# NVMe-sourced weight streaming for diffusion + LLM (Vulkan)

**Date:** 2026-06-04
**Status:** Approved design
**Target hardware:** AMD BC-250 (RADV/Vulkan), 16 GB shared CPU/GPU memory, NVMe storage. No CUDA.

## Problem

We want to run models whose weights exceed the 16 GB shared memory pool (e.g. a 20 GB+
diffusion model) on the BC-250 Vulkan box. The weights must be streamed from NVMe on
demand. They must **not** be held as a full copy in host RAM, because on this APU the GPU
"VRAM" is carved from the same physical 16 GB — when VRAM is full there is no RAM left for
a host-resident weight copy.

### Why the AGENTS.md "strict, zero host bytes" path does not apply here

The AGENTS.md spec assumes CUDA + GPUDirect Storage (cuFile), which can DMA NVMe bytes
directly into registered GPU memory. **There is no GPUDirect Storage equivalent for
RADV/Vulkan.** In ggml-Vulkan a tensor's device handle lives in `tensor->extra` and
`tensor->data` is not a host-writable pointer; the only way to fill a `VkBuffer` is
`ggml_backend_tensor_set()`, which requires a host source buffer. Therefore a small host
bounce buffer is physically unavoidable on this backend.

### Achievable invariant

- **RSS bound:** host memory never holds a full-model copy. The floor is one *aligned
  staging buffer* (sized to the largest single streamed tensor) plus the small
  non-streamed components (VAE) and normal process overhead. **Independent of model size.**
- **VRAM bound:** `--max-vram` (resident set + largest streamed segment + compute
  buffers), enforced by the existing graph-cut planner.

## Key empirical findings (validated on jarvis01, 2026-06-04)

1. The existing `--stream-layers` graph-cut streaming **already works on the Vulkan
   backend**. `sd-cli --offload-to-cpu --mmap --stream-layers --max-vram 6` generated a
   correct Z-Image result in 36.6 s; z_image was cut into 2 segments under the 6 GB budget,
   the LLM offloaded as one segment, and **VRAM stayed bounded by `--max-vram`**.
2. The streamed fill used the `offload_params` / `offload_resident_params` /
   `offload_partial_params` path (the `LayerRegistry` had no entries — logs said
   "using upstream offload path"). **That is the integration point**, not `LayerRegistry`.
3. With `--offload-to-cpu --mmap`, peak RSS was 8.65 GB and ~10.5 GB was read from NVMe via
   page faults: the weights were an mmap'd, file-backed (reclaimable) copy. RSS was that
   high only because it fit in 16 GB; the gap to close is making host RAM **explicitly,
   deterministically bounded** so it cannot starve VRAM, and so a 20 GB+ model (which
   cannot fit in RAM at all) can run.

## Approach: explicit NVMe source (Approach B)

Reuse the validated Vulkan graph-cut streaming path unchanged; change only *where the
weight bytes come from* — read each tensor from NVMe into a reusable aligned host staging
buffer and upload via `ggml_backend_tensor_set`, instead of copying from a host-resident
(mmap) copy.

### Components

#### 3.1 CLI / API
- Add `--stream-source cpu|nvme|auto` CLI flag. The `weight_stream_source` enum and
  `sd_ctx_params_t` field already exist; this exposes them. `nvme` is the new mode.
- `nvme` **requires** `--stream-layers` **and** `--max-vram > 0` **and** a GPU runtime
  backend. Violations are a **hard error with an explicit message** (no silent fallback).
- Default streamed components: `diffusion` + `llm`. VAE and other small auxiliaries load
  normally. `--direct-weight-components` is parsed but only `diffusion,llm` are honored in
  this phase; requesting others is a hard error in strict mode, a warning otherwise.
- `--strict-direct-weights` (already exists) here means: **fail closed** if any requested
  streamed component cannot use the bounded NVMe path (e.g. a tensor requires CPU-side
  dtype conversion, or has no resolvable span). Without strict, such tensors fall back to a
  one-off buffered load and a warning. Coverage is logged before generation.

#### 3.2 Index-only loading
For diffusion + LLM modules in `nvme` mode, replace "allocate + fill the monolithic params
buffer" with:
- Build a `ModelWeightIndex` mapping each param tensor name → `WeightSpan{source_path,
  file_offset, payload_bytes, type, ne, aligned_file_offset, aligned_read_bytes,
  payload_offset_inside_aligned_read, direct_streamable}`. The data comes from what the
  loader already has: `TensorStorage.offset` / dtype / shape and `ModelLoader`'s file
  paths (`file_index` → path).
- Create tensor **skeletons** (no payload) and assign each a shared lightweight **index
  buffer** so `tensor->buffer != nullptr` (this lets the tensor survive the
  `runtime_param_tensors` null-buffer filter in `ggml_graph_cut`) while **no real device or
  host memory is allocated** for payloads. `tensor->data` stays null between segments.
- A tensor is **`direct_streamable`** only if its on-disk dtype/layout already matches the
  runtime tensor layout (no CPU-side conversion). Non-matching tensors are flagged: hard
  error in strict mode; otherwise a one-time buffered load fills them like normal params.
  (Z-Image's tensors are q8_0/f32 and already match, so all stream directly.)

#### 3.3 NVMe payload source (Vulkan-correct fill)
Change the `WeightPayloadSource` interface from a raw dst-pointer signature to:

```cpp
// Read the tensor's payload from storage and upload it into the destination
// ggml tensor's backend buffer. Bounded host staging only; no full-model copy.
virtual bool read_to_tensor(const WeightSpan& span, ggml_tensor* dst) = 0;
```

The Vulkan/CPU implementation (`NvmeStagedWeightSource`):
- Keeps the file descriptor open (opened once per source).
- `pread` the aligned range (`O_DIRECT` for page-cache-free reads) into a **reusable,
  aligned host staging buffer** (`posix_memalign`, grown to the largest
  `aligned_read_bytes` seen, optionally chunked to a cap).
- `ggml_backend_tensor_set(dst, staging + payload_offset_inside_aligned_read, 0,
  payload_bytes)`.
- If `O_DIRECT` is unsupported by the filesystem, fall back to a buffered `pread` and log
  it. Short reads and I/O errors are hard failures.

The CUDA/GDS source (`weight_payload_source_gds.cpp`) remains `#ifdef SD_CUDA_GDS` for a
future direct path and is out of scope here.

#### 3.4 Integration into the offload functions
The validated path fills GPU twins via `ggml_backend_tensor_copy(cpu_tensor, twin)` in
three functions in `ggml_extend.hpp`: `offload_params` (whole module),
`offload_resident_params` (resident set, filled once and kept on GPU), and
`offload_partial_params` (streamed segment, filled each sampling step). Introduce one
shared helper:

```cpp
// Fill the GPU twin for `tensor`. In nvme mode, read directly from storage;
// otherwise use the existing host-to-device copy.
bool fill_twin(ggml_tensor* tensor, ggml_tensor* twin);
```

- If `weight_index_` has a span for `tensor` and `tensor` is a skeleton (no resident
  payload) → `weight_payload_source_->read_to_tensor(span, twin)`.
- Otherwise → `ggml_backend_tensor_copy(tensor, twin)` (unchanged; VAE and any
  non-streamable tensors keep working exactly as today).

Consequence: resident-set params are read from NVMe **once**; streamed-segment params are
re-read from NVMe **each sampling step**. Correctness-first, synchronous. **No async
prefetch** (PR #1576 disabled it after correctness regressions; re-adding it is out of
scope).

### 3.5 Memory model
- **VRAM** = resident params (chosen by `annotate_residency` under the budget) + largest
  streamed segment's params + compute buffer + graph cache. The planner already reserves
  the worst-case streamed-segment footprint + safety margin, so total ≤ `--max-vram`.
  Resident set is maximized within the budget to minimize NVMe re-reads.
- **RSS** = aligned staging buffer (largest streamed tensor's `aligned_read_bytes`) + VAE
  params + process overhead. Model-size-independent. A 20 GB+ model is bounded by
  `--max-vram` in VRAM and by the staging buffer in RAM.

## Files touched
- `examples/common/common.{h,cpp}` — `--stream-source` flag, validation, `to_string()`.
- `include/stable-diffusion.h` — params already present; ensure fully plumbed.
- `src/stable-diffusion.cpp` — nvme validation; for diffusion + LLM, drive index-only
  load, build `ModelWeightIndex`, attach source + index to the runner; strict coverage log.
- `src/ggml_extend.hpp` — index-only param allocation (shared index buffer); `fill_twin`
  helper and its use in the three offload functions; hold `weight_payload_source_` /
  `weight_index_` (already added).
- `src/weight_payload_source.h` / `weight_payload_source_cpu.cpp` — new `read_to_tensor`
  interface and `NvmeStagedWeightSource` (aligned O_DIRECT staged reads).
- `src/model_weight_index.*`, `src/weight_span.h` — offset/dtype population,
  `direct_streamable` marking, lookup.
- `src/model.cpp` / model loader — expose per-tensor `{path, offset, dtype, shape}` to the
  index-only path (read-only; reuse existing `TensorStorage`/`file_paths_`).

## Risks (validate empirically while building)
1. **Skeleton tensors with an index buffer must survive every ggml code path** (graph
   build, allocation, segment execution) with `data == null` between segments. This is the
   load-bearing assumption. **Prototype and test this first** before building the rest. If
   ggml dereferences the skeleton's data outside the swap-in window, adjust (e.g. give the
   index buffer a valid host base of size 0, or special-case the planner).
2. **dtype/layout match** — only directly-streamable tensors qualify; confirm all of
   Z-Image's diffusion + Qwen3 LLM tensors qualify (expected: yes).
3. **Performance** — streamed params re-read per step; NVMe-bound but functional. Acceptable
   for correctness-first phase.

## Verification
- **Unit tests:** `ModelWeightIndex` span registration/lookup; `WeightSpan::compute_aligned_io`
  math; strict rejection of unaligned / missing / conversion-needed tensors.
- **Integration (jarvis01):**
  1. Baseline reference image with normal load (no streaming) — already captured
     (red apple, seed 42, 512×512, 8 steps, cfg 1.0).
  2. Same prompt/seed with `--stream-source nvme --stream-layers --max-vram N`; assert the
     output image matches the reference (same numerical path).
  3. Assert **peak RSS ≪ model size** via `/usr/bin/time -v` (Maximum resident set size),
     and that no full-model RAM copy occurs.
  4. Confirm VRAM stays within `--max-vram`.
- **Regression:** `--stream-source cpu` / no streaming unchanged; VAE path unchanged.

## Out of scope (this phase)
- CUDA/GPUDirect Storage direct path (kept `#ifdef`'d).
- Async prefetch / double-buffering.
- VAE streaming and ControlNet/other auxiliaries.
- A prepacked direct-weight file format (`sdcpp-pack-direct`).
