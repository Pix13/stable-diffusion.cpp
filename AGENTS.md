# AGENTS.md — stable-diffusion.cpp server: implement `--offload-to-disk`

## Goal

Implement a new server/library feature named **offload to disk** for `leejet/stable-diffusion.cpp`, focused first on `examples/server` but wired through the shared CLI/common parameter path so it can also be reused by `sd-cli` later.

Existing behavior:

- `--offload-to-cpu` places model weights in host RAM to reduce VRAM usage, then automatically moves/copies them to VRAM when a component needs to run.
- Server startup currently creates `SDContextParams`, converts it to `sd_ctx_params_t`, and passes that to `new_sd_ctx()`.
- `sd_ctx_params_t` already exposes `offload_params_to_cpu`, `enable_mmap`, `keep_clip_on_cpu`, `keep_control_net_on_cpu`, `keep_vae_on_cpu`, `max_vram`, `backend`, and `params_backend`.
- `ModelLoader` already has mmap-related APIs: `process_model_files(bool enable_mmap, bool writable_mmap)`, `mmap_tensors(...)`, and `load_tensors(..., bool use_mmap)`.

New behavior:

- `--offload-to-disk` should keep model weights backed by disk/mmap as long as possible and upload/copy tensors to the active GPU/backend only when needed.
- It must reuse the existing CPU offload path and the existing mmap/model-loader machinery as much as possible.
- It must avoid holding a full duplicate of all model weights in RAM when the user explicitly selected disk offload.

## High-level design

### Mental model

Treat disk offload as a stricter version of CPU offload:

```text
normal GPU load:       model file -> RAM decode/load -> GPU params buffers
--offload-to-cpu:      model file -> CPU/RAM params buffers -> GPU when needed
--offload-to-disk:     model file -> mmap/disk-backed tensor views -> CPU staging -> GPU when needed
```

The disk mode should not invent a separate generation pipeline. It should share the same component boundaries already used by CPU offload:

- text encoder / CLIP / T5 / LLM
- diffusion model / UNet / DiT
- VAE / TAE
- ControlNet
- upscaler path where applicable

## User-facing interface

Add these flags to `SDContextParams::get_options()` in `examples/common/common.cpp`:

```cpp
{"", "--offload-to-disk", "keep model weights disk/mmap-backed and load them to the active backend only when needed", true, &offload_params_to_disk},
{"", "--disk-offload-dir", "directory for disk-offload cache/staging files; default: model directory or OS temp", &disk_offload_dir},
```

`--disk-offload-dir` is optional for a first implementation if pure mmap is enough. If implemented later, keep the flag but document that the default uses original model files directly.

Suggested interactions:

- `--offload-to-disk` implies `--mmap`.
- `--offload-to-disk` implies `--offload-to-cpu` for code-path reuse, but the backing store must be disk/mmap rather than fully materialized RAM where possible.
- If both `--offload-to-disk` and `--offload-to-cpu` are passed, disk mode wins for tensor storage; CPU remains the staging/backend behavior.
- If the format cannot be mmap-backed, either:
  - fail with a clear error, or
  - fall back to CPU offload with a warning. Prefer fail-fast for the first PR unless compatibility is explicitly required.

## Data model changes

### `examples/common/common.h`

Add fields to `SDContextParams`:

```cpp
bool offload_params_to_disk = false;
std::string disk_offload_dir;
```

### `include/stable-diffusion.h`

Extend `sd_ctx_params_t` near the existing memory/backend controls:

```cpp
bool offload_params_to_disk;
const char* disk_offload_dir;
```

Keep ABI concerns in mind if this project treats the C API as stable. If ABI stability matters, append fields at the end of the struct instead of inserting in the middle.

### `examples/common/common.cpp`

Update:

- `SDContextParams::get_options()` to parse the new flags.
- `SDContextParams::to_string()` to log both new fields.
- `SDContextParams::resolve()` so `offload_params_to_disk` sets `enable_mmap = true` and `offload_params_to_cpu = true`.
- `SDContextParams::to_sd_ctx_params_t()` to pass the new fields.

Validation rules:

```cpp
if (offload_params_to_disk) {
    enable_mmap = true;
    offload_params_to_cpu = true;
}
```

Also validate `disk_offload_dir` when non-empty:

- create it if that matches project conventions, or
- require that it exists and is writable.

## Core implementation plan

### 1. Reuse `ModelLoader` mmap APIs

`ModelLoader` already tracks `ModelFileData`, `MmapWrapper`, `MmapTensorStore`, and exposes mmap-oriented APIs. Start there.

Implementation target:

- When `sd_ctx_params->offload_params_to_disk` is true, initialize/process model files with mmap enabled.
- Prefer `ModelLoader::mmap_tensors(...)` for tensors that can remain backed by model files.
- Avoid `load_tensors(...)` paths that copy every tensor into CPU RAM unless a tensor format requires conversion.

Expected shape in `StableDiffusionGGML::init()`:

```cpp
const bool disk_offload = sd_ctx_params->offload_params_to_disk;
const bool use_mmap = sd_ctx_params->enable_mmap || disk_offload;

model_loader.process_model_files(use_mmap, /*writable_mmap=*/!disk_offload);

if (disk_offload) {
    // Build disk-backed tensor views and register them in `tensors`.
    // Keep existing ignore_tensors behavior.
    auto mapped = model_loader.mmap_tensors(tensors, ignore_tensors, /*writable=*/false);
    // Then let component execution upload/copy the relevant tensors when needed.
} else {
    success = model_loader.load_tensors(tensors, backend, ignore_tensors, n_threads, use_mmap);
}
```

Adjust names/signatures to the real current implementation; do not force this exact pseudo-code if the existing overloads differ.

### 2. Reuse the CPU-offload transfer point

Find the existing point where CPU-backed parameter buffers are copied to the active backend/VRAM. Disk offload should enter that same point with an mmap-backed CPU source tensor.

Do **not** create a second scheduling mechanism unless necessary.

The preferred architecture is:

```text
DiskBackedTensorSource
    -> exposes ggml tensor data pointer from mmap or reads chunk on demand
    -> CPU staging buffer only for tensors that must be converted/dequantized/transformed
    -> existing backend buffer upload path
```

The implementation should be component-granular first, tensor-granular only if the current graph execution model supports it cleanly.

### 3. Lifetime management

Ensure mmap/file-backed storage lives at least as long as the tensors that reference it.

Recommended owner:

- Add a member to the model/runtime object that owns returned `MmapTensorStore` / mmap handles.
- Do not let local vectors returned by `mmap_tensors()` drop at the end of `StableDiffusionGGML::init()` if tensors still reference their mapped memory.

Example:

```cpp
class StableDiffusionGGML {
    // ...existing fields...
    std::vector<MmapTensorStore> disk_offload_mmaps;
};
```

Use the real type from `model.h`. If the returned type differs, store the actual owner type.

### 4. Read-only safety

Disk-offloaded base tensors should be read-only. Any operation that mutates model weights must first materialize a writable copy.

Pay special attention to:

- LoRA immediate merge mode
- PhotoMaker LoRA
- tensor type conversion rules
- any code path that writes into parameter tensors in place

Rules:

- `LORA_APPLY_AT_RUNTIME` is compatible with read-only disk-backed base tensors.
- `LORA_APPLY_IMMEDIATELY` must either be rejected in disk mode or force a writable CPU materialization of affected tensors only.
- If `tensor_type_rules` require conversion, converted tensors need a cache/staging location; they cannot stay as immutable views of the original file.

### 5. Format handling

Support order:

1. GGUF mmap-backed tensors.
2. safetensors mmap-backed tensors.
3. Torch zip / legacy formats only if existing `ModelLoader` can expose stable byte ranges without decompressing the whole model.

For unsupported formats in disk mode, return a clear error:

```text
--offload-to-disk requires a mmap-compatible model format for this path; use GGUF/safetensors or fall back to --offload-to-cpu.
```

### 6. Server path

`examples/server/main.cpp` should need little or no change because it already does:

```cpp
sd_ctx_params_t sd_ctx_params = ctx_params.to_sd_ctx_params_t(false, false, false);
SDCtxPtr sd_ctx(new_sd_ctx(&sd_ctx_params));
```

Most wiring belongs in shared `examples/common` and the core library.

Update `examples/server/README.md` with an example:

```bash
sd-server \
  --diffusion-model /models/qwen-image.gguf \
  --vae /models/vae.safetensors \
  --llm /models/text_encoder.gguf \
  --diffusion-fa \
  --offload-to-disk \
  --mmap \
  -v
```

Mention that first-token/first-step latency may increase because tensors are fetched from disk/mmap instead of already-resident RAM.

## Acceptance criteria

### Functional

- `sd-server --help` shows `--offload-to-disk`.
- `sd-cli --help` shows it too if shared common options are used there.
- Starting the server with `--offload-to-disk` logs:
  - disk offload enabled
  - mmap enabled
  - selected model/component formats
  - clear fallback/error when a component cannot be disk-backed
- Image generation completes on a model that previously worked with `--offload-to-cpu`.
- Peak RAM is lower than `--offload-to-cpu` for at least one large GGUF/safetensors model.
- Peak VRAM remains comparable to `--offload-to-cpu`.
- Write tests using sd-cli that works on a Vulkan build ( you can build a Vulkan version locally ). All needed models ( var, llm, and diffusion ) are available in ~/z-image

### Safety/correctness

- No dangling mmap pointers after `ModelLoader` exits initialization.
- Read-only disk-backed tensors are not mutated.
- LoRA immediate mode is either safely materialized or rejected with a clear message.
- Disk offload works with standalone `--diffusion-model`, `--vae`, `--llm`, and full `--model` paths.
- Error messages include the component/path that failed.

### Performance expectations

- Disk mode will usually be slower than CPU offload on first use or when the OS page cache is cold.
- With a warm OS page cache, performance may approach CPU offload but should not be promised.
- Do not optimize by preloading the whole model into RAM; that defeats the feature.

## Suggested tests

### Help/parse tests

```bash
./bin/sd-server --help | grep offload-to-disk
./bin/sd-cli --help | grep offload-to-disk
```

### Smoke test

```bash
./bin/sd-server \
  --diffusion-model ./models/diffusion.gguf \
  --vae ./models/vae.safetensors \
  --llm ./models/text_encoder.gguf \
  --offload-to-disk \
  --listen-port 1234 \
  -v
```

Then run one image generation through the existing `/sdcpp/v1/...` or compatibility endpoint.

### Memory comparison

Run the same model three ways:

```bash
/usr/bin/time -v ./bin/sd-cli ...
/usr/bin/time -v ./bin/sd-cli ... --offload-to-cpu
/usr/bin/time -v ./bin/sd-cli ... --offload-to-disk
```

Record:

- maximum resident set size
- VRAM from `nvidia-smi` or backend-specific equivalent
- total generation time
- first generation time after cold cache
- second generation time after warm cache

## Documentation notes

Add README wording similar to:

```markdown
`--offload-to-disk` is a low-RAM mode derived from `--offload-to-cpu`.
Instead of keeping all offloaded weights resident in system RAM, it keeps supported model weights mmap-backed by their source files and stages/uploads them to the active backend when needed. This lowers RAM pressure but may increase latency, especially with slow disks or a cold OS page cache. Prefer GGUF or safetensors models.
```

## Non-goals for the first PR

- Do not implement a new frontend UI toggle unless the backend flag is already stable.
- Do not promise faster inference.
- Do not add tensor eviction heuristics beyond what CPU offload already does.
- Do not silently mutate mmap-backed source weights.
- Do not require a separate converted cache format unless current tensor conversion makes it unavoidable.

## Files likely to change

- `examples/common/common.h`
- `examples/common/common.cpp`
- `include/stable-diffusion.h`
- `src/stable-diffusion.cpp`
- `src/model.h`
- `src/model.cpp` or the current `ModelLoader` implementation file
- `examples/server/README.md`
- optionally `examples/cli/README.md`
