# NVMe weight streaming (Vulkan / low-memory)

Stream the diffusion model and LLM text-encoder weights from disk *during*
generation, so a model whose weights exceed available memory can still run. Host
RAM stays bounded to a small reusable staging buffer (independent of model size)
and GPU memory is bounded by `--max-vram`.

This was built for shared-memory APUs such as the AMD BC-250 (RADV/Vulkan), where
"VRAM" is carved from the same physical RAM — when the GPU is full there is no RAM
left for a host-resident copy of the weights.

## Requirements

- A GPU runtime backend (validated on **Vulkan/RADV**). Not for CPU-only runs.
- `--stream-layers` and `--max-vram <GB>` are **required** (hard error otherwise).
- `nvme` implies params-on-CPU (offload) internally, but with *index-only*
  allocation, so **no full-model host copy is made** — only skeleton tensors plus
  a metadata index of file offsets.

## Usage

```
sd-cli --diffusion-model model.gguf --vae vae.sft --llm llm.gguf \
       --stream-source nvme --stream-layers --max-vram 6 --diffusion-fa \
       -p "a red apple on a wooden table" -W 512 -H 512 --steps 8 --cfg-scale 1.0
```

- `--stream-source cpu|nvme|auto` — `nvme` enables disk streaming. Default `auto`.
- `--max-vram <GB>` bounds GPU memory (resident set + largest streamed segment +
  compute buffers). A larger budget keeps more layers resident and reduces per-step
  NVMe re-reads.
- `--strict-direct-weights` — fail closed if any requested tensor cannot stream
  directly (e.g. it would need a host-side dtype conversion).

## What streams

- **Diffusion model** and **LLM text encoder** stream from NVMe.
- The **VAE** (small) loads normally into RAM.
- Conditioners that don't support streaming (non-LLM, e.g. CLIP/T5 stacks) fall back
  to normal loading; the diffusion model still streams.

## How it works

Weights are loaded *index-only*: tensor skeletons are created with no payload and a
`ModelWeightIndex` records each tensor's file offset/size. The existing graph-cut
"offload" path then, for each segment, reads the needed tensors from NVMe into a
reusable aligned host staging buffer (`O_DIRECT` when available) and uploads them to
the GPU via `ggml_backend_tensor_set`, computes the segment, then frees/reuses the
GPU memory. No full-model copy ever lives in host RAM.

## Notes / limitations

- Streamed-segment weights are re-read from NVMe each sampling step (no async
  prefetch). Throughput is NVMe-bound; raising `--max-vram` reduces re-reads.
- A single weight source reads from one file per module (diffusion → the diffusion
  model file, LLM → `--llm`). Modules whose tensors span multiple files are not yet
  supported by the direct path.
- There is no GPUDirect Storage on Vulkan/RADV, so a small host staging buffer is
  unavoidable; it is bounded by the largest single tensor, not the model size.

## Measured (AMD BC-250, Z-Image q8_0 + Qwen3-4B, 512×512, 8 steps)

| Mode | Peak host RSS | Diffusion in RAM | LLM in RAM |
|---|---|---|---|
| `--offload-to-cpu --mmap --stream-layers` | ~8.6 GB | mmap (reclaimable) | mmap |
| `--stream-source nvme --stream-layers` | **~0.8 GB** | 0 MB | 0 MB |

Output images are identical between modes (same numerical path).
