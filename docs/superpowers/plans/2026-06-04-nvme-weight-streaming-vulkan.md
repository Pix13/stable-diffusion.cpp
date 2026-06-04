# NVMe Weight Streaming (Vulkan) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream a diffusion model + LLM text encoder's weights from NVMe on demand on the AMD BC-250 (RADV/Vulkan), so host RAM never holds a full-model copy and a 20 GB+ model runs within the 16 GB shared pool.

**Architecture:** Reuse the already-validated graph-cut "offload" streaming path. Load diffusion + LLM in *index-only* mode (skeleton tensors + a `ModelWeightIndex` of file offsets, no params buffer). When the existing offload functions fill a GPU twin, read the tensor's bytes from NVMe into a reusable aligned host staging buffer and upload via `ggml_backend_tensor_set`, instead of copying from a host-resident weight copy. Gated behind `--stream-source nvme`.

**Tech Stack:** C++17, ggml (Vulkan backend), CMake. Target box: `dev@jarvis01.dmz`, models in `~/z-image/{model.gguf,vae.sft,llm.gguf}`.

**Spec:** `docs/superpowers/specs/2026-06-04-nvme-weight-streaming-vulkan-design.md`

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/weight_span.h` | Per-tensor metadata + aligned-IO math | Modify: fix `compute_aligned_io`, add `direct_streamable` rules |
| `src/model_weight_index.{h,cpp}` | tensor-name → `WeightSpan` map | Modify: minor; used by builder |
| `src/weight_payload_source.h` | Abstract source interface | Modify: replace dst-pointer API with `read_to_tensor(span, ggml_tensor*)` |
| `src/weight_payload_source_cpu.cpp` | `NvmeStagedWeightSource` (O_DIRECT staged reads → `tensor_set`) | Rewrite the impl + factory |
| `src/weight_payload_source_gds.cpp` | CUDA/GDS source | Unchanged (`#ifdef SD_CUDA_GDS`) |
| `src/model.h` | `ModelLoader` accessors | Modify: add `get_file_path(idx)` |
| `src/stable-diffusion.cpp` | build index, attach source, validate flags, index-only load for diffusion+LLM | Modify |
| `src/ggml_extend.hpp` | index-only param alloc + `fill_twin` helper in the 3 offload functions | Modify |
| `examples/common/common.{h,cpp}` | `--stream-source` CLI flag + validation/log | Modify |
| `tests/test_weight_streaming.cpp` | Unit tests for pure-logic pieces | Create |
| `CMakeLists.txt` | opt-in `sd-tests` target | Modify |

**Build conventions (used throughout):**
- Local CPU build dir: `build-cpu` (already configured). Rebuild a target: `cmake --build build-cpu -j6 --target <t>`.
- Remote (Vulkan) box: `dev@jarvis01.dmz`, repo at `~/stable-diffusion.cpp`, build dir `build`.
- Deploy code to remote: `rsync -az --delete --exclude='/.git' --exclude='/build' --exclude='/build-cpu' --exclude='node_modules' -e 'ssh -o BatchMode=yes' ./ dev@jarvis01.dmz:~/stable-diffusion.cpp/`
- Remote build: `ssh -o BatchMode=yes dev@jarvis01.dmz 'cd ~/stable-diffusion.cpp && nice -n 10 cmake --build build -j3 --target sd-cli'`

---

## Task 0: Add opt-in unit-test target

**Files:**
- Create: `tests/test_weight_streaming.cpp`
- Modify: `CMakeLists.txt` (end of file)

- [ ] **Step 1: Create a minimal assert-based test scaffold**

`tests/test_weight_streaming.cpp`:
```cpp
#include <cstdio>
#include <cstdlib>

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_failures; } \
} while (0)

void test_weight_span();        // Task 1
void test_model_weight_index(); // Task 2
void test_nvme_source();        // Task 3

int main() {
    test_weight_span();
    test_model_weight_index();
    test_nvme_source();
    if (g_failures == 0) { printf("ALL TESTS PASSED\n"); return 0; }
    fprintf(stderr, "%d CHECK(s) FAILED\n", g_failures);
    return 1;
}
```

Add stub definitions at the bottom of the same file so it links before later tasks fill them in:
```cpp
void test_weight_span() {}
void test_model_weight_index() {}
void test_nvme_source() {}
```

- [ ] **Step 2: Add the CMake target (append to `CMakeLists.txt`)**

```cmake
option(SD_BUILD_TESTS "sd: build unit tests" OFF)
if(SD_BUILD_TESTS)
    add_executable(sd-tests tests/test_weight_streaming.cpp)
    target_link_libraries(sd-tests PRIVATE ${SD_LIB} ggml)
    target_include_directories(sd-tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include)
endif()
```

- [ ] **Step 3: Configure + build the test target locally**

Run: `cmake -S . -B build-cpu -DSD_BUILD_TESTS=ON -DSD_SERVER_BUILD_FRONTEND=OFF >/dev/null && cmake --build build-cpu -j6 --target sd-tests`
Expected: builds `build-cpu/bin/sd-tests`.

- [ ] **Step 4: Run it**

Run: `./build-cpu/bin/sd-tests`
Expected: `ALL TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add tests/test_weight_streaming.cpp CMakeLists.txt
git commit -m "test: add opt-in sd-tests target for weight-streaming logic"
```

---

## Task 1: WeightSpan aligned-IO correctness

**Problem in current code:** `WeightSpan::compute_aligned_io` always sets `direct_streamable = true`, even though `direct_streamable` should reflect dtype/layout readiness (set by the index builder in Task 2), not alignment. Alignment must always succeed (we read an aligned superset). Separate the two concerns.

**Files:**
- Modify: `src/weight_span.h`
- Modify: `tests/test_weight_streaming.cpp` (`test_weight_span`)

- [ ] **Step 1: Write the failing test** (replace the `test_weight_span` stub)

```cpp
#include "weight_span.h"
void test_weight_span() {
    int64_t ne[1] = {100};
    WeightSpan s("t", "/m.gguf", /*offset*/ 5000, /*bytes*/ 100, GGML_TYPE_F32, ne, 1);
    CHECK(s.compute_aligned_io(4096) == true);
    CHECK(s.aligned_file_offset == 4096);                       // 5000 -> down to 4096
    CHECK(s.payload_offset_inside_aligned_read == 904);         // 5000 - 4096
    CHECK(s.aligned_read_bytes == 4096);                        // ceil(904+100, 4096)
    CHECK(s.aligned_file_offset + s.aligned_read_bytes >= s.file_offset + s.payload_bytes);
    // compute_aligned_io must NOT decide streamability:
    WeightSpan s2("t2", "/m.gguf", 0, 200, GGML_TYPE_F32, ne, 1);
    s2.compute_aligned_io(4096);
    CHECK(s2.direct_streamable == false);   // unchanged by alignment
}
```

- [ ] **Step 2: Run test, verify it fails**

Run: `cmake --build build-cpu -j6 --target sd-tests && ./build-cpu/bin/sd-tests`
Expected: FAIL on `s2.direct_streamable == false` (current code sets it true).

- [ ] **Step 3: Fix `compute_aligned_io`** in `src/weight_span.h` — remove the `direct_streamable = true;` line inside it (leave the alignment math). The method becomes:

```cpp
    bool compute_aligned_io(size_t alignment = 4096) {
        if (payload_bytes == 0 || alignment == 0) {
            aligned_file_offset = file_offset;
            aligned_read_bytes  = payload_bytes;
            payload_offset_inside_aligned_read = 0;
            return true;
        }
        aligned_file_offset = (file_offset / alignment) * alignment;
        payload_offset_inside_aligned_read = file_offset - aligned_file_offset;
        uint64_t unaligned = payload_bytes + payload_offset_inside_aligned_read;
        aligned_read_bytes = ((unaligned + alignment - 1) / alignment) * alignment;
        return true;
    }
```

- [ ] **Step 4: Run test, verify pass**

Run: `cmake --build build-cpu -j6 --target sd-tests && ./build-cpu/bin/sd-tests`
Expected: `ALL TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/weight_span.h tests/test_weight_streaming.cpp
git commit -m "fix(weight-span): keep direct_streamable independent of alignment"
```

---

## Task 2: Build a ModelWeightIndex from the loader + accessor

**Files:**
- Modify: `src/model.h` (add `get_file_path`)
- Create: `src/model_weight_index.cpp` already exists; add a free function `build_weight_index(...)` in `model_weight_index.{h,cpp}`
- Modify: `tests/test_weight_streaming.cpp` (`test_model_weight_index`)

- [ ] **Step 1: Add a read-only path accessor to `ModelLoader`** (`src/model.h`, in the public section near line 276):

```cpp
    const std::string& get_file_path(size_t index) const { return file_paths_.at(index); }
    size_t get_num_files() const { return file_paths_.size(); }
```

- [ ] **Step 2: Declare the builder** in `src/model_weight_index.h` (after the class):

```cpp
class ModelLoader;  // fwd
// Build spans for the given runtime param tensors (name -> ggml_tensor*) using the
// loader's tensor storage map. A tensor is direct_streamable only when its on-disk
// dtype matches the runtime tensor dtype (no host-side conversion).
std::shared_ptr<ModelWeightIndex> build_weight_index(
    ModelLoader& loader,
    const std::map<std::string, struct ggml_tensor*>& runtime_tensors,
    size_t alignment);
```
Add `#include <memory>` and `#include <map>` to the header if not present.

- [ ] **Step 3: Implement the builder** in `src/model_weight_index.cpp`:

```cpp
#include "model.h"

std::shared_ptr<ModelWeightIndex> build_weight_index(
    ModelLoader& loader,
    const std::map<std::string, ggml_tensor*>& runtime_tensors,
    size_t alignment) {
    auto index = std::make_shared<ModelWeightIndex>();
    auto& storage_map = loader.get_tensor_storage_map();
    for (const auto& [name, tensor] : runtime_tensors) {
        auto it = storage_map.find(name);
        if (it == storage_map.end()) {
            continue;  // not all runtime tensors are file-backed (e.g. derived)
        }
        const TensorStorage& ts = it->second;
        const std::string& path = loader.get_file_path(ts.file_index);
        WeightSpan span(name, path, ts.offset, ggml_nbytes(tensor),
                        tensor->type, tensor->ne, ggml_n_dims(tensor));
        span.compute_aligned_io(alignment);
        // Direct-streamable only when on-disk dtype already matches runtime dtype
        // and the file is not a zip (zip payloads are not contiguous/aligned here).
        span.direct_streamable = (ts.type == tensor->type) && (ts.index_in_zip < 0);
        index->add_span(name, span);
    }
    return index;
}
```

- [ ] **Step 4: Write the failing test** (replace `test_model_weight_index` stub). This test exercises `ModelWeightIndex` directly (the builder is covered by integration tests since it needs a real loader):

```cpp
#include "model_weight_index.h"
void test_model_weight_index() {
    ModelWeightIndex idx;
    int64_t ne[1] = {10};
    WeightSpan a("a", "/m", 0, 40, GGML_TYPE_F32, ne, 1); a.direct_streamable = true;
    WeightSpan b("b", "/m", 64, 40, GGML_TYPE_F32, ne, 1); b.direct_streamable = false;
    CHECK(idx.add_span("a", a) == true);
    CHECK(idx.add_span("a", a) == false);   // duplicate rejected
    CHECK(idx.add_span("b", b) == true);
    CHECK(idx.find_by_name("a") != nullptr);
    CHECK(idx.find_by_name("missing") == nullptr);
    CHECK(idx.all_direct_streamable() == false);  // b is not
    CHECK(idx.size() == 2);
}
```

- [ ] **Step 5: Build + run, verify pass**

Run: `cmake --build build-cpu -j6 --target sd-tests && ./build-cpu/bin/sd-tests`
Expected: `ALL TESTS PASSED`

- [ ] **Step 6: Commit**

```bash
git add src/model.h src/model_weight_index.h src/model_weight_index.cpp tests/test_weight_streaming.cpp
git commit -m "feat(weight-index): builder from ModelLoader + file path accessor"
```

---

## Task 3: NvmeStagedWeightSource (read_to_tensor)

**Files:**
- Modify: `src/weight_payload_source.h` (interface)
- Rewrite impl: `src/weight_payload_source_cpu.cpp`
- Modify: `tests/test_weight_streaming.cpp` (`test_nvme_source`)

- [ ] **Step 1: Change the interface** in `src/weight_payload_source.h` — replace the `read_to_device(...)` pure virtual with:

```cpp
    // Read the tensor payload from storage and upload into dst's backend buffer.
    // Uses only a bounded, reusable host staging buffer; never a full-model copy.
    virtual bool read_to_tensor(const WeightSpan& span, struct ggml_tensor* dst) = 0;
```
Keep `open()`, `supports_strict_direct()`, `close()`, `path()`, the `source_path_` member, and the `create_weight_payload_source(path, strict)` factory declaration. Add `#include "ggml.h"`.

- [ ] **Step 2: Rewrite `src/weight_payload_source_cpu.cpp`** as `NvmeStagedWeightSource`:

```cpp
#include "weight_payload_source.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

#include "ggml-backend.h"

// Reads tensor payloads from a file into a reusable aligned host staging buffer,
// then uploads to the destination tensor's backend buffer via ggml_backend_tensor_set.
// Bounded host memory: the staging buffer grows to the largest aligned read, never
// the whole model. Tries O_DIRECT (page-cache-free); falls back to buffered reads.
class NvmeStagedWeightSource : public WeightPayloadSource {
public:
    explicit NvmeStagedWeightSource(const std::string& path)
        : WeightPayloadSource(path) {}
    ~NvmeStagedWeightSource() override { close(); }

    bool open() override {
        fd_ = ::open(source_path_.c_str(), O_RDONLY | O_DIRECT);
        if (fd_ >= 0) { direct_ = true; return true; }
        fd_ = ::open(source_path_.c_str(), O_RDONLY);
        if (fd_ < 0) {
            fprintf(stderr, "NvmeStagedWeightSource: cannot open '%s': %s\n",
                    source_path_.c_str(), strerror(errno));
            return false;
        }
        fprintf(stderr, "NvmeStagedWeightSource: O_DIRECT unavailable for '%s', "
                "using buffered reads\n", source_path_.c_str());
        return true;
    }

    bool supports_strict_direct() const override { return false; } // bounded host buffer, not zero-copy

    bool read_to_tensor(const WeightSpan& span, ggml_tensor* dst) override {
        if (fd_ < 0) { fprintf(stderr, "NvmeStagedWeightSource: not open\n"); return false; }
        const size_t   want   = span.aligned_read_bytes ? span.aligned_read_bytes : span.payload_bytes;
        const uint64_t off    = span.aligned_read_bytes ? span.aligned_file_offset : span.file_offset;
        const size_t   inset  = span.aligned_read_bytes ? span.payload_offset_inside_aligned_read : 0;
        if (!ensure_buffer(want)) return false;

        size_t done = 0;
        while (done < want) {
            ssize_t n = ::pread(fd_, buf_ + done, want - done, (off_t)(off + done));
            if (n < 0) {
                if (errno == EINTR) continue;
                fprintf(stderr, "NvmeStagedWeightSource: pread '%s' failed: %s\n",
                        span.tensor_name.c_str(), strerror(errno));
                return false;
            }
            if (n == 0) break;  // EOF
            done += (size_t)n;
        }
        if (done < inset + span.payload_bytes) {
            fprintf(stderr, "NvmeStagedWeightSource: short read '%s': got %zu need %zu\n",
                    span.tensor_name.c_str(), done, inset + (size_t)span.payload_bytes);
            return false;
        }
        ggml_backend_tensor_set(dst, buf_ + inset, 0, span.payload_bytes);
        return true;
    }

    void close() override {
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
        if (buf_) { free(buf_); buf_ = nullptr; cap_ = 0; }
    }
    const std::string& path() const override { return source_path_; }

private:
    bool ensure_buffer(size_t need) {
        // O_DIRECT requires 512-byte aligned buffer; 4096 is safe for NVMe.
        if (cap_ >= need && buf_) return true;
        free(buf_); buf_ = nullptr; cap_ = 0;
        void* p = nullptr;
        if (posix_memalign(&p, 4096, need) != 0 || !p) {
            fprintf(stderr, "NvmeStagedWeightSource: staging alloc %zu failed\n", need);
            return false;
        }
        buf_ = (uint8_t*)p; cap_ = need;
        return true;
    }

    int      fd_     = -1;
    bool     direct_ = false;
    uint8_t* buf_    = nullptr;
    size_t   cap_    = 0;
};

std::unique_ptr<WeightPayloadSource> create_weight_payload_source(
    const std::string& path, bool /*strict*/) {
    return std::make_unique<NvmeStagedWeightSource>(path);
}
```

- [ ] **Step 3: Write the failing test** (`test_nvme_source`): write a temp file with known bytes at an offset, read it into a CPU-backend tensor, verify contents.

```cpp
#include "weight_payload_source.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
void test_nvme_source() {
    const char* path = "/tmp/sd_nvme_test.bin";
    const int N = 256;
    float data[N];
    for (int i = 0; i < N; ++i) data[i] = (float)i * 1.5f;
    FILE* f = fopen(path, "wb");
    CHECK(f != nullptr);
    char pad[100] = {0};
    fwrite(pad, 1, sizeof(pad), f);          // 100-byte prefix -> unaligned offset
    fwrite(data, sizeof(float), N, f);
    fclose(f);

    int64_t ne[1] = {N};
    WeightSpan span("w", path, /*offset*/ 100, /*bytes*/ N * sizeof(float),
                    GGML_TYPE_F32, ne, 1);
    span.compute_aligned_io(4096);

    ggml_init_params ip{ ggml_tensor_overhead() + 1024, nullptr, true };
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    CHECK(buf != nullptr);

    auto src = create_weight_payload_source(path, false);
    CHECK(src->open() == true);
    CHECK(src->read_to_tensor(span, t) == true);

    float out[N];
    ggml_backend_tensor_get(t, out, 0, sizeof(out));
    bool ok = true;
    for (int i = 0; i < N; ++i) if (out[i] != data[i]) ok = false;
    CHECK(ok);

    src->close();
    ggml_backend_buffer_free(buf);
    ggml_backend_free(cpu);
    ggml_free(ctx);
}
```

- [ ] **Step 4: Build + run, verify pass**

Run: `cmake --build build-cpu -j6 --target sd-tests && ./build-cpu/bin/sd-tests`
Expected: `ALL TESTS PASSED` (the read offset 100 is unaligned, exercising the aligned-superset + inset path).

- [ ] **Step 5: Commit**

```bash
git add src/weight_payload_source.h src/weight_payload_source_cpu.cpp tests/test_weight_streaming.cpp
git commit -m "feat(weight-source): NvmeStagedWeightSource with aligned O_DIRECT staged reads"
```

---

## Task 4: Index-only param allocation + `fill_twin` helper (SPIKE — validate first)

> This task carries the load-bearing risk (skeleton tensors with `data==null` between segments). Build it, then validate on jarvis01 in Task 6 before relying on it.

**Files:**
- Modify: `src/ggml_extend.hpp`

- [ ] **Step 1: Add members + setter for nvme mode** in `GGMLRunner` (near the existing `weight_payload_source_` / `weight_index_` members, ~line 1716):

```cpp
    bool nvme_stream_mode_ = false;
    ggml_backend_buffer_t index_buffer_ = nullptr;  // 0-byte marker buffer for skeleton tensors
public:
    void set_nvme_stream_mode(bool on) { nvme_stream_mode_ = on; }
protected:
```

- [ ] **Step 2: Add `alloc_params_index_only()`** next to `alloc_params_buffer()` (~line 3007). It registers every params_ctx tensor against a single 0-size marker buffer so `tensor->buffer != nullptr` while no payload memory is allocated:

```cpp
    bool alloc_params_index_only() {
        size_t num_tensors = ggml_tensor_num(params_ctx);
        if (num_tensors == 0) return true;
        // A real (small) backend buffer to satisfy buffer!=null checks. Tensors keep
        // data==nullptr; their payloads are streamed into GPU twins on demand.
        index_buffer_ = ggml_backend_buft_alloc_buffer(
            ggml_backend_get_default_buffer_type(params_backend), 0);
        if (index_buffer_ == nullptr) {
            LOG_ERROR("%s alloc index buffer failed", get_desc().c_str());
            return false;
        }
        ggml_backend_buffer_set_usage(index_buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (ggml_tensor* t = ggml_get_first_tensor(params_ctx); t != nullptr;
             t = ggml_get_next_tensor(params_ctx, t)) {
            t->buffer = index_buffer_;
            t->data   = nullptr;
        }
        rebuild_params_tensor_set();
        LOG_INFO("%s params index-only (%zu tensors, 0 MB resident)",
                 get_desc().c_str(), num_tensors);
        return true;
    }
```

> Note: if `ggml_backend_buft_alloc_buffer(..., 0)` returns null on Vulkan, allocate 1 byte instead. This is verified empirically in Task 6; adjust the size argument there if needed.

- [ ] **Step 3: Add the `fill_twin` helper** (private, near the offload functions ~line 2168):

```cpp
    // Fill a GPU twin for `src` param tensor. In nvme mode, stream from storage;
    // otherwise copy from the host-resident source tensor (existing behavior).
    bool fill_twin(ggml_tensor* src, ggml_tensor* twin) {
        if (nvme_stream_mode_ && weight_index_ && weight_payload_source_) {
            const WeightSpan* span = weight_index_->find(src);
            if (span != nullptr && span->direct_streamable) {
                return weight_payload_source_->read_to_tensor(*span, twin);
            }
        }
        ggml_backend_tensor_copy(src, twin);
        return true;
    }
```

- [ ] **Step 4: Route the three fill sites through `fill_twin`.** Replace each `ggml_backend_tensor_copy(<src>, <twin>);` with `if (!fill_twin(<src>, <twin>)) return false;`* in:
  - `offload_params` (the `while` loop ~line 2143: `ggml_backend_tensor_copy(t, offload_t);` → `if (!fill_twin(t, offload_t)) return false;`)
  - `offload_resident_params` (the resident copy `ggml_backend_tensor_copy(t, twin);`)
  - `offload_partial_params` (the partial copy `ggml_backend_tensor_copy(tensor, offload_tensor);`)

  *(`offload_params` returns `bool`; confirm the others do too. If a site is `void`, log an error and `GGML_ASSERT(false)` instead of `return false`.)*

- [ ] **Step 5: Build locally (compile check only — nvme path inert without wiring)**

Run: `cmake --build build-cpu -j6 --target sd-cli`
Expected: builds with exit 0.

- [ ] **Step 6: Commit**

```bash
git add src/ggml_extend.hpp
git commit -m "feat(streaming): index-only param alloc + fill_twin (nvme source hook)"
```

---

## Task 5: Wire `--stream-source nvme` end to end for the diffusion model

**Files:**
- Modify: `examples/common/common.cpp` (add `--stream-source` option + validation)
- Modify: `src/stable-diffusion.cpp` (validation; index-only load + index build + source attach for diffusion)

- [ ] **Step 1: Add the CLI flag** in `examples/common/common.cpp` `get_options()` (near the existing `--strict-direct-weights` entry, ~line 516). Use the existing `str_to_weight_stream_source` + a string field; add a `--stream-source` arg that sets `weight_stream_source`:

```cpp
        {"", "--stream-source",
         "weight stream source: cpu, nvme, or auto (default auto). nvme streams weights "
         "from disk during generation; requires --stream-layers and --max-vram",
         false, &weight_stream_source},
```
(`weight_stream_source` is already a `std::string` in `common.h`. The string→enum conversion already happens in `to_sd_ctx_params_t`.)

- [ ] **Step 2: Add validation in `StableDiffusionGGML`** (`src/stable-diffusion.cpp`, in the params-ingest block ~line 275, after the existing `stream_layers && max_vram==0` check):

```cpp
        if (weight_stream_source == SD_WEIGHT_STREAM_SOURCE_NVME) {
            if (!stream_layers || max_vram == 0.f) {
                LOG_ERROR("--stream-source nvme requires --stream-layers and --max-vram > 0");
                return false;  // match the function's existing failure convention
            }
        }
```
*(If this constructor/initializer cannot `return false`, set an `init_failed` flag the caller already checks, following the surrounding pattern.)*

- [ ] **Step 3: Helper to enable nvme load for a module.** Add a private method to `StableDiffusionGGML` (near the diffusion load site ~line 770):

```cpp
    bool is_nvme_stream() const {
        return stream_layers && weight_stream_source == SD_WEIGHT_STREAM_SOURCE_NVME;
    }
```

- [ ] **Step 4: Replace the diffusion load path** at ~line 770 (the block that currently creates a source then calls `get_param_tensors(diffusion_model, ...)`). The current diff added a `create_weight_payload_source(...)` call; replace that whole block with index-only handling:

```cpp
            if (is_nvme_stream()) {
                // Build skeleton tensors + index, attach NVMe source; no params buffer.
                std::map<std::string, ggml_tensor*> dm_tensors;
                diffusion_model->get_param_tensors(dm_tensors);   // names -> skeleton tensors
                auto index = build_weight_index(model_loader, dm_tensors, /*alignment*/ 4096);
                index->log_direct_coverage();
                if (strict_direct_weights && !index->all_direct_streamable()) {
                    LOG_ERROR("strict direct weights: not all diffusion tensors are streamable");
                    return false;
                }
                auto source = create_weight_payload_source(
                    model_loader.get_file_path(0), strict_direct_weights);
                if (!source || !source->open()) {
                    LOG_ERROR("failed to open NVMe weight source for diffusion model");
                    return false;
                }
                diffusion_model->set_weight_index(index);
                diffusion_model->set_weight_payload_source(std::move(source));
                diffusion_model->set_nvme_stream_mode(true);
                diffusion_model->alloc_params_index_only();
            } else {
                get_param_tensors(diffusion_model, module_can_mmap(SDBackendModule::DIFFUSION));
            }
```

> Verify the exact way `get_param_tensors(model, ...)` triggers buffer alloc/fill, and that `diffusion_model->get_param_tensors(map)` (the runner method that just enumerates tensors) exists — the architecture report cites it at `diffusion_model.hpp:103`. `alloc_params_index_only()` must be public or called via a thin public wrapper on the runner. The source path uses `get_file_path(0)`; if the diffusion model is a separate file, use the file index recorded for its tensors (all diffusion tensors share one file in these models).

- [ ] **Step 5: Build locally (compile check)**

Run: `cmake --build build-cpu -j6 --target sd-cli`
Expected: exit 0.

- [ ] **Step 6: Commit**

```bash
git add examples/common/common.cpp examples/common/common.h src/stable-diffusion.cpp
git commit -m "feat(cli): --stream-source nvme wires index-only NVMe streaming for diffusion"
```

---

## Task 6: Integration validation on jarvis01 — diffusion-only NVMe streaming (SPIKE GATE)

This proves the skeleton-tensor mechanism works on Vulkan and bounds RSS. If it fails, iterate on Task 4 (index buffer size, planner null-buffer handling) before proceeding.

- [ ] **Step 1: Deploy + build on the Vulkan box**

```bash
rsync -az --delete --exclude='/.git' --exclude='/build' --exclude='/build-cpu' --exclude='node_modules' -e 'ssh -o BatchMode=yes' ./ dev@jarvis01.dmz:~/stable-diffusion.cpp/
ssh -o BatchMode=yes dev@jarvis01.dmz 'cd ~/stable-diffusion.cpp && nice -n 10 cmake --build build -j3 --target sd-cli'
```
Expected: build exit 0.

- [ ] **Step 2: Generate with NVMe streaming, capture RSS**

```bash
ssh -o BatchMode=yes dev@jarvis01.dmz 'cd ~/z-image && /usr/bin/time -v ~/stable-diffusion.cpp/build/bin/sd-cli \
  --diffusion-model model.gguf --vae vae.sft --llm llm.gguf \
  --stream-source nvme --stream-layers --max-vram 6 --diffusion-fa \
  -p "a red apple on a wooden table, studio lighting" --cfg-scale 1.0 --steps 8 \
  -W 512 -H 512 --seed 42 -o /tmp/nvme_test.png -v 2>/tmp/nvme.log; \
  echo EXIT=$?; grep -E "Maximum resident|index-only|direct weights|streaming" /tmp/nvme.log | head'
```
Expected: `EXIT=0`; log shows `z_image params index-only`; **Maximum resident set size well below the model size** (target: a few hundred MB to low-GB range, far under the ~8.6 GB mmap baseline). Diffusion-only here means the LLM still loads normally, so RSS includes the ~4 GB LLM — acceptable until Task 7. To isolate the diffusion RSS win, also run once with `--max-vram 6` and confirm the diffusion contribution to RSS is gone (no `z_image ... (VRAM/RAM)` resident params line).

- [ ] **Step 3: Verify image correctness against the reference**

```bash
scp -o BatchMode=yes dev@jarvis01.dmz:/tmp/nvme_test.png /tmp/nvme_test.png
# Compare to the non-streamed reference generated earlier (same seed/prompt/size/steps).
ssh -o BatchMode=yes dev@jarvis01.dmz 'cd ~/z-image && ~/stable-diffusion.cpp/build/bin/sd-cli \
  --diffusion-model model.gguf --vae vae.sft --llm llm.gguf --diffusion-fa \
  -p "a red apple on a wooden table, studio lighting" --cfg-scale 1.0 --steps 8 \
  -W 512 -H 512 --seed 42 -o /tmp/ref.png >/dev/null 2>&1; \
  python3 -c "import hashlib;print(\"nvme\",hashlib.md5(open(\"/tmp/nvme_test.png\",\"rb\").read()).hexdigest());print(\"ref \",hashlib.md5(open(\"/tmp/ref.png\",\"rb\").read()).hexdigest())"'
```
Expected: identical (or visually identical — minor nondeterminism from Vulkan kernels is acceptable; pull both PNGs and compare visually if hashes differ). The Read tool can render `/tmp/nvme_test.png` to confirm it's the apple.

- [ ] **Step 4: If it works, commit a note; if not, debug Task 4 and repeat**

```bash
git commit --allow-empty -m "test: validated diffusion-only NVMe streaming on jarvis01 (Vulkan)"
```

---

## Task 7: Extend NVMe streaming to the LLM text encoder

**Files:**
- Modify: `src/stable-diffusion.cpp` (apply the same index-only path to the LLM/conditioner module)

- [ ] **Step 1: Locate the LLM/text-encoder load site** (`loading llm from ...`, ~stable-diffusion.cpp:370; param-tensor registration for the conditioner). Apply the same pattern as Task 5 Step 4 to the LLM runner: enumerate its param tensors, `build_weight_index(model_loader, llm_tensors, 4096)`, open a source on the LLM file path (use the `file_index` of an LLM tensor → `get_file_path(idx)`, since the LLM is a separate file), `set_weight_index` / `set_weight_payload_source` / `set_nvme_stream_mode(true)` / `alloc_params_index_only()` on the LLM runner, all guarded by `is_nvme_stream()`.

```cpp
            if (is_nvme_stream()) {
                std::map<std::string, ggml_tensor*> te_tensors;
                cond_stage_model->get_param_tensors(te_tensors);
                size_t te_file_index = te_tensors.empty() ? 0
                    : model_loader.get_tensor_storage_map().at(te_tensors.begin()->first).file_index;
                auto index = build_weight_index(model_loader, te_tensors, 4096);
                index->log_direct_coverage();
                if (strict_direct_weights && !index->all_direct_streamable()) {
                    LOG_ERROR("strict direct weights: not all LLM tensors are streamable");
                    return false;
                }
                auto source = create_weight_payload_source(
                    model_loader.get_file_path(te_file_index), strict_direct_weights);
                if (!source || !source->open()) { LOG_ERROR("failed to open NVMe source for LLM"); return false; }
                cond_stage_model->set_weight_index(index);
                cond_stage_model->set_weight_payload_source(std::move(source));
                cond_stage_model->set_nvme_stream_mode(true);
                cond_stage_model->alloc_params_index_only();
            } else {
                get_param_tensors(cond_stage_model, module_can_mmap(SDBackendModule::TE));
            }
```
> Replace `cond_stage_model` with the actual LLM runner variable name used at that site (confirm by reading the surrounding code; the conditioner that owns the Qwen3 LLM). The LLM offloads as a whole module via `offload_params` (seen in the empirical run), which now routes through `fill_twin`.

- [ ] **Step 2: Build locally (compile check)**

Run: `cmake --build build-cpu -j6 --target sd-cli`
Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
git add src/stable-diffusion.cpp
git commit -m "feat(streaming): NVMe streaming for the LLM text encoder"
```

---

## Task 8: Full integration test (diffusion + LLM) + strict coverage

- [ ] **Step 1: Deploy + build on jarvis01** (same commands as Task 6 Step 1).

- [ ] **Step 2: Generate with both streamed; capture RSS**

```bash
ssh -o BatchMode=yes dev@jarvis01.dmz 'cd ~/z-image && /usr/bin/time -v ~/stable-diffusion.cpp/build/bin/sd-cli \
  --diffusion-model model.gguf --vae vae.sft --llm llm.gguf \
  --stream-source nvme --stream-layers --max-vram 6 --diffusion-fa \
  -p "a red apple on a wooden table, studio lighting" --cfg-scale 1.0 --steps 8 \
  -W 512 -H 512 --seed 42 -o /tmp/nvme_full.png -v 2>/tmp/nvmefull.log; \
  echo EXIT=$?; grep -E "Maximum resident|index-only|direct" /tmp/nvmefull.log'
```
Expected: `EXIT=0`; both `z_image` and the LLM show `params index-only`; **Maximum resident set size now far below the ~8.6 GB mmap baseline** (LLM no longer RAM-resident). Image still correct (compare as in Task 6 Step 3).

- [ ] **Step 3: Strict-mode smoke test**

```bash
ssh -o BatchMode=yes dev@jarvis01.dmz 'cd ~/z-image && ~/stable-diffusion.cpp/build/bin/sd-cli \
  --diffusion-model model.gguf --vae vae.sft --llm llm.gguf \
  --stream-source nvme --strict-direct-weights --stream-layers --max-vram 6 --diffusion-fa \
  -p test --steps 2 -W 256 -H 256 -o /tmp/strict.png -v 2>&1 | grep -iE "direct|strict|streamable" | head'
```
Expected: coverage logged; run proceeds if all tensors are direct-streamable, else a clear hard error (no silent fallback).

- [ ] **Step 4: Commit**

```bash
git commit --allow-empty -m "test: validated diffusion+LLM NVMe streaming + strict coverage on jarvis01"
```

---

## Task 9: Regression + docs

- [ ] **Step 1: Regression — non-streaming and cpu modes unchanged**

```bash
ssh -o BatchMode=yes dev@jarvis01.dmz 'cd ~/z-image && ~/stable-diffusion.cpp/build/bin/sd-cli \
  --diffusion-model model.gguf --vae vae.sft --llm llm.gguf --diffusion-fa \
  -p "a red apple on a wooden table" --cfg-scale 1.0 --steps 8 -W 512 -H 512 --seed 42 \
  -o /tmp/reg.png -v 2>&1 | tail -3; echo EXIT=$?'
```
Expected: `EXIT=0`, image identical to the original baseline (default path untouched).

- [ ] **Step 2: Document the feature** — create `docs/nvme_streaming.md`:

```markdown
# NVMe weight streaming (Vulkan / low-memory)

Stream diffusion + LLM weights from disk during generation so a model larger than
available memory can run. Host RAM stays bounded to a small staging buffer.

## Requirements
- A GPU runtime backend (validated on Vulkan/RADV). Not for CPU-only runs.
- `--stream-layers` and `--max-vram <GB>` are required.

## Usage
    sd-cli --diffusion-model model.gguf --vae vae.sft --llm llm.gguf \
           --stream-source nvme --stream-layers --max-vram 6 --diffusion-fa \
           -p "..." -W 512 -H 512

`--max-vram` bounds GPU memory (resident set + largest streamed segment + compute).
Larger `--max-vram` keeps more layers resident and reduces per-step NVMe reads.
`--strict-direct-weights` fails closed if any requested tensor cannot stream directly.

## Notes
- Streamed-segment weights are re-read from NVMe each sampling step (no async prefetch).
- On shared-memory APUs (e.g. BC-250), this avoids a full-model host copy competing
  with VRAM.
```

- [ ] **Step 3: Commit**

```bash
git add docs/nvme_streaming.md
git commit -m "docs: NVMe weight streaming usage"
```

---

## Self-Review notes (for the implementer)
- **Spec coverage:** flag + validation (T5), index-only load (T4/T5/T7), NVMe source (T3), fill_twin in all 3 offload sites (T4), strict coverage (T5/T7/T8), RSS/VRAM verification (T6/T8), regression (T9). VAE intentionally not streamed (out of scope).
- **Risk gate:** Task 6 must pass before Task 7. If `alloc_params_index_only` triggers ggml asserts (skeleton `data==null`), options in priority order: (a) give `index_buffer_` a 1-byte allocation; (b) special-case the planner's null-`data` handling for indexed tensors; (c) allocate real but tiny per-tensor backing only for tensors the planner dereferences pre-execute.
- **Type consistency:** `read_to_tensor(const WeightSpan&, ggml_tensor*)`, `fill_twin(ggml_tensor*, ggml_tensor*)`, `build_weight_index(ModelLoader&, const std::map<std::string,ggml_tensor*>&, size_t)`, `set_nvme_stream_mode(bool)`, `alloc_params_index_only()`, `get_file_path(size_t)` — used consistently across tasks.
- **Confirm-at-implementation:** exact runner variable names (`diffusion_model`, LLM/conditioner runner), whether the constructor returns bool vs sets a failure flag, and whether `get_param_tensors(map)` on the runner enumerates without allocating.
