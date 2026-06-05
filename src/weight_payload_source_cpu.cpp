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
        if (fd_ >= 0) { return true; }
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

    bool supports_strict_direct() const override { return false; }

    bool read_to_tensor(const WeightSpan& span, ggml_tensor* dst) override {
        if (fd_ < 0) { fprintf(stderr, "NvmeStagedWeightSource: not open\n"); return false; }
        const size_t   want  = span.aligned_read_bytes ? span.aligned_read_bytes : span.payload_bytes;
        const uint64_t off   = span.aligned_read_bytes ? span.aligned_file_offset : span.file_offset;
        const size_t   inset = span.aligned_read_bytes ? span.payload_offset_inside_aligned_read : 0;
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
    uint8_t* buf_    = nullptr;
    size_t   cap_    = 0;
};

std::unique_ptr<WeightPayloadSource> create_weight_payload_source(
    const std::string& path, bool /*strict*/) {
    return std::make_unique<NvmeStagedWeightSource>(path);
}
