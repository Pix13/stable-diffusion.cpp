#include "weight_payload_source.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "ggml-backend.h"

// HostRamWeightSource: reads tensor payloads from a model file into
// host RAM, then copies to the destination GPU buffer via ggml.
// This is the non-strict path (payload bytes pass through host RAM).
class HostRamWeightSource : public WeightPayloadSource {
public:
    explicit HostRamWeightSource(const std::string& path)
        : WeightPayloadSource(path) {}

    bool open() override {
        file_.open(source_path_, std::ios::binary | std::ios::ate);
        if (!file_.is_open()) {
            fprintf(stderr, "failed to open weight source file: %s\n", source_path_.c_str());
            return false;
        }
        file_size_ = file_.tellg();
        file_.seekg(0, std::ios::beg);
        return true;
    }

    bool supports_strict_direct() const override {
        return false;  // Host RAM staging is not strict.
    }

    bool read_to_device(const WeightSpan& span,
                        ggml_backend_buffer_t dst_buffer,
                        void* dst_device_ptr,
                        size_t dst_offset,
                        void* /*backend_stream_or_null*/) override {
        if (!file_.is_open()) {
            fprintf(stderr, "HostRamWeightSource: file not open\n");
            return false;
        }

        // Read payload bytes into a host buffer.
        std::vector<uint8_t> buffer(span.payload_bytes);
        file_.seekg(static_cast<std::streampos>(span.file_offset));
        file_.read(reinterpret_cast<char*>(buffer.data()),
                   static_cast<std::streamsize>(span.payload_bytes));

        if (file_.gcount() != static_cast<std::streamsize>(span.payload_bytes)) {
            fprintf(stderr, "HostRamWeightSource: short read for tensor '%s': "
                    "expected %zu, got %zd\n",
                    span.tensor_name.c_str(),
                    span.payload_bytes,
                    static_cast<std::streamsize>(span.payload_bytes));
            return false;
        }

        // Copy host buffer to destination device pointer.
        memcpy(static_cast<uint8_t*>(dst_device_ptr) + dst_offset,
               buffer.data(), span.payload_bytes);

        return true;
    }

    void close() override {
        if (file_.is_open()) {
            file_.close();
        }
    }

    const std::string& path() const override {
        return source_path_;
    }

private:
    std::ifstream file_;
    std::streampos file_size_ = 0;
};

// Factory implementation (CPU-only version; GDS version overrides this
// when SD_CUDA_GDS is enabled).
std::unique_ptr<WeightPayloadSource> create_weight_payload_source(
    const std::string& path, bool strict) {
    if (strict) {
        fprintf(stderr, "strict direct weights requested, but no GDS source "
                "available (rebuild with -DSD_CUDA_GDS=ON).\n");
        return nullptr;
    }
    return std::make_unique<HostRamWeightSource>(path);
}
