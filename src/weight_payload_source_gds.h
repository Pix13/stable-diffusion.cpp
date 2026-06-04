#ifndef __SD_WEIGHT_PAYLOAD_SOURCE_GDS_H__
#define __SD_WEIGHT_PAYLOAD_SOURCE_GDS_H__

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml-backend.h"
#include "weight_payload_source.h"

// CudaGdsWeightSource: reads tensor payloads directly from NVMe storage
// into registered GPU memory using NVIDIA GPUDirect Storage (cuFile).
// This is the strict direct path — no host-RAM payload staging.
class CudaGdsWeightSource : public WeightPayloadSource {
public:
    explicit CudaGdsWeightSource(const std::string& path);
    ~CudaGdsWeightSource() override;

    bool open() override;
    bool supports_strict_direct() const override { return true; }
    // NOTE: GDS path is unverified (no hardware) and out of test scope.
    bool read_to_tensor(const WeightSpan& span, struct ggml_tensor* dst) override;
    void close() override;
    const std::string& path() const override { return source_path_; }

    // Register a device buffer for cuFile reads. Must be called before
    // read_to_device targets that buffer.
    bool register_buffer(void* device_ptr, size_t size);

    // Deregister all buffers.
    void deregister_all_buffers();

private:
    bool init_cufile();
    bool register_file_handle();
    void cleanup();

    // source_path_ is owned by the WeightPayloadSource base class.
    int fd_ = -1;

    // cuFile handles (opaque types loaded dynamically).
    void* cufile_handle_ = nullptr;   // CUfileHandle_t
    void* cufile_desc_   = nullptr;   // CUfileDescr_t (owned)

    // Registered device buffers: ptr -> size
    std::unordered_map<void*, size_t> registered_buffers_;

    // cuFile library handle (for dynamic loading).
    void* libcufile_handle_ = nullptr;
};

// Factory: create a CudaGdsWeightSource for the given path.
// Returns nullptr if cuFile is unavailable.
std::unique_ptr<CudaGdsWeightSource> create_cuda_gds_source(const std::string& path);

#endif  // __SD_WEIGHT_PAYLOAD_SOURCE_GDS_H__
