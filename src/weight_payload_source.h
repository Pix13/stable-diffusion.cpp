#ifndef __SD_WEIGHT_PAYLOAD_SOURCE_H__
#define __SD_WEIGHT_PAYLOAD_SOURCE_H__

#include <cstddef>
#include <memory>
#include <string>

#include "ggml-backend.h"
#include "weight_span.h"

// Abstract interface for tensor payload sources.
// Implementations provide the mechanism to read tensor payload bytes
// from storage into a destination GPU backend buffer.
class WeightPayloadSource {
public:
    virtual ~WeightPayloadSource() = default;

    explicit WeightPayloadSource(const std::string& path)
        : source_path_(path) {}

    // Open the source (e.g., open file descriptors, initialize libraries).
    // Returns true on success.
    virtual bool open() = 0;

    // Whether this source supports strict direct mode (no host-RAM staging).
    virtual bool supports_strict_direct() const = 0;

    // Read a tensor payload directly into a device buffer.
    //   span             - the WeightSpan describing the on-disk range
    //   dst_buffer       - the ggml backend buffer containing the destination
    //   dst_device_ptr   - the device pointer within dst_buffer
    //   dst_offset       - byte offset within dst_device_ptr
    //   backend_stream   - optional backend stream (nullptr for synchronous)
    // Returns true on success.
    virtual bool read_to_device(const WeightSpan& span,
                                ggml_backend_buffer_t dst_buffer,
                                void* dst_device_ptr,
                                size_t dst_offset,
                                void* backend_stream_or_null) = 0;

    // Close the source and release resources.
    virtual void close() = 0;

    // Get the source path (model file or pack file).
    virtual const std::string& path() const = 0;

protected:
    std::string source_path_;
};

// Factory: create a WeightPayloadSource for the given path and mode.
//   strict - if true, the returned source must support strict direct mode
std::unique_ptr<WeightPayloadSource> create_weight_payload_source(
    const std::string& path, bool strict);

#endif  // __SD_WEIGHT_PAYLOAD_SOURCE_H__
