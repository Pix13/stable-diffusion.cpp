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

    // Read the tensor payload from storage and upload into dst's backend buffer.
    // Uses only a bounded, reusable host staging buffer; never a full-model copy.
    virtual bool read_to_tensor(const WeightSpan& span, struct ggml_tensor* dst) = 0;

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
