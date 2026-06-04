#ifndef __SD_MODEL_WEIGHT_INDEX_H__
#define __SD_MODEL_WEIGHT_INDEX_H__

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ggml.h"
#include "weight_span.h"

// Metadata-only index that maps ggml tensors to their WeightSpan entries.
// Built during model loading; used by direct-weight streaming to resolve
// on-disk payload ranges without reading payload bytes into host RAM.
class ModelWeightIndex {
public:
    // Register a WeightSpan for a tensor. Returns false if the name is
    // already registered.
    bool add_span(const std::string& tensor_name, WeightSpan span);

    // Find the WeightSpan for a ggml tensor by its name.
    const WeightSpan* find(const ggml_tensor* tensor) const;

    // Find the WeightSpan by exact tensor name string.
    const WeightSpan* find_by_name(const std::string& tensor_name) const;

    // Check whether all spans in the index are direct-streamable.
    bool all_direct_streamable() const;

    // Count of registered spans.
    size_t size() const { return spans_.size(); }

    // Get all registered spans (read-only).
    const std::map<std::string, WeightSpan>& spans() const { return spans_; }

    // Log direct-streaming coverage to stdout.
    void log_direct_coverage() const;

private:
    std::map<std::string, WeightSpan> spans_;
};

class ModelLoader;  // fwd
// Build spans for the given runtime param tensors (name -> ggml_tensor*) using the
// loader's tensor storage map. A tensor is direct_streamable only when its on-disk
// dtype matches the runtime tensor dtype (no host-side conversion) and it is not in a zip.
std::shared_ptr<ModelWeightIndex> build_weight_index(
    ModelLoader& loader,
    const std::map<std::string, struct ggml_tensor*>& runtime_tensors,
    size_t alignment);

#endif  // __SD_MODEL_WEIGHT_INDEX_H__
