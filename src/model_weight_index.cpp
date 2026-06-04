#include "model_weight_index.h"

#include <cstdio>
#include <sstream>

#include "model.h"

bool ModelWeightIndex::add_span(const std::string& tensor_name, WeightSpan span) {
    auto result = spans_.emplace(tensor_name, std::move(span));
    return result.second;  // true if inserted (not already present)
}

const WeightSpan* ModelWeightIndex::find(const ggml_tensor* tensor) const {
    if (!tensor) return nullptr;
    return find_by_name(tensor->name);
}

const WeightSpan* ModelWeightIndex::find_by_name(const std::string& tensor_name) const {
    auto it = spans_.find(tensor_name);
    if (it != spans_.end()) {
        return &it->second;
    }
    return nullptr;
}

bool ModelWeightIndex::all_direct_streamable() const {
    for (const auto& [name, span] : spans_) {
        if (!span.direct_streamable) {
            return false;
        }
    }
    return !spans_.empty();
}

void ModelWeightIndex::log_direct_coverage() const {
    size_t total_bytes = 0;
    size_t direct_bytes = 0;
    size_t direct_count = 0;

    for (const auto& [name, span] : spans_) {
        total_bytes += span.payload_bytes;
        if (span.direct_streamable) {
            direct_bytes += span.payload_bytes;
            direct_count++;
        }
    }

    std::stringstream ss;
    ss << "[direct weights] " << spans_.size() << " tensors, "
       << total_bytes / (1024 * 1024) << " MB total, "
       << direct_count << " direct-streamable ("
       << direct_bytes / (1024 * 1024) << " MB)";
    if (!spans_.empty() && direct_bytes < total_bytes) {
        ss << " - some tensors require host-side conversion";
    }
    printf("%s\n", ss.str().c_str());
}

std::shared_ptr<ModelWeightIndex> build_weight_index(
    ModelLoader& loader,
    const std::map<std::string, ggml_tensor*>& runtime_tensors,
    size_t alignment) {
    auto index = std::make_shared<ModelWeightIndex>();
    auto& storage_map = loader.get_tensor_storage_map();
    for (const auto& [name, tensor] : runtime_tensors) {
        auto it = storage_map.find(name);
        if (it == storage_map.end()) {
            continue;  // not all runtime tensors are file-backed
        }
        const TensorStorage& ts = it->second;
        const std::string& path = loader.get_file_path(ts.file_index);
        WeightSpan span(name, path, ts.offset, ggml_nbytes(tensor),
                        tensor->type, tensor->ne, ggml_n_dims(tensor));
        span.compute_aligned_io(alignment);
        span.direct_streamable = (ts.type == tensor->type) && (ts.index_in_zip < 0);
        index->add_span(name, span);
    }
    return index;
}
