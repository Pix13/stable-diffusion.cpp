#ifndef __SD_WEIGHT_SPAN_H__
#define __SD_WEIGHT_SPAN_H__

#include <array>
#include <cstdint>
#include <string>

#include "ggml.h"
#include "model_io/tensor_storage.h"

// Metadata-only descriptor for a tensor payload range in a model file.
// All fields are metadata; no payload bytes are held in this struct.
struct WeightSpan {
    std::string tensor_name;
    std::string source_path;
    uint64_t file_offset = 0;
    uint64_t payload_bytes = 0;

    // Aligned I/O range (for strict direct mode):
    //   aligned_file_offset         - 4 KiB-aligned file offset
    //   aligned_read_bytes          - 4 KiB-aligned byte count to read
    //   payload_offset_inside_aligned_read - offset of the actual payload
    //                                       within the aligned read range
    uint64_t aligned_file_offset = 0;
    uint64_t aligned_read_bytes = 0;
    uint64_t payload_offset_inside_aligned_read = 0;

    ggml_type type = GGML_TYPE_COUNT;
    std::array<int64_t, SD_MAX_DIMS> ne = {0, 0, 0, 0, 0};
    int n_dims = 0;

    // true when the on-disk bytes can be consumed by ggml without
    // host-side transformation (dtype and layout are already runtime-ready).
    bool exact_ggml_layout = false;

    // true when this span can be direct-streamed from storage to VRAM
    // without a host-RAM bounce buffer.
    bool direct_streamable = false;

    WeightSpan() = default;

    WeightSpan(std::string name, std::string path, uint64_t offset, uint64_t bytes,
               ggml_type type, const int64_t* ne, int n_dims)
        : tensor_name(std::move(name)),
          source_path(std::move(path)),
          file_offset(offset),
          payload_bytes(bytes),
          type(type),
          n_dims(n_dims) {
        for (int i = 0; i < n_dims; i++) {
            this->ne[i] = ne[i];
        }
    }

    // Compute the aligned I/O range for a given alignment boundary.
    // Returns true if the span can be aligned; false if alignment would
    // require host-side padding (which is forbidden in strict mode).
    bool compute_aligned_io(size_t alignment = 4096) {
        if (payload_bytes == 0 || alignment == 0) {
            aligned_file_offset = file_offset;
            aligned_read_bytes = payload_bytes;
            payload_offset_inside_aligned_read = 0;
            return true;
        }

        // Align the file offset down to the nearest boundary
        aligned_file_offset = (file_offset / alignment) * alignment;
        payload_offset_inside_aligned_read = file_offset - aligned_file_offset;

        // Align the read size up to cover the full payload
        uint64_t unaligned_read_bytes = payload_bytes + payload_offset_inside_aligned_read;
        aligned_read_bytes = ((unaligned_read_bytes + alignment - 1) / alignment) * alignment;

        return true;
    }

    // Check whether this span is naturally aligned (no padding needed).
    bool is_naturally_aligned(size_t alignment = 4096) const {
        if (payload_bytes == 0 || alignment == 0) return true;
        return (file_offset % alignment == 0) && (payload_bytes % alignment == 0);
    }
};

#endif  // __SD_WEIGHT_SPAN_H__
