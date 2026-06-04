#include <cstdio>
#include <cstdlib>
#include "weight_span.h"

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

void test_weight_span() {
    int64_t ne[1] = {100};
    WeightSpan s("t", "/m.gguf", /*offset*/ 5000, /*bytes*/ 100, GGML_TYPE_F32, ne, 1);
    CHECK(s.compute_aligned_io(4096) == true);
    CHECK(s.aligned_file_offset == 4096);                       // 5000 -> down to 4096
    CHECK(s.payload_offset_inside_aligned_read == 904);         // 5000 - 4096
    CHECK(s.aligned_read_bytes == 4096);                        // ceil(904+100, 4096)
    CHECK(s.aligned_file_offset + s.aligned_read_bytes >= s.file_offset + s.payload_bytes);
    WeightSpan s2("t2", "/m.gguf", 0, 200, GGML_TYPE_F32, ne, 1);
    s2.compute_aligned_io(4096);
    CHECK(s2.direct_streamable == false);   // unchanged by alignment
}
void test_model_weight_index() {}
void test_nvme_source() {}
