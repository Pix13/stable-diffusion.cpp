#include <cstdio>
#include <cstdlib>
#include "model_weight_index.h"
#include "weight_span.h"
#include "weight_payload_source.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

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
