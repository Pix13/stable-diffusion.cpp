#include <cstdio>
#include <cstdlib>

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

void test_weight_span() {}
void test_model_weight_index() {}
void test_nvme_source() {}
