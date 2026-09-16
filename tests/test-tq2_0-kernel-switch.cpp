// Bit-for-bit comparison of the two TQ2_0 dot-product kernels selected by
// GGML_CPU_TQ2_KERNEL, through ggml_mul_mat on a CPU_REPACK weight, at every batch size
// from 1 to 17 (1 takes the GEMV path, 4 and up take the GEMM path, and the odd sizes
// exercise the gemm2 and single-row tails).
//
//   old = shift-and-mask unpack, the kernel that shipped on moe-gemm-v8
//   new = mask-only unpack with per-scale accumulators, the candidate
//
// The bar is exact equality of the output bytes, not a tolerance. The run also asserts that
// the switch itself is live (ggml_tq2_kernel_new() follows the environment) and ends with a
// sabotage arm: the same comparison against a weight with one code byte changed, which MUST
// report a difference. Without that arm an equality test would pass on a comparator that
// cannot fail.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-cpu/tq2-kernel.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static ggml_backend_buffer_type_t find_repack_buft(void) {
    ggml_backend_reg_t reg = ggml_backend_cpu_reg();
    auto * get_extra_bufts = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    if (get_extra_bufts == nullptr) {
        return nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    ggml_backend_buffer_type_t * bufts = get_extra_bufts(dev);
    while (bufts && *bufts) {
        if (strcmp(ggml_backend_buft_name(*bufts), "CPU_REPACK") == 0) {
            return *bufts;
        }
        bufts++;
    }
    return nullptr;
}

static bool tq2_0_is_repacked(ggml_backend_buffer_type_t buft) {
    ggml_init_params wp = { ggml_tensor_overhead() * 2, nullptr, true };
    ggml_context * ctx_w = ggml_init(wp);
    ggml_tensor * w = ggml_new_tensor_2d(ctx_w, GGML_TYPE_TQ2_0, 256, 8);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
    const bool repacked = buf_w != nullptr && w->extra != nullptr;
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
    return repacked;
}

static bool run_mul_mat(ggml_backend_t backend, ggml_backend_buffer_type_t buft, const std::vector<uint8_t> & w_q,
                        int64_t K, int64_t N, const std::vector<float> & x, int64_t B, std::vector<float> & out) {
    ggml_init_params wp = { ggml_tensor_overhead() * 2, nullptr, true };
    ggml_context * ctx_w = ggml_init(wp);
    ggml_tensor * w = ggml_new_tensor_2d(ctx_w, GGML_TYPE_TQ2_0, K, N);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
    if (buf_w == nullptr || w->extra == nullptr) {
        fprintf(stderr, "weight did not get repack traits\n");
        return false;
    }
    ggml_backend_tensor_set(w, w_q.data(), 0, w_q.size());

    ggml_init_params cp = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(cp);
    ggml_tensor * xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, B);
    ggml_tensor * y  = ggml_mul_mat(ctx, w, xt);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(xt, x.data(), 0, x.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        return false;
    }

    out.resize(N * B);
    ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
    return true;
}

static void select_kernel(const char * which) {
    setenv("GGML_CPU_TQ2_KERNEL", which, 1);
    ggml_tq2_kernel_init();
}

// number of output elements that differ in any bit
static size_t count_diff(const std::vector<float> & a, const std::vector<float> & b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size(); i++) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            n++;
        }
    }
    return n;
}

int main(void) {
    ggml_backend_load_all();

    ggml_backend_buffer_type_t buft_repack = find_repack_buft();
    if (buft_repack == nullptr || !tq2_0_is_repacked(buft_repack)) {
        printf("test-tq2_0-kernel-switch: SKIP (no TQ2_0 repack variant registered for this CPU)\n");
        return 0;
    }

    // the switch must actually follow the environment, or everything below compares a
    // kernel with itself
    select_kernel("old");
    const bool flag_old = ggml_tq2_kernel_new();
    select_kernel("new");
    const bool flag_new = ggml_tq2_kernel_new();
    unsetenv("GGML_CPU_TQ2_KERNEL");
    ggml_tq2_kernel_init();
    const bool flag_default = ggml_tq2_kernel_new();
    printf("switch: old -> new_kernel=%d, new -> new_kernel=%d, unset -> new_kernel=%d\n",
           (int) flag_old, (int) flag_new, (int) flag_default);
    if (flag_old != false || flag_new != true || flag_default != true) {
        printf("test-tq2_0-kernel-switch: FAIL (GGML_CPU_TQ2_KERNEL is not live)\n");
        return 1;
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    const int n_threads = 4;
    ggml_backend_cpu_set_n_threads(backend, n_threads);

    const int64_t shapes[][2] = { { 256, 8 }, { 512, 24 }, { 2560, 128 }, { 1024, 40 } };

    std::mt19937 rng(20260916);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    bool ok = true;
    size_t cases = 0;

    for (const auto & shape : shapes) {
        const int64_t K = shape[0];
        const int64_t N = shape[1];

        std::vector<float> w_f(K * N);
        for (auto & v : w_f) {
            v = dist(rng);
        }
        std::vector<uint8_t> w_q(ggml_row_size(GGML_TYPE_TQ2_0, K) * N);
        ggml_quantize_chunk(GGML_TYPE_TQ2_0, w_f.data(), w_q.data(), 0, N, K, nullptr);

        size_t worst = 0;
        for (int64_t B = 1; B <= 17; B++) {
            std::vector<float> x(K * B);
            for (auto & v : x) {
                v = dist(rng);
            }

            std::vector<float> out_old;
            std::vector<float> out_new;

            select_kernel("old");
            if (!run_mul_mat(backend, buft_repack, w_q, K, N, x, B, out_old)) { return 1; }
            select_kernel("new");
            if (!run_mul_mat(backend, buft_repack, w_q, K, N, x, B, out_new)) { return 1; }

            const size_t ndiff = count_diff(out_old, out_new);
            worst = std::max(worst, ndiff);
            cases++;
            if (ndiff != 0) {
                float md = 0.0f;
                for (size_t i = 0; i < out_old.size(); i++) {
                    md = std::max(md, std::fabs(out_old[i] - out_new[i]));
                }
                printf("K=%5d N=%4d B=%3d  differing elements %zu of %zu  max abs diff %.3e  FAIL\n",
                       (int) K, (int) N, (int) B, ndiff, out_old.size(), md);
                ok = false;
            }
        }
        printf("K=%5d N=%4d  B=1..17, %d threads: differing elements %zu  %s\n",
               (int) K, (int) N, n_threads, worst, worst == 0 ? "OK" : "FAIL");
    }

    // Sabotage: one code byte of the weight changed under the new kernel only. The same
    // comparison must now report differences; if it does not, the comparison above proved
    // nothing.
    {
        const int64_t K = 2560, N = 128, B = 17;
        std::vector<float> w_f(K * N);
        for (auto & v : w_f) { v = dist(rng); }
        std::vector<uint8_t> w_q(ggml_row_size(GGML_TYPE_TQ2_0, K) * N);
        ggml_quantize_chunk(GGML_TYPE_TQ2_0, w_f.data(), w_q.data(), 0, N, K, nullptr);
        std::vector<float> x(K * B);
        for (auto & v : x) { v = dist(rng); }

        std::vector<float> out_old, out_sab;
        select_kernel("old");
        if (!run_mul_mat(backend, buft_repack, w_q, K, N, x, B, out_old)) { return 1; }

        std::vector<uint8_t> w_sab = w_q;
        w_sab[7] ^= 0x01;  // one 2-bit code in the first block of the first row
        select_kernel("new");
        if (!run_mul_mat(backend, buft_repack, w_sab, K, N, x, B, out_sab)) { return 1; }

        const size_t ndiff = count_diff(out_old, out_sab);
        printf("sabotage (one code byte flipped): differing elements %zu of %zu  %s\n",
               ndiff, out_old.size(), ndiff > 0 ? "OK (the comparison can fail)" : "FAIL (comparison is vacuous)");
        ok = ok && ndiff > 0;
        cases++;
    }

    ggml_backend_free(backend);
    printf("%s (%zu cases)\n", ok ? "test-tq2_0-kernel-switch: PASS" : "test-tq2_0-kernel-switch: FAIL", cases);
    return ok ? 0 : 1;
}
