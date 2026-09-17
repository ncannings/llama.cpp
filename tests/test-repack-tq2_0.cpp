// Two checks on the repacked TQ2_0 GEMV/GEMM path (CPU_REPACK buffer type).
//
// 1. AGAINST THE UN-REPACKED ggml_vec_dot_tq2_0_q8_K PATH, at a tolerance. This one cannot be
//    exact and is not meant to be: the two paths associate the floats differently. vec_dot
//    keeps an eight-lane float accumulator in which lane k is a partial sum of ONE output
//    column, converts and scales each lane separately and horizontally sums at the end; the
//    repacked path has eight output COLUMNS in its eight lanes, so a column's block sum is one
//    int32 that is converted and scaled once. Same integers, different float tree. The arm
//    exists to catch a wrong layout or wrong addressing, which shows up as a large difference,
//    not as a last-bit one.
//
// 2. AGAINST THE SCALAR GENERIC REPACK KERNELS, exactly. GGML_CPU_TQ2_KERNEL=old routes the
//    repacked path to ggml_gemv_tq2_0_8x*_q8_K_generic and ggml_gemm_tq2_0_8x*_q8_K_generic,
//    which share the float epilogue of the vectorised form operation for operation. That makes
//    equality of the output BYTES the right bar, and it is the bar the x86 VNNI kernels (and
//    the Arm ones) are held to here, at every batch size from 1 to 17. A sabotage arm follows
//    it, because an equality test that cannot fail proves nothing.

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

// out[b][n] = sum_k W[n][k] * X[b][k] with W allocated on buft
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
                        int64_t K, int64_t N, const std::vector<float> & x, int64_t B, std::vector<float> & out, bool require_repack) {
    ggml_init_params wp = { ggml_tensor_overhead() * 2, nullptr, true };
    ggml_context * ctx_w = ggml_init(wp);
    ggml_tensor * w = ggml_new_tensor_2d(ctx_w, GGML_TYPE_TQ2_0, K, N);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
    if (buf_w == nullptr) {
        fprintf(stderr, "failed to allocate weight on %s\n", ggml_backend_buft_name(buft));
        return false;
    }
    if (require_repack && w->extra == nullptr) {
        // no TQ2_0 repack variant is registered for this CPU (needs NEON dotprod or i8mm)
        ggml_backend_buffer_free(buf_w);
        ggml_free(ctx_w);
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

int main(void) {
    ggml_backend_load_all();

    ggml_backend_buffer_type_t buft_ref    = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_type_t buft_repack = find_repack_buft();
    if (buft_repack == nullptr) {
        fprintf(stderr, "CPU_REPACK buffer type not available in this build, nothing to test\n");
        return 0;
    }

    if (!tq2_0_is_repacked(buft_repack)) {
        printf("test-repack-tq2_0: SKIP (no TQ2_0 repack variant registered for this CPU, needs NEON dotprod or i8mm)\n");
        return 0;
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 4);

    const int64_t shapes[][2] = { { 256, 8 }, { 512, 24 }, { 2560, 128 } };
    const int64_t batches[]   = { 1, 4, 17 };
    const float   tol_rel     = 1e-3f;

    // shapes for the exact arm: a single block, a long row, one column group with many blocks,
    // and a column count that is a multiple of the 8-row interleave but not of 16
    const int64_t xshapes[][2] = { { 256, 8 }, { 1024, 40 }, { 3072, 8 }, { 512, 72 } };

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    bool ok = true;
    for (const auto & shape : shapes) {
        const int64_t K = shape[0];
        const int64_t N = shape[1];

        std::vector<float> w_f(K * N);
        for (auto & v : w_f) {
            v = dist(rng);
        }
        std::vector<uint8_t> w_q(ggml_row_size(GGML_TYPE_TQ2_0, K) * N);
        ggml_quantize_chunk(GGML_TYPE_TQ2_0, w_f.data(), w_q.data(), 0, N, K, nullptr);

        for (int64_t B : batches) {
            std::vector<float> x(K * B);
            for (auto & v : x) {
                v = dist(rng);
            }

            std::vector<float> out_ref;
            std::vector<float> out_rep;
            if (!run_mul_mat(backend, buft_ref, w_q, K, N, x, B, out_ref, false)) {
                return 1;
            }
            if (!run_mul_mat(backend, buft_repack, w_q, K, N, x, B, out_rep, true)) {
                return 1;
            }

            float max_abs = 0.0f;
            float max_diff = 0.0f;
            for (size_t i = 0; i < out_ref.size(); i++) {
                max_abs  = std::max(max_abs, std::fabs(out_ref[i]));
                max_diff = std::max(max_diff, std::fabs(out_ref[i] - out_rep[i]));
            }
            const float tol = tol_rel * max_abs;
            const bool pass = std::isfinite(max_diff) && max_diff <= tol;
            printf("K=%5d N=%4d B=%3d  max|ref|=%10.4f  max diff=%.3e  tol=%.3e  %s\n",
                   (int) K, (int) N, (int) B, max_abs, max_diff, tol, pass ? "OK" : "FAIL");
            ok = ok && pass;
        }
    }

    // -------------------------------------------------------------------------------------
    // exact arm: the vectorised repack kernels against the scalar generic reference, B = 1..17
    printf("-- exact arm: repacked kernels vs the scalar generic reference (bar: 0 differing bytes)\n");
    for (const auto & shape : xshapes) {
        const int64_t K = shape[0];
        const int64_t N = shape[1];

        std::vector<float> w_f(K * N);
        for (auto & v : w_f) {
            v = dist(rng);
        }
        std::vector<uint8_t> w_q(ggml_row_size(GGML_TYPE_TQ2_0, K) * N);
        ggml_quantize_chunk(GGML_TYPE_TQ2_0, w_f.data(), w_q.data(), 0, N, K, nullptr);

        size_t worst  = 0;
        float  worstd = 0.0f;
        for (int64_t B = 1; B <= 17; B++) {
            std::vector<float> x(K * B);
            for (auto & v : x) {
                v = dist(rng);
            }

            std::vector<float> out_old, out_new;
            setenv("GGML_CPU_TQ2_KERNEL", "old", 1); ggml_tq2_kernel_init();
            if (!run_mul_mat(backend, buft_repack, w_q, K, N, x, B, out_old, true)) { return 1; }
            setenv("GGML_CPU_TQ2_KERNEL", "new", 1); ggml_tq2_kernel_init();
            if (!run_mul_mat(backend, buft_repack, w_q, K, N, x, B, out_new, true)) { return 1; }

            for (size_t i = 0; i < out_old.size(); i++) {
                if (memcmp(&out_old[i], &out_new[i], sizeof(float)) != 0) {
                    worst++;
                    worstd = std::max(worstd, std::fabs(out_old[i] - out_new[i]));
                }
            }
        }
        printf("K=%5d N=%4d B=1..17  differing elements=%zu  max diff=%.3e  %s\n",
               (int) K, (int) N, worst, worstd, worst == 0 ? "OK" : "FAIL");
        ok = ok && worst == 0;
    }

    // sabotage: one code byte flipped under the new kernel only, which MUST be visible
    {
        const int64_t K = 1024, N = 40, B = 17;
        std::vector<float> w_f(K * N);
        for (auto & v : w_f) { v = dist(rng); }
        std::vector<uint8_t> w_q(ggml_row_size(GGML_TYPE_TQ2_0, K) * N);
        ggml_quantize_chunk(GGML_TYPE_TQ2_0, w_f.data(), w_q.data(), 0, N, K, nullptr);
        std::vector<float> x(K * B);
        for (auto & v : x) { v = dist(rng); }

        std::vector<float> out_old, out_sab;
        setenv("GGML_CPU_TQ2_KERNEL", "old", 1); ggml_tq2_kernel_init();
        if (!run_mul_mat(backend, buft_repack, w_q, K, N, x, B, out_old, true)) { return 1; }

        std::vector<uint8_t> w_sab = w_q;
        w_sab[9] ^= 0x04;
        setenv("GGML_CPU_TQ2_KERNEL", "new", 1); ggml_tq2_kernel_init();
        if (!run_mul_mat(backend, buft_repack, w_sab, K, N, x, B, out_sab, true)) { return 1; }

        size_t ndiff = 0;
        for (size_t i = 0; i < out_old.size(); i++) {
            if (memcmp(&out_old[i], &out_sab[i], sizeof(float)) != 0) { ndiff++; }
        }
        printf("sabotage (one code byte flipped): differing elements %zu of %zu  %s\n",
               ndiff, out_old.size(), ndiff > 0 ? "OK (the comparison can fail)" : "FAIL (comparison is vacuous)");
        ok = ok && ndiff > 0;
    }

    // the interleave gate: a row count that is not a multiple of 8 must NOT be repacked
    {
        ggml_init_params wp = { ggml_tensor_overhead() * 2, nullptr, true };
        ggml_context * ctx_w = ggml_init(wp);
        ggml_tensor * w = ggml_new_tensor_2d(ctx_w, GGML_TYPE_TQ2_0, 512, 12);
        ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft_repack);
        const bool repacked = buf_w != nullptr && w->extra != nullptr;
        printf("interleave gate: ne[1]=12 repacked=%d  %s\n", (int) repacked, repacked ? "FAIL" : "OK");
        ok = ok && !repacked;
        ggml_backend_buffer_free(buf_w);
        ggml_free(ctx_w);
    }

    unsetenv("GGML_CPU_TQ2_KERNEL");
    ggml_tq2_kernel_init();

    ggml_backend_free(backend);

    printf("%s\n", ok ? "test-repack-tq2_0: PASS" : "test-repack-tq2_0: FAIL");
    return ok ? 0 : 1;
}
