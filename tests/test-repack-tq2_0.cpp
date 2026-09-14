// Checks the repacked TQ2_0 GEMV/GEMM path (CPU_REPACK buffer type) against the
// un-repacked ggml_vec_dot_tq2_0_q8_K path for batch sizes 1, 4, 5, 8, 12, 16 and 17:
// the gemm plane count and the gemv remainder of forward_mul_mat are both exercised.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

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
    const int64_t batches[]   = { 1, 4, 5, 8, 12, 16, 17 };
    const float   tol_rel     = 1e-3f;

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

    ggml_backend_free(backend);

    printf("%s\n", ok ? "test-repack-tq2_0: PASS" : "test-repack-tq2_0: FAIL");
    return ok ? 0 : 1;
}
