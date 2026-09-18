// Checks ggml_mul_mat_id over a repacked (CPU_REPACK) 3D weight against the un-repacked
// reference path. TQ2_0 (q8_K activations) takes the gemm batching and must be identical bit
// for bit. Q4_0 (q8_0 activations) is carried as a non-regression control: its repack
// kernels are not bit-exact against the reference even for a single row on the untouched
// gemv path, so it is checked against the usual 1e-3 relative tolerance instead.
//
// The repack path dispatches the rows routed to one expert as 4-row gemm calls, then sends
// a remainder of 2 or 3 rows through the same gemm zero padded to four, then gemv for a
// remainder of 1, so the interesting variable is the number of rows one expert receives
// modulo 4: rows 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12 and 17 are all exercised, which covers
// every remainder 0, 1, 2, 3 both with and without a preceding 4-row group. Five mixed
// routings additionally scatter the dst rows: 9/9/8/8, two that give an expert exactly 2 and
// exactly 3 rows, and two 88-row routings. The results must be identical bit for bit, not
// merely within tolerance: a padded row must not be able to move a real one.
//
// expert-tiles: GGML_TEST_THREADS overrides the thread list ("1,4,5") and GGML_TEST_HASH=1
// adds an FNV-1a hash of the repacked output bytes to every line. Both exist so that this
// test can be re-run under GGML_CPU_EXPERT_TILES=1, which pins one thread per cpu and so
// cannot be oversubscribed to 10 threads on a five-core pin, and so that "tiles on equals
// tiles off" can be checked as a byte comparison and not only through the tolerance.
// Neither changes what is computed.
//
// Thread counts 1, 4 and 10 are all run, because the thread count selects the work partition
// as well as the number of workers. forward_mul_mat_id splits the op by src0 columns when
// there are fewer work items than threads and by (expert, group) work items at full column
// width otherwise, so 1 thread is always full width, and the 88-row routings carry more than
// 10 work items and therefore exercise the row partition at all three counts. 10 threads on
// a 4 or 5 core box is oversubscribed, which is fine: this test times nothing.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
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

static bool type_is_repacked(ggml_backend_buffer_type_t buft, ggml_type wtype) {
    ggml_init_params wp = { ggml_tensor_overhead() * 2, nullptr, true };
    ggml_context * ctx_w = ggml_init(wp);
    ggml_tensor * w = ggml_new_tensor_3d(ctx_w, wtype, 256, 8, 2);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
    const bool repacked = buf_w != nullptr && w->extra != nullptr;
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
    return repacked;
}

// Build a routing that gives expert e exactly counts[e] rows, dealing the experts out round
// robin so that one expert's rows are scattered over the tokens rather than contiguous.
static std::vector<int32_t> ids_from_counts(const int counts[4], int64_t n_as, int64_t n_slots) {
    std::vector<int> left(counts, counts + n_as);
    std::vector<int32_t> ids;
    ids.reserve(n_slots);
    while ((int64_t) ids.size() < n_slots) {
        for (int64_t e = 0; e < n_as; e++) {
            if (left[e] > 0 && (int64_t) ids.size() < n_slots) {
                ids.push_back((int32_t) e);
                left[e]--;
            }
        }
    }
    return ids;
}

// dst[n][iu][t] = sum_k W[ids[iu][t]][n][k] * X[t][iu][k]
static bool run_mul_mat_id(ggml_backend_t backend, ggml_backend_buffer_type_t buft, ggml_type wtype,
                           const std::vector<uint8_t> & w_q, int64_t K, int64_t N, int64_t n_as,
                           const std::vector<float> & x, int64_t n_used, int64_t n_tok,
                           const std::vector<int32_t> & ids_v, std::vector<float> & out,
                           bool require_repack) {
    ggml_init_params wp = { ggml_tensor_overhead() * 2, nullptr, true };
    ggml_context * ctx_w = ggml_init(wp);
    ggml_tensor * w = ggml_new_tensor_3d(ctx_w, wtype, K, N, n_as);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
    if (buf_w == nullptr) {
        fprintf(stderr, "failed to allocate weight on %s\n", ggml_backend_buft_name(buft));
        return false;
    }
    if (require_repack && w->extra == nullptr) {
        // no repack variant of this type is registered for this CPU
        ggml_backend_buffer_free(buf_w);
        ggml_free(ctx_w);
        return false;
    }
    ggml_backend_tensor_set(w, w_q.data(), 0, w_q.size());

    ggml_init_params cp = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(cp);
    ggml_tensor * xt  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, n_used, n_tok);
    ggml_tensor * idt = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
    ggml_tensor * y   = ggml_mul_mat_id(ctx, w, xt, idt);
    ggml_cgraph * gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(xt,  x.data(),     0, x.size() * sizeof(float));
    ggml_backend_tensor_set(idt, ids_v.data(), 0, ids_v.size() * sizeof(int32_t));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        return false;
    }

    out.resize(ggml_nelements(y));
    ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
    return true;
}

// FNV-1a over the raw output bytes. Printed only under GGML_TEST_HASH=1, so that a script
// can require the expert placement to produce the SAME BYTES as the unplaced deal rather
// than merely the same tolerance.
static std::string fnv1a(const std::vector<float> & v) {
    uint64_t h = 1469598103934665603ULL;
    const uint8_t * p = (const uint8_t *) v.data();
    for (size_t i = 0; i < v.size() * sizeof(float); i++) {
        h = (h ^ p[i]) * 1099511628211ULL;
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) h);
    return std::string(buf);
}

int main(void) {
    ggml_backend_load_all();

    ggml_backend_buffer_type_t buft_ref    = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_type_t buft_repack = find_repack_buft();
    if (buft_repack == nullptr) {
        fprintf(stderr, "CPU_REPACK buffer type not available in this build, nothing to test\n");
        return 0;
    }

    if (!type_is_repacked(buft_repack, GGML_TYPE_TQ2_0)) {
        printf("test-repack-mul-mat-id: SKIP (no TQ2_0 repack variant registered for this CPU, needs NEON dotprod or i8mm)\n");
        return 0;
    }

    const ggml_type wtypes[]  = { GGML_TYPE_TQ2_0, GGML_TYPE_Q4_0 };
    // The last two are the REAL expert shapes of the trained tile MoE
    // (tile-moe-8x2-step18000-TQ2_0.gguf): ffn_gate_exps and ffn_up_exps are ne = (768,
    // 2048, 8) and ffn_down_exps is (2048, 768, 8). They are carried here so the placement
    // is checked on the shapes it will actually run, not only on the synthetic ones. n_as
    // stays 4 rather than the model's 8: n_as changes only how many experts the routing
    // spreads over, which the four map variants in expert-tiles-identity.sh already vary,
    // whereas the SHAPE is what selects the kernel, the NB_COLS rounding and the gemv/gemm
    // dispatch.
    // ... and the last two are the two-expert model's shapes (intermediate 1024):
    // ffn_gate_exps/ffn_up_exps ne = (768, 1024, 2), ffn_down_exps ne = (1024, 768, 2).
    const int64_t shapes[][2] = { { 256, 8 }, { 512, 24 }, { 2048, 512 },
                                  { 768, 2048 }, { 2048, 768 },
                                  { 768, 1024 }, { 1024, 768 } };
    const int64_t tokens[]    = { 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 17 };
    std::vector<int> threads = { 1, 4, 10 };
    if (const char * e = getenv("GGML_TEST_THREADS")) {
        threads.clear();
        for (const char * p = e; *p; ) {
            char * end = nullptr;
            const long v = strtol(p, &end, 10);
            if (end == p) { break; }
            if (v > 0) { threads.push_back((int) v); }
            p = (*end == ',') ? end + 1 : end;
        }
        if (threads.empty()) {
            fprintf(stderr, "GGML_TEST_THREADS='%s' parsed to an empty list\n", e);
            return 1;
        }
    }
    const bool want_hash = getenv("GGML_TEST_HASH") != nullptr && atoi(getenv("GGML_TEST_HASH")) == 1;
    const int64_t n_as        = 4;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    bool ok = true;

    for (const ggml_type wtype : wtypes) {
      if (!type_is_repacked(buft_repack, wtype)) {
        printf("skipping %s, no repack variant registered for this CPU\n", ggml_type_name(wtype));
        continue;
      }
      for (const int nth : threads) {
        ggml_backend_t backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backend, nth);

        for (const auto & shape : shapes) {
            const int64_t K = shape[0];
            const int64_t N = shape[1];

            std::vector<float> w_f(K * N * n_as);
            for (auto & v : w_f) {
                v = dist(rng);
            }
            std::vector<uint8_t> w_q(ggml_row_size(wtype, K) * N * n_as);
            ggml_quantize_chunk(wtype, w_f.data(), w_q.data(), 0, N * n_as, K, nullptr);

            // case 1: every token routed to the same expert, so that expert receives
            //         exactly n_tok rows and the other three receive none
            for (const int64_t n_tok : tokens) {
                const int64_t n_used = 1;
                std::vector<float> x(K * n_used * n_tok);
                for (auto & v : x) {
                    v = dist(rng);
                }
                std::vector<int32_t> ids(n_used * n_tok, 1); // expert 1 takes everything

                std::vector<float> out_ref;
                std::vector<float> out_rep;
                if (!run_mul_mat_id(backend, buft_ref, wtype, w_q, K, N, n_as, x, n_used, n_tok, ids, out_ref, false)) {
                    return 1;
                }
                if (!run_mul_mat_id(backend, buft_repack, wtype, w_q, K, N, n_as, x, n_used, n_tok, ids, out_rep, true)) {
                    return 1;
                }

                float max_abs = 0.0f;
                float max_diff = 0.0f;
                for (size_t i = 0; i < out_ref.size(); i++) {
                    max_abs  = std::max(max_abs, std::fabs(out_ref[i]));
                    max_diff = std::max(max_diff, std::fabs(out_ref[i] - out_rep[i]));
                }
                const float tol  = (wtype == GGML_TYPE_TQ2_0) ? 0.0f : 1e-3f*max_abs;
                const bool  pass = std::isfinite(max_diff) && max_diff <= tol;
                printf("%-6s t=%d K=%5d N=%4d  one expert, rows=%3d  max|ref|=%10.4f  max diff=%.3e  %s%s%s\n",
                       ggml_type_name(wtype), nth, (int) K, (int) N, (int) n_tok, max_abs, max_diff, pass ? "OK" : "FAIL",
                       want_hash ? "  fnv " : "", want_hash ? fnv1a(out_rep).c_str() : "");
                ok = ok && pass;
            }

            // case 2: mixed routing, two experts per token, so the four experts receive
            //         different row counts and the dst rows are genuinely scattered
            {
                const int64_t n_used = 2;
                const int64_t n_tok  = 17;
                std::vector<float> x(K * n_used * n_tok);
                for (auto & v : x) {
                    v = dist(rng);
                }
                std::vector<int32_t> ids(n_used * n_tok);
                int counts[4] = { 0, 0, 0, 0 };
                for (int64_t t = 0; t < n_tok; t++) {
                    for (int64_t u = 0; u < n_used; u++) {
                        const int32_t e = (int32_t) ((3 * t + u) % n_as);
                        ids[t * n_used + u] = e;
                        counts[e]++;
                    }
                }

                std::vector<float> out_ref;
                std::vector<float> out_rep;
                if (!run_mul_mat_id(backend, buft_ref, wtype, w_q, K, N, n_as, x, n_used, n_tok, ids, out_ref, false)) {
                    return 1;
                }
                if (!run_mul_mat_id(backend, buft_repack, wtype, w_q, K, N, n_as, x, n_used, n_tok, ids, out_rep, true)) {
                    return 1;
                }

                float max_abs = 0.0f;
                float max_diff = 0.0f;
                for (size_t i = 0; i < out_ref.size(); i++) {
                    max_abs  = std::max(max_abs, std::fabs(out_ref[i]));
                    max_diff = std::max(max_diff, std::fabs(out_ref[i] - out_rep[i]));
                }
                const float tol  = (wtype == GGML_TYPE_TQ2_0) ? 0.0f : 1e-3f*max_abs;
                const bool  pass = std::isfinite(max_diff) && max_diff <= tol;
                printf("%-6s t=%d K=%5d N=%4d  mixed, rows=%d/%d/%d/%d  max|ref|=%10.4f  max diff=%.3e  %s%s%s\n",
                       ggml_type_name(wtype), nth, (int) K, (int) N, counts[0], counts[1], counts[2], counts[3],
                       max_abs, max_diff, pass ? "OK" : "FAIL",
                       want_hash ? "  fnv " : "", want_hash ? fnv1a(out_rep).c_str() : "");
                ok = ok && pass;
            }

            // case 3: mixed routings chosen so that one expert receives exactly 2 rows and
            //         another exactly 3, the two counts the 2-row gemm is there for, with
            //         the dst rows scattered as in case 2. 2/3/4/1 has no 4-row group before
            //         the pair, 6/7/2/3 does.
            {
                // The last two are 88 rows over the four experts, which is more than 10 work
                // items however they fall, so the (expert, group) row partition is live at
                // every thread count this test runs.
                const int routings[4][4] = { { 2, 3, 4, 1 }, { 6, 7, 2, 3 },
                                             { 40, 9, 23, 16 }, { 17, 31, 22, 18 } };
                for (const auto & counts : routings) {
                    const int64_t n_used = 2;
                    const int64_t n_rows = counts[0] + counts[1] + counts[2] + counts[3];
                    const int64_t n_tok  = n_rows / n_used;
                    std::vector<float> x(K * n_used * n_tok);
                    for (auto & v : x) {
                        v = dist(rng);
                    }
                    const std::vector<int32_t> ids = ids_from_counts(counts, n_as, n_used * n_tok);

                    std::vector<float> out_ref;
                    std::vector<float> out_rep;
                    if (!run_mul_mat_id(backend, buft_ref, wtype, w_q, K, N, n_as, x, n_used, n_tok, ids, out_ref, false)) {
                        return 1;
                    }
                    if (!run_mul_mat_id(backend, buft_repack, wtype, w_q, K, N, n_as, x, n_used, n_tok, ids, out_rep, true)) {
                        return 1;
                    }

                    float max_abs = 0.0f;
                    float max_diff = 0.0f;
                    for (size_t i = 0; i < out_ref.size(); i++) {
                        max_abs  = std::max(max_abs, std::fabs(out_ref[i]));
                        max_diff = std::max(max_diff, std::fabs(out_ref[i] - out_rep[i]));
                    }
                    const float tol  = (wtype == GGML_TYPE_TQ2_0) ? 0.0f : 1e-3f*max_abs;
                    const bool  pass = std::isfinite(max_diff) && max_diff <= tol;
                    printf("%-6s t=%d K=%5d N=%4d  mixed, rows=%d/%d/%d/%d  max|ref|=%10.4f  max diff=%.3e  %s%s%s\n",
                           ggml_type_name(wtype), nth, (int) K, (int) N, counts[0], counts[1], counts[2], counts[3],
                           max_abs, max_diff, pass ? "OK" : "FAIL",
                           want_hash ? "  fnv " : "", want_hash ? fnv1a(out_rep).c_str() : "");
                    ok = ok && pass;
                }
            }
        }

        ggml_backend_free(backend);
      }
    }

    printf("%s\n", ok ? "test-repack-mul-mat-id: PASS" : "test-repack-mul-mat-id: FAIL");
    return ok ? 0 : 1;
}
