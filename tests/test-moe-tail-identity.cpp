// Bit-identity gate for the moe-tail work-partition and barrier-elision changes.
//
// WHAT IT TESTS. Four changes alter WHICH THREAD does a piece of work, or WHICH BARRIER
// a thread waits at, and none of them may alter a single output byte:
//
//   GGML_CPU_NO_ROWS_FLATTEN   RMS_NORM partitions the flattened row index instead of
//                              splitting ne01 alone inside a loop over ne02
//   GGML_CPU_NO_BCAST_SCALAR   a row broadcast by one scalar runs as one loop instead of
//                              ne00 calls to a one-element vector op
//   GGML_CPU_NO_SUMROWS_PAR    SUM_ROWS is split over rows instead of run on thread 0
//   GGML_CPU_NO_BARRIER_RUN    a run of consecutive row-local elementwise nodes shares
//                              one pool-wide barrier instead of taking one each
//
// HOW. It builds the Maple 20B-A1B decoder block node for node: the two RMS_NORMs whose
// ne01 is the token count, the Q norm ([128, 16, T]) and the K norm ([128, 4, T]) whose
// ne01 is the head count and which are the shapes the flattening exists for, the F32
// router, the soft max, the top-k argsort, the GET_ROWS / SUM_ROWS / CLAMP / DIV weight
// chain, three TQ2_0 MUL_MAT_IDs over repacked expert weights, the swiglu, the broadcast
// MUL that scales each expert's output, the EIGHT consecutive ADDs that reduce the expert
// slices (the run the barrier elision exists for) and the residual ADDs, then a Q4_K head.
// Attention itself is not built: FLASH_ATTN_EXT and the matmuls around it terminate every
// elision run, so they carry no case this test could lose. The real graph including flash
// attention, real graph-allocator aliasing and the real 24 layers is covered separately by
// hashing every tensor of the real model under each switch (llama-moe-tail-hash).
//
// Every node keeps its own buffer (ggml_backend_alloc_ctx_tensors, not a graph allocator),
// so EVERY node of the graph is still readable after the run and all of them are compared,
// not just the last one. The reference is the same binary with every switch off, which is
// upstream behaviour, so a difference can only be one of these four changes.
//
// Thread counts 1, 2, 3 and 4 are all run. The thread count is the variable the partition
// is about: at nth = 4 the K norm's ne01 = 4 divides exactly and the old partition is
// already balanced, at nth = 3 it does not. Token counts 1, 4, 64 and 129 cover the decode
// shape, a short batch, the profiled batch and a batch that divides no thread count.
//
// This test times nothing.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "moe-tail.h"

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

// Maple 20B-A1B geometry, with the expert count cut from 256 to 16 so the test's weights
// fit in a few tens of megabytes. Nothing the switches touch depends on the expert count:
// the elementwise block downstream of the experts has the same shapes either way.
static const int64_t N_EMBD   = 2048;
static const int64_t N_HEAD   = 16;
static const int64_t N_HEADKV = 4;
static const int64_t D_HEAD   = 128;
static const int64_t N_EXPERT = 16;
static const int64_t N_USED   = 8;
static const int64_t N_FF_EXP = 512;
static const int64_t N_VOCAB  = 4096;
static const int     N_LAYER  = 2;

struct weights {
    ggml_context *          ctx  = nullptr;
    ggml_backend_buffer_t   bufq = nullptr;  // repack buffer (quantised)
    ggml_backend_buffer_t   buff = nullptr;  // plain CPU buffer (f32 norms and router)
    ggml_tensor * attn_norm[N_LAYER];
    ggml_tensor * q_norm   [N_LAYER];
    ggml_tensor * k_norm   [N_LAYER];
    ggml_tensor * ffn_norm [N_LAYER];
    ggml_tensor * wq[N_LAYER], * wk[N_LAYER], * wv[N_LAYER], * wo[N_LAYER];
    ggml_tensor * router[N_LAYER];
    ggml_tensor * w_gate[N_LAYER], * w_up[N_LAYER], * w_down[N_LAYER];
    ggml_tensor * out_norm = nullptr;
    ggml_tensor * head     = nullptr;
};

static void fill_random_f32(std::vector<float> & v, std::mt19937 & rng, float s = 1.0f) {
    std::normal_distribution<float> d(0.0f, s);
    for (auto & x : v) {
        x = d(rng);
    }
}

static void set_quantised(ggml_tensor * t, ggml_type ty, std::mt19937 & rng) {
    const int64_t nrow = ggml_nrows(t);
    const int64_t k    = t->ne[0];
    std::vector<float> f((size_t) nrow * k);
    fill_random_f32(f, rng);
    std::vector<uint8_t> q(ggml_row_size(ty, k) * nrow);
    ggml_quantize_chunk(ty, f.data(), q.data(), 0, nrow, k, nullptr);
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

static void set_f32(ggml_tensor * t, std::mt19937 & rng, float s = 1.0f) {
    std::vector<float> f(ggml_nelements(t));
    fill_random_f32(f, rng, s);
    ggml_backend_tensor_set(t, f.data(), 0, f.size() * sizeof(float));
}

static bool build_weights(weights & w, ggml_backend_buffer_type_t buft_repack, std::mt19937 & rng) {
    ggml_init_params p = { ggml_tensor_overhead() * 256, nullptr, true };
    w.ctx = ggml_init(p);

    ggml_context * c = w.ctx;

    for (int il = 0; il < N_LAYER; il++) {
        w.wq[il]     = ggml_new_tensor_2d(c, GGML_TYPE_TQ2_0, N_EMBD, N_EMBD);
        w.wk[il]     = ggml_new_tensor_2d(c, GGML_TYPE_TQ2_0, N_EMBD, N_HEADKV*D_HEAD);
        w.wv[il]     = ggml_new_tensor_2d(c, GGML_TYPE_TQ2_0, N_EMBD, N_HEADKV*D_HEAD);
        w.wo[il]     = ggml_new_tensor_2d(c, GGML_TYPE_TQ2_0, N_EMBD, N_EMBD);
        w.w_gate[il] = ggml_new_tensor_3d(c, GGML_TYPE_TQ2_0, N_EMBD,   N_FF_EXP, N_EXPERT);
        w.w_up  [il] = ggml_new_tensor_3d(c, GGML_TYPE_TQ2_0, N_EMBD,   N_FF_EXP, N_EXPERT);
        w.w_down[il] = ggml_new_tensor_3d(c, GGML_TYPE_TQ2_0, N_FF_EXP, N_EMBD,   N_EXPERT);
    }
    w.head = ggml_new_tensor_2d(c, GGML_TYPE_Q4_K, N_EMBD, N_VOCAB);

    w.bufq = ggml_backend_alloc_ctx_tensors_from_buft(c, buft_repack);
    if (w.bufq == nullptr) {
        return false;
    }
    if (w.wq[0]->extra == nullptr || w.w_gate[0]->extra == nullptr || w.head->extra == nullptr) {
        fprintf(stderr, "moe-tail: no repack variant registered for TQ2_0 or Q4_K on this CPU\n");
        return false;
    }

    // f32 tensors in a second context so they do not go through the repack buffer
    ggml_init_params pf = { ggml_tensor_overhead() * 256, nullptr, true };
    ggml_context * cf = ggml_init(pf);
    for (int il = 0; il < N_LAYER; il++) {
        w.attn_norm[il] = ggml_new_tensor_1d(cf, GGML_TYPE_F32, N_EMBD);
        w.q_norm   [il] = ggml_new_tensor_1d(cf, GGML_TYPE_F32, D_HEAD);
        w.k_norm   [il] = ggml_new_tensor_1d(cf, GGML_TYPE_F32, D_HEAD);
        w.ffn_norm [il] = ggml_new_tensor_1d(cf, GGML_TYPE_F32, N_EMBD);
        w.router   [il] = ggml_new_tensor_2d(cf, GGML_TYPE_F32, N_EMBD, N_EXPERT);
    }
    w.out_norm = ggml_new_tensor_1d(cf, GGML_TYPE_F32, N_EMBD);
    w.buff = ggml_backend_alloc_ctx_tensors_from_buft(cf, ggml_backend_cpu_buffer_type());
    if (w.buff == nullptr) {
        return false;
    }

    for (int il = 0; il < N_LAYER; il++) {
        set_quantised(w.wq[il],     GGML_TYPE_TQ2_0, rng);
        set_quantised(w.wk[il],     GGML_TYPE_TQ2_0, rng);
        set_quantised(w.wv[il],     GGML_TYPE_TQ2_0, rng);
        set_quantised(w.wo[il],     GGML_TYPE_TQ2_0, rng);
        set_quantised(w.w_gate[il], GGML_TYPE_TQ2_0, rng);
        set_quantised(w.w_up  [il], GGML_TYPE_TQ2_0, rng);
        set_quantised(w.w_down[il], GGML_TYPE_TQ2_0, rng);
        set_f32(w.attn_norm[il], rng, 0.5f);
        set_f32(w.q_norm   [il], rng, 0.5f);
        set_f32(w.k_norm   [il], rng, 0.5f);
        set_f32(w.ffn_norm [il], rng, 0.5f);
        set_f32(w.router   [il], rng, 0.1f);
    }
    set_quantised(w.head, GGML_TYPE_Q4_K, rng);
    set_f32(w.out_norm, rng, 0.5f);

    // the f32 context is kept alive by its buffer; the tensors are reachable through w
    (void) cf;
    return true;
}

// One run: build the graph for n_tok tokens, compute it with nth threads, and return the
// bytes of every node in graph order.
static bool run_graph(const weights & w, int64_t n_tok, int nth,
                      const std::vector<float> & inp_data,
                      std::vector<std::vector<uint8_t>> & out,
                      std::vector<std::string> & names) {
    const size_t n_tensors = 4096;
    ggml_init_params p = { ggml_tensor_overhead()*n_tensors + ggml_graph_overhead_custom(n_tensors, false), nullptr, true };
    ggml_context * c = ggml_init(p);

    ggml_tensor * inp = ggml_new_tensor_2d(c, GGML_TYPE_F32, N_EMBD, n_tok);
    ggml_set_name(inp, "inp");

    ggml_tensor * cur = inp;

    for (int il = 0; il < N_LAYER; il++) {
        ggml_tensor * x = cur;

        ggml_tensor * n0 = ggml_mul(c, ggml_rms_norm(c, x, 1e-6f), w.attn_norm[il]);

        ggml_tensor * q = ggml_mul_mat(c, w.wq[il], n0);
        ggml_tensor * k = ggml_mul_mat(c, w.wk[il], n0);
        ggml_tensor * v = ggml_mul_mat(c, w.wv[il], n0);

        ggml_tensor * q3 = ggml_reshape_3d(c, q, D_HEAD, N_HEAD,   n_tok);
        ggml_tensor * k3 = ggml_reshape_3d(c, k, D_HEAD, N_HEADKV, n_tok);

        // the two norms the flattened partition exists for: ne01 = 16 and ne01 = 4
        ggml_tensor * qn = ggml_mul(c, ggml_rms_norm(c, q3, 1e-6f), w.q_norm[il]);
        ggml_tensor * kn = ggml_mul(c, ggml_rms_norm(c, k3, 1e-6f), w.k_norm[il]);

        // stand-in for attention: mix the normalised Q with V so both norms feed the
        // output, without building flash attention, which terminates every elision run
        ggml_tensor * mix = ggml_add(c, ggml_reshape_2d(c, qn, N_EMBD, n_tok),
                                        ggml_repeat(c, ggml_reshape_2d(c, kn, N_HEADKV*D_HEAD, n_tok), ggml_reshape_2d(c, qn, N_EMBD, n_tok)));
        (void) v;
        ggml_tensor * attn = ggml_mul_mat(c, w.wo[il], mix);

        ggml_tensor * ffn_inp = ggml_add(c, x, attn);

        ggml_tensor * n1 = ggml_mul(c, ggml_rms_norm(c, ffn_inp, 1e-6f), w.ffn_norm[il]);

        // router
        ggml_tensor * logits = ggml_mul_mat(c, w.router[il], n1);           // [N_EXPERT, n_tok]
        ggml_tensor * probs  = ggml_soft_max(c, logits);
        ggml_tensor * top    = ggml_argsort_top_k(c, probs, N_USED);         // ARGSORT + view, as the Maple graph does

        ggml_tensor * wts = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, N_EXPERT, n_tok), top);
        wts = ggml_reshape_2d(c, wts, N_USED, n_tok);
        ggml_tensor * wsum = ggml_sum_rows(c, wts);                          // [1, n_tok]
        wsum = ggml_clamp(c, wsum, 1e-20f, INFINITY);
        wts  = ggml_div(c, wts, wsum);
        wts  = ggml_reshape_3d(c, wts, 1, N_USED, n_tok);

        ggml_tensor * n1_3 = ggml_reshape_3d(c, n1, N_EMBD, 1, n_tok);
        ggml_tensor * gate = ggml_mul_mat_id(c, w.w_gate[il], n1_3, top);    // [N_FF_EXP, N_USED, n_tok]
        ggml_tensor * up   = ggml_mul_mat_id(c, w.w_up  [il], n1_3, top);
        ggml_tensor * act  = ggml_swiglu_split(c, gate, up);
        ggml_tensor * down = ggml_mul_mat_id(c, w.w_down[il], act, top);     // [N_EMBD, N_USED, n_tok]

        ggml_tensor * weighted = ggml_mul(c, down, wts);                     // the broadcast MUL

        // the eight consecutive ADDs that reduce the expert slices
        ggml_tensor * moe_out = ggml_view_2d(c, weighted, N_EMBD, n_tok, weighted->nb[2], 0);
        for (int64_t e = 1; e < N_USED; e++) {
            ggml_tensor * slice = ggml_view_2d(c, weighted, N_EMBD, n_tok, weighted->nb[2], e*weighted->nb[1]);
            moe_out = ggml_add(c, moe_out, slice);
        }

        cur = ggml_add(c, moe_out, ffn_inp);
        ggml_set_name(cur, (std::string("l_out-") + std::to_string(il)).c_str());
    }

    ggml_tensor * fin  = ggml_mul(c, ggml_rms_norm(c, cur, 1e-6f), w.out_norm);
    ggml_tensor * outl = ggml_mul_mat(c, w.head, fin);
    ggml_set_name(outl, "result_output");

    ggml_cgraph * gf = ggml_new_graph_custom(c, n_tensors, false);
    ggml_build_forward_expand(gf, outl);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, nth);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, backend);
    if (buf == nullptr) {
        fprintf(stderr, "moe-tail: failed to allocate the graph context\n");
        return false;
    }
    // Zero the whole buffer before every run. Some nodes are never written: the CPU
    // backend fuses RMS_NORM with the MUL after it and writes the MUL's destination, so
    // each fused RMS_NORM node's own buffer keeps whatever the allocator handed it.
    // Without this the comparison below reads uninitialised memory for those nodes and
    // reports differences that are the allocator's, not the engine's.
    ggml_backend_buffer_clear(buf, 0);
    ggml_backend_tensor_set(inp, inp_data.data(), 0, inp_data.size()*sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "moe-tail: graph compute failed\n");
        return false;
    }

    out.clear();
    names.clear();
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        const size_t n = ggml_nbytes(t);
        std::vector<uint8_t> b(n);
        ggml_backend_tensor_get(t, b.data(), 0, n);
        out.push_back(std::move(b));
        names.push_back(std::string(ggml_op_name(t->op)) + " " + t->name);
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(c);
    return true;
}

struct switches {
    const char * name;
    int no_rows_flatten;
    int no_bcast_scalar;
    int no_sumrows_par;
    int no_barrier_run;
};

static void apply(const switches & s) {
    setenv("GGML_CPU_NO_ROWS_FLATTEN", s.no_rows_flatten ? "1" : "0", 1);
    setenv("GGML_CPU_NO_BCAST_SCALAR", s.no_bcast_scalar ? "1" : "0", 1);
    setenv("GGML_CPU_NO_SUMROWS_PAR",  s.no_sumrows_par  ? "1" : "0", 1);
    setenv("GGML_CPU_NO_BARRIER_RUN",  s.no_barrier_run  ? "1" : "0", 1);
    ggml_tail_init();
}

int main(void) {
    ggml_backend_load_all();

    ggml_backend_buffer_type_t buft_repack = find_repack_buft();
    if (buft_repack == nullptr) {
        fprintf(stderr, "moe-tail: no CPU_REPACK buffer type, skipping\n");
        return 0;
    }

    std::mt19937 rng(1234567);
    weights w;
    if (!build_weights(w, buft_repack, rng)) {
        fprintf(stderr, "moe-tail: could not build weights\n");
        return 1;
    }

    const switches cfgs[] = {
        { "reference (all off, upstream behaviour)", 1, 1, 1, 1 },
        { "rows_flatten only",                       0, 1, 1, 1 },
        { "bcast_scalar only",                       1, 0, 1, 1 },
        { "sumrows_par only",                        1, 1, 0, 1 },
        { "barrier_run only",                        1, 1, 1, 0 },
        { "all on (shipped default)",                0, 0, 0, 0 },
    };

    const int64_t toks[] = { 1, 4, 64, 129 };
    const int     nths[] = { 1, 2, 3, 4 };

    int n_fail = 0;
    int n_cmp  = 0;

    for (int64_t n_tok : toks) {
        std::vector<float> inp((size_t) N_EMBD * n_tok);
        std::mt19937 irng(99 + (unsigned) n_tok);
        fill_random_f32(inp, irng);

        for (int nth : nths) {
            std::vector<std::vector<uint8_t>> ref;
            std::vector<std::string> ref_names;

            apply(cfgs[0]);
            if (!run_graph(w, n_tok, nth, inp, ref, ref_names)) {
                return 1;
            }

            for (size_t ci = 1; ci < sizeof(cfgs)/sizeof(cfgs[0]); ci++) {
                std::vector<std::vector<uint8_t>> got;
                std::vector<std::string> got_names;

                apply(cfgs[ci]);
                if (!run_graph(w, n_tok, nth, inp, got, got_names)) {
                    return 1;
                }

                bool ok = got.size() == ref.size();
                std::string first_bad;
                size_t n_bad_nodes = 0;
                for (size_t i = 0; ok && i < ref.size(); i++) {
                    if (got[i].size() != ref[i].size() ||
                        memcmp(got[i].data(), ref[i].data(), ref[i].size()) != 0) {
                        n_bad_nodes++;
                        if (first_bad.empty()) {
                            first_bad = "node " + std::to_string(i) + " " + ref_names[i];
                        }
                    }
                }
                if (n_bad_nodes) {
                    ok = false;
                }

                n_cmp++;
                if (ok) {
                    printf("T=%4lld nth=%d  %-40s  IDENTICAL (%zu nodes)\n",
                           (long long) n_tok, nth, cfgs[ci].name, ref.size());
                } else {
                    printf("T=%4lld nth=%d  %-40s  DIFFERS %zu of %zu nodes, first %s\n",
                           (long long) n_tok, nth, cfgs[ci].name, n_bad_nodes, ref.size(), first_bad.c_str());
                    n_fail++;
                }
                fflush(stdout);
            }
        }
    }

    ggml_backend_buffer_free(w.bufq);
    ggml_backend_buffer_free(w.buff);
    ggml_free(w.ctx);

    printf("\ntest-moe-tail-identity: %d comparisons, %d failed\n", n_cmp, n_fail);
    if (n_fail) {
        printf("test-moe-tail-identity: FAIL\n");
        return 1;
    }
    printf("test-moe-tail-identity: PASS\n");
    return 0;
}
