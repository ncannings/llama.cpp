// Bit-identity gate for the mm-invoke per-invocation changes inside the repack
// forward_mul_mat.
//
// WHAT IT TESTS. Four changes alter WHICH THREAD does a piece of work, or WHERE a scratch
// byte lives, and none of them may alter a single output byte:
//
//   GGML_CPU_NO_MM_PARQUANT  at ne11 == 1 the single src1 row is quantised by thread 0
//                            alone; the change splits its WHOLE BLOCKS across the pool
//   GGML_CPU_NO_MM_STATIC1   at ne11 == 1 src0 rows are handed out through the threadpool's
//                            shared atomic chunk counter; the change gives each thread a
//                            fixed contiguous NB_COLS-aligned range and drops the counter
//   GGML_CPU_NO_MM_DISPATCH  the extra-buffer dispatch walks a function-local static
//                            std::vector; the change walks a flat array filled once
//   GGML_CPU_WBUF_SLOTS=<n>  the cplan work buffer is one region shared by every node; the
//                            change gives consecutive nodes different regions
//
// HOW. It builds MUL_MAT nodes over repacked weights directly, which is where all four
// changes live, and compares every output byte against the same binary with every switch
// off (upstream behaviour). The shapes are chosen to exercise the boundaries the changes
// introduce rather than the common case:
//
//   ne00 = 256    ONE q8_K block, so the block split has nothing to split and must fall
//                 back to the row path
//   ne00 = 768    three blocks over four threads: two threads get one, two get none
//   ne01 = 24     three NB_COLS groups over four threads, so some threads get an empty
//                 static range and must still reach the internal barrier
//   ne01 = 1544   not a multiple of the thread count, so the static split's ceil boundary
//                 and the chunk loop's must agree on the last partial group
//   ne11 = 1      the only shape the static split and the block quantisation apply to
//   ne11 = 2,3,4,5,64,129  every shape they must LEAVE ALONE, including the gemm path
//
// Thread counts 1, 2, 3, 4 and 10 are run. Ten threads on four cores is oversubscribed on
// purpose: it is the setting in which a thread is most likely to be descheduled between
// publishing its quantised blocks and the barrier that orders them.
//
// Every node keeps its own buffer, so every node is compared, not just the last.
//
// This test times nothing.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "moe-tail.h"

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

static void fill_random_f32(std::vector<float> & v, std::mt19937 & rng, float s = 1.0f) {
    std::normal_distribution<float> d(0.0f, s);
    for (auto & x : v) {
        x = d(rng);
    }
}

struct shape {
    int64_t     ne00;
    int64_t     ne01;
    ggml_type   type;
    const char * why;
};

static const shape shapes[] = {
    { 1536, 1536, GGML_TYPE_TQ2_0, "synth attn_q"          },
    { 1536, 4096, GGML_TYPE_TQ2_0, "synth ffn_gate"        },
    { 1536,  384, GGML_TYPE_TQ2_0, "synth attn_k"          },
    { 2560, 2560, GGML_TYPE_TQ2_0, "bitnet attn"           },
    { 2560, 6912, GGML_TYPE_TQ2_0, "bitnet ffn"            },
    {  256, 2048, GGML_TYPE_TQ2_0, "one q8_K block"        },
    {  768, 1544, GGML_TYPE_TQ2_0, "3 blocks, ragged rows" },
    { 1536,   24, GGML_TYPE_TQ2_0, "3 NB_COLS groups"      },
    { 1024, 1024, GGML_TYPE_Q4_0,  "q4_0 repack"           },
    {  512,  520, GGML_TYPE_Q4_0,  "q4_0 ragged rows"      },
};

struct switches {
    const char * name;
    int no_parquant;
    int no_static1;
    int no_dispatch;
    int slots;
    int static_max_mb;   // 0 = no threshold, the ungated form
};

static void apply(const switches & s) {
    setenv("GGML_CPU_NO_MM_PARQUANT", s.no_parquant ? "1" : "0", 1);
    setenv("GGML_CPU_NO_MM_STATIC1",  s.no_static1  ? "1" : "0", 1);
    setenv("GGML_CPU_NO_MM_DISPATCH", s.no_dispatch ? "1" : "0", 1);
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", s.slots);
    setenv("GGML_CPU_WBUF_SLOTS", buf, 1);
    snprintf(buf, sizeof(buf), "%d", s.static_max_mb);
    setenv("GGML_CPU_MM_STATIC_MAX_MB", buf, 1);
    ggml_tail_init();
}

// The weights are built once: repacking ten tensors is the expensive part and it does not
// depend on any switch.
struct weights {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> w;
};

static bool build_weights(weights & W, ggml_backend_buffer_type_t buft_repack) {
    const int n_shape = (int) (sizeof(shapes)/sizeof(shapes[0]));

    ggml_init_params ip = { ggml_tensor_overhead() * 256, nullptr, true };
    W.ctx = ggml_init(ip);
    if (W.ctx == nullptr) {
        return false;
    }
    W.w.resize(n_shape);
    for (int i = 0; i < n_shape; i++) {
        W.w[i] = ggml_new_tensor_2d(W.ctx, shapes[i].type, shapes[i].ne00, shapes[i].ne01);
        ggml_set_name(W.w[i], shapes[i].why);
    }
    W.buf = ggml_backend_alloc_ctx_tensors_from_buft(W.ctx, buft_repack);
    if (W.buf == nullptr) {
        return false;
    }
    std::mt19937 rng(20260916);
    for (int i = 0; i < n_shape; i++) {
        const int64_t nrow = ggml_nrows(W.w[i]);
        const int64_t k    = W.w[i]->ne[0];
        std::vector<float> f((size_t) nrow * k);
        fill_random_f32(f, rng);
        std::vector<uint8_t> q(ggml_row_size(shapes[i].type, k) * nrow);
        ggml_quantize_chunk(shapes[i].type, f.data(), q.data(), 0, nrow, k, nullptr);
        ggml_backend_tensor_set(W.w[i], q.data(), 0, q.size());
    }
    return true;
}

// One graph: every shape, with the matmuls that share an ne00 sharing one activation
// tensor, so consecutive matmuls are exactly the adjacent same-src1 pairs the work-buffer
// regions are about.
static bool run_graph(const weights & W,
                      int64_t n_tok, int nth,
                      std::vector<std::vector<uint8_t>> & out,
                      std::vector<std::string> & names) {
    const int n_shape = (int) (sizeof(shapes)/sizeof(shapes[0]));

    ggml_init_params ip = { ggml_tensor_overhead() * 4096 + ggml_graph_overhead(), nullptr, true };
    ggml_context * c = ggml_init(ip);
    if (c == nullptr) {
        return false;
    }

    std::vector<ggml_tensor *> a(n_shape, nullptr);
    for (int i = 0; i < n_shape; i++) {
        for (int j = 0; j < i; j++) {
            if (shapes[j].ne00 == shapes[i].ne00) { a[i] = a[j]; break; }
        }
        if (a[i] == nullptr) {
            a[i] = ggml_new_tensor_2d(c, GGML_TYPE_F32, shapes[i].ne00, n_tok);
        }
    }

    ggml_cgraph * gf = ggml_new_graph(c);
    std::vector<ggml_tensor *> res(n_shape);
    for (int i = 0; i < n_shape; i++) {
        res[i] = ggml_mul_mat(c, W.w[i], a[i]);
        ggml_set_name(res[i], shapes[i].why);
        ggml_build_forward_expand(gf, res[i]);
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, backend);
    if (buf == nullptr) {
        ggml_backend_free(backend);
        ggml_free(c);
        return false;
    }

    for (int i = 0; i < n_shape; i++) {
        bool done = false;
        for (int j = 0; j < i; j++) {
            if (a[j] == a[i]) { done = true; break; }
        }
        if (done) {
            continue;
        }
        std::vector<float> f(ggml_nelements(a[i]));
        std::mt19937 arng(777u + (unsigned) a[i]->ne[0] + 13u * (unsigned) n_tok);
        fill_random_f32(f, arng);
        ggml_backend_tensor_set(a[i], f.data(), 0, f.size() * sizeof(float));
    }

    // POISON every destination before the run. Without this a sabotage that leaves output
    // rows UNWRITTEN can pass: the reference run frees its buffer and the treatment run is
    // handed the same allocation back, so the unwritten rows still hold the reference's own
    // bytes and compare equal. The m3 row-hole sabotage did exactly that before this was
    // added, which is the reason it is here.
    for (int i = 0; i < n_shape; i++) {
        std::vector<uint8_t> poison(ggml_nbytes(res[i]), 0xA5);
        ggml_backend_tensor_set(res[i], poison.data(), 0, poison.size());
    }

    ggml_backend_cpu_set_n_threads(backend, nth);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(c);
        return false;
    }

    out.clear();
    names.clear();
    for (int i = 0; i < n_shape; i++) {
        std::vector<uint8_t> b(ggml_nbytes(res[i]));
        ggml_backend_tensor_get(res[i], b.data(), 0, b.size());
        out.push_back(std::move(b));
        names.push_back(std::string(shapes[i].why));
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(c);
    return true;
}

int main(void) {
    ggml_backend_load_all();

    ggml_backend_buffer_type_t buft_repack = find_repack_buft();
    if (buft_repack == nullptr) {
        fprintf(stderr, "mm-invoke: no CPU_REPACK buffer type, skipping\n");
        return 0;
    }

    weights W;
    if (!build_weights(W, buft_repack)) {
        fprintf(stderr, "mm-invoke: could not build weights\n");
        return 1;
    }

    // The weight working set of this test's ten shapes is about 9.8 MB, so the 16 MB
    // residency threshold ADMITS the static split here and "static1 only" exercises it. The
    // two extra rows walk the threshold off its default in both directions: at 1 MB the gate
    // refuses and the chunk path runs, at 0 the threshold is removed entirely. Both must be
    // identical to the reference, because both sides of the gate are the same arithmetic.
    const switches cfgs[] = {
        { "reference (all off, upstream behaviour)", 1, 1, 1, 1, 16 },
        { "parquant only",                           0, 1, 1, 1, 16 },
        { "static1 only",                            1, 0, 1, 1, 16 },
        { "static1, threshold 1 MB (gate refuses)",  1, 0, 1, 1,  1 },
        { "static1, threshold removed",              1, 0, 1, 1,  0 },
        { "dispatch only",                           1, 1, 0, 1, 16 },
        { "wbuf_slots=3 only",                       1, 1, 1, 3, 16 },
        { "wbuf_slots=2 only",                       1, 1, 1, 2, 16 },
        { "all on (shipped default)",                0, 0, 0, 3, 16 },
    };

    const int64_t toks[] = { 1, 2, 3, 4, 5, 64, 129 };
    const int     nths[] = { 1, 2, 3, 4, 10 };

    int n_fail = 0;
    int n_cmp  = 0;

    for (int64_t n_tok : toks) {
        for (int nth : nths) {
            std::vector<std::vector<uint8_t>> ref;
            std::vector<std::string> ref_names;

            apply(cfgs[0]);
            if (!run_graph(W, n_tok, nth, ref, ref_names)) {
                fprintf(stderr, "mm-invoke: reference run failed at T=%lld nth=%d\n", (long long) n_tok, nth);
                return 1;
            }

            for (size_t ci = 1; ci < sizeof(cfgs)/sizeof(cfgs[0]); ci++) {
                std::vector<std::vector<uint8_t>> got;
                std::vector<std::string> got_names;

                apply(cfgs[ci]);
                if (!run_graph(W, n_tok, nth, got, got_names)) {
                    fprintf(stderr, "mm-invoke: run failed at T=%lld nth=%d %s\n", (long long) n_tok, nth, cfgs[ci].name);
                    return 1;
                }

                size_t n_bad = 0;
                std::string first_bad;
                bool ok = got.size() == ref.size();
                for (size_t i = 0; ok && i < ref.size(); i++) {
                    if (got[i].size() != ref[i].size() ||
                        memcmp(got[i].data(), ref[i].data(), ref[i].size()) != 0) {
                        n_bad++;
                        if (first_bad.empty()) {
                            first_bad = "node " + std::to_string(i) + " " + ref_names[i];
                        }
                    }
                }
                if (n_bad) {
                    ok = false;
                }

                n_cmp++;
                if (ok) {
                    printf("T=%4lld nth=%2d  %-40s  IDENTICAL (%zu nodes)\n",
                           (long long) n_tok, nth, cfgs[ci].name, ref.size());
                } else {
                    printf("T=%4lld nth=%2d  %-40s  DIFFERS %zu of %zu nodes, first %s\n",
                           (long long) n_tok, nth, cfgs[ci].name, n_bad, ref.size(), first_bad.c_str());
                    n_fail++;
                }
            }
        }
    }

    printf("\ntest-mm-invoke-identity: %d comparisons, %d failed\n", n_cmp, n_fail);
    printf("test-mm-invoke-identity: %s\n", n_fail == 0 ? "PASS" : "FAIL");
    return n_fail == 0 ? 0 : 1;
}
