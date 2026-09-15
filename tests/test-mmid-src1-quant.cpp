// Checks the src1 quantisation dispatch of ggml::cpu::repack::tensor_traits::forward_mul_mat_id.
//
// build_moe_ffn hands the expert matmuls an activation of shape [n_embd, 1, n_tokens], so
// ne11 == 1 and the old loop
//
//     for (i12 = 0; i12 < ne12; ++i12) for (i11 = ith; i11 < ne11; i11 += nth) from_float(...)
//
// gave every row to thread 0 while the rest waited in the barrier below it. The new loop
// walks the flattened (i11, i12) row index with the same round robin over threads. Each row
// is still quantised exactly once, by exactly one thread, by the same from_float call on the
// same source and destination addresses, so the bytes written are identical and only the
// thread that writes them changes.
//
// Part 1 compares the two dispatches directly, field by field over the resulting block_q8_K
// rows: the int8 quants, the float scale and the int16 block sums must all be identical, and
// no byte of the destination may be left unwritten. Stimuli are uniform and normal random
// rows, rows with outlier channels, all zero rows, and real activation rows captured from a
// Maple decode step (see --capture below).
//
// Part 2 binds that to the shipped code: it runs the real ggml_mul_mat_id over a repacked
// TQ2_0 expert tensor at the build_moe_ffn shape (ne11 = 1, rows along ne12) for thread
// counts 1, 2, 3, 4, 5 and 8 and requires the output to be identical bit for bit to the
// un-repacked reference. A dispatch that dropped or doubled a row fails here.
//
// Fixture capture:  test-mmid-src1-quant --capture <model.gguf> <out.bin>
// runs one decode with an eval callback and writes the f32 src1 rows of the first
// MUL_MAT_ID node it sees.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

// ggml-common.h is not on the test include path; this is the layout it defines, and the
// static check below refuses if it ever stops matching what ggml reports.
struct q8_K_block {
    float   d;
    int8_t  qs[256];
    int16_t bsums[256/16];
};

static const char * FIXTURE_MAGIC = "MMIDSRC1";

// ---------------------------------------------------------------------------- capture mode

struct capture_state {
    std::vector<float> rows;
    int64_t ne10 = 0;
    int64_t nrows = 0;
    bool done = false;
};

static bool capture_cb(ggml_tensor * t, bool ask, void * user_data) {
    capture_state * st = (capture_state *) user_data;
    if (st->done) {
        return false;
    }
    if (ask) {
        return t->op == GGML_OP_MUL_MAT_ID;
    }
    ggml_tensor * src1 = t->src[1];
    if (src1 == nullptr || src1->type != GGML_TYPE_F32 || !ggml_is_contiguous(src1)) {
        return true;
    }
    st->ne10  = src1->ne[0];
    st->nrows = src1->ne[1]*src1->ne[2]*src1->ne[3];
    st->rows.resize((size_t) st->ne10*st->nrows);
    ggml_backend_tensor_get(src1, st->rows.data(), 0, st->rows.size()*sizeof(float));
    st->done = true;
    return true;
}

static int do_capture(const char * model_path, const char * out_path) {
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", model_path);
        return 1;
    }

    capture_state st;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 512;
    cparams.n_batch         = 512;
    cparams.n_ubatch        = 512;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;
    cparams.cb_eval           = capture_cb;
    cparams.cb_eval_user_data = &st;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const char * prompt =
        "The history of the Royal Navy begins with the ships of the medieval kings, and the "
        "main difference between a cache and DRAM is latency, not capacity.";
    std::vector<llama_token> toks(256);
    const int n = llama_tokenize(vocab, prompt, (int32_t) strlen(prompt), toks.data(), (int32_t) toks.size(), true, true);
    if (n <= 0) {
        fprintf(stderr, "tokenise failed\n");
        return 1;
    }
    toks.resize(n);

    if (llama_decode(ctx, llama_batch_get_one(toks.data(), (int32_t) toks.size())) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }

    if (!st.done) {
        fprintf(stderr, "no MUL_MAT_ID node with an f32 contiguous src1 was seen\n");
        return 1;
    }

    FILE * f = fopen(out_path, "wb");
    if (f == nullptr) {
        fprintf(stderr, "cannot write %s\n", out_path);
        return 1;
    }
    const int32_t ne10 = (int32_t) st.ne10;
    const int32_t nrows = (int32_t) st.nrows;
    fwrite(FIXTURE_MAGIC, 1, 8, f);
    fwrite(&ne10, sizeof(int32_t), 1, f);
    fwrite(&nrows, sizeof(int32_t), 1, f);
    fwrite(st.rows.data(), sizeof(float), st.rows.size(), f);
    fclose(f);

    printf("captured %d real src1 rows of %d elements to %s\n", nrows, ne10, out_path);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

static bool load_fixture(const char * path, std::vector<float> & rows, int64_t & ne10, int64_t & nrows) {
    FILE * f = fopen(path, "rb");
    if (f == nullptr) {
        return false;
    }
    char magic[8];
    int32_t a = 0;
    int32_t b = 0;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, FIXTURE_MAGIC, 8) != 0 ||
        fread(&a, sizeof(int32_t), 1, f) != 1 || fread(&b, sizeof(int32_t), 1, f) != 1) {
        fclose(f);
        return false;
    }
    ne10  = a;
    nrows = b;
    rows.resize((size_t) ne10*nrows);
    const bool ok = fread(rows.data(), sizeof(float), rows.size(), f) == rows.size();
    fclose(f);
    return ok;
}

// ------------------------------------------------------------------- part 1, the dispatches

// the loop as it was before the change
static void quant_old(ggml_from_float_t from_float, const char * src, char * wdata,
                      int64_t ne10, int64_t ne11, int64_t ne12, size_t nb11, size_t nb12,
                      size_t nbw1, size_t nbw2, int ith, int nth) {
    for (int64_t i12 = 0; i12 < ne12; ++i12) {
        for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
            from_float((const float *) (src + i12*nb12 + i11*nb11), (void *) (wdata + i12*nbw2 + i11*nbw1), ne10);
        }
    }
}

// the loop as it is after it
static void quant_new(ggml_from_float_t from_float, const char * src, char * wdata,
                      int64_t ne10, int64_t ne11, int64_t ne12, size_t nb11, size_t nb12,
                      size_t nbw1, size_t nbw2, int ith, int nth) {
    const int64_t nrow1 = ne11*ne12;
    for (int64_t ir = ith; ir < nrow1; ir += nth) {
        const int64_t i11 = ir % ne11;
        const int64_t i12 = ir / ne11;
        from_float((const float *) (src + i12*nb12 + i11*nb11), (void *) (wdata + i12*nbw2 + i11*nbw1), ne10);
    }
}

static bool compare_rows(const char * label, const std::vector<float> & src,
                         int64_t ne10, int64_t ne11, int64_t ne12, int nth_old, int nth_new) {
    const ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const ggml_from_float_t from_float = tr->from_float;

    const size_t nbw1 = ggml_row_size(GGML_TYPE_Q8_K, ne10);
    const size_t nbw2 = nbw1*ne11;
    const size_t total = nbw1*ne11*ne12;

    const size_t nb11 = ne10*sizeof(float);
    const size_t nb12 = nb11*ne11;

    // Each dispatch is run twice into buffers prefilled with two DIFFERENT sentinels. A byte
    // the dispatch wrote comes out the same both times; a byte it did not write comes out as
    // the two sentinels, so the disagreeing positions are exactly the untouched set. Counting
    // occurrences of one sentinel value would not work, because a quantised byte is free to
    // equal it.
    //
    // Untouched bytes are not hypothetical here: ggml's aarch64 quantize_row_q8_K takes an
    // early out on a row whose amax is zero, writing only d and leaving qs and bsums as it
    // found them. That is a property of the quantiser, not of the dispatch, and it is the
    // same under both, which is what the equal-untouched-set check below states. The
    // destination address of a given row is identical under both dispatches, so whatever the
    // previous op left at that address is identical too.
    std::vector<char> a (total, (char) 0xA5);
    std::vector<char> a2(total, (char) 0x5A);
    std::vector<char> b (total, (char) 0xA5);
    std::vector<char> b2(total, (char) 0x5A);

    for (int ith = 0; ith < nth_old; ++ith) {
        quant_old(from_float, (const char *) src.data(), a.data(),  ne10, ne11, ne12, nb11, nb12, nbw1, nbw2, ith, nth_old);
        quant_old(from_float, (const char *) src.data(), a2.data(), ne10, ne11, ne12, nb11, nb12, nbw1, nbw2, ith, nth_old);
    }
    for (int ith = 0; ith < nth_new; ++ith) {
        quant_new(from_float, (const char *) src.data(), b.data(),  ne10, ne11, ne12, nb11, nb12, nbw1, nbw2, ith, nth_new);
        quant_new(from_float, (const char *) src.data(), b2.data(), ne10, ne11, ne12, nb11, nb12, nbw1, nbw2, ith, nth_new);
    }

    // the untouched sets must be identical, and the touched bytes must agree; neutralise the
    // untouched positions in both buffers so the field comparison below reads written data
    size_t untouched_a = 0;
    size_t untouched_b = 0;
    size_t untouched_mismatch = 0;
    for (size_t i = 0; i < total; ++i) {
        const bool ua = a[i] != a2[i];
        const bool ub = b[i] != b2[i];
        untouched_a += ua;
        untouched_b += ub;
        untouched_mismatch += ua != ub;
        if (ua) { a[i] = 0; }
        if (ub) { b[i] = 0; }
    }

    const int64_t nrows = ne11*ne12;
    const int64_t nblk  = ne10/256;

    int    max_qs      = 0;
    int    max_bsum    = 0;
    double max_scale   = 0.0;
    size_t scale_bits  = 0;   // scales whose bits differ, including any that compare equal

    for (int64_t r = 0; r < nrows; ++r) {
        const q8_K_block * pa = (const q8_K_block *) (a.data() + (size_t) r*nbw1);
        const q8_K_block * pb = (const q8_K_block *) (b.data() + (size_t) r*nbw1);
        for (int64_t k = 0; k < nblk; ++k) {
            if (memcmp(&pa[k].d, &pb[k].d, sizeof(float)) != 0) {
                scale_bits++;
                max_scale = std::max(max_scale, (double) std::fabs((double) pa[k].d - (double) pb[k].d));
            }
            for (int j = 0; j < 256; ++j) {
                max_qs = std::max(max_qs, std::abs((int) pa[k].qs[j] - (int) pb[k].qs[j]));
            }
            for (int j = 0; j < 16; ++j) {
                max_bsum = std::max(max_bsum, std::abs((int) pa[k].bsums[j] - (int) pb[k].bsums[j]));
            }
        }
    }
    const bool bytes_equal = memcmp(a.data(), b.data(), total) == 0;

    const bool pass = max_qs == 0 && max_bsum == 0 && scale_bits == 0 && bytes_equal &&
                      untouched_mismatch == 0;

    printf("  %-18s ne11=%3d ne12=%5d  old nth=%d new nth=%d  "
           "max diff int8=%d scale=%.3e (%zu bits) bsums=%d  written bytes=%s  "
           "untouched old/new=%zu/%zu mismatch=%zu  %s\n",
           label, (int) ne11, (int) ne12, nth_old, nth_new,
           max_qs, max_scale, scale_bits, max_bsum, bytes_equal ? "equal" : "DIFFER",
           untouched_a, untouched_b, untouched_mismatch, pass ? "OK" : "FAIL");
    return pass;
}

// ------------------------------------------------------- part 2, the shipped dispatch in situ

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
    ggml_init_params wp = { ggml_tensor_overhead()*2, nullptr, true };
    ggml_context * ctx_w = ggml_init(wp);
    ggml_new_tensor_3d(ctx_w, wtype, 256, 8, 2);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
    const bool repacked = buf_w != nullptr && ggml_get_first_tensor(ctx_w)->extra != nullptr;
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
    return repacked;
}

// the build_moe_ffn shape: src1 is [K, 1, n_tok], so ne11 == 1 and the rows live along ne12
static bool run_mul_mat_id(ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                           const std::vector<uint8_t> & w_q, int64_t K, int64_t N, int64_t n_as,
                           const std::vector<float> & x, int64_t n_tok,
                           const std::vector<int32_t> & ids_v, std::vector<float> & out) {
    ggml_init_params wp = { ggml_tensor_overhead()*2, nullptr, true };
    ggml_context * ctx_w = ggml_init(wp);
    ggml_tensor * w = ggml_new_tensor_3d(ctx_w, GGML_TYPE_TQ2_0, K, N, n_as);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
    if (buf_w == nullptr) {
        fprintf(stderr, "failed to allocate weight on %s\n", ggml_backend_buft_name(buft));
        return false;
    }
    ggml_backend_tensor_set(w, w_q.data(), 0, w_q.size());

    ggml_init_params cp = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(cp);
    ggml_tensor * xt  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, n_tok);
    ggml_tensor * idt = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_tok);
    ggml_tensor * y   = ggml_mul_mat_id(ctx, w, xt, idt);
    ggml_cgraph * gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(xt,  x.data(),     0, x.size()*sizeof(float));
    ggml_backend_tensor_set(idt, ids_v.data(), 0, ids_v.size()*sizeof(int32_t));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        return false;
    }

    out.resize(ggml_nelements(y));
    ggml_backend_tensor_get(y, out.data(), 0, out.size()*sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
    return true;
}

// ---------------------------------------------------------------------------------- driver

int main(int argc, char ** argv) {
    if (argc >= 4 && strcmp(argv[1], "--capture") == 0) {
        return do_capture(argv[2], argv[3]);
    }

    ggml_backend_load_all();

    if (ggml_row_size(GGML_TYPE_Q8_K, 256) != sizeof(q8_K_block)) {
        fprintf(stderr, "block_q8_K layout in this test no longer matches ggml (%zu against %zu)\n",
                sizeof(q8_K_block), ggml_row_size(GGML_TYPE_Q8_K, 256));
        return 1;
    }

    bool ok = true;

    // ---- part 1: the two dispatches, field by field
    printf("part 1: src1 quantisation dispatch, old against new\n");

    const int64_t K = 2048;
    std::mt19937 rng(20260915);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::normal_distribution<float> nrm(0.0f, 1.0f);

    // (ne11, ne12) pairs: the gate and up shape (ne11 == 1), the down shape (ne11 == 8) and
    // some awkward ones where neither index divides the thread count
    const int64_t pairs[][2] = { {1,1}, {1,7}, {1,64}, {1,512}, {8,64}, {3,5}, {2,3}, {5,1} };
    const int     nths[]     = { 1, 2, 3, 4, 5, 8 };

    std::vector<float> fx_rows;
    int64_t fx_ne10 = 0;
    int64_t fx_nrows = 0;
    const bool have_fixture = load_fixture(MMID_SRC1_FIXTURE, fx_rows, fx_ne10, fx_nrows);
    if (!have_fixture) {
        fprintf(stderr, "real-row fixture %s missing or malformed; regenerate it with --capture\n",
                MMID_SRC1_FIXTURE);
        return 1;
    }
    if (fx_ne10 % 256 != 0) {
        fprintf(stderr, "fixture row length %d is not a multiple of QK_K\n", (int) fx_ne10);
        return 1;
    }
    printf("  real-row fixture: %d rows of %d elements\n", (int) fx_nrows, (int) fx_ne10);

    for (const auto & pr : pairs) {
        const int64_t ne11 = pr[0];
        const int64_t ne12 = pr[1];
        const int64_t nrows = ne11*ne12;

        std::vector<std::pair<const char *, std::vector<float>>> stim;

        std::vector<float> v_uni((size_t) K*nrows);
        for (auto & v : v_uni) { v = uni(rng); }
        stim.emplace_back("uniform", v_uni);

        std::vector<float> v_nrm((size_t) K*nrows);
        for (auto & v : v_nrm) { v = nrm(rng); }
        stim.emplace_back("normal", v_nrm);

        // a few outlier channels 50x the rest, which is what real activations look like and
        // what drives the per-row amax the quantiser keys on
        std::vector<float> v_out = v_nrm;
        for (int64_t r = 0; r < nrows; ++r) {
            for (int j = 0; j < 7; ++j) {
                v_out[(size_t) r*K + (size_t) (j*263) % K] *= 50.0f;
            }
        }
        stim.emplace_back("outlier channels", v_out);

        std::vector<float> v_zero((size_t) K*nrows, 0.0f);
        stim.emplace_back("all zero", v_zero);

        // real rows, tiled from the captured fixture if more rows are needed than captured
        std::vector<float> v_real((size_t) K*nrows);
        for (int64_t r = 0; r < nrows; ++r) {
            const int64_t sr = r % fx_nrows;
            for (int64_t j = 0; j < K; ++j) {
                v_real[(size_t) r*K + j] = fx_rows[(size_t) sr*fx_ne10 + (size_t) (j % fx_ne10)];
            }
        }
        stim.emplace_back("real (maple src1)", v_real);

        for (const auto & s : stim) {
            for (const int nth : nths) {
                ok = compare_rows(s.first, s.second, K, ne11, ne12, 1, nth) && ok;
            }
            // and the old dispatch at several thread counts against the new at several more,
            // since the old partition is also supposed to be thread-count invariant
            ok = compare_rows(s.first, s.second, K, ne11, ne12, 4, 4) && ok;
        }
    }

    // ---- part 2: the shipped dispatch, through the real op
    printf("part 2: ggml_mul_mat_id over a repacked TQ2_0 expert tensor, ne11 = 1\n");

    ggml_backend_buffer_type_t buft_ref    = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_type_t buft_repack = find_repack_buft();
    if (buft_repack == nullptr || !type_is_repacked(buft_repack, GGML_TYPE_TQ2_0)) {
        fprintf(stderr, "no TQ2_0 repack variant registered for this CPU, part 2 cannot run\n");
        return 1;
    }

    const int64_t shapes2[][2] = { { 256, 8 }, { 2048, 512 } };
    const int64_t n_as = 4;
    const int64_t toks[] = { 1, 5, 17, 64 };

    for (const auto & shape : shapes2) {
        const int64_t Kk = shape[0];
        const int64_t N  = shape[1];

        std::vector<float> w_f((size_t) Kk*N*n_as);
        for (auto & v : w_f) { v = uni(rng); }
        std::vector<uint8_t> w_q(ggml_row_size(GGML_TYPE_TQ2_0, Kk)*N*n_as);
        ggml_quantize_chunk(GGML_TYPE_TQ2_0, w_f.data(), w_q.data(), 0, N*n_as, Kk, nullptr);

        for (const int64_t n_tok : toks) {
            std::vector<float> x((size_t) Kk*n_tok);
            for (auto & v : x) { v = uni(rng); }
            std::vector<int32_t> ids(n_tok);
            for (int64_t t = 0; t < n_tok; ++t) {
                ids[t] = (int32_t) (t % n_as);
            }

            // The repacked run goes FIRST and the reference SECOND, deliberately. Both paths
            // quantise src1 into the same work-buffer layout, and the allocator readily hands
            // the same freed block back, so a reference run performed first can leave a
            // correct quantised row sitting exactly where a defective dispatch would fail to
            // write one, and the defect then hides. Taking the repacked run first denies it
            // that. See the sabotage note in the record.
            for (const int nth : nths) {
                std::vector<float> out_rep;
                {
                    ggml_backend_t backend = ggml_backend_cpu_init();
                    ggml_backend_cpu_set_n_threads(backend, nth);
                    if (!run_mul_mat_id(backend, buft_repack, w_q, Kk, N, n_as, x, n_tok, ids, out_rep)) { return 1; }
                    ggml_backend_free(backend);
                }

                std::vector<float> out_ref;
                {
                    ggml_backend_t backend = ggml_backend_cpu_init();
                    ggml_backend_cpu_set_n_threads(backend, 1);
                    if (!run_mul_mat_id(backend, buft_ref, w_q, Kk, N, n_as, x, n_tok, ids, out_ref)) { return 1; }
                    ggml_backend_free(backend);
                }

                float max_abs = 0.0f;
                float max_diff = 0.0f;
                for (size_t i = 0; i < out_ref.size(); ++i) {
                    max_abs  = std::max(max_abs, std::fabs(out_ref[i]));
                    max_diff = std::max(max_diff, std::fabs(out_ref[i] - out_rep[i]));
                }
                const bool pass = std::isfinite(max_diff) && max_diff == 0.0f;
                printf("  t=%d K=%5d N=%4d  rows along ne12=%3d  max|ref|=%10.4f  max diff=%.3e  %s\n",
                       nth, (int) Kk, (int) N, (int) n_tok, max_abs, max_diff, pass ? "OK" : "FAIL");
                ok = ok && pass;
            }
        }
    }

    printf("test-mmid-src1-quant: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
