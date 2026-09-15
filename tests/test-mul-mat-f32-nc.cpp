// Checks the column-blocked F32 mul_mat path (ggml_vec_dot_f32_nc, reached from
// ggml_compute_forward_mul_mat_one_chunk) against the result ggml produced before it existed.
//
// Two independent references, both required to agree exactly:
//
//   1. use_ref. The CPU backend's "use reference implementation" switch keeps the op on the
//      old one-dot-per-output loop, so the same binary can compute the pre-change result and
//      the new one on the same graph. This is the bit that must be exactly 0.
//   2. vec_dot. The old loop called ggml_vec_dot_f32 once per output element, so calling it
//      once per output element here reproduces the pre-change arithmetic from outside ggml
//      entirely. It is exact whenever the op takes the generic path, which the test reports.
//
// The shapes are the Maple router (ne00 = 2048, ne01 = 256) at ne11 = 1, 4 and 64, plus a
// deliberately awkward shape (ne00 = 68, ne01 = 19) whose ne00 is not a multiple of
// GGML_F32_STEP and whose ne01 is not a multiple of the column block, so the scalar leftover
// loop and the column remainder are both exercised. Each shape runs with src1 contiguous,
// which is what the model does, and with src1 strided, which forces the generic path even at
// ne11 >= 4 where llamafile sgemm would otherwise claim the contiguous case.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// dst[i1][i0] = sum_k w[i1][k] * x[i1row][k], one ggml_vec_dot_f32 call per output element,
// which is exactly what ggml_compute_forward_mul_mat_one_chunk did before the change.
static void reference_vec_dot(std::vector<float> & dst, const std::vector<float> & w,
                              const std::vector<float> & x, int64_t ne00, int64_t ne01,
                              int64_t ne11, size_t x_row_stride_f) {
    const ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu(GGML_TYPE_F32);
    dst.assign((size_t) ne01*ne11, 0.0f);
    for (int64_t i11 = 0; i11 < ne11; ++i11) {
        const float * xr = x.data() + (size_t) i11*x_row_stride_f;
        for (int64_t i01 = 0; i01 < ne01; ++i01) {
            tr->vec_dot((int) ne00, &dst[(size_t) i11*ne01 + i01], 0,
                        w.data() + (size_t) i01*ne00, 0, xr, 0, 1);
        }
    }
}

static bool run_graph(ggml_backend_t backend, bool use_ref, const std::vector<float> & w,
                      const std::vector<float> & x, int64_t ne00, int64_t ne01, int64_t ne11,
                      int64_t x_pad, std::vector<float> & out) {
    ggml_init_params cp = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(cp);

    ggml_tensor * wt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne00, ne01);
    // x is allocated with ne00 + x_pad columns and viewed as its first ne00, so x_pad > 0
    // gives a strided (non-contiguous) src1 and x_pad == 0 gives the contiguous one
    ggml_tensor * xa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne00 + x_pad, ne11);
    ggml_tensor * xt = ggml_view_2d(ctx, xa, ne00, ne11, xa->nb[1], 0);
    ggml_tensor * y  = ggml_mul_mat(ctx, wt, xt);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        fprintf(stderr, "alloc failed\n");
        return false;
    }
    ggml_backend_tensor_set(wt, w.data(), 0, w.size()*sizeof(float));
    ggml_backend_tensor_set(xa, x.data(), 0, x.size()*sizeof(float));

    ggml_backend_cpu_set_use_ref(backend, use_ref);
    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    ggml_backend_cpu_set_use_ref(backend, false);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        return false;
    }

    out.resize(ggml_nelements(y));
    ggml_backend_tensor_get(y, out.data(), 0, out.size()*sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

int main(void) {
    ggml_backend_load_all();

    const int64_t shapes[][2] = { { 2048, 256 }, { 68, 19 } };
    const int64_t ne11s[]     = { 1, 4, 64 };
    const int64_t pads[]      = { 0, 5 };
    const int     threads[]   = { 1, 4 };

    std::mt19937 rng(20260915);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    bool ok = true;

    for (const int nth : threads) {
        ggml_backend_t backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backend, nth);

        for (const auto & shape : shapes) {
            const int64_t ne00 = shape[0];
            const int64_t ne01 = shape[1];

            std::vector<float> w((size_t) ne00*ne01);
            for (auto & v : w) { v = dist(rng); }

            for (const int64_t ne11 : ne11s) {
                for (const int64_t pad : pads) {
                    std::vector<float> x((size_t) (ne00 + pad)*ne11);
                    for (auto & v : x) { v = dist(rng); }

                    std::vector<float> out_ref;
                    std::vector<float> out_new;
                    if (!run_graph(backend, true,  w, x, ne00, ne01, ne11, pad, out_ref)) { return 1; }
                    if (!run_graph(backend, false, w, x, ne00, ne01, ne11, pad, out_new)) { return 1; }

                    std::vector<float> out_vd;
                    reference_vec_dot(out_vd, w, x, ne00, ne01, ne11, (size_t) (ne00 + pad));

                    float max_abs = 0.0f;
                    for (const float v : out_ref) { max_abs = std::max(max_abs, std::fabs(v)); }

                    const float d_ref = max_abs_diff(out_ref, out_new);
                    const float d_vd  = max_abs_diff(out_vd,  out_new);

                    // the generic path is taken whenever the vec_dot reference is reproduced
                    // bit for bit; otherwise llamafile sgemm claimed the op and neither the
                    // old nor the new code ran at all, which must show as d_ref == 0 too
                    const char * path = (d_vd == 0.0f) ? "generic" : "sgemm  ";
                    const bool pass = std::isfinite(d_ref) && d_ref == 0.0f &&
                                      (d_vd == 0.0f || (pad == 0 && ne11 >= 4 && d_vd <= 1e-3f*max_abs));

                    printf("t=%d K=%5d N=%4d ne11=%3d src1=%-13s path=%s  max|ref|=%10.4f  "
                           "max diff vs use_ref=%.3e  vs vec_dot=%.3e  %s\n",
                           nth, (int) ne00, (int) ne01, (int) ne11,
                           pad == 0 ? "contiguous" : "strided", path, max_abs, d_ref, d_vd,
                           pass ? "OK" : "FAIL");
                    ok = ok && pass;
                }
            }
        }

        ggml_backend_free(backend);
    }

    printf("test-mul-mat-f32-nc: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
