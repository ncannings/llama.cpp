// Holds the inline asm q4_K 8x8 i8mm GEMM to the intrinsics kernel it replaces, BIT FOR BIT.
//
// The reference here is deliberately NOT the scalar ggml_gemm_q4_K_8x8_q8_K_generic. The
// claim being tested is that ggml_gemm_q4_K_8x8_q8_K reproduces exactly what the repacked
// i8mm path produced before the asm kernel went in, which is what the Maple output head
// (output.weight, 151936 x 2048, Q4_K, repacked to q4_K_8x8) runs through on every decode
// step. Rounding differences against the scalar path are expected and are not the subject.
//
// Shape: a 2048 x 1024 Q4_K matrix, so 8 superblocks of K and 128 interleaved column
// groups, at 1, 2, 3, 4, 5, 8, 9, 12, 16, 17 and 64 activation rows. The kernel now handles
// EIGHT rows at a time wherever two 4-row panels are available and four for a trailing odd
// panel, so the row counts cover every residue of 4 below 8 (which stays on the 4-row path),
// every residue of 8 above it, and 64 for the batch the Maple head actually sees;
// forward_mul_mat rounds the row count up to a multiple of 4 and the panel carries the
// padding, which is what the test reproduces.
//
// TWO references, and both must be exact. ggml_gemm_q4_K_8x8_q8_K_intrinsics is the original
// claim, unchanged since the asm kernel went in. ggml_gemm_q4_K_8x8_q8_K_asm4 is the 4-row
// asm kernel the 8-row tile was measured against and is the reference the 8-row work claims
// against directly; it is the same code the dispatched entry point still runs for a trailing
// odd panel, so a row count of 4 or 12 exercises it on both sides.
//
// Max diff must be exactly 0. A deliberate off-by-one anywhere in the kernel must make this
// test fail; that sabotage check is recorded in docs/findings/2026-09-15-q4k-head-gemm.md.

#include "ggml.h"
#include "ggml-cpu.h"

#include "repack.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// The arch flags that turn on i8mm are applied to the ggml-cpu target, not to this one, so
// the dispatch inside the library is not visible here. Guard only on what decides whether
// ggml_gemm_q4_K_8x8_q8_K_intrinsics exists at all.
#if defined(__aarch64__) && defined(__ARM_NEON)
#    define TEST_Q4K_HEAD_ACTIVE 1
#endif

#if defined(TEST_Q4K_HEAD_ACTIVE)

static ggml_half f32_to_half(float f) {
    return ggml_fp32_to_fp16(f);
}

int main(void) {
    const int K  = 2048;          // n, the reduction dimension
    const int NC = 1024;          // columns
    const int nb = K / QK_K;      // 8 superblocks
    const int ng = NC / 8;        // 128 interleaved column groups

    const int row_counts[] = { 1, 2, 3, 4, 5, 8, 9, 12, 16, 17, 64 };

    std::mt19937 rng(20260915);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<int> q8_dist(-127, 127);
    std::uniform_real_distribution<float> d_dist(0.0005f, 0.01f);

    // Random q4_Kx8 weights. Every bit pattern is a legal repacked block: the 6-bit scales
    // and mins are carved out of scales[] by mask and shift, and qs[] is raw nibbles.
    std::vector<block_q4_Kx8> q4((size_t) ng * nb);
    for (auto & blk : q4) {
        for (int i = 0; i < 8; i++) {
            blk.d[i]    = f32_to_half(d_dist(rng));
            blk.dmin[i] = f32_to_half(d_dist(rng) * 0.25f);
        }
        for (int i = 0; i < 96; i++) {
            blk.scales[i] = (uint8_t) byte_dist(rng);
        }
        for (int i = 0; i < 1024; i++) {
            blk.qs[i] = (uint8_t) byte_dist(rng);
        }
    }

    int    fails    = 0;
    double worst    = 0.0;
    size_t bad_bits = 0;

    for (int nr_real : row_counts) {
        const int nr_pad = (nr_real + 3) / 4 * 4;   // what forward_mul_mat hands the gemm

        // Activation panel: nr_pad/4 groups of interleaved q8_Kx4, the tail rows zero
        // filled exactly as ggml_quantize_mat_q8_K_4x8 would leave them.
        std::vector<block_q8_Kx4> q8((size_t) (nr_pad / 4) * nb);
        for (int g = 0; g < nr_pad / 4; g++) {
            for (int b = 0; b < nb; b++) {
                block_q8_Kx4 & blk = q8[(size_t) g * nb + b];
                memset(&blk, 0, sizeof(blk));
                for (int t = 0; t < 4; t++) {
                    const int row = g * 4 + t;
                    blk.d[t] = row < nr_real ? d_dist(rng) * 20.0f : 0.0f;
                }
                // qs layout: [group of 4 sub-blocks][4 quants-of-16 slots], rows interleaved
                // in 16 byte runs. Fill everything, then zero the padding rows.
                for (int i = 0; i < QK_K * 4; i++) {
                    blk.qs[i] = (int8_t) q8_dist(rng);
                }
                for (int sb4 = 0; sb4 < 16; sb4++) {
                    for (int t = 0; t < 4; t++) {
                        if (g * 4 + t >= nr_real) {
                            memset(blk.qs + sb4 * 64 + t * 16, 0, 16);
                        }
                    }
                }
                // bsums[g16][t][j] is the sum of the 16 quants of row t, sub-block j of
                // the group, matching ggml_quantize_mat_q8_K_4x8
                for (int grp = 0; grp < 4; grp++) {
                    for (int t = 0; t < 4; t++) {
                        for (int j = 0; j < 4; j++) {
                            int sum = 0;
                            const int8_t * p = blk.qs + (grp * 4 + j) * 64 + t * 16;
                            for (int e = 0; e < 16; e++) {
                                sum += p[e];
                            }
                            blk.bsums[grp * 16 + t * 4 + j] = (int16_t) sum;
                        }
                    }
                }
            }
        }

        std::vector<float> out_ref((size_t) nr_pad * NC, 0.0f);
        std::vector<float> out_a4((size_t) nr_pad * NC, 0.0f);
        std::vector<float> out_new((size_t) nr_pad * NC, 0.0f);

        ggml_gemm_q4_K_8x8_q8_K_intrinsics(K, out_ref.data(), NC, q4.data(), q8.data(), nr_pad, NC);
        ggml_gemm_q4_K_8x8_q8_K_asm4(K, out_a4.data(), NC, q4.data(), q8.data(), nr_pad, NC);
        ggml_gemm_q4_K_8x8_q8_K(K, out_new.data(), NC, q4.data(), q8.data(), nr_pad, NC);

        size_t differing = 0, differing_a4 = 0;
        double maxdiff = 0.0, maxdiff_a4 = 0.0;
        for (int r = 0; r < nr_real; r++) {          // only the real rows are the claim
            for (int c = 0; c < NC; c++) {
                const size_t idx = (size_t) r * NC + c;
                if (memcmp(&out_ref[idx], &out_new[idx], sizeof(float)) != 0) {
                    differing++;
                }
                const double d = std::fabs((double) out_ref[idx] - (double) out_new[idx]);
                if (d > maxdiff) {
                    maxdiff = d;
                }
                if (memcmp(&out_a4[idx], &out_new[idx], sizeof(float)) != 0) {
                    differing_a4++;
                }
                const double d4 = std::fabs((double) out_a4[idx] - (double) out_new[idx]);
                if (d4 > maxdiff_a4) {
                    maxdiff_a4 = d4;
                }
            }
        }
        if (maxdiff > worst) {
            worst = maxdiff;
        }
        if (maxdiff_a4 > worst) {
            worst = maxdiff_a4;
        }
        bad_bits += differing + differing_a4;

        const bool ok = (maxdiff == 0.0 && differing == 0 && maxdiff_a4 == 0.0 && differing_a4 == 0);
        printf("rows %2d (padded to %2d): vs intrinsics max diff %.17g, differing %zu | "
               "vs 4-row asm max diff %.17g, differing %zu -> %s\n",
               nr_real, nr_pad, maxdiff, differing, maxdiff_a4, differing_a4,
               ok ? "IDENTICAL" : "DIFFER");
        if (!ok) {
            fails++;
        }
    }

    printf("\nq4_K 8x8 head gemm vs intrinsics and 4-row asm kernels: %s (worst max diff %.17g, %zu differing floats)\n",
           fails == 0 ? "PASS" : "FAIL", worst, bad_bits);
    return fails == 0 ? 0 : 1;
}

#else

int main(void) {
    printf("q4_K 8x8 head gemm test: skipped, needs aarch64 NEON\n");
    return 0;
}

#endif
