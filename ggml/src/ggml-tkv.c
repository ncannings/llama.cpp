// Native packed KV formats: balanced-ternary planes with a 64-element block.
//
// Branch native-packed-kv. One block is one 64-wide head row, so the block
// divides the head dimension exactly and the cache needs no padding.
//
//   TKV1  one balanced trit per element, one scale per block
//   TKV3  three balanced-ternary planes, digit p carrying weight 3^p, which
//         represents every integer in [-13, 13] exactly
//
// Trits are packed five per byte (3^5 = 243 <= 256), so a 64-element plane is
// 13 bytes (the last byte carries four trits). The integer -> planes -> integer
// round trip is exact by construction; all loss is in the scalar quantisation
// in front of it, which is what the isolation tests measure separately.

#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#include <assert.h>
#include <math.h>
#include <string.h>

#define TKV_TRITS_PER_BYTE 5

static const int tkv_pow3[5] = {1, 3, 9, 27, 81};

// pack 64 trits in {-1,0,1} into 13 bytes
static void tkv_pack_plane(const int8_t * trits, uint8_t * out) {
    memset(out, 0, TKV_PLANE_BYTES);
    for (int i = 0; i < QK_TKV; ++i) {
        const int byte = i / TKV_TRITS_PER_BYTE;
        const int slot = i % TKV_TRITS_PER_BYTE;
        out[byte] += (uint8_t) ((trits[i] + 1) * tkv_pow3[slot]);   // digit in {0,1,2}
    }
}

static void tkv_unpack_plane(const uint8_t * in, int8_t * trits) {
    for (int i = 0; i < QK_TKV; ++i) {
        const int byte = i / TKV_TRITS_PER_BYTE;
        const int slot = i % TKV_TRITS_PER_BYTE;
        trits[i] = (int8_t) ((in[byte] / tkv_pow3[slot]) % 3 - 1);
    }
}

// integer in [-levels, levels] -> planes (exact for levels <= (3^planes-1)/2)
static void tkv_int_to_planes(const int * q, int planes, uint8_t * out) {
    int8_t trits[QK_TKV];
    int work[QK_TKV];
    for (int i = 0; i < QK_TKV; ++i) {
        work[i] = q[i];
    }
    for (int p = 0; p < planes; ++p) {
        for (int i = 0; i < QK_TKV; ++i) {
            int r = work[i] % 3;
            if (r == 2)  r = -1;
            if (r == -2) r =  1;
            trits[i] = (int8_t) r;
            work[i] = (work[i] - r) / 3;
        }
        tkv_pack_plane(trits, out + p * TKV_PLANE_BYTES);
    }
}

static void tkv_planes_to_int(const uint8_t * in, int planes, int * q) {
    int8_t trits[QK_TKV];
    for (int i = 0; i < QK_TKV; ++i) {
        q[i] = 0;
    }
    for (int p = 0; p < planes; ++p) {
        tkv_unpack_plane(in + p * TKV_PLANE_BYTES, trits);
        int w = 1;
        for (int k = 0; k < p; ++k) {
            w *= 3;
        }
        for (int i = 0; i < QK_TKV; ++i) {
            q[i] += trits[i] * w;
        }
    }
}

static int tkv_levels(int planes) {
    int r = 1;
    for (int p = 0; p < planes; ++p) {
        r *= 3;
    }
    return (r - 1) / 2;
}

// ------------------------------------------------------------------ quantise

static void tkv_quantize(const float * x, void * vy, int64_t k, int planes) {
    assert(k % QK_TKV == 0);
    const int nb = k / QK_TKV;
    const int levels = tkv_levels(planes);
    const size_t bs = planes == 1 ? sizeof(block_tkv1) : sizeof(block_tkv3);
    uint8_t * y = (uint8_t *) vy;

    for (int b = 0; b < nb; ++b) {
        const float * xb = x + b * QK_TKV;
        float amax = 0.0f;
        for (int i = 0; i < QK_TKV; ++i) {
            const float v = fabsf(xb[i]);
            if (v > amax) {
                amax = v;
            }
        }
        const float d  = amax / levels;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;

        int q[QK_TKV];
        for (int i = 0; i < QK_TKV; ++i) {
            int v = (int) roundf(xb[i] * id);
            if (v >  levels) v =  levels;
            if (v < -levels) v = -levels;
            q[i] = v;
        }
        uint8_t * blk = y + (size_t) b * bs;
        ggml_half dh = GGML_FP32_TO_FP16(d);
        memcpy(blk, &dh, sizeof(ggml_half));
        tkv_int_to_planes(q, planes, blk + sizeof(ggml_half));
    }
}

static void tkv_dequantize(const void * vx, float * y, int64_t k, int planes) {
    assert(k % QK_TKV == 0);
    const int nb = k / QK_TKV;
    const size_t bs = planes == 1 ? sizeof(block_tkv1) : sizeof(block_tkv3);
    const uint8_t * x = (const uint8_t *) vx;

    for (int b = 0; b < nb; ++b) {
        const uint8_t * blk = x + (size_t) b * bs;
        ggml_half dh;
        memcpy(&dh, blk, sizeof(ggml_half));
        const float d = GGML_FP16_TO_FP32(dh);
        int q[QK_TKV];
        tkv_planes_to_int(blk + sizeof(ggml_half), planes, q);
        for (int i = 0; i < QK_TKV; ++i) {
            y[b * QK_TKV + i] = d * q[i];
        }
    }
}

void quantize_row_tkv1_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    tkv_quantize(x, y, k, 1);
}

void quantize_row_tkv3_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    tkv_quantize(x, y, k, 3);
}

void dequantize_row_tkv1(const block_tkv1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    tkv_dequantize(x, y, k, 1);
}

void dequantize_row_tkv3(const block_tkv3 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    tkv_dequantize(x, y, k, 3);
}

size_t quantize_tkv1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights;
    const size_t row_size = ggml_row_size(GGML_TYPE_TKV1, n_per_row);
    for (int64_t r = 0; r < nrow; ++r) {
        tkv_quantize(src + r * n_per_row, (uint8_t *) dst + r * row_size, n_per_row, 1);
    }
    return nrow * row_size;
}

size_t quantize_tkv3(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights;
    const size_t row_size = ggml_row_size(GGML_TYPE_TKV3, n_per_row);
    for (int64_t r = 0; r < nrow; ++r) {
        tkv_quantize(src + r * n_per_row, (uint8_t *) dst + r * row_size, n_per_row, 3);
    }
    return nrow * row_size;
}

// ------------------------------------------------------------------- vec_dot
// The partner type is F32: the block is decoded into integers, and the dot is
// accumulated in the block's scale. Correctness first; this is the quality
// gate, and throughput is measured separately under its own kill criterion.

static void tkv_vec_dot(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx,
                        const float * GGML_RESTRICT vy, int planes) {
    assert(n % QK_TKV == 0);
    const int nb = n / QK_TKV;
    const size_t bs = planes == 1 ? sizeof(block_tkv1) : sizeof(block_tkv3);
    const uint8_t * x = (const uint8_t *) vx;

    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t * blk = x + (size_t) b * bs;
        ggml_half dh;
        memcpy(&dh, blk, sizeof(ggml_half));
        const float d = GGML_FP16_TO_FP32(dh);
        int q[QK_TKV];
        tkv_planes_to_int(blk + sizeof(ggml_half), planes, q);
        float acc = 0.0f;
        for (int i = 0; i < QK_TKV; ++i) {
            acc += (float) q[i] * vy[b * QK_TKV + i];
        }
        sum += d * acc;
    }
    *s = sum;
}

void ggml_vec_dot_tkv1_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    (void) bs; (void) bx; (void) by; (void) nrc;
    tkv_vec_dot(n, s, vx, (const float *) vy, 1);
}

void ggml_vec_dot_tkv3_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    (void) bs; (void) bx; (void) by; (void) nrc;
    tkv_vec_dot(n, s, vx, (const float *) vy, 3);
}
