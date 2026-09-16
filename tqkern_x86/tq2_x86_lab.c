// TQ2_0 kernel ceilings and variants on x86, single core, cache-resident.
//
// There is no x86 repack (CPU_REPACK) path for TQ2_0 in this tree: the repack traits for
// GGML_TYPE_TQ2_0 are gated on ggml_cpu_has_neon(), and arch/x86/repack.cpp defines no
// tq2_0 kernel, so every TQ2_0 matmul on x86 (decode AND prefill) goes through the
// single-row ggml_vec_dot_tq2_0_q8_K. That kernel is therefore the whole x86 story and
// the baseline here.
//
// Floors, all on the same cache-resident matrix and the same q8_K token:
//   (a) vec_dot AVX2    - verbatim copy of ggml_vec_dot_tq2_0_q8_K (the shipped kernel)
//   (c) load only       - streams exactly the same code bytes, one XOR per load
//   (d2) maddubs only   - same loads, same 4 vpmaddubsw + 4 vpaddw per 32 code bytes, no unpack
//   (d3) vpdpbusd only  - same loads, 4 vpdpbusd (ymm) per 32 code bytes, no unpack
//   (d4) vpdpbusd zmm   - same loads, 4 vpdpbusd (zmm) per 64 code bytes, no unpack
// Variants keep the same integer sum per block, the same bsums correction and the same
// float association, so their output is bit-identical to (a).
//
// Build: gcc -O3 -march=native -o tq2_x86_lab tq2_x86_lab.c -lm
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define QK_K 256
typedef uint16_t ggml_half;
typedef struct { uint8_t qs[QK_K/4]; ggml_half d; } block_tq2_0;             // 66 B
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;

#ifndef NROWS
#define NROWS 2048
#endif
#ifndef KDIM
#define KDIM 1024
#endif
#define NB (KDIM/QK_K)

static block_tq2_0 *W;     // [NROWS][NB]
static block_q8_K  *Y;     // [NB]
static float       *outA, *outB, *outT;
static int8_t      *Wref;  // [NROWS][KDIM] in -1/0/1

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
static float half2f(ggml_half h){ return _cvtsh_ss(h); }
static ggml_half f2half(float f){ return _cvtss_sh(f, 0); }

static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static void gen(void){
    unsigned st = 12345u;
#define RND (st = st*1664525u + 1013904223u, (int)((st>>16)&0x7fff))
    for (int r=0;r<NROWS;r++){
        for (int b=0;b<NB;b++){
            block_tq2_0 *x = &W[r*NB+b];
            x->d = f2half(0.01f + 0.0001f*(r%13));
            for (int j=0;j<64;j+=32)
                for (int m=0;m<32;m++){
                    uint8_t q=0;
                    for (int n=0;n<4;n++){
                        int v = (RND%3)-1;
                        Wref[(size_t)r*KDIM + b*QK_K + j*4 + m + n*32] = (int8_t)v;
                        q |= (uint8_t)((v+1)<<(2*n));
                    }
                    x->qs[j+m]=q;
                }
        }
    }
    for (int b=0;b<NB;b++){
        Y[b].d = 0.02f;
        for (int k=0;k<QK_K;k++) Y[b].qs[k] = (int8_t)((RND%255)-127);
        for (int g=0;g<16;g++){ int s=0; for(int k=0;k<16;k++) s+=Y[b].qs[g*16+k]; Y[b].bsums[g]=(int16_t)s; }
    }
#undef RND
}

static void ref_scalar(float *o){
    for (int r=0;r<NROWS;r++){
        float s=0;
        for (int b=0;b<NB;b++){
            int32_t a=0;
            for (int k=0;k<QK_K;k++) a += Wref[(size_t)r*KDIM+b*QK_K+k]*Y[b].qs[k];
            s += half2f(W[r*NB+b].d)*Y[b].d*(float)a;
        }
        o[r]=s;
    }
}

/* ---------------- (a) the shipped AVX2 kernel, verbatim ---------------- */
static inline float vec_dot_avx2(const block_tq2_0 *x, const block_q8_K *y){
    __m256 sumf = _mm256_setzero_ps();
    for (int i = 0; i < NB; ++i) {
        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();
        for (size_t j = 0; j < 64; j += 32) {
            __m256i qx0 = _mm256_loadu_si256((const __m256i *) (x[i].qs + j));
            __m256i qx1 = _mm256_srli_epi16(qx0, 2);
            __m256i qx2 = _mm256_srli_epi16(qx0, 4);
            __m256i qx3 = _mm256_srli_epi16(qx0, 6);
            qx0 = _mm256_and_si256(qx0, _mm256_set1_epi8(3));
            qx1 = _mm256_and_si256(qx1, _mm256_set1_epi8(3));
            qx2 = _mm256_and_si256(qx2, _mm256_set1_epi8(3));
            qx3 = _mm256_and_si256(qx3, _mm256_set1_epi8(3));
            const __m256i qy0 = _mm256_loadu_si256((const __m256i *) (y[i].qs + j*4 +  0));
            const __m256i qy1 = _mm256_loadu_si256((const __m256i *) (y[i].qs + j*4 + 32));
            const __m256i qy2 = _mm256_loadu_si256((const __m256i *) (y[i].qs + j*4 + 64));
            const __m256i qy3 = _mm256_loadu_si256((const __m256i *) (y[i].qs + j*4 + 96));
            qx0 = _mm256_maddubs_epi16(qx0, qy0);
            qx1 = _mm256_maddubs_epi16(qx1, qy1);
            qx2 = _mm256_maddubs_epi16(qx2, qy2);
            qx3 = _mm256_maddubs_epi16(qx3, qy3);
            sumi0 = _mm256_add_epi16(sumi0, _mm256_add_epi16(qx0, qx1));
            sumi1 = _mm256_add_epi16(sumi1, _mm256_add_epi16(qx2, qx3));
        }
        const __m256i ysum = _mm256_loadu_si256((const __m256i *) y[i].bsums);
        const __m256 d = _mm256_set1_ps(y[i].d * half2f(x[i].d));
        sumi0 = _mm256_add_epi16(sumi0, sumi1);
        sumi0 = _mm256_sub_epi16(sumi0, ysum);
        sumi0 = _mm256_madd_epi16(sumi0, _mm256_set1_epi16(1));
        sumf = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(sumi0), d), sumf);
    }
    return hsum_float_8(sumf);
}
static void k_avx2(float *o){ for (int r=0;r<NROWS;r++) o[r]=vec_dot_avx2(&W[r*NB],Y); }

/* the int32 lane vector that the float epilogue consumes, shared by every variant:
   lane k = sum over the four planes and both halves of code bytes 4k..4k+3, minus
   bsums[2k]+bsums[2k+1]. Identical arithmetic to (a) by exactness, not by tolerance. */
static inline __m256 epi_add(__m256 sumf, __m256i sumi, const block_tq2_0 *x, const block_q8_K *y){
    const __m256 d = _mm256_set1_ps(y->d * half2f(x->d));
    return _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(sumi), d), sumf);
}
static inline __m256i bsum32(const block_q8_K *y){
    const __m256i ysum = _mm256_loadu_si256((const __m256i *) y->bsums);
    return _mm256_madd_epi16(ysum, _mm256_set1_epi16(1));
}

/* ---------------- (c) memory floor ---------------- */
static __m256i sink_v;
static void k_loadonly(float *s){
    __m256i a0=_mm256_setzero_si256(), a1=a0;
    for (int r=0;r<NROWS;r++)
        for (int b=0;b<NB;b++){
            const uint8_t *p = W[r*NB+b].qs;
            a0=_mm256_xor_si256(a0,_mm256_loadu_si256((const __m256i*)(p)));
            a1=_mm256_xor_si256(a1,_mm256_loadu_si256((const __m256i*)(p+32)));
        }
    sink_v=_mm256_xor_si256(a0,a1);
    s[0]=(float)_mm256_extract_epi8(sink_v,0);
}

/* ---------------- (d2) AVX2 arithmetic floor: same loads, 4 maddubs, no unpack ------ */
static void k_maddubs_only(float *s){
    __m256i sink=_mm256_setzero_si256();
    for (int r=0;r<NROWS;r++){
        const block_tq2_0 *x=&W[r*NB];
        for (int i=0;i<NB;i++){
            __m256i sumi0=_mm256_setzero_si256(), sumi1=_mm256_setzero_si256();
            for (size_t j=0;j<64;j+=32){
                const __m256i qx = _mm256_loadu_si256((const __m256i *)(x[i].qs + j));
                const __m256i qy0 = _mm256_loadu_si256((const __m256i *) (Y[i].qs + j*4 +  0));
                const __m256i qy1 = _mm256_loadu_si256((const __m256i *) (Y[i].qs + j*4 + 32));
                const __m256i qy2 = _mm256_loadu_si256((const __m256i *) (Y[i].qs + j*4 + 64));
                const __m256i qy3 = _mm256_loadu_si256((const __m256i *) (Y[i].qs + j*4 + 96));
                sumi0=_mm256_add_epi16(sumi0,_mm256_add_epi16(_mm256_maddubs_epi16(qx,qy0),_mm256_maddubs_epi16(qx,qy1)));
                sumi1=_mm256_add_epi16(sumi1,_mm256_add_epi16(_mm256_maddubs_epi16(qx,qy2),_mm256_maddubs_epi16(qx,qy3)));
            }
            sink=_mm256_add_epi16(sink,_mm256_add_epi16(sumi0,sumi1));
        }
    }
    s[0]=(float)_mm256_extract_epi16(sink,0);
}

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
/* ---------------- (d3) VNNI floor, ymm: same loads, 4 vpdpbusd, no unpack ---------- */
static void k_vnni_only(float *s){
    __m256i sink=_mm256_setzero_si256();
    for (int r=0;r<NROWS;r++){
        const block_tq2_0 *x=&W[r*NB];
        for (int i=0;i<NB;i++){
            __m256i a0=_mm256_setzero_si256(),a1=a0,a2=a0,a3=a0;
            for (size_t j=0;j<64;j+=32){
                const __m256i qx = _mm256_loadu_si256((const __m256i *)(x[i].qs + j));
                a0=_mm256_dpbusd_epi32(a0,qx,_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 +  0)));
                a1=_mm256_dpbusd_epi32(a1,qx,_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 + 32)));
                a2=_mm256_dpbusd_epi32(a2,qx,_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 + 64)));
                a3=_mm256_dpbusd_epi32(a3,qx,_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 + 96)));
            }
            sink=_mm256_add_epi32(sink,_mm256_add_epi32(_mm256_add_epi32(a0,a1),_mm256_add_epi32(a2,a3)));
        }
    }
    s[0]=(float)_mm256_extract_epi32(sink,0);
}
/* ---------------- (d4) VNNI floor, zmm ---------------- */
static void k_vnni512_only(float *s){
    __m512i sink=_mm512_setzero_si512();
    for (int r=0;r<NROWS;r++){
        const block_tq2_0 *x=&W[r*NB];
        for (int i=0;i<NB;i++){
            __m512i a0=_mm512_setzero_si512(),a1=a0;
            const __m512i c0 = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(x[i].qs)));
            const __m512i c1 = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(x[i].qs+32)));
            a0=_mm512_dpbusd_epi32(a0,c0,_mm512_loadu_si512((const void *)(Y[i].qs +   0)));
            a1=_mm512_dpbusd_epi32(a1,c0,_mm512_loadu_si512((const void *)(Y[i].qs +  64)));
            a0=_mm512_dpbusd_epi32(a0,c1,_mm512_loadu_si512((const void *)(Y[i].qs + 128)));
            a1=_mm512_dpbusd_epi32(a1,c1,_mm512_loadu_si512((const void *)(Y[i].qs + 192)));
            sink=_mm512_add_epi32(sink,_mm512_add_epi32(a0,a1));
        }
    }
    s[0]=(float)_mm512_cvtsi512_si32(sink);
}
#endif

/* ---------------- X1: AVX2 mask-only for planes 0,1,2 (one shift left), 3 scale classes
   plane0 = w & 0x03 (scale 1)   plane1 = w & 0x0c (scale 4)
   plane2 = w & 0x30 (scale 16)  plane3 = (w >> 6)  (scale 1, shares acc with plane0)
   Each accumulator holds an exact multiple of its scale, so one arithmetic right shift
   per block undoes it exactly (negatives included) and the int16 lane totals are the
   same integers the shifted form produced. maddubs needs the code operand unsigned,
   which it is; the scaled products stay inside int16 (32*127*2 per lane pair). */
static void k_x1(float *o){
    const __m256i m03=_mm256_set1_epi8(0x03), m0c=_mm256_set1_epi8(0x0c), m30=_mm256_set1_epi8(0x30), m03s=_mm256_set1_epi8(0x03);
    for (int r=0;r<NROWS;r++){
        const block_tq2_0 *x=&W[r*NB];
        __m256 sumf=_mm256_setzero_ps();
        for (int i=0;i<NB;i++){
            __m256i s1=_mm256_setzero_si256(), s4=_mm256_setzero_si256(), s16=_mm256_setzero_si256();
            for (size_t j=0;j<64;j+=32){
                const __m256i w = _mm256_loadu_si256((const __m256i *)(x[i].qs + j));
                const __m256i p0=_mm256_and_si256(w,m03);
                const __m256i p1=_mm256_and_si256(w,m0c);
                const __m256i p2=_mm256_and_si256(w,m30);
                const __m256i p3=_mm256_and_si256(_mm256_srli_epi16(w,6),m03s);
                const __m256i y0=_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 +  0));
                const __m256i y1=_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 + 32));
                const __m256i y2=_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 + 64));
                const __m256i y3=_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 + 96));
                s1 =_mm256_add_epi16(s1 ,_mm256_add_epi16(_mm256_maddubs_epi16(p0,y0),_mm256_maddubs_epi16(p3,y3)));
                s4 =_mm256_add_epi16(s4 ,_mm256_maddubs_epi16(p1,y1));
                s16=_mm256_add_epi16(s16,_mm256_maddubs_epi16(p2,y2));
            }
            __m256i sumi = _mm256_add_epi16(s1, _mm256_add_epi16(_mm256_srai_epi16(s4,2), _mm256_srai_epi16(s16,4)));
            sumi = _mm256_sub_epi16(sumi, _mm256_loadu_si256((const __m256i *) Y[i].bsums));
            sumi = _mm256_madd_epi16(sumi, _mm256_set1_epi16(1));
            sumf = epi_add(sumf, sumi, &x[i], &Y[i]);
        }
        o[r]=hsum_float_8(sumf);
    }
}

/* ---------------- X2: AVX2 mask-only for all four planes, the 64x plane kept in two
   per-half accumulators so it never exceeds int16 (128*127*2 = 32512 per lane pair) --- */
static void k_x2(float *o){
    const __m256i m03=_mm256_set1_epi8(0x03), m0c=_mm256_set1_epi8(0x0c), m30=_mm256_set1_epi8(0x30);
    const __m256i mc0=_mm256_set1_epi8((char)0xc0);
    for (int r=0;r<NROWS;r++){
        const block_tq2_0 *x=&W[r*NB];
        __m256 sumf=_mm256_setzero_ps();
        for (int i=0;i<NB;i++){
            __m256i s1=_mm256_setzero_si256(), s4=_mm256_setzero_si256(), s16=_mm256_setzero_si256();
            __m256i s64a, s64b;
            for (size_t j=0;j<64;j+=32){
                const __m256i w = _mm256_loadu_si256((const __m256i *)(x[i].qs + j));
                const __m256i y0=_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 +  0));
                const __m256i y1=_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 + 32));
                const __m256i y2=_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 + 64));
                const __m256i y3=_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 + 96));
                s1 =_mm256_add_epi16(s1 ,_mm256_maddubs_epi16(_mm256_and_si256(w,m03),y0));
                s4 =_mm256_add_epi16(s4 ,_mm256_maddubs_epi16(_mm256_and_si256(w,m0c),y1));
                s16=_mm256_add_epi16(s16,_mm256_maddubs_epi16(_mm256_and_si256(w,m30),y2));
                const __m256i t =_mm256_maddubs_epi16(_mm256_and_si256(w,mc0),y3);
                if (j==0) s64a=t; else s64b=t;
            }
            // 64x class: shift each half down to the 16x class first, then fold once
            s16 = _mm256_add_epi16(s16, _mm256_add_epi16(_mm256_srai_epi16(s64a,2), _mm256_srai_epi16(s64b,2)));
            __m256i sumi = _mm256_add_epi16(s1, _mm256_add_epi16(_mm256_srai_epi16(s4,2), _mm256_srai_epi16(s16,4)));
            sumi = _mm256_sub_epi16(sumi, _mm256_loadu_si256((const __m256i *) Y[i].bsums));
            sumi = _mm256_madd_epi16(sumi, _mm256_set1_epi16(1));
            sumf = epi_add(sumf, sumi, &x[i], &Y[i]);
        }
        o[r]=hsum_float_8(sumf);
    }
}

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
/* ---------------- X3: VNNI ymm, mask-only for all four planes, four int32 accumulators.
   vpdpbusd takes the code operand unsigned and accumulates into int32, so nothing can
   overflow and no plane needs shifting down: 4 AND and no shifts, and the four
   accumulate-adds of the AVX2 form disappear into the instruction. The int32 lane
   mapping (lane k = code bytes 4k..4k+3) is exactly the lane mapping the shipped kernel
   reaches after its madd_epi16, so the epilogue is unchanged. */
static void k_x3(float *o){
    const __m256i m03=_mm256_set1_epi8(0x03), m0c=_mm256_set1_epi8(0x0c);
    const __m256i m30=_mm256_set1_epi8(0x30), mc0=_mm256_set1_epi8((char)0xc0);
    for (int r=0;r<NROWS;r++){
        const block_tq2_0 *x=&W[r*NB];
        __m256 sumf=_mm256_setzero_ps();
        for (int i=0;i<NB;i++){
            __m256i a0=_mm256_setzero_si256(),a1=a0,a2=a0,a3=a0;
            for (size_t j=0;j<64;j+=32){
                const __m256i w = _mm256_loadu_si256((const __m256i *)(x[i].qs + j));
                a0=_mm256_dpbusd_epi32(a0,_mm256_and_si256(w,m03),_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 +  0)));
                a1=_mm256_dpbusd_epi32(a1,_mm256_and_si256(w,m0c),_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 + 32)));
                a2=_mm256_dpbusd_epi32(a2,_mm256_and_si256(w,m30),_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 + 64)));
                a3=_mm256_dpbusd_epi32(a3,_mm256_and_si256(w,mc0),_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 + 96)));
            }
            __m256i sumi = _mm256_add_epi32(_mm256_add_epi32(a0,_mm256_srai_epi32(a1,2)),
                                            _mm256_add_epi32(_mm256_srai_epi32(a2,4),_mm256_srai_epi32(a3,6)));
            sumi = _mm256_sub_epi32(sumi, bsum32(&Y[i]));
            sumf = epi_add(sumf, sumi, &x[i], &Y[i]);
        }
        o[r]=hsum_float_8(sumf);
    }
}

/* ---------------- X4: VNNI zmm. The 32 code bytes of one half are broadcast into both
   256-bit halves of a zmm and masked with a two-plane mask, so one AND produces two
   planes and one 512-bit y load feeds both. Two broadcasts, four ANDs and four vpdpbusd
   cover the whole 256-weight block. The per-lane scales (1,4 in accA; 16,64 in accB)
   are undone by one variable-shift per accumulator, then the two halves are added,
   which reproduces the same eight int32 lanes. */
static void k_x4(float *o){
    const __m512i m01=_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi8(0x03)),_mm256_set1_epi8(0x0c),1);
    const __m512i m23=_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi8(0x30)),_mm256_set1_epi8((char)0xc0),1);
    const __m512i sh01=_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi32(0)),_mm256_set1_epi32(2),1);
    const __m512i sh23=_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi32(4)),_mm256_set1_epi32(6),1);
    for (int r=0;r<NROWS;r++){
        const block_tq2_0 *x=&W[r*NB];
        __m256 sumf=_mm256_setzero_ps();
        for (int i=0;i<NB;i++){
            __m512i A=_mm512_setzero_si512(), B=A;
            const __m512i c0 = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(x[i].qs)));
            const __m512i c1 = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(x[i].qs+32)));
            A=_mm512_dpbusd_epi32(A,_mm512_and_si512(c0,m01),_mm512_loadu_si512((const void *)(Y[i].qs +   0)));
            B=_mm512_dpbusd_epi32(B,_mm512_and_si512(c0,m23),_mm512_loadu_si512((const void *)(Y[i].qs +  64)));
            A=_mm512_dpbusd_epi32(A,_mm512_and_si512(c1,m01),_mm512_loadu_si512((const void *)(Y[i].qs + 128)));
            B=_mm512_dpbusd_epi32(B,_mm512_and_si512(c1,m23),_mm512_loadu_si512((const void *)(Y[i].qs + 192)));
            A=_mm512_srav_epi32(A,sh01);
            B=_mm512_srav_epi32(B,sh23);
            const __m512i S=_mm512_add_epi32(A,B);
            __m256i sumi=_mm256_add_epi32(_mm512_castsi512_si256(S),_mm512_extracti64x4_epi64(S,1));
            sumi = _mm256_sub_epi32(sumi, bsum32(&Y[i]));
            sumf = epi_add(sumf, sumi, &x[i], &Y[i]);
        }
        o[r]=hsum_float_8(sumf);
    }
}

/* ---------------- X5: X3 with two rows in flight, to see whether the ymm VNNI form is
   latency bound on the four accumulators rather than issue bound. Identity is per row. */
static void k_x5(float *o){
    const __m256i m03=_mm256_set1_epi8(0x03), m0c=_mm256_set1_epi8(0x0c);
    const __m256i m30=_mm256_set1_epi8(0x30), mc0=_mm256_set1_epi8((char)0xc0);
    for (int r=0;r<NROWS;r+=2){
        const block_tq2_0 *xa=&W[r*NB], *xb=&W[(r+1)*NB];
        __m256 sa=_mm256_setzero_ps(), sb=_mm256_setzero_ps();
        for (int i=0;i<NB;i++){
            __m256i a0=_mm256_setzero_si256(),a1=a0,a2=a0,a3=a0;
            __m256i b0=a0,b1=a0,b2=a0,b3=a0;
            for (size_t j=0;j<64;j+=32){
                const __m256i wa = _mm256_loadu_si256((const __m256i *)(xa[i].qs + j));
                const __m256i wb = _mm256_loadu_si256((const __m256i *)(xb[i].qs + j));
                const __m256i y0=_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 +  0));
                const __m256i y1=_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 + 32));
                const __m256i y2=_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 + 64));
                const __m256i y3=_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 + 96));
                a0=_mm256_dpbusd_epi32(a0,_mm256_and_si256(wa,m03),y0);
                a1=_mm256_dpbusd_epi32(a1,_mm256_and_si256(wa,m0c),y1);
                a2=_mm256_dpbusd_epi32(a2,_mm256_and_si256(wa,m30),y2);
                a3=_mm256_dpbusd_epi32(a3,_mm256_and_si256(wa,mc0),y3);
                b0=_mm256_dpbusd_epi32(b0,_mm256_and_si256(wb,m03),y0);
                b1=_mm256_dpbusd_epi32(b1,_mm256_and_si256(wb,m0c),y1);
                b2=_mm256_dpbusd_epi32(b2,_mm256_and_si256(wb,m30),y2);
                b3=_mm256_dpbusd_epi32(b3,_mm256_and_si256(wb,mc0),y3);
            }
            const __m256i bs=bsum32(&Y[i]);
            __m256i sia = _mm256_add_epi32(_mm256_add_epi32(a0,_mm256_srai_epi32(a1,2)),
                                           _mm256_add_epi32(_mm256_srai_epi32(a2,4),_mm256_srai_epi32(a3,6)));
            __m256i sib = _mm256_add_epi32(_mm256_add_epi32(b0,_mm256_srai_epi32(b1,2)),
                                           _mm256_add_epi32(_mm256_srai_epi32(b2,4),_mm256_srai_epi32(b3,6)));
            sa = epi_add(sa, _mm256_sub_epi32(sia,bs), &xa[i], &Y[i]);
            sb = epi_add(sb, _mm256_sub_epi32(sib,bs), &xb[i], &Y[i]);
        }
        o[r]=hsum_float_8(sa); o[r+1]=hsum_float_8(sb);
    }
}
#endif


/* ---------------- (d5) unpack only: the 7 unpack ops and the code loads, no y loads,
   no arithmetic against the activations. What the unpack alone costs. ---------------- */
static void k_unpack_only(float *s){
    __m256i sink=_mm256_setzero_si256();
    for (int r=0;r<NROWS;r++){
        const block_tq2_0 *x=&W[r*NB];
        for (int i=0;i<NB;i++){
            for (size_t j=0;j<64;j+=32){
                __m256i qx0 = _mm256_loadu_si256((const __m256i *)(x[i].qs + j));
                __m256i qx1 = _mm256_srli_epi16(qx0, 2);
                __m256i qx2 = _mm256_srli_epi16(qx0, 4);
                __m256i qx3 = _mm256_srli_epi16(qx0, 6);
                qx0 = _mm256_and_si256(qx0, _mm256_set1_epi8(3));
                qx1 = _mm256_and_si256(qx1, _mm256_set1_epi8(3));
                qx2 = _mm256_and_si256(qx2, _mm256_set1_epi8(3));
                qx3 = _mm256_and_si256(qx3, _mm256_set1_epi8(3));
                sink=_mm256_xor_si256(sink,_mm256_xor_si256(_mm256_xor_si256(qx0,qx1),_mm256_xor_si256(qx2,qx3)));
            }
        }
    }
    sink_v=sink; s[0]=(float)_mm256_extract_epi8(sink,0);
}

/* ---------------- (d6) half the arithmetic: 2 maddubs per 32 code bytes instead of 4,
   same loads. If the shipped kernel is bound by vpmaddubsw this halves the time. ------ */
static void k_maddubs_half(float *s){
    __m256i sink=_mm256_setzero_si256();
    for (int r=0;r<NROWS;r++){
        const block_tq2_0 *x=&W[r*NB];
        for (int i=0;i<NB;i++){
            __m256i sumi0=_mm256_setzero_si256(), sumi1=_mm256_setzero_si256();
            for (size_t j=0;j<64;j+=32){
                const __m256i qx = _mm256_loadu_si256((const __m256i *)(x[i].qs + j));
                const __m256i qy0 = _mm256_loadu_si256((const __m256i *) (Y[i].qs + j*4 +  0));
                const __m256i qy1 = _mm256_loadu_si256((const __m256i *) (Y[i].qs + j*4 + 32));
                const __m256i qy2 = _mm256_loadu_si256((const __m256i *) (Y[i].qs + j*4 + 64));
                const __m256i qy3 = _mm256_loadu_si256((const __m256i *) (Y[i].qs + j*4 + 96));
                sink=_mm256_xor_si256(sink,_mm256_xor_si256(qy0,qy2));
                sumi0=_mm256_add_epi16(sumi0,_mm256_maddubs_epi16(qx,qy1));
                sumi1=_mm256_add_epi16(sumi1,_mm256_maddubs_epi16(qx,qy3));
            }
            sink=_mm256_add_epi16(sink,_mm256_add_epi16(sumi0,sumi1));
        }
    }
    s[0]=(float)_mm256_extract_epi16(sink,0);
}

/* ---------------- X6: VNNI ymm mask-only, four rows in flight. Same per-row integer
   sums and the same per-row float epilogue, so identity is unchanged; only the order
   in which independent rows are interleaved differs. ---------------- */
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
static void k_x6(float *o){
    const __m256i m03=_mm256_set1_epi8(0x03), m0c=_mm256_set1_epi8(0x0c);
    const __m256i m30=_mm256_set1_epi8(0x30), mc0=_mm256_set1_epi8((char)0xc0);
    for (int r=0;r<NROWS;r+=4){
        const block_tq2_0 *xr[4]={&W[r*NB],&W[(r+1)*NB],&W[(r+2)*NB],&W[(r+3)*NB]};
        __m256 sf[4]={_mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps()};
        for (int i=0;i<NB;i++){
            __m256i acc[4][4];
            for (int t=0;t<4;t++) for (int c=0;c<4;c++) acc[t][c]=_mm256_setzero_si256();
            for (size_t j=0;j<64;j+=32){
                const __m256i y0=_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 +  0));
                const __m256i y1=_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 + 32));
                const __m256i y2=_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 + 64));
                const __m256i y3=_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 + 96));
                for (int t=0;t<4;t++){
                    const __m256i w = _mm256_loadu_si256((const __m256i *)(xr[t][i].qs + j));
                    acc[t][0]=_mm256_dpbusd_epi32(acc[t][0],_mm256_and_si256(w,m03),y0);
                    acc[t][1]=_mm256_dpbusd_epi32(acc[t][1],_mm256_and_si256(w,m0c),y1);
                    acc[t][2]=_mm256_dpbusd_epi32(acc[t][2],_mm256_and_si256(w,m30),y2);
                    acc[t][3]=_mm256_dpbusd_epi32(acc[t][3],_mm256_and_si256(w,mc0),y3);
                }
            }
            const __m256i bs=bsum32(&Y[i]);
            for (int t=0;t<4;t++){
                __m256i si=_mm256_add_epi32(_mm256_add_epi32(acc[t][0],_mm256_srai_epi32(acc[t][1],2)),
                                            _mm256_add_epi32(_mm256_srai_epi32(acc[t][2],4),_mm256_srai_epi32(acc[t][3],6)));
                sf[t]=epi_add(sf[t],_mm256_sub_epi32(si,bs),&xr[t][i],&Y[i]);
            }
        }
        for (int t=0;t<4;t++) o[r+t]=hsum_float_8(sf[t]);
    }
}
#endif


/* ---------------- sabotage arm: X3 with the plane-1 fold shifted by 3 instead of 2.
   The comparator MUST report a difference, or "max diff 0" above is not evidence. ---- */
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
static void k_x3_sabotage(float *o){
    const __m256i m03=_mm256_set1_epi8(0x03), m0c=_mm256_set1_epi8(0x0c);
    const __m256i m30=_mm256_set1_epi8(0x30), mc0=_mm256_set1_epi8((char)0xc0);
    for (int r=0;r<NROWS;r++){
        const block_tq2_0 *x=&W[r*NB];
        __m256 sumf=_mm256_setzero_ps();
        for (int i=0;i<NB;i++){
            __m256i a0=_mm256_setzero_si256(),a1=a0,a2=a0,a3=a0;
            for (size_t j=0;j<64;j+=32){
                const __m256i w = _mm256_loadu_si256((const __m256i *)(x[i].qs + j));
                a0=_mm256_dpbusd_epi32(a0,_mm256_and_si256(w,m03),_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 +  0)));
                a1=_mm256_dpbusd_epi32(a1,_mm256_and_si256(w,m0c),_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 + 32)));
                a2=_mm256_dpbusd_epi32(a2,_mm256_and_si256(w,m30),_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 + 64)));
                a3=_mm256_dpbusd_epi32(a3,_mm256_and_si256(w,mc0),_mm256_loadu_si256((const __m256i *)(Y[i].qs + j*4 + 96)));
            }
            __m256i sumi = _mm256_add_epi32(_mm256_add_epi32(a0,_mm256_srai_epi32(a1,3)),
                                            _mm256_add_epi32(_mm256_srai_epi32(a2,4),_mm256_srai_epi32(a3,6)));
            sumi = _mm256_sub_epi32(sumi, bsum32(&Y[i]));
            sumf = epi_add(sumf, sumi, &x[i], &Y[i]);
        }
        o[r]=hsum_float_8(sumf);
    }
}
#endif

/* ---------------- harness ---------------- */
typedef void (*kfn)(float*);
static double bench(kfn f, float *o, int iters, int reps){
    f(o);
    double best=1e18;
    for (int r=0;r<reps;r++){
        double t=now();
        for (int i=0;i<iters;i++) f(o);
        t=now()-t;
        if (t<best) best=t;
    }
    return best/iters;
}
static const double WEIGHTS = (double)NROWS*(double)KDIM;
static void report(const char *name, double t, const char *ident){
    printf("%-34s %9.3f us  %6.4f ns/weight  %7.1f G weights/s  %s\n",
           name, t*1e6, t*1e9/WEIGHTS, WEIGHTS/t/1e9, ident);
}

int main(int argc,char**argv){
    int iters = argc>1?atoi(argv[1]):50;
    int reps  = argc>2?atoi(argv[2]):7;
    const char *realfile = argc>3?argv[3]:NULL;
    W  = aligned_alloc(64, ((size_t)NROWS*NB*sizeof(block_tq2_0)+63)/64*64);
    Y  = aligned_alloc(64, ((size_t)NB*sizeof(block_q8_K)+63)/64*64);
    Wref = malloc((size_t)NROWS*KDIM);
    outA = aligned_alloc(64,NROWS*4); outB = aligned_alloc(64,NROWS*4); outT = aligned_alloc(64,NROWS*4);
    gen();
    if (realfile){
        FILE *f=fopen(realfile,"rb");
        if(!f){ perror(realfile); return 1; }
        size_t want=(size_t)NROWS*NB*sizeof(block_tq2_0);
        if (fread(W,1,want,f)!=want){ fprintf(stderr,"short read of %s\n",realfile); return 1; }
        fclose(f);
        for (int r=0;r<NROWS;r++) for (int b=0;b<NB;b++){
            const block_tq2_0 *x=&W[r*NB+b];
            for (int j=0;j<64;j+=32) for (int m=0;m<32;m++) for (int nn=0;nn<4;nn++)
                Wref[(size_t)r*KDIM+b*QK_K+j*4+m+nn*32]=(int8_t)(((x->qs[j+m]>>(2*nn))&3)-1);
        }
        printf("weights: real TQ2_0 blocks from %s\n", realfile);
    }
    double mb_plain = (double)NROWS*NB*sizeof(block_tq2_0)/1e6;
    printf("matrix %d x %d TQ2_0  codes+scales %.3f MB  (%.0f weights)\n", NROWS, KDIM, mb_plain, WEIGHTS);
    printf("build: AVX2=%d AVX512F=%d AVX512VL=%d AVX512VNNI=%d AVX512BW=%d\n",
#ifdef __AVX2__
        1,
#else
        0,
#endif
#ifdef __AVX512F__
        1,
#else
        0,
#endif
#ifdef __AVX512VL__
        1,
#else
        0,
#endif
#ifdef __AVX512VNNI__
        1,
#else
        0,
#endif
#ifdef __AVX512BW__
        1
#else
        0
#endif
        );

    float *R=malloc(NROWS*4); ref_scalar(R);
    k_avx2(outB);
    double mr=0; for(int r=0;r<NROWS;r++){ double e=fabs(outB[r]-R[r])/(fabs(R[r])+1e-6); if(e>mr)mr=e; }
    printf("shipped AVX2 vec_dot vs scalar ref maxrel %.2e\n", mr);

    report("(a) vec_dot AVX2 (shipped)",  bench(k_avx2,      outT, iters, reps), "baseline");
    report("(c) load only (mem floor)",   bench(k_loadonly,  outT, iters, reps), "-");
    report("(d2) maddubs only (no unpack)",bench(k_maddubs_only,outT,iters,reps), "-");
    report("(d5) unpack only (no maddubs)",bench(k_unpack_only,outT,iters,reps), "-");
    report("(d6) 2 maddubs per 32B codes", bench(k_maddubs_half,outT,iters,reps), "-");
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    report("(d3) vpdpbusd ymm only",      bench(k_vnni_only, outT, iters, reps), "-");
    report("(d4) vpdpbusd zmm only",      bench(k_vnni512_only,outT,iters,reps), "-");
#else
    printf("(d3)/(d4) vpdpbusd floors: skipped, no AVX512VNNI in this build\n");
#endif

    printf("\n-- variants (bitwise identity is against (a), the shipped kernel) --\n");
    double ta = bench(k_avx2,outT,iters,reps);
#define VAR(NAME,FN) do { memset(outT,0,NROWS*4); FN(outT); \
    int id = memcmp(outT,outB,NROWS*4)==0; \
    size_t nd=0; for(int q=0;q<NROWS;q++) if (memcmp(&outT[q],&outB[q],4)!=0) nd++; \
    double t = bench(FN,outT,iters,reps); \
    char buf[96]; snprintf(buf,sizeof buf,"%s  %.2fx vs (a)", id?"max diff 0":"DIFFERS", ta/t); \
    if(!id) snprintf(buf,sizeof buf,"DIFFERS in %zu of %d  %.2fx vs (a)", nd, NROWS, ta/t); \
    report(NAME,t,buf); } while(0)
    VAR("X1 AVX2 mask-only 3 classes", k_x1);
    VAR("X2 AVX2 mask-only 4 classes", k_x2);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    VAR("X3 VNNI ymm mask-only",       k_x3);
    VAR("X4 VNNI zmm broadcast-mask",  k_x4);
    VAR("X5 VNNI ymm, 2 rows",         k_x5);
    VAR("X6 VNNI ymm, 4 rows",         k_x6);
    VAR("SABOTAGE X3 fold shift 3",    k_x3_sabotage);
#endif
    return 0;
}
