#pragma once

// Shared VNNI helper for the x86 TQ2_0 kernels (arch/x86/quants.c and arch/x86/repack.cpp).
//
// Every x86 TQ2_0 kernel added on top of the shipped AVX2 vec_dot rests on one instruction,
// vpdpbusd: it takes the code operand UNSIGNED and accumulates four byte products into int32.
// Two consequences carry the whole design.
//
//   1. A 2-bit plane can be selected with a mask alone (w & 0x03, w & 0x0c, w & 0x30,
//      w & 0xc0) with no shift, because an unsigned operand of 4^sh times the code is legal
//      and the accumulated int32 comes out 4^sh times the true sum. One exact arithmetic
//      right shift per plane per block folds it back. An arithmetic shift of an exact
//      multiple of 4^sh is exact for negative values too, so this is equality, not tolerance.
//   2. Nothing can overflow: the 16-bit accumulators of the AVX2 form (and the argument that
//      256 * 127 still fits) disappear, and so do the four int16 accumulate-adds, which are
//      absorbed into the instruction.
//
// The kernels are compiled in only when the compiler is targeting a VNNI-capable level.
// GGML_TQ2_X86_VNNI is the single compile-time predicate; the callers pair it with the
// run-time ggml_cpu_has_avx512_vnni() / ggml_cpu_has_avx_vnni() check.

#if defined(__AVX2__) && ((defined(__AVX512VNNI__) && defined(__AVX512VL__)) || defined(__AVXVNNI__))
#define GGML_TQ2_X86_VNNI 1

static inline __m256i ggml_tq2_dpbusd(__m256i acc, __m256i u, __m256i s) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_dpbusd_epi32(acc, u, s);
#else
    return _mm256_dpbusd_avx_epi32(acc, u, s);
#endif
}

// A 32-bit activation group broadcast into all eight int32 lanes. In the repacked layouts
// the eight lanes are eight different weight rows sharing one activation dword, which is
// what gives the multi-row form its name: one activation load feeds eight rows.
static inline __m256i ggml_tq2_bcast32(const void * p) {
    int32_t v;
    memcpy(&v, p, sizeof(v));
    return _mm256_set1_epi32(v);
}

#endif // VNNI
