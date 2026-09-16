#pragma once

// moe-tail: work-partition and barrier-elision switches.
//
// Every switch here changes only WHICH THREAD does a piece of work, or WHICH
// BARRIER a thread waits at. None of them changes the arithmetic of a row, the
// order of an accumulation, or the order of a dot product, so every one of them
// is bit-identical by construction and the switches exist to isolate a timing
// claim, not to trade accuracy.
//
//   GGML_CPU_NO_ROWS_FLATTEN=1   restore the per-plane RMS_NORM row partition
//   GGML_CPU_NO_BCAST_SCALAR=1   restore the one-element-at-a-time broadcast in binary ops
//   GGML_CPU_NO_SUMROWS_PAR=1    restore single-threaded SUM_ROWS
//   GGML_CPU_NO_BARRIER_RUN=1    restore one pool-wide barrier after every node
//   GGML_CPU_MM_CHUNKS=<n>       chunks per thread in the repack matmul (default 4)
//
// mm-invoke: per-invocation cost inside the repack forward_mul_mat.
//
//   GGML_CPU_NO_MM_PARQUANT=1    restore the serial src1 quantisation at ne11 == 1
//   GGML_CPU_NO_MM_STATIC1=1     restore the work-stealing chunk loop at ne11 == 1
//   GGML_CPU_MM_STATIC_MAX_MB=<n> the static split applies only while the graph's whole
//                                repacked weight working set is <= n MB (default 16;
//                                0 removes the threshold entirely)
//   GGML_CPU_NO_MM_DISPATCH=1    restore the std::vector walk in the extra-buffer dispatch
//   GGML_CPU_WBUF_SLOTS=<n>      work-buffer regions in the cplan (1 = the single shared
//                                buffer, the upstream behaviour; default 1)

#include <stdbool.h>
#include <stddef.h>

// Maximum number of work-buffer regions a cplan may carry (GGML_CPU_WBUF_SLOTS).
#define GGML_WBUF_SLOTS_MAX 8

#ifdef __cplusplus
extern "C" {
#endif

void ggml_tail_init(void);

bool ggml_tail_rows_flatten(void);
bool ggml_tail_bcast_scalar(void);
bool ggml_tail_sumrows_par(void);
bool ggml_tail_barrier_run(void);

// persistent-sched: per-node dependency tracking in place of the pool-wide
// barrier. Disable with GGML_CPU_NO_PSCHED=1.
bool ggml_psched_enabled(void);
int  ggml_tail_mm_chunks(void);

// mm-invoke switches. Each changes only which thread does a piece of work or
// where a scratch byte lives, never an arithmetic result.
bool ggml_mm_parquant(void);
bool ggml_mm_static1(void);
size_t ggml_mm_static_max_bytes(void);

// The residency test itself, in ONE place. The matmul's gate and the GGML_CPU_MM_STATS
// line both call this and nothing else, so a change to the rule cannot move one without
// moving the other and tests/mm-invoke-residency-gate.sh can fail on it.
bool ggml_mm_static_resident(size_t mm_weight_bytes);
bool ggml_mm_dispatch(void);
int  ggml_wbuf_slots(void);

// Fill the flat extra-buffer dispatch cache. Called from ggml_graph_compute, before any
// worker thread exists.
void ggml_cpu_extra_cache_init(void);

#ifdef __cplusplus
}
#endif
