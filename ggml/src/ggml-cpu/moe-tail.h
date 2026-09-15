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

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void ggml_tail_init(void);

bool ggml_tail_rows_flatten(void);
bool ggml_tail_bcast_scalar(void);
bool ggml_tail_sumrows_par(void);
bool ggml_tail_barrier_run(void);
int  ggml_tail_mm_chunks(void);

#ifdef __cplusplus
}
#endif
