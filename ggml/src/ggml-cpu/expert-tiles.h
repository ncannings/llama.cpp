#pragma once

// expert-tiles: pin each MoE expert index to one L3 domain, and compute that expert's
// matmuls only on the cores of that domain.
//
// WHAT THIS IS FOR. On the GB10 the two L3 domains are 8 MB (cpus 0-9, big cores 5-9)
// and 16 MB (cpus 10-19, big cores 15-19). A cache-resident TQ2_0 weight runs 1.4x
// faster and 1.4x more energy-efficient per weight than one streamed from DRAM
// (2026-09-14-cache-residency-sweep-spark.md). With the expert tensors dealt out over
// the whole pool, every expert's bytes can be pulled into BOTH L3s, so the aggregate
// distinct cache the expert store sees is one domain's worth, not two. Pinning expert e
// to domain d means e's bytes are only ever filled into d's L3.
//
// WHAT IT DOES NOT DO. Only GGML_OP_MUL_MAT_ID is placed. Attention, the router, the
// head and every dense MUL_MAT keep the whole pool. There is no second threadpool and
// no second barrier: the pool-wide barrier across the ten big cores is 0.5 us, which is
// the measurement the blueprint's argument rests on, so the placement is a work
// ASSIGNMENT inside the existing pool, with stealing confined to the domain.
//
// BIT-IDENTICAL BY CONSTRUCTION. The placement changes only WHICH thread computes a
// piece of work and, in the column split, where the column boundaries fall. Every
// output element is still produced by exactly one kernel call reducing over the full
// ne00, from the same inputs, so the bytes written are the same at every thread count
// and under every map.
//
// SWITCHES
//   GGML_CPU_EXPERT_TILES=1        turn the placement on (default OFF)
//   GGML_CPU_EXPERT_TILES=2        PIN ONLY: build and apply exactly the same plan, one
//                                  thread per cpu, but deal the experts pool-wide as
//                                  before. This is the control the timing job needs: it
//                                  separates what the per-thread pinning costs or buys
//                                  from what the expert placement costs or buys, which a
//                                  comparison against an unpinned arm cannot.
//   GGML_CPU_EXPERT_TILE_MAP=...   the expert-to-domain map. Either a comma separated
//                                  list of domain indices, one per expert, applied
//                                  modulo its length ("0,1" interleaves, "0,0,0,0,1,1,1,1"
//                                  blocks 8 experts over 2 domains, "1" sends every
//                                  expert to domain 1), or "block" (the default: expert
//                                  e goes to domain e*ndom/n_expert, an equal split that
//                                  matches the equal thread split), or "cyclic"
//                                  (e % ndom).
//   GGML_CPU_SYSFS_ROOT=<dir>      read the topology from <dir> instead of
//                                  /sys/devices/system/cpu. For tests only: it lets the
//                                  two-domain partition be exercised on the little cores.
//   GGML_CPU_EXPERT_TILES_VERBOSE=1 print the placement once to stderr.
//
// NO SILENT FALLBACK. If the topology cannot be read, if a thread cannot be given a
// distinct cpu, or if any domain would end up with no threads (which would leave that
// domain's experts uncomputed), the planner ABORTS. It never quietly reverts to the
// unplaced deal, because a run that silently reverted would be recorded as a measurement
// of the placement.

#include <stdbool.h>

#define GGML_EXPERT_TILES_MAX_DOMAINS 8

#ifdef __cplusplus
extern "C" {
#endif

// Is the plan built and applied at all (mode 1 or 2)? Cached env read, safe from any thread.
bool ggml_expert_tiles_enabled(void);

// Is the EXPERT PARTITION on (mode 1 only)? Mode 2 pins without partitioning.
bool ggml_expert_tiles_placed(void);

// Build the placement for a pool of n_threads. MAIN THREAD ONLY, before the workers
// start on a graph. Idempotent: replanning for the same n_threads is a no-op. Aborts
// rather than falling back.
void ggml_expert_tiles_plan(int n_threads);

// Apply thread ith's pin. Called by each worker, after the plan. Cheap and idempotent.
void ggml_expert_tiles_apply(int ith);

// The thread count the current plan was built for. A caller whose pool differs from this
// is looking at a plan that does not describe it, and must refuse rather than use it.
int ggml_expert_tiles_planned_for(void);

int ggml_expert_tiles_n_domains(void);      // number of L3 domains in the pool
int ggml_expert_tiles_thread_domain(int ith);  // domain of thread ith
int ggml_expert_tiles_domain_nth(int dom);     // threads in that domain
int ggml_expert_tiles_domain_rank(int ith);    // rank of ith among its domain's threads

// Which domain owns expert index `expert` of `n_expert`. Pure function of the map.
int ggml_expert_tiles_expert_domain(int expert, int n_expert);

#ifdef __cplusplus
}
#endif
