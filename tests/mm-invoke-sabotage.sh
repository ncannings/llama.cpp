#!/bin/bash
# mm-invoke sabotage, BYTE gate.
#
# One gate per change. Each sabotage is a SINGLE source edit applied alone to the clean
# tree, rebuilt, and put in front of the gate that is supposed to catch it. A sabotage that
# nothing catches means the gate cannot fail and the change is not proven, which is the
# finding the persistent-schedule record made about its WAR rule.
#
#   m1  the block split rounds the block boundary instead of cutting on it, so one thread
#       quantises into the middle of another thread's block
#   m2  the static row range drops the NB_COLS rounding, so a thread's range starts inside
#       a repacked group
#   m3  the static row split leaves a hole: the last thread's range stops one group short
#   m4  the dispatch shortcut visits the extra buffer types in reverse order
#   m5  the work-buffer region mapping in the COMPUTE LOOP and the one in the PLANNER
#       disagree, which is the shape that makes the schedule permit an overlap the buffer
#       does not allow
#
# m1 to m3 must be caught by test-mm-invoke-identity. m4 is a no-op on a build with one
# extra buffer type and is expected NOT to be caught: it is included so that the gate's
# silence there is on the record rather than inferred. m5 is a scheduling hazard whose byte
# gate is the whole-tensor hash of the real model; the plan gate is tests/mm-invoke-sabotage-plan.sh.
set -u
T=/home/ncannings/ternary_bench/llama.cpp-mminv
R=$T/ggml/src/ggml-cpu/repack.cpp
C=$T/ggml/src/ggml-cpu/ggml-cpu.c
TR=$T/ggml/src/ggml-cpu/traits.cpp
PIN="taskset -c 0-4"
REPS=${REPS:-3}
cd $T

git diff --quiet -- $R $C $TR || { echo "REFUSED: tree already modified"; exit 3; }

build() { $PIN cmake --build build-dp -j4 --target test-mm-invoke-identity llama-moe-tail-hash > /tmp/mminv_sab_build.log 2>&1; }

unit() { timeout 1800 $PIN ./build-dp/bin/test-mm-invoke-identity 2>&1 | grep -vE "^repack:" | grep -E "DIFFERS|: (PASS|FAIL)" | head -2; }

echo "== reference"
build || { echo "reference build failed"; exit 3; }
unit

for tag in m1_block_round m2_no_align m3_row_hole m4_reverse_dispatch m5_planner_mismatch; do
    git checkout -- $R $C $TR
    python3 - "$tag" <<'PY'
import sys, pathlib
tag = sys.argv[1]
T = pathlib.Path("/home/ncannings/ternary_bench/llama.cpp-mminv/ggml/src/ggml-cpu")
M = {
 "m1_block_round": ("repack.cpp",
   """            const int64_t b0   = ((int64_t) ith       * nblk) / nth;""",
   """            const int64_t b0   = ((int64_t) ith       * nblk + nth/2) / nth;"""),
 "m2_no_align": ("repack.cpp",
   """            src0_start = (src0_start % NB_COLS) ? src0_start + NB_COLS - (src0_start % NB_COLS) : src0_start;
            src0_end   = (src0_end   % NB_COLS) ? src0_end   + NB_COLS - (src0_end   % NB_COLS) : src0_end;
            src0_end   = MIN(src0_end, ne01);""",
   """            src0_end   = MIN(src0_end, ne01);"""),
 "m3_row_hole": ("repack.cpp",
   """            src0_end   = MIN(src0_end, ne01);

            ggml_barrier(params->threadpool);""",
   """            src0_end   = MIN(src0_end, ne01 - (ith == nth - 1 ? NB_COLS : 0));

            ggml_barrier(params->threadpool);"""),
 "m4_reverse_dispatch": ("traits.cpp",
   """        for (int i = 0; i < g_extra_n; i++) {""",
   """        for (int i = g_extra_n - 1; i >= 0; i--) {"""),
 "m5_planner_mismatch": ("ggml-cpu.c",
   """        p->wslot = ggml_psched_wslot(g, (cplan->work_region > 0 && cplan->work_slots > 1) ? cplan->work_slots : 1);""",
   """        p->wslot = ggml_psched_wslot(g + 1, (cplan->work_region > 0 && cplan->work_slots > 1) ? cplan->work_slots : 1);"""),
}
f, a, b = M[tag]
p = T / f
s = p.read_text()
assert s.count(a) == 1, (tag, f, s.count(a))
p.write_text(s.replace(a, b, 1))
print("   applied", tag, "to", f)
PY
    echo "=== SABOTAGE $tag"
    build || { echo "   build FAILED (counts as caught only if it is a compile error you meant)"; continue; }
    for r in $(seq 1 $REPS); do
        echo "   rep $r unit: $(unit | tr '\n' ' ')"
    done
done

git checkout -- $R $C $TR
echo "=== tree restored: $(git diff --numstat -- $R $C $TR | wc -l) modified files"
build > /dev/null 2>&1
