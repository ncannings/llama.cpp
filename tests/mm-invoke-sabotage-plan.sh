#!/bin/bash
# mm-invoke sabotage, PLAN gate.
#
# The byte gates catch a sabotage only when the race it opens actually fires. The
# work-buffer region change (GGML_CPU_WBUF_SLOTS) is a change to what the SCHEDULE IS
# ALLOWED TO DO, so the thing to read is the schedule, not the bytes: if the region rule is
# load-bearing on a real graph then changing it MUST change the count of slots that keep a
# full wait. The persistent-schedule record built this gate for its own WAR rule after
# discovering that neither byte gate could fail on it; this is the same gate for the rule
# that replaced its WBUF refusal.
#
# It also carries the two sabotages of tests/psched-sabotage-plan.sh in the form this tree
# needs. That script edits the literal `if (pj->wbuf && ps->wbuf) {`, which no longer
# appears here because the WBUF refusal now also tests the region, so running it unchanged
# against this tree would refuse rather than measure. The psched tree's copy is untouched
# and still gates the parent.
#
#   p1  the WBUF refusal is deleted outright (the psched s3 shape). Deleting it must give
#       the SAME plan as the largest region count, because enough regions is exactly what
#       "no WBUF refusal" means. That equality is the gate on the region change: it says
#       the implementation reaches the bound the sabotage measured and does not exceed it.
#   p2  the WAR refusal is deleted (the psched s4 shape), unchanged in effect
#   p3  the planner's region mapping is shifted by one against the compute loop's, which is
#       a schedule that permits an overlap the buffer does not allow
set -u
T=/home/ncannings/ternary_bench/llama.cpp-mminv
F=$T/ggml/src/ggml-cpu/ggml-cpu.c
M=/home/ncannings/ternary_bench/models/maple-preview-TQ2_0-head-Q4_K.gguf
MB=/home/ncannings/ternary_bench/models/bitnet-2b4t-TQ2_0.gguf
PIN="taskset -c 0-4"
cd $T

git diff --quiet -- $F || { echo "REFUSED: $F already modified"; exit 3; }

plan() { # <model> <slots>
  GGML_CPU_WBUF_SLOTS=$2 GGML_PSCHED_STATS=1 timeout 900 $PIN ./build-dp/bin/llama-bench -m $1 -t 4 -p 0 -n 4 -r 1 2>&1 | grep "^psched:" | tail -1
}

echo "reference plan, by region count"
for S in 1 2 3 4; do
  echo "   slots=$S bitnet $(plan $MB $S)"
  echo "   slots=$S maple  $(plan $M  $S)"
done

for tag in p1_no_wbuf p2_no_war p3_planner_shift; do
    git checkout -- $F
    python3 - "$tag" <<'PY'
import sys, pathlib
tag = sys.argv[1]
f = pathlib.Path("/home/ncannings/ternary_bench/llama.cpp-mminv/ggml/src/ggml-cpu/ggml-cpu.c")
s = f.read_text()
M = {
 "p1_no_wbuf": ("""    if (pj->wbuf && ps->wbuf && pj->wslot == ps->wslot) {
        *why = PSCHED_WHY_WBUF;
        return false;
    }""", """    if (false) {
        *why = PSCHED_WHY_WBUF;
        return false;
    }"""),
 "p2_no_war": ("""        if (!ggml_tail_disjoint(sc, ps->dst)) {
            *why = PSCHED_WHY_WAR;
            return false;
        }""", """        if (false) {
            *why = PSCHED_WHY_WAR;
            return false;
        }"""),
 "p3_planner_shift": ("""        p->wslot = ggml_psched_wslot(g, (cplan->work_region > 0 && cplan->work_slots > 1) ? cplan->work_slots : 1);""",
                      """        p->wslot = ggml_psched_wslot(g + 1, (cplan->work_region > 0 && cplan->work_slots > 1) ? cplan->work_slots : 1);"""),
}
a, b = M[tag]
assert s.count(a) == 1, (tag, s.count(a))
f.write_text(s.replace(a, b, 1))
PY
    echo "=== SABOTAGE $tag"
    $PIN cmake --build build-dp -j4 --target llama-bench > /tmp/mminv_sabplan_build.log 2>&1 || { echo "   build FAILED"; continue; }
    for S in 1 4; do
      echo "   slots=$S bitnet $(plan $MB $S)"
      echo "   slots=$S maple  $(plan $M  $S)"
    done
done

git checkout -- $F
$PIN cmake --build build-dp -j4 --target llama-bench > /dev/null 2>&1
echo "=== tree restored: $(git diff --numstat -- $F | wc -l) modified files"
