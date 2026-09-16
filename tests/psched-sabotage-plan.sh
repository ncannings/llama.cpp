#!/bin/bash
# psched sabotage, PLAN gate.
#
# tests/psched-sabotage.sh puts each sabotage in front of two BYTE gates: the Maple decoder
# block unit test and the whole-tensor hash of the real model. Those gates catch a sabotage
# only when the race it opens actually fires, and one of the five (s4, the write-after-read
# rule) opens a hazard that the unit test cannot reach at all, because that test gives every
# node its own buffer and therefore never makes the graph allocator hand one slot the buffer
# an earlier slot is still reading. A rule with no gate that can fail on it is not proven.
#
# This gate reads the PLAN instead of the bytes. GGML_PSCHED_STATS counts, per graph, how
# many slots keep a full wait, how many skip back further and how many take none. If a rule
# is load-bearing on a real graph then deleting it MUST change those counts. If deleting it
# changes nothing, the rule is inert on this graph and this gate says so rather than
# pretending a passing byte comparison was evidence.
#
# It is a weaker claim than a byte difference and a stronger one than silence: it says the
# rule decides something here, and it can fail.
set -u
T=/home/ncannings/ternary_bench/llama.cpp-psched
F=$T/ggml/src/ggml-cpu/ggml-cpu.c
M=/home/ncannings/ternary_bench/models/maple-preview-TQ2_0-head-Q4_K.gguf
MB=/home/ncannings/ternary_bench/models/bitnet-2b4t-TQ2_0.gguf
PIN="taskset -c 0-4"
cd $T

git diff --quiet -- $F || { echo "REFUSED: $F already modified"; exit 3; }

plan() { # <model>
  GGML_PSCHED_STATS=1 timeout 600 $PIN ./build-dp/bin/llama-bench -m $1 -t 4 -p 0 -n 4 -r 1 2>&1 | grep "^psched:" | tail -1
}

echo "reference plan"
REF_M=$(plan $M); echo "   maple  $REF_M"
REF_B=$(plan $MB); echo "   bitnet $REF_B"

for tag in s3_no_wbuf s4_no_war; do
    git checkout -- $F
    python3 - "$tag" <<'PY'
import sys, pathlib
tag = sys.argv[1]
f = pathlib.Path("/home/ncannings/ternary_bench/llama.cpp-psched/ggml/src/ggml-cpu/ggml-cpu.c")
s = f.read_text()
M = {
 "s3_no_wbuf": ("""    if (pj->wbuf && ps->wbuf) {
        *why = PSCHED_WHY_WBUF;
        return false;
    }""", """    if (false) {
        *why = PSCHED_WHY_WBUF;
        return false;
    }"""),
 "s4_no_war": ("""        if (!ggml_tail_disjoint(sc, ps->dst)) {
            *why = PSCHED_WHY_WAR;
            return false;
        }""", """        if (false) {
            *why = PSCHED_WHY_WAR;
            return false;
        }"""),
}
a, b = M[tag]
assert s.count(a) == 1, ("anchor not unique for " + tag, s.count(a))
f.write_text(s.replace(a, b))
PY
    $PIN cmake --build build-dp -j4 --target llama-bench > /tmp/psched_sabplan_build.log 2>&1 || { echo "=== $tag BUILD FAILED"; continue; }
    GOT_M=$(plan $M); GOT_B=$(plan $MB)
    echo "=== SABOTAGE $tag"
    echo "   maple  $GOT_M"
    echo "   bitnet $GOT_B"
    if [ "$GOT_M" = "$REF_M" ] && [ "$GOT_B" = "$REF_B" ]; then
        echo "   plan gate: UNCHANGED (rule inert on both graphs, NOT caught)"
    else
        echo "   plan gate: CHANGED  CAUGHT"
    fi
done

git checkout -- $F
$PIN cmake --build build-dp -j4 --target llama-bench > /dev/null 2>&1
echo "=== tree restored: $(git diff --stat -- $F | wc -l) modified lines in ggml-cpu.c"
