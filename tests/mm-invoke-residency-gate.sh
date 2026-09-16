#!/bin/bash
# mm-invoke residency gate: ASSERT the routing, do not infer it from a timing.
#
# The static single-row split is taken only while the graph's whole matmul weight working
# set fits GGML_CPU_MM_STATIC_MAX_MB. Both sides of that test are the same arithmetic and
# are proven bit-identical by test-mm-invoke-identity, so NO BYTE GATE CAN FAIL on the
# threshold: get it backwards and the output is still correct, just slower on exactly the
# rows job 103 measured. What can fail is the decision itself, and GGML_CPU_MM_STATS prints
# it. This script states the expected routing for four models and two thresholds, and it
# refuses if any of the eight disagree.
#
# Two sabotages, both applied to the ONE function the rule lives in: the comparison
# reversed, and the threshold removed altogether. Both are invisible to every other gate in
# this tree, which is why this one exists.
set -u
T=/home/ncannings/ternary_bench/llama.cpp-mminv
F=$T/ggml/src/ggml-cpu/repack.cpp
C=$T/ggml/src/ggml-cpu/ggml-cpu.c
B=/home/ncannings/ternary_bench
PIN="taskset -c 0-4"
cd $T
git diff --quiet -- $F $C || { echo "REFUSED: the tree is already modified"; exit 3; }

# model, threshold MB, expected decision
CASES="synth/synth-L1-TQ2_0:16:yes synth/synth-L2-TQ2_0:16:yes synth/synth-L16-TQ2_0:16:no synth/synth-L16-TQ2_0:1024:yes"
CASES="$CASES ../bitnet-2b4t-TQ2_0:16:no ../bitnet-2b4t-TQ2_0:1024:yes synth/synth-L1-TQ2_0:1:no synth/synth-L2-TQ2_0:14:no"

check() { # prints one line per case, returns the number that disagreed
  local bad=0
  for c in $CASES; do
    local mdl=${c%%:*} rest=${c#*:} mb exp got
    mb=${rest%%:*}; exp=${rest#*:}
    got=$(GGML_CPU_MM_STATIC_MAX_MB=$mb GGML_CPU_MM_STATS=1 timeout 900 $PIN ./build-dp/bin/llama-bench \
            -m $B/models/$mdl.gguf -ngl 0 -t 4 -p 0 -n 1 -r 1 2>&1 | grep "^mm:" | tail -1)
    local dec=$(echo "$got" | sed -E 's/.*static1_resident=([a-z]+).*/\1/')
    if [ "$dec" = "$exp" ]; then
      echo "   OK       $(basename $mdl) at ${mb} MB -> $dec   [$got]"
    else
      echo "   DISAGREE $(basename $mdl) at ${mb} MB -> want $exp got '$dec'   [$got]"
      bad=$((bad+1))
    fi
  done
  return $bad
}

echo "== reference routing"
check; ref_bad=$?
echo "   reference disagreements: $ref_bad"

for tag in r1_invert r2_no_threshold; do
  echo "=== SABOTAGE $tag"
  python3 - "$tag" <<'PY'
import sys, pathlib
tag = sys.argv[1]
f = pathlib.Path("/home/ncannings/ternary_bench/llama.cpp-mminv/ggml/src/ggml-cpu/ggml-cpu.c")
s = f.read_text()
M = {
 "r1_invert":      ("    return mm_weight_bytes > 0 && mm_weight_bytes <= ggml_mm_static_max_bytes();",
                    "    return mm_weight_bytes > 0 && mm_weight_bytes >= ggml_mm_static_max_bytes();"),
 "r2_no_threshold":("    return mm_weight_bytes > 0 && mm_weight_bytes <= ggml_mm_static_max_bytes();",
                    "    return mm_weight_bytes > 0;"),
}
a, b = M[tag]
assert s.count(a) == 1, (tag, s.count(a))
f.write_text(s.replace(a, b, 1))
PY
  $PIN cmake --build build-dp -j4 --target llama-bench > /tmp/mminv_resgate_build.log 2>&1 || { echo "   build FAILED"; git checkout -- $C; continue; }
  check; bad=$?
  if [ $bad -gt 0 ]; then echo "   routing gate: $bad of 8 cases CHANGED  CAUGHT"; else echo "   routing gate: unchanged  NOT caught"; fi
  git checkout -- $C
done

$PIN cmake --build build-dp -j4 --target llama-bench > /dev/null 2>&1
echo "=== tree restored: $(git diff --numstat -- $C $F | wc -l) modified files"
[ $ref_bad -eq 0 ] && echo "mm-invoke-residency-gate: PASS" || echo "mm-invoke-residency-gate: FAIL"
exit $ref_bad
