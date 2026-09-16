#!/bin/bash
# psched sabotage: one edit each, applied alone to the clean tree, rebuilt, and put in front
# of the gates that are supposed to catch it. A gate that cannot fail is not evidence.
#
#   s1 the directed wait never waits at all
#   s2 a thread publishes its progress one slot EARLY (it claims a slot it has not finished)
#   s3 the shared-work-buffer rule is gone, so two matmuls may overlap (the 5b shape)
#   s4 the write-after-read rule is gone, so a slot may overwrite a buffer an earlier slot reads
#   s5 the wait checks thread 0 only, dropping every other thread's row completions
#
# Gates: tests/test-moe-tail-identity (the Maple decoder block, every node, every switch
# combination) and examples/moe-tail-hash on the real Maple model (every tensor of every
# graph, real graph allocator). Both are reported, caught or not.
set -u
T=/home/ncannings/ternary_bench/llama.cpp-psched
F=$T/ggml/src/ggml-cpu/ggml-cpu.c
M=/home/ncannings/ternary_bench/models/maple-preview-TQ2_0-head-Q4_K.gguf
PIN="taskset -c 0-4"
cd $T

git diff --quiet -- $F || { echo "REFUSED: $F already modified"; exit 3; }

ref=$(mktemp)
$PIN ./build-dp/bin/llama-moe-tail-hash -m $M -p "The capital of France is" -t 4 -ngl 0 > $ref 2>/dev/null
echo "reference hash: $(wc -l < $ref) lines"

apply() { # <tag>
python3 - "$1" <<'PY'
import sys, pathlib
tag = sys.argv[1]
f = pathlib.Path("/home/ncannings/ternary_bench/llama.cpp-psched/ggml/src/ggml-cpu/ggml-cpu.c")
s = f.read_text()
if tag == "s1_wait_never":
    a = """static inline void ggml_psched_wait(struct ggml_threadpool * tp, int w, int nth) {
    if (w <= 0) {
        return;
    }"""
    b = """static inline void ggml_psched_wait(struct ggml_threadpool * tp, int w, int nth) {
    if (w >= 0) {
        return;
    }"""
elif tag == "s2_publish_early":
    a = "            ggml_psched_publish(tp, state->ith, psched_seq);"
    b = "            ggml_psched_publish(tp, state->ith, psched_seq + 1);"
elif tag == "s3_no_wbuf":
    a = """    if (pj->wbuf && ps->wbuf) {
        *why = PSCHED_WHY_WBUF;
        return false;
    }"""
    b = """    if (false) {
        *why = PSCHED_WHY_WBUF;
        return false;
    }"""
elif tag == "s4_no_war":
    a = """        if (!ggml_tail_disjoint(sc, ps->dst)) {
            *why = PSCHED_WHY_WAR;
            return false;
        }"""
    b = """        if (false) {
            *why = PSCHED_WHY_WAR;
            return false;
        }"""
elif tag == "s5_wait_thread0":
    a = "    for (int u = 0; u < nth; u++) {\n        while (atomic_load_explicit(&tp->psched_prog[u].v, memory_order_acquire) < w) {"
    b = "    for (int u = 0; u < 1; u++) {\n        while (atomic_load_explicit(&tp->psched_prog[u].v, memory_order_acquire) < w) {"
else:
    raise SystemExit("unknown tag " + tag)
assert s.count(a) == 1, ("anchor not unique for " + tag, s.count(a))
f.write_text(s.replace(a, b))
PY
}

for tag in s1_wait_never s2_publish_early s3_no_wbuf s4_no_war s5_wait_thread0; do
    git checkout -- $F
    apply $tag || { echo "=== SABOTAGE $tag: COULD NOT APPLY"; continue; }
    if ! $PIN cmake --build build-dp -j4 --target test-moe-tail-identity llama-moe-tail-hash > /tmp/psched_sab_build.log 2>&1; then
        echo "=== SABOTAGE $tag: BUILD FAILED"; tail -5 /tmp/psched_sab_build.log; continue
    fi
    echo "=== SABOTAGE $tag"
    for rep in 1 2 3; do
        out=$($PIN ./build-dp/bin/test-moe-tail-identity 2>&1); rc=$?
        v=$(echo "$out" | grep -E "test-moe-tail-identity: (PASS|FAIL)" | tail -1)
        first=$(echo "$out" | grep "DIFFERS" | head -1)
        echo "   rep $rep unit: exit=$rc ${v:-<no verdict>} ${first}"
        got=$(mktemp)
        $PIN ./build-dp/bin/llama-moe-tail-hash -m $M -p "The capital of France is" -t 4 -ngl 0 > $got 2>/dev/null
        if diff -q $ref $got >/dev/null; then echo "   rep $rep hash: IDENTICAL (NOT caught)"
        else echo "   rep $rep hash: DIFFERS ($(diff $ref $got | grep -c '^<') lines) CAUGHT"; fi
        rm -f $got
    done
done

git checkout -- $F
$PIN cmake --build build-dp -j4 --target test-moe-tail-identity llama-moe-tail-hash > /dev/null 2>&1
echo "=== tree restored: $(git diff --stat -- $F | wc -l) modified lines in ggml-cpu.c"
rm -f $ref
