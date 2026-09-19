#!/bin/bash
# expert-tiles: the kernel-level identity gate for the expert-to-L3-domain placement.
#
# WHAT IT PROVES. test-repack-mul-mat-id compares the repacked MUL_MAT_ID path against the
# un-repacked reference path and demands, for TQ2_0, finite, byte-identical output. This script runs
# that comparison with the placement OFF and then ON under four different expert maps, and
# additionally requires the FNV-1a hash of every output tensor to be identical across all
# six runs. So the placement is checked twice over: against the reference arithmetic, and
# byte for byte against the unplaced deal.
#
# WHY A FAKE TOPOLOGY. The interesting partition needs at least two L3 domains, and on this
# box the two real domains are reached only by the big cores, which correctness work is not
# allowed to occupy. GGML_CPU_SYSFS_ROOT lets the placement read a topology that splits the
# little cores 0-4 into two domains, so the two-domain code path runs on the cores it is
# allowed to run on. The placement logic under test is the real one; only the map of which
# cpu shares which cache is substituted.
#
# The placement pins one thread per cpu, so the thread list is 1..5 rather than the default
# 1,4,10: ten threads on a five-cpu mask is exactly the condition the planner refuses.
set -u
ulimit -c 0
BIN=${1:-}
[ -x "$BIN/test-repack-mul-mat-id" ] || { echo "usage: expert-tiles-identity.sh <bin-dir>"; exit 2; }
[ -f "$HOME/.spark_window_open" ] && [ ! -f "$HOME/.gpu_window_open" ] && { echo "REFUSED: timing window open"; exit 2; }

OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

# a two-domain topology over cpus 0-4: {0,1} share one last-level cache, {2,3,4} another
ROOT=$OUT/sysfs
for c in 0 1 2 3 4; do
  d=$ROOT/cpu$c/cache/index3
  mkdir -p "$d"
  echo 3 > "$d/level"
  echo Unified > "$d/type"
  if [ "$c" -le 1 ]; then echo "0-1" > "$d/shared_cpu_list"; echo "8192K" > "$d/size"
  else                    echo "2-4" > "$d/shared_cpu_list"; echo "16384K" > "$d/size"; fi
done

run() { # run <label> <env...>
  local label=$1; shift
  env "$@" GGML_TEST_THREADS=1,2,3,4,5,2,1 GGML_TEST_HASH=1 GGML_TEST_REQUIRE_TQ2=1 \
      taskset -c 0-4 "$BIN/test-repack-mul-mat-id" > "$OUT/$label.txt" 2>"$OUT/$label.err"
  local rc=$?
  if [ $rc -ne 0 ]; then echo "   $label: EXIT $rc"; sed -n '1,5p' "$OUT/$label.err"; return 1; fi
  if grep -q FAIL "$OUT/$label.txt"; then echo "   $label: FAIL lines present"; grep FAIL "$OUT/$label.txt" | head -5; return 1; fi
  grep -q '^TQ2_0 cases=1001 expected=1001$' "$OUT/$label.txt" || { echo "   $label: missing TQ2_0 coverage"; return 1; }
  echo "   $label: $(grep -c ' OK' "$OUT/$label.txt") cases OK, $(grep -c 'max diff=0.000e+00' "$OUT/$label.txt") at max diff exactly 0"
  return 0
}

fail=0
echo "== expert-tiles kernel identity"
run off GGML_CPU_EXPERT_TILES=0 || fail=1
run pin_only GGML_CPU_EXPERT_TILES=2 GGML_CPU_SYSFS_ROOT=$ROOT || fail=1
run on_block  GGML_CPU_EXPERT_TILES=1 GGML_CPU_SYSFS_ROOT=$ROOT GGML_CPU_EXPERT_TILE_MAP=block  || fail=1
run on_cyclic GGML_CPU_EXPERT_TILES=1 GGML_CPU_SYSFS_ROOT=$ROOT GGML_CPU_EXPERT_TILE_MAP=cyclic || fail=1
run on_list   GGML_CPU_EXPERT_TILES=1 GGML_CPU_SYSFS_ROOT=$ROOT GGML_CPU_EXPERT_TILE_MAP=0,1,1,0 || fail=1
run on_one    GGML_CPU_EXPERT_TILES=1 GGML_CPU_SYSFS_ROOT=$ROOT GGML_CPU_EXPERT_TILE_MAP=1       || fail=1

echo "== byte identity against the unplaced deal"
for a in pin_only on_block on_cyclic on_list on_one; do
  if [ -s "$OUT/$a.txt" ] && diff -q "$OUT/off.txt" "$OUT/$a.txt" > /dev/null; then
    echo "   $a: IDENTICAL to off ($(grep -c fnv "$OUT/$a.txt") hashes)"
  else
    echo "   $a: DIFFERS from off"; diff "$OUT/off.txt" "$OUT/$a.txt" | head -10; fail=1
  fi
done

# The refusals must be reachable, or the gate above passed on a path that can never fail.
echo "== refusals"
refuse() { # refuse <what> <env...>
  local what=$1 expected=$2; shift 2
  if env "$@" GGML_TEST_THREADS=$THREADS taskset -c 0-4 "$BIN/test-repack-mul-mat-id" > /dev/null 2>"$OUT/r.err"; then
    echo "   $what: DID NOT REFUSE"; return 1
  fi
  grep -qF "$expected" "$OUT/r.err" || { echo "   $what: failed for the wrong reason"; cat "$OUT/r.err"; return 1; }
  echo "   $what: refused ($(grep -oiE 'expert-tiles:[^\"]*' "$OUT/r.err" | head -1 | cut -c1-90))"
  return 0
}
THREADS=10 refuse "10 threads on a 5-cpu mask" "threads but only" GGML_CPU_EXPERT_TILES=1 GGML_CPU_SYSFS_ROOT=$ROOT || fail=1
THREADS=4  refuse "unreadable topology" "no cache topology"        GGML_CPU_EXPERT_TILES=1 GGML_CPU_SYSFS_ROOT=$OUT/nothing-here || fail=1
THREADS=4  refuse "map beyond the domains" "map domain"     GGML_CPU_EXPERT_TILES=1 GGML_CPU_SYSFS_ROOT=$ROOT GGML_CPU_EXPERT_TILE_MAP=0,5 || fail=1
# The pool shrinking under a plan built for more threads is the one failure that would lose
# expert work rather than slow it down, so it gets its own reachable case: OMP_THREAD_LIMIT
# is the real mechanism that produces it.
THREADS=4  refuse "pool smaller than the plan" "pool is 3 threads but the plan was built for 4"  OMP_THREAD_LIMIT=3 GGML_CPU_EXPERT_TILES=1 GGML_CPU_SYSFS_ROOT=$ROOT || fail=1

THREADS=4 refuse "malformed map" "parsed to an empty map" GGML_CPU_EXPERT_TILES=1 GGML_CPU_SYSFS_ROOT=$ROOT GGML_CPU_EXPERT_TILE_MAP=0,1garbage || fail=1
cp -r "$ROOT" "$OUT/l1"
for c in 0 1 2 3 4; do echo 1 > "$OUT/l1/cpu$c/cache/index3/level"; done
THREADS=4 refuse "L1 is not LLC" "no cache topology" GGML_CPU_EXPERT_TILES=1 GGML_CPU_SYSFS_ROOT=$OUT/l1 || fail=1
cp -r "$ROOT" "$OUT/zero"
for c in 0 1 2 3 4; do echo 0 > "$OUT/zero/cpu$c/cache/index3/size"; done
THREADS=4 refuse "zero-size cache" "no cache topology" GGML_CPU_EXPERT_TILES=1 GGML_CPU_SYSFS_ROOT=$OUT/zero || fail=1
cp -r "$ROOT" "$OUT/no-type"
for c in 0 1 2 3 4; do rm "$OUT/no-type/cpu$c/cache/index3/type"; done
THREADS=4 refuse "missing cache type" "no cache topology" GGML_CPU_EXPERT_TILES=1 GGML_CPU_SYSFS_ROOT=$OUT/no-type || fail=1
if env GGML_CPU_EXPERT_TILES=0 GGML_TEST_THREADS=1 GGML_TEST_REQUIRE_TQ2=1 GGML_TEST_POISON_OUTPUT=1 \
    taskset -c 0-4 "$BIN/test-repack-mul-mat-id" > "$OUT/poison.txt" 2>&1; then
    echo "NaN sabotage was accepted"; fail=1
fi
grep -qi 'tq2_0.*FAIL' "$OUT/poison.txt" || { echo "NaN sabotage did not reach TQ2_0"; fail=1; }

echo "== EXPERT-TILES KERNEL IDENTITY: $([ $fail -eq 0 ] && echo PASS || echo FAIL)"
exit $fail
