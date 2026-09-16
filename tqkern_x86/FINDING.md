# The x86 TQ2_0 kernel: the unpack is NOT the wall, the AVX2 multiply-add is (2026-09-17)

Status: measured, and the task's stop condition fired. The ARM finding of 2026-09-16 was
that the TQ2_0 unpack cost as much as the dot products and that masking without shifting
bought 1.19x to 1.65x per core, bit-identical. That premise does not hold on x86. On a Zen 4
core, removing the entire unpack from the shipped kernel changes its speed by nothing
measurable, so the mask-only trick has nothing to win. The instruction was to stop after the
floors table if the unpack is not the wall, and that is what happened: no in-tree kernel
change, no identity test, no identity gate and no BitNet timing were run. The branch
`tq-avx512` in `/home/ncannings/ternary_bench/llama.cpp-x86k` carries the harness and these
results and nothing else; the kernel sources are untouched.

## 0. There is no x86 repack path for TQ2_0 in this tree

Asked and answered, because it decides what "the x86 kernel" means. There is none.

- `ggml/src/ggml-cpu/arch/x86/repack.cpp` contains no `tq2` symbol at all.
- The repack traits for `GGML_TYPE_TQ2_0` (repack.cpp, the `ggml_repack_get_optimal_repack_type`
  chain) are gated on `ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()` for the 8x8 variant
  and `ggml_cpu_has_neon() && ggml_cpu_has_dotprod()` for the 8x4 variant, so no x86 CPU ever
  selects a TQ2_0 CPU_REPACK buffer.
- `arch-fallback.h` maps `ggml_gemv_tq2_0_*_generic` and `ggml_gemm_tq2_0_*_generic` onto the
  plain names in the x86 block, which is the tree saying the same thing: x86 has only the
  scalar generic, and it is unreachable anyway.
- `llamafile/sgemm.cpp` (tinyBLAS) has no TQ2_0 path either.

So on x86 every TQ2_0 matmul, decode and prefill alike, goes through the single-row
`ggml_vec_dot_tq2_0_q8_K` in `arch/x86/quants.c`, whose only vector path is `#if defined(__AVX2__)`.
There is no AVX-512 path in the shipped kernel. That one function is the whole x86 story, and
it is the baseline below. Consequently the ARM comparison `(b)` (the repacked 8-column GEMV)
has no x86 counterpart and does not appear in this table.

## 1. The box

Azure `Standard_D8as_v6`, eastus, on demand, resource group `ternary-x86-0008`. AMD EPYC 9V74
(Genoa, Zen 4), 8 vCPU = 4 physical cores with SMT, 1 MiB L2 per core, 32 MiB L3, 31 GB RAM,
Ubuntu 24.04, GCC 13.3, `-O3 -march=native`. The build reports
`AVX2=1 AVX512F=1 AVX512VL=1 AVX512VNNI=1 AVX512BW=1`. All runs `taskset -c 0`, one core, the
SMT sibling (cpu 1) idle. Frequency during the runs 3.63 to 3.69 GHz.

Harness `~/ternary_bench/tqkern_x86/tq2_x86_lab.c`, the x86 counterpart of the ARM
`tq2_kernel_lab.c`: a cache-resident 2048 x 1024 TQ2_0 matrix (0.541 MB of codes plus scales,
2,097,152 weights) against one q8_K token, best of 9 reps of 50 passes. Raw output in
`~/ternary_bench/tqkern_x86/tq2_x86_lab.results.txt` and `tq2_x86_lab.real.txt`.

## 2. The floors, one Zen 4 core, cache-resident

`(a)` is `ggml_vec_dot_tq2_0_q8_K` copied verbatim out of `arch/x86/quants.c`. `(c)` streams
exactly the same code bytes with one XOR per load. `(d5)` is the unpack alone: the same code
loads plus the three `vpsrlw` and four `vpand` the shipped kernel issues, and no activations.
`(d2)` is the mirror image: the same code loads, the same four activation loads, the same four
`vpmaddubsw` and the same four `vpaddw` per 32 code bytes, with NO unpack at all, so it is what
the kernel would run at if extracting the 2-bit planes were free. `(d3)` and `(d4)` replace the
`vpmaddubsw` plus `vpaddw` pair with `vpdpbusd` on the same loads, at 256 and 512 bits.

| kernel | vector ops per 64 weights | ns per weight | G weights/s per core |
|---|---:|---:|---:|
| (a) vec_dot AVX2, the shipped kernel | 10 | 0.0135 | 74.3 |
| (c) load only, memory floor | 1 | 0.0023 | 432.5 |
| (d5) unpack only, no activations | 4 plus an XOR sink | 0.0038 | 264.3 |
| (d2) maddubs only, no unpack | 6.5 | 0.0136 | **73.7** |
| (d6) half the maddubs (2 per 32 code bytes) | 5.5 | 0.0041 | 246.9 |
| (d3) vpdpbusd only, ymm, no unpack | 4.5 | 0.0066 | 150.7 |
| (d4) vpdpbusd only, zmm, no unpack | 2.5 | 0.0054 | 185.7 |

Three readings, and the first one is the finding:

1. **The unpack is free on Zen 4.** `(d2)` deletes the whole unpack from the shipped kernel and
   measures 73.7 G weights/s against the kernel's 74.3, which is a 0.8 percent DEFICIT, not a
   gain: the difference is inside the run-to-run spread of 1 to 2 percent. The seven unpack
   operations per 32 code bytes cost nothing because they hide completely under the AVX2
   multiply-add chain. This is the opposite of the X925 and the M1, where the same removal was
   worth 1.5x to 1.8x. The ARM finding does not travel.
2. **The wall is the arithmetic primitive.** `(d3)` runs the identical loads and the identical
   loop with `vpdpbusd` instead of `vpmaddubsw` plus `vpaddw` and is 2.03x the shipped kernel.
   The shipped kernel is at 49 percent of the 256-bit VNNI floor and 40 percent of the 512-bit
   one. Zen 4 has had VNNI since launch and the TQ2_0 kernel does not use it.
3. **Memory is nowhere near the limit in cache**, same as on Arm: the load floor is 5.8x the
   kernel rate. A cache-resident ternary matmul on Zen 4 is arithmetic bound with a large
   margin, which is why the Genoa-X residency sweep found ternary flat at 39 G weights/s per
   core in and out of the 96 MB L3.

Caveat on `(d6)`: halving the `vpmaddubsw` count from four to two per 32 code bytes is worth
3.4x, not 2x, so the shipped kernel's time is not a clean linear function of the maddubs count
and no micro-architectural claim is made from that row. The disassembly was checked
(`objdump --disassemble=`): `(d2)` issues 32 `vpmaddubsw` and 40 `vmovdqu` per row and `(d6)`
issues 16, so neither arm has had its work optimised away. The two rows that carry the finding,
`(d2)` and `(d3)`, are structurally identical to `(a)` in everything but the operation under
test.

## 3. The variants, all bit-identical, none of them the ARM win

Every variant keeps the same integer sum per block, the same bsums correction and the same float
association as `(a)`: one `_mm256_cvtepi32_ps`, one multiply by `d` and one add into the same
eight-lane `sumf`, then the same `hsum_float_8`. The bar is equality of the output bytes.

| variant | ns per weight | G weights/s | max diff vs (a) | speed vs (a) |
|---|---:|---:|---|---:|
| (a) AVX2 shift-and-mask, shipped | 0.0135 | 74.3 | - | 1.00 |
| X1 AVX2 mask-only, 3 scale classes | 0.0131 | 76.1 | 0 | 1.03 |
| X2 AVX2 mask-only, 4 scale classes | 0.0130 | 76.7 | 0 | 1.04 |
| X3 VNNI ymm, mask-only, 4 int32 accumulators | 0.0120 | 83.2 | 0 | 1.13 |
| X4 VNNI zmm, broadcast-and-mask, two accumulators | 0.0128 | 78.1 | 0 | 1.06 |
| X5 X3 with two rows in flight | 0.0116 | 86.4 | 0 | 1.17 |
| X6 X3 with four rows in flight | 0.0102 | 97.8 | 0 | **1.32** |
| SABOTAGE: X3 with the plane-1 fold shifted by 3 | 0.0120 | 83.0 | DIFFERS in 2047 of 2048 | - |

**X1 and X2 are the ARM kernel ported, and they are worth 3 to 4 percent.** X1 masks planes 0,
1 and 2 without shifting (`w & 0x03`, `w & 0x0c`, `w & 0x30`) and keeps the shift for plane 3,
giving three accumulator scale classes folded once per block by an exact arithmetic right shift.
X2 masks all four planes including `w & 0xc0`, which `vpmaddubsw` tolerates because its first
operand is unsigned, and keeps the 64x class in two per-half accumulators so it never leaves
int16 (128 x 127 x 2 = 32512 fits). Both are bit-identical. Both are inside the noise of the
`(d2)` result that predicted them: you cannot win back time that was never being spent.

**X3 is the real lever and it is a different change.** `vpdpbusd` takes the code operand unsigned
and accumulates into int32, so every plane can be masked without shifting AND without any
overflow argument, and the four 16-bit accumulate-adds of the AVX2 form disappear into the
instruction. Its int32 lane mapping (lane k covers code bytes 4k to 4k+3) is exactly the lane
mapping the shipped kernel reaches after its `_mm256_madd_epi16`, so the epilogue is unchanged
and identity is structural rather than lucky. Four masked planes, four int32 accumulators, three
exact `_mm256_srai_epi32` folds per block.

**X4, the 512-bit form, loses.** It broadcasts 32 code bytes into both halves of a zmm and masks
with a two-plane mask, so one AND produces two planes and one 512-bit activation load feeds both:
two broadcasts, four ANDs and four `vpdpbusd` cover a whole 256-weight block, and the per-lane
scales are undone by one `_mm512_srav_epi32` per accumulator. It runs 14 vector operations per 256-weight block against X3's
26 and runs 6 percent SLOWER, which is Zen 4 splitting every 512-bit operation into two
256-bit halves and gaining nothing back.

**The remaining gap is latency, not issue.** X3 at 1.13x sits well below the `(d3)` floor of
2.03x. Putting two independent rows in flight takes it to 1.17x and four rows to 1.32x, with no
change to any row's arithmetic. The single-row `vec_dot` signature is what caps this: the kernel
is starved of independent work, not of issue slots.

The sabotage arm is the proof that the "max diff 0" column could have failed: X3 with one fold
shift changed from 2 to 3 is still a plausible-looking kernel and it differs on 2047 of 2048
rows.

## 4. Reproduction

Three full runs on random weights and two on real ones. The real-data runs replace the codes and
scales with `blk.0.attn_output.weight` taken verbatim out of `bitnet-2b4t-TQ2_0.gguf`
(`tqkern/real_tq2_blocks.bin`, 2048 rows x 4 blocks). Every rate reproduces within 2 percent and
every variant reports max diff 0 on both, which is expected: nothing in this kernel is data
dependent. Run 3, real blocks, verbatim:

```
=== RUN 3, real TQ2_0 blocks from bitnet-2b4t-TQ2_0.gguf ===
weights: real TQ2_0 blocks from real_tq2_blocks.bin
matrix 2048 x 1024 TQ2_0  codes+scales 0.541 MB  (2097152 weights)
build: AVX2=1 AVX512F=1 AVX512VL=1 AVX512VNNI=1 AVX512BW=1
shipped AVX2 vec_dot vs scalar ref maxrel 8.16e-05
(a) vec_dot AVX2 (shipped)            28.707 us  0.0137 ns/weight     73.1 G weights/s  baseline
(c) load only (mem floor)              4.684 us  0.0022 ns/weight    447.7 G weights/s  -
(d2) maddubs only (no unpack)         28.822 us  0.0137 ns/weight     72.8 G weights/s  -
(d5) unpack only (no maddubs)          7.964 us  0.0038 ns/weight    263.3 G weights/s  -
(d6) 2 maddubs per 32B codes           8.579 us  0.0041 ns/weight    244.5 G weights/s  -
(d3) vpdpbusd ymm only                13.831 us  0.0066 ns/weight    151.6 G weights/s  -
(d4) vpdpbusd zmm only                11.022 us  0.0053 ns/weight    190.3 G weights/s  -

-- variants (bitwise identity is against (a), the shipped kernel) --
X1 AVX2 mask-only 3 classes           27.540 us  0.0131 ns/weight     76.1 G weights/s  max diff 0  1.05x vs (a)
X2 AVX2 mask-only 4 classes           27.473 us  0.0131 ns/weight     76.3 G weights/s  max diff 0  1.06x vs (a)
X3 VNNI ymm mask-only                 25.218 us  0.0120 ns/weight     83.2 G weights/s  max diff 0  1.15x vs (a)
X4 VNNI zmm broadcast-mask            26.973 us  0.0129 ns/weight     77.8 G weights/s  max diff 0  1.08x vs (a)
X5 VNNI ymm, 2 rows                   24.329 us  0.0116 ns/weight     86.2 G weights/s  max diff 0  1.19x vs (a)
X6 VNNI ymm, 4 rows                   21.331 us  0.0102 ns/weight     98.3 G weights/s  max diff 0  1.36x vs (a)
```

and the sabotage arm, run 4, random weights, verbatim:

```
SABOTAGE X3 fold shift 3              24.902 us  0.0119 ns/weight     84.2 G weights/s  DIFFERS in 2047 of 2048  1.14x vs (a)
```

## 5. What was NOT done, and why

The task's stop rule: if the floors table says the unpack is not the wall, stop after the table.
It says exactly that, in the row that matters most (`(d2)` at 73.7 against the kernel's 74.3), so
the following were not run and nothing in this record should be read as if they were:

- No change to `ggml/src/ggml-cpu/arch/x86/quants.c`. The branch `tq-avx512` does not touch any
  kernel source.
- No `test-tq2_0-x86-kernel` in the tree. The identity evidence here is the harness comparison
  plus its sabotage arm, at 2048 rows of one shape, not the 1-to-17-row test through
  `ggml_mul_mat` that an in-tree change would require. (The tree's existing
  `test-tq2_0-kernel-switch` SKIPs on x86 by design: it needs a TQ2_0 CPU_REPACK weight, which
  x86 never gets, per section 0.)
- No identity gate. `gate_grace.sh` was not adapted or run, and `bitnet-2b4t-TQ2_0.gguf` was not
  needed on the VM.
- No BitNet tg128 or pp512 timing, old against new, because there is no new.

## 6. Cost

Resource group `ternary-x86-0008`, one `Standard_D8as_v6`. Created 00:09:24, delete requested
00:17:55, so about nine minutes of a roughly $0.40 per hour on-demand VM: about six cents,
call it a dollar with the rounding Azure does to the minute and the managed disk. The group
delete was issued with `--no-wait` and confirmed gone afterwards.

## 7. What this changes for the programme

- **The 30 to 40 G weights/s per core ceiling is not one phenomenon.** On Arm it was the unpack.
  On Zen 4 it is the AVX2 integer multiply-add, with the unpack hidden underneath it and VNNI
  unused. The number was the same three ways; the cause is not. Any claim of the form "ternary
  kernels are arithmetic bound at 30 to 40 G weights/s per core on three architectures" survives,
  but the attribution clause behind it must now be stated per architecture.
- **x86 has a named, unclaimed 1.3x that is not the ARM change.** X3 plus multi-row (X6) is
  bit-identical and measured 1.32x to 1.36x on one Zen 4 core against a 2.03x arithmetic floor,
  and it needs the single-row `vec_dot` signature to be widened to carry independent rows. That
  is a different piece of work from this one and is not authorised by this task.
- **The isolated kernel runs at 74 G weights/s per core while the engine delivered 39** on
  Genoa-X. The gap is engine overhead, not the kernel, and it is the same shape as the gap the
  residency profile found on the GB10. On x86 a kernel change of 1.3x would therefore be expected
  to show materially less than 1.3x end to end, and that prediction is untested.
