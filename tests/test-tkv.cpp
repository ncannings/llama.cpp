// Microtests for the native packed KV formats (branch native-packed-kv).
//
// Required by the pre-registration before any quality scoring: cache
// positions, wraparound, GQA head indexing and both K and V layouts, plus the
// property the whole scheme rests on, that integer -> planes -> integer is
// exact. Deterministic: fixed seeds, no timing, no model.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <random>
#include <vector>

static int failures = 0;

static void check(bool ok, const char * what) {
    printf("%-58s %s\n", what, ok ? "pass" : "FAIL");
    if (!ok) {
        failures++;
    }
}

static std::vector<float> make_row(int n, uint32_t seed, float scale = 1.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = d(rng);
    }
    return v;
}

static void roundtrip(ggml_type t, const std::vector<float> & in, std::vector<float> & out) {
    const int64_t n = (int64_t) in.size();
    std::vector<uint8_t> packed(ggml_row_size(t, n));
    const auto * tr = ggml_get_type_traits(t);
    const auto * trc = ggml_get_type_traits_cpu(t);
    trc->from_float(in.data(), packed.data(), n);
    out.resize(in.size());
    tr->to_float(packed.data(), out.data(), n);
}

// the levels each format can represent exactly
static int levels_of(ggml_type t) { return t == GGML_TYPE_TKV1 ? 1 : 13; }

int main() {
    ggml_cpu_init();
    const int HEAD = 64;

    for (ggml_type t : {GGML_TYPE_TKV1, GGML_TYPE_TKV3}) {
        const char * name = ggml_type_name(t);
        char buf[128];

        // 1. block geometry: one block is exactly one head row
        snprintf(buf, sizeof(buf), "%s: block size divides the 64-wide head", name);
        check(HEAD % ggml_blck_size(t) == 0, buf);

        // 2. integer -> planes -> integer is exact over the whole representable range
        {
            const int lv = levels_of(t);
            std::vector<float> in(HEAD);
            for (int i = 0; i < HEAD; ++i) {
                in[i] = (float) (((i % (2 * lv + 1)) - lv));   // every level, in order
            }
            std::vector<float> out;
            roundtrip(t, in, out);
            bool exact = true;
            for (int i = 0; i < HEAD; ++i) {
                if (std::fabs(out[i] - in[i]) > 1e-3f) {
                    exact = false;
                }
            }
            snprintf(buf, sizeof(buf), "%s: integers in [-%d,%d] survive the round trip", name, lv, lv);
            check(exact, buf);
        }

        // 3. every packed byte decodes to trits in {-1,0,1}: no code above 3^5-1
        {
            std::vector<float> in = make_row(HEAD, 11);
            std::vector<uint8_t> packed(ggml_row_size(t, HEAD));
            ggml_get_type_traits_cpu(t)->from_float(in.data(), packed.data(), HEAD);
            bool ok = true;
            for (size_t i = sizeof(uint16_t); i < packed.size(); ++i) {
                if (packed[i] > 242) {           // 3^5 - 1
                    ok = false;
                }
            }
            snprintf(buf, sizeof(buf), "%s: no packed byte exceeds 3^5-1", name);
            check(ok, buf);
        }

        // 4. a zero row stays zero (the scale-zero path)
        {
            std::vector<float> in(HEAD, 0.0f), out;
            roundtrip(t, in, out);
            bool ok = true;
            for (float v : out) {
                if (v != 0.0f) {
                    ok = false;
                }
            }
            snprintf(buf, sizeof(buf), "%s: an all-zero row round trips to zero", name);
            check(ok, buf);
        }

        // 5. each block is independent: a block's scale must not leak into its
        //    neighbour. Two head rows of wildly different magnitude, in one row.
        {
            std::vector<float> in = make_row(HEAD, 3, 1.0f);
            std::vector<float> big = make_row(HEAD, 4, 1000.0f);
            in.insert(in.end(), big.begin(), big.end());
            std::vector<float> out;
            roundtrip(t, in, out);
            double e_small = 0.0, ref_small = 0.0;
            for (int i = 0; i < HEAD; ++i) {
                e_small   += std::fabs(out[i] - in[i]);
                ref_small += std::fabs(in[i]);
            }
            snprintf(buf, sizeof(buf), "%s: a loud neighbouring block does not swamp a quiet one", name);
            check(e_small / ref_small < (t == GGML_TYPE_TKV1 ? 0.9 : 0.15), buf);
        }

        // 6. cache positions and wraparound: writing head rows at arbitrary
        //    offsets in a ring buffer, including a wrap, must not disturb
        //    neighbours or change the decoded values.
        {
            const int slots = 8;
            std::vector<uint8_t> cache(ggml_row_size(t, HEAD) * slots, 0);
            std::vector<std::vector<float>> written(slots);
            const int order[] = {5, 6, 7, 0, 1, 2, 3, 4};      // wraps past the end
            for (int k = 0; k < slots; ++k) {
                const int slot = order[k];
                written[slot] = make_row(HEAD, 100 + slot);
                ggml_get_type_traits_cpu(t)->from_float(
                    written[slot].data(), cache.data() + (size_t) slot * ggml_row_size(t, HEAD), HEAD);
            }
            bool ok = true;
            for (int slot = 0; slot < slots; ++slot) {
                std::vector<float> got(HEAD);
                ggml_get_type_traits(t)->to_float(
                    cache.data() + (size_t) slot * ggml_row_size(t, HEAD), got.data(), HEAD);
                std::vector<float> ref;
                roundtrip(t, written[slot], ref);
                for (int i = 0; i < HEAD; ++i) {
                    if (got[i] != ref[i]) {
                        ok = false;
                    }
                }
            }
            snprintf(buf, sizeof(buf), "%s: ring-buffer slots, written out of order, read back", name);
            check(ok, buf);
        }

        // 7. GQA head indexing: heads are contiguous blocks in one row, and
        //    reading head h must give head h, not a neighbour.
        {
            // The SAME data in every head, each multiplied by a different
            // constant. With one scale per head the relative error must be
            // identical across heads; if a scale leaked between heads it would
            // not be. Different random data per head would instead measure the
            // sampling variation of each block's maximum, which is not the
            // property under test.
            const int heads = 12;
            const std::vector<float> base = make_row(HEAD, 200);
            std::vector<float> row;
            for (int h = 0; h < heads; ++h) {
                for (int i = 0; i < HEAD; ++i) {
                    row.push_back(base[i] * (float) (1 << h));   // 1x to 2048x
                }
            }
            std::vector<float> out;
            roundtrip(t, row, out);
            // Heads differ in magnitude by 12x. If each head kept its own
            // scale the RELATIVE error is head-independent; if a scale leaked
            // across heads the loud heads would be fine and the quiet ones
            // ruined. So the test is on the spread, not on an absolute bound.
            double lo = 1e9, hi = 0.0;
            for (int h = 0; h < heads; ++h) {
                double num = 0.0, den = 0.0;
                for (int i = 0; i < HEAD; ++i) {
                    const int j = h * HEAD + i;
                    num += std::fabs(out[j] - row[j]);
                    den += std::fabs(row[j]);
                }
                const double rel = num / den;
                lo = std::min(lo, rel);
                hi = std::max(hi, rel);
            }
            snprintf(buf, sizeof(buf), "%s: 12 heads spanning 2048x in magnitude, "
                     "relative error %.4f to %.4f", name, lo, hi);
            check(hi / lo < 1.02, buf);       // identical up to fp16 scale rounding
        }

        // 8. vec_dot agrees with dequantise-then-dot (the V layout path)
        {
            std::vector<float> x = make_row(HEAD * 4, 7);
            std::vector<float> y = make_row(HEAD * 4, 8);
            std::vector<uint8_t> packed(ggml_row_size(t, HEAD * 4));
            ggml_get_type_traits_cpu(t)->from_float(x.data(), packed.data(), HEAD * 4);
            float dot = 0.0f;
            ggml_get_type_traits_cpu(t)->vec_dot(HEAD * 4, &dot, 0, packed.data(), 0, y.data(), 0, 1);
            std::vector<float> deq(HEAD * 4);
            ggml_get_type_traits(t)->to_float(packed.data(), deq.data(), HEAD * 4);
            double ref = 0.0;
            for (int i = 0; i < HEAD * 4; ++i) {
                ref += (double) deq[i] * y[i];
            }
            snprintf(buf, sizeof(buf), "%s: vec_dot matches dequantise-then-dot", name);
            check(std::fabs(dot - ref) <= 1e-3 * std::fabs(ref) + 1e-4, buf);
        }
    }

    // 9. the formats are distinct: TKV3 must be strictly better than TKV1
    {
        std::vector<float> in = make_row(64 * 8, 42);
        std::vector<float> o1, o3;
        roundtrip(GGML_TYPE_TKV1, in, o1);
        roundtrip(GGML_TYPE_TKV3, in, o3);
        double e1 = 0.0, e3 = 0.0;
        for (size_t i = 0; i < in.size(); ++i) {
            e1 += std::fabs(o1[i] - in[i]);
            e3 += std::fabs(o3[i] - in[i]);
        }
        check(e3 < e1 / 5.0, "tkv3 reconstructs at least 5x better than tkv1");
    }

    // 10. row sizes are what the byte accounting claims
    {
        const size_t r1 = ggml_row_size(GGML_TYPE_TKV1, 64);
        const size_t r3 = ggml_row_size(GGML_TYPE_TKV3, 64);
        // 15 and 41 bytes of payload, padded by the compiler to 16 and 42 for
        // the 2-byte scale. The padding is real cache traffic and is counted.
        printf("tkv1 row of 64: %zu bytes = %.3f bits/element (13 packed + 2 scale + padding)\n",
               r1, r1 * 8.0 / 64);
        printf("tkv3 row of 64: %zu bytes = %.3f bits/element (39 packed + 2 scale + padding)\n",
               r3, r3 * 8.0 / 64);
        check(r1 == 16 && r3 == 42, "row sizes are 16 and 42 bytes per 64 elements, padding included");
    }

    printf("\n%s\n", failures == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return failures == 0 ? 0 : 1;
}
