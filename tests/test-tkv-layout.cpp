// Exact-layout test for the RUNTIME tkv3 block (Codex defect 4).
//
// The accepted Python isolation codec uses a 16-element group with an FP32
// scale. The runtime block is a 64-element block with an FP16 scale packed into
// 42 bytes. Family-level MST bit-exactness says nothing about THAT packer, so
// this test reads the bytes the runtime actually writes, decodes them with an
// independent implementation written from the format description alone, and
// requires exact agreement.
//
// Independent means: this file does not call any tkv code except the public
// from_float/to_float, and it re-derives the packing from first principles
// (five balanced trits per byte, digit p weighted 3^p, fp16 block scale).

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int failures = 0;

static void check(bool ok, const char * what) {
    printf("%-64s %s\n", what, ok ? "pass" : "FAIL");
    if (!ok) {
        failures++;
    }
}

// ---- an independent reader of the documented layout -------------------------
// 42 bytes: [0..1] fp16 scale, [2..14] plane 0, [15..27] plane 1, [28..40] plane 2,
// [41] padding. Five trits per byte, trit k of a byte at 3^k, digit d in {0,1,2}
// meaning {-1,0,+1}.
struct decoded {
    float   scale;
    int     q[64];          // reconstructed integers
    uint8_t pad;
};

static decoded read_block_independently(const uint8_t * b) {
    decoded d{};
    uint16_t h;
    memcpy(&h, b, 2);
    // fp16 -> float, written out rather than using a helper
    const uint32_t sign = (h >> 15) & 0x1;
    const uint32_t exp  = (h >> 10) & 0x1f;
    const uint32_t man  =  h        & 0x3ff;
    float v;
    if (exp == 0) {
        v = std::ldexp((float) man, -24);
    } else if (exp == 31) {
        v = man ? NAN : INFINITY;
    } else {
        v = std::ldexp((float) (man | 0x400), (int) exp - 25);
    }
    d.scale = sign ? -v : v;

    static const int pow3[5] = {1, 3, 9, 27, 81};
    for (int i = 0; i < 64; ++i) {
        d.q[i] = 0;
    }
    for (int plane = 0; plane < 3; ++plane) {
        const uint8_t * p = b + 2 + plane * 13;
        int weight = 1;
        for (int k = 0; k < plane; ++k) {
            weight *= 3;
        }
        for (int i = 0; i < 64; ++i) {
            const int trit = (p[i / 5] / pow3[i % 5]) % 3 - 1;
            d.q[i] += trit * weight;
        }
    }
    d.pad = b[41];
    return d;
}

int main() {
    ggml_cpu_init();
    const ggml_type T = GGML_TYPE_TKV3;
    const auto * tr  = ggml_get_type_traits(T);
    const auto * trc = ggml_get_type_traits_cpu(T);

    check(ggml_type_size(T) == 42, "runtime block is 42 bytes");
    check(ggml_blck_size(T) == 64, "runtime block holds 64 elements");
    check(ggml_row_size(T, 64) == 42, "a 64-element row is one 42-byte block");

    std::mt19937 rng(20260923);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // 1. the bytes the runtime writes decode, under an independent reader, to
    //    exactly what the runtime itself reports
    {
        bool ok_codes = true, ok_values = true, ok_pad = true;
        for (int trial = 0; trial < 64; ++trial) {
            std::vector<float> x(64);
            for (auto & v : x) {
                v = nd(rng) * std::pow(10.0f, (float) (trial % 7) - 3);   // 1e-3 to 1e3
            }
            std::vector<uint8_t> blk(42, 0xAB);      // poisoned, so the pad byte
            trc->from_float(x.data(), blk.data(), 64);   // is tested, not the allocator

            const decoded d = read_block_independently(blk.data());
            std::vector<float> back(64);
            tr->to_float(blk.data(), back.data(), 64);

            for (int i = 0; i < 64; ++i) {
                if (std::fabs(d.scale * d.q[i] - back[i]) > 1e-6f * std::fabs(back[i]) + 1e-9f) {
                    ok_values = false;
                }
                if (d.q[i] < -13 || d.q[i] > 13) {
                    ok_codes = false;       // three planes cannot exceed 13
                }
            }
            if (d.pad != 0) {
                ok_pad = false;             // padding byte must be written, not left stale
            }
        }
        check(ok_codes, "every decoded integer lies in [-13,13], the 3-plane range");
        check(ok_values, "independent decode matches the runtime's own to_float exactly");
        check(ok_pad, "the padding byte is zeroed, so blocks are byte-reproducible");
    }

    // 2. the scale really is fp16: quantising a value that fp16 cannot represent
    //    exactly must round the same way the format says
    {
        std::vector<float> x(64, 0.0f);
        x[0] = 1.0f / 3.0f;                 // scale becomes (1/3)/13, not representable
        std::vector<uint8_t> blk(42);
        trc->from_float(x.data(), blk.data(), 64);
        const decoded d = read_block_independently(blk.data());
        const float want = (float) (double) (_Float16) (x[0] / 13.0f);
        check(std::fabs(d.scale - want) <= 1e-7f, "block scale is fp16-rounded as specified");
    }

    // 3. byte reproducibility: the same input always produces the same 42 bytes
    {
        std::vector<float> x(64);
        for (auto & v : x) {
            v = nd(rng);
        }
        std::vector<uint8_t> a(42), b(42);
        trc->from_float(x.data(), a.data(), 64);
        trc->from_float(x.data(), b.data(), 64);
        check(memcmp(a.data(), b.data(), 42) == 0, "packing is deterministic byte for byte");
    }

    // 4. K and V paths: a K row is one head (64) and a V row in the transposed
    //    cache is n_ctx long; both must be whole numbers of blocks and must not
    //    bleed between blocks.
    {
        bool ok = true;
        for (int n : {64, 128, 768, 2048}) {
            std::vector<float> x(n);
            for (int i = 0; i < n; ++i) {
                x[i] = nd(rng) * (1.0f + (float) (i / 64));   // a different scale per block
            }
            std::vector<uint8_t> packed(ggml_row_size(T, n));
            trc->from_float(x.data(), packed.data(), n);
            for (int b = 0; b < n / 64; ++b) {
                const decoded d = read_block_independently(packed.data() + (size_t) b * 42);
                float amax = 0.0f;
                for (int i = 0; i < 64; ++i) {
                    amax = std::max(amax, std::fabs(x[b * 64 + i]));
                }
                // each block's scale must come from ITS OWN maximum
                const float want = (float) (double) (_Float16) (amax / 13.0f);
                if (std::fabs(d.scale - want) > 1e-6f * want + 1e-9f) {
                    ok = false;
                }
            }
        }
        check(ok, "every block takes its scale from its own 64 elements (K and V rows)");
    }

    // 5. cache positions, wraparound and GQA indexing at the byte level: a block
    //    written at slot s must occupy exactly bytes [42s, 42s+42) and nothing else
    {
        const int slots = 16;
        std::vector<uint8_t> cache(42 * slots, 0xAB);      // poison
        std::vector<std::vector<float>> src(slots);
        const int order[] = {13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        for (int k = 0; k < slots; ++k) {
            const int s = order[k];
            src[s].resize(64);
            for (auto & v : src[s]) {
                v = nd(rng);
            }
            trc->from_float(src[s].data(), cache.data() + (size_t) s * 42, 64);
        }
        bool ok = true;
        for (int s = 0; s < slots; ++s) {
            std::vector<uint8_t> ref(42);
            trc->from_float(src[s].data(), ref.data(), 64);
            if (memcmp(cache.data() + (size_t) s * 42, ref.data(), 42) != 0) {
                ok = false;                                 // slot disturbed by a neighbour
            }
        }
        check(ok, "ring-buffer slots are byte-identical to an isolated packing");
    }

    // 6. the pad byte is never read: corrupting it must not change any decoded
    //    value. This is what decides whether zeroing it changed earlier results.
    {
        std::vector<float> x(64);
        for (auto & v : x) {
            v = nd(rng);
        }
        std::vector<uint8_t> blk(42);
        trc->from_float(x.data(), blk.data(), 64);
        std::vector<float> clean(64), dirty(64);
        tr->to_float(blk.data(), clean.data(), 64);
        blk[41] = 0xFF;                       // corrupt only the pad byte
        tr->to_float(blk.data(), dirty.data(), 64);
        float dot_clean = 0.0f, dot_dirty = 0.0f;
        std::vector<float> y(64, 1.0f);
        blk[41] = 0x00;
        trc->vec_dot(64, &dot_clean, 0, blk.data(), 0, y.data(), 0, 1);
        blk[41] = 0xFF;
        trc->vec_dot(64, &dot_dirty, 0, blk.data(), 0, y.data(), 0, 1);
        check(memcmp(clean.data(), dirty.data(), 64 * sizeof(float)) == 0 &&
              dot_clean == dot_dirty,
              "the pad byte is never read: corrupting it changes no decoded value");
    }

    printf("\n%s\n", failures == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return failures == 0 ? 0 : 1;
}
