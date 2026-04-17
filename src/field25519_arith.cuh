#pragma once

#include <cstdint>
#include "fe25519.cuh"

namespace {

    // Modulo of the field 25519
    __device__ __constant__ uint32_t kP[8] = {
        0xFFFFFFEDu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
        0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x7FFFFFFFu,
    };


    // Compare the number with the modulo p
    __device__ __forceinline__ int cmp_ge_p(const fe25519& a) {
        for (int i = 7; i >= 0; --i) {
            if (a.limb[i] > kP[i]) return 1;
            if (a.limb[i] < kP[i]) return 0;
        }
        return 1;
    }

    __device__ __forceinline__ int is_zero(const fe25519& a) {
        uint32_t acc = 0;
        for (int i = 0; i < 8; ++i) acc |= a.limb[i];
        return acc == 0;
    }

    __device__ __forceinline__ int is_one(const fe25519& a) {
        if (a.limb[0] != 1u) return 0;
        for (int i = 1; i < 8; ++i) {
            if (a.limb[i] != 0u) return 0;
        }
        return 1;
    }

    __device__ __forceinline__ fe25519 from_u32(uint32_t x) {
        fe25519 out{};
        out.limb[0] = x;
        return out;
    }

    __device__ __forceinline__ void sub_p(fe25519* a) {
        uint64_t borrow = 0;
        for (int i = 0; i < 8; ++i) {
            const uint64_t cur = static_cast<uint64_t>(a->limb[i]);
            const uint64_t sub = static_cast<uint64_t>(kP[i]) + borrow;
            a->limb[i] = static_cast<uint32_t>(cur - sub);
            borrow = (cur < sub) ? 1 : 0;
        }
    }

    __device__ __forceinline__ fe25519 neg_mod_p(const fe25519& a) {
        if (is_zero(a)) return a;
        fe25519 out{};
        uint64_t borrow = 0;
        for (int i = 0; i < 8; ++i) {
            const uint64_t cur = static_cast<uint64_t>(kP[i]);
            const uint64_t sub = static_cast<uint64_t>(a.limb[i]) + borrow;
            out.limb[i] = static_cast<uint32_t>(cur - sub);
            borrow = (cur < sub) ? 1 : 0;
        }
        return out;
    }

    // Reduce an internal 9-limb accumulator modulo p.
    // Helper for multiplication
    // Uses pseudo-Mersenne relations:
    //   2^256 = 38 (mod p), 2^255 = 19 (mod p)
    __device__ __forceinline__ fe25519 reduce_9(unsigned __int128 x[9]) {
        auto normalize32 = [&]() {
            for (int i = 0; i < 8; ++i) {
                x[i + 1] += (x[i] >> 32);
                x[i] &= 0xFFFFFFFFu;
            }
        };

        auto fold_top256 = [&]() {
            while (x[8] != 0) {
                const unsigned __int128 hi = x[8];
                x[8] = 0;
                x[0] += hi * 38u;
                normalize32();
            }
        };

        normalize32();
        fold_top256();

        // Bring bit 255 down with 2^255 = 19 (mod p).
        for (int iter = 0; iter < 3; ++iter) {
            const unsigned __int128 c255 = x[7] >> 31;
            x[7] &= 0x7FFFFFFFu;
            x[0] += c255 * 19u;
            normalize32();
            fold_top256();
        }

        fe25519 out{};
        for (int i = 0; i < 8; ++i) {
            out.limb[i] = static_cast<uint32_t>(x[i]);
        }

        if (cmp_ge_p(out)) sub_p(&out);
        if (cmp_ge_p(out)) sub_p(&out);
        return out;
    }

    __device__ __forceinline__ fe25519 add_mod_p(const fe25519& a, const fe25519& b) {
        unsigned __int128 x[9] = {};
        for (int i = 0; i < 8; ++i) {
            x[i] = static_cast<unsigned __int128>(a.limb[i]) + static_cast<unsigned __int128>(b.limb[i]);
        }
        return reduce_9(x);
    }

    __device__ __forceinline__ fe25519 sub_mod_p(const fe25519& a, const fe25519& b) {
        fe25519 out{};
        uint64_t borrow = 0;
        for (int i = 0; i < 8; ++i) {
            const uint64_t ai = static_cast<uint64_t>(a.limb[i]);
            const uint64_t bi = static_cast<uint64_t>(b.limb[i]) + borrow;
            out.limb[i] = static_cast<uint32_t>(ai - bi);
            borrow = (ai < bi) ? 1 : 0;
        }
        if (borrow) {
            uint64_t carry = 0;
            for (int i = 0; i < 8; ++i) {
                const uint64_t s = static_cast<uint64_t>(out.limb[i]) + static_cast<uint64_t>(kP[i]) + carry;
                out.limb[i] = static_cast<uint32_t>(s);
                carry = s >> 32;
            }
        }
        if (cmp_ge_p(out)) sub_p(&out);
        return out;
    }

    __device__ __forceinline__ fe25519 mul_mod_p(const fe25519& a, const fe25519& b) {
        unsigned __int128 t[16] = {};
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                t[i + j] += static_cast<unsigned __int128>(a.limb[i]) * static_cast<unsigned __int128>(b.limb[j]);
            }
        }

        // Normalize into 32-bit limbs before pseudo-Mersenne folding.
        for (int i = 0; i < 15; ++i) {
            t[i + 1] += (t[i] >> 32);
            t[i] &= 0xFFFFFFFFu;
        }

        unsigned __int128 x[9] = {};
        for (int i = 0; i < 8; ++i) x[i] += t[i];
        for (int i = 8; i < 16; ++i) x[i - 8] += t[i] * 38u;

        return reduce_9(x);
    }

    __device__ __forceinline__ fe25519 sqr_mod_p(const fe25519& a) { return mul_mod_p(a, a); }

    __device__ __forceinline__ fe25519 pow_const(const fe25519& base, const uint32_t exp[8]) {
        fe25519 result = from_u32(1u);
        for (int limb = 7; limb >= 0; --limb) {
            for (int bit = 31; bit >= 0; --bit) {
                result = sqr_mod_p(result);
                if ((exp[limb] >> bit) & 1u) {
                    result = mul_mod_p(result, base);
                }
            }
        }
        return result;
    }

    __device__ __forceinline__ fe25519 inv_mod_p(const fe25519& a) {
        // a^(p-2) with p = 2^255 - 19.
        constexpr uint32_t kExpPMinus2[8] = {
            0xFFFFFFEBu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
            0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x7FFFFFFFu,
        };
        return pow_const(a, kExpPMinus2);
    }

    __device__ __forceinline__ int is_square(const fe25519& a) {
        // Euler criterion: a^((p-1)/2) is 1 for nonzero quadratic residues.
        constexpr uint32_t kExpPM1Over2[8] = {
            0xFFFFFFF6u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
            0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x3FFFFFFFu,
        };
        if (is_zero(a)) return 1;
        const fe25519 e = pow_const(a, kExpPM1Over2);
        return is_one(e);
    }

}