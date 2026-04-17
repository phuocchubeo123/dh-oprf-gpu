#pragma once

#include <cstdint>
#include "fe25519.cuh"
#include "field25519_arith.cuh"

namespace {

__device__ __forceinline__ int tonelli_shanks_sqrt(const fe25519& n, fe25519* root) {
    // p = 2^255 - 19, p - 1 = q * 2^s with s = 2 and q odd.
    constexpr uint32_t kExpQ[8] = {
        0xFFFFFFFBu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
        0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x1FFFFFFFu,
    };
    constexpr uint32_t kExpQPlus1Over2[8] = {
        0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
        0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x0FFFFFFFu,
    };
    constexpr uint32_t kExpPM1Over2[8] = {
        0xFFFFFFF6u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
        0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x3FFFFFFFu,
    };

    if (is_zero(n)) {
        *root = from_u32(0u);
        return 1;
    }

    // Euler criterion: n^((p-1)/2) must be 1 for quadratic residues.
    const fe25519 leg = pow_const(n, kExpPM1Over2);
    if (!is_one(leg)) {
        return 0;
    }

    fe25519 x = pow_const(n, kExpQPlus1Over2);
    fe25519 t = pow_const(n, kExpQ);
    if (is_one(t)) {
        *root = x;
        return 1;
    }

    // z=2 is a quadratic non-residue for this prime.
    const fe25519 z = from_u32(2u);
    fe25519 c = pow_const(z, kExpQ);
    int m = 2;

    while (!is_one(t)) {
        int i = 1;
        fe25519 t2i = sqr_mod_p(t);
        while (i < m && !is_one(t2i)) {
            t2i = sqr_mod_p(t2i);
            ++i;
        }
        if (i == m) return 0;

        fe25519 b = c;
        for (int j = 0; j < (m - i - 1); ++j) {
            b = sqr_mod_p(b);
        }

        const fe25519 b2 = sqr_mod_p(b);
        x = mul_mod_p(x, b);
        t = mul_mod_p(t, b2);
        c = b2;
        m = i;
    }

    *root = x;
    return 1;
}


}