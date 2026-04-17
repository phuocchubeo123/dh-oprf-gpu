#pragma once

#include <cstdint>
#include "fe25519.cuh"
#include "field25519_arith.cuh"

namespace {

__device__ __forceinline__ void cswap_fe(fe25519* a, fe25519* b, uint32_t swap) {
    const uint32_t mask = 0u - swap;
    for (int i = 0; i < 8; ++i) {
        const uint32_t x = mask & (a->limb[i] ^ b->limb[i]);
        a->limb[i] ^= x;
        b->limb[i] ^= x;
    }
}

__device__ __forceinline__ uint32_t scalar_bit(const uint8_t scalar[32], int bit_idx) {
    const int byte_idx = bit_idx >> 3;
    const int bit_off = bit_idx & 7;
    return static_cast<uint32_t>((scalar[byte_idx] >> bit_off) & 1u);
}

__device__ __forceinline__ fe25519 x25519_scalarmult_u_impl(const uint8_t scalar_in[32], const fe25519& u_in, int clamp) {
    uint8_t k[32];
    for (int i = 0; i < 32; ++i) k[i] = scalar_in[i];
    if (clamp) {
        k[0] &= 248u;
        k[31] &= 127u;
        k[31] |= 64u;
    }

    const fe25519 x1 = u_in;
    fe25519 x2 = from_u32(1u);
    fe25519 z2 = from_u32(0u);
    fe25519 x3 = x1;
    fe25519 z3 = from_u32(1u);
    // RFC 7748 ladder uses a24 = (A - 2) / 4 = 121665 for Curve25519.
    const fe25519 a24 = from_u32(121665u);
    uint32_t swap = 0u;

    for (int t = 254; t >= 0; --t) {
        const uint32_t kt = scalar_bit(k, t);
        swap ^= kt;
        cswap_fe(&x2, &x3, swap);
        cswap_fe(&z2, &z3, swap);
        swap = kt;

        const fe25519 A = add_mod_p(x2, z2);
        const fe25519 AA = sqr_mod_p(A);
        const fe25519 B = sub_mod_p(x2, z2);
        const fe25519 BB = sqr_mod_p(B);
        const fe25519 E = sub_mod_p(AA, BB);
        const fe25519 C = add_mod_p(x3, z3);
        const fe25519 D = sub_mod_p(x3, z3);
        const fe25519 DA = mul_mod_p(D, A);
        const fe25519 CB = mul_mod_p(C, B);
        const fe25519 DA_plus_CB = add_mod_p(DA, CB);
        const fe25519 DA_minus_CB = sub_mod_p(DA, CB);
        x3 = sqr_mod_p(DA_plus_CB);
        z3 = mul_mod_p(x1, sqr_mod_p(DA_minus_CB));
        x2 = mul_mod_p(AA, BB);
        z2 = mul_mod_p(E, add_mod_p(AA, mul_mod_p(a24, E)));
    }

    cswap_fe(&x2, &x3, swap);
    cswap_fe(&z2, &z3, swap);
    return mul_mod_p(x2, inv_mod_p(z2));
}

__device__ __forceinline__ fe25519 x25519_scalarmult_u(const uint8_t scalar_in[32], const fe25519& u_in) {
    return x25519_scalarmult_u_impl(scalar_in, u_in, 1);
}

__device__ __forceinline__ fe25519 x25519_scalarmult_u_raw(const uint8_t scalar_in[32], const fe25519& u_in) {
    return x25519_scalarmult_u_impl(scalar_in, u_in, 0);
}

__device__ __forceinline__ fe25519 x25519_double_u(const fe25519& u) {
    // Montgomery xDBL with a24=(A+2)/4=121666 on Curve25519.
    const fe25519 one = from_u32(1u);
    const fe25519 a24 = from_u32(121666u);
    fe25519 X = u;
    fe25519 Z = one;

    const fe25519 XpZ = add_mod_p(X, Z);
    const fe25519 XmZ = sub_mod_p(X, Z);
    const fe25519 AA = sqr_mod_p(XpZ);
    const fe25519 BB = sqr_mod_p(XmZ);
    const fe25519 E = sub_mod_p(AA, BB);
    const fe25519 X2 = mul_mod_p(AA, BB);
    const fe25519 t = add_mod_p(BB, mul_mod_p(a24, E));
    const fe25519 Z2 = mul_mod_p(E, t);
    return mul_mod_p(X2, inv_mod_p(Z2));
}

}