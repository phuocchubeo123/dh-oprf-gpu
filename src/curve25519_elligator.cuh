#pragma once 
#include "fe25519.cuh"
#include "field25519_arith.cuh"
#include "curve25519_scalarmult.cuh"

namespace {

__device__ __forceinline__ fe25519 elligator2_curve25519_u(const fe25519& r) {
    // Simple Elligator2 map to Montgomery u:
    // u = -A / (1 + 2r^2); if g(u) nonsquare, use u = -u - A.
    const fe25519 A = from_u32(486662u);
    const fe25519 one = from_u32(1u);
    const fe25519 two = from_u32(2u);

    const fe25519 rr = sqr_mod_p(r);
    const fe25519 denom = add_mod_p(one, mul_mod_p(two, rr));
    const fe25519 inv_denom = inv_mod_p(denom);
    fe25519 u = mul_mod_p(neg_mod_p(A), inv_denom);

    const fe25519 u2 = sqr_mod_p(u);
    const fe25519 u3 = mul_mod_p(u2, u);
    const fe25519 g = add_mod_p(add_mod_p(u3, mul_mod_p(A, u2)), u);  // u^3 + A*u^2 + u
    if (!is_square(g)) {
        u = sub_mod_p(neg_mod_p(u), A);  // -u - A
    }

    // Match typical ed25519-from-uniform behavior by clearing cofactor (x-only doubling x8).
    u = x25519_double_u(u);
    u = x25519_double_u(u);
    u = x25519_double_u(u);
    return u;
}


}