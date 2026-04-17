#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include "fe25519.cuh"
#include "field25519_arith.cuh"
#include "curve25519_decode.cuh"
#include "curve25519_scalarmult.cuh"
#include "curve25519_elligator.cuh"
#include "field25519_sqrt.cuh"

extern "C" __global__ void fe25519_add_kernel(const fe25519* a, const fe25519* b, fe25519* out, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (idx >= n) return;
    out[idx] = add_mod_p(a[idx], b[idx]);
}

extern "C" __global__ void fe25519_mul_kernel(const fe25519* a, const fe25519* b, fe25519* out, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (idx >= n) return;
    out[idx] = mul_mod_p(a[idx], b[idx]);
}

extern "C" __global__ void fe25519_sqrt_kernel(const fe25519* a, fe25519* out, uint8_t* ok, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (idx >= n) return;
    ok[idx] = static_cast<uint8_t>(tonelli_shanks_sqrt(a[idx], &out[idx]));
}

extern "C" __global__ void curve25519_elligator2_u_kernel(const uint8_t* uniform32, fe25519* out_u, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (idx >= n) return;
    const uint8_t* in = uniform32 + (idx * 32u);
    const fe25519 r = decode_uniform_to_field(in);
    out_u[idx] = elligator2_curve25519_u(r);
}

extern "C" __global__ void curve25519_scalarmult_u_kernel(const uint8_t* scalar32, const uint8_t* u32, uint8_t* out32,
                                                           size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (idx >= n) return;

    const uint8_t* s = scalar32 + (idx * 32u);
    const uint8_t* u = u32 + (idx * 32u);
    uint8_t* out = out32 + (idx * 32u);

    const fe25519 u_fe = decode_u_coordinate(u);
    const fe25519 r_fe = x25519_scalarmult_u(s, u_fe);
    encode_u_coordinate(r_fe, out);
}

extern "C" __global__ void curve25519_scalarmult_u_raw_kernel(const uint8_t* scalar32, const uint8_t* u32, uint8_t* out32,
                                                               size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (idx >= n) return;

    const uint8_t* s = scalar32 + (idx * 32u);
    const uint8_t* u = u32 + (idx * 32u);
    uint8_t* out = out32 + (idx * 32u);

    const fe25519 u_fe = decode_u_coordinate(u);
    const fe25519 r_fe = x25519_scalarmult_u_raw(s, u_fe);
    encode_u_coordinate(r_fe, out);
}

extern "C" __global__ void curve25519_scalarmult_fe_raw_kernel(const uint8_t* scalar32, const fe25519* u_in,
                                                                fe25519* out_u, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (idx >= n) return;
    const uint8_t* s = scalar32 + (idx * 32u);
    out_u[idx] = x25519_scalarmult_u_raw(s, u_in[idx]);
}

extern "C" __global__ void curve25519_scalarmult_fe_kernel(const uint8_t* scalar32, const fe25519* u_in,
                                                            fe25519* out_u, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (idx >= n) return;
    const uint8_t* s = scalar32 + (idx * 32u);
    out_u[idx] = x25519_scalarmult_u(s, u_in[idx]);
}