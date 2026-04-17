#pragma once

#include <cstdint>
#include "fe25519.cuh"
#include "field25519_arith.cuh"

namespace {
    __device__ __forceinline__ fe25519 decode_uniform_to_field(const uint8_t in[32]) {
        fe25519 out{};
        for (int i = 0; i < 8; ++i) {
            const int o = i * 4;
            out.limb[i] = static_cast<uint32_t>(in[o]) |
                        (static_cast<uint32_t>(in[o + 1]) << 8) |
                        (static_cast<uint32_t>(in[o + 2]) << 16) |
                        (static_cast<uint32_t>(in[o + 3]) << 24);
        }
        out.limb[7] &= 0x7FFFFFFFu;
        if (cmp_ge_p(out)) sub_p(&out);
        return out;
    }

    __device__ __forceinline__ fe25519 decode_u_coordinate(const uint8_t in[32]) {
        return decode_uniform_to_field(in);
    }

    __device__ __forceinline__ void encode_u_coordinate(const fe25519& a, uint8_t out[32]) {
        fe25519 x = a;
        if (cmp_ge_p(x)) sub_p(&x);
        x.limb[7] &= 0x7FFFFFFFu;
        for (int i = 0; i < 8; ++i) {
            const uint32_t w = x.limb[i];
            const int o = i * 4;
            out[o] = static_cast<uint8_t>(w & 0xFFu);
            out[o + 1] = static_cast<uint8_t>((w >> 8) & 0xFFu);
            out[o + 2] = static_cast<uint8_t>((w >> 16) & 0xFFu);
            out[o + 3] = static_cast<uint8_t>((w >> 24) & 0xFFu);
        }
    }
}