#pragma once

#include <cuda_runtime.h>
#include <openssl/bn.h>
#include <openssl/rand.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "fe25519.cuh"

extern "C" __global__ void curve25519_elligator2_u_kernel(const uint8_t* uniform32, fe25519* out_u, size_t n);
extern "C" __global__ void curve25519_scalarmult_fe_kernel(const uint8_t* scalar32, const fe25519* u_in,
                                                           fe25519* out_u, size_t n);
extern "C" __global__ void curve25519_scalarmult_u_kernel(const uint8_t* scalar32, const uint8_t* u32, uint8_t* out32,
                                                          size_t n);

inline bool checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(result) << '\n';
        return false;
    }
    return true;
}

inline BIGNUM* subgroupOrderL() {
    BIGNUM* l = nullptr;
    if (BN_hex2bn(&l, "1000000000000000000000000000000014DEF9DEA2F79CD65812631A5CF5D3ED") == 0) {
        return nullptr;
    }
    return l;
}

inline bool sampleScalarLe32(uint8_t out[32]) {
    BIGNUM* l = subgroupOrderL();
    BIGNUM* s = BN_new();
    if (l == nullptr || s == nullptr) {
        BN_free(l);
        BN_free(s);
        return false;
    }
    const int ok = BN_rand_range(s, l);
    if (ok != 1 || BN_bn2binpad(s, out, 32) != 32) {
        BN_free(l);
        BN_free(s);
        return false;
    }
    for (int i = 0; i < 16; ++i) {
        const uint8_t tmp = out[i];
        out[i] = out[31 - i];
        out[31 - i] = tmp;
    }
    BN_free(l);
    BN_free(s);
    return true;
}

inline bool fillRandomBytes(uint8_t* out, size_t len) {
    return RAND_bytes(out, static_cast<int>(len)) == 1;
}

inline std::vector<uint8_t> repeatScalar(const uint8_t scalar[32], size_t n) {
    std::vector<uint8_t> out(n * 32u);
    for (size_t i = 0; i < n; ++i) {
        std::memcpy(out.data() + i * 32u, scalar, 32u);
    }
    return out;
}

inline void encodeU25519Le(const fe25519& a, uint8_t out[32]) {
    for (int i = 0; i < 8; ++i) {
        const uint32_t w = (i == 7) ? (a.limb[i] & 0x7FFFFFFFu) : a.limb[i];
        const int o = i * 4;
        out[o] = static_cast<uint8_t>(w & 0xFFu);
        out[o + 1] = static_cast<uint8_t>((w >> 8) & 0xFFu);
        out[o + 2] = static_cast<uint8_t>((w >> 16) & 0xFFu);
        out[o + 3] = static_cast<uint8_t>((w >> 24) & 0xFFu);
    }
}

inline std::string bytesHex(const uint8_t* bytes, size_t len) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; ++i) {
        oss << std::setw(2) << static_cast<unsigned>(bytes[i]);
    }
    return oss.str();
}

inline bool gpuMapAndBlind(const std::vector<uint8_t>& uniform_inputs, const uint8_t scalar[32],
                           std::vector<uint8_t>* out_u_bytes) {
    if (uniform_inputs.size() % 32u != 0) return false;
    const size_t n = uniform_inputs.size() / 32u;
    out_u_bytes->assign(uniform_inputs.size(), 0);
    std::vector<uint8_t> repeated = repeatScalar(scalar, n);
    std::vector<fe25519> hY(n);

    uint8_t* dX = nullptr;
    uint8_t* dA = nullptr;
    fe25519* dH = nullptr;
    fe25519* dY = nullptr;

    const size_t scalarBytes = uniform_inputs.size();
    const size_t feBytes = n * sizeof(fe25519);
    const int threads = 256;
    const int blocks = static_cast<int>((n + static_cast<size_t>(threads) - 1u) / static_cast<size_t>(threads));

    if (!checkCuda(cudaMalloc(&dX, scalarBytes), "cudaMalloc dX") ||
        !checkCuda(cudaMalloc(&dA, scalarBytes), "cudaMalloc dA") ||
        !checkCuda(cudaMalloc(&dH, feBytes), "cudaMalloc dH") ||
        !checkCuda(cudaMalloc(&dY, feBytes), "cudaMalloc dY")) {
        cudaFree(dX);
        cudaFree(dA);
        cudaFree(dH);
        cudaFree(dY);
        return false;
    }

    if (!checkCuda(cudaMemcpy(dX, uniform_inputs.data(), scalarBytes, cudaMemcpyHostToDevice), "cudaMemcpy dX") ||
        !checkCuda(cudaMemcpy(dA, repeated.data(), scalarBytes, cudaMemcpyHostToDevice), "cudaMemcpy dA")) {
        cudaFree(dX);
        cudaFree(dA);
        cudaFree(dH);
        cudaFree(dY);
        return false;
    }

    curve25519_elligator2_u_kernel<<<blocks, threads>>>(dX, dH, n);
    curve25519_scalarmult_fe_kernel<<<blocks, threads>>>(dA, dH, dY, n);
    if (!checkCuda(cudaGetLastError(), "gpuMapAndBlind kernel launch") ||
        !checkCuda(cudaDeviceSynchronize(), "gpuMapAndBlind cudaDeviceSynchronize") ||
        !checkCuda(cudaMemcpy(hY.data(), dY, feBytes, cudaMemcpyDeviceToHost), "cudaMemcpy dY")) {
        cudaFree(dX);
        cudaFree(dA);
        cudaFree(dH);
        cudaFree(dY);
        return false;
    }

    cudaFree(dX);
    cudaFree(dA);
    cudaFree(dH);
    cudaFree(dY);

    for (size_t i = 0; i < n; ++i) {
        encodeU25519Le(hY[i], out_u_bytes->data() + i * 32u);
    }
    return true;
}

inline bool gpuApplyScalarToUBytes(const std::vector<uint8_t>& u_bytes, const uint8_t scalar[32],
                                   std::vector<uint8_t>* out_u_bytes) {
    if (u_bytes.size() % 32u != 0) return false;
    const size_t n = u_bytes.size() / 32u;
    out_u_bytes->assign(u_bytes.size(), 0);
    std::vector<uint8_t> repeated = repeatScalar(scalar, n);

    uint8_t* dScalar = nullptr;
    uint8_t* dIn = nullptr;
    uint8_t* dOut = nullptr;
    const size_t bytes = u_bytes.size();
    const int threads = 256;
    const int blocks = static_cast<int>((n + static_cast<size_t>(threads) - 1u) / static_cast<size_t>(threads));

    if (!checkCuda(cudaMalloc(&dScalar, bytes), "cudaMalloc dScalar") ||
        !checkCuda(cudaMalloc(&dIn, bytes), "cudaMalloc dIn") ||
        !checkCuda(cudaMalloc(&dOut, bytes), "cudaMalloc dOut")) {
        cudaFree(dScalar);
        cudaFree(dIn);
        cudaFree(dOut);
        return false;
    }

    if (!checkCuda(cudaMemcpy(dScalar, repeated.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy dScalar") ||
        !checkCuda(cudaMemcpy(dIn, u_bytes.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy dIn")) {
        cudaFree(dScalar);
        cudaFree(dIn);
        cudaFree(dOut);
        return false;
    }

    curve25519_scalarmult_u_kernel<<<blocks, threads>>>(dScalar, dIn, dOut, n);
    if (!checkCuda(cudaGetLastError(), "gpuApplyScalarToUBytes kernel launch") ||
        !checkCuda(cudaDeviceSynchronize(), "gpuApplyScalarToUBytes cudaDeviceSynchronize") ||
        !checkCuda(cudaMemcpy(out_u_bytes->data(), dOut, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy dOut")) {
        cudaFree(dScalar);
        cudaFree(dIn);
        cudaFree(dOut);
        return false;
    }

    cudaFree(dScalar);
    cudaFree(dIn);
    cudaFree(dOut);
    return true;
}
