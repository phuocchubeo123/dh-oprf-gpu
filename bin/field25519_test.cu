#include <cuda_runtime.h>
#include <openssl/bn.h>

#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct fe25519 {
    uint32_t limb[8];
};

extern "C" __global__ void fe25519_add_kernel(const fe25519* a, const fe25519* b, fe25519* out, size_t n);
extern "C" __global__ void fe25519_mul_kernel(const fe25519* a, const fe25519* b, fe25519* out, size_t n);

namespace {

bool checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(result) << '\n';
        return false;
    }
    return true;
}

bool parsePositiveInt(const char* raw, int* out) {
    if (raw == nullptr || raw[0] == '\0') return false;
    char* end = nullptr;
    const long long parsed = std::strtoll(raw, &end, 10);
    if (*end != '\0' || parsed <= 0 || parsed > 1000000000LL) return false;
    *out = static_cast<int>(parsed);
    return true;
}

BIGNUM* modulusP() {
    BIGNUM* p = BN_new();
    if (p == nullptr) return nullptr;
    // p = 2^255 - 19
    if (BN_hex2bn(&p, "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFED") == 0) {
        BN_free(p);
        return nullptr;
    }
    return p;
}

bool bnToFe(const BIGNUM* bn, fe25519* out) {
    unsigned char be[32] = {0};
    if (BN_bn2binpad(bn, be, 32) != 32) return false;
    for (int i = 0; i < 8; ++i) {
        const int base = 31 - (i * 4);
        out->limb[i] = static_cast<uint32_t>(be[base]) |
                       (static_cast<uint32_t>(be[base - 1]) << 8) |
                       (static_cast<uint32_t>(be[base - 2]) << 16) |
                       (static_cast<uint32_t>(be[base - 3]) << 24);
    }
    return true;
}

BIGNUM* feToBn(const fe25519& x) {
    unsigned char be[32];
    for (int i = 0; i < 8; ++i) {
        const uint32_t w = x.limb[i];
        const int base = 31 - (i * 4);
        be[base] = static_cast<unsigned char>(w & 0xFFu);
        be[base - 1] = static_cast<unsigned char>((w >> 8) & 0xFFu);
        be[base - 2] = static_cast<unsigned char>((w >> 16) & 0xFFu);
        be[base - 3] = static_cast<unsigned char>((w >> 24) & 0xFFu);
    }
    return BN_bin2bn(be, 32, nullptr);
}

std::string feHex(const fe25519& x) {
    BIGNUM* bn = feToBn(x);
    if (bn == nullptr) return "<bn_err>";
    char* hex = BN_bn2hex(bn);
    std::string out = (hex != nullptr) ? hex : "<hex_err>";
    OPENSSL_free(hex);
    BN_free(bn);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    int n = 4096;
    if (argc > 2 || (argc == 2 && !parsePositiveInt(argv[1], &n))) {
        std::cerr << "Usage: " << argv[0] << " [num_elements]\n";
        return 1;
    }

    BN_CTX* ctx = BN_CTX_new();
    BIGNUM* p = modulusP();
    BIGNUM* aBn = BN_new();
    BIGNUM* bBn = BN_new();
    BIGNUM* addBn = BN_new();
    BIGNUM* mulBn = BN_new();
    if (ctx == nullptr || p == nullptr || aBn == nullptr || bBn == nullptr || addBn == nullptr || mulBn == nullptr) {
        std::cerr << "OpenSSL BN allocation failed\n";
        BN_CTX_free(ctx);
        BN_free(p);
        BN_free(aBn);
        BN_free(bBn);
        BN_free(addBn);
        BN_free(mulBn);
        return 1;
    }

    std::vector<fe25519> hA(static_cast<size_t>(n));
    std::vector<fe25519> hB(static_cast<size_t>(n));
    std::vector<fe25519> hAddRef(static_cast<size_t>(n));
    std::vector<fe25519> hMulRef(static_cast<size_t>(n));

    for (int i = 0; i < n; ++i) {
        if (BN_rand_range(aBn, p) != 1 || BN_rand_range(bBn, p) != 1) {
            std::cerr << "BN_rand_range failed at " << i << '\n';
            BN_CTX_free(ctx);
            BN_free(p);
            BN_free(aBn);
            BN_free(bBn);
            BN_free(addBn);
            BN_free(mulBn);
            return 1;
        }
        if (BN_mod_add(addBn, aBn, bBn, p, ctx) != 1 || BN_mod_mul(mulBn, aBn, bBn, p, ctx) != 1) {
            std::cerr << "BN_mod_* failed at " << i << '\n';
            BN_CTX_free(ctx);
            BN_free(p);
            BN_free(aBn);
            BN_free(bBn);
            BN_free(addBn);
            BN_free(mulBn);
            return 1;
        }

        if (!bnToFe(aBn, &hA[static_cast<size_t>(i)]) || !bnToFe(bBn, &hB[static_cast<size_t>(i)]) ||
            !bnToFe(addBn, &hAddRef[static_cast<size_t>(i)]) || !bnToFe(mulBn, &hMulRef[static_cast<size_t>(i)])) {
            std::cerr << "bnToFe conversion failed at " << i << '\n';
            BN_CTX_free(ctx);
            BN_free(p);
            BN_free(aBn);
            BN_free(bBn);
            BN_free(addBn);
            BN_free(mulBn);
            return 1;
        }
    }

    BN_CTX_free(ctx);
    BN_free(p);
    BN_free(aBn);
    BN_free(bBn);
    BN_free(addBn);
    BN_free(mulBn);

    fe25519 *dA = nullptr, *dB = nullptr, *dAddOut = nullptr, *dMulOut = nullptr;
    std::vector<fe25519> hAddOut(static_cast<size_t>(n));
    std::vector<fe25519> hMulOut(static_cast<size_t>(n));
    const auto cudaStart = std::chrono::high_resolution_clock::now();

    const size_t bytes = static_cast<size_t>(n) * sizeof(fe25519);
    if (!checkCuda(cudaMalloc(&dA, bytes), "cudaMalloc dA") || !checkCuda(cudaMalloc(&dB, bytes), "cudaMalloc dB") ||
        !checkCuda(cudaMalloc(&dAddOut, bytes), "cudaMalloc dAddOut") ||
        !checkCuda(cudaMalloc(&dMulOut, bytes), "cudaMalloc dMulOut")) {
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dAddOut);
        cudaFree(dMulOut);
        return 1;
    }

    if (!checkCuda(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy hA") ||
        !checkCuda(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy hB")) {
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dAddOut);
        cudaFree(dMulOut);
        return 1;
    }

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!checkCuda(cudaEventCreate(&start), "cudaEventCreate start") ||
        !checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop")) {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dAddOut);
        cudaFree(dMulOut);
        return 1;
    }

    if (!checkCuda(cudaEventRecord(start), "cudaEventRecord start") ) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dAddOut);
        cudaFree(dMulOut);
        return 1;
    }

    fe25519_add_kernel<<<blocks, threads>>>(dA, dB, dAddOut, static_cast<size_t>(n));
    fe25519_mul_kernel<<<blocks, threads>>>(dA, dB, dMulOut, static_cast<size_t>(n));

    if (!checkCuda(cudaGetLastError(), "kernel launch") || !checkCuda(cudaEventRecord(stop), "cudaEventRecord stop") ||
        !checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop")) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dAddOut);
        cudaFree(dMulOut);
        return 1;
    }

    float kernelMs = 0.0f;
    if (!checkCuda(cudaEventElapsedTime(&kernelMs, start, stop), "cudaEventElapsedTime")) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dAddOut);
        cudaFree(dMulOut);
        return 1;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (!checkCuda(cudaMemcpy(hAddOut.data(), dAddOut, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy add out") ||
        !checkCuda(cudaMemcpy(hMulOut.data(), dMulOut, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy mul out")) {
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dAddOut);
        cudaFree(dMulOut);
        return 1;
    }
    const auto cudaEnd = std::chrono::high_resolution_clock::now();
    const double cudaTotalMs = std::chrono::duration<double, std::milli>(cudaEnd - cudaStart).count();

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dAddOut);
    cudaFree(dMulOut);

    for (int i = 0; i < n; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (std::memcmp(&hAddOut[idx], &hAddRef[idx], sizeof(fe25519)) != 0) {
            std::cerr << "add mismatch at index " << i << '\n';
            std::cerr << "  got: " << feHex(hAddOut[idx]) << '\n';
            std::cerr << "  exp: " << feHex(hAddRef[idx]) << '\n';
            return 1;
        }
        if (std::memcmp(&hMulOut[idx], &hMulRef[idx], sizeof(fe25519)) != 0) {
            std::cerr << "mul mismatch at index " << i << '\n';
            std::cerr << "  got: " << feHex(hMulOut[idx]) << '\n';
            std::cerr << "  exp: " << feHex(hMulRef[idx]) << '\n';
            return 1;
        }
    }

    std::cout << "fe25519 add/mul test passed for " << n << " random vectors\n";
    std::cout << std::fixed << std::setprecision(6) << "CUDA total time (malloc+H2D+kernel+D2H): " << cudaTotalMs << " ms\n";
    std::cout << std::fixed << std::setprecision(6) << "Kernel time (add+mul): " << kernelMs << " ms\n";
    return 0;
}
