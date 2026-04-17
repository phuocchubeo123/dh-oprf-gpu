#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

extern "C" __global__ void curve25519_scalarmult_u_kernel(const uint8_t* scalar32, const uint8_t* u32, uint8_t* out32,
                                                            size_t n);

extern "C" int sodium_init(void);
extern "C" int crypto_scalarmult_curve25519(unsigned char* q, const unsigned char* n, const unsigned char* p);

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

std::string hex32(const uint8_t* in) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (int i = 31; i >= 0; --i) {
        oss << std::setw(2) << static_cast<unsigned>(in[i]);
    }
    return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
    int n = 4096;
    int sampleChecks = 512;
    if (argc > 3 || (argc >= 2 && !parsePositiveInt(argv[1], &n)) || (argc == 3 && !parsePositiveInt(argv[2], &sampleChecks))) {
        std::cerr << "Usage: " << argv[0] << " [num_elements] [cpu_sample_checks]\n";
        return 1;
    }
    sampleChecks = std::min(sampleChecks, n);

    if (sodium_init() < 0) {
        std::cerr << "sodium_init failed\n";
        return 1;
    }

    std::vector<uint8_t> hScalars(static_cast<size_t>(n) * 32u);
    std::vector<uint8_t> hU(static_cast<size_t>(n) * 32u);
    std::vector<uint8_t> hOut(static_cast<size_t>(n) * 32u);

    std::mt19937_64 rng(42);
    for (size_t i = 0; i < hScalars.size(); ++i) hScalars[i] = static_cast<uint8_t>(rng() & 0xFFu);
    for (size_t i = 0; i < hU.size(); ++i) hU[i] = static_cast<uint8_t>(rng() & 0xFFu);

    // Clear top bit in u input to match RFC 7748 decode behavior.
    for (int i = 0; i < n; ++i) {
        hU[static_cast<size_t>(i) * 32u + 31u] &= 0x7Fu;
    }

    uint8_t *dScalars = nullptr, *dU = nullptr, *dOut = nullptr;
    const size_t bytes = static_cast<size_t>(n) * 32u;

    const auto cudaStart = std::chrono::high_resolution_clock::now();

    if (!checkCuda(cudaMalloc(&dScalars, bytes), "cudaMalloc dScalars") ||
        !checkCuda(cudaMalloc(&dU, bytes), "cudaMalloc dU") ||
        !checkCuda(cudaMalloc(&dOut, bytes), "cudaMalloc dOut")) {
        cudaFree(dScalars);
        cudaFree(dU);
        cudaFree(dOut);
        return 1;
    }

    if (!checkCuda(cudaMemcpy(dScalars, hScalars.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy hScalars") ||
        !checkCuda(cudaMemcpy(dU, hU.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy hU")) {
        cudaFree(dScalars);
        cudaFree(dU);
        cudaFree(dOut);
        return 1;
    }

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    cudaEvent_t kernelStart = nullptr;
    cudaEvent_t kernelStop = nullptr;
    if (!checkCuda(cudaEventCreate(&kernelStart), "cudaEventCreate kernelStart") ||
        !checkCuda(cudaEventCreate(&kernelStop), "cudaEventCreate kernelStop")) {
        if (kernelStart) cudaEventDestroy(kernelStart);
        if (kernelStop) cudaEventDestroy(kernelStop);
        cudaFree(dScalars);
        cudaFree(dU);
        cudaFree(dOut);
        return 1;
    }

    if (!checkCuda(cudaEventRecord(kernelStart), "cudaEventRecord kernelStart")) {
        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelStop);
        cudaFree(dScalars);
        cudaFree(dU);
        cudaFree(dOut);
        return 1;
    }

    curve25519_scalarmult_u_kernel<<<blocks, threads>>>(dScalars, dU, dOut, static_cast<size_t>(n));

    if (!checkCuda(cudaGetLastError(), "kernel launch") ||
        !checkCuda(cudaEventRecord(kernelStop), "cudaEventRecord kernelStop") ||
        !checkCuda(cudaEventSynchronize(kernelStop), "cudaEventSynchronize kernelStop")) {
        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelStop);
        cudaFree(dScalars);
        cudaFree(dU);
        cudaFree(dOut);
        return 1;
    }

    float kernelMs = 0.0f;
    if (!checkCuda(cudaEventElapsedTime(&kernelMs, kernelStart, kernelStop), "cudaEventElapsedTime")) {
        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelStop);
        cudaFree(dScalars);
        cudaFree(dU);
        cudaFree(dOut);
        return 1;
    }
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelStop);

    if (!checkCuda(cudaMemcpy(hOut.data(), dOut, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy dOut")) {
        cudaFree(dScalars);
        cudaFree(dU);
        cudaFree(dOut);
        return 1;
    }

    const auto cudaEnd = std::chrono::high_resolution_clock::now();
    const double cudaTotalMs = std::chrono::duration<double, std::milli>(cudaEnd - cudaStart).count();

    cudaFree(dScalars);
    cudaFree(dU);
    cudaFree(dOut);

    std::vector<int> indices(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) indices[static_cast<size_t>(i)] = i;
    std::shuffle(indices.begin(), indices.end(), std::mt19937(1337));

    const auto cpuStart = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < sampleChecks; ++k) {
        const int idx = indices[static_cast<size_t>(k)];
        const uint8_t* scalar = hScalars.data() + static_cast<size_t>(idx) * 32u;
        const uint8_t* u = hU.data() + static_cast<size_t>(idx) * 32u;
        const uint8_t* gpu = hOut.data() + static_cast<size_t>(idx) * 32u;

        uint8_t ref[32];
        if (crypto_scalarmult_curve25519(ref, scalar, u) != 0) {
            std::cerr << "crypto_scalarmult_curve25519 failed at index " << idx << '\n';
            return 1;
        }

        if (std::memcmp(gpu, ref, 32) != 0) {
            std::cerr << "curve25519 scalar-mult mismatch at sampled index " << idx << '\n';
            std::cerr << "  GPU: " << hex32(gpu) << '\n';
            std::cerr << "  REF: " << hex32(ref) << '\n';
            return 1;
        }
    }
    const auto cpuEnd = std::chrono::high_resolution_clock::now();
    const double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    std::cout << "curve25519 scalar multiplication comparison passed\n";
    std::cout << "Total elements: " << n << ", CPU reference sample checks: " << sampleChecks << "\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CUDA total time (malloc+H2D+kernel+D2H): " << cudaTotalMs << " ms\n";
    std::cout << "GPU kernel time: " << kernelMs << " ms\n";
    std::cout << "CPU reference sample time (libsodium): " << cpuMs << " ms\n";
    return 0;
}
