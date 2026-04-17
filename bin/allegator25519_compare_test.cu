#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

struct fe25519 {
    uint32_t limb[8];
};

extern "C" __global__ void curve25519_elligator2_u_kernel(const uint8_t* uniform32, fe25519* out_u, size_t n);

extern "C" int sodium_init(void);
extern "C" int crypto_core_ed25519_from_uniform(unsigned char* p, const unsigned char* r);
extern "C" int crypto_sign_ed25519_pk_to_curve25519(unsigned char* x25519_pk, const unsigned char* ed25519_pk);

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

fe25519 decodeU25519Le(const uint8_t in[32]) {
    fe25519 out{};
    for (int i = 0; i < 8; ++i) {
        const int o = i * 4;
        out.limb[i] = static_cast<uint32_t>(in[o]) |
                      (static_cast<uint32_t>(in[o + 1]) << 8) |
                      (static_cast<uint32_t>(in[o + 2]) << 16) |
                      (static_cast<uint32_t>(in[o + 3]) << 24);
    }
    out.limb[7] &= 0x7FFFFFFFu;
    return out;
}

bool feEqual(const fe25519& a, const fe25519& b) {
    return std::memcmp(&a, &b, sizeof(fe25519)) == 0;
}

std::string feHex(const fe25519& x) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (int i = 7; i >= 0; --i) {
        oss << std::setw(8) << x.limb[i];
    }
    return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
    int n = 4096;
    int sampleChecks = 256;
    if (argc > 3 || (argc >= 2 && !parsePositiveInt(argv[1], &n)) || (argc == 3 && !parsePositiveInt(argv[2], &sampleChecks))) {
        std::cerr << "Usage: " << argv[0] << " [num_elements] [cpu_sample_checks]\n";
        return 1;
    }
    sampleChecks = std::min(sampleChecks, n);

    if (sodium_init() < 0) {
        std::cerr << "sodium_init failed\n";
        return 1;
    }

    std::vector<uint8_t> hInputs(static_cast<size_t>(n) * 32u);
    std::vector<fe25519> hGpuU(static_cast<size_t>(n));

    std::mt19937_64 rng(42);
    for (size_t i = 0; i < hInputs.size(); ++i) {
        hInputs[i] = static_cast<uint8_t>(rng() & 0xFFu);
    }

    uint8_t* dInputs = nullptr;
    fe25519* dOut = nullptr;

    const size_t inBytes = hInputs.size() * sizeof(uint8_t);
    const size_t outBytes = hGpuU.size() * sizeof(fe25519);

    const auto cudaStart = std::chrono::high_resolution_clock::now();

    if (!checkCuda(cudaMalloc(&dInputs, inBytes), "cudaMalloc dInputs") ||
        !checkCuda(cudaMalloc(&dOut, outBytes), "cudaMalloc dOut")) {
        cudaFree(dInputs);
        cudaFree(dOut);
        return 1;
    }

    if (!checkCuda(cudaMemcpy(dInputs, hInputs.data(), inBytes, cudaMemcpyHostToDevice), "cudaMemcpy hInputs")) {
        cudaFree(dInputs);
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
        cudaFree(dInputs);
        cudaFree(dOut);
        return 1;
    }

    if (!checkCuda(cudaEventRecord(kernelStart), "cudaEventRecord kernelStart")) {
        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelStop);
        cudaFree(dInputs);
        cudaFree(dOut);
        return 1;
    }

    curve25519_elligator2_u_kernel<<<blocks, threads>>>(dInputs, dOut, static_cast<size_t>(n));

    if (!checkCuda(cudaGetLastError(), "kernel launch") ||
        !checkCuda(cudaEventRecord(kernelStop), "cudaEventRecord kernelStop") ||
        !checkCuda(cudaEventSynchronize(kernelStop), "cudaEventSynchronize kernelStop")) {
        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelStop);
        cudaFree(dInputs);
        cudaFree(dOut);
        return 1;
    }

    float kernelMs = 0.0f;
    if (!checkCuda(cudaEventElapsedTime(&kernelMs, kernelStart, kernelStop), "cudaEventElapsedTime")) {
        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelStop);
        cudaFree(dInputs);
        cudaFree(dOut);
        return 1;
    }
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelStop);

    if (!checkCuda(cudaMemcpy(hGpuU.data(), dOut, outBytes, cudaMemcpyDeviceToHost), "cudaMemcpy dOut")) {
        cudaFree(dInputs);
        cudaFree(dOut);
        return 1;
    }

    const auto cudaEnd = std::chrono::high_resolution_clock::now();
    const double cudaTotalMs = std::chrono::duration<double, std::milli>(cudaEnd - cudaStart).count();

    cudaFree(dInputs);
    cudaFree(dOut);

    // CPU reference only on sampled indices.
    std::vector<int> indices(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) indices[static_cast<size_t>(i)] = i;
    std::shuffle(indices.begin(), indices.end(), std::mt19937(1337));

    const auto cpuRefStart = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < sampleChecks; ++k) {
        const int idx = indices[static_cast<size_t>(k)];
        const uint8_t* in = hInputs.data() + static_cast<size_t>(idx) * 32u;

        unsigned char ed[32];
        unsigned char urefBytes[32];
        if (crypto_core_ed25519_from_uniform(ed, in) != 0) {
            std::cerr << "crypto_core_ed25519_from_uniform failed at index " << idx << '\n';
            return 1;
        }
        if (crypto_sign_ed25519_pk_to_curve25519(urefBytes, ed) != 0) {
            std::cerr << "crypto_sign_ed25519_pk_to_curve25519 failed at index " << idx << '\n';
            return 1;
        }

        const fe25519 ref = decodeU25519Le(urefBytes);
        if (!feEqual(ref, hGpuU[static_cast<size_t>(idx)])) {
            std::cerr << "Allegator mismatch at sampled index " << idx << '\n';
            std::cerr << "  GPU u: " << feHex(hGpuU[static_cast<size_t>(idx)]) << '\n';
            std::cerr << "  REF u: " << feHex(ref) << '\n';
            return 1;
        }
    }
    const auto cpuRefEnd = std::chrono::high_resolution_clock::now();
    const double cpuRefMs = std::chrono::duration<double, std::milli>(cpuRefEnd - cpuRefStart).count();

    std::cout << "Allegator2 curve25519-u comparison passed\n";
    std::cout << "Total elements: " << n << ", CPU reference sample checks: " << sampleChecks << "\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CUDA total time (malloc+H2D+kernel+D2H): " << cudaTotalMs << " ms\n";
    std::cout << "GPU kernel time: " << kernelMs << " ms\n";
    std::cout << "CPU reference sample time (libsodium): " << cpuRefMs << " ms\n";
    return 0;
}
