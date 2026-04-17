#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

struct fe25519 {
    uint32_t limb[8];
};

extern "C" __global__ void fe25519_mul_kernel(const fe25519* a, const fe25519* b, fe25519* out, size_t n);
extern "C" __global__ void fe25519_sqrt_kernel(const fe25519* a, fe25519* out, uint8_t* ok, size_t n);

namespace {

constexpr uint32_t kP[8] = {
    0xFFFFFFEDu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x7FFFFFFFu,
};

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

int cmpGeP(const fe25519& a) {
    for (int i = 7; i >= 0; --i) {
        if (a.limb[i] > kP[i]) return 1;
        if (a.limb[i] < kP[i]) return 0;
    }
    return 1;
}

void subP(fe25519* a) {
    uint64_t borrow = 0;
    for (int i = 0; i < 8; ++i) {
        const uint64_t cur = a->limb[i];
        const uint64_t sub = static_cast<uint64_t>(kP[i]) + borrow;
        a->limb[i] = static_cast<uint32_t>(cur - sub);
        borrow = (cur < sub) ? 1 : 0;
    }
}

void canonicalize(fe25519* a) {
    a->limb[7] &= 0x7FFFFFFFu;
    if (cmpGeP(*a)) subP(a);
}

fe25519 randomFieldElement(std::mt19937_64* rng) {
    fe25519 out{};
    for (int i = 0; i < 8; ++i) {
        out.limb[i] = static_cast<uint32_t>((*rng)());
    }
    canonicalize(&out);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    int n = 4096;
    int iters = 5;
    if (argc > 3 || (argc >= 2 && !parsePositiveInt(argv[1], &n)) || (argc == 3 && !parsePositiveInt(argv[2], &iters))) {
        std::cerr << "Usage: " << argv[0] << " [num_elements] [timed_iterations]\n";
        return 1;
    }

    std::mt19937_64 rng(42);
    std::vector<fe25519> hX(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        hX[static_cast<size_t>(i)] = randomFieldElement(&rng);
    }

    std::vector<fe25519> hSquares(static_cast<size_t>(n));
    std::vector<fe25519> hRoots(static_cast<size_t>(n));
    std::vector<fe25519> hResquared(static_cast<size_t>(n));
    std::vector<uint8_t> hOk(static_cast<size_t>(n), 0);

    fe25519 *dX = nullptr, *dSquares = nullptr, *dRoots = nullptr, *dResquared = nullptr;
    uint8_t* dOk = nullptr;

    const size_t feBytes = static_cast<size_t>(n) * sizeof(fe25519);
    const size_t okBytes = static_cast<size_t>(n) * sizeof(uint8_t);

    if (!checkCuda(cudaFree(nullptr), "cudaFree(0) warmup")) {
        return 1;
    }
    if (!checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup")) {
        return 1;
    }

    const auto cudaStart = std::chrono::high_resolution_clock::now();

    if (!checkCuda(cudaMalloc(&dX, feBytes), "cudaMalloc dX") ||
        !checkCuda(cudaMalloc(&dSquares, feBytes), "cudaMalloc dSquares") ||
        !checkCuda(cudaMalloc(&dRoots, feBytes), "cudaMalloc dRoots") ||
        !checkCuda(cudaMalloc(&dResquared, feBytes), "cudaMalloc dResquared") ||
        !checkCuda(cudaMalloc(&dOk, okBytes), "cudaMalloc dOk")) {
        cudaFree(dX);
        cudaFree(dSquares);
        cudaFree(dRoots);
        cudaFree(dResquared);
        cudaFree(dOk);
        return 1;
    }

    if (!checkCuda(cudaMemcpy(dX, hX.data(), feBytes, cudaMemcpyHostToDevice), "cudaMemcpy hX")) {
        cudaFree(dX);
        cudaFree(dSquares);
        cudaFree(dRoots);
        cudaFree(dResquared);
        cudaFree(dOk);
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
        cudaFree(dX);
        cudaFree(dSquares);
        cudaFree(dRoots);
        cudaFree(dResquared);
        cudaFree(dOk);
        return 1;
    }

    // Warmup iteration: excluded from timing.
    fe25519_mul_kernel<<<blocks, threads>>>(dX, dX, dSquares, static_cast<size_t>(n));
    fe25519_sqrt_kernel<<<blocks, threads>>>(dSquares, dRoots, dOk, static_cast<size_t>(n));
    fe25519_mul_kernel<<<blocks, threads>>>(dRoots, dRoots, dResquared, static_cast<size_t>(n));
    if (!checkCuda(cudaGetLastError(), "warmup kernel launch") ||
        !checkCuda(cudaDeviceSynchronize(), "warmup cudaDeviceSynchronize")) {
        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelStop);
        cudaFree(dX);
        cudaFree(dSquares);
        cudaFree(dRoots);
        cudaFree(dResquared);
        cudaFree(dOk);
        return 1;
    }

    if (!checkCuda(cudaEventRecord(kernelStart), "cudaEventRecord kernelStart")) {
        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelStop);
        cudaFree(dX);
        cudaFree(dSquares);
        cudaFree(dRoots);
        cudaFree(dResquared);
        cudaFree(dOk);
        return 1;
    }

    for (int iter = 0; iter < iters; ++iter) {
        // squares = x^2
        fe25519_mul_kernel<<<blocks, threads>>>(dX, dX, dSquares, static_cast<size_t>(n));
        // roots = sqrt(squares)
        fe25519_sqrt_kernel<<<blocks, threads>>>(dSquares, dRoots, dOk, static_cast<size_t>(n));
        // re-squared roots for validation
        fe25519_mul_kernel<<<blocks, threads>>>(dRoots, dRoots, dResquared, static_cast<size_t>(n));
    }

    if (!checkCuda(cudaGetLastError(), "kernel launch") ||
        !checkCuda(cudaEventRecord(kernelStop), "cudaEventRecord kernelStop") ||
        !checkCuda(cudaEventSynchronize(kernelStop), "cudaEventSynchronize kernelStop")) {
        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelStop);
        cudaFree(dX);
        cudaFree(dSquares);
        cudaFree(dRoots);
        cudaFree(dResquared);
        cudaFree(dOk);
        return 1;
    }

    float kernelMs = 0.0f;
    if (!checkCuda(cudaEventElapsedTime(&kernelMs, kernelStart, kernelStop), "cudaEventElapsedTime")) {
        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelStop);
        cudaFree(dX);
        cudaFree(dSquares);
        cudaFree(dRoots);
        cudaFree(dResquared);
        cudaFree(dOk);
        return 1;
    }

    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelStop);

    if (!checkCuda(cudaMemcpy(hSquares.data(), dSquares, feBytes, cudaMemcpyDeviceToHost), "cudaMemcpy hSquares") ||
        !checkCuda(cudaMemcpy(hRoots.data(), dRoots, feBytes, cudaMemcpyDeviceToHost), "cudaMemcpy hRoots") ||
        !checkCuda(cudaMemcpy(hResquared.data(), dResquared, feBytes, cudaMemcpyDeviceToHost), "cudaMemcpy hResquared") ||
        !checkCuda(cudaMemcpy(hOk.data(), dOk, okBytes, cudaMemcpyDeviceToHost), "cudaMemcpy hOk")) {
        cudaFree(dX);
        cudaFree(dSquares);
        cudaFree(dRoots);
        cudaFree(dResquared);
        cudaFree(dOk);
        return 1;
    }

    const auto cudaEnd = std::chrono::high_resolution_clock::now();
    const double cudaTotalMs = std::chrono::duration<double, std::milli>(cudaEnd - cudaStart).count();

    cudaFree(dX);
    cudaFree(dSquares);
    cudaFree(dRoots);
    cudaFree(dResquared);
    cudaFree(dOk);

    for (int i = 0; i < n; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (hOk[idx] == 0) {
            std::cerr << "sqrt reported non-residue at index " << i << " for a constructed square\n";
            return 1;
        }
        if (std::memcmp(&hResquared[idx], &hSquares[idx], sizeof(fe25519)) != 0) {
            std::cerr << "sqrt verification mismatch at index " << i << '\n';
            return 1;
        }
    }

    std::cout << "fe25519 sqrt (Tonelli-Shanks) test passed for " << n << " constructed squares\n";
    std::cout << "Timed iterations (warmup excluded): " << iters << "\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CUDA total time (malloc+H2D+kernels+D2H): " << cudaTotalMs << " ms\n";
    std::cout << "Kernel time total (square+sqrt+resquare): " << kernelMs << " ms\n";
    std::cout << "Kernel time avg per iteration: " << (kernelMs / static_cast<float>(iters)) << " ms\n";
    return 0;
}
