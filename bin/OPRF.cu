#include <cuda_runtime.h>
#include <openssl/bn.h>

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

struct fe25519 {
    uint32_t limb[8];
};

extern "C" __global__ void curve25519_elligator2_u_kernel(const uint8_t* uniform32, fe25519* out_u, size_t n);
extern "C" __global__ void curve25519_scalarmult_fe_kernel(const uint8_t* scalar32, const fe25519* u_in,
                                                           fe25519* out_u, size_t n);

extern "C" int sodium_init(void);
extern "C" int crypto_core_ed25519_from_uniform(unsigned char* p, const unsigned char* r);
extern "C" int crypto_sign_ed25519_pk_to_curve25519(unsigned char* x25519_pk, const unsigned char* ed25519_pk);
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

BIGNUM* subgroupOrderL() {
    BIGNUM* l = nullptr;
    // l = 2^252 + 27742317777372353535851937790883648493
    if (BN_hex2bn(&l, "1000000000000000000000000000000014DEF9DEA2F79CD65812631A5CF5D3ED") == 0) {
        return nullptr;
    }
    return l;
}

bool bnToLe32(const BIGNUM* bn, uint8_t out[32]) {
    uint8_t be[32] = {0};
    if (BN_bn2binpad(bn, be, 32) != 32) return false;
    for (int i = 0; i < 32; ++i) out[i] = be[31 - i];
    return true;
}

std::string feHex(const fe25519& x) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (int i = 7; i >= 0; --i) {
        oss << std::setw(8) << x.limb[i];
    }
    return oss.str();
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

}  // namespace

int main(int argc, char** argv) {
    int n = 4096;
    int sampleChecks = 256;
    if (argc > 3 || (argc >= 2 && !parsePositiveInt(argv[1], &n)) || (argc == 3 && !parsePositiveInt(argv[2], &sampleChecks))) {
        std::cerr << "Usage: " << argv[0] << " [num_inputs] [cpu_sample_checks]\n";
        return 1;
    }
    sampleChecks = std::min(sampleChecks, n);

    if (sodium_init() < 0) {
        std::cerr << "sodium_init failed\n";
        return 1;
    }

    BIGNUM* l = subgroupOrderL();
    BIGNUM* a_bn = BN_new();
    BIGNUM* b_bn = BN_new();
    if (l == nullptr || a_bn == nullptr || b_bn == nullptr) {
        std::cerr << "OpenSSL BN allocation failed\n";
        BN_free(l);
        BN_free(a_bn);
        BN_free(b_bn);
        return 1;
    }

    if (BN_rand_range(a_bn, l) != 1 || BN_rand_range(b_bn, l) != 1) {
        std::cerr << "BN_rand_range failed for a/b\n";
        BN_free(l);
        BN_free(a_bn);
        BN_free(b_bn);
        return 1;
    }

    uint8_t a_bytes[32] = {0};
    uint8_t b_bytes[32] = {0};
    if (!bnToLe32(a_bn, a_bytes) || !bnToLe32(b_bn, b_bytes)) {
        std::cerr << "BN to scalar conversion failed\n";
        BN_free(l);
        BN_free(a_bn);
        BN_free(b_bn);
        return 1;
    }

    BN_free(l);
    BN_free(a_bn);
    BN_free(b_bn);

    std::vector<uint8_t> hX(static_cast<size_t>(n) * 32u);
    std::vector<uint8_t> hA(static_cast<size_t>(n) * 32u);
    std::vector<uint8_t> hB(static_cast<size_t>(n) * 32u);

    std::mt19937_64 rng(42);
    for (size_t i = 0; i < hX.size(); ++i) hX[i] = static_cast<uint8_t>(rng() & 0xFFu);

    for (int i = 0; i < n; ++i) {
        std::memcpy(hA.data() + static_cast<size_t>(i) * 32u, a_bytes, 32u);
        std::memcpy(hB.data() + static_cast<size_t>(i) * 32u, b_bytes, 32u);
    }

    uint8_t *dX = nullptr, *dA = nullptr, *dB = nullptr;
    fe25519 *dH = nullptr, *dY = nullptr, *dZ = nullptr;

    const size_t scalarBytes = static_cast<size_t>(n) * 32u;
    const size_t feBytes = static_cast<size_t>(n) * sizeof(fe25519);

    // One-time runtime/context warmup (excluded from measured timings).
    if (!checkCuda(cudaFree(nullptr), "cudaFree(0) warmup") ||
        !checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup")) {
        return 1;
    }

    const auto cudaStart = std::chrono::high_resolution_clock::now();

    if (!checkCuda(cudaMalloc(&dX, scalarBytes), "cudaMalloc dX") ||
        !checkCuda(cudaMalloc(&dA, scalarBytes), "cudaMalloc dA") ||
        !checkCuda(cudaMalloc(&dB, scalarBytes), "cudaMalloc dB") ||
        !checkCuda(cudaMalloc(&dH, feBytes), "cudaMalloc dH") ||
        !checkCuda(cudaMalloc(&dY, feBytes), "cudaMalloc dY") ||
        !checkCuda(cudaMalloc(&dZ, feBytes), "cudaMalloc dZ")) {
        cudaFree(dX); cudaFree(dA); cudaFree(dB); cudaFree(dH); cudaFree(dY); cudaFree(dZ);
        return 1;
    }

    if (!checkCuda(cudaMemcpy(dX, hX.data(), scalarBytes, cudaMemcpyHostToDevice), "cudaMemcpy hX") ||
        !checkCuda(cudaMemcpy(dA, hA.data(), scalarBytes, cudaMemcpyHostToDevice), "cudaMemcpy hA") ||
        !checkCuda(cudaMemcpy(dB, hB.data(), scalarBytes, cudaMemcpyHostToDevice), "cudaMemcpy hB")) {
        cudaFree(dX); cudaFree(dA); cudaFree(dB); cudaFree(dH); cudaFree(dY); cudaFree(dZ);
        return 1;
    }

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    cudaEvent_t kStart = nullptr;
    cudaEvent_t kStop = nullptr;
    if (!checkCuda(cudaEventCreate(&kStart), "cudaEventCreate kStart") ||
        !checkCuda(cudaEventCreate(&kStop), "cudaEventCreate kStop")) {
        if (kStart) cudaEventDestroy(kStart);
        if (kStop) cudaEventDestroy(kStop);
        cudaFree(dX); cudaFree(dA); cudaFree(dB); cudaFree(dH); cudaFree(dY); cudaFree(dZ);
        return 1;
    }

    if (!checkCuda(cudaEventRecord(kStart), "cudaEventRecord kStart")) {
        cudaEventDestroy(kStart); cudaEventDestroy(kStop);
        cudaFree(dX); cudaFree(dA); cudaFree(dB); cudaFree(dH); cudaFree(dY); cudaFree(dZ);
        return 1;
    }

    // Step 1: H(x) = allegator(x)
    curve25519_elligator2_u_kernel<<<blocks, threads>>>(dX, dH, static_cast<size_t>(n));
    // Step 2: y = H(x)^a
    curve25519_scalarmult_fe_kernel<<<blocks, threads>>>(dA, dH, dY, static_cast<size_t>(n));
    // Step 3: z = y^b
    curve25519_scalarmult_fe_kernel<<<blocks, threads>>>(dB, dY, dZ, static_cast<size_t>(n));

    if (!checkCuda(cudaGetLastError(), "kernel launch") ||
        !checkCuda(cudaEventRecord(kStop), "cudaEventRecord kStop") ||
        !checkCuda(cudaEventSynchronize(kStop), "cudaEventSynchronize kStop")) {
        cudaEventDestroy(kStart); cudaEventDestroy(kStop);
        cudaFree(dX); cudaFree(dA); cudaFree(dB); cudaFree(dH); cudaFree(dY); cudaFree(dZ);
        return 1;
    }

    float kernelMs = 0.0f;
    if (!checkCuda(cudaEventElapsedTime(&kernelMs, kStart, kStop), "cudaEventElapsedTime")) {
        cudaEventDestroy(kStart); cudaEventDestroy(kStop);
        cudaFree(dX); cudaFree(dA); cudaFree(dB); cudaFree(dH); cudaFree(dY); cudaFree(dZ);
        return 1;
    }
    cudaEventDestroy(kStart);
    cudaEventDestroy(kStop);

    std::vector<fe25519> hZ(static_cast<size_t>(n));

    if (!checkCuda(cudaMemcpy(hZ.data(), dZ, feBytes, cudaMemcpyDeviceToHost), "cudaMemcpy dZ")) {
        cudaFree(dX); cudaFree(dA); cudaFree(dB); cudaFree(dH); cudaFree(dY); cudaFree(dZ);
        return 1;
    }

    const auto cudaEnd = std::chrono::high_resolution_clock::now();
    const double cudaTotalMs = std::chrono::duration<double, std::milli>(cudaEnd - cudaStart).count();

    cudaFree(dX); cudaFree(dA); cudaFree(dB); cudaFree(dH); cudaFree(dY); cudaFree(dZ);

    // Sampled CPU reference checks with libsodium.
    std::vector<int> indices(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) indices[static_cast<size_t>(i)] = i;
    std::shuffle(indices.begin(), indices.end(), std::mt19937(1337));

    const auto cpuStart = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < sampleChecks; ++k) {
        const int idx = indices[static_cast<size_t>(k)];
        const uint8_t* x = hX.data() + static_cast<size_t>(idx) * 32u;

        unsigned char ed[32];
        unsigned char h_ref_b[32];
        unsigned char y_ref_b[32];
        unsigned char z_ref_b[32];

        if (crypto_core_ed25519_from_uniform(ed, x) != 0) {
            std::cerr << "libsodium from_uniform failed at sampled index " << idx << '\n';
            return 1;
        }
        if (crypto_sign_ed25519_pk_to_curve25519(h_ref_b, ed) != 0) {
            std::cerr << "libsodium pk_to_curve25519 failed at sampled index " << idx << '\n';
            return 1;
        }
        if (crypto_scalarmult_curve25519(y_ref_b, a_bytes, h_ref_b) != 0 ||
            crypto_scalarmult_curve25519(z_ref_b, b_bytes, y_ref_b) != 0) {
            std::cerr << "libsodium reference failed at sampled index " << idx << '\n';
            return 1;
        }

        const fe25519 z_ref = decodeU25519Le(z_ref_b);

        if (std::memcmp(&hZ[static_cast<size_t>(idx)], &z_ref, sizeof(fe25519)) != 0) {
            std::cerr << "OPRF z mismatch at sampled index " << idx << '\n';
            std::cerr << "  z gpu: " << feHex(hZ[static_cast<size_t>(idx)]) << '\n';
            std::cerr << "  z ref: " << feHex(z_ref) << '\n';
            return 1;
        }
    }
    const auto cpuEnd = std::chrono::high_resolution_clock::now();
    const double cpuSampleMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    std::cout << "OPRF pipeline completed for " << n << " inputs\n";
    std::cout << "CPU sampled reference checks: " << sampleChecks << '\n';
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CUDA total time (malloc+H2D+kernels+D2H): " << cudaTotalMs << " ms\n";
    std::cout << "Kernel time total: " << kernelMs << " ms\n";
    std::cout << "CPU sampled reference time (libsodium): " << cpuSampleMs << " ms\n";

    const int preview = std::min(n, 3);
    for (int i = 0; i < preview; ++i) {
        std::cout << "sample[" << i << "]\n";
        std::cout << "  z    = " << feHex(hZ[static_cast<size_t>(i)]) << '\n';
    }

    return 0;
}
