#include <cuda_runtime.h>
#include <openssl/sha.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kMaxMessageBytes = 55;  // single-block SHA-256 limit
constexpr int kDigestBytes = 32;

__device__ __constant__ uint32_t kSha256K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
    0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
    0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
    0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
    0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
    0xc67178f2u};

__device__ inline uint32_t rotr32(uint32_t x, uint32_t n) { return (x >> n) | (x << (32u - n)); }
__device__ inline uint32_t bigSigma0(uint32_t x) { return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22); }
__device__ inline uint32_t bigSigma1(uint32_t x) { return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25); }
__device__ inline uint32_t smallSigma0(uint32_t x) { return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3); }
__device__ inline uint32_t smallSigma1(uint32_t x) { return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10); }
__device__ inline uint32_t choose(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
__device__ inline uint32_t majority(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }

static bool checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(result) << '\n';
        return false;
    }
    return true;
}

__global__ void sha256KernelSingleBlock(const uint8_t* messages, const uint32_t* lengths, uint8_t* digests,
                                        int messageCount) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= messageCount) return;

    const uint32_t len = lengths[idx];
    uint8_t* out = digests + static_cast<size_t>(idx) * kDigestBytes;
    if (len > kMaxMessageBytes) {
        for (int i = 0; i < kDigestBytes; ++i) out[i] = 0;
        return;
    }

    uint8_t block[64] = {0};
    const uint8_t* in = messages + static_cast<size_t>(idx) * kMaxMessageBytes;
    for (uint32_t i = 0; i < len; ++i) block[i] = in[i];
    block[len] = 0x80;

    const uint64_t bitLen = static_cast<uint64_t>(len) * 8ull;
    for (int i = 0; i < 8; ++i) block[63 - i] = static_cast<uint8_t>((bitLen >> (i * 8)) & 0xffu);

    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) | (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
               (static_cast<uint32_t>(block[i * 4 + 2]) << 8) | static_cast<uint32_t>(block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) w[i] = smallSigma1(w[i - 2]) + w[i - 7] + smallSigma0(w[i - 15]) + w[i - 16];

    uint32_t a = 0x6a09e667u;
    uint32_t b = 0xbb67ae85u;
    uint32_t c = 0x3c6ef372u;
    uint32_t d = 0xa54ff53au;
    uint32_t e = 0x510e527fu;
    uint32_t f = 0x9b05688cu;
    uint32_t g = 0x1f83d9abu;
    uint32_t h = 0x5be0cd19u;

    for (int i = 0; i < 64; ++i) {
        const uint32_t t1 = h + bigSigma1(e) + choose(e, f, g) + kSha256K[i] + w[i];
        const uint32_t t2 = bigSigma0(a) + majority(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    const uint32_t state[8] = {0x6a09e667u + a, 0xbb67ae85u + b, 0x3c6ef372u + c, 0xa54ff53au + d,
                               0x510e527fu + e, 0x9b05688cu + f, 0x1f83d9abu + g, 0x5be0cd19u + h};

    for (int i = 0; i < 8; ++i) {
        out[i * 4] = static_cast<uint8_t>((state[i] >> 24) & 0xffu);
        out[i * 4 + 1] = static_cast<uint8_t>((state[i] >> 16) & 0xffu);
        out[i * 4 + 2] = static_cast<uint8_t>((state[i] >> 8) & 0xffu);
        out[i * 4 + 3] = static_cast<uint8_t>(state[i] & 0xffu);
    }
}

std::string toHex(const uint8_t* bytes, size_t len) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; ++i) oss << std::setw(2) << static_cast<unsigned>(bytes[i]);
    return oss.str();
}

bool parsePositiveInt(const char* raw, int* out) {
    if (raw == nullptr || raw[0] == '\0') return false;
    char* end = nullptr;
    const long long parsed = std::strtoll(raw, &end, 10);
    if (*end != '\0' || parsed <= 0 || parsed > static_cast<long long>(std::numeric_limits<int>::max())) {
        return false;
    }
    *out = static_cast<int>(parsed);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <num_messages> [sample_checks]\n";
        return 1;
    }

    int count = 0;
    if (!parsePositiveInt(argv[1], &count)) {
        std::cerr << "Invalid num_messages: " << argv[1] << '\n';
        return 1;
    }

    int sampleChecks = std::min(count, 1024);
    if (argc == 3 && !parsePositiveInt(argv[2], &sampleChecks)) {
        std::cerr << "Invalid sample_checks: " << argv[2] << '\n';
        return 1;
    }
    if (sampleChecks > count) sampleChecks = count;

    std::vector<std::string> messages(static_cast<size_t>(count));
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> lenDist(0, kMaxMessageBytes);
    std::uniform_int_distribution<int> charDist(0, 61);
    constexpr char kAlphabet[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

    for (int i = 0; i < count; ++i) {
        const int len = lenDist(rng);
        messages[i].resize(static_cast<size_t>(len));
        for (int j = 0; j < len; ++j) messages[i][j] = kAlphabet[charDist(rng)];
    }

    std::vector<uint8_t> hMessages(static_cast<size_t>(count) * kMaxMessageBytes, 0);
    std::vector<uint32_t> hLengths(count, 0);
    std::vector<uint8_t> hGpuDigests(static_cast<size_t>(count) * kDigestBytes, 0);
    std::vector<uint8_t> hCpuDigests(static_cast<size_t>(count) * kDigestBytes, 0);

    for (int i = 0; i < count; ++i) {
        hLengths[i] = static_cast<uint32_t>(messages[i].size());
        std::memcpy(hMessages.data() + static_cast<size_t>(i) * kMaxMessageBytes, messages[i].data(), messages[i].size());
    }

    uint8_t* dMessages = nullptr;
    uint32_t* dLengths = nullptr;
    uint8_t* dDigests = nullptr;

    const auto cudaStart = std::chrono::high_resolution_clock::now();

    const size_t messageBytes = hMessages.size() * sizeof(uint8_t);
    const size_t lengthBytes = hLengths.size() * sizeof(uint32_t);
    const size_t digestBytes = hGpuDigests.size() * sizeof(uint8_t);

    if (!checkCuda(cudaMalloc(&dMessages, messageBytes), "cudaMalloc dMessages") ||
        !checkCuda(cudaMalloc(&dLengths, lengthBytes), "cudaMalloc dLengths") ||
        !checkCuda(cudaMalloc(&dDigests, digestBytes), "cudaMalloc dDigests")) {
        cudaFree(dMessages);
        cudaFree(dLengths);
        cudaFree(dDigests);
        return 1;
    }

    if (!checkCuda(cudaMemcpy(dMessages, hMessages.data(), messageBytes, cudaMemcpyHostToDevice), "cudaMemcpy hMessages") ||
        !checkCuda(cudaMemcpy(dLengths, hLengths.data(), lengthBytes, cudaMemcpyHostToDevice), "cudaMemcpy hLengths")) {
        cudaFree(dMessages);
        cudaFree(dLengths);
        cudaFree(dDigests);
        return 1;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!checkCuda(cudaEventCreate(&start), "cudaEventCreate start") ||
        !checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop")) {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        cudaFree(dMessages);
        cudaFree(dLengths);
        cudaFree(dDigests);
        return 1;
    }

    const int threadsPerBlock = 128;
    const int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;
    if (!checkCuda(cudaEventRecord(start), "cudaEventRecord start")) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(dMessages);
        cudaFree(dLengths);
        cudaFree(dDigests);
        return 1;
    }

    sha256KernelSingleBlock<<<blocks, threadsPerBlock>>>(dMessages, dLengths, dDigests, count);

    if (!checkCuda(cudaGetLastError(), "Kernel launch") || !checkCuda(cudaEventRecord(stop), "cudaEventRecord stop") ||
        !checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop")) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(dMessages);
        cudaFree(dLengths);
        cudaFree(dDigests);
        return 1;
    }

    float gpuMs = 0.0f;
    if (!checkCuda(cudaEventElapsedTime(&gpuMs, start, stop), "cudaEventElapsedTime")) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(dMessages);
        cudaFree(dLengths);
        cudaFree(dDigests);
        return 1;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (!checkCuda(cudaMemcpy(hGpuDigests.data(), dDigests, digestBytes, cudaMemcpyDeviceToHost), "cudaMemcpy dDigests")) {
        cudaFree(dMessages);
        cudaFree(dLengths);
        cudaFree(dDigests);
        return 1;
    }

    const auto cudaEnd = std::chrono::high_resolution_clock::now();
    const double cudaMs = std::chrono::duration<double, std::milli>(cudaEnd - cudaStart).count();

    cudaFree(dMessages);
    cudaFree(dLengths);
    cudaFree(dDigests);

    const auto cpuStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < count; ++i) {
        const uint8_t* msg = hMessages.data() + static_cast<size_t>(i) * kMaxMessageBytes;
        uint8_t* out = hCpuDigests.data() + static_cast<size_t>(i) * kDigestBytes;
        if (SHA256(msg, hLengths[i], out) == nullptr) {
            std::cerr << "OpenSSL SHA256 failed at index " << i << '\n';
            return 1;
        }
    }
    const auto cpuEnd = std::chrono::high_resolution_clock::now();
    const double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    std::vector<int> sampled(static_cast<size_t>(count));
    std::iota(sampled.begin(), sampled.end(), 0);
    std::mt19937 sampleRng(1337);
    std::shuffle(sampled.begin(), sampled.end(), sampleRng);

    bool ok = true;
    for (int i = 0; i < sampleChecks; ++i) {
        const int idx = sampled[static_cast<size_t>(i)];
        const uint8_t* gpuDigest = hGpuDigests.data() + static_cast<size_t>(idx) * kDigestBytes;
        const uint8_t* cpuDigest = hCpuDigests.data() + static_cast<size_t>(idx) * kDigestBytes;
        if (std::memcmp(gpuDigest, cpuDigest, kDigestBytes) != 0) {
            std::cerr << "Digest mismatch at sampled index " << idx << '\n';
            std::cerr << "  GPU: " << toHex(gpuDigest, kDigestBytes) << '\n';
            std::cerr << "  CPU: " << toHex(cpuDigest, kDigestBytes) << '\n';
            ok = false;
            break;
        }
    }

    if (!ok) return 1;

    std::cout << "SHA-256 GPU kernel succeeded for " << count << " messages.\n";
    std::cout << "Random sample checks passed: " << sampleChecks << '\n';
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CUDA total time (malloc+H2D+kernel+D2H): " << cudaMs << " ms\n";
    std::cout << "GPU kernel time: " << gpuMs << " ms\n";
    std::cout << "CPU OpenSSL time: " << cpuMs << " ms\n";

    const int preview = std::min(sampleChecks, 3);
    for (int i = 0; i < preview; ++i) {
        const int idx = sampled[static_cast<size_t>(i)];
        const uint8_t* digest = hGpuDigests.data() + static_cast<size_t>(idx) * kDigestBytes;
        std::cout << "sample[" << idx << "] sha256=" << toHex(digest, kDigestBytes) << '\n';
    }

    return 0;
}
