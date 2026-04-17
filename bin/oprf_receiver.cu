#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <unistd.h>
#include <vector>

#include "oprf_gpu.hpp"
#include "oprf_tcp.hpp"

namespace {

bool parsePositiveInt(const char* raw, int* out) {
    if (raw == nullptr || raw[0] == '\0') return false;
    char* end = nullptr;
    const long long parsed = std::strtoll(raw, &end, 10);
    if (*end != '\0' || parsed <= 0 || parsed > 1000000000LL) return false;
    *out = static_cast<int>(parsed);
    return true;
}

void printUsage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " <host> <port> [num_inputs]\n";
}

}  // namespace

int main(int argc, char** argv) {
    int n = 4096;
    if (argc != 3 && argc != 4) {
        printUsage(argv[0]);
        return 1;
    }
    if (argc == 4 && !parsePositiveInt(argv[3], &n)) {
        printUsage(argv[0]);
        return 1;
    }

    const char* host = argv[1];
    const char* port = argv[2];
    const auto totalStart = std::chrono::high_resolution_clock::now();

    uint8_t a[32] = {0};
    if (!sampleScalarLe32(a)) {
        std::cerr << "Failed to sample receiver scalar\n";
        return 1;
    }

    std::vector<uint8_t> xBytes(static_cast<size_t>(n) * 32u);
    if (!fillRandomBytes(xBytes.data(), xBytes.size())) {
        std::cerr << "Failed to sample receiver inputs\n";
        return 1;
    }

    const auto blindStart = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> yBytes;
    if (!gpuMapAndBlind(xBytes, a, &yBytes)) {
        return 1;
    }
    const auto blindEnd = std::chrono::high_resolution_clock::now();

    const int fd = connectTcp(host, port);
    if (fd < 0) {
        std::cerr << "Failed to connect to " << host << ':' << port << '\n';
        return 1;
    }
    std::cout << "Receiver connected to sender successfully at " << host << ':' << port << '\n';

    if (!sendCount(fd, static_cast<uint64_t>(n)) || !sendAll(fd, yBytes.data(), yBytes.size(), "y payload")) {
        close(fd);
        return 1;
    }

    uint64_t returnedCount = 0;
    if (!recvCount(fd, &returnedCount) || returnedCount != static_cast<uint64_t>(n)) {
        std::cerr << "Unexpected sender response count\n";
        close(fd);
        return 1;
    }

    std::vector<uint8_t> zBytes(static_cast<size_t>(returnedCount) * 32u);
    if (!recvAll(fd, zBytes.data(), zBytes.size(), "z payload")) {
        close(fd);
        return 1;
    }
    close(fd);
    const auto totalEnd = std::chrono::high_resolution_clock::now();

    const size_t sentBytes = sizeof(uint64_t) + yBytes.size();
    const size_t recvBytes = sizeof(uint64_t) + zBytes.size();
    const size_t totalBytes = sentBytes + recvBytes;
    const double blindMs = std::chrono::duration<double, std::milli>(blindEnd - blindStart).count();
    const double totalMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
    std::cout << "Receiver sent " << n << " blinded inputs and received " << returnedCount << " evaluated outputs\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Receiver timing: total=" << totalMs
              << " ms, blind=" << blindMs << " ms\n";
    std::cout << "Receiver communication cost: sent=" << sentBytes
              << " bytes, recv=" << recvBytes
              << " bytes, total=" << totalBytes << " bytes\n";
    const size_t preview = std::min<size_t>(static_cast<size_t>(returnedCount), 3u);
    for (size_t i = 0; i < preview; ++i) {
        std::cout << "x[" << i << "] = " << bytesHex(xBytes.data() + i * 32u, 32) << '\n';
        std::cout << "y[" << i << "] = " << bytesHex(yBytes.data() + i * 32u, 32) << '\n';
        std::cout << "z[" << i << "] = " << bytesHex(zBytes.data() + i * 32u, 32) << '\n';
    }
    return 0;
}
