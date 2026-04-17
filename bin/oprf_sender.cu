#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <unistd.h>
#include <vector>

#include "oprf_gpu.hpp"
#include "oprf_tcp.hpp"

namespace {

void printUsage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " <port>\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        printUsage(argv[0]);
        return 1;
    }

    const char* port = argv[1];
    const auto totalStart = std::chrono::high_resolution_clock::now();
    uint8_t b[32] = {0};
    if (!sampleScalarLe32(b)) {
        std::cerr << "Failed to sample sender scalar\n";
        return 1;
    }

    const int listenFd = listenTcp(port);
    if (listenFd < 0) {
        std::cerr << "Failed to listen on port " << port << '\n';
        return 1;
    }

    std::cout << "Sender listening on port " << port << '\n';
    const int connFd = accept(listenFd, nullptr, nullptr);
    close(listenFd);
    if (connFd < 0) {
        std::cerr << "accept failed\n";
        return 1;
    }
    std::cout << "Sender accepted receiver connection successfully\n";

    uint64_t count = 0;
    if (!recvCount(connFd, &count)) {
        close(connFd);
        return 1;
    }

    std::vector<uint8_t> yBytes(static_cast<size_t>(count) * 32u);
    if (!recvAll(connFd, yBytes.data(), yBytes.size(), "y payload")) {
        close(connFd);
        return 1;
    }

    const auto evalStart = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> zBytes;
    if (!gpuApplyScalarToUBytes(yBytes, b, &zBytes)) {
        close(connFd);
        return 1;
    }
    const auto evalEnd = std::chrono::high_resolution_clock::now();

    if (!sendCount(connFd, count) || !sendAll(connFd, zBytes.data(), zBytes.size(), "z payload")) {
        close(connFd);
        return 1;
    }

    close(connFd);
    const auto totalEnd = std::chrono::high_resolution_clock::now();
    const size_t recvBytes = sizeof(uint64_t) + yBytes.size();
    const size_t sentBytes = sizeof(uint64_t) + zBytes.size();
    const size_t totalBytes = recvBytes + sentBytes;
    const double evalMs = std::chrono::duration<double, std::milli>(evalEnd - evalStart).count();
    const double totalMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
    const double evalSeconds = evalMs / 1000.0;
    const double throughputInputs = (evalSeconds > 0.0) ? (static_cast<double>(count) / evalSeconds) : 0.0;
    const double throughputMiB = (evalSeconds > 0.0)
        ? (static_cast<double>(yBytes.size() + zBytes.size()) / (1024.0 * 1024.0) / evalSeconds)
        : 0.0;
    std::cout << "Sender processed " << count << " blinded inputs\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Sender timing: total=" << totalMs
              << " ms, evaluation=" << evalMs << " ms\n";
    std::cout << "Sender throughput: " << throughputInputs
              << " inputs/s, " << throughputMiB << " MiB/s\n";
    std::cout << "Sender communication cost: recv=" << recvBytes
              << " bytes, sent=" << sentBytes
              << " bytes, total=" << totalBytes << " bytes\n";
    const size_t preview = std::min<size_t>(count, 3u);
    for (size_t i = 0; i < preview; ++i) {
        std::cout << "z[" << i << "] = " << bytesHex(zBytes.data() + i * 32u, 32) << '\n';
    }
    return 0;
}
