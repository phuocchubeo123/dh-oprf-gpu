#pragma once

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

inline bool sendAll(int fd, const void* data, size_t len, const char* label) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    size_t sent = 0;
    while (sent < len) {
        const ssize_t n = send(fd, p + sent, len - sent, 0);
        if (n <= 0) {
            std::cerr << label << " send failed: " << std::strerror(errno) << '\n';
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    return true;
}

inline bool recvAll(int fd, void* data, size_t len, const char* label) {
    uint8_t* p = static_cast<uint8_t*>(data);
    size_t recvd = 0;
    while (recvd < len) {
        const ssize_t n = recv(fd, p + recvd, len - recvd, 0);
        if (n <= 0) {
            std::cerr << label << " recv failed: " << std::strerror(errno) << '\n';
            return false;
        }
        recvd += static_cast<size_t>(n);
    }
    return true;
}

inline bool sendCount(int fd, uint64_t count) {
    uint8_t wire[8];
    for (int i = 0; i < 8; ++i) {
        wire[7 - i] = static_cast<uint8_t>((count >> (i * 8)) & 0xFFu);
    }
    return sendAll(fd, wire, sizeof(wire), "count");
}

inline bool recvCount(int fd, uint64_t* count) {
    uint8_t wire[8];
    if (!recvAll(fd, wire, sizeof(wire), "count")) return false;
    uint64_t out = 0;
    for (int i = 0; i < 8; ++i) {
        out = (out << 8) | static_cast<uint64_t>(wire[i]);
    }
    *count = out;
    return true;
}

inline int connectTcp(const char* host, const char* port) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* result = nullptr;
    if (getaddrinfo(host, port, &hints, &result) != 0) {
        return -1;
    }

    int fd = -1;
    for (addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
        fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (fd == -1) continue;
        if (connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(result);
    return fd;
}

inline int listenTcp(const char* port) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    addrinfo* result = nullptr;
    if (getaddrinfo(nullptr, port, &hints, &result) != 0) {
        return -1;
    }

    int fd = -1;
    for (addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
        fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (fd == -1) continue;
        int yes = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        if (bind(fd, rp->ai_addr, rp->ai_addrlen) == 0 && listen(fd, 1) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(result);
    return fd;
}
