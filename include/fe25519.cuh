#pragma once

#include <cstdint>

// Field modulus used by curve25519-dalek: p = 2^255 - 19.
// Representation: 8 little-endian 32-bit limbs.
struct fe25519 {
    uint32_t limb[8];
};
