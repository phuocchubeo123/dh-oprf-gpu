# CUDA SHA-256 Kernel

Simple CUDA C++ example that computes SHA-256 on the GPU (one message per thread, single 512-bit block), and uses OpenSSL SHA-256 on CPU for timing and correctness checks.

## Build

```bash
make
make field25519_test
```

## Run

```bash
make run ARGS="1000 128"
```

You can also run the binary directly:

```bash
./vector_mul 1000 128
```

Arguments:
- `num_messages`: number of random messages to hash.
- `sample_checks` (optional): number of random hashes to compare (GPU vs OpenSSL CPU). Default is `min(num_messages, 1024)`.

## Field Arithmetic Test

Build and run the 256-bit field arithmetic validator (mod `2^255 - 19`):

```bash
make field_test ARGS=4096
```

This compares CUDA `fe25519` add/mul kernels against OpenSSL big-number `BN_mod_add` and `BN_mod_mul` on random vectors.

## Field Sqrt Test

Build and run Tonelli-Shanks square-root validation (no bigint library):

```bash
make field_sqrt_test ARGS=4096
make field_sqrt_test ARGS="4096 10"
```

Test method:
- sample random field elements `x`
- compute `sq = x^2 (mod p)` on GPU
- compute `r = sqrt(sq)` on GPU using Tonelli-Shanks
- verify `r^2 == sq (mod p)` on GPU output

Arguments:
- `num_elements` (optional): number of random samples (default `4096`)
- `timed_iterations` (optional): repeated timed kernel passes after a warmup pass (default `5`)

To reduce first-launch JIT overhead, build for your exact GPU architecture:

```bash
make field25519_sqrt_test CUDA_ARCH=sm_86
```

## Allegator Compare Test

Build and run GPU Allegator2 map-to-curve (`Curve25519` Montgomery `u`) with sampled CPU reference checks:

```bash
make allegator25519_compare_test
make allegator_test ARGS="100000 1024"
```

Arguments:
- `num_elements` (optional): total GPU elements to process (default `4096`)
- `cpu_sample_checks` (optional): number of indices to validate on CPU reference (default `256`)

CPU reference backend:
- libsodium `crypto_core_ed25519_from_uniform`
- libsodium `crypto_sign_ed25519_pk_to_curve25519`

Only sampled CPU checks are run, as requested.

## Curve Scalar-Mult Test

Build and run Curve25519 scalar-multiplication (`scalar * u`) comparison:

```bash
make curve25519_scalarmult_test
make scalarmult_test ARGS="100000 1024"
```

Arguments:
- `num_elements` (optional): total GPU elements to process (default `4096`)
- `cpu_sample_checks` (optional): sampled CPU reference comparisons (default `512`)

CPU reference backend:
- libsodium `crypto_scalarmult_curve25519`

## OPRF Pipeline

Build and run OPRF flow with two independently sampled keys `a,b` (mod Curve25519 subgroup order):

```bash
make OPRF
make oprf ARGS="10000 512"
```

Pipeline per input `x`:
- `H(x) = allegator(x)`
- `y = H(x)^a`
- `z = y^b`

The last two steps are executed as two separate kernel launches (not combined).

Behavior:
- samples `a,b` in Curve25519 subgroup order
- runs a warmup pass before timed kernels
- performs sampled CPU reference checks (libsodium) controlled by `cpu_sample_checks`

## Files

- `main.cu`: CUDA SHA-256 kernel, host setup, and CPU digest cross-check.
- `field25519_kernel.cu`: CUDA kernels for 256-bit field add/mul/sqrt modulo `2^255 - 19`.
- `field25519_test.cu`: Host test harness that validates field kernels against OpenSSL BN arithmetic.
- `field25519_sqrt_test.cu`: Host test harness for GPU Tonelli-Shanks sqrt via resquare validation.
- `allegator25519_compare_test.cu`: GPU Allegator2 mapping test with sampled libsodium CPU reference checks.
- `curve25519_scalarmult_test.cu`: GPU X25519 scalar-multiplication test with sampled libsodium reference checks.
- `OPRF.cu`: End-to-end OPRF pipeline using Allegator map and two separate scalar multiplications.
- `Makefile`: Build and run helpers.
