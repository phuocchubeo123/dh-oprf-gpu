# CUDA PSI / OPRF Library

CUDA-based building blocks for PSI-oriented OPRF work over Curve25519, including field arithmetic, Elligator2 mapping, scalar multiplication, and a first sender/receiver TCP scaffold.

## Build

```bash
make
make field25519_test
```

## OPRF Interfaces

The main OPRF entry points are now the two TCP-facing binaries:

- `target/oprf_sender`: server-side process that listens on a TCP port, holds the sender secret scalar `b`, evaluates received blinded points, and returns the results.
- `target/oprf_receiver`: client-side process that samples receiver inputs and receiver scalar `a`, blinds inputs on the GPU, sends them to the sender, and receives the evaluated outputs.

Build them with:

```bash
make -B oprf_sender
make -B oprf_receiver
```

Run the sender in one terminal:

```bash
./target/oprf_sender 9000
```

Run the receiver in another terminal:

```bash
./target/oprf_receiver 127.0.0.1 9000 4096
```

Current scaffold behavior:
- the receiver samples random `x` inputs and a receiver scalar `a`
- the receiver computes `y = H(x)^a` on the GPU
- the receiver sends the blinded `y` values to the sender over TCP
- the sender applies its secret scalar `b` on the GPU and returns `z = y^b`
- both sides print connection success and application-level communication cost in bytes

This is the first networked split of the existing pipeline. It is not yet a complete finalized OPRF protocol with serialization hardening, transcript handling, or unblinding/finalization logic.

## Field Arithmetic Test

Build and run the 256-bit field arithmetic validator (mod `2^255 - 19`):

```bash
make field_test ARGS=4096
make field_test_run ARGS=4096
```

This compares CUDA `fe25519` add/mul kernels against OpenSSL big-number `BN_mod_add` and `BN_mod_mul` on random vectors.

## Field Sqrt Test

Build and run Tonelli-Shanks square-root validation (no bigint library):

```bash
make field_sqrt_test ARGS=4096
make field_sqrt_test_run ARGS="4096 10"
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
make allegator_test_run ARGS="100000 1024"
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
make scalarmult_test_run ARGS="100000 1024"
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
make oprf_run ARGS="10000 512"
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

- `include/fe25519.cuh`: shared `fe25519` field-element type.
- `include/oprf_gpu.hpp`: shared GPU-side host helpers for scalar sampling, encoding, and launching the existing CUDA kernels.
- `include/oprf_tcp.hpp`: TCP socket helpers for connecting, listening, and sending fixed-size payloads reliably.
- `src/field25519_arith.cuh`: finite-field arithmetic modulo `2^255 - 19`.
- `src/curve25519_decode.cuh`: byte-to-field and field-to-byte encoding helpers.
- `src/curve25519_scalarmult.cuh`: X25519 scalar-multiplication helpers on Montgomery `u` coordinates.
- `src/curve25519_elligator.cuh`: Elligator2 map-to-curve helper for Curve25519 `u`.
- `src/field25519_sqrt.cuh`: Tonelli-Shanks square-root routine over the field.
- `src/kernels.cu`: exported CUDA kernels for field arithmetic, Elligator mapping, and Curve25519 scalar multiplication.
- `bin/field25519_test.cu`: host test harness that validates field kernels against OpenSSL BN arithmetic.
- `bin/field25519_sqrt_test.cu`: host test harness for GPU Tonelli-Shanks sqrt via resquare validation.
- `bin/allegator25519_compare_test.cu`: GPU Allegator2 mapping test with sampled libsodium CPU reference checks.
- `bin/curve25519_scalarmult_test.cu`: GPU X25519 scalar-multiplication test with sampled libsodium reference checks.
- `bin/OPRF.cu`: single-process OPRF pipeline benchmark using two separate scalar multiplications.
- `bin/oprf_sender.cu`: TCP sender/server for the networked OPRF scaffold.
- `bin/oprf_receiver.cu`: TCP receiver/client for the networked OPRF scaffold.
- `Makefile`: build and run helpers.
