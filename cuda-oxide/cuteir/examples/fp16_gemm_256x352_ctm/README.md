# FP16 GEMM with Rust CTM (`ctm-rs`)

Two thread blocks (CTAs) cooperate on each 256 × 352 output tile:

```text
C[M, N] = A[M, K] @ B[N, K].T + bias[M, 1]  # bias is optional
```

Inputs and output are FP16; accumulation and optional row bias use FP32 before
conversion to FP16. M must be a positive multiple of **256**, and N and K of
**8**. Partial N352 and K64 tiles are supported. `--mnk` sets the full problem
size; the example name describes the output tile.

## The CTM flow

This version expresses the [CuTe example's](../fp16_gemm_256x352_cute) schedule
through `ctm-rs` physical formats and instruction types:

```text
A/B global ──cluster TMA──▶ shared [5 stages] ──MMA──▶ TMEM
C global   ◀─────TMA────── shared [2 stages] ◀──FP16 + bias──┘
```

Paired 256×192 and 256×160 MMA operations reuse A. The kernel explicitly
coordinates election, barrier phases, fences and TMEM completion; types retain
operand roles and storage formats. See the [API notes](reference/README.md#kernel-and-api)
for the launch contract and caller obligations.

## Build and run

You need an SM100 GPU, a compatible CUDA driver, and the configured cuda-oxide
toolchain. This kernel cannot run on RTX 5090; see
[`blockscale_gemm_ctm`](../blockscale_gemm_ctm/README.md) for SM120a.
From `cuda-oxide/`:

```bash
cargo oxide toolchain install cutlass
export CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir
export CUDA_OXIDE_MLIR_PROFILE=cutlass-primitives-mlir22
cargo oxide build fp16_gemm_256x352_ctm --arch sm_100a
cargo oxide run fp16_gemm_256x352_ctm --arch sm_100a
```

The default `256,352,64` run prints `CPU verification: PASS` and a JSON timing
report. It needs no Python packages. Timing defaults to graph replay with 25
warmups and 100 samples; `--timing-mode direct --warmup 0 --iters 1` performs
exactly one kernel launch. `--help` lists the options.

## Comparing with CUTLASS (CuTe + CTM)

Use the CuTe example's [comparison setup](../fp16_gemm_256x352_cute/public_reference/README.md#comparison-setup-and-timing).
Initial setup needs `python3`, `venv`/`ensurepip` or `uv`, and network access.
From `cuda-oxide/`, prepare its environment and compare the CTM executable:

```bash
python3 cuteir/examples/fp16_gemm_256x352_cute/setup_reference.py
cd cuteir/examples/fp16_gemm_256x352_cute
.venv/bin/python compare.py \
  --rust-bin ../fp16_gemm_256x352_ctm/target/release/fp16_gemm_256x352_ctm \
  --mnk 8192,8096,8192 --seed 0 --warmup 2000 --iters 301 \
  --json /tmp/ctm-comparison.json
```

Use the actual binary path if you changed Cargo's target directory. The report
labels the implementations `ctm-rs` and `CUTLASS (CuTe + CTM)`, verifies identical
inputs against PyTorch, and times single kernels with direct CUDA events.
Repeat runs and retain their spread; see [comparison checks](reference/README.md#comparison-checks).

## B200 validation (2026-10-11)

Historical [recorded results](b200-validation.json), M8192/N8096/K8192 without
bias: direct CUDA events, seed 0, 2,000 warmups, 301 launches, 74 clusters.
Each entry is the median of three run medians; both outputs passed the oracle.

| Implementation | Median (µs) | Run range (µs) |
| --- | ---: | ---: |
| Final Rust CTM | 778.976 | 776.864–780.576 |
| Paired CUTLASS CTM reference | 779.456 | 771.168–779.872 |

These used the earlier local Python reference, not the public reference added
on October 18. Normal event runs did not control clocks. See
[provenance and profiling](reference/README.md#recorded-b200-measurements) for the
separate controlled-clock profiles, and [validation commands](reference/README.md#validation).
