# FP16 GEMM with `cute-rs` on SM100

Two thread blocks (CTAs) cooperate on each 256 × 352 output tile:

```text
C[M, N] = A[M, K] @ B[N, K].T + bias[M, 1]  # bias is optional
```

Inputs and output are FP16; multiplication accumulates in FP32. `--has-bias`
adds one value per output row before conversion to FP16. M must be a positive
multiple of **256**, and N and K positive multiples of **8**. Partial N352 and
K64 tiles are supported. The example name describes the tile, while `--mnk`
sets the full problem size.

## The CuTe flow

`cute-rs` describes shared-memory layouts, tensor copies, pipelines, and MMA.
The host builds tensor maps from the same layout types. `ctm-rs` supplies
explicit thread coordinates, barriers, fences, and TMEM allocation/release.

```text
A/B global ──cluster TMA──▶ shared [5 stages] ──MMA──▶ TMEM
C global   ◀─────TMA────── shared [2 stages] ◀──FP16 + bias──┘
```

The paired MMA operations compute 192 and 160 columns while reusing A.
Persistent clusters advance through output tiles; `--clusters` overrides the
host's cluster count. See the [kernel and API notes](public_reference/README.md#kernel-architecture)
for the layouts and warp roles, or the [CTM-only example](../fp16_gemm_256x352_ctm)
for the instruction-level version.

## Build and run

You need an SM100 GPU, a compatible CUDA driver, and the configured cuda-oxide
toolchain. The default comparison automatically prepares a Python environment.
It requires `python3` and either `venv`/`ensurepip` (`python3-venv` on
Debian/Ubuntu) or `uv` on `PATH`, plus network access for the first install or
changed requirements. Later runs reuse the environment. See
[Python setup and overrides](public_reference/README.md#comparison-setup-and-timing).

From the `cuda-oxide/` directory:

```bash
cargo oxide toolchain install cutlass
export CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir
cargo oxide build fp16_gemm_256x352_cute --arch sm_100a
cargo oxide run fp16_gemm_256x352_cute --arch sm_100a
```

The default `256,352,64` run verifies both implementations on identical inputs
and prints:

```text
CUTLASS (CuTe + CTM): <time> us, <rate> TFLOP/s
cute-rs: <time> us, <rate> TFLOP/s
```

Both measurements use direct CUDA-event timing, with 25 warmups and 100 samples.
This small case is sensitive to launch overhead. Use the
[larger workload and measurement guidance](public_reference/README.md#repeated-performance-measurements)
for throughput comparisons. `--json result.json` saves the report; `--help`
lists all options.

`--rust-only` skips Python setup and checks Rust against a CPU oracle; large
shapes make that check expensive. It defaults to graph timing; use
`--timing-mode direct` for individual launches.

## Recorded B200 performance

Historical measurement from **2026-10-11**, **M=8192, N=8096, K=8192**, no bias,
CUTLASS 4.7, 74 clusters. Each run used 2,000 warmups and 301 direct CUDA-event
samples; the table shows the median and range of three run medians:

| Implementation | CUDA events (µs) | Run range (µs) |
| --- | ---: | ---: |
| Rust (cute-rs) | 764.672 | 761.024–765.856 |
| Paired CUTLASS reference | 761.216 | 759.360–764.160 |

Both outputs passed the independent oracle on identical inputs. Clocks were
unlocked and the ranges overlap. These measurements use the **previous local
reference**, not the current `public_reference/` kernel. No FP16 Nsight Systems
durations are recorded in this evidence. See the [measurement record](public_reference/README.md#recorded-b200-performance)
for provenance and the current comparison command.

<a id="regression-checks"></a>

The comparison uses the included Apache-2.0 [public reference](public_reference/SOURCE.md).
Detailed [contracts and run options](public_reference/README.md#shapes-verification-and-run-options),
[validation commands](public_reference/README.md#validation), and
[timing methodology](public_reference/README.md#comparison-setup-and-timing)
live alongside it.
