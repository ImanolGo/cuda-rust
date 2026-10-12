# MXFP4 GEMM with Rust CTM (`ctm-rs`)

This cuda-oxide example uses register primitives for SM120a on RTX 5090:
packed E2M1 inputs, UE8M0 scales per 32 K values, FP32 accumulators and FP16 output.
The fixed problem is **M=N=2048, K=1024**, with a 128×128 tile, three-stage TMA
ring, eight compute warps and one producer warp, and persistent scheduling.

The kernel uses typed A/B operands, scales, shared formats and tensor maps,
plus `AccumulatorTile16x8<2, 8>`. Physical lane addresses, barrier phases and
publication remain explicit Rust. It needs no CuTe operations or Python runtime.

From `cuda-oxide/`, with the pinned CUTLASS compiler 4.7 installed:

```sh
python3 scripts/check-primitives-api-compile.py --example blockscale \
  --output-dir /tmp/primitives-blockscale-evidence
```

The resulting manifest identifies the fresh
`run-*/target/release/blockscale_gemm_ctm` executable. Validate it on GPU 1:

```sh
CUDA_VISIBLE_DEVICES=1 python3 cuteir/examples/blockscale_gemm_ctm/validate.py \
  --binary /path/to/blockscale_gemm_ctm \
  --output-dir /tmp/blockscale-results --sanitizer
```

Validation checks every FP16 output bit against the CPU oracle, replay, and
170/1/17-CTA grids. `--sanitizer` adds memory/tensor and synchronization checks.
Run the executable directly to print CUDA-event timings; those intervals differ
from the Nsight Systems kernel-active durations below.

Latest recorded RTX 5090 comparison (2026-10-11), 170 CTAs, hot buffers and
unlocked clocks: three rotating rounds, 100 warmups and 31 samples per round.
Both cubins used one common host harness and the same complete oracle.

| Implementation | Compiler | Kernel-active median |
| --- | --- | ---: |
| Rust CTM | CUTLASS 4.7.0 | 11.359 µs |
| CUTLASS CTM reference | CUTLASS 4.8.0 | 10.624 µs |

Rust takes **6.9%** longer in this run. The reference is a repository-authored
port using `cutlass.experimental.primitives`. Compiler versions differ, so
this comparison does not isolate compiler effects. The latest compiler fix
produced a byte-identical Rust cubin; its small timing change is measurement
variation.

- [CUTLASS CTM reference and setup](reference/README.md)
- [Implementation details, validation, and benchmark history](reference/measurement-notes.md)
- [Latest compiler-fix results](reference/compiler-fix-results.json)
- [Source-cleanup results](reference/cleanup-results.json)
