# MXFP4 GEMM with Rust instruction primitives

SM120a port of `cuteir/examples/blockscale_gemm_cute` for the RTX 5090:
packed E2M1 inputs, UE8M0 scales per 32 K values, FP32 accumulators and FP16 output.
It retains the 128×128 tile, three-stage TMA ring, eight compute warps and one
producer warp, persistent scheduling, and shared-memory STMATRIX/TMA epilogue.
The fixed problem is M=N=2048, K=1024.

The kernel uses `cuda_device::primitives`, with distinct A/B register and scale
carriers and typed shared formats/tensor maps. Election, physical lane addresses,
barrier phases and publication remain explicit Rust. No CuTe/Python dependency
or CuTe operations/types are required. Five grid-constant tensor maps replace the
CuTe example's device pointers to descriptor allocations.

From `cuda-oxide/`, with pinned CUTLASS compiler 4.7 installed:

```sh
python3 scripts/check-primitives-api-compile.py --example blockscale \
  --output-dir /tmp/primitives-blockscale-evidence
```

The manifest records the fresh executable directory, source/compiler hashes,
MLIR and validated cubin. Use the resulting
`run-*/target/release/blockscale_gemm_primitives` executable:

```sh
python3 crates/rustc-codegen-cuda/examples/blockscale_gemm_primitives/validate.py \
  --binary /path/to/blockscale_gemm_primitives \
  --output-dir /tmp/blockscale-results --sanitizer
/path/to/blockscale_gemm_primitives --json /tmp/blockscale-timing.json
```

Each validation case compares all 4,194,304 output bit patterns with the exact
CPU oracle, starts with NaN output and rechecks after repeated launches. Resident,
one-CTA and 17-CTA grids exercise persistent stage/phase wrapping. The optional
checks run memcheck with tensor-operation checking and synccheck. Timeout/failure
reports cannot reuse previous JSON output. The full CPU reference is substantial.

A normal run performs 100 warmups and 31 individual event-timed launches, matching
the CuTe example's protocol; flags are `--ctas`, `--warmup`, `--samples`, `--json`.
The report distinguishes correctness from timing. Throughput is not a SOL claim.

## Local validation (2026-10-11)

On an RTX 5090, all 4,194,304 FP16 outputs matched the exact CPU oracle,
including repeat launches, with the resident 170-CTA grid and forced 1/17-CTA
grids. CUDA memcheck (`--check-tensor-ops yes`) and synccheck reported zero errors.
The implementation also passed 3,689 tests across the affected Rust crates,
two pinned CUTLASS consumer tests, generated-output checks and independent review.

Three serial, alternating runs used 100 warmups and 31 event-timed launches per
process, the same fixed problem and exhaustive correctness checks for both paths:

| Implementation | Run medians (µs) | Median of run medians (µs) |
| --- | --- | --- |
| Primitives | 15.392, 15.328, 15.136 | 15.328 |
| Existing CuTe | 13.600, 13.600, 13.472 | 13.600 |

Primitives is 12.7% slower in this measurement; performance parity is
not established. Cubin resource inspection reports 126 registers and 8 bytes of
stack versus CuTe's 118 registers and zero stack. Device helper calls remain in
the primitives image and are a candidate for further tuning.

Validated primitives cubin SHA-256:
`59e207e940a641aa66312f2cf54a21bed66bff7a6e9ae79eaa42e3d30a5511da`.
Local detailed evidence: `/tmp/primitives-blockscale-final/manifest.json`,
`/tmp/blockscale-gpu-validation/run-eophgihc/report.json`, and
`/tmp/blockscale-timing/comparison.json`.
