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
CUDA_VISIBLE_DEVICES=1 python3 crates/rustc-codegen-cuda/examples/blockscale_gemm_primitives/validate.py \
  --binary /path/to/blockscale_gemm_primitives \
  --output-dir /tmp/blockscale-results --sanitizer
CUDA_VISIBLE_DEVICES=1 /path/to/blockscale_gemm_primitives --json /tmp/blockscale-timing.json
```

Each validation case compares all 4,194,304 output bit patterns with the exact
CPU oracle, starts with NaN output and rechecks after repeated launches. Resident,
one-CTA and 17-CTA grids exercise persistent stage/phase wrapping. The optional
checks run memcheck with tensor-operation checking and synccheck. Timeout/failure
reports cannot reuse previous JSON output. The full CPU reference is substantial.

A normal run performs 100 warmups and 31 individual event-timed launches, matching
the CuTe example's protocol; flags are `--ctas`, `--warmup`, `--samples`, `--json`.
CUDA-event intervals include gaps between event execution and kernel execution;
they must not be compared with Nsight kernel-active durations. The report
distinguishes correctness from timing. Throughput is not a SOL claim.

## Local validation (2026-10-11)

On an RTX 5090, all 4,194,304 FP16 outputs matched the exact CPU oracle,
including repeat launches, with the resident 170-CTA grid and forced 1/17-CTA
grids. CUDA memcheck (`--check-tensor-ops yes`) and synccheck reported zero errors.
For this optimization, the fresh `cuda-device` suite passed 123 tests with
110 ignored and no failures. The production compile check and independent
code review also passed. Constructor tests cover valid pointer preservation
and invalid null, misaligned and undersized storage for all three shared formats
used here. Invalid-input tests use the host trap stub; generated MLIR was also
checked for validation branches ending in trap/unreachable.

## Kernel-active timing (2026-10-11)

Nsight Systems 2026.3.2, driver 615.71.09, physical GPU 1 (RTX 5090),
M=N=2048, K=1024, L=1, 128×128×128 tiles, E2M1 inputs, UE8M0 scales per
32 K values, FP32 accumulation and FP16 output. Each round used 100 warmups
and 31 measured launches, hot buffer reuse, no graphs and unlocked clocks.
The three Rust implementations ran serially in rotating order over three rounds;
Python ran three rounds in one process afterwards. All durations below are
Nsight kernel-active time, excluding host submission gaps.

| Implementation | Run medians (µs) | Median of run medians (µs) |
| --- | --- | --- |
| Rust primitives before this optimization | 12.960, 12.992, 12.960 | 12.960 |
| Rust primitives with trapping constructor | 11.648, 11.680, 11.616 | 11.648 |
| CuTe-RS, CUTLASS compiler 4.7.0 | 11.264, 11.264, 11.232 | 11.264 |
| Python CuTeDSL 4.6.2 | 10.080, 10.112, 10.112 | 10.112 |

The optimization reduces primitives kernel time by 10.1%; it remains 3.4%
slower than CuTe-RS and 15.2% slower than Python CuTeDSL in this measurement.
The Rust runs check all 4,194,304 FP16 bits against the CPU oracle. Python uses
the upstream reference check (`atol=0.1`, `rtol=0.01`) before timing a separate
generated workspace. This comparison matches shape, formats and timing method;
compiler versions and correctness oracles differ. It does not establish SOL.

Earlier event timings of 15.328 µs (Rust primitives) and 13.600 µs (CuTe-RS)
used the same problem size. They are a different timing metric and are not
the baseline for the improvement above.

`SharedRegion::from_raw_parts_or_trap` retains the existing constructor's null,
alignment and capacity validation and unsafe storage/lifetime contract. It
directly constructs the typed region after validation, avoiding the
pointer-bearing `Result` carrier in kernels that already trap on failure.
The Result-returning API remains available. The final cubin has no direct calls
to the three shared-region helper specializations; conditional TMA syscall
fallback calls remain. Resource usage is 128 registers and 8 bytes of stack
(previously 126/8; CuTe-RS uses 118/0). Kernel scheduling and raw TMA semantics
are unchanged.

No matching runnable upstream CUTLASS-primitives baseline was found in the local
CUTLASS/DKG checkouts. The available blockscale primitives example uses SM100
TCGen05/TMEM, which cannot run on this SM120 device; the SM120 instruction wrapper
alone is not a complete GEMM benchmark. The Python row above is CuTeDSL.

To reproduce one Rust timing round, use its freshly built executable:

```sh
CUDA_VISIBLE_DEVICES=1 nsys profile --trace=cuda,nvtx --sample=none \
  --cpuctxsw=none --output=/tmp/blockscale-active \
  /path/to/blockscale_gemm_primitives --warmup 100 --samples 31
nsys export --type sqlite --output=/tmp/blockscale-active.sqlite \
  /tmp/blockscale-active.nsys-rep
```

In `CUPTI_ACTIVITY_KIND_KERNEL`, select the last 31 `blockscale_gemm` launches
ordered by `start` (there are 132 total: one correctness launch, 100 warmups,
31 samples). Their `(end-start)/1000.0` values are microseconds. Repeat serially
for three rounds and report the median of round medians. Python sample windows
are delimited by `cutedsl_samples_round_*` NVTX ranges in the local harness.

[The recorded results](benchmark-results.json) preserve all samples, tool/source
hashes, correctness scope and provenance. Local reproduction harnesses:
`/tmp/primitives-optimization/compare.py` and
`/tmp/primitives-optimization/python/benchmark.py`.

Validated primitives cubin SHA-256:
`d55b1fe9b00bdaaf743165dd1c9fdfb046194536c094e4b48a6c53c0ff26b533`.
The sanitizer build and final formatted build have identical cubin hashes.
Local detailed evidence: `/tmp/primitives-optimization/final/manifest.json`,
`/tmp/primitives-optimization/validation/run-nuiwesaq/report.json`, and
`/tmp/primitives-optimization/timing/comparison.json` (alongside the Nsight traces).
