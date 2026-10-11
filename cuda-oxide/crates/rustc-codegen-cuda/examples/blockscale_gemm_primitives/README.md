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
