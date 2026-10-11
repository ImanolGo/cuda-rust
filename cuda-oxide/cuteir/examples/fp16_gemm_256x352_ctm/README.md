# FP16 GEMM with Rust CTM (`ctm-rs`)

Ports [`fp16_gemm_256x352_cute`](../fp16_gemm_256x352_cute) to the typed
`ctm_rs` API without CuTe operations, layout algebra or Python
kernel dependencies. The SM100a schedule retains two CTAs, five TMA stages,
M256/N192 + M256/N160 paired MMA, explicit TMEM completion and two output stages.
Types distinguish physical formats, operand roles, CTA groups and TMEM views;
the kernel explicitly coordinates election, phase changes, fences and lifetimes.

From `cuda-oxide/`:

```sh
python3 scripts/check-primitives-api-compile.py --example fp16 \
  --output-dir /tmp/primitives-fp16-evidence
```

The manifest records the fresh build directory, compiler hashes, MLIR and cubin.
On an SM100 GPU, use that directory's `target/release/fp16_gemm_256x352_ctm`:

```sh
python3 cuteir/examples/fp16_gemm_256x352_ctm/validate.py \
  --binary /path/to/fp16_gemm_256x352_ctm \
  --output-dir /tmp/fp16-results --sanitizer
```

The six cases exercise tails, bias, persistent tiles, five-stage reuse and phase
wraps. The host also supports direct/graph timing and an exhaustive CPU oracle;
see `--help`. B200 validation results are recorded below. This kernel cannot
execute on an RTX 5090; use [`blockscale_gemm_ctm`](../blockscale_gemm_ctm/README.md)
for SM120a.

## Comparing with CUTLASS (CuTe + CTM)

Use the CuTe example's [local comparison setup](../fp16_gemm_256x352_cute/README.md#compare-with-cutlass-cute--ctm).
The Python reference is labeled "CUTLASS (CuTe + CTM)": CUTLASS
`experimental.primitives` supplies its hardware operations (TMA, barriers, TMEM
and MMA), while the CuTe frontend supplies compilation and runtime helpers.
This Rust CTM-only example is labeled `ctm-rs`; the sibling layout/tensor example
is labeled `cute-rs`.

Both kernels use two CTAs, five A/B stages, a 256×192 + 256×160 MMA split
with FILL/LASTUSE collector operations, 512 TMEM columns and two C stages.

The `compare.py --rust-bin` option accepts the CTM executable. Each run
exports seeded FP16 inputs, verifies the Rust output against the independent
PyTorch oracle, and times a single kernel with direct CUDA events. From
`cuda-oxide/cuteir/examples/fp16_gemm_256x352_cute/`, with the local assets installed:

```sh
.venv/bin/python compare.py --rust-bin /path/to/fp16_gemm_256x352_ctm \
  --mnk 8192,8096,8192 --seed 0 --warmup 2000 --iters 301 \
  --json /tmp/ctm-comparison.json
```

The printed comparison identifies both implementations:

```text
CUTLASS (CuTe + CTM): <time> us, <rate> TFLOP/s
ctm-rs: <time> us, <rate> TFLOP/s
```

The aggregate JSON keys remain `cuda_oxide` and `python_reference` for
compatibility. Compare these entries in the report. Check the
`input_sha256`, matching cluster counts, launch counts and timing settings. The
report must say `verification: "both-pass"`; a standalone Rust report marked
`external-oracle-required` is not an independent correctness result. Run the
commands sequentially and repeat them to measure clock/power variation.

## B200 validation (2026-10-11)

[Recorded results](b200-validation.json) use CUTLASS 4.7.0, driver 615.71.09
and a 1,000 W power limit. The final comparison uses M8192/N8096/K8192
without bias, seed 0, 2,000 warmups, 301 timed launches, 74 clusters and direct
CUDA events. Each table entry summarizes three run medians. All comparisons
verified both outputs against the independent PyTorch oracle using identical
input bytes.

| Implementation | Median (µs) | Run range (µs) |
| --- | ---: | ---: |
| Final Rust CTM | 778.976 | 776.864–780.576 |
| CUTLASS CTM, paired with final Rust | 779.456 | 771.168–779.872 |
| Earlier Rust CTM candidate with macros (historical) | 842.464 | 836.352–849.312 |
| Original Rust CTM (historical) | 1,846.848 | 1,846.624–1,847.072 |

The final normal CUDA-event timings overlap the CUTLASS run range. These runs
were separate from profiling, and their reports do not record the ambient
clock state. The historical rows were measured earlier and are retained to
identify the previous implementations.

Hardware profiling identified local-memory traffic from temporary enum/`Result`
values. The compiler now places its enum scratch allocations in the function
entry block, allowing scalar replacement to eliminate that traffic while
retaining the original validation and error handling. The kernel also computes
shared-memory descriptors across the issuing warp, outside elected-lane
branches. Each of the eight MMA instructions and the completion commit still
executes in an elected lane, matching the CUTLASS primitive sequence.

The kernel contains no local macros. Its helpers are ordinary
`#[inline(always)]` functions, with const generics for K steps and output chunks;
the MLIR exporter preserves the inlining request for pointer-bearing helpers.
The epilogue writes four aligned 128-bit chunks per lane, and invariant TMEM
views are checked once per output tile.

A separate Nsight Compute comparison used controlled SM clocks of approximately
845 MHz. These profile timings must not be mixed with the normal event timings
above:

| Profiled implementation | Time (ms) | Executed warp instructions (millions) | Tensor-pipe activity (% elapsed) | Local-memory sectors (millions) |
| --- | ---: | ---: | ---: | ---: |
| Rust CTM before compiler/descriptor fixes | 1.420960 | 52.641 | 74.760 | 12.389 |
| Final Rust CTM | 1.160704 | 32.277 | 91.517 | 0 |
| CUTLASS CTM | 1.133056 | 15.684 | 93.609 | 0 |

All three profiles report zero register spilling. The eliminated traffic came
from temporary local storage rather than register spills. The final Rust
profile still takes 2.44% longer than CUTLASS at the controlled clock, so the
normal event results do not establish identical performance.

The original passed the six numerical cases but memcheck reported a stack
overflow in its outlined epilogue. The final kernel passes all six cases,
memcheck with tensor-operation checks, and synccheck. Two additional fractional
FP16 input cases (including bias and N/K tails) pass a CPU float64 oracle with
maximum absolute errors of 0.00390625 and 0.0078125 using the existing tolerance.
The primitives compiler check verifies no CuTe operations/types, the eight
paired MMA issues, the TMEM load/wait, and the direct kernel ABI. Its final
fresh-build cubin is byte-identical to the measured cubin.

The Rust CuTe migration is checked separately: its three-run median moved from
762.624 to 764.672 µs, within the measured run ranges. It passes the same six
cases and sanitizers; its fractional outputs match the final CTM output bytes.
Rebuilding it with the final compiler produces the same cubin. This is a
migration regression check, separate from the CTM performance baseline.
