# FP16 CTM implementation and validation notes

[Back to the example](../README.md).

All commands run from `cuda-oxide/` unless another directory is specified.
Follow the entry README's toolchain and backend setup first.

## Kernel and API

The [kernel](../src/kernel.rs) expresses its two-CTA schedule through `ctm-rs`,
without CuTe operations, layout algebra or Python kernel dependencies. Types
distinguish physical formats, operand roles, CTA groups and TMEM views. The
kernel explicitly coordinates election, phase changes, fences and lifetimes;
types do not prove collective participation or asynchronous completion.

- Five A/B stages overlap cluster TMA and MMA, with K64 per stage.
- Paired 256×192 and 256×160 MMA instructions reuse A through FILL/LASTUSE
  collector operations and accumulate into 512 TMEM columns.
- Each CTA has 192 threads and 211,072 bytes of dynamic shared memory, aligned
  to 1,024 bytes. Two shared C stages feed TMA output stores.
- Persistent clusters advance through output tiles. The host selects resident
  capacity by default; `--clusters` overrides the count.
- The [host](../src/main.rs) encodes tensor maps from the physical formats and
  keeps descriptors, allocations and the loaded module live through completion.

The final measured implementation uses ordinary `#[inline(always)]` helpers,
with const generics for K steps and output chunks. The MLIR exporter preserves
the inlining request for pointer-bearing helpers. The epilogue writes four
aligned 128-bit chunks per lane, and invariant TMEM views are checked once per
output tile.

## Comparison checks

The public Python reference uses CUTLASS `experimental.primitives` for TMA,
barriers, TMEM and MMA, and the CuTe frontend for compilation and runtime helpers.
Its label is `CUTLASS (CuTe + CTM)`; this Rust example is `ctm-rs`, and the sibling
layout/tensor example is `cute-rs`. Both CTM implementations use the same tile
split, five input stages, 512 TMEM columns and two output stages.

Follow the [comparison setup](../../fp16_gemm_256x352_cute/public_reference/README.md#comparison-setup-and-timing)
and the entry README command. `compare.py --rust-bin` accepts this executable,
exports seeded FP16 inputs, and checks both outputs against an independent
PyTorch oracle before reporting direct CUDA-event timing.

The aggregate JSON keys remain `cuda_oxide` and `python_reference` for
compatibility. Check `input_sha256`, cluster counts, launch counts and timing
settings. The report must say `verification: "both-pass"`; a standalone Rust
report marked `external-oracle-required` is not an independent correctness
result. Run comparisons sequentially, repeat them, and retain the run spread
to assess clock/power variation.

The standalone executable defaults to graph timing and a CPU oracle using FP32
accumulation rounded to FP16 (`atol=0.1`, `rtol=1e-5`). Large shapes make that
exhaustive CPU check expensive. Direct timing measures one kernel per event
pair; graph timing reports time per captured launch. A direct run with zero
warmups and one iteration launches once total and verifies that output.

## Recorded B200 measurements

[b200-validation.json](../b200-validation.json) is the October 11, 2026 record.
It identifies source revision `1755d950e7cac3791937673a69520cc38fdc9341`
with uncommitted changes, and retains source, compiler, binary and cubin hashes.
The environment was B200, driver 615.71.09, a 1,000 W power limit, CUTLASS 4.7.0,
and PyTorch 2.13.0+cu130. Normal comparison commands requested no clock locking;
their reports do not describe the ambient clock state in each timing window.

The measured Python reference was the earlier ignored local implementation,
identified by SHA-256
`82439e501f521bf211afe71f1273fafcd80a95c34a431a7a0a4f154e490bab9b`.
These measurements do not establish performance of the public reference added
on October 18; its [provenance](../../fp16_gemm_256x352_cute/public_reference/SOURCE.md)
is recorded separately. Historical artifact locations in the saved JSON are
provenance records, not bundled public reports.

The event protocol was M8192/N8096/K8192, no bias, seed 0, 2,000 warmups,
301 timed single-kernel launches and 74 clusters. Each entry below summarizes
three run medians. Every comparison checked both outputs against the independent
PyTorch oracle using identical input bytes.

| Implementation | Median (µs) | Run range (µs) |
| --- | ---: | ---: |
| Final Rust CTM | 778.976 | 776.864–780.576 |
| CUTLASS CTM, paired with final Rust | 779.456 | 771.168–779.872 |
| Earlier Rust CTM candidate with macros | 842.464 | 836.352–849.312 |
| Original Rust CTM | 1,846.848 | 1,846.624–1,847.072 |

The final event timings overlap the paired CUTLASS run range. The earlier
candidate and original measurements identify previous implementations and
were collected in different runs; comparing their standalone medians across
clock regimes does not establish a speedup.

### Separate Nsight Compute profiles

Nsight Compute 2026.3.1 used base clock control (approximately 845 MHz),
`cache_control=none` and a stable pipeline boost state. It profiled one launch
after 2,000 warmups with 23 replay passes, using the same shape and input hashes.
These durations are separate from the normal CUDA-event results above.

| Profiled implementation | Time (ms) | Executed warp instructions (millions) | Tensor-pipe activity (% elapsed) | Local-memory sectors (millions) |
| --- | ---: | ---: | ---: | ---: |
| Rust CTM before compiler/descriptor fixes | 1.420960 | 52.641 | 74.760 | 12.389 |
| Final Rust CTM | 1.160704 | 32.277 | 91.517 | 0 |
| CUTLASS CTM | 1.133056 | 15.684 | 93.609 | 0 |

All three profiles report zero register spilling. The eliminated traffic came
from temporary enum/`Result` storage. Moving enum scratch allocations to the
function entry block allowed scalar replacement to eliminate it while keeping
validation and error handling. Shared-memory descriptors are computed across
the issuing warp, outside elected-lane branches. Each of the eight MMA issues
and the completion commit still executes in an elected lane, matching the
CUTLASS primitive sequence.

The final Rust profile takes 2.44% longer than CUTLASS at the controlled clock.
The overlapping event ranges do not establish identical performance. The saved
profiling is from Nsight Compute; this record contains no Nsight Systems measurement.

## Validation

To create fresh compiler evidence, from `cuda-oxide/`:

```bash
python3 scripts/check-primitives-api-compile.py --example fp16 \
  --output-dir /tmp/primitives-fp16-evidence
```

The manifest records the fresh build directory, compiler hashes, MLIR and cubin.
The check verifies no CuTe operations/types, the eight paired MMA issues, the
TMEM load/wait, and the direct kernel ABI. It does not execute GPU work.

On Linux with an SM100 GPU, pass the resulting binary to the validation script:

```bash
python3 cuteir/examples/fp16_gemm_256x352_ctm/validate.py \
  --binary /path/to/fp16_gemm_256x352_ctm \
  --output-dir /tmp/fp16-results --sanitizer
```

Use the manifest's build directory and its `target/release/` executable, or the
ordinary build's executable from the entry README. The six cases exercise
full tiles, tails, bias, persistent tiles, five-stage reuse and phase wraps,
using the exhaustive CPU oracle and graph replay checks. `--sanitizer` adds
Compute Sanitizer memcheck with tensor-operation checks and synccheck.

In the October 11 record, the original passed the numerical cases but memcheck
reported a stack overflow in its outlined epilogue. The final kernel passed
all six cases and both sanitizers. Two additional fractional FP16 cases,
including bias and N/K tails, passed a CPU float64 oracle with maximum absolute
errors of 0.00390625 and 0.0078125 under the existing tolerance. Its final fresh
cubin was byte-identical to the measured cubin.

The separate `cute-rs` migration regression moved its three-run median from
762.624 to 764.672 µs, within the measured run ranges. It passed the same six
cases and sanitizers; fractional outputs matched the final CTM output bytes,
and rebuilding with the final compiler produced the same cubin. This checks
the migration separately from the CTM performance baseline.
