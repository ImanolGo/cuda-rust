# FP16 MLA decode with CuTe-RS — SM100a (B200)

This cuda-oxide example ports the public CUTLASS FP16 MLA decode kernel to
CuTe-RS for data center Blackwell. Two-CTA clusters process a paged latent KV
cache; a second kernel combines partial results when multiple splits are used.

The kernel uses `cute_rs` for typed layouts, structured MMA and TMA operations,
and `ctm_rs` for explicit instructions: pipeline synchronization, TMEM transfers,
packed arithmetic, warp/cluster coordination and register control. CTM also
exposes the kernel macros and canonical storage types. The manifest retains
`cuda-device` because the macro expansions still refer to that crate internally.
CuTe operations retain their CuTe dialect lowering through `cutlass-mlir`;
this example uses both APIs rather than the primitives-only compiler profile.

- **128 heads**, one query, latent dimension **512**, RoPE dimension **64**,
  and **128-token pages**; the latent cache supplies both K and V.
- FP16 inputs, probabilities and output; FP32 QK/PV accumulation and base-2 LSE.
- Positive batch and uniform KV lengths, **1–256 splits**, permuted pages and
  partial final pages. Attention is noncausal with explicit length masking.
- Softmax scale defaults to **0.0416** and output scale to **1.0**.

From `cuda-oxide/`, with CUTLASS 4.7 installed (`cargo oxide toolchain install cutlass`):

```bash
# Compile without launching a GPU kernel
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 cargo oxide build sm100_mla_decode_fp16_cute --arch sm_100a

# Check both implementations against the independent oracle on B200
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 cargo oxide run sm100_mla_decode_fp16_cute --arch sm_100a -- --batch 1 --kv-length 1024 --splits 1 --clusters 1 --check-only

# Small, unsplit decode
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 cargo oxide run sm100_mla_decode_fp16_cute --arch sm_100a -- --batch 1 --kv-length 1024 --splits 1 --clusters 1 --warmup 300 --iters 100 --windows 5

# Batched, persistent split-KV decode
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 cargo oxide run sm100_mla_decode_fp16_cute --arch sm_100a -- --batch 16 --kv-length 8192 --splits 8 --clusters 74 --warmup 300 --iters 100 --windows 5
```

Install the [Python reference dependencies](reference/README.md) before running.
The default runner compares Rust and pinned CuTeDSL on identical inputs and
checks both against an independent FP64 oracle before timing. It finishes with:

```text
CuTeDSL: <time> us
CuTe-RS: <time> us
```

Add `--json comparison.json` to save the full report or `--check-only` to
validate without timing. Pass `--help` after `--` for options.

Historical results from before the CTM migration, recorded on **NVIDIA B200**,
CUTLASS 4.7 at optimization level 2 on both sides (2026-10-12). The CuTeDSL reference is the unmodified, pinned upstream kernel.

| Batch / KV length / splits / clusters | CuTe-RS | CuTeDSL |
| --- | ---: | ---: |
| 1 / 1024 / 1 / 1 | 16.775 µs | 18.022 µs |
| 16 / 8192 / 8 / 74 | 48.514 µs | 49.213 µs |

CUDA-event medians include split reduction when enabled: ten windows across
both timing orders, **10 warmups** and 100 invocations per window. The commands
above use **300 warmups** for new runs. GPU clocks were not locked. Sustained
measurements also reach parity; full ranges and the warmup-dependent timing
shift are recorded in the measurement notes.

- [CTM migration: 9 passing B200 cases, sanitizers and compiler comparison](reference/results/ctm-migration/summary.json)
- [CuTeDSL reference and setup](reference/README.md)
- [Validation, profiling, benchmark commands, and detailed results](reference/measurement-notes.md)
- [Historical correctness results](reference/results/optimization-b200/validation.json) · [Benchmark evidence](reference/results/optimization-b200/summary.json)
- [Shared attention feature-gap roadmap](reference/attention-feature-gaps.md)
