# FP16 FMHA prefill with CuTe-RS — SM100a (B200)

This cuda-oxide example computes `softmax(Q @ K.T / sqrt(128)) @ V` on
data center Blackwell GPUs, using the public CUTLASS 4.7 FMHA schedule.

The kernel uses `cute-rs` for tensor/layout operations and `ctm-rs` for device
primitives, macros and storage types. Barrier rings distinguish TMA producers
from ordinary arrivals; packed FP32 arithmetic and TMEM allocation use CTM
types. The manifest retains `cuda-device` for macro expansion internally.
The CTM migration compiles for `sm_100a` locally; its runtime validation must
be repeated on SM100 hardware. The RTX 5090 / SM120 cannot execute this kernel.

- FP16 inputs/output, FP32 accumulation, and FP16 probabilities before PV.
- Contiguous BSHD tensors, head dimension **128**, runtime batch and lengths.
- MHA or GQA (`Hq % Hkv == 0`), dense attention (default) or top-left causal
  masking. Partial query and KV tiles are supported.

From `cuda-oxide/`, with CUTLASS 4.7 installed (`cargo oxide toolchain install cutlass`)
and the [Python reference environment](reference/README.md) available:

```bash
# Dense
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 cargo oxide run sm100_fmha_prefill_cute --arch sm_100a -- --shape 1,1024,1024,8,8 --warmup 100 --iters 100 --windows 5

# Dense persistent
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 cargo oxide run sm100_fmha_prefill_cute --arch sm_100a -- --shape 4,4096,4096,8,8 --warmup 100 --iters 100 --windows 5

# Causal grouped-query attention (GQA)
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 cargo oxide run sm100_fmha_prefill_cute --arch sm_100a -- --shape 4,4096,4096,8,2 --causal --warmup 100 --iters 100 --windows 5
```

The default runner compares Rust and pinned CuTeDSL on identical inputs and
checks both against an independent FP32 reference before timing. It finishes with:

```text
CuTeDSL: <time> us
CuTe-RS: <time> us
```

Add `--json comparison.json` to save the full report.
`--shape` is `B,Sq,Sk,Hq,Hkv`; add `--causal` for causal attention or
`--check-only` to validate without timing. Pass `--help` after `--` for options.

Previously recorded on B200 with CUTLASS 4.7 on both sides (2026-10-12): median
CUDA-event invocation times, **10 warmups**, 100 invocations per window, five
windows, hot cache. The commands above use **100 warmups** for new runs.

| Profile (B, Sq, Sk, Hq, Hkv) | CuTe-RS | CuTeDSL |
| --- | ---: | ---: |
| Dense (1, 1024, 1024, 8, 8) | 18.45 µs | 75.40 µs |
| Dense persistent (4, 4096, 4096, 8, 8) | 246.17 µs | 236.80 µs |
| Causal GQA (4, 4096, 4096, 8, 2) | 150.32 µs | 168.25 µs |

These intervals include host submission gaps; the small case does not establish
a kernel-active speedup, and the dense persistent timing ranges overlap.
With 2,000 warmups and 301 invocations per window, dense persistent medians
were **289.98 µs** (CuTe-RS) and **292.13 µs** (CuTeDSL), comparable throughput.

- [CuTeDSL reference and setup](reference/README.md)
- [Validation, benchmark commands, and detailed results](reference/measurement-notes.md)
- [Shared attention functionality and compiler gaps](../sm100_mla_decode_fp16_cute/reference/attention-feature-gaps.md)
- [Dedicated MLA decode example](../sm100_mla_decode_fp16_cute/README.md)
