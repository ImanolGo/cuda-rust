# FP16 prefill with CuTe-RS — SM120a (RTX 5090)

This cuda-oxide example computes `softmax(Q @ K.T / sqrt(64)) @ V` using
FP16 inputs, FP32 accumulation, and 64-by-64 tiles.

- Batch size **1**, head dimension **64**, contiguous BSHD tensors.
- Sequence lengths are positive multiples of 64. **Causal** (default) requires
  equal query/KV lengths; `--noncausal` permits unequal lengths and also `Sq=1`.
- Shapes are runtime parameters by default. `--shape-mode static` specializes
  **Sq=Sk=1024, H=8**; other shapes require runtime mode.

From `cuda-oxide/`, with CUTLASS 4.7 installed (`cargo oxide toolchain install cutlass`):

```bash
export CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  cargo oxide run fmha_prefill_sm120a_cute --arch sm_120a -- \
  --shape-mode static --warmup 5 --iters 20
```

This builds and runs on GPU 1; the MLIR profile is selected automatically.
Each run checks every output against a CPU FP64 reference before timing.
Add `--noncausal` for unmasked attention, or use `--shape-mode runtime` for
dynamic shapes. Pass `--help` after `--` for all options.
The command prints CUDA-event timings; the comparison below uses Nsight Systems.

Measured on RTX 5090, **B=1, Sq=Sk=1024, H=8, D=64**, CUTLASS 4.7 on both
sides (2026-10-12). These are Nsight Systems kernel-active timings; each row
uses the same shape specialization in Rust and CuTeDSL.

| Shapes | Attention | CuTe-RS | CuTeDSL | Remaining gap |
| --- | --- | ---: | ---: | ---: |
| Runtime | Causal | 15.871 µs | 15.648 µs | 1.4% |
| Runtime | Noncausal | 16.127 µs | 15.680 µs | 2.9% |
| Static | Causal | 15.519 µs | 15.455 µs | 0.4% |
| Static | Noncausal | 15.584 µs | 15.456 µs | 0.8% |

The four gaps average **1.4%**, down from 4.8% before this profiling
work. This is an equal-weight summary; full parity is not yet established.

- [CuTeDSL reference and setup](reference/README.md)
- [Validation, benchmark commands, and detailed results](reference/measurement-notes.md)
- [Profiling findings and latest validation](reference/profiling-notes.md)
- [Dedicated decode example](../fmha_decode_sm120a_cute/README.md)
