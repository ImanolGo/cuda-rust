# Single-token decode with CuTe-RS — SM120a (RTX 5090)

This cuda-oxide example computes attention for one query per head against an
existing FP16 KV cache. A dedicated kernel processes 64 cache entries per block;
a second kernel combines the partial softmax results and writes FP16 output.

- Batch size **1**, head dimension **64**, contiguous BSHD tensors, FP32 accumulation.
- Any positive cache length is supported, including partial final tiles.
- `--specialization auto` (default) uses static `(Sk,H)` shapes **(64,8), (65,3),
  (1024,8), (4097,8)** when available; otherwise it uses runtime dimensions.
  `static` requires one of those shapes; `runtime` keeps dimensions dynamic.
- The cache contains only visible tokens. Cache management, paging and grouped-query
  attention are outside this example.

From `cuda-oxide/`, with CUTLASS 4.7 installed (`cargo oxide toolchain install cutlass`):

```bash
export CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  cargo oxide run fmha_decode_sm120a_cute --arch sm_120a -- \
  --specialization static --warmup 5 --iters 20
```

This builds and runs on GPU 1 with **Sk=1024, H=8**; the MLIR profile is selected
automatically. Each run checks every output against a CPU FP64 reference before
timing. Use `--specialization runtime` for dynamic shapes, or pass `--help` after
`--` for all options. The command prints CUDA-event timings for both kernels
together; the comparison below uses Nsight Systems.

Measured on RTX 5090, **B=1, Sq=1, Sk=1024, H=8, D=64**, CUTLASS 4.7 on
both sides (2026-10-12). The CuTeDSL reference is independently authored here
using the same split-KV algorithm; it is not an upstream CUTLASS decode kernel.

| Shapes | CuTe-RS active sum | CuTeDSL active sum | CuTe-RS sequence span | CuTeDSL sequence span |
| --- | ---: | ---: | ---: | ---: |
| Runtime | 2.912 µs | 3.040 µs | 3.168 µs | 3.328 µs |
| Static | 2.400 µs | 2.208 µs | 2.688 µs | 2.624 µs |

Nsight Systems **active sum** adds the two kernel durations; **sequence span**
also includes the gap between them. The static active-sum gap is **8.7%**.
The implementations have some scheduling differences, detailed below.

- [CuTeDSL reference and setup](reference/README.md)
- [Validation, benchmark commands, and detailed results](reference/measurement-notes.md)
- [Latest compiler regression checks](reference/compiler-regression-results.json) · [Earlier validation](reference/validation-results.json)
