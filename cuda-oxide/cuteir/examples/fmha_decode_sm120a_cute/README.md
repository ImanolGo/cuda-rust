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

From `cuda-oxide/`, install CUTLASS 4.7 (`cargo oxide toolchain install cutlass`)
and prepare the shared Python environment once for both prefill and decode:

```bash
python3 -m venv cuteir/.venv
cuteir/.venv/bin/python -m pip install -r cuteir/examples/fmha_prefill_sm120a_cute/reference/requirements.txt
```

Then compare both specializations from `cuda-oxide/`:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide run fmha_decode_sm120a_cute --arch sm_120a -- \
  --all-cases --warmup 100 --iters 31
```

The command compares CuTeDSL and cute-rs on device 0; the MLIR profile is
selected automatically. Both implementations check every output against a CPU
FP64 reference, and the comparison verifies matching input hashes and output
parity before printing a table of direct CUDA-event medians for runtime and
static specialization. Both rows use noncausal single-token attention at the
selected shape. The signed relative delta is `(cute-rs / CuTeDSL - 1) * 100`;
positive means cute-rs took longer.

The default shape is **Sk=1024, H=8**. `--all-cases` requires one of the static
`(Sk,H)` shapes listed above, overrides `--specialization`, and cannot be
combined with `--rust-only`. For one case, replace `--all-cases` with
`--specialization static`; choose `runtime` for dynamic shapes.
Each event interval covers both partial and merge kernels and the gap between
them, and can include GPU idle time while the host submits those launches.

For a single case, add `--rust-only` to keep the standalone Rust validation
and timing report. `--json PATH` writes a combined comparison report by default,
with a `cases` array for `--all-cases`, and the standalone report with
`--rust-only`. `--iters 0` verifies both implementations without
collecting timings. Pass `--help` after `--` for all options.

The comparison requires CUTLASS DSL 4.7 and a compatible CUDA-enabled PyTorch
build; see [reference setup](reference/README.md) and
[requirements](reference/requirements.txt). Python is selected in order from
`--python PATH`, `CUDA_OXIDE_CUTE_PYTHON`, this example's `.venv/bin/python`,
`cuteir/.venv/bin/python`, then `python3`. The shared environment is found
automatically; no activation or Python environment-variable export is needed.
Nsight Systems is not required for the default comparison. The historical
results below use the separate Nsight Systems profiling harness.

Device 0 is the default. Add `--device N` after `--` to select another visible
device ordinal. Use `CUDA_VISIBLE_DEVICES` only when selecting or remapping
visible GPUs; `CUDA_DEVICE_ORDER=PCI_BUS_ID` controls their ordering and is
optional.

The historical table below records **Nsight Systems measurements** on RTX
5090, **B=1, Sq=1, Sk=1024, H=8, D=64**, CUTLASS 4.7 on both sides
(2026-10-12), using 100 warmups and 31 samples per implementation in each
of three alternating rounds. The CuTeDSL reference is independently authored here
using the same split-KV algorithm; it is not an upstream CUTLASS decode kernel.

| Shapes | CuTe-RS active sum | CuTeDSL active sum | CuTe-RS sequence span | CuTeDSL sequence span |
| --- | ---: | ---: | ---: | ---: |
| Runtime | 2.912 µs | 3.040 µs | 3.168 µs | 3.328 µs |
| Static | 2.400 µs | 2.208 µs | 2.688 µs | 2.624 µs |

Nsight Systems **active sum** adds the two kernel durations; **sequence span**
runs from the first kernel's start to the second kernel's end and includes
the gap between them. Direct CUDA-event intervals also cover launch submission
around the kernels, so they are a separate metric from both archived columns.
Comparing event times against this table does not establish a regression;
use the same method and sampling protocol to assess a change. See
[Nsight reproduction](reference/README.md). The static active-sum gap is
**8.7%**. The implementations have some scheduling differences, detailed below.

- [CuTeDSL reference and setup](reference/README.md)
- [Validation, benchmark commands, and detailed results](reference/measurement-notes.md)
- [Latest compiler regression checks](reference/compiler-regression-results.json) · [Earlier validation](reference/validation-results.json)
