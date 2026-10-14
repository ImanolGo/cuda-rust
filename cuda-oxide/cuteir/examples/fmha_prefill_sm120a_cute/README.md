# FP16 prefill with CuTe-RS — SM120a (RTX 5090)

This cuda-oxide example computes `softmax(Q @ K.T / sqrt(64)) @ V` using
FP16 inputs, FP32 accumulation, and 64-by-64 tiles.

The kernel uses `cute_rs` for typed layouts, copies, and matrix fragments,
and `ctm_rs` for instruction-level operations, including the FP16 MMA behind
CuTe's accumulator wrapper. CTM also exposes the kernel macros and canonical
storage types. The manifest retains `cuda-device` because the macro expansions
still refer to that crate internally.

- Batch size **1**, head dimension **64**, contiguous BSHD tensors.
- Sequence lengths are positive multiples of 64. **Causal** (default) requires
  equal query/KV lengths; `--noncausal` permits unequal lengths and also `Sq=1`.
- Shapes are runtime parameters by default. `--shape-mode static` specializes
  **Sq=Sk=1024, H=8**; other shapes require runtime mode.

From `cuda-oxide/`, install CUTLASS 4.7 (`cargo oxide toolchain install cutlass`)
and prepare the shared Python environment once for both prefill and decode:

```bash
python3 -m venv cuteir/.venv
cuteir/.venv/bin/python -m pip install -r cuteir/examples/fmha_prefill_sm120a_cute/reference/requirements.txt
```

Then compare all four cases from `cuda-oxide/`:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide run fmha_prefill_sm120a_cute --arch sm_120a -- \
  --all-cases --warmup 100 --iters 31
```

The command compares CuTeDSL and cute-rs on device 0; the MLIR profile is
selected automatically. Both implementations check every output against a CPU
FP64 reference, and the comparison verifies matching input hashes and output
parity before printing a table of direct CUDA-event medians for runtime/static
shapes crossed with causal/noncausal attention. The signed relative delta is
`(cute-rs / CuTeDSL - 1) * 100`; positive means cute-rs took longer.

`--all-cases` requires **Sq=Sk=1024, H=8**, overrides `--shape-mode` and the
causal/noncausal selection, and cannot be combined with `--rust-only`.
For one case, replace `--all-cases` with `--shape-mode static --causal`;
choose `runtime` or `--noncausal` for the other cases. Each event interval
covers one attention launch.

For a single case, add `--rust-only` to keep the standalone Rust validation
and timing report.
`--json PATH` writes a combined comparison report by default and the standalone
report with `--rust-only`. `--iters 0` verifies both implementations without
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

The historical table below records **Nsight Systems kernel-active durations**
on RTX 5090, **B=1, Sq=Sk=1024, H=8, D=64**, CUTLASS 4.7 on both sides
(2026-10-12). Each row uses matching shape specialization, 100 warmups and
31 measured launches per implementation in each of three alternating rounds;
the result is the median of round medians.

The current command prints **direct CUDA-event intervals**. They can include
GPU idle time while the host submits the launch between the start and end
events; Nsight kernel-active durations measure the kernel's execution alone.
An event result of 18 or 23 µs therefore cannot establish a regression against
the archived roughly 15 µs values. A run with `--warmup 5 --iters 20` also has
a different sampling protocol. Use the same timing method and sampling
protocol to assess a change; see [Nsight reproduction](reference/README.md).

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
- [CTM migration: correctness, instruction equivalence, and matched timings](reference/ctm-migration-results.json)
- [Dedicated decode example](../fmha_decode_sm120a_cute/README.md)
