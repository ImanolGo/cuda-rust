# FP16 GEMM with Rust instruction primitives

Ports `cuteir/examples/fp16_gemm_256x352_cute` to the typed
`cuda_device::primitives` API without CuTe operations, layout algebra or Python
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
On an SM100 GPU, use that directory's `target/release/fp16_gemm_256x352_primitives`:

```sh
python3 crates/rustc-codegen-cuda/examples/fp16_gemm_256x352_primitives/validate.py \
  --binary /path/to/fp16_gemm_256x352_primitives \
  --output-dir /tmp/fp16-results --sanitizer
```

The six cases exercise tails, bias, persistent tiles, five-stage reuse and phase
wraps. The host also supports direct/graph timing and an exhaustive CPU oracle;
see `--help`. Compilation is verified locally; GPU correctness, sanitizer and
performance results require SM100 hardware. This kernel cannot execute on an
RTX 5090; use [`blockscale_gemm_primitives`](../../../../cuteir/examples/blockscale_gemm_primitives/README.md)
for SM120a.
