# CuTe-style GPU kernels in Rust

`cuteir` is part of cuda-oxide and lives in `cuda-oxide/cuteir/`.
It lets a Rust kernel describe GPU data with typed tensors and layouts.
It starts with two ideas:

- a **Tensor** is storage plus a layout;
- a **Layout** says where a logical coordinate lives in that storage.

That is enough to express operations such as “split this vector into
four-value tiles” or “copy this matrix tile into shared memory.” The compiler
can then choose how to lower those operations without asking the kernel author
to rewrite the pointer arithmetic by hand.

```text
plain storage + layout
          │
          ▼
        Tensor ──► tile ──► slice ──► copy / add / matrix multiply
```

## Semantic frontend

An **IR** is the compiler's working form of a program. A **dialect** is a named
set of operations in that IR. A **backend** turns the IR into code another
compiler or the GPU can consume.

The frontend keeps CuTe operations in `dialect-cute` while they are still
recognizable:

```text
Rust kernel using cute-rs
Tensor → zipped_divide → slice → copy / add / matrix multiply
                         │
                         ▼
              high-level dialect-cute
                         │
                         ▼
             preparation + verification
                         │
                         ▼
                code generation
```

The verifier checks where tensors come from, how scheduler and pipeline state
flows between operations, and whether TMA, MMA, and epilogue operations agree
on layouts and buffer ownership. Both backends use these checks before
lowering CuTe operations. Dynamic phases, lane participation, and pointer
lifetimes remain the caller's responsibility.

## What lives here

| Path | What it does |
| :--- | :--- |
| [`cute-layout`](cute-layout) | Layout math shared by the Rust API and compiler. |
| [`cute-rs`](cute-rs) | Device-facing Tensor, Layout, copy, pipeline, and matrix multiply-and-accumulate (MMA) types. |
| [`ctm-rs`](ctm-rs) | CTM-style typed instruction primitives, independent of CuTe layouts, over `cuda-device`. |
| [`dialect-cute`](dialect-cute) | CuTe operations and backend-neutral whole-module verification in Pliron. |
| [`elementwise_cute`](examples/elementwise_cute) | Adds two vectors through per-thread Tensor tiles. |
| [`swiglu_cute`](examples/swiglu_cute) | Fuses SiLU and multiplication through typed register transforms, with explicit FP32 arithmetic, FP16 output conversion and scalar tails on SM120a. |
| [`rmsnorm_cute`](examples/rmsnorm_cute) | Validates FP32/BF16 RMSNorm with full-warp reductions, explicit BF16 rounding, checked shapes, and typed vector-copy tails on SM120a. |
| [`ffn_cute`](examples/ffn_cute) | Composes BF16 tensor-core gate/up/down projections with typed weight roles, checked strides, explicit SiLU/product rounding, and intermediate validation on SM120a. |
| [`nvfp4_gemv_cute`](examples/nvfp4_gemv_cute) | Multiplies a packed FP4 matrix by a packed FP4 vector. |
| [`blockscale_gemm_cute`](examples/blockscale_gemm_cute) | Multiplies two block-scaled FP4 matrices with the Tensor Memory Accelerator (TMA) and tensor cores. |
| [`blockscale_gemm_ctm`](examples/blockscale_gemm_ctm) | The same SM120a GEMM through the typed `ctm-rs` instruction API. Its Python CuTeDSL and C++ comparison sources are in the shared [reference folder](examples/blockscale_gemm_cute/reference). |
| [`fp16_gemm_256x352_cute`](examples/fp16_gemm_256x352_cute) | SM100 two-CTA GEMM with typed shared tiles, TMEM operations, collector reuse, and cluster TMA pipelines through the CUTLASS translation backend. |
| [`fp16_gemm_256x352_ctm`](examples/fp16_gemm_256x352_ctm) | The same SM100 FP16 GEMM through the typed `ctm-rs` instruction API, with explicit TMA stages, MMA issue, and TMEM completion. |

The datacenter Blackwell attention examples target **SM100a/B200**:

| Example | Matched CuTe-RS and pinned CuTeDSL implementations |
| :--- | :--- |
| [`sm100_fmha_prefill_cute`](examples/sm100_fmha_prefill_cute) | FP16 D128 prefill, dense or causal MHA/GQA, with FP32 accumulation and persistent warp specialization. |
| [`sm100_mla_decode_fp16_cute`](examples/sm100_mla_decode_fp16_cute) | FP16 MLA decode with a paged latent cache, two-CTA clusters, and optional split-KV reduction. |

Each example vendors its CUTLASS source and helpers under `reference/`, checks
both outputs against an independent oracle, and compares complete invocations
with the same event-timing policy. Kernel names distinguish `_cute_rs` and
`_cutedsl`. See the example READMEs for supported shape profiles and validation
commands. These use CuTe layouts plus NVVM operations; `cute_ext` GQA examples
belong to the separate LIR programming model.

The RTX 5090 attention examples target **SM120a**. They use `cute_rs` for
layouts and copies, and `ctm_rs` for instruction-level operations:

| Example | Purpose |
| :--- | :--- |
| [fmha_prefill_sm120a_cute](examples/fmha_prefill_sm120a_cute) | FP16 D64 prefill with causal or noncausal attention. |
| [fmha_decode_sm120a_cute](examples/fmha_decode_sm120a_cute) | Single-query FP16 D64 decode with partial-result reduction. |

The original examples are deliberately ordered from smallest to largest:

```text
elementwise       GEMV                    GEMM
vector tiles  →   scaled row tiles   →    global/shared matrix tiles
load + add         load + dot             TMA + pipeline + MMA + epilogue
```

## Maintaining the CuTe implementation

The Rust API, importer, dialect and exporter have different responsibilities:

| Location | Responsibility |
| :--- | :--- |
| [cute-rs/src/](cute-rs/src) | Public kernel abstractions, with private implementation modules for layouts, fragments, MMA, copies and pipelines. Existing root and family imports remain the public API. |
| [cute-layout/src/validation/](cute-layout/src/validation) | Independent proofs for cooperative copies, thread/value coverage, matrix loads and TMA layouts. |
| [mir-importer/src/translator/cute/](../crates/mir-importer/src/translator/cute) | Exact compiler-entry registration, defining-path recognition and operation dispatch. Each family groups its type decoder and emitter. |
| [dialect-cute/src/](dialect-cute/src) | Backend-neutral operations, attributes and types. `verify/` separates shared control-flow and value-provenance analysis from operation-family protocol checks. |
| [cuda-oxide-mlir-export/src/cute/](../crates/cuda-oxide-mlir-export/src/cute) | Mapping recipes grouped by tensor, copy, MMA, pipeline, epilogue and scheduler, with architecture-specific implementations inside those families. |

CuTe-RS's compiler boundary uses **defining paths**, not public reexport paths.
Moving a type or an inherent implementation into a child module changes its
compiler identity even when the old Rust import remains valid. Keep compiler
hooks in their existing `__compiler` modules. When moving a recognized type or
method, update the importer's exact type/layout decoders and method classifier,
and the backend collector's exact implementation-owner paths together. Preserve
the closed source-abstraction policy; accepting arbitrary suffix matches can
silently translate a compiler hook as ordinary Rust.

Validate such changes with importer and collector identity tests, compile-fail
API tests, and real kernel compilation. Host tests alone do not exercise the
compiler's recognition of device calls. Compare emitted MLIR and instructions
for representative tensor, copy, GEMV, GEMM and attention kernels.

Keep rustdoc short and close to the API: explain its purpose, supported shapes,
units and caller obligations. Add a small Rust example when it makes usage or
ordering clearer, and check it as a doctest where possible. Link shared details
instead of repeating them in each module.

## Run the examples

Run these commands from the `cuda-oxide/` directory after completing the normal
cuda-oxide setup:

```bash
cargo oxide run elementwise_cute

cargo oxide run nvfp4_gemv_cute --arch sm_120a -- \
  --m 512 --k 256 --l 1

cargo oxide run blockscale_gemm_cute --arch sm_120a
```

The elementwise example uses ordinary `f32` and `f16` operations. The FP4
examples use Blackwell instructions and should be built for `sm_120a`; their
own READMEs give the exact device and shape requirements.

The elementwise, GEMV, and both block-scaled GEMM examples compare every GPU
output bit pattern with a fixed host result. The FP16 example checks outputs numerically; its README
includes the SM100 build and validation commands. Tests for generated code
and artifacts live with each backend.

The [official CUTLASS backend](docs/translation.md) maps CuTe operations
directly to the CUTLASS 4.7 MLIR profile and embeds the resulting cubin.
