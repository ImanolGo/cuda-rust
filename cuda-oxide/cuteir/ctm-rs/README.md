# CTM-style Rust primitives

`ctm-rs` provides the instruction-level programming model for cuda-oxide:
typed MMA operands, physical shared-memory formats, tensor maps, explicit
barriers, and collective TMEM allocation. It depends on `cuda-device` for
hardware intrinsics and common device types. It has no dependency on `cute-rs`
or `cute-layout`; the compiler continues to use the existing NVVM dialect.

```rust,ignore
use ctm_rs::{
    memory::{Mxf4Tile128, SharedRegion},
    mma::{Accumulator16x8, Mxf4A, Mxf4B, ScalePairA, ScalePairB},
    tma::{Coordinates2D, TensorMap},
};
```

The default crate is `no_std` and does not require CUDA host bindings. Enable
the optional `host` feature for `ctm_rs::host::tma::encode_2d` and its checked
driver descriptor encoding. That feature uses `cuda-core` directly.

Thread election, synchronization, instruction ordering, asynchronous completion,
and cross-thread lifetimes remain explicit responsibilities of the kernel
author. The Rust types check local operand roles, formats, shape choices, and
borrowed resource lifetimes; they do not infer a collective execution schedule.

## Migration from `cuda_device::primitives`

Use `ctm_rs::{...}` instead of `cuda_device::primitives::{...}` and add `ctm-rs`
as a dependency. The old module is removed to keep dependencies one-way.

- Typed TMEM guards live at `ctm_rs::tcgen05::tmem::TmemGuard`. They expose
  explicit `alloc_collective`/`dealloc_collective` operations and borrowed
  views. They do not elect a warp or insert synchronization. The original
  `cuda_device::TmemGuard` retains its existing convenience API separately.
- Import `ctm_rs::mbarrier::BarrierAddressExt` for `shared_address()` on a ready
  `cuda_device::ManagedBarrier`. Its returned address borrows the barrier;
  an uninitialized barrier does not implement the adapter.
- Host tensor-map encoding moves from `cuda_host::tma` to
  `ctm_rs::host::tma`, behind `features = ["host"]`.

The [coverage contract](docs/primitives-coverage.md) lists the implemented
instruction subset, safety responsibilities, compiler profile, and limitations.
Examples are the [SM120 block-scaled GEMM](../examples/blockscale_gemm_ctm)
and [SM100 FP16 GEMM](../../crates/rustc-codegen-cuda/examples/fp16_gemm_256x352_primitives).

## Validation

From `cuda-oxide/` with its pinned nightly:

```sh
cargo test -p ctm-rs --all-targets
cargo test -p ctm-rs --all-targets --features host
cargo test -p ctm-rs --doc
python3 scripts/check-primitives-api-compile.py --example api --output-dir /tmp/ctm-api
python3 scripts/check-primitives-api-compile.py --example blockscale --output-dir /tmp/ctm-blockscale
python3 scripts/check-primitives-api-compile.py --example fp16 --output-dir /tmp/ctm-fp16
```

Default tests cover physical layouts, register carriers, storage checks, and
compile-time rejection of invalid contracts. Host tests validate descriptor
arguments and driver-error handling. Compilation checks verify the actual
instruction and parameter ABI through the CUTLASS consumer. GPU correctness
and sanitizer commands live in each example's README.
