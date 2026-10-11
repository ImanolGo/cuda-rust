# Rust CUTLASS primitives

`ctm_rs` is a Rust programming API over the existing NVVM
instruction dialect. It has no CuTe layout algebra or Python dependency. The
initial consumer is `cutlass-primitives-mlir22`, pinned to CUTLASS compiler
4.7.0 / LLVM 22 / `sm_100a` or `sm_120a` and Pliron
`06d99a3df938394583715aacf5eadb65b4791c09`.

The covered subsets support two-CTA SM100 FP16 GEMM and SM120 MXFP4 GEMM.
They are not full PTX parity. Reference source snapshots:
CUTLASS `0b55a2f691d69981583568fd9eb69687b1f0de8a`, DKG
`b3f067d4db6ecac574dd5f1ba5ab0e9d2f05fdbf`. Their programming interfaces inform
this facade; the installed 4.7 compiler determines accepted MLIR.

## Coverage

| Public Rust API | Importer boundary | Pliron operation | MLIR mapping | Real consumer probe |
| --- | --- | --- | --- | --- |
| `SharedRegion<F>`, typed local/cluster addresses | Existing shared allocation, pointer projections and explicit CVTA | `mir.shared_alloc`, `nvvm.cvta_generic_to_shared_offset` / `nvvm.cvta_generic_to_cluster_shared_offset` | Aligned AS3 globals; shared/cluster offsets retain address-space bits | TMA and MMA probes; checked constructors execute in the compiled path |
| `TensorMap<f16,F>`, host `ctm_rs::host::tma::encode_2d` | Transparent wrapper around canonical descriptor; grid-constant metadata | MIR array/struct and reference parameter | 128-byte by-value descriptor, alignment 64, `nvvm.grid_constant` | One-map and four-map kernels; descriptor-only cubin frame size zero |
| `TmemGuard<_,N,CtaGroup2>`, borrowed `view`/`read_32x32` | Existing literal group-2 intrinsics | `nvvm.tcgen05_alloc_cg2`, `dealloc_cg2`, `relinquish_alloc_permit_cg2` | `nvvm.tcgen05.alloc`, `.dealloc`, `.relinquish_alloc_permit` | One explicit instruction per lifecycle step; no added barriers/election |
| `SmemOperand<Role,F>`, `InstructionDescriptor<S>`, `mma<S,U>` | Existing shared-A MMA intrinsic, literal kind/group/collector | `nvvm.tcgen05_mma` | `nvvm.tcgen05.mma`, f16, CTA group 2 | Eight issues: M256/N192 and M256/N160, each fill/use/lastuse/discard |
| `load_32x32b_x32`, explicit `load_wait`, unsafe `decode_f32` | Existing raw 32-result load | `nvvm.tcgen05_ld_32x32b_x32_raw`, `tcgen05_load_wait` | One vector load, 32 ordered scalar extracts, one explicit wait | Raw results stay by value through decoding; wait precedes decode |
| `tma::load_2d_cg2`, `store_2d`, `prefetch` | Existing generated TMA and prefetch intrinsics | Multicast CG2 2D load, 2D store, prefetch | `nvvm.cp.async.bulk.tensor.*`, descriptor prefetch | Three typed loads, one store; distinct signed coordinates and mask checked |
| `bulk_commit_group`, `bulk_wait_group_read<N>` | Existing literal wait-group intrinsic | `nvvm.cp_async_bulk_commit_group`, `wait_group_read` | Bulk commit and read wait | Explicit commit followed by wait groups 0 and 7 |
| Barrier init/invalidate, arrive, expect-tx, parity try-wait | Canonical `ManagedBarrier` methods and intrinsics | Shared mbarrier operations | `nvvm.mbarrier.*` | Tokens/predicate remain observable; no software wait loop inserted |
| `cluster::map_shared`, `arrive_remote`, `CtaMask<G>` | Existing mapa and remote-arrival intrinsics | `nvvm.mapa_shared_cluster`, explicit CVTA, cluster arrival | AS7 mapa; explicit generic-to-AS7 offset; cluster-scoped arrival | Actual Rust rank-1 remote barrier path and native/MLIR address regressions |
| Counted CTA barriers, cluster barriers, fences | Existing generated intrinsics | Counted sync/arrive; cluster arrive/wait; proxy and TCGen05 fences | Corresponding NVVM operations; CTA `aligned` is a boolean property | All exposed modes in sync probe |
| `warp::elect_sync`, `warp_id`, thread/block/cluster coordinates | Existing special-register and election importers | NVVM sreg operations and `elect_sync` | NVVM election and sreg reads | Both election results and representative coordinates stored |

`cute.kernel` remains kernel-discovery **metadata** alongside `gpu.kernel`:
the pinned compiler does not produce named kernel code with `gpu.kernel` alone.
No CuTe operations or types are registered by the primitives profile; nested
CuTe types/attributes are rejected before preparation can erase them.

The [SM120 block-scaled example](../../examples/blockscale_gemm_ctm/README.md)
ports the full three-stage persistent GEMM to register primitives. The
[SM100 FP16 example](../../examples/fp16_gemm_256x352_ctm/README.md)
uses the two-CTA TMEM path. Dynamic shared allocations lower to one external
AS3 global with the maximum requested alignment; neither path needs CuTe ops.

| SM120 addition | Instruction contract |
| --- | --- |
| `mma::{Mxf4A, Mxf4B, Accumulator16x8, ScalePairA, ScalePairB}` | Distinct operand roles, A4/B2/C4 register counts, paired UE8M0 scale bytes |
| `mma::mma_mxf4` | Canonical `v1:i1030` → `nvvm.mma.block_scale`, m16n8k64, E2M1, scale_vec 2X; explicit full-warp participation |
| `load_mxf4_a`, `load_mxf4_b_pair`, `store_f16` | Explicit ldmatrix.x4 and packed FP16/stmatrix.x2; kernel supplies physical addresses |
| `Mxf4Tile128`, `Mxf4ScaleTile`, `tma::load_2d` | B64 packed-byte tiles; unswizzled scales with 128-byte TMA alignment; CTA-local completion |

The profile rejects TCGen05 on SM120a and register MXFP4 on SM100a. Native LLVM
support for the new intrinsic is tested separately; libNVVM support is not
advertised. Existing full-CuTe translation remains a separate programming model.

## Types and explicit responsibilities

Physical formats distinguish A `[128,64]`, B `[96,64]` / `[80,64]` (128-byte
swizzle, 1024-byte stage alignment), and C `[128,32]` (64-byte swizzle, 512-byte
alignment). Collective M256 is not a local shared-memory extent. Descriptor
construction checks offsets and field ranges before canonical packing. Types
reject wrong element/format/operand roles, unsupported MMA shapes and group
mixing. TMEM views borrow their allocation guard and carry checked packed
row/column addresses. No destructor synchronizes or frees hardware resources.

A typical issue site, inside an unsafe kernel section after checked resource
construction, is:

```rust,ignore
use ctm_rs::{groups::CtaGroup2, tcgen05::{descriptors::*, mma::{self, collector_a}}};
type Tile = DenseSharedF16<CtaGroup2, 256, 192, KMajor, KMajor>;
let descriptor = InstructionDescriptor::<Tile>::new();
// dst, a and b carry matching checked group, format and role types.
mma::mma::<Tile, collector_a::Fill>(
    &dst, &a.k16::<0>(), &b.k16::<0>(), &descriptor, false,
);
// Explicitly commit/wait and coordinate consumers before reuse or readback.
```

Unsafe adoption requires live storage in the claimed address space. Kernel
authors still coordinate election, collective participation, publication,
barrier phases, collector sequencing, asynchronous completion and remote
lifetimes. Static types establish local contracts, not cross-thread schedules.
`SharedRegion::from_raw_parts` returns validation errors; the alternative
`from_raw_parts_or_trap` performs the same null/alignment/capacity checks and
traps on failure. Both have the same unsafe storage and lifetime requirements.
The trapping form avoids a pointer-bearing `Result` carrier when recovery is
not useful, as in the SM120 example.
The host encoder checks shape/pitch/base constraints and uses a separate
128-byte-aligned CUDA-driver output temporary before copying opaque bytes into
the canonical 64-byte-aligned kernel descriptor.

## Reproduce validation

From `cuda-oxide/` with its pinned nightly and installed compiler:

```sh
python3 scripts/check-primitives-api-compile.py --output-dir /tmp/primitives-api-evidence
CUDA_OXIDE_TEST_CUTLASS_COMPILER="$HOME/.cargo/cuda-oxide/toolchains/cutlass/4.7.0/x86_64-cu13/lib/libCutlassCompiler.so" \
  cargo test -p cuda-oxide-codegen cutlass_primitives -- --ignored
```

The runner builds the in-tree CLI/backend, uses a fresh fixture target, checks
actual imported markers and CFG instruction order, and records source status,
compiler/backend hashes, MLIR, MIR logs and a validated cubin. It never launches
the kernels. See the [Rust probes](../../../crates/rustc-codegen-cuda/tests/fixtures/primitives_api_compile/src/kernel.rs),
[type-rejection tests](../tests/primitives_safety.rs), and
[consumer tests](../../../crates/cuda-oxide-codegen/src/tests/cutlass/primitives.rs).

Compile-only evidence does not establish numerical correctness, progress or
SOL performance. The SM120 example includes a bit-exact GPU/CPU oracle and
persistent-grid/sanitizer runner. SM100 execution still requires compatible
hardware; the local RTX 5090 cannot run its cubin.
The MMA probe still has a 128-byte final stack frame for its decoded-output
carrier; the raw load carrier is removed from the decode input path. Register
pressure and final output materialization must be measured during the port.

Deferred: LIR, further CuTe equivalences, other architectures/groups/MMA kinds, arbitrary
physical layouts, token wait convenience APIs and full intrinsic parity. Native
APIs outside this table may exist without a primitives MLIR mapping. Unsupported
selectors and operations fail at export. Checked API lowering supports Direct
and Single Rust enum layouts; niche/unknown layouts and enum payloads needing
bool-byte or named-space-pointer storage adapters are rejected explicitly.
