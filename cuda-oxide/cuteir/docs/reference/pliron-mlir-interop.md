# Using upstream Pliron–MLIR interop

Upstream can provide an extensible printing foundation without CUDA-specific
changes. Replacing our current exporter still requires downstream integration;
this is not a dependency-only swap. Source review has not established any
mandatory upstream PR.

Reviewed on 2026-10-16 at upstream revision
[`665cc56d44fb0cfd689fc013cca5b35c4d1e0b12`](https://github.com/pliron-org/pliron-mlir-interop/tree/665cc56d44fb0cfd689fc013cca5b35c4d1e0b12).
This assessment includes a small extension probe, not a completed CUTLASS
compatibility test.
Start with [the current translation flow](../pliron-mlir-flow.md).

## What already works

Upstream provides `ToMlirOp`, `ToMlirType`, and `ToMlirAttr` interfaces.
Dialect authors implement them for their own operations, types, and attributes.
`MlirPrinter` calls those implementations and produces textual MLIR.
Builtin and LLVM translations are included.
[Source: translation interfaces](https://github.com/pliron-org/pliron-mlir-interop/blob/665cc56d44fb0cfd689fc013cca5b35c4d1e0b12/src/lib.rs).

An operation implementation has full control over its output. It can emit more
than one target operation. The shared Pliron printing state can hold local
translation data, including profile choices and bookkeeping for new values.
Upstream already uses that state for recursive type printing.
[Source: recursive types](https://github.com/pliron-org/pliron-mlir-interop/blob/665cc56d44fb0cfd689fc013cca5b35c4d1e0b12/src/llvm/types.rs).

The `GenericOp` helper covers common cases: changing the operation name,
selecting results, adding properties and attributes, and printing nested
regions. More involved lowering can use a local target IR or a custom printer.
[Source: generic operation printer](https://github.com/pliron-org/pliron-mlir-interop/blob/665cc56d44fb0cfd689fc013cca5b35c4d1e0b12/src/printers.rs).

## What stays in cuda-oxide

- Rust value layouts and signed/unsigned arithmetic decisions.
- MIR, CuTe, and NVVM mapping rules, including expansions into several ops.
- CuTe safety checks, source verification, and rejection of unsupported input.
- CUTLASS version profiles, GPU capabilities, launch ABI, and cubin validation.
- Tests that preserve arithmetic, memory behavior, and generated code.

These are compiler policies supplied through extensions. They do not belong
in upstream's generic printer.

## Integration details we must resolve

| Detail | What the source shows | Downstream approach |
| --- | --- | --- |
| Where implementations live | Rust cannot implement an upstream trait on a type owned by another crate from our separate exporter crate. | Put optional implementations in the owning dialect crates, or translate into locally owned target/adapter types. |
| Integer types | Upstream preserves signedness as `si32`, `ui32`, or `i32`. Our standard MLIR arithmetic uses signless types. | Choose signed/unsigned operations first, then lower the target types. Do not globally change upstream's integer policy. |
| Expanded operations | `GenericOp` uses source operands, types, and names; it is not our target syntax tree. | Preserve typed value/block bookkeeping in a downstream lowering layer. Avoid scattering hand-built strings through mapping recipes. |
| Source locations | The generic upstream printer does not currently emit operation or block-argument locations. | Preserve them through an adapter or contribute a generic location-printing facility. |
| Extra attributes | Dedicated builtin/LLVM printers select known attributes; for example, the module printer only selects `sym_name`. | Define which attributes survive and test preservation explicitly. |
| Dependency identity | Upstream tracks Pliron Git HEAD; its current source does not compile against our `06d99a3` pin. | Resolve one compatible Pliron source/revision across the dependency graph and validate the local compiler against it. |

The integer and module behavior is visible in upstream's
[builtin types](https://github.com/pliron-org/pliron-mlir-interop/blob/665cc56d44fb0cfd689fc013cca5b35c4d1e0b12/src/builtin/types.rs)
and [module printer](https://github.com/pliron-org/pliron-mlir-interop/blob/665cc56d44fb0cfd689fc013cca5b35c4d1e0b12/src/builtin/ops.rs).
Compare our [builtin mappings](../../../crates/cuda-oxide-mlir-export/src/packs.rs)
and [translation session](../../../crates/pliron-mlir-export/src/translate.rs).

A scratch build against our Pliron pin failed on changed upstream APIs,
including `BytesAttr` and attribute accessors. This is a dependency compatibility
issue to resolve locally; it does not establish a missing interop extension.

The separate [Pliron 0.19.0 workspace check](pliron-0.19-validation.md) records
the API adaptations needed by the tested experimental snapshot. The migration
subsequently landed on `main` in [PR #1526](https://github.com/NVIDIA/cuda-rust/pull/1526).
That migration does not establish interop compatibility with 0.19.0.

## What the extension probe established

With compatible Pliron revision `d6886bd2ff997dff11b13330c257cfb99308f35f`,
unchanged upstream interop source compiled and ran a scratch example containing
a custom type, custom operations, properties, and an operation expanding into
two target operations. A later operation consumed the expansion's final value.
Only the scratch dependency paths changed; our workspace and pins did not.

LLVM 22 `mlir-opt --allow-unregistered-dialect` accepted the output. This checks
generic MLIR structure and value connections, not the semantics of an
unregistered dialect or compatibility with CUTLASS. It demonstrates that custom
extensions and operation expansion do not inherently need an upstream PR.

## Which upstream PRs would make sense?

Only reusable facilities demonstrated by a small test case:

1. **Location printing:** preserve source locations for any dialect.
2. **Attribute preservation:** define how unconsumed discardable attributes
   survive dedicated builtin/LLVM printers.
3. **Optional external translation hooks:** let a separate client supply a
   translation policy for foreign types or operations, if downstream adapters
   otherwise duplicate the generic machinery.

The third item needs a design discussion and a concrete example. Existing
interfaces may be sufficient after local lowering. Additional builder methods
are conveniences to justify individually, not assumed migration blockers.

Keep CUDA names, CuTe semantics, architecture checks, and CUTLASS requirements
out of these PRs. Tests for generic facilities can use a tiny example dialect.

## Recommended next step

First align dependencies on a compatible Pliron revision and validate that
revision against our compiler. Then adapt one small mixed MIR/NVVM/CuTe kernel
using the unchanged upstream crate.
Check signed arithmetic, one operation expansion, locations, attributes, and
unknown-input rejection, then compile its output with the pinned CUTLASS
consumer. Compare it with the existing exporter before expanding the migration.

That experiment should decide which generic improvements deserve upstream PRs.
Do not delete our current exporter or relax its checks merely because the new
text parses.
