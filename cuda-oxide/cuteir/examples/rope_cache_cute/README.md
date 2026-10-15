# BF16 rotary embeddings and owned KV cache

This cuda-oxide example starts transformer integration with already projected
and normalized Q/K, unrotated V, and explicit positions. It uses `cute-rs` for
BF16 storage and checked attention geometry, and `ctm-rs` for kernel interfaces
and instruction primitives. It does not yet compute attention or run a model.

`AttentionShape<QH,KVH,D>` keeps query heads, KV heads and head width independent
of hidden width. `CacheShape<KVH,D>` checks capacity and append token ranges.
The example's `KvCache<KVH,D>` privately owns K/V allocations; `RotatedQuery<QH,D>`
retains query dimensions. `HostTables<D>` prevents mixing different rotary head
widths. Dynamic input lengths, grouped heads, byte extents, contexts, positions
and launch sizes are checked before execution.

The runner rotates split-half pairs: coordinate `i` pairs with `i+D/2`, not
with its adjacent coordinate. Both BF16 products round before their BF16 sum.
The predictor uses ordinary RoPE; Talker interleaved MRoPE uses `[24,20,20]`
sections at D=128. Height frequencies occupy indices 1,4,…,58 and width
frequencies occupy 2,5,…,59; all remaining frequencies use temporal positions.
Three-axis positions remain separate from physical cache slots.

The cache stores rotated K and unrotated V in dense `[token,head,dimension]`
order. An append borrows inputs and the cache until GPU completion, then advances
the valid-token cursor. Input readiness is established with a context wait; all
launched work is synchronized even on enqueue-error paths. Validation failure
does not change the cursor or completed prefix. Reset discards the logical
prefix without promising to erase old allocation bytes. Empty appends launch
no kernels. This is a conservative synchronous API, without paging or eviction.

From `cuda-oxide/`:

```bash
cargo test --offline -p cute-rs --features host --all-targets
cargo test --offline -p cute-rs --features host --doc
cargo test --offline --manifest-path cuteir/examples/rope_cache_cute/Cargo.toml
CARGO_NET_OFFLINE=true CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide run rope_cache_cute --arch sm_120a
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  cuteir/examples/rope_cache_cute/target/release/rope_cache_cute
/usr/local/cuda/bin/compute-sanitizer --tool synccheck --error-exitcode 1 \
  cuteir/examples/rope_cache_cute/target/release/rope_cache_cute
```

Validation covers Q16/KV8/D128 for both rotary modes, unequal position axes,
the last supported positions, Q6/KV2/D6 tails, empty input, cache exhaustion,
invalid lengths, reset and pending writes on another stream. One-shot and
chunked appends must match. An independent FP64 oracle with integer BF16
rounding checks Q/K bits exactly; V bits must remain unchanged. Intermediate
snapshots check every completed prefix, unwritten slot and trailing guard.
An exact fixture requires `0x3eb4` where incorrectly rounding only once produces
`0x3eb5`.

The host computes FP32 angles and trigonometry, then narrows coefficients to
BF16. GPU validation checks rotation against those stored coefficients. Host
trigonometry has not been compared bit-for-bit with CUDA/PyTorch table generation;
checkpoint/reference execution is still required for model correctness. Local
execution targets RTX 5090 / SM120; no SM100 execution or performance claim is made.

Validated on RTX 5090 with CUTLASS 4.7 on 2026-10-15: 10 example host tests,
582,652 BF16 reference values, and zero errors from both memcheck and synccheck.
The pending-input case exercises ordering but is not an isolated proof of the
context wait: allocation may also serialize prior input writes.
