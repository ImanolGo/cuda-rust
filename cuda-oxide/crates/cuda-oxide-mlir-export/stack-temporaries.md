# Why move stack temporaries to the function entry?

LLVM can often replace temporary memory with ordinary values. Its scalar
replacement (SROA) and memory-to-register (mem2reg) passes look for allocations
in the function's entry block.

Simple pseudocode:

```text
Before:
    for each x:
        temp = stack_alloc<int>()
        store(temp, x + 1)
        consume(load(temp))

After hoisting:
    temp = stack_alloc<int>()
    for each x:
        store(temp, x + 1)
        consume(load(temp))

After LLVM optimizes:
    for each x:
        consume(x + 1)
```

Only the allocation moves initially. Computation, writes and reads stay in the
loop. LLVM can then remove the unnecessary store/load pair and the unused slot.
Entry placement enables this optimization; it does not guarantee elimination.

Our `hoist_enum_scratch` in [`mir_checked.rs`](src/mir_checked.rs) applies this
to fixed-size, compiler-owned enum temporaries whose addresses never escape.
It preserves checks and enum layout, and leaves ordinary Rust allocations alone.

See [LLVM's guidance on allocas](https://llvm.org/docs/Frontend/PerformanceTips.html#use-of-allocas).
