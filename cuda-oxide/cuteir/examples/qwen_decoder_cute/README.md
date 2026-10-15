# Typed BF16 Qwen decoder composition

This is the synthetic step 4c implementation for
`Qwen3-TTS-12Hz-1.7B-CustomVoice` on the local RTX 5090 / SM120. It composes
existing `cute-rs` / `ctm-rs` operators; it does not load a checkpoint or produce
speech yet. No model-specific compiler operations were added.

```text
text IDs → embedding → biased linear → SiLU → biased linear
  → Talker: 28 decoder blocks → final norm → codec-0 head
  → predictor: [Talker last hidden, code-0 embedding]
      → biased 2048→1024 adapter → 5 blocks → codebook-specific head
      → repeat to obtain codes 1–15, using a fresh cache for each frame
  → sum all 16 code embeddings + next text/pad row → next Talker call
```

| Decoder contract | Talker | Predictor |
| --- | --- | --- |
| Residual / FFN width | 2048 / 6144 | 1024 / 3072 |
| Query / KV heads; head width | 16 / 8; 128 | 16 / 8; 128 |
| Q / K / V projection width | 2048 / 1024 / 1024 | 2048 / 1024 / 1024 |
| Output projection | 2048→2048 | 2048→1024 |
| Rotary policy / position limit | Interleaved `[24,20,20]` / 32768 | Standard / 65536 |
| Cache lifetime | Persists across frames | New per frame; 2 seed rows + 14 appends |

Each block executes input RMSNorm, Q/K/V projections, per-head Q/K RMSNorm,
RoPE and cache append, causal GQA, output projection, residual addition,
post-attention RMSNorm, SwiGLU FFN and another residual addition. Final RMSNorm
is outside the layer loop. All layer projections are bias-free; the text and
predictor input adapters apply bias in FP32 before narrowing to BF16.

`cute-rs::decoder` checks projection/norm roles, residual and head widths,
embedding extents and vocabulary bounds. `Batch<W>` owns its BF16 allocation;
`DecoderCache<H,I,K>` owns a separate cache for every layer. A failed stack call
poisons the cache until reset. The runner synchronizes before reading borrowed
buffers and on every launch-error path. These types do not identify checkpoint
contents or prove arbitrary raw-pointer/warp-participation safety.

Source modules reuse the existing FFN and RoPE/attention examples and the shared
`examples/common/rmsnorm.rs`; they compile into this executable's own artifact.
Kernel source uses CTM primitives. The `cuda-device` dependency is required by
macro expansion, not called directly by authored kernels.

From `cuda-oxide/`:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir cargo oxide run qwen_decoder_cute --arch sm_120a
cargo test --offline --manifest-path cuteir/examples/qwen_decoder_cute/Cargo.toml
cargo clippy --offline --manifest-path cuteir/examples/qwen_decoder_cute/Cargo.toml --all-targets -- -D warnings
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  cuteir/examples/qwen_decoder_cute/target/release/qwen_decoder_cute
```

Validation covers both real model widths, an irregular H17/I33 two-layer stack,
65-token tails, whole/chunked/single-token equivalence, cache prefix/guard
preservation, empty work, reset, capacity rejection and context failure recovery.
Independent FP64 references check each operation from its actual GPU input and
the complete graph from its original input. Per-operation tolerance is
`2e-5 + 0.008*abs(reference)`; propagated graph tolerance is
`0.002 + 0.025*abs(reference)`. These are synthetic tolerances, not measured
checkpoint acceptance thresholds.

The full-depth wiring fixture uses identity residual blocks, a nonzero predictor
input adapter, distinct codebook tables and distinct heads. Its deterministic
code sequence detects reversed seed rows and incorrect table/head selection;
two successive frames verify fresh predictor caches and persistent Talker state.
Nonzero decoder math is checked separately at the actual widths. Text tests use
a three-token vocabulary at width 2048 to avoid a large redundant fixture table;
the typed text interface defaults to the checkpoint vocabulary of 151936.

On SM120, 935,767 intermediate/output/cache comparisons and the full-depth wiring
fixtures pass. Maximum reference error is `1.52587891e-5`. The fresh-host-thread
allocation regression was observed failing before explicit context binding.
The full example also passed Compute Sanitizer memcheck, synccheck and racecheck
with zero errors/hazards. The 33 host tests, strict clippy and dependency policy
checks pass; the existing FFN and RMSNorm GPU examples pass after kernel reuse.

This is a synchronous correctness baseline with validation readbacks and greedy
host selection. Checkpoint loading, official-reference code predictions, prompt
formatting, stochastic sampling, waveform decoding and throughput acceptance
remain subsequent work. No SM100 runtime or model-performance claim is made.

Contracts follow the [pinned model config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/blob/0c0e3051f131929182e2c023b9537f8b1c68adfe/config.json)
and [pinned model implementation](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/models/modeling_qwen3_tts.py).
