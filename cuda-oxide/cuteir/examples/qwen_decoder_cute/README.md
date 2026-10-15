# Typed BF16 Qwen inference

This example implements `Qwen3-TTS-12Hz-1.7B-CustomVoice` on RTX 5090 / SM120
using `cute-rs` / `ctm-rs` operators. It includes typed Talker/predictor composition,
bounded checkpoint loading, CustomVoice prompt preparation, greedy code
generation and the GPU waveform decoder. Python supplies offline reference
fixtures and optional text tokenization; Rust executes all model tensor work.

```text
text IDs → embedding → biased linear → SiLU → biased linear
  → Talker: 28 decoder blocks → final norm → codec-0 head
  → predictor: [Talker last hidden, code-0 embedding]
      → biased 2048→1024 adapter → 5 blocks → codebook-specific head
      → repeat to obtain codes 1–15, using a fresh cache for each frame
  → sum all 16 code embeddings + next text/pad row → next Talker call
complete 16-code frames → codec embedding tables → transformer → upsampling → 24 kHz WAV
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

The original step 4c synthetic validation on SM120 checked 935,767
intermediate/output/cache comparisons and the full-depth wiring
fixtures pass. Maximum reference error is `1.52587891e-5`. The fresh-host-thread
allocation regression was observed failing before explicit context binding.
That synthetic suite also passed Compute Sanitizer memcheck, synccheck and
racecheck with zero errors/hazards; the existing FFN and RMSNorm GPU examples
passed after kernel reuse.

## Waveform decoder

`codec_model::Runner` owns 269 required decoder weight tensors, excluding the
encoder and the two unused quantizer input projections. The stored FP32 weights
are explicitly narrowed to BF16; codebook normalization then executes on the
GPU. Each operation borrows checked weights and owns its output `Batch<W>`.
Convolution input/output/kernel widths and linear projection roles are Rust
types. Ordinary and transposed checkpoint layouts remain distinct.

| Stage | Implementation contract |
| --- | --- |
| Codec lookup | 16 tables, vocabulary 2048, width 256; sequential BF16 residual-codebook sums; separate 256→512 projections for group 0 and groups 1–15 |
| Initial convolution | 512→1024, causal kernel 3 |
| Transformer | 8 layers, hidden 512, FFN 1024, 16 query/KV heads of width 64, causal 72-position window, RMS epsilon `1e-5` |
| First upsampling | Two factor-2 stages with ConvNeXt blocks, width 1024 |
| Waveform upsampling | Factors 8, 5, 4, 3; channels 1536→768→384→192→96; SnakeBeta and dilated residual convolutions |
| Output | One channel, clamp to `[-1,1]`, 1920 samples per code frame, 24 kHz |

The full decoder matches official 300-frame chunks with 25 frames of recomputed
left context. Positions restart for each chunk, and the duplicated waveform
prefix is cropped. Learned decoder operations run on the GPU. Host code builds
rotary tables and handles sample readback, cropping, exact BF16-to-FP32 widening
and WAV serialization.

Two observable rounding boundaries are explicit. BF16 convolution/transposed
convolution round the dot product before adding bias. The transformer's input
projection also rounds before bias because the official implementation feeds
PyTorch Linear a strided view of the convolution output. Other contiguous
linear projections use FP32 bias accumulation before BF16 narrowing. GPU
discriminators prevent a future fusion from accidentally changing these rules.

Run the pinned codec comparisons after provisioning the [official reference](reference/README.md):

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir cargo oxide run qwen_decoder_cute --arch sm_120a -- \
  --codec /tmp/qwen3-tts-acceptance/model /tmp/qwen3-tts-acceptance/reference-codec
```

The reference reader checks tensor shape, dtype, byte count, SHA-256 and path
containment. Checkpoint/config files are also hashed before loading. The codec
command writes JSON measurements to `/tmp/qwen-codec-metrics.json`; override
that path with `CUTE_CODEC_REPORT`. The recorded gate is relative RMS ≤ 0.05 for
every compared stage, with separate coverage flags for the attention window
and chunk boundary.

| Pinned fixture | Samples | Waveform relative RMS | Waveform SNR |
| --- | ---: | ---: | ---: |
| 2 deterministic frames | 3,840 | 0.01977 | 34.08 dB |
| 4 deterministic frames | 7,680 | 0.01216 | 38.30 dB |
| 3 generated frames | 5,760 | 0.01011 | 39.91 dB |
| 73 frames, crosses attention window | 140,160 | 0.02903 | 30.74 dB |
| 301 frames, crosses decoder chunk boundary | 577,920 | 0.03284 | 29.67 dB |

These five cases check 4,466,688 intermediate/waveform values. Quantizer output,
initial convolution and transformer input projection match the short fixtures
exactly. The full codec passes Compute Sanitizer memcheck; convolution/element
fixtures and 57,600 D64 attention outputs pass both memcheck and synccheck, with
zero reported errors. SNR here measures agreement with the reference waveform;
it is not a speech quality rating.

## Generation, sampling and measurement

The synchronous runner supports greedy or explicitly seeded host token
selection. `SamplingConfig<3072>` and `SamplingConfig<2048>` keep the Talker and
predictor vocabularies separate. Private validated parameters enforce finite
positive temperature, bounded top-k and top-p in `(0,1]`. Filtering follows
temperature → top-k → top-p, after Talker suppression/repetition handling.
Seeded streams are reproducible within this implementation, not bit-identical
to PyTorch's RNG. Omitting `sampling` from the request keeps greedy behavior.

```bash
/tmp/qwen3-tts-acceptance/venv/bin/python \
  cuteir/examples/qwen_decoder_cute/reference/tokenize_request.py \
  --model-dir /tmp/qwen3-tts-acceptance/model \
  --text 'Hello. This is a cuda-oxide acceptance test.' \
  --speaker Vivian --sample --seed 0 --output /tmp/qwen-request.json

CUDA_VISIBLE_DEVICES=GPU-04a6d226-f1d6-0fc8-a95a-3ede21981e72 \
  cuteir/examples/qwen_decoder_cute/target/release/qwen_decoder_cute \
  --generate /tmp/qwen3-tts-acceptance/model /tmp/qwen-request.json /tmp/qwen.wav
```

The 24 kHz mono WAV has an adjacent JSON report containing actual checkpoint
hashes, binary hash, request/policy/seeds, codec IDs, EOS/cap status, waveform
hash and timing boundaries. `--benchmark MODEL_DIR REQUEST_JSON OUTPUT_DIR 20`
retains weights across three warmups and twenty requests, requiring identical
codes and waveform hashes. The first delivered buffer is the full waveform;
incremental streaming is not implemented. See the [benchmark guide](reference/benchmark.md)
for qualified baselines and [recorded results](../../../docs/cuteir/feature-parity/qwen-acceptance-results.md)
for performance and quality limits.

The optimized one/two-row projection prefetches four MMA fragments while
preserving the original operation order. Its `--decode-linear` regression
checks 19,513 accuracy/bias cases and 27,703 exact original-MMA outputs, including
strides, tails and varied BF16 values. Typed weight/bias roles and synchronized
borrows remain unchanged. Memcheck, synccheck and racecheck pass.

Both Ryan test sentences pass complete-speech checks. Seeded Vivian examples
also pass, but **greedy Vivian repeats the greeting and hits the cap**. The
strict second-frame diagnostic retains a failed code decision at a zero-margin
BF16 boundary; sampling success does not close it. Broad voice/style/language
quality, speaker consistency, incremental delivery and SM100 execution remain
separate acceptance work.

Contracts follow the [pinned model config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/blob/0c0e3051f131929182e2c023b9537f8b1c68adfe/config.json)
and [pinned model implementation](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/models/modeling_qwen3_tts.py).
The waveform graph follows the [pinned 12 Hz codec implementation](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py).
