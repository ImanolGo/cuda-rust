# Qwen text to speech

This model turns text into a 24 kHz speech WAV using
`Qwen3-TTS-12Hz-1.7B-CustomVoice`. It runs on RTX 5090 / SM120a.
Python prepares text tokens; Rust runs the model and generates the audio.

## How it works

```text
text tokens
    │
    ▼
Talker ──► first audio code
    │
    ▼
Predictor ──► fifteen more codes
    │
    └── repeat to build the speech sequence
                       │
                       ▼
                waveform decoder ──► speech.wav
```

The Talker reads the text and earlier audio. The smaller predictor completes
one group of sixteen audio codes. The waveform decoder turns those groups into
samples you can play.

Inside each GPU operation, `cute-rs` describes the data flow:

```text
BF16 buffers → tensor views → layout partitions → register tiles
                                                     │
output buffer ◄────────── copies ◄──── reductions or matrix multiplication
```

Typed dimensions keep model widths and weight roles separate. Owned buffers and
completion checks keep storage alive until GPU work finishes. The
[execution walkthrough](reference/execution.md) shows how a Talker step reuses
this storage and where the CPU waits.

## Run the checks

From `cuda-oxide/`, with the cuda-oxide toolchain and CUTLASS backend installed:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide run qwen_decoder_cute --arch sm_120a
```

Without arguments, the program checks synthetic decoder inputs, cache updates,
and the complete Talker/predictor wiring. A successful run prints `PASS` lines
and ends with `Qwen structured generation: ... passed`. It does not load a
checkpoint or generate speech in this mode.

## Generate speech

You need the pinned checkpoint and a tokenized request. The
[reference guide](reference/README.md) describes the required files;
[request preparation](reference/implementation.md#generation-sampling-and-measurement)
shows how to tokenize text and choose a voice.

Using the existing local acceptance files, after building above:

```bash
cuteir/models/qwen_decoder_cute/target/release/qwen_decoder_cute \
  --generate /tmp/qwen3-tts-acceptance/model \
  /tmp/qwen3-tts-acceptance/request-full.json /tmp/qwen.wav
```

Listen to `/tmp/qwen.wav`. The adjacent JSON report records the generation time,
audio length and output checks. Replace the paths if your files live elsewhere.
Changing the request's text requires generating new token IDs too.

## Measure generation speed

```bash
cuteir/models/qwen_decoder_cute/target/release/qwen_decoder_cute \
  --benchmark /tmp/qwen3-tts-acceptance/model \
  /tmp/qwen3-tts-acceptance/request-full.json /tmp/qwen-benchmark 20
```

This warms up three times and measures twenty complete generations. Each
printed time covers prepared text tokens through the complete waveform;
model loading, tokenization and WAV saving are excluded. The median is
`latency_p50_seconds` in `/tmp/qwen-benchmark/benchmark.json`. The generated
speech is saved there as `output.wav`.

For reference comparisons, profiling and timing definitions, see the
[benchmark guide](reference/benchmark.md). Shapes, numerical contracts,
validation commands and recorded checks are in
[implementation and validation](reference/implementation.md).

Greedy generation with Vivian still has a known repetition failure. Broader
voice and language acceptance remains open. Audio is delivered as a complete
waveform; incremental playback is not implemented.
