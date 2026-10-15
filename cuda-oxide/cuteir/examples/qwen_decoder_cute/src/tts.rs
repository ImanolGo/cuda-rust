/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! CustomVoice prompt construction and deterministic B=1 speech generation.
use crate::{
    checkpoint::QwenWeights,
    checkpoint_validation::Reference,
    generation::{self, TalkerWeights, TextWeights},
    model::{Batch, DecoderCache, Runner},
    operator::Result,
    sampling::{Sampler, SamplingConfig},
};
use cuda_core::CudaContext;
use cute_rs::Bf16;
use cute_rs::decoder::TokenIds;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Read, path::Path, time::Instant};

struct Request {
    text_ids: Vec<u32>,
    instruct_ids: Vec<u32>,
    speaker: String,
    language: String,
    non_streaming: bool,
    max_new_tokens: usize,
    sampling: SamplingPlan,
}

struct SamplingPlan {
    talker: SamplingConfig<3072>,
    predictor: SamplingConfig<2048>,
    seed: u64,
    enabled: bool,
}
impl SamplingPlan {
    fn parse(value: Option<&Value>) -> Result<Self> {
        let Some(value) = value else {
            return Ok(Self {
                talker: SamplingConfig::greedy()?,
                predictor: SamplingConfig::greedy()?,
                seed: 0,
                enabled: false,
            });
        };
        let seed = value["seed"]
            .as_u64()
            .ok_or("sampling requires an unsigned 64-bit seed")?;
        fn config<const V: usize>(value: &Value) -> Result<SamplingConfig<V>> {
            let temperature = value["temperature"]
                .as_f64()
                .ok_or("sampling requires temperature")? as f32;
            let top_k = usize::try_from(
                value["top_k"]
                    .as_u64()
                    .ok_or("sampling requires integer top_k")?,
            )?;
            let top_p = value["top_p"].as_f64().ok_or("sampling requires top_p")?;
            if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
                return Err("sampling top_p must be in (0,1] before FP32 conversion".into());
            }
            SamplingConfig::sample(temperature, top_k, top_p as f32)
        }
        Ok(Self {
            talker: config(&value["talker"])?,
            predictor: config(&value["predictor"])?,
            seed,
            enabled: true,
        })
    }
}

fn predictor_seed(seed: u64) -> u64 {
    let mut digest = Sha256::new();
    digest.update(b"cuda-oxide/qwen-predictor/splitmix64-v1");
    digest.update(seed.to_le_bytes());
    let digest = digest.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(bytes)
}
impl Request {
    fn parse(value: &Value) -> Result<Self> {
        let parse_ids = |name: &str, required: bool| -> Result<Vec<u32>> {
            let Some(ids) = value.get(name) else {
                if required {
                    return Err(format!("request requires {name}").into());
                } else {
                    return Ok(Vec::new());
                }
            };
            let ids = ids.as_array().ok_or("token IDs must be an array")?;
            if ids.len() > 32760 {
                return Err("tokenized input exceeds the Talker position limit".into());
            }
            ids.iter()
                .map(|id| {
                    let id = u32::try_from(
                        id.as_u64()
                            .ok_or("token IDs must be nonnegative integers")?,
                    )?;
                    if id >= 151936 {
                        return Err("text token exceeds Qwen vocabulary".into());
                    }
                    Ok(id)
                })
                .collect()
        };
        let text_ids = parse_ids("text_ids", true)?;
        if text_ids.len() < 9
            || !text_ids.starts_with(&[151644, 77091, 198])
            || !text_ids.ends_with(&[151645, 198, 151644, 77091, 198])
        {
            return Err("text_ids must include the official assistant wrapper and at least one content token".into());
        }
        let speaker = value["speaker"]
            .as_str()
            .ok_or("request requires a speaker")?
            .to_lowercase();
        let language = value
            .get("language")
            .map(|language| language.as_str().ok_or("language must be a string"))
            .transpose()?
            .unwrap_or("auto")
            .to_lowercase();
        if speaker.is_empty() || language.is_empty() {
            return Err("CustomVoice speaker and language cannot be empty".into());
        }
        let non_streaming = value
            .get("non_streaming_mode")
            .map(|v| v.as_bool().ok_or("non_streaming_mode must be boolean"))
            .transpose()?
            .unwrap_or(false);
        let max_new_tokens = usize::try_from(
            value
                .get("max_new_tokens")
                .map(|v| v.as_u64().ok_or("max_new_tokens must be an integer"))
                .transpose()?
                .unwrap_or(128),
        )?;
        if !(2..=4096).contains(&max_new_tokens) {
            return Err("max_new_tokens must be between 2 and 4096".into());
        }
        Ok(Self {
            text_ids,
            instruct_ids: parse_ids("instruct_ids", false)?,
            speaker,
            language,
            non_streaming,
            max_new_tokens,
            sampling: SamplingPlan::parse(value.get("sampling"))?,
        })
    }
}

struct PromptPlan {
    codec_ids: Vec<u32>,
    text_special: [u32; 3],
}
fn prompt_plan(request: &Request, config: &Value) -> Result<PromptPlan> {
    if config["tts_model_type"] != "custom_voice" {
        return Err("this runner requires the pinned CustomVoice model".into());
    }
    let id = |value: &Value, vocab: u32| -> Result<u32> {
        let id = u32::try_from(
            value
                .as_u64()
                .ok_or("configured token ID missing or invalid")?,
        )?;
        if id >= vocab {
            return Err("configured token exceeds vocabulary".into());
        }
        Ok(id)
    };
    let talker = &config["talker_config"];
    let speaker = id(&talker["spk_id"][&request.speaker], 3072)?;
    let mut language = if request.language == "auto" {
        None
    } else {
        Some(id(&talker["codec_language_id"][&request.language], 3072)?)
    };
    if matches!(request.language.as_str(), "auto" | "chinese")
        && let Some(dialect) = talker["spk_is_dialect"][&request.speaker].as_str()
    {
        language = Some(id(&talker["codec_language_id"][dialect], 3072)?);
    }
    let mut codec_ids = vec![
        id(
            &talker[if language.is_some() {
                "codec_think_id"
            } else {
                "codec_nothink_id"
            }],
            3072,
        )?,
        id(&talker["codec_think_bos_id"], 3072)?,
    ];
    if let Some(language) = language {
        codec_ids.push(language);
    }
    codec_ids.extend([
        id(&talker["codec_think_eos_id"], 3072)?,
        speaker,
        id(&talker["codec_pad_id"], 3072)?,
        id(&talker["codec_bos_id"], 3072)?,
    ]);
    Ok(PromptPlan {
        codec_ids,
        text_special: [
            id(&config["tts_bos_token_id"], 151936)?,
            id(&config["tts_eos_token_id"], 151936)?,
            id(&config["tts_pad_token_id"], 151936)?,
        ],
    })
}

fn next_code_with(logits: &[Bf16], previous: &[u32], sampler: &mut Sampler<3072>) -> Result<u32> {
    if logits.len() != 3072 {
        return Err("Talker logits require 3072 values".into());
    }
    let mut scores = vec![f32::NEG_INFINITY; 3072];
    for (index, logit) in logits.iter().enumerate() {
        let mut score = logit.to_f32();
        if !score.is_finite() {
            return Err(format!("nonfinite Talker logit at {index}").into());
        }
        if (index >= 2048 && index != 2150) || (index == 2150 && previous.len() < 2) {
            continue;
        }
        if previous.contains(&(index as u32)) {
            score = if score < 0.0 {
                score * 1.05
            } else {
                score / 1.05
            };
        }
        scores[index] = score;
    }
    sampler.select(&scores)
}

#[cfg(test)]
fn next_code(logits: &[Bf16], previous: &[u32]) -> Result<u32> {
    next_code_with(
        logits,
        previous,
        &mut Sampler::new(SamplingConfig::greedy()?, 0),
    )
}

struct PreparedPrompt {
    input: Batch<2048>,
    trailing: Batch<2048>,
    pad: Batch<2048>,
}

fn prepare(
    runner: &Runner,
    talker: &TalkerWeights<'_>,
    text: &TextWeights<'_>,
    request: &Request,
    config: &Value,
) -> Result<PreparedPrompt> {
    let plan = prompt_plan(request, config)?;
    let projected = generation::text_input(runner, text, TokenIds::new(&request.text_ids)?)?;
    let special = generation::text_input(runner, text, TokenIds::new(&plan.text_special)?)?;
    let codec = runner.embed(&talker.codec_embedding, TokenIds::new(&plan.codec_ids)?)?;
    let pad = runner.select(&special, &[2])?;
    let role = runner.select(&projected, &[0, 1, 2])?;
    let mut pad_bos = vec![2; plan.codec_ids.len() - 2];
    pad_bos.push(0);
    let text_tags = runner.select(&special, &pad_bos)?;
    let tag_indices = (0..u32::try_from(plan.codec_ids.len() - 1)?).collect::<Vec<_>>();
    let codec_tags = runner.select(&codec, &tag_indices)?;
    let tags = runner.add(&text_tags, &codec_tags)?;
    let mut input = runner.concat(&role, &tags)?;
    let codec_bos = runner.select(&codec, &[u32::try_from(plan.codec_ids.len() - 1)?])?;
    let trailing;
    if request.non_streaming {
        let content = (3..u32::try_from(request.text_ids.len() - 5)?).collect::<Vec<_>>();
        let content = runner.select(&projected, &content)?;
        let eos = runner.select(&special, &[1])?;
        let content = runner.concat(&content, &eos)?;
        let codec_pad = runner.select(&codec, &[u32::try_from(plan.codec_ids.len() - 2)?])?;
        let codec_pad = runner.select(&codec_pad, &vec![0; content.rows()])?;
        let content = runner.add(&content, &codec_pad)?;
        input = runner.concat(&input, &content)?;
        input = runner.concat(&input, &runner.add(&pad, &codec_bos)?)?;
        trailing = runner.select(&pad, &[0])?;
    } else {
        let first = runner.select(&projected, &[3])?;
        input = runner.concat(&input, &runner.add(&first, &codec_bos)?)?;
        let rest = (4..u32::try_from(request.text_ids.len() - 5)?).collect::<Vec<_>>();
        let rest = runner.select(&projected, &rest)?;
        trailing = runner.concat(&rest, &runner.select(&special, &[1])?)?;
    }
    if !request.instruct_ids.is_empty() {
        let instruction =
            generation::text_input(runner, text, TokenIds::new(&request.instruct_ids)?)?;
        input = runner.concat(&instruction, &input)?;
    }
    let capacity = input
        .rows()
        .checked_add(request.max_new_tokens - 1)
        .ok_or("Talker capacity overflow")?;
    if capacity > 32768 {
        return Err("prompt plus generation exceeds the Talker position limit".into());
    }
    Ok(PreparedPrompt {
        input,
        trailing,
        pad,
    })
}

fn load_json(path: &Path) -> Result<Value> {
    if fs::metadata(path)?.len() > 1024 * 1024 {
        return Err("request/config JSON exceeds 1 MiB".into());
    }
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

/// Compare the Rust-assembled prompt against the independently exported official
/// prompt. Existing runner and weights avoid another full checkpoint upload.
pub fn validate_prompt_with(
    runner: &Runner,
    weights: &QwenWeights,
    reference: &Reference,
    model_dir: &Path,
) -> Result<()> {
    let shape = reference.manifest()["tensors"]["prompt/text_ids"]["shape"]
        .as_array()
        .ok_or("reference text shape missing")?;
    let count = usize::try_from(
        shape
            .get(1)
            .and_then(Value::as_u64)
            .ok_or("reference text length missing")?,
    )?;
    let ids = reference
        .i64("prompt/text_ids", &[1, count])?
        .into_iter()
        .map(u32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let value = json!({"text_ids":ids,"speaker":reference.manifest()["speaker"],"language":reference.manifest()["language"],"max_new_tokens":reference.manifest()["generation"]["max_new_tokens"]});
    let request = Request::parse(&value)?;
    let prepared = prepare(
        runner,
        &weights.talker()?,
        &weights.text()?,
        &request,
        &load_json(&model_dir.join("config.json"))?,
    )?;
    for (name, batch) in [
        ("inputs_embeds", &prepared.input),
        ("trailing_text_hidden", &prepared.trailing),
        ("tts_pad_embed", &prepared.pad),
    ] {
        let actual = batch.read(runner.stream())?;
        let expected = reference.bf16(&format!("prompt/{name}"), &[1, batch.rows(), 2048])?;
        let (mut error, mut norm, mut max_abs) = (0.0f64, 0.0f64, 0.0f64);
        for (a, b) in actual.iter().zip(&expected) {
            let (a, b) = (f64::from(a.to_f32()), f64::from(b.to_f32()));
            if !a.is_finite() || !b.is_finite() {
                return Err("nonfinite prompt comparison".into());
            }
            error += (a - b) * (a - b);
            norm += b * b;
            max_abs = max_abs.max((a - b).abs());
        }
        let relative_l2 = (error / norm.max(f64::MIN_POSITIVE)).sqrt();
        println!("Rust prompt {name}: relative L2={relative_l2:.6}, max abs={max_abs:.6}");
        if relative_l2 > 0.01 {
            return Err(format!("Rust prompt {name} exceeds relative L2 0.01").into());
        }
    }
    Ok(())
}

/// Run deterministic CustomVoice generation from locally tokenized text. All
/// embeddings, projections, attention, predictor and waveform operations run
/// through the Rust GPU implementation; host code selects token IDs and writes WAV.
struct Engine {
    runner: Runner,
    weights: QwenWeights,
    codec: crate::codec_model::Runner,
    config: Value,
    load_seconds: f64,
    actual_artifacts: Value,
    binary_sha256: String,
    gpu: String,
}
impl Engine {
    fn load(model_dir: &Path) -> Result<Self> {
        let actual_artifacts = crate::checkpoint_validation::verify_pinned_artifacts(model_dir)?;
        let mut binary = fs::File::open(std::env::current_exe()?)?;
        let mut digest = Sha256::new();
        let mut bytes = [0u8; 65536];
        loop {
            let count = binary.read(&mut bytes)?;
            if count == 0 {
                break;
            }
            digest.update(&bytes[..count]);
        }
        let binary_sha256 = format!("{:x}", digest.finalize());
        let started = Instant::now();
        let config = load_json(&model_dir.join("config.json"))?;
        let context = CudaContext::new(0)?;
        let gpu = context.device_name()?;
        let stream = context.default_stream();
        let runner = Runner::new(stream.clone())?;
        let weights = QwenWeights::load(&stream, model_dir.join("model.safetensors"))?;
        let codec = crate::codec_model::Runner::load(
            stream,
            &model_dir.join("speech_tokenizer/model.safetensors"),
        )?;
        Ok(Self {
            runner,
            weights,
            codec,
            config,
            load_seconds: started.elapsed().as_secs_f64(),
            actual_artifacts,
            binary_sha256,
            gpu,
        })
    }
    fn synthesize(&self, request_value: &Value) -> Result<(Vec<f32>, Value)> {
        let request = Request::parse(request_value)?;
        // Separate RNG streams keep Talker choices independent of the number
        // of predictor draws. Equal seeds do not reproduce PyTorch's RNG.
        let mut talker_sampler = Sampler::new(request.sampling.talker, request.sampling.seed);
        let mut predictor_sampler = Sampler::new(
            request.sampling.predictor,
            predictor_seed(request.sampling.seed),
        );
        let (runner, weights, codec, config) =
            (&self.runner, &self.weights, &self.codec, &self.config);
        let stream = runner.stream();
        let synthesis = Instant::now();
        let prompt_started = Instant::now();
        let talker = weights.talker()?;
        let predictor = weights.predictor()?;
        let prompt = prepare(runner, &talker, &weights.text()?, &request, config)?;
        let prompt_seconds = prompt_started.elapsed().as_secs_f64();
        let capacity = prompt.input.rows() + request.max_new_tokens - 1;
        let mut cache = DecoderCache::<2048, 6144, 8>::new(stream, 28, capacity)?;
        let positions = (0..u32::try_from(prompt.input.rows())?)
            .map(|p| [p; 3])
            .collect::<Vec<_>>();
        let timer = Instant::now();
        let mut step =
            generation::talker_step(runner, &talker, &mut cache, &prompt.input, &positions)?;
        let mut talker_seconds = timer.elapsed().as_secs_f64();
        let mut predictor_seconds = 0.0;
        let mut sequences = vec![next_code_with(
            &step.logits.read(stream)?,
            &[],
            &mut talker_sampler,
        )?];
        let mut frames = Vec::<[u32; 16]>::new();
        let mut first_frame_seconds = None;
        let mut stop_reason = "max_new_tokens";
        // Preserve the official generate() schedule: each subsequent Talker call
        // completes the previously predicted code-zero token, then predicts another.
        for _ in 1..request.max_new_tokens {
            let code = sequences[sequences.len() - 1];
            if code == 2150 {
                stop_reason = "eos";
                break;
            }
            let timer = Instant::now();
            let code_zero = generation::code_zero_input(runner, &talker, code)?;
            let frame = generation::predictor_frame(
                runner,
                &predictor,
                &step.last_hidden,
                &code_zero,
                &mut predictor_sampler,
            )?;
            predictor_seconds += timer.elapsed().as_secs_f64();
            let mut codes = [0; 16];
            codes[0] = code;
            codes[1..].copy_from_slice(&frame.remaining_codes);
            let frame_index = frames.len();
            frames.push(codes);
            if first_frame_seconds.is_none() {
                first_frame_seconds = Some(synthesis.elapsed().as_secs_f64());
            }
            let text = if frame_index < prompt.trailing.rows() {
                runner.select(&prompt.trailing, &[u32::try_from(frame_index)?])?
            } else {
                runner.select(&prompt.pad, &[0])?
            };
            let input = runner.add(&frame.embedding_sum, &text)?;
            let position = u32::try_from(cache.len())?;
            let timer = Instant::now();
            step = generation::talker_step(runner, &talker, &mut cache, &input, &[[position; 3]])?;
            talker_seconds += timer.elapsed().as_secs_f64();
            let code = next_code_with(&step.logits.read(stream)?, &sequences, &mut talker_sampler)?;
            sequences.push(code);
            if code == 2150 {
                stop_reason = "eos";
                break;
            }
        }
        if frames.is_empty() {
            return Err("generation produced no codec frames".into());
        }
        let generation_seconds = synthesis.elapsed().as_secs_f64();
        let timer = Instant::now();
        let waveform = codec.decode(&frames)?;
        let codec_seconds = timer.elapsed().as_secs_f64();
        let synthesis_seconds = synthesis.elapsed().as_secs_f64();

        let audio_seconds = waveform.len() as f64 / crate::codec_model::SAMPLE_RATE as f64;
        // Hashing is a validation/reporting cost, outside inference timing.
        let waveform_sha256 = waveform_digest(&waveform)?;
        let report = json!({"model":"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice","model_revision":"0c0e3051f131929182e2c023b9537f8b1c68adfe","actual_artifacts":self.actual_artifacts,"binary_sha256":self.binary_sha256,"gpu":self.gpu,"cuda_visible_devices":std::env::var("CUDA_VISIBLE_DEVICES").ok(),"request":request_value,"mode":if request.sampling.enabled {"B=1 BF16 seeded sampling, synchronous"} else {"B=1 BF16 deterministic greedy, synchronous"},"sampling_rng":if request.sampling.enabled {Some(crate::sampling::RNG_VERSION)} else {None},"talker_seed":request.sampling.seed,"predictor_seed":predictor_seed(request.sampling.seed),"seed_derivation":"predictor uses first 8 little-endian bytes of SHA256(cuda-oxide/qwen-predictor/splitmix64-v1 || seed_le_u64)","ready_for_speedup_claim":false,"delivery_mode":"complete_waveform","incremental_streaming":false,"sample_rate":24_000,"samples":waveform.len(),"audio_seconds":audio_seconds,"waveform_f32le_sha256":waveform_sha256,"stop_reason":stop_reason,"code_zero_sequence":sequences,"codec_frames":frames,"model_load_seconds":self.load_seconds,"prompt_seconds":prompt_seconds,"talker_seconds":talker_seconds,"predictor_seconds":predictor_seconds,"generation_seconds":generation_seconds,"codec_seconds":codec_seconds,"synthesis_seconds":synthesis_seconds,"first_codec_frame_seconds":first_frame_seconds,"first_playable_audio_seconds":synthesis_seconds,"first_playable_audio_scope":"first complete waveform returned to host; speech quality assessed separately; no incremental delivery","waveform_available_seconds":synthesis_seconds,"real_time_factor":synthesis_seconds/audio_seconds,"codec_frames_per_second":frames.len()as f64/synthesis_seconds,"generation_codec_frames_per_second":frames.len()as f64/generation_seconds});
        Ok((waveform, report))
    }
}

fn waveform_digest(waveform: &[f32]) -> Result<String> {
    if waveform.is_empty() || waveform.iter().any(|sample| !sample.is_finite()) {
        return Err("benchmark requires a nonempty finite waveform".into());
    }
    let mut digest = Sha256::new();
    for sample in waveform {
        digest.update(sample.to_le_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn same_output(left: &Value, right: &Value) -> bool {
    [
        "codec_frames",
        "stop_reason",
        "samples",
        "waveform_f32le_sha256",
    ]
    .iter()
    .all(|key| !left[*key].is_null() && left[*key] == right[*key])
}

pub fn run(model_dir: &Path, request_json: &Path, output_wav: &Path) -> Result<()> {
    let request = load_json(request_json)?;
    Request::parse(&request)?;
    let engine = Engine::load(model_dir)?;
    let (waveform, mut report) = engine.synthesize(&request)?;
    crate::wav::write(
        output_wav,
        crate::codec_model::SAMPLE_RATE as u32,
        &waveform,
    )?;
    report["waveform"] = json!(output_wav);
    let report_path = output_wav.with_extension("json");
    fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "Wrote {} ({} samples), report {}",
        output_wav.display(),
        waveform.len(),
        report_path.display()
    );
    Ok(())
}

fn percentile(values: &[f64], fraction: f64) -> Result<f64> {
    if values.is_empty()
        || values.iter().any(|v| !v.is_finite())
        || !fraction.is_finite()
        || fraction <= 0.0
        || fraction > 1.0
    {
        return Err("percentile requires finite samples and fraction in (0,1]".into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * fraction).ceil() as usize).saturating_sub(1);
    Ok(sorted[index.min(sorted.len() - 1)])
}

/// Loaded weights persist across warmups/repetitions; all request work,
/// including cache allocation and waveform synthesis, remains in each sample.
pub fn benchmark(
    model_dir: &Path,
    request_json: &Path,
    output_dir: &Path,
    repetitions: usize,
) -> Result<()> {
    if !(2..=100).contains(&repetitions) {
        return Err("benchmark repetitions must be 2..=100".into());
    }
    let request = load_json(request_json)?;
    Request::parse(&request)?;
    let engine = Engine::load(model_dir)?;
    fs::create_dir_all(output_dir)?;
    for _ in 0..3 {
        engine.synthesize(&request)?;
    }
    let mut runs = Vec::new();
    for index in 0..repetitions {
        let (waveform, report) = engine.synthesize(&request)?;
        if runs
            .first()
            .is_some_and(|first| !same_output(first, &report))
        {
            return Err(
                "deterministic benchmark changed codes, stopping, or waveform between runs".into(),
            );
        }
        if index == 0 {
            crate::wav::write(&output_dir.join("output.wav"), 24_000, &waveform)?;
        }
        println!(
            "Benchmark {}/{repetitions}: {:.6}s",
            index + 1,
            report["synthesis_seconds"].as_f64().unwrap_or_default()
        );
        runs.push(report);
    }
    let latencies: Vec<_> = runs
        .iter()
        .map(|r| r["synthesis_seconds"].as_f64().unwrap())
        .collect();
    let rtfs: Vec<_> = runs
        .iter()
        .map(|r| r["real_time_factor"].as_f64().unwrap())
        .collect();
    let report = json!({"schema_version":2,"warmups":3,"repetitions":repetitions,"timing_boundary":"prepared token IDs to complete host waveform; includes prompt GPU work, caches, decoding, codec; excludes loading/tokenization/WAV file I/O and output validation","delivery_mode":"complete_waveform","incremental_streaming":false,"outputs_bit_identical":true,"percentile_estimator":"nearest rank (observed sample)","first_playable_audio_p50_seconds":percentile(&latencies,0.5)?,"first_playable_audio_p95_seconds":percentile(&latencies,0.95)?,"peak_gpu_memory_bytes":null,"latency_p50_seconds":percentile(&latencies,0.5)?,"latency_p95_seconds":percentile(&latencies,0.95)?,"real_time_factor_p50":percentile(&rtfs,0.5)?,"real_time_factor_p95":percentile(&rtfs,0.95)?,"real_time_factor_definition":"wall seconds / audio seconds; lower is better","runs":runs});
    fs::write(
        output_dir.join("benchmark.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn request() -> Value {
        json!({"text_ids":[151644,77091,198,9707,151645,198,151644,77091,198],"speaker":"Ryan","language":"English","max_new_tokens":4})
    }
    fn config() -> Value {
        json!({"tts_model_type":"custom_voice","tts_bos_token_id":151672,"tts_eos_token_id":151673,"tts_pad_token_id":151671,"talker_config":{"codec_think_id":2154,"codec_nothink_id":2155,"codec_think_bos_id":2156,"codec_think_eos_id":2157,"codec_pad_id":2148,"codec_bos_id":2149,"spk_id":{"ryan":3061,"eric":2875},"spk_is_dialect":{"ryan":false,"eric":"sichuan_dialect"},"codec_language_id":{"english":2050,"chinese":2055,"sichuan_dialect":2062}}})
    }
    #[test]
    fn waveform_validation_and_run_stability_include_codes_and_audio() {
        assert!(waveform_digest(&[]).is_err());
        assert!(waveform_digest(&[f32::NAN]).is_err());
        assert!(waveform_digest(&[f32::INFINITY]).is_err());
        let first = json!({"codec_frames":[[1,2]],"stop_reason":"eos","samples":2,"waveform_f32le_sha256":waveform_digest(&[0., 1.]).unwrap()});
        assert!(same_output(&first, &first));
        for (key, value) in [
            ("codec_frames", json!([[2, 1]])),
            ("stop_reason", json!("cap")),
            ("samples", json!(3)),
            (
                "waveform_f32le_sha256",
                json!(waveform_digest(&[0., -1.]).unwrap()),
            ),
        ] {
            let mut changed = first.clone();
            changed[key] = value;
            assert!(!same_output(&first, &changed));
        }
        assert!(!same_output(&json!({}), &json!({})));
    }
    #[test]
    fn benchmark_percentiles_use_observed_nearest_rank_and_reject_nonfinite() {
        assert_eq!(percentile(&[3., 1., 4., 2.], 0.5).unwrap(), 2.);
        assert_eq!(percentile(&[3., 1., 4., 2.], 0.95).unwrap(), 4.);
        assert!(percentile(&[], 0.5).is_err());
        assert!(percentile(&[f64::NAN], 0.5).is_err());
        assert!(percentile(&[1.], 1.5).is_err());
    }
    #[test]
    fn request_rejects_unwrapped_empty_out_of_vocab_and_unbounded_generation() {
        let good = request();
        assert_eq!(Request::parse(&good).unwrap().text_ids.len(), 9);
        for ids in [
            json!([]),
            json!([9707]),
            json!([151644, 77091, 198, 151936, 151645, 198, 151644, 77091, 198]),
        ] {
            let mut bad = good.clone();
            bad["text_ids"] = ids;
            assert!(Request::parse(&bad).is_err());
        }
        for limit in [0, 1, 4097] {
            let mut bad = good.clone();
            bad["max_new_tokens"] = json!(limit);
            assert!(Request::parse(&bad).is_err());
        }
    }
    #[test]
    fn request_defaults_absent_language_but_rejects_nonstring_language() {
        let mut value = request();
        value.as_object_mut().unwrap().remove("language");
        assert_eq!(Request::parse(&value).unwrap().language, "auto");
        for bad in [json!(null), json!(123), json!([]), json!(true)] {
            value["language"] = bad;
            assert!(Request::parse(&value).is_err());
        }
    }
    #[test]
    fn sampling_request_keeps_vocabularies_and_seed_checked() {
        let mut value = request();
        assert!(!Request::parse(&value).unwrap().sampling.enabled);
        value["sampling"] = json!({"seed":42,"talker":{"temperature":0.9,"top_k":50,"top_p":1.0},"predictor":{"temperature":0.9,"top_k":50,"top_p":1.0}});
        let parsed = Request::parse(&value).unwrap();
        assert!(parsed.sampling.enabled);
        assert_eq!(parsed.sampling.seed, 42);
        for field in ["seed", "talker", "predictor"] {
            let mut bad = value.clone();
            bad["sampling"][field] = Value::Null;
            assert!(Request::parse(&bad).is_err());
        }
        value["sampling"]["predictor"]["top_k"] = json!(2049);
        assert!(Request::parse(&value).is_err());
        value["sampling"]["predictor"]["top_k"] = json!(2048);
        value["sampling"]["talker"]["top_k"] = json!(3072);
        assert!(Request::parse(&value).is_ok());
        value["sampling"]["talker"]["top_p"] = json!(1.00000001);
        assert!(Request::parse(&value).is_err());
        // Domain separation must not shift one SplitMix stream by its increment,
        // including seed zero. The derivation is stable for reproducible runs.
        assert_ne!(predictor_seed(0), 0);
        assert_ne!(predictor_seed(0), 0x9e3779b97f4a7c15);
        assert_ne!(predictor_seed(0), predictor_seed(1));
        assert_eq!(predictor_seed(0), 1544247571153862893);
    }
    #[test]
    fn prompt_tags_follow_language_speaker_and_dialect_policy() {
        let mut value = request();
        assert_eq!(
            prompt_plan(&Request::parse(&value).unwrap(), &config())
                .unwrap()
                .codec_ids,
            [2154, 2156, 2050, 2157, 3061, 2148, 2149]
        );
        value["language"] = json!("Auto");
        assert_eq!(
            prompt_plan(&Request::parse(&value).unwrap(), &config())
                .unwrap()
                .codec_ids,
            [2155, 2156, 2157, 3061, 2148, 2149]
        );
        value["speaker"] = json!("Eric");
        assert_eq!(
            prompt_plan(&Request::parse(&value).unwrap(), &config())
                .unwrap()
                .codec_ids,
            [2154, 2156, 2062, 2157, 2875, 2148, 2149]
        );
        value["speaker"] = json!("missing");
        assert!(prompt_plan(&Request::parse(&value).unwrap(), &config()).is_err());
    }
    #[test]
    fn generation_policy_suppresses_tags_and_eos_then_applies_repetition_penalty() {
        let mut logits = vec![Bf16::from_f32(-10.0); 3072];
        logits[3061] = Bf16::from_f32(100.0);
        logits[2150] = Bf16::from_f32(99.0);
        logits[7] = Bf16::from_f32(1.0);
        logits[8] = Bf16::from_f32(0.98);
        assert_eq!(next_code(&logits, &[]).unwrap(), 7);
        assert_eq!(next_code(&logits, &[7]).unwrap(), 8);
        assert_eq!(next_code(&logits, &[7, 8]).unwrap(), 2150);
    }
}
