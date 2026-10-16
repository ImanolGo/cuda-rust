/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Typed Qwen Talker, code predictor and waveform decoder with reference checks.
// Share the already validated kernels and owned-cache implementation. These
// source modules compile into this binary's own embedded device artifact.
#[path = "../../rope_cache_cute/src/attention.rs"]
mod attention;
#[path = "../../rope_cache_cute/src/attention_kernel.rs"]
mod attention_kernel;
#[path = "../../rope_cache_cute/src/attention_reference.rs"]
mod attention_reference;
mod checkpoint;
mod checkpoint_validation;
mod codec_attention;
mod codec_kernel;
mod codec_model;
mod codec_ops;
mod codec_validation;
#[cfg(test)]
mod cute_boundary;
mod decode_linear;
mod decode_linear_bench;
mod element_kernel;
#[path = "../../ffn_cute/src/kernel.rs"]
mod ffn_kernel;
mod generation;
mod generation_validation;
#[path = "../../rope_cache_cute/src/kernel.rs"]
mod kernel;
mod layer_validation;
mod model;
#[path = "../../common/rmsnorm.rs"]
mod norm_kernel;
mod norm_specialized;
#[path = "../../rope_cache_cute/src/operator.rs"]
mod operator;
mod predictor_graph;
mod predictor_graph_kernel;
mod reference;
#[path = "../../rope_cache_cute/src/rotary.rs"]
mod rotary;
#[path = "../../rope_cache_cute/src/reference.rs"]
mod rotary_reference;
mod sampling;
mod tts;
mod validation;
mod wav;

fn main() -> operator::Result<()> {
    use std::path::Path;
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        None=>validation::run(),
        Some("--decode-linear") if args.len()==2 => {
            let context=cuda_core::CudaContext::new(0)?;
            decode_linear::validate(context.default_stream())
        }
        Some("--decode-linear-bench") if args.len()==2 => {
            let context=cuda_core::CudaContext::new(0)?;
            decode_linear_bench::run(context.default_stream())
        }
        Some("--predictor-graph") if args.len()==3 => {
            let model_dir=Path::new(&args[2]);
            checkpoint_validation::verify_pinned_artifacts(model_dir)?;
            let context=cuda_core::CudaContext::new(0)?;
            let stream=context.default_stream();
            let weights=std::sync::Arc::new(checkpoint::QwenWeights::load(&stream,model_dir.join("model.safetensors"))?);
            predictor_graph::validate(stream,weights)
        }
        Some("--norm-specialized") if args.len()==2 => {
            let context=cuda_core::CudaContext::new(0)?;
            norm_specialized::validate(context.default_stream())
        }
        Some("--norm-specialized-bench") if args.len()==2 => {
            let context=cuda_core::CudaContext::new(0)?;
            norm_specialized::benchmark(context.default_stream())
        }
        Some("--codec-ops") if args.len()==2 => {
            let context=cuda_core::CudaContext::new(0)?;
            codec_ops::validate(context.default_stream())
        }
        Some("--codec-attention") if args.len()==2 => {
            let context=cuda_core::CudaContext::new(0)?;
            codec_attention::validate(context.default_stream())
        }
        Some("--checkpoint") if args.len()==4 =>checkpoint_validation::run(Path::new(&args[2]),Path::new(&args[3])),
        Some("--codec") if args.len()==4 =>codec_validation::run(Path::new(&args[2]),Path::new(&args[3])),
        Some("--prompt") if args.len()==4 => {
            let reference=checkpoint_validation::Reference::open(Path::new(&args[3]))?;
            reference.verify_artifacts(Path::new(&args[2]))?;
            let context=cuda_core::CudaContext::new(0)?;
            let stream=context.default_stream();
            let weights=checkpoint::QwenWeights::load(&stream,Path::new(&args[2]).join("model.safetensors"))?;
            let runner=model::Runner::new(stream)?;
            tts::validate_prompt_with(&runner,&weights,&reference,Path::new(&args[2]))
        }
        Some("--generate") if args.len()==5 =>tts::run(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("--benchmark") if args.len()==5 || args.len()==6 => {
            let repetitions=if args.len()==6 {args[5].parse()?} else {20};
            tts::benchmark(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4]),repetitions)
        }
        _=>Err("usage: qwen_decoder_cute [--decode-linear | --decode-linear-bench | --norm-specialized | --norm-specialized-bench | --predictor-graph MODEL_DIR | --codec-ops | --codec-attention | --checkpoint MODEL_DIR REFERENCE_DIR | --codec MODEL_DIR REFERENCE_DIR | --prompt MODEL_DIR REFERENCE_DIR | --generate MODEL_DIR REQUEST_JSON OUTPUT_WAV | --benchmark MODEL_DIR REQUEST_JSON OUTPUT_DIR [REPETITIONS]]".into()),
    }
}
