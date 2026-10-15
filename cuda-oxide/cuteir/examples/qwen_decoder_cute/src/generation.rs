/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed text, Talker and code-predictor boundaries for deterministic validation.
//!
//! Dimensions and operation order follow the pinned Qwen3-TTS CustomVoice
//! configuration. This module consumes loaded weights and prepared prompt rows;
//! it does not load a checkpoint, format a speech prompt or implement sampling.

use crate::model::{Batch, DecoderCache, Runner};
use crate::operator::Result;
use crate::rotary::RotarySpec;
use cute_rs::Bf16;
use cute_rs::decoder::{
    BiasWeights, DecoderWeights, EmbeddingWeights, FinalNorm, NormWeights, TokenIds,
};
use cute_rs::ffn::{InputProjection, Logits, ProjectionWeights, TextProjection1, TextProjection2};

pub const TALKER_LAYERS: usize = 28;
pub const PREDICTOR_LAYERS: usize = 5;
pub const REMAINING_CODES: usize = 15;

/// The Talker preserves 2048-wide hidden states across 28 decoder layers.
pub struct TalkerWeights<'a> {
    pub layers: Vec<DecoderWeights<'a, 2048, 6144, 2048, 1024, 128>>,
    pub norm: NormWeights<'a, FinalNorm, 2048>,
    pub codec_head: ProjectionWeights<'a, Logits, 2048, 3072>,
    pub codec_embedding: EmbeddingWeights<'a, 3072, 2048>,
}

/// Predictor embeddings retain the Talker's width, then a biased projection
/// maps 2048 to 1024 before its five decoder layers. Each code group has its
/// own output head and embedding table.
pub struct PredictorWeights<'a> {
    pub layers: Vec<DecoderWeights<'a, 1024, 3072, 2048, 1024, 128>>,
    pub norm: NormWeights<'a, FinalNorm, 1024>,
    pub projection: ProjectionWeights<'a, InputProjection, 2048, 1024>,
    pub bias: BiasWeights<'a, InputProjection, 1024>,
    pub embeddings: [EmbeddingWeights<'a, 2048, 2048>; REMAINING_CODES],
    pub heads: [ProjectionWeights<'a, Logits, 1024, 2048>; REMAINING_CODES],
}

/// Text tokenizer rows pass through a biased two-layer SiLU adapter before
/// participating in the Talker's prompt or per-frame input.
pub struct TextWeights<'a, const VOCAB: usize = 151936> {
    pub embedding: EmbeddingWeights<'a, VOCAB, 2048>,
    pub first: ProjectionWeights<'a, TextProjection1, 2048, 2048>,
    pub first_bias: BiasWeights<'a, TextProjection1, 2048>,
    pub second: ProjectionWeights<'a, TextProjection2, 2048, 2048>,
    pub second_bias: BiasWeights<'a, TextProjection2, 2048>,
}

/// Only the final input row predicts the next code-zero token.
pub struct TalkerStep {
    pub last_hidden: Batch<2048>,
    pub logits: Batch<3072>,
}

/// The caller already supplied code zero. These are codes one through fifteen,
/// plus the sum of all sixteen embedding vectors for the next Talker input.
pub struct CodecFrame {
    pub remaining_codes: [u32; REMAINING_CODES],
    pub embedding_sum: Batch<2048>,
}

fn check_layers(actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(format!("model requires {expected} decoder layers, received {actual}").into());
    }
    Ok(())
}

/// Exact position schedule for a fresh code predictor invocation. Hidden state
/// and code-zero embedding occupy rows zero and one. Codes one through fourteen
/// are subsequently appended; code fifteen ends the frame without another append.
fn predictor_positions(step: usize) -> Result<Vec<[u32; 3]>> {
    if step >= REMAINING_CODES {
        return Err("predictor step exceeds the fifteen remaining code groups".into());
    }
    Ok(if step == 0 {
        vec![[0; 3], [1; 3]]
    } else {
        vec![[(step + 1) as u32; 3]]
    })
}

/// Deterministic validation policy. Nonfinite logits are errors, and the lowest
/// vocabulary index wins a tie. This does not reproduce stochastic sampling.
pub fn greedy_token<const VOCAB: usize>(logits: &[Bf16]) -> Result<u32> {
    if VOCAB == 0 || VOCAB > u32::MAX as usize || logits.len() != VOCAB {
        return Err("greedy logits must match a nonempty u32-indexed vocabulary".into());
    }
    let mut best_value = f32::NEG_INFINITY;
    let mut best_index = 0;
    for (index, value) in logits.iter().enumerate() {
        let value = value.to_f32();
        if !value.is_finite() {
            return Err(format!("nonfinite logit at vocabulary index {index}").into());
        }
        if value > best_value {
            best_value = value;
            best_index = index;
        }
    }
    Ok(best_index as u32)
}

pub fn text_input<const VOCAB: usize>(
    runner: &Runner,
    weights: &TextWeights<'_, VOCAB>,
    ids: TokenIds<'_, VOCAB>,
) -> Result<Batch<2048>> {
    let embedded = runner.embed(&weights.embedding, ids)?;
    let first = runner.project(&embedded, &weights.first, Some(&weights.first_bias))?;
    let activated = runner.silu(&first)?;
    runner.project(&activated, &weights.second, Some(&weights.second_bias))
}

/// Append prepared prompt or generation rows to the persistent Talker cache.
/// Explicit three-axis positions permit the pinned interleaved MRoPE policy;
/// prompt formatting and the choice of positions belong to the caller.
/// If the output projection fails after the decoder succeeds, the cache retains
/// the completed prefix but is invalidated until reset.
pub fn talker_step(
    runner: &Runner,
    weights: &TalkerWeights<'_>,
    cache: &mut DecoderCache<2048, 6144, 8>,
    input: &Batch<2048>,
    positions: &[[u32; 3]],
) -> Result<TalkerStep> {
    check_layers(weights.layers.len(), TALKER_LAYERS)?;
    if input.rows() == 0 {
        return Err("a Talker prediction requires at least one input row".into());
    }
    if !std::sync::Arc::ptr_eq(
        weights.codec_head.buffer().context(),
        runner.stream().context(),
    ) {
        return Err("Talker codec head must share the runner context".into());
    }
    let last_row = u32::try_from(input.rows() - 1)?;
    let tables = RotarySpec::interleaved(1_000_000.0, 32768, [24, 20, 20])?.tables(positions)?;
    let hidden = runner.forward::<2048, 6144, 16, 8, 2048, 1024>(
        input,
        &weights.layers,
        &weights.norm,
        cache,
        &tables,
    )?;
    let result = (|| -> Result<TalkerStep> {
        let last_hidden = runner.select(&hidden, &[last_row])?;
        let logits = runner.project(&last_hidden, &weights.codec_head, None)?;
        Ok(TalkerStep {
            last_hidden,
            logits,
        })
    })();
    if result.is_err() {
        cache.invalidate();
    }
    result
}

/// Embed a checked code-zero token using the Talker's vocabulary table.
pub fn code_zero_input(
    runner: &Runner,
    weights: &TalkerWeights<'_>,
    code: u32,
) -> Result<Batch<2048>> {
    runner.embed(&weights.codec_embedding, TokenIds::new(&[code])?)
}

/// Generate one complete set of remaining codec groups with a fresh predictor
/// cache. Host logits readback is intentional for this correctness baseline.
pub fn predictor_greedy_frame(
    runner: &Runner,
    weights: &PredictorWeights<'_>,
    talker_last: &Batch<2048>,
    code_zero_embedding: &Batch<2048>,
) -> Result<CodecFrame> {
    check_layers(weights.layers.len(), PREDICTOR_LAYERS)?;
    if talker_last.rows() != 1 || code_zero_embedding.rows() != 1 {
        return Err("predictor frame requires one Talker row and one code-zero row".into());
    }
    let mut input = runner.concat(talker_last, code_zero_embedding)?;
    let mut codec_rows = runner.select(code_zero_embedding, &[0])?;
    let mut cache =
        DecoderCache::<1024, 3072, 8>::new(runner.stream(), PREDICTOR_LAYERS, REMAINING_CODES + 1)?;
    let rotary = RotarySpec::standard(1_000_000.0, 65536)?;
    let mut remaining_codes = [0; REMAINING_CODES];

    for (step, code) in remaining_codes.iter_mut().enumerate() {
        let tables = rotary.tables(&predictor_positions(step)?)?;
        let projected = runner.project(&input, &weights.projection, Some(&weights.bias))?;
        let hidden = runner.forward::<1024, 3072, 16, 8, 2048, 1024>(
            &projected,
            &weights.layers,
            &weights.norm,
            &mut cache,
            &tables,
        )?;
        let last = runner.select(&hidden, &[(hidden.rows() - 1) as u32])?;
        let logits = runner.project(&last, &weights.heads[step], None)?;
        *code = greedy_token::<2048>(&logits.read(runner.stream())?)?;
        let embedded = runner.embed(&weights.embeddings[step], TokenIds::new(&[*code])?)?;
        codec_rows = runner.concat(&codec_rows, &embedded)?;
        input = embedded;
    }
    debug_assert_eq!(cache.len(), REMAINING_CODES + 1);
    let embedding_sum = runner.sum_rows(&codec_rows)?;
    Ok(CodecFrame {
        remaining_codes,
        embedding_sum,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cute_rs::Bf16;

    fn bf16(value: f32) -> Bf16 {
        Bf16::from_f32(value)
    }

    #[test]
    fn one_frame_predicts_fifteen_codes_from_sixteen_cached_rows() {
        let mut cached = 0;
        for step in 0..15 {
            let positions = predictor_positions(step).unwrap();
            assert_eq!(positions.len(), if step == 0 { 2 } else { 1 });
            for position in positions {
                assert_eq!(position, [cached; 3]);
                cached += 1;
            }
        }
        assert_eq!(cached, 16);
        assert!(predictor_positions(15).is_err());
        assert!(predictor_positions(usize::MAX).is_err());
    }

    #[test]
    fn greedy_selection_returns_exact_vocab_indices_and_resolves_ties() {
        assert_eq!(
            greedy_token::<4>(&[bf16(-2.0), bf16(3.0), bf16(3.0), bf16(0.5)]).unwrap(),
            1
        );
        assert_eq!(
            greedy_token::<4>(&[bf16(-2.0), bf16(-1.0), bf16(-3.0), bf16(0.5)]).unwrap(),
            3
        );
        assert_eq!(greedy_token::<1>(&[bf16(-5.0)]).unwrap(), 0);
        assert!(greedy_token::<4>(&[bf16(1.0); 3]).is_err());
        assert!(greedy_token::<4>(&[bf16(1.0); 5]).is_err());
        assert!(greedy_token::<0>(&[]).is_err());
    }

    #[test]
    fn greedy_selection_rejects_nonfinite_logits_even_when_not_winning() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(greedy_token::<3>(&[bf16(4.0), bf16(value), bf16(-1.0)]).is_err());
        }
    }

    #[test]
    fn full_model_layer_counts_are_enforced_before_work() {
        assert!(check_layers(28, 28).is_ok());
        assert!(check_layers(5, 5).is_ok());
        assert!(check_layers(1, 28).is_err());
        assert!(check_layers(0, 5).is_err());
        assert!(check_layers(28, 5).is_err());
    }
}
