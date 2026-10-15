/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Checked host token selection, independent of GPU model computation.
//!
//! Sampling follows the pinned Transformers 4.57.3 warper order: temperature,
//! top-k (including boundary ties), top-p (retaining its crossing token), then
//! categorical selection. Logit transforms and softmax use FP32; the final CDF
//! accumulates the resulting probabilities in FP64. This is not a promise of
//! bit-identical Torch reductions or RNG streams.
//!
//! SplitMix64 v1 uses wrapping integer operations and the upper 53 output bits
//! for a uniform value in [0,1). It is reproducible, not cryptographic. Request
//! code owns seed/domain separation; greedy selection consumes no random draws.

use crate::operator::Result;
use cute_rs::Bf16;

pub const RNG_VERSION: &str = "splitmix64-v1/top53-uniform";

#[derive(Clone, Copy, Debug)]
pub struct SamplingConfig<const VOCAB: usize> {
    mode: Mode,
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Greedy,
    Sample {
        temperature: f32,
        top_k: usize,
        top_p: f32,
    },
}

impl<const VOCAB: usize> SamplingConfig<VOCAB> {
    pub fn greedy() -> Result<Self> {
        if VOCAB == 0 || VOCAB > u32::MAX as usize {
            return Err("sampling vocabulary must be nonempty and u32-indexed".into());
        }
        Ok(Self { mode: Mode::Greedy })
    }

    pub fn sample(temperature: f32, top_k: usize, top_p: f32) -> Result<Self> {
        Self::greedy()?;
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err("sampling temperature must be finite and positive".into());
        }
        if top_k > VOCAB {
            return Err("top_k must be zero (disabled) or within the vocabulary".into());
        }
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            return Err("top_p must be finite and within (0,1]".into());
        }
        Ok(Self {
            mode: Mode::Sample {
                temperature,
                top_k,
                top_p,
            },
        })
    }
}

pub struct Sampler<const VOCAB: usize> {
    config: SamplingConfig<VOCAB>,
    state: u64,
}

impl<const VOCAB: usize> Sampler<VOCAB> {
    pub fn new(config: SamplingConfig<VOCAB>, seed: u64) -> Self {
        Self {
            config,
            state: seed,
        }
    }

    /// Scores may contain -infinity only for caller-supplied suppression masks.
    /// NaN, +infinity, wrong width and an entirely suppressed vocabulary fail.
    pub fn select(&mut self, scores: &[f32]) -> Result<u32> {
        validate::<VOCAB>(scores)?;
        if let Mode::Greedy = self.config.mode {
            let mut best = 0;
            for index in 1..VOCAB {
                if scores[index] > scores[best] {
                    best = index;
                }
            }
            return Ok(best as u32);
        }
        let probabilities = probabilities(scores, self.config)?;
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut bits = self.state;
        bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        bits ^= bits >> 31;
        let uniform = ((bits >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0);
        select_uniform(&probabilities, uniform)
    }

    /// Raw GPU logits cannot contain masks or nonfinite values.
    pub fn select_bf16(&mut self, scores: &[Bf16]) -> Result<u32> {
        let scores: Vec<_> = scores.iter().map(|value| value.to_f32()).collect();
        if scores.iter().any(|value| !value.is_finite()) {
            return Err("raw BF16 logits must all be finite".into());
        }
        self.select(&scores)
    }
}

fn validate<const V: usize>(scores: &[f32]) -> Result<()> {
    if scores.len() != V
        || scores
            .iter()
            .any(|&score| score.is_nan() || score == f32::INFINITY)
    {
        return Err(
            "logits require the typed vocabulary width and finite scores or -infinity masks".into(),
        );
    }
    if !scores.iter().any(|score| score.is_finite()) {
        return Err("all logits are suppressed".into());
    }
    Ok(())
}

fn softmax(scores: &[f32]) -> Result<Vec<f32>> {
    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities: Vec<_> = scores.iter().map(|score| (score - maximum).exp()).collect();
    let total: f32 = probabilities.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err("sampling softmax has no finite positive mass".into());
    }
    for value in &mut probabilities {
        *value /= total;
    }
    Ok(probabilities)
}

fn probabilities<const V: usize>(scores: &[f32], config: SamplingConfig<V>) -> Result<Vec<f32>> {
    validate::<V>(scores)?;
    let Mode::Sample {
        temperature,
        top_k,
        top_p,
    } = config.mode
    else {
        return Err("probability construction requires a sampling policy".into());
    };
    let mut scores = scores.to_vec();
    for score in &mut scores {
        if score.is_finite() {
            *score /= temperature;
            if !score.is_finite() {
                return Err("temperature scaling overflowed FP32".into());
            }
        }
    }
    if top_k > 0 && top_k < V {
        let mut ranked = scores.clone();
        ranked.sort_unstable_by(|left, right| right.total_cmp(left));
        let threshold = ranked[top_k - 1];
        for score in &mut scores {
            if *score < threshold {
                *score = f32::NEG_INFINITY;
            }
        }
    }
    if top_p < 1.0 {
        let mut indices: Vec<_> = (0..V).collect();
        // Ascending scores match Transformers' discarded lower tail. For exact
        // ties use index order explicitly; backend sort ties are not portable.
        indices.sort_by(|&left, &right| {
            scores[left]
                .total_cmp(&scores[right])
                .then(left.cmp(&right))
        });
        let ranked: Vec<_> = indices.iter().map(|&index| scores[index]).collect();
        let mass = softmax(&ranked)?;
        let mut cumulative = 0.0_f32;
        for (rank, (&index, probability)) in indices.iter().zip(mass).enumerate() {
            cumulative += probability;
            if cumulative <= 1.0 - top_p && rank + 1 < V {
                scores[index] = f32::NEG_INFINITY;
            }
        }
    }
    softmax(&scores)
}

fn select_uniform(probabilities: &[f32], uniform: f64) -> Result<u32> {
    if !uniform.is_finite()
        || !(0.0..1.0).contains(&uniform)
        || probabilities.is_empty()
        || probabilities.len() > u32::MAX as usize
        || probabilities
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(
            "categorical selection requires probabilities and a uniform draw in [0,1)".into(),
        );
    }
    let total: f64 = probabilities.iter().map(|&value| f64::from(value)).sum();
    if total <= 0.0 || !total.is_finite() {
        return Err("categorical distribution has no positive mass".into());
    }
    let threshold = uniform * total;
    let mut cumulative = 0.0;
    let mut last_positive = None;
    for (index, &probability) in probabilities.iter().enumerate() {
        cumulative += f64::from(probability);
        if probability > 0.0 {
            last_positive = Some(index as u32);
        }
        if threshold < cumulative {
            return Ok(index as u32);
        }
    }
    // Only a final FP64 rounding boundary can reach this branch.
    last_positive.ok_or_else(|| "categorical distribution has no selectable token".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rejects_invalid_vocabulary_and_sampling_parameters() {
        assert!(SamplingConfig::<0>::greedy().is_err());
        for temperature in [0., -1., f32::NAN, f32::INFINITY] {
            assert!(SamplingConfig::<4>::sample(temperature, 0, 1.).is_err());
        }
        for top_p in [0., -1., 1.01, f32::NAN, f32::INFINITY] {
            assert!(SamplingConfig::<4>::sample(1., 0, top_p).is_err());
        }
        assert!(SamplingConfig::<4>::sample(1., 5, 1.).is_err());
        assert!(SamplingConfig::<4>::sample(1., 0, 1.).is_ok());
    }

    #[test]
    fn greedy_keeps_lowest_tie_and_checks_raw_bf16_before_masks() {
        let mut sampler = Sampler::new(SamplingConfig::<4>::greedy().unwrap(), 1);
        assert_eq!(sampler.select(&[1., 3., 3., f32::NEG_INFINITY]).unwrap(), 1);
        assert_eq!(
            sampler
                .select_bf16(&[
                    Bf16::from_bits(0x3f80),
                    Bf16::from_bits(0x4040),
                    Bf16::from_bits(0x4040),
                    Bf16::from_bits(0)
                ])
                .unwrap(),
            1
        );
        for value in [f32::NAN, f32::INFINITY] {
            assert!(sampler.select(&[1., 2., 3., value]).is_err());
        }
        assert!(sampler.select(&[f32::NEG_INFINITY; 4]).is_err());
        assert!(sampler.select(&[1., 2.]).is_err());
        assert!(sampler.select_bf16(&[Bf16::from_bits(0xff80); 4]).is_err());
    }

    fn reference_case<const V: usize>(
        scores: [f32; V],
        temperature: f32,
        top_k: usize,
        top_p: f32,
        expected: [f32; V],
        selections: [u32; 5],
    ) {
        let actual = probabilities(
            &scores,
            SamplingConfig::<V>::sample(temperature, top_k, top_p).unwrap(),
        )
        .unwrap();
        assert_eq!(actual.len(), V);
        for (index, (&left, &right)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (left - right).abs() <= 2e-7,
                "probability {index}: {left} != {right}"
            );
        }
        for (uniform, expected) in [0., 0.2, 0.5, 0.8, 0.999999].into_iter().zip(selections) {
            assert_eq!(select_uniform(&actual, uniform).unwrap(), expected);
        }
    }

    #[test]
    fn matches_pinned_transformers_fp32_warpers_and_shared_uniform_reference() {
        // Independently exported with Transformers 4.57.3 Temperature/TopK/TopP
        // warpers, Torch 2.14.1 FP32 softmax and FP64 CDF, not this implementation.
        reference_case(
            [0., 1., 2., 3.],
            0.5,
            0,
            1.,
            [0.0021440089, 0.015842201, 0.117_058_9, 0.864_954_8],
            [0, 3, 3, 3, 3],
        );
        reference_case(
            [3., 2., 2., 0.],
            1.,
            2,
            1.,
            [0.5761169, 0.21194157, 0.21194157, 0.],
            [0, 0, 0, 2, 2],
        );
        reference_case(
            [3., 2., 1., 0.],
            1.,
            0,
            0.7,
            [0.7310586, 0.26894143, 0., 0.],
            [0, 0, 0, 1, 1],
        );
        reference_case(
            [1., -2., 3., 0., 2.],
            0.9,
            3,
            0.8,
            [0., 0., 0.7523362, 0., 0.24766378],
            [2, 2, 2, 4, 4],
        );
    }

    #[test]
    fn shared_draws_handle_zero_mass_boundaries_and_reject_invalid_inputs() {
        assert_eq!(select_uniform(&[0., 0.25, 0., 0.75], 0.).unwrap(), 1);
        assert_eq!(select_uniform(&[0., 0.25, 0., 0.75], 0.25).unwrap(), 3);
        for uniform in [-1., 1., f64::NAN, f64::INFINITY] {
            assert!(select_uniform(&[0.5, 0.5], uniform).is_err());
        }
        for probabilities in [
            vec![],
            vec![0., 0.],
            vec![1., -1.],
            vec![f32::NAN],
            vec![f32::INFINITY],
        ] {
            assert!(select_uniform(&probabilities, 0.5).is_err());
        }
    }

    #[test]
    fn seeded_sampling_repeats_and_separates_seeds_with_bounded_tokens() {
        let config = SamplingConfig::<4>::sample(1., 0, 1.).unwrap();
        let sequence = |seed| {
            let mut sampler = Sampler::new(config, seed);
            (0..64)
                .map(|_| sampler.select(&[0.; 4]).unwrap())
                .collect::<Vec<_>>()
        };
        let first = sequence(42);
        assert_eq!(first, sequence(42));
        assert_ne!(first, sequence(43));
        assert!(first.iter().all(|&token| token < 4));
        assert!(first.contains(&3));
        let mut one = Sampler::new(SamplingConfig::<4>::sample(0.9, 1, 1.).unwrap(), 99);
        for _ in 0..16 {
            assert_eq!(one.select(&[-10., 4., 2., 1.]).unwrap(), 1);
        }
    }

    #[test]
    fn sampling_preserves_exact_bf16_widening_and_rejects_scaling_overflow() {
        let config = SamplingConfig::<4>::sample(0.9, 3, 0.8).unwrap();
        let mut bf16 = Sampler::new(config, 13);
        let mut fp32 = Sampler::new(config, 13);
        let values = [0x3f80, 0x4000, 0x4040, 0x4080].map(Bf16::from_bits);
        for _ in 0..32 {
            assert_eq!(
                bf16.select_bf16(&values).unwrap(),
                fp32.select(&[1., 2., 3., 4.]).unwrap()
            );
        }
        let mut extreme = Sampler::new(SamplingConfig::<4>::sample(0.5, 0, 1.).unwrap(), 0);
        assert!(extreme.select(&[f32::MAX, 1., 2., 3.]).is_err());
    }
}
