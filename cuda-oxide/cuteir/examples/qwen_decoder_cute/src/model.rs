/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Typed synchronous decoder composition.
use crate::operator::Result;
use crate::{element_kernel, ffn_kernel, norm_kernel, operator, rotary::HostTables};
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaStream, DeviceBuffer};
use cute_rs::{
    Bf16,
    decoder::{
        BiasWeights, DecoderShape, DecoderWeights, EmbeddingWeights, FinalNorm, NormRole,
        NormWeights, TokenIds,
    },
    ffn::{ProjectionRole, ProjectionWeights, RowMajorShape},
};
use std::{marker::PhantomData, sync::Arc};

pub const EPSILON: f32 = 1e-6;

/// Owned dense BF16 rows. Width cannot be changed by a caller, nor can the
/// backing buffer be mutably borrowed while an operator reads it.
pub struct Batch<const W: usize> {
    buffer: DeviceBuffer<Bf16>,
    rows: usize,
}

impl<const W: usize> Batch<W> {
    pub fn from_host(stream: &CudaStream, rows: usize, values: &[Bf16]) -> Result<Self> {
        let shape = RowMajorShape::dense(rows, W)?;
        if values.len() != shape.required_len() {
            return Err("input must contain exactly rows * typed width values".into());
        }
        stream.context().bind_to_thread()?;
        Ok(Self {
            buffer: DeviceBuffer::from_host(stream, values)?,
            rows,
        })
    }
    pub(crate) fn allocate(stream: &CudaStream, rows: usize) -> Result<Self> {
        let shape = RowMajorShape::dense(rows, W)?;
        stream.context().bind_to_thread()?;
        Ok(Self {
            buffer: DeviceBuffer::from_host(
                stream,
                &vec![Bf16::from_bits(operator::GUARD); shape.required_len()],
            )?,
            rows,
        })
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn buffer(&self) -> &DeviceBuffer<Bf16> {
        &self.buffer
    }
    pub(crate) fn buffer_mut(&mut self) -> &mut DeviceBuffer<Bf16> {
        &mut self.buffer
    }
    pub fn read(&self, stream: &CudaStream) -> Result<Vec<Bf16>> {
        if !Arc::ptr_eq(self.buffer.context(), stream.context()) {
            return Err("read stream context mismatch".into());
        }
        stream.context().synchronize()?;
        Ok(self.buffer.to_host_vec(stream)?)
    }
}

/// A failed multi-layer append may leave different layer cursors. Such a cache
/// cannot be reused until reset; a successful call publishes a common prefix.
#[derive(Debug)]
struct CacheState {
    len: usize,
    capacity: usize,
    layers: usize,
    poisoned: bool,
}

impl CacheState {
    fn check(&self, layers: usize, tokens: usize) -> Result<usize> {
        if self.poisoned {
            return Err("failed decoder cache must be reset before reuse".into());
        }
        if layers == 0 || layers != self.layers {
            return Err("decoder layer count does not match owned caches".into());
        }
        let end = self
            .len
            .checked_add(tokens)
            .ok_or("cache cursor overflow")?;
        if end > self.capacity {
            return Err("decoder cache capacity exceeded".into());
        }
        Ok(end)
    }
}

/// One private K/V allocation per layer. Hidden and intermediate widths prevent
/// accidental Talker/predictor cache interchange even though both have eight KV heads.
pub struct DecoderCache<const H: usize, const I: usize, const K: usize> {
    state: CacheState,
    layers: Vec<operator::KvCache<K, 128>>,
    widths: PhantomData<([(); H], [(); I])>,
}

impl<const H: usize, const I: usize, const K: usize> DecoderCache<H, I, K> {
    pub fn new(stream: &CudaStream, layers: usize, capacity: usize) -> Result<Self> {
        if layers == 0 {
            return Err("decoder needs at least one layer".into());
        }
        DecoderShape::<H, I, K, K, 128>::new(0, H)?;
        Ok(Self {
            state: CacheState {
                len: 0,
                capacity,
                layers,
                poisoned: false,
            },
            layers: (0..layers)
                .map(|_| operator::KvCache::new(stream, capacity))
                .collect::<Result<_>>()?,
            widths: PhantomData,
        })
    }
    pub fn len(&self) -> usize {
        self.state.len
    }
    pub fn is_poisoned(&self) -> bool {
        self.state.poisoned
    }
    /// A caller that consumes a completed decoder step but fails before
    /// returning its prediction must not silently retry against the new prefix.
    pub(crate) fn invalidate(&mut self) {
        self.state.poisoned = true;
    }
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
        self.state.len = 0;
        self.state.poisoned = false;
    }
    pub fn snapshots(&self, stream: &CudaStream) -> Result<Vec<(Vec<Bf16>, Vec<Bf16>)>> {
        self.layers
            .iter()
            .map(|layer| layer.snapshot(stream))
            .collect()
    }
}

/// Optional synchronous readbacks for this numerical example, disabled in normal composition.
#[derive(Default)]
pub struct Trace {
    pub enabled: bool,
    pub stages: Vec<(&'static str, Vec<Bf16>)>,
}
impl Trace {
    fn record(
        &mut self,
        stream: &CudaStream,
        name: &'static str,
        buffer: &DeviceBuffer<Bf16>,
        length: usize,
    ) -> Result<()> {
        if self.enabled {
            let mut values = buffer.to_host_vec(stream)?;
            values.truncate(length);
            self.stages.push((name, values));
        }
        Ok(())
    }
}

fn grid(work: usize, per_block: usize) -> Result<Option<LaunchConfig>> {
    if work == 0 {
        return Ok(None);
    }
    let blocks = work.div_ceil(per_block);
    if blocks > i32::MAX as usize {
        return Err("grid exceeds CUDA grid-x range".into());
    }
    Ok(Some(LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }))
}

/// Reuses the tensor-core FFN projections, RMSNorm, RoPE and GQA kernels.
/// Safe methods establish input readiness and retain borrows through completion,
/// including launch errors. This deliberately synchronous baseline is B=1.
pub struct Runner {
    stream: Arc<CudaStream>,
    elements: element_kernel::kernels::LoadedModule,
    linear: ffn_kernel::kernels::LoadedModule,
    decode_linear: crate::decode_linear::Runner,
    norms: norm_kernel::kernels::LoadedModule,
    attention: operator::Runner,
}

impl Runner {
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        Ok(Self {
            elements: element_kernel::kernels::load(stream.context())?,
            // SAFETY: the module embeds the artifact compiled from the shared
            // FFN source; the decode runner prepares its contracted launches.
            linear: unsafe { ffn_kernel::kernels::load(stream.context())? },
            decode_linear: crate::decode_linear::Runner::new(stream.clone())?,
            norms: norm_kernel::kernels::load(stream.context())?,
            attention: operator::Runner::new(stream.clone())?,
            stream,
        })
    }
    fn ready(&self, buffers: &[&DeviceBuffer<Bf16>]) -> Result<()> {
        if buffers
            .iter()
            .any(|buffer| !Arc::ptr_eq(buffer.context(), self.stream.context()))
        {
            return Err("decoder buffers must share the runner context".into());
        }
        self.stream.context().synchronize()?;
        Ok(())
    }
    fn finish(&self, launch: Result<()>) -> Result<()> {
        let completed = self.stream.synchronize();
        launch?;
        completed?;
        Ok(())
    }
    pub fn embed<const V: usize, const W: usize>(
        &self,
        weights: &EmbeddingWeights<'_, V, W>,
        ids: TokenIds<'_, V>,
    ) -> Result<Batch<W>> {
        let shape = weights.output_shape(ids)?;
        if weights.shape().table().row_stride() != W {
            return Err("embedding kernel requires dense table rows".into());
        }
        self.ready(&[weights.buffer()])?;
        let mut output = Batch::allocate(&self.stream, ids.len())?;
        if let Some(config) = grid(shape.required_len(), 128)? {
            let ids_dev = DeviceBuffer::from_host(&self.stream, ids.as_slice())?;
            // SAFETY: TokenIds proves every index is in the checked table;
            // output ownership and one-thread-per-element writes are disjoint.
            let launch = unsafe {
                self.elements
                    .embedding_bf16(
                        &self.stream,
                        config,
                        weights.buffer(),
                        &ids_dev,
                        &mut output.buffer,
                        ids.len(),
                        W,
                    )
                    .map_err(Into::into)
            };
            self.finish(launch)?;
        }
        Ok(output)
    }
    pub fn select<const W: usize>(&self, input: &Batch<W>, ids: &[u32]) -> Result<Batch<W>> {
        if ids.iter().any(|&id| id as usize >= input.rows) {
            return Err("selected row exceeds batch length".into());
        }
        self.ready(&[input.buffer()])?;
        let mut output = Batch::allocate(&self.stream, ids.len())?;
        if let Some(config) = grid(output.buffer.len(), 128)? {
            let indices = DeviceBuffer::from_host(&self.stream, ids)?;
            // SAFETY: all row IDs were checked against the owned dense input.
            let launch = unsafe {
                self.elements
                    .embedding_bf16(
                        &self.stream,
                        config,
                        input.buffer(),
                        &indices,
                        &mut output.buffer,
                        ids.len(),
                        W,
                    )
                    .map_err(Into::into)
            };
            self.finish(launch)?;
        }
        Ok(output)
    }
    pub fn concat<const W: usize>(&self, left: &Batch<W>, right: &Batch<W>) -> Result<Batch<W>> {
        let rows = left
            .rows
            .checked_add(right.rows)
            .ok_or("concatenated row count overflow")?;
        self.ready(&[left.buffer(), right.buffer()])?;
        let mut output = Batch::allocate(&self.stream, rows)?;
        let total = output.buffer.len();
        if let Some(config) = grid(total, 128)? {
            // SAFETY: checked total extent and disjoint destination allocation.
            let launch = unsafe {
                self.elements
                    .concat_bf16(
                        &self.stream,
                        config,
                        left.buffer(),
                        right.buffer(),
                        &mut output.buffer,
                        left.buffer.len(),
                        total,
                    )
                    .map_err(Into::into)
            };
            self.finish(launch)?;
        }
        Ok(output)
    }
    /// Sum codec embedding rows in FP32, rounding only the final sum to BF16.
    pub fn sum_rows<const W: usize>(&self, input: &Batch<W>) -> Result<Batch<W>> {
        self.ready(&[input.buffer()])?;
        let mut output = Batch::allocate(&self.stream, 1)?;
        let config = grid(W, 128)?.ok_or("zero row width")?;
        // SAFETY: one thread owns each column and visits only valid input rows.
        let launch = unsafe {
            self.elements
                .sum_rows_bf16(
                    &self.stream,
                    config,
                    input.buffer(),
                    &mut output.buffer,
                    input.rows,
                    W,
                )
                .map_err(Into::into)
        };
        self.finish(launch)?;
        Ok(output)
    }
    pub fn project<Role: ProjectionRole, const IN: usize, const OUT: usize>(
        &self,
        input: &Batch<IN>,
        weight: &ProjectionWeights<'_, Role, IN, OUT>,
        bias: Option<&BiasWeights<'_, Role, OUT>>,
    ) -> Result<Batch<OUT>> {
        self.project_buffer(input.buffer(), input.rows(), weight, bias)
    }
    fn project_buffer<Role: ProjectionRole, const IN: usize, const OUT: usize>(
        &self,
        input: &DeviceBuffer<Bf16>,
        rows: usize,
        weight: &ProjectionWeights<'_, Role, IN, OUT>,
        bias: Option<&BiasWeights<'_, Role, OUT>>,
    ) -> Result<Batch<OUT>> {
        RowMajorShape::dense(rows, IN)?.validate_len(input.len())?;
        if matches!(rows, 1 | 2) {
            return self
                .decode_linear
                .project_buffer(input, rows, IN, weight, bias);
        }
        self.ready(&[input, weight.buffer()])?;
        if let Some(bias) = bias {
            self.ready(&[bias.buffer()])?;
        }
        let tiles = rows
            .div_ceil(16)
            .checked_mul(OUT.div_ceil(8))
            .ok_or("projection tile overflow")?;
        let config = grid(tiles, 4)?;
        let mut output = Batch::allocate(&self.stream, rows)?;
        if let Some(config) = config {
            // SAFETY: checked [out,in] weights and dense input/output extents;
            // the full-warp MMA map gives each output exactly one writer.
            let launch = unsafe {
                match bias {
                    Some(bias) => self.linear.linear_bias_bf16(
                        &self.stream,
                        config,
                        input,
                        weight.buffer(),
                        bias.buffer(),
                        &mut output.buffer,
                        rows,
                        IN,
                        OUT,
                        IN,
                        weight.shape().row_stride(),
                    ),
                    None => self.linear.linear_bf16(
                        &self.stream,
                        config,
                        input,
                        weight.buffer(),
                        &mut output.buffer,
                        rows,
                        IN,
                        OUT,
                        IN,
                        weight.shape().row_stride(),
                    ),
                }
                .map_err(Into::into)
            };
            self.finish(launch)?;
        }
        Ok(output)
    }
    pub fn norm<Role: NormRole, const W: usize>(
        &self,
        input: &Batch<W>,
        weight: &NormWeights<'_, Role, W>,
    ) -> Result<Batch<W>> {
        self.norm_rows(input, weight.buffer(), W)
    }
    /// Codec and language decoders retain their distinct checkpoint epsilon.
    pub fn norm_with_epsilon<Role: NormRole, const W: usize>(
        &self,
        input: &Batch<W>,
        weight: &NormWeights<'_, Role, W>,
        epsilon: f32,
    ) -> Result<Batch<W>> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err("normalization epsilon must be finite and positive".into());
        }
        self.norm_rows_epsilon(input, weight.buffer(), W, epsilon)
    }
    fn norm_rows<const W: usize>(
        &self,
        input: &Batch<W>,
        weight: &DeviceBuffer<Bf16>,
        norm_width: usize,
    ) -> Result<Batch<W>> {
        self.norm_rows_epsilon(input, weight, norm_width, EPSILON)
    }
    fn norm_rows_epsilon<const W: usize>(
        &self,
        input: &Batch<W>,
        weight: &DeviceBuffer<Bf16>,
        norm_width: usize,
        epsilon: f32,
    ) -> Result<Batch<W>> {
        if norm_width == 0 || !W.is_multiple_of(norm_width) || weight.len() < norm_width {
            return Err("normalization width does not divide the typed row".into());
        }
        let rows = input
            .rows
            .checked_mul(W / norm_width)
            .ok_or("norm row overflow")?;
        self.ready(&[input.buffer(), weight])?;
        let mut output = Batch::allocate(&self.stream, input.rows)?;
        if let Some(config) = grid(rows, 4)? {
            // SAFETY: full warps own distinct contiguous norm_width rows.
            let launch = unsafe {
                self.norms
                    .rmsnorm_bf16(
                        &self.stream,
                        config,
                        input.buffer(),
                        weight,
                        &mut output.buffer,
                        rows,
                        norm_width,
                        epsilon,
                    )
                    .map_err(Into::into)
            };
            self.finish(launch)?;
        }
        Ok(output)
    }
    pub fn add<const W: usize>(&self, left: &Batch<W>, right: &Batch<W>) -> Result<Batch<W>> {
        if left.rows != right.rows {
            return Err("residual row count mismatch".into());
        }
        self.ready(&[left.buffer(), right.buffer()])?;
        let mut output = Batch::allocate(&self.stream, left.rows)?;
        if let Some(config) = grid(left.buffer.len(), 128)? {
            // SAFETY: exact matching extents, distinct output, disjoint threads.
            let launch = unsafe {
                self.elements
                    .add_bf16(
                        &self.stream,
                        config,
                        left.buffer(),
                        right.buffer(),
                        &mut output.buffer,
                        left.buffer.len(),
                    )
                    .map_err(Into::into)
            };
            self.finish(launch)?;
        }
        Ok(output)
    }
    pub fn silu<const W: usize>(&self, input: &Batch<W>) -> Result<Batch<W>> {
        self.ready(&[input.buffer()])?;
        let mut output = Batch::allocate(&self.stream, input.rows)?;
        if let Some(config) = grid(input.buffer.len(), 128)? {
            // SAFETY: matching extents and one output element per thread.
            let launch = unsafe {
                self.elements
                    .silu_bf16(
                        &self.stream,
                        config,
                        input.buffer(),
                        &mut output.buffer,
                        input.buffer.len(),
                    )
                    .map_err(Into::into)
            };
            self.finish(launch)?;
        }
        Ok(output)
    }
    /// Apply the already validated BF16 SiLU/product rounding sequence.
    pub fn swiglu<const W: usize>(&self, gate: &Batch<W>, up: &Batch<W>) -> Result<Batch<W>> {
        if gate.rows() != up.rows() {
            return Err("SwiGLU row counts differ".into());
        }
        self.ready(&[gate.buffer(), up.buffer()])?;
        let mut silu = Batch::<W>::allocate(&self.stream, gate.rows())?;
        let mut product = Batch::<W>::allocate(&self.stream, gate.rows())?;
        if let Some(config) = grid(gate.buffer.len().div_ceil(8), 128)? {
            // SAFETY: typed equal-size buffers; each lane owns disjoint aligned
            // eight-element tiles, with the validated kernel's scalar tail.
            let launch = unsafe {
                self.linear
                    .activate_bf16(
                        &self.stream,
                        config,
                        gate.buffer(),
                        up.buffer(),
                        &mut silu.buffer,
                        &mut product.buffer,
                        gate.buffer.len(),
                    )
                    .map_err(Into::into)
            };
            self.finish(launch)?;
        }
        Ok(product)
    }
    fn validate_layer<const H: usize, const I: usize, const QW: usize, const KW: usize>(
        &self,
        weight: &DecoderWeights<'_, H, I, QW, KW, 128>,
    ) -> Result<()> {
        let a = weight.attention();
        let f = weight.ffn();
        self.ready(&[
            weight.input_norm().buffer(),
            weight.post_attention_norm().buffer(),
            a.query().buffer(),
            a.key().buffer(),
            a.value().buffer(),
            a.output().buffer(),
            a.query_norm().buffer(),
            a.key_norm().buffer(),
            f.gate().buffer(),
            f.up().buffer(),
            f.down().buffer(),
        ])
    }
    #[allow(clippy::too_many_arguments)]
    /// Single-layer numerical probe. An error after append may have advanced
    /// this raw cache; discard/reset it on error. Model calls use `forward`,
    /// which guards the complete stack with poison-on-failure state.
    pub(crate) fn layer<
        const H: usize,
        const I: usize,
        const Q: usize,
        const K: usize,
        const QW: usize,
        const KW: usize,
    >(
        &self,
        input: &Batch<H>,
        weight: &DecoderWeights<'_, H, I, QW, KW, 128>,
        cache: &mut operator::KvCache<K, 128>,
        tables: &HostTables<128>,
        trace: &mut Trace,
    ) -> Result<Batch<H>> {
        let shape = DecoderShape::<H, I, Q, K, 128>::new(input.rows, H)?;
        shape.validate_weights(weight)?;
        self.validate_layer(weight)?;
        if tables.tokens() != input.rows || input.rows > cache.capacity() - cache.len() {
            return Err("decoder positions or cache capacity mismatch".into());
        }
        let start = cache.len();
        let a = weight.attention();
        let normalized = self.norm(input, weight.input_norm())?;
        trace.record(
            &self.stream,
            "normalized",
            normalized.buffer(),
            input.rows * H,
        )?;
        let q = self.project(&normalized, a.query(), None)?;
        let k = self.project(&normalized, a.key(), None)?;
        let v = self.project(&normalized, a.value(), None)?;
        trace.record(&self.stream, "q", q.buffer(), input.rows * QW)?;
        trace.record(&self.stream, "k", k.buffer(), input.rows * KW)?;
        trace.record(&self.stream, "v", v.buffer(), input.rows * KW)?;
        let qn = self.norm_rows(&q, a.query_norm().buffer(), 128)?;
        let kn = self.norm_rows(&k, a.key_norm().buffer(), 128)?;
        trace.record(&self.stream, "q_normed", qn.buffer(), input.rows * QW)?;
        trace.record(&self.stream, "k_normed", kn.buffer(), input.rows * KW)?;
        let query = self.attention.append(
            cache,
            operator::AttentionInputs::<Q, K, 128>::new(
                qn.buffer(),
                kn.buffer(),
                v.buffer(),
                input.rows,
            )?,
            tables,
        )?;
        trace.record(
            &self.stream,
            "rotated_q",
            query.buffer(),
            query.tokens() * QW,
        )?;
        let attended = self.attention.attend(&query, cache, Some(start))?;
        trace.record(
            &self.stream,
            "attention",
            attended.buffer(),
            input.rows * QW,
        )?;
        let projected =
            self.project_buffer(attended.buffer(), attended.tokens(), a.output(), None)?;
        trace.record(
            &self.stream,
            "projected",
            projected.buffer(),
            input.rows * H,
        )?;
        let residual = self.add(input, &projected)?;
        trace.record(&self.stream, "residual", residual.buffer(), input.rows * H)?;
        let post = self.norm(&residual, weight.post_attention_norm())?;
        trace.record(&self.stream, "post_normed", post.buffer(), input.rows * H)?;
        let gate = self.project(&post, weight.ffn().gate(), None)?;
        let up = self.project(&post, weight.ffn().up(), None)?;
        trace.record(&self.stream, "gate", gate.buffer(), input.rows * I)?;
        trace.record(&self.stream, "up", up.buffer(), input.rows * I)?;
        let mut silu = Batch::<I>::allocate(&self.stream, input.rows)?;
        let mut product = Batch::<I>::allocate(&self.stream, input.rows)?;
        if let Some(config) = grid(gate.buffer.len().div_ceil(8), 128)? {
            // SAFETY: dense typed FFN buffers, aligned vector tiles plus tails;
            // each thread owns a distinct eight-element output tile.
            let launch = unsafe {
                self.linear
                    .activate_bf16(
                        &self.stream,
                        config,
                        gate.buffer(),
                        up.buffer(),
                        &mut silu.buffer,
                        &mut product.buffer,
                        gate.buffer.len(),
                    )
                    .map_err(Into::into)
            };
            self.finish(launch)?;
        }
        trace.record(&self.stream, "silu", silu.buffer(), input.rows * I)?;
        trace.record(&self.stream, "product", product.buffer(), input.rows * I)?;
        let down = self.project(&product, weight.ffn().down(), None)?;
        trace.record(&self.stream, "down", down.buffer(), input.rows * H)?;
        let output = self.add(&residual, &down)?;
        trace.record(&self.stream, "output", output.buffer(), input.rows * H)?;
        Ok(output)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn forward<
        const H: usize,
        const I: usize,
        const Q: usize,
        const K: usize,
        const QW: usize,
        const KW: usize,
    >(
        &self,
        input: &Batch<H>,
        weights: &[DecoderWeights<'_, H, I, QW, KW, 128>],
        final_norm: &NormWeights<'_, FinalNorm, H>,
        cache: &mut DecoderCache<H, I, K>,
        tables: &HostTables<128>,
    ) -> Result<Batch<H>> {
        let end = cache.state.check(weights.len(), input.rows)?;
        let shape = DecoderShape::<H, I, Q, K, 128>::new(input.rows, H)?;
        if tables.tokens() != input.rows {
            return Err("rotary rows do not match decoder input".into());
        }
        self.ready(&[input.buffer(), final_norm.buffer()])?;
        for weight in weights {
            shape.validate_weights(weight)?;
            self.validate_layer(weight)?;
        }
        if cache
            .layers
            .iter()
            .any(|layer| layer.len() != cache.state.len)
        {
            return Err("inconsistent layer cache prefix".into());
        }
        // Once a layer has appended, a later failure must invalidate the whole
        // sequence. Keep the old published length until every layer completes.
        cache.state.poisoned = true;
        let mut trace = Trace::default();
        let mut hidden = self.layer::<H, I, Q, K, QW, KW>(
            input,
            &weights[0],
            &mut cache.layers[0],
            tables,
            &mut trace,
        )?;
        for (weight, layer) in weights[1..].iter().zip(&mut cache.layers[1..]) {
            hidden =
                self.layer::<H, I, Q, K, QW, KW>(&hidden, weight, layer, tables, &mut trace)?;
        }
        let output = self.norm(&hidden, final_norm)?;
        cache.state.len = end;
        cache.state.poisoned = false;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_step_rejects_inconsistent_layers_capacity_and_poison() {
        let mut state = CacheState {
            len: 3,
            capacity: 5,
            layers: 2,
            poisoned: false,
        };
        assert_eq!(state.check(2, 2).unwrap(), 5);
        assert_eq!(state.check(2, 0).unwrap(), 3);
        assert!(state.check(1, 1).is_err());
        assert!(state.check(2, 3).is_err());
        assert!(state.check(2, usize::MAX).is_err());
        assert_eq!(state.len, 3);
        state.poisoned = true;
        assert!(state.check(2, 0).is_err());
        assert!(state.check(2, 1).is_err());
    }
}
