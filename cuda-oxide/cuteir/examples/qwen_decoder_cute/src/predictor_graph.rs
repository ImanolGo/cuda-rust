/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Exclusive, synchronous predictor execution over persistent typed storage.
//!
//! The captured region owns every referenced allocation, module and checkpoint.
//! Only completed output copies escape. Caller inputs are copied after context
//! readiness; none of the public runner or buffer-pool completion rules change.

use crate::checkpoint::QwenWeights;
use crate::generation::{CodecFrame, PREDICTOR_LAYERS, PredictorWeights, REMAINING_CODES};
use crate::model::{Batch, EPSILON};
use crate::operator::{GUARD_LEN, Result};
use crate::{
    attention_kernel, element_kernel, ffn_kernel, kernel, norm_kernel, predictor_graph_kernel,
    rotary::RotarySpec,
};
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig1D, PreparedLaunch, sys};
use cute_rs::{
    Bf16,
    decoder::DecoderShape,
    ffn::{ProjectionRole, ProjectionWeights},
};
use std::sync::Arc;

// Device guard/index helpers encode this fixed model schedule. Fail compilation
// if shared geometry changes instead of silently weakening their bounds proof.
const _: () = assert!(GUARD_LEN == 8 && attention_kernel::STATE_SIZE == 130);
const _: () = assert!(PREDICTOR_LAYERS == 5 && REMAINING_CODES == 15);

type ProjectionLaunch = PreparedLaunch<ffn_kernel::kernels::__decode_linear_bf16_CudaKernel>;
type BiasedLaunch = PreparedLaunch<ffn_kernel::kernels::__decode_linear_bias_bf16_CudaKernel>;
type PartialLaunch = PreparedLaunch<attention_kernel::kernels::__partial_CudaKernel>;
type MergeLaunch = PreparedLaunch<attention_kernel::kernels::__merge_CudaKernel>;

fn driver(result: sys::CUresult) -> Result<()> {
    if result == sys::cudaError_enum_CUDA_SUCCESS {
        Ok(())
    } else {
        Err(DriverError(result).into())
    }
}

fn grid(work: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (work.div_ceil(128) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn one_warp() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Completion is installed before the first enqueue and precedes owner drops.
struct Completion(Option<Arc<CudaStream>>);
impl Completion {
    fn new(stream: &Arc<CudaStream>) -> Self {
        Self(Some(stream.clone()))
    }
    fn finish(mut self) -> Result<()> {
        Ok(self.0.take().expect("one completion").synchronize()?)
    }
}
impl Drop for Completion {
    fn drop(&mut self) {
        if let Some(stream) = self.0.take() {
            let _ = stream.synchronize();
        }
    }
}

struct Graph {
    source: sys::CUgraph,
    executable: sys::CUgraphExec,
    stream: Arc<CudaStream>,
}
impl Drop for Graph {
    fn drop(&mut self) {
        let _ = self.stream.context().bind_to_thread();
        // The plan drains its stream before dropping this owner. Construction
        // failure never launched it. Destroy executable before source/resources.
        unsafe {
            if !self.executable.is_null() {
                let _ = sys::cuGraphExecDestroy(self.executable);
            }
            if !self.source.is_null() {
                let _ = sys::cuGraphDestroy(self.source);
            }
        }
    }
}

struct Capture {
    stream: Arc<CudaStream>,
    active: bool,
}
impl Capture {
    fn begin(stream: &Arc<CudaStream>) -> Result<Self> {
        stream.context().bind_to_thread()?;
        // SAFETY: private nondefault stream, current context, no concurrent use.
        unsafe {
            driver(sys::cuStreamBeginCapture_v2(
                stream.cu_stream(),
                sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            ))?;
        }
        Ok(Self {
            stream: stream.clone(),
            active: true,
        })
    }
    fn end(mut self) -> Result<Graph> {
        self.stream.context().bind_to_thread()?;
        let mut source = std::ptr::null_mut();
        // End must also run for invalidated captures. Never synchronize while
        // capture is active, even on submission error or unwinding.
        let result = unsafe { sys::cuStreamEndCapture(self.stream.cu_stream(), &mut source) };
        self.active = false;
        let mut graph = Graph {
            source,
            executable: std::ptr::null_mut(),
            stream: self.stream.clone(),
        };
        driver(result)?;
        unsafe {
            driver(sys::cuGraphInstantiateWithFlags(
                &mut graph.executable,
                graph.source,
                0,
            ))?;
        }
        Ok(graph)
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        if self.active {
            // Cleanup also runs during unwinding; a panic hook may have changed
            // the current context since this stream's capture began.
            let _ = self.stream.context().bind_to_thread();
            let mut source = std::ptr::null_mut();
            unsafe {
                let _ = sys::cuStreamEndCapture(self.stream.cu_stream(), &mut source);
                if !source.is_null() {
                    let _ = sys::cuGraphDestroy(source);
                }
            }
        }
    }
}

/// Guarded attention rows retain their logical width instead of carrying raw
/// indices into a universal buffer collection.
struct Guarded<const W: usize> {
    buffer: DeviceBuffer<Bf16>,
}
impl<const W: usize> Guarded<W> {
    fn new(stream: &CudaStream, rows: usize) -> Result<Self> {
        Ok(Self {
            buffer: crate::operator::allocate(
                stream,
                rows.checked_mul(W).ok_or("guarded row overflow")?,
            )?,
        })
    }
}
struct LayerCache {
    key: Guarded<1024>,
    value: Guarded<1024>,
}
struct Tables {
    cos: DeviceBuffer<Bf16>,
    sin: DeviceBuffer<Bf16>,
}

/// Each live attention geometry owns its own guard boundary. A shared max-size
/// scratch would let the next layer overwrite evidence beyond the live extent.
struct Scratch {
    rows: usize,
    hidden: Batch<1024>,
    normalized: Batch<1024>,
    query: Batch<2048>,
    key: Batch<1024>,
    value: Batch<1024>,
    query_norm: Batch<2048>,
    key_norm: Batch<1024>,
    rotated_query: Guarded<2048>,
    rotated_key: Guarded<1024>,
    attended: Guarded<2048>,
    partials: DeviceBuffer<f32>,
    projected: Batch<1024>,
    residual: Batch<1024>,
    post: Batch<1024>,
    gate: Batch<3072>,
    up: Batch<3072>,
    silu: Batch<3072>,
    product: Batch<3072>,
    down: Batch<1024>,
    final_norm: Batch<1024>,
    last: Batch<1024>,
    logits: Batch<2048>,
    partial_launch: PartialLaunch,
    merge_launch: MergeLaunch,
}
impl Scratch {
    fn new(
        stream: &CudaStream,
        module: &attention_kernel::kernels::LoadedModule,
        rows: usize,
    ) -> Result<Self> {
        if !matches!(rows, 1 | 2) {
            return Err("predictor scratch requires one or two rows".into());
        }
        Ok(Self {
            rows,
            hidden: Batch::allocate(stream, rows)?,
            normalized: Batch::allocate(stream, rows)?,
            query: Batch::allocate(stream, rows)?,
            key: Batch::allocate(stream, rows)?,
            value: Batch::allocate(stream, rows)?,
            query_norm: Batch::allocate(stream, rows)?,
            key_norm: Batch::allocate(stream, rows)?,
            rotated_query: Guarded::new(stream, rows)?,
            rotated_key: Guarded::new(stream, rows)?,
            attended: Guarded::new(stream, rows)?,
            partials: DeviceBuffer::from_host(
                stream,
                &vec![
                    f32::from_bits(0x7fc1_0001);
                    rows * 16 * attention_kernel::STATE_SIZE + GUARD_LEN
                ],
            )?,
            projected: Batch::allocate(stream, rows)?,
            residual: Batch::allocate(stream, rows)?,
            post: Batch::allocate(stream, rows)?,
            gate: Batch::allocate(stream, rows)?,
            up: Batch::allocate(stream, rows)?,
            silu: Batch::allocate(stream, rows)?,
            product: Batch::allocate(stream, rows)?,
            down: Batch::allocate(stream, rows)?,
            final_norm: Batch::allocate(stream, rows)?,
            last: Batch::allocate(stream, 1)?,
            logits: Batch::allocate(stream, 1)?,
            partial_launch: module.prepare_partial(LaunchConfig1D::new(
                (rows * 16) as u32,
                128,
                attention_kernel::PARTIAL_SHARED_BYTES,
            ))?,
            merge_launch: module.prepare_merge(LaunchConfig1D::new((rows * 16) as u32, 32, 0))?,
        })
    }
}

struct Modules {
    linear: ffn_kernel::kernels::LoadedModule,
    norms: norm_kernel::kernels::LoadedModule,
    elements: element_kernel::kernels::LoadedModule,
    rotary: kernel::kernels::LoadedModule,
    attention: attention_kernel::kernels::LoadedModule,
    schedule: predictor_graph_kernel::kernels::LoadedModule,
    projection_1024: ProjectionLaunch,
    projection_2048: ProjectionLaunch,
    projection_3072: ProjectionLaunch,
    biased_1024: BiasedLaunch,
}
impl Modules {
    fn new(stream: &CudaStream) -> Result<Self> {
        // SAFETY: modules embed this binary's matching kernel definitions.
        let linear = unsafe { ffn_kernel::kernels::load(stream.context())? };
        let attention = unsafe { attention_kernel::kernels::load(stream.context())? };
        Ok(Self {
            projection_1024: linear.prepare_decode_linear_bf16(LaunchConfig1D::new(128, 32, 0))?,
            projection_2048: linear.prepare_decode_linear_bf16(LaunchConfig1D::new(256, 32, 0))?,
            projection_3072: linear.prepare_decode_linear_bf16(LaunchConfig1D::new(384, 32, 0))?,
            biased_1024: linear.prepare_decode_linear_bias_bf16(LaunchConfig1D::new(128, 32, 0))?,
            linear,
            attention,
            norms: norm_kernel::kernels::load(stream.context())?,
            elements: element_kernel::kernels::load(stream.context())?,
            rotary: kernel::kernels::load(stream.context())?,
            schedule: predictor_graph_kernel::kernels::load(stream.context())?,
        })
    }

    /// Caller has validated typed weight geometry/context, private scratch
    /// extent and readiness. All owners outlive unconditional completion.
    unsafe fn project<Role: ProjectionRole, const IN: usize, const OUT: usize>(
        &self,
        stream: &CudaStream,
        input: &DeviceBuffer<Bf16>,
        rows: usize,
        weight: &ProjectionWeights<'_, Role, IN, OUT>,
        output: &mut Batch<OUT>,
    ) -> Result<()> {
        let launch = match OUT {
            1024 => &self.projection_1024,
            2048 => &self.projection_2048,
            3072 => &self.projection_3072,
            _ => return Err("unsupported predictor projection width".into()),
        };
        self.linear.decode_linear_bf16(
            stream,
            launch,
            input,
            weight.buffer(),
            output.buffer_mut(),
            rows,
            IN,
            OUT,
            IN,
            weight.shape().row_stride(),
        )?;
        Ok(())
    }
}

/// Owns the complete fifteen-step program. No pending work or private scratch
/// borrow escapes `run`; exclusive access prevents overlapping replays.
pub(crate) struct PredictorPlan {
    graph: Option<Graph>,
    stream: Arc<CudaStream>,
    weights: Arc<QwenWeights>,
    modules: Modules,
    prefill: Scratch,
    decode: Scratch,
    caches: Vec<LayerCache>,
    tables: Vec<Tables>,
    first_input: Batch<2048>,
    feedback: Batch<2048>,
    codec_rows: Batch<2048>,
    embedding_sum: Batch<2048>,
    codes: DeviceBuffer<u32>,
    status: DeviceBuffer<u32>,
    poisoned: bool,
}

impl Drop for PredictorPlan {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
        // Explicit drop order matters: graph nodes contain pointers into every
        // remaining field, including checkpoint buffers and loaded modules.
        drop(self.graph.take());
    }
}

fn validate_weights(stream: &CudaStream, weights: &PredictorWeights<'_>) -> Result<()> {
    if weights.layers.len() != PREDICTOR_LAYERS {
        return Err("predictor requires five layers".into());
    }
    let context = |buffer: &DeviceBuffer<Bf16>| -> Result<()> {
        if Arc::ptr_eq(buffer.context(), stream.context()) {
            Ok(())
        } else {
            Err("predictor weight context mismatch".into())
        }
    };
    let shape = DecoderShape::<1024, 3072, 16, 8, 128>::new(2, 1024)?;
    for layer in &weights.layers {
        shape.validate_weights(layer)?;
        let a = layer.attention();
        let f = layer.ffn();
        for buffer in [
            layer.input_norm().buffer(),
            layer.post_attention_norm().buffer(),
            a.query().buffer(),
            a.key().buffer(),
            a.value().buffer(),
            a.output().buffer(),
            a.query_norm().buffer(),
            a.key_norm().buffer(),
            f.gate().buffer(),
            f.up().buffer(),
            f.down().buffer(),
        ] {
            context(buffer)?;
        }
    }
    weights
        .projection
        .shape()
        .validate_len(weights.projection.buffer().len())?;
    for buffer in [
        weights.projection.buffer(),
        weights.bias.buffer(),
        weights.norm.buffer(),
    ] {
        context(buffer)?;
    }
    for (head, embedding) in weights.heads.iter().zip(&weights.embeddings) {
        head.shape().validate_len(head.buffer().len())?;
        embedding
            .shape()
            .table()
            .validate_len(embedding.buffer().len())?;
        context(head.buffer())?;
        context(embedding.buffer())?;
    }
    Ok(())
}

impl PredictorPlan {
    pub(crate) fn new(
        stream: Arc<CudaStream>,
        weights: Arc<QwenWeights>,
        capture: bool,
    ) -> Result<Self> {
        // A private nondefault stream is capturable and cannot be used by a
        // caller to create conflicting work against this plan's private owners.
        let stream = stream.context().new_stream()?;
        validate_weights(&stream, &weights.predictor()?)?;
        stream.context().synchronize()?;
        let modules = Modules::new(&stream)?;
        let prefill = Scratch::new(&stream, &modules.attention, 2)?;
        let decode = Scratch::new(&stream, &modules.attention, 1)?;
        let rotary = RotarySpec::<128>::standard(1_000_000.0, 65536)?;
        let mut tables = Vec::with_capacity(REMAINING_CODES);
        for step in 0..REMAINING_CODES {
            let positions = if step == 0 {
                vec![[0; 3], [1; 3]]
            } else {
                vec![[(step + 1) as u32; 3]]
            };
            let host = rotary.tables(&positions)?;
            tables.push(Tables {
                cos: DeviceBuffer::from_host(&stream, host.cos())?,
                sin: DeviceBuffer::from_host(&stream, host.sin())?,
            });
        }
        let caches = (0..PREDICTOR_LAYERS)
            .map(|_| {
                Ok(LayerCache {
                    key: Guarded::new(&stream, 16)?,
                    value: Guarded::new(&stream, 16)?,
                })
            })
            .collect::<Result<_>>()?;
        let mut plan = Self {
            graph: None,
            weights,
            modules,
            prefill,
            decode,
            caches,
            tables,
            first_input: Batch::from_host(&stream, 2, &[Bf16::from_bits(0); 4096])?,
            feedback: Batch::allocate(&stream, 1)?,
            codec_rows: Batch::from_host(&stream, 16, &[Bf16::from_bits(0); 32768])?,
            embedding_sum: Batch::allocate(&stream, 1)?,
            codes: DeviceBuffer::from_host(&stream, &[0u32; REMAINING_CODES])?,
            status: DeviceBuffer::from_host(&stream, &[0u32])?,
            stream,
            poisoned: false,
        };
        // Warm every actual entry before capture: lazy function resolution or
        // module code loading is setup work, never part of the captured region.
        let weights_owner = plan.weights.clone();
        let views = weights_owner.predictor()?;
        let completion = Completion::new(&plan.stream);
        let warmed = plan.enqueue(&views);
        let completed = completion.finish();
        warmed?;
        completed?;
        if capture {
            let capture = Capture::begin(&plan.stream)?;
            let enqueued = plan.enqueue(&views);
            // End capture even when enqueue returns an error. Capture's Drop
            // handles unwinding before any plan/resource completion can run.
            let graph = capture.end();
            enqueued?;
            plan.graph = Some(graph?);
        }
        Ok(plan)
    }

    fn validate_inputs(&self, last: &Batch<2048>, zero: &Batch<2048>) -> Result<()> {
        if self.poisoned {
            return Err("failed predictor plan must be rebuilt before reuse".into());
        }
        for input in [last, zero] {
            if input.rows() != 1 || input.buffer().len() != 2048 {
                return Err("predictor requires one dense 2048-wide input row".into());
            }
            if !Arc::ptr_eq(input.buffer().context(), self.stream.context()) {
                return Err("predictor input context mismatch".into());
            }
        }
        Ok(())
    }

    pub(crate) fn run(&mut self, last: &Batch<2048>, zero: &Batch<2048>) -> Result<CodecFrame> {
        self.validate_inputs(last, zero)?;
        let output = Batch::allocate(&self.stream, 1)?;
        self.run_into(last, zero, output)
    }

    // Accept a preallocated output for readiness/error tests, so allocation's
    // incidental driver synchronization cannot hide a missing input fence.
    fn run_into(
        &mut self,
        last: &Batch<2048>,
        zero: &Batch<2048>,
        mut embedding_sum: Batch<2048>,
    ) -> Result<CodecFrame> {
        self.validate_inputs(last, zero)?;
        if embedding_sum.rows() != 1
            || embedding_sum.buffer().len() != 2048
            || !Arc::ptr_eq(embedding_sum.buffer().context(), self.stream.context())
        {
            return Err("predictor output must be one dense row in the plan context".into());
        }
        // Public buffers may have pending producers on any stream. Retain both
        // borrows until copies/graph complete, including launch errors/unwind.
        self.stream.context().synchronize()?;
        let weights_owner = self.weights.clone();
        let views = if self.graph.is_none() {
            Some(weights_owner.predictor()?)
        } else {
            None
        };
        self.poisoned = true;
        let completion = Completion::new(&self.stream);
        let enqueued = (|| -> Result<()> {
            // SAFETY: exact checked caller rows, same context, distinct private
            // destinations. Fixed copy extents fit their dense typed owners.
            unsafe {
                self.modules.elements.concat_bf16(
                    &self.stream,
                    grid(4096),
                    last.buffer(),
                    zero.buffer(),
                    self.first_input.buffer_mut(),
                    2048,
                    4096,
                )?;
                self.modules.schedule.copy_range(
                    &self.stream,
                    grid(2048),
                    zero.buffer(),
                    self.codec_rows.buffer_mut(),
                    0,
                    0,
                    2048,
                )?;
                if let Some(graph) = &self.graph {
                    driver(sys::cuGraphLaunch(
                        graph.executable,
                        self.stream.cu_stream(),
                    ))?;
                } else {
                    self.enqueue(
                        views
                            .as_ref()
                            .expect("eager views prepared before submission"),
                    )?;
                }
                self.modules.schedule.copy_range(
                    &self.stream,
                    grid(2048),
                    self.embedding_sum.buffer(),
                    embedding_sum.buffer_mut(),
                    0,
                    0,
                    2048,
                )?;
            }
            Ok(())
        })();
        let completed = completion.finish();
        enqueued?;
        completed?;
        let status = self.status.to_host_vec(&self.stream)?[0];
        if status != 0 {
            return Err(format!(
                "predictor device validation failed (status {status}: nonfinite=1, guard=2)"
            )
            .into());
        }
        let remaining_codes = self
            .codes
            .to_host_vec(&self.stream)?
            .try_into()
            .map_err(|_| "predictor code count mismatch")?;
        self.poisoned = false;
        Ok(CodecFrame {
            remaining_codes,
            embedding_sum,
        })
    }

    /// Private schedule: every shape, context, pitch and allocation was checked
    /// before entry; all owners are fixed and inaccessible to outside enqueues.
    /// No allocation, upload, host readback or synchronization occurs here.
    fn enqueue(&mut self, weights: &PredictorWeights<'_>) -> Result<()> {
        let m = &self.modules;
        let stream = &self.stream;
        // SAFETY: the plan owns checked, initialized and mutually disjoint
        // resources, on one ordered stream. New/run/capture RAII retains them
        // until execution completion or discards a nonexecuted capture.
        unsafe {
            m.schedule
                .reset_status(stream, one_warp(), &mut self.status)?;
            for step in 0..REMAINING_CODES {
                let s = if step == 0 {
                    &mut self.prefill
                } else {
                    &mut self.decode
                };
                let input = if step == 0 {
                    &self.first_input
                } else {
                    &self.feedback
                };
                let start = if step == 0 { 0 } else { step + 1 };
                let key_tokens = step + 2;
                let table = &self.tables[step];
                m.linear.decode_linear_bias_bf16(
                    stream,
                    &m.biased_1024,
                    input.buffer(),
                    weights.projection.buffer(),
                    weights.bias.buffer(),
                    s.hidden.buffer_mut(),
                    s.rows,
                    2048,
                    1024,
                    2048,
                    weights.projection.shape().row_stride(),
                )?;
                for (weight, cache) in weights.layers.iter().zip(&mut self.caches) {
                    let a = weight.attention();
                    let f = weight.ffn();
                    crate::norm_specialized::enqueue(
                        &m.norms,
                        stream,
                        s.hidden.buffer(),
                        weight.input_norm().buffer(),
                        s.normalized.buffer_mut(),
                        s.rows,
                        1024,
                        EPSILON,
                    )?;
                    m.project(
                        stream,
                        s.normalized.buffer(),
                        s.rows,
                        a.query(),
                        &mut s.query,
                    )?;
                    m.project(stream, s.normalized.buffer(), s.rows, a.key(), &mut s.key)?;
                    m.project(
                        stream,
                        s.normalized.buffer(),
                        s.rows,
                        a.value(),
                        &mut s.value,
                    )?;
                    crate::norm_specialized::enqueue(
                        &m.norms,
                        stream,
                        s.query.buffer(),
                        a.query_norm().buffer(),
                        s.query_norm.buffer_mut(),
                        s.rows * 16,
                        128,
                        EPSILON,
                    )?;
                    crate::norm_specialized::enqueue(
                        &m.norms,
                        stream,
                        s.key.buffer(),
                        a.key_norm().buffer(),
                        s.key_norm.buffer_mut(),
                        s.rows * 8,
                        128,
                        EPSILON,
                    )?;
                    m.rotary.rotate_bf16(
                        stream,
                        grid(s.rows * 1024),
                        s.query_norm.buffer(),
                        &table.cos,
                        &table.sin,
                        &mut s.rotated_query.buffer,
                        s.rows,
                        16,
                        128,
                    )?;
                    m.rotary.rotate_bf16(
                        stream,
                        grid(s.rows * 512),
                        s.key_norm.buffer(),
                        &table.cos,
                        &table.sin,
                        &mut s.rotated_key.buffer,
                        s.rows,
                        8,
                        128,
                    )?;
                    m.rotary.append_bf16(
                        stream,
                        grid(s.rows * 1024),
                        &s.rotated_key.buffer,
                        s.value.buffer(),
                        &mut cache.key.buffer,
                        &mut cache.value.buffer,
                        start * 1024,
                        s.rows * 1024,
                    )?;
                    m.attention.partial(
                        stream,
                        &s.partial_launch,
                        &s.rotated_query.buffer,
                        &cache.key.buffer,
                        &cache.value.buffer,
                        &mut s.partials,
                        s.rows as u32,
                        key_tokens as u32,
                        16,
                        8,
                        start as u32,
                        true,
                    )?;
                    m.attention.merge(
                        stream,
                        &s.merge_launch,
                        &s.partials,
                        &mut s.attended.buffer,
                        s.rows as u32,
                        16,
                        1,
                    )?;
                    m.schedule.attention_guards(
                        stream,
                        one_warp(),
                        &s.partials,
                        &s.rotated_query.buffer,
                        &s.rotated_key.buffer,
                        &s.attended.buffer,
                        &cache.key.buffer,
                        &cache.value.buffer,
                        &mut self.status,
                        s.rows,
                    )?;
                    m.project(
                        stream,
                        &s.attended.buffer,
                        s.rows,
                        a.output(),
                        &mut s.projected,
                    )?;
                    m.elements.add_bf16(
                        stream,
                        grid(s.rows * 1024),
                        s.hidden.buffer(),
                        s.projected.buffer(),
                        s.residual.buffer_mut(),
                        s.rows * 1024,
                    )?;
                    crate::norm_specialized::enqueue(
                        &m.norms,
                        stream,
                        s.residual.buffer(),
                        weight.post_attention_norm().buffer(),
                        s.post.buffer_mut(),
                        s.rows,
                        1024,
                        EPSILON,
                    )?;
                    m.project(stream, s.post.buffer(), s.rows, f.gate(), &mut s.gate)?;
                    m.project(stream, s.post.buffer(), s.rows, f.up(), &mut s.up)?;
                    m.linear.activate_bf16(
                        stream,
                        grid((s.rows * 3072).div_ceil(8)),
                        s.gate.buffer(),
                        s.up.buffer(),
                        s.silu.buffer_mut(),
                        s.product.buffer_mut(),
                        s.rows * 3072,
                    )?;
                    m.project(stream, s.product.buffer(), s.rows, f.down(), &mut s.down)?;
                    m.elements.add_bf16(
                        stream,
                        grid(s.rows * 1024),
                        s.residual.buffer(),
                        s.down.buffer(),
                        s.hidden.buffer_mut(),
                        s.rows * 1024,
                    )?;
                }
                crate::norm_specialized::enqueue(
                    &m.norms,
                    stream,
                    s.hidden.buffer(),
                    weights.norm.buffer(),
                    s.final_norm.buffer_mut(),
                    s.rows,
                    1024,
                    EPSILON,
                )?;
                m.schedule.copy_range(
                    stream,
                    grid(1024),
                    s.final_norm.buffer(),
                    s.last.buffer_mut(),
                    (s.rows - 1) * 1024,
                    0,
                    1024,
                )?;
                m.project(
                    stream,
                    s.last.buffer(),
                    1,
                    &weights.heads[step],
                    &mut s.logits,
                )?;
                m.schedule.greedy(
                    stream,
                    one_warp(),
                    s.logits.buffer(),
                    &mut self.codes,
                    &mut self.status,
                    step,
                )?;
                m.schedule.embedding(
                    stream,
                    grid(2048),
                    weights.embeddings[step].buffer(),
                    &self.codes,
                    self.feedback.buffer_mut(),
                    self.codec_rows.buffer_mut(),
                    step,
                    weights.embeddings[step].shape().table().row_stride(),
                )?;
            }
            m.elements.sum_rows_bf16(
                stream,
                grid(2048),
                self.codec_rows.buffer(),
                self.embedding_sum.buffer_mut(),
                16,
                2048,
            )?;
        }
        Ok(())
    }
}

#[path = "predictor_graph_validation.rs"]
mod validation;
pub(crate) use validation::validate;
