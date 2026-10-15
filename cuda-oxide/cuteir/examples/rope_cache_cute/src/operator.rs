/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Synchronous BF16 rotary composition with privately owned key/value caches.
use crate::{kernel::kernels, rotary::HostTables};
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaStream, DeviceBuffer};
use cute_rs::{
    Bf16,
    attention::{AttentionShape, CacheShape},
};
use std::sync::Arc;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const GUARD: u16 = 0x7fc1;
pub const GUARD_LEN: usize = 8;

/// Check the BF16 allocation extent including its trailing sentinel elements.
fn guarded_len(logical_len: usize) -> Result<usize> {
    let elements = logical_len
        .checked_add(GUARD_LEN)
        .ok_or("guard extent overflow")?;
    let bytes = elements
        .checked_mul(size_of::<Bf16>())
        .ok_or("guard byte extent overflow")?;
    if bytes > isize::MAX as usize {
        return Err("guarded BF16 allocation exceeds the slice address range".into());
    }
    Ok(elements)
}

/// One thread handles one rotation pair or one appended BF16 element.
fn launch_grid(work_items: usize) -> Result<Option<LaunchConfig>> {
    if work_items == 0 {
        return Ok(None);
    }
    let blocks = work_items.div_ceil(128);
    if blocks > i32::MAX as usize {
        return Err("work exceeds the CUDA grid-x range".into());
    }
    Ok(Some(LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }))
}

/// Allocate sentinel-filled storage only after checking the full guarded extent.
pub(crate) fn allocate(stream: &CudaStream, logical_len: usize) -> Result<DeviceBuffer<Bf16>> {
    let elements = guarded_len(logical_len)?;
    stream.context().bind_to_thread()?;
    Ok(DeviceBuffer::from_host(
        stream,
        &vec![Bf16::from_bits(GUARD); elements],
    )?)
}

/// Borrow dense BF16 inputs with distinct query/KV head dimensions.
///
/// Exact lengths reject accidental reinterpretation of trailing storage as
/// tokens. Context and pending producer work are checked by the runner.
pub struct AttentionInputs<'a, const Q: usize, const K: usize, const D: usize> {
    query: &'a DeviceBuffer<Bf16>,
    key: &'a DeviceBuffer<Bf16>,
    value: &'a DeviceBuffer<Bf16>,
    shape: AttentionShape<Q, K, D>,
    tokens: usize,
}

impl<'a, const Q: usize, const K: usize, const D: usize> AttentionInputs<'a, Q, K, D> {
    pub fn new(
        query: &'a DeviceBuffer<Bf16>,
        key: &'a DeviceBuffer<Bf16>,
        value: &'a DeviceBuffer<Bf16>,
        tokens: usize,
    ) -> Result<Self> {
        let shape = AttentionShape::new()?;
        if query.len() != shape.query_len(tokens)?
            || key.len() != shape.kv_len(tokens)?
            || value.len() != shape.kv_len(tokens)?
        {
            return Err(
                "attention input lengths must exactly match the token/head dimensions".into(),
            );
        }
        Ok(Self {
            query,
            key,
            value,
            shape,
            tokens,
        })
    }
}

/// Own cache storage and publish only the prefix completed by synchronous appends.
///
/// Buffer ownership and head dimensions cannot be replaced by callers. A reset
/// discards the logical prefix; it does not expose or clear old allocation data.
pub struct KvCache<const K: usize, const D: usize> {
    key: DeviceBuffer<Bf16>,
    value: DeviceBuffer<Bf16>,
    shape: CacheShape<K, D>,
    valid_tokens: usize,
}

impl<const K: usize, const D: usize> KvCache<K, D> {
    pub fn new(stream: &CudaStream, capacity: usize) -> Result<Self> {
        let shape = CacheShape::new(capacity)?;
        Ok(Self {
            key: allocate(stream, shape.allocation_len())?,
            value: allocate(stream, shape.allocation_len())?,
            shape,
            valid_tokens: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.valid_tokens
    }

    pub fn capacity(&self) -> usize {
        self.shape.capacity()
    }

    /// Return all capacity and sentinel elements for this example's validation.
    /// Only the first `len()` tokens represent the current cache prefix.
    pub fn snapshot(&self, stream: &CudaStream) -> Result<(Vec<Bf16>, Vec<Bf16>)> {
        if !Arc::ptr_eq(self.key.context(), stream.context())
            || !Arc::ptr_eq(self.value.context(), stream.context())
        {
            return Err("snapshot stream must share the cache context".into());
        }
        stream.context().synchronize()?;
        let mut key = vec![Bf16::from_bits(0); self.key.len()];
        let mut value = vec![Bf16::from_bits(0); self.value.len()];
        self.key.copy_to_host(stream, &mut key)?;
        self.value.copy_to_host(stream, &mut value)?;
        Ok((key, value))
    }

    /// Discard the completed prefix; the next append starts at token zero.
    pub fn reset(&mut self) {
        self.valid_tokens = 0;
    }
}

/// Owned rotated query storage, retaining its query-head and head-width types.
pub struct RotatedQuery<const Q: usize, const D: usize> {
    buffer: DeviceBuffer<Bf16>,
    tokens: usize,
}

impl<const Q: usize, const D: usize> RotatedQuery<Q, D> {
    /// Read-only buffer including the trailing validation sentinel elements.
    pub fn buffer(&self) -> &DeviceBuffer<Bf16> {
        &self.buffer
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }
}

/// Bind a loaded kernel module to one stream and context.
pub struct Runner {
    module: kernels::LoadedModule,
    attention: crate::attention::AttentionRunner,
    stream: Arc<CudaStream>,
}

impl Runner {
    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        Ok(Self {
            module: kernels::load(stream.context())?,
            attention: crate::attention::AttentionRunner::new(stream.clone())?,
            stream,
        })
    }

    /// Attend over the completed cache prefix with fixed 128-wide BF16 heads.
    /// `query_start` is a physical causal cache offset, independent of RoPE
    /// positions. `None` selects all available keys for every query.
    pub fn attend<const Q: usize, const K: usize>(
        &self,
        query: &RotatedQuery<Q, 128>,
        cache: &KvCache<K, 128>,
        query_start: Option<usize>,
    ) -> Result<crate::attention::AttentionOutput<Q>> {
        self.attention.run::<Q, K>(
            &query.buffer,
            query.tokens,
            &cache.key,
            &cache.value,
            cache.valid_tokens,
            query_start,
        )
    }

    /// Rotate Q/K and append completed K/V, retaining every borrow through completion.
    ///
    /// Coefficients are explicit inputs. Their token count and head dimension
    /// are checked, while their semantic positions remain the caller's choice.
    /// The current synchronous runner orders prior work in its whole context.
    pub fn append<const Q: usize, const K: usize, const D: usize>(
        &self,
        cache: &mut KvCache<K, D>,
        inputs: AttentionInputs<'_, Q, K, D>,
        tables: &HostTables<D>,
    ) -> Result<RotatedQuery<Q, D>> {
        if !D.is_multiple_of(2) {
            return Err("split-half rotary head width must be even".into());
        }
        if tables.tokens() != inputs.tokens {
            return Err("rotary table token count does not match attention inputs".into());
        }
        for buffer in [
            inputs.query,
            inputs.key,
            inputs.value,
            &cache.key,
            &cache.value,
        ] {
            if !Arc::ptr_eq(buffer.context(), self.stream.context()) {
                return Err("attention inputs and cache must share the runner context".into());
            }
        }
        let range = cache
            .shape
            .append_range(cache.valid_tokens, inputs.tokens)?;
        let query_len = inputs.shape.query_len(inputs.tokens)?;
        let kv_len = inputs.shape.kv_len(inputs.tokens)?;
        let query_grid = launch_grid(query_len / 2)?;
        let key_grid = launch_grid(kv_len / 2)?;
        let append_grid = launch_grid(kv_len)?;
        let start_element = range
            .start
            .checked_mul(cache.shape.width())
            .ok_or("cache offset overflow")?;
        // Safe asynchronous writes may be pending on another stream. Establish
        // readiness before any kernel reads the borrowed input allocations.
        self.stream.context().synchronize()?;
        let mut query = RotatedQuery {
            buffer: allocate(&self.stream, query_len)?,
            tokens: inputs.tokens,
        };
        if inputs.tokens == 0 {
            return Ok(query);
        }
        let query_grid = query_grid.ok_or("missing query rotation grid")?;
        let key_grid = key_grid.ok_or("missing key rotation grid")?;
        let append_grid = append_grid.ok_or("missing cache append grid")?;
        let mut rotated_key = allocate(&self.stream, kv_len)?;
        let cos = DeviceBuffer::from_host(&self.stream, tables.cos())?;
        let sin = DeviceBuffer::from_host(&self.stream, tables.sin())?;
        // SAFETY: lengths, contexts, grids and capacity were checked above.
        // Inputs remain borrowed; every output is a distinct owned allocation.
        // Rotation threads own disjoint pairs, and append threads own disjoint
        // elements beyond the completed cache prefix. One stream orders all
        // launches; unconditional synchronization retains their buffers on error.
        let enqueued = (|| -> Result<()> {
            unsafe {
                self.module.rotate_bf16(
                    &self.stream,
                    query_grid,
                    inputs.query,
                    &cos,
                    &sin,
                    &mut query.buffer,
                    inputs.tokens,
                    Q,
                    D,
                )?;
                self.module.rotate_bf16(
                    &self.stream,
                    key_grid,
                    inputs.key,
                    &cos,
                    &sin,
                    &mut rotated_key,
                    inputs.tokens,
                    K,
                    D,
                )?;
                self.module.append_bf16(
                    &self.stream,
                    append_grid,
                    &rotated_key,
                    inputs.value,
                    &mut cache.key,
                    &mut cache.value,
                    start_element,
                    kv_len,
                )?;
            }
            Ok(())
        })();
        let completed = self.stream.synchronize();
        enqueued?;
        completed?;
        cache.valid_tokens = range.end;
        Ok(query)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guards_cover_empty_and_nonempty_buffers_without_exceeding_slice_extent() {
        assert_eq!(guarded_len(0).unwrap(), 8);
        assert_eq!(guarded_len(17).unwrap(), 25);
        let largest = isize::MAX as usize / size_of::<cute_rs::Bf16>();
        assert_eq!(guarded_len(largest - GUARD_LEN).unwrap(), largest);
        assert!(guarded_len(largest - GUARD_LEN + 1).is_err());
        assert!(guarded_len(usize::MAX).is_err());
    }

    #[test]
    fn launch_grid_covers_partial_blocks_and_skips_empty_work() {
        assert!(launch_grid(0).unwrap().is_none());
        for (elements, blocks) in [(1, 1), (127, 1), (128, 1), (129, 2), (257, 3)] {
            let grid = launch_grid(elements).unwrap().unwrap();
            assert_eq!(grid.grid_dim, (blocks, 1, 1));
            assert_eq!(grid.block_dim, (128, 1, 1));
            assert_eq!(grid.shared_mem_bytes, 0);
        }
    }

    #[test]
    fn launch_grid_rejects_cuda_grid_overflow_before_casting() {
        let largest = i32::MAX as usize * 128;
        assert_eq!(
            launch_grid(largest).unwrap().unwrap().grid_dim.0,
            i32::MAX as u32
        );
        assert!(launch_grid(largest + 1).is_err());
        assert!(launch_grid(usize::MAX).is_err());
    }
}
