/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Explicit cluster membership, shared-address mapping and remote arrival.
use super::{
    groups::CtaGroup,
    memory::{ClusterSharedAddress, SharedAddress},
};
use crate::barrier::Barrier;
pub use crate::cluster::{
    barrier_cluster_arrive, barrier_cluster_arrive_aligned, barrier_cluster_arrive_relaxed,
    barrier_cluster_arrive_relaxed_aligned, barrier_cluster_wait, barrier_cluster_wait_aligned,
    block_rank, cluster_ctaidX, cluster_ctaidY, cluster_ctaidZ, cluster_idx, cluster_nctaidX,
    cluster_nctaidY, cluster_nctaidZ, cluster_size, num_clusters,
};
use core::marker::PhantomData;

/// A nonempty multicast mask. Group mode does not restrict cluster membership.
pub struct CtaMask<G: CtaGroup> {
    bits: u16,
    group: PhantomData<G>,
}
impl<G: CtaGroup> Copy for CtaMask<G> {}
impl<G: CtaGroup> Clone for CtaMask<G> {
    fn clone(&self) -> Self {
        *self
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskError {
    Empty,
}
impl<G: CtaGroup> CtaMask<G> {
    #[inline(always)]
    pub const fn try_from_bits(bits: u16) -> Result<Self, MaskError> {
        if bits == 0 {
            Err(MaskError::Empty)
        } else {
            Ok(Self {
                bits,
                group: PhantomData,
            })
        }
    }
    #[inline(always)]
    pub const fn bits(self) -> u16 {
        self.bits
    }
}
/// Map local shared storage to the same offset in a cluster peer.
///
/// # Safety
/// `rank` must name an active CTA in this cluster with matching live storage.
/// Remote initialization, readiness and lifetime must be coordinated explicitly.
#[inline(always)]
pub unsafe fn map_shared<'s, T>(
    address: SharedAddress<'s, T>,
    rank: u32,
) -> ClusterSharedAddress<'s, T> {
    unsafe {
        ClusterSharedAddress::from_raw(crate::cluster::map_shared_rank_mut(address.as_raw(), rank))
    }
}
/// Issue one remote arrival without waiting or changing a software phase.
///
/// # Safety
/// The remote barrier must be initialized and expecting this arrival. All
/// participants must agree on the phase and keep its storage live.
#[inline(always)]
pub unsafe fn arrive_remote(address: ClusterSharedAddress<'_, Barrier>) {
    unsafe {
        let offset =
            crate::shared::cvta_generic_to_cluster_shared_offset(address.as_raw().cast::<u8>());
        crate::barrier::mbarrier_arrive_cluster(offset)
    }
}
