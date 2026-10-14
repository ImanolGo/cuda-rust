/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SM100 tensor-memory, collector-aware tiled MMA, and cluster pipelines.
//!
//! These operations keep ordinary MIR scalar carriers and retain the static
//! layout and collective contracts as typed attributes. A paired MMA is one
//! semantic operation: its A collector is filled and last-used within each
//! K=16 step, so a backend cannot accidentally carry it across K iterations.
//!
//! Cluster TMA loads are collective across all 32 producer-warp lanes. Native
//! CuTe lowering owns lane election and warp synchronization; callers must
//! not place a load inside an elected-lane branch. TMA stores retain a single
//! issuer, and accumulator release fences all lanes before electing an arrival.
//!
//! All operations have zero SSA results. Their docs give the operand order for
//! `new`; effects update shared memory, tensor memory, or hardware barriers.
//! TMEM addresses are encoded `u32` addresses, not byte pointers. Ring stages
//! are zero-based indices, and phases are parity bits maintained by the caller.

use dialect_mir::ops::MirConstantOp;
use dialect_mir::types::{MirFP16Type, MirPtrType, address_space};
use pliron::attribute::Attribute;
use pliron::builtin::types::{FP32Type, IntegerType};
use pliron::common_traits::Verify;
use pliron::context::{Context, Ptr};
use pliron::location::Located;
use pliron::op::{Op, OpId};
use pliron::operation::Operation;
use pliron::result::Error;
use pliron::r#type::Typed;
use pliron::value::Value;
use pliron::{verify_err, verify_err_noloc};
use pliron_derive::{pliron_attr, pliron_op};

use crate::attributes::{CuteComposedLayoutAttr, CuteTmaStorePipelineAttr};
use crate::layout::{ComposedLayout, Layout, OffsetUnit, Swizzle};

/// The first N partition fills A, and the second consumes its last use.
#[pliron_attr(name = "cute.sm100_collector", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CuteSm100CollectorAttr {
    /// Fill the A collector for the first product and consume its last use in the second.
    AFillLastUse,
}

/// Capacity of one two-CTA tensor-memory allocation, in columns per CTA.
///
/// Each CTA has 128 rows. The supported column count is a power of two in
/// `32..=512`; allocation and deallocation must use the same plan.
#[pliron_attr(
    name = "cute.sm100_tmem_plan",
    format = "`<` $columns `,` $cta_group `>`"
)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CuteSm100TmemPlanAttr {
    /// Tensor-memory columns allocated per CTA.
    pub columns: u32,
    /// Number of CTAs participating in the collective contract.
    pub cta_group: u32,
}
impl Verify for CuteSm100TmemPlanAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.cta_group != 2
            || !(32..=512).contains(&self.columns)
            || !self.columns.is_power_of_two()
        {
            return verify_err_noloc!(
                "SM100 tensor memory requires CTA group 2 and power-of-two columns in 32..=512"
            );
        }
        Ok(())
    }
}

/// Two FP16 MMA products sharing A, accumulated into adjacent TMEM partitions.
///
/// `m`, `n0`, `n1`, and `k` describe the complete two-CTA products. The shared
/// layouts describe each CTA's portion: A is `(m / 2, k)`, and each B is
/// `(n / 2, k)`, all K-major with a 128-byte swizzle. The current path fixes
/// `m = 256`, `k = 64`, and `tmem_columns = 512`; each N is a multiple of 16
/// in `16..=256`. B1's output follows B0's `n0` columns.
#[pliron_attr(
    name = "cute.sm100_mma_plan",
    format = "`<` $m `,` $n0 `,` $n1 `,` $k `,` $tmem_columns `,` $cta_group `,` $collector `,` $a_layout `,` $b0_layout `,` $b1_layout `>`"
)]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct CuteSm100MmaPlanAttr {
    /// Collective output-row extent M.
    pub m: u32,
    /// Output columns in the first product, starting at TMEM column zero.
    pub n0: u32,
    /// Output columns in the second product, immediately following the first.
    pub n1: u32,
    /// Reduction extent K, in logical elements.
    pub k: u32,
    /// TMEM capacity in columns available to both output partitions.
    pub tmem_columns: u32,
    /// Number of CTAs participating in the collective contract.
    pub cta_group: u32,
    /// Lifetime policy for the shared A collector across the two products.
    pub collector: CuteSm100CollectorAttr,
    /// Physical shared placement of each CTA's A tile.
    pub a_layout: CuteComposedLayoutAttr,
    /// Physical shared placement of each CTA's first B tile.
    pub b0_layout: CuteComposedLayoutAttr,
    /// Physical shared placement of each CTA's second B tile.
    pub b1_layout: CuteComposedLayoutAttr,
}

/// Build a zero-offset FP16 row-major layout with the requested hardware swizzle.
fn row_major(rows: u32, columns: u32, swizzle_bits: u32) -> ComposedLayout {
    let inner: Layout = format!("({rows},{columns}):({columns},1)")
        .parse()
        .expect("fixed rank-two layout");
    ComposedLayout::new(
        Swizzle::new(swizzle_bits, 3, 3),
        0,
        inner,
        OffsetUnit::Elements,
    )
    .expect("fixed SM100 FP16 swizzle")
}

impl CuteSm100MmaPlanAttr {
    /// Bytes copied by both CTAs for one complete A/B0/B1 stage.
    pub fn transaction_bytes(&self) -> Option<u32> {
        self.m
            .checked_add(self.n0)?
            .checked_add(self.n1)?
            .checked_mul(self.k)?
            .checked_mul(2)
    }
}
impl Verify for CuteSm100MmaPlanAttr {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        CuteSm100TmemPlanAttr {
            columns: self.tmem_columns,
            cta_group: self.cta_group,
        }
        .verify(ctx)?;
        if self.m != 256
            || self.k != 64
            || self.tmem_columns != 512
            || !(16..=256).contains(&self.n0)
            || !(16..=256).contains(&self.n1)
            || !self.n0.is_multiple_of(16)
            || !self.n1.is_multiple_of(16)
            || self.n0 + self.n1 > self.tmem_columns
        {
            return verify_err_noloc!(
                "SM100 paired FP16 MMA requires M256/K64, two N partitions in 16..=256 divisible by 16, and 512 TMEM columns"
            );
        }
        for (name, layout, rows) in [
            ("A", &self.a_layout, 128),
            ("B0", &self.b0_layout, self.n0 / 2),
            ("B1", &self.b1_layout, self.n1 / 2),
        ] {
            layout.verify(ctx)?;
            if layout.0 != row_major(rows, self.k, 3) {
                return verify_err_noloc!(
                    "SM100 {name} MMA layout must be the matching K-major FP16 B128 shared tile"
                );
            }
        }
        Ok(())
    }
}

/// One CTA's FP16 input tile in a two-CTA cluster TMA copy.
///
/// The layout is K-major with 64 columns and a 128-byte swizzle; its row count
/// is a multiple of eight in `8..=128`. Completion is routed to leader rank
/// zero. The caller's barrier expectation covers bytes copied by both CTAs.
#[pliron_attr(
    name = "cute.sm100_cluster_copy_plan",
    format = "`<` $layout `,` $cta_group `,` $leader_rank `>`"
)]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct CuteSm100ClusterCopyPlanAttr {
    /// Static coordinate layout used to interpret the storage.
    pub layout: CuteComposedLayoutAttr,
    /// Number of CTAs participating in the collective contract.
    pub cta_group: u32,
    /// CTA rank receiving collective completion notifications.
    pub leader_rank: u32,
}
impl Verify for CuteSm100ClusterCopyPlanAttr {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        self.layout.verify(ctx)?;
        let dimensions = self.layout.0.inner().shape.leaves();
        if self.cta_group != 2
            || self.leader_rank != 0
            || dimensions.len() != 2
            || dimensions[0] < 8
            || dimensions[0] > 128
            || dimensions[0] % 8 != 0
            || dimensions[1] != 64
            || self.layout.0 != row_major(dimensions[0] as u32, 64, 3)
        {
            return verify_err_noloc!(
                "SM100 cluster TMA requires CTA group 2, leader 0, and a K64 FP16 B128 tile with 8..=128 rows divisible by 8"
            );
        }
        Ok(())
    }
}

/// A 128-by-32 FP16 shared output tile, with a 64-byte swizzle.
///
/// The epilogue fills this tile from FP32 TMEM, and the store operation sends
/// the same physical layout to global memory. Rows and columns count elements.
#[pliron_attr(
    name = "cute.sm100_epilogue_plan",
    format = "`<` $layout `,` $rows `,` $columns `>`"
)]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct CuteSm100EpiloguePlanAttr {
    /// Static coordinate layout used to interpret the storage.
    pub layout: CuteComposedLayoutAttr,
    /// Row extent of the shared output tile, in elements.
    pub rows: u32,
    /// Allocated or copied column extent, in elements.
    pub columns: u32,
}
impl Verify for CuteSm100EpiloguePlanAttr {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        self.layout.verify(ctx)?;
        if self.rows != 128 || self.columns != 32 || self.layout.0 != row_major(128, 32, 2) {
            return verify_err_noloc!("SM100 epilogue requires the 128x32 FP16 B64 shared layout");
        }
        Ok(())
    }
}

/// Static barrier configuration for either input stages or one accumulator.
///
/// Positive `transaction_bytes` selects the input pipeline: it counts all
/// A/B0/B1 bytes copied by both CTAs per stage, with one MMA completion arrival
/// releasing the stage. Zero selects the accumulator pipeline: one stage and
/// one release arrival per consumer warp across both CTAs. The current
/// epilogue has four consumer warps per CTA, hence eight arrivals.
/// Both forms use CTA group two, leader rank zero, and multicast mask `0b11`.
///
/// ```
/// use dialect_cute::sm100_ops::CuteSm100PipelinePlanAttr;
/// use pliron::{common_traits::Verify, context::Context};
///
/// let accumulator = CuteSm100PipelinePlanAttr {
///     stages: 1,
///     transaction_bytes: 0,
///     consumer_arrivals: 8,
///     cta_group: 2,
///     leader_rank: 0,
///     multicast_mask: 0b11,
/// };
/// accumulator.verify(&Context::new()).unwrap();
/// ```
#[pliron_attr(
    name = "cute.sm100_pipeline_plan",
    format = "`<` $stages `,` $transaction_bytes `,` $consumer_arrivals `,` $cta_group `,` $leader_rank `,` $multicast_mask `>`"
)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CuteSm100PipelinePlanAttr {
    /// Number of reusable slots in the barrier or output-buffer ring.
    pub stages: u32,
    /// Bytes expected by the full barrier for one complete input stage; zero selects accumulator synchronization.
    pub transaction_bytes: u32,
    /// Slot-release arrivals: one MMA completion for input, or one per epilogue warp for accumulators.
    pub consumer_arrivals: u32,
    /// Number of CTAs participating in the collective contract.
    pub cta_group: u32,
    /// CTA rank receiving collective completion notifications.
    pub leader_rank: u32,
    /// Bit mask of CTA ranks notified by a collective completion.
    pub multicast_mask: u32,
}
impl Verify for CuteSm100PipelinePlanAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.stages == 0
            || self.cta_group != 2
            || self.leader_rank != 0
            || self.multicast_mask != 3
        {
            return verify_err_noloc!(
                "SM100 pipeline requires positive stages, CTA group 2, leader 0, and multicast mask 3"
            );
        }
        if self.transaction_bytes == 0 {
            if self.stages != 1
                || self.consumer_arrivals == 0
                || self.consumer_arrivals > 64
                || !self.consumer_arrivals.is_multiple_of(2)
            {
                return verify_err_noloc!(
                    "SM100 accumulator pipeline requires one stage and an even number of consumer arrivals in 2..=64"
                );
            }
        } else if self.consumer_arrivals != 1 || !self.transaction_bytes.is_multiple_of(16) {
            return verify_err_noloc!(
                "SM100 input pipeline requires one MMA completion arrival and a transaction size divisible by 16"
            );
        }
        Ok(())
    }
}

/// Runtime scalar or pointer contract required at one positional operand.
#[derive(Clone, Copy)]
enum Carrier {
    /// Unsigned 32-bit index or encoded tensor-memory address.
    U32,
    /// One-bit predicate selecting whether MMA retains the previous accumulator.
    Bool,
    /// FP32 scalar storage or accumulation.
    F32,
    /// Writable shared pointer to a 32-bit allocation token.
    SharedU32,
    /// Writable shared pointer to a 64-bit hardware barrier.
    SharedBarrier,
    /// Writable shared FP16 tile pointer.
    SharedF16,
    /// Readable shared FP16 tile pointer.
    ReadSharedF16,
    /// Immutable tensor-map pointer in generic, global, or constant space.
    Descriptor,
}
/// Check an operand's scalar width or pointer storage/address-space/mutability contract.
fn carrier_matches(ctx: &Context, value: Value, carrier: Carrier) -> bool {
    let ty = value.get_type(ctx);
    let ty = ty.deref(ctx);
    let integer = |width| {
        ty.downcast_ref::<IntegerType>()
            .is_some_and(|i| i.width() == width && (width == 1 || i.is_unsigned()))
    };
    match carrier {
        Carrier::U32 => integer(32),
        Carrier::Bool => integer(1),
        Carrier::F32 => ty.downcast_ref::<FP32Type>().is_some(),
        _ => {
            let Some(ptr) = ty.downcast_ref::<MirPtrType>() else {
                return false;
            };
            if matches!(carrier, Carrier::Descriptor) {
                return !ptr.is_mutable
                    && matches!(
                        ptr.address_space,
                        address_space::GENERIC | address_space::GLOBAL | address_space::CONSTANT
                    );
            }
            if !matches!(
                ptr.address_space,
                address_space::SHARED | address_space::GENERIC
            ) {
                return false;
            }
            if !matches!(carrier, Carrier::ReadSharedF16) && !ptr.is_mutable {
                return false;
            }
            let pointee = ptr.pointee.deref(ctx);
            match carrier {
                Carrier::SharedU32 => pointee
                    .downcast_ref::<IntegerType>()
                    .is_some_and(|i| i.width() == 32 && i.is_unsigned()),
                Carrier::SharedBarrier => pointee
                    .downcast_ref::<IntegerType>()
                    .is_some_and(|i| i.width() == 64 && i.is_unsigned()),
                Carrier::SharedF16 | Carrier::ReadSharedF16 => {
                    pointee.downcast_ref::<MirFP16Type>().is_some()
                }
                _ => false,
            }
        }
    }
}

/// Generate zero-result operations with a typed plan and local carrier verification.
macro_rules! semantic_op {
    ($(#[$doc:meta])* $ty:ident, $name:literal, $key:ident, $plan:ty, [$($carrier:ident),*], $check:expr) => {
        #[pliron_op(name = $name, format, attributes = ($key: $plan))]
        pub struct $ty;
        // Keep contracts on the impl and constructor: pliron's generated
        // attribute-table links do not compose correctly with local struct docs.
        $(#[$doc])*
        impl $ty {
            $(#[$doc])*
            pub fn new(ctx: &mut Context, operands: Vec<Value>, plan: $plan) -> Self {
                let op = Self { op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], operands, vec![], 0) };
                op.get_operation().deref_mut(ctx).attributes.set(stringify!($key).try_into().expect("plan key"), plan);
                op
            }
            /// View an existing operation without allocating or verifying it.
            pub fn wrap(op: Ptr<Operation>) -> Self { Self { op } }
            /// Borrow the typed plan while retaining the operation's context borrow.
            pub fn get_attr_plan<'a>(&self, ctx: &'a Context) -> Option<std::cell::Ref<'a, $plan>> {
                std::cell::Ref::filter_map(self.get_operation().deref(ctx), |op|
                    op.attributes.get::<$plan>(&stringify!($key).try_into().expect("plan key"))).ok()
            }
            /// Clone the typed plan, or return `None` when the attribute is missing.
            pub fn plan(&self, ctx: &Context) -> Option<$plan> { self.get_attr_plan(ctx).map(|attr| (*attr).clone()) }
        }
        impl Verify for $ty {
            fn verify(&self, ctx: &Context) -> Result<(), Error> {
                let op = self.get_operation().deref(ctx);
                let expected: &[Carrier] = &[$(Carrier::$carrier),*];
                if op.get_num_operands() != expected.len() || op.get_num_results() != 0 {
                    return verify_err!(op.loc(), "{} needs {} operands and no results", $name, expected.len());
                }
                let Some(plan) = self.plan(ctx) else { return verify_err!(op.loc(), "{} requires its typed plan", $name); };
                plan.verify(ctx)?;
                if !($check)(&plan) { return verify_err!(op.loc(), "{} has the wrong pipeline role", $name); }
                for (index, carrier) in expected.iter().enumerate() {
                    if !carrier_matches(ctx, op.get_operand(index), *carrier) {
                        return verify_err!(op.loc(), "{} operand {} has an invalid scalar/pointer carrier", $name, index);
                    }
                }
                Ok(())
            }
        }
    };
}
semantic_op!(
    /// Allocate TMEM columns and relinquish the allocation permit.
    ///
    /// Operands: `[token]`, a shared `u32` destination for the encoded address.
    /// All lanes of the allocating warp in both CTAs participate; synchronize
    /// readers before they load the token.
    CuteSm100TmemAllocOp,
    "cute.sm100_tmem_alloc",
    sm100_tmem_alloc_plan,
    CuteSm100TmemPlanAttr,
    [SharedU32],
    |_| true
);
semantic_op!(
    /// Free a completed two-CTA TMEM allocation.
    ///
    /// Operands: `[tmem]`, the encoded base address. Both allocating warps
    /// participate after all asynchronous users finish and the CTAs rendezvous.
    CuteSm100TmemDeallocOp,
    "cute.sm100_tmem_dealloc",
    sm100_tmem_dealloc_plan,
    CuteSm100TmemPlanAttr,
    [U32],
    |_| true
);
semantic_op!(
    /// Issue the paired A×B0 and A×B1 products into TMEM.
    ///
    /// Operands: `[tmem, a, b0, b1, accumulate]`; A/B pointers address shared
    /// FP16 tiles. One elected leader lane issues after input completion.
    /// `accumulate` controls the first K step; subsequent steps accumulate.
    /// All source tiles and destination TMEM stay live until MMA completion.
    CuteSm100TiledMmaOp,
    "cute.sm100_tiled_mma",
    sm100_tiled_mma_plan,
    CuteSm100MmaPlanAttr,
    [U32, ReadSharedF16, ReadSharedF16, ReadSharedF16, Bool],
    |_| true
);
semantic_op!(
    /// Add a row bias to FP32 TMEM values and place FP16 results in shared memory.
    ///
    /// Operands: `[tmem_slice, destination, tid, bias]`. The encoded slice
    /// selects the caller's 32 TMEM rows and 32 columns; `tid` is in `0..128`.
    /// All lanes of four epilogue warps participate after accumulator wait.
    /// Publish their shared writes before issuing a TMA store.
    CuteSm100TmemEpilogueOp,
    "cute.sm100_tmem_epilogue",
    sm100_tmem_epilogue_plan,
    CuteSm100EpiloguePlanAttr,
    [U32, SharedF16, U32, F32],
    |_| true
);
semantic_op!(
    /// Start one CTA's contribution to a cluster input transfer.
    ///
    /// Operands: `[descriptor, destination, row, column, barrier, rank]`.
    /// Coordinates count global FP16 elements; rank is the local CTA rank.
    /// All producer-warp lanes participate. Each CTA's destination covers its
    /// local tile; the leader's expected byte count covers both CTAs.
    CuteSm100ClusterTmaLoadOp,
    "cute.sm100_cluster_tma_load",
    sm100_cluster_tma_load_plan,
    CuteSm100ClusterCopyPlanAttr,
    [Descriptor, SharedF16, U32, U32, SharedBarrier, U32],
    |_| true
);
semantic_op!(
    /// Initialize input barriers from one elected lane per CTA.
    ///
    /// Operands: `[full, empty]`, shared arrays of `plan.stages` barriers.
    /// Publish initialization and synchronize the cluster before use.
    CuteSm100PipelineInitOp,
    "cute.sm100_pipeline_init",
    sm100_pipeline_init_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, SharedBarrier],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes > 0
);
semantic_op!(
    /// Wait until the producer may reuse an input stage.
    ///
    /// Operands: `[empty, stage, phase]`. The producer starts with phase one;
    /// the stage must be inside the shared empty-barrier array.
    CuteSm100PipelineAcquireOp,
    "cute.sm100_pipeline_acquire",
    sm100_pipeline_acquire_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, U32, U32],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes > 0
);
semantic_op!(
    /// Set the full barrier's expected transaction bytes for one input stage.
    ///
    /// Operands: `[full, stage]`. One elected rank-zero lane calls after
    /// acquire; the plan counts the copies from both CTAs.
    CuteSm100PipelineExpectOp,
    "cute.sm100_pipeline_expect",
    sm100_pipeline_expect_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, U32],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes > 0
);
semantic_op!(
    /// Wait for an input stage and order subsequent tensor-core reads.
    ///
    /// Operands: `[full, stage, phase]`. Rank zero's MMA warp supplies its
    /// current consumer stage and phase.
    CuteSm100PipelineWaitOp,
    "cute.sm100_pipeline_wait",
    sm100_pipeline_wait_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, U32, U32],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes > 0
);
semantic_op!(
    /// Release an input stage in both CTAs after pending MMA reads complete.
    ///
    /// Operands: `[empty, stage]`. One elected leader lane calls after all
    /// MMAs reading that stage have been issued.
    CuteSm100PipelineReleaseOp,
    "cute.sm100_pipeline_release",
    sm100_pipeline_release_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, U32],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes > 0
);
semantic_op!(
    /// Initialize the single accumulator slot from one elected lane per CTA.
    ///
    /// Operands: `[empty, full]`, two distinct shared barriers. Publish their
    /// initialization cluster-wide before either side uses the slot.
    CuteSm100AccumulatorInitOp,
    "cute.sm100_accumulator_init",
    sm100_accumulator_init_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, SharedBarrier],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes == 0
);
semantic_op!(
    /// Wait until epilogue warps have released the previous accumulator tile.
    ///
    /// Operands: `[empty, phase]`. The leader CTA's MMA warp acquires with
    /// initial phase one.
    CuteSm100AccumulatorAcquireOp,
    "cute.sm100_accumulator_acquire",
    sm100_accumulator_acquire_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, U32],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes == 0
);
semantic_op!(
    /// Publish the accumulator to both CTAs after pending MMA completes.
    ///
    /// Operands: `[full]`. One elected leader lane calls after the last K stage.
    CuteSm100AccumulatorCommitOp,
    "cute.sm100_accumulator_commit",
    sm100_accumulator_commit_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes == 0
);
semantic_op!(
    /// Wait for an accumulator tile and order subsequent TMEM loads.
    ///
    /// Operands: `[full, phase]`. All consumer-warp lanes supply their current
    /// phase, initially zero.
    CuteSm100AccumulatorWaitOp,
    "cute.sm100_accumulator_wait",
    sm100_accumulator_wait_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, U32],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes == 0
);
semantic_op!(
    /// Release one consumer warp's accumulator portion to the leader CTA.
    ///
    /// Operands: `[empty]`. All epilogue-warp lanes call after their TMEM reads;
    /// lowering fences every lane and elects one arrival per warp.
    CuteSm100AccumulatorReleaseOp,
    "cute.sm100_accumulator_release",
    sm100_accumulator_release_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes == 0
);
semantic_op!(
    /// Start an asynchronous shared-to-global FP16 output transfer.
    ///
    /// Operands: `[descriptor, source, row, column]`; coordinates count global
    /// elements. One elected issuer calls after writers publish and synchronize.
    /// Keep the source unchanged until its committed group finishes reading it.
    CuteSm100TmaStoreOp,
    "cute.sm100_tma_store",
    sm100_tma_store_plan,
    CuteSm100EpiloguePlanAttr,
    [Descriptor, ReadSharedF16, U32, U32],
    |_| true
);
semantic_op!(
    /// Commit this issuer's pending TMA stores as one hardware group.
    ///
    /// No operands. The same issuer performs the matching acquire and tail.
    CuteSm100StoreCommitOp,
    "cute.sm100_store_commit",
    sm100_store_commit_plan,
    CuteTmaStorePipelineAttr,
    [],
    |_| true
);
semantic_op!(
    /// Wait until at most `plan.stages - 1` store groups still read shared memory.
    ///
    /// No operands. The store issuer may then reuse the oldest buffer in its
    /// ring; the newest group can still be pending.
    CuteSm100StoreAcquireOp,
    "cute.sm100_store_acquire",
    sm100_store_acquire_plan,
    CuteTmaStorePipelineAttr,
    [],
    |_| true
);
semantic_op!(
    /// Wait until all committed stores stop reading shared memory.
    ///
    /// No operands. The store issuer calls before releasing its output buffers.
    CuteSm100StoreTailOp,
    "cute.sm100_store_tail",
    sm100_store_tail_plan,
    CuteTmaStorePipelineAttr,
    [],
    |_| true
);

/// Apply a callback to the closed paired-SM100 operation set.
macro_rules! for_all_ops {
    ($callback:ident) => {
        $callback!(
            CuteSm100TmemAllocOp,
            CuteSm100TmemDeallocOp,
            CuteSm100TiledMmaOp,
            CuteSm100TmemEpilogueOp,
            CuteSm100ClusterTmaLoadOp,
            CuteSm100PipelineInitOp,
            CuteSm100PipelineAcquireOp,
            CuteSm100PipelineExpectOp,
            CuteSm100PipelineWaitOp,
            CuteSm100PipelineReleaseOp,
            CuteSm100AccumulatorInitOp,
            CuteSm100AccumulatorAcquireOp,
            CuteSm100AccumulatorCommitOp,
            CuteSm100AccumulatorWaitOp,
            CuteSm100AccumulatorReleaseOp,
            CuteSm100TmaStoreOp,
            CuteSm100StoreCommitOp,
            CuteSm100StoreAcquireOp,
            CuteSm100StoreTailOp
        )
    };
}
/// All operations belonging to this semantic family.
pub fn semantic_ids() -> Vec<OpId> {
    /// Collect the registered operation IDs from the supplied type list.
    macro_rules! ids { ($($ty:ty),*) => { vec![$(<$ty>::get_opid_static()),*] }; }
    for_all_ops!(ids)
}
/// Register the SM100 operations and their typed plans.
pub fn register(ctx: &mut Context) {
    /// Register every listed semantic operation in the supplied IR context.
    macro_rules! register_ops { ($($ty:ty),*) => { $(<$ty>::register(ctx);)* }; }
    for_all_ops!(register_ops);
    CuteSm100CollectorAttr::register(ctx);
    CuteSm100TmemPlanAttr::register(ctx);
    CuteSm100MmaPlanAttr::register(ctx);
    CuteSm100ClusterCopyPlanAttr::register(ctx);
    CuteSm100EpiloguePlanAttr::register(ctx);
    CuteSm100PipelinePlanAttr::register(ctx);
}
/// Locally verify a recognized operation; `None` means another family.
pub fn verify_local(ctx: &Context, op: Ptr<Operation>) -> Option<Result<(), Error>> {
    let id = Operation::get_opid(op, ctx);
    /// Select and run the local verifier for one recognized SM100 operation.
    macro_rules! verify_ops { ($($ty:ty),*) => { $(if id == <$ty>::get_opid_static() { return Some(<$ty>::wrap(op).verify(ctx)); })* }; }
    for_all_ops!(verify_ops);
    None
}

/// Find and clone the operation attribute of the requested static plan type.
fn static_plan<T: Attribute + Clone>(ctx: &Context, op: Ptr<Operation>) -> Option<T> {
    op.deref(ctx)
        .attributes
        .0
        .values()
        .find_map(|attr| attr.downcast_ref::<T>().cloned())
}

// CTM lowers its typed group-2 TMEM guard into these canonical NVVM ops.
// Read the column count when MIR still carries it as a direct constant; the
// group-2 op name is the static group contract in all cases.
/// Read a direct literal column count from a canonical CTM/NVVM allocation operation.
fn nvvm_tmem_columns(ctx: &Context, op: Ptr<Operation>) -> Option<u64> {
    let value = op.deref(ctx).get_operand(1);
    let constant = value.defining_op()?;
    if Operation::get_opid(constant, ctx) != MirConstantOp::get_opid_static() {
        return None;
    }
    let attribute = MirConstantOp::new(constant).get_attr_value(ctx)?;
    (attribute.value().bw() <= 64).then(|| attribute.value().to_u64())
}

/// Check static contracts across the scalar-carrier SM100 graph in one function.
///
/// This checks matching tile layouts, copy transaction sizes, TMEM capacity,
/// and complete lifecycle membership. Dynamic phase arithmetic, lane election,
/// and pointer aliasing remain the documented unsafe Rust preconditions.
pub fn verify_story(ctx: &Context, ops: &[Ptr<Operation>]) -> Result<(), String> {
    let id = |op| Operation::get_opid(op, ctx);
    let find = |wanted| ops.iter().copied().find(|op| id(*op) == wanted);
    let Some(mma_op) = find(CuteSm100TiledMmaOp::get_opid_static()) else {
        return Ok(());
    };
    let mma = CuteSm100TiledMmaOp::wrap(mma_op)
        .plan(ctx)
        .ok_or("SM100 MMA plan missing")?;
    let nvvm = |name: &str| {
        ops.iter()
            .copied()
            .filter(|op| id(*op).to_string() == name)
            .collect::<Vec<_>>()
    };
    let ctm_alloc = nvvm("nvvm.tcgen05_alloc_cg2");
    let ctm_dealloc = nvvm("nvvm.tcgen05_dealloc_cg2");
    if find(CuteSm100TmemAllocOp::get_opid_static()).is_none() && ctm_alloc.is_empty() {
        if !nvvm("nvvm.tcgen05_alloc").is_empty() {
            return Err("SM100 tiled MMA requires CTA group 2 TMEM allocation".into());
        }
        return Err(
            "SM100 tiled MMA is missing required lifecycle operation cute.sm100_tmem_alloc or nvvm.tcgen05_alloc_cg2"
                .into(),
        );
    }
    if find(CuteSm100TmemDeallocOp::get_opid_static()).is_none() && ctm_dealloc.is_empty() {
        if !nvvm("nvvm.tcgen05_dealloc").is_empty() {
            return Err("SM100 tiled MMA requires CTA group 2 TMEM deallocation".into());
        }
        return Err(
            "SM100 tiled MMA is missing required lifecycle operation cute.sm100_tmem_dealloc or nvvm.tcgen05_dealloc_cg2"
                .into(),
        );
    }
    if !ctm_alloc.is_empty() && nvvm("nvvm.tcgen05_relinquish_alloc_permit_cg2").is_empty() {
        return Err(
            "SM100 CTM group-2 TMEM allocation is missing its allocation permit relinquish".into(),
        );
    }
    for op in ctm_alloc.into_iter().chain(ctm_dealloc) {
        if op.deref(ctx).get_num_operands() != 2 {
            return Err("SM100 CTM TMEM lifecycle operation must have two operands".into());
        }
        if nvvm_tmem_columns(ctx, op).is_some_and(|columns| columns != mma.tmem_columns as u64) {
            return Err("SM100 allocation capacity disagrees with tiled MMA".into());
        }
    }
    for needed in [
        CuteSm100PipelineInitOp::get_opid_static(),
        CuteSm100PipelineAcquireOp::get_opid_static(),
        CuteSm100PipelineExpectOp::get_opid_static(),
        CuteSm100PipelineWaitOp::get_opid_static(),
        CuteSm100PipelineReleaseOp::get_opid_static(),
        CuteSm100AccumulatorInitOp::get_opid_static(),
        CuteSm100AccumulatorAcquireOp::get_opid_static(),
        CuteSm100AccumulatorCommitOp::get_opid_static(),
        CuteSm100AccumulatorWaitOp::get_opid_static(),
        CuteSm100AccumulatorReleaseOp::get_opid_static(),
        CuteSm100TmemEpilogueOp::get_opid_static(),
        CuteSm100TmaStoreOp::get_opid_static(),
        CuteSm100StoreCommitOp::get_opid_static(),
        CuteSm100StoreAcquireOp::get_opid_static(),
        CuteSm100StoreTailOp::get_opid_static(),
    ] {
        if find(needed.clone()).is_none() {
            return Err(format!(
                "SM100 tiled MMA is missing required lifecycle operation {needed}"
            ));
        }
    }
    let mut copied_layouts = Vec::new();
    let mut input_plan = None;
    let mut accumulator_plan = None;
    let mut store_plan = None;
    let input_ids = [
        CuteSm100PipelineInitOp::get_opid_static(),
        CuteSm100PipelineAcquireOp::get_opid_static(),
        CuteSm100PipelineExpectOp::get_opid_static(),
        CuteSm100PipelineWaitOp::get_opid_static(),
        CuteSm100PipelineReleaseOp::get_opid_static(),
    ];
    let accumulator_ids = [
        CuteSm100AccumulatorInitOp::get_opid_static(),
        CuteSm100AccumulatorAcquireOp::get_opid_static(),
        CuteSm100AccumulatorCommitOp::get_opid_static(),
        CuteSm100AccumulatorWaitOp::get_opid_static(),
        CuteSm100AccumulatorReleaseOp::get_opid_static(),
    ];
    for &op in ops {
        let opid = id(op);
        if opid == CuteSm100TiledMmaOp::get_opid_static()
            && CuteSm100TiledMmaOp::wrap(op).plan(ctx).as_ref() != Some(&mma)
        {
            return Err("SM100 tiled MMA plans in one function must agree".into());
        }
        if opid == CuteSm100TmemAllocOp::get_opid_static()
            || opid == CuteSm100TmemDeallocOp::get_opid_static()
        {
            let plan =
                static_plan::<CuteSm100TmemPlanAttr>(ctx, op).ok_or("SM100 TMEM plan missing")?;
            if plan.columns != mma.tmem_columns || plan.cta_group != mma.cta_group {
                return Err("SM100 allocation capacity disagrees with tiled MMA".into());
            }
        }
        if input_ids.contains(&opid) || accumulator_ids.contains(&opid) {
            let plan = static_plan::<CuteSm100PipelinePlanAttr>(ctx, op)
                .ok_or("SM100 pipeline plan missing")?;
            let target = if input_ids.contains(&opid) {
                &mut input_plan
            } else {
                &mut accumulator_plan
            };
            if target.is_some_and(|previous| previous != plan) {
                return Err("SM100 pipeline lifecycle plans disagree".into());
            }
            *target = Some(plan);
            if plan.transaction_bytes > 0 && Some(plan.transaction_bytes) != mma.transaction_bytes()
            {
                return Err(format!(
                    "SM100 pipeline promises {} bytes but the MMA operand tiles need {}",
                    plan.transaction_bytes,
                    mma.transaction_bytes().unwrap()
                ));
            }
            if plan.transaction_bytes == 0 && plan.consumer_arrivals != 8 {
                return Err(
                    "SM100 128x32 epilogue requires four consumer warps in each CTA (8 arrivals)"
                        .into(),
                );
            }
        }
        if opid == CuteSm100ClusterTmaLoadOp::get_opid_static() {
            copied_layouts.push(
                CuteSm100ClusterTmaLoadOp::wrap(op)
                    .plan(ctx)
                    .ok_or("SM100 copy plan missing")?
                    .layout,
            );
        }
        if [
            CuteSm100StoreCommitOp::get_opid_static(),
            CuteSm100StoreAcquireOp::get_opid_static(),
            CuteSm100StoreTailOp::get_opid_static(),
        ]
        .contains(&opid)
        {
            let plan = static_plan::<CuteTmaStorePipelineAttr>(ctx, op)
                .ok_or("SM100 store plan missing")?;
            if store_plan.is_some_and(|previous| previous != plan) {
                return Err("SM100 store lifecycle stage counts disagree".into());
            }
            store_plan = Some(plan);
        }
    }
    for required in [&mma.a_layout, &mma.b0_layout, &mma.b1_layout] {
        if !copied_layouts.contains(required) {
            return Err("SM100 MMA operand layout has no matching cluster TMA copy".into());
        }
    }
    Ok(())
}

/// Regression fixtures covering accepted contracts and rejected malformed IR.
#[cfg(test)]
mod tests {
    use super::*;
    use dialect_mir::ops::{MirConstantOp, MirUndefOp};
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::builtin::op_interfaces::{NOpdsInterface, NResultsInterface};
    use pliron::builtin::ops::ModuleOp;
    use pliron::builtin::types::Signedness;
    use pliron::linked_list::ContainsLinkedList;
    use pliron::r#type::TypeHandle;
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    // Match importer-produced NVVM op IDs without a production dependency.
    /// Test-only canonical two-CTA allocation recognized by the semantic lifecycle audit.
    #[pliron_op(name = "nvvm.tcgen05_alloc_cg2", format, verifier = "succ", interfaces = [NOpdsInterface<2>, NResultsInterface<0>])]
    struct TestAllocCg2Op;
    /// Test-only canonical two-CTA deallocation recognized by the lifecycle audit.
    #[pliron_op(name = "nvvm.tcgen05_dealloc_cg2", format, verifier = "succ", interfaces = [NOpdsInterface<2>, NResultsInterface<0>])]
    struct TestDeallocCg2Op;
    /// Test-only marker proving the two-CTA allocation permit was relinquished.
    #[pliron_op(name = "nvvm.tcgen05_relinquish_alloc_permit_cg2", format, verifier = "succ", interfaces = [NOpdsInterface<0>, NResultsInterface<0>])]
    struct TestRelinquishCg2Op;
    /// Test-only one-CTA allocation used to reject the wrong collective mode.
    #[pliron_op(name = "nvvm.tcgen05_alloc", format, verifier = "succ", interfaces = [NOpdsInterface<2>, NResultsInterface<0>])]
    struct TestAllocCg1Op;
    /// Test-only one-CTA deallocation used to reject a mismatched collective mode.
    #[pliron_op(name = "nvvm.tcgen05_dealloc", format, verifier = "succ", interfaces = [NOpdsInterface<2>, NResultsInterface<0>])]
    struct TestDeallocCg1Op;

    /// Append a direct unsigned 32-bit MIR literal for static-capacity verification tests.
    fn literal_u32(ctx: &mut Context, block: Ptr<BasicBlock>, value: u32) -> Value {
        let ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
        let op = Operation::new(
            ctx,
            MirConstantOp::get_concrete_op_info(),
            vec![ty.into()],
            vec![],
            vec![],
            0,
        );
        MirConstantOp::new(op).set_attr_value(
            ctx,
            IntegerAttr::new(ty, APInt::from_u32(value, NonZeroUsize::new(32).unwrap())),
        );
        op.insert_at_back(block, ctx);
        op.deref(ctx).get_result(0)
    }

    /// Append a test-only canonical CTM/NVVM operation and include it in the audited operation list.
    fn append_ctm_op<T: Op>(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        ops: &mut Vec<Ptr<Operation>>,
        operands: Vec<Value>,
    ) {
        let op = Operation::new(ctx, T::get_concrete_op_info(), vec![], operands, vec![], 0);
        op.insert_at_back(block, ctx);
        ops.push(op);
    }

    /// Replace fixture allocation markers with configurable CTM lifecycle sites for compatibility and failure cases.
    fn replace_cute_tmem_with_ctm(
        ctx: &mut Context,
        ops: &mut Vec<Ptr<Operation>>,
        columns: u32,
        group: u32,
        dealloc: bool,
        relinquish: bool,
    ) {
        let block = ops[0].deref(ctx).get_parent_block().unwrap();
        let shared = ops[0].deref(ctx).get_operand(0);
        let scalar = ops
            .iter()
            .find(|op| Operation::get_opid(**op, ctx) == CuteSm100TmemDeallocOp::get_opid_static())
            .unwrap()
            .deref(ctx)
            .get_operand(0);
        let is_cute_tmem = |op| {
            [
                CuteSm100TmemAllocOp::get_opid_static(),
                CuteSm100TmemDeallocOp::get_opid_static(),
            ]
            .contains(&Operation::get_opid(op, ctx))
        };
        for op in ops.iter().copied().filter(|op| is_cute_tmem(*op)) {
            op.unlink(ctx);
        }
        ops.retain(|op| !is_cute_tmem(*op));
        let count = literal_u32(ctx, block, columns);
        if group == 2 {
            append_ctm_op::<TestAllocCg2Op>(ctx, block, ops, vec![shared, count]);
            if relinquish {
                append_ctm_op::<TestRelinquishCg2Op>(ctx, block, ops, vec![]);
            }
            if dealloc {
                append_ctm_op::<TestDeallocCg2Op>(ctx, block, ops, vec![scalar, count]);
            }
        } else {
            append_ctm_op::<TestAllocCg1Op>(ctx, block, ops, vec![shared, count]);
            if dealloc {
                append_ctm_op::<TestDeallocCg1Op>(ctx, block, ops, vec![scalar, count]);
            }
        }
    }

    /// Build the supported paired M256/N192+160/K64 plan with canonical shared layouts.
    fn mma_plan() -> CuteSm100MmaPlanAttr {
        CuteSm100MmaPlanAttr {
            m: 256,
            n0: 192,
            n1: 160,
            k: 64,
            tmem_columns: 512,
            cta_group: 2,
            collector: CuteSm100CollectorAttr::AFillLastUse,
            a_layout: CuteComposedLayoutAttr(row_major(128, 64, 3)),
            b0_layout: CuteComposedLayoutAttr(row_major(96, 64, 3)),
            b1_layout: CuteComposedLayoutAttr(row_major(80, 64, 3)),
        }
    }
    /// Build an input or accumulator pipeline fixture from expected bytes and arrival count.
    fn pipeline(bytes: u32, arrivals: u32) -> CuteSm100PipelinePlanAttr {
        CuteSm100PipelinePlanAttr {
            stages: if bytes == 0 { 1 } else { 5 },
            transaction_bytes: bytes,
            consumer_arrivals: arrivals,
            cta_group: 2,
            leader_rank: 0,
            multicast_mask: 3,
        }
    }
    /// Create a typed placeholder SSA result so a verifier test needs no runtime input.
    fn undef(ctx: &mut Context, block: Ptr<BasicBlock>, ty: TypeHandle) -> Value {
        let op = MirUndefOp::new(ctx, ty).get_operation();
        op.insert_at_back(block, ctx);
        op.deref(ctx).get_result(0)
    }
    /// Construct one complete paired-SM100 lifecycle, returning the module and all semantic operation sites.
    fn complete_story(ctx: &mut Context, bytes: u32) -> (Ptr<Operation>, Vec<Ptr<Operation>>) {
        dialect_mir::register(ctx);
        crate::register(ctx);
        let module = ModuleOp::new(ctx, "sm100_test".try_into().unwrap()).get_operation();
        let block = module
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .iter(ctx)
            .next()
            .unwrap();
        let u32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
        let u8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
        let bool_ty: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
        let f32_ty: TypeHandle = FP32Type::get(ctx).into();
        let f16_ty: TypeHandle = MirFP16Type::get(ctx).into();
        let token_ty: TypeHandle = MirPtrType::get(ctx, u32_ty, true, address_space::SHARED).into();
        let barrier_ty: TypeHandle =
            MirPtrType::get(ctx, u64_ty, true, address_space::SHARED).into();
        let shared_ty: TypeHandle =
            MirPtrType::get(ctx, f16_ty, true, address_space::SHARED).into();
        let desc_ty: TypeHandle = MirPtrType::get_generic(ctx, u8_ty, false).into();
        let scalar = undef(ctx, block, u32_ty);
        let condition = undef(ctx, block, bool_ty);
        let bias = undef(ctx, block, f32_ty);
        let token = undef(ctx, block, token_ty);
        let full = undef(ctx, block, barrier_ty);
        let empty = undef(ctx, block, barrier_ty);
        let afull = undef(ctx, block, barrier_ty);
        let aempty = undef(ctx, block, barrier_ty);
        let shared = undef(ctx, block, shared_ty);
        let desc = undef(ctx, block, desc_ty);
        let mut ops = Vec::new();
        /// Append one typed semantic operation to the test block and collected operation list.
        macro_rules! append { ($ty:ident,[$($arg:expr),*],$plan:expr) => {{
            let operation=$ty::new(ctx,vec![$($arg),*],$plan).get_operation();
            operation.insert_at_back(block,ctx);ops.push(operation);
        }}; }
        let tmem = CuteSm100TmemPlanAttr {
            columns: 512,
            cta_group: 2,
        };
        let input = pipeline(bytes, 1);
        let acc = pipeline(0, 8);
        let epi = CuteSm100EpiloguePlanAttr {
            layout: CuteComposedLayoutAttr(row_major(128, 32, 2)),
            rows: 128,
            columns: 32,
        };
        append!(CuteSm100TmemAllocOp, [token], tmem);
        append!(CuteSm100PipelineInitOp, [full, empty], input);
        append!(CuteSm100AccumulatorInitOp, [aempty, afull], acc);
        append!(CuteSm100PipelineAcquireOp, [empty, scalar, scalar], input);
        append!(CuteSm100PipelineExpectOp, [full, scalar], input);
        for rows in [128, 96, 80] {
            append!(
                CuteSm100ClusterTmaLoadOp,
                [desc, shared, scalar, scalar, full, scalar],
                CuteSm100ClusterCopyPlanAttr {
                    layout: CuteComposedLayoutAttr(row_major(rows, 64, 3)),
                    cta_group: 2,
                    leader_rank: 0
                }
            );
        }
        append!(CuteSm100AccumulatorAcquireOp, [aempty, scalar], acc);
        append!(CuteSm100PipelineWaitOp, [full, scalar, scalar], input);
        append!(
            CuteSm100TiledMmaOp,
            [scalar, shared, shared, shared, condition],
            mma_plan()
        );
        append!(CuteSm100PipelineReleaseOp, [empty, scalar], input);
        append!(CuteSm100AccumulatorCommitOp, [afull], acc);
        append!(CuteSm100AccumulatorWaitOp, [afull, scalar], acc);
        append!(
            CuteSm100TmemEpilogueOp,
            [scalar, shared, scalar, bias],
            epi.clone()
        );
        append!(CuteSm100TmaStoreOp, [desc, shared, scalar, scalar], epi);
        append!(CuteSm100StoreCommitOp, [], CuteTmaStorePipelineAttr::new(2));
        append!(
            CuteSm100StoreAcquireOp,
            [],
            CuteTmaStorePipelineAttr::new(2)
        );
        append!(CuteSm100AccumulatorReleaseOp, [aempty], acc);
        append!(CuteSm100StoreTailOp, [], CuteTmaStorePipelineAttr::new(2));
        append!(CuteSm100TmemDeallocOp, [scalar], tmem);
        (module, ops)
    }

    /// The stage-byte promise includes A and both B partitions across both CTAs.
    #[test]
    fn paired_plan_derives_complete_two_cta_transaction_bytes() {
        let ctx = Context::new();
        let plan = mma_plan();
        plan.verify(&ctx).unwrap();
        assert_eq!(plan.transaction_bytes(), Some(77_824));
        let mut bad = plan.clone();
        bad.b0_layout = plan.b1_layout.clone();
        assert!(bad.verify(&ctx).is_err());
        let mut bad = plan.clone();
        bad.n1 = 176;
        assert!(bad.verify(&ctx).is_err());
        let mut bad = plan;
        bad.cta_group = 1;
        assert!(bad.verify(&ctx).is_err());
    }
    /// Unsupported allocation capacities and collective routing configurations fail static verification.
    #[test]
    fn tmem_and_cluster_collective_contracts_fail_closed() {
        let ctx = Context::new();
        assert!(
            CuteSm100TmemPlanAttr {
                columns: 48,
                cta_group: 2
            }
            .verify(&ctx)
            .is_err()
        );
        assert!(
            CuteSm100TmemPlanAttr {
                columns: 512,
                cta_group: 1
            }
            .verify(&ctx)
            .is_err()
        );
        let mut plan = pipeline(77_824, 1);
        plan.multicast_mask = 1;
        assert!(plan.verify(&ctx).is_err());
        let mut plan = pipeline(0, 8);
        plan.stages = 2;
        assert!(plan.verify(&ctx).is_err());
    }
    /// The backend-neutral entry point accepts a complete matching paired-SM100 lifecycle.
    #[test]
    fn complete_sm100_lifecycle_is_verified_by_shared_entrypoint() {
        let mut ctx = Context::new();
        let (module, _) = complete_story(&mut ctx, 77_824);
        crate::verify::verify_cute_semantics(&ctx, module).unwrap();
    }
    /// Canonical CTM two-CTA allocation/deallocation may supply storage for semantic CuTe MMA.
    #[test]
    fn accepts_ctm_group2_tmem_lifecycle_with_cute_mma() {
        let mut ctx = Context::new();
        let (module, mut ops) = complete_story(&mut ctx, 77_824);
        replace_cute_tmem_with_ctm(&mut ctx, &mut ops, 512, 2, true, true);
        verify_story(&ctx, &ops).unwrap();
        crate::verify::verify_cute_semantics(&ctx, module).unwrap();
    }
    /// A recognized allocation cannot omit the matching release of TMEM capacity.
    #[test]
    fn rejects_ctm_tmem_lifecycle_missing_deallocation() {
        let mut ctx = Context::new();
        let (_, mut ops) = complete_story(&mut ctx, 77_824);
        replace_cute_tmem_with_ctm(&mut ctx, &mut ops, 512, 2, false, true);
        let error = verify_story(&ctx, &ops).unwrap_err();
        assert!(error.contains("dealloc"), "{error}");
    }
    /// CTM allocation must relinquish its allocation permit before the lifecycle is accepted.
    #[test]
    fn rejects_ctm_tmem_lifecycle_missing_allocation_permit() {
        let mut ctx = Context::new();
        let (_, mut ops) = complete_story(&mut ctx, 77_824);
        replace_cute_tmem_with_ctm(&mut ctx, &mut ops, 512, 2, true, false);
        let error = verify_story(&ctx, &ops).unwrap_err();
        assert!(error.contains("permit"), "{error}");
    }
    /// One-CTA allocation/deallocation cannot satisfy a two-CTA MMA storage contract.
    #[test]
    fn rejects_ctm_tmem_lifecycle_with_group1_operations() {
        let mut ctx = Context::new();
        let (_, mut ops) = complete_story(&mut ctx, 77_824);
        replace_cute_tmem_with_ctm(&mut ctx, &mut ops, 512, 1, true, false);
        let error = verify_story(&ctx, &ops).unwrap_err();
        assert!(error.contains("CTA group 2"), "{error}");
    }
    /// A directly known CTM column count must match the paired MMA's TMEM capacity.
    #[test]
    fn rejects_ctm_tmem_lifecycle_with_known_wrong_capacity() {
        let mut ctx = Context::new();
        let (_, mut ops) = complete_story(&mut ctx, 77_824);
        replace_cute_tmem_with_ctm(&mut ctx, &mut ops, 256, 2, true, true);
        let error = verify_story(&ctx, &ops).unwrap_err();
        assert!(error.contains("capacity"), "{error}");
    }
    /// The input barrier must expect both CTAs' copies, not only the local tile bytes.
    #[test]
    fn rejects_only_one_ctas_expected_transaction_bytes() {
        let mut ctx = Context::new();
        let (module, _) = complete_story(&mut ctx, 38_912);
        let error = crate::verify::verify_cute_semantics(&ctx, module)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("promises 38912 bytes") && error.contains("77824"),
            "{error}"
        );
    }
    /// Every shared MMA operand placement must have a corresponding cluster input-copy layout.
    #[test]
    fn rejects_mma_input_layout_without_matching_cluster_copy() {
        let mut ctx = Context::new();
        let (_, mut ops) = complete_story(&mut ctx, 77_824);
        ops.retain(|op| {
            Operation::get_opid(*op, &ctx) != CuteSm100ClusterTmaLoadOp::get_opid_static()
        });
        assert!(
            verify_story(&ctx, &ops)
                .unwrap_err()
                .contains("no matching cluster TMA copy")
        );
    }
    /// The epilogue must release its accumulator slot before the paired lifecycle is complete.
    #[test]
    fn rejects_missing_accumulator_release() {
        let mut ctx = Context::new();
        let (_, mut ops) = complete_story(&mut ctx, 77_824);
        ops.retain(|op| {
            Operation::get_opid(*op, &ctx) != CuteSm100AccumulatorReleaseOp::get_opid_static()
        });
        assert!(
            verify_story(&ctx, &ops)
                .unwrap_err()
                .contains("sm100_accumulator_release")
        );
    }
    /// Malformed positional scalar types are rejected by the shared verifier before backend emission.
    #[test]
    fn rejects_bad_scalar_carriers_before_backend_lowering() {
        let mut ctx = Context::new();
        let (_, ops) = complete_story(&mut ctx, 77_824);
        let mma = ops
            .iter()
            .find(|op| Operation::get_opid(**op, &ctx) == CuteSm100TiledMmaOp::get_opid_static())
            .copied()
            .unwrap();
        let scalar = mma.deref(&ctx).get_operand(0);
        let malformed = CuteSm100TiledMmaOp::new(&mut ctx, vec![scalar; 5], mma_plan());
        assert!(malformed.verify(&ctx).is_err());
        let malformed = CuteSm100TmemAllocOp::new(
            &mut ctx,
            vec![],
            CuteSm100TmemPlanAttr {
                columns: 512,
                cta_group: 2,
            },
        );
        assert!(malformed.verify(&ctx).is_err());
    }
}
