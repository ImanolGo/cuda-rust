/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#![allow(unused_imports)]
use cuda_device::primitives::{groups::{CtaGroup1,CtaGroup2},memory::*,tcgen05::{descriptors::*,mma::*,tmem::*}};
type L=DenseSharedF16<CtaGroup2,256,192,KMajor,KMajor>;
type R=DenseSharedF16<CtaGroup2,256,160,KMajor,KMajor>;
fn wrong(d:&impl TmemOperand<Group=CtaGroup2,Format=TmemF32<256,192>>,a:&SmemOperand<OperandA,F16Tile128>,b:&SmemOperand<OperandB,F16TileB96>) {
 unsafe { mma::<L,collector_a::Discard>(d,a,b,&InstructionDescriptor::<R>::new(),false) };
}
fn main() {}
