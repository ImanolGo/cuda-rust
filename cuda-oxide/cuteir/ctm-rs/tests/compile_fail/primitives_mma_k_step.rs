/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#![allow(unused_imports)]
use ctm_rs::{groups::{CtaGroup1,CtaGroup2},memory::*,tcgen05::{descriptors::*,mma::*,tmem::*}};
type L=DenseSharedF16<CtaGroup2,256,192,KMajor,KMajor>;
type R=DenseSharedF16<CtaGroup2,256,160,KMajor,KMajor>;
fn wrong(a:&SmemOperand<OperandA,F16Tile128>) { let _ = a.k16::<4>(); }
fn main() {}
