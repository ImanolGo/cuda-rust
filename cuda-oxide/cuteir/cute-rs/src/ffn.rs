/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Checked host arguments for BF16 feed-forward compositions.
//!
//! These types describe existing allocations; they neither allocate nor launch
//! work. Matrix dimensions and pitches are checked once at construction. Weight
//! roles and projection dimensions remain in Rust types when composing an FFN.
//! All weights retain checkpoint orientation: `[output, input]`, with contiguous
//! input coordinates within each row. No implicit transpose or conversion occurs.
//!
//! Assigning a role asserts the meaning of the supplied weights. Shapes cannot
//! establish that an importer labeled their contents correctly. Buffer context,
//! stream ordering, kernel participation and disjoint output writes remain launch
//! obligations; these argument types do not prove them.
//!
//! ```
//! use cuda_core::DeviceBuffer;
//! use cute_rs::Bf16;
//! use cute_rs::ffn::{Down, FfnError, FfnWeights, Gate, ProjectionWeights, Up};
//!
//! fn checked<'a>(
//!     gate: &'a DeviceBuffer<Bf16>,
//!     up: &'a DeviceBuffer<Bf16>,
//!     down: &'a DeviceBuffer<Bf16>,
//! ) -> Result<FfnWeights<'a, 1024, 3072>, FfnError> {
//!     Ok(FfnWeights::new(
//!         ProjectionWeights::<Gate, 1024, 3072>::new(gate, 1024)?,
//!         ProjectionWeights::<Up, 1024, 3072>::new(up, 1024)?,
//!         ProjectionWeights::<Down, 3072, 1024>::new(down, 3072)?,
//!     ))
//! }
//! ```

use core::{fmt, marker::PhantomData, mem::size_of};
use cuda_core::DeviceBuffer;

use crate::Bf16;

/// A rejected BF16 matrix shape or allocation extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnError {
    /// A matrix width or projection dimension is zero.
    ZeroWidth,
    /// Consecutive rows overlap because their pitch is smaller than the width.
    StrideTooSmall,
    /// An element calculation overflowed or the BF16 byte extent exceeds `isize::MAX`.
    ExtentOverflow,
    /// The allocation does not cover every logical matrix coordinate.
    BufferTooShort,
}

impl fmt::Display for FfnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ZeroWidth => "FFN matrix widths and projection dimensions must be positive",
            Self::StrideTooSmall => "matrix row stride is smaller than its column count",
            Self::ExtentOverflow => {
                "BF16 matrix or slice byte extent exceeds the addressable range"
            }
            Self::BufferTooShort => "buffer does not cover the complete matrix extent",
        })
    }
}

impl core::error::Error for FfnError {}

/// Validated row-major BF16 coordinates, with the pitch measured in elements.
///
/// The minimum allocation ends at the last logical element: a `3 x 5` matrix
/// with pitch `8` requires `21` elements. Extra trailing padding is permitted.
/// Empty row sets require no elements, but their width and pitch remain valid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowMajorShape {
    /// Number of logical rows.
    rows: usize,
    /// Number of logical columns in every row.
    cols: usize,
    /// Distance in BF16 elements between consecutive row starts.
    row_stride: usize,
    /// Minimum buffer extent including padding between rows.
    required_len: usize,
}

impl RowMajorShape {
    /// Check a row-major BF16 shape without allocating any memory.
    ///
    /// Rejects zero width, overlapping rows, overflowing element offsets and
    /// byte extents larger than `isize::MAX`, as required by Rust slice indexing.
    pub fn new(rows: usize, cols: usize, row_stride: usize) -> Result<Self, FfnError> {
        if cols == 0 {
            return Err(FfnError::ZeroWidth);
        }
        if row_stride < cols {
            return Err(FfnError::StrideTooSmall);
        }
        let required_len = if rows == 0 {
            0
        } else {
            (rows - 1)
                .checked_mul(row_stride)
                .and_then(|last_row| last_row.checked_add(cols))
                .ok_or(FfnError::ExtentOverflow)?
        };
        check_byte_extent(required_len)?;
        Ok(Self {
            rows,
            cols,
            row_stride,
            required_len,
        })
    }

    /// Check a row-major BF16 shape whose rows have no padding.
    pub fn dense(rows: usize, cols: usize) -> Result<Self, FfnError> {
        Self::new(rows, cols, cols)
    }

    /// Number of logical rows.
    #[must_use]
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Number of logical columns in each row.
    #[must_use]
    pub const fn cols(self) -> usize {
        self.cols
    }

    /// Distance in BF16 elements between consecutive row starts.
    #[must_use]
    pub const fn row_stride(self) -> usize {
        self.row_stride
    }

    /// Minimum number of BF16 elements covering every logical coordinate.
    #[must_use]
    pub const fn required_len(self) -> usize {
        self.required_len
    }

    /// Check an existing buffer's element count, permitting trailing padding.
    ///
    /// The full slice extent, including any trailing padding, must fit within
    /// `isize::MAX` bytes. This check does not establish buffer context or lifetime.
    pub fn validate_len(self, len: usize) -> Result<(), FfnError> {
        check_byte_extent(len)?;
        if len < self.required_len {
            Err(FfnError::BufferTooShort)
        } else {
            Ok(())
        }
    }
}

/// Reject BF16 buffers whose complete byte extent cannot be represented by a Rust slice.
fn check_byte_extent(len: usize) -> Result<(), FfnError> {
    match len.checked_mul(size_of::<Bf16>()) {
        Some(bytes) if bytes <= isize::MAX as usize => Ok(()),
        _ => Err(FfnError::ExtentOverflow),
    }
}

/// Private supertrait prevents unreviewed projection roles.
mod sealed {
    /// Marks the supported FFN weight roles.
    pub trait Sealed {}
}

/// One of the distinct gate, up or down projection roles.
pub trait ProjectionRole: sealed::Sealed {}

/// Weights producing the gated branch before SiLU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gate;

/// Weights producing the branch multiplied by the activated gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Up;

/// Weights projecting the multiplied intermediate values back to hidden width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Down;

impl sealed::Sealed for Gate {}
impl sealed::Sealed for Up {}
impl sealed::Sealed for Down {}
impl ProjectionRole for Gate {}
impl ProjectionRole for Up {}
impl ProjectionRole for Down {}

/// Borrowed BF16 projection weights with a checked `[OUTPUT, INPUT]` layout.
///
/// The role and dimensions prevent incompatible weights from being passed to a
/// checked FFN composition. The owning [`DeviceBuffer`] retains allocation and
/// context lifetime; this wrapper never assumes ownership or converts its data.
///
/// An FP32 allocation is not BF16 storage:
///
/// ```compile_fail,E0308
/// use cuda_core::DeviceBuffer;
/// use cute_rs::ffn::{Gate, ProjectionWeights};
/// fn wrong_dtype(weights: &DeviceBuffer<f32>) {
///     let _ = ProjectionWeights::<Gate, 8, 16>::new(weights, 8);
/// }
/// ```
///
/// The weight borrow prevents mutable access while the checked view is used:
///
/// ```compile_fail,E0502
/// use cuda_core::DeviceBuffer;
/// use cute_rs::Bf16;
/// use cute_rs::ffn::{Gate, ProjectionWeights};
/// fn exclusive_use(_: &mut DeviceBuffer<Bf16>) {}
/// fn alias(weights: &mut DeviceBuffer<Bf16>) {
///     let gate = ProjectionWeights::<Gate, 8, 16>::new(weights, 8).unwrap();
///     exclusive_use(weights);
///     let _ = gate.buffer();
/// }
/// ```
pub struct ProjectionWeights<'a, Role: ProjectionRole, const INPUT: usize, const OUTPUT: usize> {
    /// The existing, immutably borrowed allocation.
    buffer: &'a DeviceBuffer<Bf16>,
    /// Checked checkpoint orientation and row pitch.
    shape: RowMajorShape,
    /// Distinguishes equally shaped gate and up weights.
    role: PhantomData<Role>,
}

impl<'a, Role: ProjectionRole, const INPUT: usize, const OUTPUT: usize>
    ProjectionWeights<'a, Role, INPUT, OUTPUT>
{
    /// Borrow weights in checkpoint orientation, with input coordinates contiguous.
    ///
    /// Both projection dimensions must be positive. The caller explicitly labels
    /// the role; this validates dimensions and capacity, not semantic contents.
    pub fn new(buffer: &'a DeviceBuffer<Bf16>, row_stride: usize) -> Result<Self, FfnError> {
        if OUTPUT == 0 {
            return Err(FfnError::ZeroWidth);
        }
        let shape = RowMajorShape::new(OUTPUT, INPUT, row_stride)?;
        shape.validate_len(buffer.len())?;
        Ok(Self {
            buffer,
            shape,
            role: PhantomData,
        })
    }

    /// Borrow the existing allocation for a checked kernel launch.
    #[must_use]
    pub const fn buffer(&self) -> &'a DeviceBuffer<Bf16> {
        self.buffer
    }

    /// The checked `[OUTPUT, INPUT]` shape and its physical row pitch.
    #[must_use]
    pub const fn shape(&self) -> RowMajorShape {
        self.shape
    }
}

/// Gate/up/down weights whose roles and dimensions agree for one FFN.
///
/// Equal dimensions do not make gate and up weights interchangeable:
///
/// ```compile_fail,E0308
/// use cute_rs::ffn::{Down, FfnWeights, Gate, ProjectionWeights, Up};
/// fn swapped(
///     gate: ProjectionWeights<'_, Gate, 8, 16>,
///     up: ProjectionWeights<'_, Up, 8, 16>,
///     down: ProjectionWeights<'_, Down, 16, 8>,
/// ) {
///     let _ = FfnWeights::<8, 16>::new(up, gate, down);
/// }
/// ```
///
/// The down projection must reverse the hidden/intermediate dimensions:
///
/// ```compile_fail,E0308
/// use cute_rs::ffn::{Down, FfnWeights, Gate, ProjectionWeights, Up};
/// fn wrong_down_orientation(
///     gate: ProjectionWeights<'_, Gate, 8, 16>,
///     up: ProjectionWeights<'_, Up, 8, 16>,
///     down: ProjectionWeights<'_, Down, 8, 16>,
/// ) {
///     let _ = FfnWeights::<8, 16>::new(gate, up, down);
/// }
/// ```
pub struct FfnWeights<'a, const HIDDEN: usize, const INTERMEDIATE: usize> {
    /// Gate projection from hidden to intermediate width.
    gate: ProjectionWeights<'a, Gate, HIDDEN, INTERMEDIATE>,
    /// Up projection from hidden to intermediate width.
    up: ProjectionWeights<'a, Up, HIDDEN, INTERMEDIATE>,
    /// Down projection from intermediate to hidden width.
    down: ProjectionWeights<'a, Down, INTERMEDIATE, HIDDEN>,
}

impl<'a, const HIDDEN: usize, const INTERMEDIATE: usize> FfnWeights<'a, HIDDEN, INTERMEDIATE> {
    /// Compose previously checked weights; Rust verifies their roles and dimensions.
    #[must_use]
    pub const fn new(
        gate: ProjectionWeights<'a, Gate, HIDDEN, INTERMEDIATE>,
        up: ProjectionWeights<'a, Up, HIDDEN, INTERMEDIATE>,
        down: ProjectionWeights<'a, Down, INTERMEDIATE, HIDDEN>,
    ) -> Self {
        Self { gate, up, down }
    }

    /// Weights for the gate projection.
    #[must_use]
    pub const fn gate(&self) -> &ProjectionWeights<'a, Gate, HIDDEN, INTERMEDIATE> {
        &self.gate
    }

    /// Weights for the up projection.
    #[must_use]
    pub const fn up(&self) -> &ProjectionWeights<'a, Up, HIDDEN, INTERMEDIATE> {
        &self.up
    }

    /// Weights for the down projection.
    #[must_use]
    pub const fn down(&self) -> &ProjectionWeights<'a, Down, INTERMEDIATE, HIDDEN> {
        &self.down
    }
}

/// Checked input and workspace extents for fixed hidden/intermediate widths.
///
/// Input rows may be padded. Hidden-width outputs and intermediate workspace are
/// dense. Empty row sets remain valid so callers can skip all allocations/launches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FfnShape<const HIDDEN: usize, const INTERMEDIATE: usize> {
    /// The input matrix, including its caller-selected pitch.
    input: RowMajorShape,
    /// Dense hidden-width output or normalization workspace.
    hidden: RowMajorShape,
    /// Dense intermediate-width projection/activation workspace.
    intermediate: RowMajorShape,
}

impl<const HIDDEN: usize, const INTERMEDIATE: usize> FfnShape<HIDDEN, INTERMEDIATE> {
    /// Check all input and workspace extents before allocation.
    pub fn new(rows: usize, input_stride: usize) -> Result<Self, FfnError> {
        Ok(Self {
            input: RowMajorShape::new(rows, HIDDEN, input_stride)?,
            hidden: RowMajorShape::dense(rows, HIDDEN)?,
            intermediate: RowMajorShape::dense(rows, INTERMEDIATE)?,
        })
    }

    /// Number of logical token rows in every activation matrix.
    #[must_use]
    pub const fn rows(self) -> usize {
        self.input.rows()
    }

    /// Checked input shape, including any row padding.
    #[must_use]
    pub const fn input(self) -> RowMajorShape {
        self.input
    }

    /// Dense shape for hidden-width outputs and normalization workspace.
    #[must_use]
    pub const fn hidden(self) -> RowMajorShape {
        self.hidden
    }

    /// Dense shape for projection outputs and multiplied activations.
    #[must_use]
    pub const fn intermediate(self) -> RowMajorShape {
        self.intermediate
    }
}
