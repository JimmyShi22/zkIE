//! Logical tensor, storage and buffer identity types.
//!
//! Logical identity is independent of where the bytes live. Every shape,
//! element offset and byte offset is checked; there is no implicit
//! "materialize the whole tensor" path.

use crate::common::codec::{Canonical, DecodeLimits};
use crate::common::field::FieldDescriptor;
use crate::error::{Error, Result};

/// A content digest. The width is fixed so canonical encoding is unambiguous.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub struct Digest(pub [u8; 32]);

impl Digest {
    pub const ZERO: Self = Self([0u8; 32]);

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl AsRef<[u8]> for Digest {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl Canonical for Digest {
    fn encode(&self, out: &mut Vec<u8>) -> Result<()> {
        out.extend_from_slice(&self.0);
        Ok(())
    }

    fn decode(input: &mut &[u8], limits: &DecodeLimits) -> Result<Self> {
        limits.check_bytes(32)?;
        if input.len() < 32 {
            return Err(Error::InvalidEncoding);
        }
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&input[..32]);
        *input = &input[32..];
        Ok(Self(bytes))
    }
}

/// Element type of a logical tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    I32,
    I64,
    Bytes,
    BaseField(&'static FieldDescriptor),
    ExtensionField(&'static FieldDescriptor),
}

impl DType {
    pub const fn element_size(self) -> u64 {
        match self {
            DType::I32 => 4,
            DType::I64 => 8,
            DType::Bytes => 1,
            DType::BaseField(_) => 8,
            DType::ExtensionField(descriptor) => {
                8u64.saturating_mul(descriptor.extension_degree as u64)
            }
        }
    }

    pub const fn is_integer(self) -> bool {
        matches!(self, DType::I32 | DType::I64)
    }

    pub const fn is_field(self) -> bool {
        matches!(self, DType::BaseField(_) | DType::ExtensionField(_))
    }

    pub const fn field_descriptor(self) -> Option<&'static FieldDescriptor> {
        match self {
            DType::BaseField(descriptor) | DType::ExtensionField(descriptor) => Some(descriptor),
            _ => None,
        }
    }
}

/// Physical layout of a logical tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutSpec {
    RowMajor,
    ColumnMajor,
    Strided(Vec<u64>),
}

impl LayoutSpec {
    pub const fn row_major() -> Self {
        Self::RowMajor
    }
}

/// Rounding rule. The rule is part of the mathematical specification, never a
/// host-language default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundRule {
    Floor,
    NearestTiesPositive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignedInterval {
    pub lo: i128,
    pub hi: i128,
}

impl SignedInterval {
    pub const fn new(lo: i128, hi: i128) -> Self {
        Self { lo, hi }
    }

    pub const fn contains(self, value: i128) -> bool {
        value >= self.lo && value <= self.hi
    }
}

/// Fixed-point interpretation of an integer tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedSpec {
    pub scale: i32,
    pub signed_bound: u128,
    pub round: RoundRule,
    pub clamp: Option<SignedInterval>,
}

impl FixedSpec {
    pub const fn new(scale: i32, signed_bound: u128, round: RoundRule) -> Self {
        Self {
            scale,
            signed_bound,
            round,
            clamp: None,
        }
    }
}

/// Padding metadata. `valid_len = None` means the whole tensor is valid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PaddingSpec {
    pub value: i128,
    pub valid_len: Option<u64>,
    pub mask_digest: Option<Digest>,
}

impl PaddingSpec {
    pub const fn none() -> Self {
        Self {
            value: 0,
            valid_len: None,
            mask_digest: None,
        }
    }
}

/// Full specification of a logical tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub layout: LayoutSpec,
    pub fixed: Option<FixedSpec>,
    pub padding: PaddingSpec,
}

impl TensorSpec {
    pub fn new(dtype: DType, shape: Vec<u64>) -> Self {
        Self {
            dtype,
            shape,
            layout: LayoutSpec::RowMajor,
            fixed: None,
            padding: PaddingSpec::none(),
        }
    }

    /// Checked product of the logical shape. A scalar tensor has one element.
    pub fn element_count(&self) -> Result<u64> {
        self.shape
            .iter()
            .try_fold(1u64, |acc, &dim| acc.checked_mul(dim))
            .ok_or(Error::ShapeMismatch)
    }

    /// Checked byte length of the logical tensor.
    pub fn byte_len(&self) -> Result<u64> {
        self.element_count()?
            .checked_mul(self.dtype.element_size())
            .ok_or(Error::ShapeMismatch)
    }

    pub fn validate(&self) -> Result<()> {
        let elements = self.element_count()?;
        self.byte_len()?;
        if let LayoutSpec::Strided(strides) = &self.layout {
            if strides.len() != self.shape.len() {
                return Err(Error::ShapeMismatch);
            }
        }
        if let Some(valid_len) = self.padding.valid_len {
            if valid_len > elements {
                return Err(Error::ShapeMismatch);
            }
        }
        Ok(())
    }

    /// Checked logical-to-linear offset for the declared layout.
    pub fn element_offset(&self, index: &[u64]) -> Result<u64> {
        if index.len() != self.shape.len() {
            return Err(Error::ShapeMismatch);
        }
        for (axis, &value) in index.iter().enumerate() {
            if value >= self.shape[axis] {
                return Err(Error::InvalidIndex);
            }
        }
        let offset = match &self.layout {
            LayoutSpec::RowMajor => {
                let mut acc = 0u64;
                for (axis, &value) in index.iter().enumerate() {
                    acc = acc
                        .checked_mul(self.shape[axis])
                        .and_then(|acc| acc.checked_add(value))
                        .ok_or(Error::InvalidIndex)?;
                }
                acc
            }
            LayoutSpec::ColumnMajor => {
                let mut acc = 0u64;
                let mut stride = 1u64;
                for axis in (0..index.len()).rev() {
                    let term = index[axis]
                        .checked_mul(stride)
                        .ok_or(Error::InvalidIndex)?;
                    acc = acc.checked_add(term).ok_or(Error::InvalidIndex)?;
                    stride = stride
                        .checked_mul(self.shape[axis])
                        .ok_or(Error::InvalidIndex)?;
                }
                acc
            }
            LayoutSpec::Strided(strides) => {
                if strides.len() != index.len() {
                    return Err(Error::ShapeMismatch);
                }
                let mut acc = 0u64;
                for (axis, &value) in index.iter().enumerate() {
                    let term = value
                        .checked_mul(strides[axis])
                        .ok_or(Error::InvalidIndex)?;
                    acc = acc.checked_add(term).ok_or(Error::InvalidIndex)?;
                }
                acc
            }
        };
        Ok(offset)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MappedTensorId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScratchTensorId(pub u64);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ElementRange {
    pub start: u64,
    pub len: u64,
}

impl ElementRange {
    pub const fn new(start: u64, len: u64) -> Self {
        Self { start, len }
    }

    pub fn end(&self) -> Result<u64> {
        self.start
            .checked_add(self.len)
            .ok_or(Error::InvalidIndex)
    }

    pub fn validate(&self, spec: &TensorSpec) -> Result<()> {
        if self.end()? > spec.element_count()? {
            return Err(Error::InvalidIndex);
        }
        Ok(())
    }

    pub fn byte_range(&self, spec: &TensorSpec) -> Result<(u64, u64)> {
        let element_size = spec.dtype.element_size();
        let start = self
            .start
            .checked_mul(element_size)
            .ok_or(Error::InvalidIndex)?;
        let len = self
            .len
            .checked_mul(element_size)
            .ok_or(Error::InvalidIndex)?;
        Ok((start, len))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferOwner(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferGeneration(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AllocationId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferSpec {
    pub dtype: DType,
    pub element_count: u64,
}

impl BufferSpec {
    pub const fn new(dtype: DType, element_count: u64) -> Self {
        Self {
            dtype,
            element_count,
        }
    }

    pub fn byte_len(&self) -> Result<u64> {
        self.element_count
            .checked_mul(self.dtype.element_size())
            .ok_or(Error::ShapeMismatch)
    }
}

/// Handle to a resident buffer. The handle is only valid for the owner and
/// generation that produced it; it is not a pointer and never enters proof
/// math.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferHandle {
    pub owner: BufferOwner,
    pub generation: BufferGeneration,
    pub allocation_id: AllocationId,
    pub dtype: DType,
    pub element_count: u64,
}

impl BufferHandle {
    pub const fn new(
        owner: BufferOwner,
        generation: BufferGeneration,
        allocation_id: AllocationId,
        dtype: DType,
        element_count: u64,
    ) -> Self {
        Self {
            owner,
            generation,
            allocation_id,
            dtype,
            element_count,
        }
    }

    pub fn check_owner(&self, expected: BufferOwner) -> Result<()> {
        if self.owner == expected {
            Ok(())
        } else {
            Err(Error::StatementMismatch)
        }
    }

    pub fn check_generation(&self, expected: BufferGeneration) -> Result<()> {
        if self.generation == expected {
            Ok(())
        } else {
            Err(Error::DeviceLost)
        }
    }

    pub fn check_dtype(&self, expected: DType) -> Result<()> {
        if self.dtype == expected {
            Ok(())
        } else {
            Err(Error::ShapeMismatch)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fence {
    pub owner: BufferOwner,
    pub event_id: u64,
}

impl Fence {
    pub const fn new(owner: BufferOwner, event_id: u64) -> Self {
        Self { owner, event_id }
    }

    pub fn check_owner(&self, expected: BufferOwner) -> Result<()> {
        if self.owner == expected {
            Ok(())
        } else {
            Err(Error::StatementMismatch)
        }
    }
}

/// Where a tensor's bytes live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorStorageRef {
    Resident(BufferHandle),
    Mapped(MappedTensorId),
    Scratch(ScratchTensorId),
}

/// A logical view over a tensor range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorView {
    pub tensor_id: TensorId,
    pub spec: TensorSpec,
    pub storage: TensorStorageRef,
    pub element_range: ElementRange,
}

impl TensorView {
    pub fn validate(&self) -> Result<()> {
        self.spec.validate()?;
        self.element_range.validate(&self.spec)
    }
}

/// Per-task workspace declaration. The minimum must fit inside the selected
/// budget; the execution layer acquires a permit before touching bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkspaceRequirement {
    pub minimum_resident_bytes: u64,
    pub selected_resident_bytes: u64,
    pub mapped_window_bytes: u64,
    pub scratch_disk_bytes: u64,
}

impl WorkspaceRequirement {
    pub fn validate(&self) -> Result<()> {
        if self.minimum_resident_bytes > self.selected_resident_bytes {
            return Err(Error::ResourceLimit);
        }
        Ok(())
    }
}
