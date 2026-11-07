//! Index generation: clamped offset indexing and embedding index composition.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, to_i64_checked, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableIndexOp {
    pub offset: i64,
    pub table_len: usize,
}

impl PureForward for TableIndexOp {
    fn opcode(&self) -> OpCode {
        OpCode::TableIndex
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        let input = inputs.first().ok_or(Error::ShapeMismatch)?;
        Ok(vec![crate::registry::spec_1d(io_count(input)?)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        let values = inputs.first().ok_or(Error::ShapeMismatch)?;
        if self.table_len == 0 {
            return Err(Error::InvalidConfig);
        }
        let output = values
            .iter()
            .map(|value| {
                value
                    .saturating_add(self.offset)
                    .clamp(0, self.table_len as i64 - 1)
            })
            .collect();
        Ok(KernelRecord::with_output(output))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbeddingIndexOp {
    pub hidden: usize,
    pub vocab: usize,
}

impl PureForward for EmbeddingIndexOp {
    fn opcode(&self) -> OpCode {
        OpCode::EmbeddingIndex
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        let input = inputs.first().ok_or(Error::ShapeMismatch)?;
        let rows = io_count(input)?;
        Ok(vec![crate::registry::spec_1d(rows * self.hidden)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        let tokens = inputs.first().ok_or(Error::ShapeMismatch)?;
        if self.hidden == 0 || self.vocab == 0 {
            return Err(Error::InvalidConfig);
        }
        let mut indices = Vec::with_capacity(tokens.len() * self.hidden);
        for token in tokens {
            if *token < 0 || *token as usize >= self.vocab {
                return Err(Error::InvalidIndex);
            }
            for dimension in 0..self.hidden {
                indices.push(to_i64_checked(
                    (dimension * self.vocab) as i128 + *token as i128,
                )?);
            }
        }
        Ok(KernelRecord::with_output(indices))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowEmbeddingIndexOp {
    pub hidden: usize,
    pub vocab: usize,
}

impl PureForward for RowEmbeddingIndexOp {
    fn opcode(&self) -> OpCode {
        OpCode::RowEmbeddingIndex
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        let input = inputs.first().ok_or(Error::ShapeMismatch)?;
        let rows = io_count(input)?;
        Ok(vec![crate::registry::spec_1d(rows * self.hidden)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        let tokens = inputs.first().ok_or(Error::ShapeMismatch)?;
        if self.hidden == 0 || self.vocab == 0 {
            return Err(Error::InvalidConfig);
        }
        let mut indices = Vec::with_capacity(tokens.len() * self.hidden);
        for token in tokens {
            if *token < 0 || *token as usize >= self.vocab {
                return Err(Error::InvalidIndex);
            }
            for dimension in 0..self.hidden {
                indices.push(to_i64_checked(*token as i128 * self.hidden as i128 + dimension as i128)?);
            }
        }
        Ok(KernelRecord::with_output(indices))
    }
}
