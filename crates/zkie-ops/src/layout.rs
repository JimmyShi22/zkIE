//! Deterministic reshape / 2-D transpose.

use zkie_core::error::{Error, Result};
use zkie_core::types::{DType, TensorSpec};

use crate::op::OpCode;
use crate::registry::{io_count, require_len, spec_1d, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutKind {
    Reshape,
    Transpose2d,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayoutOp {
    pub kind: LayoutKind,
    pub from: Vec<u64>,
    pub to: Vec<u64>,
}

impl LayoutOp {
    pub fn reshape(from: Vec<u64>, to: Vec<u64>) -> Self {
        Self {
            kind: LayoutKind::Reshape,
            from,
            to,
        }
    }

    pub fn transpose(rows: u64, cols: u64) -> Self {
        Self {
            kind: LayoutKind::Transpose2d,
            from: vec![rows, cols],
            to: vec![cols, rows],
        }
    }
}

fn product(shape: &[u64]) -> Result<u64> {
    shape
        .iter()
        .try_fold(1u64, |accumulator, dim| accumulator.checked_mul(*dim))
        .ok_or(Error::ResourceLimit)
}

impl PureForward for LayoutOp {
    fn opcode(&self) -> OpCode {
        OpCode::Layout
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if inputs.len() != 1 {
            return Err(Error::ShapeMismatch);
        }
        if inputs[0].element_count()? != product(&self.to)? {
            return Err(Error::ShapeMismatch);
        }
        Ok(vec![TensorSpec::new(DType::I64, self.to.clone())])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        let values = inputs.first().ok_or(Error::ShapeMismatch)?;
        let count = io_count(&TensorSpec::new(DType::I64, self.to.clone()))?;
        require_len(values, count)?;
        let output = match self.kind {
            LayoutKind::Reshape => values.clone(),
            LayoutKind::Transpose2d => {
                let rows = self.from[0] as usize;
                let cols = self.from[1] as usize;
                let mut out = vec![0i64; count];
                for row in 0..rows {
                    for column in 0..cols {
                        out[column * rows + row] = values[row * cols + column];
                    }
                }
                out
            }
        };
        let _ = spec_1d(count);
        Ok(KernelRecord::with_output(output))
    }
}
