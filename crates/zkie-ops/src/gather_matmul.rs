//! Gather rows then multiply by a weight matrix.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, requantize, to_i64_checked, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatherMatMulOp {
    pub cols: usize,
    pub n: usize,
    pub shift: u32,
}

impl PureForward for GatherMatMulOp {
    fn opcode(&self) -> OpCode {
        OpCode::GatherMatMul
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if inputs.len() != 3 {
            return Err(Error::ShapeMismatch);
        }
        let rows = io_count(&inputs[1])?;
        if io_count(&inputs[0])? % self.cols != 0 || io_count(&inputs[2])? != self.cols * self.n {
            return Err(Error::ShapeMismatch);
        }
        Ok(vec![crate::registry::spec_1d(rows * self.n)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        if inputs.len() != 3 {
            return Err(Error::ShapeMismatch);
        }
        let table = &inputs[0];
        let indices = &inputs[1];
        let weights = &inputs[2];
        let table_rows = table.len() / self.cols;
        let mut output = Vec::with_capacity(indices.len() * self.n);
        let mut raw_all = Vec::with_capacity(indices.len() * self.n);
        for index in indices {
            if *index < 0 || *index as usize >= table_rows {
                return Err(Error::InvalidIndex);
            }
            let row = &table[*index as usize * self.cols..(*index as usize + 1) * self.cols];
            for column in 0..self.n {
                let mut raw = 0i128;
                for inner in 0..self.cols {
                    raw = raw
                        .checked_add((row[inner] as i128) * (weights[inner * self.n + column] as i128))
                        .ok_or(Error::BoundViolation)?;
                }
                raw_all.push(raw);
                output.push(to_i64_checked(requantize(raw, self.shift)?)?);
            }
        }
        Ok(KernelRecord {
            outputs: vec![output],
            raw_accumulators: vec![raw_all],
            ..KernelRecord::default()
        })
    }
}
