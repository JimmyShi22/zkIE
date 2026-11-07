//! Row gather from a table with checked indices.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatherOp {
    pub cols: usize,
}

impl PureForward for GatherOp {
    fn opcode(&self) -> OpCode {
        OpCode::Gather
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if inputs.len() != 2 || self.cols == 0 {
            return Err(Error::ShapeMismatch);
        }
        io_count(&inputs[0])?;
        let rows = io_count(&inputs[1])?;
        Ok(vec![crate::registry::spec_1d(rows * self.cols)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        if inputs.len() != 2 {
            return Err(Error::ShapeMismatch);
        }
        let table = &inputs[0];
        let indices = &inputs[1];
        if table.len() % self.cols != 0 {
            return Err(Error::ShapeMismatch);
        }
        let rows = table.len() / self.cols;
        let mut output = Vec::with_capacity(indices.len() * self.cols);
        for index in indices {
            if *index < 0 || *index as usize >= rows {
                return Err(Error::InvalidIndex);
            }
            let start = *index as usize * self.cols;
            output.extend_from_slice(&table[start..start + self.cols]);
        }
        Ok(KernelRecord::with_output(output))
    }
}
