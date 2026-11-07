//! Scatter with duplicate-index rejection.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScatterOp {
    pub size: usize,
}

impl PureForward for ScatterOp {
    fn opcode(&self) -> OpCode {
        OpCode::Scatter
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if inputs.len() != 2 || crate::registry::io_count(&inputs[0])? != crate::registry::io_count(&inputs[1])? {
            return Err(Error::ShapeMismatch);
        }
        Ok(vec![crate::registry::spec_1d(self.size)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        if inputs.len() != 2 || inputs[0].len() != inputs[1].len() {
            return Err(Error::ShapeMismatch);
        }
        let mut output = vec![0i64; self.size];
        let mut seen = vec![false; self.size];
        for (index, value) in inputs[0].iter().zip(&inputs[1]) {
            if *index < 0 || *index as usize >= self.size || seen[*index as usize] {
                return Err(Error::InvalidIndex);
            }
            seen[*index as usize] = true;
            output[*index as usize] = *value;
        }
        Ok(KernelRecord::with_output(output))
    }
}
