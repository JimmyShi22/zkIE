//! Same-scale addition. A scale mismatch is an explicit error.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, to_i64_checked, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AddOp;

impl PureForward for AddOp {
    fn opcode(&self) -> OpCode {
        OpCode::Add
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if inputs.len() != 2 {
            return Err(Error::ShapeMismatch);
        }
        let left = io_count(&inputs[0])?;
        if left != io_count(&inputs[1])? {
            return Err(Error::ShapeMismatch);
        }
        if let (Some(a), Some(b)) = (inputs[0].fixed, inputs[1].fixed) {
            if a.scale != b.scale {
                return Err(Error::ScaleMismatch);
            }
        }
        Ok(vec![crate::registry::spec_1d(left)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        if inputs.len() != 2 || inputs[0].len() != inputs[1].len() {
            return Err(Error::ShapeMismatch);
        }
        let mut output = Vec::with_capacity(inputs[0].len());
        for (left, right) in inputs[0].iter().zip(&inputs[1]) {
            output.push(to_i64_checked(*left as i128 + *right as i128)?);
        }
        Ok(KernelRecord::with_output(output))
    }
}
