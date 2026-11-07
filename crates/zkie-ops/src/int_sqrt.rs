//! Exact floor square root, rejecting negative inputs.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, isqrt_u128, to_i64_checked, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IntSqrtOp;

impl PureForward for IntSqrtOp {
    fn opcode(&self) -> OpCode {
        OpCode::IntSqrt
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        let count = io_count(inputs.first().ok_or(Error::ShapeMismatch)?)?;
        Ok(vec![crate::registry::spec_1d(count)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        let values = inputs.first().ok_or(Error::ShapeMismatch)?;
        let mut output = Vec::with_capacity(values.len());
        for value in values {
            if *value < 0 {
                return Err(Error::BoundViolation);
            }
            let root = isqrt_u128(*value as u128);
            output.push(to_i64_checked(root as i128)?);
        }
        Ok(KernelRecord::with_output(output))
    }
}
