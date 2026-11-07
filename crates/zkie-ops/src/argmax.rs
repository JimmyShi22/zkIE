//! Argmax with the first maximum as the deterministic tie-break.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArgmaxOp;

impl PureForward for ArgmaxOp {
    fn opcode(&self) -> OpCode {
        OpCode::Argmax
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        io_count(inputs.first().ok_or(Error::ShapeMismatch)?)?;
        Ok(vec![crate::registry::spec_1d(1)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        let values = inputs.first().ok_or(Error::ShapeMismatch)?;
        if values.is_empty() {
            return Err(Error::InvalidConfig);
        }
        let mut best = 0usize;
        for index in 1..values.len() {
            if values[index] > values[best] {
                best = index;
            }
        }
        Ok(KernelRecord::with_output(vec![best as i64]))
    }
}
