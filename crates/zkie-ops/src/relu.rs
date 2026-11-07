//! Elementwise rectified linear unit.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReluOp;

impl PureForward for ReluOp {
    fn opcode(&self) -> OpCode {
        OpCode::Relu
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        let spec = inputs.first().ok_or(Error::ShapeMismatch)?;
        Ok(vec![crate::registry::spec_1d(io_count(spec)?)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        let values = inputs.first().ok_or(Error::ShapeMismatch)?;
        Ok(KernelRecord::with_output(
            values.iter().map(|value| (*value).max(0)).collect(),
        ))
    }
}
