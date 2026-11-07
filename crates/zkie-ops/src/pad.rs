//! Pad: the valid prefix is copied, the rest receives the declared value.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PadOp {
    pub valid_len: usize,
    pub value: i64,
}

impl PureForward for PadOp {
    fn opcode(&self) -> OpCode {
        OpCode::Pad
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        let count = io_count(inputs.first().ok_or(Error::ShapeMismatch)?)?;
        if self.valid_len > count {
            return Err(Error::InvalidConfig);
        }
        Ok(vec![crate::registry::spec_1d(count)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        let values = inputs.first().ok_or(Error::ShapeMismatch)?;
        if self.valid_len > values.len() {
            return Err(Error::InvalidConfig);
        }
        let mut output = values.clone();
        for value in output.iter_mut().skip(self.valid_len) {
            *value = self.value;
        }
        Ok(KernelRecord::with_output(output))
    }
}
