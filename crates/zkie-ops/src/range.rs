//! Range predicate with explicit lower and upper bounds.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangeOp {
    pub low: i64,
    pub high: i64,
}

impl PureForward for RangeOp {
    fn opcode(&self) -> OpCode {
        OpCode::Range
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if self.low > self.high {
            return Err(Error::InvalidConfig);
        }
        let count = io_count(inputs.first().ok_or(Error::ShapeMismatch)?)?;
        Ok(vec![crate::registry::spec_1d(count)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        if self.low > self.high {
            return Err(Error::InvalidConfig);
        }
        let values = inputs.first().ok_or(Error::ShapeMismatch)?;
        let mut output = Vec::with_capacity(values.len());
        let mut outside = Vec::new();
        for (index, value) in values.iter().enumerate() {
            let inside = *value >= self.low && *value <= self.high;
            output.push(if inside { 1 } else { 0 });
            if !inside {
                outside.push(index as i64);
            }
        }
        Ok(KernelRecord {
            outputs: vec![output],
            range_helpers: vec![outside],
            ..KernelRecord::default()
        })
    }
}
