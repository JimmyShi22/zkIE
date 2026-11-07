//! Deterministic top-k: higher value first, lower index on ties.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopKOp {
    pub k: usize,
}

impl PureForward for TopKOp {
    fn opcode(&self) -> OpCode {
        OpCode::TopK
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        let count = io_count(inputs.first().ok_or(Error::ShapeMismatch)?)?;
        if self.k == 0 || self.k > count {
            return Err(Error::InvalidConfig);
        }
        Ok(vec![crate::registry::spec_1d(2 * self.k)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        let values = inputs.first().ok_or(Error::ShapeMismatch)?;
        if self.k == 0 || self.k > values.len() {
            return Err(Error::InvalidConfig);
        }
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|left, right| {
            values[*right]
                .cmp(&values[*left])
                .then_with(|| left.cmp(right))
        });
        let mut indices = Vec::with_capacity(self.k);
        let mut selected = Vec::with_capacity(self.k);
        for index in order.into_iter().take(self.k) {
            indices.push(index as i64);
            selected.push(values[index]);
        }
        let mut output = indices;
        output.extend(selected);
        Ok(KernelRecord::with_output(output))
    }
}
