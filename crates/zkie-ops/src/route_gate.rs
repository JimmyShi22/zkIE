//! Router gate: top-k experts above a threshold with deterministic ties.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouteGateOp {
    pub top_k: usize,
    pub threshold: i64,
}

impl PureForward for RouteGateOp {
    fn opcode(&self) -> OpCode {
        OpCode::RouteGate
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        let count = io_count(inputs.first().ok_or(Error::ShapeMismatch)?)?;
        if self.top_k == 0 || self.top_k > count {
            return Err(Error::InvalidConfig);
        }
        Ok(vec![crate::registry::spec_1d(1 + 2 * self.top_k)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        let scores = inputs.first().ok_or(Error::ShapeMismatch)?;
        if self.top_k == 0 || self.top_k > scores.len() {
            return Err(Error::InvalidConfig);
        }
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|left, right| {
            scores[*right]
                .cmp(&scores[*left])
                .then_with(|| left.cmp(right))
        });
        let mut selected: Vec<usize> = order
            .into_iter()
            .filter(|index| scores[*index] >= self.threshold)
            .take(self.top_k)
            .collect();
        let count = selected.len();
        selected.resize(self.top_k, usize::MAX);
        let mut output = Vec::with_capacity(1 + 2 * self.top_k);
        output.push(count as i64);
        for index in &selected {
            output.push(if *index == usize::MAX { -1 } else { *index as i64 });
        }
        for index in &selected {
            output.push(if *index == usize::MAX {
                0
            } else {
                scores[*index]
            });
        }
        Ok(KernelRecord::with_output(output))
    }
}
