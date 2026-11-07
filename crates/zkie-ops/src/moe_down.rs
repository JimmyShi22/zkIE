//! MoE down: shared input `m` and routed input `xr` have separate matrices;
//! their raw outputs are added before the single rounding.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{requantize, to_i64_checked, KernelRecord, PureForward};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoeDownOp {
    pub shared_weights: Vec<i64>,
    pub routed_weights: Vec<i64>,
    pub width: usize,
    pub out_cols: usize,
    pub shift: u32,
}

impl PureForward for MoeDownOp {
    fn opcode(&self) -> OpCode {
        OpCode::MoeDown
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if inputs.len() != 2 {
            return Err(Error::ShapeMismatch);
        }
        let width = crate::registry::io_count(&inputs[0])?;
        if crate::registry::io_count(&inputs[1])? != width
            || self.shared_weights.len() != width * self.out_cols
            || self.routed_weights.len() != width * self.out_cols
        {
            return Err(Error::ShapeMismatch);
        }
        Ok(vec![crate::registry::spec_1d(self.out_cols)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        if self.width == 0
            || self.out_cols == 0
            || self.shared_weights.len() != self.width * self.out_cols
            || self.routed_weights.len() != self.width * self.out_cols
        {
            return Err(Error::ShapeMismatch);
        }
        let m = inputs.first().ok_or(Error::ShapeMismatch)?;
        let xr = inputs.get(1).ok_or(Error::ShapeMismatch)?;
        if m.len() != self.width || xr.len() != self.width {
            return Err(Error::ShapeMismatch);
        }
        let mut shared = Vec::with_capacity(self.out_cols);
        let mut routed = Vec::with_capacity(self.out_cols);
        let mut total = Vec::with_capacity(self.out_cols);
        let mut output = Vec::with_capacity(self.out_cols);
        for column in 0..self.out_cols {
            let mut shared_raw = 0i128;
            let mut routed_raw = 0i128;
            for index in 0..self.width {
                shared_raw = shared_raw
                    .checked_add((m[index] as i128) * (self.shared_weights[index * self.out_cols + column] as i128))
                    .ok_or(Error::BoundViolation)?;
                routed_raw = routed_raw
                    .checked_add((xr[index] as i128) * (self.routed_weights[index * self.out_cols + column] as i128))
                    .ok_or(Error::BoundViolation)?;
            }
            let sum = shared_raw.checked_add(routed_raw).ok_or(Error::BoundViolation)?;
            shared.push(shared_raw);
            routed.push(routed_raw);
            total.push(sum);
            output.push(to_i64_checked(requantize(sum, self.shift)?)?);
        }
        Ok(KernelRecord {
            outputs: vec![output],
            raw_accumulators: vec![shared, routed, total],
            ..KernelRecord::default()
        })
    }
}
