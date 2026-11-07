//! Affine scale with one explicit rounding and an explicit clamp.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, requantize, to_i64_checked, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineScaleOp {
    pub multiplier: i128,
    pub addend: i128,
    pub shift: u32,
    pub clamp: Option<(i64, i64)>,
}

impl PureForward for AffineScaleOp {
    fn opcode(&self) -> OpCode {
        OpCode::AffineScale
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        let spec = inputs.first().ok_or(Error::ShapeMismatch)?;
        Ok(vec![crate::registry::spec_1d(io_count(spec)?)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        let values = inputs.first().ok_or(Error::ShapeMismatch)?;
        let mut output = Vec::with_capacity(values.len());
        let mut raw = Vec::with_capacity(values.len());
        for value in values {
            let product = self
                .multiplier
                .checked_mul(*value as i128)
                .ok_or(Error::BoundViolation)?;
            let sum = product.checked_add(self.addend).ok_or(Error::BoundViolation)?;
            raw.push(sum);
            let mut result = to_i64_checked(requantize(sum, self.shift)?)?;
            if let Some((low, high)) = self.clamp {
                if low > high {
                    return Err(Error::InvalidConfig);
                }
                result = result.clamp(low, high);
            }
            output.push(result);
        }
        Ok(KernelRecord {
            outputs: vec![output],
            raw_accumulators: vec![raw],
            ..KernelRecord::default()
        })
    }
}
