//! Rotation projection on raw Q/K values; one rounding after the rotation.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, requantize, to_i64_checked, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RopeProjectionOp {
    pub shift: u32,
}

impl PureForward for RopeProjectionOp {
    fn opcode(&self) -> OpCode {
        OpCode::RopeProjection
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if inputs.len() != 3 {
            return Err(Error::ShapeMismatch);
        }
        let count = io_count(&inputs[0])?;
        if count % 2 != 0 || io_count(&inputs[1])? != count / 2 || io_count(&inputs[2])? != count / 2
        {
            return Err(Error::ShapeMismatch);
        }
        Ok(vec![crate::registry::spec_1d(count)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        if inputs.len() != 3 {
            return Err(Error::ShapeMismatch);
        }
        let x = &inputs[0];
        let cos = &inputs[1];
        let sin = &inputs[2];
        if x.len() % 2 != 0 || cos.len() != x.len() / 2 || sin.len() != x.len() / 2 {
            return Err(Error::ShapeMismatch);
        }
        let pairs = x.len() / 2;
        let mut output = Vec::with_capacity(x.len());
        let mut raw = Vec::with_capacity(x.len());
        for pair in 0..pairs {
            let even = x[2 * pair] as i128;
            let odd = x[2 * pair + 1] as i128;
            let c = cos[pair] as i128;
            let s = sin[pair] as i128;
            let rotated_even = even
                .checked_mul(c)
                .and_then(|value| value.checked_sub(odd.checked_mul(s)?))
                .ok_or(Error::BoundViolation)?;
            let rotated_odd = even
                .checked_mul(s)
                .and_then(|value| value.checked_add(odd.checked_mul(c)?))
                .ok_or(Error::BoundViolation)?;
            raw.push(rotated_even);
            raw.push(rotated_odd);
            output.push(to_i64_checked(requantize(rotated_even, self.shift)?)?);
            output.push(to_i64_checked(requantize(rotated_odd, self.shift)?)?);
        }
        Ok(KernelRecord {
            outputs: vec![output],
            raw_accumulators: vec![raw],
            ..KernelRecord::default()
        })
    }
}
