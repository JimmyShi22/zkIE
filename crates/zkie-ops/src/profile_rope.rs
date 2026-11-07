//! Profile RoPE: halves within each head, cos/sin per position, one rounding.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, requantize, to_i64_checked, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProfileRopeOp {
    pub head_dim: usize,
    pub shift: u32,
}

impl PureForward for ProfileRopeOp {
    fn opcode(&self) -> OpCode {
        OpCode::ProfileRope
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if inputs.len() != 3 || self.head_dim == 0 || self.head_dim % 2 != 0 {
            return Err(Error::InvalidConfig);
        }
        let count = io_count(&inputs[0])?;
        if count % self.head_dim != 0 {
            return Err(Error::ShapeMismatch);
        }
        let pairs = count * 2 / self.head_dim;
        if io_count(&inputs[1])? != pairs || io_count(&inputs[2])? != pairs {
            return Err(Error::ShapeMismatch);
        }
        Ok(vec![crate::registry::spec_1d(count as usize)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        if inputs.len() != 3 || self.head_dim == 0 || self.head_dim % 2 != 0 {
            return Err(Error::InvalidConfig);
        }
        let x = &inputs[0];
        let cos = &inputs[1];
        let sin = &inputs[2];
        let half = self.head_dim / 2;
        if x.len() % self.head_dim != 0 || cos.len() != x.len() / 2 || sin.len() != x.len() / 2 {
            return Err(Error::ShapeMismatch);
        }
        let rows = x.len() / self.head_dim;
        let mut output = vec![0i64; x.len()];
        let mut raw = vec![0i64; x.len()];
        for row in 0..rows {
            for index in 0..half {
                let even = x[row * self.head_dim + index] as i128;
                let odd = x[row * self.head_dim + half + index] as i128;
                let c = cos[row * half + index] as i128;
                let s = sin[row * half + index] as i128;
                let rotated_even = even
                    .checked_mul(c)
                    .and_then(|value| value.checked_sub(odd.checked_mul(s)?))
                    .ok_or(Error::BoundViolation)?;
                let rotated_odd = even
                    .checked_mul(s)
                    .and_then(|value| value.checked_add(odd.checked_mul(c)?))
                    .ok_or(Error::BoundViolation)?;
                raw[row * self.head_dim + index] = rotated_even as i64;
                raw[row * self.head_dim + half + index] = rotated_odd as i64;
                output[row * self.head_dim + index] =
                    to_i64_checked(requantize(rotated_even, self.shift)?)?;
                output[row * self.head_dim + half + index] =
                    to_i64_checked(requantize(rotated_odd, self.shift)?)?;
            }
        }
        Ok(KernelRecord {
            outputs: vec![output],
            raw_accumulators: vec![raw.into_iter().map(|value| value as i128).collect()],
            ..KernelRecord::default()
        })
    }
}
