//! RMS norm: mean square, integer isqrt and one output rounding.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{
    io_count, isqrt_u128, requantize, to_i64_checked, KernelRecord, PureForward,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RmsNormOp {
    pub rows: usize,
    pub cols: usize,
    pub valid_cols: usize,
    pub shift: u32,
    pub clamp: Option<(i64, i64)>,
}

impl PureForward for RmsNormOp {
    fn opcode(&self) -> OpCode {
        OpCode::RmsNorm
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if inputs.len() != 2 || self.valid_cols == 0 || self.valid_cols > self.cols {
            return Err(Error::InvalidConfig);
        }
        if io_count(&inputs[0])? != self.rows * self.cols
            || io_count(&inputs[1])? != self.rows * self.cols
        {
            return Err(Error::ShapeMismatch);
        }
        Ok(vec![crate::registry::spec_1d(self.rows * self.cols)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        if inputs.len() != 2
            || self.valid_cols == 0
            || self.valid_cols > self.cols
            || inputs[0].len() != self.rows * self.cols
            || inputs[1].len() != self.rows * self.cols
        {
            return Err(Error::ShapeMismatch);
        }
        let x = &inputs[0];
        let weight = &inputs[1];
        let mut output = vec![0i64; self.rows * self.cols];
        let mut means = Vec::with_capacity(self.rows);
        let mut rstds = Vec::with_capacity(self.rows);
        let mut raw = Vec::with_capacity(self.rows * self.cols);
        for row in 0..self.rows {
            let start = row * self.cols;
            let mut sum = 0i128;
            for index in 0..self.valid_cols {
                sum += (x[start + index] as i128) * (x[start + index] as i128);
            }
            let mean = sum.div_euclid(self.valid_cols as i128);
            let scale = 1u128
                .checked_shl(2 * self.shift)
                .ok_or(Error::BoundViolation)?;
            let mean_positive = u128::try_from(mean.max(1)).map_err(|_| Error::BoundViolation)?;
            let rstd = isqrt_u128(scale / mean_positive);
            means.push(to_i64_checked(mean)?);
            rstds.push(to_i64_checked(rstd as i128)?);
            for index in 0..self.cols {
                let image = if index < self.valid_cols {
                    (x[start + index] as i128)
                        .checked_mul(rstd as i128)
                        .and_then(|value| value.checked_mul(weight[start + index] as i128))
                        .ok_or(Error::BoundViolation)?
                } else {
                    0
                };
                raw.push(image);
                let mut value = to_i64_checked(requantize(image, self.shift)?)?;
                if let Some((low, high)) = self.clamp {
                    if low > high {
                        return Err(Error::InvalidConfig);
                    }
                    value = value.clamp(low, high);
                }
                output[start + index] = value;
            }
        }
        Ok(KernelRecord {
            outputs: vec![output],
            raw_accumulators: vec![raw],
            auxiliaries: vec![means, rstds],
            ..KernelRecord::default()
        })
    }
}
