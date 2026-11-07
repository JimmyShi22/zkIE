//! Profile-backed centered LayerNorm: round-half-up mean/variance, piecewise
//! rsqrt table lookup, one output rounding and explicit bias.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{
    io_count, profile_rsqrt_index, round_div_half_up, to_i64_checked, KernelRecord, PureForward,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProfileLayerNormOp {
    pub rows: usize,
    pub cols: usize,
    pub valid_cols: usize,
    pub output_shift: u32,
    pub clamp: Option<(i64, i64)>,
}

impl PureForward for ProfileLayerNormOp {
    fn opcode(&self) -> OpCode {
        OpCode::ProfileLayerNorm
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if inputs.len() != 4 || self.valid_cols == 0 || self.valid_cols > self.cols {
            return Err(Error::InvalidConfig);
        }
        let count = self.rows * self.cols;
        if io_count(&inputs[0])? != count
            || io_count(&inputs[1])? != count
            || io_count(&inputs[2])? != count
        {
            return Err(Error::ShapeMismatch);
        }
        Ok(vec![crate::registry::spec_1d(count)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        if inputs.len() != 4
            || self.valid_cols == 0
            || self.valid_cols > self.cols
            || inputs[0].len() != self.rows * self.cols
            || inputs[1].len() != self.rows * self.cols
            || inputs[2].len() != self.rows * self.cols
            || inputs[3].is_empty()
        {
            return Err(Error::ShapeMismatch);
        }
        let x = &inputs[0];
        let weight = &inputs[1];
        let bias = &inputs[2];
        let table = &inputs[3];
        let mut output = vec![0i64; self.rows * self.cols];
        let mut means = Vec::with_capacity(self.rows);
        let mut variances = Vec::with_capacity(self.rows);
        let mut rstds = Vec::with_capacity(self.rows);
        let mut raw = Vec::with_capacity(self.rows * self.cols);
        for row in 0..self.rows {
            let start = row * self.cols;
            let sum: i128 = (0..self.valid_cols)
                .map(|index| x[start + index] as i128)
                .sum();
            let mean = round_div_half_up(sum, self.valid_cols as i128)?;
            let mut squares = 0i128;
            for index in 0..self.valid_cols {
                let centered = x[start + index] as i128 - mean;
                squares += centered * centered;
            }
            let variance = round_div_half_up(squares, self.valid_cols as i128)?;
            let variance_i64 = i64::try_from(variance).map_err(|_| Error::BoundViolation)?;
            let table_index = profile_rsqrt_index(variance_i64)?;
            if table_index as usize >= table.len() {
                return Err(Error::BoundViolation);
            }
            let rstd = table[table_index as usize];
            means.push(to_i64_checked(mean)?);
            variances.push(variance_i64);
            rstds.push(rstd);
            for index in 0..self.cols {
                let image = if index < self.valid_cols {
                    let centered = x[start + index] as i128 - mean;
                    centered
                        .checked_mul(rstd as i128)
                        .and_then(|value| value.checked_mul(weight[start + index] as i128))
                        .ok_or(Error::BoundViolation)?
                } else {
                    0
                };
                raw.push(image);
                let mut value = to_i64_checked(
                    round_div_half_up(image, 1i128 << self.output_shift)? + bias[start + index] as i128,
                )?;
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
            auxiliaries: vec![means, variances, rstds],
            ..KernelRecord::default()
        })
    }
}
