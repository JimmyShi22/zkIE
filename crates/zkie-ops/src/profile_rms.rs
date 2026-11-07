//! Profile-backed RMSNorm: round-half-up mean square, rsqrt table lookup,
//! one output rounding, optional clamp (no bias).
//!
//! The rsqrt table may be supplied as a base table plus an optional extension
//! table. Indices below `base.len()` read the base table; larger indices read
//! the extension table at `index - base.len()`.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{
    io_count, profile_rsqrt_index, round_div_half_up, to_i64_checked, KernelRecord, PureForward,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProfileRmsNormOp {
    pub rows: usize,
    pub cols: usize,
    pub valid_cols: usize,
    pub output_shift: u32,
    pub clamp: Option<(i64, i64)>,
}

impl ProfileRmsNormOp {
    fn table_value(table: &[i64], extension: Option<&[i64]>, index: usize) -> Result<i64> {
        if index < table.len() {
            return Ok(table[index]);
        }
        let extension = extension.ok_or(Error::BoundViolation)?;
        let offset = index.checked_sub(table.len()).ok_or(Error::BoundViolation)?;
        extension.get(offset).copied().ok_or(Error::BoundViolation)
    }
}

impl PureForward for ProfileRmsNormOp {
    fn opcode(&self) -> OpCode {
        OpCode::ProfileRmsNorm
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if (inputs.len() != 3 && inputs.len() != 4)
            || self.valid_cols == 0
            || self.valid_cols > self.cols
        {
            return Err(Error::InvalidConfig);
        }
        let count = self.rows * self.cols;
        if io_count(&inputs[0])? != count || io_count(&inputs[1])? != count {
            return Err(Error::ShapeMismatch);
        }
        Ok(vec![crate::registry::spec_1d(count)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        if (inputs.len() != 3 && inputs.len() != 4)
            || self.valid_cols == 0
            || self.valid_cols > self.cols
            || inputs[0].len() != self.rows * self.cols
            || inputs[1].len() != self.rows * self.cols
            || inputs[2].is_empty()
            || (inputs.len() == 4 && inputs[3].is_empty())
        {
            return Err(Error::ShapeMismatch);
        }
        let x = &inputs[0];
        let weight = &inputs[1];
        let table = &inputs[2];
        let extension = inputs.get(3).map(Vec::as_slice);
        let mut output = vec![0i64; self.rows * self.cols];
        let mut means = Vec::with_capacity(self.rows);
        let mut rstds = Vec::with_capacity(self.rows);
        let mut raw = Vec::with_capacity(self.rows * self.cols);
        for row in 0..self.rows {
            let start = row * self.cols;
            let mut mean_square = 0i128;
            for index in 0..self.valid_cols {
                let value = x[start + index] as i128;
                mean_square += value * value;
            }
            let mean_square = round_div_half_up(mean_square, self.valid_cols as i128)?;
            let mean_square_i64 = i64::try_from(mean_square).map_err(|_| Error::BoundViolation)?;
            let table_index = profile_rsqrt_index(mean_square_i64)?;
            let rstd = Self::table_value(table, extension, table_index as usize)
                .map_err(|_| Error::BoundViolation)?;
            means.push(mean_square_i64);
            rstds.push(rstd);
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
                let mut value = to_i64_checked(round_div_half_up(image, 1i128 << self.output_shift)?)?;
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
