//! Profile standardization: statistics are taken from one source row of the
//! input and applied to every row, then the result is padded with zeros.
//!
//! `mean = round(sum(x[source][0..valid]) / valid)`,
//! `var = round(sum((x[source][i] - mean)^2) / valid)`,
//! `rstd = rsqrt_table[rsqrt_index(var)]`,
//! `out[row][i] = round((x[row][i] - mean) * rstd / 2^shift)` for `i < valid`
//! and `0` for padding columns.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{
    io_count, profile_rsqrt_index, round_div_half_up, to_i64_checked, KernelRecord, PureForward,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProfileStandardizeOp {
    pub rows: usize,
    pub cols: usize,
    pub output_cols: usize,
    pub valid_cols: usize,
    pub source_row: usize,
    pub output_shift: u32,
    pub clamp: Option<(i64, i64)>,
}

impl ProfileStandardizeOp {
    fn table_value(table: &[i64], extension: Option<&[i64]>, index: usize) -> Result<i64> {
        if index < table.len() {
            return Ok(table[index]);
        }
        let extension = extension.ok_or(Error::BoundViolation)?;
        let offset = index.checked_sub(table.len()).ok_or(Error::BoundViolation)?;
        extension.get(offset).copied().ok_or(Error::BoundViolation)
    }
}

impl PureForward for ProfileStandardizeOp {
    fn opcode(&self) -> OpCode {
        OpCode::ProfileStandardize
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if (inputs.len() != 2 && inputs.len() != 3)
            || self.rows == 0
            || self.cols == 0
            || self.valid_cols == 0
            || self.valid_cols > self.cols
            || self.output_cols < self.cols
            || self.source_row >= self.rows
        {
            return Err(Error::InvalidConfig);
        }
        if io_count(&inputs[0])? != self.rows * self.cols {
            return Err(Error::ShapeMismatch);
        }
        Ok(vec![crate::registry::spec_1d(self.rows * self.output_cols)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        if (inputs.len() != 2 && inputs.len() != 3)
            || self.rows == 0
            || self.cols == 0
            || self.valid_cols == 0
            || self.valid_cols > self.cols
            || self.output_cols < self.cols
            || self.source_row >= self.rows
            || inputs[0].len() != self.rows * self.cols
            || inputs[1].is_empty()
            || (inputs.len() == 3 && inputs[2].is_empty())
        {
            return Err(Error::ShapeMismatch);
        }
        let x = &inputs[0];
        let table = &inputs[1];
        let extension = inputs.get(2).map(Vec::as_slice);
        let source = self.source_row * self.cols;
        let mut sum = 0i128;
        for index in 0..self.valid_cols {
            sum += x[source + index] as i128;
        }
        let mean = round_div_half_up(sum, self.valid_cols as i128)?;
        let mut squares = 0i128;
        for index in 0..self.valid_cols {
            let centered = x[source + index] as i128 - mean;
            squares += centered * centered;
        }
        let variance = round_div_half_up(squares, self.valid_cols as i128)?;
        let variance_i64 = i64::try_from(variance).map_err(|_| Error::BoundViolation)?;
        let table_index = profile_rsqrt_index(variance_i64)?;
        let rstd = Self::table_value(table, extension, table_index as usize)
            .map_err(|_| Error::BoundViolation)?;
        let mean_i64 = i64::try_from(mean).map_err(|_| Error::BoundViolation)?;
        let mut output = vec![0i64; self.rows * self.output_cols];
        let mut raw = vec![0i128; self.rows * self.output_cols];
        for row in 0..self.rows {
            for index in 0..self.output_cols {
                let image = if index < self.valid_cols {
                    (x[row * self.cols + index] as i128 - mean_i64 as i128)
                        .checked_mul(rstd as i128)
                        .ok_or(Error::BoundViolation)?
                } else {
                    0
                };
                raw[row * self.output_cols + index] = image;
                let mut value = to_i64_checked(round_div_half_up(image, 1i128 << self.output_shift)?)?;
                if let Some((low, high)) = self.clamp {
                    if low > high {
                        return Err(Error::InvalidConfig);
                    }
                    value = value.clamp(low, high);
                }
                output[row * self.output_cols + index] = value;
            }
        }
        Ok(KernelRecord {
            outputs: vec![output],
            raw_accumulators: vec![raw],
            auxiliaries: vec![vec![mean_i64], vec![variance_i64], vec![rstd]],
            ..KernelRecord::default()
        })
    }
}
