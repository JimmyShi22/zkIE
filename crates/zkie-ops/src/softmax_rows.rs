//! Row-batched integer softmax: max shift, exp table, one output division.
//!
//! Every row is processed independently with the same semantics as the
//! single-row softmax op. Maxima and denominators are committed as
//! auxiliaries; the per-element numerator is a wide raw accumulator. The
//! per-element `u` vector is recomputed by the verifier and is not carried in
//! the proof.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{
    exp_profile_value, io_count, round_div_half_up, to_i64_checked, KernelRecord, PureForward,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoftmaxRowsOp {
    pub rows: usize,
    pub cols: usize,
    pub shift: u32,
}

impl PureForward for SoftmaxRowsOp {
    fn opcode(&self) -> OpCode {
        OpCode::SoftmaxRows
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if inputs.len() != 2 || self.rows == 0 || self.cols == 0 {
            return Err(Error::InvalidConfig);
        }
        let count = self
            .rows
            .checked_mul(self.cols)
            .ok_or(Error::ResourceLimit)?;
        if io_count(&inputs[0])? != count {
            return Err(Error::ShapeMismatch);
        }
        Ok(vec![crate::registry::spec_1d(count)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        if inputs.len() != 2 || self.rows == 0 || self.cols == 0 {
            return Err(Error::InvalidConfig);
        }
        let count = self
            .rows
            .checked_mul(self.cols)
            .ok_or(Error::ResourceLimit)?;
        if inputs[0].len() != count || inputs[1].is_empty() {
            return Err(Error::ShapeMismatch);
        }
        let x = &inputs[0];
        let table = &inputs[1];
        let scale = 1i128.checked_shl(self.shift).ok_or(Error::BoundViolation)?;
        let mut output = vec![0i64; count];
        let mut raw = Vec::with_capacity(count);
        let mut maxima = Vec::with_capacity(self.rows);
        let mut denominators = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            let start = row * self.cols;
            let max = *x[start..start + self.cols]
                .iter()
                .max()
                .ok_or(Error::InvalidConfig)?;
            let mut entries = Vec::with_capacity(self.cols);
            let mut denominator = 0i128;
            for index in 0..self.cols {
                let distance = max.checked_sub(x[start + index]).ok_or(Error::BoundViolation)?;
                let entry = exp_profile_value(table, distance)?;
                if entry < 0 {
                    return Err(Error::InvalidConfig);
                }
                denominator = denominator
                    .checked_add(entry as i128)
                    .ok_or(Error::BoundViolation)?;
                entries.push(entry);
            }
            if denominator == 0 {
                return Err(Error::DivisionByZero);
            }
            maxima.push(max);
            denominators.push(to_i64_checked(denominator)?);
            for (index, entry) in entries.iter().enumerate() {
                let numerator = (*entry as i128)
                    .checked_mul(scale)
                    .ok_or(Error::BoundViolation)?;
                raw.push(numerator);
                output[start + index] = to_i64_checked(round_div_half_up(numerator, denominator)?)?;
            }
        }
        Ok(KernelRecord {
            outputs: vec![output],
            raw_accumulators: vec![raw],
            auxiliaries: vec![maxima, denominators],
            ..KernelRecord::default()
        })
    }
}
