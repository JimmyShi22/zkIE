//! Causal attention with max shift, exp table, denominator and one output
//! division. The probabilities are never quantized before multiplying V.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{exp_profile_value, io_count, to_i64_checked, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CausalAttentionOp {
    pub seq: usize,
    pub head_dim: usize,
    pub shift: u32,
}

impl PureForward for CausalAttentionOp {
    fn opcode(&self) -> OpCode {
        OpCode::CausalAttention
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if inputs.len() != 4 || self.head_dim == 0 || self.seq == 0 {
            return Err(Error::InvalidConfig);
        }
        let count = self.seq * self.head_dim;
        for spec in &inputs[..3] {
            if io_count(spec)? != count {
                return Err(Error::ShapeMismatch);
            }
        }
        if io_count(&inputs[3])? == 0 {
            return Err(Error::InvalidConfig);
        }
        Ok(vec![crate::registry::spec_1d(count)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        if inputs.len() != 4 {
            return Err(Error::ShapeMismatch);
        }
        let q = &inputs[0];
        let k = &inputs[1];
        let v = &inputs[2];
        let table = &inputs[3];
        let mut output = vec![0i64; self.seq * self.head_dim];
        let mut raw_all = Vec::with_capacity(self.seq * self.head_dim);
        let mut u_all = Vec::new();
        let mut denominators = Vec::with_capacity(self.seq);
        let mut maxima = Vec::with_capacity(self.seq);
        let scale = 1i128
            .checked_shl(self.shift)
            .ok_or(Error::BoundViolation)?;
        for row in 0..self.seq {
            let mut scores = Vec::with_capacity(row + 1);
            for column in 0..=row {
                let mut score = 0i128;
                for index in 0..self.head_dim {
                    score = score
                        .checked_add(
                            (q[row * self.head_dim + index] as i128)
                                * (k[column * self.head_dim + index] as i128),
                        )
                        .ok_or(Error::BoundViolation)?;
                }
                scores.push(score);
            }
            let max = *scores.iter().max().expect("non-empty");
            maxima.push(to_i64_checked(max)?);
            let mut u = Vec::with_capacity(scores.len());
            let mut denominator = 0i128;
            for score in &scores {
                let distance = max.checked_sub(*score).ok_or(Error::BoundViolation)?;
                let distance = i64::try_from(distance).map_err(|_| Error::BoundViolation)?;
                let entry = exp_profile_value(table, distance)?;
                if entry < 0 {
                    return Err(Error::InvalidConfig);
                }
                denominator = denominator
                    .checked_add(entry as i128)
                    .ok_or(Error::BoundViolation)?;
                u.push(entry);
            }
            if denominator == 0 {
                return Err(Error::DivisionByZero);
            }
            denominators.push(to_i64_checked(denominator)?);
            for index in 0..self.head_dim {
                let mut numerator = 0i128;
                for (column, entry) in u.iter().enumerate() {
                    numerator = numerator
                        .checked_add((*entry as i128) * (v[column * self.head_dim + index] as i128))
                        .ok_or(Error::BoundViolation)?;
                }
                let scaled = numerator
                    .checked_mul(scale)
                    .ok_or(Error::BoundViolation)?;
                raw_all.push(scaled);
                output[row * self.head_dim + index] =
                    to_i64_checked(scaled.div_euclid(denominator))?;
            }
            u_all.extend(u);
        }
        Ok(KernelRecord {
            outputs: vec![output],
            raw_accumulators: vec![raw_all],
            auxiliaries: vec![maxima, u_all, denominators],
            ..KernelRecord::default()
        })
    }
}
