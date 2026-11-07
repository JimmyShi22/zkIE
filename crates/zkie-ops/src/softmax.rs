//! Integer softmax with a max shift, an exp table and a single output division.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{exp_profile_value, io_count, round_div_half_up, to_i64_checked, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoftmaxOp {
    pub shift: u32,
}

impl SoftmaxOp {
    pub fn forward(&self, logits: &[i64], exp_table: &[i64]) -> Result<KernelRecord> {
        if logits.is_empty() {
            return Err(Error::InvalidConfig);
        }
        let max = *logits.iter().max().expect("non-empty");
        let mut u = Vec::with_capacity(logits.len());
        let mut denominator = 0i128;
        for value in logits {
            let distance = max.checked_sub(*value).ok_or(Error::BoundViolation)?;
            let entry = exp_profile_value(exp_table, distance)?;
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
        let scale = 1i128.checked_shl(self.shift).ok_or(Error::BoundViolation)?;
        let mut output = Vec::with_capacity(logits.len());
        let mut raw = Vec::with_capacity(logits.len());
        for entry in &u {
            let numerator = (*entry as i128)
                .checked_mul(scale)
                .ok_or(Error::BoundViolation)?;
            raw.push(numerator);
            output.push(to_i64_checked(round_div_half_up(numerator, denominator)?)?);
        }
        Ok(KernelRecord {
            outputs: vec![output],
            raw_accumulators: vec![raw],
            auxiliaries: vec![vec![max], u, vec![to_i64_checked(denominator)?]],
            ..KernelRecord::default()
        })
    }
}

impl PureForward for SoftmaxOp {
    fn opcode(&self) -> OpCode {
        OpCode::Softmax
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        let spec = inputs.first().ok_or(Error::ShapeMismatch)?;
        Ok(vec![crate::registry::spec_1d(io_count(spec)?)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        if inputs.len() != 2 {
            return Err(Error::ShapeMismatch);
        }
        self.forward(&inputs[0], &inputs[1])
    }
}
