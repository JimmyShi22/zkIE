//! Exact floor quotient and non-negative remainder.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, to_i64_checked, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DivRoundOp;

impl PureForward for DivRoundOp {
    fn opcode(&self) -> OpCode {
        OpCode::DivRound
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if inputs.len() != 2 || io_count(&inputs[0])? != io_count(&inputs[1])? {
            return Err(Error::ShapeMismatch);
        }
        Ok(vec![crate::registry::spec_1d(2 * io_count(&inputs[0])?)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        if inputs.len() != 2 || inputs[0].len() != inputs[1].len() {
            return Err(Error::ShapeMismatch);
        }
        let mut quotients = Vec::with_capacity(inputs[0].len());
        let mut remainders = Vec::with_capacity(inputs[0].len());
        for (numerator, divisor) in inputs[0].iter().zip(&inputs[1]) {
            if *divisor <= 0 {
                return Err(if *divisor == 0 {
                    Error::DivisionByZero
                } else {
                    Error::InvalidConfig
                });
            }
            let quotient = (*numerator as i128).div_euclid(*divisor as i128);
            let remainder = (*numerator as i128).rem_euclid(*divisor as i128);
            quotients.push(to_i64_checked(quotient)?);
            remainders.push(to_i64_checked(remainder)?);
        }
        let mut output = quotients.clone();
        output.extend_from_slice(&remainders);
        Ok(KernelRecord {
            outputs: vec![output],
            quotients: vec![quotients],
            ..KernelRecord::default()
        })
    }
}
