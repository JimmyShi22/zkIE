//! Gated FFN: one rounding after the gate/up product.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, requantize, to_i64_checked, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatedFfnOp {
    pub shift: u32,
}

impl PureForward for GatedFfnOp {
    fn opcode(&self) -> OpCode {
        OpCode::GatedFfn
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if inputs.len() != 2 || io_count(&inputs[0])? != io_count(&inputs[1])? {
            return Err(Error::ShapeMismatch);
        }
        Ok(vec![crate::registry::spec_1d(io_count(&inputs[0])?)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        if inputs.len() != 2 || inputs[0].len() != inputs[1].len() {
            return Err(Error::ShapeMismatch);
        }
        let mut output = Vec::with_capacity(inputs[0].len());
        let mut raw = Vec::with_capacity(inputs[0].len());
        for (gate, up) in inputs[0].iter().zip(&inputs[1]) {
            let product = (*gate as i128).checked_mul(*up as i128).ok_or(Error::BoundViolation)?;
            raw.push(product);
            output.push(to_i64_checked(requantize(product, self.shift)?)?);
        }
        Ok(KernelRecord {
            outputs: vec![output],
            raw_accumulators: vec![raw],
            ..KernelRecord::default()
        })
    }
}
