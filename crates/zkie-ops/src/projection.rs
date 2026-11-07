//! Weight projection: matmul plus bias, one rounding at the end.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, matmul_raw, requantize, to_i64_checked, KernelRecord, PureForward};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionOp {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub shift: u32,
    pub bias: Option<Vec<i64>>,
}

impl PureForward for ProjectionOp {
    fn opcode(&self) -> OpCode {
        OpCode::Projection
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if inputs.len() != 2 {
            return Err(Error::ShapeMismatch);
        }
        if io_count(&inputs[0])? != self.m * self.k || io_count(&inputs[1])? != self.k * self.n {
            return Err(Error::ShapeMismatch);
        }
        if let Some(bias) = &self.bias {
            if bias.len() != self.m * self.n {
                return Err(Error::ShapeMismatch);
            }
        }
        Ok(vec![crate::registry::spec_1d(self.m * self.n)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        let mut raw = matmul_raw(&inputs[0], &inputs[1], self.m, self.n, self.k)?;
        if let Some(bias) = &self.bias {
            for (slot, value) in raw.iter_mut().zip(bias) {
                *slot = slot.checked_add(*value as i128).ok_or(Error::BoundViolation)?;
            }
        }
        let output = raw
            .iter()
            .map(|value| to_i64_checked(requantize(*value, self.shift)?))
            .collect::<Result<_>>()?;
        Ok(KernelRecord {
            outputs: vec![output],
            raw_accumulators: vec![raw],
            ..KernelRecord::default()
        })
    }
}
