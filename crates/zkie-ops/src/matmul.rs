//! Exact integer matrix product with a single requantization.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, matmul_raw, requantize, to_i64_checked, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatMulOp {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub shift: u32,
}

impl PureForward for MatMulOp {
    fn opcode(&self) -> OpCode {
        OpCode::MatMul
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if inputs.len() != 2 {
            return Err(Error::ShapeMismatch);
        }
        if io_count(&inputs[0])? != self.m * self.k || io_count(&inputs[1])? != self.k * self.n {
            return Err(Error::ShapeMismatch);
        }
        Ok(vec![crate::registry::spec_1d(self.m * self.n)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        if inputs.len() != 2 {
            return Err(Error::ShapeMismatch);
        }
        let raw = matmul_raw(&inputs[0], &inputs[1], self.m, self.n, self.k)?;
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
