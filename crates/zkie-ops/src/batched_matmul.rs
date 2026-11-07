//! Batched matrix product, one rounding per output.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, matmul_raw, requantize, to_i64_checked, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchedMatMulOp {
    pub batch: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub shift: u32,
}

impl PureForward for BatchedMatMulOp {
    fn opcode(&self) -> OpCode {
        OpCode::BatchedMatMul
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if inputs.len() != 2 {
            return Err(Error::ShapeMismatch);
        }
        if io_count(&inputs[0])? != self.batch * self.m * self.k
            || io_count(&inputs[1])? != self.batch * self.k * self.n
        {
            return Err(Error::ShapeMismatch);
        }
        Ok(vec![crate::registry::spec_1d(
            self.batch * self.m * self.n,
        )])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        if inputs.len() != 2 {
            return Err(Error::ShapeMismatch);
        }
        let a_stride = self.m * self.k;
        let b_stride = self.k * self.n;
        let mut output = Vec::with_capacity(self.batch * self.m * self.n);
        let mut raw_all = Vec::new();
        for batch in 0..self.batch {
            let raw = matmul_raw(
                &inputs[0][batch * a_stride..(batch + 1) * a_stride],
                &inputs[1][batch * b_stride..(batch + 1) * b_stride],
                self.m,
                self.n,
                self.k,
            )?;
            for value in &raw {
                output.push(to_i64_checked(requantize(*value, self.shift)?)?);
            }
            raw_all.extend(raw);
        }
        Ok(KernelRecord {
            outputs: vec![output],
            raw_accumulators: vec![raw_all],
            ..KernelRecord::default()
        })
    }
}
