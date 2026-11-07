//! Compose two index maps: `out[i] = inner[outer[i]]`.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IndexComposeOp;

impl PureForward for IndexComposeOp {
    fn opcode(&self) -> OpCode {
        OpCode::IndexCompose
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if inputs.len() != 2 {
            return Err(Error::ShapeMismatch);
        }
        Ok(vec![crate::registry::spec_1d(io_count(&inputs[0])?)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        if inputs.len() != 2 {
            return Err(Error::ShapeMismatch);
        }
        let outer = &inputs[0];
        let inner = &inputs[1];
        let mut output = Vec::with_capacity(outer.len());
        for index in outer {
            if *index < 0 || *index as usize >= inner.len() {
                return Err(Error::InvalidIndex);
            }
            output.push(inner[*index as usize]);
        }
        Ok(KernelRecord::with_output(output))
    }
}
