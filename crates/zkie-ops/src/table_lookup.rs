//! Table lookup with checked indices.

use zkie_core::error::{Error, Result};
use zkie_core::types::TensorSpec;

use crate::op::OpCode;
use crate::registry::{io_count, KernelRecord, PureForward};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TableLookupOp;

impl PureForward for TableLookupOp {
    fn opcode(&self) -> OpCode {
        OpCode::TableLookup
    }

    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        if inputs.len() != 2 {
            return Err(Error::ShapeMismatch);
        }
        io_count(&inputs[0])?;
        Ok(vec![crate::registry::spec_1d(io_count(&inputs[1])?)])
    }

    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord> {
        if inputs.len() != 2 {
            return Err(Error::ShapeMismatch);
        }
        let table = &inputs[0];
        let mut output = Vec::with_capacity(inputs[1].len());
        for index in &inputs[1] {
            if *index < 0 || *index as usize >= table.len() {
                return Err(Error::InvalidIndex);
            }
            output.push(table[*index as usize]);
        }
        Ok(KernelRecord::with_output(output))
    }
}
