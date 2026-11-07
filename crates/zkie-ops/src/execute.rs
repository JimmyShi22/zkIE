//! Graph execution entry.
//!
//! Execution is driven by the graph plan and an explicit operator registry.
//! Operators are looked up by opcode; the executor never creates its own thread
//! pool or proof session.

use std::collections::BTreeMap;

use zkie_core::common::field::ProtocolField;
use zkie_core::compute::ExecutionPort;
use zkie_core::error::{Error, Result};
use zkie_core::types::{TensorId, TensorView};

use crate::graph::GraphPlan;
use crate::op::{OpCode, Operation};

pub trait OperationRegistry<F: ProtocolField> {
    fn operation(&self, opcode: OpCode) -> Option<&dyn Operation<F>>;
}

#[derive(Clone, Debug, Default)]
pub struct ExecuteOutcome {
    pub outputs: BTreeMap<TensorId, TensorView>,
}

pub trait GraphExecutor<F: ProtocolField> {
    fn execute(
        &mut self,
        graph: &GraphPlan,
        registry: &dyn OperationRegistry<F>,
        ctx: &mut dyn ExecutionPort<F>,
        inputs: &BTreeMap<TensorId, TensorView>,
    ) -> Result<ExecuteOutcome>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutedStep {
    pub op_id: crate::op::OpId,
    pub opcode: OpCode,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
}

/// Sequential executor in graph order. Completion order is fixed by the plan,
/// so the trace is reproducible.
#[derive(Clone, Debug, Default)]
pub struct SequentialExecutor {
    pub trace: Vec<ExecutedStep>,
}

impl<F: ProtocolField> GraphExecutor<F> for SequentialExecutor {
    fn execute(
        &mut self,
        graph: &GraphPlan,
        registry: &dyn OperationRegistry<F>,
        ctx: &mut dyn ExecutionPort<F>,
        inputs: &BTreeMap<TensorId, TensorView>,
    ) -> Result<ExecuteOutcome> {
        graph.validate()?;
        let mut values: BTreeMap<TensorId, TensorView> = inputs.clone();
        self.trace.clear();
        for node in &graph.nodes {
            let operation = registry
                .operation(node.opcode)
                .ok_or(Error::UnsupportedPrimitive)?;
            let mut node_inputs = Vec::with_capacity(node.inputs.len());
            for id in &node.inputs {
                node_inputs.push(
                    values
                        .get(id)
                        .cloned()
                        .ok_or(Error::ProverStateMissing)?,
                );
            }
            let record = operation.forward(ctx, &node_inputs)?;
            if record.outputs.len() != node.outputs.len() {
                return Err(Error::ShapeMismatch);
            }
            for (id, view) in node.outputs.iter().zip(record.outputs) {
                values.insert(*id, view);
            }
            self.trace.push(ExecutedStep {
                op_id: node.op_id,
                opcode: node.opcode,
                inputs: node.inputs.clone(),
                outputs: node.outputs.clone(),
            });
        }
        Ok(ExecuteOutcome { outputs: values })
    }
}
