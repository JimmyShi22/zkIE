//! Logical graph identity.
//!
//! The graph records tensor specs, operator nodes and edges. It is a plan, not
//! an execution schedule; scheduling belongs to the engine.

use std::collections::{BTreeMap, BTreeSet};

use zkie_core::error::{Error, Result};
use zkie_core::types::{TensorId, TensorSpec};

use crate::op::{OpCode, OpId};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphNode {
    pub op_id: OpId,
    pub opcode: OpCode,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
}

impl GraphNode {
    pub fn new(op_id: OpId, opcode: OpCode) -> Self {
        Self {
            op_id,
            opcode,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorEdge {
    pub from: TensorId,
    pub to: TensorId,
}

#[derive(Clone, Debug, Default)]
pub struct GraphPlan {
    pub tensors: BTreeMap<TensorId, TensorSpec>,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<TensorEdge>,
}

impl GraphPlan {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_tensor(&mut self, id: TensorId, spec: TensorSpec) -> Result<()> {
        spec.validate()?;
        if self.tensors.insert(id, spec).is_some() {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }

    pub fn add_node(&mut self, node: GraphNode) -> Result<()> {
        if self.nodes.iter().any(|existing| existing.op_id == node.op_id) {
            return Err(Error::StatementMismatch);
        }
        self.nodes.push(node);
        Ok(())
    }

    pub fn add_edge(&mut self, edge: TensorEdge) -> Result<()> {
        if !self.tensors.contains_key(&edge.from) || !self.tensors.contains_key(&edge.to) {
            return Err(Error::InvalidIndex);
        }
        self.edges.push(edge);
        Ok(())
    }

    pub fn tensor_spec(&self, id: TensorId) -> Option<&TensorSpec> {
        self.tensors.get(&id)
    }

    pub fn validate(&self) -> Result<()> {
        for spec in self.tensors.values() {
            spec.validate()?;
        }
        let mut op_ids = BTreeSet::new();
        for node in &self.nodes {
            if !op_ids.insert(node.op_id) {
                return Err(Error::StatementMismatch);
            }
            for tensor in node.inputs.iter().chain(node.outputs.iter()) {
                if !self.tensors.contains_key(tensor) {
                    return Err(Error::InvalidIndex);
                }
            }
        }
        for edge in &self.edges {
            if !self.tensors.contains_key(&edge.from) || !self.tensors.contains_key(&edge.to) {
                return Err(Error::InvalidIndex);
            }
        }
        Ok(())
    }
}
