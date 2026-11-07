//! Shared model asset and ground-truth helpers.
pub mod assets;
pub mod gt;

use std::collections::{BTreeMap, BTreeSet};

use zkie_core::error::{Error, Result};
use zkie_ops::op::OpCode;

/// One model-graph node. Names are the only edge identifiers; every input must
/// be produced by an earlier node or listed as an external input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphNodeSpec {
    pub name: String,
    pub opcode: OpCode,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub scale: Option<u32>,
    pub attrs: BTreeMap<String, String>,
}

impl GraphNodeSpec {
    pub fn new(
        name: impl Into<String>,
        opcode: OpCode,
        inputs: Vec<String>,
        outputs: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            opcode,
            inputs,
            outputs,
            scale: None,
            attrs: BTreeMap::new(),
        }
    }

    pub fn with_scale(mut self, scale: u32) -> Self {
        self.scale = Some(scale);
        self
    }

    pub fn with_attr(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attrs.insert(key.into(), value.into());
        self
    }
}

/// A validated model graph in topological order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelGraph {
    pub model_id: String,
    pub external_inputs: Vec<String>,
    pub nodes: Vec<GraphNodeSpec>,
}

impl ModelGraph {
    pub fn validate(&self) -> Result<()> {
        if self.model_id.is_empty() || self.nodes.is_empty() {
            return Err(Error::InvalidConfig);
        }
        let mut available: BTreeSet<String> = self.external_inputs.iter().cloned().collect();
        let mut produced = BTreeSet::new();
        for node in &self.nodes {
            if node.name.is_empty() || node.outputs.is_empty() {
                return Err(Error::InvalidConfig);
            }
            for input in &node.inputs {
                if !available.contains(input) {
                    return Err(Error::StatementMismatch);
                }
            }
            for output in &node.outputs {
                if !available.insert(output.clone()) || !produced.insert(output.clone()) {
                    return Err(Error::StatementMismatch);
                }
            }
        }
        Ok(())
    }
}
