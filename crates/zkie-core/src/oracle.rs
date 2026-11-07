//! Oracle identity.
//!
//! An oracle is a logical tensor plus its declared encoding, layout, padding and
//! source relation. The identity is independent of storage location.

use crate::claim::RelationId;
use crate::error::Result;
use crate::types::{TensorId, TensorSpec};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub struct OracleId(pub u64);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OracleBinding {
    pub tensor_id: TensorId,
    pub spec: TensorSpec,
    pub source: RelationId,
}

impl OracleBinding {
    pub fn validate(&self) -> Result<()> {
        self.spec.validate()
    }
}
