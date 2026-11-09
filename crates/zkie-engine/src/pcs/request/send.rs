//! Oracle notices and the verifier's table registry (foundation only).
//!
//! A committed table is sent as a notice: its root, its code/row/answer/salt
//! numbers and its leaf class. The verifier records a notice only when it
//! equals the notice the trusted plan expects, and gives it an oracle id that
//! later messages and column queries name. The sampling distances are plan
//! values and are deliberately absent from a notice. This module is not wired
//! into production.

use serde::{Deserialize, Serialize};
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::code::CodeParams;
use crate::pcs::request::table::{
    index_fault, tree_shape_fault, IndexShape, TableDistances,
};

/// The identifier of a sent table in one statement. The verifier assigns ids
/// in arrival order; a proof never chooses one.
pub type OracleId = u64;

/// The public description of a sent table. It carries no distances, record or
/// salt values; those are either plan values or are bound by the table name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleNotice {
    pub root: Digest,
    pub code: CodeParams,
    pub v: u32,
    pub answers: u32,
    pub salt_bytes: usize,
    pub leaf_class: u32,
}

impl OracleNotice {
    pub fn from_shape(root: Digest, shape: &IndexShape, leaf_class: u32) -> Self {
        Self {
            root,
            code: shape.code,
            v: shape.v,
            answers: shape.answers,
            salt_bytes: shape.salt_bytes,
            leaf_class,
        }
    }

    /// The notice's shape numbers agree with a planned shape. Distances are
    /// checked by the plan separately.
    pub fn matches_shape(&self, shape: &IndexShape) -> bool {
        self.code == shape.code
            && self.v == shape.v
            && self.answers == shape.answers
            && self.salt_bytes == shape.salt_bytes
    }

    /// The shape faults a notice can carry, judged without a plan. The
    /// distances are not part of a notice; zero is a placeholder that the
    /// fault checks do not read.
    pub fn fault(&self) -> Option<&'static str> {
        let shape = IndexShape {
            code: self.code,
            v: self.v,
            answers: self.answers,
            salt_bytes: self.salt_bytes,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        };
        index_fault(&shape).or_else(|| tree_shape_fault(&shape))
    }
}

/// The verifier's registry of sent tables.
#[derive(Clone, Debug, Default)]
pub struct OracleRegistry {
    entries: Vec<OracleNotice>,
}

impl OracleRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Receive one notice. The notice must equal the plan's expected notice;
    /// the verifier never takes the root, shape or leaf class from the proof
    /// alone. The id is assigned in arrival order.
    pub fn receive(&mut self, expected: &OracleNotice, sent: &OracleNotice) -> Result<OracleId> {
        if sent != expected {
            return Err(Error::StatementMismatch);
        }
        if sent.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        let id = u64::try_from(self.entries.len()).map_err(|_| Error::ResourceLimit)?;
        self.entries.push(*sent);
        Ok(id)
    }

    pub fn notice(&self, id: OracleId) -> Option<&OracleNotice> {
        self.entries.get(usize::try_from(id).ok()?)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::request::code::build_code_plan;
    use crate::pcs::request::params::{LEAF_SALT_MIN_BYTES, LEAF_SALT_MAX_BYTES};

    fn shape() -> IndexShape {
        IndexShape {
            code: build_code_plan(3, 4, 1).unwrap(),
            v: 2,
            answers: 1,
            salt_bytes: LEAF_SALT_MIN_BYTES,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        }
    }

    fn notice() -> OracleNotice {
        OracleNotice::from_shape(Digest([7; 32]), &shape(), 3)
    }

    #[test]
    fn notices_bind_the_planned_root_and_shape() {
        let notice = notice();
        assert!(notice.matches_shape(&shape()));
        assert!(notice.fault().is_none());
        let mut other = shape();
        other.distances.radius = 9;
        assert!(notice.matches_shape(&other));
        // A different salt length, code or row count is not the plan's shape.
        let mut changed = notice;
        changed.salt_bytes = LEAF_SALT_MIN_BYTES + 8;
        assert!(!changed.matches_shape(&shape()));
        let mut changed = notice;
        changed.v = 3;
        assert!(!changed.matches_shape(&shape()));
        let mut changed = notice;
        changed.code = build_code_plan(4, 4, 1).unwrap();
        assert!(!changed.matches_shape(&shape()));
        let mut changed = notice;
        changed.root = Digest([8; 32]);
        assert_ne!(changed, notice);
        assert!(changed.fault().is_none());
    }

    #[test]
    fn registry_requires_the_plan_notice() {
        let expected = notice();
        let mut registry = OracleRegistry::new();
        let id = registry.receive(&expected, &notice()).unwrap();
        assert_eq!(id, 0);
        assert_eq!(registry.notice(id), Some(&expected));
        assert_eq!(registry.len(), 1);
        assert!(registry.notice(1).is_none());
        assert!(!registry.is_empty());

        // A different root, shape or leaf class is rejected before an id is
        // assigned; the proof cannot self-report any of them.
        let mut registry = OracleRegistry::new();
        let mut sent = notice();
        sent.root = Digest([9; 32]);
        assert!(registry.receive(&expected, &sent).is_err());
        let mut sent = notice();
        sent.v = 3;
        assert!(registry.receive(&expected, &sent).is_err());
        let mut sent = notice();
        sent.leaf_class = 4;
        assert!(registry.receive(&expected, &sent).is_err());
        assert!(registry.is_empty());
    }

    #[test]
    fn invalid_notices_are_rejected_without_panicking() {
        let mut bad = notice();
        bad.v = 64;
        assert!(bad.fault().is_some());
        let mut registry = OracleRegistry::new();
        assert!(registry.receive(&bad, &bad).is_err());
        let mut bad = notice();
        bad.answers = 0;
        assert!(bad.fault().is_some());
        let mut bad = notice();
        bad.salt_bytes = LEAF_SALT_MAX_BYTES + 8;
        assert!(bad.fault().is_some());
    }
}
