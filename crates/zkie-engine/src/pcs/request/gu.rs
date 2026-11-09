//! G_u's SEND notice and its out-of-domain/record relation skeleton
//! (foundation only, not wired into production).
//!
//! G_u is a runtime oracle message: its root is the prover's commitment to the
//! folded row, sent in SEND before any seed is drawn. The verifier binds that
//! root to the frozen plan's tree and part, records it as a sampled oracle and
//! then keeps two separate relations: one claim per out-of-domain sample,
//! answered after the seed, and one claim per record answer, folded at the
//! table's row point. The two are distinct messages/rounds and are never
//! merged into one claim set. The coefficient rows and their verification are
//! open; this file gives the binding and target skeleton.

use serde::{Deserialize, Serialize};
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::field::{Fp3, E_DEGREE};
use crate::pcs::request::runtime_tree::RuntimeTreePlan;
use crate::pcs::request::table::{
    index_fault, tree_shape_fault, IndexRecord, IndexShape, SampleSeed,
};
use crate::primitives::codec::CanonicalWriter;

/// The trusted plan's shape of the G_u oracle message: the runtime tree, the
/// part that carries the folded row, and the indexed table's shape whose
/// record supplies the record relation. The tree number, leaf class, depth,
/// widths, salt and starts are plan values; the root is received, never
/// planned.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuShape {
    pub oracle: u64,
    pub group: u64,
    pub tree: RuntimeTreePlan,
    pub part: u32,
    pub index: IndexShape,
}

impl GuShape {
    pub fn main_rows(&self) -> Result<u64> {
        let part = self
            .tree
            .parts
            .get(self.part as usize)
            .ok_or(Error::InvalidIndex)?;
        if part.main_symbols == 0 || part.main_symbols % E_DEGREE as u64 != 0 {
            return Err(Error::InvalidConfig);
        }
        Ok(part.main_symbols / E_DEGREE as u64)
    }

    /// The number of sample-cube coordinates that a G_u seed must carry: the
    /// row bits plus the base code's entry bits.
    pub fn sample_bits(&self) -> Result<u32> {
        let rows = self.main_rows()?;
        let row_bits = if rows <= 1 {
            0
        } else {
            64 - (rows - 1).leading_zeros()
        };
        row_bits
            .checked_add(self.index.code.log_ell())
            .ok_or(Error::ResourceLimit)
    }

    pub fn fault(&self) -> Option<&'static str> {
        if index_fault(&self.index).is_some() || tree_shape_fault(&self.index).is_some() {
            return Some("the G_u index shape is not a table");
        }
        if self.tree.fault().is_some() {
            return Some("the G_u tree is not a runtime tree");
        }
        let part = match self.tree.parts.get(self.part as usize) {
            Some(part) => part,
            None => return Some("the G_u part is not in its tree"),
        };
        if part.main_symbols == 0 {
            return Some("the G_u part has no main symbols");
        }
        if part.main_symbols % E_DEGREE as u64 != 0 {
            return Some("the G_u part is not over the extension field");
        }
        if part.mask_symbols != 0 {
            return Some("the G_u part has mask symbols in this skeleton");
        }
        if self.tree.log_n != self.index.code.log_n {
            return Some("the G_u tree depth is not the table code's depth");
        }
        if self.index.answers < 1 {
            return Some("the G_u table has no record answer");
        }
        None
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(self.oracle);
        writer.put_u64(self.group);
        let tree = self.tree.encode();
        writer.put_u64(tree.len() as u64);
        writer.put_bytes(&tree);
        writer.put_u64(self.part as u64);
        writer.put_u64(self.index.code.log_n as u64);
        writer.put_u64(self.index.code.ell);
        writer.put_u64(self.index.code.r);
        writer.put_u64(self.index.v as u64);
        writer.put_u64(self.index.answers as u64);
        writer.put_u64(self.index.salt_bytes as u64);
        writer.put_u64(self.index.distances.radius);
        writer.put_u64(self.index.distances.later_radius);
        writer.put_u64(self.index.distances.m_once);
        writer.put_u64(self.index.distances.m_later);
        writer.into_bytes()
    }
}

/// The public notice of the G_u SEND message. The root is the only proof
/// value; the oracle/tree/part must equal the frozen plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuNotice {
    pub root: Digest,
    pub oracle: u64,
    pub tree_number: u64,
    pub part: u32,
}

impl GuNotice {
    pub fn fault(&self) -> Option<&'static str> {
        if self.root == Digest::ZERO {
            return Some("the G_u notice has no root");
        }
        None
    }

    pub fn matches(&self, shape: &GuShape) -> bool {
        self.oracle == shape.oracle
            && self.tree_number == shape.tree.tree_number
            && self.part == shape.part
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_digest(&self.root);
        writer.put_u64(self.oracle);
        writer.put_u64(self.tree_number);
        writer.put_u64(self.part as u64);
        writer.into_bytes()
    }
}

/// The oracle recorded by SEND: the received root, bound to the frozen shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuOracle {
    pub oracle: u64,
    pub group: u64,
    pub tree_number: u64,
    pub part: u32,
    pub root: Digest,
    pub answers: u32,
    pub main_rows: u64,
    pub sampled: bool,
}

/// The verifier's G_u SEND registry. A root is recorded only when its notice
/// equals the frozen plan's shape; the root is then that oracle's root for the
/// sample and record relations.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GuRegistry {
    entries: Vec<GuOracle>,
}

impl GuRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn receive(&mut self, planned: &GuShape, notice: &GuNotice) -> Result<GuOracle> {
        if planned.fault().is_some() || notice.fault().is_some() || !notice.matches(planned) {
            return Err(Error::StatementMismatch);
        }
        if self.entries.iter().any(|entry| entry.oracle == planned.oracle) {
            return Err(Error::StatementMismatch);
        }
        let oracle = GuOracle {
            oracle: planned.oracle,
            group: planned.group,
            tree_number: planned.tree.tree_number,
            part: planned.part,
            root: notice.root,
            answers: planned.index.answers,
            main_rows: planned.main_rows()?,
            sampled: true,
        };
        self.entries.push(oracle.clone());
        Ok(oracle)
    }

    pub fn get(&self, oracle: u64) -> Option<&GuOracle> {
        self.entries.iter().find(|entry| entry.oracle == oracle)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// One out-of-domain sample claim on G_u: the prover's answer after the seed
/// was drawn. The claim's coefficient row (eq over the sample cube, with the
/// pad at beta) is open; this skeleton binds the oracle, the answer number,
/// the seed and the target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuSampleClaim {
    pub oracle: u64,
    pub group: u64,
    pub answer: u32,
    pub seed: SampleSeed,
    pub target: Fp3,
}

impl GuSampleClaim {
    pub fn new(
        shape: &GuShape,
        registry: &GuRegistry,
        answer: u32,
        seed: SampleSeed,
        target: Fp3,
    ) -> Result<Self> {
        if shape.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        let oracle = registry.get(shape.oracle).ok_or(Error::StatementMismatch)?;
        if !oracle.sampled || answer >= shape.index.answers {
            return Err(Error::StatementMismatch);
        }
        if seed.beta == Fp3::ZERO {
            return Err(Error::InvalidProof);
        }
        if seed.z.len() < shape.sample_bits()? as usize {
            return Err(Error::ShapeMismatch);
        }
        Ok(Self {
            oracle: shape.oracle,
            group: shape.group,
            answer,
            seed,
            target,
        })
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(self.oracle);
        writer.put_u64(self.group);
        writer.put_u64(self.answer as u64);
        writer.put_u64(self.seed.z.len() as u64);
        for value in &self.seed.z {
            writer.put_fp3(value);
        }
        writer.put_fp3(&self.seed.beta);
        writer.put_fp3(&self.target);
        writer.into_bytes()
    }
}

/// One record claim on G_u: at the table's row point, answer i folds the
/// record to a public target. The claim's coefficient row is open; the target
/// and its oracle binding are the skeleton.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuRecordClaim {
    pub oracle: u64,
    pub group: u64,
    pub answer: u32,
    pub target: Fp3,
}

impl GuRecordClaim {
    pub fn new(
        shape: &GuShape,
        registry: &GuRegistry,
        record: &IndexRecord,
        eq_gamma: &[Fp3],
        answer: u32,
    ) -> Result<Self> {
        if shape.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        let oracle = registry.get(shape.oracle).ok_or(Error::StatementMismatch)?;
        if !oracle.sampled || answer >= shape.index.answers {
            return Err(Error::StatementMismatch);
        }
        let rows = usize::try_from(shape.index.rows()).map_err(|_| Error::ResourceLimit)?;
        let answers = shape.index.answers as usize;
        if record.y.len() != rows.checked_mul(answers).ok_or(Error::ResourceLimit)? {
            return Err(Error::ShapeMismatch);
        }
        if eq_gamma.len() != rows {
            return Err(Error::ShapeMismatch);
        }
        let mut target = Fp3::ZERO;
        for (row, weight) in eq_gamma.iter().enumerate() {
            target = target.add(&weight.mul(&record.y[row * answers + answer as usize]));
        }
        Ok(Self {
            oracle: shape.oracle,
            group: shape.group,
            answer,
            target,
        })
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(self.oracle);
        writer.put_u64(self.group);
        writer.put_u64(self.answer as u64);
        writer.put_fp3(&self.target);
        writer.into_bytes()
    }
}

/// One column-query claim on G_u: the folded row's codeword at the drawn
/// position has the prover's value. The claim's coefficient row is open; the
/// oracle, position and value are the skeleton. It is a third message/round,
/// separate from both the sample and record claims.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuColumnClaim {
    pub oracle: u64,
    pub group: u64,
    pub position: u64,
    pub value: Fp3,
}

impl GuColumnClaim {
    pub fn new(shape: &GuShape, registry: &GuRegistry, position: u64, value: Fp3) -> Result<Self> {
        if shape.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        let oracle = registry.get(shape.oracle).ok_or(Error::StatementMismatch)?;
        if !oracle.sampled || position >= shape.index.code.n() {
            return Err(Error::StatementMismatch);
        }
        Ok(Self {
            oracle: shape.oracle,
            group: shape.group,
            position,
            value,
        })
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(self.oracle);
        writer.put_u64(self.group);
        writer.put_u64(self.position);
        writer.put_fp3(&self.value);
        writer.into_bytes()
    }
}

const SAMPLE_CLAIMS_LABEL: u64 = 1;
const RECORD_CLAIMS_LABEL: u64 = 2;
const COLUMN_CLAIMS_LABEL: u64 = 3;

/// The bytes of the sample-claim round. Separate from the record bytes: the
/// two relations are bound in their own messages and are never merged.
pub fn encode_sample_claims(claims: &[GuSampleClaim]) -> Vec<u8> {
    let mut writer = CanonicalWriter::new();
    writer.put_u64(SAMPLE_CLAIMS_LABEL);
    writer.put_u64(claims.len() as u64);
    for claim in claims {
        let bytes = claim.encode();
        writer.put_u64(bytes.len() as u64);
        writer.put_bytes(&bytes);
    }
    writer.into_bytes()
}

/// The bytes of the record-claim round, separate from the sample bytes.
pub fn encode_record_claims(claims: &[GuRecordClaim]) -> Vec<u8> {
    let mut writer = CanonicalWriter::new();
    writer.put_u64(RECORD_CLAIMS_LABEL);
    writer.put_u64(claims.len() as u64);
    for claim in claims {
        let bytes = claim.encode();
        writer.put_u64(bytes.len() as u64);
        writer.put_bytes(&bytes);
    }
    writer.into_bytes()
}

/// The bytes of the column-query round, separate from the sample and record
/// bytes.
pub fn encode_column_claims(claims: &[GuColumnClaim]) -> Vec<u8> {
    let mut writer = CanonicalWriter::new();
    writer.put_u64(COLUMN_CLAIMS_LABEL);
    writer.put_u64(claims.len() as u64);
    for claim in claims {
        let bytes = claim.encode();
        writer.put_u64(bytes.len() as u64);
        writer.put_bytes(&bytes);
    }
    writer.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::request::code::build_code_plan;
    use crate::pcs::request::params::LEAF_SALT_MIN_BYTES;
    use crate::pcs::request::runtime_tree::{
        lay_runtime_trees, RuntimeOracleShape,
    };
    use crate::pcs::request::table::TableDistances;

    fn index_shape() -> IndexShape {
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

    fn tree() -> RuntimeTreePlan {
        let layout = lay_runtime_trees(
            &[RuntimeOracleShape {
                leaf_class: 2,
                base_log_n: 3,
                main_rows: 1,
                main_deg: 3,
                mask_rows: 0,
                mask_log_n: 0,
            }],
            5,
            LEAF_SALT_MIN_BYTES,
            Digest([9; 32]),
            Digest([7; 32]),
        )
        .unwrap();
        layout.plans.into_iter().next().unwrap()
    }

    fn shape() -> GuShape {
        GuShape {
            oracle: 9,
            group: 1,
            tree: tree(),
            part: 0,
            index: index_shape(),
        }
    }

    fn notice() -> GuNotice {
        GuNotice {
            root: Digest([4; 32]),
            oracle: 9,
            tree_number: 5,
            part: 0,
        }
    }

    fn registered() -> (GuShape, GuRegistry) {
        let shape = shape();
        let mut registry = GuRegistry::new();
        registry.receive(&shape, &notice()).unwrap();
        (shape, registry)
    }

    #[test]
    fn gu_send_binds_the_root_to_the_planned_tree() {
        let shape = shape();
        assert!(shape.fault().is_none());
        assert_eq!(shape.main_rows().unwrap(), 1);
        assert_eq!(shape.sample_bits().unwrap(), 2);
        let mut registry = GuRegistry::new();
        let oracle = registry.receive(&shape, &notice()).unwrap();
        assert_eq!(oracle.oracle, 9);
        assert_eq!(oracle.group, 1);
        assert_eq!(oracle.tree_number, 5);
        assert_eq!(oracle.part, 0);
        assert_eq!(oracle.root, Digest([4; 32]));
        assert_eq!(oracle.answers, 1);
        assert!(oracle.sampled);
        // Duplicate registration and a mismatched notice are rejected.
        assert!(registry.receive(&shape, &notice()).is_err());
        let mut wrong = notice();
        wrong.oracle = 10;
        assert!(GuRegistry::new().receive(&shape, &wrong).is_err());
        let mut wrong = notice();
        wrong.tree_number = 6;
        assert!(GuRegistry::new().receive(&shape, &wrong).is_err());
        let mut wrong = notice();
        wrong.part = 1;
        assert!(GuRegistry::new().receive(&shape, &wrong).is_err());
        let mut wrong = notice();
        wrong.root = Digest::ZERO;
        assert!(GuRegistry::new().receive(&shape, &wrong).is_err());
    }

    #[test]
    fn gu_notice_bytes_bind_root_and_shape() {
        let base = notice().encode();
        let mut other = notice();
        other.root = Digest([5; 32]);
        assert_ne!(base, other.encode());
        let mut other = notice();
        other.oracle = 10;
        assert_ne!(base, other.encode());
        let mut other = notice();
        other.tree_number = 6;
        assert_ne!(base, other.encode());
        let mut other = notice();
        other.part = 1;
        assert_ne!(base, other.encode());
    }

    #[test]
    fn gu_shape_faults_are_rejected() {
        let base = shape();
        let mut bad = base.clone();
        bad.part = 1;
        assert!(bad.fault().is_some());
        let mut bad = base.clone();
        bad.tree.parts[0].main_symbols = 4;
        assert!(bad.fault().is_some());
        let mut bad = base.clone();
        bad.tree.parts[0].mask_symbols = 3;
        assert!(bad.fault().is_some());
        let mut bad = base;
        bad.index.answers = 0;
        assert!(bad.fault().is_some());
        assert!(GuRegistry::new().receive(&bad, &notice()).is_err());
    }

    #[test]
    fn gu_sample_claim_binds_its_round_and_rejects_bad_seeds() {
        let (shape, registry) = registered();
        let seed = SampleSeed {
            z: vec![Fp3::ONE; shape.sample_bits().unwrap() as usize],
            beta: Fp3::ONE,
        };
        let claim = GuSampleClaim::new(&shape, &registry, 0, seed, Fp3::from_u64(7)).unwrap();
        assert_eq!(claim.oracle, 9);
        assert_eq!(claim.answer, 0);
        assert_eq!(claim.target, Fp3::from_u64(7));
        let mut short = SampleSeed {
            z: vec![Fp3::ONE; 1],
            beta: Fp3::ONE,
        };
        assert!(GuSampleClaim::new(&shape, &registry, 0, short.clone(), Fp3::ZERO).is_err());
        short.z = vec![Fp3::ONE; 2];
        short.beta = Fp3::ZERO;
        assert!(GuSampleClaim::new(&shape, &registry, 0, short, Fp3::ZERO).is_err());
        let good = SampleSeed {
            z: vec![Fp3::ONE; 2],
            beta: Fp3::ONE,
        };
        assert!(GuSampleClaim::new(&shape, &registry, 1, good, Fp3::ZERO).is_err());
        assert!(GuSampleClaim::new(&shape, &GuRegistry::new(), 0, {
            SampleSeed {
                z: vec![Fp3::ONE; 2],
                beta: Fp3::ONE,
            }
        }, Fp3::ZERO)
        .is_err());
    }

    #[test]
    fn gu_record_claim_folds_each_answer_separately() {
        let (shape, registry) = registered();
        let record = IndexRecord {
            y: (1..=4).map(Fp3::from_u64).collect(),
        };
        let eq_gamma: Vec<Fp3> = (1..=4).map(Fp3::from_u64).collect();
        let claim = GuRecordClaim::new(&shape, &registry, &record, &eq_gamma, 0).unwrap();
        assert_eq!(claim.target, Fp3::from_u64(30));
        assert_eq!(claim.oracle, 9);
        assert_eq!(claim.group, 1);
        // Wrong record length, wrong point length and an answer out of range.
        let short = IndexRecord {
            y: vec![Fp3::ONE; 3],
        };
        assert!(GuRecordClaim::new(&shape, &registry, &short, &eq_gamma, 0).is_err());
        assert!(GuRecordClaim::new(&shape, &registry, &record, &eq_gamma[..3], 0).is_err());
        assert!(GuRecordClaim::new(&shape, &registry, &record, &eq_gamma, 1).is_err());
    }

    #[test]
    fn gu_sample_and_record_bytes_stay_separate() {
        let (shape, registry) = registered();
        let seed = SampleSeed {
            z: vec![Fp3::ONE; 2],
            beta: Fp3::ONE,
        };
        let sample = GuSampleClaim::new(&shape, &registry, 0, seed, Fp3::from_u64(7)).unwrap();
        let record = IndexRecord {
            y: (1..=4).map(Fp3::from_u64).collect(),
        };
        let eq_gamma: Vec<Fp3> = (1..=4).map(Fp3::from_u64).collect();
        let record = GuRecordClaim::new(&shape, &registry, &record, &eq_gamma, 0).unwrap();
        let column = GuColumnClaim::new(&shape, &registry, 1, Fp3::from_u64(9)).unwrap();
        assert!(GuColumnClaim::new(&shape, &registry, 8, Fp3::ZERO).is_err());
        let sample_bytes = encode_sample_claims(&[sample.clone()]);
        let record_bytes = encode_record_claims(&[record.clone()]);
        let column_bytes = encode_column_claims(&[column.clone()]);
        assert_ne!(sample_bytes, record_bytes);
        assert_ne!(sample_bytes, column_bytes);
        assert_ne!(record_bytes, column_bytes);
        assert_eq!(sample_bytes, encode_sample_claims(&[sample]));
        assert_eq!(record_bytes, encode_record_claims(&[record]));
        assert_eq!(column_bytes, encode_column_claims(&[column]));
    }
}
