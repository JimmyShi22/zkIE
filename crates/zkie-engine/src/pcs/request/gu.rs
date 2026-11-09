//! G_u's SEND notice and its out-of-domain/record/column relation skeleton
//! (foundation only, not wired into production).
//!
//! G_u is a runtime oracle message with its own trusted [`GuOracleShape`]:
//! its own code, main and mask rows and its own sample answer count. The
//! indexed table's shape is a different object and supplies only the record
//! relation. The verifier binds the received root to the frozen tree, part and
//! complete G_u shape, records it as a sampled oracle, and keeps three
//! separate relations: one claim per out-of-domain sample (the seed is drawn
//! after the root and the answer follows), one claim per record answer of the
//! indexed table, and one claim per drawn column. The coefficient rows and
//! their verification are open; this module gives the binding, the capacity
//! check and the target skeleton.

use serde::{Deserialize, Serialize};
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::code::{build_code_plan, CodeParams};
use crate::pcs::request::field::{Fp3, E_DEGREE};
use crate::pcs::request::fold::VerifiedGroup;
use crate::pcs::request::relation::GuRelationRow;
use crate::pcs::request::runtime_tree::RuntimeTreePlan;
use crate::pcs::request::table::{
    index_fault, tree_shape_fault, IndexRecord, IndexShape, SampleSeed,
};
use crate::primitives::codec::CanonicalWriter;

fn ceil_log2(value: u64) -> u32 {
    if value <= 1 {
        0
    } else {
        64 - (value - 1).leading_zeros()
    }
}

/// The trusted shape of the G_u oracle message itself. Its code, rows, mask
/// rows and sample answers are not the indexed table's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuOracleShape {
    pub code: CodeParams,
    pub main_rows: u64,
    /// The main rows are over the cubic extension: 3.
    pub main_deg: u32,
    pub mask_rows: u64,
    pub mask_code: Option<CodeParams>,
    pub answers: u32,
}

impl GuOracleShape {
    pub fn fault(&self) -> Option<&'static str> {
        if build_code_plan(self.code.log_n, self.code.ell, self.code.r).is_err() {
            return Some("the G_u oracle code is not a code");
        }
        if self.main_rows < 1 || self.main_deg != E_DEGREE as u32 {
            return Some("the G_u oracle has no main rows over the extension field");
        }
        if self.answers < 1 {
            return Some("the G_u oracle has no sample answer");
        }
        match (self.mask_rows, self.mask_code) {
            (0, None) => {}
            (0, Some(_)) => return Some("the G_u oracle names a mask code without mask rows"),
            (_, None) => return Some("the G_u oracle has mask rows without a mask code"),
            (_, Some(code)) => {
                if build_code_plan(code.log_n, code.ell, code.r).is_err() {
                    return Some("the G_u mask code is not a code");
                }
                if code.ell < self.answers as u64 {
                    return Some("the G_u mask row has no room for the sample pads");
                }
            }
        }
        None
    }
}

/// The trusted plan's shape of the G_u oracle message: the runtime tree, the
/// part that carries the folded row, G_u's own oracle shape and the indexed
/// table's shape that supplies the record. The root is received, never
/// planned.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuShape {
    pub oracle: u64,
    pub group: u64,
    pub tree: RuntimeTreePlan,
    pub part: u32,
    pub rows_high: bool,
    pub u: GuOracleShape,
    pub index: IndexShape,
}

impl GuShape {
    /// The G_u table's entry count: the folded message, its tail at the next
    /// multiple of the tail's power of two, and the mask-code tail.
    pub fn entries(&self) -> Result<u64> {
        let step = 1u64
            .checked_shl(ceil_log2(self.index.code.r))
            .ok_or(Error::ResourceLimit)?;
        let tail_first = self
            .index
            .code
            .ell
            .checked_add(step - 1)
            .ok_or(Error::ResourceLimit)?
            / step
            * step;
        tail_first
            .checked_add(self.index.code.r)
            .ok_or(Error::ResourceLimit)
    }

    /// The leading entries that lie in real rows and before every main-table
    /// pad of G_u's own sample.
    pub fn vector_room(&self) -> Result<u64> {
        let row_bits = ceil_log2(self.u.main_rows);
        let place_bits = self.u.code.log_ell();
        if !self.rows_high && self.u.main_rows != (1u64 << row_bits) {
            return Ok(0);
        }
        let bits = row_bits.checked_add(place_bits).ok_or(Error::ResourceLimit)?;
        if bits > 60 {
            return Ok(0);
        }
        let mut room = self
            .u
            .main_rows
            .checked_shl(place_bits)
            .ok_or(Error::ResourceLimit)?;
        if self.u.answers > 0 && self.u.mask_rows == 0 {
            if self.u.code.ell < self.u.answers as u64 {
                return Ok(0);
            }
            for answer in 0..self.u.answers {
                let place = self.u.code.ell - self.u.answers as u64 + answer as u64;
                let index = (self.u.main_rows - 1)
                    .checked_add(place.checked_shl(row_bits).ok_or(Error::ResourceLimit)?)
                    .ok_or(Error::ResourceLimit)?;
                let entry_at = if self.rows_high {
                    (index >> row_bits)
                        | ((index & ((1u64 << row_bits) - 1)) << place_bits)
                } else {
                    index
                };
                room = room.min(entry_at);
            }
        }
        Ok(room)
    }

    /// The number of sample-cube coordinates that a G_u seed must carry.
    pub fn sample_bits(&self) -> Result<u32> {
        let main_bits = ceil_log2(self.u.main_rows)
            .checked_add(self.u.code.log_ell())
            .ok_or(Error::ResourceLimit)?;
        let mask_bits = if let Some(code) = self.u.mask_code {
            ceil_log2(self.u.mask_rows)
                .checked_add(code.log_ell())
                .ok_or(Error::ResourceLimit)?
        } else {
            0
        };
        if self.u.main_rows > 0 && self.u.mask_rows > 0 {
            main_bits
                .max(mask_bits)
                .checked_add(1)
                .ok_or(Error::ResourceLimit)
        } else {
            main_bits
                .checked_add(mask_bits)
                .ok_or(Error::ResourceLimit)
        }
    }

    pub fn fault(&self) -> Option<&'static str> {
        if index_fault(&self.index).is_some() || tree_shape_fault(&self.index).is_some() {
            return Some("the record's index shape is not a table");
        }
        if self.u.fault().is_some() {
            return Some("the G_u oracle shape is not an oracle shape");
        }
        if self.tree.fault().is_some() {
            return Some("the G_u tree is not a runtime tree");
        }
        let part = match self.tree.parts.get(self.part as usize) {
            Some(part) => part,
            None => return Some("the G_u part is not in its tree"),
        };
        let main_symbols = match self.u.main_rows.checked_mul(self.u.main_deg as u64) {
            Some(value) => value,
            None => return Some("the G_u main width overflows"),
        };
        let mask_symbols = match self.u.mask_rows.checked_mul(E_DEGREE as u64) {
            Some(value) => value,
            None => return Some("the G_u mask width overflows"),
        };
        if part.main_symbols != main_symbols || part.mask_symbols != mask_symbols {
            return Some("the G_u part's widths are not its oracle shape's");
        }
        if self.tree.log_n != self.u.code.log_n {
            return Some("the G_u tree depth is not the oracle code's depth");
        }
        if let Some(code) = self.u.mask_code {
            if self.tree.log_n_zk != code.log_n {
                return Some("the G_u tree mask depth is not the mask code's depth");
            }
        }
        let entries = match self.entries() {
            Ok(entries) => entries,
            Err(_) => return Some("the G_u entry count overflows"),
        };
        let room = match self.vector_room() {
            Ok(room) => room,
            Err(_) => return Some("the G_u vector room overflows"),
        };
        if entries > room {
            return Some("the folded row does not fit beside the pad of its oracle message");
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
        writer.put_u64(if self.rows_high { 1 } else { 0 });
        writer.put_u64(self.u.code.log_n as u64);
        writer.put_u64(self.u.code.ell);
        writer.put_u64(self.u.code.r);
        writer.put_u64(self.u.main_rows);
        writer.put_u64(self.u.main_deg as u64);
        writer.put_u64(self.u.mask_rows);
        writer.put_u64(if self.u.mask_code.is_some() { 1 } else { 0 });
        if let Some(code) = self.u.mask_code {
            writer.put_u64(code.log_n as u64);
            writer.put_u64(code.ell);
            writer.put_u64(code.r);
        }
        writer.put_u64(self.u.answers as u64);
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

/// The oracle recorded by SEND: the received root and the complete trusted
/// shape it was bound to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuOracle {
    pub oracle: u64,
    pub root: Digest,
    pub sampled: bool,
    pub shape: GuShape,
}

/// The verifier's G_u SEND registry. A root is recorded only when its notice
/// equals the frozen plan's shape; the recorded shape is the authority for
/// every later claim, not the caller's shape argument.
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
        if self
            .entries
            .iter()
            .any(|entry| entry.oracle == planned.oracle)
        {
            return Err(Error::StatementMismatch);
        }
        let oracle = GuOracle {
            oracle: planned.oracle,
            root: notice.root,
            sampled: true,
            shape: planned.clone(),
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

/// The isolated group membership of a sampled G_u oracle. The trusted group
/// supplies the base and mask codes; the trusted G_u shape supplies the
/// oracle, group, tree, part and layout; the registry supplies the received
/// root. The membership records the dynamic join and rejects a second join,
/// a static member with the same oracle number, another group/code/layout or
/// a pending row that names another oracle or group, and a G_u oracle number
/// equal to the group's fresh mask id. The statement and layout sources are
/// trusted fixtures in this isolated layer; this is not a production
/// statement-authentication claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuMembership {
    pub group: u64,
    pub code: CodeParams,
    pub mask_code: Option<CodeParams>,
    pub fresh: Option<u64>,
    pub tree_number: u64,
    pub part: u32,
    pub members: Vec<u64>,
    joined: Option<u64>,
}

impl GuMembership {
    /// Open the membership for a trusted group and a registered G_u shape.
    /// The shape must be the complete registered shape; this only checks the
    /// static group/code/layout relation, not the statement's origin.
    pub fn new(group: &VerifiedGroup, shape: &GuShape, registry: &GuRegistry) -> Result<Self> {
        if shape.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        let oracle = registry.get(shape.oracle).ok_or(Error::StatementMismatch)?;
        if oracle.shape != *shape || !oracle.sampled {
            return Err(Error::StatementMismatch);
        }
        if shape.group != group.group {
            return Err(Error::StatementMismatch);
        }
        let code = group.code.ok_or(Error::StatementMismatch)?;
        if code != shape.u.code || group.mask_code != shape.u.mask_code {
            return Err(Error::StatementMismatch);
        }
        if group.fresh == Some(shape.oracle) {
            return Err(Error::StatementMismatch);
        }
        let mut members = group.members.clone();
        members.sort_unstable();
        if members.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            group: shape.group,
            code,
            mask_code: shape.u.mask_code,
            fresh: group.fresh,
            tree_number: shape.tree.tree_number,
            part: shape.part,
            members,
            joined: None,
        })
    }

    /// Dynamically join the registered G_u oracle. Only one dynamic join is
    /// allowed; an oracle number that is already a static member is a
    /// duplicate. The complete shape, group, code, tree and part must agree.
    pub fn join(&mut self, shape: &GuShape, registry: &GuRegistry) -> Result<()> {
        if self.joined.is_some() {
            return Err(Error::InvalidSessionState);
        }
        let oracle = registry.get(shape.oracle).ok_or(Error::StatementMismatch)?;
        if oracle.shape != *shape || !oracle.sampled {
            return Err(Error::StatementMismatch);
        }
        if shape.group != self.group
            || shape.u.code != self.code
            || shape.u.mask_code != self.mask_code
            || self.fresh == Some(shape.oracle)
            || shape.tree.tree_number != self.tree_number
            || shape.part != self.part
        {
            return Err(Error::StatementMismatch);
        }
        let at = match self.members.binary_search(&shape.oracle) {
            Ok(_) => return Err(Error::StatementMismatch),
            Err(index) => index,
        };
        self.members.insert(at, shape.oracle);
        self.joined = Some(shape.oracle);
        Ok(())
    }

    /// The joined oracle and its complete shape must still be the registry's
    /// authority for every later claim.
    pub fn require(&self, shape: &GuShape, registry: &GuRegistry) -> Result<()> {
        let oracle = registry.get(shape.oracle).ok_or(Error::StatementMismatch)?;
        if oracle.shape != *shape || !oracle.sampled || self.joined != Some(shape.oracle) {
            return Err(Error::StatementMismatch);
        }
        if shape.group != self.group
            || shape.u.code != self.code
            || shape.u.mask_code != self.mask_code
            || self.fresh == Some(shape.oracle)
            || shape.tree.tree_number != self.tree_number
            || shape.part != self.part
            || self.members.binary_search(&shape.oracle).is_err()
        {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }

    /// A pending G_u relation row must belong to the joined oracle and group.
    pub fn check_row(&self, row: &GuRelationRow) -> Result<()> {
        if self.joined != Some(row.oracle) || row.group != self.group {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }
}

/// One out-of-domain sample claim on G_u: the prover's answer after the seed
/// was drawn. The claim's coefficient row is open; this skeleton binds the
/// registered oracle, the answer number, the seed and the target. The answer
/// count is G_u's own, not the record's.
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
        let oracle = registry.get(shape.oracle).ok_or(Error::StatementMismatch)?;
        if oracle.shape != *shape || !oracle.sampled {
            return Err(Error::StatementMismatch);
        }
        if answer >= shape.u.answers {
            return Err(Error::StatementMismatch);
        }
        if seed.beta == Fp3::ZERO {
            return Err(Error::InvalidProof);
        }
        if seed.z.len() < shape.sample_bits()? as usize {
            return Err(Error::ShapeMismatch);
        }
        Ok(Self {
            oracle: oracle.oracle,
            group: oracle.shape.group,
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

/// One record claim on G_u: at the indexed table's row point, answer i folds
/// the record to a public target. The record's answer count is the index's,
/// not G_u's sample count. The coefficient row is open.
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
        let oracle = registry.get(shape.oracle).ok_or(Error::StatementMismatch)?;
        if oracle.shape != *shape || !oracle.sampled {
            return Err(Error::StatementMismatch);
        }
        if answer >= shape.index.answers {
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
            oracle: oracle.oracle,
            group: oracle.shape.group,
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
/// position has the prover's value. The coefficient row is open.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuColumnClaim {
    pub oracle: u64,
    pub group: u64,
    pub position: u64,
    pub value: Fp3,
}

impl GuColumnClaim {
    pub fn new(shape: &GuShape, registry: &GuRegistry, position: u64, value: Fp3) -> Result<Self> {
        let oracle = registry.get(shape.oracle).ok_or(Error::StatementMismatch)?;
        if oracle.shape != *shape || !oracle.sampled {
            return Err(Error::StatementMismatch);
        }
        if position >= shape.index.code.n() {
            return Err(Error::StatementMismatch);
        }
        Ok(Self {
            oracle: oracle.oracle,
            group: oracle.shape.group,
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

/// The bytes of the sample-claim round. Separate from the record and column
/// bytes: the relations are bound in their own messages and are never merged.
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
    use crate::pcs::request::params::LEAF_SALT_MIN_BYTES;
    use crate::pcs::request::relation::GuRelationKind;
    use crate::pcs::request::runtime_tree::{lay_runtime_trees, RuntimeOracleShape};
    use crate::pcs::request::table::TableDistances;

    fn index_shape(log_n: u32, ell: u64, answers: u32) -> IndexShape {
        IndexShape {
            code: build_code_plan(log_n, ell, 1).unwrap(),
            v: 2,
            answers,
            salt_bytes: LEAF_SALT_MIN_BYTES,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        }
    }

    fn tree(main_rows: u64, mask_rows: u64) -> RuntimeTreePlan {
        let layout = lay_runtime_trees(
            &[RuntimeOracleShape {
                leaf_class: 2,
                base_log_n: 3,
                main_rows,
                main_deg: 3,
                mask_rows,
                mask_log_n: if mask_rows > 0 { 2 } else { 0 },
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
            tree: tree(4, 0),
            part: 0,
            rows_high: false,
            u: GuOracleShape {
                code: build_code_plan(3, 4, 1).unwrap(),
                main_rows: 4,
                main_deg: 3,
                mask_rows: 0,
                mask_code: None,
                answers: 1,
            },
            index: index_shape(3, 4, 1),
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

    fn trusted_group(shape: &GuShape) -> VerifiedGroup {
        VerifiedGroup {
            group: shape.group,
            members: vec![1, 2],
            fresh: None,
            code: Some(shape.u.code),
            mask_code: shape.u.mask_code,
        }
    }

    #[test]
    fn gu_membership_joins_once_and_checks_the_trusted_group() {
        let (shape, registry) = registered();
        let group = trusted_group(&shape);
        let mut membership = GuMembership::new(&group, &shape, &registry).unwrap();
        let row = GuRelationRow {
            oracle: shape.oracle,
            group: shape.group,
            kind: GuRelationKind::Record,
            main: Vec::new(),
            mask: Vec::new(),
            target: Fp3::ZERO,
        };
        assert!(GuMembership::check_row(&membership, &row).is_err());
        membership.join(&shape, &registry).unwrap();
        assert!(GuMembership::check_row(&membership, &row).is_ok());
        assert!(membership.join(&shape, &registry).is_err());
        assert!(membership.require(&shape, &registry).is_ok());

        let mut other_group = shape.clone();
        other_group.group = 2;
        assert!(membership.require(&other_group, &registry).is_err());

        let mut wrong_code = group.clone();
        wrong_code.code = Some(build_code_plan(3, 8, 0).unwrap());
        assert!(GuMembership::new(&wrong_code, &shape, &registry).is_err());

        let mut duplicate = group.clone();
        duplicate.members = vec![shape.oracle];
        let mut duplicate_membership = GuMembership::new(&duplicate, &shape, &registry).unwrap();
        assert!(duplicate_membership.join(&shape, &registry).is_err());

        let mut fresh_collision = group.clone();
        fresh_collision.fresh = Some(shape.oracle);
        assert!(GuMembership::new(&fresh_collision, &shape, &registry).is_err());

        let mut rebound = shape.clone();
        rebound.oracle = 10;
        assert!(GuMembership::new(&group, &rebound, &registry).is_err());

        let mut another_tree = shape.clone();
        another_tree.tree.tree_number += 1;
        assert!(GuMembership::new(&group, &another_tree, &registry).is_err());
    }

    #[test]
    fn gu_send_binds_the_root_to_the_complete_shape() {
        let shape = shape();
        assert!(shape.fault().is_none());
        assert_eq!(shape.sample_bits().unwrap(), 4);
        let mut registry = GuRegistry::new();
        let oracle = registry.receive(&shape, &notice()).unwrap();
        assert_eq!(oracle.shape, shape);
        assert!(oracle.sampled);
        assert!(registry.receive(&shape, &notice()).is_err());
        let mut wrong = notice();
        wrong.root = Digest::ZERO;
        assert!(GuRegistry::new().receive(&shape, &wrong).is_err());
        let mut wrong = notice();
        wrong.part = 1;
        assert!(GuRegistry::new().receive(&shape, &wrong).is_err());
    }

    #[test]
    fn gu_shape_faults_follow_the_oracle_and_capacity() {
        let base = shape();
        let mut bad = base.clone();
        bad.u.main_deg = 1;
        assert!(bad.fault().is_some());
        let mut bad = base.clone();
        bad.u.answers = 0;
        assert!(bad.fault().is_some());
        let mut bad = base.clone();
        bad.tree.parts[0].main_symbols = 3;
        assert!(bad.fault().is_some());
        // A folded row that is longer than the room before G_u's own pads.
        let mut bad = base.clone();
        bad.u.main_rows = 1;
        bad.tree = tree(1, 0);
        bad.index = index_shape(3, 4, 1);
        assert!(bad.fault().is_some());
        // The index's depth need not equal G_u's; the capacity, not log_n, decides.
        let mut good = base.clone();
        good.index = index_shape(4, 8, 1);
        assert!(good.fault().is_none());
        // rows_high admits a non-power-of-two row count.
        let mut good = base;
        good.rows_high = true;
        good.u.main_rows = 3;
        good.tree = tree(3, 0);
        assert!(good.fault().is_none());
    }

    #[test]
    fn gu_claims_use_the_registered_shape_not_a_lookalike() {
        let (shape, registry) = registered();
        let mut other = shape.clone();
        other.group = 2;
        let seed = SampleSeed {
            z: vec![Fp3::ONE; shape.sample_bits().unwrap() as usize],
            beta: Fp3::ONE,
        };
        assert!(GuSampleClaim::new(&other, &registry, 0, seed.clone(), Fp3::ONE).is_err());
        let record = IndexRecord {
            y: vec![Fp3::ONE; 4],
        };
        let eq_gamma = vec![Fp3::ONE; 4];
        assert!(GuRecordClaim::new(&other, &registry, &record, &eq_gamma, 0).is_err());
        assert!(GuColumnClaim::new(&other, &registry, 1, Fp3::ONE).is_err());
        assert!(GuSampleClaim::new(&shape, &registry, 0, seed, Fp3::ONE).is_ok());
        assert!(GuRecordClaim::new(&shape, &registry, &record, &eq_gamma, 0).is_ok());
        assert!(GuColumnClaim::new(&shape, &registry, 1, Fp3::ONE).is_ok());
    }

    #[test]
    fn gu_index_and_oracle_answers_stay_separate() {
        let mut shape = shape();
        shape.u.answers = 2;
        shape.index.answers = 1;
        let mut registry = GuRegistry::new();
        registry.receive(&shape, &notice()).unwrap();
        let seed = SampleSeed {
            z: vec![Fp3::ONE; shape.sample_bits().unwrap() as usize],
            beta: Fp3::ONE,
        };
        // G_u has two sample answers.
        assert!(GuSampleClaim::new(&shape, &registry, 1, seed, Fp3::ONE).is_ok());
        // The index record has one answer.
        let record = IndexRecord {
            y: vec![Fp3::ONE; 4],
        };
        let eq_gamma = vec![Fp3::ONE; 4];
        assert!(GuRecordClaim::new(&shape, &registry, &record, &eq_gamma, 1).is_err());
    }

    #[test]
    fn gu_record_and_column_targets_and_separate_bytes() {
        let (shape, registry) = registered();
        let seed = SampleSeed {
            z: vec![Fp3::ONE; shape.sample_bits().unwrap() as usize],
            beta: Fp3::ONE,
        };
        let sample = GuSampleClaim::new(&shape, &registry, 0, seed, Fp3::from_u64(7)).unwrap();
        let record = IndexRecord {
            y: (1..=4).map(Fp3::from_u64).collect(),
        };
        let eq_gamma: Vec<Fp3> = (1..=4).map(Fp3::from_u64).collect();
        let record = GuRecordClaim::new(&shape, &registry, &record, &eq_gamma, 0).unwrap();
        assert_eq!(record.target, Fp3::from_u64(30));
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

    #[test]
    fn gu_overflowing_widths_are_rejected_by_the_registry() {
        let base = shape();
        let mut bad = base.clone();
        bad.u.main_rows = u64::MAX / 3 + 1;
        assert_eq!(bad.fault(), Some("the G_u main width overflows"));
        assert!(GuRegistry::new().receive(&bad, &notice()).is_err());
        let mut bad = base.clone();
        bad.u.mask_rows = u64::MAX / 3 + 1;
        bad.u.mask_code = Some(build_code_plan(3, 4, 1).unwrap());
        assert_eq!(bad.fault(), Some("the G_u mask width overflows"));
        assert!(GuRegistry::new().receive(&bad, &notice()).is_err());
        // The legal shape remains usable.
        assert!(GuRegistry::new().receive(&base, &notice()).is_ok());
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
}
