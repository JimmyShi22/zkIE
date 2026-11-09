//! FOLDEQ session skeleton (foundation only).
//!
//! The verifier's plan holds the committed table, the fold shape, the runtime
//! mask-tree layout, the group registry and the budgets/counts; every number
//! comes from the verified statement/plan, not from a proof. The session fixes
//! the order: the complete first query, the record seeds, the gamma challenge,
//! the G_u SEND root, the column query round, the column reads and the FOLD
//! start message. The G_u table/OOD relations, the FOLD round polynomials and
//! the table test are not in this skeleton and no closure is claimed.

use std::sync::{Arc, Mutex};

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::chain::{
    Chain, KIND_ALGEBRAIC, KIND_EPSILON, KIND_POSITIONS, KIND_SEAL,
};
use crate::pcs::request::field::{Fp, Fp3};
use crate::pcs::request::runtime_tree::MaskColumnOpening;
use crate::pcs::request::fold::{
    fold_mask_shape_fault, fold_start_verify, FoldShape, FoldStartMessage, RuntimeOracles,
    VerifiedStatement,
};
use crate::pcs::request::read::{draw_positions, HidingCharge, HidingState, ReadLedger, VerifierReads};
use crate::pcs::request::table::{
    record_seeds, ColumnOpening, IndexName, IndexReader, SampleSeed,
};
use crate::primitives::codec::CanonicalWriter;
use crate::primitives::transcript::chain_start;

/// The binder that both sides build from the complete first query. A session
/// accepts only a chain whose label and start value are the binder's; a
/// pre-built chain from another prefix is rejected.
#[derive(Clone, Debug)]
pub struct FoldeqBinder {
    label: u64,
    parameters: Digest,
    first_query: Vec<u8>,
    start: Digest,
}

impl FoldeqBinder {
    pub fn new(label: u64, parameters: Digest, plan: &FoldeqPlan) -> Result<Self> {
        if plan.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        if parameters != plan.reader.configuration {
            return Err(Error::StatementMismatch);
        }
        let first_query = plan.first_query();
        let start = chain_start(label, &parameters, &first_query, &[], None)?;
        Ok(Self {
            label,
            parameters,
            first_query,
            start,
        })
    }

    pub fn first_query(&self) -> &[u8] {
        &self.first_query
    }

    pub fn start(&self) -> &Digest {
        &self.start
    }

    pub fn chain(&self) -> Result<Chain> {
        Chain::begin(self.label, &self.parameters, &self.first_query, &[], None)
    }
}

/// The provider's G_u message: the root the provider sends after its durable
/// hiding pre-charge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuMessage {
    pub root: Digest,
}

/// The provider's side of the G_u/column phase. `begin` must charge the
/// provider hiding state before `send_gu` returns a message; an uncharged
/// provider cannot send. The charge is in-memory here; durable cross-process
/// storage and rollback protection remain open.
pub struct ProverFoldeq {
    hiding: HidingState,
    table: Digest,
    lifecycle: u64,
    columns: usize,
    charged: bool,
}

impl ProverFoldeq {
    /// Bind the provider state to the actual table root, the lifecycle and the
    /// plan's column count. The maximum is still a caller-provided value; its
    /// derivation from the encoded tail/opening budgets is open.
    pub fn new(
        hiding: HidingState,
        reader: &IndexReader,
        lifecycle: u64,
        columns: usize,
    ) -> Result<Self> {
        if hiding.table() != reader.root
            || hiding.lifecycle() != lifecycle
            || columns == 0
        {
            return Err(Error::StatementMismatch);
        }
        Ok(Self {
            hiding,
            table: reader.root,
            lifecycle,
            columns,
            charged: false,
        })
    }

    /// Charge the provider hiding state before G_u and any column exposure.
    /// This is an in-memory update, not a durable pre-charge; the durable
    /// provider ledger (IndexHolder-like) is still open. A second call fails;
    /// an over-budget charge fails before any send. The returned token is
    /// required by the charged column-open path.
    pub fn begin(&mut self) -> Result<HidingCharge> {
        if self.charged {
            return Err(Error::InvalidSessionState);
        }
        let charge = self
            .hiding
            .begin_charge(self.table, self.lifecycle, self.columns)?;
        self.charged = true;
        Ok(charge)
    }

    /// Return the G_u message only after the hiding pre-charge.
    pub fn send_gu(&self, root: &Digest) -> Result<GuMessage> {
        if !self.charged || *root == Digest::ZERO {
            return Err(Error::InvalidSessionState);
        }
        Ok(GuMessage { root: *root })
    }

    pub fn used(&self) -> u64 {
        self.hiding.used()
    }
}

/// A LIN-style gamma: the suffix of a caller-drawn psi, starting at `k_lo`.
/// The FOLDEQ session does not assume this caller; a direct gamma is also
/// accepted. The caller owns the psi draw and its round.
fn lin_gamma(psi: &[Fp3], k_lo: u32, columns: u32) -> Result<Vec<Fp3>> {
    let start = k_lo as usize;
    let end = start
        .checked_add(columns as usize)
        .ok_or(Error::ResourceLimit)?;
    if end > psi.len() {
        return Err(Error::ShapeMismatch);
    }
    Ok(psi[start..end].to_vec())
}

/// The trusted challenge path of one FOLDEQ plan. A LIN caller fixes the
/// LinShape's k_lo/k_col; a caller with another reference path fixes the
/// direct length. The plan chooses the path, never the proof or an untrusted
/// caller, and the first query binds these numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GammaPath {
    Lin {
        k_lo: u32,
        k_col: u32,
        lin_shape: Digest,
    },
    Direct {
        columns: u32,
    },
}

/// The verifier's FOLDEQ plan. The reader is the committed table, the fold and
/// runtime mask-tree layouts and the group registry come from the statement,
/// and the seed/read budgets and column draw count come from the plan. None of
/// these is taken from a proof. The sampling distances are the first-query
/// exception: they stay plan values and are not in the first-query bytes.
#[derive(Clone, Debug)]
pub struct FoldeqPlan {
    pub reader: IndexReader,
    pub fold: FoldShape,
    pub statement: Arc<VerifiedStatement>,
    pub oracles: Arc<Mutex<RuntimeOracles>>,
    pub mask_tree_number: u64,
    pub group_id: u64,
    pub leaf_class: u32,
    pub seed_budget: usize,
    pub ledger: Arc<Mutex<ReadLedger>>,
    pub column_draws: usize,
    pub mask_positions: Vec<u64>,
    pub gamma_path: GammaPath,
}

impl FoldeqPlan {
    pub fn fault(&self) -> Option<&'static str> {
        if let Some(fault) = self.reader.fault() {
            return Some(fault);
        }
        if let Some(fault) = fold_mask_shape_fault(&self.fold) {
            return Some(fault);
        }
        let tree = match self.statement.mask_tree(self.mask_tree_number) {
            Some(tree) => tree,
            None => return Some("the plan names a runtime tree the statement does not hold"),
        };
        if let Some(fault) = tree.fault() {
            return Some(fault);
        }
        let rows = match self.fold.mask_rows() {
            Ok(rows) => rows,
            Err(_) => return Some("the fold shape has no mask rows"),
        };
        if tree.mask_rows != rows
            || tree.log_n != self.fold.mask_code.log_n
            || tree.leaf_class != self.fold.mask_leaf_class
        {
            return Some("the runtime mask tree does not match the fold shape");
        }
        let group = match self.statement.groups().get(self.group_id) {
            Some(group) => group,
            None => return Some("the plan names a group the verifier does not hold"),
        };
        if group.members.is_empty() {
            return Some("the plan's group has no members");
        }
        if self.column_draws == 0 {
            return Some("no column is drawn");
        }
        let draws = match u64::try_from(self.column_draws) {
            Ok(draws) => draws,
            Err(_) => return Some("the column draw count does not fit"),
        };
        let max_reads = match self.ledger.lock() {
            Ok(ledger) => ledger.max_reads(),
            Err(_) => return Some("the read ledger is poisoned"),
        };
        if draws > max_reads {
            return Some("the column draws exceed the read budget");
        }
        if self.mask_positions.is_empty() {
            return Some("no mask position is fixed");
        }
        let columns = tree.columns();
        let mut previous = None;
        for position in &self.mask_positions {
            if *position >= columns {
                return Some("a mask position is outside the runtime tree");
            }
            if let Some(previous) = previous {
                if *position <= previous {
                    return Some("the mask positions are not sorted unique");
                }
            }
            previous = Some(*position);
        }
        let answers = self.reader.shape.answers as usize;
        if self.seed_budget < answers {
            return Some("the seed budget is below the answer count");
        }
        match self.gamma_path {
            GammaPath::Lin { k_lo, k_col, lin_shape } => {
                if lin_shape == Digest::ZERO {
                    return Some("the LIN challenge path has no caller shape binding");
                }
                if k_lo > 30 || k_col > 60 {
                    return Some("the LIN challenge path has too many coordinates");
                }
                let expected = match k_lo.checked_add(self.reader.shape.v) {
                    Some(expected) => expected,
                    None => return Some("the LIN challenge path overflows"),
                };
                if k_col != expected {
                    return Some("the LIN challenge path does not end at the table width");
                }
            }
            GammaPath::Direct { columns } => {
                if columns != self.reader.shape.v {
                    return Some("the direct challenge length is not the table width");
                }
            }
        }
        None
    }

    /// The bytes the first query binds: the complete table name (commitment and
    /// record), before any challenge that reads it. Distances are not part of
    /// the first query; they stay plan values.
    pub fn first_query(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        // The complete table name: commitment and record.
        writer.put_bytes(
            &IndexName {
                commitment: self.reader.commitment(),
                record: self.reader.record.clone(),
            }
            .encode(),
        );
        // The trusted table and fold shapes. Distances are the first-query
        // exception and stay plan values; the configuration is the binder's
        // parameters digest, not this statement.
        for value in [
            self.reader.shape.code.log_n as u64,
            self.reader.shape.code.ell,
            self.reader.shape.code.r,
            self.reader.shape.v as u64,
            self.reader.shape.answers as u64,
            self.reader.shape.salt_bytes as u64,
            self.fold.code.log_n as u64,
            self.fold.code.ell,
            self.fold.code.r,
            self.fold.mask_code.log_n as u64,
            self.fold.mask_code.ell,
            self.fold.mask_code.r,
            self.fold.k as u64,
            self.fold.h as u64,
            self.fold.mask_leaf_class as u64,
        ] {
            writer.put_u64(value);
        }
        // The verified statement (groups, runtime trees and their starts), the
        // expected tree/group/leaf numbers and the complete query plan.
        writer.put_digest(self.statement.id());
        for value in [
            self.mask_tree_number,
            self.group_id,
            self.leaf_class as u64,
            self.seed_budget as u64,
            self.column_draws as u64,
        ] {
            writer.put_u64(value);
        }
        let max_reads = self
            .ledger
            .lock()
            .map(|ledger| ledger.max_reads())
            .unwrap_or(0);
        writer.put_u64(max_reads);
        writer.put_u64(self.mask_positions.len() as u64);
        for position in &self.mask_positions {
            writer.put_u64(*position);
        }
        match self.gamma_path {
            GammaPath::Lin {
                k_lo,
                k_col,
                lin_shape,
            } => {
                writer.put_u64(1);
                writer.put_u64(k_lo as u64);
                writer.put_u64(k_col as u64);
                writer.put_digest(&lin_shape);
            }
            GammaPath::Direct { columns } => {
                writer.put_u64(2);
                writer.put_u64(columns as u64);
            }
        }
        writer.into_bytes()
    }

    pub fn check_first_query(&self, sent: &[u8]) -> Result<()> {
        if sent != self.first_query() {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }

    pub fn seeds(&self) -> Result<Vec<SampleSeed>> {
        record_seeds(&self.reader.shape, &self.reader.name(), self.seed_budget)
    }

    pub fn draw_queries(&self, chain: &mut Chain) -> Result<Vec<u64>> {
        draw_positions(chain, self.column_draws, self.reader.shape.code.log_n)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    FirstQuery,
    Seeds,
    Gamma,
    Gu,
    Queried,
    Read,
    FoldStarted,
    MaskOpened,
}

/// The verifier's FOLDEQ session. It owns the chain, the plan, the mask spend
/// ledger and the read store; the prover's public messages and openings are
/// passed in. This is a joint-development harness, not a production verifier.
pub struct FoldeqSession {
    plan: FoldeqPlan,
    chain: Chain,
    reads: VerifierReads,
    stage: Stage,
    gamma: Option<Vec<Fp3>>,
    gu_root: Option<Digest>,
    start: Option<FoldStartMessage>,
    epsilon: Option<Fp3>,
    queries: Vec<u64>,
}

impl FoldeqSession {
    /// Open a session only from a chain built by the binder. A chain from a
    /// different first-query prefix, or one that has already advanced, is
    /// rejected.
    pub fn open(plan: FoldeqPlan, binder: FoldeqBinder, chain: Chain) -> Result<Self> {
        if plan.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        if binder.first_query() != plan.first_query().as_slice()
            || chain.label() != binder.label
            || chain.value() != binder.start()
            || chain.rounds() != 0
        {
            return Err(Error::StatementMismatch);
        }
        let reads = VerifierReads::new(Arc::clone(&plan.ledger));
        Ok(Self {
            plan,
            chain,
            reads,
            stage: Stage::FirstQuery,
            gamma: None,
            gu_root: None,
            start: None,
            epsilon: None,
            queries: Vec::new(),
        })
    }

    pub fn plan(&self) -> &FoldeqPlan {
        &self.plan
    }

    pub fn queries(&self) -> &[u64] {
        &self.queries
    }

    pub fn gamma(&self) -> Option<&[Fp3]> {
        self.gamma.as_deref()
    }

    pub fn gu_root(&self) -> Option<&Digest> {
        self.gu_root.as_ref()
    }

    pub fn epsilon(&self) -> Option<Fp3> {
        self.epsilon
    }

    pub fn start(&self) -> Option<&FoldStartMessage> {
        self.start.as_ref()
    }

    pub fn bind_first_query(&mut self, sent: &[u8]) -> Result<()> {
        if self.stage != Stage::FirstQuery {
            return Err(Error::InvalidSessionState);
        }
        self.plan.check_first_query(sent)?;
        self.stage = Stage::Seeds;
        Ok(())
    }

    pub fn seeds(&self) -> Result<Vec<SampleSeed>> {
        if self.stage != Stage::Seeds {
            return Err(Error::InvalidSessionState);
        }
        self.plan.seeds()
    }

    /// Draw the challenge along the plan's fixed path. The path and its
    /// numbers are trusted-plan values bound by the first query; the caller
    /// cannot choose LIN versus direct or replace the offset, and the session
    /// owns the draw so no caller value or forged psi source is accepted.
    pub fn draw_gamma(&mut self) -> Result<()> {
        if self.stage != Stage::Seeds {
            return Err(Error::InvalidSessionState);
        }
        let gamma = match self.plan.gamma_path {
            GammaPath::Lin { k_lo, k_col, .. } => {
                let psi = self.draw_challenge(k_col as usize)?;
                lin_gamma(&psi, k_lo, self.plan.reader.shape.v)?
            }
            GammaPath::Direct { columns } => self.draw_challenge(columns as usize)?,
        };
        self.set_gamma(gamma)
    }

    fn draw_challenge(&mut self, count: usize) -> Result<Vec<Fp3>> {
        let prefix = self.chain.message().prefix(KIND_ALGEBRAIC, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| Error::ResourceLimit)?;
        for _ in 0..count {
            values.push(self.chain.draw_e()?);
        }
        Ok(values)
    }

    fn set_gamma(&mut self, gamma: Vec<Fp3>) -> Result<()> {
        self.gamma = Some(gamma);
        self.stage = Stage::Gamma;
        Ok(())
    }

    /// Bind the prover's G_u SEND root. The root is a public message value;
    /// the prover's folded table and its OOD/record relations are verified by
    /// the next stages and remain open.
    pub fn send_gu(&mut self, root: &Digest) -> Result<()> {
        if self.stage != Stage::Gamma {
            return Err(Error::InvalidSessionState);
        }
        if *root == Digest::ZERO {
            return Err(Error::InvalidProof);
        }
        let mut message = self.chain.message();
        for chunk in root.0.chunks(8) {
            let word = u64::from_le_bytes(chunk.try_into().map_err(|_| Error::InvalidEncoding)?);
            message.push_u64(word);
        }
        let prefix = message.prefix(KIND_SEAL, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        self.gu_root = Some(*root);
        self.stage = Stage::Gu;
        Ok(())
    }

    pub fn draw_queries(&mut self) -> Result<()> {
        if self.stage != Stage::Gu {
            return Err(Error::InvalidSessionState);
        }
        let prefix = self.chain.message().prefix(KIND_POSITIONS, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        self.queries = self.plan.draw_queries(&mut self.chain)?;
        self.stage = Stage::Queried;
        Ok(())
    }

    pub fn read_queries(&mut self, openings: &[ColumnOpening]) -> Result<Vec<Vec<Fp>>> {
        if self.stage != Stage::Queried && self.stage != Stage::Read {
            return Err(Error::InvalidSessionState);
        }
        if openings.len() != self.queries.len() {
            return Err(Error::ShapeMismatch);
        }
        let Self {
            plan,
            reads,
            queries,
            stage,
            ..
        } = self;
        let columns = reads.read(&plan.reader, queries, openings)?;
        *stage = Stage::Read;
        Ok(columns)
    }

    /// Verify the prover's public FOLD start message and bind it in the start
    /// round; the verifier receives no coefficient, tail or coin. The mask is
    /// spent once.
    pub fn start_fold(&mut self, message: &FoldStartMessage) -> Result<()> {
        if self.stage != Stage::Read {
            return Err(Error::InvalidSessionState);
        }
        if message.tree_number != self.plan.mask_tree_number
            || message.claimed_group != self.plan.group_id
        {
            return Err(Error::StatementMismatch);
        }
        let oracles = self
            .plan
            .oracles
            .lock()
            .map_err(|_| Error::InvalidSessionState)?;
        fold_start_verify(&self.plan.fold, message, &self.plan.statement, &oracles)?;
        let mut out = self.chain.message();
        out.push_u64(message.id);
        out.push_u64(message.claimed_group);
        out.push_u64(message.notice.mask_rows);
        out.push_u64(message.notice.mask_code.log_n as u64);
        out.push_u64(message.notice.mask_code.ell);
        out.push_u64(message.notice.mask_code.r);
        out.push_u64(message.notice.answers as u64);
        out.push_u64(message.notice.leaf_class as u64);
        for chunk in message.root.0.chunks(8) {
            let word = u64::from_le_bytes(chunk.try_into().map_err(|_| Error::InvalidEncoding)?);
            out.push_u64(word);
        }
        out.push_fp3(&message.sum);
        let prefix = out.prefix(KIND_EPSILON, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        self.epsilon = Some(self.chain.draw_e()?);
        self.start = Some(message.clone());
        self.stage = Stage::FoldStarted;
        Ok(())
    }

    /// Verify the mask root against the plan's runtime tree and the plan's
    /// allowed position list. This is the root authentication the start
    /// message cannot provide by itself.
    pub fn read_mask(&mut self, openings: &[MaskColumnOpening]) -> Result<()> {
        if self.stage != Stage::FoldStarted {
            return Err(Error::InvalidSessionState);
        }
        let message = self.start.as_ref().ok_or(Error::InvalidSessionState)?;
        let tree = self
            .plan
            .statement
            .mask_tree(self.plan.mask_tree_number)
            .ok_or(Error::StatementMismatch)?;
        let mut oracles = self
            .plan
            .oracles
            .lock()
            .map_err(|_| Error::InvalidSessionState)?;
        oracles.authenticate_mask(
            tree,
            message.id,
            &self.plan.mask_positions,
            &self.plan.mask_positions,
            openings,
        )?;
        drop(oracles);
        self.stage = Stage::MaskOpened;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::request::code::build_code_plan;
    use crate::pcs::request::field::{Fp, MODULUS};
    use crate::pcs::request::runtime_tree::MaskColumnOpening;
use crate::pcs::request::fold::{
        fold_mask_make, fold_mask_make_tail, fold_mask_notice, fold_start_prove, MaskOracle,
        OracleMetadata, RuntimeOracle, RuntimeOracles, VerifiedGroup, VerifiedGroups,
        VerifiedStatement,
    };
    use crate::pcs::request::params::{LEAF_SALT_MIN_BYTES, REQUEST_CHAIN_LABEL};
    use crate::pcs::request::read::{HidingState, ProverReads};
    use crate::pcs::request::runtime_tree::{verify_mask_columns, MaskTreePlan};
        use crate::pcs::request::table::{commit_table, IndexShape, TableDistances};
    use crate::primitives::transcript::{hash_labeled, tag_word, TEST_COINS};
    use zkie_core::common::field::PrimeCharacteristicRing;

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

    fn committed() -> crate::pcs::request::table::CommittedTable {
        let shape = shape();
        let rows = shape.rows() as usize;
        let messages: Vec<Vec<Fp>> = (0..rows)
            .map(|row| {
                (0..shape.code.ell)
                    .map(|index| Fp::from_u64((row * 10 + index as usize + 1) as u64))
                    .collect()
            })
            .collect();
        let tails: Vec<Vec<Fp>> = (0..rows)
            .map(|row| {
                (0..shape.code.r)
                    .map(|index| Fp::from_u64((row + index as usize + 1) as u64))
                    .collect()
            })
            .collect();
        let salts: Vec<Vec<u8>> = (0..shape.code.n() as usize)
            .map(|position| vec![(position as u8).wrapping_add(1); shape.salt_bytes])
            .collect();
        commit_table(shape, Digest([7; 32]), &messages, &tails, &salts).unwrap()
    }

    fn fold_shape() -> FoldShape {
        FoldShape {
            code: build_code_plan(3, 4, 1).unwrap(),
            mask_code: build_code_plan(3, 4, 1).unwrap(),
            k: 2,
            h: 1,
            mask_leaf_class: 3,
        }
    }

    fn mask_tree() -> MaskTreePlan {
        let fold = fold_shape();
        MaskTreePlan {
            tree_number: 5,
            leaf_class: fold.mask_leaf_class,
            log_n: fold.mask_code.log_n,
            mask_rows: fold.mask_rows().unwrap(),
            salt_bytes: LEAF_SALT_MIN_BYTES,
            run: Digest([2; 32]),
            configuration: Digest([7; 32]),
        }
    }

    fn mask_salts(plan: &MaskTreePlan) -> Vec<Vec<u8>> {
        (0..plan.columns() as usize)
            .map(|position| vec![(position as u8).wrapping_add(1); plan.salt_bytes])
            .collect()
    }

    fn plan_fixture() -> (FoldeqPlan, crate::pcs::request::table::CommittedTable) {
        let table = committed();
        let tree = mask_tree();
        let statement = Arc::new(
            VerifiedStatement::new(
                VerifiedGroups {
                    groups: vec![VerifiedGroup {
                        group: 1,
                        members: vec![1, 2],
                        fresh: Some(9),
                    }],
                },
                vec![tree],
                vec![OracleMetadata {
                    id: 9,
                    group: 1,
                    tree_number: tree.tree_number,
                    leaf_class: tree.leaf_class,
                    mask_rows: tree.mask_rows,
                    mask_code: build_code_plan(3, 4, 1).unwrap(),
                }],
            )
            .unwrap(),
        );
        let plan = FoldeqPlan {
            reader: table.reader.clone(),
            fold: fold_shape(),
            statement,
            oracles: Arc::new(Mutex::new(RuntimeOracles::default())),
            mask_tree_number: tree.tree_number,
            group_id: 1,
            leaf_class: 3,
            seed_budget: table.reader.shape.answers as usize,
            ledger: Arc::new(Mutex::new(ReadLedger::new(2))),
            column_draws: 2,
            mask_positions: vec![1, 4],
            gamma_path: GammaPath::Lin { k_lo: 1, k_col: 3, lin_shape: Digest([3; 32]) },
        };
        (plan, table)
    }

    fn open_session() -> (FoldeqSession, crate::pcs::request::table::CommittedTable) {
        let (plan, table) = plan_fixture();
        let binder = FoldeqBinder::new(REQUEST_CHAIN_LABEL, plan.reader.configuration, &plan).unwrap();
        let chain = binder.chain().unwrap();
        (FoldeqSession::open(plan, binder, chain).unwrap(), table)
    }

    fn mask_start(plan: &FoldeqPlan) -> crate::pcs::request::fold::FoldStartProver {
        let mut source = TestCoins::new(Digest([9; 32]));
        let mut mask = fold_mask_make(&plan.fold, &mut source).unwrap();
        fold_mask_make_tail(&plan.fold, &mut mask, &mut source).unwrap();
        mask.sent_as(9).unwrap();
        let notice = fold_mask_notice(&plan.fold).unwrap();
        let oracle = MaskOracle::prover_view(9, 1, &notice);
        let tree = *plan.statement.mask_tree(plan.mask_tree_number).unwrap();
        fold_start_prove(
            &plan.fold,
            &mut mask,
            &source,
            &oracle,
            plan.statement.groups(),
            tree,
            &mask_salts(&tree),
        )
        .unwrap()
    }

    #[test]
    fn wrong_prefix_chain_is_rejected() {
        let (plan, _table) = plan_fixture();
        let binder = FoldeqBinder::new(REQUEST_CHAIN_LABEL, plan.reader.configuration, &plan).unwrap();
        let wrong = Chain::begin(
            REQUEST_CHAIN_LABEL,
            &Digest::ZERO,
            b"another first query",
            &[],
            None,
        )
        .unwrap();
        assert!(FoldeqSession::open(plan, binder, wrong).is_err());
    }

    #[test]
    fn first_query_binds_the_plan_and_configuration() {
        let (plan, _table) = plan_fixture();
        assert!(FoldeqBinder::new(REQUEST_CHAIN_LABEL, Digest::ZERO, &plan).is_err());
        let mut other = plan.clone();
        other.column_draws = 1;
        assert_ne!(plan.first_query(), other.first_query());
        // The same root and record under another plan is not this first query.
        let binder = FoldeqBinder::new(
            REQUEST_CHAIN_LABEL,
            plan.reader.configuration,
            &plan,
        )
        .unwrap();
        let chain = binder.chain().unwrap();
        let mut session = FoldeqSession::open(plan.clone(), binder, chain).unwrap();
        assert!(session.bind_first_query(&other.first_query()).is_err());
        let own = plan.first_query();
        session.bind_first_query(&own).unwrap();
    }

    #[test]
    fn lin_gamma_is_the_psi_suffix() {
        let psi = vec![
            Fp3::from_u64(1),
            Fp3::from_u64(2),
            Fp3::from_u64(3),
            Fp3::from_u64(4),
        ];
        assert_eq!(
            lin_gamma(&psi, 2, 2).unwrap(),
            vec![Fp3::from_u64(3), Fp3::from_u64(4)]
        );
        assert!(lin_gamma(&psi, 3, 2).is_err());
        assert!(lin_gamma(&psi, 0, 5).is_err());
    }

    #[test]
    fn gamma_is_drawn_from_the_fixed_prefix() {
        let draw = |plan: FoldeqPlan| {
            let binder = FoldeqBinder::new(
                REQUEST_CHAIN_LABEL,
                plan.reader.configuration,
                &plan,
            )
            .unwrap();
            let chain = binder.chain().unwrap();
            let mut session = FoldeqSession::open(plan.clone(), binder, chain).unwrap();
            let first = plan.first_query();
            session.bind_first_query(&first).unwrap();
            session.draw_gamma().unwrap();
            session.gamma().unwrap().to_vec()
        };
        let (plan, _table) = plan_fixture();
        let first = draw(plan.clone());
        let (same, _table) = plan_fixture();
        assert_eq!(first, draw(same));
        let mut other = plan.clone();
        other.gamma_path = GammaPath::Direct {
            columns: other.reader.shape.v,
        };
        assert_ne!(first, draw(other.clone()));
        // The same statement with a replaced LIN offset is a different plan
        // binding and cannot be bound to the first query.
        let mut offset = plan;
        offset.gamma_path = GammaPath::Lin { k_lo: 0, k_col: 2, lin_shape: Digest([3; 32]) };
        assert_ne!(first, draw(offset.clone()));
        let binder = FoldeqBinder::new(
            REQUEST_CHAIN_LABEL,
            offset.reader.configuration,
            &offset,
        )
        .unwrap();
        let chain = binder.chain().unwrap();
        let mut session = FoldeqSession::open(offset, binder, chain).unwrap();
        let first = session.plan().first_query();
        session.bind_first_query(&first).unwrap();
        assert!(session.draw_gamma().is_ok());
    }

    #[test]
    fn first_query_and_stage_order() {
        let (mut session, _table) = open_session();
        assert!(session.seeds().is_err());
        assert!(session.draw_gamma().is_err());
        let mut sent = session.plan().first_query();
        sent[0] ^= 1;
        assert!(session.bind_first_query(&sent).is_err());
        let good = session.plan().first_query();
        session.bind_first_query(&good).unwrap();
        assert_eq!(session.seeds().unwrap().len(), 1);
        assert!(session.send_gu(&Digest([4; 32])).is_err());
        assert!(session.draw_queries().is_err());
        assert!(session.read_queries(&[]).is_err());
        let plan = session.plan().clone();
        assert!(session.start_fold(&mask_start(&plan).message).is_err());
    }

    #[test]
    fn gamma_gu_queries_reads_and_fold_start() {
        let (mut session, table) = open_session();
        let plan = session.plan().clone();
        let good = plan.first_query();
        session.bind_first_query(&good).unwrap();
        let _seeds = session.seeds().unwrap();
        session.draw_gamma().unwrap();
        assert_eq!(session.gamma().unwrap().len(), plan.reader.shape.v as usize);
        let prover_hiding = HidingState::new(table.reader.root, 1, 8);
        let mut gu_prover =
            ProverFoldeq::new(prover_hiding, &table.reader, 1, plan.column_draws).unwrap();
        assert!(gu_prover.send_gu(&Digest([4; 32])).is_err());
        let charge = gu_prover.begin().unwrap();
        let gu = gu_prover.send_gu(&Digest([4; 32])).unwrap();
        session.send_gu(&gu.root).unwrap();
        assert_eq!(session.gu_root(), Some(&gu.root));
        assert_eq!(gu_prover.used(), plan.column_draws as u64);
        session.draw_queries().unwrap();
        assert_eq!(session.queries().len(), plan.column_draws);
        let mut prover = ProverReads::new(plan.ledger.lock().unwrap().max_reads());
        let openings = prover.open(&charge, &table, session.queries()).unwrap();
        let columns = session.read_queries(&openings).unwrap();
        assert_eq!(columns.len(), plan.column_draws);
        // A second read charges past the plan's budget.
        assert!(session.read_queries(&openings).is_err());
        let start = mask_start(&plan);
        let mask_openings = start.tree.open(&plan.mask_positions).unwrap();
        {
            let tree = plan.statement.mask_tree(plan.mask_tree_number).unwrap();
            session
                .plan()
                .oracles
                .lock()
                .unwrap()
                .receive_mask(
                    &plan.statement,
                    tree,
                    9,
                    &start.message.notice,
                    start.message.root,
                )
                .unwrap();
            // Registration of a forged root is only a claim; the later
            // opening phase must reject it.
            let mut forged_registry = RuntimeOracles::default();
            forged_registry
                .receive_mask(
                    &plan.statement,
                    tree,
                    9,
                    &start.message.notice,
                    Digest([5; 32]),
                )
                .unwrap();
            assert!(forged_registry
                .authenticate_mask(
                    tree,
                    9,
                    &plan.mask_positions,
                    &plan.mask_positions,
                    &mask_openings,
                )
                .is_err());
            // Empty or caller-chosen positions are not the verifier's list.
            let mut empty_registry = RuntimeOracles::default();
            empty_registry
                .receive_mask(
                    &plan.statement,
                    tree,
                    9,
                    &start.message.notice,
                    start.message.root,
                )
                .unwrap();
            assert!(empty_registry
                .authenticate_mask(tree, 9, &plan.mask_positions, &[], &[])
                .is_err());
            assert!(empty_registry
                .authenticate_mask(
                    tree,
                    9,
                    &plan.mask_positions,
                    &[1u64],
                    &mask_openings[..1],
                )
                .is_err());
        }
        session.start_fold(&start.message).unwrap();
        assert!(session.epsilon().is_some());
        assert_eq!(session.start(), Some(&start.message));
        session.read_mask(&mask_openings).unwrap();
        assert!(session.read_mask(&mask_openings).is_err());
        let tree = plan.statement.mask_tree(plan.mask_tree_number).unwrap();
        assert!(verify_mask_columns(
            tree,
            &Digest([5; 32]),
            &plan.mask_positions,
            &mask_openings
        )
        .is_err());
        let mut forged = mask_openings;
        forged[0].symbols[0] = forged[0].symbols[0] + Fp::ONE;
        assert!(verify_mask_columns(
            tree,
            &start.message.root,
            &plan.mask_positions,
            &forged
        )
        .is_err());
    }

    #[test]
    fn provider_hiding_is_bound_and_charged_before_send() {
        let (plan, table) = plan_fixture();
        let wrong_table = HidingState::new(Digest([9; 32]), 1, 8);
        assert!(ProverFoldeq::new(wrong_table, &table.reader, 1, plan.column_draws).is_err());
        let wrong_lifecycle = HidingState::new(table.reader.root, 2, 8);
        assert!(ProverFoldeq::new(wrong_lifecycle, &table.reader, 1, plan.column_draws).is_err());
        let too_small = HidingState::new(table.reader.root, 1, 1);
        let mut small =
            ProverFoldeq::new(too_small, &table.reader, 1, plan.column_draws).unwrap();
        assert!(small.begin().is_err());
        assert!(small.send_gu(&Digest([4; 32])).is_err());
        let state = HidingState::new(table.reader.root, 1, 8);
        let mut prover = ProverFoldeq::new(state, &table.reader, 1, plan.column_draws).unwrap();
        assert!(prover.send_gu(&Digest([4; 32])).is_err());
        let _charge = prover.begin().unwrap();
        assert!(prover.begin().is_err());
        assert_eq!(prover.used(), plan.column_draws as u64);
    }

    #[test]
    fn plan_faults_are_rejected() {
        let (plan, _table) = plan_fixture();
        let binder = FoldeqBinder::new(
            REQUEST_CHAIN_LABEL,
            plan.reader.configuration,
            &plan,
        )
        .unwrap();
        let chain = binder.chain().unwrap();

        let mut bad = plan.clone();
        bad.seed_budget = 0;
        assert!(FoldeqSession::open(bad, binder.clone(), chain.clone()).is_err());

        let mut bad = plan.clone();
        bad.column_draws = 0;
        assert!(FoldeqSession::open(bad, binder.clone(), chain.clone()).is_err());

        let mut bad = plan.clone();
        bad.mask_tree_number = 6;
        assert!(FoldeqSession::open(bad, binder.clone(), chain.clone()).is_err());

        let mut bad = plan.clone();
        bad.ledger = Arc::new(Mutex::new(ReadLedger::new(1)));
        bad.column_draws = 2;
        assert!(FoldeqSession::open(bad, binder.clone(), chain.clone()).is_err());

        let mut bad = plan.clone();
        bad.mask_positions.clear();
        assert!(FoldeqSession::open(bad, binder.clone(), chain.clone()).is_err());

        let mut bad = plan.clone();
        bad.gamma_path = GammaPath::Lin { k_lo: 0, k_col: 1, lin_shape: Digest([3; 32]) };
        assert!(FoldeqSession::open(bad, binder.clone(), chain.clone()).is_err());

        let mut bad = plan;
        bad.group_id = 7;
        assert!(FoldeqSession::open(bad, binder, chain).is_err());
    }

    // A local numbered source for the joint harness tests; it is not a
    // production randomness source and provides no hiding.
    struct TestCoins {
        seed: Digest,
        draw: u32,
    }

    impl TestCoins {
        fn new(seed: Digest) -> Self {
            Self { seed, draw: 0 }
        }

        fn value_at(&self, number: u32) -> Result<Fp3> {
            let digest = hash_labeled(
                tag_word(TEST_COINS, 0),
                number as u64,
                Some(&self.seed),
                &[],
            )?;
            let mut coefficients = [Fp::ZERO; 3];
            for (index, coefficient) in coefficients.iter_mut().enumerate() {
                let word =
                    u64::from_le_bytes(digest.0[index * 8..(index + 1) * 8].try_into().unwrap());
                *coefficient = Fp::from_u64(word % MODULUS);
            }
            Ok(Fp3::new(coefficients[0], coefficients[1], coefficients[2]))
        }
    }

    impl crate::pcs::request::fold::MaskCoins for TestCoins {
        fn draw(&mut self) -> Result<(Fp3, u32)> {
            let number = self.draw;
            self.draw = self.draw.checked_add(1).ok_or(Error::ResourceLimit)?;
            Ok((self.value_at(number)?, number))
        }

        fn redraw(&self, number: u32) -> Result<Fp3> {
            self.value_at(number)
        }
    }
}
