//! FOLDEQ session skeleton (foundation only).
//!
//! The verifier's plan holds the committed table, the fold shape, the runtime
//! mask-tree layout, the group registry and the budgets/counts; every number
//! comes from the verified statement/plan, not from a proof. With a G_u shape
//! the session orders the complete first query, the record seeds, the gamma
//! challenge, the received G_u root, the shared seed set drawn from that
//! session chain, the answer round (closed before any later challenge), the
//! group join and record rows, the column draws, the column rows and the FOLD
//! start; without one it keeps the existing path. The group join uses the
//! trusted statement group's codes and the registered G_u shape. Column rows
//! come either from the reads verified against the trusted reader root or from
//! an explicit, unauthenticated fixture; both stay pending, and only the
//! verified source can satisfy the FOLD gate. The statement and
//! runtime layout origins are trusted fixtures in this isolated layer, the
//! proof-byte answer decoder is not connected, and no production
//! statement-authentication, relation or FOLD closure is claimed.

use std::sync::{Arc, Mutex};

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::chain::{
    Chain, KIND_ALGEBRAIC, KIND_EPSILON, KIND_EMPTY, KIND_POSITIONS, KIND_SEAL, KIND_SEED,
};
use crate::pcs::request::field::{Fp, Fp3};
use crate::pcs::request::gu::{GuMembership, GuNotice, GuRegistry, GuShape};
use crate::pcs::request::relation::{
    column_plan_rows, foldeq_weights, record_row, sample_row, GuRelationRow,
};
use crate::pcs::request::runtime_tree::RuntimeTreeOpening;
use crate::pcs::request::fold::{
    fold_mask_shape_fault, fold_start_verify, fold_start_verify_with_groups, FoldShape,
    FoldStartMessage, RuntimeOracles, VerifiedStatement,
};
use crate::pcs::request::read::{
    draw_positions, unique_sorted, HidingCharge, HidingState, ReadLedger, VerifierReads,
};
use crate::pcs::request::table::{
    index_name, record_seeds, ColumnOpening, IndexName, IndexReader, SampleSeed,
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
    Lin { shape: LinShapeBinding },
    Direct { columns: u32 },
}

/// The verifier's binding of the caller's LinShape for the gamma path: the
/// matrix rows and the low/column counts. The verifier constructs it from its
/// own trusted LinShape and the first query binds these fields. Carrying the
/// fields (not a caller-supplied digest) keeps the values checkable; the
/// trusted source itself remains a verifier/plan prerequisite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinShapeBinding {
    pub matrix_rows: u64,
    pub k_lo: u32,
    pub k_col: u32,
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
    pub mask_query_count: usize,
    pub gamma_path: GammaPath,
    /// The trusted G_u shape when this plan carries the isolated G_u relation
    /// session. It is a plan/statement value, not a proof report.
    pub gu: Option<GuShape>,
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
        let answers = self.reader.shape.answers as usize;
        if self.seed_budget < answers {
            return Some("the seed budget is below the answer count");
        }
        if let Some(gu) = &self.gu {
            if gu.fault().is_some() {
                return Some("the G_u shape is not valid");
            }
            if gu.index != self.reader.shape {
                return Some("the G_u index shape is not the plan's reader shape");
            }
            if gu.group != self.group_id {
                return Some("the G_u group is not the FOLD group");
            }
            if self.statement.groups().get(gu.group).is_none() {
                return Some("the G_u group is not in the statement");
            }
            if crate::pcs::request::relation::gu_layout_fault(gu).is_some() {
                return Some("the G_u layout is not executable");
            }
        }
        match self.gamma_path {
            GammaPath::Lin { shape } => {
                if shape.matrix_rows < 1
                    || shape.matrix_rows > (1u64 << 30)
                    || shape.k_lo > 30
                    || shape.k_col > 60
                {
                    return Some("the LIN shape binding has invalid numbers");
                }
                let expected = match shape.k_lo.checked_add(self.reader.shape.v) {
                    Some(expected) => expected,
                    None => return Some("the LIN challenge path overflows"),
                };
                if shape.k_col != expected {
                    return Some("the LIN challenge path does not end at the table width");
                }
                // A shift that loses high bits is not a fit: checked_shl only
                // rejects an out-of-range shift amount, so use a checked
                // multiply by the explicit power of two instead.
                let factor = match 1u64.checked_shl(shape.k_lo) {
                    Some(factor) => factor,
                    None => return Some("the matrix width overflows"),
                };
                let width = match shape.matrix_rows.checked_mul(factor) {
                    Some(width) => width,
                    None => return Some("the matrix width overflows"),
                };
                if width > self.reader.shape.plain() {
                    return Some("the matrix does not fit the table's plain columns");
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
        writer.put_u64(self.mask_query_count as u64);
        match self.gamma_path {
            GammaPath::Lin { shape } => {
                writer.put_u64(1);
                writer.put_u64(shape.matrix_rows);
                writer.put_u64(shape.k_lo as u64);
                writer.put_u64(shape.k_col as u64);
            }
            GammaPath::Direct { columns } => {
                writer.put_u64(2);
                writer.put_u64(columns as u64);
            }
        }
        match &self.gu {
            Some(gu) => {
                writer.put_u64(1);
                let bytes = gu.encode();
                writer.put_u64(bytes.len() as u64);
                writer.put_bytes(&bytes);
            }
            None => writer.put_u64(0),
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

/// Where the joined G_u column rows came from. A fixture is useful for
/// isolated evaluation but is not a reader authentication result and cannot
/// satisfy the FOLD gate; only the verified read path can.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GuColumnSource {
    Fixture,
    Verified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    FirstQuery,
    Seeds,
    Gamma,
    Gu,
    GuAnswers,
    GuRecord,
    GuJoined,
    Queried,
    Read,
    FoldStarted,
    MaskQueried,
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
    mask_queries: Vec<u64>,
    gu_registry: GuRegistry,
    gu_seeds: Vec<SampleSeed>,
    gu_answers: Vec<Fp3>,
    gu_answered: usize,
    gu_membership: Option<GuMembership>,
    gu_read_columns: Option<Vec<Vec<Fp>>>,
    gu_columns_source: Option<GuColumnSource>,
    gu_rows: Vec<GuRelationRow>,
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
            mask_queries: Vec::new(),
            gu_registry: GuRegistry::new(),
            gu_seeds: Vec::new(),
            gu_answers: Vec::new(),
            gu_answered: 0,
            gu_membership: None,
            gu_read_columns: None,
            gu_columns_source: None,
            gu_rows: Vec::new(),
        })
    }

    pub fn plan(&self) -> &FoldeqPlan {
        &self.plan
    }

    pub fn queries(&self) -> &[u64] {
        &self.queries
    }

    pub fn mask_queries(&self) -> &[u64] {
        &self.mask_queries
    }

    pub fn gamma(&self) -> Option<&[Fp3]> {
        self.gamma.as_deref()
    }

    /// The seed set drawn from the received G_u root's SEND round. It is
    /// session state, not a proof field.
    pub fn gu_shared_seeds(&self) -> &[SampleSeed] {
        &self.gu_seeds
    }

    pub fn gu_root(&self) -> Option<&Digest> {
        self.gu_root.as_ref()
    }

    /// The current chain value. This is session state for inspection, not a
    /// proof field and not a public transcript artifact.
    pub fn transcript_value(&self) -> &Digest {
        self.chain.value()
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
            GammaPath::Lin { shape } => {
                let psi = self.draw_challenge(shape.k_col as usize)?;
                lin_gamma(&psi, shape.k_lo, self.plan.reader.shape.v)?
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
        // With a G_u shape the root opens the seed round: the verifier's
        // message is the seed set, drawn by `shared_seeds`. Without one the
        // root keeps the existing seal round.
        let kind = if self.plan.gu.is_some() {
            KIND_SEED
        } else {
            KIND_SEAL
        };
        let prefix = message.prefix(kind, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        self.gu_root = Some(*root);
        if let Some(gu) = &self.plan.gu {
            self.gu_registry.receive(
                gu,
                &GuNotice {
                    root: *root,
                    oracle: gu.oracle,
                    tree_number: gu.tree.tree_number,
                    part: gu.part,
                },
            )?;
        }
        self.stage = Stage::Gu;
        Ok(())
    }

    /// Draw the verifier's shared seed set from the session chain after the
    /// received G_u root. The seed carries the maximum sample-cube dimension
    /// and G_u's answer count; the caller cannot supply or replace it. This is
    /// the single-G_u interface; a multi-oracle SEND message is not
    /// implemented.
    pub fn shared_seeds(&mut self) -> Result<()> {
        if self.stage != Stage::Gu {
            return Err(Error::InvalidSessionState);
        }
        let gu = self.plan.gu.as_ref().ok_or(Error::InvalidSessionState)?;
        let bits = gu.sample_bits()? as usize;
        let answers = gu.u.answers as usize;
        let mut seeds = Vec::new();
        seeds
            .try_reserve_exact(answers)
            .map_err(|_| Error::ResourceLimit)?;
        for _ in 0..answers {
            let mut z = Vec::new();
            z.try_reserve_exact(bits)
                .map_err(|_| Error::ResourceLimit)?;
            for _ in 0..bits {
                z.push(self.chain.draw_e()?);
            }
            let beta = self.chain.draw_e_nonzero()?;
            seeds.push(SampleSeed { z, beta });
        }
        self.gu_seeds = seeds;
        self.gu_answers.clear();
        self.gu_answered = 0;
        self.stage = Stage::GuAnswers;
        Ok(())
    }

    /// Receive the next sample answer in the fixed single-G_u answer order.
    /// The answer is encoded into the answer round with its canonical field
    /// words; after the last answer the round is closed and its state feeds
    /// the later column draws. A value swap is not necessarily visible at this
    /// typed interface; the fixed seed and later relation check decide it.
    ///
    /// The interface takes typed field values. A proof-byte decoder with
    /// truncation/non-canonical checks is not connected here, so that byte
    /// boundary remains an integration gate and is not claimed as tested.
    pub fn receive_gu_answer(&mut self, target: Fp3) -> Result<()> {
        if self.stage != Stage::GuAnswers {
            return Err(Error::InvalidSessionState);
        }
        let gu = self.plan.gu.as_ref().ok_or(Error::InvalidSessionState)?.clone();
        if self.gu_answered >= gu.u.answers as usize {
            return Err(Error::InvalidSessionState);
        }
        let row = sample_row(
            &gu,
            &self.gu_registry,
            &self.gu_seeds,
            self.gu_answered as u32,
            target,
        )?;
        self.gu_rows.push(row);
        self.gu_answers.push(target);
        self.gu_answered += 1;
        if self.gu_answered == gu.u.answers as usize {
            let mut message = self.chain.message();
            for answer in &self.gu_answers {
                message.push_fp3(answer);
            }
            let prefix = message.prefix(KIND_EMPTY, 0)?;
            if !self.chain.step(&prefix, 0)? {
                return Err(Error::InvalidProof);
            }
            self.stage = Stage::GuRecord;
        }
        Ok(())
    }

    /// Join the trusted group and the record rows after every sample answer,
    /// using the same gamma as the row-point weights and the bound index
    /// name's record seeds. The group join checks the trusted group/code and
    /// the registered G_u shape, and every pending row must belong to the
    /// joined oracle and group.
    pub fn join_gu_record(&mut self) -> Result<()> {
        if self.stage != Stage::GuRecord {
            return Err(Error::InvalidSessionState);
        }
        let gu = self.plan.gu.as_ref().ok_or(Error::InvalidSessionState)?.clone();
        let group = self
            .plan
            .statement
            .groups()
            .get(gu.group)
            .ok_or(Error::StatementMismatch)?;
        let mut membership = GuMembership::new(group, &gu, &self.gu_registry)?;
        membership.join(&gu, &self.gu_registry)?;
        for row in &self.gu_rows {
            membership.check_row(row)?;
        }
        let gamma = self.gamma.clone().ok_or(Error::InvalidSessionState)?;
        let eq_gamma = foldeq_weights(&gamma, self.plan.reader.shape.v)?;
        let name = index_name(
            &self.plan.reader.root,
            &self.plan.reader.shape,
            &self.plan.reader.configuration,
        );
        let answers = self.plan.reader.shape.answers as usize;
        let seeds = self.plan.seeds()?;
        if seeds.len() < answers {
            return Err(Error::ShapeMismatch);
        }
        let mut rows = Vec::new();
        rows.try_reserve_exact(answers)
            .map_err(|_| Error::ResourceLimit)?;
        for answer in 0..answers {
            let row = record_row(
                &gu,
                &self.gu_registry,
                &self.plan.reader.record,
                &name,
                &seeds[..answers],
                &eq_gamma,
                answer as u32,
            )?;
            membership.check_row(&row)?;
            rows.push(row);
        }
        self.gu_rows.extend(rows);
        self.gu_membership = Some(membership);
        self.stage = Stage::GuJoined;
        Ok(())
    }

    /// Add column rows from an explicit fixture column map before any read.
    /// The fixture is not authenticated by a reader opening and the rows stay
    /// pending; this path is only for isolated evaluation and is not accepted
    /// by the FOLD gate. A caller with verified reads must use
    /// `join_gu_read_columns` after `read_queries`.
    pub fn join_gu_columns(&mut self, fixture_columns: &[(u64, Vec<Fp3>)]) -> Result<()> {
        if self.stage != Stage::Queried {
            return Err(Error::InvalidSessionState);
        }
        if self.gu_columns_source.is_some() {
            return Err(Error::InvalidSessionState);
        }
        let gu = self.plan.gu.as_ref().ok_or(Error::InvalidSessionState)?.clone();
        let membership = self
            .gu_membership
            .as_ref()
            .ok_or(Error::InvalidSessionState)?;
        membership.require(&gu, &self.gu_registry)?;
        let gamma = self.gamma.clone().ok_or(Error::InvalidSessionState)?;
        let eq_gamma = foldeq_weights(&gamma, self.plan.reader.shape.v)?;
        let rows = column_plan_rows(
            &gu,
            &self.gu_registry,
            self.plan.column_draws,
            self.plan.reader.shape.code.log_n,
            &self.queries,
            fixture_columns,
            &eq_gamma,
        )?;
        for row in &rows {
            membership.check_row(row)?;
        }
        self.gu_rows.extend(rows);
        self.gu_columns_source = Some(GuColumnSource::Fixture);
        Ok(())
    }

    /// Add column rows from the columns that `read_queries` verified against
    /// the trusted reader root for this session's draws. The draw order,
    /// repeats included, is reused. `read_queries` verifies one opening per
    /// unique physical position and maps it back to the logical draw sequence;
    /// no caller column map is accepted here, so a substituted read cannot
    /// become an authenticated claim.
    pub fn join_gu_read_columns(&mut self) -> Result<()> {
        if self.stage != Stage::Read {
            return Err(Error::InvalidSessionState);
        }
        if self.gu_columns_source.is_some() {
            return Err(Error::InvalidSessionState);
        }
        let gu = self.plan.gu.as_ref().ok_or(Error::InvalidSessionState)?.clone();
        let membership = self
            .gu_membership
            .as_ref()
            .ok_or(Error::InvalidSessionState)?;
        membership.require(&gu, &self.gu_registry)?;
        let read_columns = self
            .gu_read_columns
            .as_ref()
            .ok_or(Error::InvalidSessionState)?;
        if read_columns.len() != self.queries.len() {
            return Err(Error::ShapeMismatch);
        }
        let rows_count = usize::try_from(self.plan.reader.shape.rows())
            .map_err(|_| Error::ResourceLimit)?;
        let mut columns = Vec::new();
        for position in unique_sorted(&self.queries) {
            let mut column: Option<&Vec<Fp>> = None;
            for (query, read) in self.queries.iter().zip(read_columns) {
                if *query == position {
                    if read.len() != rows_count {
                        return Err(Error::ShapeMismatch);
                    }
                    match column {
                        Some(previous) if previous != read => {
                            return Err(Error::StatementMismatch);
                        }
                        Some(_) => {}
                        None => column = Some(read),
                    }
                }
            }
            let column = column.ok_or(Error::StatementMismatch)?;
            let values = column
                .iter()
                .map(|value| Fp3::from_base(*value))
                .collect();
            columns.push((position, values));
        }
        let gamma = self.gamma.clone().ok_or(Error::InvalidSessionState)?;
        let eq_gamma = foldeq_weights(&gamma, self.plan.reader.shape.v)?;
        let rows = column_plan_rows(
            &gu,
            &self.gu_registry,
            self.plan.column_draws,
            self.plan.reader.shape.code.log_n,
            &self.queries,
            &columns,
            &eq_gamma,
        )?;
        for row in &rows {
            membership.check_row(row)?;
        }
        self.gu_rows.extend(rows);
        self.gu_columns_source = Some(GuColumnSource::Verified);
        Ok(())
    }

    pub fn pending_gu_rows(&self) -> &[GuRelationRow] {
        &self.gu_rows
    }

    pub fn draw_queries(&mut self) -> Result<()> {
        let expected = if self.plan.gu.is_some() {
            Stage::GuJoined
        } else {
            Stage::Gu
        };
        if self.stage != expected {
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

    /// Verify one opening per unique physical position and map the verified
    /// columns back to the logical draw sequence. The caller supplies one
    /// opening per logical draw; repeats must carry the same opening. The read
    /// ledger therefore charges the actual unique openings, while the plan's
    /// column count remains the logical sampling count. Hiding accounting is a
    /// separate provider-side ledger and is not reduced by this deduplication.
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
            gu_read_columns,
            ..
        } = self;
        let mut unique: Vec<(u64, ColumnOpening)> = Vec::new();
        for (position, opening) in queries.iter().zip(openings) {
            if opening.position != *position {
                return Err(Error::InvalidIndex);
            }
            match unique.iter().find(|(at, _)| at == position) {
                Some((_, previous)) if previous != opening => {
                    return Err(Error::StatementMismatch);
                }
                Some(_) => {}
                None => unique.push((*position, opening.clone())),
            }
        }
        unique.sort_by_key(|(position, _)| *position);
        let positions: Vec<u64> = unique.iter().map(|(position, _)| *position).collect();
        let unique_openings: Vec<ColumnOpening> =
            unique.into_iter().map(|(_, opening)| opening).collect();
        let unique_columns = reads.read(&plan.reader, &positions, &unique_openings)?;
        let mut by_position = std::collections::HashMap::new();
        for (position, column) in positions.into_iter().zip(unique_columns) {
            by_position.insert(position, column);
        }
        let mut columns = Vec::new();
        columns
            .try_reserve_exact(queries.len())
            .map_err(|_| Error::ResourceLimit)?;
        for position in queries {
            columns.push(
                by_position
                    .get(position)
                    .cloned()
                    .ok_or(Error::StatementMismatch)?,
            );
        }
        if plan.gu.is_some() {
            *gu_read_columns = Some(columns.clone());
        }
        *stage = Stage::Read;
        Ok(columns)
    }

    /// Verify the prover's public FOLD start message and bind it in the start
    /// round; the verifier receives no coefficient, tail or coin. With a G_u
    /// shape the session overlays the joined membership onto the trusted group
    /// view consumed by the FOLD check, and requires the FOLD target group to
    /// be the joined group and the column rows to come from verified reads.
    pub fn start_fold(&mut self, message: &FoldStartMessage) -> Result<()> {
        if self.stage != Stage::Read {
            return Err(Error::InvalidSessionState);
        }
        if self.plan.gu.is_some() {
            if self.gu_membership.is_none() {
                return Err(Error::InvalidSessionState);
            }
            if self.gu_columns_source != Some(GuColumnSource::Verified) {
                return Err(Error::InvalidSessionState);
            }
        }
        if message.tree_number != self.plan.mask_tree_number
            || message.claimed_group != self.plan.group_id
        {
            return Err(Error::StatementMismatch);
        }
        let gu_groups = if let Some(gu) = self.plan.gu.as_ref() {
            let membership = self
                .gu_membership
                .as_ref()
                .ok_or(Error::InvalidSessionState)?;
            membership.require(gu, &self.gu_registry)?;
            if membership.group != self.plan.group_id {
                return Err(Error::StatementMismatch);
            }
            let mut groups = self.plan.statement.groups().clone();
            for group in &mut groups.groups {
                if group.group == membership.group {
                    group.members = membership.members.clone();
                }
            }
            Some(groups)
        } else {
            None
        };
        let oracles = self
            .plan
            .oracles
            .lock()
            .map_err(|_| Error::InvalidSessionState)?;
        match &gu_groups {
            Some(groups) => fold_start_verify_with_groups(
                &self.plan.fold,
                message,
                &self.plan.statement,
                &oracles,
                groups,
            )?,
            None => fold_start_verify(&self.plan.fold, message, &self.plan.statement, &oracles)?,
        }
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

    /// Draw the mask opening positions from the verifier's own transcript.
    /// The plan supplies only the query count; the positions are not the
    /// plan's list and are not supplied by the caller or the proof.
    pub fn draw_mask_queries(&mut self) -> Result<()> {
        if self.stage != Stage::FoldStarted {
            return Err(Error::InvalidSessionState);
        }
        let tree = self
            .plan
            .statement
            .mask_tree(self.plan.mask_tree_number)
            .ok_or(Error::StatementMismatch)?;
        let prefix = self.chain.message().prefix(KIND_POSITIONS, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        let mut positions = Vec::new();
        positions
            .try_reserve_exact(self.plan.mask_query_count)
            .map_err(|_| Error::ResourceLimit)?;
        for _ in 0..self.plan.mask_query_count {
            positions.push(self.chain.draw_position(tree.log_n)?);
        }
        self.mask_queries = unique_sorted(&positions);
        self.stage = Stage::MaskQueried;
        Ok(())
    }

    /// Check the mask opening at the positions the verifier's transcript drew.
    /// The plan's count is a different source from the drawn list; the
    /// low-level check accepts at most that many positions and does not prove
    /// the list's source or completeness. The list comes from this session's
    /// own chain in the harness, which is not connected to production and does
    /// not claim a complete reference round, proximity, table test or hiding
    /// closure. With a zero plan count there is nothing to open and the stage
    /// still advances; whether the block permits zero queries is a
    /// reference-block condition, and entering this stage does not mean the
    /// mask was authenticated.
    pub fn read_mask(&mut self, openings: &RuntimeTreeOpening) -> Result<()> {
        if self.stage != Stage::MaskQueried {
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
            self.plan.mask_query_count,
            &self.mask_queries,
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
    use crate::pcs::request::code::{build_code_plan, CodeParams};
    use crate::pcs::request::field::{Fp, Fp3, MODULUS};
use crate::pcs::request::fold::{
        fold_mask_make, fold_mask_make_tail, fold_mask_notice, fold_start_prove, MaskOracle,
        OracleMetadata, RuntimeOracles, VerifiedGroup, VerifiedGroups,
        VerifiedStatement,
    };
    use crate::pcs::request::params::{LEAF_SALT_MIN_BYTES, REQUEST_CHAIN_LABEL};
    use crate::pcs::request::gu::{GuNotice, GuOracleShape, GuRegistry, GuShape};
    use crate::pcs::request::read::{unique_sorted, HidingState, ProverReads};
    use crate::pcs::request::relation::{gu_table, GuRelationKind, GuWitness};
    use crate::pcs::request::runtime_tree::{RuntimeTreePart, RuntimeTreePlan};
    use crate::pcs::request::runtime_tree::{verify_mask_opening, MaskTreePlan};
    use crate::pcs::request::table::{
        commit_table, IndexReader, IndexRecord, IndexShape, TableDistances,
    };
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

    fn gu_shape() -> GuShape {
        GuShape {
            oracle: 10,
            group: 1,
            tree: RuntimeTreePlan {
                tree_number: 6,
                leaf_class: 3,
                log_n: 3,
                log_n_zk: 0,
                parts: vec![RuntimeTreePart {
                    main_symbols: 6,
                    mask_symbols: 0,
                }],
                salt_bytes: LEAF_SALT_MIN_BYTES,
                run: Digest([9; 32]),
                configuration: Digest([7; 32]),
            },
            part: 0,
            rows_high: false,
            u: GuOracleShape {
                code: build_code_plan(3, 6, 1).unwrap(),
                main_rows: 2,
                main_deg: 3,
                mask_rows: 0,
                mask_code: None,
                answers: 1,
            },
            index: shape(),
        }
    }

    fn gu_plan_with(
        group_code: Option<CodeParams>,
        members: Vec<u64>,
        oracle: u64,
    ) -> (FoldeqPlan, crate::pcs::request::table::CommittedTable) {
        let (mut plan, table) = plan_fixture_with_group(group_code, members);
        let mut gu = gu_shape();
        gu.oracle = oracle;
        plan.gu = Some(gu);
        (plan, table)
    }

    fn gu_plan_fixture() -> (FoldeqPlan, crate::pcs::request::table::CommittedTable) {
        gu_plan_with(Some(build_code_plan(3, 6, 1).unwrap()), vec![1, 2], 10)
    }
    fn gu_plan_two_groups(
        gu_group: u64,
    ) -> (FoldeqPlan, crate::pcs::request::table::CommittedTable) {
        let code = Some(build_code_plan(3, 6, 1).unwrap());
        let (mut plan, table) = plan_fixture_with_groups(vec![
            VerifiedGroup {
                group: 1,
                members: vec![1, 2],
                fresh: Some(9),
                code,
                mask_code: None,
            },
            VerifiedGroup {
                group: 2,
                members: vec![3, 4],
                fresh: Some(8),
                code,
                mask_code: None,
            },
        ]);
        let mut gu = gu_shape();
        gu.group = gu_group;
        plan.gu = Some(gu);
        (plan, table)
    }

    fn gu_plan_fresh_collision() -> (FoldeqPlan, crate::pcs::request::table::CommittedTable) {
        let (mut plan, table) =
            plan_fixture_with_group_fresh(Some(build_code_plan(3, 6, 1).unwrap()), vec![1, 2], Some(10));
        plan.gu = Some(gu_shape());
        (plan, table)
    }

    fn gu_plan_with_answers(
        answers: u32,
    ) -> (FoldeqPlan, crate::pcs::request::table::CommittedTable) {
        let (mut plan, table) = gu_plan_fixture();
        plan.gu.as_mut().unwrap().u.answers = answers;
        (plan, table)
    }

    fn open_gu_plan_from(plan: FoldeqPlan) -> FoldeqSession {
        let binder = FoldeqBinder::new(REQUEST_CHAIN_LABEL, plan.reader.configuration, &plan).unwrap();
        let chain = binder.chain().unwrap();
        let mut session = FoldeqSession::open(plan, binder, chain).unwrap();
        session
            .bind_first_query(&session.plan().first_query())
            .unwrap();
        session.seeds().unwrap();
        session.draw_gamma().unwrap();
        session
    }

    fn open_gu_plan() -> (FoldeqSession, crate::pcs::request::table::CommittedTable) {
        let (plan, table) = gu_plan_fixture();
        (open_gu_plan_from(plan), table)
    }

    fn open_gu_session() -> (FoldeqSession, crate::pcs::request::table::CommittedTable) {
        let (mut session, table) = open_gu_plan();
        session.send_gu(&Digest([4; 32])).unwrap();
        (session, table)
    }

    fn open_gu_with_root(
        root: Digest,
    ) -> (FoldeqSession, crate::pcs::request::table::CommittedTable) {
        let (mut session, table) = open_gu_plan();
        session.send_gu(&root).unwrap();
        (session, table)
    }

    fn plan_fixture_with_groups(
        groups: Vec<VerifiedGroup>,
    ) -> (FoldeqPlan, crate::pcs::request::table::CommittedTable) {
        let table = committed();
        let tree = mask_tree();
        let statement = Arc::new(
            VerifiedStatement::new(
                VerifiedGroups { groups },
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
            mask_query_count: 2,
            gamma_path: GammaPath::Lin { shape: LinShapeBinding { matrix_rows: 1, k_lo: 0, k_col: 2 } },
            gu: None,
        };
        (plan, table)
    }

    fn plan_fixture_with_group_fresh(
        group_code: Option<CodeParams>,
        members: Vec<u64>,
        fresh: Option<u64>,
    ) -> (FoldeqPlan, crate::pcs::request::table::CommittedTable) {
        plan_fixture_with_groups(vec![VerifiedGroup {
            group: 1,
            members,
            fresh,
            code: group_code,
            mask_code: None,
        }])
    }

    fn plan_fixture_with_group(
        group_code: Option<CodeParams>,
        members: Vec<u64>,
    ) -> (FoldeqPlan, crate::pcs::request::table::CommittedTable) {
        plan_fixture_with_group_fresh(group_code, members, Some(9))
    }

    fn plan_fixture() -> (FoldeqPlan, crate::pcs::request::table::CommittedTable) {
        plan_fixture_with_group(Some(build_code_plan(3, 6, 1).unwrap()), vec![1, 2])
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

    fn register_mask_start(session: &FoldeqSession, message: &FoldStartMessage) {
        let tree = *session
            .plan()
            .statement
            .mask_tree(session.plan().mask_tree_number)
            .unwrap();
        let statement = Arc::clone(&session.plan().statement);
        session
            .plan()
            .oracles
            .lock()
            .unwrap()
            .receive_mask(
                &statement,
                &tree,
                message.id,
                &message.notice,
                message.root,
            )
            .unwrap();
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
        let first_query = plan.first_query();
        let first = draw(plan.clone());
        let (same, _table) = plan_fixture();
        assert_eq!(first, draw(same));
        let mut other = plan.clone();
        other.gamma_path = GammaPath::Direct {
            columns: other.reader.shape.v,
        };
        assert_ne!(first, draw(other.clone()));
        // A replaced LIN offset does not fit this table's plain columns and is
        // rejected by the plan fault before any draw; its first query is a
        // different binding.
        let mut offset = plan;
        offset.gamma_path = GammaPath::Lin {
            shape: LinShapeBinding {
                matrix_rows: 1,
                k_lo: 1,
                k_col: 3,
            },
        };
        assert_ne!(first_query, offset.first_query());
        let binder = FoldeqBinder::new(
            REQUEST_CHAIN_LABEL,
            offset.reader.configuration,
            &offset,
        );
        assert!(binder.is_err());
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
        // Registration of a forged root is only a claim; the later opening
        // phase must reject it.
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
        session.start_fold(&start.message).unwrap();
        assert!(session.epsilon().is_some());
        assert_eq!(session.start(), Some(&start.message));
        // The opening positions come from the verifier's own transcript; the
        // plan supplies only the count.
        session.draw_mask_queries().unwrap();
        let positions = session.mask_queries().to_vec();
        assert!(positions.len() <= plan.mask_query_count);
        let mask_openings = start.tree.open(&positions).unwrap();
        assert!(forged_registry
            .authenticate_mask(
                tree,
                9,
                plan.mask_query_count,
                &positions,
                &mask_openings,
            )
            .is_err());
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
        // An empty opening list is allowed (nothing is opened or marked); a
        // list longer than the plan's draw count is rejected.
        empty_registry
            .authenticate_mask(
                tree,
                9,
                plan.mask_query_count,
                &[],
                &RuntimeTreeOpening::default(),
            )
            .unwrap();
        let long = start.tree.open(&[0u64, 1, 2]).unwrap();
        assert!(empty_registry
            .authenticate_mask(
                tree,
                9,
                plan.mask_query_count,
                &[0u64, 1, 2],
                &long,
            )
            .is_err());
        session.read_mask(&mask_openings).unwrap();
        assert!(session.read_mask(&mask_openings).is_err());
        assert!(verify_mask_opening(
            tree,
            &Digest([5; 32]),
            &positions,
            &mask_openings
        )
        .is_err());
        let mut forged = mask_openings;
        forged.columns[0].symbols[0] = forged.columns[0].symbols[0] + Fp::ONE;
        assert!(verify_mask_opening(
            tree,
            &start.message.root,
            &positions,
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
        bad.gamma_path = GammaPath::Lin { shape: LinShapeBinding { matrix_rows: 1, k_lo: 0, k_col: 1 } };
        assert!(FoldeqSession::open(bad, binder.clone(), chain.clone()).is_err());

        // A high-bit matrix is rejected by the reference row-index bound
        // before any shift; the checked multiply stays as a defensive guard,
        // since within the row and shift bounds the product cannot overflow.
        let mut bad = plan.clone();
        bad.gamma_path = GammaPath::Lin {
            shape: LinShapeBinding { matrix_rows: 1u64 << 63, k_lo: 1, k_col: 3 },
        };
        assert_eq!(bad.fault(), Some("the LIN shape binding has invalid numbers"));
        assert!(FoldeqSession::open(bad, binder.clone(), chain.clone()).is_err());

        // A matrix that does not overflow but exceeds the plain columns is
        // refused by the upper-bound check.
        let mut bad = plan.clone();
        bad.gamma_path = GammaPath::Lin {
            shape: LinShapeBinding { matrix_rows: 2, k_lo: 0, k_col: 2 },
        };
        assert_eq!(
            bad.fault(),
            Some("the matrix does not fit the table's plain columns")
        );
        assert!(FoldeqSession::open(bad, binder.clone(), chain.clone()).is_err());

        // The reference shape bound also limits the row index to 30 bits, so a
        // matrix with more than 2^30 rows is invalid regardless of the fit.
        let mut bad = plan.clone();
        bad.gamma_path = GammaPath::Lin {
            shape: LinShapeBinding { matrix_rows: (1u64 << 30) + 1, k_lo: 0, k_col: 2 },
        };
        assert_eq!(bad.fault(), Some("the LIN shape binding has invalid numbers"));
        assert!(FoldeqSession::open(bad, binder.clone(), chain.clone()).is_err());

        let mut bad = plan;
        bad.group_id = 7;
        assert!(FoldeqSession::open(bad, binder, chain).is_err());
    }

    #[test]
    fn lin_row_bound_accepts_the_limit_without_materializing() {
        let (mut plan, _table) = plan_fixture();
        // A shape whose plain width admits the largest row count; only the
        // numbers are built, no row or matrix is allocated.
        plan.reader = IndexReader {
            shape: IndexShape {
                code: build_code_plan(32, (1u64 << 30) + 3, 1).unwrap(),
                v: 0,
                answers: 1,
                salt_bytes: LEAF_SALT_MIN_BYTES,
                distances: TableDistances {
                    radius: 0,
                    later_radius: 0,
                    m_once: 0,
                    m_later: 0,
                },
            },
            configuration: Digest([9; 32]),
            root: Digest([7; 32]),
            record: IndexRecord { y: vec![Fp3::ZERO] },
        };
        plan.gamma_path = GammaPath::Lin {
            shape: LinShapeBinding { matrix_rows: 1u64 << 30, k_lo: 0, k_col: 0 },
        };
        // The plain width is exactly 2^30, so this row count is legal.
        assert_eq!(plan.fault(), None);
        plan.gamma_path = GammaPath::Lin {
            shape: LinShapeBinding { matrix_rows: (1u64 << 30) + 1, k_lo: 0, k_col: 0 },
        };
        assert_eq!(plan.fault(), Some("the LIN shape binding has invalid numbers"));
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

    #[test]
    fn gu_session_requires_seed_answers_record_then_queries() {
        let (mut session, _table) = open_gu_session();
        // The old skip path from the received root directly to draws is rejected.
        assert!(session.draw_queries().is_err());
        assert!(session.receive_gu_answer(Fp3::ONE).is_err());
        session.shared_seeds().unwrap();
        assert_eq!(session.gu_shared_seeds().len(), 1);
        assert_eq!(
            session.gu_shared_seeds()[0].z.len(),
            gu_shape().sample_bits().unwrap() as usize
        );
        assert!(session.shared_seeds().is_err());
        assert!(session.draw_queries().is_err());
        assert!(session.join_gu_record().is_err());
        session.receive_gu_answer(Fp3::from_u64(7)).unwrap();
        assert!(session.receive_gu_answer(Fp3::ONE).is_err());
        assert!(session.draw_queries().is_err());
        session.join_gu_record().unwrap();
        assert!(session.join_gu_record().is_err());
        assert!(session.join_gu_columns(&[]).is_err());
        assert_eq!(session.pending_gu_rows().len(), 2);
        session.draw_queries().unwrap();
        let queries = session.queries().to_vec();
        let unique = unique_sorted(&queries);
        let fixture: Vec<(u64, Vec<Fp3>)> = unique
            .iter()
            .map(|position| {
                (
                    *position,
                    (0..session.plan().reader.shape.rows())
                        .map(|row| Fp3::from_u64(300 + row + *position))
                        .collect(),
                )
            })
            .collect();
        let mut wrong = fixture.clone();
        if wrong.len() > 1 {
            wrong.pop();
        } else {
            let other = if wrong[0].0 == 0 { 1 } else { 0 };
            wrong.push((other, vec![Fp3::ZERO; session.plan().reader.shape.rows() as usize]));
        }
        assert!(session.join_gu_columns(&wrong).is_err());
        session.join_gu_columns(&fixture).unwrap();
        assert!(session.join_gu_columns(&fixture).is_err());
        assert!(session.draw_queries().is_err());
        let rows = session.pending_gu_rows();
        assert_eq!(rows.len(), 1 + 1 + queries.len());
        assert_eq!(rows.iter().filter(|row| row.kind == GuRelationKind::Sample).count(), 1);
        assert_eq!(rows.iter().filter(|row| row.kind == GuRelationKind::Record).count(), 1);
        assert_eq!(rows.iter().filter(|row| row.kind == GuRelationKind::Column).count(), queries.len());
        assert!(rows.iter().all(|row| row.oracle == 10 && row.group == 1));
    }

    #[test]
    fn gu_session_rejects_missing_and_out_of_order_stages() {
        let (plan, _table) = gu_plan_fixture();
        let binder = FoldeqBinder::new(REQUEST_CHAIN_LABEL, plan.reader.configuration, &plan).unwrap();
        let chain = binder.chain().unwrap();
        let mut session = FoldeqSession::open(plan, binder, chain).unwrap();
        assert!(session.shared_seeds().is_err());
        session
            .bind_first_query(&session.plan().first_query())
            .unwrap();
        assert!(session.shared_seeds().is_err());
        session.seeds().unwrap();
        session.draw_gamma().unwrap();
        assert!(session.draw_queries().is_err());
        session.send_gu(&Digest([4; 32])).unwrap();
        session.shared_seeds().unwrap();
        assert!(session.shared_seeds().is_err());
        assert!(session.join_gu_record().is_err());
        session.receive_gu_answer(Fp3::ONE).unwrap();
        assert!(session.draw_queries().is_err());
        assert!(session.join_gu_columns(&[]).is_err());
        session.join_gu_record().unwrap();
        assert!(session.join_gu_columns(&[]).is_err());
        assert!(session.read_queries(&[]).is_err());
    }

    #[test]
    fn gu_shared_seeds_are_drawn_from_the_received_root() {
        let (mut first, _) = open_gu_with_root(Digest([4; 32]));
        let (mut second, _) = open_gu_with_root(Digest([4; 32]));
        first.shared_seeds().unwrap();
        second.shared_seeds().unwrap();
        assert_eq!(first.transcript_value(), second.transcript_value());
        assert_eq!(first.gu_shared_seeds(), second.gu_shared_seeds());

        let (mut changed, _) = open_gu_with_root(Digest([5; 32]));
        assert_ne!(first.transcript_value(), changed.transcript_value());
        changed.shared_seeds().unwrap();
        assert_ne!(first.gu_shared_seeds(), changed.gu_shared_seeds());
    }

    #[test]
    fn gu_answer_round_feeds_the_following_draws() {
        let (mut first, _) = open_gu_with_root(Digest([4; 32]));
        first.shared_seeds().unwrap();
        first.receive_gu_answer(Fp3::ONE).unwrap();
        first.join_gu_record().unwrap();
        let first_value = *first.transcript_value();
        first.draw_queries().unwrap();

        let (mut second, _) = open_gu_with_root(Digest([4; 32]));
        second.shared_seeds().unwrap();
        second.receive_gu_answer(Fp3::from_u64(2)).unwrap();
        second.join_gu_record().unwrap();
        let second_value = *second.transcript_value();
        second.draw_queries().unwrap();

        assert_ne!(first_value, second_value);
        assert_ne!(first.queries(), second.queries());
    }

    #[test]
    fn gu_session_joins_the_trusted_group_once() {
        let (mut session, _) = open_gu_session();
        session.shared_seeds().unwrap();
        session.receive_gu_answer(Fp3::ONE).unwrap();
        session.join_gu_record().unwrap();
        assert!(session
            .pending_gu_rows()
            .iter()
            .all(|row| row.oracle == 10 && row.group == 1));

        let (wrong_code, _) = gu_plan_with(
            Some(build_code_plan(3, 8, 0).unwrap()),
            vec![1, 2],
            10,
        );
        let mut wrong = open_gu_plan_from(wrong_code);
        wrong.send_gu(&Digest([4; 32])).unwrap();
        wrong.shared_seeds().unwrap();
        wrong.receive_gu_answer(Fp3::ONE).unwrap();
        assert!(wrong.join_gu_record().is_err());

        let (duplicate, _) = gu_plan_with(
            Some(build_code_plan(3, 6, 1).unwrap()),
            vec![10],
            10,
        );
        let mut duplicate = open_gu_plan_from(duplicate);
        duplicate.send_gu(&Digest([4; 32])).unwrap();
        duplicate.shared_seeds().unwrap();
        duplicate.receive_gu_answer(Fp3::ONE).unwrap();
        assert!(duplicate.join_gu_record().is_err());
    }

    #[test]
    fn gu_session_requires_joined_columns_before_fold_and_uses_verified_reads() {
        let (mut session, table) = open_gu_session();
        session.shared_seeds().unwrap();
        session.receive_gu_answer(Fp3::ONE).unwrap();
        session.join_gu_record().unwrap();
        session.draw_queries().unwrap();
        let queries = session.queries().to_vec();
        let unique = unique_sorted(&queries);
        let openings = table.open_columns(&unique).unwrap();
        let mut by_position = std::collections::HashMap::new();
        for opening in &openings {
            by_position.insert(opening.position, opening.clone());
        }
        let ordered: Vec<ColumnOpening> = queries
            .iter()
            .map(|position| by_position[position].clone())
            .collect();
        let mut tampered = ordered.clone();
        tampered[0].symbols[0] = tampered[0].symbols[0] + Fp::ONE;
        assert!(session.read_queries(&tampered).is_err());
        let columns = session.read_queries(&ordered).unwrap();
        assert_eq!(columns.len(), queries.len());
        let fixture: Vec<(u64, Vec<Fp3>)> = unique
            .iter()
            .map(|position| {
                (
                    *position,
                    (0..session.plan().reader.shape.rows())
                        .map(|row| Fp3::from_u64(700 + row + *position))
                        .collect(),
                )
            })
            .collect();
        assert!(session.join_gu_columns(&fixture).is_err());
        let start = mask_start(session.plan());
        let tree = *session
            .plan()
            .statement
            .mask_tree(session.plan().mask_tree_number)
            .unwrap();
        let statement = Arc::clone(&session.plan().statement);
        session
            .plan()
            .oracles
            .lock()
            .unwrap()
            .receive_mask(
                &statement,
                &tree,
                start.message.id,
                &start.message.notice,
                start.message.root,
            )
            .unwrap();
        assert!(session.start_fold(&start.message).is_err());
        session.join_gu_read_columns().unwrap();
        assert!(session.join_gu_read_columns().is_err());
        session.start_fold(&start.message).unwrap();
        assert_eq!(session.pending_gu_rows().len(), 2 + queries.len());
    }

    #[test]
    fn gu_session_rows_check_one_independent_witness() {
        let (mut session, table) = open_gu_session();
        session.shared_seeds().unwrap();
        let gu = session.plan().gu.clone().unwrap();
        let shape = session.plan().reader.shape;
        let eq_gamma = foldeq_weights(session.gamma().unwrap(), shape.v).unwrap();
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
        let mut folded_message = vec![Fp3::ZERO; shape.code.ell as usize];
        for (row, weight) in eq_gamma.iter().enumerate() {
            for (entry, value) in messages[row].iter().enumerate() {
                folded_message[entry] =
                    folded_message[entry].add(&weight.mul(&Fp3::from_base(*value)));
            }
        }
        let mut folded_tail = vec![Fp3::ZERO; shape.code.r as usize];
        for (row, weight) in eq_gamma.iter().enumerate() {
            for (entry, value) in tails[row].iter().enumerate() {
                folded_tail[entry] =
                    folded_tail[entry].add(&weight.mul(&Fp3::from_base(*value)));
            }
        }
        let mut witness = gu_table(&gu, &folded_message, &folded_tail).unwrap();
        let row_bits = 1u64;
        let pad_virtual = (gu.u.main_rows - 1)
            + ((gu.u.code.ell - gu.u.answers as u64) << row_bits);
        let pad = ((pad_virtual & ((1u64 << row_bits) - 1)) * gu.u.code.ell
            + (pad_virtual >> row_bits)) as usize;
        witness.main[pad] = Fp3::from_u64(999);

        let notice = GuNotice {
            root: *session.gu_root().unwrap(),
            oracle: gu.oracle,
            tree_number: gu.tree.tree_number,
            part: gu.part,
        };
        let mut probe_registry = GuRegistry::new();
        probe_registry.receive(&gu, &notice).unwrap();
        let seeds = session.gu_shared_seeds().to_vec();
        let sample_probe =
            sample_row(&gu, &probe_registry, &seeds, 0, Fp3::ZERO).unwrap();
        let sample_target = sample_probe.evaluate(&witness).unwrap();
        session.receive_gu_answer(sample_target).unwrap();
        session.join_gu_record().unwrap();
        let rows = session.pending_gu_rows();
        let sample = rows.iter().find(|row| row.kind == GuRelationKind::Sample).unwrap();
        let record = rows.iter().find(|row| row.kind == GuRelationKind::Record).unwrap();
        assert_eq!(sample.evaluate(&witness).unwrap(), sample.target);
        assert_eq!(record.evaluate(&witness).unwrap(), record.target);

        session.draw_queries().unwrap();
        let queries = session.queries().to_vec();
        let unique = unique_sorted(&queries);
        let openings = table.open_columns(&unique).unwrap();
        let mut by_position = std::collections::HashMap::new();
        for opening in &openings {
            by_position.insert(opening.position, opening.clone());
        }
        let ordered: Vec<ColumnOpening> = queries
            .iter()
            .map(|position| by_position[position].clone())
            .collect();
        session.read_queries(&ordered).unwrap();
        session.join_gu_read_columns().unwrap();
        let rows = session.pending_gu_rows();
        let columns: Vec<_> = rows
            .iter()
            .filter(|row| row.kind == GuRelationKind::Column)
            .collect();
        assert_eq!(columns.len(), queries.len());
        for column in columns {
            assert_eq!(column.evaluate(&witness).unwrap(), column.target);
        }
    }

    #[test]
    fn gu_session_reuses_one_verified_column_for_repeated_draws() {
        let mut found = None;
        for answer in 0..64u64 {
            let (mut plan, table) = gu_plan_fixture();
            plan.column_draws = 8;
            plan.ledger = Arc::new(Mutex::new(ReadLedger::new(8)));
            let mut session = open_gu_plan_from(plan);
            session.send_gu(&Digest([4; 32])).unwrap();
            session.shared_seeds().unwrap();
            session
                .receive_gu_answer(Fp3::from_u64(answer))
                .unwrap();
            session.join_gu_record().unwrap();
            session.draw_queries().unwrap();
            if unique_sorted(session.queries()).len() < session.queries().len() {
                found = Some((session, table));
                break;
            }
        }
        let (mut session, table) = found.expect("a repeated draw in the searched range");
        let queries = session.queries().to_vec();
        let unique = unique_sorted(&queries);
        let openings = table.open_columns(&unique).unwrap();
        let mut by_position = std::collections::HashMap::new();
        for opening in &openings {
            by_position.insert(opening.position, opening.clone());
        }
        let ordered: Vec<ColumnOpening> = queries
            .iter()
            .map(|position| by_position[position].clone())
            .collect();
        let mut tampered = ordered.clone();
        let mut seen = std::collections::HashSet::new();
        let mut changed = false;
        for (index, position) in queries.iter().enumerate() {
            if !seen.insert(*position) {
                tampered[index].symbols[0] = tampered[index].symbols[0] + Fp::ONE;
                changed = true;
                break;
            }
        }
        assert!(changed);
        assert!(session.read_queries(&tampered).is_err());
        session.read_queries(&ordered).unwrap();
        assert_eq!(
            session.plan().ledger.lock().unwrap().used(),
            unique.len() as u64
        );
        assert_eq!(session.plan().column_draws, queries.len());
        session.join_gu_read_columns().unwrap();
        let rows = session.pending_gu_rows();
        let columns: Vec<_> = rows
            .iter()
            .filter(|row| row.kind == GuRelationKind::Column)
            .collect();
        assert_eq!(columns.len(), queries.len());
        let mut targets = std::collections::HashMap::new();
        let mut repeats = 0usize;
        for (position, row) in queries.iter().zip(columns) {
            match targets.get(position) {
                Some(target) => {
                    assert_eq!(*target, row.target);
                    repeats += 1;
                }
                None => {
                    targets.insert(*position, row.target);
                }
            }
        }
        assert!(repeats > 0);
    }

    #[test]
    fn gu_session_rejects_a_rebound_oracle_tree_or_group() {
        let (mut session, _) = open_gu_session();
        session.shared_seeds().unwrap();
        session.receive_gu_answer(Fp3::ONE).unwrap();
        session.plan.gu.as_mut().unwrap().oracle = 11;
        assert!(session.join_gu_record().is_err());

        let (mut session, _) = open_gu_session();
        session.shared_seeds().unwrap();
        session.receive_gu_answer(Fp3::ONE).unwrap();
        session.plan.gu.as_mut().unwrap().tree.tree_number += 1;
        assert!(session.join_gu_record().is_err());

        let (mut session, _) = open_gu_session();
        session.shared_seeds().unwrap();
        session.receive_gu_answer(Fp3::ONE).unwrap();
        session.plan.gu.as_mut().unwrap().group = 2;
        assert!(session.join_gu_record().is_err());
    }

    #[test]
    fn gu_session_closes_every_ordered_answer() {
        let (plan, _table) = gu_plan_with_answers(2);
        let mut session = open_gu_plan_from(plan);
        session.send_gu(&Digest([4; 32])).unwrap();
        session.shared_seeds().unwrap();
        assert_eq!(session.gu_shared_seeds().len(), 2);
        let before = *session.transcript_value();
        let gu = session.plan().gu.clone().unwrap();
        let notice = GuNotice {
            root: *session.gu_root().unwrap(),
            oracle: gu.oracle,
            tree_number: gu.tree.tree_number,
            part: gu.part,
        };
        let mut probe = GuRegistry::new();
        probe.receive(&gu, &notice).unwrap();
        let seeds = session.gu_shared_seeds().to_vec();
        let witness = GuWitness {
            main: vec![
                Fp3::ONE;
                usize::try_from(gu.u.main_rows * gu.u.code.ell).unwrap()
            ],
            mask: Vec::new(),
        };
        let first = sample_row(&gu, &probe, &seeds, 0, Fp3::ZERO)
            .unwrap()
            .evaluate(&witness)
            .unwrap();
        let second = sample_row(&gu, &probe, &seeds, 1, Fp3::ZERO)
            .unwrap()
            .evaluate(&witness)
            .unwrap();
        assert_ne!(first, second);

        session.receive_gu_answer(first).unwrap();
        assert_eq!(*session.transcript_value(), before);
        assert!(session.draw_queries().is_err());
        assert!(session.join_gu_record().is_err());
        session.receive_gu_answer(second).unwrap();
        let after = *session.transcript_value();
        assert_ne!(after, before);
        assert!(session.receive_gu_answer(Fp3::ONE).is_err());
        session.join_gu_record().unwrap();
        session.draw_queries().unwrap();
        let samples: Vec<_> = session
            .pending_gu_rows()
            .iter()
            .filter(|row| row.kind == GuRelationKind::Sample)
            .collect();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].target, first);
        assert_eq!(samples[1].target, second);

        let run = |first: Fp3, second: Fp3| {
            let (plan, _table) = gu_plan_with_answers(2);
            let mut session = open_gu_plan_from(plan);
            session.send_gu(&Digest([4; 32])).unwrap();
            session.shared_seeds().unwrap();
            let before = *session.transcript_value();
            session.receive_gu_answer(first).unwrap();
            let middle = *session.transcript_value();
            session.receive_gu_answer(second).unwrap();
            let after = *session.transcript_value();
            (before, middle, after)
        };
        let baseline = run(first, second);
        let first_changed = run(first.add(&Fp3::ONE), second);
        let second_changed = run(first, second.add(&Fp3::ONE));
        assert_eq!(baseline.0, baseline.1);
        assert_ne!(baseline.2, baseline.0);
        assert_ne!(baseline.2, first_changed.2);
        assert_ne!(baseline.2, second_changed.2);
    }

    #[test]
    fn gu_fixture_columns_do_not_pass_fold() {
        let (mut session, table) = open_gu_session();
        session.shared_seeds().unwrap();
        session.receive_gu_answer(Fp3::ONE).unwrap();
        session.join_gu_record().unwrap();
        session.draw_queries().unwrap();
        let queries = session.queries().to_vec();
        let unique = unique_sorted(&queries);
        let fixture: Vec<(u64, Vec<Fp3>)> = unique
            .iter()
            .map(|position| {
                (
                    *position,
                    (0..session.plan().reader.shape.rows())
                        .map(|row| Fp3::from_u64(900 + row + *position))
                        .collect(),
                )
            })
            .collect();
        session.join_gu_columns(&fixture).unwrap();
        let openings = table.open_columns(&unique).unwrap();
        let mut by_position = std::collections::HashMap::new();
        for opening in &openings {
            by_position.insert(opening.position, opening.clone());
        }
        let ordered: Vec<ColumnOpening> = queries
            .iter()
            .map(|position| by_position[position].clone())
            .collect();
        session.read_queries(&ordered).unwrap();
        assert!(session.join_gu_read_columns().is_err());
        let start = mask_start(session.plan());
        register_mask_start(&session, &start.message);
        assert!(session.start_fold(&start.message).is_err());
    }

    #[test]
    fn gu_wrong_fold_group_is_rejected() {
        let (plan, _table) = gu_plan_two_groups(2);
        assert!(plan.fault().is_some());
        assert!(FoldeqBinder::new(REQUEST_CHAIN_LABEL, plan.reader.configuration, &plan).is_err());
    }

    #[test]
    fn gu_fresh_id_collision_is_rejected() {
        let (plan, _table) = gu_plan_fresh_collision();
        let mut session = open_gu_plan_from(plan);
        session.send_gu(&Digest([4; 32])).unwrap();
        session.shared_seeds().unwrap();
        session.receive_gu_answer(Fp3::ONE).unwrap();
        assert!(session.join_gu_record().is_err());
    }

    #[test]
    fn gu_fold_consumes_the_joined_membership() {
        let (mut session, table) = open_gu_session();
        session.shared_seeds().unwrap();
        session.receive_gu_answer(Fp3::ONE).unwrap();
        session.join_gu_record().unwrap();
        session.draw_queries().unwrap();
        let queries = session.queries().to_vec();
        let unique = unique_sorted(&queries);
        let openings = table.open_columns(&unique).unwrap();
        let mut by_position = std::collections::HashMap::new();
        for opening in &openings {
            by_position.insert(opening.position, opening.clone());
        }
        let ordered: Vec<ColumnOpening> = queries
            .iter()
            .map(|position| by_position[position].clone())
            .collect();
        session.read_queries(&ordered).unwrap();
        session.join_gu_read_columns().unwrap();
        let start = mask_start(session.plan());
        register_mask_start(&session, &start.message);
        let membership = session.gu_membership.as_mut().unwrap();
        membership.members.push(9);
        membership.members.sort_unstable();
        assert!(session.start_fold(&start.message).is_err());
    }
}
