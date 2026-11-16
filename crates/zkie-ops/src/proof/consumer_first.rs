//! The consumer request's first stage: the plan member, the W/Eval handoff and
//! the stage seal.
//!
//! The first stage of one consumer request takes the fixed request statement,
//! the committed raw relation and the table's request-level `W = gᵀ M`:
//!
//! 1. the first-stage chain begins on the fixed statement, so the member, the
//!    W and Eval plan and the raw plan are bound before any challenge;
//! 2. the request claim draws the position coefficients, the per-head column
//!    coefficients and the batch challenge from that chain;
//! 3. the prover derives the raw target from its committed raw rows and the
//!    fixed coefficients and refuses a value that is not the plan member's;
//! 4. the committed raw relation runs: the exact rounding of the verified
//!    output rows is proven over the hidden accumulator with masked ends;
//! 5. one product sumcheck on the same chain proves the combined rank-one
//!    claim with the fixed coefficients and the plan member's target, and the
//!    derived pending row must equal the plan member;
//! 6. the request-level `W` is derived from the member's row weights over the
//!    authenticated table, committed, and its root absorbed; the Eval target
//!    is fixed and the Eval row is built on the plan's units;
//! 7. the chain value after the stage is the stage seal. The request chain's
//!    group test continues this same chain and only then closes the member.
//!
//! The module returns the BAT rows the group test must check: the Eval row,
//! the consumer mask row and the raw relation rows. It does not run the group
//! itself and does not claim the member closed.

use zkie_core::artifact::digest_bytes;
use zkie_core::common::codec::DecodeLimits;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::bat::{BatPart, BatRow, BatRowKind};
use zkie_engine::pcs::request::chain::Chain;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::table::{IndexShape, TableRowSource};
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};

use crate::proof::consumer_group::{
    prove_consumer_group, verify_consumer_group, CommittedConsumerGroupMask, ConsumerGroupPlan,
    ConsumerGroupProof, ConsumerGroupWitness,
};
use crate::proof::consumer_psc::{
    prove_consumer_psc_fixed, verify_consumer_psc_fixed, ConsumerMaskRow, ConsumerPscMasks,
    ConsumerPscProof,
};
use crate::proof::consumer_raw::{
    prove_consumer_raw, verify_consumer_raw, ConsumerRawFirst, ConsumerRawMaskWitness,
    ConsumerRawObligation, ConsumerRawPlan, ConsumerRawSecond,
};
use crate::proof::consumer_request::{
    consumer_combined_obligation, consumer_combined_raw, consumer_request_claim,
    ConsumerRequestClaim, ConsumerRequestPlan,
};
use crate::proof::hidden_member::{check_member_claim, PendingRow, WeightBlock};
use crate::proof::hidden_send_session::{
    commit_request_w, derive_w_values, pending_row_weights, request_eval_row, TableLayout, WShape,
};
use crate::proof::output_closure::CommittedValueGroup;
use crate::proof::projection_claim::{raw_target, ConsumerMatrixColumns};
use crate::proof::psc::multilinear_eval;

/// Version of the first-stage messages.
pub const CONSUMER_FIRST_VERSION: u64 = 1;
/// Chain label of the consumer request's first stage.
pub const CONSUMER_FIRST_CHAIN_LABEL: u64 = 130;
/// Transcript tag of the committed W root.
const CONSUMER_FIRST_W_TAG: u64 = 110;
/// Transcript tag of the Eval target.
const CONSUMER_FIRST_EVAL_TAG: u64 = 111;
/// Transcript tag of the plan member digest.
const CONSUMER_FIRST_MEMBER_TAG: u64 = 113;

fn push_digest(message: &mut zkie_engine::pcs::request::chain::ChainMessage, digest: &Digest) {
    for word in digest.0.chunks_exact(8) {
        message.push_u64(u64::from_le_bytes(
            word.try_into().expect("eight-byte word"),
        ));
    }
}

/// The fixed first-stage plan data of one consumer request, before the plan
/// member is known: the request statement, the committed raw plan, the W
/// geometry, the table geometry and the Eval row's mask units.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerFirstDraft {
    pub request: ConsumerRequestPlan,
    pub raw: ConsumerRawPlan,
    pub w_shape: WShape,
    pub w_oracle: u64,
    pub w_group: u64,
    pub table_shape: IndexShape,
    pub matrix_rows: u64,
    pub k_lo: u32,
    pub columns: u64,
}

impl ConsumerFirstDraft {
    pub fn fault(&self) -> Option<&'static str> {
        if self.request.fault().is_some() || self.raw.validate().is_err() {
            return Some("the consumer first-stage draft has an invalid request or raw plan");
        }
        let columns = self.matrix_rows.checked_shl(self.k_lo)?;
        if columns != self.columns || self.columns < 2 || !self.columns.is_power_of_two() {
            return Some("the W column count is not the table column count");
        }
        if self.table_shape.rows_checked().ok()? != self.matrix_rows {
            return Some("the table rows are not the matrix rows");
        }
        if self.w_shape.fault(self.columns, 1).is_some() {
            return Some("the W message shape does not cover the W vector");
        }
        if self.w_oracle == 0
            || self.w_group == 0
            || self.w_oracle == self.raw.mask.oracle
            || self.w_group == self.raw.raw.group
            || self.w_group == self.raw.bits.group
        {
            return Some("the W identifiers collide with another message");
        }
        None
    }

    pub fn check(&self) -> Result<()> {
        self.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))
    }

    fn raw_plan_bytes(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        let raw = &self.raw;
        writer.put_u64(CONSUMER_FIRST_VERSION);
        writer.put_u64(raw.shape.heads as u64);
        writer.put_u64(raw.shape.positions as u64);
        writer.put_u64(raw.shape.width);
        writer.put_u64(raw.shape.shift as u64);
        writer.put_u64(raw.shape.columns);
        for group in [&raw.raw, &raw.bits] {
            writer.put_u64(group.columns);
            writer.put_u64(group.group);
            writer.put_u64(group.oracle_base);
            writer.put_u64(group.openings as u64);
            writer.put_u64(group.shape.code.log_n as u64);
            writer.put_u64(group.shape.code.ell);
            writer.put_u64(group.shape.code.r);
            writer.put_u64(group.shape.main_rows);
            writer.put_u64(group.shape.answers as u64);
            writer.put_u64(group.shape.tree_number);
            writer.put_digest(&group.shape.run);
            writer.put_digest(&group.shape.configuration);
            writer.put_u64(group.roots.len() as u64);
            for root in &group.roots {
                writer.put_digest(root);
            }
        }
        writer.put_u64(raw.mask.oracle);
        writer.put_u64(raw.mask.base_slot);
        writer.put_digest(&raw.outputs_digest);
        writer.put_u64(raw.bound.records);
        writer.put_u64(raw.bound.ranges.x_abs_bound);
        writer.put_u64(raw.bound.ranges.weight_lo as u64);
        writer.put_u64(raw.bound.ranges.weight_hi as u64);
        writer.put_u64(raw.bound.shift as u64);
        writer.into_bytes()
    }

    /// The fixed statement of the stage, without the plan member: the member
    /// is derived by the stage from this same statement and is then plan data
    /// checked against the derived pending. The statement determines every
    /// coefficient; the member is bound separately before the stage seal.
    pub fn statement_bytes(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(CONSUMER_FIRST_VERSION);
        writer.put_digest(&self.request.statement_digest());
        writer.put_bytes(&self.raw_plan_bytes());
        writer.put_u64(self.w_shape.code.log_n as u64);
        writer.put_u64(self.w_shape.code.ell);
        writer.put_u64(self.w_shape.code.r);
        writer.put_u64(self.w_shape.main_rows);
        writer.put_u64(self.w_shape.answers as u64);
        writer.put_u64(self.w_shape.tree_number);
        writer.put_digest(&self.w_shape.run);
        writer.put_digest(&self.w_shape.configuration);
        writer.put_u64(self.w_oracle);
        writer.put_u64(self.w_group);
        writer.put_u64(self.table_shape.v as u64);
        writer.put_u64(self.matrix_rows);
        writer.put_u64(u64::from(self.k_lo));
        writer.put_u64(self.columns);
        writer.into_bytes()
    }

    pub fn statement_digest(&self) -> Digest {
        digest_bytes(&self.statement_bytes())
    }

    pub fn begin_chain(&self) -> Result<Chain> {
        self.check()?;
        Chain::begin(
            CONSUMER_FIRST_CHAIN_LABEL,
            &self.statement_digest(),
            &self.statement_bytes(),
            &self.request.table.identity.0,
            None,
        )
    }
}

/// The committed W randomness: one message, one tail row and one salt row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerWCommitRandomness {
    pub pads: Vec<Fp3>,
    pub tails: Vec<Vec<Fp3>>,
    pub salts: Vec<Vec<Vec<u8>>>,
}

impl ConsumerWCommitRandomness {
    pub fn check(&self, shape: &WShape) -> Result<()> {
        let rows = usize::try_from(shape.main_rows).map_err(|_| Error::ResourceLimit)?;
        let tail = usize::try_from(shape.code.r).map_err(|_| Error::ResourceLimit)?;
        let columns = usize::try_from(shape.code.n()).map_err(|_| Error::ResourceLimit)?;
        if self.pads.len() != shape.answers as usize
            || self.tails.len() != 1
            || self.tails[0].len() != rows * tail
            || self.salts.len() != 1
            || self.salts[0].len() != columns
        {
            return Err(Error::ShapeMismatch);
        }
        if self
            .salts
            .iter()
            .flatten()
            .any(|salt| salt.len() != shape.salt_bytes)
        {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }
}

/// The first-stage plan: the fixed draft and the plan member the stage must
/// reproduce. The member is plan data; its target is the claim the group test
/// later binds to the committed raw message, not a proof value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerFirstPlan {
    pub draft: ConsumerFirstDraft,
    pub member: PendingRow,
    /// The raw claim target of the request: the plan value the product
    /// sumcheck proves. The prover enforces it against its witness reduction;
    /// the verifier takes it from the plan, never from the proof.
    pub target: Fp3,
    /// The proximity group of the stage. With a group the stage's real first
    /// seal is the chain value after the group verification; without one the
    /// stage is a component fixture and its seal is the pre-group value.
    pub group: Option<ConsumerGroupPlan>,
}

impl ConsumerFirstPlan {
    pub fn new(draft: ConsumerFirstDraft, member: PendingRow, target: Fp3) -> Result<Self> {
        draft.check()?;
        if member.weights.is_empty()
            || member.psi.len() as u64 != draft.columns.trailing_zeros() as u64
            || member.psi.len() >= 32
        {
            return Err(Error::InvalidConfig);
        }
        for block in &member.weights {
            match block {
                WeightBlock::Scalar { row, .. } => {
                    if *row >= draft.matrix_rows {
                        return Err(Error::InvalidConfig);
                    }
                }
                WeightBlock::Dense {
                    first_row,
                    log_rows,
                    values,
                } => {
                    if values.len() as u64 != 1u64 << log_rows
                        || first_row
                            .checked_add(values.len() as u64)
                            .map_or(true, |end| end > draft.matrix_rows)
                    {
                        return Err(Error::InvalidConfig);
                    }
                }
            }
        }
        let plan = Self {
            draft,
            member,
            target,
            group: None,
        };
        plan.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))?;
        Ok(plan)
    }

    /// Attaches the proximity group to the stage. The group's member shapes
    /// must be the draft's raw, bit and W shapes and its roots must be the
    /// committed roots of the plan.
    pub fn with_group(mut self, group: ConsumerGroupPlan) -> Result<Self> {
        group.check()?;
        if group.shapes.raw_shape != self.draft.raw.raw.shape
            || group.shapes.bits_shape != self.draft.raw.bits.shape
            || group.shapes.w_shape != self.draft.w_shape
            || group.raw_roots != self.draft.raw.raw.roots
            || group.bits_roots != self.draft.raw.bits.roots
        {
            return Err(Error::StatementMismatch);
        }
        self.group = Some(group);
        Ok(self)
    }

    pub fn fault(&self) -> Option<&'static str> {
        if let Some(fault) = self.draft.fault() {
            return Some(fault);
        }
        if self.member.weights.is_empty()
            || self.member.psi.len() as u64 != self.draft.columns.trailing_zeros() as u64
        {
            return Some("the plan member does not fit the W column cube");
        }
        for block in &self.member.weights {
            match block {
                WeightBlock::Scalar { row, .. } => {
                    if *row >= self.draft.matrix_rows {
                        return Some("a plan member row weight leaves the matrix rows");
                    }
                }
                WeightBlock::Dense {
                    first_row, values, ..
                } => {
                    if first_row
                        .checked_add(values.len() as u64)
                        .map_or(true, |end| end > self.draft.matrix_rows)
                    {
                        return Some("a plan member row block leaves the matrix rows");
                    }
                }
            }
        }
        None
    }

    pub fn check(&self) -> Result<()> {
        self.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))
    }

    fn member_bytes(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(self.member.weights.len() as u64);
        for block in &self.member.weights {
            match block {
                WeightBlock::Scalar { row, value } => {
                    writer.put_u64(0);
                    writer.put_u64(*row);
                    writer.put_fp3(value);
                }
                WeightBlock::Dense {
                    first_row,
                    log_rows,
                    values,
                } => {
                    writer.put_u64(1);
                    writer.put_u64(*first_row);
                    writer.put_u64(u64::from(*log_rows));
                    writer.put_u64(values.len() as u64);
                    for value in values {
                        writer.put_fp3(value);
                    }
                }
            }
        }
        writer.put_u64(self.member.psi.len() as u64);
        for value in &self.member.psi {
            writer.put_fp3(value);
        }
        writer.put_fp3(&self.member.target);
        writer.put_u64(self.member.masks.len() as u64);
        for term in &self.member.masks {
            writer.put_u64(term.oracle);
            writer.put_u64(term.slot);
            writer.put_fp3(&term.coeff);
        }
        writer.into_bytes()
    }

    /// The stage statement is the draft's fixed statement: the plan member is
    /// not part of it, because the stage derives the member from that same
    /// statement. The member is plan data, bound into the seal after the stage
    /// checks it against the derived pending.
    pub fn statement_bytes(&self) -> Vec<u8> {
        self.draft.statement_bytes()
    }

    pub fn statement_digest(&self) -> Digest {
        self.draft.statement_digest()
    }

    pub fn begin_chain(&self) -> Result<Chain> {
        self.draft.begin_chain()
    }
}

/// The plan member is fixed before the stage seal: both sides absorb its
/// digest after the derived pending has been checked against it.
fn absorb_member(chain: &mut Chain, plan: &ConsumerFirstPlan) -> Result<()> {
    let mut bytes = plan.member_bytes();
    let mut writer = CanonicalWriter::new();
    writer.put_bytes(&bytes);
    writer.put_fp3(&plan.target);
    bytes = writer.into_bytes();
    let digest = digest_bytes(&bytes);
    let mut message = chain.message();
    message.push_u64(CONSUMER_FIRST_VERSION);
    message.push_u64(CONSUMER_FIRST_MEMBER_TAG);
    push_digest(&mut message, &digest);
    let prefix = message.prefix(zkie_engine::pcs::request::chain::KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn absorb_w_root(chain: &mut Chain, oracle: u64, root: &Digest) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(CONSUMER_FIRST_VERSION);
    message.push_u64(CONSUMER_FIRST_W_TAG);
    message.push_u64(oracle);
    push_digest(&mut message, root);
    let prefix = message.prefix(zkie_engine::pcs::request::chain::KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn absorb_eval_target(chain: &mut Chain, target: Fp3) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(CONSUMER_FIRST_VERSION);
    message.push_u64(CONSUMER_FIRST_EVAL_TAG);
    message.push_fp3(&target);
    let prefix = message.prefix(zkie_engine::pcs::request::chain::KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// The target the plan member must carry, derived from the witness rows and
/// the fixed coefficients: `Σ_h theta^h T_h`.
fn witness_target(
    plan: &ConsumerFirstPlan,
    claim: &ConsumerRequestClaim,
    raw_heads: &[Vec<Fp3>],
) -> Result<Fp3> {
    let positions = plan.draft.request.shape.positions as usize;
    let width = usize::try_from(plan.draft.request.head_width).map_err(|_| Error::ResourceLimit)?;
    if raw_heads.len() != claim.heads.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut total = Fp3::ZERO;
    for (head, rows) in raw_heads.iter().enumerate() {
        if rows.len() < positions * width {
            return Err(Error::ShapeMismatch);
        }
        let target = raw_target(
            positions,
            width,
            &rows[..positions * width],
            &claim.position,
            &claim.heads[head],
        )?;
        total = total.add(&claim.powers[head].mul(&target));
    }
    Ok(total)
}

fn check_bindings(
    plan: &ConsumerFirstPlan,
    activation: &[Fp3],
    outputs: &[Vec<i64>],
) -> Result<()> {
    plan.check()?;
    if crate::proof::consumer_request::consumer_activation_digest(activation)
        != plan.draft.request.activation_digest
    {
        return Err(Error::StatementMismatch);
    }
    if crate::proof::consumer_request::consumer_outputs_digest(&plan.draft.request, outputs)?
        != plan.draft.request.outputs_digest
    {
        return Err(Error::StatementMismatch);
    }
    Ok(())
}

fn consumer_mask_row_bat(plan: &ConsumerFirstPlan, mask_row: &ConsumerMaskRow) -> Result<BatRow> {
    let mut row = BatRow::new(BatRowKind::Plain);
    for (index, coefficient) in mask_row.coefficients.iter().enumerate() {
        let slot = mask_row
            .base_slot
            .checked_add(index as u64)
            .ok_or(Error::ResourceLimit)?;
        row.add_unit(mask_row.oracle, BatPart::Mask, slot, *coefficient);
    }
    row.target = mask_row.target;
    Ok(row)
}

fn raw_obligation_bats(
    plan: &ConsumerFirstPlan,
    obligation: &ConsumerRawObligation,
) -> Result<Vec<BatRow>> {
    let mask_oracle = plan.draft.raw.mask.oracle;
    let mut upper = BatRow::new(BatRowKind::Plain);
    if obligation.hiding_slots.len() != obligation.row.hiding.len() {
        return Err(Error::ShapeMismatch);
    }
    for (slot, coefficient) in obligation.hiding_slots.iter().zip(&obligation.row.hiding) {
        upper.add_unit(mask_oracle, BatPart::Mask, *slot, *coefficient);
    }
    upper.target = obligation.row.constant;
    let mut end_row = BatRow::new(BatRowKind::Plain);
    let end = &obligation.end;
    if !end.raw_columns.is_empty() {
        end_row.add_term(
            end.raw_oracle,
            BatPart::Main,
            end.raw_table(&plan.draft.raw.raw.shape)?,
        );
    }
    end_row.add_term(
        end.bits_oracle,
        BatPart::Main,
        end.bits_table(&plan.draft.raw.bits.shape)?,
    );
    end_row.add_unit(mask_oracle, BatPart::Mask, end.shown_slot, end.mask_scale);
    end_row.target = end.target;
    Ok(vec![upper, end_row])
}

/// Maps a stage row from the source oracles to the group's BAT members. The
/// mask units keep their compact table index: the BAT layout maps it into the
/// combined row itself.
fn remap_group_rows(
    plan: &ConsumerFirstPlan,
    group: &ConsumerGroupPlan,
    rows: &[BatRow],
) -> Result<Vec<BatRow>> {
    let heads = u64::from(group.shapes.entries.heads);
    let raw_base = plan.draft.raw.raw.oracle_base;
    let bits_base = plan.draft.raw.bits.oracle_base;
    let mask_oracle = plan.draft.raw.mask.oracle;
    let map = |oracle: u64| -> Result<u64> {
        if oracle == plan.draft.w_oracle {
            return Ok(group.w_bat);
        }
        if oracle == mask_oracle {
            return Ok(group.mask_bat);
        }
        if oracle >= raw_base && oracle - raw_base < heads {
            return Ok(group.raw_bat_base + (oracle - raw_base));
        }
        if oracle >= bits_base && oracle - bits_base < heads {
            return Ok(group.bits_bat_base + (oracle - bits_base));
        }
        Err(Error::StatementMismatch)
    };
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        if !row.runs.is_empty() || !row.forms.is_empty() {
            return Err(Error::StatementMismatch);
        }
        let mut mapped = BatRow::new(row.kind);
        mapped.target = row.target;
        for term in &row.terms {
            mapped.add_term(map(term.oracle)?, term.part, term.values.clone());
        }
        for unit in &row.units {
            let oracle = map(unit.oracle)?;
            let index = if unit.part == BatPart::Mask {
                let h = group.shapes.stages.first().ok_or(Error::InvalidConfig)?.h;
                let ell = group.shapes.first_mask.ell;
                if ell == 0 {
                    return Err(Error::InvalidConfig);
                }
                let row = unit.index / ell;
                let place = unit.index % ell;
                row.checked_add(place.checked_shl(h).ok_or(Error::ResourceLimit)?)
                    .ok_or(Error::ResourceLimit)?
            } else {
                unit.index
            };
            mapped.add_unit(oracle, unit.part, index, unit.coefficient);
        }
        out.push(mapped);
    }
    Ok(out)
}

/// The claim row of the reduction that starts the product sumcheck: the
/// request coefficients on the committed raw messages of every head must equal
/// the same target the PSC fixed. The verifier builds it from the chain claim
/// and the plan, never from the proof, so a self-consistent raw/rounding chain
/// with a different weighted PSC target is refused by this row.
fn raw_claim_row(
    plan: &ConsumerFirstPlan,
    claim: &crate::proof::consumer_request::ConsumerRequestClaim,
) -> Result<BatRow> {
    let layout = TableLayout::of(&plan.draft.raw.raw.shape);
    let mut row = BatRow::new(BatRowKind::Plain);
    for (head, coefficients) in claim.alpha_rows()?.into_iter().enumerate() {
        let table = layout.scatter(plan.draft.raw.raw.columns, &coefficients)?;
        row.add_term(
            plan.draft.raw.raw.oracle_base + head as u64,
            BatPart::Main,
            table,
        );
    }
    row.target = plan.target;
    Ok(row)
}

fn group_rows(
    plan: &ConsumerFirstPlan,
    eval_row: &BatRow,
    mask_row: &ConsumerMaskRow,
    raw_claim: &BatRow,
    raw_rows: &[ConsumerRawObligation],
) -> Result<Vec<BatRow>> {
    let mut rows = Vec::with_capacity(3 + raw_rows.len() * 2);
    rows.push(eval_row.clone());
    rows.push(consumer_mask_row_bat(plan, mask_row)?);
    rows.push(raw_claim.clone());
    for obligation in raw_rows {
        rows.extend(raw_obligation_bats(plan, obligation)?);
    }
    Ok(rows)
}

/// The BAT rows the first-stage group test must check: the Eval row on the W
/// message, the consumer mask row and the raw relation rows. The module
/// returns them as obligations; it never verifies them itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerFirstObligations {
    pub eval_row: BatRow,
    pub mask_row: ConsumerMaskRow,
    pub raw_rows: Vec<ConsumerRawObligation>,
    /// Every linear group row of the stage, in the fixed order: the Eval row,
    /// the consumer mask row, and per raw relation instance its `(U)` row then
    /// its `(E)` row. The group test takes exactly this list; the module never
    /// verifies it itself.
    pub group_rows: Vec<BatRow>,
}

/// The prover's committed raw and bit messages, the mask message and the W
/// encoding tail for the group run. The W commitment itself is built inside
/// the stage from the plan's randomness and must reproduce the group plan's W
/// root; the mask commitment is made against the group's mask root.
pub struct ConsumerFirstGroupWitness<'a> {
    pub raw: &'a CommittedValueGroup,
    pub raw_tails: &'a [Vec<Fp3>],
    pub bits: &'a CommittedValueGroup,
    pub bits_tails: &'a [Vec<Fp3>],
    pub mask: &'a CommittedConsumerGroupMask,
    pub mask_rows: Vec<Vec<Fp3>>,
    pub mask_tails: Vec<Vec<Fp3>>,
    pub w_tails: &'a [Vec<Fp3>],
    /// The run seed of the group's SEND test, from the caller's transcript
    /// source; it is never a fixed test constant in production.
    pub run_seed: Digest,
}

/// The public first-stage messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerFirstMessages {
    pub raw_first: ConsumerRawFirst,
    pub raw_second: ConsumerRawSecond,
    pub psc: ConsumerPscProof,
    pub w_root: Digest,
    pub eval_target: Fp3,
    pub psi: Vec<Fp3>,
    /// The chain value after the pre-group stage.
    pub stage_seal: Digest,
    /// The proximity group proof, when the plan carries a group.
    pub group: Option<ConsumerGroupProof>,
    /// The chain value after the group verification: the stage's real first
    /// seal when a group is present.
    pub first_seal: Digest,
}

impl ConsumerFirstMessages {
    /// Reads the wire prefix that does not depend on the plan: the committed
    /// raw and bit groups and the W root. The verifier uses it to build the
    /// plan that completes the same bytes; the full decode re-checks every
    /// field against that plan.
    pub fn peek_header(bytes: &[u8]) -> Result<(ConsumerRawFirst, Digest)> {
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        if reader.get_u64()? != CONSUMER_FIRST_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let raw_first = ConsumerRawFirst::decode(reader.get_bytes()?)?;
        let _raw_second = reader.get_bytes()?;
        let _psc = reader.get_bytes()?;
        let w_root = reader.get_digest()?;
        if w_root == Digest::ZERO {
            return Err(Error::InvalidProof);
        }
        Ok((raw_first, w_root))
    }

    pub fn encode(&self, plan: &ConsumerFirstPlan) -> Result<Vec<u8>> {
        self.check(plan)?;
        let mut writer = CanonicalWriter::new();
        writer.put_u64(CONSUMER_FIRST_VERSION);
        writer.put_bytes(&self.raw_first.encode()?);
        writer.put_bytes(&self.raw_second.encode(&plan.draft.raw)?);
        writer.put_bytes(&self.psc.encode(&plan.draft.request.shape)?);
        writer.put_digest(&self.w_root);
        writer.put_fp3(&self.eval_target);
        writer.put_u64(self.psi.len() as u64);
        for value in &self.psi {
            writer.put_fp3(value);
        }
        writer.put_digest(&self.stage_seal);
        writer.put_u64(u64::from(self.group.is_some()));
        if let Some(group) = &self.group {
            writer.put_bytes(&group.encode()?);
        }
        writer.put_digest(&self.first_seal);
        Ok(writer.into_bytes())
    }

    pub fn decode(plan: &ConsumerFirstPlan, bytes: &[u8]) -> Result<Self> {
        plan.check()?;
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        if reader.get_u64()? != CONSUMER_FIRST_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let raw_first = ConsumerRawFirst::decode(reader.get_bytes()?)?;
        let raw_second = ConsumerRawSecond::decode(&plan.draft.raw, reader.get_bytes()?)?;
        let psc = ConsumerPscProof::decode(&plan.draft.request.shape, reader.get_bytes()?)?;
        let w_root = reader.get_digest()?;
        let eval_target = reader.get_fp3()?;
        let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        let mut psi = Vec::with_capacity(count);
        for _ in 0..count {
            psi.push(reader.get_fp3()?);
        }
        let stage_seal = reader.get_digest()?;
        let group = match reader.get_u64()? {
            0 => None,
            1 => {
                let group_plan = plan.group.as_ref().ok_or(Error::InvalidProof)?;
                Some(ConsumerGroupProof::decode(group_plan, reader.get_bytes()?)?)
            }
            _ => return Err(Error::InvalidEncoding),
        };
        let first_seal = reader.get_digest()?;
        reader.finish()?;
        let messages = Self {
            raw_first,
            raw_second,
            psc,
            w_root,
            eval_target,
            psi,
            stage_seal,
            group,
            first_seal,
        };
        messages.check(plan)?;
        Ok(messages)
    }

    pub fn check(&self, plan: &ConsumerFirstPlan) -> Result<()> {
        plan.check()?;
        self.raw_first.check(&plan.draft.raw)?;
        self.raw_second.check(&plan.draft.raw)?;
        self.psc.check(&plan.draft.request.shape)?;
        if self.w_root == Digest::ZERO || self.stage_seal == Digest::ZERO {
            return Err(Error::InvalidProof);
        }
        if self.psi != plan.member.psi || self.eval_target != plan.member.target {
            return Err(Error::StatementMismatch);
        }
        match (&plan.group, &self.group) {
            (Some(group_plan), Some(group)) => {
                if group.group_seal != self.first_seal {
                    return Err(Error::StatementMismatch);
                }
                let _ = group.proximity_messages(group_plan)?;
            }
            (None, None) => {
                if self.first_seal != self.stage_seal {
                    return Err(Error::StatementMismatch);
                }
            }
            _ => return Err(Error::StatementMismatch),
        }
        Ok(())
    }
}

/// The setup pass: derives the plan member from the witness by running the
/// same first-stage statement and claim on a scratch chain. The scratch chain
/// begins exactly like the real one, so the coefficients and the pending row
/// are the stage's own; the plan member is then fixed before the real proof.
#[allow(clippy::too_many_arguments)]
pub fn derive_consumer_first_member(
    draft: &ConsumerFirstDraft,
    activation: &[Fp3],
    raw_heads: &[Vec<Fp3>],
    bits_heads: &[Vec<Fp3>],
    outputs: &[Vec<i64>],
    raw_first: &ConsumerRawFirst,
    raw_masks: &ConsumerRawMaskWitness,
    matrix_places: &dyn ConsumerMatrixColumns,
    psc_masks: &ConsumerPscMasks,
) -> Result<(PendingRow, Fp3)> {
    draft.check()?;
    // The scratch chain begins on the same fixed statement as the stage, so
    // the coefficients and the derived pending are the stage's own.
    let mut chain = draft.begin_chain()?;
    let claim = consumer_request_claim(&mut chain, &draft.request)?;
    let target = witness_target_draft(draft, &claim, raw_heads)?;
    // The member is derived with the stage's exact chain prefix: the committed
    // raw relation runs before the product sumcheck, so the pending row sees
    // the same challenges the real stage will.
    let _ = prove_consumer_raw(
        &mut chain, &draft.raw, raw_first, raw_heads, bits_heads, outputs, raw_masks,
    )?;
    let combined = consumer_combined_raw(&draft.request, raw_heads)?;
    let obligation = consumer_combined_obligation(&draft.request)?;
    let (_, outcome) = prove_consumer_psc_fixed(
        &mut chain,
        &obligation,
        activation,
        &combined,
        matrix_places,
        psc_masks,
        &claim.position,
        &claim.combined,
        target,
    )?;
    Ok((outcome.pending, target))
}

fn witness_target_draft(
    draft: &ConsumerFirstDraft,
    claim: &ConsumerRequestClaim,
    raw_heads: &[Vec<Fp3>],
) -> Result<Fp3> {
    let positions = draft.request.shape.positions as usize;
    let width = usize::try_from(draft.request.head_width).map_err(|_| Error::ResourceLimit)?;
    if raw_heads.len() != claim.heads.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut total = Fp3::ZERO;
    for (head, rows) in raw_heads.iter().enumerate() {
        if rows.len() < positions * width {
            return Err(Error::ShapeMismatch);
        }
        let target = raw_target(
            positions,
            width,
            &rows[..positions * width],
            &claim.position,
            &claim.heads[head],
        )?;
        total = total.add(&claim.powers[head].mul(&target));
    }
    Ok(total)
}

/// Prover side of the first stage. The witness rows are the committed
/// messages' rows; the plan member's target is enforced against the witness
/// reduction; the stage seal is the chain value after the implemented stage.
#[allow(clippy::too_many_arguments)]
pub fn prove_consumer_first(
    plan: &ConsumerFirstPlan,
    activation: &[Fp3],
    raw_heads: &[Vec<Fp3>],
    bits_heads: &[Vec<Fp3>],
    outputs: &[Vec<i64>],
    raw_first: &ConsumerRawFirst,
    raw_masks: &ConsumerRawMaskWitness,
    psc_masks: &ConsumerPscMasks,
    matrix_places: &dyn ConsumerMatrixColumns,
    table_source: &mut dyn TableRowSource,
    w_randomness: &ConsumerWCommitRandomness,
    group_witness: Option<&ConsumerFirstGroupWitness<'_>>,
) -> Result<(ConsumerFirstMessages, ConsumerFirstObligations)> {
    check_bindings(plan, activation, outputs)?;
    w_randomness.check(&plan.draft.w_shape)?;
    let mut chain = plan.begin_chain()?;
    let claim = consumer_request_claim(&mut chain, &plan.draft.request)?;
    let target = witness_target(plan, &claim, raw_heads)?;
    if target != plan.target {
        return Err(Error::InvalidProof);
    }
    let (raw_second, raw_rows) = prove_consumer_raw(
        &mut chain,
        &plan.draft.raw,
        raw_first,
        raw_heads,
        bits_heads,
        outputs,
        raw_masks,
    )?;
    let combined = consumer_combined_raw(&plan.draft.request, raw_heads)?;
    let obligation = consumer_combined_obligation(&plan.draft.request)?;
    let (psc, outcome) = prove_consumer_psc_fixed(
        &mut chain,
        &obligation,
        activation,
        &combined,
        matrix_places,
        psc_masks,
        &claim.position,
        &claim.combined,
        plan.target,
    )?;
    check_member_claim(&plan.member, &outcome.pending)?;
    absorb_member(&mut chain, plan)?;
    let weights = pending_row_weights(&plan.member, plan.draft.matrix_rows)?;
    let w_values = derive_w_values(
        &plan.draft.table_shape,
        plan.draft.matrix_rows,
        plan.draft.k_lo,
        plan.draft.columns,
        &weights,
        table_source,
    )?;
    let w_commit = commit_request_w(
        &plan.draft.w_shape,
        plan.draft.w_oracle,
        plan.draft.w_group,
        &w_values,
        &w_randomness.pads,
        &w_randomness.tails,
        &w_randomness.salts,
    )?;
    let w_root = *w_commit.roots.first().ok_or(Error::ShapeMismatch)?;
    absorb_w_root(&mut chain, plan.draft.w_oracle, &w_root)?;
    // The Eval row is the checked pending row itself: the same psi, target and
    // mask units the product sumcheck left. It is not an independently chosen
    // evaluation target.
    let member_masks: Vec<(u64, u64, Fp3)> = plan
        .member
        .masks
        .iter()
        .map(|term| (term.oracle, term.slot, term.coeff))
        .collect();
    absorb_eval_target(&mut chain, plan.member.target)?;
    let eval_row = request_eval_row(
        TableLayout::of(&plan.draft.w_shape),
        plan.draft.columns,
        plan.draft.w_oracle,
        &plan.member.psi,
        &member_masks,
        plan.member.target,
    )?;
    let raw_claim = raw_claim_row(plan, &claim)?;
    let stage_seal = *chain.value();
    let rows = group_rows(plan, &eval_row, &outcome.mask_row, &raw_claim, &raw_rows)?;
    let mut group = None;
    let mut first_seal = stage_seal;
    let mut chain = chain;
    if let Some(group_plan) = &plan.group {
        let witness = group_witness.ok_or(Error::InvalidConfig)?;
        if witness.raw.roots != group_plan.raw_roots
            || witness.bits.roots != group_plan.bits_roots
            || witness.mask.root != group_plan.mask_root
            || w_root != group_plan.w_roots[0]
        {
            return Err(Error::StatementMismatch);
        }
        let group_witness = ConsumerGroupWitness {
            raw: witness.raw,
            raw_tails: witness.raw_tails,
            bits: witness.bits,
            bits_tails: witness.bits_tails,
            w: &w_commit,
            w_tails: witness.w_tails,
            mask: witness.mask,
            mask_rows: witness.mask_rows.clone(),
            mask_tails: witness.mask_tails.clone(),
        };
        let bat_rows = remap_group_rows(plan, group_plan, &rows)?;
        let (proof, next) = prove_consumer_group(
            chain,
            group_plan,
            &group_witness,
            bat_rows,
            witness.run_seed,
        )?;
        first_seal = proof.group_seal;
        group = Some(proof);
        chain = next;
    } else if group_witness.is_some() {
        return Err(Error::StatementMismatch);
    }
    let _ = chain;
    let messages = ConsumerFirstMessages {
        raw_first: raw_first.clone(),
        raw_second,
        psc,
        w_root,
        eval_target: plan.member.target,
        psi: plan.member.psi.clone(),
        stage_seal,
        group,
        first_seal,
    };
    Ok((
        messages,
        ConsumerFirstObligations {
            eval_row,
            mask_row: outcome.mask_row,
            raw_rows,
            group_rows: rows,
        },
    ))
}

/// Verifier side of the first stage. The verifier checks the same statement,
/// the committed raw relation, the product sumcheck against the plan member's
/// target and the member equality, absorbs the same W root and Eval target and
/// requires the same stage seal. It never sees the matrix, the raw rows or the
/// mask entries.
pub fn verify_consumer_first(
    plan: &ConsumerFirstPlan,
    activation: &[Fp3],
    outputs: &[Vec<i64>],
    messages: &ConsumerFirstMessages,
) -> Result<ConsumerFirstObligations> {
    check_bindings(plan, activation, outputs)?;
    messages.check(plan)?;
    let mut chain = plan.begin_chain()?;
    let claim = consumer_request_claim(&mut chain, &plan.draft.request)?;
    let raw_rows = verify_consumer_raw(
        &mut chain,
        &plan.draft.raw,
        &messages.raw_first,
        outputs,
        &messages.raw_second,
    )?;
    let obligation = consumer_combined_obligation(&plan.draft.request)?;
    let outcome = verify_consumer_psc_fixed(
        &mut chain,
        &obligation,
        activation,
        &messages.psc,
        &claim.position,
        &claim.combined,
        plan.target,
    )?;
    check_member_claim(&plan.member, &outcome.pending)?;
    absorb_member(&mut chain, plan)?;
    absorb_w_root(&mut chain, plan.draft.w_oracle, &messages.w_root)?;
    // The Eval row is the checked pending row: plan member psi, target and
    // mask units. The wire target must equal the plan member target.
    absorb_eval_target(&mut chain, plan.member.target)?;
    let member_masks: Vec<(u64, u64, Fp3)> = plan
        .member
        .masks
        .iter()
        .map(|term| (term.oracle, term.slot, term.coeff))
        .collect();
    let eval_row = request_eval_row(
        TableLayout::of(&plan.draft.w_shape),
        plan.draft.columns,
        plan.draft.w_oracle,
        &plan.member.psi,
        &member_masks,
        plan.member.target,
    )?;
    if *chain.value() != messages.stage_seal {
        return Err(Error::InvalidProof);
    }
    let raw_claim = raw_claim_row(plan, &claim)?;
    let rows = group_rows(plan, &eval_row, &outcome.mask_row, &raw_claim, &raw_rows)?;
    let mut first_seal = messages.stage_seal;
    if let Some(group_plan) = &plan.group {
        let proof = messages.group.as_ref().ok_or(Error::InvalidProof)?;
        let bat_rows = remap_group_rows(plan, group_plan, &rows)?;
        let (seal, _next) = verify_consumer_group(chain, group_plan, bat_rows, proof)?;
        first_seal = seal;
    }
    if first_seal != messages.first_seal {
        return Err(Error::InvalidProof);
    }
    Ok(ConsumerFirstObligations {
        eval_row,
        mask_row: outcome.mask_row,
        raw_rows,
        group_rows: rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_core::common::field::{PrimeCharacteristicRing, PrimeField64};
    use zkie_engine::pcs::request::code::{build_code_plan, CodeParams};
    use zkie_engine::pcs::request::field::{fp_from_u64, Fp};
    use zkie_engine::pcs::request::fold::{ByteMaskCoins, MaskCoins};
    use zkie_engine::pcs::request::table::TableDistances;
    use crate::proof::projection_claim::PlaceRows;

    use crate::proof::column_session::fp_from_signed_word;
    use crate::proof::consumer_group::{
        commit_consumer_group_mask, derive_consumer_group_shapes, CommittedConsumerGroupMask,
        ConsumerGroupEntries, ConsumerGroupPlan, ConsumerGroupProof, ConsumerGroupWitness,
    };
    use crate::proof::consumer_psc::{
        ConsumerObligation, ConsumerPscMasks, ConsumerPscShape, ConsumerTable,
    };
    use crate::proof::consumer_raw::{
        ConsumerRawGroup, ConsumerRawMaskPlan, ConsumerRawPlan, ConsumerRawShape,
    };
    use crate::proof::consumer_request::{
        consumer_activation_digest, consumer_outputs_digest, ConsumerIntegerRanges,
        ConsumerRequestPlan, ConsumerSharedPlan,
    };
    use crate::proof::output_closure::{commit_value_group, CommittedValueGroup};

    const HEADS: usize = 2;
    const POSITIONS: usize = 3;
    const WIDTH: usize = 4;
    const RECORDS: usize = 16;
    const PLACES: usize = 16;
    const SHIFT: u32 = 16;
    const COLUMNS: usize = 16;
    const ELL: usize = 32;
    const TABLE_R: usize = 4;
    const SALT: usize = 40;

    fn fp(value: u64) -> Fp3 {
        Fp3::from_base(fp_from_u64(value))
    }

    fn code() -> CodeParams {
        build_code_plan(12, ELL as u64, TABLE_R as u64).unwrap()
    }

    fn raw_shape() -> ConsumerRawShape {
        ConsumerRawShape {
            heads: HEADS as u32,
            positions: POSITIONS as u32,
            width: WIDTH as u64,
            shift: SHIFT,
            columns: COLUMNS as u64,
        }
    }

    fn raw_wshape(columns: usize, tree: u64) -> WShape {
        WShape::new(
            columns as u64,
            code(),
            1,
            HEADS as u32,
            SALT,
            tree,
            1,
            Digest([0x41; 32]),
            Digest([0x51; 32]),
        )
        .unwrap()
    }

    fn w_shape() -> WShape {
        WShape::new(
            COLUMNS as u64,
            code(),
            1,
            1,
            SALT,
            800,
            1,
            Digest([0x42; 32]),
            Digest([0x52; 32]),
        )
        .unwrap()
    }

    fn commit(
        shape: &WShape,
        oracle_base: u64,
        group: u64,
        values: &[Vec<Fp3>],
        coins: &mut ByteMaskCoins,
    ) -> CommittedValueGroup {
        let positions = values.len();
        let answers = shape.answers as usize;
        let mut pads = Vec::with_capacity(positions * answers);
        for _ in 0..positions * answers {
            let mut pad = Fp3::ZERO;
            while pad == Fp3::ZERO {
                pad = coins.draw().unwrap().0;
            }
            pads.push(pad);
        }
        let rows = shape.main_rows as usize;
        let tail = shape.code.r as usize;
        let tails: Vec<Vec<Fp3>> = (0..positions)
            .map(|_| (0..rows * tail).map(|_| coins.draw().unwrap().0).collect())
            .collect();
        let salts: Vec<Vec<Vec<u8>>> = (0..positions)
            .map(|_| {
                (0..shape.code.n() as usize)
                    .map(|_| {
                        let mut salt = Vec::new();
                        while salt.len() < shape.salt_bytes {
                            let (value, _) = coins.draw().unwrap();
                            for coefficient in value.c {
                                salt.extend_from_slice(
                                    &coefficient.as_canonical_u64().to_le_bytes(),
                                );
                            }
                        }
                        salt.truncate(shape.salt_bytes);
                        salt
                    })
                    .collect()
            })
            .collect();
        commit_value_group(
            shape,
            values[0].len() as u64,
            oracle_base,
            group,
            values,
            &pads,
            &tails,
            &salts,
            positions as u32,
        )
        .unwrap()
    }

    fn table_with(identity: Digest, configuration: Digest) -> ConsumerTable {
        ConsumerTable {
            path: "weights/L0_k_w_i32.bin".to_string(),
            oracle: 30,
            root: Digest([0x11; 32]),
            identity,
            configuration,
        }
    }

    fn table() -> ConsumerTable {
        table_with(Digest([0x12; 32]), Digest([0x13; 32]))
    }

    fn mask_source() -> crate::proof::consumer_psc::ConsumerMaskSource {
        crate::proof::consumer_psc::ConsumerMaskSource {
            oracle: 60,
            hiding_base: 100,
            shown_slot: 160,
        }
    }

    fn shared() -> ConsumerSharedPlan {
        ConsumerSharedPlan {
            m_once: 8,
            m_later: 8,
            one_time_columns: 2,
            later_columns: 2,
            declared_batches: 1,
            table_tail: 4,
        }
    }

    fn ranges() -> ConsumerIntegerRanges {
        ConsumerIntegerRanges {
            x_abs_bound: 1 << 20,
            weight_lo: -(1 << 18),
            weight_hi: (1 << 18) - 1,
        }
    }

    fn obligation_with(table: ConsumerTable, head: u32, source_step: u32) -> ConsumerObligation {
        ConsumerObligation {
            request: 7,
            source_step,
            activation_step: 9,
            head,
            shape: ConsumerPscShape {
                positions: POSITIONS as u32,
                places: PLACES as u64,
                records: RECORDS as u64,
                slice: (
                    u64::from(head) * WIDTH as u64,
                    (u64::from(head) + 1) * WIDTH as u64,
                ),
            },
            shift: i64::from(SHIFT),
            table,
            mask: mask_source(),
        }
    }

    fn obligation(head: u32, source_step: u32) -> ConsumerObligation {
        obligation_with(table(), head, source_step)
    }

    struct TableSource {
        messages: Vec<Vec<Fp>>,
        tails: Vec<Vec<Fp>>,
    }

    impl TableRowSource for TableSource {
        fn rows(&self) -> u64 {
            self.messages.len() as u64
        }
        fn row(&mut self, index: u64, message: &mut [Fp], tail: &mut [Fp]) -> Result<()> {
            let source = self
                .messages
                .get(index as usize)
                .ok_or(Error::ShapeMismatch)?;
            if source.len() != message.len() {
                return Err(Error::ShapeMismatch);
            }
            message.copy_from_slice(source);
            let source_tail = self.tails.get(index as usize).ok_or(Error::ShapeMismatch)?;
            if source_tail.len() != tail.len() {
                return Err(Error::ShapeMismatch);
            }
            tail.copy_from_slice(source_tail);
            Ok(())
        }
    }

    struct Fixture {
        plan: ConsumerFirstPlan,
        activation: Vec<Fp3>,
        raw: Vec<Vec<Fp3>>,
        bits: Vec<Vec<Fp3>>,
        outputs: Vec<Vec<i64>>,
        raw_first: ConsumerRawFirst,
        raw_masks: ConsumerRawMaskWitness,
        psc_masks: ConsumerPscMasks,
        matrix: Vec<Vec<Fp3>>,
        table_source: TableSource,
        w_randomness: ConsumerWCommitRandomness,
    }

    fn fixture() -> Fixture {
        let real = POSITIONS * WIDTH;
        let activation_words: Vec<i64> = (0..POSITIONS * RECORDS)
            .map(|index| ((index as i64 * 5 + 2) % 11) + 1)
            .collect();
        let matrix_words: Vec<Vec<i64>> = (0..PLACES)
            .map(|place| {
                (0..RECORDS)
                    .map(|record| ((record as i64 * 7 + place as i64 * 3 + 1) % 17) + 1)
                    .collect()
            })
            .collect();
        let activation: Vec<Fp3> = activation_words
            .iter()
            .map(|value| Fp3::from_base(fp_from_signed_word(*value as u64)))
            .collect();
        let matrix: Vec<Vec<Fp3>> = matrix_words
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| Fp3::from_base(fp_from_signed_word(*value as u64)))
                    .collect()
            })
            .collect();
        let mut raw_int = vec![vec![0i64; real]; HEADS];
        let mut outputs = vec![vec![0i64; real]; HEADS];
        for head in 0..HEADS {
            for position in 0..POSITIONS {
                for index in 0..WIDTH {
                    let place = head * WIDTH + index;
                    let mut sum = 0i64;
                    for record in 0..RECORDS {
                        sum += activation_words[position * RECORDS + record]
                            * matrix_words[place][record];
                    }
                    raw_int[head][position * WIDTH + index] = sum;
                    outputs[head][position * WIDTH + index] =
                        i64::try_from(crate::proof::relations::rq(i128::from(sum), SHIFT).unwrap())
                            .unwrap();
                }
            }
        }
        let raw: Vec<Vec<Fp3>> = raw_int
            .iter()
            .map(|head| {
                let mut values = vec![Fp3::ZERO; COLUMNS];
                for (entry, value) in head.iter().enumerate() {
                    values[entry] = signed(*value);
                }
                values
            })
            .collect();
        let mut bits = vec![vec![Fp3::ZERO; COLUMNS * SHIFT as usize]; HEADS];
        for head in 0..HEADS {
            for entry in 0..real {
                let remainder = raw_int[head][entry] + (1i64 << (SHIFT - 1))
                    - outputs[head][entry] * (1i64 << SHIFT);
                for bit in 0..SHIFT as usize {
                    bits[head][entry * SHIFT as usize + bit] = fp(((remainder >> bit) & 1) as u64);
                }
            }
        }
        // Consumer request statement.
        let mut request = ConsumerRequestPlan::new(
            7,
            table(),
            mask_source(),
            shared(),
            ranges(),
            Digest([0x41; 32]),
            Digest([0x42; 32]),
            vec![obligation(0, 20), obligation(1, 21)],
        )
        .unwrap();
        let activation_digest = consumer_activation_digest(&activation);
        let outputs_digest = consumer_outputs_digest(&request, &outputs).unwrap();
        request = ConsumerRequestPlan::new(
            7,
            table(),
            mask_source(),
            shared(),
            ranges(),
            activation_digest,
            outputs_digest,
            vec![obligation(0, 20), obligation(1, 21)],
        )
        .unwrap();
        // Committed raw and bit messages.
        let mut coins = ByteMaskCoins::new(Digest([0x77; 32])).unwrap();
        let raw_shape_msg = raw_wshape(COLUMNS, 700);
        let bits_shape_msg = raw_wshape(COLUMNS * SHIFT as usize, 701);
        let raw_group = commit(&raw_shape_msg, 20, 41, &raw, &mut coins);
        let bits_group = commit(&bits_shape_msg, 30, 42, &bits, &mut coins);
        let raw_plan = ConsumerRawPlan {
            shape: raw_shape(),
            raw: ConsumerRawGroup {
                shape: raw_shape_msg,
                columns: COLUMNS as u64,
                group: 41,
                oracle_base: 20,
                openings: 1,
                roots: raw_group.roots.clone(),
            },
            bits: ConsumerRawGroup {
                shape: bits_shape_msg,
                columns: (COLUMNS * SHIFT as usize) as u64,
                group: 42,
                oracle_base: 30,
                openings: 1,
                roots: bits_group.roots.clone(),
            },
            mask: ConsumerRawMaskPlan {
                oracle: 60,
                base_slot: 200,
            },
            outputs_digest,
            bound: crate::proof::consumer_request::ConsumerIntegerBound::new(
                RECORDS as u64,
                ranges(),
                SHIFT,
            )
            .unwrap(),
        };
        raw_plan.validate().unwrap();
        let raw_first = ConsumerRawFirst::from_groups(&raw_group.group, &bits_group.group);
        let spaces = raw_plan.mask_spaces().unwrap();
        let mut raw_masks = ConsumerRawMaskWitness::default();
        let mut counter = 1u64;
        for space in &spaces {
            for slot in &space.hiding {
                raw_masks.insert(*slot, fp(counter));
                counter += 1;
            }
            raw_masks.insert(space.shown, fp(counter));
            counter += 1;
        }
        // PSC masks over the combined shape.
        let psc_shape = request.shape;
        let hiding = (0..psc_shape.hiding_entries().unwrap())
            .map(|index| fp(300 + index as u64))
            .collect();
        let psc_masks = ConsumerPscMasks::new(hiding, fp(400));
        // First-stage draft.
        let draft = ConsumerFirstDraft {
            request,
            raw: raw_plan,
            w_shape: w_shape(),
            w_oracle: 70,
            w_group: 43,
            table_shape: IndexShape {
                code: code(),
                v: 4,
                answers: 1,
                salt_bytes: SALT,
                distances: TableDistances {
                    radius: 0,
                    later_radius: 0,
                    m_once: 0,
                    m_later: 0,
                },
            },
            matrix_rows: PLACES as u64,
            k_lo: 0,
            columns: COLUMNS as u64,
            };
        draft.check().unwrap();
        // W randomness.
        let mut coins = ByteMaskCoins::new(Digest([0x78; 32])).unwrap();
        let mut pads = vec![Fp3::ZERO; draft.w_shape.answers as usize];
        for pad in &mut pads {
            while pad == &Fp3::ZERO {
                *pad = coins.draw().unwrap().0;
            }
        }
        let rows: usize = draft.w_shape.main_rows as usize;
        let tail_len: usize = draft.w_shape.code.r as usize;
        let tails: Vec<Vec<Fp3>> = (0..1)
            .map(|_| {
                (0..rows * tail_len)
                    .map(|_| coins.draw().unwrap().0)
                    .collect()
            })
            .collect();
        let salts: Vec<Vec<Vec<u8>>> = (0..1)
            .map(|_| {
                (0..draft.w_shape.code.n() as usize)
                    .map(|_| {
                        let mut salt = Vec::new();
                        while salt.len() < draft.w_shape.salt_bytes {
                            let (value, _) = coins.draw().unwrap();
                            for coefficient in value.c {
                                salt.extend_from_slice(
                                    &coefficient.as_canonical_u64().to_le_bytes(),
                                );
                            }
                        }
                        salt.truncate(draft.w_shape.salt_bytes);
                        salt
                    })
                    .collect()
            })
            .collect();
        let w_randomness = ConsumerWCommitRandomness { pads, tails, salts };
        w_randomness.check(&draft.w_shape).unwrap();
        // Table source for W.
        let table_source = TableSource {
            // The committed table stores one row per record: entry `place` of
            // record `record` is M[place, record], the same matrix the PSC
            // consumes.
            messages: (0..RECORDS)
                .map(|record| {
                    let mut message = vec![Fp::ZERO; ELL];
                    for place in 0..PLACES {
                        message[place] =
                            fp_from_signed_word(matrix_words[place][record] as u64);
                    }
                    message
                })
                .collect(),
            tails: (0..PLACES)
                .map(|_| (0..TABLE_R).map(|_| Fp::ZERO).collect())
                .collect(),
        };
        // Setup pass derives the plan member on the same statement.
        let (member, target) = derive_consumer_first_member(
            &draft,
            &activation,
            &raw,
            &bits,
            &outputs,
            &raw_first,
            &raw_masks,
            &PlaceRows::new(&matrix),
            &psc_masks,
        )
        .unwrap();
        let plan = ConsumerFirstPlan::new(draft, member, target).unwrap();
        Fixture {
            plan,
            activation,
            raw,
            bits,
            outputs,
            raw_first,
            raw_masks,
            psc_masks,
            matrix,
            table_source,
            w_randomness,
        }
    }

    fn signed(value: i64) -> Fp3 {
        Fp3::from_base(fp_from_signed_word(value as u64))
    }

    fn prove(fixture: &mut Fixture) -> Result<(ConsumerFirstMessages, ConsumerFirstObligations)> {
        prove_consumer_first(
            &fixture.plan,
            &fixture.activation,
            &fixture.raw,
            &fixture.bits,
            &fixture.outputs,
            &fixture.raw_first,
            &fixture.raw_masks,
            &fixture.psc_masks,
            &PlaceRows::new(&fixture.matrix),
            &mut fixture.table_source,
            &fixture.w_randomness,
            None,
        )
    }

    fn witness_tables(fixture: &Fixture) -> std::collections::BTreeMap<u64, Vec<Fp3>> {
        use std::collections::BTreeMap;
        let mut tables = BTreeMap::new();
        for head in 0..HEADS {
            tables.insert(
                20 + head as u64,
                TableLayout::of(&fixture.plan.draft.raw.raw.shape)
                    .scatter(COLUMNS as u64, &fixture.raw[head])
                    .unwrap(),
            );
            tables.insert(
                30 + head as u64,
                TableLayout::of(&fixture.plan.draft.raw.bits.shape)
                    .scatter((COLUMNS * SHIFT as usize) as u64, &fixture.bits[head])
                    .unwrap(),
            );
        }
        let weights =
            pending_row_weights(&fixture.plan.member, fixture.plan.draft.matrix_rows).unwrap();
        let mut source = FixtureSource(fixture);
        let w_values = derive_w_values(
            &fixture.plan.draft.table_shape,
            fixture.plan.draft.matrix_rows,
            fixture.plan.draft.k_lo,
            fixture.plan.draft.columns,
            &weights,
            &mut source,
        )
        .unwrap();
        tables.insert(
            fixture.plan.draft.w_oracle,
            TableLayout::of(&fixture.plan.draft.w_shape)
                .scatter(fixture.plan.draft.columns, &w_values)
                .unwrap(),
        );
        let mut mask = vec![Fp3::ZERO; 512];
        for (index, value) in fixture.psc_masks.entries.hiding.iter().enumerate() {
            mask[fixture.plan.draft.request.mask.hiding_base as usize + index] = *value;
        }
        mask[fixture.plan.draft.request.mask.shown_slot as usize] =
            fixture.psc_masks.entries.shown[1];
        for space in fixture.plan.draft.raw.mask_spaces().unwrap() {
            for slot in &space.hiding {
                mask[*slot as usize] = fixture.raw_masks.entry(*slot).unwrap();
            }
            mask[space.shown as usize] = fixture.raw_masks.entry(space.shown).unwrap();
        }
        mask[5] = fp(3);
        tables.insert(fixture.plan.draft.raw.mask.oracle, mask);
        tables
    }

    fn evaluate_bat_row(row: &BatRow, tables: &std::collections::BTreeMap<u64, Vec<Fp3>>) -> Fp3 {
        let mut value = Fp3::ZERO;
        for term in &row.terms {
            let table = tables.get(&term.oracle).expect("a member table");
            assert_eq!(term.values.len(), table.len());
            for (coefficient, entry) in term.values.iter().zip(table) {
                value = value.add(&coefficient.mul(entry));
            }
        }
        for unit in &row.units {
            let table = tables.get(&unit.oracle).expect("a member table");
            value = value.add(&unit.coefficient.mul(&table[unit.index as usize]));
        }
        assert!(row.runs.is_empty() && row.forms.is_empty());
        value
    }

    #[test]
    fn the_first_stage_binds_the_member_w_and_eval() {
        let mut fixture = fixture();
        let (messages, proven) = prove(&mut fixture).unwrap();
        let verified = verify_consumer_first(
            &fixture.plan,
            &fixture.activation,
            &fixture.outputs,
            &messages,
        )
        .unwrap();
        assert_eq!(proven, verified);
        // Every linear group row holds on the witness tables; the group test
        // checks the same rows against the commitments.
        assert_eq!(proven.group_rows.len(), 3 + proven.raw_rows.len() * 2);
        let tables = witness_tables(&fixture);
        for row in &proven.group_rows {
            assert_eq!(evaluate_bat_row(row, &tables), row.target);
        }
        assert_eq!(messages.psi, fixture.plan.member.psi);
        assert_ne!(messages.stage_seal, Digest::ZERO);
        assert_ne!(messages.w_root, Digest::ZERO);
        assert_eq!(proven.eval_row.target, messages.eval_target);
        assert_eq!(messages.eval_target, fixture.plan.member.target);
        // The Eval row's numeric identity over the witness tables is checked
        // by the group-row loop above; its target is the pending row's own.
        let decoded =
            ConsumerFirstMessages::decode(&fixture.plan, &messages.encode(&fixture.plan).unwrap())
                .unwrap();
        assert_eq!(decoded, messages);
        verify_consumer_first(
            &fixture.plan,
            &fixture.activation,
            &fixture.outputs,
            &decoded,
        )
        .unwrap();
    }

    struct FixtureSource<'a>(&'a Fixture);

    impl TableRowSource for FixtureSource<'_> {
        fn rows(&self) -> u64 {
            self.0.table_source.messages.len() as u64
        }
        fn row(&mut self, index: u64, message: &mut [Fp], tail: &mut [Fp]) -> Result<()> {
            let source = self
                .0
                .table_source
                .messages
                .get(index as usize)
                .ok_or(Error::ShapeMismatch)?;
            message.copy_from_slice(source);
            let source_tail = self
                .0
                .table_source
                .tails
                .get(index as usize)
                .ok_or(Error::ShapeMismatch)?;
            tail.copy_from_slice(source_tail);
            Ok(())
        }
    }

    #[test]
    fn a_changed_member_witness_or_seal_is_refused() {
        let mut fixture = fixture();
        let (messages, _) = prove(&mut fixture).unwrap();
        // The plan member target is fixed: another target is refused by the
        // prover's witness check.
        let bad_plan = ConsumerFirstPlan::new(
            fixture.plan.draft.clone(),
            fixture.plan.member.clone(),
            fixture.plan.target.add(&fp(1)),
        )
        .unwrap();
        assert!(prove_consumer_first(
            &bad_plan,
            &fixture.activation,
            &fixture.raw,
            &fixture.bits,
            &fixture.outputs,
            &fixture.raw_first,
            &fixture.raw_masks,
            &fixture.psc_masks,
            &PlaceRows::new(&fixture.matrix),
            &mut fixture.table_source,
            &fixture.w_randomness,
            None,
        )
        .is_err());
        // Another member point is refused at the message check.
        let mut bad_member = fixture.plan.member.clone();
        bad_member.psi[0] = bad_member.psi[0].add(&fp(1));
        let bad_plan =
            ConsumerFirstPlan::new(fixture.plan.draft.clone(), bad_member, fixture.plan.target)
                .unwrap();
        assert!(
            verify_consumer_first(&bad_plan, &fixture.activation, &fixture.outputs, &messages)
                .is_err()
        );
        // A changed W root or Eval target no longer reproduces the seal.
        let mut changed = messages.clone();
        changed.w_root = Digest([0x66; 32]);
        assert!(verify_consumer_first(
            &fixture.plan,
            &fixture.activation,
            &fixture.outputs,
            &changed
        )
        .is_err());
        let mut changed = messages.clone();
        changed.eval_target = changed.eval_target.add(&fp(1));
        assert!(verify_consumer_first(
            &fixture.plan,
            &fixture.activation,
            &fixture.outputs,
            &changed
        )
        .is_err());
        // A changed raw row breaks the witness target or the raw relation.
        let mut raw = fixture.raw.clone();
        raw[0][0] = raw[0][0].add(&fp(1));
        assert!(prove_consumer_first(
            &fixture.plan,
            &fixture.activation,
            &raw,
            &fixture.bits,
            &fixture.outputs,
            &fixture.raw_first,
            &fixture.raw_masks,
            &fixture.psc_masks,
            &PlaceRows::new(&fixture.matrix),
            &mut fixture.table_source,
            &fixture.w_randomness,
            None,
        )
        .is_err());
        // Another output row no longer matches the statement binding.
        fixture.outputs[0][0] += 1;
        assert!(verify_consumer_first(
            &fixture.plan,
            &fixture.activation,
            &fixture.outputs,
            &messages
        )
        .is_err());
    }
    // The plan row count covers the ten relation rows and one sample row per
    // member (two raw, two bits, mask and W).
    const GROUP_ROWS: u64 = 17;
    const GROUP_RUN: Digest = Digest([0x61; 32]);
    const GROUP_CONFIGURATION: Digest = Digest([0x62; 32]);

    struct GroupFixture {
        plan: ConsumerFirstPlan,
        activation: Vec<Fp3>,
        raw: Vec<Vec<Fp3>>,
        bits: Vec<Vec<Fp3>>,
        outputs: Vec<Vec<i64>>,
        raw_first: ConsumerRawFirst,
        raw_masks: ConsumerRawMaskWitness,
        psc_masks: ConsumerPscMasks,
        matrix: Vec<Vec<Fp3>>,
        table_source: TableSource,
        w_randomness: ConsumerWCommitRandomness,
        raw_group: CommittedValueGroup,
        raw_tails: Vec<Vec<Fp3>>,
        bits_group: CommittedValueGroup,
        bits_tails: Vec<Vec<Fp3>>,
        mask: CommittedConsumerGroupMask,
        mask_rows: Vec<Vec<Fp3>>,
        mask_tails: Vec<Vec<Fp3>>,
        run_seed: Digest,
    }

    fn many_salts(coins: &mut ByteMaskCoins, count: u64, len: usize) -> Vec<Vec<u8>> {
        (0..count)
            .map(|_| {
                let mut salt = Vec::new();
                while salt.len() < len {
                    let (value, _) = coins.draw().unwrap();
                    for coefficient in value.c {
                        salt.extend_from_slice(&coefficient.as_canonical_u64().to_le_bytes());
                    }
                }
                salt.truncate(len);
                salt
            })
            .collect()
    }

    fn commit_collect(
        shape: &WShape,
        oracle_base: u64,
        group: u64,
        values: &[Vec<Fp3>],
        coins: &mut ByteMaskCoins,
    ) -> (CommittedValueGroup, Vec<Vec<Fp3>>) {
        let positions = values.len();
        let mut pads = vec![Fp3::ZERO; positions * shape.answers as usize];
        for pad in &mut pads {
            while *pad == Fp3::ZERO {
                *pad = coins.draw().unwrap().0;
            }
        }
        let rows = shape.main_rows as usize;
        let tail_len = shape.code.r as usize;
        let tails: Vec<Vec<Fp3>> = (0..positions)
            .map(|_| {
                (0..rows * tail_len)
                    .map(|_| coins.draw().unwrap().0)
                    .collect()
            })
            .collect();
        let salts: Vec<Vec<Vec<u8>>> = (0..positions)
            .map(|_| many_salts(coins, shape.code.n(), shape.salt_bytes))
            .collect();
        let committed = commit_value_group(
            shape,
            values[0].len() as u64,
            oracle_base,
            group,
            values,
            &pads,
            &tails,
            &salts,
            positions as u32,
        )
        .unwrap();
        (committed, tails)
    }

    /// The stored-column view of M: row `k` (a record) carries `M[n,k]` for
    /// every place `n`. The first stage's W derivation reads the stored
    /// columns and weights the place axis, so W comes out over the record
    /// axis. The shared reader consumes the same view, and the Dot row then
    /// contracts the same M on the complementary axes.
    fn group_table_source(matrix_words: &[Vec<i64>]) -> TableSource {
        let places = matrix_words.len();
        let records = matrix_words.first().map(|row| row.len()).unwrap_or(0);
        TableSource {
            messages: (0..records)
                .map(|record| {
                    let mut message = vec![Fp::ZERO; ELL];
                    for place in 0..places {
                        message[place] = fp_from_signed_word(matrix_words[place][record] as u64);
                    }
                    message
                })
                .collect(),
            tails: (0..records)
                .map(|_| (0..TABLE_R).map(|_| Fp::ZERO).collect())
                .collect(),
        }
    }

    fn group_fixture() -> GroupFixture {
        group_fixture_with(table())
    }

    fn group_fixture_with(table: ConsumerTable) -> GroupFixture {
        let real = POSITIONS * WIDTH;
        let activation_words: Vec<i64> = (0..POSITIONS * RECORDS)
            .map(|index| ((index as i64 * 5 + 2) % 11) + 1)
            .collect();
        let matrix_words: Vec<Vec<i64>> = (0..PLACES)
            .map(|place| {
                (0..RECORDS)
                    .map(|record| ((record as i64 * 7 + place as i64 * 3 + 1) % 17) + 1)
                    .collect()
            })
            .collect();
        let activation: Vec<Fp3> = activation_words
            .iter()
            .map(|value| Fp3::from_base(fp_from_signed_word(*value as u64)))
            .collect();
        let matrix: Vec<Vec<Fp3>> = matrix_words
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| Fp3::from_base(fp_from_signed_word(*value as u64)))
                    .collect()
            })
            .collect();
        let mut raw_int = vec![vec![0i64; real]; HEADS];
        let mut outputs = vec![vec![0i64; real]; HEADS];
        for head in 0..HEADS {
            for position in 0..POSITIONS {
                for index in 0..WIDTH {
                    let place = head * WIDTH + index;
                    let mut sum = 0i64;
                    for record in 0..RECORDS {
                        sum += activation_words[position * RECORDS + record]
                            * matrix_words[place][record];
                    }
                    raw_int[head][position * WIDTH + index] = sum;
                    outputs[head][position * WIDTH + index] =
                        i64::try_from(crate::proof::relations::rq(i128::from(sum), SHIFT).unwrap())
                            .unwrap();
                }
            }
        }
        let raw: Vec<Vec<Fp3>> = raw_int
            .iter()
            .map(|head| {
                let mut values = vec![Fp3::ZERO; COLUMNS];
                for (entry, value) in head.iter().enumerate() {
                    values[entry] = signed(*value);
                }
                values
            })
            .collect();
        let mut bits = vec![vec![Fp3::ZERO; COLUMNS * SHIFT as usize]; HEADS];
        for head in 0..HEADS {
            for entry in 0..real {
                let remainder = raw_int[head][entry] + (1i64 << (SHIFT - 1))
                    - outputs[head][entry] * (1i64 << SHIFT);
                for bit in 0..SHIFT as usize {
                    bits[head][entry * SHIFT as usize + bit] = fp(((remainder >> bit) & 1) as u64);
                }
            }
        }
        let mut request = ConsumerRequestPlan::new(
            7,
            table.clone(),
            mask_source(),
            shared(),
            ranges(),
            Digest([0x41; 32]),
            Digest([0x42; 32]),
            vec![
                obligation_with(table.clone(), 0, 20),
                obligation_with(table.clone(), 1, 21),
            ],
        )
        .unwrap();
        let activation_digest = consumer_activation_digest(&activation);
        let outputs_digest = consumer_outputs_digest(&request, &outputs).unwrap();
        request = ConsumerRequestPlan::new(
            7,
            table.clone(),
            mask_source(),
            shared(),
            ranges(),
            activation_digest,
            outputs_digest,
            vec![
                obligation_with(table.clone(), 0, 20),
                obligation_with(table.clone(), 1, 21),
            ],
        )
        .unwrap();
        let entries = ConsumerGroupEntries {
            heads: HEADS as u32,
            raw_columns: COLUMNS as u64,
            bits_columns: (COLUMNS * SHIFT as usize) as u64,
            w_columns: COLUMNS as u64,
            mask_entries: 512,
            rows: GROUP_ROWS,
        };
        let shapes =
            derive_consumer_group_shapes(entries, GROUP_RUN, GROUP_CONFIGURATION, 100, 900)
                .unwrap();
        let mut coins = ByteMaskCoins::new(Digest([0x79; 32])).unwrap();
        let (raw_group, raw_tails) = commit_collect(&shapes.raw_shape, 20, 41, &raw, &mut coins);
        let (bits_group, bits_tails) =
            commit_collect(&shapes.bits_shape, 30, 42, &bits, &mut coins);
        let raw_plan = ConsumerRawPlan {
            shape: raw_shape(),
            raw: ConsumerRawGroup {
                shape: shapes.raw_shape.clone(),
                columns: COLUMNS as u64,
                group: 41,
                oracle_base: 20,
                openings: 1,
                roots: raw_group.roots.clone(),
            },
            bits: ConsumerRawGroup {
                shape: shapes.bits_shape.clone(),
                columns: (COLUMNS * SHIFT as usize) as u64,
                group: 42,
                oracle_base: 30,
                openings: 1,
                roots: bits_group.roots.clone(),
            },
            mask: ConsumerRawMaskPlan {
                oracle: 60,
                base_slot: 200,
            },
            outputs_digest,
            bound: crate::proof::consumer_request::ConsumerIntegerBound::new(
                RECORDS as u64,
                ranges(),
                SHIFT,
            )
            .unwrap(),
        };
        raw_plan.validate().unwrap();
        let raw_first = ConsumerRawFirst::from_groups(&raw_group.group, &bits_group.group);
        let spaces = raw_plan.mask_spaces().unwrap();
        let mut raw_masks = ConsumerRawMaskWitness::default();
        let mut counter = 1u64;
        for space in &spaces {
            for slot in &space.hiding {
                raw_masks.insert(*slot, fp(counter));
                counter += 1;
            }
            raw_masks.insert(space.shown, fp(counter));
            counter += 1;
        }
        let psc_shape = request.shape;
        let hiding = (0..psc_shape.hiding_entries().unwrap())
            .map(|index| fp(300 + index as u64))
            .collect();
        let psc_masks = ConsumerPscMasks::new(hiding, fp(400));
        let mask_ell = usize::try_from(shapes.first_mask.ell).unwrap();
        let mask_total = usize::try_from(shapes.mask_rows).unwrap() * mask_ell;
        let mut mask = vec![Fp3::ZERO; mask_total];
        for (index, value) in psc_masks.entries.hiding.iter().enumerate() {
            mask[request.mask.hiding_base as usize + index] = *value;
        }
        mask[request.mask.shown_slot as usize] = psc_masks.entries.shown[1];
        for space in &spaces {
            for slot in &space.hiding {
                mask[*slot as usize] = raw_masks.entry(*slot).unwrap();
            }
            mask[space.shown as usize] = raw_masks.entry(space.shown).unwrap();
        }
        mask[5] = fp(3);
        let mask_rows: Vec<Vec<Fp3>> = mask.chunks(mask_ell).map(|row| row.to_vec()).collect();
        let mask_tails: Vec<Vec<Fp3>> = (0..mask_rows.len())
            .map(|_| {
                (0..shapes.first_mask.r as usize)
                    .map(|_| coins.draw().unwrap().0)
                    .collect()
            })
            .collect();
        let mask_salts = many_salts(&mut coins, shapes.first_mask.n(), 40);
        let placeholder = ConsumerGroupPlan {
            shapes: shapes.clone(),
            group_id: 41,
            fresh: 51,
            raw_oracle_base: 20,
            bits_oracle_base: 30,
            mask_oracle: 60,
            w_oracle: 70,
            raw_bat_base: 0,
            bits_bat_base: HEADS as u64,
            mask_bat: 2 * HEADS as u64,
            w_bat: 2 * HEADS as u64 + 1,
            mask_root: Digest::ZERO,
            raw_roots: raw_group.roots.clone(),
            bits_roots: bits_group.roots.clone(),
            w_roots: vec![Digest([0x71; 32])],
        };
        let mask_commit = commit_consumer_group_mask(
            &placeholder,
            mask_rows.clone(),
            mask_tails.clone(),
            mask_salts,
        )
        .unwrap();
        let draft = ConsumerFirstDraft {
            request,
            raw: raw_plan,
            w_shape: shapes.w_shape.clone(),
            w_oracle: 70,
            w_group: 43,
            table_shape: IndexShape {
                code: code(),
                v: 4,
                answers: 1,
                salt_bytes: SALT,
                distances: TableDistances {
                    radius: 0,
                    later_radius: 0,
                    m_once: 0,
                    m_later: 0,
                },
            },
            matrix_rows: PLACES as u64,
            k_lo: 0,
            columns: COLUMNS as u64,
            };
        draft.check().unwrap();
        let mut pads = vec![Fp3::ZERO; draft.w_shape.answers as usize];
        for pad in &mut pads {
            while *pad == Fp3::ZERO {
                *pad = coins.draw().unwrap().0;
            }
        }
        let rows: usize = draft.w_shape.main_rows as usize;
        let tail_len: usize = draft.w_shape.code.r as usize;
        let tails: Vec<Vec<Fp3>> = (0..1)
            .map(|_| {
                (0..rows * tail_len)
                    .map(|_| coins.draw().unwrap().0)
                    .collect()
            })
            .collect();
        let salts: Vec<Vec<Vec<u8>>> = (0..1)
            .map(|_| many_salts(&mut coins, draft.w_shape.code.n(), draft.w_shape.salt_bytes))
            .collect();
        let w_randomness = ConsumerWCommitRandomness { pads, tails, salts };
        w_randomness.check(&draft.w_shape).unwrap();
        let table_source = group_table_source(&matrix_words);
        let (member, target) = derive_consumer_first_member(
            &draft,
            &activation,
            &raw,
            &bits,
            &outputs,
            &raw_first,
            &raw_masks,
            &PlaceRows::new(&matrix),
            &psc_masks,
        )
        .unwrap();
        let weights = pending_row_weights(&member, PLACES as u64).unwrap();
        let mut source = group_table_source(&matrix_words);
        let w_values = derive_w_values(
            &draft.table_shape,
            PLACES as u64,
            0,
            COLUMNS as u64,
            &weights,
            &mut source,
        )
        .unwrap();
        let w_commit = commit_request_w(
            &draft.w_shape,
            70,
            43,
            &w_values,
            &w_randomness.pads,
            &w_randomness.tails,
            &w_randomness.salts,
        )
        .unwrap();
        let w_root = *w_commit.roots.first().unwrap();
        let group = ConsumerGroupPlan {
            shapes: shapes.clone(),
            group_id: 41,
            fresh: 51,
            raw_oracle_base: 20,
            bits_oracle_base: 30,
            mask_oracle: 60,
            w_oracle: 70,
            raw_bat_base: 0,
            bits_bat_base: HEADS as u64,
            mask_bat: 2 * HEADS as u64,
            w_bat: 2 * HEADS as u64 + 1,
            mask_root: mask_commit.root,
            raw_roots: raw_group.roots.clone(),
            bits_roots: bits_group.roots.clone(),
            w_roots: vec![w_root],
        };
        group.check().unwrap();
        let plan = ConsumerFirstPlan::new(draft, member, target)
            .unwrap()
            .with_group(group)
            .unwrap();
        GroupFixture {
            plan,
            activation,
            raw,
            bits,
            outputs,
            raw_first,
            raw_masks,
            psc_masks,
            matrix,
            table_source,
            w_randomness,
            raw_group,
            raw_tails,
            bits_group,
            bits_tails,
            mask: mask_commit,
            mask_rows,
            mask_tails,
            run_seed: Digest([0x33; 32]),
        }
    }

    fn prove_group(
        fixture: &mut GroupFixture,
    ) -> Result<(ConsumerFirstMessages, ConsumerFirstObligations)> {
        let witness = ConsumerFirstGroupWitness {
            raw: &fixture.raw_group,
            raw_tails: &fixture.raw_tails,
            bits: &fixture.bits_group,
            bits_tails: &fixture.bits_tails,
            mask: &fixture.mask,
            mask_rows: fixture.mask_rows.clone(),
            mask_tails: fixture.mask_tails.clone(),
            w_tails: &fixture.w_randomness.tails,
            run_seed: fixture.run_seed,
        };
        prove_consumer_first(
            &fixture.plan,
            &fixture.activation,
            &fixture.raw,
            &fixture.bits,
            &fixture.outputs,
            &fixture.raw_first,
            &fixture.raw_masks,
            &fixture.psc_masks,
            &PlaceRows::new(&fixture.matrix),
            &mut fixture.table_source,
            &fixture.w_randomness,
            Some(&witness),
        )
    }

    #[test]
    #[ignore = "release group run"]
    fn the_group_seal_is_the_first_stage_seal() {
        let mut fixture = group_fixture();
        let (messages, proven) = prove_group(&mut fixture).unwrap();
        let group = messages.group.as_ref().expect("a group proof");
        assert_eq!(group.group_seal, messages.first_seal);
        assert_ne!(messages.first_seal, messages.stage_seal);
        let verified = verify_consumer_first(
            &fixture.plan,
            &fixture.activation,
            &fixture.outputs,
            &messages,
        )
        .unwrap();
        assert_eq!(proven, verified);
        let encoded = messages.encode(&fixture.plan).unwrap();
        let decoded = ConsumerFirstMessages::decode(&fixture.plan, &encoded).unwrap();
        assert_eq!(decoded, messages);
        verify_consumer_first(
            &fixture.plan,
            &fixture.activation,
            &fixture.outputs,
            &decoded,
        )
        .unwrap();
    }

    /// Replays the verifier's pre-group prefix on the honest messages and
    /// returns the chain, the PSC outcome, the claim and the raw rows.
    fn replay_pre_group(
        plan: &ConsumerFirstPlan,
        fixture: &GroupFixture,
        messages: &ConsumerFirstMessages,
    ) -> Result<(
        zkie_engine::pcs::request::chain::Chain,
        crate::proof::consumer_psc::ConsumerPscOutcome,
        crate::proof::consumer_request::ConsumerRequestClaim,
        Vec<crate::proof::consumer_raw::ConsumerRawObligation>,
    )> {
        let mut chain = plan.begin_chain()?;
        let claim = consumer_request_claim(&mut chain, &plan.draft.request)?;
        let raw_rows = verify_consumer_raw(
            &mut chain,
            &plan.draft.raw,
            &messages.raw_first,
            &fixture.outputs,
            &messages.raw_second,
        )?;
        let obligation = consumer_combined_obligation(&plan.draft.request)?;
        let outcome = verify_consumer_psc_fixed(
            &mut chain,
            &obligation,
            &fixture.activation,
            &messages.psc,
            &claim.position,
            &claim.combined,
            plan.target,
        )?;
        check_member_claim(&plan.member, &outcome.pending)?;
        absorb_member(&mut chain, plan)?;
        absorb_w_root(&mut chain, plan.draft.w_oracle, &messages.w_root)?;
        absorb_eval_target(&mut chain, plan.member.target)?;
        Ok((chain, outcome, claim, raw_rows))
    }

    #[test]
    #[ignore = "release group run"]
    fn a_pending_that_does_not_match_the_committed_w_is_refused() {
        // R1: the Eval row is the checked pending row. If the verifier's row
        // target is not the pending target the committed W satisfies, the
        // group run refuses it; nothing else in the carrier is changed.
        let mut fixture = group_fixture();
        let (messages, _) = prove_group(&mut fixture).unwrap();
        let group = fixture.plan.group.as_ref().unwrap();
        let (chain, outcome, claim, raw_rows) =
            replay_pre_group(&fixture.plan, &fixture, &messages).unwrap();
        let member_masks: Vec<(u64, u64, Fp3)> = fixture
            .plan
            .member
            .masks
            .iter()
            .map(|term| (term.oracle, term.slot, term.coeff))
            .collect();
        let honest = request_eval_row(
            TableLayout::of(&fixture.plan.draft.w_shape),
            fixture.plan.draft.columns,
            fixture.plan.draft.w_oracle,
            &fixture.plan.member.psi,
            &member_masks,
            fixture.plan.member.target,
        )
        .unwrap();
        let mut changed = honest.clone();
        changed.target = changed.target.add(&fp(1));
        let raw_claim = raw_claim_row(&fixture.plan, &claim).unwrap();
        let rows = group_rows(&fixture.plan, &changed, &outcome.mask_row, &raw_claim, &raw_rows)
            .unwrap();
        let bat_rows = remap_group_rows(&fixture.plan, group, &rows).unwrap();
        assert!(verify_consumer_group(
            chain,
            group,
            bat_rows,
            messages.group.as_ref().unwrap()
        )
        .is_err());
    }

    #[test]
    #[ignore = "release group run"]
    fn another_weighted_raw_target_is_refused() {
        // R2: the raw claim row binds the committed raw messages to the PSC
        // start target under the request's coefficients. A row carrying
        // another weighted target is refused by the same group run.
        let mut fixture = group_fixture();
        let (messages, _) = prove_group(&mut fixture).unwrap();
        let group = fixture.plan.group.as_ref().unwrap();
        let (chain, outcome, claim, raw_rows) =
            replay_pre_group(&fixture.plan, &fixture, &messages).unwrap();
        let member_masks: Vec<(u64, u64, Fp3)> = fixture
            .plan
            .member
            .masks
            .iter()
            .map(|term| (term.oracle, term.slot, term.coeff))
            .collect();
        let eval_row = request_eval_row(
            TableLayout::of(&fixture.plan.draft.w_shape),
            fixture.plan.draft.columns,
            fixture.plan.draft.w_oracle,
            &fixture.plan.member.psi,
            &member_masks,
            fixture.plan.member.target,
        )
        .unwrap();
        let mut raw_claim = raw_claim_row(&fixture.plan, &claim).unwrap();
        raw_claim.target = raw_claim.target.add(&fp(1));
        let rows = group_rows(&fixture.plan, &eval_row, &outcome.mask_row, &raw_claim, &raw_rows)
            .unwrap();
        let bat_rows = remap_group_rows(&fixture.plan, group, &rows).unwrap();
        assert!(verify_consumer_group(
            chain,
            group,
            bat_rows,
            messages.group.as_ref().unwrap()
        )
        .is_err());
    }

    #[test]
    #[ignore = "release group run"]
    fn a_changed_group_answer_or_seal_is_refused() {
        let mut fixture = group_fixture();
        let (messages, _) = prove_group(&mut fixture).unwrap();
        // A changed sample answer no longer reproduces the group.
        let mut changed = messages.clone();
        let group = changed.group.as_mut().unwrap();
        group.answers[0] = group.answers[0].add(&fp(1));
        assert!(verify_consumer_first(
            &fixture.plan,
            &fixture.activation,
            &fixture.outputs,
            &changed
        )
        .is_err());
        // A changed group seal is refused even before the proximity check.
        let mut changed = messages.clone();
        let group = changed.group.as_mut().unwrap();
        group.group_seal = Digest([0x12; 32]);
        assert!(verify_consumer_first(
            &fixture.plan,
            &fixture.activation,
            &fixture.outputs,
            &changed
        )
        .is_err());
        // A proof without the group is not the stage's real first seal.
        let mut changed = messages.clone();
        changed.group = None;
        assert!(verify_consumer_first(
            &fixture.plan,
            &fixture.activation,
            &fixture.outputs,
            &changed
        )
        .is_err());
        // The plan member still binds: another first seal is refused.
        let mut changed = messages;
        changed.first_seal = Digest([0x13; 32]);
        assert!(verify_consumer_first(
            &fixture.plan,
            &fixture.activation,
            &fixture.outputs,
            &changed
        )
        .is_err());
    }

    // ------------------------------------------------------------------
    // The full consumer composition: first stage, shared stage, own group
    // ------------------------------------------------------------------

    use std::cell::RefCell;

    use zkie_engine::pcs::request::field::E_DEGREE;
    use zkie_engine::pcs::request::foldeq::{reader_table_identity, OmegaShapePlan};
    use zkie_engine::pcs::request::gu::{GuOracleShape, GuShape};
    use zkie_engine::pcs::request::read::{DurableColumnBudget, HidingState};
    use zkie_engine::pcs::request::runtime_tree::{
        MaskTreePlan, RuntimePartRows, RuntimeTree, RuntimeTreePart, RuntimeTreePlan,
    };
    use zkie_engine::pcs::request::table::{
        commit_table_streaming, IndexReader, IndexShape, StreamingTable, VecTableSource,
    };

    use crate::proof::consumer_proof::{
        encode_own_proof, verify_consumer_chain, ConsumerProofFile,
    };
    use crate::proof::consumer_second::{
        consumer_second_chain, consumer_total_acceptance, prove_consumer_own, ConsumerChainBinding,
        ConsumerOwnPlan, ConsumerOwnWitness,
    };
    use crate::proof::consumer_shared::{
        consumer_binding_prove, consumer_shared_charge, prove_consumer_shared,
        verify_consumer_shared, ConsumerSharedGeometry, ConsumerSharedIds, ConsumerSharedStage,
    };
    use crate::proof::lin_binding::{LinBindingPath, LinBindingRoot, LinBindingShape};

    fn composition_binding(
        geometry: &ConsumerSharedGeometry,
        first_seal: &Digest,
    ) -> (LinBindingPath, LinBindingRoot) {
        let salts: Vec<u8> = (0..40).map(|index| (index as u8) ^ 0x6D).collect();
        consumer_binding_prove(
            first_seal,
            geometry.binding_shape,
            &geometry.binding_configuration,
            salts,
        )
        .unwrap()
    }
    use crate::proof::hidden_send_session::HiddenTableProver;

    const COMPOSITION_CONFIGURATION: Digest = Digest([0x55; 32]);

    struct CompositionKeeper {
        table: StreamingTable,
        source: RefCell<VecTableSource>,
        reader: IndexReader,
        hiding: HidingState,
        budget: Option<DurableColumnBudget>,
    }

    impl TableRowSource for CompositionKeeper {
        fn rows(&self) -> u64 {
            self.source.borrow().rows()
        }

        fn row(&mut self, index: u64, message: &mut [Fp], tail: &mut [Fp]) -> Result<()> {
            self.source.borrow_mut().row(index, message, tail)
        }
    }

    impl HiddenTableProver for CompositionKeeper {
        fn reader(&self) -> &IndexReader {
            &self.reader
        }

        fn open_columns(
            &mut self,
            positions: &[u64],
        ) -> Result<Vec<zkie_engine::pcs::request::table::ColumnOpening>> {
            let mut source = self.source.borrow_mut();
            self.table.open_columns(&mut *source, positions)
        }

        fn hiding_state(&mut self) -> Option<&mut HidingState> {
            Some(&mut self.hiding)
        }

        fn durable_column_budget(&mut self) -> Option<&mut DurableColumnBudget> {
            self.budget.as_mut()
        }
    }

    struct GroupSource<'a>(&'a GroupFixture);

    impl TableRowSource for GroupSource<'_> {
        fn rows(&self) -> u64 {
            self.0.table_source.messages.len() as u64
        }

        fn row(&mut self, index: u64, message: &mut [Fp], tail: &mut [Fp]) -> Result<()> {
            let source = self
                .0
                .table_source
                .messages
                .get(index as usize)
                .ok_or(Error::ShapeMismatch)?;
            message.copy_from_slice(source);
            let source_tail = self
                .0
                .table_source
                .tails
                .get(index as usize)
                .ok_or(Error::ShapeMismatch)?;
            tail.copy_from_slice(source_tail);
            Ok(())
        }
    }

    fn matrix_word(place: usize, record: usize) -> i64 {
        ((record as i64 * 7 + place as i64 * 3 + 1) % 17) + 1
    }

    /// The shared table is the same M in its stored-column view: row `k` is
    /// the record and the entries are the places, exactly the view the first
    /// stage's W derivation reads.
    fn composition_table() -> (IndexReader, CompositionKeeper) {
        let code = build_code_plan(6, 16, 1).unwrap();
        let shape = IndexShape {
            code,
            v: 4,
            answers: 1,
            salt_bytes: 40,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        };
        let mut message_rows = vec![vec![Fp::ZERO; 16]; 16];
        for record in 0..16 {
            for place in 0..16 {
                message_rows[record][place] =
                    fp_from_signed_word(matrix_word(place, record) as u64);
            }
        }
        let tails = vec![vec![Fp::ZERO; 1]; 16];
        let mut source = VecTableSource::new(message_rows, tails).unwrap();
        let salts: Vec<Vec<u8>> = (0..code.n())
            .map(|position| vec![(position as u8) ^ 0x5A; 40])
            .collect();
        let table = commit_table_streaming(
            &mut source,
            shape,
            COMPOSITION_CONFIGURATION,
            &salts,
            64,
            u64::MAX,
        )
        .unwrap();
        let reader = table.reader.clone();
        let keeper = CompositionKeeper {
            table,
            source: RefCell::new(source),
            hiding: HidingState::new(reader.root, 1, reader.shape.code.r as u64),
            reader: reader.clone(),
            budget: None,
        };
        (reader, keeper)
    }

    fn composition_geometry(
        plan: &ConsumerFirstPlan,
        reader: &IndexReader,
    ) -> ConsumerSharedGeometry {
        ConsumerSharedGeometry::derive(
            plan,
            reader_table_identity(reader).unwrap(),
            COMPOSITION_CONFIGURATION,
            reader.shape,
            LinBindingShape {
                log_leaves: 0,
                salt_bytes: 40,
            },
            Digest([0x92; 32]),
            Digest([0x98; 32]),
            b"consumer composition".to_vec(),
            reader.name(),
            ConsumerSharedIds {
                group: 9,
                gu_oracle: 960,
                omega_oracle: 961,
                gu_tree: 971,
                omega_tree: 972,
                mask_tree: 970,
                fresh: 980,
                base_positions: 1,
                column_work_bits: 0,
            },
        )
        .unwrap()
    }

    fn composition_shared_plan() -> ConsumerSharedPlan {
        ConsumerSharedPlan {
            m_once: 8,
            m_later: 8,
            one_time_columns: 1,
            later_columns: 1,
            declared_batches: 1,
            table_tail: 4,
        }
    }

    #[test]
    #[ignore = "release composition run"]
    fn the_consumer_composition_accepts_on_the_same_matrix() {
        let (reader, mut keeper) = composition_table();
        let table_identity = reader_table_identity(&reader).unwrap();
        let mut fixture = group_fixture_with(table_with(table_identity, COMPOSITION_CONFIGURATION));
        let (messages, _obligations) = prove_group(&mut fixture).unwrap();
        let plan = fixture.plan.clone();
        let geometry = composition_geometry(&plan, &reader);
        // The per-request binding root comes from this request's own first
        // seal; the proof carries the path, the verifier rebuilds the root.
        let (binding_path, binding_root) = composition_binding(&geometry, &messages.first_seal);
        // The shared stage over the same table.
        let shared_plan = composition_shared_plan();
        let stage = ConsumerSharedStage::new(&shared_plan).unwrap();
        let charge = consumer_shared_charge(&stage, &mut keeper).unwrap();
        let shared_proof = prove_consumer_shared(
            &geometry,
            &stage,
            &reader,
            &mut keeper,
            Digest([0xE2; 32]),
            &charge,
            &binding_root,
        )
        .unwrap();
        let check_stage = ConsumerSharedStage::new(&shared_plan).unwrap();
        let shared_verified = verify_consumer_shared(
            &geometry,
            &check_stage,
            &reader,
            &shared_proof.messages,
            &binding_root,
        )
        .unwrap();
        assert_eq!(shared_verified.binding_root, binding_root);
        // The W witness is the first stage's committed W rebuilt from the same
        // stored-column matrix view.
        let weights = pending_row_weights(&plan.member, plan.draft.matrix_rows).unwrap();
        let mut source = GroupSource(&fixture);
        let w_values = derive_w_values(
            &plan.draft.table_shape,
            plan.draft.matrix_rows,
            plan.draft.k_lo,
            plan.draft.columns,
            &weights,
            &mut source,
        )
        .unwrap();
        let w_commit = commit_request_w(
            &plan.draft.w_shape,
            plan.draft.w_oracle,
            plan.draft.w_group,
            &w_values,
            &fixture.w_randomness.pads,
            &fixture.w_randomness.tails,
            &fixture.w_randomness.salts,
        )
        .unwrap();
        assert_eq!(*w_commit.roots.first().unwrap(), messages.w_root);
        let tail_len = usize::try_from(plan.draft.w_shape.code.r).unwrap();
        let rows = usize::try_from(plan.draft.w_shape.main_rows).unwrap();
        let flat = &fixture.w_randomness.tails[0];
        let w_tails: Vec<Vec<Fp3>> = (0..rows)
            .map(|row| flat[row * tail_len..(row + 1) * tail_len].to_vec())
            .collect();
        // The own group and the total acceptance.
        let own_mask_tree = MaskTreePlan {
            tree_number: 986,
            leaf_class: 4,
            log_n: geometry.mask_code.log_n,
            mask_rows: 1,
            salt_bytes: 40,
            run: Digest([0x99; 32]),
            configuration: Digest([0x9A; 32]),
        };
        let own_plan = ConsumerOwnPlan::from_first_shared(
            &plan,
            &messages,
            &geometry,
            &shared_verified,
            own_mask_tree,
            985,
            1,
        )
        .unwrap();
        let binding = ConsumerChainBinding::from_first_shared(&messages, &shared_verified).unwrap();
        let chain = consumer_second_chain(&geometry, &binding).unwrap();
        let witness = ConsumerOwnWitness {
            w: &w_commit,
            w_tails: &w_tails,
            shared: &shared_proof.witness,
        };
        let own_proof = prove_consumer_own(
            chain,
            &own_plan,
            &shared_verified,
            &witness,
            Digest([0xE3; 32]),
        )
        .unwrap();
        let accept_stage = ConsumerSharedStage::new(&shared_plan).unwrap();
        let total = consumer_total_acceptance(
            &plan,
            &fixture.activation,
            &fixture.outputs,
            &messages,
            &geometry,
            &accept_stage,
            &reader,
            &shared_proof.messages,
            &binding_path,
            &own_plan,
            &own_proof,
        )
        .unwrap();
        assert_eq!(total.first_seal, messages.first_seal);
        assert_eq!(total.shared_seal, shared_verified.seal);
        assert_eq!(total.second_seal, own_proof.seal);
        assert!(!total.obligations.group_rows.is_empty());
        assert!(!plan.member.psi.is_empty());
        // The auxiliary file assembles from the accepted results and decodes
        // back to the same stage messages.
        let file = ConsumerProofFile::assemble(
            &plan,
            &messages,
            &shared_proof.messages,
            &own_proof,
            &total,
            binding_path.clone(),
        )
        .unwrap();
        let bytes = file.encode().unwrap();
        let decoded = ConsumerProofFile::decode(&bytes).unwrap();
        assert_eq!(decoded, file);
        assert_eq!(decoded.own_proof().unwrap(), own_proof);
        assert_eq!(decoded.totals.first_seal, total.first_seal);
        // The verifier-side global order runs from the file alone: decode the
        // three sections, compare the totals, then verify every stage.
        let accepted = verify_consumer_chain(
            &plan,
            &fixture.activation,
            &fixture.outputs,
            &geometry,
            &ConsumerSharedStage::new(&shared_plan).unwrap(),
            &reader,
            &own_plan,
            &file,
        )
        .unwrap();
        assert_eq!(accepted.first_seal, total.first_seal);
        assert_eq!(accepted.second_seal, total.second_seal);
        // A changed total seal is refused before any stage runs.
        let mut bad = file.clone();
        bad.totals.second_seal = Digest([0x77; 32]);
        let bad = ConsumerProofFile::decode(&bad.encode().unwrap()).unwrap();
        assert!(verify_consumer_chain(
            &plan,
            &fixture.activation,
            &fixture.outputs,
            &geometry,
            &ConsumerSharedStage::new(&shared_plan).unwrap(),
            &reader,
            &own_plan,
            &bad,
        )
        .is_err());
        // A changed binding salt or a changed path place no longer rebuilds
        // the root, so the file is refused before the stages.
        let mut bad = file.clone();
        bad.binding_path.salt[0] ^= 1;
        let bad = ConsumerProofFile::decode(&bad.encode().unwrap()).unwrap();
        assert!(verify_consumer_chain(
            &plan,
            &fixture.activation,
            &fixture.outputs,
            &geometry,
            &ConsumerSharedStage::new(&shared_plan).unwrap(),
            &reader,
            &own_plan,
            &bad,
        )
        .is_err());
        // A changed own seal with matching totals is refused by the own stage.
        let mut own_bad = own_proof.clone();
        own_bad.seal = Digest([0x78; 32]);
        let mut bad = file.clone();
        bad.own = encode_own_proof(&own_bad).unwrap();
        bad.totals.second_seal = own_bad.seal;
        let bad = ConsumerProofFile::decode(&bad.encode().unwrap()).unwrap();
        assert!(verify_consumer_chain(
            &plan,
            &fixture.activation,
            &fixture.outputs,
            &geometry,
            &ConsumerSharedStage::new(&shared_plan).unwrap(),
            &reader,
            &own_plan,
            &bad,
        )
        .is_err());
    }

    #[test]
    #[ignore = "release composition run"]
    fn the_composition_refuses_a_witness_on_the_wrong_axis() {
        let (reader, mut keeper) = composition_table();
        let table_identity = reader_table_identity(&reader).unwrap();
        let mut fixture = group_fixture_with(table_with(table_identity, COMPOSITION_CONFIGURATION));
        let (messages, _) = prove_group(&mut fixture).unwrap();
        let plan = fixture.plan.clone();
        let geometry = composition_geometry(&plan, &reader);
        let (_binding_path, binding_root) = composition_binding(&geometry, &messages.first_seal);
        let shared_plan = composition_shared_plan();
        let stage = ConsumerSharedStage::new(&shared_plan).unwrap();
        let charge = consumer_shared_charge(&stage, &mut keeper).unwrap();
        let shared_proof = prove_consumer_shared(
            &geometry,
            &stage,
            &reader,
            &mut keeper,
            Digest([0xE2; 32]),
            &charge,
            &binding_root,
        )
        .unwrap();
        let check_stage = ConsumerSharedStage::new(&shared_plan).unwrap();
        let shared_verified = verify_consumer_shared(
            &geometry,
            &check_stage,
            &reader,
            &shared_proof.messages,
            &binding_root,
        )
        .unwrap();
        // The wrong-axis witness: places as rows, records as entries.
        let records = 16usize;
        let places = 16usize;
        let mut rows = vec![vec![Fp::ZERO; ELL]; places];
        for place in 0..places {
            for record in 0..records {
                rows[place][record] = fp_from_signed_word(matrix_word(place, record) as u64);
            }
        }
        let mut wrong = TableSource {
            messages: rows,
            tails: (0..places)
                .map(|_| (0..TABLE_R).map(|_| Fp::ZERO).collect())
                .collect(),
        };
        let weights = pending_row_weights(&plan.member, plan.draft.matrix_rows).unwrap();
        let wrong_values = derive_w_values(
            &plan.draft.table_shape,
            plan.draft.matrix_rows,
            plan.draft.k_lo,
            plan.draft.columns,
            &weights,
            &mut wrong,
        )
        .unwrap();
        let wrong_commit = commit_request_w(
            &plan.draft.w_shape,
            plan.draft.w_oracle,
            plan.draft.w_group,
            &wrong_values,
            &fixture.w_randomness.pads,
            &fixture.w_randomness.tails,
            &fixture.w_randomness.salts,
        )
        .unwrap();
        assert_ne!(*wrong_commit.roots.first().unwrap(), messages.w_root);
        let tail_len = usize::try_from(plan.draft.w_shape.code.r).unwrap();
        let flat = &fixture.w_randomness.tails[0];
        let w_tails: Vec<Vec<Fp3>> = vec![flat[..tail_len].to_vec()];
        let witness = ConsumerOwnWitness {
            w: &wrong_commit,
            w_tails: &w_tails,
            shared: &shared_proof.witness,
        };
        let own_mask_tree = MaskTreePlan {
            tree_number: 986,
            leaf_class: 4,
            log_n: geometry.mask_code.log_n,
            mask_rows: 1,
            salt_bytes: 40,
            run: Digest([0x99; 32]),
            configuration: Digest([0x9A; 32]),
        };
        let own_plan = ConsumerOwnPlan::from_first_shared(
            &plan,
            &messages,
            &geometry,
            &shared_verified,
            own_mask_tree,
            985,
            1,
        )
        .unwrap();
        let binding = ConsumerChainBinding::from_first_shared(&messages, &shared_verified).unwrap();
        let chain = consumer_second_chain(&geometry, &binding).unwrap();
        assert!(prove_consumer_own(
            chain,
            &own_plan,
            &shared_verified,
            &witness,
            Digest([0xE3; 32]),
        )
        .is_err());
    }
}
