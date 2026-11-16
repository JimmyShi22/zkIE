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
use crate::proof::projection_claim::raw_target;
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
    pub eval_units: Vec<(u64, u64, Fp3)>,
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
        writer.put_u64(self.eval_units.len() as u64);
        for (oracle, index, coefficient) in &self.eval_units {
            writer.put_u64(*oracle);
            writer.put_u64(*index);
            writer.put_fp3(coefficient);
        }
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
        };
        plan.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))?;
        Ok(plan)
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

fn group_rows(
    plan: &ConsumerFirstPlan,
    eval_row: &BatRow,
    mask_row: &ConsumerMaskRow,
    raw_rows: &[ConsumerRawObligation],
) -> Result<Vec<BatRow>> {
    let mut rows = Vec::with_capacity(2 + raw_rows.len() * 2);
    rows.push(eval_row.clone());
    rows.push(consumer_mask_row_bat(plan, mask_row)?);
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

/// The public first-stage messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerFirstMessages {
    pub raw_first: ConsumerRawFirst,
    pub raw_second: ConsumerRawSecond,
    pub psc: ConsumerPscProof,
    pub w_root: Digest,
    pub eval_target: Fp3,
    pub psi: Vec<Fp3>,
    pub stage_seal: Digest,
}

impl ConsumerFirstMessages {
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
        reader.finish()?;
        let messages = Self {
            raw_first,
            raw_second,
            psc,
            w_root,
            eval_target,
            psi,
            stage_seal,
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
        if self.psi != plan.member.psi {
            return Err(Error::StatementMismatch);
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
    matrix_places: &[Vec<Fp3>],
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
    matrix_places: &[Vec<Fp3>],
    table_source: &mut dyn TableRowSource,
    w_randomness: &ConsumerWCommitRandomness,
    eval_mask: &dyn Fn(u64, u64) -> Result<Fp3>,
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
    let mut eval_target = multilinear_eval(&w_values, &plan.member.psi)?;
    for (oracle, index, coefficient) in &plan.draft.eval_units {
        eval_target = eval_target.add(&coefficient.mul(&eval_mask(*oracle, *index)?));
    }
    absorb_eval_target(&mut chain, eval_target)?;
    let eval_row = request_eval_row(
        TableLayout::of(&plan.draft.w_shape),
        plan.draft.columns,
        plan.draft.w_oracle,
        &plan.member.psi,
        &plan.draft.eval_units,
        eval_target,
    )?;
    let stage_seal = *chain.value();
    let messages = ConsumerFirstMessages {
        raw_first: raw_first.clone(),
        raw_second,
        psc,
        w_root,
        eval_target,
        psi: plan.member.psi.clone(),
        stage_seal,
    };
    let rows = group_rows(plan, &eval_row, &outcome.mask_row, &raw_rows)?;
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
    absorb_eval_target(&mut chain, messages.eval_target)?;
    let eval_row = request_eval_row(
        TableLayout::of(&plan.draft.w_shape),
        plan.draft.columns,
        plan.draft.w_oracle,
        &plan.member.psi,
        &plan.draft.eval_units,
        messages.eval_target,
    )?;
    if *chain.value() != messages.stage_seal {
        return Err(Error::InvalidProof);
    }
    let rows = group_rows(plan, &eval_row, &outcome.mask_row, &raw_rows)?;
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

    use crate::proof::column_session::fp_from_signed_word;
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

    fn table() -> ConsumerTable {
        ConsumerTable {
            path: "weights/L0_k_w_i32.bin".to_string(),
            oracle: 30,
            root: Digest([0x11; 32]),
            identity: Digest([0x12; 32]),
            configuration: Digest([0x13; 32]),
        }
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

    fn obligation(head: u32, source_step: u32) -> ConsumerObligation {
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
            table: table(),
            mask: mask_source(),
        }
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
            eval_units: vec![(60, 5, fp(2))],
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
            messages: matrix_words
                .iter()
                .map(|row| {
                    let mut message = vec![Fp::ZERO; ELL];
                    for (place, value) in row.iter().enumerate() {
                        message[place] = fp_from_signed_word(*value as u64);
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
            &matrix,
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

    fn eval_mask(oracle: u64, index: u64) -> Result<Fp3> {
        if oracle == 60 && index == 5 {
            Ok(fp(3))
        } else {
            Err(Error::ShapeMismatch)
        }
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
            &fixture.matrix,
            &mut fixture.table_source,
            &fixture.w_randomness,
            &eval_mask,
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
        assert_eq!(proven.group_rows.len(), 2 + proven.raw_rows.len() * 2);
        let tables = witness_tables(&fixture);
        for row in &proven.group_rows {
            assert_eq!(evaluate_bat_row(row, &tables), row.target);
        }
        assert_eq!(messages.psi, fixture.plan.member.psi);
        assert_ne!(messages.stage_seal, Digest::ZERO);
        assert_ne!(messages.w_root, Digest::ZERO);
        assert_eq!(proven.eval_row.target, messages.eval_target);
        // The Eval target is the W evaluation at the member's point plus the
        // plan's mask units.
        let weights =
            pending_row_weights(&fixture.plan.member, fixture.plan.draft.matrix_rows).unwrap();
        let mut source = FixtureSource(&fixture);
        let w_values = derive_w_values(
            &fixture.plan.draft.table_shape,
            fixture.plan.draft.matrix_rows,
            fixture.plan.draft.k_lo,
            fixture.plan.draft.columns,
            &weights,
            &mut source,
        )
        .unwrap();
        let expected = multilinear_eval(&w_values, &fixture.plan.member.psi)
            .unwrap()
            .add(&fp(2).mul(&fp(3)));
        assert_eq!(messages.eval_target, expected);
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
            &fixture.matrix,
            &mut fixture.table_source,
            &fixture.w_randomness,
            &eval_mask,
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
            &fixture.matrix,
            &mut fixture.table_source,
            &fixture.w_randomness,
            &eval_mask,
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
}
