//! The projection consumer request: one request, one chain, all head slices.
//!
//! A real projection request consumes one table through every head slice of
//! the layer. The request binding walks the head claims of one request in one
//! chain: the request statement fixes the request, the table, the geometry,
//! the mask source, the ordered head sources and the actual raw accumulator
//! rows before any challenge; the position coefficients and the per-head
//! column coefficients follow; then the batch challenge combines the heads
//! into one product-sumcheck claim over the whole consumed region.
//!
//! With `J` registered heads the request performs the batch step (`J > 1`) or
//! the direct step (`J = 1`):
//!
//! * every head has its own column coefficients `b_h` and its own raw target
//!   `T_h = Σ_p Σ_j a[p] b_h[j] Y_h[p, j]`, derived from the head's raw rows;
//! * the batch challenge `theta` combines the targets and the column
//!   coefficients: `T = Σ_h theta^h T_h` and
//!   `b = concat_h(theta^h b_h)`;
//! * one masked product sumcheck over the whole consumed slice proves
//!   `Σ_k U[k] V[k] = T` with the public activation reduction `U` and the
//!   private column combination `V[k] = Σ_n b[n] M[n, k]`;
//! * the verifier recomputes `U` at its own point, checks the first end value,
//!   and turns the second end value into one pending row over the same matrix,
//!   with the row weights covering exactly the consumed region.
//!
//! The raw rows are the projection's integer accumulator, verified as public
//! request inputs: every entry must satisfy the production rounding relation
//! `out = rq(Y, shift)` of the head's projection step, and the relation is
//! checked by both sides before any challenge. The verifier therefore derives
//! the raw targets itself; the prover never supplies them. The raw relation
//! quantity of every head is this exact accumulator, never its requantized
//! output. What the verifier learns is exactly the accumulator the op-level
//! relation already carries; the table itself never leaves the prover.

use zkie_core::artifact::digest_bytes;
use zkie_core::common::codec::DecodeLimits;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::chain::Chain;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};

use crate::proof::column_session::fp_from_signed_word;
use crate::proof::committed_psc::{
    get_psc_transcript, put_psc_transcript, PositionPscChallenges, PscTranscriptTags,
};
use crate::proof::consumer_psc::{
    ConsumerMaskRow, ConsumerMaskSource, ConsumerObligation, ConsumerPscMasks, ConsumerPscShape,
    ConsumerTable, CONSUMER_PSC_VERSION,
};
use crate::proof::hidden_member::{stretch_weights, MaskTerm, PendingRow};
use crate::proof::projection_claim::{activation_shown, consumer_reduction, raw_target};
use crate::proof::psc::{
    check_product_claims, multilinear_eval, prove_dot_sumcheck, verify_psc_transcript, DotClaim,
    DotMaskEntries, DotOutput, DotProof, SHOWN_MAPS,
};
use crate::proof::relations::rq;

/// Version of the request proof message and of its transcript messages.
pub const CONSUMER_REQUEST_VERSION: u64 = 1;
/// Transcript tag of the request statement.
const CONSUMER_REQUEST_TAG: u64 = 86;
/// Transcript tags of the request's product sumcheck. The claim word is
/// reserved: the plain product sumcheck carries no claim masks.
pub const CONSUMER_REQUEST_PSC_TAGS: PscTranscriptTags = PscTranscriptTags {
    version: CONSUMER_REQUEST_VERSION,
    hiding_sum: 88,
    round: 89,
    end: 90,
    claim: 91,
};

fn push_digest(message: &mut zkie_engine::pcs::request::chain::ChainMessage, digest: &Digest) {
    for word in digest.0.chunks_exact(8) {
        message.push_u64(u64::from_le_bytes(
            word.try_into().expect("eight-byte word"),
        ));
    }
}

/// The one-request consumer plan: the registered head obligations of a single
/// request, their common table and the single product-sumcheck instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerRequestPlan {
    pub request: u64,
    pub table: ConsumerTable,
    pub shift: i64,
    pub shape: ConsumerPscShape,
    pub mask: ConsumerMaskSource,
    pub head_width: u64,
    pub obligations: Vec<ConsumerObligation>,
}

impl ConsumerRequestPlan {
    /// Registers the request from its ordered head obligations. Every head is
    /// one source step of the same request, the same table and the same
    /// activation step; the heads are exactly `0..J`; their slices are the
    /// adjacent head slices of the consumed region and nothing else.
    pub fn new(
        request: u64,
        table: ConsumerTable,
        mask: ConsumerMaskSource,
        obligations: Vec<ConsumerObligation>,
    ) -> Result<Self> {
        if obligations.is_empty()
            || obligations.len() > u32::MAX as usize
            || obligations.len() > 4096
        {
            return Err(Error::InvalidConfig);
        }
        let heads = obligations.len() as u32;
        let first = obligations[0].clone();
        if first.request != request || first.table != table || first.mask != mask {
            return Err(Error::StatementMismatch);
        }
        let head_width = first.shape.slice_len()?;
        let hidden = u64::from(heads)
            .checked_mul(head_width)
            .ok_or(Error::ResourceLimit)?;
        if first.shape.positions == 0
            || first.shape.places < hidden
            || first.shape.records < 2
            || !first.shape.records.is_power_of_two()
            || !first.shape.places.is_power_of_two()
        {
            return Err(Error::InvalidConfig);
        }
        let activation_step = first.activation_step;
        for (index, obligation) in obligations.iter().enumerate() {
            if obligation.request != request
                || obligation.table != table
                || obligation.mask != mask
                || obligation.shift != first.shift
                || obligation.activation_step != activation_step
                || obligation.shape.positions != first.shape.positions
                || obligation.shape.places != first.shape.places
                || obligation.shape.records != first.shape.records
                || obligation.head as usize != index
                || obligation.shape.slice
                    != (
                        obligation.head as u64 * head_width,
                        (obligation.head as u64 + 1) * head_width,
                    )
            {
                return Err(Error::StatementMismatch);
            }
            obligation.check()?;
        }
        let shape = ConsumerPscShape {
            positions: first.shape.positions,
            places: first.shape.places,
            records: first.shape.records,
            slice: (0, hidden),
        };
        shape.check()?;
        let coverage = crate::proof::projection_claim::ConsumerCoverage::new(
            shape.positions,
            head_width,
            heads,
            hidden,
            shape.places,
        )?;
        for (index, slice) in coverage.slices().iter().enumerate() {
            if slice.head as usize != index
                || slice.columns != (index as u64 * head_width, (index as u64 + 1) * head_width)
            {
                return Err(Error::InvalidConfig);
            }
        }
        let plan = Self {
            request,
            table,
            shift: first.shift,
            shape,
            mask,
            head_width,
            obligations,
        };
        plan.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))?;
        Ok(plan)
    }

    /// The number of registered head claims.
    pub fn heads(&self) -> u32 {
        self.obligations.len() as u32
    }

    /// The consumed visible width: every head slice of the request.
    pub fn consumed(&self) -> u64 {
        self.shape.slice.1 - self.shape.slice.0
    }

    pub fn fault(&self) -> Option<&'static str> {
        if let Some(fault) = self.shape.fault() {
            return Some(fault);
        }
        if self.obligations.is_empty() || self.obligations.len() > u32::MAX as usize {
            return Some("a consumer request has no head claim");
        }
        if self.head_width == 0 || self.head_width != self.obligations[0].shape.slice_len().ok()? {
            return Some("a consumer request has no common head width");
        }
        if self.consumed() != u64::from(self.heads()) * self.head_width {
            return Some("a consumer request does not consume exactly its head slices");
        }
        None
    }

    fn absorb(&self, chain: &mut Chain, raw_digest: &Digest, output_digest: &Digest) -> Result<()> {
        let mut message = chain.message();
        message.push_u64(CONSUMER_REQUEST_VERSION);
        message.push_u64(CONSUMER_REQUEST_TAG);
        message.push_u64(self.request);
        message.push_u64(u64::from(self.heads()));
        message.push_u64(self.head_width);
        message.push_u64(u64::from(self.shape.positions));
        message.push_u64(self.shape.places);
        message.push_u64(self.shape.records);
        message.push_u64(self.shape.slice.0);
        message.push_u64(self.shape.slice.1);
        message.push_u64(self.shift as u64);
        message.push_u64(self.table.oracle);
        push_digest(&mut message, &self.table.root);
        push_digest(&mut message, &self.table.identity);
        push_digest(&mut message, &self.table.configuration);
        message.push_u64(self.mask.oracle);
        message.push_u64(self.mask.hiding_base);
        message.push_u64(self.mask.shown_slot);
        push_digest(&mut message, raw_digest);
        push_digest(&mut message, output_digest);
        for obligation in &self.obligations {
            message.push_u64(u64::from(obligation.source_step));
            message.push_u64(u64::from(obligation.activation_step));
            message.push_u64(u64::from(obligation.head));
        }
        let prefix = message.prefix(zkie_engine::pcs::request::chain::KIND_ALGEBRAIC, 0)?;
        if !chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        Ok(())
    }
}

fn check_rows(value: &[Vec<i64>], heads: usize, positions: u32, width: u64) -> Result<()> {
    if value.len() != heads {
        return Err(Error::ShapeMismatch);
    }
    let count = (positions as usize)
        .checked_mul(usize::try_from(width).map_err(|_| Error::ResourceLimit)?)
        .ok_or(Error::ResourceLimit)?;
    if value.iter().any(|rows| rows.len() != count) {
        return Err(Error::ShapeMismatch);
    }
    Ok(())
}

/// The rounding relation of the request, checked in the clear by both sides:
/// every raw accumulator entry of every head must round to the head step's
/// public output entry under the production rounding rule and shift. This is
/// the real raw relation; a requantized output substituted for the raw
/// accumulator fails it.
pub fn check_consumer_raw_rounding(
    plan: &ConsumerRequestPlan,
    raw: &[Vec<i64>],
    outputs: &[Vec<i64>],
) -> Result<()> {
    plan.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))?;
    let heads = plan.obligations.len();
    check_rows(raw, heads, plan.shape.positions, plan.head_width)?;
    check_rows(outputs, heads, plan.shape.positions, plan.head_width)?;
    let shift = u32::try_from(plan.shift).map_err(|_| Error::InvalidConfig)?;
    for (raw_head, output_head) in raw.iter().zip(outputs) {
        for (value, output) in raw_head.iter().zip(output_head) {
            if rq(i128::from(*value), shift)? != i128::from(*output) {
                return Err(Error::InvalidProof);
            }
        }
    }
    Ok(())
}

fn rows_digest(plan: &ConsumerRequestPlan, value: &[Vec<i64>], tag: u64) -> Result<Digest> {
    check_rows(
        value,
        plan.obligations.len(),
        plan.shape.positions,
        plan.head_width,
    )?;
    let mut writer = CanonicalWriter::new();
    writer.put_u64(CONSUMER_REQUEST_VERSION);
    writer.put_u64(tag);
    writer.put_u64(value.len() as u64);
    for head in value {
        for entry in head {
            writer.put_u64(*entry as u64);
        }
    }
    Ok(digest_bytes(&writer.into_bytes()))
}

/// The digest of the actual raw accumulator rows of the request.
pub fn consumer_raw_digest(plan: &ConsumerRequestPlan, raw: &[Vec<i64>]) -> Result<Digest> {
    rows_digest(plan, raw, 1)
}

/// The digest of the head steps' public output rows.
pub fn consumer_output_digest(plan: &ConsumerRequestPlan, outputs: &[Vec<i64>]) -> Result<Digest> {
    rows_digest(plan, outputs, 2)
}

/// The transcript coefficients of one request: the common position
/// coefficients, one column vector per head and, after the raw relation, the
/// batch challenge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerRequestCoefficients {
    pub position: Vec<Fp3>,
    pub heads: Vec<Vec<Fp3>>,
    pub batch: Option<Fp3>,
}

impl ConsumerRequestCoefficients {
    /// The combined column coefficients over the consumed region: the batch
    /// powers of the per-head vectors, head after head.
    pub fn combined(&self) -> Result<Vec<Fp3>> {
        let width = self.heads.first().ok_or(Error::InvalidConfig)?.len();
        if self.heads.iter().any(|head| head.len() != width) {
            return Err(Error::ShapeMismatch);
        }
        let mut combined = Vec::with_capacity(
            self.heads
                .len()
                .checked_mul(width)
                .ok_or(Error::ResourceLimit)?,
        );
        let mut power = Fp3::ONE;
        for head in &self.heads {
            for value in head {
                combined.push(power.mul(value));
            }
            power = match self.batch {
                Some(theta) => power.mul(&theta),
                None => Fp3::ONE,
            };
        }
        Ok(combined)
    }
}

fn draw_coefficients(
    chain: &mut Chain,
    plan: &ConsumerRequestPlan,
) -> Result<ConsumerRequestCoefficients> {
    let mut position = Vec::with_capacity(plan.shape.positions as usize);
    for _ in 0..plan.shape.positions {
        position.push(chain.draw_e()?);
    }
    let mut heads = Vec::with_capacity(plan.obligations.len());
    for _ in 0..plan.obligations.len() {
        let width = usize::try_from(plan.head_width).map_err(|_| Error::ResourceLimit)?;
        let mut values = Vec::with_capacity(width);
        for _ in 0..width {
            values.push(chain.draw_e()?);
        }
        heads.push(values);
    }
    Ok(ConsumerRequestCoefficients {
        position,
        heads,
        batch: None,
    })
}

/// The embedded raw rows of one head: the signed integer accumulator cells as
/// field elements.
pub fn consumer_raw_field_rows(raw: &[Vec<i64>]) -> Result<Vec<Vec<Fp3>>> {
    let mut heads = Vec::with_capacity(raw.len());
    for head in raw {
        let mut rows = Vec::with_capacity(head.len());
        for value in head {
            rows.push(Fp3::from_base(fp_from_signed_word(*value as u64)));
        }
        heads.push(rows);
    }
    Ok(heads)
}

/// The per-head raw targets from the actual raw rows: `T_h` of every head, in
/// head order. Both sides call this, so the prover never supplies a target.
pub fn consumer_head_targets(
    plan: &ConsumerRequestPlan,
    coefficients: &ConsumerRequestCoefficients,
    raw: &[Vec<i64>],
) -> Result<Vec<Fp3>> {
    check_rows(
        raw,
        plan.obligations.len(),
        plan.shape.positions,
        plan.head_width,
    )?;
    if coefficients.position.len() != plan.shape.positions as usize
        || coefficients.heads.len() != plan.obligations.len()
    {
        return Err(Error::ShapeMismatch);
    }
    let width = usize::try_from(plan.head_width).map_err(|_| Error::ResourceLimit)?;
    let mut targets = Vec::with_capacity(raw.len());
    for (head, rows) in raw.iter().enumerate() {
        if coefficients.heads[head].len() != width {
            return Err(Error::ShapeMismatch);
        }
        let field_row: Vec<Fp3> = rows
            .iter()
            .map(|value| Fp3::from_base(fp_from_signed_word(*value as u64)))
            .collect();
        targets.push(raw_target(
            plan.shape.positions as usize,
            width,
            &field_row,
            &coefficients.position,
            &coefficients.heads[head],
        )?);
    }
    Ok(targets)
}

fn combined_target(
    coefficients: &ConsumerRequestCoefficients,
    targets: &[Fp3],
    heads: usize,
) -> Result<Fp3> {
    if targets.len() != heads || coefficients.heads.len() != heads {
        return Err(Error::ShapeMismatch);
    }
    if heads > 1 && coefficients.batch.is_none() {
        return Err(Error::InvalidConfig);
    }
    let mut total = Fp3::ZERO;
    let mut power = Fp3::ONE;
    for target in targets {
        total = total.add(&power.mul(target));
        power = match coefficients.batch {
            Some(theta) => power.mul(&theta),
            None => Fp3::ONE,
        };
    }
    Ok(total)
}

/// The request proof message: the one product-sumcheck transcript and its
/// masked point values. The targets are derived from the public raw rows, not
/// sent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerRequestProof {
    pub dot: DotProof,
    pub point_values: [Fp3; SHOWN_MAPS],
}

impl ConsumerRequestProof {
    pub fn encode(&self, plan: &ConsumerRequestPlan) -> Result<Vec<u8>> {
        self.check(plan)?;
        let mut writer = CanonicalWriter::new();
        writer.put_u64(CONSUMER_REQUEST_VERSION);
        put_psc_transcript(
            &mut writer,
            CONSUMER_PSC_VERSION,
            std::slice::from_ref(&self.dot),
            std::slice::from_ref(&self.point_values),
        )?;
        Ok(writer.into_bytes())
    }

    pub fn decode(plan: &ConsumerRequestPlan, bytes: &[u8]) -> Result<Self> {
        plan.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))?;
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        if reader.get_u64()? != CONSUMER_REQUEST_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let (mut psc, point_values) = get_psc_transcript::<SHOWN_MAPS>(
            &mut reader,
            CONSUMER_PSC_VERSION,
            1,
            plan.shape.variables()?,
        )?;
        reader.finish()?;
        let dot = psc.pop().ok_or(Error::InvalidEncoding)?;
        if !psc.is_empty() {
            return Err(Error::InvalidEncoding);
        }
        let point_values = point_values
            .into_iter()
            .next()
            .ok_or(Error::InvalidEncoding)?;
        let proof = Self { dot, point_values };
        proof.check(plan)?;
        Ok(proof)
    }

    pub fn check(&self, plan: &ConsumerRequestPlan) -> Result<()> {
        plan.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))?;
        let variables = plan.shape.variables()?;
        if self.dot.rounds.len() != variables
            || self.dot.y_round.coefficients.len() != 2
            || self
                .dot
                .rounds
                .iter()
                .any(|round| round.coefficients.len() != 2)
        {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }
}

/// The closed raw obligation of the request: every head's target was derived
/// from that head's actual raw accumulator, the batch challenge combines them,
/// and the combined target is the one the product sumcheck proves. Both sides
/// derive the whole obligation from the public raw rows; no target is trusted
/// from the proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerRawBatchObligation {
    pub request: u64,
    pub activation_step: u32,
    pub shift: i64,
    pub head_width: u64,
    pub position_coefficients: Vec<Fp3>,
    pub heads: Vec<ConsumerRawHead>,
    pub batch: Option<Fp3>,
    pub target: Fp3,
    pub raw_digest: Digest,
    pub output_digest: Digest,
}

/// One head's raw claim: its source step, its slice and its derived target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerRawHead {
    pub head: u32,
    pub source_step: u32,
    pub slice: (u64, u64),
    pub target: Fp3,
    pub column_coefficients: Vec<Fp3>,
}

impl ConsumerRawBatchObligation {
    /// The combined target on the raw rows: the batch powers over the head
    /// targets, each recomputed from its own raw rows.
    pub fn value_on(&self, raw: &[i128]) -> Result<Fp3> {
        if raw.len()
            != self
                .heads
                .len()
                .checked_mul(self.position_coefficients.len())
                .and_then(|value| value.checked_mul(self.head_width as usize))
                .ok_or(Error::ResourceLimit)?
        {
            return Err(Error::ShapeMismatch);
        }
        let width = usize::try_from(self.head_width).map_err(|_| Error::ResourceLimit)?;
        let per_head = self.position_coefficients.len() * width;
        let mut total = Fp3::ZERO;
        let mut power = Fp3::ONE;
        for (head, claim) in self.heads.iter().enumerate() {
            let start = head.checked_mul(per_head).ok_or(Error::ResourceLimit)?;
            let field: Vec<Fp3> = raw[start..start + per_head]
                .iter()
                .map(|value| Fp3::from_base(fp_from_signed_word(*value as u64)))
                .collect();
            let target = raw_target(
                self.position_coefficients.len(),
                width,
                &field,
                &self.position_coefficients,
                &claim.column_coefficients,
            )?;
            total = total.add(&power.mul(&target));
            power = match self.batch {
                Some(theta) => power.mul(&theta),
                None => Fp3::ONE,
            };
        }
        Ok(total)
    }

    /// The target holds on the given flattened raw rows (head-major, then
    /// position-major).
    pub fn check_on(&self, raw: &[i128]) -> Result<()> {
        if self.value_on(raw)? != self.target {
            return Err(Error::InvalidProof);
        }
        Ok(())
    }

    /// The rounding relation on the same flattened raw rows and the public
    /// output rows: every entry must satisfy `out = rq(Y, shift)`.
    pub fn check_rounding(&self, raw: &[i128], outputs: &[i64]) -> Result<()> {
        if raw.len() != outputs.len() || raw.is_empty() {
            return Err(Error::ShapeMismatch);
        }
        let shift = u32::try_from(self.shift).map_err(|_| Error::InvalidConfig)?;
        for (value, output) in raw.iter().zip(outputs) {
            if rq(*value, shift)? != i128::from(*output) {
                return Err(Error::InvalidProof);
            }
        }
        Ok(())
    }
}

/// The checked outcome of one request: the sumcheck point, the one pending
/// row over the matrix, the mask row and the closed raw obligation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerRequestOutcome {
    pub point: Vec<Fp3>,
    pub pending: PendingRow,
    pub mask_row: ConsumerMaskRow,
    pub raw: ConsumerRawBatchObligation,
    pub coefficients: ConsumerRequestCoefficients,
}

fn outcome(
    plan: &ConsumerRequestPlan,
    coefficients: &ConsumerRequestCoefficients,
    targets: &[Fp3],
    target: Fp3,
    output: &DotOutput,
    raw_digest: Digest,
    output_digest: Digest,
) -> Result<ConsumerRequestOutcome> {
    let shape = &plan.shape;
    let variables = shape.variables()?;
    if output.point.len() != variables
        || output.hiding_form.len() != shape.hiding_entries()?
        || output.ends.iter().any(|end| end.point != output.point)
        || targets.len() != plan.obligations.len()
    {
        return Err(Error::InvalidProof);
    }
    let combined = coefficients.combined()?;
    let pending = PendingRow {
        weights: stretch_weights(shape.table_row_bits()?, shape.slice.0, &combined)?,
        psi: output.point.clone(),
        target: output.ends[1].value,
        masks: vec![MaskTerm {
            oracle: plan.mask.oracle,
            slot: plan.mask.shown_slot,
            coeff: output.ends[1].mask_scale,
        }],
    };
    let mask_row = ConsumerMaskRow {
        oracle: plan.mask.oracle,
        base_slot: plan.mask.hiding_base,
        coefficients: output.hiding_form.clone(),
        target: output.row_target,
    };
    let heads = plan
        .obligations
        .iter()
        .zip(targets)
        .map(|(obligation, target)| ConsumerRawHead {
            head: obligation.head,
            source_step: obligation.source_step,
            slice: obligation.shape.slice,
            target: *target,
            column_coefficients: coefficients.heads[obligation.head as usize].clone(),
        })
        .collect();
    let raw = ConsumerRawBatchObligation {
        request: plan.request,
        activation_step: plan.obligations[0].activation_step,
        shift: plan.shift,
        head_width: plan.head_width,
        position_coefficients: coefficients.position.clone(),
        heads,
        batch: coefficients.batch,
        target,
        raw_digest,
        output_digest,
    };
    Ok(ConsumerRequestOutcome {
        point: output.point.clone(),
        pending,
        mask_row,
        raw,
        coefficients: coefficients.clone(),
    })
}

fn derive_targets(
    plan: &ConsumerRequestPlan,
    coefficients: &mut ConsumerRequestCoefficients,
    chain: &mut Chain,
    raw: &[Vec<i64>],
) -> Result<(Vec<Fp3>, Fp3)> {
    if plan.obligations.len() > 1 {
        let theta = chain.draw_e()?;
        coefficients.batch = Some(theta);
    }
    let targets = consumer_head_targets(plan, coefficients, raw)?;
    let target = combined_target(coefficients, &targets, plan.obligations.len())?;
    Ok((targets, target))
}

fn combined_raw_rows(plan: &ConsumerRequestPlan, raw: &[Vec<i64>]) -> Result<Vec<Fp3>> {
    let positions = plan.shape.positions as usize;
    let width = usize::try_from(plan.head_width).map_err(|_| Error::ResourceLimit)?;
    let hidden = usize::try_from(plan.shape.slice_len()?).map_err(|_| Error::ResourceLimit)?;
    let mut combined = vec![Fp3::ZERO; positions * hidden];
    for (head, rows) in raw.iter().enumerate() {
        for position in 0..positions {
            for index in 0..width {
                combined[position * hidden + head * width + index] =
                    Fp3::from_base(fp_from_signed_word(rows[position * width + index] as u64));
            }
        }
    }
    Ok(combined)
}

fn check_inputs(
    plan: &ConsumerRequestPlan,
    activation: &[Fp3],
    raw: &[Vec<i64>],
    outputs: &[Vec<i64>],
) -> Result<()> {
    plan.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))?;
    let positions = plan.shape.positions as usize;
    let records = usize::try_from(plan.shape.records).map_err(|_| Error::ResourceLimit)?;
    if activation.len() != positions.checked_mul(records).ok_or(Error::ResourceLimit)? {
        return Err(Error::ShapeMismatch);
    }
    check_rows(
        raw,
        plan.obligations.len(),
        plan.shape.positions,
        plan.head_width,
    )?;
    check_rows(
        outputs,
        plan.obligations.len(),
        plan.shape.positions,
        plan.head_width,
    )?;
    check_consumer_raw_rounding(plan, raw, outputs)
}

/// Prover side of the request. One chain, one product-sumcheck instance over
/// the whole consumed region. The raw relation is checked first; the raw
/// targets are derived from the same public rows the verifier uses; the batch
/// challenge then combines the heads and the product sumcheck proves the
/// combined claim. An honest prover refuses a raw row that does not round to
/// the head's public output and refuses a relation inconsistent with its own
/// matrix.
pub fn prove_consumer_request(
    chain: &mut Chain,
    plan: &ConsumerRequestPlan,
    activation: &[Fp3],
    raw: &[Vec<i64>],
    outputs: &[Vec<i64>],
    matrix_places: &[Vec<Fp3>],
    masks: &ConsumerPscMasks,
) -> Result<(ConsumerRequestProof, ConsumerRequestOutcome)> {
    check_inputs(plan, activation, raw, outputs)?;
    masks.check(&plan.shape)?;
    let positions = plan.shape.positions as usize;
    let records = usize::try_from(plan.shape.records).map_err(|_| Error::ResourceLimit)?;
    if matrix_places.len()
        != usize::try_from(plan.shape.places).map_err(|_| Error::ResourceLimit)?
        || matrix_places.iter().any(|row| row.len() != records)
    {
        return Err(Error::ShapeMismatch);
    }
    let raw_digest = consumer_raw_digest(plan, raw)?;
    let output_digest = consumer_output_digest(plan, outputs)?;
    plan.absorb(chain, &raw_digest, &output_digest)?;
    let mut coefficients = draw_coefficients(chain, plan)?;
    let (targets, target) = derive_targets(plan, &mut coefficients, chain, raw)?;
    let combined = coefficients.combined()?;
    let combined_raw = combined_raw_rows(plan, raw)?;
    let reduction = consumer_reduction(
        positions,
        plan.shape.records,
        plan.shape.places,
        plan.shape.slice,
        activation,
        &combined_raw,
        &coefficients.position,
        &combined,
    )?;
    let columns = reduction.column_values(matrix_places)?;
    let claim = DotClaim {
        masks: Vec::new(),
        target,
    };
    let mut challenges = PositionPscChallenges::new(chain, CONSUMER_REQUEST_PSC_TAGS, 0);
    let (dot, output) = prove_dot_sumcheck(
        &reduction.shown,
        &columns,
        &claim,
        &masks.entries,
        &mut challenges,
    )?;
    check_product_claims(&output, &claim, &masks.entries, &reduction.shown, &columns)?;
    let outcome = outcome(
        plan,
        &coefficients,
        &targets,
        target,
        &output,
        raw_digest,
        output_digest,
    )?;
    crate::proof::hidden_member::check_pending(
        &outcome.pending,
        matrix_places,
        &|oracle, slot| {
            if oracle == plan.mask.oracle && slot == plan.mask.shown_slot {
                Ok(masks.entries.shown[1])
            } else {
                Err(Error::ShapeMismatch)
            }
        },
    )?;
    let proof = ConsumerRequestProof {
        point_values: [output.ends[0].value, output.ends[1].value],
        dot,
    };
    Ok((proof, outcome))
}

/// Verifier side of the request. The verifier checks the raw rounding relation
/// on the public rows, draws the same coefficients, derives the same targets
/// and the same batch challenge, replays the product sumcheck and checks the
/// public activation map at its own point. It never sees the matrix or the
/// mask entries, and no prover-supplied target enters the transcript.
pub fn verify_consumer_request(
    chain: &mut Chain,
    plan: &ConsumerRequestPlan,
    activation: &[Fp3],
    raw: &[Vec<i64>],
    outputs: &[Vec<i64>],
    proof: &ConsumerRequestProof,
) -> Result<ConsumerRequestOutcome> {
    check_inputs(plan, activation, raw, outputs)?;
    proof.check(plan)?;
    let raw_digest = consumer_raw_digest(plan, raw)?;
    let output_digest = consumer_output_digest(plan, outputs)?;
    plan.absorb(chain, &raw_digest, &output_digest)?;
    let mut coefficients = draw_coefficients(chain, plan)?;
    let (targets, target) = derive_targets(plan, &mut coefficients, chain, raw)?;
    let entries = DotMaskEntries {
        hiding: vec![Fp3::ZERO; plan.shape.hiding_entries()?],
        shown: [Fp3::ZERO; SHOWN_MAPS],
    };
    let claim = DotClaim {
        masks: Vec::new(),
        target,
    };
    let mut challenges = PositionPscChallenges::new(chain, CONSUMER_REQUEST_PSC_TAGS, 0);
    let output = verify_psc_transcript(
        &claim,
        &entries,
        &proof.dot,
        plan.shape.variables()?,
        &mut challenges,
    )?;
    if proof.point_values != [output.ends[0].value, output.ends[1].value] {
        return Err(Error::InvalidProof);
    }
    let shown = activation_shown(
        plan.shape.positions as usize,
        plan.shape.records,
        activation,
        &coefficients.position,
    )?;
    if multilinear_eval(&shown, &output.point)? != output.ends[0].value {
        return Err(Error::InvalidProof);
    }
    outcome(
        plan,
        &coefficients,
        &targets,
        target,
        &output,
        raw_digest,
        output_digest,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::field::fp_from_u64;

    use crate::proof::consumer_psc::ConsumerPscMasks;

    fn value(word: u64) -> Fp3 {
        Fp3::from_base(fp_from_u64(word))
    }

    fn table(path: &str, root: [u8; 32]) -> ConsumerTable {
        ConsumerTable {
            path: path.to_string(),
            oracle: 30,
            root: Digest(root),
            identity: Digest([0x12; 32]),
            configuration: Digest([0x13; 32]),
        }
    }

    fn mask() -> ConsumerMaskSource {
        ConsumerMaskSource {
            oracle: 31,
            hiding_base: 40,
            shown_slot: 60,
        }
    }

    fn obligation(head: u32, source_step: u32) -> ConsumerObligation {
        ConsumerObligation {
            request: 7,
            source_step,
            activation_step: 9,
            head,
            shape: ConsumerPscShape {
                positions: 3,
                places: 16,
                records: 8,
                slice: (u64::from(head) * 4, (u64::from(head) + 1) * 4),
            },
            shift: 16,
            table: table("weights/L0_k_w_i32.bin", [0x11; 32]),
            mask: mask(),
        }
    }

    fn plan(heads: u32) -> ConsumerRequestPlan {
        let obligations = (0..heads).map(|head| obligation(head, 20 + head)).collect();
        ConsumerRequestPlan::new(
            7,
            table("weights/L0_k_w_i32.bin", [0x11; 32]),
            mask(),
            obligations,
        )
        .unwrap()
    }

    /// The fixture keeps the integers small enough that the field embeddings
    /// of the activation, the matrix and every accumulator are faithful.
    fn activation_words(plan: &ConsumerRequestPlan) -> Vec<i64> {
        let count = plan.shape.positions as usize * plan.shape.records as usize;
        (0..count)
            .map(|index| ((index as i64 * 5 + 2) % 13) + 1)
            .collect()
    }

    fn matrix_words(plan: &ConsumerRequestPlan) -> Vec<Vec<i64>> {
        (0..plan.shape.places as usize)
            .map(|place| {
                (0..plan.shape.records as usize)
                    .map(|record| (((record as i64 * 7 + place as i64 * 3 + 1) % 19) + 1) * 3001)
                    .collect()
            })
            .collect()
    }

    fn embed(values: &[i64]) -> Vec<Fp3> {
        values
            .iter()
            .map(|value| Fp3::from_base(fp_from_signed_word(*value as u64)))
            .collect()
    }

    fn raw_words(
        plan: &ConsumerRequestPlan,
        activation: &[i64],
        matrix: &[Vec<i64>],
    ) -> Vec<Vec<i64>> {
        let positions = plan.shape.positions as usize;
        let records = plan.shape.records as usize;
        let width = plan.head_width as usize;
        let mut heads = Vec::with_capacity(plan.obligations.len());
        for head in 0..plan.obligations.len() {
            let start = head * width;
            let mut rows = vec![0i64; positions * width];
            for position in 0..positions {
                for index in 0..width {
                    let mut sum = 0i64;
                    for record in 0..records {
                        sum +=
                            activation[position * records + record] * matrix[start + index][record];
                    }
                    rows[position * width + index] = sum;
                }
            }
            heads.push(rows);
        }
        heads
    }

    fn output_words(plan: &ConsumerRequestPlan, raw: &[Vec<i64>]) -> Vec<Vec<i64>> {
        let shift = u32::try_from(plan.shift).unwrap();
        raw.iter()
            .map(|head| {
                head.iter()
                    .map(|value| i64::try_from(rq(i128::from(*value), shift).unwrap()).unwrap())
                    .collect()
            })
            .collect()
    }

    fn masks(shape: &ConsumerPscShape) -> ConsumerPscMasks {
        let hiding = (0..shape.hiding_entries().unwrap())
            .map(|index| value(100 + index as u64))
            .collect();
        ConsumerPscMasks::new(hiding, value(200))
    }

    fn chain() -> Chain {
        Chain::begin(
            92,
            &Digest::ZERO,
            b"consumer request",
            b"request fixture",
            None,
        )
        .unwrap()
    }

    fn fixture(
        heads: u32,
    ) -> (
        ConsumerRequestPlan,
        Vec<Fp3>,
        Vec<Vec<Fp3>>,
        Vec<Vec<i64>>,
        Vec<Vec<i64>>,
        ConsumerPscMasks,
    ) {
        let plan = plan(heads);
        let activation_words = activation_words(&plan);
        let matrix_words = matrix_words(&plan);
        let raw = raw_words(&plan, &activation_words, &matrix_words);
        let outputs = output_words(&plan, &raw);
        let activation = embed(&activation_words);
        let matrix = matrix_words.iter().map(|row| embed(row)).collect();
        let masks = masks(&plan.shape);
        (plan, activation, matrix, raw, outputs, masks)
    }

    fn run(
        heads: u32,
    ) -> (
        ConsumerRequestPlan,
        ConsumerRequestProof,
        ConsumerRequestOutcome,
        ConsumerRequestOutcome,
        Vec<Fp3>,
        Vec<Vec<Fp3>>,
        Vec<Vec<i64>>,
        Vec<Vec<i64>>,
        ConsumerPscMasks,
    ) {
        let (plan, activation, matrix, raw, outputs, masks) = fixture(heads);
        let (proof, proven) = prove_consumer_request(
            &mut chain(),
            &plan,
            &activation,
            &raw,
            &outputs,
            &matrix,
            &masks,
        )
        .unwrap();
        let verified =
            verify_consumer_request(&mut chain(), &plan, &activation, &raw, &outputs, &proof)
                .unwrap();
        (
            plan, proof, proven, verified, activation, matrix, raw, outputs, masks,
        )
    }

    #[test]
    fn one_request_batches_every_head_into_one_pending() {
        let (plan, proof, proven, verified, activation, matrix, raw, outputs, masks) = run(3);
        assert_eq!(proven, verified);
        let outcome = &verified;
        assert_eq!(plan.heads(), 3);
        assert_eq!(plan.consumed(), 12);
        assert!(outcome.coefficients.batch.is_some());
        assert_eq!(
            outcome.pending.weights,
            stretch_weights(4, 0, &outcome.coefficients.combined().unwrap()).unwrap()
        );
        assert_eq!(outcome.pending.psi, outcome.point);
        assert_eq!(outcome.pending.masks.len(), 1);
        assert_eq!(outcome.pending.masks[0].oracle, plan.mask.oracle);
        assert_eq!(outcome.pending.masks[0].slot, plan.mask.shown_slot);
        // The same-matrix pending row holds against the real matrix and the
        // one-time shown entry.
        crate::proof::hidden_member::check_pending(&outcome.pending, &matrix, &|oracle, slot| {
            if oracle == plan.mask.oracle && slot == plan.mask.shown_slot {
                Ok(masks.entries.shown[1])
            } else {
                Err(Error::ShapeMismatch)
            }
        })
        .unwrap();
        // The targets are derived from the public raw rows, and the raw
        // relation closes on them.
        let derived = consumer_head_targets(&plan, &outcome.coefficients, &raw).unwrap();
        assert_eq!(derived.len(), 3);
        let flat: Vec<i128> = raw
            .iter()
            .flat_map(|head| head.iter().map(|value| i128::from(*value)))
            .collect();
        outcome.raw.check_on(&flat).unwrap();
        assert_eq!(
            outcome.raw.raw_digest,
            consumer_raw_digest(&plan, &raw).unwrap()
        );
        assert_eq!(
            outcome.raw.output_digest,
            consumer_output_digest(&plan, &outputs).unwrap()
        );
        let flat_outputs: Vec<i64> = outputs.iter().flat_map(|head| head.clone()).collect();
        outcome.raw.check_rounding(&flat, &flat_outputs).unwrap();
        // The message round-trips and the decoded message verifies.
        let bytes = proof.encode(&plan).unwrap();
        let decoded = ConsumerRequestProof::decode(&plan, &bytes).unwrap();
        assert_eq!(decoded, proof);
        verify_consumer_request(&mut chain(), &plan, &activation, &raw, &outputs, &decoded)
            .unwrap();
    }

    #[test]
    fn one_head_takes_the_direct_step() {
        let (plan, _proof, proven, verified, _activation, matrix, raw, outputs, masks) = run(1);
        assert_eq!(proven, verified);
        assert_eq!(plan.heads(), 1);
        assert!(verified.coefficients.batch.is_none());
        assert_eq!(verified.raw.heads.len(), 1);
        crate::proof::hidden_member::check_pending(&verified.pending, &matrix, &|oracle, slot| {
            if oracle == plan.mask.oracle && slot == plan.mask.shown_slot {
                Ok(masks.entries.shown[1])
            } else {
                Err(Error::ShapeMismatch)
            }
        })
        .unwrap();
        let flat: Vec<i128> = raw
            .iter()
            .flat_map(|head| head.iter().map(|value| i128::from(*value)))
            .collect();
        let flat_outputs: Vec<i64> = outputs.iter().flat_map(|head| head.clone()).collect();
        verified.raw.check_on(&flat).unwrap();
        verified.raw.check_rounding(&flat, &flat_outputs).unwrap();
    }

    #[test]
    fn the_raw_relation_replaces_the_requantized_output() {
        let (plan, proof, _proven, _verified, activation, matrix, raw, outputs, masks) = run(2);
        // The requantized output is not the raw accumulator: substituting it
        // for the raw rows fails the rounding relation and the prover refuses.
        let substituted: Vec<Vec<i64>> = outputs.clone();
        assert!(check_consumer_raw_rounding(&plan, &substituted, &outputs).is_err());
        assert!(prove_consumer_request(
            &mut chain(),
            &plan,
            &activation,
            &substituted,
            &outputs,
            &matrix,
            &masks
        )
        .is_err());
        // A changed output row breaks the relation and the verification.
        let mut changed_outputs = outputs.clone();
        changed_outputs[0][0] += 1;
        assert!(check_consumer_raw_rounding(&plan, &raw, &changed_outputs).is_err());
        assert!(verify_consumer_request(
            &mut chain(),
            &plan,
            &activation,
            &raw,
            &changed_outputs,
            &proof
        )
        .is_err());
    }

    #[test]
    fn another_preimage_value_cannot_stand_for_the_accumulator() {
        let (plan, activation, matrix, raw, outputs, masks) = fixture(1);
        // Move one raw entry by one without changing its rounding: the raw
        // relation still holds, but the derived target changes and the matrix
        // relation no longer matches. The accumulator is not free inside its
        // rounding interval.
        let mut moved = raw.clone();
        let entry = moved[0]
            .iter()
            .position(|value| {
                rq(i128::from(*value), 16).unwrap() == rq(i128::from(*value) + 1, 16).unwrap()
            })
            .expect("a rounding interval of length above one");
        moved[0][entry] += 1;
        assert!(check_consumer_raw_rounding(&plan, &moved, &outputs).is_ok());
        assert!(prove_consumer_request(
            &mut chain(),
            &plan,
            &activation,
            &moved,
            &outputs,
            &matrix,
            &masks
        )
        .is_err());
        // A proof made on the moved rows does not verify on the original rows.
        let (proof, _outcome) = prove_consumer_request(
            &mut chain(),
            &plan,
            &activation,
            &raw,
            &outputs,
            &matrix,
            &masks,
        )
        .unwrap();
        assert!(verify_consumer_request(
            &mut chain(),
            &plan,
            &activation,
            &moved,
            &outputs,
            &proof
        )
        .is_err());
    }

    #[test]
    fn a_changed_proof_or_statement_is_refused() {
        let (plan, proof, _proven, _verified, activation, _matrix, raw, outputs, _masks) = run(2);

        let mut changed = proof.clone();
        changed.point_values[1] = changed.point_values[1].add(&value(1));
        assert!(verify_consumer_request(
            &mut chain(),
            &plan,
            &activation,
            &raw,
            &outputs,
            &changed
        )
        .is_err());

        let mut changed = proof.clone();
        changed.dot.rounds[0].coefficients[0] = value(7);
        assert!(verify_consumer_request(
            &mut chain(),
            &plan,
            &activation,
            &raw,
            &outputs,
            &changed
        )
        .is_err());

        // Another request source, another head source page or another table
        // changes the statement and the challenges.
        let mut other = plan.clone();
        other.obligations[0].source_step += 1;
        assert!(
            verify_consumer_request(&mut chain(), &other, &activation, &raw, &outputs, &proof)
                .is_err()
        );
        let mut other = plan.clone();
        other.request += 1;
        assert!(
            verify_consumer_request(&mut chain(), &other, &activation, &raw, &outputs, &proof)
                .is_err()
        );
        let mut other = plan;
        other.table.root = Digest([0x77; 32]);
        assert!(
            verify_consumer_request(&mut chain(), &other, &activation, &raw, &outputs, &proof)
                .is_err()
        );
    }

    #[test]
    fn the_request_registration_binds_the_head_sources_and_coverage() {
        // A gap or an overlap between head slices is not a request.
        let obligations = vec![obligation(0, 20), obligation(2, 22)];
        assert!(ConsumerRequestPlan::new(
            7,
            table("weights/L0_k_w_i32.bin", [0x11; 32]),
            mask(),
            obligations
        )
        .is_err());

        // A head from another activation step is another request source.
        let mut other = obligation(1, 21);
        other.activation_step = 10;
        assert!(ConsumerRequestPlan::new(
            7,
            table("weights/L0_k_w_i32.bin", [0x11; 32]),
            mask(),
            vec![obligation(0, 20), other]
        )
        .is_err());

        // A different table is not the same request.
        let mut other = obligation(1, 21);
        other.table = table("weights/L0_q_w_i32.bin", [0x21; 32]);
        assert!(ConsumerRequestPlan::new(
            7,
            table("weights/L0_k_w_i32.bin", [0x11; 32]),
            mask(),
            vec![obligation(0, 20), other]
        )
        .is_err());

        // A different mask source is not the same request.
        let mut other = obligation(1, 21);
        other.mask.shown_slot += 1;
        assert!(ConsumerRequestPlan::new(
            7,
            table("weights/L0_k_w_i32.bin", [0x11; 32]),
            mask(),
            vec![obligation(0, 20), other]
        )
        .is_err());

        // The real 12-head geometry registers and covers exactly the visible
        // width, and J is the registered head count.
        let real = ConsumerRequestPlan::new(
            1,
            table("gpt2/weights/L0_k_w_i32.bin", [0x31; 32]),
            ConsumerMaskSource {
                oracle: 41,
                hiding_base: 400,
                shown_slot: 900,
            },
            (0..12)
                .map(|head| ConsumerObligation {
                    request: 1,
                    source_step: 100 + head,
                    activation_step: 90,
                    head,
                    shape: ConsumerPscShape {
                        positions: 16,
                        places: 1024,
                        records: 1024,
                        slice: (u64::from(head) * 64, (u64::from(head) + 1) * 64),
                    },
                    shift: 16,
                    table: table("gpt2/weights/L0_k_w_i32.bin", [0x31; 32]),
                    mask: ConsumerMaskSource {
                        oracle: 41,
                        hiding_base: 400,
                        shown_slot: 900,
                    },
                })
                .collect(),
        )
        .unwrap();
        assert_eq!(real.heads(), 12);
        assert_eq!(real.consumed(), 768);
        assert_eq!(real.head_width, 64);
        assert_eq!(real.shape.slice, (0, 768));
    }
}
