//! The projection consumer request: one request, one chain, all head slices.
//!
//! A real projection request consumes one table through every head slice of
//! the layer. The request binding walks the head claims of one request in one
//! chain: the request statement fixes the request, the table, the geometry,
//! the mask source and the ordered head sources before any challenge; the
//! position coefficients and the per-head column coefficients follow; then the
//! prover fixes every head's raw target in the chain, one by one, and only
//! after them the batch challenge combines the heads into one product-sumcheck
//! claim over the whole consumed region.
//!
//! With `J` registered heads the request performs the batch step (`J > 1`) or
//! the direct step (`J = 1`):
//!
//! * every head has its own column coefficients `b_h` and its own raw target
//!   `T_h = Σ_p Σ_j a[p] b_h[j] Y_h[p, j]`, fixed before the batch challenge;
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
//! The first head target fixed before `theta` makes the batch binding: a
//! request whose head relation fails fails the combined claim for the drawn
//! challenge, exactly as the batch of `J` claims must. The per-head targets
//! are registered as the request's raw obligation; closing them is the raw
//! relation's duty, never this module's trust.

use zkie_core::common::codec::DecodeLimits;
use zkie_core::error::{Error, Result};
use zkie_engine::pcs::request::chain::Chain;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};

use crate::proof::committed_psc::{
    get_psc_transcript, put_psc_transcript, PositionPscChallenges, PscTranscriptTags,
};
use crate::proof::consumer_psc::{
    ConsumerMaskRow, ConsumerObligation, ConsumerPscMasks, ConsumerPscShape, ConsumerTable,
    CONSUMER_PSC_VERSION,
};
use crate::proof::hidden_member::{stretch_weights, MaskTerm, PendingRow};
use crate::proof::projection_claim::{activation_shown, consumer_reduction, raw_target};
use crate::proof::psc::{
    check_product_claims, hp_entries, multilinear_eval, prove_dot_sumcheck, psc_variables,
    verify_psc_transcript, DotClaim, DotMaskEntries, DotOutput, DotProof, SHOWN_MAPS,
};

/// Version of the request proof message and of its transcript messages.
pub const CONSUMER_REQUEST_VERSION: u64 = 1;
/// Transcript tag of the request statement.
const CONSUMER_REQUEST_TAG: u64 = 86;
/// Transcript tag of one head's raw target, before the batch challenge.
const CONSUMER_HEAD_TARGET_TAG: u64 = 87;
/// Transcript tags of the request's product sumcheck. The claim word is
/// reserved: the plain product sumcheck carries no claim masks.
pub const CONSUMER_REQUEST_PSC_TAGS: PscTranscriptTags = PscTranscriptTags {
    version: CONSUMER_REQUEST_VERSION,
    hiding_sum: 88,
    round: 89,
    end: 90,
    claim: 91,
};

/// The one-request consumer plan: the registered head obligations of a single
/// request, their common table and the single product-sumcheck instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerRequestPlan {
    pub request: u64,
    pub table: ConsumerTable,
    pub shift: i64,
    pub shape: ConsumerPscShape,
    pub mask: crate::proof::consumer_psc::ConsumerMaskSource,
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
        mask: crate::proof::consumer_psc::ConsumerMaskSource,
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

    fn absorb(&self, chain: &mut Chain) -> Result<()> {
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

fn push_digest(
    message: &mut zkie_engine::pcs::request::chain::ChainMessage,
    digest: &zkie_core::types::Digest,
) {
    for word in digest.0.chunks_exact(8) {
        message.push_u64(u64::from_le_bytes(
            word.try_into().expect("eight-byte word"),
        ));
    }
}

/// The transcript coefficients of one request: the common position
/// coefficients, one column vector per head and, after the head targets, the
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

fn absorb_head_target(chain: &mut Chain, head: u32, target: Fp3) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(CONSUMER_REQUEST_VERSION);
    message.push_u64(CONSUMER_HEAD_TARGET_TAG);
    message.push_u64(u64::from(head));
    message.push_fp3(&target);
    let prefix = message.prefix(zkie_engine::pcs::request::chain::KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// The request proof message: every head's raw target (fixed before the batch
/// challenge), the combined target and the one product-sumcheck transcript.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerRequestProof {
    pub head_targets: Vec<Fp3>,
    pub target: Fp3,
    pub dot: DotProof,
    pub point_values: [Fp3; SHOWN_MAPS],
}

impl ConsumerRequestProof {
    pub fn encode(&self, plan: &ConsumerRequestPlan) -> Result<Vec<u8>> {
        self.check(plan)?;
        let mut writer = CanonicalWriter::new();
        writer.put_u64(CONSUMER_REQUEST_VERSION);
        writer.put_u64(self.head_targets.len() as u64);
        for target in &self.head_targets {
            writer.put_fp3(target);
        }
        writer.put_fp3(&self.target);
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
        let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if count != plan.obligations.len() {
            return Err(Error::ShapeMismatch);
        }
        let mut head_targets = Vec::with_capacity(count);
        for _ in 0..count {
            head_targets.push(reader.get_fp3()?);
        }
        let target = reader.get_fp3()?;
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
        let proof = Self {
            head_targets,
            target,
            dot,
            point_values,
        };
        proof.check(plan)?;
        Ok(proof)
    }

    pub fn check(&self, plan: &ConsumerRequestPlan) -> Result<()> {
        plan.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))?;
        if self.head_targets.len() != plan.obligations.len() {
            return Err(Error::ShapeMismatch);
        }
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

/// The registered raw obligation of the request: every head's raw target was
/// fixed before the batch challenge, the batch challenge combines them, and
/// the combined target is the one the product sumcheck proves. The obligation
/// closes on the real raw rows of the head steps; closing it inside the proof
/// is the raw relation's duty.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerRawBatchObligation {
    pub request: u64,
    pub activation_step: u32,
    pub position_coefficients: Vec<Fp3>,
    pub head_width: u64,
    pub heads: Vec<ConsumerRawHead>,
    pub batch: Option<Fp3>,
    pub target: Fp3,
}

/// One head's registered raw claim: its source step, its slice and its target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerRawHead {
    pub head: u32,
    pub source_step: u32,
    pub slice: (u64, u64),
    pub target: Fp3,
    pub column_coefficients: Vec<Fp3>,
}

impl ConsumerRawBatchObligation {
    /// The combined target on the real raw rows: the batch powers over the
    /// head targets, each recomputed from its own raw rows.
    pub fn value_on(&self, raw: &[Vec<Fp3>]) -> Result<Fp3> {
        if raw.len() != self.heads.len() {
            return Err(Error::ShapeMismatch);
        }
        let mut total = Fp3::ZERO;
        let mut power = Fp3::ONE;
        for (head, rows) in self.heads.iter().zip(raw) {
            let target = raw_target(
                self.position_coefficients.len(),
                head.column_coefficients.len(),
                rows,
                &self.position_coefficients,
                &head.column_coefficients,
            )?;
            total = total.add(&power.mul(&target));
            power = match self.batch {
                Some(theta) => power.mul(&theta),
                None => Fp3::ONE,
            };
        }
        Ok(total)
    }

    pub fn check_on(&self, raw: &[Vec<Fp3>]) -> Result<()> {
        if self.value_on(raw)? != self.target {
            return Err(Error::InvalidProof);
        }
        Ok(())
    }
}

/// The checked outcome of one request: the sumcheck point, the one pending
/// row over the matrix, the mask row and the registered raw obligation.
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
    head_targets: &[Fp3],
    target: Fp3,
    output: &DotOutput,
) -> Result<ConsumerRequestOutcome> {
    let shape = &plan.shape;
    let variables = shape.variables()?;
    if output.point.len() != variables
        || output.hiding_form.len() != shape.hiding_entries()?
        || output.ends.iter().any(|end| end.point != output.point)
        || head_targets.len() != plan.obligations.len()
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
        .zip(head_targets)
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
        position_coefficients: coefficients.position.clone(),
        head_width: plan.head_width,
        heads,
        batch: coefficients.batch,
        target,
    };
    Ok(ConsumerRequestOutcome {
        point: output.point.clone(),
        pending,
        mask_row,
        raw,
        coefficients: coefficients.clone(),
    })
}

/// Prover side of the request. One chain, one product-sumcheck instance over
/// the whole consumed region. Every head's raw target is absorbed before the
/// batch challenge; the batch challenge then combines the heads and the
/// product sumcheck proves the combined claim. An honest prover refuses a raw
/// target that is not its own reduction.
pub fn prove_consumer_request(
    chain: &mut Chain,
    plan: &ConsumerRequestPlan,
    activation: &[Fp3],
    raw_heads: &[Vec<Fp3>],
    matrix_places: &[Vec<Fp3>],
    masks: &ConsumerPscMasks,
) -> Result<(ConsumerRequestProof, ConsumerRequestOutcome)> {
    plan.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))?;
    masks.check(&plan.shape)?;
    let positions = plan.shape.positions as usize;
    let records = usize::try_from(plan.shape.records).map_err(|_| Error::ResourceLimit)?;
    if activation.len() != positions.checked_mul(records).ok_or(Error::ResourceLimit)? {
        return Err(Error::ShapeMismatch);
    }
    if matrix_places.len()
        != usize::try_from(plan.shape.places).map_err(|_| Error::ResourceLimit)?
        || matrix_places.iter().any(|row| row.len() != records)
        || raw_heads.len() != plan.obligations.len()
    {
        return Err(Error::ShapeMismatch);
    }
    plan.absorb(chain)?;
    let mut coefficients = draw_coefficients(chain, plan)?;
    let mut head_targets = Vec::with_capacity(raw_heads.len());
    let hidden = usize::try_from(plan.shape.slice_len()?).map_err(|_| Error::ResourceLimit)?;
    let mut combined_raw = vec![Fp3::ZERO; positions * hidden];
    for (head, rows) in raw_heads.iter().enumerate() {
        let width = usize::try_from(plan.head_width).map_err(|_| Error::ResourceLimit)?;
        if rows.len() != positions * width {
            return Err(Error::ShapeMismatch);
        }
        let target = raw_target(
            positions,
            width,
            rows,
            &coefficients.position,
            &coefficients.heads[head],
        )?;
        absorb_head_target(chain, head as u32, target)?;
        head_targets.push(target);
        // The consumed region's raw rows are the head rows in position-major
        // order: position `p`, head `h`, offset `j` is the entry
        // `p * hidden + h * width + j`.
        for position in 0..positions {
            for index in 0..width {
                combined_raw[position * hidden + head * width + index] =
                    rows[position * width + index];
            }
        }
    }
    if plan.obligations.len() > 1 {
        let theta = chain.draw_e()?;
        coefficients.batch = Some(theta);
    }
    let combined = coefficients.combined()?;
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
        target: reduction.target,
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
    let outcome = outcome(plan, &coefficients, &head_targets, claim.target, &output)?;
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
        head_targets,
        target: claim.target,
        point_values: [output.ends[0].value, output.ends[1].value],
        dot,
    };
    Ok((proof, outcome))
}

/// Verifier side of the request. The verifier draws the same coefficients,
/// absorbs the same per-head targets from the proof, draws the same batch
/// challenge, replays the product sumcheck and checks the public activation
/// map at its own point. It never sees the matrix, the raw rows or the mask
/// entries.
pub fn verify_consumer_request(
    chain: &mut Chain,
    plan: &ConsumerRequestPlan,
    activation: &[Fp3],
    proof: &ConsumerRequestProof,
) -> Result<ConsumerRequestOutcome> {
    plan.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))?;
    proof.check(plan)?;
    let positions = plan.shape.positions as usize;
    let records = usize::try_from(plan.shape.records).map_err(|_| Error::ResourceLimit)?;
    if activation.len() != positions.checked_mul(records).ok_or(Error::ResourceLimit)? {
        return Err(Error::ShapeMismatch);
    }
    plan.absorb(chain)?;
    let mut coefficients = draw_coefficients(chain, plan)?;
    for (head, target) in proof.head_targets.iter().enumerate() {
        absorb_head_target(chain, head as u32, *target)?;
    }
    let target = if plan.obligations.len() > 1 {
        let theta = chain.draw_e()?;
        coefficients.batch = Some(theta);
        let mut total = Fp3::ZERO;
        let mut power = Fp3::ONE;
        for head_target in &proof.head_targets {
            total = total.add(&power.mul(head_target));
            power = power.mul(&theta);
        }
        total
    } else {
        proof.head_targets[0]
    };
    if target != proof.target {
        return Err(Error::InvalidProof);
    }
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
        positions,
        plan.shape.records,
        activation,
        &coefficients.position,
    )?;
    if multilinear_eval(&shown, &output.point)? != output.ends[0].value {
        return Err(Error::InvalidProof);
    }
    outcome(plan, &coefficients, &proof.head_targets, target, &output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_core::types::Digest;
    use zkie_engine::pcs::request::field::fp_from_u64;

    use crate::proof::consumer_psc::{ConsumerMaskSource, ConsumerPscMasks};

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

    fn activation(shape: &ConsumerPscShape) -> Vec<Fp3> {
        let count = shape.positions as usize * shape.records as usize;
        (0..count)
            .map(|index| value((index as u64 * 5 + 2) % 13))
            .collect()
    }

    fn matrix(shape: &ConsumerPscShape) -> Vec<Vec<Fp3>> {
        (0..shape.places as usize)
            .map(|place| {
                (0..shape.records as usize)
                    .map(|record| value(((record as u64 * 7 + place as u64 * 3 + 1) % 19) as u64))
                    .collect()
            })
            .collect()
    }

    fn raw_heads(
        plan: &ConsumerRequestPlan,
        activation: &[Fp3],
        matrix: &[Vec<Fp3>],
    ) -> Vec<Vec<Fp3>> {
        let positions = plan.shape.positions as usize;
        let records = plan.shape.records as usize;
        let width = plan.head_width as usize;
        let mut heads = Vec::with_capacity(plan.obligations.len());
        for head in 0..plan.obligations.len() {
            let start = head * width;
            let mut rows = vec![Fp3::ZERO; positions * width];
            for position in 0..positions {
                for index in 0..width {
                    let mut sum = Fp3::ZERO;
                    for record in 0..records {
                        sum = sum.add(
                            &activation[position * records + record]
                                .mul(&matrix[start + index][record]),
                        );
                    }
                    rows[position * width + index] = sum;
                }
            }
            heads.push(rows);
        }
        heads
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

    fn run(
        heads: u32,
    ) -> (
        ConsumerRequestPlan,
        ConsumerRequestProof,
        ConsumerRequestOutcome,
        ConsumerRequestOutcome,
        Vec<Fp3>,
        Vec<Vec<Fp3>>,
        Vec<Vec<Fp3>>,
        ConsumerPscMasks,
    ) {
        let plan = plan(heads);
        let activation = activation(&plan.shape);
        let matrix = matrix(&plan.shape);
        let raw = raw_heads(&plan, &activation, &matrix);
        let masks = masks(&plan.shape);
        let (proof, proven) =
            prove_consumer_request(&mut chain(), &plan, &activation, &raw, &matrix, &masks)
                .unwrap();
        let verified = verify_consumer_request(&mut chain(), &plan, &activation, &proof).unwrap();
        (
            plan, proof, proven, verified, activation, matrix, raw, masks,
        )
    }

    #[test]
    fn one_request_batches_every_head_into_one_pending() {
        let (plan, proof, proven, verified, _activation, matrix, raw, masks) = run(3);
        assert_eq!(proven, verified);
        let outcome = &verified;
        assert_eq!(plan.heads(), 3);
        assert_eq!(plan.consumed(), 12);
        assert_eq!(proof.head_targets.len(), 3);
        assert!(outcome.coefficients.batch.is_some());
        // One pending row, its weights over the whole consumed region only.
        assert_eq!(outcome.pending.psi, outcome.point);
        assert_eq!(
            outcome.pending.weights,
            stretch_weights(4, 0, &outcome.coefficients.combined().unwrap()).unwrap()
        );
        assert_eq!(outcome.pending.masks.len(), 1);
        assert_eq!(outcome.pending.masks[0].oracle, plan.mask.oracle);
        assert_eq!(outcome.pending.masks[0].slot, plan.mask.shown_slot);
        // The same-matrix row holds against the real matrix and shown entry.
        crate::proof::hidden_member::check_pending(&outcome.pending, &matrix, &|oracle, slot| {
            if oracle == plan.mask.oracle && slot == plan.mask.shown_slot {
                Ok(masks.entries.shown[1])
            } else {
                Err(Error::ShapeMismatch)
            }
        })
        .unwrap();
        // The per-head raw targets were fixed before the batch challenge and
        // combine to the proven target.
        let mut total = Fp3::ZERO;
        let mut power = Fp3::ONE;
        for target in &proof.head_targets {
            total = total.add(&power.mul(target));
            power = power.mul(&outcome.coefficients.batch.unwrap());
        }
        assert_eq!(total, proof.target);
        // The registered raw obligation closes on the real raw rows.
        outcome.raw.check_on(&raw).unwrap();
        let mut changed = raw.clone();
        changed[1][0] = changed[1][0].add(&value(1));
        assert!(outcome.raw.check_on(&changed).is_err());
        // The message round-trips and the decoded message verifies.
        let bytes = proof.encode(&plan).unwrap();
        let decoded = ConsumerRequestProof::decode(&plan, &bytes).unwrap();
        assert_eq!(decoded, proof);
        verify_consumer_request(&mut chain(), &plan, &_activation, &decoded).unwrap();
    }

    #[test]
    fn one_head_takes_the_direct_step() {
        let (plan, proof, proven, verified, _activation, matrix, raw, masks) = run(1);
        assert_eq!(proven, verified);
        assert_eq!(plan.heads(), 1);
        assert!(verified.coefficients.batch.is_none());
        assert_eq!(proof.target, proof.head_targets[0]);
        assert_eq!(proof.head_targets.len(), 1);
        assert_eq!(
            verified.pending.weights,
            stretch_weights(4, 0, &verified.coefficients.combined().unwrap()).unwrap()
        );
        crate::proof::hidden_member::check_pending(&verified.pending, &matrix, &|oracle, slot| {
            if oracle == plan.mask.oracle && slot == plan.mask.shown_slot {
                Ok(masks.entries.shown[1])
            } else {
                Err(Error::ShapeMismatch)
            }
        })
        .unwrap();
        verified.raw.check_on(&raw).unwrap();
    }

    #[test]
    fn a_changed_head_target_or_batch_fails_the_request() {
        let (plan, proof, _proven, _verified, activation, _matrix, _raw, _masks) = run(2);

        let mut changed = proof.clone();
        changed.head_targets[0] = changed.head_targets[0].add(&value(1));
        changed.target = changed.target.add(&value(1));
        assert!(verify_consumer_request(&mut chain(), &plan, &activation, &changed).is_err());

        let mut changed = proof.clone();
        changed.target = changed.target.add(&value(1));
        assert!(verify_consumer_request(&mut chain(), &plan, &activation, &changed).is_err());

        let mut changed = proof.clone();
        changed.point_values[1] = changed.point_values[1].add(&value(1));
        assert!(verify_consumer_request(&mut chain(), &plan, &activation, &changed).is_err());

        let mut changed = proof;
        changed.dot.rounds[0].coefficients[0] = value(7);
        assert!(verify_consumer_request(&mut chain(), &plan, &activation, &changed).is_err());
    }

    #[test]
    fn a_changed_raw_head_or_matrix_fails_the_prover() {
        let plan = plan(2);
        let activation = activation(&plan.shape);
        let matrix = matrix(&plan.shape);
        let raw = raw_heads(&plan, &activation, &matrix);
        let masks = masks(&plan.shape);

        let mut changed = raw.clone();
        changed[1][0] = changed[1][0].add(&value(1));
        assert!(prove_consumer_request(
            &mut chain(),
            &plan,
            &activation,
            &changed,
            &matrix,
            &masks
        )
        .is_err());

        let mut changed = matrix.clone();
        changed[4][0] = changed[4][0].add(&value(1));
        assert!(
            prove_consumer_request(&mut chain(), &plan, &activation, &raw, &changed, &masks)
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

        // A different mask source, request or shift is not the same request.
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
