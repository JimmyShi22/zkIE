//! The projection consumer request's first-stage relation: the fixed
//! statement, the rank-one head combination and the pending obligations.
//!
//! One request consumes one authenticated table through every head slice of a
//! layer. The low-level module below fixes the whole statement of that
//! consumption and turns it into the claims the real request chain's first
//! relation stage must prove:
//!
//! * the statement fixes the request, the table qualification, the mask
//!   objects, the activation and output bindings, every head's source step,
//!   slice and order, and the integer ranges, before any challenge;
//! * the position coefficients and the per-head column coefficients are drawn
//!   from the caller's chain after the statement;
//! * the batch challenge combines the heads: `b = concat_h theta^h b_h`,
//!   `alpha[h,p,j] = a[p] theta^h b_h[j]` and the mask terms use the same
//!   powers, so the combined relation is one rank-one claim on the table's
//!   matrix with one common column point and head-external coefficients zero;
//! * the raw accumulator stays a prover witness and a committed relation
//!   quantity: no raw row, remainder or unmasked reading is returned, sent or
//!   published. The module returns the linear claim coefficients over the
//!   committed raw entries as a pending obligation; the actual rounding
//!   relation, the mask ends and the group check are the request chain's
//!   duty.
//!
//! The chain is the caller's real first-stage chain: this module never opens a
//! chain, never invents a seal and never returns a completion credential. It
//! only absorbs the fixed statement, draws the challenges from that chain and
//! returns the claim plan and its obligations.
//!
//! Integer gate: the raw accumulator is summed over the authenticated
//! contraction of `records` entries with the declared activation magnitude and
//! the authenticated table's `R19` weight range. The gate proves the
//! accumulator and its rounding numerator stay far below half the base field,
//! so the field relation is the integer relation with no modular wrap and the
//! `i128` accumulation semantics is not silently shrunk to an arbitrary
//! `i64`: the witness-side rounding check keeps every value in `i128`.

use zkie_core::artifact::digest_bytes;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::chain::Chain;
use zkie_engine::pcs::request::field::{Fp3, MODULUS};
use zkie_engine::primitives::codec::CanonicalWriter;

use crate::proof::committed_psc::PscTranscriptTags;
use crate::proof::consumer_psc::{
    ConsumerMaskSource, ConsumerObligation, ConsumerPscShape, ConsumerTable, CONSUMER_PSC_VERSION,
};

/// Version of the request statement.
pub const CONSUMER_REQUEST_VERSION: u64 = 1;
/// Transcript tag of the request statement.
const CONSUMER_REQUEST_TAG: u64 = 86;
/// Transcript tags reserved for the request's own product sumcheck. The
/// request chain integration passes them to the masked sumcheck it runs on the
/// caller's stage.
pub const CONSUMER_REQUEST_PSC_TAGS: PscTranscriptTags = PscTranscriptTags {
    version: CONSUMER_PSC_VERSION,
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

/// The shared table stage's fixed parameters of one request: the
/// authenticated table's once and later distances, the once column count, the
/// one declared later batch's column count and the table's hiding tail. This
/// block is the table's later-mode parameter set, not the request's whole
/// reduction budget: the product-sumchecks, the mask objects and the group
/// plan account separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsumerSharedPlan {
    pub m_once: u64,
    pub m_later: u64,
    pub one_time_columns: u64,
    pub later_columns: u64,
    pub declared_batches: u32,
    pub table_tail: u64,
}

impl ConsumerSharedPlan {
    pub fn fault(&self) -> Option<&'static str> {
        if self.m_once == 0 || self.m_later == 0 || self.m_later > self.m_once {
            return Some("the later distance is not bounded by the once distance");
        }
        if self.one_time_columns == 0 || self.later_columns == 0 {
            return Some("the table column counts are empty");
        }
        if self.declared_batches != 1 {
            return Some("one request declares exactly one later batch");
        }
        let total = match self.one_time_columns.checked_add(self.later_columns) {
            Some(total) => total,
            None => return Some("the declared table batch overflows"),
        };
        if total > self.table_tail {
            return Some("the declared table batch does not fit the hiding tail");
        }
        None
    }

    pub fn check(&self) -> Result<()> {
        self.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))
    }
}

/// The authenticated integer ranges the raw gate depends on: the declared
/// activation magnitude of the request and the table's fixed weight range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsumerIntegerRanges {
    /// `max |X|` over the request's activation, from the authenticated trace.
    pub x_abs_bound: u64,
    /// The table's authenticated weight range: the `R19` declaration.
    pub weight_lo: i64,
    pub weight_hi: i64,
}

impl ConsumerIntegerRanges {
    pub fn fault(&self) -> Option<&'static str> {
        if self.x_abs_bound == 0 {
            return Some("the request declares no activation magnitude");
        }
        if self.weight_lo > self.weight_hi || self.weight_lo >= 0 || self.weight_hi < 0 {
            return Some("the authenticated weight range does not straddle zero");
        }
        None
    }

    fn weight_magnitude(&self) -> i128 {
        let low = i128::from(self.weight_lo).unsigned_abs();
        let high = i128::from(self.weight_hi).unsigned_abs();
        i128::try_from(low.max(high)).unwrap_or(i128::MAX)
    }
}

/// The integer no-wrap gate of one request. The raw accumulator is
/// `Y = Σ_k X[k] M[·,k]` over `records` entries; its magnitude is at most
/// `records * x_abs_bound * max(|weight_lo|, |weight_hi|)`, and the rounding
/// numerator adds `2^(shift-1)`. The gate refuses any request whose bound
/// reaches half the base field or the `i128` safety margin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsumerIntegerBound {
    pub records: u64,
    pub ranges: ConsumerIntegerRanges,
    pub shift: u32,
    pub raw_abs_bound: i128,
    pub numerator_abs_bound: i128,
    pub field_half: i128,
    pub i128_margin: i128,
}

impl ConsumerIntegerBound {
    pub fn new(records: u64, ranges: ConsumerIntegerRanges, shift: u32) -> Result<Self> {
        if records < 2 || !records.is_power_of_two() || shift == 0 || shift >= 127 {
            return Err(Error::InvalidConfig);
        }
        ranges
            .fault()
            .map_or(Ok(()), |_| Err(Error::InvalidConfig))?;
        let per_entry = i128::from(ranges.x_abs_bound)
            .checked_mul(ranges.weight_magnitude())
            .ok_or(Error::ResourceLimit)?;
        let raw_abs_bound = per_entry
            .checked_mul(i128::from(records))
            .ok_or(Error::ResourceLimit)?;
        let half = 1i128.checked_shl(shift - 1).ok_or(Error::ResourceLimit)?;
        let numerator_abs_bound = raw_abs_bound
            .checked_add(half)
            .ok_or(Error::ResourceLimit)?;
        let field_half = i128::from(MODULUS) / 2;
        let i128_margin = 1i128 << 126;
        let bound = Self {
            records,
            ranges,
            shift,
            raw_abs_bound,
            numerator_abs_bound,
            field_half,
            i128_margin,
        };
        bound.check()?;
        Ok(bound)
    }

    pub fn check(&self) -> Result<()> {
        if self.numerator_abs_bound >= self.field_half {
            return Err(Error::BoundViolation);
        }
        if self.numerator_abs_bound >= self.i128_margin {
            return Err(Error::BoundViolation);
        }
        Ok(())
    }
}

/// The rounding relation on hidden witness rows, prover side only: every raw
/// accumulator entry must round to the head step's public output entry under
/// the production rule and shift, with the `i128` accumulation semantics.
/// This is the prover's own witness check; the verifier never receives the raw
/// rows or the remainders.
pub fn check_raw_rounding_witness(raw: &[i128], outputs: &[i64], shift: u32) -> Result<()> {
    if raw.len() != outputs.len() || raw.is_empty() {
        return Err(Error::ShapeMismatch);
    }
    for (value, output) in raw.iter().zip(outputs) {
        if crate::proof::relations::rq(*value, shift)? != i128::from(*output) {
            return Err(Error::InvalidProof);
        }
    }
    Ok(())
}

/// The rounding remainder of one accumulator entry under the production rule:
/// `r = Y + 2^(shift-1) - out * 2^shift`, in `[0, 2^shift)` for a valid entry.
pub fn raw_rounding_remainder(value: i128, output: i64, shift: u32) -> Result<i128> {
    if shift == 0 || shift >= 127 {
        return Err(Error::InvalidConfig);
    }
    let half = 1i128.checked_shl(shift - 1).ok_or(Error::ResourceLimit)?;
    let scaled = i128::from(output)
        .checked_mul(1i128.checked_shl(shift).ok_or(Error::ResourceLimit)?)
        .ok_or(Error::BoundViolation)?;
    let remainder = value
        .checked_add(half)
        .and_then(|value| value.checked_sub(scaled))
        .ok_or(Error::BoundViolation)?;
    if remainder < 0 || remainder >= (1i128 << shift) {
        return Err(Error::InvalidProof);
    }
    Ok(remainder)
}

/// The one-request consumer plan: the registered head obligations of a single
/// request, their common table and the fixed integers the claims depend on.
/// Every field is public plan data; the raw rows and the table values are
/// never part of it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerRequestPlan {
    pub request: u64,
    pub table: ConsumerTable,
    pub shift: i64,
    pub shape: ConsumerPscShape,
    pub mask: ConsumerMaskSource,
    pub head_width: u64,
    pub shared: ConsumerSharedPlan,
    pub ranges: ConsumerIntegerRanges,
    /// The digest of the request's actual activation values, bound by the
    /// authenticated trace. The outer statement may carry it instead; this
    /// module never accepts the step number alone.
    pub activation_digest: Digest,
    /// The digest of the head steps' public output rows, in head order.
    pub outputs_digest: Digest,
    pub obligations: Vec<ConsumerObligation>,
}

impl ConsumerRequestPlan {
    /// Registers the request from its ordered head obligations. Every head is
    /// one source step of the same request, the same table and the same
    /// activation step; the heads are exactly `0..J`; their slices are the
    /// adjacent head slices of the consumed region and nothing else.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        request: u64,
        table: ConsumerTable,
        mask: ConsumerMaskSource,
        shared: ConsumerSharedPlan,
        ranges: ConsumerIntegerRanges,
        activation_digest: Digest,
        outputs_digest: Digest,
        obligations: Vec<ConsumerObligation>,
    ) -> Result<Self> {
        shared.check()?;
        ranges
            .fault()
            .map_or(Ok(()), |_| Err(Error::InvalidConfig))?;
        if activation_digest == Digest::ZERO || outputs_digest == Digest::ZERO {
            return Err(Error::InvalidConfig);
        }
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
            shared,
            ranges,
            activation_digest,
            outputs_digest,
            obligations,
        };
        plan.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))?;
        Ok(plan)
    }

    pub fn heads(&self) -> u32 {
        self.obligations.len() as u32
    }

    pub fn consumed(&self) -> u64 {
        self.shape.slice.1 - self.shape.slice.0
    }

    /// The integer no-wrap gate of the request.
    pub fn integer_bound(&self) -> Result<ConsumerIntegerBound> {
        let shift = u32::try_from(self.shift).map_err(|_| Error::InvalidConfig)?;
        ConsumerIntegerBound::new(self.shape.records, self.ranges, shift)
    }

    pub fn fault(&self) -> Option<&'static str> {
        if let Some(fault) = self.shape.fault() {
            return Some(fault);
        }
        if self.obligations.is_empty() || self.obligations.len() > u32::MAX as usize {
            return Some("a consumer request has no head claim");
        }
        let width = match self.obligations[0].shape.slice_len() {
            Ok(width) => width,
            Err(_) => return Some("a consumer request has no valid head width"),
        };
        if self.head_width == 0 || self.head_width != width {
            return Some("a consumer request has no common head width");
        }
        if self.consumed() != u64::from(self.heads()) * self.head_width {
            return Some("a consumer request does not consume exactly its head slices");
        }
        if self.activation_digest == Digest::ZERO || self.outputs_digest == Digest::ZERO {
            return Some("a consumer request has no activation or output binding");
        }
        if self.integer_bound().is_err() {
            return Some("the request's integer no-wrap gate does not hold");
        }
        None
    }

    pub(crate) fn statement_bytes(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(CONSUMER_REQUEST_VERSION);
        writer.put_u64(self.request);
        writer.put_u64(u64::from(self.heads()));
        writer.put_u64(self.head_width);
        writer.put_u64(u64::from(self.shape.positions));
        writer.put_u64(self.shape.places);
        writer.put_u64(self.shape.records);
        writer.put_u64(self.shape.slice.0);
        writer.put_u64(self.shape.slice.1);
        writer.put_u64(self.shift as u64);
        writer.put_u64(self.table.oracle);
        writer.put_digest(&self.table.root);
        writer.put_digest(&self.table.identity);
        writer.put_digest(&self.table.configuration);
        writer.put_u64(self.mask.oracle);
        writer.put_u64(self.mask.hiding_base);
        writer.put_u64(self.mask.shown_slot);
        writer.put_u64(self.shared.m_once);
        writer.put_u64(self.shared.m_later);
        writer.put_u64(self.shared.one_time_columns);
        writer.put_u64(self.shared.later_columns);
        writer.put_u64(u64::from(self.shared.declared_batches));
        writer.put_u64(self.shared.table_tail);
        writer.put_u64(self.ranges.x_abs_bound);
        writer.put_u64(self.ranges.weight_lo as u64);
        writer.put_u64(self.ranges.weight_hi as u64);
        writer.put_digest(&self.activation_digest);
        writer.put_digest(&self.outputs_digest);
        for obligation in &self.obligations {
            writer.put_u64(u64::from(obligation.source_step));
            writer.put_u64(u64::from(obligation.activation_step));
            writer.put_u64(u64::from(obligation.head));
            writer.put_u64(obligation.shape.slice.0);
            writer.put_u64(obligation.shape.slice.1);
        }
        writer.into_bytes()
    }

    /// The digest of the fixed statement. Both sides compute it from the same
    /// plan; it is never taken from the proof.
    pub fn statement_digest(&self) -> Digest {
        digest_bytes(&self.statement_bytes())
    }

    /// Absorbs the fixed statement into the caller's real first-stage chain
    /// before any challenge. The caller owns the chain: this function never
    /// opens one.
    pub fn absorb_statement(&self, chain: &mut Chain) -> Result<()> {
        let statement = self.statement_bytes();
        let digest = digest_bytes(&statement);
        let mut message = chain.message();
        message.push_u64(CONSUMER_REQUEST_VERSION);
        message.push_u64(CONSUMER_REQUEST_TAG);
        push_digest(&mut message, &digest);
        let prefix = message.prefix(zkie_engine::pcs::request::chain::KIND_ALGEBRAIC, 0)?;
        if !chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        Ok(())
    }
}

/// The digest of the request's activation values in their canonical field
/// encoding. The authenticated trace binds this digest to the request; the
/// module never accepts the step number alone.
pub fn consumer_activation_digest(values: &[Fp3]) -> Digest {
    let mut writer = CanonicalWriter::new();
    writer.put_u64(CONSUMER_REQUEST_VERSION);
    writer.put_u64(values.len() as u64);
    for value in values {
        writer.put_fp3(value);
    }
    digest_bytes(&writer.into_bytes())
}

/// The digest of the head steps' public output rows, in head order, with the
/// plan's per-head width and positions.
pub fn consumer_outputs_digest(plan: &ConsumerRequestPlan, outputs: &[Vec<i64>]) -> Result<Digest> {
    if outputs.len() != plan.obligations.len() {
        return Err(Error::ShapeMismatch);
    }
    let count = (plan.shape.positions as usize)
        .checked_mul(usize::try_from(plan.head_width).map_err(|_| Error::ResourceLimit)?)
        .ok_or(Error::ResourceLimit)?;
    if outputs.iter().any(|head| head.len() != count) {
        return Err(Error::ShapeMismatch);
    }
    let mut writer = CanonicalWriter::new();
    writer.put_u64(CONSUMER_REQUEST_VERSION);
    writer.put_u64(outputs.len() as u64);
    for head in outputs {
        for value in head {
            writer.put_u64(*value as u64);
        }
    }
    Ok(digest_bytes(&writer.into_bytes()))
}

/// The transcript coefficients of one request: the common position
/// coefficients, one column vector per head and, after the fixed statement and
/// the coefficients, the batch challenge.
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

/// The batch accounting of the head combination. The combined claim is one
/// polynomial in the batch challenge of degree `heads - 1`; a failing head
/// that survives the combination contributes one such root. The count is the
/// real registered head number, never the table's later-mode column block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsumerBatchAccounting {
    pub heads: u32,
    pub theta_degree: u32,
}

impl ConsumerBatchAccounting {
    pub fn new(heads: u32) -> Result<Self> {
        if heads == 0 {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            heads,
            theta_degree: heads.saturating_sub(1),
        })
    }
}

/// The fixed-order raw claim of the request: the logical accumulator entries
/// of every head, in head order, then position order, then slice offset. The
/// group integration maps these logical entries to the committed raw message
/// slots; the order itself is plan data and never chosen after a challenge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerRawClaimOrder {
    pub heads: u32,
    pub positions: u32,
    pub width: u64,
}

impl ConsumerRawClaimOrder {
    pub fn entries(&self) -> Result<usize> {
        (self.heads as usize)
            .checked_mul(self.positions as usize)
            .and_then(|value| value.checked_mul(self.width as usize))
            .ok_or(Error::ResourceLimit)
    }
}

/// The first-stage claim plan and its pending obligations of one request.
/// Both sides derive it from the same plan and the same caller chain. The raw
/// side is a linear claim over the committed raw entries; its target is not
/// sent and not computed from public rows. The table side is the combined
/// column coefficient vector with one common point. Every completion stays a
/// pending obligation of the request chain's group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerRequestClaim {
    pub statement: Digest,
    pub order: ConsumerRawClaimOrder,
    pub theta: Option<Fp3>,
    pub powers: Vec<Fp3>,
    pub position: Vec<Fp3>,
    pub heads: Vec<Vec<Fp3>>,
    /// `b = concat_h theta^h b_h` over the consumed region.
    pub combined: Vec<Fp3>,
    /// `alpha[h, p, j] = a[p] theta^h b_h[j]` in the fixed logical order.
    pub alpha: Vec<Fp3>,
    pub bound: ConsumerIntegerBound,
    pub accounting: ConsumerBatchAccounting,
}

impl ConsumerRequestClaim {
    /// The alpha coefficients in per-head, per-position form, in the fixed
    /// logical order.
    pub fn alpha_rows(&self) -> Result<Vec<Vec<Fp3>>> {
        let per_head = self.order.entries()? / self.order.heads as usize;
        if self.alpha.len() != self.order.entries()? {
            return Err(Error::ShapeMismatch);
        }
        let mut heads = Vec::with_capacity(self.order.heads as usize);
        for head in 0..self.order.heads as usize {
            heads.push(self.alpha[head * per_head..(head + 1) * per_head].to_vec());
        }
        Ok(heads)
    }

    /// The raw claim value on witness rows in the fixed logical order. This is
    /// a prover and integration helper; it is never a verification input.
    pub fn claim_value_on(&self, raw: &[Vec<i128>]) -> Result<Fp3> {
        if raw.len() != self.order.heads as usize {
            return Err(Error::ShapeMismatch);
        }
        let mut total = Fp3::ZERO;
        let mut at = 0usize;
        for head in raw {
            if head.len() != self.order.positions as usize * self.order.width as usize {
                return Err(Error::ShapeMismatch);
            }
            for value in head {
                let coefficient = *self.alpha.get(at).ok_or(Error::ShapeMismatch)?;
                total = total.add(&coefficient.mul(&signed_fp3(*value)?));
                at += 1;
            }
        }
        if at != self.alpha.len() {
            return Err(Error::ShapeMismatch);
        }
        Ok(total)
    }
}

pub fn signed_fp3(value: i128) -> Result<Fp3> {
    let magnitude = u64::try_from(value.unsigned_abs()).map_err(|_| Error::BoundViolation)?;
    let base = zkie_engine::pcs::request::field::fp_from_u64(magnitude);
    let positive = Fp3::from_base(base);
    Ok(if value < 0 {
        Fp3::ZERO.sub(&positive)
    } else {
        positive
    })
}

/// Derives the request's first-stage claim from the caller's real chain. The
/// statement is absorbed first; the coefficients follow; the batch challenge
/// is drawn only after every object, output binding, input, slice and order is
/// fixed. Both the prover and the verifier call this and obtain the same
/// claim; the module returns no proof and no completion credential.
pub fn consumer_request_claim(
    chain: &mut Chain,
    plan: &ConsumerRequestPlan,
) -> Result<ConsumerRequestClaim> {
    plan.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))?;
    plan.absorb_statement(chain)?;
    let mut position = Vec::with_capacity(plan.shape.positions as usize);
    for _ in 0..plan.shape.positions {
        position.push(chain.draw_e()?);
    }
    let width = usize::try_from(plan.head_width).map_err(|_| Error::ResourceLimit)?;
    let mut heads = Vec::with_capacity(plan.obligations.len());
    for _ in 0..plan.obligations.len() {
        let mut values = Vec::with_capacity(width);
        for _ in 0..width {
            values.push(chain.draw_e()?);
        }
        heads.push(values);
    }
    let mut coefficients = ConsumerRequestCoefficients {
        position: position.clone(),
        heads: heads.clone(),
        batch: None,
    };
    if plan.obligations.len() > 1 {
        coefficients.batch = Some(chain.draw_e()?);
    }
    let combined = coefficients.combined()?;
    let mut powers = Vec::with_capacity(heads.len());
    let mut power = Fp3::ONE;
    for _ in &heads {
        powers.push(power);
        power = match coefficients.batch {
            Some(theta) => power.mul(&theta),
            None => Fp3::ONE,
        };
    }
    let mut alpha = Vec::with_capacity(heads.len() * position.len() * width);
    for (head, power) in heads.iter().zip(&powers) {
        for coefficient in &position {
            for value in head {
                alpha.push(coefficient.mul(power).mul(value));
            }
        }
    }
    let bound = plan.integer_bound()?;
    let accounting = ConsumerBatchAccounting::new(plan.heads())?;
    Ok(ConsumerRequestClaim {
        statement: plan.statement_digest(),
        order: ConsumerRawClaimOrder {
            heads: plan.heads(),
            positions: plan.shape.positions,
            width: plan.head_width,
        },
        theta: coefficients.batch,
        powers,
        position,
        heads,
        combined,
        alpha,
        bound,
        accounting,
    })
}

/// The combined instance's obligation of the request: the whole consumed
/// region as one consumer shape, head zero, the request's table and mask. The
/// first stage uses it with the request claim's fixed coefficients and the
/// plan member's target.
pub fn consumer_combined_obligation(plan: &ConsumerRequestPlan) -> Result<ConsumerObligation> {
    plan.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))?;
    let obligation = ConsumerObligation {
        request: plan.request,
        source_step: plan.obligations[0].source_step,
        activation_step: plan.obligations[0].activation_step,
        head: 0,
        shape: plan.shape,
        shift: plan.shift,
        table: plan.table.clone(),
        mask: plan.mask,
    };
    obligation.check()?;
    Ok(obligation)
}

/// The combined raw field rows of the request from the per-head committed
/// value rows: position-major over the whole consumed region, real entries
/// only. Padding columns of a head are the committed message's padding and
/// never enter the claim.
pub fn consumer_combined_raw(plan: &ConsumerRequestPlan, raw: &[Vec<Fp3>]) -> Result<Vec<Fp3>> {
    let positions = plan.shape.positions as usize;
    let width = usize::try_from(plan.head_width).map_err(|_| Error::ResourceLimit)?;
    let hidden = usize::try_from(plan.shape.slice_len()?).map_err(|_| Error::ResourceLimit)?;
    if raw.len() != plan.obligations.len() || raw.iter().any(|head| head.len() < positions * width)
    {
        return Err(Error::ShapeMismatch);
    }
    let mut combined = vec![Fp3::ZERO; positions * hidden];
    for (head, rows) in raw.iter().enumerate() {
        for position in 0..positions {
            for index in 0..width {
                combined[position * hidden + head * width + index] = rows[position * width + index];
            }
        }
    }
    Ok(combined)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::field::fp_from_u64;

    use crate::proof::consumer_psc::ConsumerMaskSource;

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
            shared(),
            ranges(),
            Digest([0x41; 32]),
            Digest([0x42; 32]),
            obligations,
        )
        .unwrap()
    }

    fn chain() -> Chain {
        Chain::begin(
            96,
            &Digest::ZERO,
            b"consumer request stage",
            b"fixture",
            None,
        )
        .unwrap()
    }

    #[test]
    fn the_statement_fixes_the_objects_before_any_draw() {
        let plan = plan(2);
        let first = consumer_request_claim(&mut chain(), &plan).unwrap();
        let second = consumer_request_claim(&mut chain(), &plan).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.statement, plan.statement_digest());
        assert_eq!(first.order.heads, 2);
        assert_eq!(first.order.positions, 3);
        assert_eq!(first.order.width, 4);
        assert_eq!(first.order.entries().unwrap(), 24);
        assert_eq!(first.alpha.len(), 24);
        // Another statement is another claim.
        let mut other = plan.clone();
        other.table.root = Digest([0x77; 32]);
        assert_ne!(
            consumer_request_claim(&mut chain(), &other)
                .unwrap()
                .statement,
            first.statement
        );
        let mut other = plan.clone();
        other.activation_digest = Digest([0x43; 32]);
        assert_ne!(consumer_request_claim(&mut chain(), &other).unwrap(), first);
        let mut other = plan.clone();
        other.outputs_digest = Digest([0x44; 32]);
        assert_ne!(consumer_request_claim(&mut chain(), &other).unwrap(), first);
        let mut other = plan;
        other.obligations[1].source_step += 1;
        assert_ne!(consumer_request_claim(&mut chain(), &other).unwrap(), first);
    }

    #[test]
    fn the_heads_combine_with_one_power_series_and_one_common_point() {
        let plan = plan(3);
        let claim = consumer_request_claim(&mut chain(), &plan).unwrap();
        let theta = claim.theta.unwrap();
        assert_eq!(claim.powers.len(), 3);
        assert_eq!(claim.powers[0], Fp3::ONE);
        assert_eq!(claim.powers[1], theta);
        assert_eq!(claim.powers[2], theta.mul(&theta));
        assert_eq!(claim.combined.len(), 12);
        let width = plan.head_width as usize;
        for head in 0..3usize {
            for index in 0..width {
                assert_eq!(
                    claim.combined[head * width + index],
                    claim.powers[head].mul(&claim.heads[head][index])
                );
            }
        }
        // alpha is the linear combination of the per-head a tensor with the
        // same powers; nothing outside the registered heads is claimed.
        let rows = claim.alpha_rows().unwrap();
        assert_eq!(rows.len(), 3);
        for (head, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), 12);
            for position in 0..3usize {
                for index in 0..width {
                    assert_eq!(
                        row[position * width + index],
                        claim.position[position]
                            .mul(&claim.powers[head])
                            .mul(&claim.heads[head][index])
                    );
                }
            }
        }
        assert_eq!(claim.accounting.heads, 3);
        assert_eq!(claim.accounting.theta_degree, 2);
    }

    #[test]
    fn one_head_takes_the_direct_claim() {
        let plan = plan(1);
        let claim = consumer_request_claim(&mut chain(), &plan).unwrap();
        assert!(claim.theta.is_none());
        assert_eq!(claim.powers, vec![Fp3::ONE]);
        assert_eq!(claim.accounting.theta_degree, 0);
        assert_eq!(claim.alpha.len(), 12);
    }

    #[test]
    fn the_integer_gate_refuses_a_wrap() {
        // The honest ranges stay far below half the base field.
        let bound = ConsumerIntegerBound::new(8, ranges(), 16).unwrap();
        assert!(bound.raw_abs_bound > 0);
        assert!(bound.numerator_abs_bound < bound.field_half);
        assert!(bound.numerator_abs_bound < bound.i128_margin);
        // A huge activation magnitude or contraction cannot be bounded.
        let huge = ConsumerIntegerRanges {
            x_abs_bound: 1 << 60,
            ..ranges()
        };
        assert!(ConsumerIntegerBound::new(1 << 17, huge, 16).is_err());
        // A range that does not straddle zero is not a weight range.
        let bad = ConsumerIntegerRanges {
            x_abs_bound: 1,
            weight_lo: 1,
            weight_hi: 4,
        };
        assert!(ConsumerIntegerBound::new(8, bad, 16).is_err());
        // The plan refuses a request whose gate fails.
        let mut broken = plan(1);
        broken.ranges.x_abs_bound = 1 << 60;
        assert!(broken.fault().is_some());
    }

    #[test]
    fn the_rounding_relation_keeps_the_i128_semantics() {
        // The witness check and the remainder stay in i128; a value far above
        // the i64 range still rounds exactly and its remainder is in range.
        let shift = 16u32;
        let value = (1i128 << 70) + 12345;
        let rounded = crate::proof::relations::rq(value, shift).unwrap();
        let output = i64::try_from(rounded).unwrap();
        assert!(check_raw_rounding_witness(&[value], &[output], shift).is_ok());
        let remainder = raw_rounding_remainder(value, output, shift).unwrap();
        assert!(remainder >= 0 && remainder < (1i128 << shift));
        // A requantized output substituted for the raw value fails.
        assert!(check_raw_rounding_witness(&[i128::from(output)], &[output], shift).is_err());
        // The gate refuses an accumulator magnitude that would reach half the
        // base field, so the field relation cannot hide a modular wrap.
        let ranges = ConsumerIntegerRanges {
            x_abs_bound: 1 << 60,
            weight_lo: -(1 << 18),
            weight_hi: (1 << 18) - 1,
        };
        assert!(ConsumerIntegerBound::new(1 << 17, ranges, shift).is_err());
    }

    #[test]
    fn the_claim_value_helper_uses_the_alpha_order() {
        let plan = plan(2);
        let claim = consumer_request_claim(&mut chain(), &plan).unwrap();
        let width = plan.head_width as usize;
        let positions = plan.shape.positions as usize;
        let raw: Vec<Vec<i128>> = (0..2)
            .map(|_| {
                (0..positions * width)
                    .map(|index| (index as i128 % 7) - 3)
                    .collect()
            })
            .collect();
        let value = claim.claim_value_on(&raw).unwrap();
        let mut expected = Fp3::ZERO;
        let mut at = 0usize;
        for head in &raw {
            for entry in head {
                expected = expected.add(&claim.alpha[at].mul(&signed_fp3(*entry).unwrap()));
                at += 1;
            }
        }
        assert_eq!(value, expected);
    }

    #[test]
    fn the_request_registration_binds_the_head_sources_and_coverage() {
        // A gap or an overlap between head slices is not a request.
        let obligations = vec![obligation(0, 20), obligation(2, 22)];
        assert!(ConsumerRequestPlan::new(
            7,
            table("weights/L0_k_w_i32.bin", [0x11; 32]),
            mask(),
            shared(),
            ranges(),
            Digest([0x41; 32]),
            Digest([0x42; 32]),
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
            shared(),
            ranges(),
            Digest([0x41; 32]),
            Digest([0x42; 32]),
            vec![obligation(0, 20), other]
        )
        .is_err());

        // A zero activation or output digest is not a binding.
        assert!(ConsumerRequestPlan::new(
            7,
            table("weights/L0_k_w_i32.bin", [0x11; 32]),
            mask(),
            shared(),
            ranges(),
            Digest::ZERO,
            Digest([0x42; 32]),
            vec![obligation(0, 20), obligation(1, 21)]
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
            ConsumerSharedPlan {
                m_once: 284,
                m_later: 284,
                one_time_columns: 52,
                later_columns: 52,
                declared_batches: 1,
                table_tail: 128,
            },
            ConsumerIntegerRanges {
                x_abs_bound: 1 << 20,
                weight_lo: -(1 << 18),
                weight_hi: (1 << 18) - 1,
            },
            Digest([0x51; 32]),
            Digest([0x52; 32]),
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
        assert!(real.integer_bound().is_ok());
    }
}
