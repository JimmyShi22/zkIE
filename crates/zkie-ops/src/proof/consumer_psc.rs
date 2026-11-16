//! The projection consumer's product-sum reduction over the authenticated
//! table matrix.
//!
//! A projection consumer reads one slice of a weight matrix `A`, whose stored
//! records are the matrix columns of the encoded table matrix
//! `M[place, record] = A[record, place]`. The operator's raw accumulator is
//! `Y[p, n] = Σ_k X[p, k] M[n, k]`, where `p` ranges over the positions, `k`
//! over the stored records (the contraction index) and `n` over the stored
//! places (the output columns). The consumer relation reduces the real output
//! rows against the matrix:
//!
//! ```text
//! Σ_p Σ_n a[p] b[n] Y[p, n]
//!     = Σ_k (Σ_p a[p] X[p, k]) · (Σ_n b[n] M[n, k])
//! ```
//!
//! where `a` and `b` are the position and slice coefficients drawn from the
//! chain after the obligation is fixed, `X` is the public activation, and `Y`
//! is the raw accumulator of the projection, never its requantized output.
//! The right side is one masked product sumcheck over the contraction index.
//! Its first map `U[k] = Σ_p a[p] X[p, k]` is public: the verifier recomputes
//! it from its own activation rows and checks the first masked end value. Its
//! second map `V[k] = Σ_n b[n] M[n, k]` stays with the prover; the sumcheck
//! fixes the matrix's column variable at the challenge point, which leaves
//! one pending row on the same matrix
//!
//! ```text
//! <b ⊗ eq(point), M> + beta · shown = end1.value
//! ```
//!
//! The pending row's weights cover exactly the consumed place slice of `b`;
//! its column point is the sumcheck point, and the one-time shown mask stays a
//! mask term of the row. The pending row enters the request's member and
//! batch merge; this module never states that the matrix values are correct.
//!
//! The left side of the identity is a linear claim on the raw output rows of
//! the source step. The prover computes its target from the raw witness, the
//! target is absorbed into the transcript before the product-sumcheck
//! challenges, and the outcome carries the registered obligation with its
//! source step and the exact coefficient vectors. This module does not close
//! that claim: a verified raw relation of the source step must derive the same
//! target from the raw accumulator and its integer rounding bounds.
//!
//! The public map is never masked. The verifier checks the first end value
//! against its own reduction; a proof that masks the public map fails that
//! check. The private map is masked by one committed one-time entry, and the
//! pending row carries the entry as a mask term, so the row check consumes it
//! instead of trusting the prover's number.
//!
//! The target is a caller-supplied relation value: this primitive proves the
//! product sum equals it, never that it is the raw accumulator. A caller that
//! sources the target from a public raw reading re-introduces exactly the
//! exposure the asset-consume round removes; the request integration derives
//! the target from a committed raw claim and lets the group check it.

use zkie_core::common::codec::DecodeLimits;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::chain::{Chain, ChainMessage};
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};

use crate::op::OpCode;
use crate::proof::committed_psc::{
    get_psc_transcript, put_psc_transcript, PositionPscChallenges, PscTranscriptTags,
};
use crate::proof::hidden_member::{stretch_weights, MaskTerm, PendingRow};
use crate::proof::plan::{PlanConfigValue, PlanLayout, PlanSource, PlanStep};
use crate::proof::projection_claim::{activation_shown, consumer_reduction};
use crate::proof::psc::{
    check_product_claims, hp_entries, multilinear_eval, prove_dot_sumcheck, psc_variables,
    verify_psc_transcript, DotClaim, DotMaskEntries, DotOutput, DotProof, SHOWN_MAPS,
};

/// Version of the consumer proof message and of its transcript messages.
pub const CONSUMER_PSC_VERSION: u64 = 1;
/// Transcript tag of the obligation message.
const CONSUMER_OBLIGATION_TAG: u64 = 80;
/// Transcript tag of the raw target message.
const CONSUMER_TARGET_TAG: u64 = 81;
/// Transcript tags of the product-sumcheck. The claim word is reserved: the
/// plain product sumcheck carries no claim masks.
pub const CONSUMER_PSC_TAGS: PscTranscriptTags = PscTranscriptTags {
    version: CONSUMER_PSC_VERSION,
    hiding_sum: 82,
    round: 83,
    end: 84,
    claim: 85,
};

/// The trusted geometry of one projection consumer.
///
/// The matrix rows are the stored places (the output columns), the matrix
/// columns are the stored records (the contraction index), and the consumed
/// slice is a range of places.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsumerPscShape {
    /// Activation and output positions.
    pub positions: u32,
    /// Stored places: the matrix's row count.
    pub places: u64,
    /// Stored records: the matrix's column count and the contraction length.
    pub records: u64,
    /// The consumed place columns `[start, end)`.
    pub slice: (u64, u64),
}

impl ConsumerPscShape {
    /// The consumed slice length in places.
    pub fn slice_len(&self) -> Result<u64> {
        let (start, end) = self.slice;
        if end <= start {
            return Err(Error::InvalidConfig);
        }
        Ok(end - start)
    }

    /// The row-cube size of the matrix, in bits.
    pub fn table_row_bits(&self) -> Result<u32> {
        if self.places < 2 || !self.places.is_power_of_two() {
            return Err(Error::InvalidConfig);
        }
        Ok(self.places.trailing_zeros())
    }

    /// The number of product-sumcheck `X` variables: the bits of the
    /// contraction index.
    pub fn variables(&self) -> Result<usize> {
        if self.records < 2 || !self.records.is_power_of_two() {
            return Err(Error::InvalidConfig);
        }
        Ok(self.records.trailing_zeros() as usize)
    }

    /// The number of one-time hiding entries of one instance.
    pub fn hiding_entries(&self) -> Result<usize> {
        hp_entries(&psc_variables(self.variables()?)?)
    }

    pub fn fault(&self) -> Option<&'static str> {
        if self.positions == 0 {
            return Some("a consumer has no positions");
        }
        if self.places < 2 || !self.places.is_power_of_two() {
            return Some("the consumer matrix row count is not a power-of-two cube");
        }
        if self.records < 2 || !self.records.is_power_of_two() {
            return Some("the consumer contraction length is not a power-of-two cube");
        }
        let (start, end) = self.slice;
        if end <= start || end > self.places {
            return Some("the consumed place slice is not a non-empty range inside the places");
        }
        None
    }

    pub fn check(&self) -> Result<()> {
        self.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))
    }
}

/// The public table identity a consumer is registered against. The path is the
/// locator checked at registration; the digests are what the transcript and
/// the request statement bind. The root is the committed table root of the
/// authentication, never a prover-supplied value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerTable {
    pub path: String,
    pub oracle: u64,
    pub root: Digest,
    pub identity: Digest,
    pub configuration: Digest,
}

/// The index-mask message the one-time mask entries of one consumer live in:
/// the hiding entries at `hiding_base..hiding_base + hiding_entries`, and the
/// shown-map mask entry at `shown_slot`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsumerMaskSource {
    pub oracle: u64,
    pub hiding_base: u64,
    pub shown_slot: u64,
}

/// The registered consumer obligation of one head slice.
///
/// The obligation fixes the source step, the activation step, the request, the
/// geometry, the table identity and the mask slots before any challenge is
/// drawn. A plan step that does not consume exactly the declared slice of the
/// declared table for the declared head is not a source of this obligation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerObligation {
    pub request: u64,
    pub source_step: u32,
    pub activation_step: u32,
    pub head: u32,
    pub shape: ConsumerPscShape,
    /// The projection's rounding shift of the raw accumulator.
    pub shift: i64,
    pub table: ConsumerTable,
    pub mask: ConsumerMaskSource,
}

impl ConsumerObligation {
    /// Registers the consumer obligation from the real projection step of the
    /// plan. The step must be a projection over the activation step's whole
    /// first output and the table's place slice of the head; the slice, the
    /// contraction length and the position count all come from the step and
    /// its declared layout, never from proof data.
    pub fn from_projection_step(
        request: u64,
        source_step: u32,
        activation_step: u32,
        head: u32,
        step: &PlanStep,
        table: ConsumerTable,
        mask: ConsumerMaskSource,
    ) -> Result<Self> {
        if step.opcode != OpCode::Projection || step.config.len() != 5 || step.inputs.len() != 2 {
            return Err(Error::InvalidConfig);
        }
        let mut literals = [0i64; 5];
        for (slot, value) in literals.iter_mut().zip(&step.config) {
            match value {
                PlanConfigValue::Literal(value) => *slot = *value,
                PlanConfigValue::AssetI64 { .. } => return Err(Error::InvalidConfig),
            }
        }
        let positions = u32::try_from(literals[0]).map_err(|_| Error::InvalidConfig)?;
        let head_width = u64::try_from(literals[1]).map_err(|_| Error::InvalidConfig)?;
        let records = u64::try_from(literals[2]).map_err(|_| Error::InvalidConfig)?;
        let shift = literals[3];
        if literals[4] != 0 || shift <= 0 {
            return Err(Error::InvalidConfig);
        }
        match &step.inputs[0] {
            PlanSource::Prev {
                step: at,
                output,
                layout,
            } if *at == activation_step as usize && *output == 0 && *layout == PlanLayout::Flat => {
            }
            _ => return Err(Error::InvalidConfig),
        }
        let (path, layout) = match &step.inputs[1] {
            PlanSource::Asset { path, layout } | PlanSource::ConsumerAsset { path, layout } => {
                (path, layout)
            }
            _ => return Err(Error::InvalidConfig),
        };
        let PlanLayout::Columns {
            row_width,
            start,
            count,
        } = layout
        else {
            return Err(Error::InvalidConfig);
        };
        let places = u64::try_from(*row_width).map_err(|_| Error::InvalidConfig)?;
        let start = u64::try_from(*start).map_err(|_| Error::InvalidConfig)?;
        let count = u64::try_from(*count).map_err(|_| Error::InvalidConfig)?;
        if path != &table.path || count != head_width || head_width == 0 {
            return Err(Error::InvalidConfig);
        }
        if start % head_width != 0 || start / head_width != u64::from(head) {
            return Err(Error::InvalidConfig);
        }
        let end = start.checked_add(count).ok_or(Error::ResourceLimit)?;
        if end > places {
            return Err(Error::InvalidConfig);
        }
        let obligation = Self {
            request,
            source_step,
            activation_step,
            head,
            shape: ConsumerPscShape {
                positions,
                places,
                records,
                slice: (start, end),
            },
            shift,
            table,
            mask,
        };
        obligation.check()?;
        Ok(obligation)
    }

    pub fn fault(&self) -> Option<&'static str> {
        if let Some(fault) = self.shape.fault() {
            return Some(fault);
        }
        if self.source_step == self.activation_step {
            return Some("a consumer source step is its own activation step");
        }
        if self.shift <= 0 || self.shift > 62 {
            return Some("the consumer projection shift is outside the raw accumulator range");
        }
        let (start, end) = self.shape.slice;
        let width = end - start;
        if u64::from(self.head)
            .checked_mul(width)
            .map_or(true, |expected| expected != start)
        {
            return Some("the consumer head does not start at its own place slice");
        }
        if self.table.root == Digest::ZERO {
            return Some("the consumer table has no committed root");
        }
        let hiding = match self.shape.hiding_entries() {
            Ok(hiding) => hiding,
            Err(_) => return Some("the consumer mask entries are not bounded by the geometry"),
        };
        let hiding_end = match self.mask.hiding_base.checked_add(hiding as u64) {
            Some(end) => end,
            None => return Some("the consumer mask slot range overflows"),
        };
        if self.mask.shown_slot >= self.mask.hiding_base && self.mask.shown_slot < hiding_end {
            return Some("the consumer shown mask slot aliases its hiding entries");
        }
        None
    }

    pub fn check(&self) -> Result<()> {
        self.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))
    }

    fn absorb(&self, chain: &mut Chain) -> Result<()> {
        let mut message = chain.message();
        message.push_u64(CONSUMER_PSC_VERSION);
        message.push_u64(CONSUMER_OBLIGATION_TAG);
        message.push_u64(u64::from(self.head));
        message.push_u64(self.request);
        message.push_u64(u64::from(self.source_step));
        message.push_u64(u64::from(self.activation_step));
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
        let prefix = message.prefix(zkie_engine::pcs::request::chain::KIND_ALGEBRAIC, 0)?;
        if !chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        Ok(())
    }
}

/// The prover's one-time mask entries of one consumer instance: the hiding
/// coefficients in [`crate::proof::psc::hp_place`] order and the two shown
/// entries. The first shown entry is fixed to zero in the honest prover: the
/// first map is public and is checked directly, never masked. The second shown
/// entry is private and stays a mask term of the pending row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerPscMasks {
    pub entries: DotMaskEntries,
}

impl ConsumerPscMasks {
    /// The mask entries of the honest prover: the public map unmasked and the
    /// private map masked by `shown`.
    pub fn new(hiding: Vec<Fp3>, shown: Fp3) -> Self {
        Self {
            entries: DotMaskEntries {
                hiding,
                shown: [Fp3::ZERO, shown],
            },
        }
    }

    pub fn check(&self, shape: &ConsumerPscShape) -> Result<()> {
        if self.entries.hiding.len() != shape.hiding_entries()? {
            return Err(Error::ShapeMismatch);
        }
        if !self.entries.shown[0].is_zero() {
            return Err(Error::InvalidConfig);
        }
        if self.entries.shown[1].is_zero() {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }
}

/// The consumer proof message: the raw target and the product-sumcheck
/// transcript of one instance. The masked point values of the two shown maps
/// are sent beside the transcript and must equal its end values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerPscProof {
    /// `Σ_p Σ_n a[p] b[n] Y[p, n]`: the left side of the reduction identity.
    pub target: Fp3,
    pub dot: DotProof,
    pub point_values: [Fp3; SHOWN_MAPS],
}

impl ConsumerPscProof {
    pub fn encode(&self, shape: &ConsumerPscShape) -> Result<Vec<u8>> {
        shape.check()?;
        self.check(shape)?;
        let mut writer = CanonicalWriter::new();
        writer.put_fp3(&self.target);
        put_psc_transcript(
            &mut writer,
            CONSUMER_PSC_VERSION,
            std::slice::from_ref(&self.dot),
            std::slice::from_ref(&self.point_values),
        )?;
        Ok(writer.into_bytes())
    }

    pub fn decode(shape: &ConsumerPscShape, bytes: &[u8]) -> Result<Self> {
        shape.check()?;
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        let target = reader.get_fp3()?;
        let (mut psc, point_values) = get_psc_transcript::<SHOWN_MAPS>(
            &mut reader,
            CONSUMER_PSC_VERSION,
            1,
            shape.variables()?,
        )?;
        reader.finish()?;
        let dot = psc.pop().ok_or(Error::InvalidEncoding)?;
        if !psc.is_empty() {
            return Err(Error::InvalidEncoding);
        }
        let proof = Self {
            target,
            dot,
            point_values: point_values
                .into_iter()
                .next()
                .ok_or(Error::InvalidEncoding)?,
        };
        proof.check(shape)?;
        Ok(proof)
    }

    pub fn check(&self, shape: &ConsumerPscShape) -> Result<()> {
        shape.check()?;
        let variables = shape.variables()?;
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

/// One linear row over the index-mask message's hiding entries:
/// `Σ_i coefficients[i] · entry(hiding_base + i) = target`. The row is the
/// `U` row of the product sumcheck and is checked by the member merge, not
/// here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerMaskRow {
    pub oracle: u64,
    pub base_slot: u64,
    pub coefficients: Vec<Fp3>,
    pub target: Fp3,
}

impl ConsumerMaskRow {
    pub fn units(&self) -> Result<Vec<(u64, u64, Fp3)>> {
        let mut units = Vec::with_capacity(self.coefficients.len());
        for (index, coefficient) in self.coefficients.iter().enumerate() {
            let slot = self
                .base_slot
                .checked_add(index as u64)
                .ok_or(Error::ResourceLimit)?;
            units.push((self.oracle, slot, *coefficient));
        }
        Ok(units)
    }
}

/// The registered left-hand obligation of one consumer reduction: the raw
/// output rows of the source step, reduced by the public coefficient vectors,
/// must equal `target`. The target is fixed in the chain before the
/// product-sumcheck challenges. The obligation is not closed here; the raw
/// accumulator relation of the source step must derive the same target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerRawObligation {
    pub request: u64,
    pub source_step: u32,
    pub activation_step: u32,
    pub positions: u32,
    pub slice: (u64, u64),
    pub target: Fp3,
    pub position_coefficients: Vec<Fp3>,
    pub column_coefficients: Vec<Fp3>,
}

impl ConsumerRawObligation {
    /// The left side on the real raw rows of the source step:
    /// `Σ_p a[p] · Σ_n b[n] Y[p, n]`, with the rows stored slice-local.
    pub fn value_on(&self, raw: &[Fp3]) -> Result<Fp3> {
        crate::proof::projection_claim::raw_target(
            self.positions as usize,
            self.column_coefficients.len(),
            raw,
            &self.position_coefficients,
            &self.column_coefficients,
        )
    }

    /// Checks that the registered target is the raw rows' reduction. This is
    /// the closure the raw relation of the source step must supply; the
    /// consumer proof fixes the target in its transcript but does not close
    /// this claim by itself.
    pub fn check_on(&self, raw: &[Fp3]) -> Result<()> {
        if self.value_on(raw)? != self.target {
            return Err(Error::InvalidProof);
        }
        Ok(())
    }
}

/// The checked outcome of one consumer instance: the sumcheck point, the
/// same-matrix pending row, the mask row and the registered raw obligation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerPscOutcome {
    pub point: Vec<Fp3>,
    pub pending: PendingRow,
    pub mask_row: ConsumerMaskRow,
    pub raw: ConsumerRawObligation,
}

/// The public coefficients of one instance. Both sides draw them from the
/// chain after the obligation and before the raw target, so no coefficient
/// depends on the output under test.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ConsumerCoefficients {
    position: Vec<Fp3>,
    column: Vec<Fp3>,
}

fn draw_coefficients(chain: &mut Chain, shape: &ConsumerPscShape) -> Result<ConsumerCoefficients> {
    let slice_len = usize::try_from(shape.slice_len()?).map_err(|_| Error::ResourceLimit)?;
    let mut position = Vec::with_capacity(shape.positions as usize);
    for _ in 0..shape.positions {
        position.push(chain.draw_e()?);
    }
    let mut column = Vec::with_capacity(slice_len);
    for _ in 0..slice_len {
        column.push(chain.draw_e()?);
    }
    Ok(ConsumerCoefficients { position, column })
}

fn push_digest(message: &mut ChainMessage, digest: &Digest) {
    for word in digest.0.chunks_exact(8) {
        message.push_u64(u64::from_le_bytes(
            word.try_into().expect("eight-byte word"),
        ));
    }
}

fn absorb_target(chain: &mut Chain, head: u32, target: Fp3) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(CONSUMER_PSC_VERSION);
    message.push_u64(CONSUMER_TARGET_TAG);
    message.push_u64(u64::from(head));
    message.push_fp3(&target);
    let prefix = message.prefix(zkie_engine::pcs::request::chain::KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn outcome(
    obligation: &ConsumerObligation,
    coefficients: &ConsumerCoefficients,
    target: Fp3,
    output: &DotOutput,
) -> Result<ConsumerPscOutcome> {
    let shape = &obligation.shape;
    let variables = shape.variables()?;
    if output.point.len() != variables
        || output.hiding_form.len() != shape.hiding_entries()?
        || output.ends.iter().any(|end| end.point != output.point)
    {
        return Err(Error::InvalidProof);
    }
    let pending = PendingRow {
        weights: stretch_weights(shape.table_row_bits()?, shape.slice.0, &coefficients.column)?,
        psi: output.point.clone(),
        target: output.ends[1].value,
        masks: vec![MaskTerm {
            oracle: obligation.mask.oracle,
            slot: obligation.mask.shown_slot,
            coeff: output.ends[1].mask_scale,
        }],
    };
    let mask_row = ConsumerMaskRow {
        oracle: obligation.mask.oracle,
        base_slot: obligation.mask.hiding_base,
        coefficients: output.hiding_form.clone(),
        target: output.row_target,
    };
    let raw = ConsumerRawObligation {
        request: obligation.request,
        source_step: obligation.source_step,
        activation_step: obligation.activation_step,
        positions: shape.positions,
        slice: shape.slice,
        target,
        position_coefficients: coefficients.position.clone(),
        column_coefficients: coefficients.column.clone(),
    };
    Ok(ConsumerPscOutcome {
        point: output.point.clone(),
        pending,
        mask_row,
        raw,
    })
}

/// Prover side of one consumer instance.
///
/// The activation rows are the public verified values of the activation step;
/// the raw rows are the projection's actual raw accumulator over the consumed
/// slice; the matrix places are the authenticated table's rows. The function
/// draws the coefficients, computes the reduction target from the raw rows,
/// fixes it in the chain, runs the masked product sumcheck and returns the
/// message beside the same-matrix pending row. An honest prover refuses a raw
/// target that is not the product of its own reduction, and refuses to mask
/// the public map or to leave the private map unmasked.
pub fn prove_consumer_psc(
    chain: &mut Chain,
    obligation: &ConsumerObligation,
    activation: &[Fp3],
    raw: &[Fp3],
    matrix_places: &dyn crate::proof::projection_claim::ConsumerMatrixColumns,
    masks: &ConsumerPscMasks,
) -> Result<(ConsumerPscProof, ConsumerPscOutcome)> {
    obligation.check()?;
    masks.check(&obligation.shape)?;
    let shape = &obligation.shape;
    let positions = shape.positions as usize;
    let records = usize::try_from(shape.records).map_err(|_| Error::ResourceLimit)?;
    if activation.len() != positions.checked_mul(records).ok_or(Error::ResourceLimit)? {
        return Err(Error::ShapeMismatch);
    }
    if matrix_places.places() != shape.places || matrix_places.records() != shape.records {
        return Err(Error::ShapeMismatch);
    }
    obligation.absorb(chain)?;
    let coefficients = draw_coefficients(chain, shape)?;
    let reduction = consumer_reduction(
        positions,
        shape.records,
        shape.places,
        shape.slice,
        activation,
        raw,
        &coefficients.position,
        &coefficients.column,
    )?;
    let columns = reduction.column_values(matrix_places)?;
    let claim = DotClaim {
        masks: Vec::new(),
        target: reduction.target,
    };
    absorb_target(chain, obligation.head, claim.target)?;
    let mut challenges =
        PositionPscChallenges::new(chain, CONSUMER_PSC_TAGS, u64::from(obligation.head));
    let (dot, output) = prove_dot_sumcheck(
        &reduction.shown,
        &columns,
        &claim,
        &masks.entries,
        &mut challenges,
    )?;
    check_product_claims(&output, &claim, &masks.entries, &reduction.shown, &columns)?;
    let outcome = outcome(obligation, &coefficients, claim.target, &output)?;
    crate::proof::hidden_member::check_pending(
        &outcome.pending,
        matrix_places,
        &|oracle, slot| {
            if oracle == obligation.mask.oracle && slot == obligation.mask.shown_slot {
                Ok(masks.entries.shown[1])
            } else {
                Err(Error::ShapeMismatch)
            }
        },
    )?;
    let proof = ConsumerPscProof {
        target: claim.target,
        point_values: [output.ends[0].value, output.ends[1].value],
        dot,
    };
    Ok((proof, outcome))
}

/// Verifier side of one consumer instance.
///
/// The verifier draws the same coefficients, fixes the same target, replays
/// the transcript from its own challenges and checks the public map at its own
/// point. It never sees the matrix, the raw rows or the mask entries; the
/// pending row and the mask row it returns carry every private value as a
/// claim on the committed messages. The raw obligation stays registered and
/// open.
pub fn verify_consumer_psc(
    chain: &mut Chain,
    obligation: &ConsumerObligation,
    activation: &[Fp3],
    proof: &ConsumerPscProof,
) -> Result<ConsumerPscOutcome> {
    obligation.check()?;
    proof.check(&obligation.shape)?;
    let shape = &obligation.shape;
    let positions = shape.positions as usize;
    let records = usize::try_from(shape.records).map_err(|_| Error::ResourceLimit)?;
    if activation.len() != positions.checked_mul(records).ok_or(Error::ResourceLimit)? {
        return Err(Error::ShapeMismatch);
    }
    obligation.absorb(chain)?;
    let coefficients = draw_coefficients(chain, shape)?;
    absorb_target(chain, obligation.head, proof.target)?;
    let entries = DotMaskEntries {
        hiding: vec![Fp3::ZERO; shape.hiding_entries()?],
        shown: [Fp3::ZERO; SHOWN_MAPS],
    };
    let claim = DotClaim {
        masks: Vec::new(),
        target: proof.target,
    };
    let mut challenges =
        PositionPscChallenges::new(chain, CONSUMER_PSC_TAGS, u64::from(obligation.head));
    let output = verify_psc_transcript(
        &claim,
        &entries,
        &proof.dot,
        shape.variables()?,
        &mut challenges,
    )?;
    if proof.point_values != [output.ends[0].value, output.ends[1].value] {
        return Err(Error::InvalidProof);
    }
    let shown = activation_shown(positions, shape.records, activation, &coefficients.position)?;
    if multilinear_eval(&shown, &output.point)? != output.ends[0].value {
        return Err(Error::InvalidProof);
    }
    outcome(obligation, &coefficients, proof.target, &output)
}

/// Prover side of one consumer instance with fixed coefficients and target.
/// The caller owns the chain statement and has fixed the obligation, the
/// coefficients and the target before this call; the function neither absorbs
/// the obligation nor draws any coefficient. The sumcheck enforces the fixed
/// target against the prover's own reduction, so a plan target that is not the
/// witness reduction is refused.
#[allow(clippy::too_many_arguments)]
pub fn prove_consumer_psc_fixed(
    chain: &mut Chain,
    obligation: &ConsumerObligation,
    activation: &[Fp3],
    raw: &[Fp3],
    matrix_places: &dyn crate::proof::projection_claim::ConsumerMatrixColumns,
    masks: &ConsumerPscMasks,
    position: &[Fp3],
    column: &[Fp3],
    target: Fp3,
) -> Result<(ConsumerPscProof, ConsumerPscOutcome)> {
    obligation.check()?;
    masks.check(&obligation.shape)?;
    let shape = &obligation.shape;
    let positions = shape.positions as usize;
    let records = usize::try_from(shape.records).map_err(|_| Error::ResourceLimit)?;
    let slice_len = usize::try_from(shape.slice_len()?).map_err(|_| Error::ResourceLimit)?;
    if activation.len() != positions.checked_mul(records).ok_or(Error::ResourceLimit)?
        || position.len() != positions
        || column.len() != slice_len
    {
        return Err(Error::ShapeMismatch);
    }
    if matrix_places.places() != shape.places || matrix_places.records() != shape.records {
        return Err(Error::ShapeMismatch);
    }
    let reduction = consumer_reduction(
        positions,
        shape.records,
        shape.places,
        shape.slice,
        activation,
        raw,
        position,
        column,
    )?;
    if reduction.target != target {
        return Err(Error::InvalidProof);
    }
    let columns = reduction.column_values(matrix_places)?;
    let claim = DotClaim {
        masks: Vec::new(),
        target,
    };
    // The fixed target enters the canonical prefix before the first challenge:
    // the first sumcheck challenge multiplies it, so it cannot be fixed later.
    let mut challenges =
        PositionPscChallenges::new(chain, CONSUMER_PSC_TAGS, u64::from(obligation.head));
    challenges.send_claim(target)?;
    let (dot, output) = prove_dot_sumcheck(
        &reduction.shown,
        &columns,
        &claim,
        &masks.entries,
        &mut challenges,
    )?;
    check_product_claims(&output, &claim, &masks.entries, &reduction.shown, &columns)?;
    let coefficients = ConsumerCoefficients {
        position: position.to_vec(),
        column: column.to_vec(),
    };
    let outcome = outcome(obligation, &coefficients, target, &output)?;
    crate::proof::hidden_member::check_pending(
        &outcome.pending,
        matrix_places,
        &|oracle, slot| {
            if oracle == obligation.mask.oracle && slot == obligation.mask.shown_slot {
                Ok(masks.entries.shown[1])
            } else {
                Err(Error::ShapeMismatch)
            }
        },
    )?;
    let proof = ConsumerPscProof {
        target,
        point_values: [output.ends[0].value, output.ends[1].value],
        dot,
    };
    Ok((proof, outcome))
}

/// Verifier side of one consumer instance with fixed coefficients and target.
/// The verifier uses the caller's target, not the proof's, and refuses a proof
/// whose declared target differs.
#[allow(clippy::too_many_arguments)]
pub fn verify_consumer_psc_fixed(
    chain: &mut Chain,
    obligation: &ConsumerObligation,
    activation: &[Fp3],
    proof: &ConsumerPscProof,
    position: &[Fp3],
    column: &[Fp3],
    target: Fp3,
) -> Result<ConsumerPscOutcome> {
    obligation.check()?;
    proof.check(&obligation.shape)?;
    if proof.target != target {
        return Err(Error::InvalidProof);
    }
    let shape = &obligation.shape;
    let positions = shape.positions as usize;
    let records = usize::try_from(shape.records).map_err(|_| Error::ResourceLimit)?;
    let slice_len = usize::try_from(shape.slice_len()?).map_err(|_| Error::ResourceLimit)?;
    if activation.len() != positions.checked_mul(records).ok_or(Error::ResourceLimit)?
        || position.len() != positions
        || column.len() != slice_len
    {
        return Err(Error::ShapeMismatch);
    }
    let entries = DotMaskEntries {
        hiding: vec![Fp3::ZERO; shape.hiding_entries()?],
        shown: [Fp3::ZERO; SHOWN_MAPS],
    };
    let claim = DotClaim {
        masks: Vec::new(),
        target,
    };
    // Both sides absorb the caller-fixed target before the first challenge.
    let mut challenges =
        PositionPscChallenges::new(chain, CONSUMER_PSC_TAGS, u64::from(obligation.head));
    challenges.send_claim(target)?;
    let output = verify_psc_transcript(
        &claim,
        &entries,
        &proof.dot,
        shape.variables()?,
        &mut challenges,
    )?;
    if proof.point_values != [output.ends[0].value, output.ends[1].value] {
        return Err(Error::InvalidProof);
    }
    let shown = activation_shown(positions, shape.records, activation, position)?;
    if multilinear_eval(&shown, &output.point)? != output.ends[0].value {
        return Err(Error::InvalidProof);
    }
    let coefficients = ConsumerCoefficients {
        position: position.to_vec(),
        column: column.to_vec(),
    };
    outcome(obligation, &coefficients, target, &output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::projection_claim::PlaceRows;
    use zkie_engine::pcs::request::field::fp_from_u64;

    fn value(word: u64) -> Fp3 {
        Fp3::from_base(fp_from_u64(word))
    }

    /// The fixture geometry: three positions, a 16-place matrix and eight
    /// records. Head one consumes places [4, 8).
    fn shape() -> ConsumerPscShape {
        ConsumerPscShape {
            positions: 3,
            places: 16,
            records: 8,
            slice: (4, 8),
        }
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

    fn mask_source() -> ConsumerMaskSource {
        ConsumerMaskSource {
            oracle: 31,
            hiding_base: 40,
            shown_slot: 60,
        }
    }

    fn obligation() -> ConsumerObligation {
        ConsumerObligation {
            request: 7,
            source_step: 23,
            activation_step: 9,
            head: 1,
            shape: shape(),
            shift: 16,
            table: table(),
            mask: mask_source(),
        }
    }

    fn activation(shape: &ConsumerPscShape) -> Vec<Fp3> {
        let count = shape.positions as usize * shape.records as usize;
        (0..count)
            .map(|index| value((index as u64 * 5 + 2) % 13))
            .collect()
    }

    /// The matrix in the plan's layout: one row per place, one entry per
    /// record, the transpose of the stored weight records.
    fn matrix(shape: &ConsumerPscShape) -> Vec<Vec<Fp3>> {
        (0..shape.places as usize)
            .map(|place| {
                (0..shape.records as usize)
                    .map(|record| value(((record as u64 * 7 + place as u64 * 3 + 1) % 19) as u64))
                    .collect()
            })
            .collect()
    }

    fn raw_rows(shape: &ConsumerPscShape, activation: &[Fp3], matrix: &[Vec<Fp3>]) -> Vec<Fp3> {
        let positions = shape.positions as usize;
        let records = shape.records as usize;
        let (start, end) = shape.slice;
        let slice_len = (end - start) as usize;
        let mut raw = vec![Fp3::ZERO; positions * slice_len];
        for position in 0..positions {
            for (index, place) in (start as usize..end as usize).enumerate() {
                let mut sum = Fp3::ZERO;
                for record in 0..records {
                    sum = sum
                        .add(&activation[position * records + record].mul(&matrix[place][record]));
                }
                raw[position * slice_len + index] = sum;
            }
        }
        raw
    }

    fn masks(shape: &ConsumerPscShape) -> ConsumerPscMasks {
        let hiding = (0..shape.hiding_entries().unwrap())
            .map(|index| value(100 + index as u64))
            .collect();
        ConsumerPscMasks::new(hiding, value(200))
    }

    fn chain() -> Chain {
        Chain::begin(
            91,
            &Digest::ZERO,
            b"consumer psc",
            b"consumer fixture",
            None,
        )
        .unwrap()
    }

    fn prove_fixture() -> (
        ConsumerPscProof,
        ConsumerPscOutcome,
        ConsumerPscOutcome,
        ConsumerObligation,
        Vec<Fp3>,
        Vec<Vec<Fp3>>,
        Vec<Fp3>,
        ConsumerPscMasks,
    ) {
        let obligation = obligation();
        let shape = obligation.shape;
        let activation = activation(&shape);
        let matrix = matrix(&shape);
        let raw = raw_rows(&shape, &activation, &matrix);
        let masks = masks(&shape);
        let (proof, proven) = prove_consumer_psc(
            &mut chain(),
            &obligation,
            &activation,
            &raw,
            &PlaceRows::new(&matrix),
            &masks,
        )
        .unwrap();
        let verified = verify_consumer_psc(&mut chain(), &obligation, &activation, &proof).unwrap();
        (
            proof, proven, verified, obligation, activation, matrix, raw, masks,
        )
    }

    #[test]
    fn the_consumer_product_sum_leaves_the_same_matrix_pending() {
        let (proof, proven, verified, obligation, activation, matrix, raw, masks) = prove_fixture();
        assert_eq!(proven, verified);
        let outcome = &verified;
        assert_eq!(outcome.point.len(), 3);
        assert_eq!(outcome.pending.psi, outcome.point);
        assert_eq!(outcome.pending.weights.len(), 1);
        assert_eq!(
            outcome.pending.weights,
            stretch_weights(4, 4, &outcome.raw.column_coefficients).unwrap()
        );
        assert_eq!(outcome.pending.target, proof.point_values[1]);
        assert_eq!(outcome.pending.masks.len(), 1);
        assert_eq!(outcome.pending.masks[0].oracle, obligation.mask.oracle);
        assert_eq!(outcome.pending.masks[0].slot, obligation.mask.shown_slot);
        assert_eq!(
            outcome.mask_row.coefficients.len(),
            obligation.shape.hiding_entries().unwrap()
        );
        assert_eq!(
            outcome.mask_row.units().unwrap().len(),
            outcome.mask_row.coefficients.len()
        );
        // The same-matrix pending row holds against the real matrix and the
        // one-time shown entry.
        crate::proof::hidden_member::check_pending(&outcome.pending, &PlaceRows::new(&matrix), &|oracle, slot| {
            if oracle == obligation.mask.oracle && slot == obligation.mask.shown_slot {
                Ok(masks.entries.shown[1])
            } else {
                Err(Error::ShapeMismatch)
            }
        })
        .unwrap();
        // The mask row holds against the one-time hiding entries.
        let mut row = Fp3::ZERO;
        for (index, coefficient) in outcome.mask_row.coefficients.iter().enumerate() {
            row = row.add(&coefficient.mul(&masks.entries.hiding[index]));
        }
        assert_eq!(row, outcome.mask_row.target);
        // The raw obligation carries the drawn coefficients and the target of
        // the actual raw accumulator.
        assert_eq!(
            outcome.raw.position_coefficients.len(),
            activation.len() / 8
        );
        assert_eq!(outcome.raw.column_coefficients.len(), 4);
        assert_eq!(outcome.raw.target, proof.target);
        assert_eq!(outcome.raw.source_step, obligation.source_step);
        assert_eq!(outcome.raw.activation_step, obligation.activation_step);
        // The registered raw obligation closes on the actual raw rows, and a
        // changed raw row breaks it: this is the claim the raw relation of the
        // source step must supply.
        outcome.raw.check_on(&raw).unwrap();
        let mut changed_raw = raw.clone();
        changed_raw[0] = changed_raw[0].add(&value(1));
        assert!(outcome.raw.check_on(&changed_raw).is_err());
        // The message round-trips and the decoded message verifies.
        let bytes = proof.encode(&obligation.shape).unwrap();
        let decoded = ConsumerPscProof::decode(&obligation.shape, &bytes).unwrap();
        assert_eq!(decoded, proof);
        let replay = verify_consumer_psc(&mut chain(), &obligation, &activation, &decoded).unwrap();
        assert_eq!(replay, verified);
    }

    #[test]
    fn a_changed_public_map_or_target_is_refused() {
        let (proof, _proven, _verified, obligation, activation, _matrix, _raw, _masks) =
            prove_fixture();

        let mut changed = activation.clone();
        changed[0] = changed[0].add(&value(1));
        assert!(
            verify_consumer_psc(&mut chain(), &obligation, &changed, &proof).is_err(),
            "a changed public activation breaks the public map check"
        );

        let mut changed = proof.clone();
        changed.target = changed.target.add(&value(1));
        assert!(verify_consumer_psc(&mut chain(), &obligation, &activation, &changed).is_err());

        let mut changed = proof.clone();
        changed.point_values[1] = changed.point_values[1].add(&value(1));
        assert!(verify_consumer_psc(&mut chain(), &obligation, &activation, &changed).is_err());

        let mut changed = proof.clone();
        changed.dot.end_values[0] = changed.dot.end_values[0].add(&value(1));
        assert!(verify_consumer_psc(&mut chain(), &obligation, &activation, &changed).is_err());

        let mut changed = proof;
        changed.dot.rounds[0].coefficients[0] = value(7);
        assert!(verify_consumer_psc(&mut chain(), &obligation, &activation, &changed).is_err());
    }

    #[test]
    fn a_masked_public_map_or_an_unmasked_private_map_is_refused() {
        let obligation = obligation();
        let shape = obligation.shape;
        let activation = activation(&shape);
        let matrix = matrix(&shape);
        let raw = raw_rows(&shape, &activation, &matrix);
        let honest = masks(&shape);

        let mut masked_public = honest.clone();
        masked_public.entries.shown[0] = value(3);
        assert!(prove_consumer_psc(
            &mut chain(),
            &obligation,
            &activation,
            &raw,
            &PlaceRows::new(&matrix),
            &masked_public
        )
        .is_err());

        let mut unmasked_private = honest.clone();
        unmasked_private.entries.shown[1] = Fp3::ZERO;
        assert!(prove_consumer_psc(
            &mut chain(),
            &obligation,
            &activation,
            &raw,
            &PlaceRows::new(&matrix),
            &unmasked_private
        )
        .is_err());

        let mut wrong_count = honest;
        wrong_count.entries.hiding.pop();
        assert!(prove_consumer_psc(
            &mut chain(),
            &obligation,
            &activation,
            &raw,
            &PlaceRows::new(&matrix),
            &wrong_count
        )
        .is_err());
    }

    #[test]
    fn the_requantized_output_cannot_stand_for_the_raw_claim() {
        let obligation = obligation();
        let shape = obligation.shape;
        let activation = activation(&shape);
        let matrix = matrix(&shape);
        let raw = raw_rows(&shape, &activation, &matrix);
        // A requantized row set is not the accumulator: the reduction target
        // no longer equals the product of the two maps, so the honest prover
        // refuses the run instead of proving the wrong claim.
        let scale = value(16).inverse().unwrap();
        let requantized: Vec<Fp3> = raw.iter().map(|entry| entry.mul(&scale)).collect();
        assert!(prove_consumer_psc(
            &mut chain(),
            &obligation,
            &activation,
            &requantized,
            &PlaceRows::new(&matrix),
            &masks(&shape)
        )
        .is_err());
    }

    #[test]
    fn a_pending_from_another_mask_entry_or_position_is_refused() {
        let (_proof, _proven, verified, obligation, _activation, matrix, _raw, masks) =
            prove_fixture();
        // The pending names one committed shown entry; checking it against a
        // different entry fails.
        let mut changed_mask = masks.clone();
        changed_mask.entries.shown[1] = changed_mask.entries.shown[1].add(&value(1));
        assert!(crate::proof::hidden_member::check_pending(
            &verified.pending,
            &PlaceRows::new(&matrix),
            &|oracle, slot| {
                if oracle == obligation.mask.oracle && slot == obligation.mask.shown_slot {
                    Ok(changed_mask.entries.shown[1])
                } else {
                    Err(Error::ShapeMismatch)
                }
            },
        )
        .is_err());
        // Another request or another source step changes the transcript, so
        // the same message no longer verifies.
        let mut other = obligation.clone();
        other.request += 1;
        let (proof, _proven, _verified, _obligation, activation, _matrix, _raw, _masks) =
            prove_fixture();
        assert!(verify_consumer_psc(&mut chain(), &other, &activation, &proof).is_err());
        let mut other = obligation;
        other.source_step += 2;
        assert!(verify_consumer_psc(&mut chain(), &other, &activation, &proof).is_err());
    }

    fn projection_step() -> PlanStep {
        PlanStep {
            opcode: OpCode::Projection,
            config: crate::proof::plan::config(vec![3, 4, 8, 16, 0]),
            inputs: vec![
                PlanSource::prev(9, 0),
                PlanSource::asset_slice("weights/L0_k_w_i32.bin", PlanLayout::columns(16, 4, 4)),
            ],
            outputs: 1,
        }
    }

    #[test]
    fn registration_binds_the_real_projection_source() {
        let step = projection_step();
        let registered =
            ConsumerObligation::from_projection_step(7, 23, 9, 1, &step, table(), mask_source())
                .unwrap();
        assert_eq!(registered, obligation());

        let mut wrong_opcode = step.clone();
        wrong_opcode.opcode = OpCode::MatMul;
        assert!(ConsumerObligation::from_projection_step(
            7,
            23,
            9,
            1,
            &wrong_opcode,
            table(),
            mask_source()
        )
        .is_err());

        let mut wrong_config = step.clone();
        wrong_config.config[1] = PlanConfigValue::Literal(8);
        assert!(ConsumerObligation::from_projection_step(
            7,
            23,
            9,
            1,
            &wrong_config,
            table(),
            mask_source()
        )
        .is_err());

        let mut wrong_activation = step.clone();
        wrong_activation.inputs[0] = PlanSource::prev(10, 0);
        assert!(ConsumerObligation::from_projection_step(
            7,
            23,
            9,
            1,
            &wrong_activation,
            table(),
            mask_source()
        )
        .is_err());

        let mut wrong_slice = step.clone();
        wrong_slice.inputs[1] =
            PlanSource::asset_slice("weights/L0_k_w_i32.bin", PlanLayout::columns(16, 8, 4));
        assert!(ConsumerObligation::from_projection_step(
            7,
            23,
            9,
            1,
            &wrong_slice,
            table(),
            mask_source()
        )
        .is_err());

        let mut wrong_path = step.clone();
        wrong_path.inputs[1] =
            PlanSource::asset_slice("weights/L0_v_w_i32.bin", PlanLayout::columns(16, 4, 4));
        assert!(ConsumerObligation::from_projection_step(
            7,
            23,
            9,
            1,
            &wrong_path,
            table(),
            mask_source()
        )
        .is_err());

        let mut non_literal = step;
        non_literal.config[2] = PlanConfigValue::AssetI64 {
            path: "weights/L0_k_w_i32.bin".to_string(),
            index: 0,
        };
        assert!(ConsumerObligation::from_projection_step(
            7,
            23,
            9,
            1,
            &non_literal,
            table(),
            mask_source()
        )
        .is_err());
    }

    #[test]
    fn registration_binds_the_real_production_geometry() {
        use crate::proof::hidden_member::WeightBlock;

        let step = PlanStep {
            opcode: OpCode::Projection,
            config: crate::proof::plan::config(vec![16, 64, 1024, 16, 0]),
            inputs: vec![
                PlanSource::prev(3, 0),
                PlanSource::asset_slice(
                    "gpt2/weights/L0_k_w_i32.bin",
                    PlanLayout::columns(1024, 320, 64),
                ),
            ],
            outputs: 1,
        };
        let table = ConsumerTable {
            path: "gpt2/weights/L0_k_w_i32.bin".to_string(),
            oracle: 9,
            root: Digest([0x21; 32]),
            identity: Digest([0x22; 32]),
            configuration: Digest([0x23; 32]),
        };
        let mask = ConsumerMaskSource {
            oracle: 10,
            hiding_base: 400,
            shown_slot: 900,
        };
        let obligation =
            ConsumerObligation::from_projection_step(1, 7, 3, 5, &step, table, mask).unwrap();
        assert_eq!(obligation.shape.positions, 16);
        assert_eq!(obligation.shape.places, 1024);
        assert_eq!(obligation.shape.records, 1024);
        assert_eq!(obligation.shape.slice, (320, 384));
        assert_eq!(obligation.shape.slice_len().unwrap(), 64);
        assert_eq!(obligation.shape.table_row_bits().unwrap(), 10);
        assert_eq!(obligation.shape.variables().unwrap(), 10);
        assert_eq!(obligation.shape.hiding_entries().unwrap(), 23);
        assert_eq!(obligation.shift, 16);
        assert_eq!(obligation.head, 5);
        // The pending row's weights cover exactly the head's place slice as
        // one aligned block and claim nothing outside it.
        let weights = stretch_weights(
            obligation.shape.table_row_bits().unwrap(),
            obligation.shape.slice.0,
            &vec![value(1); 64],
        )
        .unwrap();
        assert_eq!(weights.len(), 1);
        match &weights[0] {
            WeightBlock::Dense {
                first_row,
                log_rows,
                values,
            } => {
                assert_eq!(*first_row, 320);
                assert_eq!(*log_rows, 6);
                assert_eq!(values.len(), 64);
            }
            WeightBlock::Scalar { .. } => panic!("the head slice is one aligned block"),
        }
    }

    #[test]
    fn heads_are_domain_separated_in_one_chain() {
        let (proof, _proven, _verified, obligation, activation, _matrix, _raw, _masks) =
            prove_fixture();
        let mut other_head = obligation.clone();
        other_head.head = 0;
        other_head.shape.slice = (0, 4);
        other_head.check().unwrap();
        assert!(verify_consumer_psc(&mut chain(), &other_head, &activation, &proof).is_err());
        verify_consumer_psc(&mut chain(), &obligation, &activation, &proof).unwrap();
    }

    #[test]
    fn the_coefficients_are_fixed_before_the_output() {
        // Two honest runs over different matrices and their own raw rows draw
        // the same coefficients: they enter the chain right after the
        // obligation and before any output-dependent message.
        let obligation = obligation();
        let shape = obligation.shape;
        let activation = activation(&shape);
        let masks = masks(&shape);
        let first_matrix = matrix(&shape);
        let first_raw = raw_rows(&shape, &activation, &first_matrix);
        let (_, first) = prove_consumer_psc(
            &mut chain(),
            &obligation,
            &activation,
            &first_raw,
            &PlaceRows::new(&first_matrix),
            &masks,
        )
        .unwrap();
        let mut second_matrix = first_matrix;
        second_matrix[4][0] = second_matrix[4][0].add(&value(1));
        let second_raw = raw_rows(&shape, &activation, &second_matrix);
        let (_, second) = prove_consumer_psc(
            &mut chain(),
            &obligation,
            &activation,
            &second_raw,
            &PlaceRows::new(&second_matrix),
            &masks,
        )
        .unwrap();
        assert_eq!(
            first.raw.position_coefficients,
            second.raw.position_coefficients
        );
        assert_eq!(
            first.raw.column_coefficients,
            second.raw.column_coefficients
        );
        assert_ne!(first.raw.target, second.raw.target);
        first.raw.check_on(&first_raw).unwrap();
        second.raw.check_on(&second_raw).unwrap();
    }

    #[test]
    fn obligation_and_shape_faults_are_refused() {
        let mut bad = obligation();
        bad.shape.positions = 0;
        assert!(bad.check().is_err());

        let mut bad = obligation();
        bad.shape.places = 12;
        assert!(bad.check().is_err());

        let mut bad = obligation();
        bad.shape.records = 12;
        assert!(bad.check().is_err());

        let mut bad = obligation();
        bad.shape.slice = (8, 4);
        assert!(bad.check().is_err());

        let mut bad = obligation();
        bad.shape.slice = (4, 20);
        assert!(bad.check().is_err());

        let mut bad = obligation();
        bad.head = 2;
        assert!(bad.check().is_err());

        let mut bad = obligation();
        bad.table.root = Digest::ZERO;
        assert!(bad.check().is_err());

        let mut bad = obligation();
        bad.mask.shown_slot = bad.mask.hiding_base;
        assert!(bad.check().is_err());

        let mut bad = obligation();
        bad.shift = 0;
        assert!(bad.check().is_err());

        let mut bad = obligation();
        bad.source_step = bad.activation_step;
        assert!(bad.check().is_err());
    }
}
