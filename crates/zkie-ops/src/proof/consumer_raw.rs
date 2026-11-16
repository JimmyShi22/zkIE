//! The committed raw accumulator relation of the projection consumer.
//!
//! The raw accumulator of a projection consumer is a hidden witness: it is
//! never sent, never published and never returned. It is committed as one
//! value message per head, next to one committed message per head holding the
//! rounding remainders' bits. Both messages are committed before any
//! challenge; their roots are the plan's declared roots, and the first message
//! only repeats the declaration.
//!
//! Two masked relations per head connect the hidden accumulator to the head
//! step's public output under the exact production rounding rule
//! `out = rq(Y, shift)`, which is equivalent to the integer equations
//!
//! ```text
//! Y_e - sum_j 2^j bit[e, j] = out_e * 2^shift - 2^(shift-1),
//! bit[e, j] in {0, 1}.
//! ```
//!
//! The equations are checked over the entry cube at a random point drawn
//! after the commitments: the prover's weighted map is the hidden side
//! `Y - bits`, the verifier's target is the public side from the output rows,
//! so the two sides only meet through the masked sumcheck. The Booleanity is
//! one weighted-bits sumcheck over the bit cube with target zero. No raw
//! value, no remainder and no unmasked reading leaves the prover.
//!
//! The integer gate of [`crate::proof::consumer_request::ConsumerIntegerBound`]
//! proves the accumulator and its rounding numerator stay below half the base
//! field, so the field equations are the integer equations with no modular
//! wrap and the `i128` accumulation semantics is preserved.
//!
//! The one-time hiding and shown masks of every relation instance live in the
//! request's mask message at the plan's fixed slots; the relation outputs are
//! returned as pending row obligations for the request chain's group check.
//! This module never checks the group rows themselves and never certifies the
//! raw side.

use std::collections::BTreeMap;

use zkie_core::common::codec::DecodeLimits;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::chain::Chain;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};

use crate::proof::committed_psc::{
    CommittedGroupMessage, PositionPscChallenges, PscTranscriptTags,
};
use crate::proof::consumer_request::{signed_fp3, ConsumerIntegerBound};
use crate::proof::hidden_send_session::{SendGroup, TableLayout, WShape};
use crate::proof::psc::{hp_entries, Weight};
use crate::proof::psc_general::{
    general_variables, prove_general, verify_general, GeneralChallenges, GeneralMaskEntries,
    GeneralProof, InnerSumcheck, Summand, WeightedBitsInner, WeightedMapInner,
};
use crate::proof::row_claim::RowForm;

/// Version of the consumer raw relation messages.
pub const CONSUMER_RAW_VERSION: u64 = 1;
/// Transcript tags of the rounding map relation.
pub const CONSUMER_RAW_ROUNDING_TAGS: PscTranscriptTags = PscTranscriptTags {
    version: CONSUMER_RAW_VERSION,
    hiding_sum: 100,
    round: 101,
    end: 102,
    claim: 103,
};
/// Transcript tag of the fixed first declaration.
const CONSUMER_RAW_FIRST_TAG: u64 = 108;

/// Transcript tags of the bit Booleanity relation.
pub const CONSUMER_RAW_BITS_TAGS: PscTranscriptTags = PscTranscriptTags {
    version: CONSUMER_RAW_VERSION,
    hiding_sum: 104,
    round: 105,
    end: 106,
    claim: 107,
};

/// The fixed logical geometry of the raw relation. Every message holds one
/// block per head; a block has `columns` entries covering the head's
/// `positions * width` real entries and zero padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsumerRawShape {
    pub heads: u32,
    pub positions: u32,
    pub width: u64,
    pub shift: u32,
    pub columns: u64,
}

impl ConsumerRawShape {
    pub fn fault(&self) -> Option<&'static str> {
        if self.heads == 0 || self.positions == 0 || self.width == 0 {
            return Some("the consumer raw relation has no entry");
        }
        if self.shift == 0 || !self.shift.is_power_of_two() {
            return Some("the consumer shift is not a power-of-two bit count");
        }
        let real = (self.positions as u64).checked_mul(self.width)?;
        if self.columns < real || !self.columns.is_power_of_two() {
            return Some("the raw message width does not cover the real entries");
        }
        self.bits_columns()?;
        None
    }

    /// The number of real entries of one head block.
    pub fn real_entries(&self) -> u64 {
        self.positions as u64 * self.width
    }

    /// The bit-message width of one head block.
    pub fn bits_columns(&self) -> Option<u64> {
        self.columns.checked_mul(self.shift as u64)
    }

    pub fn columns_variables(&self) -> u32 {
        self.columns.trailing_zeros()
    }

    pub fn bits_variables(&self) -> u32 {
        self.bits_columns().unwrap_or(0).trailing_zeros()
    }
}

/// The committed-message plan of one value message family: the shape, the
/// group id, the first oracle id, the opening count and the declared roots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerRawGroup {
    pub shape: WShape,
    pub columns: u64,
    pub group: u64,
    pub oracle_base: u64,
    pub openings: usize,
    pub roots: Vec<Digest>,
}

impl ConsumerRawGroup {
    pub fn fault(&self, heads: u32) -> Option<&'static str> {
        if self.columns == 0 || !self.columns.is_power_of_two() {
            return Some("a consumer raw group has no power-of-two width");
        }
        if self.shape.answers != 1 {
            return Some("a consumer raw group has no single sample answer");
        }
        if self.openings == 0 || self.openings as u64 > self.shape.code.r {
            return Some("a consumer raw group has no bounded opening count");
        }
        if self.roots.len() != heads as usize || self.roots.iter().any(|root| *root == Digest::ZERO)
        {
            return Some("a consumer raw group does not have one root per head");
        }
        if self.shape.fault(self.columns, heads).is_some() {
            return Some("a consumer raw group shape does not cover its entries");
        }
        if self.oracle_base == 0 || self.group == 0 {
            return Some("a consumer raw group has no identifier");
        }
        None
    }
}

/// The mask slots of one relation instance: the hiding entries and the shown
/// end entry. The slots index the request's committed mask message; the plan
/// fixes them before any challenge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerRawMaskSpace {
    pub hiding: Vec<u64>,
    pub shown: u64,
}

/// The mask source of the raw relation: the mask message oracle and the first
/// slot of the relation's block. Slot allocation is fixed and dense.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsumerRawMaskPlan {
    pub oracle: u64,
    pub base_slot: u64,
}

/// The fixed plan of the consumer raw relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerRawPlan {
    pub shape: ConsumerRawShape,
    pub raw: ConsumerRawGroup,
    pub bits: ConsumerRawGroup,
    pub mask: ConsumerRawMaskPlan,
    /// The digest of the head steps' verified output rows.
    pub outputs_digest: Digest,
    /// The integer no-wrap gate of the request.
    pub bound: ConsumerIntegerBound,
}

/// One relation instance's kind, in the plan's fixed order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsumerRawRelation {
    Rounding,
    Bits,
}

/// The mask-message slots one raw shape consumes from `base_slot`: every
/// instance's hiding entries and its shown entry. The group plan sizes its
/// mask member from this count before any witness is built.
pub fn consumer_raw_mask_entries(shape: &ConsumerRawShape, base_slot: u64) -> Result<u64> {
    shape
        .fault()
        .map_or(Ok(()), |_| Err(Error::InvalidConfig))?;
    let mut next = base_slot;
    for _ in 0..shape.heads {
        for hiding in [
            rounding_hiding(shape.columns_variables())?,
            bits_hiding(shape.bits_variables())?,
        ] {
            next = next
                .checked_add(hiding as u64 + 1)
                .ok_or(Error::ResourceLimit)?;
        }
    }
    Ok(next)
}

impl ConsumerRawPlan {
    pub fn validate(&self) -> Result<()> {
        self.shape
            .fault()
            .map_or(Ok(()), |_| Err(Error::InvalidConfig))?;
        self.raw
            .fault(self.shape.heads)
            .map_or(Ok(()), |_| Err(Error::InvalidConfig))?;
        self.bits
            .fault(self.shape.heads)
            .map_or(Ok(()), |_| Err(Error::InvalidConfig))?;
        if self.raw.columns != self.shape.columns
            || self.bits.columns != self.shape.bits_columns().ok_or(Error::InvalidConfig)?
        {
            return Err(Error::InvalidConfig);
        }
        if self.raw.group == self.bits.group
            || self.raw.oracle_base == self.bits.oracle_base
            || self.mask.oracle == 0
        {
            return Err(Error::InvalidConfig);
        }
        if self.outputs_digest == Digest::ZERO {
            return Err(Error::InvalidConfig);
        }
        self.bound.check()?;
        self.mask_spaces()?;
        Ok(())
    }

    /// The mask spaces of every instance, in head order with the rounding
    /// relation before the bit relation. The allocation is dense from the
    /// plan's base slot.
    pub fn mask_spaces(&self) -> Result<Vec<ConsumerRawMaskSpace>> {
        let mut spaces = Vec::with_capacity(self.instances());
        let mut next = self.mask.base_slot;
        for _head in 0..self.shape.heads {
            let rounding = rounding_hiding(self.shape.columns_variables())?;
            let bits = bits_hiding(self.shape.bits_variables())?;
            for (relation, hiding) in [
                (ConsumerRawRelation::Rounding, rounding),
                (ConsumerRawRelation::Bits, bits),
            ] {
                let mut slots = Vec::with_capacity(hiding);
                for _ in 0..hiding {
                    slots.push(next);
                    next = next.checked_add(1).ok_or(Error::ResourceLimit)?;
                }
                let shown = next;
                next = next.checked_add(1).ok_or(Error::ResourceLimit)?;
                let _ = relation;
                spaces.push(ConsumerRawMaskSpace {
                    hiding: slots,
                    shown,
                });
            }
        }
        Ok(spaces)
    }

    pub fn instances(&self) -> usize {
        self.shape.heads as usize * 2
    }

    pub fn relation(&self, instance: usize) -> Result<ConsumerRawRelation> {
        if instance >= self.instances() {
            return Err(Error::InvalidIndex);
        }
        Ok(if instance % 2 == 0 {
            ConsumerRawRelation::Rounding
        } else {
            ConsumerRawRelation::Bits
        })
    }

    pub fn head(&self, instance: usize) -> Result<u32> {
        if instance >= self.instances() {
            return Err(Error::InvalidIndex);
        }
        Ok((instance / 2) as u32)
    }
}

fn rounding_variables(columns: u32) -> Result<Vec<Weight>> {
    let point = vec![Fp3::ZERO; columns as usize];
    equality_variables(&point)
}

fn equality_variables(point: &[Fp3]) -> Result<Vec<Weight>> {
    point
        .iter()
        .map(|coordinate| Weight::equality(*coordinate, 3))
        .collect()
}

fn rounding_hiding(columns: u32) -> Result<usize> {
    let variables = general_variables(&rounding_variables(columns)?, &Summand::Map)?;
    hp_entries(&variables)
}

fn bits_hiding(variables: u32) -> Result<usize> {
    let point = vec![Fp3::ZERO; variables as usize];
    let variables = general_variables(&equality_variables(&point)?, &Summand::Bits)?;
    hp_entries(&variables)
}

/// The declared first message: the two committed messages, repeated as group
/// declarations. The roots are the plan's; the function only lets both sides
/// derive the same root list from the wire.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerRawFirst {
    pub raw: CommittedGroupMessage,
    pub bits: CommittedGroupMessage,
}

impl ConsumerRawFirst {
    pub fn from_groups(raw: &SendGroup, bits: &SendGroup) -> Self {
        Self {
            raw: CommittedGroupMessage::new(raw.clone()),
            bits: CommittedGroupMessage::new(bits.clone()),
        }
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(CONSUMER_RAW_VERSION);
        writer.put_bytes(&self.raw.group.encode());
        writer.put_bytes(&self.bits.group.encode());
        Ok(writer.into_bytes())
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        if reader.get_u64()? != CONSUMER_RAW_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let raw = CommittedGroupMessage::new(SendGroup::decode(reader.get_bytes()?)?);
        let bits = CommittedGroupMessage::new(SendGroup::decode(reader.get_bytes()?)?);
        reader.finish()?;
        Ok(Self { raw, bits })
    }

    pub fn check(&self, plan: &ConsumerRawPlan) -> Result<()> {
        plan.validate()?;
        let heads = plan.shape.heads as usize;
        for (message, group) in [(&self.raw, &plan.raw), (&self.bits, &plan.bits)] {
            if message.group.fault(plan.shape.heads).is_some()
                || message.roots.len() != heads
                || message.roots != group.roots
            {
                return Err(Error::StatementMismatch);
            }
            if message.group.group != group.group {
                return Err(Error::StatementMismatch);
            }
            for (position, entry) in message.group.entries.iter().enumerate() {
                if entry.position != position as u32
                    || entry.oracle
                        != group
                            .oracle_base
                            .checked_add(position as u64)
                            .ok_or(Error::ResourceLimit)?
                    || entry.shape != group.shape.oracle_shape()
                    || entry.tree != group.shape.tree_plan(position as u32)?
                {
                    return Err(Error::StatementMismatch);
                }
            }
        }
        Ok(())
    }
}

/// The public rounding target of one head: the equal-weighted public side
/// `out_e * 2^shift - 2^(shift-1)` of the integer equations at the drawn
/// point. The prover's map is the hidden side `Y - bits`.
pub fn consumer_rounding_target(
    shape: &ConsumerRawShape,
    point: &[Fp3],
    outputs: &[i64],
) -> Result<Fp3> {
    if point.len() != shape.columns_variables() as usize {
        return Err(Error::ShapeMismatch);
    }
    let real = shape.real_entries() as usize;
    if outputs.len() != real {
        return Err(Error::ShapeMismatch);
    }
    let shift = u32::try_from(shape.shift).map_err(|_| Error::InvalidConfig)?;
    let scale = 1i128.checked_shl(shift).ok_or(Error::ResourceLimit)?;
    let half = 1i128.checked_shl(shift - 1).ok_or(Error::ResourceLimit)?;
    let mut target = Fp3::ZERO;
    for (index, output) in outputs.iter().enumerate() {
        let weight = eq_at(point, index as u64);
        let value = i128::from(*output)
            .checked_mul(scale)
            .and_then(|value| value.checked_sub(half))
            .ok_or(Error::BoundViolation)?;
        target = target.add(&weight.mul(&signed_fp3(value)?));
    }
    Ok(target)
}

fn eq_at(point: &[Fp3], index: u64) -> Fp3 {
    let mut weight = Fp3::ONE;
    for (bit, coordinate) in point.iter().enumerate() {
        let chosen = if (index >> bit) & 1 == 1 {
            *coordinate
        } else {
            Fp3::ONE.sub(coordinate)
        };
        weight = weight.mul(&chosen);
    }
    weight
}



/// The prover's hidden rounding map of one head: `Y_e - sum_j 2^j bit[e,j]`
/// over the padded entry cube, zero on the padding. The public side is added
/// by the verifier's target, never here.
pub fn consumer_rounding_map(
    shape: &ConsumerRawShape,
    raw: &[Fp3],
    bits: &[Fp3],
) -> Result<Vec<Fp3>> {
    let columns = usize::try_from(shape.columns).map_err(|_| Error::ResourceLimit)?;
    let shift = shape.shift as usize;
    if raw.len() != columns || bits.len() != columns * shift {
        return Err(Error::ShapeMismatch);
    }
    let mut values = vec![Fp3::ZERO; columns];
    for (entry, value) in values
        .iter_mut()
        .enumerate()
        .take(shape.real_entries() as usize)
    {
        let mut reconstructed = Fp3::ZERO;
        let mut power = Fp3::ONE;
        for bit in 0..shift {
            let bit_value = bits[entry * shift + bit];
            reconstructed = reconstructed.add(&power.mul(&bit_value));
            power = power.add(&power);
        }
        *value = raw[entry].sub(&reconstructed);
    }
    Ok(values)
}

/// One relation instance's output: the transcript output and the mask slots it
/// uses. The row is the pending obligation of the request chain's group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerRawObligation {
    pub instance: usize,
    pub head: u32,
    pub relation: ConsumerRawRelation,
    pub output: crate::proof::psc_general::GeneralOutput,
    pub row: RowForm,
    pub hiding_slots: Vec<u64>,
    pub shown_slot: u64,
    /// The shown map's linear end claim of this instance: the (E) row the
    /// group must check against the same head's committed messages and this
    /// instance's own shown slot. The squared end value never appears here; it
    /// is already part of the output's row target.
    pub end: ConsumerRawEndForm,
}

/// The linear end claim of one relation instance. The row is
/// `<raw_columns, raw message> + <bits_columns, bits message>
///  + mask_scale * shown entry = target`, where every coordinate is the
/// instance's own: the same head's committed raw and bit messages and the
/// instance's shown slot. The columns are in the message column order; a
/// group scatters them onto the member tables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerRawEndForm {
    pub instance: usize,
    pub head: u32,
    pub raw_oracle: u64,
    pub raw_columns: Vec<Fp3>,
    pub bits_oracle: u64,
    pub bits_columns: Vec<Fp3>,
    pub shown_slot: u64,
    pub mask_scale: Fp3,
    pub target: Fp3,
}

impl ConsumerRawEndForm {
    pub fn raw_table(&self, shape: &WShape) -> Result<Vec<Fp3>> {
        if self.raw_columns.is_empty() {
            return Ok(Vec::new());
        }
        TableLayout::of(shape).scatter(self.raw_columns.len() as u64, &self.raw_columns)
    }

    pub fn bits_table(&self, shape: &WShape) -> Result<Vec<Fp3>> {
        if self.bits_columns.is_empty() {
            return Ok(Vec::new());
        }
        TableLayout::of(shape).scatter(self.bits_columns.len() as u64, &self.bits_columns)
    }
}

/// The second message: one masked sumcheck per instance, with its point value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerRawSecond {
    pub proofs: Vec<GeneralProof>,
    pub point_values: Vec<Fp3>,
}

impl ConsumerRawSecond {
    pub fn encode(&self, plan: &ConsumerRawPlan) -> Result<Vec<u8>> {
        self.check(plan)?;
        let mut writer = CanonicalWriter::new();
        writer.put_u64(CONSUMER_RAW_VERSION);
        writer.put_u64(self.proofs.len() as u64);
        for (proof, value) in self.proofs.iter().zip(&self.point_values) {
            writer.put_fp3(value);
            proof.encode_into(&mut writer);
        }
        Ok(writer.into_bytes())
    }

    pub fn decode(plan: &ConsumerRawPlan, bytes: &[u8]) -> Result<Self> {
        plan.validate()?;
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        if reader.get_u64()? != CONSUMER_RAW_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if count != plan.instances() {
            return Err(Error::ShapeMismatch);
        }
        let mut proofs = Vec::with_capacity(count);
        let mut point_values = Vec::with_capacity(count);
        for instance in 0..count {
            point_values.push(reader.get_fp3()?);
            let rounds = match plan.relation(instance)? {
                ConsumerRawRelation::Rounding => plan.shape.columns_variables() as usize,
                ConsumerRawRelation::Bits => plan.shape.bits_variables() as usize,
            };
            proofs.push(GeneralProof::decode(&mut reader, rounds, 1)?);
        }
        reader.finish()?;
        let message = Self {
            proofs,
            point_values,
        };
        message.check(plan)?;
        Ok(message)
    }

    pub fn check(&self, plan: &ConsumerRawPlan) -> Result<()> {
        plan.validate()?;
        if self.proofs.len() != plan.instances() || self.point_values.len() != plan.instances() {
            return Err(Error::ShapeMismatch);
        }
        for (instance, proof) in self.proofs.iter().enumerate() {
            let rounds = match plan.relation(instance)? {
                ConsumerRawRelation::Rounding => plan.shape.columns_variables() as usize,
                ConsumerRawRelation::Bits => plan.shape.bits_variables() as usize,
            };
            if proof.rounds.len() != rounds || proof.end_values.len() != 1 {
                return Err(Error::ShapeMismatch);
            }
        }
        Ok(())
    }
}

/// The prover's mask entries by slot, in the plan's dense block.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConsumerRawMaskWitness {
    values: BTreeMap<u64, Fp3>,
}

impl ConsumerRawMaskWitness {
    pub fn insert(&mut self, slot: u64, value: Fp3) {
        self.values.insert(slot, value);
    }

    pub fn entry(&self, slot: u64) -> Result<Fp3> {
        self.values.get(&slot).copied().ok_or(Error::ShapeMismatch)
    }
}

fn mask_entries(
    space: &ConsumerRawMaskSpace,
    witness: &ConsumerRawMaskWitness,
) -> Result<GeneralMaskEntries> {
    let mut hiding = Vec::with_capacity(space.hiding.len());
    for slot in &space.hiding {
        hiding.push(witness.entry(*slot)?);
    }
    Ok(GeneralMaskEntries {
        hiding,
        shown: vec![witness.entry(space.shown)?],
    })
}

fn zero_entries(space: &ConsumerRawMaskSpace) -> GeneralMaskEntries {
    GeneralMaskEntries {
        hiding: vec![Fp3::ZERO; space.hiding.len()],
        shown: vec![Fp3::ZERO],
    }
}

fn draw_point(chain: &mut Chain, variables: u32) -> Result<Vec<Fp3>> {
    let mut point = Vec::with_capacity(variables as usize);
    for _ in 0..variables {
        point.push(chain.draw_e()?);
    }
    Ok(point)
}

impl GeneralChallenges for ConsumerRawChallenges<'_> {
    fn draw(&mut self) -> Result<Fp3> {
        self.inner.draw()
    }

    fn draw_nonzero(&mut self) -> Result<Fp3> {
        self.inner.draw_nonzero()
    }

    fn send_hiding_sum(&mut self, value: Fp3) -> Result<()> {
        self.inner.send_hiding_sum(value)
    }

    fn send_round(&mut self, index: usize, round: &crate::proof::psc::PscRound) -> Result<()> {
        self.inner.send_round(index, round)
    }

    fn send_end_values(&mut self, values: &[Fp3]) -> Result<()> {
        self.inner.send_end_values(values)
    }
}

struct ConsumerRawChallenges<'a> {
    inner: PositionPscChallenges<'a>,
}

impl ConsumerRawChallenges<'_> {
    /// Absorbs the public relation target before the first sumcheck
    /// challenge, so the public output rows are part of the transcript
    /// before any value that multiplies them.
    fn send_claim(&mut self, target: Fp3) -> Result<()> {
        self.inner.send_claim(target)
    }
}

/// The first declaration is fixed in the transcript before any relation
/// challenge: both sides absorb the same declaration bytes.
fn absorb_first(chain: &mut Chain, first: &ConsumerRawFirst) -> Result<()> {
    let digest = zkie_core::artifact::digest_bytes(&first.encode()?);
    let mut message = chain.message();
    message.push_u64(CONSUMER_RAW_VERSION);
    message.push_u64(CONSUMER_RAW_FIRST_TAG);
    for word in digest.0.chunks_exact(8) {
        message.push_u64(u64::from_le_bytes(
            word.try_into().expect("eight-byte word"),
        ));
    }
    let prefix = message.prefix(zkie_engine::pcs::request::chain::KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn obligations(
    plan: &ConsumerRawPlan,
    instance: usize,
    output: crate::proof::psc_general::GeneralOutput,
) -> Result<ConsumerRawObligation> {
    let spaces = plan.mask_spaces()?;
    let space = spaces.get(instance).ok_or(Error::InvalidIndex)?;
    let relation = plan.relation(instance)?;
    let head = plan.head(instance)?;
    let row = RowForm::from_output(&output);
    let columns = usize::try_from(plan.shape.columns).map_err(|_| Error::ResourceLimit)?;
    let real = plan.shape.real_entries() as usize;
    let shift = plan.shape.shift as usize;
    let raw_oracle = plan
        .raw
        .oracle_base
        .checked_add(u64::from(head))
        .ok_or(Error::ResourceLimit)?;
    let bits_oracle = plan
        .bits
        .oracle_base
        .checked_add(u64::from(head))
        .ok_or(Error::ResourceLimit)?;
    let (raw_columns, bits_columns) = match relation {
        ConsumerRawRelation::Rounding => {
            let mut raw = vec![Fp3::ZERO; columns];
            let mut bits = vec![Fp3::ZERO; columns * shift];
            for entry in 0..real {
                let weight = eq_at(&output.point, entry as u64);
                raw[entry] = weight;
                let mut power = Fp3::ONE;
                for bit in 0..shift {
                    bits[entry * shift + bit] = Fp3::ZERO.sub(&weight.mul(&power));
                    power = power.add(&power);
                }
            }
            (raw, bits)
        }
        ConsumerRawRelation::Bits => {
            let mut bits = vec![Fp3::ZERO; columns * shift];
            for (entry, coefficient) in bits.iter_mut().enumerate() {
                *coefficient = eq_at(&output.point, entry as u64);
            }
            (Vec::new(), bits)
        }
    };
    let end = ConsumerRawEndForm {
        instance,
        head,
        raw_oracle,
        raw_columns,
        bits_oracle,
        bits_columns,
        shown_slot: space.shown,
        mask_scale: output.ends[0].mask_scale,
        target: output.ends[0].value,
    };
    Ok(ConsumerRawObligation {
        instance,
        head,
        relation,
        output,
        row,
        hiding_slots: space.hiding.clone(),
        shown_slot: space.shown,
        end,
    })
}

/// Prover side of the raw relation. The raw and bit values are the committed
/// witness rows; the outputs are the public verified rows; the masks are the
/// committed one-time entries. The caller's chain carries the transcript; the
/// first declaration must already be absorbed in it.
pub fn prove_consumer_raw(
    chain: &mut Chain,
    plan: &ConsumerRawPlan,
    first: &ConsumerRawFirst,
    raw: &[Vec<Fp3>],
    bits: &[Vec<Fp3>],
    outputs: &[Vec<i64>],
    masks: &ConsumerRawMaskWitness,
) -> Result<(ConsumerRawSecond, Vec<ConsumerRawObligation>)> {
    plan.validate()?;
    first.check(plan)?;
    absorb_first(chain, first)?;
    let heads = plan.shape.heads as usize;
    let columns = usize::try_from(plan.shape.columns).map_err(|_| Error::ResourceLimit)?;
    let bits_columns = usize::try_from(plan.shape.bits_columns().ok_or(Error::InvalidConfig)?)
        .map_err(|_| Error::ResourceLimit)?;
    let real = plan.shape.real_entries() as usize;
    if raw.len() != heads
        || bits.len() != heads
        || outputs.len() != heads
        || raw.iter().any(|head| head.len() != columns)
        || bits.iter().any(|head| head.len() != bits_columns)
        || outputs.iter().any(|head| head.len() != real)
    {
        return Err(Error::ShapeMismatch);
    }
    let spaces = plan.mask_spaces()?;
    let mut proofs = Vec::with_capacity(plan.instances());
    let mut point_values = Vec::with_capacity(plan.instances());
    let mut rows_out = Vec::with_capacity(plan.instances());
    for head in 0..heads {
        // Rounding relation: hidden map Y - bits against the public target.
        let point = draw_point(chain, plan.shape.columns_variables())?;
        let target = consumer_rounding_target(&plan.shape, &point, &outputs[head])?;
        let map = consumer_rounding_map(&plan.shape, &raw[head], &bits[head])?;
        let mut inner = WeightedMapInner::new(&map, &point)?;
        let variables = equality_variables(&point)?;
        let space = &spaces[head * 2];
        let entries = mask_entries(space, masks)?;
        let mut challenges = ConsumerRawChallenges {
            inner: PositionPscChallenges::new(chain, CONSUMER_RAW_ROUNDING_TAGS, (head * 2) as u64),
        };
        challenges.send_claim(target)?;
        let (proof, output) = prove_general(
            &variables,
            &[],
            &|_| Ok(Fp3::ZERO),
            target,
            &Summand::Map,
            None,
            &mut inner,
            &entries,
            &mut challenges,
        )?;
        point_values.push(output.ends[0].value);
        rows_out.push(obligations(plan, head * 2, output)?);
        proofs.push(proof);

        // Bit relation: Booleanity of every remainders' bit.
        let bit_point = draw_point(chain, plan.shape.bits_variables())?;
        let mut inner: Box<dyn InnerSumcheck> =
            Box::new(WeightedBitsInner::new(&bits[head], &bit_point)?);
        let bit_variables = equality_variables(&bit_point)?;
        let space = &spaces[head * 2 + 1];
        let entries = mask_entries(space, masks)?;
        let mut challenges = ConsumerRawChallenges {
            inner: PositionPscChallenges::new(chain, CONSUMER_RAW_BITS_TAGS, (head * 2 + 1) as u64),
        };
        challenges.send_claim(Fp3::ZERO)?;
        let (proof, output) = prove_general(
            &bit_variables,
            &[],
            &|_| Ok(Fp3::ZERO),
            Fp3::ZERO,
            &Summand::Bits,
            None,
            &mut *inner,
            &entries,
            &mut challenges,
        )?;
        point_values.push(output.ends[0].value);
        rows_out.push(obligations(plan, head * 2 + 1, output)?);
        proofs.push(proof);
    }
    // The prover self-checks every relation row against the one-time mask
    // entries it committed: the row must hold on the actual entries, or the
    // group check would reject the relation.
    for row in &rows_out {
        let mut hiding = Vec::with_capacity(row.hiding_slots.len());
        for slot in &row.hiding_slots {
            hiding.push(masks.entry(*slot)?);
        }
        if row
            .row
            .evaluate(&hiding, &|_| Err(Error::ShapeMismatch), &[])?
            != row.row.constant
        {
            return Err(Error::InvalidProof);
        }
    }
    // The end forms are the other half of every relation: the shown map's
    // linear claim on the same head's committed messages and this instance's
    // shown slot. The prover checks them on its witness; the group checks them
    // against the commitments.
    for obligation in &rows_out {
        let end = &obligation.end;
        let mut value = Fp3::ZERO;
        if !end.raw_columns.is_empty() {
            let raw = raw.get(end.head as usize).ok_or(Error::ShapeMismatch)?;
            if raw.len() < end.raw_columns.len() {
                return Err(Error::ShapeMismatch);
            }
            for (coefficient, entry) in end.raw_columns.iter().zip(raw) {
                value = value.add(&coefficient.mul(entry));
            }
        }
        if !end.bits_columns.is_empty() {
            let bits = bits.get(end.head as usize).ok_or(Error::ShapeMismatch)?;
            if bits.len() < end.bits_columns.len() {
                return Err(Error::ShapeMismatch);
            }
            for (coefficient, entry) in end.bits_columns.iter().zip(bits) {
                value = value.add(&coefficient.mul(entry));
            }
        }
        value = value.add(&end.mask_scale.mul(&masks.entry(end.shown_slot)?));
        if value != end.target {
            return Err(Error::InvalidProof);
        }
    }
    let second = ConsumerRawSecond {
        proofs,
        point_values,
    };
    second.check(plan)?;
    Ok((second, rows_out))
}

/// Verifier side of the raw relation. The verifier checks the same transcripts
/// from its own challenges and the public output target; it never receives the
/// raw rows, the bits or the mask entries.
pub fn verify_consumer_raw(
    chain: &mut Chain,
    plan: &ConsumerRawPlan,
    first: &ConsumerRawFirst,
    outputs: &[Vec<i64>],
    second: &ConsumerRawSecond,
) -> Result<Vec<ConsumerRawObligation>> {
    plan.validate()?;
    first.check(plan)?;
    second.check(plan)?;
    absorb_first(chain, first)?;
    let heads = plan.shape.heads as usize;
    let real = plan.shape.real_entries() as usize;
    if outputs.len() != heads || outputs.iter().any(|head| head.len() != real) {
        return Err(Error::ShapeMismatch);
    }
    let spaces = plan.mask_spaces()?;
    let mut rows_out = Vec::with_capacity(plan.instances());
    for head in 0..heads {
        let point = draw_point(chain, plan.shape.columns_variables())?;
        let target = consumer_rounding_target(&plan.shape, &point, &outputs[head])?;
        let variables = equality_variables(&point)?;
        let space = &spaces[head * 2];
        let entries = zero_entries(space);
        let mut challenges = ConsumerRawChallenges {
            inner: PositionPscChallenges::new(chain, CONSUMER_RAW_ROUNDING_TAGS, (head * 2) as u64),
        };
        challenges.send_claim(target)?;
        let output = verify_general(
            &variables,
            &[],
            target,
            &Summand::Map,
            None,
            &second.proofs[head * 2],
            entries.hiding.len(),
            &mut challenges,
        )?;
        if second.point_values[head * 2] != output.ends[0].value {
            return Err(Error::InvalidProof);
        }
        rows_out.push(obligations(plan, head * 2, output)?);

        let bit_point = draw_point(chain, plan.shape.bits_variables())?;
        let bit_variables = equality_variables(&bit_point)?;
        let space = &spaces[head * 2 + 1];
        let entries = zero_entries(space);
        let mut challenges = ConsumerRawChallenges {
            inner: PositionPscChallenges::new(chain, CONSUMER_RAW_BITS_TAGS, (head * 2 + 1) as u64),
        };
        challenges.send_claim(Fp3::ZERO)?;
        let output = verify_general(
            &bit_variables,
            &[],
            Fp3::ZERO,
            &Summand::Bits,
            None,
            &second.proofs[head * 2 + 1],
            entries.hiding.len(),
            &mut challenges,
        )?;
        if second.point_values[head * 2 + 1] != output.ends[0].value {
            return Err(Error::InvalidProof);
        }
        rows_out.push(obligations(plan, head * 2 + 1, output)?);
    }
    Ok(rows_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_core::common::field::PrimeField64;
    use zkie_engine::pcs::request::code::{build_code_plan, CodeParams};
    use zkie_engine::pcs::request::field::fp_from_u64;
    use zkie_engine::pcs::request::fold::{ByteMaskCoins, MaskCoins};

    use crate::proof::consumer_request::ConsumerIntegerRanges;
    use crate::proof::output_closure::{commit_value_group, CommittedValueGroup};

    const HEADS: usize = 2;
    const POSITIONS: usize = 3;
    const WIDTH: usize = 4;
    const COLUMNS: usize = 16;
    const SHIFT: u32 = 16;

    fn fp(value: u64) -> Fp3 {
        Fp3::from_base(fp_from_u64(value))
    }

    fn shape() -> ConsumerRawShape {
        ConsumerRawShape {
            heads: HEADS as u32,
            positions: POSITIONS as u32,
            width: WIDTH as u64,
            shift: SHIFT,
            columns: COLUMNS as u64,
        }
    }

    fn code() -> CodeParams {
        build_code_plan(12, 32, 4).unwrap()
    }

    fn wshape(columns: usize, tree: u64) -> WShape {
        WShape::new(
            columns as u64,
            code(),
            1,
            HEADS as u32,
            40,
            tree,
            1,
            Digest([0x41; 32]),
            Digest([0x51; 32]),
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

    struct Fixture {
        plan: ConsumerRawPlan,
        first: ConsumerRawFirst,
        raw: Vec<Vec<Fp3>>,
        bits: Vec<Vec<Fp3>>,
        outputs: Vec<Vec<i64>>,
        masks: ConsumerRawMaskWitness,
    }

    fn fixture() -> Fixture {
        let real = POSITIONS * WIDTH;
        let mut raw_int = vec![vec![0i64; COLUMNS]; HEADS];
        let mut outputs = vec![vec![0i64; real]; HEADS];
        for head in 0..HEADS {
            for entry in 0..real {
                raw_int[head][entry] =
                    ((entry as i64 + 1) * 7919 + head as i64 * 104729) % 1_000_003;
                outputs[head][entry] = i64::try_from(
                    crate::proof::relations::rq(i128::from(raw_int[head][entry]), SHIFT).unwrap(),
                )
                .unwrap();
            }
        }
        let raw: Vec<Vec<Fp3>> = raw_int
            .iter()
            .map(|head| {
                let mut values = vec![Fp3::ZERO; COLUMNS];
                for (entry, value) in head.iter().take(real).enumerate() {
                    values[entry] = signed_fp3(i128::from(*value)).unwrap();
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
        let mut coins = ByteMaskCoins::new(Digest([0x77; 32])).unwrap();
        let raw_shape = wshape(COLUMNS, 700);
        let bits_shape = wshape(COLUMNS * SHIFT as usize, 701);
        let raw_group = commit(&raw_shape, 20, 41, &raw, &mut coins);
        let bits_group = commit(&bits_shape, 30, 42, &bits, &mut coins);
        let mut plan = ConsumerRawPlan {
            shape: shape(),
            raw: ConsumerRawGroup {
                shape: raw_shape,
                columns: COLUMNS as u64,
                group: 41,
                oracle_base: 20,
                openings: 1,
                roots: raw_group.roots.clone(),
            },
            bits: ConsumerRawGroup {
                shape: bits_shape,
                columns: (COLUMNS * SHIFT as usize) as u64,
                group: 42,
                oracle_base: 30,
                openings: 1,
                roots: bits_group.roots.clone(),
            },
            mask: ConsumerRawMaskPlan {
                oracle: 60,
                base_slot: 100,
            },
            outputs_digest: Digest([0x99; 32]),
            bound: ConsumerIntegerBound::new(
                8,
                ConsumerIntegerRanges {
                    x_abs_bound: 1 << 20,
                    weight_lo: -(1 << 18),
                    weight_hi: (1 << 18) - 1,
                },
                SHIFT,
            )
            .unwrap(),
        };
        plan.validate().unwrap();
        let first = ConsumerRawFirst::from_groups(&raw_group.group, &bits_group.group);
        let spaces = plan.mask_spaces().unwrap();
        let mut masks = ConsumerRawMaskWitness::default();
        let mut counter = 1u64;
        for space in &spaces {
            for slot in &space.hiding {
                masks.insert(*slot, fp(counter));
                counter += 1;
            }
            masks.insert(space.shown, fp(counter));
            counter += 1;
        }
        Fixture {
            plan,
            first,
            raw,
            bits,
            outputs,
            masks,
        }
    }

    fn chains() -> (Chain, Chain) {
        let begin = || Chain::begin(120, &Digest::ZERO, b"consumer raw", b"fixture", None).unwrap();
        (begin(), begin())
    }

    fn prove(
        fixture: &Fixture,
        raw: &[Vec<Fp3>],
        bits: &[Vec<Fp3>],
        outputs: &[Vec<i64>],
    ) -> Result<ConsumerRawSecond> {
        let mut chain = chains().0;
        let (second, _) = prove_consumer_raw(
            &mut chain,
            &fixture.plan,
            &fixture.first,
            raw,
            bits,
            outputs,
            &fixture.masks,
        )?;
        Ok(second)
    }

    #[test]
    fn the_hidden_rounding_relation_passes_and_leaves_rows() {
        let fixture = fixture();
        let (mut prover, mut verifier) = chains();
        let (second, proven_rows) = prove_consumer_raw(
            &mut prover,
            &fixture.plan,
            &fixture.first,
            &fixture.raw,
            &fixture.bits,
            &fixture.outputs,
            &fixture.masks,
        )
        .unwrap();
        let verified_rows = verify_consumer_raw(
            &mut verifier,
            &fixture.plan,
            &fixture.first,
            &fixture.outputs,
            &second,
        )
        .unwrap();
        assert_eq!(proven_rows, verified_rows);
        assert_eq!(proven_rows.len(), fixture.plan.instances());
        let spaces = fixture.plan.mask_spaces().unwrap();
        for (row, space) in proven_rows.iter().zip(&spaces) {
            assert_eq!(row.hiding_slots, space.hiding);
            assert_eq!(row.shown_slot, space.shown);
            assert_eq!(row.output.ends.len(), 1);
            assert_eq!(row.row.constant, row.output.row_target);
            // The (E) end form holds on the same head's committed witness and
            // this instance's shown slot.
            let end = &row.end;
            assert_eq!(
                end.raw_oracle,
                fixture.plan.raw.oracle_base + u64::from(end.head)
            );
            assert_eq!(
                end.bits_oracle,
                fixture.plan.bits.oracle_base + u64::from(end.head)
            );
            let mut value = Fp3::ZERO;
            for (coefficient, entry) in end.raw_columns.iter().zip(&fixture.raw[end.head as usize])
            {
                value = value.add(&coefficient.mul(entry));
            }
            for (coefficient, entry) in end
                .bits_columns
                .iter()
                .zip(&fixture.bits[end.head as usize])
            {
                value = value.add(&coefficient.mul(entry));
            }
            value = value.add(
                &end.mask_scale
                    .mul(&fixture.masks.entry(end.shown_slot).unwrap()),
            );
            assert_eq!(value, end.target);
        }
        let bytes = second.encode(&fixture.plan).unwrap();
        let decoded = ConsumerRawSecond::decode(&fixture.plan, &bytes).unwrap();
        assert_eq!(decoded, second);
        let mut fresh = chains().1;
        verify_consumer_raw(
            &mut fresh,
            &fixture.plan,
            &fixture.first,
            &fixture.outputs,
            &decoded,
        )
        .unwrap();
    }

    #[test]
    fn a_changed_raw_bit_or_output_is_refused() {
        let fixture = fixture();
        let _ = prove(&fixture, &fixture.raw, &fixture.bits, &fixture.outputs).unwrap();
        let mut raw = fixture.raw.clone();
        raw[0][0] = raw[0][0].add(&fp(1));
        assert!(prove(&fixture, &raw, &fixture.bits, &fixture.outputs).is_err());
        let mut bits = fixture.bits.clone();
        bits[0][0] = if bits[0][0] == Fp3::ZERO {
            fp(1)
        } else {
            Fp3::ZERO
        };
        assert!(prove(&fixture, &fixture.raw, &bits, &fixture.outputs).is_err());
        let mut bits = fixture.bits.clone();
        bits[0][(POSITIONS * WIDTH) * SHIFT as usize] = fp(2);
        assert!(prove(&fixture, &fixture.raw, &bits, &fixture.outputs).is_err());
        let second = prove(&fixture, &fixture.raw, &fixture.bits, &fixture.outputs).unwrap();
        // The honest rows hold on the committed one-time mask entries.
        let (mut prover, _) = chains();
        let (_, honest_rows) = prove_consumer_raw(
            &mut prover,
            &fixture.plan,
            &fixture.first,
            &fixture.raw,
            &fixture.bits,
            &fixture.outputs,
            &fixture.masks,
        )
        .unwrap();
        for row in &honest_rows {
            let hiding: Vec<Fp3> = row
                .hiding_slots
                .iter()
                .map(|slot| fixture.masks.entry(*slot).unwrap())
                .collect();
            assert_eq!(
                row.row
                    .evaluate(&hiding, &|_| Err(Error::ShapeMismatch), &[])
                    .unwrap(),
                row.row.constant
            );
        }
        // Changed public outputs derive different row targets; the same
        // committed mask entries no longer satisfy them, so the group check
        // rejects the relation.
        let mut outputs = fixture.outputs.clone();
        outputs[0][0] += 1;
        let mut verifier = chains().1;
        let changed_rows = verify_consumer_raw(
            &mut verifier,
            &fixture.plan,
            &fixture.first,
            &outputs,
            &second,
        )
        .unwrap();
        assert_ne!(changed_rows, honest_rows);
        for row in &changed_rows {
            let hiding: Vec<Fp3> = row
                .hiding_slots
                .iter()
                .map(|slot| fixture.masks.entry(*slot).unwrap())
                .collect();
            assert_ne!(
                row.row
                    .evaluate(&hiding, &|_| Err(Error::ShapeMismatch), &[])
                    .unwrap(),
                row.row.constant
            );
            let end = &row.end;
            let mut value = Fp3::ZERO;
            for (coefficient, entry) in end.raw_columns.iter().zip(&fixture.raw[end.head as usize])
            {
                value = value.add(&coefficient.mul(entry));
            }
            for (coefficient, entry) in end
                .bits_columns
                .iter()
                .zip(&fixture.bits[end.head as usize])
            {
                value = value.add(&coefficient.mul(entry));
            }
            value = value.add(
                &end.mask_scale
                    .mul(&fixture.masks.entry(end.shown_slot).unwrap()),
            );
            assert_ne!(value, end.target);
        }
    }

    #[test]
    fn the_first_declaration_and_mask_slots_are_fixed() {
        let fixture = fixture();
        assert_eq!(
            ConsumerRawFirst::decode(&fixture.first.encode().unwrap()).unwrap(),
            fixture.first
        );
        let mut changed = fixture.first.clone();
        changed.raw.roots[0] = Digest([0x66; 32]);
        assert!(changed.check(&fixture.plan).is_err());
        let mut changed = fixture.first.clone();
        changed.bits.group.group += 1;
        assert!(changed.encode().is_ok());
        // The slot allocation is dense from the plan's base and every instance
        // takes its own block.
        let spaces = fixture.plan.mask_spaces().unwrap();
        let mut next = fixture.plan.mask.base_slot;
        for space in &spaces {
            for slot in &space.hiding {
                assert_eq!(*slot, next);
                next += 1;
            }
            assert_eq!(space.shown, next);
            next += 1;
        }
        let mut shifted = fixture.plan.clone();
        shifted.mask.base_slot += 7;
        let shifted_spaces = shifted.mask_spaces().unwrap();
        assert_eq!(shifted_spaces[0].hiding[0], fixture.plan.mask.base_slot + 7);
        // A changed group root or a broken shape is refused.
        let mut broken = fixture.plan.clone();
        broken.raw.roots.clear();
        assert!(broken.validate().is_err());
        let mut broken = fixture.plan.clone();
        broken.shape.columns = 12;
        assert!(broken.validate().is_err());
        let mut broken = fixture.plan;
        broken.shape.shift = 12;
        assert!(broken.validate().is_err());
    }

    #[test]
    fn the_public_target_uses_the_verified_outputs() {
        let fixture = fixture();
        let point = vec![fp(2), fp(3), fp(5), fp(7)];
        let target =
            consumer_rounding_target(&fixture.plan.shape, &point, &fixture.outputs[0]).unwrap();
        let mut outputs = fixture.outputs[0].clone();
        outputs[0] += 1;
        assert_ne!(
            consumer_rounding_target(&fixture.plan.shape, &point, &outputs).unwrap(),
            target
        );
        assert!(
            consumer_rounding_target(&fixture.plan.shape, &point, &fixture.outputs[0][..3])
                .is_err()
        );
    }
}
