//! Byte-range lookup by product trees, over one public table.
//!
//! The lookup proves that every committed key is a row of the public byte table
//! `0, 1, .., 255`, following the product-tree lookup: for a challenge alpha,
//!
//! ```text
//! prod_i (alpha - K[i]) = prod_t (alpha - tau(t))^(m_t),
//! ```
//!
//! with multiplicities `m_t` committed bit by bit. The key side and the table
//! side are product trees; each level is a general masked sumcheck over the two
//! halves of the next level, and the halving line reduces the two shown half
//! claims to one claim on the next level. The tie combines the table roots and
//! the key root; the multiplicity bits are checked by a bit summand and the
//! multiplicity sum row.
//!
//! Scope: one table, one key part, no shift, no padding. The key and
//! multiplicity messages are supplied by the caller; this module produces
//! claims and rows but does not yet bind them to committed groups. That wiring,
//! and the norm digit rows, is the next step.

use zkie_core::common::field::PrimeField64;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::bat::{BatPart, BatRow, BatRowKind};
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};

use crate::proof::hidden_send_session::WShape;
use crate::proof::lookup_messages::{
    commit_lookup, LookupCommitPlan, LookupMessages, LookupRandomSource, LookupRoots,
};
use crate::proof::psc::Weight;
use crate::proof::psc_general::{
    general_variables, line, prove_general, verify_general, GeneralChallenges, GeneralClaim,
    GeneralMaskEntries, GeneralOutput, GeneralProof, InnerSumcheck, MaskTerm, ScalarInner, Summand,
    TableAt, WeightedBitsInner, WeightedProductInner,
};

/// Version of the lookup plan and messages.
pub const LOOKUP_VERSION: u64 = 1;

/// The public byte table, as a multilinear extension over eight coordinates.
#[derive(Clone, Copy, Debug, Default)]
pub struct ByteTable;

impl TableAt for ByteTable {
    fn at(&self, point: &[Fp3]) -> Result<Fp3> {
        if point.len() != 8 {
            return Err(Error::ShapeMismatch);
        }
        let mut values = (0..256u64).map(Fp3::from_u64).collect::<Vec<_>>();
        for coordinate in point {
            let half = values.len() / 2;
            for index in 0..half {
                let low = values[index];
                let high = values[index + half];
                values[index] = low.add(&coordinate.mul(&high.sub(&low)));
            }
            values.truncate(half);
        }
        values.first().copied().ok_or(Error::ShapeMismatch)
    }
}

/// The public plan of one lookup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupPlan {
    /// Number of key entries is `2^key_bits`.
    pub key_bits: u32,
    /// Multiplicity planes are `2^mu`; each multiplicity must be below
    /// `2^(2^mu)`.
    pub mu: u32,
    /// The table has `2^table_log` rows; the byte table fixes eight.
    pub table_log: u32,
}

impl LookupPlan {
    pub fn validate(&self) -> Result<()> {
        if self.table_log != 8 || self.key_bits == 0 || self.mu == 0 {
            return Err(Error::InvalidConfig);
        }
        let keys = 1u64
            .checked_shl(self.key_bits)
            .ok_or(Error::ResourceLimit)?;
        let planes = 1usize
            .checked_shl(self.mu)
            .ok_or(Error::ResourceLimit)?;
        if keys >= (1u64 << planes.min(63)) {
            return Err(Error::InvalidConfig);
        }
        let table = 1usize
            .checked_shl(self.table_log)
            .ok_or(Error::ResourceLimit)?;
        planes
            .checked_mul(table)
            .and_then(|value| value.checked_mul(1usize << self.key_bits))
            .ok_or(Error::ResourceLimit)?;
        Ok(())
    }

    pub fn planes(&self) -> Result<usize> {
        self.validate()?;
        1usize.checked_shl(self.mu).ok_or(Error::ResourceLimit)
    }

    pub fn table_rows(&self) -> Result<usize> {
        1usize
            .checked_shl(self.table_log)
            .ok_or(Error::ResourceLimit)
    }

    pub fn key_count(&self) -> Result<usize> {
        1usize
            .checked_shl(self.key_bits)
            .ok_or(Error::ResourceLimit)
    }

    pub fn digest(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[..8].copy_from_slice(&LOOKUP_VERSION.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.key_bits.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.mu.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.table_log.to_le_bytes());
        bytes
    }
}

/// One key's physical source in a committed rounding message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LookupKeySource {
    pub oracle: u64,
    pub position: u32,
    pub row: u32,
    pub place: u32,
    pub coefficient: Fp3,
    pub offset: Fp3,
}

/// The trusted map from each key index to a committed rounding cell, plus the
/// padding value. The key vector itself is never committed as a message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupKeyPlan {
    pub sources: Vec<LookupKeySource>,
    pub padding_value: Fp3,
    pub table_identity: Digest,
}

impl LookupKeyPlan {
    pub fn validate(&self, plan: &LookupPlan) -> Result<()> {
        let keys = plan.key_count()?;
        if self.sources.len() > keys || self.table_identity == Digest::ZERO {
            return Err(Error::InvalidConfig);
        }
        let mut seen = std::collections::HashSet::new();
        for source in &self.sources {
            if source.oracle == 0
                || !seen.insert((source.oracle, source.position, source.row, source.place))
            {
                return Err(Error::InvalidConfig);
            }
        }
        Ok(())
    }

    /// Reads the key vector from the committed rounding rows. `rows[position]`
    /// is the unencoded message of that position; a source names its row and
    /// place directly.
    pub fn read(&self, plan: &LookupPlan, rows: &[Vec<Vec<Fp3>>]) -> Result<Vec<Fp3>> {
        self.validate(plan)?;
        let count = plan.key_count()?;
        let mut keys = Vec::with_capacity(count);
        for source in &self.sources {
            let value = rows
                .get(source.position as usize)
                .and_then(|message| message.get(source.row as usize))
                .and_then(|row| row.get(source.place as usize))
                .copied()
                .ok_or(Error::ShapeMismatch)?;
            keys.push(source.coefficient.mul(&value).add(&source.offset));
        }
        while keys.len() < count {
            keys.push(self.padding_value);
        }
        Ok(keys)
    }
}

/// One key's contribution to the linear claim on its committed rounding cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeyLinearTerm {
    pub oracle: u64,
    pub position: u32,
    pub row: u32,
    pub place: u32,
    pub coefficient: Fp3,
}

/// The key-leaf claim expanded into linear terms on the committed rounding
/// cells, a public padding constant and the mask terms of the claim. The
/// caller registers each term on its physical member; no virtual tree or root
/// is involved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyClaimExpansion {
    pub terms: Vec<KeyLinearTerm>,
    pub constant: Fp3,
    pub masks: Vec<MaskTerm>,
    pub value: Fp3,
}

impl LookupKeyPlan {
    /// Expands the key-leaf claim at its random point into linear claims on the
    /// committed rounding cells named by this plan.
    pub fn expand_claim(
        &self,
        plan: &LookupPlan,
        claim: &GeneralClaim,
    ) -> Result<KeyClaimExpansion> {
        self.validate(plan)?;
        let keys = plan.key_count()?;
        let eq = eq_weights(&claim.point);
        if eq.len() != keys {
            return Err(Error::ShapeMismatch);
        }
        let mut terms = Vec::with_capacity(self.sources.len());
        let mut constant = Fp3::ZERO;
        for (index, source) in self.sources.iter().enumerate() {
            let weight = eq[index];
            terms.push(KeyLinearTerm {
                oracle: source.oracle,
                position: source.position,
                row: source.row,
                place: source.place,
                coefficient: weight.mul(&source.coefficient),
            });
            constant = constant.add(&weight.mul(&source.offset));
        }
        for weight in eq.iter().skip(self.sources.len()) {
            constant = constant.add(&weight.mul(&self.padding_value));
        }
        Ok(KeyClaimExpansion {
            terms,
            constant,
            masks: claim.masks.clone(),
            value: claim.value,
        })
    }
}

/// The physical member a key source is registered on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyClaimMember {
    pub source_oracle: u64,
    pub bat_oracle: u64,
    pub shape: WShape,
    pub columns: u64,
}

impl LookupKeyPlan {
    /// Builds the actual BAT row that carries the expanded key claim. Each
    /// source term is written on its physical member table; the public
    /// constant is moved into the target and every mask term becomes a unit
    /// on the mask member. The verifier builds the same row from the same
    /// plan and claim.
    pub fn registration_row(
        &self,
        plan: &LookupPlan,
        claim: &GeneralClaim,
        members: &[KeyClaimMember],
        mask_bat_oracle: u64,
        mask_index: &dyn Fn(usize) -> Result<u64>,
    ) -> Result<BatRow> {
        let expansion = self.expand_claim(plan, claim)?;
        let mut row = BatRow::new(BatRowKind::Plain);
        for member in members {
            let rows = usize::try_from(member.shape.main_rows).map_err(|_| Error::ResourceLimit)?;
            let ell = usize::try_from(member.shape.code.ell).map_err(|_| Error::ResourceLimit)?;
            let mut values = vec![Fp3::ZERO; rows * ell];
            let mut used = false;
            for term in &expansion.terms {
                if term.oracle != member.source_oracle {
                    continue;
                }
                let index = (term.row as usize)
                    .checked_mul(ell)
                    .and_then(|base| base.checked_add(term.place as usize))
                    .ok_or(Error::ResourceLimit)?;
                let entry = values.get_mut(index).ok_or(Error::ShapeMismatch)?;
                *entry = entry.add(&term.coefficient);
                used = true;
            }
            if used {
                row.add_term(member.bat_oracle, BatPart::Main, values);
            }
        }
        for mask in &expansion.masks {
            row.add_unit(
                mask_bat_oracle,
                BatPart::Mask,
                mask_index(mask.slot)?,
                mask.coefficient,
            );
        }
        row.target = expansion.value.sub(&expansion.constant);
        Ok(row)
    }
}

/// A deterministic challenge source for the standalone lookup. The send hooks
/// mix every prover message into the state, so a tampered message changes the
/// later challenges.
#[derive(Clone)]
pub struct LookupChallenges {
    state: u64,
}

impl LookupChallenges {
    pub fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn mix_u64(&mut self, value: u64) {
        self.state = self
            .state
            .rotate_left(13)
            .wrapping_add(value)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    fn mix(&mut self, value: Fp3) {
        for part in value.c {
            self.mix_u64(part.as_canonical_u64());
        }
    }

    pub fn absorb(&mut self, tag: u64, bytes: &[u8]) {
        self.mix_u64(tag);
        for byte in bytes {
            self.mix_u64(*byte as u64);
        }
    }
}

impl GeneralChallenges for LookupChallenges {
    fn draw(&mut self) -> Result<Fp3> {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        Ok(Fp3::from_u64(self.state))
    }

    fn draw_nonzero(&mut self) -> Result<Fp3> {
        for _ in 0..64 {
            let value = self.draw()?;
            if value != Fp3::ZERO {
                return Ok(value);
            }
        }
        Err(Error::EntropyUnavailable)
    }

    fn send_hiding_sum(&mut self, value: Fp3) -> Result<()> {
        self.mix_u64(1);
        self.mix(value);
        Ok(())
    }

    fn send_round(&mut self, index: usize, round: &crate::proof::psc::PscRound) -> Result<()> {
        self.mix_u64(2);
        self.mix_u64(index as u64);
        for coefficient in &round.coefficients {
            self.mix(*coefficient);
        }
        Ok(())
    }

    fn send_end_values(&mut self, values: &[Fp3]) -> Result<()> {
        self.mix_u64(3);
        for value in values {
            self.mix(*value);
        }
        Ok(())
    }
}

/// A cursor over the plan's mask entries. Prover and verifier allocate the
/// same slots in the same order.
#[derive(Clone, Debug, Default)]
pub struct MaskSlots {
    next: usize,
}

impl MaskSlots {
    pub fn take(&mut self, count: usize) -> Result<Vec<usize>> {
        let end = self
            .next
            .checked_add(count)
            .ok_or(Error::ResourceLimit)?;
        let slots = (self.next..end).collect();
        self.next = end;
        Ok(slots)
    }
}

/// The prover's mask entries in slot order.
#[derive(Clone, Debug)]
pub struct ProverMasks {
    pub values: Vec<Fp3>,
    pub slots: MaskSlots,
}

impl ProverMasks {
    fn from_entries(values: Vec<Fp3>) -> Self {
        Self {
            values,
            slots: MaskSlots::default(),
        }
    }

    fn value(&self, slot: usize) -> Result<Fp3> {
        self.values.get(slot).copied().ok_or(Error::InvalidIndex)
    }
}

/// The prover's witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupWitness {
    /// The multiplication bits, plane-major: `b[j + planes * t]`.
    pub bits: Vec<Fp3>,
}

/// The prover's messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupProof {
    pub tie: Vec<GeneralProof>,
    pub key_levels: Vec<GeneralProof>,
    pub table_levels: Vec<GeneralProof>,
    pub leaves: GeneralProof,
    pub bits: GeneralProof,
}

/// What the lookup leaves to its caller: the key claim and the multiplicity
/// sum row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupOutcome {
    /// `sum_i eq(point, i) K[i] + <masks> = value`, after the shift of alpha.
    pub key: GeneralClaim,
    /// Coefficients on the multiplicity bits and the number of keys.
    pub multiplicity: (Vec<Fp3>, Fp3),
    /// The `(U)` and end claims of every sumcheck, for the caller's rows.
    pub rows: Vec<GeneralOutput>,
}

fn tree_levels(values: &[Fp3], fixed: usize, variables: usize) -> Vec<Vec<Fp3>> {
    let mut levels = vec![Vec::new(); variables + 1];
    levels[variables] = values.to_vec();
    let mut current = values.to_vec();
    for level in (0..variables).rev() {
        let half = 1usize
            .checked_shl((fixed + level) as u32)
            .expect("tree level fits");
        let mut next = vec![Fp3::ZERO; half];
        for index in 0..half {
            next[index] = current[index].mul(&current[index + half]);
        }
        levels[level] = next.clone();
        current = next;
    }
    levels
}

/// The equality weights of a point, in little-endian index order: the last
/// processed coordinate is the least significant index bit. This matches the
/// sumcheck's variable order, where the first point coordinate is the least
/// significant bit of the map index.
fn eq_weights(point: &[Fp3]) -> Vec<Fp3> {
    let mut weights = vec![Fp3::ONE];
    for coordinate in point.iter().rev() {
        let mut next = vec![Fp3::ZERO; weights.len() * 2];
        for (index, weight) in weights.iter().enumerate() {
            next[2 * index] = weight.mul(&Fp3::ONE.sub(coordinate));
            next[2 * index + 1] = weight.mul(coordinate);
        }
        weights = next;
    }
    weights
}

fn end_claim(output: &GeneralOutput, index: usize, slot: usize) -> Result<GeneralClaim> {
    let end = output.ends.get(index).ok_or(Error::ShapeMismatch)?;
    Ok(GeneralClaim {
        point: end.point.clone(),
        value: end.value,
        masks: vec![MaskTerm {
            slot,
            coefficient: end.mask_scale,
        }],
    })
}

fn claim_masks_negated(claim: &GeneralClaim) -> Vec<MaskTerm> {
    claim
        .masks
        .iter()
        .map(|term| MaskTerm {
            slot: term.slot,
            coefficient: Fp3::ZERO.sub(&term.coefficient),
        })
        .collect()
}

struct ProverStep {
    proof: GeneralProof,
    output: GeneralOutput,
    shown: Vec<usize>,
}

fn prove_step(
    masks: &mut ProverMasks,
    challenges: &mut LookupChallenges,
    variables: &[Weight],
    claim_masks: &[MaskTerm],
    target: Fp3,
    summand: &Summand,
    table: Option<&dyn TableAt>,
    inner: &mut dyn InnerSumcheck,
) -> Result<ProverStep> {
    let all_variables = general_variables(variables, summand)?;
    let hiding = crate::proof::psc::hp_entries(&all_variables)?;
    let shown = summand.shown();
    let hiding_slots = masks.slots.take(hiding)?;
    let shown_slots = masks.slots.take(shown)?;
    let mut hiding_values = Vec::with_capacity(hiding);
    for slot in &hiding_slots {
        hiding_values.push(masks.value(*slot)?);
    }
    let mut shown_values = Vec::with_capacity(shown);
    for slot in &shown_slots {
        shown_values.push(masks.value(*slot)?);
    }
    let entries = GeneralMaskEntries {
        hiding: hiding_values,
        shown: shown_values,
    };
    let provider = |slot: usize| masks.value(slot);
    let (proof, output) = prove_general(
        variables,
        claim_masks,
        &provider,
        target,
        summand,
        table,
        inner,
        &entries,
        challenges,
    )?;
    Ok(ProverStep {
        proof,
        output,
        shown: shown_slots,
    })
}

fn verify_step(
    slots: &mut MaskSlots,
    challenges: &mut LookupChallenges,
    variables: &[Weight],
    claim_masks: &[MaskTerm],
    target: Fp3,
    summand: &Summand,
    table: Option<&dyn TableAt>,
    proof: &GeneralProof,
) -> Result<(GeneralOutput, Vec<usize>)> {
    let all_variables = general_variables(variables, summand)?;
    let hiding = crate::proof::psc::hp_entries(&all_variables)?;
    let shown = summand.shown();
    let _hiding_slots = slots.take(hiding)?;
    let shown_slots = slots.take(shown)?;
    let output = verify_general(
        variables,
        claim_masks,
        target,
        summand,
        table,
        proof,
        hiding,
        challenges,
    )?;
    Ok((output, shown_slots))
}

fn key_and_table_levels(
    plan: &LookupPlan,
    keys_vec: &[Fp3],
    witness: &LookupWitness,
    alpha: Fp3,
) -> Result<(Vec<Vec<Fp3>>, Vec<Vec<Fp3>>)> {
    let keys = plan.key_count()?;
    let planes = plan.planes()?;
    let rows = plan.table_rows()?;
    if keys_vec.len() != keys
        || witness.bits.len() != planes * rows
        || witness.bits.iter().any(|bit| *bit != Fp3::ZERO && *bit != Fp3::ONE)
    {
        return Err(Error::ShapeMismatch);
    }
    let key_values = keys_vec
        .iter()
        .map(|key| alpha.sub(key))
        .collect::<Vec<_>>();
    let key_levels = tree_levels(&key_values, 0, plan.key_bits as usize);
    let table = (0..rows).map(|index| Fp3::from_u64(index as u64)).collect::<Vec<_>>();
    let mut leaves = vec![Fp3::ZERO; planes * rows];
    for plane in 0..planes {
        for row in 0..rows {
            let bit = witness.bits[plane + planes * row];
            leaves[plane + planes * row] = Fp3::ONE.add(
                &bit.mul(&alpha.sub(&Fp3::ONE).sub(&table[row])),
            );
        }
    }
    let table_levels = tree_levels(&leaves, plan.mu as usize, plan.table_log as usize);
    Ok((key_levels, table_levels))
}

fn table_merge_claim(
    planes: &[GeneralClaim],
    point: &[Fp3],
) -> Result<GeneralClaim> {
    let weights = eq_weights(point);
    if weights.len() != planes.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut value = Fp3::ZERO;
    let mut masks = Vec::new();
    for (weight, claim) in weights.iter().zip(planes) {
        value = value.add(&weight.mul(&claim.value));
        for term in &claim.masks {
            masks.push(MaskTerm {
                slot: term.slot,
                coefficient: weight.mul(&term.coefficient),
            });
        }
    }
    Ok(GeneralClaim {
        point: point.to_vec(),
        value,
        masks,
    })
}

fn equality_variables(point: &[Fp3], plane_count: usize) -> Result<Vec<Weight>> {
    point
        .iter()
        .enumerate()
        .map(|(index, coordinate)| {
            let degree = if index < plane_count { 2 } else { 3 };
            Weight::equality(*coordinate, degree)
        })
        .collect()
}

/// Commits the lookup messages and proves the byte lookup for one key vector.
pub fn prove_lookup(
    plan: &LookupPlan,
    commit_plan: &LookupCommitPlan,
    key_plan: &LookupKeyPlan,
    key_rows: &[Vec<Vec<Fp3>>],
    witness: &LookupWitness,
    seed: u64,
) -> Result<(LookupMessages, LookupProof, LookupOutcome)> {
    plan.validate()?;
    let mask_entries = mask_count(plan)?;
    commit_plan.validate(witness.bits.len(), mask_entries)?;
    let mut random_source = LookupRandomSource::new(seed ^ 0x5A5A_5A5A);
    let messages = commit_lookup(
        commit_plan,
        &witness.bits,
        mask_entries,
        &mut random_source,
    )?;
    let (proof, outcome) = prove_lookup_with_messages(
        plan,
        commit_plan,
        key_plan,
        key_rows,
        witness,
        &messages,
        seed,
    )?;
    Ok((messages, proof, outcome))
}

/// Proves the byte lookup on messages that are already committed and sampled
/// by the enclosing session. The challenges are drawn from the transcript in
/// the fixed order: plan and message roots first, then the lookup challenge,
/// then the product-sumcheck transcript.
pub fn prove_lookup_with_messages(
    plan: &LookupPlan,
    commit_plan: &LookupCommitPlan,
    key_plan: &LookupKeyPlan,
    key_rows: &[Vec<Vec<Fp3>>],
    witness: &LookupWitness,
    messages: &LookupMessages,
    seed: u64,
) -> Result<(LookupProof, LookupOutcome)> {
    plan.validate()?;
    let keys = plan.key_count()?;
    let planes = plan.planes()?;
    let rows = plan.table_rows()?;
    let key_values = key_plan.read(plan, key_rows)?;
    if key_values.len() != keys {
        return Err(Error::ShapeMismatch);
    }
    let mask_entries = mask_count(plan)?;
    commit_plan.validate(witness.bits.len(), mask_entries)?;
    let mask_rows = usize::try_from(commit_plan.mask_rows).map_err(|_| Error::ResourceLimit)?;
    let ell = usize::try_from(commit_plan.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
    let bit_rows = usize::try_from(commit_plan.bits_shape.main_rows)
        .map_err(|_| Error::ResourceLimit)?;
    if messages.bits.group.group != commit_plan.group
        || messages.bits.roots.len() != 1
        || messages.bits.roots[0] == Digest::ZERO
        || messages.bits.messages.len() != 1
        || messages.bits.messages[0].len() != bit_rows
        || messages.mask_root == Digest::ZERO
        || messages.mask_rows.len() != mask_rows
        || messages.mask_rows.iter().any(|row| row.len() != ell)
        || messages.mask_tails.len() != mask_rows
        || messages.mask_salts.is_empty()
    {
        return Err(Error::ShapeMismatch);
    }
    let mut challenges = LookupChallenges::new(seed);
    challenges.absorb(LOOKUP_VERSION, &plan.digest());
    challenges.absorb(commit_plan.bits_oracle, &messages.bits_root().0);
    challenges.absorb(commit_plan.mask_oracle, &messages.mask_root.0);
    let alpha = challenges.draw()?;
    let (key_levels, table_levels) = key_and_table_levels(plan, &key_values, witness, alpha)?;
    let mut mask_values = Vec::with_capacity(mask_entries);
    for slot in 0..mask_entries {
        mask_values.push(messages.mask_entry(commit_plan, slot)?);
    }
    let mut masks = ProverMasks::from_entries(mask_values);
    let mut rows_out = Vec::new();
    let table = ByteTable;

    let plane_roots = &table_levels[0];
    let planes_count = plane_roots.len();
    let key_root = key_levels[0][0];
    let mut power = vec![Fp3::ZERO; planes_count];
    power[planes_count - 1] = plane_roots[planes_count - 1];
    for index in (0..planes_count - 1).rev() {
        power[index] = power[index + 1]
            .mul(&power[index + 1])
            .mul(&plane_roots[index]);
    }

    let mut tie_proofs = Vec::new();
    let mut on_plane = Vec::new();
    let mut on_root: Option<GeneralClaim> = None;
    let mut on_power: Option<GeneralClaim> = None;
    for index in 0..planes_count - 1 {
        let (summand, values) = if index == 0 {
            (
                Summand::TieFirst { a: Fp3::ONE },
                vec![key_root, power[index + 1], plane_roots[index]],
            )
        } else {
            (
                Summand::TieNext,
                vec![power[index + 1], plane_roots[index]],
            )
        };
        let (claim_masks, target) = if index == 0 {
            (Vec::new(), Fp3::ZERO)
        } else {
            let claim = on_power.as_ref().ok_or(Error::InvalidSessionState)?;
            (claim.masks.clone(), claim.value)
        };
        let mut inner = ScalarInner::new(&values)?;
        let step = match prove_step(
            &mut masks,
            &mut challenges,
            &[],
            &claim_masks,
            target,
            &summand,
            None,
            &mut inner,
        ) {
            Ok(step) => step,
            Err(error) => return Err(error),
        };
        tie_proofs.push(step.proof.clone());
        rows_out.push(step.output.clone());
        let mut at = 0;
        if index == 0 {
            on_root = Some(end_claim(&step.output, at, step.shown[at])?);
            at += 1;
        }
        on_power = Some(end_claim(&step.output, at, step.shown[at])?);
        on_plane.push(end_claim(
            &step.output,
            at + 1,
            step.shown[at + 1],
        )?);
    }
    on_plane.push(on_power.ok_or(Error::InvalidSessionState)?);
    let on_root = on_root.ok_or(Error::InvalidSessionState)?;

    let plane_point = {
        let mut point = Vec::with_capacity(plan.mu as usize);
        for _ in 0..plan.mu {
            point.push(challenges.draw()?);
        }
        point
    };
    let on_table = table_merge_claim(&on_plane, &plane_point)?;

    let mut key_proofs = Vec::new();
    let mut key_claim = on_root;
    for level in 0..plan.key_bits as usize {
        let next = &key_levels[level + 1];
        let half = next.len() / 2;
        let mut inner = if level == 0 {
            let mut scalar = ScalarInner::new(&[next[0], next[1]])?;
            let step = prove_step(
                &mut masks,
                &mut challenges,
                &[],
                &key_claim.masks,
                key_claim.value,
                &Summand::Product,
                None,
                &mut scalar,
            )?;
            key_proofs.push(step.proof.clone());
            rows_out.push(step.output.clone());
            let half0 = end_claim(&step.output, 0, step.shown[0])?;
            let half1 = end_claim(&step.output, 1, step.shown[1])?;
            key_claim = line(&mut challenges, half0, half1)?;
            continue;
        } else {
            WeightedProductInner::new(
                &next[..half],
                &next[half..],
                &key_claim.point,
            )?
        };
        let variables = equality_variables(&key_claim.point, 0)?;
        let step = prove_step(
            &mut masks,
            &mut challenges,
            &variables,
            &key_claim.masks,
            key_claim.value,
            &Summand::Product,
            None,
            &mut inner,
        )?;
        key_proofs.push(step.proof.clone());
        rows_out.push(step.output.clone());
        let half0 = end_claim(&step.output, 0, step.shown[0])?;
        let half1 = end_claim(&step.output, 1, step.shown[1])?;
        key_claim = line(&mut challenges, half0, half1)?;
    }
    let on_key_leaves = key_claim;
    let key = GeneralClaim {
        point: on_key_leaves.point.clone(),
        value: alpha.sub(&on_key_leaves.value),
        masks: claim_masks_negated(&on_key_leaves),
    };

    let mut table_proofs = Vec::new();
    let mut table_claim = on_table;
    for level in 0..plan.table_log as usize {
        let next = &table_levels[level + 1];
        let half = next.len() / 2;
        let variables = equality_variables(&table_claim.point, 0)?;
        let mut inner =
            WeightedProductInner::new(&next[..half], &next[half..], &table_claim.point)?;
        let step = prove_step(
            &mut masks,
            &mut challenges,
            &variables,
            &table_claim.masks,
            table_claim.value,
            &Summand::Product,
            None,
            &mut inner,
        )?;
        table_proofs.push(step.proof.clone());
        rows_out.push(step.output.clone());
        let half0 = end_claim(&step.output, 0, step.shown[0])?;
        let half1 = end_claim(&step.output, 1, step.shown[1])?;
        table_claim = line(&mut challenges, half0, half1)?;
    }

    let leaves_variables = equality_variables(&table_claim.point, plan.mu as usize)?;
    let mut leaves_inner = crate::proof::psc_general::LeavesInner::new(
        &witness.bits,
        &(0..witness.bits.len())
            .map(|index| Fp3::from_u64((index / planes) as u64))
            .collect::<Vec<_>>(),
        &table_claim.point,
        alpha,
    )?;
    let leaves_step = prove_step(
        &mut masks,
        &mut challenges,
        &leaves_variables,
        &table_claim.masks,
        table_claim.value,
        &Summand::Leaves { alpha, mu: plan.mu as usize },
        Some(&table),
        &mut leaves_inner,
    )?;
    rows_out.push(leaves_step.output.clone());

    let bit_point = {
        let mut point = Vec::with_capacity(plan.mu as usize + plan.table_log as usize);
        for _ in 0..(plan.mu as usize + plan.table_log as usize) {
            point.push(challenges.draw()?);
        }
        point
    };
    let bit_variables = equality_variables(&bit_point, 0)?;
    let mut bits_inner = WeightedBitsInner::new(&witness.bits, &bit_point)?;
    let bits_step = prove_step(
        &mut masks,
        &mut challenges,
        &bit_variables,
        &[],
        Fp3::ZERO,
        &Summand::Bits,
        None,
        &mut bits_inner,
    )?;
    rows_out.push(bits_step.output.clone());

    let mut coefficients = vec![Fp3::ZERO; planes * rows];
    for plane in 0..planes {
        let weight = Fp3::from_u64(1u64 << plane);
        for row in 0..rows {
            coefficients[plane + planes * row] = weight;
        }
    }
    let multiplicity = (coefficients, Fp3::from_u64(keys as u64));

    Ok((
        LookupProof {
            tie: tie_proofs,
            key_levels: key_proofs,
            table_levels: table_proofs,
            leaves: leaves_step.proof,
            bits: bits_step.proof,
        },
        LookupOutcome {
            key,
            multiplicity,
            rows: rows_out,
        },
    ))
}

/// Verifies the lookup messages and returns the same claims and rows.
pub fn verify_lookup(
    plan: &LookupPlan,
    commit_plan: &LookupCommitPlan,
    roots: &LookupRoots,
    proof: &LookupProof,
    seed: u64,
) -> Result<LookupOutcome> {
    plan.validate()?;
    let planes = plan.planes()?;
    let table_rows = plan.table_rows()?;
    commit_plan.validate(planes * table_rows, mask_count(plan)?)?;
    let mut challenges = LookupChallenges::new(seed);
    challenges.absorb(LOOKUP_VERSION, &plan.digest());
    challenges.absorb(commit_plan.bits_oracle, &roots.bits.0);
    challenges.absorb(commit_plan.mask_oracle, &roots.mask.0);
    let alpha = challenges.draw()?;
    let mut slots = MaskSlots::default();
    let mut rows_out = Vec::new();
    let table = ByteTable;

    if proof.tie.len() != planes - 1 {
        return Err(Error::ShapeMismatch);
    }
    let mut on_plane = Vec::with_capacity(planes);
    let mut on_root: Option<GeneralClaim> = None;
    let mut on_power: Option<GeneralClaim> = None;
    for index in 0..planes - 1 {
        let summand = if index == 0 {
            Summand::TieFirst { a: Fp3::ONE }
        } else {
            Summand::TieNext
        };
        let (claim_masks, target) = if index == 0 {
            (Vec::new(), Fp3::ZERO)
        } else {
            let claim = on_power.as_ref().ok_or(Error::InvalidSessionState)?;
            (claim.masks.clone(), claim.value)
        };
        let (output, shown) = verify_step(
            &mut slots,
            &mut challenges,
            &[],
            &claim_masks,
            target,
            &summand,
            None,
            &proof.tie[index],
        )?;
        rows_out.push(output.clone());
        let mut at = 0;
        if index == 0 {
            on_root = Some(end_claim(&output, at, shown[at])?);
            at += 1;
        }
        on_power = Some(end_claim(&output, at, shown[at])?);
        on_plane.push(end_claim(&output, at + 1, shown[at + 1])?);
    }
    on_plane.push(on_power.ok_or(Error::InvalidSessionState)?);
    let on_root = on_root.ok_or(Error::InvalidSessionState)?;
    let plane_point = {
        let mut point = Vec::with_capacity(plan.mu as usize);
        for _ in 0..plan.mu {
            point.push(challenges.draw()?);
        }
        point
    };
    let on_table = table_merge_claim(&on_plane, &plane_point)?;

    if proof.key_levels.len() != plan.key_bits as usize
        || proof.table_levels.len() != plan.table_log as usize
    {
        return Err(Error::ShapeMismatch);
    }
    let mut key_claim = on_root;
    for level in 0..plan.key_bits as usize {
        let variables = equality_variables(&key_claim.point, 0)?;
        let (output, shown) = verify_step(
            &mut slots,
            &mut challenges,
            &variables,
            &key_claim.masks,
            key_claim.value,
            &Summand::Product,
            None,
            &proof.key_levels[level],
        )?;
        rows_out.push(output.clone());
        let half0 = end_claim(&output, 0, shown[0])?;
        let half1 = end_claim(&output, 1, shown[1])?;
        key_claim = line(&mut challenges, half0, half1)?;
    }
    let on_key_leaves = key_claim;
    let key = GeneralClaim {
        point: on_key_leaves.point.clone(),
        value: alpha.sub(&on_key_leaves.value),
        masks: claim_masks_negated(&on_key_leaves),
    };
    let mut table_claim = on_table;
    for level in 0..plan.table_log as usize {
        let variables = equality_variables(&table_claim.point, 0)?;
        let (output, shown) = verify_step(
            &mut slots,
            &mut challenges,
            &variables,
            &table_claim.masks,
            table_claim.value,
            &Summand::Product,
            None,
            &proof.table_levels[level],
        )?;
        rows_out.push(output.clone());
        let half0 = end_claim(&output, 0, shown[0])?;
        let half1 = end_claim(&output, 1, shown[1])?;
        table_claim = line(&mut challenges, half0, half1)?;
    }
    let leaves_variables = equality_variables(&table_claim.point, plan.mu as usize)?;
    let (leaves_output, _) = verify_step(
        &mut slots,
        &mut challenges,
        &leaves_variables,
        &table_claim.masks,
        table_claim.value,
        &Summand::Leaves { alpha, mu: plan.mu as usize },
        Some(&table),
        &proof.leaves,
    )?;
    rows_out.push(leaves_output);

    let bit_point = {
        let mut point = Vec::with_capacity(plan.mu as usize + plan.table_log as usize);
        for _ in 0..(plan.mu as usize + plan.table_log as usize) {
            point.push(challenges.draw()?);
        }
        point
    };
    let bit_variables = equality_variables(&bit_point, 0)?;
    let (bits_output, shown) = verify_step(
        &mut slots,
        &mut challenges,
        &bit_variables,
        &[],
        Fp3::ZERO,
        &Summand::Bits,
        None,
        &proof.bits,
    )?;
    rows_out.push(bits_output);
    let _ = shown;

    let mut coefficients = vec![Fp3::ZERO; planes * table_rows];
    for plane in 0..planes {
        let weight = Fp3::from_u64(1u64 << plane);
        for row in 0..table_rows {
            coefficients[plane + planes * row] = weight;
        }
    }
    let multiplicity = (
        coefficients,
        Fp3::from_u64(plan.key_count()? as u64),
    );

    Ok(LookupOutcome {
        key,
        multiplicity,
        rows: rows_out,
    })
}

/// The number of mask entries the lookup uses.
pub fn mask_count(plan: &LookupPlan) -> Result<usize> {
    plan.validate()?;
    let planes = plan.planes()?;
    let mut count = 0usize;
    for index in 0..planes - 1 {
        let summand = if index == 0 {
            Summand::TieFirst { a: Fp3::ONE }
        } else {
            Summand::TieNext
        };
        count = count
            .checked_add(crate::proof::psc::hp_entries(&general_variables(&[], &summand)?)?)
            .and_then(|value| value.checked_add(summand.shown()))
            .ok_or(Error::ResourceLimit)?;
    }
    for level in 0..plan.key_bits as usize {
        let variables = if level == 0 {
            Vec::new()
        } else {
            (0..level)
                .map(|_| Weight::equality(Fp3::ZERO, 3).unwrap())
                .collect()
        };
        let summand = Summand::Product;
        count = count
            .checked_add(crate::proof::psc::hp_entries(&general_variables(
                &variables, &summand,
            )?)?)
            .and_then(|value| value.checked_add(summand.shown()))
            .ok_or(Error::ResourceLimit)?;
    }
    for level in 0..plan.table_log as usize {
        let variables = (0..(plan.mu as usize + level))
            .map(|_| Weight::equality(Fp3::ZERO, 3).unwrap())
            .collect::<Vec<_>>();
        let summand = Summand::Product;
        count = count
            .checked_add(crate::proof::psc::hp_entries(&general_variables(
                &variables, &summand,
            )?)?)
            .and_then(|value| value.checked_add(summand.shown()))
            .ok_or(Error::ResourceLimit)?;
    }
    let leaves_variables = (0..(plan.mu as usize + plan.table_log as usize))
        .map(|index| {
            Weight::equality(
                Fp3::ZERO,
                if index < plan.mu as usize { 2 } else { 3 },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    count = count
        .checked_add(crate::proof::psc::hp_entries(&general_variables(
            &leaves_variables,
            &Summand::Leaves { alpha: Fp3::ZERO, mu: plan.mu as usize },
        )?)?)
        .ok_or(Error::ResourceLimit)?;
    let bits_variables = (0..(plan.mu as usize + plan.table_log as usize))
        .map(|_| Weight::equality(Fp3::ZERO, 3).unwrap())
        .collect::<Vec<_>>();
    count = count
        .checked_add(crate::proof::psc::hp_entries(&general_variables(
            &bits_variables,
            &Summand::Bits,
        )?)?)
        .and_then(|value| value.checked_add(1))
        .ok_or(Error::ResourceLimit)?;
    Ok(count)
}

fn put_round(writer: &mut CanonicalWriter, round: &crate::proof::psc::PscRound) -> Result<()> {
    writer.put_u64(round.coefficients.len() as u64);
    for coefficient in &round.coefficients {
        writer.put_fp3(coefficient);
    }
    Ok(())
}

fn get_round(reader: &mut CanonicalReader<'_>, expected: usize) -> Result<crate::proof::psc::PscRound> {
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if count != expected {
        return Err(Error::InvalidEncoding);
    }
    let mut coefficients = Vec::with_capacity(count);
    for _ in 0..count {
        coefficients.push(reader.get_fp3()?);
    }
    Ok(crate::proof::psc::PscRound { coefficients })
}

fn put_general(writer: &mut CanonicalWriter, proof: &GeneralProof) -> Result<()> {
    writer.put_fp3(&proof.hiding_sum);
    writer.put_u64(proof.rounds.len() as u64);
    for round in &proof.rounds {
        put_round(writer, round)?;
    }
    put_round(writer, &proof.y_round)?;
    writer.put_u64(proof.end_values.len() as u64);
    for value in &proof.end_values {
        writer.put_fp3(value);
    }
    Ok(())
}

fn get_general(
    reader: &mut CanonicalReader<'_>,
    rounds: usize,
    round_coefficients: &[usize],
    y_coefficients: usize,
    ends: usize,
) -> Result<GeneralProof> {
    let hiding_sum = reader.get_fp3()?;
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if count != rounds {
        return Err(Error::InvalidEncoding);
    }
    let mut parsed = Vec::with_capacity(count);
    for index in 0..count {
        parsed.push(get_round(
            reader,
            round_coefficients.get(index).copied().ok_or(Error::InvalidEncoding)?,
        )?);
    }
    let y_round = get_round(reader, y_coefficients)?;
    let end_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if end_count != ends {
        return Err(Error::InvalidEncoding);
    }
    let mut end_values = Vec::with_capacity(end_count);
    for _ in 0..end_count {
        end_values.push(reader.get_fp3()?);
    }
    Ok(GeneralProof {
        hiding_sum,
        rounds: parsed,
        y_round,
        end_values,
    })
}

impl LookupProof {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(LOOKUP_VERSION);
        writer.put_u64(self.tie.len() as u64);
        for proof in &self.tie {
            put_general(&mut writer, proof)?;
        }
        writer.put_u64(self.key_levels.len() as u64);
        for proof in &self.key_levels {
            put_general(&mut writer, proof)?;
        }
        writer.put_u64(self.table_levels.len() as u64);
        for proof in &self.table_levels {
            put_general(&mut writer, proof)?;
        }
        put_general(&mut writer, &self.leaves)?;
        put_general(&mut writer, &self.bits)?;
        Ok(writer.into_bytes())
    }

    pub fn decode(plan: &LookupPlan, bytes: &[u8]) -> Result<Self> {
        plan.validate()?;
        let planes = plan.planes()?;
        let all = plan.mu as usize + plan.table_log as usize;
        let mut reader = CanonicalReader::new(bytes, zkie_core::common::codec::DecodeLimits::unlimited());
        if reader.get_u64()? != LOOKUP_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let tie_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if tie_count != planes - 1 {
            return Err(Error::InvalidEncoding);
        }
        let mut tie = Vec::with_capacity(tie_count);
        for index in 0..tie_count {
            let (shown, degree) = if index == 0 { (3, 3) } else { (2, 3) };
            tie.push(get_general(&mut reader, 0, &[], degree, shown)?);
        }
        let key_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if key_count != plan.key_bits as usize {
            return Err(Error::InvalidEncoding);
        }
        let mut key_levels = Vec::with_capacity(key_count);
        for level in 0..key_count {
            let rounds = if level == 0 { 0 } else { level };
            let coefficients = vec![2usize; rounds];
            key_levels.push(get_general(&mut reader, rounds, &coefficients, 2, 2)?);
        }
        let table_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if table_count != plan.table_log as usize {
            return Err(Error::InvalidEncoding);
        }
        let mut table_levels = Vec::with_capacity(table_count);
        for level in 0..table_count {
            let rounds = plan.mu as usize + level;
            let coefficients = vec![2usize; rounds];
            table_levels.push(get_general(&mut reader, rounds, &coefficients, 2, 2)?);
        }
        let leaves_coefficients = (0..all)
            .map(|index| if index < plan.mu as usize { 1 } else { 2 })
            .collect::<Vec<_>>();
        let leaves = get_general(&mut reader, all, &leaves_coefficients, 0, 0)?;
        let bits_coefficients = vec![2usize; all];
        let bits = get_general(&mut reader, all, &bits_coefficients, 2, 1)?;
        reader.finish()?;
        Ok(Self {
            tie,
            key_levels,
            table_levels,
            leaves,
            bits,
        })
    }
}

fn put_claim(writer: &mut CanonicalWriter, claim: &GeneralClaim) {
    writer.put_u64(claim.point.len() as u64);
    for coordinate in &claim.point {
        writer.put_fp3(coordinate);
    }
    writer.put_fp3(&claim.value);
    writer.put_u64(claim.masks.len() as u64);
    for term in &claim.masks {
        writer.put_u64(term.slot as u64);
        writer.put_fp3(&term.coefficient);
    }
}

fn get_claim(
    reader: &mut CanonicalReader<'_>,
    max_point: usize,
    max_masks: usize,
) -> Result<GeneralClaim> {
    let point_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if point_count > max_point {
        return Err(Error::InvalidEncoding);
    }
    let mut point = Vec::with_capacity(point_count);
    for _ in 0..point_count {
        point.push(reader.get_fp3()?);
    }
    let value = reader.get_fp3()?;
    let mask_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if mask_count > max_masks {
        return Err(Error::InvalidEncoding);
    }
    let mut masks = Vec::with_capacity(mask_count);
    for _ in 0..mask_count {
        masks.push(MaskTerm {
            slot: usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?,
            coefficient: reader.get_fp3()?,
        });
    }
    Ok(GeneralClaim {
        point,
        value,
        masks,
    })
}

fn put_general_output(writer: &mut CanonicalWriter, output: &GeneralOutput) {
    put_claim(
        writer,
        &GeneralClaim {
            point: output.point.clone(),
            value: output.claim_scale,
            masks: output.claim_masks.clone(),
        },
    );
    writer.put_fp3(&output.row_target);
    writer.put_u64(output.hiding_form.len() as u64);
    for coefficient in &output.hiding_form {
        writer.put_fp3(coefficient);
    }
    writer.put_u64(output.ends.len() as u64);
    for end in &output.ends {
        writer.put_u64(end.point.len() as u64);
        for coordinate in &end.point {
            writer.put_fp3(coordinate);
        }
        writer.put_fp3(&end.mask_scale);
        writer.put_fp3(&end.value);
    }
    writer.put_u64(output.row_maps.len() as u64);
    for (map, coefficient) in &output.row_maps {
        writer.put_u64(*map as u64);
        writer.put_fp3(coefficient);
    }
}

fn get_general_output(
    reader: &mut CanonicalReader<'_>,
    max_point: usize,
    max_masks: usize,
) -> Result<GeneralOutput> {
    let claim = get_claim(reader, max_point, max_masks)?;
    let row_target = reader.get_fp3()?;
    let hiding_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if hiding_count > max_masks {
        return Err(Error::InvalidEncoding);
    }
    let mut hiding_form = Vec::with_capacity(hiding_count);
    for _ in 0..hiding_count {
        hiding_form.push(reader.get_fp3()?);
    }
    let end_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if end_count > 4 {
        return Err(Error::InvalidEncoding);
    }
    let mut ends = Vec::with_capacity(end_count);
    for _ in 0..end_count {
        let point_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if point_count > max_point {
            return Err(Error::InvalidEncoding);
        }
        let mut point = Vec::with_capacity(point_count);
        for _ in 0..point_count {
            point.push(reader.get_fp3()?);
        }
        ends.push(crate::proof::psc::PointClaim {
            point,
            mask_scale: reader.get_fp3()?,
            value: reader.get_fp3()?,
        });
    }
    let map_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if map_count > 4 {
        return Err(Error::InvalidEncoding);
    }
    let mut row_maps = Vec::with_capacity(map_count);
    for _ in 0..map_count {
        row_maps.push((
            usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?,
            reader.get_fp3()?,
        ));
    }
    Ok(GeneralOutput {
        point: claim.point,
        hiding_form,
        claim_scale: claim.value,
        row_target,
        ends,
        row_maps,
        claim_masks: claim.masks,
    })
}

/// The number of general-sumcheck outputs the lookup transcript leaves.
pub fn outcome_rows(plan: &LookupPlan) -> Result<usize> {
    plan.validate()?;
    let planes = plan.planes()?;
    planes
        .checked_sub(1)
        .and_then(|value| value.checked_add(plan.key_bits as usize))
        .and_then(|value| value.checked_add(plan.table_log as usize))
        .and_then(|value| value.checked_add(2))
        .ok_or(Error::ResourceLimit)
}

impl LookupOutcome {
    /// Fixes the expected end maps and X-variable counts of one output row.
    fn row_shape(plan: &LookupPlan, index: usize) -> Result<(usize, usize, usize)> {
        let planes = plan.planes()?;
        let key_bits = plan.key_bits as usize;
        let mu = plan.mu as usize;
        let table_log = plan.table_log as usize;
        if index < planes - 1 {
            let ends = if index == 0 { 3 } else { 2 };
            return Ok((0, ends, 0));
        }
        let index = index - (planes - 1);
        if index < key_bits {
            return Ok((index, 2, 0));
        }
        let index = index - key_bits;
        if index < table_log {
            return Ok((mu + index, 2, 0));
        }
        let index = index - table_log;
        match index {
            0 => Ok((mu + table_log, 0, 1)),
            1 => Ok((mu + table_log, 1, 0)),
            _ => Err(Error::ShapeMismatch),
        }
    }

    pub fn check(&self, plan: &LookupPlan) -> Result<()> {
        plan.validate()?;
        let planes = plan.planes()?;
        let rows = plan.table_rows()?;
        let keys = plan.key_count()?;
        let count = mask_count(plan)?;
        if self.rows.len() != outcome_rows(plan)?
            || self.key.point.len() != plan.key_bits as usize
            || self.multiplicity.0.len() != planes * rows
            || self.multiplicity.1 != Fp3::from_u64(keys as u64)
        {
            return Err(Error::ShapeMismatch);
        }
        if self
            .key
            .masks
            .iter()
            .any(|term| term.slot >= count)
        {
            return Err(Error::InvalidIndex);
        }
        for (index, output) in self.rows.iter().enumerate() {
            let (point, ends, maps) = Self::row_shape(plan, index)?;
            if output.point.len() != point
                || output.ends.len() != ends
                || output.row_maps.len() != maps
                || output.hiding_form.len() > count
                || output.claim_masks.iter().any(|term| term.slot >= count)
                || output
                    .row_maps
                    .iter()
                    .any(|(map, _)| *map >= maps.max(1))
            {
                return Err(Error::ShapeMismatch);
            }
            for end in &output.ends {
                if end.point.len() != point {
                    return Err(Error::ShapeMismatch);
                }
            }
        }
        Ok(())
    }

    pub fn encode(&self, plan: &LookupPlan) -> Result<Vec<u8>> {
        self.check(plan)?;
        Ok(self.encode_unchecked())
    }

    /// Writes the outcome without re-checking it against a plan. The enclosing
    /// message checks it on decode; the prover path checks it before calling.
    pub(crate) fn encode_unchecked(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(LOOKUP_VERSION);
        put_claim(&mut writer, &self.key);
        writer.put_u64(self.multiplicity.0.len() as u64);
        for coefficient in &self.multiplicity.0 {
            writer.put_fp3(coefficient);
        }
        writer.put_fp3(&self.multiplicity.1);
        writer.put_u64(self.rows.len() as u64);
        for output in &self.rows {
            put_general_output(&mut writer, output);
        }
        writer.into_bytes()
    }

    pub fn decode(plan: &LookupPlan, bytes: &[u8]) -> Result<Self> {
        plan.validate()?;
        let mut reader = CanonicalReader::new(
            bytes,
            zkie_core::common::codec::DecodeLimits::unlimited(),
        );
        if reader.get_u64()? != LOOKUP_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let max_point = plan.key_bits as usize + plan.mu as usize + plan.table_log as usize;
        let count = mask_count(plan)?;
        let key = get_claim(&mut reader, plan.key_bits as usize, count)?;
        let coefficient_count =
            usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        let planes = plan.planes()?;
        let rows = plan.table_rows()?;
        if coefficient_count != planes * rows {
            return Err(Error::ShapeMismatch);
        }
        let mut coefficients = Vec::with_capacity(coefficient_count);
        for _ in 0..coefficient_count {
            coefficients.push(reader.get_fp3()?);
        }
        let total = reader.get_fp3()?;
        let row_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if row_count != outcome_rows(plan)? {
            return Err(Error::ShapeMismatch);
        }
        let mut outputs = Vec::with_capacity(row_count);
        for _ in 0..row_count {
            outputs.push(get_general_output(&mut reader, max_point, count)?);
        }
        reader.finish()?;
        let outcome = Self {
            key,
            multiplicity: (coefficients, total),
            rows: outputs,
        };
        outcome.check(plan)?;
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_core::types::Digest;
    use zkie_engine::pcs::request::code::build_code_plan;
    use zkie_engine::pcs::request::runtime_tree::{RuntimeTreePart, RuntimeTreePlan};

    use crate::proof::hidden_send_session::WShape;

    fn witness(keys: &[u64], plan: &LookupPlan) -> LookupWitness {
        let planes = plan.planes().unwrap();
        let rows = plan.table_rows().unwrap();
        let mut counts = vec![0u64; rows];
        for key in keys {
            if (*key as usize) < rows {
                counts[*key as usize] += 1;
            }
        }
        let mut bits = vec![Fp3::ZERO; planes * rows];
        for row in 0..rows {
            for plane in 0..planes {
                if (counts[row] >> plane) & 1 == 1 {
                    bits[plane + planes * row] = Fp3::ONE;
                }
            }
        }
        LookupWitness { bits }
    }

    fn key_plan(keys: &[u64]) -> LookupKeyPlan {
        LookupKeyPlan {
            sources: keys
                .iter()
                .enumerate()
                .map(|(index, key)| LookupKeySource {
                    oracle: 900 + index as u64,
                    position: 0,
                    row: 0,
                    place: index as u32,
                    coefficient: Fp3::ONE,
                    offset: Fp3::ZERO,
                })
                .collect(),
            padding_value: Fp3::ZERO,
            table_identity: Digest([0x44; 32]),
        }
    }

    fn key_rows(keys: &[u64]) -> Vec<Vec<Vec<Fp3>>> {
        vec![vec![keys.iter().map(|key| Fp3::from_u64(*key)).collect()]]
    }

    #[test]
    fn key_claim_registers_one_row_on_its_member() {
        let plan = plan();
        let keys = [0u64, 1, 2];
        let mut key_plan = key_plan(&keys);
        key_plan.sources[0].coefficient = Fp3::from_u64(2);
        key_plan.sources[0].offset = Fp3::from_u64(3);
        key_plan.sources.truncate(1);
        key_plan.padding_value = Fp3::ZERO;
        let point = [Fp3::from_u64(5), Fp3::from_u64(7)];
        let cell = Fp3::from_u64(11);
        let eq = eq_weights(&point);
        let value = eq[0].mul(&Fp3::from_u64(2).mul(&cell).add(&Fp3::from_u64(3)));
        let claim = GeneralClaim { point: point.to_vec(), value, masks: Vec::new() };
        let member = KeyClaimMember {
            source_oracle: 900,
            bat_oracle: 77,
            shape: WShape::new(4, build_code_plan(4, 8, 1).unwrap(), 1, 1, 40, 700, 1, Digest([0x41; 32]), Digest([0x51; 32])).unwrap(),
            columns: 4,
        };
        let row = key_plan
            .registration_row(&plan, &claim, &[member], 88, &|slot| Ok(slot as u64))
            .unwrap();
        assert_eq!(row.terms.len(), 1);
        assert_eq!(row.terms[0].oracle, 77);
        assert_eq!(row.terms[0].values[0], eq[0].mul(&Fp3::from_u64(2)));
        assert_eq!(row.target, value.sub(&eq[0].mul(&Fp3::from_u64(3))));
    }

    #[test]
    fn key_claim_expands_to_linear_terms() {
        let plan = plan();
        let keys = [0u64, 1, 2];
        let cells = [Fp3::from_u64(3), Fp3::from_u64(5), Fp3::from_u64(7)];
        let mut key_plan = key_plan(&keys);
        key_plan.padding_value = Fp3::from_u64(11);
        key_plan.sources[0].coefficient = Fp3::from_u64(2);
        key_plan.sources[1].offset = Fp3::from_u64(4);
        let point = [Fp3::from_u64(9), Fp3::from_u64(13)];
        let eq = eq_weights(&point);
        let mut value = Fp3::ZERO;
        for index in 0..cells.len() {
            let key = key_plan.sources[index]
                .coefficient
                .mul(&cells[index])
                .add(&key_plan.sources[index].offset);
            value = value.add(&eq[index].mul(&key));
        }
        value = value.add(&eq[3].mul(&key_plan.padding_value));
        let claim = GeneralClaim { point: point.to_vec(), value, masks: Vec::new() };
        let expansion = key_plan.expand_claim(&plan, &claim).unwrap();
        let mut lhs = expansion.constant;
        for (index, term) in expansion.terms.iter().enumerate() {
            lhs = lhs.add(&term.coefficient.mul(&cells[index]));
        }
        assert_eq!(lhs, expansion.value);
    }

    fn plan() -> LookupPlan {
        LookupPlan {
            key_bits: 2,
            mu: 2,
            table_log: 8,
        }
    }

    fn commit_plan() -> LookupCommitPlan {
        let bits_shape = WShape::new(
            1024,
            build_code_plan(12, 2048, 1).unwrap(),
            1,
            1,
            40,
            701,
            1,
            Digest([0x42; 32]),
            Digest([0x51; 32]),
        )
        .unwrap();
        LookupCommitPlan {
            bits_shape,
            bits_oracle: 901,
            group: 7,
            bits_openings: 1,
            mask_oracle: 902,
            mask_code: build_code_plan(6, 32, 1).unwrap(),
            mask_rows: 8,
            mask_openings: 1,
            mask_tree: RuntimeTreePlan {
                tree_number: 702,
                leaf_class: 1,
                log_n: 6,
                log_n_zk: 6,
                parts: vec![RuntimeTreePart {
                    main_symbols: 0,
                    mask_symbols: 24,
                }],
                salt_bytes: 40,
                run: Digest([0x43; 32]),
                configuration: Digest([0x51; 32]),
            },
        }
    }

    #[test]
    fn honest_byte_lookup_proves_and_verifies() {
        let plan = plan();
        let keys = [0u64, 1, 2, 255];
        let witness = witness(&keys, &plan);
        let (messages, proof, proven) = prove_lookup(&plan, &commit_plan(), &key_plan(&keys), &key_rows(&keys), &witness, 7).unwrap();
        let verified = verify_lookup(&plan, &commit_plan(), &messages.roots(), &proof, 7).unwrap();
        assert_eq!(proven, verified);
        assert_eq!(proven.multiplicity.0.len(), plan.planes().unwrap() * 256);
        assert_eq!(proven.multiplicity.1, Fp3::from_u64(4));
    }

    #[test]
    fn duplicate_keys_prove_and_verify() {
        let plan = plan();
        let keys = [0u64, 0, 1, 2];
        let witness = witness(&keys, &plan);
        let (messages, proof, proven) = prove_lookup(
            &plan,
            &commit_plan(),
            &key_plan(&keys),
            &key_rows(&keys),
            &witness,
            7,
        )
        .unwrap();
        let verified =
            verify_lookup(&plan, &commit_plan(), &messages.roots(), &proof, 7).unwrap();
        assert_eq!(proven, verified);
    }

    #[test]
    fn committed_lookup_messages_prove_without_recommitting() {
        let plan = plan();
        let keys = [0u64, 1, 2, 255];
        let witness = witness(&keys, &plan);
        let commit_plan = commit_plan();
        let mask_entries = mask_count(&plan).unwrap();
        let mut source = LookupRandomSource::new(7 ^ 0x5A5A_5A5A);
        let messages = commit_lookup(
            &commit_plan,
            &witness.bits,
            mask_entries,
            &mut source,
        )
        .unwrap();
        let (proof, proven) = prove_lookup_with_messages(
            &plan,
            &commit_plan,
            &key_plan(&keys),
            &key_rows(&keys),
            &witness,
            &messages,
            7,
        )
        .unwrap();
        let verified =
            verify_lookup(&plan, &commit_plan, &messages.roots(), &proof, 7).unwrap();
        assert_eq!(proven, verified);
        let bytes = proven.encode(&plan).unwrap();
        assert_eq!(LookupOutcome::decode(&plan, &bytes).unwrap(), proven);
    }

    #[test]
    fn lookup_messages_round_trip_and_reject_bad_versions() {
        let plan = plan();
        let keys = [0u64, 1, 2, 255];
        let witness = witness(&keys, &plan);
        let (messages, proof, _) = prove_lookup(&plan, &commit_plan(), &key_plan(&keys), &key_rows(&keys), &witness, 13).unwrap();
        let bytes = proof.encode().unwrap();
        let decoded = LookupProof::decode(&plan, &bytes).unwrap();
        assert_eq!(decoded, proof);
        assert!(verify_lookup(&plan, &commit_plan(), &messages.roots(), &decoded, 13).is_ok());
        let mut changed = bytes.clone();
        changed[0] ^= 1;
        assert!(LookupProof::decode(&plan, &changed).is_err());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(LookupProof::decode(&plan, &trailing).is_err());
        let mut truncated = bytes.clone();
        truncated.pop();
        assert!(LookupProof::decode(&plan, &truncated).is_err());
    }

    #[test]
    fn malformed_lookup_messages_are_rejected() {
        let plan = plan();
        let keys = [0u64, 1, 2, 255];
        let witness = witness(&keys, &plan);
        let (messages, proof, _) = prove_lookup(&plan, &commit_plan(), &key_plan(&keys), &key_rows(&keys), &witness, 9).unwrap();
        assert!(verify_lookup(&plan, &commit_plan(), &messages.roots(), &proof, 9).is_ok());

        let mut changed = proof.clone();
        changed.tie[0].y_round.coefficients.pop();
        assert!(verify_lookup(&plan, &commit_plan(), &messages.roots(), &changed, 9).is_err());

        let mut changed = proof.clone();
        changed.table_levels[0].rounds[0].coefficients.clear();
        assert!(verify_lookup(&plan, &commit_plan(), &messages.roots(), &changed, 9).is_err());

        let mut changed = proof.clone();
        changed.bits.end_values.clear();
        assert!(verify_lookup(&plan, &commit_plan(), &messages.roots(), &changed, 9).is_err());

        let mut changed = proof.clone();
        changed.key_levels.pop();
        assert!(verify_lookup(&plan, &commit_plan(), &messages.roots(), &changed, 9).is_err());

        let honest =
            verify_lookup(&plan, &commit_plan(), &messages.roots(), &proof, 9).unwrap();
        let mut changed = messages.clone();
        changed.bits.roots[0] = Digest([0x99; 32]);
        let tampered =
            verify_lookup(&plan, &commit_plan(), &changed.roots(), &proof, 9).unwrap();
        assert_ne!(honest.key, tampered.key);
        let mut changed = messages.clone();
        changed.mask_root = Digest([0x98; 32]);
        let tampered =
            verify_lookup(&plan, &commit_plan(), &changed.roots(), &proof, 9).unwrap();
        assert_ne!(honest.key, tampered.key);
    }

    #[test]
    fn an_out_of_table_key_cannot_be_proved() {
        let plan = plan();
        let keys = [0u64, 1, 2, 999];
        let witness = witness(&keys, &plan);
        assert!(prove_lookup(&plan, &commit_plan(), &key_plan(&keys), &key_rows(&keys), &witness, 11).is_err());
    }
}
