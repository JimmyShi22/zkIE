//! PSC message and verifier skeleton for the norm core.
//!
//! This module builds the committed messages of the norm's `x`, `sum`,
//! `squares`, `mean` and `variance` groups and a verifier skeleton that checks
//! their group declarations, openings, challenge order and product-sumcheck
//! transcripts.
//!
//! Scope, stated explicitly: the skeleton does not yet bind the product-
//! sumcheck point values to the committed rows through BAT/FOLD/CS/BASE. The
//! row-level relation (including carry/range, the rsqrt lookup, the weight
//! product and the output commitment) is the next slice. The transcript
//! targets are zero. The first-stage openings are product-form encoded and
//! salted; the second-stage point values are masked with prover-private
//! one-time entries, and those masks are not committed or verified in this
//! skeleton. The verifier backend is the commitment-free transcript checker:
//! it checks group identity, roots, opening positions, message versions,
//! challenge order and the equality of each shown point value with the
//! transcript's shown end value. It does not reject arbitrary transcript or
//! challenge tampering; that rejection requires the BAT/FOLD/CS/BASE row
//! binding and is deliberately not claimed here. What this module does
//! establish is the exact message layout, group identity, opening positions,
//! challenge order and PSC transcript consistency that the later row binding
//! consumes.
//!
//! The five group roots are trusted plan inputs in this skeleton. The proof
//! does not yet derive the sum/squares/mean/variance roots from the x rows;
//! that derivation is exactly the deferred row binding.

use zkie_core::artifact::digest_bytes;
use zkie_core::common::codec::DecodeLimits;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::chain::Chain;
use zkie_engine::pcs::request::fold::{ByteMaskCoins, MaskCoins};
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};

use crate::proof::committed_psc::{
    get_psc_transcript, get_tree_plan, put_psc_transcript, put_tree_plan, CommittedPscGroup,
    PositionPscChallenges, PscTranscriptTags,
};
use crate::proof::hidden_norm::{NormCoreRow, NormShape};
use crate::proof::hidden_send_session::{SendSamples, WShape};
use crate::proof::psc::{
    hp_entries, prove_dot_sumcheck, psc_variables, verify_psc_transcript, DotClaim,
    DotMaskEntries, DotProof, SHOWN_MAPS,
};

/// Version of the norm PSC message skeleton.
pub const NORM_PSC_VERSION: u64 = 1;
/// Number of product-sumcheck instances per position: sum, squares, mean and
/// variance.
pub const NORM_PSC_RELATIONS: usize = 4;
/// The shared chain label of the norm PSC skeleton.
const NORM_PSC_CHAIN_LABEL: u64 = 77;
/// Transcript tags of the norm product-sumcheck, distinct from the closure
/// tags and the hidden Dot tags.
const NORM_PSC_TAGS: PscTranscriptTags = PscTranscriptTags {
    version: NORM_PSC_VERSION,
    hiding_sum: 50,
    round: 51,
    end: 52,
};

/// The trusted plan of one committed value group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormPscGroupPlan {
    pub shape: WShape,
    /// The number of committed columns in one message row.
    pub columns: u64,
    /// The SEND group id.
    pub group: u64,
    /// The first oracle id of the group entries.
    pub oracle_base: u64,
    /// The number of opened positions per position.
    pub openings: usize,
    /// The committed roots in position order.
    pub roots: Vec<Digest>,
}

impl NormPscGroupPlan {
    fn fault(&self, positions: u32) -> Option<&'static str> {
        if self.columns == 0 || !self.columns.is_power_of_two() || self.columns > (1 << 20) {
            return Some("a norm group has no power-of-two column count");
        }
        if self.openings == 0 || self.openings as u64 > self.shape.code.r {
            return Some("a norm group has no bounded opening count");
        }
        if self.shape.answers == 0 {
            return Some("a norm group has no sample answer");
        }
        if self.shape.tree_plan(0).is_err() {
            return Some("a norm group has no runtime tree plan");
        }
        if self.roots.len() != positions as usize {
            return Some("a norm group does not have one root per position");
        }
        if self.roots.iter().any(|root| *root == Digest::ZERO) {
            return Some("a norm group has no root");
        }
        None
    }
}

/// The five committed groups of the norm core.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormPscGroups {
    pub x: NormPscGroupPlan,
    pub sum: NormPscGroupPlan,
    pub squares: NormPscGroupPlan,
    pub mean: NormPscGroupPlan,
    pub variance: NormPscGroupPlan,
}

impl NormPscGroups {
    fn all(&self) -> [&NormPscGroupPlan; 5] {
        [&self.x, &self.sum, &self.squares, &self.mean, &self.variance]
    }
}

/// The trusted plan of the norm PSC skeleton.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormPscPlan {
    pub shape: NormShape,
    /// The padded input width; the product-sumcheck maps have `2*columns`
    /// entries and therefore `log2(columns)+1` variables.
    pub columns: u64,
    pub variables: usize,
    pub groups: NormPscGroups,
}

impl NormPscPlan {
    pub fn validate(&self) -> Result<()> {
        if self.shape.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        if self.columns < 2
            || !self.columns.is_power_of_two()
            || self.columns < self.shape.hidden as u64
            || self.columns > (1 << 20)
        {
            return Err(Error::InvalidConfig);
        }
        let expected_variables = self.columns.trailing_zeros() as usize + 1;
        if self.variables != expected_variables {
            return Err(Error::InvalidConfig);
        }
        if self.groups.x.columns != self.columns
            || self.groups.sum.columns != 1
            || self.groups.squares.columns != 1
            || self.groups.mean.columns != 1
            || self.groups.variance.columns != 1
        {
            return Err(Error::InvalidConfig);
        }
        for group in self.groups.all() {
            if group.fault(self.shape.positions).is_some() {
                return Err(Error::InvalidConfig);
            }
        }
        let groups = self.groups.all();
        for left in 0..groups.len() {
            for right in left + 1..groups.len() {
                if groups[left].group == groups[right].group
                    || groups[left].oracle_base == groups[right].oracle_base
                {
                    return Err(Error::InvalidConfig);
                }
            }
        }
        Ok(())
    }

    pub fn encode_into(&self, writer: &mut CanonicalWriter) {
        writer.put_u64(NORM_PSC_VERSION);
        writer.put_u64(self.shape.positions as u64);
        writer.put_u64(self.shape.hidden as u64);
        writer.put_u64(self.shape.valid as u64);
        writer.put_u64(self.columns);
        writer.put_u64(self.variables as u64);
        for group in self.groups.all() {
            writer.put_u64(group.columns);
            writer.put_u64(group.group);
            writer.put_u64(group.oracle_base);
            writer.put_u64(group.openings as u64);
            put_wshape(writer, &group.shape);
            writer.put_u64(group.roots.len() as u64);
            for root in &group.roots {
                writer.put_digest(root);
            }
        }
    }

    pub fn digest(&self) -> Digest {
        let mut writer = CanonicalWriter::new();
        self.encode_into(&mut writer);
        digest_bytes(&writer.into_bytes())
    }
}

fn put_wshape(writer: &mut CanonicalWriter, shape: &WShape) {
    writer.put_u64(shape.code.log_n as u64);
    writer.put_u64(shape.code.ell);
    writer.put_u64(shape.code.r);
    writer.put_u64(shape.main_rows);
    writer.put_u64(shape.answers as u64);
    writer.put_u64(shape.salt_bytes as u64);
    writer.put_u64(shape.tree_number);
    writer.put_u64(shape.leaf_class as u64);
    writer.put_digest(&shape.run);
    writer.put_digest(&shape.configuration);
}

/// The public first-stage norm message: the five declared and opened groups.
/// It carries no value rows and no plaintext.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormPscFirst {
    pub x: CommittedPscGroup,
    pub sum: CommittedPscGroup,
    pub squares: CommittedPscGroup,
    pub mean: CommittedPscGroup,
    pub variance: CommittedPscGroup,
}

impl NormPscFirst {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(NORM_PSC_VERSION);
        self.x.encode_into(&mut writer)?;
        self.sum.encode_into(&mut writer)?;
        self.squares.encode_into(&mut writer)?;
        self.mean.encode_into(&mut writer)?;
        self.variance.encode_into(&mut writer)?;
        Ok(writer.into_bytes())
    }

    pub fn decode(plan: &NormPscPlan, bytes: &[u8]) -> Result<Self> {
        plan.validate()?;
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        if reader.get_u64()? != NORM_PSC_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let positions = plan.shape.positions;
        let message = Self {
            x: CommittedPscGroup::decode(&mut reader, &plan.groups.x.shape, positions)?,
            sum: CommittedPscGroup::decode(&mut reader, &plan.groups.sum.shape, positions)?,
            squares: CommittedPscGroup::decode(
                &mut reader,
                &plan.groups.squares.shape,
                positions,
            )?,
            mean: CommittedPscGroup::decode(&mut reader, &plan.groups.mean.shape, positions)?,
            variance: CommittedPscGroup::decode(
                &mut reader,
                &plan.groups.variance.shape,
                positions,
            )?,
        };
        reader.finish()?;
        message.check(plan)?;
        Ok(message)
    }

    pub fn check(&self, plan: &NormPscPlan) -> Result<()> {
        plan.validate()?;
        let groups = [
            (&self.x, &plan.groups.x),
            (&self.sum, &plan.groups.sum),
            (&self.squares, &plan.groups.squares),
            (&self.mean, &plan.groups.mean),
            (&self.variance, &plan.groups.variance),
        ];
        for (message, group_plan) in groups {
            if message.message.group.group != group_plan.group {
                return Err(Error::StatementMismatch);
            }
            if message.message.roots != group_plan.roots {
                return Err(Error::StatementMismatch);
            }
            if message.fault(&group_plan.shape, plan.shape.positions).is_some() {
                return Err(Error::ShapeMismatch);
            }
            for (position, entry) in message.message.group.entries.iter().enumerate() {
                if entry.oracle
                    != group_plan
                        .oracle_base
                        .checked_add(position as u64)
                        .ok_or(Error::ResourceLimit)?
                {
                    return Err(Error::StatementMismatch);
                }
            }
        }
        Ok(())
    }
}

/// The public second-stage norm message: one product-sumcheck transcript and
/// its two shown point values per instance, in position-major and
/// sum/squares/mean/variance-major order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormPscSecond {
    pub psc: Vec<DotProof>,
    pub point_values: Vec<[Fp3; 2]>,
}

impl NormPscSecond {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut writer = CanonicalWriter::new();
        put_psc_transcript(
            &mut writer,
            NORM_PSC_VERSION,
            &self.psc,
            &self.point_values,
        )?;
        Ok(writer.into_bytes())
    }

    pub fn decode(plan: &NormPscPlan, bytes: &[u8]) -> Result<Self> {
        plan.validate()?;
        let instances = plan
            .shape
            .positions
            .checked_mul(NORM_PSC_RELATIONS as u32)
            .ok_or(Error::ResourceLimit)?;
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        let (psc, point_values) = get_psc_transcript::<2>(
            &mut reader,
            NORM_PSC_VERSION,
            instances,
            plan.variables,
        )?;
        reader.finish()?;
        let message = Self { psc, point_values };
        message.check(plan)?;
        Ok(message)
    }

    pub fn decode_for(plan: &NormPlan, bytes: &[u8]) -> Result<Self> {
        plan.validate()?;
        let instances = plan
            .shape
            .positions
            .checked_mul(NORM_PSC_RELATIONS as u32)
            .ok_or(Error::ResourceLimit)?;
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        let (psc, point_values) = get_psc_transcript::<2>(
            &mut reader,
            NORM_PSC_VERSION,
            instances,
            plan.variables,
        )?;
        reader.finish()?;
        let message = Self { psc, point_values };
        message.check_for(plan)?;
        Ok(message)
    }

    pub fn check_for(&self, plan: &NormPlan) -> Result<()> {
        plan.validate()?;
        let instances = plan.shape.positions as usize * NORM_PSC_RELATIONS;
        if self.psc.len() != instances || self.point_values.len() != instances {
            return Err(Error::ShapeMismatch);
        }
        for proof in &self.psc {
            if proof.rounds.len() != plan.variables || proof.y_round.coefficients.len() != 2 {
                return Err(Error::ShapeMismatch);
            }
        }
        Ok(())
    }

    pub fn check(&self, plan: &NormPscPlan) -> Result<()> {
        plan.validate()?;
        let instances = plan.shape.positions as usize * NORM_PSC_RELATIONS;
        if self.psc.len() != instances || self.point_values.len() != instances {
            return Err(Error::ShapeMismatch);
        }
        for proof in &self.psc {
            if proof.rounds.len() != plan.variables
                || proof.y_round.coefficients.len() != 2
            {
                return Err(Error::ShapeMismatch);
            }
        }
        Ok(())
    }
}

fn norm_chain(plan: &NormPscPlan, first: &NormPscFirst) -> Result<Chain> {
    Chain::begin(
        NORM_PSC_CHAIN_LABEL,
        &plan.digest(),
        &first.encode()?,
        &[],
        None,
    )
}

fn empty_masks(variables: usize) -> Result<DotMaskEntries> {
    let weights = psc_variables(variables)?;
    Ok(DotMaskEntries {
        hiding: vec![Fp3::ZERO; hp_entries(&weights)?],
        shown: [Fp3::ZERO; SHOWN_MAPS],
    })
}

fn random_masks(variables: usize, coins: &mut ByteMaskCoins) -> Result<DotMaskEntries> {
    let weights = psc_variables(variables)?;
    let count = hp_entries(&weights)?;
    let mut hiding = Vec::with_capacity(count);
    for _ in 0..count {
        hiding.push(coins.draw()?.0);
    }
    Ok(DotMaskEntries {
        hiding,
        shown: [coins.draw()?.0, coins.draw()?.0],
    })
}

fn map_pair(columns: usize) -> (Vec<Fp3>, Vec<Fp3>) {
    (vec![Fp3::ZERO; 2 * columns], vec![Fp3::ZERO; 2 * columns])
}

fn rel_map(
    row: &NormCoreRow,
    values: &[i64],
    valid: usize,
    columns: usize,
    relation: usize,
) -> Result<(Vec<Fp3>, Vec<Fp3>)> {
    let (mut left, mut right) = map_pair(columns);
    match relation {
        0 => {
            for (index, value) in values.iter().take(valid).enumerate() {
                left[index] = fp3_signed(*value as i128)?;
                right[index] = Fp3::ONE;
            }
            left[columns] = fp3_signed(-row.sum)?;
            right[columns] = Fp3::ONE;
        }
        1 => {
            let mean = fp3_signed(row.mean as i128)?;
            for (index, value) in values.iter().take(valid).enumerate() {
                let centered = fp3_signed(*value as i128)?.sub(&mean);
                left[index] = centered;
                right[index] = centered;
            }
            left[columns] = fp3_signed(-row.squares)?;
            right[columns] = Fp3::ONE;
        }
        2 => {
            left[0] = fp3_signed(row.sum)?;
            left[1] = fp3_signed(-(row.mean as i128) * valid as i128)?;
            left[2] = fp3_signed(-row.mean_remainder)?;
            right[0] = Fp3::ONE;
            right[1] = Fp3::ONE;
            right[2] = Fp3::ONE;
        }
        3 => {
            left[0] = fp3_signed(row.squares)?;
            left[1] = fp3_signed(-(row.variance as i128) * valid as i128)?;
            left[2] = fp3_signed(-row.variance_remainder)?;
            right[0] = Fp3::ONE;
            right[1] = Fp3::ONE;
            right[2] = Fp3::ONE;
        }
        _ => return Err(Error::InvalidConfig),
    }
    Ok((left, right))
}

/// Prover side of the skeleton: computes the four product-sumcheck instances
/// of every position from the exact integer core rows and the public first
/// message.
pub fn prove_norm_psc(
    plan: &NormPscPlan,
    rows: &[Vec<i64>],
    first: &NormPscFirst,
    coins: &mut ByteMaskCoins,
) -> Result<NormPscSecond> {
    plan.validate()?;
    first.check(plan)?;
    let positions = plan.shape.positions as usize;
    let hidden = plan.shape.hidden as usize;
    let valid = plan.shape.valid as usize;
    let columns = plan.columns as usize;
    if rows.len() != positions || rows.iter().any(|row| row.len() != hidden) {
        return Err(Error::ShapeMismatch);
    }
    let core_rows = rows
        .iter()
        .map(|row| NormCoreRow::from_values(row, valid))
        .collect::<Result<Vec<_>>>()?;
    let mut chain = norm_chain(plan, first)?;
    let claim = DotClaim {
        masks: Vec::new(),
        target: Fp3::ZERO,
    };
    let instances = positions * NORM_PSC_RELATIONS;
    let mut psc = Vec::with_capacity(instances);
    let mut point_values = Vec::with_capacity(instances);
    for (position, row) in core_rows.iter().enumerate() {
        for relation in 0..NORM_PSC_RELATIONS {
            let (left, right) = rel_map(
                row,
                &rows[position],
                valid,
                columns,
                relation,
            )?;
            let mut challenges = PositionPscChallenges::new(
                &mut chain,
                NORM_PSC_TAGS,
                (position * NORM_PSC_RELATIONS + relation) as u64,
            );
            let entries = random_masks(plan.variables, coins)?;
            let (proof, output) =
                prove_dot_sumcheck(&left, &right, &claim, &entries, &mut challenges)?;
            psc.push(proof);
            point_values.push([output.ends[0].value, output.ends[1].value]);
        }
    }
    Ok(NormPscSecond { psc, point_values })
}

/// Verifier side of the skeleton: checks the plan, the first message, the
/// challenge-bound transcripts and the shown point values. It does not yet
/// check the point values against the committed rows; that is the deferred
/// BAT/FOLD/CS/BASE row binding.
pub fn verify_norm_psc(
    plan: &NormPscPlan,
    first: &NormPscFirst,
    second: &NormPscSecond,
) -> Result<()> {
    plan.validate()?;
    first.check(plan)?;
    second.check(plan)?;
    let mut chain = norm_chain(plan, first)?;
    let claim = DotClaim {
        masks: Vec::new(),
        target: Fp3::ZERO,
    };
    let entries = empty_masks(plan.variables)?;
    for (index, proof) in second.psc.iter().enumerate() {
        let mut challenges = PositionPscChallenges::new(
            &mut chain,
            NORM_PSC_TAGS,
            index as u64,
        );
        let output = verify_psc_transcript(
            &claim,
            &entries,
            proof,
            plan.variables,
            &mut challenges,
        )?;
        let expected = [output.ends[0].value, output.ends[1].value];
        if second.point_values[index] != expected {
            return Err(Error::InvalidProof);
        }
    }
    Ok(())
}

fn fp3_signed(value: i128) -> Result<Fp3> {
    crate::proof::output_closure::fp3_signed(value)
}


// ---------------------------------------------------------------------------
// Stage A: the integrated norm plan and first/second messages.
//
// The trusted fixture input is the product-form x group: its roots are plan
// values. The sum/squares/mean/variance groups are proof-fixed witness
// messages: the plan fixes their shapes, identifiers and opening counts, while
// their roots arrive in the first message. The mask message is optional in
// this stage and becomes required when the committed mask entries are wired.

/// Version of the integrated norm plan and first message.
pub const NORM_BIND_VERSION: u64 = 1;

/// One committed witness group of the integrated norm plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormGroupPlan {
    pub shape: WShape,
    pub columns: u64,
    pub group: u64,
    pub oracle_base: u64,
    pub openings: usize,
}

impl NormGroupPlan {
    fn fault(&self, positions: u32) -> Option<&'static str> {
        if self.columns == 0 || !self.columns.is_power_of_two() || self.columns > (1 << 20) {
            return Some("a norm group has no power-of-two column count");
        }
        if self.openings == 0 || self.openings as u64 > self.shape.code.r {
            return Some("a norm group has no bounded opening count");
        }
        if self.shape.fault(self.columns, positions).is_some() {
            return Some("a norm group is not a message shape");
        }
        None
    }
}

/// The trusted plan of the integrated norm relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormPlan {
    pub shape: NormShape,
    pub columns: u64,
    pub variables: usize,
    /// The trusted product-form x group. Its roots are plan values.
    pub x: NormGroupPlan,
    pub x_roots: Vec<Digest>,
    /// The proof-fixed sum/squares/mean/variance witness groups.
    pub sum: NormGroupPlan,
    pub squares: NormGroupPlan,
    pub mean: NormGroupPlan,
    pub variance: NormGroupPlan,
    /// The committed mask group; optional in this stage.
    pub mask: Option<NormMaskPlan>,
}

impl NormPlan {
    pub fn validate(&self) -> Result<()> {
        if self.shape.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        if self.columns < 2
            || !self.columns.is_power_of_two()
            || self.columns < self.shape.hidden as u64
            || self.columns > (1 << 20)
        {
            return Err(Error::InvalidConfig);
        }
        if self.variables != self.columns.trailing_zeros() as usize + 1 {
            return Err(Error::InvalidConfig);
        }
        if self.x.columns != self.columns
            || self.sum.columns != 1
            || self.squares.columns != 1
            || self.mean.columns != 1
            || self.variance.columns != 1
        {
            return Err(Error::InvalidConfig);
        }
        for group in [&self.x, &self.sum, &self.squares, &self.mean, &self.variance] {
            if group.fault(self.shape.positions).is_some() {
                return Err(Error::InvalidConfig);
            }
            if group.shape.answers != 1 {
                return Err(Error::InvalidConfig);
            }
        }
        if self.x_roots.len() != self.shape.positions as usize
            || self.x_roots.iter().any(|root| *root == Digest::ZERO)
        {
            return Err(Error::InvalidConfig);
        }
        let groups = [&self.x, &self.sum, &self.squares, &self.mean, &self.variance];
        for left in 0..groups.len() {
            for right in left + 1..groups.len() {
                if groups[left].group == groups[right].group
                    || groups[left].oracle_base == groups[right].oracle_base
                {
                    return Err(Error::InvalidConfig);
                }
            }
        }
        if let Some(mask) = &self.mask {
            mask.validate(self.shape.positions)?;
        }
        Ok(())
    }

    pub fn encode_into(&self, writer: &mut CanonicalWriter) {
        writer.put_u64(NORM_BIND_VERSION);
        writer.put_u64(self.shape.positions as u64);
        writer.put_u64(self.shape.hidden as u64);
        writer.put_u64(self.shape.valid as u64);
        writer.put_u64(self.columns);
        writer.put_u64(self.variables as u64);
        put_norm_group(writer, &self.x);
        writer.put_u64(self.x_roots.len() as u64);
        for root in &self.x_roots {
            writer.put_digest(root);
        }
        put_norm_group(writer, &self.sum);
        put_norm_group(writer, &self.squares);
        put_norm_group(writer, &self.mean);
        put_norm_group(writer, &self.variance);
        match &self.mask {
            Some(mask) => {
                writer.put_u64(1);
                mask.encode_into(writer);
            }
            None => writer.put_u64(0),
        }
    }

    pub fn digest(&self) -> Digest {
        let mut writer = CanonicalWriter::new();
        self.encode_into(&mut writer);
        digest_bytes(&writer.into_bytes())
    }
}

fn put_norm_group(writer: &mut CanonicalWriter, group: &NormGroupPlan) {
    writer.put_u64(group.columns);
    writer.put_u64(group.group);
    writer.put_u64(group.oracle_base);
    writer.put_u64(group.openings as u64);
    put_wshape(writer, &group.shape);
}

/// The committed mask-only group of the norm relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormMaskPlan {
    pub oracle: u64,
    pub group: u64,
    pub mask_code: zkie_engine::pcs::request::code::CodeParams,
    pub mask_rows: u64,
    pub answers: u32,
    pub openings: usize,
    pub tree: zkie_engine::pcs::request::runtime_tree::RuntimeTreePlan,
    pub positions: u32,
    pub hiding: usize,
    pub shown: usize,
    pub claims: usize,
}

impl NormMaskPlan {
    fn validate(&self, positions: u32) -> Result<()> {
        if self.oracle == 0 || self.group == 0 {
            return Err(Error::InvalidConfig);
        }
        if self.mask_rows == 0 || self.answers == 0 || self.openings == 0 {
            return Err(Error::InvalidConfig);
        }
        if self.positions != positions || self.hiding == 0 || self.shown == 0 || self.claims == 0 {
            return Err(Error::InvalidConfig);
        }
        if zkie_engine::pcs::request::code::build_code_plan(
            self.mask_code.log_n,
            self.mask_code.ell,
            self.mask_code.r,
        )
        .is_err()
        {
            return Err(Error::InvalidConfig);
        }
        if self.tree.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }

    fn encode_into(&self, writer: &mut CanonicalWriter) {
        writer.put_u64(self.oracle);
        writer.put_u64(self.group);
        writer.put_u64(self.mask_code.log_n as u64);
        writer.put_u64(self.mask_code.ell);
        writer.put_u64(self.mask_code.r);
        writer.put_u64(self.mask_rows);
        writer.put_u64(self.answers as u64);
        writer.put_u64(self.openings as u64);
        put_tree_plan(writer, &self.tree);
        writer.put_u64(self.positions as u64);
        writer.put_u64(self.hiding as u64);
        writer.put_u64(self.shown as u64);
        writer.put_u64(self.claims as u64);
    }
}

/// The public mask-only norm message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormMaskMessage {
    pub tree: zkie_engine::pcs::request::runtime_tree::RuntimeTreePlan,
    pub root: Digest,
    pub sample: SendSamples,
    pub positions: Vec<u64>,
    pub opening: zkie_engine::pcs::request::runtime_tree::RuntimeTreeOpening,
}

impl NormMaskMessage {
    fn encode(&self) -> Result<Vec<u8>> {
        let mut writer = CanonicalWriter::new();
        put_tree_plan(&mut writer, &self.tree);
        writer.put_digest(&self.root);
        writer.put_bytes(&self.sample.encode());
        writer.put_u64(self.positions.len() as u64);
        for position in &self.positions {
            writer.put_u64(*position);
        }
        writer.put_bytes(
            &zkie_engine::pcs::request::runtime_tree::encode_runtime_opening(
                &self.tree,
                &self.opening,
            )?,
        );
        Ok(writer.into_bytes())
    }

    fn decode(plan: &NormMaskPlan, bytes: &[u8]) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        let tree = get_tree_plan(&mut reader)?;
        if tree != plan.tree {
            return Err(Error::StatementMismatch);
        }
        let root = reader.get_digest()?;
        if root == Digest::ZERO {
            return Err(Error::InvalidEncoding);
        }
        let sample = SendSamples::decode(reader.get_bytes()?)?;
        if sample.seed.len() != plan.answers as usize
            || sample.answers.len() != plan.answers as usize
        {
            return Err(Error::ShapeMismatch);
        }
        let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if count != plan.openings {
            return Err(Error::ShapeMismatch);
        }
        let mut positions = Vec::with_capacity(count);
        for _ in 0..count {
            positions.push(reader.get_u64()?);
        }
        let opening = zkie_engine::pcs::request::runtime_tree::decode_runtime_opening(
            &tree,
            &positions,
            reader.get_bytes()?,
        )?;
        reader.finish()?;
        Ok(Self {
            tree,
            root,
            sample,
            positions,
            opening,
        })
    }
}

/// The prover-side message rows and encoding tails of one norm group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormGroupWitness {
    /// `messages[position][row][place]`, the unencoded message rows.
    pub messages: Vec<Vec<Vec<Fp3>>>,
    /// `tails[position][row * r + i]`.
    pub tails: Vec<Vec<Fp3>>,
}

/// The prover-side witness of the integrated norm groups.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormWitness {
    pub x: NormGroupWitness,
    pub sum: NormGroupWitness,
    pub squares: NormGroupWitness,
    pub mean: NormGroupWitness,
    pub variance: NormGroupWitness,
    pub mask_rows: Vec<Vec<Fp3>>,
    pub mask_tails: Vec<Vec<Fp3>>,
}

/// The public first-stage message of the integrated norm relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormFirstMessages {
    pub x: CommittedPscGroup,
    pub sum: CommittedPscGroup,
    pub squares: CommittedPscGroup,
    pub mean: CommittedPscGroup,
    pub variance: CommittedPscGroup,
    pub mask: Option<NormMaskMessage>,
}

impl NormFirstMessages {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(NORM_BIND_VERSION);
        self.x.encode_into(&mut writer)?;
        self.sum.encode_into(&mut writer)?;
        self.squares.encode_into(&mut writer)?;
        self.mean.encode_into(&mut writer)?;
        self.variance.encode_into(&mut writer)?;
        match &self.mask {
            Some(mask) => {
                writer.put_u64(1);
                writer.put_bytes(&mask.encode()?);
            }
            None => writer.put_u64(0),
        }
        Ok(writer.into_bytes())
    }

    pub fn decode(plan: &NormPlan, bytes: &[u8]) -> Result<Self> {
        plan.validate()?;
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        if reader.get_u64()? != NORM_BIND_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let positions = plan.shape.positions;
        let x = CommittedPscGroup::decode(&mut reader, &plan.x.shape, positions)?;
        let sum = CommittedPscGroup::decode(&mut reader, &plan.sum.shape, positions)?;
        let squares = CommittedPscGroup::decode(&mut reader, &plan.squares.shape, positions)?;
        let mean = CommittedPscGroup::decode(&mut reader, &plan.mean.shape, positions)?;
        let variance = CommittedPscGroup::decode(&mut reader, &plan.variance.shape, positions)?;
        let mask = match reader.get_u64()? {
            0 => {
                if plan.mask.is_some() {
                    return Err(Error::ShapeMismatch);
                }
                None
            }
            1 => {
                let mask_plan = plan.mask.as_ref().ok_or(Error::ShapeMismatch)?;
                Some(NormMaskMessage::decode(mask_plan, reader.get_bytes()?)?)
            }
            _ => return Err(Error::InvalidEncoding),
        };
        reader.finish()?;
        let message = Self {
            x,
            sum,
            squares,
            mean,
            variance,
            mask,
        };
        message.check_trusted(plan)?;
        Ok(message)
    }

    /// Checks the trusted x roots and every public group declaration. The
    /// witness group roots are proof values and are not compared with the plan.
    pub fn check_trusted(&self, plan: &NormPlan) -> Result<()> {
        plan.validate()?;
        let groups = [
            (&self.x, &plan.x, true),
            (&self.sum, &plan.sum, false),
            (&self.squares, &plan.squares, false),
            (&self.mean, &plan.mean, false),
            (&self.variance, &plan.variance, false),
        ];
        for (message, group_plan, trusted) in groups {
            if message.message.group.group != group_plan.group {
                return Err(Error::StatementMismatch);
            }
            if message.fault(&group_plan.shape, plan.shape.positions).is_some() {
                return Err(Error::ShapeMismatch);
            }
            if trusted && message.message.roots != plan.x_roots {
                return Err(Error::StatementMismatch);
            }
            for (position, entry) in message.message.group.entries.iter().enumerate() {
                if entry.oracle
                    != group_plan
                        .oracle_base
                        .checked_add(position as u64)
                        .ok_or(Error::ResourceLimit)?
                {
                    return Err(Error::StatementMismatch);
                }
            }
        }
        if let (Some(plan_mask), Some(message_mask)) = (&plan.mask, &self.mask) {
            if message_mask.root == Digest::ZERO
                || message_mask.tree != plan_mask.tree
                || message_mask.sample.seed.len() != plan_mask.answers as usize
                || message_mask.sample.answers.len() != plan_mask.answers as usize
                || message_mask.positions.len() != plan_mask.openings
                || message_mask.opening.positions != message_mask.positions
            {
                return Err(Error::ShapeMismatch);
            }
        }
        if plan.mask.is_some() != self.mask.is_some() {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::code::build_code_plan;
    use zkie_engine::pcs::request::runtime_tree::RuntimeTreePart;
    use zkie_engine::pcs::request::fold::{ByteMaskCoins, MaskCoins};
    use zkie_engine::pcs::request::table::SampleSeed;

    use zkie_core::common::field::PrimeField64;

    use crate::proof::committed_psc::CommittedGroupMessage;
    use crate::proof::output_closure::commit_value_group;

    fn code() -> zkie_engine::pcs::request::code::CodeParams {
        build_code_plan(4, 8, 1).unwrap()
    }

    fn group_shape(columns: u64, tree: u64) -> WShape {
        WShape::new(
            columns,
            code(),
            1,
            2,
            40,
            tree,
            1,
            Digest([0x41; 32]),
            Digest([0x51; 32]),
        )
        .unwrap()
    }

    fn group_plan(
        shape: WShape,
        columns: u64,
        group: u64,
        oracle: u64,
    ) -> NormPscGroupPlan {
        NormPscGroupPlan {
            shape,
            columns,
            group,
            oracle_base: oracle,
            openings: 1,
            roots: Vec::new(),
        }
    }

    fn group_randomness(
        shape: &WShape,
        positions: usize,
        coins: &mut ByteMaskCoins,
    ) -> (Vec<Vec<Fp3>>, Vec<Fp3>, Vec<Vec<Vec<u8>>>) {
        let rows = shape.main_rows as usize;
        let tail = shape.code.r as usize;
        let columns = shape.code.n() as usize;
        let answers = shape.answers as usize;
        let mut tails = Vec::with_capacity(positions);
        for _ in 0..positions {
            let mut values = Vec::with_capacity(rows * tail);
            for _ in 0..rows * tail {
                values.push(coins.draw().unwrap().0);
            }
            tails.push(values);
        }
        let mut pads = Vec::with_capacity(positions * answers);
        for _ in 0..positions * answers {
            let mut pad = Fp3::ZERO;
            for _ in 0..64 {
                pad = coins.draw().unwrap().0;
                if pad != Fp3::ZERO {
                    break;
                }
            }
            assert_ne!(pad, Fp3::ZERO);
            pads.push(pad);
        }
        let mut salts = Vec::with_capacity(positions);
        for _ in 0..positions {
            let mut row_salts = Vec::with_capacity(columns);
            for _ in 0..columns {
                let mut salt = Vec::with_capacity(shape.salt_bytes);
                while salt.len() < shape.salt_bytes {
                    let (value, _) = coins.draw().unwrap();
                    for coefficient in value.c {
                        salt.extend_from_slice(
                            &coefficient.as_canonical_u64().to_le_bytes(),
                        );
                    }
                }
                salt.truncate(shape.salt_bytes);
                row_salts.push(salt);
            }
            salts.push(row_salts);
        }
        (tails, pads, salts)
    }

    fn commit_group(
        plan: &mut NormPscGroupPlan,
        values: &[Vec<Fp3>],
        coins: &mut ByteMaskCoins,
    ) -> CommittedPscGroup {
        let positions = values.len();
        let (tails, pads, salts) =
            group_randomness(&plan.shape, positions, coins);
        let committed = commit_value_group(
            &plan.shape,
            plan.columns,
            plan.oracle_base,
            plan.group,
            values,
            &pads,
            &tails,
            &salts,
            positions as u32,
        )
        .unwrap();
        plan.roots = committed.roots.clone();
        let positions = (0..positions as u32)
            .map(|_| vec![0])
            .collect::<Vec<_>>();
        let openings = positions
            .iter()
            .enumerate()
            .map(|(position, list)| {
                committed
                    .trees
                    .get(position)
                    .unwrap()
                    .open(list)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        CommittedPscGroup {
            message: CommittedGroupMessage::new(committed.group.clone()),
            sample: SendSamples {
                seed: (0..plan.shape.answers)
                    .map(|_| SampleSeed {
                        z: Vec::new(),
                        beta: Fp3::ONE,
                    })
                    .collect(),
                answers: vec![Fp3::ONE; values.len()],
            },
            positions,
            openings,
        }
    }

    fn values(rows: &[Vec<i64>]) -> Vec<Vec<Fp3>> {
        rows.iter()
            .map(|row| row.iter().map(|value| fp3_signed(*value as i128).unwrap()).collect())
            .collect()
    }

    fn fixture() -> (NormPscPlan, Vec<Vec<i64>>, NormPscFirst, NormPscSecond) {
        let shape = NormShape::new(2, 4, 4).unwrap();
        let rows = vec![vec![3, -2, 5, 1], vec![-4, 0, 2, 7]];
        let core = rows
            .iter()
            .map(|row| NormCoreRow::from_values(row, 4).unwrap())
            .collect::<Vec<_>>();
        let mut groups = NormPscGroups {
            x: group_plan(group_shape(4, 1), 4, 1, 100),
            sum: group_plan(group_shape(1, 2), 1, 2, 200),
            squares: group_plan(group_shape(1, 3), 1, 3, 300),
            mean: group_plan(group_shape(1, 4), 1, 4, 400),
            variance: group_plan(group_shape(1, 5), 1, 5, 500),
        };
        let mut coins = ByteMaskCoins::new(Digest([0x72; 32])).unwrap();
        let x_message = commit_group(&mut groups.x, &values(&rows), &mut coins);
        let sum_message = commit_group(
            &mut groups.sum,
            &values(&core.iter().map(|row| vec![row.sum as i64]).collect::<Vec<_>>()),
            &mut coins,
        );
        let squares_message = commit_group(
            &mut groups.squares,
            &values(
                &core
                    .iter()
                    .map(|row| vec![row.squares as i64])
                    .collect::<Vec<_>>(),
            ),
            &mut coins,
        );
        let mean_message = commit_group(
            &mut groups.mean,
            &values(&core.iter().map(|row| vec![row.mean]).collect::<Vec<_>>()),
            &mut coins,
        );
        let variance_message = commit_group(
            &mut groups.variance,
            &values(
                &core
                    .iter()
                    .map(|row| vec![row.variance])
                    .collect::<Vec<_>>(),
            ),
            &mut coins,
        );
        let plan = NormPscPlan {
            shape,
            columns: 4,
            variables: 3,
            groups,
        };
        let first = NormPscFirst {
            x: x_message,
            sum: sum_message,
            squares: squares_message,
            mean: mean_message,
            variance: variance_message,
        };
        let mut psc_coins = ByteMaskCoins::new(Digest([0x91; 32])).unwrap();
        let second = prove_norm_psc(&plan, &rows, &first, &mut psc_coins).unwrap();
        (plan, rows, first, second)
    }

    #[test]
    fn skeleton_accepts_the_honest_toy_messages() {
        let (plan, _rows, first, second) = fixture();
        verify_norm_psc(&plan, &first, &second).unwrap();
    }

    #[test]
    fn encoded_messages_round_trip_and_still_verify() {
        let (plan, _rows, first, second) = fixture();
        let first_bytes = first.encode().unwrap();
        let second_bytes = second.encode().unwrap();
        let decoded_first = NormPscFirst::decode(&plan, &first_bytes).unwrap();
        let decoded_second = NormPscSecond::decode(&plan, &second_bytes).unwrap();
        assert_eq!(decoded_first, first);
        assert_eq!(decoded_second, second);
        verify_norm_psc(&plan, &decoded_first, &decoded_second).unwrap();
    }

    #[test]
    fn wrong_root_or_opening_is_rejected() {
        let (plan, _rows, first, _second) = fixture();
        let mut changed = first.clone();
        changed.x.message.roots[0] = Digest([0x99; 32]);
        assert!(verify_norm_psc(&plan, &changed, &_second).is_err());

        let mut changed = first.clone();
        changed.sum.positions[0] = vec![1];
        assert!(verify_norm_psc(&plan, &changed, &_second).is_err());

        let mut changed = first.clone();
        changed.mean.message.group.group = 9;
        assert!(verify_norm_psc(&plan, &changed, &_second).is_err());
    }

    #[test]
    fn tampered_point_or_end_value_is_rejected() {
        let (plan, _rows, first, second) = fixture();

        let mut changed = second.clone();
        changed.point_values[1][0] = changed.point_values[1][0].add(&Fp3::ONE);
        assert!(verify_norm_psc(&plan, &first, &changed).is_err());

        let mut changed = second.clone();
        changed.psc[0].end_values[0] = changed.psc[0].end_values[0].add(&Fp3::ONE);
        assert!(verify_norm_psc(&plan, &first, &changed).is_err());
    }

    #[test]
    fn changed_plan_or_version_is_rejected() {
        let (mut plan, _rows, first, second) = fixture();
        plan.groups.x.roots[0] = Digest([0x99; 32]);
        assert!(verify_norm_psc(&plan, &first, &second).is_err());

        let (plan, _rows, first, _second) = fixture();
        let mut bytes = first.encode().unwrap();
        bytes[0] ^= 1;
        assert!(NormPscFirst::decode(&plan, &bytes).is_err());

        let mut bytes = second.encode().unwrap();
        bytes.push(0);
        assert!(NormPscSecond::decode(&plan, &bytes).is_err());
    }
    fn integrated_group(plan: &NormPscGroupPlan) -> NormGroupPlan {
        NormGroupPlan {
            shape: plan.shape.clone(),
            columns: plan.columns,
            group: plan.group,
            oracle_base: plan.oracle_base,
            openings: plan.openings,
        }
    }

    fn integrated_fixture() -> (NormPlan, NormFirstMessages, NormPscSecond) {
        let (psc, _rows, first, second) = fixture();
        let plan = NormPlan {
            shape: psc.shape,
            columns: psc.columns,
            variables: psc.variables,
            x: integrated_group(&psc.groups.x),
            x_roots: psc.groups.x.roots.clone(),
            sum: integrated_group(&psc.groups.sum),
            squares: integrated_group(&psc.groups.squares),
            mean: integrated_group(&psc.groups.mean),
            variance: integrated_group(&psc.groups.variance),
            mask: None,
        };
        let message = NormFirstMessages {
            x: first.x.clone(),
            sum: first.sum.clone(),
            squares: first.squares.clone(),
            mean: first.mean.clone(),
            variance: first.variance.clone(),
            mask: None,
        };
        (plan, message, second)
    }

    #[test]
    fn integrated_first_round_trips_with_trusted_x_and_witness_roots() {
        let (plan, message, second) = integrated_fixture();
        let bytes = message.encode().unwrap();
        let decoded = NormFirstMessages::decode(&plan, &bytes).unwrap();
        assert_eq!(decoded, message);
        decoded.check_trusted(&plan).unwrap();
        let second_bytes = second.encode().unwrap();
        let decoded_second = NormPscSecond::decode_for(&plan, &second_bytes).unwrap();
        assert_eq!(decoded_second, second);
    }

    #[test]
    fn integrated_trusted_x_root_and_opening_budget_are_checked() {
        let (plan, message, second) = integrated_fixture();
        let mut changed = message.clone();
        changed.x.message.roots[0] = Digest([0x99; 32]);
        assert!(changed.check_trusted(&plan).is_err());

        let mut changed = message.clone();
        changed.sum.message.group.entries[0].root = Digest([0x99; 32]);
        changed.sum.message.roots[0] = Digest([0x99; 32]);
        assert!(changed.check_trusted(&plan).is_ok());

        let mut changed = plan.clone();
        changed.x.openings = 0;
        assert!(changed.validate().is_err());

        let mut changed = plan.clone();
        changed.squares.shape.main_rows = 0;
        assert!(changed.validate().is_err());

        assert!(NormPscSecond::decode_for(&plan, &second.encode().unwrap()).is_ok());
    }

    fn mask_plan() -> NormMaskPlan {
        NormMaskPlan {
            oracle: 7000,
            group: 7001,
            mask_code: build_code_plan(4, 8, 1).unwrap(),
            mask_rows: 1,
            answers: 1,
            openings: 1,
            tree: zkie_engine::pcs::request::runtime_tree::RuntimeTreePlan {
                tree_number: 7002,
                leaf_class: 1,
                log_n: 4,
                log_n_zk: 4,
                parts: vec![RuntimeTreePart {
                    main_symbols: 0,
                    mask_symbols: 1,
                }],
                salt_bytes: 40,
                run: Digest([0x71; 32]),
                configuration: Digest([0x72; 32]),
            },
            positions: 2,
            hiding: 1,
            shown: 2,
            claims: 1,
        }
    }

    fn mask_message(plan: &NormMaskPlan) -> NormMaskMessage {
        NormMaskMessage {
            tree: plan.tree.clone(),
            root: Digest([0x73; 32]),
            sample: SendSamples {
                seed: vec![zkie_engine::pcs::request::table::SampleSeed {
                    z: Vec::new(),
                    beta: Fp3::ONE,
                }],
                answers: vec![Fp3::ONE],
            },
            positions: vec![0],
            opening: zkie_engine::pcs::request::runtime_tree::RuntimeTreeOpening {
                positions: vec![0],
                columns: Vec::new(),
                given: Vec::new(),
            },
        }
    }

    #[test]
    fn integrated_mask_root_and_opening_shape_are_checked() {
        let (mut plan, mut message, _second) = integrated_fixture();
        let mask = mask_plan();
        plan.mask = Some(mask.clone());
        message.mask = Some(mask_message(&mask));
        plan.validate().unwrap();
        message.check_trusted(&plan).unwrap();

        let mut changed = message.clone();
        changed.mask.as_mut().unwrap().root = Digest::ZERO;
        assert!(changed.check_trusted(&plan).is_err());

        let mut changed = message.clone();
        changed.mask.as_mut().unwrap().tree.tree_number = 9999;
        assert!(changed.check_trusted(&plan).is_err());

        let mut changed = message.clone();
        changed.mask.as_mut().unwrap().positions = vec![1];
        assert!(changed.check_trusted(&plan).is_err());
    }

}
