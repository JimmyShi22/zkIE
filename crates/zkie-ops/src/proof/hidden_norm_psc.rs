//! Committed messages, product-sumcheck instances and row bindings for the
//! final norm core.
//!
//! `NormPlan` fixes the five committed group shapes, their identifiers, the
//! opening counts and the relation domain. The x group is the trusted
//! product-form input: its roots are plan values. The sum, squares, mean and
//! variance groups are proof-fixed witness messages, and the mask group is a
//! committed one-time mask message.
//!
//! The integrated product-sumcheck runs one instance per position and per
//! relation listed by the plan. The supported row binding covers the sum,
//! squares, mean and variance relations: each shown endpoint is a linear form
//! of the committed x, sum, squares, mean and variance messages, and the
//! one-time hiding and shown masks are committed entries of the same mask
//! group. The remainder range and digit checks, the rsqrt lookup, the weight
//! product and the output consumer are not part of this module. No soundness
//! or privacy proof is claimed.

use zkie_core::artifact::digest_bytes;
use zkie_core::common::codec::DecodeLimits;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::chain::Chain;
use zkie_engine::pcs::request::code::encode_row_fp3;
use zkie_engine::pcs::request::fold::{ByteMaskCoins, MaskCoins};
use zkie_engine::pcs::request::field::{Fp3, MODULUS};
use zkie_engine::pcs::request::runtime_tree::{
    RuntimePartRows, RuntimeTree, RuntimeTreeOpening, RuntimeTreePlan,
};
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};

use crate::proof::committed_psc::{
    get_psc_transcript, get_tree_plan, put_psc_transcript, put_tree_plan, CommittedPscGroup,
    PositionPscChallenges, PscTranscriptTags,
};
use crate::proof::hidden_norm::{NormCoreRow, NormShape};
use crate::proof::hidden_send_session::{SendSamples, WShape};
use crate::proof::output_closure::{CommittedValueGroup, RetainedTrees};
use crate::proof::psc::{
    hp_entries, prove_dot_sumcheck, psc_variables, verify_psc_transcript, DotClaim,
    DotMaskEntries, DotOutput, DotProof, SHOWN_MAPS,
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
pub(crate) const NORM_PSC_TAGS: PscTranscriptTags = PscTranscriptTags {
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
            .checked_mul(plan.relations.len() as u32)
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
        let instances = plan.shape.positions as usize * plan.relations.len();
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

pub(crate) fn rel_map(
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
// The integrated norm plan and first/second messages.
//
// The trusted fixture input is the product-form x group: its roots are plan
// values. The sum/squares/mean/variance groups are proof-fixed witness
// messages: the plan fixes their shapes, identifiers and opening counts, while
// their roots arrive in the first message. The mask message is committed in
// the first stage and its entries are consumed by the second-stage product-
// sumcheck and the row binding.

/// Version of the integrated norm plan and first message.
pub const NORM_BIND_VERSION: u64 = 3;

/// The relation indices the bound core supports. The map construction of
/// every relation is fixed by the relation index; the plan may list only
/// relations whose row binding is implemented.
pub const NORM_BOUND_RELATIONS: [usize; 4] = [0, 1, 2, 3];

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

/// The trusted plan of the rounding witness group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormRoundPlan {
    pub group: NormGroupPlan,
    pub layout: NormRoundLayout,
    /// The trusted input bound `X` of the fixture contract, in integer units.
    pub input_bound: u64,
}

/// The trusted plan of the integrated norm relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormPlan {
    pub shape: NormShape,
    pub columns: u64,
    pub variables: usize,
    /// The relation indices of the product-sumcheck instances, in transcript
    /// order. The plan fixes the domain; only the relations whose row binding
    /// is implemented may be listed.
    pub relations: Vec<usize>,
    /// The trusted product-form x group. Its roots are plan values.
    pub x: NormGroupPlan,
    pub x_roots: Vec<Digest>,
    /// The proof-fixed sum/squares/mean/variance witness groups.
    pub sum: NormGroupPlan,
    pub squares: NormGroupPlan,
    pub mean: NormGroupPlan,
    pub variance: NormGroupPlan,
    /// The committed rounding witness group; optional in this stage.
    pub round: Option<NormRoundPlan>,
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
        if self.relations.is_empty()
            || self.relations.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(Error::InvalidConfig);
        }
        if self
            .relations
            .iter()
            .any(|relation| !NORM_BOUND_RELATIONS.contains(relation))
        {
            return Err(Error::UnsupportedPrimitive);
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
                if groups[left].oracle_base == groups[right].oracle_base {
                    return Err(Error::InvalidConfig);
                }
            }
        }
        if let Some(round) = &self.round {
            if round.input_bound == 0 {
                return Err(Error::InvalidConfig);
            }
            let columns = round.layout.widths.columns()?;
            if round.group.columns != columns
                || round.group.fault(self.shape.positions).is_some()
                || round.group.shape.answers != 1
            {
                return Err(Error::InvalidConfig);
            }
            if norm_no_wrap(round.input_bound, self.shape.valid, &round.layout.widths).is_err()
            {
                return Err(Error::InvalidConfig);
            }
            for group in [&self.x, &self.sum, &self.squares, &self.mean, &self.variance] {
                if round.group.oracle_base == group.oracle_base {
                    return Err(Error::InvalidConfig);
                }
            }
        }
        if let Some(mask) = &self.mask {
            mask.validate(
                self.shape.positions,
                self.variables,
                self.relations.len(),
            )?;
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
        writer.put_u64(self.relations.len() as u64);
        for relation in &self.relations {
            writer.put_u64(*relation as u64);
        }
        put_norm_group(writer, &self.x);
        writer.put_u64(self.x_roots.len() as u64);
        for root in &self.x_roots {
            writer.put_digest(root);
        }
        put_norm_group(writer, &self.sum);
        put_norm_group(writer, &self.squares);
        put_norm_group(writer, &self.mean);
        put_norm_group(writer, &self.variance);
        match &self.round {
            Some(round) => {
                writer.put_u64(1);
                writer.put_u64(round.input_bound);
                writer.put_u64(round.layout.widths.sum as u64);
                writer.put_u64(round.layout.widths.squares as u64);
                writer.put_u64(round.layout.widths.variance as u64);
                writer.put_u64(round.layout.widths.mean as u64);
                writer.put_u64(round.layout.widths.offset as u64);
                put_norm_group(writer, &round.group);
            }
            None => writer.put_u64(0),
        }
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
    /// The number of allocated entries of one product-sumcheck instance.
    pub fn per_instance(&self) -> Result<usize> {
        self.hiding
            .checked_add(self.shown)
            .ok_or(Error::ResourceLimit)
    }

    /// The number of product-sumcheck instances of the plan.
    pub fn instances(&self, relations: usize) -> Result<usize> {
        (self.positions as usize)
            .checked_mul(relations)
            .ok_or(Error::ResourceLimit)
    }

    /// The total number of allocated entries: one block per instance plus the
    /// reserved claim entries.
    pub fn total_entries(&self, relations: usize) -> Result<usize> {
        self.instances(relations)?
            .checked_mul(self.per_instance()?)
            .and_then(|value| value.checked_add(self.claims))
            .ok_or(Error::ResourceLimit)
    }

    /// The compact slot of one hiding entry of one instance.
    pub fn hiding_slot(&self, instance: usize, index: usize) -> Result<usize> {
        if index >= self.hiding {
            return Err(Error::InvalidIndex);
        }
        instance
            .checked_mul(self.per_instance()?)
            .and_then(|base| base.checked_add(index))
            .ok_or(Error::ResourceLimit)
    }

    /// The compact slot of one shown-value mask of one instance.
    pub fn shown_slot(&self, instance: usize, index: usize) -> Result<usize> {
        if index >= self.shown {
            return Err(Error::InvalidIndex);
        }
        instance
            .checked_mul(self.per_instance()?)
            .and_then(|base| base.checked_add(self.hiding))
            .and_then(|base| base.checked_add(index))
            .ok_or(Error::ResourceLimit)
    }

    /// The compact slot of one reserved claim entry.
    pub fn claim_slot(&self, instance_count: usize, index: usize) -> Result<usize> {
        if index >= self.claims {
            return Err(Error::InvalidIndex);
        }
        instance_count
            .checked_mul(self.per_instance()?)
            .and_then(|base| base.checked_add(index))
            .ok_or(Error::ResourceLimit)
    }

    /// The compact row-major offset of an allocated entry.
    pub fn entry_offset(&self, slot: usize) -> Result<usize> {
        let rows = usize::try_from(self.mask_rows).map_err(|_| Error::ResourceLimit)?;
        let ell = usize::try_from(self.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        let length = rows.checked_mul(ell).ok_or(Error::ResourceLimit)?;
        if slot >= length {
            return Err(Error::InvalidIndex);
        }
        Ok(slot)
    }

    /// The BAT virtual table index of an allocated entry.
    pub fn entry_virtual_index(&self, slot: usize) -> Result<u64> {
        let offset = self.entry_offset(slot)?;
        let ell = usize::try_from(self.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        let row = offset / ell;
        let place = offset % ell;
        let row_bits = ceil_log2_u64(self.mask_rows);
        (row as u64)
            .checked_add((place as u64) << row_bits)
            .ok_or(Error::ResourceLimit)
    }

    /// The compact offset of the SEND pad: the last cell of the last row.
    pub fn pad_offset(&self) -> Result<usize> {
        let rows = usize::try_from(self.mask_rows).map_err(|_| Error::ResourceLimit)?;
        let ell = usize::try_from(self.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        rows.checked_mul(ell)
            .and_then(|value| value.checked_sub(1))
            .ok_or(Error::ShapeMismatch)
    }

    /// The number of seed coordinates the mask sample form reads.
    pub fn sample_bits(&self) -> Result<u32> {
        ceil_log2_u64(self.mask_rows)
            .checked_add(self.mask_code.log_ell())
            .ok_or(Error::ResourceLimit)
    }

    /// The free entries of the mask rows, excluding the SEND pad.
    pub fn free_entries(&self) -> Result<usize> {
        let rows = usize::try_from(self.mask_rows).map_err(|_| Error::ResourceLimit)?;
        let ell = usize::try_from(self.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        let answers = self.answers as usize;
        if answers == 0 || answers > ell {
            return Err(Error::InvalidConfig);
        }
        let per_row = ell.checked_sub(answers).ok_or(Error::ResourceLimit)?;
        rows.checked_mul(per_row)
            .and_then(|value| value.checked_add(rows.saturating_sub(1)))
            .ok_or(Error::ResourceLimit)
    }

    fn validate(&self, positions: u32, variables: usize, relations: usize) -> Result<()> {
        if self.oracle == 0 || self.group == 0 {
            return Err(Error::InvalidConfig);
        }
        if self.mask_rows == 0
            || !self.mask_rows.is_power_of_two()
            || self.answers != 1
            || self.openings == 0
        {
            return Err(Error::InvalidConfig);
        }
        if self.openings as u64 > self.mask_code.r {
            return Err(Error::InvalidConfig);
        }
        if self.positions != positions || self.shown != SHOWN_MAPS || self.claims == 0 {
            return Err(Error::InvalidConfig);
        }
        let variables = psc_variables(variables).map_err(|_| Error::InvalidConfig)?;
        if self.hiding != hp_entries(&variables).map_err(|_| Error::InvalidConfig)? {
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
        if self.tree.log_n != self.mask_code.log_n
            || self.tree.log_n_zk != self.mask_code.log_n
            || self.tree.parts.len() != 1
        {
            return Err(Error::InvalidConfig);
        }
        let mask_symbols = self
            .mask_rows
            .checked_mul(
                zkie_engine::pcs::request::field::E_DEGREE as u64,
            )
            .ok_or(Error::ResourceLimit)?;
        if self.tree.parts[0].main_symbols != 0
            || self.tree.parts[0].mask_symbols != mask_symbols
        {
            return Err(Error::InvalidConfig);
        }
        if self.total_entries(relations)? > self.free_entries()? {
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

fn ceil_log2_u64(value: u64) -> u32 {
    if value <= 1 {
        return 0;
    }
    u64::BITS - (value - 1).leading_zeros()
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

/// The prover-side message rows and encoding material of one norm group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormGroupWitness {
    /// `messages[position][row][place]`, the unencoded message rows.
    pub messages: Vec<Vec<Vec<Fp3>>>,
    /// `tails[position][row * r + i]`.
    pub tails: Vec<Vec<Fp3>>,
    /// `salts[position][column]`.
    pub salts: Vec<Vec<Vec<u8>>>,
    /// One non-zero sample pad per position.
    pub pads: Vec<Fp3>,
    /// The committed group, including the retained runtime trees.
    pub committed: CommittedValueGroup,
}

/// The committed mask-only norm message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommittedNormMask {
    pub root: Digest,
    pub rows: Vec<Vec<Fp3>>,
    pub tails: Vec<Vec<Fp3>>,
    pub salts: Vec<Vec<u8>>,
}

impl CommittedNormMask {
    pub fn tree(&self, plan: &NormMaskPlan) -> Result<RuntimeTree> {
        let mut codewords = Vec::with_capacity(self.rows.len());
        for (row, tail) in self.rows.iter().zip(&self.tails) {
            codewords.push(encode_row_fp3(&plan.mask_code, row, tail)?);
        }
        RuntimeTree::commit(
            plan.tree.clone(),
            &[RuntimePartRows {
                main: Vec::new(),
                mask: codewords,
                main_deg: 1,
            }],
            &self.salts,
        )
    }

    /// The compact mask entry at one slot.
    pub fn entry(&self, plan: &NormMaskPlan, slot: usize) -> Result<Fp3> {
        let offset = plan.entry_offset(slot)?;
        let ell = usize::try_from(plan.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        self.rows
            .get(offset / ell)
            .and_then(|row| row.get(offset % ell))
            .copied()
            .ok_or(Error::ShapeMismatch)
    }
}

/// Commits the mask-only norm message from its private rows.
pub fn commit_norm_mask(
    plan: &NormMaskPlan,
    rows: Vec<Vec<Fp3>>,
    tails: Vec<Vec<Fp3>>,
    salts: Vec<Vec<u8>>,
) -> Result<CommittedNormMask> {
    let committed = CommittedNormMask {
        root: Digest::ZERO,
        rows,
        tails,
        salts,
    };
    let root = *committed.tree(plan)?.root();
    Ok(CommittedNormMask { root, ..committed })
}

/// The compact mask entries of every product-sumcheck instance, in instance
/// order: hiding entries first and the two shown masks last.
pub fn norm_mask_entries(
    plan: &NormMaskPlan,
    mask: &CommittedNormMask,
    relations: usize,
) -> Result<Vec<DotMaskEntries>> {
    let instances = plan.instances(relations)?;
    let mut entries = Vec::with_capacity(instances);
    for instance in 0..instances {
        let mut hiding = Vec::with_capacity(plan.hiding);
        for index in 0..plan.hiding {
            hiding.push(mask.entry(plan, plan.hiding_slot(instance, index)?)?);
        }
        let shown = [
            mask.entry(plan, plan.shown_slot(instance, 0)?)?,
            mask.entry(plan, plan.shown_slot(instance, 1)?)?,
        ];
        entries.push(DotMaskEntries { hiding, shown });
    }
    Ok(entries)
}

/// The prover-side mask witness of the integrated norm relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormMaskWitness {
    pub committed: CommittedNormMask,
    /// One mask entry set per product-sumcheck instance.
    pub entries: Vec<DotMaskEntries>,
}

/// The prover-side witness of the integrated norm groups.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormWitness {
    /// The private integer rows the norm relation consumes.
    pub rows: Vec<Vec<i64>>,
    /// The exact integer core of every row.
    pub core: Vec<NormCoreRow>,
    pub x: NormGroupWitness,
    pub sum: NormGroupWitness,
    pub squares: NormGroupWitness,
    pub mean: NormGroupWitness,
    pub variance: NormGroupWitness,
    pub mask: Option<NormMaskWitness>,
}

/// One checked product-sumcheck instance of the norm relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormPositionBinding {
    pub position: u32,
    /// The relation index, in the plan's relation order.
    pub relation: usize,
    pub output: DotOutput,
}

/// The checked product-sumcheck bindings of every instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormRowBinding {
    pub positions: Vec<NormPositionBinding>,
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
        if plan.round.is_some() {
            return Err(Error::UnsupportedPrimitive);
        }
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

/// The fixed byte widths of the rounding witness rows.
///
/// The widths are plan values. The offset width bounds both `u = r - lo` and
/// `s = valid - 1 - u`; the other widths bound the sum, squares, variance and
/// mean magnitudes. A width may not exceed seven bytes: a wider row could
/// exceed the base modulus and the digit reconstruction would stop being a
/// canonical field representative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormRoundWidths {
    pub sum: usize,
    pub squares: usize,
    pub variance: usize,
    pub mean: usize,
    pub offset: usize,
}

impl NormRoundWidths {
    pub fn fault(&self, valid: u32) -> Option<&'static str> {
        for width in [self.sum, self.squares, self.variance, self.mean, self.offset] {
            if width == 0 || width > 7 {
                return Some("a rounding witness width is out of range");
            }
            if digit_bound(width).ok()? >= MODULUS as u128 {
                return Some("a rounding witness width does not fit the modulus");
            }
        }
        let capacity = digit_bound(self.offset).ok()? + 1;
        if capacity < valid as u128 {
            return Some("the offset width does not cover the valid column range");
        }
        None
    }

    pub fn slots(&self) -> Result<usize> {
        self.offset
            .checked_mul(4)
            .and_then(|offset| offset.checked_add(self.sum))
            .and_then(|value| value.checked_add(self.squares))
            .and_then(|value| value.checked_add(self.variance))
            .and_then(|value| value.checked_add(self.mean))
            .and_then(|value| value.checked_add(NormRoundLayout::FIXED_SLOTS))
            .ok_or(Error::ResourceLimit)
    }

    pub fn columns(&self) -> Result<u64> {
        let slots = self.slots()?;
        u64::try_from(slots)
            .ok()
            .and_then(|value| value.checked_next_power_of_two())
            .ok_or(Error::ResourceLimit)
    }
}

fn digit_bound(width: usize) -> Result<u128> {
    256u128
        .checked_pow(u32::try_from(width).map_err(|_| Error::ResourceLimit)?)
        .and_then(|value| value.checked_sub(1))
        .ok_or(Error::ResourceLimit)
}

fn digit_capacity(width: usize) -> Result<u128> {
    256u128
        .checked_pow(u32::try_from(width).map_err(|_| Error::ResourceLimit)?)
        .ok_or(Error::ResourceLimit)
}

/// The slot positions of the rounding witness message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormRoundSlots {
    pub mean_remainder: usize,
    pub mean_offset: usize,
    pub mean_complement: usize,
    pub variance_remainder: usize,
    pub variance_offset: usize,
    pub variance_complement: usize,
    pub mean_negative: usize,
    pub sum_negative: usize,
    pub sum_digits: usize,
    pub mean_digits: usize,
    pub squares_digits: usize,
    pub variance_digits: usize,
    pub mean_offset_digits: usize,
    pub mean_complement_digits: usize,
    pub variance_offset_digits: usize,
    pub variance_complement_digits: usize,
    pub total: usize,
}

/// The plan-fixed layout of one rounding witness message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormRoundLayout {
    pub widths: NormRoundWidths,
    pub slots: NormRoundSlots,
}

impl NormRoundLayout {
    pub const FIXED_SLOTS: usize = 8;

    pub fn new(widths: NormRoundWidths, valid: u32) -> Result<Self> {
        if widths.fault(valid).is_some() {
            return Err(Error::InvalidConfig);
        }
        let mean_remainder = 0;
        let mean_offset = 1;
        let mean_complement = 2;
        let variance_remainder = 3;
        let variance_offset = 4;
        let variance_complement = 5;
        let mean_negative = 6;
        let sum_negative = 7;
        let mut next = Self::FIXED_SLOTS;
        let sum_digits = next;
        next = next.checked_add(widths.sum).ok_or(Error::ResourceLimit)?;
        let squares_digits = next;
        next = next
            .checked_add(widths.squares)
            .ok_or(Error::ResourceLimit)?;
        let variance_digits = next;
        next = next
            .checked_add(widths.variance)
            .ok_or(Error::ResourceLimit)?;
        let mean_digits = next;
        next = next
            .checked_add(widths.mean)
            .ok_or(Error::ResourceLimit)?;
        let mean_offset_digits = next;
        next = next
            .checked_add(widths.offset)
            .ok_or(Error::ResourceLimit)?;
        let mean_complement_digits = next;
        next = next
            .checked_add(widths.offset)
            .ok_or(Error::ResourceLimit)?;
        let variance_offset_digits = next;
        next = next
            .checked_add(widths.offset)
            .ok_or(Error::ResourceLimit)?;
        let variance_complement_digits = next;
        next = next
            .checked_add(widths.offset)
            .ok_or(Error::ResourceLimit)?;
        if next != widths.slots()? {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            widths,
            slots: NormRoundSlots {
                mean_remainder,
                mean_offset,
                mean_complement,
                variance_remainder,
                variance_offset,
                variance_complement,
                mean_negative,
                sum_negative,
                sum_digits,
                mean_digits,
                squares_digits,
                variance_digits,
                mean_offset_digits,
                mean_complement_digits,
                variance_offset_digits,
                variance_complement_digits,
                total: next,
            },
        })
    }

    /// Writes one integer core row into the fixed slot vector. Digit rows are
    /// zero padded to their fixed width.
    pub fn values(&self, core: &NormCoreRow, columns: u64) -> Result<Vec<Fp3>> {
        let columns = usize::try_from(columns).map_err(|_| Error::ResourceLimit)?;
        if columns < self.slots.total || !columns.is_power_of_two() {
            return Err(Error::ShapeMismatch);
        }
        let mut values = vec![Fp3::ZERO; columns];
        values[self.slots.mean_remainder] = fp3_signed(core.mean_remainder)?;
        values[self.slots.mean_offset] = fp3_signed(core.mean_offset)?;
        values[self.slots.mean_complement] = fp3_signed(core.mean_complement)?;
        values[self.slots.variance_remainder] = fp3_signed(core.variance_remainder)?;
        values[self.slots.variance_offset] = fp3_signed(core.variance_offset)?;
        values[self.slots.variance_complement] = fp3_signed(core.variance_complement)?;
        values[self.slots.mean_negative] = if core.mean_negative {
            Fp3::ONE
        } else {
            Fp3::ZERO
        };
        values[self.slots.sum_negative] = if core.sum_negative {
            Fp3::ONE
        } else {
            Fp3::ZERO
        };
        write_digits(
            &mut values[self.slots.sum_digits..self.slots.sum_digits + self.widths.sum],
            &core.sum_digits,
        )?;
        write_digits(
            &mut values[self.slots.squares_digits..self.slots.squares_digits + self.widths.squares],
            &core.squares_digits,
        )?;
        write_digits(
            &mut values
                [self.slots.variance_digits..self.slots.variance_digits + self.widths.variance],
            &core.variance_digits,
        )?;
        write_digits(
            &mut values[self.slots.mean_digits..self.slots.mean_digits + self.widths.mean],
            &core.mean_digits,
        )?;
        write_digits(
            &mut values
                [self.slots.mean_offset_digits..self.slots.mean_offset_digits + self.widths.offset],
            &core.mean_offset_digits,
        )?;
        write_digits(
            &mut values[self.slots.mean_complement_digits
                ..self.slots.mean_complement_digits + self.widths.offset],
            &core.mean_complement_digits,
        )?;
        write_digits(
            &mut values[self.slots.variance_offset_digits
                ..self.slots.variance_offset_digits + self.widths.offset],
            &core.variance_offset_digits,
        )?;
        write_digits(
            &mut values[self.slots.variance_complement_digits
                ..self.slots.variance_complement_digits + self.widths.offset],
            &core.variance_complement_digits,
        )?;
        Ok(values)
    }
}

fn write_digits(slot: &mut [Fp3], digits: &[u8]) -> Result<()> {
    if digits.len() > slot.len() {
        return Err(Error::BoundViolation);
    }
    for (target, digit) in slot.iter_mut().zip(digits) {
        *target = Fp3::from_u64(*digit as u64);
    }
    Ok(())
}

/// The integer no-wrap bounds of one position with the trusted input bound
/// `X`, the valid width `d` and the fixed digit widths. Every bound must stay
/// below `p/2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormNoWrap {
    pub relation_sum: u128,
    pub relation_squares: u128,
    pub relation_mean: u128,
    pub relation_variance: u128,
    pub reconstruction: u128,
    pub offsets: u128,
    pub remainder: u128,
}

pub fn norm_no_wrap(
    input_bound: u64,
    valid: u32,
    widths: &NormRoundWidths,
) -> Result<NormNoWrap> {
    if valid == 0 || widths.fault(valid).is_some() {
        return Err(Error::InvalidConfig);
    }
    let x = input_bound as u128;
    let d = valid as u128;
    let bv = digit_bound(widths.sum)?;
    let bs = digit_bound(widths.squares)?;
    let bvar = digit_bound(widths.variance)?;
    let bq = digit_bound(widths.mean)?;
    let bu = digit_bound(widths.offset)?;
    let relation_sum = d
        .checked_mul(x)
        .and_then(|value| value.checked_add(bv))
        .ok_or(Error::ResourceLimit)?;
    let sum_x_q = x.checked_add(bq).ok_or(Error::ResourceLimit)?;
    let relation_squares = d
        .checked_mul(sum_x_q.checked_mul(sum_x_q).ok_or(Error::ResourceLimit)?)
        .and_then(|value| value.checked_add(bs))
        .ok_or(Error::ResourceLimit)?;
    let relation_mean = bv
        .checked_add(bq.checked_mul(d).ok_or(Error::ResourceLimit)?)
        .and_then(|value| value.checked_add(d))
        .ok_or(Error::ResourceLimit)?;
    let relation_variance = bs
        .checked_add(bvar.checked_mul(d).ok_or(Error::ResourceLimit)?)
        .and_then(|value| value.checked_add(d))
        .ok_or(Error::ResourceLimit)?;
    let reconstruction = [
        digit_capacity(widths.sum)?,
        digit_capacity(widths.squares)?,
        digit_capacity(widths.variance)?,
        digit_capacity(widths.mean)?,
        digit_capacity(widths.offset)?,
    ]
    .into_iter()
    .max()
    .ok_or(Error::InvalidConfig)?;
    let offsets = bu
        .checked_mul(2)
        .and_then(|value| value.checked_add(d))
        .ok_or(Error::ResourceLimit)?;
    let remainder = d.checked_mul(3).ok_or(Error::ResourceLimit)?;
    let half = (MODULUS as u128) / 2;
    if relation_sum >= half
        || relation_squares >= half
        || relation_mean >= half
        || relation_variance >= half
        || reconstruction >= half
        || offsets >= half
        || remainder >= half
    {
        return Err(Error::InvalidConfig);
    }
    Ok(NormNoWrap {
        relation_sum,
        relation_squares,
        relation_mean,
        relation_variance,
        reconstruction,
        offsets,
        remainder,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real_widths() -> NormRoundWidths {
        NormRoundWidths {
            sum: 5,
            squares: 7,
            variance: 6,
            mean: 3,
            offset: 2,
        }
    }

    #[test]
    fn round_layout_fixes_the_slot_map_and_pads_digits() {
        let widths = real_widths();
        let layout = NormRoundLayout::new(widths, 768).unwrap();
        assert_eq!(layout.slots.total, 37);
        assert_eq!(widths.columns().unwrap(), 64);
        let core = NormCoreRow::from_values(&[3, -2, 5, 1], 4).unwrap();
        let values = layout.values(&core, 64).unwrap();
        assert_eq!(values.len(), 64);
        assert_eq!(
            values[layout.slots.mean_remainder],
            fp3_signed(core.mean_remainder).unwrap()
        );
        assert_eq!(
            values[layout.slots.variance_complement],
            fp3_signed(core.variance_complement).unwrap()
        );
        assert_eq!(values[layout.slots.sum_negative], Fp3::ZERO);
        for (offset, digit) in core.mean_digits.iter().enumerate() {
            assert_eq!(
                values[layout.slots.mean_digits + offset],
                Fp3::from_u64(*digit as u64)
            );
        }
        for slot in layout.slots.mean_digits + core.mean_digits.len()
            ..layout.slots.mean_digits + widths.mean
        {
            assert_eq!(values[slot], Fp3::ZERO);
        }
        for slot in layout.slots.total..64 {
            assert_eq!(values[slot], Fp3::ZERO);
        }
    }

    #[test]
    fn round_widths_and_digit_rows_fail_closed() {
        let mut widths = real_widths();
        widths.squares = 8;
        assert!(NormRoundLayout::new(widths, 768).is_err());
        let mut widths = real_widths();
        widths.offset = 1;
        assert!(NormRoundLayout::new(widths, 768).is_err());
        let mut widths = real_widths();
        widths.squares = 6;
        let layout = NormRoundLayout::new(widths, 768).unwrap();
        let core = NormCoreRow::from_values(&[22_855_985, -22_855_985], 2).unwrap();
        assert_eq!(core.squares_digits.len(), 7);
        assert!(layout.values(&core, 64).is_err());
        let layout = NormRoundLayout::new(real_widths(), 768).unwrap();
        assert!(layout.values(&core, 32).is_err());
    }

    #[test]
    fn real_bounds_stay_below_the_modulus_half() {
        let bounds = norm_no_wrap(22_855_985, 768, &real_widths()).unwrap();
        assert_eq!(bounds.relation_sum, 1_117_065_024_255);
        assert_eq!(bounds.relation_squares, 1_278_424_730_478_247_935);
        assert_eq!(bounds.relation_mean, 1_112_396_529_663);
        assert_eq!(bounds.relation_variance, 288_230_376_151_711_743);
        assert_eq!(bounds.reconstruction, 72_057_594_037_927_936);
        assert_eq!(bounds.offsets, 131_838);
        assert_eq!(bounds.remainder, 2_304);
        let mut widths = real_widths();
        widths.mean = 4;
        assert!(norm_no_wrap(22_855_985, 768, &widths).is_err());
    }

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
            relations: vec![0, 1, 2, 3],
            x: integrated_group(&psc.groups.x),
            x_roots: psc.groups.x.roots.clone(),
            sum: integrated_group(&psc.groups.sum),
            squares: integrated_group(&psc.groups.squares),
            mean: integrated_group(&psc.groups.mean),
            variance: integrated_group(&psc.groups.variance),
            round: None,
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
    #[test]
    fn round_plan_fixes_the_layout_and_the_input_bound() {
        let (mut plan, _, _) = integrated_fixture();
        let widths = NormRoundWidths {
            sum: 2,
            squares: 2,
            variance: 2,
            mean: 2,
            offset: 1,
        };
        let layout = NormRoundLayout::new(widths, plan.shape.valid).unwrap();
        let columns = widths.columns().unwrap();
        let group = NormGroupPlan {
            shape: group_shape(columns, 6),
            columns,
            group: plan.variance.group,
            oracle_base: 9500,
            openings: 1,
        };
        plan.round = Some(NormRoundPlan {
            group,
            layout,
            input_bound: 1000,
        });
        plan.validate().unwrap();
        let with_round = plan.digest();

        let mut wide = plan.clone();
        let widths = NormRoundWidths {
            mean: 4,
            ..widths
        };
        let layout = NormRoundLayout::new(widths, plan.shape.valid).unwrap();
        let columns = widths.columns().unwrap();
        wide.round = Some(NormRoundPlan {
            group: NormGroupPlan {
                shape: group_shape(columns, 7),
                columns,
                group: wide.variance.group,
                oracle_base: 9501,
                openings: 1,
            },
            layout,
            input_bound: 1000,
        });
        assert!(wide.validate().is_err());

        let mut colliding = plan.clone();
        let x_oracle = colliding.x.oracle_base;
        if let Some(round) = &mut colliding.round {
            round.group.oracle_base = x_oracle;
        }
        assert!(colliding.validate().is_err());
        assert_ne!(with_round, {
            let mut plain = plan.clone();
            plain.round = None;
            plain.digest()
        });
    }

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
            mask_rows: 16,
            answers: 1,
            openings: 1,
            tree: zkie_engine::pcs::request::runtime_tree::RuntimeTreePlan {
                tree_number: 7002,
                leaf_class: 1,
                log_n: 4,
                log_n_zk: 4,
                parts: vec![RuntimeTreePart {
                    main_symbols: 0,
                    mask_symbols: 48,
                }],
                salt_bytes: 40,
                run: Digest([0x71; 32]),
                configuration: Digest([0x72; 32]),
            },
            positions: 2,
            hiding: 9,
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
