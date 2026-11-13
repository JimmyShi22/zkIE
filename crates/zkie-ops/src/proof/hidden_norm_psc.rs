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
use zkie_engine::pcs::request::table::{ColumnOpening, SampleSeed};
use zkie_engine::pcs::request::runtime_tree::{
    RuntimePartRows, RuntimeTree, RuntimeTreePart, RuntimeTreePlan,
};
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};

use crate::proof::committed_psc::{
    get_psc_transcript, get_tree_plan, put_psc_transcript, put_tree_plan, CommittedPscGroup,
    PositionPscChallenges, PscTranscriptTags,
};
use crate::proof::hidden_norm::{NormCoreRow, NormShape};
use crate::proof::hidden_send_session::{
    get_column_opening, put_column_opening, SendSamples, WShape,
};
use crate::proof::lookup::{
    LookupKeyPlan, LookupKeySource, LookupOutcome, LookupPlan, LookupProof,
};
use crate::proof::lookup_table::{TableConfig, TABLE_ENCODING_VERSION};
use crate::proof::lookup_messages::LookupCommitPlan;
use crate::proof::norm_param_table::{
    NormParamClaimPlan, NormParamTableIdentity, NormParamTablesPlan,
};
use crate::proof::output_closure::CommittedValueGroup;
use crate::registry::{RSQRT_COARSE, RSQRT_COARSE_END, RSQRT_FINE, RSQRT_HALF, RSQRT_SHIFT};
use crate::proof::psc::{
    hp_entries, prove_dot_sumcheck, psc_variables, verify_psc_transcript, DotClaim,
    DotMaskEntries, DotOutput, DotProof, PscRound, SHOWN_MAPS,
};
use crate::proof::psc_general::GeneralProof;

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
    claim: 53,
};
/// Transcript tags of the pointwise product instances, distinct from the
/// relation transcript tags.
pub(crate) const NORM_OUTPUT_PSC_TAGS: PscTranscriptTags = PscTranscriptTags {
    version: NORM_PSC_VERSION,
    hiding_sum: 60,
    round: 61,
    end: 62,
    claim: 68,
};
/// The prefix tag of the pointwise column point.
pub(crate) const NORM_OUTPUT_POINT_TAG: u64 = 63;
/// The prefix tag of the bias addition point.
pub(crate) const NORM_BIAS_POINT_TAG: u64 = 67;
/// Transcript tags of the tail relation instances, distinct from the relation
/// and pointwise tags.
pub(crate) const NORM_TAIL_PSC_TAGS: PscTranscriptTags = PscTranscriptTags {
    version: NORM_PSC_VERSION,
    hiding_sum: 64,
    round: 65,
    end: 66,
    claim: 69,
};
/// Version of the pointwise transcript section.
pub const NORM_OUTPUT_PSC_VERSION: u64 = 1;

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

/// The public second-stage transcript of the pointwise product instances:
/// the claim value and the general product-sumcheck proof of every instance,
/// in position-major order (broadcast/gain product, then centered/product
/// image).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormOutputSecond {
    pub claims: Vec<Fp3>,
    pub proofs: Vec<GeneralProof>,
}

fn put_general_proof(writer: &mut CanonicalWriter, proof: &GeneralProof) {
    writer.put_fp3(&proof.hiding_sum);
    writer.put_u64(proof.rounds.len() as u64);
    for round in &proof.rounds {
        writer.put_u64(round.coefficients.len() as u64);
        for coefficient in &round.coefficients {
            writer.put_fp3(coefficient);
        }
    }
    writer.put_u64(proof.y_round.coefficients.len() as u64);
    for coefficient in &proof.y_round.coefficients {
        writer.put_fp3(coefficient);
    }
    writer.put_u64(proof.end_values.len() as u64);
    for value in &proof.end_values {
        writer.put_fp3(value);
    }
}

fn get_general_proof(reader: &mut CanonicalReader, rounds: usize) -> Result<GeneralProof> {
    let hiding_sum = reader.get_fp3()?;
    if reader.get_u64()? != rounds as u64 {
        return Err(Error::InvalidEncoding);
    }
    let mut round_values = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if count > 8 {
            return Err(Error::InvalidEncoding);
        }
        let mut coefficients = Vec::with_capacity(count);
        for _ in 0..count {
            coefficients.push(reader.get_fp3()?);
        }
        round_values.push(PscRound { coefficients });
    }
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if count > 8 {
        return Err(Error::InvalidEncoding);
    }
    let mut y_round = Vec::with_capacity(count);
    for _ in 0..count {
        y_round.push(reader.get_fp3()?);
    }
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if count != 2 {
        return Err(Error::InvalidEncoding);
    }
    let mut end_values = Vec::with_capacity(count);
    for _ in 0..count {
        end_values.push(reader.get_fp3()?);
    }
    Ok(GeneralProof {
        hiding_sum,
        rounds: round_values,
        y_round: PscRound {
            coefficients: y_round,
        },
        end_values,
    })
}

impl NormOutputSecond {
    pub fn encode(&self) -> Result<Vec<u8>> {
        if self.claims.len() != self.proofs.len() {
            return Err(Error::ShapeMismatch);
        }
        let mut writer = CanonicalWriter::new();
        writer.put_u64(NORM_OUTPUT_PSC_VERSION);
        writer.put_u64(self.claims.len() as u64);
        for (claim, proof) in self.claims.iter().zip(&self.proofs) {
            writer.put_fp3(claim);
            put_general_proof(&mut writer, proof);
        }
        Ok(writer.into_bytes())
    }

    pub fn decode(plan: &NormOutputPlan, positions: u32, bytes: &[u8]) -> Result<Self> {
        let variables = plan.broadcast.columns.trailing_zeros() as usize;
        let instances = usize::try_from(positions)
            .map_err(|_| Error::ResourceLimit)?
            .checked_mul(2)
            .ok_or(Error::ResourceLimit)?;
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        if reader.get_u64()? != NORM_OUTPUT_PSC_VERSION {
            return Err(Error::InvalidEncoding);
        }
        if reader.get_u64()? != instances as u64 {
            return Err(Error::InvalidEncoding);
        }
        let mut claims = Vec::with_capacity(instances);
        let mut proofs = Vec::with_capacity(instances);
        for _ in 0..instances {
            claims.push(reader.get_fp3()?);
            proofs.push(get_general_proof(&mut reader, variables)?);
        }
        reader.finish()?;
        let message = Self { claims, proofs };
        message.check(plan, positions)?;
        Ok(message)
    }

    pub fn check(&self, plan: &NormOutputPlan, positions: u32) -> Result<()> {
        let variables = plan.broadcast.columns.trailing_zeros() as usize;
        let instances = positions as usize * 2;
        if self.claims.len() != instances || self.proofs.len() != instances {
            return Err(Error::ShapeMismatch);
        }
        for proof in &self.proofs {
            if proof.rounds.len() != variables
                || proof.rounds.iter().any(|round| round.coefficients.len() >= 8)
                || proof.y_round.coefficients.len() != 2
                || proof.end_values.len() != 2
            {
                return Err(Error::ShapeMismatch);
            }
        }
        Ok(())
    }
}

/// The public second-stage transcript of the tail relations: one
/// product-sumcheck instance and its two shown point values per position and
/// relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormTailPscSecond {
    pub psc: Vec<DotProof>,
    pub point_values: Vec<[Fp3; 2]>,
}

impl NormTailPscSecond {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut writer = CanonicalWriter::new();
        put_psc_transcript(&mut writer, NORM_PSC_VERSION, &self.psc, &self.point_values)?;
        Ok(writer.into_bytes())
    }

    pub fn decode(plan: &NormPlan, bytes: &[u8]) -> Result<Self> {
        let tail_plan = plan
            .output
            .as_ref()
            .and_then(|output| output.tail.as_ref())
            .ok_or(Error::InvalidConfig)?;
        let instances = usize::try_from(plan.shape.positions)
            .map_err(|_| Error::ResourceLimit)?
            .checked_mul(NORM_OUTPUT_TAIL_RELATIONS)
            .ok_or(Error::ResourceLimit)?;
        let variables = output_tail_variables(plan.columns)?;
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        let (psc, point_values) = get_psc_transcript::<2>(
            &mut reader,
            NORM_PSC_VERSION,
            instances as u32,
            variables,
        )?;
        reader.finish()?;
        let message = Self { psc, point_values };
        message.check(plan, tail_plan)?;
        Ok(message)
    }

    pub fn check(&self, plan: &NormPlan, tail_plan: &NormOutputTailPlan) -> Result<()> {
        let instances = plan.shape.positions as usize * NORM_OUTPUT_TAIL_RELATIONS;
        let variables = output_tail_variables(plan.columns)?;
        let _ = tail_plan;
        if self.psc.len() != instances || self.point_values.len() != instances {
            return Err(Error::ShapeMismatch);
        }
        for proof in &self.psc {
            if proof.rounds.len() != variables || proof.y_round.coefficients.len() != 2 {
                return Err(Error::ShapeMismatch);
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
    /// The lookup transcript and claims, when the plan carries the lookup.
    /// The wire layout omits the field when the plan has no lookup.
    pub lookup: Option<NormLookupSecond>,
    /// The frozen-table lookup transcript and claims, when the plan carries it.
    pub rstd: Option<NormLookupSecond>,
    /// The pointwise product transcripts, when the plan carries the output
    /// tail.
    pub output: Option<NormOutputSecond>,
    /// The tail relation transcripts, when the plan carries the tail section.
    pub tail: Option<NormTailPscSecond>,
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
        if let Some(lookup) = &self.lookup {
            writer.put_bytes(&lookup.encode()?);
        }
        if let Some(rstd) = &self.rstd {
            writer.put_bytes(&rstd.encode()?);
        }
        if let Some(output) = &self.output {
            writer.put_bytes(&output.encode()?);
        }
        if let Some(tail) = &self.tail {
            writer.put_bytes(&tail.encode()?);
        }
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
        let message = Self {
            psc,
            point_values,
            lookup: None,
            rstd: None,
            output: None,
            tail: None,
        };
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
        let lookup = match &plan.lookup {
            Some(lookup_plan) => Some(NormLookupSecond::decode(
                lookup_plan,
                reader.get_bytes()?,
            )?),
            None => None,
        };
        let rstd = match &plan.rstd {
            Some(rstd_plan) => Some(NormLookupSecond::decode_for_rstd(
                rstd_plan,
                reader.get_bytes()?,
            )?),
            None => None,
        };
        let output = match &plan.output {
            Some(output_plan) => Some(NormOutputSecond::decode(
                output_plan,
                plan.shape.positions,
                reader.get_bytes()?,
            )?),
            None => None,
        };
        let tail = match plan.output.as_ref().and_then(|output| output.tail.as_ref()) {
            Some(_) => Some(NormTailPscSecond::decode(plan, reader.get_bytes()?)?),
            None => None,
        };
        reader.finish()?;
        let message = Self {
            psc,
            point_values,
            lookup,
            rstd,
            output,
            tail,
        };
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
        match (&plan.lookup, &self.lookup) {
            (Some(lookup_plan), Some(lookup)) => lookup.check(lookup_plan)?,
            (None, None) => {}
            _ => return Err(Error::ShapeMismatch),
        }
        match (&plan.rstd, &self.rstd) {
            (Some(rstd_plan), Some(rstd)) => rstd.check_rstd(rstd_plan)?,
            (None, None) => {}
            _ => return Err(Error::ShapeMismatch),
        }
        match (&plan.output, &self.output) {
            (Some(output_plan), Some(output)) => {
                output.check(output_plan, plan.shape.positions)?
            }
            (None, None) => {}
            _ => return Err(Error::ShapeMismatch),
        }
        let tail_plan = plan.output.as_ref().and_then(|output| output.tail.as_ref());
        match (tail_plan, &self.tail) {
            (Some(tail_plan), Some(tail)) => tail.check(plan, tail_plan)?,
            (None, None) => {}
            _ => return Err(Error::ShapeMismatch),
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
        if self.lookup.is_some() || self.rstd.is_some() {
            return Err(Error::ShapeMismatch);
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
    round: Option<&NormRoundPlan>,
    rsqrt: Option<&crate::proof::hidden_norm::RsqrtRow>,
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
        4 => {
            left[0] = fp3_signed(row.mean_offset)?;
            left[1] = fp3_signed(row.mean_complement)?;
            left[2] = fp3_signed(-(valid as i128 - 1))?;
            right[0] = Fp3::ONE;
            right[1] = Fp3::ONE;
            right[2] = Fp3::ONE;
        }
        5 => {
            left[0] = fp3_signed(row.variance_offset)?;
            left[1] = fp3_signed(row.variance_complement)?;
            left[2] = fp3_signed(-(valid as i128 - 1))?;
            right[0] = Fp3::ONE;
            right[1] = Fp3::ONE;
            right[2] = Fp3::ONE;
        }
        6 => {
            let low = -((valid / 2) as i128);
            left[0] = fp3_signed(row.mean_remainder)?;
            left[1] = fp3_signed(-row.mean_offset)?;
            left[2] = fp3_signed(-low)?;
            right[0] = Fp3::ONE;
            right[1] = Fp3::ONE;
            right[2] = Fp3::ONE;
        }
        7 => {
            let low = -((valid / 2) as i128);
            left[0] = fp3_signed(row.variance_remainder)?;
            left[1] = fp3_signed(-row.variance_offset)?;
            left[2] = fp3_signed(-low)?;
            right[0] = Fp3::ONE;
            right[1] = Fp3::ONE;
            right[2] = Fp3::ONE;
        }
        8..=15 => {
            let round = round.ok_or(Error::InvalidConfig)?;
            let layout = &round.layout;
            let (width, digits, value, offset) = match relation {
                8 => (
                    layout.widths.sum,
                    shifted_digits(row.sum, layout.sum_offset)?,
                    row.sum,
                    layout.sum_offset,
                ),
                9 => (
                    layout.widths.squares,
                    row.squares_digits.clone(),
                    row.squares,
                    0,
                ),
                10 => (
                    layout.widths.variance,
                    row.variance_digits.clone(),
                    row.variance as i128,
                    0,
                ),
                11 => (
                    layout.widths.mean,
                    shifted_digits(row.mean as i128, layout.mean_offset)?,
                    row.mean as i128,
                    layout.mean_offset,
                ),
                12 => (
                    layout.widths.offset,
                    row.mean_offset_digits.clone(),
                    row.mean_offset,
                    0,
                ),
                13 => (
                    layout.widths.offset,
                    row.mean_complement_digits.clone(),
                    row.mean_complement,
                    0,
                ),
                14 => (
                    layout.widths.offset,
                    row.variance_offset_digits.clone(),
                    row.variance_offset,
                    0,
                ),
                _ => (
                    layout.widths.offset,
                    row.variance_complement_digits.clone(),
                    row.variance_complement,
                    0,
                ),
            };
            if width >= columns || digits.len() > width {
                return Err(Error::ShapeMismatch);
            }
            let mut power: i128 = 1;
            for index in 0..width {
                let digit = digits.get(index).copied().unwrap_or(0) as i128;
                left[index] = fp3_signed(power * digit)?;
                right[index] = Fp3::ONE;
                power = power.checked_mul(256).ok_or(Error::ResourceLimit)?;
            }
            if offset != 0 {
                left[width] = fp3_signed(-(offset as i128))?;
                right[width] = Fp3::ONE;
            }
            left[columns] = fp3_signed(-value)?;
            right[columns] = Fp3::ONE;
        }
        16..=25 => {
            let round = round.ok_or(Error::InvalidConfig)?;
            let rsqrt = rsqrt.ok_or(Error::InvalidConfig)?;
            let signed = fp3_signed;
            match relation {
                16 => {
                    // variance + 2^17 - 2^18 * raw - raw_remainder = 0
                    left[0] = signed(rsqrt.variance as i128)?;
                    right[0] = Fp3::ONE;
                    left[1] = Fp3::ONE;
                    right[1] = signed(RSQRT_HALF as i128)?;
                    left[2] = signed(rsqrt.raw as i128)?;
                    right[2] = signed(-(RSQRT_SHIFT as i128))?;
                    left[3] = signed(rsqrt.raw_remainder as i128)?;
                    right[3] = signed(-1)?;
                }
                17 => {
                    // raw_remainder + raw_remainder_slack = 2^18 - 1
                    left[0] = signed(rsqrt.raw_remainder as i128)?;
                    right[0] = Fp3::ONE;
                    left[1] = signed(rsqrt.raw_remainder_slack as i128)?;
                    right[1] = Fp3::ONE;
                    left[2] = Fp3::ONE;
                    right[2] = signed(-(RSQRT_SHIFT as i128 - 1))?;
                }
                18 => {
                    // s * (1 - s) = 0 for every selector
                    for (index, selector) in
                        [rsqrt.fine, rsqrt.mid, rsqrt.coarse].iter().enumerate()
                    {
                        let value = signed(*selector as i128)?;
                        left[index] = value;
                        right[index] = Fp3::ONE.sub(&value);
                    }
                }
                19 => {
                    // fine + middle + coarse = 1
                    for (index, selector) in
                        [rsqrt.fine, rsqrt.mid, rsqrt.coarse].iter().enumerate()
                    {
                        left[index] = signed(*selector as i128)?;
                        right[index] = Fp3::ONE;
                    }
                    left[3] = Fp3::ONE;
                    right[3] = signed(-1)?;
                }
                20 => {
                    // fine * (FINE - 1) - fine * raw - fine_slack = 0
                    let fine = signed(rsqrt.fine as i128)?;
                    left[0] = fine;
                    right[0] = signed(RSQRT_FINE as i128 - 1)?;
                    left[1] = fine;
                    right[1] = signed(-(rsqrt.raw as i128))?;
                    left[2] = signed(rsqrt.fine_slack as i128)?;
                    right[2] = signed(-1)?;
                }
                21 => {
                    // middle * raw - middle * FINE - 256 * q - r = 0
                    let middle = signed(rsqrt.mid as i128)?;
                    left[0] = middle;
                    right[0] = signed(rsqrt.raw as i128)?;
                    left[1] = middle;
                    right[1] = signed(-(RSQRT_FINE as i128))?;
                    left[2] = signed(rsqrt.mid_q as i128)?;
                    right[2] = signed(-256)?;
                    left[3] = signed(rsqrt.mid_r as i128)?;
                    right[3] = signed(-1)?;
                }
                22 => {
                    // middle * (COARSE_END - 1) - middle * raw - slack = 0
                    let middle = signed(rsqrt.mid as i128)?;
                    left[0] = middle;
                    right[0] = signed(RSQRT_COARSE_END as i128 - 1)?;
                    left[1] = middle;
                    right[1] = signed(-(rsqrt.raw as i128))?;
                    left[2] = signed(rsqrt.mid_high_slack as i128)?;
                    right[2] = signed(-1)?;
                }
                23 => {
                    // coarse * raw - coarse * END - 65536 * q - 256 * hi - lo = 0
                    let coarse = signed(rsqrt.coarse as i128)?;
                    left[0] = coarse;
                    right[0] = signed(rsqrt.raw as i128)?;
                    left[1] = coarse;
                    right[1] = signed(-(RSQRT_COARSE_END as i128))?;
                    left[2] = signed(rsqrt.coarse_q as i128)?;
                    right[2] = signed(-65_536)?;
                    left[3] = Fp3::from_u64((rsqrt.coarse_r & 0xff) as u64);
                    right[3] = signed(-256)?;
                    left[4] = Fp3::from_u64(((rsqrt.coarse_r >> 8) & 0xffff) as u64);
                    right[4] = signed(-1)?;
                }
                24 => {
                    // index = fine*raw + middle*(FINE+q) + coarse*(END+q)
                    left[0] = signed(rsqrt.index as i128)?;
                    right[0] = Fp3::ONE;
                    left[1] = signed(rsqrt.fine as i128)?;
                    right[1] = signed(-(rsqrt.raw as i128))?;
                    left[2] = signed(rsqrt.mid as i128)?;
                    right[2] = signed(-(rsqrt.mid_q as i128))?;
                    left[3] = signed(rsqrt.coarse as i128)?;
                    right[3] = signed(-(rsqrt.coarse_q as i128))?;
                    left[4] = signed(rsqrt.mid as i128)?;
                    right[4] = signed(-(RSQRT_FINE as i128))?;
                    left[5] = signed(rsqrt.coarse as i128)?;
                    right[5] = signed(-((RSQRT_FINE + RSQRT_COARSE) as i128))?;
                }
                _ => {
                    // index + index_slack = table_real_len - 1
                    left[0] = signed(rsqrt.index as i128)?;
                    right[0] = Fp3::ONE;
                    left[1] = signed(rsqrt.index_slack as i128)?;
                    right[1] = Fp3::ONE;
                    left[2] = Fp3::ONE;
                    right[2] = signed(-((round.table_real_len - 1) as i128))?;
                }
            }
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
                None,
                None,
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
    Ok(NormPscSecond {
        psc,
        point_values,
        lookup: None,
        rstd: None,
        output: None,
        tail: None,
    })
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
pub const NORM_BIND_VERSION: u64 = 14;

/// The relation indices the bound core supports. The map construction of
/// every relation is fixed by the relation index; the plan may list only
/// relations whose row binding is implemented.
pub const NORM_BOUND_RELATIONS: [usize; 26] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25,
];

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
    /// The real (unpadded) length of the frozen rsqrt table.
    pub table_real_len: u64,
}

/// The lookup plan carried by the norm relation. The key vector is read from
/// the committed rounding rows according to `key`; only the multiplicity-bit
/// and mask messages are committed for the lookup itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormLookupPlan {
    pub lookup: LookupPlan,
    pub commit: LookupCommitPlan,
    pub key: crate::proof::lookup::LookupKeyPlan,
    /// The committed groups the key sources address. A key map that covers the
    /// rounding digits and the tail digits has two groups; the frozen-table
    /// view keeps none and falls back to the rounding group at the row
    /// registration.
    pub key_groups: Vec<LookupKeyGroup>,
}

/// One committed message group a lookup key map addresses: its first oracle,
/// its message shape, its column count and its position count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupKeyGroup {
    pub oracle_base: u64,
    pub shape: WShape,
    pub columns: u64,
    pub positions: u32,
}

impl LookupKeyGroup {
    pub fn covers(&self, oracle: u64) -> bool {
        oracle >= self.oracle_base
            && oracle < self.oracle_base.saturating_add(self.positions as u64)
    }

    fn fault(&self) -> Option<&'static str> {
        if self.oracle_base == 0
            || self.positions == 0
            || self.columns == 0
            || self.shape.fault(self.columns, self.positions).is_some()
        {
            return Some("a lookup key group is not a message group");
        }
        None
    }
}

impl NormLookupPlan {
    pub fn validate(&self) -> Result<()> {
        self.lookup.validate()?;
        self.key.validate(&self.lookup)?;
        if self.key_groups.is_empty() {
            if !self.key.sources.is_empty() {
                return Err(Error::InvalidConfig);
            }
        } else if self.key_groups.iter().any(|group| group.fault().is_some()) {
            return Err(Error::InvalidConfig);
        }
        for source in &self.key.sources {
            if self
                .key_groups
                .iter()
                .filter(|group| group.covers(source.oracle))
                .count()
                != 1
            {
                return Err(Error::InvalidConfig);
            }
        }
        let planes = self.lookup.planes()?;
        let rows = self.lookup.table_rows()?;
        self.commit.validate(
            planes.checked_mul(rows).ok_or(Error::ResourceLimit)?,
            crate::proof::lookup::mask_count(&self.lookup)?,
        )?;
        Ok(())
    }

    /// Binds the lookup plan, the key map and the committed message plans.
    pub fn digest(&self) -> Digest {
        let mut writer = CanonicalWriter::new();
        writer.put_bytes(&self.lookup.digest());
        writer.put_u64(self.key_groups.len() as u64);
        for group in &self.key_groups {
            writer.put_u64(group.oracle_base);
            writer.put_u64(group.columns);
            writer.put_u64(group.positions as u64);
            put_wshape(&mut writer, &group.shape);
        }
        writer.put_u64(self.key.sources.len() as u64);
        for source in &self.key.sources {
            writer.put_u64(source.column as u64);
            writer.put_u64(source.oracle);
            writer.put_u64(source.position as u64);
            writer.put_u64(source.row as u64);
            writer.put_u64(source.place as u64);
            writer.put_fp3(&source.coefficient);
            writer.put_fp3(&source.offset);
        }
        writer.put_u64(self.key.padding_values.len() as u64);
        for padding in &self.key.padding_values {
            writer.put_fp3(padding);
        }
        writer.put_digest(&self.key.table_identity);
        writer.put_u64(self.commit.bits_oracle);
        writer.put_u64(self.commit.bits_openings as u64);
        writer.put_u64(self.commit.group);
        writer.put_u64(self.commit.mask_oracle);
        writer.put_u64(self.commit.mask_openings as u64);
        put_wshape(&mut writer, &self.commit.bits_shape);
        writer.put_u64(self.commit.mask_code.log_n as u64);
        writer.put_u64(self.commit.mask_code.ell);
        writer.put_u64(self.commit.mask_code.r);
        writer.put_u64(self.commit.mask_rows);
        put_tree_plan(&mut writer, &self.commit.mask_tree);
        digest_bytes(&writer.into_bytes())
    }
}

/// The trusted plan of the frozen-table lookup. The key vector is the pair
/// (committed index, committed table value); the table view fixes the trusted
/// identity, content root, real length and encoding, and the key plan fixes
/// the two committed round slots and the public first-row padding pair.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormRstdPlan {
    pub lookup: LookupPlan,
    pub commit: LookupCommitPlan,
    pub key: crate::proof::lookup::LookupKeyPlan,
    pub table: TableConfig,
}

impl NormRstdPlan {
    pub fn validate(&self) -> Result<()> {
        self.lookup.validate()?;
        if self.lookup.columns != 2 {
            return Err(Error::InvalidConfig);
        }
        self.key.validate(&self.lookup)?;
        let planes = self.lookup.planes()?;
        let rows = self.lookup.table_rows()?;
        self.commit.validate(
            planes.checked_mul(rows).ok_or(Error::ResourceLimit)?,
            crate::proof::lookup::mask_count(&self.lookup)?,
        )?;
        if self.table.version != TABLE_ENCODING_VERSION
            || self.table.identity == Digest::ZERO
            || self.table.content_root == Digest::ZERO
            || self.table.real_len == 0
            || self.table.real_len > rows
            || self.key.table_identity != self.table.identity
            || self.key.padding_values.len() != 2
        {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }

    /// Binds the lookup plan, the key map, the committed message plans, the
    /// trusted table configuration and the first-row padding pair.
    pub fn digest(&self) -> Digest {
        let mut writer = CanonicalWriter::new();
        writer.put_bytes(&self.lookup.digest());
        writer.put_u64(self.key.sources.len() as u64);
        for source in &self.key.sources {
            writer.put_u64(source.column as u64);
            writer.put_u64(source.oracle);
            writer.put_u64(source.position as u64);
            writer.put_u64(source.row as u64);
            writer.put_u64(source.place as u64);
            writer.put_fp3(&source.coefficient);
            writer.put_fp3(&source.offset);
        }
        writer.put_u64(self.key.padding_values.len() as u64);
        for padding in &self.key.padding_values {
            writer.put_fp3(padding);
        }
        writer.put_digest(&self.key.table_identity);
        writer.put_u64(self.commit.bits_oracle);
        writer.put_u64(self.commit.bits_openings as u64);
        writer.put_u64(self.commit.group);
        writer.put_u64(self.commit.mask_oracle);
        writer.put_u64(self.commit.mask_openings as u64);
        put_wshape(&mut writer, &self.commit.bits_shape);
        writer.put_u64(self.commit.mask_code.log_n as u64);
        writer.put_u64(self.commit.mask_code.ell);
        writer.put_u64(self.commit.mask_code.r);
        writer.put_u64(self.commit.mask_rows);
        put_tree_plan(&mut writer, &self.commit.mask_tree);
        writer.put_digest(&self.table.identity);
        writer.put_digest(&self.table.content_root);
        writer.put_u64(self.table.real_len as u64);
        writer.put_u64(self.table.version);
        writer.put_u64(match self.table.encoding {
            crate::proof::lookup_table::TableEncoding::I32Le => 4,
            crate::proof::lookup_table::TableEncoding::I64Le => 8,
        });
        digest_bytes(&writer.into_bytes())
    }

    /// Sizes and builds the frozen-table lookup from the round value slots.
    /// Column zero reads the committed index value slot, column one the
    /// committed table value slot; the padding pair is the first table row.
    pub fn from_round(
        round: &NormRoundPlan,
        positions: u32,
        table: TableConfig,
        first_value: Fp3,
        params: &NormRstdParams,
    ) -> Result<Self> {
        if positions == 0
            || table.real_len == 0
            || table.version != TABLE_ENCODING_VERSION
            || table.identity == Digest::ZERO
            || table.content_root == Digest::ZERO
            || params.bits_oracle == 0
            || params.mask_oracle == 0
            || params.bits_oracle == params.mask_oracle
            || params.group == 0
            || params.bits_tree == params.mask_tree
            || params.code.r < 1
            || params.mask_code.r < 1
        {
            return Err(Error::InvalidConfig);
        }
        let table_log = ceil_log2_u64(table.real_len as u64);
        let (key_bits, mu) = lookup_parameters(positions as usize)?;
        let lookup = LookupPlan {
            key_bits,
            mu,
            table_log,
            columns: 2,
        };
        lookup.validate()?;
        let rows = lookup.table_rows()?;
        if table.real_len > rows {
            return Err(Error::InvalidConfig);
        }
        let round_rows = usize::try_from(round.group.shape.main_rows)
            .map_err(|_| Error::ResourceLimit)?;
        let ell = usize::try_from(round.group.shape.code.ell)
            .map_err(|_| Error::ResourceLimit)?;
        if round_rows == 0 || ell == 0 {
            return Err(Error::InvalidConfig);
        }
        let mut sources = Vec::with_capacity(positions as usize * 2);
        for position in 0..positions {
            let oracle = round
                .group
                .oracle_base
                .checked_add(position as u64)
                .ok_or(Error::ResourceLimit)?;
            for (column, slot) in [
                (0u8, round.layout.slots.index_value),
                (1u8, round.layout.slots.rstd),
            ] {
                let row = slot % round_rows;
                let place = slot / round_rows;
                if place >= ell {
                    return Err(Error::ShapeMismatch);
                }
                sources.push(LookupKeySource {
                    column,
                    oracle,
                    position,
                    row: row as u32,
                    place: place as u32,
                    coefficient: Fp3::ONE,
                    offset: Fp3::ZERO,
                });
            }
        }
        let key = LookupKeyPlan {
            sources,
            padding_values: vec![Fp3::ZERO, first_value],
            table_identity: table.identity,
        };
        let bit_count = lookup
            .planes()?
            .checked_mul(rows)
            .ok_or(Error::ResourceLimit)?;
        let bits_shape = WShape::new(
            bit_count as u64,
            params.code,
            1,
            1,
            params.salt_bytes,
            params.bits_tree,
            params.leaf_class,
            params.run,
            params.configuration,
        )?;
        if bits_shape.main_rows > (1u64 << params.code.log_n) {
            return Err(Error::InvalidConfig);
        }
        let mask_rows = lookup_mask_rows(&lookup, params.mask_code)?;
        let mask_tree = RuntimeTreePlan {
            tree_number: params.mask_tree,
            leaf_class: params.leaf_class,
            log_n: params.mask_code.log_n,
            log_n_zk: params.mask_code.log_n,
            parts: vec![RuntimeTreePart {
                main_symbols: 0,
                mask_symbols: mask_rows
                    .checked_mul(zkie_engine::pcs::request::field::E_DEGREE as u64)
                    .ok_or(Error::ResourceLimit)?,
            }],
            salt_bytes: params.salt_bytes,
            run: params.run,
            configuration: params.configuration,
        };
        let commit = LookupCommitPlan {
            bits_shape,
            bits_oracle: params.bits_oracle,
            group: params.group,
            bits_openings: 1,
            mask_oracle: params.mask_oracle,
            mask_code: params.mask_code,
            mask_rows,
            mask_openings: 1,
            mask_tree,
        };
        let plan = Self {
            lookup,
            commit,
            key,
            table,
        };
        plan.validate()?;
        Ok(plan)
    }
}

/// The plan fields the row registration reads from either lookup plan.
pub trait NormLookupView {
    fn lookup_plan(&self) -> &LookupPlan;
    fn commit_plan(&self) -> &LookupCommitPlan;
    fn key_plan(&self) -> &LookupKeyPlan;
    /// The committed groups the key sources address, when the plan fixes
    /// them. The frozen-table view keeps none and falls back to the rounding
    /// group at the row registration.
    fn key_groups(&self) -> &[LookupKeyGroup];
}

impl NormLookupView for NormLookupPlan {
    fn lookup_plan(&self) -> &LookupPlan {
        &self.lookup
    }

    fn commit_plan(&self) -> &LookupCommitPlan {
        &self.commit
    }

    fn key_plan(&self) -> &LookupKeyPlan {
        &self.key
    }

    fn key_groups(&self) -> &[LookupKeyGroup] {
        &self.key_groups
    }
}

impl NormLookupView for NormRstdPlan {
    fn lookup_plan(&self) -> &LookupPlan {
        &self.lookup
    }

    fn commit_plan(&self) -> &LookupCommitPlan {
        &self.commit
    }

    fn key_plan(&self) -> &LookupKeyPlan {
        &self.key
    }

    fn key_groups(&self) -> &[LookupKeyGroup] {
        &[]
    }
}

/// The parent-plan values of the frozen-table lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormRstdParams {
    pub code: zkie_engine::pcs::request::code::CodeParams,
    pub mask_code: zkie_engine::pcs::request::code::CodeParams,
    pub group: u64,
    pub bits_oracle: u64,
    pub mask_oracle: u64,
    pub bits_tree: u64,
    pub mask_tree: u64,
    pub leaf_class: u32,
    pub salt_bytes: usize,
    pub run: Digest,
    pub configuration: Digest,
}

/// The smallest key-bit count and multiplicity-plane count of a byte lookup
/// over `slots` key entries: `2^key_bits >= slots` and `2^(2^mu) > 2^key_bits`.
pub fn lookup_parameters(slots: usize) -> Result<(u32, u32)> {
    if slots == 0 || slots > (1usize << 30) {
        return Err(Error::InvalidConfig);
    }
    let key_bits = ceil_log2_u64(slots as u64);
    let keys = 1u64.checked_shl(key_bits).ok_or(Error::ResourceLimit)?;
    let mut mu = 1u32;
    loop {
        let planes = 1u64.checked_shl(mu).ok_or(Error::ResourceLimit)?;
        if planes >= 63 || keys < (1u64 << planes) {
            break;
        }
        mu = mu.checked_add(1).ok_or(Error::ResourceLimit)?;
        if mu >= 63 {
            return Err(Error::ResourceLimit);
        }
    }
    Ok((key_bits, mu))
}

/// The trusted key map of the byte lookup over every committed digit slot of
/// the rounding rows. Key `position * digits + digit` reads that digit cell of
/// the position's rounding message; the remaining keys are the public padding
/// byte. The table identity is a plan value.
pub fn round_digit_lookup_key(
    round: &NormRoundPlan,
    positions: u32,
    key_bits: u32,
    padding_byte: u64,
    table_identity: Digest,
) -> Result<LookupKeyPlan> {
    if positions == 0 || padding_byte > 255 || table_identity == Digest::ZERO {
        return Err(Error::InvalidConfig);
    }
    let keys = 1u64.checked_shl(key_bits).ok_or(Error::ResourceLimit)?;
    let digits = round.layout.digit_slots()?;
    let count = digits
        .len()
        .checked_mul(positions as usize)
        .ok_or(Error::ResourceLimit)?;
    if count == 0 || count as u64 > keys {
        return Err(Error::InvalidConfig);
    }
    let rows = round.group.shape.main_rows;
    let ell = round.group.shape.code.ell;
    if rows == 0 || ell == 0 {
        return Err(Error::InvalidConfig);
    }
    let mut sources = Vec::with_capacity(count);
    let mut seen = std::collections::HashSet::new();
    for position in 0..positions {
        let oracle = round
            .group
            .oracle_base
            .checked_add(position as u64)
            .ok_or(Error::ResourceLimit)?;
        for slot in &digits {
            let slot = *slot as u64;
            if slot >= round.group.columns {
                return Err(Error::ShapeMismatch);
            }
            let row = slot % rows;
            let place = slot / rows;
            if place >= ell {
                return Err(Error::ShapeMismatch);
            }
            if !seen.insert((oracle, position, row as u32, place as u32)) {
                return Err(Error::InvalidConfig);
            }
            sources.push(LookupKeySource {
                column: 0,
                oracle,
                position,
                row: row as u32,
                place: place as u32,
                coefficient: Fp3::ONE,
                offset: Fp3::ZERO,
            });
        }
    }
    Ok(LookupKeyPlan {
        sources,
        padding_values: vec![Fp3::from_u64(padding_byte)],
        table_identity,
    })
}

/// The parent-plan values a lookup plan inherits: the main and mask codes of
/// the own proximity group, the message and tree identifiers, the tree
/// material, and the public padding and table identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormLookupParams {
    pub code: zkie_engine::pcs::request::code::CodeParams,
    pub mask_code: zkie_engine::pcs::request::code::CodeParams,
    pub group: u64,
    pub bits_oracle: u64,
    pub mask_oracle: u64,
    pub bits_tree: u64,
    pub mask_tree: u64,
    pub leaf_class: u32,
    pub salt_bytes: usize,
    pub run: Digest,
    pub configuration: Digest,
    pub padding_byte: u64,
    pub table_identity: Digest,
}

impl NormLookupPlan {
    fn check_params(params: &NormLookupParams) -> Result<()> {
        if params.bits_oracle == 0
            || params.mask_oracle == 0
            || params.bits_oracle == params.mask_oracle
            || params.group == 0
            || params.bits_tree == params.mask_tree
        {
            return Err(Error::InvalidConfig);
        }
        if params.code.r < 1 || params.mask_code.r < 1 {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }

    fn from_parts(
        lookup: LookupPlan,
        key: crate::proof::lookup::LookupKeyPlan,
        key_groups: Vec<LookupKeyGroup>,
        params: &NormLookupParams,
    ) -> Result<Self> {
        Self::check_params(params)?;
        lookup.validate()?;
        let bit_count = lookup
            .planes()?
            .checked_mul(lookup.table_rows()?)
            .ok_or(Error::ResourceLimit)?;
        let bits_shape = WShape::new(
            bit_count as u64,
            params.code,
            1,
            1,
            params.salt_bytes,
            params.bits_tree,
            params.leaf_class,
            params.run,
            params.configuration,
        )?;
        if bits_shape.main_rows > (1u64 << params.code.log_n) {
            return Err(Error::InvalidConfig);
        }
        let mask_rows = lookup_mask_rows(&lookup, params.mask_code)?;
        let mask_symbols = mask_rows
            .checked_mul(zkie_engine::pcs::request::field::E_DEGREE as u64)
            .ok_or(Error::ResourceLimit)?;
        let mask_tree = RuntimeTreePlan {
            tree_number: params.mask_tree,
            leaf_class: params.leaf_class,
            log_n: params.mask_code.log_n,
            log_n_zk: params.mask_code.log_n,
            parts: vec![RuntimeTreePart {
                main_symbols: 0,
                mask_symbols,
            }],
            salt_bytes: params.salt_bytes,
            run: params.run,
            configuration: params.configuration,
        };
        let plan = Self {
            lookup,
            commit: LookupCommitPlan {
                bits_shape,
                bits_oracle: params.bits_oracle,
                group: params.group,
                bits_openings: 1,
                mask_oracle: params.mask_oracle,
                mask_code: params.mask_code,
                mask_rows,
                mask_openings: 1,
                mask_tree,
            },
            key,
            key_groups,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// Sizes and builds the lookup plan from the rounding layout and the
    /// parent plan values. The key map covers every committed digit slot; the
    /// multiplicity-bit shape and mask rows follow from the derived key and
    /// plane counts, the parent codes and the mask budget.
    pub fn from_round(
        round: &NormRoundPlan,
        positions: u32,
        params: &NormLookupParams,
    ) -> Result<Self> {
        Self::check_params(params)?;
        let digits = round.layout.digit_slots()?;
        let slots = digits
            .len()
            .checked_mul(positions as usize)
            .ok_or(Error::ResourceLimit)?;
        let (key_bits, mu) = lookup_parameters(slots)?;
        let lookup = LookupPlan {
            key_bits,
            mu,
            table_log: 8,
            columns: 1,
        };
        let key = round_digit_lookup_key(
            round,
            positions,
            key_bits,
            params.padding_byte,
            params.table_identity,
        )?;
        let group = LookupKeyGroup {
            oracle_base: round.group.oracle_base,
            shape: round.group.shape,
            columns: round.group.columns,
            positions,
        };
        Self::from_parts(lookup, key, vec![group], params)
    }

    /// Sizes and builds one byte-table lookup that covers the rounding digits
    /// and every tail digit cell. The key map spans two committed groups: the
    /// rounding group and the output tail group. Both groups are bound in the
    /// plan digest, so the integrated chain proves every digit of both
    /// messages is a byte.
    pub fn from_round_and_tail(
        round: &NormRoundPlan,
        tail: &NormOutputTailPlan,
        columns: u64,
        positions: u32,
        params: &NormLookupParams,
    ) -> Result<Self> {
        Self::check_params(params)?;
        if params.table_identity != crate::proof::lookup::BYTE_TABLE_IDENTITY
            || params.padding_byte > 255
            || positions == 0
        {
            return Err(Error::InvalidConfig);
        }
        if tail.group.columns != tail.columns(columns)?
            || tail.group.fault(positions).is_some()
        {
            return Err(Error::InvalidConfig);
        }
        let round_slots = round
            .layout
            .digit_slots()?
            .len()
            .checked_mul(positions as usize)
            .ok_or(Error::ResourceLimit)?;
        let tail_slots = tail_digit_count(tail, positions, columns)?;
        let total = round_slots
            .checked_add(tail_slots)
            .ok_or(Error::ResourceLimit)?;
        let (key_bits, mu) = lookup_parameters(total)?;
        let lookup = LookupPlan {
            key_bits,
            mu,
            table_log: 8,
            columns: 1,
        };
        let mut key = round_digit_lookup_key(
            round,
            positions,
            key_bits,
            params.padding_byte,
            params.table_identity,
        )?;
        let tail_key = tail_digit_lookup_key(
            tail,
            positions,
            columns,
            key_bits,
            params.padding_byte,
            params.table_identity,
        )?;
        key.sources.extend(tail_key.sources);
        let groups = vec![
            LookupKeyGroup {
                oracle_base: round.group.oracle_base,
                shape: round.group.shape,
                columns: round.group.columns,
                positions,
            },
            LookupKeyGroup {
                oracle_base: tail.group.oracle_base,
                shape: tail.group.shape,
                columns: tail.group.columns,
                positions,
            },
        ];
        Self::from_parts(lookup, key, groups, params)
    }
}

/// The digit cells of one tail column: quotient, remainder, doubled remainder,
/// tie slack, shifted value, pre-clamp value, the two clamp distance rows and
/// the four clamp slack rows (strict and non-strict per side). The selectors
/// are excluded because the booleanity and sign-consistency relations force
/// them to zero or one once every slack word is a byte.
pub fn tail_digit_groups(tail: &NormOutputTailPlan) -> Result<Vec<(usize, usize)>> {
    let clamp = tail.clamp.as_ref().ok_or(Error::InvalidConfig)?;
    let widths = &tail.widths;
    let q_at = 0usize;
    let r_at = widths.quotient;
    let u_at = r_at + widths.remainder;
    let w_at = u_at + widths.doubled_remainder;
    let sh_at = w_at + widths.tie_slack + 1;
    let pre_at = sh_at + widths.shifted;
    let low_at = pre_at + widths.pre_clamp;
    let high_at = low_at + clamp.low;
    let low_selector = high_at + clamp.high;
    let low_strict_at = low_selector + 2;
    let low_nonstrict_at = low_strict_at + clamp.low_strict;
    let high_strict_at = low_nonstrict_at + clamp.low_nonstrict;
    let high_nonstrict_at = high_strict_at + clamp.high_strict;
    Ok(vec![
        (q_at, widths.quotient),
        (r_at, widths.remainder),
        (u_at, widths.doubled_remainder),
        (w_at, widths.tie_slack),
        (sh_at, widths.shifted),
        (pre_at, widths.pre_clamp),
        (low_at, clamp.low),
        (high_at, clamp.high),
        (low_strict_at, clamp.low_strict),
        (low_nonstrict_at, clamp.low_nonstrict),
        (high_strict_at, clamp.high_strict),
        (high_nonstrict_at, clamp.high_nonstrict),
    ])
}

/// The number of digit key cells of the tail group and column domain.
pub fn tail_digit_count(
    tail: &NormOutputTailPlan,
    positions: u32,
    columns: u64,
) -> Result<usize> {
    let groups = tail_digit_groups(tail)?;
    let digits = groups
        .iter()
        .try_fold(0usize, |sum, (_, width)| sum.checked_add(*width))
        .ok_or(Error::ResourceLimit)?;
    usize::try_from(columns)
        .map_err(|_| Error::ResourceLimit)?
        .checked_mul(digits)
        .and_then(|value| value.checked_mul(positions as usize))
        .ok_or(Error::ResourceLimit)
}

/// The trusted key map of the byte lookup over every tail digit cell. The keys
/// are ordered position-major, then column-major, then by the digit groups.
pub fn tail_digit_lookup_key(
    tail: &NormOutputTailPlan,
    positions: u32,
    columns: u64,
    key_bits: u32,
    padding_byte: u64,
    table_identity: Digest,
) -> Result<crate::proof::lookup::LookupKeyPlan> {
    use crate::proof::lookup::LookupKeySource;

    if positions == 0 || padding_byte > 255 || table_identity == Digest::ZERO {
        return Err(Error::InvalidConfig);
    }
    let keys = 1u64.checked_shl(key_bits).ok_or(Error::ResourceLimit)?;
    let count = tail_digit_count(tail, positions, columns)?;
    if count == 0 || count as u64 > keys {
        return Err(Error::InvalidConfig);
    }
    let groups = tail_digit_groups(tail)?;
    let slots = tail.slots()?;
    let rows = usize::try_from(tail.group.shape.main_rows)
        .map_err(|_| Error::ResourceLimit)?;
    let ell = usize::try_from(tail.group.shape.code.ell)
        .map_err(|_| Error::ResourceLimit)?;
    let columns = usize::try_from(columns).map_err(|_| Error::ResourceLimit)?;
    if rows == 0 || ell == 0 {
        return Err(Error::InvalidConfig);
    }
    let mut sources = Vec::with_capacity(count);
    for position in 0..positions {
        let oracle = tail
            .group
            .oracle_base
            .checked_add(position as u64)
            .ok_or(Error::ResourceLimit)?;
        for column in 0..columns {
            let base = column.checked_mul(slots).ok_or(Error::ResourceLimit)?;
            for (offset, width) in &groups {
                for digit in 0..*width {
                    let slot = base
                        .checked_add(*offset)
                        .and_then(|value| value.checked_add(digit))
                        .ok_or(Error::ResourceLimit)?;
                    let row = slot % rows;
                    let place = slot / rows;
                    if place >= ell {
                        return Err(Error::ShapeMismatch);
                    }
                    sources.push(LookupKeySource {
                        column: 0,
                        oracle,
                        position,
                        row: row as u32,
                        place: place as u32,
                        coefficient: Fp3::ONE,
                        offset: Fp3::ZERO,
                    });
                }
            }
        }
    }
    Ok(crate::proof::lookup::LookupKeyPlan {
        sources,
        padding_values: vec![Fp3::from_u64(padding_byte)],
        table_identity,
    })
}

/// The smallest power-of-two mask row count whose free entries hold every
/// allocated lookup mask entry.
pub fn lookup_mask_rows(
    plan: &LookupPlan,
    mask_code: zkie_engine::pcs::request::code::CodeParams,
) -> Result<u64> {
    let count = crate::proof::lookup::mask_count(plan)?;
    let ell = mask_code.ell;
    if ell < 2 {
        return Err(Error::InvalidConfig);
    }
    let columns = 1u64
        .checked_shl(mask_code.log_n)
        .ok_or(Error::ResourceLimit)?;
    let mut rows = 1u64;
    while rows <= columns {
        let free = rows
            .checked_mul(ell - 1)
            .and_then(|value| value.checked_add(rows - 1))
            .ok_or(Error::ResourceLimit)?;
        if free >= count as u64 {
            return Ok(rows);
        }
        rows = rows.checked_mul(2).ok_or(Error::ResourceLimit)?;
    }
    Err(Error::ResourceLimit)
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
    /// The lookup plan over the committed rounding digit rows; optional in
    /// this stage.
    pub lookup: Option<NormLookupPlan>,
    /// The frozen-table lookup over the committed index and table-value slots;
    /// optional in this stage.
    pub rstd: Option<NormRstdPlan>,
    /// The norm gain/bias parameter tables; optional in this stage.
    pub params: Option<NormParamTablesPlan>,
    /// The output tail groups (the broadcast rstd row, the rstd-times-gain
    /// product, the centered-times-product image, and the output/hd rows);
    /// optional in this stage.
    pub output: Option<NormOutputPlan>,
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
        if self.relations.iter().any(|relation| *relation >= 4) && self.round.is_none() {
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
            if round.input_bound == 0
                || round.table_real_len == 0
                || round.table_real_len > (1u64 << 32)
            {
                return Err(Error::InvalidConfig);
            }
            let columns = round.layout.widths.columns()?;
            if round.group.columns != columns
                || round.group.fault(self.shape.positions).is_some()
                || round.group.shape.answers != 1
            {
                return Err(Error::InvalidConfig);
            }
            if norm_no_wrap(
                round.input_bound,
                self.shape.valid,
                &round.layout.widths,
                round.layout.sum_offset,
                round.layout.mean_offset,
            )
            .is_err()
            {
                return Err(Error::InvalidConfig);
            }
            for group in [&self.x, &self.sum, &self.squares, &self.mean, &self.variance] {
                if round.group.oracle_base == group.oracle_base {
                    return Err(Error::InvalidConfig);
                }
            }
        }
        if let Some(lookup) = &self.lookup {
            lookup.validate()?;
            let plan_group = |group: &NormGroupPlan| LookupKeyGroup {
                oracle_base: group.oracle_base,
                shape: group.shape,
                columns: group.columns,
                positions: self.shape.positions,
            };
            let round_group = self.round.as_ref().map(|round| plan_group(&round.group));
            let tail_plan = self
                .output
                .as_ref()
                .and_then(|output| output.tail.as_ref());
            let tail_group = tail_plan.map(|tail| plan_group(&tail.group));
            for group in &lookup.key_groups {
                let known = round_group
                    .as_ref()
                    .map(|plan| plan == group)
                    .unwrap_or(false)
                    || tail_group
                        .as_ref()
                        .map(|plan| plan == group)
                        .unwrap_or(false);
                if !known {
                    return Err(Error::InvalidConfig);
                }
            }
            // The integrated chain must prove the byte range of every digit
            // of both committed message groups.
            if !lookup.key.sources.is_empty() {
                if round_group
                    .as_ref()
                    .map(|group| !lookup.key_groups.contains(group))
                    .unwrap_or(false)
                {
                    return Err(Error::InvalidConfig);
                }
                if tail_group
                    .as_ref()
                    .map(|group| !lookup.key_groups.contains(group))
                    .unwrap_or(false)
                {
                    return Err(Error::InvalidConfig);
                }
            }
            // The key map is canonical: it addresses every digit cell of the
            // committed rounding message and, when the tail exists, every
            // digit cell of the tail message. A lookup that drops or
            // redirects a cell would leave that cell without a byte-range
            // proof, so the plan is rejected.
            if let Some(round) = &self.round {
                let key_bits = lookup.lookup.key_bits;
                let expected = round_digit_lookup_key(
                    round,
                    self.shape.positions,
                    key_bits,
                    0,
                    lookup.key.table_identity,
                )?;
                let round_sources = expected.sources.len();
                if lookup.key.sources.len() < round_sources
                    || lookup.key.sources[..round_sources] != expected.sources[..]
                {
                    return Err(Error::InvalidConfig);
                }
                match tail_plan {
                    Some(tail) => {
                        let expected_tail = tail_digit_lookup_key(
                            tail,
                            self.shape.positions,
                            self.columns,
                            key_bits,
                            0,
                            lookup.key.table_identity,
                        )?;
                        if lookup.key.sources[round_sources..] != expected_tail.sources[..] {
                            return Err(Error::InvalidConfig);
                        }
                    }
                    None => {
                        if lookup.key.sources.len() != round_sources {
                            return Err(Error::InvalidConfig);
                        }
                    }
                }
            }
        }
        if let Some(rstd) = &self.rstd {
            rstd.validate()?;
            if let Some(round) = &self.round {
                if rstd.table.real_len as u64 != round.table_real_len {
                    return Err(Error::InvalidConfig);
                }
            }
        }
        if let Some(params) = &self.params {
            params.validate()?;
            if params.gain.width != self.columns || params.bias.width != self.columns {
                return Err(Error::InvalidConfig);
            }
        }
        if let Some(output) = &self.output {
            if self.params.is_none() {
                return Err(Error::InvalidConfig);
            }
            output.validate(self)?;
            if output.tail.is_some() {
                self.tail_no_wrap()?;
            }
        }
        if let Some(mask) = &self.mask {
            mask.validate(
                self.shape.positions,
                self.variables,
                self.relations.len(),
            )?;
            let expected = if self.output.is_some() { 2 } else { 0 };
            if mask.pointwise != expected {
                return Err(Error::InvalidConfig);
            }
            let tail_expected = if self
                .output
                .as_ref()
                .and_then(|output| output.tail.as_ref())
                .is_some()
            {
                NORM_OUTPUT_TAIL_RELATIONS
            } else {
                0
            };
            if mask.tail != tail_expected {
                return Err(Error::InvalidConfig);
            }
        }
        Ok(())
    }

    /// The plan-level no-wrap derivation of the output tail. Every integer
    /// magnitude is bounded from the trusted widths, the shift, the input
    /// bound and the published parameter ranges; each reconstruction and
    /// relation term must stay below half the base modulus. The witness-level
    /// bound is not a substitute for this plan check.
    fn tail_no_wrap(&self) -> Result<()> {
        let output = self.output.as_ref().ok_or(Error::InvalidConfig)?;
        let tail = output.tail.as_ref().ok_or(Error::InvalidConfig)?;
        let round = self.round.as_ref().ok_or(Error::InvalidConfig)?;
        let params = self.params.as_ref().ok_or(Error::InvalidConfig)?;
        let widths = &tail.widths;
        let half = (MODULUS as u128) / 2;
        let capacity = |width: usize| -> Result<u128> {
            if width == 0 || width > 7 {
                return Err(Error::InvalidConfig);
            }
            1u128
                .checked_shl(8 * width as u32)
                .ok_or(Error::ResourceLimit)
        };
        let half_range = |width: usize| -> Result<u128> {
            capacity(width)?
                .checked_div(2)
                .ok_or(Error::ResourceLimit)
        };
        if output.shift >= 63 {
            return Err(Error::InvalidConfig);
        }
        let divisor = 1u128 << output.shift;
        let q_cap = capacity(widths.quotient)?;
        let q_half = half_range(widths.quotient)?;
        let r_cap = capacity(widths.remainder)?;
        let u_cap = capacity(widths.doubled_remainder)?;
        let w_cap = capacity(widths.tie_slack)?;
        let sh_cap = capacity(widths.shifted)?;
        let pre_cap = capacity(widths.pre_clamp)?;
        let pre_half = half_range(widths.pre_clamp)?;
        let round_bounds = norm_no_wrap(
            round.input_bound,
            self.shape.valid,
            &round.layout.widths,
            round.layout.sum_offset,
            round.layout.mean_offset,
        )?;
        let gain = params
            .gain
            .valid_max
            .unsigned_abs()
            .max(params.gain.valid_min.unsigned_abs())
            as u128;
        let bias = params
            .bias
            .valid_max
            .unsigned_abs()
            .max(params.bias.valid_min.unsigned_abs())
            as u128;
        let centered = (round.input_bound as u128)
            .checked_mul(2)
            .ok_or(Error::ResourceLimit)?;
        // The normalized image: the centered input times the reciprocal
        // standard deviation and the gain, at the committed fixed-point
        // scale. The scale division only shrinks the true value, so the
        // product of the three published magnitudes plus the rounding term
        // is a safe integer bound over the tile.
        let image = (self.shape.valid as u128)
            .checked_mul(centered)
            .and_then(|value| value.checked_mul(round_bounds.rsqrt))
            .and_then(|value| value.checked_mul(gain))
            .and_then(|value| value.checked_add(1))
            .ok_or(Error::ResourceLimit)?;
        let shifted = q_half
            .checked_add(1)
            .ok_or(Error::ResourceLimit)?;
        let bias_sum = shifted
            .checked_add(bias)
            .ok_or(Error::ResourceLimit)?;
        if shifted >= half || bias_sum >= half || image >= half {
            return Err(Error::InvalidConfig);
        }
        // Relation 0: image plus the quotient and remainder reconstruction.
        let reconstruction = (q_cap - 1)
            .checked_mul(divisor)
            .and_then(|value| value.checked_add(r_cap - 1))
            .and_then(|value| value.checked_add(divisor.checked_mul(q_half)?))
            .and_then(|value| value.checked_add(image))
            .ok_or(Error::ResourceLimit)?;
        if reconstruction >= half {
            return Err(Error::InvalidConfig);
        }
        // Relation 1: shifted value from the quotient and the half selector.
        let shifted_relation = (q_cap - 1)
            .checked_add(sh_cap - 1)
            .and_then(|value| value.checked_add(q_half))
            .and_then(|value| value.checked_add(1))
            .ok_or(Error::ResourceLimit)?;
        if shifted_relation >= half {
            return Err(Error::InvalidConfig);
        }
        // Relations 3 and 4: doubled remainder and tie slack.
        let doubled = (r_cap - 1)
            .checked_mul(2)
            .and_then(|value| value.checked_add(u_cap - 1))
            .ok_or(Error::ResourceLimit)?;
        let tie = (u_cap - 1)
            .checked_add(w_cap - 1)
            .and_then(|value| value.checked_add(divisor))
            .ok_or(Error::ResourceLimit)?;
        if doubled >= half || tie >= half {
            return Err(Error::InvalidConfig);
        }
        // Relation 5 without a clamp interval: out from pre and the half.
        let clamp_relation = bias_sum
            .checked_add(pre_cap - 1)
            .and_then(|value| value.checked_add(pre_half))
            .ok_or(Error::ResourceLimit)?;
        if clamp_relation >= half {
            return Err(Error::InvalidConfig);
        }
        let Some(clamp) = tail.clamp.as_ref() else {
            return Ok(());
        };
        let (low_bound, high_bound) = output.clamp.ok_or(Error::InvalidConfig)?;
        let low_cap = capacity(clamp.low)?;
        let high_cap = capacity(clamp.high)?;
        let low_half = half_range(clamp.low)?;
        let high_half = half_range(clamp.high)?;
        // The clamped form of relation 5 adds both distance words.
        let clamped_relation = clamp_relation
            .checked_add(low_cap - 1)
            .and_then(|value| value.checked_add(high_cap - 1))
            .ok_or(Error::ResourceLimit)?;
        if clamped_relation >= half {
            return Err(Error::InvalidConfig);
        }
        // Relations 7 to 10: one selector times one distance and one slack.
        let low_slack = capacity(clamp.low_strict)?
            .max(capacity(clamp.low_nonstrict)?)
            .checked_sub(1)
            .ok_or(Error::ResourceLimit)?;
        let high_slack = capacity(clamp.high_strict)?
            .max(capacity(clamp.high_nonstrict)?)
            .checked_sub(1)
            .ok_or(Error::ResourceLimit)?;
        let sign_low = (low_cap - 1)
            .checked_add(low_slack)
            .and_then(|value| value.checked_add(1))
            .ok_or(Error::ResourceLimit)?;
        let sign_high = (high_cap - 1)
            .checked_add(high_slack)
            .and_then(|value| value.checked_add(1))
            .ok_or(Error::ResourceLimit)?;
        if sign_low >= half || sign_high >= half {
            return Err(Error::InvalidConfig);
        }
        // The distance rows: the bound word, the pre word and the public
        // target (the two half ranges and the clamp constant) in one row.
        let distance_low = (low_cap - 1)
            .checked_add(pre_cap - 1)
            .and_then(|value| value.checked_add(low_half))
            .and_then(|value| value.checked_add(pre_half))
            .and_then(|value| value.checked_add(low_bound.unsigned_abs() as u128))
            .ok_or(Error::ResourceLimit)?;
        let distance_high = (high_cap - 1)
            .checked_add(pre_cap - 1)
            .and_then(|value| value.checked_add(high_half))
            .and_then(|value| value.checked_add(pre_half))
            .and_then(|value| value.checked_add(high_bound.unsigned_abs() as u128))
            .ok_or(Error::ResourceLimit)?;
        if distance_low >= half || distance_high >= half {
            return Err(Error::InvalidConfig);
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
                writer.put_u64(round.table_real_len);
                writer.put_u64(round.layout.widths.sum as u64);
                writer.put_u64(round.layout.widths.squares as u64);
                writer.put_u64(round.layout.widths.variance as u64);
                writer.put_u64(round.layout.widths.mean as u64);
                writer.put_u64(round.layout.widths.offset as u64);
                writer.put_u64(round.layout.sum_offset);
                writer.put_u64(round.layout.mean_offset);
                put_norm_group(writer, &round.group);
            }
            None => writer.put_u64(0),
        }
        match &self.lookup {
            Some(lookup) => {
                writer.put_u64(1);
                writer.put_digest(&lookup.digest());
            }
            None => writer.put_u64(0),
        }
        match &self.rstd {
            Some(rstd) => {
                writer.put_u64(1);
                writer.put_digest(&rstd.digest());
            }
            None => writer.put_u64(0),
        }
        match &self.params {
            Some(params) => {
                writer.put_u64(1);
                writer.put_digest(&params.digest());
            }
            None => writer.put_u64(0),
        }
        match &self.output {
            Some(output) => {
                writer.put_u64(1);
                output.encode_into(writer);
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
    /// The number of equality-weighted pointwise product instances per
    /// position; zero when the plan carries no output tail.
    pub pointwise: usize,
    /// The hiding entries of one pointwise instance.
    pub pointwise_hiding: usize,
    /// The shown-map masks of one pointwise instance.
    pub pointwise_shown: usize,
    /// The number of tail relation instances per position; zero when the
    /// plan carries no tail section.
    pub tail: usize,
    /// The hiding entries of one tail relation instance.
    pub tail_hiding: usize,
    /// The shown-map masks of one tail relation instance.
    pub tail_shown: usize,
}

/// The hiding and shown mask counts of one equality-weighted pointwise
/// product instance over the column domain of a plan with `variables`
/// product-sumcheck variables (one more than the column bits).
pub fn norm_pointwise_masks(variables: usize) -> Result<(usize, usize)> {
    if variables < 2 {
        return Err(Error::InvalidConfig);
    }
    let weights = (0..variables - 1)
        .map(|_| crate::proof::psc::Weight::equality(Fp3::ZERO, 3))
        .collect::<Result<Vec<_>>>()?;
    let summand = crate::proof::psc_general::Summand::Product;
    let all = crate::proof::psc_general::general_variables(&weights, &summand)?;
    Ok((hp_entries(&all)?, summand.shown()))
}

/// The hiding and shown mask counts of one tail relation instance over the
/// column domain.
pub fn norm_tail_mask_masks(columns: u64) -> Result<(usize, usize)> {
    let variables = output_tail_variables(columns)?;
    let weights = psc_variables(variables).map_err(|_| Error::InvalidConfig)?;
    Ok((hp_entries(&weights)?, SHOWN_MAPS))
}

/// The power-of-two own-mask row count from the index-mask rows and the
/// norm-mask rows. This is the single derivation used by the proximity plan
/// check and by the tests.
pub fn norm_own_mask_rows(index_rows: u64, norm_rows: u64) -> Result<u64> {
    let sum = index_rows
        .checked_add(norm_rows)
        .ok_or(Error::ResourceLimit)?;
    if sum == 0 {
        return Ok(1);
    }
    sum.checked_next_power_of_two().ok_or(Error::ResourceLimit)
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

    /// The number of pointwise product instances of the plan.
    pub fn pointwise_instances(&self) -> Result<usize> {
        (self.positions as usize)
            .checked_mul(self.pointwise)
            .ok_or(Error::ResourceLimit)
    }

    /// The allocated entries of one pointwise instance.
    pub fn pointwise_per_instance(&self) -> Result<usize> {
        self.pointwise_hiding
            .checked_add(self.pointwise_shown)
            .ok_or(Error::ResourceLimit)
    }

    fn pointwise_entries(&self) -> Result<usize> {
        self.pointwise_instances()?
            .checked_mul(self.pointwise_per_instance()?)
            .ok_or(Error::ResourceLimit)
    }

    /// The number of tail relation instances of the plan.
    pub fn tail_instances(&self) -> Result<usize> {
        (self.positions as usize)
            .checked_mul(self.tail)
            .ok_or(Error::ResourceLimit)
    }

    /// The allocated entries of one tail instance.
    pub fn tail_per_instance(&self) -> Result<usize> {
        self.tail_hiding
            .checked_add(self.tail_shown)
            .ok_or(Error::ResourceLimit)
    }

    fn tail_entries(&self) -> Result<usize> {
        self.tail_instances()?
            .checked_mul(self.tail_per_instance()?)
            .ok_or(Error::ResourceLimit)
    }

    fn tail_base(&self, instances: usize) -> Result<usize> {
        self.pointwise_base(instances)?
            .checked_add(self.pointwise_entries()?)
            .ok_or(Error::ResourceLimit)
    }

    /// The compact slot of one hiding entry of one tail instance.
    pub fn tail_hiding_slot(
        &self,
        instances: usize,
        instance: usize,
        index: usize,
    ) -> Result<usize> {
        if self.tail == 0 || instance >= self.tail_instances()? || index >= self.tail_hiding {
            return Err(Error::InvalidIndex);
        }
        self.tail_base(instances)?
            .checked_add(
                instance
                    .checked_mul(self.tail_per_instance()?)
                    .ok_or(Error::ResourceLimit)?,
            )
            .and_then(|base| base.checked_add(index))
            .ok_or(Error::ResourceLimit)
    }

    /// The compact slot of one shown-value mask of one tail instance.
    pub fn tail_shown_slot(
        &self,
        instances: usize,
        instance: usize,
        index: usize,
    ) -> Result<usize> {
        if self.tail == 0 || instance >= self.tail_instances()? || index >= self.tail_shown {
            return Err(Error::InvalidIndex);
        }
        self.tail_base(instances)?
            .checked_add(
                instance
                    .checked_mul(self.tail_per_instance()?)
                    .ok_or(Error::ResourceLimit)?,
            )
            .and_then(|base| base.checked_add(self.tail_hiding))
            .and_then(|base| base.checked_add(index))
            .ok_or(Error::ResourceLimit)
    }

    fn pointwise_base(&self, instances: usize) -> Result<usize> {
        instances
            .checked_mul(self.per_instance()?)
            .ok_or(Error::ResourceLimit)
    }

    /// The compact slot of one hiding entry of one pointwise instance.
    pub fn pointwise_hiding_slot(
        &self,
        instances: usize,
        instance: usize,
        index: usize,
    ) -> Result<usize> {
        if self.pointwise == 0
            || instance >= self.pointwise_instances()?
            || index >= self.pointwise_hiding
        {
            return Err(Error::InvalidIndex);
        }
        self.pointwise_base(instances)?
            .checked_add(
                instance
                    .checked_mul(self.pointwise_per_instance()?)
                    .ok_or(Error::ResourceLimit)?,
            )
            .and_then(|base| base.checked_add(index))
            .ok_or(Error::ResourceLimit)
    }

    /// The compact slot of one shown-value mask of one pointwise instance.
    pub fn pointwise_shown_slot(
        &self,
        instances: usize,
        instance: usize,
        index: usize,
    ) -> Result<usize> {
        if self.pointwise == 0
            || instance >= self.pointwise_instances()?
            || index >= self.pointwise_shown
        {
            return Err(Error::InvalidIndex);
        }
        self.pointwise_base(instances)?
            .checked_add(
                instance
                    .checked_mul(self.pointwise_per_instance()?)
                    .ok_or(Error::ResourceLimit)?,
            )
            .and_then(|base| base.checked_add(self.pointwise_hiding))
            .and_then(|base| base.checked_add(index))
            .ok_or(Error::ResourceLimit)
    }

    /// The total number of allocated entries: one block per product-sumcheck
    /// instance, one block per pointwise instance, plus the reserved claim
    /// entries.
    pub fn total_entries(&self, relations: usize) -> Result<usize> {
        let instances = self.instances(relations)?;
        self.tail_base(instances)?
            .checked_add(self.tail_entries()?)
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
        self.tail_base(instance_count)?
            .checked_add(self.tail_entries()?)
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

    /// The free entries of `rows` mask rows, excluding the SEND pads.
    fn free_entries_for(&self, rows: usize) -> Result<usize> {
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

    /// The free entries of the plan's mask rows, excluding the SEND pad.
    pub fn free_entries(&self) -> Result<usize> {
        let rows = usize::try_from(self.mask_rows).map_err(|_| Error::ResourceLimit)?;
        self.free_entries_for(rows)
    }

    /// The smallest power-of-two row count whose free entries hold one mask
    /// block per instance and the reserved claims. This is the single
    /// derivation of the mask size from the relation count, the mask code
    /// answers and the per-instance allocation.
    pub fn required_rows(&self, relations: usize) -> Result<u64> {
        let total = self.total_entries(relations)?;
        let columns = 1u64
            .checked_shl(self.mask_code.log_n)
            .ok_or(Error::ResourceLimit)?;
        let mut rows = 1usize;
        while (rows as u64) <= columns {
            if self.free_entries_for(rows)? >= total {
                return Ok(rows as u64);
            }
            rows = rows.checked_mul(2).ok_or(Error::ResourceLimit)?;
        }
        Err(Error::ResourceLimit)
    }

    fn validate(&self, positions: u32, variables: usize, relations: usize) -> Result<()> {
        if self.oracle == 0 || self.group == 0 {
            return Err(Error::InvalidConfig);
        }
        if self.mask_rows == 0
            || !self.mask_rows.is_power_of_two()
            || self.mask_rows > (1u64 << self.mask_code.log_n)
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
        if variables < 2 {
            return Err(Error::InvalidConfig);
        }
        let columns = 1u64 << (variables as u32 - 1);
        let (tail_hiding, tail_shown) = norm_tail_mask_masks(columns)
            .map_err(|_| Error::InvalidConfig)?;
        if self.tail_hiding != tail_hiding
            || self.tail_shown != tail_shown
            || self.tail > NORM_OUTPUT_TAIL_RELATIONS
        {
            return Err(Error::InvalidConfig);
        }
        let (pointwise_hiding, pointwise_shown) =
            norm_pointwise_masks(variables).map_err(|_| Error::InvalidConfig)?;
        let variables = psc_variables(variables).map_err(|_| Error::InvalidConfig)?;
        if self.hiding != hp_entries(&variables).map_err(|_| Error::InvalidConfig)? {
            return Err(Error::InvalidConfig);
        }
        if self.pointwise_hiding != pointwise_hiding
            || self.pointwise_shown != pointwise_shown
            || self.pointwise > 2
        {
            return Err(Error::InvalidConfig);
        }
        if self.pointwise > 0 && self.claims < self.pointwise_instances()? {
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
        if self.mask_rows < self.required_rows(relations)? {
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
        writer.put_u64(self.pointwise as u64);
        writer.put_u64(self.pointwise_hiding as u64);
        writer.put_u64(self.pointwise_shown as u64);
        writer.put_u64(self.tail as u64);
        writer.put_u64(self.tail_hiding as u64);
        writer.put_u64(self.tail_shown as u64);
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
        Self::decode_with(&plan.tree, plan.answers, plan.openings, bytes)
    }

    /// Reads a mask-only message against a trusted tree plan, answer count and
    /// opening count. The norm mask and the lookup mask share this form.
    fn decode_with(
        plan_tree: &zkie_engine::pcs::request::runtime_tree::RuntimeTreePlan,
        answers: u32,
        openings: usize,
        bytes: &[u8],
    ) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        let tree = get_tree_plan(&mut reader)?;
        if tree != *plan_tree {
            return Err(Error::StatementMismatch);
        }
        let root = reader.get_digest()?;
        if root == Digest::ZERO {
            return Err(Error::InvalidEncoding);
        }
        let sample = SendSamples::decode(reader.get_bytes()?)?;
        if sample.seed.len() != answers as usize
            || sample.answers.len() != answers as usize
        {
            return Err(Error::ShapeMismatch);
        }
        let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if count != openings {
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

/// The committed mask entries of the pointwise product instances, in
/// position-major order (product, image).
pub fn norm_pointwise_entries(
    plan: &NormMaskPlan,
    mask: &CommittedNormMask,
    instances: usize,
) -> Result<Vec<crate::proof::psc_general::GeneralMaskEntries>> {
    let count = plan.pointwise_instances()?;
    let mut entries = Vec::with_capacity(count);
    for instance in 0..count {
        let mut hiding = Vec::with_capacity(plan.pointwise_hiding);
        for index in 0..plan.pointwise_hiding {
            hiding.push(mask.entry(
                plan,
                plan.pointwise_hiding_slot(instances, instance, index)?,
            )?);
        }
        let mut shown = Vec::with_capacity(plan.pointwise_shown);
        for index in 0..plan.pointwise_shown {
            shown.push(mask.entry(
                plan,
                plan.pointwise_shown_slot(instances, instance, index)?,
            )?);
        }
        entries.push(crate::proof::psc_general::GeneralMaskEntries { hiding, shown });
    }
    Ok(entries)
}

/// The committed mask entries of the tail relation instances, in
/// position-major and relation order.
pub fn norm_tail_entries(
    plan: &NormMaskPlan,
    mask: &CommittedNormMask,
    instances: usize,
) -> Result<Vec<DotMaskEntries>> {
    let count = plan.tail_instances()?;
    let mut entries = Vec::with_capacity(count);
    for instance in 0..count {
        let mut hiding = Vec::with_capacity(plan.tail_hiding);
        for index in 0..plan.tail_hiding {
            hiding.push(mask.entry(plan, plan.tail_hiding_slot(instances, instance, index)?)?);
        }
        let shown = [
            mask.entry(plan, plan.tail_shown_slot(instances, instance, 0)?)?,
            mask.entry(plan, plan.tail_shown_slot(instances, instance, 1)?)?,
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

/// The prover-side committed material of one folded parameter claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormParamClaimWitness {
    pub message: Vec<Vec<Fp3>>,
    pub tails: Vec<Vec<Fp3>>,
    pub salts: Vec<Vec<u8>>,
}

/// The fixed byte widths of the output tail witness rows. The quotient,
/// shifted and pre-clamp values are carried as offset digit rows; the
/// remainder is nonnegative and bounded by the shift divisor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormOutputTailWidths {
    pub quotient: usize,
    pub remainder: usize,
    /// The doubled remainder `2 r` used by the half-up selector row.
    pub doubled_remainder: usize,
    /// The tie slack `2 r - 2^shift * selector`, nonnegative below the
    /// divisor.
    pub tie_slack: usize,
    pub shifted: usize,
    pub pre_clamp: usize,
}

impl NormOutputTailWidths {
    pub fn fault(&self, shift: u32) -> Option<&'static str> {
        for width in [
            self.quotient,
            self.remainder,
            self.doubled_remainder,
            self.tie_slack,
            self.shifted,
            self.pre_clamp,
        ] {
            if width == 0 || width > 7 {
                return Some("an output tail width is out of range");
            }
            if digit_bound(width).ok()? >= MODULUS as u128 {
                return Some("an output tail width does not fit the modulus");
            }
        }
        if shift >= 63 {
            return Some("the output shift is too large");
        }
        let divisor = 1u128 << shift;
        if digit_bound(self.remainder).ok()? + 1 < divisor {
            return Some("the remainder width does not cover the shift divisor");
        }
        if digit_bound(self.doubled_remainder).ok()? + 1 < 2 * divisor {
            return Some("the doubled remainder width does not cover twice the divisor");
        }
        if digit_bound(self.tie_slack).ok()? + 1 < divisor {
            return Some("the tie slack width does not cover the divisor");
        }
        None
    }

    /// The slots of one column: quotient, remainder, half selector, shifted
    /// value and pre-clamp value.
    pub fn slots(&self) -> Result<usize> {
        self.quotient
            .checked_add(self.remainder)
            .and_then(|value| value.checked_add(self.doubled_remainder))
            .and_then(|value| value.checked_add(self.tie_slack))
            .and_then(|value| value.checked_add(1))
            .and_then(|value| value.checked_add(self.shifted))
            .and_then(|value| value.checked_add(self.pre_clamp))
            .ok_or(Error::ResourceLimit)
    }

    /// The message width of one tail row: every column carries all of its
    /// slots, so the width is the column count times the slot count rounded
    /// up to a power of two.
    pub fn columns(&self, columns: u64) -> Result<u64> {
        columns
            .checked_mul(u64::try_from(self.slots()?).map_err(|_| Error::ResourceLimit)?)
            .and_then(|value| value.checked_next_power_of_two())
            .ok_or(Error::ResourceLimit)
    }
}

/// The byte widths of the clamp comparison rows: the offset distances to the
/// low and high bounds and the four sign-consistency slacks (strict and
/// non-strict per bound).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormClampWidths {
    pub low: usize,
    pub high: usize,
    pub low_strict: usize,
    pub low_nonstrict: usize,
    pub high_strict: usize,
    pub high_nonstrict: usize,
}

/// The trusted plan of the output tail witness rows: the quotient, remainder,
/// half selector, shifted value and pre-clamp value of every column, the
/// optional clamp rows, and the byte widths of their digit rows. The shift and
/// clamp bounds are the output plan's trusted values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormOutputTailPlan {
    pub group: NormGroupPlan,
    pub widths: NormOutputTailWidths,
    pub clamp: Option<NormClampWidths>,
}

impl NormOutputTailPlan {
    /// The slots of one column including the clamp rows.
    pub fn slots(&self) -> Result<usize> {
        let clamp = match &self.clamp {
            Some(clamp) => clamp
                .low
                .checked_add(clamp.high)
                .and_then(|value| value.checked_add(clamp.low_strict))
                .and_then(|value| value.checked_add(clamp.low_nonstrict))
                .and_then(|value| value.checked_add(clamp.high_strict))
                .and_then(|value| value.checked_add(clamp.high_nonstrict))
                .and_then(|value| value.checked_add(2))
                .ok_or(Error::ResourceLimit)?,
            None => 0,
        };
        self.widths
            .slots()?
            .checked_add(clamp)
            .ok_or(Error::ResourceLimit)
    }

    /// The message width of one tail row over the column domain.
    pub fn columns(&self, columns: u64) -> Result<u64> {
        columns
            .checked_mul(u64::try_from(self.slots()?).map_err(|_| Error::ResourceLimit)?)
            .and_then(|value| value.checked_next_power_of_two())
            .ok_or(Error::ResourceLimit)
    }

    fn validate(&self, output: &NormOutputPlan, plan: &NormPlan) -> Result<()> {
        if self.widths.fault(output.shift).is_some() {
            return Err(Error::InvalidConfig);
        }
        match (&self.clamp, output.clamp) {
            (Some(clamp), Some((low, high))) => {
                for width in [
                    clamp.low,
                    clamp.high,
                    clamp.low_strict,
                    clamp.low_nonstrict,
                    clamp.high_strict,
                    clamp.high_nonstrict,
                ] {
                    if width == 0 || width > 7 {
                        return Err(Error::InvalidConfig);
                    }
                    if digit_bound(width).map_err(|_| Error::InvalidConfig)? >= MODULUS as u128 {
                        return Err(Error::InvalidConfig);
                    }
                }
                let pre_capacity = 1u128 << (8 * self.widths.pre_clamp as u32);
                let bound = (low.unsigned_abs() as u128).max(high.unsigned_abs() as u128);
                let span = pre_capacity
                    .checked_add(bound)
                    .ok_or(Error::InvalidConfig)?;
                if digit_bound(clamp.low).map_err(|_| Error::InvalidConfig)? + 1 < span
                    || digit_bound(clamp.high).map_err(|_| Error::InvalidConfig)? + 1 < span
                {
                    return Err(Error::InvalidConfig);
                }
            }
            (None, None) => {}
            _ => return Err(Error::InvalidConfig),
        }
        if self.group.columns != self.columns(plan.columns)?
            || self.group.fault(plan.shape.positions).is_some()
            || self.group.shape.answers != 1
        {
            return Err(Error::InvalidConfig);
        }
        let mut bases = vec![
            plan.x.oracle_base,
            plan.sum.oracle_base,
            plan.squares.oracle_base,
            plan.mean.oracle_base,
            plan.variance.oracle_base,
            output.broadcast.oracle_base,
            output.product.oracle_base,
            output.image.oracle_base,
            output.out.oracle_base,
            self.group.oracle_base,
        ];
        if let Some(round) = &plan.round {
            bases.push(round.group.oracle_base);
        }
        bases.sort_unstable();
        if bases.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }

    fn encode_into(&self, writer: &mut CanonicalWriter) {
        put_norm_group(writer, &self.group);
        writer.put_u64(self.widths.quotient as u64);
        writer.put_u64(self.widths.remainder as u64);
        writer.put_u64(self.widths.doubled_remainder as u64);
        writer.put_u64(self.widths.tie_slack as u64);
        writer.put_u64(self.widths.shifted as u64);
        writer.put_u64(self.widths.pre_clamp as u64);
        match &self.clamp {
            Some(clamp) => {
                writer.put_u64(1);
                writer.put_u64(clamp.low as u64);
                writer.put_u64(clamp.high as u64);
                writer.put_u64(clamp.low_strict as u64);
                writer.put_u64(clamp.low_nonstrict as u64);
                writer.put_u64(clamp.high_strict as u64);
                writer.put_u64(clamp.high_nonstrict as u64);
            }
            None => writer.put_u64(0),
        }
    }

    /// The slot values of one position, position-major over the column
    /// domain: per column the offset quotient digits, the remainder digits,
    /// the half selector, the offset shifted digits and the offset pre-clamp
    /// digits. Signed quantities use a public half-range offset.
    pub fn values(
        &self,
        row: &crate::proof::hidden_norm::NormOutputRow,
        columns: u64,
    ) -> Result<Vec<Fp3>> {
        let width = usize::try_from(self.group.columns).map_err(|_| Error::ResourceLimit)?;
        let count = usize::try_from(columns).map_err(|_| Error::ResourceLimit)?;
        let slots = self.slots()?;
        let needed = count.checked_mul(slots).ok_or(Error::ResourceLimit)?;
        if width < needed {
            return Err(Error::ShapeMismatch);
        }
        for list in [&row.quotient, &row.remainder, &row.shifted, &row.pre_clamp] {
            if list.len() != count {
                return Err(Error::ShapeMismatch);
            }
        }
        if row.selector.len() != count {
            return Err(Error::ShapeMismatch);
        }
        let half = |bytes: usize| -> Result<i128> {
            if bytes == 0 || bytes > 7 {
                return Err(Error::InvalidConfig);
            }
            Ok(1i128 << (8 * bytes - 1))
        };
        let put = |values: &mut [Fp3], at: usize, value: i128, bytes: usize| -> Result<()> {
            if value < 0 {
                return Err(Error::BoundViolation);
            }
            let mut remaining = value as u128;
            let mut used = 0usize;
            while used < bytes {
                values[at + used] = Fp3::from_u64((remaining & 0xff) as u64);
                remaining >>= 8;
                used += 1;
            }
            if remaining != 0 {
                return Err(Error::BoundViolation);
            }
            Ok(())
        };
        let half_q = half(self.widths.quotient)?;
        let half_s = half(self.widths.shifted)?;
        let half_p = half(self.widths.pre_clamp)?;
        let mut values = vec![Fp3::ZERO; width];
        for column in 0..count {
            let base = column.checked_mul(slots).ok_or(Error::ResourceLimit)?;
            let quotient = row.quotient[column]
                .checked_add(half_q)
                .ok_or(Error::BoundViolation)?;
            put(&mut values, base, quotient, self.widths.quotient)?;
            let remainder_at = base + self.widths.quotient;
            put(
                &mut values,
                remainder_at,
                row.remainder[column],
                self.widths.remainder,
            )?;
            let doubled_at = remainder_at + self.widths.remainder;
            let divisor = 1i128 << row.shift;
            let doubled = row.remainder[column]
                .checked_mul(2)
                .ok_or(Error::BoundViolation)?;
            let selector_value = if row.selector[column] { 1i128 } else { 0 };
            let tie = doubled
                .checked_sub(divisor * selector_value)
                .ok_or(Error::BoundViolation)?;
            put(
                &mut values,
                doubled_at,
                doubled,
                self.widths.doubled_remainder,
            )?;
            let tie_at = doubled_at + self.widths.doubled_remainder;
            put(&mut values, tie_at, tie, self.widths.tie_slack)?;
            let selector = tie_at + self.widths.tie_slack;
            if row.selector[column] {
                values[selector] = Fp3::ONE;
            }
            let shifted_at = selector + 1;
            let shifted = row.shifted[column]
                .checked_add(half_s)
                .ok_or(Error::BoundViolation)?;
            put(&mut values, shifted_at, shifted, self.widths.shifted)?;
            let pre_at = shifted_at + self.widths.shifted;
            let pre_clamp = row.pre_clamp[column]
                .checked_add(half_p)
                .ok_or(Error::BoundViolation)?;
            put(&mut values, pre_at, pre_clamp, self.widths.pre_clamp)?;
            if let (Some(clamp_widths), Some((low, high))) = (&self.clamp, row.clamp) {
                let low_at = pre_at + self.widths.pre_clamp;
                let high_at = low_at + clamp_widths.low;
                let low_half = 1i128 << (8 * clamp_widths.low as u32 - 1);
                let high_half = 1i128 << (8 * clamp_widths.high as u32 - 1);
                let below = row.pre_clamp[column].checked_sub(low as i128);
                let above = row.pre_clamp[column].checked_sub(high as i128);
                let low_value = below
                    .and_then(|value| value.checked_add(low_half))
                    .ok_or(Error::BoundViolation)?;
                let high_value = above
                    .and_then(|value| value.checked_add(high_half))
                    .ok_or(Error::BoundViolation)?;
                put(&mut values, low_at, low_value, clamp_widths.low)?;
                put(&mut values, high_at, high_value, clamp_widths.high)?;
                let low_selector = high_at + clamp_widths.high;
                let pre = row.pre_clamp[column];
                let below = pre < low as i128;
                let above = pre > high as i128;
                if below {
                    values[low_selector] = Fp3::ONE;
                }
                if above {
                    values[low_selector + 1] = Fp3::ONE;
                }
                let low_strict_at = low_selector + 2;
                let low_nonstrict_at = low_strict_at + clamp_widths.low_strict;
                let high_strict_at = low_nonstrict_at + clamp_widths.low_nonstrict;
                let high_nonstrict_at = high_strict_at + clamp_widths.high_strict;
                let low_strict = if below {
                    (low as i128)
                        .checked_sub(pre)
                        .and_then(|value| value.checked_sub(1))
                        .ok_or(Error::BoundViolation)?
                } else {
                    0
                };
                let low_nonstrict = if below {
                    0
                } else {
                    pre.checked_sub(low as i128)
                        .ok_or(Error::BoundViolation)?
                };
                let high_strict = if above {
                    pre.checked_sub(high as i128)
                        .and_then(|value| value.checked_sub(1))
                        .ok_or(Error::BoundViolation)?
                } else {
                    0
                };
                let high_nonstrict = if above {
                    0
                } else {
                    (high as i128)
                        .checked_sub(pre)
                        .ok_or(Error::BoundViolation)?
                };
                put(&mut values, low_strict_at, low_strict, clamp_widths.low_strict)?;
                put(
                    &mut values,
                    low_nonstrict_at,
                    low_nonstrict,
                    clamp_widths.low_nonstrict,
                )?;
                put(&mut values, high_strict_at, high_strict, clamp_widths.high_strict)?;
                put(
                    &mut values,
                    high_nonstrict_at,
                    high_nonstrict,
                    clamp_widths.high_nonstrict,
                )?;
            }
        }
        Ok(values)
    }
}

/// The number of product slots one column uses in the tail relation packing.
pub const NORM_OUTPUT_TAIL_PAIRS: usize = 32;
/// The number of tail relations: rounding reconstruction, shifted value,
/// selector booleanity, doubled remainder, tie slack, clamp value, selector
/// exclusivity, and the four sign-consistency relations of the clamp
/// selectors.
pub const NORM_OUTPUT_TAIL_RELATIONS: usize = 11;

/// The product-sumcheck variable count of the tail relations.
pub fn output_tail_variables(columns: u64) -> Result<usize> {
    let domain = columns
        .checked_mul(NORM_OUTPUT_TAIL_PAIRS as u64)
        .ok_or(Error::ResourceLimit)?
        .checked_next_power_of_two()
        .ok_or(Error::ResourceLimit)?;
    Ok(domain.trailing_zeros() as usize)
}

/// One factor of a tail relation product. Slot factors name a committed tail
/// slot and an optional scalar; the image factor names the committed image
/// value of the term's column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TailFactor {
    Slot { index: usize, scale: Fp3 },
    Image,
    Out,
    Constant(Fp3),
}

/// One product of a tail relation, at the column and the per-column pair
/// position it occupies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TailTerm {
    pub column: usize,
    pub left: TailFactor,
    pub right: TailFactor,
}

/// The products of one tail relation. Relation 0 is the rounding
/// reconstruction, 1 the shifted value, 2 the selector booleanity, 3 the
/// doubled remainder and 4 the tie slack.
pub fn output_rel_terms(
    plan: &NormOutputTailPlan,
    columns: u64,
    shift: u32,
    relation: usize,
) -> Result<Vec<TailTerm>> {
    let count = usize::try_from(columns).map_err(|_| Error::ResourceLimit)?;
    let slots = plan.slots()?;
    let q_at = 0usize;
    let r_at = plan.widths.quotient;
    let u_at = r_at + plan.widths.remainder;
    let w_at = u_at + plan.widths.doubled_remainder;
    let s_at = w_at + plan.widths.tie_slack;
    let sh_at = s_at + 1;
    if shift >= 63 {
        return Err(Error::InvalidConfig);
    }
    let divisor = 1u128 << shift;
    let q_half = 1u128 << (8 * plan.widths.quotient as u32 - 1);
    let s_half = 1u128 << (8 * plan.widths.shifted as u32 - 1);
    let field = |value: u128| -> Result<Fp3> {
        Ok(Fp3::from_u64(
            u64::try_from(value).map_err(|_| Error::BoundViolation)?,
        ))
    };
    let times = |base: u128, digits: usize| -> Result<Fp3> {
        let mut value = field(base)?;
        for _ in 0..digits {
            value = value.mul(&Fp3::from_u64(256));
        }
        Ok(value)
    };
    let neg = |value: Fp3| Fp3::ZERO.sub(&value);
    let slot = |index: usize| TailFactor::Slot {
        index,
        scale: Fp3::ONE,
    };
    let neg_slot = |index: usize| TailFactor::Slot {
        index,
        scale: neg(Fp3::ONE),
    };
    let mut terms = Vec::new();
    for column in 0..count {
        let base = column * slots;
        match relation {
            0 => {
                terms.push(TailTerm {
                    column,
                    left: TailFactor::Image,
                    right: TailFactor::Constant(Fp3::ONE),
                });
                for digit in 0..plan.widths.quotient {
                    terms.push(TailTerm {
                        column,
                        left: slot(base + q_at + digit),
                        right: TailFactor::Constant(neg(times(divisor, digit)?)),
                    });
                }
                for digit in 0..plan.widths.remainder {
                    terms.push(TailTerm {
                        column,
                        left: slot(base + r_at + digit),
                        right: TailFactor::Constant(neg(times(1, digit)?)),
                    });
                }
                terms.push(TailTerm {
                    column,
                    left: TailFactor::Constant(
                        field(divisor)?.mul(&field(q_half)?),
                    ),
                    right: TailFactor::Constant(Fp3::ONE),
                });
            }
            1 => {
                for digit in 0..plan.widths.shifted {
                    terms.push(TailTerm {
                        column,
                        left: slot(base + sh_at + digit),
                        right: TailFactor::Constant(times(1, digit)?),
                    });
                }
                for digit in 0..plan.widths.quotient {
                    terms.push(TailTerm {
                        column,
                        left: slot(base + q_at + digit),
                        right: TailFactor::Constant(neg(times(1, digit)?)),
                    });
                }
                terms.push(TailTerm {
                    column,
                    left: slot(base + s_at),
                    right: TailFactor::Constant(neg(Fp3::ONE)),
                });
                terms.push(TailTerm {
                    column,
                    left: TailFactor::Constant(field(q_half)?.sub(&field(s_half)?)),
                    right: TailFactor::Constant(Fp3::ONE),
                });
            }
            2 => {
                terms.push(TailTerm {
                    column,
                    left: slot(base + s_at),
                    right: TailFactor::Constant(Fp3::ONE),
                });
                terms.push(TailTerm {
                    column,
                    left: slot(base + s_at),
                    right: neg_slot(base + s_at),
                });
            }
            3 => {
                for digit in 0..plan.widths.doubled_remainder {
                    terms.push(TailTerm {
                        column,
                        left: slot(base + u_at + digit),
                        right: TailFactor::Constant(times(1, digit)?),
                    });
                }
                for digit in 0..plan.widths.remainder {
                    terms.push(TailTerm {
                        column,
                        left: slot(base + r_at + digit),
                        right: TailFactor::Constant(neg(times(2, digit)?)),
                    });
                }
            }
            4 => {
                for digit in 0..plan.widths.tie_slack {
                    terms.push(TailTerm {
                        column,
                        left: slot(base + w_at + digit),
                        right: TailFactor::Constant(times(1, digit)?),
                    });
                }
                for digit in 0..plan.widths.doubled_remainder {
                    terms.push(TailTerm {
                        column,
                        left: slot(base + u_at + digit),
                        right: TailFactor::Constant(neg(times(1, digit)?)),
                    });
                }
                terms.push(TailTerm {
                    column,
                    left: slot(base + s_at),
                    right: TailFactor::Constant(field(divisor)?),
                });
            }
            5 => {
                terms.push(TailTerm {
                    column,
                    left: TailFactor::Out,
                    right: TailFactor::Constant(Fp3::ONE),
                });
                let pre_at = sh_at + plan.widths.shifted;
                for digit in 0..plan.widths.pre_clamp {
                    terms.push(TailTerm {
                        column,
                        left: slot(base + pre_at + digit),
                        right: TailFactor::Constant(neg(times(1, digit)?)),
                    });
                }
                if let Some(clamp) = &plan.clamp {
                    let low_at = pre_at + plan.widths.pre_clamp;
                    let high_at = low_at + clamp.low;
                    let low_half = 1u128 << (8 * clamp.low as u32 - 1);
                    let high_half = 1u128 << (8 * clamp.high as u32 - 1);
                    let low_selector = high_at + clamp.high;
                    for digit in 0..clamp.low {
                        terms.push(TailTerm {
                            column,
                            left: slot(base + low_selector),
                            right: TailFactor::Slot {
                                index: base + low_at + digit,
                                scale: times(1, digit)?,
                            },
                        });
                    }
                    for digit in 0..clamp.high {
                        terms.push(TailTerm {
                            column,
                            left: slot(base + low_selector + 1),
                            right: TailFactor::Slot {
                                index: base + high_at + digit,
                                scale: times(1, digit)?,
                            },
                        });
                    }
                    terms.push(TailTerm {
                        column,
                        left: slot(base + low_selector),
                        right: TailFactor::Constant(neg(field(low_half)?)),
                    });
                    terms.push(TailTerm {
                        column,
                        left: slot(base + low_selector + 1),
                        right: TailFactor::Constant(neg(field(high_half)?)),
                    });
                }
                terms.push(TailTerm {
                    column,
                    left: TailFactor::Constant(field(1u128 << (8 * plan.widths.pre_clamp as u32 - 1))?),
                    right: TailFactor::Constant(Fp3::ONE),
                });
            }
            6 => {
                if plan.clamp.is_some() {
                    let pre_at = sh_at + plan.widths.shifted;
                    let low_at = pre_at + plan.widths.pre_clamp;
                    let clamp = plan.clamp.as_ref().ok_or(Error::InvalidConfig)?;
                    let high_at = low_at + clamp.low;
                    let low_selector = high_at + clamp.high;
                    terms.push(TailTerm {
                        column,
                        left: slot(base + low_selector),
                        right: slot(base + low_selector + 1),
                    });
                }
            }
            7..=10 => {
                let Some(clamp_widths) = plan.clamp.as_ref() else {
                    continue;
                };
                let pre_at = sh_at + plan.widths.shifted;
                let low_at = pre_at + plan.widths.pre_clamp;
                let high_at = low_at + clamp_widths.low;
                let low_selector = high_at + clamp_widths.high;
                let high_selector = low_selector + 1;
                let low_strict_at = low_selector + 2;
                let low_nonstrict_at = low_strict_at + clamp_widths.low_strict;
                let high_strict_at = low_nonstrict_at + clamp_widths.low_nonstrict;
                let high_nonstrict_at = high_strict_at + clamp_widths.high_strict;
                let low_half = 1u128 << (8 * clamp_widths.low as u32 - 1);
                let high_half = 1u128 << (8 * clamp_widths.high as u32 - 1);
                let sel = |index: usize| slot(base + index);
                let digit_at = |offset: usize, digit: usize| TailFactor::Slot {
                    index: base + offset + digit,
                    scale: times(1, digit).unwrap_or(Fp3::ZERO),
                };
                match relation {
                    7 => {
                        // s_lo * (u + 1 + alpha_lo) = 0, u = sum low digits - half_l
                        for digit in 0..clamp_widths.low {
                            terms.push(TailTerm { column, left: sel(low_selector), right: digit_at(low_at, digit) });
                        }
                        terms.push(TailTerm { column, left: sel(low_selector), right: TailFactor::Constant(neg(field(low_half)?)) });
                        terms.push(TailTerm { column, left: sel(low_selector), right: TailFactor::Constant(Fp3::ONE) });
                        for digit in 0..clamp_widths.low_strict {
                            terms.push(TailTerm { column, left: sel(low_selector), right: digit_at(low_strict_at, digit) });
                        }
                    }
                    8 => {
                        // (1 - s_lo) * (u - beta_lo) = 0
                        for digit in 0..clamp_widths.low {
                            terms.push(TailTerm { column, left: slot(base + low_at + digit), right: TailFactor::Constant(times(1, digit)?) });
                            terms.push(TailTerm { column, left: sel(low_selector), right: TailFactor::Slot { index: base + low_at + digit, scale: neg(times(1, digit)?) } });
                        }
                        for digit in 0..clamp_widths.low_nonstrict {
                            terms.push(TailTerm { column, left: slot(base + low_nonstrict_at + digit), right: TailFactor::Constant(neg(times(1, digit)?)) });
                            terms.push(TailTerm { column, left: sel(low_selector), right: digit_at(low_nonstrict_at, digit) });
                        }
                        terms.push(TailTerm { column, left: TailFactor::Constant(neg(field(low_half)?)), right: TailFactor::Constant(Fp3::ONE) });
                        terms.push(TailTerm { column, left: sel(low_selector), right: TailFactor::Constant(field(low_half)?) });
                    }
                    9 => {
                        // s_hi * (v - 1 - alpha_hi) = 0, v = sum high digits - half_h
                        for digit in 0..clamp_widths.high {
                            terms.push(TailTerm { column, left: sel(high_selector), right: digit_at(high_at, digit) });
                        }
                        terms.push(TailTerm { column, left: sel(high_selector), right: TailFactor::Constant(neg(field(high_half)?)) });
                        terms.push(TailTerm { column, left: sel(high_selector), right: TailFactor::Constant(neg(Fp3::ONE)) });
                        for digit in 0..clamp_widths.high_strict {
                            terms.push(TailTerm { column, left: sel(high_selector), right: TailFactor::Slot { index: base + high_strict_at + digit, scale: neg(times(1, digit)?) } });
                        }
                    }
                    _ => {
                        // (1 - s_hi) * (v + beta_hi) = 0
                        for digit in 0..clamp_widths.high {
                            terms.push(TailTerm { column, left: slot(base + high_at + digit), right: TailFactor::Constant(times(1, digit)?) });
                            terms.push(TailTerm { column, left: sel(high_selector), right: TailFactor::Slot { index: base + high_at + digit, scale: neg(times(1, digit)?) } });
                        }
                        for digit in 0..clamp_widths.high_nonstrict {
                            terms.push(TailTerm { column, left: slot(base + high_nonstrict_at + digit), right: TailFactor::Constant(times(1, digit)?) });
                            terms.push(TailTerm { column, left: sel(high_selector), right: TailFactor::Slot { index: base + high_nonstrict_at + digit, scale: neg(times(1, digit)?) } });
                        }
                        terms.push(TailTerm { column, left: TailFactor::Constant(neg(field(high_half)?)), right: TailFactor::Constant(Fp3::ONE) });
                        terms.push(TailTerm { column, left: sel(high_selector), right: TailFactor::Constant(field(high_half)?) });
                    }
                }
            }
            _ => return Err(Error::UnsupportedPrimitive),
        }
    }
    Ok(terms)
}



fn factor_value(
    factor: &TailFactor,
    column: usize,
    values: &[Fp3],
    image: &[Fp3],
    out: &[Fp3],
) -> Result<Fp3> {
    Ok(match factor {
        TailFactor::Slot { index, scale } => values
            .get(*index)
            .ok_or(Error::ShapeMismatch)?
            .mul(scale),
        TailFactor::Image => *image.get(column).ok_or(Error::ShapeMismatch)?,
        TailFactor::Out => *out.get(column).ok_or(Error::ShapeMismatch)?,
        TailFactor::Constant(value) => *value,
    })
}

/// The value-level maps of one tail relation: the two product vectors whose
/// inner product is zero on the exact witness. The row binding consumes the
/// same term enumeration.
pub fn output_rel_map(
    plan: &NormOutputTailPlan,
    row: &crate::proof::hidden_norm::NormOutputRow,
    columns: u64,
    shift: u32,
    relation: usize,
) -> Result<(Vec<Fp3>, Vec<Fp3>)> {
    use crate::proof::output_closure::fp3_signed;

    let count = usize::try_from(columns).map_err(|_| Error::ResourceLimit)?;
    let values = plan.values(row, columns)?;
    let terms = output_rel_terms(plan, columns, shift, relation)?;
    let domain = usize::try_from(
        columns
            .checked_mul(NORM_OUTPUT_TAIL_PAIRS as u64)
            .and_then(|value| value.checked_next_power_of_two())
            .ok_or(Error::ResourceLimit)?,
    )
    .map_err(|_| Error::ResourceLimit)?;
    let mut image = Vec::with_capacity(count);
    let mut out = Vec::with_capacity(count);
    if row.out.len() != count {
        return Err(Error::ShapeMismatch);
    }
    for column in 0..count {
        image.push(fp3_signed(row.image[column])?);
        out.push(fp3_signed(row.out[column] as i128)?);
    }
    let mut left = vec![Fp3::ZERO; domain];
    let mut right = vec![Fp3::ZERO; domain];
    let mut used = vec![0usize; count];
    for term in terms {
        if term.column >= count {
            return Err(Error::ShapeMismatch);
        }
        let at = term.column * NORM_OUTPUT_TAIL_PAIRS + used[term.column];
        used[term.column] += 1;
        if used[term.column] > NORM_OUTPUT_TAIL_PAIRS {
            return Err(Error::ShapeMismatch);
        }
        left[at] = factor_value(&term.left, term.column, &values, &image, &out)?;
        right[at] = factor_value(&term.right, term.column, &values, &image, &out)?;
    }
    Ok((left, right))
}

/// The trusted plan of the norm output tail. The first three groups are the
/// committed column vectors the two pointwise products chain through; the out
/// group is the hd object the next layer consumes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormOutputPlan {
    /// The rstd value repeated over every column of the position.
    pub broadcast: NormGroupPlan,
    /// The pointwise product of the broadcast row and the gain row.
    pub product: NormGroupPlan,
    /// The pointwise product of the centered row and the product row.
    pub image: NormGroupPlan,
    /// The rounded image after the output shift and the bias addition, clamped.
    pub out: NormGroupPlan,
    /// The output shift of the runtime rounding.
    pub shift: u32,
    /// The optional clamp interval of the runtime op.
    pub clamp: Option<(i64, i64)>,
    /// The committed witness rows of the rounding, bias and clamp tail.
    pub tail: Option<NormOutputTailPlan>,
}

impl NormOutputPlan {
    /// The hidden output object the next layer consumes. The out group is the
    /// same committed group the tail relation rows address, at the same
    /// oracles and roots; the consumer reads this object, not a copy and not a
    /// second root.
    pub fn hd_group(&self) -> &NormGroupPlan {
        &self.out
    }

    fn validate(&self, plan: &NormPlan) -> Result<()> {
        for group in [&self.broadcast, &self.product, &self.image, &self.out] {
            if group.columns != plan.columns
                || group.fault(plan.shape.positions).is_some()
                || group.shape.answers != 1
            {
                return Err(Error::InvalidConfig);
            }
        }
        if self.shift >= 63 {
            return Err(Error::InvalidConfig);
        }
        if let Some((low, high)) = self.clamp {
            if low > high {
                return Err(Error::InvalidConfig);
            }
        }
        let mut bases = [
            self.broadcast.oracle_base,
            self.product.oracle_base,
            self.image.oracle_base,
            self.out.oracle_base,
        ];
        bases.sort_unstable();
        if bases.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(Error::InvalidConfig);
        }
        let others = [
            plan.x.oracle_base,
            plan.sum.oracle_base,
            plan.squares.oracle_base,
            plan.mean.oracle_base,
            plan.variance.oracle_base,
        ];
        if bases.iter().any(|base| others.contains(base)) {
            return Err(Error::InvalidConfig);
        }
        if let Some(tail) = &self.tail {
            tail.validate(self, plan)?;
        }
        Ok(())
    }

    fn encode_into(&self, writer: &mut CanonicalWriter) {
        put_norm_group(writer, &self.broadcast);
        put_norm_group(writer, &self.product);
        put_norm_group(writer, &self.image);
        put_norm_group(writer, &self.out);
        writer.put_u64(self.shift as u64);
        match self.clamp {
            Some((low, high)) => {
                writer.put_u64(1);
                writer.put_u64(low as u64);
                writer.put_u64(high as u64);
            }
            None => writer.put_u64(0),
        }
        match &self.tail {
            Some(tail) => {
                writer.put_u64(1);
                tail.encode_into(writer);
            }
            None => writer.put_u64(0),
        }
    }
}

/// The public first-stage commitments of the four output tail groups.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormOutputMessage {
    pub broadcast: CommittedPscGroup,
    pub product: CommittedPscGroup,
    pub image: CommittedPscGroup,
    pub out: CommittedPscGroup,
}

impl NormOutputMessage {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut writer = CanonicalWriter::new();
        for group in [&self.broadcast, &self.product, &self.image, &self.out] {
            group.encode_into(&mut writer)?;
        }
        Ok(writer.into_bytes())
    }

    pub fn decode(plan: &NormOutputPlan, positions: u32, bytes: &[u8]) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        let broadcast = CommittedPscGroup::decode(&mut reader, &plan.broadcast.shape, positions)?;
        let product = CommittedPscGroup::decode(&mut reader, &plan.product.shape, positions)?;
        let image = CommittedPscGroup::decode(&mut reader, &plan.image.shape, positions)?;
        let out = CommittedPscGroup::decode(&mut reader, &plan.out.shape, positions)?;
        reader.finish()?;
        let message = Self {
            broadcast,
            product,
            image,
            out,
        };
        message.check(plan, positions)?;
        Ok(message)
    }

    /// Checks every group declaration against the trusted output plan. The
    /// roots are proof values; only the shape, oracle, tree and group
    /// declarations are compared with the plan.
    pub fn check(&self, plan: &NormOutputPlan, positions: u32) -> Result<()> {
        for (message, group) in [
            (&self.broadcast, &plan.broadcast),
            (&self.product, &plan.product),
            (&self.image, &plan.image),
            (&self.out, &plan.out),
        ] {
            if message.message.group.group != group.group
                || message.fault(&group.shape, positions).is_some()
                || message.message.roots.len() != positions as usize
            {
                return Err(Error::StatementMismatch);
            }
            for (position, entry) in message.message.group.entries.iter().enumerate() {
                if entry.position != position as u32
                    || entry.oracle
                        != group
                            .oracle_base
                            .checked_add(position as u64)
                            .ok_or(Error::ResourceLimit)?
                    || entry.shape != group.shape.oracle_shape()
                    || entry.tree != group.shape.tree_plan(position as u32)?
                    || entry.root == Digest::ZERO
                {
                    return Err(Error::StatementMismatch);
                }
            }
        }
        Ok(())
    }
}

/// The prover-side parameter vectors and folded claim material.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormParamWitness {
    pub gain: Vec<i64>,
    pub bias: Vec<i64>,
    pub gain_claim: Option<NormParamClaimWitness>,
    pub bias_claim: Option<NormParamClaimWitness>,
}

/// The committed output tail group of the first stage. The slot values are
/// recomputed from the exact output rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormOutputTailWitness {
    pub committed: NormGroupWitness,
}

/// The prover-side output tail: the exact integer rows, the committed column
/// groups of the broadcast, product, image and out rows, and the committed
/// tail group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormOutputWitness {
    pub rows: Vec<crate::proof::hidden_norm::NormOutputRow>,
    pub broadcast: NormGroupWitness,
    pub product: NormGroupWitness,
    pub image: NormGroupWitness,
    pub out: NormGroupWitness,
    pub tail: Option<NormOutputTailWitness>,
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
    pub round: Option<NormGroupWitness>,
    /// The private rsqrt index rows, one per position.
    pub rsqrt: Option<Vec<crate::proof::hidden_norm::RsqrtRow>>,
    pub mask: Option<NormMaskWitness>,
    /// The private lookup material, when the plan carries the lookup.
    pub lookup: Option<NormLookupWitness>,
    /// The private frozen-table lookup material, when the plan carries it.
    pub rstd: Option<NormLookupWitness>,
    /// The private parameter vectors, when the plan carries the tables.
    pub params: Option<NormParamWitness>,
    /// The private output tail rows and commitments, when the plan carries
    /// the output groups.
    pub output: Option<NormOutputWitness>,
}

/// The private lookup material of one norm session. The multiplicity bits are
/// supplied by the caller; the committed messages are filled by the first
/// stage and consumed by the second-stage lookup transcript.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormLookupWitness {
    /// The multiplicity-bit values, plane-major.
    pub bits: Vec<Fp3>,
    pub messages: Option<crate::proof::lookup_messages::LookupMessages>,
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
    /// The checked lookup transcript and claims, when the plan carries the
    /// lookup. The row builder consumes the claims on both sides.
    pub lookup: Option<NormLookupSecond>,
    /// The checked frozen-table lookup transcript and claims.
    pub rstd: Option<NormLookupSecond>,
    /// The checked pointwise product instances, when the plan carries the
    /// output tail. The row builder consumes them on both sides.
    pub output: Option<Vec<NormOutputBinding>>,
    /// The checked tail relation instances, when the plan carries the tail
    /// section.
    pub tail: Option<Vec<NormTailBinding>>,
    /// The bias addition point, when the plan carries the tail section.
    pub bias_point: Option<Vec<Fp3>>,
}

/// One checked tail relation instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormTailBinding {
    pub position: u32,
    pub relation: usize,
    pub output: DotOutput,
}

/// One checked pointwise product instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormOutputBinding {
    pub position: u32,
    /// Zero is the broadcast/gain product; one is the centered/product
    /// image.
    pub kind: u8,
    /// The original column point the instance claim was drawn at. The claim
    /// row's equality coefficients come from this point, not from the
    /// transcript's final challenge point.
    pub point: Vec<Fp3>,
    /// The claim value the verifier was given.
    pub claim: Fp3,
    /// The checked transcript output claims.
    pub output: crate::proof::psc_general::GeneralOutput,
}

/// The lookup commitments of the first stage: the multiplicity-bit group and
/// the mask-only message, each with its SEND sample answer, opening positions
/// and runtime opening. The key vector is read from the committed rounding
/// rows and is not a message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormLookupMessage {
    pub bits: CommittedPscGroup,
    pub mask: NormMaskMessage,
}

impl NormLookupMessage {
    fn encode(&self) -> Result<Vec<u8>> {
        let mut writer = CanonicalWriter::new();
        self.bits.encode_into(&mut writer)?;
        writer.put_bytes(&self.mask.encode()?);
        Ok(writer.into_bytes())
    }

    fn decode(plan: &NormLookupPlan, bytes: &[u8]) -> Result<Self> {
        let commit = &plan.commit;
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        let bits = CommittedPscGroup::decode(&mut reader, &commit.bits_shape, 1)?;
        let mask = NormMaskMessage::decode_with(
            &commit.mask_tree,
            1,
            commit.mask_openings,
            reader.get_bytes()?,
        )?;
        reader.finish()?;
        let message = Self { bits, mask };
        message.check(plan)?;
        Ok(message)
    }

    fn encode_rstd(&self) -> Result<Vec<u8>> {
        self.encode()
    }

    fn decode_rstd_for(plan: &NormRstdPlan, bytes: &[u8]) -> Result<Self> {
        let commit = &plan.commit;
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        let bits = CommittedPscGroup::decode(&mut reader, &commit.bits_shape, 1)?;
        let mask = NormMaskMessage::decode_with(
            &commit.mask_tree,
            1,
            commit.mask_openings,
            reader.get_bytes()?,
        )?;
        reader.finish()?;
        let message = Self { bits, mask };
        message.check_rstd_for(plan)?;
        Ok(message)
    }

    fn check_rstd_for(&self, plan: &NormRstdPlan) -> Result<()> {
        let commit = &plan.commit;
        if self.bits.fault(&commit.bits_shape, 1).is_some() {
            return Err(Error::ShapeMismatch);
        }
        let entry = self
            .bits
            .message
            .group
            .entries
            .first()
            .ok_or(Error::ShapeMismatch)?;
        if self.bits.message.group.group != commit.group
            || entry.position != 0
            || entry.oracle != commit.bits_oracle
            || entry.shape != commit.bits_shape.oracle_shape()
            || entry.tree != commit.bits_shape.tree_plan(0)?
            || entry.root == Digest::ZERO
            || self.bits.message.roots.len() != 1
            || self.bits.sample.seed.len() != commit.bits_shape.answers as usize
            || self.bits.sample.answers.len() != 1
            || self.bits.positions.len() != 1
            || self.bits.positions[0].len() != commit.bits_openings
            || self.bits.openings.len() != 1
            || self.bits.openings[0].positions != self.bits.positions[0]
            || self.mask.tree != commit.mask_tree
            || self.mask.root == Digest::ZERO
            || self.mask.sample.seed.len() != 1
            || self.mask.sample.answers.len() != 1
            || self.mask.positions.len() != commit.mask_openings
            || self.mask.opening.positions != self.mask.positions
        {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }

    fn check(&self, plan: &NormLookupPlan) -> Result<()> {
        let commit = &plan.commit;
        if self.bits.fault(&commit.bits_shape, 1).is_some() {
            return Err(Error::ShapeMismatch);
        }
        let entry = self
            .bits
            .message
            .group
            .entries
            .first()
            .ok_or(Error::ShapeMismatch)?;
        if self.bits.message.group.group != commit.group
            || entry.position != 0
            || entry.oracle != commit.bits_oracle
            || entry.shape != commit.bits_shape.oracle_shape()
            || entry.tree != commit.bits_shape.tree_plan(0)?
            || entry.root == Digest::ZERO
            || self.bits.message.roots.len() != 1
            || self.bits.sample.seed.len() != commit.bits_shape.answers as usize
            || self.bits.sample.answers.len() != 1
            || self.bits.positions.len() != 1
            || self.bits.positions[0].len() != commit.bits_openings
            || self.bits.openings.len() != 1
            || self.bits.openings[0].positions != self.bits.positions[0]
        {
            return Err(Error::ShapeMismatch);
        }
        if self.mask.tree != commit.mask_tree
            || self.mask.root == Digest::ZERO
            || self.mask.sample.seed.len() != 1
            || self.mask.sample.answers.len() != 1
            || self.mask.positions.len() != commit.mask_openings
            || self.mask.opening.positions != self.mask.positions
        {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }
}

/// The lookup proof and the claims it leaves for the later row registration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormLookupSecond {
    pub proof: LookupProof,
    pub claims: LookupOutcome,
}

impl NormLookupSecond {
    fn encode(&self) -> Result<Vec<u8>> {
        let mut writer = CanonicalWriter::new();
        writer.put_bytes(&self.proof.encode()?);
        writer.put_bytes(&self.claims.encode_unchecked());
        Ok(writer.into_bytes())
    }

    fn decode(plan: &NormLookupPlan, bytes: &[u8]) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        let proof = LookupProof::decode(&plan.lookup, reader.get_bytes()?)?;
        let claims = LookupOutcome::decode(&plan.lookup, reader.get_bytes()?)?;
        reader.finish()?;
        Ok(Self { proof, claims })
    }

    fn check(&self, plan: &NormLookupPlan) -> Result<()> {
        self.claims.check(&plan.lookup)
    }

    fn decode_for_rstd(plan: &NormRstdPlan, bytes: &[u8]) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        let proof = LookupProof::decode(&plan.lookup, reader.get_bytes()?)?;
        let claims = LookupOutcome::decode(&plan.lookup, reader.get_bytes()?)?;
        reader.finish()?;
        Ok(Self { proof, claims })
    }

    fn check_rstd(&self, plan: &NormRstdPlan) -> Result<()> {
        self.claims.check(&plan.lookup)
    }
}

/// The first-stage claim message of one norm parameter table: the committed
/// folded row (G_u), its sample seeds and answers, and the openings of the
/// drawn table columns. The table has one row, so the fold weights are the
/// single entry one and there is no gamma.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormParamClaimMessage {
    pub root: Digest,
    pub seeds: Vec<SampleSeed>,
    pub answers: Vec<Fp3>,
    pub openings: Vec<ColumnOpening>,
}

impl NormParamClaimMessage {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut writer = CanonicalWriter::new();
        writer.put_digest(&self.root);
        writer.put_u64(self.seeds.len() as u64);
        for seed in &self.seeds {
            writer.put_u64(seed.z.len() as u64);
            for coordinate in &seed.z {
                writer.put_fp3(coordinate);
            }
            writer.put_fp3(&seed.beta);
        }
        writer.put_u64(self.answers.len() as u64);
        for answer in &self.answers {
            writer.put_fp3(answer);
        }
        writer.put_u64(self.openings.len() as u64);
        for opening in &self.openings {
            put_column_opening(&mut writer, opening);
        }
        Ok(writer.into_bytes())
    }

    fn decode(
        claim: &NormParamClaimPlan,
        identity: &NormParamTableIdentity,
        bytes: &[u8],
    ) -> Result<Self> {
        let gu = &claim.gu;
        let shape = &identity.reader.shape;
        let rows = shape.rows() as usize;
        let log_n = shape.code.log_n as usize;
        let n = shape.code.n();
        let bits = gu.sample_bits()? as usize;
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        let root = reader.get_digest()?;
        if root == Digest::ZERO {
            return Err(Error::InvalidEncoding);
        }
        let seed_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if seed_count != gu.u.answers as usize {
            return Err(Error::ShapeMismatch);
        }
        let mut seeds = Vec::with_capacity(seed_count);
        for _ in 0..seed_count {
            let z_len = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
            if z_len != bits {
                return Err(Error::ShapeMismatch);
            }
            let mut z = Vec::with_capacity(z_len);
            for _ in 0..z_len {
                z.push(reader.get_fp3()?);
            }
            let beta = reader.get_fp3()?;
            if beta == Fp3::ZERO {
                return Err(Error::InvalidEncoding);
            }
            seeds.push(SampleSeed { z, beta });
        }
        let answer_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if answer_count != gu.u.answers as usize {
            return Err(Error::ShapeMismatch);
        }
        let mut answers = Vec::with_capacity(answer_count);
        for _ in 0..answer_count {
            answers.push(reader.get_fp3()?);
        }
        let opening_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if opening_count == 0
            || opening_count > claim.column_draws
            || opening_count as u64 > n
        {
            return Err(Error::ResourceLimit);
        }
        let mut openings = Vec::with_capacity(opening_count);
        let mut previous = None;
        for _ in 0..opening_count {
            let opening = get_column_opening(&mut reader, rows.max(log_n))?;
            if opening.symbols.len() != rows
                || opening.path.len() != log_n
                || opening.salt.len() != shape.salt_bytes
                || opening.position >= n
            {
                return Err(Error::ShapeMismatch);
            }
            if let Some(previous) = previous {
                if opening.position <= previous {
                    return Err(Error::ShapeMismatch);
                }
            }
            previous = Some(opening.position);
            openings.push(opening);
        }
        reader.finish()?;
        Ok(Self {
            root,
            seeds,
            answers,
            openings,
        })
    }

    pub fn check(
        &self,
        claim: &NormParamClaimPlan,
        identity: &NormParamTableIdentity,
    ) -> Result<()> {
        let gu = &claim.gu;
        let bits = gu.sample_bits()? as usize;
        let shape = &identity.reader.shape;
        if self.root == Digest::ZERO
            || self.seeds.len() != gu.u.answers as usize
            || self.answers.len() != gu.u.answers as usize
            || self
                .seeds
                .iter()
                .any(|seed| seed.z.len() != bits || seed.beta == Fp3::ZERO)
            || self.openings.is_empty()
            || self.openings.len() > claim.column_draws
            || self
                .openings
                .iter()
                .any(|opening| opening.symbols.len() != shape.rows() as usize)
        {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }
}

/// The first-stage claims of the norm parameter tables. The bias claim is
/// absent when the bias row shares the gain table; one claim then covers both
/// rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormParamMessage {
    pub gain: NormParamClaimMessage,
    pub bias: Option<NormParamClaimMessage>,
}

impl NormParamMessage {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut writer = CanonicalWriter::new();
        writer.put_bytes(&self.gain.encode()?);
        match &self.bias {
            Some(bias) => {
                writer.put_u64(1);
                writer.put_bytes(&bias.encode()?);
            }
            None => writer.put_u64(0),
        }
        Ok(writer.into_bytes())
    }

    fn decode(plan: &NormParamTablesPlan, bytes: &[u8]) -> Result<Self> {
        let gain_claim = plan.gain_claim.as_ref().ok_or(Error::ShapeMismatch)?;
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        let gain = NormParamClaimMessage::decode(gain_claim, &plan.gain, reader.get_bytes()?)?;
        let bias = match reader.get_u64()? {
            0 => {
                if plan.bias.reader != plan.gain.reader {
                    return Err(Error::ShapeMismatch);
                }
                None
            }
            1 => {
                let bias_claim = plan.bias_claim.as_ref().ok_or(Error::ShapeMismatch)?;
                Some(NormParamClaimMessage::decode(
                    bias_claim,
                    &plan.bias,
                    reader.get_bytes()?,
                )?)
            }
            _ => return Err(Error::InvalidEncoding),
        };
        reader.finish()?;
        Ok(Self { gain, bias })
    }

    pub fn check(&self, plan: &NormParamTablesPlan) -> Result<()> {
        let gain_claim = plan.gain_claim.as_ref().ok_or(Error::ShapeMismatch)?;
        self.gain.check(gain_claim, &plan.gain)?;
        match (&self.bias, plan.bias.reader == plan.gain.reader) {
            (None, true) => {}
            (Some(bias), false) => {
                let bias_claim = plan.bias_claim.as_ref().ok_or(Error::ShapeMismatch)?;
                bias.check(bias_claim, &plan.bias)?;
            }
            _ => return Err(Error::ShapeMismatch),
        }
        Ok(())
    }
}

/// The public first-stage commitment of the output tail group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormOutputTailMessage {
    pub group: CommittedPscGroup,
}

impl NormOutputTailMessage {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut writer = CanonicalWriter::new();
        self.group.encode_into(&mut writer)?;
        Ok(writer.into_bytes())
    }

    pub fn decode(plan: &NormOutputTailPlan, positions: u32, bytes: &[u8]) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        let group = CommittedPscGroup::decode(&mut reader, &plan.group.shape, positions)?;
        reader.finish()?;
        let message = Self { group };
        message.check(plan, positions)?;
        Ok(message)
    }

    pub fn check(&self, plan: &NormOutputTailPlan, positions: u32) -> Result<()> {
        if self.group.message.group.group != plan.group.group
            || self.group.fault(&plan.group.shape, positions).is_some()
            || self.group.message.roots.len() != positions as usize
        {
            return Err(Error::StatementMismatch);
        }
        for (position, entry) in self.group.message.group.entries.iter().enumerate() {
            if entry.position != position as u32
                || entry.oracle
                    != plan
                        .group
                        .oracle_base
                        .checked_add(position as u64)
                        .ok_or(Error::ResourceLimit)?
                || entry.shape != plan.group.shape.oracle_shape()
                || entry.tree != plan.group.shape.tree_plan(position as u32)?
                || entry.root == Digest::ZERO
            {
                return Err(Error::StatementMismatch);
            }
        }
        Ok(())
    }
}

/// The public first-stage message of the integrated norm relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormFirstMessages {
    pub x: CommittedPscGroup,
    pub sum: CommittedPscGroup,
    pub squares: CommittedPscGroup,
    pub mean: CommittedPscGroup,
    pub variance: CommittedPscGroup,
    pub round: Option<CommittedPscGroup>,
    pub mask: Option<NormMaskMessage>,
    /// The lookup commitments, samples and openings, when the plan carries
    /// the lookup. The wire layout omits the field entirely when the plan has
    /// no lookup, so the lookup-free bytes stay unchanged.
    pub lookup: Option<NormLookupMessage>,
    /// The frozen-table lookup commitments, samples and openings, when the
    /// plan carries it.
    pub rstd: Option<NormLookupMessage>,
    /// The norm parameter table openings, when the plan carries the tables.
    pub params: Option<NormParamMessage>,
    /// The output tail group commitments, when the plan carries them.
    pub output: Option<NormOutputMessage>,
    /// The rounding/bias/clamp tail group, when the plan carries it.
    pub tail: Option<NormOutputTailMessage>,
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
        match &self.round {
            Some(round) => {
                writer.put_u64(1);
                round.encode_into(&mut writer)?;
            }
            None => writer.put_u64(0),
        }
        match &self.mask {
            Some(mask) => {
                writer.put_u64(1);
                writer.put_bytes(&mask.encode()?);
            }
            None => writer.put_u64(0),
        }
        if let Some(lookup) = &self.lookup {
            writer.put_bytes(&lookup.encode()?);
        }
        if let Some(rstd) = &self.rstd {
            writer.put_bytes(&rstd.encode_rstd()?);
        }
        if let Some(params) = &self.params {
            writer.put_bytes(&params.encode()?);
        }
        if let Some(output) = &self.output {
            writer.put_bytes(&output.encode()?);
        }
        if let Some(tail) = &self.tail {
            writer.put_bytes(&tail.encode()?);
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
        let round = match reader.get_u64()? {
            0 => {
                if plan.round.is_some() {
                    return Err(Error::ShapeMismatch);
                }
                None
            }
            1 => {
                let round_plan = plan.round.as_ref().ok_or(Error::ShapeMismatch)?;
                Some(CommittedPscGroup::decode(
                    &mut reader,
                    &round_plan.group.shape,
                    positions,
                )?)
            }
            _ => return Err(Error::InvalidEncoding),
        };
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
        let lookup = match &plan.lookup {
            Some(lookup_plan) => Some(NormLookupMessage::decode(
                lookup_plan,
                reader.get_bytes()?,
            )?),
            None => None,
        };
        let rstd = match &plan.rstd {
            Some(rstd_plan) => Some(NormLookupMessage::decode_rstd_for(
                rstd_plan,
                reader.get_bytes()?,
            )?),
            None => None,
        };
        let params = match &plan.params {
            Some(params_plan) => Some(NormParamMessage::decode(
                params_plan,
                reader.get_bytes()?,
            )?),
            None => None,
        };
        let output = match &plan.output {
            Some(output_plan) => Some(NormOutputMessage::decode(
                output_plan,
                positions,
                reader.get_bytes()?,
            )?),
            None => None,
        };
        let tail = match plan.output.as_ref().and_then(|output| output.tail.as_ref()) {
            Some(tail_plan) => Some(NormOutputTailMessage::decode(
                tail_plan,
                positions,
                reader.get_bytes()?,
            )?),
            None => None,
        };
        reader.finish()?;
        let message = Self {
            x,
            sum,
            squares,
            mean,
            variance,
            round,
            mask,
            lookup,
            rstd,
            params,
            output,
            tail,
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
        match (&plan.round, &self.round) {
            (Some(round_plan), Some(message)) => {
                if message.message.group.group != round_plan.group.group
                    || message.fault(&round_plan.group.shape, plan.shape.positions).is_some()
                    || message.message.roots.len() != plan.shape.positions as usize
                {
                    return Err(Error::ShapeMismatch);
                }
                for (position, entry) in message.message.group.entries.iter().enumerate() {
                    if entry.position != position as u32
                        || entry.oracle
                            != round_plan
                                .group
                                .oracle_base
                                .checked_add(position as u64)
                                .ok_or(Error::ResourceLimit)?
                        || entry.shape != round_plan.group.shape.oracle_shape()
                        || entry.tree != round_plan.group.shape.tree_plan(position as u32)?
                        || entry.root == Digest::ZERO
                    {
                        return Err(Error::StatementMismatch);
                    }
                }
            }
            (None, None) => {}
            _ => return Err(Error::ShapeMismatch),
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
        match (&plan.lookup, &self.lookup) {
            (Some(lookup_plan), Some(message)) => message.check(lookup_plan)?,
            (None, None) => {}
            _ => return Err(Error::ShapeMismatch),
        }
        match (&plan.rstd, &self.rstd) {
            (Some(rstd_plan), Some(message)) => message.check_rstd_for(rstd_plan)?,
            (None, None) => {}
            _ => return Err(Error::ShapeMismatch),
        }
        match (&plan.params, &self.params) {
            (Some(params_plan), Some(message)) => message.check(params_plan)?,
            (None, None) => {}
            _ => return Err(Error::ShapeMismatch),
        }
        match (&plan.output, &self.output) {
            (Some(output_plan), Some(message)) => {
                message.check(output_plan, plan.shape.positions)?
            }
            (None, None) => {}
            _ => return Err(Error::ShapeMismatch),
        }
        let tail_plan = plan.output.as_ref().and_then(|output| output.tail.as_ref());
        match (tail_plan, &self.tail) {
            (Some(tail_plan), Some(message)) => {
                message.check(tail_plan, plan.shape.positions)?
            }
            (None, None) => {}
            _ => return Err(Error::ShapeMismatch),
        }
        Ok(())
    }
}

/// The fixed byte widths of the rsqrt witness blocks.
pub const RSQRT_RAW_REMAINDER_WIDTH: usize = 3;
pub const RSQRT_FINE_SLACK_WIDTH: usize = 3;
pub const RSQRT_MID_HIGH_WIDTH: usize = 4;
pub const RSQRT_INDEX_WIDTH: usize = 4;
pub const RSQRT_INDEX_SLACK_WIDTH: usize = 4;

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

    /// The byte width of the rsqrt raw index digits, derived from the
    /// variance width and capped so the rounding product stays below `p/2`.
    pub fn raw(&self) -> usize {
        self.variance.saturating_sub(2).clamp(1, 5)
    }

    /// The byte width of the middle-segment quotient digits.
    pub fn mid_q(&self) -> usize {
        self.raw().saturating_sub(1).max(1)
    }

    /// The byte width of the coarse-segment quotient digits.
    pub fn coarse_q(&self) -> usize {
        self.raw().saturating_sub(2).max(1)
    }

    pub fn slots(&self) -> Result<usize> {
        self.offset
            .checked_mul(4)
            .and_then(|offset| offset.checked_add(self.sum))
            .and_then(|value| value.checked_add(self.squares))
            .and_then(|value| value.checked_add(self.variance))
            .and_then(|value| value.checked_add(self.mean))
            .and_then(|value| value.checked_add(NormRoundLayout::FIXED_SLOTS))
            .and_then(|value| value.checked_add(self.raw()))
            .and_then(|value| value.checked_add(RSQRT_RAW_REMAINDER_WIDTH))
            .and_then(|value| value.checked_add(RSQRT_RAW_REMAINDER_WIDTH))
            .and_then(|value| value.checked_add(3))
            .and_then(|value| value.checked_add(RSQRT_FINE_SLACK_WIDTH))
            .and_then(|value| value.checked_add(self.mid_q()))
            .and_then(|value| value.checked_add(1))
            .and_then(|value| value.checked_add(RSQRT_MID_HIGH_WIDTH))
            .and_then(|value| value.checked_add(self.coarse_q()))
            .and_then(|value| value.checked_add(2))
            .and_then(|value| value.checked_add(1))
            .and_then(|value| value.checked_add(RSQRT_INDEX_WIDTH))
            .and_then(|value| value.checked_add(RSQRT_INDEX_SLACK_WIDTH))
            .and_then(|value| value.checked_add(1))
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
///
/// The six fixed slots hold the mean and variance remainders, offsets and
/// complements. The digit blocks hold the base-256 digits of the sum and mean
/// offset quantities, the squares, the variance, and the four offset and
/// complement values. There is no sign slot: a signed value is represented by
/// its fixed-offset quantity, whose digits are nonnegative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormRoundSlots {
    pub mean_remainder: usize,
    pub mean_offset: usize,
    pub mean_complement: usize,
    pub variance_remainder: usize,
    pub variance_offset: usize,
    pub variance_complement: usize,
    pub sum_digits: usize,
    pub mean_digits: usize,
    pub squares_digits: usize,
    pub variance_digits: usize,
    pub mean_offset_digits: usize,
    pub mean_complement_digits: usize,
    pub variance_offset_digits: usize,
    pub variance_complement_digits: usize,
    pub raw_digits: usize,
    pub raw_remainder: usize,
    pub raw_remainder_slack: usize,
    pub fine_selector: usize,
    pub mid_selector: usize,
    pub coarse_selector: usize,
    pub fine_slack: usize,
    pub mid_q_digits: usize,
    pub mid_r: usize,
    pub mid_high_slack: usize,
    pub coarse_q_digits: usize,
    pub coarse_r: usize,
    pub index_value: usize,
    pub index_digits: usize,
    pub index_slack: usize,
    pub rstd: usize,
    pub total: usize,
}

/// The plan-fixed layout of one rounding witness message.
///
/// `sum_offset` and `mean_offset` are the public fixed offsets of the signed
/// sum and mean: their digit rows hold `value + offset`, and the reconstruction
/// relation subtracts the offset from the digit sum. The nonnegative values
/// (squares, variance, the four offset/complement values) use offset zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormRoundLayout {
    pub widths: NormRoundWidths,
    pub sum_offset: u64,
    pub mean_offset: u64,
    pub slots: NormRoundSlots,
}

impl NormRoundLayout {
    pub const FIXED_SLOTS: usize = 6;

    pub fn new(
        widths: NormRoundWidths,
        valid: u32,
        sum_offset: u64,
        mean_offset: u64,
    ) -> Result<Self> {
        if widths.fault(valid).is_some() {
            return Err(Error::InvalidConfig);
        }
        let sum_capacity = digit_capacity(widths.sum)?;
        let mean_capacity = digit_capacity(widths.mean)?;
        if sum_offset as u128 >= sum_capacity
            || mean_offset as u128 >= mean_capacity
            || sum_capacity
                .checked_add(sum_offset as u128)
                .filter(|value| *value < MODULUS as u128)
                .is_none()
            || mean_capacity
                .checked_add(mean_offset as u128)
                .filter(|value| *value < MODULUS as u128)
                .is_none()
        {
            return Err(Error::InvalidConfig);
        }
        let mean_remainder = 0;
        let mean_offset_slot = 1;
        let mean_complement = 2;
        let variance_remainder = 3;
        let variance_offset = 4;
        let variance_complement = 5;
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
        let raw_digits = next;
        next = next.checked_add(widths.raw()).ok_or(Error::ResourceLimit)?;
        let raw_remainder = next;
        next = next
            .checked_add(RSQRT_RAW_REMAINDER_WIDTH)
            .ok_or(Error::ResourceLimit)?;
        let raw_remainder_slack = next;
        next = next
            .checked_add(RSQRT_RAW_REMAINDER_WIDTH)
            .ok_or(Error::ResourceLimit)?;
        let fine_selector = next;
        next = next.checked_add(1).ok_or(Error::ResourceLimit)?;
        let mid_selector = next;
        next = next.checked_add(1).ok_or(Error::ResourceLimit)?;
        let coarse_selector = next;
        next = next.checked_add(1).ok_or(Error::ResourceLimit)?;
        let fine_slack = next;
        next = next
            .checked_add(RSQRT_FINE_SLACK_WIDTH)
            .ok_or(Error::ResourceLimit)?;
        let mid_q_digits = next;
        next = next
            .checked_add(widths.mid_q())
            .ok_or(Error::ResourceLimit)?;
        let mid_r = next;
        next = next.checked_add(1).ok_or(Error::ResourceLimit)?;
        let mid_high_slack = next;
        next = next
            .checked_add(RSQRT_MID_HIGH_WIDTH)
            .ok_or(Error::ResourceLimit)?;
        let coarse_q_digits = next;
        next = next
            .checked_add(widths.coarse_q())
            .ok_or(Error::ResourceLimit)?;
        let coarse_r = next;
        next = next.checked_add(2).ok_or(Error::ResourceLimit)?;
        let index_value = next;
        next = next.checked_add(1).ok_or(Error::ResourceLimit)?;
        let index_digits = next;
        next = next
            .checked_add(RSQRT_INDEX_WIDTH)
            .ok_or(Error::ResourceLimit)?;
        let index_slack = next;
        next = next
            .checked_add(RSQRT_INDEX_SLACK_WIDTH)
            .ok_or(Error::ResourceLimit)?;
        let rstd = next;
        next = next.checked_add(1).ok_or(Error::ResourceLimit)?;
        if next != widths.slots()? {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            widths,
            sum_offset,
            mean_offset,
            slots: NormRoundSlots {
                mean_remainder,
                mean_offset: mean_offset_slot,
                mean_complement,
                variance_remainder,
                variance_offset,
                variance_complement,
                sum_digits,
                mean_digits,
                squares_digits,
                variance_digits,
                mean_offset_digits,
                mean_complement_digits,
                variance_offset_digits,
                variance_complement_digits,
                raw_digits,
                raw_remainder,
                raw_remainder_slack,
                fine_selector,
                mid_selector,
                coarse_selector,
                fine_slack,
                mid_q_digits,
                mid_r,
                mid_high_slack,
                coarse_q_digits,
                coarse_r,
                index_value,
                index_digits,
                index_slack,
                rstd,
                total: next,
            },
        })
    }

    /// The committed digit slots the byte lookup covers, in plan order: the
    /// fixed-offset sum and mean rows, the squares and variance rows, and the
    /// four offset/complement digit rows. The six value slots (remainders,
    /// offsets and complements) are not digits and are not keys.
    pub fn digit_slots(&self) -> Result<Vec<usize>> {
        let ranges = [
            (self.slots.sum_digits, self.widths.sum),
            (self.slots.squares_digits, self.widths.squares),
            (self.slots.variance_digits, self.widths.variance),
            (self.slots.mean_digits, self.widths.mean),
            (self.slots.mean_offset_digits, self.widths.offset),
            (self.slots.mean_complement_digits, self.widths.offset),
            (self.slots.variance_offset_digits, self.widths.offset),
            (self.slots.variance_complement_digits, self.widths.offset),
            (self.slots.raw_digits, self.widths.raw()),
            (self.slots.raw_remainder, RSQRT_RAW_REMAINDER_WIDTH),
            (self.slots.raw_remainder_slack, RSQRT_RAW_REMAINDER_WIDTH),
            (self.slots.fine_slack, RSQRT_FINE_SLACK_WIDTH),
            (self.slots.mid_q_digits, self.widths.mid_q()),
            (self.slots.mid_r, 1),
            (self.slots.mid_high_slack, RSQRT_MID_HIGH_WIDTH),
            (self.slots.coarse_q_digits, self.widths.coarse_q()),
            (self.slots.coarse_r, 2),
            (self.slots.index_digits, RSQRT_INDEX_WIDTH),
            (self.slots.index_slack, RSQRT_INDEX_SLACK_WIDTH),
        ];
        let mut slots = Vec::new();
        for (start, width) in ranges {
            for index in 0..width {
                slots.push(start.checked_add(index).ok_or(Error::ResourceLimit)?);
            }
        }
        Ok(slots)
    }

    /// Writes one integer core row into the fixed slot vector. The sum and mean
    /// digit blocks hold the fixed-offset quantities; the other digit rows are
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
        let sum_digits = shifted_digits(core.sum, self.sum_offset)?;
        let mean_digits = shifted_digits(core.mean as i128, self.mean_offset)?;
        write_digits(
            &mut values[self.slots.sum_digits..self.slots.sum_digits + self.widths.sum],
            &sum_digits,
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
            &mean_digits,
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

    /// Writes the rsqrt witness slots on top of the rounding core row. The
    /// slot widths are fixed by the plan; a value that does not fit fails
    /// closed.
    pub fn values_with_rsqrt(
        &self,
        core: &NormCoreRow,
        rsqrt: &crate::proof::hidden_norm::RsqrtRow,
        columns: u64,
    ) -> Result<Vec<Fp3>> {
        let mut values = self.values(core, columns)?;
        write_digits(
            &mut values[self.slots.raw_digits..self.slots.raw_digits + self.widths.raw()],
            &rsqrt.raw_digits,
        )?;
        write_digits(
            &mut values[self.slots.raw_remainder
                ..self.slots.raw_remainder + RSQRT_RAW_REMAINDER_WIDTH],
            &rsqrt.raw_remainder_digits,
        )?;
        write_digits(
            &mut values[self.slots.raw_remainder_slack
                ..self.slots.raw_remainder_slack + RSQRT_RAW_REMAINDER_WIDTH],
            &rsqrt.raw_remainder_slack_digits,
        )?;
        values[self.slots.fine_selector] = Fp3::from_u64(rsqrt.fine as u64);
        values[self.slots.mid_selector] = Fp3::from_u64(rsqrt.mid as u64);
        values[self.slots.coarse_selector] = Fp3::from_u64(rsqrt.coarse as u64);
        write_digits(
            &mut values[self.slots.fine_slack..self.slots.fine_slack + RSQRT_FINE_SLACK_WIDTH],
            &rsqrt.fine_slack_digits,
        )?;
        write_digits(
            &mut values[self.slots.mid_q_digits..self.slots.mid_q_digits + self.widths.mid_q()],
            &rsqrt.mid_q_digits,
        )?;
        values[self.slots.mid_r] = Fp3::from_u64(rsqrt.mid_r as u64);
        write_digits(
            &mut values[self.slots.mid_high_slack
                ..self.slots.mid_high_slack + RSQRT_MID_HIGH_WIDTH],
            &rsqrt.mid_high_slack_digits,
        )?;
        write_digits(
            &mut values
                [self.slots.coarse_q_digits..self.slots.coarse_q_digits + self.widths.coarse_q()],
            &rsqrt.coarse_q_digits,
        )?;
        values[self.slots.coarse_r] = Fp3::from_u64((rsqrt.coarse_r & 0xff) as u64);
        values[self.slots.coarse_r + 1] =
            Fp3::from_u64(((rsqrt.coarse_r >> 8) & 0xffff) as u64);
        values[self.slots.index_value] = fp3_signed(rsqrt.index as i128)?;
        write_digits(
            &mut values[self.slots.index_digits
                ..self.slots.index_digits + RSQRT_INDEX_WIDTH],
            &rsqrt.index_digits,
        )?;
        write_digits(
            &mut values[self.slots.index_slack
                ..self.slots.index_slack + RSQRT_INDEX_SLACK_WIDTH],
            &rsqrt.index_slack_digits,
        )?;
        values[self.slots.rstd] = fp3_signed(rsqrt.rstd as i128)?;
        Ok(values)
    }
}

fn shifted_digits(value: i128, offset: u64) -> Result<Vec<u8>> {
    let shifted = value
        .checked_add(offset as i128)
        .ok_or(Error::BoundViolation)?;
    if shifted < 0 {
        return Err(Error::BoundViolation);
    }
    let mut rest = shifted as u128;
    let mut digits = Vec::new();
    loop {
        digits.push((rest & 0xff) as u8);
        rest >>= 8;
        if rest == 0 {
            break;
        }
    }
    Ok(digits)
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
/// `X`, the valid width `d`, the fixed digit widths and the two signed-value
/// offsets. Every bound must stay below `p/2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormNoWrap {
    pub relation_sum: u128,
    pub relation_squares: u128,
    pub relation_mean: u128,
    pub relation_variance: u128,
    pub reconstruction: u128,
    pub offsets: u128,
    pub remainder: u128,
    pub rsqrt: u128,
}

pub fn norm_no_wrap(
    input_bound: u64,
    valid: u32,
    widths: &NormRoundWidths,
    sum_offset: u64,
    mean_offset: u64,
) -> Result<NormNoWrap> {
    if valid == 0 || widths.fault(valid).is_some() {
        return Err(Error::InvalidConfig);
    }
    let x = input_bound as u128;
    let d = valid as u128;
    let sum_capacity = digit_capacity(widths.sum)?;
    let mean_capacity = digit_capacity(widths.mean)?;
    if sum_offset as u128 >= sum_capacity || mean_offset as u128 >= mean_capacity {
        return Err(Error::InvalidConfig);
    }
    let bv = digit_bound(widths.sum)?
        .checked_add(sum_offset as u128)
        .ok_or(Error::ResourceLimit)?;
    let bs = digit_bound(widths.squares)?;
    let bvar = digit_bound(widths.variance)?;
    let bq = digit_bound(widths.mean)?
        .checked_add(mean_offset as u128)
        .ok_or(Error::ResourceLimit)?;
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
        sum_capacity
            .checked_add(sum_offset as u128)
            .ok_or(Error::ResourceLimit)?,
        mean_capacity
            .checked_add(mean_offset as u128)
            .ok_or(Error::ResourceLimit)?,
        digit_capacity(widths.squares)?,
        digit_capacity(widths.variance)?,
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
    let raw_bound = digit_capacity(widths.variance)?
        .checked_div(1u128 << 18)
        .and_then(|value| value.checked_add(1))
        .ok_or(Error::ResourceLimit)?;
    let rounding = digit_capacity(widths.variance)?
        .checked_add(RSQRT_HALF as u128)
        .and_then(|value| value.checked_add((1u128 << 18).checked_mul(raw_bound)?))
        .and_then(|value| value.checked_add(1u128 << 18))
        .ok_or(Error::ResourceLimit)?;
    let segment = raw_bound
        .checked_add(RSQRT_COARSE_END as u128)
        .and_then(|value| value.checked_add(1u128 << 32))
        .and_then(|value| value.checked_add(1u128 << 32))
        .ok_or(Error::ResourceLimit)?;
    let rsqrt = rounding.max(segment);
    let half = (MODULUS as u128) / 2;
    if relation_sum >= half
        || relation_squares >= half
        || relation_mean >= half
        || relation_variance >= half
        || reconstruction >= half
        || offsets >= half
        || remainder >= half
        || rsqrt >= half
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
        rsqrt,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SUM_OFFSET: u64 = 1 << 25;
    const MEAN_OFFSET: u64 = 1 << 16;

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
    fn mask_rows_are_derived_from_the_relation_count() {
        use zkie_engine::pcs::request::runtime_tree::RuntimeTreePlan;
        let plan = NormMaskPlan {
            oracle: 1,
            group: 1,
            mask_code: zkie_engine::pcs::request::code::build_code_plan(6, 32, 1).unwrap(),
            mask_rows: 1,
            answers: 1,
            openings: 1,
            tree: RuntimeTreePlan {
                tree_number: 1,
                leaf_class: 1,
                log_n: 6,
                log_n_zk: 6,
                parts: vec![zkie_engine::pcs::request::runtime_tree::RuntimeTreePart {
                    main_symbols: 0,
                    mask_symbols: 3,
                }],
                salt_bytes: 40,
                run: Digest([0x61; 32]),
                configuration: Digest([0x62; 32]),
            },
            positions: 2,
            hiding: 9,
            shown: 2,
            claims: 1,
            pointwise: 0,
            pointwise_hiding: norm_pointwise_masks(3).unwrap().0,
            pointwise_shown: norm_pointwise_masks(3).unwrap().1,
            tail: 0,
            tail_hiding: norm_tail_mask_masks(4).unwrap().0,
            tail_shown: norm_tail_mask_masks(4).unwrap().1,
        };
        assert_eq!(plan.required_rows(4).unwrap(), 4);
        assert_eq!(plan.required_rows(8).unwrap(), 8);
        assert_eq!(plan.required_rows(17).unwrap(), 16);
        assert_eq!(norm_own_mask_rows(0, 4).unwrap(), 4);
        assert_eq!(norm_own_mask_rows(0, 8).unwrap(), 8);
        assert_eq!(norm_own_mask_rows(4, 8).unwrap(), 16);
        assert_eq!(norm_own_mask_rows(0, 0).unwrap(), 1);
    }

    #[test]
    fn digit_reconstruction_maps_sum_to_zero() {
        use crate::proof::hidden_norm_psc::{
            rel_map, NormGroupPlan, NormRoundLayout, NormRoundPlan, NormRoundWidths,
        };
        let widths = NormRoundWidths {
            sum: 2,
            squares: 2,
            variance: 2,
            mean: 2,
            offset: 1,
        };
        let layout = NormRoundLayout::new(widths, 4, 8, 4).unwrap();
        let columns = 8usize;
        let round = NormRoundPlan {
            group: NormGroupPlan {
                shape: group_shape(2, 1),
                columns: 2,
                group: 1,
                oracle_base: 1,
                openings: 1,
            },
            layout,
            input_bound: 16,
            table_real_len: 8,
        };
        for (row, valid) in [(vec![1i64, 2], 2usize), (vec![1i64, 2, 3, 4], 4usize)] {
            let core = NormCoreRow::from_values(&row, valid).unwrap();
            for relation in 8..=15 {
                let (left, right) =
                    rel_map(&core, &row, valid, columns, relation, Some(&round), None).unwrap();
                let dot = left
                    .iter()
                    .zip(right.iter())
                    .fold(Fp3::ZERO, |sum, (a, b)| sum.add(&a.mul(b)));
                assert_eq!(dot, Fp3::ZERO, "relation {relation} row {row:?}");
            }
        }
    }

    #[test]
    fn norm_plan_accepts_and_binds_a_lookup_carrier() {
        use crate::proof::lookup::{LookupKeyPlan, LookupPlan};
        use crate::proof::lookup_messages::LookupCommitPlan;
        use zkie_engine::pcs::request::code::build_code_plan;
        use zkie_engine::pcs::request::runtime_tree::{RuntimeTreePlan, RuntimeTreePart};

        let (mut plan, _, _) = integrated_fixture();
        let lookup = LookupPlan {
            key_bits: 2,
            mu: 2,
            table_log: 8,
            columns: 1,
        };
        let commit = LookupCommitPlan {
            bits_shape: WShape::new(
                1024,
                build_code_plan(12, 2048, 1).unwrap(),
                1,
                1,
                40,
                60701,
                1,
                Digest([0x64; 32]),
                Digest([0x62; 32]),
            )
            .unwrap(),
            bits_oracle: 9700,
            group: 1,
            bits_openings: 1,
            mask_oracle: 9701,
            mask_code: build_code_plan(6, 32, 1).unwrap(),
            mask_rows: 8,
            mask_openings: 1,
            mask_tree: RuntimeTreePlan {
                tree_number: 60700,
                leaf_class: 1,
                log_n: 6,
                log_n_zk: 6,
                parts: vec![RuntimeTreePart {
                    main_symbols: 0,
                    mask_symbols: 24,
                }],
                salt_bytes: 40,
                run: Digest([0x61; 32]),
                configuration: Digest([0x62; 32]),
            },
        };
        let key = LookupKeyPlan {
            sources: Vec::new(),
            padding_values: vec![Fp3::ZERO],
            table_identity: crate::proof::lookup::BYTE_TABLE_IDENTITY,
        };
        let key_shape = WShape::new(
            4,
            build_code_plan(4, 7, 2).unwrap(),
            1,
            1,
            40,
            1,
            1,
            Digest([0x61; 32]),
            Digest([0x62; 32]),
        )
        .unwrap();
        plan.lookup = Some(NormLookupPlan {
            lookup,
            commit,
            key,
            key_groups: Vec::new(),
        });
        plan.validate().unwrap();
        let with = plan.digest();
        plan.lookup = None;
        assert_ne!(with, plan.digest());
    }

    #[test]
    fn round_layout_fixes_the_slot_map_and_pads_digits() {
        let widths = real_widths();
        let layout = NormRoundLayout::new(widths, 768, SUM_OFFSET, MEAN_OFFSET).unwrap();
        assert_eq!(layout.slots.total, 73);
        assert_eq!(widths.columns().unwrap(), 128);
        let core = NormCoreRow::from_values(&[3, -2, 5, 1], 4).unwrap();
        let values = layout.values(&core, widths.columns().unwrap()).unwrap();
        assert_eq!(values.len(), 128);
        assert_eq!(
            values[layout.slots.mean_remainder],
            fp3_signed(core.mean_remainder).unwrap()
        );
        assert_eq!(
            values[layout.slots.variance_complement],
            fp3_signed(core.variance_complement).unwrap()
        );
        let shifted = core.sum + SUM_OFFSET as i128;
        assert_eq!(
            values[layout.slots.sum_digits],
            Fp3::from_u64((shifted & 0xff) as u64)
        );
        let mut shifted = core.mean as i128 + MEAN_OFFSET as i128;
        let mut mean_digits = Vec::new();
        loop {
            mean_digits.push((shifted & 0xff) as u64);
            shifted >>= 8;
            if shifted == 0 {
                break;
            }
        }
        for (offset, digit) in mean_digits.iter().enumerate() {
            assert_eq!(
                values[layout.slots.mean_digits + offset],
                Fp3::from_u64(*digit)
            );
        }
        for slot in layout.slots.mean_digits + mean_digits.len()
            ..layout.slots.mean_digits + widths.mean
        {
            assert_eq!(values[slot], Fp3::ZERO);
        }
        for slot in layout.slots.total..128 {
            assert_eq!(values[slot], Fp3::ZERO);
        }
    }

    #[test]
    fn lookup_plan_is_sized_from_the_parent_plan() {
        use crate::proof::lookup::mask_count;
        use crate::proof::lookup_messages::{commit_lookup, LookupRandomSource};

        let widths = NormRoundWidths {
            sum: 2,
            squares: 3,
            variance: 2,
            mean: 2,
            offset: 2,
        };
        let layout = NormRoundLayout::new(widths, 8, 5, 3).unwrap();
        let columns = widths.columns().unwrap();
        let round = NormRoundPlan {
            group: NormGroupPlan {
                shape: group_shape(columns, 1),
                columns,
                group: 1,
                oracle_base: 9000,
                openings: 1,
            },
            layout,
            input_bound: 8,
            table_real_len: 8,
        };
        let params = NormLookupParams {
            // The parent main code must cover the multiplicity-bit message
            // rows; a digit-heavy round needs one more tree level.
            code: zkie_engine::pcs::request::code::build_code_plan(7, 32, 4).unwrap(),
            mask_code: zkie_engine::pcs::request::code::build_code_plan(6, 32, 4).unwrap(),
            group: 7,
            bits_oracle: 9700,
            mask_oracle: 9701,
            bits_tree: 60701,
            mask_tree: 60700,
            leaf_class: 1,
            salt_bytes: 40,
            run: Digest([0x41; 32]),
            configuration: Digest([0x51; 32]),
            padding_byte: 0,
            table_identity: Digest([0x63; 32]),
        };
        let plan = NormLookupPlan::from_round(&round, 2, &params).unwrap();
        let digits = layout.digit_slots().unwrap();
        let (key_bits, mu) = lookup_parameters(digits.len() * 2).unwrap();
        assert_eq!(plan.lookup.key_bits, key_bits);
        assert_eq!(plan.lookup.mu, mu);
        assert_eq!(plan.lookup.table_log, 8);
        assert_eq!(plan.key.sources.len(), digits.len() * 2);
        let bit_count = plan.lookup.planes().unwrap() * plan.lookup.table_rows().unwrap();
        assert_eq!(plan.commit.bits_shape.fault(bit_count as u64, 1), None);
        assert!(plan.commit.bits_shape.main_rows <= (1u64 << params.code.log_n));
        let expected_rows = lookup_mask_rows(&plan.lookup, params.mask_code).unwrap();
        assert_eq!(plan.commit.mask_rows, expected_rows);
        let free = plan.commit.mask_rows * (params.mask_code.ell - 1)
            + plan.commit.mask_rows
            - 1;
        assert!(free >= mask_count(&plan.lookup).unwrap() as u64);
        assert_eq!(
            plan.commit.mask_tree.parts[0].mask_symbols,
            plan.commit.mask_rows
                * zkie_engine::pcs::request::field::E_DEGREE as u64
        );
        let digest = plan.digest();
        let mut changed = plan.clone();
        changed.key.padding_values = vec![Fp3::ONE];
        assert_ne!(changed.digest(), digest);

        // The constructed message plans commit from a synthetic bit witness.
        let bits = vec![Fp3::ZERO; bit_count];
        let mut source = LookupRandomSource::new(5);
        let messages = commit_lookup(
            &plan.commit,
            &bits,
            mask_count(&plan.lookup).unwrap(),
            &mut source,
        );
        assert!(messages.is_ok());

        let mut bad = params.clone();
        bad.bits_tree = bad.mask_tree;
        assert!(NormLookupPlan::from_round(&round, 2, &bad).is_err());
        bad = params.clone();
        bad.padding_byte = 256;
        assert!(NormLookupPlan::from_round(&round, 2, &bad).is_err());
        bad = params.clone();
        bad.table_identity = Digest::ZERO;
        assert!(NormLookupPlan::from_round(&round, 2, &bad).is_err());
        bad = params.clone();
        bad.code = zkie_engine::pcs::request::code::build_code_plan(6, 32, 4).unwrap();
        assert!(NormLookupPlan::from_round(&round, 2, &bad).is_err());
    }

    #[test]
    fn unit_valid_row_has_zero_offset_digits() {
        let core = NormCoreRow::from_values(&[5], 1).unwrap();
        assert_eq!(core.mean_remainder, 0);
        assert_eq!(core.mean_offset, 0);
        assert_eq!(core.mean_complement, 0);
        let widths = NormRoundWidths {
            sum: 1,
            squares: 1,
            variance: 1,
            mean: 1,
            offset: 1,
        };
        let layout = NormRoundLayout::new(widths, 1, 0, 0).unwrap();
        let columns = widths.columns().unwrap();
        let values = layout.values(&core, columns).unwrap();
        // d = 1 gives u = s = r = 0: the four offset/complement digit rows,
        // the squares row and the variance row are all zero. The sum and mean
        // magnitude rows still hold their values.
        for (start, width) in [
            (layout.slots.squares_digits, widths.squares),
            (layout.slots.variance_digits, widths.variance),
            (layout.slots.mean_offset_digits, widths.offset),
            (layout.slots.mean_complement_digits, widths.offset),
            (layout.slots.variance_offset_digits, widths.offset),
            (layout.slots.variance_complement_digits, widths.offset),
        ] {
            for slot in start..start + width {
                assert_eq!(values[slot], Fp3::ZERO, "a d=1 range digit is not zero");
            }
        }
        assert_eq!(values[layout.slots.mean_digits], Fp3::from_u64(5));
        assert_eq!(values[layout.slots.sum_digits], Fp3::from_u64(5));
    }

    #[test]
    fn round_digit_key_plan_covers_the_committed_digits() {
        use crate::proof::lookup::LookupPlan;

        let widths = NormRoundWidths {
            sum: 2,
            squares: 3,
            variance: 2,
            mean: 2,
            offset: 2,
        };
        let layout = NormRoundLayout::new(widths, 8, 5, 3).unwrap();
        let columns = widths.columns().unwrap();
        let round = NormRoundPlan {
            group: NormGroupPlan {
                shape: group_shape(columns, 1),
                columns,
                group: 1,
                oracle_base: 9000,
                openings: 1,
            },
            layout,
            input_bound: 8,
            table_real_len: 8,
        };
        let digits = layout.digit_slots().unwrap();
        assert_eq!(
            digits.len(),
            widths.sum
                + widths.squares
                + widths.variance
                + widths.mean
                + 4 * widths.offset
                + widths.raw()
                + 2 * RSQRT_RAW_REMAINDER_WIDTH
                + RSQRT_FINE_SLACK_WIDTH
                + widths.mid_q()
                + 1
                + RSQRT_MID_HIGH_WIDTH
                + widths.coarse_q()
                + 2
                + RSQRT_INDEX_WIDTH
                + RSQRT_INDEX_SLACK_WIDTH
        );
        for slot in [
            layout.slots.mean_remainder,
            layout.slots.mean_offset,
            layout.slots.mean_complement,
            layout.slots.variance_remainder,
            layout.slots.variance_offset,
            layout.slots.variance_complement,
        ] {
            assert!(!digits.contains(&slot), "a value slot is a lookup key");
        }
        for pair in digits.windows(2) {
            assert!(pair[0] < pair[1], "the digit slots are not in plan order");
        }

        let positions = 2u32;
        let (key_bits, mu) = lookup_parameters(digits.len() * positions as usize).unwrap();
        let key = round_digit_lookup_key(
            &round,
            positions,
            key_bits,
            0,
            Digest([0x63; 32]),
        )
        .unwrap();
        let plan = LookupPlan {
            key_bits,
            mu,
            table_log: 8,
            columns: 1,
        };
        plan.validate().unwrap();
        assert_eq!(key.sources.len(), digits.len() * positions as usize);
        let rows = usize::try_from(round.group.shape.main_rows).unwrap();
        let ell = usize::try_from(round.group.shape.code.ell).unwrap();
        for position in 0..positions {
            for (index, slot) in digits.iter().enumerate() {
                let source = &key.sources[position as usize * digits.len() + index];
                assert_eq!(source.oracle, 9000 + position as u64);
                assert_eq!(source.position, position);
                assert_eq!(source.row as usize, slot % rows);
                assert_eq!(source.place as usize, slot / rows);
                assert_eq!(source.coefficient, Fp3::ONE);
                assert_eq!(source.offset, Fp3::ZERO);
            }
        }

        // The key vector reads the digit cells of the real message layout.
        let digit_of = |position: usize, index: usize| {
            Fp3::from_u64(((position * 37 + index * 11 + 3) % 256) as u64)
        };
        let messages = (0..positions as usize)
            .map(|position| {
                let mut message = vec![vec![Fp3::ZERO; ell]; rows];
                for (index, slot) in digits.iter().enumerate() {
                    message[slot % rows][slot / rows] = digit_of(position, index);
                }
                message
            })
            .collect::<Vec<_>>();
        let keys = key.read(&plan, &messages, Fp3::ONE).unwrap();
        let key_count = plan.key_count().unwrap();
        assert_eq!(keys.len(), key_count);
        for position in 0..positions as usize {
            for index in 0..digits.len() {
                assert_eq!(
                    keys[position * digits.len() + index],
                    digit_of(position, index)
                );
            }
        }
        for extra in keys.iter().skip(key.sources.len()) {
            assert_eq!(*extra, Fp3::ZERO);
        }
        assert!(round_digit_lookup_key(
            &round,
            positions,
            key_bits,
            256,
            Digest([0x63; 32]),
        )
        .is_err());
        assert!(round_digit_lookup_key(
            &round,
            positions,
            key_bits,
            0,
            Digest::ZERO,
        )
        .is_err());
        assert!(round_digit_lookup_key(&round, positions, 3, 0, Digest([0x63; 32]))
            .is_err());
    }

    #[test]
    fn offset_rows_cover_zero_negative_and_the_fixed_offset() {
        let widths = real_widths();
        let layout = NormRoundLayout::new(widths, 768, SUM_OFFSET, MEAN_OFFSET).unwrap();
        for (row, shifted_sum) in [
            (vec![0i64, 0, 0, 0], SUM_OFFSET as i128),
            (vec![-1i64, 0, 0, 0], SUM_OFFSET as i128 - 1),
            (vec![3i64, -2, 5, 1], SUM_OFFSET as i128 + 7),
        ] {
            let core = NormCoreRow::from_values(&row, 4).unwrap();
            let values = layout.values(&core, widths.columns().unwrap()).unwrap();
            let mut rest = shifted_sum;
            for index in 0..widths.sum {
                let digit = (rest & 0xff) as u64;
                assert_eq!(values[layout.slots.sum_digits + index], Fp3::from_u64(digit));
                rest >>= 8;
            }
            assert_eq!(rest, 0);
            let shifted_mean = core.mean as i128 + MEAN_OFFSET as i128;
            let mut rest = shifted_mean;
            for index in 0..widths.mean {
                let digit = (rest & 0xff) as u64;
                assert_eq!(
                    values[layout.slots.mean_digits + index],
                    Fp3::from_u64(digit)
                );
                rest >>= 8;
            }
            assert_eq!(rest, 0);
        }
    }

    #[test]
    fn rstd_plan_reads_the_committed_index_and_value_slots() {
        use crate::proof::lookup_table::{TableConfig, TableEncoding, TABLE_ENCODING_VERSION};

        let widths = NormRoundWidths {
            sum: 1,
            squares: 1,
            variance: 1,
            mean: 1,
            offset: 1,
        };
        let layout = NormRoundLayout::new(widths, 4, 8, 4).unwrap();
        let columns = widths.columns().unwrap();
        let round = NormRoundPlan {
            group: NormGroupPlan {
                shape: group_shape(columns, 1),
                columns,
                group: 1,
                oracle_base: 9000,
                openings: 1,
            },
            layout,
            input_bound: 8,
            table_real_len: 8,
        };
        let table = TableConfig {
            identity: Digest([0x71; 32]),
            content_root: Digest([0x72; 32]),
            real_len: 8,
            encoding: TableEncoding::I32Le,
            version: TABLE_ENCODING_VERSION,
        };
        let params = NormRstdParams {
            code: build_code_plan(6, 32, 4).unwrap(),
            mask_code: build_code_plan(6, 32, 4).unwrap(),
            group: 7,
            bits_oracle: 9800,
            mask_oracle: 9801,
            bits_tree: 60800,
            mask_tree: 60801,
            leaf_class: 1,
            salt_bytes: 40,
            run: Digest([0x73; 32]),
            configuration: Digest([0x74; 32]),
        };
        let plan = NormRstdPlan::from_round(&round, 2, table, Fp3::from_u64(100), &params)
            .unwrap();
        assert_eq!(plan.lookup.columns, 2);
        assert_eq!(plan.lookup.table_log, 3);
        assert_eq!(plan.key.sources.len(), 4);
        assert_eq!(plan.key.sources[0].column, 0);
        assert_eq!(plan.key.sources[1].column, 1);
        assert_eq!(plan.key.sources[2].column, 0);
        assert_eq!(plan.key.sources[3].column, 1);
        assert_eq!(
            plan.key.padding_values,
            vec![Fp3::ZERO, Fp3::from_u64(100)]
        );
        assert_eq!(plan.key.table_identity, table.identity);
        plan.validate().unwrap();

        let mut bad = table;
        bad.identity = Digest::ZERO;
        assert!(NormRstdPlan::from_round(&round, 2, bad, Fp3::ZERO, &params).is_err());
        let mut bad = plan.clone();
        bad.lookup.columns = 1;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn rsqrt_slots_take_the_segmentation_values() {
        use crate::proof::hidden_norm::RsqrtRow;

        let widths = real_widths();
        let layout = NormRoundLayout::new(widths, 768, SUM_OFFSET, MEAN_OFFSET).unwrap();
        let core = NormCoreRow::from_values(&[3, -2, 5, 1], 4).unwrap();
        let row = RsqrtRow::from_variance(core.variance, 7_340_032, 1_234).unwrap();
        let columns = widths.columns().unwrap();
        let values = layout
            .values_with_rsqrt(&core, &row, columns)
            .unwrap();
        assert_eq!(
            values[layout.slots.fine_selector],
            Fp3::from_u64(row.fine as u64)
        );
        assert_eq!(
            values[layout.slots.mid_selector],
            Fp3::from_u64(row.mid as u64)
        );
        assert_eq!(
            values[layout.slots.index_value],
            fp3_signed(row.index as i128).unwrap()
        );
        assert_eq!(
            values[layout.slots.rstd],
            fp3_signed(row.rstd as i128).unwrap()
        );
        for (index, digit) in row.raw_digits.iter().enumerate() {
            assert_eq!(
                values[layout.slots.raw_digits + index],
                Fp3::from_u64(*digit as u64)
            );
        }
        for index in row.raw_digits.len()..widths.raw() {
            assert_eq!(values[layout.slots.raw_digits + index], Fp3::ZERO);
        }
        for (index, digit) in row.index_digits.iter().enumerate() {
            assert_eq!(
                values[layout.slots.index_digits + index],
                Fp3::from_u64(*digit as u64)
            );
        }

        let mut narrow = widths;
        narrow.variance = 3;
        let layout = NormRoundLayout::new(narrow, 768, SUM_OFFSET, MEAN_OFFSET).unwrap();
        let wide = RsqrtRow::from_variance(1 << 30, 7_340_032, 1).unwrap();
        assert!(
            layout
                .values_with_rsqrt(&core, &wide, narrow.columns().unwrap())
                .is_err(),
            "a raw index wider than its plan block must fail closed"
        );
    }

    #[test]
    fn rsqrt_relation_maps_sum_to_zero_at_the_segment_boundaries() {
        use crate::proof::hidden_norm::RsqrtRow;

        let widths = NormRoundWidths {
            sum: 1,
            squares: 1,
            variance: 1,
            mean: 1,
            offset: 1,
        };
        let layout = NormRoundLayout::new(widths, 2, 0, 0).unwrap();
        let round = NormRoundPlan {
            group: NormGroupPlan {
                shape: group_shape(64, 1),
                columns: 64,
                group: 1,
                oracle_base: 1,
                openings: 1,
            },
            layout,
            input_bound: 2,
            table_real_len: 8_388_608,
        };
        let core = NormCoreRow::from_values(&[1, 3], 2).unwrap();
        let rows = [1i64, 1i64, 1i64];
        let mut variances = vec![0i64, 1, 1 << 18, 1 << 30];
        for raw in [
            RSQRT_FINE - 1,
            RSQRT_FINE,
            RSQRT_COARSE_END - 1,
            RSQRT_COARSE_END,
            RSQRT_COARSE_END + 65_536,
        ] {
            variances.push(raw * RSQRT_SHIFT - RSQRT_HALF);
        }
        for variance in variances {
            let rsqrt = RsqrtRow::from_variance(variance, round.table_real_len, 7).unwrap();
            for relation in 16..=25 {
                let (left, right) =
                    rel_map(&core, &rows, 2, 8, relation, Some(&round), Some(&rsqrt)).unwrap();
                let dot = left
                    .iter()
                    .zip(right.iter())
                    .fold(Fp3::ZERO, |sum, (a, b)| sum.add(&a.mul(b)));
                assert_eq!(
                    dot, Fp3::ZERO,
                    "relation {relation} variance {variance} does not sum to zero"
                );
            }
        }
    }

    #[test]
    fn round_widths_and_digit_rows_fail_closed() {
        let mut widths = real_widths();
        widths.squares = 8;
        assert!(NormRoundLayout::new(widths, 768, SUM_OFFSET, MEAN_OFFSET).is_err());
        let mut widths = real_widths();
        widths.offset = 1;
        assert!(NormRoundLayout::new(widths, 768, SUM_OFFSET, MEAN_OFFSET).is_err());
        let mut widths = real_widths();
        widths.squares = 6;
        let layout = NormRoundLayout::new(widths, 768, SUM_OFFSET, MEAN_OFFSET).unwrap();
        let core = NormCoreRow::from_values(&[22_855_985, -22_855_985], 2).unwrap();
        assert_eq!(core.squares_digits.len(), 7);
        assert!(layout.values(&core, widths.columns().unwrap()).is_err());
        let layout = NormRoundLayout::new(real_widths(), 768, SUM_OFFSET, MEAN_OFFSET).unwrap();
        assert!(layout.values(&core, 32).is_err());
    }

    #[test]
    fn real_bounds_stay_below_the_modulus_half() {
        let bounds =
            norm_no_wrap(22_855_985, 768, &real_widths(), SUM_OFFSET, MEAN_OFFSET).unwrap();
        assert_eq!(bounds.relation_sum, 1_117_098_578_687);
        assert_eq!(bounds.relation_squares, 1_282_417_637_556_158_463);
        assert_eq!(bounds.relation_mean, 1_112_480_415_743);
        assert_eq!(bounds.relation_variance, 288_230_376_151_711_743);
        assert_eq!(bounds.reconstruction, 72_057_594_037_927_936);
        assert_eq!(bounds.offsets, 131_838);
        assert_eq!(bounds.remainder, 2_304);
        let mut widths = real_widths();
        widths.mean = 4;
        assert!(norm_no_wrap(22_855_985, 768, &widths, SUM_OFFSET, MEAN_OFFSET).is_err());
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
            lookup: None,
            rstd: None,
            params: None,
            output: None,
            mask: None,
        };
        let message = NormFirstMessages {
            x: first.x.clone(),
            sum: first.sum.clone(),
            squares: first.squares.clone(),
            mean: first.mean.clone(),
            variance: first.variance.clone(),
            round: None,
            mask: None,
            lookup: None,
            rstd: None,
            params: None,
            output: None,
            tail: None,
        };
        (plan, message, second)
    }

    #[test]
    fn parameter_tables_are_bound_in_the_plan() {
        use crate::proof::norm_param_table::{
            NormParamKind, NormParamTableIdentity, NormParamTablesPlan,
            NORM_PARAM_TABLE_VERSION,
        };
        use zkie_engine::pcs::request::code::build_code_plan;
        use zkie_engine::pcs::request::field::Fp3;
        use zkie_engine::pcs::request::params::LEAF_SALT_MIN_BYTES;
        use zkie_engine::pcs::request::table::{
            IndexReader, IndexRecord, IndexShape, TableDistances,
        };

        let (mut plan, _, _) = integrated_fixture();
        let reader = || IndexReader {
            shape: IndexShape {
                code: build_code_plan(4, 7, 2).unwrap(),
                v: 1,
                answers: 1,
                salt_bytes: LEAF_SALT_MIN_BYTES,
                distances: TableDistances {
                    radius: 0,
                    later_radius: 0,
                    m_once: 0,
                    m_later: 0,
                },
            },
            configuration: Digest([0x21; 32]),
            root: Digest([0x22; 32]),
            record: IndexRecord {
                y: vec![Fp3::ZERO; 2],
            },
        };
        let identity = |kind: NormParamKind, row: u64| NormParamTableIdentity {
            version: NORM_PARAM_TABLE_VERSION,
            kind,
            model_identity: Digest([0x23; 32]),
            asset_table_identity: Digest([0x24; 32]),
            reader: reader(),
            row,
            width: plan.columns,
            valid_min: -128,
            valid_max: 127,
            trusted_i32_mapping: true,
        };
        let digest = plan.digest();
        plan.params = Some(NormParamTablesPlan {
            gain: identity(NormParamKind::Gain, 0),
            bias: identity(NormParamKind::Bias, 1),
            column_draws: 2,
            seed_budget: 1,
            gain_claim: None,
            bias_claim: None,
        });
        plan.validate().unwrap();
        assert_ne!(plan.digest(), digest);

        let mut changed = plan.clone();
        changed.params.as_mut().unwrap().gain.width = 2;
        assert!(changed.validate().is_err());

        let mut changed = plan.clone();
        changed.params.as_mut().unwrap().bias.kind = NormParamKind::Gain;
        assert!(changed.validate().is_err());

        let mut changed = plan;
        changed.params.as_mut().unwrap().bias.row = 0;
        assert!(changed.validate().is_err());
    }

    fn claim_gu(
        kind: crate::proof::norm_param_table::NormParamKind,
        shape: zkie_engine::pcs::request::table::IndexShape,
    ) -> crate::proof::norm_param_table::NormParamClaimPlan {
        use crate::proof::norm_param_table::NormParamClaimPlan;
        use zkie_engine::pcs::request::field::E_DEGREE;
        use zkie_engine::pcs::request::gu::GuOracleShape;
        use zkie_engine::pcs::request::runtime_tree::{RuntimeTreePart, RuntimeTreePlan};
        use zkie_engine::pcs::request::params::LEAF_SALT_MIN_BYTES;

        let main_rows = 2u64;
        let group_code = zkie_engine::pcs::request::code::build_code_plan(7, 32, 4).unwrap();
        NormParamClaimPlan {
            kind,
            gu: zkie_engine::pcs::request::gu::GuShape {
                oracle: 0x8800 + kind.encode(),
                group: 0x42,
                tree: RuntimeTreePlan {
                    tree_number: 0x1234,
                    leaf_class: 1,
                    log_n: group_code.log_n,
                    log_n_zk: group_code.log_n,
                    parts: vec![RuntimeTreePart {
                        main_symbols: main_rows * E_DEGREE as u64,
                        mask_symbols: 0,
                    }],
                    salt_bytes: LEAF_SALT_MIN_BYTES,
                    run: Digest([0x71; 32]),
                    configuration: Digest([0x72; 32]),
                },
                part: 0,
                rows_high: true,
                u: GuOracleShape {
                    code: group_code,
                    main_rows,
                    main_deg: E_DEGREE as u32,
                    mask_rows: 0,
                    mask_code: None,
                    answers: 1,
                },
                index: shape,
            },
            column_draws: 2,
        }
    }

    #[test]
    fn output_tail_plan_binds_the_groups_and_the_runtime_parameters() {
        use crate::proof::norm_param_table::{
            NormParamKind, NormParamTableIdentity, NormParamTablesPlan,
            NORM_PARAM_TABLE_VERSION,
        };
        use zkie_engine::pcs::request::code::build_code_plan;
        use zkie_engine::pcs::request::field::Fp3;
        use zkie_engine::pcs::request::params::LEAF_SALT_MIN_BYTES;
        use zkie_engine::pcs::request::table::{
            IndexReader, IndexRecord, IndexShape, TableDistances,
        };

        let (mut plan, _, _) = integrated_fixture();
        let shape = || IndexShape {
            code: build_code_plan(4, 7, 2).unwrap(),
            v: 0,
            answers: 1,
            salt_bytes: LEAF_SALT_MIN_BYTES,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        };
        let reader = |root: u8, configuration: u8| IndexReader {
            shape: shape(),
            configuration: Digest([configuration; 32]),
            root: Digest([root; 32]),
            record: IndexRecord {
                y: vec![Fp3::ZERO; 1],
            },
        };
        let identity = |kind: NormParamKind, reader: IndexReader| NormParamTableIdentity {
            version: NORM_PARAM_TABLE_VERSION,
            kind,
            model_identity: Digest([0x33; 32]),
            asset_table_identity: Digest([0x34; 32]),
            reader,
            row: 0,
            width: plan.columns,
            valid_min: -128,
            valid_max: 127,
            trusted_i32_mapping: true,
        };
        plan.params = Some(NormParamTablesPlan {
            gain: identity(NormParamKind::Gain, reader(0x32, 0x31)),
            bias: identity(NormParamKind::Bias, reader(0x37, 0x38)),
            column_draws: 2,
            seed_budget: 1,
            gain_claim: None,
            bias_claim: None,
        });
        let group = |tree: u64, oracle: u64| NormGroupPlan {
            shape: WShape::new(
                plan.columns,
                code(),
                1,
                2,
                LEAF_SALT_MIN_BYTES,
                tree,
                1,
                Digest([0x41; 32]),
                Digest([0x51; 32]),
            )
            .unwrap(),
            columns: plan.columns,
            group: 7,
            oracle_base: oracle,
            openings: 1,
        };
        let output = NormOutputPlan {
            broadcast: group(700, 7000),
            product: group(701, 7100),
            image: group(702, 7200),
            out: group(703, 7300),
            shift: 4,
            clamp: Some((-5, 5)),
            tail: None,
        };
        let digest = plan.digest();
        plan.output = Some(output);
        plan.validate().unwrap();
        assert_ne!(plan.digest(), digest);
        let round_widths = NormRoundWidths {
            sum: 2,
            squares: 2,
            variance: 2,
            mean: 2,
            offset: 1,
        };
        let round_layout =
            NormRoundLayout::new(round_widths, plan.shape.valid, 64, 64).unwrap();
        let round_columns = round_widths.columns().unwrap();
        plan.round = Some(NormRoundPlan {
            group: NormGroupPlan {
                shape: group_shape(round_columns, 6),
                columns: round_columns,
                group: plan.variance.group,
                oracle_base: 9500,
                openings: 1,
            },
            layout: round_layout,
            input_bound: 1000,
            table_real_len: 8,
        });
        plan.validate().unwrap();

        let mut changed = plan.clone();
        changed.output.as_mut().unwrap().shift = 63;
        assert!(changed.validate().is_err());

        let mut changed = plan.clone();
        changed.output.as_mut().unwrap().clamp = Some((6, 5));
        assert!(changed.validate().is_err());

        let mut changed = plan.clone();
        changed.output.as_mut().unwrap().broadcast.columns = 2;
        assert!(changed.validate().is_err());

        let mut changed = plan.clone();
        let base = changed.output.as_ref().unwrap().broadcast.oracle_base;
        changed.output.as_mut().unwrap().product.oracle_base = base;
        assert!(changed.validate().is_err());

        let widths = NormOutputTailWidths {
            quotient: 3,
            remainder: 2,
            doubled_remainder: 2,
            tie_slack: 2,
            shifted: 3,
            pre_clamp: 3,
        };
        let clamp_widths = NormClampWidths {
            low: 4,
            high: 4,
            low_strict: 4,
            low_nonstrict: 4,
            high_strict: 4,
            high_nonstrict: 4,
        };
        let tail_columns = 256u64;
        let tail_group = NormGroupPlan {
            shape: WShape::new(
                tail_columns,
                code(),
                1,
                2,
                LEAF_SALT_MIN_BYTES,
                704,
                1,
                Digest([0x42; 32]),
                Digest([0x52; 32]),
            )
            .unwrap(),
            columns: tail_columns,
            group: 8,
            oracle_base: 7400,
            openings: 1,
        };
        let mut with_tail = plan.clone();
        with_tail.output.as_mut().unwrap().tail = Some(NormOutputTailPlan {
            group: tail_group.clone(),
            widths,
            clamp: Some(clamp_widths),
        });
        with_tail.validate().unwrap();
        // The plan-level no-wrap derivation rejects a plan whose trusted
        // input bound and parameter range push the normalized image past the
        // safe magnitude.
        let mut huge = with_tail.clone();
        huge.round.as_mut().unwrap().input_bound = 1u64 << 42;
        huge.params.as_mut().unwrap().gain.valid_max = i64::MAX;
        assert!(huge.validate().is_err());
        assert_eq!(
            with_tail
                .output
                .as_ref()
                .unwrap()
                .tail
                .as_ref()
                .unwrap()
                .columns(plan.columns)
                .unwrap(),
            256
        );

        let mut changed = with_tail.clone();
        changed.output.as_mut().unwrap().tail.as_mut().unwrap().group.columns = 16;
        assert!(changed.validate().is_err());

        let mut changed = with_tail.clone();
        changed.output.as_mut().unwrap().shift = 30;
        assert!(changed.validate().is_err());

        let mut changed = with_tail.clone();
        let base = changed.output.as_ref().unwrap().out.oracle_base;
        changed
            .output
            .as_mut()
            .unwrap()
            .tail
            .as_mut()
            .unwrap()
            .group
            .oracle_base = base;
        assert!(changed.validate().is_err());

        let mut changed = with_tail.clone();
        changed
            .output
            .as_mut()
            .unwrap()
            .tail
            .as_mut()
            .unwrap()
            .widths
            .quotient = 8;
        assert!(changed.validate().is_err());

        assert_eq!(widths.slots().unwrap(), 16);
    }

    #[test]
    fn slack_digit_lookup_plan_addresses_the_tail_cells() {
        let (plan, _, _) = integrated_fixture();
        let widths = NormOutputTailWidths {
            quotient: 3,
            remainder: 2,
            doubled_remainder: 2,
            tie_slack: 2,
            shifted: 3,
            pre_clamp: 3,
        };
        let clamp_widths = NormClampWidths {
            low: 4,
            high: 4,
            low_strict: 4,
            low_nonstrict: 4,
            high_strict: 4,
            high_nonstrict: 4,
        };
        let tail_columns = 256u64;
        let tail_group = NormGroupPlan {
            shape: WShape::new(
                tail_columns,
                code(),
                1,
                2,
                zkie_engine::pcs::request::params::LEAF_SALT_MIN_BYTES,
                720,
                1,
                Digest([0x44; 32]),
                Digest([0x54; 32]),
            )
            .unwrap(),
            columns: tail_columns,
            group: 10,
            oracle_base: 7600,
            openings: 1,
        };
        let tail = NormOutputTailPlan {
            group: tail_group,
            widths,
            clamp: Some(clamp_widths),
        };
        let identity = Digest([0x55; 32]);
        let positions = plan.shape.positions;
        let count = tail_digit_count(&tail, positions, plan.columns).unwrap();
        let (key_bits, mu) = lookup_parameters(count).unwrap();
        let lookup = crate::proof::lookup::LookupPlan {
            key_bits,
            mu,
            table_log: 8,
            columns: 1,
        };
        lookup.validate().unwrap();
        let key =
            tail_digit_lookup_key(&tail, positions, plan.columns, key_bits, 0, identity)
                .unwrap();
        let slack_digits = tail_digit_groups(&tail)
            .unwrap()
            .iter()
            .map(|(_, width)| *width)
            .sum::<usize>();
        let per_position = plan.columns as usize * slack_digits;
        assert_eq!(key.sources.len(), positions as usize * per_position);
        assert_eq!(lookup.columns, 1);
        assert_eq!(lookup.table_log, 8);
        assert_eq!(
            lookup.key_count().unwrap(),
            (key.sources.len() as u64).next_power_of_two() as usize
        );
        for source in &key.sources {
            assert_eq!(source.column, 0);
            assert_eq!(
                source.oracle,
                tail.group.oracle_base + source.position as u64
            );
        }
        assert_eq!(key.padding_values, vec![Fp3::ZERO]);
        key.validate(&lookup).unwrap();

        let mut without_clamp = tail.clone();
        without_clamp.clamp = None;
        assert!(tail_digit_count(&without_clamp, positions, plan.columns).is_err());
    }

    #[test]
    fn output_tail_relations_hold_on_the_exact_rows() {
        use crate::proof::hidden_norm::{NormCoreRow, NormOutputRow};
        use zkie_engine::pcs::request::params::LEAF_SALT_MIN_BYTES;

        let (plan, _, _) = integrated_fixture();
        let columns = plan.columns;
        let widths = NormOutputTailWidths {
            quotient: 3,
            remainder: 2,
            doubled_remainder: 2,
            tie_slack: 2,
            shifted: 3,
            pre_clamp: 3,
        };
        let tail_columns = widths.columns(columns).unwrap();
        let group = NormGroupPlan {
            shape: WShape::new(
                tail_columns,
                code(),
                1,
                2,
                LEAF_SALT_MIN_BYTES,
                710,
                1,
                Digest([0x43; 32]),
                Digest([0x53; 32]),
            )
            .unwrap(),
            columns: tail_columns,
            group: 9,
            oracle_base: 7500,
            openings: 1,
        };
        let tail_plan = NormOutputTailPlan {
            group,
            widths,
            clamp: None,
        };
        let weight = vec![2i64, -1, 3, 1];
        let bias = vec![1i64, 2, -1, 2];
        let inner = |left: &[Fp3], right: &[Fp3]| -> Fp3 {
            let mut sum = Fp3::ZERO;
            for (left, right) in left.iter().zip(right) {
                sum = sum.add(&left.mul(right));
            }
            sum
        };
        for values in [vec![3i64, -2], vec![-4, 7]] {
            let core = NormCoreRow::from_values(&values, 2).unwrap();
            let mut padded = values.clone();
            padded.resize(columns as usize, 0);
            let row = NormOutputRow::from_core(
                &core,
                &padded,
                100,
                &weight,
                &bias,
                2,
                3,
                None,
            )
            .unwrap();
            for relation in 0..NORM_OUTPUT_TAIL_RELATIONS {
                let (left, right) = output_rel_map(&tail_plan, &row, columns, 3, relation).unwrap();
                assert_eq!(inner(&left, &right), Fp3::ZERO, "relation {relation}");
            }
            let mut changed = row.clone();
            changed.image[0] += 1;
            let (left, right) = output_rel_map(&tail_plan, &changed, columns, 3, 0).unwrap();
            assert_ne!(inner(&left, &right), Fp3::ZERO);
            let mut changed = row.clone();
            changed.selector[1] = !changed.selector[1];
            let (left, right) = output_rel_map(&tail_plan, &changed, columns, 3, 1).unwrap();
            assert_ne!(inner(&left, &right), Fp3::ZERO);
            let mut changed = row.clone();
            changed.remainder[0] += 1;
            let (left, right) = output_rel_map(&tail_plan, &changed, columns, 3, 0).unwrap();
            assert_ne!(inner(&left, &right), Fp3::ZERO);
        }
        assert_eq!(output_tail_variables(columns).unwrap(), 7);
        let row = {
            let core = NormCoreRow::from_values(&[1, 2], 2).unwrap();
            NormOutputRow::from_core(&core, &[1, 2, 0, 0], 100, &weight, &bias, 2, 3, None)
                .unwrap()
        };
        assert!(output_rel_map(&tail_plan, &row, columns, 3, NORM_OUTPUT_TAIL_RELATIONS).is_err());
    }

    #[test]
    fn parameter_table_claims_round_trip_and_are_plan_gated() {
        use crate::proof::norm_param_table::{
            NormParamKind, NormParamTableIdentity, NormParamTablesPlan,
            NORM_PARAM_TABLE_VERSION,
        };
        use zkie_core::common::field::{PrimeCharacteristicRing, PrimeField64};
        use zkie_engine::pcs::request::code::build_code_plan;
        use zkie_engine::pcs::request::field::Fp;
        use zkie_engine::pcs::request::params::LEAF_SALT_MIN_BYTES;
        use zkie_engine::pcs::request::table::{
            ColumnOpening, IndexReader, IndexRecord, IndexShape, TableDistances,
        };

        let (mut plan, message, _) = integrated_fixture();
        let shape = || IndexShape {
            code: build_code_plan(4, 7, 2).unwrap(),
            v: 0,
            answers: 1,
            salt_bytes: LEAF_SALT_MIN_BYTES,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        };
        let reader = |root: u8, configuration: u8| IndexReader {
            shape: shape(),
            configuration: Digest([configuration; 32]),
            root: Digest([root; 32]),
            record: IndexRecord {
                y: vec![Fp3::ZERO; 1],
            },
        };
        let identity = |kind: NormParamKind, reader: IndexReader| NormParamTableIdentity {
            version: NORM_PARAM_TABLE_VERSION,
            kind,
            model_identity: Digest([0x33; 32]),
            asset_table_identity: Digest([0x34; 32]),
            reader,
            row: 0,
            width: plan.columns,
            valid_min: -128,
            valid_max: 127,
            trusted_i32_mapping: true,
        };
        let gain = identity(NormParamKind::Gain, reader(0x32, 0x31));
        let bias = identity(NormParamKind::Bias, reader(0x37, 0x38));
        plan.params = Some(NormParamTablesPlan {
            gain: gain.clone(),
            bias: bias.clone(),
            column_draws: 2,
            seed_budget: 1,
            gain_claim: Some(claim_gu(NormParamKind::Gain, gain.reader.shape)),
            bias_claim: Some(claim_gu(NormParamKind::Bias, bias.reader.shape)),
        });
        plan.validate().unwrap();
        let opening = |position: u64| ColumnOpening {
            position,
            salt: vec![0u8; LEAF_SALT_MIN_BYTES],
            symbols: vec![Fp::from_u64(0); 1],
            path: vec![Digest::ZERO; 4],
        };
        let claim_message = || NormParamClaimMessage {
            root: Digest([0x77; 32]),
            seeds: vec![SampleSeed {
                z: vec![Fp3::ZERO; 6],
                beta: Fp3::ONE,
            }],
            answers: vec![Fp3::ZERO],
            openings: vec![opening(1), opening(3)],
        };
        let params = NormParamMessage {
            gain: claim_message(),
            bias: Some(claim_message()),
        };
        params.check(plan.params.as_ref().unwrap()).unwrap();

        let mut first = message;
        first.params = Some(params.clone());
        let bytes = first.encode().unwrap();
        let decoded = NormFirstMessages::decode(&plan, &bytes).unwrap();
        assert_eq!(decoded.params, Some(params.clone()));

        let (plain_plan, _, _) = integrated_fixture();
        assert!(NormFirstMessages::decode(&plain_plan, &bytes).is_err());

        let mut swapped = params.clone();
        swapped.gain.openings.swap(0, 1);
        let encoded = swapped.gain.encode().unwrap();
        assert!(NormParamClaimMessage::decode(
            plan.params.as_ref().unwrap().gain_claim.as_ref().unwrap(),
            &plan.params.as_ref().unwrap().gain,
            &encoded
        )
        .is_err());

        let mut packed = plan.clone();
        packed.params.as_mut().unwrap().bias.reader = packed.params.as_ref().unwrap().gain.reader.clone();
        packed.params.as_mut().unwrap().bias.row = 1;
        assert!(packed.validate().is_err());

        let mut missing = params;
        missing.bias = None;
        assert!(missing.check(plan.params.as_ref().unwrap()).is_err());
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
        let layout = NormRoundLayout::new(widths, plan.shape.valid, 64, 64).unwrap();
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
            table_real_len: 8,
        });
        plan.validate().unwrap();
        let with_round = plan.digest();

        let mut wide = plan.clone();
        let widths = NormRoundWidths {
            mean: 4,
            ..widths
        };
        let layout = NormRoundLayout::new(widths, plan.shape.valid, 64, 64).unwrap();
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
            table_real_len: 8,
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
            pointwise: 0,
            pointwise_hiding: norm_pointwise_masks(3).unwrap().0,
            pointwise_shown: norm_pointwise_masks(3).unwrap().1,
            tail: 0,
            tail_hiding: norm_tail_mask_masks(4).unwrap().0,
            tail_shown: norm_tail_mask_masks(4).unwrap().1,
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
