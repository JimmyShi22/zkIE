//! The multi-stage hidden-session IOPP loop.
//!
//! Stage 0 is a BAT/FOLD result supplied by the caller. Every later trusted
//! stage runs the multi-row code switch over the previous stage's rows, then a
//! stage FOLD that produces the next stage's rows, except the last stage which
//! ends in BASE. The stage list, every switch target, every switch tree plan
//! and every stage FOLD shape, layout and mask tree are derived from the
//! trusted plan; proof bytes carry only the per-switch messages, the stage
//! FOLD messages and the final BASE artifact.
//!
//! The verifier derives the same plan, checks the proof's switch targets
//! against it before using any value, and replays the transcript challenges.
//! The hidden-session adapter supplies the initial member trees and the fresh
//! mask tree for the first switch, and the previous switch's committed message
//! for later switches.

use std::sync::Arc;

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::chain::{Chain, KIND_ALGEBRAIC, KIND_EPSILON};
use zkie_engine::pcs::request::code::CodeParams;
use zkie_engine::pcs::request::cs_stage::{
    switch_tree_plan_multi_at, Cut, CutPart, SwitchTarget, SwitchedClaims, SwitchedRows,
};
use zkie_engine::pcs::request::field::{Fp3, E_DEGREE};
use zkie_engine::pcs::request::fold::{
    fold_mask_make, fold_mask_make_tail, fold_mask_notice, fold_mask_weights,
    fold_output, fold_start_prove, fold_start_verify, fold_mask_shape_fault, ByteMaskCoins,
    FoldGroupLayout, FoldInputRow, FoldMemberLayout, FoldOutput, FoldProverSession,
    FoldRoundVerifier, FoldShape, FoldStartMessage, MaskOracle, OracleMetadata, RuntimeOracles,
    VerifiedGroup, VerifiedGroups, VerifiedStatement,
};
use zkie_engine::pcs::request::gu::GuOracleShape;
use zkie_engine::pcs::request::proximity::SingleRows;
use zkie_engine::pcs::request::read::unique_sorted;
use zkie_engine::pcs::request::relation::foldeq_weights;
use zkie_engine::pcs::request::runtime_tree::{
    MaskTree, MaskTreePlan, RuntimeTree, RuntimeTreeOpening, RuntimeTreePlan,
};
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};
use zkie_core::common::codec::DecodeLimits;

use crate::proof::base_final_session::{fold_identity, FoldBasis, FoldMember};
use crate::proof::base_last_session::{
    encode_last_base_proof, verify_last_base_proof_with_positions, LastBaseProver,
    LastBaseRandomness, LastBaseSession,
};
use crate::proof::cs_multi_session::{CsMultiProver, CsMultiSession, OldMaskSource};
use crate::proof::fold_handoff::{FoldHandoff, FoldHandoffAnchor};
use crate::proof::hidden_send_session::{
    get_code_public, get_fold_start, get_runtime_opening, iopp_stage_fault, put_code_public,
    put_fold_start, put_runtime_opening, IoppStagePlan,
};

/// The leaf class of the mask-only trees this loop commits. It is distinct
/// from the switched message class and from the caller's BASE star class.
pub const IOPP_MASK_LEAF_CLASS: u32 = 3;

/// The leaf class of a switched oracle message tree.
pub const IOPP_SWITCH_LEAF_CLASS: u32 = 7;

/// The decode bound of one runtime opening: positions, columns, symbols and
/// supplied digests each decode within this count. The session still checks
/// every opening against the trusted tree and drawn positions.
const IOPP_OPENING_DECODE_BOUND: usize = 1 << 20;

/// The decode bound of one switch's old-member opening count. The session
/// rejects a count that disagrees with the trusted group layout.
const IOPP_MEMBER_DECODE_BOUND: usize = 1 << 12;

/// One stage's trusted code and row-variable shape.
///
/// Entry 0 describes the stage 0 FOLD shape; entries 1.. are switch stages.
/// A switch at plan index `i` uses entry `i` as its target and reads the old
/// rows drawn by entry `i - 1`, so entry `i - 1` supplies that switch's
/// position count and work bits. The last entry has `k = h = 0`; its position
/// count is the BASE opening count.
#[derive(Clone, Debug)]
pub struct IoppStage {
    pub plan: IoppStagePlan,
    pub target: Option<SwitchTarget>,
    pub tree: Option<RuntimeTreePlan>,
    pub fold: Option<IoppStageFold>,
}

/// One non-last stage's trusted FOLD plan: the shape, the group layout and the
/// fresh mask tree that the following stage FOLD commits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IoppStageFold {
    pub shape: FoldShape,
    pub layout: FoldGroupLayout,
    pub mask_tree: MaskTreePlan,
    pub group: u64,
    pub member: u64,
    pub fresh: u64,
}

/// The identifiers one non-last stage FOLD binds in its own registry.
///
/// The identifiers are allocated by the trusted plan from the bound identifier
/// set in canonical (chain, stage, role) order; the loop checks that the
/// sequence carries no duplicate and that stage 0's identifiers are not reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IoppStageFoldIds {
    pub group: u64,
    pub member: u64,
    pub fresh: u64,
}

/// The public derivation values of one stage's committed trees.
///
/// The switch tree and the following FOLD mask tree of stage `i` carry these
/// values, so every stage's public labels are derived from the trusted stage
/// ordinal and purpose instead of repeating one chain-wide value. Entry 0 is
/// the stage 0 FOLD's own fresh mask tree; the loop never derives a tree from
/// it, and it is carried so the caller binds the same trusted values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IoppStageSeed {
    pub switch_run: Digest,
    pub mask_run: Digest,
    pub configuration: Digest,
}

/// The caller's plan values that are not part of the stage list itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IoppChainConfig {
    /// The stage 0 FOLD's codes, used to check the supplied handoff.
    pub first_main: CodeParams,
    pub first_mask: CodeParams,
    /// The BASE position count; it must equal the last stage's count.
    pub base_positions: usize,
    /// One identifier triple per non-last switch, in stage order.
    pub fold_ids: Vec<IoppStageFoldIds>,
    /// One allocated tree number per switch, in stage order.
    pub switch_tree_numbers: Vec<u64>,
    /// One allocated tree number per non-last switch's stage FOLD mask tree,
    /// in stage order.
    pub fold_tree_numbers: Vec<u64>,
    /// The leaf class of every stage FOLD mask tree.
    pub mask_leaf_class: u32,
    pub switch_salt_bytes: usize,
    /// One derivation entry per stage, in stage order.
    pub stage_seeds: Vec<IoppStageSeed>,
    /// The BASE masking tree, distinct from the last switch's tree.
    pub base_star: RuntimeTreePlan,
}

/// The trusted stage chain: every stage entry with its derived switch target,
/// committed tree plan and following stage FOLD plan.
#[derive(Clone, Debug)]
pub struct IoppChain {
    pub stages: Vec<IoppStage>,
    pub config: IoppChainConfig,
}

impl IoppChain {
    /// Derives the complete stage chain from the trusted stage list and
    /// configuration. Every switch target, tree plan and stage FOLD plan is
    /// derived here and checked against the stage-list invariants; a stage
    /// list or configuration that disagrees is rejected before any session.
    pub fn derive(stages: &[IoppStagePlan], config: &IoppChainConfig) -> Result<Self> {
        if iopp_stage_fault(
            stages,
            config.first_main,
            config.first_mask,
            config.base_positions,
        )
        .is_some()
        {
            return Err(Error::InvalidConfig);
        }
        if stages
            .iter()
            .take(stages.len() - 1)
            .any(|stage| stage.positions == 0)
        {
            return Err(Error::InvalidConfig);
        }
        if config.base_positions == 0
            || config.mask_leaf_class == IOPP_SWITCH_LEAF_CLASS
            || config.mask_leaf_class == config.base_star.leaf_class
            || config.stage_seeds.len() != stages.len()
            || config.stage_seeds.iter().any(|seed| {
                seed.switch_run == Digest::ZERO
                    || seed.mask_run == Digest::ZERO
                    || seed.configuration == Digest::ZERO
            })
            || config.switch_salt_bytes == 0
        {
            return Err(Error::InvalidConfig);
        }
        let switch_count = stages.len() - 1;
        if config.fold_ids.len() != switch_count - 1
            || config.switch_tree_numbers.len() != switch_count
            || config.fold_tree_numbers.len() != switch_count - 1
        {
            return Err(Error::ShapeMismatch);
        }
        if fold_ids_fault(&config.fold_ids) {
            return Err(Error::InvalidConfig);
        }
        let mut used_trees = Vec::with_capacity(switch_count * 2);
        used_trees.extend_from_slice(&config.switch_tree_numbers);
        used_trees.extend_from_slice(&config.fold_tree_numbers);
        used_trees.push(config.base_star.tree_number);
        used_trees.sort_unstable();
        if used_trees[0] == 0 || used_trees.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(Error::InvalidConfig);
        }
        let mut derived = Vec::with_capacity(stages.len());
        derived.push(IoppStage {
            plan: stages[0].clone(),
            target: None,
            tree: None,
            fold: None,
        });
        for (index, plan) in stages.iter().enumerate().skip(1) {
            if plan.positions == 0 {
                return Err(Error::InvalidConfig);
            }
            let target = SwitchTarget::from_stage(
                plan.main,
                plan.mask,
                plan.k,
                plan.h,
                plan.samples,
            )?;
            let tree = switch_tree_plan_multi_at(
                config.switch_tree_numbers[index - 1],
                &target,
                config.switch_salt_bytes,
                config.stage_seeds[index].switch_run,
                config.stage_seeds[index].configuration,
            )?;
            if tree.fault().is_some() {
                return Err(Error::InvalidConfig);
            }
            let fold = if index + 1 < stages.len() {
                let ids = config.fold_ids[index - 1];
                let shape = FoldShape {
                    code: target.code,
                    mask_code: target.mask_code,
                    k: plan.k,
                    h: plan.h,
                    mask_leaf_class: config.mask_leaf_class,
                };
                if fold_mask_shape_fault(&shape).is_some() {
                    return Err(Error::InvalidConfig);
                }
                let layout = FoldGroupLayout {
                    group: ids.group,
                    k: plan.k,
                    h: plan.h,
                    members: vec![FoldMemberLayout {
                        oracle: ids.member,
                        main_rows: target.main_rows,
                        mask_rows: target.mask_rows,
                        kept: false,
                    }],
                };
                if layout.fault(&shape).is_some() {
                    return Err(Error::InvalidConfig);
                }
                let mask_tree = MaskTreePlan {
                    tree_number: config.fold_tree_numbers[index - 1],
                    leaf_class: config.mask_leaf_class,
                    log_n: target.mask_code.log_n,
                    mask_rows: target.mask_rows,
                    salt_bytes: config.switch_salt_bytes,
                    run: config.stage_seeds[index].mask_run,
                    configuration: config.stage_seeds[index].configuration,
                };
                if mask_tree.fault().is_some() {
                    return Err(Error::InvalidConfig);
                }
                Some(IoppStageFold {
                    shape,
                    layout,
                    mask_tree,
                    group: ids.group,
                    member: ids.member,
                    fresh: ids.fresh,
                })
            } else {
                None
            };
            derived.push(IoppStage {
                plan: plan.clone(),
                target: Some(target),
                tree: Some(tree),
                fold,
            });
        }
        let last = derived.last().ok_or(Error::InvalidConfig)?;
        let last_tree = last.tree.clone().ok_or(Error::InvalidConfig)?;
        let star = &config.base_star;
        if star.fault().is_some()
            || star.parts.len() != 1
            || star.parts[0].main_symbols != E_DEGREE as u64
            || star.parts[0].mask_symbols != E_DEGREE as u64
            || star.log_n != last_tree.log_n
            || star.log_n_zk != last_tree.log_n_zk
            || *star == last_tree
        {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            stages: derived,
            config: config.clone(),
        })
    }

    /// The proof-visible switch targets, in stage order.
    pub fn switch_targets(&self) -> Vec<SwitchTarget> {
        self.stages
            .iter()
            .skip(1)
            .filter_map(|stage| stage.target)
            .collect()
    }

    pub fn base_positions(&self) -> usize {
        self.config.base_positions
    }
}

/// One initial member commitment supplied by the hidden session.
#[derive(Clone, Copy, Debug)]
pub struct IoppInitialMember<'a> {
    pub oracle: u64,
    pub tree: &'a RuntimeTree,
}

/// The hidden-session adapter for the switch stage's old-row commitments.
///
/// `Initial` is the first switch: the old main rows come from the stage 0
/// member commitments and the old mask rows from either the dedicated shared
/// mask tree or the members' mask parts. `Stage` is every later switch: the
/// previous switch's committed message is the only member.
pub enum IoppSwitchMembers<'a> {
    Initial {
        members: &'a [IoppInitialMember<'a>],
        old_mask: Option<&'a MaskTree>,
        fresh: &'a MaskTree,
    },
    Stage {
        member: &'a CsMultiProver,
        fresh: &'a MaskTree,
    },
}

impl<'a> IoppSwitchMembers<'a> {
    /// The old-mask source the session must register, matching this adapter.
    /// A dedicated mask tree wins; otherwise the members' mask parts are used,
    /// or the trusted layout's zero mask when no member carries mask rows.
    pub fn old_mask_source(&self, layout: &FoldGroupLayout) -> OldMaskSource {
        match self {
            Self::Initial {
                old_mask: Some(tree),
                ..
            } => OldMaskSource::Tree {
                plan: tree.plan,
                root: tree.root,
            },
            Self::Initial { .. } => {
                if layout.members.iter().any(|member| member.mask_rows > 0) {
                    OldMaskSource::Member
                } else {
                    OldMaskSource::Zero
                }
            }
            Self::Stage { .. } => OldMaskSource::Member,
        }
    }

    /// Opens every old member at the drawn main positions, the fresh mask at
    /// the drawn mask positions, and the dedicated old mask tree when present.
    pub fn open(
        &self,
        positions: &[u64],
        mask_positions: &[u64],
    ) -> Result<(
        Vec<RuntimeTreeOpening>,
        RuntimeTreeOpening,
        Option<RuntimeTreeOpening>,
    )> {
        match self {
            Self::Initial {
                members,
                old_mask,
                fresh,
            } => {
                let mut openings = Vec::with_capacity(members.len());
                for member in members.iter() {
                    openings.push(member.tree.open(positions)?);
                }
                let old = match old_mask {
                    Some(tree) => Some(tree.open(mask_positions)?),
                    None => None,
                };
                Ok((openings, fresh.open(mask_positions)?, old))
            }
            Self::Stage { member, fresh } => Ok((
                vec![member.open(positions)?],
                fresh.open(mask_positions)?,
                None,
            )),
        }
    }
}

/// The prover's stage 0 input: the BAT/FOLD handoff, its authentication basis,
/// the folded rows, the member commitments and the two mask trees. The member
/// trees are shared with the session's own commitments, never copied.
pub struct IoppStageZero {
    pub handoff: FoldHandoff,
    pub basis: FoldBasis,
    pub rows: SingleRows,
    pub member_trees: Vec<Arc<RuntimeTree>>,
    pub old_mask_tree: Option<MaskTree>,
    pub fresh_tree: MaskTree,
}

/// The verifier's stage 0 input. Member roots and tree plans are the trusted
/// session commitments; the optional dedicated old mask tree is the trusted
/// first switch mask commitment.
pub struct IoppVerifyStageZero {
    pub handoff: FoldHandoff,
    pub basis: FoldBasis,
    pub first_old_mask: Option<(MaskTreePlan, Digest)>,
}

/// One switch's private row encodings: the new message's per-row tails, the
/// private pad and one salt per new tree column.
#[derive(Clone, Debug)]
pub struct IoppSwitchPrivate {
    pub main_tails: Vec<Vec<Fp3>>,
    pub mask_tails: Vec<Vec<Fp3>>,
    pub pad: Fp3,
    pub salts: Vec<Vec<u8>>,
}

/// One stage FOLD's private fresh-mask seed and column salts.
#[derive(Clone, Debug)]
pub struct IoppStageFoldPrivate {
    pub seed: Digest,
    pub salts: Vec<Vec<u8>>,
}

/// The BASE private masking inputs.
#[derive(Clone, Debug)]
pub struct IoppBasePrivate {
    pub salts: Vec<Vec<u8>>,
    pub randomness: LastBaseRandomness,
}

/// One stage FOLD's public messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IoppStageFoldMessage {
    pub start: FoldStartMessage,
    pub rounds: Vec<(Fp3, Fp3)>,
}

/// One switch's public messages and openings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IoppSwitchMessage {
    pub target: SwitchTarget,
    pub identity: Digest,
    pub declared_root: Digest,
    pub sample_answer: Fp3,
    pub nonce: u64,
    pub member_openings: Vec<RuntimeTreeOpening>,
    pub fresh_opening: RuntimeTreeOpening,
    pub old_mask_opening: Option<RuntimeTreeOpening>,
    pub stage_fold: Option<IoppStageFoldMessage>,
}

/// The complete multi-stage proof: the switch messages in stage order and the
/// final BASE artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IoppProof {
    pub switches: Vec<IoppSwitchMessage>,
    pub base: Vec<u8>,
}

impl IoppProof {
    /// Encodes the proof. Every field is written field by field, including
    /// the switch count, each target and the stage FOLD presence and round
    /// count; the decoder checks them all against the trusted chain.
    pub fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(self.switches.len() as u64);
        for record in &self.switches {
            put_switch_target(&mut writer, &record.target);
            writer.put_digest(&record.identity);
            writer.put_digest(&record.declared_root);
            writer.put_fp3(&record.sample_answer);
            writer.put_u64(record.nonce);
            writer.put_u64(record.member_openings.len() as u64);
            for opening in &record.member_openings {
                put_runtime_opening(&mut writer, opening);
            }
            put_runtime_opening(&mut writer, &record.fresh_opening);
            match &record.old_mask_opening {
                Some(opening) => {
                    writer.put_u64(1);
                    put_runtime_opening(&mut writer, opening);
                }
                None => writer.put_u64(0),
            }
            match &record.stage_fold {
                Some(fold) => {
                    writer.put_u64(1);
                    writer.put_u64(fold.rounds.len() as u64);
                    put_fold_start(&mut writer, &fold.start);
                    for (c0, c2) in &fold.rounds {
                        writer.put_fp3(c0);
                        writer.put_fp3(c2);
                    }
                }
                None => writer.put_u64(0),
            }
        }
        writer.put_bytes(&self.base);
        writer.into_bytes()
    }

    /// Decodes the proof against the trusted chain. The switch count, every
    /// target field, the stage FOLD presence and its round count are checked
    /// against the plan before any value is used, and trailing bytes are
    /// rejected.
    pub fn decode(plan: &IoppChain, bytes: &[u8]) -> Result<Self> {
        let expected_switches = plan.stages.len() - 1;
        let last_index = plan.stages.len() - 1;
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        let switch_count =
            usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if switch_count != expected_switches {
            return Err(Error::ShapeMismatch);
        }
        let mut switches = Vec::with_capacity(switch_count);
        for index in 0..switch_count {
            let stage_index = index + 1;
            let expected = plan.stages[stage_index]
                .target
                .ok_or(Error::InvalidConfig)?;
            let target = get_switch_target(&mut reader, &expected)?;
            let identity = reader.get_digest()?;
            let declared_root = reader.get_digest()?;
            let sample_answer = reader.get_fp3()?;
            let nonce = reader.get_u64()?;
            let member_count =
                usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
            if member_count == 0 || member_count > IOPP_MEMBER_DECODE_BOUND {
                return Err(Error::ResourceLimit);
            }
            let mut member_openings = Vec::with_capacity(member_count);
            for _ in 0..member_count {
                member_openings.push(get_runtime_opening(
                    &mut reader,
                    IOPP_OPENING_DECODE_BOUND,
                )?);
            }
            let fresh_opening =
                get_runtime_opening(&mut reader, IOPP_OPENING_DECODE_BOUND)?;
            let old_mask_opening = match reader.get_u64()? {
                0 => None,
                1 => Some(get_runtime_opening(
                    &mut reader,
                    IOPP_OPENING_DECODE_BOUND,
                )?),
                _ => return Err(Error::InvalidEncoding),
            };
            let stage_fold = match reader.get_u64()? {
                0 => {
                    if stage_index != last_index {
                        return Err(Error::ShapeMismatch);
                    }
                    None
                }
                1 => {
                    if stage_index == last_index {
                        return Err(Error::ShapeMismatch);
                    }
                    let fold = plan.stages[stage_index]
                        .fold
                        .as_ref()
                        .ok_or(Error::InvalidConfig)?;
                    let rounds = usize::try_from(reader.get_u64()?)
                        .map_err(|_| Error::ResourceLimit)?;
                    if rounds != fold.shape.k as usize {
                        return Err(Error::ShapeMismatch);
                    }
                    let start = get_fold_start(&mut reader)?;
                    let mut values = Vec::with_capacity(rounds);
                    for _ in 0..rounds {
                        values.push((reader.get_fp3()?, reader.get_fp3()?));
                    }
                    Some(IoppStageFoldMessage {
                        start,
                        rounds: values,
                    })
                }
                _ => return Err(Error::InvalidEncoding),
            };
            switches.push(IoppSwitchMessage {
                target,
                identity,
                declared_root,
                sample_answer,
                nonce,
                member_openings,
                fresh_opening,
                old_mask_opening,
                stage_fold,
            });
        }
        let base = reader.get_bytes()?.to_vec();
        reader.finish()?;
        Ok(Self { switches, base })
    }
}

/// Writes one switch target field by field.
fn put_switch_target(writer: &mut CanonicalWriter, target: &SwitchTarget) {
    put_code_public(writer, target.code);
    put_code_public(writer, target.mask_code);
    writer.put_u64(target.main_rows);
    writer.put_u64(target.mask_rows);
    writer.put_u64(target.answers as u64);
    put_cut(writer, target.main);
    put_cut(writer, target.rest);
}

/// Reads one switch target and rejects it unless it equals the trusted target.
fn get_switch_target(
    reader: &mut CanonicalReader<'_>,
    expected: &SwitchTarget,
) -> Result<SwitchTarget> {
    let target = SwitchTarget {
        code: get_code_public(reader)?,
        mask_code: get_code_public(reader)?,
        main_rows: reader.get_u64()?,
        mask_rows: reader.get_u64()?,
        answers: u32::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?,
        main: get_cut(reader)?,
        rest: get_cut(reader)?,
    };
    if target != *expected {
        return Err(Error::StatementMismatch);
    }
    Ok(target)
}

fn put_cut(writer: &mut CanonicalWriter, cut: Cut) {
    let part = match cut.part {
        CutPart::Main => 0,
        CutPart::Mask => 1,
    };
    writer.put_u64(part);
    writer.put_u64(cut.bits as u64);
    writer.put_u64(cut.first_row);
}

fn get_cut(reader: &mut CanonicalReader<'_>) -> Result<Cut> {
    let part = match reader.get_u64()? {
        0 => CutPart::Main,
        1 => CutPart::Mask,
        _ => return Err(Error::InvalidEncoding),
    };
    let bits = u32::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    let first_row = reader.get_u64()?;
    Ok(Cut {
        part,
        bits,
        first_row,
    })
}

struct StageFoldOutcome {
    output: FoldOutput,
    chain: Chain,
    record: IoppStageFoldMessage,
    end_mu: Vec<Fp3>,
    end_fresh: Vec<Fp3>,
    folded_main: Vec<Fp3>,
    fresh_tree: MaskTree,
    fresh_root: Digest,
    fresh_tail: Vec<Fp3>,
}

/// Runs the complete multi-stage loop and returns the proof plus the prover's
/// final transcript value. The prover's BASE session mirrors the verifier's
/// exactly, so the returned seal equals `verify_multi_stage_iopp`'s.
pub fn prove_multi_stage_iopp(
    mut chain: Chain,
    plan: &IoppChain,
    stage_zero: IoppStageZero,
    switches: &[IoppSwitchPrivate],
    stage_folds: &[IoppStageFoldPrivate],
    base: &IoppBasePrivate,
) -> Result<(IoppProof, Digest)> {
    let switch_count = plan.stages.len() - 1;
    if switches.len() != switch_count || stage_folds.len() != switch_count - 1 {
        return Err(Error::ShapeMismatch);
    }
    let IoppStageZero {
        handoff,
        basis,
        rows,
        member_trees,
        old_mask_tree,
        fresh_tree,
    } = stage_zero;
    validate_stage_zero(
        &plan.stages[0].plan,
        &handoff,
        &basis,
        Some(&rows),
        old_mask_tree.as_ref(),
        Some(&member_trees),
        fresh_tree.plan,
        fresh_tree.root,
    )?;
    check_stage_zero_ids(plan, &handoff, &basis)?;
    let initial_members: Vec<IoppInitialMember> = member_trees
        .iter()
        .zip(basis.members.iter())
        .map(|(tree, member)| IoppInitialMember {
            oracle: member.oracle,
            tree: tree.as_ref(),
        })
        .collect();
    let mut handoff = handoff;
    let mut basis = basis;
    let mut rows = rows;
    let mut fresh_tree = fresh_tree;
    let mut previous: Option<CsMultiProver> = None;
    let mut records = Vec::with_capacity(switch_count);
    let last_index = plan.stages.len() - 1;
    for index in 1..plan.stages.len() {
        let stage = &plan.stages[index];
        let target = stage.target.ok_or(Error::InvalidConfig)?;
        let tree_plan = stage.tree.clone().ok_or(Error::InvalidConfig)?;
        let private = &switches[index - 1];
        let prover = CsMultiProver::new(
            rows.clone(),
            target,
            private.main_tails.clone(),
            private.mask_tails.clone(),
            private.pad,
            tree_plan.clone(),
            &private.salts,
        )?;
        let members = match previous.as_ref() {
            Some(member) => IoppSwitchMembers::Stage {
                member,
                fresh: &fresh_tree,
            },
            None => IoppSwitchMembers::Initial {
                members: &initial_members,
                old_mask: old_mask_tree.as_ref(),
                fresh: &fresh_tree,
            },
        };
        let mut session = CsMultiSession::new(
            chain,
            handoff.clone(),
            basis.clone(),
            target,
            *prover.shape(),
            tree_plan.clone(),
            members.old_mask_source(&basis.layout),
        )?;
        let identity = fold_identity(&handoff);
        session.receive_declarations(identity, prover.root())?;
        let seed = session.draw_sample_seed()?;
        let sample_answer = prover.sample_answer(&seed)?;
        session.receive_answer(sample_answer)?;
        let previous_stage = &plan.stages[index - 1].plan;
        session.set_work_bits(previous_stage.work_bits)?;
        let (positions, _theta, nonce) =
            session.draw_positions_theta_with_nonce(previous_stage.positions, None)?;
        let unique = unique_sorted(&positions);
        let mask_positions: Vec<u64> = positions
            .iter()
            .map(|position| position % handoff.mask_code.n())
            .collect();
        let mask_unique = unique_sorted(&mask_positions);
        let (member_openings, fresh_opening, old_mask_opening) =
            members.open(&unique, &mask_unique)?;
        let claims = session
            .authenticate_and_register(
                &member_openings,
                &fresh_opening,
                old_mask_opening.as_ref(),
            )?
            .clone();
        let declared_root = session
            .declared_root()
            .ok_or(Error::InvalidSessionState)?;
        let mut record = IoppSwitchMessage {
            target,
            identity,
            declared_root,
            sample_answer,
            nonce,
            member_openings,
            fresh_opening,
            old_mask_opening,
            stage_fold: None,
        };
        chain = session.into_chain();
        if index == last_index {
            let base_rows = last_stage_single_rows(target, &claims, prover.message())?;
            let star_plan = plan.config.base_star.clone();
            let base_prover = LastBaseProver::new(
                base_rows.clone(),
                star_plan.clone(),
                &base.salts,
                &base.randomness,
            )?;
            let messages = base_prover.messages()?;
            let mut base_session = LastBaseSession::new(
                chain,
                base_rows,
                tree_plan.clone(),
                declared_root,
                star_plan,
            )?;
            base_session.receive_star(messages.star_root, messages.target_star)?;
            let lambda = base_session.draw_lambda()?;
            let prepared = base_prover.base_rows(lambda)?;
            base_session.receive_rows(prepared.clone())?;
            let base_positions = base_session.draw_base_positions(plan.base_positions())?;
            let star_opening = base_prover.open_star(&base_positions)?;
            let final_opening = prover.open(&base_positions)?;
            base_session.verify_openings(&star_opening, &final_opening)?;
            let seal = base_session.finish()?;
            let bytes = encode_last_base_proof(
                &messages,
                &prepared,
                &star_opening,
                &final_opening,
                base_prover.plan(),
                &tree_plan,
            )?;
            records.push(record);
            return Ok((
                IoppProof {
                    switches: records,
                    base: bytes,
                },
                seal,
            ));
        }
        let fold = stage.fold.as_ref().ok_or(Error::InvalidConfig)?;
        let fold_private = &stage_folds[index - 1];
        let row = FoldInputRow {
            group: fold.group,
            k: fold.shape.k,
            h: fold.shape.h,
            members: vec![fold.member],
            coefficients: claims.combined.main.clone(),
            mask_coefficients: claims.combined.mask.clone(),
            kept: Vec::new(),
            target: claims.combined.target,
        };
        let outcome = run_stage_fold(
            chain,
            fold,
            row,
            prover.message().main.clone(),
            prover.message().mask.clone(),
            fold_private,
        )?;
        record.stage_fold = Some(outcome.record.clone());
        records.push(record);
        let (next_handoff, next_basis, next_rows) = build_stage_fold_stage(
            fold,
            &outcome,
            prover.root(),
            prover.plan().clone(),
            prover.main_tails(),
            prover.mask_tails(),
        )?;
        handoff = next_handoff;
        basis = next_basis;
        rows = next_rows;
        chain = outcome.chain;
        fresh_tree = outcome.fresh_tree;
        previous = Some(prover);
    }
    Err(Error::InvalidSessionState)
}

/// Verifies the multi-stage loop from the trusted stage chain and the proof.
/// The proof's switch sequence must equal the plan's sequence before any
/// proof-supplied root, opening or stage FOLD message is used.
pub fn verify_multi_stage_iopp(
    chain: Chain,
    plan: &IoppChain,
    stage_zero: IoppVerifyStageZero,
    proof: &IoppProof,
) -> Result<Digest> {
    verify_multi_stage_iopp_with_last_positions(chain, plan, stage_zero, proof)
        .map(|(seal, _)| seal)
}

/// Verifies the loop and also returns the last-stage BASE positions. The
/// external opening ledger uses the positions for the LastBase artifact.
pub fn verify_multi_stage_iopp_with_last_positions(
    mut chain: Chain,
    plan: &IoppChain,
    stage_zero: IoppVerifyStageZero,
    proof: &IoppProof,
) -> Result<(Digest, Vec<u64>)> {
    let switch_count = plan.stages.len() - 1;
    if proof.switches.len() != switch_count {
        return Err(Error::StatementMismatch);
    }
    let IoppVerifyStageZero {
        handoff,
        basis,
        first_old_mask,
    } = stage_zero;
    validate_stage_zero(
        &plan.stages[0].plan,
        &handoff,
        &basis,
        None,
        None,
        None,
        basis.fresh_tree,
        basis.fresh_root,
    )?;
    if let Some((old_plan, old_root)) = first_old_mask {
        check_old_mask(&old_plan, old_root, &handoff)?;
        if old_root == basis.fresh_root || old_plan == basis.fresh_tree {
            // The stage 0 FOLD's fresh mask is not the group's old mask.
            return Err(Error::StatementMismatch);
        }
    }
    check_stage_zero_ids(plan, &handoff, &basis)?;
    let provided: Vec<SwitchTarget> = proof
        .switches
        .iter()
        .map(|record| record.target)
        .collect();
    check_stage_sequence(&plan.switch_targets(), &provided)?;
    let mut handoff = handoff;
    let mut basis = basis;
    let last_index = plan.stages.len() - 1;
    for index in 1..plan.stages.len() {
        let stage = &plan.stages[index];
        let target = stage.target.ok_or(Error::InvalidConfig)?;
        let tree_plan = stage.tree.clone().ok_or(Error::InvalidConfig)?;
        let record = &proof.switches[index - 1];
        let source = if index == 1 {
            match first_old_mask {
                Some((plan, root)) => OldMaskSource::Tree { plan, root },
                None if basis
                    .layout
                    .members
                    .iter()
                    .any(|member| member.mask_rows > 0) =>
                {
                    OldMaskSource::Member
                }
                None => OldMaskSource::Zero,
            }
        } else {
            OldMaskSource::Member
        };
        let mut session = CsMultiSession::new(
            chain,
            handoff.clone(),
            basis.clone(),
            target,
            target_shape(&target),
            tree_plan.clone(),
            source,
        )?;
        session.receive_declarations(record.identity, record.declared_root)?;
        let _seed = session.draw_sample_seed()?;
        session.receive_answer(record.sample_answer)?;
        let previous_stage = &plan.stages[index - 1].plan;
        session.set_work_bits(previous_stage.work_bits)?;
        let _ = session.draw_positions_theta_with_nonce(
            previous_stage.positions,
            Some(record.nonce),
        )?;
        let claims = session
            .authenticate_and_register(
                &record.member_openings,
                &record.fresh_opening,
                record.old_mask_opening.as_ref(),
            )?
            .clone();
        let declared_root = session
            .declared_root()
            .ok_or(Error::InvalidSessionState)?;
        chain = session.into_chain();
        if index == last_index {
            if record.stage_fold.is_some() {
                return Err(Error::StatementMismatch);
            }
            let rows = last_stage_verify_rows(target, &claims)?;
            let final_plan = tree_plan;
            let base_session = LastBaseSession::new(
                chain,
                rows,
                final_plan,
                declared_root,
                plan.config.base_star.clone(),
            )?;
            return verify_last_base_proof_with_positions(
                base_session,
                plan.base_positions(),
                &proof.base,
            );
        }
        let fold = stage.fold.as_ref().ok_or(Error::InvalidConfig)?;
        let fold_record = record
            .stage_fold
            .as_ref()
            .ok_or(Error::StatementMismatch)?;
        let row = FoldInputRow {
            group: fold.group,
            k: fold.shape.k,
            h: fold.shape.h,
            members: vec![fold.member],
            coefficients: claims.combined.main.clone(),
            mask_coefficients: claims.combined.mask.clone(),
            kept: Vec::new(),
            target: claims.combined.target,
        };
        let (next_chain, next_handoff, next_basis) = verify_stage_fold(
            chain,
            fold,
            row,
            fold_record,
            declared_root,
            tree_plan,
        )?;
        chain = next_chain;
        handoff = next_handoff;
        basis = next_basis;
    }
    Err(Error::InvalidSessionState)
}

fn target_shape(target: &SwitchTarget) -> GuOracleShape {
    GuOracleShape {
        code: target.code,
        main_rows: target.main_rows,
        main_deg: E_DEGREE as u32,
        mask_rows: target.mask_rows,
        mask_code: Some(target.mask_code),
        answers: target.answers,
    }
}

/// Checks that the proof carries exactly the trusted switch sequence. A
/// missing, added or reordered switch fails before its contents are used.
fn check_stage_sequence(expected: &[SwitchTarget], provided: &[SwitchTarget]) -> Result<()> {
    if expected.len() != provided.len() {
        return Err(Error::StatementMismatch);
    }
    for (want, got) in expected.iter().zip(provided) {
        if want != got {
            return Err(Error::StatementMismatch);
        }
    }
    Ok(())
}

/// Validates the supplied stage 0 handoff, basis and (on the prover side) rows
/// and commitments against the trusted first stage.
#[allow(clippy::too_many_arguments)]
fn validate_stage_zero(
    stage0: &IoppStagePlan,
    handoff: &FoldHandoff,
    basis: &FoldBasis,
    rows: Option<&SingleRows>,
    old_mask_tree: Option<&MaskTree>,
    member_trees: Option<&[Arc<RuntimeTree>]>,
    fresh_plan: MaskTreePlan,
    fresh_root: Digest,
) -> Result<()> {
    let shape = FoldShape {
        code: stage0.main,
        mask_code: stage0.mask,
        k: stage0.k,
        h: stage0.h,
        mask_leaf_class: fresh_plan.leaf_class,
    };
    if handoff.code != stage0.main
        || handoff.mask_code != stage0.mask
        || handoff.k != stage0.k
        || handoff.h != stage0.h
        || handoff.gamma.len() != stage0.k as usize
        || handoff.kept_rows != 0
        || basis.layout != handoff.layout
        || basis.layout.group != handoff.group
        || basis.layout.k != handoff.k
        || basis.layout.h != handoff.h
        || basis.layout.fault(&shape).is_some()
        || handoff.members.len() != basis.layout.members.len()
    {
        return Err(Error::StatementMismatch);
    }
    for (member, layout) in handoff.members.iter().zip(basis.layout.members.iter()) {
        if *member != layout.oracle {
            return Err(Error::StatementMismatch);
        }
    }
    let mask_rows = shape.mask_rows()?;
    let mask_ell =
        usize::try_from(shape.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
    if handoff.main_claim.len() != stage0.main.ell as usize
        || handoff.mask_claim.len() != mask_ell
        || handoff.fresh_claim.len() != mask_ell
        || handoff.fresh_claim
            != fold_mask_weights(handoff.k, handoff.h, mask_ell, &handoff.gamma)?
        || handoff.fresh_root != basis.fresh_root
        || handoff.fresh_root != fresh_root
        || handoff.fresh_root == Digest::ZERO
        || basis.fresh_tree.tree_number != handoff.tree_number
        || basis.fresh_tree.log_n != shape.mask_code.log_n
        || basis.fresh_tree.mask_rows != mask_rows
        || fresh_plan != basis.fresh_tree
        || fresh_root != basis.fresh_root
        || fresh_plan.fault().is_some()
    {
        return Err(Error::StatementMismatch);
    }
    if basis.members.len() != basis.layout.members.len() {
        return Err(Error::ShapeMismatch);
    }
    for (member, layout) in basis.members.iter().zip(basis.layout.members.iter()) {
        if member.oracle != layout.oracle
            || member.root == Digest::ZERO
            || member.tree.fault().is_some()
            || member.tree.log_n != handoff.code.log_n
            || member.tree.log_n_zk != handoff.mask_code.log_n
            || member.tree.parts.len() != 1
            || member.tree.parts[0].main_symbols
                != layout
                    .main_rows
                    .checked_mul(E_DEGREE as u64)
                    .ok_or(Error::ResourceLimit)?
            || member.tree.parts[0].mask_symbols
                != layout
                    .mask_rows
                    .checked_mul(E_DEGREE as u64)
                    .ok_or(Error::ResourceLimit)?
            || member.main_code != handoff.code
            || member.mask_code != handoff.mask_code
        {
            return Err(Error::StatementMismatch);
        }
    }
    if let Some(tree) = old_mask_tree {
        check_old_mask(&tree.plan, tree.root, handoff)?;
        if tree.root == basis.fresh_root || tree.plan == basis.fresh_tree {
            // The stage 0 FOLD's fresh mask is not the group's old mask.
            return Err(Error::StatementMismatch);
        }
    } else if !basis
        .layout
        .members
        .iter()
        .any(|member| member.mask_rows > 0)
    {
        // No member carries a mask part, so the fold input's mask table is the
        // zero vector by the trusted layout: the fold hides no mask variable
        // and the FOLD output's mask claim must be zero. It is derived here,
        // never taken from a proof-declared zero tree.
        if handoff.h != 0
            || handoff
                .mask_claim
                .iter()
                .any(|value| *value != Fp3::ZERO)
        {
            return Err(Error::StatementMismatch);
        }
    }
    if let Some(trees) = member_trees {
        if trees.len() != basis.members.len() {
            return Err(Error::ShapeMismatch);
        }
        for (tree, member) in trees.iter().zip(basis.members.iter()) {
            if *tree.plan() != member.tree || *tree.root() != member.root {
                return Err(Error::StatementMismatch);
            }
        }
    }
    if let Some(rows) = rows {
        rows.check()?;
        if rows.code != handoff.code
            || rows.mask_code != handoff.mask_code
            || rows.masks.len() != 2
            || rows.mask_tails.len() != 2
            || rows.mask_claims.len() != 2
            || rows.claim != handoff.main_claim
            || rows.mask_claims[0] != handoff.mask_claim
            || rows.mask_claims[1] != handoff.fresh_claim
            || rows.target != handoff.target
        {
            return Err(Error::StatementMismatch);
        }
        if old_mask_tree.is_none()
            && !basis
                .layout
                .members
                .iter()
                .any(|member| member.mask_rows > 0)
            && (rows.masks[0].iter().any(|value| *value != Fp3::ZERO)
                || rows.mask_tails[0]
                    .iter()
                    .any(|value| *value != Fp3::ZERO))
        {
            return Err(Error::StatementMismatch);
        }
    }
    Ok(())
}

/// Rejects zero identifiers, a fresh identifier equal to its member, and any
/// identifier repeated in the stage sequence. Each stage's fresh mask and
/// switched-member registration is a distinct object in the shared registry.
fn fold_ids_fault(ids: &[IoppStageFoldIds]) -> bool {
    if ids.iter().any(|entry| {
        entry.group == 0 || entry.member == 0 || entry.fresh == 0 || entry.fresh == entry.member
    }) {
        return true;
    }
    ids.iter().enumerate().any(|(index, entry)| {
        ids[index + 1..].iter().any(|other| {
            entry.member == other.member
                || entry.fresh == other.fresh
                || entry.group == other.group
        })
    })
}

/// Checks the allocated stage FOLD identifiers against stage 0's bound
/// identifiers. Stage 0 and every later stage share one identifier registry,
/// so no later stage may reuse a group, member or fresh identifier.
fn check_stage_zero_ids(
    chain_plan: &IoppChain,
    handoff: &FoldHandoff,
    basis: &FoldBasis,
) -> Result<()> {
    let mut bound = Vec::with_capacity(basis.layout.members.len() + 2);
    bound.push(handoff.group);
    bound.push(handoff.fresh);
    bound.extend(basis.layout.members.iter().map(|member| member.oracle));
    for stage in chain_plan
        .stages
        .iter()
        .filter_map(|stage| stage.fold.as_ref())
    {
        if bound.contains(&stage.group)
            || bound.contains(&stage.member)
            || bound.contains(&stage.fresh)
        {
            return Err(Error::InvalidConfig);
        }
    }
    Ok(())
}

/// Checks one dedicated old-mask tree against the stage 0 FOLD handoff.
fn check_old_mask(plan: &MaskTreePlan, root: Digest, handoff: &FoldHandoff) -> Result<()> {
    let mask_rows = 1u64
        .checked_shl(handoff.h)
        .ok_or(Error::ResourceLimit)?;
    if plan.fault().is_some()
        || plan.log_n != handoff.mask_code.log_n
        || plan.mask_rows != mask_rows
        || root == Digest::ZERO
    {
        return Err(Error::StatementMismatch);
    }
    Ok(())
}

/// Runs one non-last stage FOLD over the switched oracle message and a fresh
/// mask, mirroring the verifier's checks in the same transcript order.
fn run_stage_fold(
    mut chain: Chain,
    spec: &IoppStageFold,
    row: FoldInputRow,
    main_messages: Vec<Fp3>,
    mask_messages: Vec<Fp3>,
    private: &IoppStageFoldPrivate,
) -> Result<StageFoldOutcome> {
    let metadata = OracleMetadata {
        id: spec.fresh,
        group: spec.group,
        tree_number: spec.mask_tree.tree_number,
        leaf_class: spec.mask_tree.leaf_class,
        mask_rows: spec.mask_tree.mask_rows,
        mask_code: spec.shape.mask_code,
    };
    let statement = VerifiedStatement::new(
        VerifiedGroups {
            groups: vec![VerifiedGroup {
                group: spec.group,
                members: vec![spec.member],
                fresh: Some(spec.fresh),
                code: None,
                mask_code: None,
            }],
        },
        vec![spec.mask_tree],
        vec![metadata],
    )?;
    let mask_oracle = MaskOracle {
        id: spec.fresh,
        claimed_group: spec.group,
        main_rows: 0,
        mask_rows: spec.mask_tree.mask_rows,
        mask_code: spec.shape.mask_code,
        answers: 0,
        sampled: false,
    };
    if private.salts.len()
        != usize::try_from(spec.mask_tree.columns()).map_err(|_| Error::ResourceLimit)?
        || private
            .salts
            .iter()
            .any(|salt| salt.len() != spec.mask_tree.salt_bytes)
    {
        return Err(Error::ShapeMismatch);
    }
    let mut coins = ByteMaskCoins::new(private.seed)?;
    let mut mask = fold_mask_make(&spec.shape, &mut coins)?;
    fold_mask_make_tail(&spec.shape, &mut mask, &mut coins)?;
    mask.sent_as(spec.fresh)?;
    let start = fold_start_prove(
        &spec.shape,
        &mut mask,
        &coins,
        &mask_oracle,
        statement.groups(),
        spec.mask_tree,
        &private.salts,
    )?;
    let fresh_tail = mask
        .tail()
        .ok_or(Error::InvalidSessionState)?
        .values
        .clone();
    let mut runtime = RuntimeOracles::default();
    runtime.receive_mask(
        &statement,
        &spec.mask_tree,
        spec.fresh,
        &fold_mask_notice(&spec.shape)?,
        start.message.root,
    )?;
    fold_start_verify(&spec.shape, &start.message, &statement, &runtime)?;
    absorb_stage_fold_start(&mut chain, &start.message)?;
    let eps = chain.draw_e()?;
    let row_target = row.target;
    let mut prover = FoldProverSession::new(
        &spec.shape,
        &spec.layout,
        &mask,
        row,
        main_messages,
        mask_messages,
        Vec::new(),
    )?;
    if prover.sum() != start.message.sum {
        return Err(Error::StatementMismatch);
    }
    prover.start(eps, &coins)?;
    let mut claim = FoldRoundVerifier::new(start.message.sum, eps, row_target);
    let mut gammas = Vec::with_capacity(spec.shape.k as usize);
    let mut rounds = Vec::with_capacity(spec.shape.k as usize);
    for _ in 0..spec.shape.k {
        let round = prover.round()?;
        rounds.push((round[0], round[2]));
        absorb_stage_fold_round(&mut chain, round[0], round[2])?;
        let gamma = chain.draw_e()?;
        claim.round(round[0], round[2], gamma);
        gammas.push(gamma);
        prover.bind(gamma)?;
    }
    let end = prover.end()?;
    let output = prover.output(
        &spec.layout,
        spec.fresh,
        start.message.root,
        spec.mask_tree.tree_number,
        eps,
        start.message.sum,
        &gammas,
        claim.claim(),
    )?;
    Ok(StageFoldOutcome {
        output,
        chain,
        record: IoppStageFoldMessage {
            start: start.message,
            rounds,
        },
        end_mu: end.mu,
        end_fresh: end.fresh,
        folded_main: prover.main_messages().to_vec(),
        fresh_root: start.tree.root,
        fresh_tree: start.tree,
        fresh_tail,
    })
}

/// Verifies one stage FOLD message and rebuilds the next stage's handoff and
/// authentication basis from the trusted plan and the received messages.
fn verify_stage_fold(
    mut chain: Chain,
    spec: &IoppStageFold,
    row: FoldInputRow,
    record: &IoppStageFoldMessage,
    member_root: Digest,
    member_plan: RuntimeTreePlan,
) -> Result<(Chain, FoldHandoff, FoldBasis)> {
    let metadata = OracleMetadata {
        id: spec.fresh,
        group: spec.group,
        tree_number: spec.mask_tree.tree_number,
        leaf_class: spec.mask_tree.leaf_class,
        mask_rows: spec.mask_tree.mask_rows,
        mask_code: spec.shape.mask_code,
    };
    let statement = VerifiedStatement::new(
        VerifiedGroups {
            groups: vec![VerifiedGroup {
                group: spec.group,
                members: vec![spec.member],
                fresh: Some(spec.fresh),
                code: None,
                mask_code: None,
            }],
        },
        vec![spec.mask_tree],
        vec![metadata],
    )?;
    let mut runtime = RuntimeOracles::default();
    runtime.receive_mask(
        &statement,
        &spec.mask_tree,
        spec.fresh,
        &fold_mask_notice(&spec.shape)?,
        record.start.root,
    )?;
    fold_start_verify(&spec.shape, &record.start, &statement, &runtime)?;
    absorb_stage_fold_start(&mut chain, &record.start)?;
    let eps = chain.draw_e()?;
    if record.rounds.len() != spec.shape.k as usize {
        return Err(Error::ShapeMismatch);
    }
    let mut claim = FoldRoundVerifier::new(record.start.sum, eps, row.target);
    let mut gammas = Vec::with_capacity(spec.shape.k as usize);
    for (c0, c2) in &record.rounds {
        absorb_stage_fold_round(&mut chain, *c0, *c2)?;
        let gamma = chain.draw_e()?;
        claim.round(*c0, *c2, gamma);
        gammas.push(gamma);
    }
    let output = fold_output(
        &spec.shape,
        &spec.layout,
        &row,
        spec.fresh,
        record.start.root,
        spec.mask_tree.tree_number,
        eps,
        record.start.sum,
        &gammas,
        claim.claim(),
    )?;
    let anchor = FoldHandoffAnchor {
        shape: spec.shape,
        group: spec.group,
        members: vec![spec.member],
        fresh: spec.fresh,
        fresh_root: record.start.root,
        tree_number: spec.mask_tree.tree_number,
        layout: spec.layout.clone(),
        fold_gamma: output.gamma.clone(),
        fold_epsilon: output.eps,
        fold_sum: output.sum,
    };
    let handoff = FoldHandoff::from_output(&output, &anchor)?;
    if handoff.members != vec![spec.member] {
        return Err(Error::StatementMismatch);
    }
    let basis = FoldBasis {
        layout: handoff.layout.clone(),
        fresh_root: handoff.fresh_root,
        fresh_tree: spec.mask_tree,
        members: vec![FoldMember {
            oracle: spec.member,
            main_code: handoff.code,
            mask_code: handoff.mask_code,
            root: member_root,
            tree: member_plan,
        }],
    };
    Ok((chain, handoff, basis))
}

fn absorb_stage_fold_start(chain: &mut Chain, message: &FoldStartMessage) -> Result<()> {
    let mut out = chain.message();
    out.push_u64(message.id);
    out.push_u64(message.claimed_group);
    out.push_u64(message.notice.mask_rows);
    out.push_u64(message.notice.mask_code.log_n as u64);
    out.push_u64(message.notice.mask_code.ell);
    out.push_u64(message.notice.mask_code.r);
    out.push_u64(message.notice.answers as u64);
    out.push_u64(message.notice.leaf_class as u64);
    for chunk in message.root.0.chunks(8) {
        let word = u64::from_le_bytes(chunk.try_into().map_err(|_| Error::InvalidEncoding)?);
        out.push_u64(word);
    }
    out.push_fp3(&message.sum);
    let prefix = out.prefix(KIND_EPSILON, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn absorb_stage_fold_round(chain: &mut Chain, c0: Fp3, c2: Fp3) -> Result<()> {
    let mut out = chain.message();
    out.push_fp3(&c0);
    out.push_fp3(&c2);
    let prefix = out.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// Builds the next stage's rows and authentication basis from one verified
/// stage FOLD output and the switched member's own commitment.
fn build_stage_fold_stage(
    spec: &IoppStageFold,
    outcome: &StageFoldOutcome,
    member_root: Digest,
    member_tree: RuntimeTreePlan,
    main_tails: &[Vec<Fp3>],
    mask_tails: &[Vec<Fp3>],
) -> Result<(FoldHandoff, FoldBasis, SingleRows)> {
    let output = &outcome.output;
    if spec.layout.members.len() != 1 || spec.layout.members[0].oracle != spec.member {
        return Err(Error::StatementMismatch);
    }
    let anchor = FoldHandoffAnchor {
        shape: spec.shape,
        group: spec.group,
        members: vec![spec.member],
        fresh: spec.fresh,
        fresh_root: outcome.fresh_root,
        tree_number: spec.mask_tree.tree_number,
        layout: spec.layout.clone(),
        fold_gamma: output.gamma.clone(),
        fold_epsilon: output.eps,
        fold_sum: output.sum,
    };
    let handoff = FoldHandoff::from_output(output, &anchor)?;
    if handoff.members != vec![spec.member] {
        return Err(Error::StatementMismatch);
    }
    let basis = FoldBasis {
        layout: handoff.layout.clone(),
        fresh_root: handoff.fresh_root,
        fresh_tree: spec.mask_tree,
        members: vec![FoldMember {
            oracle: spec.member,
            main_code: handoff.code,
            mask_code: handoff.mask_code,
            root: member_root,
            tree: member_tree,
        }],
    };
    let main_coeff = foldeq_weights(&handoff.gamma, handoff.k)?;
    let mask_coeff = foldeq_weights(&handoff.gamma[..handoff.h as usize], handoff.h)?;
    let main_rows = usize::try_from(spec.layout.members[0].main_rows)
        .map_err(|_| Error::ResourceLimit)?;
    let mask_rows = usize::try_from(spec.layout.members[0].mask_rows)
        .map_err(|_| Error::ResourceLimit)?;
    if main_tails.len() != main_rows
        || mask_tails.len() != mask_rows
        || main_coeff.len() != main_rows
        || mask_coeff.len() != mask_rows
    {
        return Err(Error::ShapeMismatch);
    }
    let main_tail_len =
        usize::try_from(handoff.code.r).map_err(|_| Error::ResourceLimit)?;
    let mask_tail_len =
        usize::try_from(handoff.mask_code.r).map_err(|_| Error::ResourceLimit)?;
    let mut main_tail = vec![Fp3::ZERO; main_tail_len];
    for (row, tail) in main_tails.iter().enumerate() {
        if tail.len() != main_tail_len {
            return Err(Error::ShapeMismatch);
        }
        for (value, entry) in main_tail.iter_mut().zip(tail) {
            *value = value.add(&main_coeff[row].mul(entry));
        }
    }
    let mut mask_tail = vec![Fp3::ZERO; mask_tail_len];
    for (row, tail) in mask_tails.iter().enumerate() {
        if tail.len() != mask_tail_len {
            return Err(Error::ShapeMismatch);
        }
        for (value, entry) in mask_tail.iter_mut().zip(tail) {
            *value = value.add(&mask_coeff[row].mul(entry));
        }
    }
    if outcome.fresh_tail.len() != mask_rows * mask_tail_len {
        return Err(Error::ShapeMismatch);
    }
    let mut fresh_tail = vec![Fp3::ZERO; mask_tail_len];
    for (row, values) in outcome
        .fresh_tail
        .chunks_exact(mask_tail_len)
        .enumerate()
    {
        for (value, entry) in fresh_tail.iter_mut().zip(values) {
            *value = value.add(&mask_coeff[row].mul(entry));
        }
    }
    let rows = SingleRows {
        code: handoff.code,
        mask_code: handoff.mask_code,
        main: outcome.folded_main.clone(),
        main_tail,
        masks: vec![outcome.end_mu.clone(), outcome.end_fresh.clone()],
        mask_tails: vec![mask_tail, fresh_tail],
        claim: handoff.main_claim.clone(),
        mask_claims: vec![handoff.mask_claim.clone(), handoff.fresh_claim.clone()],
        target: handoff.target,
    };
    rows.check()?;
    Ok((handoff, basis, rows))
}

/// The prover's last-stage rows: the final switched message with its one main
/// row and one mask row, carrying the combined claims.
fn last_stage_single_rows(
    target: SwitchTarget,
    claims: &SwitchedClaims,
    message: &SwitchedRows,
) -> Result<SingleRows> {
    if target.main_rows != 1 || target.mask_rows != 1 {
        return Err(Error::UnsupportedPrimitive);
    }
    if message.main_tails.len() != 1 || message.mask_tails.len() != 1 {
        return Err(Error::ShapeMismatch);
    }
    let rows = SingleRows {
        code: target.code,
        mask_code: target.mask_code,
        main: message.main.clone(),
        main_tail: message.main_tails[0].clone(),
        masks: vec![message.mask.clone()],
        mask_tails: vec![message.mask_tails[0].clone()],
        claim: claims.combined.main.clone(),
        mask_claims: vec![claims.combined.mask.clone()],
        target: claims.combined.target,
    };
    rows.check()?;
    if rows.claim.len() != usize::try_from(rows.code.ell).map_err(|_| Error::ResourceLimit)?
        || rows.masks.len() != 1
    {
        return Err(Error::StatementMismatch);
    }
    Ok(rows)
}

/// The verifier's last-stage rows: only the shape and the combined claims are
/// needed. The opened values are supplied by the BASE artifact and checked
/// against them, so the row contents stay unconstrained.
fn last_stage_verify_rows(
    target: SwitchTarget,
    claims: &SwitchedClaims,
) -> Result<SingleRows> {
    if target.main_rows != 1 || target.mask_rows != 1 {
        return Err(Error::UnsupportedPrimitive);
    }
    let ell = usize::try_from(target.code.ell).map_err(|_| Error::ResourceLimit)?;
    let r = usize::try_from(target.code.r).map_err(|_| Error::ResourceLimit)?;
    let mask_ell =
        usize::try_from(target.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
    let mask_r =
        usize::try_from(target.mask_code.r).map_err(|_| Error::ResourceLimit)?;
    let rows = SingleRows {
        code: target.code,
        mask_code: target.mask_code,
        main: vec![Fp3::ZERO; ell],
        main_tail: vec![Fp3::ZERO; r],
        masks: vec![vec![Fp3::ZERO; mask_ell]],
        mask_tails: vec![vec![Fp3::ZERO; mask_r]],
        claim: claims.combined.main.clone(),
        mask_claims: vec![claims.combined.mask.clone()],
        target: claims.combined.target,
    };
    rows.check()?;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::code::{build_code_plan, encode_row_fp3};
    use zkie_engine::pcs::request::field::fp_from_u64;
    use zkie_engine::pcs::request::runtime_tree::{RuntimePartRows, RuntimeTreePart};

    use crate::proof::base_final_session::fold_coefficients;

    const CONFIGURATION: Digest = Digest([0x5a; 32]);
    const SWITCH_RUN: Digest = Digest([0x91; 32]);
    const MASK_RUN: Digest = Digest([0x92; 32]);
    const BASE_RUN: Digest = Digest([0x93; 32]);
    const GROUP: u64 = 7;
    const MEMBER0: u64 = 11;
    const MEMBER1: u64 = 12;
    const FRESH0: u64 = 21;
    const FRESH1: u64 = 22;
    const SALT_BYTES: usize = 40;
    const CHAIN_LABEL: u64 = 0x4d32;
    const CHAIN_PARAMETERS: Digest = Digest([0x77; 32]);

    fn code(log_n: u32, ell: u64, r: u64) -> CodeParams {
        build_code_plan(log_n, ell, r).unwrap()
    }

    fn stage_plan(
        main: CodeParams,
        mask: CodeParams,
        k: u32,
        h: u32,
        positions: usize,
    ) -> IoppStagePlan {
        IoppStagePlan {
            main,
            mask,
            k,
            h,
            positions,
            samples: 1,
            work_bits: 0,
        }
    }

    fn stages() -> Vec<IoppStagePlan> {
        // Adjacent stages differ in both the old-row draw count and the work
        // bits, so reading the wrong entry is rejected by the transcript.
        vec![
            IoppStagePlan {
                work_bits: 2,
                ..stage_plan(code(5, 8, 1), code(5, 16, 1), 2, 0, 1)
            },
            IoppStagePlan {
                positions: 2,
                work_bits: 1,
                ..stage_plan(code(6, 4, 2), code(6, 32, 2), 1, 1, 1)
            },
            stage_plan(code(8, 4, 1), code(8, 128, 1), 0, 0, 1),
        ]
    }

    fn base_star() -> RuntimeTreePlan {
        RuntimeTreePlan {
            tree_number: 0x600,
            leaf_class: 5,
            log_n: 8,
            log_n_zk: 8,
            parts: vec![RuntimeTreePart {
                main_symbols: E_DEGREE as u64,
                mask_symbols: E_DEGREE as u64,
            }],
            salt_bytes: SALT_BYTES,
            run: BASE_RUN,
            configuration: CONFIGURATION,
        }
    }

    fn config() -> IoppChainConfig {
        IoppChainConfig {
            first_main: code(5, 8, 1),
            first_mask: code(5, 16, 1),
            base_positions: 1,
            fold_ids: vec![IoppStageFoldIds {
                group: GROUP + 1,
                member: MEMBER1,
                fresh: FRESH1,
            }],
            switch_tree_numbers: vec![0x310, 0x311],
            fold_tree_numbers: vec![0x510],
            mask_leaf_class: IOPP_MASK_LEAF_CLASS,
            switch_salt_bytes: SALT_BYTES,
            stage_seeds: vec![
                IoppStageSeed {
                    switch_run: SWITCH_RUN,
                    mask_run: MASK_RUN,
                    configuration: CONFIGURATION,
                },
                IoppStageSeed {
                    switch_run: Digest([0xb1; 32]),
                    mask_run: Digest([0xb2; 32]),
                    configuration: Digest([0xb3; 32]),
                },
                IoppStageSeed {
                    switch_run: Digest([0xc1; 32]),
                    mask_run: Digest([0xc2; 32]),
                    configuration: Digest([0xc3; 32]),
                },
            ],
            base_star: base_star(),
        }
    }

    fn salts(columns: u64, seed: u8) -> Vec<Vec<u8>> {
        (0..columns)
            .map(|column| vec![(column as u8) ^ seed; SALT_BYTES])
            .collect()
    }

    fn member_plan() -> RuntimeTreePlan {
        RuntimeTreePlan {
            tree_number: 0x210,
            leaf_class: 2,
            log_n: 5,
            log_n_zk: 5,
            parts: vec![RuntimeTreePart {
                main_symbols: 12,
                mask_symbols: 3,
            }],
            salt_bytes: SALT_BYTES,
            run: Digest([0xa0; 32]),
            configuration: CONFIGURATION,
        }
    }

    fn mask_plan(tree_number: u64, seed: u8) -> MaskTreePlan {
        MaskTreePlan {
            tree_number,
            leaf_class: IOPP_MASK_LEAF_CLASS,
            log_n: 5,
            mask_rows: 1,
            salt_bytes: SALT_BYTES,
            run: Digest([seed; 32]),
            configuration: CONFIGURATION,
        }
    }

    fn member_tree(
        member_main: &[Vec<Fp3>],
        member_tails: &[Vec<Fp3>],
        mask_row: &[Fp3],
        mask_tail: &[Fp3],
    ) -> RuntimeTree {
        let plan = member_plan();
        let mut main = Vec::with_capacity(member_main.len());
        for (row, tail) in member_main.iter().zip(member_tails) {
            main.push(encode_row_fp3(&code(5, 8, 1), row, tail).unwrap());
        }
        let mask = vec![encode_row_fp3(&code(5, 16, 1), mask_row, mask_tail).unwrap()];
        RuntimeTree::commit(
            plan,
            &[RuntimePartRows {
                main,
                mask,
                main_deg: E_DEGREE as u32,
            }],
            &salts(32, 0x11),
        )
        .unwrap()
    }

    fn mask_tree(
        plan: MaskTreePlan,
        shape: &FoldShape,
        row: &[Fp3],
        tail: &[Fp3],
        salt_seed: u8,
    ) -> MaskTree {
        MaskTree::commit(plan, shape, &[row.to_vec()], &[tail.to_vec()], &salts(32, salt_seed))
            .unwrap()
    }

    fn stage_zero() -> (IoppStageZero, IoppVerifyStageZero) {
        let shape = FoldShape {
            code: code(5, 8, 1),
            mask_code: code(5, 16, 1),
            k: 2,
            h: 0,
            mask_leaf_class: IOPP_MASK_LEAF_CLASS,
        };
        let layout = FoldGroupLayout {
            group: GROUP,
            k: 2,
            h: 0,
            members: vec![FoldMemberLayout {
                oracle: MEMBER0,
                main_rows: 4,
                mask_rows: 1,
                kept: false,
            }],
        };
        assert!(layout.fault(&shape).is_none());
        let gamma = vec![Fp3::from_u64(3), Fp3::from_u64(5)];
        let eps = Fp3::from_u64(7);
        let main_claim = (1..=8).map(Fp3::from_u64).collect::<Vec<_>>();
        let mask_claim = (0..16)
            .map(|index| Fp3::from_u64(1000 + index))
            .collect::<Vec<_>>();
        let fresh_claim = fold_mask_weights(2, 0, 16, &gamma).unwrap();
        let old_mask_row = (0..16)
            .map(|index| Fp3::from_u64(2000 + index))
            .collect::<Vec<_>>();
        let fresh_mask_row = (0..16)
            .map(|index| Fp3::from_u64(3000 + index))
            .collect::<Vec<_>>();
        let old_mask_tail = vec![Fp3::from_u64(51)];
        let fresh_mask_tail = vec![Fp3::from_u64(52)];
        let member_main = (0..4)
            .map(|row| {
                (0..8)
                    .map(|column| Fp3::from_u64((row * 8 + column + 1) as u64))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let member_tails = (0..4)
            .map(|row| vec![Fp3::from_u64(900 + row as u64)])
            .collect::<Vec<_>>();
        let weights = fold_coefficients(2, &gamma).unwrap();
        let mut main_row = vec![Fp3::ZERO; 8];
        for (weight, row) in weights.iter().zip(&member_main) {
            for (value, entry) in main_row.iter_mut().zip(row) {
                *value = value.add(&weight.mul(entry));
            }
        }
        let mut main_tail = Fp3::ZERO;
        for (weight, tail) in weights.iter().zip(&member_tails) {
            main_tail = main_tail.add(&weight.mul(&tail[0]));
        }
        let mut target = Fp3::ZERO;
        for (coefficient, value) in main_claim.iter().zip(&main_row) {
            target = target.add(&coefficient.mul(value));
        }
        for (coefficient, value) in mask_claim.iter().zip(&old_mask_row) {
            target = target.add(&coefficient.mul(value));
        }
        for (coefficient, value) in fresh_claim.iter().zip(&fresh_mask_row) {
            target = target.add(&coefficient.mul(value));
        }
        let mut kept_factor = eps;
        for challenge in &gamma {
            kept_factor = kept_factor.mul(&Fp3::ONE.sub(challenge));
        }
        let sum = Fp3::from_u64(17);
        let tree = member_tree(&member_main, &member_tails, &old_mask_row, &old_mask_tail);
        let old_plan = mask_plan(0x481, 0xa2);
        let fresh_plan = mask_plan(0x480, 0xa1);
        let old_tree = mask_tree(old_plan, &shape, &old_mask_row, &old_mask_tail, 0x22);
        let old_root = old_tree.root;
        let fresh_tree = mask_tree(fresh_plan, &shape, &fresh_mask_row, &fresh_mask_tail, 0x23);
        let handoff = FoldHandoff {
            code: shape.code,
            mask_code: shape.mask_code,
            main_claim: main_claim.clone(),
            mask_claim: mask_claim.clone(),
            target,
            fresh_claim: fresh_claim.clone(),
            group: GROUP,
            k: 2,
            h: 0,
            members: vec![MEMBER0],
            gamma: gamma.clone(),
            eps,
            sum,
            fresh: FRESH0,
            fresh_root: fresh_tree.root,
            tree_number: fresh_plan.tree_number,
            kept_factor,
            kept_rows: 0,
            layout: layout.clone(),
        };
        let basis = FoldBasis {
            layout: layout.clone(),
            fresh_root: fresh_tree.root,
            fresh_tree: fresh_plan,
            members: vec![FoldMember {
                oracle: MEMBER0,
                main_code: shape.code,
                mask_code: shape.mask_code,
                root: *tree.root(),
                tree: member_plan(),
            }],
        };
        let rows = SingleRows {
            code: shape.code,
            mask_code: shape.mask_code,
            main: main_row,
            main_tail: vec![main_tail],
            masks: vec![old_mask_row, fresh_mask_row],
            mask_tails: vec![old_mask_tail, fresh_mask_tail],
            claim: main_claim,
            mask_claims: vec![mask_claim, fresh_claim],
            target,
        };
        rows.check().unwrap();
        let prover = IoppStageZero {
            handoff: handoff.clone(),
            basis: basis.clone(),
            rows,
            member_trees: vec![Arc::new(tree)],
            old_mask_tree: Some(old_tree),
            fresh_tree,
        };
        let verifier = IoppVerifyStageZero {
            handoff,
            basis,
            first_old_mask: Some((old_plan, old_root)),
        };
        (prover, verifier)
    }

    /// Stage 0 whose members carry no mask rows: the trusted layout makes the
    /// old mask the zero vector, so no dedicated old mask tree is supplied.
    fn zero_mask_stage_zero() -> (IoppStageZero, IoppVerifyStageZero) {
        let shape = FoldShape {
            code: code(5, 8, 1),
            mask_code: code(5, 16, 1),
            k: 2,
            h: 0,
            mask_leaf_class: IOPP_MASK_LEAF_CLASS,
        };
        let layout = FoldGroupLayout {
            group: GROUP,
            k: 2,
            h: 0,
            members: vec![FoldMemberLayout {
                oracle: MEMBER0,
                main_rows: 4,
                mask_rows: 0,
                kept: false,
            }],
        };
        assert!(layout.fault(&shape).is_none());
        let gamma = vec![Fp3::from_u64(3), Fp3::from_u64(5)];
        let eps = Fp3::from_u64(7);
        let main_claim = (1..=8).map(Fp3::from_u64).collect::<Vec<_>>();
        let mask_claim = vec![Fp3::ZERO; 16];
        let fresh_claim = fold_mask_weights(2, 0, 16, &gamma).unwrap();
        let old_mask_row = vec![Fp3::ZERO; 16];
        let fresh_mask_row = (0..16)
            .map(|index| Fp3::from_u64(3000 + index))
            .collect::<Vec<_>>();
        let old_mask_tail = vec![Fp3::ZERO];
        let fresh_mask_tail = vec![Fp3::from_u64(52)];
        let member_main = (0..4)
            .map(|row| {
                (0..8)
                    .map(|column| Fp3::from_u64((row * 8 + column + 1) as u64))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let member_tails = (0..4)
            .map(|row| vec![Fp3::from_u64(900 + row as u64)])
            .collect::<Vec<_>>();
        let weights = fold_coefficients(2, &gamma).unwrap();
        let mut main_row = vec![Fp3::ZERO; 8];
        for (weight, row) in weights.iter().zip(&member_main) {
            for (value, entry) in main_row.iter_mut().zip(row) {
                *value = value.add(&weight.mul(entry));
            }
        }
        let mut main_tail = Fp3::ZERO;
        for (weight, tail) in weights.iter().zip(&member_tails) {
            main_tail = main_tail.add(&weight.mul(&tail[0]));
        }
        let mut target = Fp3::ZERO;
        for (coefficient, value) in main_claim.iter().zip(&main_row) {
            target = target.add(&coefficient.mul(value));
        }
        for (coefficient, value) in mask_claim.iter().zip(&old_mask_row) {
            target = target.add(&coefficient.mul(value));
        }
        for (coefficient, value) in fresh_claim.iter().zip(&fresh_mask_row) {
            target = target.add(&coefficient.mul(value));
        }
        let mut kept_factor = eps;
        for challenge in &gamma {
            kept_factor = kept_factor.mul(&Fp3::ONE.sub(challenge));
        }
        let sum = Fp3::from_u64(17);
        let member_plan = RuntimeTreePlan {
            parts: vec![RuntimeTreePart {
                main_symbols: 12,
                mask_symbols: 0,
            }],
            ..member_plan()
        };
        let mut main = Vec::with_capacity(member_main.len());
        for (row, tail) in member_main.iter().zip(&member_tails) {
            main.push(encode_row_fp3(&code(5, 8, 1), row, tail).unwrap());
        }
        let tree = RuntimeTree::commit(
            member_plan.clone(),
            &[RuntimePartRows {
                main,
                mask: Vec::new(),
                main_deg: E_DEGREE as u32,
            }],
            &salts(32, 0x11),
        )
        .unwrap();
        let fresh_plan = mask_plan(0x480, 0xa1);
        let fresh_tree = mask_tree(
            fresh_plan,
            &shape,
            &fresh_mask_row,
            &fresh_mask_tail,
            0x23,
        );
        let handoff = FoldHandoff {
            code: shape.code,
            mask_code: shape.mask_code,
            main_claim: main_claim.clone(),
            mask_claim: mask_claim.clone(),
            target,
            fresh_claim: fresh_claim.clone(),
            group: GROUP,
            k: 2,
            h: 0,
            members: vec![MEMBER0],
            gamma: gamma.clone(),
            eps,
            sum,
            fresh: FRESH0,
            fresh_root: fresh_tree.root,
            tree_number: fresh_plan.tree_number,
            kept_factor,
            kept_rows: 0,
            layout: layout.clone(),
        };
        let basis = FoldBasis {
            layout: layout.clone(),
            fresh_root: fresh_tree.root,
            fresh_tree: fresh_plan,
            members: vec![FoldMember {
                oracle: MEMBER0,
                main_code: shape.code,
                mask_code: shape.mask_code,
                root: *tree.root(),
                tree: member_plan,
            }],
        };
        let rows = SingleRows {
            code: shape.code,
            mask_code: shape.mask_code,
            main: main_row,
            main_tail: vec![main_tail],
            masks: vec![old_mask_row, fresh_mask_row],
            mask_tails: vec![old_mask_tail, fresh_mask_tail],
            claim: main_claim,
            mask_claims: vec![mask_claim, fresh_claim],
            target,
        };
        rows.check().unwrap();
        let prover = IoppStageZero {
            handoff: handoff.clone(),
            basis: basis.clone(),
            rows,
            member_trees: vec![Arc::new(tree)],
            old_mask_tree: None,
            fresh_tree,
        };
        let verifier = IoppVerifyStageZero {
            handoff,
            basis,
            first_old_mask: None,
        };
        (prover, verifier)
    }

    fn switches(chain_plan: &IoppChain) -> Vec<IoppSwitchPrivate> {
        chain_plan
            .stages
            .iter()
            .skip(1)
            .map(|stage| {
                let target = stage.target.unwrap();
                let tree = stage.tree.as_ref().unwrap();
                IoppSwitchPrivate {
                    main_tails: (0..target.main_rows as usize)
                        .map(|row| {
                            (0..target.code.r as usize)
                                .map(|at| Fp3::from_u64(400 + (row * 8 + at) as u64))
                                .collect()
                        })
                        .collect(),
                    mask_tails: (0..target.mask_rows as usize)
                        .map(|row| {
                            (0..target.mask_code.r as usize)
                                .map(|at| Fp3::from_u64(500 + (row * 8 + at) as u64))
                                .collect()
                        })
                        .collect(),
                    pad: Fp3::from_u64(31),
                    salts: salts(tree.columns(), 0x33),
                }
            })
            .collect()
    }

    fn stage_folds(chain_plan: &IoppChain) -> Vec<IoppStageFoldPrivate> {
        chain_plan
            .stages
            .iter()
            .filter_map(|stage| stage.fold.as_ref())
            .map(|fold| IoppStageFoldPrivate {
                seed: Digest([0xc1; 32]),
                salts: salts(fold.mask_tree.columns(), 0x44),
            })
            .collect()
    }

    fn base_private(chain_plan: &IoppChain) -> IoppBasePrivate {
        IoppBasePrivate {
            salts: salts(chain_plan.config.base_star.columns(), 0x55),
            randomness: LastBaseRandomness::Fixed(Digest([0xd1; 32])),
        }
    }

    fn chain() -> Chain {
        Chain::begin(
            CHAIN_LABEL,
            &CHAIN_PARAMETERS,
            b"multi-stage iopp fixture",
            b"",
            None,
        )
        .unwrap()
    }

    fn run_proof() -> (IoppChain, IoppProof, Digest, IoppVerifyStageZero) {
        let plan = IoppChain::derive(&stages(), &config()).unwrap();
        let (zero, verify_zero) = stage_zero();
        let (proof, seal) = prove_multi_stage_iopp(
            chain(),
            &plan,
            zero,
            &switches(&plan),
            &stage_folds(&plan),
            &base_private(&plan),
        )
        .unwrap();
        (plan, proof, seal, verify_zero)
    }

    #[test]
    fn fast_fixture_round_trips() {
        let (plan, proof, seal, zero) = run_proof();
        assert_eq!(proof.switches.len(), 2);
        assert!(proof.switches[0].stage_fold.is_some());
        assert!(proof.switches[1].stage_fold.is_none());
        let verified = verify_multi_stage_iopp(chain(), &plan, zero, &proof).unwrap();
        assert_eq!(verified, seal);
    }

    #[test]
    fn proof_codec_round_trips_against_the_plan() {
        let (plan, proof, seal, zero) = run_proof();
        let bytes = proof.encode();
        let decoded = IoppProof::decode(&plan, &bytes).unwrap();
        assert_eq!(decoded, proof);
        let verified = verify_multi_stage_iopp(chain(), &plan, zero, &decoded).unwrap();
        assert_eq!(verified, seal);
    }

    #[test]
    fn proof_codec_rejects_truncated_and_trailing_bytes() {
        let (plan, proof, _, _) = run_proof();
        let bytes = proof.encode();
        assert!(IoppProof::decode(&plan, &bytes[..bytes.len() - 1]).is_err());
        let mut trailing = bytes;
        trailing.push(0);
        assert!(IoppProof::decode(&plan, &trailing).is_err());
    }

    #[test]
    fn proof_codec_rejects_missing_and_reordered_stages() {
        let (plan, proof, _, _) = run_proof();
        let mut missing = proof.clone();
        missing.switches.pop();
        assert!(IoppProof::decode(&plan, &missing.encode()).is_err());
        let mut reordered = proof;
        reordered.switches.swap(0, 1);
        assert!(IoppProof::decode(&plan, &reordered.encode()).is_err());
    }

    #[test]
    fn proof_codec_rejects_a_target_that_differs_from_the_plan() {
        let (plan, proof, _, _) = run_proof();
        let mut bytes = proof.encode();
        // The encoded proof starts with its switch count; the first target
        // follows. Flipping an answer-count bit makes the decoded target
        // differ from the trusted one.
        bytes[50] ^= 1;
        assert!(IoppProof::decode(&plan, &bytes).is_err());
    }

    #[test]
    fn last_switch_work_comes_from_the_previous_entry() {
        let (_, proof, _, _) = run_proof();
        let mut list = stages();
        // The last entry describes the BASE target, not an old stage: its work
        // bits never enter a switch replay.
        list[2].work_bits = 3;
        let plan = IoppChain::derive(&list, &config()).unwrap();
        assert!(verify_multi_stage_iopp(chain(), &plan, stage_zero().1, &proof).is_ok());
    }

    #[test]
    fn last_switch_positions_come_from_the_previous_entry() {
        let (_, proof, _, _) = run_proof();
        let mut list = stages();
        // Entry 1 supplies the last switch's old-row count; shrinking it must
        // break the replay even though entry 2 still matches the BASE count.
        list[1].positions = 1;
        let plan = IoppChain::derive(&list, &config()).unwrap();
        assert!(verify_multi_stage_iopp(chain(), &plan, stage_zero().1, &proof).is_err());
    }

    #[test]
    fn a_proof_from_another_chain_is_rejected() {
        let (plan, proof, _seal, zero) = run_proof();
        let other = Chain::begin(
            CHAIN_LABEL + 1,
            &CHAIN_PARAMETERS,
            b"multi-stage iopp fixture",
            b"",
            None,
        )
        .unwrap();
        assert!(verify_multi_stage_iopp(other, &plan, zero, &proof).is_err());
    }

    #[test]
    fn nonzero_kept_rows_are_rejected_by_the_verifier() {
        let (plan, proof, _seal, mut zero) = run_proof();
        zero.handoff.kept_rows = 1;
        assert!(verify_multi_stage_iopp(chain(), &plan, zero, &proof).is_err());
    }

    #[test]
    fn a_tampered_last_base_is_rejected() {
        let (plan, proof, _seal, zero) = run_proof();
        let mut tampered = proof.clone();
        let last = tampered.base.len() - 1;
        tampered.base[last] ^= 1;
        assert!(verify_multi_stage_iopp(chain(), &plan, zero, &tampered).is_err());
    }

    #[test]
    fn zero_old_mask_round_trips_without_a_tree() {
        let plan = IoppChain::derive(&stages(), &config()).unwrap();
        let (zero, verify_zero) = zero_mask_stage_zero();
        let (proof, seal) = prove_multi_stage_iopp(
            chain(),
            &plan,
            zero,
            &switches(&plan),
            &stage_folds(&plan),
            &base_private(&plan),
        )
        .unwrap();
        assert!(proof.switches[0].old_mask_opening.is_none());
        assert_eq!(
            verify_multi_stage_iopp(chain(), &plan, verify_zero, &proof).unwrap(),
            seal
        );
    }

    #[test]
    fn nonzero_fold_mask_claim_without_members_is_rejected() {
        let plan = IoppChain::derive(&stages(), &config()).unwrap();
        let (mut zero, _) = zero_mask_stage_zero();
        zero.handoff.mask_claim[0] = Fp3::from_u64(9);
        assert!(prove_multi_stage_iopp(
            chain(),
            &plan,
            zero,
            &switches(&plan),
            &stage_folds(&plan),
            &base_private(&plan),
        )
        .is_err());
    }

    #[test]
    fn verifier_derives_the_zero_old_mask_from_the_layout() {
        let plan = IoppChain::derive(&stages(), &config()).unwrap();
        let (_, proof, _, _) = run_proof();
        let (_, mut verify_zero) = zero_mask_stage_zero();
        // A handoff that declares a nonzero mask claim where the trusted
        // layout has no member mask rows is rejected before any transcript
        // value is used.
        verify_zero.handoff.mask_claim[0] = Fp3::from_u64(9);
        assert!(verify_multi_stage_iopp(chain(), &plan, verify_zero, &proof).is_err());
    }

    #[test]
    fn fresh_mask_tree_cannot_stand_in_for_the_old_mask() {
        let plan = IoppChain::derive(&stages(), &config()).unwrap();
        let (_, proof, _, mut zero) = run_proof();
        let (fresh_plan, fresh_root) = (zero.basis.fresh_tree, zero.basis.fresh_root);
        zero.first_old_mask = Some((fresh_plan, fresh_root));
        assert!(verify_multi_stage_iopp(chain(), &plan, zero, &proof).is_err());
    }

    #[test]
    fn missing_stage_is_rejected() {
        let (plan, mut proof, _, zero) = run_proof();
        proof.switches.pop();
        assert!(verify_multi_stage_iopp(chain(), &plan, zero, &proof).is_err());
    }

    #[test]
    fn reordered_stage_is_rejected() {
        let (plan, mut proof, _, zero) = run_proof();
        proof.switches.swap(0, 1);
        assert!(verify_multi_stage_iopp(chain(), &plan, zero, &proof).is_err());
    }

    #[test]
    fn wrong_old_mask_root_is_rejected() {
        let (plan, proof, _, mut zero) = run_proof();
        let (old_plan, root) = zero.first_old_mask.take().unwrap();
        zero.first_old_mask = Some((old_plan, Digest([0xee; 32])));
        assert_ne!(root, Digest([0xee; 32]));
        assert!(verify_multi_stage_iopp(chain(), &plan, zero, &proof).is_err());
    }

    #[test]
    fn wrong_old_opening_is_rejected() {
        let (plan, mut proof, _, zero) = run_proof();
        proof.switches[0]
            .old_mask_opening
            .as_mut()
            .unwrap()
            .columns[0]
            .symbols[0] = fp_from_u64(999);
        assert!(verify_multi_stage_iopp(chain(), &plan, zero, &proof).is_err());
    }

    #[test]
    fn wrong_fresh_opening_is_rejected() {
        let (plan, mut proof, _, zero) = run_proof();
        proof.switches[0].fresh_opening.columns[0].symbols[0] = fp_from_u64(999);
        assert!(verify_multi_stage_iopp(chain(), &plan, zero, &proof).is_err());
    }

    #[test]
    fn wrong_mask_room_is_rejected_by_the_plan() {
        let mut list = stages();
        list[1].mask = code(6, 16, 1);
        assert!(IoppChain::derive(&list, &config()).is_err());
    }

    #[test]
    fn wrong_leaf_class_is_rejected_by_the_stage_fold() {
        let (_, proof, _, zero) = run_proof();
        let mut bad = config();
        bad.mask_leaf_class = 4;
        let plan = IoppChain::derive(&stages(), &bad).unwrap();
        assert!(verify_multi_stage_iopp(chain(), &plan, zero, &proof).is_err());
    }

    #[test]
    fn duplicate_stage_identifiers_are_rejected() {
        let base = IoppStageFoldIds {
            group: GROUP + 1,
            member: MEMBER1,
            fresh: FRESH1,
        };
        assert!(!fold_ids_fault(&[base]));
        assert!(fold_ids_fault(&[
            base,
            IoppStageFoldIds {
                fresh: base.fresh,
                ..base
            }
        ]));
        assert!(fold_ids_fault(&[
            base,
            IoppStageFoldIds {
                member: base.member,
                ..base
            }
        ]));
        assert!(fold_ids_fault(&[
            base,
            IoppStageFoldIds {
                group: base.group,
                ..base
            }
        ]));
        assert!(fold_ids_fault(&[IoppStageFoldIds {
            fresh: base.member,
            ..base
        }]));
    }

    #[test]
    fn stage_fold_identifiers_must_not_reuse_stage_zero_identifiers() {
        let (_, proof, _, zero) = run_proof();
        let mut bad = config();
        bad.fold_ids[0].member = MEMBER0;
        let plan = IoppChain::derive(&stages(), &bad).unwrap();
        assert!(verify_multi_stage_iopp(chain(), &plan, zero, &proof).is_err());
    }

    #[test]
    fn stage_fold_group_must_not_reuse_the_stage_zero_group() {
        let (_, proof, _, zero) = run_proof();
        let mut bad = config();
        bad.fold_ids[0].group = GROUP;
        let plan = IoppChain::derive(&stages(), &bad).unwrap();
        assert!(verify_multi_stage_iopp(chain(), &plan, zero, &proof).is_err());
    }

    #[test]
    fn actual_stage_lists_derive() {
        let shared = vec![
            stage_plan(code(17, 65536, 8192), code(17, 512, 256), 2, 0, 2),
            stage_plan(code(17, 65536, 256), code(17, 16384, 256), 0, 0, 2),
        ];
        let shared_config = IoppChainConfig {
            first_main: code(17, 65_536, 8_192),
            first_mask: code(17, 512, 256),
            base_positions: 2,
            fold_ids: Vec::new(),
            switch_tree_numbers: vec![0x710],
            fold_tree_numbers: Vec::new(),
            stage_seeds: vec![
                config().stage_seeds[0],
                config().stage_seeds[0],
            ],
            base_star: RuntimeTreePlan {
                tree_number: 0x700,
                leaf_class: 5,
                log_n: 17,
                log_n_zk: 17,
                parts: vec![RuntimeTreePart {
                    main_symbols: E_DEGREE as u64,
                    mask_symbols: E_DEGREE as u64,
                }],
                salt_bytes: SALT_BYTES,
                run: BASE_RUN,
                configuration: CONFIGURATION,
            },
            ..config()
        };
        assert!(IoppChain::derive(&shared, &shared_config).is_ok());
        let own = vec![
            stage_plan(code(17, 65536, 8192), code(17, 512, 256), 14, 2, 2),
            stage_plan(code(17, 8, 256), code(17, 16, 256), 13, 13, 2),
            stage_plan(code(17, 8, 256), code(17, 1024, 256), 0, 0, 2),
        ];
        let own_config = IoppChainConfig {
            first_main: code(17, 65_536, 8_192),
            first_mask: code(17, 512, 256),
            base_positions: 2,
            fold_ids: vec![IoppStageFoldIds {
                group: GROUP + 1,
                member: MEMBER1,
                fresh: FRESH1,
            }],
            stage_seeds: config().stage_seeds,
            base_star: RuntimeTreePlan {
                tree_number: 0x701,
                leaf_class: 5,
                log_n: 17,
                log_n_zk: 17,
                parts: vec![RuntimeTreePart {
                    main_symbols: E_DEGREE as u64,
                    mask_symbols: E_DEGREE as u64,
                }],
                salt_bytes: SALT_BYTES,
                run: BASE_RUN,
                configuration: CONFIGURATION,
            },
            ..config()
        };
        assert!(IoppChain::derive(&own, &own_config).is_ok());
    }

    #[test]
    fn stage_derivation_values_are_per_stage() {
        let plan = IoppChain::derive(&stages(), &config()).unwrap();
        let mut changed = config();
        changed.stage_seeds[1].switch_run = Digest([0xd1; 32]);
        changed.stage_seeds[1].mask_run = Digest([0xd2; 32]);
        let changed = IoppChain::derive(&stages(), &changed).unwrap();
        assert_ne!(
            plan.stages[1].tree.as_ref().unwrap().run,
            changed.stages[1].tree.as_ref().unwrap().run
        );
        assert_ne!(
            plan.stages[1].fold.as_ref().unwrap().mask_tree.run,
            changed.stages[1].fold.as_ref().unwrap().mask_tree.run
        );
        assert_eq!(
            plan.stages[2].tree.as_ref().unwrap().run,
            changed.stages[2].tree.as_ref().unwrap().run
        );
    }

    #[test]
    fn stage_derivation_values_must_cover_every_stage_and_be_nonzero() {
        let mut short = config();
        short.stage_seeds.pop();
        assert!(IoppChain::derive(&stages(), &short).is_err());
        let mut zero = config();
        zero.stage_seeds[2].configuration = Digest::ZERO;
        assert!(IoppChain::derive(&stages(), &zero).is_err());
    }
}
