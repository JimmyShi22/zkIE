//! The consumer first stage's proximity group: members, samples, plan and run.
//!
//! The group takes the committed first-stage messages (raw and bit messages per
//! head, the mask message and the request-level W) and the linear relation rows
//! of the first stage, and runs the repository's proximity test over them: the
//! roots are absorbed first, one sample seed per member is drawn, the sample
//! answers are absorbed, and only then the group's own challenges are drawn by
//! the proximity prover. The linear rows and the sample rows are checked by the
//! BAT/FOLD/CS/BASE engine; the chain value after the group is the real
//! first-stage seal.
//!
//! Every member message is committed under the first codes the plan derives
//! from the entry and row counts, so a message committed under other codes is
//! refused. Identifiers, tree numbers and tags are fixed in the plan before any
//! challenge; the dependency order (roots, seeds, answers, then the group's own
//! challenges) is the engine's own and is never reordered.

use std::sync::Arc;

use zkie_core::common::codec::DecodeLimits;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::bat::BatRow;
use zkie_engine::pcs::request::chain::Chain;
use zkie_engine::pcs::request::code::{encode_row_fp3, CodeParams};
use zkie_engine::pcs::request::field::{Fp3, E_DEGREE};
use zkie_engine::pcs::request::gu::GuOracleShape;
use zkie_engine::pcs::request::relation::{
    oracle_sample_row, sample_cube_of, GuRelationKind, GuRelationRow, GuWitness,
};
use zkie_engine::pcs::request::runtime_tree::{
    MaskTreePlan, RuntimePartRows, RuntimeTree, RuntimeTreePlan,
};
use zkie_engine::pcs::request::table::SampleSeed;
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};

use crate::proof::hidden_send_session::{
    decode_proximity_messages, encode_proximity_messages, member_fold_shape, proximity_prove,
    proximity_verify, IoppStagePlan, ProximityMember, ProximityMessages, ProximityProof,
    ProximityTreeSource, WShape,
};
use crate::proof::multi_stage_iopp::IOPP_MASK_LEAF_CLASS;
use crate::proof::output_closure::CommittedValueGroup;
use crate::proof::proximity_iopp::{
    derive_proximity_iopp, IoppBoundIds, ProximityGroupRole, ProximityIoppPlan,
};
use crate::proof::proximity_planner::{
    derive_single_group_plan_with_extra_mask_rows, lin_group_bits, rows_for_columns,
};

/// Version of the consumer group messages.
pub const CONSUMER_GROUP_VERSION: u64 = 1;
/// Transcript tag of the group root round.
const CONSUMER_GROUP_ROOT_TAG: u64 = 120;
/// Transcript tag of the group answer round.
const CONSUMER_GROUP_ANSWER_TAG: u64 = 121;

/// The fixed entry counts the group plan derives from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsumerGroupEntries {
    pub heads: u32,
    pub raw_columns: u64,
    pub bits_columns: u64,
    pub w_columns: u64,
    /// The number of mask entries the group rows address.
    pub mask_entries: u64,
    /// The number of linear relation rows.
    pub rows: u64,
}

impl ConsumerGroupEntries {
    pub fn fault(&self) -> Option<&'static str> {
        if self.heads == 0
            || self.raw_columns == 0
            || self.bits_columns == 0
            || self.w_columns == 0
            || self.mask_entries == 0
            || self.rows == 0
        {
            return Some("a consumer group has an empty entry count");
        }
        None
    }
}

/// The derived group geometry: the first codes, the mask tree and the member
/// shapes. Every committed message of the group must use exactly these shapes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerGroupShapes {
    pub entries: ConsumerGroupEntries,
    pub first_main: CodeParams,
    pub first_mask: CodeParams,
    pub mask_rows: u64,
    pub stages: Vec<IoppStagePlan>,
    pub base_positions: usize,
    pub raw_shape: WShape,
    pub bits_shape: WShape,
    pub w_shape: WShape,
    pub mask_tree: MaskTreePlan,
}

fn stage_plan(stage: &crate::proof::proximity_planner::SolvedStage) -> IoppStagePlan {
    IoppStagePlan {
        main: CodeParams {
            log_n: stage.log_n,
            ell: stage.main_ell,
            r: stage.main_r,
        },
        mask: CodeParams {
            log_n: stage.mask_log_n,
            ell: stage.mask_ell,
            r: stage.mask_r,
        },
        k: stage.k,
        h: stage.h,
        positions: stage.positions as usize,
        samples: 1,
        work_bits: 0,
    }
}

/// The members' main and mask row counts under one first main code: raw and
/// bits messages one per head, the W message, and the mask-only member.
fn member_rows(
    entries: ConsumerGroupEntries,
    ell: u64,
    mask_rows: u64,
) -> Result<(Vec<u64>, Vec<u64>)> {
    let mut main = Vec::with_capacity(2 * entries.heads as usize + 2);
    let mut mask = Vec::with_capacity(2 * entries.heads as usize + 2);
    for _ in 0..entries.heads {
        main.push(rows_for_columns(entries.raw_columns, ell, entries.heads)?);
        mask.push(0);
    }
    for _ in 0..entries.heads {
        main.push(rows_for_columns(entries.bits_columns, ell, entries.heads)?);
        mask.push(0);
    }
    main.push(rows_for_columns(entries.w_columns, ell, 1)?);
    mask.push(0);
    main.push(0);
    mask.push(mask_rows);
    Ok((main, mask))
}

/// Derives the group geometry from the entry counts.
pub fn derive_consumer_group_shapes(
    entries: ConsumerGroupEntries,
    run: Digest,
    configuration: Digest,
    tree_base: u64,
    mask_tree_number: u64,
) -> Result<ConsumerGroupShapes> {
    entries
        .fault()
        .map_or(Ok(()), |_| Err(Error::InvalidConfig))?;
    let mask_entries = usize::try_from(entries.mask_entries).map_err(|_| Error::ResourceLimit)?;
    let derive = |bit_entries: u64| {
        derive_single_group_plan_with_extra_mask_rows(
            bit_entries,
            mask_entries,
            0,
            entries.rows,
            1,
            false,
        )
        .map_err(|_| Error::InvalidConfig)
    };
    // The single-group solver takes one main-table row count, while the group
    // joins one main table per member (raw and bits per head, and W). The
    // solver's row input must be the power-of-two row count that holds the
    // members' aligned main blocks, so the derived group bits equal the bits
    // of the real BAT layout. The fixed point below feeds the solver that
    // aggregate until the derived first code reproduces it exactly.
    let mut bit_entries = entries.bits_columns;
    let mut derived = derive(bit_entries)?;
    for _ in 0..8 {
        let (main_rows, mask_rows) =
            member_rows(entries, derived.first_main.ell, derived.mask_rows)?;
        let (k, _h) = lin_group_bits(&main_rows, &mask_rows)?;
        let (_, planner_h) = lin_group_bits(&[1], &[derived.mask_rows])?;
        let required = 1u64
            .checked_shl(k.max(planner_h).max(1))
            .ok_or(Error::ResourceLimit)?;
        let capacity = derived
            .first_main
            .ell
            .checked_sub(1)
            .ok_or(Error::ResourceLimit)?;
        let target = required.checked_mul(capacity).ok_or(Error::ResourceLimit)?;
        if target == bit_entries {
            break;
        }
        bit_entries = target;
        derived = derive(bit_entries)?;
    }
    let (main_rows, mask_rows) = member_rows(entries, derived.first_main.ell, derived.mask_rows)?;
    let (k_actual, h_actual) = lin_group_bits(&main_rows, &mask_rows)?;
    let first = derived.stages.first().ok_or(Error::InvalidConfig)?;
    if first.k != k_actual || first.h != h_actual {
        return Err(Error::InvalidConfig);
    }
    let stages: Vec<IoppStagePlan> = derived.stages.iter().map(stage_plan).collect();
    let base_positions = stages.last().ok_or(Error::InvalidConfig)?.positions;
    let raw_shape = WShape::new(
        entries.raw_columns,
        derived.first_main,
        1,
        entries.heads,
        40,
        tree_base,
        1,
        run,
        configuration,
    )?;
    let bits_shape = WShape::new(
        entries.bits_columns,
        derived.first_main,
        1,
        entries.heads,
        40,
        tree_base
            .checked_add(entries.heads as u64)
            .ok_or(Error::ResourceLimit)?,
        1,
        run,
        configuration,
    )?;
    let w_shape = WShape::new(
        entries.w_columns,
        derived.first_main,
        1,
        1,
        40,
        tree_base
            .checked_add(2 * entries.heads as u64)
            .ok_or(Error::ResourceLimit)?,
        1,
        run,
        configuration,
    )?;
    let mask_tree = MaskTreePlan {
        tree_number: mask_tree_number,
        leaf_class: IOPP_MASK_LEAF_CLASS,
        log_n: derived.first_mask.log_n,
        mask_rows: derived.mask_rows,
        salt_bytes: 40,
        run,
        configuration,
    };
    if mask_tree.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    let shapes = ConsumerGroupShapes {
        entries,
        first_main: derived.first_main,
        first_mask: derived.first_mask,
        mask_rows: derived.mask_rows,
        stages,
        base_positions,
        raw_shape,
        bits_shape,
        w_shape,
        mask_tree,
    };
    shapes
        .check_members()
        .map_or(Ok(()), |_| Err(Error::InvalidConfig))?;
    Ok(shapes)
}

impl ConsumerGroupShapes {
    pub fn check_members(&self) -> Option<&'static str> {
        if self
            .raw_shape
            .fault(self.entries.raw_columns, self.entries.heads)
            .is_some()
            || self
                .bits_shape
                .fault(self.entries.bits_columns, self.entries.heads)
                .is_some()
            || self.w_shape.fault(self.entries.w_columns, 1).is_some()
        {
            return Some("a consumer group member shape does not fit the first code");
        }
        if self.raw_shape.code != self.first_main
            || self.bits_shape.code != self.first_main
            || self.w_shape.code != self.first_main
        {
            return Some("a consumer group member is not committed under the first main code");
        }
        None
    }

    /// The number of mask message entries (rows times the mask code width).
    pub fn mask_message_entries(&self) -> Result<u64> {
        self.mask_rows
            .checked_mul(self.first_mask.ell)
            .ok_or(Error::ResourceLimit)
    }
}

/// The fixed identifiers and roots of one group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerGroupPlan {
    pub shapes: ConsumerGroupShapes,
    pub group_id: u64,
    pub fresh: u64,
    pub raw_oracle_base: u64,
    pub bits_oracle_base: u64,
    pub mask_oracle: u64,
    pub w_oracle: u64,
    pub raw_bat_base: u64,
    pub bits_bat_base: u64,
    pub mask_bat: u64,
    pub w_bat: u64,
    pub mask_root: Digest,
    pub raw_roots: Vec<Digest>,
    pub bits_roots: Vec<Digest>,
    pub w_roots: Vec<Digest>,
}

impl ConsumerGroupPlan {
    pub fn fault(&self) -> Option<&'static str> {
        if let Some(fault) = self.shapes.check_members() {
            return Some(fault);
        }
        let heads = self.shapes.entries.heads as usize;
        if self.raw_roots.len() != heads
            || self.bits_roots.len() != heads
            || self.w_roots.len() != 1
            || self.mask_root == Digest::ZERO
            || self.raw_roots.iter().any(|root| *root == Digest::ZERO)
            || self.bits_roots.iter().any(|root| *root == Digest::ZERO)
            || self.w_roots.iter().any(|root| *root == Digest::ZERO)
        {
            return Some("a consumer group root is missing");
        }
        let heads = u64::from(self.shapes.entries.heads);
        let contiguous = (self.raw_bat_base == 0 && self.bits_bat_base == heads)
            || (self.bits_bat_base == 0 && self.raw_bat_base == heads);
        if !contiguous || self.mask_bat != 2 * heads || self.w_bat != 2 * heads + 1 {
            return Some("the BAT member identifiers are not the contiguous member order");
        }
        let mut ids = vec![
            self.group_id,
            self.fresh,
            self.raw_oracle_base,
            self.bits_oracle_base,
            self.mask_oracle,
            self.w_oracle,
            self.raw_bat_base,
            self.bits_bat_base,
            self.mask_bat,
            self.w_bat,
        ];
        let count = ids.len();
        ids.sort_unstable();
        ids.dedup();
        if ids.len() != count {
            return Some("a consumer group identifier is not distinct");
        }
        None
    }

    pub fn check(&self) -> Result<()> {
        self.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))
    }

    pub fn member_count(&self) -> usize {
        2 * self.shapes.entries.heads as usize + 2
    }

    pub fn iopp(&self) -> Result<ProximityIoppPlan> {
        let mut oracles = vec![self.group_id, self.fresh, self.mask_bat];
        let mut trees = vec![self.shapes.mask_tree.tree_number];
        for head in 0..self.shapes.entries.heads {
            oracles.push(self.raw_bat_base + head as u64);
            oracles.push(self.bits_bat_base + head as u64);
            trees.push(self.shapes.raw_shape.tree_number + head as u64);
            trees.push(self.shapes.bits_shape.tree_number + head as u64);
        }
        oracles.push(self.w_bat);
        trees.push(self.shapes.w_shape.tree_number);
        let bound = IoppBoundIds::new(oracles, trees);
        derive_proximity_iopp(
            &self.shapes.stages,
            ProximityGroupRole::Own,
            &self.shapes.mask_tree,
            self.group_id,
            self.fresh,
            self.shapes.base_positions,
            &bound,
        )
    }

    /// The group end to decode against the plan.
    pub fn proximity_plan(&self) -> Option<ProximityIoppPlan> {
        if self.shapes.stages.is_empty() {
            None
        } else {
            self.iopp().ok()
        }
    }
}

/// The committed mask-only message: its root and tree.
pub struct CommittedConsumerGroupMask {
    pub root: Digest,
    pub tree: Arc<RuntimeTree>,
}

/// Commits the mask-only message under the group's first mask code.
pub fn commit_consumer_group_mask(
    plan: &ConsumerGroupPlan,
    rows: Vec<Vec<Fp3>>,
    tails: Vec<Vec<Fp3>>,
    salts: Vec<Vec<u8>>,
) -> Result<CommittedConsumerGroupMask> {
    plan.shapes
        .check_members()
        .map_or(Ok(()), |_| Err(Error::InvalidConfig))?;
    let mask_rows = usize::try_from(plan.shapes.mask_rows).map_err(|_| Error::ResourceLimit)?;
    let tail_len = usize::try_from(plan.shapes.first_mask.r).map_err(|_| Error::ResourceLimit)?;
    let columns = usize::try_from(plan.shapes.first_mask.n()).map_err(|_| Error::ResourceLimit)?;
    if rows.len() != mask_rows
        || tails.len() != mask_rows
        || tails.iter().any(|tail| tail.len() != tail_len)
        || salts.len() != columns
    {
        return Err(Error::ShapeMismatch);
    }
    let mut codewords = Vec::with_capacity(mask_rows);
    for (row, tail) in rows.iter().zip(&tails) {
        codewords.push(encode_row_fp3(&plan.shapes.first_mask, row, tail)?);
    }
    let tree = RuntimeTree::commit(
        plan.shapes.mask_tree.runtime_plan()?,
        &[RuntimePartRows {
            main: Vec::new(),
            mask: codewords,
            main_deg: 1,
        }],
        &salts,
    )?;
    Ok(CommittedConsumerGroupMask {
        root: *tree.root(),
        tree: Arc::new(tree),
    })
}

/// The prover's committed messages, their encoding tails and the mask message.
pub struct ConsumerGroupWitness<'a> {
    pub raw: &'a CommittedValueGroup,
    pub raw_tails: &'a [Vec<Fp3>],
    pub bits: &'a CommittedValueGroup,
    pub bits_tails: &'a [Vec<Fp3>],
    pub w: &'a CommittedValueGroup,
    pub w_tails: &'a [Vec<Fp3>],
    pub mask: &'a CommittedConsumerGroupMask,
    pub mask_rows: Vec<Vec<Fp3>>,
    pub mask_tails: Vec<Vec<Fp3>>,
}

/// The plan's row count is the group's real BAT row count: the plain relation
/// rows plus one sample row per member. A group run whose actual rows differ
/// from the plan is refused before any challenge.
fn check_planned_rows(
    plan: &ConsumerGroupPlan,
    members: &[ProximityMember],
    plain_rows: usize,
) -> Result<()> {
    let samples: u64 = members.iter().map(|member| u64::from(member.answers)).sum();
    let actual = u64::try_from(plain_rows)
        .map_err(|_| Error::ResourceLimit)?
        .checked_add(samples)
        .ok_or(Error::ResourceLimit)?;
    if actual != plan.shapes.entries.rows {
        return Err(Error::StatementMismatch);
    }
    Ok(())
}

fn split_tails(flat: &[Fp3], rows: u64, tail_len: u64) -> Result<Vec<Vec<Fp3>>> {
    let rows = usize::try_from(rows).map_err(|_| Error::ResourceLimit)?;
    let tail_len = usize::try_from(tail_len).map_err(|_| Error::ResourceLimit)?;
    if flat.len() != rows.checked_mul(tail_len).ok_or(Error::ResourceLimit)? {
        return Err(Error::ShapeMismatch);
    }
    Ok(flat
        .chunks_exact(tail_len)
        .map(|tail| tail.to_vec())
        .collect())
}

fn make_member(
    bat_oracle: u64,
    source_oracle: u64,
    main_code: CodeParams,
    mask_code: CodeParams,
    main_rows: u64,
    mask_rows: u64,
    answers: u32,
    root: Digest,
    tree_plan: RuntimeTreePlan,
    message: Vec<Vec<Fp3>>,
    tails: Vec<Vec<Fp3>>,
    mask_message: Vec<Vec<Fp3>>,
    mask_tails: Vec<Vec<Fp3>>,
) -> ProximityMember {
    ProximityMember {
        bat_oracle,
        source_oracle,
        main_code,
        mask_code,
        main_rows,
        mask_rows,
        answers,
        root,
        tree_plan,
        message,
        tails,
        mask_message,
        mask_tails,
    }
}

fn members(
    plan: &ConsumerGroupPlan,
    witness: Option<&ConsumerGroupWitness<'_>>,
) -> Result<Vec<ProximityMember>> {
    plan.check()?;
    let heads = plan.shapes.entries.heads as usize;
    let shapes = &plan.shapes;
    let mask_tree = shapes.mask_tree.runtime_plan()?;
    let mut members = Vec::with_capacity(plan.member_count());
    // The isolated FOLD handoff places the members' main blocks in descending
    // block size, so the plan's own BAT order fixes which family is built
    // first: the family whose base is zero. Both sides read the same plan.
    let order = if plan.raw_bat_base < plan.bits_bat_base {
        [0u8, 1u8]
    } else {
        [1u8, 0u8]
    };
    for kind in order {
    if kind == 0 {
    for head in 0..heads {
        let (message, tails) = match witness {
            Some(witness) => (
                witness
                    .raw
                    .messages
                    .get(head)
                    .cloned()
                    .ok_or(Error::ShapeMismatch)?,
                split_tails(
                    witness.raw_tails.get(head).ok_or(Error::ShapeMismatch)?,
                    shapes.raw_shape.main_rows,
                    shapes.raw_shape.code.r,
                )?,
            ),
            None => (Vec::new(), Vec::new()),
        };
        members.push(make_member(
            plan.raw_bat_base + head as u64,
            plan.raw_oracle_base + head as u64,
            shapes.first_main,
            shapes.first_mask,
            shapes.raw_shape.main_rows,
            0,
            1,
            plan.raw_roots[head],
            shapes.raw_shape.tree_plan(head as u32)?,
            message,
            tails,
            Vec::new(),
            Vec::new(),
        ));
    }
    } else {
    for head in 0..heads {
        let (message, tails) = match witness {
            Some(witness) => (
                witness
                    .bits
                    .messages
                    .get(head)
                    .cloned()
                    .ok_or(Error::ShapeMismatch)?,
                split_tails(
                    witness.bits_tails.get(head).ok_or(Error::ShapeMismatch)?,
                    shapes.bits_shape.main_rows,
                    shapes.bits_shape.code.r,
                )?,
            ),
            None => (Vec::new(), Vec::new()),
        };
        members.push(make_member(
            plan.bits_bat_base + head as u64,
            plan.bits_oracle_base + head as u64,
            shapes.first_main,
            shapes.first_mask,
            shapes.bits_shape.main_rows,
            0,
            1,
            plan.bits_roots[head],
            shapes.bits_shape.tree_plan(head as u32)?,
            message,
            tails,
            Vec::new(),
            Vec::new(),
        ));
    }
    }
    }
    let (mask_message, mask_tails) = match witness {
        Some(witness) => (witness.mask_rows.clone(), witness.mask_tails.clone()),
        None => (Vec::new(), Vec::new()),
    };
    members.push(make_member(
        plan.mask_bat,
        plan.mask_oracle,
        shapes.first_main,
        shapes.first_mask,
        0,
        shapes.mask_rows,
        1,
        plan.mask_root,
        mask_tree,
        Vec::new(),
        Vec::new(),
        mask_message,
        mask_tails,
    ));
    let (message, tails) = match witness {
        Some(witness) => (
            witness
                .w
                .messages
                .first()
                .cloned()
                .ok_or(Error::ShapeMismatch)?,
            split_tails(
                witness.w_tails.first().ok_or(Error::ShapeMismatch)?,
                shapes.w_shape.main_rows,
                shapes.w_shape.code.r,
            )?,
        ),
        None => (Vec::new(), Vec::new()),
    };
    members.push(make_member(
        plan.w_bat,
        plan.w_oracle,
        shapes.first_main,
        shapes.first_mask,
        shapes.w_shape.main_rows,
        0,
        1,
        plan.w_roots[0],
        shapes.w_shape.tree_plan(0)?,
        message,
        tails,
        Vec::new(),
        Vec::new(),
    ));
    if members.len() != plan.member_count() {
        return Err(Error::InvalidConfig);
    }
    Ok(members)
}

fn oracle_shape(member: &ProximityMember) -> GuOracleShape {
    GuOracleShape {
        code: member.main_code,
        main_rows: member.main_rows,
        main_deg: E_DEGREE as u32,
        mask_rows: member.mask_rows,
        mask_code: if member.mask_rows > 0 {
            Some(member.mask_code)
        } else {
            None
        },
        answers: member.answers,
    }
}

fn absorb_roots(
    chain: &mut Chain,
    plan: &ConsumerGroupPlan,
    members: &[ProximityMember],
) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(CONSUMER_GROUP_VERSION);
    message.push_u64(CONSUMER_GROUP_ROOT_TAG);
    message.push_u64(plan.group_id);
    message.push_u64(plan.fresh);
    message.push_u64(members.len() as u64);
    for member in members {
        message.push_u64(member.bat_oracle);
        message.push_u64(member.source_oracle);
        message.push_u64(member.main_rows);
        message.push_u64(member.mask_rows);
        message.push_u64(member.answers as u64);
        message.push_u64(member.tree_plan.tree_number);
        for word in member.root.0.chunks_exact(8) {
            message.push_u64(u64::from_le_bytes(
                word.try_into().expect("eight-byte word"),
            ));
        }
    }
    let prefix = message.prefix(zkie_engine::pcs::request::chain::KIND_SEED, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn absorb_answers(chain: &mut Chain, answers: &[Fp3]) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(CONSUMER_GROUP_VERSION);
    message.push_u64(CONSUMER_GROUP_ANSWER_TAG);
    message.push_u64(answers.len() as u64);
    for value in answers {
        message.push_fp3(value);
    }
    let prefix = message.prefix(zkie_engine::pcs::request::chain::KIND_EMPTY, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn draw_seed(chain: &mut Chain, bits: u32) -> Result<SampleSeed> {
    let mut z = Vec::with_capacity(bits as usize);
    for _ in 0..bits {
        z.push(chain.draw_e()?);
    }
    let beta = chain.draw_e_nonzero()?;
    Ok(SampleSeed { z, beta })
}

fn eq_index(point: &[Fp3], index: u64, bits: u32) -> Result<Fp3> {
    if point.len() != bits as usize {
        return Err(Error::ShapeMismatch);
    }
    let mut value = Fp3::ONE;
    for (bit, coordinate) in point.iter().enumerate() {
        let chosen = if (index >> bit) & 1 == 1 {
            *coordinate
        } else {
            Fp3::ONE.sub(coordinate)
        };
        value = value.mul(&chosen);
    }
    Ok(value)
}

fn ceil_log2_u64(value: u64) -> u32 {
    if value <= 1 {
        0
    } else {
        64 - (value - 1).leading_zeros()
    }
}

/// The mask-table sample row at one drawn seed: the eq coefficients over the
/// mask cube, with the last entry scaled by beta, placed in the compact
/// row-major table. The consumer group and the second own group share this
/// one form.
pub(crate) fn mask_sample_row_at(
    oracle: u64,
    group: u64,
    mask_rows: u64,
    mask_ell: u64,
    seed: &SampleSeed,
    bits: u32,
    target: Fp3,
) -> Result<GuRelationRow> {
    if seed.z.len() < bits as usize || seed.beta == Fp3::ZERO {
        return Err(Error::ShapeMismatch);
    }
    let ell = usize::try_from(mask_ell).map_err(|_| Error::ResourceLimit)?;
    let mask_rows = usize::try_from(mask_rows).map_err(|_| Error::ResourceLimit)?;
    let row_bits = ceil_log2_u64(mask_rows as u64);
    let stride = 1usize.checked_shl(row_bits).ok_or(Error::ResourceLimit)?;
    let point = &seed.z[..bits as usize];
    let count = 1u64.checked_shl(bits).ok_or(Error::ResourceLimit)?;
    let mut mask = vec![Fp3::ZERO; mask_rows * ell];
    for index in 0..count {
        let row = (index & (stride as u64 - 1)) as usize;
        let place = (index >> row_bits) as usize;
        if row >= mask_rows || place >= ell {
            continue;
        }
        let coefficient = if row + 1 == mask_rows && place + 1 == ell {
            seed.beta
        } else {
            eq_index(point, index, bits)?
        };
        mask[row * ell + place] = coefficient;
    }
    Ok(GuRelationRow {
        oracle,
        group,
        kind: GuRelationKind::Sample,
        main: Vec::new(),
        mask,
        target,
    })
}

fn mask_sample_row(
    plan: &ConsumerGroupPlan,
    seed: &SampleSeed,
    bits: u32,
    target: Fp3,
) -> Result<GuRelationRow> {
    mask_sample_row_at(
        plan.mask_oracle,
        plan.group_id,
        plan.shapes.mask_rows,
        plan.shapes.first_mask.ell,
        seed,
        bits,
        target,
    )
}

fn sample_row(
    plan: &ConsumerGroupPlan,
    member: &ProximityMember,
    seed: &SampleSeed,
    bits: u32,
    target: Fp3,
) -> Result<GuRelationRow> {
    if member.mask_rows > 0 {
        mask_sample_row(plan, seed, bits, target)
    } else {
        oracle_sample_row(
            member.source_oracle,
            plan.group_id,
            &oracle_shape(member),
            std::slice::from_ref(seed),
            0,
            target,
        )
    }
}

/// The prover's sample rows: one row per member, its target the answer from the
/// member's own committed message.
fn prover_sample_rows(
    chain: &mut Chain,
    plan: &ConsumerGroupPlan,
    members: &[ProximityMember],
) -> Result<Vec<Vec<GuRelationRow>>> {
    let mut rows = Vec::with_capacity(members.len());
    for member in members {
        if member.answers != 1 {
            return Err(Error::InvalidConfig);
        }
        let bits = sample_cube_of(&oracle_shape(member))?.bits;
        let seed = draw_seed(chain, bits)?;
        let zero = sample_row(plan, member, &seed, bits, Fp3::ZERO)?;
        let witness = GuWitness {
            main: member.message.concat(),
            mask: member.mask_message.concat(),
        };
        let answer = zero.evaluate(&witness)?;
        let mut row = zero;
        row.target = answer;
        rows.push(vec![row]);
    }
    Ok(rows)
}

/// The verifier's sample rows: the same seeds and the proof's answers.
fn verifier_sample_rows(
    chain: &mut Chain,
    plan: &ConsumerGroupPlan,
    members: &[ProximityMember],
    answers: &[Fp3],
) -> Result<Vec<Vec<GuRelationRow>>> {
    if answers.len() != members.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut rows = Vec::with_capacity(members.len());
    for (member, answer) in members.iter().zip(answers) {
        let bits = sample_cube_of(&oracle_shape(member))?.bits;
        let seed = draw_seed(chain, bits)?;
        rows.push(vec![sample_row(plan, member, &seed, bits, *answer)?]);
    }
    Ok(rows)
}

/// The group proof message: the sample answers, the proximity messages and the
/// chain value after the group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerGroupProof {
    pub answers: Vec<Fp3>,
    pub proximity: Vec<u8>,
    pub group_seal: Digest,
}

impl ConsumerGroupProof {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(CONSUMER_GROUP_VERSION);
        writer.put_u64(self.answers.len() as u64);
        for value in &self.answers {
            writer.put_fp3(value);
        }
        writer.put_bytes(&self.proximity);
        writer.put_digest(&self.group_seal);
        Ok(writer.into_bytes())
    }

    pub fn decode(plan: &ConsumerGroupPlan, bytes: &[u8]) -> Result<Self> {
        plan.check()?;
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        if reader.get_u64()? != CONSUMER_GROUP_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if count != plan.member_count() {
            return Err(Error::ShapeMismatch);
        }
        let mut answers = Vec::with_capacity(count);
        for _ in 0..count {
            answers.push(reader.get_fp3()?);
        }
        let proximity = reader.get_bytes()?.to_vec();
        let group_seal = reader.get_digest()?;
        reader.finish()?;
        Ok(Self {
            answers,
            proximity,
            group_seal,
        })
    }

    pub fn proximity_messages(&self, plan: &ConsumerGroupPlan) -> Result<ProximityMessages> {
        let iopp = plan.proximity_plan();
        decode_proximity_messages(iopp.as_ref(), &self.proximity)
    }
}

/// Prover side of the group over the caller's first-stage chain.
pub fn prove_consumer_group(
    chain: Chain,
    plan: &ConsumerGroupPlan,
    witness: &ConsumerGroupWitness<'_>,
    plain_rows: Vec<BatRow>,
    run_seed: Digest,
) -> Result<(ConsumerGroupProof, Chain)> {
    plan.check()?;
    let members = members(plan, Some(witness))?;
    check_planned_rows(plan, &members, plain_rows.len())?;
    let mut chain = chain;
    absorb_roots(&mut chain, plan, &members)?;
    let sample_rows = prover_sample_rows(&mut chain, plan, &members)?;
    let answers: Vec<Fp3> = sample_rows
        .iter()
        .map(|rows| rows.first().map_or(Fp3::ZERO, |row| row.target))
        .collect();
    absorb_answers(&mut chain, &answers)?;
    let fold = member_fold_shape(
        plan.shapes.first_main,
        plan.shapes.first_mask,
        &members,
        plan.shapes.mask_tree.leaf_class,
    )?;
    let iopp = plan.iopp()?;
    let mut trees: Vec<(u64, Arc<RuntimeTree>)> = Vec::with_capacity(members.len());
    for head in 0..plan.shapes.entries.heads as usize {
        trees.push((
            plan.raw_oracle_base + head as u64,
            Arc::clone(witness.raw.trees.0.get(head).ok_or(Error::ShapeMismatch)?),
        ));
        trees.push((
            plan.bits_oracle_base + head as u64,
            Arc::clone(witness.bits.trees.0.get(head).ok_or(Error::ShapeMismatch)?),
        ));
    }
    trees.push((plan.mask_oracle, Arc::clone(&witness.mask.tree)));
    trees.push((
        plan.w_oracle,
        Arc::clone(witness.w.trees.0.first().ok_or(Error::ShapeMismatch)?),
    ));
    let proof = proximity_prove(
        chain,
        fold,
        members,
        ProximityTreeSource::Ready { trees, omega: None },
        &sample_rows,
        plain_rows,
        plan.fresh,
        plan.shapes.mask_tree,
        run_seed,
        plan.shapes.base_positions,
        Some(&iopp),
    )?;
    let ProximityProof {
        messages, chain, ..
    } = proof;
    let group_seal = *chain.value();
    Ok((
        ConsumerGroupProof {
            answers,
            proximity: encode_proximity_messages(&messages)?,
            group_seal,
        },
        chain,
    ))
}

/// Verifier side of the group over the caller's first-stage chain.
pub fn verify_consumer_group(
    chain: Chain,
    plan: &ConsumerGroupPlan,
    plain_rows: Vec<BatRow>,
    proof: &ConsumerGroupProof,
) -> Result<(Digest, Chain)> {
    plan.check()?;
    let members = members(plan, None)?;
    check_planned_rows(plan, &members, plain_rows.len())?;
    let mut chain = chain;
    absorb_roots(&mut chain, plan, &members)?;
    let sample_rows = verifier_sample_rows(&mut chain, plan, &members, &proof.answers)?;
    absorb_answers(&mut chain, &proof.answers)?;
    let fold = member_fold_shape(
        plan.shapes.first_main,
        plan.shapes.first_mask,
        &members,
        plan.shapes.mask_tree.leaf_class,
    )?;
    let iopp = plan.iopp()?;
    let messages = proof.proximity_messages(plan)?;
    let verification = proximity_verify(
        chain,
        fold,
        &members,
        &sample_rows,
        plain_rows,
        &messages,
        plan.fresh,
        plan.shapes.mask_tree,
        plan.shapes.base_positions,
        Some(&iopp),
    )?;
    let seal = *verification.chain.value();
    if seal != proof.group_seal {
        return Err(Error::InvalidProof);
    }
    Ok((seal, verification.chain))
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_core::common::field::PrimeField64;
    use zkie_engine::pcs::request::fold::MaskCoins;

    fn entries() -> ConsumerGroupEntries {
        ConsumerGroupEntries {
            heads: 1,
            raw_columns: 16,
            bits_columns: 256,
            w_columns: 16,
            mask_entries: 512,
            rows: 5,
        }
    }

    fn fp(value: u64) -> Fp3 {
        Fp3::from_base(zkie_engine::pcs::request::field::fp_from_u64(value))
    }

    fn commit(
        shape: &WShape,
        oracle_base: u64,
        group: u64,
        values: &[Vec<Fp3>],
        coins: &mut zkie_engine::pcs::request::fold::ByteMaskCoins,
    ) -> (CommittedValueGroup, Vec<Vec<Fp3>>) {
        use zkie_core::common::field::PrimeField64;
        use zkie_engine::pcs::request::fold::MaskCoins;
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
        let committed = crate::proof::output_closure::commit_value_group(
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
        .unwrap();
        (committed, tails)
    }

    struct GroupFixture {
        plan: ConsumerGroupPlan,
        raw: CommittedValueGroup,
        raw_tails: Vec<Vec<Fp3>>,
        bits: CommittedValueGroup,
        bits_tails: Vec<Vec<Fp3>>,
        w: CommittedValueGroup,
        w_tails: Vec<Vec<Fp3>>,
        mask: CommittedConsumerGroupMask,
        mask_rows: Vec<Vec<Fp3>>,
        mask_tails: Vec<Vec<Fp3>>,
        plain_rows: Vec<BatRow>,
    }

    fn fixture() -> GroupFixture {
        use zkie_engine::pcs::request::fold::ByteMaskCoins;
        let shapes = derive_consumer_group_shapes(
            entries(),
            Digest([0x41; 32]),
            Digest([0x51; 32]),
            900,
            999,
        )
        .unwrap();
        let mut coins = ByteMaskCoins::new(Digest([0x77; 32])).unwrap();
        let raw_values = vec![(0..16u64).map(|i| fp(i + 1)).collect::<Vec<_>>()];
        let bits_values = vec![(0..256usize)
            .map(|i| if i % 3 == 0 { fp(1) } else { Fp3::ZERO })
            .collect::<Vec<_>>()];
        let w_values = vec![(0..16u64).map(|i| fp(2 * i + 5)).collect::<Vec<_>>()];
        let (raw, raw_tails) = commit(&shapes.raw_shape, 20, 41, &raw_values, &mut coins);
        let (bits, bits_tails) = commit(&shapes.bits_shape, 21, 42, &bits_values, &mut coins);
        let (w, w_tails) = commit(&shapes.w_shape, 23, 44, &w_values, &mut coins);
        let mask_rows: Vec<Vec<Fp3>> = (0..shapes.mask_rows as usize)
            .map(|row| {
                (0..shapes.first_mask.ell as usize)
                    .map(|place| fp(((row * 7 + place * 3) % 19) as u64))
                    .collect()
            })
            .collect();
        let mask_tails: Vec<Vec<Fp3>> = (0..shapes.mask_rows as usize)
            .map(|_| {
                (0..shapes.first_mask.r as usize)
                    .map(|_| coins.draw().unwrap().0)
                    .collect()
            })
            .collect();
        let mask_salts: Vec<Vec<u8>> = (0..shapes.first_mask.n() as usize)
            .map(|_| {
                let mut salt = Vec::new();
                while salt.len() < 40 {
                    let (value, _) = coins.draw().unwrap();
                    for coefficient in value.c {
                        salt.extend_from_slice(&coefficient.as_canonical_u64().to_le_bytes());
                    }
                }
                salt.truncate(40);
                salt
            })
            .collect();
        let mut plan = ConsumerGroupPlan {
            shapes,
            group_id: 1000,
            fresh: 1001,
            raw_oracle_base: 20,
            bits_oracle_base: 21,
            mask_oracle: 22,
            w_oracle: 23,
            raw_bat_base: 0,
            bits_bat_base: 1,
            mask_bat: 2,
            w_bat: 3,
            mask_root: Digest::ZERO,
            raw_roots: raw.roots.clone(),
            bits_roots: bits.roots.clone(),
            w_roots: w.roots.clone(),
        };
        let mask =
            commit_consumer_group_mask(&plan, mask_rows.clone(), mask_tails.clone(), mask_salts)
                .unwrap();
        plan.mask_root = mask.root;
        plan.check().unwrap();
        let mut plain = BatRow::new(zkie_engine::pcs::request::bat::BatRowKind::Plain);
        let mut coefficients = vec![
            Fp3::ZERO;
            plan.shapes.raw_shape.main_rows as usize
                * plan.shapes.raw_shape.code.ell as usize
        ];
        coefficients[0] = Fp3::ONE;
        plain.add_term(
            plan.raw_bat_base,
            zkie_engine::pcs::request::bat::BatPart::Main,
            coefficients,
        );
        plain.target = raw.messages[0][0][0];
        GroupFixture {
            plan,
            raw,
            raw_tails,
            bits,
            bits_tails,
            w,
            w_tails,
            mask,
            mask_rows,
            mask_tails,
            plain_rows: vec![plain],
        }
    }

    fn chain() -> Chain {
        Chain::begin(131, &Digest::ZERO, b"consumer group", b"fixture", None).unwrap()
    }

    fn run(fixture: &GroupFixture) -> Result<(ConsumerGroupProof, Chain)> {
        let witness = ConsumerGroupWitness {
            raw: &fixture.raw,
            raw_tails: &fixture.raw_tails,
            bits: &fixture.bits,
            bits_tails: &fixture.bits_tails,
            w: &fixture.w,
            w_tails: &fixture.w_tails,
            mask: &fixture.mask,
            mask_rows: fixture.mask_rows.clone(),
            mask_tails: fixture.mask_tails.clone(),
        };
        prove_consumer_group(
            chain(),
            &fixture.plan,
            &witness,
            fixture.plain_rows.clone(),
            Digest([0x91; 32]),
        )
    }

    #[test]
    #[ignore = "the derived group codes make the proximity run heavy; run in release"]
    fn the_consumer_group_proves_and_verifies() {
        let fixture = fixture();
        let (proof, prover_chain) = run(&fixture).unwrap();
        assert_eq!(proof.answers.len(), fixture.plan.member_count());
        assert_ne!(proof.group_seal, Digest::ZERO);
        let encoded = proof.encode().unwrap();
        let decoded = ConsumerGroupProof::decode(&fixture.plan, &encoded).unwrap();
        assert_eq!(decoded, proof);
        let (seal, verifier_chain) =
            verify_consumer_group(chain(), &fixture.plan, fixture.plain_rows.clone(), &decoded)
                .unwrap();
        assert_eq!(seal, proof.group_seal);
        assert_eq!(*prover_chain.value(), seal);
        assert_eq!(*verifier_chain.value(), seal);
    }

    #[test]
    #[ignore = "the derived group codes make the proximity run heavy; run in release"]
    fn a_changed_answer_or_seal_is_refused() {
        let fixture = fixture();
        let (proof, _) = run(&fixture).unwrap();
        let mut changed = proof.clone();
        changed.answers[0] = changed.answers[0].add(&fp(1));
        assert!(verify_consumer_group(
            chain(),
            &fixture.plan,
            fixture.plain_rows.clone(),
            &changed
        )
        .is_err());
        let mut changed = proof;
        changed.group_seal = Digest([0x66; 32]);
        assert!(verify_consumer_group(
            chain(),
            &fixture.plan,
            fixture.plain_rows.clone(),
            &changed
        )
        .is_err());
    }

    #[test]
    fn the_group_shapes_derive_from_the_entry_counts() {
        let shapes = derive_consumer_group_shapes(
            entries(),
            Digest([0x41; 32]),
            Digest([0x51; 32]),
            900,
            999,
        )
        .unwrap();
        assert_eq!(shapes.raw_shape.code, shapes.first_main);
        assert!(shapes.mask_message_entries().unwrap() >= shapes.entries.mask_entries);
    }
}
