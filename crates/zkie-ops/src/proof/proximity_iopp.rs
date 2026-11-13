//! Trusted derivation of the two hidden proximity groups' IOPP chain.
//!
//! The shared and own groups each carry a trusted stage list, a fresh mask
//! tree and a BASE position count. This module turns those plan values into
//! the loop's chain configuration: it allocates the stage FOLD identifiers
//! from the request's bound identifier set, derives one public tree seed per
//! stage and purpose, and derives the BASE star tree. The verifier calls the
//! same function on the same trusted values and therefore obtains the same
//! chain; no proof byte contributes to any derived value.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::code::CodeParams;
use zkie_engine::pcs::request::field::E_DEGREE;
use zkie_engine::pcs::request::runtime_tree::{MaskTreePlan, RuntimeTreePart, RuntimeTreePlan};
use zkie_engine::primitives::codec::CanonicalWriter;
use zkie_engine::primitives::transcript::{hash_labeled, tag_word, PARAMETERS};

use crate::proof::hidden_send_session::{
    iopp_stage_fault, HiddenProximityPlan, HiddenSendPlan, IoppStagePlan,
};
use crate::proof::multi_stage_iopp::{
    IoppChain, IoppChainConfig, IoppStageFoldIds, IoppStageSeed,
};
use crate::proof::output_closure::ClosurePlan;
use crate::proof::hidden_norm_psc::NormPlan;

/// The leaf class of the BASE star tree. It is distinct from the mask-only
/// class and from the switched-message class, so a star tree can never be
/// confused with one of those trees.
pub const PROXIMITY_BASE_STAR_LEAF_CLASS: u32 = 5;

/// The derivation format version. It is part of every derived label.
const DERIVATION_VERSION: u64 = 1;

/// The base of the BASE star tree-number namespace, one range per group.
const STAR_TREE_BASE_SHARED: u64 = 0x900;
const STAR_TREE_BASE_OWN: u64 = 0xa00;

/// The base of the switch and stage FOLD mask tree-number namespaces. Both
/// ranges are below the star bases, and every allocation walks upward from its
/// base skipping bound and already allocated numbers.
const SWITCH_TREE_BASE: u64 = 0x800;
const FOLD_TREE_BASE: u64 = 0x880;

/// The two proximity groups. The role is part of every derivation label and
/// selects the group's stage list, mask tree and BASE position count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProximityGroupRole {
    Shared,
    Own,
}

impl ProximityGroupRole {
    fn tag(self) -> u64 {
        match self {
            Self::Shared => 1,
            Self::Own => 2,
        }
    }

    fn star_tree_base(self) -> u64 {
        match self {
            Self::Shared => STAR_TREE_BASE_SHARED,
            Self::Own => STAR_TREE_BASE_OWN,
        }
    }
}

/// The identifiers and tree numbers already bound by the trusted request.
/// The stage FOLD identifiers are allocated from this set; every derived tree
/// number must stay outside it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IoppBoundIds {
    pub oracles: Vec<u64>,
    pub trees: Vec<u64>,
}

impl IoppBoundIds {
    pub fn new(mut oracles: Vec<u64>, mut trees: Vec<u64>) -> Self {
        oracles.sort_unstable();
        oracles.dedup();
        trees.sort_unstable();
        trees.dedup();
        Self { oracles, trees }
    }

    /// Reserves one derived chain's stage identifiers and tree numbers in this
    /// registry. The two groups derive in canonical shared-then-own order over
    /// one registry, so a later group never repeats an identifier or a tree
    /// number that an earlier group already bound.
    fn reserve(&mut self, config: &IoppChainConfig) {
        for ids in &config.fold_ids {
            self.oracles.extend([ids.group, ids.member, ids.fresh]);
        }
        self.trees.extend(&config.switch_tree_numbers);
        self.trees.extend(&config.fold_tree_numbers);
        self.trees.push(config.base_star.tree_number);
        self.oracles.sort_unstable();
        self.oracles.dedup();
        self.trees.sort_unstable();
        self.trees.dedup();
    }
}

/// Collects the oracle identifiers and tree numbers already bound by the
/// trusted plan: the hidden SEND oracles, the proximity member identifier
/// range, the two fresh masks, the optional shown-map and index-mask members
/// and the output closure. The stage FOLD allocation may not reuse any of
/// them, and no derived tree may take one of their tree numbers.
pub fn proximity_bound_ids(
    proximity: &HiddenProximityPlan,
    send: &HiddenSendPlan,
    closure: Option<&ClosurePlan>,
    norm: Option<&NormPlan>,
) -> Result<IoppBoundIds> {
    let positions = send.positions() as u64;
    let mut oracles = vec![0u64, send.group];
    // The two shared members and the declared bound of the own group's
    // members are BAT-local identifiers in [0, bound).
    oracles.extend(0..2);
    // The own BAT group has one W member per position, one Omega member,
    // and optional Dot and index-mask members. The closure contributes one
    // slack, one packed-bits and one product-map member per position plus one
    // closure-mask member. The stage FOLD identifiers share this registry, so
    // every member identifier must be inside the bound.
    let mut own_member_bound = positions.checked_add(1).ok_or(Error::ResourceLimit)?;
    if send.dot.is_some() {
        own_member_bound = own_member_bound
            .checked_add(1)
            .ok_or(Error::ResourceLimit)?;
    }
    if send.index_mask.is_some() {
        own_member_bound = own_member_bound
            .checked_add(1)
            .ok_or(Error::ResourceLimit)?;
    }
    if closure.is_some() {
        let closure_members = positions
            .checked_mul(3)
            .and_then(|value| value.checked_add(1))
            .ok_or(Error::ResourceLimit)?;
        own_member_bound = own_member_bound
            .checked_add(closure_members)
            .ok_or(Error::ResourceLimit)?;
    }
    if let Some(norm) = norm {
        let norm_groups = if norm.round.is_some() { 6 } else { 5 };
        let norm_members = positions
            .checked_mul(norm_groups)
            .and_then(|value| {
                value.checked_add(if norm.mask.is_some() { 1 } else { 0 })
            })
            .and_then(|value| {
                value.checked_add(if norm.lookup.is_some() { 2 } else { 0 })
            })
            .and_then(|value| {
                value.checked_add(if norm.rstd.is_some() { 2 } else { 0 })
            })
            .ok_or(Error::ResourceLimit)?;
        own_member_bound = own_member_bound
            .checked_add(norm_members)
            .ok_or(Error::ResourceLimit)?;
    }
    oracles.extend(0..own_member_bound);
    oracles.push(proximity.gu.oracle);
    oracles.push(proximity.omega.oracle);
    oracles.push(proximity.shared_fresh);
    oracles.push(proximity.own_fresh);
    push_range(&mut oracles, send.oracle_base, positions)?;
    if let Some(index) = &send.index_mask {
        oracles.push(index.oracle);
    }
    if let Some(dot) = &send.dot {
        oracles.push(dot.oracle);
    }
    if let Some(closure) = closure {
        push_range(&mut oracles, closure.slack_oracle_base, positions)?;
        push_range(&mut oracles, closure.bits_oracle_base, positions)?;
        push_range(&mut oracles, closure.map_oracle_base, positions)?;
        oracles.push(closure.mask.oracle);
    }
    if let Some(norm) = norm {
        for group in [
            &norm.x,
            &norm.sum,
            &norm.squares,
            &norm.mean,
            &norm.variance,
        ] {
            push_range(&mut oracles, group.oracle_base, positions)?;
        }
        if let Some(round) = &norm.round {
            push_range(&mut oracles, round.group.oracle_base, positions)?;
        }
        if let Some(mask) = &norm.mask {
            oracles.push(mask.oracle);
        }
        if let Some(lookup) = &norm.lookup {
            oracles.push(lookup.commit.bits_oracle);
            oracles.push(lookup.commit.mask_oracle);
        }
        if let Some(rstd) = &norm.rstd {
            oracles.push(rstd.commit.bits_oracle);
            oracles.push(rstd.commit.mask_oracle);
        }
    }

    let mut trees = vec![
        proximity.gu.tree.tree_number,
        proximity.omega.tree.tree_number,
        proximity.shared_mask_tree.tree_number,
        proximity.own_mask_tree.tree_number,
    ];
    push_range(&mut trees, send.w.tree_number, positions)?;
    if let Some(index) = &send.index_mask {
        trees.push(index.tree.tree_number);
    }
    if let Some(dot) = &send.dot {
        trees.push(dot.shape.tree_number);
    }
    if let Some(closure) = closure {
        push_range(&mut trees, closure.slack.tree_number, positions)?;
        push_range(&mut trees, closure.bits.tree_number, positions)?;
        push_range(&mut trees, closure.map.tree_number, positions)?;
        trees.push(closure.mask.tree.tree_number);
    }
    if let Some(norm) = norm {
        for group in [
            &norm.x,
            &norm.sum,
            &norm.squares,
            &norm.mean,
            &norm.variance,
        ] {
            push_range(&mut trees, group.shape.tree_number, positions)?;
        }
        if let Some(round) = &norm.round {
            push_range(&mut trees, round.group.shape.tree_number, positions)?;
        }
        if let Some(mask) = &norm.mask {
            trees.push(mask.tree.tree_number);
        }
        if let Some(lookup) = &norm.lookup {
            trees.push(lookup.commit.bits_shape.tree_number);
            trees.push(lookup.commit.mask_tree.tree_number);
        }
        if let Some(rstd) = &norm.rstd {
            trees.push(rstd.commit.bits_shape.tree_number);
            trees.push(rstd.commit.mask_tree.tree_number);
        }
    }
    Ok(IoppBoundIds::new(oracles, trees))
}

fn push_range(values: &mut Vec<u64>, start: u64, count: u64) -> Result<()> {
    let end = start.checked_add(count).ok_or(Error::ResourceLimit)?;
    values.extend(start..end);
    Ok(())
}

/// Derives one group's trusted IOPP chain from the hidden plan. An empty stage
/// list keeps the single-stage path and returns `None`; a non-empty list is
/// derived against the request's bound identifier and tree sets.
///
/// The two groups and stage 0 allocate from one registry in canonical
/// shared-then-own order: the shared group derives first and everything it
/// bound is reserved before the own group allocates. The verifier calls the
/// same function on the same trusted plan and obtains the same chains.
pub fn proximity_iopp_plan(
    proximity: &HiddenProximityPlan,
    send: &HiddenSendPlan,
    closure: Option<&ClosurePlan>,
    norm: Option<&NormPlan>,
    role: ProximityGroupRole,
) -> Result<Option<ProximityIoppPlan>> {
    let mut bound = proximity_bound_ids(proximity, send, closure, norm)?;
    let shared = derive_group(proximity, &bound, ProximityGroupRole::Shared)?;
    if let Some(plan) = &shared {
        bound.reserve(&plan.chain.config);
    }
    let own = derive_group(proximity, &bound, ProximityGroupRole::Own)?;
    match role {
        ProximityGroupRole::Shared => Ok(shared),
        ProximityGroupRole::Own => Ok(own),
    }
}

fn derive_group(
    proximity: &HiddenProximityPlan,
    bound: &IoppBoundIds,
    role: ProximityGroupRole,
) -> Result<Option<ProximityIoppPlan>> {
    let (stages, mask_tree, fresh, base_positions) = match role {
        ProximityGroupRole::Shared => (
            proximity.shared_stages.as_slice(),
            &proximity.shared_mask_tree,
            proximity.shared_fresh,
            proximity.shared_base_positions,
        ),
        ProximityGroupRole::Own => (
            proximity.own_stages.as_slice(),
            &proximity.own_mask_tree,
            proximity.own_fresh,
            proximity.own_base_positions,
        ),
    };
    if stages.is_empty() {
        return Ok(None);
    }
    derive_proximity_iopp(stages, role, mask_tree, 0, fresh, base_positions, bound).map(Some)
}

/// The derived trusted chain of one proximity group.
#[derive(Clone, Debug)]
pub struct ProximityIoppPlan {
    pub role: ProximityGroupRole,
    pub chain: IoppChain,
    /// The stage 0 FOLD's fresh identifier, bound by the trusted plan.
    pub stage_zero_fresh: u64,
}

/// Derives one group's complete IOPP chain from its trusted stage list, fresh
/// mask tree, stage 0 fresh identifier and BASE position count.
///
/// The caller supplies the request's bound identifiers. The function allocates
/// the stage FOLD group, member and fresh identifiers in stage order, always
/// taking the smallest unused value, and derives one public seed triple per
/// stage. It then checks that no derived tree number collides with a bound
/// tree number or with another derived tree number.
pub fn derive_proximity_iopp(
    stages: &[IoppStagePlan],
    role: ProximityGroupRole,
    mask_tree: &MaskTreePlan,
    stage_zero_group: u64,
    stage_zero_fresh: u64,
    base_positions: usize,
    bound: &IoppBoundIds,
) -> Result<ProximityIoppPlan> {
    if mask_tree.fault().is_some()
        || bound.oracles.binary_search(&stage_zero_group).is_err()
        || bound.oracles.binary_search(&stage_zero_fresh).is_err()
    {
        return Err(Error::InvalidConfig);
    }
    if iopp_stage_fault(
        stages,
        first_main(stages)?,
        first_mask(stages)?,
        base_positions,
    )
    .is_some()
    {
        return Err(Error::InvalidConfig);
    }
    let stage_seeds = derive_stage_seeds(stages, role, mask_tree, base_positions)?;
    let mut used_trees = bound.trees.clone();
    let (switch_tree_numbers, fold_tree_numbers) = allocate_stage_trees(stages.len(), &mut used_trees)?;
    let base_star = derive_base_star(stages, role, mask_tree, &used_trees)?;
    let config = IoppChainConfig {
        first_main: first_main(stages)?,
        first_mask: first_mask(stages)?,
        base_positions,
        fold_ids: allocate_fold_ids(stages.len(), &bound.oracles)?,
        switch_tree_numbers,
        fold_tree_numbers,
        mask_leaf_class: mask_tree.leaf_class,
        switch_salt_bytes: mask_tree.salt_bytes,
        stage_seeds,
        base_star,
    };
    let chain = IoppChain::derive(stages, &config)?;
    check_tree_bindings(&chain, bound)?;
    Ok(ProximityIoppPlan {
        role,
        chain,
        stage_zero_fresh,
    })
}

fn first_main(stages: &[IoppStagePlan]) -> Result<CodeParams> {
    stages
        .first()
        .map(|stage| stage.main)
        .ok_or(Error::InvalidConfig)
}

fn first_mask(stages: &[IoppStagePlan]) -> Result<CodeParams> {
    stages
        .first()
        .map(|stage| stage.mask)
        .ok_or(Error::InvalidConfig)
}

/// Allocates the stage FOLD identifiers. Stage `i` (1 .. len-1) owns one
/// group, one member and one fresh identifier; the allocation walks the stages
/// in order and, inside one stage, the group then member then fresh role,
/// always choosing the smallest value that is not bound and not yet
/// allocated. The stage 0 group and fresh identifiers are part of the bound
/// set supplied by the caller.
fn allocate_fold_ids(stage_count: usize, bound: &[u64]) -> Result<Vec<IoppStageFoldIds>> {
    if stage_count < 2 {
        return Err(Error::InvalidConfig);
    }
    let mut used = bound.to_vec();
    used.sort_unstable();
    used.dedup();
    let mut ids = Vec::with_capacity(stage_count - 2);
    for _ in 1..stage_count.saturating_sub(1) {
        let group = next_unused(&used)?;
        insert_sorted(&mut used, group);
        let member = next_unused(&used)?;
        insert_sorted(&mut used, member);
        let fresh = next_unused(&used)?;
        insert_sorted(&mut used, fresh);
        ids.push(IoppStageFoldIds {
            group,
            member,
            fresh,
        });
    }
    Ok(ids)
}

fn next_unused(used: &[u64]) -> Result<u64> {
    let mut candidate = 1u64;
    for value in used {
        if *value < candidate {
            continue;
        }
        if *value > candidate {
            break;
        }
        candidate = candidate.checked_add(1).ok_or(Error::ResourceLimit)?;
    }
    if candidate == 0 {
        return Err(Error::ResourceLimit);
    }
    Ok(candidate)
}

fn insert_sorted(values: &mut Vec<u64>, value: u64) {
    match values.binary_search(&value) {
        Ok(_) => {}
        Err(index) => values.insert(index, value),
    }
}

/// Allocates one switch tree number per switch and one stage FOLD mask tree
/// number per non-last switch, in stage order and purpose order. The switches
/// walk up from the switch namespace base and the stage FOLD trees from the
/// fold namespace base; both skip every bound tree number and every number
/// allocated before them. `used_trees` grows with the allocation so the BASE
/// star tree, derived next, skips the same values.
fn allocate_stage_trees(
    stage_count: usize,
    used_trees: &mut Vec<u64>,
) -> Result<(Vec<u64>, Vec<u64>)> {
    if stage_count < 2 {
        return Err(Error::InvalidConfig);
    }
    let mut switches = Vec::with_capacity(stage_count - 1);
    for _ in 1..stage_count {
        switches.push(next_unused_tree(used_trees, SWITCH_TREE_BASE)?);
    }
    let mut folds = Vec::with_capacity(stage_count - 2);
    for _ in 1..stage_count.saturating_sub(1) {
        folds.push(next_unused_tree(used_trees, FOLD_TREE_BASE)?);
    }
    Ok((switches, folds))
}

fn next_unused_tree(used: &mut Vec<u64>, mut candidate: u64) -> Result<u64> {
    while used.binary_search(&candidate).is_ok() {
        candidate = candidate.checked_add(1).ok_or(Error::ResourceLimit)?;
    }
    let index = used
        .binary_search(&candidate)
        .err()
        .ok_or(Error::InvalidConfig)?;
    used.insert(index, candidate);
    Ok(candidate)
}

/// The canonical bytes that every derived label of this chain depends on.
/// They carry the derivation version, the group role, the complete stage list,
/// the trusted stage 0 mask tree and the BASE position count. Only trusted plan
/// values enter, so no label depends on a root, a challenge or a proof byte.
fn derivation_domain(
    stages: &[IoppStagePlan],
    role: ProximityGroupRole,
    mask_tree: &MaskTreePlan,
    base_positions: usize,
) -> Vec<u8> {
    let mut writer = CanonicalWriter::new();
    writer.put_u64(DERIVATION_VERSION);
    writer.put_u64(role.tag());
    writer.put_u64(stages.len() as u64);
    for stage in stages {
        put_code(&mut writer, &stage.main);
        put_code(&mut writer, &stage.mask);
        writer.put_u64(stage.k as u64);
        writer.put_u64(stage.h as u64);
        writer.put_u64(stage.positions as u64);
        writer.put_u64(stage.samples as u64);
        writer.put_u64(stage.work_bits as u64);
    }
    writer.put_u64(mask_tree.tree_number);
    writer.put_u64(mask_tree.leaf_class as u64);
    writer.put_u64(mask_tree.log_n as u64);
    writer.put_u64(mask_tree.mask_rows);
    writer.put_u64(mask_tree.salt_bytes as u64);
    writer.put_digest(&mask_tree.run);
    writer.put_digest(&mask_tree.configuration);
    writer.put_u64(base_positions as u64);
    writer.into_bytes()
}

fn put_code(writer: &mut CanonicalWriter, code: &CodeParams) {
    writer.put_u64(code.log_n as u64);
    writer.put_u64(code.ell);
    writer.put_u64(code.r);
}

/// Derives one seed triple per stage. The stage ordinal and the switch, mask
/// and configuration purposes are separate labels over the same trusted
/// domain, so two stages never share a public run or configuration.
fn derive_stage_seeds(
    stages: &[IoppStagePlan],
    role: ProximityGroupRole,
    mask_tree: &MaskTreePlan,
    base_positions: usize,
) -> Result<Vec<IoppStageSeed>> {
    let domain = derivation_domain(stages, role, mask_tree, base_positions);
    let mut seeds = Vec::with_capacity(stages.len());
    for index in 0..stages.len() {
        let stage = index as u64;
        let label = |purpose: u64| -> Result<Digest> {
            let mut bytes = domain.clone();
            bytes.extend_from_slice(&stage.to_le_bytes());
            bytes.extend_from_slice(&purpose.to_le_bytes());
            let digest = hash_labeled(tag_word(PARAMETERS, 6), stage, None, &bytes)?;
            if digest == Digest::ZERO {
                return Err(Error::InvalidConfig);
            }
            Ok(digest)
        };
        let switch_run = label(1)?;
        let mask_run = label(2)?;
        let configuration = label(3)?;
        if switch_run == mask_run
            || switch_run == configuration
            || mask_run == configuration
        {
            return Err(Error::InvalidConfig);
        }
        seeds.push(IoppStageSeed {
            switch_run,
            mask_run,
            configuration,
        });
    }
    Ok(seeds)
}

/// Derives the BASE star tree. The tree shape comes from the last stage's
/// codes; the tree number is allocated from the group's star namespace,
/// skipping every bound tree number. The run and configuration are derived
/// from the trusted domain with the star purpose, so the star tree is not a
/// renamed switch tree.
fn derive_base_star(
    stages: &[IoppStagePlan],
    role: ProximityGroupRole,
    mask_tree: &MaskTreePlan,
    used_trees: &[u64],
) -> Result<RuntimeTreePlan> {
    let last = stages.last().ok_or(Error::InvalidConfig)?;
    let domain = derivation_domain(stages, role, mask_tree, 0);
    let mut bytes = domain;
    bytes.extend_from_slice(&1u64.to_le_bytes());
    let run = hash_labeled(tag_word(PARAMETERS, 7), role.tag(), None, &bytes)?;
    let mut bytes = bytes.clone();
    bytes.extend_from_slice(&2u64.to_le_bytes());
    let configuration = hash_labeled(tag_word(PARAMETERS, 7), role.tag(), None, &bytes)?;
    if run == Digest::ZERO || configuration == Digest::ZERO {
        return Err(Error::InvalidConfig);
    }
    let mut tree_number = role.star_tree_base();
    while used_trees.binary_search(&tree_number).is_ok() {
        tree_number = tree_number.checked_add(1).ok_or(Error::ResourceLimit)?;
    }
    let plan = RuntimeTreePlan {
        tree_number,
        leaf_class: PROXIMITY_BASE_STAR_LEAF_CLASS,
        log_n: last.main.log_n,
        log_n_zk: last.mask.log_n,
        parts: vec![RuntimeTreePart {
            main_symbols: E_DEGREE as u64,
            mask_symbols: E_DEGREE as u64,
        }],
        salt_bytes: mask_tree.salt_bytes,
        run,
        configuration,
    };
    if plan.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    Ok(plan)
}

/// Checks that every tree the chain commits owns a tree number that is not
/// already bound and not used by another derived tree. This also catches a
/// stage list long enough for the switch tree numbers to wrap.
fn check_tree_bindings(chain: &IoppChain, bound: &IoppBoundIds) -> Result<()> {
    let mut seen = bound.trees.clone();
    for stage in &chain.stages {
        if let Some(tree) = &stage.tree {
            allocate_tree_number(&mut seen, tree.tree_number)?;
        }
        if let Some(fold) = &stage.fold {
            allocate_tree_number(&mut seen, fold.mask_tree.tree_number)?;
        }
    }
    allocate_tree_number(&mut seen, chain.config.base_star.tree_number)?;
    Ok(())
}

fn allocate_tree_number(seen: &mut Vec<u64>, tree_number: u64) -> Result<()> {
    let index = seen
        .binary_search(&tree_number)
        .err()
        .ok_or(Error::InvalidConfig)?;
    seen.insert(index, tree_number);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::code::build_code_plan;

    const SALT_BYTES: usize = 40;

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
        vec![
            stage_plan(code(5, 8, 1), code(5, 16, 1), 2, 0, 1),
            stage_plan(code(6, 4, 2), code(6, 32, 2), 1, 1, 2),
            stage_plan(code(8, 4, 1), code(8, 128, 1), 0, 0, 1),
        ]
    }

    fn mask_tree() -> MaskTreePlan {
        MaskTreePlan {
            tree_number: 0x120,
            leaf_class: crate::proof::multi_stage_iopp::IOPP_MASK_LEAF_CLASS,
            log_n: 5,
            mask_rows: 1,
            salt_bytes: SALT_BYTES,
            run: Digest([0x11; 32]),
            configuration: Digest([0x22; 32]),
        }
    }

    fn bound() -> IoppBoundIds {
        IoppBoundIds::new(
            vec![0, 1, 2, 5, 7, 11, 12, 50, 51],
            vec![0x120, 0x210, 0x300],
        )
    }

    #[test]
    fn derivation_parts_are_valid() {
        assert!(iopp_stage_fault(
            &stages(),
            first_main(&stages()).unwrap(),
            first_mask(&stages()).unwrap(),
            1,
        )
        .is_none());
        assert!(derive_stage_seeds(&stages(), ProximityGroupRole::Shared, &mask_tree(), 1).is_ok());
        assert!(derive_base_star(&stages(), ProximityGroupRole::Shared, &mask_tree(), &bound().trees).is_ok());
        let mut used = bound().trees.clone();
        let (switch_tree_numbers, fold_tree_numbers) =
            allocate_stage_trees(stages().len(), &mut used).unwrap();
        let base_star =
            derive_base_star(&stages(), ProximityGroupRole::Shared, &mask_tree(), &used).unwrap();
        let config = IoppChainConfig {
            first_main: first_main(&stages()).unwrap(),
            first_mask: first_mask(&stages()).unwrap(),
            base_positions: 1,
            fold_ids: allocate_fold_ids(stages().len(), &bound().oracles).unwrap(),
            switch_tree_numbers,
            fold_tree_numbers,
            mask_leaf_class: mask_tree().leaf_class,
            switch_salt_bytes: mask_tree().salt_bytes,
            stage_seeds: derive_stage_seeds(&stages(), ProximityGroupRole::Shared, &mask_tree(), 1)
                .unwrap(),
            base_star,
        };
        assert!(IoppChain::derive(&stages(), &config).is_ok());
    }

    #[test]
    fn stage_zero_identifiers_must_be_bound() {
        assert!(derive_proximity_iopp(
            &stages(),
            ProximityGroupRole::Shared,
            &mask_tree(),
            0,
            999,
            1,
            &bound(),
        )
        .is_err());
        assert!(derive_proximity_iopp(
            &stages(),
            ProximityGroupRole::Shared,
            &mask_tree(),
            999,
            50,
            1,
            &bound(),
        )
        .is_err());
    }

    #[test]
    fn identifiers_are_allocated_in_stage_and_role_order() {
        let ids = allocate_fold_ids(5, &[0, 1, 2, 5]).unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(
            ids[0],
            IoppStageFoldIds {
                group: 3,
                member: 4,
                fresh: 6,
            }
        );
        assert_eq!(
            ids[1],
            IoppStageFoldIds {
                group: 7,
                member: 8,
                fresh: 9,
            }
        );
        assert_eq!(
            ids[2],
            IoppStageFoldIds {
                group: 10,
                member: 11,
                fresh: 12,
            }
        );
    }

    #[test]
    fn stage_zero_identifiers_are_never_reused() {
        let plan = derive_proximity_iopp(
            &stages(),
            ProximityGroupRole::Shared,
            &mask_tree(),
            0,
            50,
            1,
            &bound(),
        )
        .unwrap();
        let mut oracles = bound().oracles;
        oracles.push(50);
        for stage in &plan.chain.stages {
            if let Some(fold) = &stage.fold {
                assert!(!oracles.contains(&fold.group));
                assert!(!oracles.contains(&fold.member));
                assert!(!oracles.contains(&fold.fresh));
            }
        }
        assert_eq!(plan.stage_zero_fresh, 50);
    }

    #[test]
    fn derivation_is_deterministic_and_role_separated() {
        let shared = derive_proximity_iopp(
            &stages(),
            ProximityGroupRole::Shared,
            &mask_tree(),
            0,
            50,
            1,
            &bound(),
        )
        .unwrap();
        let again = derive_proximity_iopp(
            &stages(),
            ProximityGroupRole::Shared,
            &mask_tree(),
            0,
            50,
            1,
            &bound(),
        )
        .unwrap();
        let own = derive_proximity_iopp(
            &stages(),
            ProximityGroupRole::Own,
            &mask_tree(),
            0,
            51,
            1,
            &bound(),
        )
        .unwrap();
        assert_eq!(
            shared.chain.stages[1].tree.as_ref().unwrap(),
            again.chain.stages[1].tree.as_ref().unwrap()
        );
        assert_eq!(shared.chain.config.stage_seeds, again.chain.config.stage_seeds);
        assert_ne!(
            shared.chain.stages[1].tree.as_ref().unwrap().run,
            own.chain.stages[1].tree.as_ref().unwrap().run
        );
        assert_ne!(
            shared.chain.config.base_star.tree_number,
            own.chain.config.base_star.tree_number
        );
    }

    #[test]
    fn stage_labels_differ_by_stage_and_purpose() {
        let seeds = derive_stage_seeds(&stages(), ProximityGroupRole::Shared, &mask_tree(), 1)
            .unwrap();
        assert_eq!(seeds.len(), 3);
        for index in 1..seeds.len() {
            assert_ne!(seeds[index - 1].switch_run, seeds[index].switch_run);
            assert_ne!(seeds[index - 1].mask_run, seeds[index].mask_run);
            assert_ne!(
                seeds[index - 1].configuration,
                seeds[index].configuration
            );
        }
        for seed in &seeds {
            assert_ne!(seed.switch_run, seed.mask_run);
            assert_ne!(seed.switch_run, seed.configuration);
            assert_ne!(seed.mask_run, seed.configuration);
        }
    }

    #[test]
    fn a_bound_tree_number_is_skipped_by_the_switch_allocation() {
        let mut bound = bound();
        bound.trees.push(SWITCH_TREE_BASE);
        let plan = derive_proximity_iopp(
            &stages(),
            ProximityGroupRole::Shared,
            &mask_tree(),
            0,
            50,
            1,
            &bound,
        )
        .unwrap();
        assert_eq!(
            plan.chain.stages[1].tree.as_ref().unwrap().tree_number,
            SWITCH_TREE_BASE + 1
        );
        assert_eq!(
            plan.chain.stages[2].tree.as_ref().unwrap().tree_number,
            SWITCH_TREE_BASE + 2
        );
    }

    #[test]
    fn a_bound_tree_number_is_skipped_by_the_fold_allocation() {
        let mut bound = bound();
        bound.trees.push(FOLD_TREE_BASE);
        let plan = derive_proximity_iopp(
            &stages(),
            ProximityGroupRole::Shared,
            &mask_tree(),
            0,
            50,
            1,
            &bound,
        )
        .unwrap();
        assert_eq!(
            plan.chain.stages[1].fold.as_ref().unwrap().mask_tree.tree_number,
            FOLD_TREE_BASE + 1
        );
    }

    #[test]
    fn the_star_tree_is_allocated_outside_the_bound_set() {
        let mut bound = bound();
        bound.trees.push(STAR_TREE_BASE_SHARED);
        let plan = derive_proximity_iopp(
            &stages(),
            ProximityGroupRole::Shared,
            &mask_tree(),
            0,
            50,
            1,
            &bound,
        )
        .unwrap();
        assert_eq!(
            plan.chain.config.base_star.tree_number,
            STAR_TREE_BASE_SHARED + 1
        );
        assert_ne!(
            plan.chain.config.base_star.tree_number,
            plan.chain.stages.last().unwrap().tree.as_ref().unwrap().tree_number
        );
    }
}
