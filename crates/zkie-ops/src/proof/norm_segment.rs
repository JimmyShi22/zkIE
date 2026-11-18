//! Production builders for the concise final-norm segment.
//!
//! The builders assemble the accepted hidden norm relation at the production
//! shape: the trusted input group, the four statistic groups, the rounding
//! group and the committed mask group. The x group is the trusted input; its
//! roots are derived from the materialized rows with the plan's fixed public
//! coins, so the prover cannot supply the group or change its roots. The
//! remaining witness values are prover-side and never enter the plan.
//!
//! The parameter tables, the rstd view, the output tail and the byte lookups
//! are attached by later builders on the same plan; this module starts with
//! the core groups that every later stage reuses.

use zkie_core::common::field::PrimeField64;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::code::CodeParams;
use zkie_engine::pcs::request::field::{Fp3, E_DEGREE};
use zkie_engine::pcs::request::fold::{ByteMaskCoins, MaskCoins};
use zkie_engine::pcs::request::runtime_tree::{RuntimeTreePart, RuntimeTreePlan};

use crate::proof::hidden_norm::NormShape;
use crate::proof::hidden_norm_psc::{
    norm_pointwise_masks, norm_tail_mask_masks, NormGroupPlan, NormGroupWitness,
    NormMaskPlan, NormPlan, NormRoundLayout, NormRoundPlan, NormRoundWidths,
};
use crate::proof::hidden_send_session::WShape;
use crate::proof::output_closure::commit_value_group;
use crate::proof::psc::{hp_entries, psc_variables, SHOWN_MAPS};

/// One oracle/tree pair of a committed group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupIds {
    pub oracle: u64,
    pub tree: u64,
}

/// The trusted plan space the norm groups occupy inside one hidden session.
/// Every group shares the session's own group number, code family and run and
/// configuration; the oracle and tree numbers are disjoint from the session's
/// other messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormSegmentSpace {
    pub group: u64,
    pub code: CodeParams,
    pub mask_code: CodeParams,
    pub run: Digest,
    pub configuration: Digest,
    pub salt_bytes: usize,
    pub leaf_class: u32,
    pub x: GroupIds,
    pub sum: GroupIds,
    pub squares: GroupIds,
    pub mean: GroupIds,
    pub variance: GroupIds,
    pub round: GroupIds,
    pub mask: GroupIds,
}

/// The trusted arithmetic profile of one final norm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormSegmentProfile {
    pub positions: u32,
    pub hidden: u32,
    pub valid: u32,
    pub columns: u64,
    pub shift: u32,
    pub clamp: Option<(i64, i64)>,
    pub input_bound: u64,
    pub sum_offset: u64,
    pub mean_offset: u64,
    pub round_widths: NormRoundWidths,
    pub table_real_len: u64,
    /// The fixed public coin seed of the trusted x group derivation.
    pub x_seed: Digest,
}

impl NormSegmentProfile {
    pub fn validate(&self) -> Result<()> {
        NormShape::new(self.positions, self.hidden, self.valid)?;
        if self.columns < u64::from(self.hidden) || !self.columns.is_power_of_two() {
            return Err(Error::InvalidConfig);
        }
        if self.shift >= 63 || self.table_real_len == 0 {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }
}

fn group_plan(
    space: &NormSegmentSpace,
    profile: &NormSegmentProfile,
    columns: u64,
    ids: GroupIds,
) -> Result<NormGroupPlan> {
    Ok(NormGroupPlan {
        shape: WShape::new(
            columns,
            space.code,
            1,
            profile.positions,
            space.salt_bytes,
            ids.tree,
            space.leaf_class,
            space.run,
            space.configuration,
        )?,
        columns,
        group: space.group,
        oracle_base: ids.oracle,
        openings: 1,
    })
}

fn draw_tails(shape: &WShape, positions: usize, coins: &mut ByteMaskCoins) -> Vec<Vec<Fp3>> {
    let rows = shape.main_rows as usize;
    let tail = shape.code.r as usize;
    (0..positions)
        .map(|_| (0..rows * tail).map(|_| coins.draw().unwrap().0).collect())
        .collect()
}

fn draw_pads(shape: &WShape, positions: usize, coins: &mut ByteMaskCoins) -> Result<Vec<Fp3>> {
    let answers = shape.answers as usize;
    let mut pads = Vec::with_capacity(positions * answers);
    for _ in 0..positions * answers {
        let mut pad = Fp3::ZERO;
        for _ in 0..64 {
            pad = coins.draw().unwrap().0;
            if pad != Fp3::ZERO {
                break;
            }
        }
        if pad == Fp3::ZERO {
            return Err(Error::InvalidConfig);
        }
        pads.push(pad);
    }
    Ok(pads)
}

fn draw_salts(shape: &WShape, positions: usize, coins: &mut ByteMaskCoins) -> Vec<Vec<Vec<u8>>> {
    let columns = shape.code.n() as usize;
    (0..positions)
        .map(|_| {
            (0..columns)
                .map(|_| {
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
                    salt
                })
                .collect()
        })
        .collect()
}

/// Commits one norm group with the plan-fixed public coins.
pub fn commit_group(
    group: &NormGroupPlan,
    values: &[Vec<Fp3>],
    seed: Digest,
) -> Result<NormGroupWitness> {
    let positions = values.len();
    if positions == 0 {
        return Err(Error::ShapeMismatch);
    }
    let mut coins = ByteMaskCoins::new(seed)?;
    let pads = draw_pads(&group.shape, positions, &mut coins)?;
    let tails = draw_tails(&group.shape, positions, &mut coins);
    let salts = draw_salts(&group.shape, positions, &mut coins);
    let committed = commit_value_group(
        &group.shape,
        group.columns,
        group.oracle_base,
        group.group,
        values,
        &pads,
        &tails,
        &salts,
        positions as u32,
    )?;
    Ok(NormGroupWitness {
        messages: committed.messages.clone(),
        tails,
        salts,
        pads,
        committed,
    })
}

/// Maps one materialized row into the x group's column domain.
pub fn x_values(row: &[i64], columns: u64) -> Result<Vec<Fp3>> {
    let width = usize::try_from(columns).map_err(|_| Error::ResourceLimit)?;
    if row.len() > width {
        return Err(Error::ShapeMismatch);
    }
    let mut values = vec![Fp3::ZERO; width];
    for (slot, value) in values.iter_mut().zip(row) {
        *slot = crate::proof::output_closure::fp3_signed(i128::from(*value))?;
    }
    Ok(values)
}

/// The committed x group of the materialized rows. Its roots are the plan
/// values the verifier derives itself; the prover cannot supply this group.
pub fn commit_x(
    space: &NormSegmentSpace,
    profile: &NormSegmentProfile,
    x_rows: &[Vec<i64>],
) -> Result<NormGroupWitness> {
    if x_rows.len() != profile.positions as usize {
        return Err(Error::ShapeMismatch);
    }
    let mut values = Vec::with_capacity(x_rows.len());
    for row in x_rows {
        if row.len() != profile.hidden as usize {
            return Err(Error::ShapeMismatch);
        }
        values.push(x_values(row, profile.columns)?);
    }
    let group = group_plan(space, profile, profile.columns, space.x)?;
    commit_group(&group, &values, profile.x_seed)
}

/// The trusted x group roots of the plan.
pub fn x_roots(
    space: &NormSegmentSpace,
    profile: &NormSegmentProfile,
    x_rows: &[Vec<i64>],
) -> Result<Vec<Digest>> {
    Ok(commit_x(space, profile, x_rows)?.committed.roots.clone())
}

fn base_mask_plan(
    space: &NormSegmentSpace,
    profile: &NormSegmentProfile,
    variables: usize,
    relations: usize,
) -> Result<NormMaskPlan> {
    let mask_variables = psc_variables(variables)?;
    let (pointwise_hiding, pointwise_shown) = norm_pointwise_masks(variables)?;
    let (tail_hiding, tail_shown) = norm_tail_mask_masks(profile.columns)?;
    let mut mask = NormMaskPlan {
        oracle: space.mask.oracle,
        group: space.group,
        mask_code: space.mask_code,
        mask_rows: 1,
        answers: 1,
        openings: 1,
        tree: RuntimeTreePlan {
            tree_number: space.mask.tree,
            leaf_class: space.leaf_class,
            log_n: space.mask_code.log_n,
            log_n_zk: space.mask_code.log_n,
            parts: vec![RuntimeTreePart {
                main_symbols: 0,
                mask_symbols: 0,
            }],
            salt_bytes: space.salt_bytes,
            run: space.run,
            configuration: space.configuration,
        },
        positions: profile.positions,
        hiding: hp_entries(&mask_variables)?,
        shown: SHOWN_MAPS,
        claims: 1,
        pointwise: 0,
        pointwise_hiding,
        pointwise_shown,
        tail: 0,
        tail_hiding,
        tail_shown,
    };
    mask.mask_rows = mask.required_rows(relations)?;
    mask.tree.parts[0].mask_symbols = mask
        .mask_rows
        .checked_mul(E_DEGREE as u64)
        .ok_or(Error::ResourceLimit)?;
    Ok(mask)
}

/// Builds the core norm plan (x, the four statistic groups, rounding and the
/// committed mask group). The output tail, the parameter tables, the rstd
/// view and the byte lookups are attached by later builders.
pub fn build_core_plan(
    space: &NormSegmentSpace,
    profile: &NormSegmentProfile,
    x_rows: &[Vec<i64>],
) -> Result<NormPlan> {
    profile.validate()?;
    let shape = NormShape::new(profile.positions, profile.hidden, profile.valid)?;
    let columns = profile.columns;
    let variables = columns.trailing_zeros() as usize + 1;
    let relations: Vec<usize> = (0..26).collect();
    let round_layout =
        NormRoundLayout::new(profile.round_widths, profile.valid, profile.sum_offset, profile.mean_offset)?;
    let round_columns = profile.round_widths.columns()?;
    let x_roots = x_roots(space, profile, x_rows)?;
    let mask = base_mask_plan(space, profile, variables, relations.len())?;
    let mut plan = NormPlan {
        shape,
        columns,
        variables,
        relations,
        x: group_plan(space, profile, columns, space.x)?,
        x_roots,
        sum: group_plan(space, profile, 1, space.sum)?,
        squares: group_plan(space, profile, 1, space.squares)?,
        mean: group_plan(space, profile, 1, space.mean)?,
        variance: group_plan(space, profile, 1, space.variance)?,
        round: Some(NormRoundPlan {
            group: group_plan(space, profile, round_columns, space.round)?,
            layout: round_layout,
            input_bound: profile.input_bound,
            table_real_len: profile.table_real_len,
        }),
        lookup: None,
        rstd: None,
        params: None,
        output: None,
        mask: Some(mask),
    };
    plan.validate()?;
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::code::build_code_plan;

    fn space() -> NormSegmentSpace {
        NormSegmentSpace {
            group: 7,
            code: build_code_plan(12, 2048, 4).unwrap(),
            mask_code: build_code_plan(17, 512, 256).unwrap(),
            run: Digest([0x41; 32]),
            configuration: Digest([0x44; 32]),
            salt_bytes: 40,
            leaf_class: 1,
            x: GroupIds { oracle: 9000, tree: 60000 },
            sum: GroupIds { oracle: 9100, tree: 60100 },
            squares: GroupIds { oracle: 9200, tree: 60200 },
            mean: GroupIds { oracle: 9300, tree: 60300 },
            variance: GroupIds { oracle: 9400, tree: 60400 },
            round: GroupIds { oracle: 9600, tree: 60600 },
            mask: GroupIds { oracle: 9500, tree: 60500 },
        }
    }

    fn real_profile() -> NormSegmentProfile {
        NormSegmentProfile {
            positions: 16,
            hidden: 1024,
            valid: 768,
            columns: 1024,
            shift: 32,
            clamp: None,
            input_bound: 1 << 25,
            sum_offset: 1 << 25,
            mean_offset: 1 << 16,
            round_widths: NormRoundWidths {
                sum: 5,
                squares: 7,
                variance: 6,
                mean: 3,
                offset: 2,
            },
            table_real_len: 7_340_032,
            x_seed: Digest([0x91; 32]),
        }
    }

    #[test]
    fn the_real_shape_core_plan_is_admitted() {
        let profile = real_profile();
        let space = space();
        let x_rows = vec![vec![0i64; profile.hidden as usize]; profile.positions as usize];
        let plan = build_core_plan(&space, &profile, &x_rows).unwrap();
        assert_eq!(plan.x_roots.len(), profile.positions as usize);
        assert!(plan.mask.is_some());
        assert!(plan.round.is_some());
    }

    #[test]
    fn the_x_roots_are_reproducible_and_value_bound() {
        let profile = real_profile();
        let space = space();
        let mut x_rows = vec![vec![0i64; profile.hidden as usize]; profile.positions as usize];
        x_rows[0][0] = 7;
        let first = x_roots(&space, &profile, &x_rows).unwrap();
        let second = x_roots(&space, &profile, &x_rows).unwrap();
        assert_eq!(first, second);
        let mut changed = x_rows.clone();
        changed[0][0] = 8;
        let third = x_roots(&space, &profile, &changed).unwrap();
        assert_ne!(first, third);
    }
}
