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

use crate::proof::hidden_norm::{NormCoreRow, NormOutputRow, NormShape, RsqrtRow};
use crate::proof::hidden_norm_psc::{
    norm_pointwise_masks, norm_tail_mask_masks, NormGroupPlan, NormGroupWitness,
    commit_norm_mask, norm_mask_entries, NormMaskPlan, NormMaskWitness, NormOutputPlan,
    NormOutputTailPlan, NormOutputTailWidths, NormPlan, NormRoundLayout, NormRoundPlan,
    NormClampWidths, NormLookupParams, NormLookupPlan, NormLookupWitness, NormRoundWidths, NormRstdParams,
    NormRstdPlan, NormWitness, NormOutputTailWitness, NormOutputWitness, NormParamWitness,
    NORM_OUTPUT_TAIL_RELATIONS,
};
use crate::proof::hidden_send_session::WShape;
use crate::proof::lookup_table::TableConfig;
use crate::proof::hidden_send_session::norm_lookup_key_values;
use crate::proof::lookup::{LookupPlan, BYTE_TABLE_IDENTITY};
use crate::proof::norm_param_table::NormParamTablesPlan;
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
    pub clamp_widths: Option<NormClampWidths>,
    pub no_op_clamp: bool,
    pub input_bound: u64,
    pub sum_offset: u64,
    pub mean_offset: u64,
    pub round_widths: NormRoundWidths,
    pub tail_widths: NormOutputTailWidths,
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




/// The public derivation label of every non-x group commitment. The x group
/// uses the profile seed because its roots are plan values.
const SEGMENT_DERIVATION: &[u8] = b"zkie/norm-segment/v1";

fn group_seed(label: &[u8]) -> Digest {
    let mut bytes = Vec::with_capacity(SEGMENT_DERIVATION.len() + label.len());
    bytes.extend_from_slice(SEGMENT_DERIVATION);
    bytes.extend_from_slice(label);
    zkie_core::artifact::digest_bytes(&bytes)
}

fn scalar_values(rows: &[NormCoreRow], value: impl Fn(&NormCoreRow) -> i128) -> Result<Vec<Vec<Fp3>>> {
    rows.iter()
        .map(|row| Ok(vec![crate::proof::output_closure::fp3_signed(value(row))?]))
        .collect()
}

/// Commits the mask-only message and derives its entry list.
fn commit_mask(
    plan: &NormMaskPlan,
    relations: usize,
    seed: Digest,
) -> Result<NormMaskWitness> {
    let mut coins = ByteMaskCoins::new(seed)?;
    let mask_rows = usize::try_from(plan.mask_rows).map_err(|_| Error::ResourceLimit)?;
    let ell = usize::try_from(plan.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
    let mut values = Vec::with_capacity(mask_rows);
    for _ in 0..mask_rows {
        values.push((0..ell).map(|_| coins.draw().unwrap().0).collect::<Vec<_>>());
    }
    let mut pad = Fp3::ZERO;
    for _ in 0..64 {
        pad = coins.draw().unwrap().0;
        if pad != Fp3::ZERO {
            break;
        }
    }
    if pad == Fp3::ZERO || mask_rows == 0 || ell == 0 {
        return Err(Error::InvalidConfig);
    }
    values[mask_rows - 1][ell - 1] = pad;
    let tails = (0..mask_rows)
        .map(|_| {
            (0..plan.mask_code.r as usize)
                .map(|_| coins.draw().unwrap().0)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let salts = (0..plan.tree.columns() as usize)
        .map(|_| {
            let mut salt = Vec::with_capacity(plan.tree.salt_bytes);
            while salt.len() < plan.tree.salt_bytes {
                let (value, _) = coins.draw().unwrap();
                for coefficient in value.c {
                    salt.extend_from_slice(
                        &PrimeField64::as_canonical_u64(&coefficient).to_le_bytes(),
                    );
                }
            }
            salt.truncate(plan.tree.salt_bytes);
            salt
        })
        .collect::<Vec<_>>();
    let committed = commit_norm_mask(plan, values, tails, salts)?;
    let entries = norm_mask_entries(plan, &committed, relations)?;
    Ok(NormMaskWitness { committed, entries })
}

/// Builds the private witness of the core groups, the rounding group, the
/// rsqrt rows, the committed mask and the interval x group. The caller passes
/// the rstd value of every position; the chain verifies them against the
/// frozen table.
pub fn build_core_witness(
    plan: &NormPlan,
    profile: &NormSegmentProfile,
    space: &NormSegmentSpace,
    x_rows: &[Vec<i64>],
    rstd_values: &[i64],
) -> Result<NormWitness> {
    if x_rows.len() != profile.positions as usize
        || rstd_values.len() != profile.positions as usize
    {
        return Err(Error::ShapeMismatch);
    }
    let mut rows = Vec::with_capacity(x_rows.len());
    let mut core = Vec::with_capacity(x_rows.len());
    for row in x_rows {
        if row.len() != profile.hidden as usize {
            return Err(Error::ShapeMismatch);
        }
        rows.push(row.clone());
        core.push(NormCoreRow::from_values(row, profile.valid as usize)?);
    }
    let rsqrt = core
        .iter()
        .zip(rstd_values)
        .map(|(row, rstd)| RsqrtRow::from_variance(row.variance, profile.table_real_len, *rstd))
        .collect::<Result<Vec<_>>>()?;
    let x = commit_x(space, profile, x_rows)?;
    if x.committed.roots != plan.x_roots {
        return Err(Error::StatementMismatch);
    }
    let sum = commit_group(&plan.sum, &scalar_values(&core, |row| row.sum)?, group_seed(b"sum"))?;
    let squares = commit_group(
        &plan.squares,
        &scalar_values(&core, |row| row.squares)?,
        group_seed(b"squares"),
    )?;
    let mean = commit_group(&plan.mean, &scalar_values(&core, |row| row.mean as i128)?, group_seed(b"mean"))?;
    let variance = commit_group(
        &plan.variance,
        &scalar_values(&core, |row| row.variance as i128)?,
        group_seed(b"variance"),
    )?;
    let round = match &plan.round {
        Some(round) => {
            let values = core
                .iter()
                .zip(&rsqrt)
                .map(|(row, rsqrt)| {
                    round
                        .layout
                        .values_with_rsqrt(row, rsqrt, round.group.columns)
                })
                .collect::<Result<Vec<_>>>()?;
            Some(commit_group(&round.group, &values, group_seed(b"round"))?)
        }
        None => None,
    };
    let mask = match &plan.mask {
        Some(mask) => Some(commit_mask(mask, plan.relations.len(), group_seed(b"mask"))?),
        None => None,
    };
    Ok(NormWitness {
        rows,
        core,
        x,
        sum,
        squares,
        mean,
        variance,
        round,
        rsqrt: Some(rsqrt),
        mask,
        lookup: None,
        rstd: None,
        params: None,
        output: None,
    })
}



/// Re-commits the final output rows with the plan-fixed public coins and
/// returns the out group roots. The verifier uses this on the trace's
/// materialized MatMul activation rows; equality with the chain's committed
/// out message roots is the materialized identity binding.
pub fn out_group_roots(
    plan: &NormPlan,
    profile: &NormSegmentProfile,
    rows: &[Vec<i64>],
) -> Result<Vec<Digest>> {
    let output = plan.output.as_ref().ok_or(Error::InvalidConfig)?;
    if rows.len() != profile.positions as usize
        || rows.iter().any(|row| row.len() != profile.columns as usize)
    {
        return Err(Error::ShapeMismatch);
    }
    let values = rows
        .iter()
        .map(|row| {
            signed_column(&row.iter().map(|value| *value as i128).collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(commit_group(&output.out, &values, group_seed(b"out"))?
        .committed
        .roots)
}

/// Fills the private gain and bias vectors of the parameter tables.
pub fn attach_params_witness(
    witness: &mut NormWitness,
    gain: &[i64],
    bias: &[i64],
) -> Result<()> {
    if gain.len() != bias.len() || gain.is_empty() {
        return Err(Error::ShapeMismatch);
    }
    witness.params = Some(NormParamWitness {
        gain: gain.to_vec(),
        bias: bias.to_vec(),
        gain_claim: None,
        bias_claim: None,
    });
    Ok(())
}

fn signed_column(values: &[i128]) -> Result<Vec<Fp3>> {
    values
        .iter()
        .map(|value| crate::proof::output_closure::fp3_signed(*value))
        .collect()
}

/// Commits the output tail, the broadcast row, the gain product, the image and
/// the final output row of every position. The rows consume the same rstd
/// values the frozen-table witness proves.
pub fn attach_output_witness(
    witness: &mut NormWitness,
    plan: &NormPlan,
    profile: &NormSegmentProfile,
    gain: &[i64],
    bias: &[i64],
    rstd_values: &[i64],
) -> Result<()> {
    let output = plan.output.as_ref().ok_or(Error::InvalidConfig)?;
    let tail_plan = output.tail.as_ref().ok_or(Error::InvalidConfig)?;
    if gain.len() != profile.columns as usize
        || bias.len() != profile.columns as usize
        || rstd_values.len() != witness.rows.len()
    {
        return Err(Error::ShapeMismatch);
    }
    let mut rows = Vec::with_capacity(witness.rows.len());
    for ((row, core), rstd) in witness
        .rows
        .iter()
        .zip(&witness.core)
        .zip(rstd_values)
    {
        let mut padded = row.clone();
        padded.resize(profile.columns as usize, 0);
        rows.push(NormOutputRow::from_core(
            core,
            &padded,
            *rstd,
            gain,
            bias,
            profile.valid as usize,
            output.shift,
            output.clamp,
        )?);
    }
    let tail = NormOutputTailWitness {
        committed: commit_group(
            &tail_plan.group,
            &rows
                .iter()
                .map(|row| tail_plan.values(row, profile.columns))
                .collect::<Result<Vec<_>>>()?,
            group_seed(b"tail"),
        )?,
    };
    let mut commit_rows = |group: &crate::proof::hidden_norm_psc::NormGroupPlan,
                           values: Vec<Vec<Fp3>>,
                           label: &[u8]| {
        commit_group(group, &values, group_seed(label))
    };
    let broadcast = commit_rows(
        &output.broadcast,
        rows.iter().map(|row| signed_column(&row.broadcast.iter().map(|value| *value as i128).collect::<Vec<_>>())).collect::<Result<Vec<_>>>()?,
        b"broadcast",
    )?;
    let product = commit_rows(
        &output.product,
        rows.iter().map(|row| signed_column(&row.product.iter().map(|value| *value as i128).collect::<Vec<_>>())).collect::<Result<Vec<_>>>()?,
        b"product",
    )?;
    let image = commit_rows(
        &output.image,
        rows.iter().map(|row| signed_column(&row.image.iter().map(|value| *value as i128).collect::<Vec<_>>())).collect::<Result<Vec<_>>>()?,
        b"image",
    )?;
    let out = commit_rows(
        &output.out,
        rows.iter()
            .map(|row| signed_column(&row.out.iter().map(|value| *value as i128).collect::<Vec<_>>()))
            .collect::<Result<Vec<_>>>()?,
        b"out",
    )?;
    witness.output = Some(NormOutputWitness {
        tail: Some(tail),
        broadcast,
        product,
        image,
        out,
        rows,
    });
    Ok(())
}


/// The oracle/tree and code space of the committed byte lookup over the
/// rounding and tail digit rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LookupSpace {
    pub bits_oracle: u64,
    pub mask_oracle: u64,
    pub bits_tree: u64,
    pub mask_tree: u64,
    pub code: CodeParams,
    pub mask_code: CodeParams,
}

/// Attaches the byte lookup over every committed rounding and tail digit
/// cell. The key map is the canonical one derived from the rounding and tail
/// plans; a plan that drops or redirects a digit cell fails closed.
pub fn attach_lookup(
    plan: &mut NormPlan,
    space: &NormSegmentSpace,
    lookup: &LookupSpace,
) -> Result<()> {
    let round = plan.round.as_ref().ok_or(Error::InvalidConfig)?;
    let tail = plan
        .output
        .as_ref()
        .and_then(|output| output.tail.as_ref())
        .ok_or(Error::InvalidConfig)?;
    let params = NormLookupParams {
        code: lookup.code,
        mask_code: lookup.mask_code,
        group: space.group,
        bits_oracle: lookup.bits_oracle,
        mask_oracle: lookup.mask_oracle,
        bits_tree: lookup.bits_tree,
        mask_tree: lookup.mask_tree,
        leaf_class: space.leaf_class,
        salt_bytes: space.salt_bytes,
        run: space.run,
        configuration: space.configuration,
        padding_byte: 0,
        table_identity: BYTE_TABLE_IDENTITY,
    };
    let plan_lookup =
        NormLookupPlan::from_round_and_tail(round, tail, plan.columns, plan.shape.positions, &params)?;
    plan.lookup = Some(plan_lookup);
    plan.validate()
}

fn counts_to_bits(plan: &LookupPlan, counts: &[u64]) -> Result<Vec<Fp3>> {
    let planes = plan.planes()?;
    let rows = plan.table_rows()?;
    if counts.len() != rows {
        return Err(Error::ShapeMismatch);
    }
    let mut bits = vec![Fp3::ZERO; planes.checked_mul(rows).ok_or(Error::ResourceLimit)?];
    for row in 0..rows {
        for plane in 0..planes {
            if (counts[row] >> plane) & 1 == 1 {
                bits[plane + planes * row] = Fp3::ONE;
            }
        }
    }
    Ok(bits)
}

fn key_occurrence_bits(plan: &LookupPlan, keys: &[Fp3]) -> Result<Vec<Fp3>> {
    let rows = plan.table_rows()?;
    let mut counts = vec![0u64; rows];
    for key in keys {
        let mut found = false;
        for (row, count) in counts.iter_mut().enumerate() {
            if *key == Fp3::from_u64(row as u64) {
                *count += 1;
                found = true;
                break;
            }
        }
        if !found {
            return Err(Error::BoundViolation);
        }
    }
    counts_to_bits(plan, &counts)
}

fn fp3_row(value: &Fp3, rows: usize) -> Result<usize> {
    if value.c[1].as_canonical_u64() != 0 || value.c[2].as_canonical_u64() != 0 {
        return Err(Error::BoundViolation);
    }
    let row = usize::try_from(value.c[0].as_canonical_u64()).map_err(|_| Error::ResourceLimit)?;
    if row >= rows {
        return Err(Error::BoundViolation);
    }
    Ok(row)
}

/// Fills the committed byte lookup witness from the committed rounding and
/// tail messages.
pub fn attach_lookup_witness(witness: &mut NormWitness, plan: &NormPlan) -> Result<()> {
    let lookup = plan.lookup.as_ref().ok_or(Error::InvalidConfig)?;
    let round = witness.round.as_ref().ok_or(Error::ShapeMismatch)?;
    let tail = witness
        .output
        .as_ref()
        .and_then(|output| output.tail.as_ref())
        .ok_or(Error::ShapeMismatch)?;
    let keys = norm_lookup_key_values(plan, lookup, &round.messages, &tail.committed.messages)?;
    let bits = key_occurrence_bits(&lookup.lookup, &keys)?;
    witness.lookup = Some(NormLookupWitness {
        bits,
        messages: None,
    });
    Ok(())
}

/// Fills the frozen-table lookup witness from the committed rounding slots.
pub fn attach_rstd_witness(witness: &mut NormWitness, plan: &NormPlan) -> Result<()> {
    let rstd = plan.rstd.as_ref().ok_or(Error::InvalidConfig)?;
    let round = witness.round.as_ref().ok_or(Error::ShapeMismatch)?;
    let columns = rstd.key.read_columns(&rstd.lookup, &round.messages)?;
    if columns.len() != 2 {
        return Err(Error::ShapeMismatch);
    }
    let rows = usize::try_from(rstd.lookup.table_rows()?).map_err(|_| Error::ResourceLimit)?;
    let mut counts = vec![0u64; rows];
    let entries = rstd.key.sources.len() / 2;
    for key in 0..entries {
        let index = fp3_row(&columns[0][key], rows)?;
        counts[index] += 1;
    }
    let keys = rstd.lookup.key_count()?;
    for _ in entries..keys {
        counts[0] += 1;
    }
    let bits = counts_to_bits(&rstd.lookup, &counts)?;
    witness.rstd = Some(NormLookupWitness {
        bits,
        messages: None,
    });
    Ok(())
}

/// The oracle/tree pair of the frozen-table lookup messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RstdSpace {
    pub bits_oracle: u64,
    pub mask_oracle: u64,
    pub bits_tree: u64,
    pub mask_tree: u64,
    /// The code of the bits message; its length must cover the table rows.
    pub code: CodeParams,
    /// The mask code of the lookup mask message.
    pub mask_code: CodeParams,
}

/// Attaches the frozen-table lookup over the existing public rsqrt table.
/// The table view is a trusted plan value; its content root is the public
/// table digest already frozen for the model. The value bound is the trusted
/// maximum of the rstd entries used by this segment, recorded with its
/// derivation evidence and absorbed into the plan digest.
pub fn attach_rstd(
    plan: &mut NormPlan,
    space: &NormSegmentSpace,
    rstd: &RstdSpace,
    table: TableConfig,
    first_value: Fp3,
    value_bound: u64,
) -> Result<()> {
    let round = plan.round.as_ref().ok_or(Error::InvalidConfig)?.clone();
    let params = NormRstdParams {
        code: rstd.code,
        mask_code: rstd.mask_code,
        group: space.group,
        bits_oracle: rstd.bits_oracle,
        mask_oracle: rstd.mask_oracle,
        bits_tree: rstd.bits_tree,
        mask_tree: rstd.mask_tree,
        leaf_class: space.leaf_class,
        salt_bytes: space.salt_bytes,
        run: space.run,
        configuration: space.configuration,
    };
    let rstd_plan = NormRstdPlan::from_round(
        &round,
        plan.shape.positions,
        table,
        first_value,
        value_bound,
        &params,
    )?;
    plan.rstd = Some(rstd_plan);
    plan.validate()
}

/// The trusted shape space of the four output rows and the tail message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputGroups {
    pub broadcast: GroupIds,
    pub product: GroupIds,
    pub image: GroupIds,
    pub out: GroupIds,
    pub tail: GroupIds,
}

/// Attaches the gain/bias parameter tables. The identities are trust plan
/// values: the committed readers and their roots come from the commit bundle,
/// never from a proof message.
pub fn attach_params(plan: &mut NormPlan, params: NormParamTablesPlan) -> Result<()> {
    plan.params = Some(params);
    plan.validate()
}

/// Attaches the output tail and resizes the committed mask block so the
/// pointwise and tail relation instances are covered.
pub fn attach_output(
    plan: &mut NormPlan,
    profile: &NormSegmentProfile,
    space: &NormSegmentSpace,
    groups: &OutputGroups,
) -> Result<()> {
    let group = |columns: u64, ids: GroupIds| group_plan(space, profile, columns, ids);
    let provisional = NormOutputTailPlan {
        group: group(profile.columns, groups.tail)?,
        widths: profile.tail_widths,
        clamp: profile.clamp_widths,
    };
    let tail_columns = provisional.columns(profile.columns)?;
    let tail = NormOutputTailPlan {
        group: group(tail_columns, groups.tail)?,
        widths: profile.tail_widths,
        clamp: profile.clamp_widths,
    };
    plan.output = Some(NormOutputPlan {
        broadcast: group(profile.columns, groups.broadcast)?,
        product: group(profile.columns, groups.product)?,
        image: group(profile.columns, groups.image)?,
        out: group(profile.columns, groups.out)?,
        shift: profile.shift,
        clamp: profile.clamp,
        no_op_clamp: profile.no_op_clamp,
        tail: Some(tail),
    });
    if let Some(mask) = plan.mask.as_mut() {
        mask.pointwise = 2;
        mask.tail = NORM_OUTPUT_TAIL_RELATIONS;
        mask.claims = (profile.positions as usize)
            .checked_mul(2)
            .ok_or(Error::ResourceLimit)?;
        mask.mask_rows = mask.required_rows(plan.relations.len())?;
        mask.tree.parts[0].mask_symbols = mask
            .mask_rows
            .checked_mul(E_DEGREE as u64)
            .ok_or(Error::ResourceLimit)?;
    }
    plan.validate()
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::code::build_code_plan;

    fn space() -> NormSegmentSpace {
        NormSegmentSpace {
            group: 7,
            code: build_code_plan(12, 2049, 4).unwrap(),
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
            clamp: Some((-(1 << 31), (1 << 31) - 1)),
            clamp_widths: Some(NormClampWidths {
                low: 5,
                high: 5,
                low_strict: 4,
                low_nonstrict: 4,
                high_strict: 4,
                high_nonstrict: 4,
            }),
            no_op_clamp: true,
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
            tail_widths: NormOutputTailWidths {
                quotient: 4,
                remainder: 4,
                doubled_remainder: 5,
                tie_slack: 4,
                shifted: 4,
                pre_clamp: 4,
            },
            table_real_len: 7_340_032,
            x_seed: Digest([0x91; 32]),
        }
    }


    fn param_table(values: &[i64]) -> zkie_engine::pcs::request::table::IndexReader {
        use zkie_engine::pcs::request::table::{
            commit_table_streaming, IndexShape, TableDistances, VecTableSource,
        };
        let code = build_code_plan(12, 2048, 4).unwrap();
        let shape = IndexShape {
            code,
            v: 0,
            answers: 1,
            salt_bytes: 40,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        };
        let ell = shape.code.ell as usize;
        let mut message = vec![crate::proof::column_session::fp_from_signed_word(0); ell];
        for (index, value) in values.iter().take(ell).enumerate() {
            message[index] = crate::proof::column_session::fp_from_signed_word(*value as u64);
        }
        let tails = vec![vec![crate::proof::column_session::fp_from_signed_word(0); shape.code.r as usize]];
        let mut source = VecTableSource::new(vec![message], tails).unwrap();
        let salts = vec![vec![0u8; shape.salt_bytes]; shape.code.n() as usize];
        let table = commit_table_streaming(
            &mut source,
            shape,
            Digest([0x81; 32]),
            &salts,
            8,
            1 << 20,
        )
        .unwrap();
        table.reader.clone()
    }

    fn param_identity(
        kind: crate::proof::norm_param_table::NormParamKind,
        reader: zkie_engine::pcs::request::table::IndexReader,
        width: u64,
    ) -> crate::proof::norm_param_table::NormParamTableIdentity {
        crate::proof::norm_param_table::NormParamTableIdentity {
            version: crate::proof::norm_param_table::NORM_PARAM_TABLE_VERSION,
            kind,
            model_identity: Digest([0x63; 32]),
            asset_table_identity: Digest([0x64; 32]),
            reader,
            row: 0,
            width,
            valid_min: -128,
            valid_max: 127,
            trusted_i32_mapping: true,
        }
    }

    fn param_claim(
        kind: crate::proof::norm_param_table::NormParamKind,
        shape: zkie_engine::pcs::request::table::IndexShape,
        code: zkie_engine::pcs::request::code::CodeParams,
    ) -> crate::proof::norm_param_table::NormParamClaimPlan {
        use zkie_engine::pcs::request::gu::{GuOracleShape, GuShape};
        use zkie_engine::pcs::request::runtime_tree::{RuntimeTreePart, RuntimeTreePlan};
        let main_rows = 2u64;
        crate::proof::norm_param_table::NormParamClaimPlan {
            kind,
            gu: GuShape {
                oracle: 0x8800 + kind.encode(),
                group: 0x42,
                tree: RuntimeTreePlan {
                    tree_number: 0x1234,
                    leaf_class: 1,
                    log_n: code.log_n,
                    log_n_zk: code.log_n,
                    parts: vec![RuntimeTreePart {
                        main_symbols: main_rows * E_DEGREE as u64,
                        mask_symbols: 0,
                    }],
                    salt_bytes: 40,
                    run: Digest([0x71; 32]),
                    configuration: Digest([0x72; 32]),
                },
                part: 0,
                rows_high: true,
                u: GuOracleShape {
                    code,
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

    fn lookup_space() -> LookupSpace {
        LookupSpace {
            bits_oracle: 9900,
            mask_oracle: 9901,
            bits_tree: 60980,
            mask_tree: 60981,
            code: build_code_plan(12, 2049, 4).unwrap(),
            mask_code: build_code_plan(17, 512, 256).unwrap(),
        }
    }

    fn rstd_space() -> RstdSpace {
        RstdSpace {
            bits_oracle: 9800,
            mask_oracle: 9801,
            bits_tree: 60900,
            mask_tree: 60901,
            code: build_code_plan(17, 65539, 8192).unwrap(),
            mask_code: build_code_plan(17, 512, 256).unwrap(),
        }
    }

    fn rstd_table() -> (TableConfig, Fp3) {
        use crate::proof::lookup_table::{TableEncoding, TABLE_ENCODING_VERSION};
        let table = TableConfig {
            identity: Digest([0x82; 32]),
            content_root: Digest([
                124, 186, 106, 124, 136, 5, 59, 153, 17, 44, 18, 113, 162, 213, 147, 225, 20,
                122, 75, 59, 74, 146, 254, 16, 145, 193, 72, 226, 227, 53, 115, 237,
            ]),
            real_len: 7_340_032,
            encoding: TableEncoding::I32Le,
            version: TABLE_ENCODING_VERSION,
        };
        (table, Fp3::from_u64(65_536_000))
    }

    fn output_groups() -> OutputGroups {
        OutputGroups {
            broadcast: GroupIds { oracle: 20000, tree: 61000 },
            product: GroupIds { oracle: 20100, tree: 61100 },
            image: GroupIds { oracle: 20200, tree: 61200 },
            out: GroupIds { oracle: 20300, tree: 61300 },
            tail: GroupIds { oracle: 20400, tree: 61400 },
        }
    }

    #[test]
    fn the_real_shape_output_plan_is_admitted() {
        use crate::proof::norm_param_table::{NormParamKind, NormParamTablesPlan};
        let profile = real_profile();
        let space = space();
        let x_rows = vec![vec![0i64; profile.hidden as usize]; profile.positions as usize];
        let mut plan = build_core_plan(&space, &profile, &x_rows).unwrap();
        let gain_reader = param_table(&vec![1i64; profile.columns as usize]);
        let bias_reader = param_table(&vec![0i64; profile.columns as usize]);
        let params = NormParamTablesPlan {
            gain_claim: Some(param_claim(NormParamKind::Gain, gain_reader.shape, space.code)),
            bias_claim: Some(param_claim(NormParamKind::Bias, bias_reader.shape, space.code)),
            gain: param_identity(NormParamKind::Gain, gain_reader, profile.columns),
            bias: param_identity(NormParamKind::Bias, bias_reader, profile.columns),
            column_draws: 2,
            seed_budget: 1,
        };
        attach_params(&mut plan, params).unwrap();
        let (table, first_value) = rstd_table();
        attach_rstd(&mut plan, &space, &rstd_space(), table, first_value, 8_192).unwrap();
        attach_output(&mut plan, &profile, &space, &output_groups()).unwrap();
        assert!(plan.output.is_some());
        assert!(plan.params.is_some());
        let mask = plan.mask.as_ref().unwrap();
        assert_eq!(mask.pointwise, 2);
        assert_eq!(mask.tail, NORM_OUTPUT_TAIL_RELATIONS);
    }

    #[test]
    fn the_core_witness_matches_the_plan_roots() {
        let profile = real_profile();
        let space = space();
        let mut x_rows = vec![vec![0i64; profile.hidden as usize]; profile.positions as usize];
        x_rows[0][0] = 11;
        let plan = build_core_plan(&space, &profile, &x_rows).unwrap();
        let rstd = vec![1i64; profile.positions as usize];
        let witness = build_core_witness(&plan, &profile, &space, &x_rows, &rstd).unwrap();
        assert_eq!(witness.x.committed.roots, plan.x_roots);
        assert!(witness.mask.is_some());
        assert!(witness.round.is_some());
    }

    #[test]
    fn the_output_witness_is_committed_at_the_real_shape() {
        use crate::proof::norm_param_table::{NormParamKind, NormParamTablesPlan};
        let profile = real_profile();
        let space = space();
        let mut x_rows = vec![vec![1i64; profile.hidden as usize]; profile.positions as usize];
        x_rows[0][0] = 11;
        let mut plan = build_core_plan(&space, &profile, &x_rows).unwrap();
        let gain_reader = param_table(&vec![1i64; profile.columns as usize]);
        let bias_reader = param_table(&vec![0i64; profile.columns as usize]);
        let params = NormParamTablesPlan {
            gain_claim: Some(param_claim(NormParamKind::Gain, gain_reader.shape, space.code)),
            bias_claim: Some(param_claim(NormParamKind::Bias, bias_reader.shape, space.code)),
            gain: param_identity(NormParamKind::Gain, gain_reader, profile.columns),
            bias: param_identity(NormParamKind::Bias, bias_reader, profile.columns),
            column_draws: 2,
            seed_budget: 1,
        };
        attach_params(&mut plan, params).unwrap();
        let (table, first_value) = rstd_table();
        attach_rstd(&mut plan, &space, &rstd_space(), table, first_value, 8_192).unwrap();
        attach_output(&mut plan, &profile, &space, &output_groups()).unwrap();
        let rstd = vec![1i64; profile.positions as usize];
        let mut witness =
            build_core_witness(&plan, &profile, &space, &x_rows, &rstd).unwrap();
        let gain = vec![1i64; profile.columns as usize];
        let bias = vec![0i64; profile.columns as usize];
        attach_params_witness(&mut witness, &gain, &bias).unwrap();
        attach_output_witness(&mut witness, &plan, &profile, &gain, &bias, &rstd).unwrap();
        attach_lookup(&mut plan, &space, &lookup_space()).unwrap();
        attach_lookup_witness(&mut witness, &plan).unwrap();
        attach_rstd_witness(&mut witness, &plan).unwrap();
        assert!(witness.output.is_some());
        assert!(witness.params.is_some());
        assert!(witness.lookup.is_some());
        assert!(witness.rstd.is_some());
        let out_rows = witness
            .output
            .as_ref()
            .unwrap()
            .rows
            .iter()
            .map(|row| row.out.clone())
            .collect::<Vec<_>>();
        let roots = out_group_roots(&plan, &profile, &out_rows).unwrap();
        assert_eq!(
            roots,
            witness.output.as_ref().unwrap().out.committed.roots
        );
    }

    #[test]
    fn a_changed_no_op_clamp_constant_is_rejected() {
        use crate::proof::norm_param_table::{NormParamKind, NormParamTablesPlan};
        let profile = real_profile();
        let space = space();
        let x_rows = vec![vec![0i64; profile.hidden as usize]; profile.positions as usize];
        let mut plan = build_core_plan(&space, &profile, &x_rows).unwrap();
        let gain_reader = param_table(&vec![1i64; profile.columns as usize]);
        let bias_reader = param_table(&vec![0i64; profile.columns as usize]);
        let params = NormParamTablesPlan {
            gain_claim: Some(param_claim(NormParamKind::Gain, gain_reader.shape, space.code)),
            bias_claim: Some(param_claim(NormParamKind::Bias, bias_reader.shape, space.code)),
            gain: param_identity(NormParamKind::Gain, gain_reader, profile.columns),
            bias: param_identity(NormParamKind::Bias, bias_reader, profile.columns),
            column_draws: 2,
            seed_budget: 1,
        };
        attach_params(&mut plan, params).unwrap();
        let (table, first_value) = rstd_table();
        attach_rstd(&mut plan, &space, &rstd_space(), table, first_value, 8_192).unwrap();
        attach_output(&mut plan, &profile, &space, &output_groups()).unwrap();
        let digest = plan.digest();
        plan.output.as_mut().unwrap().clamp = Some((-(1 << 31), 1 << 31));
        assert!(plan.validate().is_err());
        plan.output.as_mut().unwrap().clamp = Some((-(1 << 31), (1 << 31) - 1));
        assert_eq!(plan.digest(), digest);
        plan.output.as_mut().unwrap().clamp = Some((-(1 << 30), (1 << 30) - 1));
        assert!(plan.validate().is_err());
        assert_ne!(plan.digest(), digest);
    }

    #[test]
    fn the_tail_widths_cover_the_real_shift() {
        use crate::proof::norm_param_table::{NormParamKind, NormParamTablesPlan};
        let profile = real_profile();
        let space = space();
        let x_rows = vec![vec![0i64; profile.hidden as usize]; profile.positions as usize];
        let mut plan = build_core_plan(&space, &profile, &x_rows).unwrap();
        let gain_reader = param_table(&vec![1i64; profile.columns as usize]);
        let bias_reader = param_table(&vec![0i64; profile.columns as usize]);
        let params = NormParamTablesPlan {
            gain_claim: Some(param_claim(NormParamKind::Gain, gain_reader.shape, space.code)),
            bias_claim: Some(param_claim(NormParamKind::Bias, bias_reader.shape, space.code)),
            gain: param_identity(NormParamKind::Gain, gain_reader, profile.columns),
            bias: param_identity(NormParamKind::Bias, bias_reader, profile.columns),
            column_draws: 2,
            seed_budget: 1,
        };
        attach_params(&mut plan, params).unwrap();
        let (table, first_value) = rstd_table();
        attach_rstd(&mut plan, &space, &rstd_space(), table, first_value, 8_192).unwrap();
        attach_output(&mut plan, &profile, &space, &output_groups()).unwrap();
        assert!(plan.validate().is_ok());
        for (name, width) in [("remainder", 2usize), ("tie", 2usize)] {
            let mut changed = plan.clone();
            let tail = changed.output.as_mut().unwrap().tail.as_mut().unwrap();
            if name == "remainder" {
                tail.widths.remainder = width;
            } else {
                tail.widths.tie_slack = width;
            }
            assert!(changed.validate().is_err(), "{name} width was admitted");
        }
    }

    #[test]
    fn the_real_magnitude_tail_row_is_committed() {
        use crate::proof::norm_param_table::{NormParamKind, NormParamTablesPlan};
        let profile = real_profile();
        let space = space();
        let x_rows = vec![vec![0i64; profile.hidden as usize]; profile.positions as usize];
        let mut plan = build_core_plan(&space, &profile, &x_rows).unwrap();
        let gain_reader = param_table(&vec![1i64; profile.columns as usize]);
        let bias_reader = param_table(&vec![0i64; profile.columns as usize]);
        let params = NormParamTablesPlan {
            gain_claim: Some(param_claim(NormParamKind::Gain, gain_reader.shape, space.code)),
            bias_claim: Some(param_claim(NormParamKind::Bias, bias_reader.shape, space.code)),
            gain: param_identity(NormParamKind::Gain, gain_reader, profile.columns),
            bias: param_identity(NormParamKind::Bias, bias_reader, profile.columns),
            column_draws: 2,
            seed_budget: 1,
        };
        attach_params(&mut plan, params).unwrap();
        let (table, first_value) = rstd_table();
        attach_rstd(&mut plan, &space, &rstd_space(), table, first_value, 8_192).unwrap();
        attach_output(&mut plan, &profile, &space, &output_groups()).unwrap();
        let tail = plan.output.as_ref().unwrap().tail.as_ref().unwrap();
        let columns = profile.columns as usize;
        let mut row = crate::proof::hidden_norm::NormOutputRow {
            rstd: 5_078,
            shift: profile.shift,
            clamp: profile.clamp,
            broadcast: vec![5_078; columns],
            product: vec![5_798_000_000; columns],
            image: vec![60_469_830_945_683_544; columns],
            quotient: vec![14_079_265; columns],
            remainder: vec![4_294_889_424; columns],
            selector: vec![true; columns],
            shifted: vec![14_079_265; columns],
            pre_clamp: vec![14_562_155; columns],
            out: vec![14_562_155; columns],
        };
        let values = tail.values(&row, profile.columns).unwrap();
        assert_eq!(values.len() as u64, tail.group.columns);
        let mut out_of_half = row.clone();
        out_of_half.quotient[0] = 1 << 31;
        assert!(tail.values(&out_of_half, profile.columns).is_err());
        let mut out_of_pre = row.clone();
        out_of_pre.pre_clamp[0] = 1 << 31;
        assert!(tail.values(&out_of_pre, profile.columns).is_err());
    }

    #[test]
    fn an_image_bound_over_half_the_modulus_is_rejected() {
        use crate::proof::norm_param_table::{NormParamKind, NormParamTablesPlan};
        let profile = real_profile();
        let space = space();
        let x_rows = vec![vec![0i64; profile.hidden as usize]; profile.positions as usize];
        let mut plan = build_core_plan(&space, &profile, &x_rows).unwrap();
        let gain_reader = param_table(&vec![1i64; profile.columns as usize]);
        let bias_reader = param_table(&vec![0i64; profile.columns as usize]);
        let params = NormParamTablesPlan {
            gain_claim: Some(param_claim(NormParamKind::Gain, gain_reader.shape, space.code)),
            bias_claim: Some(param_claim(NormParamKind::Bias, bias_reader.shape, space.code)),
            gain: param_identity(NormParamKind::Gain, gain_reader, profile.columns),
            bias: param_identity(NormParamKind::Bias, bias_reader, profile.columns),
            column_draws: 2,
            seed_budget: 1,
        };
        attach_params(&mut plan, params).unwrap();
        let (table, first_value) = rstd_table();
        attach_rstd(&mut plan, &space, &rstd_space(), table, first_value, 1u64 << 32).unwrap();
        assert!(attach_output(&mut plan, &profile, &space, &output_groups()).is_err());
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
