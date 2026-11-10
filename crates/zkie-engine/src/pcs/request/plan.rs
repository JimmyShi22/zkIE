//! Fixed request plan: the committed table, the two sets of group lines, the
//! linear shape and the work scopes.
//!
//! A request plan is protocol data, not proof data. The values are fixed by
//! the protocol definition and are checked here against the table shape and
//! the code parameters before any proof is made or read. A plan with a
//! misfit table message, a group whose mask code is longer than its main code,
//! a stage that does not fold, a folded and an Omega group with different mask
//! codes, an own group with another main code than Omega, or work above the
//! bound is rejected.

use serde::{Deserialize, Serialize};
use zkie_core::error::{Error, Result};

use crate::pcs::request::code::{build_code_plan, CodeParams, MOST_LOG_N};
use crate::pcs::request::field::E_DEGREE;
use crate::pcs::request::cs_stage::SWITCH_LEAF_CLASS;
use crate::pcs::request::params::{LEAF_SALT_MAX_BYTES, LEAF_SALT_MIN_BYTES};
use crate::pcs::request::table::{index_fault, tree_shape_fault, IndexShape, TableDistances};

/// The largest number of work bits a scope may state.
pub const MAX_WORK_BITS: u8 = 32;

/// Who pays for the proof of work of a round and how often it is paid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkScope {
    Request,
    Shared,
    Once,
}

/// The number of work bits before a verifier's message.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopeWork {
    pub positions: u8,
    pub closeness: u8,
}

impl ScopeWork {
    pub fn new(positions: u8, closeness: u8) -> Result<Self> {
        if positions > MAX_WORK_BITS || closeness > MAX_WORK_BITS {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            positions,
            closeness,
        })
    }
}

/// The work bits of the three scopes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkPlan {
    pub request: ScopeWork,
    pub shared: ScopeWork,
    pub once: ScopeWork,
}

impl WorkPlan {
    pub fn new(request: ScopeWork, shared: ScopeWork, once: ScopeWork) -> Result<Self> {
        for work in [request, shared, once] {
            if work.positions > MAX_WORK_BITS || work.closeness > MAX_WORK_BITS {
                return Err(Error::InvalidConfig);
            }
        }
        Ok(Self {
            request,
            shared,
            once,
        })
    }
}

/// One code line of a group: the code length and the message length, both as
/// powers of two. The random tail is derived by the protocol from the number
/// of positions at which the code is opened; it is not a line of the plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeLine {
    pub log_n: u32,
    pub log_ell: u32,
}

impl CodeLine {
    pub fn n(&self) -> Result<u64> {
        1u64
            .checked_shl(self.log_n)
            .ok_or(Error::InvalidConfig)
    }

    pub fn ell(&self) -> Result<u64> {
        1u64
            .checked_shl(self.log_ell)
            .ok_or(Error::InvalidConfig)
    }
}

/// A stage after the first: the fold's k and h and the two codes' lengths,
/// each as a power of two. The number of positions is derived by the protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageChoice {
    pub k: u32,
    pub h: u32,
    pub log_n: u32,
    pub mask_log_n: u32,
}

/// One group line: the main and mask codes and the later stages.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupLine {
    pub main: CodeLine,
    pub mask: CodeLine,
    pub stages: Vec<StageChoice>,
}

/// The three group lines of one test set: the folded row, Omega and a
/// request's own message. `folded` and `omega` have one mask code, over which
/// the message that splits a claim on both groups is sent; when their main
/// codes are equal the two are tested together and no split is needed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinLines {
    pub folded: GroupLine,
    pub omega: GroupLine,
    pub own: GroupLine,
}

/// The committed table: its code, its rows and its one-time distance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TablePlan {
    pub code: CodeParams,
    pub v: u32,
    pub delta_once_num: u64,
    pub delta_once_den: u64,
}

/// The linear relation's shape: matrix rows, column bits and instances.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinPlan {
    pub matrix_rows_log: u32,
    pub k_col: u32,
    pub k_lo: u32,
    pub key_instances: u64,
    pub proof_instances: u64,
}

/// One complete request plan: the table, the committer's one-time test lines
/// and a batch's lines.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestPlan {
    pub table: TablePlan,
    pub once: LinLines,
    pub proof: LinLines,
    pub lin: LinPlan,
    pub work: WorkPlan,
    pub samples: u64,
    pub salt_bytes: usize,
}

impl RequestPlan {
    /// Checks every relation that follows from the plan's own numbers.
    pub fn validate(&self) -> Result<()> {
        if self.samples != 1 {
            return Err(Error::InvalidConfig);
        }
        if self.salt_bytes < LEAF_SALT_MIN_BYTES || self.salt_bytes > LEAF_SALT_MAX_BYTES {
            return Err(Error::InvalidConfig);
        }
        if self.lin.matrix_rows_log < 1
            || self.lin.matrix_rows_log > 63
            || self.lin.k_col > 63
            || self.lin.k_lo > 63
        {
            return Err(Error::InvalidConfig);
        }
        if self.lin.matrix_rows_log + self.lin.k_lo != self.lin.k_col {
            return Err(Error::InvalidConfig);
        }
        if self.lin.key_instances != 1 || self.lin.proof_instances < 1 {
            return Err(Error::InvalidConfig);
        }
        if self.table.v > 63
            || self.table.v < self.lin.matrix_rows_log
            || self.table.delta_once_num == 0
            || self.table.delta_once_den == 0
            || self.table.delta_once_num >= self.table.delta_once_den
        {
            return Err(Error::InvalidConfig);
        }
        self.table
            .code
            .ell
            .checked_add(self.table.code.r)
            .ok_or(Error::InvalidConfig)?;
        build_code_plan(
            self.table.code.log_n,
            self.table.code.ell,
            self.table.code.r,
        )?;
        let pads = (E_DEGREE as u64)
            .checked_mul(self.samples)
            .ok_or(Error::InvalidConfig)?;
        let plain = self
            .table
            .code
            .ell
            .checked_sub(pads)
            .ok_or(Error::InvalidConfig)?;
        let matrix_rows = 1u64
            .checked_shl(self.lin.matrix_rows_log)
            .ok_or(Error::InvalidConfig)?;
        let matrix_entries = matrix_rows
            .checked_shl(self.lin.k_lo)
            .ok_or(Error::InvalidConfig)?;
        if matrix_entries > plain {
            return Err(Error::InvalidConfig);
        }
        self.table_shape()?;
        for lines in [&self.once, &self.proof] {
            validate_lin_lines(lines)?;
        }
        for work in [self.work.request, self.work.shared, self.work.once] {
            if work.positions > MAX_WORK_BITS || work.closeness > MAX_WORK_BITS {
                return Err(Error::InvalidConfig);
            }
        }
        Ok(())
    }

    /// The committed table's shape, as the runtime constructs it before the
    /// table is read. The one-time distance is a number of positions.
    pub fn table_shape(&self) -> Result<IndexShape> {
        let n = self.table.code.n();
        let radius = n
            .checked_mul(self.table.delta_once_num)
            .ok_or(Error::InvalidConfig)?
            .checked_div(self.table.delta_once_den)
            .ok_or(Error::InvalidConfig)?;
        if radius == 0 || radius >= n {
            return Err(Error::InvalidConfig);
        }
        let answers = u32::try_from(self.samples).map_err(|_| Error::InvalidConfig)?;
        let shape = IndexShape {
            code: self.table.code,
            v: self.table.v,
            answers,
            salt_bytes: self.salt_bytes,
            distances: TableDistances {
                radius,
                later_radius: radius,
                m_once: 0,
                m_later: 0,
            },
        };
        if index_fault(&shape).is_some() || tree_shape_fault(&shape).is_some() {
            return Err(Error::InvalidConfig);
        }
        Ok(shape)
    }
}

/// Checks one set of group lines: the codes fit, the mask code is no longer
/// than its main code, the stages fold the main message, and the folded and
/// Omega groups share a mask code while an own group has Omega's main code.
pub fn validate_lin_lines(lines: &LinLines) -> Result<()> {
    validate_group(&lines.folded)?;
    validate_group(&lines.omega)?;
    validate_group(&lines.own)?;
    if lines.folded.mask != lines.omega.mask {
        return Err(Error::InvalidConfig);
    }
    if lines.own.main != lines.omega.main {
        return Err(Error::InvalidConfig);
    }
    let split = lines.folded.main != lines.omega.main;
    if split && lines.folded.mask != lines.omega.mask {
        return Err(Error::InvalidConfig);
    }
    Ok(())
}

fn validate_group(group: &GroupLine) -> Result<()> {
    validate_code(&group.main)?;
    validate_code(&group.mask)?;
    if group.mask.log_n > group.main.log_n {
        return Err(Error::InvalidConfig);
    }
    if group.stages.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let mut main_log_ell = group.main.log_ell;
    for (index, stage) in group.stages.iter().enumerate() {
        let last = index + 1 == group.stages.len();
        if stage.k > 32 || stage.h > stage.k {
            return Err(Error::InvalidConfig);
        }
        if last {
            if stage.k != 0 || stage.h != 0 {
                return Err(Error::InvalidConfig);
            }
        } else if stage.k == 0 {
            return Err(Error::InvalidConfig);
        }
        if stage.log_n < 1
            || stage.log_n > MOST_LOG_N
            || stage.mask_log_n < 1
            || stage.mask_log_n > MOST_LOG_N
        {
            return Err(Error::InvalidConfig);
        }
        if stage.mask_log_n > stage.log_n {
            return Err(Error::InvalidConfig);
        }
        if stage.k > 0 {
            if main_log_ell < stage.k {
                return Err(Error::InvalidConfig);
            }
            main_log_ell -= stage.k;
        }
        if stage.log_n < main_log_ell {
            return Err(Error::InvalidConfig);
        }
    }
    Ok(())
}

fn validate_code(code: &CodeLine) -> Result<()> {
    if code.log_n < 1 || code.log_n > MOST_LOG_N || code.log_ell > code.log_n {
        return Err(Error::InvalidConfig);
    }
    Ok(())
}

/// The largest number of bits needed to write `value` as a power of two at
/// least it; an empty value is rejected.
pub fn ceil_log2(value: u64) -> Result<u32> {
    if value == 0 {
        return Err(Error::InvalidConfig);
    }
    Ok(64 - (value - 1).leading_zeros())
}


/// One compiled stage: the fold's k and h, the two codes with their random
/// tails, and the number of positions. The reference frame derives these from
/// the plan lines and the table; a runtime consumes them and checks that they
/// agree with the lines.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledStage {
    pub k: u32,
    pub h: u32,
    pub main: CodeParams,
    pub mask: CodeParams,
    pub positions: u64,
}

/// One compiled group: its row variables, the two codes and the later stages.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledGroup {
    pub k: u32,
    pub h: u32,
    pub main: CodeParams,
    pub mask: CodeParams,
    pub positions: u64,
    pub stages: Vec<CompiledStage>,
}

/// The compiled form of one set of group lines. Omega is absent when its main
/// code equals the folded row's; the two are then tested together.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledLines {
    pub folded: CompiledGroup,
    pub omega: Option<CompiledGroup>,
    pub own: CompiledGroup,
}

/// One oracle message of the compiled linear shape.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledOracle {
    pub code: CodeParams,
    pub main_rows: u64,
    pub main_deg: u32,
    pub mask_rows: u64,
    pub mask_code: Option<CodeParams>,
    pub answers: u32,
}

/// The compiled linear shape's oracle messages and leaf classes: the folded
/// row, Omega, a request's W, the split masks and the leaf classes that keep
/// messages read at different lists in different trees.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledFoldeq {
    pub u: CompiledOracle,
    pub omega: CompiledOracle,
    pub w: CompiledOracle,
    pub shared_split: Option<CompiledOracle>,
    pub own_split: Option<CompiledOracle>,
    pub rows_high: bool,
    pub masks_ahead: bool,
    pub masks_ride: bool,
    pub fold_leaf_class: u32,
    pub omega_leaf_class: u32,
    pub w_leaf_class: u32,
    pub shared_split_leaf_class: u32,
    pub own_split_leaf_class: u32,
    pub base_leaf_class: u32,
    pub base_tree_number: u64,
    pub base_salt_bytes: usize,
}

/// The compiled form of the one-time and batch group sets.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledPlan {
    pub once: CompiledLines,
    pub proof: CompiledLines,
    pub once_foldeq: CompiledFoldeq,
}

impl RequestPlan {
    /// Checks the compiled groups against the plan lines and each other.
    pub fn validate_compiled(&self, compiled: &CompiledPlan) -> Result<()> {
        validate_compiled_lines(&self.once, &compiled.once, self.samples)?;
        validate_compiled_lines(&self.proof, &compiled.proof, self.samples)?;
        check_compiled_foldeq(self, &compiled.once_foldeq)?;
        check_compiled_base(&compiled.once, &compiled.once_foldeq)
    }
}

/// Checks the last stage's independent masking oracle configuration: the
/// final stage has no fold variables, its star tree uses the final codes'
/// depths, and its leaf class differs from the switched oracle's so their
/// leaves never share a tree.
fn check_compiled_base(lines: &CompiledLines, shape: &CompiledFoldeq) -> Result<()> {
    let last = lines.folded.stages.last().ok_or(Error::InvalidConfig)?;
    if last.k != 0 || last.h != 0 {
        return Err(Error::InvalidConfig);
    }
    if shape.base_tree_number >= (1u64 << 16)
        || shape.base_salt_bytes < LEAF_SALT_MIN_BYTES
        || shape.base_salt_bytes > LEAF_SALT_MAX_BYTES
        || shape.base_salt_bytes % 8 != 0
        || shape.base_leaf_class == SWITCH_LEAF_CLASS
    {
        return Err(Error::InvalidConfig);
    }
    Ok(())
}

/// Checks one compiled set against its lines.
pub fn validate_compiled_lines(
    lines: &LinLines,
    compiled: &CompiledLines,
    samples: u64,
) -> Result<()> {
    validate_lin_lines(lines)?;
    check_compiled_group(&lines.folded, &compiled.folded, samples)?;
    check_compiled_group(&lines.own, &compiled.own, samples)?;
    let merged = lines.folded.main == lines.omega.main;
    match (&compiled.omega, merged) {
        (None, true) => Ok(()),
        (Some(group), false) => check_compiled_group(&lines.omega, group, samples),
        _ => Err(Error::InvalidConfig),
    }
}

fn check_compiled_group(line: &GroupLine, group: &CompiledGroup, samples: u64) -> Result<()> {
    if group.main.log_n != line.main.log_n
        || group.main.ell != line.main.ell()?
        || group.mask.log_n != line.mask.log_n
        || group.mask.ell != line.mask.ell()?
    {
        return Err(Error::InvalidConfig);
    }
    build_code_plan(group.main.log_n, group.main.ell, group.main.r)?;
    build_code_plan(group.mask.log_n, group.mask.ell, group.mask.r)?;
    if group.k == 0
        || group.h > group.k
        || group.positions == 0
        || group.mask.r != group.positions
        || group.main.r < group.positions
        || group.stages.len() != line.stages.len()
    {
        return Err(Error::InvalidConfig);
    }
    let mut main_ell = group.main.ell;
    let mut prev_main_r = group.main.r;
    let mut prev_mask_ell = group.mask.ell;
    let mut prev_mask_r = group.mask.r;
    for (index, (line_stage, stage)) in line.stages.iter().zip(&group.stages).enumerate() {
        let last = index + 1 == group.stages.len();
        if stage.k != line_stage.k
            || stage.h != line_stage.h
            || stage.main.log_n != line_stage.log_n
            || stage.mask.log_n != line_stage.mask_log_n
            || stage.positions == 0
            || stage.main.r != stage.positions
            || stage.mask.r != stage.positions
        {
            return Err(Error::InvalidConfig);
        }
        if last && (stage.k != 0 || stage.h != 0) {
            return Err(Error::InvalidConfig);
        }
        let divisor = 1u64 << stage.k;
        if main_ell % divisor != 0 {
            return Err(Error::InvalidConfig);
        }
        main_ell /= divisor;
        if stage.main.ell != main_ell {
            return Err(Error::InvalidConfig);
        }
        let carried = (2u64
            .checked_mul(prev_mask_ell)
            .ok_or(Error::InvalidConfig)?)
        .checked_add(prev_main_r)
        .ok_or(Error::InvalidConfig)?
        .checked_add(
            2u64.checked_mul(prev_mask_r)
                .ok_or(Error::InvalidConfig)?,
        )
        .ok_or(Error::InvalidConfig)?;
        let rounded = carried
            .checked_add(samples)
            .ok_or(Error::InvalidConfig)?
            .checked_add(1u64 << stage.h)
            .ok_or(Error::InvalidConfig)?
            .checked_sub(1)
            .ok_or(Error::InvalidConfig)?;
        let own_fold = if stage.k > 0 {
            2u64.checked_mul(stage.k as u64)
                .ok_or(Error::InvalidConfig)?
                .checked_add(1)
                .ok_or(Error::InvalidConfig)?
                .checked_sub(stage.h as u64)
                .ok_or(Error::InvalidConfig)?
        } else {
            1
        };
        let mask_ell = pow2_at_least(rounded >> stage.h)?.max(pow2_at_least(own_fold)?);
        if stage.mask.ell != mask_ell {
            return Err(Error::InvalidConfig);
        }
        build_code_plan(stage.main.log_n, stage.main.ell, stage.main.r)?;
        build_code_plan(stage.mask.log_n, stage.mask.ell, stage.mask.r)?;
        prev_main_r = stage.main.r;
        prev_mask_ell = stage.mask.ell;
        prev_mask_r = stage.mask.r;
    }
    Ok(())
}

/// Checks the compiled linear shape's oracle messages against the plan's
/// once lines and the table's shape. The row counts follow the reference's
/// placement: the folded row and Omega include the sample's pad; W's pad is in
/// its mask row.
fn check_compiled_foldeq(plan: &RequestPlan, shape: &CompiledFoldeq) -> Result<()> {
    let lines = &plan.once;
    let u_line = lines.folded.main;
    let omega_line = lines.omega.main;
    let w_line = lines.own.main;
    let fold_mask_line = lines.folded.mask;
    let own_mask_line = lines.own.mask;
    let u_ell = u_line.ell()?;
    let omega_ell = omega_line.ell()?;
    let w_ell = w_line.ell()?;
    let table = &plan.table;
    let step = pow2_at_least(table.code.r)?;
    let tail_first = table
        .code
        .ell
        .checked_add(step - 1)
        .and_then(|value| value.checked_div(step))
        .and_then(|value| value.checked_mul(step))
        .ok_or(Error::InvalidConfig)?;
    let entries = tail_first
        .checked_add(table.code.r)
        .ok_or(Error::InvalidConfig)?;
    let u_rows = entries
        .checked_add(plan.samples)
        .and_then(|value| value.checked_add(u_ell - 1))
        .and_then(|value| value.checked_div(u_ell))
        .ok_or(Error::InvalidConfig)?;
    let matrix_rows = 1u64
        .checked_shl(plan.lin.matrix_rows_log)
        .ok_or(Error::InvalidConfig)?;
    let omega_rows = matrix_rows
        .checked_add(plan.samples)
        .and_then(|value| value.checked_add(omega_ell - 1))
        .and_then(|value| value.checked_div(omega_ell))
        .ok_or(Error::InvalidConfig)?;
    let matrix_columns = 1u64
        .checked_shl(plan.lin.k_col)
        .ok_or(Error::InvalidConfig)?;
    let w_rows = matrix_columns
        .checked_add(w_ell - 1)
        .and_then(|value| value.checked_div(w_ell))
        .ok_or(Error::InvalidConfig)?;
    if shape.u.code.log_n != u_line.log_n
        || shape.u.code.ell != u_ell
        || shape.u.main_rows != u_rows
        || shape.u.main_deg != E_DEGREE as u32
        || shape.u.mask_rows != 0
        || shape.u.answers != plan.samples as u32
        || shape.omega.code.log_n != omega_line.log_n
        || shape.omega.code.ell != omega_ell
        || shape.omega.main_rows != omega_rows
        || shape.omega.mask_rows != 0
        || shape.omega.answers != plan.samples as u32
        || shape.w.code.log_n != w_line.log_n
        || shape.w.code.ell != w_ell
        || shape.w.main_rows != w_rows
        || shape.w.mask_rows != 1
        || shape
            .w
            .mask_code
            .map(|code| (code.log_n, code.ell))
            != Some((own_mask_line.log_n, own_mask_line.ell()?))
        || shape.w.answers != plan.samples as u32
        || !shape.rows_high
        || !shape.masks_ahead
        || !shape.masks_ride
        || shape.fold_leaf_class != 0
        || shape.w_leaf_class != 0
        || shape.omega_leaf_class == shape.fold_leaf_class
        || shape.shared_split_leaf_class == shape.omega_leaf_class
        || shape.own_split_leaf_class == shape.w_leaf_class
    {
        return Err(Error::InvalidConfig);
    }
    if w_rows
        .checked_mul(w_ell)
        .and_then(|capacity| capacity.checked_sub(matrix_columns))
        .ok_or(Error::InvalidConfig)?
        >= plan.samples
    {
        return Err(Error::InvalidConfig);
    }
    match &shape.shared_split {
        Some(shared) => {
            if shared.main_rows != 0
                || shared.mask_rows != 1
                || shared
                    .mask_code
                    .map(|code| (code.log_n, code.ell))
                    != Some((fold_mask_line.log_n, fold_mask_line.ell()?))
                || shared.answers != plan.samples as u32
            {
                return Err(Error::InvalidConfig);
            }
        }
        None => return Err(Error::InvalidConfig),
    }
    if shape.own_split.is_some() {
        return Err(Error::InvalidConfig);
    }
    Ok(())
}

/// The smallest power of two at least `value`.
pub fn pow2_at_least(value: u64) -> Result<u64> {
    let bits = ceil_log2(value)?;
    if bits >= 64 {
        return Err(Error::InvalidConfig);
    }
    Ok(1u64 << bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::request::table::record_seeds;

    fn code(log_n: u32, ell: u64, r: u64) -> CodeParams {
        build_code_plan(log_n, ell, r).expect("code")
    }

    fn scoped(positions: u8) -> ScopeWork {
        ScopeWork::new(positions, 0).expect("work")
    }

    fn line(log_n: u32, log_ell: u32) -> CodeLine {
        CodeLine { log_n, log_ell }
    }

    fn stage(k: u32, h: u32, log_n: u32, mask_log_n: u32) -> StageChoice {
        StageChoice {
            k,
            h,
            log_n,
            mask_log_n,
        }
    }

    fn stages() -> Vec<StageChoice> {
        vec![
            stage(4, 3, 12, 12),
            stage(3, 3, 12, 11),
            stage(0, 0, 13, 12),
        ]
    }

    fn group(main: CodeLine, mask: CodeLine) -> GroupLine {
        GroupLine {
            main,
            mask,
            stages: stages(),
        }
    }

    fn lines() -> LinLines {
        LinLines {
            folded: group(line(17, 12), line(14, 6)),
            omega: group(line(17, 12), line(14, 6)),
            own: group(line(17, 12), line(14, 8)),
        }
    }

    fn base_plan() -> RequestPlan {
        RequestPlan {
            table: TablePlan {
                code: code(17, 65539, 8192),
                v: 10,
                delta_once_num: 1,
                delta_once_den: 8,
            },
            once: lines(),
            proof: lines(),
            lin: LinPlan {
                matrix_rows_log: 10,
                k_col: 16,
                k_lo: 6,
                key_instances: 1,
                proof_instances: 16,
            },
            work: WorkPlan::new(scoped(0), scoped(12), scoped(12)).expect("work"),
            samples: 1,
            salt_bytes: 40,
        }
    }

    #[test]
    fn a_complete_plan_passes() {
        base_plan().validate().expect("plan");
    }

    #[test]
    fn the_table_shape_and_record_seeds_are_built_from_the_plan() {
        let plan = base_plan();
        let shape = plan.table_shape().expect("shape");
        assert_eq!(shape.cube_bits(), 17);
        assert_eq!(shape.plain(), 1 << plan.lin.k_col);
        let seeds = record_seeds(&shape, b"shape-check", shape.answers as usize).expect("seeds");
        assert_eq!(seeds.len(), shape.answers as usize);
        assert_eq!(seeds[0].z.len(), shape.cube_bits() as usize);
    }

    #[test]
    fn a_folded_and_omega_mask_mismatch_is_rejected() {
        let mut plan = base_plan();
        plan.proof.omega.mask = line(14, 7);
        assert!(plan.validate().is_err());
    }

    #[test]
    fn an_own_group_with_another_main_code_is_rejected() {
        let mut plan = base_plan();
        plan.once.own.main = line(16, 11);
        assert!(plan.validate().is_err());
    }

    #[test]
    fn a_mask_code_longer_than_its_main_code_is_rejected() {
        let mut plan = base_plan();
        plan.proof.own.mask = line(18, 6);
        assert!(plan.validate().is_err());
    }

    #[test]
    fn a_last_stage_that_folds_is_rejected() {
        let mut plan = base_plan();
        plan.proof.folded.stages[2] = stage(1, 0, 13, 12);
        assert!(plan.validate().is_err());
    }

    #[test]
    fn a_stage_that_cannot_fold_the_message_is_rejected() {
        let mut plan = base_plan();
        plan.proof.folded.stages[0] = stage(13, 3, 12, 12);
        assert!(plan.validate().is_err());
    }

    #[test]
    fn a_misfit_table_message_is_rejected() {
        let mut plan = base_plan();
        plan.table.code.ell = 65538;
        assert!(plan.validate().is_err());
        let mut plan = base_plan();
        plan.table.code.ell = 65536;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn work_above_the_bound_is_rejected() {
        assert!(ScopeWork::new(33, 0).is_err());
        let bad = ScopeWork {
            positions: 33,
            closeness: 0,
        };
        assert!(WorkPlan::new(scoped(0), bad, scoped(0)).is_err());
    }

    #[test]
    fn more_than_one_sample_is_rejected() {
        let mut plan = base_plan();
        plan.samples = 2;
        assert!(plan.validate().is_err());
    }
}
