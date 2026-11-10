//! Fixed request plan: codes, groups, stages, trees, table and work.
//!
//! A request plan is protocol data, not proof data. The values are fixed by
//! the protocol definition and are checked here against the table shape and
//! the code parameters before any proof is made or read. A plan with an
//! unknown code, a message that does not fit its code, a stage that cannot
//! fold the previous message, a mask code with too little room, a matrix that
//! does not fit beside the table's answer pads, or work above the bound is
//! rejected.

use serde::{Deserialize, Serialize};
use zkie_core::error::{Error, Result};

use crate::pcs::request::code::{build_code_plan, CodeParams, MOST_LOG_N};
use crate::pcs::request::field::E_DEGREE;
use crate::pcs::request::params::{LEAF_SALT_MAX_BYTES, LEAF_SALT_MIN_BYTES};
use crate::pcs::request::table::{index_fault, tree_shape_fault, IndexShape, TableDistances};

/// The largest number of work bits a scope may state.
pub const MAX_WORK_BITS: u8 = 32;

/// The identifier of a code in one request plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodeId {
    U,
    Om,
    MaskU,
    MaskO,
    MaskC,
}

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

/// One group: `2^k` main rows and `2^h` mask rows, read at one list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupPlan {
    pub main: CodeId,
    pub mask: CodeId,
    pub k: u32,
    pub h: u32,
    pub rows_log: u32,
    /// The sample cube's largest number of variables, as the protocol plan
    /// states it. It must equal the value derived from the codes and k/h.
    pub cube: u32,
    pub scope: WorkScope,
}

impl GroupPlan {
    /// The sample cube's largest number of variables, derived from the codes.
    pub fn cube_bits(&self, codes: &CodesPlan) -> Result<u32> {
        let main = codes.get(self.main);
        let mask = codes.get(self.mask);
        let main_bits = self
            .k
            .checked_add(ceil_log2(main.ell)?)
            .ok_or(Error::InvalidConfig)?;
        if mask.ell == 0 {
            return Ok(main_bits);
        }
        let mask_bits = self
            .h
            .checked_add(ceil_log2(mask.ell)?)
            .ok_or(Error::InvalidConfig)?;
        main_bits
            .max(mask_bits)
            .checked_add(1)
            .ok_or(Error::InvalidConfig)
    }
}

/// A stage after the first: the fold's k and h, the two codes' lengths, and
/// the number of positions at which the stage is opened.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageChoice {
    pub k: u32,
    pub h: u32,
    pub log_n: u32,
    pub mask_log_n: u32,
    pub positions: u64,
}

/// A tree's number of columns and the number of entries in one leaf.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreePlan {
    pub log_n: u32,
    pub rows: u64,
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

/// The five codes that are not the table's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodesPlan {
    pub u: CodeParams,
    pub om: CodeParams,
    pub mask_u: CodeParams,
    pub mask_o: CodeParams,
    pub mask_c: CodeParams,
}

impl CodesPlan {
    pub fn get(&self, id: CodeId) -> &CodeParams {
        match id {
            CodeId::U => &self.u,
            CodeId::Om => &self.om,
            CodeId::MaskU => &self.mask_u,
            CodeId::MaskO => &self.mask_o,
            CodeId::MaskC => &self.mask_c,
        }
    }
}

/// The five groups of the request plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupsPlan {
    pub guo: GroupPlan,
    pub omo: GroupPlan,
    pub gu: GroupPlan,
    pub om: GroupPlan,
    pub own: GroupPlan,
}

impl GroupsPlan {
    fn iter(&self) -> impl Iterator<Item = &GroupPlan> {
        [&self.guo, &self.omo, &self.gu, &self.om, &self.own].into_iter()
    }
}

/// The later stages of each proximity test.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StagesPlan {
    pub guo: Vec<StageChoice>,
    pub omo: Vec<StageChoice>,
    pub gu: Vec<StageChoice>,
    pub om: Vec<StageChoice>,
    pub own: Vec<StageChoice>,
}

/// The eight trees of the request plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreesPlan {
    pub t: TreePlan,
    pub gu: TreePlan,
    pub om: TreePlan,
    pub wc: TreePlan,
    pub gu2: TreePlan,
    pub om2: TreePlan,
    pub oc2: TreePlan,
    pub wc2: TreePlan,
}

/// One complete request plan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestPlan {
    pub table: TablePlan,
    pub codes: CodesPlan,
    pub groups: GroupsPlan,
    pub stages: StagesPlan,
    pub trees: TreesPlan,
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
        for code in [
            self.codes.u,
            self.codes.om,
            self.codes.mask_u,
            self.codes.mask_o,
            self.codes.mask_c,
        ] {
            build_code_plan(code.log_n, code.ell, code.r)?;
        }
        for group in self.groups.iter() {
            if group.rows_log != self.table.v
                || group.k > 32
                || group.h > group.k
                || group.k == 0
            {
                return Err(Error::InvalidConfig);
            }
            let main = self.codes.get(group.main);
            let mask = self.codes.get(group.mask);
            if group.k > 0
                && group.h == 0
                && (mask.dim() as u128) * (main.n() as u128)
                    > (main.dim() as u128) * (mask.n() as u128)
            {
                return Err(Error::InvalidConfig);
            }
            if group.cube_bits(&self.codes)? != group.cube {
                return Err(Error::InvalidConfig);
            }
        }
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
        for (group, stages) in [
            (&self.groups.guo, self.stages.guo.as_slice()),
            (&self.groups.omo, self.stages.omo.as_slice()),
            (&self.groups.gu, self.stages.gu.as_slice()),
            (&self.groups.om, self.stages.om.as_slice()),
            (&self.groups.own, self.stages.own.as_slice()),
        ] {
            derive_stages(&self.codes, group, stages, self.samples)?;
        }
        for tree in [
            self.trees.t,
            self.trees.gu,
            self.trees.om,
            self.trees.wc,
            self.trees.gu2,
            self.trees.om2,
            self.trees.oc2,
            self.trees.wc2,
        ] {
            if tree.log_n < 1 || tree.log_n > MOST_LOG_N || tree.rows == 0 {
                return Err(Error::InvalidConfig);
            }
        }
        if self.trees.t.log_n != self.table.code.log_n || self.trees.t.rows != 1u64 << self.table.v
        {
            return Err(Error::InvalidConfig);
        }
        if self.trees.gu.log_n != self.codes.u.log_n
            || self.trees.gu2.log_n != self.codes.u.log_n
            || self.trees.om.log_n != self.codes.om.log_n
            || self.trees.om2.log_n != self.codes.om.log_n
        {
            return Err(Error::InvalidConfig);
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

/// The largest number of bits needed to write `value` as a power of two at
/// least it; an empty value is rejected.
pub fn ceil_log2(value: u64) -> Result<u32> {
    if value == 0 {
        return Err(Error::InvalidConfig);
    }
    Ok(64 - (value - 1).leading_zeros())
}

/// The smallest power of two at least `value`.
pub fn pow2_at_least(value: u64) -> Result<u64> {
    let bits = ceil_log2(value)?;
    if bits >= 64 {
        return Err(Error::InvalidConfig);
    }
    Ok(1u64 << bits)
}

fn is_power_of_two(value: u64) -> bool {
    value != 0 && (value & (value - 1)) == 0
}

/// Derives the main and mask codes of every later stage from the group's
/// codes, the fold's k and h, the number of positions and the sample count.
/// The formulas are the protocol's: the folded main message is the previous
/// message cut into `2^k` rows; the mask message carries the previous mask
/// message and both tails, plus the pads, cut into `2^h` rows; both codes carry
/// as many random coordinates as the stage has positions.
pub fn derive_stages(
    codes: &CodesPlan,
    first: &GroupPlan,
    stages: &[StageChoice],
    samples: u64,
) -> Result<Vec<(CodeParams, CodeParams)>> {
    if stages.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let main0 = codes.get(first.main);
    let mask0 = codes.get(first.mask);
    if first.k > 32
        || first.h > first.k
        || mask0.log_n > main0.log_n
        || mask0.ell < 2u64 * first.k as u64 + 1 - first.h as u64
        || !is_power_of_two(main0.ell)
        || !is_power_of_two(mask0.ell)
    {
        return Err(Error::InvalidConfig);
    }
    let mut main_ell = main0.ell;
    let mut prev_main_r = main0.r;
    let mut prev_mask_ell = mask0.ell;
    let mut prev_mask_r = mask0.r;
    let mut derived = Vec::with_capacity(stages.len());
    for (index, stage) in stages.iter().enumerate() {
        let last = index + 1 == stages.len();
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
        if stage.positions == 0
            || stage.log_n < 1
            || stage.log_n > MOST_LOG_N
            || stage.mask_log_n < 1
            || stage.mask_log_n > MOST_LOG_N
        {
            return Err(Error::InvalidConfig);
        }
        let divisor = 1u64 << stage.k;
        if main_ell % divisor != 0 {
            return Err(Error::InvalidConfig);
        }
        main_ell /= divisor;
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
        let rounding = 1u64 << stage.h;
        let rounded = carried
            .checked_add(samples)
            .ok_or(Error::InvalidConfig)?
            .checked_add(rounding)
            .ok_or(Error::InvalidConfig)?
            .checked_sub(1)
            .ok_or(Error::InvalidConfig)?;
        let scaled = rounded >> stage.h;
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
        let mask_ell = pow2_at_least(scaled)?.max(pow2_at_least(own_fold)?);
        if !is_power_of_two(main_ell)
            || !is_power_of_two(mask_ell)
            || stage.mask_log_n > stage.log_n
            || mask_ell
                .checked_shl(stage.h)
                .ok_or(Error::InvalidConfig)?
                < carried
                    .checked_add(samples)
                    .ok_or(Error::InvalidConfig)?
            || samples > mask_ell
        {
            return Err(Error::InvalidConfig);
        }
        let main = CodeParams {
            log_n: stage.log_n,
            ell: main_ell,
            r: stage.positions,
        };
        let mask = CodeParams {
            log_n: stage.mask_log_n,
            ell: mask_ell,
            r: stage.positions,
        };
        build_code_plan(main.log_n, main.ell, main.r)?;
        build_code_plan(mask.log_n, mask.ell, mask.r)?;
        derived.push((main, mask));
        prev_main_r = stage.positions;
        prev_mask_ell = mask_ell;
        prev_mask_r = stage.positions;
    }
    Ok(derived)
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

    fn group(main: CodeId, mask: CodeId, k: u32, cube: u32, scope: WorkScope) -> GroupPlan {
        GroupPlan {
            main,
            mask,
            k,
            h: 0,
            rows_log: 10,
            cube,
            scope,
        }
    }

    fn stage(k: u32, h: u32, log_n: u32, mask_log_n: u32, positions: u64) -> StageChoice {
        StageChoice {
            k,
            h,
            log_n,
            mask_log_n,
            positions,
        }
    }

    fn base_plan() -> RequestPlan {
        let table = code(17, 65539, 8192);
        let u = code(11, 512, 300);
        let om = code(17, 4096, 3640);
        let mask_u = code(10, 64, 300);
        let mask_o = code(14, 64, 300);
        let mask_c = code(14, 256, 300);
        RequestPlan {
            table: TablePlan {
                code: table,
                v: 10,
                delta_once_num: 1,
                delta_once_den: 8,
            },
            codes: CodesPlan {
                u,
                om,
                mask_u,
                mask_o,
                mask_c,
            },
            groups: GroupsPlan {
                guo: group(CodeId::U, CodeId::MaskU, 7, 17, WorkScope::Once),
                omo: group(CodeId::Om, CodeId::MaskO, 4, 17, WorkScope::Once),
                gu: group(CodeId::U, CodeId::MaskU, 7, 17, WorkScope::Shared),
                om: group(CodeId::Om, CodeId::MaskO, 4, 17, WorkScope::Shared),
                own: group(CodeId::Om, CodeId::MaskC, 4, 17, WorkScope::Request),
            },
            stages: StagesPlan {
                guo: vec![
                    stage(4, 2, 12, 11, 141),
                    stage(3, 2, 12, 11, 141),
                    stage(0, 0, 13, 13, 122),
                ],
                omo: vec![
                    stage(4, 3, 12, 12, 127),
                    stage(3, 3, 12, 11, 141),
                    stage(0, 0, 13, 12, 287),
                ],
                gu: vec![
                    stage(4, 2, 12, 11, 141),
                    stage(3, 2, 12, 11, 141),
                    stage(0, 0, 13, 13, 122),
                ],
                om: vec![
                    stage(4, 3, 12, 12, 127),
                    stage(3, 3, 12, 11, 141),
                    stage(0, 0, 13, 12, 287),
                ],
                own: vec![
                    stage(4, 3, 12, 12, 142),
                    stage(3, 3, 12, 11, 160),
                    stage(0, 0, 13, 12, 327),
                ],
            },
            trees: TreesPlan {
                t: TreePlan {
                    log_n: 17,
                    rows: 1024,
                },
                gu: TreePlan {
                    log_n: 11,
                    rows: 1,
                },
                om: TreePlan {
                    log_n: 17,
                    rows: 1,
                },
                wc: TreePlan {
                    log_n: 17,
                    rows: 1,
                },
                gu2: TreePlan {
                    log_n: 11,
                    rows: 1,
                },
                om2: TreePlan {
                    log_n: 17,
                    rows: 1,
                },
                oc2: TreePlan {
                    log_n: 17,
                    rows: 1,
                },
                wc2: TreePlan {
                    log_n: 17,
                    rows: 1,
                },
            },
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
    fn the_later_stage_codes_follow_the_fold() {
        let plan = base_plan();
        let derived = derive_stages(
            &plan.codes,
            &plan.groups.guo,
            &plan.stages.guo,
            plan.samples,
        )
        .expect("stages");
        let shapes: Vec<(u64, u64, u64, u64)> = derived
            .iter()
            .map(|(main, mask)| (main.ell, main.r, mask.ell, mask.r))
            .collect();
        assert_eq!(
            shapes,
            vec![(32, 141, 512, 141), (4, 141, 512, 141), (4, 122, 2048, 122)]
        );
        let derived = derive_stages(
            &plan.codes,
            &plan.groups.omo,
            &plan.stages.omo,
            plan.samples,
        )
        .expect("stages");
        let shapes: Vec<(u64, u64, u64, u64)> = derived
            .iter()
            .map(|(main, mask)| (main.ell, main.r, mask.ell, mask.r))
            .collect();
        assert_eq!(
            shapes,
            vec![
                (256, 127, 1024, 127),
                (32, 141, 512, 141),
                (32, 287, 2048, 287)
            ]
        );
    }

    #[test]
    fn a_misfit_table_message_is_rejected() {
        let mut plan = base_plan();
        plan.table.code.ell = 1 << 15;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn a_misfit_lo_split_is_rejected() {
        let mut plan = base_plan();
        plan.lin.k_lo = 5;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn a_wrong_cube_is_rejected() {
        let mut plan = base_plan();
        plan.groups.guo.cube = 16;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn a_last_stage_that_folds_is_rejected() {
        let mut plan = base_plan();
        plan.stages.guo[2] = stage(1, 0, 13, 13, 122);
        assert!(plan.validate().is_err());
    }

    #[test]
    fn a_stage_that_does_not_divide_is_rejected() {
        let mut plan = base_plan();
        plan.stages.guo[0] = stage(10, 2, 12, 11, 141);
        assert!(plan.validate().is_err());
    }

    #[test]
    fn a_stage_mask_code_without_room_is_rejected() {
        let mut plan = base_plan();
        plan.stages.guo[0] = stage(4, 2, 12, 9, 141);
        assert!(plan.validate().is_err());
    }

    #[test]
    fn a_stage_with_no_positions_is_rejected() {
        let mut plan = base_plan();
        plan.stages.guo[0] = stage(4, 2, 12, 11, 0);
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
    fn a_mask_code_with_a_larger_rate_is_rejected() {
        let mut plan = base_plan();
        plan.codes.mask_u = code(10, 512, 300);
        assert!(plan.validate().is_err());
    }

    #[test]
    fn more_than_one_sample_is_rejected() {
        let mut plan = base_plan();
        plan.samples = 2;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn a_group_over_the_wrong_rows_is_rejected() {
        let mut plan = base_plan();
        plan.groups.om.rows_log = 9;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn a_table_without_room_for_the_matrix_beside_its_pads_is_rejected() {
        let mut plan = base_plan();
        plan.table.code.ell = 65538;
        assert!(plan.validate().is_err());
        plan.table.code.ell = 65536;
        assert!(plan.validate().is_err());
        let mut underflow = base_plan();
        underflow.table.code.ell = 1;
        underflow.table.code.r = 0;
        assert!(underflow.validate().is_err());
    }

    #[test]
    fn the_smallest_table_message_that_fits_beside_the_pads_passes() {
        let mut plan = base_plan();
        plan.table.code.ell = 65539;
        plan.validate().expect("the matrix fits beside the pads");
    }

    #[test]
    fn a_group_mask_code_without_room_for_the_fold_is_rejected() {
        let mut plan = base_plan();
        plan.codes.mask_u = code(10, 8, 300);
        assert!(plan.validate().is_err());
    }

    #[test]
    fn a_stage_mask_code_longer_than_its_main_code_is_rejected() {
        let mut plan = base_plan();
        plan.stages.guo[0] = stage(4, 2, 12, 13, 141);
        assert!(plan.validate().is_err());
    }
}
