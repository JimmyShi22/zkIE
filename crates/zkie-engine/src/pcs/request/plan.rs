//! Fixed request plan: codes, groups, stages, trees, table and work.
//!
//! A request plan is protocol data, not proof data. The values are fixed by
//! the protocol definition and are checked here against the table shape and
//! the code parameters before any proof is made or read. A plan with an
//! unknown code, a message that does not fit its code, a stage that cannot
//! fold the previous message, a mask code with a larger rate than its main
//! code, or work above the bound is rejected.

use serde::{Deserialize, Serialize};
use zkie_core::error::{Error, Result};

use crate::pcs::request::code::{build_code_plan, CodeParams, MOST_LOG_N};

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

/// A stage after the first: the fold's k and h, and the two codes' lengths.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageChoice {
    pub k: u32,
    pub h: u32,
    pub log_n: u32,
    pub mask_log_n: u32,
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

impl StagesPlan {
    // The group references are supplied by `RequestPlan::validate`, which has
    // both the groups and this struct; keeping them out of this struct avoids
    // a second copy of the group numbers.
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
}

impl RequestPlan {
    /// Checks every relation that follows from the plan's own numbers.
    pub fn validate(&self) -> Result<()> {
        if self.samples != 1 {
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
        build_code_plan(
            self.table.code.log_n,
            self.table.code.ell,
            self.table.code.r,
        )?;
        if self.table.code.ell != 1u64 << self.lin.k_col {
            return Err(Error::InvalidConfig);
        }
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
            if (mask.dim() as u128) * (main.n() as u128)
                > (main.dim() as u128) * (mask.n() as u128)
            {
                return Err(Error::InvalidConfig);
            }
            if group.cube_bits(&self.codes)? != group.cube {
                return Err(Error::InvalidConfig);
            }
        }
        for (group, stages) in [
            (&self.groups.guo, self.stages.guo.as_slice()),
            (&self.groups.omo, self.stages.omo.as_slice()),
            (&self.groups.gu, self.stages.gu.as_slice()),
            (&self.groups.om, self.stages.om.as_slice()),
            (&self.groups.own, self.stages.own.as_slice()),
        ] {
            validate_stages(&self.codes, group, stages)?;
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
        for work in [
            self.work.request,
            self.work.shared,
            self.work.once,
        ] {
            if work.positions > MAX_WORK_BITS || work.closeness > MAX_WORK_BITS {
                return Err(Error::InvalidConfig);
            }
        }
        Ok(())
    }
}

/// The largest number of bits needed to write `value` as a power of two at
/// least it; an empty message is rejected.
pub fn ceil_log2(value: u64) -> Result<u32> {
    if value == 0 {
        return Err(Error::InvalidConfig);
    }
    Ok(64 - (value - 1).leading_zeros())
}

fn validate_stages(codes: &CodesPlan, first: &GroupPlan, stages: &[StageChoice]) -> Result<()> {
    if stages.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let mut ell = codes.get(first.main).ell;
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
        if stage.log_n < 1
            || stage.log_n > MOST_LOG_N
            || stage.mask_log_n < 1
            || stage.mask_log_n > MOST_LOG_N
        {
            return Err(Error::InvalidConfig);
        }
        if stage.k > 0 {
            let divisor = 1u64 << stage.k;
            if ell % divisor != 0 {
                return Err(Error::InvalidConfig);
            }
            ell /= divisor;
        }
        if (1u64 << stage.log_n) < ell {
            return Err(Error::InvalidConfig);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn stage(k: u32, h: u32, log_n: u32, mask_log_n: u32) -> StageChoice {
        StageChoice {
            k,
            h,
            log_n,
            mask_log_n,
        }
    }

    fn base_plan() -> RequestPlan {
        let table = code(17, 1 << 16, 8192);
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
                guo: vec![stage(4, 2, 12, 11), stage(3, 2, 12, 11), stage(0, 0, 13, 13)],
                omo: vec![stage(4, 3, 12, 12), stage(3, 3, 12, 11), stage(0, 0, 13, 12)],
                gu: vec![stage(4, 2, 12, 11), stage(3, 2, 12, 11), stage(0, 0, 13, 13)],
                om: vec![stage(4, 3, 12, 12), stage(3, 3, 12, 11), stage(0, 0, 13, 12)],
                own: vec![stage(4, 3, 12, 12), stage(3, 3, 12, 11), stage(0, 0, 13, 12)],
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
        }
    }

    #[test]
    fn a_complete_plan_passes() {
        base_plan().validate().expect("plan");
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
        plan.stages.guo[2] = stage(1, 0, 13, 13);
        assert!(plan.validate().is_err());
    }

    #[test]
    fn a_stage_that_does_not_divide_is_rejected() {
        let mut plan = base_plan();
        plan.stages.guo[0] = stage(10, 2, 12, 11);
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
}
