//! Fixed request plan for the GPT-2 output-head linear relation.
//!
//! The plan binds the committed table (2^10 rows of 2^16 entries), the main
//! and mask codes, the five groups, the later stages and the eight trees. The
//! numbers are protocol data; a runtime validates them with
//! [`RequestPlan::validate`] before any proof is made or read.

use zkie_core::error::Result;
use zkie_engine::pcs::request::code::{build_code_plan, CodeParams};
use zkie_engine::pcs::request::plan::{
    CodeId, GroupPlan, GroupsPlan, LinPlan, RequestPlan, ScopeWork, StageChoice, StagesPlan,
    TablePlan, TreePlan, TreesPlan, WorkPlan, WorkScope,
};

fn code(log_n: u32, ell: u64, r: u64) -> Result<CodeParams> {
    build_code_plan(log_n, ell, r)
}

fn scope(positions: u8) -> Result<ScopeWork> {
    ScopeWork::new(positions, 0)
}

fn group(main: CodeId, mask: CodeId, k: u32, scope: WorkScope) -> GroupPlan {
    GroupPlan {
        main,
        mask,
        k,
        h: 0,
        rows_log: 10,
        cube: 17,
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

fn tree(log_n: u32, rows: u64) -> TreePlan {
    TreePlan { log_n, rows }
}

fn codes() -> Result<zkie_engine::pcs::request::plan::CodesPlan> {
    Ok(zkie_engine::pcs::request::plan::CodesPlan {
        u: code(11, 512, 300)?,
        om: code(17, 4096, 3640)?,
        mask_u: code(10, 64, 300)?,
        mask_o: code(14, 64, 300)?,
        mask_c: code(14, 256, 300)?,
    })
}

/// The fixed request plan of the output-head relation.
pub fn lm_head_plan() -> Result<RequestPlan> {
    let codes = codes()?;
    Ok(RequestPlan {
        table: TablePlan {
            code: code(17, 1 << 16, 8192)?,
            v: 10,
            delta_once_num: 1,
            delta_once_den: 8,
        },
        codes,
        groups: GroupsPlan {
            guo: group(CodeId::U, CodeId::MaskU, 7, WorkScope::Once),
            omo: group(CodeId::Om, CodeId::MaskO, 4, WorkScope::Once),
            gu: group(CodeId::U, CodeId::MaskU, 7, WorkScope::Shared),
            om: group(CodeId::Om, CodeId::MaskO, 4, WorkScope::Shared),
            own: group(CodeId::Om, CodeId::MaskC, 4, WorkScope::Request),
        },
        stages: StagesPlan {
            guo: vec![
                stage(4, 2, 12, 11),
                stage(3, 2, 12, 11),
                stage(0, 0, 13, 13),
            ],
            omo: vec![
                stage(4, 3, 12, 12),
                stage(3, 3, 12, 11),
                stage(0, 0, 13, 12),
            ],
            gu: vec![
                stage(4, 2, 12, 11),
                stage(3, 2, 12, 11),
                stage(0, 0, 13, 13),
            ],
            om: vec![
                stage(4, 3, 12, 12),
                stage(3, 3, 12, 11),
                stage(0, 0, 13, 12),
            ],
            own: vec![
                stage(4, 3, 12, 12),
                stage(3, 3, 12, 11),
                stage(0, 0, 13, 12),
            ],
        },
        trees: TreesPlan {
            t: tree(17, 1024),
            gu: tree(11, 1),
            om: tree(17, 1),
            wc: tree(17, 1),
            gu2: tree(11, 1),
            om2: tree(17, 1),
            oc2: tree(17, 1),
            wc2: tree(17, 1),
        },
        lin: LinPlan {
            matrix_rows_log: 10,
            k_col: 16,
            k_lo: 6,
            key_instances: 1,
            proof_instances: 16,
        },
        work: WorkPlan::new(scope(0)?, scope(12)?, scope(12)?)?,
        samples: 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fixed_plan_is_admissible() {
        let plan = lm_head_plan().expect("plan");
        plan.validate().expect("plan validates");
        assert_eq!(plan.table.code.ell, 1 << plan.lin.k_col);
        assert_eq!(plan.table.v, plan.lin.matrix_rows_log);
        assert_eq!(plan.lin.matrix_rows_log + plan.lin.k_lo, plan.lin.k_col);
        assert_eq!(plan.groups.guo.cube_bits(&plan.codes).expect("cube"), 17);
        assert_eq!(plan.groups.omo.cube_bits(&plan.codes).expect("cube"), 17);
        assert_eq!(plan.work.request.positions, 0);
        assert_eq!(plan.work.shared.positions, 12);
        assert_eq!(plan.work.once.positions, 12);
        assert_eq!(plan.work.request.closeness, 0);
        assert_eq!(plan.work.shared.closeness, 0);
        assert_eq!(plan.work.once.closeness, 0);
    }

    #[test]
    fn a_wrong_table_row_count_is_rejected() {
        let mut plan = lm_head_plan().expect("plan");
        plan.table.v = 9;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn a_wrong_table_tree_is_rejected() {
        let mut plan = lm_head_plan().expect("plan");
        plan.trees.t.log_n = 18;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn a_wrong_last_stage_is_rejected() {
        let mut plan = lm_head_plan().expect("plan");
        plan.stages.omo[2] = stage(1, 0, 13, 12);
        assert!(plan.validate().is_err());
    }

    #[test]
    fn a_wrong_group_cube_is_rejected() {
        let mut plan = lm_head_plan().expect("plan");
        plan.groups.own.cube = 16;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn a_wrong_instance_count_is_rejected() {
        let mut plan = lm_head_plan().expect("plan");
        plan.lin.proof_instances = 0;
        assert!(plan.validate().is_err());
        let mut plan = lm_head_plan().expect("plan");
        plan.lin.key_instances = 2;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn a_mask_code_with_a_larger_rate_is_rejected() {
        let mut plan = lm_head_plan().expect("plan");
        plan.codes.mask_c = code(14, 4096, 300).expect("code");
        assert!(plan.validate().is_err());
    }
}
