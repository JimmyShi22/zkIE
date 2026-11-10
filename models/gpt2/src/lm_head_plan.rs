//! Fixed request plan for the GPT-2 output-head linear relation.
//!
//! The plan binds the committed table (2^10 rows of 2^16 entries beside the
//! answer pads), the two sets of group lines (the committer's one-time test
//! and a batch's test), the linear shape and the work scopes. The folded row
//! and Omega share one main code and one mask code here, so the two are tested
//! together and no split message is needed.

use zkie_core::error::Result;
use zkie_engine::pcs::request::code::{build_code_plan, CodeParams};
use zkie_engine::pcs::request::field::E_DEGREE;
use zkie_engine::pcs::request::plan::{
    CodeLine, GroupLine, LinLines, LinPlan, RequestPlan, ScopeWork, StageChoice, TablePlan,
    WorkPlan,
};

fn code(log_n: u32, ell: u64, r: u64) -> Result<CodeParams> {
    build_code_plan(log_n, ell, r)
}

fn scope(positions: u8) -> Result<ScopeWork> {
    ScopeWork::new(positions, 0)
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

/// The fixed request plan of the output-head relation.
pub fn lm_head_plan() -> Result<RequestPlan> {
    Ok(RequestPlan {
        table: TablePlan {
            code: code(17, 65539, 8192)?,
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
        work: WorkPlan::new(scope(0)?, scope(12)?, scope(12)?)?,
        samples: 1,
        salt_bytes: 40,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fixed_plan_is_admissible() {
        let plan = lm_head_plan().expect("plan");
        plan.validate().expect("plan validates");
        let shape = plan.table_shape().expect("shape");
        let pads = E_DEGREE as u64 * plan.samples;
        assert_eq!(
            plan.table.code.ell - pads,
            1 << plan.lin.k_col,
            "the matrix fills the message beside the pads"
        );
        assert_eq!(plan.table.v, plan.lin.matrix_rows_log);
        assert_eq!(plan.lin.matrix_rows_log + plan.lin.k_lo, plan.lin.k_col);
        assert_eq!(shape.cube_bits(), 17);
        assert_eq!(plan.work.request.positions, 0);
        assert_eq!(plan.work.shared.positions, 12);
        assert_eq!(plan.work.once.positions, 12);
        assert_eq!(plan.work.request.closeness, 0);
        assert_eq!(plan.work.shared.closeness, 0);
        assert_eq!(plan.work.once.closeness, 0);
    }

    #[test]
    fn the_folded_and_omega_groups_share_their_mask_code() {
        let plan = lm_head_plan().expect("plan");
        assert_eq!(plan.proof.folded.mask, plan.proof.omega.mask);
        assert_eq!(plan.once.folded.mask, plan.once.omega.mask);
        assert_eq!(plan.proof.own.main, plan.proof.omega.main);
    }

    #[test]
    fn a_wrong_table_row_count_is_rejected() {
        let mut plan = lm_head_plan().expect("plan");
        plan.table.v = 9;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn a_wrong_last_stage_is_rejected() {
        let mut plan = lm_head_plan().expect("plan");
        plan.proof.omega.stages[2] = stage(1, 0, 13, 12);
        assert!(plan.validate().is_err());
    }

    #[test]
    fn a_wrong_group_mask_is_rejected() {
        let mut plan = lm_head_plan().expect("plan");
        plan.proof.omega.mask = line(14, 7);
        assert!(plan.validate().is_err());
    }

    #[test]
    fn a_wrong_own_main_code_is_rejected() {
        let mut plan = lm_head_plan().expect("plan");
        plan.once.own.main = line(16, 11);
        assert!(plan.validate().is_err());
    }

    #[test]
    fn a_table_without_room_for_the_matrix_beside_its_pads_is_rejected() {
        let mut plan = lm_head_plan().expect("plan");
        plan.table.code.ell = 65538;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn the_once_and_proof_sets_are_checked_separately() {
        let mut plan = lm_head_plan().expect("plan");
        plan.once.folded.mask = line(14, 7);
        assert!(plan.validate().is_err());
        let mut plan = lm_head_plan().expect("plan");
        plan.proof.folded.mask = line(14, 7);
        assert!(plan.validate().is_err());
    }
}
