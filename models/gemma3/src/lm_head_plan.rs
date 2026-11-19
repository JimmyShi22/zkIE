//! Fixed request plan for the Gemma 3 output-head linear relation.
//!
//! The plan binds the committed table (2^10 rows of 2^18 entries beside the
//! answer pads), the two sets of group lines (the committer's one-time test
//! and a batch's test), the linear shape and the work scopes. The table row
//! holds the low eight column bits times the 2^10 hidden rows, so one row
//! covers 2^18 columns; the folded row and Omega share one main code and one
//! mask code here, so the two are tested together and no split message is
//! needed.

use zkie_core::error::Result;
use zkie_engine::pcs::request::code::{build_code_plan, CodeParams};
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
            code: code(19, 262_147, 8192)?,
            v: 10,
            delta_once_num: 1,
            delta_once_den: 8,
        },
        once: lines(),
        proof: lines(),
        lin: LinPlan {
            matrix_rows_log: 10,
            k_col: 18,
            k_lo: 8,
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
    fn the_plan_is_admitted_and_covers_the_full_column_domain() {
        let plan = lm_head_plan().expect("the fixed plan");
        plan.validate().expect("the plan checks its own numbers");
        let shape = plan.table_shape().expect("the table shape");
        assert_eq!(shape.rows(), 1024);
        assert_eq!(shape.code.n(), 1 << 19);
        assert_eq!(shape.plain(), 262_144);
        assert_eq!(plan.lin.matrix_rows_log, plan.table.v);
        assert_eq!(
            plan.lin.matrix_rows_log + plan.lin.k_lo,
            plan.lin.k_col
        );
        let matrix_entries = 1u64 << plan.lin.k_col;
        assert_eq!(matrix_entries, shape.plain());
    }

    #[test]
    fn a_forged_column_count_is_rejected() {
        let mut plan = lm_head_plan().unwrap();
        plan.lin.k_col = 17;
        assert!(plan.validate().is_err());
        let mut plan = lm_head_plan().unwrap();
        plan.table.code.ell = 262_146;
        assert!(plan.validate().is_err());
    }
}
