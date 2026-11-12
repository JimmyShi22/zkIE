//! Trusted two-stage planner for the hidden own proximity group.
//!
//! This is a small, independent check of the reference planning formulas:
//! Johnson distance/list size, mutual-correlated-agreement error, FOLD/BASE
//! error, sample and field terms, the CS batching term and the position
//! draws. It chooses the largest Johnson `m` whose rounds meet the reference
//! `2^-128` target with a small planning margin, caps the stage-0 positions at
//! the frozen mask tail minus the SEND openings, adds proof of work for the
//! remaining gap, and solves the last-stage tail from the BASE position count.
//!
//! The planner is recomputed from the trusted codes and stage choices; no
//! proof byte is an input. The floating-point comparisons use a fixed planning
//! margin, so a plan accepted here is not an exact interval proof; the margin
//! is part of the candidate parameter set and is recorded with the trace.

use zkie_core::error::{Error, Result};
use zkie_engine::pcs::request::code::CodeParams;

const FIELD_LOG2: f64 = 64.0;
const EXTENSION_LOG2: f64 = 3.0 * FIELD_LOG2 - 0.00000000101;
const TARGET_BITS: f64 = 128.0;
const PLANNING_MARGIN_BITS: f64 = 0.2;
const MAX_M: u64 = 4096;
const MAX_PLAN_ROUNDS: usize = 64;

/// The planner's result for the own two-stage chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TwoStagePlan {
    pub stage_zero_positions: usize,
    pub stage_zero_work_bits: u8,
    pub stage_zero_m: u64,
    pub last_positions: usize,
    pub last_main_r: u64,
    pub last_mask_r: u64,
    pub last_m: u64,
}

#[derive(Clone, Copy, Debug)]
struct Word {
    n: f64,
    k: f64,
    rho: f64,
}

#[derive(Clone, Copy, Debug)]
struct Group {
    main: Word,
    mask: Word,
    k: u32,
    h: u32,
}

fn word(code: CodeParams) -> Result<Word> {
    if code.log_n == 0 || code.log_n >= 63 || code.ell == 0 {
        return Err(Error::InvalidConfig);
    }
    let n = 2f64.powi(code.log_n as i32);
    let dim = (code.ell as f64) + (code.r as f64);
    if dim > n {
        return Err(Error::InvalidConfig);
    }
    Ok(Word {
        n,
        k: dim - 1.0,
        rho: (dim - 1.0) / n,
    })
}

fn johnson_count(w: Word, m: u64) -> f64 {
    let root = w.rho.sqrt();
    let big_m = (m as f64) + 0.5;
    let agree = root + root / (m as f64);
    let delta = 1.0 - agree;
    (2.0 * big_m.powi(5) + 3.0 * big_m * delta * w.rho) / (3.0 * w.rho * root) * w.n
        + big_m / root
}

fn johnson_list(w: Word, agree: f64) -> f64 {
    let x = (agree - w.rho) / (agree * agree - w.rho);
    if x.is_finite() && x > 0.0 {
        x.floor().max(1.0)
    } else {
        1.0
    }
}

fn johnson(w: Word, m: u64) -> (f64, f64, f64) {
    let root = w.rho.sqrt();
    let agree = root + root / (m as f64);
    (agree, johnson_count(w, m), johnson_list(w, agree))
}

fn at_distance(w: Word, agree: f64) -> (f64, f64) {
    let delta = 1.0 - agree;
    let errors = (delta * w.n).floor();
    let d = w.n - w.k;
    if 3.0 * errors < d {
        return (errors + 1.0, 1.0);
    }
    let root = w.rho.sqrt();
    let eta = agree - root;
    let m = (root / eta).ceil().max(3.0) as u64;
    let (_, count, list) = johnson(w, m);
    (count, list)
}

fn mca_log(count: f64) -> f64 {
    count.ln() - EXTENSION_LOG2 * std::f64::consts::LN_2
}

fn field_log(degree: f64) -> f64 {
    degree.ln() - EXTENSION_LOG2 * std::f64::consts::LN_2
}

fn powers_log(rows: f64) -> f64 {
    if rows <= 1.0 {
        f64::NEG_INFINITY
    } else {
        (rows - 1.0).ln() - EXTENSION_LOG2 * std::f64::consts::LN_2
    }
}

fn log_add(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    if m == f64::NEG_INFINITY {
        return m;
    }
    m + ((a - m).exp2() + (b - m).exp2()).log2()
}

fn group_error_log(group: Group, m: u64, cube_bits: u32, batched_rows: u64) -> Result<f64> {
    let (agree, main_count, main_list) = johnson(group.main, m);
    let (mask_count, mask_list) = at_distance(group.mask, agree);
    let list = main_list.max(mask_list);
    let list_log = list.ln();
    let mut closeness = mca_log(main_count);
    if group.k == 0 {
        closeness = log_add(closeness, list_log + field_log(1.0));
        closeness = log_add(closeness, mca_log(mask_count));
    } else {
        closeness = log_add(closeness, list_log + field_log(2.0));
        if group.h > 0 {
            closeness =
                log_add(closeness, mca_log(mask_count) + std::f64::consts::LN_2);
        }
    }
    let sample = list_log * 2.0
        - std::f64::consts::LN_2
        + (cube_bits as f64).ln()
        - EXTENSION_LOG2 * std::f64::consts::LN_2;
    let field = list_log + log_add(powers_log(batched_rows as f64), field_log(1.0));
    let bound = -(TARGET_BITS + PLANNING_MARGIN_BITS) * std::f64::consts::LN_2;
    if !closeness.is_finite() || closeness > bound {
        return Err(Error::InvalidConfig);
    }
    if sample > bound || field > bound {
        return Err(Error::InvalidConfig);
    }
    Ok(closeness)
}

fn largest_m(group: Group, cube_bits: u32, batched_rows: u64) -> Result<(u64, f64, f64)> {
    let mut best = None;
    for m in 3..=MAX_M {
        if let Ok(error) = group_error_log(group, m, cube_bits, batched_rows) {
            let (agree, _, list) = johnson(group.main, m);
            best = Some((m, agree, list, error));
        }
    }
    best.map(|(m, agree, list, _)| (m, agree, list))
        .ok_or(Error::InvalidConfig)
}

fn cube_bits(main: CodeParams, mask: CodeParams, k: u32, h: u32) -> u32 {
    let a = k + main.ell.trailing_zeros();
    let b = h + mask.ell.trailing_zeros();
    a.max(b) + 1
}

fn batched_rows(main: CodeParams) -> u64 {
    4 + 3 * main.r.max(1024)
}

/// Plan the own two-stage chain. `max_stage_zero_positions` is the frozen
/// group mask tail minus the SEND openings that share it. The last main and
/// mask codes carry the caller's chosen `log_n` and `ell`; their tails are
/// solved here.
pub fn plan_own_two_stage(
    main: CodeParams,
    mask: CodeParams,
    last_main: CodeParams,
    last_mask: CodeParams,
    k: u32,
    h: u32,
    max_stage_zero_positions: usize,
) -> Result<TwoStagePlan> {
    if k < 1 || h > k {
        return Err(Error::InvalidConfig);
    }
    if last_main.log_n != main.log_n
        || last_main.ell != main.ell
        || last_mask.log_n != mask.log_n
        || last_mask.ell <= mask.ell
    {
        return Err(Error::InvalidConfig);
    }
    if max_stage_zero_positions == 0 {
        return Err(Error::InvalidConfig);
    }
    let main_word = word(main)?;
    let mask_word = word(mask)?;
    let zero_group = Group {
        main: main_word,
        mask: mask_word,
        k,
        h,
    };
    let zero_cube = cube_bits(main, mask, k, h);
    let zero_rows = batched_rows(main);

    let mut last_r = 1u64;
    let mut last_m = 0u64;
    let mut last_list = 0.0;
    for _ in 0..MAX_PLAN_ROUNDS {
        let last_main_word = word(CodeParams {
            log_n: last_main.log_n,
            ell: last_main.ell,
            r: last_r,
        })?;
        let last_mask_word = word(CodeParams {
            log_n: last_mask.log_n,
            ell: last_mask.ell,
            r: last_r,
        })?;
        let last_group = Group {
            main: last_main_word,
            mask: last_mask_word,
            k: 0,
            h: 0,
        };
        let last_cube = cube_bits(last_main, last_mask, 0, 0);
        let (m, agree, list) = largest_m(last_group, last_cube, zero_rows)?;
        let positions = ((-(TARGET_BITS)) / agree.log2()).ceil();
        if !positions.is_finite() || positions <= 0.0 {
            return Err(Error::InvalidConfig);
        }
        let positions = positions as u64;
        if positions == last_r {
            last_m = m;
            last_list = list;
            break;
        }
        last_r = positions;
    }
    if last_m == 0 || last_r == 0 {
        return Err(Error::InvalidConfig);
    }
    if last_r > (1u64 << last_main.log_n) {
        return Err(Error::ResourceLimit);
    }

    let zero = largest_m(zero_group, zero_cube, zero_rows)?;
    let (zero_m, zero_agree, _zero_list) = zero;
    let t = max_stage_zero_positions as f64;
    let miss_log = t * zero_agree.log2();
    let cs_rows = 2.0 + 3.0 * t;
    let extra_log = last_list.log2() + (cs_rows - 1.0).log2() - EXTENSION_LOG2;
    let total_log = log_add(miss_log, extra_log);
    let work = (TARGET_BITS + total_log).ceil();
    if !work.is_finite() || work < 0.0 || work > 58.0 {
        return Err(Error::InvalidConfig);
    }
    Ok(TwoStagePlan {
        stage_zero_positions: max_stage_zero_positions,
        stage_zero_work_bits: work as u8,
        stage_zero_m: zero_m,
        last_positions: last_r as usize,
        last_main_r: last_r,
        last_mask_r: last_r,
        last_m,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::code::build_code_plan;

    #[test]
    fn real_two_stage_plan_is_the_candidate() {
        let main = build_code_plan(17, 65_536, 8_192).unwrap();
        let mask = build_code_plan(17, 512, 256).unwrap();
        let last_main = build_code_plan(17, 65_536, 1).unwrap();
        let last_mask = build_code_plan(17, 16_384, 1).unwrap();
        let plan = plan_own_two_stage(main, mask, last_main, last_mask, 14, 2, 254).unwrap();
        assert_eq!(plan.stage_zero_positions, 254);
        assert_eq!(plan.stage_zero_work_bits, 24);
        assert_eq!(plan.last_positions, 259);
        assert_eq!(plan.last_main_r, 259);
        assert_eq!(plan.last_mask_r, 259);
    }

    #[test]
    fn a_smaller_mask_tail_rejects_the_plan() {
        let main = build_code_plan(17, 65_536, 8_192).unwrap();
        let mask = build_code_plan(17, 512, 256).unwrap();
        let last_main = build_code_plan(17, 65_536, 1).unwrap();
        let last_mask = build_code_plan(17, 16_384, 1).unwrap();
        assert!(plan_own_two_stage(main, mask, last_main, last_mask, 14, 2, 0).is_err());
    }
}
