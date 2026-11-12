//! Trusted interval planner for the hidden proximity chains.
//!
//! The planner recomputes every stage distance, list size, round error and
//! position count from the trusted codes and stage choices. It deliberately
//! does not use floating point for a decision: `interval` keeps exact integers
//! where possible and rounds the rest outward, so a plan is accepted only when
//! the safe end of every bound meets the target. The planner is called before
//! the first query by both prover and verifier; a proof byte is never an input.
//!
//! The estimated values are only a planning result. The final total-bound
//! accounting and its coverage limits are recorded with the acceptance
//! evidence, not claimed as a formal proof here.

use std::cmp::Ordering;

use zkie_core::error::{Error, Result};
use zkie_engine::pcs::request::code::CodeParams;

use super::interval::{surely_le, surely_lt, Big, Bound, Real};

const TARGET_BITS: u32 = 128;
const FIELD_DEGREE: u32 = 3;
const MAX_M: u64 = 4096;
const MAX_STAGE_ROUNDS: usize = 64;

/// The planner's result for the two-stage own chain.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Word {
    log_n: u32,
    ell: u64,
    k: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    Third,
    Johnson,
}

#[derive(Clone, Debug)]
struct Proximity {
    agree: Bound,
    delta: Bound,
    count: Bound,
    list: Big,
    m: u64,
    source: Source,
}

impl Proximity {
    fn failed() -> Self {
        Self {
            agree: Bound::failed(),
            delta: Bound::failed(),
            count: Bound::failed(),
            list: Big::from_u64(1),
            m: 0,
            source: Source::Johnson,
        }
    }

    fn bad(&self) -> bool {
        self.agree.bad || self.delta.bad || self.count.bad
    }
}

#[derive(Clone, Debug)]
struct GroupShape {
    main: Word,
    mask: Word,
    k: u32,
    h: u32,
}

#[derive(Clone, Debug)]
struct Analysis {
    at: Proximity,
    mask_at: Option<Proximity>,
    list: Big,
    bad: bool,
}

#[derive(Clone, Copy, Debug)]
struct GroupLoad {
    cube_bits: u32,
    messages: u64,
    batched_rows: u64,
}

fn word(code: CodeParams) -> Result<Word> {
    if code.log_n == 0 || code.log_n >= 63 || code.ell == 0 {
        return Err(Error::InvalidConfig);
    }
    let n = code.n();
    let dim = code.ell.checked_add(code.r).ok_or(Error::InvalidConfig)?;
    if dim > n || dim == 0 {
        return Err(Error::InvalidConfig);
    }
    Ok(Word {
        log_n: code.log_n,
        ell: code.ell,
        k: dim - 1,
    })
}

fn word_n(word: &Word) -> Result<u64> {
    1u64
        .checked_shl(word.log_n)
        .ok_or(Error::ResourceLimit)
}

fn rho_of(word: &Word) -> Result<Bound> {
    let n = word_n(word)?;
    Ok(Bound::ratio(
        &Big::from_u64(word.k),
        &Big::from_u64(n),
    ))
}

fn card_e() -> Bound {
    let p = Big::two_to(64)
        .sub(&Big::two_to(32))
        .add(&Big::from_u64(1));
    Bound::from_big(Big::pow(p, FIELD_DEGREE as u64))
}

fn field_error(degree: u64) -> Bound {
    Bound::from_big(Big::from_u64(degree)).div(&card_e())
}

fn field_error_nonzero(degree: u64) -> Bound {
    let card = card_e();
    let divisor = Bound::pair(
        card.lo.sub(&Real::one(), super::interval::Dir::Down),
        card.hi.sub(&Real::one(), super::interval::Dir::Up),
    );
    Bound::from_big(Big::from_u64(degree)).div(&divisor)
}

fn powers_error(rows: u64) -> Bound {
    Bound::from_big(Big::from_u64(rows.saturating_sub(1))).div(&card_e())
}

fn mca_error(proximity: &Proximity) -> Bound {
    proximity.count.clone().div(&card_e())
}

fn johnson_list(agree: &Bound, lambda: &Bound) -> (Big, bool) {
    let numerator = agree.sub(lambda);
    let denominator = agree.mul(agree).sub(lambda);
    let ratio = numerator.div(&denominator);
    let bad = ratio.bad;
    let list = if bad {
        Big::from_u64(1)
    } else {
        ratio.hi.floor()
    };
    (list, bad)
}

fn johnson_count(word: &Word, rho: &Bound, delta: &Bound, m: u64) -> Bound {
    let numerator = Big::from_u64(2)
        .mul(&Big::from_u64(m))
        .add(&Big::from_u64(1));
    let big_m = Bound::from_big(numerator).div(&Bound::from_u64(2));
    let root = rho.sqrt();
    let five = Bound::from_u64(2)
        .mul(&big_m.pow(5))
        .add(
            &Bound::from_u64(3)
                .mul(&big_m)
                .mul(delta)
                .mul(rho),
        );
    let denominator = Bound::from_u64(3).mul(rho).mul(&root);
    let length = word_n(word).map(Bound::from_u64).unwrap_or_else(|_| Bound::failed());
    five.div(&denominator).mul(&length).add(&big_m.div(&root))
}

fn johnson(word: &Word, m: u64) -> Result<Proximity> {
    let rho = rho_of(word)?;
    let root = rho.sqrt();
    let eta = root.div(&Bound::from_u64(m));
    let agree = root.add(&eta);
    let delta = Bound::from_u64(1).sub(&agree);
    let count = johnson_count(word, &rho, &delta, m);
    let (list, bad) = johnson_list(&agree, &rho);
    let mut result = Proximity {
        agree,
        delta,
        count,
        list,
        m,
        source: Source::Johnson,
    };
    if m < 3 || word.k >= word_n(word)? || bad {
        result.count = Bound::failed();
    }
    Ok(result)
}

fn repeated(word: &Word, log_n: u32) -> Result<Word> {
    let n = word_n(word)?;
    if log_n <= word.log_n {
        return Ok(*word);
    }
    let extra = log_n - word.log_n;
    let k = word
        .k
        .checked_shl(extra)
        .ok_or(Error::ResourceLimit)?;
    if k >= (1u64 << log_n) {
        return Err(Error::InvalidConfig);
    }
    Ok(Word {
        log_n,
        ell: word.ell,
        k,
    })
}

fn at_distance(word: &Word, agree: &Bound) -> Result<Proximity> {
    if agree.bad || word.k >= word_n(word)? {
        return Ok(Proximity::failed());
    }
    let n = word_n(word)?;
    let distance = n - word.k;
    let delta = Bound::from_u64(1).sub(agree);
    if delta.bad {
        let mut failed = Proximity::failed();
        failed.agree = agree.clone();
        failed.delta = delta;
        return Ok(failed);
    }
    let n_bound = Bound::from_u64(n);
    let errors = delta.mul(&n_bound).hi.floor();
    let three_errors = Big::from_u64(3).mul(&errors);
    if three_errors.cmp(&Big::from_u64(distance)) == Ordering::Less {
        return Ok(Proximity {
            agree: agree.clone(),
            delta,
            count: Bound::from_big(errors.add(&Big::from_u64(1))),
            list: Big::from_u64(1),
            m: 0,
            source: Source::Third,
        });
    }
    let rho = rho_of(word)?;
    let root = rho.sqrt();
    let eta = agree.sub(&root);
    let q = root.div(&eta);
    if q.bad || !q.hi.ceil().fits_u64() {
        let mut failed = Proximity::failed();
        failed.agree = agree.clone();
        failed.delta = delta;
        return Ok(failed);
    }
    let m = q.hi.ceil().to_u64().max(3);
    let count = johnson_count(word, &rho, &delta, m);
    let list = if !surely_lt(
        &Bound::from_u64(2).mul(&delta).mul(&n_bound),
        &Bound::from_u64(distance),
    ) {
        let (list, bad) = johnson_list(agree, &rho);
        if bad {
            let mut failed = Proximity::failed();
            failed.agree = agree.clone();
            failed.delta = delta;
            return Ok(failed);
        }
        list
    } else {
        Big::from_u64(1)
    };
    Ok(Proximity {
        agree: agree.clone(),
        delta,
        count,
        list,
        m,
        source: Source::Johnson,
    })
}

fn group_fault(group: &GroupShape) -> bool {
    if group.h > group.k || group.mask.log_n > group.main.log_n {
        return true;
    }
    if group.k > 0 && group.h == 0 {
        let mask_rate = (group.mask.k as u128) * (1u128 << group.main.log_n);
        let main_rate = (group.main.k as u128) * (1u128 << group.mask.log_n);
        if mask_rate > main_rate {
            return true;
        }
    }
    false
}

fn analysis_at(group: &GroupShape, at: Proximity) -> Result<Analysis> {
    let mut analysis = Analysis {
        at,
        mask_at: None,
        list: Big::from_u64(1),
        bad: group_fault(group),
    };
    analysis.list = analysis.at.list.clone();
    if analysis.bad {
        analysis.at.count = Bound::failed();
    }
    let mask_word = repeated(&group.mask, group.main.log_n)?;
    let mask_at = at_distance(&mask_word, &analysis.at.agree)?;
    if mask_at.list.cmp(&analysis.list) == Ordering::Greater {
        analysis.list = mask_at.list.clone();
    }
    if mask_at.source != Source::Third {
        let mut mask_at = mask_at;
        mask_at.count = mask_at
            .count
            .add(&Bound::from_big(mask_at.list.clone()));
        analysis.mask_at = Some(mask_at);
    } else {
        analysis.mask_at = Some(mask_at);
    }
    if analysis.at.bad() || analysis.mask_at.as_ref().is_some_and(Proximity::bad) {
        analysis.bad = true;
    }
    Ok(analysis)
}

fn mask_rules(group: &GroupShape) -> bool {
    (group.mask.k as u128) * (1u128 << group.main.log_n)
        > (group.main.k as u128) * (1u128 << group.mask.log_n)
}

fn analysis_of_m(group: &GroupShape, m: u64) -> Result<Analysis> {
    if mask_rules(group) {
        let z = johnson(&repeated(&group.mask, group.main.log_n)?, m)?;
        let mut analysis = analysis_at(group, at_distance(&group.main, &z.agree)?)?;
        let mut z = z;
        z.count = z.count.add(&Bound::from_big(z.list.clone()));
        if z.list.cmp(&analysis.list) == Ordering::Greater {
            analysis.list = z.list.clone();
        }
        analysis.mask_at = Some(z);
        if analysis.at.bad() || analysis.mask_at.as_ref().is_some_and(Proximity::bad) {
            analysis.bad = true;
        }
        Ok(analysis)
    } else {
        analysis_at(group, johnson(&group.main, m)?)
    }
}

fn fold_round_error(group: &GroupShape, analysis: &Analysis, round: u32) -> Bound {
    let mut error = mca_error(&analysis.at)
        .add(&Bound::from_big(analysis.list.clone()).mul(&field_error(2)));
    if group.h > 0 && round <= group.h {
        if let Some(mask_at) = &analysis.mask_at {
            error = error.add(&Bound::from_u64(2).mul(&mca_error(mask_at)));
        }
    }
    error
}

fn base_error(group: &GroupShape, analysis: &Analysis) -> Bound {
    let mut error = mca_error(&analysis.at)
        .add(&Bound::from_big(analysis.list.clone()).mul(&field_error(1)));
    if let Some(mask_at) = &analysis.mask_at {
        error = error.add(&mca_error(mask_at));
    }
    error
}

fn sample_error(list: &Big, cube_bits: u32, answers: u32) -> Bound {
    let pair = Bound::from_big(list.mul(list)).div(&Bound::from_u64(2));
    let per_answer = if cube_bits >= 2 {
        field_error(cube_bits as u64)
    } else {
        field_error_nonzero(1)
    };
    pair.mul(&per_answer.pow(answers as u64))
}

fn goal(work_bits: u32) -> Bound {
    let bits = TARGET_BITS.saturating_sub(work_bits);
    Bound::from_real(Real::two_to(-(bits as i64)))
}

fn holds_for_m(group: &GroupShape, load: GroupLoad, m: u64) -> bool {
    let Ok(analysis) = analysis_of_m(group, m) else {
        return false;
    };
    if analysis.bad {
        return false;
    }
    let worst = if group.k > 0 {
        fold_round_error(group, &analysis, 1)
    } else {
        base_error(group, &analysis)
    };
    let closeness_ok = surely_le(&Bound::from_real(worst.error()), &goal(0));
    let sample = Bound::from_u64(load.messages).mul(&sample_error(
        &analysis.list,
        load.cube_bits,
        1,
    ));
    let sample_ok = surely_le(&Bound::from_real(sample.error()), &goal(0));
    let field = Bound::from_big(analysis.list.clone()).mul(
        &powers_error(load.batched_rows).add(&field_error(1)),
    );
    let field_ok = surely_le(&Bound::from_real(field.error()), &goal(0));
    closeness_ok && sample_ok && field_ok
}

fn largest_m(group: &GroupShape, load: GroupLoad) -> u64 {
    let good = |m: u64| holds_for_m(group, load, m);
    let mut low = 3;
    while low < 67 && !good(low) {
        low += 1;
    }
    if low == 67 {
        return 0;
    }
    let mut high = low + 1;
    while high < MAX_M && good(high) {
        low = high;
        high *= 2;
    }
    if high >= MAX_M {
        if good(MAX_M) {
            return MAX_M;
        }
        high = MAX_M;
    }
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        if good(middle) {
            low = middle;
        } else {
            high = middle;
        }
    }
    low
}

fn positions(agree: &Bound, bits: u32, extra: &Bound, factor: u64) -> u64 {
    if factor == 0 {
        return 0;
    }
    let goal = Real::div(
        &Real::two_to(-(bits as i64)),
        &Real::from_u64(factor),
        super::interval::Dir::Down,
    );
    let goal = Bound::from_real(goal);
    if agree.bad
        || extra.bad
        || agree.hi.cmp(&Real::one()) != Ordering::Less
        || extra.hi.cmp(&goal.hi) != Ordering::Less
    {
        return 0;
    }
    let ok = |t: u64| {
        let error = Real::pow(agree.hi.clone(), t, super::interval::Dir::Up);
        let sum = error.add(&extra.hi, super::interval::Dir::Up);
        sum.cmp(&goal.hi) != Ordering::Greater
    };
    let mut high = 1u64;
    while !ok(high) {
        match high.checked_mul(2) {
            Some(next) if next <= (1u64 << 32) => high = next,
            _ => return 0,
        }
    }
    let mut low = high / 2;
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        if ok(middle) {
            high = middle;
        } else {
            low = middle;
        }
    }
    high
}

fn scope_positions(agree: &Bound, work_bits: u32, extra: &Bound) -> u64 {
    let bits = TARGET_BITS.saturating_sub(work_bits);
    positions(agree, bits, extra, 1)
}

fn stage_cube_bits(group: &GroupShape) -> u32 {
    let main = group.k + ceil_log2(group.main.ell);
    let mask = group.h + ceil_log2(group.mask.ell);
    main.max(mask) + 1
}

fn ceil_log2(value: u64) -> u32 {
    if value <= 1 {
        return 0;
    }
    u64::BITS - (value - 1).leading_zeros()
}

fn batched_rows(word: &Word) -> u64 {
    4 + 3 * word.k.max(1024)
}

fn group_load(group: &GroupShape, previous_main_r: u64) -> GroupLoad {
    GroupLoad {
        cube_bits: stage_cube_bits(group),
        messages: 1,
        batched_rows: 4 + 3 * previous_main_r.max(1024),
    }
}

fn group_shape(main: CodeParams, mask: CodeParams, k: u32, h: u32) -> Result<GroupShape> {
    Ok(GroupShape {
        main: word(main)?,
        mask: word(mask)?,
        k,
        h,
    })
}

fn plan_last(
    last_main: CodeParams,
    last_mask: CodeParams,
    previous_main_r: u64,
) -> Result<(u64, u64, Analysis)> {
    let mut last_r = 1u64;
    for _ in 0..MAX_STAGE_ROUNDS {
        let main = CodeParams {
            log_n: last_main.log_n,
            ell: last_main.ell,
            r: last_r,
        };
        let mask = CodeParams {
            log_n: last_mask.log_n,
            ell: last_mask.ell,
            r: last_r,
        };
        let group = group_shape(main, mask, 0, 0)?;
        let load = group_load(&group, previous_main_r);
        let m = largest_m(&group, load);
        if m == 0 {
            return Err(Error::InvalidConfig);
        }
        let analysis = analysis_of_m(&group, m)?;
        if analysis.bad {
            return Err(Error::InvalidConfig);
        }
        let next = scope_positions(&analysis.at.agree, 0, &Bound::from_u64(0));
        if next == 0 {
            return Err(Error::InvalidConfig);
        }
        if next == last_r {
            return Ok((last_r, m, analysis));
        }
        if next > (1u64 << last_main.log_n) {
            return Err(Error::ResourceLimit);
        }
        last_r = next;
    }
    Err(Error::InvalidConfig)
}

/// Plan the own two-stage chain with outward-rounded bounds. The stage-0
/// position count comes from the reference iteration with the smallest proof
/// of work that fits `max_stage_zero_positions`; the last tail is the fixed
/// point of the BASE position count.
pub fn plan_own_two_stage(
    main: CodeParams,
    mask: CodeParams,
    last_main: CodeParams,
    last_mask: CodeParams,
    k: u32,
    h: u32,
    max_stage_zero_positions: usize,
) -> Result<TwoStagePlan> {
    if k == 0 || h > k || max_stage_zero_positions == 0 {
        return Err(Error::InvalidConfig);
    }
    if last_main.log_n != main.log_n
        || last_main.ell != main.ell
        || last_mask.log_n != mask.log_n
        || last_mask.ell <= mask.ell
    {
        return Err(Error::InvalidConfig);
    }
    let stage_zero = group_shape(main, mask, k, h)?;
    let zero_load = group_load(&stage_zero, main.r);
    let zero_m = largest_m(&stage_zero, zero_load);
    if zero_m == 0 {
        return Err(Error::InvalidConfig);
    }
    let zero_analysis = analysis_of_m(&stage_zero, zero_m)?;
    if zero_analysis.bad {
        return Err(Error::InvalidConfig);
    }
    let (last_r, last_m, last_analysis) = plan_last(last_main, last_mask, main.r)?;

    let mut chosen = None;
    for work in 0u32..=58 {
        let mut t = 0u64;
        let mut converged = false;
        for _ in 0..MAX_STAGE_ROUNDS {
            let rows = 1u64
                .checked_add(1)
                .and_then(|value| value.checked_add(3u64.checked_mul(t)?))
                .ok_or(Error::ResourceLimit)?;
            let extra = Bound::from_big(last_analysis.list.clone())
                .mul(&powers_error(rows));
            let next = scope_positions(&zero_analysis.at.agree, work, &extra);
            if next == 0 {
                break;
            }
            if next == t {
                converged = true;
                break;
            }
            t = next;
        }
        if !converged || t == 0 {
            continue;
        }
        if t as usize <= max_stage_zero_positions {
            chosen = Some((t as usize, work as u8));
            break;
        }
    }
    let (stage_zero_positions, stage_zero_work_bits) =
        chosen.ok_or(Error::InvalidConfig)?;
    Ok(TwoStagePlan {
        stage_zero_positions,
        stage_zero_work_bits,
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
        assert_eq!(plan.stage_zero_positions, 252);
        assert_eq!(plan.stage_zero_work_bits, 24);
        assert_eq!(plan.stage_zero_m, 616);
        assert_eq!(plan.last_positions, 259);
        assert_eq!(plan.last_main_r, 259);
        assert_eq!(plan.last_mask_r, 259);
        assert_eq!(plan.last_m, 595);
    }

    #[test]
    fn reference_interval_accounting_is_visible() {
        use super::super::interval::bits_of;

        let main = build_code_plan(17, 65_536, 8_192).unwrap();
        let mask = build_code_plan(17, 512, 256).unwrap();
        let last_main = build_code_plan(17, 65_536, 259).unwrap();
        let last_mask = build_code_plan(17, 16_384, 259).unwrap();
        let stage_zero = group_shape(main, mask, 14, 2).unwrap();
        let last = group_shape(last_main, last_mask, 0, 0).unwrap();
        let zero_load = group_load(&stage_zero, main.r);
        let zero_m = largest_m(&stage_zero, zero_load);
        let zero = analysis_of_m(&stage_zero, zero_m).unwrap();
        let last_load = group_load(&last, main.r);
        let last_m = largest_m(&last, last_load);
        let last_analysis = analysis_of_m(&last, last_m).unwrap();
        assert_eq!(zero_m, 616);
        assert_eq!(last_m, 595);

        let mut terms: Vec<(&str, Bound)> = Vec::new();
        terms.push((
            "own_fold_epsilon",
            Bound::from_big(zero.list.clone()).mul(&field_error(1)),
        ));
        for round in 1..=14u32 {
            terms.push((
                "own_fold_round",
                fold_round_error(&stage_zero, &zero, round),
            ));
        }
        terms.push((
            "own_bat",
            Bound::from_big(zero.list.clone()).mul(&powers_error(4 + 3 * 8_192)),
        ));
        terms.push((
            "own_cs_query",
            Bound::from_real(Real::two_to(-24))
                .mul(&zero.at.agree.pow(252))
                .add(
                    &Bound::from_big(last_analysis.list.clone())
                        .mul(&powers_error(1 + 1 + 3 * 252)),
                ),
        ));
        let next_cube = 17;
        terms.push((
            "own_cs_sample",
            sample_error(&last_analysis.list, next_cube, 1),
        ));
        terms.push(("own_base", base_error(&last, &last_analysis)));
        terms.push(("own_base_query", last_analysis.at.agree.pow(259)));
        terms.push((
            "own_send_sample",
            Bound::from_u64(4).mul(&sample_error(&zero.list, 31, 1)),
        ));

        let shared_main = build_code_plan(17, 65_536, 8_192).unwrap();
        let shared_mask = build_code_plan(17, 512, 256).unwrap();
        let shared = group_shape(shared_main, shared_mask, 2, 0).unwrap();
        let shared_m = largest_m(&shared, group_load(&shared, shared_main.r));
        let shared_analysis = analysis_of_m(&shared, shared_m).unwrap();
        terms.push((
            "shared_fold_epsilon",
            Bound::from_big(shared_analysis.list.clone()).mul(&field_error(1)),
        ));
        for round in 1..=2u32 {
            terms.push((
                "shared_fold_round",
                fold_round_error(&shared, &shared_analysis, round),
            ));
        }
        terms.push((
            "shared_bat",
            Bound::from_big(shared_analysis.list.clone())
                .mul(&powers_error(4 + 3 * 8_192)),
        ));
        terms.push((
            "shared_send_sample",
            Bound::from_u64(4).mul(&sample_error(&shared_analysis.list, 31, 1)),
        ));
        terms.push((
            "shared_cs_sample",
            sample_error(&last_analysis.list, 17, 1),
        ));
        let table = word(build_code_plan(17, 65_539, 8_192).unwrap()).unwrap();
        let table_agree = Bound::from_u64(1)
            .sub(&Bound::ratio(&Big::from_u64(16_384), &Big::from_u64(131_072)));
        let table_proximity = at_distance(&table, &table_agree).unwrap();
        terms.push((
            "table_columns_603",
            Bound::from_real(Real::two_to(-12))
                .mul(&table_proximity.agree.pow(603)),
        ));
        terms.push((
            "table_record",
            sample_error(&table_proximity.list, 17, 1),
        ));
        let table_fold = Bound::from_u64(10)
            .mul(&mca_error(&table_proximity))
            .add(
                &Bound::from_big(table_proximity.list.clone())
                    .mul(&field_error(10)),
            );
        terms.push(("table_fold", table_fold));
        let card = card_e();
        terms.push((
            "closure_range",
            Bound::from_big(Big::from_u64(16 * 166)).div(&card),
        ));
        terms.push((
            "psc_rounds",
            Bound::from_big(Big::from_u64(30 + 16 * 69)).div(&card),
        ));
        terms.push((
            "chi_zero",
            Bound::from_big(Big::from_u64(1)).div(&card),
        ));

        let mut union = Bound::from_u64(0);
        let mut worst = Bound::from_u64(0);
        for (name, term) in &terms {
            println!("term={name} bits={:.6}", bits_of(&term.error()));
            union = union.add(term);
            if worst.error().cmp(&term.error()) == Ordering::Less {
                worst = term.clone();
            }
        }
        println!("union_bits={:.6}", bits_of(&union.error()));
        println!("worst_round_bits={:.6}", bits_of(&worst.error()));
        assert!(!union.bad && !worst.bad);
        assert!(bits_of(&union.error()) < 128.0);
    }

    #[test]
    fn a_smaller_mask_tail_rejects_the_plan() {
        let main = build_code_plan(17, 65_536, 8_192).unwrap();
        let mask = build_code_plan(17, 512, 256).unwrap();
        let last_main = build_code_plan(17, 65_536, 1).unwrap();
        let last_mask = build_code_plan(17, 16_384, 1).unwrap();
        assert!(plan_own_two_stage(main, mask, last_main, last_mask, 14, 2, 0).is_err());
    }
    #[test]
    fn position_search_boundaries_and_bad_searches() {
        let zero = Bound::from_u64(0);
        let half = Bound::from_u64(1).div(&Bound::from_u64(2));
        assert_eq!(positions(&half, 128, &zero, 1), 128);
        let three_quarters = Bound::from_u64(3).div(&Bound::from_u64(4));
        assert_eq!(positions(&three_quarters, 128, &zero, 1), 309);
        let one = Bound::from_u64(1);
        assert_eq!(positions(&one, 128, &zero, 1), 0);
        let goal = Bound::from_real(Real::two_to(-128));
        assert_eq!(positions(&half, 128, &goal, 1), 0);
        assert_eq!(positions(&half, 128, &zero, 0), 0);
        let near_one = Bound::from_u64(1)
            .sub(&Bound::from_real(Real::two_to(-40)));
        assert_eq!(positions(&near_one, 128, &zero, 1), 0);

        let tiny = GroupShape {
            main: Word { log_n: 2, ell: 3, k: 2 },
            mask: Word { log_n: 2, ell: 3, k: 2 },
            k: 1,
            h: 0,
        };
        let load = GroupLoad {
            cube_bits: 3,
            messages: 1,
            batched_rows: u64::MAX,
        };
        assert_eq!(largest_m(&tiny, load), 0);
    }

}
