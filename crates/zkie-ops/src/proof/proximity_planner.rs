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

use super::interval::{bits_of, surely_le, surely_lt, Big, Bound, Real};

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
    /// Whether the group's mask rows exist. The two-stage solver always uses
    /// them; the first-stage derivation also sizes a main-only group on the
    /// way up.
    mask_enabled: bool,
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
    if !group.mask_enabled {
        return Ok(analysis);
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
    if !group.mask_enabled {
        return false;
    }
    (group.mask.k as u128) * (1u128 << group.main.log_n)
        > (group.main.k as u128) * (1u128 << group.mask.log_n)
}

fn analysis_of_m(group: &GroupShape, m: u64) -> Result<Analysis> {
    if !group.mask_enabled {
        return analysis_at(group, johnson(&group.main, m)?);
    }
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
    if group.mask_enabled && group.h > 0 && round <= group.h {
        if let Some(mask_at) = &analysis.mask_at {
            error = error.add(&Bound::from_u64(2).mul(&mca_error(mask_at)));
        }
    }
    error
}

fn base_error(group: &GroupShape, analysis: &Analysis) -> Bound {
    let mut error = mca_error(&analysis.at)
        .add(&Bound::from_big(analysis.list.clone()).mul(&field_error(1)));
    if group.mask_enabled {
        if let Some(mask_at) = &analysis.mask_at {
            error = error.add(&mca_error(mask_at));
        }
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

/// The trusted parameters of one shared linear stage: the table's row
/// variables, the shared point's low variables, how many requests share the
/// point, and the comparison point's variables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedLinShape {
    pub v: u32,
    pub k_lo: u32,
    pub instances: u64,
    pub rows_log: u32,
}

/// The derived query count and error terms of one shared linear stage.
#[derive(Clone, Debug)]
pub struct SharedLinBounds {
    /// The fold point's error: `instances * field_error(v + k_lo)` plus the
    /// folded table's error.
    pub point: Bound,
    /// The number of columns drawn for the table's agreement.
    pub columns: u64,
    /// `agree^columns`, the column round's error.
    pub column_error: Bound,
    /// `field_error(rows_log)`, the comparison point's error.
    pub zeta: Bound,
}

fn table_fold_error(v: u32, proximity_error: &Bound, list: &Big) -> Bound {
    let mut error = Bound::from_u64(0);
    for _ in 1..=v {
        error = error.add(proximity_error);
    }
    error.add(&Bound::from_big(list.clone()).mul(&field_error(u64::from(v))))
}

/// Derives a shared linear stage's column query count and error terms from the
/// table's tested distance. `agree` is the distance's agreement, `list` its
/// list size and `proximity_error` its per-variable proximity error; the
/// folded table's error is `v * proximity_error + list * field_error(v)`. The
/// column count is the smallest `nu` whose `agree^nu` meets the query target
/// with no extra term. Only this layer is derived; the fold round count and
/// the hiding polynomial are not.
pub fn shared_lin_bounds(
    shape: SharedLinShape,
    agree: &Bound,
    list: &Big,
    proximity_error: &Bound,
    closeness_work_bits: u32,
    query_work_bits: u32,
) -> Result<SharedLinBounds> {
    if shape.v == 0 || shape.instances == 0 || shape.rows_log == 0 {
        return Err(Error::InvalidConfig);
    }
    let fold = table_fold_error(shape.v, proximity_error, list);
    let point = Bound::from_u64(shape.instances)
        .mul(&field_error(u64::from(shape.v) + u64::from(shape.k_lo)))
        .add(&fold);
    if !surely_le(&Bound::from_real(point.error()), &goal(closeness_work_bits)) {
        return Err(Error::InvalidConfig);
    }
    let columns = scope_positions(agree, query_work_bits, &Bound::from_u64(0));
    if columns == 0 {
        return Err(Error::InvalidConfig);
    }
    let column_error = agree.pow(columns);
    if !surely_le(&column_error, &goal(query_work_bits)) {
        return Err(Error::InvalidConfig);
    }
    Ok(SharedLinBounds {
        point,
        columns,
        column_error,
        zeta: field_error(u64::from(shape.rows_log)),
    })
}

/// The trusted parameters of one shared linear stage's bound check. The
/// instance count is not carried: one request is one instance, and the check
/// does not accept a caller's instance count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedLinPlanFault {
    pub v: u32,
    pub k_lo: u32,
    pub rows_log: u32,
    pub m: u64,
    pub column_draws: u64,
    pub point_work_bits: u32,
    pub column_work_bits: u32,
    pub zeta_work_bits: u32,
}

fn table_proximity(code: CodeParams, m: u64) -> Result<Proximity> {
    if code.log_n >= 64 || m < 3 {
        return Err(Error::InvalidConfig);
    }
    let word = Word {
        log_n: code.log_n,
        ell: code.ell,
        k: code.r,
    };
    let at = johnson(&word, m)?;
    if at.bad() {
        return Err(Error::InvalidConfig);
    }
    Ok(at)
}

/// The smallest number of table columns whose agreement power meets the query
/// target at the given one-time distance and work bits.
pub fn shared_lin_columns(
    shape: SharedLinShape,
    code: CodeParams,
    m: u64,
    point_work_bits: u32,
    column_work_bits: u32,
) -> Result<u64> {
    let at = table_proximity(code, m)?;
    let bounds = shared_lin_bounds(
        shape,
        &at.agree,
        &at.list,
        &mca_error(&at),
        point_work_bits,
        column_work_bits,
    )?;
    Ok(bounds.columns)
}

/// Checks a frozen shared linear plan: the table's one-time distance must
/// exist, the plan's actual column count must be at least the derived minimum
/// (a larger, more conservative count is accepted), and the actual column and
/// comparison-point errors must meet the targets with the plan's work bits.
pub fn shared_lin_plan_fault(fault: SharedLinPlanFault, code: CodeParams) -> Result<()> {
    if fault.v == 0 || fault.rows_log == 0 || fault.column_draws == 0 {
        return Err(Error::InvalidConfig);
    }
    let at = table_proximity(code, fault.m)?;
    let shape = SharedLinShape {
        v: fault.v,
        k_lo: fault.k_lo,
        instances: 1,
        rows_log: fault.rows_log,
    };
    let bounds = shared_lin_bounds(
        shape,
        &at.agree,
        &at.list,
        &mca_error(&at),
        fault.point_work_bits,
        fault.column_work_bits,
    )?;
    if fault.column_draws < bounds.columns {
        return Err(Error::InvalidConfig);
    }
    let column_error = at.agree.pow(fault.column_draws);
    if !surely_le(&column_error, &goal(fault.column_work_bits)) {
        return Err(Error::InvalidConfig);
    }
    let zeta = field_error(u64::from(fault.rows_log));
    if !surely_le(&zeta, &goal(fault.zeta_work_bits)) {
        return Err(Error::InvalidConfig);
    }
    Ok(())
}

/// The largest table distance whose fold point and (optionally) record sample
/// meet the same target. The one-time rule checks the record sample as well;
/// the later rule does not repeat that check. The point bound always uses the
/// caller's sharing count and the same security target of the plan's mode.
fn largest_table_m(
    code: CodeParams,
    v: u32,
    answers: u32,
    point_variables: u32,
    sharing: u64,
    work_bits: u32,
    with_record: bool,
) -> Result<u64> {
    if v == 0 || answers == 0 || sharing == 0 || point_variables == 0 {
        return Err(Error::InvalidConfig);
    }
    let word = Word {
        log_n: code.log_n,
        ell: code.ell,
        k: code.r,
    };
    let cube_bits = ceil_log2(code.ell);
    let good = |m: u64| -> bool {
        let Ok(at) = johnson(&word, m) else {
            return false;
        };
        if at.bad() {
            return false;
        }
        let point = Bound::from_u64(sharing)
            .mul(&field_error(u64::from(point_variables)))
            .add(&Bound::from_u64(2).mul(&table_fold_error(v, &mca_error(&at), &at.list)));
        let point_ok = surely_le(&Bound::from_real(point.error()), &goal(work_bits));
        if !with_record {
            return point_ok;
        }
        let sample = sample_error(&at.list, cube_bits, answers);
        let sample_ok = surely_le(&Bound::from_real(sample.error()), &goal(0));
        point_ok && sample_ok
    };
    if !good(3) {
        return Err(Error::InvalidConfig);
    }
    let mut low = 3u64;
    let mut high = 6u64;
    while high <= MAX_M && good(high) {
        low = high;
        high = high.saturating_mul(2);
    }
    if high > MAX_M {
        if good(MAX_M) {
            return Ok(MAX_M);
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
    Ok(low)
}

/// The largest one-time table distance whose fold point and record sample meet
/// the target, with the given sharing and point variables.
pub fn recorded_table_distance(
    code: CodeParams,
    v: u32,
    answers: u32,
    point_variables: u32,
    sharing: u64,
    work_bits: u32,
) -> Result<u64> {
    largest_table_m(
        code,
        v,
        answers,
        point_variables,
        sharing,
        work_bits,
        true,
    )
}

/// The largest later table distance at the same point goal with `requests`
/// sharing and no repeated record check. The caller caps it at the one-time
/// distance; the later test is only defined for a table that already passed
/// the one-time test at the same geometry.
pub fn later_table_distance(
    code: CodeParams,
    v: u32,
    answers: u32,
    point_variables: u32,
    requests: u64,
    work_bits: u32,
) -> Result<u64> {
    if requests == 0 {
        return Err(Error::InvalidConfig);
    }
    largest_table_m(
        code,
        v,
        answers,
        point_variables,
        requests,
        work_bits,
        false,
    )
}

/// The derived later-parameter candidate of one declared table: the later
/// distance capped at the one-time one, the later column count, the batch
/// capacity of the random tail and the declared one-batch total. The tail is
/// shared with the one-time columns already spent; the capacity is not a
/// licence to run more batches than the declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaterTableCandidate {
    pub m_once: u64,
    pub m_later: u64,
    pub one_time_columns: u64,
    pub later_columns: u64,
    /// The one-time columns already spent from the table's random tail.
    pub spent: u64,
    /// floor((tail - spent) / later_columns); not the declared batch count.
    pub capacity_batches: u64,
    /// spent + one declared later batch's columns.
    pub total_columns_one_batch: u64,
}

/// Derives the later-parameter candidate from the same public geometry and
/// security target as the one-time rule. The later distance is the later rule
/// capped at `m_once`; the later columns are the smallest count meeting the
/// column target at the later distance; the spent columns and the declared one
/// batch must fit the table's random tail.
#[allow(clippy::too_many_arguments)]
pub fn derive_later_table_candidate(
    shape: SharedLinShape,
    code: CodeParams,
    answers: u32,
    point_variables: u32,
    m_once: u64,
    one_time_columns: u64,
    table_tail: u64,
    work_bits: u32,
) -> Result<LaterTableCandidate> {
    if m_once < 3
        || one_time_columns == 0
        || table_tail == 0
        || shape.instances == 0
        || shape.v == 0
        || shape.rows_log == 0
    {
        return Err(Error::InvalidConfig);
    }
    let later_raw = later_table_distance(
        code,
        shape.v,
        answers,
        point_variables,
        shape.instances,
        work_bits,
    )?;
    let m_later = m_once.min(later_raw);
    let later_columns = shared_lin_columns(shape, code, m_later, work_bits, work_bits)?;
    if later_columns == 0 {
        return Err(Error::InvalidConfig);
    }
    let spent = one_time_columns;
    let remaining = table_tail.saturating_sub(spent);
    let capacity_batches = remaining / later_columns;
    let total_columns_one_batch = spent
        .checked_add(later_columns)
        .ok_or(Error::ResourceLimit)?;
    if total_columns_one_batch > table_tail {
        return Err(Error::ResourceLimit);
    }
    Ok(LaterTableCandidate {
        m_once,
        m_later,
        one_time_columns,
        later_columns,
        spent,
        capacity_batches,
        total_columns_one_batch,
    })
}

/// The production later-parameter derivation. Every input is fixed public
/// geometry; the once distance and columns are derived here instead of being
/// supplied, the table tail is the code's own random tail, the request count
/// is one, and the work bits are zero. The derived later stage must also pass
/// the ordinary plan fault over its point, columns and comparison point.
pub fn derive_production_later_candidate(
    code: CodeParams,
    v: u32,
    answers: u32,
    point_variables: u32,
    rows_log: u32,
    k_lo: u32,
) -> Result<LaterTableCandidate> {
    let shape = SharedLinShape {
        v,
        k_lo,
        instances: 1,
        rows_log,
    };
    let m_once = recorded_table_distance(code, v, answers, point_variables, 1, 0)?;
    let one_time_columns = shared_lin_columns(shape, code, m_once, 0, 0)?;
    let candidate = derive_later_table_candidate(
        shape,
        code,
        answers,
        point_variables,
        m_once,
        one_time_columns,
        code.r,
        0,
    )?;
    shared_lin_plan_fault(
        SharedLinPlanFault {
            v,
            k_lo,
            rows_log,
            m: candidate.m_later,
            column_draws: candidate.later_columns,
            point_work_bits: 0,
            column_work_bits: 0,
            zeta_work_bits: 0,
        },
        code,
    )?;
    Ok(candidate)
}

fn stage_cube_bits(group: &GroupShape) -> u32 {
    let main = group.k + ceil_log2(group.main.ell);
    let mask = group.h + ceil_log2(group.mask.ell);
    main.max(mask) + 1
}

/// The row bits of one group of oracle members. Each member occupies an
/// aligned power-of-two block of main rows and of mask rows; the group has
/// `2^k` main rows and `2^h` mask rows, with `k >= h` and `k >= 1`. The
/// members are the oracle messages that actually join the group, in any
/// order: the sums only depend on the aligned block sizes.
pub fn lin_group_bits(main_rows: &[u64], mask_rows: &[u64]) -> Result<(u32, u32)> {
    use zkie_engine::pcs::request::plan::{ceil_log2 as plan_ceil_log2, pow2_at_least};
    if main_rows.is_empty() || main_rows.len() != mask_rows.len() {
        return Err(Error::InvalidConfig);
    }
    let mut main = 0u64;
    let mut mask = 0u64;
    for (rows, mask_rows) in main_rows.iter().zip(mask_rows) {
        if *rows > 0 {
            main = main
                .checked_add(pow2_at_least(*rows)?)
                .ok_or(Error::ResourceLimit)?;
        }
        if *mask_rows > 0 {
            mask = mask
                .checked_add(pow2_at_least(*mask_rows)?)
                .ok_or(Error::ResourceLimit)?;
        }
    }
    let h = if mask > 0 { plan_ceil_log2(mask)? } else { 0 };
    let main_bits = if main > 0 { plan_ceil_log2(main)? } else { 0 };
    Ok((main_bits.max(h).max(1), h))
}

/// The random tail for `t` opened positions. A circle domain needs an odd
/// tail, so an even count is raised by one; other domains use the count
/// itself. The flag is plan data, never a proof value.
pub fn stage_tail(t: u64, circle: bool) -> u64 {
    if circle && t != 0 && t % 2 == 0 {
        t + 1
    } else {
        t
    }
}

/// The sizes of one middle stage of a proximity-test stage list, derived from
/// the stage before it by the trusted rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageSizes {
    pub k: u32,
    pub h: u32,
    pub main_ell: u64,
    pub mask_ell: u64,
    /// The out-of-domain entries a mask row carries, the pad included.
    pub r_ood: u64,
    /// The row's random tail at `positions` opened positions.
    pub tail: u64,
}

fn pow2_at_least_u64(value: u64) -> Result<u64> {
    if value == 0 {
        return Ok(1);
    }
    let bits = 64 - (value - 1).leading_zeros();
    if bits >= 64 {
        return Err(Error::ResourceLimit);
    }
    Ok(1u64 << bits)
}

/// One middle stage: the folded row is cut into `2^k` rows, the mask capacity
/// carries the previous stage's two tails and slots plus its own two tails,
/// the mask length is the power of two that holds the carried entries, the
/// samples and the pad, and the out-of-domain entries are what remains.
#[allow(clippy::too_many_arguments)]
pub fn solve_next_stage(
    prev_main_ell: u64,
    prev_mask_ell: u64,
    prev_slots: u64,
    prev_main_r: u64,
    prev_mask_r: u64,
    k: u32,
    h: u32,
    samples: u32,
    positions: u64,
    circle: bool,
) -> Result<StageSizes> {
    if k == 0 || k > 32 || h > k {
        return Err(Error::InvalidConfig);
    }
    let divisor = 1u64.checked_shl(k).ok_or(Error::ResourceLimit)?;
    let scaled = prev_main_ell
        .checked_mul(prev_slots)
        .ok_or(Error::ResourceLimit)?;
    if scaled % divisor != 0 {
        return Err(Error::InvalidConfig);
    }
    let main_ell = scaled / divisor;
    let carried = 2u64
        .checked_mul(prev_mask_ell)
        .and_then(|value| value.checked_add(prev_slots.checked_mul(prev_main_r)?))
        .and_then(|value| value.checked_add(2u64.checked_mul(prev_mask_r)?))
        .ok_or(Error::ResourceLimit)?;
    let own_fold = 2u64
        .checked_mul(k as u64)
        .and_then(|value| value.checked_add(1))
        .and_then(|value| value.checked_sub(h as u64))
        .ok_or(Error::ResourceLimit)?;
    let per_row = 1u64.checked_shl(h).ok_or(Error::ResourceLimit)?;
    let need = carried
        .checked_add(samples as u64)
        .and_then(|value| value.checked_add(per_row - 1))
        .ok_or(Error::ResourceLimit)?;
    let mask_ell = pow2_at_least_u64(need >> h)?.max(pow2_at_least_u64(own_fold)?);
    let capacity = mask_ell.checked_shl(h).ok_or(Error::ResourceLimit)?;
    if capacity < carried {
        return Err(Error::ShapeMismatch);
    }
    Ok(StageSizes {
        k,
        h,
        main_ell,
        mask_ell,
        r_ood: capacity - carried,
        tail: stage_tail(positions, circle),
    })
}

/// The last stage of a stage list: one row, no folding (`k = h = 0`). Its
/// mask length is the power of two that holds the carried entries and the
/// samples; the out-of-domain entries are what remains.
pub fn solve_last_stage(
    prev_main_ell: u64,
    prev_mask_ell: u64,
    prev_slots: u64,
    prev_main_r: u64,
    prev_mask_r: u64,
    samples: u32,
    positions: u64,
    circle: bool,
) -> Result<StageSizes> {
    let main_ell = prev_main_ell
        .checked_mul(prev_slots)
        .ok_or(Error::ResourceLimit)?;
    let carried = 2u64
        .checked_mul(prev_mask_ell)
        .and_then(|value| value.checked_add(prev_slots.checked_mul(prev_main_r)?))
        .and_then(|value| value.checked_add(2u64.checked_mul(prev_mask_r)?))
        .ok_or(Error::ResourceLimit)?;
    let mask_ell = pow2_at_least_u64(
        carried
            .checked_add(samples as u64)
            .ok_or(Error::ResourceLimit)?,
    )?;
    if mask_ell < carried {
        return Err(Error::ShapeMismatch);
    }
    Ok(StageSizes {
        k: 0,
        h: 0,
        main_ell,
        mask_ell,
        r_ood: mask_ell - carried,
        tail: stage_tail(positions, circle),
    })
}

/// One public choice of a middle or last stage: the two code lengths and the
/// fold bits. It is plan data; no proof value enters it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageChoice {
    pub log_n: u32,
    pub mask_log_n: u32,
    pub k: u32,
    pub h: u32,
}

/// The first stage of a chain, as chosen by the public configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FirstStage {
    pub k: u32,
    pub h: u32,
    pub main_ell: u64,
    pub mask_ell: u64,
    pub slots: u64,
    pub main_r: u64,
    pub mask_r: u64,
}

/// One derived stage of a chain, with its code lengths and capacities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DerivedStage {
    pub log_n: u32,
    pub mask_log_n: u32,
    pub k: u32,
    pub h: u32,
    pub main_ell: u64,
    pub mask_ell: u64,
    pub r_ood: u64,
    pub tail: u64,
}

/// Why one public configuration is not a valid chain. The stage index is the
/// position in the choice list plus one (the first stage is zero), and the
/// point names the rule that failed: the stage shape rule, the divisibility
/// of the folded row, or the code's capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageFailure {
    pub stage: usize,
    pub point: &'static str,
}

fn code_fits(log_n: u32, ell: u64, tail: u64) -> bool {
    if log_n > 32 {
        return false;
    }
    let n = 1u64 << log_n;
    ell.checked_add(tail).map_or(false, |used| used <= n)
}

/// Derives the complete stage list of one chain from the public
/// configuration: the first stage and the per-stage choices. The last choice
/// must be the single-row last stage (k = h = 0), every other choice is a
/// middle stage with 1 <= k and h <= k. Each stage's capacities and random
/// tail follow the stage rules; the code lengths must fit the message and the
/// tail. A configuration the rules reject is reported with its stage and the
/// failing point instead of a generic error.
pub fn derive_stage_list(
    first: FirstStage,
    samples: u32,
    positions: u64,
    circle: bool,
    choices: &[StageChoice],
) -> std::result::Result<Vec<DerivedStage>, StageFailure> {
    if choices.is_empty() {
        return Err(StageFailure {
            stage: 0,
            point: "a chain folds at its first stage and has a last stage",
        });
    }
    let first_tail = stage_tail(positions, circle);
    if !code_fits(choices[0].log_n, first.main_ell, first_tail)
        || !code_fits(choices[0].mask_log_n, first.mask_ell, first_tail)
    {
        return Err(StageFailure {
            stage: 0,
            point: "the first stage's message and tail do not fit its code length",
        });
    }
    let mut stages = Vec::with_capacity(choices.len());
    stages.push(DerivedStage {
        log_n: choices[0].log_n,
        mask_log_n: choices[0].mask_log_n,
        k: first.k,
        h: first.h,
        main_ell: first.main_ell,
        mask_ell: first.mask_ell,
        r_ood: 0,
        tail: first_tail,
    });
    let mut prev = first;
    let mut prev_tail = first_tail;
    for (index, choice) in choices.iter().enumerate() {
        if index == 0 {
            continue;
        }
        let last = index + 1 == choices.len();
        if choice.k > 32 || choice.h > choice.k || (last && (choice.k != 0 || choice.h != 0)) || (!last && choice.k == 0) {
            return Err(StageFailure {
                stage: index,
                point: if last {
                    "the last stage has k = h = 0"
                } else {
                    "a middle stage has 1 <= k and h <= k"
                },
            });
        }
        let sizes = if last {
            solve_last_stage(
                prev.main_ell,
                prev.mask_ell,
                prev.slots,
                prev_tail,
                prev_tail,
                samples,
                positions,
                circle,
            )
        } else {
            solve_next_stage(
                prev.main_ell,
                prev.mask_ell,
                prev.slots,
                prev_tail,
                prev_tail,
                choice.k,
                choice.h,
                samples,
                positions,
                circle,
            )
        };
        let sizes = match sizes {
            Ok(sizes) => sizes,
            Err(Error::InvalidConfig) => {
                return Err(StageFailure {
                    stage: index,
                    point: "the folded row cannot be cut into the chosen rows",
                })
            }
            Err(_) => {
                return Err(StageFailure {
                    stage: index,
                    point: "the stage's capacities overflow",
                })
            }
        };
        if !code_fits(choice.log_n, sizes.main_ell, sizes.tail)
            || !code_fits(choice.mask_log_n, sizes.mask_ell, sizes.tail)
        {
            return Err(StageFailure {
                stage: index,
                point: "the stage's message and tail do not fit its code length",
            });
        }
        prev = FirstStage {
            k: sizes.k,
            h: sizes.h,
            main_ell: sizes.main_ell,
            mask_ell: sizes.mask_ell,
            slots: 1,
            main_r: sizes.tail,
            mask_r: sizes.tail,
        };
        prev_tail = sizes.tail;
        stages.push(DerivedStage {
            log_n: choice.log_n,
            mask_log_n: choice.mask_log_n,
            k: sizes.k,
            h: sizes.h,
            main_ell: sizes.main_ell,
            mask_ell: sizes.mask_ell,
            r_ood: sizes.r_ood,
            tail: sizes.tail,
        });
    }
    Ok(stages)
}

/// The main rows of a `W`-like message of `entries` entries under a code of
/// message length `ell` with `answers` sample pads: the smallest power of two
/// that leaves the last row's pads free. This is the one row rule of every
/// column message, so the shape constructor and the planner read it here.
pub fn rows_for_columns(entries: u64, ell: u64, answers: u32) -> Result<u64> {
    if entries == 0 || answers == 0 || answers as u64 > ell {
        return Err(Error::InvalidConfig);
    }
    let capacity = ell - answers as u64;
    if capacity == 0 {
        return Err(Error::InvalidConfig);
    }
    let rows_needed = entries
        .checked_add(capacity - 1)
        .ok_or(Error::ResourceLimit)?
        / capacity;
    let main_rows = rows_needed
        .checked_next_power_of_two()
        .ok_or(Error::ResourceLimit)?;
    let last_place = (entries - 1) / main_rows;
    if last_place >= capacity {
        return Err(Error::InvalidConfig);
    }
    Ok(main_rows)
}

/// One solved stage of a chain: the public choices with the derived message
/// lengths, out-of-domain entries, position count and random tail. The
/// position count and the tail follow from the stage rules; neither is a
/// free parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SolvedStage {
    pub log_n: u32,
    pub mask_log_n: u32,
    pub k: u32,
    pub h: u32,
    pub main_ell: u64,
    pub mask_ell: u64,
    pub r_ood: u64,
    pub positions: u64,
    /// The main code's random tail of this stage. The first stage's tail is
    /// the code's joint tail; a later stage's tail is the solved position
    /// count's tail, the same for both codes.
    pub main_r: u64,
    pub mask_r: u64,
}

/// The first stage of a chain as the trusted plan fixes it: the message
/// lengths of both codes, the fold bits derived from the real member layout,
/// and the number of positions the joint solve settled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainStart {
    pub main_ell: u64,
    pub mask_ell: u64,
    pub k: u32,
    pub h: u32,
    pub positions: u64,
    pub main_r: u64,
    pub mask_r: u64,
}

/// How many positions one middle or last stage is opened at, and its random
/// tail. The iteration mirrors the reference stage rule: start from zero, size
/// the group with the tail that the current count implies, and raise the count
/// until the round error at the target stops moving it. A group that cannot
/// hold its rounds at the target, or a code that cannot hold message and tail,
/// is reported at its stage.
fn solve_stage_positions(
    log_n: u32,
    mask_log_n: u32,
    sizes: StageSizes,
    previous_main_r: u64,
    circle: bool,
) -> std::result::Result<(u64, u64), StageFailure> {
    let mut t = 0u64;
    for turn in 0..=32 {
        let tail = stage_tail(t, circle);
        let main = CodeParams {
            log_n,
            ell: sizes.main_ell,
            r: tail,
        };
        let mask = CodeParams {
            log_n: mask_log_n,
            ell: sizes.mask_ell,
            r: tail,
        };
        let shape = match group_shape(main, mask, sizes.k, sizes.h) {
            Ok(shape) => shape,
            Err(_) => {
                return Err(StageFailure {
                    stage: 0,
                    point: "the stage's message and random tail do not fit its code length",
                })
            }
        };
        let load = GroupLoad {
            cube_bits: stage_cube_bits(&shape),
            messages: 1,
            batched_rows: 4 + 3 * previous_main_r.max(1024),
        };
        let m = largest_m(&shape, load);
        let serves = m > 0 && holds_for_m(&shape, load, m);
        let extra = margin_bound();
        let next = if serves {
            match analysis_of_m(&shape, m) {
                Ok(analysis) => scope_positions(&analysis.at.agree, 0, &extra),
                Err(_) => 0,
            }
        } else {
            positions_from_below(&shape, load, &extra)
        };
        if !serves && next <= t {
            return Err(StageFailure {
                stage: 0,
                point: "no distance under its rule holds the stage's rounds at the target",
            });
        }
        if next == 0 {
            return Err(StageFailure {
                stage: 0,
                point: "no number of positions reaches the target",
            });
        }
        if next <= t {
            return Ok((t, stage_tail(t, circle)));
        }
        if turn == 32 {
            return Err(StageFailure {
                stage: 0,
                point: "positions and random coordinates do not settle",
            });
        }
        t = next;
    }
    Err(StageFailure {
        stage: 0,
        point: "positions and random coordinates do not settle",
    })
}

/// One later stage's sizes from the choice and the stage before it. The error
/// is reported at `stage` with the rule that failed.
#[allow(clippy::too_many_arguments)]
fn sized_choice(
    prev: &FirstStage,
    previous_positions: u64,
    k: u32,
    h: u32,
    last: bool,
    samples: u32,
    circle: bool,
    stage: usize,
) -> std::result::Result<StageSizes, StageFailure> {
    let sized = if last {
        solve_last_stage(
            prev.main_ell,
            prev.mask_ell,
            prev.slots,
            prev.main_r,
            prev.mask_r,
            samples,
            previous_positions,
            circle,
        )
    } else {
        solve_next_stage(
            prev.main_ell,
            prev.mask_ell,
            prev.slots,
            prev.main_r,
            prev.mask_r,
            k,
            h,
            samples,
            previous_positions,
            circle,
        )
    };
    sized.map_err(|error| match error {
        Error::InvalidConfig => StageFailure {
            stage,
            point: "the folded row cannot be cut into the chosen rows",
        },
        _ => StageFailure {
            stage,
            point: "the stage's capacities overflow",
        },
    })
}

/// Derives a complete chain and solves each stage's position count. The first
/// choice carries the first stage's two code lengths; the first stage's fold
/// bits, message lengths and position count come from `start`. Every later
/// choice is one middle stage (`1 <= k`, `h <= k`) or, as the last choice, the
/// single-row stage `k = h = 0`. Each later stage's message lengths, its
/// out-of-domain entries and its position count follow from the stage rules.
pub fn solve_chain_stages(
    start: ChainStart,
    samples: u32,
    circle: bool,
    choices: &[StageChoice],
) -> std::result::Result<Vec<SolvedStage>, StageFailure> {
    if choices.len() < 2 || start.positions == 0 || start.k == 0 {
        return Err(StageFailure {
            stage: 0,
            point: "a chain folds at its first stage and has a last stage",
        });
    }
    let first = FirstStage {
        k: start.k,
        h: start.h,
        main_ell: start.main_ell,
        mask_ell: start.mask_ell,
        slots: 1,
        main_r: start.main_r,
        mask_r: start.mask_r,
    };
    if !code_fits(choices[0].log_n, first.main_ell, first.main_r)
        || !code_fits(choices[0].mask_log_n, first.mask_ell, first.mask_r)
    {
        return Err(StageFailure {
            stage: 0,
            point: "the first stage's message and tail do not fit its code length",
        });
    }
    let mut stages = Vec::with_capacity(choices.len());
    stages.push(SolvedStage {
        log_n: choices[0].log_n,
        mask_log_n: choices[0].mask_log_n,
        k: start.k,
        h: start.h,
        main_ell: start.main_ell,
        mask_ell: start.mask_ell,
        r_ood: 0,
        positions: start.positions,
        main_r: start.main_r,
        mask_r: start.mask_r,
    });
    let mut prev = first;
    for (index, choice) in choices.iter().enumerate() {
        if index == 0 {
            continue;
        }
        let last = index + 1 == choices.len();
        if choice.k > 8
            || choice.h > choice.k
            || (last && (choice.k != 0 || choice.h != 0))
            || (!last && choice.k == 0)
        {
            return Err(StageFailure {
                stage: index,
                point: if last {
                    "the last stage has k = h = 0"
                } else {
                    "a middle stage has 1 <= k and h <= k"
                },
            });
        }
        let previous_positions = stages[index - 1].positions;
        let sizes = sized_choice(&prev, previous_positions, choice.k, choice.h, last, samples, circle, index)?;
        let (positions, tail) = solve_stage_positions(
            choice.log_n,
            choice.mask_log_n,
            sizes,
            prev.main_r,
            circle,
        )
        .map_err(|failure| StageFailure {
            stage: index,
            point: failure.point,
        })?;
        prev = FirstStage {
            k: sizes.k,
            h: sizes.h,
            main_ell: sizes.main_ell,
            mask_ell: sizes.mask_ell,
            slots: 1,
            main_r: tail,
            mask_r: tail,
        };
        stages.push(SolvedStage {
            log_n: choice.log_n,
            mask_log_n: choice.mask_log_n,
            k: sizes.k,
            h: sizes.h,
            main_ell: sizes.main_ell,
            mask_ell: sizes.mask_ell,
            r_ood: sizes.r_ood,
            positions,
            main_r: tail,
            mask_r: tail,
        });
    }
    Ok(stages)
}


/// One member of a group as the assembly sizes it: a message with fixed rows,
/// or a column message whose rows follow from the first-stage message length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssemblyMember {
    Fixed { main_rows: u64, mask_rows: u64 },
    Columns { entries: u64, answers: u32 },
}

/// One object of the own group with what it has already spent before the
/// stage-0 list: the SEND readings on its own tree, and whether the shared
/// test opens this same physical object as well (the joined `Omega`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssemblyOwnObject {
    pub member: AssemblyMember,
    pub shared_too: bool,
    pub spent: u64,
}

/// The trusted inputs of one assembly: the real member layouts of both
/// groups, the fixed request count, sample count and table width, and the
/// circle flag. Every number here is plan data; no proof byte enters it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssemblyInput {
    pub shared_members: Vec<AssemblyMember>,
    pub own_objects: Vec<AssemblyOwnObject>,
    /// What the shared fresh mask has already spent before its list.
    pub shared_mask_spent: u64,
    pub requests: u64,
    pub samples: u32,
    pub columns: u64,
    pub circle: bool,
}

/// One fully validated assembled proximity plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssembledProximity {
    pub first_main: CodeParams,
    pub first_mask: CodeParams,
    pub shared: Vec<SolvedStage>,
    pub own: Vec<SolvedStage>,
    /// The smallest remaining openable capacity of the own group's stage-0
    /// objects; the own stage-0 list fits it by construction.
    pub max_zero: u64,
    /// The exact round count of the frozen complete plan, formed before any
    /// run. It is an output of the plan, never a search variable.
    pub reserved_rounds: u64,
    pub choices: AssemblyCandidate,
}

fn assembly_rows(member: AssemblyMember, ell: u64) -> std::result::Result<(u64, u64), StageFailure> {
    match member {
        AssemblyMember::Fixed {
            main_rows,
            mask_rows,
        } => Ok((main_rows, mask_rows)),
        AssemblyMember::Columns { entries, answers } => rows_for_columns(entries, ell, answers)
            .map(|rows| (rows, 0))
            .map_err(|_| StageFailure {
                stage: 0,
                point: "a column message does not fit the first-stage code length",
            }),
    }
}

fn derived_stage(stage: &SolvedStage) -> DerivedStage {
    DerivedStage {
        log_n: stage.log_n,
        mask_log_n: stage.mask_log_n,
        k: stage.k,
        h: stage.h,
        main_ell: stage.main_ell,
        mask_ell: stage.mask_ell,
        r_ood: stage.r_ood,
        tail: stage.main_r.max(stage.mask_r),
    }
}

/// Checks one solved chain against the stage-list rules a run enforces. The
/// first stage's codes and fold bits were checked against the real layout by
/// the caller; here every later stage's codes, fold bits, capacities and
/// position counts are checked again on the solved values.
fn validate_solved_chain(stages: &[SolvedStage], samples: u32) -> std::result::Result<(), StageFailure> {
    if stages.len() < 2 {
        return Err(StageFailure {
            stage: 0,
            point: "a proximity test has at least two stages",
        });
    }
    for (index, stage) in stages.iter().enumerate() {
        use zkie_engine::pcs::request::code::build_code_plan;
        if build_code_plan(stage.log_n, stage.main_ell, stage.main_r).is_err()
            || build_code_plan(stage.mask_log_n, stage.mask_ell, stage.mask_r).is_err()
        {
            return Err(StageFailure {
                stage: index,
                point: "a stage code is not a code",
            });
        }
        if stage.mask_log_n > stage.log_n {
            return Err(StageFailure {
                stage: index,
                point: "a stage mask code is longer than its main code",
            });
        }
        if stage.positions == 0 || stage.positions > stage.main_r.min(stage.mask_r) {
            return Err(StageFailure {
                stage: index,
                point: "a stage opens more positions than its tail hides",
            });
        }
        let last = index + 1 == stages.len();
        if last {
            if stage.k != 0 || stage.h != 0 {
                return Err(StageFailure {
                    stage: index,
                    point: "the last stage has k = h = 0",
                });
            }
        } else {
            if stage.k < 1 || stage.h > stage.k {
                return Err(StageFailure {
                    stage: index,
                    point: "a middle stage has 1 <= k and h <= k",
                });
            }
            if stage.mask_ell < 2 * stage.k as u64 + 1 - stage.h as u64 {
                return Err(StageFailure {
                    stage: index,
                    point: "a stage mask message is too short for the fold hiding polynomial",
                });
            }
        }
        if index > 0 {
            let previous = &stages[index - 1];
            match stage.main_ell.checked_shl(stage.k) {
                Some(folded) if folded == previous.main_ell => {}
                _ => {
                    return Err(StageFailure {
                        stage: index,
                        point: "the folded main message does not fill the next rows",
                    })
                }
            }
            let carried = 2u64
                .checked_mul(previous.mask_ell)
                .and_then(|value| value.checked_add(previous.main_r))
                .and_then(|value| value.checked_add(2u64.checked_mul(previous.mask_r)?))
                .ok_or(StageFailure {
                    stage: index,
                    point: "the stage's capacities overflow",
                })?;
            match stage.mask_ell.checked_shl(stage.h) {
                Some(room) if room >= carried + samples as u64 => {}
                _ => {
                    return Err(StageFailure {
                        stage: index,
                        point: "the stage's mask rows have no room for the fold masks and randomness",
                    })
                }
            }
        }
    }
    Ok(())
}

/// One fully public configuration candidate: the first stage's two code
/// lengths and each group's post-first stages, in the reference's public
/// form `(k, h, log_n, mask_log_n)`. The first stage's fold bits are not here:
/// they come from the real member layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssemblyCandidate {
    pub first_log_n: u32,
    pub first_log_ell: u32,
    pub first_mask_log_n: u32,
    pub first_mask_log_ell: u32,
    pub shared_choices: Vec<StageChoice>,
    pub own_choices: Vec<StageChoice>,
}

/// The reference production policy's group lines, expressed as candidates for
/// the one-main-code layout. The folded line names another main code than the
/// own and Omega lines, a split this executor does not support; the folded
/// group therefore uses the own/Omega main code and the fold bits of its own
/// line. The mask code is one for both groups, so the own line's mask code is
/// tried first and the folded line's split mask code second.
pub fn production_candidates() -> Vec<AssemblyCandidate> {
    let shared_choices = vec![
        StageChoice { log_n: 14, mask_log_n: 12, k: 4, h: 2 },
        StageChoice { log_n: 14, mask_log_n: 14, k: 0, h: 0 },
    ];
    let own_choices = vec![
        StageChoice { log_n: 14, mask_log_n: 13, k: 4, h: 3 },
        StageChoice { log_n: 15, mask_log_n: 15, k: 0, h: 0 },
    ];
    vec![
        AssemblyCandidate {
            first_log_n: 18,
            first_log_ell: 14,
            first_mask_log_n: 18,
            first_mask_log_ell: 8,
            shared_choices: shared_choices.clone(),
            own_choices: own_choices.clone(),
        },
        AssemblyCandidate {
            first_log_n: 18,
            first_log_ell: 14,
            first_mask_log_n: 18,
            first_mask_log_ell: 6,
            shared_choices,
            own_choices,
        },
    ]
}

/// Why one configuration was not assembled: a named failing point, or a
/// search that could not be completed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssemblyError {
    Reject(StageFailure),
    Exhausted,
}

impl From<StageFailure> for AssemblyError {
    fn from(failure: StageFailure) -> Self {
        AssemblyError::Reject(failure)
    }
}

/// Validates one public configuration end to end. Every value the plan needs
/// is derived here from the real member layout and the public choices; a
/// configuration that violates any rule is reported with its stage and the
/// failing point. The prover's assembly and the verifier's independent check
/// both call this function on the same trusted inputs; the verifier does not
/// search for another configuration.
pub fn validate_assembled(
    input: &AssemblyInput,
    candidate: &AssemblyCandidate,
) -> std::result::Result<AssembledProximity, AssemblyError> {
    if candidate.first_log_ell >= 32
        || candidate.first_mask_log_ell >= 32
        || candidate.first_log_ell >= candidate.first_log_n
        || candidate.first_mask_log_ell >= candidate.first_mask_log_n
        || candidate.first_mask_log_n > candidate.first_log_n
    {
        return Err(AssemblyError::Reject(StageFailure {
            stage: 0,
            point: "the first-stage code lengths are not a legal candidate",
        }));
    }
    let ell = 1u64 << candidate.first_log_ell;
    let mask_ell = 1u64 << candidate.first_mask_log_ell;
    let mut shared_main = Vec::with_capacity(input.shared_members.len());
    let mut shared_mask = Vec::with_capacity(input.shared_members.len());
    for member in &input.shared_members {
        let (main_rows, mask_rows) = assembly_rows(*member, ell)?;
        shared_main.push(main_rows);
        shared_mask.push(mask_rows);
    }
    let mut own_main = Vec::with_capacity(input.own_objects.len());
    let mut own_mask = Vec::with_capacity(input.own_objects.len());
    for object in &input.own_objects {
        let (main_rows, mask_rows) = assembly_rows(object.member, ell)?;
        own_main.push(main_rows);
        own_mask.push(mask_rows);
    }
    let (shared_k, shared_h) = lin_group_bits(&shared_main, &shared_mask).map_err(|_| StageFailure {
        stage: 0,
        point: "the shared group's member rows do not form a group",
    })?;
    let (own_k, own_h) = lin_group_bits(&own_main, &own_mask).map_err(|_| StageFailure {
        stage: 0,
        point: "the own group's member rows do not form a group",
    })?;
    // A stage that is not the last one needs mask room for the fold hiding
    // polynomial; the first stage is never the last one, and its fold bits
    // come from the real member layout. A mask message below the bound is
    // rejected before any solve.
    let shared_fold_min = 2u64
        .checked_mul(shared_k as u64)
        .and_then(|value| value.checked_add(1))
        .and_then(|value| value.checked_sub(shared_h as u64))
        .ok_or(StageFailure { stage: 0, point: "the first stage's fold bits overflow" })?;
    let own_fold_min = 2u64
        .checked_mul(own_k as u64)
        .and_then(|value| value.checked_add(1))
        .and_then(|value| value.checked_sub(own_h as u64))
        .ok_or(StageFailure { stage: 0, point: "the first stage's fold bits overflow" })?;
    if mask_ell < shared_fold_min || mask_ell < own_fold_min {
        return Err(AssemblyError::Reject(StageFailure {
            stage: 0,
            point: "the first stage's mask message is too short for the fold hiding polynomial",
        }));
    }
    if ell > (1u64 << candidate.first_log_n) || mask_ell > (1u64 << candidate.first_mask_log_n) {
        return Err(AssemblyError::Reject(StageFailure {
            stage: 0,
            point: "the first stage's message does not fit its code length",
        }));
    }
    let main_code = CodeParams {
        log_n: candidate.first_log_n,
        ell,
        r: 0,
    };
    let mask_code = CodeParams {
        log_n: candidate.first_mask_log_n,
        ell: mask_ell,
        r: 0,
    };
    let mut main_spends = Vec::new();
    let mut mask_spends = Vec::new();
    for object in &input.own_objects {
        let (main_rows, mask_rows) = assembly_rows(object.member, ell)?;
        if main_rows > 0 {
            main_spends.push(object.spent);
        } else if mask_rows > 0 {
            mask_spends.push(object.spent);
        }
    }
    let folded = JointGroup {
        main: main_code,
        mask: mask_code,
        k: shared_k,
        h: shared_h,
    };
    let own = JointGroup {
        main: main_code,
        mask: mask_code,
        k: own_k,
        h: own_h,
    };
    let stage = solve_first_stage_one_mask(
        &folded,
        &folded,
        &own,
        input.requests,
        input.samples,
        input.columns,
        input.circle,
        &main_spends,
        &mask_spends,
    )?;
    let shared_positions = stage.folded_positions;
    let own_positions = stage.own_positions;
    let main_tail = stage.own_main_tail;
    let mask_tail = stage.folded_mask_tail;
    if shared_positions == 0 || own_positions == 0 {
        return Err(AssemblyError::Reject(StageFailure {
            stage: 0,
            point: "the first stage opens no position",
        }));
    }
    if !code_fits(candidate.first_log_n, ell, main_tail)
        || !code_fits(candidate.first_mask_log_n, mask_ell, mask_tail)
    {
        return Err(AssemblyError::Reject(StageFailure {
            stage: 0,
            point: "the first stage's message and random tail do not fit its code length",
        }));
    }
    let mut max_zero = u64::MAX;
    for (index, object) in input.own_objects.iter().enumerate() {
        let (main_rows, mask_rows) = assembly_rows(object.member, ell)?;
        let (tail, spent) = if main_rows > 0 {
            let shared_reads = if object.shared_too { shared_positions } else { 0 };
            let spent = object.spent.checked_add(shared_reads).ok_or(StageFailure {
                stage: 0,
                point: "an own object's spent positions overflow",
            })?;
            (main_tail, spent)
        } else if mask_rows > 0 {
            (mask_tail, object.spent)
        } else {
            continue;
        };
        let remaining = tail.checked_sub(spent).ok_or(StageFailure {
            stage: 0,
            point: "an own object has spent more positions than its tail hides",
        })?;
        if own_positions > remaining {
            return Err(AssemblyError::Reject(StageFailure {
                stage: 0,
                point: "the own stage-0 list exceeds an object's remaining capacity",
            }));
        }
        max_zero = max_zero.min(remaining);
        let _ = index;
    }
    if max_zero == u64::MAX {
        return Err(AssemblyError::Reject(StageFailure {
            stage: 0,
            point: "the own group has no stage-0 object",
        }));
    }
    for member in &input.shared_members {
        let (main_rows, mask_rows) = assembly_rows(*member, ell)?;
        let tail = if main_rows > 0 {
            main_tail
        } else if mask_rows > 0 {
            mask_tail
        } else {
            continue;
        };
        if shared_positions > tail {
            return Err(AssemblyError::Reject(StageFailure {
                stage: 0,
                point: "the shared stage-0 list exceeds a member's tail",
            }));
        }
    }
    if shared_positions > mask_tail.saturating_sub(input.shared_mask_spent) {
        return Err(AssemblyError::Reject(StageFailure {
            stage: 0,
            point: "the shared stage-0 list exceeds the fresh mask's remaining capacity",
        }));
    }
    let shared_start = ChainStart {
        main_ell: ell,
        mask_ell,
        k: shared_k,
        h: shared_h,
        positions: shared_positions,
        main_r: main_tail,
        mask_r: mask_tail,
    };
    let own_start = ChainStart {
        main_ell: ell,
        mask_ell,
        k: own_k,
        h: own_h,
        positions: own_positions,
        main_r: main_tail,
        mask_r: mask_tail,
    };
    let mut shared_choices = Vec::with_capacity(candidate.shared_choices.len() + 1);
    shared_choices.push(StageChoice {
        log_n: candidate.first_log_n,
        mask_log_n: candidate.first_mask_log_n,
        k: 0,
        h: 0,
    });
    shared_choices.extend(candidate.shared_choices.iter().copied());
    let shared = solve_chain_stages(shared_start, input.samples, input.circle, &shared_choices)?;
    let mut own_choices = Vec::with_capacity(candidate.own_choices.len() + 1);
    own_choices.push(StageChoice {
        log_n: candidate.first_log_n,
        mask_log_n: candidate.first_mask_log_n,
        k: 0,
        h: 0,
    });
    own_choices.extend(candidate.own_choices.iter().copied());
    let own_stages = solve_chain_stages(own_start, input.samples, input.circle, &own_choices)?;
    validate_solved_chain(&shared, input.samples)?;
    validate_solved_chain(&own_stages, input.samples)?;
    let shared_budget = chain_round_budget(
        &shared.iter().map(derived_stage).collect::<Vec<_>>(),
    )
    .map_err(|failure| StageFailure {
        stage: failure.stage,
        point: failure.point,
    })?;
    let own_budget = chain_round_budget(
        &own_stages.iter().map(derived_stage).collect::<Vec<_>>(),
    )
    .map_err(|failure| StageFailure {
        stage: failure.stage,
        point: failure.point,
    })?;
    let reserved_rounds = shared_budget
        .total
        .checked_add(own_budget.total)
        .ok_or(StageFailure {
            stage: 0,
            point: "the chain round count overflows",
        })?;
    Ok(AssembledProximity {
        first_main: CodeParams {
            log_n: candidate.first_log_n,
            ell,
            r: main_tail,
        },
        first_mask: CodeParams {
            log_n: candidate.first_mask_log_n,
            ell: mask_ell,
            r: mask_tail,
        },
        shared,
        own: own_stages,
        max_zero,
        reserved_rounds,
        choices: candidate.clone(),
    })
}

/// Assembles the production proximity plan from the trusted public
/// configuration. Every candidate is the reference production policy's group
/// lines, mapped to the one-main-code layout this executor supports; each is
/// validated end to end against the real member layout. The first candidate
/// that passes every check is the plan. When none passes, the error names each
/// candidate's failing stage and point: no configuration is invented to make
/// one fit.
pub fn assemble_proximity(
    input: &AssemblyInput,
) -> std::result::Result<AssembledProximity, String> {
    let candidates = production_candidates();
    if candidates.is_empty() {
        return Err("no candidate found within the searched domain (the candidate list is empty)".into());
    }
    let mut failures = Vec::with_capacity(candidates.len());
    for candidate in &candidates {
        match validate_assembled(input, candidate) {
            Ok(plan) => return Ok(plan),
            Err(AssemblyError::Reject(failure)) => {
                failures.push(format!("stage {}: {}", failure.stage, failure.point));
            }
            Err(AssemblyError::Exhausted) => {
                failures.push("the candidate could not be fully checked".into());
            }
        }
    }
    Err(format!(
        "no candidate found within the searched domain ({} candidates; failures: {})",
        candidates.len(),
        failures.join("; ")
    ))
}

/// The positions every code's random coordinates must cover when the shared
/// and own groups are solved together. `folded` and `omega` are the shared
/// group's position counts, `own` a request's, and `requests` the batch size.
/// Omega is read by the shared test and by every request's test; a split
/// shared group reads Omega's own message as well.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JointDraws {
    pub folded_main: u64,
    pub folded_mask: u64,
    pub omega_main: u64,
    pub omega_mask: u64,
    pub own_main: u64,
    pub own_mask: u64,
}

/// The joint draw counts of the shared and own groups. `requests` is the
/// public batch size of the request under test: a single request has
/// `requests = 1`, which is not the same as omitting the batch term.
pub fn joint_draws(
    folded: u64,
    omega: u64,
    own: u64,
    requests: u64,
    split: bool,
) -> Result<JointDraws> {
    if requests == 0 {
        return Err(Error::InvalidConfig);
    }
    let own_total = requests
        .checked_mul(own)
        .ok_or(Error::ResourceLimit)?;
    let omega_reads = if split { omega } else { folded };
    let omega_main = omega_reads
        .checked_add(own_total)
        .ok_or(Error::ResourceLimit)?;
    Ok(JointDraws {
        folded_main: if split { folded } else { omega_main },
        folded_mask: if split {
            folded.checked_add(omega).ok_or(Error::ResourceLimit)?
        } else {
            folded
        },
        omega_main,
        omega_mask: if split {
            folded.checked_add(omega).ok_or(Error::ResourceLimit)?
        } else {
            folded
        },
        own_main: omega_main,
        own_mask: own,
    })
}

fn with_tail(word: Word, tail: u64) -> Result<Word> {
    let dim = word.ell.checked_add(tail).ok_or(Error::ResourceLimit)?;
    let n = 1u64 << word.log_n;
    if dim == 0 || dim > n {
        return Err(Error::InvalidConfig);
    }
    Ok(Word {
        log_n: word.log_n,
        ell: word.ell,
        k: dim - 1,
    })
}

fn group_with_tails(shape: &GroupShape, main_tail: u64, mask_tail: u64) -> Result<GroupShape> {
    let mut group = shape.clone();
    group.main = with_tail(shape.main, main_tail)?;
    group.mask = with_tail(shape.mask, mask_tail)?;
    Ok(group)
}

fn margin_bound() -> Bound {
    Bound::from_real(Real::two_to(-((TARGET_BITS as i64) + 30)))
}

/// The main code's own position count on the way up: the mask rows are
/// cleared and the main code alone is sized. Zero when not even the main code
/// reaches the target under its rule.
fn positions_from_below(group: &GroupShape, load: GroupLoad, extra: &Bound) -> u64 {
    let mut main_only = group.clone();
    main_only.h = 0;
    main_only.mask_enabled = false;
    let m = largest_m(&main_only, load);
    if m == 0 || !holds_for_m(&main_only, load, m) {
        return 0;
    }
    match analysis_of_m(&main_only, m) {
        Ok(analysis) => scope_positions(&analysis.at.agree, 0, extra),
        Err(_) => 0,
    }
}

/// One group of the first-stage joint solve: the two codes and the fold bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JointGroup {
    pub main: CodeParams,
    pub mask: CodeParams,
    pub k: u32,
    pub h: u32,
}

/// The converged first-stage counts of the joint solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JointFirstStage {
    pub folded_positions: u64,
    pub omega_positions: u64,
    pub own_positions: u64,
    pub folded_main_tail: u64,
    pub folded_mask_tail: u64,
    pub omega_main_tail: u64,
    pub omega_mask_tail: u64,
    pub own_main_tail: u64,
    pub own_mask_tail: u64,
}

fn sized_step_tails(
    shape: &GroupShape,
    main_tail: u64,
    mask_tail: u64,
    rows: u64,
    extra: &Bound,
) -> std::result::Result<(u64, bool), &'static str> {
    let group = group_with_tails(shape, main_tail, mask_tail)
        .map_err(|_| "the group's message and random tail do not fit its code length")?;
    let load = GroupLoad {
        cube_bits: stage_cube_bits(&group),
        messages: 2,
        batched_rows: rows,
    };
    let m = largest_m(&group, load);
    if m > 0 && holds_for_m(&group, load, m) {
        let analysis = analysis_of_m(&group, m)
            .map_err(|_| "the group's codes have no admissible distance under its rule")?;
        return Ok((scope_positions(&analysis.at.agree, 0, extra), true));
    }
    let below = positions_from_below(&group, load, extra);
    if below == 0 {
        return Err(
            "no distance under its rule, or no number of positions, holds the group's rounds at the target",
        );
    }
    Ok((below, false))
}

fn sized_step(
    shape: &GroupShape,
    main_draws: u64,
    mask_draws: u64,
    rows: u64,
    circle: bool,
    extra: &Bound,
) -> std::result::Result<(u64, bool), &'static str> {
    sized_step_tails(
        shape,
        stage_tail(main_draws, circle),
        stage_tail(mask_draws, circle),
        rows,
        extra,
    )
}

/// Solves the shared and own groups together, upward from zero, on the joint
/// draw counts. Each round recomputes the random tails from the current
/// positions and the groups' sizes; the loop stops when no group's position
/// count grows. A settled solution must have every sized group holding its
/// rounds at the target; otherwise the configuration is reported with its
/// failing point.
#[allow(clippy::too_many_arguments)]
pub fn solve_first_stage(
    folded: &JointGroup,
    omega: &JointGroup,
    own: &JointGroup,
    requests: u64,
    samples: u32,
    columns: u64,
    circle: bool,
) -> std::result::Result<JointFirstStage, StageFailure> {
    let folded_shape = group_shape(folded.main, folded.mask, folded.k, folded.h)
        .map_err(|_| StageFailure { stage: 0, point: "the folded group's codes do not form a group" })?;
    let omega_shape = group_shape(omega.main, omega.mask, omega.k, omega.h)
        .map_err(|_| StageFailure { stage: 0, point: "the Omega group's codes do not form a group" })?;
    let own_shape = group_shape(own.main, own.mask, own.k, own.h)
        .map_err(|_| StageFailure { stage: 0, point: "the request group's codes do not form a group" })?;
    let split = folded_shape.main.log_n != omega_shape.main.log_n
        || folded_shape.main.ell != omega_shape.main.ell;
    let samples64 = samples as u64;
    let rows_folded = columns
        .checked_add(1)
        .and_then(|value| value.checked_add(3u64.checked_mul(samples64)?))
        .ok_or(StageFailure { stage: 0, point: "the folded group's batched rows overflow" })?;
    let rows_omega = 1u64
        .checked_add(2u64.checked_mul(samples64).ok_or(StageFailure { stage: 0, point: "the Omega group's batched rows overflow" })?)
        .ok_or(StageFailure { stage: 0, point: "the Omega group's batched rows overflow" })?;
    let rows_own = 2u64
        .checked_add(3u64.checked_mul(samples64).ok_or(StageFailure { stage: 0, point: "the request group's batched rows overflow" })?)
        .ok_or(StageFailure { stage: 0, point: "the request group's batched rows overflow" })?;
    let extra = margin_bound();
    let mut folded_t = 0u64;
    let mut omega_t = 0u64;
    let mut own_t = 0u64;
    let mut serves_folded = false;
    let mut serves_omega = false;
    let mut serves_own = false;
    for turn in 0..=32 {
        let draws = joint_draws(folded_t, omega_t, own_t, requests, split)
            .map_err(|_| StageFailure { stage: 0, point: "the joint draw counts overflow" })?;
        let folded_step = sized_step(
            &folded_shape,
            draws.folded_main,
            draws.folded_mask,
            rows_folded,
            circle,
            &extra,
        )
        .map_err(|point| StageFailure { stage: 0, point })?;
        let omega_step = if split {
            Some(
                sized_step(
                    &omega_shape,
                    draws.omega_main,
                    draws.omega_mask,
                    rows_omega,
                    circle,
                    &extra,
                )
                .map_err(|point| StageFailure { stage: 0, point })?,
            )
        } else {
            None
        };
        let own_step = sized_step(
            &own_shape,
            draws.own_main,
            draws.own_mask,
            rows_own,
            circle,
            &extra,
        )
        .map_err(|point| StageFailure { stage: 0, point })?;
        serves_folded = folded_step.1;
        serves_omega = omega_step.map_or(false, |step| step.1);
        serves_own = own_step.1;
        let next_folded = folded_t.max(folded_step.0);
        let next_omega = omega_t.max(omega_step.map_or(0, |step| step.0));
        let next_own = own_t.max(own_step.0);
        if next_folded == folded_t && next_omega == omega_t && next_own == own_t {
            if !serves_folded || (split && !serves_omega) || !serves_own {
                return Err(StageFailure {
                    stage: 0,
                    point: "no distance under its rule holds the group's rounds at the target",
                });
            }
            let draws = joint_draws(next_folded, next_omega, next_own, requests, split)
                .map_err(|_| StageFailure { stage: 0, point: "the joint draw counts overflow" })?;
            return Ok(JointFirstStage {
                folded_positions: next_folded,
                omega_positions: next_omega,
                own_positions: next_own,
                folded_main_tail: stage_tail(draws.folded_main, circle),
                folded_mask_tail: stage_tail(draws.folded_mask, circle),
                omega_main_tail: stage_tail(draws.omega_main, circle),
                omega_mask_tail: stage_tail(draws.omega_mask, circle),
                own_main_tail: stage_tail(draws.own_main, circle),
                own_mask_tail: stage_tail(draws.own_mask, circle),
            });
        }
        if turn == 32 {
            return Err(StageFailure {
                stage: 0,
                point: "positions and random coordinates do not settle",
            });
        }
        folded_t = next_folded;
        omega_t = next_omega;
        own_t = next_own;
    }
    Err(StageFailure {
        stage: 0,
        point: "positions and random coordinates do not settle",
    })
}


/// The converged first stage of one group solved on its own draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SingleFirstStage {
    pub k: u32,
    pub h: u32,
    pub positions: u64,
    pub main_tail: u64,
    pub mask_tail: u64,
}

/// Solves the first stage of one group on its own draws. It is the own-group
/// branch of [`solve_first_stage`] without the groups that read the same
/// messages: the main and mask words are opened at the group's own position
/// count, and the batched rows are the group's statement rows. The error
/// analysis is the same `sized_step` the joint solve uses.
pub fn solve_single_first_stage(
    main: CodeParams,
    mask: CodeParams,
    main_rows: u64,
    mask_rows: u64,
    rows: u64,
    circle: bool,
) -> std::result::Result<SingleFirstStage, StageFailure> {
    if rows == 0 || main_rows == 0 {
        return Err(StageFailure {
            stage: 0,
            point: "the group has no rows",
        });
    }
    let (k, h) = lin_group_bits(&[main_rows], &[mask_rows]).map_err(|_| StageFailure {
        stage: 0,
        point: "the group's member rows do not form a group",
    })?;
    let shape = group_shape(main, mask, k, h).map_err(|_| StageFailure {
        stage: 0,
        point: "the group's codes do not form a group",
    })?;
    let extra = margin_bound();
    let mut t = 0u64;
    for _ in 0..MAX_STAGE_ROUNDS {
        let (step, serves) =
            sized_step(&shape, t, t, rows, circle, &extra).map_err(|point| StageFailure {
                stage: 0,
                point,
            })?;
        let next = t.max(step);
        if next == t {
            if !serves {
                return Err(StageFailure {
                    stage: 0,
                    point: "no distance under its rule holds the group's rounds at the target",
                });
            }
            return Ok(SingleFirstStage {
                k,
                h,
                positions: t,
                main_tail: stage_tail(t, circle),
                mask_tail: stage_tail(t, circle),
            });
        }
        t = next;
    }
    Err(StageFailure {
        stage: 0,
        point: "positions and random coordinates do not settle",
    })
}

/// Solves the first stage of both groups when they share one mask code, as
/// the current executor's layout requires. The mask code's random tail is one
/// number: it must hide both groups' stage-0 lists, so the coupled solve uses
/// the larger of the two mask draw counts for both groups' mask words. The
/// main code is shared by construction; a split group is refused because this
/// layout has no message for a second main code.
#[allow(clippy::too_many_arguments)]
pub fn solve_first_stage_one_mask(
    folded: &JointGroup,
    omega: &JointGroup,
    own: &JointGroup,
    requests: u64,
    samples: u32,
    columns: u64,
    circle: bool,
    main_spends: &[u64],
    mask_spends: &[u64],
) -> std::result::Result<JointFirstStage, StageFailure> {
    if folded.main.log_n != omega.main.log_n
        || folded.main.ell != omega.main.ell
        || folded.mask.log_n != omega.mask.log_n
        || folded.mask.ell != omega.mask.ell
        || folded.main.log_n != own.main.log_n
        || folded.main.ell != own.main.ell
        || folded.mask.log_n != own.mask.log_n
        || folded.mask.ell != own.mask.ell
    {
        return Err(StageFailure {
            stage: 0,
            point: "the groups do not share one main and one mask code",
        });
    }
    let folded_shape = group_shape(folded.main, folded.mask, folded.k, folded.h)
        .map_err(|_| StageFailure { stage: 0, point: "the folded group's codes do not form a group" })?;
    let own_shape = group_shape(own.main, own.mask, own.k, own.h)
        .map_err(|_| StageFailure { stage: 0, point: "the request group's codes do not form a group" })?;
    let samples64 = samples as u64;
    let rows_folded = columns
        .checked_add(1)
        .and_then(|value| value.checked_add(3u64.checked_mul(samples64)?))
        .ok_or(StageFailure { stage: 0, point: "the folded group's batched rows overflow" })?;
    let rows_own = 2u64
        .checked_add(3u64.checked_mul(samples64).ok_or(StageFailure { stage: 0, point: "the request group's batched rows overflow" })?)
        .ok_or(StageFailure { stage: 0, point: "the request group's batched rows overflow" })?;
    let extra = margin_bound();
    let mut folded_t = 0u64;
    let mut own_t = 0u64;
    for turn in 0..=32 {
        let draws = joint_draws(folded_t, 0, own_t, requests, false)
            .map_err(|_| StageFailure { stage: 0, point: "the joint draw counts overflow" })?;
        // The one main code and the one mask code each carry every object that
        // shares them. An object is opened at its SEND readings and at the
        // request's stage-0 list; the code's random tail must hide every such
        // object's openings, so it is the largest of the per-object tails.
        let mut main_tail = stage_tail(draws.folded_main, circle).max(stage_tail(draws.own_main, circle));
        for spend in main_spends {
            let total = spend
                .checked_add(own_t)
                .ok_or(StageFailure { stage: 0, point: "an object's openings overflow" })?;
            main_tail = main_tail.max(stage_tail(total, circle));
        }
        let mut mask_tail = stage_tail(draws.folded_mask.max(draws.own_mask), circle);
        for spend in mask_spends {
            let total = spend
                .checked_add(own_t)
                .ok_or(StageFailure { stage: 0, point: "an object's openings overflow" })?;
            mask_tail = mask_tail.max(stage_tail(total, circle));
        }
        let folded_step = sized_step_tails(&folded_shape, main_tail, mask_tail, rows_folded, &extra)
            .map_err(|point| StageFailure { stage: 0, point })?;
        let own_step = sized_step_tails(&own_shape, main_tail, mask_tail, rows_own, &extra)
            .map_err(|point| StageFailure { stage: 0, point })?;
        let next_folded = folded_t.max(folded_step.0);
        let next_own = own_t.max(own_step.0);
        if next_folded == folded_t && next_own == own_t {
            if !folded_step.1 || !own_step.1 {
                return Err(StageFailure {
                    stage: 0,
                    point: "no distance under its rule holds the group's rounds at the target",
                });
            }
            return Ok(JointFirstStage {
                folded_positions: next_folded,
                omega_positions: 0,
                own_positions: next_own,
                folded_main_tail: main_tail,
                folded_mask_tail: mask_tail,
                omega_main_tail: main_tail,
                omega_mask_tail: mask_tail,
                own_main_tail: main_tail,
                own_mask_tail: mask_tail,
            });
        }
        if turn == 32 {
            return Err(StageFailure {
                stage: 0,
                point: "positions and random coordinates do not settle",
            });
        }
        folded_t = next_folded;
        own_t = next_own;
    }
    Err(StageFailure {
        stage: 0,
        point: "positions and random coordinates do not settle",
    })
}

/// One encoded object's position-hiding account: the random tail hides
/// `tail` opened positions in all, `spent` of them are already opened against
/// this object. A main object and a mask object keep separate accounts: one
/// object's tail never covers another's openings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HidingAccount {
    pub tail: u64,
    pub spent: u64,
}

impl HidingAccount {
    /// The positions this object's tail can still hide.
    pub fn remaining(&self) -> u64 {
        self.tail.saturating_sub(self.spent)
    }

    /// Whether one opened list of `list_len` positions still fits.
    pub fn covers(&self, list_len: u64) -> bool {
        list_len > 0 && list_len <= self.remaining()
    }
}

/// Checks that one unified stage list fits the remaining capacity of every
/// affected object, each named. The first object that does not cover the list
/// is named with the gap; the objects are never pooled.
pub fn hiding_capacity(
    names: &[&str],
    accounts: &[HidingAccount],
    list_len: u64,
) -> std::result::Result<(), String> {
    if names.len() != accounts.len() {
        return Err("hiding accounting: the object list and the account list differ".into());
    }
    if list_len == 0 {
        return Err("hiding accounting: empty stage list".into());
    }
    for (name, account) in names.iter().zip(accounts) {
        if !account.covers(list_len) {
            return Err(format!(
                "{name}: the stage list of {list_len} positions exceeds the remaining capacity {}",
                account.remaining()
            ));
        }
    }
    Ok(())
}

/// Whether the sampled answers fit their message: each answer takes a pad at
/// the end of the message's rows. The pads are not position-hiding capacity
/// and are counted on the message's own rows only.
pub fn ood_pads_fit(entries: u64, rows: u64, ell: u64, answers: u32) -> bool {
    if rows == 0 || ell == 0 {
        return false;
    }
    let room = rows.checked_mul(ell).and_then(|value| value.checked_sub(entries));
    match room {
        Some(room) => room >= answers as u64,
        None => false,
    }
}

/// The opened position counts of one stage: the shared group opens `shared`
/// positions and every request opens `own`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StagePositions {
    pub shared: u64,
    pub own: u64,
}

/// The positions each encoded object accumulates over the whole stage list.
/// The folded message and, in a split shared group, Omega are opened by the
/// shared test alone; Omega is also opened by every request's test; a
/// request's own objects (W, the index mask and the shown map) are opened by
/// the request's tests alone. Main and mask objects keep separate counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpenedLists {
    pub folded_main: u64,
    pub folded_mask: u64,
    pub omega_main: u64,
    pub omega_mask: u64,
    pub own_main: u64,
    pub own_mask: u64,
}

/// Sums the opened lists of the stage list. `requests` is the public batch
/// size; a single request has one.
pub fn opened_lists(
    stages: &[StagePositions],
    requests: u64,
    split: bool,
) -> Result<OpenedLists> {
    if stages.is_empty() || requests == 0 {
        return Err(Error::InvalidConfig);
    }
    let mut shared_total = 0u64;
    let mut own_total = 0u64;
    for stage in stages {
        shared_total = shared_total
            .checked_add(stage.shared)
            .ok_or(Error::ResourceLimit)?;
        own_total = own_total
            .checked_add(stage.own)
            .ok_or(Error::ResourceLimit)?;
    }
    let request_total = requests
        .checked_mul(own_total)
        .ok_or(Error::ResourceLimit)?;
    let omega = shared_total
        .checked_add(request_total)
        .ok_or(Error::ResourceLimit)?;
    Ok(OpenedLists {
        folded_main: shared_total,
        folded_mask: shared_total,
        omega_main: if split { shared_total } else { omega },
        omega_mask: if split { shared_total } else { omega },
        own_main: own_total,
        own_mask: own_total,
    })
}

/// Checks the opened lists against each object's position account, one
/// object per main and mask row set. Every affected object is checked; the
/// first that does not fit is named with its gap.
pub fn opened_capacity(
    opened: &OpenedLists,
    folded: HidingAccount,
    omega: HidingAccount,
    own: HidingAccount,
) -> std::result::Result<(), String> {
    let checks = [
        ("folded main", folded, opened.folded_main),
        ("folded mask", folded, opened.folded_mask),
        ("Omega main", omega, opened.omega_main),
        ("Omega mask", omega, opened.omega_mask),
        ("request main", own, opened.own_main),
        ("request mask", own, opened.own_mask),
    ];
    for (name, account, list) in checks {
        if !account.covers(list) {
            return Err(format!(
                "{name}: the opened list of {list} positions exceeds the remaining capacity {}",
                account.remaining()
            ));
        }
    }
    Ok(())
}

/// One stage as the run sizes it: the two codes' lengths and the fold bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunStage {
    pub log_n: u32,
    pub mask_log_n: u32,
    pub main_ell: u64,
    pub mask_ell: u64,
    pub k: u32,
    pub h: u32,
}

/// A measurement or message list longer than the stage was sized for opens a
/// row at more positions than its random tail hides. The stage is named.
pub fn stage_list_within(
    list_lengths: &[u64],
    sized_positions: &[u64],
) -> std::result::Result<(), StageFailure> {
    if list_lengths.len() != sized_positions.len() || list_lengths.is_empty() {
        return Err(StageFailure {
            stage: 0,
            point: "the planned lists and the stages differ",
        });
    }
    for (stage, (list, sized)) in list_lengths.iter().zip(sized_positions).enumerate() {
        if list > sized {
            return Err(StageFailure {
                stage,
                point: "a list is longer than the stage's random coordinates were sized for",
            });
        }
    }
    Ok(())
}

/// Compares the run's stage sizes with the solved stages: the same count, and
/// each stage's fold bits, main and mask lengths and code lengths must match.
/// A mismatch names the stage and the field instead of being accepted.
pub fn iopp_sized_fault(run: &[RunStage], solved: &[DerivedStage]) -> std::result::Result<(), StageFailure> {
    if run.len() != solved.len() {
        return Err(StageFailure {
            stage: 0,
            point: "the proximity test is sized for another number of stages than it has",
        });
    }
    for (stage, (actual, solved)) in run.iter().zip(solved).enumerate() {
        if actual.k != solved.k || actual.h != solved.h {
            return Err(StageFailure {
                stage,
                point: "a stage of the proximity test is sized for another number of rows than it has",
            });
        }
        if actual.log_n != solved.log_n
            || actual.mask_log_n != solved.mask_log_n
            || actual.main_ell != solved.main_ell
            || actual.mask_ell != solved.mask_ell
        {
            return Err(StageFailure {
                stage,
                point: "a stage of the proximity test names another code than the solved one",
            });
        }
    }
    Ok(())
}

/// The rounds of one proximity-test chain, counted from the stage list: one
/// BAT round, for every stage but the last one epsilon round and its fold
/// rounds and one code-switch round, and the last stage's masked-target round
/// and base round. Roots and scalars are message parts, not rounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoundBudget {
    pub bat: u64,
    pub fold_epsilon: u64,
    pub fold_rounds: u64,
    pub code_switch: u64,
    pub base_lambda: u64,
    pub base_positions: u64,
    pub total: u64,
}

/// Counts the chain's rounds. A chain with fewer than two stages is refused
/// with its failing point.
pub fn chain_round_budget(stages: &[DerivedStage]) -> std::result::Result<RoundBudget, StageFailure> {
    if stages.len() < 2 {
        return Err(StageFailure {
            stage: 0,
            point: "a proximity test has at least two stages",
        });
    }
    let mut budget = RoundBudget {
        bat: 1,
        fold_epsilon: 0,
        fold_rounds: 0,
        code_switch: 0,
        base_lambda: 1,
        base_positions: 1,
        total: 0,
    };
    for (index, stage) in stages.iter().enumerate() {
        if index + 1 == stages.len() {
            break;
        }
        budget.fold_epsilon += 1;
        budget.fold_rounds = budget
            .fold_rounds
            .checked_add(stage.k as u64)
            .ok_or(StageFailure { stage: index, point: "the fold round count overflows" })?;
        budget.code_switch += 1;
    }
    budget.total = budget
        .bat
        .checked_add(budget.fold_epsilon)
        .and_then(|value| value.checked_add(budget.fold_rounds))
        .and_then(|value| value.checked_add(budget.code_switch))
        .and_then(|value| value.checked_add(budget.base_lambda))
        .and_then(|value| value.checked_add(budget.base_positions))
        .ok_or(StageFailure { stage: 0, point: "the chain round count overflows" })?;
    Ok(budget)
}

/// The chain's rounds must fit the reserved budget. A chain that does not fit
/// is named with its overage; the reserved number is never raised here.
pub fn budget_within(budget: RoundBudget, reserved: u64) -> std::result::Result<(), String> {
    if budget.total > reserved {
        return Err(format!(
            "the chain needs {} rounds but only {reserved} are reserved",
            budget.total
        ));
    }
    Ok(())
}

/// The capacity recheck over a whole chain: the opened lists derived from the
/// stage positions must fit each object's account. Positions only; the table,
/// range and link conditions stay outside this check.
#[allow(clippy::too_many_arguments)]
pub fn capacity_recheck(
    positions: &[StagePositions],
    requests: u64,
    split: bool,
    folded: HidingAccount,
    omega: HidingAccount,
    own: HidingAccount,
) -> std::result::Result<(), String> {
    let opened = opened_lists(positions, requests, split)
        .map_err(|_| "capacity recheck: the opened lists overflow".to_string())?;
    opened_capacity(&opened, folded, omega, own)
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
        mask_enabled: true,
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

/// One candidate's recomputed own-chain accounting. The values are the
/// planner's outward-rounded upper bounds; they are reported to the acceptance
/// note and never feed a planner decision.
#[derive(Clone, Copy, Debug)]
pub struct TwoStageReport {
    pub plan: TwoStagePlan,
    /// `agree^t` of the stage-0 query round, in bits.
    pub stage_zero_query_bits: f64,
    /// The last stage's carried term `list * powers_error(2 + 3t)`, in bits.
    pub stage_zero_carried_bits: f64,
    /// The stage-0 target `2^-(128 - work)`, in bits.
    pub stage_zero_target_bits: f64,
    /// `query + carried` in bits plus the stage-0 work bits.
    pub stage_zero_joint_bits: f64,
    /// The last stage's BASE error, in bits.
    pub last_base_bits: f64,
    /// The last stage's target `2^-128`, in bits.
    pub last_target_bits: f64,
}

/// Recomputes the candidate's error components from the trusted codes. The
/// returned plan is exactly the plan `plan_own_two_stage` returns; the bits
/// only expose the bounds that the planner already checked.
pub fn plan_own_two_stage_report(
    main: CodeParams,
    mask: CodeParams,
    last_main: CodeParams,
    last_mask: CodeParams,
    k: u32,
    h: u32,
    max_stage_zero_positions: usize,
) -> Result<TwoStageReport> {
    let plan = plan_own_two_stage(main, mask, last_main, last_mask, k, h, max_stage_zero_positions)?;
    let stage_zero = group_shape(main, mask, k, h)?;
    let zero_load = group_load(&stage_zero, main.r);
    let zero_m = largest_m(&stage_zero, zero_load);
    let zero_analysis = analysis_of_m(&stage_zero, zero_m)?;
    if zero_analysis.bad || zero_m != plan.stage_zero_m {
        return Err(Error::InvalidConfig);
    }
    let (last_r, last_m, last_analysis) = plan_last(last_main, last_mask, main.r)?;
    if last_m != plan.last_m || last_r as usize != plan.last_positions {
        return Err(Error::InvalidConfig);
    }
    let t = u64::try_from(plan.stage_zero_positions).map_err(|_| Error::ResourceLimit)?;
    let rows = 1u64
        .checked_add(1)
        .and_then(|value| value.checked_add(3u64.checked_mul(t)?))
        .ok_or(Error::ResourceLimit)?;
    let carried = Bound::from_big(last_analysis.list.clone()).mul(&powers_error(rows));
    let query = zero_analysis.at.agree.pow(t);
    let sum = query.add(&carried);
    let last_main_code = CodeParams {
        log_n: last_main.log_n,
        ell: last_main.ell,
        r: last_r,
    };
    let last_mask_code = CodeParams {
        log_n: last_mask.log_n,
        ell: last_mask.ell,
        r: last_r,
    };
    let last_group = group_shape(last_main_code, last_mask_code, 0, 0)?;
    let last_again = analysis_of_m(&last_group, last_m)?;
    if last_again.bad {
        return Err(Error::InvalidConfig);
    }
    let last_base = base_error(&last_group, &last_again);
    let work = u32::from(plan.stage_zero_work_bits);
    Ok(TwoStageReport {
        plan,
        stage_zero_query_bits: bits_of(&query.error()),
        stage_zero_carried_bits: bits_of(&carried.error()),
        stage_zero_target_bits: bits_of(&goal(work).error()),
        stage_zero_joint_bits: bits_of(&sum.error()) + f64::from(work),
        last_base_bits: bits_of(&last_base.error()),
        last_target_bits: bits_of(&goal(0).error()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::code::build_code_plan;

    /// The shared linear layer derives exactly the three recorded terms: the
    /// fold point's error, the agreement power at the column count, and the
    /// comparison point's field error. An unreachable agreement and a zero
    /// instance or row count are refused.
    #[test]
    fn the_shared_lin_bounds_derive_the_query_count_and_error_terms() {
        let word = Word {
            log_n: 12,
            ell: 64,
            k: 16,
        };
        let at = johnson(&word, 8).unwrap();
        assert!(!at.bad());
        let proximity = mca_error(&at);
        let shape = SharedLinShape {
            v: 10,
            k_lo: 6,
            instances: 2,
            rows_log: 10,
        };
        let bounds = shared_lin_bounds(shape, &at.agree, &at.list, &proximity, 0, 0).unwrap();
        let mut fold = Bound::from_u64(0);
        for _ in 1..=shape.v {
            fold = fold.add(&proximity);
        }
        let fold = fold.add(
            &Bound::from_big(at.list.clone()).mul(&field_error(u64::from(shape.v))),
        );
        let expected = Bound::from_u64(shape.instances)
            .mul(&field_error(u64::from(shape.v) + u64::from(shape.k_lo)))
            .add(&fold);
        assert_eq!(expected, bounds.point);
        assert!(bounds.columns > 0);
        assert!(surely_le(&bounds.column_error, &goal(0)));
        let zeta = field_error(u64::from(shape.rows_log));
        assert_eq!(zeta, bounds.zeta);

        let one = Bound::from_u64(1);
        assert!(shared_lin_bounds(shape, &one, &at.list, &proximity, 0, 0).is_err());
        let mut bad = shape;
        bad.instances = 0;
        assert!(shared_lin_bounds(bad, &at.agree, &at.list, &proximity, 0, 0).is_err());
        let mut bad = shape;
        bad.v = 0;
        assert!(shared_lin_bounds(bad, &at.agree, &at.list, &proximity, 0, 0).is_err());
    }

    /// The plan check derives a column count from a realistic table distance
    /// and refuses a plan that draws fewer columns or carries no distance; a
    /// more conservative count is accepted.
    #[test]
    fn the_shared_lin_plan_check_rejects_under_met_candidates() {
        let code = build_code_plan(12, 2048, 128).unwrap();
        let m = recorded_table_distance(code, 10, 1, 10, 1, 0).unwrap();
        assert!(m >= 3);
        let shape = SharedLinShape {
            v: 10,
            k_lo: 0,
            instances: 1,
            rows_log: 10,
        };
        let columns = shared_lin_columns(shape, code, m, 0, 0).unwrap();
        assert!(columns > 0);
        let fault = SharedLinPlanFault {
            v: 10,
            k_lo: 0,
            rows_log: 10,
            m,
            column_draws: columns,
            point_work_bits: 0,
            column_work_bits: 0,
            zeta_work_bits: 0,
        };
        shared_lin_plan_fault(fault, code).unwrap();
        let mut conservative = fault;
        conservative.column_draws = columns + 1;
        shared_lin_plan_fault(conservative, code).unwrap();
        let mut fewer = fault;
        fewer.column_draws = columns - 1;
        assert!(shared_lin_plan_fault(fewer, code).is_err());
        let mut no_distance = fault;
        no_distance.m = 0;
        assert!(shared_lin_plan_fault(no_distance, code).is_err());
    }

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

    /// The candidate report recomputes the same plan and the stage bounds the
    /// planner checked; the joint accounting never falls below 128 bits.
    #[test]
    fn the_candidate_report_recomputes_the_stage_bounds() {
        let main = build_code_plan(17, 65_536, 8_192).unwrap();
        let mask = build_code_plan(17, 512, 256).unwrap();
        let last_main = build_code_plan(17, 65_536, 1).unwrap();
        let last_mask = build_code_plan(17, 16_384, 1).unwrap();
        let report =
            plan_own_two_stage_report(main, mask, last_main, last_mask, 14, 2, 252).unwrap();
        assert_eq!(
            report.plan,
            plan_own_two_stage(main, mask, last_main, last_mask, 14, 2, 252).unwrap()
        );
        assert!(report.stage_zero_target_bits <= 128.0);
        assert!(report.stage_zero_joint_bits >= 127.999);
        assert!(report.last_target_bits <= 128.0);
        assert!(report.last_base_bits >= 127.999);
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
            mask_enabled: true,
        };
        let load = GroupLoad {
            cube_bits: 3,
            messages: 1,
            batched_rows: u64::MAX,
        };
        assert_eq!(largest_m(&tiny, load), 0);
    }

}

#[cfg(test)]
mod lin_group_bits_tests {
    use super::*;

    #[test]
    fn group_bits_follow_the_member_row_blocks() {
        // A request's group of an input mask message, W and Omega with one
        // main row each: two main rows and one mask row.
        assert_eq!(lin_group_bits(&[1, 1, 0], &[0, 0, 1]).unwrap(), (1, 0));
        // Unaligned main rows are rounded up to their aligned blocks.
        assert_eq!(lin_group_bits(&[3, 2, 0], &[0, 0, 1]).unwrap(), (3, 0));
        // The table-sized shape: a 1024-row W, a 513-row Omega and a four-row
        // mask message.
        assert_eq!(lin_group_bits(&[1024, 513, 0], &[0, 0, 4]).unwrap(), (11, 2));
        // A group with only mask rows still needs one main row bit.
        assert_eq!(lin_group_bits(&[0], &[1]).unwrap(), (1, 0));
        assert!(lin_group_bits(&[], &[]).is_err());
        assert!(lin_group_bits(&[1], &[1, 1]).is_err());
    }
}

#[cfg(test)]
mod stage_sizes_tests {
    use super::*;

    #[test]
    fn middle_stage_follows_the_capacity_rules() {
        // A 1024-entry row cut into 2^1 rows leaves 512 entries; the mask
        // carries two previous mask tails, one slot of the previous main tail
        // and two of its own mask tails.
        let sizes = solve_next_stage(1024, 32, 2, 3, 1, 1, 1, 1, 7, false).unwrap();
        assert_eq!(sizes.main_ell, 1024);
        assert_eq!(sizes.tail, 7);
        let carried = 2 * 32 + 2 * 3 + 2 * 1;
        let need = carried + 1;
        let rows = (need + (1u64 << 1) - 1) >> 1;
        let mask_ell = pow2_at_least_u64(rows).unwrap().max(2);
        assert_eq!(sizes.mask_ell, mask_ell);
        assert_eq!(sizes.r_ood, (mask_ell << 1) - carried);
        // An indivisible cut is rejected.
        assert!(solve_next_stage(1023, 32, 1, 3, 1, 1, 0, 1, 7, false).is_err());
        // The mask bits cannot exceed the main bits.
        assert!(solve_next_stage(1024, 32, 1, 3, 1, 1, 2, 1, 7, false).is_err());
    }

    #[test]
    fn last_stage_is_one_row_and_the_tail_follows_the_domain() {
        let sizes = solve_last_stage(512, 64, 2, 5, 1, 1, 9, false).unwrap();
        assert_eq!(sizes.k, 0);
        assert_eq!(sizes.h, 0);
        assert_eq!(sizes.main_ell, 1024);
        let carried = 2 * 64 + 2 * 5 + 2 * 1;
        assert_eq!(sizes.mask_ell, pow2_at_least_u64(carried + 1).unwrap());
        assert_eq!(sizes.r_ood, sizes.mask_ell - carried);
        assert_eq!(sizes.tail, 9);
        assert_eq!(stage_tail(8, false), 8);
        assert_eq!(stage_tail(8, true), 9);
        assert_eq!(stage_tail(9, true), 9);
        assert_eq!(stage_tail(0, true), 0);
    }
}

#[cfg(test)]
mod stage_list_tests {
    use super::*;

    fn choice(log_n: u32, mask_log_n: u32, k: u32, h: u32) -> StageChoice {
        StageChoice { log_n, mask_log_n, k, h }
    }

    fn first() -> FirstStage {
        FirstStage { k: 1, h: 0, main_ell: 2048, mask_ell: 64, slots: 2, main_r: 5, mask_r: 5 }
    }

    #[test]
    fn a_configured_chain_derives_and_reports_the_failing_point() {
        let choices = vec![choice(14, 13, 0, 1), choice(14, 13, 1, 1), choice(14, 14, 0, 0)];
        let stages = derive_stage_list(first(), 1, 7, false, &choices).unwrap();
        assert_eq!(stages.len(), 3);
        assert_eq!(stages[1].k, 1);
        assert_eq!(stages[2].k, 0);
        assert_eq!(stages[2].h, 0);
        assert!(stages.windows(2).all(|pair| pair[1].tail == 7));

        // A middle stage cannot be the single-row stage.
        let bad = vec![choice(14, 13, 0, 1), choice(14, 14, 0, 0), choice(14, 14, 0, 0)];
        let failure = derive_stage_list(first(), 1, 7, false, &bad).unwrap_err();
        assert_eq!(failure.stage, 1);
        assert_eq!(failure.point, "a middle stage has 1 <= k and h <= k");

        // The last stage must be the single-row stage.
        let bad = vec![choice(14, 13, 0, 1), choice(14, 14, 1, 0)];
        let failure = derive_stage_list(first(), 1, 7, false, &bad).unwrap_err();
        assert_eq!(failure.stage, 1);
        assert_eq!(failure.point, "the last stage has k = h = 0");

        // An indivisible cut is a configuration failure at its stage.
        let odd = FirstStage { k: 1, h: 0, main_ell: 2046, mask_ell: 64, slots: 1, main_r: 5, mask_r: 5 };
        let bad = vec![choice(14, 13, 0, 1), choice(14, 14, 3, 0), choice(14, 14, 0, 0)];
        let failure = derive_stage_list(odd, 1, 7, false, &bad).unwrap_err();
        assert_eq!(failure.stage, 1);
        assert!(failure.point.contains("cut"));

        // A code length that does not hold the message and its tail fails.
        let bad = vec![choice(10, 13, 0, 1)];
        let failure = derive_stage_list(first(), 1, 7, false, &bad).unwrap_err();
        assert_eq!(failure.stage, 0);
        assert!(failure.point.contains("fit its code length"));
    }
}

#[cfg(test)]
mod joint_draw_tests {
    use super::*;

    #[test]
    fn joint_draws_account_every_reader() {
        // One shared code: Omega is read by the shared test and by the one
        // request's test.
        let draws = joint_draws(11, 11, 7, 1, false).unwrap();
        assert_eq!(draws.omega_main, 18);
        assert_eq!(draws.folded_main, 18);
        assert_eq!(draws.own_main, 18);
        assert_eq!(draws.own_mask, 7);
        // A split shared group reads Omega's own message in both tests.
        let draws = joint_draws(11, 13, 7, 1, true).unwrap();
        assert_eq!(draws.omega_main, 20);
        assert_eq!(draws.folded_main, 11);
        assert_eq!(draws.folded_mask, 24);
        assert_eq!(draws.omega_mask, 24);
        // A batch multiplies only the request term.
        let draws = joint_draws(11, 11, 7, 2, false).unwrap();
        assert_eq!(draws.omega_main, 25);
        assert_eq!(draws.own_mask, 7);
        assert!(joint_draws(11, 11, 7, 0, false).is_err());
    }
}

#[cfg(test)]
mod first_stage_tests {
    use super::*;

    fn group(log_n: u32, ell: u64, k: u32, h: u32) -> JointGroup {
        JointGroup {
            main: CodeParams { log_n, ell, r: 0 },
            mask: CodeParams { log_n, ell, r: 0 },
            k,
            h,
        }
    }

    #[test]
    fn the_first_stage_settles_or_names_a_failing_point() {
        let g = group(14, 256, 1, 0);
        match solve_first_stage(&g, &g, &g, 1, 1, 64, false) {
            Ok(stage) => {
                assert!(stage.folded_positions > 0);
                assert!(stage.own_positions > 0);
                assert_eq!(stage.omega_positions, 0);
                // One request: Omega's read count is the folded positions plus
                // the request's positions; the mask draw of the joined group
                // is the folded positions.
                assert_eq!(
                    stage.folded_main_tail,
                    stage.folded_positions + stage.own_positions
                );
                assert_eq!(stage.folded_mask_tail, stage.folded_positions);
                assert_eq!(stage.own_main_tail, stage.folded_main_tail);
                assert_eq!(stage.own_mask_tail, stage.own_positions);
            }
            Err(failure) => {
                assert_eq!(failure.stage, 0);
                assert!(!failure.point.is_empty());
            }
        }
    }

    #[test]
    fn a_tail_that_cannot_fit_the_code_is_reported() {
        // The message fills the whole code length, so any positive random tail
        // is a labeled capacity failure rather than a panic.
        let g = group(3, 8, 1, 0);
        let failure = solve_first_stage(&g, &g, &g, 1, 1, 8, false).unwrap_err();
        assert_eq!(failure.stage, 0);
        assert!(!failure.point.is_empty());
    }

    #[test]
    fn a_zero_batch_is_rejected() {
        let g = group(14, 256, 1, 0);
        assert!(solve_first_stage(&g, &g, &g, 0, 1, 64, false).is_err());
    }
}

#[cfg(test)]
mod hiding_account_tests {
    use super::*;

    #[test]
    fn spent_and_remaining_are_per_object() {
        let main = HidingAccount { tail: 100, spent: 40 };
        let mask = HidingAccount { tail: 60, spent: 60 };
        assert_eq!(main.remaining(), 60);
        assert_eq!(mask.remaining(), 0);
        assert!(hiding_capacity(&["main"], &[main], 60).is_ok());
        // The two objects are not pooled: the main object cannot lend its
        // remaining capacity to the mask object.
        let error = hiding_capacity(&["main", "mask"], &[main, mask], 30).unwrap_err();
        assert!(error.contains("mask"));
        assert!(error.contains("remaining capacity 0"));
        // A list longer than one object's remaining capacity is named.
        let error = hiding_capacity(&["main"], &[main], 61).unwrap_err();
        assert!(error.contains("main"));
        assert!(hiding_capacity(&["main"], &[main], 0).is_err());
    }

    #[test]
    fn pads_and_hiding_capacity_are_counted_separately() {
        // The message has room for the entries and one answer pad.
        assert!(ood_pads_fit(100, 2, 64, 1));
        assert!(!ood_pads_fit(127, 2, 64, 2));
        // A message with pads left can still fail the position account, and
        // the reverse.
        let spent = HidingAccount { tail: 10, spent: 10 };
        assert!(!hiding_capacity(&["mask"], &[spent], 1).is_ok());
        assert!(ood_pads_fit(10, 1, 16, 1));
    }
}

#[cfg(test)]
mod opened_list_tests {
    use super::*;

    #[test]
    fn opened_lists_split_by_object() {
        let stages = [
            StagePositions { shared: 11, own: 7 },
            StagePositions { shared: 5, own: 3 },
        ];
        let opened = opened_lists(&stages, 1, false).unwrap();
        assert_eq!(opened.folded_main, 16);
        assert_eq!(opened.omega_main, 26);
        assert_eq!(opened.own_main, 10);
        let split = opened_lists(&stages, 2, true).unwrap();
        assert_eq!(split.folded_main, 16);
        assert_eq!(split.omega_main, 16);
        assert_eq!(split.own_main, 10);
        assert!(opened_lists(&[], 1, false).is_err());
        assert!(opened_lists(&stages, 0, false).is_err());
    }

    #[test]
    fn opened_capacity_names_the_failing_object() {
        let stages = [StagePositions { shared: 11, own: 7 }];
        let opened = opened_lists(&stages, 1, false).unwrap();
        let folded = HidingAccount { tail: 20, spent: 0 };
        let omega = HidingAccount { tail: 17, spent: 0 };
        let own = HidingAccount { tail: 7, spent: 0 };
        // Omega needs 18 and has 17: named, with the gap.
        let error = opened_capacity(&opened, folded, omega, own).unwrap_err();
        assert!(error.contains("Omega"));
        assert!(error.contains("18"));
        // With enough tail everywhere the lists fit.
        let omega = HidingAccount { tail: 18, spent: 0 };
        assert!(opened_capacity(&opened, folded, omega, own).is_ok());
        // The request object is checked on its own list, not Omega's.
        let own = HidingAccount { tail: 6, spent: 0 };
        let error = opened_capacity(&opened, folded, omega, own).unwrap_err();
        assert!(error.contains("request"));
    }
}

#[cfg(test)]
mod plan_iopp_tests {
    use super::*;

    fn solved(k: u32, h: u32, main_ell: u64, mask_ell: u64) -> DerivedStage {
        DerivedStage {
            log_n: 14,
            mask_log_n: 14,
            k,
            h,
            main_ell,
            mask_ell,
            r_ood: 0,
            tail: 7,
        }
    }

    fn run(k: u32, h: u32, main_ell: u64, mask_ell: u64) -> RunStage {
        RunStage {
            log_n: 14,
            mask_log_n: 14,
            main_ell,
            mask_ell,
            k,
            h,
        }
    }

    #[test]
    fn lists_longer_than_the_stage_are_rejected_and_named() {
        let sized = [4u64, 6];
        assert!(stage_list_within(&[4, 6], &sized).is_ok());
        let failure = stage_list_within(&[5, 6], &sized).unwrap_err();
        assert_eq!(failure.stage, 0);
        assert!(failure.point.contains("longer than the stage"));
        let failure = stage_list_within(&[4, 7], &sized).unwrap_err();
        assert_eq!(failure.stage, 1);
        assert!(stage_list_within(&[4], &sized).is_err());
    }

    #[test]
    fn sized_fault_names_the_mismatching_stage_and_field() {
        let solved = [solved(1, 0, 512, 64), solved(0, 0, 512, 128)];
        let matching = [run(1, 0, 512, 64), run(0, 0, 512, 128)];
        assert!(iopp_sized_fault(&matching, &solved).is_ok());
        let rows = [run(2, 0, 512, 64), run(0, 0, 512, 128)];
        let failure = iopp_sized_fault(&rows, &solved).unwrap_err();
        assert_eq!(failure.stage, 0);
        assert!(failure.point.contains("number of rows"));
        let codes = [run(1, 0, 512, 64), run(0, 0, 256, 128)];
        let failure = iopp_sized_fault(&codes, &solved).unwrap_err();
        assert_eq!(failure.stage, 1);
        assert!(failure.point.contains("another code"));
        assert!(iopp_sized_fault(&matching[..1], &solved).is_err());
    }
}

#[cfg(test)]
mod round_budget_tests {
    use super::*;

    fn stage(k: u32) -> DerivedStage {
        DerivedStage {
            log_n: 14,
            mask_log_n: 14,
            k,
            h: 0,
            main_ell: 512,
            mask_ell: 64,
            r_ood: 0,
            tail: 7,
        }
    }

    #[test]
    fn rounds_follow_the_stage_list() {
        let stages = [stage(1), stage(1), stage(0)];
        let budget = chain_round_budget(&stages).unwrap();
        assert_eq!(budget.bat, 1);
        assert_eq!(budget.fold_epsilon, 2);
        assert_eq!(budget.fold_rounds, 2);
        assert_eq!(budget.code_switch, 2);
        assert_eq!(budget.base_lambda, 1);
        assert_eq!(budget.base_positions, 1);
        assert_eq!(budget.total, 9);
        assert!(budget_within(budget, 9).is_ok());
        let error = budget_within(budget, 8).unwrap_err();
        assert!(error.contains("9"));
        assert!(error.contains("8"));
        assert!(chain_round_budget(&stages[..1]).is_err());
    }

    #[test]
    fn the_capacity_recheck_composes_the_object_checks() {
        let stages = [StagePositions { shared: 11, own: 7 }];
        let folded = HidingAccount { tail: 20, spent: 0 };
        let omega = HidingAccount { tail: 18, spent: 0 };
        let own = HidingAccount { tail: 7, spent: 0 };
        assert!(capacity_recheck(&stages, 1, false, folded, omega, own).is_ok());
        let own = HidingAccount { tail: 6, spent: 0 };
        let error = capacity_recheck(&stages, 1, false, folded, omega, own).unwrap_err();
        assert!(error.contains("request"));
    }
}

#[cfg(test)]
mod stage_solve_tests {
    use super::*;

    fn group(log_n: u32, ell: u64, k: u32, h: u32) -> JointGroup {
        JointGroup {
            main: CodeParams { log_n, ell, r: 0 },
            mask: CodeParams { log_n, ell, r: 0 },
            k,
            h,
        }
    }

    #[test]
    fn one_mask_solution_uses_one_mask_tail() {
        let g = group(14, 256, 1, 0);
        let stage = solve_first_stage_one_mask(&g, &g, &g, 1, 1, 64, false, &[], &[]).unwrap();
        assert_eq!(stage.folded_positions, 47);
        assert_eq!(stage.own_positions, 47);
        assert_eq!(stage.omega_positions, 0);
        // The one mask code hides the larger of the two stage-0 lists; the one
        // main code hides the shared list plus the request's list.
        assert_eq!(stage.folded_mask_tail, stage_tail(47, false));
        assert_eq!(stage.folded_mask_tail, stage.own_mask_tail);
        assert_eq!(stage.omega_mask_tail, stage.folded_mask_tail);
        assert_eq!(stage.folded_main_tail, stage_tail(94, false));
        assert_eq!(stage.folded_main_tail, stage.own_main_tail);
    }

    #[test]
    fn groups_with_other_mask_codes_are_refused() {
        let g = group(14, 256, 1, 0);
        let other = group(14, 128, 1, 0);
        let failure = solve_first_stage_one_mask(&g, &g, &other, 1, 1, 64, false, &[], &[]).unwrap_err();
        assert_eq!(failure.stage, 0);
        assert!(failure.point.contains("one main and one mask code"));
    }

    #[test]
    fn a_chain_solves_every_stage_from_its_choice() {
        let choices = vec![
            StageChoice { log_n: 14, mask_log_n: 14, k: 0, h: 0 },
            StageChoice { log_n: 14, mask_log_n: 14, k: 2, h: 1 },
            StageChoice { log_n: 14, mask_log_n: 14, k: 0, h: 0 },
        ];
        let start = ChainStart {
            main_ell: 256,
            mask_ell: 256,
            k: 1,
            h: 0,
            positions: 4,
            main_r: 4,
            mask_r: 4,
        };
        let stages = solve_chain_stages(start, 1, false, &choices).unwrap();
        assert_eq!(stages.len(), 3);
        assert_eq!(stages[0].k, 1);
        assert_eq!(stages[0].positions, 4);
        assert_eq!(stages[0].main_r, 4);
        assert_eq!(stages[0].mask_r, 4);
        // The first choice carries the first stage's code lengths, not a
        // post-first stage: the middle stage is choices[1].
        assert_eq!(stages[1].k, 2);
        assert_eq!(stages[1].h, 1);
        assert_eq!(stages[1].main_ell, 64);
        assert_eq!(stages[1].mask_ell, 512);
        assert_eq!(stages[1].positions, 53);
        assert_eq!(stages[1].main_r, 53);
        assert_eq!(stages[1].mask_r, 53);
        assert_eq!(stages[2].k, 0);
        assert_eq!(stages[2].h, 0);
        assert_eq!(stages[2].main_ell, 64);
        assert_eq!(stages[2].mask_ell, 2048);
        assert_eq!(stages[2].positions, 88);
        assert_eq!(stages[2].main_r, 88);
        assert_eq!(stages[2].mask_r, 88);
        // Every stage opens no more positions than its random tail hides.
        for stage in &stages {
            assert!(stage.positions <= stage.main_r.min(stage.mask_r));
        }
    }

    #[test]
    fn an_infeasible_choice_names_its_stage() {
        let choices = vec![
            StageChoice { log_n: 14, mask_log_n: 14, k: 0, h: 0 },
            StageChoice { log_n: 14, mask_log_n: 14, k: 1, h: 0 },
            StageChoice { log_n: 14, mask_log_n: 14, k: 0, h: 0 },
        ];
        let start = ChainStart {
            main_ell: 256,
            mask_ell: 256,
            k: 1,
            h: 0,
            positions: 4,
            main_r: 4,
            mask_r: 4,
        };
        let failure = solve_chain_stages(start, 1, false, &choices).unwrap_err();
        assert_eq!(failure.stage, 1);
        assert!(failure.point.contains("holds the stage's rounds"));
    }

    #[test]
    fn a_chain_without_a_last_stage_is_refused() {
        let choices = vec![StageChoice { log_n: 14, mask_log_n: 14, k: 1, h: 0 }];
        let start = ChainStart {
            main_ell: 256,
            mask_ell: 256,
            k: 1,
            h: 0,
            positions: 4,
            main_r: 4,
            mask_r: 4,
        };
        let failure = solve_chain_stages(start, 1, false, &choices).unwrap_err();
        assert_eq!(failure.stage, 0);
        assert!(failure.point.contains("last stage"));
    }
}


#[cfg(test)]
mod assembly_tests {
    use super::*;

    fn diagnostic_input() -> AssemblyInput {
        let mut own_objects = Vec::new();
        for _ in 0..16 {
            own_objects.push(AssemblyOwnObject {
                member: AssemblyMember::Columns {
                    entries: 1024,
                    answers: 1,
                },
                shared_too: false,
                spent: 2,
            });
        }
        own_objects.push(AssemblyOwnObject {
            member: AssemblyMember::Fixed {
                main_rows: 1,
                mask_rows: 0,
            },
            shared_too: true,
            spent: 0,
        });
        own_objects.push(AssemblyOwnObject {
            member: AssemblyMember::Fixed {
                main_rows: 0,
                mask_rows: 4,
            },
            shared_too: false,
            spent: 1,
        });
        own_objects.push(AssemblyOwnObject {
            member: AssemblyMember::Columns {
                entries: 16 * 1024,
                answers: 1,
            },
            shared_too: false,
            spent: 0,
        });
        AssemblyInput {
            shared_members: vec![
                AssemblyMember::Fixed {
                    main_rows: 1,
                    mask_rows: 0,
                },
                AssemblyMember::Fixed {
                    main_rows: 1,
                    mask_rows: 0,
                },
            ],
            own_objects,
            shared_mask_spent: 0,
            requests: 1,
            samples: 1,
            columns: 1024,
            circle: false,
        }
    }

    #[test]
    fn production_candidates_assemble_for_the_consumer_layout() {
        let plan = assemble_proximity(&diagnostic_input()).unwrap();
        // The first stage's codes are the production lines' own/Omega main
        // code and own mask code; the fold bits come from the real members.
        assert_eq!(plan.first_main.log_n, 18);
        assert_eq!(plan.first_main.ell, 1 << 14);
        assert_eq!(plan.first_mask.log_n, 18);
        assert_eq!(plan.first_mask.ell, 1 << 8);
        assert_eq!(plan.shared[0].k, 1);
        assert_eq!(plan.shared[0].h, 0);
        assert_eq!(plan.own[0].k, 5);
        assert_eq!(plan.own[0].h, 2);
        assert_eq!(plan.shared.len(), 3);
        assert_eq!(plan.own.len(), 3);
        // The own stage-0 list fits every object's remaining capacity, and the
        // minimum is recorded.
        assert!(plan.own[0].positions <= plan.max_zero);
        assert!(plan.shared[0].positions <= plan.own[0].main_r.min(plan.own[0].mask_r));
        assert_eq!(plan.first_main.r, plan.own[0].main_r);
        assert_eq!(plan.first_mask.r, plan.own[0].mask_r);
        // The reserved round count is the frozen plan's exact output.
        let shared_budget = chain_round_budget(
            &plan.shared.iter().map(|stage| DerivedStage {
                log_n: stage.log_n,
                mask_log_n: stage.mask_log_n,
                k: stage.k,
                h: stage.h,
                main_ell: stage.main_ell,
                mask_ell: stage.mask_ell,
                r_ood: stage.r_ood,
                tail: stage.main_r.max(stage.mask_r),
            }).collect::<Vec<_>>(),
        )
        .unwrap();
        let own_budget = chain_round_budget(
            &plan.own.iter().map(|stage| DerivedStage {
                log_n: stage.log_n,
                mask_log_n: stage.mask_log_n,
                k: stage.k,
                h: stage.h,
                main_ell: stage.main_ell,
                mask_ell: stage.mask_ell,
                r_ood: stage.r_ood,
                tail: stage.main_r.max(stage.mask_r),
            }).collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(plan.reserved_rounds, shared_budget.total + own_budget.total);
    }

    #[test]
    fn an_unformable_group_names_the_failure() {
        let mut input = diagnostic_input();
        input.own_objects.clear();
        let error = assemble_proximity(&input).unwrap_err();
        assert!(error.contains("no candidate found within the searched domain"));
        assert!(error.contains("the own group's member rows do not form a group"));
    }
}



/// One group's derived chain: the first-stage codes with their tails and the
/// solved stage list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DerivedSingleGroup {
    pub first_main: CodeParams,
    pub first_mask: CodeParams,
    pub stages: Vec<SolvedStage>,
}

/// Derives one group's stage chain from the message it carries. The first
/// stage is solved on the group's own draws by [`solve_single_first_stage`];
/// the later stages are the production candidate's own chain, solved by
/// [`solve_chain_stages`] exactly as the joint assembly does. The candidate
/// whose first codes hold the message and whose chain validates is the plan;
/// when none passes, the error names the last failing point.
/// One group's derived plan: the chosen first-stage codes, the mask message's
/// row count and the solved stage chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DerivedSingleGroupPlan {
    pub first_main: CodeParams,
    pub first_mask: CodeParams,
    pub mask_rows: u64,
    pub stages: Vec<SolvedStage>,
}

/// The smallest power-of-two row count of a mask message whose free entries
/// (`rows * ell - 1`) hold `entries`.
pub fn smallest_mask_rows(entries: usize, ell: u64) -> Result<u64> {
    if entries == 0 || ell == 0 {
        return Err(Error::InvalidConfig);
    }
    let mut rows = 1u64;
    loop {
        let capacity = rows
            .checked_mul(ell)
            .and_then(|value| value.checked_sub(1))
            .ok_or(Error::ResourceLimit)?;
        if capacity >= entries as u64 {
            return Ok(rows);
        }
        rows = rows.checked_mul(2).ok_or(Error::ResourceLimit)?;
    }
}

fn derive_candidate_group(
    candidate: &AssemblyCandidate,
    bit_entries: u64,
    mask_rows: u64,
    extra_mask_rows: u64,
    rows: u64,
    samples: u32,
    circle: bool,
) -> std::result::Result<DerivedSingleGroup, StageFailure> {
    let ell = 1u64 << candidate.first_log_ell;
    let mask_ell = 1u64 << candidate.first_mask_log_ell;
    let main_rows = rows_for_columns(bit_entries, ell, 1).map_err(|_| StageFailure {
        stage: 0,
        point: "the bit message does not fit the candidate's main code",
    })?;
    if main_rows == 0 || !main_rows.is_power_of_two() {
        return Err(StageFailure {
            stage: 0,
            point: "the bit message has no power-of-two row count",
        });
    }
    let total_mask_rows = mask_rows
        .checked_add(extra_mask_rows)
        .ok_or(StageFailure {
            stage: 0,
            point: "the group's mask blocks overflow",
        })?;
    let main = CodeParams {
        log_n: candidate.first_log_n,
        ell,
        r: 0,
    };
    let mask = CodeParams {
        log_n: candidate.first_mask_log_n,
        ell: mask_ell,
        r: 0,
    };
    let first = solve_single_first_stage(main, mask, main_rows, total_mask_rows, rows, circle)?;
    let start = ChainStart {
        main_ell: ell,
        mask_ell,
        k: first.k,
        h: first.h,
        positions: first.positions,
        main_r: first.main_tail,
        mask_r: first.mask_tail,
    };
    let mut last = StageFailure {
        stage: 0,
        point: "no candidate chain holds the group",
    };
    for later in [&candidate.own_choices, &candidate.shared_choices] {
        let mut choices = Vec::with_capacity(later.len() + 1);
        choices.push(StageChoice {
            log_n: candidate.first_log_n,
            mask_log_n: candidate.first_mask_log_n,
            k: 0,
            h: 0,
        });
        choices.extend(later.iter().copied());
        match solve_chain_stages(start, samples, circle, &choices) {
            Ok(stages) => match validate_solved_chain(&stages, samples) {
                Ok(()) => {
                    return Ok(DerivedSingleGroup {
                        first_main: CodeParams {
                            log_n: candidate.first_log_n,
                            ell,
                            r: first.main_tail,
                        },
                        first_mask: CodeParams {
                            log_n: candidate.first_mask_log_n,
                            ell: mask_ell,
                            r: first.mask_tail,
                        },
                        stages,
                    });
                }
                Err(failure) => last = failure,
            },
            Err(failure) => last = failure,
        }
    }
    Err(last)
}

/// Derives one group's chain for a known mask row count.
pub fn derive_single_group_stages(
    bit_entries: u64,
    mask_rows: u64,
    rows: u64,
    samples: u32,
    circle: bool,
) -> std::result::Result<DerivedSingleGroup, StageFailure> {
    let mut last = StageFailure {
        stage: 0,
        point: "the candidate list is empty",
    };
    for candidate in production_candidates() {
        match derive_candidate_group(&candidate, bit_entries, mask_rows, 0, rows, samples, circle) {
            Ok(group) => return Ok(group),
            Err(failure) => last = failure,
        }
    }
    Err(last)
}

/// Derives one group's plan with the mask message's row count taken from its
/// entry count and the candidate's mask code.
pub fn derive_single_group_plan(
    bit_entries: u64,
    mask_entries: usize,
    rows: u64,
    samples: u32,
    circle: bool,
) -> std::result::Result<DerivedSingleGroupPlan, StageFailure> {
    derive_plan_with_extra_mask_rows(bit_entries, mask_entries, 0, rows, samples, circle)
}

/// Derives one group's plan when the group carries another mask-only member
/// whose aligned row block is `extra_mask_rows`. The returned `mask_rows` is
/// the lookup mask message's own row count; the extra block only enters the
/// fold bits.
pub fn derive_single_group_plan_with_extra_mask_rows(
    bit_entries: u64,
    mask_entries: usize,
    extra_mask_rows: u64,
    rows: u64,
    samples: u32,
    circle: bool,
) -> std::result::Result<DerivedSingleGroupPlan, StageFailure> {
    derive_plan_with_extra_mask_rows(
        bit_entries,
        mask_entries,
        extra_mask_rows,
        rows,
        samples,
        circle,
    )
}

fn derive_plan_with_extra_mask_rows(
    bit_entries: u64,
    mask_entries: usize,
    extra_mask_rows: u64,
    rows: u64,
    samples: u32,
    circle: bool,
) -> std::result::Result<DerivedSingleGroupPlan, StageFailure> {
    let mut last = StageFailure {
        stage: 0,
        point: "the candidate list is empty",
    };
    for candidate in production_candidates() {
        let mask_ell = 1u64 << candidate.first_mask_log_ell;
        let mask_rows = match smallest_mask_rows(mask_entries, mask_ell) {
            Ok(rows) => rows,
            Err(_) => {
                last = StageFailure {
                    stage: 0,
                    point: "the mask entries do not fit the candidate's mask code",
                };
                continue;
            }
        };
        match derive_candidate_group(
            &candidate,
            bit_entries,
            mask_rows,
            extra_mask_rows,
            rows,
            samples,
            circle,
        ) {
            Ok(group) => {
                return Ok(DerivedSingleGroupPlan {
                    first_main: group.first_main,
                    first_mask: group.first_mask,
                    mask_rows,
                    stages: group.stages,
                });
            }
            Err(failure) => last = failure,
        }
    }
    Err(last)
}

#[cfg(test)]
mod single_group_stage_tests {
    use super::*;
    use zkie_engine::pcs::request::code::build_code_plan;

    #[test]
    fn the_single_group_first_stage_settles() {
        let main = build_code_plan(17, 65_536, 8_192).unwrap();
        let mask = build_code_plan(17, 512, 256).unwrap();
        let stage = solve_single_first_stage(main, mask, 16_384, 4, 5, false).unwrap();
        assert_eq!(stage.k, 14);
        assert_eq!(stage.h, 2);
        assert!(stage.positions > 0);
        assert_eq!(stage.main_tail, stage.positions);
        assert_eq!(stage.mask_tail, stage.positions);
    }


    #[test]
    fn the_frozen_message_derives_a_group_chain() {
        // The frozen geometry's bit message: 2^23 entries, one mask row, and
        // every row the lookup registers on its own group (its 71 outcome
        // rows, the multiplicity row and the mask sample row).
        match derive_single_group_stages(1 << 23, 1, 73, 1, false) {
            Ok(group) => {
                assert!(!group.stages.is_empty());
                assert_eq!(group.stages[0].main_ell, group.first_main.ell);
                assert_eq!(group.stages[0].mask_ell, group.first_mask.ell);
                assert!(group.first_main.r > 0);
                assert!(group.first_mask.r > 0);
            }
            Err(failure) => panic!(
                "the frozen message must derive a chain: stage {}: {}",
                failure.stage, failure.point
            ),
        }
    }


    #[test]
    fn the_frozen_range_group_derives_its_plan() {
        use crate::proof::table_range::{
            derive_entry_range, range_lookup_plan, EntryRange, EntryRangeLimits,
        };
        let range = EntryRange::new(-131072, 131071).unwrap();
        let plan = derive_entry_range(
            1024,
            10,
            range,
            EntryRangeLimits {
                max_table_log: 19,
                max_bit_entries: 1 << 24,
            },
        )
        .unwrap();
        let lookup = range_lookup_plan(&plan).unwrap();
        let mask_entries = crate::proof::lookup::mask_count(&lookup).unwrap();
        let group =
            derive_single_group_plan(plan.bit_entries, mask_entries, 73, 1, false).unwrap();
        assert_eq!(group.stages[0].main_ell, group.first_main.ell);
        assert_eq!(group.stages[0].mask_ell, group.first_mask.ell);
        assert!(group.first_main.r > 0);
        assert!(group.mask_rows.is_power_of_two());
        assert!(group.mask_rows * group.first_mask.ell - 1 >= mask_entries as u64);
    }

    #[test]
    fn a_group_without_rows_is_refused() {
        let main = build_code_plan(17, 65_536, 8_192).unwrap();
        let mask = build_code_plan(17, 512, 256).unwrap();
        assert!(solve_single_first_stage(main, mask, 16_384, 4, 0, false).is_err());
        assert!(solve_single_first_stage(main, mask, 0, 4, 5, false).is_err());
    }

    #[test]
    fn a_code_that_cannot_hold_the_group_is_refused() {
        let main = build_code_plan(4, 8, 1).unwrap();
        let mask = build_code_plan(4, 8, 1).unwrap();
        assert!(solve_single_first_stage(main, mask, 16_384, 4, 5, false).is_err());
    }

    #[test]
    fn the_later_rule_removes_only_the_record_check() {
        let code = build_code_plan(12, 2048, 128).unwrap();
        let once = recorded_table_distance(code, 10, 1, 10, 1, 0).unwrap();
        let later = later_table_distance(code, 10, 1, 10, 1, 0).unwrap();
        assert!(once >= 3);
        assert!(
            later >= once,
            "the later rule drops only the record check"
        );
        // More sharing can only lower the largest point distance.
        let shared = later_table_distance(code, 10, 1, 10, 12, 0).unwrap();
        assert!(shared <= later);
    }

    #[test]
    fn the_production_later_candidate_fits_one_declared_batch() {
        let code = build_code_plan(12, 2048, 128).unwrap();
        let shape = SharedLinShape {
            v: 10,
            k_lo: 0,
            instances: 1,
            rows_log: 10,
        };
        let m_once = recorded_table_distance(code, 10, 1, 10, 1, 0).unwrap();
        let one_time_columns = shared_lin_columns(shape, code, m_once, 0, 0).unwrap();
        let candidate = derive_later_table_candidate(
            shape,
            code,
            1,
            10,
            m_once,
            one_time_columns,
            128,
            0,
        )
        .unwrap();
        eprintln!("later-table candidate: {:?}", candidate);
        assert_eq!(candidate.m_once, m_once);
        assert!(candidate.m_later <= candidate.m_once);
        assert!(candidate.later_columns >= 1);
        assert_eq!(candidate.spent, one_time_columns);
        assert!(candidate.total_columns_one_batch <= 128);
        assert!(candidate.capacity_batches >= 1);
    }

    #[test]
    fn the_production_later_candidate_is_derived_and_fault_checked() {
        let code = build_code_plan(12, 2048, 128).unwrap();
        let candidate = derive_production_later_candidate(code, 10, 1, 10, 10, 0).unwrap();
        eprintln!("production later-table candidate: {:?}", candidate);
        assert_eq!(candidate.m_once, 284);
        assert_eq!(candidate.m_later, 284);
        assert_eq!(candidate.one_time_columns, 52);
        assert_eq!(candidate.later_columns, 52);
        assert_eq!(candidate.total_columns_one_batch, 104);
        assert_eq!(candidate.capacity_batches, 1);
        // The ordinary later plan fault rejects a smaller column draw than the
        // derived one; the production wrapper itself runs the same fault.
        let fault = SharedLinPlanFault {
            v: 10,
            k_lo: 0,
            rows_log: 10,
            m: candidate.m_later,
            column_draws: candidate.later_columns - 1,
            point_work_bits: 0,
            column_work_bits: 0,
            zeta_work_bits: 0,
        };
        assert!(shared_lin_plan_fault(fault, code).is_err());
    }
}
