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

fn sized_step(
    shape: &GroupShape,
    main_draws: u64,
    mask_draws: u64,
    rows: u64,
    circle: bool,
    extra: &Bound,
) -> std::result::Result<(u64, bool), &'static str> {
    let main_tail = stage_tail(main_draws, circle);
    let mask_tail = stage_tail(mask_draws, circle);
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
