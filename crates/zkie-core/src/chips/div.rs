//! `DivChip`: I18 fixed-point division `numerator / divisor` where **both**
//! operands are witnessed I18 values (unlike `EltwiseMulChip`'s rescale gate,
//! where the divisor is the compile-time constant `SCALE_18`). This
//! generalization is needed for softmax's normalization step, which divides
//! each `exp(x_i)` by the sum of all exponentials -- a value only known at
//! proving time.
//!
//! # Precondition: `divisor > 0`
//!
//! This chip requires (and enforces in-circuit) `divisor.raw() > 0`. It is
//! intended for use where the divisor is a sum of positive lookup-table
//! outputs (e.g. softmax's sum-of-exponentials), which is always positive.
//! Division by a non-positive divisor is rejected by `assign` with a typed
//! error, since the `0 <= r < divisor` bound baked into this chip's gates
//! only has meaning for a strictly positive divisor.
//!
//! # Fixed-point division semantics
//!
//! For raw i64 numerator `n` and divisor `d` (`d > 0`), the I18-scaled
//! quotient is `result_raw = round-towards-negative-infinity(n * SCALE_18 / d)`,
//! i.e. `n * SCALE_18 == q * d + r` with `0 <= r < d` (Euclidean division).
//! Note the bound on `r` is `d`, a **witnessed** value -- not the constant
//! `SCALE_18` as in `EltwiseMulChip` -- which is the key generalization this
//! chip introduces.
//!
//! # Soundness: linking range-checked cells across regions
//!
//! `RangeCheckChip::assign` always creates its own fresh `layouter` region for
//! its bit-decomposition, so the "value" cell it range-checks lives at a
//! different absolute row than the cell used in this chip's main polynomial
//! gates, even when the same physical column is reused. Per the halo2
//! `Region` documentation ("Chips must use `Region::constrain_equal` to copy
//! in variables assigned in other regions"), those two cells are **not**
//! automatically identified -- an explicit copy constraint is required, or a
//! synthesize implementation sharing the same `configure()` could assign
//! divergent values to them and still satisfy every gate.
//!
//! This was verified empirically against the existing `EltwiseMulChip`: a
//! probe circuit sharing its `configure()` but assigning an out-of-bounds `r`
//! in the main "mul rescale" gate row while assigning a compliant, unrelated
//! `r` into the range-check sub-region satisfied `MockProver` -- i.e. the
//! `r`/`slack` (and shifted `q`) cells used by `EltwiseMulChip`'s gates are
//! not tied to the cells its range checks actually bound. Because this chip
//! is explicitly the highest-soundness-risk gadget introduced so far, it
//! deviates from that established (but gap-prone) template by explicitly
//! constraining every range-checked witness to the cell used in the main
//! gates via `region.constrain_equal`, closing that gap here. (The same
//! hardening would be worth back-porting to `EltwiseMulChip`, `ReduceMeanChip`
//! and `DotProductChip`, but that is out of scope for this change.)

use crate::chips::range_check::{RangeCheckChip, RangeCheckConfig};
use crate::field_convert::{i128_to_fr, i64_to_fr, shifted_i64_witness, Fr, SIGNED_SHIFT};
use crate::fixed_point::{FixedPointError, I18, SCALE_18};
use halo2_proofs::circuit::{AssignedCell, Layouter, Value};
use halo2_proofs::plonk::{Advice, Column, ConstraintSystem, ErrorFront, Expression, Selector};
use halo2_proofs::poly::Rotation;
use std::fmt;

/// `r` and `slack = divisor - 1 - r` are bounded by the *witnessed* divisor,
/// whose I18 raw magnitude can be as large as `i64::MAX` -- unlike
/// `EltwiseMulChip`'s remainder bound (the constant `SCALE_18`, which fits in
/// 60 bits), so both need the full unsigned 64-bit range here.
const REMAINDER_BITS: usize = 64;
/// Proves `divisor >= 1` by range-checking `divisor - 1` as an unsigned
/// value. I18's representable range tops out at `i64::MAX < 2^63`, so 63
/// bits suffices to rule out a non-positive divisor.
const DIVISOR_MINUS_ONE_BITS: usize = 63;

/// Errors that can occur while computing or assigning an I18 division.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DivError {
    /// The divisor was not strictly positive; division by it is undefined
    /// for this chip's `0 <= r < divisor` bound.
    NonPositiveDivisor { divisor: i64 },
    /// The requantized quotient overflowed I18's representable `i64` range.
    QuotientOverflow(FixedPointError),
    /// A halo2 circuit-synthesis error occurred while assigning cells.
    Circuit(String),
}

impl fmt::Display for DivError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DivError::NonPositiveDivisor { divisor } => {
                write!(f, "divisor {divisor} is not strictly positive")
            }
            DivError::QuotientOverflow(e) => write!(f, "division overflow: {e}"),
            DivError::Circuit(e) => write!(f, "div circuit error: {e}"),
        }
    }
}

impl std::error::Error for DivError {}

impl From<ErrorFront> for DivError {
    fn from(e: ErrorFront) -> Self {
        DivError::Circuit(format!("{e:?}"))
    }
}

/// Computes the I18 quotient and Euclidean remainder for `numerator / divisor`.
///
/// Precondition (checked, not assumed): `divisor.raw() > 0`. Returns a typed
/// error -- never panics -- if the divisor is non-positive, or if the
/// requantized quotient overflows I18's `i64` range.
pub fn div_quotient_remainder(numerator: I18, divisor: I18) -> Result<(I18, i128), DivError> {
    let d_raw = divisor.raw();
    if d_raw <= 0 {
        return Err(DivError::NonPositiveDivisor { divisor: d_raw });
    }
    let n_i128 = (numerator.raw() as i128) * SCALE_18;
    let d_i128 = d_raw as i128;
    let q_i128 = n_i128.div_euclid(d_i128);
    let r_i128 = n_i128.rem_euclid(d_i128);
    if q_i128 < i64::MIN as i128 || q_i128 > i64::MAX as i128 {
        return Err(DivError::QuotientOverflow(FixedPointError(format!(
            "{} / {} requantizes to a quotient that overflows I18 range",
            numerator.to_f64(),
            divisor.to_f64()
        ))));
    }
    Ok((I18::from_raw(q_i128 as i64), r_i128))
}

#[derive(Clone, Debug)]
pub struct DivConfig {
    numerator: Column<Advice>,
    divisor: Column<Advice>,
    q_shift: Column<Advice>,
    r: Column<Advice>,
    slack: Column<Advice>,
    dm1: Column<Advice>,
    s_div: Selector,
    s_slack: Selector,
    s_dm1: Selector,
    range_q: RangeCheckConfig,
    range_r: RangeCheckConfig,
    range_slack: RangeCheckConfig,
    range_dm1: RangeCheckConfig,
}

pub struct DivChip {
    config: DivConfig,
}

impl DivChip {
    /// `q_shift` stores the *shifted* quotient (`q + 2^63`), the same
    /// shift-by-`2^63` trick `EltwiseMulChip` uses to range-check a signed
    /// value with an unsigned bit-decomposition; the main "div rescale" gate
    /// subtracts the shift back out before using it, so callers only ever
    /// deal with unshifted `I18` values.
    #[allow(clippy::too_many_arguments)]
    pub fn configure(
        meta: &mut ConstraintSystem<Fr>,
        numerator: Column<Advice>,
        divisor: Column<Advice>,
        q_shift: Column<Advice>,
        r: Column<Advice>,
        slack: Column<Advice>,
        dm1: Column<Advice>,
        bits: Column<Advice>,
    ) -> DivConfig {
        meta.enable_equality(numerator);
        meta.enable_equality(divisor);
        meta.enable_equality(q_shift);
        meta.enable_equality(r);
        meta.enable_equality(slack);
        meta.enable_equality(dm1);

        // numerator * SCALE_18 == q * divisor + r, where q is recovered from
        // the shifted witness q_shift as (q_shift - 2^63).
        let s_div = meta.selector();
        meta.create_gate("div rescale", |meta| {
            let n = meta.query_advice(numerator, Rotation::cur());
            let d = meta.query_advice(divisor, Rotation::cur());
            let q_shift_expr = meta.query_advice(q_shift, Rotation::cur());
            let r_expr = meta.query_advice(r, Rotation::cur());
            let s_div = meta.query_selector(s_div);
            let scale = Expression::Constant(i128_to_fr(SCALE_18));
            let shift = Expression::Constant(i128_to_fr(SIGNED_SHIFT));
            let q_actual = q_shift_expr - shift;
            vec![s_div * (n * scale - q_actual * d - r_expr)]
        });

        // slack == divisor - 1 - r, i.e. slack + r - divisor + 1 == 0.
        // Unlike EltwiseMulChip's slack gate (subtracting the CONSTANT
        // SCALE_18 - 1), here `divisor` is a witnessed column value, so the
        // gate keeps `divisor` as a linear term instead of folding `- 1`
        // into a precomputed constant.
        let s_slack = meta.selector();
        meta.create_gate(
            "div slack equals divisor minus one minus remainder",
            |meta| {
                let d = meta.query_advice(divisor, Rotation::cur());
                let r_expr = meta.query_advice(r, Rotation::cur());
                let slack_expr = meta.query_advice(slack, Rotation::cur());
                let s_slack = meta.query_selector(s_slack);
                let one = Expression::Constant(Fr::one());
                vec![s_slack * (slack_expr + r_expr - d + one)]
            },
        );

        // dm1 == divisor - 1, i.e. dm1 - divisor + 1 == 0. Range-checking
        // dm1 as an unsigned DIVISOR_MINUS_ONE_BITS-bit value (below) proves
        // divisor >= 1, i.e. divisor > 0.
        let s_dm1 = meta.selector();
        meta.create_gate("divisor_minus_one equals divisor minus one", |meta| {
            let d = meta.query_advice(divisor, Rotation::cur());
            let dm1_expr = meta.query_advice(dm1, Rotation::cur());
            let s_dm1 = meta.query_selector(s_dm1);
            let one = Expression::Constant(Fr::one());
            vec![s_dm1 * (dm1_expr - d + one)]
        });

        let range_q = RangeCheckChip::configure(meta, q_shift, bits, 64);
        let range_r = RangeCheckChip::configure(meta, r, bits, REMAINDER_BITS);
        let range_slack = RangeCheckChip::configure(meta, slack, bits, REMAINDER_BITS);
        let range_dm1 = RangeCheckChip::configure(meta, dm1, bits, DIVISOR_MINUS_ONE_BITS);

        DivConfig {
            numerator,
            divisor,
            q_shift,
            r,
            slack,
            dm1,
            s_div,
            s_slack,
            s_dm1,
            range_q,
            range_r,
            range_slack,
            range_dm1,
        }
    }

    pub fn construct(config: DivConfig) -> Self {
        DivChip { config }
    }

    /// Computes `numerator / divisor` as I18 fixed-point division, witnesses
    /// the quotient/remainder/slack/`divisor - 1` gadget, range-checks each
    /// of them, and links every range-checked cell back to the cell used in
    /// the main gates via `region.constrain_equal` (see module docs for why
    /// this differs from the sibling chips' pattern). Returns the quotient
    /// as an `I18`, along with the `AssignedCell`s holding the `numerator`
    /// and `divisor` witnessed here -- composing chips (e.g. `SoftmaxChip`,
    /// whose divisor is a `ReduceSumChip` sum) must `region.constrain_equal`
    /// these back to the cell holding the same value elsewhere, rather than
    /// leave this chip's copy disconnected from it -- or a typed `DivError`
    /// -- never panics -- if `divisor` is not strictly positive or the
    /// quotient overflows I18's range.
    #[allow(clippy::type_complexity)]
    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fr>,
        numerator: I18,
        divisor: I18,
    ) -> Result<(I18, AssignedCell<Fr, Fr>, AssignedCell<Fr, Fr>), DivError> {
        let (quotient, r_i128) = div_quotient_remainder(numerator, divisor)?;
        let d_i128 = divisor.raw() as i128;
        let slack = d_i128 - 1 - r_i128;
        let dm1 = d_i128 - 1;
        let (q_shift_fr, q_shift_raw) = shifted_i64_witness(quotient.raw());

        let (numerator_cell, divisor_cell, q_shift_cell, r_cell, slack_cell, dm1_cell) = layouter
            .assign_region(
            || "div core",
            |mut region| {
                self.config.s_div.enable(&mut region, 0)?;
                self.config.s_slack.enable(&mut region, 0)?;
                self.config.s_dm1.enable(&mut region, 0)?;
                let numerator_cell = region.assign_advice(
                    || "numerator",
                    self.config.numerator,
                    0,
                    || Value::known(i64_to_fr(numerator.raw())),
                )?;
                let divisor_cell = region.assign_advice(
                    || "divisor",
                    self.config.divisor,
                    0,
                    || Value::known(i64_to_fr(divisor.raw())),
                )?;
                let q_shift_cell =
                    region.assign_advice(|| "q_shift", self.config.q_shift, 0, || q_shift_fr)?;
                let r_cell = region.assign_advice(
                    || "r",
                    self.config.r,
                    0,
                    || Value::known(i128_to_fr(r_i128)),
                )?;
                let slack_cell = region.assign_advice(
                    || "slack",
                    self.config.slack,
                    0,
                    || Value::known(i128_to_fr(slack)),
                )?;
                let dm1_cell = region.assign_advice(
                    || "divisor_minus_one",
                    self.config.dm1,
                    0,
                    || Value::known(i128_to_fr(dm1)),
                )?;
                Ok((
                    numerator_cell,
                    divisor_cell,
                    q_shift_cell,
                    r_cell,
                    slack_cell,
                    dm1_cell,
                ))
            },
        )?;

        let range_q_chip = RangeCheckChip::construct(self.config.range_q.clone());
        let q_shift_range_cell = range_q_chip.assign(
            layouter.namespace(|| "range q_shift"),
            q_shift_fr,
            q_shift_raw,
        )?;

        let range_r_chip = RangeCheckChip::construct(self.config.range_r.clone());
        let r_range_cell = range_r_chip.assign(
            layouter.namespace(|| "range r"),
            Value::known(i128_to_fr(r_i128)),
            Value::known(r_i128),
        )?;

        let range_slack_chip = RangeCheckChip::construct(self.config.range_slack.clone());
        let slack_range_cell = range_slack_chip.assign(
            layouter.namespace(|| "range slack"),
            Value::known(i128_to_fr(slack)),
            Value::known(slack),
        )?;

        let range_dm1_chip = RangeCheckChip::construct(self.config.range_dm1.clone());
        let dm1_range_cell = range_dm1_chip.assign(
            layouter.namespace(|| "range divisor_minus_one"),
            Value::known(i128_to_fr(dm1)),
            Value::known(dm1),
        )?;

        // Tie each range-checked "value" cell (assigned in its own
        // RangeCheckChip region above) back to the corresponding cell used
        // in the "div core" gates -- see module docs.
        layouter.assign_region(
            || "div range check links",
            |mut region| {
                region.constrain_equal(q_shift_cell.cell(), q_shift_range_cell.cell())?;
                region.constrain_equal(r_cell.cell(), r_range_cell.cell())?;
                region.constrain_equal(slack_cell.cell(), slack_range_cell.cell())?;
                region.constrain_equal(dm1_cell.cell(), dm1_range_cell.cell())?;
                Ok(())
            },
        )?;

        Ok((quotient, numerator_cell, divisor_cell))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::circuit::SimpleFloorPlanner;
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::plonk::{Circuit, ConstraintSystem, ErrorFront};

    #[derive(Clone)]
    struct DivTestConfig {
        div: DivConfig,
    }

    struct DivTestCircuit {
        numerator: I18,
        divisor: I18,
    }

    impl Circuit<Fr> for DivTestCircuit {
        type Config = DivTestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            DivTestCircuit {
                numerator: I18::from_raw(0),
                divisor: I18::from_raw(1),
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let numerator = meta.advice_column();
            let divisor = meta.advice_column();
            let q_shift = meta.advice_column();
            let r = meta.advice_column();
            let slack = meta.advice_column();
            let dm1 = meta.advice_column();
            let bits = meta.advice_column();
            DivTestConfig {
                div: DivChip::configure(meta, numerator, divisor, q_shift, r, slack, dm1, bits),
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = DivChip::construct(config.div);
            chip.assign(layouter, self.numerator, self.divisor)
                .map(|_| ())
                .map_err(|e| match e {
                    DivError::Circuit(_) => ErrorFront::Synthesis,
                    _ => panic!("unexpected DivError in test synthesize: {e}"),
                })
        }
    }

    #[test]
    fn five_divided_by_two_is_exact_two_point_five() {
        // I18's representable range tops out at i64::MAX / SCALE_18 ~= 9.22,
        // so 10.0/4.0 (the task's original example ratio) doesn't fit; 5.0/2.0
        // is the same 2.5 ratio, exact in I18, and representable.
        let numerator = I18::from_f64(5.0).unwrap();
        let divisor = I18::from_f64(2.0).unwrap();
        let (quotient, remainder) = div_quotient_remainder(numerator, divisor).unwrap();
        assert!((quotient.to_f64() - 2.5).abs() < 1e-9);
        assert_eq!(remainder, 0);

        let circuit = DivTestCircuit { numerator, divisor };
        let prover = MockProver::run(11, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn seven_divided_by_three_matches_host_requantize_within_tolerance() {
        let numerator = I18::from_f64(7.0).unwrap();
        let divisor = I18::from_f64(3.0).unwrap();

        // Host-side "requantize"-style expectation: (7 * SCALE_18) / 3,
        // Euclidean division, matching this chip's own semantics exactly
        // (not just an approximate float comparison).
        let n_i128 = (numerator.raw() as i128) * SCALE_18;
        let d_i128 = divisor.raw() as i128;
        let expected_q = n_i128.div_euclid(d_i128);
        let expected_r = n_i128.rem_euclid(d_i128);

        let (quotient, remainder) = div_quotient_remainder(numerator, divisor).unwrap();
        assert_eq!(quotient.raw() as i128, expected_q);
        assert_eq!(remainder, expected_r);
        assert!((0..d_i128).contains(&remainder));
        // 7/3 ~= 2.3333...; fixed-point quantization tolerance should be
        // well within 1 part in SCALE_18.
        assert!((quotient.to_f64() - 7.0 / 3.0).abs() < 1e-9);

        let circuit = DivTestCircuit { numerator, divisor };
        let prover = MockProver::run(11, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn negative_divisor_is_rejected_with_typed_error_not_panic() {
        let numerator = I18::from_f64(1.0).unwrap();
        let divisor = I18::from_f64(-2.0).unwrap();
        let result = div_quotient_remainder(numerator, divisor);
        assert_eq!(
            result,
            Err(DivError::NonPositiveDivisor {
                divisor: divisor.raw()
            })
        );
    }

    #[test]
    fn zero_divisor_is_rejected_with_typed_error_not_panic() {
        let numerator = I18::from_f64(1.0).unwrap();
        let divisor = I18::from_raw(0);
        let result = div_quotient_remainder(numerator, divisor);
        assert_eq!(result, Err(DivError::NonPositiveDivisor { divisor: 0 }));
    }

    #[test]
    fn div_with_forged_quotient_is_rejected() {
        struct ForgedDivCircuit {
            numerator: I18,
            divisor: I18,
        }

        impl Circuit<Fr> for ForgedDivCircuit {
            type Config = DivTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                ForgedDivCircuit {
                    numerator: I18::from_raw(0),
                    divisor: I18::from_raw(1),
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                DivTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let (quotient, r_i128) =
                    div_quotient_remainder(self.numerator, self.divisor).unwrap();
                let d_i128 = self.divisor.raw() as i128;
                let forged_q = quotient.raw() + 1; // violates n*SCALE_18 == q*divisor + r
                let slack = d_i128 - 1 - r_i128;
                let dm1 = d_i128 - 1;
                let (q_shift_fr, q_shift_raw) = shifted_i64_witness(forged_q);

                let (q_shift_cell, r_cell, slack_cell, dm1_cell) = layouter.assign_region(
                    || "forged div core",
                    |mut region| {
                        config.div.s_div.enable(&mut region, 0)?;
                        config.div.s_slack.enable(&mut region, 0)?;
                        config.div.s_dm1.enable(&mut region, 0)?;
                        region.assign_advice(
                            || "numerator",
                            config.div.numerator,
                            0,
                            || Value::known(i64_to_fr(self.numerator.raw())),
                        )?;
                        region.assign_advice(
                            || "divisor",
                            config.div.divisor,
                            0,
                            || Value::known(i64_to_fr(self.divisor.raw())),
                        )?;
                        let q_shift_cell = region.assign_advice(
                            || "q_shift",
                            config.div.q_shift,
                            0,
                            || q_shift_fr,
                        )?;
                        let r_cell = region.assign_advice(
                            || "r",
                            config.div.r,
                            0,
                            || Value::known(i128_to_fr(r_i128)),
                        )?;
                        let slack_cell = region.assign_advice(
                            || "slack",
                            config.div.slack,
                            0,
                            || Value::known(i128_to_fr(slack)),
                        )?;
                        let dm1_cell = region.assign_advice(
                            || "divisor_minus_one",
                            config.div.dm1,
                            0,
                            || Value::known(i128_to_fr(dm1)),
                        )?;
                        Ok((q_shift_cell, r_cell, slack_cell, dm1_cell))
                    },
                )?;

                let range_q_chip = RangeCheckChip::construct(config.div.range_q.clone());
                let q_shift_range_cell = range_q_chip.assign(
                    layouter.namespace(|| "range q_shift"),
                    q_shift_fr,
                    q_shift_raw,
                )?;
                let range_r_chip = RangeCheckChip::construct(config.div.range_r.clone());
                let r_range_cell = range_r_chip.assign(
                    layouter.namespace(|| "range r"),
                    Value::known(i128_to_fr(r_i128)),
                    Value::known(r_i128),
                )?;
                let range_slack_chip = RangeCheckChip::construct(config.div.range_slack.clone());
                let slack_range_cell = range_slack_chip.assign(
                    layouter.namespace(|| "range slack"),
                    Value::known(i128_to_fr(slack)),
                    Value::known(slack),
                )?;
                let range_dm1_chip = RangeCheckChip::construct(config.div.range_dm1.clone());
                let dm1_range_cell = range_dm1_chip.assign(
                    layouter.namespace(|| "range divisor_minus_one"),
                    Value::known(i128_to_fr(dm1)),
                    Value::known(dm1),
                )?;

                layouter.assign_region(
                    || "forged div range check links",
                    |mut region| {
                        region.constrain_equal(q_shift_cell.cell(), q_shift_range_cell.cell())?;
                        region.constrain_equal(r_cell.cell(), r_range_cell.cell())?;
                        region.constrain_equal(slack_cell.cell(), slack_range_cell.cell())?;
                        region.constrain_equal(dm1_cell.cell(), dm1_range_cell.cell())?;
                        Ok(())
                    },
                )?;
                Ok(())
            }
        }

        let circuit = ForgedDivCircuit {
            numerator: I18::from_f64(5.0).unwrap(),
            divisor: I18::from_f64(2.0).unwrap(),
        };
        let prover = MockProver::run(11, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    fn div_with_remainder_forged_to_violate_r_less_than_divisor_is_rejected() {
        // Directly witness r >= divisor (bypassing `assign`'s honest
        // computation), keeping q and the main "div rescale" polynomial
        // satisfied by compensating q, but forging slack to be negative
        // (which must fail its own range check since slack cannot be
        // represented as a small nonnegative value).
        struct ForgedRemainderCircuit {
            numerator: I18,
            divisor: I18,
        }

        impl Circuit<Fr> for ForgedRemainderCircuit {
            type Config = DivTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                ForgedRemainderCircuit {
                    numerator: I18::from_raw(0),
                    divisor: I18::from_raw(1),
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                DivTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let (quotient, r_i128) =
                    div_quotient_remainder(self.numerator, self.divisor).unwrap();
                let d_i128 = self.divisor.raw() as i128;

                // Forge r' = r + divisor (still >= 0, but >= divisor,
                // violating the intended 0 <= r < divisor bound), and
                // compensate q' = q - 1 so the main div-rescale polynomial
                // n*SCALE_18 == q*divisor + r still holds exactly.
                let forged_r = r_i128 + d_i128;
                let forged_q = quotient.raw() - 1;
                let forged_slack = d_i128 - 1 - forged_r; // now negative
                let dm1 = d_i128 - 1;
                let (q_shift_fr, q_shift_raw) = shifted_i64_witness(forged_q);

                let (q_shift_cell, r_cell, slack_cell, dm1_cell) = layouter.assign_region(
                    || "forged remainder div core",
                    |mut region| {
                        config.div.s_div.enable(&mut region, 0)?;
                        config.div.s_slack.enable(&mut region, 0)?;
                        config.div.s_dm1.enable(&mut region, 0)?;
                        region.assign_advice(
                            || "numerator",
                            config.div.numerator,
                            0,
                            || Value::known(i64_to_fr(self.numerator.raw())),
                        )?;
                        region.assign_advice(
                            || "divisor",
                            config.div.divisor,
                            0,
                            || Value::known(i64_to_fr(self.divisor.raw())),
                        )?;
                        let q_shift_cell = region.assign_advice(
                            || "q_shift",
                            config.div.q_shift,
                            0,
                            || q_shift_fr,
                        )?;
                        let r_cell = region.assign_advice(
                            || "r",
                            config.div.r,
                            0,
                            || Value::known(i128_to_fr(forged_r)),
                        )?;
                        let slack_cell = region.assign_advice(
                            || "slack",
                            config.div.slack,
                            0,
                            || Value::known(i128_to_fr(forged_slack)),
                        )?;
                        let dm1_cell = region.assign_advice(
                            || "divisor_minus_one",
                            config.div.dm1,
                            0,
                            || Value::known(i128_to_fr(dm1)),
                        )?;
                        Ok((q_shift_cell, r_cell, slack_cell, dm1_cell))
                    },
                )?;

                let range_q_chip = RangeCheckChip::construct(config.div.range_q.clone());
                let q_shift_range_cell = range_q_chip.assign(
                    layouter.namespace(|| "range q_shift"),
                    q_shift_fr,
                    q_shift_raw,
                )?;
                let range_r_chip = RangeCheckChip::construct(config.div.range_r.clone());
                let r_range_cell = range_r_chip.assign(
                    layouter.namespace(|| "range r"),
                    Value::known(i128_to_fr(forged_r)),
                    Value::known(forged_r),
                )?;
                let range_slack_chip = RangeCheckChip::construct(config.div.range_slack.clone());
                let slack_range_cell = range_slack_chip.assign(
                    layouter.namespace(|| "range slack"),
                    Value::known(i128_to_fr(forged_slack)),
                    Value::known(forged_slack),
                )?;
                let range_dm1_chip = RangeCheckChip::construct(config.div.range_dm1.clone());
                let dm1_range_cell = range_dm1_chip.assign(
                    layouter.namespace(|| "range divisor_minus_one"),
                    Value::known(i128_to_fr(dm1)),
                    Value::known(dm1),
                )?;

                layouter.assign_region(
                    || "forged remainder div range check links",
                    |mut region| {
                        region.constrain_equal(q_shift_cell.cell(), q_shift_range_cell.cell())?;
                        region.constrain_equal(r_cell.cell(), r_range_cell.cell())?;
                        region.constrain_equal(slack_cell.cell(), slack_range_cell.cell())?;
                        region.constrain_equal(dm1_cell.cell(), dm1_range_cell.cell())?;
                        Ok(())
                    },
                )?;
                Ok(())
            }
        }

        let circuit = ForgedRemainderCircuit {
            numerator: I18::from_f64(5.0).unwrap(),
            divisor: I18::from_f64(2.0).unwrap(),
        };
        let prover = MockProver::run(11, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    fn div_with_nonpositive_divisor_forged_in_circuit_is_rejected() {
        // Bypass `assign`'s Rust-level `divisor > 0` precondition check
        // entirely: directly witness divisor = 0 (with n = 0 so the main
        // div-rescale polynomial 0*SCALE_18 == q*0 + 0 is trivially
        // satisfiable with q = r = 0), and confirm the divisor_minus_one
        // range check (proving divisor >= 1) rejects it.
        struct ForgedZeroDivisorCircuit;

        impl Circuit<Fr> for ForgedZeroDivisorCircuit {
            type Config = DivTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                ForgedZeroDivisorCircuit
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                DivTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let dm1 = -1i128; // divisor - 1 where divisor = 0

                let (dm1_cell,) = layouter.assign_region(
                    || "forged zero divisor core",
                    |mut region| {
                        config.div.s_div.enable(&mut region, 0)?;
                        config.div.s_slack.enable(&mut region, 0)?;
                        config.div.s_dm1.enable(&mut region, 0)?;
                        region.assign_advice(
                            || "numerator",
                            config.div.numerator,
                            0,
                            || Value::known(i64_to_fr(0)),
                        )?;
                        region.assign_advice(
                            || "divisor",
                            config.div.divisor,
                            0,
                            || Value::known(i64_to_fr(0)),
                        )?;
                        let (q_shift_fr, _) = shifted_i64_witness(0);
                        region.assign_advice(|| "q_shift", config.div.q_shift, 0, || q_shift_fr)?;
                        region.assign_advice(
                            || "r",
                            config.div.r,
                            0,
                            || Value::known(i128_to_fr(0)),
                        )?;
                        // slack = divisor - 1 - r = 0 - 1 - 0 = -1, matching s_slack.
                        region.assign_advice(
                            || "slack",
                            config.div.slack,
                            0,
                            || Value::known(i128_to_fr(-1)),
                        )?;
                        let dm1_cell = region.assign_advice(
                            || "divisor_minus_one",
                            config.div.dm1,
                            0,
                            || Value::known(i128_to_fr(dm1)),
                        )?;
                        Ok((dm1_cell,))
                    },
                )?;

                let range_dm1_chip = RangeCheckChip::construct(config.div.range_dm1.clone());
                let dm1_range_cell = range_dm1_chip.assign(
                    layouter.namespace(|| "range divisor_minus_one"),
                    Value::known(i128_to_fr(dm1)),
                    Value::known(dm1),
                )?;

                layouter.assign_region(
                    || "forged zero divisor link",
                    |mut region| region.constrain_equal(dm1_cell.cell(), dm1_range_cell.cell()),
                )?;
                Ok(())
            }
        }

        let circuit = ForgedZeroDivisorCircuit;
        let prover = MockProver::run(11, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    fn div_cross_region_forgery_is_rejected_by_explicit_link() {
        // Probes the exact gap found (and documented) in EltwiseMulChip: a
        // synthesize sharing the same `configure()` assigns an
        // out-of-bounds r/slack in the main "div core" row (compensating q
        // so the main polynomial still holds), while assigning a
        // compliant, disconnected r/slack into the range-check
        // sub-regions -- but this time DOES call the explicit
        // `constrain_equal` links this chip adds. Since the linked cells
        // must match, MockProver must reject.
        struct CrossRegionForgeCircuit {
            numerator: I18,
            divisor: I18,
        }

        impl Circuit<Fr> for CrossRegionForgeCircuit {
            type Config = DivTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                CrossRegionForgeCircuit {
                    numerator: I18::from_raw(0),
                    divisor: I18::from_raw(1),
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                DivTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let (quotient, r_i128) =
                    div_quotient_remainder(self.numerator, self.divisor).unwrap();
                let d_i128 = self.divisor.raw() as i128;

                // Main-row values: out-of-range r' (compensated by q'), the
                // same attack as the forged-remainder test above.
                let forged_r = r_i128 + d_i128;
                let forged_q = quotient.raw() - 1;
                let forged_slack = d_i128 - 1 - forged_r;
                let dm1 = d_i128 - 1;
                let (forged_q_shift_fr, _) = shifted_i64_witness(forged_q);

                let (q_shift_cell, r_cell, slack_cell, dm1_cell) = layouter.assign_region(
                    || "cross region forge div core",
                    |mut region| {
                        config.div.s_div.enable(&mut region, 0)?;
                        config.div.s_slack.enable(&mut region, 0)?;
                        config.div.s_dm1.enable(&mut region, 0)?;
                        region.assign_advice(
                            || "numerator",
                            config.div.numerator,
                            0,
                            || Value::known(i64_to_fr(self.numerator.raw())),
                        )?;
                        region.assign_advice(
                            || "divisor",
                            config.div.divisor,
                            0,
                            || Value::known(i64_to_fr(self.divisor.raw())),
                        )?;
                        let q_shift_cell = region.assign_advice(
                            || "q_shift",
                            config.div.q_shift,
                            0,
                            || forged_q_shift_fr,
                        )?;
                        let r_cell = region.assign_advice(
                            || "r",
                            config.div.r,
                            0,
                            || Value::known(i128_to_fr(forged_r)),
                        )?;
                        let slack_cell = region.assign_advice(
                            || "slack",
                            config.div.slack,
                            0,
                            || Value::known(i128_to_fr(forged_slack)),
                        )?;
                        let dm1_cell = region.assign_advice(
                            || "divisor_minus_one",
                            config.div.dm1,
                            0,
                            || Value::known(i128_to_fr(dm1)),
                        )?;
                        Ok((q_shift_cell, r_cell, slack_cell, dm1_cell))
                    },
                )?;

                // Range-check sub-regions get the ORIGINAL, honest, in-range
                // q/r/slack -- deliberately disconnected in VALUE from the
                // forged main row above.
                let (honest_q_shift_fr, honest_q_shift_raw) = shifted_i64_witness(quotient.raw());
                let range_q_chip = RangeCheckChip::construct(config.div.range_q.clone());
                let q_shift_range_cell = range_q_chip.assign(
                    layouter.namespace(|| "range q_shift"),
                    honest_q_shift_fr,
                    honest_q_shift_raw,
                )?;
                let range_r_chip = RangeCheckChip::construct(config.div.range_r.clone());
                let r_range_cell = range_r_chip.assign(
                    layouter.namespace(|| "range r"),
                    Value::known(i128_to_fr(r_i128)),
                    Value::known(r_i128),
                )?;
                let honest_slack = d_i128 - 1 - r_i128;
                let range_slack_chip = RangeCheckChip::construct(config.div.range_slack.clone());
                let slack_range_cell = range_slack_chip.assign(
                    layouter.namespace(|| "range slack"),
                    Value::known(i128_to_fr(honest_slack)),
                    Value::known(honest_slack),
                )?;
                let range_dm1_chip = RangeCheckChip::construct(config.div.range_dm1.clone());
                let dm1_range_cell = range_dm1_chip.assign(
                    layouter.namespace(|| "range divisor_minus_one"),
                    Value::known(i128_to_fr(dm1)),
                    Value::known(dm1),
                )?;

                // The links this chip adds: since the main-row cells and
                // range-checked cells now hold DIFFERENT values, these
                // copy constraints must fail.
                layouter.assign_region(
                    || "cross region forge links",
                    |mut region| {
                        region.constrain_equal(q_shift_cell.cell(), q_shift_range_cell.cell())?;
                        region.constrain_equal(r_cell.cell(), r_range_cell.cell())?;
                        region.constrain_equal(slack_cell.cell(), slack_range_cell.cell())?;
                        region.constrain_equal(dm1_cell.cell(), dm1_range_cell.cell())?;
                        Ok(())
                    },
                )?;
                Ok(())
            }
        }

        let circuit = CrossRegionForgeCircuit {
            numerator: I18::from_f64(5.0).unwrap(),
            divisor: I18::from_f64(2.0).unwrap(),
        };
        let prover = MockProver::run(11, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }
}
