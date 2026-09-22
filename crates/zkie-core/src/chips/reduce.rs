use crate::chips::range_check::{RangeCheckChip, RangeCheckConfig};
use crate::field_convert::{i128_to_fr, i64_to_fr, shifted_i64_witness, Fr, SIGNED_SHIFT};
use crate::fixed_point::{requantize_mul, I18, SCALE_18};
use halo2_proofs::circuit::{AssignedCell, Layouter, Value};
use halo2_proofs::plonk::{Advice, Column, ConstraintSystem, ErrorFront, Expression, Selector};
use halo2_proofs::poly::Rotation;

const REMAINDER_BITS: usize = 60; // 2^60 > SCALE_18 - 1, matches eltwise.rs's mul gadget.

// See the soundness note in chips/eltwise.rs: any column that needs both to
// participate in this chip's own gates and to be range-checked must have its
// range-checked (possibly signed-shifted) representation copy-constrained
// back to the cell used in the gate via `region.constrain_equal` -- a
// `RangeCheckChip::assign` call on its own only witnesses a fresh,
// disconnected cell in its own region.

/// Running-sum reduction over a fixed, compile-time-known count of I18
/// inputs (`k`, provided to `configure`). I18 + I18 needs no fixed-point
/// rescale (only multiplication changes scale), so this is just `k - 1`
/// chained additions with coefficient 1, in the same spirit as
/// `RangeCheckChip`'s running-sum gate but summing raw values instead of
/// bit-weighted powers of two.
#[derive(Clone, Debug)]
pub struct ReduceSumConfig {
    values: Column<Advice>,
    sum: Column<Advice>,
    sum_shift: Column<Advice>,
    s_first: Selector,
    s_running: Selector,
    s_shift: Selector,
    range_sum: RangeCheckConfig,
    k: usize,
}

pub struct ReduceSumChip {
    config: ReduceSumConfig,
}

impl ReduceSumChip {
    pub fn configure(
        meta: &mut ConstraintSystem<Fr>,
        values: Column<Advice>,
        sum: Column<Advice>,
        sum_shift: Column<Advice>,
        bits: Column<Advice>,
        k: usize,
    ) -> ReduceSumConfig {
        assert!(k >= 1, "ReduceSumChip requires at least one input");
        meta.enable_equality(values);
        meta.enable_equality(sum);
        meta.enable_equality(sum_shift);

        // Row 0: sum == values (base case of the running total).
        let s_first = meta.selector();
        meta.create_gate("reduce sum base case", |meta| {
            let value = meta.query_advice(values, Rotation::cur());
            let sum = meta.query_advice(sum, Rotation::cur());
            let s_first = meta.query_selector(s_first);
            vec![s_first * (sum - value)]
        });

        // Row i > 0: sum[i] == sum[i - 1] + values[i].
        let s_running = meta.selector();
        meta.create_gate("reduce sum running total", |meta| {
            let value = meta.query_advice(values, Rotation::cur());
            let sum_prev = meta.query_advice(sum, Rotation::prev());
            let sum_cur = meta.query_advice(sum, Rotation::cur());
            let s_running = meta.query_selector(s_running);
            vec![s_running * (sum_prev + value - sum_cur)]
        });

        // At the final row (k - 1): sum_shift == sum + 2^63, so the exact
        // cell fed to the range check (sum_shift) is tied, via this gate, to
        // the same `sum` cell used by the accumulation gates above -- and
        // `constrain_equal` (in `assign`) ties the range check's own cell
        // back to this `sum_shift` cell.
        let s_shift = meta.selector();
        meta.create_gate("reduce sum shift for range check", |meta| {
            let sum = meta.query_advice(sum, Rotation::cur());
            let sum_shift = meta.query_advice(sum_shift, Rotation::cur());
            let s_shift = meta.query_selector(s_shift);
            let shift = Expression::Constant(i128_to_fr(SIGNED_SHIFT));
            vec![s_shift * (sum_shift - sum - shift)]
        });

        let range_sum = RangeCheckChip::configure(meta, sum_shift, bits, 64);

        ReduceSumConfig {
            values,
            sum,
            sum_shift,
            s_first,
            s_running,
            s_shift,
            range_sum,
            k,
        }
    }

    pub fn construct(config: ReduceSumConfig) -> Self {
        ReduceSumChip { config }
    }

    /// Assigns the running-sum region for `inputs` (must have length `k`,
    /// the count fixed at `configure` time), range-checks the final sum as
    /// a signed 64-bit value, and returns the sum as an `I18`, the
    /// `AssignedCell` holding the (raw, unshifted) final sum, and the
    /// per-input `AssignedCell`s (in `inputs` order) holding this region's
    /// own witnessed copies of each input value -- composing chips (e.g.
    /// `ReduceMeanChip` for the sum cell, `SoftmaxChip` for both) must
    /// `region.constrain_equal` these cells to any cell where they
    /// re-witness the same value, rather than re-assigning it disconnected
    /// from this one.
    #[allow(clippy::type_complexity)]
    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fr>,
        inputs: &[I18],
    ) -> Result<(I18, AssignedCell<Fr, Fr>, Vec<AssignedCell<Fr, Fr>>), ErrorFront> {
        assert_eq!(
            inputs.len(),
            self.config.k,
            "ReduceSumChip configured for {} inputs, got {}",
            self.config.k,
            inputs.len()
        );

        let mut partial_sums: Vec<i64> = Vec::with_capacity(inputs.len());
        for (i, v) in inputs.iter().enumerate() {
            let next = if i == 0 {
                v.raw()
            } else {
                partial_sums[i - 1]
                    .checked_add(v.raw())
                    .expect("I18 reduce sum overflow")
            };
            partial_sums.push(next);
        }
        let sum_raw = *partial_sums
            .last()
            .expect("k >= 1 guarantees a last element");
        let (sum_shift_fr, sum_shift_raw) = shifted_i64_witness(sum_raw);

        let (sum_cell, sum_shift_cell, value_cells) = layouter.assign_region(
            || "reduce sum",
            |mut region| {
                let mut last_sum_cell = None;
                let mut value_cells = Vec::with_capacity(inputs.len());
                for (i, (v, s)) in inputs.iter().zip(partial_sums.iter()).enumerate() {
                    let value_cell = region.assign_advice(
                        || format!("value {i}"),
                        self.config.values,
                        i,
                        || Value::known(i64_to_fr(v.raw())),
                    )?;
                    value_cells.push(value_cell);
                    if i == 0 {
                        self.config.s_first.enable(&mut region, 0)?;
                    } else {
                        self.config.s_running.enable(&mut region, i)?;
                    }
                    let cell = region.assign_advice(
                        || format!("sum {i}"),
                        self.config.sum,
                        i,
                        || Value::known(i64_to_fr(*s)),
                    )?;
                    last_sum_cell = Some(cell);
                }
                let last_row = inputs.len() - 1;
                self.config.s_shift.enable(&mut region, last_row)?;
                let sum_shift_cell = region.assign_advice(
                    || "sum shift",
                    self.config.sum_shift,
                    last_row,
                    || sum_shift_fr,
                )?;
                Ok((last_sum_cell.expect("k >= 1"), sum_shift_cell, value_cells))
            },
        )?;

        let range_sum_chip = RangeCheckChip::construct(self.config.range_sum.clone());
        let range_cell = range_sum_chip.assign(
            layouter.namespace(|| "range sum"),
            sum_shift_fr,
            sum_shift_raw,
        )?;

        layouter.assign_region(
            || "reduce sum range check link",
            |mut region| region.constrain_equal(sum_shift_cell.cell(), range_cell.cell()),
        )?;

        Ok((I18::from_raw(sum_raw), sum_cell, value_cells))
    }
}

/// Mean reduction over `k` I18 inputs: computes the running sum (via
/// `ReduceSumChip`) and then rescales it by a compile-time-known reciprocal
/// constant `1/k` (an `I18` computed once at `configure` time), using the
/// same quotient/remainder rescale gadget as `EltwiseMulChip`:
/// `sum * (1/k) == q * SCALE_18 + r`, with `q` range-checked as a signed
/// 64-bit value and `r`/`slack = SCALE_18 - 1 - r` each range-checked into
/// `REMAINDER_BITS` to pin `r` into `[0, SCALE_18)`. Unlike `EltwiseMulChip`,
/// the second multiplicand (`1/k`) is a constant baked into the gate rather
/// than a witnessed column, since it is fixed once `k` is known.
///
/// NOTE: multiplying by a precomputed reciprocal instead of performing exact
/// division introduces the usual fixed-point quantization error (e.g. 1/3
/// is not exactly representable in I18) — expected and acceptable at this
/// foundation layer.
#[derive(Clone, Debug)]
pub struct ReduceMeanConfig {
    sum: ReduceSumConfig,
    q: Column<Advice>,
    r: Column<Advice>,
    slack: Column<Advice>,
    s_rescale: Selector,
    s_slack: Selector,
    range_q: RangeCheckConfig,
    range_r: RangeCheckConfig,
    range_r_slack: RangeCheckConfig,
    reciprocal: I18,
}

pub struct ReduceMeanChip {
    config: ReduceMeanConfig,
}

impl ReduceMeanChip {
    #[allow(clippy::too_many_arguments)]
    pub fn configure(
        meta: &mut ConstraintSystem<Fr>,
        values: Column<Advice>,
        sum: Column<Advice>,
        sum_shift: Column<Advice>,
        q: Column<Advice>,
        r: Column<Advice>,
        slack: Column<Advice>,
        bits: Column<Advice>,
        k: usize,
    ) -> ReduceMeanConfig {
        let sum_config = ReduceSumChip::configure(meta, values, sum, sum_shift, bits, k);

        meta.enable_equality(q);
        meta.enable_equality(r);
        meta.enable_equality(slack);

        let reciprocal = I18::from_f64(1.0 / (k as f64))
            .expect("1/k must be representable as an I18 fixed-point value");
        let reciprocal_fr = i64_to_fr(reciprocal.raw());

        // `q` holds the *shifted* (q_raw + 2^63) representation (see the
        // soundness note at the top of this file and in chips/eltwise.rs),
        // so the rescale gate is written in terms of the shifted quotient.
        let s_rescale = meta.selector();
        meta.create_gate("mean rescale (shifted q)", |meta| {
            let sum = meta.query_advice(sum, Rotation::cur());
            let q_shift = meta.query_advice(q, Rotation::cur());
            let r = meta.query_advice(r, Rotation::cur());
            let s_rescale = meta.query_selector(s_rescale);
            let scale = Expression::Constant(i128_to_fr(SCALE_18));
            let reciprocal_const = Expression::Constant(reciprocal_fr);
            let shift_scaled = Expression::Constant(i128_to_fr(SIGNED_SHIFT * SCALE_18));
            vec![
                s_rescale * (sum * reciprocal_const - (q_shift * scale.clone() - shift_scaled) - r),
            ]
        });

        // slack = (SCALE_18 - 1) - r, enforced at the same row as s_rescale's inputs.
        let s_slack = meta.selector();
        meta.create_gate("mean slack equals bound minus remainder", |meta| {
            let r = meta.query_advice(r, Rotation::cur());
            let slack = meta.query_advice(slack, Rotation::cur());
            let s_slack = meta.query_selector(s_slack);
            let bound_minus_one = Expression::Constant(i128_to_fr(SCALE_18 - 1));
            vec![s_slack * (slack + r - bound_minus_one)]
        });

        let range_q = RangeCheckChip::configure(meta, q, bits, 64);
        let range_r = RangeCheckChip::configure(meta, r, bits, REMAINDER_BITS);
        let range_r_slack = RangeCheckChip::configure(meta, slack, bits, REMAINDER_BITS);

        ReduceMeanConfig {
            sum: sum_config,
            q,
            r,
            slack,
            s_rescale,
            s_slack,
            range_q,
            range_r,
            range_r_slack,
            reciprocal,
        }
    }

    pub fn construct(config: ReduceMeanConfig) -> Self {
        ReduceMeanChip { config }
    }

    /// Assigns the running-sum region for `inputs`, then rescales the sum by
    /// the precomputed `1/k` reciprocal, range-checking the quotient (the
    /// I18 mean) and remainder/slack. Returns the mean as an `I18`, the
    /// `AssignedCell` holding the (signed-shifted) mean -- the same
    /// representation `RangeCheckChip`/this chip's own rescale gate use, see
    /// `chips/eltwise.rs`'s soundness note -- and the per-input
    /// `AssignedCell`s (in `inputs` order) holding `ReduceSumChip`'s own
    /// witnessed copies of each input, mirroring `ReduceSumChip::assign`'s
    /// own return shape. Composing chips (e.g. `LayerNormChip`, which calls
    /// this twice: once over raw inputs, once over squared deviations) must
    /// `region.constrain_equal` these cells to any cell where they re-derive
    /// or re-witness the same value, rather than leaving this chip's copies
    /// disconnected from them.
    #[allow(clippy::type_complexity)]
    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fr>,
        inputs: &[I18],
    ) -> Result<(I18, AssignedCell<Fr, Fr>, Vec<AssignedCell<Fr, Fr>>), ErrorFront> {
        let sum_chip = ReduceSumChip::construct(self.config.sum.clone());
        let (sum, sum_cell, value_cells) =
            sum_chip.assign(layouter.namespace(|| "mean sum"), inputs)?;

        let (mean, r) =
            requantize_mul(sum, self.config.reciprocal).expect("I18 mean rescale overflow");
        let slack = SCALE_18 - 1 - r;
        let (q_shift_fr, q_shift_raw) = shifted_i64_witness(mean.raw());

        let (sum_link_cell, q_cell, r_cell, slack_cell) = layouter.assign_region(
            || "mean rescale",
            |mut region| {
                self.config.s_rescale.enable(&mut region, 0)?;
                self.config.s_slack.enable(&mut region, 0)?;
                let sum_link_cell = region.assign_advice(
                    || "sum",
                    self.config.sum.sum,
                    0,
                    || Value::known(i64_to_fr(sum.raw())),
                )?;
                let q_cell = region.assign_advice(|| "q", self.config.q, 0, || q_shift_fr)?;
                let r_cell = region.assign_advice(
                    || "r",
                    self.config.r,
                    0,
                    || Value::known(i128_to_fr(r)),
                )?;
                let slack_cell = region.assign_advice(
                    || "slack",
                    self.config.slack,
                    0,
                    || Value::known(i128_to_fr(slack)),
                )?;
                Ok((sum_link_cell, q_cell, r_cell, slack_cell))
            },
        )?;

        let range_q_chip = RangeCheckChip::construct(self.config.range_q.clone());
        let q_range_cell =
            range_q_chip.assign(layouter.namespace(|| "range q"), q_shift_fr, q_shift_raw)?;

        let range_r_chip = RangeCheckChip::construct(self.config.range_r.clone());
        let r_range_cell = range_r_chip.assign(
            layouter.namespace(|| "range r"),
            Value::known(i128_to_fr(r)),
            Value::known(r),
        )?;

        let range_r_slack_chip = RangeCheckChip::construct(self.config.range_r_slack.clone());
        let slack_range_cell = range_r_slack_chip.assign(
            layouter.namespace(|| "range r slack"),
            Value::known(i128_to_fr(slack)),
            Value::known(slack),
        )?;

        layouter.assign_region(
            || "mean range check links",
            |mut region| {
                // Ties this region's re-witnessed `sum` cell back to
                // ReduceSumChip's own final sum cell (same column, different
                // row/region) -- without this, the value fed into the mean
                // rescale gate would be disconnected from the actual sum.
                region.constrain_equal(sum_link_cell.cell(), sum_cell.cell())?;
                region.constrain_equal(q_cell.cell(), q_range_cell.cell())?;
                region.constrain_equal(r_cell.cell(), r_range_cell.cell())?;
                region.constrain_equal(slack_cell.cell(), slack_range_cell.cell())?;
                Ok(())
            },
        )?;

        Ok((mean, q_cell, value_cells))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_convert::Fr;
    use crate::fixed_point::I18;
    use halo2_proofs::circuit::{Layouter, SimpleFloorPlanner};
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::plonk::{Circuit, ConstraintSystem, ErrorFront};

    #[derive(Clone)]
    struct SumTestConfig {
        reduce: ReduceSumConfig,
    }

    struct SumTestCircuit {
        inputs: Vec<I18>,
    }

    impl Circuit<Fr> for SumTestCircuit {
        type Config = SumTestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            SumTestCircuit {
                inputs: vec![I18::from_raw(0); self.inputs.len()],
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let values = meta.advice_column();
            let sum = meta.advice_column();
            let sum_shift = meta.advice_column();
            let bits = meta.advice_column();
            SumTestConfig {
                reduce: ReduceSumChip::configure(meta, values, sum, sum_shift, bits, 4),
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = ReduceSumChip::construct(config.reduce);
            chip.assign(layouter, &self.inputs)?;
            Ok(())
        }
    }

    #[test]
    fn sum_of_four_mixed_sign_values_is_satisfied() {
        let inputs = vec![
            I18::from_f64(2.0).unwrap(),
            I18::from_f64(-1.5).unwrap(),
            I18::from_f64(3.25).unwrap(),
            I18::from_f64(-0.75).unwrap(),
        ];
        let circuit = SumTestCircuit { inputs };
        let prover = MockProver::run(12, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn sum_with_forged_running_total_is_rejected() {
        struct ForgedSumCircuit {
            inputs: Vec<I18>,
        }

        impl Circuit<Fr> for ForgedSumCircuit {
            type Config = SumTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                ForgedSumCircuit {
                    inputs: vec![I18::from_raw(0); self.inputs.len()],
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                SumTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                layouter.assign_region(
                    || "forged reduce sum",
                    |mut region| {
                        let mut running = 0i64;
                        for (i, v) in self.inputs.iter().enumerate() {
                            region.assign_advice(
                                || format!("value {i}"),
                                config.reduce.values,
                                i,
                                || Value::known(crate::field_convert::i64_to_fr(v.raw())),
                            )?;
                            if i == 0 {
                                config.reduce.s_first.enable(&mut region, 0)?;
                                running = v.raw();
                            } else {
                                config.reduce.s_running.enable(&mut region, i)?;
                                running += v.raw();
                            }
                            // Forge the last row's running total to be off by one.
                            let forged = if i == self.inputs.len() - 1 {
                                running + 1
                            } else {
                                running
                            };
                            region.assign_advice(
                                || format!("sum {i}"),
                                config.reduce.sum,
                                i,
                                || Value::known(crate::field_convert::i64_to_fr(forged)),
                            )?;
                        }
                        Ok(())
                    },
                )?;
                Ok(())
            }
        }

        let inputs = vec![
            I18::from_f64(2.0).unwrap(),
            I18::from_f64(-1.5).unwrap(),
            I18::from_f64(3.25).unwrap(),
            I18::from_f64(-0.75).unwrap(),
        ];
        let circuit = ForgedSumCircuit { inputs };
        let prover = MockProver::run(12, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    fn sum_constrain_equal_rejects_mismatched_range_check_witness() {
        // See the analogous test in chips/eltwise.rs for why this must
        // actually call `constrain_equal` (with mismatched values) rather
        // than merely omit the link, to be a meaningful probe.
        struct MismatchedLinkCircuit {
            inputs: Vec<I18>,
        }

        impl Circuit<Fr> for MismatchedLinkCircuit {
            type Config = SumTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                MismatchedLinkCircuit {
                    inputs: vec![I18::from_raw(0); self.inputs.len()],
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                SumTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let mut partial_sums: Vec<i64> = Vec::with_capacity(self.inputs.len());
                for (i, v) in self.inputs.iter().enumerate() {
                    let next = if i == 0 {
                        v.raw()
                    } else {
                        partial_sums[i - 1] + v.raw()
                    };
                    partial_sums.push(next);
                }
                let sum_raw = *partial_sums.last().unwrap();
                let last_row = self.inputs.len() - 1;

                let sum_shift_cell = layouter.assign_region(
                    || "reduce sum",
                    |mut region| {
                        let mut cell = None;
                        for (i, (v, s)) in self.inputs.iter().zip(partial_sums.iter()).enumerate() {
                            region.assign_advice(
                                || format!("value {i}"),
                                config.reduce.values,
                                i,
                                || Value::known(i64_to_fr(v.raw())),
                            )?;
                            if i == 0 {
                                config.reduce.s_first.enable(&mut region, 0)?;
                            } else {
                                config.reduce.s_running.enable(&mut region, i)?;
                            }
                            region.assign_advice(
                                || format!("sum {i}"),
                                config.reduce.sum,
                                i,
                                || Value::known(i64_to_fr(*s)),
                            )?;
                            if i == last_row {
                                config.reduce.s_shift.enable(&mut region, i)?;
                                let (sum_shift_fr, _) = shifted_i64_witness(sum_raw);
                                cell = Some(region.assign_advice(
                                    || "sum shift",
                                    config.reduce.sum_shift,
                                    i,
                                    || sum_shift_fr,
                                )?);
                            }
                        }
                        Ok(cell.unwrap())
                    },
                )?;

                // Mismatch: range-check a decoy (0) instead of the real
                // shifted sum, but still link it via constrain_equal.
                let (decoy_fr, decoy_raw) = shifted_i64_witness(0);
                let range_sum_chip = RangeCheckChip::construct(config.reduce.range_sum.clone());
                let range_cell = range_sum_chip.assign(
                    layouter.namespace(|| "range sum"),
                    decoy_fr,
                    decoy_raw,
                )?;

                layouter.assign_region(
                    || "reduce sum range check link",
                    |mut region| region.constrain_equal(sum_shift_cell.cell(), range_cell.cell()),
                )?;

                Ok(())
            }
        }

        let inputs = vec![
            I18::from_f64(2.0).unwrap(),
            I18::from_f64(-1.5).unwrap(),
            I18::from_f64(3.25).unwrap(),
            I18::from_f64(-0.75).unwrap(),
        ];
        let circuit = MismatchedLinkCircuit { inputs };
        let prover = MockProver::run(12, &circuit, vec![]).unwrap();
        assert!(
            prover.verify().is_err(),
            "constrain_equal must reject a sum_shift cell tied to a mismatched decoy value"
        );
    }

    #[derive(Clone)]
    struct MeanTestConfig {
        reduce: ReduceMeanConfig,
    }

    struct MeanTestCircuit {
        inputs: Vec<I18>,
    }

    impl Circuit<Fr> for MeanTestCircuit {
        type Config = MeanTestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            MeanTestCircuit {
                inputs: vec![I18::from_raw(0); self.inputs.len()],
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let values = meta.advice_column();
            let sum = meta.advice_column();
            let sum_shift = meta.advice_column();
            let q = meta.advice_column();
            let r = meta.advice_column();
            let slack = meta.advice_column();
            let bits = meta.advice_column();
            MeanTestConfig {
                reduce: ReduceMeanChip::configure(
                    meta, values, sum, sum_shift, q, r, slack, bits, 4,
                ),
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = ReduceMeanChip::construct(config.reduce);
            chip.assign(layouter, &self.inputs)?;
            Ok(())
        }
    }

    #[test]
    fn mean_of_four_mixed_sign_values_is_satisfied() {
        let inputs = vec![
            I18::from_f64(2.0).unwrap(),
            I18::from_f64(-1.5).unwrap(),
            I18::from_f64(3.25).unwrap(),
            I18::from_f64(-0.75).unwrap(),
        ];
        let circuit = MeanTestCircuit {
            inputs: inputs.clone(),
        };
        let prover = MockProver::run(12, &circuit, vec![]).unwrap();
        prover.assert_satisfied();

        // Cross-check the expected mean value within fixed-point tolerance;
        // 1/4 is exactly representable in I18, so the only quantization
        // error comes from the inputs' own rounding.
        let expected: f64 = inputs.iter().map(I18::to_f64).sum::<f64>() / 4.0;
        let sum_raw: i64 = inputs.iter().map(|v| v.raw()).sum();
        let mean = crate::fixed_point::requantize_mul(
            I18::from_raw(sum_raw),
            I18::from_f64(0.25).unwrap(),
        )
        .unwrap()
        .0;
        assert!((mean.to_f64() - expected).abs() < 1e-9);
    }

    #[test]
    fn mean_with_forged_quotient_is_rejected() {
        struct ForgedMeanCircuit {
            inputs: Vec<I18>,
        }

        impl Circuit<Fr> for ForgedMeanCircuit {
            type Config = MeanTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                ForgedMeanCircuit {
                    inputs: vec![I18::from_raw(0); self.inputs.len()],
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                MeanTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let sum_chip = ReduceSumChip::construct(config.reduce.sum.clone());
                let (sum, _sum_cell, _value_cells) =
                    sum_chip.assign(layouter.namespace(|| "mean sum"), &self.inputs)?;

                let (q, r) =
                    crate::fixed_point::requantize_mul(sum, config.reduce.reciprocal).unwrap();
                let forged_q = q.raw() + 1; // violates sum * (1/k) == q * SCALE_18 + r
                let (forged_q_shift_fr, _) = shifted_i64_witness(forged_q);
                let slack = crate::fixed_point::SCALE_18 - 1 - r;

                layouter.assign_region(
                    || "forged mean rescale",
                    |mut region| {
                        config.reduce.s_rescale.enable(&mut region, 0)?;
                        config.reduce.s_slack.enable(&mut region, 0)?;
                        region.assign_advice(
                            || "sum",
                            config.reduce.sum.sum,
                            0,
                            || Value::known(crate::field_convert::i64_to_fr(sum.raw())),
                        )?;
                        region.assign_advice(|| "q", config.reduce.q, 0, || forged_q_shift_fr)?;
                        region.assign_advice(
                            || "r",
                            config.reduce.r,
                            0,
                            || Value::known(crate::field_convert::i128_to_fr(r)),
                        )?;
                        region.assign_advice(
                            || "slack",
                            config.reduce.slack,
                            0,
                            || Value::known(crate::field_convert::i128_to_fr(slack)),
                        )
                    },
                )?;
                Ok(())
            }
        }

        let inputs = vec![
            I18::from_f64(2.0).unwrap(),
            I18::from_f64(-1.5).unwrap(),
            I18::from_f64(3.25).unwrap(),
            I18::from_f64(-0.75).unwrap(),
        ];
        let circuit = ForgedMeanCircuit { inputs };
        let prover = MockProver::run(12, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    fn mean_constrain_equal_rejects_mismatched_range_check_witness() {
        struct MismatchedLinkCircuit {
            inputs: Vec<I18>,
        }

        impl Circuit<Fr> for MismatchedLinkCircuit {
            type Config = MeanTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                MismatchedLinkCircuit {
                    inputs: vec![I18::from_raw(0); self.inputs.len()],
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                MeanTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let sum_chip = ReduceSumChip::construct(config.reduce.sum.clone());
                let (sum, sum_cell, _value_cells) =
                    sum_chip.assign(layouter.namespace(|| "mean sum"), &self.inputs)?;

                let (mean, r) =
                    crate::fixed_point::requantize_mul(sum, config.reduce.reciprocal).unwrap();
                let slack = crate::fixed_point::SCALE_18 - 1 - r;
                let (q_shift_fr, _) = shifted_i64_witness(mean.raw());

                let (sum_link_cell, q_cell, r_cell, slack_cell) = layouter.assign_region(
                    || "mean rescale",
                    |mut region| {
                        config.reduce.s_rescale.enable(&mut region, 0)?;
                        config.reduce.s_slack.enable(&mut region, 0)?;
                        let sum_link_cell = region.assign_advice(
                            || "sum",
                            config.reduce.sum.sum,
                            0,
                            || Value::known(i64_to_fr(sum.raw())),
                        )?;
                        let q_cell =
                            region.assign_advice(|| "q", config.reduce.q, 0, || q_shift_fr)?;
                        let r_cell = region.assign_advice(
                            || "r",
                            config.reduce.r,
                            0,
                            || Value::known(i128_to_fr(r)),
                        )?;
                        let slack_cell = region.assign_advice(
                            || "slack",
                            config.reduce.slack,
                            0,
                            || Value::known(i128_to_fr(slack)),
                        )?;
                        Ok((sum_link_cell, q_cell, r_cell, slack_cell))
                    },
                )?;

                let range_r_chip = RangeCheckChip::construct(config.reduce.range_r.clone());
                let r_range_cell = range_r_chip.assign(
                    layouter.namespace(|| "range r"),
                    Value::known(i128_to_fr(r)),
                    Value::known(r),
                )?;
                let range_r_slack_chip =
                    RangeCheckChip::construct(config.reduce.range_r_slack.clone());
                let slack_range_cell = range_r_slack_chip.assign(
                    layouter.namespace(|| "range r slack"),
                    Value::known(i128_to_fr(slack)),
                    Value::known(slack),
                )?;

                // Mismatch: range-check a decoy (0) for `q` instead of the
                // real shifted mean, but still link it via constrain_equal.
                let (decoy_fr, decoy_raw) = shifted_i64_witness(0);
                let range_q_chip = RangeCheckChip::construct(config.reduce.range_q.clone());
                let q_range_cell =
                    range_q_chip.assign(layouter.namespace(|| "range q"), decoy_fr, decoy_raw)?;

                layouter.assign_region(
                    || "mean range check links",
                    |mut region| {
                        region.constrain_equal(sum_link_cell.cell(), sum_cell.cell())?;
                        region.constrain_equal(q_cell.cell(), q_range_cell.cell())?;
                        region.constrain_equal(r_cell.cell(), r_range_cell.cell())?;
                        region.constrain_equal(slack_cell.cell(), slack_range_cell.cell())?;
                        Ok(())
                    },
                )?;

                Ok(())
            }
        }

        let inputs = vec![
            I18::from_f64(2.0).unwrap(),
            I18::from_f64(-1.5).unwrap(),
            I18::from_f64(3.25).unwrap(),
            I18::from_f64(-0.75).unwrap(),
        ];
        let circuit = MismatchedLinkCircuit { inputs };
        let prover = MockProver::run(12, &circuit, vec![]).unwrap();
        assert!(
            prover.verify().is_err(),
            "constrain_equal must reject a q cell tied to a mismatched decoy range-check value"
        );
    }
}
