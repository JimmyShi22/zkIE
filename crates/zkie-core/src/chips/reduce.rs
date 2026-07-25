use crate::chips::range_check::{RangeCheckChip, RangeCheckConfig};
use crate::field_convert::{i128_to_fr, i64_to_fr, shifted_i64_witness, Fr};
use crate::fixed_point::{requantize_mul, I18, SCALE_18};
use halo2_proofs::circuit::{Layouter, Value};
use halo2_proofs::plonk::{Advice, Column, ConstraintSystem, ErrorFront, Expression, Selector};
use halo2_proofs::poly::Rotation;

const REMAINDER_BITS: usize = 60; // 2^60 > SCALE_18 - 1, matches eltwise.rs's mul gadget.

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
    s_first: Selector,
    s_running: Selector,
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
        bits: Column<Advice>,
        k: usize,
    ) -> ReduceSumConfig {
        assert!(k >= 1, "ReduceSumChip requires at least one input");
        meta.enable_equality(values);
        meta.enable_equality(sum);

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

        let range_sum = RangeCheckChip::configure(meta, sum, bits, 64);

        ReduceSumConfig {
            values,
            sum,
            s_first,
            s_running,
            range_sum,
            k,
        }
    }

    pub fn construct(config: ReduceSumConfig) -> Self {
        ReduceSumChip { config }
    }

    /// Assigns the running-sum region for `inputs` (must have length `k`,
    /// the count fixed at `configure` time), range-checks the final sum as
    /// a signed 64-bit value, and returns the sum as an `I18`.
    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fr>,
        inputs: &[I18],
    ) -> Result<I18, ErrorFront> {
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

        layouter.assign_region(
            || "reduce sum",
            |mut region| {
                for (i, (v, s)) in inputs.iter().zip(partial_sums.iter()).enumerate() {
                    region.assign_advice(
                        || format!("value {i}"),
                        self.config.values,
                        i,
                        || Value::known(i64_to_fr(v.raw())),
                    )?;
                    if i == 0 {
                        self.config.s_first.enable(&mut region, 0)?;
                    } else {
                        self.config.s_running.enable(&mut region, i)?;
                    }
                    region.assign_advice(
                        || format!("sum {i}"),
                        self.config.sum,
                        i,
                        || Value::known(i64_to_fr(*s)),
                    )?;
                }
                Ok(())
            },
        )?;

        let (sum_shift_fr, sum_shift_raw) = shifted_i64_witness(sum_raw);
        let range_sum_chip = RangeCheckChip::construct(self.config.range_sum.clone());
        range_sum_chip.assign(
            layouter.namespace(|| "range sum"),
            sum_shift_fr,
            sum_shift_raw,
        )?;

        Ok(I18::from_raw(sum_raw))
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
        q: Column<Advice>,
        r: Column<Advice>,
        slack: Column<Advice>,
        bits: Column<Advice>,
        k: usize,
    ) -> ReduceMeanConfig {
        let sum_config = ReduceSumChip::configure(meta, values, sum, bits, k);

        meta.enable_equality(q);
        meta.enable_equality(r);
        meta.enable_equality(slack);

        let reciprocal = I18::from_f64(1.0 / (k as f64))
            .expect("1/k must be representable as an I18 fixed-point value");
        let reciprocal_fr = i64_to_fr(reciprocal.raw());

        let s_rescale = meta.selector();
        meta.create_gate("mean rescale", |meta| {
            let sum = meta.query_advice(sum, Rotation::cur());
            let q = meta.query_advice(q, Rotation::cur());
            let r = meta.query_advice(r, Rotation::cur());
            let s_rescale = meta.query_selector(s_rescale);
            let scale = Expression::Constant(i128_to_fr(SCALE_18));
            let reciprocal_const = Expression::Constant(reciprocal_fr);
            vec![s_rescale * (sum * reciprocal_const - q * scale - r)]
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
    /// I18 mean) and remainder/slack. Returns the mean as an `I18`.
    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fr>,
        inputs: &[I18],
    ) -> Result<I18, ErrorFront> {
        let sum_chip = ReduceSumChip::construct(self.config.sum.clone());
        let sum = sum_chip.assign(layouter.namespace(|| "mean sum"), inputs)?;

        let (mean, r) =
            requantize_mul(sum, self.config.reciprocal).expect("I18 mean rescale overflow");
        let slack = SCALE_18 - 1 - r;

        layouter.assign_region(
            || "mean rescale",
            |mut region| {
                self.config.s_rescale.enable(&mut region, 0)?;
                self.config.s_slack.enable(&mut region, 0)?;
                region.assign_advice(
                    || "sum",
                    self.config.sum.sum,
                    0,
                    || Value::known(i64_to_fr(sum.raw())),
                )?;
                region.assign_advice(
                    || "q",
                    self.config.q,
                    0,
                    || Value::known(i64_to_fr(mean.raw())),
                )?;
                region.assign_advice(|| "r", self.config.r, 0, || Value::known(i128_to_fr(r)))?;
                region.assign_advice(
                    || "slack",
                    self.config.slack,
                    0,
                    || Value::known(i128_to_fr(slack)),
                )?;
                Ok(())
            },
        )?;

        let (q_shift_fr, q_shift_raw) = shifted_i64_witness(mean.raw());
        let range_q_chip = RangeCheckChip::construct(self.config.range_q.clone());
        range_q_chip.assign(layouter.namespace(|| "range q"), q_shift_fr, q_shift_raw)?;

        let range_r_chip = RangeCheckChip::construct(self.config.range_r.clone());
        range_r_chip.assign(
            layouter.namespace(|| "range r"),
            Value::known(i128_to_fr(r)),
            Value::known(r),
        )?;

        let range_r_slack_chip = RangeCheckChip::construct(self.config.range_r_slack.clone());
        range_r_slack_chip.assign(
            layouter.namespace(|| "range r slack"),
            Value::known(i128_to_fr(slack)),
            Value::known(slack),
        )?;

        Ok(mean)
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
            let bits = meta.advice_column();
            SumTestConfig {
                reduce: ReduceSumChip::configure(meta, values, sum, bits, 4),
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
            let q = meta.advice_column();
            let r = meta.advice_column();
            let slack = meta.advice_column();
            let bits = meta.advice_column();
            MeanTestConfig {
                reduce: ReduceMeanChip::configure(meta, values, sum, q, r, slack, bits, 4),
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
                let sum = sum_chip.assign(layouter.namespace(|| "mean sum"), &self.inputs)?;

                let (q, r) =
                    crate::fixed_point::requantize_mul(sum, config.reduce.reciprocal).unwrap();
                let forged_q = q.raw() + 1; // violates sum * (1/k) == q * SCALE_18 + r
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
                        region.assign_advice(
                            || "q",
                            config.reduce.q,
                            0,
                            || Value::known(crate::field_convert::i64_to_fr(forged_q)),
                        )?;
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
}
