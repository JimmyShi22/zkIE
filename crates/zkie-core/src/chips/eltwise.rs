use crate::chips::range_check::{RangeCheckChip, RangeCheckConfig};
use crate::field_convert::{i128_to_fr, i64_to_fr, Fr};
use crate::fixed_point::{requantize_mul, I18, SCALE_18};
use halo2_proofs::circuit::{Layouter, Value};
use halo2_proofs::plonk::{Advice, Column, ConstraintSystem, ErrorFront, Expression, Selector};
use halo2_proofs::poly::Rotation;

const SIGNED_SHIFT: i128 = 1i128 << 63;
const REMAINDER_BITS: usize = 60; // 2^60 > SCALE_18 - 1.

fn shifted_i64_witness(v: i64) -> (Value<Fr>, Value<i128>) {
    let shifted = (v as i128) + SIGNED_SHIFT;
    (Value::known(i128_to_fr(shifted)), Value::known(shifted))
}

#[derive(Clone, Debug)]
pub struct EltwiseAddConfig {
    a: Column<Advice>,
    b: Column<Advice>,
    c: Column<Advice>,
    s_add: Selector,
    range_a: RangeCheckConfig,
    range_b: RangeCheckConfig,
    range_c: RangeCheckConfig,
}

pub struct EltwiseAddChip {
    config: EltwiseAddConfig,
}

impl EltwiseAddChip {
    pub fn configure(
        meta: &mut ConstraintSystem<Fr>,
        a: Column<Advice>,
        b: Column<Advice>,
        c: Column<Advice>,
        bits: Column<Advice>,
    ) -> EltwiseAddConfig {
        meta.enable_equality(a);
        meta.enable_equality(b);
        meta.enable_equality(c);

        let s_add = meta.selector();
        meta.create_gate("add", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let c = meta.query_advice(c, Rotation::cur());
            let s_add = meta.query_selector(s_add);
            vec![s_add * (a + b - c)]
        });

        let range_a = RangeCheckChip::configure(meta, a, bits, 64);
        let range_b = RangeCheckChip::configure(meta, b, bits, 64);
        let range_c = RangeCheckChip::configure(meta, c, bits, 64);

        EltwiseAddConfig { a, b, c, s_add, range_a, range_b, range_c }
    }

    pub fn construct(config: EltwiseAddConfig) -> Self {
        EltwiseAddChip { config }
    }

    pub fn assign(&self, mut layouter: impl Layouter<Fr>, a: I18, b: I18) -> Result<(), ErrorFront> {
        let c_raw = a.raw().checked_add(b.raw()).expect("I18 add overflow");

        layouter.assign_region(
            || "eltwise add",
            |mut region| {
                self.config.s_add.enable(&mut region, 0)?;
                region.assign_advice(|| "a", self.config.a, 0, || Value::known(i64_to_fr(a.raw())))?;
                region.assign_advice(|| "b", self.config.b, 0, || Value::known(i64_to_fr(b.raw())))?;
                region.assign_advice(|| "c", self.config.c, 0, || Value::known(i64_to_fr(c_raw)))?;
                Ok(())
            },
        )?;

        let (a_shift_fr, a_shift_raw) = shifted_i64_witness(a.raw());
        let range_a_chip = RangeCheckChip::construct(self.config.range_a.clone());
        range_a_chip.assign(layouter.namespace(|| "range a"), a_shift_fr, a_shift_raw)?;

        let (b_shift_fr, b_shift_raw) = shifted_i64_witness(b.raw());
        let range_b_chip = RangeCheckChip::construct(self.config.range_b.clone());
        range_b_chip.assign(layouter.namespace(|| "range b"), b_shift_fr, b_shift_raw)?;

        let (c_shift_fr, c_shift_raw) = shifted_i64_witness(c_raw);
        let range_c_chip = RangeCheckChip::construct(self.config.range_c.clone());
        range_c_chip.assign(layouter.namespace(|| "range c"), c_shift_fr, c_shift_raw)?;

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct EltwiseMulConfig {
    a: Column<Advice>,
    b: Column<Advice>,
    q: Column<Advice>,
    r: Column<Advice>,
    slack: Column<Advice>,
    s_mul: Selector,
    s_slack: Selector,
    range_q: RangeCheckConfig,
    range_r: RangeCheckConfig,
    range_r_slack: RangeCheckConfig,
}

pub struct EltwiseMulChip {
    config: EltwiseMulConfig,
}

impl EltwiseMulChip {
    #[allow(clippy::too_many_arguments)]
    pub fn configure(
        meta: &mut ConstraintSystem<Fr>,
        a: Column<Advice>,
        b: Column<Advice>,
        q: Column<Advice>,
        r: Column<Advice>,
        slack: Column<Advice>,
        bits: Column<Advice>,
    ) -> EltwiseMulConfig {
        meta.enable_equality(a);
        meta.enable_equality(b);
        meta.enable_equality(q);
        meta.enable_equality(r);
        meta.enable_equality(slack);

        let s_mul = meta.selector();
        meta.create_gate("mul rescale", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let q = meta.query_advice(q, Rotation::cur());
            let r = meta.query_advice(r, Rotation::cur());
            let s_mul = meta.query_selector(s_mul);
            let scale = Expression::Constant(i128_to_fr(SCALE_18));
            vec![s_mul * (a * b - q * scale - r)]
        });

        // slack = (SCALE_18 - 1) - r, enforced at the same row as s_mul's inputs.
        let s_slack = meta.selector();
        meta.create_gate("slack equals bound minus remainder", |meta| {
            let r = meta.query_advice(r, Rotation::cur());
            let slack = meta.query_advice(slack, Rotation::cur());
            let s_slack = meta.query_selector(s_slack);
            let bound_minus_one = Expression::Constant(i128_to_fr(SCALE_18 - 1));
            vec![s_slack * (slack + r - bound_minus_one)]
        });

        let range_q = RangeCheckChip::configure(meta, q, bits, 64);
        let range_r = RangeCheckChip::configure(meta, r, bits, REMAINDER_BITS);
        let range_r_slack = RangeCheckChip::configure(meta, slack, bits, REMAINDER_BITS);

        EltwiseMulConfig {
            a,
            b,
            q,
            r,
            slack,
            s_mul,
            s_slack,
            range_q,
            range_r,
            range_r_slack,
        }
    }

    pub fn construct(config: EltwiseMulConfig) -> Self {
        EltwiseMulChip { config }
    }

    pub fn assign(&self, mut layouter: impl Layouter<Fr>, a: I18, b: I18) -> Result<(), ErrorFront> {
        let (q, r) = requantize_mul(a, b).expect("I18 mul overflow");
        let slack = SCALE_18 - 1 - r;

        layouter.assign_region(
            || "eltwise mul",
            |mut region| {
                self.config.s_mul.enable(&mut region, 0)?;
                self.config.s_slack.enable(&mut region, 0)?;
                region.assign_advice(|| "a", self.config.a, 0, || Value::known(i64_to_fr(a.raw())))?;
                region.assign_advice(|| "b", self.config.b, 0, || Value::known(i64_to_fr(b.raw())))?;
                region.assign_advice(|| "q", self.config.q, 0, || Value::known(i64_to_fr(q.raw())))?;
                region.assign_advice(|| "r", self.config.r, 0, || Value::known(i128_to_fr(r)))?;
                region.assign_advice(|| "slack", self.config.slack, 0, || Value::known(i128_to_fr(slack)))?;
                Ok(())
            },
        )?;

        let (q_shift_fr, q_shift_raw) = shifted_i64_witness(q.raw());
        let range_q_chip = RangeCheckChip::construct(self.config.range_q.clone());
        range_q_chip.assign(layouter.namespace(|| "range q"), q_shift_fr, q_shift_raw)?;

        let range_r_chip = RangeCheckChip::construct(self.config.range_r.clone());
        range_r_chip.assign(layouter.namespace(|| "range r"), Value::known(i128_to_fr(r)), Value::known(r))?;

        let range_r_slack_chip = RangeCheckChip::construct(self.config.range_r_slack.clone());
        range_r_slack_chip.assign(
            layouter.namespace(|| "range r slack"),
            Value::known(i128_to_fr(slack)),
            Value::known(slack),
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed_point::I18;
    use halo2_proofs::circuit::SimpleFloorPlanner;
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::plonk::{Circuit, ConstraintSystem, ErrorFront};

    #[derive(Clone)]
    struct AddTestConfig {
        add: EltwiseAddConfig,
    }

    struct AddTestCircuit {
        a: I18,
        b: I18,
    }

    impl Circuit<Fr> for AddTestCircuit {
        type Config = AddTestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            AddTestCircuit { a: I18::from_raw(0), b: I18::from_raw(0) }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let a = meta.advice_column();
            let b = meta.advice_column();
            let c = meta.advice_column();
            let bits = meta.advice_column();
            AddTestConfig { add: EltwiseAddChip::configure(meta, a, b, c, bits) }
        }

        fn synthesize(&self, config: Self::Config, layouter: impl Layouter<Fr>) -> Result<(), ErrorFront> {
            let chip = EltwiseAddChip::construct(config.add);
            chip.assign(layouter, self.a, self.b)
        }
    }

    #[test]
    fn add_positive_plus_positive_satisfied() {
        let circuit = AddTestCircuit { a: I18::from_f64(2.0).unwrap(), b: I18::from_f64(3.5).unwrap() };
        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn add_negative_plus_positive_satisfied() {
        let circuit = AddTestCircuit { a: I18::from_f64(-2.0).unwrap(), b: I18::from_f64(3.5).unwrap() };
        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn add_with_forged_sum_is_rejected() {
        struct ForgedAddCircuit {
            a: I18,
            b: I18,
        }

        impl Circuit<Fr> for ForgedAddCircuit {
            type Config = AddTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                ForgedAddCircuit { a: I18::from_raw(0), b: I18::from_raw(0) }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                AddTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                layouter.assign_region(
                    || "forged add",
                    |mut region| {
                        config.add.s_add.enable(&mut region, 0)?;
                        region.assign_advice(|| "a", config.add.a, 0, || Value::known(i64_to_fr(self.a.raw())))?;
                        region.assign_advice(|| "b", config.add.b, 0, || Value::known(i64_to_fr(self.b.raw())))?;
                        let forged_c = self.a.raw() + self.b.raw() + 1;
                        region.assign_advice(|| "c", config.add.c, 0, || Value::known(i64_to_fr(forged_c)))
                    },
                )?;
                Ok(())
            }
        }

        let circuit = ForgedAddCircuit { a: I18::from_f64(2.0).unwrap(), b: I18::from_f64(3.0).unwrap() };
        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[derive(Clone)]
    struct MulTestConfig {
        mul: EltwiseMulConfig,
    }

    struct MulTestCircuit {
        a: I18,
        b: I18,
    }

    impl Circuit<Fr> for MulTestCircuit {
        type Config = MulTestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            MulTestCircuit { a: I18::from_raw(0), b: I18::from_raw(0) }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let a = meta.advice_column();
            let b = meta.advice_column();
            let q = meta.advice_column();
            let r = meta.advice_column();
            let slack = meta.advice_column();
            let bits = meta.advice_column();
            MulTestConfig { mul: EltwiseMulChip::configure(meta, a, b, q, r, slack, bits) }
        }

        fn synthesize(&self, config: Self::Config, layouter: impl Layouter<Fr>) -> Result<(), ErrorFront> {
            let chip = EltwiseMulChip::construct(config.mul);
            chip.assign(layouter, self.a, self.b)
        }
    }

    #[test]
    fn mul_positive_times_positive_satisfied() {
        let circuit = MulTestCircuit { a: I18::from_f64(2.0).unwrap(), b: I18::from_f64(3.0).unwrap() };
        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn mul_negative_times_positive_satisfied() {
        let circuit = MulTestCircuit { a: I18::from_f64(-2.5).unwrap(), b: I18::from_f64(2.0).unwrap() };
        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn mul_with_forged_quotient_is_rejected() {
        struct ForgedMulCircuit {
            a: I18,
            b: I18,
        }

        impl Circuit<Fr> for ForgedMulCircuit {
            type Config = MulTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                ForgedMulCircuit { a: I18::from_raw(0), b: I18::from_raw(0) }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                MulTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let (q, r) = requantize_mul(self.a, self.b).unwrap();
                let forged_q = q.raw() + 1; // violates a*b == q*SCALE_18 + r
                layouter.assign_region(
                    || "forged mul",
                    |mut region| {
                        config.mul.s_mul.enable(&mut region, 0)?;
                        config.mul.s_slack.enable(&mut region, 0)?;
                        region.assign_advice(|| "a", config.mul.a, 0, || Value::known(i64_to_fr(self.a.raw())))?;
                        region.assign_advice(|| "b", config.mul.b, 0, || Value::known(i64_to_fr(self.b.raw())))?;
                        region.assign_advice(|| "q", config.mul.q, 0, || Value::known(i64_to_fr(forged_q)))?;
                        region.assign_advice(|| "r", config.mul.r, 0, || Value::known(i128_to_fr(r)))?;
                        let slack = SCALE_18 - 1 - r;
                        region.assign_advice(|| "slack", config.mul.slack, 0, || Value::known(i128_to_fr(slack)))
                    },
                )?;
                Ok(())
            }
        }

        let circuit = ForgedMulCircuit { a: I18::from_f64(2.0).unwrap(), b: I18::from_f64(3.0).unwrap() };
        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }
}
