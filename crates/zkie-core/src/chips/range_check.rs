use crate::field_convert::Fr;
use halo2_proofs::circuit::{AssignedCell, Layouter, Value};
use halo2_proofs::plonk::{Advice, Column, ConstraintSystem, ErrorFront, Expression, Selector};
use halo2_proofs::poly::Rotation;

#[derive(Clone, Debug)]
pub struct RangeCheckConfig {
    pub value: Column<Advice>,
    pub bits: Column<Advice>,
    pub s_bit: Selector,
    pub s_sum: Selector,
    pub n_bits: usize,
}

pub struct RangeCheckChip {
    config: RangeCheckConfig,
}

impl RangeCheckChip {
    pub fn configure(
        meta: &mut ConstraintSystem<Fr>,
        value: Column<Advice>,
        bits: Column<Advice>,
        n_bits: usize,
    ) -> RangeCheckConfig {
        meta.enable_equality(bits);

        let s_bit = meta.selector();
        meta.create_gate("bit is boolean", |meta| {
            let bit = meta.query_advice(bits, Rotation::cur());
            let s_bit = meta.query_selector(s_bit);
            let one = Expression::Constant(Fr::one());
            vec![s_bit * bit.clone() * (one - bit)]
        });

        let s_sum = meta.selector();
        meta.create_gate("running sum equals value", |meta| {
            let value = meta.query_advice(value, Rotation::cur());
            let s_sum = meta.query_selector(s_sum);
            let mut sum = Expression::Constant(Fr::zero());
            let mut coeff = Fr::one();
            for i in 0..n_bits {
                // `assign` puts bit `i` (2^i) at row `i`, with `s_sum` enabled
                // at row `n_bits - 1`. Relative to that row, bit `i` sits at
                // rotation `i - (n_bits - 1)`.
                let rotation = Rotation(-((n_bits - 1 - i) as i32));
                let bit = meta.query_advice(bits, rotation);
                sum = sum + bit * Expression::Constant(coeff);
                coeff = coeff.double();
            }
            vec![s_sum * (sum - value)]
        });

        RangeCheckConfig {
            value,
            bits,
            s_bit,
            s_sum,
            n_bits,
        }
    }

    pub fn construct(config: RangeCheckConfig) -> Self {
        RangeCheckChip { config }
    }

    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fr>,
        value: Value<Fr>,
        raw_value: Value<i128>,
    ) -> Result<AssignedCell<Fr, Fr>, ErrorFront> {
        let n_bits = self.config.n_bits;
        layouter.assign_region(
            || "range check",
            |mut region| {
                for i in 0..n_bits {
                    self.config.s_bit.enable(&mut region, i)?;
                    let bit_value = raw_value.map(|v| ((v >> i) & 1) as u64);
                    region.assign_advice(
                        || format!("bit {i}"),
                        self.config.bits,
                        i,
                        || bit_value.map(Fr::from),
                    )?;
                }
                self.config.s_sum.enable(&mut region, n_bits - 1)?;
                region.assign_advice(|| "value", self.config.value, n_bits - 1, || value)
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_convert::Fr;
    use halo2_proofs::circuit::{SimpleFloorPlanner, Value};
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::plonk::{Circuit, ConstraintSystem, ErrorFront};

    #[derive(Clone)]
    struct TestConfig {
        range: RangeCheckConfig,
    }

    struct TestCircuit {
        value: Value<Fr>,
        raw_value: Value<i128>,
        n_bits: usize,
    }

    impl Circuit<Fr> for TestCircuit {
        type Config = TestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            TestCircuit {
                value: Value::unknown(),
                raw_value: Value::unknown(),
                n_bits: self.n_bits,
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let value = meta.advice_column();
            let bits = meta.advice_column();
            meta.enable_equality(value);
            let range = RangeCheckChip::configure(meta, value, bits, 8);
            TestConfig { range }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = RangeCheckChip::construct(config.range);
            chip.assign(layouter, self.value, self.raw_value)?;
            Ok(())
        }
    }

    #[test]
    fn value_within_8_bit_range_is_satisfied() {
        let circuit = TestCircuit {
            value: Value::known(Fr::from(200u64)),
            raw_value: Value::known(200i128),
            n_bits: 8,
        };
        let prover = MockProver::run(6, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn value_equal_to_zero_is_satisfied() {
        let circuit = TestCircuit {
            value: Value::known(Fr::zero()),
            raw_value: Value::known(0i128),
            n_bits: 8,
        };
        let prover = MockProver::run(6, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn value_at_max_of_range_is_satisfied() {
        let circuit = TestCircuit {
            value: Value::known(Fr::from(255u64)),
            raw_value: Value::known(255i128),
            n_bits: 8,
        };
        let prover = MockProver::run(6, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn value_exceeding_8_bit_range_is_rejected() {
        // 256 does not fit in 8 bits: the running-sum decomposition of the
        // claimed bits cannot equal 256 if all 8 bits are constrained boolean,
        // so the assigned `value` witness (256) will mismatch the sum (<=255).
        let circuit = TestCircuit {
            value: Value::known(Fr::from(256u64)),
            raw_value: Value::known(256i128),
            n_bits: 8,
        };
        let prover = MockProver::run(6, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[derive(Clone)]
    struct WideTestConfig {
        range: RangeCheckConfig,
    }

    struct WideTestCircuit {
        value: Value<Fr>,
        raw_value: Value<i128>,
    }

    impl Circuit<Fr> for WideTestCircuit {
        type Config = WideTestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            WideTestCircuit {
                value: Value::unknown(),
                raw_value: Value::unknown(),
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let value = meta.advice_column();
            let bits = meta.advice_column();
            meta.enable_equality(value);
            WideTestConfig {
                range: RangeCheckChip::configure(meta, value, bits, 64),
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            RangeCheckChip::construct(config.range).assign(layouter, self.value, self.raw_value)?;
            Ok(())
        }
    }

    #[test]
    fn value_within_64_bit_range_is_satisfied() {
        let raw: i128 = (1i128 << 63) + 12345; // an "offset by 2^63" style shifted value
        let circuit = WideTestCircuit {
            value: Value::known(crate::field_convert::i128_to_fr(raw)),
            raw_value: Value::known(raw),
        };
        let prover = MockProver::run(8, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}
