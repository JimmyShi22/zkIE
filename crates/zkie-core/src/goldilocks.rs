//! Non-native Goldilocks arithmetic in the BN254 scalar field.
//!
//! The GKR/sum-check layer proves over Goldilocks (`p = 2^64 - 2^32 + 1`, a
//! 64-bit prime), but the on-chain verifier circuit runs over BN254 `Fr`
//! (~254-bit). This module is the first building block for closing that gap:
//! it proves, natively in `Fr`, that `c == a*b (mod p)` by witnessing the
//! quotient `q` with `a*b = q*p + c` and range-checking every value to 64 bits.

use crate::chips::range_check::{RangeCheckChip, RangeCheckConfig};
use crate::field_convert::Fr;
use halo2_proofs::circuit::{AssignedCell, Layouter, Value};
use halo2_proofs::plonk::{Advice, Column, ConstraintSystem, ErrorFront, Expression, Selector};
use halo2_proofs::poly::Rotation;

/// Goldilocks characteristic `2^64 - 2^32 + 1`.
pub const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;

#[derive(Clone, Debug)]
pub struct GoldilocksMulConfig {
    a: Column<Advice>,
    b: Column<Advice>,
    c: Column<Advice>,
    q: Column<Advice>,
    s_mul: Selector,
    range_a: RangeCheckConfig,
    range_b: RangeCheckConfig,
    range_c: RangeCheckConfig,
    range_q: RangeCheckConfig,
}

pub struct GoldilocksMulChip {
    config: GoldilocksMulConfig,
}

impl GoldilocksMulChip {
    pub fn configure(meta: &mut ConstraintSystem<Fr>) -> GoldilocksMulConfig {
        let a = meta.advice_column();
        let b = meta.advice_column();
        let c = meta.advice_column();
        let q = meta.advice_column();
        for col in [a, b, c, q] {
            meta.enable_equality(col);
        }

        // One bit-decomposition column per range-checked value.
        let a_bits = meta.advice_column();
        let b_bits = meta.advice_column();
        let c_bits = meta.advice_column();
        let q_bits = meta.advice_column();
        let range_a = RangeCheckChip::configure(meta, a, a_bits, 64);
        let range_b = RangeCheckChip::configure(meta, b, b_bits, 64);
        let range_c = RangeCheckChip::configure(meta, c, c_bits, 64);
        let range_q = RangeCheckChip::configure(meta, q, q_bits, 64);

        let s_mul = meta.selector();
        meta.create_gate("a*b - c = q*p", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let c = meta.query_advice(c, Rotation::cur());
            let q = meta.query_advice(q, Rotation::cur());
            let s = meta.query_selector(s_mul);
            let p = Expression::Constant(Fr::from(GOLDILOCKS_P));
            vec![s * (a * b - c - q * p)]
        });

        GoldilocksMulConfig {
            a,
            b,
            c,
            q,
            s_mul,
            range_a,
            range_b,
            range_c,
            range_q,
        }
    }

    pub fn construct(config: GoldilocksMulConfig) -> Self {
        GoldilocksMulChip { config }
    }

    /// Witness `a*b = q*p + c`, where every value is a canonical `u64`.
    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fr>,
        a: u64,
        b: u64,
        c: u64,
        q: u64,
    ) -> Result<AssignedCell<Fr, Fr>, ErrorFront> {
        let c_cell = layouter.assign_region(
            || "goldilocks mul",
            |mut region| {
                self.config.s_mul.enable(&mut region, 0)?;
                region.assign_advice(|| "a", self.config.a, 0, || Value::known(Fr::from(a)))?;
                region.assign_advice(|| "b", self.config.b, 0, || Value::known(Fr::from(b)))?;
                region.assign_advice(|| "q", self.config.q, 0, || Value::known(Fr::from(q)))?;
                region.assign_advice(|| "c", self.config.c, 0, || Value::known(Fr::from(c)))
            },
        )?;

        let fr = |v: u64| Value::known(Fr::from(v));
        let raw = |v: u64| Value::known(v as i128);
        RangeCheckChip::construct(self.config.range_a.clone()).assign(
            layouter.namespace(|| "range a"),
            fr(a),
            raw(a),
        )?;
        RangeCheckChip::construct(self.config.range_b.clone()).assign(
            layouter.namespace(|| "range b"),
            fr(b),
            raw(b),
        )?;
        RangeCheckChip::construct(self.config.range_c.clone()).assign(
            layouter.namespace(|| "range c"),
            fr(c),
            raw(c),
        )?;
        RangeCheckChip::construct(self.config.range_q.clone()).assign(
            layouter.namespace(|| "range q"),
            fr(q),
            raw(q),
        )?;
        Ok(c_cell)
    }
}

/// Host-side helper: `(c, q)` such that `a*b = q*p + c` and `0 <= c < p`.
pub fn mul_mod(a: u64, b: u64) -> (u64, u64) {
    let prod = (a as u128) * (b as u128);
    let p = GOLDILOCKS_P as u128;
    ((prod % p) as u64, (prod / p) as u64)
}

#[derive(Clone, Debug)]
pub struct GoldilocksAddConfig {
    a: Column<Advice>,
    b: Column<Advice>,
    c: Column<Advice>,
    ov: Column<Advice>,
    s_add: Selector,
    s_ov_bit: Selector,
    range_a: RangeCheckConfig,
    range_b: RangeCheckConfig,
    range_c: RangeCheckConfig,
}

pub struct GoldilocksAddChip {
    config: GoldilocksAddConfig,
}

impl GoldilocksAddChip {
    pub fn configure(meta: &mut ConstraintSystem<Fr>) -> GoldilocksAddConfig {
        let a = meta.advice_column();
        let b = meta.advice_column();
        let c = meta.advice_column();
        let ov = meta.advice_column();
        for col in [a, b, c] {
            meta.enable_equality(col);
        }

        let s_ov_bit = meta.selector();
        meta.create_gate("ov is boolean", |meta| {
            let ov = meta.query_advice(ov, Rotation::cur());
            let s = meta.query_selector(s_ov_bit);
            let one = Expression::Constant(Fr::one());
            vec![s * ov.clone() * (one - ov)]
        });

        let s_add = meta.selector();
        meta.create_gate("a + b = c + ov*p", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let c = meta.query_advice(c, Rotation::cur());
            let ov = meta.query_advice(ov, Rotation::cur());
            let s = meta.query_selector(s_add);
            let p = Expression::Constant(Fr::from(GOLDILOCKS_P));
            vec![s * (a + b - c - ov * p)]
        });

        let a_bits = meta.advice_column();
        let b_bits = meta.advice_column();
        let c_bits = meta.advice_column();
        let range_a = RangeCheckChip::configure(meta, a, a_bits, 64);
        let range_b = RangeCheckChip::configure(meta, b, b_bits, 64);
        let range_c = RangeCheckChip::configure(meta, c, c_bits, 64);

        GoldilocksAddConfig {
            a,
            b,
            c,
            ov,
            s_add,
            s_ov_bit,
            range_a,
            range_b,
            range_c,
        }
    }

    pub fn construct(config: GoldilocksAddConfig) -> Self {
        GoldilocksAddChip { config }
    }

    /// Witness `a + b = c + ov*p` with `ov in {0,1}`.
    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fr>,
        a: u64,
        b: u64,
        c: u64,
        ov: u64,
    ) -> Result<AssignedCell<Fr, Fr>, ErrorFront> {
        let c_cell = layouter.assign_region(
            || "goldilocks add",
            |mut region| {
                self.config.s_add.enable(&mut region, 0)?;
                self.config.s_ov_bit.enable(&mut region, 0)?;
                region.assign_advice(|| "a", self.config.a, 0, || Value::known(Fr::from(a)))?;
                region.assign_advice(|| "b", self.config.b, 0, || Value::known(Fr::from(b)))?;
                region.assign_advice(|| "ov", self.config.ov, 0, || Value::known(Fr::from(ov)))?;
                region.assign_advice(|| "c", self.config.c, 0, || Value::known(Fr::from(c)))
            },
        )?;

        let fr = |v: u64| Value::known(Fr::from(v));
        let raw = |v: u64| Value::known(v as i128);
        RangeCheckChip::construct(self.config.range_a.clone()).assign(
            layouter.namespace(|| "range a"),
            fr(a),
            raw(a),
        )?;
        RangeCheckChip::construct(self.config.range_b.clone()).assign(
            layouter.namespace(|| "range b"),
            fr(b),
            raw(b),
        )?;
        RangeCheckChip::construct(self.config.range_c.clone()).assign(
            layouter.namespace(|| "range c"),
            fr(c),
            raw(c),
        )?;
        Ok(c_cell)
    }
}

/// Host-side helper: `(c, ov)` such that `a + b = ov*p + c` and `0 <= c < p`.
pub fn add_mod(a: u64, b: u64) -> (u64, u64) {
    let sum = (a as u128) + (b as u128);
    let p = GOLDILOCKS_P as u128;
    if sum >= p {
        ((sum - p) as u64, 1)
    } else {
        (sum as u64, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::circuit::SimpleFloorPlanner;
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::plonk::{Circuit, ConstraintSystem, ErrorFront};

    #[derive(Clone)]
    struct TestConfig {
        mul: GoldilocksMulConfig,
    }

    struct TestCircuit {
        a: u64,
        b: u64,
        c: u64,
        q: u64,
    }

    impl Circuit<Fr> for TestCircuit {
        type Config = TestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            TestCircuit {
                a: 0,
                b: 0,
                c: 0,
                q: 0,
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            TestConfig {
                mul: GoldilocksMulChip::configure(meta),
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            GoldilocksMulChip::construct(config.mul).assign(
                layouter,
                self.a,
                self.b,
                self.c,
                self.q,
            )?;
            Ok(())
        }
    }

    #[test]
    fn product_without_wraparound_is_satisfied() {
        let (a, b) = (5u64, 7u64);
        let (c, q) = mul_mod(a, b);
        assert_eq!((c, q), (35, 0));
        let prover = MockProver::run(12, &TestCircuit { a, b, c, q }, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn product_with_wraparound_is_satisfied() {
        let a = GOLDILOCKS_P - 1;
        let b = 2u64;
        let (c, q) = mul_mod(a, b);
        assert_eq!(c, GOLDILOCKS_P - 2);
        assert_eq!(q, 1);
        let prover = MockProver::run(12, &TestCircuit { a, b, c, q }, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn wrong_product_is_rejected() {
        let (a, b) = (5u64, 7u64);
        let (_, q) = mul_mod(a, b);
        let wrong_c = 36u64; // 5*7 != 36 mod p
        let prover = MockProver::run(12, &TestCircuit { a, b, c: wrong_c, q }, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[derive(Clone)]
    struct AddTestConfig {
        add: GoldilocksAddConfig,
    }

    struct AddTestCircuit {
        a: u64,
        b: u64,
        c: u64,
        ov: u64,
    }

    impl Circuit<Fr> for AddTestCircuit {
        type Config = AddTestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            AddTestCircuit { a: 0, b: 0, c: 0, ov: 0 }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            AddTestConfig {
                add: GoldilocksAddChip::configure(meta),
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            GoldilocksAddChip::construct(config.add).assign(
                layouter, self.a, self.b, self.c, self.ov,
            )?;
            Ok(())
        }
    }

    #[test]
    fn add_without_wraparound_is_satisfied() {
        let (a, b) = (5u64, 7u64);
        let (c, ov) = add_mod(a, b);
        assert_eq!((c, ov), (12, 0));
        let prover = MockProver::run(12, &AddTestCircuit { a, b, c, ov }, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn add_with_wraparound_is_satisfied() {
        let a = GOLDILOCKS_P - 1;
        let b = 2u64;
        let (c, ov) = add_mod(a, b);
        assert_eq!((c, ov), (1, 1));
        let prover = MockProver::run(12, &AddTestCircuit { a, b, c, ov }, vec![]).unwrap();
        prover.assert_satisfied();
    }
}
