//! `DotProductChip`: the atomic per-output-element primitive backing the
//! `DOT_GENERAL` instruction. Computes `result = sum_{i=0}^{K-1}(a_i * b_i)`
//! over length-K vectors of I18 values, requantized back to I18.
//!
//! Full M x N x K matrix tiling is out of scope here: a later ONNX-compiler
//! sub-project will instantiate this chip once per output element.

use crate::chips::range_check::{RangeCheckChip, RangeCheckConfig};
use crate::field_convert::{i128_to_fr, i64_to_fr, shifted_i64_witness, Fr};
use crate::fixed_point::{requantize_raw, FixedPointError, I18, SCALE_18};
use halo2_proofs::circuit::{Layouter, Value};
use halo2_proofs::plonk::{Advice, Column, ConstraintSystem, ErrorFront, Expression, Selector};
use halo2_proofs::poly::Rotation;
use std::fmt;

// 2^60 > SCALE_18 - 1, so 60 bits is enough to bound a remainder in [0, SCALE_18).
const REMAINDER_BITS: usize = 60;

/// Errors that can occur while assigning a `DotProductChip` region.
#[derive(Debug)]
pub enum DotProductError {
    /// `a`/`b` did not both have exactly `K` (the configured length) elements.
    LengthMismatch {
        expected: usize,
        got_a: usize,
        got_b: usize,
    },
    /// The accumulated raw product sum, or its requantized quotient,
    /// overflowed the representable range.
    Overflow(FixedPointError),
    /// A halo2 circuit-synthesis error occurred while assigning cells.
    Circuit(ErrorFront),
}

impl fmt::Display for DotProductError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DotProductError::LengthMismatch {
                expected,
                got_a,
                got_b,
            } => write!(
                f,
                "dot product expects vectors of length {expected}, got a.len()={got_a}, b.len()={got_b}"
            ),
            DotProductError::Overflow(e) => write!(f, "dot product overflow: {e}"),
            DotProductError::Circuit(e) => write!(f, "dot product circuit error: {e:?}"),
        }
    }
}

impl std::error::Error for DotProductError {}

impl From<ErrorFront> for DotProductError {
    fn from(e: ErrorFront) -> Self {
        DotProductError::Circuit(e)
    }
}

#[derive(Clone, Debug)]
pub struct DotProductConfig {
    a: Column<Advice>,
    b: Column<Advice>,
    accumulator: Column<Advice>,
    q: Column<Advice>,
    r: Column<Advice>,
    slack: Column<Advice>,
    s_acc_start: Selector,
    s_acc_step: Selector,
    s_final: Selector,
    s_slack: Selector,
    range_q: RangeCheckConfig,
    range_r: RangeCheckConfig,
    range_r_slack: RangeCheckConfig,
    k: usize,
}

pub struct DotProductChip {
    config: DotProductConfig,
}

impl DotProductChip {
    /// `k` is the compile-time-known dot-product length (vector size); it is
    /// fixed per configured circuit, matching how `RangeCheckChip::configure`
    /// takes `n_bits`.
    #[allow(clippy::too_many_arguments)]
    pub fn configure(
        meta: &mut ConstraintSystem<Fr>,
        a: Column<Advice>,
        b: Column<Advice>,
        accumulator: Column<Advice>,
        q: Column<Advice>,
        r: Column<Advice>,
        slack: Column<Advice>,
        bits: Column<Advice>,
        k: usize,
    ) -> DotProductConfig {
        assert!(k > 0, "dot product length K must be positive");

        meta.enable_equality(a);
        meta.enable_equality(b);
        meta.enable_equality(accumulator);
        meta.enable_equality(q);
        meta.enable_equality(r);
        meta.enable_equality(slack);

        // Row 0: accumulator = a_0 * b_0 (base case of the running sum).
        let s_acc_start = meta.selector();
        meta.create_gate("dot product accumulation start", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let acc = meta.query_advice(accumulator, Rotation::cur());
            let s_acc_start = meta.query_selector(s_acc_start);
            vec![s_acc_start * (acc - a * b)]
        });

        // Row i (i >= 1): accumulator_i = accumulator_{i-1} + a_i * b_i.
        let s_acc_step = meta.selector();
        meta.create_gate("dot product accumulation step", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let acc_cur = meta.query_advice(accumulator, Rotation::cur());
            let acc_prev = meta.query_advice(accumulator, Rotation::prev());
            let s_acc_step = meta.query_selector(s_acc_step);
            vec![s_acc_step * (acc_cur - acc_prev - a * b)]
        });

        // At the final row (K - 1): final_accumulator = q * SCALE_18 + r,
        // the same quotient/remainder rescale gadget as EltwiseMulChip.
        let s_final = meta.selector();
        meta.create_gate("dot product final rescale", |meta| {
            let acc = meta.query_advice(accumulator, Rotation::cur());
            let q = meta.query_advice(q, Rotation::cur());
            let r = meta.query_advice(r, Rotation::cur());
            let s_final = meta.query_selector(s_final);
            let scale = Expression::Constant(i128_to_fr(SCALE_18));
            vec![s_final * (acc - q * scale - r)]
        });

        // slack = (SCALE_18 - 1) - r, enforced at the same row as s_final.
        let s_slack = meta.selector();
        meta.create_gate("dot product slack equals bound minus remainder", |meta| {
            let r = meta.query_advice(r, Rotation::cur());
            let slack = meta.query_advice(slack, Rotation::cur());
            let s_slack = meta.query_selector(s_slack);
            let bound_minus_one = Expression::Constant(i128_to_fr(SCALE_18 - 1));
            vec![s_slack * (slack + r - bound_minus_one)]
        });

        let range_q = RangeCheckChip::configure(meta, q, bits, 64);
        let range_r = RangeCheckChip::configure(meta, r, bits, REMAINDER_BITS);
        let range_r_slack = RangeCheckChip::configure(meta, slack, bits, REMAINDER_BITS);

        DotProductConfig {
            a,
            b,
            accumulator,
            q,
            r,
            slack,
            s_acc_start,
            s_acc_step,
            s_final,
            s_slack,
            range_q,
            range_r,
            range_r_slack,
            k,
        }
    }

    pub fn construct(config: DotProductConfig) -> Self {
        DotProductChip { config }
    }

    /// Assigns the dot-product region for `a` and `b` (each must have exactly
    /// the configured `K` elements), returning the requantized I18 result.
    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fr>,
        a: Vec<I18>,
        b: Vec<I18>,
    ) -> Result<I18, DotProductError> {
        let k = self.config.k;
        if a.len() != k || b.len() != k {
            return Err(DotProductError::LengthMismatch {
                expected: k,
                got_a: a.len(),
                got_b: b.len(),
            });
        }

        // Host-side computation of the running sum of raw Q36-scaled products.
        let mut raw_sum: i128 = 0;
        let mut partial_sums: Vec<i128> = Vec::with_capacity(k);
        for i in 0..k {
            let term = (a[i].raw() as i128) * (b[i].raw() as i128);
            raw_sum = raw_sum.checked_add(term).ok_or_else(|| {
                DotProductError::Overflow(FixedPointError(
                    "dot product raw accumulation overflowed i128".to_string(),
                ))
            })?;
            partial_sums.push(raw_sum);
        }

        let (q, r) = requantize_raw(raw_sum).map_err(DotProductError::Overflow)?;
        let slack = SCALE_18 - 1 - r;

        layouter.assign_region(
            || "dot product accumulation",
            |mut region| {
                for i in 0..k {
                    region.assign_advice(
                        || format!("a_{i}"),
                        self.config.a,
                        i,
                        || Value::known(i64_to_fr(a[i].raw())),
                    )?;
                    region.assign_advice(
                        || format!("b_{i}"),
                        self.config.b,
                        i,
                        || Value::known(i64_to_fr(b[i].raw())),
                    )?;
                    region.assign_advice(
                        || format!("accumulator_{i}"),
                        self.config.accumulator,
                        i,
                        || Value::known(i128_to_fr(partial_sums[i])),
                    )?;
                    if i == 0 {
                        self.config.s_acc_start.enable(&mut region, i)?;
                    } else {
                        self.config.s_acc_step.enable(&mut region, i)?;
                    }
                }

                let last = k - 1;
                self.config.s_final.enable(&mut region, last)?;
                self.config.s_slack.enable(&mut region, last)?;
                region.assign_advice(
                    || "q",
                    self.config.q,
                    last,
                    || Value::known(i64_to_fr(q.raw())),
                )?;
                region.assign_advice(
                    || "r",
                    self.config.r,
                    last,
                    || Value::known(i128_to_fr(r)),
                )?;
                region.assign_advice(
                    || "slack",
                    self.config.slack,
                    last,
                    || Value::known(i128_to_fr(slack)),
                )?;
                Ok(())
            },
        )?;

        let (q_shift_fr, q_shift_raw) = shifted_i64_witness(q.raw());
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

        Ok(q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::circuit::SimpleFloorPlanner;
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::plonk::{Circuit, ConstraintSystem, ErrorFront};

    const K: usize = 3;

    #[derive(Clone)]
    struct DotTestConfig {
        dot: DotProductConfig,
    }

    struct DotTestCircuit {
        a: Vec<I18>,
        b: Vec<I18>,
    }

    impl Circuit<Fr> for DotTestCircuit {
        type Config = DotTestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            DotTestCircuit {
                a: vec![I18::from_raw(0); K],
                b: vec![I18::from_raw(0); K],
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let a = meta.advice_column();
            let b = meta.advice_column();
            let accumulator = meta.advice_column();
            let q = meta.advice_column();
            let r = meta.advice_column();
            let slack = meta.advice_column();
            let bits = meta.advice_column();
            DotTestConfig {
                dot: DotProductChip::configure(meta, a, b, accumulator, q, r, slack, bits, K),
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = DotProductChip::construct(config.dot);
            chip.assign(layouter, self.a.clone(), self.b.clone())
                .map(|_| ())
                .map_err(|e| match e {
                    DotProductError::Circuit(err) => err,
                    other => panic!("unexpected non-circuit error in synthesize: {other}"),
                })
        }
    }

    #[test]
    fn dot_product_of_mixed_sign_length_3_vectors_is_satisfied_and_correct() {
        let a = vec![
            I18::from_f64(2.0).unwrap(),
            I18::from_f64(-3.0).unwrap(),
            I18::from_f64(1.5).unwrap(),
        ];
        let b = vec![
            I18::from_f64(3.0).unwrap(),
            I18::from_f64(2.0).unwrap(),
            I18::from_f64(-4.0).unwrap(),
        ];

        // Independently compute the expected result the same way the chip does:
        // raw Q36 accumulation, requantized once at the end.
        let raw_sum: i128 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x.raw() as i128) * (y.raw() as i128))
            .sum();
        let (expected_q, _expected_r) = requantize_raw(raw_sum).unwrap();

        let circuit = DotTestCircuit {
            a: a.clone(),
            b: b.clone(),
        };
        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        prover.assert_satisfied();

        // 2*3 + (-3)*2 + 1.5*(-4) = 6 - 6 - 6 = -6
        assert!((expected_q.to_f64() - (-6.0)).abs() < 1e-9);
    }

    #[test]
    fn dot_product_all_zero_is_satisfied() {
        let a = vec![I18::from_raw(0); K];
        let b = vec![I18::from_raw(0); K];
        let circuit = DotTestCircuit { a, b };
        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn assign_rejects_mismatched_vector_lengths() {
        // The length check happens before any layouter interaction, so we
        // drive it through a minimal Circuit whose synthesize() asserts that
        // DotProductChip::assign itself returns a LengthMismatch error (not a
        // panic) when a.len()/b.len() don't match the configured K.
        struct LenTestCircuit {
            a: Vec<I18>,
            b: Vec<I18>,
        }

        impl Circuit<Fr> for LenTestCircuit {
            type Config = DotTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                LenTestCircuit {
                    a: vec![I18::from_raw(0); K],
                    b: vec![I18::from_raw(0); K],
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                DotTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let chip = DotProductChip::construct(config.dot);
                match chip.assign(layouter, self.a.clone(), self.b.clone()) {
                    Err(DotProductError::LengthMismatch {
                        expected,
                        got_a,
                        got_b,
                    }) => {
                        assert_eq!(expected, K);
                        assert_eq!(got_a, K);
                        assert_eq!(got_b, K - 1);
                    }
                    Ok(_) => panic!("expected LengthMismatch error but assign succeeded"),
                    Err(other) => panic!("expected LengthMismatch error, got {other}"),
                }
                Ok(())
            }
        }

        let circuit = LenTestCircuit {
            a: vec![I18::from_raw(1); K],
            b: vec![I18::from_raw(1); K - 1],
        };
        // synthesize() above asserts internally; we only need to drive it.
        let _ = MockProver::run(10, &circuit, vec![]);
    }

    #[test]
    fn dot_product_with_forged_final_quotient_is_rejected() {
        struct ForgedDotCircuit {
            a: Vec<I18>,
            b: Vec<I18>,
        }

        impl Circuit<Fr> for ForgedDotCircuit {
            type Config = DotTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                ForgedDotCircuit {
                    a: vec![I18::from_raw(0); K],
                    b: vec![I18::from_raw(0); K],
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                DotTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let a = &self.a;
                let b = &self.b;
                let raw_sum: i128 = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x.raw() as i128) * (y.raw() as i128))
                    .sum();
                let mut partial_sums = Vec::with_capacity(K);
                let mut acc = 0i128;
                for i in 0..K {
                    acc += (a[i].raw() as i128) * (b[i].raw() as i128);
                    partial_sums.push(acc);
                }
                let (q, r) = requantize_raw(raw_sum).unwrap();
                let forged_q = q.raw() + 1; // violates final_accumulator == q*SCALE_18 + r
                let slack = SCALE_18 - 1 - r;

                layouter.assign_region(
                    || "forged dot product",
                    |mut region| {
                        for i in 0..K {
                            region.assign_advice(
                                || format!("a_{i}"),
                                config.dot.a,
                                i,
                                || Value::known(i64_to_fr(a[i].raw())),
                            )?;
                            region.assign_advice(
                                || format!("b_{i}"),
                                config.dot.b,
                                i,
                                || Value::known(i64_to_fr(b[i].raw())),
                            )?;
                            region.assign_advice(
                                || format!("accumulator_{i}"),
                                config.dot.accumulator,
                                i,
                                || Value::known(i128_to_fr(partial_sums[i])),
                            )?;
                            if i == 0 {
                                config.dot.s_acc_start.enable(&mut region, i)?;
                            } else {
                                config.dot.s_acc_step.enable(&mut region, i)?;
                            }
                        }
                        let last = K - 1;
                        config.dot.s_final.enable(&mut region, last)?;
                        config.dot.s_slack.enable(&mut region, last)?;
                        region.assign_advice(
                            || "q",
                            config.dot.q,
                            last,
                            || Value::known(i64_to_fr(forged_q)),
                        )?;
                        region.assign_advice(
                            || "r",
                            config.dot.r,
                            last,
                            || Value::known(i128_to_fr(r)),
                        )?;
                        region.assign_advice(
                            || "slack",
                            config.dot.slack,
                            last,
                            || Value::known(i128_to_fr(slack)),
                        )
                    },
                )?;
                Ok(())
            }
        }

        let circuit = ForgedDotCircuit {
            a: vec![
                I18::from_f64(2.0).unwrap(),
                I18::from_f64(-3.0).unwrap(),
                I18::from_f64(1.5).unwrap(),
            ],
            b: vec![
                I18::from_f64(3.0).unwrap(),
                I18::from_f64(2.0).unwrap(),
                I18::from_f64(-4.0).unwrap(),
            ],
        };
        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }
}
