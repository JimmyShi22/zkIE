//! `DotProductChip`: the atomic per-output-element primitive backing the
//! `DOT_GENERAL` instruction. Computes `result = sum_{i=0}^{K-1}(a_i * b_i)`
//! over length-K vectors of I18 values, requantized back to I18.
//!
//! Full M x N x K matrix tiling is out of scope here: a later ONNX-compiler
//! sub-project will instantiate this chip once per output element.

use crate::chips::lookup_range_check::{LookupRangeCheckChip, LookupRangeCheckConfig};
use crate::chips::range_check::{RangeCheckChip, RangeCheckConfig};
use crate::field_convert::{i128_to_fr, i64_to_fr, shifted_i64_witness, Fr, SIGNED_SHIFT};
use crate::fixed_point::{requantize_raw, FixedPointError, I18, SCALE_18};
use halo2_proofs::circuit::{Layouter, Value};
use halo2_proofs::plonk::{Advice, Column, ConstraintSystem, ErrorFront, Expression, Selector};
use halo2_proofs::poly::Rotation;
use std::fmt;

// 2^60 > SCALE_18 - 1, so 60 bits is enough to bound a remainder in [0, SCALE_18).
const REMAINDER_BITS: usize = 60;

// See the soundness note in chips/eltwise.rs: `q` holds the *signed-shifted*
// representation directly so it can be range-checked and then
// `region.constrain_equal`'d back to the cell used in the `s_final` gate --
// `RangeCheckChip::assign` on its own only witnesses a disconnected cell in
// its own region. `r`/`slack` don't need shifting (non-negative by
// construction) so only the copy-constraint link is needed for them.

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
    // Crate-visible (not fully private) so that composed chips such as
    // `PatchEmbedChip` (which reuses a single configured `DotProductChip`
    // across several regions) and their tests can reach into the raw region
    // layout, e.g. to forge a witness in one region for a negative test —
    // mirroring the pattern used in this module's own `tests` submodule.
    pub(crate) a: Column<Advice>,
    pub(crate) b: Column<Advice>,
    pub(crate) accumulator: Column<Advice>,
    pub(crate) q: Column<Advice>,
    pub(crate) r: Column<Advice>,
    pub(crate) slack: Column<Advice>,
    pub(crate) s_acc_start: Selector,
    pub(crate) s_acc_step: Selector,
    pub(crate) s_final: Selector,
    pub(crate) s_slack: Selector,
    // Crate-visible for the same reason as the columns/selectors above --
    // `AssemblerChip` (see `crate::assembler`) needs to build its own fully
    // per-element-linked dot-product rows (mirroring `LayerNormChip`'s
    // `assign_add_row`/`assign_mul_row` precedent for `EltwiseAddConfig`/
    // `EltwiseMulConfig`), which requires witnessing the range checks
    // directly against these configs rather than only through
    // `DotProductChip::assign`'s black-box API.
    pub(crate) range_q: RangeCheckConfig,
    pub(crate) range_r: RangeCheckConfig,
    pub(crate) range_r_slack: RangeCheckConfig,
    // `a`/`b` must hold *raw* values because the accumulator gate multiplies
    // them, so unlike `EltwiseAddChip` they cannot be range-checked in place.
    // These columns carry the signed-shifted copy of each operand, tied to
    // `a`/`b` by `s_shift` and range-checked at 64 bits, which is what bounds
    // the operands themselves to i64.
    pub(crate) a_shift: Column<Advice>,
    pub(crate) b_shift: Column<Advice>,
    pub(crate) s_shift: Selector,
    pub(crate) range_a: LookupRangeCheckConfig,
    pub(crate) range_b: LookupRangeCheckConfig,
    pub(crate) k: usize,
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
        // the same quotient/remainder rescale gadget as EltwiseMulChip, with
        // `q` held in its shifted representation (see module note).
        let s_final = meta.selector();
        meta.create_gate("dot product final rescale (shifted q)", |meta| {
            let acc = meta.query_advice(accumulator, Rotation::cur());
            let q_shift = meta.query_advice(q, Rotation::cur());
            let r = meta.query_advice(r, Rotation::cur());
            let s_final = meta.query_selector(s_final);
            let scale = Expression::Constant(i128_to_fr(SCALE_18));
            let shift_scaled = Expression::Constant(i128_to_fr(SIGNED_SHIFT * SCALE_18));
            vec![s_final * (acc - (q_shift * scale.clone() - shift_scaled) - r)]
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

        // Operand bounds. Allocated here rather than taken as parameters so the
        // fix stays local to the chip; the trade-off is two more advice columns
        // per configured dot product.
        let a_shift = meta.advice_column();
        let b_shift = meta.advice_column();
        meta.enable_equality(a_shift);
        meta.enable_equality(b_shift);

        let s_shift = meta.selector();
        meta.create_gate("dot product operand shift", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let a_shift = meta.query_advice(a_shift, Rotation::cur());
            let b_shift = meta.query_advice(b_shift, Rotation::cur());
            let s_shift = meta.query_selector(s_shift);
            let shift = Expression::Constant(i128_to_fr(SIGNED_SHIFT));
            vec![
                s_shift.clone() * (a_shift - a - shift.clone()),
                s_shift * (b_shift - b - shift),
            ]
        });

        // Lookup-based, not bit decomposition: there are 2k of these per
        // instance, so 8 rows each instead of 64 is the difference between a
        // usable bound and an unusable one. See `chips::lookup_range_check`.
        let range_a = LookupRangeCheckChip::configure(meta, a_shift, bits, 64);
        let range_b = LookupRangeCheckChip::configure(meta, b_shift, bits, 64);
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
            a_shift,
            b_shift,
            s_shift,
            range_a,
            range_b,
            k,
        }
    }

    pub fn construct(config: DotProductConfig) -> Self {
        DotProductChip { config }
    }

    /// Assigns the dot-product region for `a` and `b` (each must have exactly
    /// the configured `K` elements), returning the requantized I18 result.
    /// Loads the byte table backing the operand range checks. Must be called
    /// once per circuit synthesis, independently of [`Self::assign`].
    pub fn load_range_table(&self, mut layouter: impl Layouter<Fr>) -> Result<(), ErrorFront> {
        LookupRangeCheckChip::construct(self.config.range_a.clone())
            .load_table(layouter.namespace(|| "dot operand byte table a"))?;
        LookupRangeCheckChip::construct(self.config.range_b.clone())
            .load_table(layouter.namespace(|| "dot operand byte table b"))
    }

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
        let (q_shift_fr, q_shift_raw) = shifted_i64_witness(q.raw());

        let (q_cell, r_cell, slack_cell, a_shift_cells, b_shift_cells) = layouter.assign_region(
            || "dot product accumulation",
            |mut region| {
                let mut a_shift_cells = Vec::with_capacity(k);
                let mut b_shift_cells = Vec::with_capacity(k);
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
                    self.config.s_shift.enable(&mut region, i)?;
                    let (a_shift_fr, _) = shifted_i64_witness(a[i].raw());
                    let (b_shift_fr, _) = shifted_i64_witness(b[i].raw());
                    a_shift_cells.push(region.assign_advice(
                        || format!("a_shift_{i}"),
                        self.config.a_shift,
                        i,
                        || a_shift_fr,
                    )?);
                    b_shift_cells.push(region.assign_advice(
                        || format!("b_shift_{i}"),
                        self.config.b_shift,
                        i,
                        || b_shift_fr,
                    )?);
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
                let q_cell = region.assign_advice(|| "q", self.config.q, last, || q_shift_fr)?;
                let r_cell = region.assign_advice(
                    || "r",
                    self.config.r,
                    last,
                    || Value::known(i128_to_fr(r)),
                )?;
                let slack_cell = region.assign_advice(
                    || "slack",
                    self.config.slack,
                    last,
                    || Value::known(i128_to_fr(slack)),
                )?;
                Ok((q_cell, r_cell, slack_cell, a_shift_cells, b_shift_cells))
            },
        )?;

        // Bound each operand to i64 via its shifted copy.
        let range_a_chip = LookupRangeCheckChip::construct(self.config.range_a.clone());
        let range_b_chip = LookupRangeCheckChip::construct(self.config.range_b.clone());
        let mut operand_links = Vec::with_capacity(2 * k);
        for i in 0..k {
            let (a_shift_fr, a_shift_raw) = shifted_i64_witness(a[i].raw());
            let cell = range_a_chip.assign(
                layouter.namespace(|| format!("range a_{i}")),
                a_shift_fr,
                a_shift_raw,
            )?;
            operand_links.push((a_shift_cells[i].cell(), cell.cell()));

            let (b_shift_fr, b_shift_raw) = shifted_i64_witness(b[i].raw());
            let cell = range_b_chip.assign(
                layouter.namespace(|| format!("range b_{i}")),
                b_shift_fr,
                b_shift_raw,
            )?;
            operand_links.push((b_shift_cells[i].cell(), cell.cell()));
        }

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
            || "dot product range check links",
            |mut region| {
                region.constrain_equal(q_cell.cell(), q_range_cell.cell())?;
                region.constrain_equal(r_cell.cell(), r_range_cell.cell())?;
                region.constrain_equal(slack_cell.cell(), slack_range_cell.cell())?;
                for (lhs, rhs) in &operand_links {
                    region.constrain_equal(*lhs, *rhs)?;
                }
                Ok(())
            },
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
        type Params = ();

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
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = DotProductChip::construct(config.dot);
            chip.load_range_table(layouter.namespace(|| "range tables"))?;
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
            type Params = ();

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
            type Params = ();

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
                // Load the byte tables the operand range checks look up against.
                // Without them the lookup fails for lack of a table and the circuit
                // is rejected for the wrong reason, so the forgery below would never
                // be the thing this test exercises.
                LookupRangeCheckChip::construct(config.dot.range_a.clone())
                    .load_table(layouter.namespace(|| "byte table a"))?;
                LookupRangeCheckChip::construct(config.dot.range_b.clone())
                    .load_table(layouter.namespace(|| "byte table b"))?;
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
                let (forged_q_shift_fr, _) = shifted_i64_witness(forged_q);
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
                        region.assign_advice(|| "q", config.dot.q, last, || forged_q_shift_fr)?;
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

    #[test]
    fn dot_product_constrain_equal_rejects_mismatched_range_check_witness() {
        // See the analogous test in chips/eltwise.rs for why this must
        // actually call `constrain_equal` (with mismatched values) rather
        // than merely omit the link, to be a meaningful probe.
        struct MismatchedLinkCircuit {
            a: Vec<I18>,
            b: Vec<I18>,
        }

        impl Circuit<Fr> for MismatchedLinkCircuit {
            type Params = ();

            type Config = DotTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                MismatchedLinkCircuit {
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
                // Load the byte tables the operand range checks look up against.
                // Without them the lookup fails for lack of a table and the circuit
                // is rejected for the wrong reason, so the forgery below would never
                // be the thing this test exercises.
                LookupRangeCheckChip::construct(config.dot.range_a.clone())
                    .load_table(layouter.namespace(|| "byte table a"))?;
                LookupRangeCheckChip::construct(config.dot.range_b.clone())
                    .load_table(layouter.namespace(|| "byte table b"))?;
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
                let slack = SCALE_18 - 1 - r;
                let (q_shift_fr, _) = shifted_i64_witness(q.raw());

                let (q_cell, r_cell, slack_cell) = layouter.assign_region(
                    || "dot product accumulation",
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
                        let q_cell =
                            region.assign_advice(|| "q", config.dot.q, last, || q_shift_fr)?;
                        let r_cell = region.assign_advice(
                            || "r",
                            config.dot.r,
                            last,
                            || Value::known(i128_to_fr(r)),
                        )?;
                        let slack_cell = region.assign_advice(
                            || "slack",
                            config.dot.slack,
                            last,
                            || Value::known(i128_to_fr(slack)),
                        )?;
                        Ok((q_cell, r_cell, slack_cell))
                    },
                )?;

                let range_r_chip = RangeCheckChip::construct(config.dot.range_r.clone());
                let r_range_cell = range_r_chip.assign(
                    layouter.namespace(|| "range r"),
                    Value::known(i128_to_fr(r)),
                    Value::known(r),
                )?;
                let range_r_slack_chip =
                    RangeCheckChip::construct(config.dot.range_r_slack.clone());
                let slack_range_cell = range_r_slack_chip.assign(
                    layouter.namespace(|| "range r slack"),
                    Value::known(i128_to_fr(slack)),
                    Value::known(slack),
                )?;

                // Mismatch: range-check a decoy (0) for `q` instead of the
                // real shifted q, but still link it via constrain_equal.
                let (decoy_fr, decoy_raw) = shifted_i64_witness(0);
                let range_q_chip = RangeCheckChip::construct(config.dot.range_q.clone());
                let q_range_cell =
                    range_q_chip.assign(layouter.namespace(|| "range q"), decoy_fr, decoy_raw)?;

                layouter.assign_region(
                    || "dot product range check links",
                    |mut region| {
                        region.constrain_equal(q_cell.cell(), q_range_cell.cell())?;
                        region.constrain_equal(r_cell.cell(), r_range_cell.cell())?;
                        region.constrain_equal(slack_cell.cell(), slack_range_cell.cell())?;
                        Ok(())
                    },
                )?;

                Ok(())
            }
        }

        let circuit = MismatchedLinkCircuit {
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
        assert!(
            prover.verify().is_err(),
            "constrain_equal must reject a q cell tied to a mismatched decoy range-check value"
        );
    }

    #[test]
    fn dot_product_rejects_out_of_range_operand() {
        // `a_0` sits above i64::MAX, so it is not a representable I18 at all,
        // yet every witness the circuit checks stays in range: q = 10, r = 0,
        // slack = SCALE_18 - 1. Only a bound on the operands catches it.
        //
        // The region mirrors `DotProductChip::assign` exactly, selectors
        // included -- selectors live in fixed columns and are pinned by the
        // verifying key, so a prover cannot switch one off. That leaves two
        // ways to witness the operand, and all three must be rejected:
        // shift it honestly and the 64-bit range check fails; shift it into
        // range and the `s_shift` gate fails; witness the range check against
        // an unrelated in-range value and the copy constraint fails.
        #[derive(Clone, Copy, Debug)]
        enum Forge {
            HonestShift,
            ShiftIntoRange,
            DisconnectedRangeWitness,
        }

        struct OutOfRangeOperandCircuit {
            a_raw: Vec<i128>,
            b_raw: Vec<i128>,
            forge: Forge,
        }

        impl Circuit<Fr> for OutOfRangeOperandCircuit {
            type Params = ();

            type Config = DotTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                OutOfRangeOperandCircuit {
                    a_raw: vec![0; K],
                    b_raw: vec![0; K],
                    forge: Forge::HonestShift,
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
                let mut raw_sum: i128 = 0;
                let mut partial_sums = Vec::with_capacity(K);
                for (a, b) in self.a_raw.iter().zip(self.b_raw.iter()) {
                    raw_sum += a * b;
                    partial_sums.push(raw_sum);
                }
                let (q, r) = requantize_raw(raw_sum).unwrap();
                let slack = SCALE_18 - 1 - r;
                let (q_shift_fr, q_shift_raw) = shifted_i64_witness(q.raw());

                // The shifted operand the prover puts on the row. Shifting
                // honestly overflows 64 bits; the alternative is to lie.
                let shifted = |v: i128| -> i128 {
                    match self.forge {
                        Forge::ShiftIntoRange => 0,
                        _ => v + SIGNED_SHIFT,
                    }
                };
                // What the prover feeds the range check, which need not be
                // what sits on the row unless the copy constraint says so.
                let range_witness = |v: i128| -> i128 {
                    match self.forge {
                        Forge::DisconnectedRangeWitness => 0,
                        _ => shifted(v),
                    }
                };

                let (q_cell, r_cell, slack_cell, a_shift_cells, b_shift_cells) = layouter
                    .assign_region(
                        || "out of range operand",
                        |mut region| {
                            let mut a_shift_cells = Vec::with_capacity(K);
                            let mut b_shift_cells = Vec::with_capacity(K);
                            for (i, partial) in partial_sums.iter().enumerate() {
                                region.assign_advice(
                                    || format!("a_{i}"),
                                    config.dot.a,
                                    i,
                                    || Value::known(i128_to_fr(self.a_raw[i])),
                                )?;
                                region.assign_advice(
                                    || format!("b_{i}"),
                                    config.dot.b,
                                    i,
                                    || Value::known(i128_to_fr(self.b_raw[i])),
                                )?;
                                config.dot.s_shift.enable(&mut region, i)?;
                                a_shift_cells.push(region.assign_advice(
                                    || format!("a_shift_{i}"),
                                    config.dot.a_shift,
                                    i,
                                    || Value::known(i128_to_fr(shifted(self.a_raw[i]))),
                                )?);
                                b_shift_cells.push(region.assign_advice(
                                    || format!("b_shift_{i}"),
                                    config.dot.b_shift,
                                    i,
                                    || Value::known(i128_to_fr(shifted(self.b_raw[i]))),
                                )?);
                                region.assign_advice(
                                    || format!("accumulator_{i}"),
                                    config.dot.accumulator,
                                    i,
                                    || Value::known(i128_to_fr(*partial)),
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
                            let q_cell =
                                region.assign_advice(|| "q", config.dot.q, last, || q_shift_fr)?;
                            let r_cell = region.assign_advice(
                                || "r",
                                config.dot.r,
                                last,
                                || Value::known(i128_to_fr(r)),
                            )?;
                            let slack_cell = region.assign_advice(
                                || "slack",
                                config.dot.slack,
                                last,
                                || Value::known(i128_to_fr(slack)),
                            )?;
                            Ok((q_cell, r_cell, slack_cell, a_shift_cells, b_shift_cells))
                        },
                    )?;

                let range_a_chip = LookupRangeCheckChip::construct(config.dot.range_a.clone());
                let range_b_chip = LookupRangeCheckChip::construct(config.dot.range_b.clone());
                // Without this the lookup fails for lack of a table and the
                // circuit would be rejected for the wrong reason.
                range_a_chip.load_table(layouter.namespace(|| "byte table a"))?;
                range_b_chip.load_table(layouter.namespace(|| "byte table b"))?;
                let mut operand_links = Vec::with_capacity(2 * K);
                for i in 0..K {
                    let a_s = range_witness(self.a_raw[i]);
                    let cell = range_a_chip.assign(
                        layouter.namespace(|| format!("range a_{i}")),
                        Value::known(i128_to_fr(a_s)),
                        Value::known(a_s),
                    )?;
                    operand_links.push((a_shift_cells[i].cell(), cell.cell()));
                    let b_s = range_witness(self.b_raw[i]);
                    let cell = range_b_chip.assign(
                        layouter.namespace(|| format!("range b_{i}")),
                        Value::known(i128_to_fr(b_s)),
                        Value::known(b_s),
                    )?;
                    operand_links.push((b_shift_cells[i].cell(), cell.cell()));
                }

                let range_q_chip = RangeCheckChip::construct(config.dot.range_q.clone());
                let q_range_cell = range_q_chip.assign(
                    layouter.namespace(|| "range q"),
                    q_shift_fr,
                    q_shift_raw,
                )?;
                let range_r_chip = RangeCheckChip::construct(config.dot.range_r.clone());
                let r_range_cell = range_r_chip.assign(
                    layouter.namespace(|| "range r"),
                    Value::known(i128_to_fr(r)),
                    Value::known(r),
                )?;
                let range_slack_chip = RangeCheckChip::construct(config.dot.range_r_slack.clone());
                let slack_range_cell = range_slack_chip.assign(
                    layouter.namespace(|| "range r slack"),
                    Value::known(i128_to_fr(slack)),
                    Value::known(slack),
                )?;

                layouter.assign_region(
                    || "out of range operand links",
                    |mut region| {
                        region.constrain_equal(q_cell.cell(), q_range_cell.cell())?;
                        region.constrain_equal(r_cell.cell(), r_range_cell.cell())?;
                        region.constrain_equal(slack_cell.cell(), slack_range_cell.cell())?;
                        for (lhs, rhs) in &operand_links {
                            region.constrain_equal(*lhs, *rhs)?;
                        }
                        Ok(())
                    },
                )?;
                Ok(())
            }
        }

        // 10 * SCALE_18 = 1e19, above i64::MAX ~= 9.22e18. b_0 = 1 keeps the
        // rescale witnesses small: q = 10, r = 0.
        for forge in [
            Forge::HonestShift,
            Forge::ShiftIntoRange,
            Forge::DisconnectedRangeWitness,
        ] {
            let circuit = OutOfRangeOperandCircuit {
                a_raw: vec![10 * SCALE_18, 0, 0],
                b_raw: vec![1, 0, 0],
                forge,
            };
            let prover = MockProver::run(11, &circuit, vec![]).unwrap();
            assert!(
                prover.verify().is_err(),
                "an operand above i64::MAX must not satisfy the circuit ({forge:?})"
            );
        }
    }
}
