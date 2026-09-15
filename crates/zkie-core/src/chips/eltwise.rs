use crate::chips::lookup_range_check::{LookupRangeCheckChip, LookupRangeCheckConfig};
use crate::chips::range_check::{RangeCheckChip, RangeCheckConfig};
use crate::field_convert::{i128_to_fr, i64_to_fr, shifted_i64_witness, Fr, SIGNED_SHIFT};
use crate::fixed_point::{requantize_mul, I18, SCALE_18};
use halo2_proofs::circuit::{AssignedCell, Layouter, Region, Value};
use halo2_proofs::plonk::{Advice, Column, ConstraintSystem, ErrorFront, Expression, Selector};
use halo2_proofs::poly::Rotation;

const REMAINDER_BITS: usize = 60; // 2^60 > SCALE_18 - 1.

// NOTE ON SOUNDNESS: `RangeCheckChip::assign` witnesses its own copy of the
// value being range-checked in its own region. If a chip only computes a
// "shifted" (for signed values) or otherwise-derived value host-side, calls
// `RangeCheckChip::assign` with it, and never ties that witnessed cell back
// to the cell used in this chip's own arithmetic gates via
// `region.constrain_equal`, the range check is checking a value that is
// *disconnected* from the one actually used in the gate -- a prover could
// satisfy the gate with an out-of-range value while range-checking an
// unrelated in-range decoy. This was confirmed empirically (a MockProver
// probe that assigned a decoy `0` to the range-check regions while the main
// gate held a real, unrelated value, and the proof still verified).
//
// The fix used throughout this file: any column that needs both (a) to
// participate in this chip's own polynomial gates and (b) to be range-checked
// is assigned its *signed-shifted* representation directly in the main gate
// region (not a separate raw value), so the exact same field element used in
// the gate is the one handed to `RangeCheckChip::assign` -- and the cell
// returned by `assign_advice` in the main region is explicitly
// `region.constrain_equal`'d to the cell returned by `RangeCheckChip::assign`.
// `r`/`slack` in `EltwiseMulChip` don't need shifting (they're already
// non-negative by construction), so only the copy-constraint is needed there.

#[derive(Clone, Debug)]
pub struct EltwiseAddConfig {
    // Crate-visible (not fully private) so that composed chips such as
    // `LayerNormChip` (which needs to witness its own `a`/`b`/`c` cells
    // directly, rather than only receiving the output cell back from
    // `EltwiseAddChip::assign`, in order to link an operand to whichever
    // other chip produced it) and their tests can reach into the raw
    // region layout — mirroring `DotProductConfig`'s precedent in
    // `dot_general.rs`.
    pub(crate) a: Column<Advice>,
    pub(crate) b: Column<Advice>,
    pub(crate) c: Column<Advice>,
    pub(crate) s_add: Selector,
    pub(crate) range_a: RangeCheckConfig,
    pub(crate) range_b: RangeCheckConfig,
    pub(crate) range_c: RangeCheckConfig,
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

        // `a`, `b`, `c` hold the *shifted* (a_raw + 2^63) representation so
        // the same cells assigned here can be range-checked directly and
        // copy-constrained to the range check's cells (see module note).
        // Shifted add: (a_raw+S) + (b_raw+S) - (c_raw+S) - S == a_raw+b_raw-c_raw == 0.
        let s_add = meta.selector();
        meta.create_gate("add (shifted)", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let c = meta.query_advice(c, Rotation::cur());
            let s_add = meta.query_selector(s_add);
            let shift = Expression::Constant(i128_to_fr(SIGNED_SHIFT));
            vec![s_add * (a + b - c - shift)]
        });

        let range_a = RangeCheckChip::configure(meta, a, bits, 64);
        let range_b = RangeCheckChip::configure(meta, b, bits, 64);
        let range_c = RangeCheckChip::configure(meta, c, bits, 64);

        EltwiseAddConfig {
            a,
            b,
            c,
            s_add,
            range_a,
            range_b,
            range_c,
        }
    }

    pub fn construct(config: EltwiseAddConfig) -> Self {
        EltwiseAddChip { config }
    }

    /// Assigns `a + b` in this chip's region, range-checks each of `a`, `b`,
    /// `c`, and links every range-checked witness back to the cell used in
    /// the main gate. Returns the assigned (signed-shifted) `c` (output)
    /// cell -- composing chips (e.g. `LayerNormChip`) must
    /// `region.constrain_equal` this cell to any cell where they re-witness
    /// the same value, rather than re-assigning it disconnected from this
    /// one (see the soundness note at the top of this file).
    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fr>,
        a: I18,
        b: I18,
    ) -> Result<AssignedCell<Fr, Fr>, ErrorFront> {
        let c_raw = a.raw().checked_add(b.raw()).expect("I18 add overflow");

        let (a_shift_fr, a_shift_raw) = shifted_i64_witness(a.raw());
        let (b_shift_fr, b_shift_raw) = shifted_i64_witness(b.raw());
        let (c_shift_fr, c_shift_raw) = shifted_i64_witness(c_raw);

        let (a_cell, b_cell, c_cell) = layouter.assign_region(
            || "eltwise add",
            |mut region| {
                self.config.s_add.enable(&mut region, 0)?;
                let a_cell = region.assign_advice(|| "a", self.config.a, 0, || a_shift_fr)?;
                let b_cell = region.assign_advice(|| "b", self.config.b, 0, || b_shift_fr)?;
                let c_cell = region.assign_advice(|| "c", self.config.c, 0, || c_shift_fr)?;
                Ok((a_cell, b_cell, c_cell))
            },
        )?;

        let range_a_chip = RangeCheckChip::construct(self.config.range_a.clone());
        let a_range_cell =
            range_a_chip.assign(layouter.namespace(|| "range a"), a_shift_fr, a_shift_raw)?;

        let range_b_chip = RangeCheckChip::construct(self.config.range_b.clone());
        let b_range_cell =
            range_b_chip.assign(layouter.namespace(|| "range b"), b_shift_fr, b_shift_raw)?;

        let range_c_chip = RangeCheckChip::construct(self.config.range_c.clone());
        let c_range_cell =
            range_c_chip.assign(layouter.namespace(|| "range c"), c_shift_fr, c_shift_raw)?;

        layouter.assign_region(
            || "eltwise add range check links",
            |mut region| {
                region.constrain_equal(a_cell.cell(), a_range_cell.cell())?;
                region.constrain_equal(b_cell.cell(), b_range_cell.cell())?;
                region.constrain_equal(c_cell.cell(), c_range_cell.cell())?;
                Ok(())
            },
        )?;

        Ok(c_cell)
    }
}

#[derive(Clone, Debug)]
pub struct EltwiseMulConfig {
    // Crate-visible for the same reason as `EltwiseAddConfig`'s fields above
    // -- `LayerNormChip` needs direct access to witness its own `a`/`b`
    // operand cells so it can link them to whichever chip produced that
    // operand's value.
    pub(crate) a: Column<Advice>,
    pub(crate) b: Column<Advice>,
    pub(crate) q: Column<Advice>,
    pub(crate) r: Column<Advice>,
    pub(crate) slack: Column<Advice>,
    pub(crate) s_mul: Selector,
    pub(crate) s_slack: Selector,
    pub(crate) range_q: RangeCheckConfig,
    pub(crate) range_r: RangeCheckConfig,
    pub(crate) range_r_slack: RangeCheckConfig,
    // `a`/`b` must stay raw because `s_mul` multiplies them, so unlike
    // `EltwiseAddChip` they cannot be range-checked in place. These carry the
    // signed-shifted copy of each operand, tied to `a`/`b` by `s_shift` and
    // bounded at 64 bits, which is what bounds the operands to i64.
    pub(crate) a_shift: Column<Advice>,
    pub(crate) b_shift: Column<Advice>,
    pub(crate) s_shift: Selector,
    pub(crate) range_a: LookupRangeCheckConfig,
    pub(crate) range_b: LookupRangeCheckConfig,
}

type MulOperandShiftCells = (AssignedCell<Fr, Fr>, AssignedCell<Fr, Fr>);

/// Loads the byte table backing a multiply's operand range checks. Must be
/// called once per circuit synthesis by whoever configured the chip.
pub(crate) fn load_mul_operand_range_table(
    mul: &EltwiseMulConfig,
    mut layouter: impl Layouter<Fr>,
) -> Result<(), ErrorFront> {
    LookupRangeCheckChip::construct(mul.range_a.clone())
        .load_table(layouter.namespace(|| "mul operand byte table a"))?;
    LookupRangeCheckChip::construct(mul.range_b.clone())
        .load_table(layouter.namespace(|| "mul operand byte table b"))
}

/// Witnesses the signed-shifted copies of a multiply's operands at `offset`
/// of the caller's region, so they can be range-checked. Pairs with
/// [`link_mul_operand_ranges`], which does the checking; both are needed for
/// the bound to hold, and callers that assemble mul rows by hand (see
/// `layer_norm::assign_mul_row`) must call both.
#[allow(clippy::type_complexity)]
pub(crate) fn assign_mul_operand_shifts(
    mul: &EltwiseMulConfig,
    region: &mut Region<Fr>,
    offset: usize,
    a_val: I18,
    b_val: I18,
) -> Result<MulOperandShiftCells, ErrorFront> {
    assign_mul_operand_shifts_with_witness_mode(mul, region, offset, a_val, b_val, true)
}

pub(crate) fn assign_mul_operand_shifts_with_witness_mode(
    mul: &EltwiseMulConfig,
    region: &mut Region<Fr>,
    offset: usize,
    a_val: I18,
    b_val: I18,
    witnesses_known: bool,
) -> Result<MulOperandShiftCells, ErrorFront> {
    mul.s_shift.enable(region, offset)?;
    let (a_shift_fr, _) = shifted_i64_witness(a_val.raw());
    let (b_shift_fr, _) = shifted_i64_witness(b_val.raw());
    let a_cell = region.assign_advice(
        || "a_shift",
        mul.a_shift,
        offset,
        || witness_value(witnesses_known, a_shift_fr),
    )?;
    let b_cell = region.assign_advice(
        || "b_shift",
        mul.b_shift,
        offset,
        || witness_value(witnesses_known, b_shift_fr),
    )?;
    Ok((a_cell, b_cell))
}

/// Range-checks the shifted operands witnessed by [`assign_mul_operand_shifts`]
/// and copy-constrains the checked cells back to them. Without the copy
/// constraint the range check would bound an unrelated cell.
pub(crate) fn link_mul_operand_ranges(
    mul: &EltwiseMulConfig,
    layouter: impl Layouter<Fr>,
    a_val: I18,
    b_val: I18,
    a_shift_cell: &AssignedCell<Fr, Fr>,
    b_shift_cell: &AssignedCell<Fr, Fr>,
) -> Result<(), ErrorFront> {
    link_mul_operand_ranges_with_witness_mode(
        mul,
        layouter,
        a_val,
        b_val,
        a_shift_cell,
        b_shift_cell,
        true,
    )
}

pub(crate) fn link_mul_operand_ranges_with_witness_mode(
    mul: &EltwiseMulConfig,
    mut layouter: impl Layouter<Fr>,
    a_val: I18,
    b_val: I18,
    a_shift_cell: &AssignedCell<Fr, Fr>,
    b_shift_cell: &AssignedCell<Fr, Fr>,
    witnesses_known: bool,
) -> Result<(), ErrorFront> {
    let (a_fr, a_raw) = shifted_i64_witness(a_val.raw());
    let a_range_cell = LookupRangeCheckChip::construct(mul.range_a.clone()).assign(
        layouter.namespace(|| "range a"),
        witness_value(witnesses_known, a_fr),
        witness_value(witnesses_known, a_raw),
    )?;
    let (b_fr, b_raw) = shifted_i64_witness(b_val.raw());
    let b_range_cell = LookupRangeCheckChip::construct(mul.range_b.clone()).assign(
        layouter.namespace(|| "range b"),
        witness_value(witnesses_known, b_fr),
        witness_value(witnesses_known, b_raw),
    )?;
    layouter.assign_region(
        || "mul operand range check links",
        |mut region| {
            region.constrain_equal(a_shift_cell.cell(), a_range_cell.cell())?;
            region.constrain_equal(b_shift_cell.cell(), b_range_cell.cell())?;
            Ok(())
        },
    )
}

fn witness_value<T: Copy>(known: bool, value: Value<T>) -> Value<T> {
    if known {
        value
    } else {
        Value::unknown()
    }
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

        // `q` holds the *shifted* (q_raw + 2^63) representation (see module
        // note), so the rescale gate is written in terms of the shifted
        // quotient: a*b == (q_shift - 2^63)*SCALE_18 + r.
        let s_mul = meta.selector();
        meta.create_gate("mul rescale (shifted q)", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let q_shift = meta.query_advice(q, Rotation::cur());
            let r = meta.query_advice(r, Rotation::cur());
            let s_mul = meta.query_selector(s_mul);
            let scale = Expression::Constant(i128_to_fr(SCALE_18));
            let shift_scaled = Expression::Constant(i128_to_fr(SIGNED_SHIFT * SCALE_18));
            vec![s_mul * (a * b - (q_shift * scale.clone() - shift_scaled) - r)]
        });

        // slack = (SCALE_18 - 1) - r, enforced at the same row as s_mul's inputs.
        // `r`/`slack` are already non-negative by construction, so they don't
        // need shifting -- the raw values used here are exactly what gets
        // range-checked (only the copy-constraint link is needed for them).
        let s_slack = meta.selector();
        meta.create_gate("slack equals bound minus remainder", |meta| {
            let r = meta.query_advice(r, Rotation::cur());
            let slack = meta.query_advice(slack, Rotation::cur());
            let s_slack = meta.query_selector(s_slack);
            let bound_minus_one = Expression::Constant(i128_to_fr(SCALE_18 - 1));
            vec![s_slack * (slack + r - bound_minus_one)]
        });

        // Operand bounds. Allocated here rather than taken as parameters so the
        // fix stays local to the chip; the trade-off is two more advice columns
        // per configured multiply.
        let a_shift = meta.advice_column();
        let b_shift = meta.advice_column();
        meta.enable_equality(a_shift);
        meta.enable_equality(b_shift);

        let s_shift = meta.selector();
        meta.create_gate("mul operand shift", |meta| {
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

        // Lookup-based for the same reason as the dot product's: these are
        // per-operand, so 8 rows each instead of 64.
        let range_a = LookupRangeCheckChip::configure(meta, a_shift, bits, 64);
        let range_b = LookupRangeCheckChip::configure(meta, b_shift, bits, 64);
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
            a_shift,
            b_shift,
            s_shift,
            range_a,
            range_b,
        }
    }

    pub fn construct(config: EltwiseMulConfig) -> Self {
        EltwiseMulChip { config }
    }

    /// Assigns `a * b` (via the quotient/remainder rescale gadget) in this
    /// chip's region, range-checks `q`/`r`/`slack`, and links every
    /// range-checked witness back to the cell used in the main gate.
    /// Returns the assigned (signed-shifted) `q` (output/quotient) cell --
    /// composing chips (e.g. `LayerNormChip`) must `region.constrain_equal`
    /// this cell to any cell where they re-witness the same value, rather
    /// than re-assigning it disconnected from this one (see the soundness
    /// note at the top of this file).
    /// Loads the byte table backing the operand range checks. Must be called
    /// once per circuit synthesis, independently of [`Self::assign`].
    pub fn load_range_table(&self, layouter: impl Layouter<Fr>) -> Result<(), ErrorFront> {
        load_mul_operand_range_table(&self.config, layouter)
    }

    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fr>,
        a: I18,
        b: I18,
    ) -> Result<AssignedCell<Fr, Fr>, ErrorFront> {
        let (q, r) = requantize_mul(a, b).expect("I18 mul overflow");
        let slack = SCALE_18 - 1 - r;
        let (q_shift_fr, q_shift_raw) = shifted_i64_witness(q.raw());

        let (q_cell, r_cell, slack_cell, a_shift_cell, b_shift_cell) = layouter.assign_region(
            || "eltwise mul",
            |mut region| {
                self.config.s_mul.enable(&mut region, 0)?;
                self.config.s_slack.enable(&mut region, 0)?;
                region.assign_advice(
                    || "a",
                    self.config.a,
                    0,
                    || Value::known(i64_to_fr(a.raw())),
                )?;
                region.assign_advice(
                    || "b",
                    self.config.b,
                    0,
                    || Value::known(i64_to_fr(b.raw())),
                )?;
                let (a_shift_cell, b_shift_cell) =
                    assign_mul_operand_shifts(&self.config, &mut region, 0, a, b)?;
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
                Ok((q_cell, r_cell, slack_cell, a_shift_cell, b_shift_cell))
            },
        )?;

        link_mul_operand_ranges(
            &self.config,
            layouter.namespace(|| "mul operand ranges"),
            a,
            b,
            &a_shift_cell,
            &b_shift_cell,
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
            || "eltwise mul range check links",
            |mut region| {
                region.constrain_equal(q_cell.cell(), q_range_cell.cell())?;
                region.constrain_equal(r_cell.cell(), r_range_cell.cell())?;
                region.constrain_equal(slack_cell.cell(), slack_range_cell.cell())?;
                Ok(())
            },
        )?;

        Ok(q_cell)
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
        type Params = ();

        type Config = AddTestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            AddTestCircuit {
                a: I18::from_raw(0),
                b: I18::from_raw(0),
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let a = meta.advice_column();
            let b = meta.advice_column();
            let c = meta.advice_column();
            let bits = meta.advice_column();
            AddTestConfig {
                add: EltwiseAddChip::configure(meta, a, b, c, bits),
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = EltwiseAddChip::construct(config.add);
            chip.assign(layouter, self.a, self.b).map(|_| ())
        }
    }

    #[test]
    fn add_positive_plus_positive_satisfied() {
        let circuit = AddTestCircuit {
            a: I18::from_f64(2.0).unwrap(),
            b: I18::from_f64(3.5).unwrap(),
        };
        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn add_negative_plus_positive_satisfied() {
        let circuit = AddTestCircuit {
            a: I18::from_f64(-2.0).unwrap(),
            b: I18::from_f64(3.5).unwrap(),
        };
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
            type Params = ();

            type Config = AddTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                ForgedAddCircuit {
                    a: I18::from_raw(0),
                    b: I18::from_raw(0),
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                AddTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let (a_shift_fr, _) = shifted_i64_witness(self.a.raw());
                let (b_shift_fr, _) = shifted_i64_witness(self.b.raw());
                let forged_c = self.a.raw() + self.b.raw() + 1;
                let (c_shift_fr, _) = shifted_i64_witness(forged_c);
                layouter.assign_region(
                    || "forged add",
                    |mut region| {
                        config.add.s_add.enable(&mut region, 0)?;
                        region.assign_advice(|| "a", config.add.a, 0, || a_shift_fr)?;
                        region.assign_advice(|| "b", config.add.b, 0, || b_shift_fr)?;
                        region.assign_advice(|| "c", config.add.c, 0, || c_shift_fr)
                    },
                )?;
                Ok(())
            }
        }

        let circuit = ForgedAddCircuit {
            a: I18::from_f64(2.0).unwrap(),
            b: I18::from_f64(3.0).unwrap(),
        };
        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    fn add_constrain_equal_rejects_mismatched_range_check_witness() {
        // Regression test proving `region.constrain_equal` genuinely binds
        // the main gate's `c` cell to the range-check's cell: this synthesize
        // mimics EltwiseAddChip::assign's exact structure (main region, range
        // check region, and the link), but deliberately range-checks a decoy
        // value (0) instead of the real shifted `c`, while STILL calling
        // `constrain_equal` between the two (mismatched) cells -- exactly as
        // production code would if it accidentally computed the wrong
        // decoy/shifted value to range-check. The permutation argument must
        // catch that mismatch.
        //
        // (A synthesize that range-checks a decoy *without* calling
        // constrain_equal at all isn't a meaningful test: MockProver only
        // checks permutation ties that were actually registered during that
        // circuit's own synthesis, so omitting the call trivially "passes"
        // regardless of whether the real EltwiseAddChip links its cells --
        // it simply isn't exercising the mechanism under test.)
        struct MismatchedLinkCircuit {
            a: I18,
            b: I18,
        }

        impl Circuit<Fr> for MismatchedLinkCircuit {
            type Params = ();

            type Config = AddTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                MismatchedLinkCircuit {
                    a: I18::from_raw(0),
                    b: I18::from_raw(0),
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                AddTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let (a_shift_fr, a_shift_raw) = shifted_i64_witness(self.a.raw());
                let (b_shift_fr, b_shift_raw) = shifted_i64_witness(self.b.raw());
                let c_raw = self.a.raw() + self.b.raw();
                let (c_shift_fr, _) = shifted_i64_witness(c_raw);

                let (a_cell, b_cell, c_cell) = layouter.assign_region(
                    || "eltwise add",
                    |mut region| {
                        config.add.s_add.enable(&mut region, 0)?;
                        let a_cell =
                            region.assign_advice(|| "a", config.add.a, 0, || a_shift_fr)?;
                        let b_cell =
                            region.assign_advice(|| "b", config.add.b, 0, || b_shift_fr)?;
                        let c_cell =
                            region.assign_advice(|| "c", config.add.c, 0, || c_shift_fr)?;
                        Ok((a_cell, b_cell, c_cell))
                    },
                )?;

                let range_a_chip = RangeCheckChip::construct(config.add.range_a.clone());
                let a_range_cell = range_a_chip.assign(
                    layouter.namespace(|| "range a"),
                    a_shift_fr,
                    a_shift_raw,
                )?;
                let range_b_chip = RangeCheckChip::construct(config.add.range_b.clone());
                let b_range_cell = range_b_chip.assign(
                    layouter.namespace(|| "range b"),
                    b_shift_fr,
                    b_shift_raw,
                )?;

                // Mismatch: range-check a decoy (0) for `c` instead of the
                // real shifted c, but still link it via constrain_equal --
                // simulating a caller that wired the link correctly but
                // computed the wrong value to check.
                let (decoy_fr, decoy_raw) = shifted_i64_witness(0);
                let range_c_chip = RangeCheckChip::construct(config.add.range_c.clone());
                let c_range_cell =
                    range_c_chip.assign(layouter.namespace(|| "range c"), decoy_fr, decoy_raw)?;

                layouter.assign_region(
                    || "eltwise add range check links",
                    |mut region| {
                        region.constrain_equal(a_cell.cell(), a_range_cell.cell())?;
                        region.constrain_equal(b_cell.cell(), b_range_cell.cell())?;
                        region.constrain_equal(c_cell.cell(), c_range_cell.cell())?;
                        Ok(())
                    },
                )?;

                Ok(())
            }
        }

        let circuit = MismatchedLinkCircuit {
            a: I18::from_f64(2.0).unwrap(),
            b: I18::from_f64(3.0).unwrap(),
        };
        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        assert!(
            prover.verify().is_err(),
            "constrain_equal must reject a c cell tied to a mismatched decoy range-check value"
        );
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
        type Params = ();

        type Config = MulTestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            MulTestCircuit {
                a: I18::from_raw(0),
                b: I18::from_raw(0),
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let a = meta.advice_column();
            let b = meta.advice_column();
            let q = meta.advice_column();
            let r = meta.advice_column();
            let slack = meta.advice_column();
            let bits = meta.advice_column();
            MulTestConfig {
                mul: EltwiseMulChip::configure(meta, a, b, q, r, slack, bits),
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = EltwiseMulChip::construct(config.mul);
            chip.load_range_table(layouter.namespace(|| "range tables"))?;
            chip.assign(layouter, self.a, self.b).map(|_| ())
        }
    }

    #[test]
    fn mul_positive_times_positive_satisfied() {
        let circuit = MulTestCircuit {
            a: I18::from_f64(2.0).unwrap(),
            b: I18::from_f64(3.0).unwrap(),
        };
        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn mul_negative_times_positive_satisfied() {
        let circuit = MulTestCircuit {
            a: I18::from_f64(-2.5).unwrap(),
            b: I18::from_f64(2.0).unwrap(),
        };
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
            type Params = ();

            type Config = MulTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                ForgedMulCircuit {
                    a: I18::from_raw(0),
                    b: I18::from_raw(0),
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                MulTestCircuit::configure(meta)
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
                LookupRangeCheckChip::construct(config.mul.range_a.clone())
                    .load_table(layouter.namespace(|| "byte table a"))?;
                LookupRangeCheckChip::construct(config.mul.range_b.clone())
                    .load_table(layouter.namespace(|| "byte table b"))?;
                let (q, r) = requantize_mul(self.a, self.b).unwrap();
                let forged_q = q.raw() + 1; // violates a*b == q*SCALE_18 + r
                let (forged_q_shift_fr, _) = shifted_i64_witness(forged_q);
                layouter.assign_region(
                    || "forged mul",
                    |mut region| {
                        config.mul.s_mul.enable(&mut region, 0)?;
                        config.mul.s_slack.enable(&mut region, 0)?;
                        region.assign_advice(
                            || "a",
                            config.mul.a,
                            0,
                            || Value::known(i64_to_fr(self.a.raw())),
                        )?;
                        region.assign_advice(
                            || "b",
                            config.mul.b,
                            0,
                            || Value::known(i64_to_fr(self.b.raw())),
                        )?;
                        region.assign_advice(|| "q", config.mul.q, 0, || forged_q_shift_fr)?;
                        region.assign_advice(
                            || "r",
                            config.mul.r,
                            0,
                            || Value::known(i128_to_fr(r)),
                        )?;
                        let slack = SCALE_18 - 1 - r;
                        region.assign_advice(
                            || "slack",
                            config.mul.slack,
                            0,
                            || Value::known(i128_to_fr(slack)),
                        )
                    },
                )?;
                Ok(())
            }
        }

        let circuit = ForgedMulCircuit {
            a: I18::from_f64(2.0).unwrap(),
            b: I18::from_f64(3.0).unwrap(),
        };
        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    fn mul_constrain_equal_rejects_mismatched_range_check_witness() {
        // Regression test proving `region.constrain_equal` genuinely binds
        // the main gate's `q` cell to the range-check's cell (see the
        // analogous add-chip test's doc comment for why the link must
        // actually be established, not merely omitted, to be a meaningful
        // probe of this mechanism).
        struct MismatchedLinkCircuit {
            a: I18,
            b: I18,
        }

        impl Circuit<Fr> for MismatchedLinkCircuit {
            type Params = ();

            type Config = MulTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                MismatchedLinkCircuit {
                    a: I18::from_raw(0),
                    b: I18::from_raw(0),
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                MulTestCircuit::configure(meta)
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
                LookupRangeCheckChip::construct(config.mul.range_a.clone())
                    .load_table(layouter.namespace(|| "byte table a"))?;
                LookupRangeCheckChip::construct(config.mul.range_b.clone())
                    .load_table(layouter.namespace(|| "byte table b"))?;
                let (q, r) = requantize_mul(self.a, self.b).unwrap();
                let slack = SCALE_18 - 1 - r;
                let (q_shift_fr, _) = shifted_i64_witness(q.raw());

                let (q_cell, r_cell, slack_cell) = layouter.assign_region(
                    || "eltwise mul",
                    |mut region| {
                        config.mul.s_mul.enable(&mut region, 0)?;
                        config.mul.s_slack.enable(&mut region, 0)?;
                        region.assign_advice(
                            || "a",
                            config.mul.a,
                            0,
                            || Value::known(i64_to_fr(self.a.raw())),
                        )?;
                        region.assign_advice(
                            || "b",
                            config.mul.b,
                            0,
                            || Value::known(i64_to_fr(self.b.raw())),
                        )?;
                        let q_cell =
                            region.assign_advice(|| "q", config.mul.q, 0, || q_shift_fr)?;
                        let r_cell = region.assign_advice(
                            || "r",
                            config.mul.r,
                            0,
                            || Value::known(i128_to_fr(r)),
                        )?;
                        let slack_cell = region.assign_advice(
                            || "slack",
                            config.mul.slack,
                            0,
                            || Value::known(i128_to_fr(slack)),
                        )?;
                        Ok((q_cell, r_cell, slack_cell))
                    },
                )?;

                let range_r_chip = RangeCheckChip::construct(config.mul.range_r.clone());
                let r_range_cell = range_r_chip.assign(
                    layouter.namespace(|| "range r"),
                    Value::known(i128_to_fr(r)),
                    Value::known(r),
                )?;
                let range_r_slack_chip =
                    RangeCheckChip::construct(config.mul.range_r_slack.clone());
                let slack_range_cell = range_r_slack_chip.assign(
                    layouter.namespace(|| "range r slack"),
                    Value::known(i128_to_fr(slack)),
                    Value::known(slack),
                )?;

                // Mismatch: range-check a decoy (0) for `q` instead of the
                // real shifted q, but still link it via constrain_equal.
                let (decoy_fr, decoy_raw) = shifted_i64_witness(0);
                let range_q_chip = RangeCheckChip::construct(config.mul.range_q.clone());
                let q_range_cell =
                    range_q_chip.assign(layouter.namespace(|| "range q"), decoy_fr, decoy_raw)?;

                layouter.assign_region(
                    || "eltwise mul range check links",
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
            a: I18::from_f64(2.0).unwrap(),
            b: I18::from_f64(3.0).unwrap(),
        };
        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        assert!(
            prover.verify().is_err(),
            "constrain_equal must reject a q cell tied to a mismatched decoy range-check value"
        );
    }

    #[test]
    fn mul_rejects_out_of_range_operand() {
        // Same gap as `dot_product_rejects_out_of_range_operand`: `a` sits
        // above i64::MAX while q = 10, r = 0 and slack = SCALE_18 - 1 all stay
        // in range, so only a bound on the operands themselves catches it.
        // The region mirrors the real assignment, selectors included, since
        // selectors are fixed columns a prover cannot switch off.
        #[derive(Clone, Copy, Debug)]
        enum Forge {
            HonestShift,
            ShiftIntoRange,
            DisconnectedRangeWitness,
        }

        struct OutOfRangeMulCircuit {
            a_raw: i128,
            b_raw: i128,
            forge: Forge,
        }

        impl Circuit<Fr> for OutOfRangeMulCircuit {
            type Params = ();

            type Config = MulTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                OutOfRangeMulCircuit {
                    a_raw: 0,
                    b_raw: 0,
                    forge: Forge::HonestShift,
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                MulTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let product = self.a_raw * self.b_raw;
                let q = product.div_euclid(SCALE_18);
                let r = product.rem_euclid(SCALE_18);
                let slack = SCALE_18 - 1 - r;
                let (q_shift_fr, q_shift_raw) = shifted_i64_witness(q as i64);

                let shifted = |v: i128| -> i128 {
                    match self.forge {
                        Forge::ShiftIntoRange => 0,
                        _ => v + SIGNED_SHIFT,
                    }
                };
                let range_witness = |v: i128| -> i128 {
                    match self.forge {
                        Forge::DisconnectedRangeWitness => 0,
                        _ => shifted(v),
                    }
                };

                let (q_cell, r_cell, slack_cell, a_shift_cell, b_shift_cell) = layouter
                    .assign_region(
                        || "out of range mul",
                        |mut region| {
                            config.mul.s_mul.enable(&mut region, 0)?;
                            config.mul.s_slack.enable(&mut region, 0)?;
                            config.mul.s_shift.enable(&mut region, 0)?;
                            region.assign_advice(
                                || "a",
                                config.mul.a,
                                0,
                                || Value::known(i128_to_fr(self.a_raw)),
                            )?;
                            region.assign_advice(
                                || "b",
                                config.mul.b,
                                0,
                                || Value::known(i128_to_fr(self.b_raw)),
                            )?;
                            let a_shift_cell = region.assign_advice(
                                || "a_shift",
                                config.mul.a_shift,
                                0,
                                || Value::known(i128_to_fr(shifted(self.a_raw))),
                            )?;
                            let b_shift_cell = region.assign_advice(
                                || "b_shift",
                                config.mul.b_shift,
                                0,
                                || Value::known(i128_to_fr(shifted(self.b_raw))),
                            )?;
                            let q_cell =
                                region.assign_advice(|| "q", config.mul.q, 0, || q_shift_fr)?;
                            let r_cell = region.assign_advice(
                                || "r",
                                config.mul.r,
                                0,
                                || Value::known(i128_to_fr(r)),
                            )?;
                            let slack_cell = region.assign_advice(
                                || "slack",
                                config.mul.slack,
                                0,
                                || Value::known(i128_to_fr(slack)),
                            )?;
                            Ok((q_cell, r_cell, slack_cell, a_shift_cell, b_shift_cell))
                        },
                    )?;

                // Without the byte tables the lookups fail for lack of a
                // table and the circuit would be rejected for the wrong reason.
                load_mul_operand_range_table(
                    &config.mul,
                    layouter.namespace(|| "mul byte tables"),
                )?;

                let a_s = range_witness(self.a_raw);
                let a_range_cell = LookupRangeCheckChip::construct(config.mul.range_a.clone())
                    .assign(
                        layouter.namespace(|| "range a"),
                        Value::known(i128_to_fr(a_s)),
                        Value::known(a_s),
                    )?;
                let b_s = range_witness(self.b_raw);
                let b_range_cell = LookupRangeCheckChip::construct(config.mul.range_b.clone())
                    .assign(
                        layouter.namespace(|| "range b"),
                        Value::known(i128_to_fr(b_s)),
                        Value::known(b_s),
                    )?;
                let q_range_cell = RangeCheckChip::construct(config.mul.range_q.clone()).assign(
                    layouter.namespace(|| "range q"),
                    q_shift_fr,
                    q_shift_raw,
                )?;
                let r_range_cell = RangeCheckChip::construct(config.mul.range_r.clone()).assign(
                    layouter.namespace(|| "range r"),
                    Value::known(i128_to_fr(r)),
                    Value::known(r),
                )?;
                let slack_range_cell = RangeCheckChip::construct(config.mul.range_r_slack.clone())
                    .assign(
                        layouter.namespace(|| "range r slack"),
                        Value::known(i128_to_fr(slack)),
                        Value::known(slack),
                    )?;

                layouter.assign_region(
                    || "out of range mul links",
                    |mut region| {
                        region.constrain_equal(a_shift_cell.cell(), a_range_cell.cell())?;
                        region.constrain_equal(b_shift_cell.cell(), b_range_cell.cell())?;
                        region.constrain_equal(q_cell.cell(), q_range_cell.cell())?;
                        region.constrain_equal(r_cell.cell(), r_range_cell.cell())?;
                        region.constrain_equal(slack_cell.cell(), slack_range_cell.cell())?;
                        Ok(())
                    },
                )?;
                Ok(())
            }
        }

        // 10 * SCALE_18 = 1e19, above i64::MAX ~= 9.22e18. b = 1 keeps the
        // rescale witnesses small: q = 10, r = 0.
        for forge in [
            Forge::HonestShift,
            Forge::ShiftIntoRange,
            Forge::DisconnectedRangeWitness,
        ] {
            let circuit = OutOfRangeMulCircuit {
                a_raw: 10 * SCALE_18,
                b_raw: 1,
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
