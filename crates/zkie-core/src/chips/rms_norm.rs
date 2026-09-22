//! `RmsNormChip`: backs a new `Instruction::RmsNorm` (see `crate::isa`) --
//! RMS normalization (as used by TimesFM's `RMSNorm` class, not the
//! mean-centering `LayerNorm`) over a fixed, compile-time-known count `k` of
//! I18 inputs, with a real per-channel learned scale (`weight`):
//!
//! ```text
//! mean_sq  = (1/k) * sum_i x_i^2
//! rsqrt_v  = rsqrt(mean_sq + epsilon)
//! output_i = x_i * rsqrt_v * weight_i
//! ```
//!
//! This is the exact math TimesFM's own `RMSNorm._norm` +
//! `RMSNorm.forward` (with `add_unit_offset=False`, the case
//! `TimesFMDecoderLayer.input_layernorm` uses) computes: `x *
//! rsqrt(mean(x^2, -1) + eps) * weight`. See
//! `docs/superpowers/specs/2026-07-26-zkie-smaller-timesfm-attempt.md` for
//! where this is wired into the real op-mapper/assembler pipeline against a
//! real exported TimesFM-architecture ONNX graph.
//!
//! ## Relationship to `LayerNormChip`
//!
//! [`crate::chips::layer_norm::LayerNormChip`] computes real `LayerNorm`
//! (mean-centered, no learnable affine). RMSNorm is numerically *simpler*
//! (no mean-centering step at all -- `mean_sq` is the mean of `x_i^2`
//! directly, not of `(x_i - mean)^2`) but *adds back* a real per-channel
//! learned `weight` multiply that `LayerNormChip` deliberately scopes out.
//! Rather than force one chip to serve both shapes (and risk a subtly wrong
//! fusion), this is a distinct chip that reuses the same already-sound
//! building blocks (`ReduceMeanChip`, `EltwiseMulChip`, `EltwiseAddChip`,
//! `RsqrtChip` -- the last imported directly from `chips::layer_norm`, which
//! is where it's defined) in a shorter pipeline, following the exact same
//! composition/soundness discipline `LayerNormChip` established.
//!
//! ## Soundness: linking cells across composed-chip boundaries
//!
//! Exactly the same discipline as `LayerNormChip` (see that module's
//! top-level docs): every intermediate value that crosses a sub-chip
//! boundary is both recomputed host-side *and* tied back to the producing
//! chip's real cell via `region.constrain_equal`, bridging representations
//! (`shifted` vs `unshifted`, see `chips/eltwise.rs`'s module note) with the
//! same small dedicated bridge gadgets `LayerNormChip` uses. Two differences
//! from `LayerNormChip`'s pattern, both because this chip's `x_i` and
//! `weight_i` are genuine *external* inputs (no internal producer to derive
//! a canonical cell from, unlike `LayerNormChip`'s `mean_input_cells`, which
//! comes from its own first `ReduceMeanChip::assign` call):
//!
//! - Each `x_i` and `weight_i` is witnessed exactly once into a dedicated
//!   "anchor" column (`x_anchor`/`weight_anchor`, both unshifted, equality
//!   enabled, no gate needed -- a pure witness point) up front, and every
//!   other row that uses `x_i` or `weight_i` links back to that one anchor
//!   cell via `constrain_equal`, rather than trusting two independently
//!   witnessed copies of the same host value to coincide (they would not be
//!   constrained equal to each other otherwise -- exactly the disconnected-
//!   witness trap this codebase's soundness docs warn about).
//! - `RmsNormChip::assign` returns those anchor cells as `input_cells`/
//!   `weight_cells`, so `crate::assembler::AssemblerChip` (which dispatches
//!   this instruction) can link them to whichever register produced `x_i`/
//!   `weight_i`, the same way it already does for `DotGeneral`/`Eltwise`.
//!
//! ## CRITICAL NUMERIC LIMITATION -- inherited from `RsqrtChip`/`LayerNormChip`
//!
//! Same as `LayerNormChip`'s own "CRITICAL NUMERIC LIMITATION" docs: I18's
//! representable range (`~9.22` in magnitude), the `rsqrt` lookup domain's
//! exact-quantized-point requirement, and epsilon's milli-unit granularity
//! all apply identically here -- see that module's docs for the full
//! reasoning, which is not repeated verbatim in this file.

use crate::chips::eltwise::{EltwiseAddConfig, EltwiseMulConfig};
use crate::chips::layer_norm::{
    assign_add_row, assign_mul_row, RsqrtChip, RsqrtConfig, RsqrtDomain,
};
use crate::chips::lookup::LookupError;
use crate::chips::reduce::{ReduceMeanChip, ReduceMeanConfig};
use crate::field_convert::{i64_to_fr, shifted_i64_witness, Fr};
use crate::fixed_point::{requantize_mul, I18};
use halo2_proofs::circuit::{AssignedCell, Layouter, Value};
use halo2_proofs::plonk::{Advice, Column, ConstraintSystem, ErrorFront, Expression, Selector};
use halo2_proofs::poly::Rotation;
use std::fmt;

/// Errors from [`RmsNormChip::assign`].
#[derive(Debug)]
pub enum RmsNormError {
    /// `inputs.len()` or `weight.len()` didn't match `k` (the count fixed at
    /// `configure` time).
    InputCountMismatch { expected: usize, got: usize },
    /// A wrapped synthesis-time error from a composed sub-chip.
    Synthesis(ErrorFront),
    /// `mean(x^2) + epsilon` was not an exact point of the configured
    /// `rsqrt` lookup domain -- see `chips::layer_norm`'s identical
    /// limitation.
    Rsqrt(LookupError),
}

impl fmt::Display for RmsNormError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RmsNormError::InputCountMismatch { expected, got } => {
                write!(f, "rms norm expects {expected} inputs (k), got {got}")
            }
            RmsNormError::Synthesis(err) => write!(f, "rms norm synthesis error: {err:?}"),
            RmsNormError::Rsqrt(err) => write!(f, "rms norm rsqrt lookup failed: {err}"),
        }
    }
}

impl std::error::Error for RmsNormError {}

impl From<ErrorFront> for RmsNormError {
    fn from(err: ErrorFront) -> Self {
        RmsNormError::Synthesis(err)
    }
}

/// Configuration for an [`RmsNormChip`]. See this module's top-level docs
/// for the composition strategy.
#[derive(Clone)]
pub struct RmsNormConfig {
    reduce_mean: ReduceMeanConfig,
    add: EltwiseAddConfig,
    mul: EltwiseMulConfig,
    rsqrt: RsqrtConfig,
    k: usize,
    epsilon: I18,
    rsqrt_domain: RsqrtDomain,
    // Pure witness-anchor columns for the chip's two genuine external
    // inputs (`x_i`, `weight_i`) -- see this module's top-level docs on why
    // `LayerNormChip`'s `mean_input_cells` trick doesn't apply here.
    x_anchor: Column<Advice>,
    weight_anchor: Column<Advice>,
    // "Unshift bridge": ties a value produced in `EltwiseMulChip`'s
    // shifted `q` column to a fresh unshifted copy, so it can be linked
    // into another `EltwiseMulChip`/`LookupChip` row (whose `a`/`b`/
    // `input` columns hold the unshifted representation) -- identical
    // gadget to `LayerNormChip`'s own `s_unshift` gate.
    unshift_in: Column<Advice>,
    unshift_out: Column<Advice>,
    s_unshift: Selector,
}

/// The result of [`RmsNormChip::assign`]: the normalized+scaled output
/// values, plus every `AssignedCell` a composing caller (namely
/// `crate::assembler::AssemblerChip`) needs to link this instruction's
/// inputs/output to neighboring instructions' registers.
pub struct RmsNormOutput {
    pub outputs: Vec<I18>,
    /// Final per-channel output cell, in `EltwiseMulChip`'s *shifted*
    /// representation (matching `DotProductChip`/`EltwiseMulChip`'s own
    /// external `assign` convention -- see `crate::assembler`'s module
    /// docs on representation bridging).
    pub output_cells: Vec<AssignedCell<Fr, Fr>>,
    /// Canonical (unshifted) witness cell for each `x_i`.
    pub input_cells: Vec<AssignedCell<Fr, Fr>>,
    /// Canonical (unshifted) witness cell for each `weight_i`.
    pub weight_cells: Vec<AssignedCell<Fr, Fr>>,
}

pub struct RmsNormChip {
    config: RmsNormConfig,
    rsqrt_chip: RsqrtChip,
}

impl RmsNormChip {
    #[allow(clippy::too_many_arguments)]
    pub fn configure(
        meta: &mut ConstraintSystem<Fr>,
        values: Column<Advice>,
        sum: Column<Advice>,
        sum_shift: Column<Advice>,
        mean_q: Column<Advice>,
        mean_r: Column<Advice>,
        mean_slack: Column<Advice>,
        add_a: Column<Advice>,
        add_b: Column<Advice>,
        add_c: Column<Advice>,
        mul_a: Column<Advice>,
        mul_b: Column<Advice>,
        mul_q: Column<Advice>,
        mul_r: Column<Advice>,
        mul_slack: Column<Advice>,
        bits: Column<Advice>,
        rsqrt_input: Column<Advice>,
        rsqrt_output: Column<Advice>,
        x_anchor: Column<Advice>,
        weight_anchor: Column<Advice>,
        unshift_in: Column<Advice>,
        unshift_out: Column<Advice>,
        k: usize,
        epsilon_milli: u64,
        rsqrt_domain: RsqrtDomain,
    ) -> RmsNormConfig {
        assert!(k >= 1, "RmsNormChip requires at least one input");

        let reduce_mean = ReduceMeanChip::configure(
            meta, values, sum, sum_shift, mean_q, mean_r, mean_slack, bits, k,
        );
        let add = crate::chips::eltwise::EltwiseAddChip::configure(meta, add_a, add_b, add_c, bits);
        let mul = crate::chips::eltwise::EltwiseMulChip::configure(
            meta, mul_a, mul_b, mul_q, mul_r, mul_slack, bits,
        );
        let rsqrt = RsqrtChip::configure(meta, rsqrt_input, rsqrt_output);

        let epsilon = I18::from_f64(epsilon_milli as f64 / 1000.0)
            .expect("epsilon_milli / 1000 must be representable as an I18 fixed-point value");

        meta.enable_equality(x_anchor);
        meta.enable_equality(weight_anchor);
        meta.enable_equality(unshift_in);
        meta.enable_equality(unshift_out);

        let s_unshift = meta.selector();
        meta.create_gate("rms norm unshift bridge", |meta| {
            let shifted_expr = meta.query_advice(unshift_in, Rotation::cur());
            let unshifted_expr = meta.query_advice(unshift_out, Rotation::cur());
            let s_unshift = meta.query_selector(s_unshift);
            let shift = Expression::Constant(crate::field_convert::i128_to_fr(
                crate::field_convert::SIGNED_SHIFT,
            ));
            vec![s_unshift * (unshifted_expr + shift - shifted_expr)]
        });

        RmsNormConfig {
            reduce_mean,
            add,
            mul,
            rsqrt,
            k,
            epsilon,
            rsqrt_domain,
            x_anchor,
            weight_anchor,
            unshift_in,
            unshift_out,
            s_unshift,
        }
    }

    pub fn construct(config: RmsNormConfig) -> Self {
        let rsqrt_chip =
            RsqrtChip::construct_with_domain(config.rsqrt.clone(), config.rsqrt_domain.clone());
        RmsNormChip { config, rsqrt_chip }
    }

    /// Loads the fixed `rsqrt` table. Must be called exactly once per
    /// circuit synthesis. Delegates to [`RsqrtChip::load_table`].
    pub fn load_table(&self, layouter: impl Layouter<Fr>) -> Result<(), ErrorFront> {
        self.rsqrt_chip.load_table(layouter)
    }

    fn unshift(
        &self,
        mut layouter: impl Layouter<Fr>,
        shifted_cell: &AssignedCell<Fr, Fr>,
        raw_value: i64,
    ) -> Result<AssignedCell<Fr, Fr>, ErrorFront> {
        let (shifted_fr, _) = shifted_i64_witness(raw_value);
        let unshifted_fr = Value::known(i64_to_fr(raw_value));
        let (shifted_copy_cell, unshifted_cell) = layouter.assign_region(
            || "rms norm unshift",
            |mut region| {
                self.config.s_unshift.enable(&mut region, 0)?;
                let shifted_copy_cell = region.assign_advice(
                    || "shifted in",
                    self.config.unshift_in,
                    0,
                    || shifted_fr,
                )?;
                let unshifted_cell = region.assign_advice(
                    || "unshifted out",
                    self.config.unshift_out,
                    0,
                    || unshifted_fr,
                )?;
                Ok((shifted_copy_cell, unshifted_cell))
            },
        )?;
        layouter.assign_region(
            || "rms norm unshift link",
            |mut region| region.constrain_equal(shifted_copy_cell.cell(), shifted_cell.cell()),
        )?;
        Ok(unshifted_cell)
    }

    /// Assigns the full RMS-norm pipeline for `inputs`/`weight` (each must
    /// have length `k`) and returns the length-`k` `output = x * rsqrt(mean(x^2)
    /// + eps) * weight` vector, plus every cell a composing caller needs --
    /// see [`RmsNormOutput`] and this module's top-level soundness docs.
    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fr>,
        inputs: &[I18],
        weight: &[I18],
    ) -> Result<RmsNormOutput, RmsNormError> {
        let k = self.config.k;
        if inputs.len() != k {
            return Err(RmsNormError::InputCountMismatch {
                expected: k,
                got: inputs.len(),
            });
        }
        if weight.len() != k {
            return Err(RmsNormError::InputCountMismatch {
                expected: k,
                got: weight.len(),
            });
        }

        // Anchor: witness each x_i / weight_i exactly once as the canonical
        // cell every other row linking to that value constrains against.
        let mut x_anchor_cells = Vec::with_capacity(k);
        let mut weight_anchor_cells = Vec::with_capacity(k);
        for (i, (x, w)) in inputs.iter().zip(weight.iter()).enumerate() {
            let (xc, wc) = layouter.assign_region(
                || format!("rms norm anchor {i}"),
                |mut region| {
                    let xc = region.assign_advice(
                        || "x anchor",
                        self.config.x_anchor,
                        0,
                        || Value::known(i64_to_fr(x.raw())),
                    )?;
                    let wc = region.assign_advice(
                        || "weight anchor",
                        self.config.weight_anchor,
                        0,
                        || Value::known(i64_to_fr(w.raw())),
                    )?;
                    Ok((xc, wc))
                },
            )?;
            x_anchor_cells.push(xc);
            weight_anchor_cells.push(wc);
        }

        let mean_chip = ReduceMeanChip::construct(self.config.reduce_mean.clone());

        // Step 1: sq_i = x_i * x_i.
        let mut squares = Vec::with_capacity(k);
        let mut square_cells = Vec::with_capacity(k);
        for (i, x) in inputs.iter().enumerate() {
            let (a_cell, b_cell, sq_cell) = assign_mul_row(
                &self.config.mul,
                layouter.namespace(|| format!("rms norm square {i}")),
                *x,
                *x,
            )?;
            layouter.assign_region(
                || format!("rms norm square {i} anchor links"),
                |mut region| {
                    region.constrain_equal(a_cell.cell(), x_anchor_cells[i].cell())?;
                    region.constrain_equal(b_cell.cell(), x_anchor_cells[i].cell())?;
                    Ok(())
                },
            )?;

            let (sq, _) = requantize_mul(*x, *x).expect("I18 rms norm square overflow");
            squares.push(sq);
            square_cells.push(sq_cell);
        }

        // Step 2: mean_sq = (1/k) * sum_i sq_i.
        let (mean_sq, mean_sq_cell, sq_input_cells) =
            mean_chip.assign(layouter.namespace(|| "rms norm mean of squares"), &squares)?;
        for (i, sq) in squares.iter().enumerate() {
            let sq_unshifted_cell = self.unshift(
                layouter.namespace(|| format!("rms norm square {i} unshift")),
                &square_cells[i],
                sq.raw(),
            )?;
            layouter.assign_region(
                || format!("rms norm mean input {i} link"),
                |mut region| {
                    region.constrain_equal(sq_unshifted_cell.cell(), sq_input_cells[i].cell())
                },
            )?;
        }

        // Step 3: variance_plus_eps = mean_sq + epsilon. Both `mean_sq_cell`
        // (from `ReduceMeanChip`) and `EltwiseAddChip`'s `a` operand hold the
        // shifted representation, so no bridge is needed here.
        let (vpe_a_cell, _vpe_b_cell, vpe_cell) = assign_add_row(
            &self.config.add,
            layouter.namespace(|| "rms norm variance plus epsilon"),
            mean_sq.raw(),
            self.config.epsilon.raw(),
        )?;
        layouter.assign_region(
            || "rms norm variance plus epsilon link",
            |mut region| region.constrain_equal(vpe_a_cell.cell(), mean_sq_cell.cell()),
        )?;
        let vpe_raw = mean_sq
            .raw()
            .checked_add(self.config.epsilon.raw())
            .expect("I18 rms norm variance+epsilon overflow");
        let vpe = I18::from_raw(vpe_raw);

        // Step 4: rsqrt_value = rsqrt(variance_plus_eps).
        let rsqrt_value = self
            .rsqrt_chip
            .assign(layouter.namespace(|| "rms norm rsqrt"), vpe)
            .map_err(RmsNormError::Rsqrt)?;

        let vpe_unshifted_cell = self.unshift(
            layouter.namespace(|| "rms norm variance plus epsilon unshift"),
            &vpe_cell,
            vpe.raw(),
        )?;

        let rsqrt_config = self.config.rsqrt.clone();
        let (rsqrt_input_cell, rsqrt_output_cell) = layouter.assign_region(
            || "rms norm rsqrt link row",
            |mut region| {
                rsqrt_config.selector().enable(&mut region, 0)?;
                let input_cell = region.assign_advice(
                    || "rsqrt input",
                    rsqrt_config.input_column(),
                    0,
                    || Value::known(i64_to_fr(vpe.raw())),
                )?;
                let output_cell = region.assign_advice(
                    || "rsqrt output",
                    rsqrt_config.output_column(),
                    0,
                    || Value::known(i64_to_fr(rsqrt_value.raw())),
                )?;
                Ok((input_cell, output_cell))
            },
        )?;
        layouter.assign_region(
            || "rms norm rsqrt input link",
            |mut region| region.constrain_equal(rsqrt_input_cell.cell(), vpe_unshifted_cell.cell()),
        )?;

        // Step 5: normed_i = x_i * rsqrt_value.
        let mut normed = Vec::with_capacity(k);
        let mut normed_cells = Vec::with_capacity(k);
        for (i, x) in inputs.iter().enumerate() {
            let (a_cell, b_cell, normed_cell) = assign_mul_row(
                &self.config.mul,
                layouter.namespace(|| format!("rms norm scale {i}")),
                *x,
                rsqrt_value,
            )?;
            layouter.assign_region(
                || format!("rms norm scale {i} links"),
                |mut region| {
                    region.constrain_equal(a_cell.cell(), x_anchor_cells[i].cell())?;
                    region.constrain_equal(b_cell.cell(), rsqrt_output_cell.cell())?;
                    Ok(())
                },
            )?;
            let (n, _) = requantize_mul(*x, rsqrt_value).expect("I18 rms norm scale overflow");
            normed.push(n);
            normed_cells.push(normed_cell);
        }

        // Step 6: output_i = normed_i * weight_i.
        let mut outputs = Vec::with_capacity(k);
        let mut output_cells = Vec::with_capacity(k);
        for (i, n) in normed.iter().enumerate() {
            let normed_unshifted_cell = self.unshift(
                layouter.namespace(|| format!("rms norm normed {i} unshift")),
                &normed_cells[i],
                n.raw(),
            )?;
            let (a_cell, b_cell, out_cell) = assign_mul_row(
                &self.config.mul,
                layouter.namespace(|| format!("rms norm weight scale {i}")),
                *n,
                weight[i],
            )?;
            layouter.assign_region(
                || format!("rms norm weight scale {i} links"),
                |mut region| {
                    region.constrain_equal(a_cell.cell(), normed_unshifted_cell.cell())?;
                    region.constrain_equal(b_cell.cell(), weight_anchor_cells[i].cell())?;
                    Ok(())
                },
            )?;
            let (out, _) = requantize_mul(*n, weight[i]).expect("I18 rms norm output overflow");
            outputs.push(out);
            output_cells.push(out_cell);
        }

        Ok(RmsNormOutput {
            outputs,
            output_cells,
            input_cells: x_anchor_cells,
            weight_cells: weight_anchor_cells,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::circuit::{SimpleFloorPlanner, Value};
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::plonk::{Circuit, ConstraintSystem, ErrorFront};
    use std::cell::RefCell;

    const K: usize = 4;
    // epsilon = 10 / 1000 = 0.01.
    const EPSILON_MILLI: u64 = 10;
    // mean(x^2) for sample_inputs() below is exactly 1.25 (see comment on
    // sample_inputs), so mean(x^2)+eps = 1.26 -- domain chosen (as
    // `LayerNormChip`'s own tests do) so this is an exact grid point.
    const RSQRT_DOMAIN_MIN: f64 = 0.1;
    const RSQRT_DOMAIN_MAX: f64 = 3.0;
    const RSQRT_DOMAIN_N: usize = 101;

    const CIRCUIT_K: u32 = 12;

    fn sample_inputs() -> Vec<I18> {
        // x^2: 2.25, 0.25, 0.25, 2.25 -> mean = 1.25 exactly.
        vec![
            I18::from_f64(-1.5).unwrap(),
            I18::from_f64(-0.5).unwrap(),
            I18::from_f64(0.5).unwrap(),
            I18::from_f64(1.5).unwrap(),
        ]
    }

    fn sample_weight() -> Vec<I18> {
        vec![
            I18::from_f64(1.0).unwrap(),
            I18::from_f64(2.0).unwrap(),
            I18::from_f64(0.5).unwrap(),
            I18::from_f64(-1.0).unwrap(),
        ]
    }

    /// Independent host-side (f64) computation of the real RMSNorm formula
    /// TimesFM's own `RMSNorm._norm`/`forward` (add_unit_offset=False) use:
    /// `x * rsqrt(mean(x^2, -1) + eps) * weight`. Used to cross-check the
    /// chip's returned `I18` outputs are not merely internally consistent,
    /// but numerically the right computation.
    fn expected_outputs() -> Vec<f64> {
        let xs = [-1.5_f64, -0.5, 0.5, 1.5];
        let ws = [1.0_f64, 2.0, 0.5, -1.0];
        let mean_sq = xs.iter().map(|x| x * x).sum::<f64>() / xs.len() as f64;
        let rsqrt_v = 1.0 / (mean_sq + 0.01_f64).sqrt();
        xs.iter()
            .zip(ws.iter())
            .map(|(x, w)| x * rsqrt_v * w)
            .collect()
    }

    #[derive(Clone)]
    struct RmsNormTestConfig {
        rms: RmsNormConfig,
    }

    #[allow(clippy::too_many_arguments)]
    fn configure_test_chip(meta: &mut ConstraintSystem<Fr>) -> RmsNormTestConfig {
        let values = meta.advice_column();
        let sum = meta.advice_column();
        let sum_shift = meta.advice_column();
        let mean_q = meta.advice_column();
        let mean_r = meta.advice_column();
        let mean_slack = meta.advice_column();
        let add_a = meta.advice_column();
        let add_b = meta.advice_column();
        let add_c = meta.advice_column();
        let mul_a = meta.advice_column();
        let mul_b = meta.advice_column();
        let mul_q = meta.advice_column();
        let mul_r = meta.advice_column();
        let mul_slack = meta.advice_column();
        let bits = meta.advice_column();
        let rsqrt_input = meta.advice_column();
        let rsqrt_output = meta.advice_column();
        let x_anchor = meta.advice_column();
        let weight_anchor = meta.advice_column();
        let unshift_in = meta.advice_column();
        let unshift_out = meta.advice_column();

        let rms = RmsNormChip::configure(
            meta,
            values,
            sum,
            sum_shift,
            mean_q,
            mean_r,
            mean_slack,
            add_a,
            add_b,
            add_c,
            mul_a,
            mul_b,
            mul_q,
            mul_r,
            mul_slack,
            bits,
            rsqrt_input,
            rsqrt_output,
            x_anchor,
            weight_anchor,
            unshift_in,
            unshift_out,
            K,
            EPSILON_MILLI,
            RsqrtDomain::Range {
                min: RSQRT_DOMAIN_MIN,
                max: RSQRT_DOMAIN_MAX,
                n: RSQRT_DOMAIN_N,
            },
        );
        RmsNormTestConfig { rms }
    }

    struct RmsNormTestCircuit {
        inputs: Vec<I18>,
        weight: Vec<I18>,
        // Captures `assign`'s returned I18 outputs so the test can assert on
        // them after `MockProver::run` -- `synthesize` has no return value,
        // so this is the standard interior-mutability escape hatch other
        // chips' tests in this codebase also don't need (they only check
        // constraint satisfaction), but this test additionally wants a
        // numeric cross-check against `expected_outputs()`.
        captured_outputs: RefCell<Option<Vec<I18>>>,
    }

    impl Circuit<Fr> for RmsNormTestCircuit {
        type Config = RmsNormTestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            RmsNormTestCircuit {
                inputs: vec![I18::from_raw(0); self.inputs.len()],
                weight: vec![I18::from_raw(0); self.weight.len()],
                captured_outputs: RefCell::new(None),
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            configure_test_chip(meta)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = RmsNormChip::construct(config.rms);
            chip.load_table(layouter.namespace(|| "load rsqrt table"))?;
            let result = chip
                .assign(
                    layouter.namespace(|| "rms norm"),
                    &self.inputs,
                    &self.weight,
                )
                .map_err(|e| panic!("rms norm assign failed: {e}"))?;
            *self.captured_outputs.borrow_mut() = Some(result.outputs);
            Ok(())
        }
    }

    #[test]
    fn rms_norm_matches_expected_output() {
        let circuit = RmsNormTestCircuit {
            inputs: sample_inputs(),
            weight: sample_weight(),
            captured_outputs: RefCell::new(None),
        };
        let prover = MockProver::run(CIRCUIT_K, &circuit, vec![]).unwrap();
        prover.assert_satisfied();

        let got = circuit
            .captured_outputs
            .borrow()
            .clone()
            .expect("assign should have run during synthesize");
        let expected = expected_outputs();
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(expected.iter()) {
            let diff = (g.to_f64() - e).abs();
            assert!(
                diff < 1e-3,
                "rms norm output mismatch: got {}, expected {e} (diff {diff})",
                g.to_f64()
            );
        }
    }

    #[test]
    fn assign_rejects_wrong_input_count_at_the_rust_level() {
        struct GuardCircuit;
        impl Circuit<Fr> for GuardCircuit {
            type Config = RmsNormTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                GuardCircuit
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                configure_test_chip(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let chip = RmsNormChip::construct(config.rms);
                chip.load_table(layouter.namespace(|| "load rsqrt table"))?;
                let result = chip.assign(
                    layouter.namespace(|| "rms norm"),
                    &sample_inputs()[..3],
                    &sample_weight(),
                );
                assert!(matches!(
                    result,
                    Err(RmsNormError::InputCountMismatch {
                        expected: 4,
                        got: 3
                    })
                ));
                Ok(())
            }
        }

        MockProver::run(CIRCUIT_K, &GuardCircuit, vec![])
            .unwrap()
            .assert_satisfied();
    }

    /// Soundness regression test: witnesses a "weight anchor" cell holding
    /// one value, then a weight-multiply row whose `b` operand holds a
    /// genuinely DIFFERENT value, and claims (via `constrain_equal`) that
    /// they're the same cell -- exactly the disconnected-witness bug shape
    /// this codebase's soundness docs warn about (see this module's
    /// top-level docs and `docs/superpowers/plans/2026-07-26-zkie-subproject1-foundation.md`'s
    /// addendum), scoped to `RmsNormChip`'s new weight-anchor link. Must be
    /// REJECTED by the real permutation argument.
    #[test]
    fn forged_weight_link_is_rejected() {
        struct MismatchedWeightCircuit;
        impl Circuit<Fr> for MismatchedWeightCircuit {
            type Config = RmsNormTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                MismatchedWeightCircuit
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                configure_test_chip(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                // Witness a weight anchor for value 1.0...
                let anchor_cell = layouter.assign_region(
                    || "weight anchor",
                    |mut region| {
                        region.assign_advice(
                            || "weight anchor",
                            config.rms.weight_anchor,
                            0,
                            || Value::known(i64_to_fr(I18::from_f64(1.0).unwrap().raw())),
                        )
                    },
                )?;
                // ...then a weight-scale mul row using a DIFFERENT value
                // (2.0) for `b`...
                let (_a, b_cell, _q) = assign_mul_row(
                    &config.rms.mul,
                    layouter.namespace(|| "mismatched weight row"),
                    I18::from_f64(1.0).unwrap(),
                    I18::from_f64(2.0).unwrap(),
                )?;
                // ...and claim they're equal. This must be REJECTED.
                layouter.assign_region(
                    || "forged link",
                    |mut region| region.constrain_equal(b_cell.cell(), anchor_cell.cell()),
                )?;
                Ok(())
            }
        }

        let forged_prover = MockProver::run(CIRCUIT_K, &MismatchedWeightCircuit, vec![]).unwrap();
        assert!(
            forged_prover.verify().is_err(),
            "forging a constrain_equal between two genuinely different witnessed values must be rejected"
        );

        // Sanity: the real, untampered circuit still passes.
        let circuit = RmsNormTestCircuit {
            inputs: sample_inputs(),
            weight: sample_weight(),
            captured_outputs: RefCell::new(None),
        };
        MockProver::run(CIRCUIT_K, &circuit, vec![])
            .unwrap()
            .assert_satisfied();
    }
}
