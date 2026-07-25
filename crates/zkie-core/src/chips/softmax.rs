//! `SoftmaxChip`: backs the `Softmax` ISA instruction -- computes
//! `softmax(x_i) = exp(x_i) / sum_j exp(x_j)` over a fixed, compile-time
//! known count `K` of I18 inputs, by composing three already-fixed building
//! blocks:
//!
//! 1. an exp lookup (`LookupChip` + `build_domain(f64::exp, ...)`) computes
//!    `e_i = exp(x_i)` for each input, exactly the way `GeluChip` wraps
//!    `LookupChip` for GELU;
//! 2. `ReduceSumChip` sums the `e_i` into `s = sum(e_i)`;
//! 3. `DivChip`, invoked once per `i` (reusing one configured instance across
//!    `K` sequential regions, the same way `PatchEmbedChip` reuses one
//!    `DotProductChip` across `embed_dim` dimensions), computes `e_i / s`.
//!
//! # Soundness: linking cells across the three composed chips
//!
//! Each of `LookupChip::assign`, `ReduceSumChip::assign` and
//! `DivChip::assign` witnesses its own value in its own layouter region, so
//! passing an `I18` returned by one into another as a plain host-side value
//! (without an explicit `region.constrain_equal`) would leave the two chips'
//! cells disconnected -- exactly the class of bug fixed elsewhere in this
//! crate for `RangeCheckChip` composition (see `chips/reduce.rs` and
//! `chips/div.rs`'s module docs). `SoftmaxChip::assign` therefore explicitly
//! links:
//!
//! - each exp lookup's output cell to `ReduceSumChip`'s own re-witnessed
//!   copy of that same input (`ReduceSumChip::assign`'s returned
//!   `value_cells`);
//! - each exp lookup's output cell to the corresponding `DivChip::assign`
//!   call's `numerator` cell;
//! - `ReduceSumChip::assign`'s returned final-sum cell to every
//!   `DivChip::assign` call's `divisor` cell.
//!
//! # CRITICAL NUMERIC LIMITATION: I18's representable range
//!
//! I18 is a plain `i64`-backed fixed-point type (scale `10^18`) with **no**
//! wider host-level accumulator: its representable range is only
//! `i64::MAX / 1e18 ~= +/-9.22`. `exp(x)` grows explosively -- `exp(3) ~=
//! 20.1` already overflows this range -- and softmax's sum of `K`
//! exponentials must *also* fit in that same `+/-9.22` range (there is no
//! `i128` intermediate accumulator here, unlike `DotProductChip`'s raw
//! product-sum, which is rescaled back down to I18 before any other chip
//! ever sees it).
//!
//! This chip's tests therefore restrict the exp lookup domain to
//! `[EXP_DOMAIN_MIN, EXP_DOMAIN_MAX] = [-4.0, 0.0]`: `exp(x)` over this
//! range is bounded by `[~0.0183, 1.0]`, so even summing `K = 8` such values
//! tops out at `~8.0`, safely under `9.22`. Softmax's shift-invariance
//! (`softmax(x) == softmax(x - max(x))`) means any real-valued input vector
//! can in principle be renormalized into this domain by subtracting its max
//! off host-side before calling this chip -- but that shift is **not**
//! performed by this chip itself (an in-circuit max-reduction gadget is out
//! of scope here); callers are responsible for pre-shifting inputs into
//! whatever domain this chip is configured with. This is a known, documented
//! limitation of this foundation layer, to be revisited once real model
//! activation ranges are analyzed -- not something to silently work around
//! by only ever testing with unrealistically small values without saying so.

use crate::chips::div::{DivChip, DivConfig, DivError};
use crate::chips::lookup::{build_domain, LookupChip, LookupConfig, LookupError};
use crate::chips::reduce::{ReduceSumChip, ReduceSumConfig};
use crate::field_convert::Fr;
use crate::fixed_point::I18;
use halo2_proofs::circuit::Layouter;
use halo2_proofs::plonk::{Advice, Column, ConstraintSystem, ErrorFront};
use std::fmt;

/// Errors that can occur while assigning a [`SoftmaxChip`] region.
#[derive(Debug)]
pub enum SoftmaxError {
    /// `inputs.len()` did not match the compile-time-configured `K`.
    InputCountMismatch { expected: usize, got: usize },
    /// One of the inputs' `exp` lookup failed (not an exact domain point).
    Lookup(LookupError),
    /// One of the per-output division sub-circuits failed (e.g. a
    /// non-positive divisor, or a requantized quotient overflow -- neither
    /// should occur in practice since the divisor is a sum of positive
    /// lookup-table outputs, but the typed error is still surfaced rather
    /// than unwrapped).
    Div(DivError),
    /// A halo2 circuit-synthesis error occurred while assigning cells or
    /// linking them via `region.constrain_equal`.
    Circuit(String),
}

impl fmt::Display for SoftmaxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SoftmaxError::InputCountMismatch { expected, got } => {
                write!(f, "softmax expects {expected} inputs (K), got {got}")
            }
            SoftmaxError::Lookup(e) => write!(f, "softmax exp lookup failed: {e}"),
            SoftmaxError::Div(e) => write!(f, "softmax division failed: {e}"),
            SoftmaxError::Circuit(e) => write!(f, "softmax circuit error: {e}"),
        }
    }
}

impl std::error::Error for SoftmaxError {}

impl From<LookupError> for SoftmaxError {
    fn from(e: LookupError) -> Self {
        SoftmaxError::Lookup(e)
    }
}

impl From<DivError> for SoftmaxError {
    fn from(e: DivError) -> Self {
        SoftmaxError::Div(e)
    }
}

impl From<ErrorFront> for SoftmaxError {
    fn from(e: ErrorFront) -> Self {
        SoftmaxError::Circuit(format!("{e:?}"))
    }
}

/// Configuration for a [`SoftmaxChip`]: an exp lookup, a `K`-input reduce
/// sum, and a division gadget, each configured once and reused across the
/// `K` per-input regions [`SoftmaxChip::assign`] creates.
#[derive(Clone, Debug)]
pub struct SoftmaxConfig {
    lookup: LookupConfig,
    sum: ReduceSumConfig,
    div: DivConfig,
    k: usize,
}

/// A chip that proves "I correctly computed softmax over `K` fixed-point
/// inputs" by composing an exp lookup, a running sum, and `K` divisions.
/// See the module docs for the domain/range limitation this chip's callers
/// must respect, and for how cells are linked across the composed chips.
pub struct SoftmaxChip {
    config: SoftmaxConfig,
    exp_lookup: LookupChip,
}

impl SoftmaxChip {
    /// Configures the three composed sub-chips. `k` is the compile-time-known
    /// input count (softmax's `axis_dim`). See each sub-chip's own
    /// `configure` for what each column is used for; `bits` is shared across
    /// `ReduceSumChip`'s and `DivChip`'s internal range checks, mirroring
    /// `ReduceMeanChip::configure`'s single shared `bits` column.
    #[allow(clippy::too_many_arguments)]
    pub fn configure(
        meta: &mut ConstraintSystem<Fr>,
        exp_input: Column<Advice>,
        exp_output: Column<Advice>,
        sum_values: Column<Advice>,
        sum: Column<Advice>,
        sum_shift: Column<Advice>,
        div_numerator: Column<Advice>,
        div_divisor: Column<Advice>,
        div_q_shift: Column<Advice>,
        div_r: Column<Advice>,
        div_slack: Column<Advice>,
        div_dm1: Column<Advice>,
        bits: Column<Advice>,
        k: usize,
    ) -> SoftmaxConfig {
        assert!(k >= 1, "SoftmaxChip requires at least one input");
        let lookup = LookupChip::configure(meta, exp_input, exp_output);
        let sum_config = ReduceSumChip::configure(meta, sum_values, sum, sum_shift, bits, k);
        let div = DivChip::configure(
            meta,
            div_numerator,
            div_divisor,
            div_q_shift,
            div_r,
            div_slack,
            div_dm1,
            bits,
        );
        SoftmaxConfig {
            lookup,
            sum: sum_config,
            div,
            k,
        }
    }

    /// Builds a chip backed by an exp table: `n` points evenly quantized
    /// over `[domain_min, domain_max]`, each paired with `f64::exp(x)`
    /// computed host-side (see the module docs' numeric-range warning for
    /// how to choose `domain_min`/`domain_max` safely).
    pub fn construct(config: SoftmaxConfig, domain_min: f64, domain_max: f64, n: usize) -> Self {
        let (domain, values) = build_domain(f64::exp, domain_min, domain_max, n);
        let exp_lookup = LookupChip::construct(config.lookup.clone(), domain, values);
        SoftmaxChip { config, exp_lookup }
    }

    /// Loads the fixed exp table backing the lookup argument. Must be called
    /// exactly once per circuit synthesis. Delegates to
    /// [`LookupChip::load_table`].
    pub fn load_table(&self, layouter: impl Layouter<Fr>) -> Result<(), ErrorFront> {
        self.exp_lookup.load_table(layouter)
    }

    /// Computes `softmax(inputs)`, returning the length-`K` I18 output
    /// vector. `inputs` must have exactly `K` elements (the count fixed at
    /// `configure` time) and each element must be an exact point of the exp
    /// lookup's domain (see [`LookupChip::assign`]); returns a typed
    /// [`SoftmaxError`] -- never panics -- otherwise.
    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fr>,
        inputs: &[I18],
    ) -> Result<Vec<I18>, SoftmaxError> {
        let k = self.config.k;
        if inputs.len() != k {
            return Err(SoftmaxError::InputCountMismatch {
                expected: k,
                got: inputs.len(),
            });
        }

        // Step 1: e_i = exp(x_i) via the shared exp lookup table, keeping
        // each output's AssignedCell so it can be linked (rather than
        // re-witnessed disconnected) into both the sum below and each
        // input's own division.
        let mut exp_values = Vec::with_capacity(k);
        let mut exp_cells = Vec::with_capacity(k);
        for (i, x) in inputs.iter().enumerate() {
            let (e, cell) = self
                .exp_lookup
                .assign(layouter.namespace(|| format!("softmax exp {i}")), *x)?;
            exp_values.push(e);
            exp_cells.push(cell);
        }

        // Step 2: s = sum(e_i) via ReduceSumChip. `value_cells[i]` are
        // ReduceSumChip's own (freshly witnessed, otherwise disconnected)
        // copies of e_i -- link each back to the real lookup output cell.
        let sum_chip = ReduceSumChip::construct(self.config.sum.clone());
        let (sum, sum_cell, value_cells) =
            sum_chip.assign(layouter.namespace(|| "softmax sum"), &exp_values)?;

        layouter.assign_region(
            || "softmax exp-to-sum links",
            |mut region| {
                for (exp_cell, value_cell) in exp_cells.iter().zip(value_cells.iter()) {
                    region.constrain_equal(exp_cell.cell(), value_cell.cell())?;
                }
                Ok(())
            },
        )?;

        // Step 3: softmax_i = e_i / s, one DivChip region per input, reusing
        // the same configured columns (mirroring PatchEmbedChip's reuse of
        // one DotProductChip across dimensions). Each division's numerator
        // and divisor cells are linked back to the real e_i / s cells above
        // rather than left as disconnected re-witnessed copies.
        let mut outputs = Vec::with_capacity(k);
        for i in 0..k {
            let div_chip = DivChip::construct(self.config.div.clone());
            let (quotient, numerator_cell, divisor_cell) = div_chip.assign(
                layouter.namespace(|| format!("softmax div {i}")),
                exp_values[i],
                sum,
            )?;

            layouter.assign_region(
                || format!("softmax div {i} links"),
                |mut region| {
                    region.constrain_equal(numerator_cell.cell(), exp_cells[i].cell())?;
                    region.constrain_equal(divisor_cell.cell(), sum_cell.cell())?;
                    Ok(())
                },
            )?;

            outputs.push(quotient);
        }

        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chips::div::div_quotient_remainder;
    use crate::field_convert::i64_to_fr;
    use halo2_proofs::circuit::{SimpleFloorPlanner, Value};
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::plonk::{Circuit, ConstraintSystem, ErrorFront};

    // See the module-level numeric-limitation doc: exp over this domain is
    // bounded by [~0.0183, 1.0], so summing even K=8 such values stays well
    // under I18's ~+/-9.22 representable range. We use K=4 in these tests,
    // for a worst-case sum of ~4.0.
    const EXP_DOMAIN_MIN: f64 = -4.0;
    const EXP_DOMAIN_MAX: f64 = 0.0;
    const EXP_DOMAIN_N: usize = 33;
    const K: usize = 4;

    fn exp_domain() -> (Vec<I18>, Vec<I18>) {
        build_domain(f64::exp, EXP_DOMAIN_MIN, EXP_DOMAIN_MAX, EXP_DOMAIN_N)
    }

    /// Independently recomputes exactly what `SoftmaxChip::assign` will
    /// produce, using the same deterministic host-side building blocks
    /// (`build_domain`'s table values and `div_quotient_remainder`) -- the
    /// same idiom `GeluChip`'s and `PatchEmbedChip`'s own tests use to check
    /// numeric correctness independently of running the circuit itself.
    fn expected_softmax(inputs: &[I18]) -> Vec<I18> {
        let (domain, values) = exp_domain();
        let exps: Vec<I18> = inputs
            .iter()
            .map(|x| {
                let idx = domain
                    .iter()
                    .position(|d| d.raw() == x.raw())
                    .expect("input must be an exact domain point");
                values[idx]
            })
            .collect();
        // I18 + I18 needs no rescale, so plain raw addition matches
        // ReduceSumChip's own semantics exactly.
        let sum_raw: i64 = exps.iter().map(|e| e.raw()).sum();
        let sum = I18::from_raw(sum_raw);
        exps.iter()
            .map(|&e| div_quotient_remainder(e, sum).unwrap().0)
            .collect()
    }

    #[derive(Clone)]
    struct SoftmaxTestConfig {
        softmax: SoftmaxConfig,
    }

    struct SoftmaxTestCircuit {
        inputs: Vec<I18>,
    }

    impl Circuit<Fr> for SoftmaxTestCircuit {
        type Config = SoftmaxTestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            // 0.0 is the domain's upper bound (an exact grid point), so this
            // remains a valid input during keygen's `without_witnesses` pass.
            SoftmaxTestCircuit {
                inputs: vec![I18::from_raw(0); self.inputs.len()],
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let exp_input = meta.advice_column();
            let exp_output = meta.advice_column();
            let sum_values = meta.advice_column();
            let sum = meta.advice_column();
            let sum_shift = meta.advice_column();
            let div_numerator = meta.advice_column();
            let div_divisor = meta.advice_column();
            let div_q_shift = meta.advice_column();
            let div_r = meta.advice_column();
            let div_slack = meta.advice_column();
            let div_dm1 = meta.advice_column();
            let bits = meta.advice_column();
            SoftmaxTestConfig {
                softmax: SoftmaxChip::configure(
                    meta,
                    exp_input,
                    exp_output,
                    sum_values,
                    sum,
                    sum_shift,
                    div_numerator,
                    div_divisor,
                    div_q_shift,
                    div_r,
                    div_slack,
                    div_dm1,
                    bits,
                    K,
                ),
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = SoftmaxChip::construct(
                config.softmax,
                EXP_DOMAIN_MIN,
                EXP_DOMAIN_MAX,
                EXP_DOMAIN_N,
            );
            chip.load_table(layouter.namespace(|| "table"))?;
            chip.assign(layouter.namespace(|| "assign"), &self.inputs)
                .expect("inputs should be exact domain points in this test");
            Ok(())
        }
    }

    #[test]
    fn softmax_of_four_inputs_is_satisfied_and_close_to_true_softmax() {
        let (domain, _values) = exp_domain();
        // Four distinct exact domain points spread across [-4.0, 0.0].
        let indices = [0usize, 12, 20, 32];
        let inputs: Vec<I18> = indices.iter().map(|&i| domain[i]).collect();

        let circuit = SoftmaxTestCircuit {
            inputs: inputs.clone(),
        };
        let prover = MockProver::run(14, &circuit, vec![]).unwrap();
        prover.assert_satisfied();

        let outputs = expected_softmax(&inputs);

        // True (real-valued) softmax, computed independently in f64.
        let xs: Vec<f64> = inputs.iter().map(I18::to_f64).collect();
        let true_exps: Vec<f64> = xs.iter().map(|&x| x.exp()).collect();
        let true_sum: f64 = true_exps.iter().sum();
        let true_softmax: Vec<f64> = true_exps.iter().map(|e| e / true_sum).collect();

        // Tolerance: the chip's own lookup table already stores exact
        // (grid-point) exp(x) values for these on-domain inputs, so the
        // dominant error sources are (a) I18's rounding of exp(x) into raw
        // units (bounded by 0.5 raw unit == 5e-19, i.e. negligible) and (b)
        // each division's Euclidean-rounding error (bounded by
        // 1/divisor_raw, also far below 1e-9 for divisors on the order of
        // 1.0-4.0 in I18 scale). We still budget generously against the
        // domain's quantization step (for off-grid inputs this chip would
        // reject anyway, but this documents the general bound) by using the
        // larger of the two.
        let step = (domain[1].to_f64() - domain[0].to_f64()).abs();
        let tolerance = step.max(1e-6);
        for (i, (out, expected)) in outputs.iter().zip(true_softmax.iter()).enumerate() {
            assert!(
                (out.to_f64() - expected).abs() < tolerance,
                "softmax[{i}] = {} vs true {} (tolerance {tolerance})",
                out.to_f64(),
                expected
            );
        }

        // Basic sanity property: softmax outputs sum to ~1.0.
        let output_sum: f64 = outputs.iter().map(I18::to_f64).sum();
        assert!(
            (output_sum - 1.0).abs() < tolerance,
            "softmax outputs should sum to ~1.0, got {output_sum}"
        );
    }

    #[test]
    fn softmax_rejects_wrong_input_count() {
        struct CountTestCircuit {
            inputs: Vec<I18>,
        }

        impl Circuit<Fr> for CountTestCircuit {
            type Config = SoftmaxTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                CountTestCircuit {
                    inputs: vec![I18::from_raw(0); self.inputs.len()],
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                SoftmaxTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let chip = SoftmaxChip::construct(
                    config.softmax,
                    EXP_DOMAIN_MIN,
                    EXP_DOMAIN_MAX,
                    EXP_DOMAIN_N,
                );
                chip.load_table(layouter.namespace(|| "table"))?;
                match chip.assign(layouter.namespace(|| "assign"), &self.inputs) {
                    Err(SoftmaxError::InputCountMismatch { expected, got }) => {
                        assert_eq!(expected, K);
                        assert_eq!(got, K - 1);
                    }
                    Ok(_) => panic!("expected InputCountMismatch but assign succeeded"),
                    Err(other) => panic!("expected InputCountMismatch, got {other}"),
                }
                Ok(())
            }
        }

        let (domain, _values) = exp_domain();
        let circuit = CountTestCircuit {
            inputs: domain[0..K - 1].to_vec(),
        };
        let _ = MockProver::run(14, &circuit, vec![]);
    }

    #[test]
    fn forged_exp_lookup_output_within_softmax_config_is_rejected() {
        // Bypasses the exp lookup step of SoftmaxChip::assign and directly
        // witnesses a valid domain input paired with the wrong output, on
        // the exact lookup columns/selector SoftmaxChip::configure wires up
        // for its exp step -- mirroring lookup.rs's and gelu.rs's own
        // forged-output tests, applied to SoftmaxConfig's `lookup` field.
        // This exercises the underlying halo2 lookup argument itself (not
        // just Rust-level bookkeeping), confirming that any attempt to
        // forge softmax's exp step is rejected.
        struct ForgedExpCircuit {
            input: I18,
            forged_output: I18,
        }

        impl Circuit<Fr> for ForgedExpCircuit {
            type Config = SoftmaxTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                ForgedExpCircuit {
                    input: self.input,
                    forged_output: self.forged_output,
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                SoftmaxTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let softmax_config = config.softmax.clone();
                let chip = SoftmaxChip::construct(
                    config.softmax,
                    EXP_DOMAIN_MIN,
                    EXP_DOMAIN_MAX,
                    EXP_DOMAIN_N,
                );
                chip.load_table(layouter.namespace(|| "table"))?;
                layouter.assign_region(
                    || "forged softmax exp lookup",
                    |mut region| {
                        // Must enable the selector: the lookup is gated (see
                        // `LookupConfig`'s doc comment in `chips/lookup.rs`),
                        // so this forged row would otherwise silently
                        // collapse to the always-satisfied padding row
                        // instead of actually checking the forged values.
                        softmax_config.lookup.selector.enable(&mut region, 0)?;
                        region.assign_advice(
                            || "input",
                            softmax_config.lookup.input,
                            0,
                            || Value::known(i64_to_fr(self.input.raw())),
                        )?;
                        region.assign_advice(
                            || "output",
                            softmax_config.lookup.output,
                            0,
                            || Value::known(i64_to_fr(self.forged_output.raw())),
                        )
                    },
                )?;
                Ok(())
            }
        }

        let (domain, values) = exp_domain();
        let input = domain[20];
        let correct_output = values[20];
        // Off by one raw unit: not equal to exp(input), and (input, forged)
        // is not a row of the table since domain inputs are unique.
        let forged_output = I18::from_raw(correct_output.raw() + 1);

        let circuit = ForgedExpCircuit {
            input,
            forged_output,
        };
        let prover = MockProver::run(14, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }
}
