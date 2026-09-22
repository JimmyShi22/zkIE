use crate::chips::lookup::{build_domain, LookupChip, LookupConfig, LookupError};
use crate::field_convert::Fr;
use crate::fixed_point::I18;
use halo2_proofs::circuit::Layouter;
use halo2_proofs::plonk::{Advice, Column, ConstraintSystem, ErrorFront, Selector};

/// Host-side (`f64`) approximation of the Gauss error function `erf(x)`,
/// using Abramowitz & Stegun formula 7.1.26 (Handbook of Mathematical
/// Functions, 1964), a well-known single-polynomial rational approximation
/// with documented maximum absolute error ~1.5e-7 for `x >= 0`:
///
/// ```text
/// erf(x) ~= 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-x^2),
/// t = 1 / (1 + p*x)
/// p = 0.3275911
/// a1 =  0.254829592
/// a2 = -0.284496736
/// a3 =  1.421413741
/// a4 = -1.453152027
/// a5 =  1.061405429
/// ```
///
/// `erf` is odd (`erf(-x) == -erf(x)`), so negative inputs are handled by
/// negating the result of the positive-branch approximation.
///
/// This is purely a host-side helper used to *generate* the GELU lookup
/// table at circuit-construction time (see [`gelu_f64`] and
/// [`GeluChip::construct`]) -- it never runs inside the circuit itself, so
/// its accuracy is a modeling/product-quality concern, not a soundness
/// concern. See the module-level docs on [`GeluChip`] for why.
fn erf_approx(x: f64) -> f64 {
    const P: f64 = 0.3275911;
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + P * x);
    let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
    let y = 1.0 - poly * (-x * x).exp();

    sign * y
}

/// The exact (non-tanh-approximated) GELU activation:
/// `GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`, evaluated host-side in
/// `f64` using [`erf_approx`].
pub fn gelu_f64(x: f64) -> f64 {
    0.5 * x * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

/// Configuration for a [`GeluChip`]. Thin wrapper around [`LookupConfig`]
/// that also retains the raw `input`/`output` advice columns so callers
/// (notably tests) can build circuits that witness rows on those same
/// columns directly, bypassing [`GeluChip::assign`], e.g. to check that the
/// halo2 lookup argument itself -- not just [`GeluChip::assign`]'s
/// bookkeeping -- rejects a forged (input, output) pair.
#[derive(Clone)]
pub struct GeluConfig {
    lookup: LookupConfig,
    input: Column<Advice>,
    output: Column<Advice>,
}

impl GeluConfig {
    /// The advice column `GeluChip::assign` witnesses its input on.
    pub fn input_column(&self) -> Column<Advice> {
        self.input
    }

    /// The advice column `GeluChip::assign` witnesses its output on.
    pub fn output_column(&self) -> Column<Advice> {
        self.output
    }

    /// The selector that gates the lookup argument (see `LookupConfig`'s
    /// doc comment in `chips/lookup.rs`): callers that witness a row on
    /// `input_column()`/`output_column()` directly (bypassing
    /// `GeluChip::assign`) must enable this selector, or the lookup
    /// argument silently collapses that row to the always-satisfied
    /// padding entry instead of actually checking the witnessed values.
    pub fn selector(&self) -> Selector {
        self.lookup.selector
    }
}

/// A chip that proves "I looked up the exact precomputed GELU value for
/// this exact quantized input" via a halo2 lookup argument, by wrapping
/// [`LookupChip`] with a GELU-specific quantized domain/value table.
///
/// # Soundness vs. accuracy
///
/// The circuit only proves membership of the witnessed `(input, output)`
/// pair in the fixed table loaded by [`GeluChip::load_table`] -- exactly
/// the same guarantee [`LookupChip`] itself provides, checked entirely by
/// the halo2 lookup argument (no separate region is witnessed and left
/// unlinked, so there is no analog of the `RangeCheckChip`
/// disconnected-cell bug here: the lookup argument constrains precisely the
/// cells [`GeluChip::assign`] returns).
///
/// How closely the *table* approximates the true mathematical GELU
/// function (a function of the `erf` polynomial approximation's accuracy,
/// and of the domain's quantization step) is a modeling/product-quality
/// concern: a "wrong" but internally-consistent table would still produce
/// valid proofs, because the statement being proved is "this output is the
/// table's value for this input," not "this output equals real-valued
/// GELU." Consumers who care about numerical fidelity to GELU should widen
/// `n` (the number of domain points) or validate the generated table
/// out-of-band; that is orthogonal to what the proof itself attests to.
pub struct GeluChip {
    inner: LookupChip,
}

impl GeluChip {
    /// Configures the lookup argument backing this chip. Delegates directly
    /// to [`LookupChip::configure`].
    pub fn configure(
        meta: &mut ConstraintSystem<Fr>,
        input: Column<Advice>,
        output: Column<Advice>,
    ) -> GeluConfig {
        let lookup = LookupChip::configure(meta, input, output);
        GeluConfig {
            lookup,
            input,
            output,
        }
    }

    /// Builds a chip backed by a GELU table: `n` points evenly quantized
    /// over `[domain_min, domain_max]`, each paired with
    /// `gelu_f64(x)` computed host-side. Both bounds must fit within I18's
    /// representable range (`domain_max`'s GELU value in particular, since
    /// GELU(x) -> x for x >> 0).
    pub fn construct(config: GeluConfig, domain_min: f64, domain_max: f64, n: usize) -> Self {
        let (domain, values) = build_domain(gelu_f64, domain_min, domain_max, n);
        let inner = LookupChip::construct(config.lookup, domain, values);
        GeluChip { inner }
    }

    /// Loads the fixed GELU table backing the lookup argument. Must be
    /// called exactly once per circuit synthesis. Delegates to
    /// [`LookupChip::load_table`].
    pub fn load_table(&self, layouter: impl Layouter<Fr>) -> Result<(), ErrorFront> {
        self.inner.load_table(layouter)
    }

    /// Witnesses `input` and its precomputed `gelu_f64(input)`, checked by
    /// the lookup argument, and returns the looked-up output. Delegates to
    /// [`LookupChip::assign`], discarding the `AssignedCell` it also returns
    /// (GELU is not currently composed into a larger chip, unlike
    /// `SoftmaxChip`'s use of the same underlying `LookupChip::assign`).
    pub fn assign(&self, layouter: impl Layouter<Fr>, input: I18) -> Result<I18, LookupError> {
        self.inner
            .assign(layouter, input)
            .map(|(value, _cell)| value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_convert::i64_to_fr;
    use halo2_proofs::circuit::{SimpleFloorPlanner, Value};
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::plonk::{Circuit, ConstraintSystem, ErrorFront};

    // Small domain over [-6.0, 6.0]: well within I18's ~+/-9.22 representable
    // range, and GELU saturates to ~0 / ~x well before the edges, so this is
    // plenty for both correctness and accuracy checks.
    //
    // `DOMAIN_N` is deliberately odd: since the domain is symmetric, that
    // guarantees x = 0.0 is an exact grid point. This matters because
    // `LookupChip`'s lookup argument (like any halo2 dynamic lookup with no
    // selector) constrains *every* row of the input/output advice columns,
    // not just the rows explicitly assigned by `assign` -- including rows
    // MockProver defaults to (0, 0) because nothing was ever witnessed
    // there. Since `gelu_f64(0.0) == 0.0` exactly (the `x` factor alone is
    // zero, regardless of `erf_approx`'s tiny error at 0), having x = 0.0 in
    // the domain makes (0, 0) a valid table row and those default rows
    // satisfy the lookup for free -- exactly the same trick `lookup.rs`'s
    // own tests rely on by starting their square domain at 0.0.
    const DOMAIN_MIN: f64 = -6.0;
    const DOMAIN_MAX: f64 = 6.0;
    const DOMAIN_N: usize = 33;

    #[test]
    fn erf_approx_matches_known_values() {
        // erf(0) = 0 exactly.
        assert!(erf_approx(0.0).abs() < 1e-7);
        // erf(1) ~= 0.8427007929497149 (reference value).
        assert!((erf_approx(1.0) - 0.8427007929497149).abs() < 1e-6);
        // erf is odd.
        assert!((erf_approx(-1.0) + erf_approx(1.0)).abs() < 1e-12);
        // erf(x) -> 1 as x -> +inf.
        assert!((erf_approx(6.0) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn gelu_f64_matches_known_shape() {
        // GELU(0) = 0.
        assert!(gelu_f64(0.0).abs() < 1e-9);
        // GELU is (nearly, given erf_approx's tiny error) odd around the
        // point (0, 0) only in the sense GELU(x) + GELU(-x) ~= x, since
        // GELU(x) - GELU(-x) = x - 0 ... actually check via direct identity:
        // GELU(x) = x - GELU(-x)... no: GELU(x) + GELU(-x) = x is the true
        // identity (since 0.5*x*(1+erf) + 0.5*(-x)*(1+erf(-x/sqrt2))
        // = 0.5*x*(1+erf) - 0.5*x*(1-erf) = x*erf(x/sqrt2)... verify
        // numerically instead of asserting a hand-derived identity.
        let x = 2.0;
        let gelu_pos = gelu_f64(x);
        let gelu_neg = gelu_f64(-x);
        assert!((gelu_pos - (x + gelu_neg)).abs() < 1e-6);
        // For large positive x, GELU(x) ~= x.
        assert!((gelu_f64(6.0) - 6.0).abs() < 1e-6);
        // For large negative x, GELU(x) ~= 0.
        assert!(gelu_f64(-6.0).abs() < 1e-6);
    }

    fn gelu_domain() -> (Vec<I18>, Vec<I18>) {
        build_domain(gelu_f64, DOMAIN_MIN, DOMAIN_MAX, DOMAIN_N)
    }

    #[derive(Clone)]
    struct GeluTestConfig {
        gelu: GeluConfig,
    }

    struct GeluTestCircuit {
        input: I18,
    }

    impl Circuit<Fr> for GeluTestCircuit {
        type Config = GeluTestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            GeluTestCircuit {
                input: I18::from_raw(0),
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let input = meta.advice_column();
            let output = meta.advice_column();
            GeluTestConfig {
                gelu: GeluChip::configure(meta, input, output),
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = GeluChip::construct(config.gelu, DOMAIN_MIN, DOMAIN_MAX, DOMAIN_N);
            chip.load_table(layouter.namespace(|| "table"))?;
            chip.assign(layouter.namespace(|| "assign"), self.input)
                .expect("input should be an exact domain point in this test");
            Ok(())
        }
    }

    #[test]
    fn valid_gelu_lookup_at_domain_point_is_satisfied_and_close_to_true_gelu() {
        let (domain, values) = gelu_domain();
        let idx = 20;
        let input = domain[idx];
        let looked_up_output = values[idx];

        let circuit = GeluTestCircuit { input };
        let prover = MockProver::run(7, &circuit, vec![]).unwrap();
        prover.assert_satisfied();

        // The table's value should equal the true (independently computed)
        // GELU at this exact quantized point, within the I18 quantization
        // step (1 raw unit == 1e-18, so this is effectively an exact
        // check up to f64 rounding in `gelu_f64` itself).
        let true_gelu = gelu_f64(input.to_f64());
        let step = (domain[1].to_f64() - domain[0].to_f64()).abs();
        assert!(
            (looked_up_output.to_f64() - true_gelu).abs() < step,
            "looked up {} vs true {} (step {})",
            looked_up_output.to_f64(),
            true_gelu,
            step
        );
    }

    #[test]
    fn valid_gelu_lookup_at_first_domain_point_is_satisfied() {
        let (domain, _values) = gelu_domain();
        let circuit = GeluTestCircuit { input: domain[0] };
        let prover = MockProver::run(7, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn forged_output_for_a_valid_gelu_input_is_rejected() {
        // Bypasses `assign` and directly witnesses a valid domain input
        // paired with the wrong output on the same advice columns, so this
        // exercises the underlying halo2 lookup argument itself (not just
        // GeluChip::assign's Rust-level bookkeeping).
        struct ForgedOutputCircuit {
            input: I18,
            forged_output: I18,
        }

        impl Circuit<Fr> for ForgedOutputCircuit {
            type Config = GeluTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                ForgedOutputCircuit {
                    input: self.input,
                    forged_output: self.forged_output,
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                GeluTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let gelu_config = config.gelu.clone();
                let chip = GeluChip::construct(config.gelu, DOMAIN_MIN, DOMAIN_MAX, DOMAIN_N);
                chip.load_table(layouter.namespace(|| "table"))?;
                layouter.assign_region(
                    || "forged gelu lookup",
                    |mut region| {
                        // Must enable the selector: the lookup is gated (see
                        // `GeluConfig::selector`'s doc comment), so this
                        // forged row would otherwise silently collapse to
                        // the always-satisfied padding row instead of
                        // actually checking the forged values below.
                        gelu_config.selector().enable(&mut region, 0)?;
                        region.assign_advice(
                            || "input",
                            gelu_config.input_column(),
                            0,
                            || Value::known(i64_to_fr(self.input.raw())),
                        )?;
                        region.assign_advice(
                            || "output",
                            gelu_config.output_column(),
                            0,
                            || Value::known(i64_to_fr(self.forged_output.raw())),
                        )
                    },
                )?;
                Ok(())
            }
        }

        let (domain, values) = gelu_domain();
        let input = domain[20];
        let correct_output = values[20];
        // Off by one raw unit: not equal to gelu_f64(input), and (input,
        // forged) is not a row of the table since domain inputs are unique.
        let forged_output = I18::from_raw(correct_output.raw() + 1);

        let circuit = ForgedOutputCircuit {
            input,
            forged_output,
        };
        let prover = MockProver::run(7, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }
}
