use crate::field_convert::{i64_to_fr, Fr};
use crate::fixed_point::I18;
use halo2_proofs::circuit::{AssignedCell, Layouter, Value};
use halo2_proofs::plonk::{
    Advice, Column, ConstraintSystem, ErrorFront, Expression, Selector, TableColumn,
};
use halo2_proofs::poly::Rotation;
use std::fmt;

/// Sentinel raw `I18` value reserved as the lookup argument's "disabled row"
/// fallback (see [`LookupConfig`]). Chosen as `i64::MIN` because it is not a
/// value any realistic fixed-point domain point would legitimately take (it
/// corresponds to an astronomically large-magnitude negative real number),
/// so it can never collide with a genuine caller-supplied domain point.
const PAD_INPUT_RAW: i64 = i64::MIN;
/// Fallback output paired with [`PAD_INPUT_RAW`] in the reserved padding
/// table row. The value is arbitrary (never observed by a real query) but
/// fixed so the padding row is a single well-defined table entry.
const PAD_OUTPUT_RAW: i64 = 0;

/// Error raised by [`LookupChip::assign`] when the requested input is not an
/// exact point of the chip's quantized domain, or when the underlying circuit
/// synthesis fails.
#[derive(Debug)]
pub enum LookupError {
    /// The witnessed input does not exactly match any domain point loaded
    /// into the lookup table.
    InputNotInDomain(I18),
    /// Wraps a synthesis-time error from the halo2 layouter.
    Synthesis(ErrorFront),
}

impl fmt::Display for LookupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LookupError::InputNotInDomain(input) => {
                write!(
                    f,
                    "input {} is not a point of the lookup domain",
                    input.to_f64()
                )
            }
            LookupError::Synthesis(err) => write!(f, "lookup synthesis error: {err:?}"),
        }
    }
}

impl std::error::Error for LookupError {}

impl From<ErrorFront> for LookupError {
    fn from(err: ErrorFront) -> Self {
        LookupError::Synthesis(err)
    }
}

/// Configuration for a [`LookupChip`]: two advice columns (`input`, `output`)
/// constrained via a lookup argument to be a row of a fixed table
/// (`table_input`, `table_output`) representing a quantized function
/// `f: I18 -> I18`.
///
/// The lookup argument is gated by `selector`: on rows where `assign` enables
/// it, the actual `(input, output)` pair is checked against the table. On
/// every other row of the circuit (including rows this chip's columns never
/// touch, which halo2 treats as holding field-zero), the check instead
/// trivially collapses to the reserved `(PAD_INPUT_RAW, PAD_OUTPUT_RAW)` row
/// that `load_table` always loads. Without this gating, a halo2 lookup
/// argument applies unconditionally to *every* row of the whole circuit, so
/// any row this chip's dedicated columns don't explicitly assign would
/// otherwise have to satisfy `(0, 0) ∈ table` — which is false whenever the
/// domain doesn't happen to map `0 -> 0`, silently making the circuit
/// unsatisfiable for otherwise-correct witnesses. See the `padding` tests
/// below.
#[derive(Clone, Debug)]
pub struct LookupConfig {
    // Crate-visible (not fully private) so that composed chips such as
    // `EmbedLookupChip` (which configures several independent `LookupChip`s,
    // one per output dimension) and their tests can reach into the raw
    // region layout, e.g. to forge a witness for a negative test — mirroring
    // the pattern used by `DotProductConfig` in `dot_general.rs`.
    pub(crate) input: Column<Advice>,
    pub(crate) output: Column<Advice>,
    pub(crate) selector: Selector,
    table_input: TableColumn,
    table_output: TableColumn,
}

/// A generic chip that proves "I correctly evaluated a fixed, public,
/// quantized function `f` at a quantized point" via a halo2 lookup argument.
///
/// The chip is parameterized over the domain/value pairs supplied at
/// construction time, so the same chip machinery can back GELU, softmax's
/// `exp`, layer norm's `rsqrt`, or any other function with a precomputed
/// lookup table.
pub struct LookupChip {
    config: LookupConfig,
    /// `(domain point, f(domain point))` pairs, in the order they will be
    /// loaded into the table.
    domain: Vec<(I18, I18)>,
}

impl LookupChip {
    /// Configures the lookup argument: witnessed `(input, output)` pairs on
    /// the given advice columns must match a row of the table loaded by
    /// [`LookupChip::load_table`].
    pub fn configure(
        meta: &mut ConstraintSystem<Fr>,
        input: Column<Advice>,
        output: Column<Advice>,
    ) -> LookupConfig {
        meta.enable_equality(input);
        meta.enable_equality(output);

        // A complex selector is required (rather than a simple one) because
        // it appears inside a lookup argument expression below.
        let selector = meta.complex_selector();
        let table_input = meta.lookup_table_column();
        let table_output = meta.lookup_table_column();

        meta.lookup("input/output is a row of the function table", |meta| {
            let s = meta.query_selector(selector);
            let one = Expression::Constant(Fr::from(1u64));
            let input_expr = meta.query_advice(input, Rotation::cur());
            let output_expr = meta.query_advice(output, Rotation::cur());
            let pad_input = Expression::Constant(i64_to_fr(PAD_INPUT_RAW));
            let pad_output = Expression::Constant(i64_to_fr(PAD_OUTPUT_RAW));
            // On disabled rows (`s == 0`), the checked pair collapses to the
            // fixed `(PAD_INPUT_RAW, PAD_OUTPUT_RAW)` row, which `load_table`
            // always includes — trivially satisfied regardless of the
            // (irrelevant, possibly zero-valued) advice cells at that row.
            vec![
                (
                    s.clone() * input_expr + (one.clone() - s.clone()) * pad_input,
                    table_input,
                ),
                (
                    s.clone() * output_expr + (one - s) * pad_output,
                    table_output,
                ),
            ]
        });

        LookupConfig {
            input,
            output,
            selector,
            table_input,
            table_output,
        }
    }

    /// Builds a chip backed by the given quantized domain and precomputed
    /// function values (`values[i] == f(domain[i])`). `domain` and `values`
    /// must have the same length.
    pub fn construct(config: LookupConfig, domain: Vec<I18>, values: Vec<I18>) -> Self {
        assert_eq!(
            domain.len(),
            values.len(),
            "domain and values must have the same length"
        );
        let pairs = domain.into_iter().zip(values).collect();
        LookupChip {
            config,
            domain: pairs,
        }
    }

    /// Loads the fixed `(input, output)` table backing the lookup argument.
    /// Must be called exactly once per circuit synthesis, independently of
    /// any [`LookupChip::assign`] calls (which only witness rows to be
    /// checked against this table).
    pub fn load_table(&self, mut layouter: impl Layouter<Fr>) -> Result<(), ErrorFront> {
        layouter.assign_table(
            || "lookup function table",
            |mut table| {
                for (i, (input, output)) in self.domain.iter().enumerate() {
                    table.assign_cell(
                        || "table input",
                        self.config.table_input,
                        i,
                        || Value::known(i64_to_fr(input.raw())),
                    )?;
                    table.assign_cell(
                        || "table output",
                        self.config.table_output,
                        i,
                        || Value::known(i64_to_fr(output.raw())),
                    )?;
                }
                // Reserved padding row (see `LookupConfig`'s doc comment):
                // every row of the circuit outside an `assign`-enabled row
                // collapses to this pair, so it must always be a member of
                // the table, independently of the caller-supplied domain.
                let pad_row = self.domain.len();
                table.assign_cell(
                    || "table input (padding)",
                    self.config.table_input,
                    pad_row,
                    || Value::known(i64_to_fr(PAD_INPUT_RAW)),
                )?;
                table.assign_cell(
                    || "table output (padding)",
                    self.config.table_output,
                    pad_row,
                    || Value::known(i64_to_fr(PAD_OUTPUT_RAW)),
                )?;
                Ok(())
            },
        )
    }

    /// Witnesses `input` and its precomputed `f(input)` so that the lookup
    /// argument checks the pair against the table, and returns `f(input)`
    /// along with the `AssignedCell` holding the output -- composing chips
    /// (e.g. `SoftmaxChip`) must `region.constrain_equal` this cell to any
    /// cell where they re-witness the same value, rather than re-assigning
    /// it disconnected from this one (see the soundness notes in
    /// `chips/reduce.rs` and `chips/div.rs`).
    ///
    /// Returns [`LookupError::InputNotInDomain`] without touching the
    /// layouter if `input` is not an exact point of the chip's domain.
    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fr>,
        input: I18,
    ) -> Result<(I18, AssignedCell<Fr, Fr>), LookupError> {
        let output = self
            .domain
            .iter()
            .find(|(x, _)| x.raw() == input.raw())
            .map(|(_, y)| *y)
            .ok_or(LookupError::InputNotInDomain(input))?;

        let output_cell = layouter.assign_region(
            || "lookup assign",
            |mut region| {
                self.config.selector.enable(&mut region, 0)?;
                region.assign_advice(
                    || "input",
                    self.config.input,
                    0,
                    || Value::known(i64_to_fr(input.raw())),
                )?;
                region.assign_advice(
                    || "output",
                    self.config.output,
                    0,
                    || Value::known(i64_to_fr(output.raw())),
                )
            },
        )?;

        Ok((output, output_cell))
    }
}

/// Builds a domain from explicit RAW `I18` anchor points (`raw_points`),
/// evaluating `f` at each point's own `to_f64()` value.
///
/// # Why this exists, distinct from [`build_domain`]
///
/// `build_domain` derives every domain point via `I18::from_f64(some_f64)`.
/// That is fundamentally unable to hit an arbitrary, already-known raw `i64`
/// target exactly at magnitudes around `1e18`: `f64` multiplication by
/// `SCALE_18` (`1e18`, itself exactly representable) still produces a
/// *product* that is itself a representable `f64`, and `f64`'s ULP spacing
/// at that magnitude is roughly `128`-`256` raw units -- so most individual
/// raw integers (spaced `1` apart) are simply not reachable as `(value *
/// SCALE_18).round()` for *any* `f64` value, no matter how finely `value` is
/// searched. This was discovered empirically composing `RmsNormChip` against
/// a real, already-computed `mean(x^2) + epsilon` raw value (see
/// `crates/zkie-compiler/tests/rms_norm_fintext_real_weights.rs` and the
/// accompanying report) -- a real, non-hypothetical gap in `build_domain`'s
/// applicability, not merely a theoretical concern. This function sidesteps
/// it entirely by constructing domain points directly via [`I18::from_raw`]
/// (exact, no `f64` round-trip at all) -- [`LookupChip::assign`]'s own match
/// is already a plain `raw() == raw()` integer comparison (see its
/// implementation above), so a domain built this way works identically to
/// one built via `build_domain`, just without the reachability gap.
pub fn build_domain_from_raw(f: impl Fn(f64) -> f64, raw_points: &[i64]) -> (Vec<I18>, Vec<I18>) {
    let domain: Vec<I18> = raw_points.iter().map(|&r| I18::from_raw(r)).collect();
    let values: Vec<I18> = domain
        .iter()
        .map(|d| I18::from_f64(f(d.to_f64())).expect("function value must fit in I18 range"))
        .collect();
    (domain, values)
}

/// Evenly quantizes `[min, max]` into `n` points and evaluates `f` (a host
/// side `f64 -> f64` function, e.g. `f64::exp`, a GELU implementation, or
/// `|v| 1.0 / v.sqrt()`) at each, producing the `(domain, values)` I18
/// vectors used to build a [`LookupChip`].
pub fn build_domain(f: impl Fn(f64) -> f64, min: f64, max: f64, n: usize) -> (Vec<I18>, Vec<I18>) {
    assert!(n > 0, "domain must have at least one point");
    let mut domain = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    for i in 0..n {
        let t = if n == 1 {
            0.0
        } else {
            i as f64 / (n - 1) as f64
        };
        let x = min + t * (max - min);
        let y = f(x);
        domain.push(I18::from_f64(x).expect("domain point must fit in I18 range"));
        values.push(I18::from_f64(y).expect("function value must fit in I18 range"));
    }
    (domain, values)
}

#[cfg(test)]
mod build_domain_from_raw_tests {
    use super::*;

    #[test]
    fn anchors_exactly_at_an_unreachable_via_f64_raw_target() {
        // A raw value deliberately NOT of the form `(v * 1e18).round()` for
        // any convenient `f64` `v` -- an arbitrary large-magnitude odd raw
        // integer, matching the real-world shape this function exists for
        // (see its own module docs).
        let target_raw: i64 = 1_012_174_153_815_396_446;
        let (domain, values) = build_domain_from_raw(|x| 1.0 / x.sqrt(), &[target_raw]);
        assert_eq!(domain.len(), 1);
        assert_eq!(domain[0].raw(), target_raw);
        // Sanity: the looked-up value is a real rsqrt evaluation, not zero
        // or garbage.
        let expected = 1.0 / I18::from_raw(target_raw).to_f64().sqrt();
        assert!((values[0].to_f64() - expected).abs() < 1e-9);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::circuit::SimpleFloorPlanner;
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::plonk::{Circuit, ConstraintSystem, ErrorFront};

    /// A small 16-point domain over [0.0, 2.0] with f(x) = x * x (max output
    /// 4.0, comfortably within I18's representable range), in the spirit of
    /// the spike this task references.
    fn square_domain() -> (Vec<I18>, Vec<I18>) {
        build_domain(|x| x * x, 0.0, 2.0, 16)
    }

    #[derive(Clone)]
    struct LookupTestConfig {
        lookup: LookupConfig,
    }

    struct LookupTestCircuit {
        domain: Vec<I18>,
        values: Vec<I18>,
        input: I18,
    }

    impl Circuit<Fr> for LookupTestCircuit {
        type Config = LookupTestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            LookupTestCircuit {
                domain: self.domain.clone(),
                values: self.values.clone(),
                input: I18::from_raw(0),
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let input = meta.advice_column();
            let output = meta.advice_column();
            LookupTestConfig {
                lookup: LookupChip::configure(meta, input, output),
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip =
                LookupChip::construct(config.lookup, self.domain.clone(), self.values.clone());
            chip.load_table(layouter.namespace(|| "table"))?;
            chip.assign(layouter.namespace(|| "assign"), self.input)
                .expect("input should be an exact domain point in this test");
            Ok(())
        }
    }

    #[test]
    fn valid_lookup_at_domain_point_is_satisfied() {
        let (domain, values) = square_domain();
        let circuit = LookupTestCircuit {
            input: domain[5],
            domain,
            values,
        };
        let prover = MockProver::run(6, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn valid_lookup_at_first_domain_point_is_satisfied() {
        let (domain, values) = square_domain();
        let circuit = LookupTestCircuit {
            input: domain[0],
            domain,
            values,
        };
        let prover = MockProver::run(6, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn assign_rejects_input_not_in_domain_at_the_rust_level() {
        // Exercises the host-side domain-membership guard directly: `assign`
        // must refuse an out-of-domain input by returning a typed error,
        // without ever generating an invalid circuit region.
        struct GuardTestCircuit {
            domain: Vec<I18>,
            values: Vec<I18>,
            off_domain_input: I18,
        }

        impl Circuit<Fr> for GuardTestCircuit {
            type Config = LookupTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                GuardTestCircuit {
                    domain: self.domain.clone(),
                    values: self.values.clone(),
                    off_domain_input: self.off_domain_input,
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                LookupTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let chip =
                    LookupChip::construct(config.lookup, self.domain.clone(), self.values.clone());
                chip.load_table(layouter.namespace(|| "table"))?;
                match chip.assign(layouter.namespace(|| "assign"), self.off_domain_input) {
                    Err(LookupError::InputNotInDomain(input)) => {
                        assert_eq!(input, self.off_domain_input);
                    }
                    other => panic!("expected InputNotInDomain, got {other:?}"),
                }
                Ok(())
            }
        }

        let (domain, values) = square_domain();
        // Halfway between two grid points: not an exact domain point.
        let step = (domain[1].to_f64() - domain[0].to_f64()) / 2.0;
        let off_domain_input = I18::from_f64(domain[0].to_f64() + step).unwrap();

        let circuit = GuardTestCircuit {
            domain,
            values,
            off_domain_input,
        };
        let prover = MockProver::run(6, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn forged_output_for_a_valid_input_is_rejected() {
        // Bypasses `assign` and directly witnesses a valid domain input
        // paired with the wrong output, confirming the lookup argument
        // (not just Rust-level bookkeeping) enforces correctness.
        struct ForgedOutputCircuit {
            domain: Vec<I18>,
            values: Vec<I18>,
            input: I18,
            forged_output: I18,
        }

        impl Circuit<Fr> for ForgedOutputCircuit {
            type Config = LookupTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                ForgedOutputCircuit {
                    domain: self.domain.clone(),
                    values: self.values.clone(),
                    input: self.input,
                    forged_output: self.forged_output,
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                LookupTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let lookup_config = config.lookup.clone();
                let chip =
                    LookupChip::construct(config.lookup, self.domain.clone(), self.values.clone());
                chip.load_table(layouter.namespace(|| "table"))?;
                layouter.assign_region(
                    || "forged lookup",
                    |mut region| {
                        // Must enable the selector here too: the lookup is
                        // gated (see `LookupConfig`'s doc comment), so this
                        // forged row would otherwise silently collapse to
                        // the always-satisfied padding row instead of
                        // actually checking the forged values below.
                        lookup_config.selector.enable(&mut region, 0)?;
                        region.assign_advice(
                            || "input",
                            lookup_config.input,
                            0,
                            || Value::known(i64_to_fr(self.input.raw())),
                        )?;
                        region.assign_advice(
                            || "output",
                            lookup_config.output,
                            0,
                            || Value::known(i64_to_fr(self.forged_output.raw())),
                        )
                    },
                )?;
                Ok(())
            }
        }

        let (domain, values) = square_domain();
        let input = domain[5];
        let correct_output = values[5];
        // Off by one raw unit: not equal to f(input), and (input, forged)
        // is not a row of the table since domain inputs are unique.
        let forged_output = I18::from_raw(correct_output.raw() + 1);

        let circuit = ForgedOutputCircuit {
            domain,
            values,
            input,
            forged_output,
        };
        let prover = MockProver::run(6, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    fn forged_input_not_in_domain_is_rejected() {
        // Bypasses `assign` and directly witnesses an input that is not any
        // domain point at all, confirming MockProver rejects it regardless
        // of what output is paired with it.
        struct ForgedInputCircuit {
            domain: Vec<I18>,
            values: Vec<I18>,
            off_domain_input: I18,
            paired_output: I18,
        }

        impl Circuit<Fr> for ForgedInputCircuit {
            type Config = LookupTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                ForgedInputCircuit {
                    domain: self.domain.clone(),
                    values: self.values.clone(),
                    off_domain_input: self.off_domain_input,
                    paired_output: self.paired_output,
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                LookupTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let lookup_config = config.lookup.clone();
                let chip =
                    LookupChip::construct(config.lookup, self.domain.clone(), self.values.clone());
                chip.load_table(layouter.namespace(|| "table"))?;
                layouter.assign_region(
                    || "forged lookup",
                    |mut region| {
                        // See the comment in `forged_output_for_a_valid_input_is_rejected`:
                        // the lookup is gated, so the selector must be
                        // enabled for this row's forged values to actually
                        // be checked.
                        lookup_config.selector.enable(&mut region, 0)?;
                        region.assign_advice(
                            || "input",
                            lookup_config.input,
                            0,
                            || Value::known(i64_to_fr(self.off_domain_input.raw())),
                        )?;
                        region.assign_advice(
                            || "output",
                            lookup_config.output,
                            0,
                            || Value::known(i64_to_fr(self.paired_output.raw())),
                        )
                    },
                )?;
                Ok(())
            }
        }

        let (domain, values) = square_domain();
        let step = (domain[1].to_f64() - domain[0].to_f64()) / 2.0;
        let off_domain_input = I18::from_f64(domain[0].to_f64() + step).unwrap();

        let paired_output = values[0];
        let circuit = ForgedInputCircuit {
            domain,
            values,
            off_domain_input,
            paired_output,
        };
        let prover = MockProver::run(6, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    fn domain_not_mapping_zero_to_zero_is_still_satisfied() {
        // Regression test for the "unassigned rows default to (0, 0)" trap:
        // a halo2 lookup argument applies to every row of the circuit, not
        // just the one row `assign` touches. Before the selector-gating fix,
        // this would fail on every one of the circuit's other (unassigned,
        // hence zero-valued) rows unless the domain happened to map 0 -> 0.
        // This domain deliberately does not: it starts at raw input 1.
        let domain: Vec<I18> = (1..=16).map(I18::from_raw).collect();
        let values: Vec<I18> = domain.iter().map(|x| I18::from_raw(x.raw() * 10)).collect();
        let circuit = LookupTestCircuit {
            input: domain[3],
            domain,
            values,
        };
        let prover = MockProver::run(6, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}
