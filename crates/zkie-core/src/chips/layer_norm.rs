//! `LayerNormChip`: backs the `LayerNorm` instruction (see `crate::isa`) --
//! layer normalization over a fixed, compile-time-known count `K` of I18
//! inputs:
//!
//! ```text
//! mean     = (1/K) * sum_i x_i
//! variance = (1/K) * sum_i (x_i - mean)^2
//! output_i = (x_i - mean) * rsqrt(variance + epsilon)
//! ```
//!
//! ## Scope: no learnable affine parameters
//!
//! This chip deliberately does **not** implement the usual `gamma * output_i +
//! beta` learnable affine transform that follows normalization in most real
//! layer-norm implementations. That is an explicit, deliberate scope decision
//! for this foundation layer (not an oversight): `gamma`/`beta` would just be
//! one more `EltwiseMulChip`/`EltwiseAddChip` pair per output element once
//! per-channel weight vectors are threaded through the ISA, and adding that
//! is straightforward follow-up work once the surrounding instruction set
//! has a place to carry those learned weights. Keeping this chip to the
//! normalization core (mean/variance/rsqrt/rescale) keeps it composable and
//! easy to verify in isolation first.
//!
//! ## Composition strategy
//!
//! Structurally this chip is a pipeline of already-existing, already-sound
//! building blocks, threaded together the same way
//! [`crate::chips::patch_embed::PatchEmbedChip`] reuses a single configured
//! [`crate::chips::dot_general::DotProductChip`] across several sequential
//! regions rather than configuring independent copies:
//!
//! 1. [`crate::chips::reduce::ReduceMeanChip`] computes `mean` over the `K`
//!    inputs.
//! 2. [`crate::chips::eltwise::EltwiseAddChip`] computes each `diff_i = x_i -
//!    mean` (as `x_i + (-mean)`, since subtraction is just addition with a
//!    host-negated operand -- I18's field-natural sign handling means no new
//!    circuit machinery is needed for this).
//! 3. [`crate::chips::eltwise::EltwiseMulChip`] computes each `sq_i =
//!    diff_i^2` (`diff_i * diff_i`).
//! 4. The **same** `ReduceMeanChip` instance (reused, not reconfigured --
//!    it's data-independent: mean of `K` values is mean of `K` values,
//!    whether they're the raw inputs or their squared deviations) computes
//!    `variance` over the `sq_i` values.
//! 5. `EltwiseAddChip` (the same instance again) adds the constant `epsilon`
//!    to `variance`.
//! 6. A small dedicated [`RsqrtChip`] (a thin lookup-backed wrapper in the
//!    exact spirit of [`crate::chips::gelu::GeluChip`]) looks up
//!    `rsqrt(variance + epsilon) = 1 / sqrt(variance + epsilon)`.
//! 7. `EltwiseMulChip` (the same instance used for squaring) computes each
//!    `output_i = diff_i * rsqrt_value`.
//!
//! Because [`crate::chips::eltwise::EltwiseAddChip::assign`] and
//! [`crate::chips::eltwise::EltwiseMulChip::assign`] only *witness and
//! constrain* a region from host-supplied `I18` values (they don't hand back
//! an `AssignedCell` for their output the way
//! [`crate::chips::dot_general::DotProductChip::assign`] or
//! [`crate::chips::reduce::ReduceMeanChip::assign`] do), this chip's `assign`
//! must independently recompute each intermediate host-side value (`diff_i`,
//! `sq_i`, `variance_plus_eps`, `output_i`) using *exactly* the same
//! arithmetic those chips' own gates enforce (plain `i64` addition for sums,
//! [`crate::fixed_point::requantize_mul`] for products) before feeding it
//! into the next step. This mirrors the same pattern
//! [`crate::chips::reduce::ReduceMeanChip::assign`] already uses internally
//! (it recomputes `mean`/`r`/`slack` host-side with the identical formula its
//! own gate checks, then witnesses those exact values) -- there is no gap
//! here relative to the rest of the crate, just the same discipline applied
//! one level up the composition.
//!
//! ## CRITICAL NUMERIC LIMITATION -- I18's representable range
//!
//! I18 is backed by a plain `i64` at scale `10^18`, giving a representable
//! range of only about `±9.22` in real terms (`i64::MAX / 1e18`), with **no**
//! wider accumulator type at the host level for this chip's intermediate
//! values. Every one of `x_i`, `mean`, `variance`, `rsqrt(variance +
//! epsilon)`, and each `output_i` must individually stay within that range,
//! or the corresponding `requantize_mul`/`checked_add` call inside `assign`
//! panics (`.expect(...)`) rather than silently wrapping.
//!
//! This chip's own tests use `K = 4` and inputs in `[-1.5, 1.5]`, which keeps
//! every intermediate comfortably in range: `diff_i` up to `3.0` in
//! magnitude, `sq_i` up to `9.0` (well under `9.22`, but leaves very little
//! headroom -- a wider input range would risk overflowing the square step
//! well before it overflows the inputs themselves), `variance` a mean of
//! those squares (so no larger than the largest square), and
//! `rsqrt(variance + epsilon)` bounded by the *lower* end of the domain (see
//! below). Callers choosing their own `K`/input ranges must re-derive these
//! bounds for their own case; nothing here checks them beyond the panics
//! `requantize_mul`/`checked_add` raise on actual overflow.
//!
//! ## CRITICAL NUMERIC LIMITATION -- the rsqrt lookup domain
//!
//! `rsqrt(v) = 1/sqrt(v)` blows up as `v -> 0`, so the domain lower bound
//! must be chosen carefully: e.g. `rsqrt(0.0001) = 100`, which already
//! overflows I18's `~9.22` range on its own, long before any multiplication
//! by `diff_i` even happens. This chip's tests use a domain of `[0.1, 3.0]`,
//! giving `rsqrt` outputs in `[rsqrt(3.0), rsqrt(0.1)] ~= [0.577, 3.162]` --
//! comfortably within range with a wide margin.
//!
//! This means, as a real and deliberately-not-silently-papered-over
//! limitation of this foundation layer: **if the true `variance + epsilon`
//! for a given `K` inputs falls below the configured domain's lower bound
//! (e.g. all `K` inputs nearly identical, driving `variance` towards zero),
//! [`LayerNormChip::assign`] returns [`LayerNormError::Rsqrt`] rather than a
//! result**, because [`crate::chips::lookup::LookupChip`] only proves
//! membership of an *exact* quantized domain point -- it has no "nearest
//! point" snapping or extrapolation. Widening the domain outward (lower
//! `rsqrt_domain_min`) is the fix for callers who need to support
//! near-degenerate (very low variance) inputs, at the cost of needing a
//! larger `rsqrt_domain_n` to keep the same quantization density, and of
//! `rsqrt`'s own output-range headroom shrinking as the lower bound drops.
//!
//! Relatedly, because [`crate::chips::lookup::LookupChip::assign`] requires
//! its query to be an *exact* point of the quantized domain (not merely
//! within `[rsqrt_domain_min, rsqrt_domain_max]`), a `variance + epsilon`
//! that falls strictly between two domain grid points is *also* rejected
//! with [`LayerNormError::Rsqrt`] -- exactly the same "quantization grid"
//! limitation already present in [`crate::chips::gelu::GeluChip`] (see that
//! module's docs). Production use of this foundation layer needs either a
//! much finer grid tuned to the upstream quantization scheme in use, or a
//! "snap to nearest domain point plus a proximity proof" mechanism -- future
//! work, out of scope here.
//!
//! ## CRITICAL NUMERIC LIMITATION -- epsilon's milli-unit granularity
//!
//! Per [`crate::isa::Instruction::LayerNorm`]'s `epsilon_milli` field,
//! epsilon is expressed as `epsilon_milli as f64 / 1000.0`, i.e. in
//! thousandths. That gives a minimum representable epsilon of `0.001`, far
//! coarser than the tiny epsilons (typically `1e-5` to `1e-8`) real-world
//! layer-norm implementations use to guard against division by (near) zero.
//! This is an accepted limitation of the current milli-unit ISA convention,
//! not something this chip works around -- a finer-grained epsilon
//! convention would need a wider integer field or the sort of `I18`-based
//! (rather than milli-unit) epsilon input this codebase's other constants
//! use elsewhere.

use crate::chips::eltwise::{EltwiseAddChip, EltwiseAddConfig, EltwiseMulChip, EltwiseMulConfig};
use crate::chips::lookup::{build_domain, LookupChip, LookupConfig, LookupError};
use crate::chips::reduce::{ReduceMeanChip, ReduceMeanConfig};
use crate::field_convert::Fr;
use crate::fixed_point::{requantize_mul, I18};
use halo2_proofs::circuit::Layouter;
use halo2_proofs::plonk::{Advice, Column, ConstraintSystem, ErrorFront, Selector};
use std::fmt;

/// Host-side (`f64`) `rsqrt(x) = 1 / sqrt(x)`, used only to *generate* the
/// [`RsqrtChip`] lookup table at circuit-construction time -- it never runs
/// inside the circuit itself (see the module-level docs on
/// [`crate::chips::gelu::GeluChip`] for why this is a modeling/accuracy
/// concern, not a soundness one: the analogous discussion applies here
/// verbatim).
pub fn rsqrt_f64(x: f64) -> f64 {
    1.0 / x.sqrt()
}

/// Configuration for an [`RsqrtChip`]. Thin wrapper around
/// [`LookupConfig`], mirroring
/// [`crate::chips::gelu::GeluConfig`] exactly, including its accessor
/// methods for tests that need to witness a row on these columns directly
/// (bypassing [`RsqrtChip::assign`]) to probe the underlying lookup argument
/// itself.
#[derive(Clone)]
pub struct RsqrtConfig {
    lookup: LookupConfig,
    input: Column<Advice>,
    output: Column<Advice>,
}

impl RsqrtConfig {
    /// The advice column `RsqrtChip::assign` witnesses its input on.
    pub fn input_column(&self) -> Column<Advice> {
        self.input
    }

    /// The advice column `RsqrtChip::assign` witnesses its output on.
    pub fn output_column(&self) -> Column<Advice> {
        self.output
    }

    /// The selector gating the lookup argument (see `LookupConfig`'s doc
    /// comment in `chips/lookup.rs`): callers witnessing a row on
    /// `input_column()`/`output_column()` directly must enable this
    /// selector, or the lookup argument silently collapses that row to the
    /// always-satisfied padding entry instead of actually checking it.
    pub fn selector(&self) -> Selector {
        self.lookup.selector
    }
}

/// A chip that proves "I looked up the exact precomputed `rsqrt` value for
/// this exact quantized input" via a halo2 lookup argument -- a thin
/// [`LookupChip`] wrapper in the exact spirit of
/// [`crate::chips::gelu::GeluChip`]. See this module's numeric-limitation
/// docs for how the domain bounds must be chosen to keep `rsqrt`'s output in
/// I18's representable range.
pub struct RsqrtChip {
    inner: LookupChip,
}

impl RsqrtChip {
    /// Configures the lookup argument backing this chip. Delegates directly
    /// to [`LookupChip::configure`].
    pub fn configure(
        meta: &mut ConstraintSystem<Fr>,
        input: Column<Advice>,
        output: Column<Advice>,
    ) -> RsqrtConfig {
        let lookup = LookupChip::configure(meta, input, output);
        RsqrtConfig {
            lookup,
            input,
            output,
        }
    }

    /// Builds a chip backed by an `rsqrt` table: `n` points evenly
    /// quantized over `[domain_min, domain_max]` (`domain_min` must be
    /// strictly positive -- `rsqrt` is undefined at/below zero), each paired
    /// with `rsqrt_f64(x)` computed host-side.
    pub fn construct(config: RsqrtConfig, domain_min: f64, domain_max: f64, n: usize) -> Self {
        assert!(
            domain_min > 0.0,
            "rsqrt domain minimum must be strictly positive"
        );
        let (domain, values) = build_domain(rsqrt_f64, domain_min, domain_max, n);
        let inner = LookupChip::construct(config.lookup, domain, values);
        RsqrtChip { inner }
    }

    /// Loads the fixed `rsqrt` table backing the lookup argument. Must be
    /// called exactly once per circuit synthesis. Delegates to
    /// [`LookupChip::load_table`].
    pub fn load_table(&self, layouter: impl Layouter<Fr>) -> Result<(), ErrorFront> {
        self.inner.load_table(layouter)
    }

    /// Witnesses `input` and its precomputed `rsqrt_f64(input)`, checked by
    /// the lookup argument, and returns the looked-up output. Delegates to
    /// [`LookupChip::assign`], discarding the `AssignedCell` it also returns
    /// (this chip's own output is independently recomputed and re-witnessed
    /// by its caller, the same discipline `EltwiseAddChip`/`EltwiseMulChip`
    /// already require of their callers -- see `LayerNormChip::assign`).
    pub fn assign(&self, layouter: impl Layouter<Fr>, input: I18) -> Result<I18, LookupError> {
        self.inner
            .assign(layouter, input)
            .map(|(value, _cell)| value)
    }
}

/// Errors that can occur while assigning a [`LayerNormChip`] region.
#[derive(Debug)]
pub enum LayerNormError {
    /// `inputs` did not have exactly the configured `k` elements.
    InputCountMismatch { expected: usize, got: usize },
    /// A wrapped synthesis-time error from one of the composed
    /// (non-lookup-backed) sub-chips (`ReduceMeanChip`, `EltwiseAddChip`,
    /// `EltwiseMulChip`).
    Synthesis(ErrorFront),
    /// The `variance + epsilon` value was not an exact point of the
    /// configured `rsqrt` lookup domain (see this module's docs for why
    /// this is a real, deliberate limitation of the current foundation
    /// layer, not a bug).
    Rsqrt(LookupError),
}

impl fmt::Display for LayerNormError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerNormError::InputCountMismatch { expected, got } => {
                write!(f, "layer norm expects {expected} inputs (k), got {got}")
            }
            LayerNormError::Synthesis(err) => write!(f, "layer norm synthesis error: {err:?}"),
            LayerNormError::Rsqrt(err) => write!(f, "layer norm rsqrt lookup failed: {err}"),
        }
    }
}

impl std::error::Error for LayerNormError {}

impl From<ErrorFront> for LayerNormError {
    fn from(err: ErrorFront) -> Self {
        LayerNormError::Synthesis(err)
    }
}

/// Configuration for a [`LayerNormChip`]: a single [`ReduceMeanConfig`]
/// (reused for both the mean-of-inputs and mean-of-squared-deviations
/// steps, since both are just "mean of `k` values"), a single
/// [`EltwiseAddConfig`] (reused for each `x_i - mean` and for `variance +
/// epsilon`), a single [`EltwiseMulConfig`] (reused for each `diff_i^2` and
/// each final `diff_i * rsqrt_value`), and an [`RsqrtConfig`] for the
/// `rsqrt` lookup -- mirroring how
/// [`crate::chips::patch_embed::PatchEmbedChip`] reuses one configured
/// [`crate::chips::dot_general::DotProductChip`] across several regions
/// rather than configuring independent copies per use site.
#[derive(Clone)]
pub struct LayerNormConfig {
    reduce_mean: ReduceMeanConfig,
    add: EltwiseAddConfig,
    mul: EltwiseMulConfig,
    rsqrt: RsqrtConfig,
    k: usize,
    epsilon: I18,
    rsqrt_domain_min: f64,
    rsqrt_domain_max: f64,
    rsqrt_domain_n: usize,
}

pub struct LayerNormChip {
    config: LayerNormConfig,
    rsqrt_chip: RsqrtChip,
}

impl LayerNormChip {
    /// `k` is the compile-time-known input count (fixed per configured
    /// circuit, matching `ReduceMeanChip::configure`'s `k`). `epsilon_milli`
    /// is epsilon expressed in thousandths (`epsilon = epsilon_milli as f64
    /// / 1000.0`), per `Instruction::LayerNorm`'s convention -- baked in
    /// here as a compile-time constant (like `ReduceMeanChip`'s reciprocal),
    /// not a witnessed circuit input, since it is fixed once the instruction
    /// is known. `rsqrt_domain_min`/`rsqrt_domain_max`/`rsqrt_domain_n`
    /// configure the underlying `RsqrtChip`'s lookup domain -- see this
    /// module's numeric-limitation docs for how to choose them safely.
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
        k: usize,
        epsilon_milli: u64,
        rsqrt_domain_min: f64,
        rsqrt_domain_max: f64,
        rsqrt_domain_n: usize,
    ) -> LayerNormConfig {
        assert!(k >= 1, "LayerNormChip requires at least one input");

        // Shared `bits` column across all three composed chips' internal
        // range checks -- safe because `RangeCheckChip::configure` creates
        // fresh, independently-gated selectors on each call (see
        // `ReduceMeanChip::configure`'s own reuse of a single `bits` column
        // across its three range checks for the same reasoning).
        let reduce_mean = ReduceMeanChip::configure(
            meta, values, sum, sum_shift, mean_q, mean_r, mean_slack, bits, k,
        );
        let add = EltwiseAddChip::configure(meta, add_a, add_b, add_c, bits);
        let mul = EltwiseMulChip::configure(meta, mul_a, mul_b, mul_q, mul_r, mul_slack, bits);
        let rsqrt = RsqrtChip::configure(meta, rsqrt_input, rsqrt_output);

        let epsilon = I18::from_f64(epsilon_milli as f64 / 1000.0)
            .expect("epsilon_milli / 1000 must be representable as an I18 fixed-point value");

        LayerNormConfig {
            reduce_mean,
            add,
            mul,
            rsqrt,
            k,
            epsilon,
            rsqrt_domain_min,
            rsqrt_domain_max,
            rsqrt_domain_n,
        }
    }

    pub fn construct(config: LayerNormConfig) -> Self {
        let rsqrt_chip = RsqrtChip::construct(
            config.rsqrt.clone(),
            config.rsqrt_domain_min,
            config.rsqrt_domain_max,
            config.rsqrt_domain_n,
        );
        LayerNormChip { config, rsqrt_chip }
    }

    /// Loads the fixed `rsqrt` table backing this chip's lookup argument.
    /// Must be called exactly once per circuit synthesis, independently of
    /// how many times `assign` is called. Delegates to
    /// [`RsqrtChip::load_table`].
    pub fn load_table(&self, layouter: impl Layouter<Fr>) -> Result<(), ErrorFront> {
        self.rsqrt_chip.load_table(layouter)
    }

    /// Assigns the full layer-norm pipeline for `inputs` (must have length
    /// `k`, the count fixed at `configure` time) and returns the
    /// length-`k` normalized output vector. See the module-level docs for
    /// the exact sequence of composed sub-chip calls and why each
    /// intermediate value is independently recomputed host-side (matching
    /// the sub-chips' own gate arithmetic exactly) between calls.
    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fr>,
        inputs: &[I18],
    ) -> Result<Vec<I18>, LayerNormError> {
        let k = self.config.k;
        if inputs.len() != k {
            return Err(LayerNormError::InputCountMismatch {
                expected: k,
                got: inputs.len(),
            });
        }

        let mean_chip = ReduceMeanChip::construct(self.config.reduce_mean.clone());
        let add_chip = EltwiseAddChip::construct(self.config.add.clone());
        let mul_chip = EltwiseMulChip::construct(self.config.mul.clone());

        // Step 1: mean = (1/k) * sum_i x_i.
        let mean = mean_chip.assign(layouter.namespace(|| "layer norm mean"), inputs)?;

        // Step 2: diff_i = x_i - mean = x_i + (-mean), for each i. `assign`
        // only witnesses/constrains the region -- it does not hand back the
        // sum -- so `diff_i` is independently recomputed here with the
        // exact same `checked_add` arithmetic `EltwiseAddChip::assign` uses
        // internally, guaranteeing this value matches what the gate
        // enforces.
        let neg_mean = I18::from_raw(-mean.raw());
        let mut diffs = Vec::with_capacity(k);
        for (i, x) in inputs.iter().enumerate() {
            add_chip.assign(
                layouter.namespace(|| format!("layer norm diff {i}")),
                *x,
                neg_mean,
            )?;
            let diff_raw = x
                .raw()
                .checked_add(neg_mean.raw())
                .expect("I18 layer norm diff overflow");
            diffs.push(I18::from_raw(diff_raw));
        }

        // Step 3: sq_i = diff_i^2, for each i.
        let mut squares = Vec::with_capacity(k);
        for (i, diff) in diffs.iter().enumerate() {
            mul_chip.assign(
                layouter.namespace(|| format!("layer norm square {i}")),
                *diff,
                *diff,
            )?;
            let (sq, _) = requantize_mul(*diff, *diff).expect("I18 layer norm square overflow");
            squares.push(sq);
        }

        // Step 4: variance = (1/k) * sum_i sq_i (same `mean_chip` instance,
        // reused: mean of k values is mean of k values, whether they're the
        // raw inputs or their squared deviations).
        let variance = mean_chip.assign(layouter.namespace(|| "layer norm variance"), &squares)?;

        // Step 5: variance_plus_eps = variance + epsilon.
        add_chip.assign(
            layouter.namespace(|| "layer norm variance plus epsilon"),
            variance,
            self.config.epsilon,
        )?;
        let variance_plus_eps_raw = variance
            .raw()
            .checked_add(self.config.epsilon.raw())
            .expect("I18 layer norm variance+epsilon overflow");
        let variance_plus_eps = I18::from_raw(variance_plus_eps_raw);

        // Step 6: rsqrt_value = rsqrt(variance_plus_eps), via the lookup
        // argument.
        let rsqrt_value = self
            .rsqrt_chip
            .assign(layouter.namespace(|| "layer norm rsqrt"), variance_plus_eps)
            .map_err(LayerNormError::Rsqrt)?;

        // Step 7: output_i = diff_i * rsqrt_value, for each i (same
        // `mul_chip` instance, reused from the squaring step).
        let mut outputs = Vec::with_capacity(k);
        for (i, diff) in diffs.iter().enumerate() {
            mul_chip.assign(
                layouter.namespace(|| format!("layer norm scale {i}")),
                *diff,
                rsqrt_value,
            )?;
            let (out, _) =
                requantize_mul(*diff, rsqrt_value).expect("I18 layer norm output overflow");
            outputs.push(out);
        }

        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_convert::i64_to_fr;
    use halo2_proofs::circuit::{SimpleFloorPlanner, Value};
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::plonk::{Circuit, ConstraintSystem, ErrorFront};

    const K: usize = 4;
    // epsilon = 10 / 1000 = 0.01.
    const EPSILON_MILLI: u64 = 10;
    // Domain chosen so that rsqrt's output stays comfortably within I18's
    // representable range: rsqrt(0.1) ~= 3.162, rsqrt(3.0) ~= 0.577 -- see
    // this module's "CRITICAL NUMERIC LIMITATION" docs for the full
    // reasoning, including why variance below 0.1 (e.g. near-identical
    // inputs) is out of scope for this particular test domain.
    const RSQRT_DOMAIN_MIN: f64 = 0.1;
    const RSQRT_DOMAIN_MAX: f64 = 3.0;
    // Chosen (see this module's development notes / task derivation) so the
    // domain's evenly-spaced grid includes x = 1.26 (index 40) exactly --
    // matching the exact `variance + epsilon` this test's chosen inputs
    // produce (verified independently: sample_inputs()'s mean is exactly
    // 0.0, variance exactly 1.25, so variance + epsilon = 1.26 exactly, all
    // with zero fixed-point remainder at every step).
    const RSQRT_DOMAIN_N: usize = 101;

    const CIRCUIT_K: u32 = 12;

    fn sample_inputs() -> Vec<I18> {
        vec![
            I18::from_f64(-1.5).unwrap(),
            I18::from_f64(-0.5).unwrap(),
            I18::from_f64(0.5).unwrap(),
            I18::from_f64(1.5).unwrap(),
        ]
    }

    #[derive(Clone)]
    struct LayerNormTestConfig {
        layer_norm: LayerNormConfig,
    }

    struct LayerNormTestCircuit {
        inputs: Vec<I18>,
    }

    impl Circuit<Fr> for LayerNormTestCircuit {
        type Config = LayerNormTestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            LayerNormTestCircuit {
                inputs: vec![I18::from_raw(0); self.inputs.len()],
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
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

            LayerNormTestConfig {
                layer_norm: LayerNormChip::configure(
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
                    K,
                    EPSILON_MILLI,
                    RSQRT_DOMAIN_MIN,
                    RSQRT_DOMAIN_MAX,
                    RSQRT_DOMAIN_N,
                ),
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = LayerNormChip::construct(config.layer_norm);
            chip.load_table(layouter.namespace(|| "table"))?;
            chip.assign(layouter.namespace(|| "assign"), &self.inputs)
                .expect("layer norm assign should not fail in this test");
            Ok(())
        }
    }

    #[test]
    fn layer_norm_of_four_symmetric_inputs_is_satisfied_and_close_to_true_layer_norm() {
        let inputs = sample_inputs();
        let circuit = LayerNormTestCircuit {
            inputs: inputs.clone(),
        };
        let prover = MockProver::run(CIRCUIT_K, &circuit, vec![]).unwrap();
        prover.assert_satisfied();

        // Independently compute the expected outputs using exactly the same
        // fixed-point arithmetic the chip's own sub-chips use internally
        // (mirroring how `reduce.rs`'s and `patch_embed.rs`'s own tests
        // cross-check their chips: MockProver confirms the circuit's
        // internal witnesses are self-consistent; this separately confirms
        // that consistent computation is also numerically close to the true
        // real-valued layer norm).
        let reciprocal = I18::from_f64(1.0 / (K as f64)).unwrap();

        let sum_raw: i64 = inputs.iter().map(I18::raw).sum();
        let (mean, _) = requantize_mul(I18::from_raw(sum_raw), reciprocal).unwrap();

        let diffs: Vec<I18> = inputs
            .iter()
            .map(|x| I18::from_raw(x.raw() - mean.raw()))
            .collect();
        let squares: Vec<I18> = diffs
            .iter()
            .map(|d| requantize_mul(*d, *d).unwrap().0)
            .collect();

        let sq_sum_raw: i64 = squares.iter().map(I18::raw).sum();
        let (variance, _) = requantize_mul(I18::from_raw(sq_sum_raw), reciprocal).unwrap();

        let epsilon = I18::from_f64(EPSILON_MILLI as f64 / 1000.0).unwrap();
        let variance_plus_eps = I18::from_raw(variance.raw() + epsilon.raw());
        // Sanity check this test's carefully chosen inputs actually hit the
        // domain grid point this test relies on (raw 1.26 == index 40).
        assert_eq!(variance_plus_eps.raw(), 1_260_000_000_000_000_000);

        let rsqrt_value = I18::from_f64(rsqrt_f64(variance_plus_eps.to_f64())).unwrap();
        let expected_outputs: Vec<I18> = diffs
            .iter()
            .map(|d| requantize_mul(*d, rsqrt_value).unwrap().0)
            .collect();

        // True (non-fixed-point) layer norm, for the accuracy cross-check.
        let true_inputs: Vec<f64> = inputs.iter().map(I18::to_f64).collect();
        let true_mean = true_inputs.iter().sum::<f64>() / (K as f64);
        let true_var = true_inputs
            .iter()
            .map(|x| (x - true_mean).powi(2))
            .sum::<f64>()
            / (K as f64);
        let true_eps = EPSILON_MILLI as f64 / 1000.0;
        let true_rsqrt = 1.0 / (true_var + true_eps).sqrt();
        let true_outputs: Vec<f64> = true_inputs
            .iter()
            .map(|x| (x - true_mean) * true_rsqrt)
            .collect();

        for (expected, true_val) in expected_outputs.iter().zip(true_outputs.iter()) {
            assert!(
                (expected.to_f64() - true_val).abs() < 1e-6,
                "expected {} vs true {}",
                expected.to_f64(),
                true_val
            );
        }
    }

    #[test]
    fn assign_rejects_wrong_input_count_at_the_rust_level() {
        struct GuardCircuit {
            inputs: Vec<I18>,
        }

        impl Circuit<Fr> for GuardCircuit {
            type Config = LayerNormTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                GuardCircuit {
                    inputs: self.inputs.clone(),
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                LayerNormTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let chip = LayerNormChip::construct(config.layer_norm);
                chip.load_table(layouter.namespace(|| "table"))?;
                match chip.assign(layouter.namespace(|| "assign"), &self.inputs) {
                    Err(LayerNormError::InputCountMismatch { expected, got }) => {
                        assert_eq!(expected, K);
                        assert_eq!(got, K - 1);
                    }
                    Ok(_) => panic!("expected InputCountMismatch error but assign succeeded"),
                    Err(other) => panic!("expected InputCountMismatch error, got {other}"),
                }
                Ok(())
            }
        }

        let mut inputs = sample_inputs();
        inputs.pop();
        let circuit = GuardCircuit { inputs };
        let _ = MockProver::run(CIRCUIT_K, &circuit, vec![]);
    }

    /// Bypasses `LayerNormChip::assign`'s internal `RsqrtChip::assign` call
    /// (well, runs alongside it -- the honest computation still happens
    /// first) and separately witnesses one additional, standalone forged
    /// `(input, output)` pair directly on the shared `rsqrt` lookup's
    /// columns, mirroring `GeluChip`'s own
    /// `forged_output_for_a_valid_gelu_input_is_rejected` test pattern (see
    /// `chips/gelu.rs`). This is the "lower-level forged pattern from one of
    /// the composed sub-chips" this chip's negative test reuses: since
    /// `RsqrtChip`/`LookupConfig` expose the selector/column accessors
    /// needed to witness a row directly, this is the natural choice (the
    /// other composed chips -- `EltwiseAddChip`/`EltwiseMulChip`/
    /// `ReduceMeanChip` -- keep their internal columns/selectors private to
    /// their own modules, so forging them isn't reachable from here without
    /// widening their visibility).
    #[test]
    fn forged_rsqrt_output_within_full_layer_norm_circuit_is_rejected() {
        struct ForgedCircuit {
            inputs: Vec<I18>,
        }

        impl Circuit<Fr> for ForgedCircuit {
            type Config = LayerNormTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                ForgedCircuit {
                    inputs: vec![I18::from_raw(0); self.inputs.len()],
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                LayerNormTestCircuit::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                let rsqrt_config = config.layer_norm.rsqrt.clone();
                let chip = LayerNormChip::construct(config.layer_norm);
                chip.load_table(layouter.namespace(|| "table"))?;
                chip.assign(layouter.namespace(|| "assign"), &self.inputs)
                    .expect("honest layer norm assign should succeed");

                // Additional standalone forged row: same query point this
                // test's inputs resolve to (raw 1.26, domain index 40), but
                // an output one raw unit away from the table's real value.
                let (domain, values) = build_domain(
                    rsqrt_f64,
                    RSQRT_DOMAIN_MIN,
                    RSQRT_DOMAIN_MAX,
                    RSQRT_DOMAIN_N,
                );
                let idx = 40;
                let query_input = domain[idx];
                let correct_output = values[idx];
                let forged_output = I18::from_raw(correct_output.raw() + 1);

                layouter.assign_region(
                    || "forged rsqrt lookup",
                    |mut region| {
                        // Must enable the selector: the lookup is gated (see
                        // `RsqrtConfig::selector`'s doc comment), so this
                        // forged row would otherwise silently collapse to
                        // the always-satisfied padding row instead of
                        // actually checking the forged values below.
                        rsqrt_config.selector().enable(&mut region, 0)?;
                        region.assign_advice(
                            || "input",
                            rsqrt_config.input_column(),
                            0,
                            || Value::known(i64_to_fr(query_input.raw())),
                        )?;
                        region.assign_advice(
                            || "forged output",
                            rsqrt_config.output_column(),
                            0,
                            || Value::known(i64_to_fr(forged_output.raw())),
                        )
                    },
                )?;
                Ok(())
            }
        }

        let circuit = ForgedCircuit {
            inputs: sample_inputs(),
        };
        let prover = MockProver::run(CIRCUIT_K, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }
}
