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
//! ## Soundness: linking cells across composed-chip boundaries
//!
//! [`crate::chips::eltwise::EltwiseAddChip::assign`] and
//! [`crate::chips::eltwise::EltwiseMulChip::assign`] now return an
//! `AssignedCell` for their output (mirroring
//! [`crate::chips::dot_general::DotProductChip::assign`] and
//! [`crate::chips::reduce::ReduceMeanChip::assign`]), and
//! [`crate::chips::reduce::ReduceMeanChip::assign`] additionally returns the
//! per-input `AssignedCell`s it re-witnesses internally. `LayerNormChip`'s
//! own `assign` still independently recomputes each intermediate host-side
//! value (`diff_i`, `sq_i`, `variance_plus_eps`, `output_i`), but critically
//! it **also** ties every one of those recomputed values' cells back to the
//! producing chip's real cell via `region.constrain_equal` -- exactly the
//! same discipline [`crate::chips::softmax::SoftmaxChip`] applies one level
//! down (see that module's docs), applied here one level up across five more
//! composed chip calls. Two wrinkles specific to this chip, not present in
//! `SoftmaxChip`:
//!
//! - `EltwiseAddChip`'s `a`/`b`/`c` columns all hold the *signed-shifted*
//!   representation (`raw + 2^63`), while `EltwiseMulChip`'s `a`/`b` and
//!   `crate::chips::lookup::LookupChip`'s `input`/`output` hold the
//!   *unshifted* natural representation -- so a value produced in shifted
//!   form (e.g. `diff_i`, out of `EltwiseAddChip`) needs a small dedicated
//!   "unshift bridge" gadget (a linear gate `unshifted + 2^63 == shifted`)
//!   before it can be tied into an unshifted consumer. See `assign_unshift`
//!   below.
//! - Computing `diff_i = x_i - mean` needs `-mean` as an operand, but
//!   `mean`'s only produced cell holds the shifted `mean_raw + 2^63`; a
//!   small dedicated "negation bridge" gate (`neg_mean_shift + mean_shift ==
//!   2 * 2^63`) ties a freshly witnessed shifted `-mean` cell back to the
//!   real mean cell. See the `s_neg_mean` gate below.
//!
//! Because `EltwiseAddChip::assign`/`EltwiseMulChip::assign` only expose
//! their *output* cell (not their `a`/`b` input cells), the specific
//! Add/Mul rows that need their *input* cells linked (the diff computation,
//! the variance+epsilon computation, and both squaring/scaling
//! multiplications) are assigned directly against `EltwiseAddConfig`'s and
//! `EltwiseMulConfig`'s columns/selectors (`assign_add_row`/
//! `assign_mul_row` below) rather than through `EltwiseAddChip::assign`/
//! `EltwiseMulChip::assign` as black boxes -- mirroring the precedent
//! `DotProductConfig` already set for this in `dot_general.rs`.
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
use crate::chips::lookup::{
    build_domain, build_domain_from_raw, LookupChip, LookupConfig, LookupError,
};
use crate::chips::range_check::RangeCheckChip;
use crate::chips::reduce::{ReduceMeanChip, ReduceMeanConfig};
use crate::field_convert::{i128_to_fr, i64_to_fr, shifted_i64_witness, Fr, SIGNED_SHIFT};
use crate::fixed_point::{requantize_mul, I18, SCALE_18};
use halo2_proofs::circuit::{AssignedCell, Layouter, Value};
use halo2_proofs::plonk::{Advice, Column, ConstraintSystem, ErrorFront, Expression, Selector};
use halo2_proofs::poly::Rotation;
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

/// How an [`RsqrtChip`]'s lookup domain is specified -- see
/// [`RsqrtChip::construct_with_domain`].
#[derive(Clone)]
pub enum RsqrtDomain {
    /// `n` points evenly quantized over `[min, max]` (`min` strictly
    /// positive) -- the original, `f64`-based construction
    /// [`RsqrtChip::construct`] also uses.
    Range { min: f64, max: f64, n: usize },
    /// Exact raw `I18` anchor points -- see
    /// `crate::chips::lookup::build_domain_from_raw`'s docs on why this
    /// exists (an `f64`-based range cannot always hit an arbitrary,
    /// already-known raw target exactly).
    RawAnchors(Vec<i64>),
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

    /// Builds a chip from an explicit [`RsqrtDomain`] spec -- either the
    /// same evenly-quantized-range construction as [`RsqrtChip::construct`],
    /// or an exact-raw-anchor domain (see [`RsqrtDomain::RawAnchors`] and
    /// `crate::chips::lookup::build_domain_from_raw`'s docs on why the
    /// latter exists).
    pub fn construct_with_domain(config: RsqrtConfig, domain: RsqrtDomain) -> Self {
        let (points, values) = match domain {
            RsqrtDomain::Range { min, max, n } => {
                assert!(min > 0.0, "rsqrt domain minimum must be strictly positive");
                build_domain(rsqrt_f64, min, max, n)
            }
            RsqrtDomain::RawAnchors(raw_points) => build_domain_from_raw(rsqrt_f64, &raw_points),
        };
        let inner = LookupChip::construct(config.lookup, points, values);
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
    // "Negation bridge": ties a freshly witnessed shifted `-mean` cell
    // (`neg_mean`) back to `ReduceMeanChip`'s own returned (shifted) mean
    // cell (copied into `mean_link` for the gate below) via
    // `neg_mean + mean_link == 2 * SIGNED_SHIFT` -- see this module's
    // top-level soundness docs.
    mean_link: Column<Advice>,
    neg_mean: Column<Advice>,
    s_neg_mean: Selector,
    // "Unshift bridge": ties a value produced in `EltwiseAddChip`'s
    // signed-shifted representation (`unshift_in`) to a fresh unshifted
    // copy (`unshift_out`) via `unshift_out + SIGNED_SHIFT == unshift_in`,
    // so it can be linked into an `EltwiseMulChip`/`LookupChip` consumer,
    // whose columns hold the unshifted representation. Reused (like
    // `RangeCheckChip`'s columns) across every value that needs this
    // bridge, each in its own small region.
    unshift_in: Column<Advice>,
    unshift_out: Column<Advice>,
    s_unshift: Selector,
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
        mean_link: Column<Advice>,
        neg_mean: Column<Advice>,
        unshift_in: Column<Advice>,
        unshift_out: Column<Advice>,
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

        meta.enable_equality(mean_link);
        meta.enable_equality(neg_mean);
        meta.enable_equality(unshift_in);
        meta.enable_equality(unshift_out);

        // See this module's top-level soundness docs: neg_mean_shift +
        // mean_shift == 2 * SIGNED_SHIFT, i.e. neg_mean_shift ==
        // 2*SIGNED_SHIFT - mean_shift == SIGNED_SHIFT - mean_raw ==
        // (-mean_raw) + SIGNED_SHIFT, the shifted representation of -mean.
        let s_neg_mean = meta.selector();
        meta.create_gate("layer norm mean negation bridge", |meta| {
            let mean_link_expr = meta.query_advice(mean_link, Rotation::cur());
            let neg_mean_expr = meta.query_advice(neg_mean, Rotation::cur());
            let s_neg_mean = meta.query_selector(s_neg_mean);
            let two_shift = Expression::Constant(i128_to_fr(2 * SIGNED_SHIFT));
            vec![s_neg_mean * (mean_link_expr + neg_mean_expr - two_shift)]
        });

        // unshifted_out == shifted_in - SIGNED_SHIFT, i.e. unshifted_out +
        // SIGNED_SHIFT == shifted_in. Reused across every value produced in
        // `EltwiseAddChip`'s shifted representation that needs bridging into
        // an unshifted consumer (see this module's top-level docs).
        let s_unshift = meta.selector();
        meta.create_gate("layer norm unshift bridge", |meta| {
            let shifted_expr = meta.query_advice(unshift_in, Rotation::cur());
            let unshifted_expr = meta.query_advice(unshift_out, Rotation::cur());
            let s_unshift = meta.query_selector(s_unshift);
            let shift = Expression::Constant(i128_to_fr(SIGNED_SHIFT));
            vec![s_unshift * (unshifted_expr + shift - shifted_expr)]
        });

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
            mean_link,
            neg_mean,
            s_neg_mean,
            unshift_in,
            unshift_out,
            s_unshift,
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
    /// the exact sequence of composed sub-chip calls, and for why every
    /// intermediate value that crosses a chip boundary is both
    /// independently recomputed host-side (matching the sub-chips' own gate
    /// arithmetic exactly) *and* tied back to the producing chip's real
    /// cell via `region.constrain_equal` (using a shift/negation "bridge"
    /// gadget where the two chips' columns hold different representations).
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

        // Step 1: mean = (1/k) * sum_i x_i. `mean_cell` holds the *signed-
        // shifted* representation (mean.raw() + SIGNED_SHIFT) -- the same
        // representation `ReduceMeanChip`'s own rescale gate/range check
        // use -- and `mean_input_cells[i]` are `ReduceSumChip`'s own
        // (unshifted) re-witnessed copies of each `inputs[i]`.
        let (mean, mean_cell, mean_input_cells) =
            mean_chip.assign(layouter.namespace(|| "layer norm mean"), inputs)?;

        // Negation bridge: prove `neg_mean` (shifted) is genuinely `-mean`,
        // tied to the real `mean_cell` above via the `s_neg_mean` gate +
        // constrain_equal, rather than host-recomputing `-mean.raw()` and
        // re-witnessing it disconnected from the real mean.
        let neg_mean = I18::from_raw(-mean.raw());
        let (mean_shift_fr, _) = shifted_i64_witness(mean.raw());
        let (neg_mean_shift_fr, _) = shifted_i64_witness(neg_mean.raw());
        let (mean_link_cell, neg_mean_cell) = layouter.assign_region(
            || "layer norm mean negation bridge",
            |mut region| {
                self.config.s_neg_mean.enable(&mut region, 0)?;
                let mean_link_cell = region.assign_advice(
                    || "mean link",
                    self.config.mean_link,
                    0,
                    || mean_shift_fr,
                )?;
                let neg_mean_cell = region.assign_advice(
                    || "neg mean",
                    self.config.neg_mean,
                    0,
                    || neg_mean_shift_fr,
                )?;
                Ok((mean_link_cell, neg_mean_cell))
            },
        )?;
        layouter.assign_region(
            || "layer norm mean negation bridge link",
            |mut region| region.constrain_equal(mean_link_cell.cell(), mean_cell.cell()),
        )?;

        // Step 2: diff_i = x_i - mean = x_i + neg_mean, for each i.
        // Reimplements `EltwiseAddChip::assign`'s internal region directly
        // (via `assign_add_row`) so this call site can capture the `a`/`b`
        // operand cells and tie `a` back to `x_i`'s own `ReduceSumChip`
        // witness (via an unshift bridge) and `b` back to `neg_mean_cell`
        // above.
        let mut diffs = Vec::with_capacity(k);
        let mut diff_cells = Vec::with_capacity(k);
        for (i, x) in inputs.iter().enumerate() {
            let (a_cell, b_cell, c_cell) = assign_add_row(
                &self.config.add,
                layouter.namespace(|| format!("layer norm diff {i}")),
                x.raw(),
                neg_mean.raw(),
            )?;

            let x_unshifted_cell = assign_unshift(
                &self.config,
                layouter.namespace(|| format!("layer norm diff {i} input unshift")),
                &a_cell,
                x.raw(),
            )?;

            layouter.assign_region(
                || format!("layer norm diff {i} links"),
                |mut region| {
                    region.constrain_equal(x_unshifted_cell.cell(), mean_input_cells[i].cell())?;
                    region.constrain_equal(b_cell.cell(), neg_mean_cell.cell())?;
                    Ok(())
                },
            )?;

            let diff_raw = x
                .raw()
                .checked_add(neg_mean.raw())
                .expect("I18 layer norm diff overflow");
            diffs.push(I18::from_raw(diff_raw));
            diff_cells.push(c_cell);
        }

        // Step 3: sq_i = diff_i^2, for each i. Reimplements
        // `EltwiseMulChip::assign`'s internal region directly (via
        // `assign_mul_row`) so both the `a` and `b` operand cells can be
        // linked back to `diff_i`'s real (shifted) `c_cell` above -- via an
        // unshift bridge, since `EltwiseAddChip`'s `c` column holds the
        // shifted representation while `EltwiseMulChip`'s `a`/`b` hold the
        // unshifted one.
        let mut squares = Vec::with_capacity(k);
        let mut square_cells = Vec::with_capacity(k);
        let mut diff_unshifted_cells = Vec::with_capacity(k);
        for (i, diff) in diffs.iter().enumerate() {
            let diff_unshifted_cell = assign_unshift(
                &self.config,
                layouter.namespace(|| format!("layer norm diff {i} unshift")),
                &diff_cells[i],
                diff.raw(),
            )?;

            let (mul_a_cell, mul_b_cell, sq_cell) = assign_mul_row(
                &self.config.mul,
                layouter.namespace(|| format!("layer norm square {i}")),
                *diff,
                *diff,
            )?;
            layouter.assign_region(
                || format!("layer norm square {i} diff links"),
                |mut region| {
                    region.constrain_equal(mul_a_cell.cell(), diff_unshifted_cell.cell())?;
                    region.constrain_equal(mul_b_cell.cell(), diff_unshifted_cell.cell())?;
                    Ok(())
                },
            )?;

            let (sq, _) = requantize_mul(*diff, *diff).expect("I18 layer norm square overflow");
            squares.push(sq);
            square_cells.push(sq_cell);
            diff_unshifted_cells.push(diff_unshifted_cell);
        }

        // Step 4: variance = (1/k) * sum_i sq_i (same `mean_chip` instance,
        // reused: mean of k values is mean of k values, whether they're the
        // raw inputs or their squared deviations). `sq_input_cells[i]` are
        // `ReduceSumChip`'s own re-witnessed (unshifted) copies of `sq_i`
        // inside this second mean call -- link each back to the real
        // (shifted) `square_cells[i]` via an unshift bridge.
        let (variance, variance_cell, sq_input_cells) =
            mean_chip.assign(layouter.namespace(|| "layer norm variance"), &squares)?;
        for (i, sq) in squares.iter().enumerate() {
            let sq_unshifted_cell = assign_unshift(
                &self.config,
                layouter.namespace(|| format!("layer norm square {i} unshift")),
                &square_cells[i],
                sq.raw(),
            )?;
            layouter.assign_region(
                || format!("layer norm variance input {i} link"),
                |mut region| {
                    region.constrain_equal(sq_unshifted_cell.cell(), sq_input_cells[i].cell())
                },
            )?;
        }

        // Step 5: variance_plus_eps = variance + epsilon. `variance_cell`
        // (the shifted mean cell `ReduceMeanChip` returned above) is tied
        // directly into this Add row's `a` operand -- both hold the shifted
        // representation, so no bridge is needed. `epsilon` is a
        // compile-time constant, so its `b` operand needs no producer-cell
        // link.
        let (vpe_a_cell, _vpe_b_cell, vpe_cell) = assign_add_row(
            &self.config.add,
            layouter.namespace(|| "layer norm variance plus epsilon"),
            variance.raw(),
            self.config.epsilon.raw(),
        )?;
        layouter.assign_region(
            || "layer norm variance plus epsilon link",
            |mut region| region.constrain_equal(vpe_a_cell.cell(), variance_cell.cell()),
        )?;
        let variance_plus_eps_raw = variance
            .raw()
            .checked_add(self.config.epsilon.raw())
            .expect("I18 layer norm variance+epsilon overflow");
        let variance_plus_eps = I18::from_raw(variance_plus_eps_raw);

        // Step 6: rsqrt_value = rsqrt(variance_plus_eps), via the lookup
        // argument. The `rsqrt_chip.assign` call recovers the correct I18
        // value (with a typed error for off-domain inputs); a second,
        // explicitly linked row on the same lookup columns re-checks the
        // pair and ties `variance_plus_eps`'s real cell (via an unshift
        // bridge from `vpe_cell`) into the lookup argument's input.
        let rsqrt_value = self
            .rsqrt_chip
            .assign(layouter.namespace(|| "layer norm rsqrt"), variance_plus_eps)
            .map_err(LayerNormError::Rsqrt)?;

        let vpe_unshifted_cell = assign_unshift(
            &self.config,
            layouter.namespace(|| "layer norm variance plus epsilon unshift"),
            &vpe_cell,
            variance_plus_eps.raw(),
        )?;

        let rsqrt_config = self.config.rsqrt.clone();
        let (rsqrt_input_cell, rsqrt_output_cell) = layouter.assign_region(
            || "layer norm rsqrt link row",
            |mut region| {
                rsqrt_config.selector().enable(&mut region, 0)?;
                let input_cell = region.assign_advice(
                    || "rsqrt input",
                    rsqrt_config.input_column(),
                    0,
                    || Value::known(i64_to_fr(variance_plus_eps.raw())),
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
            || "layer norm rsqrt input link",
            |mut region| region.constrain_equal(rsqrt_input_cell.cell(), vpe_unshifted_cell.cell()),
        )?;

        // Step 7: output_i = diff_i * rsqrt_value, for each i (same
        // Mul-row shape as the squaring step, reused). `diff_i`'s unshifted
        // cell is reused from step 3; `rsqrt_value`'s cell comes straight
        // from the lookup output above (both already unshifted, so no
        // bridge is needed for either operand here).
        let mut outputs = Vec::with_capacity(k);
        for (i, diff) in diffs.iter().enumerate() {
            let (out_a_cell, out_b_cell, _out_cell) = assign_mul_row(
                &self.config.mul,
                layouter.namespace(|| format!("layer norm scale {i}")),
                *diff,
                rsqrt_value,
            )?;
            layouter.assign_region(
                || format!("layer norm scale {i} links"),
                |mut region| {
                    region.constrain_equal(out_a_cell.cell(), diff_unshifted_cells[i].cell())?;
                    region.constrain_equal(out_b_cell.cell(), rsqrt_output_cell.cell())?;
                    Ok(())
                },
            )?;

            let (out, _) =
                requantize_mul(*diff, rsqrt_value).expect("I18 layer norm output overflow");
            outputs.push(out);
        }

        Ok(outputs)
    }
}

/// Reimplements `EltwiseAddChip::assign`'s internal region + range-check +
/// link logic directly against a shared `EltwiseAddConfig`, but -- unlike
/// `EltwiseAddChip::assign` itself, which only exposes its output cell --
/// also returns the `a`/`b` operand cells, so `LayerNormChip::assign` can
/// tie each operand to whichever other chip's cell produced it (see this
/// module's top-level soundness docs). Mirrors `DotProductConfig`'s
/// already-established precedent (in `dot_general.rs`) for composed chips
/// reaching into a shared sub-chip's raw columns/selector.
///
/// `pub(crate)` (not private): `crate::assembler::AssemblerChip` reuses this
/// exact helper for its own `Eltwise { op: Add }` dispatch, rather than
/// duplicating it -- the composition need (an operand-cell-exposing variant
/// of `EltwiseAddChip::assign`) is identical one level up.
#[allow(clippy::type_complexity)]
pub(crate) fn assign_add_row(
    add: &EltwiseAddConfig,
    mut layouter: impl Layouter<Fr>,
    a_raw: i64,
    b_raw: i64,
) -> Result<
    (
        AssignedCell<Fr, Fr>,
        AssignedCell<Fr, Fr>,
        AssignedCell<Fr, Fr>,
    ),
    ErrorFront,
> {
    let c_raw = a_raw
        .checked_add(b_raw)
        .expect("I18 layer norm add overflow");
    let (a_shift_fr, a_shift_raw) = shifted_i64_witness(a_raw);
    let (b_shift_fr, b_shift_raw) = shifted_i64_witness(b_raw);
    let (c_shift_fr, c_shift_raw) = shifted_i64_witness(c_raw);

    let (a_cell, b_cell, c_cell) = layouter.assign_region(
        || "layer norm add row",
        |mut region| {
            add.s_add.enable(&mut region, 0)?;
            let a_cell = region.assign_advice(|| "a", add.a, 0, || a_shift_fr)?;
            let b_cell = region.assign_advice(|| "b", add.b, 0, || b_shift_fr)?;
            let c_cell = region.assign_advice(|| "c", add.c, 0, || c_shift_fr)?;
            Ok((a_cell, b_cell, c_cell))
        },
    )?;

    let range_a_chip = RangeCheckChip::construct(add.range_a.clone());
    let a_range_cell =
        range_a_chip.assign(layouter.namespace(|| "range a"), a_shift_fr, a_shift_raw)?;
    let range_b_chip = RangeCheckChip::construct(add.range_b.clone());
    let b_range_cell =
        range_b_chip.assign(layouter.namespace(|| "range b"), b_shift_fr, b_shift_raw)?;
    let range_c_chip = RangeCheckChip::construct(add.range_c.clone());
    let c_range_cell =
        range_c_chip.assign(layouter.namespace(|| "range c"), c_shift_fr, c_shift_raw)?;

    layouter.assign_region(
        || "layer norm add row range check links",
        |mut region| {
            region.constrain_equal(a_cell.cell(), a_range_cell.cell())?;
            region.constrain_equal(b_cell.cell(), b_range_cell.cell())?;
            region.constrain_equal(c_cell.cell(), c_range_cell.cell())?;
            Ok(())
        },
    )?;

    Ok((a_cell, b_cell, c_cell))
}

/// Reimplements `EltwiseMulChip::assign`'s internal region + range-check +
/// link logic directly against a shared `EltwiseMulConfig`, but -- unlike
/// `EltwiseMulChip::assign` itself, which only exposes its output cell --
/// also returns the `a`/`b` operand cells, so `LayerNormChip::assign` can
/// tie each operand to whichever other chip's cell produced it. See
/// `assign_add_row`'s doc comment for the same reasoning.
///
/// `pub(crate)` for the same reason as `assign_add_row`: reused directly by
/// `crate::assembler::AssemblerChip`.
#[allow(clippy::type_complexity)]
pub(crate) fn assign_mul_row(
    mul: &EltwiseMulConfig,
    mut layouter: impl Layouter<Fr>,
    a_val: I18,
    b_val: I18,
) -> Result<
    (
        AssignedCell<Fr, Fr>,
        AssignedCell<Fr, Fr>,
        AssignedCell<Fr, Fr>,
    ),
    ErrorFront,
> {
    let (q, r) = requantize_mul(a_val, b_val).expect("I18 layer norm mul overflow");
    let slack = SCALE_18 - 1 - r;
    let (q_shift_fr, q_shift_raw) = shifted_i64_witness(q.raw());

    let (a_cell, b_cell, q_cell, r_cell, slack_cell) = layouter.assign_region(
        || "layer norm mul row",
        |mut region| {
            mul.s_mul.enable(&mut region, 0)?;
            mul.s_slack.enable(&mut region, 0)?;
            let a_cell =
                region.assign_advice(|| "a", mul.a, 0, || Value::known(i64_to_fr(a_val.raw())))?;
            let b_cell =
                region.assign_advice(|| "b", mul.b, 0, || Value::known(i64_to_fr(b_val.raw())))?;
            let q_cell = region.assign_advice(|| "q", mul.q, 0, || q_shift_fr)?;
            let r_cell = region.assign_advice(|| "r", mul.r, 0, || Value::known(i128_to_fr(r)))?;
            let slack_cell = region.assign_advice(
                || "slack",
                mul.slack,
                0,
                || Value::known(i128_to_fr(slack)),
            )?;
            Ok((a_cell, b_cell, q_cell, r_cell, slack_cell))
        },
    )?;

    let range_q_chip = RangeCheckChip::construct(mul.range_q.clone());
    let q_range_cell =
        range_q_chip.assign(layouter.namespace(|| "range q"), q_shift_fr, q_shift_raw)?;
    let range_r_chip = RangeCheckChip::construct(mul.range_r.clone());
    let r_range_cell = range_r_chip.assign(
        layouter.namespace(|| "range r"),
        Value::known(i128_to_fr(r)),
        Value::known(r),
    )?;
    let range_r_slack_chip = RangeCheckChip::construct(mul.range_r_slack.clone());
    let slack_range_cell = range_r_slack_chip.assign(
        layouter.namespace(|| "range r slack"),
        Value::known(i128_to_fr(slack)),
        Value::known(slack),
    )?;

    layouter.assign_region(
        || "layer norm mul row range check links",
        |mut region| {
            region.constrain_equal(q_cell.cell(), q_range_cell.cell())?;
            region.constrain_equal(r_cell.cell(), r_range_cell.cell())?;
            region.constrain_equal(slack_cell.cell(), slack_range_cell.cell())?;
            Ok(())
        },
    )?;

    Ok((a_cell, b_cell, q_cell))
}

/// Bridges a value produced in `EltwiseAddChip`'s signed-shifted
/// representation (`shifted_cell`, holding `raw_value + SIGNED_SHIFT`) into
/// a fresh cell holding the unshifted `raw_value` -- tied together by the
/// `s_unshift` gate configured in `LayerNormChip::configure` and a
/// `region.constrain_equal` back to `shifted_cell`. Needed because
/// `EltwiseMulChip`'s `a`/`b` and `LookupChip`'s `input`/`output` columns
/// hold the unshifted representation, unlike `EltwiseAddChip`'s columns
/// (see this module's top-level soundness docs).
fn assign_unshift(
    config: &LayerNormConfig,
    mut layouter: impl Layouter<Fr>,
    shifted_cell: &AssignedCell<Fr, Fr>,
    raw_value: i64,
) -> Result<AssignedCell<Fr, Fr>, ErrorFront> {
    let (shifted_fr, _) = shifted_i64_witness(raw_value);
    let unshifted_fr = Value::known(i64_to_fr(raw_value));

    let (shifted_copy_cell, unshifted_cell) = layouter.assign_region(
        || "layer norm unshift",
        |mut region| {
            config.s_unshift.enable(&mut region, 0)?;
            let shifted_copy_cell =
                region.assign_advice(|| "shifted in", config.unshift_in, 0, || shifted_fr)?;
            let unshifted_cell =
                region.assign_advice(|| "unshifted out", config.unshift_out, 0, || unshifted_fr)?;
            Ok((shifted_copy_cell, unshifted_cell))
        },
    )?;

    layouter.assign_region(
        || "layer norm unshift link",
        |mut region| region.constrain_equal(shifted_copy_cell.cell(), shifted_cell.cell()),
    )?;

    Ok(unshifted_cell)
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
            let mean_link = meta.advice_column();
            let neg_mean = meta.advice_column();
            let unshift_in = meta.advice_column();
            let unshift_out = meta.advice_column();

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
                    mean_link,
                    neg_mean,
                    unshift_in,
                    unshift_out,
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

    /// Regression test proving `region.constrain_equal` genuinely binds
    /// `diff_i`'s real cell to the squaring step's `a`/`b` operand cells
    /// (the link `LayerNormChip::assign`'s step 3 establishes via
    /// `assign_unshift`/`assign_mul_row`) -- mirroring the exact reasoning
    /// of `chips/eltwise.rs`'s own `*_constrain_equal_rejects_mismatched_range_check_witness`
    /// tests: a synthesize that merely *omitted* the constrain_equal call
    /// would prove nothing (MockProver only checks permutation ties that
    /// were actually registered during that circuit's own synthesis), so
    /// this reproduces the real mean -> diff -> square structure honestly up
    /// through `diff_0`'s real (linked) cell, then squares a deliberately
    /// mismatched decoy value (`diff_0 + 1`) instead of the real one --
    /// while STILL calling `constrain_equal` between the (mismatched) `a`/`b`
    /// operand cells and the real `diff_0` cell, exactly as production code
    /// would if it accidentally computed the wrong value to square. The
    /// permutation argument must reject that mismatch.
    #[test]
    fn diff_to_square_constrain_equal_rejects_mismatched_decoy_value() {
        struct MismatchedSquareCircuit {
            inputs: Vec<I18>,
        }

        impl Circuit<Fr> for MismatchedSquareCircuit {
            type Config = LayerNormTestConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                MismatchedSquareCircuit {
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
                // The rsqrt lookup table must still be loaded: its lookup
                // argument applies to every row of the circuit (see
                // `LookupConfig`'s doc comment), even though this test never
                // touches the rsqrt columns.
                let layer_norm = config.layer_norm.clone();
                let chip = LayerNormChip::construct(config.layer_norm);
                chip.load_table(layouter.namespace(|| "table"))?;

                // Honestly compute mean and diff_0 = inputs[0] - mean, with
                // the exact same real links `LayerNormChip::assign` itself
                // establishes for steps 1-2.
                let mean_chip = ReduceMeanChip::construct(layer_norm.reduce_mean.clone());
                let (mean, _mean_cell, _mean_input_cells) =
                    mean_chip.assign(layouter.namespace(|| "mean"), &self.inputs)?;
                let neg_mean = I18::from_raw(-mean.raw());

                let (_a_cell, _b_cell, diff_cell) = assign_add_row(
                    &layer_norm.add,
                    layouter.namespace(|| "diff 0"),
                    self.inputs[0].raw(),
                    neg_mean.raw(),
                )?;
                let diff_raw = self.inputs[0]
                    .raw()
                    .checked_add(neg_mean.raw())
                    .expect("diff overflow");

                let diff_unshifted_cell = assign_unshift(
                    &layer_norm,
                    layouter.namespace(|| "diff 0 unshift"),
                    &diff_cell,
                    diff_raw,
                )?;

                // Mismatch: square a decoy value (diff_0 + 1) instead of the
                // real diff_0, but still link both operand cells via
                // constrain_equal to the real diff_0 cell above.
                let decoy_diff = I18::from_raw(diff_raw + 1);
                let (mul_a_cell, mul_b_cell, _sq_cell) = assign_mul_row(
                    &layer_norm.mul,
                    layouter.namespace(|| "decoy square"),
                    decoy_diff,
                    decoy_diff,
                )?;
                layouter.assign_region(
                    || "decoy square links",
                    |mut region| {
                        region.constrain_equal(mul_a_cell.cell(), diff_unshifted_cell.cell())?;
                        region.constrain_equal(mul_b_cell.cell(), diff_unshifted_cell.cell())?;
                        Ok(())
                    },
                )?;

                Ok(())
            }
        }

        let circuit = MismatchedSquareCircuit {
            inputs: sample_inputs(),
        };
        let prover = MockProver::run(CIRCUIT_K, &circuit, vec![]).unwrap();
        assert!(
            prover.verify().is_err(),
            "constrain_equal must reject a squared decoy diff tied to the real diff cell"
        );
    }
}
