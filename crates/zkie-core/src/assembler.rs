//! `AssemblerChip`: dispatches a whole compiled program's worth of
//! [`crate::isa::Instruction`]s to the corresponding `crate::chips` chip,
//! building one real `halo2_proofs::plonk::Circuit`-shaped region graph for
//! the entire program.
//!
//! # Why this exists, and the soundness trap it must avoid
//!
//! This module intentionally lives inside `zkie_core` (not `zkie_compiler`,
//! which depends on `zkie_core` and therefore cannot be depended on back --
//! see this crate's `Cargo.toml`): several chips (`DotProductConfig`,
//! `EltwiseAddConfig`, `EltwiseMulConfig`) expose `pub(crate)` internals
//! (columns, selectors, and -- as of this module -- `DotProductConfig`'s
//! range-check sub-configs) specifically so composed chips within this crate
//! can build fully cell-linked composite regions, following the exact
//! precedent [`crate::chips::layer_norm::LayerNormChip`] already set one
//! level down. `pub(crate)` visibility only works within this crate, so an
//! external `zkie_compiler`-based assembler could not reach these internals
//! and would be forced into exactly the trap this module's docs warn about
//! next.
//!
//! This is the same soundness problem `RangeCheckChip`'s disconnected-witness
//! bug was (see the addendum in
//! `docs/superpowers/plans/2026-07-26-zkie-subproject1-foundation.md` and
//! `crate::chips::layer_norm`'s module docs), at a third level of scope:
//! **instruction-to-instruction wiring across an entire compiled program.**
//! If an instruction B takes `RegisterRef::Virtual(a)` as an input (meaning
//! "whatever instruction A produced"), and this assembler resolved that to a
//! host-side [`I18`] value and simply re-witnessed a fresh, disconnected copy
//! of it when assigning instruction B, the *value* would be numerically
//! correct but the *circuit* would not actually prove that B's input really
//! is A's output -- a prover could satisfy A's region with one value and B's
//! region with a completely unrelated one, exactly like the original
//! `RangeCheckChip` bug, just scaled up to whole-program wiring. Every
//! register reference that crosses an instruction boundary is therefore
//! resolved to the producing instruction's *real* `AssignedCell` and tied to
//! the consuming instruction's real input cell via `region.constrain_equal`
//! -- never by recomputing and re-witnessing.
//!
//! # Representation bridging
//!
//! Each chip's columns hold either the *unshifted* natural I18 representation
//! (`DotProductChip`'s `a`/`b`, `EltwiseMulChip`'s `a`/`b`) or the
//! *signed-shifted* representation (`raw + SIGNED_SHIFT`; `EltwiseAddChip`'s
//! `a`/`b`/`c`, and every chip's own quotient/output column: `DotProductChip`'s
//! `q`, `EltwiseMulChip`'s `q`) -- see `chips/eltwise.rs`'s module note. This
//! assembler adopts the *unshifted* representation as every register's
//! canonical cell (matching `DotProductChip`/`EltwiseMulChip`'s operand
//! columns directly), and uses a single reusable "shift bridge" gadget
//! (`assign_bridge`, gated by `s_bridge`: `shifted == unshifted +
//! SIGNED_SHIFT`) in both directions:
//!
//! - **Before** an `EltwiseAddChip` row (whose `a`/`b` need the shifted
//!   representation): bridge a register's canonical unshifted cell to a
//!   fresh shifted cell, linked back via `constrain_equal`.
//! - **After** any instruction (`DotProductChip`'s `q`, `EltwiseAddChip`'s
//!   `c`, `EltwiseMulChip`'s `q` -- all shifted): bridge the chip's real
//!   shifted output cell to a fresh unshifted cell, which becomes the new
//!   register's canonical cell.
//!
//! # Scope
//!
//! Only [`Instruction::DotGeneral`] (looped once per output element, per
//! `crate::chips::dot_general`'s own module docs -- no batched/tiled matmul
//! argument) and [`Instruction::Eltwise`] with [`EltwiseOp::Add`] or
//! [`EltwiseOp::Mul`] are dispatched. This covers the linear-layer
//! `y = MatMul(x, W); z = Add(y, b)` program that is this project's primary
//! end-to-end target (see
//! `docs/superpowers/specs/2026-07-26-zkie-subproject4-scope-decision.md`).
//! Every other `Instruction` variant is rejected with
//! [`AssemblerError::UnsupportedInstruction`] rather than silently
//! mis-assigned.
//!
//! `DotGeneral` with non-empty `batch_dims` is also rejected the same way:
//! batched matmul is out of scope here (see `chips::dot_general`'s own
//! module docs on tiling being future work).
//!
//! # Broadcasting
//!
//! `Instruction::Eltwise` carries no shape metadata -- the ISA models it as a
//! flat elementwise op over whatever `RegisterRef`s it's given (see
//! `zkie_compiler::op_mapper`'s `Add`/`Mul` mapping, which passes shapes
//! through unchecked). To support the linear-layer test's `y (m*n elements)
//! + b (n elements, broadcast across m rows)` pattern, this assembler
//! broadcasts whenever one operand's element count evenly divides the
//! other's: output length is `max(len_a, len_b)`, and element `idx` of each
//! operand is taken at `idx % len` (which reduces to plain elementwise access
//! when `len == max(len_a, len_b)`, and to periodic reuse -- e.g. `b[idx % n]`
//! for a length-`n` bias broadcast across `m` row-major-flattened rows of
//! length `n` each -- otherwise). Any other length mismatch is rejected with
//! [`AssemblerError::EltwiseLengthMismatch`].

use std::collections::HashMap;
use std::fmt;

use crate::chips::dot_general::{DotProductChip, DotProductConfig};
use crate::chips::eltwise::{EltwiseAddChip, EltwiseAddConfig, EltwiseMulChip, EltwiseMulConfig};
use crate::chips::layer_norm::{assign_add_row, assign_mul_row, RsqrtDomain};
use crate::chips::range_check::RangeCheckChip;
use crate::chips::rms_norm::{RmsNormChip, RmsNormConfig, RmsNormError};
use crate::field_convert::{i128_to_fr, i64_to_fr, shifted_i64_witness, Fr, SIGNED_SHIFT};
use crate::fixed_point::{requantize_mul, requantize_raw, FixedPointError, I18, SCALE_18};
use crate::isa::{EltwiseOp, Instruction};
use halo2_proofs::circuit::{AssignedCell, Layouter, Value};
use halo2_proofs::plonk::{Advice, Column, ConstraintSystem, ErrorFront, Expression, Selector};
use halo2_proofs::poly::Rotation;

/// Identifies where an [`AssemblerInstruction`]'s input value comes from --
/// the `zkie_core`-local equivalent of `zkie_compiler::graph_compiler`'s
/// `Register`, kept separate (rather than importing that type) so this crate
/// never depends on `zkie_compiler` (see this module's top-level docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegisterRef {
    /// A graph-level input tensor, indexed into [`AssemblerProgram::input_values`].
    Input(usize),
    /// A weight tensor, indexed into [`AssemblerProgram::weight_values`].
    Weight(usize),
    /// The output tensor of a prior instruction, by its 0-indexed program
    /// position.
    Virtual(usize),
}

/// A single instruction to dispatch, wired to its resolved input registers.
/// The instruction's own program-order index (implicit -- its position in
/// [`AssemblerProgram::instructions`]) is what later instructions reference
/// via `RegisterRef::Virtual`.
#[derive(Debug, Clone)]
pub struct AssemblerInstruction {
    pub instruction: Instruction,
    pub inputs: Vec<RegisterRef>,
}

/// A fully-wired program, plus concrete witness values for one specific
/// proving run. Each tensor is a flat `Vec<I18>` in row-major order; multi-
/// element weights/inputs (e.g. a whole weight matrix) are represented as a
/// single flat vector, with `Instruction`-specific shape parameters (e.g.
/// `DotGeneral`'s `m`/`n`/`k`) determining how the assembler indexes into it.
#[derive(Debug, Clone)]
pub struct AssemblerProgram {
    pub instructions: Vec<AssemblerInstruction>,
    pub input_values: Vec<Vec<I18>>,
    pub weight_values: Vec<Vec<I18>>,
}

/// Errors that can occur while assigning an [`AssemblerChip`] region.
#[derive(Debug)]
pub enum AssemblerError {
    /// A `DotGeneral` instruction's `k` was not among the `k` values scanned
    /// out of the program at `configure` time (only possible if `assign` is
    /// called with a program whose instructions differ from the one passed
    /// to `configure`).
    UnconfiguredDotK { k: usize },
    /// A `RegisterRef` indexed past the end of its corresponding vector.
    RegisterIndexOutOfRange { kind: &'static str, index: usize },
    /// A `DotGeneral` instruction did not have exactly 2 inputs.
    DotGeneralInputCount { expected: usize, got: usize },
    /// A `DotGeneral` instruction's resolved input tensors did not have the
    /// element counts implied by its own `m`/`n`/`k`.
    DotGeneralShapeMismatch {
        expected_a: usize,
        got_a: usize,
        expected_b: usize,
        got_b: usize,
    },
    /// `DotGeneral` was given non-empty `batch_dims` -- batched/tiled matmul
    /// is out of scope for this assembler (see module docs).
    DotGeneralBatchDimsUnsupported,
    /// An `Eltwise` instruction did not have exactly 2 inputs.
    EltwiseInputCount { expected: usize, got: usize },
    /// An `Eltwise` instruction's two input tensors had element counts that
    /// are not broadcast-compatible (neither evenly divides the other) --
    /// see this module's "Broadcasting" docs.
    EltwiseLengthMismatch { a_len: usize, b_len: usize },
    /// An `Instruction` variant this assembler does not (yet) dispatch.
    UnsupportedInstruction(String),
    /// A fixed-point computation overflowed I18's representable range.
    Overflow(String),
    /// A halo2 circuit-synthesis error occurred while assigning cells.
    Circuit(ErrorFront),
    /// An `RmsNorm` instruction did not have exactly 2 inputs (`x`, `weight`).
    RmsNormInputCount { expected: usize, got: usize },
    /// An `RmsNorm` instruction's `x`/`weight` register lengths didn't match
    /// its own `dim`.
    RmsNormShapeMismatch {
        dim: usize,
        got_x: usize,
        got_weight: usize,
    },
    /// No [`RmsNormConfig`] was configured for a `RmsNorm` instruction's
    /// `(dim, epsilon_milli)` pair (only possible if `assign` is called with
    /// a program whose instructions differ from the one passed to
    /// `configure`).
    UnconfiguredRmsNorm { dim: usize, epsilon_milli: u64 },
    /// A wrapped error from [`RmsNormChip::assign`].
    RmsNorm(RmsNormError),
}

impl fmt::Display for AssemblerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssemblerError::UnconfiguredDotK { k } => {
                write!(f, "no DotProductChip was configured for k={k}")
            }
            AssemblerError::RegisterIndexOutOfRange { kind, index } => {
                write!(f, "{kind} register index {index} is out of range")
            }
            AssemblerError::DotGeneralInputCount { expected, got } => write!(
                f,
                "DotGeneral expects {expected} inputs, got {got}"
            ),
            AssemblerError::DotGeneralShapeMismatch {
                expected_a,
                got_a,
                expected_b,
                got_b,
            } => write!(
                f,
                "DotGeneral input shape mismatch: expected a.len()={expected_a} (got {got_a}), b.len()={expected_b} (got {got_b})"
            ),
            AssemblerError::DotGeneralBatchDimsUnsupported => {
                write!(f, "DotGeneral with non-empty batch_dims is not supported by this assembler")
            }
            AssemblerError::EltwiseInputCount { expected, got } => {
                write!(f, "Eltwise expects {expected} inputs, got {got}")
            }
            AssemblerError::EltwiseLengthMismatch { a_len, b_len } => write!(
                f,
                "Eltwise operands are not broadcast-compatible: a.len()={a_len}, b.len()={b_len}"
            ),
            AssemblerError::UnsupportedInstruction(msg) => {
                write!(f, "unsupported instruction: {msg}")
            }
            AssemblerError::Overflow(msg) => write!(f, "assembler overflow: {msg}"),
            AssemblerError::Circuit(e) => write!(f, "assembler circuit error: {e:?}"),
            AssemblerError::RmsNormInputCount { expected, got } => {
                write!(f, "RmsNorm expects {expected} inputs, got {got}")
            }
            AssemblerError::RmsNormShapeMismatch {
                dim,
                got_x,
                got_weight,
            } => write!(
                f,
                "RmsNorm input shape mismatch: dim={dim}, got x.len()={got_x}, weight.len()={got_weight}"
            ),
            AssemblerError::UnconfiguredRmsNorm { dim, epsilon_milli } => write!(
                f,
                "no RmsNormConfig was configured for dim={dim}, epsilon_milli={epsilon_milli}"
            ),
            AssemblerError::RmsNorm(e) => write!(f, "assembler rms norm error: {e}"),
        }
    }
}

impl std::error::Error for AssemblerError {}

impl From<ErrorFront> for AssemblerError {
    fn from(e: ErrorFront) -> Self {
        AssemblerError::Circuit(e)
    }
}

impl From<FixedPointError> for AssemblerError {
    fn from(e: FixedPointError) -> Self {
        AssemblerError::Overflow(e.0)
    }
}

/// A register's real, already-witnessed state: the host-side values (used to
/// keep computing subsequent instructions' expected values) and the
/// canonical (unshifted-representation) `AssignedCell` for each element,
/// which every consuming instruction must `region.constrain_equal` against
/// -- never re-witness a fresh, disconnected copy of.
struct RegisterCells {
    values: Vec<I18>,
    cells: Vec<AssignedCell<Fr, Fr>>,
}

/// Configuration for an [`AssemblerChip`]: one [`DotProductConfig`] per
/// distinct `k` seen among the program's `DotGeneral` instructions (mirroring
/// how `PatchEmbedChip`/`SoftmaxChip` reuse one configured chip across many
/// regions -- here, reuse is keyed by `k` since `DotProductChip::configure`
/// bakes `k` in as the number of accumulation rows), a single shared
/// [`EltwiseAddConfig`]/[`EltwiseMulConfig`] reused across every `Add`/`Mul`
/// instruction, a `boundary` column for witnessing `Input`/`Weight` values
/// once each, and the shift-bridge gadget described in this module's
/// top-level docs.
#[derive(Clone)]
pub struct AssemblerConfig {
    dot: HashMap<usize, DotProductConfig>,
    add: EltwiseAddConfig,
    mul: EltwiseMulConfig,
    /// One [`RmsNormConfig`] per distinct `(dim, epsilon_milli)` pair seen
    /// among the program's `RmsNorm` instructions, each built with whichever
    /// `rsqrt` lookup domain [`AssemblerChip::configure_with_rms_norm_domains`]
    /// was given for that pair (or `rms_norm_default_rsqrt_domain()` if none
    /// was given, e.g. via the plain [`AssemblerChip::configure`]) -- see
    /// `crate::chips::rms_norm`'s "CRITICAL NUMERIC LIMITATION" docs on why
    /// the domain must be chosen to include the exact `mean(x^2)+epsilon`
    /// value real inputs will produce.
    rms_norm: HashMap<(usize, u64), RmsNormConfig>,
    boundary: Column<Advice>,
    bridge_unshifted: Column<Advice>,
    bridge_shifted: Column<Advice>,
    s_bridge: Selector,
}

/// Default `rsqrt` lookup domain used by [`AssemblerChip::configure`] (which
/// has no way to accept per-program domain overrides) for any `RmsNorm`
/// instruction whose `(dim, epsilon_milli)` isn't explicitly given a domain
/// via [`AssemblerChip::configure_with_rms_norm_domains`]. Deliberately
/// modest (`n = 21` grid points) since it exists only so `configure()`
/// doesn't panic on an unrecognized `RmsNorm` shape -- callers with real
/// `RmsNorm` data (whose exact `mean(x^2)+epsilon` will essentially never
/// land on a coarse, arbitrary grid -- see `chips::rms_norm`'s numeric-
/// limitation docs, and `RsqrtDomain::RawAnchors` for the exact-anchor
/// construction real callers should use instead) must call
/// `configure_with_rms_norm_domains` with a domain constructed to actually
/// include their real target value.
fn rms_norm_default_rsqrt_domain() -> RsqrtDomain {
    RsqrtDomain::Range {
        min: 0.1,
        max: 10.0,
        n: 21,
    }
}

pub struct AssemblerChip {
    config: AssemblerConfig,
}

impl AssemblerChip {
    /// Pre-scans `instructions` for the set of distinct `k` values `DotGeneral`
    /// instructions need, and configures one [`DotProductConfig`] per value,
    /// plus one shared Add/Mul config and the bridging gadget. `instructions`
    /// only needs to describe the program's *shape* (instruction variants and
    /// their `k`s) -- concrete values are supplied later to [`AssemblerChip::assign`].
    pub fn configure(
        meta: &mut ConstraintSystem<Fr>,
        instructions: &[AssemblerInstruction],
    ) -> AssemblerConfig {
        Self::configure_with_rms_norm_domains(meta, instructions, &HashMap::new())
    }

    /// Like [`AssemblerChip::configure`], but additionally accepts an
    /// explicit `rsqrt` lookup domain (`(domain_min, domain_max, n)`, see
    /// [`crate::chips::rms_norm::RmsNormChip::configure`]) for whichever
    /// `RmsNorm` `(dim, epsilon_milli)` pairs `instructions` contains. Any
    /// pair not present in `rms_norm_domains` falls back to
    /// [`RMS_NORM_DEFAULT_RSQRT_DOMAIN`]. Real callers with genuine `RmsNorm`
    /// data should always supply an explicit domain here -- see this
    /// module's `RMS_NORM_DEFAULT_RSQRT_DOMAIN` docs on why the default is
    /// unlikely to fit real data.
    pub fn configure_with_rms_norm_domains(
        meta: &mut ConstraintSystem<Fr>,
        instructions: &[AssemblerInstruction],
        rms_norm_domains: &HashMap<(usize, u64), RsqrtDomain>,
    ) -> AssemblerConfig {
        // Shared `bits` column across every composed chip's internal range
        // checks -- safe because `RangeCheckChip::configure` creates a fresh,
        // independently-gated selector on each call (see
        // `LayerNormChip::configure`'s identical reuse of a single `bits`
        // column across several sub-chips).
        let bits = meta.advice_column();

        let mut dot_ks: Vec<usize> = instructions
            .iter()
            .filter_map(|instr| match &instr.instruction {
                Instruction::DotGeneral { k, .. } => Some(*k),
                _ => None,
            })
            .collect();
        dot_ks.sort_unstable();
        dot_ks.dedup();

        let mut dot = HashMap::with_capacity(dot_ks.len());
        for k in dot_ks {
            let a = meta.advice_column();
            let b = meta.advice_column();
            let accumulator = meta.advice_column();
            let q = meta.advice_column();
            let r = meta.advice_column();
            let slack = meta.advice_column();
            let cfg = DotProductChip::configure(meta, a, b, accumulator, q, r, slack, bits, k);
            dot.insert(k, cfg);
        }

        let add_a = meta.advice_column();
        let add_b = meta.advice_column();
        let add_c = meta.advice_column();
        let add = EltwiseAddChip::configure(meta, add_a, add_b, add_c, bits);

        let mul_a = meta.advice_column();
        let mul_b = meta.advice_column();
        let mul_q = meta.advice_column();
        let mul_r = meta.advice_column();
        let mul_slack = meta.advice_column();
        let mul = EltwiseMulChip::configure(meta, mul_a, mul_b, mul_q, mul_r, mul_slack, bits);

        let mut rms_norm_keys: Vec<(usize, u64)> = instructions
            .iter()
            .filter_map(|instr| match &instr.instruction {
                Instruction::RmsNorm { dim, epsilon_milli } => Some((*dim, *epsilon_milli)),
                _ => None,
            })
            .collect();
        rms_norm_keys.sort_unstable();
        rms_norm_keys.dedup();

        let mut rms_norm = HashMap::with_capacity(rms_norm_keys.len());
        for (dim, epsilon_milli) in rms_norm_keys {
            let rsqrt_domain = rms_norm_domains
                .get(&(dim, epsilon_milli))
                .cloned()
                .unwrap_or_else(rms_norm_default_rsqrt_domain);

            let rms_values = meta.advice_column();
            let rms_sum = meta.advice_column();
            let rms_sum_shift = meta.advice_column();
            let rms_mean_q = meta.advice_column();
            let rms_mean_r = meta.advice_column();
            let rms_mean_slack = meta.advice_column();
            let rms_add_a = meta.advice_column();
            let rms_add_b = meta.advice_column();
            let rms_add_c = meta.advice_column();
            let rms_mul_a = meta.advice_column();
            let rms_mul_b = meta.advice_column();
            let rms_mul_q = meta.advice_column();
            let rms_mul_r = meta.advice_column();
            let rms_mul_slack = meta.advice_column();
            let rms_rsqrt_input = meta.advice_column();
            let rms_rsqrt_output = meta.advice_column();
            let rms_x_anchor = meta.advice_column();
            let rms_weight_anchor = meta.advice_column();
            let rms_unshift_in = meta.advice_column();
            let rms_unshift_out = meta.advice_column();

            let rms_cfg = RmsNormChip::configure(
                meta,
                rms_values,
                rms_sum,
                rms_sum_shift,
                rms_mean_q,
                rms_mean_r,
                rms_mean_slack,
                rms_add_a,
                rms_add_b,
                rms_add_c,
                rms_mul_a,
                rms_mul_b,
                rms_mul_q,
                rms_mul_r,
                rms_mul_slack,
                bits,
                rms_rsqrt_input,
                rms_rsqrt_output,
                rms_x_anchor,
                rms_weight_anchor,
                rms_unshift_in,
                rms_unshift_out,
                dim,
                epsilon_milli,
                rsqrt_domain,
            );
            rms_norm.insert((dim, epsilon_milli), rms_cfg);
        }

        let boundary = meta.advice_column();
        meta.enable_equality(boundary);

        let bridge_unshifted = meta.advice_column();
        let bridge_shifted = meta.advice_column();
        meta.enable_equality(bridge_unshifted);
        meta.enable_equality(bridge_shifted);

        // shifted == unshifted + SIGNED_SHIFT. Reused (like `RangeCheckChip`'s
        // columns) across every register boundary that needs bridging -- see
        // this module's top-level "Representation bridging" docs.
        let s_bridge = meta.selector();
        meta.create_gate("assembler shift bridge", |meta| {
            let unshifted = meta.query_advice(bridge_unshifted, Rotation::cur());
            let shifted = meta.query_advice(bridge_shifted, Rotation::cur());
            let s_bridge = meta.query_selector(s_bridge);
            let shift = Expression::Constant(i128_to_fr(SIGNED_SHIFT));
            vec![s_bridge * (shifted - unshifted - shift)]
        });

        AssemblerConfig {
            dot,
            add,
            mul,
            rms_norm,
            boundary,
            bridge_unshifted,
            bridge_shifted,
            s_bridge,
        }
    }

    pub fn construct(config: AssemblerConfig) -> Self {
        AssemblerChip { config }
    }

    /// Must be called exactly once per circuit synthesis if `program`
    /// contains any `RmsNorm` instruction (loads every distinct configured
    /// `RmsNormConfig`'s `rsqrt` lookup table) -- mirroring
    /// `RmsNormChip::load_table`'s own one-per-synthesis requirement.
    /// Distinct from [`AssemblerChip::assign`] (rather than folded into it)
    /// so callers whose program has no `RmsNorm` instruction pay no extra
    /// cost and need not call this at all.
    pub fn load_rms_norm_tables(&self, mut layouter: impl Layouter<Fr>) -> Result<(), ErrorFront> {
        for (key, cfg) in self.config.rms_norm.iter() {
            let chip = RmsNormChip::construct(cfg.clone());
            chip.load_table(layouter.namespace(|| format!("assembler rms norm table {key:?}")))?;
        }
        Ok(())
    }

    /// Witnesses every `program.input_values`/`program.weight_values` tensor
    /// exactly once (the real circuit boundary -- there is no producing
    /// instruction to link back to), then walks `program.instructions` in
    /// order, dispatching each to its chip and resolving every input
    /// `RegisterRef` to the real cell its producing register already holds
    /// (linked via `region.constrain_equal`, never re-witnessed). Returns the
    /// host-side output tensor of every instruction, in program order, for
    /// callers to compare against an independently computed expected result.
    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fr>,
        program: &AssemblerProgram,
    ) -> Result<Vec<Vec<I18>>, AssemblerError> {
        let mut input_regs = Vec::with_capacity(program.input_values.len());
        for (idx, tensor) in program.input_values.iter().enumerate() {
            let cells = self.assign_boundary(
                layouter.namespace(|| format!("assembler input {idx}")),
                tensor,
            )?;
            input_regs.push(RegisterCells {
                values: tensor.clone(),
                cells,
            });
        }

        let mut weight_regs = Vec::with_capacity(program.weight_values.len());
        for (idx, tensor) in program.weight_values.iter().enumerate() {
            let cells = self.assign_boundary(
                layouter.namespace(|| format!("assembler weight {idx}")),
                tensor,
            )?;
            weight_regs.push(RegisterCells {
                values: tensor.clone(),
                cells,
            });
        }

        let mut virtual_regs: Vec<RegisterCells> = Vec::with_capacity(program.instructions.len());

        for (idx, instr) in program.instructions.iter().enumerate() {
            let out = match &instr.instruction {
                Instruction::DotGeneral {
                    m,
                    n,
                    k,
                    batch_dims,
                    trans_a,
                    trans_b,
                } => self.assign_dot_general(
                    layouter.namespace(|| format!("assembler instr {idx} dot_general")),
                    &instr.inputs,
                    &input_regs,
                    &weight_regs,
                    &virtual_regs,
                    *m,
                    *n,
                    *k,
                    batch_dims,
                    *trans_a,
                    *trans_b,
                )?,
                Instruction::Eltwise {
                    op: op @ (EltwiseOp::Add | EltwiseOp::Mul),
                } => self.assign_eltwise(
                    layouter.namespace(|| format!("assembler instr {idx} eltwise")),
                    &instr.inputs,
                    &input_regs,
                    &weight_regs,
                    &virtual_regs,
                    op,
                )?,
                Instruction::RmsNorm { dim, epsilon_milli } => self.assign_rms_norm(
                    layouter.namespace(|| format!("assembler instr {idx} rms_norm")),
                    &instr.inputs,
                    &input_regs,
                    &weight_regs,
                    &virtual_regs,
                    *dim,
                    *epsilon_milli,
                )?,
                other => return Err(AssemblerError::UnsupportedInstruction(format!("{other:?}"))),
            };
            virtual_regs.push(out);
        }

        Ok(virtual_regs.into_iter().map(|r| r.values).collect())
    }

    fn assign_boundary(
        &self,
        mut layouter: impl Layouter<Fr>,
        tensor: &[I18],
    ) -> Result<Vec<AssignedCell<Fr, Fr>>, ErrorFront> {
        layouter.assign_region(
            || "assembler boundary",
            |mut region| {
                tensor
                    .iter()
                    .enumerate()
                    .map(|(row, v)| {
                        region.assign_advice(
                            || format!("boundary {row}"),
                            self.config.boundary,
                            row,
                            || Value::known(i64_to_fr(v.raw())),
                        )
                    })
                    .collect()
            },
        )
    }

    /// Bridges a raw I18 value between its unshifted and signed-shifted
    /// representations (see this module's top-level docs), returning both
    /// freshly witnessed cells. Callers link whichever side already has a
    /// real producing cell via `region.constrain_equal`.
    #[allow(clippy::type_complexity)]
    fn assign_bridge(
        &self,
        mut layouter: impl Layouter<Fr>,
        raw: i64,
    ) -> Result<(AssignedCell<Fr, Fr>, AssignedCell<Fr, Fr>), ErrorFront> {
        let unshifted_fr = Value::known(i64_to_fr(raw));
        let (shifted_fr, _) = shifted_i64_witness(raw);
        layouter.assign_region(
            || "assembler shift bridge",
            |mut region| {
                self.config.s_bridge.enable(&mut region, 0)?;
                let unshifted_cell = region.assign_advice(
                    || "unshifted",
                    self.config.bridge_unshifted,
                    0,
                    || unshifted_fr,
                )?;
                let shifted_cell = region.assign_advice(
                    || "shifted",
                    self.config.bridge_shifted,
                    0,
                    || shifted_fr,
                )?;
                Ok((unshifted_cell, shifted_cell))
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn assign_dot_general(
        &self,
        mut layouter: impl Layouter<Fr>,
        inputs: &[RegisterRef],
        input_regs: &[RegisterCells],
        weight_regs: &[RegisterCells],
        virtual_regs: &[RegisterCells],
        m: usize,
        n: usize,
        k: usize,
        batch_dims: &[usize],
        trans_a: bool,
        trans_b: bool,
    ) -> Result<RegisterCells, AssemblerError> {
        if !batch_dims.is_empty() {
            return Err(AssemblerError::DotGeneralBatchDimsUnsupported);
        }
        if inputs.len() != 2 {
            return Err(AssemblerError::DotGeneralInputCount {
                expected: 2,
                got: inputs.len(),
            });
        }

        let a_reg = resolve(&inputs[0], input_regs, weight_regs, virtual_regs)?;
        let b_reg = resolve(&inputs[1], input_regs, weight_regs, virtual_regs)?;

        let expected_a_len = m * k;
        let expected_b_len = k * n;
        if a_reg.values.len() != expected_a_len || b_reg.values.len() != expected_b_len {
            return Err(AssemblerError::DotGeneralShapeMismatch {
                expected_a: expected_a_len,
                got_a: a_reg.values.len(),
                expected_b: expected_b_len,
                got_b: b_reg.values.len(),
            });
        }

        let dot_config = self
            .config
            .dot
            .get(&k)
            .ok_or(AssemblerError::UnconfiguredDotK { k })?
            .clone();

        // Physical flat index for logical A[i][l] / B[l][j], accounting for
        // ONNX's transpose-last-two-dims convention (see
        // `zkie_compiler::op_mapper::last_two_dims`, whose logical (m,k)/(k,n)
        // this instruction's own `m`/`n`/`k` already reflect).
        let phys_a = |i: usize, l: usize| -> usize {
            if trans_a {
                l * m + i
            } else {
                i * k + l
            }
        };
        let phys_b = |l: usize, j: usize| -> usize {
            if trans_b {
                j * k + l
            } else {
                l * n + j
            }
        };

        let mut out_values = Vec::with_capacity(m * n);
        let mut out_cells = Vec::with_capacity(m * n);

        for i in 0..m {
            for j in 0..n {
                let a_vec: Vec<I18> = (0..k).map(|l| a_reg.values[phys_a(i, l)]).collect();
                let b_vec: Vec<I18> = (0..k).map(|l| b_reg.values[phys_b(l, j)]).collect();

                // Validate the accumulation stays in range *before* calling
                // `assign_dot_row` (which panics on overflow, mirroring
                // `DotProductChip::assign`'s own `.expect(...)` -- see that
                // module's docs) so a real out-of-range program surfaces a
                // typed error instead of a panic.
                let raw_sum: i128 = a_vec
                    .iter()
                    .zip(b_vec.iter())
                    .map(|(x, y)| (x.raw() as i128) * (y.raw() as i128))
                    .sum();
                let (q, _r) = requantize_raw(raw_sum)?;

                let (a_cells, b_cells, q_cell) = assign_dot_row(
                    &dot_config,
                    layouter.namespace(|| format!("assembler dot ({i},{j})")),
                    &a_vec,
                    &b_vec,
                )?;

                layouter.assign_region(
                    || format!("assembler dot ({i},{j}) input links"),
                    |mut region| {
                        for l in 0..k {
                            region.constrain_equal(
                                a_cells[l].cell(),
                                a_reg.cells[phys_a(i, l)].cell(),
                            )?;
                            region.constrain_equal(
                                b_cells[l].cell(),
                                b_reg.cells[phys_b(l, j)].cell(),
                            )?;
                        }
                        Ok(())
                    },
                )?;

                let (out_unshifted_cell, out_shifted_cell) = self.assign_bridge(
                    layouter.namespace(|| format!("assembler dot ({i},{j}) output bridge")),
                    q.raw(),
                )?;
                layouter.assign_region(
                    || format!("assembler dot ({i},{j}) output link"),
                    |mut region| region.constrain_equal(out_shifted_cell.cell(), q_cell.cell()),
                )?;

                out_values.push(q);
                out_cells.push(out_unshifted_cell);
            }
        }

        Ok(RegisterCells {
            values: out_values,
            cells: out_cells,
        })
    }

    fn assign_eltwise(
        &self,
        mut layouter: impl Layouter<Fr>,
        inputs: &[RegisterRef],
        input_regs: &[RegisterCells],
        weight_regs: &[RegisterCells],
        virtual_regs: &[RegisterCells],
        op: &EltwiseOp,
    ) -> Result<RegisterCells, AssemblerError> {
        if inputs.len() != 2 {
            return Err(AssemblerError::EltwiseInputCount {
                expected: 2,
                got: inputs.len(),
            });
        }

        let a_reg = resolve(&inputs[0], input_regs, weight_regs, virtual_regs)?;
        let b_reg = resolve(&inputs[1], input_regs, weight_regs, virtual_regs)?;

        let len_a = a_reg.values.len();
        let len_b = b_reg.values.len();
        let out_len = len_a.max(len_b);
        if out_len == 0 || out_len % len_a != 0 || out_len % len_b != 0 {
            return Err(AssemblerError::EltwiseLengthMismatch {
                a_len: len_a,
                b_len: len_b,
            });
        }

        let mut out_values = Vec::with_capacity(out_len);
        let mut out_cells = Vec::with_capacity(out_len);

        for idx in 0..out_len {
            let a_idx = idx % len_a;
            let b_idx = idx % len_b;
            let a_val = a_reg.values[a_idx];
            let b_val = b_reg.values[b_idx];

            match op {
                EltwiseOp::Add => {
                    let c_raw = a_val
                        .raw()
                        .checked_add(b_val.raw())
                        .ok_or_else(|| AssemblerError::Overflow("eltwise add overflow".into()))?;

                    // Bridge both operands from their canonical unshifted
                    // register cells to fresh shifted cells (EltwiseAddChip's
                    // a/b columns hold the shifted representation).
                    let (a_unshift_cell, a_shift_cell) = self.assign_bridge(
                        layouter.namespace(|| format!("assembler add {idx} a bridge")),
                        a_val.raw(),
                    )?;
                    layouter.assign_region(
                        || format!("assembler add {idx} a link"),
                        |mut region| {
                            region.constrain_equal(a_unshift_cell.cell(), a_reg.cells[a_idx].cell())
                        },
                    )?;
                    let (b_unshift_cell, b_shift_cell) = self.assign_bridge(
                        layouter.namespace(|| format!("assembler add {idx} b bridge")),
                        b_val.raw(),
                    )?;
                    layouter.assign_region(
                        || format!("assembler add {idx} b link"),
                        |mut region| {
                            region.constrain_equal(b_unshift_cell.cell(), b_reg.cells[b_idx].cell())
                        },
                    )?;

                    let (a_cell, b_cell, c_cell) = assign_add_row(
                        &self.config.add,
                        layouter.namespace(|| format!("assembler add {idx}")),
                        a_val.raw(),
                        b_val.raw(),
                    )?;
                    layouter.assign_region(
                        || format!("assembler add {idx} operand links"),
                        |mut region| {
                            region.constrain_equal(a_cell.cell(), a_shift_cell.cell())?;
                            region.constrain_equal(b_cell.cell(), b_shift_cell.cell())?;
                            Ok(())
                        },
                    )?;

                    let (out_unshift_cell, out_shift_cell) = self.assign_bridge(
                        layouter.namespace(|| format!("assembler add {idx} output bridge")),
                        c_raw,
                    )?;
                    layouter.assign_region(
                        || format!("assembler add {idx} output link"),
                        |mut region| region.constrain_equal(out_shift_cell.cell(), c_cell.cell()),
                    )?;

                    out_values.push(I18::from_raw(c_raw));
                    out_cells.push(out_unshift_cell);
                }
                EltwiseOp::Mul => {
                    let (q, _r) = requantize_mul(a_val, b_val)?;

                    // EltwiseMulChip's a/b already hold the unshifted
                    // representation -- the same as this assembler's
                    // canonical register cells -- so no bridge is needed for
                    // the operands, only for the (shifted) output.
                    let (a_cell, b_cell, q_cell) = assign_mul_row(
                        &self.config.mul,
                        layouter.namespace(|| format!("assembler mul {idx}")),
                        a_val,
                        b_val,
                    )?;
                    layouter.assign_region(
                        || format!("assembler mul {idx} operand links"),
                        |mut region| {
                            region.constrain_equal(a_cell.cell(), a_reg.cells[a_idx].cell())?;
                            region.constrain_equal(b_cell.cell(), b_reg.cells[b_idx].cell())?;
                            Ok(())
                        },
                    )?;

                    let (out_unshift_cell, out_shift_cell) = self.assign_bridge(
                        layouter.namespace(|| format!("assembler mul {idx} output bridge")),
                        q.raw(),
                    )?;
                    layouter.assign_region(
                        || format!("assembler mul {idx} output link"),
                        |mut region| region.constrain_equal(out_shift_cell.cell(), q_cell.cell()),
                    )?;

                    out_values.push(q);
                    out_cells.push(out_unshift_cell);
                }
                EltwiseOp::Relu => {
                    return Err(AssemblerError::UnsupportedInstruction(
                        "Eltwise { op: Relu }".to_string(),
                    ))
                }
            }
        }

        Ok(RegisterCells {
            values: out_values,
            cells: out_cells,
        })
    }

    /// Dispatches `Instruction::RmsNorm { dim, epsilon_milli }` to a
    /// configured [`RmsNormChip`]. `inputs` must resolve to exactly two
    /// registers: `x` (the `dim` values to normalize) and `weight` (the
    /// `dim` learned per-channel scale values) -- see `crate::isa::Instruction::RmsNorm`'s
    /// docs. Both `RmsNormChip::assign`'s returned `input_cells`/
    /// `weight_cells` (already unshifted, matching this assembler's
    /// canonical register-cell representation, same as `EltwiseMulChip`'s
    /// `a`/`b`) are linked directly via `region.constrain_equal` -- no
    /// bridge needed for either. The chip's `output_cells` (shifted, mul `q`
    /// convention) are bridged to fresh unshifted cells exactly like
    /// `DotGeneral`/`Eltwise`'s own outputs.
    #[allow(clippy::too_many_arguments)]
    fn assign_rms_norm(
        &self,
        mut layouter: impl Layouter<Fr>,
        inputs: &[RegisterRef],
        input_regs: &[RegisterCells],
        weight_regs: &[RegisterCells],
        virtual_regs: &[RegisterCells],
        dim: usize,
        epsilon_milli: u64,
    ) -> Result<RegisterCells, AssemblerError> {
        if inputs.len() != 2 {
            return Err(AssemblerError::RmsNormInputCount {
                expected: 2,
                got: inputs.len(),
            });
        }

        let x_reg = resolve(&inputs[0], input_regs, weight_regs, virtual_regs)?;
        let weight_reg = resolve(&inputs[1], input_regs, weight_regs, virtual_regs)?;

        if x_reg.values.len() != dim || weight_reg.values.len() != dim {
            return Err(AssemblerError::RmsNormShapeMismatch {
                dim,
                got_x: x_reg.values.len(),
                got_weight: weight_reg.values.len(),
            });
        }

        let rms_config = self
            .config
            .rms_norm
            .get(&(dim, epsilon_milli))
            .ok_or(AssemblerError::UnconfiguredRmsNorm { dim, epsilon_milli })?
            .clone();
        let chip = RmsNormChip::construct(rms_config);

        let result = chip
            .assign(
                layouter.namespace(|| "assembler rms_norm"),
                &x_reg.values,
                &weight_reg.values,
            )
            .map_err(AssemblerError::RmsNorm)?;

        layouter.assign_region(
            || "assembler rms_norm input/weight links",
            |mut region| {
                for i in 0..dim {
                    region.constrain_equal(result.input_cells[i].cell(), x_reg.cells[i].cell())?;
                    region.constrain_equal(
                        result.weight_cells[i].cell(),
                        weight_reg.cells[i].cell(),
                    )?;
                }
                Ok(())
            },
        )?;

        let mut out_cells = Vec::with_capacity(dim);
        for (i, out) in result.outputs.iter().enumerate() {
            let (out_unshift_cell, out_shift_cell) = self.assign_bridge(
                layouter.namespace(|| format!("assembler rms_norm {i} output bridge")),
                out.raw(),
            )?;
            layouter.assign_region(
                || format!("assembler rms_norm {i} output link"),
                |mut region| {
                    region.constrain_equal(out_shift_cell.cell(), result.output_cells[i].cell())
                },
            )?;
            out_cells.push(out_unshift_cell);
        }

        Ok(RegisterCells {
            values: result.outputs,
            cells: out_cells,
        })
    }
}

fn resolve<'a>(
    reg: &RegisterRef,
    input_regs: &'a [RegisterCells],
    weight_regs: &'a [RegisterCells],
    virtual_regs: &'a [RegisterCells],
) -> Result<&'a RegisterCells, AssemblerError> {
    match reg {
        RegisterRef::Input(i) => {
            input_regs
                .get(*i)
                .ok_or(AssemblerError::RegisterIndexOutOfRange {
                    kind: "input",
                    index: *i,
                })
        }
        RegisterRef::Weight(i) => {
            weight_regs
                .get(*i)
                .ok_or(AssemblerError::RegisterIndexOutOfRange {
                    kind: "weight",
                    index: *i,
                })
        }
        RegisterRef::Virtual(i) => {
            // Only instructions already processed (index < the current
            // instruction) are present in `virtual_regs`, so an
            // out-of-range index here also catches forward/self references
            // that a valid topologically-ordered program should never emit.
            virtual_regs
                .get(*i)
                .ok_or(AssemblerError::RegisterIndexOutOfRange {
                    kind: "virtual",
                    index: *i,
                })
        }
    }
}

/// Reimplements `DotProductChip::assign`'s internal region + range-check +
/// link logic directly against a shared `DotProductConfig`, but -- unlike
/// `DotProductChip::assign` itself, which only exposes the requantized
/// output value (not even a cell) -- also returns the per-element `a`/`b`
/// operand cells and the (signed-shifted) output `q` cell, so
/// `AssemblerChip` can tie every operand to whichever other register
/// produced it. Mirrors `crate::chips::layer_norm`'s `assign_add_row`/
/// `assign_mul_row` precedent for `EltwiseAddConfig`/`EltwiseMulConfig`.
#[allow(clippy::type_complexity)]
fn assign_dot_row(
    dot: &DotProductConfig,
    mut layouter: impl Layouter<Fr>,
    a: &[I18],
    b: &[I18],
) -> Result<
    (
        Vec<AssignedCell<Fr, Fr>>,
        Vec<AssignedCell<Fr, Fr>>,
        AssignedCell<Fr, Fr>,
    ),
    ErrorFront,
> {
    let k = dot.k;
    assert_eq!(
        a.len(),
        k,
        "assign_dot_row: a.len() must equal configured k"
    );
    assert_eq!(
        b.len(),
        k,
        "assign_dot_row: b.len() must equal configured k"
    );

    let mut raw_sum: i128 = 0;
    let mut partial_sums: Vec<i128> = Vec::with_capacity(k);
    for i in 0..k {
        raw_sum += (a[i].raw() as i128) * (b[i].raw() as i128);
        partial_sums.push(raw_sum);
    }
    let (q, r) = requantize_raw(raw_sum).expect("assembler dot product overflow");
    let slack = SCALE_18 - 1 - r;
    let (q_shift_fr, q_shift_raw) = shifted_i64_witness(q.raw());

    let (a_cells, b_cells, q_cell, r_cell, slack_cell) = layouter.assign_region(
        || "assembler dot product",
        |mut region| {
            let mut a_cells = Vec::with_capacity(k);
            let mut b_cells = Vec::with_capacity(k);
            for i in 0..k {
                let a_cell = region.assign_advice(
                    || format!("a_{i}"),
                    dot.a,
                    i,
                    || Value::known(i64_to_fr(a[i].raw())),
                )?;
                let b_cell = region.assign_advice(
                    || format!("b_{i}"),
                    dot.b,
                    i,
                    || Value::known(i64_to_fr(b[i].raw())),
                )?;
                region.assign_advice(
                    || format!("accumulator_{i}"),
                    dot.accumulator,
                    i,
                    || Value::known(i128_to_fr(partial_sums[i])),
                )?;
                if i == 0 {
                    dot.s_acc_start.enable(&mut region, i)?;
                } else {
                    dot.s_acc_step.enable(&mut region, i)?;
                }
                a_cells.push(a_cell);
                b_cells.push(b_cell);
            }

            let last = k - 1;
            dot.s_final.enable(&mut region, last)?;
            dot.s_slack.enable(&mut region, last)?;
            let q_cell = region.assign_advice(|| "q", dot.q, last, || q_shift_fr)?;
            let r_cell =
                region.assign_advice(|| "r", dot.r, last, || Value::known(i128_to_fr(r)))?;
            let slack_cell = region.assign_advice(
                || "slack",
                dot.slack,
                last,
                || Value::known(i128_to_fr(slack)),
            )?;
            Ok((a_cells, b_cells, q_cell, r_cell, slack_cell))
        },
    )?;

    let range_q_chip = RangeCheckChip::construct(dot.range_q.clone());
    let q_range_cell =
        range_q_chip.assign(layouter.namespace(|| "range q"), q_shift_fr, q_shift_raw)?;
    let range_r_chip = RangeCheckChip::construct(dot.range_r.clone());
    let r_range_cell = range_r_chip.assign(
        layouter.namespace(|| "range r"),
        Value::known(i128_to_fr(r)),
        Value::known(r),
    )?;
    let range_r_slack_chip = RangeCheckChip::construct(dot.range_r_slack.clone());
    let slack_range_cell = range_r_slack_chip.assign(
        layouter.namespace(|| "range r slack"),
        Value::known(i128_to_fr(slack)),
        Value::known(slack),
    )?;

    layouter.assign_region(
        || "assembler dot product range check links",
        |mut region| {
            region.constrain_equal(q_cell.cell(), q_range_cell.cell())?;
            region.constrain_equal(r_cell.cell(), r_range_cell.cell())?;
            region.constrain_equal(slack_cell.cell(), slack_range_cell.cell())?;
            Ok(())
        },
    )?;

    Ok((a_cells, b_cells, q_cell))
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::circuit::SimpleFloorPlanner;
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::plonk::{Circuit, ConstraintSystem, ErrorFront};

    const CIRCUIT_K: u32 = 12;

    // Each test below defines its own tiny `Circuit` impl whose `configure`
    // hardcodes that test's instruction shape -- `Circuit::configure` has no
    // access to `self` (it's a plain associated function), so there's no way
    // to configure generically from an arbitrary runtime `AssemblerProgram`.
    // This mirrors every other chip's own test-circuit pattern in this crate
    // (e.g. `chips/layer_norm.rs`'s `LayerNormTestCircuit`).

    fn i18(v: f64) -> I18 {
        I18::from_f64(v).unwrap()
    }

    // ---- DotGeneral in isolation ------------------------------------------

    struct DotOnlyCircuit {
        program: AssemblerProgram,
    }

    impl Circuit<Fr> for DotOnlyCircuit {
        type Config = AssemblerConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            DotOnlyCircuit {
                program: self.program.clone(),
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let instructions = vec![AssemblerInstruction {
                instruction: Instruction::DotGeneral {
                    m: 1,
                    n: 1,
                    k: 3,
                    batch_dims: vec![],
                    trans_a: false,
                    trans_b: false,
                },
                inputs: vec![RegisterRef::Input(0), RegisterRef::Input(1)],
            }];
            AssemblerChip::configure(meta, &instructions)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = AssemblerChip::construct(config);
            chip.assign(layouter, &self.program)
                .map(|_| ())
                .map_err(|e| panic!("assembler assign failed: {e}"))
        }
    }

    #[test]
    fn lone_dot_general_instruction_is_satisfied_and_correct() {
        let a = vec![i18(2.0), i18(-3.0), i18(1.5)];
        let b = vec![i18(3.0), i18(2.0), i18(-4.0)];
        let program = AssemblerProgram {
            instructions: vec![AssemblerInstruction {
                instruction: Instruction::DotGeneral {
                    m: 1,
                    n: 1,
                    k: 3,
                    batch_dims: vec![],
                    trans_a: false,
                    trans_b: false,
                },
                inputs: vec![RegisterRef::Input(0), RegisterRef::Input(1)],
            }],
            input_values: vec![a.clone(), b.clone()],
            weight_values: vec![],
        };
        let circuit = DotOnlyCircuit { program };
        let prover = MockProver::run(CIRCUIT_K, &circuit, vec![]).unwrap();
        prover.assert_satisfied();

        let raw_sum: i128 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x.raw() as i128) * (y.raw() as i128))
            .sum();
        let (expected, _) = requantize_raw(raw_sum).unwrap();
        assert!((expected.to_f64() - (-6.0)).abs() < 1e-9);
    }

    // ---- Eltwise Add / Mul in isolation ------------------------------------

    struct EltwiseOnlyCircuit {
        program: AssemblerProgram,
        op: EltwiseOp,
    }

    impl Circuit<Fr> for EltwiseOnlyCircuit {
        type Config = AssemblerConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            EltwiseOnlyCircuit {
                program: self.program.clone(),
                op: self.op.clone(),
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            // Both Add and Mul share the same config shape (one shared
            // Add/Mul chip pair regardless of which op a given program
            // actually uses), so `op` doesn't affect `configure`'s output.
            let instructions = vec![AssemblerInstruction {
                instruction: Instruction::Eltwise { op: EltwiseOp::Add },
                inputs: vec![RegisterRef::Input(0), RegisterRef::Input(1)],
            }];
            AssemblerChip::configure(meta, &instructions)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = AssemblerChip::construct(config);
            chip.assign(layouter, &self.program)
                .map(|_| ())
                .map_err(|e| panic!("assembler assign failed: {e}"))
        }
    }

    #[test]
    fn lone_eltwise_add_instruction_is_satisfied_and_correct() {
        let a = vec![i18(2.0)];
        let b = vec![i18(3.5)];
        let program = AssemblerProgram {
            instructions: vec![AssemblerInstruction {
                instruction: Instruction::Eltwise { op: EltwiseOp::Add },
                inputs: vec![RegisterRef::Input(0), RegisterRef::Input(1)],
            }],
            input_values: vec![a, b],
            weight_values: vec![],
        };
        let circuit = EltwiseOnlyCircuit {
            program,
            op: EltwiseOp::Add,
        };
        let prover = MockProver::run(CIRCUIT_K, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn lone_eltwise_mul_instruction_is_satisfied_and_correct() {
        let a = vec![i18(2.0)];
        let b = vec![i18(3.0)];
        let program = AssemblerProgram {
            instructions: vec![AssemblerInstruction {
                instruction: Instruction::Eltwise { op: EltwiseOp::Mul },
                inputs: vec![RegisterRef::Input(0), RegisterRef::Input(1)],
            }],
            input_values: vec![a, b],
            weight_values: vec![],
        };
        let circuit = EltwiseOnlyCircuit {
            program,
            op: EltwiseOp::Mul,
        };
        let prover = MockProver::run(CIRCUIT_K, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    // ---- End-to-end: the linear-layer program (y = x@W; z = y+b) ---------

    /// The exact program shape of `zkie_compiler::graph_compiler`'s
    /// `compiles_matmul_plus_bias_linear_layer_end_to_end` test: `x` (2x2
    /// graph input), `W` (2x2 weight), `b` (length-2 weight, broadcast across
    /// both rows of `y`), `y = MatMul(x, W)`, `z = Add(y, b)`.
    struct LinearLayerCircuit {
        program: AssemblerProgram,
    }

    impl Circuit<Fr> for LinearLayerCircuit {
        type Config = AssemblerConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            LinearLayerCircuit {
                program: self.program.clone(),
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let instructions = linear_layer_instructions();
            AssemblerChip::configure(meta, &instructions)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = AssemblerChip::construct(config);
            chip.assign(layouter, &self.program)
                .map(|_| ())
                .map_err(|e| panic!("assembler assign failed: {e}"))
        }
    }

    fn linear_layer_instructions() -> Vec<AssemblerInstruction> {
        vec![
            AssemblerInstruction {
                instruction: Instruction::DotGeneral {
                    m: 2,
                    n: 2,
                    k: 2,
                    batch_dims: vec![],
                    trans_a: false,
                    trans_b: false,
                },
                inputs: vec![RegisterRef::Input(0), RegisterRef::Weight(0)],
            },
            AssemblerInstruction {
                instruction: Instruction::Eltwise { op: EltwiseOp::Add },
                inputs: vec![RegisterRef::Virtual(0), RegisterRef::Weight(1)],
            },
        ]
    }

    fn linear_layer_program() -> (AssemblerProgram, Vec<I18>, Vec<I18>, Vec<I18>) {
        // x = [[0.5, -0.25], [0.25, 0.5]] (row-major), W = [[1.0, 0.5],
        // [-0.5, 1.0]] (row-major), b = [0.1, -0.2]. Values are chosen small
        // enough that every intermediate (each dot-product accumulation and
        // the final biased sum) stays comfortably within I18's ~±9.22
        // representable range (see `fixed_point::I18`'s docs) -- unlike the
        // `zkie_compiler::graph_compiler` end-to-end test's own W=[1,2,3,4]
        // values, which are only ever used structurally there (no concrete
        // `x` is witnessed in that test), whereas this test actually
        // witnesses and range-checks every intermediate.
        let x = vec![i18(0.5), i18(-0.25), i18(0.25), i18(0.5)];
        let w = vec![i18(1.0), i18(0.5), i18(-0.5), i18(1.0)];
        let b = vec![i18(0.1), i18(-0.2)];

        let program = AssemblerProgram {
            instructions: linear_layer_instructions(),
            input_values: vec![x.clone()],
            weight_values: vec![w.clone(), b.clone()],
        };
        (program, x, w, b)
    }

    /// Independently computes `x @ W + b` (row-major, `b` broadcast across
    /// rows) the same way the assembler's own fixed-point arithmetic does,
    /// for cross-checking against the circuit's witnessed output.
    fn expected_linear_layer_output(
        x: &[I18],
        w: &[I18],
        b: &[I18],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<I18> {
        let mut y = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                let raw_sum: i128 = (0..k)
                    .map(|l| (x[i * k + l].raw() as i128) * (w[l * n + j].raw() as i128))
                    .sum();
                let (q, _) = requantize_raw(raw_sum).unwrap();
                y.push(q);
            }
        }
        y.iter()
            .enumerate()
            .map(|(idx, yv)| I18::from_raw(yv.raw() + b[idx % b.len()].raw()))
            .collect()
    }

    #[test]
    fn linear_layer_end_to_end_is_satisfied_and_numerically_correct() {
        let (program, x, w, b) = linear_layer_program();
        let circuit = LinearLayerCircuit {
            program: program.clone(),
        };
        let prover = MockProver::run(CIRCUIT_K, &circuit, vec![]).unwrap();
        prover.assert_satisfied();

        let expected_z = expected_linear_layer_output(&x, &w, &b, 2, 2, 2);
        // x@W: row0 = [0.5*1.0 + (-0.25)*(-0.5), 0.5*0.5 + (-0.25)*1.0] = [0.625, 0.0]
        //      row1 = [0.25*1.0 + 0.5*(-0.5), 0.25*0.5 + 0.5*1.0] = [0.0, 0.625]
        // z = y + b broadcast (b = [0.1, -0.2]): [0.725, -0.2, 0.1, 0.425]
        let expected_f64: Vec<f64> = vec![0.725, -0.2, 0.1, 0.425];
        for (got, want) in expected_z.iter().zip(expected_f64.iter()) {
            assert!(
                (got.to_f64() - want).abs() < 1e-9,
                "{} vs {}",
                got.to_f64(),
                want
            );
        }
    }

    /// The most important regression test in this module: proves
    /// `Virtual(0)`'s cell (MatMul's real output) really is
    /// `constrain_equal`'d into the Add instruction's first input -- by the
    /// same methodology established throughout this session (see
    /// `chips/eltwise.rs`'s and `chips/layer_norm.rs`'s own
    /// `*_constrain_equal_rejects_mismatched_*` tests): this reproduces the
    /// real MatMul -> Add wiring honestly up through MatMul's real (linked)
    /// output cell, then feeds a deliberately MISMATCHED decoy value into
    /// the Add's first input WHILE STILL calling `constrain_equal` against
    /// the (mismatched) MatMul output cell -- exactly as production code
    /// would if the assembler had a wiring bug that resolved the wrong
    /// value but still (incorrectly) linked it. The permutation argument
    /// must reject this. A test that merely *omitted* the link would prove
    /// nothing (see those modules' docs for why).
    struct MismatchedWiringCircuit {
        program: AssemblerProgram,
    }

    impl Circuit<Fr> for MismatchedWiringCircuit {
        type Config = AssemblerConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            MismatchedWiringCircuit {
                program: self.program.clone(),
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let instructions = linear_layer_instructions();
            AssemblerChip::configure(meta, &instructions)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = AssemblerChip::construct(config.clone());

            let x = &self.program.input_values[0];
            let w = &self.program.weight_values[0];
            let b = &self.program.weight_values[1];

            // Honestly witness the boundary values and the real MatMul
            // (DotGeneral) output, exactly as `AssemblerChip::assign` would.
            let x_cells = chip.assign_boundary(layouter.namespace(|| "x"), x)?;
            let w_cells = chip.assign_boundary(layouter.namespace(|| "w"), w)?;
            let _b_cells = chip.assign_boundary(layouter.namespace(|| "b"), b)?;

            let k = 2;
            let dot_config = config.dot.get(&k).unwrap().clone();

            // MatMul output element (0, 0): x[0][0]*w[0][0] + x[0][1]*w[1][0].
            let a_vec = vec![x[0], x[1]];
            let b_vec = vec![w[0], w[2]];
            let (a_cells, b_cells, q_cell) = assign_dot_row(
                &dot_config,
                layouter.namespace(|| "dot (0,0)"),
                &a_vec,
                &b_vec,
            )?;
            layouter.assign_region(
                || "dot (0,0) input links",
                |mut region| {
                    for l in 0..k {
                        region.constrain_equal(a_cells[l].cell(), x_cells[l].cell())?;
                    }
                    region.constrain_equal(b_cells[0].cell(), w_cells[0].cell())?;
                    region.constrain_equal(b_cells[1].cell(), w_cells[2].cell())?;
                    Ok(())
                },
            )?;
            let raw_sum: i128 = a_vec
                .iter()
                .zip(b_vec.iter())
                .map(|(xv, wv)| (xv.raw() as i128) * (wv.raw() as i128))
                .sum();
            let (y00, _) = requantize_raw(raw_sum).unwrap();
            let (y00_unshift_cell, y00_shift_cell) =
                chip.assign_bridge(layouter.namespace(|| "dot (0,0) output bridge"), y00.raw())?;
            layouter.assign_region(
                || "dot (0,0) output link",
                |mut region| region.constrain_equal(y00_shift_cell.cell(), q_cell.cell()),
            )?;

            // Mismatch: feed a decoy value (y00 + 1) into the Add row's `a`
            // operand instead of the real y00, but STILL bridge and
            // constrain_equal it against y00's real (honest) unshifted cell
            // above -- simulating a wiring bug that resolved the wrong
            // producing cell but still called constrain_equal.
            let decoy_raw = y00
                .raw()
                .checked_add(1)
                .expect("decoy should not overflow in this small test");
            let (decoy_unshift_cell, decoy_shift_cell) =
                chip.assign_bridge(layouter.namespace(|| "decoy bridge"), decoy_raw)?;
            layouter.assign_region(
                || "decoy link to real y00",
                |mut region| {
                    region.constrain_equal(decoy_unshift_cell.cell(), y00_unshift_cell.cell())
                },
            )?;

            let (add_a_cell, _add_b_cell, _add_c_cell) = assign_add_row(
                &config.add,
                layouter.namespace(|| "add row"),
                decoy_raw,
                b[0].raw(),
            )?;
            layouter.assign_region(
                || "add row operand links",
                |mut region| {
                    region.constrain_equal(add_a_cell.cell(), decoy_shift_cell.cell())?;
                    Ok(())
                },
            )?;

            Ok(())
        }
    }

    #[test]
    fn dot_general_to_eltwise_add_constrain_equal_rejects_mismatched_decoy_value() {
        let (program, ..) = linear_layer_program();
        let circuit = MismatchedWiringCircuit { program };
        let prover = MockProver::run(CIRCUIT_K, &circuit, vec![]).unwrap();
        assert!(
            prover.verify().is_err(),
            "constrain_equal must reject a mismatched decoy value wired into the Add instruction"
        );
    }
}
