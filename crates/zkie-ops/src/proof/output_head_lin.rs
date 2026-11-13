//! Output-head MatMul wiring for the merged three-stage LIN request.
//!
//! The output head computes, for each position `t`,
//!
//! ```text
//! Y[t][j] = sum_y X[t][y] * M[y][j]
//! ```
//!
//! over the padded hidden dimension and the full padded vocabulary. The
//! merged session folds the committed encoded table, samples the reduced
//! weight rows and runs both proximity tests. The verifier never receives an
//! activation row, a table message row or a reduced weight row in the clear.
//!
//! The module has two paths.
//!
//! The formal path in the parent module keeps `X` and `Y` out of the verifier
//! boundary. `X` is an independent trusted input whose hiding commitment is
//! pinned by the operator policy; `Y` is the prover's reduced row `W = X M`,
//! committed under the hiding product-form encoding. The trusted output
//! relation (vocabulary mask, valid-vocabulary range, ordering comparison,
//! first-maximum tie rule and expected output) is bound into the first
//! statement and checked over the full padded rows. The verifier receives the
//! policy-pinned commitment, the trusted relation, the committed table reader
//! and the proof message only.
//!
//! The [`diagnostic`] module below keeps the earlier development entries that
//! read plaintext `X` and `Y` from a verified shared trace. They are a
//! controlled diagnostic and real-fixture harness, not the hidden production
//! path; the formal path does not call them.
//!
//! Neither path claims a formal soundness bound, production CLI wiring or a
//! full-model proof. The formal path proves the Dot identity
//! `<X_t, Omega> = W_t(psi*)` over the committed shown-map message and the
//! committed reduced row, through the masked product-sumcheck transcript and
//! the own-group BAT/FOLD/BASE test. The equality of the shown-map commitment
//! to the operator-pinned activation commitment is a trusted-input warranty:
//! the verifier pins both commitments, and no cross-commitment proof is
//! claimed.

use zkie_core::common::codec::DecodeLimits;
use zkie_core::common::field::PrimeField64;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::binding::BindingCandidate;
use zkie_engine::pcs::request::field::{Fp3, MODULUS};
use zkie_engine::pcs::request::foldeq::reader_table_identity;
use zkie_engine::pcs::request::lin::{LinMask, LinShape};
use zkie_engine::pcs::request::table::IndexReader;
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};
use zkie_engine::primitives::transcript::{hash_labeled, tag_word, STATEMENT};

use crate::op::OpCode;
use crate::proof::column_session::fp_from_signed_word;
use crate::proof::hidden_send_session::{
    prove_hidden_lin, prove_hidden_lin_hidden, verify_hidden_lin, verify_hidden_lin_hidden,
    ActivationCommitment, ActivationShape, CommittedActivation, HiddenLinMessages, HiddenLinOutcome,
    HiddenLinPlan, HiddenLinWitness, HiddenOutputShape, HiddenProximityPlan, HiddenSendPlan,
    HiddenTableProver, IndexMaskPlan, WShape,
};
use crate::proof::output_closure::{ClosureMaskPlan, ClosurePlan};
use crate::proof::model_commitment::{
    verify_shared_trace_outputs, ModelCommitmentPublic, SharedTrace,
};
use crate::proof::plan::{
    PlanConfigValue, PlanConstant, PlanLayout, PlanSource, PlanStep, PreviousObjectPlan, TracePlan,
};
use crate::proof::session_baseline::{assemble_binding, SelectedMatmulStep};


/// Canonical encoding version of the merged request proof. Version 5 carries
/// the proximity group's end as a plan-selected union: the single-stage BASE
/// artifact or the multi-stage loop proof. Older versions are rejected
/// explicitly instead of being read under the new layout.
const OUTPUT_HEAD_PROOF_VERSION: u64 = 5;

/// Positions of one output-head MatMul request.
pub const OUTPUT_HEAD_POSITIONS: usize = 16;
/// Padded hidden dimension, one table/matrix row per entry.
pub const OUTPUT_HEAD_HIDDEN: usize = 1024;
/// Full padded vocabulary, one matrix column per entry.
pub const OUTPUT_HEAD_COLUMNS: usize = 65536;
/// Low column bits placed inside one table row.
pub const OUTPUT_HEAD_K_LO: u32 = 6;
/// The public integer bound `D` of one reduced output cell: `|Y_j| < 2^D`.
pub const OUTPUT_HEAD_RANGE_BITS: u32 = 40;
/// The public slack bound exponent `b`: `0 <= S_j < 2^b`.
pub const OUTPUT_HEAD_SLACK_BITS: u32 = 42;
/// Table rows; `2^(k_col - k_lo)` with `k_col = 16`.
pub const OUTPUT_HEAD_TABLE_ROWS: u64 = 1024;

/// The output-head weight asset named by the trusted plan.
pub const OUTPUT_HEAD_ASSET_PATH: &str = "gpt2/weights/lm_head_w_i32.bin";

/// The real output-head shape: 1024 hidden rows and 65536 output columns.
/// The table has 1024 rows and 65536 entries per row, and every one of its
/// entries is read by the matrix mapping.
pub fn output_head_shape() -> Result<LinShape> {
    LinShape::new(
        OUTPUT_HEAD_HIDDEN as u64,
        OUTPUT_HEAD_K_LO,
        OUTPUT_HEAD_TABLE_ROWS,
        OUTPUT_HEAD_COLUMNS as u64,
    )
}

/// The trusted plan items of one output-head step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputHeadStep {
    /// The MatMul step index.
    pub step: usize,
    /// The step whose output supplies the activation, and its output port.
    pub activation_step: usize,
    pub activation_output: usize,
    /// The separate masking `Add` step fed by the MatMul output.
    pub masked_step: usize,
    pub positions: usize,
    pub hidden: usize,
    pub columns: usize,
    /// The weight asset named by the MatMul input port.
    pub asset_path: String,
    /// The plan-declared previous object consumed by the final norm.
    pub previous: PreviousObjectPlan,
}

fn literal_config(step: &PlanStep) -> Option<Vec<i64>> {
    step.config
        .iter()
        .map(|value| match value {
            PlanConfigValue::Literal(value) => Some(*value),
            PlanConfigValue::AssetI64 { .. } => None,
        })
        .collect()
}

fn flat_previous(source: &PlanSource) -> Option<(usize, usize)> {
    match source {
        PlanSource::Prev {
            step,
            output,
            layout: PlanLayout::Flat,
        } => Some((*step, *output)),
        _ => None,
    }
}

fn flat_asset(source: &PlanSource) -> Option<&str> {
    match source {
        PlanSource::Asset {
            path,
            layout: PlanLayout::Flat,
        } => Some(path.as_str()),
        _ => None,
    }
}

fn select_output_head_step_for(
    plan: &TracePlan,
    positions: usize,
    hidden: usize,
    columns: usize,
    asset_path: &str,
) -> Result<OutputHeadStep> {
    if positions == 0 || hidden == 0 || columns == 0 {
        return Err(Error::InvalidConfig);
    }
    let mut candidates = Vec::new();
    for (index, step) in plan.steps.iter().enumerate() {
        if step.opcode != OpCode::MatMul || step.inputs.len() != 2 || step.outputs != 1 {
            continue;
        }
        let Some(config) = literal_config(step) else {
            continue;
        };
        if config.len() != 5
            || config[0] != positions as i64
            || config[1] != columns as i64
            || config[2] != hidden as i64
            || config[3] != 0
            || config[4] != 0
        {
            continue;
        }
        let Some((activation_step, activation_output)) = flat_previous(&step.inputs[0]) else {
            continue;
        };
        if flat_asset(&step.inputs[1]) != Some(asset_path) {
            continue;
        }
        candidates.push((index, activation_step, activation_output));
    }
    let (step, activation_step, activation_output) = match candidates.as_slice() {
        [candidate] => *candidate,
        [] => return Err(Error::StatementMismatch),
        _ => return Err(Error::InvalidConfig),
    };

    // The vocabulary mask must stay a separate plan item over the full column
    // domain; columns at or above the valid vocabulary still take part in the
    // MatMul that this request checks.
    let mut masked = None;
    for (index, plan_step) in plan.steps.iter().enumerate() {
        if plan_step.opcode != OpCode::Add || plan_step.outputs != 1 {
            continue;
        }
        let feeds = plan_step
            .inputs
            .iter()
            .any(|source| flat_previous(source) == Some((step, 0)));
        if !feeds {
            continue;
        }
        let mut mask = None;
        for source in &plan_step.inputs {
            if let PlanSource::Constant(value @ PlanConstant::VocabMask { .. }) = source {
                if mask.replace(value).is_some() {
                    return Err(Error::InvalidConfig);
                }
            }
        }
        let Some(PlanConstant::VocabMask {
            rows,
            vocab,
            valid,
            ..
        }) = mask
        else {
            return Err(Error::StatementMismatch);
        };
        if *rows != positions || *vocab != columns || *valid == 0 || *valid >= columns {
            return Err(Error::StatementMismatch);
        }
        if masked.replace(index).is_some() {
            return Err(Error::InvalidConfig);
        }
    }
    let masked_step = masked.ok_or(Error::StatementMismatch)?;

    // Argmax stays a separate graph item over the masked output, one row per
    // position. Selecting the output head does not depend on any sampled
    // column.
    if plan.argmax_steps.len() != positions {
        return Err(Error::StatementMismatch);
    }
    let expected_rows: Vec<PlanLayout> = (0..positions)
        .map(|row| PlanLayout::rows(columns, row, 1))
        .collect();
    for layout in &expected_rows {
        let count = plan
            .argmax_steps
            .iter()
            .filter(|index| {
                plan.steps.get(**index).is_some_and(|plan_step| {
                    plan_step.opcode == OpCode::Argmax
                        && plan_step.inputs.len() == 1
                        && plan_step.inputs[0]
                            == PlanSource::Prev {
                                step: masked_step,
                                output: 0,
                                layout: layout.clone(),
                            }
                })
            })
            .count();
        if count != 1 {
            return Err(Error::StatementMismatch);
        }
    }

    let previous =
        PreviousObjectPlan::derive_for_activation(plan, activation_step, positions, hidden)?;
    let declared = plan
        .previous_object
        .as_ref()
        .ok_or(Error::StatementMismatch)?;
    declared.validate_for_activation(plan, activation_step, positions, hidden)?;

    Ok(OutputHeadStep {
        step,
        activation_step,
        activation_output,
        masked_step,
        positions,
        hidden,
        columns,
        asset_path: asset_path.to_owned(),
        previous,
    })
}

/// Selects the unique output-head step of the trusted plan and checks the
/// separate vocabulary-mask and argmax items around it.
pub fn select_output_head_step(plan: &TracePlan) -> Result<OutputHeadStep> {
    select_output_head_step_for(
        plan,
        OUTPUT_HEAD_POSITIONS,
        OUTPUT_HEAD_HIDDEN,
        OUTPUT_HEAD_COLUMNS,
        OUTPUT_HEAD_ASSET_PATH,
    )
}

fn fp3_from_signed_i64(value: i64) -> Fp3 {
    Fp3::from_base(fp_from_signed_word(value as u64))
}

/// The trusted plan items and private inputs of one merged output-head
/// request. `send` carries the table identity, the shape of the reduced weight
/// message and the group/oracle numbers; its activation fields are filled from
/// the committed activation. The proximity plan fixes both groups, the fresh
/// masks, the column draws and the BASE position counts. All fields are plan
/// or caller inputs, never proof bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputHeadLinSetup {
    /// The weight asset path named by the trusted MatMul step.
    pub asset_path: String,
    pub send: HiddenSendPlan,
    pub proximity: HiddenProximityPlan,
    /// The sampled index-mask message of the hidden-input path. The diagnostic
    /// path leaves it empty and keeps plaintext output rows.
    pub index_mask: Option<IndexMaskPlan>,
    pub activation_configuration: Digest,
    pub activation_salts: Vec<Vec<u8>>,
    pub seed: Digest,
    pub masks: Vec<Vec<LinMask>>,
    /// The diagnostic verifier's pinned activation root. The prover ignores
    /// it. The formal path takes its pin from the operator policy instead and
    /// never reads this field.
    pub expected_activation_root: Option<Digest>,
    /// An optional fixture expectation for the reduced roots. The prover may
    /// leave it empty; the hidden-input verifier accepts proof-valued roots
    /// and checks them through the openings. It is a test hook, not a policy
    /// pin.
    pub expected_roots: Option<Vec<Digest>>,
    /// The trusted shapes and ids of the output-closure messages. The formal
    /// hidden-input path requires it.
    pub closure: Option<OutputClosureSetup>,
}

/// The trusted setup of the output-closure messages: the slack and packed-bit
/// message shapes, their oracle bases and opening counts, and the mask-only
/// message plan. The integer bounds and the public winner come from the
/// trusted output relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputClosureSetup {
    pub slack: WShape,
    pub slack_oracle_base: u64,
    pub slack_openings: usize,
    pub bits: WShape,
    pub bits_oracle_base: u64,
    pub bits_openings: usize,
    pub map: WShape,
    pub map_oracle_base: u64,
    pub map_openings: usize,
    pub mask: ClosureMaskPlan,
}

/// The merged request proof. One session covers every position; the canonical
/// encoding is the session's own message encoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputHeadLinProof {
    pub session: HiddenLinMessages,
}

impl OutputHeadLinProof {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(OUTPUT_HEAD_PROOF_VERSION);
        writer.put_bytes(&self.session.encode()?);
        Ok(writer.into_bytes())
    }

    pub fn decode(plan: &HiddenLinPlan, bytes: &[u8]) -> Result<Self> {
        let limits = DecodeLimits {
            max_bytes: bytes.len(),
            max_elements: bytes.len(),
        };
        let mut reader = CanonicalReader::new(bytes, limits);
        let version = reader.get_u64()?;
        if version < OUTPUT_HEAD_PROOF_VERSION {
            return Err(Error::UnsupportedVersion);
        }
        if version != OUTPUT_HEAD_PROOF_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let session = HiddenLinMessages::decode(plan, reader.get_bytes()?)?;
        reader.finish()?;
        Ok(Self { session })
    }

    pub fn encoded_len(&self) -> Result<usize> {
        Ok(8 + 8 + self.session.encode()?.len())
    }
}

/// The verifier outcome over all positions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputHeadLinOutcome {
    pub outcome: HiddenLinOutcome,
    pub proof_bytes: usize,
}


// ---------------------------------------------------------------------------
// Trusted-input closure (formal path)
// ---------------------------------------------------------------------------
//
// This is the formal path: the verifier receives neither `X` nor `Y`. `X` is
// an independent trusted input whose hiding commitment is pinned by the
// operator policy; the proof session carries the same commitment and the
// verifier compares the two. `Y` is the prover's reduced row `W = X M`: it is
// committed as the SEND group members under a hiding product-form encoding.
// The Eval targets are masked as `W_t(psi_t) + e_t`, with the one-time `e_t`
// entry of the sampled index-mask message, and the masked Dot right-hand side
// is `W_t(psi*) + g_t`; both rows are checked by the own-group proximity test
// against the committed roots. The activation rows are also committed as one
// shown-map message; after the tilde values and the batching challenge, the
// masked product-sumcheck transcript yields the `(U)` row and the two `(E_j)`
// end-value rows; one public difference row joins the chi-weighted claim and
// tilde masks to the residual `R = sum_t chi^t (g_t - c_t)`, which also
// adjusts the product-sumcheck target. The claim and tilde masks are
// independent; the difference row does not force them to be equal. Those rows
// are checked by the same own-group test, so the
// verifier checks `<X_t, Omega> = W_t(psi*)` over the committed shown maps and
// the committed reduced rows. The vocabulary mask, validity range, ordering
// comparison, first-maximum tie rule and final output are trusted graph
// values; their digest is absorbed into the first statement.
//
// The shown-map commitment and the activation commitment are two commitments
// of the same `X`; the verifier pins both, and their equality is an
// operator-policy warranty, not a cross-commitment proof. The diagnostic
// module below keeps the plaintext-`X`/plaintext-`Y` paths for development and
// real trace fixtures only.

/// The trusted output relation of the output-head graph: the vocabulary mask,
/// the valid-vocabulary range, the ordering comparison, the first-maximum tie
/// rule and the expected final output. The proof does not supply any field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputHeadOutputPlan {
    /// The selected MatMul step and its graph neighbours.
    pub step: OutputHeadStep,
    /// Number of valid vocabulary columns. Columns at or above this index are
    /// masked out of the ordering comparison.
    pub valid: usize,
    /// The complete padded column domain of the mask.
    pub vocab: usize,
    /// The constant added to every column at or above `valid`.
    pub mask_value: i64,
    /// The separate argmax step of every position, in position order.
    pub argmax_steps: Vec<usize>,
    /// The expected final output, one index per position.
    pub expected_argmax: Vec<u32>,
    /// Version of the output relation. Version 2 carries the exact integer
    /// range and comparison closure.
    pub relation_version: u64,
    /// The integer bound `D` of `|Y_j|`: `A = 2^D - 1`.
    pub range_bits: u32,
    /// The slack bound exponent `b`: `B = 2^b`.
    pub slack_bits: u32,
}

impl OutputHeadOutputPlan {
    /// Checks the declaration against the matrix shape and its own ranges. A
    /// missing argmax step, an out-of-range name or an expected index outside
    /// the valid vocabulary is rejected.
    pub fn validate(&self, shape: LinShape) -> Result<()> {
        shape.check()?;
        if self.step.positions == 0
            || self.step.hidden as u64 != shape.matrix_rows
            || self.step.columns as u64 != shape.matrix_columns()?
            || self.vocab != self.step.columns
            || self.valid == 0
            || self.valid >= self.vocab
            || self.mask_value >= 0
        {
            return Err(Error::InvalidConfig);
        }
        if self.argmax_steps.len() != self.step.positions
            || self.expected_argmax.len() != self.step.positions
        {
            return Err(Error::ShapeMismatch);
        }
        if self
            .expected_argmax
            .iter()
            .any(|index| *index as usize >= self.valid)
        {
            return Err(Error::BoundViolation);
        }
        if self.relation_version != 2
            || self.range_bits == 0
            || self.range_bits > 62
            || self.slack_bits == 0
            || self.slack_bits > 62
        {
            return Err(Error::InvalidConfig);
        }
        let offset = (1u128 << self.range_bits) - 1;
        if (self.mask_value as i128) > -((offset as i128) + 1) {
            return Err(Error::InvalidConfig);
        }
        let bound = 1u128 << self.slack_bits;
        let no_wrap = bound
            .checked_add(2 * offset)
            .and_then(|value| value.checked_add((self.mask_value as i128).unsigned_abs()));
        match no_wrap {
            Some(value) if value < MODULUS as u128 => {}
            _ => return Err(Error::BoundViolation),
        }
        Ok(())
    }

    /// The canonical digest absorbed into the first statement.
    pub fn digest(&self) -> Result<Digest> {
        let mut writer = CanonicalWriter::new();
        writer.put_bytes(b"zkie/output-head-relation");
        writer.put_u64(self.relation_version);
        writer.put_u64(self.step.step as u64);
        writer.put_u64(self.step.activation_step as u64);
        writer.put_u64(self.step.activation_output as u64);
        writer.put_u64(self.step.masked_step as u64);
        writer.put_u64(self.step.positions as u64);
        writer.put_u64(self.step.hidden as u64);
        writer.put_u64(self.step.columns as u64);
        writer.put_bytes(self.step.asset_path.as_bytes());
        writer.put_u64(self.valid as u64);
        writer.put_u64(self.vocab as u64);
        writer.put_u64(self.mask_value as u64);
        writer.put_u64(self.argmax_steps.len() as u64);
        for step in &self.argmax_steps {
            writer.put_u64(*step as u64);
        }
        writer.put_u64(self.expected_argmax.len() as u64);
        for value in &self.expected_argmax {
            writer.put_u64(*value as u64);
        }
        writer.put_u64(self.range_bits as u64);
        writer.put_u64(self.slack_bits as u64);
        hash_labeled(tag_word(STATEMENT, 2), 0, None, &writer.into_bytes())
    }
}

/// The local result of the full-row output closure: the first-maximum index of
/// every position and a digest over the complete masked rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputHeadClosure {
    pub argmax: Vec<u32>,
    pub masked_digest: Digest,
}

/// Runs the complete masked output relation over the full rows. Every column
/// takes part: the mask is applied to the padded domain, the ordering
/// comparison is a strict `>` scan from the first column so a tie keeps the
/// earliest index, and the resulting index must equal the trusted expected
/// output. This is the prover-side local closure; it never samples columns.
pub fn close_output_relation(
    plan: &OutputHeadOutputPlan,
    rows: &[Vec<i64>],
) -> Result<OutputHeadClosure> {
    if rows.len() != plan.step.positions || rows.iter().any(|row| row.len() != plan.step.columns) {
        return Err(Error::ShapeMismatch);
    }
    let mut writer = CanonicalWriter::new();
    writer.put_bytes(b"zkie/output-head-masked");
    writer.put_u64(rows.len() as u64);
    writer.put_u64(plan.step.columns as u64);
    let mut argmax = Vec::with_capacity(rows.len());
    for (position, row) in rows.iter().enumerate() {
        let mut best = 0usize;
        let mut best_value = i64::MIN;
        for (column, value) in row.iter().enumerate() {
            let masked = if column >= plan.valid {
                value.checked_add(plan.mask_value).ok_or(Error::BoundViolation)?
            } else {
                *value
            };
            writer.put_u64(masked as u64);
            if column == 0 || masked > best_value {
                best = column;
                best_value = masked;
            }
        }
        if best >= plan.valid {
            return Err(Error::StatementMismatch);
        }
        if plan.expected_argmax.get(position) != Some(&(best as u32)) {
            return Err(Error::StatementMismatch);
        }
        argmax.push(best as u32);
    }
    let masked_digest = hash_labeled(
        tag_word(STATEMENT, 3),
        0,
        None,
        &writer.into_bytes(),
    )?;
    Ok(OutputHeadClosure {
        argmax,
        masked_digest,
    })
}

/// The verifier's trusted inputs: the operator-pinned activation commitment
/// and the trusted output relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputHeadTrustedInputs {
    pub activation: ActivationCommitment,
    pub output: OutputHeadOutputPlan,
}

fn hidden_plan(
    shape: LinShape,
    setup: &OutputHeadLinSetup,
    reader: &IndexReader,
    activation: ActivationCommitment,
    output: &OutputHeadOutputPlan,
) -> Result<HiddenLinPlan> {
    shape.check()?;
    if activation.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    if setup.send.table_shape != reader.shape
        || setup.send.table_identity != reader_table_identity(reader)?
        || setup.send.matrix_rows != shape.matrix_rows
        || setup.send.k_lo != shape.k_lo
        || setup.send.columns != shape.matrix_columns()?
    {
        return Err(Error::StatementMismatch);
    }
    if output.step.positions != activation.shape.positions() as usize
        || output.step.hidden as u64 != shape.matrix_rows
        || output.step.columns as u64 != shape.matrix_columns()?
    {
        return Err(Error::ShapeMismatch);
    }
    let mut send = setup.send.clone();
    send.activation = activation.shape;
    send.activation_configuration = activation.configuration;
    send.activation_root = activation.root;
    send.index_mask = setup.index_mask.clone();
    send.dot = if setup.index_mask.is_some() {
        setup.send.dot.clone()
    } else {
        None
    };
    send.expected_roots = setup.expected_roots.clone();
    let closure = match &setup.closure {
        Some(closure) => {
            let width = (output.range_bits as u64 + 1)
                .checked_mul(2)
                .and_then(|value| value.checked_add(output.slack_bits as u64))
                .ok_or(Error::ResourceLimit)?;
            let columns = (output.vocab as u64)
                .checked_mul(width)
                .ok_or(Error::ResourceLimit)?;
            if columns == 0 || columns > (1u64 << 30) {
                return Err(Error::ResourceLimit);
            }
            let bits_columns = columns.next_power_of_two();
            let boolean_variables = bits_columns.trailing_zeros();
            let plan = ClosurePlan {
                range_bits: output.range_bits,
                slack_bits: output.slack_bits,
                valid: output.valid,
                vocab: output.vocab,
                mask_value: output.mask_value,
                winner: output.expected_argmax.clone(),
                slack: closure.slack.clone(),
                slack_oracle_base: closure.slack_oracle_base,
                slack_openings: closure.slack_openings,
                bits: closure.bits.clone(),
                bits_oracle_base: closure.bits_oracle_base,
                bits_openings: closure.bits_openings,
                map: closure.map.clone(),
                map_oracle_base: closure.map_oracle_base,
                map_openings: closure.map_openings,
                bits_columns,
                bits_per_cell: width as u32,
                boolean_variables,
                mask: closure.mask.clone(),
            };
            if plan.fault(&send).is_some() {
                return Err(Error::InvalidConfig);
            }
            Some(plan)
        }
        None => None,
    };
    let plan = HiddenLinPlan {
        norm: None,
        send,
        output: HiddenOutputShape {
            positions: activation.shape.positions,
            columns: shape.matrix_columns()?,
        },
        weights: Vec::new(),
        masks: vec![Vec::new(); activation.shape.positions()],
        proximity: setup.proximity.clone(),
        output_relation: Some(output.digest()?),
        closure,
    };
    plan.validate()?;
    Ok(plan)
}

/// Builds the verifier's hidden-input plan from trusted values. The caller
/// supplies the trusted setup (table identity, proximity shapes and seed
/// policy), the committed table reader and the pin/relation pair; no prover
/// message contributes to this plan.
pub fn build_output_head_hidden_plan(
    shape: LinShape,
    setup: &OutputHeadLinSetup,
    reader: &IndexReader,
    trusted: &OutputHeadTrustedInputs,
) -> Result<HiddenLinPlan> {
    trusted.output.validate(shape)?;
    hidden_plan(
        shape,
        setup,
        reader,
        trusted.activation,
        &trusted.output,
    )
}

/// Converts one committed base-field value back to its signed integer. A
/// non-base extension value or a magnitude outside `i64` is rejected, so an
/// unsigned modular reduction of a negative value cannot pass as the same
/// signed value.
fn fp3_to_signed_i64(value: &Fp3) -> Result<i64> {
    if value.c[1].as_canonical_u64() != 0 || value.c[2].as_canonical_u64() != 0 {
        return Err(Error::BoundViolation);
    }
    let canonical = value.c[0].as_canonical_u64();
    if canonical <= i64::MAX as u64 {
        return Ok(canonical as i64);
    }
    let magnitude = MODULUS
        .checked_sub(canonical)
        .ok_or(Error::BoundViolation)?;
    if magnitude > (1u64 << 63) {
        return Err(Error::BoundViolation);
    }
    if magnitude == (1u64 << 63) {
        Ok(i64::MIN)
    } else {
        Ok(-(magnitude as i64))
    }
}

/// Reconstructs the full reduced rows from the prover's committed message
/// rows. This is a prover-side helper; the rows never enter the verifier.
pub fn witness_output_rows(witness: &HiddenLinWitness) -> Result<Vec<Vec<i64>>> {
    let positions = witness.send.positions as usize;
    let columns = usize::try_from(witness.send.columns).map_err(|_| Error::ResourceLimit)?;
    let rows = usize::try_from(witness.send.w.main_rows).map_err(|_| Error::ResourceLimit)?;
    let places = usize::try_from(witness.send.w.code.ell).map_err(|_| Error::ResourceLimit)?;
    let mut output = Vec::with_capacity(positions);
    for position in 0..positions {
        let message = witness
            .send
            .messages
            .get(position)
            .ok_or(Error::ShapeMismatch)?;
        if message.len() != rows || message.iter().any(|row| row.len() != places) {
            return Err(Error::ShapeMismatch);
        }
        let mut values = Vec::with_capacity(columns);
        for column in 0..columns {
            let row = column % rows;
            let place = column / rows;
            let value = message
                .get(row)
                .and_then(|row| row.get(place))
                .ok_or(Error::ShapeMismatch)?;
            values.push(fp3_to_signed_i64(value)?);
        }
        output.push(values);
    }
    Ok(output)
}

/// Builds the trusted output relation from the trusted graph plan and the
/// trusted expected output. The mask constant, its valid range and its
/// separate argmax consumers are read from the plan, not from the proof.
pub fn build_output_head_output_plan(
    plan: &TracePlan,
    expected_argmax: &[u32],
) -> Result<OutputHeadOutputPlan> {
    build_output_head_output_plan_for(
        plan,
        OUTPUT_HEAD_POSITIONS,
        OUTPUT_HEAD_HIDDEN,
        OUTPUT_HEAD_COLUMNS,
        OUTPUT_HEAD_ASSET_PATH,
        OUTPUT_HEAD_RANGE_BITS,
        OUTPUT_HEAD_SLACK_BITS,
        expected_argmax,
    )
}

/// The explicit-dimension form of [`build_output_head_output_plan`]. The
/// trusted caller supplies the graph dimensions and the weight asset path; the
/// production entry fixes them to the output-head constants.
pub fn build_output_head_output_plan_for(
    plan: &TracePlan,
    positions: usize,
    hidden: usize,
    columns: usize,
    asset_path: &str,
    range_bits: u32,
    slack_bits: u32,
    expected_argmax: &[u32],
) -> Result<OutputHeadOutputPlan> {
    let step = select_output_head_step_for(plan, positions, hidden, columns, asset_path)?;
    let masked = plan
        .steps
        .get(step.masked_step)
        .ok_or(Error::StatementMismatch)?;
    let mut mask = None;
    for source in &masked.inputs {
        if let PlanSource::Constant(PlanConstant::VocabMask {
            rows,
            vocab,
            valid,
            masked,
        }) = source
        {
            if mask.replace((*rows, *vocab, *valid, *masked)).is_some() {
                return Err(Error::InvalidConfig);
            }
        }
    }
    let (rows, vocab, valid, mask_value) = mask.ok_or(Error::StatementMismatch)?;
    if rows != step.positions
        || vocab != step.columns
        || valid == 0
        || valid >= step.columns
        || mask_value >= 0
    {
        return Err(Error::StatementMismatch);
    }
    if plan.argmax_steps.len() != step.positions || expected_argmax.len() != step.positions {
        return Err(Error::ShapeMismatch);
    }
    for (position, index) in plan.argmax_steps.iter().enumerate() {
        let layout = PlanLayout::rows(step.columns, position, 1);
        let expected_source = PlanSource::Prev {
            step: step.masked_step,
            output: 0,
            layout,
        };
        let argmax = plan
            .steps
            .get(*index)
            .ok_or(Error::StatementMismatch)?;
        if argmax.opcode != OpCode::Argmax
            || argmax.inputs.len() != 1
            || argmax.inputs[0] != expected_source
        {
            return Err(Error::StatementMismatch);
        }
    }
    let output = OutputHeadOutputPlan {
        step,
        valid,
        vocab,
        mask_value,
        argmax_steps: plan.argmax_steps.clone(),
        expected_argmax: expected_argmax.to_vec(),
        relation_version: 2,
        range_bits,
        slack_bits,
    };
    Ok(output)
}

/// Prover entry of the formal path. `X` is a private trusted input supplied
/// outside the proof; the prover commits it with fresh salts, derives the
/// reduced rows from the committed table, runs the hidden three-stage session
/// and closes the full-row output relation. The returned proof carries only
/// the hiding commitments and the public message; the closure result is for
/// the caller's own checks.
pub fn prove_output_head_lin_hidden(
    shape: LinShape,
    setup: &OutputHeadLinSetup,
    output: &OutputHeadOutputPlan,
    table: &mut dyn HiddenTableProver,
    activation_rows: &[Vec<i64>],
    activation_salts: &[Vec<u8>],
) -> Result<(OutputHeadLinProof, HiddenLinWitness, OutputHeadClosure)> {
    let _profile_total = zkie_engine::profile::Scope::new("total.prove_output_head_lin_hidden");
    shape.check()?;
    output.validate(shape)?;
    let positions = output.step.positions;
    let hidden = output.step.hidden;
    if activation_rows.len() != positions
        || activation_rows.iter().any(|row| row.len() != hidden)
        || activation_salts.len() != positions
    {
        return Err(Error::ShapeMismatch);
    }
    let salt_bytes = activation_salts
        .first()
        .map(Vec::len)
        .ok_or(Error::InvalidConfig)?;
    if activation_salts.iter().any(|salt| salt.len() != salt_bytes) {
        return Err(Error::ShapeMismatch);
    }
    let activation_shape =
        ActivationShape::new(positions as u32, hidden as u32, salt_bytes)?;
    let _profile_activation = zkie_engine::profile::Scope::new("activation.commit_signed");
    let activation = CommittedActivation::commit_signed(
        activation_shape,
        activation_rows,
        activation_salts,
        setup.activation_configuration,
    )?;
    drop(_profile_activation);
    let reader = table.reader().clone();
    let _profile_plan = zkie_engine::profile::Scope::new("plan.hidden_plan");
    let plan = hidden_plan(shape, setup, &reader, *activation.commitment(), output)
        ?;
    drop(_profile_plan);
    let (session, witness) = prove_hidden_lin_hidden(&plan, &activation, table, setup.seed)
        ?;
    let _profile_rows = zkie_engine::profile::Scope::new("post.witness_output_rows");
    let rows = witness_output_rows(&witness)
        ?;
    drop(_profile_rows);
    let _profile_close = zkie_engine::profile::Scope::new("post.close_output_relation");
    let closure = close_output_relation(output, &rows)
        ?;
    drop(_profile_close);
    Ok((OutputHeadLinProof { session }, witness, closure))
}

/// Verifier entry of the formal path. The plan is rebuilt by the verifier from
/// trusted values; the activation commitment in the proof session must equal
/// the operator-pinned commitment. Neither `X` nor `Y` enters this entry.
pub fn verify_output_head_lin_hidden(
    shape: LinShape,
    plan: &HiddenLinPlan,
    trusted: &OutputHeadTrustedInputs,
    reader: &IndexReader,
    proof: &OutputHeadLinProof,
) -> Result<OutputHeadLinOutcome> {
    let _profile = zkie_engine::profile::Scope::new("verify.output_head_lin_hidden");
    shape.check()?;
    trusted.output.validate(shape)?;
    if !plan.weights.is_empty() {
        return Err(Error::InvalidConfig);
    }
    if plan.send.activation_root != trusted.activation.root
        || plan.send.activation != trusted.activation.shape
        || plan.send.activation_configuration != trusted.activation.configuration
    {
        return Err(Error::StatementMismatch);
    }
    if plan.output.positions != trusted.output.step.positions as u32
        || plan.output.columns != trusted.output.step.columns as u64
    {
        return Err(Error::StatementMismatch);
    }
    if plan.output_relation != Some(trusted.output.digest()?) {
        return Err(Error::StatementMismatch);
    }
    let closure = plan.closure.as_ref().ok_or(Error::InvalidConfig)?;
    if closure.range_bits != trusted.output.range_bits
        || closure.slack_bits != trusted.output.slack_bits
        || closure.valid != trusted.output.valid
        || closure.vocab != trusted.output.vocab
        || closure.mask_value != trusted.output.mask_value
        || closure.winner != trusted.output.expected_argmax
    {
        return Err(Error::StatementMismatch);
    }
    if proof.session.first.activation != trusted.activation {
        return Err(Error::StatementMismatch);
    }
    let outcome = verify_hidden_lin_hidden(plan, reader, &proof.session)?;
    if outcome.activation != trusted.activation {
        return Err(Error::StatementMismatch);
    }
    Ok(OutputHeadLinOutcome {
        outcome,
        proof_bytes: proof.encoded_len()?,
    })
}

/// Verifier entry of the formal path over the canonical request encoding.
pub fn verify_output_head_lin_hidden_bytes(
    shape: LinShape,
    plan: &HiddenLinPlan,
    trusted: &OutputHeadTrustedInputs,
    reader: &IndexReader,
    bytes: &[u8],
) -> Result<OutputHeadLinOutcome> {
    let proof = OutputHeadLinProof::decode(plan, bytes)?;
    let mut outcome = verify_output_head_lin_hidden(shape, plan, trusted, reader, &proof)?;
    outcome.proof_bytes = bytes.len();
    Ok(outcome)
}


pub mod diagnostic {
//! Non-production diagnostic path.
//!
//! These entries read plaintext `X` and `Y` from a verified shared trace and
//! exist for development, unit tests and the real-fixture harness. They are
//! not the hidden production path; the formal entries in the parent module do
//! not call them and never accept plaintext `X` or `Y`.
use super::*;

/// The activation and output tensors of the verified trace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputHeadInputs {
    /// Row-major `positions x hidden` activation `X`.
    pub activation: Vec<i64>,
    /// Row-major `positions x columns` output `Y`.
    pub output: Vec<i64>,
}

/// The plan binding and trace tensors one request consumes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputHeadTrace {
    pub step: OutputHeadStep,
    pub binding: BindingCandidate,
    pub inputs: OutputHeadInputs,
    /// Index of the output-head asset opening in the verified trace.
    pub asset_index: usize,
}

pub(crate) fn prepare_output_head_trace_for(
    public: &ModelCommitmentPublic,
    plan: &TracePlan,
    trace: &SharedTrace,
    verified_outputs: &[Vec<Vec<i64>>],
    shape: LinShape,
    positions: usize,
    hidden: usize,
    columns: usize,
    asset_path: &str,
) -> Result<OutputHeadTrace> {
    shape.check()?;
    if shape.matrix_rows != hidden as u64 || shape.matrix_columns()? != columns as u64 {
        return Err(Error::ShapeMismatch);
    }
    let step = select_output_head_step_for(plan, positions, hidden, columns, asset_path)?;

    let activation = verified_outputs
        .get(step.activation_step)
        .and_then(|outputs| outputs.get(step.activation_output))
        .ok_or(Error::StatementMismatch)?;
    let activation_len = positions
        .checked_mul(hidden)
        .ok_or(Error::ResourceLimit)?;
    if activation.len() != activation_len {
        return Err(Error::ShapeMismatch);
    }
    for value in activation {
        i32::try_from(*value).map_err(|_| Error::BoundViolation)?;
    }

    let output = verified_outputs
        .get(step.step)
        .and_then(|outputs| outputs.first())
        .ok_or(Error::StatementMismatch)?;
    let output_len = positions.checked_mul(columns).ok_or(Error::ResourceLimit)?;
    if output.len() != output_len {
        return Err(Error::ShapeMismatch);
    }

    let meta = public
        .assets
        .iter()
        .find(|asset| asset.path == asset_path)
        .ok_or(Error::AssetMismatch)?;
    if meta.row_width < columns {
        return Err(Error::ShapeMismatch);
    }
    let asset_index = trace
        .assets
        .iter()
        .position(|opening| opening.asset_id == meta.asset_id)
        .ok_or(Error::StatementMismatch)?;
    let opening = trace
        .assets
        .get(asset_index)
        .ok_or(Error::IncompleteArtifact)?;
    if opening.start != meta.start
        || opening.root != meta.root
        || opening.row_count != meta.rows
        || opening.rows.len() as u64 != meta.rows
    {
        return Err(Error::StatementMismatch);
    }

    let selection = SelectedMatmulStep::select(plan, step.step)?;
    let binding = assemble_binding(public, plan, &selection)?;

    Ok(OutputHeadTrace {
        step,
        binding,
        inputs: OutputHeadInputs {
            activation: activation.clone(),
            output: output.clone(),
        },
        asset_index,
    })
}

/// Extracts the output-head activation and output from the verified trace
/// outputs, authenticates the plan step and its asset identity, and assembles
/// the request binding. `verified_outputs` must be the result of
/// `verify_shared_trace_outputs` for the same trace and plan.
pub fn prepare_output_head_trace(
    public: &ModelCommitmentPublic,
    plan: &TracePlan,
    trace: &SharedTrace,
    verified_outputs: &[Vec<Vec<i64>>],
    shape: LinShape,
) -> Result<OutputHeadTrace> {
    if shape != output_head_shape()? {
        return Err(Error::ShapeMismatch);
    }
    prepare_output_head_trace_for(
        public,
        plan,
        trace,
        verified_outputs,
        shape,
        OUTPUT_HEAD_POSITIONS,
        OUTPUT_HEAD_HIDDEN,
        OUTPUT_HEAD_COLUMNS,
        OUTPUT_HEAD_ASSET_PATH,
    )
}


// ---------------------------------------------------------------------------
// Public claims and the merged setup
// ---------------------------------------------------------------------------

/// The per-position public weights and field values derived from the verified
/// trace. `weights` is the activation row `g_t`; `output_values` is `Fp(Y_t)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputHeadLinClaims {
    pub shape: LinShape,
    pub weights: Vec<Vec<Fp3>>,
    pub output_values: Vec<Vec<Fp3>>,
    pub masks: Vec<Vec<LinMask>>,
}

/// Builds the public claim items. The activation must fit `i32`; the output is
/// the checked `i64` MatMul result. The weights and the output are public and
/// are never proof bytes. Masks are empty when the caller passes an empty
/// outer slice.
pub fn output_head_claims(
    shape: LinShape,
    inputs: &OutputHeadInputs,
    masks: &[Vec<LinMask>],
) -> Result<OutputHeadLinClaims> {
    shape.check()?;
    let rows = usize::try_from(shape.matrix_rows).map_err(|_| Error::ResourceLimit)?;
    let columns = usize::try_from(shape.matrix_columns()?).map_err(|_| Error::ResourceLimit)?;
    if rows == 0 || inputs.activation.len() % rows != 0 {
        return Err(Error::ShapeMismatch);
    }
    let positions = inputs.activation.len() / rows;
    if positions == 0 {
        return Err(Error::InvalidConfig);
    }
    let expected_output = positions.checked_mul(columns).ok_or(Error::ResourceLimit)?;
    if inputs.output.len() != expected_output {
        return Err(Error::ShapeMismatch);
    }
    let masks = if masks.is_empty() {
        vec![Vec::new(); positions]
    } else if masks.len() == positions {
        masks.to_vec()
    } else {
        return Err(Error::ShapeMismatch);
    };
    let mut weights = Vec::with_capacity(positions);
    let mut output_values = Vec::with_capacity(positions);
    for position in 0..positions {
        let mut weight = Vec::with_capacity(rows);
        for row in 0..rows {
            let value = inputs.activation[position * rows + row];
            let value = i32::try_from(value).map_err(|_| Error::BoundViolation)?;
            weight.push(fp3_from_signed_i64(value as i64));
        }
        let mut output = Vec::with_capacity(columns);
        for column in 0..columns {
            output.push(fp3_from_signed_i64(
                inputs.output[position * columns + column],
            ));
        }
        weights.push(weight);
        output_values.push(output);
    }
    Ok(OutputHeadLinClaims {
        shape,
        weights,
        output_values,
        masks,
    })
}

pub(crate) fn merged_plan(
    shape: LinShape,
    setup: &OutputHeadLinSetup,
    reader: &IndexReader,
    claims: &OutputHeadLinClaims,
    activation_root: Digest,
    expected_roots: Option<Vec<Digest>>,
) -> Result<HiddenLinPlan> {
    shape.check()?;
    if claims.shape != shape
        || claims.weights.len() != claims.masks.len()
        || claims.weights.is_empty()
        || claims.output_values.len() != claims.weights.len()
    {
        return Err(Error::ShapeMismatch);
    }
    if setup.send.table_shape != reader.shape
        || setup.send.table_identity != reader_table_identity(reader)?
        || setup.send.matrix_rows != shape.matrix_rows
        || setup.send.k_lo != shape.k_lo
        || setup.send.columns != shape.matrix_columns()?
    {
        return Err(Error::StatementMismatch);
    }
    let positions = u32::try_from(claims.weights.len()).map_err(|_| Error::ResourceLimit)?;
    let hidden = u32::try_from(shape.matrix_rows).map_err(|_| Error::ResourceLimit)?;
    if setup.activation_salts.len() != positions as usize {
        return Err(Error::ShapeMismatch);
    }
    let salt_bytes = setup
        .activation_salts
        .first()
        .map(Vec::len)
        .ok_or(Error::InvalidConfig)?;
    if setup
        .activation_salts
        .iter()
        .any(|salt| salt.len() != salt_bytes)
    {
        return Err(Error::ShapeMismatch);
    }
    let activation = ActivationShape::new(positions, hidden, salt_bytes)?;
    let mut send = setup.send.clone();
    send.activation = activation;
    send.activation_configuration = setup.activation_configuration;
    send.activation_root = activation_root;
    send.index_mask = None;
    send.dot = None;
    send.expected_roots = expected_roots;
    // The diagnostic mode has no mask message, so its fresh own mask has the
    // single row of a maskless group. The setup's proximity plan describes the
    // formal mode.
    let mut proximity = setup.proximity.clone();
    proximity.own_mask_tree.mask_rows = 1;
    // The setup's stage lists describe the formal hidden group, which carries
    // the Dot, index-mask and closure members. The diagnostic group drops
    // those members, so its fold shape differs; it keeps the single-stage
    // tail instead of claiming a stage chain that the trusted inputs do not
    // describe.
    proximity.shared_stages.clear();
    proximity.own_stages.clear();
    proximity.planner_checked = false;
    // The diagnostic group has no staged BASE; it keeps the two-position
    // single-stage basis, not the staged own BASE count.
    proximity.own_base_positions = 2;
    let plan = HiddenLinPlan {
        norm: None,
        send,
        output: HiddenOutputShape {
            positions,
            columns: shape.matrix_columns()?,
        },
        weights: claims.weights.clone(),
        masks: claims.masks.clone(),
        proximity,
        output_relation: None,
        closure: None,
    };
    plan.validate()?;
    Ok(plan)
}

pub(crate) fn verify_trace(
    public: &ModelCommitmentPublic,
    plan: &TracePlan,
    trace: &SharedTrace,
    expected_tokens: &[u32],
    shape: LinShape,
    setup: &OutputHeadLinSetup,
) -> Result<(OutputHeadTrace, OutputHeadLinClaims)> {
    shape.check()?;
    let outputs = verify_shared_trace_outputs(public, trace, expected_tokens, plan)?;
    let rows = usize::try_from(shape.matrix_rows).map_err(|_| Error::ResourceLimit)?;
    let columns = usize::try_from(shape.matrix_columns()?).map_err(|_| Error::ResourceLimit)?;
    let positions = outputs
        .first()
        .and_then(|outputs| outputs.first())
        .map(|values| values.len() / rows)
        .filter(|positions| *positions > 0 && values_match(shape, *positions))
        .ok_or(Error::ShapeMismatch)?;
    let request = prepare_output_head_trace_for(
        public,
        plan,
        trace,
        &outputs,
        shape,
        positions,
        rows,
        columns,
        &setup.asset_path,
    )?;
    let claims = output_head_claims(shape, &request.inputs, &[])?;
    Ok((request, claims))
}

pub(crate) fn values_match(shape: LinShape, positions: usize) -> bool {
    positions
        .checked_mul(usize::try_from(shape.matrix_rows).unwrap_or(0))
        .is_some()
}

/// Prover entry. The trace is verified for the model/anchor/step binding; the
/// activation rows are committed with fresh salts; the committed table is read
/// through the caller's table handle. Returns the merged proof and the private
/// witness used by unit tests.
pub fn prove_output_head_lin_request_with_witness(
    public: &ModelCommitmentPublic,
    plan: &TracePlan,
    trace: &SharedTrace,
    expected_tokens: &[u32],
    shape: LinShape,
    setup: &OutputHeadLinSetup,
    table: &mut dyn HiddenTableProver,
) -> Result<(OutputHeadLinProof, HiddenLinWitness)> {
    let (_request, claims) = verify_trace(public, plan, trace, expected_tokens, shape, setup)?;
    let masks = if setup.masks.is_empty() {
        vec![Vec::new(); claims.weights.len()]
    } else {
        setup.masks.clone()
    };
    let claims = OutputHeadLinClaims { masks, ..claims };
    let reader = table.reader().clone();
    let activation = CommittedActivation::commit(
        ActivationShape::new(
            claims.weights.len() as u32,
            shape.matrix_rows as u32,
            setup
                .activation_salts
                .first()
                .map(Vec::len)
                .ok_or(Error::InvalidConfig)?,
        )?,
        &claims.weights,
        &setup.activation_salts,
        setup.activation_configuration,
    )?;
    let merged = merged_plan(
        shape,
        setup,
        &reader,
        &claims,
        activation.root(),
        setup.expected_roots.clone(),
    )?;
    let (session, witness) =
        prove_hidden_lin(&merged, &activation, table, &claims.output_values, setup.seed)?;
    Ok((OutputHeadLinProof { session }, witness))
}

/// Prover entry without the private witness.
pub fn prove_output_head_lin_request(
    public: &ModelCommitmentPublic,
    plan: &TracePlan,
    trace: &SharedTrace,
    expected_tokens: &[u32],
    shape: LinShape,
    setup: &OutputHeadLinSetup,
    table: &mut dyn HiddenTableProver,
) -> Result<OutputHeadLinProof> {
    prove_output_head_lin_request_with_witness(
        public,
        plan,
        trace,
        expected_tokens,
        shape,
        setup,
        table,
    )
    .map(|(proof, _)| proof)
}

/// Prover entry over an already prepared request. The caller must have built
/// the request from the verified trace outputs and the trusted plan; this
/// entry does not re-read the trace. It is the fixture entry used by the unit
/// tests and by callers that already hold the verified tensors.
pub fn prove_output_head_lin_prepared(
    request: &OutputHeadTrace,
    shape: LinShape,
    setup: &OutputHeadLinSetup,
    table: &mut dyn HiddenTableProver,
) -> Result<(OutputHeadLinProof, HiddenLinWitness)> {
    let _ = &request.binding;
    let claims = output_head_claims(shape, &request.inputs, &setup.masks)?;
    let reader = table.reader().clone();
    let activation = CommittedActivation::commit(
        ActivationShape::new(
            claims.weights.len() as u32,
            shape.matrix_rows as u32,
            setup
                .activation_salts
                .first()
                .map(Vec::len)
                .ok_or(Error::InvalidConfig)?,
        )?,
        &claims.weights,
        &setup.activation_salts,
        setup.activation_configuration,
    )?;
    let merged = merged_plan(
        shape,
        setup,
        &reader,
        &claims,
        activation.root(),
        setup.expected_roots.clone(),
    )?;
    let (session, witness) =
        prove_hidden_lin(&merged, &activation, table, &claims.output_values, setup.seed)?;
    Ok((OutputHeadLinProof { session }, witness))
}

/// Verifier entry over an already prepared request.
pub fn verify_output_head_lin_prepared(
    request: &OutputHeadTrace,
    shape: LinShape,
    setup: &OutputHeadLinSetup,
    reader: &IndexReader,
    proof: &OutputHeadLinProof,
) -> Result<OutputHeadLinOutcome> {
    let _ = &request.binding;
    let claims = output_head_claims(shape, &request.inputs, &[])?;
    let masks = if setup.masks.is_empty() {
        vec![Vec::new(); claims.weights.len()]
    } else {
        setup.masks.clone()
    };
    let claims = OutputHeadLinClaims { masks, ..claims };
    let activation_root = setup
        .expected_activation_root
        .ok_or(Error::StatementMismatch)?;
    let expected_roots = setup.expected_roots.clone().ok_or(Error::StatementMismatch)?;
    if proof.session.first.activation.root != activation_root {
        return Err(Error::StatementMismatch);
    }
    let merged = merged_plan(
        shape,
        setup,
        reader,
        &claims,
        activation_root,
        Some(expected_roots),
    )?;
    let proof_bytes = proof.encoded_len()?;
    let outcome = verify_hidden_lin(&merged, reader, &claims.output_values, &proof.session)?;
    Ok(OutputHeadLinOutcome {
        outcome,
        proof_bytes,
    })
}

fn verify_merged(
    plan: &TracePlan,
    trace: &SharedTrace,
    expected_tokens: &[u32],
    shape: LinShape,
    setup: &OutputHeadLinSetup,
    public: &ModelCommitmentPublic,
    reader: &IndexReader,
    proof: &OutputHeadLinProof,
    proof_bytes: usize,
) -> Result<OutputHeadLinOutcome> {
    let (_request, claims) = verify_trace(public, plan, trace, expected_tokens, shape, setup)?;
    let masks = if setup.masks.is_empty() {
        vec![Vec::new(); claims.weights.len()]
    } else {
        setup.masks.clone()
    };
    let claims = OutputHeadLinClaims { masks, ..claims };
    let activation_root = setup
        .expected_activation_root
        .ok_or(Error::StatementMismatch)?;
    let expected_roots = setup.expected_roots.clone().ok_or(Error::StatementMismatch)?;
    if proof.session.first.activation.root != activation_root {
        return Err(Error::StatementMismatch);
    }
    let merged = merged_plan(
        shape,
        setup,
        reader,
        &claims,
        activation_root,
        Some(expected_roots),
    )?;
    let outcome = verify_hidden_lin(&merged, reader, &claims.output_values, &proof.session)?;
    Ok(OutputHeadLinOutcome {
        outcome,
        proof_bytes,
    })
}

/// Verifier entry. It rebuilds the trace-derived public claims, checks the
/// pinned activation and weight roots, and runs the merged verifier. All
/// private rows stay on the prover side.
pub fn verify_output_head_lin_request(
    public: &ModelCommitmentPublic,
    plan: &TracePlan,
    trace: &SharedTrace,
    expected_tokens: &[u32],
    shape: LinShape,
    setup: &OutputHeadLinSetup,
    reader: &IndexReader,
    proof: &OutputHeadLinProof,
) -> Result<OutputHeadLinOutcome> {
    let proof_bytes = proof.encoded_len()?;
    verify_merged(
        plan,
        trace,
        expected_tokens,
        shape,
        setup,
        public,
        reader,
        proof,
        proof_bytes,
    )
}

/// Verifier entry over the canonical request encoding.
#[allow(clippy::too_many_arguments)]
pub fn verify_output_head_lin_request_bytes(
    public: &ModelCommitmentPublic,
    plan: &TracePlan,
    trace: &SharedTrace,
    expected_tokens: &[u32],
    shape: LinShape,
    setup: &OutputHeadLinSetup,
    reader: &IndexReader,
    bytes: &[u8],
) -> Result<OutputHeadLinOutcome> {
    let (_request, claims) = verify_trace(public, plan, trace, expected_tokens, shape, setup)?;
    let masks = if setup.masks.is_empty() {
        vec![Vec::new(); claims.weights.len()]
    } else {
        setup.masks.clone()
    };
    let claims = OutputHeadLinClaims { masks, ..claims };
    let activation_root = setup
        .expected_activation_root
        .ok_or(Error::StatementMismatch)?;
    let expected_roots = setup.expected_roots.clone().ok_or(Error::StatementMismatch)?;
    let merged = merged_plan(
        shape,
        setup,
        reader,
        &claims,
        activation_root,
        Some(expected_roots),
    )?;
    let proof = OutputHeadLinProof::decode(&merged, bytes)?;
    if proof.session.first.activation.root != activation_root {
        return Err(Error::StatementMismatch);
    }
    let outcome = verify_hidden_lin(&merged, reader, &claims.output_values, &proof.session)?;
    Ok(OutputHeadLinOutcome {
        outcome,
        proof_bytes: bytes.len(),
    })
}


}

#[cfg(test)]
mod tests {
    use super::diagnostic::*;
    use super::*;
    use crate::proof::plan::{PreviousObjectLayout, PreviousObjectOwner, PreviousObjectShape};
    use crate::proof::output_closure::{ClosureMaskPlan, ClosurePlan};
    use crate::proof::psc::{hp_entries, psc_variables};
    use std::cell::RefCell;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use zkie_core::common::field::{PrimeCharacteristicRing, PrimeField64};
    use zkie_engine::pcs::request::code::{build_code_plan, CodeParams};
    use zkie_engine::pcs::request::field::{Fp, E_DEGREE};
    use zkie_engine::pcs::request::foldeq::OmegaShapePlan;
    use zkie_engine::pcs::request::gu::{GuOracleShape, GuShape};
    use zkie_engine::pcs::request::runtime_tree::{MaskTreePlan, RuntimeTreePart, RuntimeTreePlan};
    use zkie_engine::pcs::request::table::{
        commit_table_streaming, IndexShape, StreamingTable, TableDistances, VecTableSource,
    };

    use crate::matmul::MatMulOp;
    use crate::proof::hidden_send_session::{
        DotMapPlan, HiddenProximityPlan, HiddenSendPlan, HiddenTableProver, IndexMaskPlan,
        MaskEntryAllocation, WShape,
    };
    use crate::proof::model_commitment::{commit_assets, AssetStore, ModelAssetSpec};
    use crate::proof::plan::{config, PlanBuilder};
    use crate::proof::session_baseline::TrustedModel;
    use crate::registry::PureForward;

    const TOY_POSITIONS: usize = 16;
    const TOY_HIDDEN: usize = 4;
    const TOY_COLUMNS: usize = 16;
    const TOY_K_LO: u32 = 2;
    const TOY_TABLE_ROWS: u64 = 4;
    const TOY_VALID: usize = 6;
    const TOY_ASSET: &str = "toy_head_i32.bin";
    const GROUP_CODE_LOG_N: u32 = 10;
    /// The toy fixture carries i32 extreme activations, so its output bound is
    /// larger than the production output-head bound.
    const TOY_RANGE_BITS: u32 = 40;
    const TOY_SLACK_BITS: u32 = 42;
    const GROUP_CODE_ELL: u64 = 32;
    const GROUP_CODE_R: u64 = 8;
    const TABLE_CODE_LOG_N: u32 = 5;
    const TABLE_CODE_ELL: u64 = 24;
    const TABLE_CODE_R: u64 = 1;
    const SALT: usize = zkie_engine::pcs::request::params::LEAF_SALT_MIN_BYTES;
    const ORACLE_BASE: u64 = 20;
    const SEND_GROUP: u64 = 7;
    const GU_ORACLE: u64 = 10;
    const OMEGA_ORACLE: u64 = 11;
    const GU_GROUP: u64 = 5;
    const W_TREE_NUMBER: u64 = 100;
    const GU_TREE_NUMBER: u64 = 300;
    const OMEGA_TREE_NUMBER: u64 = 301;
    const SHARED_MASK_TREE: u64 = 200;
    const OWN_MASK_TREE: u64 = 201;
    const INDEX_MASK_ORACLE: u64 = 40;
    const INDEX_MASK_TREE: u64 = 202;
    const INDEX_MASK_RUN: Digest = Digest([0x65; 32]);
    const SLACK_ORACLE_BASE: u64 = 100;
    const SLACK_TREE_NUMBER: u64 = 400;
    const SLACK_RUN: Digest = Digest([0x68; 32]);
    const BITS_ORACLE_BASE: u64 = 120;
    const BITS_TREE_NUMBER: u64 = 420;
    const BITS_RUN: Digest = Digest([0x69; 32]);
    const MAP_ORACLE_BASE: u64 = 160;
    const MAP_TREE_NUMBER: u64 = 460;
    const MAP_RUN: Digest = Digest([0x6B; 32]);
    const CLOSURE_MASK_ORACLE: u64 = 140;
    const CLOSURE_MASK_TREE: u64 = 500;
    const CLOSURE_MASK_RUN: Digest = Digest([0x6A; 32]);
    const ACTIVATION_CONFIGURATION: Digest = Digest([0x31; 32]);
    const W_RUN: Digest = Digest([0x41; 32]);
    const W_CONFIGURATION: Digest = Digest([0x51; 32]);

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "zkie-output-head-merged-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn build_plan(activation_steps: usize) -> TracePlan {
        let mut builder = PlanBuilder::new();
        let producer = builder.step(
            OpCode::Add,
            Vec::new(),
            vec![PlanSource::Constant(PlanConstant::Zeros {
                count: TOY_POSITIONS * TOY_HIDDEN,
            })],
            1,
        );
        let mut activation = producer;
        for _ in 0..activation_steps.max(1) {
            activation = builder.step(
                OpCode::ProfileLayerNorm,
                config(vec![
                    TOY_POSITIONS as i64,
                    TOY_HIDDEN as i64,
                    TOY_HIDDEN as i64,
                    0,
                    0,
                    0,
                    0,
                ]),
                vec![
                    PlanSource::prev(activation, 0),
                    PlanSource::asset(TOY_ASSET),
                    PlanSource::asset(TOY_ASSET),
                    PlanSource::asset(TOY_ASSET),
                ],
                1,
            );
        }
        let logits = builder.step(
            OpCode::MatMul,
            config(vec![
                TOY_POSITIONS as i64,
                TOY_COLUMNS as i64,
                TOY_HIDDEN as i64,
                0,
                0,
            ]),
            vec![
                PlanSource::prev(activation, 0),
                PlanSource::asset(TOY_ASSET),
            ],
            1,
        );
        let masked = builder.step(
            OpCode::Add,
            Vec::new(),
            vec![
                PlanSource::prev(logits, 0),
                PlanSource::Constant(PlanConstant::VocabMask {
                    rows: TOY_POSITIONS,
                    vocab: TOY_COLUMNS,
                    valid: TOY_VALID,
                    masked: -(1i64 << 41),
                }),
            ],
            1,
        );
        for row in 0..TOY_POSITIONS {
            let argmax = builder.step(
                OpCode::Argmax,
                Vec::new(),
                vec![PlanSource::prev_slice(
                    masked,
                    0,
                    PlanLayout::rows(TOY_COLUMNS, row, 1),
                )],
                1,
            );
            builder.argmax_step(argmax);
        }
        let mut plan = builder.finish();
        plan.model_version = 1;
        plan.declare_previous_object_for_activation(activation, TOY_POSITIONS, TOY_HIDDEN)
            .unwrap();
        plan
    }

    #[test]
    fn previous_object_is_plan_derived_and_tampering_is_rejected() {
        let plan = build_plan(1);
        let activation = plan
            .steps
            .iter()
            .position(|step| step.opcode == OpCode::ProfileLayerNorm)
            .unwrap();
        let object = plan.previous_object.clone().unwrap();
        let derived = PreviousObjectPlan::derive_for_activation(
            &plan,
            activation,
            TOY_POSITIONS,
            TOY_HIDDEN,
        )
        .unwrap();
        assert_eq!(object, derived);
        assert_eq!(object.consumer_step, activation);
        assert_eq!(object.consumer_port, 0);
        assert_eq!(object.shape.positions, TOY_POSITIONS);
        assert_eq!(object.shape.hidden, TOY_HIDDEN);
        assert_eq!(object.layout.rows, TOY_POSITIONS);
        assert_eq!(object.layout.columns, TOY_HIDDEN);
        assert_eq!(object.layout.row_start, 0);
        assert_eq!(object.layout.column_start, 0);
        let (producer_step, producer_output) = match &plan.steps[activation].inputs[0] {
            PlanSource::Prev {
                step,
                output,
                layout: PlanLayout::Flat,
            } => (*step, *output),
            _ => panic!("the activation must read a flat previous step"),
        };
        assert_eq!(object.owner.step, producer_step);
        assert_eq!(object.producer_output, producer_output);

        let wrong_owner = PreviousObjectPlan {
            owner: PreviousObjectOwner {
                id: Digest([0xA1; 32]),
                ..object.owner.clone()
            },
            ..object.clone()
        };
        assert!(wrong_owner
            .validate_for_activation(&plan, activation, TOY_POSITIONS, TOY_HIDDEN)
            .is_err());
        let wrong_shape = PreviousObjectPlan {
            shape: PreviousObjectShape {
                positions: TOY_POSITIONS + 1,
                ..object.shape.clone()
            },
            ..object.clone()
        };
        assert!(wrong_shape
            .validate_for_activation(&plan, activation, TOY_POSITIONS, TOY_HIDDEN)
            .is_err());
        let wrong_layout = PreviousObjectPlan {
            layout: PreviousObjectLayout {
                row_start: 1,
                ..object.layout.clone()
            },
            ..object.clone()
        };
        assert!(wrong_layout
            .validate_for_activation(&plan, activation, TOY_POSITIONS, TOY_HIDDEN)
            .is_err());
        let wrong_object = PreviousObjectPlan {
            id: Digest([0xA2; 32]),
            ..object.clone()
        };
        assert!(wrong_object
            .validate_for_activation(&plan, activation, TOY_POSITIONS, TOY_HIDDEN)
            .is_err());
        let wrong_consumer = PreviousObjectPlan {
            consumer_port: 1,
            ..object.clone()
        };
        assert!(wrong_consumer
            .validate_for_activation(&plan, activation, TOY_POSITIONS, TOY_HIDDEN)
            .is_err());

        let mut missing = plan.clone();
        missing.previous_object = None;
        assert!(select_output_head_step_for(
            &missing,
            TOY_POSITIONS,
            TOY_HIDDEN,
            TOY_COLUMNS,
            TOY_ASSET,
        )
        .is_err());
        let mut swapped = plan.clone();
        swapped.previous_object.as_mut().unwrap().owner.step += 1;
        assert!(select_output_head_step_for(
            &swapped,
            TOY_POSITIONS,
            TOY_HIDDEN,
            TOY_COLUMNS,
            TOY_ASSET,
        )
        .is_err());
    }

    fn group_code() -> CodeParams {
        build_code_plan(GROUP_CODE_LOG_N, GROUP_CODE_ELL, GROUP_CODE_R).unwrap()
    }

    fn table_code() -> CodeParams {
        build_code_plan(TABLE_CODE_LOG_N, TABLE_CODE_ELL, TABLE_CODE_R).unwrap()
    }

    fn table_shape() -> IndexShape {
        IndexShape {
            code: table_code(),
            v: 2,
            answers: 1,
            salt_bytes: SALT,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        }
    }

    fn tree_plan(tree_number: u64, leaf_class: u32, main_rows: u64) -> RuntimeTreePlan {
        RuntimeTreePlan {
            tree_number,
            leaf_class,
            log_n: GROUP_CODE_LOG_N,
            log_n_zk: GROUP_CODE_LOG_N,
            parts: vec![RuntimeTreePart {
                main_symbols: main_rows * E_DEGREE as u64,
                mask_symbols: 0,
            }],
            salt_bytes: SALT,
            run: Digest([0x61; 32]),
            configuration: Digest([0x62; 32]),
        }
    }

    fn w_shape() -> WShape {
        WShape::new(
            TOY_COLUMNS as u64,
            group_code(),
            1,
            TOY_POSITIONS as u32,
            SALT,
            W_TREE_NUMBER,
            1,
            W_RUN,
            W_CONFIGURATION,
        )
        .unwrap()
    }

    fn gu_shape() -> GuShape {
        GuShape {
            oracle: GU_ORACLE,
            group: GU_GROUP,
            tree: tree_plan(GU_TREE_NUMBER, 2, 1),
            part: 0,
            rows_high: true,
            u: GuOracleShape {
                code: group_code(),
                main_rows: 1,
                main_deg: E_DEGREE as u32,
                mask_rows: 0,
                mask_code: None,
                answers: 1,
            },
            index: table_shape(),
        }
    }

    fn omega_shape() -> OmegaShapePlan {
        OmegaShapePlan {
            oracle: OMEGA_ORACLE,
            group: GU_GROUP,
            tree: tree_plan(OMEGA_TREE_NUMBER, 2, 1),
            u: GuOracleShape {
                code: group_code(),
                main_rows: 1,
                main_deg: E_DEGREE as u32,
                mask_rows: 0,
                mask_code: None,
                answers: 1,
            },
            rows_high: true,
            matrix_rows: TOY_HIDDEN as u64,
            k_lo: TOY_K_LO,
        }
    }

    fn mask_tree(tree_number: u64) -> MaskTreePlan {
        MaskTreePlan {
            tree_number,
            leaf_class: 3,
            log_n: GROUP_CODE_LOG_N,
            mask_rows: 1,
            salt_bytes: SALT,
            run: Digest([0x63; 32]),
            configuration: Digest([0x64; 32]),
        }
    }

    fn index_mask_plan() -> IndexMaskPlan {
        IndexMaskPlan {
            oracle: INDEX_MASK_ORACLE,
            group: SEND_GROUP,
            mask_code: group_code(),
            mask_rows: 3,
            answers: 1,
            openings: 1,
            tree: RuntimeTreePlan {
                tree_number: INDEX_MASK_TREE,
                leaf_class: 1,
                log_n: GROUP_CODE_LOG_N,
                log_n_zk: GROUP_CODE_LOG_N,
                parts: vec![RuntimeTreePart {
                    main_symbols: 0,
                    mask_symbols: 3 * E_DEGREE as u64,
                }],
                salt_bytes: SALT,
                run: INDEX_MASK_RUN,
                configuration: W_CONFIGURATION,
            },
            entries: MaskEntryAllocation::for_variables(
                TOY_HIDDEN.trailing_zeros() as usize,
                TOY_POSITIONS as u32,
            )
            .unwrap(),
        }
    }

    fn dot_plan() -> DotMapPlan {
        DotMapPlan {
            oracle: 12,
            group: SEND_GROUP,
            shape: WShape::new(
                (TOY_POSITIONS * TOY_HIDDEN) as u64,
                group_code(),
                1,
                TOY_POSITIONS as u32,
                SALT,
                302,
                2,
                Digest([0x67; 32]),
                W_CONFIGURATION,
            )
            .unwrap(),
            hidden: TOY_HIDDEN as u32,
        }
    }

    fn proximity() -> HiddenProximityPlan {
        HiddenProximityPlan {
            gu: gu_shape(),
            omega: omega_shape(),
            code: group_code(),
            mask_code: group_code(),
            shared_mask_tree: mask_tree(SHARED_MASK_TREE),
            own_mask_tree: MaskTreePlan {
                mask_rows: 32,
                ..mask_tree(OWN_MASK_TREE)
            },
            shared_fresh: 50,
            own_fresh: 1000,
            column_draws: 1,
            column_budget: 1,
            column_work_bits: 0,
            shared_base_positions: 2,
            own_base_positions: 2,
            shared_stages: Vec::new(),
            own_stages: Vec::new(),
            planner_checked: false,
        }
    }

    fn table_values() -> Vec<i32> {
        (0..(TOY_HIDDEN * TOY_COLUMNS))
            .map(|index| ((index as i32 * 37) % 23) - 11)
            .collect()
    }

    fn table_messages(values: &[i32]) -> Vec<Vec<Fp>> {
        let rows = TOY_TABLE_ROWS as usize;
        let ell = TABLE_CODE_ELL as usize;
        let mut messages = vec![vec![Fp::ZERO; ell]; rows];
        for row in 0..rows {
            for y in 0..TOY_HIDDEN {
                for low in 0..(1usize << TOY_K_LO) {
                    let column = low + (row << TOY_K_LO);
                    let entry = low + (y << TOY_K_LO);
                    messages[row][entry] = fp_from_signed_word(
                        values[y * TOY_COLUMNS + column] as i64 as u64,
                    );
                }
            }
        }
        messages
    }

    fn table_tails() -> Vec<Vec<Fp>> {
        vec![vec![Fp::ZERO; TABLE_CODE_R as usize]; TOY_TABLE_ROWS as usize]
    }

    struct Table {
        table: StreamingTable,
        source: RefCell<VecTableSource>,
        reader: IndexReader,
    }

    impl zkie_engine::pcs::request::table::TableRowSource for Table {
        fn rows(&self) -> u64 {
            self.source.borrow().rows()
        }

        fn row(&mut self, index: u64, message: &mut [Fp], tail: &mut [Fp]) -> Result<()> {
            self.source.borrow_mut().row(index, message, tail)
        }
    }

    impl HiddenTableProver for Table {
        fn reader(&self) -> &IndexReader {
            &self.reader
        }

        fn open_columns(
            &mut self,
            positions: &[u64],
        ) -> Result<Vec<zkie_engine::pcs::request::table::ColumnOpening>> {
            let mut source = self.source.borrow_mut();
            self.table.open_columns(&mut *source, positions)
        }
    }

    fn encoded_table(values: &[i32]) -> Table {
        let messages = table_messages(values);
        let tails = table_tails();
        let mut source = VecTableSource::new(messages, tails).unwrap();
        let salts: Vec<Vec<u8>> = (0..table_code().n())
            .map(|position| vec![(position as u8) ^ 0x5A; SALT])
            .collect();
        let table = commit_table_streaming(
            &mut source,
            table_shape(),
            Digest([0x21; 32]),
            &salts,
            64,
            u64::MAX,
        )
        .unwrap();
        let reader = table.reader.clone();
        Table {
            table,
            source: RefCell::new(source),
            reader,
        }
    }

    struct Fixture {
        root: PathBuf,
        public: ModelCommitmentPublic,
        specs: Vec<ModelAssetSpec>,
        anchor_id: Digest,
        model_id: Digest,
        parameters_digest: Digest,
        plan: TracePlan,
        trace: SharedTrace,
        outputs: Vec<Vec<Vec<i64>>>,
        shape: LinShape,
        table: Table,
        setup: OutputHeadLinSetup,
        activation_salts: Vec<Vec<u8>>,
        proof: OutputHeadLinProof,
        witness: HiddenLinWitness,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn fixture() -> Fixture {
        let root = temp_dir();
        let values = table_values();
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        fs::write(root.join(TOY_ASSET), bytes).unwrap();
        let specs = vec![ModelAssetSpec {
            asset_id: 0,
            path: TOY_ASSET.into(),
            element_count: (TOY_HIDDEN * TOY_COLUMNS) as u64,
            row_width: TOY_COLUMNS,
            encoding: "i32-le".into(),
        }];
        let commitment =
            commit_assets(&root, Digest([1; 32]), Digest([2; 32]), &specs).unwrap();
        let public = ModelCommitmentPublic::from_commitment(&commitment);
        let opening = AssetStore::open(&commitment)
            .unwrap()
            .open_all()
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let anchor_id = public.anchor.anchor_id;
        let model_id = Digest([1; 32]);
        let parameters_digest = Digest([2; 32]);
        let shape = LinShape::new(
            TOY_HIDDEN as u64,
            TOY_K_LO,
            TOY_TABLE_ROWS,
            TOY_COLUMNS as u64,
        )
        .unwrap();

        let pool = [
            i32::MIN,
            i32::MAX,
            -1,
            0,
            1,
            123_456_789,
            -987_654_321,
            42,
        ];
        let activation: Vec<i64> = (0..(TOY_POSITIONS * TOY_HIDDEN))
            .map(|index| pool[index % pool.len()] as i64)
            .collect();
        let table: Vec<i64> = values.iter().map(|value| *value as i64).collect();
        let output = MatMulOp {
            m: TOY_POSITIONS,
            n: TOY_COLUMNS,
            k: TOY_HIDDEN,
            shift: 0,
        }
        .forward_values(&[activation.clone(), table])
        .unwrap()
        .outputs
        .into_iter()
        .next()
        .unwrap();

        let plan = build_plan(1);
        let trace = SharedTrace {
            tokens: (1..=TOY_POSITIONS as u32).collect(),
            assets: vec![opening],
            steps: Vec::new(),
            argmax: (0..TOY_POSITIONS as u32).collect(),
            output: Vec::new(),
        };
        let outputs = vec![
            vec![activation.clone()],
            vec![activation.clone()],
            vec![output.clone()],
        ];

        let encoded = encoded_table(&values);
        let send = HiddenSendPlan {
            activation: ActivationShape::new(TOY_POSITIONS as u32, TOY_HIDDEN as u32, SALT).unwrap(),
            activation_configuration: ACTIVATION_CONFIGURATION,
            activation_root: Digest([0x71; 32]),
            activation_source: crate::proof::hidden_send_session::ActivationSource::Commitment,
            table_shape: encoded.reader.shape,
            table_identity: reader_table_identity(&encoded.reader).unwrap(),
            matrix_rows: TOY_HIDDEN as u64,
            k_lo: TOY_K_LO,
            columns: TOY_COLUMNS as u64,
            group: SEND_GROUP,
            oracle_base: ORACLE_BASE,
            openings_per_oracle: 2,
            w: w_shape(),
            index_mask: None,
            dot: Some(dot_plan()),
            expected_dot_root: None,
            expected_roots: None,
        };
        let salts: Vec<Vec<u8>> = (0..TOY_POSITIONS)
            .map(|position| vec![(position as u8).wrapping_mul(11) ^ 0x3C; SALT])
            .collect();
        let closure_width = 2 * (TOY_RANGE_BITS + 1) + TOY_SLACK_BITS;
        let bits_columns = (TOY_COLUMNS as u64)
            .checked_mul(closure_width as u64)
            .unwrap()
            .next_power_of_two();
        let boolean_variables = bits_columns.trailing_zeros() as usize;
        let closure_variables = psc_variables(boolean_variables).unwrap();
        let closure_mask = ClosureMaskPlan {
            oracle: CLOSURE_MASK_ORACLE,
            group: SEND_GROUP,
            mask_code: group_code(),
            mask_rows: 16,
            answers: 1,
            openings: 1,
            tree: RuntimeTreePlan {
                tree_number: CLOSURE_MASK_TREE,
                leaf_class: 1,
                log_n: GROUP_CODE_LOG_N,
                log_n_zk: GROUP_CODE_LOG_N,
                parts: vec![RuntimeTreePart {
                    main_symbols: 0,
                    mask_symbols: 16 * E_DEGREE as u64,
                }],
                salt_bytes: SALT,
                run: CLOSURE_MASK_RUN,
                configuration: W_CONFIGURATION,
            },
            positions: TOY_POSITIONS as u32,
            hiding: hp_entries(&closure_variables).unwrap(),
            shown: 5,
        };
        let closure_setup = OutputClosureSetup {
            slack: WShape::new(
                TOY_COLUMNS as u64,
                group_code(),
                1,
                TOY_POSITIONS as u32,
                SALT,
                SLACK_TREE_NUMBER,
                1,
                SLACK_RUN,
                W_CONFIGURATION,
            )
            .unwrap(),
            slack_oracle_base: SLACK_ORACLE_BASE,
            slack_openings: 1,
            bits: WShape::new(
                bits_columns,
                group_code(),
                1,
                TOY_POSITIONS as u32,
                SALT,
                BITS_TREE_NUMBER,
                1,
                BITS_RUN,
                W_CONFIGURATION,
            )
            .unwrap(),
            bits_oracle_base: BITS_ORACLE_BASE,
            bits_openings: 1,
            map: WShape::new(
                bits_columns,
                group_code(),
                1,
                TOY_POSITIONS as u32,
                SALT,
                MAP_TREE_NUMBER,
                1,
                MAP_RUN,
                W_CONFIGURATION,
            )
            .unwrap(),
            map_oracle_base: MAP_ORACLE_BASE,
            map_openings: 1,
            mask: closure_mask,
        };
        let setup = OutputHeadLinSetup {
            asset_path: TOY_ASSET.into(),
            send,
            proximity: proximity(),
            index_mask: Some(index_mask_plan()),
            activation_configuration: ACTIVATION_CONFIGURATION,
            activation_salts: salts.clone(),
            seed: Digest([0x72; 32]),
            masks: Vec::new(),
            expected_activation_root: None,
            expected_roots: None,
            closure: Some(closure_setup),
        };
        let mut encoded = encoded;
        let request = prepare_output_head_trace_for(
            &public,
            &plan,
            &trace,
            &outputs,
            shape,
            TOY_POSITIONS,
            TOY_HIDDEN,
            TOY_COLUMNS,
            TOY_ASSET,
        )
        .unwrap();
        let (proof, witness) =
            prove_output_head_lin_prepared(&request, shape, &setup, &mut encoded).unwrap();
        Fixture {
            root,
            public,
            specs,
            anchor_id,
            model_id,
            parameters_digest,
            plan,
            trace,
            outputs,
            shape,
            table: encoded,
            setup,
            activation_salts: salts,
            proof,
            witness,
        }
    }

    fn trusted(fixture: &Fixture) -> TrustedModel<'_> {
        TrustedModel {
            anchor_id: fixture.anchor_id,
            model_id: fixture.model_id,
            parameters_digest: fixture.parameters_digest,
            asset_specs: &fixture.specs,
        }
    }

    fn verifier_setup(fixture: &Fixture) -> OutputHeadLinSetup {
        let mut setup = fixture.setup.clone();
        setup.expected_activation_root = Some(fixture.proof.session.first.activation.root);
        setup.expected_roots = Some(fixture.witness.send.roots.clone());
        setup
    }

    fn trace_request(fixture: &Fixture, outputs: &[Vec<Vec<i64>>]) -> Result<OutputHeadTrace> {
        prepare_output_head_trace_for(
            &fixture.public,
            &fixture.plan,
            &fixture.trace,
            outputs,
            fixture.shape,
            TOY_POSITIONS,
            TOY_HIDDEN,
            TOY_COLUMNS,
            TOY_ASSET,
        )
    }

    fn verify(
        fixture: &Fixture,
        setup: &OutputHeadLinSetup,
        reader: &IndexReader,
        proof: &OutputHeadLinProof,
    ) -> Result<OutputHeadLinOutcome> {
        let request = trace_request(fixture, &fixture.outputs)?;
        verify_output_head_lin_prepared(&request, fixture.shape, setup, reader, proof)
    }

    #[test]
    fn merged_request_accepts_and_round_trips() {
        let fixture = fixture();
        let setup = verifier_setup(&fixture);
        let outcome = verify(&fixture, &setup, &fixture.table.reader, &fixture.proof).unwrap();
        assert_eq!(outcome.outcome.eval_rows.len(), TOY_POSITIONS);
        assert!(outcome
            .outcome
            .shared_base
            .single()
            .expect("the fixture ends in BASE")
            .fold_main);
        assert!(outcome
            .outcome
            .own_base
            .single()
            .expect("the fixture ends in BASE")
            .fold_main);
        let bytes = fixture.proof.encode().unwrap();
        let merged = merged_plan(
            fixture.shape,
            &setup,
            &fixture.table.reader,
            &output_head_claims(
                fixture.shape,
                &OutputHeadInputs {
                    activation: fixture.outputs[1][0].clone(),
                    output: fixture.outputs[2][0].clone(),
                },
                &[],
            )
            .unwrap(),
            setup.expected_activation_root.unwrap(),
            setup.expected_roots.clone(),
        )
        .unwrap();
        let decoded = OutputHeadLinProof::decode(&merged, &bytes).unwrap();
        assert_eq!(decoded, fixture.proof);
        let request = trace_request(&fixture, &fixture.outputs).unwrap();
        let outcome = verify_output_head_lin_prepared(
            &request,
            fixture.shape,
            &setup,
            &fixture.table.reader,
            &decoded,
        )
        .unwrap();
        assert_eq!(outcome.proof_bytes, bytes.len());
        let mut trailing = bytes;
        trailing.push(0);
        assert!(OutputHeadLinProof::decode(&merged, &trailing).is_err());
        assert_eq!(trusted(&fixture).model_id, Digest([1; 32]));
        assert_eq!(fixture.activation_salts.len(), TOY_POSITIONS);
    }

    #[test]
    fn old_or_unknown_envelope_version_is_rejected() {
        let fixture = fixture();
        let setup = verifier_setup(&fixture);
        let merged = merged_plan(
            fixture.shape,
            &setup,
            &fixture.table.reader,
            &output_head_claims(
                fixture.shape,
                &OutputHeadInputs {
                    activation: fixture.outputs[1][0].clone(),
                    output: fixture.outputs[2][0].clone(),
                },
                &[],
            )
            .unwrap(),
            setup.expected_activation_root.unwrap(),
            setup.expected_roots.clone(),
        )
        .unwrap();
        let session = fixture.proof.session.encode().unwrap();
        // A proof from the previous envelope version is unsupported, not a
        // silent fallback to the single-stage tail.
        let mut old = CanonicalWriter::new();
        old.put_u64(4);
        old.put_bytes(&session);
        assert_eq!(
            OutputHeadLinProof::decode(&merged, &old.into_bytes()).unwrap_err(),
            Error::UnsupportedVersion
        );
        // A newer unknown version cannot decode either.
        let mut newer = CanonicalWriter::new();
        newer.put_u64(OUTPUT_HEAD_PROOF_VERSION + 1);
        newer.put_bytes(&session);
        assert_eq!(
            OutputHeadLinProof::decode(&merged, &newer.into_bytes()).unwrap_err(),
            Error::InvalidEncoding
        );
        // The current envelope still decodes.
        let bytes = fixture.proof.encode().unwrap();
        assert_eq!(OutputHeadLinProof::decode(&merged, &bytes).unwrap(), fixture.proof);
    }

    #[test]
    fn changed_activation_or_output_is_rejected() {
        let mut fixture = fixture();
        let setup = verifier_setup(&fixture);
        let original_zero = fixture.outputs[1][0][0];
        fixture.outputs[1][0][0] = original_zero.wrapping_add(1);
        let outputs = fixture.outputs.clone();
        let request = trace_request(&fixture, &outputs).unwrap();
        let claims = output_head_claims(fixture.shape, &request.inputs, &[]).unwrap();
        let mut plan = merged_plan(
            fixture.shape,
            &setup,
            &fixture.table.reader,
            &claims,
            setup.expected_activation_root.unwrap(),
            setup.expected_roots.clone(),
        )
        .unwrap();
        plan.send.expected_roots = setup.expected_roots.clone();
        assert!(verify_hidden_lin(
            &plan,
            &fixture.table.reader,
            &claims.output_values,
            &fixture.proof.session,
        )
        .is_err());
        fixture.outputs[1][0][0] = original_zero;

        fixture.outputs[2][0][0] = fixture.outputs[2][0][0].wrapping_add(1);
        let outputs = fixture.outputs.clone();
        let request = trace_request(&fixture, &outputs).unwrap();
        let claims = output_head_claims(fixture.shape, &request.inputs, &[]).unwrap();
        assert!(verify_hidden_lin(
            &merged_plan(
                fixture.shape,
                &setup,
                &fixture.table.reader,
                &claims,
                setup.expected_activation_root.unwrap(),
                setup.expected_roots.clone(),
            )
            .unwrap(),
            &fixture.table.reader,
            &claims.output_values,
            &fixture.proof.session,
        )
        .is_err());
    }

    #[test]
    fn changed_roots_or_plan_are_rejected() {
        let fixture = fixture();
        let mut setup = verifier_setup(&fixture);
        setup.expected_activation_root = Some(Digest([0x77; 32]));
        assert!(verify(&fixture, &setup, &fixture.table.reader, &fixture.proof).is_err());
        let mut setup = verifier_setup(&fixture);
        let mut roots = setup.expected_roots.clone().unwrap();
        roots[0] = Digest([0x66; 32]);
        setup.expected_roots = Some(roots);
        assert!(verify(&fixture, &setup, &fixture.table.reader, &fixture.proof).is_err());
        let mut reader = fixture.table.reader.clone();
        reader.root.0[0] ^= 1;
        assert!(verify(
            &fixture,
            &verifier_setup(&fixture),
            &reader,
            &fixture.proof
        )
        .is_err());
        let mut changed = fixture.plan.clone();
        let matmul = changed
            .steps
            .iter()
            .position(|step| step.opcode == OpCode::MatMul)
            .unwrap();
        changed.steps[matmul].inputs[1] = PlanSource::asset("missing_head_i32.bin");
        let _setup = verifier_setup(&fixture);
        let request = prepare_output_head_trace_for(
            &fixture.public,
            &changed,
            &fixture.trace,
            &fixture.outputs,
            fixture.shape,
            TOY_POSITIONS,
            TOY_HIDDEN,
            TOY_COLUMNS,
            TOY_ASSET,
        );
        assert!(request.is_err());
    }

    #[test]
    fn changed_or_missing_proof_stage_is_rejected() {
        let fixture = fixture();
        let setup = verifier_setup(&fixture);
        let mut tampered = fixture.proof.clone();
        tampered.session.second.seal.0[0] ^= 1;
        assert!(verify(&fixture, &setup, &fixture.table.reader, &tampered).is_err());
        let mut tampered = fixture.proof.clone();
        tampered.session.shared.gu_root.0[0] ^= 1;
        assert!(verify(&fixture, &setup, &fixture.table.reader, &tampered).is_err());
        let mut tampered = fixture.proof.clone();
        tampered.session.second.fold_rounds.clear();
        assert!(verify(&fixture, &setup, &fixture.table.reader, &tampered).is_err());
        let mut tampered = fixture.proof.clone();
        match &mut tampered.session.second.tail {
            crate::proof::hidden_send_session::ProximityTail::Single { base_rows, .. } => {
                base_rows.base.lambda = base_rows.base.lambda.add(&Fp3::ONE);
            }
            crate::proof::hidden_send_session::ProximityTail::Stages(_) => {
                panic!("the fixture has no stages")
            }
        }
        assert!(verify(&fixture, &setup, &fixture.table.reader, &tampered).is_err());
        let bytes = fixture.proof.encode().unwrap();
        let merged = merged_plan(
            fixture.shape,
            &setup,
            &fixture.table.reader,
            &output_head_claims(
                fixture.shape,
                &OutputHeadInputs {
                    activation: fixture.outputs[1][0].clone(),
                    output: fixture.outputs[2][0].clone(),
                },
                &[],
            )
            .unwrap(),
            setup.expected_activation_root.unwrap(),
            setup.expected_roots.clone(),
        )
        .unwrap();
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(OutputHeadLinProof::decode(&merged, &trailing).is_err());
        let mut truncated = bytes;
        truncated.truncate(truncated.len() - 3);
        assert!(OutputHeadLinProof::decode(&merged, &truncated).is_err());
    }
    fn masked_row_argmax(row: &[i64]) -> u32 {
        let mut best = 0usize;
        let mut best_value = i64::MIN;
        for (column, value) in row.iter().enumerate() {
            let masked = if column >= TOY_VALID {
                value - (1i64 << 41)
            } else {
                *value
            };
            if column == 0 || masked > best_value {
                best = column;
                best_value = masked;
            }
        }
        best as u32
    }

    fn toy_expected_argmax(output: &[i64]) -> Vec<u32> {
        (0..TOY_POSITIONS)
            .map(|position| {
                let row = &output[position * TOY_COLUMNS..(position + 1) * TOY_COLUMNS];
                masked_row_argmax(row)
            })
            .collect()
    }

    fn formal_inputs(
        fixture: &Fixture,
    ) -> (Vec<Vec<i64>>, OutputHeadOutputPlan, ActivationCommitment) {
        let rows: Vec<Vec<i64>> = (0..TOY_POSITIONS)
            .map(|position| {
                fixture.outputs[1][0][position * TOY_HIDDEN..(position + 1) * TOY_HIDDEN].to_vec()
            })
            .collect();
        let expected = toy_expected_argmax(&fixture.outputs[2][0]);
        let output = build_output_head_output_plan_for(
            &fixture.plan,
            TOY_POSITIONS,
            TOY_HIDDEN,
            TOY_COLUMNS,
            TOY_ASSET,
            TOY_RANGE_BITS,
            TOY_SLACK_BITS,
            &expected,
        )
        .unwrap();
        let activation = CommittedActivation::commit_signed(
            ActivationShape::new(TOY_POSITIONS as u32, TOY_HIDDEN as u32, SALT).unwrap(),
            &rows,
            &fixture.activation_salts,
            ACTIVATION_CONFIGURATION,
        )
        .unwrap()
        .commitment()
        .clone();
        (rows, output, activation)
    }

    #[test]
    fn hidden_formal_closure_accepts_and_rejects_tampering() {
        let mut fixture = fixture();
        let (rows, output, activation) = formal_inputs(&fixture);
        let setup = fixture.setup.clone();
        let (proof, witness, closure) = prove_output_head_lin_hidden(
            fixture.shape,
            &setup,
            &output,
            &mut fixture.table,
            &rows,
            &fixture.activation_salts,
        )
        .unwrap();
        assert_eq!(closure.argmax, output.expected_argmax);
        assert_eq!(proof.session.first.activation, activation);
        let expected_output: Vec<Vec<i64>> = (0..TOY_POSITIONS)
            .map(|position| {
                fixture.outputs[2][0]
                    [position * TOY_COLUMNS..(position + 1) * TOY_COLUMNS]
                    .to_vec()
            })
            .collect();
        assert_eq!(witness_output_rows(&witness).unwrap(), expected_output);

        let plan =
            hidden_plan(fixture.shape, &setup, &fixture.table.reader, activation, &output).unwrap();
        let trusted = OutputHeadTrustedInputs {
            activation,
            output: output.clone(),
        };
        let outcome = verify_output_head_lin_hidden(
            fixture.shape,
            &plan,
            &trusted,
            &fixture.table.reader,
            &proof,
        )
        .unwrap();
        assert_eq!(outcome.outcome.eval_rows.len(), TOY_POSITIONS);

        let bytes = proof.encode().unwrap();
        let decoded = verify_output_head_lin_hidden_bytes(
            fixture.shape,
            &plan,
            &trusted,
            &fixture.table.reader,
            &bytes,
        )
        .unwrap();
        assert_eq!(decoded.proof_bytes, bytes.len());

        // Changed activation commitment: the policy pin no longer matches.
        let mut changed = trusted.clone();
        changed.activation.root.0[0] ^= 1;
        assert!(verify_output_head_lin_hidden(
            fixture.shape,
            &plan,
            &changed,
            &fixture.table.reader,
            &proof
        )
        .is_err());

        // Changed Y commitment: the group root is absorbed and opened.
        let mut changed_proof = proof.clone();
        changed_proof.session.first.group.entries[0].root.0[0] ^= 1;
        assert!(verify_output_head_lin_hidden(
            fixture.shape,
            &plan,
            &trusted,
            &fixture.table.reader,
            &changed_proof
        )
        .is_err());

        // Changed Eval target: no longer the committed row's evaluation.
        let mut changed_proof = proof.clone();
        changed_proof.session.first.eval_targets[0] =
            changed_proof.session.first.eval_targets[0].add(&Fp3::ONE);
        assert!(verify_output_head_lin_hidden(
            fixture.shape,
            &plan,
            &trusted,
            &fixture.table.reader,
            &changed_proof
        )
        .is_err());

        // Changed model request position: the relation digest no longer
        // matches the trusted declaration.
        let mut changed_output = output.clone();
        changed_output.step.step += 1;
        let mut changed_trusted = trusted.clone();
        changed_trusted.output = changed_output;
        assert!(verify_output_head_lin_hidden(
            fixture.shape,
            &plan,
            &changed_trusted,
            &fixture.table.reader,
            &proof
        )
        .is_err());

        // Missing declaration: the first statement carries no relation digest.
        let mut missing = plan.clone();
        missing.output_relation = None;
        assert!(verify_output_head_lin_hidden(
            fixture.shape,
            &missing,
            &trusted,
            &fixture.table.reader,
            &proof
        )
        .is_err());

        // Spliced request: an activation commitment of the same shape made
        // with other salts is a different request.
        let other_salts: Vec<Vec<u8>> = (0..TOY_POSITIONS)
            .map(|position| vec![(position as u8).wrapping_mul(7) ^ 0xA5; SALT])
            .collect();
        let other = CommittedActivation::commit_signed(
            ActivationShape::new(TOY_POSITIONS as u32, TOY_HIDDEN as u32, SALT).unwrap(),
            &rows,
            &other_salts,
            ACTIVATION_CONFIGURATION,
        )
        .unwrap()
        .commitment()
        .clone();
        let mut spliced = trusted.clone();
        spliced.activation = other;
        assert!(verify_output_head_lin_hidden(
            fixture.shape,
            &plan,
            &spliced,
            &fixture.table.reader,
            &proof
        )
        .is_err());

        // Opened-row leakage: an activation opening appended to the request
        // bytes is not part of the formal encoding.
        let opening = CommittedActivation::commit_signed(
            ActivationShape::new(TOY_POSITIONS as u32, TOY_HIDDEN as u32, SALT).unwrap(),
            &rows,
            &fixture.activation_salts,
            ACTIVATION_CONFIGURATION,
        )
        .unwrap()
        .open(0)
        .unwrap();
        let mut leaky = bytes.clone();
        leaky.extend_from_slice(&opening.encode());
        assert!(verify_output_head_lin_hidden_bytes(
            fixture.shape,
            &plan,
            &trusted,
            &fixture.table.reader,
            &leaky
        )
        .is_err());

        // Output-closure negatives: every new message and every booleanity
        // transcript value is bound by the final verifier.
        let reject = |candidate: &OutputHeadLinProof| {
            verify_output_head_lin_hidden(
                fixture.shape,
                &plan,
                &trusted,
                &fixture.table.reader,
                candidate,
            )
            .is_err()
        };
        let mut tampered = proof.clone();
        tampered
            .session
            .second
            .closure
            .as_mut()
            .unwrap()
            .psc[0]
            .rounds[0]
            .coefficients[0] = Fp3::from_u64(1);
        assert!(reject(&tampered));
        let mut tampered = proof.clone();
        tampered.session.second.closure.as_mut().unwrap().point_values[0][0] =
            Fp3::from_u64(1);
        assert!(reject(&tampered));
        let mut tampered = proof.clone();
        tampered
            .session
            .first
            .closure
            .as_mut()
            .unwrap()
            .bits
            .group
            .entries[0]
            .root
            .0[0] ^= 1;
        assert!(reject(&tampered));
        let mut tampered = proof.clone();
        tampered
            .session
            .first
            .closure
            .as_mut()
            .unwrap()
            .map
            .group
            .entries[0]
            .root
            .0[0] ^= 1;
        assert!(reject(&tampered));
        let mut tampered = proof.clone();
        tampered
            .session
            .first
            .closure
            .as_mut()
            .unwrap()
            .slack
            .group
            .entries[0]
            .root
            .0[0] ^= 1;
        assert!(reject(&tampered));
        let mut tampered = proof.clone();
        tampered.session.first.closure.as_mut().unwrap().slack_sample.answers[0] =
            Fp3::from_u64(1);
        assert!(reject(&tampered));
        let mut tampered = proof.clone();
        tampered.session.first.closure.as_mut().unwrap().mask.root.0[0] ^= 1;
        assert!(reject(&tampered));
        let mut missing = proof.clone();
        missing.session.first.closure = None;
        assert!(reject(&missing));

        // A different trusted winner changes the closure plan and rows.
        let mut wrong_output = output.clone();
        wrong_output.expected_argmax[0] =
            (wrong_output.expected_argmax[0] + 1) % TOY_VALID as u32;
        let wrong_trusted = OutputHeadTrustedInputs {
            activation,
            output: wrong_output.clone(),
        };
        let wrong_plan = hidden_plan(
            fixture.shape,
            &setup,
            &fixture.table.reader,
            activation,
            &wrong_output,
        )
        .unwrap();
        assert!(verify_output_head_lin_hidden(
            fixture.shape,
            &wrong_plan,
            &wrong_trusted,
            &fixture.table.reader,
            &proof
        )
        .is_err());

        // A wider integer bound changes the packed bit shape and is rejected
        // before any proof runs with the old setup.
        let mut wider = output.clone();
        wider.range_bits += 10;
        assert!(build_output_head_hidden_plan(
            fixture.shape,
            &setup,
            &fixture.table.reader,
            &OutputHeadTrustedInputs {
                activation,
                output: wider,
            },
        )
        .is_err());
    }

    #[test]
    fn arbitrary_activation_is_rejected_by_the_policy_pin() {
        let mut fixture = fixture();
        let (mut rows, output, activation) = formal_inputs(&fixture);
        rows[0][0] = rows[0][0].wrapping_add(1);
        let setup = fixture.setup.clone();
        let arbitrary = CommittedActivation::commit_signed(
            ActivationShape::new(TOY_POSITIONS as u32, TOY_HIDDEN as u32, SALT).unwrap(),
            &rows,
            &fixture.activation_salts,
            ACTIVATION_CONFIGURATION,
        )
        .unwrap();
        let prover_plan = hidden_plan(
            fixture.shape,
            &setup,
            &fixture.table.reader,
            *arbitrary.commitment(),
            &output,
        )
        .unwrap();
        let (session, _) =
            prove_hidden_lin_hidden(&prover_plan, &arbitrary, &mut fixture.table, setup.seed)
                .unwrap();
        let proof = OutputHeadLinProof { session };
        assert_ne!(proof.session.first.activation, activation);
        let plan =
            hidden_plan(fixture.shape, &setup, &fixture.table.reader, activation, &output).unwrap();
        let trusted = OutputHeadTrustedInputs {
            activation,
            output,
        };
        assert!(verify_output_head_lin_hidden(
            fixture.shape,
            &plan,
            &trusted,
            &fixture.table.reader,
            &proof
        )
        .is_err());
    }

    #[test]
    fn forged_mask_and_argmax_tie_are_rejected() {
        let fixture = fixture();
        let (rows, output, _) = formal_inputs(&fixture);
        // A forged tie resolution claims a later index.
        let mut forged = output.clone();
        forged.expected_argmax[0] = (forged.expected_argmax[0] + 1) % TOY_VALID as u32;
        assert!(close_output_relation(&forged, &rows).is_err());
        // A forged validity range makes the declared winner invalid.
        let mut forged = output.clone();
        forged.valid = 1;
        assert!(close_output_relation(&forged, &rows).is_err());
        // A forged mask constant is not a negative mask.
        let mut bad_plan = fixture.plan.clone();
        for step in &mut bad_plan.steps {
            for source in &mut step.inputs {
                if let PlanSource::Constant(PlanConstant::VocabMask { masked, .. }) = source {
                    *masked = 0;
                }
            }
        }
        assert!(build_output_head_output_plan_for(
            &bad_plan,
            TOY_POSITIONS,
            TOY_HIDDEN,
            TOY_COLUMNS,
            TOY_ASSET,
            OUTPUT_HEAD_RANGE_BITS,
            OUTPUT_HEAD_SLACK_BITS,
            &output.expected_argmax,
        )
        .is_err());
        // A missing argmax declaration is rejected before the proof runs.
        let mut missing = fixture.plan.clone();
        missing.argmax_steps.clear();
        assert!(build_output_head_output_plan_for(
            &missing,
            TOY_POSITIONS,
            TOY_HIDDEN,
            TOY_COLUMNS,
            TOY_ASSET,
            OUTPUT_HEAD_RANGE_BITS,
            OUTPUT_HEAD_SLACK_BITS,
            &output.expected_argmax,
        )
        .is_err());
        // A changed expected output is rejected by the full-row closure.
        let mut changed = output.clone();
        changed.expected_argmax[0] = (changed.expected_argmax[0] + 1) % TOY_VALID as u32;
        assert!(close_output_relation(&changed, &rows).is_err());
    }

    #[test]
    fn signed_encoding_and_masked_overflow_are_rejected() {
        let extension = Fp3::new(Fp::from_u64(1), Fp::from_u64(1), Fp::from_u64(0));
        assert!(fp3_to_signed_i64(&extension).is_err());
        let fixture = fixture();
        let (rows, output, _) = formal_inputs(&fixture);
        let unsigned_rows: Vec<Vec<Fp3>> = rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| Fp3::from_base(Fp::from_u64(*value as u64)))
                    .collect()
            })
            .collect();
        let signed = CommittedActivation::commit_signed(
            ActivationShape::new(TOY_POSITIONS as u32, TOY_HIDDEN as u32, SALT).unwrap(),
            &rows,
            &fixture.activation_salts,
            ACTIVATION_CONFIGURATION,
        )
        .unwrap();
        let unsigned = CommittedActivation::commit(
            ActivationShape::new(TOY_POSITIONS as u32, TOY_HIDDEN as u32, SALT).unwrap(),
            &unsigned_rows,
            &fixture.activation_salts,
            ACTIVATION_CONFIGURATION,
        )
        .unwrap();
        assert_ne!(signed.root(), unsigned.root());
        let output_rows: Vec<Vec<i64>> = (0..TOY_POSITIONS)
            .map(|position| {
                fixture.outputs[2][0]
                    [position * TOY_COLUMNS..(position + 1) * TOY_COLUMNS]
                    .to_vec()
            })
            .collect();
        let mut overflowing = output_rows.clone();
        overflowing[0][TOY_VALID] = i64::MIN;
        assert!(close_output_relation(&output, &overflowing).is_err());
    }

    #[test]
    fn closure_tail_union_is_enforced_by_the_plan() {
        let mut fixture = fixture();
        let (rows, output, _activation) = formal_inputs(&fixture);
        let mut setup = fixture.setup.clone();
        let closure = setup.closure.as_mut().expect("closure setup");
        closure.slack_openings = closure.slack.code.r as usize;
        assert!(prove_output_head_lin_hidden(
            fixture.shape,
            &setup,
            &output,
            &mut fixture.table,
            &rows,
            &fixture.activation_salts,
        )
        .is_err());
    }
}
