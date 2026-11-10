//! Output-head MatMul wiring for the three-stage LIN request.
//!
//! The output head computes, for each position `t`,
//!
//! ```text
//! Y[t][j] = sum_y X[t][y] * M[y][j]
//! ```
//!
//! over the padded hidden dimension (`y` in `0..1024`) and the full padded
//! vocabulary (`j` in `0..65536`). The LIN request for position `t` uses the
//! row weight vector `g_t[y] = Fp(X[t][y])`, the challenge point `psi_t` over
//! the `k_col = 16` column bits, and the claim
//!
//! ```text
//! sum_j eq(psi_t, j) * Fp(Y[t][j]) + masks = sum_y g_t[y] * W_t[y]
//! ```
//!
//! where `W_t = g_t^T M`. The left side is recomputed from the verified trace
//! output; the right side is the LIN Eval relation and the full-value `W`
//! control. The Dot and Fold relations are checked by the existing three-stage
//! component.
//!
//! The sixteen positions are independent claims: one three-stage message set
//! per position, each with its own challenge point and its own `Omega` and
//! folded commitments. No combination coefficients are used and no batched
//! claim or error bound is claimed. The component supplies local equality
//! checks, not a soundness bound.
//!
//! Signed values enter the field through the two's-complement sign extension:
//! `fp_from_signed_word` maps a negative word to the negative field element,
//! never to its unsigned reduction. The activation values must fit `i32`; the
//! output values are the checked `i64` output of the MatMul relation. The
//! equality checked here is field equality, so it shows equality modulo the
//! base field order only. The integer bounds and the checked `i128` product are
//! enforced by the traced MatMul relation, not by this module.
//!
//! The table is authenticated once per request from the output-head asset
//! opening carried by the verified trace. The verifier takes `X` and `Y` from
//! `verify_shared_trace_outputs`, so the activation and output are fixed by the
//! verified trace before `psi` is drawn. The vocabulary mask, the masking `Add`
//! and the argmax steps remain separate graph items bound by the trusted plan;
//! this module never reads an argmax result. All 65536 output columns take part
//! in the request, including the columns above the valid vocabulary size.
//!
//! The full verification bytes are the trace proof (which carries plaintext
//! weight rows and activations) plus this request's encoding plus the public
//! bundle. This is a controlled diagnostic: the trace openings are not hidden
//! and this module adds no hiding.

use zkie_core::common::codec::DecodeLimits;
use zkie_core::error::{Error, Result};
use zkie_engine::pcs::request::binding::BindingCandidate;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::lin::{eval, LinMask, LinShape, LinTable};
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};

use crate::op::OpCode;
use crate::proof::column_session::fp_from_signed_word;
use crate::proof::lin_session::{
    build_authenticated_lin_table, prove_lin_request_with_table, verify_lin_request_with_table,
    LinRequestMessages, LinRequestOutcome,
};
use crate::proof::model_commitment::{
    verify_shared_trace_outputs, ModelCommitmentPublic, SharedTrace,
};
use crate::proof::plan::{
    PlanConfigValue, PlanConstant, PlanLayout, PlanSource, PlanStep, TracePlan,
};
use crate::proof::session_baseline::{assemble_binding, SelectedMatmulStep, TrustedModel};

const OUTPUT_HEAD_PROOF_VERSION: u64 = 1;

/// Positions of one output-head MatMul request.
pub const OUTPUT_HEAD_POSITIONS: usize = 16;
/// Padded hidden dimension, one table/matrix row per entry.
pub const OUTPUT_HEAD_HIDDEN: usize = 1024;
/// Full padded vocabulary, one matrix column per entry.
pub const OUTPUT_HEAD_COLUMNS: usize = 65536;
/// Low column bits placed inside one table row.
pub const OUTPUT_HEAD_K_LO: u32 = 6;
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

    Ok(OutputHeadStep {
        step,
        activation_step,
        activation_output,
        masked_step,
        positions,
        hidden,
        columns,
        asset_path: asset_path.to_owned(),
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

fn prepare_output_head_trace_for(
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

fn fp3_from_signed_i64(value: i64) -> Fp3 {
    Fp3::from_base(fp_from_signed_word(value as u64))
}

/// The per-position weights and field values derived from `X`, `Y` and the
/// authenticated table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputHeadLinClaims {
    pub shape: LinShape,
    /// `g_t` for every position.
    pub weights: Vec<Vec<Fp3>>,
    /// `W_t = g_t^T M` for every position.
    pub w_values: Vec<Vec<Fp3>>,
    /// `Fp(Y_t)` for every position.
    pub output_values: Vec<Vec<Fp3>>,
    /// The caller's claim masks, one list per position.
    pub masks: Vec<Vec<LinMask>>,
}

/// Builds the sixteen claims. The activation must fit `i32`; the output is the
/// checked `i64` MatMul result. Masks are empty when the caller passes an empty
/// outer slice; otherwise one mask list per position is required.
pub fn output_head_claims(
    shape: LinShape,
    table: &LinTable,
    inputs: &OutputHeadInputs,
    masks: &[Vec<LinMask>],
) -> Result<OutputHeadLinClaims> {
    shape.check()?;
    if table.shape != shape {
        return Err(Error::ShapeMismatch);
    }
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
    let mut w_values = Vec::with_capacity(positions);
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
            let value = inputs.output[position * columns + column];
            output.push(fp3_from_signed_i64(value));
        }
        w_values.push(table.w(&weight)?);
        output_values.push(output);
        weights.push(weight);
    }

    Ok(OutputHeadLinClaims {
        shape,
        weights,
        w_values,
        output_values,
        masks,
    })
}

/// The independent three-stage message sets of one output-head request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputHeadLinProof {
    pub messages: Vec<LinRequestMessages>,
}

impl OutputHeadLinProof {
    /// Canonical encoding: one version word, the claim count, then one
    /// length-prefixed message set per position.
    pub fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(OUTPUT_HEAD_PROOF_VERSION);
        writer.put_u64(self.messages.len() as u64);
        for message in &self.messages {
            writer.put_bytes(&message.encode());
        }
        writer.into_bytes()
    }

    /// Decodes exactly one canonical request proof and rejects trailing bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let limits = DecodeLimits {
            max_bytes: bytes.len(),
            max_elements: bytes.len(),
        };
        let mut reader = CanonicalReader::new(bytes, limits);
        if reader.get_u64()? != OUTPUT_HEAD_PROOF_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let count = reader.get_u64()?;
        let count = usize::try_from(count).map_err(|_| Error::InvalidEncoding)?;
        if count > reader.remaining().len() / 8 {
            return Err(Error::InvalidEncoding);
        }
        let mut messages = Vec::with_capacity(count);
        for _ in 0..count {
            messages.push(LinRequestMessages::decode(reader.get_bytes()?)?);
        }
        reader.finish()?;
        Ok(Self { messages })
    }

    /// Size of the canonical encoding in bytes.
    pub fn encoded_len(&self) -> usize {
        16 + self
            .messages
            .iter()
            .map(|message| 8 + message.encode().len())
            .sum::<usize>()
    }
}

/// Verifier outcome over all positions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputHeadLinOutcome {
    /// One component outcome per position, in position order.
    pub outcomes: Vec<LinRequestOutcome>,
    /// Canonical size of the request proof checked by the verifier.
    pub proof_bytes: usize,
}

fn prove_prepared(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    trace: &SharedTrace,
    request: &OutputHeadTrace,
    shape: LinShape,
    masks: &[Vec<LinMask>],
) -> Result<OutputHeadLinProof> {
    let opening = trace
        .assets
        .get(request.asset_index)
        .ok_or(Error::IncompleteArtifact)?;
    let table = build_authenticated_lin_table(
        public,
        trusted,
        &request.step.asset_path,
        opening,
        shape,
    )?;
    let claims = output_head_claims(shape, &table, &request.inputs, masks)?;
    let mut messages = Vec::with_capacity(claims.weights.len());
    for position in 0..claims.weights.len() {
        let message = prove_lin_request_with_table(
            &table,
            &request.binding,
            shape,
            &claims.weights[position],
            &claims.masks[position],
            &claims.w_values[position],
        )?;
        let right = eval(&claims.w_values[position], &message.first.psi)?;
        let left = eval(&claims.output_values[position], &message.first.psi)?;
        if left != right {
            return Err(Error::InvalidProof);
        }
        messages.push(message);
    }
    Ok(OutputHeadLinProof { messages })
}

fn verify_prepared(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    trace: &SharedTrace,
    request: &OutputHeadTrace,
    shape: LinShape,
    proof: &OutputHeadLinProof,
    masks: &[Vec<LinMask>],
) -> Result<OutputHeadLinOutcome> {
    let opening = trace
        .assets
        .get(request.asset_index)
        .ok_or(Error::IncompleteArtifact)?;
    let table = build_authenticated_lin_table(
        public,
        trusted,
        &request.step.asset_path,
        opening,
        shape,
    )?;
    let claims = output_head_claims(shape, &table, &request.inputs, masks)?;
    let positions = claims.weights.len();
    if proof.messages.len() != positions {
        return Err(Error::ShapeMismatch);
    }
    let rows = usize::try_from(shape.matrix_rows).map_err(|_| Error::ResourceLimit)?;
    let slots = 1usize
        .checked_shl(shape.k_row())
        .ok_or(Error::ResourceLimit)?;
    let mut outcomes = Vec::with_capacity(positions);
    for (position, message) in proof.messages.iter().enumerate() {
        if message.first.masks != claims.masks[position] {
            return Err(Error::StatementMismatch);
        }
        let weights = &message.first.weights.values;
        if weights.len() != slots
            || weights.get(..rows) != Some(claims.weights[position].as_slice())
            || weights[rows..].iter().any(|value| !value.is_zero())
        {
            return Err(Error::StatementMismatch);
        }
        let outcome = verify_lin_request_with_table(&table, &request.binding, shape, message)?;
        let right = eval(&claims.w_values[position], &outcome.psi)?;
        let left = eval(&claims.output_values[position], &outcome.psi)?;
        if left != right {
            return Err(Error::InvalidProof);
        }
        outcomes.push(outcome);
    }
    Ok(OutputHeadLinOutcome {
        outcomes,
        proof_bytes: 0,
    })
}

/// Verifies the shared trace, extracts `X` and `Y` from its verified step
/// outputs, and proves the sixteen LIN requests.
pub fn prove_output_head_lin_request(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    plan: &TracePlan,
    trace: &SharedTrace,
    expected_tokens: &[u32],
    shape: LinShape,
    masks: &[Vec<LinMask>],
) -> Result<OutputHeadLinProof> {
    let outputs = verify_shared_trace_outputs(public, trace, expected_tokens, plan)?;
    let request = prepare_output_head_trace(public, plan, trace, &outputs, shape)?;
    prove_prepared(public, trusted, trace, &request, shape, masks)
}

/// Verifies the shared trace independently, rebuilds the request from the
/// verified step outputs and the trusted plan, and verifies every position.
pub fn verify_output_head_lin_request(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    plan: &TracePlan,
    trace: &SharedTrace,
    expected_tokens: &[u32],
    shape: LinShape,
    proof: &OutputHeadLinProof,
    masks: &[Vec<LinMask>],
) -> Result<OutputHeadLinOutcome> {
    let outputs = verify_shared_trace_outputs(public, trace, expected_tokens, plan)?;
    let request = prepare_output_head_trace(public, plan, trace, &outputs, shape)?;
    let mut outcome = verify_prepared(public, trusted, trace, &request, shape, proof, masks)?;
    outcome.proof_bytes = proof.encoded_len();
    Ok(outcome)
}

/// Verifier path over the canonical request encoding. Decoding rejects trailing
/// bytes before the trace and the request are checked.
#[allow(clippy::too_many_arguments)]
pub fn verify_output_head_lin_request_bytes(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    plan: &TracePlan,
    trace: &SharedTrace,
    expected_tokens: &[u32],
    shape: LinShape,
    bytes: &[u8],
    masks: &[Vec<LinMask>],
) -> Result<OutputHeadLinOutcome> {
    let proof = OutputHeadLinProof::decode(bytes)?;
    let outputs = verify_shared_trace_outputs(public, trace, expected_tokens, plan)?;
    let request = prepare_output_head_trace(public, plan, trace, &outputs, shape)?;
    let mut outcome = verify_prepared(public, trusted, trace, &request, shape, &proof, masks)?;
    outcome.proof_bytes = bytes.len();
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use zkie_core::common::field::PrimeCharacteristicRing;
    use zkie_core::types::Digest;
    use zkie_engine::pcs::request::field::Fp;

    use crate::matmul::MatMulOp;
    use crate::proof::model_commitment::{commit_assets, AssetStore, ModelAssetSpec};
    use crate::proof::plan::{config, PlanBuilder};
    use crate::registry::PureForward;

    const TOY_POSITIONS: usize = 16;
    const TOY_HIDDEN: usize = 4;
    const TOY_COLUMNS: usize = 16;
    const TOY_K_LO: u32 = 2;
    const TOY_TABLE_ROWS: u64 = 4;
    const TOY_VALID: usize = 6;
    const TOY_ASSET: &str = "toy_head_i32.bin";

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "zkie-output-head-lin-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn build_plan(activation_steps: usize) -> TracePlan {
        let mut builder = PlanBuilder::new();
        let mut activation = 0;
        for _ in 0..activation_steps {
            activation = builder.step(
                OpCode::Add,
                Vec::new(),
                vec![PlanSource::Constant(PlanConstant::Zeros {
                    count: TOY_POSITIONS * TOY_HIDDEN,
                })],
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
        plan
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
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn fixture() -> Fixture {
        let root = temp_dir();
        let table_values: Vec<i32> = (0..(TOY_HIDDEN * TOY_COLUMNS))
            .map(|index| ((index as i32 * 37) % 23) - 11)
            .collect();
        let bytes: Vec<u8> = table_values
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
        let shape =
            LinShape::new(TOY_HIDDEN as u64, TOY_K_LO, TOY_TABLE_ROWS, TOY_COLUMNS as u64)
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
        let table: Vec<i64> = table_values.iter().map(|value| *value as i64).collect();
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
        let outputs = vec![vec![activation.clone()], vec![output.clone()]];

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

    fn request(
        fixture: &Fixture,
        plan: &TracePlan,
        outputs: &[Vec<Vec<i64>>],
    ) -> Result<OutputHeadTrace> {
        prepare_output_head_trace_for(
            &fixture.public,
            plan,
            &fixture.trace,
            outputs,
            fixture.shape,
            TOY_POSITIONS,
            TOY_HIDDEN,
            TOY_COLUMNS,
            TOY_ASSET,
        )
    }

    fn prove_request(
        fixture: &Fixture,
        plan: &TracePlan,
        outputs: &[Vec<Vec<i64>>],
        masks: &[Vec<LinMask>],
    ) -> Result<OutputHeadLinProof> {
        let request = request(fixture, plan, outputs)?;
        let trusted = trusted(fixture);
        prove_prepared(
            &fixture.public,
            &trusted,
            &fixture.trace,
            &request,
            fixture.shape,
            masks,
        )
    }

    fn verify_request(
        fixture: &Fixture,
        plan: &TracePlan,
        outputs: &[Vec<Vec<i64>>],
        proof: &OutputHeadLinProof,
        masks: &[Vec<LinMask>],
    ) -> Result<OutputHeadLinOutcome> {
        let request = request(fixture, plan, outputs)?;
        let trusted = trusted(fixture);
        let mut outcome = verify_prepared(
            &fixture.public,
            &trusted,
            &fixture.trace,
            &request,
            fixture.shape,
            proof,
            masks,
        )?;
        outcome.proof_bytes = proof.encoded_len();
        Ok(outcome)
    }

    fn verify_bytes(
        fixture: &Fixture,
        outputs: &[Vec<Vec<i64>>],
        bytes: &[u8],
        masks: &[Vec<LinMask>],
    ) -> Result<OutputHeadLinOutcome> {
        let request = request(fixture, &fixture.plan, outputs)?;
        let trusted = trusted(fixture);
        let proof = OutputHeadLinProof::decode(bytes)?;
        let mut outcome = verify_prepared(
            &fixture.public,
            &trusted,
            &fixture.trace,
            &request,
            fixture.shape,
            &proof,
            masks,
        )?;
        outcome.proof_bytes = bytes.len();
        Ok(outcome)
    }

    fn table(fixture: &Fixture) -> LinTable {
        let trusted = trusted(fixture);
        build_authenticated_lin_table(
            &fixture.public,
            &trusted,
            TOY_ASSET,
            &fixture.trace.assets[0],
            fixture.shape,
        )
        .unwrap()
    }

    fn empty_masks() -> Vec<Vec<LinMask>> {
        vec![Vec::new(); TOY_POSITIONS]
    }

    #[test]
    fn toy_request_prover_and_verifier_agree_on_every_position() {
        let fixture = fixture();
        let masks = empty_masks();
        let proof = prove_request(&fixture, &fixture.plan, &fixture.outputs, &masks).unwrap();
        assert_eq!(proof.messages.len(), TOY_POSITIONS);
        let outcome =
            verify_request(&fixture, &fixture.plan, &fixture.outputs, &proof, &masks).unwrap();
        assert_eq!(outcome.outcomes.len(), TOY_POSITIONS);
        for position in 0..TOY_POSITIONS {
            let value = &outcome.outcomes[position];
            assert!(value.eval_holds && value.dot_holds && value.fold_holds);
            assert!(value.control_w && value.control_omega);
            assert_eq!(value.first_query, proof.messages[position].second.first_query);
        }

        let request = request(&fixture, &fixture.plan, &fixture.outputs).unwrap();
        let table = table(&fixture);
        let claims = output_head_claims(
            fixture.shape,
            &table,
            &request.inputs,
            &masks,
        )
        .unwrap();
        for position in 0..TOY_POSITIONS {
            assert_eq!(claims.weights[position].len(), TOY_HIDDEN);
            for weight in &claims.weights[position] {
                assert_eq!(weight.c[1], Fp::ZERO);
                assert_eq!(weight.c[2], Fp::ZERO);
            }
            assert_eq!(
                claims.w_values[position],
                table.w(&claims.weights[position]).unwrap()
            );
            assert_eq!(
                claims.output_values[position],
                (0..TOY_COLUMNS)
                    .map(|column| {
                        fp3_from_signed_i64(
                            fixture.outputs[1][0][position * TOY_COLUMNS + column],
                        )
                    })
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                proof.messages[position].first.weights.values[..TOY_HIDDEN],
                claims.weights[position][..]
            );
            assert_eq!(
                proof.messages[position].first.w.values,
                claims.w_values[position]
            );
        }
    }

    #[test]
    fn signed_words_are_not_unsigned_reductions() {
        assert_eq!(fp3_from_signed_i64(-1), Fp3::ZERO.sub(&Fp3::ONE));
        assert_ne!(fp3_from_signed_i64(-1), Fp3::from_u64(u64::MAX));
        assert_eq!(
            fp3_from_signed_i64(i32::MIN as i64),
            Fp3::ZERO.sub(&Fp3::from_u64(1u64 << 31))
        );
        assert_eq!(
            fp3_from_signed_i64(i32::MAX as i64),
            Fp3::from_u64(i32::MAX as u64)
        );
        for value in [i32::MIN, -1, 0, i32::MAX] {
            let mapped = fp3_from_signed_i64(value as i64);
            assert_eq!(mapped.c[1], Fp::ZERO);
            assert_eq!(mapped.c[2], Fp::ZERO);
        }
    }

    #[test]
    fn representative_overflow_negative_is_rejected_by_the_checked_product() {
        let activation = vec![i32::MIN as i64; TOY_HIDDEN];
        let weights = vec![i32::MIN as i64; TOY_HIDDEN * TOY_COLUMNS];
        let result = MatMulOp {
            m: 1,
            n: TOY_COLUMNS,
            k: TOY_HIDDEN,
            shift: 0,
        }
        .forward_values(&[activation, weights]);
        assert!(matches!(result, Err(Error::BoundViolation)));
        // The field mapping alone represents both factors; the checked
        // product bound is what rejects the overflowing output.
        assert_ne!(fp3_from_signed_i64(i32::MIN as i64), Fp3::ZERO);
    }

    #[test]
    fn activation_outside_the_signed_word_domain_is_rejected() {
        let fixture = fixture();
        let mut outputs = fixture.outputs.clone();
        outputs[0][0][0] = i32::MAX as i64 + 1;
        assert!(matches!(
            request(&fixture, &fixture.plan, &outputs),
            Err(Error::BoundViolation)
        ));
        outputs[0][0][0] = i32::MIN as i64 - 1;
        assert!(matches!(
            request(&fixture, &fixture.plan, &outputs),
            Err(Error::BoundViolation)
        ));
    }

    #[test]
    fn changed_activation_or_output_is_rejected() {
        let fixture = fixture();
        let masks = empty_masks();
        let proof = prove_request(&fixture, &fixture.plan, &fixture.outputs, &masks).unwrap();
        let outcome =
            verify_request(&fixture, &fixture.plan, &fixture.outputs, &proof, &masks).unwrap();
        let psi = outcome.outcomes[0].psi.clone();

        let mut changed_activation = fixture.outputs.clone();
        let original = changed_activation[0][0][0];
        changed_activation[0][0][0] = original.wrapping_add(1);
        assert!(verify_request(
            &fixture,
            &fixture.plan,
            &changed_activation,
            &proof,
            &masks
        )
        .is_err());

        let column = (TOY_VALID..TOY_COLUMNS)
            .find(|column| {
                let mut unit = vec![Fp3::ZERO; TOY_COLUMNS];
                unit[*column] = Fp3::ONE;
                eval(&unit, &psi).unwrap() != Fp3::ZERO
            })
            .expect("a padded column with a nonzero evaluation weight");
        let mut changed_output = fixture.outputs.clone();
        changed_output[1][0][column] = changed_output[1][0][column].checked_add(1).unwrap();
        assert!(verify_request(
            &fixture,
            &fixture.plan,
            &changed_output,
            &proof,
            &masks
        )
        .is_err());
    }

    #[test]
    fn changed_graph_edge_or_binding_is_rejected() {
        let fixture = fixture();
        let masks = empty_masks();
        let proof = prove_request(&fixture, &fixture.plan, &fixture.outputs, &masks).unwrap();

        let shifted = build_plan(2);
        let mut shifted_outputs = vec![
            vec![fixture.outputs[0][0].clone()],
            vec![fixture.outputs[0][0].clone()],
        ];
        shifted_outputs.extend(fixture.outputs[1..].iter().cloned());
        let shifted_request = request(&fixture, &shifted, &shifted_outputs).unwrap();
        let original_request = request(&fixture, &fixture.plan, &fixture.outputs).unwrap();
        assert_ne!(shifted_request.binding, original_request.binding);
        assert!(verify_request(
            &fixture,
            &shifted,
            &shifted_outputs,
            &proof,
            &masks
        )
        .is_err());

        let mut swapped = fixture.plan.clone();
        swapped.steps[1].inputs.swap(0, 1);
        assert!(request(&fixture, &swapped, &fixture.outputs).is_err());

        let mut missing = fixture.plan.clone();
        missing.steps[1].inputs[1] = PlanSource::asset("missing_head_i32.bin");
        assert!(request(&fixture, &missing, &fixture.outputs).is_err());

        let mut wrong_root = fixture.trace.clone();
        wrong_root.assets[0].root.0[0] ^= 1;
        let request = request(&fixture, &fixture.plan, &fixture.outputs).unwrap();
        let trusted = trusted(&fixture);
        assert!(build_authenticated_lin_table(
            &fixture.public,
            &trusted,
            TOY_ASSET,
            &wrong_root.assets[0],
            fixture.shape
        )
        .is_err());
        assert_eq!(request.asset_index, 0);
    }

    #[test]
    fn changed_table_opening_value_is_rejected_through_the_request() {
        let fixture = fixture();
        let masks = empty_masks();
        let mut trace = fixture.trace.clone();
        trace.assets[0].rows[0].values[0] ^= 1;
        let request = prepare_output_head_trace_for(
            &fixture.public,
            &fixture.plan,
            &trace,
            &fixture.outputs,
            fixture.shape,
            TOY_POSITIONS,
            TOY_HIDDEN,
            TOY_COLUMNS,
            TOY_ASSET,
        )
        .unwrap();
        let trusted = trusted(&fixture);
        assert!(prove_prepared(
            &fixture.public,
            &trusted,
            &trace,
            &request,
            fixture.shape,
            &masks
        )
        .is_err());
    }

    #[test]
    fn masks_change_the_target_and_are_bound_by_the_verifier() {
        let fixture = fixture();
        let masks: Vec<Vec<LinMask>> = (0..TOY_POSITIONS)
            .map(|position| {
                vec![LinMask {
                    value: Fp3::from_u64(position as u64 + 1),
                    coefficient: Fp3::from_u64(3),
                }]
            })
            .collect();
        let proof = prove_request(&fixture, &fixture.plan, &fixture.outputs, &masks).unwrap();
        let outcome =
            verify_request(&fixture, &fixture.plan, &fixture.outputs, &proof, &masks).unwrap();
        assert_eq!(outcome.outcomes.len(), TOY_POSITIONS);

        let request = request(&fixture, &fixture.plan, &fixture.outputs).unwrap();
        let table = table(&fixture);
        let claims = output_head_claims(fixture.shape, &table, &request.inputs, &masks).unwrap();
        for position in 0..TOY_POSITIONS {
            let unmasked = eval(
                &claims.w_values[position],
                &proof.messages[position].first.psi,
            )
            .unwrap();
            let mask = &masks[position][0];
            assert_eq!(
                proof.messages[position].first.target,
                unmasked.add(&mask.value.mul(&mask.coefficient))
            );
        }

        let mut tampered = proof.clone();
        tampered.messages[0].first.masks[0].coefficient = Fp3::from_u64(4);
        assert!(verify_request(
            &fixture,
            &fixture.plan,
            &fixture.outputs,
            &tampered,
            &masks
        )
        .is_err());

        let empty = empty_masks();
        assert!(verify_request(
            &fixture,
            &fixture.plan,
            &fixture.outputs,
            &proof,
            &empty
        )
        .is_err());
    }

    #[test]
    fn canonical_encoding_round_trips_and_rejects_trailing_bytes() {
        let fixture = fixture();
        let masks = empty_masks();
        let proof = prove_request(&fixture, &fixture.plan, &fixture.outputs, &masks).unwrap();
        let bytes = proof.encode();
        assert_eq!(OutputHeadLinProof::decode(&bytes).unwrap(), proof);
        assert_eq!(proof.encoded_len(), bytes.len());
        let outcome = verify_bytes(&fixture, &fixture.outputs, &bytes, &masks).unwrap();
        assert_eq!(outcome.proof_bytes, bytes.len());

        let mut trailing = bytes;
        trailing.push(0);
        assert!(OutputHeadLinProof::decode(&trailing).is_err());
        assert!(verify_bytes(&fixture, &fixture.outputs, &trailing, &masks).is_err());
    }

    #[test]
    fn toy_claims_use_the_signed_activation_and_full_output_domain() {
        let fixture = fixture();
        let masks = empty_masks();
        let request = request(&fixture, &fixture.plan, &fixture.outputs).unwrap();
        let table = table(&fixture);
        let claims = output_head_claims(fixture.shape, &table, &request.inputs, &masks).unwrap();
        let proof = prove_request(&fixture, &fixture.plan, &fixture.outputs, &masks).unwrap();
        let table_values = table.rows.clone();

        for position in 0..TOY_POSITIONS {
            let psi = &proof.messages[position].first.psi;
            for row in 0..TOY_HIDDEN {
                let value = fixture.outputs[0][0][position * TOY_HIDDEN + row] as i32;
                assert_eq!(
                    claims.weights[position][row],
                    fp3_from_signed_i64(value as i64)
                );
            }
            let mut column_weights = Vec::with_capacity(TOY_COLUMNS);
            for column in 0..TOY_COLUMNS {
                let mut unit = vec![Fp3::ZERO; TOY_COLUMNS];
                unit[column] = Fp3::ONE;
                column_weights.push(eval(&unit, psi).unwrap());
            }
            let mut right = Fp3::ZERO;
            for row in 0..TOY_HIDDEN {
                let mut sum = Fp3::ZERO;
                for column in 0..TOY_COLUMNS {
                    let table_row = (column as u64) >> fixture.shape.k_lo;
                    let place = (column as u64) & ((1u64 << fixture.shape.k_lo) - 1);
                    let entry = place + (row as u64) * (1u64 << fixture.shape.k_lo);
                    let value = table_values[table_row as usize][entry as usize];
                    sum = sum.add(&value.mul(&column_weights[column]));
                }
                right = right.add(&claims.weights[position][row].mul(&sum));
            }
            let left = eval(&claims.output_values[position], psi).unwrap();
            assert_eq!(left, right);
        }
    }

    #[test]
    fn real_output_head_shape_covers_the_padded_matrix_domain() {
        let shape = output_head_shape().unwrap();
        assert_eq!(shape.matrix_rows, OUTPUT_HEAD_HIDDEN as u64);
        assert_eq!(shape.k_lo, OUTPUT_HEAD_K_LO);
        assert_eq!(shape.k_col, 16);
        assert_eq!(shape.k_row(), 10);
        assert_eq!(shape.matrix_columns().unwrap(), OUTPUT_HEAD_COLUMNS as u64);
        assert_eq!(shape.table_rows, OUTPUT_HEAD_TABLE_ROWS);
        assert_eq!(shape.table_columns, OUTPUT_HEAD_COLUMNS as u64);
        assert_eq!(
            shape.table_columns_used().unwrap(),
            OUTPUT_HEAD_COLUMNS as u64
        );
    }

    #[test]
    fn real_plan_shape_selects_the_output_head() {
        let mut builder = PlanBuilder::new();
        let activation = builder.step(
            OpCode::Add,
            Vec::new(),
            vec![PlanSource::Constant(PlanConstant::Zeros {
                count: OUTPUT_HEAD_POSITIONS * OUTPUT_HEAD_HIDDEN,
            })],
            1,
        );
        let logits = builder.step(
            OpCode::MatMul,
            config(vec![
                OUTPUT_HEAD_POSITIONS as i64,
                OUTPUT_HEAD_COLUMNS as i64,
                OUTPUT_HEAD_HIDDEN as i64,
                0,
                0,
            ]),
            vec![
                PlanSource::prev(activation, 0),
                PlanSource::asset(OUTPUT_HEAD_ASSET_PATH),
            ],
            1,
        );
        let masked = builder.step(
            OpCode::Add,
            Vec::new(),
            vec![
                PlanSource::prev(logits, 0),
                PlanSource::Constant(PlanConstant::VocabMask {
                    rows: OUTPUT_HEAD_POSITIONS,
                    vocab: OUTPUT_HEAD_COLUMNS,
                    valid: 50257,
                    masked: -(1i64 << 41),
                }),
            ],
            1,
        );
        for row in 0..OUTPUT_HEAD_POSITIONS {
            let argmax = builder.step(
                OpCode::Argmax,
                Vec::new(),
                vec![PlanSource::prev_slice(
                    masked,
                    0,
                    PlanLayout::rows(OUTPUT_HEAD_COLUMNS, row, 1),
                )],
                1,
            );
            builder.argmax_step(argmax);
        }
        let plan = builder.finish();
        let step = select_output_head_step(&plan).unwrap();
        assert_eq!(step.step, logits);
        assert_eq!(step.activation_step, activation);
        assert_eq!(step.masked_step, masked);
        assert_eq!(step.positions, OUTPUT_HEAD_POSITIONS);
        assert_eq!(step.hidden, OUTPUT_HEAD_HIDDEN);
        assert_eq!(step.columns, OUTPUT_HEAD_COLUMNS);
        assert_eq!(step.asset_path, OUTPUT_HEAD_ASSET_PATH);
    }
}
