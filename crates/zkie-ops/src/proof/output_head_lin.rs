//! Output-head MatMul wiring for the merged three-stage LIN request.
//!
//! The output head computes, for each position `t`,
//!
//! ```text
//! Y[t][j] = sum_y X[t][y] * M[y][j]
//! ```
//!
//! over the padded hidden dimension and the full padded vocabulary. This
//! module binds the trusted MatMul step and its trace tensors to the merged
//! hidden session: `X` is committed under a hiding row commitment, the public
//! row weights and output `Y` are plan values derived from the independently
//! verified trace, and the session itself folds the committed encoded table,
//! samples the reduced weight rows and runs both proximity tests. The
//! verifier takes only the committed table reader, the public weights, `Y`
//! and the proof; it never receives an activation row, a table message row or
//! a reduced weight row in the clear.
//!
//! The trace is still verified for the model/anchor/step binding and for the
//! activation and output values. The old plaintext weight-table path is gone:
//! the table comes from the caller's committed table handle, not from a trace
//! asset opening.
//!
//! This is a diagnostic and development entry. It does not claim a formal
//! soundness bound, a production CLI wiring or a full-model proof. The
//! proximity tests run on toy or stage-harness plans supplied by the caller;
//! the plan values are trusted inputs, not proof bytes.

use zkie_core::common::codec::DecodeLimits;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::binding::BindingCandidate;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::foldeq::reader_table_identity;
use zkie_engine::pcs::request::lin::{LinMask, LinShape};
use zkie_engine::pcs::request::table::IndexReader;
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};

use crate::op::OpCode;
use crate::proof::column_session::fp_from_signed_word;
use crate::proof::hidden_send_session::{
    prove_hidden_lin, verify_hidden_lin, ActivationShape, CommittedActivation, HiddenLinMessages,
    HiddenLinOutcome, HiddenLinPlan, HiddenLinWitness, HiddenOutputShape, HiddenProximityPlan,
    HiddenSendPlan, HiddenTableProver,
};
use crate::proof::model_commitment::{
    verify_shared_trace_outputs, ModelCommitmentPublic, SharedTrace,
};
use crate::proof::plan::{
    PlanConfigValue, PlanConstant, PlanLayout, PlanSource, PlanStep, TracePlan,
};
use crate::proof::session_baseline::{assemble_binding, SelectedMatmulStep};


const OUTPUT_HEAD_PROOF_VERSION: u64 = 2;

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


// ---------------------------------------------------------------------------
// Public claims and the merged setup
// ---------------------------------------------------------------------------

fn fp3_from_signed_i64(value: i64) -> Fp3 {
    Fp3::from_base(fp_from_signed_word(value as u64))
}

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
    pub activation_configuration: Digest,
    pub activation_salts: Vec<Vec<u8>>,
    pub seed: Digest,
    pub masks: Vec<Vec<LinMask>>,
    /// The verifier's pinned activation root. The prover ignores it.
    pub expected_activation_root: Option<Digest>,
    /// The verifier's pinned reduced weight roots. The prover may leave it
    /// empty; the verifier requires a value.
    pub expected_roots: Option<Vec<Digest>>,
}

fn merged_plan(
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
    send.expected_roots = expected_roots;
    let plan = HiddenLinPlan {
        send,
        output: HiddenOutputShape {
            positions,
            columns: shape.matrix_columns()?,
        },
        weights: claims.weights.clone(),
        masks: claims.masks.clone(),
        proximity: setup.proximity.clone(),
    };
    plan.validate()?;
    Ok(plan)
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
        if reader.get_u64()? != OUTPUT_HEAD_PROOF_VERSION {
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

fn verify_trace(
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

fn values_match(shape: LinShape, positions: usize) -> bool {
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


#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use zkie_core::common::field::PrimeCharacteristicRing;
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
        WShape, HiddenProximityPlan, HiddenSendPlan, HiddenTableProver,
    };
    use crate::proof::model_commitment::{commit_assets, AssetStore, ModelAssetSpec};
    use crate::proof::plan::{config, PlanBuilder};
    use crate::proof::session_baseline::{SelectedMatmulStep, TrustedModel};
    use crate::registry::PureForward;

    const TOY_POSITIONS: usize = 16;
    const TOY_HIDDEN: usize = 4;
    const TOY_COLUMNS: usize = 16;
    const TOY_K_LO: u32 = 2;
    const TOY_TABLE_ROWS: u64 = 4;
    const TOY_VALID: usize = 6;
    const TOY_ASSET: &str = "toy_head_i32.bin";
    const GROUP_CODE_LOG_N: u32 = 6;
    const GROUP_CODE_ELL: u64 = 32;
    const GROUP_CODE_R: u64 = 1;
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

    fn proximity() -> HiddenProximityPlan {
        HiddenProximityPlan {
            gu: gu_shape(),
            omega: omega_shape(),
            code: group_code(),
            mask_code: group_code(),
            shared_mask_tree: mask_tree(SHARED_MASK_TREE),
            own_mask_tree: mask_tree(OWN_MASK_TREE),
            shared_fresh: 50,
            own_fresh: 51,
            column_draws: 2,
            column_budget: 4,
            column_work_bits: 0,
            shared_base_positions: 2,
            own_base_positions: 2,
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
        let outputs = vec![vec![activation.clone()], vec![output.clone()]];

        let encoded = encoded_table(&values);
        let send = HiddenSendPlan {
            activation: ActivationShape::new(TOY_POSITIONS as u32, TOY_HIDDEN as u32, SALT).unwrap(),
            activation_configuration: ACTIVATION_CONFIGURATION,
            activation_root: Digest([0x71; 32]),
            table_shape: encoded.reader.shape,
            table_identity: reader_table_identity(&encoded.reader).unwrap(),
            matrix_rows: TOY_HIDDEN as u64,
            k_lo: TOY_K_LO,
            columns: TOY_COLUMNS as u64,
            group: SEND_GROUP,
            oracle_base: ORACLE_BASE,
            openings_per_oracle: 2,
            w: w_shape(),
            expected_roots: None,
        };
        let salts: Vec<Vec<u8>> = (0..TOY_POSITIONS)
            .map(|position| vec![(position as u8).wrapping_mul(11) ^ 0x3C; SALT])
            .collect();
        let setup = OutputHeadLinSetup {
            asset_path: TOY_ASSET.into(),
            send,
            proximity: proximity(),
            activation_configuration: ACTIVATION_CONFIGURATION,
            activation_salts: salts.clone(),
            seed: Digest([0x72; 32]),
            masks: Vec::new(),
            expected_activation_root: None,
            expected_roots: None,
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
        assert!(outcome.outcome.shared_base.fold_main);
        assert!(outcome.outcome.own_base.fold_main);
        let bytes = fixture.proof.encode().unwrap();
        let merged = merged_plan(
            fixture.shape,
            &setup,
            &fixture.table.reader,
            &output_head_claims(
                fixture.shape,
                &OutputHeadInputs {
                    activation: fixture.outputs[0][0].clone(),
                    output: fixture.outputs[1][0].clone(),
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
    fn changed_activation_or_output_is_rejected() {
        let mut fixture = fixture();
        let setup = verifier_setup(&fixture);
        let original_zero = fixture.outputs[0][0][0];
        fixture.outputs[0][0][0] = original_zero.wrapping_add(1);
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
        fixture.outputs[0][0][0] = original_zero;

        fixture.outputs[1][0][0] = fixture.outputs[1][0][0].wrapping_add(1);
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
        changed.steps[1].inputs[1] = PlanSource::asset("missing_head_i32.bin");
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
        tampered.session.second.base_rows.base.lambda =
            tampered.session.second.base_rows.base.lambda.add(&Fp3::ONE);
        assert!(verify(&fixture, &setup, &fixture.table.reader, &tampered).is_err());
        let bytes = fixture.proof.encode().unwrap();
        let merged = merged_plan(
            fixture.shape,
            &setup,
            &fixture.table.reader,
            &output_head_claims(
                fixture.shape,
                &OutputHeadInputs {
                    activation: fixture.outputs[0][0].clone(),
                    output: fixture.outputs[1][0].clone(),
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
}
