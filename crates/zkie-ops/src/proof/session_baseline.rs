//! Isolated selected-step relation session baseline.
//!
//! The trusted plan selects one MatMul step. The verifier absorbs that step's
//! commitment references and full plaintext opening content into an existing
//! chain session, authenticates the same content with Merkle paths, and then
//! recomputes the relation in full. This is a replaceable baseline: the field
//! order, messages and challenge positions below are candidate-internal and a
//! later production closure will change them. It is not a production verifier
//! and it does not claim relation soundness, hiding or single-shot semantics.
//!
//! The baseline checks the selected step's own ports and external asset
//! openings. It does not verify upstream step outputs, the complete model graph
//! or external weight authorization beyond the selected asset opening. The
//! session challenge only checks transcript wiring; a transcript consistency
//! test is not evidence that the relation is sound. Hash authentication and
//! the surrounding system keep their own assumptions, so this module makes no
//! probability-free security claim.

use std::collections::BTreeSet;

use zkie_core::artifact::digest_bytes;
use zkie_core::error::{Error, Result};
use zkie_core::proof_header::PROOF_VERSION_WORD;
use zkie_core::types::Digest;
use zkie_engine::pcs::request::binding::{
    assemble, BindingCandidate, BindingParameters, BindingStatement,
};
use zkie_engine::pcs::request::chain::{Chain, KIND_ALGEBRAIC};
use zkie_engine::pcs::request::field::Fp;
use zkie_engine::pcs::request::params::{RequestParams, REQUEST_CHAIN_LABEL};
use zkie_engine::primitives::codec::CanonicalWriter;

use crate::op::OpCode;
use crate::proof::binding::{OracleId, OracleRegistry};
use crate::proof::model_commitment::{
    verify_model_anchor, AssetMeta, AssetOpening, ModelAssetSpec, ModelCommitmentPublic, SharedTrace,
    SharedTraceStep,
};
use crate::proof::plan::{PlanConfigValue, PlanLayout, PlanSource, TracePlan};
use crate::proof::relations::{
    verify_group, verify_op_relation_with_external, OpRelationPublic,
};

const SESSION_SCHEMA_VERSION: u64 = 1;

/// Candidate protocol identity for this baseline. It is not a final external
/// protocol identity.
pub const SESSION_PROTOCOL_IDENTITY: Digest = Digest([0x5a; 32]);

const GROUP_INPUTS: u64 = 1;
const GROUP_OUTPUTS: u64 = 2;
const GROUP_RAW: u64 = 3;
const GROUP_QUOTIENTS: u64 = 4;
const GROUP_HELPERS: u64 = 5;
const GROUP_AUX: u64 = 6;

/// Trusted model metadata from the same static plan source used by production.
/// Successful anchor verification establishes internal consistency with this
/// metadata; it does not decide external weight authorization, which remains
/// deferred.
pub struct TrustedModel<'a> {
    pub anchor_id: Digest,
    pub model_id: Digest,
    pub parameters_digest: Digest,
    pub asset_specs: &'a [ModelAssetSpec],
}

/// One MatMul step selected by the trusted plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedMatmulStep {
    pub step_index: usize,
    pub config: Vec<i64>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub shift: u32,
    pub inputs: Vec<PlanSource>,
}

impl SelectedMatmulStep {
    /// Selects a concrete MatMul step from the trusted plan. Only flat previous
    /// outputs and flat asset inputs are supported by this baseline.
    pub fn select(plan: &TracePlan, step_index: usize) -> Result<Self> {
        let plan_step = plan.steps.get(step_index).ok_or(Error::InvalidIndex)?;
        if plan_step.opcode != OpCode::MatMul {
            return Err(Error::UnsupportedPrimitive);
        }
        if plan_step.inputs.len() != 2 || plan_step.outputs != 1 {
            return Err(Error::ShapeMismatch);
        }
        let mut config = Vec::with_capacity(plan_step.config.len());
        for value in &plan_step.config {
            match value {
                PlanConfigValue::Literal(value) => config.push(*value),
                _ => return Err(Error::InvalidConfig),
            }
        }
        if config.len() != 5 || config[4] != 0 {
            return Err(Error::InvalidConfig);
        }
        let m = usize::try_from(config[0]).map_err(|_| Error::InvalidConfig)?;
        let n = usize::try_from(config[1]).map_err(|_| Error::InvalidConfig)?;
        let k = usize::try_from(config[2]).map_err(|_| Error::InvalidConfig)?;
        let shift = u32::try_from(config[3]).map_err(|_| Error::InvalidConfig)?;
        if m == 0 || n == 0 || k == 0 {
            return Err(Error::InvalidConfig);
        }
        for source in &plan_step.inputs {
            if !supported_source(source) {
                return Err(Error::UnsupportedPrimitive);
            }
        }
        Ok(Self {
            step_index,
            config,
            m,
            n,
            k,
            shift,
            inputs: plan_step.inputs.clone(),
        })
    }
}

fn supported_source(source: &PlanSource) -> bool {
    matches!(
        source,
        PlanSource::Prev {
            layout: PlanLayout::Flat,
            ..
        } | PlanSource::Asset {
            layout: PlanLayout::Flat,
            ..
        }
    )
}

/// One selected external asset opening and its planned values.
struct AssetPlaintext<'a> {
    port: usize,
    pool: usize,
    meta: &'a AssetMeta,
    opening: &'a AssetOpening,
    values: Vec<i64>,
}

/// The observed session values. They are not production acceptance objects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssetReference {
    pub port: usize,
    pub pool: usize,
    pub asset_id: u64,
    pub start: Digest,
    pub root: Digest,
    pub rows: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionOutcome {
    pub step_index: usize,
    pub parameters: Digest,
    pub statement_bytes: Vec<u8>,
    pub record_bytes: Vec<u8>,
    pub rho: Digest,
    pub challenge: Fp,
    pub assets: Vec<AssetReference>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionTranscript {
    pub rho: Digest,
    pub challenge: Fp,
}

fn digest_words(digest: &Digest) -> [u64; 4] {
    let mut words = [0u64; 4];
    for (index, chunk) in digest.0.chunks_exact(8).enumerate() {
        words[index] = u64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
    }
    words
}

fn push_digest(words: &mut Vec<u64>, digest: &Digest) {
    words.extend_from_slice(&digest_words(digest));
}

fn encode_plan_source(
    writer: &mut CanonicalWriter,
    public: &ModelCommitmentPublic,
    source: &PlanSource,
) -> Result<()> {
    match source {
        PlanSource::Prev {
            step,
            output,
            layout: PlanLayout::Flat,
        } => {
            writer.put_u64(1);
            writer.put_u64(*step as u64);
            writer.put_u64(*output as u64);
            Ok(())
        }
        PlanSource::Asset {
            path,
            layout: PlanLayout::Flat,
        } => {
            let meta = public
                .assets
                .iter()
                .find(|asset| asset.path == path.as_str())
                .ok_or(Error::AssetMismatch)?;
            writer.put_u64(2);
            writer.put_bytes(path.as_bytes());
            writer.put_u64(meta.asset_id);
            writer.put_digest(&meta.start);
            writer.put_digest(&meta.root);
            writer.put_u64(meta.rows);
            writer.put_u64(meta.row_width as u64);
            writer.put_u64(meta.element_count);
            writer.put_u64(meta.leaf_record.leaf_id);
            writer.put_u64(meta.leaf_record.generation);
            writer.put_digest(&meta.leaf_record.table_identity);
            writer.put_digest(&meta.leaf_record.leaf_root);
            Ok(())
        }
        _ => Err(Error::UnsupportedPrimitive),
    }
}

/// Candidate identifier for the verifier's selected static plan. It covers
/// the selected step, its config, every input port and source, and the
/// identity of every asset named by those sources. It does not cover the rest
/// of the trace graph, the case, the request, the protocol/security
/// parameters, or the source authenticity of the plan. The item order is
/// candidate-internal and replaceable.
pub fn selected_plan_identifier(
    public: &ModelCommitmentPublic,
    plan: &TracePlan,
    selection: &SelectedMatmulStep,
) -> Result<Digest> {
    let mut writer = CanonicalWriter::new();
    writer.put_u64(SESSION_SCHEMA_VERSION);
    writer.put_u64(plan.model_version as u64);
    writer.put_u64(selection.step_index as u64);
    writer.put_u64(OpCode::MatMul as u64);
    for value in &selection.config {
        writer.put_u64(*value as u64);
    }
    writer.put_u64(selection.inputs.len() as u64);
    for (port, source) in selection.inputs.iter().enumerate() {
        writer.put_u64(port as u64);
        encode_plan_source(&mut writer, public, source)?;
    }
    writer.put_u64(1);
    Ok(digest_bytes(&writer.into_bytes()))
}

/// Reuses the existing binding candidate to assemble the parameters,
/// statement and record for this session. The encoding is candidate-internal
/// and replaceable.
pub fn assemble_binding(
    public: &ModelCommitmentPublic,
    plan: &TracePlan,
    selection: &SelectedMatmulStep,
) -> Result<BindingCandidate> {
    let plan_digest = selected_plan_identifier(public, plan, selection)?;
    let request_parameters = RequestParams::default_with_code(4, 4, 0)?;
    let parameters = BindingParameters::with_plan_parts(
        PROOF_VERSION_WORD,
        SESSION_PROTOCOL_IDENTITY,
        &request_parameters,
        &[plan_digest],
    )?;
    let statement = BindingStatement::new(
        PROOF_VERSION_WORD,
        SESSION_PROTOCOL_IDENTITY,
        plan_digest.0.to_vec(),
        plan_digest,
        public.anchor.anchor_id,
    )?;
    assemble(&parameters, &statement, None)
}

fn check_plan_step(step: &SharedTraceStep, selection: &SelectedMatmulStep) -> Result<()> {
    if step.message.opcode != OpCode::MatMul as u16 {
        return Err(Error::StatementMismatch);
    }
    if step.message.config != selection.config {
        return Err(Error::StatementMismatch);
    }
    let proof = &step.message.proof;
    if proof.inputs.len() != 2
        || proof.outputs.len() != 1
        || step.external.len() != 2
        || step.sources.len() != 2
    {
        return Err(Error::ShapeMismatch);
    }
    let mut expected_external = Vec::new();
    for (port, source) in selection.inputs.iter().enumerate() {
        let external = step.external.get(port).copied().flatten();
        let edge = step.sources.get(port).copied().flatten();
        match source {
            PlanSource::Prev {
                step: previous,
                output,
                layout: PlanLayout::Flat,
            } => {
                if *previous >= selection.step_index
                    || edge != Some((*previous, *output))
                    || external.is_some()
                {
                    return Err(Error::StatementMismatch);
                }
            }
            PlanSource::Asset {
                layout: PlanLayout::Flat,
                ..
            } => {
                if edge.is_some() || external.is_none() {
                    return Err(Error::StatementMismatch);
                }
                expected_external.push(port);
            }
            _ => return Err(Error::UnsupportedPrimitive),
        }
    }
    if step.message.external_inputs != expected_external {
        return Err(Error::StatementMismatch);
    }
    if step.message.roots.is_empty() {
        return Err(Error::InvalidConfig);
    }
    Ok(())
}

fn collect_assets<'a>(
    public: &'a ModelCommitmentPublic,
    trace: &'a SharedTrace,
    selection: &SelectedMatmulStep,
    step: &SharedTraceStep,
) -> Result<Vec<AssetPlaintext<'a>>> {
    let mut assets = Vec::new();
    for (port, source) in selection.inputs.iter().enumerate() {
        let PlanSource::Asset {
            path,
            layout: PlanLayout::Flat,
        } = source
        else {
            continue;
        };
        let pool = step
            .external
            .get(port)
            .copied()
            .flatten()
            .ok_or(Error::StatementMismatch)?;
        let meta = public
            .assets
            .iter()
            .find(|asset| asset.path.as_str() == path.as_str())
            .ok_or(Error::AssetMismatch)?;
        let opening = trace.assets.get(pool).ok_or(Error::StatementMismatch)?;
        if opening.asset_id != meta.asset_id
            || opening.start != meta.start
            || opening.root != meta.root
            || opening.row_count != meta.rows
            || opening.rows.len() != meta.rows as usize
        {
            return Err(Error::StatementMismatch);
        }
        if meta.row_width == 0 || meta.element_count == 0 || !meta.rows.is_power_of_two() {
            return Err(Error::InvalidConfig);
        }
        let window = meta
            .rows
            .checked_mul(meta.row_width as u64)
            .ok_or(Error::ResourceLimit)?;
        if meta.element_count > window {
            return Err(Error::ShapeMismatch);
        }
        for (index, row) in opening.rows.iter().enumerate() {
            if row.row != index as u64 || row.values.len() != meta.row_width {
                return Err(Error::ShapeMismatch);
            }
        }
        let count = usize::try_from(meta.element_count).map_err(|_| Error::ResourceLimit)?;
        let mut values = Vec::with_capacity(count);
        for row in &opening.rows {
            for value in &row.values {
                if values.len() == count {
                    break;
                }
                values.push(*value as i64);
            }
            if values.len() == count {
                break;
            }
        }
        if values.len() != count {
            return Err(Error::IncompleteArtifact);
        }
        assets.push(AssetPlaintext {
            port,
            pool,
            meta,
            opening,
            values,
        });
    }
    Ok(assets)
}

fn build_external_values(
    step: &SharedTraceStep,
    assets: &mut [AssetPlaintext<'_>],
) -> Result<Vec<Option<Vec<i64>>>> {
    let mut values: Vec<Option<Vec<i64>>> = vec![None; step.message.proof.lengths.inputs.len()];
    for asset in assets {
        let slot = values.get_mut(asset.port).ok_or(Error::ShapeMismatch)?;
        if slot.replace(std::mem::take(&mut asset.values)).is_some() {
            return Err(Error::StatementMismatch);
        }
    }
    Ok(values)
}

fn checked_dimensions(selection: &SelectedMatmulStep) -> Result<(usize, usize, usize)> {
    let left = selection
        .m
        .checked_mul(selection.k)
        .ok_or(Error::ResourceLimit)?;
    let right = selection
        .k
        .checked_mul(selection.n)
        .ok_or(Error::ResourceLimit)?;
    let output = selection
        .m
        .checked_mul(selection.n)
        .ok_or(Error::ResourceLimit)?;
    Ok((left, right, output))
}

fn group_count(data_rows: usize) -> Result<usize> {
    data_rows
        .max(1)
        .checked_next_power_of_two()
        .ok_or(Error::ResourceLimit)
}

fn check_relation_shape(
    selection: &SelectedMatmulStep,
    step: &SharedTraceStep,
    external_values: &[Option<Vec<i64>>],
) -> Result<()> {
    let proof = &step.message.proof;
    let (left, right, output) = checked_dimensions(selection)?;
    if proof.lengths.inputs != vec![left, right]
        || proof.lengths.outputs != vec![output]
        || proof.lengths.raw != vec![output]
        || !proof.lengths.quotients.is_empty()
        || !proof.lengths.helpers.is_empty()
        || !proof.lengths.aux.is_empty()
    {
        return Err(Error::ShapeMismatch);
    }
    if proof.inputs.len() != 2
        || proof.outputs.len() != 1
        || proof.raw.len() != 1
        || !proof.quotients.is_empty()
        || !proof.helpers.is_empty()
        || !proof.aux.is_empty()
    {
        return Err(Error::ShapeMismatch);
    }
    if !proof.inputs[1].is_empty() {
        return Err(Error::ShapeMismatch);
    }
    if external_values.len() != 2 || external_values[0].is_some() {
        return Err(Error::StatementMismatch);
    }
    let external = external_values[1]
        .as_ref()
        .ok_or(Error::StatementMismatch)?;
    if external.len() != right {
        return Err(Error::ShapeMismatch);
    }
    let count = group_count(selection.m)?;
    if proof.inputs[0].len() != count || proof.outputs[0].len() != count {
        return Err(Error::ShapeMismatch);
    }
    for (position, opening) in proof.inputs[0].iter().enumerate() {
        if opening.index != position as u64 || opening.values.len() != selection.k {
            return Err(Error::ShapeMismatch);
        }
    }
    for (position, opening) in proof.outputs[0].iter().enumerate() {
        if opening.index != position as u64 || opening.values.len() != selection.n {
            return Err(Error::ShapeMismatch);
        }
    }
    let raw = proof.raw[0].get(0).ok_or(Error::ShapeMismatch)?;
    if raw.index != 0 || raw.values.len() != output.checked_mul(2).ok_or(Error::ResourceLimit)? {
        return Err(Error::ShapeMismatch);
    }
    if !raw.path.is_empty() {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn verify_oracle_authentication(
    step: &SharedTraceStep,
    selection: &SelectedMatmulStep,
) -> Result<OracleRegistry> {
    let mut registry = OracleRegistry::new();
    for (root, start, count) in &step.message.roots {
        registry.insert(*root, *start, *count)?;
    }
    let proof = &step.message.proof;
    let lengths = &proof.lengths;
    verify_group(
        &registry,
        &proof.inputs[0],
        *lengths.inputs.first().ok_or(Error::ShapeMismatch)?,
        selection.k,
        false,
    )?;
    verify_group(
        &registry,
        &proof.outputs[0],
        *lengths.outputs.first().ok_or(Error::ShapeMismatch)?,
        selection.n,
        false,
    )?;
    verify_group(
        &registry,
        &proof.raw[0],
        *lengths.raw.first().ok_or(Error::ShapeMismatch)?,
        0,
        true,
    )?;
    let used: BTreeSet<u64> = proof
        .inputs
        .iter()
        .chain(proof.outputs.iter())
        .chain(proof.raw.iter())
        .chain(proof.quotients.iter())
        .chain(proof.helpers.iter())
        .chain(proof.aux.iter())
        .flat_map(|group| group.iter().map(|opening| opening.oracle.0))
        .collect();
    let registered: BTreeSet<u64> = registry.ids().map(|id| id.0).collect();
    if used != registered {
        return Err(Error::StatementMismatch);
    }
    Ok(registry)
}

fn verify_asset_authentication(
    public: &ModelCommitmentPublic,
    asset: &AssetPlaintext<'_>,
) -> Result<()> {
    if asset.opening.asset_id != asset.meta.asset_id
        || asset.opening.start != asset.meta.start
        || asset.opening.root != asset.meta.root
        || asset.opening.row_count != asset.meta.rows
        || asset.opening.rows.len() != asset.meta.rows as usize
    {
        return Err(Error::StatementMismatch);
    }
    for (index, row) in asset.opening.rows.iter().enumerate() {
        if row.row != index as u64 || row.values.len() != asset.meta.row_width {
            return Err(Error::ShapeMismatch);
        }
    }
    public.verify_opening(asset.opening)
}

fn build_relation_public(
    step: &SharedTraceStep,
    selection: &SelectedMatmulStep,
    registry: OracleRegistry,
) -> Result<OpRelationPublic> {
    let proof = &step.message.proof;
    let input_oracles: Vec<OracleId> = proof
        .inputs
        .iter()
        .map(|group| {
            group
                .first()
                .map(|opening| opening.oracle)
                .unwrap_or(OracleId(u64::MAX))
        })
        .collect();
    let output_oracles: Vec<OracleId> = proof
        .outputs
        .iter()
        .map(|group| {
            group
                .first()
                .map(|opening| opening.oracle)
                .ok_or(Error::InvalidProof)
        })
        .collect::<Result<_>>()?;
    Ok(OpRelationPublic {
        opcode: OpCode::MatMul,
        config: selection.config.clone(),
        registry,
        input_oracles,
        output_oracles,
    })
}

fn push_openings(words: &mut Vec<u64>, tag: u64, groups: &[Vec<crate::proof::binding::OracleOpening>]) {
    words.push(tag);
    words.push(groups.len() as u64);
    for group in groups {
        words.push(group.len() as u64);
        for opening in group {
            words.push(opening.oracle.0);
            words.push(opening.index);
            words.push(opening.values.len() as u64);
            words.extend(opening.values.iter().copied());
            words.push(opening.path.len() as u64);
            for digest in &opening.path {
                push_digest(words, digest);
            }
        }
    }
}

fn add_word_count(total: &mut usize, value: usize) -> Result<()> {
    *total = total.checked_add(value).ok_or(Error::ResourceLimit)?;
    Ok(())
}

fn absorbed_word_count(
    selection: &SelectedMatmulStep,
    step: &SharedTraceStep,
    assets: &[AssetPlaintext<'_>],
) -> Result<usize> {
    let mut total = 3usize;
    add_word_count(&mut total, selection.config.len())?;
    add_word_count(
        &mut total,
        1 + step
            .message
            .roots
            .len()
            .checked_mul(9)
            .ok_or(Error::ResourceLimit)?,
    )?;
    add_word_count(&mut total, 1)?;
    for asset in assets {
        add_word_count(&mut total, 15)?;
        for row in &asset.opening.rows {
            add_word_count(&mut total, 3)?;
            add_word_count(&mut total, row.values.len())?;
            add_word_count(
                &mut total,
                row.path.len().checked_mul(4).ok_or(Error::ResourceLimit)?,
            )?;
        }
    }
    let groups = [
        &step.message.proof.inputs,
        &step.message.proof.outputs,
        &step.message.proof.raw,
        &step.message.proof.quotients,
        &step.message.proof.helpers,
        &step.message.proof.aux,
    ];
    for all in groups {
        add_word_count(&mut total, 2)?;
        for group in all {
            add_word_count(&mut total, 1)?;
            add_word_count(
                &mut total,
                group.len().checked_mul(4).ok_or(Error::ResourceLimit)?,
            )?;
            for opening in group {
                add_word_count(&mut total, opening.values.len())?;
                add_word_count(
                    &mut total,
                    opening.path.len().checked_mul(4).ok_or(Error::ResourceLimit)?,
                )?;
            }
        }
    }
    Ok(total)
}

fn absorb_step_words(
    selection: &SelectedMatmulStep,
    step: &SharedTraceStep,
    assets: &[AssetPlaintext<'_>],
) -> Result<Vec<u64>> {
    let total = absorbed_word_count(selection, step, assets)?;
    let mut words = Vec::new();
    words.try_reserve(total).map_err(|_| Error::ResourceLimit)?;
    words.push(SESSION_SCHEMA_VERSION);
    words.push(selection.step_index as u64);
    words.push(OpCode::MatMul as u64);
    words.extend(selection.config.iter().map(|value| *value as u64));
    words.push(step.message.roots.len() as u64);
    for (root, start, count) in &step.message.roots {
        push_digest(&mut words, root);
        push_digest(&mut words, start);
        words.push(*count);
    }
    words.push(assets.len() as u64);
    for asset in assets {
        words.push(asset.port as u64);
        words.push(asset.pool as u64);
        words.push(asset.meta.asset_id);
        words.push(asset.meta.row_width as u64);
        words.push(asset.meta.element_count);
        words.push(asset.meta.rows);
        push_digest(&mut words, &asset.meta.start);
        push_digest(&mut words, &asset.meta.root);
        words.push(asset.opening.rows.len() as u64);
        for row in &asset.opening.rows {
            words.push(row.row);
            words.push(row.values.len() as u64);
            words.extend(row.values.iter().copied());
            words.push(row.path.len() as u64);
            for digest in &row.path {
                push_digest(&mut words, digest);
            }
        }
    }
    push_openings(&mut words, GROUP_INPUTS, &step.message.proof.inputs);
    push_openings(&mut words, GROUP_OUTPUTS, &step.message.proof.outputs);
    push_openings(&mut words, GROUP_RAW, &step.message.proof.raw);
    push_openings(
        &mut words,
        GROUP_QUOTIENTS,
        &step.message.proof.quotients,
    );
    push_openings(&mut words, GROUP_HELPERS, &step.message.proof.helpers);
    push_openings(&mut words, GROUP_AUX, &step.message.proof.aux);
    Ok(words)
}

/// Derives the session value and one challenge from the absorbed words using
/// the existing chain interface. The challenge is drawn only after the step
/// that absorbs the words.
pub fn derive_transcript(
    parameters: &Digest,
    statement: &[u8],
    record: &[u8],
    words: &[u64],
) -> Result<SessionTranscript> {
    let mut chain = Chain::begin(REQUEST_CHAIN_LABEL, parameters, statement, record, None)?;
    let mut message = chain.message();
    for word in words {
        message.push_u64(*word);
    }
    let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    let rho = *chain.value();
    let challenge = chain.draw_f()?;
    Ok(SessionTranscript { rho, challenge })
}

/// Verifies one selected MatMul step: trusted plan and port checks, absorption
/// of the same plaintext that is authenticated and recomputed, Merkle
/// authentication and full relation recomputation.
pub fn verify_selected_step(
    public: &ModelCommitmentPublic,
    trace: &SharedTrace,
    plan: &TracePlan,
    selection: &SelectedMatmulStep,
    trusted_model: &TrustedModel<'_>,
) -> Result<SessionOutcome> {
    // The caller supplies trusted model metadata from the same static plan
    // source used by production. This establishes internal consistency with
    // that metadata before any asset or opening is used; it is not the deferred
    // external weight authorization decision.
    verify_model_anchor(
        public,
        trusted_model.anchor_id,
        trusted_model.model_id,
        trusted_model.parameters_digest,
        trusted_model.asset_specs,
    )?;
    let trusted = SelectedMatmulStep::select(plan, selection.step_index)?;
    if trusted != *selection {
        return Err(Error::StatementMismatch);
    }
    if trace.steps.len() != plan.steps.len() {
        return Err(Error::StatementMismatch);
    }
    let step = trace
        .steps
        .get(selection.step_index)
        .ok_or(Error::InvalidIndex)?;
    check_plan_step(step, selection)?;
    let mut assets = collect_assets(public, trace, selection, step)?;
    let external_values = build_external_values(step, &mut assets)?;
    check_relation_shape(selection, step, &external_values)?;

    let candidate = assemble_binding(public, plan, selection)?;
    // The same immutable step and asset opening references feed absorption,
    // Merkle authentication and the full recomputation below. No value is
    // re-read or replaced after absorption.
    let words = absorb_step_words(selection, step, &assets)?;
    let transcript = derive_transcript(
        &candidate.parameters_digest,
        &candidate.statement_bytes,
        &candidate.record_bytes,
        &words,
    )?;

    for asset in &assets {
        verify_asset_authentication(public, asset)?;
    }
    let registry = verify_oracle_authentication(step, selection)?;
    let relation_public = build_relation_public(step, selection, registry)?;
    verify_op_relation_with_external(
        &relation_public,
        &step.message.proof,
        &selection.config,
        &external_values,
    )?;

    Ok(SessionOutcome {
        step_index: selection.step_index,
        parameters: candidate.parameters_digest,
        statement_bytes: candidate.statement_bytes,
        record_bytes: candidate.record_bytes,
        rho: transcript.rho,
        challenge: transcript.challenge,
        assets: assets
            .iter()
            .map(|asset| AssetReference {
                port: asset.port,
                pool: asset.pool,
                asset_id: asset.meta.asset_id,
                start: asset.meta.start,
                root: asset.meta.root,
                rows: asset.meta.rows,
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::matmul::MatMulOp;
    use crate::proof::binding::{commit_rows_with_openings, OracleOpening};
    use crate::proof::model_commitment::{
        commit_assets, AssetStore, ModelAssetSpec, SharedTraceStep,
    };
    use crate::proof::plan::{PlanStep, TracePlan};
    use crate::proof::relations::{
        prove_op_relation_with_inputs, OpRelationProof, RelationLengths, RelationReceiptMessage,
    };

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "zkie-session-baseline-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn empty_step() -> SharedTraceStep {
        SharedTraceStep {
            message: RelationReceiptMessage {
                opcode: OpCode::Add as u16,
                config: Vec::new(),
                roots: Vec::new(),
                external_inputs: Vec::new(),
                proof: OpRelationProof {
                    lengths: RelationLengths {
                        inputs: Vec::new(),
                        outputs: Vec::new(),
                        raw: Vec::new(),
                        quotients: Vec::new(),
                        helpers: Vec::new(),
                        aux: Vec::new(),
                    },
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    raw: Vec::new(),
                    quotients: Vec::new(),
                    helpers: Vec::new(),
                    aux: Vec::new(),
                },
            },
            external: Vec::new(),
            sources: Vec::new(),
        }
    }

    struct Fixture {
        root: PathBuf,
        public: ModelCommitmentPublic,
        trace: SharedTrace,
        plan: TracePlan,
        selection: SelectedMatmulStep,
        rhs: Vec<i64>,
        specs: Vec<ModelAssetSpec>,
        anchor_id: Digest,
        model_id: Digest,
        parameters_digest: Digest,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn fixture() -> Fixture {
        let m = 2usize;
        let n = 2usize;
        let k = 2usize;
        let shift = 2u32;
        let lhs = vec![3i64, 1, 2, 4];
        let rhs = vec![1i64, 2, 3, 4];
        let op = MatMulOp { m, n, k, shift };
        let config = vec![m as i64, n as i64, k as i64, shift as i64, 0];
        let (relation, proof) =
            prove_op_relation_with_inputs(&op, &[lhs, rhs.clone()], &config, &[1]).unwrap();
        let mut roots = Vec::new();
        for id in relation.registry.ids() {
            let root = relation.registry.get(id).unwrap();
            roots.push((root.root, root.start, root.count));
        }
        let step = SharedTraceStep {
            message: RelationReceiptMessage {
                opcode: OpCode::MatMul as u16,
                config: config.clone(),
                roots,
                external_inputs: vec![1],
                proof,
            },
            external: vec![None, Some(0)],
            sources: vec![Some((0, 0)), None],
        };

        let root = temp_dir();
        let bytes: Vec<u8> = rhs
            .iter()
            .flat_map(|value| (*value as i32).to_le_bytes())
            .collect();
        fs::write(root.join("rhs_i32.bin"), bytes).unwrap();
        let specs = vec![ModelAssetSpec {
            asset_id: 0,
            path: "rhs_i32.bin".into(),
            element_count: rhs.len() as u64,
            row_width: n,
            encoding: "i32-le".into(),
        }];
        let commitment =
            commit_assets(&root, Digest([1; 32]), Digest([2; 32]), &specs).unwrap();
        let public = ModelCommitmentPublic::from_commitment(&commitment);
        let anchor_id = public.anchor.anchor_id;
        let assets = AssetStore::open(&commitment).unwrap().open_all().unwrap();

        let plan = TracePlan {
            steps: vec![
                PlanStep {
                    opcode: OpCode::Add,
                    config: Vec::new(),
                    inputs: Vec::new(),
                    outputs: 1,
                },
                PlanStep {
                    opcode: OpCode::MatMul,
                    config: vec![
                        PlanConfigValue::Literal(m as i64),
                        PlanConfigValue::Literal(n as i64),
                        PlanConfigValue::Literal(k as i64),
                        PlanConfigValue::Literal(shift as i64),
                        PlanConfigValue::Literal(0),
                    ],
                    inputs: vec![PlanSource::prev(0, 0), PlanSource::asset("rhs_i32.bin")],
                    outputs: 1,
                },
            ],
            argmax_steps: Vec::new(),
            final_output: None,
            model_version: 1,
            previous_object: None,
        };
        let selection = SelectedMatmulStep::select(&plan, 1).unwrap();
        let trace = SharedTrace {
            tokens: Vec::new(),
            assets,
            steps: vec![empty_step(), step],
            argmax: Vec::new(),
            output: Vec::new(),
        };
        Fixture {
            root,
            public,
            trace,
            plan,
            selection,
            rhs,
            specs,
            anchor_id,
            model_id: Digest([1; 32]),
            parameters_digest: Digest([2; 32]),
        }
    }

    fn verify(fixture: &Fixture) -> Result<SessionOutcome> {
        let trusted = TrustedModel {
            anchor_id: fixture.anchor_id,
            model_id: fixture.model_id,
            parameters_digest: fixture.parameters_digest,
            asset_specs: &fixture.specs,
        };
        verify_selected_step(
            &fixture.public,
            &fixture.trace,
            &fixture.plan,
            &fixture.selection,
            &trusted,
        )
    }

    fn assert_rejected(fixture: &Fixture) {
        assert!(verify(fixture).is_err());
    }

    fn rebuild_group(fixture: &mut Fixture, target: u8, rows: Vec<Vec<u64>>) {
        let step = &fixture.trace.steps[1];
        let original = match target {
            0 => step.message.proof.inputs[0][0].oracle,
            1 => step.message.proof.outputs[0][0].oracle,
            2 => step.message.proof.raw[0][0].oracle,
            _ => unreachable!(),
        };
        let mut registry = OracleRegistry::new();
        let (_, mut openings) =
            commit_rows_with_openings(&mut registry, &Digest([0x55; 32]), &rows).unwrap();
        for opening in &mut openings {
            opening.oracle = original;
        }
        let root = registry.get(OracleId(0)).unwrap().clone();
        fixture.trace.steps[1].message.roots[original.0 as usize] =
            (root.root, root.start, root.count);
        match target {
            0 => fixture.trace.steps[1].message.proof.inputs[0] = openings,
            1 => fixture.trace.steps[1].message.proof.outputs[0] = openings,
            2 => fixture.trace.steps[1].message.proof.raw[0] = openings,
            _ => unreachable!(),
        }
    }

    fn group_rows(openings: &[OracleOpening]) -> Vec<Vec<u64>> {
        openings.iter().map(|opening| opening.values.clone()).collect()
    }

    #[test]
    fn positive_selected_matmul_step_verifies() {
        let fixture = self::fixture();
        let outcome = verify(&fixture).unwrap();
        assert_eq!(outcome.step_index, 1);
        assert_eq!(outcome.assets.len(), 1);
        assert_eq!(outcome.assets[0].port, 1);
        assert_eq!(outcome.assets[0].asset_id, 0);
    }

    #[test]
    fn forged_asset_meta_root_is_rejected_before_use() {
        let mut fixture = self::fixture();
        fixture.public.assets[0].root.0[0] ^= 1;
        assert_rejected(&fixture);
    }

    #[test]
    fn forged_trusted_anchor_identity_is_rejected() {
        let mut fixture = self::fixture();
        fixture.anchor_id.0[0] ^= 1;
        assert_rejected(&fixture);
    }

    #[test]
    fn prover_and_verifier_derive_the_same_challenge() {
        let fixture = self::fixture();
        let step = &fixture.trace.steps[1];
        let assets = collect_assets(&fixture.public, &fixture.trace, &fixture.selection, step).unwrap();
        let words = absorb_step_words(&fixture.selection, step, &assets).unwrap();
        let candidate =
            assemble_binding(&fixture.public, &fixture.plan, &fixture.selection).unwrap();
        let prover = derive_transcript(
            &candidate.parameters_digest,
            &candidate.statement_bytes,
            &candidate.record_bytes,
            &words,
        )
        .unwrap();
        let verifier = derive_transcript(
            &candidate.parameters_digest,
            &candidate.statement_bytes,
            &candidate.record_bytes,
            &words,
        )
        .unwrap();
        assert_eq!(prover, verifier);
    }

    #[test]
    fn binding_assembly_outputs_are_used() {
        let fixture = self::fixture();
        let candidate =
            assemble_binding(&fixture.public, &fixture.plan, &fixture.selection).unwrap();
        let outcome = verify(&fixture).unwrap();
        assert_eq!(outcome.parameters, candidate.parameters_digest);
        assert_eq!(outcome.statement_bytes, candidate.statement_bytes);
        assert_eq!(outcome.record_bytes, candidate.record_bytes);
        let expected = Chain::begin(
            REQUEST_CHAIN_LABEL,
            &candidate.parameters_digest,
            &candidate.statement_bytes,
            &candidate.record_bytes,
            None,
        )
        .unwrap();
        assert_eq!(*expected.value(), candidate.first_query);
    }

    #[test]
    fn changing_an_absorbed_input_value_changes_the_words() {
        let fixture = self::fixture();
        let step = &fixture.trace.steps[1];
        let assets = collect_assets(&fixture.public, &fixture.trace, &fixture.selection, step).unwrap();
        let original = absorb_step_words(&fixture.selection, step, &assets).unwrap();

        let mut changed_step = step.clone();
        changed_step.message.proof.inputs[0][0].values[0] ^= 1;
        let changed_assets =
            collect_assets(&fixture.public, &fixture.trace, &fixture.selection, &changed_step).unwrap();
        let changed = absorb_step_words(&fixture.selection, &changed_step, &changed_assets).unwrap();
        assert_ne!(original, changed);

        let candidate =
            assemble_binding(&fixture.public, &fixture.plan, &fixture.selection).unwrap();
        let prover_changed = derive_transcript(
            &candidate.parameters_digest,
            &candidate.statement_bytes,
            &candidate.record_bytes,
            &changed,
        )
        .unwrap();
        let verifier_changed = derive_transcript(
            &candidate.parameters_digest,
            &candidate.statement_bytes,
            &candidate.record_bytes,
            &changed,
        )
        .unwrap();
        assert_eq!(prover_changed, verifier_changed);
    }

    #[test]
    fn changing_an_absorbed_asset_value_changes_the_words() {
        let fixture = self::fixture();
        let step = &fixture.trace.steps[1];
        let assets = collect_assets(&fixture.public, &fixture.trace, &fixture.selection, step).unwrap();
        let original = absorb_step_words(&fixture.selection, step, &assets).unwrap();

        let mut changed_trace = fixture.trace.clone();
        changed_trace.assets[0].rows[0].values[0] ^= 1;
        let changed_assets = collect_assets(
            &fixture.public,
            &changed_trace,
            &fixture.selection,
            &changed_trace.steps[1],
        )
        .unwrap();
        let changed = absorb_step_words(
            &fixture.selection,
            &changed_trace.steps[1],
            &changed_assets,
        )
        .unwrap();
        assert_ne!(original, changed);
    }

    #[test]
    fn authenticated_wrong_input_or_output_or_raw_is_rejected() {
        for target in 0..3u8 {
            let mut fixture = self::fixture();
            let rows = match target {
                0 => group_rows(&fixture.trace.steps[1].message.proof.inputs[0]),
                1 => group_rows(&fixture.trace.steps[1].message.proof.outputs[0]),
                _ => group_rows(&fixture.trace.steps[1].message.proof.raw[0]),
            };
            let mut rows = rows;
            rows[0][0] ^= 1;
            rebuild_group(&mut fixture, target, rows);
            let error = verify(&fixture).unwrap_err();
            assert_eq!(error.code(), "invalid_proof", "target {target}");
        }
    }

    #[test]
    fn rounded_output_mismatch_is_rejected() {
        let mut fixture = self::fixture();
        let mut rows = group_rows(&fixture.trace.steps[1].message.proof.outputs[0]);
        rows[0][0] = rows[0][0].wrapping_add(1);
        rebuild_group(&mut fixture, 1, rows);
        let error = verify(&fixture).unwrap_err();
        assert_eq!(error.code(), "invalid_proof");
    }

    #[test]
    fn authenticated_wrong_asset_value_is_rejected() {
        let mut fixture = self::fixture();
        let mut rhs = fixture.rhs.clone();
        rhs[0] ^= 1;
        let root = temp_dir();
        let bytes: Vec<u8> = rhs
            .iter()
            .flat_map(|value| (*value as i32).to_le_bytes())
            .collect();
        fs::write(root.join("rhs_i32.bin"), bytes).unwrap();
        let specs = vec![ModelAssetSpec {
            asset_id: 0,
            path: "rhs_i32.bin".into(),
            element_count: rhs.len() as u64,
            row_width: 2,
            encoding: "i32-le".into(),
        }];
        let commitment =
            commit_assets(&root, Digest([1; 32]), Digest([2; 32]), &specs).unwrap();
        fixture.public = ModelCommitmentPublic::from_commitment(&commitment);
        fixture.anchor_id = fixture.public.anchor.anchor_id;
        fixture.specs = specs;
        fixture.trace.assets = AssetStore::open(&commitment).unwrap().open_all().unwrap();
        let error = verify(&fixture).unwrap_err();
        assert_eq!(error.code(), "invalid_proof");
    }

    #[test]
    fn any_opening_position_is_checked() {
        for target in 0..3u8 {
            for position in 0..2usize {
                let mut fixture = self::fixture();
                let rows = match target {
                    0 => group_rows(&fixture.trace.steps[1].message.proof.inputs[0]),
                    1 => group_rows(&fixture.trace.steps[1].message.proof.outputs[0]),
                    _ => group_rows(&fixture.trace.steps[1].message.proof.raw[0]),
                };
                let row = if position == 0 { 0 } else { rows.len() - 1 };
                let width = rows[row].len() - 1;
                let mut rows = rows;
                rows[row][width] ^= 1;
                rebuild_group(&mut fixture, target, rows);
                assert_rejected(&fixture);
            }
        }
        for row in [0usize, 1] {
            let mut fixture = self::fixture();
            fixture.trace.assets[0].rows[row].values[0] ^= 1;
            assert_rejected(&fixture);
        }
    }

    #[test]
    fn plan_port_asset_and_root_tampering_is_rejected() {
        let mut fixture = self::fixture();
        fixture.trace.steps[1].message.config[0] += 1;
        assert_rejected(&fixture);

        let mut fixture = self::fixture();
        fixture.trace.steps[1].sources[0] = Some((1, 0));
        assert_rejected(&fixture);

        let mut fixture = self::fixture();
        fixture.trace.steps[1].external[1] = None;
        assert_rejected(&fixture);

        let mut fixture = self::fixture();
        fixture.trace.assets[0].root.0[0] ^= 1;
        assert_rejected(&fixture);

        let mut fixture = self::fixture();
        let root = &mut fixture.trace.steps[1].message.roots[0].0;
        root.0[0] ^= 1;
        assert_rejected(&fixture);
        let mut fixture = self::fixture();
        fixture.trace.assets[0].rows.swap(0, 1);
        assert_rejected(&fixture);
    }

    #[test]
    fn opening_position_and_malformed_shapes_are_rejected() {
        let mut fixture = self::fixture();
        fixture.trace.steps[1].message.proof.inputs[0].swap(0, 1);
        assert_rejected(&fixture);

        let mut fixture = self::fixture();
        fixture.trace.steps[1].message.proof.inputs[0][0].index = 7;
        assert_rejected(&fixture);

        let mut fixture = self::fixture();
        fixture.trace.steps[1].message.proof.inputs[0].pop();
        assert_rejected(&fixture);

        let mut fixture = self::fixture();
        let extra = fixture.trace.steps[1].message.proof.inputs[0][0].clone();
        fixture.trace.steps[1].message.proof.inputs[0].push(extra);
        assert_rejected(&fixture);

        let mut fixture = self::fixture();
        fixture.trace.steps[1].message.proof.inputs[0][0].values.pop();
        assert_rejected(&fixture);

        let mut fixture = self::fixture();
        fixture.trace.steps[1].message.proof.inputs[0][0].values.push(0);
        assert_rejected(&fixture);

        let mut fixture = self::fixture();
        fixture.trace.steps[1].message.proof.raw[0][0].values.pop();
        assert_rejected(&fixture);

        let mut fixture = self::fixture();
        fixture.trace.steps[1].message.proof.raw[0][0].values.push(0);
        assert_rejected(&fixture);

        let mut fixture = self::fixture();
        fixture.trace.steps[1].message.proof.lengths.inputs[0] += 1;
        assert_rejected(&fixture);
    }

    #[test]
    fn malformed_plan_dimensions_are_rejected() {
        let fixture = self::fixture();
        let mut plan = fixture.plan.clone();
        plan.steps[1].config[0] = PlanConfigValue::Literal(0);
        assert!(SelectedMatmulStep::select(&plan, 1).is_err());

        let mut plan = fixture.plan.clone();
        plan.steps[1].config[4] = PlanConfigValue::Literal(1);
        assert!(SelectedMatmulStep::select(&plan, 1).is_err());

        let mut plan = fixture.plan.clone();
        plan.steps[1].opcode = OpCode::Add;
        assert!(SelectedMatmulStep::select(&plan, 1).is_err());

        let mut fixture = self::fixture();
        fixture.public.assets[0].element_count = 1u64 << 40;
        assert_rejected(&fixture);
    }
}

