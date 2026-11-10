//! S7a callable adapter: the trusted gpt2 context and the lm_head asset identity.
//!
//! This module reuses the production verification checks (policy selection, model
//! commitment, anchor rebuild, asset specification checks) and exposes the
//! lm_head MatMul step and a read-only row path. It creates no commitment and
//! does not change `verify_request`; the chain that consumes these values is a
//! later step.
//!
//! The weight file is opened read-only through the existing mmap reader; only
//! the requested row is copied. The file path itself is not a trusted source:
//! the identity comes from the compiled asset specification, the public anchor
//! record and the policy anchor id, all checked here.

use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use zkie_core::compute::{MemoryGovernor, MemoryRequest};
use zkie_core::error::{Error, Result};
use zkie_core::storage::TensorReader;
use zkie_core::types::{DType, Digest, TensorSpec};
use zkie_engine::memory::BudgetedMemoryGovernor;
use zkie_engine::storage::mmap::MmapTensorReader;
use zkie_ops::op::OpCode;
use zkie_ops::proof::binding::oracle_start;
use zkie_ops::proof::model_commitment::{
    asset_table_identity, expected_asset_rows, verify_model_anchor, ModelAssetSpec,
    ModelCommitmentPublic,
};
use zkie_ops::proof::plan::{PlanConfigValue, PlanSource, TracePlan};
use zkie_ops::proof::session_baseline::{selected_plan_identifier, SelectedMatmulStep};

use crate::policy::{self, TrustedRequest};
use crate::provider::StatementData;

const LM_HEAD_SUFFIX: &str = "gpt2/weights/lm_head_w_i32.bin";

/// The lm_head weight table's identity in the public anchor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LmHeadAsset {
    pub asset_id: u64,
    pub path: String,
    pub table_identity: Digest,
    pub element_count: u64,
    pub row_width: u64,
    pub rows: u64,
    pub start: Digest,
    pub root: Digest,
    pub leaf_root: Digest,
}

/// The trace step that consumes the lm_head weight table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LmHeadStep {
    pub index: usize,
    pub opcode: u16,
    pub config: Vec<i64>,
    pub activation_port: usize,
    pub activation_step: usize,
    pub activation_output: usize,
    pub weight_port: usize,
    pub weight_path: String,
    pub output_index: usize,
}

/// The trusted context consumed by the later chain steps.
#[derive(Clone, Debug)]
pub struct LmHeadContext {
    pub policy: TrustedRequest,
    pub statement: StatementData,
    pub anchor_id: Digest,
    pub model_id: Digest,
    pub asset_root: PathBuf,
    pub asset: LmHeadAsset,
    pub matmul: LmHeadStep,
    pub plan_identifier: Digest,
    pub plan_steps: usize,
}

fn deserialize_exact<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    let mut cursor = Cursor::new(bytes);
    let value = bincode::deserialize_from(&mut cursor).map_err(|_| Error::InvalidEncoding)?;
    if cursor.position() != bytes.len() as u64 {
        return Err(Error::ArtifactIntegrityMismatch);
    }
    Ok(value)
}

/// The asset root that the public case file declares for the gpt2 assets. This
/// mirrors the production case parser: the `[assets] root` value, then the
/// `models` directory below it.
fn case_asset_root(case: &Path) -> Result<PathBuf> {
    let text = fs::read_to_string(case).map_err(|_| Error::AssetMismatch)?;
    let mut root = PathBuf::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("root = ") {
            root = PathBuf::from(rest.trim().trim_matches('"'));
        }
    }
    if root.as_os_str().is_empty() {
        return Err(Error::InvalidConfig);
    }
    Ok(root.join("models"))
}

fn literal_config(config: &[PlanConfigValue]) -> Result<Vec<i64>> {
    config
        .iter()
        .map(|value| match value {
            PlanConfigValue::Literal(number) => Ok(*number),
            PlanConfigValue::AssetI64 { .. } => Err(Error::UnsupportedPrimitive),
        })
        .collect()
}

/// Finds the MatMul step that consumes the lm_head weight table and records
/// the activation source and output. The plan is compiled into the verifier;
/// this function reads it, it does not accept a caller-supplied step list.
pub fn find_lm_head_matmul(plan: &TracePlan) -> Result<LmHeadStep> {
    for (index, step) in plan.steps.iter().enumerate() {
        if step.opcode != OpCode::MatMul {
            continue;
        }
        let weight = step.inputs.iter().enumerate().find_map(|(port, input)| match input {
            PlanSource::Asset { path, .. } if path.ends_with(LM_HEAD_SUFFIX) => {
                Some((port, path.clone()))
            }
            _ => None,
        });
        let Some((weight_port, weight_path)) = weight else {
            continue;
        };
        let mut activation: Option<(usize, usize, usize)> = None;
        for (port, input) in step.inputs.iter().enumerate() {
            if let PlanSource::Prev {
                step,
                output,
                layout: _,
            } = input
            {
                if activation.is_some() {
                    return Err(Error::UnsupportedPrimitive);
                }
                activation = Some((port, *step, *output));
            }
        }
        let (activation_port, activation_step, activation_output) =
            activation.ok_or(Error::StatementMismatch)?;
        if step.outputs != 1 {
            return Err(Error::UnsupportedPrimitive);
        }
        return Ok(LmHeadStep {
            index,
            opcode: step.opcode as u16,
            config: literal_config(&step.config)?,
            activation_port,
            activation_step,
            activation_output,
            weight_port,
            weight_path,
            output_index: 0,
        });
    }
    Err(Error::StatementMismatch)
}

fn lm_head_spec<'a>(specs: &'a [ModelAssetSpec], path: &str) -> Result<&'a ModelAssetSpec> {
    specs
        .iter()
        .find(|spec| spec.path == path)
        .ok_or(Error::AssetMismatch)
}

/// Loads the trusted context for the gpt2 lm_head LIN relation from the same
/// policy, public bundle and statement that the production verifier uses. All
/// checks reuse the production rebuilding path; nothing is committed here.
pub fn load_gpt2_lm_head_context(
    case_path: &Path,
    policy_path: &Path,
    public_bundle_dir: &Path,
    statement_path: &Path,
    expected_model_id: Digest,
) -> Result<LmHeadContext> {
    let asset_root = case_asset_root(case_path)?;
    let policy_bytes = fs::read(policy_path).map_err(|_| Error::IncompleteArtifact)?;
    let policy = policy::parse(&policy_bytes)?;
    let commitment_bytes = fs::read(public_bundle_dir.join("model-commitment.bin"))
        .map_err(|_| Error::IncompleteArtifact)?;
    let public: ModelCommitmentPublic =
        deserialize_exact(&commitment_bytes)?;
    let statement_bytes = fs::read(statement_path).map_err(|_| Error::IncompleteArtifact)?;
    let statement: StatementData = deserialize_exact(&statement_bytes)?;
    let trusted = policy::select(
        &policy,
        &statement.case_id,
        &statement.model,
        statement.tokens.len(),
    )?;
    if trusted.model != "gpt2" {
        return Err(Error::UnsupportedPrimitive);
    }
    let model_id = trusted.model_id;
    if expected_model_id != model_id
        || statement.model_commitment != model_id
        || public.anchor.model_commitment != model_id
    {
        return Err(Error::StatementMismatch);
    }
    let plan = zkie_models_gpt2::plan::trace_plan(statement.tokens.len())?;
    if zkie_models_gpt2::plan::model_commitment(&statement.tokens)? != model_id {
        return Err(Error::StatementMismatch);
    }
    let parameters = zkie_models_gpt2::plan::parameters_digest()?;
    let specs = zkie_models_gpt2::plan::asset_specs();
    verify_model_anchor(&public, trusted.anchor_id, model_id, parameters, &specs)?;
    let matmul = find_lm_head_matmul(&plan)?;
    let selection = SelectedMatmulStep::select(&plan, matmul.index)?;
    let plan_identifier = selected_plan_identifier(&public, &plan, &selection)?;
    let spec = lm_head_spec(&specs, &matmul.weight_path)?;
    let meta = public
        .assets
        .iter()
        .find(|asset| asset.path == spec.path)
        .ok_or(Error::AssetMismatch)?;
    let identity = asset_table_identity(spec);
    if meta.asset_id != spec.asset_id
        || meta.element_count != spec.element_count
        || meta.row_width != spec.row_width
        || meta.rows != expected_asset_rows(spec)?
        || meta.start != oracle_start(&identity)
        || meta.root != meta.leaf_record.leaf_root
        || meta.leaf_record.table_identity != identity
    {
        return Err(Error::StatementMismatch);
    }
    let asset = LmHeadAsset {
        asset_id: meta.asset_id,
        path: meta.path.clone(),
        table_identity: identity,
        element_count: meta.element_count,
        row_width: meta.row_width as u64,
        rows: meta.rows,
        start: meta.start,
        root: meta.root,
        leaf_root: meta.leaf_record.leaf_root,
    };
    Ok(LmHeadContext {
        policy: trusted,
        statement,
        anchor_id: public.anchor.anchor_id,
        model_id,
        asset_root,
        asset,
        matmul,
        plan_identifier,
        plan_steps: plan.steps.len(),
    })
}

impl LmHeadContext {
    /// Reads one weight row read-only through the mmap path. Only the requested
    /// row is copied; the row index is checked against the real element count.
    pub fn read_lm_head_row(&self, row: u64) -> Result<Vec<i64>> {
        if self.asset.row_width == 0 {
            return Err(Error::InvalidConfig);
        }
        let actual_rows = self.asset.element_count.div_ceil(self.asset.row_width);
        if row >= actual_rows {
            return Err(Error::InvalidIndex);
        }
        let bytes = self
            .asset
            .element_count
            .checked_mul(4)
            .ok_or(Error::ResourceLimit)?;
        let governor = BudgetedMemoryGovernor::new(bytes);
        let permit = governor.acquire(&MemoryRequest {
            mapped_window_bytes: bytes,
            ..MemoryRequest::default()
        })?;
        let reader = MmapTensorReader::open(
            self.asset_root.join(&self.asset.path),
            TensorSpec::new(DType::I32, vec![self.asset.element_count]),
            self.asset.start,
            permit,
        )?;
        if reader.mapped_len() as u64 != bytes {
            return Err(Error::ShapeMismatch);
        }
        let width = usize::try_from(self.asset.row_width).map_err(|_| Error::ResourceLimit)?;
        let start = row
            .checked_mul(self.asset.row_width)
            .ok_or(Error::ResourceLimit)?;
        let mut buffer = vec![0u8; width.checked_mul(4).ok_or(Error::ResourceLimit)?];
        reader.read_range(start, width, &mut buffer)?;
        buffer
            .chunks_exact(4)
            .map(|chunk| {
                let bytes: [u8; 4] = chunk.try_into().map_err(|_| Error::InvalidEncoding)?;
                Ok(i32::from_le_bytes(bytes) as i64)
            })
            .collect()
    }
}
