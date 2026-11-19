//! Production pipeline provider for the three shared model traces.
//!
//! Commit publishes the shared model commitment and the public case inputs.
//! Prove re-opens the commitment, executes the shared trace and writes the
//! proof. Verify reads only the public commitment/proof/inputs.

use std::fs;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use zkie_core::artifact::{
    digest_bytes, ArtifactFile, ArtifactKind, ArtifactManifest, PrivacyMode, Sha256Stream,
};
use zkie_core::error::{Error, Result};
use zkie_core::progress::{ProgressEvent, ProgressKind, ProgressSink, RunOutcome};
use zkie_core::types::Digest;
use zkie_engine::artifact::{load_manifest, ArtifactPublisher};
use zkie_models_gemma3::executor as gemma3;
use zkie_models_gpt2::executor as gpt2;
use zkie_models_timesfm::executor as timesfm;
use zkie_ops::pipeline::{
    CommitReceipt, CommitRequest, OutputBindingMode, PipelineContext, PipelineProvider,
    ProofReceipt, ProveRequest, VerifyReceipt, VerifyRequest,
};
use zkie_ops::proof::model_commitment::{
    verify_model_anchor_with_tables, verify_model_anchor_with_tables_on_identity,
    verify_shared_trace, ModelCommitmentPublic, SharedTrace,
};
use zkie_ops::proof::asset_key::{
    prove_asset_key, verify_asset_key, AssetKeyColumns, ASSET_KEY_BUNDLE,
};
use zkie_ops::proof::asset_table::{
    decode_asset_table_randomness_set, encode_asset_table_randomness_set,
    sample_asset_table_randomness, sample_asset_table_randomness_public, AssetTableInput, AssetTableMeta, AssetTableParams,
};
use zkie_ops::proof::consumer_proof::ConsumerProofFile;
use zkie_ops::proof::consumer_request::ConsumerSharedPlan;
use zkie_ops::proof::consumer_session::verify_consumer_request;
use crate::norm_parent::{
    build_norm_plan, build_norm_segment, build_segment_plan, check_authenticated_bounds,
    final_norm_rows, read_anchored_params, read_rstd_values, segment_session_seed, segment_x_seed,
};
use crate::norm_segment_provider::{
    build_param_tables, build_segment_table, configuration_for, lm_head_asset, plan_identifier,
    prove_segment_bytes, rstd_table_view, segment_identity_binding, segment_table_key,
    segment_table_source_identity, segment_work_dir, verify_segment_bytes, NormSegmentCarrier,
    SegmentTableIdentity, NORM_SEGMENT_CARRIER_FILE, NORM_SEGMENT_CARRIER_VERSION,
};

const PROTOCOL_ID: Digest = Digest([0xA1; 32]);
const PARAMETERS_DIGEST: Digest = Digest([0xA2; 32]);
const MODEL_CONFIG_DIGEST: Digest = Digest([0xA3; 32]);
const QUANTIZATION_DIGEST: Digest = Digest([0xA4; 32]);
const TABLES_DIGEST: Digest = Digest([0xA5; 32]);
const BACKEND_ID: u32 = 0x5A4B_4945;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub asset_root: String,
    pub case_root: String,
    pub case_id: String,
    pub model: String,
    pub seq_tokens: usize,
    pub tokens: Vec<u32>,
    pub input: Vec<i32>,
    pub expected_argmax: Vec<u32>,
    pub expected_output: Vec<i32>,
}

impl ProviderConfig {
    /// The ledger directory of one column table identity under this provider's
    /// asset directory. The ledger is a new directory; the asset bodies are
    /// never touched.
    pub fn column_ledger_dir(&self, table_root: Digest) -> PathBuf {
        zkie_ops::proof::column_session::column_ledger_dir(Path::new(&self.asset_root), table_root)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StatementData {
    pub case_id: String,
    pub model: String,
    pub model_commitment: Digest,
    pub tokens: Vec<u32>,
    pub expected_argmax: Vec<u32>,
    pub expected_output: Vec<i32>,
    /// The segment's own output head table identity. The policy block names
    /// the consumer table; this one rides the statement and its digest is
    /// absorbed into the segment session derivation.
    pub segment_identity: Option<SegmentTableIdentity>,
    /// The timesfm norm-only segment declaration. It is bound into the
    /// segment session seed; gpt2 keeps false.
    pub norm_only: bool,
}

fn read_i32(path: &Path) -> Result<Vec<i32>> {
    let bytes = fs::read(path).map_err(|_| Error::AssetMismatch)?;
    if bytes.len() % 4 != 0 {
        return Err(Error::InvalidEncoding);
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap_or([0; 4])))
        .collect())
}

fn deserialize_exact<T: DeserializeOwned>(bytes: &[u8], trailing_error: Error) -> Result<T> {
    let mut cursor = Cursor::new(bytes);
    let value = bincode::deserialize_from(&mut cursor).map_err(|_| Error::InvalidEncoding)?;
    if cursor.position() != bytes.len() as u64 {
        return Err(trailing_error);
    }
    Ok(value)
}

fn read_proof_exact(path: &Path, memory_budget_bytes: u64) -> Result<SharedTrace> {
    crate::bounded_trace::read_file(path, memory_budget_bytes)
}

fn parse_case(case: &Path) -> Result<ProviderConfig> {
    let text = fs::read_to_string(case).map_err(|_| Error::AssetMismatch)?;
    let mut case_root = PathBuf::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("root = ") {
            case_root = PathBuf::from(rest.trim_matches('"'));
        }
    }
    if case_root.as_os_str().is_empty() {
        return Err(Error::InvalidConfig);
    }
    let requested = std::env::var("ZKIE_CASE_ID").ok();
    let mut selected: Option<(String, String, usize)> = None;
    for block in text.split("[[case]]").skip(1) {
        let mut id = String::new();
        let mut input = String::new();
        let mut seq_tokens = 0usize;
        for line in block.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("id = ") {
                id = rest.trim().trim_matches('"').to_owned();
            } else if let Some(rest) = trimmed.strip_prefix("seq_tokens = ") {
                seq_tokens = rest.trim().parse().unwrap_or(0);
            } else if let Some(rest) = trimmed.strip_prefix("patch_tokens = ") {
                seq_tokens = rest.trim().parse().unwrap_or(0);
            } else if let Some(rest) = trimmed.strip_prefix("input = {") {
                if let Some(position) = rest.find("path = \"") {
                    let after = &rest[position + 8..];
                    if let Some(end) = after.find('"') {
                        input = after[..end].to_owned();
                    }
                }
            }
        }
        if id.is_empty() || input.is_empty() {
            continue;
        }
        let matches = requested.as_deref().map_or(true, |wanted| wanted == id);
        if matches && selected.is_none() {
            selected = Some((id, input, seq_tokens));
        }
        if requested.is_none() {
            break;
        }
    }
    let (case_id, input, seq_tokens) = selected.ok_or(Error::InvalidConfig)?;
    let model = case_id.split('-').next().unwrap_or("gpt2").to_owned();
    let supported = match model.as_str() {
        "gpt2" | "gemma3" => matches!(seq_tokens, 16 | 512),
        "timesfm" => seq_tokens == 16,
        _ => false,
    };
    if !supported {
        return Err(Error::UnsupportedPrimitive);
    }
    let input_path = case_root.join(&input);
    let (asset_root, tokens, input_values, expected_argmax, expected_output) =
        if model == "timesfm" {
            let input_values = read_i32(&input_path)?;
            if input_values.len() != 512 {
                return Err(Error::InvalidConfig);
            }
            let tokens = input_values.iter().map(|value| *value as u32).collect();
            let gt_path = case_root.join("cases/timesfm/gt_int_output_i32.bin");
            let expected_output = read_i32(&gt_path)?;
            (
                case_root.clone(),
                tokens,
                input_values,
                Vec::new(),
                expected_output,
            )
        } else {
            let values = read_i32(&input_path)?;
            if values.len() != seq_tokens {
                return Err(Error::InvalidConfig);
            }
            let tokens: Vec<u32> = values.into_iter().map(|value| value as u32).collect();
            let gt_path =
                case_root.join(format!("cases/{model}/gt_int_argmax_{seq_tokens}_i32.bin"));
            let expected_argmax: Vec<u32> = read_i32(&gt_path)?
                .into_iter()
                .map(|value| value as u32)
                .collect();
            let asset_root = if model == "gpt2" {
                case_root.join("models")
            } else {
                case_root.clone()
            };
            (asset_root, tokens, Vec::new(), expected_argmax, Vec::new())
        };
    Ok(ProviderConfig {
        asset_root: asset_root.to_string_lossy().into_owned(),
        case_root: case_root.to_string_lossy().into_owned(),
        case_id,
        model,
        seq_tokens,
        tokens,
        input: input_values,
        expected_argmax,
        expected_output,
    })
}

fn commit_manifest(model: Digest, files: Vec<ArtifactFile>) -> ArtifactManifest {
    ArtifactManifest {
        artifact_schema_version: 1,
        artifact_kind: ArtifactKind::Commit,
        protocol_id: PROTOCOL_ID,
        protocol_version: 1,
        backend_id: BACKEND_ID,
        backend_format_version: 1,
        model_commitment_id: model,
        parameters_digest: PARAMETERS_DIGEST,
        model_config_digest: MODEL_CONFIG_DIGEST,
        quantization_digest: QUANTIZATION_DIGEST,
        tables_digest: TABLES_DIGEST,
        privacy_mode: PrivacyMode::Hidden,
        files,
    }
}

/// Loads one artifact manifest, requires its kind and the fixed protocol
/// identity, and returns it. The artifact root's `COMPLETE` marker must carry
/// the manifest digest.
fn load_checked_manifest(root: &Path, kind: ArtifactKind) -> Result<ArtifactManifest> {
    let manifest = load_manifest(root)?;
    let privacy_ok = match kind {
        ArtifactKind::Commit => manifest.privacy_mode == PrivacyMode::Hidden,
        _ => manifest.privacy_mode == PrivacyMode::Public,
    };
    if manifest.artifact_kind != kind || !privacy_ok {
        return Err(Error::StatementMismatch);
    }
    if manifest.protocol_id != PROTOCOL_ID
        || manifest.protocol_version != 1
        || manifest.backend_id != BACKEND_ID
        || manifest.backend_format_version != 1
        || manifest.parameters_digest != PARAMETERS_DIGEST
        || manifest.model_config_digest != MODEL_CONFIG_DIGEST
        || manifest.quantization_digest != QUANTIZATION_DIGEST
        || manifest.tables_digest != TABLES_DIGEST
    {
        return Err(Error::UnsupportedVersion);
    }
    Ok(manifest)
}

/// Reads one manifest-listed file and requires its length and digest. The
/// relative path is joined to the artifact root; the caller never reads a file
/// the manifest does not name.
fn checked_artifact_file(
    root: &Path,
    manifest: &ArtifactManifest,
    relative_path: &str,
) -> Result<Vec<u8>> {
    let entry = manifest
        .find_file(relative_path)
        .ok_or(Error::IncompleteArtifact)?;
    let bytes = fs::read(root.join(relative_path)).map_err(|_| Error::IncompleteArtifact)?;
    if bytes.len() as u64 != entry.byte_length || digest_bytes(&bytes) != entry.content_digest {
        return Err(Error::ArtifactIntegrityMismatch);
    }
    Ok(bytes)
}

/// Compares bytes already read with one manifest entry.
fn check_manifest_bytes(
    manifest: &ArtifactManifest,
    relative_path: &str,
    bytes: &[u8],
) -> Result<()> {
    let entry = manifest
        .find_file(relative_path)
        .ok_or(Error::IncompleteArtifact)?;
    if bytes.len() as u64 != entry.byte_length || digest_bytes(bytes) != entry.content_digest {
        return Err(Error::ArtifactIntegrityMismatch);
    }
    Ok(())
}

/// Requires one proof input file to be the manifest's named file at the
/// artifact root and its bytes to match the entry.
fn check_proof_input(
    path: &Path,
    root: &Path,
    manifest: &ArtifactManifest,
    relative_path: &str,
    bytes: &[u8],
) -> Result<()> {
    if path.parent() != Some(root)
        || path.file_name().and_then(|name| name.to_str()) != Some(relative_path)
    {
        return Err(Error::StatementMismatch);
    }
    check_manifest_bytes(manifest, relative_path, bytes)
}

/// Hashes a large manifest-listed file in bounded chunks without loading it.
fn checked_large_file(path: &Path, entry: &ArtifactFile) -> Result<()> {
    let metadata = fs::metadata(path).map_err(|_| Error::IncompleteArtifact)?;
    if metadata.len() != entry.byte_length {
        return Err(Error::ArtifactIntegrityMismatch);
    }
    let mut file = fs::File::open(path).map_err(|_| Error::IncompleteArtifact)?;
    let mut stream = Sha256Stream::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer).map_err(|_| Error::Io)?;
        if read == 0 {
            break;
        }
        stream.update(&buffer[..read]);
    }
    if stream.finish() != entry.content_digest {
        return Err(Error::ArtifactIntegrityMismatch);
    }
    Ok(())
}

fn proof_manifest(model: Digest, files: Vec<ArtifactFile>) -> ArtifactManifest {
    ArtifactManifest {
        artifact_schema_version: 1,
        artifact_kind: ArtifactKind::Proof,
        protocol_id: PROTOCOL_ID,
        protocol_version: 1,
        backend_id: BACKEND_ID,
        backend_format_version: 1,
        model_commitment_id: model,
        parameters_digest: PARAMETERS_DIGEST,
        model_config_digest: MODEL_CONFIG_DIGEST,
        quantization_digest: QUANTIZATION_DIGEST,
        tables_digest: TABLES_DIGEST,
        privacy_mode: PrivacyMode::Public,
        files,
    }
}

/// The registered first projection table: its trusted parameters, its asset
/// specification and the derived key-stage columns.
fn gpt2_key_columns() -> Result<(
    AssetTableParams,
    zkie_ops::proof::model_commitment::ModelAssetSpec,
    AssetKeyColumns,
)> {
    const FIRST_PROJECTION: &str = "gpt2/weights/L0_k_w_i32.bin";
    let params = zkie_models_gpt2::plan::projection_table_params(FIRST_PROJECTION)
        .ok_or(Error::UnsupportedPrimitive)?;
    let spec = zkie_models_gpt2::plan::asset_specs()
        .into_iter()
        .find(|spec| spec.path == FIRST_PROJECTION)
        .ok_or(Error::UnsupportedPrimitive)?;
    let derived = zkie_models_gpt2::plan::derive_table_plan(
        &zkie_models_gpt2::plan::TABLE_PLAN_CANDIDATE,
    )?;
    let columns = AssetKeyColumns {
        key_columns: derived.one_time_columns,
        instance_columns: derived.total_columns,
        k_lo: zkie_models_gpt2::plan::TABLE_PLAN_CANDIDATE.k_lo,
        lin_check: derived.lin_check,
    };
    Ok((params, spec, columns))
}

/// The registered first projection table of TimesFM: its trusted parameters,
/// its asset specification and the derived key-stage columns.
fn timesfm_key_columns() -> Result<(
    AssetTableParams,
    zkie_ops::proof::model_commitment::ModelAssetSpec,
    AssetKeyColumns,
)> {
    let path = zkie_models_timesfm::plan::FIRST_PROJECTION;
    let params = zkie_models_timesfm::plan::projection_table_params(path)
        .ok_or(Error::UnsupportedPrimitive)?;
    let spec = zkie_models_timesfm::plan::asset_specs()
        .into_iter()
        .find(|spec| spec.path == path)
        .ok_or(Error::UnsupportedPrimitive)?;
    let derived = zkie_models_timesfm::plan::derive_table_plan(
        &zkie_models_timesfm::plan::TABLE_PLAN_CANDIDATE,
    )?;
    let columns = AssetKeyColumns {
        key_columns: derived.one_time_columns,
        instance_columns: derived.total_columns,
        k_lo: zkie_models_timesfm::plan::TABLE_PLAN_CANDIDATE.k_lo,
        lin_check: derived.lin_check,
    };
    Ok((params, spec, columns))
}

/// Samples the commit table material. The final-norm parameter tables use a
/// plan-fixed public seed, so their roots are a reproducible function of the
/// trusted asset and the public derivation; the projection table keeps fresh
/// material.
fn sample_production_tables(params: &[AssetTableParams]) -> Result<Vec<AssetTableInput>> {
    let mut tables = Vec::with_capacity(params.len());
    for params in params {
        let randomness = if params.path.ends_with("ln_f_w_i32.bin")
            || params.path.ends_with("ln_f_b_i32.bin")
            || params.path.ends_with("L0_lnw_i32.bin")
        {
            sample_asset_table_randomness_public(params, norm_param_seed(params))?
        } else {
            sample_asset_table_randomness(params)?
        };
        tables.push(AssetTableInput {
            params: params.clone(),
            randomness,
        });
    }
    Ok(tables)
}

fn norm_param_seed(params: &AssetTableParams) -> zkie_core::types::Digest {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"zkie/norm-param-table/v1");
    bytes.extend_from_slice(&params.asset_id.to_le_bytes());
    bytes.extend_from_slice(params.path.as_bytes());
    zkie_core::artifact::digest_bytes(&bytes)
}

fn trusted_table_params(model: &str) -> Result<Vec<AssetTableParams>> {
    match model {
        "gpt2" => zkie_models_gpt2::plan::asset_table_params(),
        "gemma3" => Ok(Vec::new()),
        "timesfm" => zkie_models_timesfm::plan::asset_table_params(),
        _ => Err(Error::UnsupportedPrimitive),
    }
}

fn build_commitment(
    config: &ProviderConfig,
    tables: &[AssetTableInput],
) -> Result<zkie_ops::proof::model_commitment::ModelCommitment> {
    match config.model.as_str() {
        "gpt2" => gpt2::build_commitment(Path::new(&config.asset_root), &config.tokens, tables),
        "gemma3" => {
            if !tables.is_empty() {
                return Err(Error::InvalidConfig);
            }
            gemma3::build_commitment(Path::new(&config.case_root), &config.tokens)
        }
        "timesfm" => timesfm::build_commitment(
            Path::new(&config.case_root),
            &config.input.iter().map(|value| *value as i64).collect::<Vec<_>>(),
            tables,
        ),
        _ => Err(Error::UnsupportedPrimitive),
    }
}

fn execute_shared(
    config: &ProviderConfig,
    commitment: &zkie_ops::proof::model_commitment::ModelCommitment,
    tables: &[AssetTableInput],
    progress: &dyn ProgressSink,
) -> Result<(SharedTrace, Option<ConsumerProofFile>)> {
    match config.model.as_str() {
        "gpt2" => gpt2::execute_shared_with_progress_public_consumer(
            commitment,
            &config.tokens.iter().map(|value| *value as i64).collect::<Vec<_>>(),
            tables,
            Some(progress),
        )
        .map(|(trace, file)| (trace, Some(file))),
        "gemma3" => gemma3::execute_shared_with_progress_public(
            commitment,
            &config.tokens.iter().map(|value| *value as i64).collect::<Vec<_>>(),
            Some(progress),
        )
        .map(|trace| (trace, None)),
        "timesfm" => timesfm::execute_shared_with_progress_public_consumer(
            commitment,
            &config.input.iter().map(|value| *value as i64).collect::<Vec<_>>(),
            tables,
            Some(progress),
        )
        .map(|(trace, file)| (trace, Some(file))),
        _ => Err(Error::UnsupportedPrimitive),
    }
}


/// Proves the final norm segment and returns its public carrier. The segment
/// is assembled only at the accepted hidden session's position count.
fn prove_norm_segment(
    config: &ProviderConfig,
    commitment: &zkie_ops::proof::model_commitment::ModelCommitment,
    tables: &[AssetTableInput],
    trace: &SharedTrace,
    output_dir: &Path,
    ctx: &mut PipelineContext<'_>,
) -> Result<Option<(Vec<u8>, SegmentTableIdentity)>> {
    let positions = config.tokens.len();
    if positions != zkie_ops::proof::output_head_lin::OUTPUT_HEAD_POSITIONS {
        return Ok(None);
    }
    let model = commitment.anchor.record.model_commitment;
    let public = ModelCommitmentPublic::from_commitment(commitment);
    let trace_plan = zkie_models_gpt2::plan::trace_plan(positions)?;
    let matmul = crate::s7_context::find_lm_head_matmul(&trace_plan)?;
    let identifier = plan_identifier(&public, &trace_plan, matmul.index)?;
    let request_plan = zkie_models_gpt2::lm_head_plan::lm_head_plan()?;
    let configuration = configuration_for(&request_plan, identifier)?;
    let plan_positions = u32::try_from(positions).map_err(|_| Error::ResourceLimit)?;
    let profile = zkie_models_gpt2::norm_profile::segment_profile(
        plan_positions,
        segment_x_seed(model, positions),
    )?;
    let (norm_index, x_rows, hd_rows) = final_norm_rows(trace)?;
    if matmul.activation_step != norm_index {
        return Err(Error::StatementMismatch);
    }
    let asset_root = Path::new(&config.asset_root);
    let (gain, bias) = read_anchored_params(asset_root, &public.assets)?;
    let rstd = read_rstd_values(asset_root, &profile, &x_rows)?;
    let (norm_plan, witness) = build_norm_segment(
        &profile,
        configuration,
        &commitment.tables,
        &x_rows,
        &gain,
        &bias,
        &rstd,
    )?;
    let rstd_table = rstd_table_view(asset_root)?;
    check_authenticated_bounds(
        &norm_plan,
        &profile,
        &x_rows,
        &gain,
        &bias,
        &rstd_table,
    )?;
    let key = segment_table_key(request_plan.table_shape()?);
    let asset = lm_head_asset(commitment, &matmul.weight_path)?;
    let work_dir = segment_work_dir(output_dir)?;
    std::fs::create_dir_all(&work_dir).map_err(|_| Error::Io)?;
    let started = Instant::now();
    let mut table = build_segment_table(
        asset_root,
        &asset,
        &request_plan,
        configuration,
        &key,
        &work_dir,
    )?;
    let reader = table.reader().clone();
    let specs = zkie_models_gpt2::plan::asset_specs();
    let spec = specs
        .iter()
        .find(|spec| spec.path == matmul.weight_path)
        .ok_or(Error::UnsupportedPrimitive)?;
    let identity = SegmentTableIdentity {
        asset_id: spec.asset_id,
        asset_table_identity: zkie_ops::proof::model_commitment::asset_table_identity(spec),
        code_log_n: reader.shape.code.log_n,
        code_ell: reader.shape.code.ell,
        code_r: reader.shape.code.r,
        table_v: reader.shape.v,
        answers: reader.shape.answers,
        salt_bytes: reader.shape.salt_bytes as u64,
        distance_radius: reader.shape.distances.radius,
        distance_later_radius: reader.shape.distances.later_radius,
        distance_m_once: reader.shape.distances.m_once,
        distance_m_later: reader.shape.distances.m_later,
        matrix_rows_log: request_plan.lin.matrix_rows_log,
        k_lo: request_plan.lin.k_lo,
        k_col: request_plan.lin.k_col,
        table_root: reader.root,
        table_record_digest: zkie_core::artifact::digest_bytes(&reader.record.encode()),
        table_configuration: configuration,
        source_identity: segment_table_source_identity(),
    };
    identity.validate()?;
    let binding = segment_identity_binding(&identity);
    let session_seed = segment_session_seed(model, positions, binding);
    let plan = build_segment_plan(
        &trace_plan,
        &config.expected_argmax,
        &reader,
        reader.shape,
        configuration,
        &norm_plan,
        &profile,
        &hd_rows,
        session_seed,
    )?;
    let mut params = build_param_tables(commitment, tables)?;
    let messages = prove_segment_bytes(
        &plan,
        &mut table,
        session_seed,
        &witness,
        &mut params,
        &rstd_table,
    )?;
    let _ = std::fs::remove_dir_all(&work_dir);
    ctx.metrics
        .record("norm_segment_prove_ms", started.elapsed().as_millis() as u64);
    ctx.metrics.record("norm_segment_bytes", messages.len() as u64);
    let carrier = NormSegmentCarrier {
        version: NORM_SEGMENT_CARRIER_VERSION,
        reader,
        messages,
    };
    Ok(Some((carrier.encode()?, identity)))
}

/// Verifies the final norm segment carrier when the proof publishes one. The
/// published reader is pinned to the operator policy, the plan is rebuilt from
/// the authenticated trace and the chain verifies against the frozen table.
fn verify_norm_segment(
    trusted: &crate::policy::TrustedRequest,
    public: &ModelCommitmentPublic,
    statement: &StatementData,
    trace: &SharedTrace,
    proof_dir: &Path,
    proof_manifest: &ArtifactManifest,
    asset_root: &Path,
    table_metas: &[AssetTableMeta],
    ctx: &mut PipelineContext<'_>,
) -> Result<()> {
    let applies =
        statement.tokens.len() == zkie_ops::proof::output_head_lin::OUTPUT_HEAD_POSITIONS;
    let entry = match proof_manifest.find_file(NORM_SEGMENT_CARRIER_FILE) {
        Some(entry) => entry,
        None => {
            if applies {
                return Err(Error::IncompleteArtifact);
            }
            return Ok(());
        }
    };
    if !applies {
        return Err(Error::StatementMismatch);
    }
    let bytes = fs::read(proof_dir.join(NORM_SEGMENT_CARRIER_FILE))
        .map_err(|_| Error::IncompleteArtifact)?;
    if bytes.len() as u64 != entry.byte_length
        || digest_bytes(&bytes) != entry.content_digest
    {
        return Err(Error::ArtifactIntegrityMismatch);
    }
    let carrier = NormSegmentCarrier::decode(&bytes)?;
    let identity = statement
        .segment_identity
        .as_ref()
        .ok_or(Error::IncompleteArtifact)?;
    identity.validate()?;
    if identity.source_identity != segment_table_source_identity() {
        return Err(Error::StatementMismatch);
    }
    let positions = statement.tokens.len();
    if positions != zkie_ops::proof::output_head_lin::OUTPUT_HEAD_POSITIONS
        || statement.model != trusted.model
        || statement.case_id != trusted.case_id
        || statement.model_commitment != public.anchor.model_commitment
    {
        return Err(Error::StatementMismatch);
    }
    let reader = &carrier.reader;
    if reader.fault().is_some()
        || reader.root != identity.table_root
        || reader.configuration != identity.table_configuration
        || reader.shape != identity.shape()?
        || digest_bytes(&reader.record.encode()) != identity.table_record_digest
        || identity.matrix_rows()? != reader.shape.rows()
    {
        return Err(Error::StatementMismatch);
    }
    let trace_plan = zkie_models_gpt2::plan::trace_plan(positions)?;
    let matmul = crate::s7_context::find_lm_head_matmul(&trace_plan)?;
    let specs = zkie_models_gpt2::plan::asset_specs();
    let spec = specs
        .iter()
        .find(|spec| spec.path == matmul.weight_path)
        .ok_or(Error::UnsupportedPrimitive)?;
    let asset_meta = public
        .assets
        .iter()
        .find(|asset| asset.path == spec.path)
        .ok_or(Error::AssetMismatch)?;
    if identity.asset_id != asset_meta.asset_id
        || identity.asset_table_identity
            != zkie_ops::proof::model_commitment::asset_table_identity(spec)
        || asset_meta.leaf_record.table_identity != identity.asset_table_identity
    {
        return Err(Error::StatementMismatch);
    }
    let identifier = plan_identifier(public, &trace_plan, matmul.index)?;
    let request_plan = zkie_models_gpt2::lm_head_plan::lm_head_plan()?;
    let configuration = configuration_for(&request_plan, identifier)?;
    if configuration != reader.configuration
        || identity.table_configuration != configuration
        || identity.shape()? != request_plan.table_shape()?
        || identity.matrix_rows_log != request_plan.lin.matrix_rows_log
        || identity.k_lo != request_plan.lin.k_lo
        || identity.k_col != request_plan.lin.k_col
    {
        return Err(Error::StatementMismatch);
    }
    let model = statement.model_commitment;
    let plan_positions = u32::try_from(positions).map_err(|_| Error::ResourceLimit)?;
    let profile = zkie_models_gpt2::norm_profile::segment_profile(
        plan_positions,
        segment_x_seed(model, positions),
    )?;
    let (norm_index, x_rows, hd_rows) = final_norm_rows(trace)?;
    if matmul.activation_step != norm_index {
        return Err(Error::StatementMismatch);
    }
    let norm_plan = build_norm_plan(&profile, configuration, table_metas, &x_rows)?;
    let (gain, bias) = read_anchored_params(asset_root, &public.assets)?;
    let session_seed =
        segment_session_seed(model, positions, segment_identity_binding(identity));
    let plan = build_segment_plan(
        &trace_plan,
        &statement.expected_argmax,
        reader,
        reader.shape,
        configuration,
        &norm_plan,
        &profile,
        &hd_rows,
        session_seed,
    )?;
    let rstd = rstd_table_view(asset_root)?;
    check_authenticated_bounds(&norm_plan, &profile, &x_rows, &gain, &bias, &rstd)?;
    let started = Instant::now();
    let outcome = verify_segment_bytes(&plan, reader, &carrier.messages, &rstd)?;
    ctx.metrics
        .record("norm_segment_verify_ms", started.elapsed().as_millis() as u64);
    if outcome.activation.root != plan.send.activation_root {
        return Err(Error::InvalidProof);
    }
    Ok(())
}


/// Proves the timesfm L0 norm-only segment and returns its carrier bytes.
fn prove_timesfm_norm_segment(
    config: &ProviderConfig,
    commitment: &zkie_ops::proof::model_commitment::ModelCommitment,
    tables: &[AssetTableInput],
    trace: &SharedTrace,
    ctx: &mut PipelineContext<'_>,
) -> Result<Option<Vec<u8>>> {
    use crate::timesfm_norm_parent;
    use crate::timesfm_norm_segment_provider as segment;

    // The segment positions are the patch tokens, not the raw input points:
    // the timesfm statement carries the 512 raw points.
    let positions = config.seq_tokens;
    if positions != timesfm_norm_parent::SEGMENT_POSITIONS {
        return Ok(None);
    }
    let model = commitment.anchor.record.model_commitment;
    let profile = zkie_models_timesfm::norm_profile::segment_profile(
        u32::try_from(positions).map_err(|_| Error::ResourceLimit)?,
        timesfm_norm_parent::segment_x_seed(model, positions),
    )?;
    let (_, x_rows, hd_rows) = timesfm_norm_parent::first_rms_norm_rows(trace)?;
    let public = ModelCommitmentPublic::from_commitment(commitment);
    let asset_root = Path::new(&config.asset_root);
    let gain = timesfm_norm_parent::read_gain(asset_root, &public.assets)?;
    let rstd = timesfm_norm_parent::read_rstd_values(asset_root, &profile, &x_rows)?;
    let configuration = segment::configuration(model);
    let plan = timesfm_norm_parent::build_norm_only_plan(
        &profile,
        configuration,
        &commitment.tables,
        &x_rows,
        &hd_rows,
    )?;
    let norm_plan = plan.norm.clone().ok_or(Error::IncompleteArtifact)?;
    let witness = timesfm_norm_parent::build_norm_witness(
        &norm_plan,
        &profile,
        configuration,
        &x_rows,
        &gain,
        &rstd,
    )?;
    let table = segment::rstd_table_view(asset_root)?;
    timesfm_norm_parent::check_authenticated_bounds(
        &norm_plan,
        &profile,
        &x_rows,
        &gain,
        &table,
    )?;
    let mut params = segment::build_param_tables(commitment, tables)?;
    let seed = timesfm_norm_parent::segment_session_seed(model, positions, true);
    let started = Instant::now();
    let carrier = segment::prove_bytes(&plan, seed, witness, &mut params, &table)?;
    ctx.metrics
        .record("timesfm_norm_prove_ms", started.elapsed().as_millis() as u64);
    let bytes = carrier.encode()?;
    ctx.metrics
        .record("timesfm_norm_bytes", bytes.len() as u64);
    Ok(Some(bytes))
}

/// Verifies the timesfm norm-only carrier when the statement declares it.
fn verify_timesfm_norm_segment(
    trusted: &crate::policy::TrustedRequest,
    public: &ModelCommitmentPublic,
    statement: &StatementData,
    trace: &SharedTrace,
    proof_dir: &Path,
    proof_manifest: &ArtifactManifest,
    asset_root: &Path,
    table_metas: &[AssetTableMeta],
    ctx: &mut PipelineContext<'_>,
) -> Result<()> {
    use crate::timesfm_norm_parent;
    use crate::timesfm_norm_segment_provider as segment;

    if statement.model != "timesfm" {
        return Ok(());
    }
    // The declaration must follow the trusted case, not the artifact: the
    // statement is prover-supplied, so turning the declaration off and
    // dropping the carrier must not switch the segment verification off for
    // a case the policy defines as a norm-only segment.
    let positions = trusted.patch_tokens.ok_or(Error::InvalidConfig)?;
    let required = positions == timesfm_norm_parent::SEGMENT_POSITIONS;
    if statement.norm_only != required {
        return Err(Error::StatementMismatch);
    }
    let entry = proof_manifest.find_file(segment::TIMESFM_NORM_CARRIER_FILE);
    if !required {
        return if entry.is_some() {
            Err(Error::StatementMismatch)
        } else {
            Ok(())
        };
    }
    let entry = entry.ok_or(Error::IncompleteArtifact)?;
    let bytes = fs::read(proof_dir.join(segment::TIMESFM_NORM_CARRIER_FILE))
        .map_err(|_| Error::IncompleteArtifact)?;
    if bytes.len() as u64 != entry.byte_length || digest_bytes(&bytes) != entry.content_digest {
        return Err(Error::ArtifactIntegrityMismatch);
    }
    let carrier = segment::TimesfmNormCarrier::decode(&bytes)?;
    let model = statement.model_commitment;
    let profile = zkie_models_timesfm::norm_profile::segment_profile(
        u32::try_from(positions).map_err(|_| Error::ResourceLimit)?,
        timesfm_norm_parent::segment_x_seed(model, positions),
    )?;
    let (_, x_rows, hd_rows) = timesfm_norm_parent::first_rms_norm_rows(trace)?;
    let configuration = segment::configuration(model);
    let plan = timesfm_norm_parent::build_norm_only_plan(
        &profile,
        configuration,
        table_metas,
        &x_rows,
        &hd_rows,
    )?;
    let norm_plan = plan.norm.clone().ok_or(Error::IncompleteArtifact)?;
    let gain = timesfm_norm_parent::read_gain(asset_root, &public.assets)?;
    let table = segment::rstd_table_view(asset_root)?;
    timesfm_norm_parent::check_authenticated_bounds(
        &norm_plan,
        &profile,
        &x_rows,
        &gain,
        &table,
    )?;
    let started = Instant::now();
    segment::verify_bytes(&plan, &carrier, &table)?;
    ctx.metrics
        .record("timesfm_norm_verify_ms", started.elapsed().as_millis() as u64);
    Ok(())
}

pub struct ProductionProvider;

impl PipelineProvider for ProductionProvider {
    fn commit_model(
        &self,
        request: &CommitRequest,
        ctx: &mut PipelineContext<'_>,
    ) -> Result<CommitReceipt> {
        let config = parse_case(&request.case_config)?;
        let table_params = trusted_table_params(&config.model)?;
        let tables = sample_production_tables(&table_params)?;
        let commitment = build_commitment(&config, &tables)?;
        // The one-time key qualification is proved once over this committed
        // instance and published in the commit's public bundle. Every request
        // re-verifies the same proof; the proof is never regenerated per
        // request and re-verification does not spend the durable column budget.
        let mut key_bundle: Option<Vec<u8>> = None;
        if config.model == "gpt2" {
            let (params, spec, columns) = gpt2_key_columns()?;
            let identity = zkie_models_gpt2::plan::table_model_identity()?;
            let meta = commitment
                .tables
                .iter()
                .find(|meta| meta.path == params.path)
                .ok_or(Error::IncompleteArtifact)?;
            let input = tables
                .iter()
                .find(|table| table.params.path == params.path)
                .ok_or(Error::IncompleteArtifact)?;
            let started = Instant::now();
            let range = zkie_ops::proof::table_range::production_table_range()?;
            let bytes = prove_asset_key(&commitment, meta, &params, identity, &spec, input, columns, &range)?;
            ctx.metrics.record("key_prove_ms", started.elapsed().as_millis() as u64);
            let started = Instant::now();
            verify_asset_key(meta, &params, identity, &spec, columns, &range, &bytes)?;
            ctx.metrics.record("key_verify_ms", started.elapsed().as_millis() as u64);
            ctx.metrics.record("key_bytes", bytes.len() as u64);
            key_bundle = Some(bytes);
        }
        if config.model == "timesfm" {
            let (params, spec, columns) = timesfm_key_columns()?;
            let identity = zkie_models_timesfm::plan::table_model_identity()?;
            let meta = commitment
                .tables
                .iter()
                .find(|meta| meta.path == params.path)
                .ok_or(Error::IncompleteArtifact)?;
            let input = tables
                .iter()
                .find(|table| table.params.path == params.path)
                .ok_or(Error::IncompleteArtifact)?;
            let started = Instant::now();
            let range = zkie_models_timesfm::plan::production_table_range()?;
            let bytes = prove_asset_key(&commitment, meta, &params, identity, &spec, input, columns, &range)?;
            ctx.metrics.record("key_prove_ms", started.elapsed().as_millis() as u64);
            let started = Instant::now();
            verify_asset_key(meta, &params, identity, &spec, columns, &range, &bytes)?;
            ctx.metrics.record("key_verify_ms", started.elapsed().as_millis() as u64);
            ctx.metrics.record("key_bytes", bytes.len() as u64);
            key_bundle = Some(bytes);
        }
        let asset_units = commitment.assets.len() as u64;
        let mut progress_event =
            ProgressEvent::new("commit", "commit", ProgressKind::WorkAdvanced);
        progress_event.completed_units = asset_units;
        progress_event.total_units = Some(asset_units);
        progress_event.unit = Some("assets".into());
        let _ = ctx.progress.emit(progress_event);
        let public = ModelCommitmentPublic::from_commitment(&commitment);
        let model = commitment.anchor.record.model_commitment;
        let commitment_bytes = bincode::serialize(&public).map_err(|_| Error::InvalidEncoding)?;
        let config_bytes = bincode::serialize(&config).map_err(|_| Error::InvalidEncoding)?;
        let protocol_bytes = fs::read(&request.case_config).map_err(|_| Error::Io)?;
        let parameters_bytes = fs::read(&request.parameters).map_err(|_| Error::Io)?;
        let tables_bytes = bincode::serialize(&commitment.tables).map_err(|_| Error::InvalidEncoding)?;
        let randomness_bytes = encode_asset_table_randomness_set(&tables)?;
        let commitment_digest = digest_bytes(&commitment_bytes);
        let config_digest = digest_bytes(&config_bytes);
        let protocol_digest = digest_bytes(&protocol_bytes);
        let parameters_digest = digest_bytes(&parameters_bytes);
        let tables_digest = digest_bytes(&tables_bytes);
        let randomness_digest = digest_bytes(&randomness_bytes);
        let mut files: Vec<(&str, &[u8])> = vec![
            ("public/model-commitment.bin", commitment_bytes.as_slice()),
            ("public/provider-config.bin", config_bytes.as_slice()),
            ("public/protocol.bin", protocol_bytes.as_slice()),
            ("public/asset-tables.bin", tables_bytes.as_slice()),
            ("private/parameters.bin", parameters_bytes.as_slice()),
            ("private/asset-table-randomness.bin", randomness_bytes.as_slice()),
        ];
        let mut digests: Vec<(&str, Digest)> = vec![
            ("public/model-commitment.bin", commitment_digest),
            ("public/provider-config.bin", config_digest),
            ("public/protocol.bin", protocol_digest),
            ("public/asset-tables.bin", tables_digest),
            ("private/parameters.bin", parameters_digest),
            ("private/asset-table-randomness.bin", randomness_digest),
        ];
        let mut manifest_entries = vec![
            ArtifactFile::new("public/model-commitment.bin", commitment_bytes.len() as u64, commitment_digest),
            ArtifactFile::new("public/provider-config.bin", config_bytes.len() as u64, config_digest),
            ArtifactFile::new("public/protocol.bin", protocol_bytes.len() as u64, protocol_digest),
            ArtifactFile::new("public/asset-tables.bin", tables_bytes.len() as u64, tables_digest),
            ArtifactFile::new("private/parameters.bin", parameters_bytes.len() as u64, parameters_digest),
            ArtifactFile::new("private/asset-table-randomness.bin", randomness_bytes.len() as u64, randomness_digest),
        ];
        if let Some(bytes) = &key_bundle {
            let digest = digest_bytes(bytes);
            files.push((ASSET_KEY_BUNDLE, bytes.as_slice()));
            digests.push((ASSET_KEY_BUNDLE, digest));
            manifest_entries.push(ArtifactFile::new(ASSET_KEY_BUNDLE, bytes.len() as u64, digest));
        }
        let manifest = commit_manifest(model, manifest_entries);
        ArtifactPublisher::publish_with_digests(&request.output_dir, &manifest, &files, &digests)?;
        ctx.metrics.record("commit_bytes", commitment_bytes.len() as u64);
        Ok(CommitReceipt {
            artifact_dir: request.output_dir.clone(),
            manifest_digest: manifest.manifest_digest()?,
        })
    }

    fn prove_request(
        &self,
        request: &ProveRequest,
        ctx: &mut PipelineContext<'_>,
    ) -> Result<ProofReceipt> {
        let config_bytes = fs::read(request.commit_dir.join("public/provider-config.bin"))
            .map_err(|_| Error::MissingCommitArtifact)?;
        let config: ProviderConfig = bincode::deserialize(&config_bytes).map_err(|_| Error::InvalidEncoding)?;
        let table_params = trusted_table_params(&config.model)?;
        let randomness_bytes =
            fs::read(request.commit_dir.join("private/asset-table-randomness.bin"))
                .map_err(|_| Error::MissingCommitArtifact)?;
        let tables = decode_asset_table_randomness_set(&randomness_bytes, &table_params)?;
        let commitment = build_commitment(&config, &tables)?;
        let published_bytes = fs::read(request.commit_dir.join("public/asset-tables.bin"))
            .map_err(|_| Error::MissingCommitArtifact)?;
        let published_tables: Vec<AssetTableMeta> = bincode::deserialize(&published_bytes)
            .map_err(|_| Error::InvalidEncoding)?;
        if published_tables != commitment.tables {
            return Err(Error::StatementMismatch);
        }
        let model = commitment.anchor.record.model_commitment;
        if request.expected_model_id != model {
            return Err(Error::StatementMismatch);
        }
        // The prover's inputs must be this artifact's entries: the commit
        // manifest binds the public config, the public table metadata and the
        // private randomness by length and digest. The key bundle is required
        // and must verify over the same committed instance before the request
        // is executed.
        let commit_manifest = load_checked_manifest(&request.commit_dir, ArtifactKind::Commit)?;
        if commit_manifest.model_commitment_id != model {
            return Err(Error::StatementMismatch);
        }
        check_manifest_bytes(&commit_manifest, "public/provider-config.bin", &config_bytes)?;
        check_manifest_bytes(&commit_manifest, "public/asset-tables.bin", &published_bytes)?;
        check_manifest_bytes(
            &commit_manifest,
            "private/asset-table-randomness.bin",
            &randomness_bytes,
        )?;
        if config.model == "gpt2" {
            let (params, spec, columns) = gpt2_key_columns()?;
            let identity = zkie_models_gpt2::plan::table_model_identity()?;
            let meta = commitment
                .tables
                .iter()
                .find(|meta| meta.path == params.path)
                .ok_or(Error::IncompleteArtifact)?;
            let bytes =
                checked_artifact_file(&request.commit_dir, &commit_manifest, ASSET_KEY_BUNDLE)?;
            let started = Instant::now();
            let range = zkie_ops::proof::table_range::production_table_range()?;
            verify_asset_key(meta, &params, identity, &spec, columns, &range, &bytes)?;
            ctx.metrics.record("key_verify_ms", started.elapsed().as_millis() as u64);
            ctx.metrics.record("key_bytes", bytes.len() as u64);
        }
        if config.model == "timesfm" {
            let (params, spec, columns) = timesfm_key_columns()?;
            let identity = zkie_models_timesfm::plan::table_model_identity()?;
            let meta = commitment
                .tables
                .iter()
                .find(|meta| meta.path == params.path)
                .ok_or(Error::IncompleteArtifact)?;
            let bytes =
                checked_artifact_file(&request.commit_dir, &commit_manifest, ASSET_KEY_BUNDLE)?;
            let started = Instant::now();
            let range = zkie_models_timesfm::plan::production_table_range()?;
            verify_asset_key(meta, &params, identity, &spec, columns, &range, &bytes)?;
            ctx.metrics.record("key_verify_ms", started.elapsed().as_millis() as u64);
            ctx.metrics.record("key_bytes", bytes.len() as u64);
        }
        let (trace, consumer) = execute_shared(&config, &commitment, &tables, ctx.progress)?;
        let (norm_carrier, segment_identity) = if config.model == "gpt2" {
            match prove_norm_segment(
                &config,
                &commitment,
                &tables,
                &trace,
                &request.output_dir,
                ctx,
            )? {
                Some((bytes, identity)) => (Some(bytes), Some(identity)),
                None => (None, None),
            }
        } else if config.model == "timesfm" {
            match prove_timesfm_norm_segment(&config, &commitment, &tables, &trace, ctx)? {
                Some(bytes) => (Some(bytes), None),
                None => (None, None),
            }
        } else {
            (None, None)
        };
        let statement = StatementData {
            case_id: config.case_id.clone(),
            model: config.model.clone(),
            model_commitment: model,
            tokens: config.tokens.clone(),
            expected_argmax: config.expected_argmax.clone(),
            expected_output: config.expected_output.clone(),
            segment_identity,
            norm_only: config.model == "timesfm"
                && config.seq_tokens == crate::timesfm_norm_parent::SEGMENT_POSITIONS,
        };
        let statement_bytes = bincode::serialize(&statement).map_err(|_| Error::InvalidEncoding)?;
        let consumer_bytes = consumer
            .as_ref()
            .map(|file| file.encode())
            .transpose()?;
        let proof_bytes = bincode::serialize(&trace).map_err(|_| Error::InvalidEncoding)?;
        let public_inputs = if config.model == "timesfm" {
            bincode::serialize(&config.expected_output).map_err(|_| Error::InvalidEncoding)?
        } else {
            bincode::serialize(&config.expected_argmax).map_err(|_| Error::InvalidEncoding)?
        };
        let statement_digest = digest_bytes(&statement_bytes);
        let proof_digest = digest_bytes(&proof_bytes);
        let public_inputs_digest = digest_bytes(&public_inputs);
        let mut files: Vec<(&str, &[u8])> = vec![
            ("statement.bin", statement_bytes.as_slice()),
            ("proof.bin", proof_bytes.as_slice()),
            ("public-inputs.bin", public_inputs.as_slice()),
        ];
        let mut digests: Vec<(&str, Digest)> = vec![
            ("statement.bin", statement_digest),
            ("proof.bin", proof_digest),
            ("public-inputs.bin", public_inputs_digest),
        ];
        let mut manifest_files = vec![
            ArtifactFile::new("statement.bin", statement_bytes.len() as u64, statement_digest),
            ArtifactFile::new("proof.bin", proof_bytes.len() as u64, proof_digest),
            ArtifactFile::new("public-inputs.bin", public_inputs.len() as u64, public_inputs_digest),
        ];
        if let Some(bytes) = &consumer_bytes {
            let digest = digest_bytes(bytes);
            files.push((zkie_ops::proof::consumer_proof::CONSUMER_PROOF_FILE, bytes.as_slice()));
            digests.push((zkie_ops::proof::consumer_proof::CONSUMER_PROOF_FILE, digest));
            manifest_files.push(ArtifactFile::new(
                zkie_ops::proof::consumer_proof::CONSUMER_PROOF_FILE,
                bytes.len() as u64,
                digest,
            ));
            ctx.metrics.record("consumer_proof_bytes", bytes.len() as u64);
        }
        if let Some(bytes) = &norm_carrier {
            let name = if config.model == "timesfm" {
                crate::timesfm_norm_segment_provider::TIMESFM_NORM_CARRIER_FILE
            } else {
                NORM_SEGMENT_CARRIER_FILE
            };
            let digest = digest_bytes(bytes);
            files.push((name, bytes.as_slice()));
            digests.push((name, digest));
            manifest_files.push(ArtifactFile::new(name, bytes.len() as u64, digest));
        }
        let manifest = proof_manifest(model, manifest_files);
        ArtifactPublisher::publish_with_digests(&request.output_dir, &manifest, &files, &digests)?;
        ctx.metrics.record("proof_bytes", proof_bytes.len() as u64);
        Ok(ProofReceipt {
            proof_dir: request.output_dir.clone(),
            statement_digest: digest_bytes(&statement_bytes),
            test_only: false,
        })
    }

    fn verify_request(
        &self,
        request: &VerifyRequest,
        ctx: &mut PipelineContext<'_>,
    ) -> Result<VerifyReceipt> {
        let policy_bytes =
            fs::read(&request.expected_policy).map_err(|_| Error::IncompleteArtifact)?;
        let policy = crate::policy::parse(&policy_bytes)?;
        // Both artifacts are read through their manifests: the commit public
        // bundle and the request proof are bound by artifact kind, protocol
        // identity, file length and content digest. A missing, swapped or
        // truncated file is refused before any stage runs.
        let commit_root = request
            .public_bundle_dir
            .parent()
            .ok_or(Error::IncompleteArtifact)?;
        if request.public_bundle_dir.file_name().and_then(|name| name.to_str())
            != Some("public")
        {
            return Err(Error::InvalidConfig);
        }
        let commit_manifest = load_checked_manifest(commit_root, ArtifactKind::Commit)?;
        let commitment_bytes =
            checked_artifact_file(commit_root, &commit_manifest, "public/model-commitment.bin")?;
        let public: ModelCommitmentPublic =
            deserialize_exact(&commitment_bytes, Error::ArtifactIntegrityMismatch)?;
        let statement_bytes = fs::read(&request.statement).map_err(|_| Error::IncompleteArtifact)?;
        let statement: StatementData =
            deserialize_exact(&statement_bytes, Error::InvalidProof)?;
        let proof_dir = request.proof.parent().ok_or(Error::IncompleteArtifact)?;
        let proof_manifest = load_checked_manifest(proof_dir, ArtifactKind::Proof)?;
        check_proof_input(
            &request.statement,
            proof_dir,
            &proof_manifest,
            zkie_core::artifact::PROOF_STATEMENT_FILE,
            &statement_bytes,
        )?;
        if request.proof.file_name().and_then(|name| name.to_str())
            != Some(zkie_core::artifact::PROOF_FILE)
        {
            return Err(Error::StatementMismatch);
        }
        let trace = read_proof_exact(&request.proof, ctx.runtime.memory_budget_bytes)?;
        let proof_entry = proof_manifest
            .find_file(zkie_core::artifact::PROOF_FILE)
            .ok_or(Error::IncompleteArtifact)?;
        checked_large_file(&request.proof, proof_entry)?;
        let public_inputs =
            fs::read(&request.public_inputs).map_err(|_| Error::IncompleteArtifact)?;
        check_proof_input(
            &request.public_inputs,
            proof_dir,
            &proof_manifest,
            zkie_core::artifact::PROOF_PUBLIC_INPUTS_FILE,
            &public_inputs,
        )?;
        let table_bytes =
            checked_artifact_file(commit_root, &commit_manifest, "public/asset-tables.bin")?;
        let table_metas: Vec<AssetTableMeta> =
            bincode::deserialize(&table_bytes).map_err(|_| Error::InvalidEncoding)?;

        let trusted = crate::policy::select(
            &policy,
            &statement.case_id,
            &statement.model,
            statement.tokens.len(),
        )?;
        let model_id = trusted.model_id;
        if request.expected_model_id != model_id
            || statement.model_commitment != model_id
            || public.anchor.model_commitment != model_id
            || commit_manifest.model_commitment_id != model_id
            || proof_manifest.model_commitment_id != model_id
        {
            return Err(Error::StatementMismatch);
        }
        let table_params = trusted_table_params(&trusted.model)?;
        // The public case config fixes the asset root that locates the fixed
        // public tables. It is public material from the bundle, not a weight
        // path and never a proof value.
        let config_bytes =
            checked_artifact_file(commit_root, &commit_manifest, "public/provider-config.bin")?;
        let public_config: ProviderConfig =
            bincode::deserialize(&config_bytes).map_err(|_| Error::InvalidEncoding)?;
        let public_root = std::path::Path::new(&public_config.asset_root);

        let (bound, statement_matches) = match trusted.model.as_str() {
            "gpt2" => {
                if trusted.model_version != zkie_models_gpt2::plan::MODEL_VERSION {
                    return Err(Error::InvalidConfig);
                }
                let plan = zkie_models_gpt2::plan::trace_plan_public(statement.tokens.len())?;
                if zkie_models_gpt2::plan::model_commitment(&statement.tokens)? != model_id {
                    return Err(Error::StatementMismatch);
                }
                let parameters = zkie_models_gpt2::plan::parameters_digest()?;
                let specs = zkie_models_gpt2::plan::asset_specs();
                verify_model_anchor_with_tables_on_identity(
                    &public,
                    trusted.anchor_id,
                    model_id,
                    zkie_models_gpt2::plan::table_model_identity()?,
                    parameters,
                    &specs,
                    &table_metas,
                    &table_params,
                )?;
                // The table's one-time qualification is re-verified per
                // request from the commit's public key bundle into an
                // in-memory accepted key; every accepted field is compared
                // with this commit's table, record, range and plan before the
                // request consumes the table. The key bytes and time are
                // recorded separately.
                let (key_params, key_spec, key_columns) = gpt2_key_columns()?;
                let identity = zkie_models_gpt2::plan::table_model_identity()?;
                let key_bytes =
                    checked_artifact_file(commit_root, &commit_manifest, ASSET_KEY_BUNDLE)?;
                let key_meta = table_metas
                    .iter()
                    .find(|meta| meta.path == key_params.path)
                    .ok_or(Error::IncompleteArtifact)?;
                let started = Instant::now();
                let key_range = zkie_ops::proof::table_range::production_table_range()?;
                verify_asset_key(
                    key_meta,
                    &key_params,
                    identity,
                    &key_spec,
                    key_columns,
                    &key_range,
                    &key_bytes,
                )?;
                ctx.metrics.record("key_verify_ms", started.elapsed().as_millis() as u64);
                ctx.metrics.record("key_bytes", key_bytes.len() as u64);
                // The request's own stages' auxiliary file is part of the same
                // proof artifact: its manifest entry must exist and its bytes
                // must match before it is decoded.
                let sidecar_entry = proof_manifest
                    .find_file(zkie_ops::proof::consumer_proof::CONSUMER_PROOF_FILE)
                    .ok_or(Error::IncompleteArtifact)?;
                let sidecar_bytes = fs::read(
                    proof_dir.join(zkie_ops::proof::consumer_proof::CONSUMER_PROOF_FILE),
                )
                .map_err(|_| Error::IncompleteArtifact)?;
                if sidecar_bytes.len() as u64 != sidecar_entry.byte_length
                    || digest_bytes(&sidecar_bytes) != sidecar_entry.content_digest
                {
                    return Err(Error::ArtifactIntegrityMismatch);
                }
                let consumer = ConsumerProofFile::decode(&sidecar_bytes)?;
                let outputs = zkie_ops::proof::model_commitment::verify_shared_trace_outputs(
                    &public,
                    &trace,
                    &statement.tokens,
                    &plan,
                    Some(public_root),
                )?;
                let path = "gpt2/weights/L0_k_w_i32.bin";
                let consumer_params =
                    zkie_models_gpt2::plan::projection_table_params(path)
                        .ok_or(Error::UnsupportedPrimitive)?;
                let derived = zkie_models_gpt2::plan::derive_table_plan(
                    &zkie_models_gpt2::plan::TABLE_PLAN_CANDIDATE,
                )?;
                let shared = ConsumerSharedPlan {
                    m_once: derived.table_shape.distances.m_once,
                    m_later: derived.table_shape.distances.m_later,
                    one_time_columns: derived.one_time_columns,
                    later_columns: derived.later_batch_columns,
                    declared_batches: 1,
                    table_tail: derived.table_shape.code.r,
                };
                let spec = zkie_models_gpt2::plan::asset_specs()
                    .into_iter()
                    .find(|spec| spec.path == path)
                    .ok_or(Error::UnsupportedPrimitive)?;
                let meta = table_metas
                    .iter()
                    .find(|meta| meta.path == path)
                    .ok_or(Error::IncompleteArtifact)?;
                verify_consumer_request(
                    &plan,
                    &outputs,
                    zkie_models_gpt2::plan::table_model_identity()?,
                    meta,
                    &consumer_params,
                    &spec,
                    shared,
                    &consumer,
                )?;
                verify_norm_segment(
                    &trusted,
                    &public,
                    &statement,
                    &trace,
                    proof_dir,
                    &proof_manifest,
                    public_root,
                    &table_metas,
                    ctx,
                )?;
                let argmax = trace.argmax.clone();
                let computed: Vec<i64> = argmax.iter().map(|value| *value as i64).collect();
                let bound = crate::output::bind(
                    &trusted.output,
                    &computed,
                    Some(zkie_models_gpt2::plan::valid_vocab()),
                    &public_inputs,
                )?;
                let statement_matches = statement
                    .expected_argmax
                    .iter()
                    .map(|value| *value as i64)
                    .eq(computed.iter().copied());
                (bound, statement_matches)
            }
            "gemma3" => {
                if trusted.model_version != zkie_models_gemma3::plan::MODEL_VERSION {
                    return Err(Error::InvalidConfig);
                }
                let plan = zkie_models_gemma3::plan::trace_plan_public(statement.tokens.len())?;
                if zkie_models_gemma3::plan::model_commitment(&statement.tokens)? != model_id {
                    return Err(Error::StatementMismatch);
                }
                let parameters = zkie_models_gemma3::plan::parameters_digest()?;
                let specs = zkie_models_gemma3::plan::asset_specs();
                verify_model_anchor_with_tables(
                    &public,
                    trusted.anchor_id,
                    model_id,
                    parameters,
                    &specs,
                    &table_metas,
                    &table_params,
                )?;
                let argmax =
                    verify_shared_trace(&public, &trace, &statement.tokens, &plan, Some(public_root))?;
                let computed: Vec<i64> = argmax.iter().map(|value| *value as i64).collect();
                let bound = crate::output::bind(
                    &trusted.output,
                    &computed,
                    Some(zkie_models_gemma3::plan::valid_vocab()),
                    &public_inputs,
                )?;
                let statement_matches = statement
                    .expected_argmax
                    .iter()
                    .map(|value| *value as i64)
                    .eq(computed.iter().copied());
                (bound, statement_matches)
            }
            "timesfm" => {
                let patches = trusted
                    .patch_tokens
                    .ok_or(Error::InvalidConfig)?;
                if trusted.model_version != zkie_models_timesfm::plan::MODEL_VERSION {
                    return Err(Error::InvalidConfig);
                }
                let plan = zkie_models_timesfm::plan::trace_plan_public(patches)?;
                if zkie_models_timesfm::plan::model_commitment(&statement.tokens)? != model_id {
                    return Err(Error::StatementMismatch);
                }
                let parameters = zkie_models_timesfm::plan::parameters_digest()?;
                let specs = zkie_models_timesfm::plan::asset_specs();
                let identity = zkie_models_timesfm::plan::table_model_identity()?;
                verify_model_anchor_with_tables_on_identity(
                    &public,
                    trusted.anchor_id,
                    model_id,
                    identity,
                    parameters,
                    &specs,
                    &table_metas,
                    &table_params,
                )?;
                let (key_params, key_spec, key_columns) = timesfm_key_columns()?;
                let key_bytes =
                    checked_artifact_file(commit_root, &commit_manifest, ASSET_KEY_BUNDLE)?;
                let key_meta = table_metas
                    .iter()
                    .find(|meta| meta.path == key_params.path)
                    .ok_or(Error::IncompleteArtifact)?;
                let started = Instant::now();
                let key_range = zkie_models_timesfm::plan::production_table_range()?;
                verify_asset_key(
                    key_meta,
                    &key_params,
                    identity,
                    &key_spec,
                    key_columns,
                    &key_range,
                    &key_bytes,
                )?;
                ctx.metrics.record("key_verify_ms", started.elapsed().as_millis() as u64);
                ctx.metrics.record("key_bytes", key_bytes.len() as u64);
                let sidecar_entry = proof_manifest
                    .find_file(zkie_ops::proof::consumer_proof::CONSUMER_PROOF_FILE)
                    .ok_or(Error::IncompleteArtifact)?;
                let sidecar_bytes = fs::read(
                    proof_dir.join(zkie_ops::proof::consumer_proof::CONSUMER_PROOF_FILE),
                )
                .map_err(|_| Error::IncompleteArtifact)?;
                if sidecar_bytes.len() as u64 != sidecar_entry.byte_length
                    || digest_bytes(&sidecar_bytes) != sidecar_entry.content_digest
                {
                    return Err(Error::ArtifactIntegrityMismatch);
                }
                let consumer = ConsumerProofFile::decode(&sidecar_bytes)?;
                let outputs = zkie_ops::proof::model_commitment::verify_shared_trace_outputs(
                    &public,
                    &trace,
                    &statement.tokens,
                    &plan,
                    Some(public_root),
                )?;
                let last = outputs.last().ok_or(Error::StatementMismatch)?;
                let output: Vec<i64> = last.iter().flatten().copied().collect();
                if output != trace.output {
                    return Err(Error::StatementMismatch);
                }
                let path = key_params.path.as_str();
                let consumer_params = zkie_models_timesfm::plan::projection_table_params(path)
                    .ok_or(Error::UnsupportedPrimitive)?;
                let derived = zkie_models_timesfm::plan::derive_table_plan(
                    &zkie_models_timesfm::plan::TABLE_PLAN_CANDIDATE,
                )?;
                let shared = ConsumerSharedPlan {
                    m_once: derived.table_shape.distances.m_once,
                    m_later: derived.table_shape.distances.m_later,
                    one_time_columns: derived.one_time_columns,
                    later_columns: derived.later_batch_columns,
                    declared_batches: 1,
                    table_tail: derived.table_shape.code.r,
                };
                let spec = zkie_models_timesfm::plan::asset_specs()
                    .into_iter()
                    .find(|spec| spec.path == path)
                    .ok_or(Error::UnsupportedPrimitive)?;
                let meta = table_metas
                    .iter()
                    .find(|meta| meta.path == path)
                    .ok_or(Error::IncompleteArtifact)?;
                verify_consumer_request(
                    &plan,
                    &outputs,
                    identity,
                    meta,
                    &consumer_params,
                    &spec,
                    shared,
                    &consumer,
                )?;
                verify_timesfm_norm_segment(
                    &trusted,
                    &public,
                    &statement,
                    &trace,
                    proof_dir,
                    &proof_manifest,
                    public_root,
                    &table_metas,
                    ctx,
                )?;
                if output.len() != zkie_models_timesfm::plan::output_elements(patches)? {
                    return Err(Error::ShapeMismatch);
                }
                let bound = crate::output::bind(
                    &trusted.output,
                    &output,
                    None,
                    &public_inputs,
                )?;
                let statement_matches = statement
                    .expected_output
                    .iter()
                    .map(|value| *value as i64)
                    .eq(output.iter().copied());
                (bound, statement_matches)
            }
            _ => return Err(Error::UnsupportedPrimitive),
        };

        let accepted = bound.accepted && statement_matches;
        if accepted {
            let step_units = trace.steps.len() as u64;
            let mut progress_event =
                ProgressEvent::new("verify", "verify", ProgressKind::WorkAdvanced);
            progress_event.completed_units = step_units;
            progress_event.total_units = Some(step_units);
            progress_event.unit = Some("relation_steps".into());
            let _ = ctx.progress.emit(progress_event);
        }
        Ok(VerifyReceipt {
            accepted,
            outcome: if accepted {
                RunOutcome::Passed
            } else {
                RunOutcome::Rejected
            },
            output_binding: Some(match bound.mode {
                crate::policy::OutputMode::Expected => OutputBindingMode::Expected,
                crate::policy::OutputMode::None => OutputBindingMode::None,
            }),
            authenticated_output: Some(bound.values),
            authenticated_output_digest: Some(bound.digest),
        })
    }
}

pub fn production_provider() -> ProductionProvider {
    ProductionProvider
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, Cursor, Read, Seek, SeekFrom};

    /// The CLI provider resolves one table identity's ledger below its real
    /// asset root, through the single ledger derivation.
    #[test]
    fn the_provider_derives_the_column_ledger_under_the_asset_root() {
        let config = ProviderConfig {
            asset_root: "/srv/zkie/assets".into(),
            case_root: String::new(),
            case_id: String::new(),
            model: "gpt2".into(),
            seq_tokens: 16,
            tokens: Vec::new(),
            input: Vec::new(),
            expected_argmax: Vec::new(),
            expected_output: Vec::new(),
        };
        let root = Digest([0xCD; 32]);
        assert_eq!(
            config.column_ledger_dir(root),
            zkie_ops::proof::column_session::column_ledger_dir(
                Path::new("/srv/zkie/assets"),
                root
            )
        );
    }

    fn trace() -> SharedTrace {
        SharedTrace {
            tokens: Vec::new(),
            assets: Vec::new(),
            steps: Vec::new(),
            argmax: Vec::new(),
            output: Vec::new(),
        }
    }

    fn statement() -> StatementData {
        StatementData {
            case_id: "case".into(),
            model: "gpt2".into(),
            model_commitment: Digest([1; 32]),
            tokens: Vec::new(),
            expected_argmax: Vec::new(),
            expected_output: Vec::new(),
            segment_identity: None,
            norm_only: false,
        }
    }

    fn temp_path(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "zkie-provider-bytes-{}-{name}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn exact_bincode_rejects_trailing_and_truncation() {
        let bytes = bincode::serialize(&statement()).unwrap();
        let value: StatementData =
            deserialize_exact(&bytes, Error::InvalidProof).unwrap();
        assert_eq!(value.case_id, "case");
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert_eq!(
            deserialize_exact::<StatementData>(&trailing, Error::InvalidProof)
                .unwrap_err()
                .code(),
            "invalid_proof"
        );
        assert_eq!(
            deserialize_exact::<StatementData>(&bytes[..bytes.len() - 1], Error::InvalidProof)
                .unwrap_err()
                .code(),
            "invalid_encoding"
        );
    }

    #[test]
    fn proof_bincode_requires_eof() {
        let root = temp_path("proof");
        let path = root.join("proof.bin");
        let bytes = bincode::serialize(&trace()).unwrap();
        fs::write(&path, &bytes).unwrap();
        read_proof_exact(&path, 1 << 20).unwrap();
        let mut trailing = bytes.clone();
        trailing.push(0);
        fs::write(&path, &trailing).unwrap();
        assert_eq!(
            read_proof_exact(&path, 1 << 20).unwrap_err().code(),
            "invalid_proof"
        );
        fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
        assert_eq!(
            read_proof_exact(&path, 1 << 20).unwrap_err().code(),
            "invalid_encoding"
        );
        let _ = fs::remove_dir_all(&root);
    }

    struct ErrorAfter {
        bytes: Cursor<Vec<u8>>,
    }

    impl Read for ErrorAfter {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            let position = self.bytes.position() as usize;
            if position < self.bytes.get_ref().len() {
                self.bytes.read(out)
            } else {
                Err(io::Error::other("injected read error"))
            }
        }
    }

    impl Seek for ErrorAfter {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            self.bytes.seek(position)
        }
    }

    #[test]
    fn proof_eof_read_error_is_not_success() {
        let bytes = bincode::serialize(&trace()).unwrap();
        let mut reader = ErrorAfter {
            bytes: Cursor::new(bytes),
        };
        assert_eq!(
            crate::bounded_trace::read_from(&mut reader, 1 << 20)
                .unwrap_err()
                .code(),
            "io"
        );
    }

    #[test]
    fn proof_directory_is_not_success() {
        let root = temp_path("proof-dir");
        let path = root.join("proof.bin");
        fs::create_dir_all(&path).unwrap();
        assert!(read_proof_exact(&path, 1 << 20).is_err());
        let _ = fs::remove_dir_all(&root);
    }
}
