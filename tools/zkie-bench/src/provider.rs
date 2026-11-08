//! Production pipeline provider for the three shared model traces.
//!
//! Commit publishes the shared model commitment and the public case inputs.
//! Prove re-opens the commitment, executes the shared trace and writes the
//! proof. Verify reads only the public commitment/proof/inputs.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use zkie_core::artifact::{
    digest_bytes, ArtifactFile, ArtifactKind, ArtifactManifest, PrivacyMode,
};
use zkie_core::error::{Error, Result};
use zkie_core::progress::{ProgressEvent, ProgressKind, ProgressSink, RunOutcome};
use zkie_core::types::Digest;
use zkie_engine::artifact::ArtifactPublisher;
use zkie_models_gemma3::executor as gemma3;
use zkie_models_gpt2::executor as gpt2;
use zkie_models_timesfm::executor as timesfm;
use zkie_ops::pipeline::{
    CommitReceipt, CommitRequest, PipelineContext, PipelineProvider, ProofReceipt, ProveRequest,
    VerifyReceipt, VerifyRequest,
};
use zkie_ops::proof::model_commitment::{
    trace_prefix, verify_shared_trace, verify_shared_trace_output, ModelCommitmentPublic,
    SharedTrace,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StatementData {
    pub case_id: String,
    pub model: String,
    pub model_commitment: Digest,
    pub tokens: Vec<u32>,
    pub expected_argmax: Vec<u32>,
    pub expected_output: Vec<i32>,
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

fn build_commitment(config: &ProviderConfig) -> Result<zkie_ops::proof::model_commitment::ModelCommitment> {
    match config.model.as_str() {
        "gpt2" => gpt2::build_commitment(Path::new(&config.asset_root), &config.tokens),
        "gemma3" => gemma3::build_commitment(Path::new(&config.case_root), &config.tokens),
        "timesfm" => timesfm::build_commitment(
            Path::new(&config.case_root),
            &config.input.iter().map(|value| *value as i64).collect::<Vec<_>>(),
        ),
        _ => Err(Error::UnsupportedPrimitive),
    }
}

fn execute_shared(
    config: &ProviderConfig,
    commitment: &zkie_ops::proof::model_commitment::ModelCommitment,
    prefix: &Digest,
    progress: &dyn ProgressSink,
) -> Result<SharedTrace> {
    match config.model.as_str() {
        "gpt2" => gpt2::execute_shared_with_prefix(
            commitment,
            &config.tokens.iter().map(|value| *value as i64).collect::<Vec<_>>(),
            prefix,
            Some(progress),
        ),
        "gemma3" => gemma3::execute_shared_with_prefix(
            commitment,
            &config.tokens.iter().map(|value| *value as i64).collect::<Vec<_>>(),
            prefix,
            Some(progress),
        ),
        "timesfm" => timesfm::execute_shared_with_prefix(
            commitment,
            &config.input.iter().map(|value| *value as i64).collect::<Vec<_>>(),
            prefix,
            Some(progress),
        ),
        _ => Err(Error::UnsupportedPrimitive),
    }
}

pub struct ProductionProvider;

impl PipelineProvider for ProductionProvider {
    fn commit_model(
        &self,
        request: &CommitRequest,
        ctx: &mut PipelineContext<'_>,
    ) -> Result<CommitReceipt> {
        let config = parse_case(&request.case_config)?;
        let commitment = build_commitment(&config)?;
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
        let commitment_digest = digest_bytes(&commitment_bytes);
        let config_digest = digest_bytes(&config_bytes);
        let protocol_digest = digest_bytes(&protocol_bytes);
        let parameters_digest = digest_bytes(&parameters_bytes);
        let files: Vec<(&str, &[u8])> = vec![
            ("public/model-commitment.bin", commitment_bytes.as_slice()),
            ("public/provider-config.bin", config_bytes.as_slice()),
            ("public/protocol.bin", protocol_bytes.as_slice()),
            ("private/parameters.bin", parameters_bytes.as_slice()),
        ];
        let digests: Vec<(&str, Digest)> = vec![
            ("public/model-commitment.bin", commitment_digest),
            ("public/provider-config.bin", config_digest),
            ("public/protocol.bin", protocol_digest),
            ("private/parameters.bin", parameters_digest),
        ];
        let manifest = commit_manifest(
            model,
            vec![
                ArtifactFile::new("public/model-commitment.bin", commitment_bytes.len() as u64, commitment_digest),
                ArtifactFile::new("public/provider-config.bin", config_bytes.len() as u64, config_digest),
                ArtifactFile::new("public/protocol.bin", protocol_bytes.len() as u64, protocol_digest),
                ArtifactFile::new("private/parameters.bin", parameters_bytes.len() as u64, parameters_digest),
            ],
        );
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
        let commitment = build_commitment(&config)?;
        let model = commitment.anchor.record.model_commitment;
        if request.expected_model_id != model {
            return Err(Error::StatementMismatch);
        }
        let statement = StatementData {
            case_id: config.case_id.clone(),
            model: config.model.clone(),
            model_commitment: model,
            tokens: config.tokens.clone(),
            expected_argmax: config.expected_argmax.clone(),
            expected_output: config.expected_output.clone(),
        };
        let statement_bytes = bincode::serialize(&statement).map_err(|_| Error::InvalidEncoding)?;
        let statement_digest = digest_bytes(&statement_bytes);
        let public = ModelCommitmentPublic::from_commitment(&commitment);
        let protocol_bytes = fs::read(request.commit_dir.join("public/protocol.bin"))
            .map_err(|_| Error::MissingCommitArtifact)?;
        let graph_identity = digest_bytes(&protocol_bytes);
        let prefix = trace_prefix(&public, &statement_digest, &graph_identity)?;
        let trace = execute_shared(&config, &commitment, &prefix, ctx.progress)?;
        if std::env::var("ZKIE_MEASURE_PROOF_SECTIONS").ok().as_deref() == Some("1") {
            let asset_bytes = bincode::serialized_size(&trace.assets)
                .map_err(|_| Error::InvalidEncoding)?;
            let step_bytes = bincode::serialized_size(&trace.steps)
                .map_err(|_| Error::InvalidEncoding)?;
            eprintln!(
                "ZKIE_ASSET_BYTES={asset_bytes} ZKIE_STEP_BYTES={step_bytes}"
            );
        }
        let proof_bytes = bincode::serialize(&trace).map_err(|_| Error::InvalidEncoding)?;
        let public_inputs = if config.model == "timesfm" {
            bincode::serialize(&config.expected_output).map_err(|_| Error::InvalidEncoding)?
        } else {
            bincode::serialize(&config.expected_argmax).map_err(|_| Error::InvalidEncoding)?
        };
        let proof_digest = digest_bytes(&proof_bytes);
        let public_inputs_digest = digest_bytes(&public_inputs);
        let files: Vec<(&str, &[u8])> = vec![
            ("statement.bin", statement_bytes.as_slice()),
            ("proof.bin", proof_bytes.as_slice()),
            ("public-inputs.bin", public_inputs.as_slice()),
        ];
        let digests: Vec<(&str, Digest)> = vec![
            ("statement.bin", statement_digest),
            ("proof.bin", proof_digest),
            ("public-inputs.bin", public_inputs_digest),
        ];
        let manifest = proof_manifest(
            model,
            vec![
                ArtifactFile::new("statement.bin", statement_bytes.len() as u64, statement_digest),
                ArtifactFile::new("proof.bin", proof_bytes.len() as u64, proof_digest),
                ArtifactFile::new("public-inputs.bin", public_inputs.len() as u64, public_inputs_digest),
            ],
        );
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
        let commitment_bytes = fs::read(request.public_bundle_dir.join("model-commitment.bin"))
            .map_err(|_| Error::IncompleteArtifact)?;
        let public: ModelCommitmentPublic =
            bincode::deserialize(&commitment_bytes).map_err(|_| Error::InvalidEncoding)?;
        if public.anchor.model_commitment != request.expected_model_id {
            return Err(Error::StatementMismatch);
        }
        let statement_bytes = fs::read(&request.statement).map_err(|_| Error::IncompleteArtifact)?;
        let statement: StatementData =
            bincode::deserialize(&statement_bytes).map_err(|_| Error::InvalidEncoding)?;
        let statement_digest = digest_bytes(&statement_bytes);
        let protocol_bytes = fs::read(request.public_bundle_dir.join("protocol.bin"))
            .map_err(|_| Error::IncompleteArtifact)?;
        let graph_identity = digest_bytes(&protocol_bytes);
        let prefix = trace_prefix(&public, &statement_digest, &graph_identity)?;
        if statement.model_commitment != request.expected_model_id {
            return Err(Error::StatementMismatch);
        }
        let proof_file =
            fs::File::open(&request.proof).map_err(|_| Error::IncompleteArtifact)?;
        let trace: SharedTrace = bincode::deserialize_from(std::io::BufReader::new(proof_file))
            .map_err(|_| Error::InvalidEncoding)?;
        let public_inputs = fs::read(&request.public_inputs).map_err(|_| Error::IncompleteArtifact)?;
        if statement.model == "timesfm" {
            let expected: Vec<i32> =
                bincode::deserialize(&public_inputs).map_err(|_| Error::InvalidEncoding)?;
            let output = verify_shared_trace_output(&public, &trace, &statement.tokens, &prefix)?;
            let matched = output.len() == expected.len()
                && output.iter().zip(&expected).all(|(value, wanted)| *value as i32 == *wanted);
            if !matched || expected != statement.expected_output {
                return Ok(VerifyReceipt {
                    accepted: false,
                    outcome: RunOutcome::Rejected,
                });
            }
        } else {
            let expected: Vec<u32> =
                bincode::deserialize(&public_inputs).map_err(|_| Error::InvalidEncoding)?;
            let argmax = verify_shared_trace(&public, &trace, &statement.tokens, &prefix)?;
            if argmax != expected || expected != statement.expected_argmax {
                return Ok(VerifyReceipt {
                    accepted: false,
                    outcome: RunOutcome::Rejected,
                });
            }
        }
        let step_units = trace.steps.len() as u64;
        let mut progress_event =
            ProgressEvent::new("verify", "verify", ProgressKind::WorkAdvanced);
        progress_event.completed_units = step_units;
        progress_event.total_units = Some(step_units);
        progress_event.unit = Some("relation_steps".into());
        let _ = ctx.progress.emit(progress_event);
        Ok(VerifyReceipt {
            accepted: true,
            outcome: RunOutcome::Passed,
        })
    }
}

pub fn production_provider() -> ProductionProvider {
    ProductionProvider
}
