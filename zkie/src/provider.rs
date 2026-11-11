//! Production pipeline provider for the three shared model traces.
//!
//! Commit publishes the shared model commitment and the public case inputs.
//! Prove re-opens the commitment, executes the shared trace and writes the
//! proof. Verify reads only the public commitment/proof/inputs.

use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
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
    CommitReceipt, CommitRequest, OutputBindingMode, PipelineContext, PipelineProvider,
    ProofReceipt, ProveRequest, VerifyReceipt, VerifyRequest,
};
use zkie_ops::proof::model_commitment::{
    verify_model_anchor, verify_shared_trace, verify_shared_trace_output,
    ModelCommitmentPublic, SharedTrace,
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
    progress: &dyn ProgressSink,
) -> Result<SharedTrace> {
    match config.model.as_str() {
        "gpt2" => gpt2::execute_shared_with_progress(
            commitment,
            &config.tokens.iter().map(|value| *value as i64).collect::<Vec<_>>(),
            Some(progress),
        ),
        "gemma3" => gemma3::execute_shared_with_progress(
            commitment,
            &config.tokens.iter().map(|value| *value as i64).collect::<Vec<_>>(),
            Some(progress),
        ),
        "timesfm" => timesfm::execute_shared_with_progress(
            commitment,
            &config.input.iter().map(|value| *value as i64).collect::<Vec<_>>(),
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
        let trace = execute_shared(&config, &commitment, ctx.progress)?;
        let proof_bytes = bincode::serialize(&trace).map_err(|_| Error::InvalidEncoding)?;
        let public_inputs = if config.model == "timesfm" {
            bincode::serialize(&config.expected_output).map_err(|_| Error::InvalidEncoding)?
        } else {
            bincode::serialize(&config.expected_argmax).map_err(|_| Error::InvalidEncoding)?
        };
        let statement_digest = digest_bytes(&statement_bytes);
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
        let policy_bytes =
            fs::read(&request.expected_policy).map_err(|_| Error::IncompleteArtifact)?;
        let policy = crate::policy::parse(&policy_bytes)?;
        let commitment_bytes = fs::read(request.public_bundle_dir.join("model-commitment.bin"))
            .map_err(|_| Error::IncompleteArtifact)?;
        let public: ModelCommitmentPublic =
            deserialize_exact(&commitment_bytes, Error::ArtifactIntegrityMismatch)?;
        let statement_bytes = fs::read(&request.statement).map_err(|_| Error::IncompleteArtifact)?;
        let statement: StatementData =
            deserialize_exact(&statement_bytes, Error::InvalidProof)?;
        let trace = read_proof_exact(&request.proof, ctx.runtime.memory_budget_bytes)?;
        let public_inputs =
            fs::read(&request.public_inputs).map_err(|_| Error::IncompleteArtifact)?;

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
        {
            return Err(Error::StatementMismatch);
        }

        let (bound, statement_matches) = match trusted.model.as_str() {
            "gpt2" => {
                if trusted.model_version != zkie_models_gpt2::plan::MODEL_VERSION {
                    return Err(Error::InvalidConfig);
                }
                let plan = zkie_models_gpt2::plan::trace_plan(statement.tokens.len())?;
                if zkie_models_gpt2::plan::model_commitment(&statement.tokens)? != model_id {
                    return Err(Error::StatementMismatch);
                }
                let parameters = zkie_models_gpt2::plan::parameters_digest()?;
                let specs = zkie_models_gpt2::plan::asset_specs();
                verify_model_anchor(
                    &public,
                    trusted.anchor_id,
                    model_id,
                    parameters,
                    &specs,
                )?;
                let argmax =
                    verify_shared_trace(&public, &trace, &statement.tokens, &plan)?;
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
                let plan = zkie_models_gemma3::plan::trace_plan(statement.tokens.len())?;
                if zkie_models_gemma3::plan::model_commitment(&statement.tokens)? != model_id {
                    return Err(Error::StatementMismatch);
                }
                let parameters = zkie_models_gemma3::plan::parameters_digest()?;
                let specs = zkie_models_gemma3::plan::asset_specs();
                verify_model_anchor(
                    &public,
                    trusted.anchor_id,
                    model_id,
                    parameters,
                    &specs,
                )?;
                let argmax =
                    verify_shared_trace(&public, &trace, &statement.tokens, &plan)?;
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
                let plan = zkie_models_timesfm::plan::trace_plan(patches)?;
                if zkie_models_timesfm::plan::model_commitment(&statement.tokens)? != model_id {
                    return Err(Error::StatementMismatch);
                }
                let parameters = zkie_models_timesfm::plan::parameters_digest()?;
                let specs = zkie_models_timesfm::plan::asset_specs();
                verify_model_anchor(
                    &public,
                    trusted.anchor_id,
                    model_id,
                    parameters,
                    &specs,
                )?;
                let output = verify_shared_trace_output(
                    &public,
                    &trace,
                    &statement.tokens,
                    &plan,
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
