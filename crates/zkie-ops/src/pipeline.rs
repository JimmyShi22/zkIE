//! Public three-phase lifecycle.
//!
//! `commit`, `prove` and `verify` are separate process entry points. `prove`
//! consumes an existing commit artifact and never auto-commits. `verify` only
//! sees public inputs and policy.

use std::path::PathBuf;

use zkie_core::artifact::COMPLETE_FILE;
use zkie_core::error::{Error, Result};
use zkie_core::progress::{ProgressEvent, ProgressKind, ProgressSink, RunOutcome};
use zkie_core::types::Digest;
use zkie_engine::context::{Metrics, RuntimeConfig};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitRequest {
    pub case_config: PathBuf,
    pub assets_manifest: PathBuf,
    pub parameters: PathBuf,
    pub output_dir: PathBuf,
    pub run_dir: PathBuf,
}

impl CommitRequest {
    pub fn validate(&self) -> Result<()> {
        if self.case_config.as_os_str().is_empty()
            || self.assets_manifest.as_os_str().is_empty()
            || self.parameters.as_os_str().is_empty()
            || self.output_dir.as_os_str().is_empty()
            || self.run_dir.as_os_str().is_empty()
        {
            return Err(Error::InvalidConfig);
        }
        if !self.case_config.exists()
            || !self.assets_manifest.exists()
            || !self.parameters.exists()
        {
            return Err(Error::AssetMismatch);
        }
        if self.output_dir.exists() {
            return Err(Error::ArtifactAlreadyExists);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProveRequest {
    pub commit_dir: PathBuf,
    pub input_request: PathBuf,
    pub expected_model_id: Digest,
    pub output_dir: PathBuf,
    pub run_dir: PathBuf,
}

impl ProveRequest {
    pub fn validate(&self) -> Result<()> {
        if !self.commit_dir.join(COMPLETE_FILE).exists() {
            return Err(Error::MissingCommitArtifact);
        }
        if self.expected_model_id == Digest::ZERO
            || self.input_request.as_os_str().is_empty()
            || self.output_dir.as_os_str().is_empty()
            || self.run_dir.as_os_str().is_empty()
        {
            return Err(Error::InvalidConfig);
        }
        if !self.input_request.exists() {
            return Err(Error::IncompleteArtifact);
        }
        if self.output_dir.exists() {
            return Err(Error::ArtifactAlreadyExists);
        }
        Ok(())
    }
}

/// Verifier inputs. There is deliberately no private directory and no weight
/// path in this type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyRequest {
    pub public_bundle_dir: PathBuf,
    pub statement: PathBuf,
    pub proof: PathBuf,
    pub public_inputs: PathBuf,
    pub expected_model_id: Digest,
    pub expected_policy: PathBuf,
    pub run_dir: PathBuf,
}

impl VerifyRequest {
    pub fn validate(&self) -> Result<()> {
        if self.expected_model_id == Digest::ZERO
            || self.public_bundle_dir.as_os_str().is_empty()
            || self.statement.as_os_str().is_empty()
            || self.proof.as_os_str().is_empty()
            || self.public_inputs.as_os_str().is_empty()
            || self.expected_policy.as_os_str().is_empty()
            || self.run_dir.as_os_str().is_empty()
        {
            return Err(Error::InvalidConfig);
        }
        if !self.public_bundle_dir.exists()
            || !self.statement.exists()
            || !self.proof.exists()
            || !self.public_inputs.exists()
            || !self.expected_policy.exists()
        {
            return Err(Error::IncompleteArtifact);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitReceipt {
    pub artifact_dir: PathBuf,
    pub manifest_digest: Digest,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofReceipt {
    pub proof_dir: PathBuf,
    pub statement_digest: Digest,
    /// Set only by a test provider; production providers never produce this.
    pub test_only: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputBindingMode {
    Expected,
    None,
}

pub struct VerifyReceipt {
    pub accepted: bool,
    pub outcome: RunOutcome,
    pub output_binding: Option<OutputBindingMode>,
    pub authenticated_output: Option<Vec<i64>>,
    pub authenticated_output_digest: Option<Digest>,
}

pub struct PipelineContext<'a> {
    pub progress: &'a dyn ProgressSink,
    pub metrics: &'a mut Metrics,
    pub runtime: &'a RuntimeConfig,
}

pub struct PipelineServices<'a> {
    pub provider: &'a dyn PipelineProvider,
    pub context: PipelineContext<'a>,
}

impl<'a> PipelineServices<'a> {
    pub fn new(
        provider: &'a dyn PipelineProvider,
        progress: &'a dyn ProgressSink,
        metrics: &'a mut Metrics,
        runtime: &'a RuntimeConfig,
    ) -> Self {
        Self {
            provider,
            context: PipelineContext {
                progress,
                metrics,
                runtime,
            },
        }
    }
}

pub trait PipelineProvider: Send + Sync {
    fn commit_model(
        &self,
        request: &CommitRequest,
        ctx: &mut PipelineContext<'_>,
    ) -> Result<CommitReceipt>;

    fn prove_request(
        &self,
        request: &ProveRequest,
        ctx: &mut PipelineContext<'_>,
    ) -> Result<ProofReceipt>;

    fn verify_request(
        &self,
        request: &VerifyRequest,
        ctx: &mut PipelineContext<'_>,
    ) -> Result<VerifyReceipt>;
}

pub fn commit_model(
    request: &CommitRequest,
    services: &mut PipelineServices<'_>,
) -> Result<CommitReceipt> {
    if let Err(error) = request.validate() {
        let _ = stage_event(
            &mut services.context,
            "commit",
            ProgressKind::StageFailed,
            Some(error.code()),
        );
        return Err(error);
    }
    stage_event(&mut services.context, "commit", ProgressKind::StageStarted, None)?;
    let provider = services.provider;
    let result = provider.commit_model(request, &mut services.context);
    finish_event(&mut services.context, "commit", result.is_ok())?;
    result
}

pub fn prove_request(
    request: &ProveRequest,
    services: &mut PipelineServices<'_>,
) -> Result<ProofReceipt> {
    if let Err(error) = request.validate() {
        let _ = stage_event(
            &mut services.context,
            "prove",
            ProgressKind::StageFailed,
            Some(error.code()),
        );
        return Err(error);
    }
    stage_event(&mut services.context, "prove", ProgressKind::StageStarted, None)?;
    let provider = services.provider;
    let result = provider.prove_request(request, &mut services.context);
    finish_event(&mut services.context, "prove", result.is_ok())?;
    result
}

pub fn verify_request(
    request: &VerifyRequest,
    services: &mut PipelineServices<'_>,
) -> Result<VerifyReceipt> {
    if let Err(error) = request.validate() {
        let _ = stage_event(
            &mut services.context,
            "verify",
            ProgressKind::StageFailed,
            Some(error.code()),
        );
        return Err(error);
    }
    stage_event(&mut services.context, "verify", ProgressKind::StageStarted, None)?;
    let provider = services.provider;
    let result = provider.verify_request(request, &mut services.context);
    finish_event(&mut services.context, "verify", result.is_ok())?;
    result
}

/// Production entry point: every lifecycle method is explicitly unsupported
/// until the real protocol provider is installed.
pub struct UnimplementedProvider;

impl PipelineProvider for UnimplementedProvider {
    fn commit_model(
        &self,
        _request: &CommitRequest,
        _ctx: &mut PipelineContext<'_>,
    ) -> Result<CommitReceipt> {
        Err(Error::UnsupportedPrimitive)
    }

    fn prove_request(
        &self,
        _request: &ProveRequest,
        _ctx: &mut PipelineContext<'_>,
    ) -> Result<ProofReceipt> {
        Err(Error::UnsupportedPrimitive)
    }

    fn verify_request(
        &self,
        _request: &VerifyRequest,
        _ctx: &mut PipelineContext<'_>,
    ) -> Result<VerifyReceipt> {
        Err(Error::UnsupportedPrimitive)
    }
}

fn stage_event(
    ctx: &mut PipelineContext<'_>,
    stage: &str,
    kind: ProgressKind,
    error: Option<&str>,
) -> Result<()> {
    let mut event = ProgressEvent::new("pipeline", stage, kind);
    event.elapsed_ms = ctx.metrics.get("elapsed_ms");
    event.error_code = error.map(str::to_owned);
    ctx.progress.emit(event)
}

fn finish_event(ctx: &mut PipelineContext<'_>, stage: &str, ok: bool) -> Result<()> {
    let kind = if ok {
        ProgressKind::StageFinished
    } else {
        ProgressKind::StageFailed
    };
    stage_event(ctx, stage, kind, if ok { None } else { Some("stage_failed") })
}
