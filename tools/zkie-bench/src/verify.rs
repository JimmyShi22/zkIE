//! `verify` subcommand.
//!
//! The parser rejects private-directory and weight flags: verification only
//! receives public material.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use zkie_core::artifact::digest_from_hex;
use zkie_core::error::Result;
use zkie_core::progress::RunOutcome;
use zkie_engine::context::{Metrics, RuntimeConfig};
use zkie_ops::pipeline::{
    self, PipelineProvider, PipelineServices, VerifyReceipt, VerifyRequest,
};

use crate::progress::{HeartbeatGuard, JsonlProgressSink};
use crate::report::{outcome_for_error, parse_flags, path_flag, require_flag, vm_hwm_kib, RunReport};

#[derive(Clone, Debug)]
pub struct VerifyArgs {
    pub public_bundle_dir: PathBuf,
    pub statement: PathBuf,
    pub proof: PathBuf,
    pub public_inputs: PathBuf,
    pub expected_model_id: zkie_core::types::Digest,
    pub expected_policy: PathBuf,
    pub run_dir: PathBuf,
}

impl VerifyArgs {
    pub fn parse(args: &[String]) -> Result<Self> {
        let flags = parse_flags(
            args,
            &[
                "public",
                "statement",
                "proof",
                "public-inputs",
                "model-id",
                "policy",
                "run-dir",
            ],
        )?;
        Ok(Self {
            public_bundle_dir: path_flag(&flags, "public")?,
            statement: path_flag(&flags, "statement")?,
            proof: path_flag(&flags, "proof")?,
            public_inputs: path_flag(&flags, "public-inputs")?,
            expected_model_id: digest_from_hex(require_flag(&flags, "model-id")?)?,
            expected_policy: path_flag(&flags, "policy")?,
            run_dir: path_flag(&flags, "run-dir")?,
        })
    }

    pub fn request(&self) -> VerifyRequest {
        VerifyRequest {
            public_bundle_dir: self.public_bundle_dir.clone(),
            statement: self.statement.clone(),
            proof: self.proof.clone(),
            public_inputs: self.public_inputs.clone(),
            expected_model_id: self.expected_model_id,
            expected_policy: self.expected_policy.clone(),
            run_dir: self.run_dir.clone(),
        }
    }
}

pub fn run(args: &[String], provider: &dyn PipelineProvider) -> Result<VerifyReceipt> {
    let parsed = VerifyArgs::parse(args)?;
    let request = parsed.request();
    let started = Instant::now();
    let progress = Arc::new(JsonlProgressSink::create(&request.run_dir, "verify")?);
    let _heartbeat = HeartbeatGuard::start(Arc::clone(&progress));
    let mut metrics = Metrics::new();
    let runtime = RuntimeConfig::default();
    let mut services = PipelineServices::new(provider, progress.as_ref(), &mut metrics, &runtime);
    let result = pipeline::verify_request(&request, &mut services);

    let mut report = match &result {
        Ok(receipt) => RunReport::new(
            "verify",
            if receipt.accepted {
                RunOutcome::Passed
            } else {
                RunOutcome::Rejected
            },
        ),
        Err(error) => {
            let mut report = RunReport::new("verify", outcome_for_error(error));
            report.error_code = Some(error.code().to_owned());
            report
        }
    };
    let _ = progress.run_finished(
        "verify",
        report.outcome,
        result.as_ref().err().map(|error| error.code()),
    );
    report.stage_wall_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    report.protocol_wall_ms = report.stage_wall_ms;
    report.vmhwm_kib = vm_hwm_kib();
    report.progress_path = Some(progress.path().to_path_buf());
    report.last_event_seq = progress.last_event_seq();
    report.last_completed_stage = Some("verify".into());
    let _ = report.write(&request.run_dir);
    let _ = crate::report::write_profile(&request.run_dir);
    result
}
