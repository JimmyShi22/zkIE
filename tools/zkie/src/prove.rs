//! `prove` subcommand.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use zkie_core::artifact::digest_from_hex;
use zkie_core::error::Result;
use zkie_core::progress::RunOutcome;
use zkie_engine::context::{Metrics, RuntimeConfig};
use zkie_ops::pipeline::{
    self, PipelineProvider, PipelineServices, ProofReceipt, ProveRequest,
};

use crate::progress::{HeartbeatGuard, JsonlProgressSink};
use crate::report::{
    dir_size_bytes, outcome_for_error, parse_flags, path_flag, require_flag, vm_hwm_kib, RunReport,
};

#[derive(Clone, Debug)]
pub struct ProveArgs {
    pub commit_dir: PathBuf,
    pub input_request: PathBuf,
    pub expected_model_id: zkie_core::types::Digest,
    pub output_dir: PathBuf,
    pub run_dir: PathBuf,
}

impl ProveArgs {
    pub fn parse(args: &[String]) -> Result<Self> {
        let flags = parse_flags(
            args,
            &["commit", "input", "model-id", "out", "run-dir"],
        )?;
        Ok(Self {
            commit_dir: path_flag(&flags, "commit")?,
            input_request: path_flag(&flags, "input")?,
            expected_model_id: digest_from_hex(require_flag(&flags, "model-id")?)?,
            output_dir: path_flag(&flags, "out")?,
            run_dir: path_flag(&flags, "run-dir")?,
        })
    }

    pub fn request(&self) -> ProveRequest {
        ProveRequest {
            commit_dir: self.commit_dir.clone(),
            input_request: self.input_request.clone(),
            expected_model_id: self.expected_model_id,
            output_dir: self.output_dir.clone(),
            run_dir: self.run_dir.clone(),
        }
    }
}

pub fn run(args: &[String], provider: &dyn PipelineProvider) -> Result<ProofReceipt> {
    let parsed = ProveArgs::parse(args)?;
    let request = parsed.request();
    let started = Instant::now();
    let progress = Arc::new(JsonlProgressSink::create(&request.run_dir, "prove")?);
    let _heartbeat = HeartbeatGuard::start(Arc::clone(&progress));
    let mut metrics = Metrics::new();
    let runtime = RuntimeConfig::default();
    let mut services = PipelineServices::new(provider, progress.as_ref(), &mut metrics, &runtime);
    let result = pipeline::prove_request(&request, &mut services);

    let mut report = match &result {
        Ok(_) => RunReport::new("prove", RunOutcome::Passed),
        Err(error) => {
            let mut report = RunReport::new("prove", outcome_for_error(error));
            report.error_code = Some(error.code().to_owned());
            report
        }
    };
    let _ = progress.run_finished(
        "prove",
        report.outcome,
        result.as_ref().err().map(|error| error.code()),
    );
    report.stage_wall_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    report.protocol_wall_ms = report.stage_wall_ms;
    report.vmhwm_kib = vm_hwm_kib();
    report.proof_bytes = dir_size_bytes(&request.output_dir);
    report.progress_path = Some(progress.path().to_path_buf());
    report.last_event_seq = progress.last_event_seq();
    report.last_completed_stage = Some("prove".into());
    let _ = report.write(&request.run_dir);
    let _ = crate::report::write_profile(&request.run_dir);
    result
}
