//! `commit` subcommand.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use zkie_core::error::Result;
use zkie_core::progress::RunOutcome;
use zkie_engine::context::{Metrics, RuntimeConfig};
use zkie_ops::pipeline::{
    self, CommitReceipt, CommitRequest, PipelineProvider, PipelineServices,
};

use crate::progress::{HeartbeatGuard, JsonlProgressSink};
use crate::report::{outcome_for_error, parse_flags, path_flag, vm_hwm_kib, RunReport};

#[derive(Clone, Debug)]
pub struct CommitArgs {
    pub case_config: PathBuf,
    pub assets_manifest: PathBuf,
    pub parameters: PathBuf,
    pub output_dir: PathBuf,
    pub run_dir: PathBuf,
}

impl CommitArgs {
    pub fn parse(args: &[String]) -> Result<Self> {
        let flags = parse_flags(
            args,
            &["case", "assets-manifest", "parameters", "out", "run-dir"],
        )?;
        Ok(Self {
            case_config: path_flag(&flags, "case")?,
            assets_manifest: path_flag(&flags, "assets-manifest")?,
            parameters: path_flag(&flags, "parameters")?,
            output_dir: path_flag(&flags, "out")?,
            run_dir: path_flag(&flags, "run-dir")?,
        })
    }

    pub fn request(&self) -> CommitRequest {
        CommitRequest {
            case_config: self.case_config.clone(),
            assets_manifest: self.assets_manifest.clone(),
            parameters: self.parameters.clone(),
            output_dir: self.output_dir.clone(),
            run_dir: self.run_dir.clone(),
        }
    }
}

pub fn run(args: &[String], provider: &dyn PipelineProvider) -> Result<CommitReceipt> {
    let parsed = CommitArgs::parse(args)?;
    let request = parsed.request();
    let started = Instant::now();
    let progress = Arc::new(JsonlProgressSink::create(&request.run_dir, "commit")?);
    let _heartbeat = HeartbeatGuard::start(Arc::clone(&progress));
    let mut metrics = Metrics::new();
    let runtime = RuntimeConfig::default();
    let mut services = PipelineServices::new(provider, progress.as_ref(), &mut metrics, &runtime);
    let result = pipeline::commit_model(&request, &mut services);

    let mut report = match &result {
        Ok(_) => RunReport::new("commit", RunOutcome::Passed),
        Err(error) => {
            let mut report = RunReport::new("commit", outcome_for_error(error));
            report.error_code = Some(error.code().to_owned());
            report
        }
    };
    let _ = progress.run_finished(
        "commit",
        report.outcome,
        result.as_ref().err().map(|error| error.code()),
    );
    report.stage_wall_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    report.protocol_wall_ms = report.stage_wall_ms;
    report.vmhwm_kib = vm_hwm_kib();
    report.progress_path = Some(progress.path().to_path_buf());
    report.last_event_seq = progress.last_event_seq();
    report.last_completed_stage = Some("commit".into());
    let _ = report.write(&request.run_dir);
    let _ = crate::report::write_profile(&request.run_dir);
    result
}
