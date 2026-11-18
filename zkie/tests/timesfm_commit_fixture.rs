//! Commit-only fixture for the TimesFM first asset consumer path.
//!
//! This test does not run prove or verify and does not claim a three-stage
//! result. It checks that the registered first projection gets its encoded
//! table and one-time key qualification in the commit artifact.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use zkie_core::error::Result;
use zkie_core::progress::{ProgressEvent, ProgressSink};
use zkie_engine::context::{Metrics, RuntimeConfig};
use zkie_ops::pipeline::{
    commit_model, CommitRequest, PipelineServices,
};
use zkie_ops::proof::asset_key::ASSET_KEY_BUNDLE;
use zkie_ops::proof::asset_table::AssetTableMeta;

struct Quiet;

impl ProgressSink for Quiet {
    fn emit(&self, _event: ProgressEvent) -> Result<()> {
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        Ok(())
    }
}

fn runtime() -> RuntimeConfig {
    RuntimeConfig {
        memory_budget_bytes: 64 << 30,
        ..RuntimeConfig::default()
    }
}

#[test]
#[ignore = "timesfm commit fixture; run explicitly"]
fn timesfm_commit_publishes_the_first_projection_key() {
    let case = std::env::var_os("ZKIE_TIMESFM_CASE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/timesfm/case.toml")
        });
    let root = PathBuf::from(
        std::env::var_os("ZKIE_TIMESFM_FIXTURE_ROOT")
            .unwrap_or_else(|| "/data/zkie-a01-fixture".into()),
    );
    let run = root.join("timesfm-commit");
    if run.exists() {
        fs::remove_dir_all(&run).unwrap();
    }
    fs::create_dir_all(&run).unwrap();
    let parameters = run.join("parameters.toml");
    fs::write(&parameters, "profile = \"case\"\n").unwrap();

    let provider = zkie::provider::production_provider();
    let mut metrics = Metrics::new();
    let runtime = runtime();
    let quiet = Quiet;
    let commit_dir = run.join("commit");
    let started = Instant::now();
    {
        let mut services = PipelineServices::new(&provider, &quiet, &mut metrics, &runtime);
        commit_model(
            &CommitRequest {
                case_config: case.clone(),
                assets_manifest: case,
                parameters,
                output_dir: commit_dir.clone(),
                run_dir: run.join("commit-run"),
            },
            &mut services,
        )
        .unwrap();
    }
    let commit_ms = started.elapsed().as_millis();

    let key = commit_dir.join(ASSET_KEY_BUNDLE);
    assert!(key.exists(), "the commit publishes the first asset key bundle");
    let metas: Vec<AssetTableMeta> = bincode::deserialize(
        &fs::read(commit_dir.join("public/asset-tables.bin")).unwrap(),
    )
    .unwrap();
    assert_eq!(metas.len(), 1, "one registered consumer table");
    assert_eq!(
        metas[0].path,
        "models/timesfm/weights/pro_out_w_i32.bin",
        "the first compatible square projection is the registered consumer asset"
    );
    assert_eq!(metas[0].shape.rows(), 2048);
    println!(
        "commit_ms={commit_ms} key_bytes={} key_prove_ms={} key_verify_ms={} table_rows={} table_root={:?}",
        fs::metadata(&key).unwrap().len(),
        metrics.get("key_prove_ms"),
        metrics.get("key_verify_ms"),
        metas[0].shape.rows(),
        metas[0].root,
    );
}
