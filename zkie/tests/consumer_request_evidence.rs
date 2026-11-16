//! Release evidence for one request's consumer chain through the production
//! pipeline. Ignored by default; run explicitly with `ZKIE_CONSUMER_EV`
//! pointing at the round's evidence directory.
//!
//! The driver runs the real gpt2 case through commit, prove and verify. Prove
//! builds the consumer witness from the executed trace and writes the request
//! proof file beside the proof artifact; verify rebuilds the request plans from
//! the trusted model plan and the verified trace outputs and runs the first,
//! shared and second stages before it accepts. The negatives tamper the
//! auxiliary file and require the production verify entry to refuse.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use zkie_core::artifact::{digest_bytes, digest_to_hex};
use zkie_core::error::Result;
use zkie_core::progress::{ProgressEvent, ProgressSink};
use zkie_engine::context::{Metrics, RuntimeConfig};
use zkie_engine::pcs::request::field::Fp3;
use zkie_ops::pipeline::{
    commit_model, prove_request, verify_request, CommitRequest, PipelineProvider, PipelineServices,
    ProveRequest, VerifyRequest,
};
use zkie_ops::proof::consumer_proof::{ConsumerProofFile, CONSUMER_PROOF_FILE};
use zkie_ops::proof::model_commitment::ModelCommitmentPublic;

struct Quiet;

impl ProgressSink for Quiet {
    fn emit(&self, _event: ProgressEvent) -> Result<()> {
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        Ok(())
    }
}

fn memory_kib() -> (u64, u64) {
    let text = fs::read_to_string("/proc/self/status").unwrap_or_default();
    let mut hwm = 0u64;
    let mut rss = 0u64;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            hwm = rest.trim().trim_end_matches(" kB").trim().parse().unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("VmRSS:") {
            rss = rest.trim().trim_end_matches(" kB").trim().parse().unwrap_or(0);
        }
    }
    (hwm, rss)
}

fn case_path() -> PathBuf {
    std::env::var_os("ZKIE_CONSUMER_CASE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/data/i1-cases/gpt2/case.toml"))
}

#[allow(clippy::too_many_arguments)]
fn write_policy(
    path: &Path,
    model: &str,
    model_version: u32,
    case_id: &str,
    sequence: usize,
    anchor_id: &str,
    model_id: &str,
    identity: &str,
) {
    let text = format!(
        "policy_version = 1\nmodel = \"{model}\"\nmodel_version = {model_version}\n\n[[case]]\ncase_id = \"{case_id}\"\nsequence_length = {sequence}\nanchor_id = \"{anchor_id}\"\nmodel_id = \"{model_id}\"\noutput_mode = \"none\"\n{identity}"
    );
    fs::write(path, text).unwrap();
}

fn table_identity_text(
    model: &str,
    model_version: u32,
    case_id: &str,
    anchor_id: &str,
    model_id: &str,
    meta: &zkie_ops::proof::asset_table::AssetTableMeta,
) -> String {
    format!(
        "\n[case.limited_table_identity]\nversion = 1\nmodel = \"{model}\"\nmodel_version = {model_version}\ncase_id = \"{case_id}\"\nanchor_id = \"{anchor_id}\"\nmodel_id = \"{model_id}\"\nasset_id = {}\nasset_table_identity = \"{}\"\ntable_root = \"{}\"\ntable_record_digest = \"{}\"\ntable_configuration = \"{}\"\ncode_log_n = {}\ncode_ell = {}\ncode_r = {}\ntable_v = {}\nanswers = {}\nsalt_bytes = {}\ndistance_radius = {}\ndistance_later_radius = {}\ndistance_m_once = {}\ndistance_m_later = {}\nmatrix_rows_log = {}\nk_lo = 0\nk_col = {}\nprotocol_identity = \"{}\"\n",
        meta.asset_id,
        digest_to_hex(&meta.identity),
        digest_to_hex(&meta.root),
        digest_to_hex(&digest_bytes(&meta.record.encode())),
        digest_to_hex(&meta.configuration),
        meta.shape.code.log_n,
        meta.shape.code.ell,
        meta.shape.code.r,
        meta.shape.v,
        meta.shape.answers,
        meta.shape.salt_bytes,
        meta.shape.distances.radius,
        meta.shape.distances.later_radius,
        meta.shape.distances.m_once,
        meta.shape.distances.m_later,
        meta.shape.v,
        meta.shape.v,
        digest_to_hex(&zkie_core::types::Digest([0x5A; 32])),
    )
}

fn copy_proof(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).unwrap();
    for name in ["statement.bin", "proof.bin", "public-inputs.bin", CONSUMER_PROOF_FILE] {
        fs::copy(src.join(name), dst.join(name)).unwrap();
    }
}

fn runtime() -> RuntimeConfig {
    RuntimeConfig {
        memory_budget_bytes: 64 << 30,
        ..RuntimeConfig::default()
    }
}

fn verify(provider: &dyn PipelineProvider, request: &VerifyRequest) -> Result<bool> {
    let mut metrics = Metrics::new();
    let runtime = runtime();
    let quiet = Quiet;
    let mut services = PipelineServices::new(provider, &quiet, &mut metrics, &runtime);
    verify_request(request, &mut services).map(|receipt| receipt.accepted)
}

fn verify_dir(provider: &dyn PipelineProvider, policy: &Path, base: &Path, commit: &Path, model: &str) -> Result<bool> {
    let request = VerifyRequest {
        public_bundle_dir: commit.join("public"),
        statement: base.join("statement.bin"),
        proof: base.join("proof.bin"),
        public_inputs: base.join("public-inputs.bin"),
        expected_model_id: model_id(commit),
        expected_policy: policy.to_path_buf(),
        run_dir: base.join("verify-run"),
    };
    verify(provider, &request)
}

fn model_id(commit: &Path) -> zkie_core::types::Digest {
    let public: ModelCommitmentPublic =
        bincode::deserialize(&fs::read(commit.join("public/model-commitment.bin")).unwrap())
            .unwrap();
    public.anchor.model_commitment
}

#[test]
#[ignore = "release consumer evidence"]
fn the_request_consumer_chain_runs_through_the_production_pipeline() {
    let Some(ev) = std::env::var_os("ZKIE_CONSUMER_EV") else {
        return;
    };
    let ev = PathBuf::from(ev);
    let run = ev.join("run");
    if run.exists() {
        fs::remove_dir_all(&run).unwrap();
    }
    fs::create_dir_all(&run).unwrap();
    let case = case_path();
    let parameters = run.join("parameters.toml");
    fs::write(&parameters, "profile = \"case\"\n").unwrap();

    let provider = zkie::provider::production_provider();
    let commit_dir = run.join("commit");
    let mut metrics = Metrics::new();
    let runtime = runtime();
    let quiet = Quiet;

    let started = Instant::now();
    {
        let mut services = PipelineServices::new(&provider, &quiet, &mut metrics, &runtime);
        commit_model(
            &CommitRequest {
                case_config: case.clone(),
                assets_manifest: case.clone(),
                parameters: parameters.clone(),
                output_dir: commit_dir.clone(),
                run_dir: run.join("commit-run"),
            },
            &mut services,
        )
        .unwrap();
    }
    let commit_ms = started.elapsed().as_millis() as u64;

    let public: ModelCommitmentPublic =
        bincode::deserialize(&fs::read(commit_dir.join("public/model-commitment.bin")).unwrap())
            .unwrap();
    let config: zkie::provider::ProviderConfig = bincode::deserialize(
        &fs::read(commit_dir.join("public/provider-config.bin")).unwrap(),
    )
    .unwrap();
    let metas: Vec<zkie_ops::proof::asset_table::AssetTableMeta> = bincode::deserialize(
        &fs::read(commit_dir.join("public/asset-tables.bin")).unwrap(),
    )
    .unwrap();
    let identity = table_identity_text(
        &config.model,
        zkie_models_gpt2::plan::MODEL_VERSION,
        &config.case_id,
        &digest_to_hex(&public.anchor.anchor_id),
        &digest_to_hex(&public.anchor.model_commitment),
        &metas[0],
    );
    let policy = run.join("policy.toml");
    write_policy(
        &policy,
        &config.model,
        zkie_models_gpt2::plan::MODEL_VERSION,
        &config.case_id,
        config.seq_tokens,
        &digest_to_hex(&public.anchor.anchor_id),
        &digest_to_hex(&public.anchor.model_commitment),
        &identity,
    );

    let proof_dir = run.join("proof");
    let started = Instant::now();
    {
        let mut services = PipelineServices::new(&provider, &quiet, &mut metrics, &runtime);
        prove_request(
            &ProveRequest {
                commit_dir: commit_dir.clone(),
                input_request: case.clone(),
                expected_model_id: public.anchor.model_commitment,
                output_dir: proof_dir.clone(),
                run_dir: run.join("prove-run"),
            },
            &mut services,
        )
        .unwrap();
    }
    let prove_ms = started.elapsed().as_millis() as u64;
    let (hwm_after_prove, _) = memory_kib();

    let sidecar = proof_dir.join(CONSUMER_PROOF_FILE);
    assert!(sidecar.exists(), "prove writes the request proof file");
    let sidecar_bytes = fs::metadata(&sidecar).unwrap().len();
    let trace_bytes = fs::metadata(proof_dir.join("proof.bin")).unwrap().len();
    let statement_bytes = fs::metadata(proof_dir.join("statement.bin")).unwrap().len();

    let started = Instant::now();
    let accepted = verify_dir(&provider, &policy, &proof_dir, &commit_dir, &config.model).unwrap();
    let verify_ms = started.elapsed().as_millis() as u64;
    let (hwm_after_verify, _) = memory_kib();
    assert!(accepted, "the request with its consumer chain is accepted");

    // Negatives: the production verify entry refuses a missing or changed
    // auxiliary file before it can report an accepted request.
    let broken = run.join("broken");
    fs::create_dir_all(&broken).unwrap();

    let missing = broken.join("missing");
    copy_proof(&proof_dir, &missing);
    fs::remove_file(missing.join(CONSUMER_PROOF_FILE)).unwrap();
    let refused = verify_dir(&provider, &policy, &missing, &commit_dir, &config.model).is_err();
    assert!(refused, "a missing request proof file is refused");

    let file = ConsumerProofFile::read_from(&proof_dir).unwrap();

    let mut changed_member = file.plan_assembly().unwrap();
    changed_member.member.psi[0] = changed_member.member.psi[0].add(&Fp3::ONE);
    let mut tampered = file.clone();
    tampered.assembly = changed_member.encode().unwrap();
    let dir = broken.join("member");
    copy_proof(&proof_dir, &dir);
    tampered.write_to(&dir).unwrap();
    assert!(
        verify_dir(&provider, &policy, &dir, &commit_dir, &config.model).is_err(),
        "a changed plan member is refused"
    );

    let mut changed_target = file.plan_assembly().unwrap();
    changed_target.target = changed_target.target.add(&Fp3::ONE);
    let mut tampered = file.clone();
    tampered.assembly = changed_target.encode().unwrap();
    let dir = broken.join("target");
    copy_proof(&proof_dir, &dir);
    tampered.write_to(&dir).unwrap();
    assert!(
        verify_dir(&provider, &policy, &dir, &commit_dir, &config.model).is_err(),
        "a changed plan target is refused"
    );

    let mut changed_mask = file.plan_assembly().unwrap();
    changed_mask.mask_root = zkie_core::types::Digest([0x77; 32]);
    let mut tampered = file.clone();
    tampered.assembly = changed_mask.encode().unwrap();
    let dir = broken.join("mask");
    copy_proof(&proof_dir, &dir);
    tampered.write_to(&dir).unwrap();
    assert!(
        verify_dir(&provider, &policy, &dir, &commit_dir, &config.model).is_err(),
        "a changed mask root is refused"
    );

    let mut changed_raw = file.plan_assembly().unwrap();
    changed_raw.raw_roots[0] = zkie_core::types::Digest([0x78; 32]);
    let mut tampered = file.clone();
    tampered.assembly = changed_raw.encode().unwrap();
    let dir = broken.join("raw");
    copy_proof(&proof_dir, &dir);
    tampered.write_to(&dir).unwrap();
    assert!(
        verify_dir(&provider, &policy, &dir, &commit_dir, &config.model).is_err(),
        "a changed raw root is refused"
    );

    let mut bytes = file.encode().unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    let dir = broken.join("bytes");
    copy_proof(&proof_dir, &dir);
    fs::write(dir.join(CONSUMER_PROOF_FILE), &bytes).unwrap();
    assert!(
        verify_dir(&provider, &policy, &dir, &commit_dir, &config.model).is_err(),
        "a changed container byte is refused"
    );

    let evidence = format!(
        "{{\n  \"case\": \"{}\",\n  \"commit_ms\": {},\n  \"prove_ms\": {},\n  \"verify_ms\": {},\n  \"trace_bytes\": {},\n  \"statement_bytes\": {},\n  \"consumer_proof_bytes\": {},\n  \"vmhwm_after_prove_kib\": {},\n  \"vmhwm_after_verify_kib\": {},\n  \"accepted\": true,\n  \"negatives\": 6\n}}\n",
        config.case_id,
        commit_ms,
        prove_ms,
        verify_ms,
        trace_bytes,
        statement_bytes,
        sidecar_bytes,
        hwm_after_prove,
        hwm_after_verify,
    );
    fs::write(ev.join("consumer-evidence.json"), evidence).unwrap();
    let mut log = fs::File::create(ev.join("consumer-evidence.log")).unwrap();
    writeln!(log, "case {}", config.case_id).unwrap();
    writeln!(log, "commit {} ms", commit_ms).unwrap();
    writeln!(log, "prove {} ms", prove_ms).unwrap();
    writeln!(log, "verify {} ms", verify_ms).unwrap();
    writeln!(log, "trace bytes {}", trace_bytes).unwrap();
    writeln!(log, "statement bytes {}", statement_bytes).unwrap();
    writeln!(log, "consumer proof bytes {}", sidecar_bytes).unwrap();
    writeln!(log, "accepted; six negatives refused").unwrap();
}
