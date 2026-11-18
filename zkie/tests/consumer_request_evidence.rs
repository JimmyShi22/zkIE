//! Release evidence for one request's consumer chain through the production
//! pipeline. Ignored by default; run explicitly with `ZKIE_CONSUMER_EV`
//! pointing at the round's evidence directory.
//!
//! The driver runs the real model case selected by ZKIE_CONSUMER_MODEL through
//! commit, prove and verify. The
//! commit proves the table's one-time key qualification and publishes it in
//! the public bundle; prove and verify re-verify that bundle per request into
//! an in-memory accepted key and check every accepted field. Prove builds the
//! consumer witness from the executed trace and writes the request proof file
//! beside the proof artifact; verify rebuilds the request plans from the
//! trusted model plan and the verified trace outputs and runs the first,
//! shared and second stages before it accepts. The trace carries no opening
//! for the consumer-authenticated weight: the consumer chain is its actual
//! proof, and a residual opening or a changed deferred output is refused. The
//! negatives tamper the auxiliary file, the trace and the key bundle and
//! require the production verify entry to refuse.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use zkie_core::artifact::{digest_bytes, digest_to_hex, COMPLETE_FILE, MANIFEST_FILE};
use zkie_core::error::Result;
use zkie_core::progress::{ProgressEvent, ProgressSink};
use zkie_engine::context::{Metrics, RuntimeConfig};
use zkie_engine::pcs::request::field::Fp3;
use zkie_ops::pipeline::{
    commit_model, prove_request, verify_request, CommitRequest, PipelineProvider, PipelineServices,
    ProveRequest, VerifyRequest,
};
use zkie_engine::artifact::load_manifest;
use zkie_ops::proof::asset_key::ASSET_KEY_BUNDLE;
use zkie_ops::proof::asset_table::decode_asset_table_randomness_set;
use zkie_ops::proof::consumer_proof::{ConsumerProofFile, CONSUMER_PROOF_FILE};
use zkie_ops::proof::asset_table::{AssetTableInput, AssetTableParams};
use zkie_ops::proof::model_commitment::{
    AssetStore, ModelCommitment, ModelCommitmentPublic, SharedTrace,
};
use zkie_ops::proof::plan::{PlanSource, TracePlan};

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

fn consumer_model() -> String {
    std::env::var("ZKIE_CONSUMER_MODEL").unwrap_or_else(|_| "gpt2".into())
}

fn case_path(model: &str) -> PathBuf {
    std::env::var_os("ZKIE_CONSUMER_CASE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("/data/zkIE/models/{model}/case.toml")))
}

fn model_version(model: &str) -> u32 {
    match model {
        "gpt2" => zkie_models_gpt2::plan::MODEL_VERSION,
        "gemma3" => zkie_models_gemma3::plan::MODEL_VERSION,
        _ => panic!("unsupported consumer model"),
    }
}

fn consumer_path(model: &str) -> &'static str {
    match model {
        "gpt2" => "gpt2/weights/L0_k_w_i32.bin",
        "gemma3" => zkie_models_gemma3::plan::FIRST_PROJECTION,
        _ => panic!("unsupported consumer model"),
    }
}

fn table_params(model: &str) -> Vec<AssetTableParams> {
    match model {
        "gpt2" => zkie_models_gpt2::plan::asset_table_params().unwrap(),
        "gemma3" => zkie_models_gemma3::plan::asset_table_params().unwrap(),
        _ => panic!("unsupported consumer model"),
    }
}

fn public_table_paths(model: &str) -> &'static [&'static str] {
    match model {
        "gpt2" => zkie_models_gpt2::plan::public_table_paths(),
        "gemma3" => zkie_models_gemma3::plan::public_table_paths(),
        _ => panic!("unsupported consumer model"),
    }
}

fn consumer_asset_paths(model: &str) -> &'static [&'static str] {
    match model {
        "gpt2" => zkie_models_gpt2::plan::consumer_asset_paths(),
        "gemma3" => zkie_models_gemma3::plan::consumer_asset_paths(),
        _ => panic!("unsupported consumer model"),
    }
}

fn trace_plan_public(model: &str, m: usize) -> TracePlan {
    match model {
        "gpt2" => zkie_models_gpt2::plan::trace_plan_public(m).unwrap(),
        "gemma3" => zkie_models_gemma3::plan::trace_plan_public(m).unwrap(),
        _ => panic!("unsupported consumer model"),
    }
}

fn build_commitment(
    model: &str,
    root: &Path,
    tokens: &[u32],
    tables: &[AssetTableInput],
) -> ModelCommitment {
    match model {
        "gpt2" => zkie_models_gpt2::executor::build_commitment(root, tokens, tables).unwrap(),
        "gemma3" => zkie_models_gemma3::executor::build_commitment(root, tokens, tables).unwrap(),
        _ => panic!("unsupported consumer model"),
    }
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
    for name in [
        "statement.bin",
        "proof.bin",
        "public-inputs.bin",
        CONSUMER_PROOF_FILE,
        MANIFEST_FILE,
        COMPLETE_FILE,
    ] {
        fs::copy(src.join(name), dst.join(name)).unwrap();
    }
}

fn copy_commit(src: &Path, dst: &Path) {
    fs::create_dir_all(dst.join("public")).unwrap();
    fs::create_dir_all(dst.join("private")).unwrap();
    for name in [MANIFEST_FILE, COMPLETE_FILE] {
        fs::copy(src.join(name), dst.join(name)).unwrap();
    }
    for dir in ["public", "private"] {
        for entry in fs::read_dir(src.join(dir)).unwrap() {
            let entry = entry.unwrap();
            fs::copy(entry.path(), dst.join(dir).join(entry.file_name())).unwrap();
        }
    }
}

/// Rewrites one file and its manifest entry so the artifact is internally
/// consistent: the refusal must then come from the protocol checks, not only
/// from the container digest.
fn rewrite_manifest_file(root: &Path, relative_path: &str, bytes: &[u8]) {
    fs::write(root.join(relative_path), bytes).unwrap();
    let mut manifest = load_manifest(root).unwrap();
    let entry = manifest
        .files
        .iter_mut()
        .find(|entry| entry.relative_path == relative_path)
        .expect("the manifest lists the file");
    entry.byte_length = bytes.len() as u64;
    entry.content_digest = digest_bytes(bytes);
    fs::write(root.join(MANIFEST_FILE), manifest.to_json().unwrap()).unwrap();
    fs::write(
        root.join(COMPLETE_FILE),
        format!("{}\n", digest_to_hex(&manifest.manifest_digest().unwrap())),
    )
    .unwrap();
}

fn runtime() -> RuntimeConfig {
    RuntimeConfig {
        memory_budget_bytes: 64 << 30,
        ..RuntimeConfig::default()
    }
}

fn verify(
    provider: &dyn PipelineProvider,
    request: &VerifyRequest,
) -> Result<(bool, Metrics)> {
    let mut metrics = Metrics::new();
    let runtime = runtime();
    let quiet = Quiet;
    let mut services = PipelineServices::new(provider, &quiet, &mut metrics, &runtime);
    verify_request(request, &mut services).map(|receipt| (receipt.accepted, metrics))
}

fn verify_dir(
    provider: &dyn PipelineProvider,
    policy: &Path,
    base: &Path,
    commit: &Path,
    _model: &str,
) -> Result<(bool, Metrics)> {
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
    let model = consumer_model();
    let case = case_path(&model);
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
    let key_prove_ms = metrics.get("key_prove_ms");
    let key_commit_verify_ms = metrics.get("key_verify_ms");
    let key_file_bytes = fs::metadata(commit_dir.join(ASSET_KEY_BUNDLE)).unwrap().len();

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
        model_version(&model),
        &config.case_id,
        &digest_to_hex(&public.anchor.anchor_id),
        &digest_to_hex(&public.anchor.model_commitment),
        &metas[0],
    );
    let policy = run.join("policy.toml");
    write_policy(
        &policy,
        &config.model,
        model_version(&model),
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
    let (accepted, verify_metrics) =
        verify_dir(&provider, &policy, &proof_dir, &commit_dir, &config.model).unwrap();
    let verify_ms = started.elapsed().as_millis() as u64;
    let key_request_verify_ms = verify_metrics.get("key_verify_ms");
    let (hwm_after_verify, _) = memory_kib();
    assert!(accepted, "the request with its consumer chain is accepted");

    // The trace carries no opening for the consumer-authenticated weight. The
    // opening is rebuilt from the same commit to record the exact bytes the
    // deletion removed, so the before/after comparison stays on one basis.
    let trace_file_bytes = fs::read(proof_dir.join("proof.bin")).unwrap();
    let trace: SharedTrace = bincode::deserialize(&trace_file_bytes).unwrap();
    let consumer_path = consumer_path(&model);
    let consumer_meta = public
        .assets
        .iter()
        .find(|asset| asset.path == consumer_path)
        .unwrap();
    assert!(
        !trace
            .assets
            .iter()
            .any(|opening| opening.asset_id == consumer_meta.asset_id),
        "the trace carries no opening for the consumer asset"
    );
    let table_params = table_params(&model);
    let randomness_bytes =
        fs::read(commit_dir.join("private/asset-table-randomness.bin")).unwrap();
    let tables = decode_asset_table_randomness_set(&randomness_bytes, &table_params).unwrap();
    let commitment = build_commitment(
        &model,
        Path::new(&config.asset_root),
        &config.tokens,
        &tables,
    );
    assert_eq!(
        trace.assets.len(),
        commitment.assets.len()
            - public_table_paths(&model).len()
            - consumer_asset_paths(&model).len(),
        "the trace omits exactly the public tables and the consumer asset"
    );
    let consumer_index = commitment
        .assets
        .iter()
        .position(|asset| asset.path == consumer_path)
        .unwrap();
    let removed_opening = AssetStore::open(&commitment)
        .unwrap()
        .open_asset(consumer_index)
        .unwrap();
    let removed_opening_bytes = bincode::serialize(&removed_opening).unwrap().len() as u64;
    let deferred: Vec<usize> = trace_plan_public(&model, config.tokens.len())
        .steps
        .iter()
        .enumerate()
        .filter(|(_, step)| {
            step.inputs
                .iter()
                .any(|source| matches!(source, PlanSource::ConsumerAsset { .. }))
        })
        .map(|(index, _)| index)
        .collect();
    let expected_deferred = if model == "gpt2" { 12 } else { 4 };
    assert_eq!(
        deferred.len(),
        expected_deferred,
        "the deferred projection steps of this model"
    );

    // Negatives: the production verify entry refuses a missing or changed
    // auxiliary file, a residual opening and a changed deferred output before
    // it can report an accepted request.
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
    rewrite_manifest_file(&dir, CONSUMER_PROOF_FILE, &tampered.encode().unwrap());
    assert!(
        verify_dir(&provider, &policy, &dir, &commit_dir, &config.model).is_err(),
        "a changed plan member is refused"
    );

    let mut changed_member_target = file.plan_assembly().unwrap();
    changed_member_target.member.target = changed_member_target.member.target.add(&Fp3::ONE);
    let mut tampered = file.clone();
    tampered.assembly = changed_member_target.encode().unwrap();
    let dir = broken.join("member-target");
    copy_proof(&proof_dir, &dir);
    rewrite_manifest_file(&dir, CONSUMER_PROOF_FILE, &tampered.encode().unwrap());
    assert!(
        verify_dir(&provider, &policy, &dir, &commit_dir, &config.model).is_err(),
        "a plan member target that is not the PSC pending is refused"
    );

    let mut changed_target = file.plan_assembly().unwrap();
    changed_target.target = changed_target.target.add(&Fp3::ONE);
    let mut tampered = file.clone();
    tampered.assembly = changed_target.encode().unwrap();
    let dir = broken.join("target");
    copy_proof(&proof_dir, &dir);
    rewrite_manifest_file(&dir, CONSUMER_PROOF_FILE, &tampered.encode().unwrap());
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
    rewrite_manifest_file(&dir, CONSUMER_PROOF_FILE, &tampered.encode().unwrap());
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
    rewrite_manifest_file(&dir, CONSUMER_PROOF_FILE, &tampered.encode().unwrap());
    assert!(
        verify_dir(&provider, &policy, &dir, &commit_dir, &config.model).is_err(),
        "a changed raw root is refused"
    );

    let mut bytes = file.encode().unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    let dir = broken.join("bytes");
    copy_proof(&proof_dir, &dir);
    rewrite_manifest_file(&dir, CONSUMER_PROOF_FILE, &bytes);
    assert!(
        verify_dir(&provider, &policy, &dir, &commit_dir, &config.model).is_err(),
        "a changed container byte is refused"
    );

    // Deletion negatives: the removed opening put back into the trace, and one
    // deferred step's output changed. Each proof file is rewritten with an
    // internally consistent manifest, so the refusal comes from the trace and
    // consumer checks, not from the container digest.
    let mut residual = trace.clone();
    residual.assets.push(removed_opening);
    let residual_bytes = bincode::serialize(&residual).unwrap();
    let dir = broken.join("residual-opening");
    copy_proof(&proof_dir, &dir);
    rewrite_manifest_file(&dir, "proof.bin", &residual_bytes);
    assert!(
        verify_dir(&provider, &policy, &dir, &commit_dir, &config.model).is_err(),
        "a residual consumer asset opening is refused"
    );

    // The same residue with a changed row is refused as well, so neither a
    // valid nor a tampered leftover opening can re-enter the proof.
    residual.assets.last_mut().unwrap().rows[0].values[0] ^= 1;
    let tampered_residual_bytes = bincode::serialize(&residual).unwrap();
    let dir = broken.join("residual-opening-row");
    copy_proof(&proof_dir, &dir);
    rewrite_manifest_file(&dir, "proof.bin", &tampered_residual_bytes);
    assert!(
        verify_dir(&provider, &policy, &dir, &commit_dir, &config.model).is_err(),
        "a tampered residual consumer asset opening is refused"
    );

    let mut changed_trace = trace.clone();
    let opening = changed_trace.steps[deferred[0]].message.proof.outputs[0]
        .first_mut()
        .unwrap();
    opening.values[0] ^= 1;
    let changed_bytes = bincode::serialize(&changed_trace).unwrap();
    let dir = broken.join("deferred-output");
    copy_proof(&proof_dir, &dir);
    rewrite_manifest_file(&dir, "proof.bin", &changed_bytes);
    assert!(
        verify_dir(&provider, &policy, &dir, &commit_dir, &config.model).is_err(),
        "a changed deferred output is refused by the consumer chain"
    );

    // Qualification negatives: a missing key bundle and a byte-changed key
    // bundle whose commit manifest is internally consistent. Both are refused
    // by the production verify entry.
    let broken_commit = broken.join("commit-missing-key");
    copy_commit(&commit_dir, &broken_commit);
    fs::remove_file(broken_commit.join(ASSET_KEY_BUNDLE)).unwrap();
    let refused = verify_dir(&provider, &policy, &proof_dir, &broken_commit, &config.model).is_err();
    assert!(refused, "a missing key bundle is refused");

    let broken_commit = broken.join("commit-bad-key");
    copy_commit(&commit_dir, &broken_commit);
    let mut key_bytes = fs::read(broken_commit.join(ASSET_KEY_BUNDLE)).unwrap();
    let last = key_bytes.len() - 1;
    key_bytes[last] ^= 1;
    rewrite_manifest_file(&broken_commit, ASSET_KEY_BUNDLE, &key_bytes);
    let refused = verify_dir(&provider, &policy, &proof_dir, &broken_commit, &config.model).is_err();
    assert!(refused, "a changed key bundle is refused");

    // Version negative: a policy written before the committed table identity
    // became mandatory is refused by the production entry.
    let old_policy = run.join("old-policy.toml");
    let policy_text = fs::read_to_string(&policy).unwrap();
    let cut = policy_text
        .find("[case.limited_table_identity]")
        .unwrap_or(policy_text.len());
    fs::write(&old_policy, &policy_text[..cut]).unwrap();
    let refused = verify_dir(&provider, &old_policy, &proof_dir, &commit_dir, &config.model).is_err();
    assert!(
        refused,
        "a policy without the committed table identity is refused"
    );

    let evidence = format!(
        "{{\n  \"case\": \"{}\",\n  \"commit_ms\": {},\n  \"prove_ms\": {},\n  \"verify_ms\": {},\n  \"trace_bytes\": {},\n  \"trace_assets\": {},\n  \"removed_opening_bytes\": {},\n  \"trace_with_opening_bytes\": {},\n  \"statement_bytes\": {},\n  \"consumer_proof_bytes\": {},\n  \"key_prove_ms\": {},\n  \"key_commit_verify_ms\": {},\n  \"key_request_verify_ms\": {},\n  \"key_file_bytes\": {},\n  \"vmhwm_after_prove_kib\": {},\n  \"vmhwm_after_verify_kib\": {},\n  \"accepted\": true,\n  \"negatives\": 13,\n  \"consumer_matrix_negatives\": 12,\n  \"version_negatives\": 1\n}}\n",
        config.case_id,
        commit_ms,
        prove_ms,
        verify_ms,
        trace_bytes,
        trace.assets.len(),
        removed_opening_bytes,
        trace_bytes + removed_opening_bytes,
        statement_bytes,
        sidecar_bytes,
        key_prove_ms,
        key_commit_verify_ms,
        key_request_verify_ms,
        key_file_bytes,
        hwm_after_prove,
        hwm_after_verify,
    );
    fs::write(ev.join("consumer-evidence.json"), evidence).unwrap();
    let mut log = fs::File::create(ev.join("consumer-evidence.log")).unwrap();
    writeln!(log, "case {}", config.case_id).unwrap();
    writeln!(log, "commit {} ms", commit_ms).unwrap();
    writeln!(log, "prove {} ms", prove_ms).unwrap();
    writeln!(log, "verify {} ms", verify_ms).unwrap();
    writeln!(log, "trace bytes {} ({} assets; {} bytes of the consumer opening removed; with opening {})", trace_bytes, trace.assets.len(), removed_opening_bytes, trace_bytes + removed_opening_bytes).unwrap();
    writeln!(log, "statement bytes {}", statement_bytes).unwrap();
    writeln!(log, "consumer proof bytes {}", sidecar_bytes).unwrap();
    writeln!(log, "key prove {} ms, commit verify {} ms, request verify {} ms, file {} bytes", key_prove_ms, key_commit_verify_ms, key_request_verify_ms, key_file_bytes).unwrap();
    writeln!(log, "accepted; twelve consumer-matrix negatives and one version negative refused").unwrap();
}
