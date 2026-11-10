use std::env;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use zkie_core::common::field::PrimeField64;
use zkie_core::types::Digest;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::lin::LinShape;
use zkie_ops::proof::lin_session::{
    build_authenticated_lin_table, fp3_from_signed_word, prove_lin_request, verify_lin_request,
    verify_lin_request_bytes, LinRequestMessages,
};
use zkie_ops::proof::model_commitment::{ModelCommitmentPublic, SharedTrace};
use zkie_ops::proof::session_baseline::{
    assemble_binding, verify_selected_step, SelectedMatmulStep, TrustedModel,
};

fn fp3(value: u64) -> Fp3 {
    fp3_from_signed_word(value)
}

fn next(value: &Fp3) -> Fp3 {
    fp3(value.c[0].as_canonical_u64().wrapping_add(1))
}

#[test]
#[ignore = "requires ZKIE_SESSION_FIXTURE with the multi-GiB gpt2-seq16 artifact"]
fn selected_gpt2_linear_claim_session() {
    let fixture = env::var("ZKIE_SESSION_FIXTURE").expect("set ZKIE_SESSION_FIXTURE");
    let root = Path::new(&fixture).join("gpt2-seq16");
    let public: ModelCommitmentPublic =
        bincode::deserialize(&fs::read(root.join("public/model-commitment.bin")).unwrap()).unwrap();
    let file = fs::File::open(root.join("proof.bin")).unwrap();
    let mut trace: SharedTrace = bincode::deserialize_from(BufReader::new(file)).unwrap();
    let plan = zkie_models_gpt2::plan::trace_plan(16).unwrap();
    let selection = SelectedMatmulStep::select(&plan, 4444).unwrap();

    let policy = zkie::policy::parse(&fs::read(root.join("policy.toml")).unwrap()).unwrap();
    let trusted_request =
        zkie::policy::select(&policy, "gpt2-seq16", "gpt2", trace.tokens.len()).unwrap();
    let specs = zkie_models_gpt2::plan::asset_specs();
    let trusted = TrustedModel {
        anchor_id: trusted_request.anchor_id,
        model_id: trusted_request.model_id,
        parameters_digest: zkie_models_gpt2::plan::parameters_digest().unwrap(),
        asset_specs: &specs,
    };

    let control_started = Instant::now();
    verify_selected_step(&public, &trace, &plan, &selection, &trusted)
        .expect("authenticated full-recompute control");
    println!(
        "full_recompute_control_ms={}",
        control_started.elapsed().as_millis()
    );

    let binding = assemble_binding(&public, &plan, &selection).unwrap();
    let shape = LinShape::new(1024, 0, 64, 1024).unwrap();
    let asset_path = "gpt2/weights/lm_head_w_i32.bin";
    let weights: Vec<Fp3> = trace.steps[4444].message.proof.inputs[0][0]
        .values
        .iter()
        .map(|value| fp3(*value))
        .collect();
    let w_values: Vec<Fp3> = trace.steps[4444].message.proof.outputs[0][0]
        .values
        .iter()
        .take(64)
        .map(|value| fp3(*value))
        .collect();
    let masks = Vec::new();

    let started = Instant::now();
    let prover = prove_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &weights,
        &masks,
        &w_values,
    )
    .unwrap();
    let verifier = verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &prover,
    )
    .unwrap();
    println!("lin_request_ms={}", started.elapsed().as_millis());
    assert_eq!(prover.first.psi, verifier.psi);
    assert_eq!(prover.shared.point, verifier.shared);
    assert_eq!(prover.shared.zeta, verifier.zeta);
    assert_eq!(prover.shared.first_seal, verifier.first_seal);
    assert_eq!(prover.second.shared_seal, verifier.shared_seal);
    assert_eq!(prover.second.seal, verifier.seal);
    assert_eq!(verifier.shared.len(), shape.k_col as usize);
    assert!(verifier.eval_holds && verifier.dot_holds && verifier.fold_holds);
    assert!(verifier.control_w && verifier.control_omega);

    let bytes = prover.encode();
    let decoded = LinRequestMessages::decode(&bytes).unwrap();
    assert_eq!(decoded, prover);
    let mut trailing = bytes;
    trailing.push(0);
    assert!(LinRequestMessages::decode(&trailing).is_err());
    assert!(verify_lin_request_bytes(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &trailing
    )
    .is_err());

    let mut messages = prover.clone();
    messages.first.w.values[0] = next(&messages.first.w.values[0]);
    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &messages
    )
    .is_err());

    let mut messages = prover.clone();
    messages.first.weights.values[0] = next(&messages.first.weights.values[0]);
    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &messages
    )
    .is_err());

    let mut messages = prover.clone();
    messages.shared.omega.values[0] = next(&messages.shared.omega.values[0]);
    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &messages
    )
    .is_err());

    let mut messages = prover.clone();
    messages.shared.folded.values[0] = next(&messages.shared.folded.values[0]);
    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &messages
    )
    .is_err());

    let mut messages = prover.clone();
    messages.first.psi[0] = next(&messages.first.psi[0]);
    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &messages
    )
    .is_err());

    let mut messages = prover.clone();
    messages.shared.point[0] = next(&messages.shared.point[0]);
    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &messages
    )
    .is_err());

    let mut messages = prover.clone();
    messages.shared.zeta[0] = next(&messages.shared.zeta[0]);
    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &messages
    )
    .is_err());

    let mut messages = prover.clone();
    messages.shared.first_seal.0[0] ^= 1;
    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &messages
    )
    .is_err());

    let mut messages = prover.clone();
    messages.second.shared_seal.0[0] ^= 1;
    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &messages
    )
    .is_err());

    let mut messages = prover.clone();
    messages.second.seal.0[0] ^= 1;
    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &messages
    )
    .is_err());

    let mut messages = prover.clone();
    messages.first.w.root.0[0] ^= 1;
    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &messages
    )
    .is_err());

    let original = trace.assets[192].rows[0].values[0];
    trace.assets[192].rows[0].values[0] ^= 1;
    assert!(build_authenticated_lin_table(
        &public,
        &trusted,
        asset_path,
        &trace.assets[192],
        shape
    )
    .is_err());
    trace.assets[192].rows[0].values[0] = original;

    trace.assets[192].rows[0].path.push(Digest([9; 32]));
    assert!(build_authenticated_lin_table(
        &public,
        &trusted,
        asset_path,
        &trace.assets[192],
        shape
    )
    .is_err());
    trace.assets[192].rows[0].path.pop();

    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &prover
    )
    .is_ok());
}

#[test]
#[ignore = "requires ZKIE_SESSION_FIXTURE with the multi-GiB gpt2-seq16 artifact"]
fn selected_gpt2_output_head_request_session() {
    use zkie_ops::proof::output_head_lin::{
        output_head_shape, prove_output_head_lin_request, select_output_head_step,
        verify_output_head_lin_request_bytes,
    };

    let fixture = env::var("ZKIE_SESSION_FIXTURE").expect("set ZKIE_SESSION_FIXTURE");
    let root = Path::new(&fixture).join("gpt2-seq16");
    let public: ModelCommitmentPublic =
        bincode::deserialize(&fs::read(root.join("public/model-commitment.bin")).unwrap()).unwrap();
    let file = fs::File::open(root.join("proof.bin")).unwrap();
    let trace_bytes = file.metadata().unwrap().len();
    let mut trace: SharedTrace = bincode::deserialize_from(BufReader::new(file)).unwrap();
    let plan = zkie_models_gpt2::plan::trace_plan(16).unwrap();
    let step = select_output_head_step(&plan).unwrap();

    let policy = zkie::policy::parse(&fs::read(root.join("policy.toml")).unwrap()).unwrap();
    let trusted_request =
        zkie::policy::select(&policy, "gpt2-seq16", "gpt2", trace.tokens.len()).unwrap();
    let specs = zkie_models_gpt2::plan::asset_specs();
    let trusted = TrustedModel {
        anchor_id: trusted_request.anchor_id,
        model_id: trusted_request.model_id,
        parameters_digest: zkie_models_gpt2::plan::parameters_digest().unwrap(),
        asset_specs: &specs,
    };
    let shape = output_head_shape().unwrap();

    let started = Instant::now();
    let proof = prove_output_head_lin_request(
        &public,
        &trusted,
        &plan,
        &trace,
        &trace.tokens,
        shape,
        &[],
    )
    .unwrap();
    let prove_ms = started.elapsed().as_millis();

    let bytes = proof.encode();
    let started = Instant::now();
    let outcome = verify_output_head_lin_request_bytes(
        &public,
        &trusted,
        &plan,
        &trace,
        &trace.tokens,
        shape,
        &bytes,
        &[],
    )
    .unwrap();
    let verify_ms = started.elapsed().as_millis();
    assert_eq!(outcome.outcomes.len(), 16);
    for value in &outcome.outcomes {
        assert!(value.eval_holds && value.dot_holds && value.fold_holds);
        assert!(value.control_w && value.control_omega);
    }
    assert_eq!(outcome.proof_bytes, bytes.len());
    println!("output_head_step={}", step.step);
    println!("trace_proof_bytes={trace_bytes}");
    println!("request_proof_bytes={}", bytes.len());
    println!(
        "total_verification_bytes={}",
        trace_bytes + bytes.len() as u64
    );
    println!("prove_ms={prove_ms}");
    println!("verify_ms={verify_ms}");

    let padded_column = 60000;
    let original = trace.steps[step.step].message.proof.outputs[0][0].values[padded_column];
    trace.steps[step.step].message.proof.outputs[0][0].values[padded_column] = original ^ 1;
    assert!(verify_output_head_lin_request_bytes(
        &public,
        &trusted,
        &plan,
        &trace,
        &trace.tokens,
        shape,
        &bytes,
        &[]
    )
    .is_err());
    trace.steps[step.step].message.proof.outputs[0][0].values[padded_column] = original;

    let original = trace.steps[step.step].message.proof.inputs[0][0].values[0];
    trace.steps[step.step].message.proof.inputs[0][0].values[0] = original ^ 1;
    assert!(verify_output_head_lin_request_bytes(
        &public,
        &trusted,
        &plan,
        &trace,
        &trace.tokens,
        shape,
        &bytes,
        &[]
    )
    .is_err());
    trace.steps[step.step].message.proof.inputs[0][0].values[0] = original;

    let meta = public
        .assets
        .iter()
        .find(|asset| asset.path == step.asset_path)
        .unwrap();
    let asset_index = trace
        .assets
        .iter()
        .position(|opening| opening.asset_id == meta.asset_id)
        .unwrap();
    let original = trace.assets[asset_index].root;
    trace.assets[asset_index].root.0[0] ^= 1;
    assert!(verify_output_head_lin_request_bytes(
        &public,
        &trusted,
        &plan,
        &trace,
        &trace.tokens,
        shape,
        &bytes,
        &[]
    )
    .is_err());
    trace.assets[asset_index].root = original;

    let mut changed_plan = plan.clone();
    changed_plan.steps[step.step].inputs[0] =
        zkie_ops::proof::plan::PlanSource::prev(step.activation_step.saturating_sub(1), 0);
    assert!(verify_output_head_lin_request_bytes(
        &public,
        &trusted,
        &changed_plan,
        &trace,
        &trace.tokens,
        shape,
        &bytes,
        &[]
    )
    .is_err());

    assert!(verify_output_head_lin_request_bytes(
        &public,
        &trusted,
        &plan,
        &trace,
        &trace.tokens,
        shape,
        &bytes,
        &[]
    )
    .is_ok());
}
