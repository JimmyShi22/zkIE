use std::env;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use zkie_core::common::field::PrimeField64;
use zkie_core::types::Digest;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::lin::{eval, LinClaim, LinShape};
use zkie_ops::proof::lin_session::{
    build_authenticated_lin_table, fp3_from_signed_word, prove_lin_session, verify_lin_session,
};
use zkie_ops::proof::model_commitment::{ModelCommitmentPublic, SharedTrace};
use zkie_ops::proof::session_baseline::{
    assemble_binding, verify_selected_step, SelectedMatmulStep, TrustedModel,
};

fn fp3(value: u64) -> Fp3 {
    fp3_from_signed_word(value)
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
    let psi: Vec<Fp3> = (1..=shape.k_col as u64).map(fp3).collect();
    let target = eval(&w_values, &psi).unwrap();
    let claim = LinClaim {
        weights,
        psi,
        masks: Vec::new(),
        target,
    };

    let started = Instant::now();
    let prover = prove_lin_session(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &claim,
        &w_values,
    )
    .unwrap();
    let verifier = verify_lin_session(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &claim,
        &prover.messages,
    )
    .unwrap();
    println!("lin_session_ms={}", started.elapsed().as_millis());
    assert_eq!(prover.shared_rho, verifier.shared_rho);
    assert_eq!(prover.shared, verifier.shared);
    assert_eq!(prover.zeta_rho, verifier.zeta_rho);
    assert_eq!(prover.zeta, verifier.zeta);
    assert_eq!(verifier.shared_dimension, shape.k_col as usize);
    assert!(verifier.eval_holds && verifier.dot_holds && verifier.fold_holds);
    assert!(verifier.control_w && verifier.control_omega);

    let mut messages = prover.messages.clone();
    messages.w.values[0] = fp3(
        messages.w.values[0]
            .c[0]
            .as_canonical_u64()
            .wrapping_add(1),
    );
    assert!(verify_lin_session(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &claim,
        &messages
    )
    .is_err());

    let mut messages = prover.messages.clone();
    messages.omega.values[0] = fp3(
        messages.omega.values[0]
            .c[0]
            .as_canonical_u64()
            .wrapping_add(1),
    );
    assert!(verify_lin_session(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &claim,
        &messages
    )
    .is_err());

    let mut messages = prover.messages.clone();
    messages.folded.values[0] = fp3(
        messages.folded.values[0]
            .c[0]
            .as_canonical_u64()
            .wrapping_add(1),
    );
    assert!(verify_lin_session(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &claim,
        &messages
    )
    .is_err());

    let mut messages = prover.messages.clone();
    messages.w.root.0[0] ^= 1;
    assert!(verify_lin_session(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &claim,
        &messages
    )
    .is_err());

    let mut claim_tamper = claim.clone();
    claim_tamper.target = fp3(claim.target.c[0].as_canonical_u64().wrapping_add(1));
    assert!(verify_lin_session(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &claim_tamper,
        &prover.messages
    )
    .is_err());

    let mut claim_tamper = claim.clone();
    claim_tamper.weights[0] = fp3(
        claim_tamper.weights[0]
            .c[0]
            .as_canonical_u64()
            .wrapping_add(1),
    );
    assert!(verify_lin_session(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &claim_tamper,
        &prover.messages
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

    assert!(verify_lin_session(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &claim,
        &prover.messages
    )
    .is_ok());
}
