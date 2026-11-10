use std::env;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use zkie_core::error::Error;
use zkie_engine::pcs::request::bat::BatRowKind;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::relation::GuRelationKind;
use zkie_ops::proof::column_session::ColumnBindingInput;
use zkie_ops::proof::model_commitment::{
    verify_model_anchor, ModelCommitmentPublic, SharedTrace,
};
use zkie_ops::proof::send_session::{
    verify_send_session, verify_send_session_with_provider, FixtureSendProvider,
    FIXTURE_GU_ROOT,
};
use zkie_ops::proof::session_baseline::{
    selected_plan_identifier, SelectedMatmulStep, TrustedModel,
};

fn sample_target(outcome: &zkie_ops::proof::send_session::SendOutcome) -> Fp3 {
    outcome
        .combine
        .declarations
        .iter()
        .find(|row| row.kind == GuRelationKind::Sample)
        .expect("sample declaration")
        .target
}

#[test]
#[ignore = "requires ZKIE_SESSION_FIXTURE with the multi-GiB gpt2-seq16 artifact"]
fn selected_gpt2_fixture_send_data_flow() {
    let fixture = env::var("ZKIE_SESSION_FIXTURE").expect("set ZKIE_SESSION_FIXTURE");
    let root = Path::new(&fixture).join("gpt2-seq16");
    let public: ModelCommitmentPublic =
        bincode::deserialize(&fs::read(root.join("public/model-commitment.bin")).unwrap()).unwrap();
    let file = fs::File::open(root.join("proof.bin")).unwrap();
    let trace: SharedTrace = bincode::deserialize_from(BufReader::new(file)).unwrap();
    let policy = zkie::policy::parse(&fs::read(root.join("policy.toml")).unwrap()).unwrap();
    let request =
        zkie::policy::select(&policy, "gpt2-seq16", "gpt2", trace.tokens.len()).unwrap();
    let specs = zkie_models_gpt2::plan::asset_specs();
    let trusted = TrustedModel {
        anchor_id: request.anchor_id,
        model_id: request.model_id,
        parameters_digest: zkie_models_gpt2::plan::parameters_digest().unwrap(),
        asset_specs: &specs,
    };
    verify_model_anchor(
        &public,
        trusted.anchor_id,
        trusted.model_id,
        trusted.parameters_digest,
        trusted.asset_specs,
    )
    .unwrap();
    let opening = &trace.assets[192];
    public.verify_opening(opening).unwrap();

    let model_plan = zkie_models_gpt2::plan::trace_plan(16).unwrap();
    let selection = SelectedMatmulStep::select(&model_plan, 4444).unwrap();
    let plan_identifier = selected_plan_identifier(&public, &model_plan, &selection).unwrap();
    let input =
        ColumnBindingInput::from_plan(plan_identifier, request.case_id.as_bytes().to_vec()).unwrap();

    let started = Instant::now();
    let outcome = verify_send_session(
        &public,
        &trusted,
        &input,
        "gpt2/weights/lm_head_w_i32.bin",
        opening,
    )
    .unwrap();
    println!("send_session_ms={}", started.elapsed().as_millis());
    println!("seeds={:?}", outcome.seeds);
    println!("provider_answers={:?}", outcome.provider_answers);
    println!("sample_target={:?}", sample_target(&outcome));
    println!(
        "bat_order={:?} bat_targets={:?}",
        outcome.combine.bat_input.order, outcome.combine.bat_input.targets
    );

    assert_eq!(outcome.seeds.len(), 1);
    assert_eq!(outcome.provider_answers.len(), 1);
    assert_eq!(outcome.combine.sample_count, 1);
    assert_eq!(sample_target(&outcome), outcome.provider_answers[0]);
    assert!(outcome.combine.sample_answers_provider_constructed);
    assert!(!outcome.combine.verifier_standard_answer_comparison);
    assert_eq!(
        outcome.combine.bat_order[outcome.combine.bat_order.len() - 1],
        BatRowKind::Sample
    );
    assert_eq!(
        outcome.combine.bat_input.targets[outcome.combine.bat_input.count - 1],
        outcome.provider_answers[0]
    );
    assert!(outcome.combine.chain_handoff);

    // Auxiliary test oracle: recompute the provider's sample-form answers from
    // the plaintext fixture witness and the recorded seed. The verifier path
    // does not call this comparison.
    let provider = FixtureSendProvider::new(&outcome.gu_shape, outcome.gu_root).unwrap();
    let oracle_answers = provider.answers(&outcome.seeds).unwrap();
    assert_eq!(oracle_answers, outcome.provider_answers);

    // A syntactically valid wrong answer is registered as a pending sample
    // claim, not rejected or self-certified.
    let wrong = verify_send_session_with_provider(
        &public,
        &trusted,
        &input,
        "gpt2/weights/lm_head_w_i32.bin",
        opening,
        FIXTURE_GU_ROOT,
        |shape, seeds| {
            let mut answers =
                FixtureSendProvider::new(shape, FIXTURE_GU_ROOT)?.answers(seeds)?;
            answers[0] = answers[0].add(&Fp3::ONE);
            Ok(answers)
        },
    )
    .unwrap();
    assert_eq!(sample_target(&wrong), outcome.provider_answers[0].add(&Fp3::ONE));
    assert!(!wrong.combine.verifier_standard_answer_comparison);
    assert_eq!(wrong.combine.bat_input.targets[wrong.combine.bat_input.count - 1], sample_target(&wrong));

    // Truncated answer list fails closed before the verifier registers a claim.
    let truncated = verify_send_session_with_provider(
        &public,
        &trusted,
        &input,
        "gpt2/weights/lm_head_w_i32.bin",
        opening,
        FIXTURE_GU_ROOT,
        |_, _| Ok(Vec::new()),
    );
    assert!(matches!(truncated, Err(Error::ShapeMismatch)));
}
