use std::env;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;
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
    verify_send_session, verify_send_session_with_committed, CommittedGuWitness,
    FixtureSendProvider,
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
    println!("committed_root={:?}", outcome.committed.root());
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
    assert_eq!(outcome.committed.root(), outcome.gu_root);
    assert_eq!(
        outcome.committed.recompute_root().unwrap(),
        outcome.committed.root()
    );

    // Auxiliary test oracle: recompute the provider's sample-form answers from
    // the same committed witness object. The verifier path does not call this
    // comparison.
    let provider = FixtureSendProvider::from_committed(outcome.committed.clone()).unwrap();
    assert!(std::sync::Arc::ptr_eq(
        provider.committed(),
        &outcome.committed
    ));
    let oracle_answers = provider.answers(&outcome.seeds).unwrap();
    assert_eq!(oracle_answers, outcome.provider_answers);
    assert_eq!(
        provider.answers(&outcome.seeds).unwrap(),
        oracle_answers
    );

    // The mask pad is a reserved message value encoded before the seed; the
    // seed only supplies its linear coefficient. A different nonzero pad must
    // change both the committed root and the sample answer.
    let mask_code = outcome.gu_shape.u.mask_code.unwrap();
    let pad_index = (mask_code.ell - outcome.gu_shape.u.answers as u64) as usize;
    let zero_pad = Arc::new(
        CommittedGuWitness::new_with_mask_pad(&outcome.gu_shape, Fp3::ZERO).unwrap(),
    );
    let nonzero_pad = Arc::new(
        CommittedGuWitness::new_with_mask_pad(&outcome.gu_shape, Fp3::from_u64(0x1234))
            .unwrap(),
    );
    assert_eq!(zero_pad.mask_messages()[0][pad_index], Fp3::ZERO);
    assert_eq!(
        nonzero_pad.mask_messages()[0][pad_index],
        Fp3::from_u64(0x1234)
    );
    assert_ne!(zero_pad.root(), nonzero_pad.root());
    assert_eq!(zero_pad.recompute_root().unwrap(), zero_pad.root());
    let zero_provider = FixtureSendProvider::from_committed(Arc::clone(&zero_pad)).unwrap();
    let nonzero_provider =
        FixtureSendProvider::from_committed(Arc::clone(&nonzero_pad)).unwrap();
    let zero_answers = zero_provider.answers(&outcome.seeds).unwrap();
    let nonzero_answers = nonzero_provider.answers(&outcome.seeds).unwrap();
    assert_ne!(zero_answers, nonzero_answers);

    // A syntactically valid wrong answer is registered as a pending sample
    // claim, not rejected or self-certified.
    let committed = Arc::clone(&outcome.committed);
    let wrong = verify_send_session_with_committed(
        &public,
        &trusted,
        &input,
        "gpt2/weights/lm_head_w_i32.bin",
        opening,
        Arc::clone(&outcome.committed),
        move |committed, _, seeds| {
            let mut answers = FixtureSendProvider::from_committed(committed)?.answers(seeds)?;
            answers[0] = answers[0].add(&Fp3::ONE);
            Ok(answers)
        },
    )
    .unwrap();
    assert_eq!(sample_target(&wrong), outcome.provider_answers[0].add(&Fp3::ONE));
    assert!(!wrong.combine.verifier_standard_answer_comparison);
    assert_eq!(wrong.combine.bat_input.targets[wrong.combine.bat_input.count - 1], sample_target(&wrong));

    // Truncated answer list fails closed before the verifier registers a claim.
    let truncated = verify_send_session_with_committed(
        &public,
        &trusted,
        &input,
        "gpt2/weights/lm_head_w_i32.bin",
        opening,
        Arc::clone(&outcome.committed),
        |_, _, _| Ok(Vec::new()),
    );
    assert!(matches!(truncated, Err(Error::ShapeMismatch)));

    let other = Arc::new(CommittedGuWitness::new_with_tag(&outcome.gu_shape, 7).unwrap());
    assert_ne!(other.root(), outcome.committed.root());
    assert!(matches!(
        FixtureSendProvider::from_committed_with_root(other, Some(outcome.committed.root())),
        Err(Error::StatementMismatch)
    ));

    let mut unsupported = outcome.gu_shape.clone();
    let part = unsupported.tree.parts[0];
    unsupported.tree.parts.push(part);
    assert!(matches!(
        CommittedGuWitness::new(&unsupported),
        Err(Error::UnsupportedPrimitive) | Err(Error::InvalidConfig)
    ));
    let _ = committed;
}
