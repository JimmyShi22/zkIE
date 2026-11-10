use std::env;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use zkie_engine::pcs::request::code::{build_code_plan, evaluate_ood_fp3, CodeParams};
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::cs_stage;
use zkie_engine::pcs::request::proximity::{
    ProximityPlan, ProximityStage, ProximityStatement, SingleRows, StageRole,
};
use zkie_ops::proof::lin_session::fp3_from_signed_word;
use zkie_ops::proof::model_commitment::{
    verify_model_anchor, ModelCommitmentPublic, SharedTrace,
};
use zkie_ops::proof::proximity_session::{
    verify_proximity_session, ProximityProver, SampleOracle,
};
use zkie_ops::proof::session_baseline::TrustedModel;

struct FixtureOracle {
    code: CodeParams,
    main: Vec<Fp3>,
    tail: Vec<Fp3>,
}

impl SampleOracle for FixtureOracle {
    fn expected(&self, seed: Fp3) -> zkie_core::Result<Fp3> {
        evaluate_ood_fp3(&self.code, &self.main, &self.tail, seed)
    }
}

fn dot(a: &[Fp3], b: &[Fp3]) -> Fp3 {
    let mut sum = Fp3::ZERO;
    for (x, y) in a.iter().zip(b) {
        sum = sum.add(&x.mul(y));
    }
    sum
}

#[test]
#[ignore = "requires ZKIE_SESSION_FIXTURE with the multi-GiB gpt2-seq16 artifact"]
fn selected_gpt2_cs_base_component() {
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

    let mut source = opening.rows.iter().flat_map(|row| row.values.iter().copied());
    let mut take = |count: usize| -> Vec<Fp3> {
        (0..count)
            .map(|_| fp3_from_signed_word(source.next().unwrap()))
            .collect()
    };
    let main = take(4);
    let main_tail = take(1);
    let mask = take(4);
    let mask_tail = take(1);
    let star_main = take(6);
    let star_tail = take(1);
    let star_mask = take(6);
    let star_mask_tail = take(1);
    let claim = vec![Fp3::ONE, Fp3::ZERO, Fp3::ZERO, Fp3::ZERO];
    let mask_claim = vec![Fp3::ZERO, Fp3::ONE, Fp3::ZERO, Fp3::ZERO];
    let target = dot(&claim, &main).add(&dot(&mask_claim, &mask));

    let rows = SingleRows {
        code: build_code_plan(3, 4, 1).unwrap(),
        mask_code: build_code_plan(3, 4, 1).unwrap(),
        main,
        main_tail,
        masks: vec![mask],
        mask_tails: vec![mask_tail],
        claim,
        mask_claims: vec![mask_claim],
        target,
    };
    let new_code = build_code_plan(3, 6, 1).unwrap();
    let plan = ProximityPlan {
        input_code: rows.code,
        input_mask_code: rows.mask_code,
        stages: vec![
            ProximityStage {
                role: StageRole::CodeSwitch,
                code: new_code,
                mask_code: new_code,
                k: 0,
                h: 0,
                answers: 1,
                samples: 3,
                folds: false,
                choice: zkie_engine::pcs::request::proximity::ChoiceReference::VerifierPlan,
            },
            ProximityStage {
                role: StageRole::Base,
                code: new_code,
                mask_code: new_code,
                k: 0,
                h: 0,
                answers: 1,
                samples: 4,
                folds: false,
                choice: zkie_engine::pcs::request::proximity::ChoiceReference::VerifierPlan,
            },
        ],
    };

    let statement = ProximityStatement::from_rows(&rows).unwrap();
    let bad_rows = rows.clone();
    let star_main_original = star_main.clone();
    let star_tail_original = star_tail.clone();
    let star_mask_original = star_mask.clone();
    let star_mask_tail_original = star_mask_tail.clone();
    let switched = cs_stage::switched_message(&rows, new_code, new_code).unwrap();
    let oracle = FixtureOracle {
        code: new_code,
        main: switched.main,
        tail: switched.main_tail,
    };
    let prover = ProximityProver::new(
        &plan,
        rows,
        star_main,
        star_tail,
        star_mask,
        star_mask_tail,
    )
    .unwrap();
    let messages = prover.messages().unwrap();
    let started = Instant::now();
    let outcome = verify_proximity_session(&plan, &statement, messages, &prover, &oracle).unwrap();
    println!("proximity_session_ms={}", started.elapsed().as_millis());
    println!("old_main_root={:?}", outcome.old_main_root);
    println!("old_mask_root={:?}", outcome.old_mask_root);
    println!("switched_root={:?}", outcome.switched_root);
    println!("star_root={:?}", outcome.star_root);
    println!("cs_positions={:?}", outcome.cs_positions);
    println!("base_positions={:?}", outcome.base_positions);
    println!("registration_count={}", outcome.registration_count);
    println!("theta={:?}", outcome.theta);
    println!("lambda={:?}", outcome.lambda);
    assert!(outcome.stage_handoff);
    assert!(outcome.sample_oracle_match);
    assert!(outcome.base_linear);
    assert!(outcome.symbol_checks);
    assert!(outcome.component_checks_passed);
    assert_eq!(outcome.component_complete().unwrap(), zkie_ops::proof::proximity_session::ProximityComponent::ComponentComplete);
    assert_eq!(outcome.registration_count, 7);

    let mut bad_rows = bad_rows;
    bad_rows.target = bad_rows.target.add(&Fp3::ONE);
    let bad_statement = ProximityStatement::from_rows(&bad_rows).unwrap();
    let bad_prover = ProximityProver::new(
        &plan,
        bad_rows,
        star_main_original,
        star_tail_original,
        star_mask_original,
        star_mask_tail_original,
    )
    .unwrap();
    let bad_messages = bad_prover.messages().unwrap();
    assert!(verify_proximity_session(&plan, &bad_statement, bad_messages, &bad_prover, &oracle).is_err());

    let mut bad_plan = plan.clone();
    bad_plan.stages.truncate(1);
    assert!(verify_proximity_session(&bad_plan, &statement, messages, &prover, &oracle).is_err());

    let mut bad_plan = plan.clone();
    bad_plan.input_code = new_code;
    assert!(verify_proximity_session(&bad_plan, &statement, messages, &prover, &oracle).is_err());

    assert!(verify_proximity_session(&plan, &statement, messages, &prover, &oracle).is_ok());
}
