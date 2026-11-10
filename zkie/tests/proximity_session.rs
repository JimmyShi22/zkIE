use std::env;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use zkie_core::error::Error;
use zkie_engine::pcs::request::code::{build_code_plan, evaluate_ood_fp3, CodeParams};
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::cs_stage;
use zkie_engine::pcs::request::proximity::{
    ChoiceReference, ProximityPlan, ProximityStage, ProximityStatement, SingleRows, StageRole,
};
use zkie_ops::proof::lin_session::fp3_from_signed_word;
use zkie_ops::proof::model_commitment::{
    verify_model_anchor, ModelCommitmentPublic, SharedTrace,
};
use zkie_ops::proof::proximity_session::{
    verify_proximity_session, ProximityProver, SampleOracle,
};
use zkie_ops::proof::column_session::{verify_column_session, ColumnBindingInput};
use zkie_ops::proof::fold_handoff::FoldHandoff;
use zkie_ops::proof::session_baseline::{
    selected_plan_identifier, SelectedMatmulStep, TrustedModel,
};

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


fn handoff_plan(handoff: &FoldHandoff) -> ProximityPlan {
    let switch_code = build_code_plan(
        handoff.code.log_n,
        handoff.code.ell + handoff.code.r,
        handoff.code.r,
    )
    .unwrap();
    let switch_mask = build_code_plan(
        handoff.mask_code.log_n,
        handoff.mask_code.ell + handoff.mask_code.r,
        handoff.mask_code.r,
    )
    .unwrap();
    ProximityPlan {
        input_code: handoff.code,
        input_mask_code: handoff.mask_code,
        stages: vec![
            ProximityStage {
                role: StageRole::CodeSwitch,
                code: switch_code,
                mask_code: switch_mask,
                k: 0,
                h: 0,
                answers: 1,
                samples: 3,
                folds: false,
                choice: ChoiceReference::VerifierPlan,
            },
            ProximityStage {
                role: StageRole::Base,
                code: switch_code,
                mask_code: switch_mask,
                k: 0,
                h: 0,
                answers: 1,
                samples: 4,
                folds: false,
                choice: ChoiceReference::VerifierPlan,
            },
        ],
    }
}

/// Fixture witness for the candidate claim: the plaintext rows are not part of
/// the FOLD output. One coordinate is set so the fixture satisfies the mapped
/// claim target; this is not the real FOLD witness.
fn handoff_rows(handoff: &FoldHandoff) -> SingleRows {
    let mut main = vec![Fp3::ZERO; handoff.code.ell as usize];
    let mut mask = vec![Fp3::ZERO; handoff.mask_code.ell as usize];
    let mut assigned = false;
    for (index, coefficient) in handoff.main_claim.iter().enumerate() {
        if *coefficient != Fp3::ZERO {
            main[index] = handoff.target.mul(&coefficient.inverse().unwrap());
            assigned = true;
            break;
        }
    }
    if !assigned {
        for (index, coefficient) in handoff.mask_claim.iter().enumerate() {
            if *coefficient != Fp3::ZERO {
                mask[index] = handoff.target.mul(&coefficient.inverse().unwrap());
                assigned = true;
                break;
            }
        }
    }
    assert!(assigned || handoff.target == Fp3::ZERO);
    handoff
        .single_rows(
            main,
            vec![Fp3::ZERO; handoff.code.r as usize],
            mask,
            vec![Fp3::ZERO; handoff.mask_code.r as usize],
        )
        .unwrap()
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

#[test]
#[ignore = "requires ZKIE_SESSION_FIXTURE with the multi-GiB gpt2-seq16 artifact"]
fn selected_gpt2_fold_projection_component() {
    // Field projection fixture: the fresh term is not authenticated, so no
    // valid FOLD-to-CS/BASE handoff is produced. The component result below
    // describes the algebraic behaviour of the projected data only.
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
    let public_opening = &trace.assets[192];
    public.verify_opening(public_opening).unwrap();

    let model_plan = zkie_models_gpt2::plan::trace_plan(16).unwrap();
    let selection = SelectedMatmulStep::select(&model_plan, 4444).unwrap();
    let plan_identifier = selected_plan_identifier(&public, &model_plan, &selection).unwrap();
    let input =
        ColumnBindingInput::from_plan(plan_identifier, request.case_id.as_bytes().to_vec()).unwrap();
    let column = verify_column_session(
        &public,
        &trusted,
        &input,
        "gpt2/weights/lm_head_w_i32.bin",
        public_opening,
    )
    .unwrap();
    let handoff = FoldHandoff::from_output(&column.fold_output, &column.fold_anchor).unwrap();
    assert_eq!(handoff.main_claim, column.fold_output.coefficients);
    assert_eq!(handoff.mask_claim, column.fold_output.mask_coefficients);
    assert_eq!(handoff.fresh_claim, column.fold_output.fresh_coefficients);
    assert_eq!(handoff.target, column.fold_output.target);
    assert_eq!(handoff.group, column.fold_output.group);
    assert_eq!(handoff.members, column.fold_output.members);
    assert_eq!(handoff.kept_rows, 0);
    assert!(column.fold_output.kept.is_empty());
    assert!(handoff.handoff_statement().is_err());

    let statement = handoff.projection_statement().unwrap();
    let plan = handoff_plan(&handoff);
    let switch_code = plan.stages[0].code;
    let switch_mask = plan.stages[0].mask_code;
    let rows = handoff_rows(&handoff);
    let switched = cs_stage::switched_message(&rows, switch_code, switch_mask).unwrap();
    let oracle = FixtureOracle {
        code: switch_code,
        main: switched.main,
        tail: switched.main_tail,
    };
    let star_main = vec![Fp3::from_u64(11); switch_code.ell as usize];
    let star_tail = vec![Fp3::from_u64(12); switch_code.r as usize];
    let star_mask = vec![Fp3::from_u64(13); switch_mask.ell as usize];
    let star_mask_tail = vec![Fp3::from_u64(14); switch_mask.r as usize];
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
    println!(
        "handoff_session_ms={} registration_count={} cs_positions={:?} base_positions={:?}",
        started.elapsed().as_millis(),
        outcome.registration_count,
        outcome.cs_positions,
        outcome.base_positions
    );
    assert_eq!(
        outcome.component_complete().unwrap(),
        zkie_ops::proof::proximity_session::ProximityComponent::ComponentComplete
    );

    let mut bad_output = column.fold_output.clone();
    bad_output.target = bad_output.target.add(&Fp3::ONE);
    let bad_handoff = FoldHandoff::from_output(&bad_output, &column.fold_anchor).unwrap();
    let bad_statement = bad_handoff.projection_statement().unwrap();
    assert!(matches!(
        verify_proximity_session(&plan, &bad_statement, messages, &prover, &oracle),
        Err(Error::InvalidProof)
    ));

    let mut bad_output = column.fold_output.clone();
    bad_output.coefficients[0] = bad_output.coefficients[0].add(&Fp3::ONE);
    let bad_handoff = FoldHandoff::from_output(&bad_output, &column.fold_anchor).unwrap();
    let bad_statement = bad_handoff.projection_statement().unwrap();
    assert!(matches!(
        verify_proximity_session(&plan, &bad_statement, messages, &prover, &oracle),
        Err(Error::InvalidProof)
    ));

    let mut bad_output = column.fold_output.clone();
    bad_output.mask_coefficients[0] = bad_output.mask_coefficients[0].add(&Fp3::ONE);
    let bad_handoff = FoldHandoff::from_output(&bad_output, &column.fold_anchor).unwrap();
    let bad_statement = bad_handoff.projection_statement().unwrap();
    assert!(matches!(
        verify_proximity_session(&plan, &bad_statement, messages, &prover, &oracle),
        Err(Error::InvalidProof)
    ));

    let mut bad_output = column.fold_output.clone();
    bad_output.group += 1;
    assert!(matches!(
        FoldHandoff::from_output(&bad_output, &column.fold_anchor),
        Err(Error::StatementMismatch)
    ));
    let mut bad_output = column.fold_output.clone();
    bad_output.fresh_root = zkie_core::types::Digest::ZERO;
    assert!(matches!(
        FoldHandoff::from_output(&bad_output, &column.fold_anchor),
        Err(Error::StatementMismatch)
    ));
    let mut bad_output = column.fold_output.clone();
    bad_output.fresh_coefficients[0] = bad_output.fresh_coefficients[0].add(&Fp3::ONE);
    assert!(matches!(
        FoldHandoff::from_output(&bad_output, &column.fold_anchor),
        Err(Error::StatementMismatch)
    ));
    let mut bad_output = column.fold_output.clone();
    bad_output.coefficients.pop();
    assert!(matches!(
        FoldHandoff::from_output(&bad_output, &column.fold_anchor),
        Err(Error::ShapeMismatch)
    ));
    let mut bad_output = column.fold_output.clone();
    bad_output.kept_rows = 1;
    assert!(matches!(
        FoldHandoff::from_output(&bad_output, &column.fold_anchor),
        Err(Error::StatementMismatch)
    ));
}
