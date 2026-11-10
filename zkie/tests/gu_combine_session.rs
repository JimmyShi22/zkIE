use std::env;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::relation::GuRelationKind;
use zkie_ops::proof::column_session::ColumnBindingInput;
use zkie_ops::proof::gu_combine_session::{
    combine_declarations, verify_gu_combine_session,
};
use zkie_ops::proof::model_commitment::{
    verify_model_anchor, ModelCommitmentPublic, SharedTrace,
};
use zkie_ops::proof::session_baseline::{
    selected_plan_identifier, SelectedMatmulStep, TrustedModel,
};

#[test]
#[ignore = "requires ZKIE_SESSION_FIXTURE with the multi-GiB gpt2-seq16 artifact"]
fn selected_gpt2_gu_declaration_combination() {
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
    let outcome = verify_gu_combine_session(
        &public,
        &trusted,
        &input,
        "gpt2/weights/lm_head_w_i32.bin",
        opening,
    )
    .unwrap();
    println!("gu_combine_session_ms={}", started.elapsed().as_millis());
    println!("reader_root={:?}", outcome.reader_root);
    println!("gu_root={:?}", outcome.gu_root);
    println!("gu_identity={:?}/{:?}/{:?}", outcome.gu_oracle, outcome.gu_group, outcome.gu_tree_number);
    println!("queries={:?}", outcome.queries);
    println!("sample_count={}", outcome.sample_count);
    println!("record_count={}", outcome.record_count);
    println!("column_count={}", outcome.column_count);
    println!("declarations={}", outcome.declarations.len());
    println!("x={:?}", outcome.x);
    println!(
        "combined={{oracle:{},group:{},count:{},kinds:{:?},target:{:?}}}",
        outcome.combined.oracle,
        outcome.combined.group,
        outcome.combined.count,
        outcome.combined.kinds,
        outcome.combined.target
    );

    assert!(outcome.same_session);
    assert!(outcome.column_root_only);
    assert!(outcome.self_reported_gu_root);
    assert!(outcome.fake_send_answer);
    assert!(!outcome.bat_decides_truth);
    assert!(!outcome.three_c_connected);
    assert_ne!(outcome.gu_root, zkie_core::types::Digest::ZERO);
    assert_eq!(outcome.sample_count, 1);
    assert_eq!(outcome.record_count, 1);
    assert!(outcome.column_count > 0);
    assert_eq!(
        outcome.declarations.len(),
        outcome.sample_count + outcome.record_count + outcome.column_count
    );
    assert_eq!(outcome.combined.count, outcome.declarations.len());
    assert_eq!(outcome.combined.kinds.len(), outcome.declarations.len());
    assert!(outcome.combined.kinds.contains(&GuRelationKind::Sample));
    assert!(outcome.combined.kinds.contains(&GuRelationKind::Record));
    assert!(outcome.combined.kinds.contains(&GuRelationKind::Column));
    assert_eq!(outcome.combined.oracle, outcome.gu_oracle);
    assert_eq!(outcome.combined.group, outcome.gu_group);
    assert_eq!(outcome.combined.main.len(), 16);
    assert_eq!(outcome.combined.mask.len(), 6);
    assert_eq!(outcome.combined.kinds[0], GuRelationKind::Sample);
    assert_eq!(outcome.combined.kinds[1], GuRelationKind::Record);
    assert!(outcome.combined.kinds[2..]
        .iter()
        .all(|kind| *kind == GuRelationKind::Column));
    assert!(!outcome.queries.is_empty());
    assert_ne!(outcome.transcript_after_columns, zkie_core::types::Digest::ZERO);
    assert!(outcome.unsupported.contains(&"multi-oracle SEND"));
    assert!(outcome.unsupported.contains(&"real SEND provider"));
    assert!(outcome.unsupported.contains(&"fresh-mask declaration"));

    let second = verify_gu_combine_session(
        &public,
        &trusted,
        &input,
        "gpt2/weights/lm_head_w_i32.bin",
        opening,
    )
    .unwrap();
    assert_eq!(second.transcript_after_columns, outcome.transcript_after_columns);
    assert_eq!(second.x, outcome.x);
    assert_eq!(second.combined, outcome.combined);

    let mut wrong = outcome.declarations.clone();
    wrong[0].target = wrong[0].target.add(&Fp3::ONE);
    let wrong_combined = combine_declarations(&wrong, outcome.x).unwrap();
    assert_ne!(wrong_combined.target, outcome.combined.target);
}
