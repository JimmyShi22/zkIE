use std::env;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use zkie_core::error::Error;
use zkie_ops::proof::column_session::ColumnBindingInput;
use zkie_ops::proof::group_row_session::{verify_group_row_session, GroupRowComparison};
use zkie_ops::proof::model_commitment::{
    verify_model_anchor, ModelCommitmentPublic, SharedTrace,
};
use zkie_ops::proof::session_baseline::{
    selected_plan_identifier, SelectedMatmulStep, TrustedModel,
};

#[test]
#[ignore = "requires ZKIE_SESSION_FIXTURE with the multi-GiB gpt2-seq16 artifact"]
fn selected_gpt2_group_row_component() {
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
    let outcome = verify_group_row_session(
        &public,
        &trusted,
        &input,
        "gpt2/weights/lm_head_w_i32.bin",
        opening,
    )
    .unwrap();
    println!("group_row_session_ms={}", started.elapsed().as_millis());
    println!("asset_root={:?}", outcome.asset_root);
    println!("table_root={:?}", outcome.table_root);
    println!("fresh_root={:?}", outcome.fresh_root);
    println!("comparison={:?}", outcome.comparison);
    println!("same_source={}", outcome.same_source);
    println!("eps={:?}", outcome.eps);
    println!("gammas={:?}", outcome.gammas);
    println!("round_messages={:?}", outcome.round_messages);
    println!("target={:?}", outcome.target);
    println!("fold_target={:?}", outcome.fold_output.target);
    assert_eq!(
        outcome.comparison,
        GroupRowComparison {
            asset_opening: true,
            main_rows: true,
            table_commitment: true,
            fresh_local_consistency: true,
        }
    );
    assert!(outcome.same_source);
    assert!(outcome.fold_next_stage_closed);
    assert!(outcome.handoff_rejected);
    assert!(!outcome.main_tail_folded);
    assert!(!outcome.old_mask_tail_folded);
    assert!(!outcome.fresh_tail_folded);
    assert_ne!(outcome.asset_root, outcome.table_root);
    assert_ne!(outcome.table_root, outcome.fresh_root);
    assert_eq!(outcome.main_messages.len(), 16);
    assert_eq!(outcome.old_mask_messages.len(), 6);
    assert_eq!(outcome.fresh_mask.len(), 6);
    assert_eq!(outcome.coefficients.len(), 16);
    assert_eq!(outcome.mask_coefficients.len(), 6);
    assert_ne!(outcome.main_messages[0], outcome.old_mask_messages[0]);
    assert_ne!(outcome.old_mask_messages[0], outcome.fresh_mask[0]);
    assert_eq!(&outcome.order[..3], &["fresh_commit", "start_sum", "epsilon"][..]);
    assert_eq!(outcome.gammas.len(), 2);
    assert_eq!(outcome.round_messages.len(), 2);
    assert_eq!(outcome.fold_output.gamma, outcome.gammas);
    assert_eq!(outcome.fold_output.eps, outcome.eps);
    assert!(outcome.fold_output.next_stage().is_err());

    let mut bad_opening = opening.clone();
    bad_opening.rows[0].values[0] = bad_opening.rows[0].values[0].wrapping_add(1);
    let rejected = verify_group_row_session(
        &public,
        &trusted,
        &input,
        "gpt2/weights/lm_head_w_i32.bin",
        &bad_opening,
    );
    assert!(matches!(
        rejected,
        Err(Error::InvalidProof) | Err(Error::StatementMismatch) | Err(Error::AssetMismatch)
    ));
}
