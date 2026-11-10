use std::env;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use zkie_core::types::Digest;
use zkie_engine::pcs::request::fold::FoldOutput;
use zkie_engine::pcs::request::table::verify_columns;
use zkie_ops::proof::column_session::{
    authenticate_columns, build_authenticated_encoding_table, check_fold_component,
    check_full_table, check_merge_targets, demo_eq_gamma, register_gu,
    register_gu_with_columns, verify_column_session,
};
use zkie_ops::proof::model_commitment::{ModelCommitmentPublic, SharedTrace};
use zkie_ops::proof::session_baseline::{
    assemble_binding, verify_selected_step, SelectedMatmulStep, TrustedModel,
};

#[test]
#[ignore = "requires ZKIE_SESSION_FIXTURE with the multi-GiB gpt2-seq16 artifact"]
fn selected_gpt2_column_verification_components() {
    let fixture = env::var("ZKIE_SESSION_FIXTURE").expect("set ZKIE_SESSION_FIXTURE");
    let root = Path::new(&fixture).join("gpt2-seq16");
    let public: ModelCommitmentPublic =
        bincode::deserialize(&fs::read(root.join("public/model-commitment.bin")).unwrap()).unwrap();
    let file = fs::File::open(root.join("proof.bin")).unwrap();
    let trace: SharedTrace = bincode::deserialize_from(BufReader::new(file)).unwrap();
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
    let asset_path = "gpt2/weights/lm_head_w_i32.bin";
    let started = Instant::now();
    let outcome = verify_column_session(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
    )
    .unwrap();
    println!("column_session_ms={}", started.elapsed().as_millis());
    println!("asset_root={:?}", outcome.asset_root);
    println!("table_root={:?}", outcome.table_root);
    println!("gu_root={:?}", outcome.gu_root);
    println!("logical_draws={:?}", outcome.logical_draws);
    println!("opened_positions={:?}", outcome.opened_positions);
    println!("column_cells={}", outcome.column_cells);
    println!("gu_rows={}", outcome.gu_rows);
    assert!(outcome.column_authentication);
    assert!(outcome.gu_registration);
    assert!(outcome.merge_targets);
    assert!(outcome.full_table_comparison);
    assert!(outcome.fold_component);
    assert!(outcome.fold_next_stage_closed);
    assert_eq!(outcome.logical_draws.len(), 4);
    assert_eq!(outcome.gu_rows, 4);
    assert_ne!(outcome.asset_root, outcome.table_root);
    assert_ne!(outcome.asset_root, outcome.gu_root);
    assert_ne!(outcome.table_root, outcome.gu_root);

    let table = build_authenticated_encoding_table(
        &public,
        &trusted,
        asset_path,
        &trace.assets[192],
    )
    .unwrap();
    let authentication = authenticate_columns(&table, &outcome.logical_draws).unwrap();
    let eq_gamma = demo_eq_gamma(authentication.rows).unwrap();
    let good = register_gu(&table, &authentication, &eq_gamma).unwrap();
    assert!(good
        .rows
        .iter()
        .all(|row| good.membership.check_row(row).is_ok()));
    assert!(check_merge_targets(&table, &authentication, &good, &eq_gamma).unwrap());
    assert!(check_full_table(&table, &authentication).unwrap().matches);

    let mut wrong_columns = authentication.columns_by_position();
    wrong_columns[0].1[0] = wrong_columns[0].1[0].add(&zkie_engine::pcs::request::field::Fp3::ONE);
    let wrong = register_gu_with_columns(&table, &authentication, &eq_gamma, &wrong_columns)
        .expect("membership registration does not evaluate the target");
    assert!(wrong
        .rows
        .iter()
        .all(|row| wrong.membership.check_row(row).is_ok()));
    assert!(!check_merge_targets(&table, &authentication, &wrong, &eq_gamma).unwrap());

    let mut wrong_opened = authentication.clone();
    wrong_opened.columns[0] = zkie_engine::pcs::request::field::Fp3::from_u64(999).c[0];
    assert!(!check_full_table(&table, &wrong_opened).unwrap().matches);

    let mut path_tamper = authentication.clone();
    path_tamper.openings[0].path.push(Digest([9; 32]));
    assert!(verify_columns(
        &table.reader,
        &path_tamper.positions,
        &path_tamper.openings
    )
    .is_err());

    let mut root_tamper = trace.assets[192].clone();
    root_tamper.root.0[0] ^= 1;
    assert!(build_authenticated_encoding_table(&public, &trusted, asset_path, &root_tamper).is_err());

    let fold = check_fold_component(
        table.shape,
        zkie_engine::pcs::request::field::Fp3::from_u64(7),
        zkie_engine::pcs::request::field::Fp3::from_u64(11),
        zkie_engine::pcs::request::field::Fp3::from_u64(13),
    )
    .unwrap();
    fold.input.check(&fold.shape, &fold.layout).unwrap();
    assert!(fold.next_stage_closed);
    let closed: Result<(), zkie_core::error::Error> = fold.output.next_stage();
    assert!(closed.is_err());
    let _: &FoldOutput = &fold.output;

    assert!(verify_column_session(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192]
    )
    .is_ok());
}
