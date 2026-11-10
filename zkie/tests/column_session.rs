use std::env;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use zkie_core::types::Digest;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::foldeq::reader_table_identity;
use zkie_engine::pcs::request::relation::{foldeq_weights, GuRelationKind, GuRelationRow};
use zkie_engine::pcs::request::table::verify_columns;
use zkie_ops::proof::column_session::{
    build_authenticated_encoding_table_with_configuration, check_full_columns, check_merge_targets,
    open_columns_for_queries, verify_column_session, ColumnBindingInput,
};
use zkie_ops::proof::model_commitment::{ModelCommitmentPublic, SharedTrace};
use zkie_ops::proof::session_baseline::{
    selected_plan_identifier, verify_selected_step, SelectedMatmulStep, TrustedModel,
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

    let plan_identifier = selected_plan_identifier(&public, &plan, &selection).unwrap();
    let input = ColumnBindingInput::from_plan(
        plan_identifier,
        trusted_request.case_id.as_bytes().to_vec(),
    )
    .unwrap();
    assert_eq!(input.parameters.plan_identifier, plan_identifier);
    assert_eq!(input.case_id, trusted_request.case_id.as_bytes());
    let asset_path = "gpt2/weights/lm_head_w_i32.bin";
    let started = Instant::now();
    let outcome = verify_column_session(
        &public,
        &trusted,
        &input,
        asset_path,
        &trace.assets[192],
    )
    .unwrap();
    println!("column_session_ms={}", started.elapsed().as_millis());
    println!("asset_root={:?}", outcome.asset_root);
    println!("table_root={:?}", outcome.table_root);
    println!("gu_root={:?}", outcome.gu_root);
    println!("queries={:?}", outcome.queries);
    println!("opened_positions={:?}", outcome.opened_positions);
    println!("gamma={:?}", outcome.gamma);
    println!("eq_gamma={:?}", outcome.eq_gamma);
    println!("column_cells={}", outcome.column_cells);
    println!("gu_rows={}", outcome.gu_rows);
    assert!(outcome.column_authentication);
    assert!(outcome.gu_registration);
    assert!(outcome.merge_targets);
    assert!(outcome.full_table_comparison);
    assert!(outcome.fold_component);
    assert!(outcome.fold_next_stage_closed);
    assert_eq!(outcome.queries.len(), 4);
    assert_eq!(outcome.gamma.len(), 2);
    assert_eq!(outcome.eq_gamma, foldeq_weights(&outcome.gamma, 2).unwrap());
    assert_eq!(outcome.column_cells, outcome.queries.len() * 4);
    assert!(outcome.gu_rows >= 4);
    assert_ne!(outcome.asset_root, outcome.table_root);
    assert_ne!(outcome.asset_root, outcome.gu_root);
    assert_ne!(outcome.table_root, outcome.gu_root);

    let table = build_authenticated_encoding_table_with_configuration(
        &public,
        &trusted,
        asset_path,
        &trace.assets[192],
        input.parameters.digest().unwrap(),
    )
    .unwrap();
    assert_eq!(table.reader.root, outcome.table_root);
    assert_eq!(
        outcome.binding_parameters_digest,
        input.parameters.digest().unwrap()
    );
    assert_eq!(
        outcome.binding_table_identity,
        reader_table_identity(&table.reader).unwrap()
    );
    let openings = open_columns_for_queries(&table, &outcome.queries).unwrap();
    let positions = zkie_engine::pcs::request::read::unique_sorted(&outcome.queries);
    let unique: Vec<_> = positions
        .iter()
        .map(|position| {
            openings
                .iter()
                .find(|opening| opening.position == *position)
                .unwrap()
                .clone()
        })
        .collect();
    let flat = verify_columns(&table.reader, &positions, &unique).unwrap();
    let rows = 4usize;
    let by_position: Vec<(u64, Vec<_>)> = positions
        .iter()
        .enumerate()
        .map(|(index, position)| {
            (*position, flat[index * rows..(index + 1) * rows].to_vec())
        })
        .collect();
    let mut logical_flat = Vec::new();
    for query in &outcome.queries {
        let column = by_position
            .iter()
            .find(|(position, _)| position == query)
            .map(|(_, column)| column)
            .unwrap();
        logical_flat.extend_from_slice(column);
    }
    assert!(check_full_columns(&table, &outcome.queries, &logical_flat)
        .unwrap()
        .matches);
    let mut wrong_columns = logical_flat.clone();
    wrong_columns[0] = Fp3::from_u64(999).c[0];
    assert!(!check_full_columns(&table, &outcome.queries, &wrong_columns)
        .unwrap()
        .matches);

    let fake_targets = vec![Fp3::ZERO; outcome.queries.len()];
    assert!(!check_merge_targets(
        &table,
        &outcome.queries,
        &fake_targets,
        &outcome.eq_gamma
    )
    .unwrap());

    let mut path_tamper = unique.clone();
    path_tamper[0].path.push(Digest([9; 32]));
    assert!(verify_columns(&table.reader, &positions, &path_tamper).is_err());

    let mut root_tamper = trace.assets[192].clone();
    root_tamper.root.0[0] ^= 1;
    assert!(build_authenticated_encoding_table_with_configuration(
        &public,
        &trusted,
        asset_path,
        &root_tamper,
        input.parameters.digest().unwrap()
    )
    .is_err());

    assert!(verify_column_session(
        &public,
        &trusted,
        &input,
        asset_path,
        &trace.assets[192]
    )
    .is_ok());

    let case_changed = ColumnBindingInput::from_plan(
        plan_identifier,
        b"gpt2-seq16-other".to_vec(),
    )
    .unwrap();
    let case_outcome = verify_column_session(
        &public,
        &trusted,
        &case_changed,
        asset_path,
        &trace.assets[192],
    )
    .unwrap();
    assert_eq!(case_outcome.table_root, outcome.table_root);
    assert_ne!(
        case_outcome.transcript_after_queries,
        outcome.transcript_after_queries
    );

    let plan_changed = ColumnBindingInput::from_plan(
        Digest([0x77; 32]),
        trusted_request.case_id.as_bytes().to_vec(),
    )
    .unwrap();
    let plan_outcome = verify_column_session(
        &public,
        &trusted,
        &plan_changed,
        asset_path,
        &trace.assets[192],
    )
    .unwrap();
    assert_ne!(plan_outcome.table_root, outcome.table_root);
    assert_ne!(
        plan_outcome.transcript_after_queries,
        outcome.transcript_after_queries
    );
}
