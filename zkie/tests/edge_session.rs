use std::env;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use zkie_engine::pcs::request::foldeq::{EdgeLayout, FoldeqEdgeBinding};
use zkie_ops::proof::column_session::{verify_column_session, ColumnBindingInput};
use zkie_ops::proof::edge_session::{
    authenticate_consumer_input, check_edge_against_plan, compare_edge_values, edge_from_plan,
    verify_edge_session, verify_profile_norm_step,
};
use zkie_ops::proof::model_commitment::{ModelCommitmentPublic, SharedTrace};
use zkie_ops::proof::session_baseline::{
    selected_plan_identifier, verify_selected_step, SelectedMatmulStep, TrustedModel,
};

#[test]
#[ignore = "requires ZKIE_SESSION_FIXTURE with the multi-GiB gpt2-seq16 artifact"]
fn selected_gpt2_two_step_edge() {
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
        .expect("2c selected MatMul control");
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
    let asset_path = "gpt2/weights/lm_head_w_i32.bin";

    let started = Instant::now();
    let outcome = verify_edge_session(
        &public,
        &trace,
        &plan,
        &trusted,
        &input,
        asset_path,
        &trace.assets[192],
    )
    .unwrap();
    println!("edge_session_ms={}", started.elapsed().as_millis());
    println!("producer_root={:?}", outcome.producer_root);
    println!("consumer_root={:?}", outcome.consumer_root);
    println!("producer_len={}", outcome.producer_output.len());
    println!("consumer_len={}", outcome.consumer_input.len());
    println!("producer_first={:?}", &outcome.producer_output[..4]);
    println!("consumer_first={:?}", &outcome.consumer_input[..4]);
    println!("edge_bound={}", outcome.column_session.edge_bound);
    println!("table_root={:?}", outcome.column_session.table_root);
    println!("queries={:?}", outcome.column_session.queries);
    println!("gu_rows={}", outcome.column_session.gu_rows);
    assert!(outcome.relation_verified);
    assert!(outcome.input_authenticated);
    assert!(outcome.values_equal);
    assert!(outcome.column_session.edge_bound);
    assert!(outcome.column_session.column_authentication);
    assert!(outcome.column_session.full_table_comparison);
    assert_eq!(outcome.producer_output, outcome.consumer_input);
    assert_eq!(outcome.producer_output.len(), 16 * 1024);
    assert_eq!(outcome.consumer_input.len(), 16 * 1024);
    assert!(matches!(
        outcome.edge,
        FoldeqEdgeBinding::Prev(_)
    ));

    let no_edge = verify_column_session(
        &public,
        &trusted,
        &input,
        asset_path,
        &trace.assets[192],
    )
    .unwrap();
    assert_eq!(no_edge.table_root, outcome.column_session.table_root);
    assert_ne!(
        no_edge.transcript_after_queries,
        outcome.column_session.transcript_after_queries
    );

    let actual_edge = edge_from_plan(&public, &plan, 4443, 4444, 0).unwrap();
    check_edge_against_plan(&public, &plan, &actual_edge).unwrap();

    let mut wrong_producer = actual_edge.clone();
    if let FoldeqEdgeBinding::Prev(edge) = &mut wrong_producer {
        edge.producer_step = 4442;
    }
    assert!(check_edge_against_plan(&public, &plan, &wrong_producer).is_err());

    let mut wrong_port = actual_edge.clone();
    if let FoldeqEdgeBinding::Prev(edge) = &mut wrong_port {
        edge.input_port = 1;
    }
    assert!(check_edge_against_plan(&public, &plan, &wrong_port).is_err());

    let mut wrong_layout = actual_edge.clone();
    if let FoldeqEdgeBinding::Prev(edge) = &mut wrong_layout {
        edge.layout = EdgeLayout::Broadcast { rows: 16 };
    }
    assert!(check_edge_against_plan(&public, &plan, &wrong_layout).is_err());

    let producer = verify_profile_norm_step(&public, &trace, &plan, 4443).unwrap();
    let consumer = authenticate_consumer_input(&plan, &trace, 4444, 0).unwrap();
    let mut changed_producer = producer.output.clone();
    changed_producer[0] ^= 1;
    assert!(compare_edge_values(&changed_producer, &consumer.values).is_err());

    let mut output_tamper = trace.clone();
    output_tamper.steps[4443].message.proof.outputs[0][0].values[0] ^= 1;
    assert!(verify_profile_norm_step(&public, &output_tamper, &plan, 4443).is_err());

    let mut input_tamper = trace.clone();
    input_tamper.steps[4444].message.proof.inputs[0][0].values[0] ^= 1;
    assert!(authenticate_consumer_input(&plan, &input_tamper, 4444, 0).is_err());

    let mut root_tamper = trace.clone();
    let output_root_position = root_tamper.steps[4443]
        .message
        .roots
        .iter()
        .position(|(root, _, _)| *root == producer.output_root)
        .expect("producer output root is registered");
    root_tamper.steps[4443].message.roots[output_root_position].0.0[0] ^= 1;
    assert!(verify_profile_norm_step(&public, &root_tamper, &plan, 4443).is_err());

    let mut missing = actual_edge.clone();
    if let FoldeqEdgeBinding::Prev(edge) = &mut missing {
        edge.layout = EdgeLayout::Chain(Vec::new());
    }
    assert!(missing.check().is_err());

    assert!(verify_edge_session(
        &public,
        &trace,
        &plan,
        &trusted,
        &input,
        asset_path,
        &trace.assets[192]
    )
    .is_ok());
}
