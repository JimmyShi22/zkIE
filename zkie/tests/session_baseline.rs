use std::env;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use zkie_ops::proof::model_commitment::{ModelCommitmentPublic, SharedTrace};
use zkie_ops::proof::session_baseline::{
    verify_selected_step, SelectedMatmulStep, TrustedModel,
};

#[test]
#[ignore = "requires ZKIE_SESSION_FIXTURE with the multi-GiB gpt2-seq16 artifact"]
fn selected_gpt2_matmul_step_session() {
    let fixture = env::var("ZKIE_SESSION_FIXTURE").expect("set ZKIE_SESSION_FIXTURE");
    let root = Path::new(&fixture).join("gpt2-seq16");
    let mut public: ModelCommitmentPublic =
        bincode::deserialize(&fs::read(root.join("public/model-commitment.bin")).unwrap()).unwrap();
    let file = fs::File::open(root.join("proof.bin")).unwrap();
    let mut trace: SharedTrace =
        bincode::deserialize_from(BufReader::new(file)).unwrap();
    let plan = zkie_models_gpt2::plan::trace_plan(16).unwrap();
    let selection = SelectedMatmulStep::select(&plan, 4444).unwrap();
    assert_eq!(selection.config, vec![16, 65536, 1024, 0, 0]);
    assert_eq!(selection.inputs.len(), 2);

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

    let started = Instant::now();
    let outcome = verify_selected_step(&public, &trace, &plan, &selection, &trusted).unwrap();
    println!("positive_elapsed_ms={}", started.elapsed().as_millis());
    assert_eq!(outcome.step_index, 4444);
    assert_eq!(outcome.assets.len(), 1);
    assert_eq!(outcome.assets[0].port, 1);
    assert_eq!(outcome.assets[0].pool, 192);
    assert_eq!(outcome.assets[0].asset_id, 192);

    let original_config = trace.steps[4444].message.config.clone();
    trace.steps[4444].message.config[0] += 1;
    assert!(verify_selected_step(&public, &trace, &plan, &selection, &trusted).is_err());
    trace.steps[4444].message.config = original_config;

    let original_source = trace.steps[4444].sources[0];
    trace.steps[4444].sources[0] = None;
    assert!(verify_selected_step(&public, &trace, &plan, &selection, &trusted).is_err());
    trace.steps[4444].sources[0] = original_source;

    let first = trace.steps[4444].message.proof.inputs[0][0].clone();
    let second = trace.steps[4444].message.proof.inputs[0][1].clone();
    trace.steps[4444].message.proof.inputs[0][0] = second.clone();
    trace.steps[4444].message.proof.inputs[0][1] = first.clone();
    assert!(verify_selected_step(&public, &trace, &plan, &selection, &trusted).is_err());
    trace.steps[4444].message.proof.inputs[0][0] = first;
    trace.steps[4444].message.proof.inputs[0][1] = second;

    let original_root = trace.steps[4444].message.roots[0];
    {
        let root = &mut trace.steps[4444].message.roots[0].0;
        root.0[0] ^= 1;
    }
    assert!(verify_selected_step(&public, &trace, &plan, &selection, &trusted).is_err());
    trace.steps[4444].message.roots[0] = original_root;

    let original_meta_root = public.assets[192].root;
    public.assets[192].root.0[0] ^= 1;
    assert!(verify_selected_step(&public, &trace, &plan, &selection, &trusted).is_err());
    public.assets[192].root = original_meta_root;

    let original_value = trace.assets[192].rows[0].values[0];
    trace.assets[192].rows[0].values[0] ^= 1;
    assert!(verify_selected_step(&public, &trace, &plan, &selection, &trusted).is_err());
    trace.assets[192].rows[0].values[0] = original_value;

    let mut forged = selection.clone();
    forged.config[2] = 64;
    assert!(verify_selected_step(&public, &trace, &plan, &forged, &trusted).is_err());
    assert!(SelectedMatmulStep::select(&plan, 12).is_err());
    assert!(SelectedMatmulStep::select(&plan, 4443).is_err());
}
