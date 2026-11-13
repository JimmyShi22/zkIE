use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use zkie_core::types::Digest;
use zkie_ops::argmax::ArgmaxOp;
use zkie_ops::index_gen::EmbeddingIndexOp;
use zkie_ops::op::OpCode;
use zkie_ops::proof::model_commitment::{
    commit_assets, AssetStore, ModelAssetSpec, ModelCommitmentPublic, SharedTrace,
    SharedTraceStep,
};
use zkie_ops::proof::plan::{config, PlanBuilder, PlanConstant, PlanLayout, PlanSource, TracePlan};
use zkie_ops::proof::relations::{
    proof_output_values, prove_op_relation_shared, prove_op_relation_with_inputs,
    RelationReceiptMessage,
};
use zkie_ops::registry::PureForward;
use zkie_ops::table_lookup::TableLookupOp;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "zkie-plan-check-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

fn write_i32(path: &PathBuf, values: &[i32]) {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    fs::write(path, bytes).unwrap();
}

fn make_step(
    op: &dyn PureForward,
    inputs: &[Vec<i64>],
    config: &[i64],
    external: &[usize],
    external_map: Vec<Option<usize>>,
    sources: Vec<Option<(usize, usize)>>,
) -> (SharedTraceStep, Vec<Vec<i64>>) {
    let (public, proof) = prove_op_relation_shared(op, inputs, config, external).unwrap();
    let mut roots = Vec::new();
    for id in public.registry.ids() {
        let root = public.registry.get(id).unwrap();
        roots.push((root.root, root.start, root.count));
    }
    let outputs = proof_output_values(&proof).unwrap();
    (
        SharedTraceStep {
            message: RelationReceiptMessage {
                opcode: op.opcode() as u16,
                config: config.to_vec(),
                roots,
                external_inputs: external.to_vec(),
                proof,
            },
            external: external_map,
            sources,
        },
        outputs,
    )
}

struct Fixture {
    public: ModelCommitmentPublic,
    trace: SharedTrace,
    tokens: Vec<u32>,
    plan: TracePlan,
}

fn table_lookup_plan(final_output: bool, indices: PlanSource) -> TracePlan {
    let mut builder = PlanBuilder::new();
    let embed = builder.step(
        OpCode::EmbeddingIndex,
        config(vec![1, 4]),
        vec![PlanSource::Constant(PlanConstant::Tokens)],
        1,
    );
    let _ = embed;
    let lookup = builder.step(
        OpCode::TableLookup,
        config(Vec::new()),
        vec![PlanSource::asset("table_i32.bin"), indices],
        1,
    );
    if final_output {
        builder.final_output(lookup, 0);
    } else {
        let argmax = builder.step(
            OpCode::Argmax,
            config(Vec::new()),
            vec![PlanSource::prev(lookup, 0)],
            1,
        );
        builder.argmax_step(argmax);
    }
    builder.finish()
}

fn one_asset_fixture(tokens: &[u32], with_argmax: bool) -> Fixture {
    let root = temp_dir();
    write_i32(&root.join("table_i32.bin"), &[10, 20, 30, 40]);
    let specs = vec![ModelAssetSpec {
        asset_id: 0,
        path: "table_i32.bin".into(),
        element_count: 4,
        row_width: 4,
        encoding: "i32-le".into(),
    }];
    let commitment =
        commit_assets(&root, Digest([1; 32]), Digest([2; 32]), &specs).unwrap();
    let public = ModelCommitmentPublic::from_commitment(&commitment);
    let opening = AssetStore::open(&commitment).unwrap().open_all().unwrap();
    let token_values: Vec<i64> = tokens.iter().map(|value| *value as i64).collect();
    let (embed_step, embed_out) = make_step(
        &EmbeddingIndexOp {
            hidden: 1,
            vocab: 4,
        },
        &[token_values],
        &[1, 4],
        &[],
        vec![None],
        vec![None],
    );
    let table = vec![10, 20, 30, 40];
    let (lookup_step, lookup_out) = make_step(
        &TableLookupOp,
        &[table, embed_out[0].clone()],
        &[],
        &[0],
        vec![Some(0), None],
        vec![None, Some((0, 0))],
    );
    let mut steps = vec![embed_step, lookup_step];
    let mut argmax = Vec::new();
    if with_argmax {
        let (argmax_step, argmax_out) = make_step(
            &ArgmaxOp,
            &[lookup_out[0].clone()],
            &[],
            &[],
            vec![None],
            vec![Some((1, 0))],
        );
        argmax.push(argmax_out[0][0] as u32);
        steps.push(argmax_step);
    }
    let trace = SharedTrace {
        tokens: tokens.to_vec(),
        assets: opening,
        steps,
        argmax,
        output: if with_argmax {
            Vec::new()
        } else {
            lookup_out[0].clone()
        },
    };
    let plan = table_lookup_plan(!with_argmax, PlanSource::prev(0, 0));
    Fixture {
        public,
        trace,
        tokens: tokens.to_vec(),
        plan,
    }
}

fn rejection_code(fixture: &Fixture) -> String {
    let bytes = bincode::serialize(&fixture.trace).unwrap();
    let trace: SharedTrace = bincode::deserialize(&bytes).unwrap();
    let result = if fixture.plan.final_output.is_some() {
        zkie_ops::proof::model_commitment::verify_shared_trace_output(
            &fixture.public,
            &trace,
            &fixture.tokens,
            &fixture.plan,
        )
        .map(|_| ())
    } else {
        zkie_ops::proof::model_commitment::verify_shared_trace(
            &fixture.public,
            &trace,
            &fixture.tokens,
            &fixture.plan,
        )
        .map(|_| ())
    };
    result.err().expect("expected rejection").code().to_owned()
}

fn assert_rejected(fixture: &Fixture) {
    let _ = rejection_code(fixture);
}

#[test]
fn positive_and_missing_extra_reorder() {
    let fixture = one_asset_fixture(&[2], true);
    let argmax = zkie_ops::proof::model_commitment::verify_shared_trace(
        &fixture.public,
        &fixture.trace,
        &fixture.tokens,
        &fixture.plan,
    )
    .unwrap();
    assert_eq!(argmax, vec![0]);

    let mut negative = Fixture {
        public: fixture.public.clone(),
        trace: fixture.trace.clone(),
        tokens: fixture.tokens.clone(),
        plan: fixture.plan.clone(),
    };

    negative.trace.steps.pop();
    assert_rejected(&negative);

    negative.trace.steps.push(negative.trace.steps[1].clone());
    assert_rejected(&negative);

    negative.trace.steps.swap(0, 1);
    assert_rejected(&negative);

    negative.trace.steps.clear();
    assert_rejected(&negative);
}

#[test]
fn negative_sources_and_argmax() {
    let fixture = one_asset_fixture(&[2], true);
    let mut negative = Fixture {
        public: fixture.public.clone(),
        trace: fixture.trace.clone(),
        tokens: fixture.tokens.clone(),
        plan: fixture.plan.clone(),
    };

    negative.trace.steps[1].sources[1] = Some((0, 1));
    assert_rejected(&negative);

    negative.trace = fixture.trace.clone();
    negative.trace.steps[1].sources[1] = Some((1, 0));
    assert_rejected(&negative);

    negative.trace = fixture.trace.clone();
    negative.trace.steps[1].sources[1] = None;
    assert_rejected(&negative);

    negative.trace = fixture.trace.clone();
    negative.trace.argmax[0] = 1;
    assert_rejected(&negative);

    negative.trace = fixture.trace.clone();
    negative.trace.output = vec![1];
    assert_rejected(&negative);

    negative.trace = fixture.trace.clone();
    negative.trace.assets[0].rows[0].values[0] ^= 1;
    assert_rejected(&negative);

    negative.trace = fixture.trace.clone();
    negative.trace.steps[0].message.config[0] = 2;
    assert_rejected(&negative);
}

#[test]
fn negative_external_and_missing_ports() {
    let fixture = one_asset_fixture(&[2], true);
    let mut negative = Fixture {
        public: fixture.public.clone(),
        trace: fixture.trace.clone(),
        tokens: fixture.tokens.clone(),
        plan: fixture.plan.clone(),
    };

    negative.trace.steps[1].external.push(None);
    assert_eq!(rejection_code(&negative), "shape_mismatch");

    negative.trace = fixture.trace.clone();
    negative.trace.steps[1].message.proof.outputs.clear();
    negative.trace.steps[1].message.proof.lengths.outputs.clear();
    assert_eq!(rejection_code(&negative), "shape_mismatch");
}

fn with_step_zero(
    fixture: &Fixture,
    proof: zkie_ops::proof::relations::OpRelationProof,
    public: &zkie_ops::proof::relations::OpRelationPublic,
) -> Fixture {
    let mut roots = Vec::new();
    for id in public.registry.ids() {
        let root = public.registry.get(id).unwrap();
        roots.push((root.root, root.start, root.count));
    }
    let mut changed = Fixture {
        public: fixture.public.clone(),
        trace: fixture.trace.clone(),
        tokens: fixture.tokens.clone(),
        plan: fixture.plan.clone(),
    };
    changed.trace.steps[0] = SharedTraceStep {
        message: RelationReceiptMessage {
            opcode: OpCode::EmbeddingIndex as u16,
            config: vec![1, 4],
            roots,
            external_inputs: Vec::new(),
            proof,
        },
        external: vec![None],
        sources: vec![None],
    };
    changed
}

#[test]
fn old_form_input_openings_are_rejected() {
    let fixture = one_asset_fixture(&[2], true);
    let op = EmbeddingIndexOp { hidden: 1, vocab: 4 };
    let token_values: Vec<i64> = fixture.tokens.iter().map(|value| *value as i64).collect();
    let (public, proof) =
        prove_op_relation_with_inputs(&op, &[token_values], &[1, 4], &[]).unwrap();
    let negative = with_step_zero(&fixture, proof, &public);
    assert_eq!(rejection_code(&negative), "unsupported_version");
}

#[test]
fn materialized_input_mismatch_is_rejected() {
    let fixture = one_asset_fixture(&[2], true);
    let op = EmbeddingIndexOp { hidden: 1, vocab: 4 };
    // The plan materializes token 2; the step was proved for token 3.
    let (public, proof) = prove_op_relation_shared(&op, &[vec![3]], &[1, 4], &[]).unwrap();
    let negative = with_step_zero(&fixture, proof, &public);
    assert_eq!(rejection_code(&negative), "invalid_proof");
}

#[test]
fn step_output_opening_tamper_is_rejected() {
    let fixture = one_asset_fixture(&[2], true);
    let mut negative = Fixture {
        public: fixture.public.clone(),
        trace: fixture.trace.clone(),
        tokens: fixture.tokens.clone(),
        plan: fixture.plan.clone(),
    };
    negative.trace.steps[1].message.proof.outputs[0][0].values[0] ^= 1;
    assert_eq!(rejection_code(&negative), "invalid_proof");
}

#[test]
fn negative_cross_step_edge_with_equal_values() {
    let root = temp_dir();
    write_i32(&root.join("table_i32.bin"), &[0, 1, 2, 3]);
    let specs = vec![ModelAssetSpec {
        asset_id: 0,
        path: "table_i32.bin".into(),
        element_count: 4,
        row_width: 4,
        encoding: "i32-le".into(),
    }];
    let commitment =
        commit_assets(&root, Digest([7; 32]), Digest([8; 32]), &specs).unwrap();
    let public = ModelCommitmentPublic::from_commitment(&commitment);
    let opening = AssetStore::open(&commitment).unwrap().open_all().unwrap();
    let tokens = vec![2u32];
    let token_values: Vec<i64> = tokens.iter().map(|value| *value as i64).collect();
    let (embed_step, embed_out) = make_step(
        &EmbeddingIndexOp {
            hidden: 1,
            vocab: 4,
        },
        &[token_values],
        &[1, 4],
        &[],
        vec![None],
        vec![None],
    );
    let (lookup_step, lookup_out) = make_step(
        &TableLookupOp,
        &[vec![0, 1, 2, 3], embed_out[0].clone()],
        &[],
        &[0],
        vec![Some(0), None],
        vec![None, Some((0, 0))],
    );
    // The table maps token 2 to the value 2, so step 0 and step 1 expose
    // equal values; only the planned edge identity distinguishes them.
    assert_eq!(embed_out[0], lookup_out[0]);
    let (argmax_step, argmax_out) = make_step(
        &ArgmaxOp,
        &[lookup_out[0].clone()],
        &[],
        &[],
        vec![None],
        vec![Some((1, 0))],
    );
    let trace = SharedTrace {
        tokens,
        assets: opening,
        steps: vec![embed_step, lookup_step, argmax_step],
        argmax: vec![argmax_out[0][0] as u32],
        output: Vec::new(),
    };
    let plan = table_lookup_plan(false, PlanSource::prev(0, 0));
    let fixture = Fixture {
        public,
        trace,
        tokens: vec![2],
        plan,
    };
    zkie_ops::proof::model_commitment::verify_shared_trace(
        &fixture.public,
        &fixture.trace,
        &fixture.tokens,
        &fixture.plan,
    )
    .unwrap();

    let mut negative = Fixture {
        public: fixture.public.clone(),
        trace: fixture.trace.clone(),
        tokens: fixture.tokens.clone(),
        plan: fixture.plan.clone(),
    };
    negative.trace.steps[2].sources[0] = Some((0, 0));
    assert_rejected(&negative);
}

#[test]
fn negative_slice() {
    let root = temp_dir();
    write_i32(&root.join("table_i32.bin"), &[10, 20, 30, 40]);
    let specs = vec![ModelAssetSpec {
        asset_id: 0,
        path: "table_i32.bin".into(),
        element_count: 4,
        row_width: 4,
        encoding: "i32-le".into(),
    }];
    let commitment =
        commit_assets(&root, Digest([5; 32]), Digest([6; 32]), &specs).unwrap();
    let public = ModelCommitmentPublic::from_commitment(&commitment);
    let opening = AssetStore::open(&commitment).unwrap().open_all().unwrap();
    let tokens = vec![0u32, 2, 1, 0];
    let token_values: Vec<i64> = tokens.iter().map(|value| *value as i64).collect();
    let (embed_step, embed_out) = make_step(
        &EmbeddingIndexOp {
            hidden: 1,
            vocab: 4,
        },
        &[token_values],
        &[1, 4],
        &[],
        vec![None],
        vec![None],
    );
    let table = vec![10, 20, 30, 40];
    let first = embed_out[0][0..2].to_vec();
    let (lookup_step, lookup_out) = make_step(
        &TableLookupOp,
        &[table, first],
        &[],
        &[0],
        vec![Some(0), None],
        vec![None, None],
    );
    let trace = SharedTrace {
        tokens,
        assets: opening,
        steps: vec![embed_step, lookup_step],
        argmax: Vec::new(),
        output: lookup_out[0].clone(),
    };
    let mut builder = PlanBuilder::new();
    builder.step(
        OpCode::EmbeddingIndex,
        config(vec![1, 4]),
        vec![PlanSource::Constant(PlanConstant::Tokens)],
        1,
    );
    let lookup = builder.step(
        OpCode::TableLookup,
        config(Vec::new()),
        vec![
            PlanSource::asset("table_i32.bin"),
            PlanSource::prev_slice(0, 0, PlanLayout::range(0, 2)),
        ],
        1,
    );
    builder.final_output(lookup, 0);
    let plan = builder.finish();
    let positive = Fixture {
        public: public.clone(),
        trace: trace.clone(),
        tokens: trace.tokens.clone(),
        plan: plan.clone(),
    };
    let output = zkie_ops::proof::model_commitment::verify_shared_trace_output(
        &positive.public,
        &positive.trace,
        &positive.tokens,
        &positive.plan,
    )
    .unwrap();
    assert_eq!(output, vec![10, 30]);

    let wrong = embed_out[0][1..3].to_vec();
    let (wrong_step, wrong_out) = make_step(
        &TableLookupOp,
        &[vec![10, 20, 30, 40], wrong],
        &[],
        &[0],
        vec![Some(0), None],
        vec![None, None],
    );
    let negative = Fixture {
        public,
        trace: SharedTrace {
            tokens: trace.tokens.clone(),
            assets: trace.assets.clone(),
            steps: vec![trace.steps[0].clone(), wrong_step],
            argmax: Vec::new(),
            output: wrong_out[0].clone(),
        },
        tokens: trace.tokens.clone(),
        plan,
    };
    assert_rejected(&negative);
}

#[test]
fn negative_final_output() {
    let fixture = one_asset_fixture(&[2], false);
    let result = zkie_ops::proof::model_commitment::verify_shared_trace_output(
        &fixture.public,
        &fixture.trace,
        &fixture.tokens,
        &fixture.plan,
    )
    .unwrap();
    assert_eq!(result, vec![30]);
    let mut negative = Fixture {
        public: fixture.public.clone(),
        trace: fixture.trace.clone(),
        tokens: fixture.tokens.clone(),
        plan: fixture.plan.clone(),
    };
    negative.trace.output = vec![20];
    assert_rejected(&negative);
}

#[test]
fn negative_wrong_asset() {
    let root = temp_dir();
    write_i32(&root.join("a_i32.bin"), &[10, 20, 30, 40]);
    write_i32(&root.join("b_i32.bin"), &[1, 2, 3, 4]);
    let specs = vec![
        ModelAssetSpec {
            asset_id: 0,
            path: "a_i32.bin".into(),
            element_count: 4,
            row_width: 4,
            encoding: "i32-le".into(),
        },
        ModelAssetSpec {
            asset_id: 1,
            path: "b_i32.bin".into(),
            element_count: 4,
            row_width: 4,
            encoding: "i32-le".into(),
        },
    ];
    let commitment =
        commit_assets(&root, Digest([3; 32]), Digest([4; 32]), &specs).unwrap();
    let public = ModelCommitmentPublic::from_commitment(&commitment);
    let opening = AssetStore::open(&commitment).unwrap().open_all().unwrap();
    let (embed_step, embed_out) = make_step(
        &EmbeddingIndexOp {
            hidden: 1,
            vocab: 4,
        },
        &[vec![0]],
        &[1, 4],
        &[],
        vec![None],
        vec![None],
    );
    let (lookup_step, lookup_out) = make_step(
        &TableLookupOp,
        &[vec![1, 2, 3, 4, 5, 6], embed_out[0].clone()],
        &[],
        &[1],
        vec![Some(1), None],
        vec![None, Some((0, 0))],
    );
    let trace = SharedTrace {
        tokens: vec![0],
        assets: opening,
        steps: vec![embed_step, lookup_step],
        argmax: Vec::new(),
        output: lookup_out[0].clone(),
    };
    let mut builder = PlanBuilder::new();
    let embed = builder.step(
        OpCode::EmbeddingIndex,
        config(vec![1, 4]),
        vec![PlanSource::Constant(PlanConstant::Tokens)],
        1,
    );
    let lookup = builder.step(
        OpCode::TableLookup,
        config(Vec::new()),
        vec![
            PlanSource::asset("a_i32.bin"),
            PlanSource::prev(embed, 0),
        ],
        1,
    );
    builder.final_output(lookup, 0);
    let plan = builder.finish();
    let fixture = Fixture {
        public,
        trace,
        tokens: vec![0],
        plan,
    };
    assert_rejected(&fixture);
}

#[test]
fn negative_external_opening_must_cover_the_full_asset() {
    let root = temp_dir();
    let table = vec![10i64, 20, 30, 40, 50, 60, 70, 80];
    let table_i32: Vec<i32> = table.iter().map(|value| *value as i32).collect();
    write_i32(&root.join("table_i32.bin"), &table_i32);
    let specs = vec![ModelAssetSpec {
        asset_id: 0,
        path: "table_i32.bin".into(),
        element_count: table.len() as u64,
        row_width: 4,
        encoding: "i32-le".into(),
    }];
    let commitment =
        commit_assets(&root, Digest([9; 32]), Digest([10; 32]), &specs).unwrap();
    let public = ModelCommitmentPublic::from_commitment(&commitment);
    let opening = AssetStore::open(&commitment).unwrap().open_all().unwrap();
    let tokens = vec![2u32];
    let token_values: Vec<i64> = tokens.iter().map(|value| *value as i64).collect();
    let (embed_step, embed_out) = make_step(
        &EmbeddingIndexOp {
            hidden: 1,
            vocab: 4,
        },
        &[token_values],
        &[1, 4],
        &[],
        vec![None],
        vec![None],
    );
    let (lookup_step, lookup_out) = make_step(
        &TableLookupOp,
        &[table, embed_out[0].clone()],
        &[],
        &[0],
        vec![Some(0), None],
        vec![None, Some((0, 0))],
    );
    let trace = SharedTrace {
        tokens,
        assets: opening,
        steps: vec![embed_step, lookup_step],
        argmax: Vec::new(),
        output: lookup_out[0].clone(),
    };
    let mut builder = PlanBuilder::new();
    let embed = builder.step(
        OpCode::EmbeddingIndex,
        config(vec![1, 4]),
        vec![PlanSource::Constant(PlanConstant::Tokens)],
        1,
    );
    let lookup = builder.step(
        OpCode::TableLookup,
        config(Vec::new()),
        vec![
            PlanSource::asset("table_i32.bin"),
            PlanSource::prev(embed, 0),
        ],
        1,
    );
    builder.final_output(lookup, 0);
    let plan = builder.finish();
    let fixture = Fixture {
        public,
        trace,
        tokens: vec![2],
        plan,
    };
    zkie_ops::proof::model_commitment::verify_shared_trace_output(
        &fixture.public,
        &fixture.trace,
        &fixture.tokens,
        &fixture.plan,
    )
    .unwrap();

    let mut negative = Fixture {
        public: fixture.public.clone(),
        trace: fixture.trace.clone(),
        tokens: fixture.tokens.clone(),
        plan: fixture.plan.clone(),
    };
    negative.trace.assets[0].rows.pop();
    negative.trace.assets[0].row_count = 1;
    assert_eq!(rejection_code(&negative), "statement_mismatch");

    // Structural mutations keep the original rows and Merkle paths valid.
    // Only the position/uniqueness/coverage checks can reject them.
    let mut duplicate = Fixture {
        public: fixture.public.clone(),
        trace: fixture.trace.clone(),
        tokens: fixture.tokens.clone(),
        plan: fixture.plan.clone(),
    };
    duplicate.trace.assets[0].rows[1] = duplicate.trace.assets[0].rows[0].clone();
    assert_eq!(rejection_code(&duplicate), "invalid_index");

    let mut swapped = Fixture {
        public: fixture.public.clone(),
        trace: fixture.trace.clone(),
        tokens: fixture.tokens.clone(),
        plan: fixture.plan.clone(),
    };
    swapped.trace.assets[0].rows.swap(0, 1);
    assert_eq!(rejection_code(&swapped), "invalid_index");

    let mut width = Fixture {
        public: fixture.public.clone(),
        trace: fixture.trace.clone(),
        tokens: fixture.tokens.clone(),
        plan: fixture.plan.clone(),
    };
    width.trace.assets[0].rows[0].values.pop();
    assert_eq!(rejection_code(&width), "shape_mismatch");
}
