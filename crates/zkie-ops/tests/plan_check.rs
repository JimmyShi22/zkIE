use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use zkie_core::types::{Digest, TensorSpec};
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
use zkie_ops::registry::{KernelRecord, PureForward};
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
            None,
        )
        .map(|_| ())
    } else {
        zkie_ops::proof::model_commitment::verify_shared_trace(
            &fixture.public,
            &trace,
            &fixture.tokens,
            &fixture.plan,
            None,
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
        None,
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
        None,
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
        None,
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
        None,
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
        None,
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

// ---- trusted public table sources ----

#[derive(Clone)]
struct PublicTableFixture {
    public: ModelCommitmentPublic,
    trace: SharedTrace,
    tokens: Vec<u32>,
    plan: TracePlan,
    root: PathBuf,
    opening: Vec<zkie_ops::proof::model_commitment::AssetOpening>,
}

fn public_table_plan(table: zkie_ops::proof::plan::PublicTable) -> TracePlan {
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
            PlanSource::public_table(table),
            PlanSource::prev(embed, 0),
        ],
        1,
    );
    let argmax = builder.step(
        OpCode::Argmax,
        config(Vec::new()),
        vec![PlanSource::prev(lookup, 0)],
        1,
    );
    builder.argmax_step(argmax);
    builder.finish()
}

fn public_table_identity(
    root: &PathBuf,
    locator: &str,
    entries: u64,
    digest: Digest,
) -> zkie_ops::proof::plan::PublicTable {
    zkie_ops::proof::plan::PublicTable {
        locator: locator.into(),
        digest,
        entries,
        encoding: zkie_ops::proof::plan::PublicTableEncoding::I32LeSigned,
    }
}

/// The production form: the trace carries no opening for the table; the
/// relation inputs are proved directly and the verifier materializes the
/// table from the trusted public root.
fn public_table_fixture(proof_values: &[i32], file_values: &[i32]) -> PublicTableFixture {
    public_table_fixture_at(proof_values, file_values, 2)
}

fn public_table_fixture_at(
    proof_values: &[i32],
    file_values: &[i32],
    token: u32,
) -> PublicTableFixture {
    let root = temp_dir();
    write_i32(&root.join("table_i32.bin"), file_values);
    let bytes = fs::read(root.join("table_i32.bin")).unwrap();
    let table = public_table_identity(
        &root,
        "table_i32.bin",
        file_values.len() as u64,
        zkie_core::artifact::digest_bytes(&bytes),
    );
    let specs = vec![ModelAssetSpec {
        asset_id: 0,
        path: "table_i32.bin".into(),
        element_count: file_values.len() as u64,
        row_width: file_values.len(),
        encoding: "i32-le".into(),
    }];
    let commitment = commit_assets(&root, Digest([5; 32]), Digest([6; 32]), &specs).unwrap();
    let public = ModelCommitmentPublic::from_commitment(&commitment);
    let opening = AssetStore::open(&commitment).unwrap().open_all().unwrap();
    let tokens = vec![token];
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
    let lookup_values: Vec<i64> = proof_values.iter().map(|value| *value as i64).collect();
    let (lookup_step, lookup_out) = make_step(
        &TableLookupOp,
        &[lookup_values, embed_out[0].clone()],
        &[],
        &[],
        vec![None, None],
        vec![None, Some((0, 0))],
    );
    let (argmax_step, argmax_out) = make_step(
        &ArgmaxOp,
        &[lookup_out[0].clone()],
        &[],
        &[],
        vec![None],
        vec![Some((1, 0))],
    );
    let plan = public_table_plan(table);
    let trace = SharedTrace {
        tokens: tokens.clone(),
        assets: Vec::new(),
        steps: vec![embed_step, lookup_step, argmax_step],
        argmax: vec![argmax_out[0][0] as u32],
        output: Vec::new(),
    };
    PublicTableFixture {
        public,
        trace,
        tokens,
        plan,
        root,
        opening,
    }
}

fn verify_public_trace(
    fixture: &PublicTableFixture,
    plan: &TracePlan,
    root: Option<&std::path::Path>,
) -> Result<Vec<u32>, zkie_core::error::Error> {
    let bytes = bincode::serialize(&fixture.trace).unwrap();
    let trace: SharedTrace = bincode::deserialize(&bytes).unwrap();
    zkie_ops::proof::model_commitment::verify_shared_trace(
        &fixture.public,
        &trace,
        &fixture.tokens,
        plan,
        root,
    )
}

#[test]
fn public_table_source_verifies_from_the_trusted_file() {
    let fixture = public_table_fixture(&[10, 20, 30, 40], &[10, 20, 30, 40]);
    assert!(fixture.trace.assets.is_empty(), "no table opening is carried");
    let argmax = verify_public_trace(&fixture, &fixture.plan, Some(&fixture.root)).unwrap();
    assert_eq!(argmax, vec![0]);
    // The same proof verifies twice: the read cache is per verification run.
    let argmax = verify_public_trace(&fixture, &fixture.plan, Some(&fixture.root)).unwrap();
    assert_eq!(argmax, vec![0]);
}

#[test]
fn public_table_without_a_trusted_root_is_rejected() {
    let fixture = public_table_fixture(&[10, 20, 30, 40], &[10, 20, 30, 40]);
    assert!(verify_public_trace(&fixture, &fixture.plan, None).is_err());
}

#[test]
fn public_table_file_swap_is_rejected() {
    let fixture = public_table_fixture(&[10, 20, 30, 40], &[10, 20, 30, 40]);
    let bytes: Vec<u8> = [11i32, 21, 31, 41]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    fs::write(fixture.root.join("table_i32.bin"), bytes).unwrap();
    assert!(verify_public_trace(&fixture, &fixture.plan, Some(&fixture.root)).is_err());
}

#[test]
fn public_table_truncated_or_padded_file_is_rejected() {
    let truncated = public_table_fixture(&[10, 20, 30, 40], &[10, 20, 30, 40]);
    let bytes = fs::read(truncated.root.join("table_i32.bin")).unwrap();
    fs::write(truncated.root.join("table_i32.bin"), &bytes[..bytes.len() - 1]).unwrap();
    assert!(verify_public_trace(&truncated, &truncated.plan, Some(&truncated.root)).is_err());

    let padded = public_table_fixture(&[10, 20, 30, 40], &[10, 20, 30, 40]);
    let mut bytes = fs::read(padded.root.join("table_i32.bin")).unwrap();
    bytes.push(0);
    fs::write(padded.root.join("table_i32.bin"), bytes).unwrap();
    assert!(verify_public_trace(&padded, &padded.plan, Some(&padded.root)).is_err());
}

#[test]
fn public_table_missing_file_is_rejected() {
    let fixture = public_table_fixture(&[10, 20, 30, 40], &[10, 20, 30, 40]);
    fs::remove_file(fixture.root.join("table_i32.bin")).unwrap();
    assert!(verify_public_trace(&fixture, &fixture.plan, Some(&fixture.root)).is_err());
}

#[test]
fn public_table_wrong_fixed_identity_is_rejected() {
    let fixture = public_table_fixture(&[10, 20, 30, 40], &[10, 20, 30, 40]);
    let bytes = fs::read(fixture.root.join("table_i32.bin")).unwrap();
    let real = zkie_core::artifact::digest_bytes(&bytes);

    // A fixed entry count that does not match the file length.
    let plan = public_table_plan(public_table_identity(
        &fixture.root,
        "table_i32.bin",
        3,
        real,
    ));
    assert!(verify_public_trace(&fixture, &plan, Some(&fixture.root)).is_err());

    // A fixed digest that is not the file digest.
    let plan = public_table_plan(public_table_identity(
        &fixture.root,
        "table_i32.bin",
        4,
        Digest([9; 32]),
    ));
    assert!(verify_public_trace(&fixture, &plan, Some(&fixture.root)).is_err());
}

#[test]
fn public_table_value_mismatch_is_rejected() {
    // The file digest is trusted, but the entry the plan reads differs from
    // the value the relation was proved over.
    let fixture = public_table_fixture(&[10, 20, 30, 40], &[10, 20, 31, 40]);
    let code = verify_public_trace(&fixture, &fixture.plan, Some(&fixture.root))
        .err()
        .expect("expected rejection")
        .code()
        .to_owned();
    assert_eq!(code, "invalid_proof");
}

#[test]
fn prover_supplied_table_opening_is_rejected() {
    let mut fixture = public_table_fixture(&[10, 20, 30, 40], &[10, 20, 30, 40]);
    fixture.trace.assets = fixture.opening.clone();
    fixture.trace.steps[1].external[0] = Some(0);
    let code = verify_public_trace(&fixture, &fixture.plan, Some(&fixture.root))
        .err()
        .expect("expected rejection")
        .code()
        .to_owned();
    assert_eq!(code, "statement_mismatch");
}

#[test]
fn public_table_short_table_rejects_index_past_the_end() {
    // The proof reads entry 3 of a four-entry table; the trusted file holds
    // three entries with its own fixed digest. The verifier must reject the
    // length change instead of clamping the index.
    let fixture = public_table_fixture_at(&[10, 20, 30, 40], &[10, 20, 30, 40], 3);
    write_i32(&fixture.root.join("table_i32.bin"), &[10, 20, 30]);
    let bytes = fs::read(fixture.root.join("table_i32.bin")).unwrap();
    let plan = public_table_plan(public_table_identity(
        &fixture.root,
        "table_i32.bin",
        3,
        zkie_core::artifact::digest_bytes(&bytes),
    ));
    assert!(verify_public_trace(&fixture, &plan, Some(&fixture.root)).is_err());
}

#[test]
fn public_table_wrong_output_is_rejected() {
    let mut fixture = public_table_fixture(&[10, 20, 30, 40], &[10, 20, 30, 40]);
    fixture.trace.argmax[0] ^= 1;
    assert!(verify_public_trace(&fixture, &fixture.plan, Some(&fixture.root)).is_err());
}

// ---- trusted public extension table in the rms norm consumer ----

#[derive(Clone)]
struct RmsPublicFixture {
    public: ModelCommitmentPublic,
    trace: SharedTrace,
    tokens: Vec<u32>,
    plan: TracePlan,
    root: PathBuf,
    opening: Vec<zkie_ops::proof::model_commitment::AssetOpening>,
}

fn rms_public_table(root: &PathBuf, locator: &str, values: &[i32]) -> zkie_ops::proof::plan::PublicTable {
    let bytes = fs::read(root.join(locator)).unwrap();
    zkie_ops::proof::plan::PublicTable {
        locator: locator.into(),
        digest: zkie_core::artifact::digest_bytes(&bytes),
        entries: values.len() as u64,
        encoding: zkie_ops::proof::plan::PublicTableEncoding::I32LeSigned,
    }
}

fn rms_public_fixture(
    x: i32,
    base_values: &[i32],
    ext_values: &[i32],
    plan_ext_entries: Option<u64>,
    plan_ext_digest: Option<Digest>,
) -> RmsPublicFixture {
    rms_public_fixture_with(x, base_values, ext_values, plan_ext_entries, plan_ext_digest, None)
}

#[derive(Clone, Copy)]
enum RmsTamper {
    MeanSquare,
    Rstd,
    Output,
}

struct TamperedRmsNorm {
    inner: zkie_ops::profile_rms::ProfileRmsNormOp,
    target: RmsTamper,
}

impl PureForward for TamperedRmsNorm {
    fn opcode(&self) -> OpCode {
        self.inner.opcode()
    }

    fn infer_specs(
        &self,
        inputs: &[TensorSpec],
    ) -> Result<Vec<TensorSpec>, zkie_core::error::Error> {
        self.inner.infer_specs(inputs)
    }

    fn forward_values(
        &self,
        inputs: &[Vec<i64>],
    ) -> Result<KernelRecord, zkie_core::error::Error> {
        // The relation prover commits exactly this record. Altering one
        // declared value before the record is returned makes the prover
        // rebuild every commitment, root and opening over the alteration,
        // so the opening checks describe the altered declaration rather
        // than a broken one.
        let mut record = self.inner.forward_values(inputs)?;
        match self.target {
            RmsTamper::MeanSquare => record.auxiliaries[0][0] ^= 1,
            RmsTamper::Rstd => record.auxiliaries[1][0] ^= 1,
            RmsTamper::Output => record.outputs[0][0] ^= 1,
        }
        Ok(record)
    }
}

fn rms_public_fixture_with(
    x: i32,
    base_values: &[i32],
    ext_values: &[i32],
    plan_ext_entries: Option<u64>,
    plan_ext_digest: Option<Digest>,
    tamper: Option<RmsTamper>,
) -> RmsPublicFixture {
    let root = temp_dir();
    write_i32(&root.join("base_i32.bin"), base_values);
    write_i32(&root.join("ext_i32.bin"), ext_values);
    let base = rms_public_table(&root, "base_i32.bin", base_values);
    let mut ext = rms_public_table(&root, "ext_i32.bin", ext_values);
    if let Some(entries) = plan_ext_entries {
        ext.entries = entries;
    }
    if let Some(digest) = plan_ext_digest {
        ext.digest = digest;
    }
    // A small committed asset keeps the public commitment well-formed; the
    // rms inputs come from the trusted public files, not from an opening.
    let specs = vec![ModelAssetSpec {
        asset_id: 0,
        path: "dummy_i32.bin".into(),
        element_count: 4,
        row_width: 4,
        encoding: "i32-le".into(),
    }];
    write_i32(&root.join("dummy_i32.bin"), &[1, 2, 3, 4]);
    let commitment = commit_assets(&root, Digest([11; 32]), Digest([12; 32]), &specs).unwrap();
    let public = ModelCommitmentPublic::from_commitment(&commitment);
    let opening = AssetStore::open(&commitment).unwrap().open_all().unwrap();

    let honest = zkie_ops::profile_rms::ProfileRmsNormOp {
        rows: 1,
        cols: 1,
        valid_cols: 1,
        output_shift: 0,
        clamp: None,
    };
    let tampered = tamper.map(|target| TamperedRmsNorm {
        inner: honest,
        target,
    });
    let op: &dyn PureForward = match &tampered {
        Some(wrapper) => wrapper,
        None => &honest,
    };
    let rms_config = vec![1, 1, 1, 0, 0, 0, 0];
    let inputs = vec![vec![x as i64], vec![1], base_values.iter().map(|v| *v as i64).collect(), ext_values.iter().map(|v| *v as i64).collect()];
    let (rms_step, rms_out) = make_step(
        op,
        &inputs,
        &rms_config,
        &[],
        vec![None, None, None, None],
        vec![None, None, None, None],
    );
    let (argmax_step, argmax_out) = make_step(
        &ArgmaxOp,
        &[rms_out[0].clone()],
        &[],
        &[],
        vec![None],
        vec![Some((0, 0))],
    );

    let mut builder = PlanBuilder::new();
    let rms = builder.step(
        OpCode::ProfileRmsNorm,
        config(rms_config.clone()),
        vec![
            PlanSource::Constant(PlanConstant::Literal(vec![x as i64])),
            PlanSource::Constant(PlanConstant::Literal(vec![1])),
            PlanSource::public_table(base),
            PlanSource::public_table(ext),
        ],
        1,
    );
    let argmax = builder.step(
        OpCode::Argmax,
        config(Vec::new()),
        vec![PlanSource::prev(rms, 0)],
        1,
    );
    builder.argmax_step(argmax);
    let plan = builder.finish();

    let trace = SharedTrace {
        tokens: vec![1],
        assets: Vec::new(),
        steps: vec![rms_step, argmax_step],
        argmax: vec![argmax_out[0][0] as u32],
        output: Vec::new(),
    };
    RmsPublicFixture {
        public,
        trace,
        tokens: vec![1],
        plan,
        root,
        opening,
    }
}

fn verify_rms_public(fixture: &RmsPublicFixture) -> Result<Vec<u32>, zkie_core::error::Error> {
    let bytes = bincode::serialize(&fixture.trace).unwrap();
    let trace: SharedTrace = bincode::deserialize(&bytes).unwrap();
    zkie_ops::proof::model_commitment::verify_shared_trace(
        &fixture.public,
        &trace,
        &fixture.tokens,
        &fixture.plan,
        Some(&fixture.root),
    )
}

#[test]
fn rms_extension_reads_the_trusted_public_file() {
    // x^2 / 2^18 = 4: the index is exactly the base length, so the extension's
    // first entry is consumed.
    let fixture = rms_public_fixture(1024, &[100, 200, 300, 400], &[500, 600, 700, 800], None, None);
    assert!(fixture.trace.assets.is_empty());
    let argmax = verify_rms_public(&fixture).unwrap();
    assert_eq!(argmax, vec![0]);
    // x^2 / 2^18 = 3: the last base entry is consumed, not the extension.
    let base_side = rms_public_fixture(887, &[100, 200, 300, 400], &[500, 600, 700, 800], None, None);
    let argmax = verify_rms_public(&base_side).unwrap();
    assert_eq!(argmax, vec![0]);
}

#[test]
fn rms_extension_file_tamper_is_rejected() {
    let swapped = rms_public_fixture(1024, &[100, 200, 300, 400], &[500, 600, 700, 800], None, None);
    let bytes: Vec<u8> = [501i32, 600, 700, 800]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    fs::write(swapped.root.join("ext_i32.bin"), bytes).unwrap();
    assert!(verify_rms_public(&swapped).is_err());

    let missing = rms_public_fixture(1024, &[100, 200, 300, 400], &[500, 600, 700, 800], None, None);
    fs::remove_file(missing.root.join("ext_i32.bin")).unwrap();
    assert!(verify_rms_public(&missing).is_err());

    let truncated = rms_public_fixture(1024, &[100, 200, 300, 400], &[500, 600, 700, 800], None, None);
    let bytes = fs::read(truncated.root.join("ext_i32.bin")).unwrap();
    fs::write(truncated.root.join("ext_i32.bin"), &bytes[..bytes.len() - 4]).unwrap();
    assert!(verify_rms_public(&truncated).is_err());

    let padded = rms_public_fixture(1024, &[100, 200, 300, 400], &[500, 600, 700, 800], None, None);
    let mut bytes = fs::read(padded.root.join("ext_i32.bin")).unwrap();
    bytes.push(0);
    fs::write(padded.root.join("ext_i32.bin"), bytes).unwrap();
    assert!(verify_rms_public(&padded).is_err());
}

#[test]
fn rms_extension_wrong_identity_or_short_table_is_rejected() {
    let wrong_digest = rms_public_fixture(
        1024,
        &[100, 200, 300, 400],
        &[500, 600, 700, 800],
        None,
        Some(Digest([9; 32])),
    );
    assert!(verify_rms_public(&wrong_digest).is_err());

    // The fixed entry count does not match the trusted file's byte length.
    let short = rms_public_fixture(
        1024,
        &[100, 200, 300, 400],
        &[500, 600, 700, 800],
        Some(3),
        None,
    );
    assert!(verify_rms_public(&short).is_err());
}

#[test]
fn rms_extension_prover_supplied_opening_is_rejected() {
    let mut fixture = rms_public_fixture(1024, &[100, 200, 300, 400], &[500, 600, 700, 800], None, None);
    fixture.trace.assets = fixture.opening.clone();
    fixture.trace.steps[0].external[3] = Some(0);
    let code = verify_rms_public(&fixture)
        .err()
        .expect("expected rejection")
        .code()
        .to_owned();
    assert_eq!(code, "statement_mismatch");
}

fn set_plan_activation(plan: &mut TracePlan, value: i64) {
    let step = &mut plan.steps[0];
    match &mut step.inputs[0] {
        PlanSource::Constant(PlanConstant::Literal(values)) => values[0] = value,
        _ => panic!("the activation is a literal"),
    }
}

#[test]
fn rms_extension_index_past_the_end_is_rejected() {
    // With the two-entry extension and the 1024 activation the derived index
    // is exactly the base length and the first extension entry is consumed:
    // this instance verifies. Raising the trusted activation to 1280 makes the
    // derived index six, past the base-four plus extension-two end; the file
    // identity and the input lengths stay legal.
    let mut fixture = rms_public_fixture(1024, &[100, 200, 300, 400], &[500, 600], None, None);
    verify_rms_public(&fixture).expect("the legal boundary instance verifies");
    set_plan_activation(&mut fixture.plan, 1280);
    let code = verify_rms_public(&fixture)
        .err()
        .expect("the past-end index is rejected")
        .code()
        .to_owned();
    // The past-end index enters the consumption relation check and the
    // verifier rejects the proof there (the file identity and the input
    // lengths are legal, so the rejection is not a length check).
    assert_eq!(code, "invalid_proof");
}

/// Re-verifies every committed declaration group of the step against the
/// roots carried in the message, using the production Merkle primitives. A
/// full-coverage group carries every leaf and no path, so its root is rebuilt
/// from those leaves; a path form is checked per opening.
fn committed_openings_authenticate(
    step: &SharedTraceStep,
) -> Result<(), zkie_core::error::Error> {
    let mut registry = zkie_ops::proof::binding::OracleRegistry::new();
    for (root, start, count) in &step.message.roots {
        registry.insert(*root, *start, *count)?;
    }
    let groups = step
        .message
        .proof
        .outputs
        .iter()
        .chain(&step.message.proof.raw)
        .chain(&step.message.proof.quotients)
        .chain(&step.message.proof.helpers)
        .chain(&step.message.proof.aux);
    for group in groups {
        if group.is_empty() {
            continue;
        }
        let oracle = group[0].oracle;
        let root = registry
            .get(oracle)
            .ok_or(zkie_core::error::Error::ProverStateMissing)?;
        if group.len() as u64 != root.count
            || group.iter().any(|opening| opening.oracle != oracle)
        {
            return Err(zkie_core::error::Error::StatementMismatch);
        }
        if group.iter().all(|opening| opening.path.is_empty()) {
            let mut leaves = Vec::with_capacity(group.len());
            for opening in group {
                leaves.push(zkie_ops::proof::binding::oracle_leaf(
                    opening.index,
                    &opening.values,
                ));
            }
            let rebuilt = zkie_engine::compute::cpu::merkle_build(&root.start, &leaves)?;
            if rebuilt.root != root.root {
                return Err(zkie_core::error::Error::InvalidProof);
            }
        } else {
            for opening in group {
                zkie_ops::proof::binding::verify_opening(&registry, opening, root.count)?;
            }
        }
    }
    Ok(())
}

#[test]
fn rms_extension_consumption_tamper_is_rejected() {
    // Each instance is proved over a record whose declared value was altered
    // before the relation prover committed it: the message roots are the
    // roots of the altered declarations, so the opening checks accept the
    // material. The full verification then rejects because the ProfileRmsNorm
    // relation recomputes the declaration from the trusted inputs and
    // compares it with the committed value (mean square, rstd, norm output in
    // that check order). The raw product is no longer a declaration.
    let base = [100, 200, 300, 400];
    let ext = [500, 600, 700, 800];
    for (target, label) in [
        (RmsTamper::MeanSquare, "mean-square"),
        (RmsTamper::Rstd, "rstd"),
        (RmsTamper::Output, "norm output"),
    ] {
        let fixture =
            rms_public_fixture_with(1024, &base, &ext, None, None, Some(target));
        committed_openings_authenticate(&fixture.trace.steps[0]).unwrap_or_else(|error| {
            panic!("{label} declaration: the committed material is authenticated: {error:?}")
        });
        eprintln!("{label} declaration: authentication accepts the committed value");
        let code = verify_rms_public(&fixture)
            .err()
            .unwrap_or_else(|| panic!("{label} declaration: the altered value is rejected"))
            .code()
            .to_owned();
        eprintln!("{label} declaration: full verification rejects with {code}");
        assert_eq!(code, "invalid_proof");
    }
}

#[test]
fn rms_extension_legacy_raw_form_is_rejected() {
    // The same relation proved with the ordinary input-committed form still
    // carries raw; the shared path rejects that form instead of accepting a
    // relaxed second representation.
    let fixture = rms_public_fixture(1024, &[100, 200, 300, 400], &[500, 600, 700, 800], None, None);
    let op = zkie_ops::profile_rms::ProfileRmsNormOp {
        rows: 1,
        cols: 1,
        valid_cols: 1,
        output_shift: 0,
        clamp: None,
    };
    let inputs = vec![
        vec![1024],
        vec![1],
        vec![100, 200, 300, 400],
        vec![500, 600, 700, 800],
    ];
    let config = vec![1, 1, 1, 0, 0, 0, 0];
    let (public, proof) = prove_op_relation_with_inputs(&op, &inputs, &config, &[0, 1, 2, 3]).unwrap();
    assert!(!proof.raw.is_empty());
    let mut roots = Vec::new();
    for id in public.registry.ids() {
        let root = public.registry.get(id).unwrap();
        roots.push((root.root, root.start, root.count));
    }
    let legacy = SharedTraceStep {
        message: RelationReceiptMessage {
            opcode: OpCode::ProfileRmsNorm as u16,
            config: config.clone(),
            roots,
            external_inputs: vec![0, 1, 2, 3],
            proof,
        },
        external: vec![None, None, None, None],
        sources: vec![None, None, None, None],
    };
    let mut trace = fixture.trace.clone();
    trace.steps[0] = legacy;
    let changed = RmsPublicFixture {
        public: fixture.public.clone(),
        trace,
        tokens: fixture.tokens.clone(),
        plan: fixture.plan.clone(),
        root: fixture.root.clone(),
        opening: fixture.opening.clone(),
    };
    let code = verify_rms_public(&changed)
        .err()
        .expect("expected rejection")
        .code()
        .to_owned();
    assert_eq!(code, "statement_mismatch");
}

// ---- trusted public rope coefficient tables ----

#[derive(Clone)]
struct RopePublicFixture {
    public: ModelCommitmentPublic,
    trace: SharedTrace,
    tokens: Vec<u32>,
    plan: TracePlan,
    root: PathBuf,
    opening: Vec<zkie_ops::proof::model_commitment::AssetOpening>,
    cos: zkie_ops::proof::plan::PublicTable,
    sin: zkie_ops::proof::plan::PublicTable,
}

fn rope_public_fixture(x: &[i64], cos_values: &[i32], sin_values: &[i32]) -> RopePublicFixture {
    let root = temp_dir();
    write_i32(&root.join("cos_i32.bin"), cos_values);
    write_i32(&root.join("sin_i32.bin"), sin_values);
    let cos = rms_public_table(&root, "cos_i32.bin", cos_values);
    let sin = rms_public_table(&root, "sin_i32.bin", sin_values);
    let specs = vec![ModelAssetSpec {
        asset_id: 0,
        path: "dummy_i32.bin".into(),
        element_count: 4,
        row_width: 4,
        encoding: "i32-le".into(),
    }];
    write_i32(&root.join("dummy_i32.bin"), &[1, 2, 3, 4]);
    let commitment = commit_assets(&root, Digest([21; 32]), Digest([22; 32]), &specs).unwrap();
    let public = ModelCommitmentPublic::from_commitment(&commitment);
    let opening = AssetStore::open(&commitment).unwrap().open_all().unwrap();

    let head_dim = 4usize;
    let half = head_dim / 2;
    let op = zkie_ops::profile_rope::ProfileRopeOp {
        head_dim,
        shift: 0,
    };
    let rope_config = vec![head_dim as i64, 0];
    let cos_input: Vec<i64> = cos_values[..half].iter().map(|value| *value as i64).collect();
    let sin_input: Vec<i64> = sin_values[..half].iter().map(|value| *value as i64).collect();
    let inputs = vec![x.to_vec(), cos_input, sin_input];
    let (rope_step, rope_out) = make_step(
        &op,
        &inputs,
        &rope_config,
        &[],
        vec![None, None, None],
        vec![None, None, None],
    );
    let (argmax_step, argmax_out) = make_step(
        &ArgmaxOp,
        &[rope_out[0].clone()],
        &[],
        &[],
        vec![None],
        vec![Some((0, 0))],
    );

    let mut builder = PlanBuilder::new();
    let rope = builder.step(
        OpCode::ProfileRope,
        config(rope_config.clone()),
        vec![
            PlanSource::Constant(PlanConstant::Literal(x.to_vec())),
            PlanSource::PublicTable {
                table: cos.clone(),
                layout: PlanLayout::range(0, half),
            },
            PlanSource::PublicTable {
                table: sin.clone(),
                layout: PlanLayout::range(0, half),
            },
        ],
        1,
    );
    let argmax = builder.step(
        OpCode::Argmax,
        config(Vec::new()),
        vec![PlanSource::prev(rope, 0)],
        1,
    );
    builder.argmax_step(argmax);
    let plan = builder.finish();

    let trace = SharedTrace {
        tokens: vec![1],
        assets: Vec::new(),
        steps: vec![rope_step, argmax_step],
        argmax: vec![argmax_out[0][0] as u32],
        output: Vec::new(),
    };
    RopePublicFixture {
        public,
        trace,
        tokens: vec![1],
        plan,
        root,
        opening,
        cos,
        sin,
    }
}

fn verify_rope_public(fixture: &RopePublicFixture) -> Result<Vec<u32>, zkie_core::error::Error> {
    let bytes = bincode::serialize(&fixture.trace).unwrap();
    let trace: SharedTrace = bincode::deserialize(&bytes).unwrap();
    zkie_ops::proof::model_commitment::verify_shared_trace(
        &fixture.public,
        &trace,
        &fixture.tokens,
        &fixture.plan,
        Some(&fixture.root),
    )
}

fn set_plan_rope_source(plan: &mut TracePlan, port: usize, source: PlanSource) {
    plan.steps[0].inputs[port] = source;
}

#[test]
fn rope_public_tables_are_read_from_the_trusted_files() {
    let fixture = rope_public_fixture(&[3, 5, 7, 11], &[10, 20, 30, 40], &[5, 6, 7, 8]);
    assert!(fixture.trace.assets.is_empty());
    let argmax = verify_rope_public(&fixture).unwrap();
    assert_eq!(argmax, vec![3]);
}

#[test]
fn rope_public_file_identity_is_checked() {
    let missing = rope_public_fixture(&[3, 5, 7, 11], &[10, 20, 30, 40], &[5, 6, 7, 8]);
    fs::remove_file(missing.root.join("cos_i32.bin")).unwrap();
    let code = verify_rope_public(&missing)
        .err()
        .expect("expected rejection")
        .code()
        .to_owned();
    assert_eq!(code, "io");

    let truncated = rope_public_fixture(&[3, 5, 7, 11], &[10, 20, 30, 40], &[5, 6, 7, 8]);
    let bytes = fs::read(truncated.root.join("cos_i32.bin")).unwrap();
    fs::write(truncated.root.join("cos_i32.bin"), &bytes[..bytes.len() - 4]).unwrap();
    let code = verify_rope_public(&truncated)
        .err()
        .expect("expected rejection")
        .code()
        .to_owned();
    assert_eq!(code, "shape_mismatch");

    let padded = rope_public_fixture(&[3, 5, 7, 11], &[10, 20, 30, 40], &[5, 6, 7, 8]);
    let mut bytes = fs::read(padded.root.join("cos_i32.bin")).unwrap();
    bytes.push(0);
    fs::write(padded.root.join("cos_i32.bin"), bytes).unwrap();
    let code = verify_rope_public(&padded)
        .err()
        .expect("expected rejection")
        .code()
        .to_owned();
    assert_eq!(code, "shape_mismatch");

    let mut wrong_digest =
        rope_public_fixture(&[3, 5, 7, 11], &[10, 20, 30, 40], &[5, 6, 7, 8]);
    let mut cos = wrong_digest.cos.clone();
    cos.digest = Digest([9; 32]);
    set_plan_rope_source(
        &mut wrong_digest.plan,
        1,
        PlanSource::PublicTable {
            table: cos,
            layout: PlanLayout::range(0, 2),
        },
    );
    let code = verify_rope_public(&wrong_digest)
        .err()
        .expect("expected rejection")
        .code()
        .to_owned();
    assert_eq!(code, "asset_mismatch");

    // The fixed entry count does not match the trusted file's byte length.
    let mut short = rope_public_fixture(&[3, 5, 7, 11], &[10, 20, 30, 40], &[5, 6, 7, 8]);
    let mut cos = short.cos.clone();
    cos.entries = 3;
    set_plan_rope_source(
        &mut short.plan,
        1,
        PlanSource::PublicTable {
            table: cos,
            layout: PlanLayout::range(0, 2),
        },
    );
    let code = verify_rope_public(&short)
        .err()
        .expect("expected rejection")
        .code()
        .to_owned();
    assert_eq!(code, "shape_mismatch");
}

#[test]
fn rope_public_wrong_table_or_slice_is_rejected_by_the_rotation_relation() {
    // The proof is produced over the cos prefix [10, 20]; the trusted plan
    // maps the cos port to the sin table, whose file identity is legal. The
    // proof material stays valid and the rotation relation rejects the
    // committed raw and output values.
    let mut swapped = rope_public_fixture(&[3, 5, 7, 11], &[10, 20, 30, 40], &[5, 6, 7, 8]);
    let sin = swapped.sin.clone();
    set_plan_rope_source(
        &mut swapped.plan,
        1,
        PlanSource::PublicTable {
            table: sin,
            layout: PlanLayout::range(0, 2),
        },
    );
    let code = verify_rope_public(&swapped)
        .err()
        .expect("expected rejection")
        .code()
        .to_owned();
    assert_eq!(code, "invalid_proof");

    // The same table with a different prefix slice: [20, 30] instead of the
    // proved [10, 20]. The lengths are unchanged, so the rejection is the
    // rotation relation, not a shape check.
    let mut sliced = rope_public_fixture(&[3, 5, 7, 11], &[10, 20, 30, 40], &[5, 6, 7, 8]);
    let cos = sliced.cos.clone();
    set_plan_rope_source(
        &mut sliced.plan,
        1,
        PlanSource::PublicTable {
            table: cos,
            layout: PlanLayout::range(1, 2),
        },
    );
    let code = verify_rope_public(&sliced)
        .err()
        .expect("expected rejection")
        .code()
        .to_owned();
    assert_eq!(code, "invalid_proof");
}

#[test]
fn rope_public_port_injection_is_rejected() {
    let mut fixture = rope_public_fixture(&[3, 5, 7, 11], &[10, 20, 30, 40], &[5, 6, 7, 8]);
    fixture.trace.assets = fixture.opening.clone();
    fixture.trace.steps[0].external[1] = Some(0);
    let code = verify_rope_public(&fixture)
        .err()
        .expect("expected rejection")
        .code()
        .to_owned();
    assert_eq!(code, "statement_mismatch");
}

// ---- shared matrix relations without the raw declaration ----

#[derive(Clone)]
struct MatMulSharedFixture {
    public: ModelCommitmentPublic,
    trace: SharedTrace,
    tokens: Vec<u32>,
    plan: TracePlan,
    root: PathBuf,
}

struct TamperedMatMul {
    inner: zkie_ops::matmul::MatMulOp,
    output_xor: i64,
}

impl PureForward for TamperedMatMul {
    fn opcode(&self) -> OpCode {
        self.inner.opcode()
    }

    fn infer_specs(
        &self,
        inputs: &[TensorSpec],
    ) -> Result<Vec<TensorSpec>, zkie_core::error::Error> {
        self.inner.infer_specs(inputs)
    }

    fn forward_values(
        &self,
        inputs: &[Vec<i64>],
    ) -> Result<KernelRecord, zkie_core::error::Error> {
        let mut record = self.inner.forward_values(inputs)?;
        record.outputs[0][0] ^= self.output_xor;
        Ok(record)
    }
}

fn matmul_shared_config() -> Vec<i64> {
    vec![2, 2, 2, 0, 0]
}

fn matmul_shared_fixture_with(
    plan_lhs: &[i64],
    plan_rhs: &[i64],
    op: &dyn PureForward,
    proved_lhs: &[i64],
    proved_rhs: &[i64],
) -> MatMulSharedFixture {
    let root = temp_dir();
    let specs = vec![ModelAssetSpec {
        asset_id: 0,
        path: "dummy_i32.bin".into(),
        element_count: 4,
        row_width: 4,
        encoding: "i32-le".into(),
    }];
    write_i32(&root.join("dummy_i32.bin"), &[1, 2, 3, 4]);
    let commitment = commit_assets(&root, Digest([31; 32]), Digest([32; 32]), &specs).unwrap();
    let public = ModelCommitmentPublic::from_commitment(&commitment);
    let relation_config = matmul_shared_config();
    let (matmul_step, matmul_out) = make_step(
        op,
        &[proved_lhs.to_vec(), proved_rhs.to_vec()],
        &relation_config,
        &[],
        vec![None, None],
        vec![None, None],
    );
    let (argmax_step, argmax_out) = make_step(
        &ArgmaxOp,
        &[matmul_out[0].clone()],
        &[],
        &[],
        vec![None],
        vec![Some((0, 0))],
    );
    let mut builder = PlanBuilder::new();
    let matmul = builder.step(
        OpCode::MatMul,
        config(relation_config.clone()),
        vec![
            PlanSource::Constant(PlanConstant::Literal(plan_lhs.to_vec())),
            PlanSource::Constant(PlanConstant::Literal(plan_rhs.to_vec())),
        ],
        1,
    );
    let argmax = builder.step(
        OpCode::Argmax,
        config(Vec::new()),
        vec![PlanSource::prev(matmul, 0)],
        1,
    );
    builder.argmax_step(argmax);
    let plan = builder.finish();
    let trace = SharedTrace {
        tokens: vec![1],
        assets: Vec::new(),
        steps: vec![matmul_step, argmax_step],
        argmax: vec![argmax_out[0][0] as u32],
        output: Vec::new(),
    };
    MatMulSharedFixture {
        public,
        trace,
        tokens: vec![1],
        plan,
        root,
    }
}

fn matmul_shared_fixture(plan_lhs: &[i64], plan_rhs: &[i64]) -> MatMulSharedFixture {
    let op = zkie_ops::matmul::MatMulOp {
        m: 2,
        n: 2,
        k: 2,
        shift: 0,
    };
    matmul_shared_fixture_with(plan_lhs, plan_rhs, &op, plan_lhs, plan_rhs)
}

fn verify_matmul_shared(
    fixture: &MatMulSharedFixture,
) -> Result<Vec<u32>, zkie_core::error::Error> {
    let bytes = bincode::serialize(&fixture.trace).unwrap();
    let trace: SharedTrace = bincode::deserialize(&bytes).unwrap();
    zkie_ops::proof::model_commitment::verify_shared_trace(
        &fixture.public,
        &trace,
        &fixture.tokens,
        &fixture.plan,
        Some(&fixture.root),
    )
}

#[test]
fn shared_matmul_carries_no_raw_declaration() {
    let fixture = matmul_shared_fixture(&[3, 1, 2, 4], &[1, 2, 3, 4]);
    let proof = &fixture.trace.steps[0].message.proof;
    assert!(proof.raw.is_empty());
    assert!(proof.lengths.raw.is_empty());
    committed_openings_authenticate(&fixture.trace.steps[0])
        .expect("the committed material is authenticated");
    let argmax = verify_matmul_shared(&fixture).unwrap();
    assert_eq!(argmax, vec![3]);
}

#[test]
fn shared_matmul_output_tamper_is_rejected_by_the_relation() {
    let op = TamperedMatMul {
        inner: zkie_ops::matmul::MatMulOp {
            m: 2,
            n: 2,
            k: 2,
            shift: 0,
        },
        output_xor: 1,
    };
    let fixture = matmul_shared_fixture_with(
        &[3, 1, 2, 4],
        &[1, 2, 3, 4],
        &op,
        &[3, 1, 2, 4],
        &[1, 2, 3, 4],
    );
    committed_openings_authenticate(&fixture.trace.steps[0])
        .expect("the committed material is authenticated");
    let code = verify_matmul_shared(&fixture)
        .err()
        .expect("expected rejection")
        .code()
        .to_owned();
    assert_eq!(code, "invalid_proof");
}

#[test]
fn shared_matmul_plan_input_replacement_is_rejected_by_the_relation() {
    let fixture = matmul_shared_fixture_with(
        &[4, 1, 2, 4],
        &[1, 2, 3, 4],
        &zkie_ops::matmul::MatMulOp {
            m: 2,
            n: 2,
            k: 2,
            shift: 0,
        },
        &[3, 1, 2, 4],
        &[1, 2, 3, 4],
    );
    let code = verify_matmul_shared(&fixture)
        .err()
        .expect("expected rejection")
        .code()
        .to_owned();
    assert_eq!(code, "invalid_proof");
}

#[test]
fn shared_matmul_legacy_raw_declaration_is_rejected() {
    let fixture = matmul_shared_fixture(&[3, 1, 2, 4], &[1, 2, 3, 4]);
    let op = zkie_ops::matmul::MatMulOp {
        m: 2,
        n: 2,
        k: 2,
        shift: 0,
    };
    let config = matmul_shared_config();
    let (public, proof) = prove_op_relation_with_inputs(
        &op,
        &[vec![3, 1, 2, 4], vec![1, 2, 3, 4]],
        &config,
        &[0, 1],
    )
    .unwrap();
    assert!(!proof.raw.is_empty());
    let mut roots = Vec::new();
    for id in public.registry.ids() {
        let root = public.registry.get(id).unwrap();
        roots.push((root.root, root.start, root.count));
    }
    let legacy = SharedTraceStep {
        message: RelationReceiptMessage {
            opcode: OpCode::MatMul as u16,
            config: config.clone(),
            roots,
            external_inputs: vec![0, 1],
            proof,
        },
        external: vec![None, None],
        sources: vec![None, None],
    };
    let mut trace = fixture.trace.clone();
    trace.steps[0] = legacy;
    let changed = MatMulSharedFixture {
        public: fixture.public.clone(),
        trace,
        tokens: fixture.tokens.clone(),
        plan: fixture.plan.clone(),
        root: fixture.root.clone(),
    };
    let code = verify_matmul_shared(&changed)
        .err()
        .expect("expected rejection")
        .code()
        .to_owned();
    assert_eq!(code, "statement_mismatch");
}

#[test]
fn shared_matmul_raw_books_that_do_not_match_the_empty_form_are_rejected() {
    let mut length_only = matmul_shared_fixture(&[3, 1, 2, 4], &[1, 2, 3, 4]);
    length_only.trace.steps[0].message.proof.lengths.raw = vec![4];
    let code = verify_matmul_shared(&length_only)
        .err()
        .expect("expected rejection")
        .code()
        .to_owned();
    assert_eq!(code, "statement_mismatch");

    let mut opening_only = matmul_shared_fixture(&[3, 1, 2, 4], &[1, 2, 3, 4]);
    opening_only.trace.steps[0].message.proof.raw.push(Vec::new());
    let code = verify_matmul_shared(&opening_only)
        .err()
        .expect("expected rejection")
        .code()
        .to_owned();
    assert_eq!(code, "statement_mismatch");
}

#[test]
fn shared_matmul_malformed_bias_config_is_rejected_before_relation() {
    // The proof declares one bias value for four output slots while the
    // trusted plan declares none; the real entry rejects the config, it does
    // not reach a slot index.
    let mut fixture = matmul_shared_fixture(&[3, 1, 2, 4], &[1, 2, 3, 4]);
    fixture.trace.steps[0].message.config = vec![2, 2, 2, 0, 1, 7];
    let code = verify_matmul_shared(&fixture)
        .err()
        .expect("expected rejection")
        .code()
        .to_owned();
    assert_eq!(code, "statement_mismatch");
}

// ---- shared softmax declarations ----

#[derive(Clone)]
struct SoftmaxSharedFixture {
    public: ModelCommitmentPublic,
    trace: SharedTrace,
    tokens: Vec<u32>,
    plan: TracePlan,
    root: PathBuf,
}

#[derive(Clone, Copy)]
enum SoftmaxTamper {
    ExpEntry,
    Denominator,
}

struct TamperedSoftmax {
    inner: zkie_ops::softmax::SoftmaxOp,
    target: SoftmaxTamper,
}

impl PureForward for TamperedSoftmax {
    fn opcode(&self) -> OpCode {
        self.inner.opcode()
    }

    fn infer_specs(
        &self,
        inputs: &[TensorSpec],
    ) -> Result<Vec<TensorSpec>, zkie_core::error::Error> {
        self.inner.infer_specs(inputs)
    }

    fn forward_values(
        &self,
        inputs: &[Vec<i64>],
    ) -> Result<KernelRecord, zkie_core::error::Error> {
        let mut record = self.inner.forward_values(inputs)?;
        match self.target {
            SoftmaxTamper::ExpEntry => record.auxiliaries[1][0] ^= 1,
            SoftmaxTamper::Denominator => record.auxiliaries[2][0] ^= 1,
        }
        Ok(record)
    }
}

fn softmax_shared_fixture_with(
    plan_table: &[i64],
    op: &dyn PureForward,
    proved_table: &[i64],
) -> SoftmaxSharedFixture {
    let root = temp_dir();
    let specs = vec![ModelAssetSpec {
        asset_id: 0,
        path: "dummy_i32.bin".into(),
        element_count: 4,
        row_width: 4,
        encoding: "i32-le".into(),
    }];
    write_i32(&root.join("dummy_i32.bin"), &[1, 2, 3, 4]);
    let commitment = commit_assets(&root, Digest([41; 32]), Digest([42; 32]), &specs).unwrap();
    let public = ModelCommitmentPublic::from_commitment(&commitment);
    let logits = vec![1i64, 2, 3];
    let relation_config = vec![0];
    let (softmax_step, softmax_out) = make_step(
        op,
        &[logits.clone(), proved_table.to_vec()],
        &relation_config,
        &[],
        vec![None, None],
        vec![None, None],
    );
    let (argmax_step, argmax_out) = make_step(
        &ArgmaxOp,
        &[softmax_out[0].clone()],
        &[],
        &[],
        vec![None],
        vec![Some((0, 0))],
    );
    let mut builder = PlanBuilder::new();
    let softmax = builder.step(
        OpCode::Softmax,
        config(relation_config.clone()),
        vec![
            PlanSource::Constant(PlanConstant::Literal(logits)),
            PlanSource::Constant(PlanConstant::Literal(plan_table.to_vec())),
        ],
        1,
    );
    let argmax = builder.step(
        OpCode::Argmax,
        config(Vec::new()),
        vec![PlanSource::prev(softmax, 0)],
        1,
    );
    builder.argmax_step(argmax);
    let plan = builder.finish();
    let trace = SharedTrace {
        tokens: vec![1],
        assets: Vec::new(),
        steps: vec![softmax_step, argmax_step],
        argmax: vec![argmax_out[0][0] as u32],
        output: Vec::new(),
    };
    SoftmaxSharedFixture {
        public,
        trace,
        tokens: vec![1],
        plan,
        root,
    }
}

fn softmax_shared_fixture(plan_table: &[i64], proved_table: &[i64]) -> SoftmaxSharedFixture {
    let op = zkie_ops::softmax::SoftmaxOp { shift: 0 };
    softmax_shared_fixture_with(plan_table, &op, proved_table)
}

fn verify_softmax_shared(
    fixture: &SoftmaxSharedFixture,
) -> Result<Vec<u32>, zkie_core::error::Error> {
    let bytes = bincode::serialize(&fixture.trace).unwrap();
    let trace: SharedTrace = bincode::deserialize(&bytes).unwrap();
    zkie_ops::proof::model_commitment::verify_shared_trace(
        &fixture.public,
        &trace,
        &fixture.tokens,
        &fixture.plan,
        Some(&fixture.root),
    )
}

#[test]
fn shared_softmax_carries_no_raw_and_verifies() {
    let table = vec![1i64, 2, 3, 4, 5, 6, 7, 8];
    let fixture = softmax_shared_fixture(&table, &table);
    assert!(fixture.trace.steps[0].message.proof.raw.is_empty());
    assert!(fixture.trace.steps[0].message.proof.lengths.raw.is_empty());
    verify_softmax_shared(&fixture).unwrap();
}

#[test]
fn shared_softmax_aux_tamper_is_rejected_by_the_relation() {
    // The exp entry and the denominator are checked against the actual table
    // lookup; altering either declaration is rebuilt into the commitment and
    // then rejected by the numerical relation.
    let table = vec![1i64, 2, 3, 4, 5, 6, 7, 8];
    for target in [SoftmaxTamper::ExpEntry, SoftmaxTamper::Denominator] {
        let op = TamperedSoftmax {
            inner: zkie_ops::softmax::SoftmaxOp { shift: 0 },
            target,
        };
        let fixture = softmax_shared_fixture_with(&table, &op, &table);
        committed_openings_authenticate(&fixture.trace.steps[0])
            .expect("the committed material is authenticated");
        let code = verify_softmax_shared(&fixture)
            .err()
            .expect("expected rejection")
            .code()
            .to_owned();
        assert_eq!(code, "invalid_proof");
    }
}

#[test]
fn shared_softmax_table_replacement_is_rejected_by_the_relation() {
    // The proof is produced with one trusted table while the plan materializes
    // another of the same length; the verifier recomputes the exp entries and
    // rejects the committed auxiliary values.
    let proved = vec![1i64, 2, 3, 4, 5, 6, 7, 8];
    let planned = vec![2i64, 2, 2, 2, 2, 2, 2, 2];
    let fixture = softmax_shared_fixture(&planned, &proved);
    let code = verify_softmax_shared(&fixture)
        .err()
        .expect("expected rejection")
        .code()
        .to_owned();
    assert_eq!(code, "invalid_proof");
}

/// A two-step production-form fixture: an activation step and one consumer
/// step whose weight input is the consumer-authenticated asset. The trace can
/// be built with or without the old asset opening, and the plan with or
/// without the consumer promotion.
#[allow(clippy::too_many_arguments)]
fn consumer_trace_fixture(
    opcode: OpCode,
    layout: PlanLayout,
    with_opening: bool,
    consumer_source: bool,
    promote: bool,
) -> Fixture {
    let root = temp_dir();
    write_i32(&root.join("table_i32.bin"), &[1, 2, 3, 4]);
    let specs = vec![ModelAssetSpec {
        asset_id: 0,
        path: "table_i32.bin".into(),
        element_count: 4,
        row_width: 2,
        encoding: "i32-le".into(),
    }];
    let commitment = commit_assets(&root, Digest([1; 32]), Digest([2; 32]), &specs).unwrap();
    let public = ModelCommitmentPublic::from_commitment(&commitment);
    let mut assets = AssetStore::open(&commitment).unwrap().open_all().unwrap();
    if !with_opening {
        assets.clear();
    }
    let tokens = vec![1u32];
    let token_values: Vec<i64> = tokens.iter().map(|value| *value as i64).collect();
    let (embed_step, embed_out) = make_step(
        &EmbeddingIndexOp { hidden: 2, vocab: 4 },
        &[token_values],
        &[2, 4],
        &[],
        vec![None],
        vec![None],
    );
    let matrix = vec![1, 2, 2, 16, 0];
    let weight = vec![1, 2, 3, 4];
    let sources = vec![Some((0, 0)), None];
    let (consumer_step, consumer_out) = match opcode {
        OpCode::Projection => {
            let op = zkie_ops::projection::ProjectionOp {
                m: 1,
                n: 2,
                k: 2,
                shift: 16,
                bias: None,
            };
            make_step(
                &op,
                &[embed_out[0].clone(), weight],
                &matrix,
                &[],
                vec![None, None],
                sources,
            )
        }
        OpCode::MatMul => {
            let op = zkie_ops::matmul::MatMulOp {
                m: 1,
                n: 2,
                k: 2,
                shift: 16,
            };
            make_step(
                &op,
                &[embed_out[0].clone(), weight],
                &matrix,
                &[],
                vec![None, None],
                sources,
            )
        }
        _ => unreachable!(),
    };
    let trace = SharedTrace {
        tokens: tokens.clone(),
        assets,
        steps: vec![embed_step, consumer_step],
        argmax: Vec::new(),
        output: consumer_out[0].clone(),
    };
    let mut builder = PlanBuilder::new();
    let embed = builder.step(
        OpCode::EmbeddingIndex,
        config(vec![2, 4]),
        vec![PlanSource::Constant(PlanConstant::Tokens)],
        1,
    );
    let weight_source = if consumer_source {
        PlanSource::consumer_asset_slice("table_i32.bin", layout)
    } else {
        PlanSource::asset_slice("table_i32.bin", layout)
    };
    let consumer = builder.step(
        opcode,
        config(vec![1, 2, 2, 16, 0]),
        vec![PlanSource::prev(embed, 0), weight_source],
        1,
    );
    builder.final_output(consumer, 0);
    let mut plan = builder.finish();
    if promote {
        zkie_ops::proof::plan::promote_consumer_assets(&mut plan, &["table_i32.bin"]).unwrap();
    }
    Fixture {
        public,
        trace,
        tokens,
        plan,
    }
}

#[test]
fn consumer_asset_input_needs_no_trace_opening() {
    let fixture = consumer_trace_fixture(
        OpCode::Projection,
        PlanLayout::columns(2, 0, 2),
        false,
        false,
        true,
    );
    let output = zkie_ops::proof::model_commitment::verify_shared_trace_output(
        &fixture.public,
        &fixture.trace,
        &fixture.tokens,
        &fixture.plan,
        None,
    )
    .unwrap();
    assert_eq!(output, fixture.trace.output);
}

#[test]
fn a_flat_asset_input_without_a_pool_verifies_from_its_opening() {
    // A flat weight that is not named by a proof pool is materialized from its
    // own verified trace opening. This is the ordinary production form of
    // every non-pooled flat asset and must keep verifying.
    let mut fixture = one_asset_fixture(&[2], true);
    fixture.trace.steps[1].external[0] = None;
    let argmax = zkie_ops::proof::model_commitment::verify_shared_trace(
        &fixture.public,
        &fixture.trace,
        &fixture.tokens,
        &fixture.plan,
        None,
    )
    .unwrap();
    assert_eq!(argmax, fixture.trace.argmax);
}

#[test]
fn residual_consumer_asset_opening_is_rejected() {
    let fixture = consumer_trace_fixture(
        OpCode::Projection,
        PlanLayout::columns(2, 0, 2),
        true,
        false,
        true,
    );
    assert_eq!(rejection_code(&fixture), "statement_mismatch");
}

#[test]
fn prover_supplied_consumer_opening_is_rejected() {
    let mut fixture = consumer_trace_fixture(
        OpCode::Projection,
        PlanLayout::columns(2, 0, 2),
        false,
        false,
        true,
    );
    fixture.trace.steps[1].external[1] = Some(0);
    assert_eq!(rejection_code(&fixture), "statement_mismatch");
}

#[test]
fn consumer_source_in_a_non_projection_step_is_rejected() {
    let fixture =
        consumer_trace_fixture(OpCode::MatMul, PlanLayout::columns(2, 0, 2), false, false, true);
    assert_eq!(rejection_code(&fixture), "invalid_config");
}

#[test]
fn consumer_source_requires_the_columns_form() {
    let fixture =
        consumer_trace_fixture(OpCode::Projection, PlanLayout::Flat, false, true, false);
    assert_eq!(rejection_code(&fixture), "invalid_config");
}

#[test]
fn unpromoted_asset_source_needs_its_opening() {
    let fixture = consumer_trace_fixture(
        OpCode::Projection,
        PlanLayout::columns(2, 0, 2),
        false,
        false,
        false,
    );
    assert_eq!(rejection_code(&fixture), "statement_mismatch");
}
