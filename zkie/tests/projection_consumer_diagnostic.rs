//! Diagnostic closure of one real projection consumer.
//!
//! The test rebuilds the registered model commitment from the saved private
//! table randomness, verifies the real gpt2-seq16 trace, takes the verified
//! activation and output rows of one layer-0 key projection consumer, arranges
//! the real weight rows in the table layout the hidden session reduces, and
//! runs the production diagnostic hidden LIN prover and verifier on that
//! table. The full-weight relation checks stay in place; this is evidence for
//! the projection reduction core on real consumer values, not a protocol
//! change.

use std::cell::RefCell;
use std::fs;
use std::path::{Path, PathBuf};

use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::types::Digest;
use zkie_engine::pcs::request::code::{build_code_plan, CodeParams};
use zkie_engine::pcs::request::field::{fp_from_u64, Fp, Fp3, E_DEGREE};
use zkie_engine::pcs::request::foldeq::{reader_table_identity, OmegaShapePlan};
use zkie_engine::pcs::request::gu::{GuOracleShape, GuShape};
use zkie_engine::pcs::request::runtime_tree::{MaskTreePlan, RuntimeTreePart, RuntimeTreePlan};
use zkie_engine::pcs::request::table::{
    commit_table_streaming, ColumnOpening, IndexReader, IndexShape, StreamingTable, TableDistances,
    TableRowSource, VecTableSource,
};
use zkie_ops::op::OpCode;
use zkie_ops::proof::asset_table::decode_asset_table_randomness_set;
use zkie_ops::proof::column_session::fp_from_signed_word;
use zkie_ops::proof::hidden_send_session::{
    prove_hidden_lin, prove_hidden_lin_hidden, verify_hidden_lin, verify_hidden_lin_hidden,
    ActivationShape, CommittedActivation, DotMapPlan, HiddenLinPlan, HiddenOutputShape,
    HiddenProximityPlan, HiddenSendPlan, HiddenTableProver, IndexMaskPlan, IoppStagePlan,
    MaskEntryAllocation, MemberPlan, WShape,
};
use zkie_ops::proof::hidden_member::{
    stretch_weights, MaskTerm, PendingRow, ReductionTable,
};
use zkie_ops::proof::proximity_planner::{
    assemble_proximity, AssemblyInput, AssemblyMember, AssemblyOwnObject, SolvedStage,
};
use zkie::lm_head_stage::{
    read_manifest, write_manifest, ManifestFile, VerificationManifest,
};
use zkie_ops::proof::model_commitment::{
    verify_shared_trace_outputs, ModelCommitmentPublic, SharedTrace,
};
use zkie_ops::proof::plan::{PlanConfigValue, PlanSource};
use zkie_ops::proof::projection_claim::{
    check_projection_reduction, projection_claims, ProjectionShape,
};
use zkie_ops::registry::requantize;

const HIDDEN: usize = 1024;
const HEAD_DIM: usize = 64;
const HEADS: usize = 12;
const MATRIX_ROWS: u64 = 1024;
const K_LO: u32 = 0;
const COLUMNS: u64 = 1024;
const TABLE_LOG_N: u32 = 12;
const TABLE_ELL: u64 = 2048;
const TABLE_R: u64 = 8;
const CODE_LOG_N: u32 = 14;
const CODE_ELL: u64 = 8192;
const CODE_R: u64 = 4;
const SALT: usize = 40;
const GROUP: u64 = 7;
const ORACLE_BASE: u64 = 20;
const TABLE_ROWS: u64 = 1024;
const TABLE_CONFIGURATION: Digest = Digest([0x21; 32]);
const ACTIVATION_CONFIGURATION: Digest = Digest([0x31; 32]);
const W_CONFIGURATION: Digest = Digest([0x51; 32]);
const W_RUN: Digest = Digest([0x41; 32]);
const GU_ORACLE: u64 = 10;
const OMEGA_ORACLE: u64 = 11;
const GU_GROUP: u64 = 5;
const SEED: Digest = Digest([0x71; 32]);
const SEQUENCE_TOKENS: usize = 16;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

/// The registered instance trace: a seq16 commit/public bundle and a verified
/// statement and proof on disk. A checkout without them skips the test.
fn artifacts() -> Option<PathBuf> {
    let root = repo_root();
    if let Ok(dir) = std::env::var("ZKIE_PROJECTION_DIAGNOSTIC_DIR") {
        let dir = PathBuf::from(dir);
        if complete(&dir) {
            return Some(dir);
        }
        return None;
    }
    let mut dirs: Vec<PathBuf> = fs::read_dir(root.join("docs/benchmarks"))
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map_or(false, |name| name.contains("-model-table-instance-"))
        })
        .collect();
    dirs.sort();
    dirs.into_iter()
        .rev()
        .find(|dir| complete(&dir.join("smoke/gpt2-seq16")))
        .map(|dir| dir.join("smoke/gpt2-seq16"))
}

fn complete(base: &Path) -> bool {
    base.join("commit/public/model-commitment.bin").exists()
        && base.join("commit/public/provider-config.bin").exists()
        && base.join("commit/private/asset-table-randomness.bin").exists()
        && base.join("proof/statement.bin").exists()
        && base.join("proof/proof.bin").exists()
}

fn table_code() -> CodeParams {
    build_code_plan(TABLE_LOG_N, TABLE_ELL, TABLE_R).unwrap()
}

fn code() -> CodeParams {
    build_code_plan(CODE_LOG_N, CODE_ELL, CODE_R).unwrap()
}

fn table_shape() -> IndexShape {
    IndexShape {
        code: table_code(),
        v: 10,
        answers: 1,
        salt_bytes: SALT,
        distances: TableDistances {
            radius: 0,
            later_radius: 0,
            m_once: 0,
            m_later: 0,
        },
    }
}

fn w_shape(positions: u32) -> WShape {
    WShape::new(COLUMNS, code(), 1, positions, SALT, 100, 1, W_RUN, W_CONFIGURATION).unwrap()
}

fn gu_shape(code: CodeParams, mask_log_n: u32) -> GuShape {
    GuShape {
        oracle: GU_ORACLE,
        group: GU_GROUP,
        tree: tree_plan_with_depth(300, 2, 1, code.log_n, mask_log_n),
        part: 0,
        rows_high: true,
        u: GuOracleShape {
            code,
            main_rows: 1,
            main_deg: E_DEGREE as u32,
            mask_rows: 0,
            mask_code: None,
            answers: 1,
        },
        index: table_shape(),
    }
}

fn omega_shape(code: CodeParams, mask_log_n: u32) -> OmegaShapePlan {
    OmegaShapePlan {
        oracle: OMEGA_ORACLE,
        group: GU_GROUP,
        tree: tree_plan_with_depth(301, 2, 1, code.log_n, mask_log_n),
        u: GuOracleShape {
            code,
            main_rows: 1,
            main_deg: E_DEGREE as u32,
            mask_rows: 0,
            mask_code: None,
            answers: 1,
        },
        rows_high: true,
        matrix_rows: MATRIX_ROWS,
        k_lo: K_LO,
    }
}

fn tree_plan_with_depth(
    tree_number: u64,
    leaf_class: u32,
    main_rows: u64,
    log_n: u32,
    log_n_zk: u32,
) -> RuntimeTreePlan {
    RuntimeTreePlan {
        tree_number,
        leaf_class,
        log_n,
        log_n_zk,
        parts: vec![RuntimeTreePart {
            main_symbols: main_rows * E_DEGREE as u64,
            mask_symbols: 0,
        }],
        salt_bytes: SALT,
        run: Digest([0x61; 32]),
        configuration: Digest([0x62; 32]),
    }
}

fn mask_tree(tree_number: u64, log_n: u32, mask_rows: u64) -> MaskTreePlan {
    MaskTreePlan {
        tree_number,
        leaf_class: 3,
        log_n,
        mask_rows,
        salt_bytes: SALT,
        run: Digest([0x63; 32]),
        configuration: Digest([0x64; 32]),
    }
}

struct Table {
    table: StreamingTable,
    source: RefCell<VecTableSource>,
    reader: IndexReader,
}

impl TableRowSource for Table {
    fn rows(&self) -> u64 {
        self.source.borrow().rows()
    }

    fn row(
        &mut self,
        index: u64,
        message: &mut [Fp],
        tail: &mut [Fp],
    ) -> zkie_core::error::Result<()> {
        self.source.borrow_mut().row(index, message, tail)
    }
}

impl HiddenTableProver for Table {
    fn reader(&self) -> &IndexReader {
        &self.reader
    }

    fn open_columns(&mut self, positions: &[u64]) -> zkie_core::error::Result<Vec<ColumnOpening>> {
        let mut source = self.source.borrow_mut();
        self.table.open_columns(&mut *source, positions)
    }
}

/// The session table holds, in row `c`, the weights of output column `c` over
/// the input rows: `message[c][i] = W[i][c]`. The production weight asset is
/// stored input-major, so the diagnostic test writes the transposed rows; this
/// is the layout the hidden session reduces.
fn diagnostic_table(weights: &[i64]) -> Table {
    let shape = table_shape();
    let rows = shape.rows() as usize;
    let ell = shape.code.ell as usize;
    let mut messages = vec![vec![Fp::ZERO; ell]; rows];
    for (row, message) in messages.iter_mut().enumerate() {
        for input in 0..HIDDEN {
            let value = weights[input * HIDDEN + row] as i64;
            message[input] = fp_from_signed_word(value as u64);
        }
    }
    let tails = vec![vec![Fp::ZERO; TABLE_R as usize]; TABLE_ROWS as usize];
    let mut source = VecTableSource::new(messages, tails).unwrap();
    let salts: Vec<Vec<u8>> = (0..table_code().n())
        .map(|position| vec![(position as u8) ^ 0x5A; SALT])
        .collect();
    let table = commit_table_streaming(
        &mut source,
        shape,
        TABLE_CONFIGURATION,
        &salts,
        64,
        u64::MAX,
    )
    .unwrap();
    let reader = table.reader.clone();
    Table {
        table,
        source: RefCell::new(source),
        reader,
    }
}


fn activation(activation_rows: &[Vec<i64>], positions: u32) -> CommittedActivation {
    let salts: Vec<Vec<u8>> = (0..positions)
        .map(|position| vec![(position as u8).wrapping_mul(7) ^ 0x3C; SALT])
        .collect();
    CommittedActivation::commit_signed(
        ActivationShape::new(positions, HIDDEN as u32, SALT).unwrap(),
        activation_rows,
        &salts,
        ACTIVATION_CONFIGURATION,
    )
    .unwrap()
}

fn field_rows(rows: &[Vec<i64>]) -> Vec<Vec<Fp3>> {
    rows.iter()
        .map(|row| {
            row.iter()
                .map(|value| Fp3::from_base(fp_from_signed_word(*value as u64)))
                .collect()
        })
        .collect()
}

#[test]
#[ignore = "runs against the registered production trace artifacts"]
fn real_projection_consumer_closes_on_the_diagnostic_hidden_session() {
    let Some(base) = artifacts() else {
        return;
    };
    let public_bytes =
        fs::read(base.join("commit/public/model-commitment.bin")).expect("public bundle");
    let public: ModelCommitmentPublic = bincode::deserialize(&public_bytes).expect("public bundle");
    let config_bytes =
        fs::read(base.join("commit/public/provider-config.bin")).expect("provider config");
    let config: zkie::provider::ProviderConfig =
        bincode::deserialize(&config_bytes).expect("provider config");
    let statement_bytes = fs::read(base.join("proof/statement.bin")).expect("statement");
    let statement: zkie::provider::StatementData =
        bincode::deserialize(&statement_bytes).expect("statement");
    assert_eq!(statement.tokens.len(), SEQUENCE_TOKENS);

    let params = zkie_models_gpt2::plan::asset_table_params().expect("table params");
    let randomness =
        fs::read(base.join("commit/private/asset-table-randomness.bin")).expect("randomness");
    let tables = decode_asset_table_randomness_set(&randomness, &params).expect("randomness set");
    let commitment = zkie_models_gpt2::executor::build_commitment(
        Path::new(&config.asset_root),
        &config.tokens,
        &tables,
    )
    .expect("commitment");
    assert_eq!(
        commitment.anchor.record.anchor_id, public.anchor.anchor_id,
        "the rebuilt instance must be the registered one"
    );

    let trace_bytes = fs::read(base.join("proof/proof.bin")).expect("proof bytes");
    let trace: SharedTrace = bincode::deserialize(&trace_bytes).expect("trace");
    let plan = zkie_models_gpt2::plan::trace_plan(statement.tokens.len()).expect("plan");
    let outputs = verify_shared_trace_outputs(&public, &trace, &statement.tokens, &plan)
        .expect("verified trace outputs");

    let mut consumers = Vec::new();
    for (index, step) in plan.steps.iter().enumerate() {
        if step.opcode != OpCode::Projection {
            continue;
        }
        let Some(PlanSource::Asset { path, .. }) = step.inputs.get(1) else {
            continue;
        };
        if path != "gpt2/weights/L0_k_w_i32.bin" {
            continue;
        }
        let Some(PlanSource::Prev { step: activation_step, .. }) = step.inputs.first() else {
            panic!("the consumer activation is a previous step");
        };
        consumers.push((index, *activation_step));
    }
    assert_eq!(consumers.len(), HEADS);
    let activation_step = consumers[0].1;
    assert!(consumers.iter().all(|(_, step)| *step == activation_step));

    let positions = std::env::var("ZKIE_PROJECTION_DIAGNOSTIC_POSITIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2)
        .min(statement.tokens.len());
    let positions = positions as u32;

    let mut activation_rows = Vec::with_capacity(positions as usize);
    for position in 0..positions as usize {
        let row = outputs[activation_step][0]
            [position * HIDDEN..(position + 1) * HIDDEN]
            .to_vec();
        activation_rows.push(row);
    }

    let mut head_outputs = vec![vec![0i64; HIDDEN]; positions as usize];
    for (head, (step, _)) in consumers.iter().enumerate() {
        let values = &outputs[*step][0];
        for position in 0..positions as usize {
            head_outputs[position][head * HEAD_DIM..(head + 1) * HEAD_DIM]
                .copy_from_slice(&values[position * HEAD_DIM..(position + 1) * HEAD_DIM]);
        }
    }

    let store = zkie_ops::proof::model_commitment::AssetStore::open(&commitment).expect("store");
    let weights = store
        .value("gpt2/weights/L0_k_w_i32.bin")
        .expect("weight rows");
    assert_eq!(weights.len(), HIDDEN * HIDDEN);

    // The raw integer accumulator of the consumer, recomputed from the
    // verified activation rows and the trusted weight rows, must requantize to
    // the verified trace output.
    let shift = match plan.steps[consumers[0].0].config.get(3) {
        Some(PlanConfigValue::Literal(value)) => *value as u32,
        _ => panic!("the consumer shift is a literal"),
    };
    let mut raw = vec![vec![0i128; HIDDEN]; positions as usize];
    for position in 0..positions as usize {
        for input in 0..HIDDEN {
            let value = activation_rows[position][input] as i128;
            if value == 0 {
                continue;
            }
            for output in 0..HIDDEN {
                raw[position][output] += value * weights[input * HIDDEN + output] as i128;
            }
        }
    }
    for position in 0..positions as usize {
        for output in 0..HIDDEN {
            let expected = requantize(raw[position][output], shift).expect("requantize") as i64;
            assert_eq!(
                expected, head_outputs[position][output],
                "the verified trace output must be the requantized real reduction"
            );
        }
    }

    let mut session_table = diagnostic_table(&weights);
    let activation = activation(&activation_rows, positions);
    let plan = plan_for(&activation, &session_table, positions, &activation_rows);
    plan.send.validate().expect("send plan");
    if let Some(reason) = plan.proximity.fault(&plan.send, None, None) {
        panic!("proximity fault: {reason}");
    }
    plan.validate().expect("lin plan");

    let output: Vec<Vec<Fp3>> = raw
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| Fp3::from_base(fp_from_signed_word(*value as i64 as u64)))
                .collect()
        })
        .collect();
    let (messages, witness) =
        prove_hidden_lin(&plan, &activation, &mut session_table, &output, SEED).expect("prove");
    let mut verifier_plan = plan.clone();
    verifier_plan.send.expected_roots = Some(witness.send.roots.clone());
    let outcome = verify_hidden_lin(&verifier_plan, session_table.reader(), &output, &messages)
        .expect("verify");
    assert!(!outcome.eval_rows.is_empty());

    let shape = ProjectionShape {
        matrix_rows: MATRIX_ROWS,
        k_lo: K_LO,
        columns: COLUMNS,
        positions,
    };
    let flat_activation: Vec<Fp3> = field_rows(&activation_rows).into_iter().flatten().collect();
    let flat_output: Vec<Fp3> = output.iter().flatten().copied().collect();
    let claims = projection_claims(shape, &flat_activation, &flat_output).expect("claims");
    let plain_table: Vec<Vec<Fp3>> = table_messages(&weights);
    check_projection_reduction(shape, &claims.weights, &plain_table, &claims.output)
        .expect("reduction");

    // Negatives through the same diagnostic session.
    let mut changed = output.clone();
    changed[0][0] = changed[0][0].add(&Fp3::from_base(fp_from_u64(1)));
    assert!(verify_hidden_lin(&verifier_plan, session_table.reader(), &changed, &messages).is_err());

    let mut changed_plan = verifier_plan.clone();
    changed_plan.weights[0][0] = changed_plan.weights[0][0].add(&Fp3::from_base(fp_from_u64(1)));
    assert!(verify_hidden_lin(&changed_plan, session_table.reader(), &output, &messages).is_err());

    let mut changed_identity = plan.clone();
    changed_identity.send.table_identity.0[0] ^= 1;
    let mut source_table = diagnostic_table(&weights);
    assert!(prove_hidden_lin(&changed_identity, &activation, &mut source_table, &output, SEED)
        .is_err());
}

fn plan_for(
    activation: &CommittedActivation,
    table: &Table,
    positions: u32,
    activation_rows: &[Vec<i64>],
) -> HiddenLinPlan {
    HiddenLinPlan {
        norm: None,
        send: HiddenSendPlan {
            activation: activation.shape(),
            activation_configuration: ACTIVATION_CONFIGURATION,
            activation_root: activation.root(),
            activation_source: zkie_ops::proof::hidden_send_session::ActivationSource::Commitment,
            table_shape: table_shape(),
            table_identity: reader_table_identity(&table.reader).unwrap(),
            matrix_rows: MATRIX_ROWS,
            k_lo: K_LO,
            columns: COLUMNS,
            group: GROUP,
            oracle_base: ORACLE_BASE,
            openings_per_oracle: 2,
            w: w_shape(positions),
            index_mask: None,
            dot: None,
            expected_dot_root: None,
            expected_roots: None,
            member: None,
            request_chain: None,
            request_reads: None,
        },
        output: HiddenOutputShape {
            positions,
            columns: COLUMNS,
        },
        masks: vec![Vec::new(); positions as usize],
        weights: field_rows(activation_rows),
        proximity: HiddenProximityPlan {
            gu: gu_shape(code(), CODE_LOG_N),
            omega: omega_shape(code(), CODE_LOG_N),
            code: code(),
            mask_code: code(),
            shared_mask_tree: mask_tree(200, CODE_LOG_N, 1),
            own_mask_tree: mask_tree(201, CODE_LOG_N, 1),
            shared_fresh: 90,
            own_fresh: 91,
            column_draws: 2,
            column_budget: 4,
            column_work_bits: 0,
            shared_base_positions: 2,
            own_base_positions: 2,
            shared_stages: Vec::new(),
            own_stages: Vec::new(),
            planner_checked: false,
        },
        output_relation: None,
        closure: None,
    }
}

fn table_messages(weights: &[i64]) -> Vec<Vec<Fp3>> {
    (0..HIDDEN)
        .map(|output| {
            (0..HIDDEN)
                .map(|input| {
                    Fp3::from_base(fp_from_signed_word(weights[input * HIDDEN + output] as u64))
                })
                .collect()
        })
        .collect()
}

const INDEX_MASK_ORACLE: u64 = 60;
const INDEX_MASK_TREE: u64 = 202;
const INDEX_MASK_RUN: Digest = Digest([0x66; 32]);
const DOT_ORACLE: u64 = 12;
const DOT_RUN: Digest = Digest([0x67; 32]);

fn index_mask_plan(positions: u32, mask_code: CodeParams) -> IndexMaskPlan {
    IndexMaskPlan {
        oracle: INDEX_MASK_ORACLE,
        group: GROUP,
        mask_code,
        mask_rows: 4,
        answers: 1,
        openings: 1,
        tree: RuntimeTreePlan {
            tree_number: INDEX_MASK_TREE,
            leaf_class: 8,
            log_n: mask_code.log_n,
            log_n_zk: mask_code.log_n,
            parts: vec![RuntimeTreePart {
                main_symbols: 0,
                mask_symbols: 4 * E_DEGREE as u64,
            }],
            salt_bytes: SALT,
            run: INDEX_MASK_RUN,
            configuration: W_CONFIGURATION,
        },
        entries: MaskEntryAllocation::for_variables(
            MATRIX_ROWS.trailing_zeros() as usize,
            positions,
        )
        .unwrap(),
    }
}

fn dot_plan(positions: u32, code: CodeParams) -> DotMapPlan {
    DotMapPlan {
        oracle: DOT_ORACLE,
        group: GROUP,
        shape: WShape::new(
            positions as u64 * HIDDEN as u64,
            code,
            1,
            positions,
            SALT,
            400,
            2,
            DOT_RUN,
            W_CONFIGURATION,
        )
        .unwrap(),
        hidden: HIDDEN as u32,
    }
}

/// The production assembly input of the projection consumer's hidden
/// session. The own group is one test over the request's W messages, so the
/// request count is one; Omega is also opened by the shared test.
fn assembly_input(positions: u32) -> AssemblyInput {
    let mut own_objects = Vec::new();
    for _ in 0..positions {
        own_objects.push(AssemblyOwnObject {
            member: AssemblyMember::Columns {
                entries: COLUMNS,
                answers: 1,
            },
            shared_too: false,
            spent: 2,
        });
    }
    own_objects.push(AssemblyOwnObject {
        member: AssemblyMember::Fixed {
            main_rows: 1,
            mask_rows: 0,
        },
        shared_too: true,
        spent: 0,
    });
    own_objects.push(AssemblyOwnObject {
        member: AssemblyMember::Fixed {
            main_rows: 0,
            mask_rows: 4,
        },
        shared_too: false,
        spent: 1,
    });
    own_objects.push(AssemblyOwnObject {
        member: AssemblyMember::Columns {
            entries: positions as u64 * HIDDEN as u64,
            answers: 1,
        },
        shared_too: false,
        spent: 0,
    });
    AssemblyInput {
        shared_members: vec![
            AssemblyMember::Fixed {
                main_rows: 1,
                mask_rows: 0,
            },
            AssemblyMember::Fixed {
                main_rows: 1,
                mask_rows: 0,
            },
        ],
        own_objects,
        shared_mask_spent: 0,
        requests: 1,
        samples: 1,
        columns: COLUMNS,
        circle: false,
    }
}

fn stage_plan(stage: &SolvedStage) -> IoppStagePlan {
    IoppStagePlan {
        main: CodeParams {
            log_n: stage.log_n,
            ell: stage.main_ell,
            r: stage.main_r,
        },
        mask: CodeParams {
            log_n: stage.mask_log_n,
            ell: stage.mask_ell,
            r: stage.mask_r,
        },
        k: stage.k,
        h: stage.h,
        positions: stage.positions as usize,
        samples: 1,
        work_bits: 0,
    }
}

fn w_shape_with(positions: u32, code: CodeParams) -> WShape {
    WShape::new(COLUMNS, code, 1, positions, SALT, 100, 1, W_RUN, W_CONFIGURATION).unwrap()
}

/// The production hidden session plan: the trusted assembly fixes both codes,
/// the fold bits, the position counts and the BASE counts from the real member
/// layout. `planner_checked` is on, so the verifier re-derives every value.
fn hidden_plan_for(
    activation: &CommittedActivation,
    table: &Table,
    positions: u32,
) -> HiddenLinPlan {
    let assembled = assemble_proximity(&assembly_input(positions))
        .expect("the trusted assembly finds the production proximity plan");
    let code = assembled.first_main;
    let mask_code = assembled.first_mask;
    let mut plan = HiddenLinPlan {
        norm: None,
        send: HiddenSendPlan {
            activation: activation.shape(),
            activation_configuration: ACTIVATION_CONFIGURATION,
            activation_root: activation.root(),
            activation_source: zkie_ops::proof::hidden_send_session::ActivationSource::Commitment,
            table_shape: table_shape(),
            table_identity: reader_table_identity(&table.reader).unwrap(),
            matrix_rows: MATRIX_ROWS,
            k_lo: K_LO,
            columns: COLUMNS,
            group: GROUP,
            oracle_base: ORACLE_BASE,
            openings_per_oracle: 2,
            w: w_shape_with(positions, code),
            index_mask: None,
            dot: None,
            expected_dot_root: None,
            expected_roots: None,
            member: None,
            request_chain: None,
            request_reads: None,
        },
        output: HiddenOutputShape {
            positions,
            columns: COLUMNS,
        },
        masks: vec![Vec::new(); positions as usize],
        weights: Vec::new(),
        proximity: HiddenProximityPlan {
            gu: gu_shape(code, mask_code.log_n),
            omega: omega_shape(code, mask_code.log_n),
            code,
            mask_code,
            shared_mask_tree: mask_tree(
                200,
                mask_code.log_n,
                1u64 << assembled.shared[0].h,
            ),
            own_mask_tree: mask_tree(201, mask_code.log_n, 1u64 << assembled.own[0].h),
            shared_fresh: 90,
            own_fresh: 91,
            column_draws: 2,
            column_budget: 4,
            column_work_bits: 0,
            shared_base_positions: assembled.shared.last().unwrap().positions as usize,
            own_base_positions: assembled.own.last().unwrap().positions as usize,
            shared_stages: assembled.shared.iter().map(stage_plan).collect(),
            own_stages: assembled.own.iter().map(stage_plan).collect(),
            planner_checked: true,
        },
        output_relation: None,
        closure: None,
    };
    let index_plan = index_mask_plan(positions, mask_code);
    let pending_mask_slot = index_plan.entries.total().expect("mask allocation") as u64;
    plan.send.index_mask = Some(index_plan);
    plan.send.dot = Some(dot_plan(positions, code));
    // The consumer's pending claim and reduced witness. The claim's weights are
    // the first position's activation row over the matrix rows; the tables cut
    // each row into eight runs of the columns. The pending claim is not yet
    // consumed by the second chain; this run binds the witness root into the
    // first message and the session encoding.
    let row = activation
        .weights(0, MATRIX_ROWS)
        .expect("activation row");
    let pending = PendingRow {
        weights: stretch_weights(MATRIX_ROWS.trailing_zeros(), 0, &row).expect("weights"),
        psi: vec![Fp3::from_u64(3); COLUMNS.trailing_zeros() as usize],
        target: Fp3::from_u64(7),
        masks: vec![MaskTerm {
            oracle: INDEX_MASK_ORACLE,
            slot: pending_mask_slot,
            coeff: Fp3::from_u64(5),
        }],
    };
    let run_length = COLUMNS / 8;
    let entries = MATRIX_ROWS * (COLUMNS / run_length);
    plan.send.member = Some(MemberPlan {
        index_oracle: INDEX_MASK_ORACLE,
        row_bits: MATRIX_ROWS.trailing_zeros(),
        column_bits: COLUMNS.trailing_zeros(),
        pending,
        tables: vec![ReductionTable {
            matrix: 0,
            parts: 1,
            run_length,
            rows_per_part: MATRIX_ROWS,
            columns: COLUMNS,
            at: 0,
            entries,
            phi: (1..=run_length).map(Fp3::from_u64).collect(),
            later: false,
        }],
        tree: RuntimeTreePlan {
            tree_number: 700,
            leaf_class: 4,
            log_n: entries.trailing_zeros(),
            log_n_zk: entries.trailing_zeros(),
            parts: vec![RuntimeTreePart {
                main_symbols: E_DEGREE as u64,
                mask_symbols: 0,
            }],
            salt_bytes: SALT,
            run: Digest([0x71; 32]),
            configuration: Digest([0x72; 32]),
        },
    });
    plan
}

#[test]
#[ignore = "runs against the registered production trace artifacts"]
fn real_projection_consumer_closes_in_hidden_input_mode() {
    let Some(base) = artifacts() else {
        return;
    };
    let public_bytes =
        fs::read(base.join("commit/public/model-commitment.bin")).expect("public bundle");
    let public: ModelCommitmentPublic = bincode::deserialize(&public_bytes).expect("public bundle");
    let config_bytes =
        fs::read(base.join("commit/public/provider-config.bin")).expect("provider config");
    let config: zkie::provider::ProviderConfig =
        bincode::deserialize(&config_bytes).expect("provider config");
    let statement_bytes = fs::read(base.join("proof/statement.bin")).expect("statement");
    let statement: zkie::provider::StatementData =
        bincode::deserialize(&statement_bytes).expect("statement");
    assert_eq!(statement.tokens.len(), SEQUENCE_TOKENS);

    let params = zkie_models_gpt2::plan::asset_table_params().expect("table params");
    let randomness =
        fs::read(base.join("commit/private/asset-table-randomness.bin")).expect("randomness");
    let tables = decode_asset_table_randomness_set(&randomness, &params).expect("randomness set");
    let commitment = zkie_models_gpt2::executor::build_commitment(
        Path::new(&config.asset_root),
        &config.tokens,
        &tables,
    )
    .expect("commitment");
    assert_eq!(commitment.anchor.record.anchor_id, public.anchor.anchor_id);

    let trace_bytes = fs::read(base.join("proof/proof.bin")).expect("proof bytes");
    let trace: SharedTrace = bincode::deserialize(&trace_bytes).expect("trace");
    let plan = zkie_models_gpt2::plan::trace_plan(statement.tokens.len()).expect("plan");
    let outputs = verify_shared_trace_outputs(&public, &trace, &statement.tokens, &plan)
        .expect("verified trace outputs");

    let mut consumers = Vec::new();
    for (index, step) in plan.steps.iter().enumerate() {
        if step.opcode != OpCode::Projection {
            continue;
        }
        let Some(PlanSource::Asset { path, .. }) = step.inputs.get(1) else {
            continue;
        };
        if path != "gpt2/weights/L0_k_w_i32.bin" {
            continue;
        }
        let Some(PlanSource::Prev { step: activation_step, .. }) = step.inputs.first() else {
            panic!("the consumer activation is a previous step");
        };
        consumers.push((index, *activation_step));
    }
    assert_eq!(consumers.len(), HEADS);
    let activation_step = consumers[0].1;
    assert!(consumers.iter().all(|(_, step)| *step == activation_step));

    let positions = std::env::var("ZKIE_PROJECTION_DIAGNOSTIC_POSITIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16)
        .min(statement.tokens.len());
    let positions = positions as u32;

    let mut activation_rows = Vec::with_capacity(positions as usize);
    for position in 0..positions as usize {
        let row = outputs[activation_step][0]
            [position * HIDDEN..(position + 1) * HIDDEN]
            .to_vec();
        activation_rows.push(row);
    }

    let store = zkie_ops::proof::model_commitment::AssetStore::open(&commitment).expect("store");
    let weights = store
        .value("gpt2/weights/L0_k_w_i32.bin")
        .expect("weight rows");

    let mut session_table = diagnostic_table(&weights);
    let activation = activation(&activation_rows, positions);
    let hidden_plan = hidden_plan_for(&activation, &session_table, positions);
    hidden_plan
        .proximity
        .fault(&hidden_plan.send, None, None)
        .map(|reason| panic!("proximity fault: {reason}"));
    hidden_plan.validate().expect("hidden plan");

    let (messages, witness) =
        prove_hidden_lin_hidden(&hidden_plan, &activation, &mut session_table, SEED)
            .expect("hidden prove");
    let mut verifier_plan = hidden_plan.clone();
    verifier_plan.send.expected_roots = Some(witness.send.roots.clone());
    let outcome = verify_hidden_lin_hidden(&verifier_plan, session_table.reader(), &messages)
        .expect("hidden verify");
    assert!(!outcome.eval_rows.is_empty());

    // The reduced-witness root is a first-message commitment: a changed root, a
    // missing root or a different witness tree in the plan is rejected.
    let mut tampered_root = messages.clone();
    if let Some(root) = tampered_root.first.reduction_root.as_mut() {
        root.0[0] ^= 1;
    } else {
        panic!("the assembled plan carries the reduced witness");
    }
    assert!(
        verify_hidden_lin_hidden(&verifier_plan, session_table.reader(), &tampered_root).is_err()
    );
    let mut missing_root = messages.clone();
    missing_root.first.reduction_root = None;
    assert!(
        verify_hidden_lin_hidden(&verifier_plan, session_table.reader(), &missing_root).is_err()
    );
    let mut changed_tree = verifier_plan.clone();
    if let Some(member) = changed_tree.send.member.as_mut() {
        member.tree.tree_number += 1;
    }
    assert!(
        verify_hidden_lin_hidden(&changed_tree, session_table.reader(), &messages).is_err()
    );

    // The at_start mask message shares no leaf class with any other tree, its
    // mask code is the own group's, and its entries cover the pending slots
    // besides the Eval allocation.
    let mut shared_class = hidden_plan.clone();
    if let Some(index) = shared_class.send.index_mask.as_mut() {
        index.tree.leaf_class = 7;
    }
    assert!(shared_class.validate().is_err());
    let mut wrong_code = hidden_plan.clone();
    if let Some(index) = wrong_code.send.index_mask.as_mut() {
        index.mask_code.r = index.mask_code.r.saturating_add(1);
    }
    assert!(wrong_code.validate().is_err());
    let mut low_slot = hidden_plan.clone();
    if let Some(member) = low_slot.send.member.as_mut() {
        for term in &mut member.pending.masks {
            term.slot = 0;
        }
    }
    assert!(low_slot.validate().is_err());
    let mut high_slot = hidden_plan.clone();
    if let Some(member) = high_slot.send.member.as_mut() {
        for term in &mut member.pending.masks {
            term.slot = 1_000_000;
        }
    }
    assert!(high_slot.validate().is_err());

    // The session slot after the trusted table: the canonical session
    // encoding enters the verification package as session.bin, and the
    // verifier consumes it from the package, not from the in-memory proof.
    let session_bytes = messages.encode().expect("session encoding");
    if let Some(path) = std::env::var_os("ZKIE_PROJECTION_SESSION_DUMP") {
        fs::write(path, &session_bytes).expect("write session dump");
    }
    let package_dir = base.join(format!(".session-package-{}", std::process::id()));
    let _ = fs::remove_dir_all(&package_dir);
    fs::create_dir_all(package_dir.join("public")).expect("package dir");
    fs::hard_link(
        base.join("commit/public/model-commitment.bin"),
        package_dir.join("public/model-commitment.bin"),
    )
    .expect("link public commitment");
    fs::hard_link(
        base.join("proof/proof.bin"),
        package_dir.join("proof.bin"),
    )
    .expect("link trace");
    fs::write(package_dir.join("session.bin"), &session_bytes).expect("write session");
    let entry = |role: &str, rel: &str| {
        let data = fs::read(package_dir.join(rel)).expect("read package file");
        ManifestFile {
            role: role.to_string(),
            path: rel.to_string(),
            bytes: data.len() as u64,
            digest: zkie_core::artifact::digest_bytes(&data),
        }
    };
    let manifest = VerificationManifest::new(vec![
        entry("public-commitment", "public/model-commitment.bin"),
        entry("trace", "proof.bin"),
        entry("lin-session", "session.bin"),
    ])
    .expect("session package manifest");
    write_manifest(&package_dir, &manifest).expect("write session manifest");
    read_manifest(&package_dir)
        .expect("read session manifest")
        .verify(&package_dir)
        .expect("verify session manifest");
    let package_total: u64 = manifest.files.iter().map(|file| file.bytes).sum();
    println!("session_package_total_bytes={package_total}");
    println!("session_bin_bytes={}", session_bytes.len());
    // The verifier decodes the session from the package and checks the proof
    // again from those bytes alone.
    let packaged = fs::read(package_dir.join("session.bin")).expect("read session.bin");
    let decoded =
        zkie_ops::proof::hidden_send_session::HiddenLinMessages::decode(&hidden_plan, &packaged)
            .expect("decode session.bin");
    let from_package = verify_hidden_lin_hidden(&verifier_plan, session_table.reader(), &decoded)
        .expect("verify from session.bin");
    assert_eq!(from_package.seal, outcome.seal);
    assert_eq!(from_package.eval_rows.len(), positions as usize);
    let mut tampered_session = packaged.clone();
    tampered_session[0] ^= 1;
    fs::write(package_dir.join("session.bin"), &tampered_session).expect("tamper session");
    assert!(manifest.verify(&package_dir).is_err());
    let mut trailing_session = packaged.clone();
    trailing_session.push(0);
    assert!(
        zkie_ops::proof::hidden_send_session::HiddenLinMessages::decode(
            &hidden_plan,
            &trailing_session
        )
        .is_err()
    );
    fs::write(package_dir.join("session.bin"), &packaged).expect("restore session");
    manifest.verify(&package_dir).expect("restored session package");
    fs::remove_dir_all(&package_dir).expect("remove package");

    // The planner-checked stage lists are re-derived by the verifier: a
    // changed position, tail, code or BASE count no longer matches.
    let plan_fault = |plan: &HiddenLinPlan| plan.proximity.fault(&plan.send, None, None);
    let mut changed = hidden_plan.clone();
    changed.proximity.own_stages[0].positions += 1;
    assert!(plan_fault(&changed).is_some());
    let mut changed = hidden_plan.clone();
    changed.proximity.code.r += 1;
    assert!(plan_fault(&changed).is_some());
    let mut changed = hidden_plan.clone();
    changed.proximity.mask_code.r += 1;
    assert!(plan_fault(&changed).is_some());
    let mut changed = hidden_plan.clone();
    changed.proximity.own_stages[1].main.ell *= 2;
    assert!(plan_fault(&changed).is_some());
    let mut changed = hidden_plan.clone();
    changed.proximity.own_base_positions += 1;
    assert!(plan_fault(&changed).is_some());
    let mut changed = hidden_plan.clone();
    changed.proximity.shared_stages[0].k = 2;
    assert!(plan_fault(&changed).is_some());
    let mut changed = hidden_plan.clone();
    changed.proximity.planner_checked = false;
    assert!(plan_fault(&changed).is_none());

    // The hidden mode requires the sampled index-mask message.
    let mut missing_mask = hidden_plan.clone();
    missing_mask.send.index_mask = None;
    assert!(missing_mask.validate().is_err());

    // A changed table identity is rejected by the hidden prover.
    let mut changed_identity = hidden_plan.clone();
    changed_identity.send.table_identity.0[0] ^= 1;
    let mut source_table = diagnostic_table(&weights);
    assert!(
        prove_hidden_lin_hidden(&changed_identity, &activation, &mut source_table, SEED).is_err()
    );
}
