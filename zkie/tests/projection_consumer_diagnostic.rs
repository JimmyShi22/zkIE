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
    derive_chain_pending, diagnostic_chain_stack, diagnostic_hiding_entries,
    diagnostic_randomness,
    prove_first_part_segment_evidence, prove_hidden_lin, prove_hidden_lin_hidden,
    prove_hidden_lin_hidden_shared_seal, prove_hidden_lin_hidden_with_chain_segment,
    verify_first_part_segment_evidence, verify_hidden_lin, verify_hidden_lin_hidden,
    verify_hidden_lin_hidden_shared_seal, verify_hidden_lin_hidden_with_chain_segment,
    ActivationShape, CommittedActivation, DotMapPlan, HiddenLinPlan, HiddenOutputShape,
    HiddenProximityPlan, HiddenSendMessages, HiddenSendPlan, HiddenSharedMessages,
    HiddenTableProver, IndexMaskPlan, IoppStagePlan,
    MaskEntryAllocation, MemberPlan, WShape,
};
use zkie_ops::proof::psc::Weight;
use zkie_ops::proof::psc_general::{general_variables, GeneralMaskEntries, Summand};
use zkie_ops::proof::request_chain::{ChainInputs, START_HIDING_SLOT};
use zkie_ops::proof::request_reduction::{token_kept_claim, token_variables};
use zkie_ops::proof::request_stack::{StackPanel, StackReduction};
use zkie_ops::proof::request_weights::{ClaimFactor, WeightClaim, WeightGroup, WeightLayout};
use zkie_ops::proof::hidden_member::{
    stretch_weights, MaskTerm as PendingMaskTerm, PendingRow, ReductionTable,
};
use zkie_ops::proof::psc_general::MaskTerm;
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
const TABLE_R: u64 = 128;
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

/// The twelve K-projection consumers of the trusted plan and the shared
/// activation step they read: the plan's own step order, not a rebuilt list.
fn projection_consumers(
    plan: &zkie_ops::proof::plan::TracePlan,
) -> (usize, Vec<(usize, usize)>) {
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
        let Some(PlanSource::Prev { step, .. }) = step.inputs.first() else {
            panic!("the consumer activation is a previous step");
        };
        consumers.push((index, *step));
    }
    assert_eq!(consumers.len(), HEADS);
    let activation_step = consumers[0].1;
    assert!(consumers.iter().all(|(_, step)| *step == activation_step));
    (activation_step, consumers)
}

fn table_code() -> CodeParams {
    build_code_plan(TABLE_LOG_N, TABLE_ELL, TABLE_R).unwrap()
}

fn code() -> CodeParams {
    build_code_plan(CODE_LOG_N, CODE_ELL, CODE_R).unwrap()
}

/// The table's one-time distance, derived from its code and the plan's point
/// variables. The plan carries this frozen value and the verifier checks it.
fn table_distance() -> u64 {
    zkie_ops::proof::proximity_planner::recorded_table_distance(table_code(), 10, 1, 10, 1, 0)
        .expect("the table distance")
}

/// The shared linear stage's column draws, derived from the table's distance,
/// the plan's point variables and the comparison point's variables.
fn shared_lin_columns() -> u64 {
    zkie_ops::proof::proximity_planner::shared_lin_columns(
        zkie_ops::proof::proximity_planner::SharedLinShape {
            v: 10,
            k_lo: K_LO,
            instances: 1,
            rows_log: MATRIX_ROWS.trailing_zeros(),
        },
        table_code(),
        table_distance(),
        0,
        0,
    )
    .expect("the shared column count")
}

fn table_shape() -> IndexShape {
    let distance = table_distance();
    IndexShape {
        code: table_code(),
        v: 10,
        answers: 1,
        salt_bytes: SALT,
        distances: TableDistances {
            radius: 0,
            later_radius: 0,
            m_once: distance,
            m_later: distance,
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
    hiding: zkie_engine::pcs::request::read::HidingState,
    /// The table's durable column budget. The real entry always carries one,
    /// so the first part cannot silently run without a persistent count.
    budget: zkie_engine::pcs::request::read::DurableColumnBudget,
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

    fn hiding_state(
        &mut self,
    ) -> Option<&mut zkie_engine::pcs::request::read::HidingState> {
        Some(&mut self.hiding)
    }

    fn durable_column_budget(
        &mut self,
    ) -> Option<&mut zkie_engine::pcs::request::read::DurableColumnBudget> {
        Some(&mut self.budget)
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
    let ledger_dir = std::env::temp_dir().join(format!(
        "zkie-diagnostic-ledger-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = fs::remove_dir_all(&ledger_dir);
    fs::create_dir_all(&ledger_dir).unwrap();
    let count = ledger_dir.join("columns.count");
    zkie_engine::pcs::request::read::DurableColumnBudget::initialize(
        ledger_dir.clone(),
        count.clone(),
    )
    .unwrap();
    let columns = shared_lin_columns();
    let budget = zkie_engine::pcs::request::read::DurableColumnBudget::open(
        ledger_dir,
        count,
        columns,
        columns * 4,
    )
    .unwrap();
    Table {
        table,
        source: RefCell::new(source),
        hiding: zkie_engine::pcs::request::read::HidingState::new(
            reader.root,
            1,
            columns * 4,
        ),
        budget,
        reader,
    }
}


/// The number of balanced base-256 limbs the tensor needs: the largest count
/// any of its values requires. The trace outputs are public verified values,
/// so both sides derive the same count from the tensor itself.
fn tensor_limbs(rows: &[Vec<i64>]) -> u32 {
    rows.iter()
        .flatten()
        .map(|value| {
            zkie_ops::proof::packed_matrix::balanced_limb_count(*value as i128)
                .expect("the value has a balanced limb count")
        })
        .max()
        .unwrap_or(1)
}

/// The diagnostic plan's shared linear values are derived, and the column
/// count fits the table's random tail.
#[test]
fn the_diagnostic_plan_values_are_derived() {
    let distance = table_distance();
    let columns = shared_lin_columns();
    eprintln!("diagnostic plan values: distance={distance} columns={columns}");
    assert!(distance >= 3);
    assert!(columns > 0);
    assert!(columns <= table_code().r);
}

#[test]
fn the_tensor_limb_count_covers_every_value() {
    assert_eq!(tensor_limbs(&[vec![0]]), 1);
    assert_eq!(tensor_limbs(&[vec![127, -128]]), 1);
    assert_eq!(tensor_limbs(&[vec![128, -129]]), 2);
    assert_eq!(tensor_limbs(&[vec![300, -1], vec![-300, 0]]), 2);
    assert_eq!(tensor_limbs(&[vec![1 << 20]]), 3);
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

    let (activation_step, consumers) = projection_consumers(&plan);

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


#[test]
#[ignore = "runs against the registered production trace artifacts"]
fn the_committed_source_follows_the_trusted_layout() {
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
    let (activation_step, consumers) = projection_consumers(&plan);

    let positions = 2usize;
    let mut activation_rows = Vec::with_capacity(positions);
    for token in 0..positions {
        activation_rows.push(
            outputs[activation_step][0][token * HIDDEN..(token + 1) * HIDDEN].to_vec(),
        );
    }
    // The committed activation layout: logical entry (token, input) reads the
    // verified row at token * HIDDEN + input, padded to the matrix rows.
    let activation = activation(&activation_rows, positions as u32);
    for token in 0..positions {
        let row = activation
            .weights(token as u32, MATRIX_ROWS)
            .expect("activation weights");
        for input in [0usize, 1, HIDDEN - 1] {
            assert_eq!(
                row[input],
                Fp3::from_base(fp_from_signed_word(activation_rows[token][input] as u64))
            );
        }
    }

    // The committed table layout: logical weight entry (input, output) is the
    // table cell at row output and place input, i.e. the message value; the
    // file's linear index input * HIDDEN + output is not a table column.
    let store = zkie_ops::proof::model_commitment::AssetStore::open(&commitment).expect("store");
    let weights = store
        .value("gpt2/weights/L0_k_w_i32.bin")
        .expect("weight rows");
    let session_table = diagnostic_table(&weights);
    let ell = table_shape().code.ell as usize;
    let mut message = vec![Fp::ZERO; ell];
    let mut tail = vec![Fp::ZERO; table_shape().code.r as usize];
    for output in [0usize, 1, HIDDEN - 1] {
        {
            let mut source = session_table.source.borrow_mut();
            source
                .row(output as u64, &mut message, &mut tail)
                .expect("table row");
        }
        for input in [0usize, 1, HIDDEN - 1] {
            assert_eq!(
                message[input],
                fp_from_signed_word(weights[input * HIDDEN + output] as u64)
            );
        }
    }
    // The encoded column object is the same commitment the session opens.
    let columns = [0u64, 1, (HIDDEN - 1) as u64];
    let openings = {
        let mut source = session_table.source.borrow_mut();
        session_table
            .table
            .open_columns(&mut *source, &columns)
            .expect("open columns")
    };
    zkie_engine::pcs::request::table::verify_columns(&session_table.reader, &columns, &openings)
        .expect("table columns");

    // Every head's verified output is the requantized reduction over its own
    // slice, with the plan's consumer order and head width; the twelve heads
    // share one activation step.
    let shift = match plan.steps[consumers[0].0].config.get(3) {
        Some(PlanConfigValue::Literal(value)) => *value as u32,
        _ => panic!("the consumer shift is a literal"),
    };
    for (head, (step, source)) in consumers.iter().enumerate() {
        assert_eq!(*source, activation_step);
        let head_outputs = &outputs[*step][0];
        for token in 0..positions {
            for k in [0usize, 1, HEAD_DIM - 1] {
                let output = head * HEAD_DIM + k;
                let mut raw = 0i128;
                for input in 0..HIDDEN {
                    raw += activation_rows[token][input] as i128
                        * weights[input * HIDDEN + output] as i128;
                }
                assert_eq!(
                    requantize(raw, shift).expect("requantize") as i64,
                    head_outputs[token * HEAD_DIM + k],
                    "head {head} token {token} output {k}"
                );
            }
        }
    }
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
            chain_mask: None,
            request_reads: None,
            request_w: None,
            source_data: None,
            segment_mask_tree: None,
            range: None,
            digit_lookups: Vec::new(),
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
            ledger: zkie_ops::proof::hidden_send_session::ReadLedgerCell::new(4 as u64),
            column_work_bits: 0,
            shared_base_positions: 2,
            own_base_positions: 2,
            shared_stages: Vec::new(),
            own_stages: Vec::new(),
            planner_checked: false,
            lin_check: None,
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
            // The at_start index message keeps the highest leaf class of the
            // session's mask ledger; the segment's chain-mask table sits at 9.
            leaf_class: 10,
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
    // The segment's chain-mask table shares the first chain's mask namespace:
    // its block joins the own group's fresh mask space even though it carries
    // no value member.
    own_objects.push(AssemblyOwnObject {
        member: AssemblyMember::Fixed {
            main_rows: 0,
            mask_rows: 4,
        },
        shared_too: false,
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

/// The segment's chain mask table: its own oracle, tree and mask ledger.
fn chain_mask_plan(code: CodeParams, mask_code: CodeParams) -> zkie_ops::proof::request_chain::ChainMaskPlan {
    zkie_ops::proof::request_chain::ChainMaskPlan {
        oracle: ORACLE_BASE + 1_500,
        group: GROUP,
        mask_code,
        mask_rows: 4,
        entries: 4,
        openings: 4,
        tree: RuntimeTreePlan {
            tree_number: 800,
            leaf_class: 9,
            log_n: mask_code.log_n,
            log_n_zk: mask_code.log_n,
            parts: vec![RuntimeTreePart {
                main_symbols: 0,
                mask_symbols: 4 * E_DEGREE as u64,
            }],
            salt_bytes: SALT,
            run: Digest([0x71; 32]),
            configuration: W_CONFIGURATION,
        },
    }
}

/// The segment's source-cell message: one committed value message over the
/// planned source cells.
fn source_data_plan(code: CodeParams, positions: u32) -> zkie_ops::proof::hidden_send_session::RequestWPlan {
    zkie_ops::proof::hidden_send_session::RequestWPlan {
        shape: WShape::new(4, code, 1, positions, SALT, 705, 1, W_RUN, W_CONFIGURATION)
            .unwrap(),
        oracle: ORACLE_BASE + 1_510,
        group: GROUP,
        columns: 4,
    }
}

/// The request-level W message: one logical vector over the table columns.
fn request_w_plan(code: CodeParams, positions: u32) -> zkie_ops::proof::hidden_send_session::RequestWPlan {
    zkie_ops::proof::hidden_send_session::RequestWPlan {
        shape: WShape::new(COLUMNS, code, 1, positions, SALT, 600, 1, W_RUN, W_CONFIGURATION)
            .unwrap(),
        oracle: ORACLE_BASE + 1_520,
        group: GROUP,
        columns: COLUMNS,
    }
}

/// The production hidden session plan: the trusted assembly fixes both codes,
/// the fold bits, the position counts and the BASE counts from the real member
/// layout. The trusted re-derivation runs without the segment's chain-mask
/// block, whose rows still join the own fresh mask space, so the plan keeps
/// the assembly's stage lists and freezes the request parameters separately.
fn hidden_plan_for(
    activation: &CommittedActivation,
    table: &Table,
    positions: u32,
    limbs: u32,
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
            chain_mask: None,
            request_reads: None,
            request_w: None,
            source_data: None,
            segment_mask_tree: None,
            range: None,
            digit_lookups: Vec::new(),
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
            column_draws: shared_lin_columns() as usize,
            column_budget: shared_lin_columns() as usize,
            ledger: zkie_ops::proof::hidden_send_session::ReadLedgerCell::new(
                shared_lin_columns(),
            ),
            column_work_bits: 0,
            shared_base_positions: assembled.shared.last().unwrap().positions as usize,
            own_base_positions: assembled.own.last().unwrap().positions as usize,
            shared_stages: assembled.shared.iter().map(stage_plan).collect(),
            own_stages: assembled.own.iter().map(stage_plan).collect(),
            // The trusted re-derivation's model does not carry the segment's
            // chain mask block (the fixture's chain plans leave it off too);
            // the stage lists above are still the assembly's derivation of the
            // same members, and the request parameters are frozen separately.
            planner_checked: false,
            // The shared linear stage's plan check: the table's one-time
            // distance, the derived column draws and the point, column and
            // comparison-point error terms come from this plan.
            lin_check: Some(zkie_ops::proof::hidden_send_session::SharedLinCheck {
                point_work_bits: 0,
                zeta_work_bits: 0,
            }),
        },
        output_relation: None,
        closure: None,
    };
    let index_plan = index_mask_plan(positions, mask_code);
    let pending_mask_slot = index_plan.entries.total().expect("mask allocation") as u64;
    plan.send.index_mask = Some(index_plan);
    // The fresh own mask covers the index-mask block plus the segment chain
    // mask block, exactly as the fold rule composes the mask space; the
    // assembly first stage only sees the proximity members.
    plan.proximity.own_mask_tree.mask_rows = zkie_ops::proof::hidden_norm_psc::norm_own_mask_rows(
        plan.send
            .index_mask
            .as_ref()
            .unwrap()
            .tree_rows()
            .expect("index mask rows")
            + chain_mask_plan(code, mask_code).mask_rows,
        0,
    )
    .expect("own mask rows");
    plan.send.dot = Some(dot_plan(positions, code));
    // The consumer's pending claim and reduced witness. The claim's weights are
    // the first position's activation row over the matrix rows. The reduced
    // witness reduces the original computation matrix the functional reads:
    // the activation tensor, positions by hidden features, one run per table
    // cut. The pending claim is not yet consumed by the second chain; this run
    // binds the witness root into the first message and the session encoding.
    let row = activation
        .weights(0, MATRIX_ROWS)
        .expect("activation row");
    let pending = PendingRow {
        weights: stretch_weights(MATRIX_ROWS.trailing_zeros(), 0, &row).expect("weights"),
        psi: vec![Fp3::from_u64(3); COLUMNS.trailing_zeros() as usize],
        target: Fp3::from_u64(7),
        masks: vec![PendingMaskTerm {
            oracle: INDEX_MASK_ORACLE,
            slot: pending_mask_slot,
            coeff: Fp3::from_u64(5),
        }],
    };
    let run_length = HIDDEN as u64 / 8;
    let entries = positions as u64 * (HIDDEN as u64 / run_length);
    // One read table per balanced byte limb of the activation tensor, each
    // reducing the same functional over its limb's cells.
    let mut tables = Vec::with_capacity(limbs as usize);
    for limb in 0..limbs as u64 {
        tables.push(ReductionTable {
            matrix: limb,
            parts: 1,
            run_length,
            rows_per_part: positions as u64,
            columns: HIDDEN as u64,
            at: limb * entries,
            entries,
            phi: (1..=run_length).map(Fp3::from_u64).collect(),
            later: false,
        });
    }
    let total = entries * limbs as u64;
    // The range object over the tensor's limbs and its two digit lookups.
    let matrix = zkie_ops::proof::packed_matrix::PackedMatrixPlan::new(
        HIDDEN as u64,
        positions as u64,
    )
    .unwrap();
    let mut objects = Vec::with_capacity(limbs as usize);
    for limb in 0..limbs as u64 {
        let array_entries = ((HIDDEN + 2) / 3) as u64 * positions as u64;
        objects.push(zkie_ops::proof::request_chain::RangeObject {
            matrix,
            array_oracle: ORACLE_BASE + 1_100 + 2 * limb,
            array_shape: WShape::new(
                array_entries.next_power_of_two(),
                code,
                1,
                positions,
                SALT,
                910 + limb,
                6,
                W_RUN,
                W_CONFIGURATION,
            )
            .unwrap(),
        });
    }
    let mz = limbs as u64 * HIDDEN as u64;
    let unit_shape = WShape::new(
        total.next_power_of_two(),
        code,
        1,
        positions,
        SALT,
        700,
        4,
        Digest([0x71; 32]),
        Digest([0x72; 32]),
    )
    .unwrap();
    // The digit message's width is the real entry count padded to a power of
    // two; its row count follows from the width and the code's sample pad.
    let digit_width =
        zkie_ops::proof::request_chain::range_digit_columns(32, mz).unwrap();
    let digit_rows =
        zkie_ops::proof::request_chain::range_digit_rows(digit_width, code.ell, 1).unwrap();
    let digit_shape = WShape {
        code,
        main_rows: digit_rows,
        answers: 1,
        salt_bytes: SALT,
        tree_number: 940,
        leaf_class: 6,
        run: Digest([0x6B; 32]),
        configuration: W_CONFIGURATION,
    };
    plan.send.range = Some(zkie_ops::proof::request_chain::RangePlan {
        objects,
        combination: zkie_ops::proof::packed_matrix::RangeCombinationPlan::new(mz, 2, 10, 0)
            .unwrap(),
        combinations: 32,
        high_bits: 8,
        window: 1i128 << 17,
        group: GROUP,
        // The range identities sit above the per-limb message oracles and
        // below the lookup identities; every object keeps its own number.
        unit_oracle: ORACLE_BASE + 1_180,
        unit_shape,
        digit_oracle: ORACLE_BASE + 1_181,
        digit_shape,
    });
    let range = plan.send.range.as_ref().unwrap();
    let lookup_params = |plane: u64| zkie_ops::proof::digit_lookup::DigitLookupParams {
        code,
        mask_code,
        group: GROUP,
        bits_oracle: ORACLE_BASE + 1_200 + 10 * plane,
        mask_oracle: ORACLE_BASE + 1_202 + 10 * plane,
        bits_tree: 920 + 5 * plane,
        mask_tree: 923 + 5 * plane,
        leaf_class: 4,
        salt_bytes: SALT,
        run: Digest([0x6B; 32]),
        configuration: W_CONFIGURATION,
    };
    let low = zkie_ops::proof::digit_lookup::digit_lookup_plan(range, 0, &lookup_params(0))
        .unwrap();
    let high = zkie_ops::proof::digit_lookup::digit_lookup_plan(range, 1, &lookup_params(1))
        .unwrap();
    // The fresh mask covers the group's whole mask space: the chain mask's
    // block plus the two lookup mask blocks, each padded to a power of two.
    let mask_blocks = 4u64
        + low.commit.mask_rows.next_power_of_two()
        + high.commit.mask_rows.next_power_of_two();
    let segment_h = 64 - (mask_blocks - 1).leading_zeros();
    plan.send.chain_mask = Some(chain_mask_plan(code, mask_code));
    plan.send.source_data = Some(source_data_plan(code, positions));
    plan.send.segment_mask_tree = Some(MaskTreePlan {
        tree_number: 203,
        leaf_class: 3,
        log_n: mask_code.log_n,
        mask_rows: 1u64 << segment_h,
        salt_bytes: SALT,
        run: Digest([0x63; 32]),
        configuration: Digest([0x64; 32]),
    });
    plan.send.request_reads =
        Some(zkie_ops::proof::hidden_send_session::RequestReadPlan {
            left: vec![(0, 0)],
            right: vec![(1, 1)],
        });
    plan.send.request_w = Some(request_w_plan(code, positions));
    plan.send.digit_lookups = vec![low, high];
    plan.send.request_chain = Some(zkie_ops::proof::request_chain::RequestChainShape {
        token_rounds: 2,
        token_ends: 1,
        stack_rounds: 1,
        stack_ends: 3,
        merge_rounds: 10,
        merge_ends: 1,
    });
    plan.send.member = Some(MemberPlan {
        index_oracle: INDEX_MASK_ORACLE,
        row_bits: MATRIX_ROWS.trailing_zeros(),
        column_bits: COLUMNS.trailing_zeros(),
        pending,
        tables,
        tree: RuntimeTreePlan {
            tree_number: 700,
            leaf_class: 4,
            log_n: code.log_n,
            log_n_zk: code.log_n,
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

    let (activation_step, consumers) = projection_consumers(&plan);

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
    let limbs = tensor_limbs(&activation_rows);
    let hidden_plan = hidden_plan_for(&activation, &session_table, positions, limbs);
    hidden_plan
        .proximity
        .fault(&hidden_plan.send, None, None)
        .map(|reason| panic!("proximity fault: {reason}"));
    hidden_plan.validate().expect("hidden plan");

    // The request chain's segment inputs over the real table and activation:
    // the token kept proof, the two stack panels and the merge's weight claim.
    let f = |value: u64| Fp3::from_base(fp_from_u64(value));
    let chain_mask = |slot: usize| -> zkie_core::error::Result<Fp3> {
        if (20..23).contains(&slot) {
            return Ok(f(800 + (slot - 20) as u64));
        }
        if slot == 3 {
            return Ok(f(953));
        }
        if slot >= START_HIDING_SLOT {
            return Ok(f(700 + (slot - START_HIDING_SLOT) as u64));
        }
        Ok(f(950 + slot as u64))
    };
    let point = [f(2), f(3)];
    let token_shown = [f(1), f(2), f(3), f(4)];
    let token_kept = [Fp3::ZERO; 4];
    let hidden_claim = token_kept_claim(&point, &token_shown, &token_kept).unwrap();
    let factor = f(11);
    let claim_masks = [MaskTerm {
        slot: 3,
        coefficient: factor,
    }];
    let shown_claim = hidden_claim.add(&factor.mul(&chain_mask(3).unwrap()));
    let token_variables = token_variables(&point).unwrap();
    let token_hiding = diagnostic_hiding_entries(
        &general_variables(&token_variables, &Summand::ShownKept).unwrap(),
    )
    .unwrap();
    let token_masks = GeneralMaskEntries {
        hiding: (0..token_hiding).map(|index| f(700 + index as u64)).collect(),
        shown: vec![f(800)],
    };
    let om = [Fp3::ONE];
    let reduction = StackReduction::new(vec![
        StackPanel {
            claim: 0,
            ec: Fp3::ONE,
            ni: 1,
            ny: 1,
            weight: vec![Fp3::ONE],
            first_claim: None,
            w: 0,
            off: 0,
        },
        StackPanel {
            claim: 0,
            ec: Fp3::ONE,
            ni: 1,
            ny: 1,
            weight: vec![Fp3::ONE],
            first_claim: Some(0),
            w: 0,
            off: 0,
        },
    ])
    .unwrap();
    let stack_variables = (0..reduction.lam)
        .map(|_| Weight::plain(3))
        .collect::<zkie_core::error::Result<Vec<_>>>()
        .unwrap();
    let stack_hiding = diagnostic_hiding_entries(
        &general_variables(
            &stack_variables,
            &Summand::PanelProduct {
                lam: reduction.lam,
                panels: reduction.panel_ends(),
            },
        )
        .unwrap(),
    )
    .unwrap();
    let stack_masks = GeneralMaskEntries {
        hiding: (0..stack_hiding).map(|index| f(600 + index as u64)).collect(),
        shown: vec![f(800), f(801), f(802)],
    };
    let end_slots = [20usize, 21, 22];
    // The merge's weight cube is the request W's row cube: ten row bits, the
    // same count the member row declares. The one group takes every column
    // bit and leaves the rows to the matrix.
    let mut layout = WeightLayout::new(10).unwrap();
    layout
        .add(0, 0, 1, vec![WeightGroup::e("x", 1 << 20, false).unwrap()])
        .unwrap();
    let claims = vec![WeightClaim {
        unit: 0,
        factors: vec![ClaimFactor::e(vec![f(2)], vec![f(3); 20])],
    }];
    let maps = vec![vec![Fp3::ZERO; 1 << 10]];
    let merge_variables = (0..layout.lk)
        .map(|_| Weight::plain(2))
        .collect::<zkie_core::error::Result<Vec<_>>>()
        .unwrap();
    let merge_hiding = diagnostic_hiding_entries(
        &general_variables(&merge_variables, &Summand::Map).unwrap(),
    )
    .unwrap();
    let index = hidden_plan.send.index_mask.clone().unwrap();
    let index_slot = index.entries.total().unwrap();
    let randomness = diagnostic_randomness(&hidden_plan.send, &SEED).unwrap();
    let mask_rows = randomness.index_mask.as_ref().unwrap().rows.clone();
    let ell = usize::try_from(index.mask_code.ell).unwrap();
    let offset = index.entry_offset(index_slot).unwrap();
    let index_entry = mask_rows[offset / ell][offset % ell];
    let merge_masks = GeneralMaskEntries {
        hiding: (0..merge_hiding).map(|index| f(500 + index as u64)).collect(),
        shown: vec![index_entry],
    };
    let empty_cells: [Vec<Fp3>; 0] = [];
    let placeholder = ChainInputs {
        token_point: &point,
        token_shown: &token_shown,
        token_kept: &token_kept,
        token_claim_masks: &claim_masks[..],
        token_shown_claim: shown_claim,
        token_claim_slot: 3,
        token_masks: &token_masks,
        reduction: &reduction,
        stack_left: &empty_cells,
        stack_right: &empty_cells,
        om: &om,
        stack_masks: &stack_masks,
        layout: &layout,
        claims: &claims,
        claim_maps: &maps,
        end_slots: &end_slots,
        merge_masks: &merge_masks,
        mask_value: &chain_mask,
        index_oracle: index.oracle,
        index_slot: index_slot as u64,
    };
    // The request's stack reads are reconstructed from the plan's own start
    // form; the diagnostic carries no recorded request cells.
    let (stack_left, stack_right) = diagnostic_chain_stack(
        &hidden_plan,
        &activation,
        &mut session_table,
        &SEED,
        &placeholder,
    )
    .expect("chain stack");
    let inputs = ChainInputs {
        token_point: &point,
        token_shown: &token_shown,
        token_kept: &token_kept,
        token_claim_masks: &claim_masks,
        token_shown_claim: shown_claim,
        token_claim_slot: 3,
        token_masks: &token_masks,
        reduction: &reduction,
        stack_left: &stack_left,
        stack_right: &stack_right,
        om: &om,
        stack_masks: &stack_masks,
        layout: &layout,
        claims: &claims,
        claim_maps: &maps,
        end_slots: &end_slots,
        merge_masks: &merge_masks,
        mask_value: &chain_mask,
        index_oracle: index.oracle,
        index_slot: index_slot as u64,
    };

    // Discovery: the member row is the segment's own output, frozen before the
    // full run; the production verifier derives the same claim in the segment.
    let mut hidden_plan = hidden_plan;
    let _discovery = std::time::Instant::now();
    hidden_plan.send.member.as_mut().unwrap().pending =
        derive_chain_pending(&hidden_plan, &activation, &mut session_table, &SEED, &inputs)
            .expect("pending discovery");
    let discovery = _discovery.elapsed();
    hidden_plan.validate().expect("hidden plan");

    if std::env::var("ZKIE_SHARED_SEAL_EVIDENCE").is_ok() {
        // The plan check runs at the actual entry: a plan below the derived
        // column count and a plan without the one-time distance are refused.
        let mut under_met = hidden_plan.clone();
        under_met.proximity.column_draws = under_met.proximity.column_draws.saturating_sub(1);
        assert!(under_met.validate().is_err(), "the under-met column plan is refused");
        let mut no_distance = hidden_plan.clone();
        no_distance.send.table_shape.distances.m_once = 0;
        no_distance.proximity.gu.index.distances.m_once = 0;
        assert!(no_distance.validate().is_err(), "the plan without a distance is refused");
        // The first two chains (the first part and the shared stage) up to
        // their seals, on the real plan and table with the plan's derived
        // column draws and one-time distance. The second stage is outside
        // this path.
        let seal_prove_start = std::time::Instant::now();
        let (seal_first, seal_shared) = prove_hidden_lin_hidden_shared_seal(
            &hidden_plan,
            &activation,
            &mut session_table,
            SEED,
            Some(&inputs),
        )
        .expect("shared seal prove");
        let seal_prove_time = seal_prove_start.elapsed();
        let seal_bytes = seal_first.encode().expect("first bytes").len()
            + seal_shared.encode().expect("shared bytes").len();
        let seal_verify_start = std::time::Instant::now();
        verify_hidden_lin_hidden_shared_seal(
            &hidden_plan,
            session_table.reader(),
            &seal_first,
            &seal_shared,
            Some(&inputs),
        )
        .expect("shared seal verify");
        let seal_verify_time = seal_verify_start.elapsed();

        let mut attempted = 0u32;
        let mut rejected = 0u32;
        let mut check =
            |label: &str, first: HiddenSendMessages, shared: HiddenSharedMessages| {
                attempted += 1;
                match verify_hidden_lin_hidden_shared_seal(
                    &hidden_plan,
                    session_table.reader(),
                    &first,
                    &shared,
                    Some(&inputs),
                ) {
                    Ok(_) => eprintln!("SHARED SEAL EVIDENCE negative accepted: {label}"),
                    Err(_) => rejected += 1,
                }
            };
        let mut changed = seal_shared.clone();
        changed.column_openings[0].symbols[0] = Fp::from_u64(0xDEAD_BEEF);
        check("changed column opening", seal_first.clone(), changed);
        let mut changed = seal_shared.clone();
        changed.fold_rounds[0].0 = changed.fold_rounds[0].0.add(&Fp3::ONE);
        check("changed fold round", seal_first.clone(), changed);
        let mut changed = seal_shared.clone();
        changed.omega_answers[0] = changed.omega_answers[0].add(&Fp3::ONE);
        check("changed Omega answer", seal_first.clone(), changed);
        let mut changed = seal_shared.clone();
        changed.seal.0[0] ^= 1;
        check("changed shared seal", seal_first.clone(), changed);
        let mut changed = seal_first.clone();
        changed.chain_value.0[0] ^= 1;
        check("changed first seal", changed, seal_shared.clone());
        assert_eq!(rejected, attempted, "every shared seal negative is rejected");
        let hwm = fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|status| {
                status
                    .lines()
                    .find(|line| line.starts_with("VmHWM:"))
                    .map(|line| line.trim().to_string())
            })
            .unwrap_or_else(|| "VmHWM unavailable".to_string());
        eprintln!(
            "SHARED SEAL EVIDENCE (local first+shared chains, not R5): limbs={} discovery={:?} seal_bytes={} seal_prove={:?} seal_verify={:?} negatives={}/{} {}",
            limbs,
            discovery,
            seal_bytes,
            seal_prove_time,
            seal_verify_time,
            rejected,
            attempted,
            hwm,
        );
        return;
    }

    if std::env::var("ZKIE_SEGMENT_EVIDENCE").is_ok() {
        // Local segment evidence, not the full request session and not R5:
        // the first chain up to its seal carries the committed group, the
        // request segment (range, digit lookups, tie/X4, masks) and its
        // proximity test. The shared and second chains are outside this path.
        let segment_prove_start = std::time::Instant::now();
        let segment_messages = prove_first_part_segment_evidence(
            &hidden_plan,
            &activation,
            &mut session_table,
            SEED,
            &inputs,
        )
        .expect("segment prove");
        let segment_prove_time = segment_prove_start.elapsed();
        let segment_bytes = segment_messages.encode().expect("segment bytes").len();
        let segment_verify_start = std::time::Instant::now();
        verify_first_part_segment_evidence(
            &hidden_plan,
            session_table.reader(),
            &segment_messages,
            &inputs,
        )
        .expect("segment verify");
        let segment_verify_time = segment_verify_start.elapsed();

        let mut attempted = 0u32;
        let mut rejected = 0u32;
        let mut check = |label: &str, changed: HiddenSendMessages| {
            attempted += 1;
            match verify_first_part_segment_evidence(
                &hidden_plan,
                session_table.reader(),
                &changed,
                &inputs,
            ) {
                Ok(()) => eprintln!("SEGMENT EVIDENCE negative accepted: {label}"),
                Err(_) => rejected += 1,
            }
        };
        let mut changed = segment_messages.clone();
        changed.digit_lookups[0].bits_root = Digest([0x5F; 32]);
        check("changed digit lookup bits root", changed);
        let mut changed = segment_messages.clone();
        changed.digit_lookups[1].claims.multiplicity.1 =
            changed.digit_lookups[1].claims.multiplicity.1.add(&Fp3::ONE);
        check("changed digit lookup multiplicity", changed);
        let mut changed = segment_messages.clone();
        changed.range_array[0].root = Digest([0x5E; 32]);
        check("changed packed array root", changed);
        let mut changed = segment_messages.clone();
        changed
            .range_units
            .as_mut()
            .expect("range units")
            .root = Digest([0x5D; 32]);
        check("changed reduced table root", changed);
        let mut changed = segment_messages.clone();
        changed
            .range_digits
            .as_mut()
            .expect("range digits")
            .root = Digest([0x5C; 32]);
        check("changed digit object root", changed);
        let mut changed = segment_messages.clone();
        changed
            .chain_mask
            .as_mut()
            .expect("chain mask")
            .root = Digest([0x5B; 32]);
        check("changed segment mask root", changed);
        let mut changed = segment_messages.clone();
        changed
            .source_data
            .as_mut()
            .expect("source data")
            .root = Digest([0x5A; 32]);
        check("changed source cell root", changed);
        let mut changed = segment_messages.clone();
        let merge_end = changed
            .request
            .as_ref()
            .expect("request section")
            .merge
            .end_values[0]
            .add(&Fp3::ONE);
        changed
            .request
            .as_mut()
            .expect("request section")
            .merge
            .end_values[0] = merge_end;
        check("changed merge end value", changed);
        let mut changed = segment_messages.clone();
        changed
            .segment_proximity
            .as_mut()
            .expect("segment proximity")
            .seal
            .0[0] ^= 1;
        check("changed segment proximity seal", changed);
        assert_eq!(rejected, attempted, "every segment negative is rejected");
        let hwm = fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|status| {
                status
                    .lines()
                    .find(|line| line.starts_with("VmHWM:"))
                    .map(|line| line.trim().to_string())
            })
            .unwrap_or_else(|| "VmHWM unavailable".to_string());
        eprintln!(
            "SEGMENT EVIDENCE (local segment, not R5): limbs={} discovery={:?} segment_bytes={} segment_prove={:?} segment_verify={:?} negatives={}/{} {}",
            limbs,
            discovery,
            segment_bytes,
            segment_prove_time,
            segment_verify_time,
            rejected,
            attempted,
            hwm,
        );
        return;
    }

    let prove_start = std::time::Instant::now();
    let (messages, witness) = prove_hidden_lin_hidden_with_chain_segment(
        &hidden_plan,
        &activation,
        &mut session_table,
        SEED,
        &inputs,
    )
    .expect("hidden prove");
    let prove_time = prove_start.elapsed();
    let verify_start = std::time::Instant::now();
    let mut verifier_plan = hidden_plan.clone();
    verifier_plan.send.expected_roots = Some(witness.send.roots.clone());
    let outcome = verify_hidden_lin_hidden_with_chain_segment(
        &verifier_plan,
        session_table.reader(),
        &messages,
        &inputs,
    )
    .expect("hidden verify");
    let verify_time = verify_start.elapsed();
    assert!(!outcome.eval_rows.is_empty());
    let proof_bytes = messages.first.encode().expect("message bytes").len();
    eprintln!(
        "R5 evidence: limbs {} discovery {:?} prove {:?} verify {:?} bytes {}",
        limbs, discovery, prove_time, verify_time, proof_bytes
    );

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
