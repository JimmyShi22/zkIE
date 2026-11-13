use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use zkie_core::common::field::PrimeField64;
use zkie_core::types::Digest;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::lin::LinShape;
use zkie_ops::proof::lin_session::{
    build_authenticated_lin_table, fp3_from_signed_word, prove_lin_request, verify_lin_request,
    verify_lin_request_bytes, LinRequestMessages,
};
use zkie_ops::proof::model_commitment::{ModelCommitmentPublic, SharedTrace};
use zkie_ops::proof::session_baseline::{
    assemble_binding, verify_selected_step, SelectedMatmulStep, TrustedModel,
};

fn fp3(value: u64) -> Fp3 {
    fp3_from_signed_word(value)
}

fn next(value: &Fp3) -> Fp3 {
    fp3(value.c[0].as_canonical_u64().wrapping_add(1))
}

#[test]
#[ignore = "requires ZKIE_SESSION_FIXTURE with the multi-GiB gpt2-seq16 artifact"]
fn selected_gpt2_linear_claim_session() {
    let fixture = env::var("ZKIE_SESSION_FIXTURE").expect("set ZKIE_SESSION_FIXTURE");
    let root = Path::new(&fixture).join("gpt2-seq16");
    let public: ModelCommitmentPublic =
        bincode::deserialize(&fs::read(root.join("public/model-commitment.bin")).unwrap()).unwrap();
    let file = fs::File::open(root.join("proof.bin")).unwrap();
    let mut trace: SharedTrace = bincode::deserialize_from(BufReader::new(file)).unwrap();
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
    let shape = LinShape::new(1024, 0, 64, 1024).unwrap();
    let asset_path = "gpt2/weights/lm_head_w_i32.bin";
    let weights: Vec<Fp3> = trace.steps[4444].message.proof.inputs[0][0]
        .values
        .iter()
        .map(|value| fp3(*value))
        .collect();
    let w_values: Vec<Fp3> = trace.steps[4444].message.proof.outputs[0][0]
        .values
        .iter()
        .take(64)
        .map(|value| fp3(*value))
        .collect();
    let masks = Vec::new();

    let started = Instant::now();
    let prover = prove_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &weights,
        &masks,
        &w_values,
    )
    .unwrap();
    let verifier = verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &prover,
    )
    .unwrap();
    println!("lin_request_ms={}", started.elapsed().as_millis());
    assert_eq!(prover.first.psi, verifier.psi);
    assert_eq!(prover.shared.point, verifier.shared);
    assert_eq!(prover.shared.zeta, verifier.zeta);
    assert_eq!(prover.shared.first_seal, verifier.first_seal);
    assert_eq!(prover.second.shared_seal, verifier.shared_seal);
    assert_eq!(prover.second.seal, verifier.seal);
    assert_eq!(verifier.shared.len(), shape.k_col as usize);
    assert!(verifier.eval_holds && verifier.dot_holds && verifier.fold_holds);
    assert!(verifier.control_w && verifier.control_omega);

    let bytes = prover.encode();
    let decoded = LinRequestMessages::decode(&bytes).unwrap();
    assert_eq!(decoded, prover);
    let mut trailing = bytes;
    trailing.push(0);
    assert!(LinRequestMessages::decode(&trailing).is_err());
    assert!(verify_lin_request_bytes(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &trailing
    )
    .is_err());

    let mut messages = prover.clone();
    messages.first.w.values[0] = next(&messages.first.w.values[0]);
    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &messages
    )
    .is_err());

    let mut messages = prover.clone();
    messages.first.weights.values[0] = next(&messages.first.weights.values[0]);
    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &messages
    )
    .is_err());

    let mut messages = prover.clone();
    messages.shared.omega.values[0] = next(&messages.shared.omega.values[0]);
    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &messages
    )
    .is_err());

    let mut messages = prover.clone();
    messages.shared.folded.values[0] = next(&messages.shared.folded.values[0]);
    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &messages
    )
    .is_err());

    let mut messages = prover.clone();
    messages.first.psi[0] = next(&messages.first.psi[0]);
    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &messages
    )
    .is_err());

    let mut messages = prover.clone();
    messages.shared.point[0] = next(&messages.shared.point[0]);
    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &messages
    )
    .is_err());

    let mut messages = prover.clone();
    messages.shared.zeta[0] = next(&messages.shared.zeta[0]);
    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &messages
    )
    .is_err());

    let mut messages = prover.clone();
    messages.shared.first_seal.0[0] ^= 1;
    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &messages
    )
    .is_err());

    let mut messages = prover.clone();
    messages.second.shared_seal.0[0] ^= 1;
    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &messages
    )
    .is_err());

    let mut messages = prover.clone();
    messages.second.seal.0[0] ^= 1;
    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &messages
    )
    .is_err());

    let mut messages = prover.clone();
    messages.first.w.root.0[0] ^= 1;
    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &messages
    )
    .is_err());

    let original = trace.assets[192].rows[0].values[0];
    trace.assets[192].rows[0].values[0] ^= 1;
    assert!(build_authenticated_lin_table(
        &public,
        &trusted,
        asset_path,
        &trace.assets[192],
        shape
    )
    .is_err());
    trace.assets[192].rows[0].values[0] = original;

    trace.assets[192].rows[0].path.push(Digest([9; 32]));
    assert!(build_authenticated_lin_table(
        &public,
        &trusted,
        asset_path,
        &trace.assets[192],
        shape
    )
    .is_err());
    trace.assets[192].rows[0].path.pop();

    assert!(verify_lin_request(
        &public,
        &trusted,
        &binding,
        asset_path,
        &trace.assets[192],
        shape,
        &prover
    )
    .is_ok());
}

/// The output-head request over the production trace and the production
/// weight asset.
///
/// The activation `X` and the output `Y` come only from the verified trace
/// outputs; the weight table is committed from the bounded read-only asset
/// path used by the limited key chain and reaches the verifier only as a
/// committed encoded table with its public reader. One merged session covers
/// all 16 positions through the first, shared and second stages, and the
/// stage manifest binds the trace, the merged request proof and the public
/// model commitment.
#[test]
#[ignore = "requires ZKIE_SESSION_FIXTURE with the multi-GiB gpt2-seq16 artifact"]
fn selected_gpt2_output_head_request_session() {
    use zkie_core::artifact::digest_bytes;
    use zkie_core::common::field::PrimeCharacteristicRing;
    use zkie_core::error::Result;
    use zkie_engine::pcs::request::code::build_code_plan;
    use zkie_engine::pcs::request::field::{Fp, Fp3, E_DEGREE};
    use zkie_engine::pcs::request::foldeq::{reader_table_identity, OmegaShapePlan};
    use zkie_engine::pcs::request::gu::{GuOracleShape, GuShape};
    use zkie_engine::pcs::request::runtime_tree::{
        MaskTreePlan, RuntimeTreePart, RuntimeTreePlan,
    };
    use zkie_engine::pcs::request::table::{
        ColumnOpening, EncodedRowStore, IndexReader, StreamingTable, TableRowSource,
    };
    use zkie_engine::storage::encoded_store::FileRowStore;
    use zkie_ops::proof::hidden_send_session::{
        ActivationShape, CommittedActivation, DotMapPlan, HiddenLinPlan, HiddenOutputShape,
        HiddenProximityPlan, HiddenSendPlan, HiddenTableProver, IndexMaskPlan,
        IoppStagePlan, MaskEntryAllocation, ProximityTail, WShape,
    };
    use zkie_ops::proof::model_commitment::verify_shared_trace_outputs;
    use zkie_ops::proof::proximity_iopp::{proximity_iopp_plan, ProximityGroupRole};
    use zkie_ops::proof::output_head_lin::diagnostic::{
        output_head_claims, prepare_output_head_trace, prove_output_head_lin_prepared,
        verify_output_head_lin_prepared, verify_output_head_lin_request_bytes, OutputHeadTrace,
    };
    use zkie_ops::proof::output_head_lin::{
        build_output_head_hidden_plan, build_output_head_output_plan, output_head_shape,
        prove_output_head_lin_hidden, select_output_head_step, verify_output_head_lin_hidden,
        verify_output_head_lin_hidden_bytes,
        OutputClosureSetup, OutputHeadLinProof, OutputHeadLinSetup, OutputHeadTrustedInputs,
        OUTPUT_HEAD_ASSET_PATH, OUTPUT_HEAD_COLUMNS, OUTPUT_HEAD_HIDDEN, OUTPUT_HEAD_K_LO,
        OUTPUT_HEAD_POSITIONS, OUTPUT_HEAD_RANGE_BITS, OUTPUT_HEAD_SLACK_BITS,
    };
    use zkie_ops::proof::output_closure::ClosureMaskPlan;
    use zkie_ops::proof::psc::{hp_entries, psc_variables};
    use zkie_ops::proof::plan::PlanSource;
    use zkie_ops::proof::session_baseline::SESSION_PROTOCOL_IDENTITY;

    use zkie::lm_head_key::{
        build_lm_head_holder, limited_key_binding_parameters, LmHeadAssetSource,
        LmHeadKeyRandomness,
    };
    use zkie::commitment_ledger::{
        CommitmentLedger, LedgerPolicy, OpeningCounts, OpeningSource, OracleBudget, OracleKey,
        OracleKind, RequestState, Stage, StreamKey,
    };
    use zkie::lm_head_stage::{
        read_manifest, write_manifest, ManifestFile, VerificationManifest,
        VERIFICATION_MANIFEST_FILE,
    };
    use zkie::provider::StatementData;

    const SALT: usize = 40;
    const ACTIVATION_CONFIGURATION: Digest = Digest([0x31; 32]);
    const W_RUN: Digest = Digest([0x41; 32]);
    const MASK_RUN: Digest = Digest([0x63; 32]);
    const GU_RUN: Digest = Digest([0x32; 32]);
    const OMEGA_RUN: Digest = Digest([0x33; 32]);
    const SEED: Digest = Digest([0x72; 32]);
    const GROUP: u64 = 7;
    const ORACLE_BASE: u64 = 20;
    const GU_ORACLE: u64 = 10;
    const OMEGA_ORACLE: u64 = 11;
    const PROXIMITY_GROUP: u64 = 5;
    const SHARED_FRESH: u64 = 50;
    const OWN_FRESH: u64 = 1000;
    const W_TREE_NUMBER: u64 = 100;
    const SHARED_MASK_TREE: u64 = 200;
    const OWN_MASK_TREE: u64 = 201;
    const GU_TREE_NUMBER: u64 = 300;
    const OMEGA_TREE_NUMBER: u64 = 301;
    const INDEX_MASK_ORACLE: u64 = 40;
    const INDEX_MASK_TREE: u64 = 202;
    const INDEX_MASK_RUN: Digest = Digest([0x66; 32]);
    const DOT_ORACLE: u64 = 41;
    const DOT_TREE_NUMBER: u64 = 302;
    const DOT_RUN: Digest = Digest([0x67; 32]);
    const SLACK_ORACLE_BASE: u64 = 60;
    const SLACK_TREE_NUMBER: u64 = 400;
    const SLACK_RUN: Digest = Digest([0x68; 32]);
    const BITS_ORACLE_BASE: u64 = 100;
    const BITS_TREE_NUMBER: u64 = 420;
    const BITS_RUN: Digest = Digest([0x69; 32]);
    const MAP_ORACLE_BASE: u64 = 180;
    const MAP_TREE_NUMBER: u64 = 440;
    const MAP_RUN: Digest = Digest([0x6B; 32]);
    const CLOSURE_MASK_ORACLE: u64 = 140;
    const CLOSURE_MASK_TREE: u64 = 500;
    const CLOSURE_MASK_RUN: Digest = Digest([0x6A; 32]);
    /// The real compiled plan draws this many table columns in the shared
    /// stage; the merged session charges the same number from the reader.
    const COLUMN_DRAWS: usize = 603;
    /// The real compiled plan grinds this many column nonce bits.
    const COLUMN_WORK_BITS: u8 = 12;
    /// The fixed-material commit never enters this session; the test commits
    /// the same asset with its own private material and keeps the public
    /// reader it produces.
    const CAP_BYTES: u64 = 1_700_000_000;
    const WINDOW_BYTES: u64 = 1 << 20;
    const TOTAL_BYTES: u64 = 1 << 30;
    const COMMIT_PANEL_COLUMNS: u64 = 4096;
    const COMMIT_MAX_BYTES: u64 = 2 << 30;
    const STORE_MAX_BYTES: u64 = 4 << 30;

    fn ledger_seed_commitment(seed: &Digest) -> Digest {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"zkie/ledger/seed-commitment");
        bytes.extend_from_slice(&seed.0);
        digest_bytes(&bytes)
    }

    fn ledger_identity(kind: OracleKind, oracle_id: u64, tree_number: u64) -> Digest {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"zkie/ledger/oracle-identity");
        bytes.extend_from_slice(kind.label().as_bytes());
        bytes.extend_from_slice(&oracle_id.to_le_bytes());
        bytes.extend_from_slice(&tree_number.to_le_bytes());
        digest_bytes(&bytes)
    }

    fn ledger_oracle(
        kind: OracleKind,
        identity: Digest,
        oracle_id: u64,
        tree_number: u64,
        run: Digest,
        configuration: Digest,
    ) -> OracleKey {
        OracleKey::new(kind, identity, oracle_id, tree_number, run, configuration)
            .expect("oracle key")
    }

    fn ledger_stream(oracle: &OracleKey, material: Digest) -> StreamKey {
        StreamKey::new(
            oracle.kind,
            oracle.oracle_id,
            oracle.tree_number,
            oracle.run,
            oracle.configuration,
            material,
        )
        .expect("stream key")
    }

    fn ledger_material(seed_commitment: &Digest, oracle: &OracleKey) -> Digest {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"zkie/ledger/material");
        bytes.extend_from_slice(&seed_commitment.0);
        bytes.extend_from_slice(oracle.kind.label().as_bytes());
        bytes.extend_from_slice(&oracle.oracle_id.to_le_bytes());
        bytes.extend_from_slice(&oracle.tree_number.to_le_bytes());
        bytes.extend_from_slice(&oracle.run.0);
        bytes.extend_from_slice(&oracle.configuration.0);
        digest_bytes(&bytes)
    }

    fn ledger_table_run(root: &Digest) -> Digest {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"zkie/ledger/table-run");
        bytes.extend_from_slice(&root.0);
        digest_bytes(&bytes)
    }

    fn ledger_table_material(root: &Digest) -> Digest {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"zkie/ledger/table-material");
        bytes.extend_from_slice(&root.0);
        digest_bytes(&bytes)
    }

    fn ledger_star_run(own: u8) -> Digest {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"zkie/ledger/star-run");
        bytes.push(own);
        digest_bytes(&bytes)
    }

    fn ledger_unique(positions: &[u64]) -> Vec<u64> {
        let mut unique = positions.to_vec();
        unique.sort_unstable();
        unique.dedup();
        unique
    }

    fn fixed_key_randomness(shape: zkie_engine::pcs::request::table::IndexShape) -> LmHeadKeyRandomness {
        let rows = shape.rows() as usize;
        let n = shape.code.n() as usize;
        let tail_len = shape.code.r as usize;
        let pads_len = E_DEGREE * shape.answers as usize;
        let word = |seed: u64| Fp::from_u64(seed % 1021 + 1);
        LmHeadKeyRandomness {
            tails: (0..rows)
                .map(|row| {
                    (0..tail_len)
                        .map(|entry| word(row as u64 * 31 + entry as u64))
                        .collect()
                })
                .collect(),
            salts: (0..n)
                .map(|position| vec![(position as u8) ^ 0x5A; SALT])
                .collect(),
            pads: (0..rows)
                .map(|row| {
                    (0..pads_len)
                        .map(|entry| word(row as u64 * 7 + entry as u64))
                        .collect()
                })
                .collect(),
        }
    }

    /// The committed encoded table and the read-only row source that produced
    /// it. Folding and opening go back through the same source, so the
    /// prover never holds a second copy of the weight matrix.
    struct CommittedOutputHeadTable {
        table: StreamingTable,
        source: LmHeadAssetSource,
    }

    impl TableRowSource for CommittedOutputHeadTable {
        fn rows(&self) -> u64 {
            self.source.rows()
        }

        fn row(&mut self, row: u64, message: &mut [Fp], tail: &mut [Fp]) -> Result<()> {
            self.source.row(row, message, tail)
        }
    }

    impl HiddenTableProver for CommittedOutputHeadTable {
        fn reader(&self) -> &IndexReader {
            &self.table.reader
        }

        fn open_columns(&mut self, positions: &[u64]) -> Result<Vec<ColumnOpening>> {
            self.table.open_columns(&mut self.source, positions)
        }

    }

    let fixture = env::var("ZKIE_SESSION_FIXTURE").expect("set ZKIE_SESSION_FIXTURE");
    let root = Path::new(&fixture).join("gpt2-seq16");
    let public: ModelCommitmentPublic =
        bincode::deserialize(&fs::read(root.join("public/model-commitment.bin")).unwrap()).unwrap();
    let file = fs::File::open(root.join("proof.bin")).unwrap();
    let trace_bytes = file.metadata().unwrap().len();
    let trace: SharedTrace = bincode::deserialize_from(BufReader::new(file)).unwrap();
    let plan = zkie_models_gpt2::plan::trace_plan(16).unwrap();
    let step = select_output_head_step(&plan).unwrap();
    let shape = output_head_shape().unwrap();

    let policy = zkie::policy::parse(&fs::read(root.join("policy.toml")).unwrap()).unwrap();
    let trusted_request =
        zkie::policy::select(&policy, "gpt2-seq16", "gpt2", trace.tokens.len()).unwrap();

    // The trusted context and the real limited commit over the read-only
    // weight asset. The configuration is the plan's, not a proof value.
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("repository root")
        .to_path_buf();
    let case_path = repo.join("models/gpt2/case.toml");
    let statement = StatementData {
        case_id: "gpt2-seq16".to_string(),
        model: "gpt2".to_string(),
        model_commitment: trusted_request.model_id,
        tokens: trace.tokens.clone(),
        expected_argmax: Vec::new(),
        expected_output: Vec::new(),
    };
    let work_dir = env::temp_dir().join(format!(
        "zkie-output-head-session-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&work_dir);
    fs::create_dir_all(&work_dir).unwrap();
    let statement_path = work_dir.join("statement.bin");
    fs::write(&statement_path, bincode::serialize(&statement).unwrap()).unwrap();
    let context = zkie::s7_context::load_gpt2_lm_head_context(
        &case_path,
        &root.join("policy.toml"),
        &root.join("public"),
        &statement_path,
        trusted_request.model_id,
    )
    .unwrap();
    let request_plan = zkie_models_gpt2::lm_head_plan::lm_head_plan().unwrap();
    let table_shape = request_plan.table_shape().unwrap();
    assert_eq!(table_shape, context.limited_table_identity.shape().unwrap());
    let parameters =
        limited_key_binding_parameters(&request_plan, context.plan_identifier).unwrap();
    let configuration = parameters.digest().unwrap();
    assert_eq!(
        configuration,
        context.limited_table_identity.table_configuration
    );
    let key = fixed_key_randomness(table_shape);

    let started = Instant::now();
    let store_dir = work_dir.join("commit-store");
    fs::create_dir_all(&store_dir).unwrap();
    let mut session_seed = [0u8; 32];
    session_seed[..4].copy_from_slice(&std::process::id().to_le_bytes());
    session_seed[8..16].copy_from_slice(
        &(std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64)
            .to_le_bytes(),
    );
    let mut store = FileRowStore::new(
        &store_dir,
        Digest(session_seed),
        table_shape.rows(),
        table_shape.code.n(),
        configuration,
        STORE_MAX_BYTES,
    )
    .unwrap();
    let holder = build_lm_head_holder(
        &context,
        &request_plan,
        configuration,
        &key,
        &mut store,
        COMMIT_PANEL_COLUMNS,
        COMMIT_MAX_BYTES,
        CAP_BYTES,
        WINDOW_BYTES,
        TOTAL_BYTES,
    )
    .unwrap();
    let reader = holder.reader().clone();
    let table_name = holder.table().name.clone();
    let table_leaves = holder.table().leaves().to_vec();
    let table_salts = holder.table().salts().to_vec();
    let table_peak = holder.table().peak_bytes();
    drop(holder);
    store.cleanup().unwrap();
    let committed = StreamingTable::rebuild(
        reader.clone(),
        table_name,
        table_leaves,
        table_salts,
        table_peak,
    )
    .unwrap();
    let source = LmHeadAssetSource::open(
        &context,
        &request_plan,
        configuration,
        &key,
        CAP_BYTES,
        WINDOW_BYTES,
        TOTAL_BYTES,
    )
    .unwrap();
    let mut table = CommittedOutputHeadTable {
        table: committed,
        source,
    };
    let table_commit_ms = started.elapsed().as_millis();
    println!("table_root={:?}", reader.root);
    println!("table_configuration={:?}", reader.configuration);
    println!("table_record_digest={:?}", digest_bytes(&reader.record.encode()));
    println!("table_commit_ms={table_commit_ms}");

    // The merged three-stage plan of the request. The table identity, shape
    // and record are the committed reader's; the proximity members are plan
    // values with one shared main code, as the merged verifier requires.
    let code = build_code_plan(17, OUTPUT_HEAD_COLUMNS as u64, 8192).unwrap();
    let mask_code = build_code_plan(17, 512, 256).unwrap();
    let tree = |tree_number: u64, run: Digest, main_rows: u64| RuntimeTreePlan {
        tree_number,
        leaf_class: 2,
        log_n: code.log_n,
        log_n_zk: mask_code.log_n,
        parts: vec![RuntimeTreePart {
            main_symbols: main_rows * E_DEGREE as u64,
            mask_symbols: 0,
        }],
        salt_bytes: SALT,
        run,
        configuration,
    };
    let mask_tree = |tree_number: u64| MaskTreePlan {
        tree_number,
        leaf_class: 3,
        log_n: mask_code.log_n,
        mask_rows: 1,
        salt_bytes: SALT,
        run: MASK_RUN,
        configuration,
    };
    let w = WShape::new(
        OUTPUT_HEAD_COLUMNS as u64,
        code,
        1,
        OUTPUT_HEAD_POSITIONS as u32,
        SALT,
        W_TREE_NUMBER,
        1,
        W_RUN,
        configuration,
    )
    .unwrap();
    let main_rows = w.main_rows;
    let proximity = HiddenProximityPlan {
        gu: GuShape {
            oracle: GU_ORACLE,
            group: PROXIMITY_GROUP,
            tree: tree(GU_TREE_NUMBER, GU_RUN, main_rows),
            part: 0,
            rows_high: true,
            u: GuOracleShape {
                code,
                main_rows,
                main_deg: E_DEGREE as u32,
                mask_rows: 0,
                mask_code: None,
                answers: 1,
            },
            index: table_shape,
        },
        omega: OmegaShapePlan {
            oracle: OMEGA_ORACLE,
            group: PROXIMITY_GROUP,
            // The second stage carries the Omega witness as one message row.
            tree: tree(OMEGA_TREE_NUMBER, OMEGA_RUN, 1),
            u: GuOracleShape {
                code,
                main_rows: 1,
                main_deg: E_DEGREE as u32,
                mask_rows: 0,
                mask_code: None,
                answers: 1,
            },
            rows_high: true,
            matrix_rows: OUTPUT_HEAD_HIDDEN as u64,
            k_lo: OUTPUT_HEAD_K_LO,
        },
        code,
        mask_code,
        shared_mask_tree: mask_tree(SHARED_MASK_TREE),
        own_mask_tree: MaskTreePlan {
            mask_rows: 4,
            ..mask_tree(OWN_MASK_TREE)
        },
        shared_fresh: SHARED_FRESH,
        own_fresh: OWN_FRESH,
        column_draws: COLUMN_DRAWS,
        column_budget: COLUMN_DRAWS,
        column_work_bits: COLUMN_WORK_BITS,
        shared_base_positions: 2,
        own_base_positions: 259,
        shared_stages: vec![
            IoppStagePlan {
                main: code,
                mask: mask_code,
                k: 2,
                h: 0,
                positions: 2,
                samples: 1,
                work_bits: 0,
            },
            IoppStagePlan {
                main: build_code_plan(17, OUTPUT_HEAD_COLUMNS as u64, 256).unwrap(),
                mask: build_code_plan(17, 16_384, 256).unwrap(),
                k: 0,
                h: 0,
                positions: 2,
                samples: 1,
                work_bits: 0,
            },
        ],
        own_stages: {
            let mut stages = vec![
                IoppStagePlan {
                    main: code,
                    mask: mask_code,
                    k: 14,
                    h: 2,
                    positions: 252,
                    samples: 1,
                    work_bits: 24,
                },
                IoppStagePlan {
                    main: build_code_plan(17, 65_536, 259).unwrap(),
                    mask: build_code_plan(17, 16_384, 259).unwrap(),
                    k: 0,
                    h: 0,
                    positions: 259,
                    samples: 1,
                    work_bits: 0,
                },
            ];
            if env::var_os("ZKIE_SESSION_THREE_STAGE").is_some() {
                stages.insert(
                    1,
                    IoppStagePlan {
                        main: build_code_plan(17, 8, 256).unwrap(),
                        mask: build_code_plan(17, 16, 256).unwrap(),
                        k: 13,
                        h: 13,
                        positions: 252,
                        samples: 1,
                        work_bits: 24,
                    },
                );
                stages[2].main = build_code_plan(17, 8, 259).unwrap();
                stages[2].mask = build_code_plan(17, 1_024, 259).unwrap();
            }
            stages
        },
        planner_checked: env::var_os("ZKIE_SESSION_THREE_STAGE").is_none(),
    };
    let activation_salts: Vec<Vec<u8>> = (0..OUTPUT_HEAD_POSITIONS)
        .map(|position| vec![(position as u8).wrapping_mul(11) ^ 0x3C; SALT])
        .collect();
    let activation = ActivationShape::new(
        OUTPUT_HEAD_POSITIONS as u32,
        OUTPUT_HEAD_HIDDEN as u32,
        SALT,
    )
    .unwrap();
    let index_mask = IndexMaskPlan {
        oracle: INDEX_MASK_ORACLE,
        group: GROUP,
        mask_code,
        mask_rows: 2,
        answers: 1,
        openings: 2,
        tree: RuntimeTreePlan {
            tree_number: INDEX_MASK_TREE,
            leaf_class: 2,
            log_n: mask_code.log_n,
            log_n_zk: mask_code.log_n,
            parts: vec![RuntimeTreePart {
                main_symbols: 0,
                mask_symbols: 2 * E_DEGREE as u64,
            }],
            salt_bytes: SALT,
            run: INDEX_MASK_RUN,
            configuration,
        },
        entries: MaskEntryAllocation::new(OUTPUT_HEAD_POSITIONS as u32).unwrap(),
    };
    let closure_width = 2 * (OUTPUT_HEAD_RANGE_BITS + 1) + OUTPUT_HEAD_SLACK_BITS;
    let bits_columns = (OUTPUT_HEAD_COLUMNS as u64)
        .checked_mul(closure_width as u64)
        .unwrap()
        .next_power_of_two();
    let boolean_variables = bits_columns.trailing_zeros() as usize;
    let closure_slack = WShape::new(
        OUTPUT_HEAD_COLUMNS as u64,
        code,
        1,
        OUTPUT_HEAD_POSITIONS as u32,
        SALT,
        SLACK_TREE_NUMBER,
        2,
        SLACK_RUN,
        configuration,
    )
    .unwrap();
    let closure_bits = WShape::new(
        bits_columns,
        code,
        1,
        OUTPUT_HEAD_POSITIONS as u32,
        SALT,
        BITS_TREE_NUMBER,
        2,
        BITS_RUN,
        configuration,
    )
    .unwrap();
    let closure_variables = psc_variables(boolean_variables).unwrap();
    let closure_mask = ClosureMaskPlan {
        oracle: CLOSURE_MASK_ORACLE,
        group: GROUP,
        mask_code,
        mask_rows: 2,
        answers: 1,
        openings: 2,
        tree: RuntimeTreePlan {
            tree_number: CLOSURE_MASK_TREE,
            leaf_class: 3,
            log_n: mask_code.log_n,
            log_n_zk: mask_code.log_n,
            parts: vec![RuntimeTreePart {
                main_symbols: 0,
                mask_symbols: 2 * E_DEGREE as u64,
            }],
            salt_bytes: SALT,
            run: CLOSURE_MASK_RUN,
            configuration,
        },
        positions: OUTPUT_HEAD_POSITIONS as u32,
        hiding: hp_entries(&closure_variables).unwrap(),
        shown: 5,
    };
    let closure_setup = OutputClosureSetup {
        slack: closure_slack,
        slack_oracle_base: SLACK_ORACLE_BASE,
        slack_openings: 2,
        bits: closure_bits,
        bits_oracle_base: BITS_ORACLE_BASE,
        bits_openings: 2,
        map: WShape::new(
            bits_columns,
            code,
            1,
            OUTPUT_HEAD_POSITIONS as u32,
            SALT,
            MAP_TREE_NUMBER,
            2,
            MAP_RUN,
            configuration,
        )
        .unwrap(),
        map_oracle_base: MAP_ORACLE_BASE,
        map_openings: 2,
        mask: closure_mask,
    };
    let setup = OutputHeadLinSetup {
        asset_path: OUTPUT_HEAD_ASSET_PATH.to_string(),
        send: HiddenSendPlan {
            activation,
            activation_configuration: ACTIVATION_CONFIGURATION,
            activation_root: Digest([0x71; 32]),
            activation_source:
                zkie_ops::proof::hidden_send_session::ActivationSource::Commitment,
            table_shape,
            table_identity: reader_table_identity(&reader).unwrap(),
            matrix_rows: OUTPUT_HEAD_HIDDEN as u64,
            k_lo: OUTPUT_HEAD_K_LO,
            columns: OUTPUT_HEAD_COLUMNS as u64,
            group: GROUP,
            oracle_base: ORACLE_BASE,
            openings_per_oracle: 2,
            w,
            index_mask: None,
            dot: Some(DotMapPlan {
                oracle: DOT_ORACLE,
                group: GROUP,
                shape: WShape::new(
                    (OUTPUT_HEAD_POSITIONS * OUTPUT_HEAD_HIDDEN) as u64,
                    code,
                    1,
                    OUTPUT_HEAD_POSITIONS as u32,
                    SALT,
                    DOT_TREE_NUMBER,
                    2,
                    DOT_RUN,
                    configuration,
                )
                .unwrap(),
                hidden: OUTPUT_HEAD_HIDDEN as u32,
            }),
            expected_dot_root: None,
            expected_roots: None,
        },
        proximity,
        index_mask: Some(index_mask),
        activation_configuration: ACTIVATION_CONFIGURATION,
        activation_salts,
        seed: SEED,
        masks: Vec::new(),
        expected_activation_root: None,
        expected_roots: None,
        closure: Some(closure_setup),
    };

    // X and Y come only from the verified trace outputs.
    let _profile_trace = zkie_engine::profile::Scope::new("harness.trace_verify");
    let started = Instant::now();
    let outputs = verify_shared_trace_outputs(&public, &trace, &trace.tokens, &plan).unwrap();
    println!("trace_verify_ms={}", started.elapsed().as_millis());
    drop(_profile_trace);
    let request = prepare_output_head_trace(&public, &plan, &trace, &outputs, shape).unwrap();
    let claims = output_head_claims(shape, &request.inputs, &[]).unwrap();
    assert_eq!(request.inputs.activation.len(), OUTPUT_HEAD_POSITIONS * OUTPUT_HEAD_HIDDEN);
    assert_eq!(request.inputs.output.len(), OUTPUT_HEAD_POSITIONS * OUTPUT_HEAD_COLUMNS);
    assert_eq!(claims.weights.len(), OUTPUT_HEAD_POSITIONS);
    assert_eq!(claims.output_values.len(), OUTPUT_HEAD_POSITIONS);
    assert!(claims
        .output_values
        .iter()
        .all(|row| row.len() == OUTPUT_HEAD_COLUMNS));
    // The operator pins the independent trusted activation commitment through
    // the policy channel. The commitment is built from the verified trace
    // activation values; the prover does not supply this policy entry.
    let activation_rows: Vec<Vec<i64>> = (0..OUTPUT_HEAD_POSITIONS)
        .map(|position| {
            request.inputs.activation
                [position * OUTPUT_HEAD_HIDDEN..(position + 1) * OUTPUT_HEAD_HIDDEN]
                .to_vec()
        })
        .collect();
    let activation_commitment = CommittedActivation::commit_signed(
        ActivationShape::new(
            OUTPUT_HEAD_POSITIONS as u32,
            OUTPUT_HEAD_HIDDEN as u32,
            SALT,
        )
        .unwrap(),
        &activation_rows,
        &setup.activation_salts,
        ACTIVATION_CONFIGURATION,
    )
    .unwrap();
    let activation_identity = zkie::policy::ActivationCommitmentIdentity {
        version: zkie::policy::ACTIVATION_COMMITMENT_IDENTITY_VERSION,
        model: "gpt2".into(),
        model_version: 1,
        case_id: "gpt2-seq16".into(),
        anchor_id: trusted_request.anchor_id,
        model_id: trusted_request.model_id,
        activation_root: activation_commitment.root(),
        positions: OUTPUT_HEAD_POSITIONS as u32,
        hidden: OUTPUT_HEAD_HIDDEN as u32,
        salt_bytes: SALT as u64,
        configuration: ACTIVATION_CONFIGURATION,
        protocol_identity: SESSION_PROTOCOL_IDENTITY,
    };
    let bound_policy =
        zkie::policy::bind_activation_commitment(&policy, "gpt2-seq16", activation_identity)
            .unwrap();
    let bound_request =
        zkie::policy::select(&bound_policy, "gpt2-seq16", "gpt2", trace.tokens.len()).unwrap();
    assert_eq!(
        bound_request
            .trusted_activation_commitment()
            .unwrap()
            .commitment()
            .unwrap(),
        *activation_commitment.commitment()
    );
    drop(outputs);

    // The diagnostic path keeps its own proximity basis: the closure mask is
    // not part of the diagnostic proof, so its fresh-mask table stays at the
    // pre-closure row count.
    let mut diagnostic_setup = setup.clone();
    diagnostic_setup.proximity.own_mask_tree.mask_rows = 2;
    let _profile_prove = zkie_engine::profile::Scope::new("harness.diagnostic_prove");
    let started = Instant::now();
    let (proof, witness) =
        prove_output_head_lin_prepared(&request, shape, &diagnostic_setup, &mut table).unwrap();
    let prove_ms = started.elapsed().as_millis();
    drop(_profile_prove);
    let bytes = proof.encode().unwrap();
    assert_eq!(bytes.len(), proof.encoded_len().unwrap());
    println!("output_head_step={}", step.step);
    println!("request_proof_bytes={}", bytes.len());
    println!("prove_ms={prove_ms}");

    let mut verifier_setup = diagnostic_setup.clone();
    verifier_setup.expected_activation_root = Some(proof.session.first.activation.root);
    verifier_setup.expected_roots = Some(witness.send.roots.clone());

    let _profile_prepared = zkie_engine::profile::Scope::new("harness.diagnostic_verify_prepared");
    let started = Instant::now();
    let prepared = verify_output_head_lin_prepared(
        &request,
        shape,
        &verifier_setup,
        &table.reader(),
        &proof,
    )
    .unwrap();
    println!("prepared_verify_ms={}", started.elapsed().as_millis());
    drop(_profile_prepared);
    assert_eq!(prepared.proof_bytes, bytes.len());
    assert_eq!(prepared.outcome.eval_rows.len(), OUTPUT_HEAD_POSITIONS);
    assert!(prepared
        .outcome
        .shared_base
        .single()
        .expect("the diagnostic shared group ends in BASE")
        .fold_main);
    assert!(prepared
        .outcome
        .own_base
        .single()
        .expect("the diagnostic own group ends in BASE")
        .fold_main);

    // The byte-level entry reruns the trace check and decodes the canonical
    // request encoding against the same merged plan.
    let _profile_bytes = zkie_engine::profile::Scope::new("harness.diagnostic_verify_bytes");
    let started = Instant::now();
    let outcome = verify_output_head_lin_request_bytes(
        &public,
        &plan,
        &trace,
        &trace.tokens,
        shape,
        &verifier_setup,
        &table.reader(),
        &bytes,
    )
    .unwrap();
    let verify_ms = started.elapsed().as_millis();
    println!("verify_ms={verify_ms}");
    drop(_profile_bytes);
    assert_eq!(outcome.proof_bytes, bytes.len());
    assert_eq!(outcome.outcome.eval_rows.len(), OUTPUT_HEAD_POSITIONS);
    assert_eq!(outcome.outcome.first_seal, prepared.outcome.first_seal);
    assert_eq!(outcome.outcome.shared_seal, prepared.outcome.shared_seal);
    assert_eq!(outcome.outcome.seal, prepared.outcome.seal);
    assert_ne!(outcome.outcome.first_seal, outcome.outcome.shared_seal);
    assert_ne!(outcome.outcome.shared_seal, outcome.outcome.seal);
    assert_eq!(prepared.outcome.first_seal, proof.session.first_seal);
    assert_eq!(prepared.outcome.shared_seal, proof.session.shared.seal);
    assert_eq!(prepared.outcome.seal, proof.session.second.seal);

    // The versioned manifest binds every file the verifier reads. The trace
    // remains a required input; the request proof is reported separately from
    // the full verification bytes.
    let fixture_dir = root.clone();
    let request_rel = "output-head-request.bin";
    let request_path = fixture_dir.join(request_rel);
    fs::write(&request_path, &bytes).expect("write request");
    let public_rel = "public/model-commitment.bin";
    let public_bytes = fs::metadata(fixture_dir.join(public_rel)).unwrap().len();
    let entry = |role: &str, rel: &str| {
        let data = fs::read(fixture_dir.join(rel)).expect("read manifest file");
        ManifestFile {
            role: role.to_string(),
            path: rel.to_string(),
            bytes: data.len() as u64,
            digest: digest_bytes(&data),
        }
    };
    let manifest = VerificationManifest::new(vec![
        entry("public-commitment", public_rel),
        entry("trace", "proof.bin"),
        entry("lin-request", request_rel),
    ])
    .expect("manifest");
    let _ = fs::remove_file(fixture_dir.join(VERIFICATION_MANIFEST_FILE));
    write_manifest(&fixture_dir, &manifest).expect("write manifest");
    read_manifest(&fixture_dir)
        .expect("read manifest")
        .verify(&fixture_dir)
        .expect("verify manifest");
    let manifest_total: u64 = manifest.files.iter().map(|file| file.bytes).sum();
    assert_eq!(manifest_total, trace_bytes + bytes.len() as u64 + public_bytes);
    println!("manifest_total_bytes={manifest_total}");
    println!("manifest_request_bytes={}", bytes.len());
    println!(
        "total_verification_bytes={}",
        trace_bytes + bytes.len() as u64
    );

    // Every file the manifest names is checked, and a missing trace stays a
    // missing input even when the other files are intact. The links are hard
    // links, so the production fixture keeps its own trace.
    // Hard links cannot cross filesystems, so the link set lives beside the
    // fixture on the same device.
    let link_dir = fixture_dir.join(format!(".manifest-links-{}", std::process::id()));
    fs::create_dir_all(link_dir.join("public")).unwrap();
    fs::hard_link(fixture_dir.join("proof.bin"), link_dir.join("proof.bin")).unwrap();
    fs::hard_link(
        fixture_dir.join(public_rel),
        link_dir.join(public_rel),
    )
    .unwrap();
    fs::hard_link(&request_path, link_dir.join(request_rel)).unwrap();
    manifest.verify(&link_dir).expect("linked manifest");
    fs::remove_file(link_dir.join("proof.bin")).unwrap();
    assert!(manifest.verify(&link_dir).is_err());

    let mut tampered = bytes.clone();
    tampered[0] ^= 1;
    fs::write(&request_path, &tampered).expect("tamper request");
    assert!(manifest.verify(&fixture_dir).is_err());
    fs::write(&request_path, &bytes).expect("restore request");

    // The canonical request encoding is checked with the same merged plan the
    // verifier builds: a missing stage, a truncated tail and a trailing byte
    // are all rejected.
    let mut decode_send = verifier_setup.send.clone();
    decode_send.activation = proof.session.first.activation.shape;
    decode_send.activation_configuration = proof.session.first.activation.configuration;
    decode_send.activation_root = proof.session.first.activation.root;
    decode_send.expected_roots = verifier_setup.expected_roots.clone();
    decode_send.dot = None;
    decode_send.expected_dot_root = None;
    let mut decode_proximity = verifier_setup.proximity.clone();
    decode_proximity.shared_stages.clear();
    decode_proximity.own_stages.clear();
    let merged = HiddenLinPlan {
        norm: None,
        send: decode_send,
        output: HiddenOutputShape {
            positions: OUTPUT_HEAD_POSITIONS as u32,
            columns: OUTPUT_HEAD_COLUMNS as u64,
        },
        weights: claims.weights,
        masks: claims.masks,
        proximity: decode_proximity,
        output_relation: None,
        closure: None,
    };
    let decoded = OutputHeadLinProof::decode(&merged, &bytes).unwrap();
    assert_eq!(decoded, proof);
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(OutputHeadLinProof::decode(&merged, &trailing).is_err());
    let mut truncated = bytes.clone();
    truncated.truncate(bytes.len() - 1);
    assert!(OutputHeadLinProof::decode(&merged, &truncated).is_err());

    let reject = |request: &OutputHeadTrace,
                  setup: &OutputHeadLinSetup,
                  reader: &IndexReader,
                  proof: &OutputHeadLinProof| {
        verify_output_head_lin_prepared(request, shape, setup, reader, proof).is_err()
    };

    // Changed X or Y.
    let mut changed = request.clone();
    changed.inputs.activation[0] ^= 1;
    assert!(reject(&changed, &verifier_setup, &table.reader(), &proof));
    let mut changed = request.clone();
    changed.inputs.output[0] ^= 1;
    assert!(reject(&changed, &verifier_setup, &table.reader(), &proof));

    // Changed committed table root or record.
    let mut changed_reader = table.reader().clone();
    changed_reader.root.0[0] ^= 1;
    assert!(reject(&request, &verifier_setup, &changed_reader, &proof));
    let mut changed_reader = table.reader().clone();
    changed_reader.record.y[0] = changed_reader.record.y[0].add(&Fp3::from_u64(1));
    assert!(reject(&request, &verifier_setup, &changed_reader, &proof));

    // Changed plan: the output-head step is selected from the trusted plan.
    let mut changed_plan = plan.clone();
    changed_plan.steps[step.step].inputs[1] = PlanSource::asset("gpt2/weights/missing_head_i32.bin");
    assert!(select_output_head_step(&changed_plan).is_err());

    // Changed statement and a missing stage.
    let mut changed_setup = verifier_setup.clone();
    changed_setup.send.group ^= 1;
    assert!(reject(&request, &changed_setup, &table.reader(), &proof));
    let mut tampered_proof = proof.clone();
    tampered_proof.session.second.fold_rounds.clear();
    assert!(reject(
        &request,
        &verifier_setup,
        &table.reader(),
        &tampered_proof
    ));

    // Changed drawn position and column nonce.
    let mut tampered_proof = proof.clone();
    tampered_proof.session.first.positions[0][0] ^= 1;
    assert!(reject(
        &request,
        &verifier_setup,
        &table.reader(),
        &tampered_proof
    ));
    let mut tampered_proof = proof.clone();
    tampered_proof.session.shared.column_nonce ^= 1;
    assert!(reject(
        &request,
        &verifier_setup,
        &table.reader(),
        &tampered_proof
    ));

    // The unaltered proof still verifies after every negative.
    assert!(verify_output_head_lin_prepared(
        &request,
        shape,
        &verifier_setup,
        &table.reader(),
        &proof
    )
    .is_ok());

    // The formal hidden-input closure on the same real fixture. The verifier
    // receives the policy-pinned activation commitment and the trusted output
    // relation; neither plaintext X nor plaintext Y enters this path. This
    // section re-reads the full weight asset and is enabled explicitly so the
    // existing diagnostic fixture run keeps its own measurement basis.
    if env::var_os("ZKIE_SESSION_FORMAL").is_some() {
        // Close the pre-formal profile window (table commit and the
        // diagnostic prove), then profile the formal hidden path alone.
        zkie_engine::profile::dump();
        zkie_engine::profile::reset();
        // The formal section runs on a fresh read-only source, as a separate
        // prover process would. The diagnostic section above has already
        // consumed the bounded read budget of the shared source.
        let _profile_formal_open =
            zkie_engine::profile::Scope::new("harness.formal_table_open");
        table.source = LmHeadAssetSource::open(
            &context,
            &request_plan,
            configuration,
            &key,
            CAP_BYTES,
            WINDOW_BYTES,
            TOTAL_BYTES,
        )
        .unwrap();
        drop(_profile_formal_open);
        let output_plan = build_output_head_output_plan(&plan, &trace.argmax).unwrap();
        let trusted = OutputHeadTrustedInputs {
            activation: *activation_commitment.commitment(),
            output: output_plan.clone(),
        };
        // The authoritative opening ledger reserves the generation stream and
        // every oracle material before the hidden proof starts. The actual
        // roots are proof values and are bound as soon as the prover returns
        // them; success is recorded only after the verifier accepts.
        let ledger_dir = work_dir.join("commitment-ledger");
        let ledger = CommitmentLedger::initialize(&ledger_dir, LedgerPolicy::default())
            .expect("initialize commitment ledger");
        let request_id = digest_bytes(b"formal-output-head-request");
        let seed_commitment = ledger_seed_commitment(&SEED);
        ledger
            .begin_request(request_id, seed_commitment, setup.send.parameters_digest())
            .expect("begin the ledger request");
        let material = |oracle: &OracleKey| ledger_material(&seed_commitment, oracle);
        let reserve = |oracle: OracleKey, budget: OracleBudget| {
            ledger
                .reserve_oracle(
                    request_id,
                    oracle,
                    ledger_stream(&oracle, material(&oracle)),
                    budget,
                )
                .expect("reserve an oracle material stream");
        };

        let own_ledger_positions = setup
            .proximity
            .own_stages
            .first()
            .map(|stage| stage.positions)
            .unwrap_or(setup.proximity.own_base_positions);
        let shared_ledger_positions = setup
            .proximity
            .shared_stages
            .first()
            .map(|stage| stage.positions)
            .unwrap_or(setup.proximity.shared_base_positions);
        // The loop's switch, stage-FOLD and BASE-star trees are reserved under
        // the plan-derived keys before any proof work. Their actual openings
        // are recorded after the prover/verifier returns.
        let ledger_plan =
            build_output_head_hidden_plan(shape, &setup, &table.reader(), &trusted)
                .expect("build the ledger plan");
        let own_iopp = proximity_iopp_plan(
            &ledger_plan.proximity,
            &ledger_plan.send,
            ledger_plan.closure.as_ref(),
            ledger_plan.norm.as_ref(),
            ProximityGroupRole::Own,
        )
        .expect("derive the own loop chain")
        .expect("the own group has stages");
        let shared_iopp = proximity_iopp_plan(
            &ledger_plan.proximity,
            &ledger_plan.send,
            ledger_plan.closure.as_ref(),
            ledger_plan.norm.as_ref(),
            ProximityGroupRole::Shared,
        )
        .expect("derive the shared loop chain")
        .expect("the shared group has stages");
        let mut own_switch_oracles = Vec::new();
        let mut own_fold_oracles = Vec::new();
        let mut shared_switch_oracles = Vec::new();
        let mut shared_fold_oracles = Vec::new();
        for (oracles, folds, iopp) in [
            (&mut own_switch_oracles, &mut own_fold_oracles, &own_iopp),
            (
                &mut shared_switch_oracles,
                &mut shared_fold_oracles,
                &shared_iopp,
            ),
        ] {
            for (index, stage) in iopp.chain.stages.iter().enumerate().skip(1) {
                let tree = stage.tree.as_ref().expect("a switch tree plan");
                let oracle_id = index as u64;
                let key = ledger_oracle(
                    OracleKind::Cs,
                    ledger_identity(OracleKind::Cs, oracle_id, tree.tree_number),
                    oracle_id,
                    tree.tree_number,
                    tree.run,
                    tree.configuration,
                );
                reserve(
                    key,
                    OracleBudget::new(
                        stage.plan.positions as u64,
                        stage.plan.main.r.min(stage.plan.mask.r),
                    )
                    .expect("switch-tree budget"),
                );
                oracles.push(key);
                if let Some(fold) = &stage.fold {
                    let fold_key = ledger_oracle(
                        OracleKind::Fold,
                        ledger_identity(OracleKind::Fold, oracle_id, fold.mask_tree.tree_number),
                        oracle_id,
                        fold.mask_tree.tree_number,
                        fold.mask_tree.run,
                        fold.mask_tree.configuration,
                    );
                    reserve(
                        fold_key,
                        OracleBudget::new(
                            stage.plan.positions as u64,
                            fold.shape.mask_code.r,
                        )
                        .expect("stage-fold budget"),
                    );
                    folds.push(fold_key);
                }
            }
        }
        let own_star = own_iopp.chain.config.base_star;
        let own_star_key = ledger_oracle(
            OracleKind::Star,
            ledger_identity(OracleKind::Star, 1, own_star.tree_number),
            1,
            own_star.tree_number,
            own_star.run,
            own_star.configuration,
        );
        reserve(
            own_star_key,
            OracleBudget::new(setup.proximity.own_base_positions as u64, 0)
                .expect("own loop star budget"),
        );
        let own_star_mask_key = ledger_oracle(
            OracleKind::StarMask,
            ledger_identity(OracleKind::StarMask, 1, own_star.tree_number),
            1,
            own_star.tree_number,
            own_star.run,
            own_star.configuration,
        );
        reserve(
            own_star_mask_key,
            OracleBudget::new(setup.proximity.own_base_positions as u64, 0)
                .expect("own loop star-mask budget"),
        );
        let shared_star = shared_iopp.chain.config.base_star;
        let shared_star_key = ledger_oracle(
            OracleKind::Star,
            ledger_identity(OracleKind::Star, 2, shared_star.tree_number),
            2,
            shared_star.tree_number,
            shared_star.run,
            shared_star.configuration,
        );
        reserve(
            shared_star_key,
            OracleBudget::new(setup.proximity.shared_base_positions as u64, 0)
                .expect("shared loop star budget"),
        );
        let shared_star_mask_key = ledger_oracle(
            OracleKind::StarMask,
            ledger_identity(OracleKind::StarMask, 2, shared_star.tree_number),
            2,
            shared_star.tree_number,
            shared_star.run,
            shared_star.configuration,
        );
        reserve(
            shared_star_mask_key,
            OracleBudget::new(setup.proximity.shared_base_positions as u64, 0)
                .expect("shared loop star-mask budget"),
        );
        let activation_root = activation_commitment.root();
        let activation_oracle = ledger_oracle(
            OracleKind::Activation,
            activation_root,
            0,
            0xE001,
            ACTIVATION_CONFIGURATION,
            ACTIVATION_CONFIGURATION,
        );
        reserve(
            activation_oracle,
            OracleBudget {
                logical_draws: 0,
                tail_rank: 0,
                allocated_rows: OUTPUT_HEAD_POSITIONS as u64,
                committed_entries: 0,
            },
        );
        ledger
            .bind_commitment(request_id, activation_oracle, activation_root)
            .expect("bind the activation commitment");

        let table_identity = reader_table_identity(&reader).unwrap();
        let table_oracle = ledger_oracle(
            OracleKind::Table,
            table_identity,
            0,
            0,
            ledger_table_run(&reader.root),
            reader.configuration,
        );
        ledger
            .reserve_oracle(
                request_id,
                table_oracle,
                ledger_stream(&table_oracle, ledger_table_material(&reader.root)),
                OracleBudget::new(COLUMN_DRAWS as u64, reader.shape.code.r)
                    .expect("table budget"),
            )
            .expect("reserve the table material");
        ledger
            .bind_commitment(request_id, table_oracle, reader.root)
            .expect("bind the table commitment");

        let mut weight_oracles = Vec::with_capacity(OUTPUT_HEAD_POSITIONS);
        for position in 0..OUTPUT_HEAD_POSITIONS as u64 {
            let oracle_id = setup.send.oracle_base + position;
            let tree_number = setup.send.w.tree_number + position;
            let oracle = ledger_oracle(
                OracleKind::Weight,
                ledger_identity(OracleKind::Weight, oracle_id, tree_number),
                oracle_id,
                tree_number,
                setup.send.w.run,
                setup.send.w.configuration,
            );
            reserve(
                oracle,
                OracleBudget::new(
                    (setup.send.openings_per_oracle as usize + own_ledger_positions) as u64,
                    setup.send.w.code.r,
                )
                .expect("W budget"),
            );
            weight_oracles.push(oracle);
        }

        let index_plan = setup
            .index_mask
            .as_ref()
            .expect("the formal plan carries an index mask");
        let index_oracle = ledger_oracle(
            OracleKind::IndexMask,
            ledger_identity(
                OracleKind::IndexMask,
                index_plan.oracle,
                index_plan.tree.tree_number,
            ),
            index_plan.oracle,
            index_plan.tree.tree_number,
            index_plan.tree.run,
            index_plan.tree.configuration,
        );
        reserve(
            index_oracle,
            OracleBudget::new(
                (index_plan.openings as usize + own_ledger_positions) as u64,
                index_plan.mask_code.r,
            )
            .expect("index-mask budget")
            .with_allocation(0, index_plan.entries.total().unwrap() as u64 + 1),
        );

        let dot_plan = setup
            .send
            .dot
            .as_ref()
            .expect("the formal plan carries a shown map");
        let dot_oracle = ledger_oracle(
            OracleKind::DotMap,
            ledger_identity(OracleKind::DotMap, dot_plan.oracle, dot_plan.shape.tree_number),
            dot_plan.oracle,
            dot_plan.shape.tree_number,
            dot_plan.shape.run,
            dot_plan.shape.configuration,
        );
        reserve(
            dot_oracle,
            OracleBudget::new(
                own_ledger_positions as u64,
                dot_plan.shape.code.r,
            )
            .expect("shown-map budget"),
        );

        let omega_plan = &setup.proximity.omega;
        let omega_oracle = ledger_oracle(
            OracleKind::Omega,
            ledger_identity(OracleKind::Omega, omega_plan.oracle, omega_plan.tree.tree_number),
            omega_plan.oracle,
            omega_plan.tree.tree_number,
            omega_plan.tree.run,
            omega_plan.tree.configuration,
        );
        reserve(
            omega_oracle,
            OracleBudget::new(
                (shared_ledger_positions + own_ledger_positions) as u64,
                omega_plan.u.code.r,
            )
            .expect("Omega budget"),
        );

        let gu_plan = &setup.proximity.gu;
        let gu_oracle = ledger_oracle(
            OracleKind::Gu,
            ledger_identity(OracleKind::Gu, gu_plan.oracle, gu_plan.tree.tree_number),
            gu_plan.oracle,
            gu_plan.tree.tree_number,
            gu_plan.tree.run,
            gu_plan.tree.configuration,
        );
        reserve(
            gu_oracle,
            OracleBudget::new(
                shared_ledger_positions as u64,
                gu_plan.u.code.r,
            )
            .expect("G_u budget"),
        );

        let fresh_shared_oracle = ledger_oracle(
            OracleKind::FreshMask,
            ledger_identity(
                OracleKind::FreshMask,
                setup.proximity.shared_fresh,
                setup.proximity.shared_mask_tree.tree_number,
            ),
            setup.proximity.shared_fresh,
            setup.proximity.shared_mask_tree.tree_number,
            setup.proximity.shared_mask_tree.run,
            setup.proximity.shared_mask_tree.configuration,
        );
        reserve(
            fresh_shared_oracle,
            OracleBudget::new(
                shared_ledger_positions as u64,
                setup.proximity.mask_code.r,
            )
            .expect("shared fresh-mask budget"),
        );
        let fresh_own_oracle = ledger_oracle(
            OracleKind::FreshMask,
            ledger_identity(
                OracleKind::FreshMask,
                setup.proximity.own_fresh,
                setup.proximity.own_mask_tree.tree_number,
            ),
            setup.proximity.own_fresh,
            setup.proximity.own_mask_tree.tree_number,
            setup.proximity.own_mask_tree.run,
            setup.proximity.own_mask_tree.configuration,
        );
        reserve(
            fresh_own_oracle,
            OracleBudget::new(
                own_ledger_positions as u64,
                setup.proximity.mask_code.r,
            )
            .expect("own fresh-mask budget"),
        );

        let star_shared_run = ledger_star_run(0);
        let star_own_run = ledger_star_run(1);
        let star_shared = ledger_oracle(
            OracleKind::Star,
            ledger_identity(OracleKind::Star, 0, 0),
            0,
            0,
            star_shared_run,
            setup.send.w.configuration,
        );
        reserve(
            star_shared,
            OracleBudget::new(setup.proximity.shared_base_positions as u64, 0)
                .expect("shared star budget"),
        );
        let star_mask_shared = ledger_oracle(
            OracleKind::StarMask,
            ledger_identity(OracleKind::StarMask, 1, 0),
            1,
            0,
            star_shared_run,
            setup.send.w.configuration,
        );
        reserve(
            star_mask_shared,
            OracleBudget::new(setup.proximity.shared_base_positions as u64, 0)
                .expect("shared star-mask budget"),
        );
        let star_own = ledger_oracle(
            OracleKind::Star,
            ledger_identity(OracleKind::Star, 0, 0),
            0,
            0,
            star_own_run,
            setup.send.w.configuration,
        );
        reserve(
            star_own,
            OracleBudget::new(setup.proximity.own_base_positions as u64, 0)
                .expect("own star budget"),
        );
        let star_mask_own = ledger_oracle(
            OracleKind::StarMask,
            ledger_identity(OracleKind::StarMask, 1, 0),
            1,
            0,
            star_own_run,
            setup.send.w.configuration,
        );
        reserve(
            star_mask_own,
            OracleBudget::new(setup.proximity.own_base_positions as u64, 0)
                .expect("own star-mask budget"),
        );

        let closure_setup = setup
            .closure
            .as_ref()
            .expect("the formal plan carries the output closure");
        let mut closure_slack_oracles = Vec::with_capacity(OUTPUT_HEAD_POSITIONS);
        let mut closure_bits_oracles = Vec::with_capacity(OUTPUT_HEAD_POSITIONS);
        let mut closure_map_oracles = Vec::with_capacity(OUTPUT_HEAD_POSITIONS);
        for position in 0..OUTPUT_HEAD_POSITIONS as u64 {
            for (kind, base, shape, openings, oracles) in [
                (
                    OracleKind::ClosureSlack,
                    closure_setup.slack_oracle_base,
                    closure_setup.slack,
                    closure_setup.slack_openings,
                    &mut closure_slack_oracles,
                ),
                (
                    OracleKind::ClosureBits,
                    closure_setup.bits_oracle_base,
                    closure_setup.bits,
                    closure_setup.bits_openings,
                    &mut closure_bits_oracles,
                ),
                (
                    OracleKind::ClosureMap,
                    closure_setup.map_oracle_base,
                    closure_setup.map,
                    closure_setup.map_openings,
                    &mut closure_map_oracles,
                ),
            ] {
                let oracle_id = base + position;
                let tree_number = shape.tree_number + position;
                let oracle = ledger_oracle(
                    kind,
                    ledger_identity(kind, oracle_id, tree_number),
                    oracle_id,
                    tree_number,
                    shape.run,
                    shape.configuration,
                );
                reserve(
                    oracle,
                    OracleBudget::new(
                        (openings as usize + own_ledger_positions) as u64,
                        shape.code.r,
                    )
                    .expect("closure value budget"),
                );
                oracles.push(oracle);
            }
        }
        let closure_mask_plan = &closure_setup.mask;
        let closure_mask_oracle = ledger_oracle(
            OracleKind::ClosureMask,
            ledger_identity(
                OracleKind::ClosureMask,
                closure_mask_plan.oracle,
                closure_mask_plan.tree.tree_number,
            ),
            closure_mask_plan.oracle,
            closure_mask_plan.tree.tree_number,
            closure_mask_plan.tree.run,
            closure_mask_plan.tree.configuration,
        );
        reserve(
            closure_mask_oracle,
            OracleBudget::new(
                (closure_mask_plan.openings as usize + own_ledger_positions) as u64,
                closure_mask_plan.mask_code.r,
            )
            .expect("closure-mask budget")
            .with_allocation(0, closure_mask_plan.total_entries().unwrap() as u64 + 1),
        );
        // The actual model-bound witness: the largest magnitude of one
        // reduced output cell and the largest comparison slack under the
        // trusted mask and winner.
        let mut max_abs_output = 0i64;
        let mut max_slack = 0i64;
        for position in 0..OUTPUT_HEAD_POSITIONS {
            let row =
                &request.inputs.output[position * OUTPUT_HEAD_COLUMNS..(position + 1) * OUTPUT_HEAD_COLUMNS];
            let winner = output_plan.expected_argmax[position] as usize;
            let winner_value = row[winner];
            for (column, value) in row.iter().enumerate() {
                max_abs_output = max_abs_output.max(value.saturating_abs());
                let masked = value
                    .saturating_add(if column >= output_plan.valid {
                        output_plan.mask_value
                    } else {
                        0
                    });
                let low = if column < winner { 1 } else { 0 };
                max_slack = max_slack.max(winner_value.saturating_sub(masked).saturating_sub(low));
            }
        }
        println!("output_abs_max={max_abs_output}");
        println!("output_slack_max={max_slack}");
        println!(
            "closure_range_bits={} closure_slack_bits={}",
            OUTPUT_HEAD_RANGE_BITS, OUTPUT_HEAD_SLACK_BITS
        );
        // Test-only historical-artifact branch: decode and verify a proof
        // produced by a previous build with the current verifier. It is off
        // unless the load path is set, so the formal run is unchanged.
        if let Some(path) = env::var_os("ZKIE_SESSION_PROOF_LOAD") {
            let bytes = fs::read(path).expect("read the loaded proof");
            let load_setup = setup.clone();
            let load_plan = build_output_head_hidden_plan(
                shape,
                &load_setup,
                &table.reader(),
                &trusted,
            )
            .expect("loaded proof plan");
            let loaded = verify_output_head_lin_hidden_bytes(
                shape,
                &load_plan,
                &trusted,
                &table.reader(),
                &bytes,
            );
            match &loaded {
                Ok(outcome) => println!("loaded_proof_ok=1 bytes={}", outcome.proof_bytes),
                Err(error) => println!("loaded_proof_ok=0 error={error:?}"),
            }
            loaded.unwrap();
            return;
        }
        // The planner is part of the trusted plan fault path: a changed
        // own or shared position must be rejected before any proof work.
        let fault_plan = build_output_head_hidden_plan(
            shape,
            &setup,
            &table.reader(),
            &trusted,
        )
        .expect("build the fault-path plan");
        let mut bad_own = fault_plan.proximity.clone();
        bad_own.own_stages[0].positions += 1;
        assert!(
            bad_own
                .fault(
                    &fault_plan.send,
                    fault_plan.closure.as_ref(),
                    fault_plan.norm.as_ref(),
                )
                .is_some(),
            "a changed own position must fail the trusted plan"
        );
        let mut bad_shared = fault_plan.proximity.clone();
        bad_shared.shared_stages[0].positions = 3;
        assert!(
            bad_shared
                .fault(
                    &fault_plan.send,
                    fault_plan.closure.as_ref(),
                    fault_plan.norm.as_ref(),
                )
                .is_some(),
            "a changed shared join count must fail the trusted plan"
        );

        let started = Instant::now();
        let (hidden_proof, hidden_witness, closure) = prove_output_head_lin_hidden(
            shape,
            &setup,
            &output_plan,
            &mut table,
            &activation_rows,
            &setup.activation_salts,
        )
        .unwrap();
        let mut hidden_witness = Some(hidden_witness);
        let hidden_prove_ms = started.elapsed().as_millis();
        let first = &hidden_proof.session.first;
        let first_closure = first
            .closure
            .as_ref()
            .expect("the hidden path commits the closure messages");
        let index_message = first
            .index_mask
            .as_ref()
            .expect("the hidden path commits the mask message");
        let dot_message = first
            .dot
            .as_ref()
            .expect("the hidden path commits the shown-map message");

        for (position, oracle) in weight_oracles.iter().enumerate() {
            ledger
                .bind_commitment(
                    request_id,
                    *oracle,
                    hidden_witness
                        .as_ref()
                        .expect("the hidden witness is alive before the verifier")
                        .send
                        .roots[position],
                )
                .expect("bind a W commitment");
        }
        ledger
            .bind_commitment(request_id, index_oracle, index_message.root)
            .expect("bind the mask commitment");
        ledger
            .bind_commitment(request_id, dot_oracle, dot_message.root)
            .expect("bind the shown-map commitment");
        for (position, oracle) in closure_slack_oracles.iter().enumerate() {
            ledger
                .bind_commitment(request_id, *oracle, first_closure.slack.roots[position])
                .expect("bind a slack commitment");
        }
        for (position, oracle) in closure_bits_oracles.iter().enumerate() {
            ledger
                .bind_commitment(request_id, *oracle, first_closure.bits.roots[position])
                .expect("bind a packed-bit commitment");
        }
        for (position, oracle) in closure_map_oracles.iter().enumerate() {
            ledger
                .bind_commitment(request_id, *oracle, first_closure.map.roots[position])
                .expect("bind a product-map commitment");
        }
        ledger
            .bind_commitment(request_id, closure_mask_oracle, first_closure.mask.root)
            .expect("bind the closure-mask commitment");

        // The proximity group's end decides which openings the message
        // carries. A group with stages ends in the loop proof: its switch
        // records carry the previous stage's member and fresh openings, and
        // the last-stage BASE openings live in the canonical LastBase
        // artifact, which the loop verifier replays. A group without stages
        // ends in the single-stage BASE artifact and carries those openings
        // directly. The ledger records the artifacts the message exposes; the
        // loop's measured budget is recorded by the acceptance run.
        let mut single_stage = true;
        let (shared_base, shared_fresh, shared_star_raw, shared_star_mask_raw) =
            match &hidden_proof.session.shared.tail {
                ProximityTail::Single { base_openings, .. } => {
                    let shared_members = &base_openings.members;
                    let shared_base = shared_members
                        .first()
                        .expect("the shared group opens its members")
                        .positions
                        .clone();
                    assert!(
                        shared_members
                            .iter()
                            .all(|opening| opening.positions == shared_base),
                        "every shared member opens the same BASE position set"
                    );
                    let shared_fresh = base_openings.fresh.positions.clone();
                    let shared_star_raw: Vec<u64> = base_openings
                        .star
                        .iter()
                        .map(|opening| opening.index)
                        .collect();
                    let shared_star_mask_raw: Vec<u64> = base_openings
                        .star_mask
                        .iter()
                        .map(|opening| opening.index)
                        .collect();
                    (
                        shared_base,
                        shared_fresh,
                        shared_star_raw,
                        shared_star_mask_raw,
                    )
                }
                ProximityTail::Stages(proof) => {
                    single_stage = false;
                    assert_eq!(
                        proof.switches.len(),
                        setup.proximity.shared_stages.len() - 1,
                        "the shared loop carries one switch per later stage"
                    );
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new())
                }
            };
        let (own_base, own_fresh, own_star_raw, own_star_mask_raw) =
            match &hidden_proof.session.second.tail {
                ProximityTail::Single { base_openings, .. } => {
                    let own_members = &base_openings.members;
                    let own_base = own_members
                        .first()
                        .expect("the own group opens its members")
                        .positions
                        .clone();
                    assert!(
                        own_members
                            .iter()
                            .all(|opening| opening.positions == own_base),
                        "every own member opens the same BASE position set"
                    );
                    let own_fresh = base_openings.fresh.positions.clone();
                    let own_star_raw: Vec<u64> =
                        base_openings.star.iter().map(|opening| opening.index).collect();
                    let own_star_mask_raw: Vec<u64> = base_openings
                        .star_mask
                        .iter()
                        .map(|opening| opening.index)
                        .collect();
                    (own_base, own_fresh, own_star_raw, own_star_mask_raw)
                }
                ProximityTail::Stages(proof) => {
                    single_stage = false;
                    assert_eq!(
                        proof.switches.len(),
                        setup.proximity.own_stages.len() - 1,
                        "the own loop carries one switch per later stage"
                    );
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new())
                }
            };
        let shared_star = ledger_unique(&shared_star_raw);
        let shared_star_mask = ledger_unique(&shared_star_mask_raw);
        let own_star = ledger_unique(&own_star_raw);
        let own_star_mask = ledger_unique(&own_star_mask_raw);

        let opening_counts = |logical: u64, positions: &[u64]| {
            OpeningCounts::new(logical, positions.len() as u64, positions.len() as u64)
                .expect("opening counts")
        };
        let star_counts = |raw: &[u64], unique: &[u64]| {
            OpeningCounts::new(raw.len() as u64, raw.len() as u64, unique.len() as u64)
                .expect("star opening counts")
        };

        let column_draws = hidden_proof.session.shared.column_positions.clone();
        let column_unique = ledger_unique(&column_draws);
        let mut expected_unions: BTreeMap<OracleKey, BTreeSet<u64>> = BTreeMap::new();
        let mut expected_logical: BTreeMap<OracleKey, u64> = BTreeMap::new();
        {
        let mut record = |oracle: OracleKey,
                          stage: Stage,
                          source: OpeningSource,
                          counts: OpeningCounts,
                          positions: Vec<u64>| {
            let tail = ledger
                .record_openings(request_id, oracle, stage, source, counts, &positions)
                .expect("record an opening event");
            let union = expected_unions.entry(oracle).or_default();
            for position in &positions {
                union.insert(*position);
            }
            *expected_logical.entry(oracle).or_insert(0) += counts.logical_draws;
            assert!(
                tail.rank == 0 || tail.new_union <= tail.rank,
                "the ledger keeps the tail union inside the rank"
            );
        };

        for (position, oracle) in weight_oracles.iter().enumerate() {
            let positions = first.positions[position].clone();
            record(
                *oracle,
                Stage::Send,
                OpeningSource::SendSample,
                opening_counts(setup.send.openings_per_oracle as u64, &positions),
                positions,
            );
            if single_stage {
                record(
                    *oracle,
                    Stage::Base,
                    OpeningSource::BaseMember,
                    opening_counts(setup.proximity.own_base_positions as u64, &own_base),
                    own_base.clone(),
                );
            }
        }
        record(
            table_oracle,
            Stage::Table,
            OpeningSource::TableColumn,
            opening_counts(column_draws.len() as u64, &column_unique),
            column_unique.clone(),
        );
        record(
            index_oracle,
            Stage::Send,
            OpeningSource::SendSample,
            opening_counts(index_plan.openings as u64, &index_message.positions),
            index_message.positions.clone(),
        );
        if single_stage {
            record(
                index_oracle,
                Stage::Base,
                OpeningSource::BaseMember,
                opening_counts(setup.proximity.own_base_positions as u64, &own_base),
                own_base.clone(),
            );
            record(
                dot_oracle,
                Stage::Base,
                OpeningSource::BaseMember,
                opening_counts(setup.proximity.own_base_positions as u64, &own_base),
                own_base.clone(),
            );
            record(
                omega_oracle,
                Stage::Shared,
                OpeningSource::BaseMember,
                opening_counts(setup.proximity.shared_base_positions as u64, &shared_base),
                shared_base.clone(),
            );
            record(
                omega_oracle,
                Stage::Base,
                OpeningSource::BaseMember,
                opening_counts(setup.proximity.own_base_positions as u64, &own_base),
                own_base.clone(),
            );
            record(
                gu_oracle,
                Stage::Shared,
                OpeningSource::BaseMember,
                opening_counts(setup.proximity.shared_base_positions as u64, &shared_base),
                shared_base.clone(),
            );
            record(
                fresh_shared_oracle,
                Stage::Shared,
                OpeningSource::BaseFresh,
                opening_counts(setup.proximity.shared_base_positions as u64, &shared_fresh),
                shared_fresh.clone(),
            );
            record(
                fresh_own_oracle,
                Stage::Base,
                OpeningSource::BaseFresh,
                opening_counts(setup.proximity.own_base_positions as u64, &own_fresh),
                own_fresh.clone(),
            );
            record(
                star_shared,
                Stage::Shared,
                OpeningSource::BaseStar,
                star_counts(&shared_star_raw, &shared_star),
                shared_star.clone(),
            );
            record(
                star_mask_shared,
                Stage::Shared,
                OpeningSource::BaseStar,
                star_counts(&shared_star_mask_raw, &shared_star_mask),
                shared_star_mask.clone(),
            );
            record(
                star_own,
                Stage::Base,
                OpeningSource::BaseStar,
                star_counts(&own_star_raw, &own_star),
                own_star.clone(),
            );
            record(
                star_mask_own,
                Stage::Base,
                OpeningSource::BaseStar,
                star_counts(&own_star_mask_raw, &own_star_mask),
                own_star_mask.clone(),
            );
        }
        for (position, oracle) in closure_slack_oracles.iter().enumerate() {
            let positions = first_closure.slack_positions[position].clone();
            record(
                *oracle,
                Stage::Send,
                OpeningSource::SendSample,
                opening_counts(closure_setup.slack_openings as u64, &positions),
                positions,
            );
            if single_stage {
                record(
                    *oracle,
                    Stage::Base,
                    OpeningSource::BaseMember,
                    opening_counts(setup.proximity.own_base_positions as u64, &own_base),
                    own_base.clone(),
                );
            }
        }
        for (position, oracle) in closure_bits_oracles.iter().enumerate() {
            let positions = first_closure.bits_positions[position].clone();
            record(
                *oracle,
                Stage::Send,
                OpeningSource::SendSample,
                opening_counts(closure_setup.bits_openings as u64, &positions),
                positions,
            );
            if single_stage {
                record(
                    *oracle,
                    Stage::Base,
                    OpeningSource::BaseMember,
                    opening_counts(setup.proximity.own_base_positions as u64, &own_base),
                    own_base.clone(),
                );
            }
        }
        for (position, oracle) in closure_map_oracles.iter().enumerate() {
            let positions = first_closure.map_positions[position].clone();
            record(
                *oracle,
                Stage::Send,
                OpeningSource::SendSample,
                opening_counts(closure_setup.map_openings as u64, &positions),
                positions,
            );
            if single_stage {
                record(
                    *oracle,
                    Stage::Base,
                    OpeningSource::BaseMember,
                    opening_counts(setup.proximity.own_base_positions as u64, &own_base),
                    own_base.clone(),
                );
            }
        }
        record(
            closure_mask_oracle,
            Stage::Send,
            OpeningSource::SendSample,
            opening_counts(closure_mask_plan.openings as u64, &first_closure.mask.positions),
            first_closure.mask.positions.clone(),
        );
        if single_stage {
            record(
                closure_mask_oracle,
                Stage::Base,
                OpeningSource::BaseMember,
                opening_counts(setup.proximity.own_base_positions as u64, &own_base),
                own_base.clone(),
            );
        }

        // The loop's switch openings are recorded against the plan-derived
        // keys. Stage 0 opens every original member at the same position set;
        // later switches open the previous switch tree.
        if let ProximityTail::Stages(proof) = &hidden_proof.session.shared.tail {
            for (index, switch) in proof.switches.iter().enumerate() {
                let logical = shared_iopp.chain.stages[index].plan.positions as u64;
                let main_positions = switch
                    .member_openings
                    .first()
                    .map(|opening| opening.positions.clone())
                    .unwrap_or_default();
                if index == 0 {
                    assert_eq!(switch.member_openings.len(), 2);
                    assert!(
                        switch
                            .member_openings
                            .iter()
                            .all(|opening| opening.positions == main_positions),
                        "every shared member opens the same stage-0 positions"
                    );
                    record(
                        gu_oracle,
                        Stage::Switch,
                        OpeningSource::Switch,
                        opening_counts(logical, &main_positions),
                        main_positions.clone(),
                    );
                    record(
                        omega_oracle,
                        Stage::Switch,
                        OpeningSource::Switch,
                        opening_counts(logical, &main_positions),
                        main_positions,
                    );
                } else {
                    assert_eq!(switch.member_openings.len(), 1);
                    record(
                        shared_switch_oracles[index - 1],
                        Stage::Switch,
                        OpeningSource::Switch,
                        opening_counts(logical, &main_positions),
                        main_positions,
                    );
                }
                let fresh = if index == 0 {
                    fresh_shared_oracle
                } else {
                    shared_fold_oracles[index - 1]
                };
                record(
                    fresh,
                    Stage::Switch,
                    OpeningSource::Switch,
                    opening_counts(logical, &switch.fresh_opening.positions),
                    switch.fresh_opening.positions.clone(),
                );
                assert!(
                    switch.old_mask_opening.is_none(),
                    "the current shared old-mask source has no opening"
                );
            }
        }
        if let ProximityTail::Stages(proof) = &hidden_proof.session.second.tail {
            for (index, switch) in proof.switches.iter().enumerate() {
                let logical = own_iopp.chain.stages[index].plan.positions as u64;
                let main_positions = switch
                    .member_openings
                    .first()
                    .map(|opening| opening.positions.clone())
                    .unwrap_or_default();
                if index == 0 {
                    assert_eq!(
                        switch.member_openings.len(),
                        weight_oracles.len() + 1 + 1 + closure_slack_oracles.len()
                            + closure_bits_oracles.len()
                            + closure_map_oracles.len()
                            + 1
                            + 1
                    );
                    assert!(
                        switch
                            .member_openings
                            .iter()
                            .all(|opening| opening.positions == main_positions),
                        "every own member opens the same stage-0 positions"
                    );
                    for oracle in weight_oracles
                        .iter()
                        .chain(std::iter::once(&index_oracle))
                        .chain(std::iter::once(&dot_oracle))
                        .chain(closure_slack_oracles.iter())
                        .chain(closure_bits_oracles.iter())
                        .chain(closure_map_oracles.iter())
                        .chain(std::iter::once(&closure_mask_oracle))
                        .chain(std::iter::once(&omega_oracle))
                    {
                        record(
                            *oracle,
                            Stage::Switch,
                            OpeningSource::Switch,
                            opening_counts(logical, &main_positions),
                            main_positions.clone(),
                        );
                    }
                } else {
                    assert_eq!(switch.member_openings.len(), 1);
                    record(
                        own_switch_oracles[index - 1],
                        Stage::Switch,
                        OpeningSource::Switch,
                        opening_counts(logical, &main_positions),
                        main_positions,
                    );
                }
                let fresh = if index == 0 {
                    fresh_own_oracle
                } else {
                    own_fold_oracles[index - 1]
                };
                record(
                    fresh,
                    Stage::Switch,
                    OpeningSource::Switch,
                    opening_counts(logical, &switch.fresh_opening.positions),
                    switch.fresh_opening.positions.clone(),
                );
                assert!(
                    switch.old_mask_opening.is_none(),
                    "the current own old-mask source has no opening"
                );
            }
        }

        }
        if single_stage {
            ledger.mark_proved(request_id).expect("the prover returned");
        }
        assert_eq!(
            ledger.succeed_request(request_id).unwrap_err(),
            zkie_core::error::Error::InvalidSessionState,
            "success before the complete verification is rejected"
        );
        let ledger_state = ledger.read_state().expect("read the opening ledger");
        // The pinned totals describe the single-stage delivery basis: every
        // group member and its fresh tree are opened at the BASE positions, so
        // the ledger sees one event per oracle and source. The loop moves those
        // openings into the switch and LastBase artifacts; its measured totals
        // are recorded by the acceptance run.
        if single_stage {
            assert_eq!(
                ledger_state.total_event_draws().expect("logical draws"),
                887,
                "the closed relation charges every logical draw"
            );
            assert_eq!(ledger_state.events.len(), 143);
            let event_count = |kind: OracleKind| {
                ledger_state
                    .events
                    .iter()
                    .filter(|event| event.oracle.kind == kind)
                    .count()
            };
            assert_eq!(event_count(OracleKind::Weight), 32);
            assert_eq!(event_count(OracleKind::Table), 1);
            assert_eq!(event_count(OracleKind::IndexMask), 2);
            assert_eq!(event_count(OracleKind::DotMap), 1);
            assert_eq!(event_count(OracleKind::Omega), 2);
            assert_eq!(event_count(OracleKind::Gu), 1);
            assert_eq!(event_count(OracleKind::FreshMask), 2);
            assert_eq!(event_count(OracleKind::Star), 2);
            assert_eq!(event_count(OracleKind::StarMask), 2);
            assert_eq!(event_count(OracleKind::ClosureSlack), 32);
            assert_eq!(event_count(OracleKind::ClosureBits), 32);
            assert_eq!(event_count(OracleKind::ClosureMap), 32);
            assert_eq!(event_count(OracleKind::ClosureMask), 2);
            assert_eq!(event_count(OracleKind::Activation), 0);
            assert_eq!(event_count(OracleKind::Switch), 0);
            assert_eq!(event_count(OracleKind::Cs), 0);
        }
        let activation_record = ledger_state
            .oracle(activation_oracle)
            .expect("activation record");
        assert_eq!(activation_record.allocated_rows, OUTPUT_HEAD_POSITIONS as u64);
        assert_eq!(activation_record.logical_openings, 0);
        assert_eq!(activation_record.opened_positions.len(), 0);
        let table_record = ledger_state.oracle(table_oracle).expect("table record");
        assert_eq!(table_record.logical_openings, COLUMN_DRAWS as u64);
        assert_eq!(table_record.unique_positions, column_unique.len() as u64);
        let closure_draws = ledger_state
            .totals_by_kind(OracleKind::ClosureSlack)
            .expect("slack")
            .logical_draws
            + ledger_state
                .totals_by_kind(OracleKind::ClosureBits)
                .expect("bits")
                .logical_draws
            + ledger_state
                .totals_by_kind(OracleKind::ClosureMap)
                .expect("map")
                .logical_draws
            + ledger_state
                .totals_by_kind(OracleKind::ClosureMask)
                .expect("mask")
                .logical_draws;
        if single_stage {
            assert_eq!(closure_draws, 196);
            assert_eq!(
                ledger_state.total_event_draws().expect("draws") - closure_draws,
                691
            );
        }
        for (oracle, union) in &expected_unions {
            let record = ledger_state.oracle(*oracle).expect("oracle record");
            assert_eq!(&record.opened_positions, union);
            assert_eq!(record.logical_openings, expected_logical[oracle]);
            assert_eq!(record.unique_positions, union.len() as u64);
            assert_eq!(record.totals().logical_draws, expected_logical[oracle]);
        }
        let expected_unique: u64 = expected_unions
            .values()
            .map(|union| union.len() as u64)
            .sum();
        assert_eq!(
            ledger_state
                .total_event_unique_positions()
                .expect("unique positions"),
            expected_unique
        );
        println!("ledger_events={}", ledger_state.events.len());
        println!(
            "ledger_logical_openings={}",
            ledger_state.total_event_draws().expect("draws")
        );
        println!(
            "ledger_unique_positions={}",
            ledger_state.total_event_unique_positions().expect("unique")
        );
        println!("ledger_table_logical={}", table_record.logical_openings);
        println!("ledger_table_unique={}", table_record.unique_positions);
        println!(
            "ledger_activation_openings={}",
            activation_record.logical_openings
        );
        println!("ledger_activation_rows={}", activation_record.allocated_rows);
        println!("ledger_closure_logical={closure_draws}");
        println!(
            "ledger_pre_closure_logical={}",
            ledger_state.total_event_draws().expect("draws") - closure_draws
        );
        let pre_closure_unique: u64 = expected_unions
            .iter()
            .filter(|(oracle, _)| {
                !matches!(
                    oracle.kind,
                    OracleKind::ClosureSlack
                        | OracleKind::ClosureBits
                        | OracleKind::ClosureMap
                        | OracleKind::ClosureMask
                )
            })
            .map(|(_, union)| union.len() as u64)
            .sum();
        println!("ledger_pre_closure_unique={pre_closure_unique}");
        println!(
            "ledger_table_reuse={}",
            ledger_state.total_table_reuse().expect("table reuse")
        );
        let budget = ledger_state.opening_budget_summary();
        println!("ledger_code_oracles={}", budget.code_oracles);
        println!("ledger_min_tail_remaining={}", budget.min_remaining);
        println!("ledger_total_tail_rank={}", budget.total_rank);
        println!("ledger_total_tail_unique={}", budget.total_unique);
        println!("ledger_exhausted_tail_oracles={}", budget.exhausted);
        println!("hidden_prove_ms={hidden_prove_ms}");
        let mut hidden_encode_profile =
            zkie_engine::profile::Scope::new("harness.hidden_proof_encode");
        let hidden_proof_encoded = hidden_proof.encode().unwrap();
        let hidden_proof_bytes = hidden_proof_encoded.len();
        hidden_encode_profile.add_bytes(hidden_proof_bytes as u64);
        if let Some(path) = env::var_os("ZKIE_SESSION_PROOF_DUMP") {
            fs::write(path, &hidden_proof_encoded).unwrap();
        }
        println!("hidden_proof_bytes={hidden_proof_bytes}");
        drop(hidden_encode_profile);
        zkie_engine::profile::dump();
        zkie_engine::profile::reset();
        assert_eq!(closure.argmax, output_plan.expected_argmax);
        let trace_output_rows: Vec<Vec<i64>> = (0..OUTPUT_HEAD_POSITIONS)
            .map(|position| {
                request.inputs.output
                    [position * OUTPUT_HEAD_COLUMNS..(position + 1) * OUTPUT_HEAD_COLUMNS]
                    .to_vec()
            })
            .collect();
        assert_eq!(
            zkie_ops::proof::output_head_lin::witness_output_rows(
                hidden_witness
                    .as_ref()
                    .expect("the hidden witness is alive before the verifier"),
            )
            .unwrap(),
            trace_output_rows
        );
        // The prover witness holds the retained runtime trees and value rows;
        // the verifier path does not read it, so release it before verifying.
        // The optional agreement check below needs it only when enabled.
        if env::var_os("ZKIE_SESSION_WITNESS").is_none() {
            hidden_witness = None;
        }
        let mut hidden_setup = setup.clone();
        hidden_setup.send.expected_dot_root = Some(dot_message.root);
        let _profile_verify_setup = zkie_engine::profile::Scope::new("harness.verify_setup");
        let hidden_plan =
            build_output_head_hidden_plan(shape, &hidden_setup, &table.reader(), &trusted).unwrap();
        drop(_profile_verify_setup);
        let _profile_hidden_verify = zkie_engine::profile::Scope::new("harness.hidden_verify");
        let started = Instant::now();
        let hidden_outcome = verify_output_head_lin_hidden(
            shape,
            &hidden_plan,
            &trusted,
            &table.reader(),
            &hidden_proof,
        )
        .unwrap();
        println!("hidden_verify_ms={}", started.elapsed().as_millis());
        drop(_profile_hidden_verify);
        if !single_stage {
            let mut record = |oracle: OracleKey,
                              stage: Stage,
                              source: OpeningSource,
                              counts: OpeningCounts,
                              positions: Vec<u64>| {
                let tail = ledger
                    .record_openings(request_id, oracle, stage, source, counts, &positions)
                    .expect("record a LastBase opening event");
                let union = expected_unions.entry(oracle).or_default();
                for position in &positions {
                    union.insert(*position);
                }
                *expected_logical.entry(oracle).or_insert(0) += counts.logical_draws;
                assert!(
                    tail.rank == 0 || tail.new_union <= tail.rank,
                    "the ledger keeps the tail union inside the rank"
                );
            };
            if let Some(positions) = hidden_outcome.outcome.shared_base.last_positions() {
                record(
                    *shared_switch_oracles
                        .last()
                        .expect("the shared last switch key"),
                    Stage::Base,
                    OpeningSource::BaseMember,
                    opening_counts(setup.proximity.shared_base_positions as u64, positions),
                    positions.to_vec(),
                );
                record(
                    shared_star_key,
                    Stage::Base,
                    OpeningSource::BaseStar,
                    opening_counts(setup.proximity.shared_base_positions as u64, positions),
                    positions.to_vec(),
                );
                record(
                    shared_star_mask_key,
                    Stage::Base,
                    OpeningSource::BaseStar,
                    opening_counts(setup.proximity.shared_base_positions as u64, positions),
                    positions.to_vec(),
                );
            } else {
                panic!("the shared loop did not return BASE positions");
            }
            if let Some(positions) = hidden_outcome.outcome.own_base.last_positions() {
                record(
                    *own_switch_oracles
                        .last()
                        .expect("the own last switch key"),
                    Stage::Base,
                    OpeningSource::BaseMember,
                    opening_counts(setup.proximity.own_base_positions as u64, positions),
                    positions.to_vec(),
                );
                record(
                    own_star_key,
                    Stage::Base,
                    OpeningSource::BaseStar,
                    opening_counts(setup.proximity.own_base_positions as u64, positions),
                    positions.to_vec(),
                );
                record(
                    own_star_mask_key,
                    Stage::Base,
                    OpeningSource::BaseStar,
                    opening_counts(setup.proximity.own_base_positions as u64, positions),
                    positions.to_vec(),
                );
            } else {
                panic!("the own loop did not return BASE positions");
            }
            ledger.mark_proved(request_id).expect("the prover returned");
        }
        let _profile_ledger_checks = zkie_engine::profile::Scope::new("harness.ledger_checks");
        ledger
            .mark_verified(request_id)
            .expect("the complete verification accepted");
        ledger
            .succeed_request(request_id)
            .expect("close the verified request");
        let closed = ledger
            .request_record(request_id)
            .expect("read the request")
            .expect("request");
        assert_eq!(closed.state, RequestState::Succeeded);
        let final_ledger = ledger.read_state().expect("read the final ledger");
        println!("ledger_final_events={}", final_ledger.events.len());
        println!(
            "ledger_final_logical={}",
            final_ledger.total_event_draws().expect("final draws")
        );
        println!(
            "ledger_final_unique={}",
            final_ledger
                .total_event_unique_positions()
                .expect("final unique")
        );
        let final_budget = final_ledger.opening_budget_summary();
        println!("ledger_final_min_tail_remaining={}", final_budget.min_remaining);
        println!("ledger_final_exhausted={}", final_budget.exhausted);
        if env::var_os("ZKIE_SESSION_KEY_MAP").is_some() {
            for (key, record) in &final_ledger.oracles {
                println!(
                    "ledger_key kind={:?} oracle={} tree={} run={:?} configuration={:?} rank={} unique={} logical={}",
                    key.kind,
                    key.oracle_id,
                    key.tree_number,
                    key.run,
                    key.configuration,
                    record.tail_rank,
                    record.unique_positions,
                    record.logical_openings
                );
            }
        }
        for (key, record) in &final_ledger.oracles {
            if record.tail_rank != 0 && record.remaining_tail() == 0 {
                println!(
                    "ledger_exhausted_key kind={:?} oracle={} tree={} rank={} unique={} logical={}",
                    key.kind,
                    key.oracle_id,
                    key.tree_number,
                    record.tail_rank,
                    record.unique_positions,
                    record.logical_openings
                );
            }
        }
        for (oracle, union) in &expected_unions {
            let record = final_ledger.oracle(*oracle).expect("final oracle record");
            assert_eq!(&record.opened_positions, union);
            assert_eq!(record.logical_openings, expected_logical[oracle]);
            assert_eq!(record.unique_positions, union.len() as u64);
            assert_eq!(record.totals().logical_draws, expected_logical[oracle]);
        }
        let expected_unique: u64 = expected_unions
            .values()
            .map(|union| union.len() as u64)
            .sum();
        assert_eq!(
            final_ledger
                .total_event_unique_positions()
                .expect("final unique"),
            expected_unique
        );
        if single_stage {
            assert_eq!(closed.event_count(), 143);
        }
        // A second request with the same generation seed, or the same material
        // stream under another seed, is rejected even though the roots would
        // differ.
        assert_eq!(
            ledger
                .begin_request(
                    digest_bytes(b"formal-output-head-second-request"),
                    seed_commitment,
                    setup.send.parameters_digest(),
                )
                .unwrap_err(),
            zkie_core::error::Error::StateRollbackSuspected
        );
        let other_request = digest_bytes(b"formal-output-head-third-request");
        ledger
            .begin_request(
                other_request,
                digest_bytes(b"formal-output-head-other-seed"),
                setup.send.parameters_digest(),
            )
            .expect("independent request");
        assert_eq!(
            ledger
                .reserve_oracle(
                    other_request,
                    weight_oracles[0],
                    ledger_stream(&weight_oracles[0], material(&weight_oracles[0])),
                    OracleBudget::new(
                        (setup.send.openings_per_oracle
                            + setup.proximity.own_base_positions) as u64,
                        setup.send.w.code.r,
                    )
                    .expect("retry budget"),
                )
                .unwrap_err(),
            zkie_core::error::Error::StateRollbackSuspected
        );
        drop(_profile_ledger_checks);

        println!(
            "hidden_own_fold_rounds={}",
            hidden_proof.session.second.fold_rounds.len()
        );
        // The optional auxiliary agreement check uses the already-computed
        // public outcome and checks the prover's private mask entries against
        // the tilde row, the residual and the product-sumcheck output rows.
        // It is enabled explicitly so the formal run keeps its established
        // memory and wall-time measurement basis.
        if env::var_os("ZKIE_SESSION_WITNESS").is_some() {
            let started = Instant::now();
            zkie_ops::proof::hidden_send_session::verify_hidden_lin_witness_agreement(
                &hidden_plan,
                &hidden_proof.session,
                hidden_witness
                    .as_ref()
                    .expect("the witness agreement check requires the witness"),
                &hidden_outcome.outcome,
            )
            .unwrap();
            println!(
                "hidden_witness_check_ms={}",
                started.elapsed().as_millis()
            );
            assert_eq!(
                hidden_outcome
                    .outcome
                    .dot
                    .as_ref()
                    .map(|binding| binding.residual),
                hidden_proof.session.second.residual
            );
        }
        hidden_witness = None;
        // The formal manifest binds the hidden proof, not the diagnostic
        // request. The diagnostic byte count stays a separate diagnostic
        // figure; the formal total is trace + hidden proof + public.
        let hidden_bytes = hidden_proof.encode().unwrap();
        let fixture_dir = Path::new(&fixture).join("gpt2-seq16");
        let hidden_rel = "output-head-hidden-request.bin";
        let hidden_path = fixture_dir.join(hidden_rel);
        let _ = fs::remove_file(&hidden_path);
        fs::write(&hidden_path, &hidden_bytes).expect("write hidden request");
        let entry = |role: &str, rel: &str| {
            let data = fs::read(fixture_dir.join(rel)).expect("read formal manifest file");
            ManifestFile {
                role: role.to_string(),
                path: rel.to_string(),
                bytes: data.len() as u64,
                digest: zkie_core::artifact::digest_bytes(&data),
            }
        };
        let formal_manifest = VerificationManifest::new(vec![
            entry("public-commitment", "public/model-commitment.bin"),
            entry("trace", "proof.bin"),
            entry("lin-request", hidden_rel),
        ])
        .expect("formal manifest");
        let _ = fs::remove_file(fixture_dir.join(zkie::lm_head_stage::VERIFICATION_MANIFEST_FILE));
        write_manifest(&fixture_dir, &formal_manifest).expect("write formal manifest");
        read_manifest(&fixture_dir)
            .expect("read formal manifest")
            .verify(&fixture_dir)
            .expect("verify formal manifest");
        let formal_total: u64 = formal_manifest.files.iter().map(|file| file.bytes).sum();
        println!("formal_manifest_total_bytes={formal_total}");
        let public_bytes = fs::metadata(fixture_dir.join("public/model-commitment.bin"))
            .expect("public metadata")
            .len();
        assert_eq!(
            formal_total,
            trace_bytes + hidden_bytes.len() as u64 + public_bytes
        );
        let mut tampered_hidden = hidden_bytes.clone();
        tampered_hidden[0] ^= 1;
        fs::write(&hidden_path, &tampered_hidden).expect("tamper hidden request");
        assert!(formal_manifest.verify(&fixture_dir).is_err());
        fs::write(&hidden_path, &hidden_bytes).expect("restore hidden request");
        assert_eq!(hidden_outcome.outcome.eval_rows.len(), OUTPUT_HEAD_POSITIONS);
        assert_eq!(hidden_proof.session.first.activation, trusted.activation);
        // The verifier outcome carries the reconstructed claim rows; the
        // negative section replays the verifier from the proof alone.
        drop(hidden_outcome);
        drop(hidden_proof_encoded);

        zkie_engine::profile::dump();
        zkie_engine::profile::reset();
        let _profile_negatives = zkie_engine::profile::Scope::new("harness.verify_negatives");
        let clone_proof = |proof: &OutputHeadLinProof| {
            let _profile = zkie_engine::profile::Scope::light("harness.negative_proof_clone");
            proof.clone()
        };
        // A changed activation commitment against the policy pin is rejected.
        let mut changed_trusted = trusted.clone();
        changed_trusted.activation.root.0[0] ^= 1;
        assert!(verify_output_head_lin_hidden(
            shape,
            &hidden_plan,
            &changed_trusted,
            &table.reader(),
            &hidden_proof
        )
        .is_err());

        // Output-closure negatives on the real proof: the booleanity
        // transcript, the fresh-point evaluations and every new commitment
        // reach the final verifier.
        let reject = |candidate: &OutputHeadLinProof| {
            verify_output_head_lin_hidden(
                shape,
                &hidden_plan,
                &trusted,
                &table.reader(),
                candidate,
            )
            .is_err()
        };
        let _profile_closure_negatives =
            zkie_engine::profile::Scope::new("harness.negative_closure");
        let mut tampered = clone_proof(&hidden_proof);
        tampered
            .session
            .second
            .closure
            .as_mut()
            .unwrap()
            .psc[0]
            .rounds[0]
            .coefficients[0] = Fp3::from_u64(1);
        assert!(reject(&tampered));
        let mut tampered = clone_proof(&hidden_proof);
        tampered.session.second.closure.as_mut().unwrap().point_values[0][1] =
            Fp3::from_u64(1);
        assert!(reject(&tampered));
        let mut tampered = clone_proof(&hidden_proof);
        tampered
            .session
            .first
            .closure
            .as_mut()
            .unwrap()
            .bits
            .group
            .entries[0]
            .root
            .0[0] ^= 1;
        assert!(reject(&tampered));
        let mut tampered = clone_proof(&hidden_proof);
        tampered
            .session
            .first
            .closure
            .as_mut()
            .unwrap()
            .map
            .group
            .entries[0]
            .root
            .0[0] ^= 1;
        assert!(reject(&tampered));
        let mut tampered = clone_proof(&hidden_proof);
        tampered
            .session
            .first
            .closure
            .as_mut()
            .unwrap()
            .slack
            .group
            .entries[0]
            .root
            .0[0] ^= 1;
        assert!(reject(&tampered));
        let mut tampered = clone_proof(&hidden_proof);
        tampered.session.first.closure.as_mut().unwrap().mask.root.0[0] ^= 1;
        assert!(reject(&tampered));
        let mut tampered = clone_proof(&hidden_proof);
        tampered.session.first.closure.as_mut().unwrap().slack_sample.answers[0] =
            Fp3::from_u64(1);
        assert!(reject(&tampered));
        let mut missing = clone_proof(&hidden_proof);
        missing.session.first.closure = None;
        assert!(reject(&missing));
        drop(_profile_closure_negatives);

        // A different trusted public winner changes the closure plan and the
        // comparison rows.
        let _profile_wrong_plan = zkie_engine::profile::Scope::new("harness.negative_wrong_plan");
        let mut wrong_output = output_plan.clone();
        wrong_output.expected_argmax[0] =
            (wrong_output.expected_argmax[0] + 1) % wrong_output.valid as u32;
        let wrong_trusted = OutputHeadTrustedInputs {
            activation: trusted.activation,
            output: wrong_output.clone(),
        };
        let wrong_plan = build_output_head_hidden_plan(
            shape,
            &hidden_setup,
            &table.reader(),
            &wrong_trusted,
        )
        .unwrap();
        assert!(verify_output_head_lin_hidden(
            shape,
            &wrong_plan,
            &wrong_trusted,
            &table.reader(),
            &hidden_proof
        )
        .is_err());
        drop(_profile_wrong_plan);

        // The canonical hidden encoding rejects a trailing byte and a
        // truncated tail against the closure-bearing plan.
        let _profile_encoding = zkie_engine::profile::Scope::new("harness.negative_encoding");
        let mut trailing = hidden_bytes.clone();
        trailing.push(0);
        assert!(OutputHeadLinProof::decode(&hidden_plan, &trailing).is_err());
        assert!(verify_output_head_lin_hidden_bytes(
            shape,
            &hidden_plan,
            &trusted,
            &table.reader(),
            &trailing
        )
        .is_err());
        let mut truncated = hidden_bytes.clone();
        truncated.truncate(truncated.len() - 1);
        assert!(OutputHeadLinProof::decode(&hidden_plan, &truncated).is_err());
        assert!(verify_output_head_lin_hidden_bytes(
            shape,
            &hidden_plan,
            &trusted,
            &table.reader(),
            &truncated
        )
        .is_err());
        // Version gate on the real staged envelope: v4 is unsupported and a
        // newer unknown version cannot decode; neither falls back to a
        // single-stage tail.
        let session_bytes = hidden_proof.session.encode().unwrap();
        let mut old_version = zkie_engine::primitives::codec::CanonicalWriter::new();
        old_version.put_u64(4);
        old_version.put_bytes(&session_bytes);
        let old_version = old_version.into_bytes();
        assert_eq!(
            OutputHeadLinProof::decode(&hidden_plan, &old_version).unwrap_err(),
            zkie_core::error::Error::UnsupportedVersion
        );
        assert!(verify_output_head_lin_hidden_bytes(
            shape,
            &hidden_plan,
            &trusted,
            &table.reader(),
            &old_version
        )
        .is_err());
        let mut new_version = zkie_engine::primitives::codec::CanonicalWriter::new();
        new_version.put_u64(6);
        new_version.put_bytes(&session_bytes);
        assert_eq!(
            OutputHeadLinProof::decode(&hidden_plan, &new_version.into_bytes()).unwrap_err(),
            zkie_core::error::Error::InvalidEncoding
        );

        // Per-source outer negatives on the real staged proof. Each mutation
        // reaches the final verifier, not only a module-level decoder. The
        // default candidate is two stages per group, so each group carries one
        // switch; the three-stage comparison uses the same code path.
        let mut tampered = clone_proof(&hidden_proof);
        if let ProximityTail::Stages(proof) = &mut tampered.session.shared.tail {
            proof.switches[0].declared_root.0[0] ^= 1;
        } else {
            panic!("the shared group is staged");
        }
        assert!(reject(&tampered));

        let mut tampered = clone_proof(&hidden_proof);
        if let ProximityTail::Stages(proof) = &mut tampered.session.second.tail {
            proof.switches[0].declared_root.0[0] ^= 1;
        } else {
            panic!("the own group is staged");
        }
        assert!(reject(&tampered));

        let mut tampered = clone_proof(&hidden_proof);
        if let ProximityTail::Stages(proof) = &mut tampered.session.second.tail {
            proof.switches[0].target.main_rows += 1;
        } else {
            panic!("the own group is staged");
        }
        assert!(reject(&tampered));

        let mut tampered = clone_proof(&hidden_proof);
        if let ProximityTail::Stages(proof) = &mut tampered.session.second.tail {
            proof.switches[0].member_openings[0].columns[0].symbols[0] = Fp::from_u64(1);
        } else {
            panic!("the own group is staged");
        }
        assert!(reject(&tampered));

        let mut tampered = clone_proof(&hidden_proof);
        if let ProximityTail::Stages(proof) = &mut tampered.session.second.tail {
            proof.switches[0].fresh_opening.columns[0].symbols[0] = Fp::from_u64(1);
        } else {
            panic!("the own group is staged");
        }
        assert!(reject(&tampered));

        let mut tampered = clone_proof(&hidden_proof);
        if let ProximityTail::Stages(proof) = &mut tampered.session.shared.tail {
            proof.switches[0].old_mask_opening = Some(proof.switches[0].fresh_opening.clone());
        } else {
            panic!("the shared group is staged");
        }
        assert!(reject(&tampered));

        let mut tampered = clone_proof(&hidden_proof);
        if let ProximityTail::Stages(proof) = &mut tampered.session.second.tail {
            proof.switches[0].old_mask_opening = Some(proof.switches[0].fresh_opening.clone());
        } else {
            panic!("the own group is staged");
        }
        assert!(reject(&tampered));

        let mut tampered = clone_proof(&hidden_proof);
        if let ProximityTail::Stages(proof) = &mut tampered.session.second.tail {
            proof.base[0] ^= 1;
        } else {
            panic!("the own group is staged");
        }
        assert!(reject(&tampered));

        // A missing switch leaves the stage claim pending; a duplicated stage
        // changes the sequence. Both are rejected before the pending claim can
        // be used.
        let mut tampered = clone_proof(&hidden_proof);
        if let ProximityTail::Stages(proof) = &mut tampered.session.second.tail {
            proof.switches.pop();
        } else {
            panic!("the own group is staged");
        }
        assert!(reject(&tampered));

        let mut tampered = clone_proof(&hidden_proof);
        if let ProximityTail::Stages(proof) = &mut tampered.session.second.tail {
            let duplicate = proof.switches[0].clone();
            proof.switches.push(duplicate);
        } else {
            panic!("the own group is staged");
        }
        assert!(reject(&tampered));

        // A real three-stage proof has two own switches; swapping them changes
        // the final verifier's sequence and must be rejected. The default
        // two-stage proof has one switch, so this row runs only with the
        // three-stage comparison list.
        if env::var_os("ZKIE_SESSION_THREE_STAGE").is_some() {
            let mut tampered = clone_proof(&hidden_proof);
            if let ProximityTail::Stages(proof) = &mut tampered.session.second.tail {
                assert!(proof.switches.len() >= 2);
                proof.switches.swap(0, 1);
            } else {
                panic!("the own group is staged");
            }
            assert!(reject(&tampered));
        }

        // The final seal is bound to the staged base transcript.
        let mut tampered = clone_proof(&hidden_proof);
        tampered.session.second.seal.0[0] ^= 1;
        assert!(reject(&tampered));

        // Swapping the two groups tails mixes opening sources across the
        // trusted plans and cannot verify.
        let mut tampered = clone_proof(&hidden_proof);
        std::mem::swap(&mut tampered.session.shared.tail, &mut tampered.session.second.tail);
        assert!(reject(&tampered));

        // A changed W root is rejected by the first group trusted roots.
        let mut tampered = clone_proof(&hidden_proof);
        tampered.session.first.group.entries[0].root.0[0] ^= 1;
        assert!(reject(&tampered));
        drop(_profile_encoding);
        drop(_profile_negatives);
        zkie_engine::profile::dump();
    }

    fs::remove_dir_all(&link_dir).ok();
    if env::var_os("ZKIE_SAMPLE_COUNTS").is_some() {
        let (accepts, rejects) = zkie_engine::pcs::request::field::sampling_counts();
        println!("sample_accepts={accepts}");
        println!("sample_rejects={rejects}");
    }
    fs::remove_dir_all(&work_dir).ok();
}
