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
        MaskEntryAllocation, WShape,
    };
    use zkie_ops::proof::model_commitment::verify_shared_trace_outputs;
    use zkie_ops::proof::output_head_lin::diagnostic::{
        output_head_claims, prepare_output_head_trace, prove_output_head_lin_prepared,
        verify_output_head_lin_prepared, verify_output_head_lin_request_bytes, OutputHeadTrace,
    };
    use zkie_ops::proof::output_head_lin::{
        build_output_head_hidden_plan, build_output_head_output_plan, output_head_shape,
        prove_output_head_lin_hidden, select_output_head_step, verify_output_head_lin_hidden,
        OutputHeadLinProof, OutputHeadLinSetup, OutputHeadTrustedInputs, OUTPUT_HEAD_ASSET_PATH,
        OUTPUT_HEAD_COLUMNS, OUTPUT_HEAD_HIDDEN, OUTPUT_HEAD_K_LO, OUTPUT_HEAD_POSITIONS,
    };
    use zkie_ops::proof::plan::PlanSource;
    use zkie_ops::proof::session_baseline::SESSION_PROTOCOL_IDENTITY;

    use zkie::lm_head_key::{
        build_lm_head_holder, limited_key_binding_parameters, LmHeadAssetSource,
        LmHeadKeyRandomness,
    };
    use zkie::commitment_ledger::{CommitmentKind, CommitmentLedger, LedgerPolicy};
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
    const OWN_FRESH: u64 = 51;
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
            mask_rows: 2,
            ..mask_tree(OWN_MASK_TREE)
        },
        shared_fresh: SHARED_FRESH,
        own_fresh: OWN_FRESH,
        column_draws: COLUMN_DRAWS,
        column_budget: COLUMN_DRAWS,
        column_work_bits: COLUMN_WORK_BITS,
        shared_base_positions: 2,
        own_base_positions: 2,
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
    let setup = OutputHeadLinSetup {
        asset_path: OUTPUT_HEAD_ASSET_PATH.to_string(),
        send: HiddenSendPlan {
            activation,
            activation_configuration: ACTIVATION_CONFIGURATION,
            activation_root: Digest([0x71; 32]),
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
    };

    // X and Y come only from the verified trace outputs.
    let started = Instant::now();
    let outputs = verify_shared_trace_outputs(&public, &trace, &trace.tokens, &plan).unwrap();
    println!("trace_verify_ms={}", started.elapsed().as_millis());
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

    let started = Instant::now();
    let (proof, witness) =
        prove_output_head_lin_prepared(&request, shape, &setup, &mut table).unwrap();
    let prove_ms = started.elapsed().as_millis();
    let bytes = proof.encode().unwrap();
    assert_eq!(bytes.len(), proof.encoded_len().unwrap());
    println!("output_head_step={}", step.step);
    println!("request_proof_bytes={}", bytes.len());
    println!("prove_ms={prove_ms}");

    let mut verifier_setup = setup.clone();
    verifier_setup.expected_activation_root = Some(proof.session.first.activation.root);
    verifier_setup.expected_roots = Some(witness.send.roots.clone());

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
    assert_eq!(prepared.proof_bytes, bytes.len());
    assert_eq!(prepared.outcome.eval_rows.len(), OUTPUT_HEAD_POSITIONS);
    assert!(prepared.outcome.shared_base.fold_main);
    assert!(prepared.outcome.own_base.fold_main);
    assert!(prepared.outcome.shared_base.base_linear);
    assert!(prepared.outcome.own_base.base_linear);

    // The byte-level entry reruns the trace check and decodes the canonical
    // request encoding against the same merged plan.
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
    let merged = HiddenLinPlan {
        send: decode_send,
        output: HiddenOutputShape {
            positions: OUTPUT_HEAD_POSITIONS as u32,
            columns: OUTPUT_HEAD_COLUMNS as u64,
        },
        weights: claims.weights,
        masks: claims.masks,
        proximity: verifier_setup.proximity.clone(),
        output_relation: None,
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
        // The formal section runs on a fresh read-only source, as a separate
        // prover process would. The diagnostic section above has already
        // consumed the bounded read budget of the shared source.
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
        let output_plan = build_output_head_output_plan(&plan, &trace.argmax).unwrap();
        let trusted = OutputHeadTrustedInputs {
            activation: *activation_commitment.commitment(),
            output: output_plan.clone(),
        };
        // The authoritative commitment ledger is reserved before the hidden
        // proof starts. X and the table root are known here; the W roots are
        // proof values and are reserved as soon as the prover returns them.
        // The ledger is a test-run directory, so a repeated formal run starts
        // from a fresh ledger instead of reusing a spent commitment.
        let ledger_dir = work_dir.join("commitment-ledger");
        let ledger = CommitmentLedger::initialize(&ledger_dir, LedgerPolicy::default())
            .expect("initialize commitment ledger");
        let request_id = digest_bytes(b"formal-output-head-request");
        let activation_root = activation_commitment.root();
        ledger
            .record_commitment(
                CommitmentKind::Activation,
                activation_root,
                activation_root,
            )
            .expect("record the activation commitment");
        let table_identity = reader_table_identity(&reader).unwrap();
        ledger
            .record_commitment(CommitmentKind::Table, table_identity, reader.root)
            .expect("record the table commitment");
        let activation_claim = ledger
            .reserve(
                CommitmentKind::Activation,
                activation_root,
                request_id,
                OUTPUT_HEAD_POSITIONS as u64,
            )
            .expect("reserve the activation openings");
        let table_claim = ledger
            .reserve(
                CommitmentKind::Table,
                table_identity,
                request_id,
                COLUMN_DRAWS as u64,
            )
            .expect("reserve the table openings");
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
        let hidden_prove_ms = started.elapsed().as_millis();
        let mut weight_claims = Vec::new();
        for weight_root in &hidden_witness.send.roots {
            ledger
                .record_commitment(CommitmentKind::Weight, *weight_root, *weight_root)
                .expect("record a W commitment");
            weight_claims.push(
                ledger
                    .reserve(
                        CommitmentKind::Weight,
                        *weight_root,
                        request_id,
                        setup.send.openings_per_oracle as u64,
                    )
                    .expect("reserve the W openings"),
            );
        }
        // The sampled mask message is one committed oracle; its one-time
        // entries, including every Eval mask, are charged to this reservation.
        let mask_root = hidden_proof
            .session
            .first
            .index_mask
            .as_ref()
            .expect("the hidden path commits the mask message")
            .root;
        ledger
            .record_commitment(CommitmentKind::Mask, mask_root, mask_root)
            .expect("record the mask commitment");
        let mask_claim = ledger
            .reserve(
                CommitmentKind::Mask,
                mask_root,
                request_id,
                setup
                    .index_mask
                    .as_ref()
                    .expect("the mask plan")
                    .openings as u64,
            )
            .expect("reserve the mask openings");
        // The shown-map message of the hidden Dot sumcheck is a commitment of
        // this run. Its own tree is opened by the own-group BASE round at the
        // drawn positions, so the reservation charges those logical openings.
        let dot_root = hidden_proof
            .session
            .first
            .dot
            .as_ref()
            .expect("the hidden path commits the shown-map message")
            .root;
        ledger
            .record_commitment(CommitmentKind::Dot, dot_root, dot_root)
            .expect("record the shown-map commitment");
        let dot_claim = ledger
            .reserve(
                CommitmentKind::Dot,
                dot_root,
                request_id,
                setup.proximity.own_base_positions as u64,
            )
            .expect("reserve the shown-map openings");
        activation_claim.succeed().expect("finish the activation attempt");
        table_claim.succeed().expect("finish the table attempt");
        mask_claim.succeed().expect("finish the mask attempt");
        dot_claim.succeed().expect("finish the shown-map attempt");
        for claim in weight_claims {
            claim.succeed().expect("finish a W attempt");
        }
        let second_request = digest_bytes(b"formal-output-head-second-request");
        assert_eq!(
            ledger
                .reserve(
                    CommitmentKind::Activation,
                    activation_root,
                    second_request,
                    OUTPUT_HEAD_POSITIONS as u64,
                )
                .unwrap_err(),
            zkie_core::error::Error::StateRollbackSuspected
        );
        assert_eq!(
            ledger
                .reserve(
                    CommitmentKind::Table,
                    table_identity,
                    second_request,
                    COLUMN_DRAWS as u64,
                )
                .unwrap_err(),
            zkie_core::error::Error::StateRollbackSuspected
        );
        assert_eq!(
            ledger
                .reserve(
                    CommitmentKind::Mask,
                    mask_root,
                    second_request,
                    setup.index_mask.as_ref().unwrap().openings as u64,
                )
                .unwrap_err(),
            zkie_core::error::Error::StateRollbackSuspected
        );
        assert_eq!(
            ledger
                .reserve(
                    CommitmentKind::Dot,
                    dot_root,
                    second_request,
                    setup.proximity.own_base_positions as u64,
                )
                .unwrap_err(),
            zkie_core::error::Error::StateRollbackSuspected
        );
        assert_eq!(
            ledger
                .reserve(
                    CommitmentKind::Activation,
                    activation_root,
                    request_id,
                    OUTPUT_HEAD_POSITIONS as u64,
                )
                .unwrap_err(),
            zkie_core::error::Error::LeafBudgetExhausted
        );
        let ledger_state = ledger.read_state().expect("read commitment ledger");
        println!(
            "commitment_ledger_openings={}",
            ledger_state.total_logical_openings().unwrap()
        );
        println!(
            "commitment_ledger_table_reuse={}",
            ledger_state.total_table_reuse().unwrap()
        );
        println!("hidden_prove_ms={hidden_prove_ms}");
        println!(
            "hidden_proof_bytes={}",
            hidden_proof.encode().unwrap().len()
        );
        assert_eq!(closure.argmax, output_plan.expected_argmax);
        let trace_output_rows: Vec<Vec<i64>> = (0..OUTPUT_HEAD_POSITIONS)
            .map(|position| {
                request.inputs.output
                    [position * OUTPUT_HEAD_COLUMNS..(position + 1) * OUTPUT_HEAD_COLUMNS]
                    .to_vec()
            })
            .collect();
        assert_eq!(
            zkie_ops::proof::output_head_lin::witness_output_rows(&hidden_witness).unwrap(),
            trace_output_rows
        );
        let mut hidden_setup = setup.clone();
        hidden_setup.send.expected_dot_root = Some(dot_root);
        let hidden_plan =
            build_output_head_hidden_plan(shape, &hidden_setup, &table.reader(), &trusted).unwrap();
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
    }

    fs::remove_dir_all(&link_dir).ok();
    fs::remove_dir_all(&work_dir).ok();
}
