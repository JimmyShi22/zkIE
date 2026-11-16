//! Release evidence for the complete key chain over the registered candidate
//! instance. Ignored by default; run explicitly with `ZKIE_KEY_EV` pointing at
//! the round's evidence directory (the instance lives below it). The driver
//! rebuilds the table keeper from the committed private randomness, derives
//! the public key plan, proves first + shared + second with a private run seed,
//! writes the complete canonical bundle to disk, reads it back and verifies it
//! through the independent public entry only.

use std::cell::RefCell;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use zkie_core::artifact::{digest_bytes, digest_to_hex};
use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_engine::pcs::request::field::{Fp, Fp3};
use zkie_engine::pcs::request::foldeq::reader_table_identity;
use zkie_engine::pcs::request::read::{DurableColumnBudget, HidingState};
use zkie_engine::pcs::request::table::{
    commit_table_streaming, ColumnOpening, IndexReader, StreamingTable, TableRowSource,
    VecTableSource,
};
use zkie_models_gpt2::plan::{
    asset_table_params, derive_table_plan, TABLE_PLAN_CANDIDATE,
};
use zkie_ops::proof::asset_table::{decode_asset_table_randomness_set, AssetTableMeta};
use zkie_ops::proof::column_session::{column_ledger_dir, fp_from_signed_word};
use zkie_ops::proof::hidden_send_session::{
    prove_key, production_key_first_plan, verify_key, HiddenTableProver, KeyPlanInputs,
    KeyRunRandomness,
};
use zkie_ops::proof::table_range::{
    production_table_range, range_bit_values, range_key_values, range_multiplicities,
    IntegerRangeTable,
};

struct Table {
    table: StreamingTable,
    source: RefCell<VecTableSource>,
    reader: IndexReader,
    hiding: HidingState,
    budget: DurableColumnBudget,
}

impl TableRowSource for Table {
    fn rows(&self) -> u64 {
        self.source.borrow().rows()
    }
    fn row(&mut self, index: u64, message: &mut [Fp], tail: &mut [Fp]) -> zkie_core::Result<()> {
        self.source.borrow_mut().row(index, message, tail)
    }
}

impl HiddenTableProver for Table {
    fn reader(&self) -> &IndexReader {
        &self.reader
    }
    fn open_columns(&mut self, positions: &[u64]) -> zkie_core::Result<Vec<ColumnOpening>> {
        let mut source = self.source.borrow_mut();
        self.table.open_columns(&mut *source, positions)
    }
    fn hiding_state(&mut self) -> Option<&mut HidingState> {
        Some(&mut self.hiding)
    }
    fn durable_column_budget(&mut self) -> Option<&mut DurableColumnBudget> {
        Some(&mut self.budget)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct MemoryKib {
    hwm: u64,
    rss: u64,
}

fn parse_kib(text: &str) -> u64 {
    text.trim()
        .trim_end_matches(" kB")
        .trim()
        .parse()
        .unwrap_or(0)
}

/// The process-wide high-water mark and the current resident set. The high
/// water mark is cumulative over the process and never decreases; the resident
/// set is the live memory at this point.
fn process_memory_kib() -> MemoryKib {
    let text = fs::read_to_string("/proc/self/status").unwrap_or_default();
    let mut memory = MemoryKib::default();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            memory.hwm = parse_kib(rest);
        } else if let Some(rest) = line.strip_prefix("VmRSS:") {
            memory.rss = parse_kib(rest);
        }
    }
    memory
}

#[test]
#[ignore]
fn the_complete_key_chain_over_the_registered_instance() {
    let Some(ev) = std::env::var_os("ZKIE_KEY_EV") else {
        return;
    };
    let ev = PathBuf::from(ev);
    // The registered instance lives in its registration trace; the run's
    // results are written to this trace directory.
    let instance = std::env::var_os("ZKIE_KEY_INSTANCE")
        .map(PathBuf::from)
        .unwrap_or_else(|| ev.join("instance/gpt2-seq16"));
    let commit = instance.join("commit");

    let config: zkie::provider::ProviderConfig =
        bincode::deserialize(&fs::read(commit.join("public/provider-config.bin")).unwrap()).unwrap();
    let metas: Vec<AssetTableMeta> =
        bincode::deserialize(&fs::read(commit.join("public/asset-tables.bin")).unwrap()).unwrap();
    let derived = derive_table_plan(&TABLE_PLAN_CANDIDATE).unwrap();
    let params = asset_table_params().unwrap().pop().unwrap();
    let meta = metas
        .iter()
        .find(|meta| meta.path == params.path)
        .expect("the registered table");
    assert_eq!(meta.identity, derived.table_identity);
    assert_eq!(meta.configuration, derived.table_configuration);
    assert_eq!(meta.shape, derived.table_shape);

    let randomness = decode_asset_table_randomness_set(
        &fs::read(commit.join("private/asset-table-randomness.bin")).unwrap(),
        std::slice::from_ref(&params),
    )
    .unwrap();
    let salts = randomness[0].randomness.salts.clone();
    let tails = randomness[0].randomness.tails.clone();

    let weight_path = Path::new(&config.asset_root).join("gpt2/weights/L0_k_w_i32.bin");
    let weight_bytes = fs::read(&weight_path).unwrap();
    let values: Vec<i64> = weight_bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as i64)
        .collect();
    let rows = meta.shape.rows() as usize;
    let row_width = params.layout.row_width as usize;
    let ell = meta.shape.code.ell as usize;
    let mut messages = vec![vec![Fp::ZERO; ell]; rows];
    for row in 0..rows {
        for column in 0..row_width {
            messages[row][column] = fp_from_signed_word(values[row * row_width + column] as u64);
        }
    }
    let mut source = VecTableSource::new(messages.clone(), tails.clone()).unwrap();
    let table = commit_table_streaming(
        &mut source,
        meta.shape,
        meta.configuration,
        &salts,
        64,
        u64::MAX,
    )
    .unwrap();
    assert_eq!(table.reader.root, meta.root);
    assert_eq!(table.reader.record, meta.record);
    let reader = table.reader.clone();

    let ledger = column_ledger_dir(Path::new(&config.asset_root), meta.root);
    let count_path = ledger.join("columns.count");
    assert!(count_path.exists(), "the commit initialized the ledger");
    // The batch size is this key stage's one-time columns; the life limit is
    // the instance's total (one-time plus the declared later batch).
    let budget = DurableColumnBudget::open(
        ledger.clone(),
        count_path.clone(),
        derived.one_time_columns,
        derived.total_columns,
    )
    .unwrap();
    let count_before = budget.count().unwrap();
    let mut keeper = Table {
        table,
        source: RefCell::new(source),
        reader: reader.clone(),
        hiding: HidingState::new(reader.root, 1, derived.total_columns),
        budget,
    };

    // The declared range is a public plan parameter: it comes from the fixed
    // public configuration and the verifier rebuilds the same declaration.
    let range_plan = production_table_range().unwrap();
    // The key plan fixes the matrix order: its matrix rows are the record
    // places and its matrix columns are the records, and a key number's low
    // coordinates are the matrix columns. The asset file stores the records in
    // its own order, so the enumerated entry `row * columns + column` is the
    // cell of record `column` at place `row`: the transpose of the file order.
    let matrix_rows = rows;
    let columns = (meta.shape.rows() as usize) << TABLE_PLAN_CANDIDATE.k_lo;
    assert_eq!(columns, row_width);
    assert_eq!(values.len(), matrix_rows * columns);
    let mut entries = Vec::with_capacity(values.len());
    for row in 0..matrix_rows {
        for column in 0..columns {
            entries.push(Fp3::from_base(fp_from_signed_word(
                values[column * row_width + row] as u64,
            )));
        }
    }
    let keys = range_key_values(&range_plan, &entries).unwrap();
    let counts = range_multiplicities(&range_plan, &keys).unwrap();
    let bits = range_bit_values(&range_plan, &counts).unwrap();
    let lookup_table = IntegerRangeTable::new(&range_plan, 1 << 20).unwrap();

    let plan = production_key_first_plan(&KeyPlanInputs {
        // The plan binds the registered instance's reader identity: its
        // committed root, record, configuration and shape. The model-level
        // table identity is a separate public fact and does not include the
        // instance root.
        table_identity: reader_table_identity(&reader).unwrap(),
        table_configuration: derived.table_configuration,
        table_shape: derived.table_shape,
        matrix_rows: meta.shape.rows(),
        k_lo: TABLE_PLAN_CANDIDATE.k_lo,
        columns: meta.shape.rows() << TABLE_PLAN_CANDIDATE.k_lo,
        total_columns: derived.one_time_columns,
        lin_check: derived.lin_check,
        range: range_plan.clone(),
    })
    .unwrap();
    let summary = plan.summary();
    // The run seed comes from the operating system generator; it stays private
    // and is never logged, written or returned beside the proof.
    let seed = KeyRunRandomness::Private.seed().unwrap();

    let prove_start = Instant::now();
    let (bytes, parts) = prove_key(
        &plan,
        &reader,
        &mut keeper,
        &lookup_table,
        &keys,
        &bits,
        seed,
    )
    .unwrap();
    let prove_ms = prove_start.elapsed().as_millis();
    let count_after = keeper.budget.count().unwrap();
    assert_eq!(count_after, count_before + derived.one_time_columns);

    // The complete canonical bundle is kept on disk beside the trace; it is a
    // binary artifact and is not committed. The verification below reads the
    // same bytes back and uses only the public plan, the reader, the range
    // lookup table and the bundle.
    let bundle_path = ev.join("key-proof.bin");
    fs::write(&bundle_path, &bytes).unwrap();
    let disk_bytes = fs::read(&bundle_path).unwrap();
    assert_eq!(disk_bytes, bytes, "the disk bundle is the proved bundle");
    let bundle_sha256 = digest_bytes(&disk_bytes);

    let memory_before_verify = process_memory_kib();
    let verify_start = Instant::now();
    let verify_plan = plan.clone();
    let accepted = verify_key(&verify_plan, &reader, &lookup_table, &disk_bytes).unwrap();
    let verify_ms = verify_start.elapsed().as_millis();
    let memory_after_verify = process_memory_kib();
    assert_eq!(accepted.plan_digest(), plan.statement_digest());
    assert_eq!(accepted.table_root(), meta.root);
    // The verification consumes the plan's read ledger once; the prover's
    // durable column budget is not charged again.
    assert_eq!(keeper.budget.count().unwrap(), count_after);
    // A repeated verification on the same plan finds the read budget spent.
    assert!(verify_key(&verify_plan, &reader, &lookup_table, &disk_bytes).is_err());
    // Byte negatives on the real bundle read from disk.
    let mut truncated = disk_bytes.clone();
    truncated.pop();
    assert!(verify_key(&plan.clone(), &reader, &lookup_table, &truncated).is_err());
    let mut trailing = disk_bytes.clone();
    trailing.push(0);
    assert!(verify_key(&plan.clone(), &reader, &lookup_table, &trailing).is_err());
    let mut bad_version = disk_bytes.clone();
    bad_version[0] ^= 1;
    assert!(verify_key(&plan.clone(), &reader, &lookup_table, &bad_version).is_err());

    let summary_json = serde_json::json!({
        "plan_digest": digest_to_hex(&summary.plan_digest),
        "table_identity": digest_to_hex(&summary.table_identity),
        "table_configuration": digest_to_hex(&summary.table_configuration),
        "matrix_rows": summary.matrix_rows,
        "k_lo": summary.k_lo,
        "columns": summary.columns,
        "shared_code": {
            "log_n": summary.shared_code.log_n,
            "ell": summary.shared_code.ell,
            "r": summary.shared_code.r,
        },
        "shared_mask_code": {
            "log_n": summary.shared_mask_code.log_n,
            "ell": summary.shared_mask_code.ell,
            "r": summary.shared_mask_code.r,
        },
        "own_code": {
            "log_n": summary.own_code.log_n,
            "ell": summary.own_code.ell,
            "r": summary.own_code.r,
        },
        "own_mask_code": {
            "log_n": summary.own_mask_code.log_n,
            "ell": summary.own_mask_code.ell,
            "r": summary.own_mask_code.r,
        },
        "shared_stages": summary.shared_stages,
        "own_stages": summary.own_stages,
        "shared_base_positions": summary.shared_base_positions,
        "own_base_positions": summary.own_base_positions,
        "index_rows": summary.index_rows,
        "lookup_bits_oracle": summary.lookup_bits_oracle,
        "lookup_mask_oracle": summary.lookup_mask_oracle,
        "w_oracle": summary.w_oracle,
        "index_oracle": summary.index_oracle,
        "group": summary.group,
        "bits_bat": summary.bits_bat,
        "mask_bat": summary.mask_bat,
        "index_bat": summary.index_bat,
        "group_fresh": summary.group_fresh,
        "group_mask_tree": summary.group_mask_tree,
        "shared_fresh": summary.shared_fresh,
        "shared_mask_tree": summary.shared_mask_tree,
        "own_fresh": summary.own_fresh,
        "own_mask_tree": summary.own_mask_tree,
        "gu_oracle": summary.gu_oracle,
        "omega_oracle": summary.omega_oracle,
        "column_draws": summary.column_draws,
        "column_budget": summary.column_budget,
    });
    let result = serde_json::json!({
        "plan_digest": digest_to_hex(&summary.plan_digest),
        "table_identity": digest_to_hex(&summary.table_identity),
        "table_configuration": digest_to_hex(&summary.table_configuration),
        "table_root": digest_to_hex(&meta.root),
        "record_len": meta.record.y.len(),
        "proof_bytes": disk_bytes.len(),
        "first_bytes": parts.first,
        "shared_bytes": parts.shared,
        "second_bytes": parts.second,
        "prove_ms": prove_ms,
        "verify_ms": verify_ms,
        "vmhwm_process_kib": memory_after_verify.hwm,
        "verify_vmhwm_before_kib": memory_before_verify.hwm,
        "verify_vmhwm_after_kib": memory_after_verify.hwm,
        "verify_hwm_delta_kib": memory_after_verify
            .hwm
            .saturating_sub(memory_before_verify.hwm),
        "verify_vmrss_before_kib": memory_before_verify.rss,
        "verify_vmrss_after_kib": memory_after_verify.rss,
        "verify_rss_delta_kib": memory_after_verify
            .rss
            .saturating_sub(memory_before_verify.rss),
        "ledger_before": count_before,
        "ledger_after": count_after,
        "instance_total_columns": derived.total_columns,
        "declared_later_batches": derived.declared_later_batches,
        "later_batch_columns": derived.later_batch_columns,
        "read_ledger_spent": true,
        "seed_source": "operating system generator (private)",
        "bundle": {
            "path": bundle_path.display().to_string(),
            "bytes": disk_bytes.len(),
            "sha256": digest_to_hex(&bundle_sha256),
        },
        "summary": summary_json,
    });
    let json = serde_json::to_string_pretty(&result).unwrap();
    fs::write(ev.join("key-chain-evidence.json"), &json).unwrap();
    eprintln!("KEY CHAIN EVIDENCE {json}");
}
