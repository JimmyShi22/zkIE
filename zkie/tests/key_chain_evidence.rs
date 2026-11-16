//! Release evidence for the complete key chain over the registered candidate
//! instance. Ignored by default; run explicitly with `ZKIE_KEY_EV` pointing at
//! the round's evidence directory (the instance lives below it). The driver
//! rebuilds the table keeper from the committed private randomness, derives
//! the public key plan, proves first + shared + second, encodes the proof
//! bytes, and verifies them through the independent public entry only.

use std::cell::RefCell;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::types::Digest;
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

fn vm_hwm_kib() -> u64 {
    let text = fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            return rest
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .unwrap_or(0);
        }
    }
    0
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
    let budget = DurableColumnBudget::open(
        ledger.clone(),
        count_path.clone(),
        derived.total_columns,
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
    let entries: Vec<Fp3> = values
        .iter()
        .map(|value| Fp3::from_base(fp_from_signed_word(*value as u64)))
        .collect();
    let keys = range_key_values(&range_plan, &entries).unwrap();
    let counts = range_multiplicities(&range_plan, &keys).unwrap();
    let bits = range_bit_values(&range_plan, &counts).unwrap();
    let lookup_table = IntegerRangeTable::new(&range_plan, 1 << 20).unwrap();

    let plan = production_key_first_plan(&KeyPlanInputs {
        table_identity: derived.table_identity,
        table_configuration: derived.table_configuration,
        table_shape: derived.table_shape,
        matrix_rows: meta.shape.rows(),
        k_lo: TABLE_PLAN_CANDIDATE.k_lo,
        columns: meta.shape.rows() << TABLE_PLAN_CANDIDATE.k_lo,
        total_columns: derived.total_columns,
        lin_check: derived.lin_check,
        range: range_plan.clone(),
    })
    .unwrap();
    let summary = plan.summary();
    let seed = Digest([0x6B; 32]);

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
    let hwm_prove = vm_hwm_kib();
    let count_after = keeper.budget.count().unwrap();
    assert_eq!(count_after, count_before + derived.total_columns);

    let verify_start = Instant::now();
    let verify_plan = plan.clone();
    let accepted = verify_key(&verify_plan, &reader, &lookup_table, &bytes).unwrap();
    let verify_ms = verify_start.elapsed().as_millis();
    let hwm_verify = vm_hwm_kib();
    assert_eq!(accepted.plan_digest(), plan.statement_digest());
    assert_eq!(accepted.table_root(), meta.root);
    // A repeated verification on the same plan finds the read budget spent.
    assert!(verify_key(&verify_plan, &reader, &lookup_table, &bytes).is_err());
    // Byte negatives on the real bundle.
    let mut truncated = bytes.clone();
    truncated.pop();
    assert!(verify_key(&plan.clone(), &reader, &lookup_table, &truncated).is_err());
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(verify_key(&plan.clone(), &reader, &lookup_table, &trailing).is_err());
    let mut bad_version = bytes.clone();
    bad_version[0] ^= 1;
    assert!(verify_key(&plan.clone(), &reader, &lookup_table, &bad_version).is_err());

    let result = format!(
        "{{\"plan_digest\":\"{}\",\"table_identity\":\"{}\",\"table_root\":\"{}\",\"proof_bytes\":{},\"first_bytes\":{},\"shared_bytes\":{},\"second_bytes\":{},\"prove_ms\":{},\"verify_ms\":{},\"vmhwm_prove_kib\":{},\"vmhwm_verify_kib\":{},\"ledger_before\":{},\"ledger_after\":{},\"summary\":{:?}}}",
        zkie_core::artifact::digest_to_hex(&summary.plan_digest),
        zkie_core::artifact::digest_to_hex(&summary.table_identity),
        zkie_core::artifact::digest_to_hex(&meta.root),
        bytes.len(),
        parts.first,
        parts.shared,
        parts.second,
        prove_ms,
        verify_ms,
        hwm_prove,
        hwm_verify,
        count_before,
        count_after,
        summary,
    );
    fs::write(ev.join("key-chain-evidence.json"), &result).unwrap();
    eprintln!("KEY CHAIN EVIDENCE {result}");
}
