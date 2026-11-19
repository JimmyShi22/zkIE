//! Real-size projection consumer closure on the production hidden LIN session.
//!
//! The fixture builds a 1024 by 1024 projection weight table, commits a real
//! activation, computes the actual output rows, and runs the production
//! diagnostic hidden LIN prover and verifier on the same table. The claims are
//! the row weights and output rows from the projection claim core.

use std::cell::RefCell;

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
use zkie_ops::proof::column_session::fp_from_signed_word;
use zkie_ops::proof::hidden_send_session::{
    prove_hidden_lin, verify_hidden_lin, ActivationShape,
    CommittedActivation, HiddenLinPlan, HiddenOutputShape, HiddenProximityPlan, HiddenSendPlan,
    HiddenTableProver, WShape,
};
use zkie_ops::proof::projection_claim::{ProjectionShape, check_projection_reduction, projection_claims};

const POSITIONS: u32 = 2;
const HIDDEN: u32 = 1024;
const MATRIX_ROWS: u64 = 1024;
const K_LO: u32 = 0;
const COLUMNS: u64 = 1024;
const TABLE_ROWS: u64 = 1024;
const TABLE_LOG_N: u32 = 12;
const TABLE_ELL: u64 = 2048;
const TABLE_R: u64 = 8;
const CODE_LOG_N: u32 = 14;
const CODE_ELL: u64 = 8192;
const CODE_R: u64 = 4;
const SALT: usize = 40;
const GROUP: u64 = 7;
const ORACLE_BASE: u64 = 20;
const TABLE_CONFIGURATION: Digest = Digest([0x21; 32]);
const ACTIVATION_CONFIGURATION: Digest = Digest([0x31; 32]);
const W_CONFIGURATION: Digest = Digest([0x51; 32]);
const W_RUN: Digest = Digest([0x41; 32]);
const GU_ORACLE: u64 = 10;
const OMEGA_ORACLE: u64 = 11;
const GU_GROUP: u64 = 5;
const SEED: Digest = Digest([0x71; 32]);

struct Table {
    table: StreamingTable,
    source: RefCell<VecTableSource>,
    reader: IndexReader,
    hiding: zkie_engine::pcs::request::read::HidingState,
}

impl TableRowSource for Table {
    fn rows(&self) -> u64 {
        self.source.borrow().rows()
    }

    fn row(&mut self, index: u64, message: &mut [Fp], tail: &mut [Fp]) -> zkie_core::error::Result<()> {
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

fn w_shape() -> WShape {
    WShape::new(COLUMNS, code(), 1, POSITIONS, SALT, 100, 1, W_RUN, W_CONFIGURATION).unwrap()
}

fn tree_plan(tree_number: u64, leaf_class: u32, main_rows: u64) -> RuntimeTreePlan {
    RuntimeTreePlan {
        tree_number,
        leaf_class,
        log_n: CODE_LOG_N,
        log_n_zk: CODE_LOG_N,
        parts: vec![RuntimeTreePart {
            main_symbols: main_rows * E_DEGREE as u64,
            mask_symbols: 0,
        }],
        salt_bytes: SALT,
        run: Digest([0x61; 32]),
        configuration: Digest([0x62; 32]),
    }
}

fn gu_shape() -> GuShape {
    GuShape {
        oracle: GU_ORACLE,
        group: GU_GROUP,
        tree: tree_plan(300, 2, 1),
        part: 0,
        rows_high: true,
        u: GuOracleShape {
            code: code(),
            main_rows: 1,
            main_deg: E_DEGREE as u32,
            mask_rows: 0,
            mask_code: None,
            answers: 1,
        },
        index: table_shape(),
    }
}

fn omega_shape() -> OmegaShapePlan {
    OmegaShapePlan {
        oracle: OMEGA_ORACLE,
        group: GU_GROUP,
        tree: tree_plan(301, 2, 1),
        u: GuOracleShape {
            code: code(),
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

fn mask_tree(tree_number: u64) -> MaskTreePlan {
    MaskTreePlan {
        tree_number,
        leaf_class: 3,
        log_n: CODE_LOG_N,
        mask_rows: 1,
        salt_bytes: SALT,
        run: Digest([0x63; 32]),
        configuration: Digest([0x64; 32]),
    }
}

fn table_messages() -> Vec<Vec<Fp>> {
    let shape = table_shape();
    let rows = shape.rows() as usize;
    let ell = shape.code.ell as usize;
    let mut messages = vec![vec![Fp::ZERO; ell]; rows];
    for (row, message) in messages.iter_mut().enumerate() {
        for entry in 0..COLUMNS as usize {
            let word = ((row as i64 * 13 + entry as i64 * 5) % 11) - 5;
            message[entry] = fp_from_signed_word(word as u64);
        }
        for entry in shape.plain() as usize..ell {
            message[entry] = Fp::ZERO;
        }
    }
    messages
}

fn table() -> Table {
    let messages = table_messages();
    let tails = vec![vec![Fp::ZERO; TABLE_R as usize]; TABLE_ROWS as usize];
    let mut source = VecTableSource::new(messages, tails).unwrap();
    let salts: Vec<Vec<u8>> = (0..table_code().n())
        .map(|position| vec![(position as u8) ^ 0x5A; SALT])
        .collect();
    let table = commit_table_streaming(
        &mut source,
        table_shape(),
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
        hiding: zkie_engine::pcs::request::read::HidingState::new(
            reader.root,
            1,
            reader.shape.code.r as u64,
        ),
        reader,
    }
}

fn activation_rows() -> Vec<Vec<i64>> {
    (0..POSITIONS as usize)
        .map(|position| {
            (0..HIDDEN as usize)
                .map(|index| (((position * 31 + index * 7) % 13) as i64) - 6)
                .collect()
        })
        .collect()
}

fn activation() -> CommittedActivation {
    let salts: Vec<Vec<u8>> = (0..POSITIONS)
        .map(|position| vec![(position as u8).wrapping_mul(7) ^ 0x3C; SALT])
        .collect();
    CommittedActivation::commit_signed(
        ActivationShape::new(POSITIONS, HIDDEN, SALT).unwrap(),
        &activation_rows(),
        &salts,
        ACTIVATION_CONFIGURATION,
    )
    .unwrap()
}

fn weights() -> Vec<Vec<Fp3>> {
    activation_rows()
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|value| Fp3::from_base(fp_from_signed_word(value as u64)))
                .collect()
        })
        .collect()
}

fn output_values() -> Vec<Vec<Fp3>> {
    let messages = table_messages();
    let weights = weights();
    let mut output = vec![vec![Fp3::ZERO; COLUMNS as usize]; POSITIONS as usize];
    for position in 0..POSITIONS as usize {
        for column in 0..COLUMNS as usize {
            let table_row = column >> K_LO as usize;
            let mut sum = Fp3::ZERO;
            for row in 0..MATRIX_ROWS as usize {
                let index = ((column & ((1usize << K_LO) - 1)) + row * (1 << K_LO)) as usize;
                let value = Fp3::from_base(messages[table_row][index]);
                sum = sum.add(&weights[position][row].mul(&value));
            }
            output[position][column] = sum;
        }
    }
    output
}

fn plan(activation: &CommittedActivation, table: &Table) -> HiddenLinPlan {
    let send = HiddenSendPlan {
        activation: activation.shape(),
        activation_configuration: ACTIVATION_CONFIGURATION,
        activation_root: activation.root(),
        activation_source:
            zkie_ops::proof::hidden_send_session::ActivationSource::Commitment,
        table_shape: table_shape(),
        table_identity: reader_table_identity(&table.reader).unwrap(),
        matrix_rows: MATRIX_ROWS,
        k_lo: K_LO,
        columns: COLUMNS,
        group: GROUP,
        oracle_base: ORACLE_BASE,
        openings_per_oracle: 2,
        w: w_shape(),
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
    };
    HiddenLinPlan {
        norm: None,
        send,
        output: HiddenOutputShape {
            positions: POSITIONS,
            columns: COLUMNS,
        },
        masks: vec![Vec::new(); POSITIONS as usize],
        weights: weights(),
        proximity: HiddenProximityPlan {
            gu: gu_shape(),
            omega: omega_shape(),
            code: code(),
            mask_code: code(),
            shared_mask_tree: mask_tree(200),
            own_mask_tree: mask_tree(201),
            shared_fresh: 50,
            own_fresh: 51,
            column_draws: 2,
            column_budget: 4,
            ledger: zkie_ops::proof::hidden_send_session::ReadLedgerCell::new(4 as u64),
            column_work_bits: 0,
            shared_base_positions: 2,
            own_base_positions: 2,
            own_deferred_samples: true,
            shared_stages: Vec::new(),
            own_stages: Vec::new(),
            planner_checked: false,
            lin_check: None,
        },
        output_relation: None,
        closure: None,
    }
}

fn run() -> (HiddenLinPlan, HiddenLinMessages, Table, Vec<Vec<Fp3>>) {
    let activation = activation();
    let mut table = table();
    let output = output_values();
    let plan = plan(&activation, &table);
    plan.send.validate().expect("send plan");
    if let Some(reason) = plan.proximity.fault(&plan.send, None, None) {
        panic!("proximity fault: {reason}");
    }
    plan.validate().expect("lin plan");
    let (messages, witness) =
        prove_hidden_lin(&plan, &activation, &mut table, &output, SEED).unwrap();
    let mut verifier_plan = plan;
    verifier_plan.send.expected_roots = Some(witness.send.roots.clone());
    (verifier_plan, messages, table, output)
}

type HiddenLinMessages = zkie_ops::proof::hidden_send_session::HiddenLinMessages;

#[test]
fn real_size_projection_consumer_closes_on_the_hidden_lin_session() {
    let (plan, messages, table, output) = run();
    verify_hidden_lin(&plan, table.reader(), &output, &messages).unwrap();
    let shape = ProjectionShape {
        matrix_rows: MATRIX_ROWS,
        k_lo: K_LO,
        columns: COLUMNS,
        positions: POSITIONS,
    };
    let flat_activation: Vec<Fp3> = weights().into_iter().flatten().collect();
    let flat_output: Vec<Fp3> = output.iter().flatten().copied().collect();
    let claims = projection_claims(shape, &flat_activation, &flat_output).unwrap();
    check_projection_reduction(
        shape,
        &claims.weights,
        &table_messages()
            .iter()
            .map(|row| row.iter().map(|value| Fp3::from_base(*value)).collect())
            .collect::<Vec<Vec<Fp3>>>(),
        &claims.output,
    )
    .unwrap();
}

#[test]
fn changed_output_is_rejected_by_the_session() {
    let (plan, messages, table, mut output) = run();
    output[0][0] = output[0][0].add(&Fp3::from_base(fp_from_u64(1)));
    assert!(verify_hidden_lin(&plan, table.reader(), &output, &messages).is_err());
}

#[test]
fn changed_coefficients_are_rejected_by_the_session() {
    let (mut plan, messages, table, output) = run();
    plan.weights[0][0] = plan.weights[0][0].add(&Fp3::from_base(fp_from_u64(1)));
    assert!(verify_hidden_lin(&plan, table.reader(), &output, &messages).is_err());
}

#[test]
fn changed_table_identity_is_rejected() {
    let activation = activation();
    let mut table = table();
    let output = output_values();
    let mut plan = plan(&activation, &table);
    plan.send.table_identity.0[0] ^= 1;
    assert!(prove_hidden_lin(&plan, &activation, &mut table, &output, SEED).is_err());
}
