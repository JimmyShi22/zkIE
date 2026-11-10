//! Isolated column verification component demonstration.
//!
//! This module builds a small encoding table deterministically from a slice of
//! an authenticated asset opening, commits it with the existing table
//! primitives, and exercises four component layers:
//!
//! 1. authenticated column openings against the table's fixed column root;
//! 2. G_u SEND registration and the membership row check;
//! 3. a deterministic full reconstruction comparison of the opened columns
//!    against the encoded rows;
//! 4. the public FOLD input check and output computation, ending at the
//!    deliberately fail-closed next-stage boundary.
//!
//! It is not a proof verifier, and its result must not be used to accept a
//! proof. The G_u root is a self-reported notice; the G_u relation rows are
//! constructed from already verified columns, but the membership check only
//! checks the oracle and group, not the target. The FOLD input row is a
//! self-reported fixture and is not derived from the asset table. The FOLD
//! output's next stage is not connected, so the FOLD layer is component-level
//! only and no statement is made that the full FOLD gate passed.
//!
//! Three roots are kept separate and recorded: the authenticated asset's meta
//! root, the encoding table's column-tree root, and the G_u notice root. No
//! equality between them is claimed.
//!
//! The encoding table is a test foundation, not a production commit path. It
//! materializes a four-row, eight-column table in memory; the asset itself is
//! read only through the verified opening and is never materialized. The
//! message layout, zero padding, zero tails, deterministic salts and the
//! slice mapping are demonstration choices and are replaceable. The table
//! configuration, the G_u notice root, the runtime-tree starts and the FOLD
//! fresh root are demo placeholders; the G_u notice root in particular is
//! self-reported and not derived from the asset or the table root.
//!
//! The asset-to-FOLD witness derivation interface remains unclosed: there is
//! no public construction path from an authenticated asset table to the FOLD
//! input coefficients, mask, target or challenges. The FOLD row here is an
//! explicitly labelled self-reported fixture. Adding such a derivation would
//! be a new interface and is not part of this demonstration.
//!
//! The base field has `p < 2^64` and `Fp3` is a cubic extension, so
//! `|Fp3| = p^3 < 2^192`. This demonstration gives no soundness bound: the
//! transcript is a candidate-internal, test-only ordering, and the checks are
//! local component and full-value comparisons, not a proximity or soundness
//! proof. Hiding and single-shot semantics are not connected. The production
//! verify entry is unchanged.

use zkie_core::common::field::{PrimeCharacteristicRing, PrimeField64};
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::binding::BindingCandidate;
use zkie_engine::pcs::request::chain::{
    Chain, ChainMessage, KIND_ALGEBRAIC,
};
use zkie_engine::pcs::request::code::{build_code_plan, encode_row_fp, CodeParams};
use zkie_engine::pcs::request::field::{Fp, Fp3, E_DEGREE};
use zkie_engine::pcs::request::fold::{
    fold_mask_shape_fault, fold_output, FoldGroupLayout, FoldInputRow, FoldMemberLayout,
    FoldOutput, FoldShape,
};
use zkie_engine::pcs::request::gu::{
    GuMembership, GuNotice, GuOracleShape, GuRegistry, GuShape,
};
use zkie_engine::pcs::request::params::{
    LEAF_SALT_MIN_BYTES, REQUEST_CHAIN_LABEL,
};
use zkie_engine::pcs::request::read::{
    draw_positions, unique_sorted, HidingState,
};
use zkie_engine::pcs::request::relation::{
    column_plan_rows, GuRelationRow,
};
use zkie_engine::pcs::request::runtime_tree::{RuntimeTreePart, RuntimeTreePlan};
use zkie_engine::pcs::request::table::{
    commit_table, verify_columns, ColumnOpening, CommittedTable, IndexReader, IndexShape,
    TableDistances,
};

use crate::proof::model_commitment::{
    verify_model_anchor, AssetMeta, AssetOpening, ModelCommitmentPublic,
};
use crate::proof::session_baseline::TrustedModel;

const SESSION_SCHEMA_VERSION: u64 = 1;
const STAGE_TABLE: u64 = 1;
const STAGE_GU: u64 = 2;
const STAGE_FOLD: u64 = 3;

const TABLE_CONFIGURATION: Digest = Digest([0x5a; 32]);
const GU_NOTICE_ROOT: Digest = Digest([0x47; 32]);
const FOLD_FRESH_ROOT: Digest = Digest([0x46; 32]);

const TABLE_LOG_N: u32 = 3;
const TABLE_ELL: u64 = 4;
const TABLE_R: u64 = 1;
const TABLE_V: u32 = 2;
const TABLE_ANSWERS: u32 = 1;
const TABLE_SALT_BYTES: usize = LEAF_SALT_MIN_BYTES;
const DRAW_COUNT: usize = 4;

const GU_ORACLE: u64 = 11;
const GU_GROUP: u64 = 5;
const GU_TREE_NUMBER: u64 = 7;
const GU_MAIN_ROWS: u64 = 4;
const GU_TREE_RUN: Digest = Digest([0x31; 32]);
const GU_TREE_CONFIGURATION: Digest = Digest([0x32; 32]);

const FOLD_MAIN_ROWS: u64 = 2;

/// A self-contained table built from the authenticated asset slice, together
/// with the messages and tails used to build it.
pub struct AssetEncodingTable {
    pub table: CommittedTable,
    pub reader: IndexReader,
    pub asset_root: Digest,
    pub shape: IndexShape,
    pub messages: Vec<Vec<Fp>>,
    pub tails: Vec<Vec<Fp>>,
}

/// The result of authenticated column opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnAuthentication {
    pub logical_draws: Vec<u64>,
    pub positions: Vec<u64>,
    pub rows: usize,
    pub openings: Vec<ColumnOpening>,
    pub columns: Vec<Fp>,
}

/// The result of the G_u registration component.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuRegistration {
    pub shape: GuShape,
    pub notice: GuNotice,
    pub registry: GuRegistry,
    pub membership: GuMembership,
    pub rows: Vec<GuRelationRow>,
}

/// The result of the deterministic full-table comparison.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableComparison {
    pub cells: usize,
    pub matches: bool,
}

/// The public FOLD component result. `next_stage_closed` records the current
/// fail-closed boundary; it is not a successful connection to the next stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldComponent {
    pub shape: FoldShape,
    pub layout: FoldGroupLayout,
    pub input: FoldInputRow,
    pub output: FoldOutput,
    pub next_stage_closed: bool,
}

/// The four component layers and the three separately recorded roots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnSessionOutcome {
    pub asset_root: Digest,
    pub table_root: Digest,
    pub gu_root: Digest,
    pub shape: IndexShape,
    pub logical_draws: Vec<u64>,
    pub opened_positions: Vec<u64>,
    pub column_cells: usize,
    pub gu_rows: usize,
    pub column_authentication: bool,
    pub gu_registration: bool,
    pub merge_targets: bool,
    pub full_table_comparison: bool,
    pub fold_component: bool,
    pub fold_next_stage_closed: bool,
}

/// Converts a signed fixed-point word stored as two's-complement `u64` into a
/// base-field element. Asset and opening values are signed integers, so an
/// unsigned reduction would change their field representation.
pub fn fp_from_signed_word(value: u64) -> Fp {
    let signed = value as i64;
    let magnitude = Fp3::from_u64(signed.unsigned_abs()).c[0];
    if signed < 0 {
        Fp3::ZERO.sub(&Fp3::from_base(magnitude)).c[0]
    } else {
        magnitude
    }
}

fn fp3_words(value: &Fp3) -> [u64; 3] {
    [
        value.c[0].as_canonical_u64(),
        value.c[1].as_canonical_u64(),
        value.c[2].as_canonical_u64(),
    ]
}

fn push_digest(message: &mut ChainMessage, digest: &Digest) {
    for word in digest_words(digest) {
        message.push_u64(word);
    }
}

fn digest_words(digest: &Digest) -> [u64; 4] {
    let mut words = [0u64; 4];
    for (index, chunk) in digest.0.chunks_exact(8).enumerate() {
        words[index] = u64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
    }
    words
}

fn push_fp3(message: &mut ChainMessage, value: &Fp3) {
    for word in fp3_words(value) {
        message.push_u64(word);
    }
}

fn push_bytes(message: &mut ChainMessage, bytes: &[u8]) {
    message.push_u64(bytes.len() as u64);
    for byte in bytes {
        message.push_u64(*byte as u64);
    }
}

/// The demonstration table shape. It has four rows and eight columns, one
/// plain message entry per row, three answer pads, and one tail entry.
pub fn demo_index_shape() -> Result<IndexShape> {
    let code = build_code_plan(TABLE_LOG_N, TABLE_ELL, TABLE_R)?;
    let shape = IndexShape {
        code,
        v: TABLE_V,
        answers: TABLE_ANSWERS,
        salt_bytes: TABLE_SALT_BYTES,
        distances: TableDistances {
            radius: 0,
            later_radius: 0,
            m_once: 0,
            m_later: 0,
        },
    };
    if index_shape_fault(&shape).is_some() {
        return Err(Error::InvalidConfig);
    }
    Ok(shape)
}

fn index_shape_fault(shape: &IndexShape) -> Option<&'static str> {
    zkie_engine::pcs::request::table::index_fault(shape).or_else(|| {
        zkie_engine::pcs::request::table::tree_shape_fault(shape)
    })
}

/// Builds the encoding table from the first `plain` values of the first
/// `2^v` rows of an authenticated asset opening. The mapping is deterministic:
///
/// - table row `r` takes asset row `r`, columns `0 .. plain` (the first
///   `plain` = `ell - 3 * answers` values);
/// - message entries `plain .. ell` are zero pads, including the answer pads;
/// - each tail is `r` zeros;
/// - the salt of column `p` is the byte `0xA5 ^ (p mod 256)`, repeated
///   `salt_bytes` times;
/// - the code is Reed-Solomon with the shape's `log_n`, `ell` and `r`.
///
/// The trusted anchor and the opening are verified first. The asset file is
/// never materialized; only the slice values in the opening are used.
pub fn build_authenticated_encoding_table(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    asset_path: &str,
    opening: &AssetOpening,
) -> Result<AssetEncodingTable> {
    verify_model_anchor(
        public,
        trusted.anchor_id,
        trusted.model_id,
        trusted.parameters_digest,
        trusted.asset_specs,
    )?;
    let meta: &AssetMeta = public
        .assets
        .iter()
        .find(|asset| asset.path == asset_path)
        .ok_or(Error::AssetMismatch)?;
    if opening.asset_id != meta.asset_id
        || opening.start != meta.start
        || opening.root != meta.root
        || opening.row_count != meta.rows
        || opening.rows.len() != meta.rows as usize
    {
        return Err(Error::StatementMismatch);
    }
    for (index, row) in opening.rows.iter().enumerate() {
        if row.row != index as u64 || row.values.len() != meta.row_width {
            return Err(Error::ShapeMismatch);
        }
    }
    public.verify_opening(opening)?;

    let shape = demo_index_shape()?;
    let rows = usize::try_from(shape.rows_checked()?).map_err(|_| Error::ResourceLimit)?;
    let plain = usize::try_from(shape.plain()).map_err(|_| Error::ResourceLimit)?;
    let code_len = usize::try_from(shape.code.ell).map_err(|_| Error::ResourceLimit)?;
    let tail_len = usize::try_from(shape.code.r).map_err(|_| Error::ResourceLimit)?;
    let columns = usize::try_from(shape.code.n()).map_err(|_| Error::ResourceLimit)?;
    if meta.row_width < plain || opening.rows.len() < rows {
        return Err(Error::ShapeMismatch);
    }

    let mut messages = vec![vec![Fp::ZERO; code_len]; rows];
    let tails = vec![vec![Fp::ZERO; tail_len]; rows];
    for (row, message) in messages.iter_mut().enumerate() {
        let source = opening.rows.get(row).ok_or(Error::IncompleteArtifact)?;
        for (column, entry) in message.iter_mut().take(plain).enumerate() {
            let value = *source
                .values
                .get(column)
                .ok_or(Error::ShapeMismatch)?;
            *entry = fp_from_signed_word(value);
        }
    }
    let salts: Vec<Vec<u8>> = (0..columns)
        .map(|position| vec![(position as u8) ^ 0xA5; shape.salt_bytes])
        .collect();

    let table = commit_table(shape, TABLE_CONFIGURATION, &messages, &tails, &salts)?;
    let reader = table.reader.clone();
    Ok(AssetEncodingTable {
        table,
        reader,
        asset_root: meta.root,
        shape,
        messages,
        tails,
    })
}

/// Opens the unique drawn columns with the existing hiding charge and verifies
/// every opening against the table's fixed column root. The logical draw list
/// is kept in its drawn order; the physical opening list is sorted and
/// deduplicated.
pub fn authenticate_columns(
    table: &AssetEncodingTable,
    logical_draws: &[u64],
) -> Result<ColumnAuthentication> {
    if logical_draws.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let positions = unique_sorted(logical_draws);
    let rows = usize::try_from(table.shape.rows_checked()?).map_err(|_| Error::ResourceLimit)?;
    let mut state = HidingState::new(table.reader.root, 1, positions.len() as u64);
    let charge = state.begin_charge(table.reader.root, 1, positions.len())?;
    let openings = table.table.open_columns_charged(&charge, &positions)?;
    let columns = verify_columns(&table.reader, &positions, &openings)?;
    Ok(ColumnAuthentication {
        logical_draws: logical_draws.to_vec(),
        positions,
        rows,
        openings,
        columns,
    })
}

impl ColumnAuthentication {
    /// The verified columns in the same `(position, column)` shape expected by
    /// the G_u column row builder.
    pub fn columns_by_position(&self) -> Vec<(u64, Vec<Fp3>)> {
        self.positions
            .iter()
            .enumerate()
            .map(|(index, position)| {
                let start = index * self.rows;
                let end = start + self.rows;
                let column = self.columns[start..end]
                    .iter()
                    .map(|value| Fp3::from_base(*value))
                    .collect();
                (*position, column)
            })
            .collect()
    }
}

/// Re-encodes every row and compares every opened cell with the expected
/// codeword symbol. This is a deterministic full-value comparison on the small
/// in-memory table; it is not a proximity test.
pub fn check_full_table(
    table: &AssetEncodingTable,
    authentication: &ColumnAuthentication,
) -> Result<TableComparison> {
    let rows = usize::try_from(table.shape.rows_checked()?).map_err(|_| Error::ResourceLimit)?;
    let expected = authentication
        .positions
        .len()
        .checked_mul(rows)
        .ok_or(Error::ResourceLimit)?;
    if authentication.columns.len() != expected || authentication.rows != rows {
        return Err(Error::ShapeMismatch);
    }
    let mut matches = true;
    let mut cells = 0usize;
    for (index, position) in authentication.positions.iter().enumerate() {
        if *position >= table.shape.code.n() {
            return Err(Error::InvalidIndex);
        }
        for row in 0..rows {
            let codeword = encode_row_fp(
                &table.shape.code,
                table.messages.get(row).ok_or(Error::ShapeMismatch)?,
                table.tails.get(row).ok_or(Error::ShapeMismatch)?,
            )?;
            let expected = *codeword
                .get(*position as usize)
                .ok_or(Error::InvalidIndex)?;
            let actual = authentication.columns[index * rows + row];
            if expected != actual {
                matches = false;
            }
            cells += 1;
        }
    }
    Ok(TableComparison { cells, matches })
}

/// The trusted shape for the demonstration G_u oracle. The G_u code, main
/// rows, tree, part and indexed-table shape are fixed test values; only the
/// notice root is received.
pub fn demo_gu_shape(index: IndexShape) -> Result<GuShape> {
    let main_symbols = GU_MAIN_ROWS
        .checked_mul(E_DEGREE as u64)
        .ok_or(Error::ResourceLimit)?;
    let tree = RuntimeTreePlan {
        tree_number: GU_TREE_NUMBER,
        leaf_class: 0,
        log_n: TABLE_LOG_N,
        log_n_zk: TABLE_LOG_N,
        parts: vec![RuntimeTreePart {
            main_symbols,
            mask_symbols: 0,
        }],
        salt_bytes: TABLE_SALT_BYTES,
        run: GU_TREE_RUN,
        configuration: GU_TREE_CONFIGURATION,
    };
    let u = GuOracleShape {
        code: build_code_plan(TABLE_LOG_N, TABLE_ELL, TABLE_R)?,
        main_rows: GU_MAIN_ROWS,
        main_deg: E_DEGREE as u32,
        mask_rows: 0,
        mask_code: None,
        answers: 1,
    };
    let shape = GuShape {
        oracle: GU_ORACLE,
        group: GU_GROUP,
        tree,
        part: 0,
        rows_high: false,
        u,
        index,
    };
    if shape.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    Ok(shape)
}

fn demo_gu_notice(shape: &GuShape) -> GuNotice {
    GuNotice {
        root: GU_NOTICE_ROOT,
        oracle: shape.oracle,
        tree_number: shape.tree.tree_number,
        part: shape.part,
    }
}

/// Registers the self-reported G_u notice and joins the isolated group. The
/// rows are built from the provided fixture columns, which must be one
/// strictly increasing entry per unique drawn position. The membership check
/// checks the oracle and group only; it does not evaluate the target.
pub fn register_gu_with_columns(
    table: &AssetEncodingTable,
    authentication: &ColumnAuthentication,
    eq_gamma: &[Fp3],
    fixture_columns: &[(u64, Vec<Fp3>)],
) -> Result<GuRegistration> {
    if eq_gamma.len() != authentication.rows {
        return Err(Error::ShapeMismatch);
    }
    let shape = demo_gu_shape(table.shape)?;
    let notice = demo_gu_notice(&shape);
    let mut registry = GuRegistry::new();
    registry.receive(&shape, &notice)?;
    let group = zkie_engine::pcs::request::fold::VerifiedGroup {
        group: shape.group,
        members: Vec::new(),
        fresh: None,
        code: Some(shape.u.code),
        mask_code: shape.u.mask_code,
    };
    let mut membership = GuMembership::new(&group, &shape, &registry)?;
    membership.join(&shape, &registry)?;
    let rows = column_plan_rows(
        &shape,
        &registry,
        authentication.logical_draws.len(),
        table.shape.code.log_n,
        &authentication.logical_draws,
        fixture_columns,
        eq_gamma,
    )?;
    for row in &rows {
        membership.check_row(row)?;
    }
    Ok(GuRegistration {
        shape,
        notice,
        registry,
        membership,
        rows,
    })
}

/// Registers G_u using the verified columns themselves.
pub fn register_gu(
    table: &AssetEncodingTable,
    authentication: &ColumnAuthentication,
    eq_gamma: &[Fp3],
) -> Result<GuRegistration> {
    let columns = authentication.columns_by_position();
    register_gu_with_columns(table, authentication, eq_gamma, &columns)
}

/// Recomputes each G_u column target from the full re-encoded table and the
/// fixed equation weights, then compares it with the target in the registered
/// row. The membership check is separate and does not evaluate targets.
pub fn check_merge_targets(
    table: &AssetEncodingTable,
    authentication: &ColumnAuthentication,
    registration: &GuRegistration,
    eq_gamma: &[Fp3],
) -> Result<bool> {
    let rows = usize::try_from(table.shape.rows_checked()?).map_err(|_| Error::ResourceLimit)?;
    if eq_gamma.len() != rows || registration.rows.len() != authentication.logical_draws.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut matches = true;
    for (row, draw) in registration.rows.iter().zip(&authentication.logical_draws) {
        if *draw >= table.shape.code.n() {
            return Err(Error::InvalidIndex);
        }
        let mut expected = Fp3::ZERO;
        for (index, weight) in eq_gamma.iter().enumerate() {
            let codeword = encode_row_fp(
                &table.shape.code,
                table.messages.get(index).ok_or(Error::ShapeMismatch)?,
                table.tails.get(index).ok_or(Error::ShapeMismatch)?,
            )?;
            let value = Fp3::from_base(
                *codeword
                    .get(*draw as usize)
                    .ok_or(Error::InvalidIndex)?,
            );
            expected = expected.add(&weight.mul(&value));
        }
        if row.target != expected {
            matches = false;
        }
    }
    Ok(matches)
}

/// Fixed test-only equation weights for the four table rows.
pub fn demo_eq_gamma(rows: usize) -> Result<Vec<Fp3>> {
    if rows == 0 {
        return Err(Error::InvalidConfig);
    }
    Ok((0..rows)
        .map(|index| Fp3::from_u64(index as u64 + 1))
        .collect())
}

/// The demonstration FOLD shape. The main code is the table's code; the mask
/// code is a valid three-entry code at the same depth. The shape is a public
/// component input, not a production group layout.
pub fn demo_fold_shape(index: IndexShape) -> Result<FoldShape> {
    let mask_code: CodeParams = build_code_plan(index.code.log_n, 3, 1)?;
    let shape = FoldShape {
        code: index.code,
        mask_code,
        k: 1,
        h: 0,
        mask_leaf_class: 0,
    };
    if fold_mask_shape_fault(&shape).is_some() {
        return Err(Error::InvalidConfig);
    }
    Ok(shape)
}

fn demo_fold_layout(shape: &FoldShape) -> Result<FoldGroupLayout> {
    let layout = FoldGroupLayout {
        group: GU_GROUP,
        k: shape.k,
        h: shape.h,
        members: vec![FoldMemberLayout {
            oracle: GU_ORACLE,
            main_rows: FOLD_MAIN_ROWS,
            mask_rows: 0,
            kept: false,
        }],
    };
    if layout.fault(shape).is_some() {
        return Err(Error::InvalidConfig);
    }
    Ok(layout)
}

fn demo_fold_input(shape: &FoldShape) -> Result<FoldInputRow> {
    let rows = 1usize.checked_shl(shape.k).ok_or(Error::ResourceLimit)?;
    let main_len = rows
        .checked_mul(usize::try_from(shape.code.ell).map_err(|_| Error::ResourceLimit)?)
        .ok_or(Error::ResourceLimit)?;
    let mask_len = 1usize
        .checked_shl(shape.h)
        .ok_or(Error::ResourceLimit)?
        .checked_mul(
            usize::try_from(shape.mask_code.ell).map_err(|_| Error::ResourceLimit)?,
        )
        .ok_or(Error::ResourceLimit)?;
    Ok(FoldInputRow {
        group: GU_GROUP,
        k: shape.k,
        h: shape.h,
        members: vec![GU_ORACLE],
        coefficients: (0..main_len)
            .map(|index| Fp3::from_u64(index as u64 + 1))
            .collect(),
        mask_coefficients: (0..mask_len)
            .map(|index| Fp3::from_u64(index as u64 + 0x20))
            .collect(),
        kept: Vec::new(),
        target: Fp3::from_u64(0x2e),
    })
}

/// Builds the public FOLD shape, layout and self-reported input row. The input
/// row is not derived from the asset table.
pub fn demo_fold_material(index: IndexShape) -> Result<(FoldShape, FoldGroupLayout, FoldInputRow)> {
    let shape = demo_fold_shape(index)?;
    let layout = demo_fold_layout(&shape)?;
    let input = demo_fold_input(&shape)?;
    input.check(&shape, &layout)?;
    Ok((shape, layout, input))
}

/// Computes the public FOLD output for the fixed, self-reported fixture row
/// and the supplied challenges. The output's next stage is deliberately
/// fail-closed.
pub fn check_fold_component_with_input(
    shape: FoldShape,
    layout: FoldGroupLayout,
    input: FoldInputRow,
    eps: Fp3,
    gamma: Fp3,
    sum: Fp3,
) -> Result<FoldComponent> {
    let output = fold_output(
        &shape,
        &layout,
        &input,
        1,
        FOLD_FRESH_ROOT,
        GU_TREE_NUMBER,
        eps,
        sum,
        &[gamma],
        input.target,
    )?;
    let next_stage_closed = output.next_stage().is_err();
    Ok(FoldComponent {
        shape,
        layout,
        input,
        output,
        next_stage_closed,
    })
}

/// Builds the self-reported input row and computes the FOLD output for the
/// supplied challenges. Callers that need transcript ordering first call
/// `demo_fold_material`, absorb the row, and then draw the challenges.
pub fn check_fold_component(
    index: IndexShape,
    eps: Fp3,
    gamma: Fp3,
    sum: Fp3,
) -> Result<FoldComponent> {
    let (shape, layout, input) = demo_fold_material(index)?;
    check_fold_component_with_input(shape, layout, input, eps, gamma, sum)
}

fn begin_chain(binding: &BindingCandidate) -> Result<Chain> {
    Chain::begin(
        REQUEST_CHAIN_LABEL,
        &binding.parameters_digest,
        &binding.statement_bytes,
        &binding.record_bytes,
        None,
    )
}

fn absorb_table(chain: &mut Chain, table: &AssetEncodingTable) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(SESSION_SCHEMA_VERSION);
    message.push_u64(STAGE_TABLE);
    push_digest(&mut message, &table.asset_root);
    push_digest(&mut message, &table.reader.root);
    push_digest(&mut message, &table.reader.configuration);
    for value in [
        table.shape.code.log_n as u64,
        table.shape.code.ell,
        table.shape.code.r,
        table.shape.v as u64,
        table.shape.answers as u64,
        table.shape.salt_bytes as u64,
    ] {
        message.push_u64(value);
    }
    message.push_u64(table.reader.record.y.len() as u64);
    for value in &table.reader.record.y {
        push_fp3(&mut message, value);
    }
    let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn absorb_gu(chain: &mut Chain, registration: &GuRegistration) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(SESSION_SCHEMA_VERSION);
    message.push_u64(STAGE_GU);
    push_digest(&mut message, &registration.notice.root);
    message.push_u64(registration.shape.oracle);
    message.push_u64(registration.shape.group);
    message.push_u64(registration.shape.tree.tree_number);
    message.push_u64(registration.shape.part as u64);
    push_bytes(&mut message, &registration.shape.encode());
    let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn absorb_fold(chain: &mut Chain, input: &FoldInputRow) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(SESSION_SCHEMA_VERSION);
    message.push_u64(STAGE_FOLD);
    message.push_u64(input.group);
    message.push_u64(input.k as u64);
    message.push_u64(input.h as u64);
    message.push_u64(input.members.len() as u64);
    for member in &input.members {
        message.push_u64(*member);
    }
    message.push_u64(input.coefficients.len() as u64);
    for value in &input.coefficients {
        push_fp3(&mut message, value);
    }
    message.push_u64(input.mask_coefficients.len() as u64);
    for value in &input.mask_coefficients {
        push_fp3(&mut message, value);
    }
    push_fp3(&mut message, &input.target);
    let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// Runs the four component layers in transcript order: the table roots and
/// record are fixed before the column draws; the verified columns are then
/// used for the G_u registration and merge-target control; the full table
/// comparison follows; the self-reported FOLD row is fixed before the FOLD
/// challenges. A failed real component returns an error. The FOLD next stage
/// is expected to remain fail-closed.
pub fn verify_column_session(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    binding: &BindingCandidate,
    asset_path: &str,
    opening: &AssetOpening,
) -> Result<ColumnSessionOutcome> {
    let table = build_authenticated_encoding_table(public, trusted, asset_path, opening)?;
    let mut chain = begin_chain(binding)?;
    absorb_table(&mut chain, &table)?;
    let logical_draws = draw_positions(&mut chain, DRAW_COUNT, table.shape.code.log_n)?;
    if logical_draws.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let authentication = authenticate_columns(&table, &logical_draws)?;
    let eq_gamma = demo_eq_gamma(authentication.rows)?;
    let registration = register_gu(&table, &authentication, &eq_gamma)?;
    absorb_gu(&mut chain, &registration)?;
    let gu_registration = registration
        .rows
        .iter()
        .all(|row| registration.membership.check_row(row).is_ok());
    let merge_targets = check_merge_targets(&table, &authentication, &registration, &eq_gamma)?;
    let comparison = check_full_table(&table, &authentication)?;

    let (fold_shape, fold_layout, fold_input) = demo_fold_material(table.shape)?;
    absorb_fold(&mut chain, &fold_input)?;
    let eps = chain.draw_e_nonzero()?;
    let gamma = chain.draw_e()?;
    let fold = check_fold_component_with_input(
        fold_shape,
        fold_layout,
        fold_input,
        eps,
        gamma,
        registration
            .rows
            .first()
            .map(|row| row.target)
            .unwrap_or(Fp3::ZERO),
    )?;

    if !gu_registration || !merge_targets || !comparison.matches || !fold.next_stage_closed {
        return Err(Error::InvalidProof);
    }
    Ok(ColumnSessionOutcome {
        asset_root: table.asset_root,
        table_root: table.reader.root,
        gu_root: registration.notice.root,
        shape: table.shape,
        logical_draws,
        opened_positions: authentication.positions.clone(),
        column_cells: authentication.columns.len(),
        gu_rows: registration.rows.len(),
        column_authentication: true,
        gu_registration,
        merge_targets,
        full_table_comparison: comparison.matches,
        fold_component: true,
        fold_next_stage_closed: fold.next_stage_closed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use zkie_core::proof_header::PROOF_VERSION_WORD;
    use zkie_engine::pcs::request::binding::{assemble, BindingParameters, BindingStatement};
    use zkie_engine::pcs::request::params::RequestParams;

    use crate::proof::model_commitment::{
        commit_assets, AssetStore, ModelAssetSpec,
    };

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "zkie-column-session-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    struct Fixture {
        root: PathBuf,
        public: ModelCommitmentPublic,
        opening: AssetOpening,
        specs: Vec<ModelAssetSpec>,
        anchor_id: Digest,
        model_id: Digest,
        parameters_digest: Digest,
        binding: BindingCandidate,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn binding(public: &ModelCommitmentPublic) -> BindingCandidate {
        let request_parameters =
            RequestParams::default_with_code(TABLE_LOG_N, TABLE_ELL, TABLE_R).unwrap();
        let plan_digest = Digest([0x33; 32]);
        let protocol_identity = Digest([0x44; 32]);
        let parameters = BindingParameters::with_plan_parts(
            PROOF_VERSION_WORD,
            protocol_identity,
            &request_parameters,
            &[plan_digest],
        )
        .unwrap();
        let statement = BindingStatement::new(
            PROOF_VERSION_WORD,
            protocol_identity,
            b"case".to_vec(),
            plan_digest,
            public.anchor.anchor_id,
        )
        .unwrap();
        assemble(&parameters, &statement, None).unwrap()
    }

    fn fixture() -> Fixture {
        let root = temp_dir();
        fs::write(
            root.join("table_i32.bin"),
            [1i32, -2, 3, -4]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let specs = vec![ModelAssetSpec {
            asset_id: 0,
            path: "table_i32.bin".into(),
            element_count: 4,
            row_width: 1,
            encoding: "i32-le".into(),
        }];
        let commitment =
            commit_assets(&root, Digest([1; 32]), Digest([2; 32]), &specs).unwrap();
        let public = ModelCommitmentPublic::from_commitment(&commitment);
        let opening = AssetStore::open(&commitment)
            .unwrap()
            .open_all()
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let anchor_id = public.anchor.anchor_id;
        let model_id = Digest([1; 32]);
        let parameters_digest = Digest([2; 32]);
        let binding = binding(&public);
        Fixture {
            root,
            public,
            opening,
            specs,
            anchor_id,
            model_id,
            parameters_digest,
            binding,
        }
    }

    fn trusted(fixture: &Fixture) -> TrustedModel<'_> {
        TrustedModel {
            anchor_id: fixture.anchor_id,
            model_id: fixture.model_id,
            parameters_digest: fixture.parameters_digest,
            asset_specs: &fixture.specs,
        }
    }

    fn table(fixture: &Fixture) -> AssetEncodingTable {
        let trusted = trusted(fixture);
        build_authenticated_encoding_table(
            &fixture.public,
            &trusted,
            "table_i32.bin",
            &fixture.opening,
        )
        .unwrap()
    }

    fn verify(fixture: &Fixture) -> Result<ColumnSessionOutcome> {
        let trusted = trusted(fixture);
        verify_column_session(
            &fixture.public,
            &trusted,
            &fixture.binding,
            "table_i32.bin",
            &fixture.opening,
        )
    }

    #[test]
    fn session_passes_all_four_component_layers_and_records_three_roots() {
        let fixture = fixture();
        let outcome = verify(&fixture).unwrap();
        assert!(outcome.column_authentication);
        assert!(outcome.gu_registration);
        assert!(outcome.merge_targets);
        assert!(outcome.full_table_comparison);
        assert!(outcome.fold_component);
        assert!(outcome.fold_next_stage_closed);
        assert_eq!(outcome.logical_draws.len(), DRAW_COUNT);
        assert!(outcome.opened_positions.len() <= outcome.logical_draws.len());
        assert_eq!(outcome.column_cells, outcome.opened_positions.len() * 4);
        assert_eq!(outcome.gu_rows, outcome.logical_draws.len());
        assert_ne!(outcome.asset_root, outcome.table_root);
        assert_ne!(outcome.asset_root, outcome.gu_root);
        assert_ne!(outcome.table_root, outcome.gu_root);
    }

    #[test]
    fn fixed_fixture_is_reproducible() {
        let fixture = fixture();
        let first = verify(&fixture).unwrap();
        let second = verify(&fixture).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn asset_root_tamper_is_rejected_before_the_table_is_used() {
        let fixture = fixture();
        let mut opening = fixture.opening.clone();
        opening.root.0[0] ^= 1;
        let trusted = trusted(&fixture);
        assert!(build_authenticated_encoding_table(
            &fixture.public,
            &trusted,
            "table_i32.bin",
            &opening
        )
        .is_err());
    }

    #[test]
    fn column_path_tamper_is_rejected_by_authentication() {
        let fixture = fixture();
        let table = table(&fixture);
        let draws = vec![0, 1, 2, 3];
        let mut authentication = authenticate_columns(&table, &draws).unwrap();
        authentication.openings[0].path.push(Digest([9; 32]));
        assert!(verify_columns(
            &table.reader,
            &authentication.positions,
            &authentication.openings
        )
        .is_err());
        assert!(authenticate_columns(&table, &draws).is_ok());
    }

    #[test]
    fn path_valid_but_wrong_merge_is_seen_by_the_target_control() {
        let fixture = fixture();
        let table = table(&fixture);
        let draws = vec![0, 1, 2, 3];
        let authentication = authenticate_columns(&table, &draws).unwrap();
        let eq_gamma = demo_eq_gamma(authentication.rows).unwrap();
        let mut columns = authentication.columns_by_position();
        let first = &mut columns[0].1;
        first.swap(0, 1);
        let wrong = register_gu_with_columns(&table, &authentication, &eq_gamma, &columns)
            .expect("membership registration does not evaluate the target");
        assert!(wrong
            .rows
            .iter()
            .all(|row| wrong.membership.check_row(row).is_ok()));
        assert!(!check_merge_targets(&table, &authentication, &wrong, &eq_gamma).unwrap());
        let good = register_gu(&table, &authentication, &eq_gamma).unwrap();
        assert!(check_merge_targets(&table, &authentication, &good, &eq_gamma).unwrap());
    }

    #[test]
    fn gu_registration_rejects_a_wrong_oracle_or_group() {
        let fixture = fixture();
        let table = table(&fixture);
        let authentication = authenticate_columns(&table, &[0, 1, 2, 3]).unwrap();
        let eq_gamma = demo_eq_gamma(authentication.rows).unwrap();
        let registration = register_gu(&table, &authentication, &eq_gamma).unwrap();
        assert!(registration
            .rows
            .iter()
            .all(|row| registration.membership.check_row(row).is_ok()));
        let mut wrong_oracle = registration.rows[0].clone();
        wrong_oracle.oracle += 1;
        assert!(registration.membership.check_row(&wrong_oracle).is_err());
        let mut wrong_group = registration.rows[0].clone();
        wrong_group.group += 1;
        assert!(registration.membership.check_row(&wrong_group).is_err());
    }

    #[test]
    fn full_table_comparison_detects_a_wrong_opened_cell() {
        let fixture = fixture();
        let table = table(&fixture);
        let mut authentication = authenticate_columns(&table, &[0, 1, 2, 3]).unwrap();
        assert!(check_full_table(&table, &authentication).unwrap().matches);
        authentication.columns[0] = Fp3::from_u64(999).c[0];
        assert!(!check_full_table(&table, &authentication).unwrap().matches);
    }

    #[test]
    fn fold_component_checks_the_public_row_and_stops_at_the_closed_boundary() {
        let fixture = fixture();
        let table = table(&fixture);
        let shape = demo_fold_shape(table.shape).unwrap();
        assert!(fold_mask_shape_fault(&shape).is_none());
        let fold = check_fold_component(
            table.shape,
            Fp3::from_u64(7),
            Fp3::from_u64(11),
            Fp3::from_u64(13),
        )
        .unwrap();
        fold.input.check(&fold.shape, &fold.layout).unwrap();
        assert!(fold.next_stage_closed);
        assert!(fold.output.next_stage().is_err());
        let mut bad = fold.input.clone();
        bad.coefficients.pop();
        assert!(bad.check(&fold.shape, &fold.layout).is_err());
    }
}
