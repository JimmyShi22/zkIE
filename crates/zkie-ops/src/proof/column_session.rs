//! Isolated column verification component demonstration.
//!
//! This module builds a small encoding table deterministically from a slice of
//! an authenticated asset opening, commits it with the existing table
//! primitives, and runs the candidate FOLDEQ session in its fixed order:
//!
//! 1. bind the table first query, derive the session gamma, and fix/send the
//!    G_u message before any column draw;
//! 2. draw the plan's column queries and authenticate the opened columns
//!    against the table's fixed column root;
//! 3. join the G_u record and verified column rows, using equation weights
//!    derived from the session gamma;
//! 4. run the deterministic full reconstruction comparison and the public
//!    FOLD component, ending at the deliberately fail-closed next-stage
//!    boundary.
//!
//! It is not a proof verifier, and its result must not be used to accept a
//! proof. The G_u root is a public message value sent into the session; the
//! membership check still checks only the oracle and group, not the target.
//! The column target control separately compares the registered target with
//! the full re-encoded table. The FOLD input row is a self-reported fixture
//! and is not derived from the asset table. The FOLD output's next stage is
//! not connected, so the FOLD layer is component-level only and no statement
//! is made that the full FOLD gate passed.
//!
//! Three roots are kept separate and recorded: the authenticated asset's meta
//! root, the encoding table's column-tree root, and the sent G_u root. No
//! equality between them is claimed.
//!
//! The encoding table is a test foundation, not a production commit path. It
//! verifies the complete asset opening row by row and the complete opening is
//! resident; only the first four rows and the first plain column participate
//! in the table. This is an expensive isolation control, not evidence that the
//! production mmap/ranged commit or memory path is closed. The message layout,
//! zero padding, zero tails, deterministic salts and the slice mapping are
//! demonstration choices and are replaceable. The table/chain configuration is
//! an independently constructed candidate parameters digest (version word,
//! candidate protocol identity, request parameters and plan identifier). The
//! typed 2b statement is built only after the reader exists and enters the
//! plan's first query before gamma; its table identity is the reader's actual
//! root, shape, configuration and record. The real fixture supplies a plan
//! identifier rebuilt from the selected static plan and the actual case id;
//! unit tests use a fixed demo input. The request id remains an explicit
//! placeholder: this does not claim a real request identity or that different
//! real requests are distinguished. The interface guarantees that the
//! supplied bytes enter the challenge prefix; it does not by itself make the
//! parameters, case or plan trusted. The G_u runtime-tree starts and the FOLD
//! fresh root are also demo placeholders.
//!
//! The asset-to-FOLD witness derivation interface remains unclosed: there is
//! no public construction path from an authenticated asset table to the FOLD
//! input coefficients, mask, target or rounds. The FOLD row here is an
//! explicitly labelled self-reported fixture; the FOLD round messages are
//! fixed component values, not a prover session. Adding such a derivation
//! would be a new interface and is not part of this demonstration.
//!
//! The base field has `p < 2^64` and `Fp3` is a cubic extension, so
//! `|Fp3| = p^3 < 2^192`. This demonstration gives no soundness bound: the
//! transcript is a candidate-internal, test-only ordering, and the checks are
//! local component and full-value comparisons, not a proximity or soundness
//! proof. Hiding and single-shot semantics are not connected. The production
//! verify entry is unchanged.

use std::sync::{Arc, Mutex};

use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::error::{Error, Result};
use zkie_core::proof_header::PROOF_VERSION_WORD;
use zkie_core::types::Digest;
use zkie_engine::pcs::request::binding::{BindingParameters, BindingStatement};
use zkie_engine::pcs::request::chain::Chain;
use zkie_engine::pcs::request::code::{build_code_plan, encode_row_fp, CodeParams};
use zkie_engine::pcs::request::field::{Fp, Fp3, E_DEGREE};
use zkie_engine::pcs::request::fold::{
    fold_mask_notice, fold_mask_shape_fault, FoldGroupLayout, FoldInputRow, FoldMemberLayout,
    FoldOutput, FoldShape, FoldStartMessage, OracleMetadata, RuntimeOracles,
    VerifiedGroup, VerifiedGroups, VerifiedStatement,
};
use crate::proof::fold_handoff::FoldHandoffAnchor;
use zkie_engine::pcs::request::foldeq::{
    reader_table_identity, FoldeqBinder, FoldeqBinding, FoldeqEdgeBinding, FoldeqPlan, FoldeqSession,
    GammaPath,
};
use zkie_engine::pcs::request::gu::{GuOracleShape, GuShape};
use zkie_engine::pcs::request::params::{RequestParams, LEAF_SALT_MIN_BYTES, REQUEST_CHAIN_LABEL};
use zkie_engine::pcs::request::read::{unique_sorted, HidingState, ReadLedger};
use zkie_engine::pcs::request::relation::{foldeq_weights, GuRelationRow};
#[cfg(test)]
use zkie_engine::pcs::request::relation::GuRelationKind;
use zkie_engine::pcs::request::runtime_tree::{MaskTreePlan, RuntimeTreePart, RuntimeTreePlan};
use std::cell::RefCell;

use zkie_engine::pcs::request::table::{
    commit_table_streaming, ColumnOpening, IndexReader, IndexShape, SampleSeed, StreamingTable,
    VecTableSource,
    TableDistances,
};

use crate::proof::model_commitment::{
    verify_model_anchor, AssetMeta, AssetOpening, ModelCommitmentPublic,
};
use crate::proof::session_baseline::TrustedModel;

const TABLE_CONFIGURATION: Digest = Digest([0x5a; 32]);
const SESSION_PROTOCOL_IDENTITY: Digest = Digest([0x53; 32]);
#[cfg(test)]
const PLAN_IDENTIFIER: Digest = Digest([0x54; 32]);
const DEMO_REQUEST_ID: Digest = Digest([0x52; 32]);
#[cfg(test)]
const DEMO_CASE_ID: &[u8] = b"demo-column-case";
pub(crate) const GU_NOTICE_ROOT: Digest = Digest([0x47; 32]);
const FRESH_MASK_ROOT: Digest = Digest([0x46; 32]);

const TABLE_LOG_N: u32 = 3;
const TABLE_ELL: u64 = 4;
const TABLE_R: u64 = 1;
const TABLE_V: u32 = 2;
const TABLE_ANSWERS: u32 = 1;
const TABLE_SALT_BYTES: usize = LEAF_SALT_MIN_BYTES;
const DRAW_COUNT: usize = 4;

const GU_ORACLE: u64 = 11;
const GU_STATIC_MEMBER: u64 = 12;
const GU_GROUP: u64 = 5;
const GU_TREE_NUMBER: u64 = 7;
const GU_MAIN_ROWS: u64 = 4;
const GU_ANSWERS: u32 = 1;
const FRESH_MASK_ID: u64 = 99;
const FOLD_K: u32 = 2;
const FOLD_H: u32 = 0;
const MASK_ELL: u64 = 6;
const MASK_R: u64 = 1;

const GU_TREE_RUN: Digest = Digest([0x31; 32]);
const GU_TREE_CONFIGURATION: Digest = Digest([0x32; 32]);
const MASK_TREE_RUN: Digest = Digest([0x33; 32]);
const MASK_TREE_CONFIGURATION: Digest = Digest([0x34; 32]);

/// A self-contained table built from the authenticated asset slice, together
/// with the messages and tails used to build it.
pub struct AssetEncodingTable {
    pub table: StreamingTable,
    pub source: RefCell<VecTableSource>,
    pub reader: IndexReader,
    pub asset_root: Digest,
    pub shape: IndexShape,
    pub messages: Vec<Vec<Fp>>,
    pub tails: Vec<Vec<Fp>>,
}

/// The result of the deterministic full-table comparison.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableComparison {
    pub cells: usize,
    pub matches: bool,
}

/// The independent candidate configuration for the column session. The
/// parameters are built from the version word, protocol identity, request
/// parameters and plan identifier; the statement is built only after the
/// reader exists. The real fixture supplies the actual case id through
/// `from_plan`; unit tests use `demo`. `request_id` remains an explicit demo
/// placeholder: this does not claim a real request identity or that different
/// real requests are distinguished.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnBindingInput {
    pub parameters: BindingParameters,
    pub case_id: Vec<u8>,
    pub request_id: Digest,
    pub edge: FoldeqEdgeBinding,
}

impl ColumnBindingInput {
    /// Builds the candidate binding input from the verifier's selected static
    /// plan identifier and the real case identifier. The request id is the
    /// explicit demo placeholder. The plan identifier source and coverage are
    /// recorded by the caller; this constructor does not make them trusted.
    pub fn from_plan(plan_identifier: Digest, case_id: impl Into<Vec<u8>>) -> Result<Self> {
        let request_parameters =
            RequestParams::default_with_code(TABLE_LOG_N, TABLE_ELL, TABLE_R)?;
        let parameters = BindingParameters::new(
            PROOF_VERSION_WORD,
            SESSION_PROTOCOL_IDENTITY,
            &request_parameters,
            plan_identifier,
        )?;
        let case_id = case_id.into();
        if case_id.is_empty() {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            parameters,
            case_id,
            request_id: DEMO_REQUEST_ID,
            edge: FoldeqEdgeBinding::NotAnEdge,
        })
    }

    /// Adds a typed edge identity to this candidate input. The caller is
    /// responsible for checking it against its independent plan; this method
    /// only carries the typed value into the first query.
    pub fn with_edge(mut self, edge: FoldeqEdgeBinding) -> Self {
        self.edge = edge;
        self
    }

    /// The unit-test-only fixed input. It is not used by the real fixture and
    /// does not represent a real plan or case.
    #[cfg(test)]
    pub fn demo() -> Result<Self> {
        Self::from_plan(PLAN_IDENTIFIER, DEMO_CASE_ID)
    }
}

/// One session run. This is a component harness result, not an accepted proof.
struct SessionRun {
    gu_root: Digest,
    binding_parameters_digest: Digest,
    binding_table_identity: Digest,
    edge_bound: bool,
    gamma: Vec<Fp3>,
    eq_gamma: Vec<Fp3>,
    queries: Vec<u64>,
    opened_positions: Vec<u64>,
    transcript_after_queries: Digest,
    columns: Vec<Fp>,
    rows: Vec<GuRelationRow>,
    comparison: TableComparison,
    merge_targets: bool,
    fold_output: FoldOutput,
    fold_anchor: FoldHandoffAnchor,
    next_stage_closed: bool,
}

/// The four component layers and the three separately recorded roots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnSessionOutcome {
    pub asset_root: Digest,
    pub table_root: Digest,
    pub gu_root: Digest,
    pub shape: IndexShape,
    pub binding_parameters_digest: Digest,
    pub binding_table_identity: Digest,
    pub edge_bound: bool,
    pub queries: Vec<u64>,
    pub opened_positions: Vec<u64>,
    pub gamma: Vec<Fp3>,
    pub eq_gamma: Vec<Fp3>,
    pub transcript_after_queries: Digest,
    pub column_cells: usize,
    pub gu_rows: usize,
    pub column_authentication: bool,
    pub gu_registration: bool,
    pub merge_targets: bool,
    pub full_table_comparison: bool,
    pub fold_component: bool,
    pub fold_next_stage_closed: bool,
    pub fold_output: FoldOutput,
    pub fold_anchor: FoldHandoffAnchor,
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

fn index_shape_fault(shape: &IndexShape) -> Option<&'static str> {
    zkie_engine::pcs::request::table::index_fault(shape)
        .or_else(|| zkie_engine::pcs::request::table::tree_shape_fault(shape))
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
/// The trusted anchor and the complete opening are verified first. The full
/// opening is resident and every row is checked; only the first four rows and
/// the first plain column are used by the table. The asset file itself is not
/// opened or materialized by this path.
pub fn build_authenticated_encoding_table(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    asset_path: &str,
    opening: &AssetOpening,
) -> Result<AssetEncodingTable> {
    build_authenticated_encoding_table_with_configuration(
        public,
        trusted,
        asset_path,
        opening,
        TABLE_CONFIGURATION,
    )
}

/// The same table construction with the caller's table configuration. The
/// independent candidate parameters digest is the configuration used by the
/// session entry; the fixed demo configuration remains available for isolated
/// component probes. The table root and record are produced after this
/// configuration and are never part of it.
pub fn build_authenticated_encoding_table_with_configuration(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    asset_path: &str,
    opening: &AssetOpening,
    configuration: Digest,
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
            let value = *source.values.get(column).ok_or(Error::ShapeMismatch)?;
            *entry = fp_from_signed_word(value);
        }
    }
    let salts: Vec<Vec<u8>> = (0..columns)
        .map(|position| vec![(position as u8) ^ 0xA5; shape.salt_bytes])
        .collect();

    let mut source = VecTableSource::new(messages.clone(), tails.clone())?;
    let table =
        commit_table_streaming(&mut source, shape, configuration, &salts, 64, u64::MAX)?;
    let reader = table.reader.clone();
    Ok(AssetEncodingTable {
        table,
        source: RefCell::new(source),
        reader,
        asset_root: meta.root,
        shape,
        messages,
        tails,
    })
}

/// Opens one charged physical column per unique query position and returns one
/// opening per logical query in the query order. Repeated queries reuse the
/// same opening. The caller must keep the table's immutable root for the
/// session; this function is not a verifier by itself.
pub fn open_columns_for_queries(
    table: &AssetEncodingTable,
    queries: &[u64],
) -> Result<Vec<ColumnOpening>> {
    if queries.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let positions = unique_sorted(queries);
    let mut state = HidingState::new(table.reader.root, 1, positions.len() as u64);
    let _charge = state.begin_charge(table.reader.root, 1, positions.len())?;
    let mut source = table.source.borrow_mut();
    let unique = table.table.open_columns(&mut *source, &positions)?;
    let mut openings = Vec::with_capacity(queries.len());
    for query in queries {
        let opening = unique
            .iter()
            .find(|opening| opening.position == *query)
            .ok_or(Error::InvalidIndex)?
            .clone();
        openings.push(opening);
    }
    Ok(openings)
}

/// Re-encodes every row and compares every opened cell with the expected
/// codeword symbol. This is a deterministic full-value comparison on the small
/// in-memory table; it is not a proximity test.
pub fn check_full_columns(
    table: &AssetEncodingTable,
    queries: &[u64],
    columns: &[Fp],
) -> Result<TableComparison> {
    let rows = usize::try_from(table.shape.rows_checked()?).map_err(|_| Error::ResourceLimit)?;
    let expected = queries.len().checked_mul(rows).ok_or(Error::ResourceLimit)?;
    if columns.len() != expected {
        return Err(Error::ShapeMismatch);
    }
    let mut matches = true;
    let mut cells = 0usize;
    for (index, query) in queries.iter().enumerate() {
        if *query >= table.shape.code.n() {
            return Err(Error::InvalidIndex);
        }
        for row in 0..rows {
            let codeword = encode_row_fp(
                &table.shape.code,
                table.messages.get(row).ok_or(Error::ShapeMismatch)?,
                table.tails.get(row).ok_or(Error::ShapeMismatch)?,
            )?;
            let expected = *codeword.get(*query as usize).ok_or(Error::InvalidIndex)?;
            let actual = columns[index * rows + row];
            if expected != actual {
                matches = false;
            }
            cells += 1;
        }
    }
    Ok(TableComparison { cells, matches })
}

/// Recomputes each G_u column target from the full re-encoded table and the
/// session's equation weights, then compares it with the target in the
/// registered column row. The membership check is separate and does not
/// evaluate targets.
pub fn check_merge_targets(
    table: &AssetEncodingTable,
    queries: &[u64],
    targets: &[Fp3],
    eq_gamma: &[Fp3],
) -> Result<bool> {
    let row_count =
        usize::try_from(table.shape.rows_checked()?).map_err(|_| Error::ResourceLimit)?;
    if eq_gamma.len() != row_count {
        return Err(Error::ShapeMismatch);
    }
    if targets.len() != queries.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut matches = true;
    for (target, query) in targets.iter().zip(queries) {
        if *query >= table.shape.code.n() {
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
                    .get(*query as usize)
                    .ok_or(Error::InvalidIndex)?,
            );
            expected = expected.add(&weight.mul(&value));
        }
        if *target != expected {
            matches = false;
        }
    }
    Ok(matches)
}

/// The demonstration FOLD shape. The main code is the table's code; the mask
/// code is a valid six-entry code at the same depth. The shape is a public
/// component input, not a production group layout.
pub fn demo_fold_shape(index: IndexShape) -> Result<FoldShape> {
    let mask_code: CodeParams = build_code_plan(index.code.log_n, MASK_ELL, MASK_R)?;
    let shape = FoldShape {
        code: index.code,
        mask_code,
        k: FOLD_K,
        h: FOLD_H,
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
        members: vec![
            FoldMemberLayout {
                oracle: GU_ORACLE,
                main_rows: 1,
                mask_rows: 0,
                kept: false,
            },
            FoldMemberLayout {
                oracle: GU_STATIC_MEMBER,
                main_rows: 1,
                mask_rows: 0,
                kept: false,
            },
        ],
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
        .checked_mul(usize::try_from(shape.mask_code.ell).map_err(|_| Error::ResourceLimit)?)
        .ok_or(Error::ResourceLimit)?;
    Ok(FoldInputRow {
        group: GU_GROUP,
        k: shape.k,
        h: shape.h,
        members: vec![GU_ORACLE, GU_STATIC_MEMBER],
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

/// The trusted G_u shape for the demonstration. Its mask code is the FOLD mask
/// code so the dynamic G_u oracle and the FOLD group use the same trusted
/// code; the notice root is supplied later by the session's SEND message.
pub fn demo_gu_shape(index: IndexShape, mask_code: CodeParams) -> Result<GuShape> {
    let main_symbols = GU_MAIN_ROWS
        .checked_mul(E_DEGREE as u64)
        .ok_or(Error::ResourceLimit)?;
    let mask_symbols = (1u64)
        .checked_mul(E_DEGREE as u64)
        .ok_or(Error::ResourceLimit)?;
    let tree = RuntimeTreePlan {
        tree_number: GU_TREE_NUMBER,
        leaf_class: 0,
        log_n: TABLE_LOG_N,
        log_n_zk: TABLE_LOG_N,
        parts: vec![RuntimeTreePart {
            main_symbols,
            mask_symbols,
        }],
        salt_bytes: TABLE_SALT_BYTES,
        run: GU_TREE_RUN,
        configuration: GU_TREE_CONFIGURATION,
    };
    let u = GuOracleShape {
        code: build_code_plan(TABLE_LOG_N, TABLE_ELL, TABLE_R)?,
        main_rows: GU_MAIN_ROWS,
        main_deg: E_DEGREE as u32,
        mask_rows: 1,
        mask_code: Some(mask_code),
        answers: GU_ANSWERS,
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
    if shape.fault().is_some()
        || zkie_engine::pcs::request::relation::gu_layout_fault(&shape).is_some()
    {
        return Err(Error::InvalidConfig);
    }
    Ok(shape)
}

fn demo_mask_tree() -> Result<MaskTreePlan> {
    let tree = MaskTreePlan {
        tree_number: GU_TREE_NUMBER,
        leaf_class: 0,
        log_n: TABLE_LOG_N,
        mask_rows: 1,
        salt_bytes: TABLE_SALT_BYTES,
        run: MASK_TREE_RUN,
        configuration: MASK_TREE_CONFIGURATION,
    };
    if tree.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    Ok(tree)
}

pub(crate) fn bind_table(
    table: &AssetEncodingTable,
    input: &ColumnBindingInput,
) -> Result<FoldeqBinding> {
    let table_identity = reader_table_identity(&table.reader)?;
    let statement = BindingStatement::new(
        input.parameters.version_word,
        input.parameters.protocol_identity,
        input.case_id.clone(),
        input.request_id,
        table_identity,
    )?;
    FoldeqBinding::new(&input.parameters, statement, &table.reader)
}

pub(crate) struct DemoSetup {
    pub(crate) plan: FoldeqPlan,
    pub(crate) binder: FoldeqBinder,
    pub(crate) chain: Chain,
    pub(crate) first_query: Vec<u8>,
    pub(crate) fold_shape: FoldShape,
    pub(crate) fold_row: FoldInputRow,
    pub(crate) fold_start: FoldStartMessage,
}

pub(crate) fn build_demo_setup(
    table: &AssetEncodingTable,
    binding: FoldeqBinding,
    edge: FoldeqEdgeBinding,
) -> Result<DemoSetup> {
    let fold_shape = demo_fold_shape(table.shape)?;
    let gu_shape = demo_gu_shape(table.shape, fold_shape.mask_code)?;
    let fold_layout = demo_fold_layout(&fold_shape)?;
    let mask_tree = demo_mask_tree()?;
    let mask_notice = fold_mask_notice(&fold_shape)?;

    let group = VerifiedGroup {
        group: GU_GROUP,
        members: vec![GU_STATIC_MEMBER],
        fresh: Some(FRESH_MASK_ID),
        code: Some(gu_shape.u.code),
        mask_code: Some(fold_shape.mask_code),
    };
    let statement = Arc::new(VerifiedStatement::new(
        VerifiedGroups {
            groups: vec![group],
        },
        vec![mask_tree],
        vec![OracleMetadata {
            id: FRESH_MASK_ID,
            group: GU_GROUP,
            tree_number: GU_TREE_NUMBER,
            leaf_class: 0,
            mask_rows: 1,
            mask_code: fold_shape.mask_code,
        }],
    )?);
    let mut oracles = RuntimeOracles::default();
    oracles.receive_mask(
        &statement,
        &mask_tree,
        FRESH_MASK_ID,
        &mask_notice,
        FRESH_MASK_ROOT,
    )?;
    let ledger = Arc::new(Mutex::new(ReadLedger::new(DRAW_COUNT as u64)));
    let plan = FoldeqPlan {
        reader: table.reader.clone(),
        binding,
        edge,
        fold: fold_shape,
        statement,
        oracles: Arc::new(Mutex::new(oracles)),
        mask_tree_number: GU_TREE_NUMBER,
        group_id: GU_GROUP,
        leaf_class: 0,
        seed_budget: table.shape.answers as usize,
        ledger,
        column_draws: DRAW_COUNT,
        mask_query_count: 0,
        gamma_path: GammaPath::Direct {
            columns: table.shape.v,
        },
        gu: Some(gu_shape),
        omega: None,
        fold_layout: Some(fold_layout),
    };
    if plan.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    let binder = FoldeqBinder::new(REQUEST_CHAIN_LABEL, table.reader.configuration, &plan)?;
    let first_query = binder.first_query().to_vec();
    let chain = binder.chain()?;
    let fold_row = demo_fold_input(&fold_shape)?;
    let fold_start = FoldStartMessage {
        id: FRESH_MASK_ID,
        claimed_group: GU_GROUP,
        tree_number: GU_TREE_NUMBER,
        notice: mask_notice,
        root: FRESH_MASK_ROOT,
        sum: Fp3::from_u64(0x2e),
    };
    Ok(DemoSetup {
        plan,
        binder,
        chain,
        first_query,
        fold_shape,
        fold_row,
        fold_start,
    })
}

/// Runs the candidate FOLDEQ session with the fixed G_u root and the demo
/// binding. The same path as `run_session` with a default input.
#[cfg(test)]
fn run_session_with_demo_binding(
    table: &AssetEncodingTable,
    gu_root: &Digest,
) -> Result<SessionRun> {
    let input = ColumnBindingInput::demo()?;
    let binding = bind_table(table, &input)?;
    run_session(table, gu_root, binding, FoldeqEdgeBinding::NotAnEdge)
}

/// Runs the candidate FOLDEQ session with the fixed G_u root. The root is an
/// immutable reference: the same value is sent in the G_u message and carried
/// into the run result, so the fixed root and the used root are the same value.
/// The session derives gamma before G_u, sends G_u before drawing columns, and
/// only then reads and verifies the columns.
fn run_session(
    table: &AssetEncodingTable,
    gu_root: &Digest,
    binding: FoldeqBinding,
    edge: FoldeqEdgeBinding,
) -> Result<SessionRun> {
    let binding_parameters_digest = binding.parameters_digest();
    let binding_table_identity = binding.table_identity();
    let edge_bound = edge.is_bound();
    let setup = build_demo_setup(table, binding, edge)?;
    let fold_shape = setup.fold_shape;
    let fold_start = setup.fold_start.clone();
    let fold_group = setup.plan.group_id;
    let fold_layout = setup
        .plan
        .fold_layout
        .as_ref()
        .ok_or(Error::InvalidConfig)?
        .clone();
    let mut session =
        FoldeqSession::open(setup.plan, setup.binder, setup.chain)?;
    session.bind_first_query(&setup.first_query)?;
    session.draw_gamma()?;
    let gamma = session
        .gamma()
        .ok_or(Error::InvalidSessionState)?
        .to_vec();

    // The G_u message is fixed and absorbed here, after gamma and before the
    // column query draw. The same immutable reference is the message root and
    // the recorded root below.
    session.send_gu(gu_root)?;
    session.shared_seeds()?;
    for answer in 0..GU_ANSWERS {
        session.receive_gu_answer(Fp3::from_u64(0x31 + answer as u64))?;
    }
    session.join_gu_record()?;
    session.draw_queries()?;
    let queries = session.queries().to_vec();
    let transcript_after_queries = *session.transcript_value();
    let openings = open_columns_for_queries(table, &queries)?;
    let read_columns = session.read_queries(&openings)?;
    let columns: Vec<Fp> = read_columns.into_iter().flatten().collect();
    session.join_gu_read_columns()?;
    let mut rows = session.pending_gu_rows().to_vec();
    for draw in 0..session.pending_column_count() {
        rows.push(session.column_relation_row(draw)?);
    }
    let eq_gamma = foldeq_weights(&gamma, table.shape.v)?;
    let merge_targets = check_merge_targets(table, &queries, session.column_targets(), &eq_gamma)?;
    let comparison = check_full_columns(table, &queries, &columns)?;

    session.fold_bind_row(setup.fold_row)?;
    session.start_fold(&fold_start)?;
    session.fold_rounds_start()?;
    for _ in 0..fold_shape.k {
        session.receive_fold_round(Fp3::ZERO, Fp3::ZERO)?;
    }
    let fold_output = session.fold_finish()?;
    let fold_anchor = FoldHandoffAnchor {
        shape: fold_shape,
        group: fold_group,
        members: fold_layout.members.iter().map(|member| member.oracle).collect(),
        fresh: fold_start.id,
        fresh_root: fold_start.root,
        tree_number: fold_start.tree_number,
        layout: fold_layout,
        fold_gamma: session.fold_gamma().to_vec(),
        fold_epsilon: session.epsilon().ok_or(Error::InvalidSessionState)?,
        fold_sum: fold_start.sum,
    };
    let next_stage_closed = session.fold_next_stage().is_err();
    let opened_positions = unique_sorted(&queries);

    Ok(SessionRun {
        gu_root: *gu_root,
        binding_parameters_digest,
        binding_table_identity,
        edge_bound,
        gamma,
        eq_gamma,
        queries,
        opened_positions,
        transcript_after_queries,
        columns,
        rows,
        comparison,
        merge_targets,
        fold_output,
        fold_anchor,
        next_stage_closed,
    })
}

/// The session state after the G_u, record, column-query and column-read
/// registration, before any FOLD round. It is an isolated inspection input for
/// the candidate G_u declaration combination.
pub(crate) struct GuFlow {
    pub(crate) session: FoldeqSession,
    pub(crate) gu_root: Digest,
    pub(crate) queries: Vec<u64>,
    pub(crate) table_identity: Digest,
    pub(crate) transcript_after_columns: Digest,
}

/// Runs the fixed pre-FOLD G_u/record/column flow and returns the session at
/// the read stage. The G_u root, SEND answers and the sample row are the same
/// demonstration values used by the column session; they are not real SEND
/// values. Column openings are authenticated only against the table column
/// root.
/// Runs the fixed pre-FOLD flow with caller-supplied SEND answers. The caller
/// receives the actual session G_u shape and the independent seed set after the
/// fixed root, and returns the complete ordered answer list. The verifier only
/// registers those answers as pending sample claims; it does not compare them
/// with a standard answer.
pub(crate) fn open_gu_flow_with_answers<F>(
    table: &AssetEncodingTable,
    binding: FoldeqBinding,
    edge: FoldeqEdgeBinding,
    gu_root: Digest,
    answer_provider: F,
) -> Result<GuFlow>
where
    F: FnOnce(&GuShape, &[SampleSeed]) -> Result<Vec<Fp3>>,
{
    let setup = build_demo_setup(table, binding, edge)?;
    let mut session = FoldeqSession::open(setup.plan, setup.binder, setup.chain)?;
    session.bind_first_query(&setup.first_query)?;
    session.draw_gamma()?;
    session.send_gu(&gu_root)?;
    session.shared_seeds()?;
    let gu = session
        .plan()
        .gu
        .as_ref()
        .ok_or(Error::InvalidSessionState)?
        .clone();
    let seeds = session.gu_shared_seeds().to_vec();
    let answers = answer_provider(&gu, &seeds)?;
    if answers.len() != gu.u.answers as usize {
        return Err(Error::ShapeMismatch);
    }
    for answer in answers {
        session.receive_gu_answer(answer)?;
    }
    session.join_gu_record()?;
    session.draw_queries()?;
    let queries = session.queries().to_vec();
    let openings = open_columns_for_queries(table, &queries)?;
    session.read_queries(&openings)?;
    session.join_gu_read_columns()?;
    let transcript_after_columns = *session.transcript_value();
    let table_identity = reader_table_identity(&table.reader)?;
    Ok(GuFlow {
        session,
        gu_root,
        queries,
        table_identity,
        transcript_after_columns,
    })
}

/// Runs the four component layers in the session's fixed order and records the
/// three roots. A failed real component returns an error. The FOLD next stage
/// is expected to remain fail-closed.
pub fn verify_column_session(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    input: &ColumnBindingInput,
    asset_path: &str,
    opening: &AssetOpening,
) -> Result<ColumnSessionOutcome> {
    // The independent candidate parameters are the table/chain configuration;
    // the table root/record are produced after it, and the typed statement is
    // built only from the resulting reader.
    let parameters_digest = input.parameters.digest()?;
    let table = build_authenticated_encoding_table_with_configuration(
        public,
        trusted,
        asset_path,
        opening,
        parameters_digest,
    )?;
    let binding = bind_table(&table, input)?;
    let gu_root = GU_NOTICE_ROOT;
    let run = run_session(&table, &gu_root, binding, input.edge.clone())?;
    if run.gu_root != gu_root
        || !run.merge_targets
        || !run.comparison.matches
        || !run.next_stage_closed
        || run.fold_output.next_stage().is_ok()
    {
        return Err(Error::InvalidProof);
    }
    let column_cells = run.columns.len();
    Ok(ColumnSessionOutcome {
        asset_root: table.asset_root,
        table_root: table.reader.root,
        gu_root: run.gu_root,
        shape: table.shape,
        binding_parameters_digest: run.binding_parameters_digest,
        binding_table_identity: run.binding_table_identity,
        edge_bound: run.edge_bound,
        queries: run.queries,
        opened_positions: run.opened_positions,
        gamma: run.gamma,
        eq_gamma: run.eq_gamma,
        transcript_after_queries: run.transcript_after_queries,
        column_cells,
        gu_rows: run.rows.len(),
        column_authentication: true,
        gu_registration: true,
        merge_targets: run.merge_targets,
        full_table_comparison: run.comparison.matches,
        fold_component: true,
        fold_next_stage_closed: run.next_stage_closed,
        fold_output: run.fold_output,
        fold_anchor: run.fold_anchor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use zkie_engine::pcs::request::gu::{GuMembership, GuNotice, GuRegistry};
    use zkie_engine::pcs::request::table::verify_columns;

    use crate::proof::model_commitment::{commit_assets, AssetStore, ModelAssetSpec};

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
        input: ColumnBindingInput,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
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
        let input = ColumnBindingInput::demo().unwrap();
        Fixture {
            root,
            public,
            opening,
            specs,
            anchor_id,
            model_id,
            parameters_digest,
            input,
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
        build_authenticated_encoding_table_with_configuration(
            &fixture.public,
            &trusted,
            "table_i32.bin",
            &fixture.opening,
            fixture.input.parameters.digest().unwrap(),
        )
        .unwrap()
    }

    fn verify(fixture: &Fixture) -> Result<ColumnSessionOutcome> {
        let trusted = trusted(fixture);
        verify_column_session(
            &fixture.public,
            &trusted,
            &fixture.input,
            "table_i32.bin",
            &fixture.opening,
        )
    }

    fn session_to_queried(
        table: &AssetEncodingTable,
        gu_root: &Digest,
    ) -> Result<(FoldeqSession, Vec<Fp3>)> {
        let input = ColumnBindingInput::demo()?;
        let binding = bind_table(table, &input)?;
        let setup = build_demo_setup(table, binding, input.edge)?;
        let mut session = FoldeqSession::open(setup.plan, setup.binder, setup.chain)?;
        session.bind_first_query(&setup.first_query)?;
        session.draw_gamma()?;
        let gamma = session
            .gamma()
            .ok_or(Error::InvalidSessionState)?
            .to_vec();
        session.send_gu(gu_root)?;
        session.shared_seeds()?;
        for answer in 0..GU_ANSWERS {
            session.receive_gu_answer(Fp3::from_u64(0x31 + answer as u64))?;
        }
        session.join_gu_record()?;
        session.draw_queries()?;
        Ok((session, gamma))
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
        assert_eq!(outcome.queries.len(), DRAW_COUNT);
        assert_eq!(outcome.gamma.len(), TABLE_V as usize);
        assert_eq!(outcome.eq_gamma, foldeq_weights(&outcome.gamma, TABLE_V).unwrap());
        assert_eq!(outcome.column_cells, outcome.queries.len() * 4);
        assert!(outcome.gu_rows >= DRAW_COUNT);
        assert_ne!(outcome.asset_root, outcome.table_root);
        assert_ne!(outcome.asset_root, outcome.gu_root);
        assert_ne!(outcome.table_root, outcome.gu_root);
        let input = ColumnBindingInput::demo().unwrap();
        assert_eq!(
            outcome.binding_parameters_digest,
            input.parameters.digest().unwrap()
        );
        assert_eq!(
            outcome.binding_table_identity,
            reader_table_identity(&table(&fixture).reader).unwrap()
        );
    }

    #[test]
    fn fixed_fixture_is_reproducible() {
        let fixture = fixture();
        let first = verify(&fixture).unwrap();
        let second = verify(&fixture).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn changed_gu_message_changes_the_query_transcript_and_order() {
        let fixture = fixture();
        let table = table(&fixture);
        let base = run_session_with_demo_binding(&table, &GU_NOTICE_ROOT).unwrap();
        let changed_root = Digest([0x48; 32]);
        let changed = run_session_with_demo_binding(&table, &changed_root).unwrap();
        assert_eq!(base.gu_root, GU_NOTICE_ROOT);
        assert_eq!(changed.gu_root, changed_root);
        assert_ne!(
            base.transcript_after_queries,
            changed.transcript_after_queries
        );
        assert_ne!(base.queries, changed.queries);
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
        let run = run_session_with_demo_binding(&table, &GU_NOTICE_ROOT).unwrap();
        let openings = open_columns_for_queries(&table, &run.queries).unwrap();
        let positions = unique_sorted(&run.queries);
        let mut unique: Vec<ColumnOpening> = positions
            .iter()
            .map(|position| {
                openings
                    .iter()
                    .find(|opening| opening.position == *position)
                    .unwrap()
                    .clone()
            })
            .collect();
        unique[0].path.push(Digest([9; 32]));
        assert!(verify_columns(&table.reader, &positions, &unique).is_err());
        assert!(run_session_with_demo_binding(&table, &GU_NOTICE_ROOT).is_ok());
    }

    #[test]
    fn path_valid_but_wrong_merge_is_seen_by_the_target_control() {
        let fixture = fixture();
        let table = table(&fixture);
        let (mut session, gamma) = session_to_queried(&table, &GU_NOTICE_ROOT).unwrap();
        let queries = session.queries().to_vec();
        let openings = open_columns_for_queries(&table, &queries).unwrap();
        let positions = unique_sorted(&queries);
        let unique: Vec<ColumnOpening> = positions
            .iter()
            .map(|position| {
                openings
                    .iter()
                    .find(|opening| opening.position == *position)
                    .unwrap()
                    .clone()
            })
            .collect();
        let flat = verify_columns(&table.reader, &positions, &unique).unwrap();
        let rows = usize::try_from(table.shape.rows_checked().unwrap()).unwrap();
        let mut fixture_columns: Vec<(u64, Vec<Fp3>)> = positions
            .iter()
            .enumerate()
            .map(|(index, position)| {
                let column = flat[index * rows..(index + 1) * rows]
                    .iter()
                    .map(|value| Fp3::from_base(*value))
                    .collect();
                (*position, column)
            })
            .collect();
        fixture_columns[0].1[0] = fixture_columns[0].1[0].add(&Fp3::ONE);
        session
            .join_gu_columns(&fixture_columns)
            .expect("membership registration does not evaluate the target");
        let eq_gamma = foldeq_weights(&gamma, TABLE_V).unwrap();
        assert!(!check_merge_targets(
            &table,
            &queries,
            session.column_targets(),
            &eq_gamma
        )
        .unwrap());
    }

    #[test]
    fn gu_registration_rejects_a_wrong_oracle_or_group() {
        let fixture = fixture();
        let table = table(&fixture);
        let fold = demo_fold_shape(table.shape).unwrap();
        let shape = demo_gu_shape(table.shape, fold.mask_code).unwrap();
        let notice = GuNotice {
            root: GU_NOTICE_ROOT,
            oracle: shape.oracle,
            tree_number: shape.tree.tree_number,
            part: shape.part,
        };
        let mut registry = GuRegistry::new();
        registry.receive(&shape, &notice).unwrap();
        let group = VerifiedGroup {
            group: shape.group,
            members: vec![GU_STATIC_MEMBER],
            fresh: Some(FRESH_MASK_ID),
            code: Some(shape.u.code),
            mask_code: shape.u.mask_code,
        };
        let mut membership = GuMembership::new(&group, &shape, &registry).unwrap();
        membership.join(&shape, &registry).unwrap();
        let mut row = GuRelationRow {
            oracle: shape.oracle,
            group: shape.group,
            kind: GuRelationKind::Column,
            main: Vec::new(),
            mask: Vec::new(),
            target: Fp3::ZERO,
        };
        assert!(membership.check_row(&row).is_ok());
        row.oracle += 1;
        assert!(membership.check_row(&row).is_err());
        row.oracle = shape.oracle;
        row.group += 1;
        assert!(membership.check_row(&row).is_err());
    }

    #[test]
    fn full_table_comparison_detects_a_wrong_opened_cell() {
        let fixture = fixture();
        let table = table(&fixture);
        let run = run_session_with_demo_binding(&table, &GU_NOTICE_ROOT).unwrap();
        assert!(run.comparison.matches);
        let mut columns = run.columns.clone();
        columns[0] = Fp3::from_u64(999).c[0];
        assert!(!check_full_columns(&table, &run.queries, &columns)
            .unwrap()
            .matches);
    }

    #[test]
    fn fold_component_checks_the_public_row_and_stops_at_the_closed_boundary() {
        let fixture = fixture();
        let table = table(&fixture);
        let run = run_session_with_demo_binding(&table, &GU_NOTICE_ROOT).unwrap();
        assert!(run.next_stage_closed);
        assert!(run.fold_output.next_stage().is_err());
        let shape = demo_fold_shape(table.shape).unwrap();
        let layout = demo_fold_layout(&shape).unwrap();
        let mut row = demo_fold_input(&shape).unwrap();
        row.check(&shape, &layout).unwrap();
        row.coefficients.pop();
        assert!(row.check(&shape, &layout).is_err());
    }
}
