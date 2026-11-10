//! The output-head table's key material: the committed table, its record and
//! the mapping from the authenticated weights.
//!
//! The builder reads the authenticated weight rows through the read-only row
//! path, never the whole asset, and fills one table row at a time. The table
//! entry at row `r`, matrix row `y` and low column `j` is the source weight at
//! matrix row `y`, column `2^k_lo r + j`. Random tails, pads and salts are the
//! caller's private values; the builder checks their sizes and binds them
//! before the table's root is computed.

use std::sync::{Arc, Mutex};

use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::error::{Error, Result};
use zkie_core::proof_header::PROOF_VERSION_WORD;
use zkie_core::types::Digest;
use zkie_engine::pcs::request::binding::{BindingParameters, BindingStatement};
use zkie_engine::pcs::request::chain::Chain;
use zkie_engine::pcs::request::code::encode_row_fp3;
use zkie_engine::pcs::request::field::{Fp, Fp3, E_DEGREE};
use zkie_engine::pcs::request::fold::{
    fold_mask_notice, fold_mask_shape_fault, FoldGroupLayout, FoldMemberLayout, FoldShape,
    OracleMetadata, RuntimeOracles, VerifiedGroup, VerifiedGroups, VerifiedStatement,
};
use zkie_engine::pcs::request::foldeq::{
    reader_table_identity, FoldeqBinder, FoldeqBinding, FoldeqEdgeBinding, FoldeqPlan,
    FoldeqSession, GammaPath, LinShapeBinding,
};
use zkie_engine::pcs::request::gu::{GuNotice, GuOracleShape, GuRegistry, GuShape};
use zkie_engine::pcs::request::params::{RequestParams, SHARED_CHAIN_LABEL};
use zkie_engine::pcs::request::plan::{CompiledPlan, RequestPlan};
use zkie_engine::pcs::request::read::{unique_sorted, HidingState, ReadLedger};
use zkie_engine::pcs::request::relation::{
    foldeq_weights, gu_sample_pad_offsets, gu_table, oracle_sample_row, sample_row, GuWitness,
};
use zkie_engine::pcs::request::runtime_tree::{
    MaskTreePlan, RuntimePartRows, RuntimeTree, RuntimeTreePart, RuntimeTreePlan,
};
use zkie_engine::pcs::request::table::{
    commit_table, verify_columns, ColumnOpening, CommittedTable, IndexReader, IndexShape,
    SampleSeed,
};
use zkie_engine::primitives::transcript::run_value;
use zkie_ops::proof::column_session::fp_from_signed_word;
use zkie_ops::proof::session_baseline::SESSION_PROTOCOL_IDENTITY;

use crate::s7_context::LmHeadContext;

/// The private values of one table: one random tail for each row, one salt for
/// each column and one pad for each answer of each row.
pub struct LmHeadKeyRandomness {
    pub tails: Vec<Vec<Fp>>,
    pub salts: Vec<Vec<u8>>,
    pub pads: Vec<Vec<Fp>>,
}

/// The committed table and its public reader.
pub struct LmHeadKey {
    pub shape: IndexShape,
    pub reader: IndexReader,
    pub table: CommittedTable,
    pub name: Vec<u8>,
    /// The committed message rows, kept for the one-time fold of this table.
    pub messages: Vec<Vec<Fp>>,
}

/// The table entry of one matrix cell, read through the authenticated row path.
pub fn lm_head_entry(
    context: &LmHeadContext,
    plan: &RequestPlan,
    table_row: u64,
    matrix_row: u64,
    low: u64,
) -> Result<Fp> {
    let factor = 1u64
        .checked_shl(plan.lin.k_lo)
        .ok_or(Error::ResourceLimit)?;
    if table_row >= 1u64 << plan.table.v
        || matrix_row >= 1u64 << plan.lin.matrix_rows_log
        || low >= factor
    {
        return Err(Error::InvalidIndex);
    }
    let column = table_row
        .checked_mul(factor)
        .and_then(|value| value.checked_add(low))
        .ok_or(Error::ResourceLimit)?;
    let row = context.read_lm_head_row(matrix_row)?;
    let value = *row
        .get(usize::try_from(column).map_err(|_| Error::ResourceLimit)?)
        .ok_or(Error::ShapeMismatch)?;
    Ok(fp_from_signed_word(value as u64))
}

/// Builds the key material of the output-head table.
///
/// `max_bytes` bounds the derived buffers before any of them is allocated: the
/// messages, the codewords, the tails and the column tree.
pub fn build_lm_head_key(
    context: &LmHeadContext,
    plan: &RequestPlan,
    configuration: Digest,
    randomness: &LmHeadKeyRandomness,
    max_bytes: u64,
) -> Result<LmHeadKey> {
    plan.validate()?;
    if configuration == Digest::ZERO {
        return Err(Error::InvalidConfig);
    }
    let shape = plan.table_shape()?;
    let rows = usize::try_from(shape.rows_checked()?).map_err(|_| Error::ResourceLimit)?;
    let factor = 1usize
        .checked_shl(plan.lin.k_lo)
        .ok_or(Error::ResourceLimit)?;
    let columns = 1usize
        .checked_shl(plan.lin.k_col)
        .ok_or(Error::ResourceLimit)?;
    if context.asset.rows != shape.rows()
        || context.asset.row_width < columns as u64
        || context.asset.element_count
            != context.asset.rows.saturating_mul(context.asset.row_width)
    {
        return Err(Error::ShapeMismatch);
    }
    if factor.checked_mul(rows).ok_or(Error::ResourceLimit)? != columns {
        return Err(Error::InvalidConfig);
    }
    let ell = usize::try_from(shape.code.ell).map_err(|_| Error::ResourceLimit)?;
    let tail = usize::try_from(shape.code.r).map_err(|_| Error::ResourceLimit)?;
    let code_len = usize::try_from(shape.code.n()).map_err(|_| Error::ResourceLimit)?;
    let plain = usize::try_from(shape.plain()).map_err(|_| Error::ResourceLimit)?;
    if plain.checked_add(E_DEGREE * shape.answers as usize).ok_or(Error::ResourceLimit)? != ell {
        return Err(Error::InvalidConfig);
    }
    let messages_bytes = (rows as u64)
        .checked_mul(ell as u64)
        .and_then(|value| value.checked_mul(8))
        .ok_or(Error::ResourceLimit)?;
    let codewords_bytes = (rows as u64)
        .checked_mul(code_len as u64)
        .and_then(|value| value.checked_mul(8))
        .ok_or(Error::ResourceLimit)?;
    let tails_bytes = (rows as u64)
        .checked_mul(tail as u64)
        .and_then(|value| value.checked_mul(8))
        .ok_or(Error::ResourceLimit)?;
    let tree_bytes = (code_len as u64)
        .checked_mul((rows as u64).saturating_mul(8).saturating_add(shape.salt_bytes as u64))
        .ok_or(Error::ResourceLimit)?;
    let needed = messages_bytes
        .checked_add(codewords_bytes)
        .and_then(|value| value.checked_add(tails_bytes))
        .and_then(|value| value.checked_add(tree_bytes))
        .ok_or(Error::ResourceLimit)?;
    if needed > max_bytes {
        return Err(Error::ResourceLimit);
    }
    if randomness.pads.len() != rows
        || randomness.pads.iter().any(|row| row.len() != E_DEGREE * shape.answers as usize)
        || randomness.tails.len() != rows
        || randomness.tails.iter().any(|row| row.len() != tail)
        || randomness.salts.len() != code_len
        || randomness
            .salts
            .iter()
            .any(|salt| salt.len() != shape.salt_bytes)
    {
        return Err(Error::ShapeMismatch);
    }

    let mut messages = vec![vec![Fp::ZERO; ell]; rows];
    for row in 0..rows {
        for (offset, pad) in randomness.pads[row].iter().enumerate() {
            messages[row][plain + offset] = *pad;
        }
    }
    for matrix_row in 0..rows {
        let source = context.read_lm_head_row(matrix_row as u64)?;
        if source.len() < columns {
            return Err(Error::ShapeMismatch);
        }
        for table_row in 0..rows {
            let base = table_row
                .checked_mul(factor)
                .ok_or(Error::ResourceLimit)?;
            let destination = &mut messages[table_row];
            for low in 0..factor {
                let column = base.checked_add(low).ok_or(Error::ResourceLimit)?;
                let place = low
                    .checked_add(
                        matrix_row
                            .checked_mul(factor)
                            .ok_or(Error::ResourceLimit)?,
                    )
                    .ok_or(Error::ResourceLimit)?;
                destination[place] =
                    fp_from_signed_word(source[column] as u64);
            }
        }
    }
    let table = commit_table(
        shape,
        configuration,
        &messages,
        &randomness.tails,
        &randomness.salts,
    )?;
    let reader = table.reader.clone();
    let name = table.name.clone();
    Ok(LmHeadKey {
        shape,
        reader,
        table,
        name,
        messages,
    })
}


/// The version of the limited key statement and plan. This is an adaptation
/// with no range proof: it binds the trusted publication assumption that every
/// valid matrix entry is the canonical i32 weight under the fixed mapping.
/// It is not a reference key file and promises no range proof.
pub const LIMITED_KEY_VERSION: u64 = 1;

/// The public statement of the limited key. The version, the model identity,
/// the table root/record/configuration, the table shape, the matrix mapping's
/// dimensions and the admitted element range are all bound here. The two flags
/// say what the statement assumes rather than proves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LimitedKeyStatement {
    pub version: u64,
    pub model_commitment: Digest,
    pub table_root: Digest,
    pub table_configuration: Digest,
    pub table_shape: IndexShape,
    pub matrix_rows: u64,
    pub k_lo: u32,
    pub k_col: u32,
    pub valid_min: i64,
    pub valid_max: i64,
    pub no_wf: bool,
    pub trusted_i32_mapping: bool,
}

impl LimitedKeyStatement {
    pub fn encode(&self) -> Vec<u8> {
        let mut writer = zkie_engine::primitives::codec::CanonicalWriter::new();
        writer.put_u64(self.version);
        writer.put_digest(&self.model_commitment);
        writer.put_digest(&self.table_root);
        writer.put_digest(&self.table_configuration);
        writer.put_u64(self.table_shape.code.log_n as u64);
        writer.put_u64(self.table_shape.code.ell);
        writer.put_u64(self.table_shape.code.r);
        writer.put_u64(self.table_shape.v as u64);
        writer.put_u64(self.table_shape.answers as u64);
        writer.put_u64(self.table_shape.salt_bytes as u64);
        writer.put_u64(self.table_shape.distances.radius);
        writer.put_u64(self.table_shape.distances.later_radius);
        writer.put_u64(self.table_shape.distances.m_once);
        writer.put_u64(self.table_shape.distances.m_later);
        writer.put_u64(self.matrix_rows);
        writer.put_u64(self.k_lo as u64);
        writer.put_u64(self.k_col as u64);
        writer.put_u64(self.valid_min as u64);
        writer.put_u64(self.valid_max as u64);
        writer.put_u64(self.no_wf as u64);
        writer.put_u64(self.trusted_i32_mapping as u64);
        writer.into_bytes()
    }

    pub fn digest(&self) -> Digest {
        zkie_core::artifact::digest_bytes(&self.encode())
    }

    /// Checks the limited statement's own version, semantics and range.
    pub fn validate(&self) -> Result<()> {
        if self.version != LIMITED_KEY_VERSION
            || !self.no_wf
            || !self.trusted_i32_mapping
            || self.model_commitment == Digest::ZERO
            || self.table_root == Digest::ZERO
            || self.table_configuration == Digest::ZERO
            || self.valid_min > self.valid_max
            || self.valid_min < i32::MIN as i64
            || self.valid_max > i32::MAX as i64
        {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }
}

/// The retained stages of the limited key, in the reference order. There is no
/// range-proof stage: the statement's flags say that the range property is a
/// trusted publication assumption, not a checked one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitedKeyStage {
    IndexMasks,
    LinFirst,
    Seal,
    Place,
    LinShared,
    LinSecond,
}

/// The limited key's plan: the retained stages and the compiled one-time and
/// batch groups.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LimitedKeyPlan {
    pub version: u64,
    pub stages: Vec<LimitedKeyStage>,
    pub once: zkie_engine::pcs::request::plan::CompiledLines,
    pub proof: zkie_engine::pcs::request::plan::CompiledLines,
}

impl LimitedKeyPlan {
    pub fn validate(&self) -> Result<()> {
        let expected = [
            LimitedKeyStage::IndexMasks,
            LimitedKeyStage::LinFirst,
            LimitedKeyStage::Seal,
            LimitedKeyStage::Place,
            LimitedKeyStage::LinShared,
            LimitedKeyStage::LinSecond,
        ];
        if self.version != LIMITED_KEY_VERSION || self.stages != expected.to_vec() {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }
}

/// Builds the limited key statement from the trusted context, the plan and the
/// committed table. The admitted range is the canonical i32 range of the
/// mapping; the statement does not prove it.
pub fn build_limited_key_statement(
    context: &LmHeadContext,
    plan: &RequestPlan,
    key: &LmHeadKey,
) -> Result<LimitedKeyStatement> {
    plan.validate()?;
    let shape = plan.table_shape()?;
    if key.shape != shape || key.reader.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    let factor = 1u64
        .checked_shl(plan.lin.k_lo)
        .ok_or(Error::InvalidConfig)?;
    let rows = 1u64 << plan.table.v;
    let columns = 1u64 << plan.lin.k_col;
    if factor.checked_mul(rows).ok_or(Error::InvalidConfig)? != columns
        || rows != 1u64 << plan.lin.matrix_rows_log
    {
        return Err(Error::InvalidConfig);
    }
    let statement = LimitedKeyStatement {
        version: LIMITED_KEY_VERSION,
        model_commitment: context.model_id,
        table_root: key.reader.root,
        table_configuration: key.reader.configuration,
        table_shape: shape,
        matrix_rows: rows,
        k_lo: plan.lin.k_lo,
        k_col: plan.lin.k_col,
        valid_min: i32::MIN as i64,
        valid_max: i32::MAX as i64,
        no_wf: true,
        trusted_i32_mapping: true,
    };
    statement.validate()?;
    Ok(statement)
}

/// Builds the limited key plan. Every retained stage is present, and the
/// compiled groups are checked against the plan's lines.
pub fn limited_key_plan(
    plan: &RequestPlan,
    compiled: &zkie_engine::pcs::request::plan::CompiledPlan,
) -> Result<LimitedKeyPlan> {
    plan.validate_compiled(compiled)?;
    let key_plan = LimitedKeyPlan {
        version: LIMITED_KEY_VERSION,
        stages: vec![
            LimitedKeyStage::IndexMasks,
            LimitedKeyStage::LinFirst,
            LimitedKeyStage::Seal,
            LimitedKeyStage::Place,
            LimitedKeyStage::LinShared,
            LimitedKeyStage::LinSecond,
        ],
        once: compiled.once.clone(),
        proof: compiled.proof.clone(),
    };
    key_plan.validate()?;
    Ok(key_plan)
}


/// The fixed plan identities of the output-head shared-only test: the
/// one-time FOLD group, its two folded members, the fresh mask oracle and its
/// runtime tree. They are plan values, not proof values; the first query binds
/// every one of them.
pub const LM_HEAD_FOLD_GROUP: u64 = 1;
pub const LM_HEAD_GU_ORACLE: u64 = 1;
pub const LM_HEAD_OMEGA_ORACLE: u64 = 2;
pub const LM_HEAD_FRESH_MASK: u64 = 3;
pub const LM_HEAD_MASK_TREE: u64 = 0;
pub const LM_HEAD_GU_TREE: u64 = 2;

/// The table columns drawn by the one-time shared test.
pub const LM_HEAD_COLUMN_DRAWS: usize = 603;

/// The verifier's plan for one shared-only key session and the binder that
/// starts it. The table root, record and configuration come from the reader;
/// the run and both runtime trees are derived from the caller's private
/// random bytes; the binding parameters are the table's configuration.
pub struct LimitedSharedSession {
    pub plan: FoldeqPlan,
    pub binder: FoldeqBinder,
    pub chain: Chain,
    pub first_query: Vec<u8>,
}

/// Builds the binding parameters of the limited key. The plan identifier is
/// the selected static plan's digest; the request parameters are the table
/// code's. The table's configuration is this value's digest.
pub fn limited_key_binding_parameters(
    plan: &RequestPlan,
    plan_identifier: Digest,
) -> Result<BindingParameters> {
    plan.validate()?;
    let request_parameters = RequestParams::default_with_code(
        plan.table.code.log_n,
        plan.table.code.ell,
        plan.table.code.r,
    )?;
    BindingParameters::with_plan_parts(
        PROOF_VERSION_WORD,
        SESSION_PROTOCOL_IDENTITY,
        &request_parameters,
        &[plan_identifier],
    )
}

/// Builds the real shared-only FOLDEQ plan for the committed key table. The
/// fold shape comes from the compiled one-time lines, the fresh mask root from
/// the provider's pre-committed mask message, and every runtime tree's run and
/// configuration from the session chain and the table. The plan is rejected
/// unless the binding parameters digest is exactly the reader's configuration.
#[allow(clippy::too_many_arguments)]
pub fn build_limited_shared_session(
    request_plan: &RequestPlan,
    compiled: &CompiledPlan,
    statement: &LimitedKeyStatement,
    reader: &IndexReader,
    plan_identifier: Digest,
    case_id: &[u8],
    random_bytes: Digest,
    fresh_mask_root: Digest,
) -> Result<LimitedSharedSession> {
    request_plan.validate()?;
    request_plan.validate_compiled(compiled)?;
    if fresh_mask_root == Digest::ZERO
        || random_bytes == Digest::ZERO
        || case_id.is_empty()
    {
        return Err(Error::InvalidConfig);
    }
    let parameters = limited_key_binding_parameters(request_plan, plan_identifier)?;
    let configuration = parameters.digest()?;
    if configuration != reader.configuration {
        return Err(Error::StatementMismatch);
    }
    statement.validate()?;
    if statement.table_root != reader.root
        || statement.table_configuration != reader.configuration
        || statement.table_shape != reader.shape
    {
        return Err(Error::StatementMismatch);
    }
    let factor = 1u64
        .checked_shl(request_plan.lin.k_lo)
        .ok_or(Error::InvalidConfig)?;
    if statement.matrix_rows != 1u64 << request_plan.table.v
        || statement.matrix_rows != 1u64 << request_plan.lin.matrix_rows_log
        || statement.k_lo != request_plan.lin.k_lo
        || statement.k_col != request_plan.lin.k_col
        || factor
            .checked_mul(statement.matrix_rows)
            .ok_or(Error::InvalidConfig)?
            != 1u64 << request_plan.lin.k_col
    {
        return Err(Error::InvalidConfig);
    }

    let run = run_value(SHARED_CHAIN_LABEL, &configuration, &random_bytes)?;
    if run == Digest::ZERO {
        return Err(Error::InvalidConfig);
    }

    let folded = &compiled.once.folded;
    let fold = FoldShape {
        code: folded.main,
        mask_code: folded.mask,
        k: folded.k,
        h: folded.h,
        mask_leaf_class: compiled.once_foldeq.shared_split_leaf_class,
    };
    if fold_mask_shape_fault(&fold).is_some() {
        return Err(Error::InvalidConfig);
    }
    let fresh = compiled
        .once_foldeq
        .shared_split
        .as_ref()
        .ok_or(Error::InvalidConfig)?;
    if fresh.main_rows != 0
        || fresh.mask_rows != fold.mask_rows()?
        || fresh.mask_code != Some(fold.mask_code)
    {
        return Err(Error::InvalidConfig);
    }

    let gu_oracle = compiled.once_foldeq.u.clone();
    let omega = compiled.once_foldeq.omega.clone();
    let gu_tree = RuntimeTreePlan {
        tree_number: LM_HEAD_GU_TREE,
        leaf_class: compiled.once_foldeq.fold_leaf_class,
        log_n: gu_oracle.code.log_n,
        log_n_zk: gu_oracle.code.log_n,
        parts: vec![RuntimeTreePart {
            main_symbols: gu_oracle
                .main_rows
                .checked_mul(gu_oracle.main_deg as u64)
                .ok_or(Error::ResourceLimit)?,
            mask_symbols: 0,
        }],
        salt_bytes: request_plan.salt_bytes,
        run,
        configuration,
    };
    let gu = GuShape {
        oracle: LM_HEAD_GU_ORACLE,
        group: LM_HEAD_FOLD_GROUP,
        tree: gu_tree,
        part: 0,
        rows_high: compiled.once_foldeq.rows_high,
        u: GuOracleShape {
            code: gu_oracle.code,
            main_rows: gu_oracle.main_rows,
            main_deg: gu_oracle.main_deg,
            mask_rows: gu_oracle.mask_rows,
            mask_code: gu_oracle.mask_code,
            answers: gu_oracle.answers,
        },
        index: reader.shape,
    };

    let mask_tree = MaskTreePlan {
        tree_number: LM_HEAD_MASK_TREE,
        leaf_class: fold.mask_leaf_class,
        log_n: fold.mask_code.log_n,
        mask_rows: fold.mask_rows()?,
        salt_bytes: request_plan.salt_bytes,
        run,
        configuration,
    };
    let group = VerifiedGroup {
        group: LM_HEAD_FOLD_GROUP,
        members: vec![LM_HEAD_OMEGA_ORACLE],
        fresh: Some(LM_HEAD_FRESH_MASK),
        code: Some(fold.code),
        mask_code: gu.u.mask_code,
    };
    let verified = Arc::new(VerifiedStatement::new(
        VerifiedGroups {
            groups: vec![group],
        },
        vec![mask_tree],
        vec![OracleMetadata {
            id: LM_HEAD_FRESH_MASK,
            group: LM_HEAD_FOLD_GROUP,
            tree_number: mask_tree.tree_number,
            leaf_class: mask_tree.leaf_class,
            mask_rows: mask_tree.mask_rows,
            mask_code: fold.mask_code,
        }],
    )?);
    let mut oracles = RuntimeOracles::default();
    oracles.receive_mask(
        &verified,
        &mask_tree,
        LM_HEAD_FRESH_MASK,
        &fold_mask_notice(&fold)?,
        fresh_mask_root,
    )?;
    let layout = FoldGroupLayout {
        group: LM_HEAD_FOLD_GROUP,
        k: fold.k,
        h: fold.h,
        members: vec![
            FoldMemberLayout {
                oracle: LM_HEAD_GU_ORACLE,
                main_rows: gu.u.main_rows,
                mask_rows: gu.u.mask_rows,
                kept: false,
            },
            FoldMemberLayout {
                oracle: LM_HEAD_OMEGA_ORACLE,
                main_rows: omega.main_rows,
                mask_rows: omega.mask_rows,
                kept: false,
            },
        ],
    };
    if layout.fault(&fold).is_some() {
        return Err(Error::InvalidConfig);
    }

    let table_identity = reader_table_identity(reader)?;
    let binding_statement = BindingStatement::new(
        parameters.version_word,
        parameters.protocol_identity,
        case_id.to_vec(),
        statement.digest(),
        table_identity,
    )?;
    let binding = FoldeqBinding::new(&parameters, binding_statement, reader)?;

    let plan = FoldeqPlan {
        reader: reader.clone(),
        binding,
        edge: FoldeqEdgeBinding::NotAnEdge,
        fold,
        statement: verified,
        oracles: Arc::new(Mutex::new(oracles)),
        mask_tree_number: mask_tree.tree_number,
        group_id: LM_HEAD_FOLD_GROUP,
        leaf_class: compiled.once_foldeq.fold_leaf_class,
        seed_budget: reader.shape.answers as usize,
        ledger: Arc::new(Mutex::new(ReadLedger::new(LM_HEAD_COLUMN_DRAWS as u64))),
        column_draws: LM_HEAD_COLUMN_DRAWS,
        mask_query_count: 0,
        gamma_path: GammaPath::Lin {
            shape: LinShapeBinding {
                matrix_rows: 1u64 << request_plan.lin.matrix_rows_log,
                k_lo: request_plan.lin.k_lo,
                k_col: request_plan.lin.k_col,
            },
        },
        gu: Some(gu),
        fold_layout: Some(layout),
    };
    if plan.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    let binder = FoldeqBinder::new_with_random_bytes(
        SHARED_CHAIN_LABEL,
        configuration,
        &plan,
        Some(random_bytes),
    )?;
    let first_query = binder.first_query().to_vec();
    let chain = binder.chain()?;
    Ok(LimitedSharedSession {
        plan,
        binder,
        chain,
        first_query,
    })
}

/// Builds the shared-only session for the trusted output-head context. The
/// plan identifier and the case id are the context's; the caller supplies the
/// private random bytes and the provider's pre-committed fresh mask root.
pub fn build_limited_shared_session_for_context(
    context: &LmHeadContext,
    request_plan: &RequestPlan,
    compiled: &CompiledPlan,
    statement: &LimitedKeyStatement,
    reader: &IndexReader,
    random_bytes: Digest,
    fresh_mask_root: Digest,
) -> Result<LimitedSharedSession> {
    build_limited_shared_session(
        request_plan,
        compiled,
        statement,
        reader,
        context.plan_identifier,
        context.policy.case_id.as_bytes(),
        random_bytes,
        fresh_mask_root,
    )
}

/// The private values of one G_u commitment: the oracle code tails, one salt
/// per runtime-tree column and one pad per sample answer.
pub struct LmHeadGuRandomness {
    pub code_tails: Vec<Vec<Fp3>>,
    pub salts: Vec<Vec<u8>>,
    pub pads: Vec<Fp3>,
}

/// The provider's committed G_u oracle: the folded relation view, its root and
/// the registry entry that the answer rows are built from.
pub struct LmHeadGuProvider {
    shape: GuShape,
    witness: GuWitness,
    folded: Vec<Fp3>,
    root: Digest,
    registry: GuRegistry,
}

impl LmHeadGuProvider {
    pub fn shape(&self) -> &GuShape {
        &self.shape
    }

    pub fn relation(&self) -> &GuWitness {
        &self.witness
    }

    pub fn root(&self) -> Digest {
        self.root
    }

    /// The folded message before its placement in the G_u table. Omega is the
    /// block sum of this vector under the low column point.
    pub fn folded(&self) -> &[Fp3] {
        &self.folded
    }

    /// The complete ordered sample answers for the verifier's seeds. The
    /// target is the evaluation of the verifier's own coefficient row on the
    /// committed witness; it is not supplied by the verifier.
    pub fn answers(&self, seeds: &[SampleSeed]) -> Result<Vec<Fp3>> {
        let mut answers = Vec::with_capacity(self.shape.u.answers as usize);
        for answer in 0..self.shape.u.answers {
            let row = sample_row(&self.shape, &self.registry, seeds, answer, Fp3::ZERO)?;
            answers.push(row.evaluate(&self.witness)?);
        }
        Ok(answers)
    }
}

/// Folds the committed table's message rows with the session's equation
/// weights, places the folded message and its tail in the trusted G_u layout,
/// writes the private sample pads, and commits the G_u runtime tree. The G_u
/// message rows and salts are the caller's private values.
pub fn fold_lm_head_gu(
    key: &LmHeadKey,
    table_tails: &[Vec<Fp>],
    shape: &GuShape,
    gamma: &[Fp3],
    randomness: &LmHeadGuRandomness,
) -> Result<LmHeadGuProvider> {
    if key.shape != shape.index || key.reader.fault().is_some() {
        return Err(Error::StatementMismatch);
    }
    if shape.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    let rows = usize::try_from(shape.index.rows())
        .map_err(|_| Error::ResourceLimit)?;
    if gamma.len() != shape.index.v as usize
        || key.messages.len() != rows
        || table_tails.len() != rows
    {
        return Err(Error::ShapeMismatch);
    }
    let ell = usize::try_from(shape.index.code.ell)
        .map_err(|_| Error::ResourceLimit)?;
    let tail = usize::try_from(shape.index.code.r)
        .map_err(|_| Error::ResourceLimit)?;
    if key
        .messages
        .iter()
        .any(|message| message.len() != ell)
        || table_tails.iter().any(|values| values.len() != tail)
    {
        return Err(Error::ShapeMismatch);
    }
    let main_rows = usize::try_from(shape.u.main_rows)
        .map_err(|_| Error::ResourceLimit)?;
    let main_tail = usize::try_from(shape.u.code.r)
        .map_err(|_| Error::ResourceLimit)?;
    if randomness.code_tails.len() != main_rows
        || randomness
            .code_tails
            .iter()
            .any(|values| values.len() != main_tail)
        || randomness.salts.len() != usize::try_from(shape.tree.columns())
            .map_err(|_| Error::ResourceLimit)?
        || randomness
            .salts
            .iter()
            .any(|salt| salt.len() != shape.tree.salt_bytes)
    {
        return Err(Error::ShapeMismatch);
    }

    let eq_gamma = foldeq_weights(gamma, shape.index.v)?;
    let mut folded = vec![Fp3::ZERO; ell];
    let mut folded_tail = vec![Fp3::ZERO; tail];
    for (row, weight) in eq_gamma.iter().enumerate() {
        for (place, value) in key.messages[row].iter().enumerate() {
            folded[place] = folded[place].add(&weight.mul(&Fp3::from_base(*value)));
        }
        for (place, value) in table_tails[row].iter().enumerate() {
            folded_tail[place] =
                folded_tail[place].add(&weight.mul(&Fp3::from_base(*value)));
        }
    }

    let mut witness = gu_table(shape, &folded, &folded_tail)?;
    let (main_pads, mask_pads) = gu_sample_pad_offsets(shape)?;
    if !mask_pads.is_empty() || main_pads.len() != randomness.pads.len() {
        return Err(Error::ShapeMismatch);
    }
    for (offset, pad) in main_pads.iter().zip(&randomness.pads) {
        let slot = witness
            .main
            .get_mut(*offset)
            .ok_or(Error::ShapeMismatch)?;
        *slot = *pad;
    }

    let mut codewords = Vec::with_capacity(main_rows);
    let message_len = usize::try_from(shape.u.code.ell)
        .map_err(|_| Error::ResourceLimit)?;
    for row in 0..main_rows {
        let start = row.checked_mul(message_len).ok_or(Error::ResourceLimit)?;
        let end = start.checked_add(message_len).ok_or(Error::ResourceLimit)?;
        let message = witness
            .main
            .get(start..end)
            .ok_or(Error::ShapeMismatch)?;
        let tail = randomness
            .code_tails
            .get(row)
            .ok_or(Error::ShapeMismatch)?;
        codewords.push(encode_row_fp3(&shape.u.code, message, tail)?);
    }
    let tree = RuntimeTree::commit(
        shape.tree.clone(),
        &[RuntimePartRows {
            main: codewords,
            mask: Vec::new(),
            main_deg: shape.u.main_deg,
        }],
        &randomness.salts,
    )?;
    let root = *tree.root();
    let mut registry = GuRegistry::new();
    registry.receive(
        shape,
        &GuNotice {
            root,
            oracle: shape.oracle,
            tree_number: shape.tree.tree_number,
            part: shape.part,
        },
    )?;
    Ok(LmHeadGuProvider {
        shape: shape.clone(),
        witness,
        folded,
        root,
        registry,
    })
}

/// The private values of one Omega commitment: the oracle code tails, one
/// salt per runtime-tree column and one sample pad.
pub struct LmHeadOmegaRandomness {
    pub code_tails: Vec<Vec<Fp3>>,
    pub salts: Vec<Vec<u8>>,
    pub pad: Fp3,
}

/// The provider's committed Omega message: its row view, root and the plan
/// shape it was committed under.
pub struct LmHeadOmegaProvider {
    oracle: u64,
    group: u64,
    u: GuOracleShape,
    tree: RuntimeTreePlan,
    message: Vec<Fp3>,
    root: Digest,
}

impl LmHeadOmegaProvider {
    pub fn oracle(&self) -> u64 {
        self.oracle
    }

    pub fn group(&self) -> u64 {
        self.group
    }

    pub fn shape(&self) -> &GuOracleShape {
        &self.u
    }

    pub fn root(&self) -> Digest {
        self.root
    }

    pub fn message(&self) -> &[Fp3] {
        &self.message
    }

    /// The complete ordered sample answers for the verifier's seeds. It uses
    /// the same ordinary-oracle sample relation the verifier will apply.
    pub fn answers(&self, seeds: &[SampleSeed]) -> Result<Vec<Fp3>> {
        let witness = GuWitness {
            main: self.message.clone(),
            mask: Vec::new(),
        };
        let mut answers = Vec::with_capacity(self.u.answers as usize);
        for answer in 0..self.u.answers {
            let row = oracle_sample_row(
                self.oracle,
                self.group,
                &self.u,
                seeds,
                answer,
                Fp3::ZERO,
            )?;
            answers.push(row.evaluate(&witness)?);
        }
        Ok(answers)
    }
}

/// The block sums of the folded message under the low column point:
/// `Omega[y] = Σ_{low < 2^k_lo} eq(psi_lo, low) · folded[low + (y << k_lo)]`.
/// It reads only the matrix rows, never the folded tail.
pub fn omega_blocks(
    folded: &[Fp3],
    matrix_rows: u64,
    k_lo: u32,
    psi_lo: &[Fp3],
) -> Result<Vec<Fp3>> {
    if matrix_rows == 0 || psi_lo.len() != k_lo as usize || k_lo >= 32 {
        return Err(Error::InvalidConfig);
    }
    let weights = foldeq_weights(psi_lo, k_lo)?;
    let required = matrix_rows
        .checked_shl(k_lo)
        .ok_or(Error::ResourceLimit)?;
    if required > folded.len() as u64 {
        return Err(Error::ShapeMismatch);
    }
    let rows = usize::try_from(matrix_rows).map_err(|_| Error::ResourceLimit)?;
    let low_count = 1usize
        .checked_shl(k_lo)
        .ok_or(Error::ResourceLimit)?;
    let mut omega = Vec::with_capacity(rows);
    for row in 0..rows {
        let base = row.checked_mul(low_count).ok_or(Error::ResourceLimit)?;
        let mut value = Fp3::ZERO;
        for (low, weight) in weights.iter().enumerate() {
            value = value.add(&weight.mul(
                folded
                    .get(base + low)
                    .ok_or(Error::ShapeMismatch)?,
            ));
        }
        omega.push(value);
    }
    Ok(omega)
}

/// Commits the Omega message for the folded row. The omega vector occupies the
/// leading message slots; the sample pad sits in the reserved slot and the
/// code tails are the caller's private values. The runtime tree is the plan's.
#[allow(clippy::too_many_arguments)]
pub fn commit_lm_head_omega(
    gu: &LmHeadGuProvider,
    oracle: u64,
    group: u64,
    u: &GuOracleShape,
    tree: &RuntimeTreePlan,
    matrix_rows: u64,
    k_lo: u32,
    psi_lo: &[Fp3],
    randomness: &LmHeadOmegaRandomness,
) -> Result<LmHeadOmegaProvider> {
    if u.fault().is_some() || u.mask_rows != 0 || tree.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    let ell = usize::try_from(u.code.ell).map_err(|_| Error::ResourceLimit)?;
    let rows = usize::try_from(u.main_rows).map_err(|_| Error::ResourceLimit)?;
    let tail = usize::try_from(u.code.r).map_err(|_| Error::ResourceLimit)?;
    if u.answers != 1
        || randomness.code_tails.len() != rows
        || randomness
            .code_tails
            .iter()
            .any(|values| values.len() != tail)
        || randomness.salts.len() != usize::try_from(tree.columns())
            .map_err(|_| Error::ResourceLimit)?
        || randomness
            .salts
            .iter()
            .any(|salt| salt.len() != tree.salt_bytes)
    {
        return Err(Error::ShapeMismatch);
    }
    let omega = omega_blocks(gu.folded(), matrix_rows, k_lo, psi_lo)?;
    let mut message = vec![Fp3::ZERO; rows.checked_mul(ell).ok_or(Error::ResourceLimit)?];
    let pad_slot = ell - u.answers as usize;
    if omega.len() > pad_slot {
        return Err(Error::ShapeMismatch);
    }
    message[..omega.len()].copy_from_slice(&omega);
    message[pad_slot] = randomness.pad;
    let mut codewords = Vec::with_capacity(rows);
    for row in 0..rows {
        let start = row.checked_mul(ell).ok_or(Error::ResourceLimit)?;
        let end = start.checked_add(ell).ok_or(Error::ResourceLimit)?;
        codewords.push(encode_row_fp3(
            &u.code,
            message.get(start..end).ok_or(Error::ShapeMismatch)?,
            &randomness.code_tails[row],
        )?);
    }
    let committed = RuntimeTree::commit(
        tree.clone(),
        &[RuntimePartRows {
            main: codewords,
            mask: Vec::new(),
            main_deg: u.main_deg,
        }],
        &randomness.salts,
    )?;
    Ok(LmHeadOmegaProvider {
        oracle,
        group,
        u: *u,
        tree: tree.clone(),
        message,
        root: *committed.root(),
    })
}

/// The values one shared-only run records before the FOLD stage: the gamma,
/// the seed set, the provider's ordered answers, the drawn table columns and
/// the transcript value after the two joins.
pub struct LimitedSharedColumns {
    pub gamma: Vec<Fp3>,
    pub seeds: Vec<SampleSeed>,
    pub answers: Vec<Fp3>,
    pub queries: Vec<u64>,
    pub root: Digest,
    pub transcript: Digest,
}

/// Runs the shared-only chain through the record and column joins. The
/// provider folds the committed table after the gamma challenge, sends G_u,
/// answers the seeds, and opens the verifier's drawn table columns. The FOLD
/// group still needs its second member (Omega), so no FOLD round is entered
/// here.
pub fn run_limited_shared_to_columns(
    setup: LimitedSharedSession,
    key: &LmHeadKey,
    table_tails: &[Vec<Fp>],
    randomness: &LmHeadGuRandomness,
) -> Result<(FoldeqSession, LimitedSharedColumns)> {
    let mut session = FoldeqSession::open(setup.plan, setup.binder, setup.chain)?;
    session.bind_first_query(&setup.first_query)?;
    session.draw_gamma()?;
    let gamma = session
        .gamma()
        .ok_or(Error::InvalidSessionState)?
        .to_vec();
    let gu_shape = session
        .plan()
        .gu
        .as_ref()
        .ok_or(Error::InvalidConfig)?
        .clone();
    let provider = fold_lm_head_gu(key, table_tails, &gu_shape, &gamma, randomness)?;
    session.send_gu(&provider.root())?;
    session.shared_seeds()?;
    let seeds = session.gu_shared_seeds().to_vec();
    let answers = provider.answers(&seeds)?;
    for answer in &answers {
        session.receive_gu_answer(*answer)?;
    }
    session.join_gu_record()?;
    session.draw_queries()?;
    let queries = session.queries().to_vec();
    let positions = unique_sorted(&queries);
    let unique = open_shared_columns(
        &key.table,
        &positions,
        LM_HEAD_COLUMN_DRAWS as u64,
    )?;
    let mut openings = Vec::with_capacity(queries.len());
    for query in &queries {
        openings.push(
            unique
                .iter()
                .find(|opening| opening.position == *query)
                .cloned()
                .ok_or(Error::InvalidIndex)?,
        );
    }
    session.read_queries(&openings)?;
    session.join_gu_read_columns()?;
    let transcript = *session.transcript_value();
    Ok((
        session,
        LimitedSharedColumns {
            gamma,
            seeds,
            answers,
            queries,
            root: provider.root(),
            transcript,
        },
    ))
}

/// The checks that the shared-only chain must report before a limited tested
/// state may exist. Each flag is set by a real check of the chain; a false flag
/// keeps the state from being created.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SharedChainChecks {
    pub columns_verified: bool,
    pub record_rows_verified: bool,
    pub omega_verified: bool,
    pub iopp_all_stages_verified: bool,
    pub artifact_ended: bool,
    pub record_matches_root_and_configuration: bool,
    pub premises_hold: bool,
}

impl SharedChainChecks {
    pub fn all(&self) -> bool {
        self.columns_verified
            && self.record_rows_verified
            && self.omega_verified
            && self.iopp_all_stages_verified
            && self.artifact_ended
            && self.record_matches_root_and_configuration
            && self.premises_hold
    }
}

/// A limited tested state: the table's proximity and record passed the
/// shared-only checks under the stated premises. It is not the reference WF
/// tested state and carries no range claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LimitedTableTested {
    statement_digest: Digest,
    root: Digest,
    record_digest: Digest,
    configuration: Digest,
    shape: IndexShape,
    plan_version: u64,
}

impl LimitedTableTested {
    pub fn statement_digest(&self) -> Digest {
        self.statement_digest
    }

    pub fn root(&self) -> Digest {
        self.root
    }

    pub fn record_digest(&self) -> Digest {
        self.record_digest
    }

    pub fn configuration(&self) -> Digest {
        self.configuration
    }

    pub fn shape(&self) -> &IndexShape {
        &self.shape
    }

    pub fn plan_version(&self) -> u64 {
        self.plan_version
    }
}

/// Creates the limited tested state only when every required check passed and
/// the reader's root, record, configuration, shape and name match the
/// statement and the table's own derivation.
pub fn establish_limited_tested(
    statement: &LimitedKeyStatement,
    plan: &LimitedKeyPlan,
    reader: &IndexReader,
    name: &[u8],
    checks: &SharedChainChecks,
) -> Result<LimitedTableTested> {
    statement.validate()?;
    plan.validate()?;
    if reader.fault().is_some()
        || reader.shape != statement.table_shape
        || reader.root != statement.table_root
        || reader.configuration != statement.table_configuration
        || !checks.all()
    {
        return Err(Error::InvalidConfig);
    }
    let answers = reader.shape.answers as usize;
    let rows = usize::try_from(reader.shape.rows_checked()?).map_err(|_| Error::ResourceLimit)?;
    let record_rows = rows.checked_mul(answers).ok_or(Error::ResourceLimit)?;
    if reader.record.y.len() != record_rows {
        return Err(Error::ShapeMismatch);
    }
    let seeds = zkie_engine::pcs::request::table::record_seeds(&reader.shape, name, answers)?;
    if seeds.len() != answers {
        return Err(Error::InvalidConfig);
    }
    let expected_name = zkie_engine::pcs::request::table::index_name(
        &reader.root,
        &reader.shape,
        &reader.configuration,
    );
    if expected_name != name {
        return Err(Error::StatementMismatch);
    }
    Ok(LimitedTableTested {
        statement_digest: statement.digest(),
        root: reader.root,
        record_digest: zkie_core::artifact::digest_bytes(&reader.record.encode()),
        configuration: reader.configuration,
        shape: reader.shape,
        plan_version: plan.version,
    })
}


/// Draws and normalizes the column positions of the shared chain.
pub fn draw_shared_columns(chain: &mut Chain, log_n: u32, count: usize) -> Result<Vec<u64>> {
    let mut positions = Vec::with_capacity(count);
    for _ in 0..count {
        positions.push(chain.draw_position(log_n)?);
    }
    positions.sort_unstable();
    positions.dedup();
    Ok(positions)
}

/// The verifier's column check against the table's reader.
pub fn verify_shared_columns(
    reader: &IndexReader,
    positions: &[u64],
    openings: &[ColumnOpening],
) -> Result<Vec<Fp>> {
    verify_columns(reader, positions, openings)
}

/// The prover's charged opening of the drawn columns.
pub fn open_shared_columns(
    table: &CommittedTable,
    positions: &[u64],
    max_columns: u64,
) -> Result<Vec<ColumnOpening>> {
    let mut state = HidingState::new(table.reader.root, 1, max_columns);
    let charge = state.begin_charge(table.reader.root, 1, positions.len())?;
    table.open_columns_charged(&charge, positions)
}

/// Reports only the column check. The remaining shared checks stay false, so
/// the limited tested gate stays closed until they are produced.
pub fn shared_column_check(
    reader: &IndexReader,
    positions: &[u64],
    openings: &[ColumnOpening],
) -> Result<SharedChainChecks> {
    verify_shared_columns(reader, positions, openings)?;
    Ok(SharedChainChecks {
        columns_verified: true,
        ..SharedChainChecks::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    use crate::s7_context::load_gpt2_lm_head_context;
    use crate::provider::StatementData;

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .to_path_buf()
    }

    fn artifacts() -> Option<(PathBuf, PathBuf, PathBuf, PathBuf)> {
        let root = repo_root();
        let case = root.join("models/gpt2/case.toml");
        let policy = root.join("models/gpt2/case.policy");
        let base = root.join(
            "docs/benchmarks/20261110-0547-verify-bytes-109934bd-670a98c9/prod/gpt2-seq16",
        );
        let public = base.join("commit/public");
        let statement = base.join("proof/statement.bin");
        if case.exists()
            && policy.exists()
            && public.join("model-commitment.bin").exists()
            && statement.exists()
        {
            Some((case, policy, public, statement))
        } else {
            None
        }
    }

    fn context() -> Option<(LmHeadContext, RequestPlan)> {
        let (case, policy, public, statement) = artifacts()?;
        let statement_data: StatementData =
            bincode::deserialize(&std::fs::read(&statement).ok()?).ok()?;
        let context = load_gpt2_lm_head_context(
            &case,
            &policy,
            &public,
            &statement,
            statement_data.model_commitment,
        )
        .ok()?;
        let plan = zkie_models_gpt2::lm_head_plan::lm_head_plan().ok()?;
        let weight = context.asset_root.join(&context.asset.path);
        if !weight.exists() {
            return None;
        }
        Some((context, plan))
    }

    #[test]
    fn the_mapping_matches_sampled_source_entries() {
        let Some((context, plan)) = context() else {
            return;
        };
        for (table_row, matrix_row, low) in [(0, 0, 0), (0, 1023, 63), (1023, 0, 63), (17, 511, 31)] {
            let entry = lm_head_entry(&context, &plan, table_row, matrix_row, low)
                .expect("entry");
            let source = context.read_lm_head_row(matrix_row).expect("row");
            let column = (table_row * (1 << plan.lin.k_lo) + low) as usize;
            assert_eq!(entry, fp_from_signed_word(source[column] as u64));
        }
    }

    #[test]
    fn the_mapping_covers_the_matrix() {
        let Some((context, plan)) = context() else {
            return;
        };
        let rows = 1usize << plan.table.v;
        let factor = 1usize << plan.lin.k_lo;
        let columns = 1usize << plan.lin.k_col;
        assert_eq!(context.asset.rows as usize, rows);
        assert_eq!(rows * factor, columns);
        assert_eq!(context.asset.row_width as usize, columns);
    }

    #[test]
    fn a_wrong_asset_row_count_is_rejected() {
        let Some((mut context, plan)) = context() else {
            return;
        };
        context.asset.rows += 1;
        let randomness = empty_randomness();
        let error = build_lm_head_key(&context, &plan, Digest([0x42; 32]), &randomness, u64::MAX);
        assert!(error.is_err());
    }

    #[test]
    fn randomness_of_the_wrong_size_is_rejected() {
        let Some((context, plan)) = context() else {
            return;
        };
        let randomness = LmHeadKeyRandomness {
            tails: Vec::new(),
            salts: Vec::new(),
            pads: Vec::new(),
        };
        let error = build_lm_head_key(&context, &plan, Digest([0x42; 32]), &randomness, u64::MAX);
        assert!(error.is_err());
    }

    #[test]
    fn the_buffer_bound_is_checked_before_the_build() {
        let Some((context, plan)) = context() else {
            return;
        };
        let randomness = empty_randomness();
        let error = build_lm_head_key(&context, &plan, Digest([0x42; 32]), &randomness, 1);
        assert!(matches!(error, Err(Error::ResourceLimit)));
    }

    fn empty_randomness() -> LmHeadKeyRandomness {
        LmHeadKeyRandomness {
            tails: Vec::new(),
            salts: Vec::new(),
            pads: Vec::new(),
        }
    }

    #[test]
    fn the_limited_statement_binds_its_semantics_and_range() {
        let Some((context, plan)) = context() else {
            return;
        };
        let randomness = empty_randomness();
        let error = build_lm_head_key(
            &context,
            &plan,
            Digest([0x42; 32]),
            &randomness,
            u64::MAX,
        );
        assert!(error.is_err());
        let statement = LimitedKeyStatement {
            version: LIMITED_KEY_VERSION,
            model_commitment: context.model_id,
            table_root: Digest([0x11; 32]),
            table_configuration: Digest([0x22; 32]),
            table_shape: plan.table_shape().expect("shape"),
            matrix_rows: 1 << plan.table.v,
            k_lo: plan.lin.k_lo,
            k_col: plan.lin.k_col,
            valid_min: i32::MIN as i64,
            valid_max: i32::MAX as i64,
            no_wf: true,
            trusted_i32_mapping: true,
        };
        statement.validate().expect("statement");
        let digest = statement.digest();
        let mut other = statement.clone();
        other.table_root = Digest([0x33; 32]);
        assert_ne!(digest, other.digest());
        other.version = LIMITED_KEY_VERSION + 1;
        assert!(other.validate().is_err());
        let mut other = statement.clone();
        other.valid_max = i32::MAX as i64 + 1;
        assert!(other.validate().is_err());
        let mut other = statement.clone();
        other.model_commitment = Digest::ZERO;
        assert!(other.validate().is_err());
        let mut other = statement.clone();
        other.no_wf = false;
        assert!(other.validate().is_err());
    }

    fn test_reader(statement: &LimitedKeyStatement) -> (IndexReader, Vec<u8>) {
        use zkie_engine::pcs::request::table::{index_name, IndexRecord, IndexReader};
        let name = index_name(
            &statement.table_root,
            &statement.table_shape,
            &statement.table_configuration,
        );
        let rows = statement.table_shape.rows() as usize;
        let reader = IndexReader {
            shape: statement.table_shape,
            configuration: statement.table_configuration,
            root: statement.table_root,
            record: IndexRecord {
                y: vec![Fp3::ZERO; rows],
            },
        };
        (reader, name)
    }

    fn test_statement() -> LimitedKeyStatement {
        let plan = zkie_models_gpt2::lm_head_plan::lm_head_plan().expect("plan");
        let shape = plan.table_shape().expect("shape");
        LimitedKeyStatement {
            version: LIMITED_KEY_VERSION,
            model_commitment: Digest([0x11; 32]),
            table_root: Digest([0x22; 32]),
            table_configuration: Digest([0x33; 32]),
            table_shape: shape,
            matrix_rows: 1 << plan.table.v,
            k_lo: plan.lin.k_lo,
            k_col: plan.lin.k_col,
            valid_min: i32::MIN as i64,
            valid_max: i32::MAX as i64,
            no_wf: true,
            trusted_i32_mapping: true,
        }
    }

    #[test]
    fn the_limited_tested_state_needs_every_check() {
        let plan = zkie_models_gpt2::lm_head_plan::lm_head_plan().expect("plan");
        let compiled = zkie_models_gpt2::lm_head_plan::lm_head_compiled().expect("compiled");
        let key_plan = limited_key_plan(&plan, &compiled).expect("key plan");
        let statement = test_statement();
        let (reader, name) = test_reader(&statement);
        let checks = SharedChainChecks {
            columns_verified: true,
            record_rows_verified: true,
            omega_verified: true,
            iopp_all_stages_verified: true,
            artifact_ended: true,
            record_matches_root_and_configuration: true,
            premises_hold: true,
        };
        let tested = establish_limited_tested(&statement, &key_plan, &reader, &name, &checks)
            .expect("tested");
        assert_eq!(tested.root(), statement.table_root);
        assert_eq!(tested.statement_digest(), statement.digest());
        let mut missing = checks;
        missing.iopp_all_stages_verified = false;
        assert!(establish_limited_tested(&statement, &key_plan, &reader, &name, &missing).is_err());
        let mut missing = checks;
        missing.artifact_ended = false;
        assert!(establish_limited_tested(&statement, &key_plan, &reader, &name, &missing).is_err());
        let mut missing = checks;
        missing.premises_hold = false;
        assert!(establish_limited_tested(&statement, &key_plan, &reader, &name, &missing).is_err());
        let mut replaced = statement.clone();
        replaced.table_root = Digest([0x44; 32]);
        assert!(establish_limited_tested(&replaced, &key_plan, &reader, &name, &checks).is_err());
        let mut wrong_record = reader.clone();
        wrong_record.record.y.pop();
        assert!(establish_limited_tested(
            &statement,
            &key_plan,
            &wrong_record,
            &name,
            &checks
        )
        .is_err());
    }

    #[test]
    fn the_limited_key_plan_keeps_every_stage_in_order() {
        let Some((_context, plan)) = context() else {
            return;
        };
        let compiled = zkie_models_gpt2::lm_head_plan::lm_head_compiled().expect("compiled");
        let key_plan = limited_key_plan(&plan, &compiled).expect("key plan");
        key_plan.validate().expect("plan");
        let mut missing = key_plan.clone();
        missing.stages.pop();
        assert!(missing.validate().is_err());
        let mut reordered = key_plan.clone();
        reordered.stages.swap(1, 2);
        assert!(reordered.validate().is_err());
    }

    #[test]
    fn the_shared_column_check_follows_the_chain() {
        use zkie_engine::pcs::request::code::build_code_plan;
        use zkie_engine::pcs::request::table::TableDistances;
        let shape = IndexShape {
            code: build_code_plan(5, 8, 2).expect("code"),
            v: 2,
            answers: 1,
            salt_bytes: 40,
            distances: TableDistances {
                radius: 1,
                later_radius: 1,
                m_once: 0,
                m_later: 0,
            },
        };
        let messages: Vec<Vec<Fp>> = (0..4)
            .map(|row| (0..8).map(|column| Fp::from_u64((row * 8 + column) as u64)).collect())
            .collect();
        let tails: Vec<Vec<Fp>> = (0..4).map(|_| vec![Fp::ZERO; 2]).collect();
        let salts: Vec<Vec<u8>> = (0..32).map(|position| vec![position as u8; 40]).collect();
        let table = commit_table(shape, Digest([0x55; 32]), &messages, &tails, &salts)
            .expect("table");
        let mut chain = Chain::begin(
            0x1234,
            &table.reader.configuration,
            b"statement",
            &table.reader.record.encode(),
            None,
        )
        .expect("chain");
        let prefix = chain.message().prefix(0, 0).expect("prefix");
        assert!(chain.step(&prefix, 0).expect("step"));
        let positions = draw_shared_columns(&mut chain, shape.code.log_n, 4).expect("positions");
        assert!(!positions.is_empty());
        let openings = open_shared_columns(&table, &positions, 100).expect("openings");
        let columns = verify_shared_columns(&table.reader, &positions, &openings)
            .expect("columns");
        assert_eq!(columns.len(), positions.len() * 4);
        let checks = shared_column_check(&table.reader, &positions, &openings).expect("check");
        assert!(checks.columns_verified);
        assert!(!checks.all());
        let mut tampered = openings.clone();
        tampered[0].symbols[0] = Fp::from_u64(999);
        assert!(verify_shared_columns(&table.reader, &positions, &tampered).is_err());
    }

    fn shared_plan_fixture() -> (
        RequestPlan,
        CompiledPlan,
        LimitedKeyStatement,
        IndexReader,
        Digest,
    ) {
        use zkie_engine::pcs::request::table::IndexRecord;
        let request_plan =
            zkie_models_gpt2::lm_head_plan::lm_head_plan().expect("request plan");
        let compiled =
            zkie_models_gpt2::lm_head_plan::lm_head_compiled().expect("compiled");
        let plan_identifier = Digest([0x91; 32]);
        let parameters = limited_key_binding_parameters(&request_plan, plan_identifier)
            .expect("parameters");
        let configuration = parameters.digest().expect("configuration");
        let shape = request_plan.table_shape().expect("shape");
        let rows = usize::try_from(shape.rows()).expect("rows");
        let reader = IndexReader {
            shape,
            configuration,
            root: Digest([0x92; 32]),
            record: IndexRecord {
                y: vec![Fp3::ZERO; rows],
            },
        };
        let statement = LimitedKeyStatement {
            version: LIMITED_KEY_VERSION,
            model_commitment: Digest([0x93; 32]),
            table_root: reader.root,
            table_configuration: reader.configuration,
            table_shape: reader.shape,
            matrix_rows: 1 << request_plan.table.v,
            k_lo: request_plan.lin.k_lo,
            k_col: request_plan.lin.k_col,
            valid_min: i32::MIN as i64,
            valid_max: i32::MAX as i64,
            no_wf: true,
            trusted_i32_mapping: true,
        };
        (request_plan, compiled, statement, reader, plan_identifier)
    }

    fn build_shared(
        request_plan: &RequestPlan,
        compiled: &CompiledPlan,
        statement: &LimitedKeyStatement,
        reader: &IndexReader,
        plan_identifier: Digest,
        salt: Digest,
        mask_root: Digest,
    ) -> Result<LimitedSharedSession> {
        build_limited_shared_session(
            request_plan,
            compiled,
            statement,
            reader,
            plan_identifier,
            b"gpt2-seq16",
            salt,
            mask_root,
        )
    }

    #[test]
    fn the_limited_shared_plan_binds_the_real_configuration_run_and_trees() {
        let (request_plan, compiled, statement, reader, plan_identifier) =
            shared_plan_fixture();
        let salt = Digest([0x94; 32]);
        let mask_root = Digest([0x95; 32]);
        let session = build_shared(
            &request_plan,
            &compiled,
            &statement,
            &reader,
            plan_identifier,
            salt,
            mask_root,
        )
        .expect("shared session");
        let plan = &session.plan;
        assert!(plan.fault().is_none());
        assert_eq!(
            plan.reader.configuration,
            plan.binding.parameters_digest()
        );
        let expected_run = run_value(
            SHARED_CHAIN_LABEL,
            &plan.reader.configuration,
            &salt,
        )
        .expect("run");
        assert_eq!(session.binder.run(), Some(expected_run));
        let gu = plan.gu.as_ref().expect("G_u shape");
        assert_eq!(gu.tree.run, expected_run);
        assert_eq!(gu.tree.configuration, plan.reader.configuration);
        assert_eq!(gu.tree.log_n, gu.u.code.log_n);
        assert_eq!(gu.tree.parts.len(), 1);
        assert_eq!(
            gu.tree.parts[0].main_symbols,
            gu.u.main_rows * gu.u.main_deg as u64
        );
        assert_eq!(gu.tree.parts[0].mask_symbols, 0);
        let mask_tree = plan
            .statement
            .mask_tree(plan.mask_tree_number)
            .expect("mask tree");
        assert_eq!(mask_tree.run, expected_run);
        assert_eq!(mask_tree.configuration, plan.reader.configuration);
        assert_eq!(mask_tree.leaf_class, plan.fold.mask_leaf_class);
        assert_eq!(mask_tree.mask_rows, plan.fold.mask_rows().expect("rows"));
        assert_eq!(
            plan.gamma_path,
            GammaPath::Lin {
                shape: LinShapeBinding {
                    matrix_rows: 1u64 << request_plan.lin.matrix_rows_log,
                    k_lo: request_plan.lin.k_lo,
                    k_col: request_plan.lin.k_col,
                },
            }
        );
        assert_eq!(plan.column_draws, LM_HEAD_COLUMN_DRAWS);
        assert_eq!(
            plan.ledger.lock().expect("ledger").max_reads(),
            LM_HEAD_COLUMN_DRAWS as u64
        );
        assert_eq!(session.binder.start(), session.chain.value());
        assert_eq!(session.first_query, session.binder.first_query());
        plan.check_first_query(&session.first_query)
            .expect("first query");
    }

    #[test]
    fn the_limited_shared_plan_rejects_a_reader_with_another_configuration() {
        let (request_plan, compiled, statement, mut reader, plan_identifier) =
            shared_plan_fixture();
        reader.configuration = Digest([0x96; 32]);
        assert!(build_shared(
            &request_plan,
            &compiled,
            &statement,
            &reader,
            plan_identifier,
            Digest([0x94; 32]),
            Digest([0x95; 32]),
        )
        .is_err());
    }

    #[test]
    fn the_limited_shared_plan_rejects_a_statement_that_does_not_name_the_reader() {
        let (request_plan, compiled, statement, reader, plan_identifier) =
            shared_plan_fixture();
        let mut other = statement.clone();
        other.table_root = Digest([0x97; 32]);
        assert!(build_shared(
            &request_plan,
            &compiled,
            &other,
            &reader,
            plan_identifier,
            Digest([0x94; 32]),
            Digest([0x95; 32]),
        )
        .is_err());
        let mut other = statement.clone();
        other.k_col = request_plan.lin.k_col - 1;
        assert!(build_shared(
            &request_plan,
            &compiled,
            &other,
            &reader,
            plan_identifier,
            Digest([0x94; 32]),
            Digest([0x95; 32]),
        )
        .is_err());
    }

    #[test]
    fn the_limited_shared_plan_rejects_a_compiled_shape_that_does_not_fit() {
        let (request_plan, mut compiled, statement, reader, plan_identifier) =
            shared_plan_fixture();
        compiled.once.folded.h = 1;
        assert!(build_shared(
            &request_plan,
            &compiled,
            &statement,
            &reader,
            plan_identifier,
            Digest([0x94; 32]),
            Digest([0x95; 32]),
        )
        .is_err());
    }

    #[test]
    fn the_limited_shared_plan_needs_private_bytes_and_a_mask_root() {
        let (request_plan, compiled, statement, reader, plan_identifier) =
            shared_plan_fixture();
        assert!(build_shared(
            &request_plan,
            &compiled,
            &statement,
            &reader,
            plan_identifier,
            Digest::ZERO,
            Digest([0x95; 32]),
        )
        .is_err());
        assert!(build_shared(
            &request_plan,
            &compiled,
            &statement,
            &reader,
            plan_identifier,
            Digest([0x94; 32]),
            Digest::ZERO,
        )
        .is_err());
    }

    #[test]
    fn the_limited_shared_run_and_binding_follow_the_private_bytes() {
        let (request_plan, compiled, statement, reader, plan_identifier) =
            shared_plan_fixture();
        let first = build_shared(
            &request_plan,
            &compiled,
            &statement,
            &reader,
            plan_identifier,
            Digest([0x94; 32]),
            Digest([0x95; 32]),
        )
        .expect("first");
        let second = build_shared(
            &request_plan,
            &compiled,
            &statement,
            &reader,
            plan_identifier,
            Digest([0x96; 32]),
            Digest([0x95; 32]),
        )
        .expect("second");
        assert_ne!(first.binder.run(), second.binder.run());
        assert_ne!(first.binder.start(), second.binder.start());
        // The G_u tree run is a plan value in the first query, so a different
        // private run changes the first query as well as the chain start.
        assert_ne!(first.first_query, second.first_query);
    }

    #[test]
    fn the_selected_plan_identifier_enters_the_first_query() {
        let (request_plan, compiled, statement, reader, plan_identifier) =
            shared_plan_fixture();
        let first = build_shared(
            &request_plan,
            &compiled,
            &statement,
            &reader,
            plan_identifier,
            Digest([0x94; 32]),
            Digest([0x95; 32]),
        )
        .expect("first");
        let other_identifier = Digest([0x98; 32]);
        let parameters =
            limited_key_binding_parameters(&request_plan, other_identifier).expect("parameters");
        let mut other_reader = reader.clone();
        other_reader.configuration = parameters.digest().expect("configuration");
        let mut other_statement = statement.clone();
        other_statement.table_configuration = other_reader.configuration;
        let second = build_shared(
            &request_plan,
            &compiled,
            &other_statement,
            &other_reader,
            other_identifier,
            Digest([0x94; 32]),
            Digest([0x95; 32]),
        )
        .expect("second");
        assert_ne!(first.first_query, second.first_query);
    }


    fn small_gu_fixture() -> (
        LmHeadKey,
        GuShape,
        LmHeadGuRandomness,
        Vec<Vec<Fp>>,
        BindingParameters,
    ) {
        use zkie_engine::pcs::request::code::build_code_plan;
        use zkie_engine::pcs::request::table::TableDistances;
        let shape = IndexShape {
            code: build_code_plan(3, 4, 1).expect("code"),
            v: 2,
            answers: 1,
            salt_bytes: 40,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        };
        let messages: Vec<Vec<Fp>> = (0..4)
            .map(|row| {
                (0..4)
                    .map(|place| Fp::from_u64((row * 4 + place + 1) as u64))
                    .collect()
            })
            .collect();
        let tails: Vec<Vec<Fp>> = (0..4)
            .map(|row| vec![Fp::from_u64(row as u64 + 7)])
            .collect();
        let salts: Vec<Vec<u8>> = (0..8)
            .map(|position| vec![(position as u8) ^ 0x5a; 40])
            .collect();
        let request_parameters = RequestParams::default_with_code(3, 4, 1).expect("params");
        let parameters = BindingParameters::with_plan_parts(
            PROOF_VERSION_WORD,
            SESSION_PROTOCOL_IDENTITY,
            &request_parameters,
            &[Digest([0x30; 32])],
        )
        .expect("binding parameters");
        let configuration = parameters.digest().expect("configuration");
        let table = commit_table(shape, configuration, &messages, &tails, &salts)
            .expect("table");
        let reader = table.reader.clone();
        let name = table.name.clone();
        let key = LmHeadKey {
            shape,
            reader,
            table,
            name,
            messages,
        };
        let tree = RuntimeTreePlan {
            tree_number: 2,
            leaf_class: 0,
            log_n: shape.code.log_n,
            log_n_zk: shape.code.log_n,
            parts: vec![RuntimeTreePart {
                main_symbols: 12,
                mask_symbols: 0,
            }],
            salt_bytes: 40,
            run: Digest([0x32; 32]),
            configuration,
        };
        let gu = GuShape {
            oracle: 1,
            group: 1,
            tree,
            part: 0,
            rows_high: true,
            u: GuOracleShape {
                code: shape.code,
                main_rows: 4,
                main_deg: 3,
                mask_rows: 0,
                mask_code: None,
                answers: 1,
            },
            index: shape,
        };
        let randomness = LmHeadGuRandomness {
            code_tails: (0..4)
                .map(|row| vec![Fp3::from_u64(0x100 + row)])
                .collect(),
            salts: (0..8)
                .map(|position| vec![(position as u8) ^ 0xa5; 40])
                .collect(),
            pads: vec![Fp3::from_u64(0x777)],
        };
        (key, gu, randomness, tails, parameters)
    }

    #[test]
    fn the_g_u_fold_commits_the_folded_table_and_answers_the_seed() {
        let (key, shape, randomness, tails, _parameters) = small_gu_fixture();
        let gamma = vec![Fp3::from_u64(2), Fp3::from_u64(3)];
        let provider =
            fold_lm_head_gu(&key, &tails, &shape, &gamma, &randomness).expect("fold");
        assert_ne!(provider.root(), Digest::ZERO);
        let seeds = vec![SampleSeed {
            z: vec![Fp3::from_u64(5); 4],
            beta: Fp3::from_u64(7),
        }];
        let answers = provider.answers(&seeds).expect("answers");
        assert_eq!(answers.len(), 1);
        let mut registry = GuRegistry::new();
        registry
            .receive(
                &shape,
                &GuNotice {
                    root: provider.root(),
                    oracle: shape.oracle,
                    tree_number: shape.tree.tree_number,
                    part: shape.part,
                },
            )
            .expect("notice");
        let row = sample_row(&shape, &registry, &seeds, 0, answers[0]).expect("row");
        assert_eq!(
            row.evaluate(provider.relation()).expect("evaluate"),
            answers[0]
        );
    }

    #[test]
    fn the_g_u_fold_binds_the_challenge_and_the_private_pads() {
        let (key, shape, randomness, tails, _parameters) = small_gu_fixture();
        let gamma = vec![Fp3::from_u64(2), Fp3::from_u64(3)];
        let first =
            fold_lm_head_gu(&key, &tails, &shape, &gamma, &randomness).expect("first");
        let mut other_gamma = gamma.clone();
        other_gamma[0] = Fp3::from_u64(4);
        let second =
            fold_lm_head_gu(&key, &tails, &shape, &other_gamma, &randomness).expect("second");
        assert_ne!(first.root(), second.root());
        let changed = LmHeadGuRandomness {
            code_tails: randomness.code_tails.clone(),
            salts: randomness.salts.clone(),
            pads: vec![Fp3::from_u64(0x778)],
        };
        let third =
            fold_lm_head_gu(&key, &tails, &shape, &gamma, &changed).expect("third");
        assert_ne!(first.root(), third.root());
    }

    #[test]
    fn the_g_u_fold_rejects_mismatched_inputs() {
        let (key, shape, randomness, tails, _parameters) = small_gu_fixture();
        let gamma = vec![Fp3::from_u64(2), Fp3::from_u64(3)];
        assert!(fold_lm_head_gu(&key, &tails, &shape, &gamma[..1], &randomness).is_err());
        let mut wrong = LmHeadGuRandomness {
            code_tails: randomness.code_tails[..3].to_vec(),
            salts: randomness.salts.clone(),
            pads: randomness.pads.clone(),
        };
        assert!(fold_lm_head_gu(&key, &tails, &shape, &gamma, &wrong).is_err());
        wrong.code_tails = randomness.code_tails.clone();
        wrong.salts.pop();
        assert!(fold_lm_head_gu(&key, &tails, &shape, &gamma, &wrong).is_err());
        wrong.salts = randomness.salts.clone();
        wrong.pads.clear();
        assert!(fold_lm_head_gu(&key, &tails, &shape, &gamma, &wrong).is_err());
    }


    fn small_shared_run() -> Result<(FoldeqSession, LimitedSharedColumns)> {
        use zkie_engine::pcs::request::code::build_code_plan;
        let (key, gu, randomness, tails, parameters) = small_gu_fixture();
        let mask_code = build_code_plan(3, 7, 1)?;
        let fold = FoldShape {
            code: key.shape.code,
            mask_code,
            k: 3,
            h: 0,
            mask_leaf_class: 0,
        };
        let mask_tree = MaskTreePlan {
            tree_number: 0,
            leaf_class: 0,
            log_n: mask_code.log_n,
            mask_rows: 1,
            salt_bytes: 40,
            run: Digest([0x32; 32]),
            configuration: key.reader.configuration,
        };
        let statement = Arc::new(VerifiedStatement::new(
            VerifiedGroups {
                groups: vec![VerifiedGroup {
                    group: 1,
                    members: vec![2],
                    fresh: Some(3),
                    code: Some(gu.u.code),
                    mask_code: None,
                }],
            },
            vec![mask_tree],
            vec![OracleMetadata {
                id: 3,
                group: 1,
                tree_number: 0,
                leaf_class: 0,
                mask_rows: 1,
                mask_code,
            }],
        )?);
        let mut oracles = RuntimeOracles::default();
        oracles.receive_mask(
            &statement,
            &mask_tree,
            3,
            &fold_mask_notice(&fold)?,
            Digest([0x77; 32]),
        )?;
        let layout = FoldGroupLayout {
            group: 1,
            k: 3,
            h: 0,
            members: vec![
                FoldMemberLayout {
                    oracle: 1,
                    main_rows: 4,
                    mask_rows: 0,
                    kept: false,
                },
                FoldMemberLayout {
                    oracle: 2,
                    main_rows: 1,
                    mask_rows: 0,
                    kept: false,
                },
            ],
        };
        let table_identity = reader_table_identity(&key.reader)?;
        let binding_statement = BindingStatement::new(
            parameters.version_word,
            parameters.protocol_identity,
            b"small-shared".to_vec(),
            Digest([0x55; 32]),
            table_identity,
        )?;
        let binding = FoldeqBinding::new(&parameters, binding_statement, &key.reader)?;
        let plan = FoldeqPlan {
            reader: key.reader.clone(),
            binding,
            edge: FoldeqEdgeBinding::NotAnEdge,
            fold,
            statement,
            oracles: Arc::new(Mutex::new(oracles)),
            mask_tree_number: 0,
            group_id: 1,
            leaf_class: 0,
            seed_budget: 1,
            ledger: Arc::new(Mutex::new(ReadLedger::new(2))),
            column_draws: 2,
            mask_query_count: 0,
            gamma_path: GammaPath::Direct {
                columns: key.shape.v,
            },
            gu: Some(gu),
            fold_layout: Some(layout),
        };
        if plan.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        let binder = FoldeqBinder::new_with_random_bytes(
            SHARED_CHAIN_LABEL,
            key.reader.configuration,
            &plan,
            Some(Digest([0x44; 32])),
        )?;
        let first_query = binder.first_query().to_vec();
        let chain = binder.chain()?;
        let setup = LimitedSharedSession {
            plan,
            binder,
            chain,
            first_query,
        };
        run_limited_shared_to_columns(setup, &key, &tails, &randomness)
    }

    #[test]
    fn the_shared_chain_reaches_the_record_and_column_joins() {
        let (session, run) = small_shared_run().expect("shared run");
        assert_ne!(run.root, Digest::ZERO);
        assert_ne!(run.transcript, Digest::ZERO);
        assert_eq!(run.gamma.len(), 2);
        assert_eq!(run.seeds.len(), 1);
        assert_eq!(run.answers.len(), 1);
        assert_eq!(run.queries.len(), 2);
        assert_eq!(session.pending_gu_rows().len(), 4);
    }


    #[test]
    fn omega_blocks_sum_the_folded_matrix_rows() {
        let psi_lo = vec![Fp3::from_u64(2), Fp3::from_u64(3)];
        let folded: Vec<Fp3> = (0..16).map(|index| Fp3::from_u64(index + 1)).collect();
        let omega = omega_blocks(&folded, 4, 2, &psi_lo).expect("omega");
        let weights = foldeq_weights(&psi_lo, 2).expect("weights");
        for row in 0..4usize {
            let mut expected = Fp3::ZERO;
            for low in 0..4usize {
                expected = expected.add(&weights[low].mul(&folded[row * 4 + low]));
            }
            assert_eq!(omega[row], expected);
        }
        assert!(omega_blocks(&folded, 4, 3, &psi_lo).is_err());
        assert!(omega_blocks(&folded[..8], 4, 2, &psi_lo).is_err());
    }

    #[test]
    fn the_omega_message_commits_and_answers_the_seed() {
        use zkie_engine::pcs::request::code::build_code_plan;
        let (key, gu_shape, randomness, tails, _parameters) = small_gu_fixture();
        let gamma = vec![Fp3::from_u64(2), Fp3::from_u64(3)];
        let gu = fold_lm_head_gu(&key, &tails, &gu_shape, &gamma, &randomness)
            .expect("fold");
        let omega_code = build_code_plan(4, 8, 1).expect("omega code");
        let u = GuOracleShape {
            code: omega_code,
            main_rows: 1,
            main_deg: 3,
            mask_rows: 0,
            mask_code: None,
            answers: 1,
        };
        let tree = RuntimeTreePlan {
            tree_number: 3,
            leaf_class: 1,
            log_n: omega_code.log_n,
            log_n_zk: omega_code.log_n,
            parts: vec![RuntimeTreePart {
                main_symbols: 3,
                mask_symbols: 0,
            }],
            salt_bytes: 40,
            run: Digest([0x34; 32]),
            configuration: key.reader.configuration,
        };
        let psi_lo: Vec<Fp3> = Vec::new();
        let omega_randomness = LmHeadOmegaRandomness {
            code_tails: vec![vec![Fp3::from_u64(9)]],
            salts: (0..16)
                .map(|position| vec![(position as u8) ^ 0x33; 40])
                .collect(),
            pad: Fp3::from_u64(0x888),
        };
        let omega = commit_lm_head_omega(
            &gu,
            2,
            1,
            &u,
            &tree,
            4,
            0,
            &psi_lo,
            &omega_randomness,
        )
        .expect("omega commit");
        assert_ne!(omega.root(), Digest::ZERO);
        let seeds = vec![SampleSeed {
            z: vec![Fp3::from_u64(3); 3],
            beta: Fp3::from_u64(7),
        }];
        let answers = omega.answers(&seeds).expect("answers");
        assert_eq!(answers.len(), 1);
        let row = oracle_sample_row(2, 1, &u, &seeds, 0, answers[0]).expect("row");
        let witness = GuWitness {
            main: omega.message().to_vec(),
            mask: Vec::new(),
        };
        assert_eq!(row.evaluate(&witness).expect("eval"), answers[0]);
    }

}
