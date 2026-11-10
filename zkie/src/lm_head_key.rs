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
use zkie_engine::pcs::request::bat::{
    combine, group_index, BatOracleShape, BatPart, BatRow, BatRowKind, BatStatement,
};
use zkie_engine::pcs::request::binding::{BindingParameters, BindingStatement};
use zkie_engine::pcs::request::chain::{Chain, KIND_ALGEBRAIC, KIND_EPSILON};
use zkie_engine::pcs::request::cs_stage::{switch_tree_plan_multi, SwitchTarget, SwitchedClaims, SwitchedRows};
use zkie_engine::pcs::request::code::encode_row_fp3;
use zkie_engine::pcs::request::field::{Fp, Fp3, E_DEGREE};
use zkie_engine::pcs::request::fold::{
    fold_member_firsts, FoldPart,
    fold_mask_make, fold_mask_make_tail, fold_mask_notice, fold_mask_shape_fault,
    fold_output, fold_start_prove, fold_start_verify, FoldRoundVerifier, ByteMaskCoins,
    FoldGroupLayout, FoldInputRow, FoldMask, FoldMemberLayout,
    FoldOutput, FoldProverSession, FoldShape, FoldStartMessage, FoldStartProver, MaskOracle,
    OracleMetadata, RuntimeOracles,
    VerifiedGroup, VerifiedGroups, VerifiedStatement,
};
use zkie_engine::pcs::request::foldeq::{
    reader_table_identity, FoldeqBinder, FoldeqBinding, FoldeqEdgeBinding, FoldeqPlan,
    FoldeqSession, GammaPath, LinShapeBinding, OmegaShapePlan,
};
use zkie_engine::pcs::request::gu::{GuNotice, GuOracleShape, GuRegistry, GuShape};
use zkie_engine::pcs::request::params::{RequestParams, SHARED_CHAIN_LABEL};
use zkie_engine::pcs::request::plan::{CompiledPlan, RequestPlan};
use zkie_engine::pcs::request::read::{unique_sorted, HidingState, ReadLedger};
use zkie_engine::pcs::request::relation::{
    foldeq_weights, gu_sample_pad_offsets, gu_table, gu_vector_offset, oracle_sample_row,
    sample_row, GuRelationRow, GuWitness,
};
use zkie_engine::pcs::request::runtime_tree::{
    MaskTree, MaskTreePlan, RuntimePartRows, RuntimeTree, RuntimeTreeOpening, RuntimeTreePart,
    RuntimeTreePlan,
};
use zkie_engine::pcs::request::table::{
    commit_table, verify_columns, ColumnOpening, CommittedTable, IndexReader, IndexShape,
    SampleSeed,
};
use zkie_engine::primitives::transcript::run_value;
use zkie_engine::pcs::request::proximity::SingleRows;
use zkie_ops::proof::base_final_session::{fold_identity, FoldBasis, FoldMember};
use zkie_ops::proof::column_session::fp_from_signed_word;
use zkie_ops::proof::cs_multi_session::{CsMultiProver, CsMultiSession, OldMaskSource};
use zkie_ops::proof::fold_handoff::{FoldHandoff, FoldHandoffAnchor};
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
pub const LM_HEAD_OMEGA_TREE: u64 = 3;

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
    pub mask: LmHeadMaskProvider,
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
    mask_randomness: &LmHeadMaskRandomness,
) -> Result<LimitedSharedSession> {
    request_plan.validate()?;
    request_plan.validate_compiled(compiled)?;
    if random_bytes == Digest::ZERO || case_id.is_empty() {
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

    let omega_oracle = compiled.once_foldeq.omega.clone();
    let omega = OmegaShapePlan {
        oracle: LM_HEAD_OMEGA_ORACLE,
        group: LM_HEAD_FOLD_GROUP,
        tree: RuntimeTreePlan {
            tree_number: LM_HEAD_OMEGA_TREE,
            leaf_class: compiled.once_foldeq.omega_leaf_class,
            log_n: omega_oracle.code.log_n,
            log_n_zk: omega_oracle.code.log_n,
            parts: vec![RuntimeTreePart {
                main_symbols: omega_oracle
                    .main_rows
                    .checked_mul(omega_oracle.main_deg as u64)
                    .ok_or(Error::ResourceLimit)?,
                mask_symbols: 0,
            }],
            salt_bytes: request_plan.salt_bytes,
            run,
            configuration,
        },
        u: GuOracleShape {
            code: omega_oracle.code,
            main_rows: omega_oracle.main_rows,
            main_deg: omega_oracle.main_deg,
            mask_rows: omega_oracle.mask_rows,
            mask_code: omega_oracle.mask_code,
            answers: omega_oracle.answers,
        },
        rows_high: true,
        matrix_rows: 1u64 << request_plan.lin.matrix_rows_log,
        k_lo: request_plan.lin.k_lo,
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
    let mask_oracle = MaskOracle {
        id: LM_HEAD_FRESH_MASK,
        claimed_group: LM_HEAD_FOLD_GROUP,
        main_rows: 0,
        mask_rows: mask_tree.mask_rows,
        mask_code: fold.mask_code,
        answers: 0,
        sampled: false,
    };
    let mask = make_lm_head_mask(
        &fold,
        &mask_oracle,
        verified.groups(),
        mask_tree,
        mask_randomness,
    )?;
    let mut oracles = RuntimeOracles::default();
    oracles.receive_mask(
        &verified,
        &mask_tree,
        LM_HEAD_FRESH_MASK,
        &fold_mask_notice(&fold)?,
        mask.root(),
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
                main_rows: omega_oracle.main_rows,
                mask_rows: omega_oracle.mask_rows,
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
        omega: Some(omega),
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
        mask,
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
    mask_randomness: &LmHeadMaskRandomness,
) -> Result<LimitedSharedSession> {
    build_limited_shared_session(
        request_plan,
        compiled,
        statement,
        reader,
        context.plan_identifier,
        context.policy.case_id.as_bytes(),
        random_bytes,
        mask_randomness,
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
    main_tails: Vec<Vec<Fp3>>,
    root: Digest,
    registry: GuRegistry,
    tree: RuntimeTree,
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

    pub fn open(&self, positions: &[u64]) -> Result<RuntimeTreeOpening> {
        self.tree.open(positions)
    }

    pub fn main_tails(&self) -> &[Vec<Fp3>] {
        &self.main_tails
    }

    pub fn tree_plan(&self) -> RuntimeTreePlan {
        self.tree.plan().clone()
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
        main_tails: randomness.code_tails.clone(),
        root,
        registry,
        tree,
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
    committed: RuntimeTree,
    message: Vec<Fp3>,
    main_tails: Vec<Vec<Fp3>>,
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

    pub fn open(&self, positions: &[u64]) -> Result<RuntimeTreeOpening> {
        self.committed.open(positions)
    }

    pub fn main_tails(&self) -> &[Vec<Fp3>] {
        &self.main_tails
    }

    pub fn tree_plan(&self) -> RuntimeTreePlan {
        self.committed.plan().clone()
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
    let root = *committed.root();
    Ok(LmHeadOmegaProvider {
        oracle,
        group,
        u: *u,
        tree: tree.clone(),
        committed,
        message,
        main_tails: randomness.code_tails.clone(),
        root,
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
    pub omega_root: Option<Digest>,
    pub zeta: Vec<Fp3>,
    pub fold_output: Option<FoldOutput>,
    pub folded_main_message: Vec<Fp3>,
    pub fold_stage: Option<LimitedFoldStage>,
    pub cs_claims: Option<SwitchedClaims>,
    pub cs_message: Option<SwitchedRows>,
    pub stage_fold: Option<StageFoldOutcome>,
    pub stage_rows: Option<LimitedFoldStage>,
    pub pending_rows: usize,
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
    omega_randomness: Option<&LmHeadOmegaRandomness>,
    old_mask: Option<&LmHeadOldMaskProvider>,
    switch: Option<LimitedSwitchInputs>,
) -> Result<LimitedSharedColumns> {
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
    let mut omega_root = None;
    let mut zeta = Vec::new();
    let mut fold_output = None;
    let mut folded_main_message = Vec::new();
    let mut fold_stage = None;
    let mut cs_claims = None;
    let mut cs_message = None;
    let mut stage_fold = None;
    let mut stage_rows = None;
    let mut pending_rows = 0usize;
    let mut transcript = *session.transcript_value();
    if let Some(omega) = session.plan().omega.as_ref().cloned() {
        let omega_randomness = omega_randomness.ok_or(Error::InvalidConfig)?;
        let psi_lo: Vec<Fp3> = match session.psi() {
            Some(psi) => psi
                .get(..omega.k_lo as usize)
                .ok_or(Error::ShapeMismatch)?
                .to_vec(),
            None if omega.k_lo == 0 => Vec::new(),
            None => return Err(Error::StatementMismatch),
        };
        let omega_provider = commit_lm_head_omega(
            &provider,
            omega.oracle,
            omega.group,
            &omega.u,
            &omega.tree,
            omega.matrix_rows,
            omega.k_lo,
            &psi_lo,
            omega_randomness,
        )?;
        session.send_omega(&omega_provider.root())?;
        let seeds = session.omega_seeds()?;
        for answer in omega_provider.answers(&seeds)? {
            session.receive_omega_answer(answer)?;
        }
        session.join_omega()?;
        zeta = session.draw_zeta()?;
        omega_root = Some(omega_provider.root());
        if let Some(old_mask) = old_mask {
        let fold_shape = session.plan().fold;
        let mut bat = BatStatement::new();
        bat.add_oracle(BatOracleShape {
            oracle: 0,
            main_rows: provider.shape().u.main_rows,
            mask_rows: provider.shape().u.mask_rows,
            main_code: Some(provider.shape().u.code),
            mask_code: provider.shape().u.mask_code,
            answers: provider.shape().u.answers,
        })?;
        bat.add_oracle(BatOracleShape {
            oracle: 1,
            main_rows: omega_provider.shape().main_rows,
            mask_rows: 0,
            main_code: Some(omega_provider.shape().code),
            mask_code: None,
            answers: omega_provider.shape().answers,
        })?;
        let sample_g = session
            .pending_gu_rows()
            .iter()
            .find(|row| row.kind == zkie_engine::pcs::request::relation::GuRelationKind::Sample)
            .ok_or(Error::InvalidSessionState)?;
        let sample_o = session
            .omega_rows()
            .first()
            .ok_or(Error::InvalidSessionState)?;
        bat.sample(
            0,
            vec![bat_row_from_relation(
                sample_g,
                0,
                provider.shape().u.main_rows,
                provider.shape().u.code.ell,
                BatRowKind::Sample,
            )?],
        )?;
        bat.sample(
            1,
            vec![bat_row_from_relation(
                sample_o,
                1,
                omega_provider.shape().main_rows,
                omega_provider.shape().code.ell,
                BatRowKind::Sample,
            )?],
        )?;
        let group = bat.new_group(fold_shape.code, fold_shape.mask_code)?;
        bat.join(group, 0, false)?;
        bat.join(group, 1, false)?;
        for row in session.pending_gu_rows().iter().filter(|row| {
            row.kind != zkie_engine::pcs::request::relation::GuRelationKind::Sample
        }) {
            bat.add_row(
                group,
                bat_row_from_relation(
                    row,
                    0,
                    provider.shape().u.main_rows,
                    provider.shape().u.code.ell,
                    BatRowKind::Plain,
                )?,
            )?;
        }
        bat.add_row(
            group,
            bat_lin_fold_row(
                &provider,
                &omega_provider,
                &psi_lo,
                &zeta,
                omega.matrix_rows,
                omega.k_lo,
            )?,
        )?;
        let input = bat.begin_run(group)?;
        let x = session.draw_bat_challenge()?;
        let combined = combine(&bat, &input.layout, &input.rows, x)?;
        let mut fold_row = bat.to_fold_input_row(&combined, &input.layout)?;
        fold_row.group = session.plan().group_id;
        fold_row.members = vec![provider.shape().oracle, omega_provider.oracle()];
        let prover_row = fold_row.clone();
        session.fold_bind_row(fold_row)?;
        session.start_fold(setup.mask.message())?;
        let eps = session.epsilon().ok_or(Error::InvalidSessionState)?;
        let coins = ByteMaskCoins::new(setup.mask.coins_seed())?;
        let fold_layout = session
            .plan()
            .fold_layout
            .clone()
            .ok_or(Error::InvalidConfig)?;
        let main_messages = group_main_messages(&provider, &omega_provider, &input.layout)?;
        let old_mu = old_mask.mu()?;
    let mut prover = FoldProverSession::new(
            &fold_shape,
            &fold_layout,
            setup.mask.mask(),
            &prover_row,
            main_messages.clone(),
            old_mu.clone(),
            Vec::new(),
        )?;
    prover.start(eps, &coins)?;
        session.fold_rounds_start()?;
        for _ in 0..fold_shape.k {
            let round = prover.round()?;
            session.receive_fold_round(round[0], round[2])?;
            let gamma = *session
                .fold_gamma()
                .last()
                .ok_or(Error::InvalidSessionState)?;
            prover.bind(gamma)?;
        }
        let end = prover.end()?;
        let output = session.fold_finish()?;
        folded_main_message = prover.main_messages().to_vec();
        fold_stage = Some(build_limited_fold_stage(
            session.plan(),
            &setup.mask,
            &provider,
            &omega_provider,
            old_mask,
            &output,
            &folded_main_message,
            &end.mu,
            &end.fresh,
        )?);
        fold_output = Some(output);
        pending_rows = session.pending_gu_rows().len();
        transcript = *session.transcript_value();
        if let (Some(inputs), Some(stage)) = (switch, fold_stage.as_ref()) {
            let chain = session.take_chain_for_base()?;
            let first = run_switch(
                chain,
                stage,
                inputs.target,
                inputs.plan.clone(),
                inputs.main_tails.clone(),
                inputs.mask_tails.clone(),
                inputs.pad,
                &inputs.salts,
                SwitchMembers::Initial {
                    gu: &provider,
                    omega: &omega_provider,
                    old_mask,
                    fresh: &setup.mask,
                },
            )?;
            cs_claims = Some(first.claims.clone());
            cs_message = Some(first.message.clone());
            transcript = *first.chain.value();
            if let Some(spec) = inputs.stage_fold {
                let row = FoldInputRow {
                    group: spec.group,
                    k: spec.shape.k,
                    h: spec.shape.h,
                    members: vec![spec.member],
                    coefficients: first.claims.combined.main.clone(),
                    mask_coefficients: first.claims.combined.mask.clone(),
                    kept: Vec::new(),
                    target: first.claims.combined.target,
                };
                let outcome = run_stage_fold(
                    first.chain.clone(),
                    &spec,
                    row,
                    first.message.main.clone(),
                    first.message.mask.clone(),
                )?;
                transcript = *outcome.chain.value();
                let rows = build_stage_fold_stage(
                    &spec,
                    &outcome,
                    first.prover.root(),
                    first.prover.plan().clone(),
                    first.prover.main_tails(),
                    first.prover.mask_tails(),
                )?;
                if let Some(next) = inputs.next {
                    let second = run_switch(
                        outcome.chain.clone(),
                        &rows,
                        next.target,
                        next.plan.clone(),
                        next.main_tails.clone(),
                        next.mask_tails.clone(),
                        next.pad,
                        &next.salts,
                        SwitchMembers::Stage {
                            member: &first.prover,
                            fresh: &outcome.fresh,
                        },
                    )?;
                    transcript = *second.chain.value();
                    cs_claims = Some(second.claims);
                    cs_message = Some(second.message);
                }
                stage_fold = Some(outcome);
                stage_rows = Some(rows);
            }
        }
        }
    } else if omega_randomness.is_some() {
        return Err(Error::InvalidConfig);
    }
    Ok(LimitedSharedColumns {
        gamma,
        seeds,
        answers,
        queries,
        root: provider.root(),
        omega_root,
        zeta,
        fold_output,
        folded_main_message,
        fold_stage,
        cs_claims,
        cs_message,
        stage_fold,
        stage_rows,
        pending_rows,
        transcript,
    })
}

/// Builds the FOLD group's input row from the real G_u and Omega messages.
///
/// The G_u part maps every entry of the dense folded row (the `k_lo` place
/// bits and the `k_row` row bits) through the trusted G_u layout to its
/// message slot, weighted by `eq(psi_lo, place) * eq(zeta, row)`. The Omega
/// part is `eq(zeta, row)` on the leading slots of its single message row.
/// The fresh mask's coefficients are supplied by the mask provider; the
/// relation itself is homogeneous, so the row target is zero.
#[allow(clippy::too_many_arguments)]
pub fn build_fold_row(
    gu: &LmHeadGuProvider,
    omega: &LmHeadOmegaProvider,
    fold: &FoldShape,
    layout: &FoldGroupLayout,
    matrix_rows: u64,
    k_lo: u32,
    psi_lo: &[Fp3],
    zeta: &[Fp3],
    mask_coefficients: &[Fp3],
) -> Result<FoldInputRow> {
    if layout.fault(fold).is_some()
        || gu.shape().u.code != fold.code
        || omega.shape().code != fold.code
        || layout.members.len() != 2
        || layout.members[0].oracle != gu.shape().oracle
        || layout.members[1].oracle != omega.oracle()
        || layout.members[0].main_rows != gu.shape().u.main_rows
        || layout.members[1].main_rows != omega.shape().main_rows
    {
        return Err(Error::InvalidConfig);
    }
    if matrix_rows < 1 || !matrix_rows.is_power_of_two() || psi_lo.len() != k_lo as usize {
        return Err(Error::InvalidConfig);
    }
    let k_row = matrix_rows.trailing_zeros();
    if zeta.len() != k_row as usize {
        return Err(Error::ShapeMismatch);
    }
    let eq_psi = foldeq_weights(psi_lo, k_lo)?;
    let eq_zeta = foldeq_weights(zeta, k_row)?;
    let rows = 1usize
        .checked_shl(fold.k)
        .ok_or(Error::ResourceLimit)?;
    let ell = usize::try_from(fold.code.ell).map_err(|_| Error::ResourceLimit)?;
    let mut coefficients = vec![
        Fp3::ZERO;
        rows.checked_mul(ell).ok_or(Error::ResourceLimit)?
    ];
    let gu_base = 0usize;
    let omega_base = usize::try_from(layout.members[0].main_rows)
        .map_err(|_| Error::ResourceLimit)?
        .checked_mul(ell)
        .ok_or(Error::ResourceLimit)?;
    let entries = matrix_rows
        .checked_shl(k_lo)
        .ok_or(Error::ResourceLimit)?;
    let low_count = 1u64
        .checked_shl(k_lo)
        .ok_or(Error::ResourceLimit)?;
    for entry in 0..entries {
        let row = entry / low_count;
        let place = entry % low_count;
        let Some(offset) = gu_vector_offset(gu.shape(), entry)? else {
            return Err(Error::ShapeMismatch);
        };
        let weight = eq_psi[place as usize].mul(&eq_zeta[row as usize]);
        let slot = gu_base
            .checked_add(offset)
            .ok_or(Error::ResourceLimit)?;
        let target = coefficients.get_mut(slot).ok_or(Error::ShapeMismatch)?;
        *target = target.sub(&weight);
    }
    for (row, weight) in eq_zeta.iter().enumerate() {
        let slot = omega_base.checked_add(row).ok_or(Error::ResourceLimit)?;
        let target = coefficients.get_mut(slot).ok_or(Error::ShapeMismatch)?;
        *target = target.add(weight);
    }
    let mask_len = 1usize
        .checked_shl(fold.h)
        .and_then(|rows| {
            usize::try_from(fold.mask_code.ell)
                .ok()
                .and_then(|ell| rows.checked_mul(ell))
        })
        .ok_or(Error::ResourceLimit)?;
    if mask_coefficients.len() != mask_len {
        return Err(Error::ShapeMismatch);
    }
    let row = FoldInputRow {
        group: layout.group,
        k: fold.k,
        h: fold.h,
        members: layout.members.iter().map(|member| member.oracle).collect(),
        coefficients,
        mask_coefficients: mask_coefficients.to_vec(),
        kept: Vec::new(),
        target: Fp3::ZERO,
    };
    row.check(fold, layout)?;
    Ok(row)
}

/// The private values of the fresh FOLD mask: the coin seed and one salt per
/// committed mask-tree column.
pub struct LmHeadMaskRandomness {
    pub seed: Digest,
    pub salts: Vec<Vec<u8>>,
}

/// The provider's fresh mask: the start message with its committed runtime
/// tree, and the mask message coefficients the FOLD row consumes.
pub struct LmHeadMaskProvider {
    start: FoldStartProver,
    mask: FoldMask,
    coins_seed: Digest,
    mask_coefficients: Vec<Fp3>,
}

impl LmHeadMaskProvider {
    pub fn root(&self) -> Digest {
        self.start.message.root
    }

    pub fn message(&self) -> &FoldStartMessage {
        &self.start.message
    }

    pub fn coefficients(&self) -> &[Fp3] {
        &self.mask_coefficients
    }

    pub fn mask(&self) -> &FoldMask {
        &self.mask
    }

    pub fn coins_seed(&self) -> Digest {
        self.coins_seed
    }

    pub fn tree(&self) -> &MaskTree {
        &self.start.tree
    }

    pub fn open(&self, positions: &[u64]) -> Result<RuntimeTreeOpening> {
        self.start.tree.open(positions)
    }

    pub fn tails(&self) -> Result<&[Fp3]> {
        self.mask
            .tail()
            .map(|tail| tail.values.as_slice())
            .ok_or(Error::InvalidSessionState)
    }
}

/// Builds the fresh mask from the existing mask maker: draws the coefficients
/// and the encoding tail from private bytes, checks the mask against the
/// verifier's group registry, commits the runtime mask tree under the plan,
/// and returns the start message and the mask message coefficients. The root
/// must equal the fresh mask root registered when the plan was built.
pub fn make_lm_head_mask(
    fold: &FoldShape,
    oracle: &MaskOracle,
    groups: &VerifiedGroups,
    tree_plan: MaskTreePlan,
    randomness: &LmHeadMaskRandomness,
) -> Result<LmHeadMaskProvider> {
    if tree_plan.fault().is_some()
        || randomness.salts.len() != usize::try_from(tree_plan.columns())
            .map_err(|_| Error::ResourceLimit)?
        || randomness
            .salts
            .iter()
            .any(|salt| salt.len() != tree_plan.salt_bytes)
    {
        return Err(Error::ShapeMismatch);
    }
    let seed = randomness.seed;
    let mut coins = ByteMaskCoins::new(seed)?;
    let mut mask = fold_mask_make(fold, &mut coins)?;
    fold_mask_make_tail(fold, &mut mask, &mut coins)?;
    mask.sent_as(oracle.id)?;
    let start = fold_start_prove(
        fold,
        &mut mask,
        &coins,
        oracle,
        groups,
        tree_plan,
        &randomness.salts,
    )?;
    let mask_coefficients = start.draft.table.values.clone();
    Ok(LmHeadMaskProvider {
        start,
        mask,
        coins_seed: seed,
        mask_coefficients,
    })
}

/// The private values of the group's old mask message: its message rows, the
/// encoding tails and one salt per mask-tree column. This message is sent with
/// G_u; it is not the FOLD fresh mask.
pub struct LmHeadOldMaskRandomness {
    pub rows: Vec<Vec<Fp3>>,
    pub tails: Vec<Vec<Fp3>>,
    pub salts: Vec<Vec<u8>>,
}

/// The committed old mask message of the shared group.
pub struct LmHeadOldMaskProvider {
    root: Digest,
    rows: Vec<Vec<Fp3>>,
    tails: Vec<Vec<Fp3>>,
    tree: MaskTree,
}

impl LmHeadOldMaskProvider {
    pub fn root(&self) -> Digest {
        self.root
    }

    /// The mask message rows in the group layout, row-major. They are the
    /// `mu` half of the original group declaration.
    pub fn mu(&self) -> Result<Vec<Fp3>> {
        let mut values = Vec::new();
        for row in &self.rows {
            values.extend_from_slice(row);
        }
        Ok(values)
    }

    pub fn open(&self, positions: &[u64]) -> Result<RuntimeTreeOpening> {
        self.tree.open(positions)
    }

    pub fn tails(&self) -> &[Vec<Fp3>] {
        &self.tails
    }

    pub fn tree_plan(&self) -> MaskTreePlan {
        self.tree.plan
    }
}

/// Commits the group's old mask message under the plan's mask tree. The root
/// is the oracle root sent with G_u; the FOLD fresh mask is a separate
/// message with its own root.
pub fn commit_lm_head_old_mask(
    fold: &FoldShape,
    tree: MaskTreePlan,
    randomness: &LmHeadOldMaskRandomness,
) -> Result<LmHeadOldMaskProvider> {
    if tree.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    let committed = MaskTree::commit(
        tree,
        fold,
        &randomness.rows,
        &randomness.tails,
        &randomness.salts,
    )?;
    Ok(LmHeadOldMaskProvider {
        root: committed.root,
        rows: randomness.rows.clone(),
        tails: randomness.tails.clone(),
        tree: committed,
    })
}

fn bit_count(value: u64) -> u32 {
    if value <= 1 {
        0
    } else {
        64 - (value - 1).leading_zeros()
    }
}

/// Expands one oracle relation row (compact row-major rows x ell) into the
/// BAT term order, where an oracle table index is `row + (place << row_bits)`
/// and the virtual padded rows are zero.
fn bat_term_values(
    main: &[Fp3],
    main_rows: u64,
    ell: u64,
) -> Result<Vec<Fp3>> {
    let row_bits = bit_count(main_rows);
    let ell = usize::try_from(ell).map_err(|_| Error::ResourceLimit)?;
    let rows = usize::try_from(main_rows).map_err(|_| Error::ResourceLimit)?;
    let virtual_rows = 1usize
        .checked_shl(row_bits)
        .ok_or(Error::ResourceLimit)?;
    let mut values = vec![
        Fp3::ZERO;
        virtual_rows.checked_mul(ell).ok_or(Error::ResourceLimit)?
    ];
    for row in 0..rows {
        for place in 0..ell {
            let value = *main
                .get(row.checked_mul(ell).and_then(|base| base.checked_add(place)).ok_or(Error::ResourceLimit)?)
                .ok_or(Error::ShapeMismatch)?;
            values[row + (place << row_bits)] = value;
        }
    }
    Ok(values)
}

fn bat_row_from_relation(
    row: &GuRelationRow,
    oracle: u64,
    main_rows: u64,
    ell: u64,
    kind: BatRowKind,
) -> Result<BatRow> {
    let values = bat_term_values(&row.main, main_rows, ell)?;
    let mut out = BatRow::new(kind);
    out.terms.push(zkie_engine::pcs::request::bat::BatTerm {
        oracle,
        part: BatPart::Main,
        values,
    });
    out.target = row.target;
    Ok(out)
}

/// The LIN fold row as BAT terms: Omega at eq(zeta, row) on its leading
/// slots, and the folded row at eq(psi_lo, place) * eq(zeta, row) through the
/// trusted G_u layout.
fn bat_lin_fold_row(
    gu: &LmHeadGuProvider,
    omega: &LmHeadOmegaProvider,
    psi_lo: &[Fp3],
    zeta: &[Fp3],
    matrix_rows: u64,
    k_lo: u32,
) -> Result<BatRow> {
    let shape = gu.shape();
    let k_row = bit_count(matrix_rows);
    if !matrix_rows.is_power_of_two() || zeta.len() != k_row as usize {
        return Err(Error::ShapeMismatch);
    }
    let eq_psi = foldeq_weights(psi_lo, k_lo)?;
    let eq_zeta = foldeq_weights(zeta, k_row)?;
    let gu_rows = usize::try_from(shape.u.main_rows).map_err(|_| Error::ResourceLimit)?;
    let ell = usize::try_from(shape.u.code.ell).map_err(|_| Error::ResourceLimit)?;
    let row_bits = bit_count(shape.u.main_rows);
    let mut gu_values = vec![
        Fp3::ZERO;
        (1usize
            .checked_shl(row_bits)
            .ok_or(Error::ResourceLimit)?)
        .checked_mul(ell)
        .ok_or(Error::ResourceLimit)?
    ];
    let entries = matrix_rows
        .checked_shl(k_lo)
        .ok_or(Error::ResourceLimit)?;
    let low_count = 1u64.checked_shl(k_lo).ok_or(Error::ResourceLimit)?;
    for entry in 0..entries {
        let row = entry / low_count;
        let place = entry % low_count;
        let offset = gu_vector_offset(shape, entry)?.ok_or(Error::ShapeMismatch)?;
        if offset >= gu_rows * ell {
            return Err(Error::ShapeMismatch);
        }
        let real_row = offset / ell;
        let real_place = offset % ell;
        let weight = eq_psi[place as usize].mul(&eq_zeta[row as usize]);
        let at = real_row + (real_place << row_bits);
        gu_values[at] = gu_values[at].sub(&weight);
    }
    let omega_ell = usize::try_from(omega.shape().code.ell).map_err(|_| Error::ResourceLimit)?;
    let mut omega_values = vec![Fp3::ZERO; omega_ell];
    for (row, weight) in eq_zeta.iter().enumerate().take(matrix_rows as usize) {
        if row >= omega_values.len() {
            return Err(Error::ShapeMismatch);
        }
        omega_values[row] = omega_values[row].add(weight);
    }
    let mut out = BatRow::new(BatRowKind::Plain);
    out.terms.push(zkie_engine::pcs::request::bat::BatTerm {
        oracle: 0,
        part: BatPart::Main,
        values: gu_values,
    });
    out.terms.push(zkie_engine::pcs::request::bat::BatTerm {
        oracle: 1,
        part: BatPart::Main,
        values: omega_values,
    });
    Ok(out)
}

/// The group's main message table in the same virtual order as the combined
/// row: each oracle's message entries are placed at the BAT layout offset.
fn group_main_messages(
    gu: &LmHeadGuProvider,
    omega: &LmHeadOmegaProvider,
    layout: &zkie_engine::pcs::request::bat::BatGroupLayout,
) -> Result<Vec<Fp3>> {
    let mut out = vec![
        Fp3::ZERO;
        layout.group_len(BatPart::Main)?
    ];
    for member in &layout.members {
        let (main_rows, ell, compact): (u64, usize, Vec<Fp3>) = if member.oracle == 0 {
            (
                gu.shape().u.main_rows,
                usize::try_from(gu.shape().u.code.ell).map_err(|_| Error::ResourceLimit)?,
                gu.relation().main.clone(),
            )
        } else if member.oracle == 1 {
            (
                omega.shape().main_rows,
                usize::try_from(omega.shape().code.ell).map_err(|_| Error::ResourceLimit)?,
                omega.message().to_vec(),
            )
        } else {
            return Err(Error::StatementMismatch);
        };
        let row_bits = bit_count(main_rows);
        let first = member.main_first.ok_or(Error::ShapeMismatch)?;
        let rows = usize::try_from(main_rows).map_err(|_| Error::ResourceLimit)?;
        for row in 0..rows {
            for place in 0..ell {
                let index = row
                    .checked_add(place << row_bits)
                    .ok_or(Error::ResourceLimit)?;
                let at = usize::try_from(group_index(
                    index as u64,
                    row_bits,
                    first,
                    layout.k,
                )?)
                .map_err(|_| Error::ResourceLimit)?;
                let value = *compact
                    .get(row.checked_mul(ell).and_then(|base| base.checked_add(place)).ok_or(Error::ResourceLimit)?)
                    .ok_or(Error::ShapeMismatch)?;
                *out.get_mut(at).ok_or(Error::ShapeMismatch)? = value;
            }
        }
    }
    Ok(out)
}

/// The FOLD-to-CS handoff of one shared run: the verified handoff, the
/// original member commitments and the prover's two-mask single rows.
pub struct LimitedFoldStage {
    pub handoff: FoldHandoff,
    pub basis: FoldBasis,
    pub rows: SingleRows,
}

/// The private inputs of one switch stage: the next stage's target, the new
/// oracle tree plan and salts, the per-row encoding tails and the private
/// sample pad.
pub struct LimitedSwitchInputs {
    pub target: SwitchTarget,
    pub plan: RuntimeTreePlan,
    pub main_tails: Vec<Vec<Fp3>>,
    pub mask_tails: Vec<Vec<Fp3>>,
    pub pad: Fp3,
    pub salts: Vec<Vec<u8>>,
    pub stage_fold: Option<StageFoldSpec>,
    pub next: Option<Box<LimitedSwitchInputs>>,
}

/// Builds the FOLD-to-CS handoff from the verified FOLD output and the real
/// committed member trees. The single rows carry the folded main message, the
/// folded tails of the member encodings, the old mask message and the FOLD
/// fresh mask, with the handoff's main/mask/fresh claims.
#[allow(clippy::too_many_arguments)]
pub fn build_limited_fold_stage(
    plan: &FoldeqPlan,
    fresh: &LmHeadMaskProvider,
    gu: &LmHeadGuProvider,
    omega: &LmHeadOmegaProvider,
    old_mask: &LmHeadOldMaskProvider,
    output: &FoldOutput,
    folded_main: &[Fp3],
    old_mask_message: &[Fp3],
    fresh_message: &[Fp3],
) -> Result<LimitedFoldStage> {
    let layout = plan
        .fold_layout
        .clone()
        .ok_or(Error::InvalidConfig)?;
    let fresh_root = fresh.root();
    let anchor = FoldHandoffAnchor {
        shape: plan.fold,
        group: plan.group_id,
        members: layout.members.iter().map(|member| member.oracle).collect(),
        fresh: LM_HEAD_FRESH_MASK,
        fresh_root,
        tree_number: plan.mask_tree_number,
        layout: layout.clone(),
        fold_gamma: output.gamma.clone(),
        fold_epsilon: output.eps,
        fold_sum: output.sum,
    };
    let handoff = FoldHandoff::from_output(output, &anchor)?;
    if handoff.members.len() != 2
        || handoff.members[0] != gu.shape().oracle
        || handoff.members[1] != omega.oracle()
    {
        return Err(Error::StatementMismatch);
    }
    let mut members = Vec::with_capacity(2);
    for member in &layout.members {
        if member.oracle == gu.shape().oracle {
            members.push(FoldMember {
                oracle: member.oracle,
                main_code: gu.shape().u.code,
                mask_code: handoff.mask_code,
                root: gu.root(),
                tree: gu.tree_plan(),
            });
        } else if member.oracle == omega.oracle() {
            members.push(FoldMember {
                oracle: member.oracle,
                main_code: omega.shape().code,
                mask_code: handoff.mask_code,
                root: omega.root(),
                tree: omega.tree_plan(),
            });
        } else {
            return Err(Error::StatementMismatch);
        }
    }
    let basis = FoldBasis {
        layout: handoff.layout.clone(),
        fresh_root: handoff.fresh_root,
        fresh_tree: fresh.tree().plan,
        members,
    };
    let main_coeff = foldeq_weights(&handoff.gamma, handoff.k)?;
    let mask_coeff = foldeq_weights(&handoff.gamma[..handoff.h as usize], handoff.h)?;
    let main_firsts = fold_member_firsts(&layout, FoldPart::Main)?;
    let mask_firsts = fold_member_firsts(&layout, FoldPart::Mask)?;
    let main_tail_len = usize::try_from(handoff.code.r).map_err(|_| Error::ResourceLimit)?;
    let mask_tail_len = usize::try_from(handoff.mask_code.r).map_err(|_| Error::ResourceLimit)?;
    let mut main_tail = vec![Fp3::ZERO; main_tail_len];
    let mut mask_tail = vec![Fp3::ZERO; mask_tail_len];
    for (index, member) in layout.members.iter().enumerate() {
        let first = main_firsts[index];
        if member.main_rows > 0 && !member.kept {
            let first = first.ok_or(Error::StatementMismatch)?;
            let tails: &[Vec<Fp3>] = if member.oracle == gu.shape().oracle {
                gu.main_tails()
            } else if member.oracle == omega.oracle() {
                omega.main_tails()
            } else {
                return Err(Error::StatementMismatch);
            };
            for row in 0..usize::try_from(member.main_rows).map_err(|_| Error::ResourceLimit)? {
                let tail = tails.get(row).ok_or(Error::ShapeMismatch)?;
                let group = first
                    .checked_add(row as u64)
                    .ok_or(Error::ResourceLimit)?;
                let coefficient = *main_coeff
                    .get(group as usize)
                    .ok_or(Error::InvalidIndex)?;
                for (value, entry) in main_tail.iter_mut().zip(tail) {
                    *value = value.add(&coefficient.mul(entry));
                }
            }
        }
        if member.mask_rows > 0 {
            let first = mask_firsts[index].ok_or(Error::StatementMismatch)?;
            let _ = first;
            return Err(Error::UnsupportedPrimitive);
        }
    }
    let mut masks = vec![old_mask_message.to_vec(), fresh_message.to_vec()];
    let mut mask_tails = vec![
        old_mask
            .tails()
            .first()
            .cloned()
            .ok_or(Error::ShapeMismatch)?,
        fresh.tails()?.to_vec(),
    ];
    if masks.iter().any(|mask| mask.len() != handoff.mask_code.ell as usize)
        || mask_tails
            .iter()
            .any(|tail| tail.len() != handoff.mask_code.r as usize)
        || mask_coeff.len() != 1
    {
        return Err(Error::ShapeMismatch);
    }
    let rows = SingleRows {
        code: handoff.code,
        mask_code: handoff.mask_code,
        main: folded_main.to_vec(),
        main_tail,
        masks: std::mem::take(&mut masks),
        mask_tails: std::mem::take(&mut mask_tails),
        claim: handoff.main_claim.clone(),
        mask_claims: vec![handoff.mask_claim.clone(), handoff.fresh_claim.clone()],
        target: handoff.target,
    };
    rows.check()?;
    Ok(LimitedFoldStage {
        handoff,
        basis,
        rows,
    })
}

/// The plan inputs of one stage FOLD: its shape and group layout, the fresh
/// mask tree, the group/member/fresh numbers and the private mask randomness.
pub struct StageFoldSpec {
    pub shape: FoldShape,
    pub layout: FoldGroupLayout,
    pub mask_tree: MaskTreePlan,
    pub group: u64,
    pub member: u64,
    pub fresh: u64,
    pub randomness: LmHeadMaskRandomness,
}

/// The result of one stage FOLD: the verified output, the advanced chain and
/// the folded messages the next switch consumes.
pub struct StageFoldOutcome {
    pub output: FoldOutput,
    pub chain: Chain,
    pub end_mu: Vec<Fp3>,
    pub end_fresh: Vec<Fp3>,
    pub folded_main: Vec<Fp3>,
    pub fresh: LmHeadMaskProvider,
}

/// Runs one non-initial FOLD stage over a switched oracle message and a fresh
/// mask: the same prover/verifier round arithmetic as the shared chain, with
/// the stage's own shape and mask tree.
pub fn run_stage_fold(
    mut chain: Chain,
    spec: &StageFoldSpec,
    row: FoldInputRow,
    main_messages: Vec<Fp3>,
    mask_messages: Vec<Fp3>,
) -> Result<StageFoldOutcome> {
    let metadata = OracleMetadata {
        id: spec.fresh,
        group: spec.group,
        tree_number: spec.mask_tree.tree_number,
        leaf_class: spec.mask_tree.leaf_class,
        mask_rows: spec.mask_tree.mask_rows,
        mask_code: spec.shape.mask_code,
    };
    let statement = Arc::new(VerifiedStatement::new(
        VerifiedGroups {
            groups: vec![VerifiedGroup {
                group: spec.group,
                members: vec![spec.member],
                fresh: Some(spec.fresh),
                code: None,
                mask_code: None,
            }],
        },
        vec![spec.mask_tree],
        vec![metadata],
    )?);
    let mask_oracle = MaskOracle {
        id: spec.fresh,
        claimed_group: spec.group,
        main_rows: 0,
        mask_rows: spec.mask_tree.mask_rows,
        mask_code: spec.shape.mask_code,
        answers: 0,
        sampled: false,
    };
    let fresh = make_lm_head_mask(
        &spec.shape,
        &mask_oracle,
        statement.groups(),
        spec.mask_tree,
        &spec.randomness,
    )?;
    let mut oracles = RuntimeOracles::default();
    oracles.receive_mask(
        &statement,
        &spec.mask_tree,
        spec.fresh,
        &fold_mask_notice(&spec.shape)?,
        fresh.root(),
    )?;
    fold_start_verify(&spec.shape, fresh.message(), &statement, &oracles)?;
    let mut message = chain.message();
    message.push_u64(fresh.message().id);
    message.push_u64(fresh.message().claimed_group);
    message.push_u64(fresh.message().notice.mask_rows);
    message.push_u64(fresh.message().notice.mask_code.log_n as u64);
    message.push_u64(fresh.message().notice.mask_code.ell);
    message.push_u64(fresh.message().notice.mask_code.r);
    message.push_u64(fresh.message().notice.answers as u64);
    message.push_u64(fresh.message().notice.leaf_class as u64);
    for chunk in fresh.message().root.0.chunks(8) {
        let word = u64::from_le_bytes(chunk.try_into().map_err(|_| Error::InvalidEncoding)?);
        message.push_u64(word);
    }
    message.push_fp3(&fresh.message().sum);
    let prefix = message.prefix(KIND_EPSILON, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    let eps = chain.draw_e()?;
    let coins = ByteMaskCoins::new(spec.randomness.seed)?;
    let mut prover = FoldProverSession::new(
        &spec.shape,
        &spec.layout,
        fresh.mask(),
        &row,
        main_messages,
        mask_messages,
        Vec::new(),
    )?;
    if prover.sum() != fresh.message().sum {
        return Err(Error::StatementMismatch);
    }
    prover.start(eps, &coins)?;
    let mut claim = FoldRoundVerifier::new(fresh.message().sum, eps, row.target);
    let mut gammas = Vec::with_capacity(spec.shape.k as usize);
    for _ in 0..spec.shape.k {
        let round = prover.round()?;
        let mut message = chain.message();
        message.push_fp3(&round[0]);
        message.push_fp3(&round[2]);
        let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
        if !chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        let gamma = chain.draw_e()?;
        claim.round(round[0], round[2], gamma);
        gammas.push(gamma);
        prover.bind(gamma)?;
    }
    let end = prover.end()?;
    let output = fold_output(
        &spec.shape,
        &spec.layout,
        &row,
        spec.fresh,
        fresh.root(),
        spec.mask_tree.tree_number,
        eps,
        fresh.message().sum,
        &gammas,
        claim.claim(),
    )?;
    Ok(StageFoldOutcome {
        output,
        chain,
        end_mu: end.mu,
        end_fresh: end.fresh,
        folded_main: prover.main_messages().to_vec(),
        fresh,
    })
}

/// Builds the next FOLD-to-CS stage from the verified stage output and the
/// switched member's own commitment: the folded main message, the folded
/// per-row tails, the two mask messages and their folded tails. The member is
/// the switched oracle, so its old mask rows come from its mask part and its
/// basis carries that tree root and plan.
pub fn build_stage_fold_stage(
    spec: &StageFoldSpec,
    outcome: &StageFoldOutcome,
    member_root: Digest,
    member_tree: RuntimeTreePlan,
    main_tails: &[Vec<Fp3>],
    mask_tails: &[Vec<Fp3>],
) -> Result<LimitedFoldStage> {
    let output = &outcome.output;
    let fresh = &outcome.fresh;
    if spec.layout.members.len() != 1 || spec.layout.members[0].oracle != spec.member {
        return Err(Error::StatementMismatch);
    }
    let anchor = FoldHandoffAnchor {
        shape: spec.shape,
        group: spec.group,
        members: vec![spec.member],
        fresh: spec.fresh,
        fresh_root: fresh.root(),
        tree_number: spec.mask_tree.tree_number,
        layout: spec.layout.clone(),
        fold_gamma: output.gamma.clone(),
        fold_epsilon: output.eps,
        fold_sum: output.sum,
    };
    let handoff = FoldHandoff::from_output(output, &anchor)?;
    if handoff.members != vec![spec.member] {
        return Err(Error::StatementMismatch);
    }
    let basis = FoldBasis {
        layout: handoff.layout.clone(),
        fresh_root: handoff.fresh_root,
        fresh_tree: fresh.tree().plan,
        members: vec![FoldMember {
            oracle: spec.member,
            main_code: handoff.code,
            mask_code: handoff.mask_code,
            root: member_root,
            tree: member_tree,
        }],
    };
    let main_coeff = foldeq_weights(&handoff.gamma, handoff.k)?;
    let mask_coeff = foldeq_weights(&handoff.gamma[..handoff.h as usize], handoff.h)?;
    let main_rows = usize::try_from(spec.layout.members[0].main_rows)
        .map_err(|_| Error::ResourceLimit)?;
    let mask_rows = usize::try_from(spec.layout.members[0].mask_rows)
        .map_err(|_| Error::ResourceLimit)?;
    if main_tails.len() != main_rows
        || mask_tails.len() != mask_rows
        || main_coeff.len() != main_rows
        || mask_coeff.len() != mask_rows
    {
        return Err(Error::ShapeMismatch);
    }
    let main_tail_len = usize::try_from(handoff.code.r).map_err(|_| Error::ResourceLimit)?;
    let mask_tail_len =
        usize::try_from(handoff.mask_code.r).map_err(|_| Error::ResourceLimit)?;
    let mut main_tail = vec![Fp3::ZERO; main_tail_len];
    for (row, tail) in main_tails.iter().enumerate() {
        if tail.len() != main_tail_len {
            return Err(Error::ShapeMismatch);
        }
        for (value, entry) in main_tail.iter_mut().zip(tail) {
            *value = value.add(&main_coeff[row].mul(entry));
        }
    }
    let mut mask_tail = vec![Fp3::ZERO; mask_tail_len];
    for (row, tail) in mask_tails.iter().enumerate() {
        if tail.len() != mask_tail_len {
            return Err(Error::ShapeMismatch);
        }
        for (value, entry) in mask_tail.iter_mut().zip(tail) {
            *value = value.add(&mask_coeff[row].mul(entry));
        }
    }
    let fresh_values = fresh.tails()?;
    if fresh_values.len() != mask_rows * mask_tail_len {
        return Err(Error::ShapeMismatch);
    }
    let mut fresh_tail = vec![Fp3::ZERO; mask_tail_len];
    for (row, values) in fresh_values.chunks_exact(mask_tail_len).enumerate() {
        for (value, entry) in fresh_tail.iter_mut().zip(values) {
            *value = value.add(&mask_coeff[row].mul(entry));
        }
    }
    let masks = vec![outcome.end_mu.clone(), outcome.end_fresh.clone()];
    let mask_tails = vec![mask_tail, fresh_tail];
    let rows = SingleRows {
        code: handoff.code,
        mask_code: handoff.mask_code,
        main: outcome.folded_main.clone(),
        main_tail,
        masks,
        mask_tails,
        claim: handoff.main_claim.clone(),
        mask_claims: vec![handoff.mask_claim.clone(), handoff.fresh_claim.clone()],
        target: handoff.target,
    };
    rows.check()?;
    Ok(LimitedFoldStage {
        handoff,
        basis,
        rows,
    })
}

/// One switch stage's public result: the advanced chain, the registered
/// claims, the switched message and the prover that committed it.
struct SwitchRun {
    chain: Chain,
    claims: SwitchedClaims,
    message: SwitchedRows,
    prover: CsMultiProver,
}

/// Where one switch stage's member openings and old mask come from. The first
/// stage opens the shared group's members and its dedicated mask; every later
/// stage opens the previous switched member and folds the old mask out of its
/// mask part.
enum SwitchMembers<'a> {
    Initial {
        gu: &'a LmHeadGuProvider,
        omega: &'a LmHeadOmegaProvider,
        old_mask: &'a LmHeadOldMaskProvider,
        fresh: &'a LmHeadMaskProvider,
    },
    Stage {
        member: &'a CsMultiProver,
        fresh: &'a LmHeadMaskProvider,
    },
}

/// Runs one multi-row switch stage: commit the switched message, declare it,
/// draw the sample seed and positions, authenticate the old rows through the
/// supplied member and fresh commitments, and register the claims.
#[allow(clippy::too_many_arguments)]
fn run_switch(
    chain: Chain,
    stage: &LimitedFoldStage,
    target: SwitchTarget,
    plan: RuntimeTreePlan,
    main_tails: Vec<Vec<Fp3>>,
    mask_tails: Vec<Vec<Fp3>>,
    pad: Fp3,
    salts: &[Vec<u8>],
    members: SwitchMembers<'_>,
) -> Result<SwitchRun> {
    let prover = CsMultiProver::new(
        stage.rows.clone(),
        target,
        main_tails,
        mask_tails,
        pad,
        plan.clone(),
        salts,
    )?;
    let (source, fresh) = match &members {
        SwitchMembers::Initial {
            old_mask, fresh, ..
        } => (
            OldMaskSource::Tree {
                plan: old_mask.tree_plan(),
                root: old_mask.root(),
            },
            *fresh,
        ),
        SwitchMembers::Stage { fresh, .. } => (OldMaskSource::Member, *fresh),
    };
    let mut cs = CsMultiSession::new(
        chain,
        stage.handoff.clone(),
        stage.basis.clone(),
        target,
        *prover.shape(),
        plan,
        source,
    )?;
    cs.receive_declarations(fold_identity(&stage.handoff), prover.root())?;
    let seed = cs.draw_sample_seed()?;
    cs.receive_answer(prover.sample_answer(&seed)?)?;
    let (positions, _theta) = cs.draw_positions_theta(2)?;
    let unique = unique_sorted(&positions);
    let mask_positions: Vec<u64> = positions
        .iter()
        .map(|position| *position % stage.handoff.mask_code.n())
        .collect();
    let mask_unique = unique_sorted(&mask_positions);
    let (member_openings, old_mask_opening) = match &members {
        SwitchMembers::Initial {
            gu,
            omega,
            old_mask,
            ..
        } => (
            vec![gu.open(&unique)?, omega.open(&unique)?],
            Some(old_mask.open(&mask_unique)?),
        ),
        SwitchMembers::Stage { member, .. } => (vec![member.open(&unique)?], None),
    };
    let fresh_opening = fresh.open(&mask_unique)?;
    let claims = cs
        .authenticate_and_register(&member_openings, &fresh_opening, old_mask_opening.as_ref())?
        .clone();
    let message = prover.message().clone();
    let chain = cs.into_chain();
    Ok(SwitchRun {
        chain,
        claims,
        message,
        prover,
    })
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

    fn real_mask_randomness() -> &'static LmHeadMaskRandomness {
        static MASK: std::sync::OnceLock<LmHeadMaskRandomness> = std::sync::OnceLock::new();
        MASK.get_or_init(|| LmHeadMaskRandomness {
            seed: Digest([0x95; 32]),
            salts: (0..1usize << 14)
                .map(|position| vec![(position as u8) ^ 0x77; 40])
                .collect(),
        })
    }

    fn zero_seed_mask_randomness() -> &'static LmHeadMaskRandomness {
        static MASK: std::sync::OnceLock<LmHeadMaskRandomness> = std::sync::OnceLock::new();
        MASK.get_or_init(|| LmHeadMaskRandomness {
            seed: Digest::ZERO,
            salts: (0..1usize << 14)
                .map(|position| vec![(position as u8) ^ 0x77; 40])
                .collect(),
        })
    }

    fn build_shared(
        request_plan: &RequestPlan,
        compiled: &CompiledPlan,
        statement: &LimitedKeyStatement,
        reader: &IndexReader,
        plan_identifier: Digest,
        salt: Digest,
        mask_randomness: &LmHeadMaskRandomness,
    ) -> Result<LimitedSharedSession> {
        build_limited_shared_session(
            request_plan,
            compiled,
            statement,
            reader,
            plan_identifier,
            b"gpt2-seq16",
            salt,
            mask_randomness,
        )
    }

    #[test]
    fn the_limited_shared_plan_binds_the_real_configuration_run_and_trees() {
        let (request_plan, compiled, statement, reader, plan_identifier) =
            shared_plan_fixture();
        let salt = Digest([0x94; 32]);
        let mask_randomness = real_mask_randomness();
        let session = build_shared(
            &request_plan,
            &compiled,
            &statement,
            &reader,
            plan_identifier,
            salt,
            mask_randomness,
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
        let omega = plan.omega.as_ref().expect("Omega plan");
        assert_eq!(omega.oracle, LM_HEAD_OMEGA_ORACLE);
        assert_eq!(omega.group, LM_HEAD_FOLD_GROUP);
        assert_eq!(omega.tree.run, expected_run);
        assert_eq!(omega.tree.configuration, plan.reader.configuration);
        assert_eq!(omega.tree.leaf_class, 1);
        assert_eq!(omega.matrix_rows, 1u64 << request_plan.lin.matrix_rows_log);
        assert_eq!(omega.k_lo, request_plan.lin.k_lo);
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
            real_mask_randomness(),
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
            real_mask_randomness(),
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
            real_mask_randomness(),
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
            real_mask_randomness(),
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
            real_mask_randomness(),
        )
        .is_err());
        assert!(build_shared(
            &request_plan,
            &compiled,
            &statement,
            &reader,
            plan_identifier,
            Digest([0x94; 32]),
            zero_seed_mask_randomness(),
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
            real_mask_randomness(),
        )
        .expect("first");
        let second = build_shared(
            &request_plan,
            &compiled,
            &statement,
            &reader,
            plan_identifier,
            Digest([0x96; 32]),
            real_mask_randomness(),
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
            real_mask_randomness(),
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
            real_mask_randomness(),
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
            code: build_code_plan(4, 4, 1).expect("code"),
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
        let salts: Vec<Vec<u8>> = (0..16)
            .map(|position| vec![(position as u8) ^ 0x5a; 40])
            .collect();
        let request_parameters = RequestParams::default_with_code(4, 4, 1).expect("params");
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
            salts: (0..16)
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
        assert!(provider.open(&[0]).is_ok());
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


    fn small_shared_run() -> Result<LimitedSharedColumns> {
        use zkie_engine::pcs::request::code::build_code_plan;
        let (key, gu, randomness, tails, parameters) = small_gu_fixture();
        let mask_code = build_code_plan(4, 8, 1)?;
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
        let mask_oracle = MaskOracle {
            id: 3,
            claimed_group: 1,
            main_rows: 0,
            mask_rows: 1,
            mask_code,
            answers: 0,
            sampled: false,
        };
        let fresh_mask = make_lm_head_mask(
            &fold,
            &mask_oracle,
            statement.groups(),
            mask_tree,
            &LmHeadMaskRandomness {
                seed: Digest([0x77; 32]),
                salts: (0..16)
                    .map(|position| vec![(position as u8) ^ 0x66; 40])
                    .collect(),
            },
        )?;
        let mut oracles = RuntimeOracles::default();
        oracles.receive_mask(
            &statement,
            &mask_tree,
            3,
            &fold_mask_notice(&fold)?,
            fresh_mask.root(),
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
        let omega_code = key.shape.code;
        let omega = OmegaShapePlan {
            oracle: 2,
            group: 1,
            tree: RuntimeTreePlan {
                tree_number: 3,
                leaf_class: 1,
                log_n: omega_code.log_n,
                log_n_zk: omega_code.log_n,
                parts: vec![RuntimeTreePart {
                    main_symbols: 3,
                    mask_symbols: 0,
                }],
                salt_bytes: 40,
                run: Digest([0x36; 32]),
                configuration: key.reader.configuration,
            },
            u: GuOracleShape {
                code: omega_code,
                main_rows: 1,
                main_deg: 3,
                mask_rows: 0,
                mask_code: None,
                answers: 1,
            },
            rows_high: true,
            matrix_rows: 1,
            k_lo: 0,
        };
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
            omega: Some(omega),
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
            mask: fresh_mask,
        };
        let omega_randomness = LmHeadOmegaRandomness {
            code_tails: vec![vec![Fp3::from_u64(9)]],
            salts: (0..16)
                .map(|position| vec![(position as u8) ^ 0x33; 40])
                .collect(),
            pad: Fp3::from_u64(0x888),
        };
        let old_mask = commit_lm_head_old_mask(
            &fold,
            mask_tree,
            &LmHeadOldMaskRandomness {
                rows: vec![vec![Fp3::from_u64(0x91); mask_code.ell as usize]],
                tails: vec![vec![Fp3::from_u64(0x92)]],
                salts: (0..16)
                    .map(|position| vec![(position as u8) ^ 0x67; 40])
                    .collect(),
            },
        )?;
        let next_code = build_code_plan(5, 4, 1)?;
        let next_mask = build_code_plan(5, 8, 1)?;
        let target = SwitchTarget::from_stage(next_code, next_mask, 2, 2, 1)?;
        let main_tails: Vec<Vec<Fp3>> = (0..4)
            .map(|row| vec![Fp3::from_u64(0x200 + row as u64)])
            .collect();
        let mask_tails: Vec<Vec<Fp3>> = (0..4)
            .map(|row| vec![Fp3::from_u64(0x300 + row as u64)])
            .collect();
        let plan = switch_tree_plan_multi(
            1,
            &target,
            40,
            Digest([0x41; 32]),
            Digest([0x42; 32]),
        )?;
        let salts: Vec<Vec<u8>> = (0..plan.columns())
            .map(|position| vec![(position as u8) ^ 0x7a; 40])
            .collect();
        let target2 = SwitchTarget::from_stage(next_code, next_mask, 2, 2, 1)?;
        let plan2 = switch_tree_plan_multi(
            2,
            &target2,
            40,
            Digest([0x61; 32]),
            Digest([0x62; 32]),
        )?;
        let salts2: Vec<Vec<u8>> = (0..plan2.columns())
            .map(|position| vec![(position as u8) ^ 0x7b; 40])
            .collect();
        run_limited_shared_to_columns(
            setup,
            &key,
            &tails,
            &randomness,
            Some(&omega_randomness),
            Some(&old_mask),
            Some(LimitedSwitchInputs {
                target,
                plan,
                main_tails,
                mask_tails,
                pad: Fp3::from_u64(0x9a),
                salts,
                stage_fold: Some(StageFoldSpec {
                    shape: FoldShape {
                        code: next_code,
                        mask_code: next_mask,
                        k: 2,
                        h: 2,
                        mask_leaf_class: 0,
                    },
                    layout: FoldGroupLayout {
                        group: 1,
                        k: 2,
                        h: 2,
                        members: vec![FoldMemberLayout {
                            oracle: 1,
                            main_rows: 4,
                            mask_rows: 4,
                            kept: false,
                        }],
                    },
                    mask_tree: MaskTreePlan {
                        tree_number: 1,
                        leaf_class: 0,
                        log_n: next_mask.log_n,
                        mask_rows: 4,
                        salt_bytes: 40,
                        run: Digest([0x51; 32]),
                        configuration: key.reader.configuration,
                    },
                    group: 1,
                    member: 1,
                    fresh: 3,
                    randomness: LmHeadMaskRandomness {
                        seed: Digest([0x53; 32]),
                        salts: (0..32)
                            .map(|position| vec![(position as u8) ^ 0x5b; 40])
                            .collect(),
                    },
                }),
                next: Some(Box::new(LimitedSwitchInputs {
                    target: target2,
                    plan: plan2,
                    main_tails: (0..4)
                        .map(|row| vec![Fp3::from_u64(0x220 + row as u64)])
                        .collect(),
                    mask_tails: (0..4)
                        .map(|row| vec![Fp3::from_u64(0x320 + row as u64)])
                        .collect(),
                    pad: Fp3::from_u64(0x9b),
                    salts: salts2,
                    stage_fold: None,
                    next: None,
                })),
            }),
        )
    }

    #[test]
    fn the_shared_chain_reaches_the_record_and_column_joins() {
        let run = small_shared_run().expect("shared run");
        assert_ne!(run.root, Digest::ZERO);
        assert_ne!(run.transcript, Digest::ZERO);
        assert_eq!(run.gamma.len(), 2);
        assert_eq!(run.seeds.len(), 1);
        assert_eq!(run.answers.len(), 1);
        assert_eq!(run.queries.len(), 2);
        assert_eq!(run.pending_rows, 4);
        assert!(run.omega_root.is_some());
        assert_eq!(run.zeta.len(), 0);
        assert!(run.fold_output.is_some());
        let claims = run.cs_claims.as_ref().expect("cs claims");
        let message = run.cs_message.as_ref().expect("cs message");
        assert_eq!(claims.claims.len(), 8);
        assert_eq!(claims.combined.main.len(), 16);
        assert_eq!(claims.combined.mask.len(), 32);
        let mut value = Fp3::ZERO;
        for (coefficient, symbol) in claims.combined.main.iter().zip(&message.main) {
            value = value.add(&coefficient.mul(symbol));
        }
        for (coefficient, symbol) in claims.combined.mask.iter().zip(&message.mask) {
            value = value.add(&coefficient.mul(symbol));
        }
        assert_eq!(value, claims.combined.target);
        let stage = run.fold_stage.as_ref().expect("fold stage");
        assert_eq!(stage.handoff.code, stage.rows.code);
        assert_eq!(stage.handoff.k, 3);
        assert_eq!(stage.handoff.h, 0);
        assert_eq!(stage.basis.members.len(), 2);
        assert_eq!(stage.rows.main.len(), 4);
        assert_eq!(stage.rows.masks.len(), 2);
        assert_eq!(stage.rows.main_tail.len(), 1);
        assert_eq!(stage.rows.mask_tails.len(), 2);
        let stage_fold = run.stage_fold.as_ref().expect("stage fold");
        assert_eq!(stage_fold.output.k, 2);
        assert_eq!(stage_fold.output.h, 2);
        assert_eq!(stage_fold.output.members, vec![1]);
        assert_eq!(stage_fold.output.fresh, 3);
        let mut value = Fp3::ZERO;
        for (coefficient, symbol) in
            stage_fold.output.coefficients.iter().zip(&stage_fold.folded_main)
        {
            value = value.add(&coefficient.mul(symbol));
        }
        for (coefficient, symbol) in
            stage_fold.output.mask_coefficients.iter().zip(&stage_fold.end_mu)
        {
            value = value.add(&coefficient.mul(symbol));
        }
        for (coefficient, symbol) in
            stage_fold.output.fresh_coefficients.iter().zip(&stage_fold.end_fresh)
        {
            value = value.add(&coefficient.mul(symbol));
        }
        assert_eq!(value, stage_fold.output.target);
        let stage_rows = run.stage_rows.as_ref().expect("stage rows");
        assert_eq!(stage_rows.basis.members.len(), 1);
        assert_eq!(stage_rows.basis.members[0].oracle, 1);
        assert_eq!(stage_rows.rows.main.len(), 4);
        assert_eq!(stage_rows.rows.code.ell, 4);
        assert_eq!(Some(stage_rows.rows.mask_code), run.stage_fold.as_ref().map(|s| s.output.mask_code));
        assert_eq!(stage_rows.rows.masks.len(), 2);

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
        assert!(omega.open(&[0]).is_ok());
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


    #[test]
    fn the_fold_row_binds_the_g_u_and_omega_coefficients() {
        use zkie_engine::pcs::request::code::build_code_plan;
        let (key, gu_shape, randomness, tails, _parameters) = small_gu_fixture();
        let gamma = vec![Fp3::from_u64(2), Fp3::from_u64(3)];
        let gu = fold_lm_head_gu(&key, &tails, &gu_shape, &gamma, &randomness)
            .expect("fold");
        let omega_code = key.shape.code;
        let u = GuOracleShape {
            code: omega_code,
            main_rows: 1,
            main_deg: 3,
            mask_rows: 0,
            mask_code: None,
            answers: 1,
        };
        let omega_tree = RuntimeTreePlan {
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
            &omega_tree,
            1,
            0,
            &[],
            &omega_randomness,
        )
        .expect("omega");
        let mask_code = build_code_plan(3, 7, 1).expect("mask code");
        let fold = FoldShape {
            code: key.shape.code,
            mask_code,
            k: 3,
            h: 0,
            mask_leaf_class: 0,
        };
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
        let zeta: Vec<Fp3> = Vec::new();
        let mask_coefficients = vec![Fp3::from_u64(1); 7];
        let row = build_fold_row(
            &gu,
            &omega,
            &fold,
            &layout,
            1,
            0,
            &[],
            &zeta,
            &mask_coefficients,
        )
        .expect("fold row");
        assert_eq!(row.coefficients.len(), 8 * 4);
        assert_eq!(row.members, vec![1, 2]);
        assert_eq!(row.coefficients[0], Fp3::ZERO.sub(&Fp3::ONE));
        assert_eq!(row.coefficients[4 * 4], Fp3::ONE);
        assert_eq!(row.target, Fp3::ZERO);
    }


    #[test]
    fn the_fresh_mask_commits_and_is_reproducible_from_private_bytes() {
        use zkie_engine::pcs::request::code::build_code_plan;
        let mask_code = build_code_plan(3, 7, 1).expect("mask code");
        let fold = FoldShape {
            code: build_code_plan(3, 4, 1).expect("main code"),
            mask_code,
            k: 3,
            h: 0,
            mask_leaf_class: 0,
        };
        let groups = VerifiedGroups {
            groups: vec![VerifiedGroup {
                group: 1,
                members: vec![1, 2],
                fresh: Some(3),
                code: None,
                mask_code: None,
            }],
        };
        let oracle = MaskOracle {
            id: 3,
            claimed_group: 1,
            main_rows: 0,
            mask_rows: 1,
            mask_code,
            answers: 0,
            sampled: false,
        };
        let tree = MaskTreePlan {
            tree_number: 0,
            leaf_class: 0,
            log_n: mask_code.log_n,
            mask_rows: 1,
            salt_bytes: 40,
            run: Digest([0x37; 32]),
            configuration: Digest([0x38; 32]),
        };
        let salts: Vec<Vec<u8>> = (0..8)
            .map(|position| vec![(position as u8) ^ 0x44; 40])
            .collect();
        let first = make_lm_head_mask(
            &fold,
            &oracle,
            &groups,
            tree,
            &LmHeadMaskRandomness {
                seed: Digest([0x39; 32]),
                salts: salts.clone(),
            },
        )
        .expect("mask");
        assert_ne!(first.root(), Digest::ZERO);
        assert!(first.open(&[0]).is_ok());
        assert_eq!(first.coefficients().len(), 7);
        assert_eq!(first.message().id, 3);
        let second = make_lm_head_mask(
            &fold,
            &oracle,
            &groups,
            tree,
            &LmHeadMaskRandomness {
                seed: Digest([0x39; 32]),
                salts,
            },
        )
        .expect("second mask");
        assert_eq!(first.root(), second.root());
        assert_eq!(first.coefficients(), second.coefficients());
    }


    #[test]
    fn the_old_mask_commits_its_message_rows_separately_from_the_fresh_mask() {
        use zkie_engine::pcs::request::code::build_code_plan;
        let mask_code = build_code_plan(3, 7, 1).expect("mask code");
        let fold = FoldShape {
            code: build_code_plan(3, 4, 1).expect("main code"),
            mask_code,
            k: 3,
            h: 0,
            mask_leaf_class: 0,
        };
        let tree = MaskTreePlan {
            tree_number: 0,
            leaf_class: 0,
            log_n: mask_code.log_n,
            mask_rows: 1,
            salt_bytes: 40,
            run: Digest([0x51; 32]),
            configuration: Digest([0x52; 32]),
        };
        let rows = vec![vec![Fp3::from_u64(0x60); 7]];
        let tails = vec![vec![Fp3::from_u64(0x61)]];
        let salts: Vec<Vec<u8>> = (0..8)
            .map(|position| vec![(position as u8) ^ 0x55; 40])
            .collect();
        let provider = commit_lm_head_old_mask(
            &fold,
            tree,
            &LmHeadOldMaskRandomness {
                rows: rows.clone(),
                tails,
                salts,
            },
        )
        .expect("old mask");
        assert_ne!(provider.root(), Digest::ZERO);
        assert!(provider.open(&[0]).is_ok());
        assert_eq!(provider.mu().expect("mu"), rows.concat());
    }

}
