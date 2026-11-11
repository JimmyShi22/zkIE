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

use serde::{Deserialize, Serialize};

use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::compute::{MemoryGovernor, MemoryRequest};
use zkie_core::error::{Error, Result};
use zkie_core::proof_header::PROOF_VERSION_WORD;
use zkie_core::types::{Digest, DType, TensorSpec};
use zkie_engine::pcs::request::bat::{
    carry_compact_table, combine, group_index, BatOracleShape, BatPart, BatRow, BatRowKind,
    BatStatement,
};
use zkie_engine::pcs::request::binding::{BindingParameters, BindingStatement};
use zkie_engine::pcs::request::chain::{Chain, KIND_ALGEBRAIC, KIND_EPSILON};
use zkie_engine::pcs::request::cs_stage::{
    switch_tree_plan_multi, SwitchTarget, SwitchedClaims, SwitchedRows,
};
use zkie_engine::pcs::request::code::encode_row_fp3;
use zkie_engine::pcs::request::field::{Fp, Fp3, E_DEGREE};
use zkie_engine::pcs::request::fold::{
    fold_member_firsts, FoldPart,
    fold_mask_make, fold_mask_make_tail, fold_mask_notice, fold_mask_shape_fault,
    fold_output, fold_start_prove, fold_start_verify, FoldRoundVerifier, ByteMaskCoins, MaskCoins,
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
use zkie_engine::pcs::request::read::{unique_sorted, ReadLedger};
use zkie_engine::pcs::request::relation::{
    foldeq_weights, gu_sample_pad_offsets, gu_table, gu_vector_offset, oracle_sample_row,
    sample_row, GuRelationRow, GuWitness,
};
use zkie_engine::pcs::request::runtime_tree::{
    MaskTree, MaskTreePlan, RuntimePartRows, RuntimeTree, RuntimeTreeOpening, RuntimeTreePart,
    RuntimeTreePlan,
};
use zkie_engine::storage::mmap::BoundedAssetReader;
use zkie_engine::pcs::request::table::{
    commit_table_streaming, verify_columns, ColumnOpening, IndexReader, IndexShape, SampleSeed,
    EncodedRowStore, StreamingHolder, TableRowSource, VecTableSource,
};
use zkie_engine::memory::BudgetedMemoryGovernor;
use zkie_engine::primitives::transcript::run_value;
use zkie_engine::pcs::request::proximity::SingleRows;
use zkie_ops::proof::base_final_session::{fold_identity, FoldBasis, FoldMember};
use zkie_ops::proof::column_session::fp_from_signed_word;
use zkie_ops::proof::base_last_session::{
    encode_last_base_proof, verify_last_base_proof, LastBaseProver, LastBaseRandomness,
    LastBaseSession,
};
use zkie_ops::proof::cs_multi_session::{CsMultiProver, CsMultiSession, OldMaskSource};
use zkie_ops::proof::fold_handoff::{FoldHandoff, FoldHandoffAnchor};
use zkie_ops::proof::session_baseline::SESSION_PROTOCOL_IDENTITY;

use crate::policy::LimitedTableIdentity;
use crate::s7_context::LmHeadContext;

/// One LIN request declaration of the output-head relation. It names the
/// trusted MatMul step, its activation source and its output consumer, binds
/// the weight asset identity and the matrix shape, and carries the canonical
/// digests of X, Y and the final output. It is a declaration, not a proof; the
/// LIN first/shared/second checks and the trace binding are verified beside
/// it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinRequestDeclaration {
    pub version: u64,
    pub model_id: Digest,
    pub plan_identifier: Digest,
    pub activation_step: u64,
    pub activation_output: u64,
    pub matmul_step: u64,
    pub output_step: u64,
    pub weight_asset_id: u64,
    pub weight_table_identity: Digest,
    pub matrix_rows: u64,
    pub k_col: u32,
    pub k_lo: u32,
    pub activation_digest: Digest,
    pub output_digest: Digest,
    pub final_output_digest: Digest,
}

pub const LIN_REQUEST_DECLARATION_VERSION: u64 = 1;
const LIN_REQUEST_DOMAIN: &[u8] = b"zkie/lin-request/v1";

impl LinRequestDeclaration {
    pub fn validate(&self) -> Result<()> {
        if self.version != LIN_REQUEST_DECLARATION_VERSION
            || self.model_id == Digest::ZERO
            || self.plan_identifier == Digest::ZERO
            || self.weight_table_identity == Digest::ZERO
            || self.activation_digest == Digest::ZERO
            || self.output_digest == Digest::ZERO
            || self.final_output_digest == Digest::ZERO
        {
            return Err(Error::InvalidConfig);
        }
        if !(self.activation_step < self.matmul_step && self.matmul_step < self.output_step) {
            return Err(Error::InvalidConfig);
        }
        if self.matrix_rows == 0
            || !self.matrix_rows.is_power_of_two()
            || self.k_lo >= self.k_col
            || self.k_col > 63
        {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        use bincode::Options;
        let mut bytes = self.version.to_le_bytes().to_vec();
        bytes.extend_from_slice(LIN_REQUEST_DOMAIN);
        bytes.extend_from_slice(
            &bincode::DefaultOptions::new()
                .with_fixint_encoding()
                .serialize(self)
                .map_err(|_| Error::InvalidEncoding)?,
        );
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        use bincode::Options;
        if bytes.len() < 8 + LIN_REQUEST_DOMAIN.len() {
            return Err(Error::InvalidEncoding);
        }
        let version = u64::from_le_bytes(
            bytes[..8]
                .try_into()
                .map_err(|_| Error::InvalidEncoding)?,
        );
        if version != LIN_REQUEST_DECLARATION_VERSION || &bytes[8..8 + LIN_REQUEST_DOMAIN.len()] != LIN_REQUEST_DOMAIN {
            return Err(Error::InvalidConfig);
        }
        let value: Self = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .reject_trailing_bytes()
            .deserialize(&bytes[8 + LIN_REQUEST_DOMAIN.len()..])
            .map_err(|_| Error::InvalidEncoding)?;
        value.validate()?;
        Ok(value)
    }

    pub fn digest(&self) -> Result<Digest> {
        let mut writer = zkie_engine::primitives::codec::CanonicalWriter::new();
        writer.put_bytes(LIN_REQUEST_DOMAIN);
        writer.put_bytes(&self.encode()?);
        Ok(zkie_core::artifact::digest_bytes(&writer.into_bytes()))
    }
}

/// Canonical, domain-separated digest of one tensor: a label, the shape and
/// the values in little-endian two's complement. The label keeps values from
/// two positions or two tensors from colliding.
pub fn lin_values_digest(label: &[u8], shape: &[u64], values: &[i64]) -> Digest {
    let mut writer = zkie_engine::primitives::codec::CanonicalWriter::new();
    writer.put_bytes(label);
    writer.put_u64(shape.len() as u64);
    for dim in shape {
        writer.put_u64(*dim);
    }
    writer.put_u64(values.len() as u64);
    for value in values {
        writer.put_u64(*value as u64);
    }
    zkie_core::artifact::digest_bytes(&writer.into_bytes())
}

/// The private values of one table: one random tail for each row, one salt for
/// each column and one pad for each answer of each row.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LmHeadKeyRandomness {
    pub tails: Vec<Vec<Fp>>,
    pub salts: Vec<Vec<u8>>,
    pub pads: Vec<Vec<Fp>>,
}

/// The committed table and its public reader.
pub struct LmHeadKey {
    pub shape: IndexShape,
    pub reader: IndexReader,
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

    /// Decodes the canonical statement bytes. Bounded and strict: unknown
    /// fields, truncation and trailing bytes are rejected.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        use zkie_engine::primitives::codec::CanonicalReader;
        use zkie_core::common::codec::DecodeLimits;
        let limits = DecodeLimits {
            max_bytes: 1 << 20,
            max_elements: 1 << 12,
        };
        let mut reader = CanonicalReader::new(bytes, limits);
        let version = reader.get_u64()?;
        let model_commitment = reader.get_digest()?;
        let table_root = reader.get_digest()?;
        let table_configuration = reader.get_digest()?;
        let log_n = u32::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?;
        let ell = reader.get_u64()?;
        let r = reader.get_u64()?;
        let v = u32::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?;
        let answers = u32::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?;
        let salt_bytes = u32::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?;
        let code = zkie_engine::pcs::request::code::build_code_plan(log_n, ell, r)?;
        let table_shape = zkie_engine::pcs::request::table::IndexShape {
            code,
            v,
            answers,
            salt_bytes: salt_bytes as usize,
            distances: zkie_engine::pcs::request::table::TableDistances {
                radius: reader.get_u64()?,
                later_radius: reader.get_u64()?,
                m_once: reader.get_u64()?,
                m_later: reader.get_u64()?,
            },
        };
        let statement = Self {
            version,
            model_commitment,
            table_root,
            table_configuration,
            table_shape,
            matrix_rows: reader.get_u64()?,
            k_lo: u32::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?,
            k_col: u32::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?,
            valid_min: reader.get_u64()? as i64,
            valid_max: reader.get_u64()? as i64,
            no_wf: reader.get_u64()? != 0,
            trusted_i32_mapping: reader.get_u64()? != 0,
        };
        reader.finish()?;
        Ok(statement)
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
    pub base: BaseStarBinding,
    pub switches: Vec<SwitchTarget>,
    pub switch_positions: Vec<u64>,
    /// The trusted-derived public view of every switch plan, carried to the
    /// verifier setup.
    pub switch_public_plans: Vec<SwitchPlanPublic>,
    pub column_work_bits: u8,
    pub switch_work_bits: u8,
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

/// The trusted-derived public fields of one switch's runtime tree plan. The
/// run seed is not part of this view: it stays a committed private value
/// carried by the handoff. Every other field is derived from the compiled
/// plan by the same rules the prover's plan follows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeTreePlanPublic {
    pub tree_number: u64,
    pub leaf_class: u32,
    pub log_n: u32,
    pub log_n_zk: u32,
    pub parts: Vec<RuntimeTreePart>,
    pub salt_bytes: usize,
    pub configuration: Digest,
}

impl RuntimeTreePlanPublic {
    fn from_plan(plan: &RuntimeTreePlan) -> Self {
        Self {
            tree_number: plan.tree_number,
            leaf_class: plan.leaf_class,
            log_n: plan.log_n,
            log_n_zk: plan.log_n_zk,
            parts: plan.parts.clone(),
            salt_bytes: plan.salt_bytes,
            configuration: plan.configuration,
        }
    }
}

/// The trusted-derived public fields of a fresh mask tree plan. The run seed
/// is not part of this view: it stays a committed private value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskTreePlanPublic {
    pub tree_number: u64,
    pub leaf_class: u32,
    pub log_n: u32,
    pub mask_rows: u64,
    pub salt_bytes: usize,
    pub configuration: Digest,
}

impl MaskTreePlanPublic {
    fn from_plan(plan: &MaskTreePlan) -> Self {
        Self {
            tree_number: plan.tree_number,
            leaf_class: plan.leaf_class,
            log_n: plan.log_n,
            mask_rows: plan.mask_rows,
            salt_bytes: plan.salt_bytes,
            configuration: plan.configuration,
        }
    }

    fn to_plan(self, run: Digest) -> MaskTreePlan {
        MaskTreePlan {
            tree_number: self.tree_number,
            leaf_class: self.leaf_class,
            log_n: self.log_n,
            mask_rows: self.mask_rows,
            salt_bytes: self.salt_bytes,
            run,
            configuration: self.configuration,
        }
    }
}

/// The trusted-derived public fields of one stage FOLD plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageFoldPlanPublic {
    pub shape: FoldShape,
    pub layout: FoldGroupLayout,
    pub mask_tree: MaskTreePlanPublic,
    pub group: u64,
    pub member: u64,
    pub fresh: u64,
}

impl StageFoldPlanPublic {
    fn from_plan(plan: &LimitedStageFoldPlan) -> Self {
        Self {
            shape: plan.shape,
            layout: plan.layout.clone(),
            mask_tree: MaskTreePlanPublic::from_plan(&plan.mask_tree),
            group: plan.group,
            member: plan.member,
            fresh: plan.fresh,
        }
    }
}

/// The trusted-derived public fields of one switch plan: its position count,
/// its runtime tree plan and its following stage FOLD plan, if any. The run
/// seeds and the column salts are absent because they stay committed private
/// values rather than values the verifier rebuilds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwitchPlanPublic {
    pub positions: u64,
    pub tree: RuntimeTreePlanPublic,
    pub stage_fold: Option<StageFoldPlanPublic>,
}

impl SwitchPlanPublic {
    fn from_plan(plan: &LimitedSwitchPlan) -> Self {
        Self {
            positions: plan.positions,
            tree: RuntimeTreePlanPublic::from_plan(&plan.plan),
            stage_fold: plan.stage_fold.as_ref().map(StageFoldPlanPublic::from_plan),
        }
    }
}

/// A nonzero constructor witness for a runtime tree's run seed. The expected
/// public view never compares the run, so this value is neither committed nor
/// sent anywhere.
const PUBLIC_PLAN_RUN_PLACEHOLDER: Digest = Digest([1; 32]);

/// Derives one stage FOLD's trusted public fields from the compiled plan and
/// its switch target. The prover's stage plan and the verifier's expected plan
/// both use this function; only the fresh mask tree's run seed is supplied
/// separately as a private value.
fn stage_fold_public_plan(
    compiled: &CompiledPlan,
    target: &SwitchTarget,
    stage_index: u64,
    configuration: Digest,
) -> Result<StageFoldPlanPublic> {
    let shape = FoldShape {
        code: target.code,
        mask_code: target.mask_code,
        k: target.main.bits,
        h: target.rest.bits,
        mask_leaf_class: compiled.once_foldeq.shared_split_leaf_class,
    };
    if fold_mask_shape_fault(&shape).is_some() {
        return Err(Error::InvalidConfig);
    }
    let layout = FoldGroupLayout {
        group: LM_HEAD_FOLD_GROUP,
        k: shape.k,
        h: shape.h,
        members: vec![FoldMemberLayout {
            oracle: LM_HEAD_FOLD_GROUP,
            main_rows: target.main_rows,
            mask_rows: target.mask_rows,
            kept: false,
        }],
    };
    if layout.fault(&shape).is_some() {
        return Err(Error::InvalidConfig);
    }
    let mask_tree = MaskTreePlanPublic {
        tree_number: 0x500u64
            .checked_add(stage_index)
            .ok_or(Error::ResourceLimit)?,
        leaf_class: compiled.once_foldeq.shared_split_leaf_class,
        log_n: target.mask_code.log_n,
        mask_rows: target.mask_rows,
        salt_bytes: compiled.once_foldeq.base_salt_bytes,
        configuration,
    };
    if mask_tree
        .to_plan(PUBLIC_PLAN_RUN_PLACEHOLDER)
        .fault()
        .is_some()
    {
        return Err(Error::InvalidConfig);
    }
    Ok(StageFoldPlanPublic {
        shape,
        layout,
        mask_tree,
        group: LM_HEAD_FOLD_GROUP,
        member: LM_HEAD_FOLD_GROUP,
        fresh: LM_HEAD_FRESH_MASK,
    })
}

/// Derives every switch's trusted public plan view from the compiled one-time
/// stages and the switch targets built from them. The run seeds and the column
/// salts stay private commitments carried by the handoff.
fn expected_switch_public_plans(
    compiled: &CompiledPlan,
    targets: &[SwitchTarget],
    configuration: Digest,
) -> Result<Vec<SwitchPlanPublic>> {
    let stages = &compiled.once.folded.stages;
    if stages.len() != targets.len() {
        return Err(Error::InvalidConfig);
    }
    let mut plans = Vec::with_capacity(targets.len());
    for (index, target) in targets.iter().enumerate() {
        let stage_index = u64::try_from(index).map_err(|_| Error::ResourceLimit)?;
        let tree = switch_tree_plan_multi(
            stage_index + 1,
            target,
            compiled.once_foldeq.base_salt_bytes,
            PUBLIC_PLAN_RUN_PLACEHOLDER,
            configuration,
        )?;
        let stage_fold = if index + 1 < targets.len() {
            Some(stage_fold_public_plan(
                compiled,
                target,
                stage_index,
                configuration,
            )?)
        } else {
            None
        };
        plans.push(SwitchPlanPublic {
            positions: if index == 0 {
                compiled.once.folded.positions
            } else {
                stages[index - 1].positions
            },
            tree: RuntimeTreePlanPublic::from_plan(&tree),
            stage_fold,
        });
    }
    Ok(plans)
}

/// Builds the real shared-only FOLDEQ plan for the committed key table. The
/// fold shape comes from the compiled one-time lines, the fresh mask root from
/// the provider's pre-committed mask message, and every runtime tree's run and
/// configuration from the session chain and the table. The plan is rejected
/// unless the binding parameters digest is exactly the reader's configuration.
#[allow(clippy::too_many_arguments)]
struct LimitedPlanParts {
    parameters: BindingParameters,
    configuration: Digest,
    random_bytes: Digest,
    fold: FoldShape,
    mask_tree: MaskTreePlan,
    verified: Arc<VerifiedStatement>,
    gu: GuShape,
    omega: OmegaShapePlan,
    base: BaseStarBinding,
    switches: Vec<SwitchTarget>,
    switch_positions: Vec<u64>,
    /// The trusted-derived public view of every switch plan.
    switch_public_plans: Vec<SwitchPlanPublic>,
    column_work_bits: u8,
    switch_work_bits: u8,
    reader: IndexReader,
    statement_digest: Digest,
    leaf_class: u32,
    lin: LinShapeBinding,
}

/// The public verifier setup: the same plan, binder, chain and targets the
/// prover uses, without the private fresh-mask provider.
pub struct LimitedVerifySetup {
    pub plan: FoldeqPlan,
    pub binder: FoldeqBinder,
    pub chain: Chain,
    pub first_query: Vec<u8>,
    pub base: BaseStarBinding,
    pub switches: Vec<SwitchTarget>,
    /// The old group's real position count at each switch, from the trusted
    /// compiled plan. The verifier rejects a public switch plan that differs.
    pub switch_positions: Vec<u64>,
    /// The trusted-derived public fields of every switch plan. A handoff plan
    /// whose public fields differ is rejected. Trusted-derived: the position
    /// count; the runtime tree's tree_number, leaf_class, log_n, log_n_zk,
    /// parts, salt_bytes and configuration; the stage FOLD's shape, layout,
    /// group, member and fresh numbers; and its mask tree's tree_number,
    /// leaf_class, log_n, mask_rows, salt_bytes and configuration. Not
    /// rebuilt: the runtime tree and mask tree run seeds and all column
    /// salts, which stay committed private values carried by the handoff.
    pub switch_public_plans: Vec<SwitchPlanPublic>,
    /// Work bits of the column draw and of each switch draw, from the trusted
    /// plan's once and shared scopes.
    pub column_work_bits: u8,
    pub switch_work_bits: u8,
    pub fresh_plan: MaskTreePlan,
}

fn build_limited_plan_parts(
    request_plan: &RequestPlan,
    compiled: &CompiledPlan,
    statement: &LimitedKeyStatement,
    reader: &IndexReader,
    plan_identifier: Digest,
    case_id: &[u8],
    random_bytes: Digest,
) -> Result<LimitedPlanParts> {
    request_plan.validate()?;
    request_plan.validate_compiled(compiled)?;
    let column_work_bits = request_plan.work.once.positions;
    let switch_work_bits = request_plan.work.shared.positions;
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
    let answers = u32::try_from(request_plan.samples).map_err(|_| Error::InvalidConfig)?;
    let mut switches = Vec::with_capacity(compiled.once.folded.stages.len());
    let mut switch_positions = Vec::with_capacity(compiled.once.folded.stages.len());
    for (index, stage) in compiled.once.folded.stages.iter().enumerate() {
        switch_positions.push(if index == 0 {
            compiled.once.folded.positions
        } else {
            compiled.once.folded.stages[index - 1].positions
        });
        switches.push(SwitchTarget::from_stage(
            stage.main,
            stage.mask,
            stage.k,
            stage.h,
            answers,
        )?);
    }
    let switch_public_plans = expected_switch_public_plans(compiled, &switches, configuration)?;
    let last = compiled
        .once
        .folded
        .stages
        .last()
        .ok_or(Error::InvalidConfig)?;
    if last.k != 0 || last.h != 0 {
        return Err(Error::InvalidConfig);
    }
    let base_plan = RuntimeTreePlan {
        tree_number: compiled.once_foldeq.base_tree_number,
        leaf_class: compiled.once_foldeq.base_leaf_class,
        log_n: last.main.log_n,
        log_n_zk: last.mask.log_n,
        parts: vec![RuntimeTreePart {
            main_symbols: E_DEGREE as u64,
            mask_symbols: E_DEGREE as u64,
        }],
        salt_bytes: compiled.once_foldeq.base_salt_bytes,
        run,
        configuration,
    };
    if base_plan.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    let base = BaseStarBinding {
        plan: base_plan,
        positions: usize::try_from(last.positions).map_err(|_| Error::ResourceLimit)?,
    };

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
    let lin = LinShapeBinding {
        matrix_rows: 1u64 << request_plan.lin.matrix_rows_log,
        k_lo: request_plan.lin.k_lo,
        k_col: request_plan.lin.k_col,
    };
    Ok(LimitedPlanParts {
        parameters,
        configuration,
        random_bytes,
        fold,
        mask_tree,
        verified,
        gu,
        omega,
        base,
        switches,
        switch_positions,
        switch_public_plans,
        column_work_bits,
        switch_work_bits,
        reader: reader.clone(),
        statement_digest: statement.digest(),
        leaf_class: compiled.once_foldeq.fold_leaf_class,
        lin,
    })
}

fn finish_limited_setup(
    parts: LimitedPlanParts,
    case_id: &[u8],
    fresh_root: Digest,
) -> Result<LimitedVerifySetup> {
    let mut oracles = RuntimeOracles::default();
    oracles.receive_mask(
        &parts.verified,
        &parts.mask_tree,
        LM_HEAD_FRESH_MASK,
        &fold_mask_notice(&parts.fold)?,
        fresh_root,
    )?;
    let layout = FoldGroupLayout {
        group: LM_HEAD_FOLD_GROUP,
        k: parts.fold.k,
        h: parts.fold.h,
        members: vec![
            FoldMemberLayout {
                oracle: LM_HEAD_GU_ORACLE,
                main_rows: parts.gu.u.main_rows,
                mask_rows: parts.gu.u.mask_rows,
                kept: false,
            },
            FoldMemberLayout {
                oracle: LM_HEAD_OMEGA_ORACLE,
                main_rows: parts.omega.u.main_rows,
                mask_rows: parts.omega.u.mask_rows,
                kept: false,
            },
        ],
    };
    if layout.fault(&parts.fold).is_some() {
        return Err(Error::InvalidConfig);
    }

    let table_identity = reader_table_identity(&parts.reader)?;
    let binding_statement = BindingStatement::new(
        parts.parameters.version_word,
        parts.parameters.protocol_identity,
        case_id.to_vec(),
        parts.statement_digest,
        table_identity,
    )?;
    let binding = FoldeqBinding::new(&parts.parameters, binding_statement, &parts.reader)?;

    let plan = FoldeqPlan {
        reader: parts.reader.clone(),
        binding,
        edge: FoldeqEdgeBinding::NotAnEdge,
        fold: parts.fold,
        statement: parts.verified.clone(),
        oracles: Arc::new(Mutex::new(oracles)),
        mask_tree_number: parts.mask_tree.tree_number,
        group_id: LM_HEAD_FOLD_GROUP,
        leaf_class: parts.leaf_class,
        seed_budget: parts.reader.shape.answers as usize,
        ledger: Arc::new(Mutex::new(ReadLedger::new(LM_HEAD_COLUMN_DRAWS as u64))),
        column_draws: LM_HEAD_COLUMN_DRAWS,
        mask_query_count: 0,
        gamma_path: GammaPath::Lin { shape: parts.lin },
        gu: Some(parts.gu),
        fold_layout: Some(layout),
        omega: Some(parts.omega),
    };
    if plan.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    let binder = FoldeqBinder::new_with_random_bytes(
        SHARED_CHAIN_LABEL,
        parts.configuration,
        &plan,
        Some(parts.random_bytes),
    )?;
    let first_query = binder.first_query().to_vec();
    let chain = binder.chain()?;
    Ok(LimitedVerifySetup {
        plan,
        binder,
        chain,
        first_query,
        base: parts.base,
        switches: parts.switches,
        switch_positions: parts.switch_positions,
        switch_public_plans: parts.switch_public_plans,
        column_work_bits: parts.column_work_bits,
        switch_work_bits: parts.switch_work_bits,
        fresh_plan: parts.mask_tree,
    })
}

impl LimitedSharedSession {
    /// The public verifier view of this session. The private mask provider is
    /// not carried into it.
    pub fn verify_setup(&self) -> LimitedVerifySetup {
        LimitedVerifySetup {
            plan: self.plan.clone(),
            binder: self.binder.clone(),
            chain: self.chain.clone(),
            first_query: self.first_query.clone(),
            base: self.base.clone(),
            switches: self.switches.clone(),
            switch_positions: self.switch_positions.clone(),
            switch_public_plans: self.switch_public_plans.clone(),
            column_work_bits: self.column_work_bits,
            switch_work_bits: self.switch_work_bits,
            fresh_plan: self.mask.tree().plan.clone(),
        }
    }
}

/// The public plan values the commit stage persists beside the statement and
/// reader. They are recomputed by the verify process and must agree.
#[derive(Clone, Debug)]
pub struct LimitedPublicPlan {
    pub base: BaseStarBinding,
    pub switches: Vec<SwitchTarget>,
    pub fresh_plan: MaskTreePlan,
    pub fold: FoldShape,
    pub gu: GuShape,
    pub omega: OmegaShapePlan,
    pub configuration: Digest,
}

pub fn build_limited_public_plan(
    request_plan: &RequestPlan,
    compiled: &CompiledPlan,
    statement: &LimitedKeyStatement,
    reader: &IndexReader,
    plan_identifier: Digest,
    case_id: &[u8],
    random_bytes: Digest,
) -> Result<LimitedPublicPlan> {
    let parts = build_limited_plan_parts(
        request_plan,
        compiled,
        statement,
        reader,
        plan_identifier,
        case_id,
        random_bytes,
    )?;
    Ok(LimitedPublicPlan {
        base: parts.base.clone(),
        switches: parts.switches.clone(),
        fresh_plan: parts.mask_tree,
        fold: parts.fold,
        gu: parts.gu.clone(),
        omega: parts.omega,
        configuration: parts.configuration,
    })
}

pub fn build_limited_verify_setup(
    request_plan: &RequestPlan,
    compiled: &CompiledPlan,
    statement: &LimitedKeyStatement,
    reader: &IndexReader,
    plan_identifier: Digest,
    case_id: &[u8],
    random_bytes: Digest,
    fresh_root: Digest,
) -> Result<LimitedVerifySetup> {
    let parts = build_limited_plan_parts(
        request_plan,
        compiled,
        statement,
        reader,
        plan_identifier,
        case_id,
        random_bytes,
    )?;
    finish_limited_setup(parts, case_id, fresh_root)
}

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
    let parts = build_limited_plan_parts(
        request_plan,
        compiled,
        statement,
        reader,
        plan_identifier,
        case_id,
        random_bytes,
    )?;
    let mask_oracle = MaskOracle {
        id: LM_HEAD_FRESH_MASK,
        claimed_group: LM_HEAD_FOLD_GROUP,
        main_rows: 0,
        mask_rows: parts.mask_tree.mask_rows,
        mask_code: parts.fold.mask_code,
        answers: 0,
        sampled: false,
    };
    let mask = make_lm_head_mask(
        &parts.fold,
        &mask_oracle,
        parts.verified.groups(),
        parts.mask_tree,
        mask_randomness,
    )?;
    let root = mask.root();
    let setup = finish_limited_setup(parts, case_id, root)?;
    Ok(LimitedSharedSession {
        plan: setup.plan,
        binder: setup.binder,
        chain: setup.chain,
        first_query: setup.first_query,
        mask,
        base: setup.base,
        switches: setup.switches,
        switch_positions: setup.switch_positions,
        switch_public_plans: setup.switch_public_plans,
        column_work_bits: setup.column_work_bits,
        switch_work_bits: setup.switch_work_bits,
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
#[derive(Clone, Debug, Serialize, Deserialize)]
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

    finish_lm_head_gu(shape, folded, folded_tail, randomness)
}

/// The shared G_u assembly: place the folded row, add the private sample
/// pads, encode each main row with its tail and commit the runtime tree.
fn finish_lm_head_gu(
    shape: &GuShape,
    folded: Vec<Fp3>,
    folded_tail: Vec<Fp3>,
    randomness: &LmHeadGuRandomness,
) -> Result<LmHeadGuProvider> {
    let main_rows = usize::try_from(shape.u.main_rows).map_err(|_| Error::ResourceLimit)?;
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

/// Folds the committed table through the holder and assembles G_u. The
/// holder reads the rows in bounded windows; no complete weight message is
/// kept by this path.
pub fn fold_lm_head_gu_holder<S: TableRowSource>(
    holder: &mut StreamingHolder<S>,
    shape: &GuShape,
    gamma: &[Fp3],
    randomness: &LmHeadGuRandomness,
) -> Result<LmHeadGuProvider> {
    if holder.reader().shape != shape.index || holder.reader().fault().is_some() {
        return Err(Error::StatementMismatch);
    }
    if shape.fault().is_some() || gamma.len() != shape.index.v as usize {
        return Err(Error::InvalidConfig);
    }
    let (folded, folded_tail) = holder.fold(gamma)?;
    finish_lm_head_gu(shape, folded, folded_tail, randomness)
}

/// The private values of one Omega commitment: the oracle code tails, one
/// salt per runtime-tree column and one sample pad.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
/// The public messages of one switch stage. Prover witness (the new rows,
/// their tails and the private encoding) stays out of this record; openings
/// are verifier inputs.
#[derive(Clone, Debug)]
pub struct LimitedSwitchRecord {
    pub identity: Digest,
    pub declared_root: Digest,
    pub sample_answer: Fp3,
    pub nonce: u64,
    pub member_openings: Vec<RuntimeTreeOpening>,
    pub fresh_opening: RuntimeTreeOpening,
    pub old_mask_root: Option<Digest>,
    pub old_mask_opening: Option<RuntimeTreeOpening>,
    pub stage_fold: Option<LimitedStageFoldRecord>,
}

/// The public messages of one FOLD stage: the fresh mask start message and
/// the round polynomials.
#[derive(Clone, Debug)]
pub struct LimitedStageFoldRecord {
    pub start: FoldStartMessage,
    pub rounds: Vec<(Fp3, Fp3)>,
}

/// The public prover messages of one chain. The statement and the trusted
/// plan are supplied by the verifier; this record holds only what the prover
/// sent and the final BASE artifact.
#[derive(Clone, Debug, Default)]
pub struct LimitedChainRecord {
    pub column_nonce: u64,
    pub gu_root: Option<Digest>,
    pub gu_answers: Vec<Fp3>,
    pub omega_root: Option<Digest>,
    pub omega_answers: Vec<Fp3>,
    pub column_openings: Vec<ColumnOpening>,
    pub fold_start: Option<FoldStartMessage>,
    pub fold_rounds: Vec<(Fp3, Fp3)>,
    pub switches: Vec<LimitedSwitchRecord>,
    pub base_proof: Option<Vec<u8>>,
    pub sealed: Option<Digest>,
}

/// One versioned whole-chain artifact. It carries the statement and every
/// public prover message; no folded message, switched row, tail or mask
/// coefficient is included.
pub const LIMITED_CHAIN_ARTIFACT_VERSION: u64 = 2;

#[derive(Clone, Debug)]
pub struct LimitedStageFoldArtifact {
    pub start: FoldStartMessage,
    pub rounds: Vec<(Fp3, Fp3)>,
}

#[derive(Clone, Debug)]
pub struct LimitedSwitchArtifact {
    pub identity: Digest,
    pub declared_root: Digest,
    pub sample_answer: Fp3,
    pub nonce: u64,
    pub member_openings: Vec<RuntimeTreeOpening>,
    pub fresh_opening: RuntimeTreeOpening,
    pub old_mask_root: Option<Digest>,
    pub old_mask_opening: Option<RuntimeTreeOpening>,
    pub stage_fold: Option<LimitedStageFoldArtifact>,
}

#[derive(Clone, Debug)]
pub struct LimitedChainArtifact {
    pub version: u64,
    pub protocol_identity: Digest,
    pub statement: LimitedKeyStatement,
    pub gu_root: Digest,
    pub gu_answers: Vec<Fp3>,
    pub omega_root: Option<Digest>,
    pub omega_answers: Vec<Fp3>,
    pub column_openings: Vec<ColumnOpening>,
    pub column_nonce: u64,
    pub fold_start: FoldStartMessage,
    pub fold_rounds: Vec<(Fp3, Fp3)>,
    pub switches: Vec<LimitedSwitchArtifact>,
    pub base_proof: Vec<u8>,
    pub sealed: Digest,
}

/// Keeps one physical opening per unique position. Repeats of a draw carry
/// the same opening; a repeated position with different values is rejected.
/// The list is sorted by position so the decoder can check the order.
fn unique_column_openings(openings: &[ColumnOpening]) -> Result<Vec<ColumnOpening>> {
    if openings.is_empty() {
        return Err(Error::IncompleteArtifact);
    }
    let mut sorted = openings.to_vec();
    sorted.sort_by_key(|opening| opening.position);
    let mut unique: Vec<ColumnOpening> = Vec::new();
    for opening in sorted {
        match unique.last() {
            Some(last) if last.position == opening.position => {
                if *last != opening {
                    return Err(Error::InvalidProof);
                }
            }
            _ => unique.push(opening),
        }
    }
    Ok(unique)
}

fn artifact_limits() -> zkie_core::common::codec::DecodeLimits {
    zkie_core::common::codec::DecodeLimits {
        max_bytes: 1 << 30,
        max_elements: 1 << 22,
    }
}

fn put_digest_vec(
    writer: &mut zkie_engine::primitives::codec::CanonicalWriter,
    values: &[Digest],
) {
    writer.put_u64(values.len() as u64);
    for value in values {
        writer.put_digest(value);
    }
}

fn put_fp_vec(
    writer: &mut zkie_engine::primitives::codec::CanonicalWriter,
    values: &[Fp],
) {
    writer.put_u64(values.len() as u64);
    for value in values {
        writer.put_fp(*value);
    }
}

fn put_fp3_vec(
    writer: &mut zkie_engine::primitives::codec::CanonicalWriter,
    values: &[Fp3],
) {
    writer.put_u64(values.len() as u64);
    for value in values {
        writer.put_fp3(value);
    }
}

fn put_column_opening(
    writer: &mut zkie_engine::primitives::codec::CanonicalWriter,
    opening: &ColumnOpening,
) {
    writer.put_u64(opening.position);
    writer.put_bytes(&opening.salt);
    put_fp_vec(writer, &opening.symbols);
    put_digest_vec(writer, &opening.path);
}

fn put_runtime_opening(
    writer: &mut zkie_engine::primitives::codec::CanonicalWriter,
    opening: &RuntimeTreeOpening,
) {
    writer.put_u64(opening.positions.len() as u64);
    for position in &opening.positions {
        writer.put_u64(*position);
    }
    writer.put_u64(opening.columns.len() as u64);
    for column in &opening.columns {
        writer.put_u64(column.position);
        writer.put_bytes(&column.salt);
        put_fp_vec(writer, &column.symbols);
    }
    put_digest_vec(writer, &opening.given);
}

fn put_mask_notice(
    writer: &mut zkie_engine::primitives::codec::CanonicalWriter,
    notice: &zkie_engine::pcs::request::fold::MaskNotice,
) {
    writer.put_u64(notice.mask_rows);
    writer.put_u64(notice.mask_code.log_n as u64);
    writer.put_u64(notice.mask_code.ell);
    writer.put_u64(notice.mask_code.r);
    writer.put_u64(notice.answers as u64);
    writer.put_u64(notice.leaf_class as u64);
}

fn put_fold_start(
    writer: &mut zkie_engine::primitives::codec::CanonicalWriter,
    start: &FoldStartMessage,
) {
    writer.put_u64(start.id);
    writer.put_u64(start.claimed_group);
    writer.put_u64(start.tree_number);
    put_mask_notice(writer, &start.notice);
    writer.put_digest(&start.root);
    writer.put_fp3(&start.sum);
}

fn put_stage_fold(
    writer: &mut zkie_engine::primitives::codec::CanonicalWriter,
    stage: &LimitedStageFoldArtifact,
) {
    put_fold_start(writer, &stage.start);
    writer.put_u64(stage.rounds.len() as u64);
    for (first, second) in &stage.rounds {
        writer.put_fp3(first);
        writer.put_fp3(second);
    }
}

fn put_switch(
    writer: &mut zkie_engine::primitives::codec::CanonicalWriter,
    switch: &LimitedSwitchArtifact,
) {
    writer.put_digest(&switch.identity);
    writer.put_digest(&switch.declared_root);
    writer.put_fp3(&switch.sample_answer);
    writer.put_u64(switch.nonce);
    writer.put_u64(switch.member_openings.len() as u64);
    for opening in &switch.member_openings {
        put_runtime_opening(writer, opening);
    }
    put_runtime_opening(writer, &switch.fresh_opening);
    match switch.old_mask_root {
        Some(root) => {
            writer.put_u64(1);
            writer.put_digest(&root);
        }
        None => writer.put_u64(0),
    }
    match &switch.old_mask_opening {
        Some(opening) => {
            writer.put_u64(1);
            put_runtime_opening(writer, opening);
        }
        None => writer.put_u64(0),
    }
    match &switch.stage_fold {
        Some(stage) => {
            writer.put_u64(1);
            put_stage_fold(writer, stage);
        }
        None => writer.put_u64(0),
    }
}

fn read_count(
    reader: &mut zkie_engine::primitives::codec::CanonicalReader<'_>,
    limits: zkie_core::common::codec::DecodeLimits,
) -> Result<usize> {
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?;
    limits.check_elements(count)?;
    Ok(count)
}

fn read_digest_vec(
    reader: &mut zkie_engine::primitives::codec::CanonicalReader<'_>,
    limits: zkie_core::common::codec::DecodeLimits,
) -> Result<Vec<Digest>> {
    let count = read_count(reader, limits)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::ResourceLimit)?;
    for _ in 0..count {
        values.push(reader.get_digest()?);
    }
    Ok(values)
}

fn read_fp_vec(
    reader: &mut zkie_engine::primitives::codec::CanonicalReader<'_>,
    limits: zkie_core::common::codec::DecodeLimits,
) -> Result<Vec<Fp>> {
    let count = read_count(reader, limits)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::ResourceLimit)?;
    for _ in 0..count {
        values.push(reader.get_fp()?);
    }
    Ok(values)
}

fn read_fp3_vec(
    reader: &mut zkie_engine::primitives::codec::CanonicalReader<'_>,
    limits: zkie_core::common::codec::DecodeLimits,
) -> Result<Vec<Fp3>> {
    let count = read_count(reader, limits)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::ResourceLimit)?;
    for _ in 0..count {
        values.push(reader.get_fp3()?);
    }
    Ok(values)
}

fn read_column_opening(
    reader: &mut zkie_engine::primitives::codec::CanonicalReader<'_>,
    limits: zkie_core::common::codec::DecodeLimits,
) -> Result<ColumnOpening> {
    let position = reader.get_u64()?;
    let salt = reader.get_bytes()?.to_vec();
    let symbols = read_fp_vec(reader, limits)?;
    let path = read_digest_vec(reader, limits)?;
    Ok(ColumnOpening {
        position,
        salt,
        symbols,
        path,
    })
}

fn read_runtime_opening(
    reader: &mut zkie_engine::primitives::codec::CanonicalReader<'_>,
    limits: zkie_core::common::codec::DecodeLimits,
) -> Result<RuntimeTreeOpening> {
    let positions_count = read_count(reader, limits)?;
    let mut positions = Vec::new();
    positions
        .try_reserve_exact(positions_count)
        .map_err(|_| Error::ResourceLimit)?;
    for _ in 0..positions_count {
        positions.push(reader.get_u64()?);
    }
    let columns_count = read_count(reader, limits)?;
    let mut columns = Vec::new();
    columns
        .try_reserve_exact(columns_count)
        .map_err(|_| Error::ResourceLimit)?;
    for _ in 0..columns_count {
        let position = reader.get_u64()?;
        let salt = reader.get_bytes()?.to_vec();
        let symbols = read_fp_vec(reader, limits)?;
        columns.push(zkie_engine::pcs::request::runtime_tree::RuntimeLeafColumn {
            position,
            salt,
            symbols,
        });
    }
    let given = read_digest_vec(reader, limits)?;
    Ok(RuntimeTreeOpening {
        positions,
        columns,
        given,
    })
}

fn read_mask_notice(
    reader: &mut zkie_engine::primitives::codec::CanonicalReader<'_>,
) -> Result<zkie_engine::pcs::request::fold::MaskNotice> {
    let mask_rows = reader.get_u64()?;
    let log_n = u32::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?;
    let ell = reader.get_u64()?;
    let r = reader.get_u64()?;
    let code = zkie_engine::pcs::request::code::build_code_plan(log_n, ell, r)?;
    let answers = u32::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?;
    let leaf_class =
        u32::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?;
    Ok(zkie_engine::pcs::request::fold::MaskNotice {
        mask_rows,
        mask_code: code,
        answers,
        leaf_class,
    })
}

fn read_fold_start(
    reader: &mut zkie_engine::primitives::codec::CanonicalReader<'_>,
) -> Result<FoldStartMessage> {
    Ok(FoldStartMessage {
        id: reader.get_u64()?,
        claimed_group: reader.get_u64()?,
        tree_number: reader.get_u64()?,
        notice: read_mask_notice(reader)?,
        root: reader.get_digest()?,
        sum: reader.get_fp3()?,
    })
}

fn read_stage_fold(
    reader: &mut zkie_engine::primitives::codec::CanonicalReader<'_>,
    limits: zkie_core::common::codec::DecodeLimits,
) -> Result<LimitedStageFoldArtifact> {
    let start = read_fold_start(reader)?;
    let count = read_count(reader, limits)?;
    let mut rounds = Vec::new();
    rounds
        .try_reserve_exact(count)
        .map_err(|_| Error::ResourceLimit)?;
    for _ in 0..count {
        rounds.push((reader.get_fp3()?, reader.get_fp3()?));
    }
    Ok(LimitedStageFoldArtifact { start, rounds })
}

fn read_switch(
    reader: &mut zkie_engine::primitives::codec::CanonicalReader<'_>,
    limits: zkie_core::common::codec::DecodeLimits,
) -> Result<LimitedSwitchArtifact> {
    let identity = reader.get_digest()?;
    let declared_root = reader.get_digest()?;
    let sample_answer = reader.get_fp3()?;
    let nonce = reader.get_u64()?;
    let member_count = read_count(reader, limits)?;
    let mut member_openings = Vec::new();
    member_openings
        .try_reserve_exact(member_count)
        .map_err(|_| Error::ResourceLimit)?;
    for _ in 0..member_count {
        member_openings.push(read_runtime_opening(reader, limits)?);
    }
    let fresh_opening = read_runtime_opening(reader, limits)?;
    let old_mask_root = match reader.get_u64()? {
        0 => None,
        1 => Some(reader.get_digest()?),
        _ => return Err(Error::InvalidEncoding),
    };
    let old_mask_opening = match reader.get_u64()? {
        0 => None,
        1 => Some(read_runtime_opening(reader, limits)?),
        _ => return Err(Error::InvalidEncoding),
    };
    let stage_fold = match reader.get_u64()? {
        0 => None,
        1 => Some(read_stage_fold(reader, limits)?),
        _ => return Err(Error::InvalidEncoding),
    };
    Ok(LimitedSwitchArtifact {
        identity,
        declared_root,
        sample_answer,
        nonce,
        member_openings,
        fresh_opening,
        old_mask_root,
        old_mask_opening,
        stage_fold,
    })
}

impl LimitedChainArtifact {
    pub fn from_record(
        statement: &LimitedKeyStatement,
        record: &LimitedChainRecord,
    ) -> Result<Self> {
        statement.validate()?;
        let fold_start = record
            .fold_start
            .clone()
            .ok_or(Error::IncompleteArtifact)?;
        let base_proof = record
            .base_proof
            .clone()
            .ok_or(Error::IncompleteArtifact)?;
        let sealed = record.sealed.ok_or(Error::IncompleteArtifact)?;
        let omega_root = record.omega_root;
        if omega_root.is_some() != !record.omega_answers.is_empty() {
            return Err(Error::IncompleteArtifact);
        }
        let mut switches = Vec::with_capacity(record.switches.len());
        for switch in &record.switches {
            switches.push(LimitedSwitchArtifact {
                identity: switch.identity,
                declared_root: switch.declared_root,
                sample_answer: switch.sample_answer,
                nonce: switch.nonce,
                member_openings: switch.member_openings.clone(),
                fresh_opening: switch.fresh_opening.clone(),
                old_mask_root: switch.old_mask_root,
                old_mask_opening: switch.old_mask_opening.clone(),
                stage_fold: switch.stage_fold.as_ref().map(|stage| LimitedStageFoldArtifact {
                    start: stage.start.clone(),
                    rounds: stage.rounds.clone(),
                }),
            });
        }
        Ok(Self {
            version: LIMITED_CHAIN_ARTIFACT_VERSION,
            protocol_identity: SESSION_PROTOCOL_IDENTITY,
            statement: statement.clone(),
            gu_root: record.gu_root.ok_or(Error::IncompleteArtifact)?,
            gu_answers: record.gu_answers.clone(),
            omega_root,
            omega_answers: record.omega_answers.clone(),
            column_openings: unique_column_openings(&record.column_openings)?,
            column_nonce: record.column_nonce,
            fold_start,
            fold_rounds: record.fold_rounds.clone(),
            switches,
            base_proof,
            sealed,
        })
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut writer = zkie_engine::primitives::codec::CanonicalWriter::new();
        writer.put_u64(self.version);
        writer.put_digest(&self.protocol_identity);
        let statement = self.statement.encode();
        writer.put_bytes(&statement);
        writer.put_digest(&self.gu_root);
        put_fp3_vec(&mut writer, &self.gu_answers);
        match self.omega_root {
            Some(root) => {
                writer.put_u64(1);
                writer.put_digest(&root);
            }
            None => writer.put_u64(0),
        }
        put_fp3_vec(&mut writer, &self.omega_answers);
        writer.put_u64(self.column_openings.len() as u64);
        for opening in &self.column_openings {
            put_column_opening(&mut writer, opening);
        }
        writer.put_u64(self.column_nonce);
        put_fold_start(&mut writer, &self.fold_start);
        writer.put_u64(self.fold_rounds.len() as u64);
        for (first, second) in &self.fold_rounds {
            writer.put_fp3(first);
            writer.put_fp3(second);
        }
        writer.put_u64(self.switches.len() as u64);
        for switch in &self.switches {
            put_switch(&mut writer, switch);
        }
        writer.put_bytes(&self.base_proof);
        writer.put_digest(&self.sealed);
        writer.into_bytes()
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        use zkie_engine::primitives::codec::CanonicalReader;
        let limits = artifact_limits();
        let mut reader = CanonicalReader::new(bytes, limits);
        let version = reader.get_u64()?;
        if version != LIMITED_CHAIN_ARTIFACT_VERSION {
            return Err(Error::InvalidConfig);
        }
        let protocol_identity = reader.get_digest()?;
        if protocol_identity != SESSION_PROTOCOL_IDENTITY {
            return Err(Error::InvalidConfig);
        }
        let statement_bytes = reader.get_bytes()?;
        let statement = LimitedKeyStatement::decode(statement_bytes)?;
        statement.validate()?;
        let gu_root = reader.get_digest()?;
        let gu_answers = read_fp3_vec(&mut reader, limits)?;
        let omega_root = match reader.get_u64()? {
            0 => None,
            1 => Some(reader.get_digest()?),
            _ => return Err(Error::InvalidEncoding),
        };
        let omega_answers = read_fp3_vec(&mut reader, limits)?;
        if omega_root.is_some() != !omega_answers.is_empty() {
            return Err(Error::IncompleteArtifact);
        }
        let opening_count = read_count(&mut reader, limits)?;
        let mut column_openings = Vec::new();
        column_openings
            .try_reserve_exact(opening_count)
            .map_err(|_| Error::ResourceLimit)?;
        for _ in 0..opening_count {
            column_openings.push(read_column_opening(&mut reader, limits)?);
        }
        if column_openings
            .windows(2)
            .any(|pair| pair[0].position >= pair[1].position)
        {
            return Err(Error::InvalidEncoding);
        }
        let column_nonce = reader.get_u64()?;
        let fold_start = read_fold_start(&mut reader)?;
        let round_count = read_count(&mut reader, limits)?;
        let mut fold_rounds = Vec::new();
        fold_rounds
            .try_reserve_exact(round_count)
            .map_err(|_| Error::ResourceLimit)?;
        for _ in 0..round_count {
            fold_rounds.push((reader.get_fp3()?, reader.get_fp3()?));
        }
        let switch_count = read_count(&mut reader, limits)?;
        let mut switches = Vec::new();
        switches
            .try_reserve_exact(switch_count)
            .map_err(|_| Error::ResourceLimit)?;
        for _ in 0..switch_count {
            switches.push(read_switch(&mut reader, limits)?);
        }
        let base_proof = reader.get_bytes()?.to_vec();
        let sealed = reader.get_digest()?;
        reader.finish()?;
        Ok(Self {
            version,
            protocol_identity,
            statement,
            gu_root,
            gu_answers,
            omega_root,
            omega_answers,
            column_openings,
            column_nonce,
            fold_start,
            fold_rounds,
            switches,
            base_proof,
            sealed,
        })
    }
}

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
    pub sealed: Option<Digest>,
    pub base_proof: Option<Vec<u8>>,
    pub pending_rows: usize,
    pub transcript: Digest,
    pub record: LimitedChainRecord,
}

/// Runs the shared-only chain through the record and column joins. The
/// provider folds the committed table after the gamma challenge, sends G_u,
/// answers the seeds, and opens the verifier's drawn table columns. The FOLD
/// group still needs its second member (Omega), so no FOLD round is entered
/// here.
/// The read-only weight asset as a bounded table row source. For one table
/// row it reads, for every matrix row, the factor entries that row owns at
/// that table row's column segment; the complete weight matrix is never read
/// into one buffer. The asset stays mapped read-only and every copy is charged
/// to the reader's window budget.
pub struct LmHeadAssetSource {
    reader: BoundedAssetReader,
    shape: IndexShape,
    matrix_rows: u64,
    factor: u64,
    row_width: u64,
    plain: u64,
    pads: Vec<Vec<Fp>>,
    tails: Vec<Vec<Fp>>,
    buffer: Vec<u8>,
}

impl LmHeadAssetSource {
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        context: &LmHeadContext,
        plan: &RequestPlan,
        configuration: Digest,
        randomness: &LmHeadKeyRandomness,
        cap_bytes: u64,
        window_bytes: u64,
        total_bytes: u64,
    ) -> Result<Self> {
        plan.validate()?;
        if configuration == Digest::ZERO || cap_bytes == 0 {
            return Err(Error::InvalidConfig);
        }
        let shape = plan.table_shape()?;
        let factor = 1u64
            .checked_shl(plan.lin.k_lo)
            .ok_or(Error::InvalidConfig)?;
        let matrix_rows = 1u64
            .checked_shl(plan.lin.matrix_rows_log)
            .ok_or(Error::InvalidConfig)?;
        let columns = 1u64
            .checked_shl(plan.lin.k_col)
            .ok_or(Error::InvalidConfig)?;
        if context.asset.rows != shape.rows()
            || context.asset.row_width < columns
            || context.asset.element_count != context.asset.rows.saturating_mul(context.asset.row_width)
            || matrix_rows != shape.rows()
            || factor
                .checked_mul(matrix_rows)
                .ok_or(Error::InvalidConfig)?
                != columns
        {
            return Err(Error::ShapeMismatch);
        }
        let ell = usize::try_from(shape.code.ell).map_err(|_| Error::ResourceLimit)?;
        let tail = usize::try_from(shape.code.r).map_err(|_| Error::ResourceLimit)?;
        let plain = shape.plain();
        if randomness.pads.len() != shape.rows() as usize
            || randomness.tails.len() != shape.rows() as usize
            || randomness
                .pads
                .iter()
                .any(|row| row.len() != E_DEGREE * shape.answers as usize)
            || randomness
                .tails
                .iter()
                .any(|row| row.len() != tail)
            || randomness.salts.len() != shape.code.n() as usize
        {
            return Err(Error::ShapeMismatch);
        }
        let bytes = context
            .asset
            .element_count
            .checked_mul(4)
            .ok_or(Error::ResourceLimit)?;
        if bytes > cap_bytes {
            return Err(Error::ResourceLimit);
        }
        let governor = BudgetedMemoryGovernor::new(cap_bytes);
        let permit = governor.acquire(&MemoryRequest {
            mapped_window_bytes: bytes,
            ..MemoryRequest::default()
        })?;
        let reader = BoundedAssetReader::open(
            context.asset_root.join(&context.asset.path),
            TensorSpec::new(DType::I32, vec![context.asset.element_count]),
            context.asset.start,
            permit,
            window_bytes,
            total_bytes,
        )?;
        let buffer = vec![0u8; usize::try_from(factor).map_err(|_| Error::ResourceLimit)? * 4];
        let _ = ell;
        Ok(Self {
            reader,
            shape,
            matrix_rows,
            factor,
            row_width: context.asset.row_width,
            plain,
            pads: randomness.pads.clone(),
            tails: randomness.tails.clone(),
            buffer,
        })
    }

    pub fn reader(&self) -> &BoundedAssetReader {
        &self.reader
    }

    pub fn shape(&self) -> IndexShape {
        self.shape
    }
}

impl TableRowSource for LmHeadAssetSource {
    fn rows(&self) -> u64 {
        self.shape.rows()
    }

    fn row(&mut self, row: u64, message: &mut [Fp], tail: &mut [Fp]) -> Result<()> {
        let rows = self.shape.rows();
        if row >= rows
            || message.len() as u64 != self.shape.code.ell
            || tail.len() as u64 != self.shape.code.r
        {
            return Err(Error::ShapeMismatch);
        }
        let factor = usize::try_from(self.factor).map_err(|_| Error::ResourceLimit)?;
        let table_row = row;
        for matrix_row in 0..self.matrix_rows {
            let start = matrix_row
                .checked_mul(self.row_width)
                .and_then(|value| {
                    table_row
                        .checked_mul(self.factor)
                        .and_then(|offset| value.checked_add(offset))
                })
                .ok_or(Error::ResourceLimit)?;
            self.reader
                .read_range(start, factor, &mut self.buffer)?;
            for low in 0..factor {
                let value = i32::from_le_bytes(
                    self.buffer[low * 4..low * 4 + 4]
                        .try_into()
                        .map_err(|_| Error::InvalidEncoding)?,
                ) as i64;
                let place = usize::try_from(
                    matrix_row
                        .checked_mul(self.factor)
                        .and_then(|base| base.checked_add(low as u64))
                        .ok_or(Error::ResourceLimit)?,
                )
                .map_err(|_| Error::ResourceLimit)?;
                *message.get_mut(place).ok_or(Error::ShapeMismatch)? =
                    fp_from_signed_word(value as u64);
            }
        }
        let pads = self.pads.get(row as usize).ok_or(Error::ShapeMismatch)?;
        for (offset, pad) in pads.iter().enumerate() {
            let at = usize::try_from(self.plain).map_err(|_| Error::ResourceLimit)? + offset;
            *message.get_mut(at).ok_or(Error::ShapeMismatch)? = *pad;
        }
        let tail_row = self.tails.get(row as usize).ok_or(Error::ShapeMismatch)?;
        tail.copy_from_slice(tail_row);
        zkie_engine::profile::add_table_bytes(
            self.matrix_rows
                .saturating_mul(self.factor)
                .saturating_mul(4),
        );
        Ok(())
    }
}

/// Commits the real weight asset through the bounded holder and returns the
/// ops-facing source. The table's root, record and openings come from the same
/// identity, configuration and private tails as the key path.
#[allow(clippy::too_many_arguments)]
pub fn build_lm_head_holder(
    context: &LmHeadContext,
    plan: &RequestPlan,
    configuration: Digest,
    key_randomness: &LmHeadKeyRandomness,
    store: &mut dyn EncodedRowStore,
    panel_columns: u64,
    max_bytes: u64,
    cap_bytes: u64,
    window_bytes: u64,
    total_bytes: u64,
) -> Result<StreamingHolder<LmHeadAssetSource>> {
    let shape = plan.table_shape()?;
    let _profile_open = zkie_engine::profile::Scope::new("table.open");
    let mut source = LmHeadAssetSource::open(
        context,
        plan,
        configuration,
        key_randomness,
        cap_bytes,
        window_bytes,
        total_bytes,
    )?;
    drop(_profile_open);
    let _profile_commit = zkie_engine::profile::Scope::new("table.commit");
    let table = zkie_engine::pcs::request::table::commit_table_stored(
        &mut source,
        store,
        shape,
        configuration,
        &key_randomness.salts,
        panel_columns,
        max_bytes,
    )?;
    drop(_profile_commit);
    StreamingHolder::new(table, source, LM_HEAD_COLUMN_DRAWS as u64)
}

/// One committed table as the ops layer sees it: the caller starts a proof
/// with a column reservation, folds and opens through the holder, and never
/// receives the complete weight matrix. The production implementation wraps
/// the bounded holder; the small fixture uses an in-memory row source behind
/// the same interface.
pub trait SharedTableSource {
    fn reader(&self) -> &IndexReader;
    fn shape(&self) -> IndexShape;
    fn begin(&mut self, columns: usize) -> Result<()>;
    fn fold(&mut self, gamma: &[Fp3], shape: &GuShape) -> Result<LmHeadGuProvider>;
    fn open(&mut self, positions: &[u64]) -> Result<Vec<ColumnOpening>>;
    fn forget(&mut self) -> Result<()>;

    /// Releases the row source after its last use. Implementations that keep
    /// no source may leave the default.
    fn release(&mut self) -> Result<()> {
        Ok(())
    }
}

/// The holder-backed source: folding and opening use the holder's bounded
/// passes and its column reservation; the G_u assembly uses the same private
/// randomness the key path uses.
pub struct HolderSharedSource<S: TableRowSource> {
    holder: StreamingHolder<S>,
    randomness: LmHeadGuRandomness,
}

impl<S: TableRowSource> HolderSharedSource<S> {
    pub fn new(holder: StreamingHolder<S>, randomness: LmHeadGuRandomness) -> Self {
        Self { holder, randomness }
    }

    pub fn holder(&self) -> &StreamingHolder<S> {
        &self.holder
    }
}

impl<S: TableRowSource> SharedTableSource for HolderSharedSource<S> {
    fn reader(&self) -> &IndexReader {
        self.holder.reader()
    }

    fn shape(&self) -> IndexShape {
        self.holder.reader().shape
    }

    fn begin(&mut self, columns: usize) -> Result<()> {
        self.holder.begin(columns)
    }

    fn fold(&mut self, gamma: &[Fp3], shape: &GuShape) -> Result<LmHeadGuProvider> {
        fold_lm_head_gu_holder(&mut self.holder, shape, gamma, &self.randomness)
    }

    fn open(&mut self, positions: &[u64]) -> Result<Vec<ColumnOpening>> {
        self.holder.open(positions)
    }

    fn forget(&mut self) -> Result<()> {
        self.holder.forget()
    }

    fn release(&mut self) -> Result<()> {
        self.holder.release_source()
    }
}

#[cfg(not(test))]
fn mem(_stage: &str) {}

#[cfg(test)]
fn mem(stage: &str) {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return;
    };
    let mut hwm = 0u64;
    let mut rss = 0u64;
    let mut anon = 0u64;
    let mut file = 0u64;
    let mut shmem = 0u64;
    for line in status.lines() {
        let mut parts = line.split_whitespace();
        let Some(name) = parts.next() else { continue };
        let value = parts.next().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
        match name {
            "VmHWM:" => hwm = value,
            "VmRSS:" => rss = value,
            "RssAnon:" => anon = value,
            "RssFile:" => file = value,
            "RssShmem:" => shmem = value,
            _ => {}
        }
    }
    eprintln!(
        "MEM {stage} hwm_kb {hwm} rss_kb {rss} anon_kb {anon} file_kb {file} shmem_kb {shmem}"
    );
}

pub fn run_limited_shared_to_columns(
    setup: LimitedSharedSession,
    table: &mut dyn SharedTableSource,
    omega_randomness: Option<&LmHeadOmegaRandomness>,
    old_mask: Option<&LmHeadOldMaskProvider>,
    switch: Option<LimitedSwitchInputs>,
) -> Result<LimitedSharedColumns> {
    let base_binding = setup.base.clone();
    let expected_switches = setup.switches.clone();
    let expected_positions = setup.switch_positions.clone();
    let column_work_bits = setup.column_work_bits;
    let switch_work_bits = setup.switch_work_bits;
    let mut session = FoldeqSession::open(setup.plan, setup.binder, setup.chain)?;
    let mut record = LimitedChainRecord::default();
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
    table.begin(LM_HEAD_COLUMN_DRAWS as usize)?;
    let provider = table.fold(&gamma, &gu_shape)?;
    record.gu_root = Some(provider.root());
    mem("fold");
    session.send_gu(&provider.root())?;
    session.shared_seeds()?;
    let seeds = session.gu_shared_seeds().to_vec();
    let answers = provider.answers(&seeds)?;
    record.gu_answers = answers.clone();
    for answer in &answers {
        session.receive_gu_answer(*answer)?;
    }
    session.join_gu_record()?;
    mem("record");
    record.column_nonce = session.draw_queries_with_work(column_work_bits, None)?;
    mem("queries");
    let queries = session.queries().to_vec();
    let positions = unique_sorted(&queries);
    let unique = table.open(&positions)?;
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
    record.column_openings = openings;
    table.forget()?;
    table.release()?;
    mem("released");
    session.join_gu_read_columns()?;
    mem("columns");
    let mut omega_root = None;
    let mut zeta = Vec::new();
    let mut fold_output = None;
    let mut folded_main_message = Vec::new();
    let mut fold_stage = None;
    let mut cs_claims = None;
    let mut cs_message = None;
    let mut stage_fold = None;
    let mut stage_rows = None;
    let mut sealed = None;
    let mut base_proof = None;
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
        record.omega_root = Some(omega_provider.root());
        record.omega_answers = session.omega_rows().iter().map(|row| row.target).collect();
        mem("omega");
        zeta = session.draw_zeta()?;
        omega_root = Some(omega_provider.root());
        if let Some(old_mask) = old_mask {
        let fold_shape = session.plan().fold;
        let group_id = session.plan().group_id;
        let (fold_row, bat_layout) = assemble_limited_fold_row(
            &mut session,
            provider.shape(),
            omega_provider.shape(),
            fold_shape,
            group_id,
            &psi_lo,
            &zeta,
            omega.matrix_rows,
            omega.k_lo,
            omega_provider.oracle(),
        )?;
        let prover_row = fold_row.clone();
        session.fold_bind_row(fold_row)?;
        session.start_fold(setup.mask.message())?;
        record.fold_start = Some(setup.mask.message().clone());
        let eps = session.epsilon().ok_or(Error::InvalidSessionState)?;
        let coins = ByteMaskCoins::new(setup.mask.coins_seed())?;
        let fold_layout = session
            .plan()
            .fold_layout
            .clone()
            .ok_or(Error::InvalidConfig)?;
        let main_messages = group_main_messages(&provider, &omega_provider, &bat_layout)?;
        let old_mu = old_mask.mu()?;
    let mut prover = FoldProverSession::new(
            &fold_shape,
            &fold_layout,
            setup.mask.mask(),
            prover_row,
            main_messages.clone(),
            old_mu.clone(),
            Vec::new(),
        )?;
    prover.start(eps, &coins)?;
        session.fold_rounds_start()?;
        for _ in 0..fold_shape.k {
            let round = prover.round()?;
            record.fold_rounds.push((round[0], round[2]));
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
        mem("fold-stage");
        pending_rows = session.pending_gu_rows().len() + session.pending_column_count();
        transcript = *session.transcript_value();
        if let Some(inputs) = switch.as_ref() {
            let mut provided = Vec::new();
            let mut provided_positions = Vec::new();
            let mut cursor = Some(inputs);
            while let Some(step) = cursor {
                provided.push(step.target);
                provided_positions.push(step.positions);
                cursor = step.next.as_deref();
            }
            check_stage_sequence(&expected_switches, &provided)?;
            if provided_positions != expected_positions {
                return Err(Error::StatementMismatch);
            }
        }
        if let (Some(inputs), Some(stage0)) = (switch, fold_stage.as_ref()) {
            let mut chain = session.take_chain_for_base()?;
            let mut stage = stage0.clone();
            let mut step = Some(inputs);
            let mut previous: Option<CsMultiProver> = None;
            let mut folds: Vec<StageFoldOutcome> = Vec::new();
            let mut stages: Vec<LimitedFoldStage> = Vec::new();
            let mut first = true;
            let mut last_target: Option<SwitchTarget> = None;
            let mut last_root: Option<Digest> = None;
            let mut base_step: Option<BaseStep> = None;
            while let Some(inputs) = step {
                let members = if first {
                    SwitchMembers::Initial {
                        gu: &provider,
                        omega: &omega_provider,
                        old_mask,
                        fresh: &setup.mask,
                    }
                } else {
                    let member = previous.as_ref().ok_or(Error::InvalidSessionState)?;
                    let fresh = &folds
                        .last()
                        .ok_or(Error::InvalidSessionState)?
                        .fresh;
                    SwitchMembers::Stage { member, fresh }
                };
                let run = run_switch(
                    chain,
                    &stage,
                    inputs.target,
                    inputs.positions,
                    switch_work_bits,
                    inputs.plan.clone(),
                    inputs.main_tails.clone(),
                    inputs.mask_tails.clone(),
                    inputs.pad,
                    &inputs.salts,
                    members,
                )?;
                chain = run.chain;
                last_target = Some(inputs.target);
                last_root = Some(run.declared_root);
                base_step = inputs.base.clone();
                cs_claims = Some(run.claims.clone());
                cs_message = Some(run.message.clone());
                transcript = *chain.value();
                record.switches.push(run.record.clone());
                previous = Some(run.prover);
                mem("switch");
                if let Some(spec) = inputs.stage_fold {
                    let row = FoldInputRow {
                        group: spec.group,
                        k: spec.shape.k,
                        h: spec.shape.h,
                        members: vec![spec.member],
                        coefficients: run.claims.combined.main.clone(),
                        mask_coefficients: run.claims.combined.mask.clone(),
                        kept: Vec::new(),
                        target: run.claims.combined.target,
                    };
                    let outcome = run_stage_fold(
                        chain.clone(),
                        &spec,
                        row,
                        run.message.main.clone(),
                        run.message.mask.clone(),
                    )?;
                    if let Some(last) = record.switches.last_mut() {
                        last.stage_fold = Some(outcome.record.clone());
                    }
                    chain = outcome.chain.clone();
                    transcript = *chain.value();
                    let member = previous.as_ref().ok_or(Error::InvalidSessionState)?;
                    let rows = build_stage_fold_stage(
                        &spec,
                        &outcome,
                        member.root(),
                        member.plan().clone(),
                        member.main_tails(),
                        member.mask_tails(),
                    )?;
                    stage = rows.clone();
                    folds.push(outcome);
                    stages.push(rows);
                }
                step = inputs.next.map(|next| *next);
                first = false;
            }
            stage_fold = folds.pop();
            stage_rows = stages.pop();
            if let Some(base) = base_step {
                let target = last_target.ok_or(Error::InvalidSessionState)?;
                let root = last_root.ok_or(Error::InvalidSessionState)?;
                let claims = cs_claims.as_ref().ok_or(Error::InvalidSessionState)?;
                let message = cs_message.as_ref().ok_or(Error::InvalidSessionState)?;
                let last = previous.as_ref().ok_or(Error::InvalidSessionState)?;
                let rows = last_stage_single_rows(target, claims, message)?;
                let prover = LastBaseProver::new(
                    rows.clone(),
                    base_binding.plan.clone(),
                    &base.salts,
                    &base.randomness,
                )?;
                let messages = prover.messages()?;
                let mut session = LastBaseSession::new(
                    chain.clone(),
                    rows.clone(),
                    last.plan().clone(),
                    root,
                    base_binding.plan.clone(),
                )?;
                session.receive_star(messages.star_root, messages.target_star)?;
                let lambda = session.draw_lambda()?;
                let prepared = prover.base_rows(lambda)?;
                session.receive_rows(prepared.clone())?;
                let positions = session.draw_base_positions(base_binding.positions)?;
                let star_opening = prover.open_star(&positions)?;
                let final_opening = last.open(&positions)?;
                session.verify_openings(&star_opening, &final_opening)?;
                let typed_seal = session.finish()?;
                let proof = encode_last_base_proof(
                    &messages,
                    &prepared,
                    &star_opening,
                    &final_opening,
                    prover.plan(),
                    last.plan(),
                )?;
                let verifier = LastBaseSession::new(
                    chain.clone(),
                    rows,
                    last.plan().clone(),
                    root,
                    base_binding.plan.clone(),
                )?;
                let decoded_seal =
                    verify_last_base_proof(verifier, base_binding.positions, &proof)?;
                if decoded_seal != typed_seal {
                    return Err(Error::InvalidProof);
                }
                sealed = Some(decoded_seal);
                mem("base");
                base_proof = Some(proof);
                transcript = decoded_seal;
            }
        }
        }
    } else if omega_randomness.is_some() {
        return Err(Error::InvalidConfig);
    }
    record.base_proof = base_proof.clone();
    record.sealed = sealed;
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
        sealed,
        base_proof,
        pending_rows,
        transcript,
        record,
    })
}

/// The public part of one switch input: the target, the tree plan and the
/// optional following stage FOLD shape. Private tails, salts and pads are not
/// copied into the verifier plan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LimitedStageFoldPlan {
    pub shape: FoldShape,
    pub layout: FoldGroupLayout,
    pub mask_tree: MaskTreePlan,
    pub group: u64,
    pub member: u64,
    pub fresh: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LimitedSwitchPlan {
    pub target: SwitchTarget,
    pub positions: u64,
    pub plan: RuntimeTreePlan,
    pub stage_fold: Option<LimitedStageFoldPlan>,
}

impl LimitedSwitchPlan {
    pub fn from_inputs(mut inputs: Option<&LimitedSwitchInputs>) -> Vec<Self> {
        let mut plans = Vec::new();
        while let Some(input) = inputs {
            plans.push(Self {
                target: input.target.clone(),
                positions: input.positions,
                plan: input.plan.clone(),
                stage_fold: input.stage_fold.as_ref().map(|spec| LimitedStageFoldPlan {
                    shape: spec.shape,
                    layout: spec.layout.clone(),
                    mask_tree: spec.mask_tree.clone(),
                    group: spec.group,
                    member: spec.member,
                    fresh: spec.fresh,
                }),
            });
            inputs = input.next.as_deref();
        }
        plans
    }
}

/// Rejects a handoff switch plan whose trusted-derived public fields differ
/// from the compiled plan's. The runtime tree's run seed and the column salts
/// are committed private values and are not compared.
fn check_switch_public_plans(
    expected: &[SwitchPlanPublic],
    switches: &[LimitedSwitchPlan],
) -> Result<()> {
    if switches.len() != expected.len() {
        return Err(Error::IncompleteArtifact);
    }
    for (plan, expected) in switches.iter().zip(expected) {
        if SwitchPlanPublic::from_plan(plan) != *expected {
            return Err(Error::InvalidProof);
        }
    }
    Ok(())
}

/// Checks the trusted policy identity against the committed reader and the
/// statement the verifier is about to use. The committed values come from the
/// reader and the fixed protocol identity; the proof's self-reported metadata
/// cannot replace the policy value.
fn check_trusted_table_identity(
    trusted_identity: &LimitedTableIdentity,
    reader: &IndexReader,
    trusted: &LimitedKeyStatement,
) -> Result<()> {
    trusted.validate()?;
    let rows = 1u64
        .checked_shl(reader.shape.v)
        .ok_or(Error::InvalidConfig)?;
    if trusted_identity.model_id != trusted.model_commitment
        || trusted.table_root != reader.root
        || trusted.table_configuration != reader.configuration
        || trusted.table_shape != reader.shape
        || !trusted.matrix_rows.is_power_of_two()
        || trusted.matrix_rows != rows
    {
        return Err(Error::StatementMismatch);
    }
    trusted_identity.check_committed(
        reader.root,
        zkie_core::artifact::digest_bytes(&reader.record.encode()),
        reader.configuration,
        reader.shape,
        reader.shape.v,
        trusted.k_lo,
        trusted.k_col,
        SESSION_PROTOCOL_IDENTITY,
    )
}

/// Replays the whole chain from the public setup, the public switch plans and
/// the artifact messages. The replay builds only zero coefficient vectors of
/// the required shapes; every claim used for a check is recomputed from the
/// verifier session or the artifact. Returns the final seal.
pub fn verify_limited_chain(
    setup: &LimitedVerifySetup,
    trusted_identity: &LimitedTableIdentity,
    trusted: &LimitedKeyStatement,
    switches: &[LimitedSwitchPlan],
    old_mask_plan: Option<MaskTreePlan>,
    old_mask_root: Option<Digest>,
    artifact: &LimitedChainArtifact,
) -> Result<Digest> {
    check_trusted_table_identity(trusted_identity, &setup.plan.reader, trusted)?;
    if artifact.switches.len() != switches.len()
        || setup.switch_positions.len() != switches.len()
        || setup.switch_public_plans.len() != switches.len()
    {
        return Err(Error::IncompleteArtifact);
    }
    check_switch_public_plans(&setup.switch_public_plans, switches)?;
    let provided_targets: Vec<SwitchTarget> =
        switches.iter().map(|plan| plan.target.clone()).collect();
    if provided_targets != setup.switches {
        return Err(Error::InvalidProof);
    }
    let (session, fold_output) = verify_limited_fold(setup, trusted, artifact)?;
    let mut chain = session.take_chain_for_base()?;
    let initial_fresh_plan = setup.fresh_plan.clone();
    let mut stage = public_initial_basis(setup, artifact, &fold_output, &initial_fresh_plan)?;
    let mut last_plan = None;
    let mut last_root = None;
    let mut last_claims = None;
    let mut last_target = None;
    for (index, plan) in switches.iter().enumerate() {
        let record = artifact
            .switches
            .get(index)
            .ok_or(Error::IncompleteArtifact)?;
        let target = plan.target.clone();
        if plan.positions != setup.switch_positions[index] {
            return Err(Error::InvalidProof);
        }
        let source = if index == 0 {
            OldMaskSource::Tree {
                plan: old_mask_plan.clone().ok_or(Error::IncompleteArtifact)?,
                root: old_mask_root.ok_or(Error::IncompleteArtifact)?,
            }
        } else {
            OldMaskSource::Member
        };
        if index == 0 && record.old_mask_root != old_mask_root {
            return Err(Error::InvalidProof);
        }
        if index > 0 && record.old_mask_opening.is_some() {
            return Err(Error::InvalidProof);
        }
        let u = GuOracleShape {
            code: target.code,
            main_rows: target.main_rows,
            main_deg: 3,
            mask_rows: target.mask_rows,
            mask_code: Some(target.mask_code),
            answers: target.answers,
        };
        let identity = fold_identity(&stage.0);
        if record.identity != identity {
            return Err(Error::InvalidProof);
        }
        let mut cs = CsMultiSession::new(
            chain.clone(),
            stage.0.clone(),
            stage.1.clone(),
            target.clone(),
            u,
            plan.plan.clone(),
            source,
        )?;
        cs.receive_declarations(record.identity, record.declared_root)?;
        cs.draw_sample_seed()?;
        cs.receive_answer(record.sample_answer)?;
        cs.set_work_bits(setup.switch_work_bits)?;
        let count = usize::try_from(plan.positions).map_err(|_| Error::ResourceLimit)?;
        if count == 0 {
            return Err(Error::IncompleteArtifact);
        }
        cs.draw_positions_theta_with_nonce(count, Some(record.nonce))?;
        let claims = cs
            .authenticate_and_register(
                &record.member_openings,
                &record.fresh_opening,
                record.old_mask_opening.as_ref(),
            )?
            .clone();
        chain = cs.into_chain();
        last_plan = Some(plan.plan.clone());
        last_root = Some(record.declared_root);
        last_target = Some(target);
        last_claims = Some(claims.clone());
        if let Some(spec) = &plan.stage_fold {
            let record_stage = record
                .stage_fold
                .as_ref()
                .ok_or(Error::IncompleteArtifact)?;
            let row = FoldInputRow {
                group: spec.group,
                k: spec.shape.k,
                h: spec.shape.h,
                members: vec![spec.member],
                coefficients: claims.combined.main.clone(),
                mask_coefficients: claims.combined.mask.clone(),
                kept: Vec::new(),
                target: claims.combined.target,
            };
            let (new_chain, output) = verify_stage_fold(chain, spec, &row, record_stage)?;
            chain = new_chain;
            stage = public_stage_basis(
                spec,
                &output,
                record.declared_root,
                plan.plan.clone(),
                record_stage.start.root,
            )?;
        } else if index + 1 < switches.len() {
            return Err(Error::IncompleteArtifact);
        }
    }
    let claims = last_claims.ok_or(Error::IncompleteArtifact)?;
    let target = last_target.ok_or(Error::IncompleteArtifact)?;
    let final_plan = last_plan.ok_or(Error::IncompleteArtifact)?;
    let final_root = last_root.ok_or(Error::IncompleteArtifact)?;
    if artifact.base_proof.is_empty() || target.main_rows != 1 || target.mask_rows != 1 {
        return Err(Error::IncompleteArtifact);
    }
    let ell = usize::try_from(target.code.ell).map_err(|_| Error::ResourceLimit)?;
    let r = usize::try_from(target.code.r).map_err(|_| Error::ResourceLimit)?;
    let mask_ell = usize::try_from(target.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
    let mask_r = usize::try_from(target.mask_code.r).map_err(|_| Error::ResourceLimit)?;
    let rows = SingleRows {
        code: target.code,
        mask_code: target.mask_code,
        main: vec![Fp3::ZERO; ell],
        main_tail: vec![Fp3::ZERO; r],
        masks: vec![vec![Fp3::ZERO; mask_ell]],
        mask_tails: vec![vec![Fp3::ZERO; mask_r]],
        claim: claims.combined.main.clone(),
        mask_claims: vec![claims.combined.mask.clone()],
        target: claims.combined.target,
    };
    rows.check()?;
    let verifier = LastBaseSession::new(
        chain,
        rows,
        final_plan,
        final_root,
        setup.base.plan.clone(),
    )?;
    let sealed = verify_last_base_proof(verifier, setup.base.positions, &artifact.base_proof)?;
    if sealed != artifact.sealed {
        return Err(Error::InvalidProof);
    }
    Ok(sealed)
}

fn public_initial_basis(
    setup: &LimitedVerifySetup,
    artifact: &LimitedChainArtifact,
    output: &FoldOutput,
    fresh_plan: &MaskTreePlan,
) -> Result<(FoldHandoff, FoldBasis)> {
    let layout = setup
        .plan
        .fold_layout
        .clone()
        .ok_or(Error::InvalidConfig)?;
    let anchor = FoldHandoffAnchor {
        shape: setup.plan.fold,
        group: setup.plan.group_id,
        members: layout.members.iter().map(|member| member.oracle).collect(),
        fresh: LM_HEAD_FRESH_MASK,
        fresh_root: artifact.fold_start.root,
        tree_number: artifact.fold_start.tree_number,
        layout: layout.clone(),
        fold_gamma: output.gamma.clone(),
        fold_epsilon: output.eps,
        fold_sum: output.sum,
    };
    let handoff = FoldHandoff::from_output(output, &anchor)?;
    let mut members = Vec::with_capacity(layout.members.len());
    for member in &layout.members {
        if let Some(gu) = setup.plan.gu.as_ref() {
            if member.oracle == gu.oracle {
                members.push(FoldMember {
                    oracle: member.oracle,
                    main_code: gu.u.code,
                    mask_code: handoff.mask_code,
                    root: artifact.gu_root,
                    tree: gu.tree.clone(),
                });
                continue;
            }
        }
        if let Some(omega) = setup.plan.omega.as_ref() {
            if member.oracle == omega.oracle {
                members.push(FoldMember {
                    oracle: member.oracle,
                    main_code: omega.u.code,
                    mask_code: handoff.mask_code,
                    root: artifact.omega_root.ok_or(Error::IncompleteArtifact)?,
                    tree: omega.tree.clone(),
                });
                continue;
            }
        }
        return Err(Error::StatementMismatch);
    }
    let basis = FoldBasis {
        layout: handoff.layout.clone(),
        fresh_root: handoff.fresh_root,
        fresh_tree: fresh_plan.clone(),
        members,
    };
    Ok((handoff, basis))
}

fn public_stage_basis(
    spec: &LimitedStageFoldPlan,
    output: &FoldOutput,
    member_root: Digest,
    member_plan: RuntimeTreePlan,
    fresh_root: Digest,
) -> Result<(FoldHandoff, FoldBasis)> {
    let anchor = FoldHandoffAnchor {
        shape: spec.shape,
        group: spec.group,
        members: vec![spec.member],
        fresh: spec.fresh,
        fresh_root,
        tree_number: spec.mask_tree.tree_number,
        layout: spec.layout.clone(),
        fold_gamma: output.gamma.clone(),
        fold_epsilon: output.eps,
        fold_sum: output.sum,
    };
    let handoff = FoldHandoff::from_output(output, &anchor)?;
    let basis = FoldBasis {
        layout: handoff.layout.clone(),
        fresh_root: handoff.fresh_root,
        fresh_tree: spec.mask_tree.clone(),
        members: vec![FoldMember {
            oracle: spec.member,
            main_code: spec.shape.code,
            mask_code: handoff.mask_code,
            root: member_root,
            tree: member_plan,
        }],
    };
    Ok((handoff, basis))
}

fn verify_stage_fold(
    mut chain: Chain,
    spec: &LimitedStageFoldPlan,
    row: &FoldInputRow,
    record: &LimitedStageFoldArtifact,
) -> Result<(Chain, FoldOutput)> {
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
        vec![spec.mask_tree.clone()],
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
    let mut oracles = RuntimeOracles::default();
    oracles.receive_mask(
        &statement,
        &spec.mask_tree,
        spec.fresh,
        &fold_mask_notice(&spec.shape)?,
        record.start.root,
    )?;
    fold_start_verify(&spec.shape, &record.start, &statement, &oracles)?;
    let mut message = chain.message();
    message.push_u64(record.start.id);
    message.push_u64(record.start.claimed_group);
    message.push_u64(record.start.notice.mask_rows);
    message.push_u64(record.start.notice.mask_code.log_n as u64);
    message.push_u64(record.start.notice.mask_code.ell);
    message.push_u64(record.start.notice.mask_code.r);
    message.push_u64(record.start.notice.answers as u64);
    message.push_u64(record.start.notice.leaf_class as u64);
    for chunk in record.start.root.0.chunks(8) {
        let word = u64::from_le_bytes(chunk.try_into().map_err(|_| Error::InvalidEncoding)?);
        message.push_u64(word);
    }
    message.push_fp3(&record.start.sum);
    let prefix = message.prefix(KIND_EPSILON, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    let eps = chain.draw_e()?;
    if record.rounds.len() != spec.shape.k as usize {
        return Err(Error::InvalidProof);
    }
    let mut claim = FoldRoundVerifier::new(record.start.sum, eps, row.target);
    let mut gammas = Vec::with_capacity(record.rounds.len());
    for (first, second) in &record.rounds {
        let mut message = chain.message();
        message.push_fp3(first);
        message.push_fp3(second);
        let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
        if !chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        let gamma = chain.draw_e()?;
        claim.round(*first, *second, gamma);
        gammas.push(gamma);
    }
    let output = fold_output(
        &spec.shape,
        &spec.layout,
        row,
        record.start.id,
        record.start.root,
        record.start.tree_number,
        eps,
        record.start.sum,
        &gammas,
        claim.claim(),
    )?;
    if output.fresh_root != record.start.root || output.sum != record.start.sum {
        return Err(Error::InvalidProof);
    }
    Ok((chain, output))
}

/// Replays the shared part of the chain from public inputs and the artifact
/// messages, then verifies the first FOLD. The only constructed coefficients
/// are zeros of the shape's length: the claim vector itself is a prover
/// witness, while the target is recomputed from the public claim targets and
/// the drawn challenge.
pub fn verify_limited_fold(
    setup: &LimitedVerifySetup,
    trusted: &LimitedKeyStatement,
    artifact: &LimitedChainArtifact,
) -> Result<(FoldeqSession, FoldOutput)> {
    if artifact.statement != *trusted {
        return Err(Error::InvalidProof);
    }
    if artifact.column_openings.is_empty()
        || artifact.column_openings.len() > setup.plan.column_draws
        || artifact.fold_rounds.len() != setup.plan.fold.k as usize
    {
        return Err(Error::InvalidProof);
    }
    let mut session = FoldeqSession::open(
        setup.plan.clone(),
        setup.binder.clone(),
        setup.chain.clone(),
    )?;
    session.bind_first_query(&setup.first_query)?;
    session.draw_gamma()?;
    session.send_gu(&artifact.gu_root)?;
    session.shared_seeds()?;
    for answer in &artifact.gu_answers {
        session.receive_gu_answer(*answer)?;
    }
    session.join_gu_record()?;
    session.draw_queries_with_work(
        setup.column_work_bits,
        Some(artifact.column_nonce),
    )?;
    let mut openings = Vec::with_capacity(session.queries().len());
    for query in session.queries() {
        let at = artifact
            .column_openings
            .binary_search_by_key(query, |opening| opening.position)
            .map_err(|_| Error::InvalidProof)?;
        openings.push(artifact.column_openings[at].clone());
    }
    session.read_queries(&openings)?;
    session.join_gu_read_columns()?;
    let mut zeta = Vec::new();
    if let Some(root) = artifact.omega_root {
        session.send_omega(&root)?;
        session.omega_seeds()?;
        for answer in &artifact.omega_answers {
            session.receive_omega_answer(*answer)?;
        }
        session.join_omega()?;
        zeta = session.draw_zeta()?;
    } else if !artifact.omega_answers.is_empty() {
        return Err(Error::IncompleteArtifact);
    }
    let gu = setup.plan.gu.as_ref().ok_or(Error::InvalidConfig)?;
    let omega_plan = setup.plan.omega.as_ref().ok_or(Error::InvalidConfig)?;
    let psi_lo: Vec<Fp3> = match session.psi() {
        Some(psi) => psi
            .get(..omega_plan.k_lo as usize)
            .ok_or(Error::ShapeMismatch)?
            .to_vec(),
        None if omega_plan.k_lo == 0 => Vec::new(),
        None => return Err(Error::StatementMismatch),
    };
    let (row, _) = assemble_limited_fold_row(
        &mut session,
        gu,
        &omega_plan.u,
        setup.plan.fold,
        setup.plan.group_id,
        &psi_lo,
        &zeta,
        omega_plan.matrix_rows,
        omega_plan.k_lo,
        omega_plan.oracle,
    )?;
    session.fold_bind_row(row)?;
    session.start_fold(&artifact.fold_start)?;
    session.fold_rounds_start()?;
    for (first, second) in &artifact.fold_rounds {
        session.receive_fold_round(*first, *second)?;
    }
    let output = session.fold_finish()?;
    if output.sum != artifact.fold_start.sum
        || output.fresh_root != artifact.fold_start.root
        || output.tree_number != artifact.fold_start.tree_number
    {
        return Err(Error::InvalidProof);
    }
    Ok((session, output))
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
#[derive(Clone, Debug, Serialize, Deserialize)]
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
#[derive(Clone, Debug, Serialize, Deserialize)]
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
/// Builds the FOLD input row from the public claim set and the BAT challenge.
/// Sample and record rows come from the verifier session, column rows from the
/// verified openings, and the LIN row from the public psi/zeta weights. Both
/// the prover driver and the independent verifier use this function, so the
/// handoff is derived from the same declarations and challenge.
#[allow(clippy::too_many_arguments)]
fn assemble_limited_fold_row(
    session: &mut FoldeqSession,
    gu: &GuShape,
    omega: &GuOracleShape,
    fold_shape: FoldShape,
    group_id: u64,
    psi_lo: &[Fp3],
    zeta: &[Fp3],
    matrix_rows: u64,
    k_lo: u32,
    omega_oracle_id: u64,
) -> Result<(FoldInputRow, zkie_engine::pcs::request::bat::BatGroupLayout)> {
    let mut bat = BatStatement::new();
    bat.add_oracle(BatOracleShape {
        oracle: 0,
        main_rows: gu.u.main_rows,
        mask_rows: gu.u.mask_rows,
        main_code: Some(gu.u.code),
        mask_code: gu.u.mask_code,
        answers: gu.u.answers,
    })?;
    bat.add_oracle(BatOracleShape {
        oracle: 1,
        main_rows: omega.main_rows,
        mask_rows: 0,
        main_code: Some(omega.code),
        mask_code: None,
        answers: omega.answers,
    })?;
    let mut gu_samples = Vec::new();
    let mut plain = Vec::new();
    for row in session.pending_gu_rows() {
        match row.kind {
            zkie_engine::pcs::request::relation::GuRelationKind::Sample => {
                gu_samples.push(bat_row_from_relation(
                    row,
                    0,
                    gu.u.main_rows,
                    gu.u.code.ell,
                    BatRowKind::Sample,
                )?);
            }
            zkie_engine::pcs::request::relation::GuRelationKind::Record => {
                plain.push(bat_row_from_relation(
                    row,
                    0,
                    gu.u.main_rows,
                    gu.u.code.ell,
                    BatRowKind::Plain,
                )?);
            }
            zkie_engine::pcs::request::relation::GuRelationKind::Column => {
                return Err(Error::InvalidSessionState);
            }
        }
    }
    if gu_samples.len() != gu.u.answers as usize {
        return Err(Error::ShapeMismatch);
    }
    bat.sample(0, gu_samples)?;
    let mut omega_samples = Vec::new();
    for row in session.omega_rows() {
        if row.kind != zkie_engine::pcs::request::relation::GuRelationKind::Sample {
            return Err(Error::InvalidSessionState);
        }
        omega_samples.push(bat_row_from_relation(
            row,
            1,
            omega.main_rows,
            omega.code.ell,
            BatRowKind::Sample,
        )?);
    }
    if omega_samples.len() != omega.answers as usize {
        return Err(Error::ShapeMismatch);
    }
    bat.sample(1, omega_samples)?;
    let group = bat.new_group(fold_shape.code, fold_shape.mask_code)?;
    bat.join(group, 0, false)?;
    bat.join(group, 1, false)?;
    let mut plain_rows = 0usize;
    for row in plain {
        bat.add_row(group, row)?;
        plain_rows += 1;
    }
    let column_start = plain_rows;
    let column_targets = session.column_targets().to_vec();
    for target in &column_targets {
        let mut row = BatRow::new(BatRowKind::Plain);
        row.target = *target;
        bat.add_row(group, row)?;
    }
    bat.add_row(
        group,
        bat_lin_fold_row(gu, omega, psi_lo, zeta, matrix_rows, k_lo)?,
    )?;
    mem("bat-rows");
    let input = bat.begin_run(group)?;
    mem("bat-frozen");
    let x = session.draw_bat_challenge()?;
    let mut combined = combine(&bat, &input.layout, &input.rows, x)?;
    mem("bat-combine");
    let mut power = Fp3::ONE;
    for _ in 0..column_start {
        power = power.mul(&x);
    }
    for (draw, target) in column_targets.iter().enumerate() {
        let at = column_start
            .checked_add(draw)
            .ok_or(Error::ResourceLimit)?;
        let frozen = input.rows.get(at).ok_or(Error::StatementMismatch)?;
        if frozen.target != *target
            || !frozen.terms.is_empty()
            || !frozen.units.is_empty()
            || !frozen.runs.is_empty()
            || !frozen.forms.is_empty()
        {
            return Err(Error::StatementMismatch);
        }
        let row = session.column_relation_row(draw)?;
        if row.target != *target {
            return Err(Error::StatementMismatch);
        }
        carry_compact_table(
            &bat,
            &input.layout,
            0,
            &row.main,
            gu.u.main_rows,
            gu.u.code.ell,
            power,
            &mut combined,
        )?;
        power = power.mul(&x);
    }
    mem("bat-columns");
    let mut fold_row = bat.to_fold_input_row(combined, &input.layout)?;
    fold_row.group = group_id;
    fold_row.members = vec![gu.oracle, omega_oracle_id];
    Ok((fold_row, input.layout))
}

fn bat_lin_fold_row(
    gu: &GuShape,
    omega: &GuOracleShape,
    psi_lo: &[Fp3],
    zeta: &[Fp3],
    matrix_rows: u64,
    k_lo: u32,
) -> Result<BatRow> {
    let shape = gu;
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
    let omega_ell = usize::try_from(omega.code.ell).map_err(|_| Error::ResourceLimit)?;
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
#[derive(Clone)]
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
    /// The number of positions drawn from the old group's compiled plan at
    /// this switch. It is the previous stage's real count, never a fixture.
    pub positions: u64,
    pub plan: RuntimeTreePlan,
    pub main_tails: Vec<Vec<Fp3>>,
    pub mask_tails: Vec<Vec<Fp3>>,
    pub pad: Fp3,
    pub salts: Vec<Vec<u8>>,
    pub stage_fold: Option<StageFoldSpec>,
    pub next: Option<Box<LimitedSwitchInputs>>,
    pub base: Option<BaseStep>,
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
    pub record: LimitedStageFoldRecord,
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
    let start_message = fresh.message().clone();
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
    let row_target = row.target;
    let mut prover = FoldProverSession::new(
        &spec.shape,
        &spec.layout,
        fresh.mask(),
        row,
        main_messages,
        mask_messages,
        Vec::new(),
    )?;
    if prover.sum() != fresh.message().sum {
        return Err(Error::StatementMismatch);
    }
    prover.start(eps, &coins)?;
    let mut claim = FoldRoundVerifier::new(fresh.message().sum, eps, row_target);
    let mut gammas = Vec::with_capacity(spec.shape.k as usize);
    let mut rounds = Vec::with_capacity(spec.shape.k as usize);
    for _ in 0..spec.shape.k {
        let round = prover.round()?;
        rounds.push((round[0], round[2]));
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
    let output = prover.output(
        &spec.layout,
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
        record: LimitedStageFoldRecord {
            start: start_message,
            rounds,
        },
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

/// The label that separates the stage private-input expansion from every
/// transcript challenge domain.
const PRIVATE_INPUTS_LABEL: u64 = 0x5052_4956;

fn index_digest(words: &[u64]) -> Digest {
    let mut bytes = [0u8; 32];
    for (slot, word) in bytes.chunks_exact_mut(8).zip(words) {
        slot.copy_from_slice(&word.to_le_bytes());
    }
    Digest(bytes)
}

fn derive_salt(
    master: &Digest,
    stage: u64,
    column: u64,
    salt_bytes: usize,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(salt_bytes);
    let mut block = 0u64;
    while bytes.len() < salt_bytes {
        let digest = run_value(
            PRIVATE_INPUTS_LABEL,
            master,
            &index_digest(&[stage, column, block, 1]),
        )?;
        bytes.extend_from_slice(&digest.0);
        block = block.checked_add(1).ok_or(Error::ResourceLimit)?;
    }
    bytes.truncate(salt_bytes);
    Ok(bytes)
}

fn derive_seed(master: &Digest, stage: u64, domain: u64) -> Result<Digest> {
    run_value(
        PRIVATE_INPUTS_LABEL,
        master,
        &index_digest(&[stage, domain, 0, 2]),
    )
}

/// The private switch-stage inputs generated from one master seed: the new
/// main and mask encoding tails, the private pad and one salt per new column.
pub struct StagePrivateInputs {
    pub main_tails: Vec<Vec<Fp3>>,
    pub mask_tails: Vec<Vec<Fp3>>,
    pub pad: Fp3,
    pub salts: Vec<Vec<u8>>,
}

pub fn expand_stage_private(
    target: &SwitchTarget,
    columns: u64,
    salt_bytes: usize,
    master: &Digest,
    stage: u64,
) -> Result<StagePrivateInputs> {
    if master == &Digest::ZERO || salt_bytes == 0 {
        return Err(Error::InvalidConfig);
    }
    let mut coins = ByteMaskCoins::new(derive_seed(master, stage, 0)?)?;
    let main_r = usize::try_from(target.code.r).map_err(|_| Error::ResourceLimit)?;
    let mask_r = usize::try_from(target.mask_code.r).map_err(|_| Error::ResourceLimit)?;
    let main_rows = usize::try_from(target.main_rows).map_err(|_| Error::ResourceLimit)?;
    let mask_rows = usize::try_from(target.mask_rows).map_err(|_| Error::ResourceLimit)?;
    let mut main_tails = Vec::with_capacity(main_rows);
    for _ in 0..main_rows {
        let mut tail = Vec::with_capacity(main_r);
        for _ in 0..main_r {
            tail.push(coins.draw()?.0);
        }
        main_tails.push(tail);
    }
    let mut mask_tails = Vec::with_capacity(mask_rows);
    for _ in 0..mask_rows {
        let mut tail = Vec::with_capacity(mask_r);
        for _ in 0..mask_r {
            tail.push(coins.draw()?.0);
        }
        mask_tails.push(tail);
    }
    let pad = coins.draw()?.0;
    let columns = usize::try_from(columns).map_err(|_| Error::ResourceLimit)?;
    let mut salts = Vec::with_capacity(columns);
    for column in 0..columns {
        salts.push(derive_salt(
            master,
            stage,
            column as u64,
            salt_bytes,
        )?);
    }
    Ok(StagePrivateInputs {
        main_tails,
        mask_tails,
        pad,
        salts,
    })
}

fn expand_fold_mask(
    target: &SwitchTarget,
    salt_bytes: usize,
    master: &Digest,
    stage: u64,
) -> Result<LmHeadMaskRandomness> {
    let plan = MaskTreePlan {
        tree_number: 0x500 + stage,
        leaf_class: 0,
        log_n: target.mask_code.log_n,
        mask_rows: target.mask_rows,
        salt_bytes,
        run: derive_seed(master, stage, 10)?,
        configuration: derive_seed(master, stage, 11)?,
    };
    if plan.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    let columns = usize::try_from(plan.columns()).map_err(|_| Error::ResourceLimit)?;
    let mut salts = Vec::with_capacity(columns);
    for column in 0..columns {
        salts.push(derive_salt(master, stage + 0x1000, column as u64, salt_bytes)?);
    }
    Ok(LmHeadMaskRandomness {
        seed: derive_seed(master, stage, 12)?,
        salts,
    })
}

/// Builds the switch chain of the compiled once group from one private master
/// seed. Each step's target comes from the compiled stages, its tree plan from
/// the plan's salt length and the session run/configuration, its private tails
/// and salts from the seed, and the following FOLD shape from the same target.
/// The last step carries the plan-bound BASE masking salts.
pub fn plan_stage_chain(
    compiled: &CompiledPlan,
    answers: u32,
    configuration: Digest,
    run: Digest,
    master: Digest,
    base: &BaseStarBinding,
) -> Result<Option<LimitedSwitchInputs>> {
    let stages = &compiled.once.folded.stages;
    if stages.is_empty() || master == Digest::ZERO {
        return Err(Error::InvalidConfig);
    }
    let salt_bytes = compiled.once_foldeq.base_salt_bytes;
    let mut steps: Vec<LimitedSwitchInputs> = Vec::with_capacity(stages.len());
    for (index, stage) in stages.iter().enumerate() {
        let stage_index = index as u64;
        let positions = if index == 0 {
            compiled.once.folded.positions
        } else {
            stages[index - 1].positions
        };
        if positions == 0 {
            return Err(Error::InvalidConfig);
        }
        let target = SwitchTarget::from_stage(stage.main, stage.mask, stage.k, stage.h, answers)?;
        let tree = switch_tree_plan_multi(
            stage_index + 1,
            &target,
            salt_bytes,
            run,
            configuration,
        )?;
        let private = expand_stage_private(
            &target,
            tree.columns(),
            salt_bytes,
            &master,
            stage_index,
        )?;
        let stage_fold = if index + 1 < stages.len() {
            let public = stage_fold_public_plan(compiled, &target, stage_index, configuration)?;
            Some(StageFoldSpec {
                shape: public.shape,
                layout: public.layout,
                mask_tree: public
                    .mask_tree
                    .to_plan(derive_seed(&master, stage_index, 10)?),
                group: public.group,
                member: public.member,
                fresh: public.fresh,
                randomness: expand_fold_mask(&target, salt_bytes, &master, stage_index)?,
            })
        } else {
            None
        };
        let base_step = if index + 1 == stages.len() {
            let columns =
                usize::try_from(base.plan.columns()).map_err(|_| Error::ResourceLimit)?;
            let mut salts = Vec::with_capacity(columns);
            for column in 0..columns {
                salts.push(derive_salt(
                    &master,
                    stage_index + 0x2000,
                    column as u64,
                    base.plan.salt_bytes,
                )?);
            }
            Some(BaseStep {
                salts,
                randomness: LastBaseRandomness::Private,
            })
        } else {
            None
        };
        steps.push(LimitedSwitchInputs {
            target,
            positions,
            plan: tree,
            main_tails: private.main_tails,
            mask_tails: private.mask_tails,
            pad: private.pad,
            salts: private.salts,
            stage_fold,
            next: None,
            base: base_step,
        });
    }
    let mut tail = steps.pop().ok_or(Error::InvalidConfig)?;
    while let Some(mut step) = steps.pop() {
        step.next = Some(Box::new(tail));
        tail = step;
    }
    Ok(Some(tail))
}

/// The last-stage BASE private inputs: salts and randomness. The masking
/// oracle's tree plan and position count come from the bound compiled plan.
#[derive(Clone)]
pub struct BaseStep {
    pub salts: Vec<Vec<u8>>,
    pub randomness: LastBaseRandomness,
}

/// The plan-bound last-stage BASE configuration: the independent masking
/// oracle's tree plan and the stage's position count.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaseStarBinding {
    pub plan: RuntimeTreePlan,
    pub positions: usize,
}

/// One switch stage's public result: the advanced chain, the registered
/// claims, the switched message and the prover that committed it.
struct SwitchRun {
    chain: Chain,
    claims: SwitchedClaims,
    message: SwitchedRows,
    prover: CsMultiProver,
    declared_root: Digest,
    record: LimitedSwitchRecord,
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
    positions: u64,
    work_bits: u8,
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
    let identity = fold_identity(&stage.handoff);
    cs.receive_declarations(identity, prover.root())?;
    let seed = cs.draw_sample_seed()?;
    let sample_answer = prover.sample_answer(&seed)?;
    cs.receive_answer(sample_answer)?;
    cs.set_work_bits(work_bits)?;
    let count = usize::try_from(positions).map_err(|_| Error::ResourceLimit)?;
    let (positions, _theta, nonce) = cs.draw_positions_theta_with_nonce(count, None)?;
    let unique = unique_sorted(&positions);
    let mask_positions: Vec<u64> = positions
        .iter()
        .map(|position| *position % stage.handoff.mask_code.n())
        .collect();
    let mask_unique = unique_sorted(&mask_positions);
    let (member_openings, old_mask_root, old_mask_opening) = match &members {
        SwitchMembers::Initial {
            gu,
            omega,
            old_mask,
            ..
        } => (
            vec![gu.open(&unique)?, omega.open(&unique)?],
            Some(old_mask.root()),
            Some(old_mask.open(&mask_unique)?),
        ),
        SwitchMembers::Stage { member, .. } => (vec![member.open(&unique)?], None, None),
    };
    let fresh_opening = fresh.open(&mask_unique)?;
    let claims = cs
        .authenticate_and_register(&member_openings, &fresh_opening, old_mask_opening.as_ref())?
        .clone();
    let message = prover.message().clone();
    let declared_root = cs.declared_root().ok_or(Error::InvalidSessionState)?;
    let chain = cs.into_chain();
    Ok(SwitchRun {
        chain,
        claims,
        message,
        prover,
        declared_root,
        record: LimitedSwitchRecord {
            identity,
            declared_root,
            sample_answer,
            nonce,
            member_openings,
            fresh_opening,
            old_mask_root,
            old_mask_opening,
            stage_fold: None,
        },
    })
}

/// Checks that the switch sequence is exactly the compiled stage sequence:
/// the same number of switches, in the same order, with the same codes, row
/// variables and answer count. A skipped, added or reordered stage fails.
fn check_stage_sequence(expected: &[SwitchTarget], provided: &[SwitchTarget]) -> Result<()> {
    if expected.len() != provided.len() {
        return Err(Error::StatementMismatch);
    }
    for (want, got) in expected.iter().zip(provided) {
        if want != got {
            return Err(Error::StatementMismatch);
        }
    }
    Ok(())
}

/// Turns the final switched GroupRow into the last stage's SingleRows. The
/// final target has one real main row and one mask row (k = h = 0). The
/// reference layout carries a virtual second main row that was never sent and
/// is fixed to zero, so the direct one-row form has the same coefficients,
/// target and openings without a new commitment; a target with folded rows is
/// rejected instead of approximated.
fn last_stage_single_rows(
    target: SwitchTarget,
    claims: &SwitchedClaims,
    message: &SwitchedRows,
) -> Result<SingleRows> {
    if target.main_rows != 1 || target.mask_rows != 1 {
        return Err(Error::UnsupportedPrimitive);
    }
    if message.main_tails.len() != 1 || message.mask_tails.len() != 1 {
        return Err(Error::ShapeMismatch);
    }
    let rows = SingleRows {
        code: target.code,
        mask_code: target.mask_code,
        main: message.main.clone(),
        main_tail: message.main_tails[0].clone(),
        masks: vec![message.mask.clone()],
        mask_tails: vec![message.mask_tails[0].clone()],
        claim: claims.combined.main.clone(),
        mask_claims: vec![claims.combined.mask.clone()],
        target: claims.combined.target,
    };
    rows.check()?;
    if rows.claim.len() != usize::try_from(rows.code.ell).map_err(|_| Error::ResourceLimit)?
        || rows.masks.len() != 1
    {
        return Err(Error::StatementMismatch);
    }
    Ok(rows)
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
    fn the_limited_statement_binds_its_semantics_and_range() {
        let Some((context, plan)) = context() else {
            return;
        };
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

    fn sample_artifact() -> LimitedChainArtifact {
        use zkie_engine::pcs::request::code::build_code_plan;
        use zkie_engine::pcs::request::fold::MaskNotice;
        let code = build_code_plan(3, 4, 1).expect("code");
        LimitedChainArtifact {
            version: LIMITED_CHAIN_ARTIFACT_VERSION,
            column_nonce: 0,
            protocol_identity: SESSION_PROTOCOL_IDENTITY,
            statement: test_statement(),
            gu_root: Digest([0x44; 32]),
            gu_answers: vec![Fp3::from_u64(5)],
            omega_root: Some(Digest([0x45; 32])),
            omega_answers: vec![Fp3::from_u64(6)],
            column_openings: vec![ColumnOpening {
                position: 1,
                salt: vec![0x5a; 4],
                symbols: vec![Fp::from_u64(7)],
                path: vec![Digest([0x55; 32])],
            }],
            fold_start: FoldStartMessage {
                id: 1,
                claimed_group: 1,
                tree_number: 0x600,
                notice: MaskNotice {
                    mask_rows: 2,
                    mask_code: code,
                    answers: 0,
                    leaf_class: 0,
                },
                root: Digest([0x46; 32]),
                sum: Fp3::from_u64(8),
            },
            fold_rounds: vec![(Fp3::ONE, Fp3::from_u64(9))],
            switches: vec![LimitedSwitchArtifact {
                nonce: 0,
                identity: Digest([0x47; 32]),
                declared_root: Digest([0x48; 32]),
                sample_answer: Fp3::from_u64(10),
                member_openings: Vec::new(),
                fresh_opening: RuntimeTreeOpening::default(),
                old_mask_root: Some(Digest([0x49; 32])),
                old_mask_opening: Some(RuntimeTreeOpening::default()),
                stage_fold: None,
            }],
            base_proof: vec![1, 2, 3, 4],
            sealed: Digest([0x4a; 32]),
        }
    }

    #[test]
    fn the_chain_artifact_round_trips_and_rejects_damage() {
        let artifact = sample_artifact();
        let bytes = artifact.encode();
        let decoded = LimitedChainArtifact::decode(&bytes).expect("decode");
        assert_eq!(decoded.encode(), bytes);
        assert_eq!(decoded.statement, artifact.statement);
        assert_eq!(decoded.gu_root, artifact.gu_root);
        assert_eq!(decoded.column_openings[0].position, 1);
        assert_eq!(decoded.switches.len(), 1);
        assert_eq!(decoded.base_proof, vec![1, 2, 3, 4]);

        let mut version = bytes.clone();
        version[0] ^= 1;
        assert!(LimitedChainArtifact::decode(&version).is_err());

        let mut identity = bytes.clone();
        identity[8] ^= 1;
        assert!(LimitedChainArtifact::decode(&identity).is_err());

        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(LimitedChainArtifact::decode(&trailing).is_err());

        let mut truncated = bytes.clone();
        truncated.truncate(truncated.len() - 4);
        assert!(LimitedChainArtifact::decode(&truncated).is_err());

        // The statement starts after the version, identity and its length.
        let mut statement = bytes.clone();
        statement[48] ^= 1;
        assert!(LimitedChainArtifact::decode(&statement).is_err());
    }

    #[test]
    fn repeated_column_openings_are_stored_once() {
        let first = ColumnOpening {
            position: 1,
            salt: vec![1],
            symbols: vec![Fp::from_u64(2)],
            path: vec![Digest([3; 32])],
        };
        let same = first.clone();
        let unique = unique_column_openings(&[first.clone(), same]).expect("unique");
        assert_eq!(unique.len(), 1);
        let mut changed = first.clone();
        changed.symbols[0] = Fp::from_u64(9);
        assert!(unique_column_openings(&[first.clone(), changed]).is_err());
        let second = ColumnOpening {
            position: 2,
            ..first.clone()
        };
        let unique = unique_column_openings(&[first, second]).expect("unique");
        assert_eq!(unique.len(), 2);
        assert_eq!(unique[0].position, 1);
        assert_eq!(unique[1].position, 2);
    }

    #[test]
    fn an_incomplete_record_has_no_artifact() {
        let statement = test_statement();
        let record = LimitedChainRecord::default();
        assert!(LimitedChainArtifact::from_record(&statement, &record).is_err());
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
        let mut source = VecTableSource::new(messages, tails).expect("source");
        let table = commit_table_streaming(
            &mut source,
            shape,
            Digest([0x55; 32]),
            &salts,
            64,
            u64::MAX,
        )
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
        let openings = table.open_columns(&mut source, &positions).expect("openings");
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
        let mut source = VecTableSource::new(messages.clone(), tails.clone()).expect("source");
        let table = commit_table_streaming(
            &mut source,
            shape,
            configuration,
            &salts,
            64,
            u64::MAX,
        )
        .expect("table");
        let reader = table.reader.clone();
        let name = table.name.clone();
        let key = LmHeadKey {
            shape,
            reader,
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
        let switch_plan = switch_tree_plan_multi(
            1,
            &target,
            40,
            Digest([0x41; 32]),
            Digest([0x42; 32]),
        )?;
        let salts: Vec<Vec<u8>> = (0..switch_plan.columns())
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
        let final_code = build_code_plan(6, 8, 1)?;
        let final_mask = build_code_plan(6, 32, 1)?;
        let final_target = SwitchTarget::from_stage(final_code, final_mask, 0, 0, 1)?;
        let final_plan = switch_tree_plan_multi(
            3,
            &final_target,
            40,
            Digest([0x71; 32]),
            Digest([0x72; 32]),
        )?;
        let final_salts: Vec<Vec<u8>> = (0..final_plan.columns())
            .map(|position| vec![(position as u8) ^ 0x7c; 40])
            .collect();
        let base = BaseStarBinding {
            plan: RuntimeTreePlan {
                tree_number: 0x401,
                leaf_class: 0,
                log_n: final_code.log_n,
                log_n_zk: final_mask.log_n,
                parts: vec![RuntimeTreePart {
                    main_symbols: 3,
                    mask_symbols: 3,
                }],
                salt_bytes: 40,
                run: Digest([0x83; 32]),
                configuration: key.reader.configuration,
            },
            positions: 2,
        };
        let setup = LimitedSharedSession {
            plan,
            binder,
            chain,
            first_query,
            mask: fresh_mask,
            base,
            switches: vec![target, target2, final_target],
            switch_positions: vec![2, 2, 2],
            switch_public_plans: vec![
                SwitchPlanPublic {
                    positions: 2,
                    tree: RuntimeTreePlanPublic::from_plan(&switch_plan),
                    stage_fold: Some(StageFoldPlanPublic {
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
                        mask_tree: MaskTreePlanPublic {
                            tree_number: 1,
                            leaf_class: 0,
                            log_n: next_mask.log_n,
                            mask_rows: 4,
                            salt_bytes: 40,
                            configuration: key.reader.configuration,
                        },
                        group: 1,
                        member: 1,
                        fresh: 3,
                    }),
                },
                SwitchPlanPublic {
                    positions: 2,
                    tree: RuntimeTreePlanPublic::from_plan(&plan2),
                    stage_fold: Some(StageFoldPlanPublic {
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
                        mask_tree: MaskTreePlanPublic {
                            tree_number: 2,
                            leaf_class: 0,
                            log_n: next_mask.log_n,
                            mask_rows: 4,
                            salt_bytes: 40,
                            configuration: key.reader.configuration,
                        },
                        group: 1,
                        member: 1,
                        fresh: 3,
                    }),
                },
                SwitchPlanPublic {
                    positions: 2,
                    tree: RuntimeTreePlanPublic::from_plan(&final_plan),
                    stage_fold: None,
                },
            ],
            column_work_bits: 0,
            switch_work_bits: 0,
        };
        let source_salts: Vec<Vec<u8>> = (0..16)
            .map(|position| vec![(position as u8) ^ 0x5a; 40])
            .collect();
        let mut source = VecSource {
            messages: key.messages.clone(),
            tails: tails.clone(),
        };
        let streamed = zkie_engine::pcs::request::table::commit_table_streaming(
            &mut source,
            key.shape,
            key.reader.configuration,
            &source_salts,
            16,
            u64::MAX,
        )?;
        if streamed.reader.root != key.reader.root {
            return Err(Error::StatementMismatch);
        }
        let holder = StreamingHolder::new(
            streamed,
            source,
            LM_HEAD_COLUMN_DRAWS as u64,
        )?;
        let mut table = HolderSharedSource::new(holder, randomness);
        run_limited_shared_to_columns(
            setup,
            &mut table,
            Some(&omega_randomness),
            Some(&old_mask),
            Some(LimitedSwitchInputs {
                target,
                positions: 2,
                plan: switch_plan,
                main_tails,
                mask_tails,
                pad: Fp3::from_u64(0x9a),
                salts,
                base: None,
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
                    positions: 2,
                    plan: plan2,
                    main_tails: (0..4)
                        .map(|row| vec![Fp3::from_u64(0x220 + row as u64)])
                        .collect(),
                    mask_tails: (0..4)
                        .map(|row| vec![Fp3::from_u64(0x320 + row as u64)])
                        .collect(),
                    pad: Fp3::from_u64(0x9b),
                    salts: salts2,
                    base: None,
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
                            tree_number: 2,
                            leaf_class: 0,
                            log_n: next_mask.log_n,
                            mask_rows: 4,
                            salt_bytes: 40,
                            run: Digest([0x63; 32]),
                            configuration: key.reader.configuration,
                        },
                        group: 1,
                        member: 1,
                        fresh: 3,
                        randomness: LmHeadMaskRandomness {
                            seed: Digest([0x65; 32]),
                            salts: (0..32)
                                .map(|position| vec![(position as u8) ^ 0x6b; 40])
                                .collect(),
                        },
                    }),
                    next: Some(Box::new(LimitedSwitchInputs {
                        target: final_target,
                        positions: 2,
                        plan: final_plan,
                        main_tails: vec![vec![Fp3::from_u64(0x240)]],
                        mask_tails: vec![vec![Fp3::from_u64(0x340)]],
                        pad: Fp3::from_u64(0x9c),
                        salts: final_salts,
                        stage_fold: None,
                        next: None,
                        base: Some(BaseStep {
                            salts: (0..64)
                                .map(|position| vec![(position as u8) ^ 0x7d; 40])
                                .collect(),
                            randomness: LastBaseRandomness::Fixed(Digest([0x81; 32])),
                        }),
                    })),
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
        assert_eq!(run.record.gu_root, Some(run.root));
        assert_eq!(run.record.gu_answers, run.answers);
        assert_eq!(run.record.omega_root, run.omega_root);
        assert_eq!(run.record.column_openings.len(), run.queries.len());
        assert!(run.record.fold_start.is_some());
        assert_eq!(
            run.record.fold_rounds.len(),
            run.fold_output.as_ref().expect("fold output").gamma.len()
        );
        assert!(!run.record.switches.is_empty());
        assert!(run.record.switches.iter().all(|switch| {
            switch.identity != Digest::ZERO
                && switch.declared_root != Digest::ZERO
                && !switch.member_openings.is_empty()
        }));
        assert_eq!(run.record.base_proof, run.base_proof);
        assert_eq!(run.record.sealed, run.sealed);
        assert!(run.omega_root.is_some());
        assert_eq!(run.zeta.len(), 0);
        assert!(run.fold_output.is_some());
        let claims = run.cs_claims.as_ref().expect("cs claims");
        let message = run.cs_message.as_ref().expect("cs message");
        assert_eq!(claims.claims.len(), 8);
        assert_eq!(claims.combined.main.len(), 8);
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
        let sealed = run.sealed.expect("sealed transcript");
        assert_ne!(sealed, Digest::ZERO);
        let proof = run.base_proof.as_ref().expect("base proof bytes");
        assert!(!proof.is_empty());

    }


    #[test]
    fn the_last_stage_carries_the_group_row_unchanged() {
        use zkie_engine::pcs::request::code::build_code_plan;
        use zkie_engine::pcs::request::cs_stage::SwitchClaim;
        let code = build_code_plan(6, 8, 1).expect("code");
        let mask_code = build_code_plan(6, 32, 1).expect("mask code");
        let target = SwitchTarget::from_stage(code, mask_code, 0, 0, 1).expect("target");
        let claims = SwitchedClaims {
            positions: vec![1, 2],
            old_symbols: vec![Fp3::from_u64(3), Fp3::from_u64(4)],
            claims: Vec::new(),
            theta: Fp3::from_u64(5),
            combined: SwitchClaim {
                main: vec![Fp3::from_u64(6); code.ell as usize],
                mask: vec![Fp3::from_u64(7); mask_code.ell as usize],
                target: Fp3::from_u64(8),
            },
            main_len: code.ell,
            main_tail_len: code.r,
            mask_len: mask_code.ell,
            mask_tail_len: mask_code.r,
        };
        let message = SwitchedRows {
            main: vec![Fp3::from_u64(9); code.ell as usize],
            main_tails: vec![vec![Fp3::from_u64(10); code.r as usize]],
            mask: vec![Fp3::from_u64(11); mask_code.ell as usize],
            mask_tails: vec![vec![Fp3::from_u64(12); mask_code.r as usize]],
        };
        let rows = last_stage_single_rows(target, &claims, &message).expect("last rows");
        assert_eq!(rows.main, message.main);
        assert_eq!(rows.main_tail, message.main_tails[0]);
        assert_eq!(rows.masks, vec![message.mask.clone()]);
        assert_eq!(rows.mask_tails, vec![message.mask_tails[0].clone()]);
        assert_eq!(rows.claim, claims.combined.main);
        assert_eq!(rows.mask_claims, vec![claims.combined.mask.clone()]);
        assert_eq!(rows.target, claims.combined.target);
        assert_eq!(rows.main.len(), code.ell as usize);
        assert_eq!(rows.masks.len(), 1);
    }

    fn random_bytes(count: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; count];
        getrandom::fill(&mut bytes).expect("entropy");
        bytes
    }

    fn random_digest() -> Digest {
        let mut bytes = [0u8; 32];
        getrandom::fill(&mut bytes).expect("entropy");
        Digest(bytes)
    }

    fn random_fp() -> Fp {
        loop {
            let mut bytes = [0u8; 8];
            getrandom::fill(&mut bytes).expect("entropy");
            let word = u64::from_le_bytes(bytes);
            if word < zkie_engine::pcs::request::field::MODULUS {
                return Fp::from_u64(word);
            }
        }
    }

    fn random_fp3() -> Fp3 {
        Fp3::new(random_fp(), random_fp(), random_fp())
    }

    fn vm_hwm_kb() -> u64 {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                if let Some(value) = rest.split_whitespace().next() {
                    return value.parse().unwrap_or(0);
                }
            }
        }
        0
    }

    fn random_fps(count: usize) -> Vec<Fp> {
        let mut bytes = vec![0u8; count * 8];
        getrandom::fill(&mut bytes).expect("entropy");
        let mut out = Vec::with_capacity(count);
        for chunk in bytes.chunks_exact(8) {
            let word = u64::from_le_bytes(chunk.try_into().expect("word"));
            out.push(if word < zkie_engine::pcs::request::field::MODULUS {
                Fp::from_u64(word)
            } else {
                random_fp()
            });
        }
        out
    }

    #[test]
    #[ignore]
    fn real_gpt2_lm_head_chain() {
        use std::path::Path;
        use std::time::Instant;
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("repository root");
        let case = root.join("models/gpt2/case.toml");
        let policy_path = root.join("models/gpt2/case.policy");
        let public = root.join(
            "docs/benchmarks/20261110-0547-verify-bytes-109934bd-670a98c9/prod/gpt2-seq16/commit/public",
        );
        let statement_path = root.join(
            "docs/benchmarks/20261110-0547-verify-bytes-109934bd-670a98c9/prod/gpt2-seq16/proof/statement.bin",
        );
        let statement_bytes = std::fs::read(&statement_path).expect("statement");
        let statement_data: crate::provider::StatementData =
            bincode::deserialize(&statement_bytes).expect("statement data");
        let context = crate::s7_context::load_gpt2_lm_head_context(
            &case,
            &policy_path,
            &public,
            &statement_path,
            statement_data.model_commitment,
        )
        .expect("context");
        let request_plan = zkie_models_gpt2::lm_head_plan::lm_head_plan().expect("plan");
        let compiled = zkie_models_gpt2::lm_head_plan::lm_head_compiled().expect("compiled");
        request_plan.validate().expect("plan validate");
        request_plan
            .validate_compiled(&compiled)
            .expect("compiled validate");
        let parameters = limited_key_binding_parameters(&request_plan, context.plan_identifier)
            .expect("parameters");
        let configuration = parameters.digest().expect("configuration");
        let shape = request_plan.table_shape().expect("shape");
        let rows = shape.rows() as usize;
        let n = shape.code.n() as usize;
        let tail_len = shape.code.r as usize;
        let answers = shape.answers as usize;
        let pads_len = E_DEGREE * answers;
        let flat_tails = random_fps(rows.saturating_mul(tail_len));
        let tails = (0..rows)
            .map(|row| flat_tails[row * tail_len..(row + 1) * tail_len].to_vec())
            .collect();
        let salts = (0..n)
            .map(|_| random_bytes(shape.salt_bytes))
            .collect();
        let flat_pads = random_fps(rows.saturating_mul(pads_len));
        let pads = (0..rows)
            .map(|row| flat_pads[row * pads_len..(row + 1) * pads_len].to_vec())
            .collect();
        let key_randomness = LmHeadKeyRandomness {
            tails,
            salts,
            pads,
        };
        let private_dir = std::env::temp_dir().join("zkie-real-lm-head");
        std::fs::create_dir_all(&private_dir).expect("private dir");
        let session_seed = random_digest();
        let max_temp = 4u64 << 30;
        let mut store = zkie_engine::storage::encoded_store::FileRowStore::new(
            &private_dir,
            session_seed,
            shape.rows(),
            shape.code.n(),
            configuration,
            max_temp,
        )
        .expect("store");
        let cap_bytes = 1_700_000_000u64;
        let max_commit_bytes = 2u64 << 30;
        let commit_start = Instant::now();
        let holder = build_lm_head_holder(
            &context,
            &request_plan,
            configuration,
            &key_randomness,
            &mut store,
            4096,
            max_commit_bytes,
            cap_bytes,
            1 << 20,
            1 << 30,
        )
        .expect("holder");
        let commit_wall = commit_start.elapsed();
        let commit_hwm = vm_hwm_kb();
        let store_written = store.bytes_written();
        let store_read = store.bytes_read();
        let store_len = store.file_len().expect("file len");
        store.cleanup().expect("cleanup after commit");
        drop(key_randomness);
        mem("commit-keys-dropped");
        let masked = LimitedKeyStatement {
            version: LIMITED_KEY_VERSION,
            model_commitment: context.model_id,
            table_root: holder.reader().root,
            table_configuration: holder.reader().configuration,
            table_shape: shape,
            matrix_rows: 1 << request_plan.table.v,
            k_lo: request_plan.lin.k_lo,
            k_col: request_plan.lin.k_col,
            valid_min: i32::MIN as i64,
            valid_max: i32::MAX as i64,
            no_wf: true,
            trusted_i32_mapping: true,
        };
        masked.validate().expect("statement");
        // The trusted publisher is a separate operator step: after commit, the
        // identity is computed from the committed public reader and record and
        // an updated policy copy is written. Prove and verify consume that
        // copy; the table root is never accepted without it.
        let committed_identity = crate::s7_context::committed_limited_table_identity(
            &context,
            holder.reader(),
            &masked,
        )
        .expect("committed identity");
        let base_policy = std::fs::read(root.join("models/gpt2/case.policy")).expect("policy");
        let document = crate::policy::parse(&base_policy).expect("policy document");
        let document = crate::policy::bind_limited_table_identity(
            &document,
            &context.policy.case_id,
            committed_identity,
        )
        .expect("bound policy");
        let operator_dir = std::env::temp_dir().join("zkie-real-lm-head-policy");
        std::fs::create_dir_all(&operator_dir).expect("operator dir");
        let operator_policy = operator_dir.join("case.policy");
        crate::policy::write_document(&operator_policy, &document).expect("write policy");
        let operator_context = crate::s7_context::load_gpt2_lm_head_context(
            &case,
            &operator_policy,
            &public,
            &statement_path,
            statement_data.model_commitment,
        )
        .expect("operator context");
        let trusted_identity = operator_context.limited_table_identity.clone();
        // A policy whose old side does not match the public anchor and asset is
        // rejected when the trusted context is loaded.
        let case_position = document
            .case
            .iter()
            .position(|entry| entry.case_id == context.policy.case_id)
            .expect("selected case");
        let mut asset_swap = document.clone();
        asset_swap.case[case_position]
            .limited_table_identity
            .as_mut()
            .expect("identity")
            .asset_id += 1;
        let mut version_swap = document.clone();
        version_swap.model_version = zkie_models_gpt2::plan::MODEL_VERSION + 1;
        for entry in &mut version_swap.case {
            if let Some(identity) = &mut entry.limited_table_identity {
                identity.model_version = zkie_models_gpt2::plan::MODEL_VERSION + 1;
            }
        }
        for (name, tampered) in [("asset", asset_swap), ("version", version_swap)] {
            let path = operator_dir.join(format!("tampered-{name}.policy"));
            crate::policy::write_document(&path, &tampered).expect("tampered policy");
            assert!(crate::s7_context::load_gpt2_lm_head_context(
                &case,
                &path,
                &public,
                &statement_path,
                statement_data.model_commitment,
            )
            .is_err());
        }
        let fold_mask_columns = 1u64 << compiled.once.folded.mask.log_n;
        let mask_randomness = LmHeadMaskRandomness {
            seed: random_digest(),
            salts: (0..fold_mask_columns)
                .map(|_| random_bytes(request_plan.salt_bytes))
                .collect(),
        };
        let random_digest_bytes = random_digest();
        let session = build_limited_shared_session_for_context(
            &context,
            &request_plan,
            &compiled,
            &masked,
            holder.reader(),
            random_digest_bytes,
            &mask_randomness,
        )
        .expect("session");
        let verify_reader = holder.reader().clone();
        let gu_shape = session
            .plan
            .gu
            .as_ref()
            .expect("gu shape")
            .clone();
        let main_rows = gu_shape.u.main_rows as usize;
        let gu_tail_len = gu_shape.u.code.r as usize;
        let gu_randomness = LmHeadGuRandomness {
            code_tails: (0..main_rows)
                .map(|_| (0..gu_tail_len).map(|_| random_fp3()).collect())
                .collect(),
            salts: (0..gu_shape.tree.columns())
                .map(|_| random_bytes(gu_shape.tree.salt_bytes))
                .collect(),
            pads: (0..gu_sample_pad_offsets(&gu_shape)
                .expect("gu pads")
                .0
                .len())
                .map(|_| random_fp3())
                .collect(),
        };
        let omega_shape = session
            .plan
            .omega
            .as_ref()
            .expect("omega")
            .clone();
        let omega_randomness = LmHeadOmegaRandomness {
            code_tails: vec![
                (0..omega_shape.u.code.r as usize)
                    .map(|_| random_fp3())
                    .collect(),
            ],
            salts: (0..omega_shape.tree.columns())
                .map(|_| random_bytes(omega_shape.tree.salt_bytes))
                .collect(),
            pad: random_fp3(),
        };
        let fold = session.plan.fold;
        let old_mask_tree = MaskTreePlan {
            tree_number: 0x600,
            leaf_class: compiled.once_foldeq.shared_split_leaf_class,
            log_n: fold.mask_code.log_n,
            mask_rows: fold.mask_rows().expect("mask rows"),
            salt_bytes: request_plan.salt_bytes,
            run: session.base.plan.run,
            configuration,
        };
        if let Some(fault) = old_mask_tree.fault() {
            panic!("old mask tree: {fault}");
        }
        let old_rows_count = old_mask_tree.mask_rows as usize;
        let old_mask_randomness = LmHeadOldMaskRandomness {
            rows: (0..old_rows_count)
                .map(|_| (0..fold.mask_code.ell as usize).map(|_| random_fp3()).collect())
                .collect(),
            tails: (0..old_rows_count)
                .map(|_| (0..fold.mask_code.r as usize).map(|_| random_fp3()).collect())
                .collect(),
            salts: (0..old_mask_tree.columns())
                .map(|_| random_bytes(old_mask_tree.salt_bytes))
                .collect(),
        };
        let old_mask = commit_lm_head_old_mask(&fold, old_mask_tree, &old_mask_randomness)
            .expect("old mask");
        let master = random_digest();
        let switches = plan_stage_chain(
            &compiled,
            request_plan.samples as u32,
            configuration,
            session.base.plan.run,
            master,
            &session.base,
        )
        .expect("stage chain");
        let verify_plans = LimitedSwitchPlan::from_inputs(switches.as_ref());
        let old_mask_plan = old_mask.tree_plan();
        let old_mask_root = old_mask.root();
        let mut source = HolderSharedSource::new(holder, gu_randomness);
        let prove_start = Instant::now();
        let result = run_limited_shared_to_columns(
            session,
            &mut source,
            Some(&omega_randomness),
            Some(&old_mask),
            switches,
        )
        .expect("real chain");
        let prove_wall = prove_start.elapsed();
        let prove_hwm = vm_hwm_kb();
        // A second public-only setup is used by the independent fold replay;
        // the private mask provider it carries is never read by the verifier.
        let verify_setup = build_limited_shared_session_for_context(
            &context,
            &request_plan,
            &compiled,
            &masked,
            &verify_reader,
            random_digest_bytes,
            &mask_randomness,
        )
        .expect("verify session")
        .verify_setup();
        let artifact = LimitedChainArtifact::from_record(&masked, &result.record)
            .expect("artifact");
        let artifact_bytes = artifact.encode();
        let unique_count = unique_sorted(&result.queries).len();
        let symbol_bytes: usize = artifact
            .column_openings
            .iter()
            .map(|opening| 8 * opening.symbols.len())
            .sum();
        let path_bytes: usize = artifact
            .column_openings
            .iter()
            .map(|opening| 32 * opening.path.len())
            .sum();
        let opening_bytes: usize = artifact
            .column_openings
            .iter()
            .map(|opening| {
                8 + 8
                    + opening.salt.len()
                    + 8
                    + 8 * opening.symbols.len()
                    + 8
                    + 32 * opening.path.len()
            })
            .sum();
        eprintln!(
            "artifact breakdown: total {} openings {} unique {} opening_bytes {} symbol_bytes {} path_bytes {} base {} switches {}",
            artifact_bytes.len(),
            artifact.column_openings.len(),
            unique_count,
            opening_bytes,
            symbol_bytes,
            path_bytes,
            artifact.base_proof.len(),
            artifact.switches.len(),
        );
        let _ = std::fs::write(
            std::env::temp_dir().join("zkie-lm-head-artifact.bin"),
            &artifact_bytes,
        );
        let decoded = LimitedChainArtifact::decode(&artifact_bytes).expect("artifact decode");
        assert_eq!(verify_setup.column_work_bits, request_plan.work.once.positions);
        assert_eq!(verify_setup.switch_work_bits, request_plan.work.shared.positions);
        assert!(verify_setup.column_work_bits > 0 && verify_setup.switch_work_bits > 0);
        let verified_seal = verify_limited_chain(
            &verify_setup,
            &trusted_identity,
            &masked,
            &verify_plans,
            Some(old_mask_plan),
            Some(old_mask_root),
            &decoded,
        )
        .expect("independent verify");
        let prover_fold = result.fold_output.as_ref().expect("fold output");
        assert_eq!(prover_fold.fresh_root, decoded.fold_start.root);
        assert_eq!(verified_seal, decoded.sealed);

        let verify = |artifact: &LimitedChainArtifact| {
            verify_limited_chain(
                &verify_setup,
                &trusted_identity,
                &masked,
                &verify_plans,
                Some(old_mask_plan.clone()),
                Some(old_mask_root),
                artifact,
            )
        };
        let mut wrong_target = verify_plans.clone();
        wrong_target[0].target = verify_setup.switches[1].clone();
        assert!(verify_limited_chain(
            &verify_setup,
            &trusted_identity,
            &masked,
            &wrong_target,
            Some(old_mask_plan.clone()),
            Some(old_mask_root),
            &decoded,
        )
        .is_err());
        let reject_public_plan = |edit: &dyn Fn(&mut LimitedSwitchPlan)| {
            let mut plans = verify_plans.clone();
            edit(&mut plans[0]);
            assert!(verify_limited_chain(
                &verify_setup,
                &trusted_identity,
                &masked,
                &plans,
                Some(old_mask_plan.clone()),
                Some(old_mask_root),
                &decoded,
            )
            .is_err());
        };
        reject_public_plan(&|plan| plan.plan.log_n += 1);
        reject_public_plan(&|plan| plan.plan.leaf_class += 1);
        reject_public_plan(&|plan| plan.plan.tree_number ^= 1);
        reject_public_plan(&|plan| plan.plan.salt_bytes += 8);
        reject_public_plan(&|plan| plan.plan.parts[0].main_symbols += E_DEGREE as u64);
        reject_public_plan(&|plan| {
            plan.stage_fold
                .as_mut()
                .expect("stage fold")
                .mask_tree
                .log_n += 1
        });
        reject_public_plan(&|plan| {
            plan.stage_fold
                .as_mut()
                .expect("stage fold")
                .mask_tree
                .tree_number ^= 1
        });
        let mut wrong_statement = decoded.clone();
        wrong_statement.statement.table_root = Digest([0x77; 32]);
        assert!(verify(&wrong_statement).is_err());
        // The policy identity pins the committed table. Replacing its root,
        // record digest, configuration, shape, mapping, protocol identity or
        // model id without an operator policy update rejects.
        let verify_with = |artifact: &LimitedChainArtifact, identity: &LimitedTableIdentity| {
            verify_limited_chain(
                &verify_setup,
                identity,
                &masked,
                &verify_plans,
                Some(old_mask_plan.clone()),
                Some(old_mask_root),
                artifact,
            )
        };
        let mut wrong_root = trusted_identity.clone();
        wrong_root.table_root = Digest([0x71; 32]);
        assert!(verify_with(&decoded, &wrong_root).is_err());
        let mut wrong_record = trusted_identity.clone();
        wrong_record.table_record_digest = Digest([0x72; 32]);
        assert!(verify_with(&decoded, &wrong_record).is_err());
        let mut wrong_configuration = trusted_identity.clone();
        wrong_configuration.table_configuration = Digest([0x73; 32]);
        assert!(verify_with(&decoded, &wrong_configuration).is_err());
        let mut wrong_shape = trusted_identity.clone();
        wrong_shape.code_log_n += 1;
        assert!(verify_with(&decoded, &wrong_shape).is_err());
        let mut wrong_mapping = trusted_identity.clone();
        wrong_mapping.k_lo += 1;
        assert!(verify_with(&decoded, &wrong_mapping).is_err());
        let mut wrong_protocol = trusted_identity.clone();
        wrong_protocol.protocol_identity = Digest([0x74; 32]);
        assert!(verify_with(&decoded, &wrong_protocol).is_err());
        let mut wrong_model_id = trusted_identity.clone();
        wrong_model_id.model_id = Digest([0x75; 32]);
        assert!(verify_with(&decoded, &wrong_model_id).is_err());
        // The random table this test commits is not the table the pre-commit
        // policy identity pins; without the operator update it rejects.
        assert!(verify_with(&decoded, &context.limited_table_identity).is_err());
        let mut wrong_gu_root = decoded.clone();
        wrong_gu_root.gu_root = Digest([0x78; 32]);
        assert!(verify(&wrong_gu_root).is_err());
        let mut wrong_omega_root = decoded.clone();
        wrong_omega_root.omega_root = Some(Digest([0x79; 32]));
        assert!(verify(&wrong_omega_root).is_err());
        let mut wrong_column = decoded.clone();
        wrong_column.column_openings[0].path[0] = Digest([0x7a; 32]);
        assert!(verify(&wrong_column).is_err());
        let mut wrong_round = decoded.clone();
        wrong_round.fold_rounds[0].0 = wrong_round.fold_rounds[0].0.add(&Fp3::ONE);
        assert!(verify(&wrong_round).is_err());
        let mut wrong_identity = decoded.clone();
        wrong_identity.switches[0].identity = Digest([0x7b; 32]);
        assert!(verify(&wrong_identity).is_err());
        let mut wrong_declared = decoded.clone();
        wrong_declared.switches[0].declared_root = Digest([0x7c; 32]);
        assert!(verify(&wrong_declared).is_err());
        let mut wrong_member = decoded.clone();
        wrong_member.switches[0].member_openings[0].columns[0].symbols[0] =
            wrong_member.switches[0].member_openings[0].columns[0].symbols[0] + Fp::ONE;
        assert!(verify(&wrong_member).is_err());
        let mut wrong_column_nonce = decoded.clone();
        wrong_column_nonce.column_nonce ^= 1;
        assert!(verify(&wrong_column_nonce).is_err());
        let mut wrong_switch_nonce = decoded.clone();
        wrong_switch_nonce.switches[0].nonce ^= 1;
        assert!(verify(&wrong_switch_nonce).is_err());
        let mut wrong_order = decoded.clone();
        wrong_order.switches.swap(0, 1);
        assert!(verify(&wrong_order).is_err());
        let mut missing_switch = decoded.clone();
        missing_switch.switches.pop();
        assert!(verify(&missing_switch).is_err());
        let mut missing_stage = decoded.clone();
        missing_stage.switches[0].stage_fold = None;
        assert!(verify(&missing_stage).is_err());
        let mut missing_answer = decoded.clone();
        missing_answer.gu_answers.clear();
        assert!(verify(&missing_answer).is_err());
        let mut wrong_answer = decoded.clone();
        wrong_answer.gu_answers[0] = wrong_answer.gu_answers[0].add(&Fp3::ONE);
        assert!(verify(&wrong_answer).is_err());
        let mut over_budget = decoded.clone();
        over_budget.column_openings.push(ColumnOpening {
            position: u64::MAX,
            salt: Vec::new(),
            symbols: Vec::new(),
            path: Vec::new(),
        });
        assert!(verify(&over_budget).is_err());

        let mut version = decoded.encode();
        version[0] ^= 1;
        assert!(LimitedChainArtifact::decode(&version).is_err());
        let mut trailing = decoded.encode();
        trailing.push(0);
        assert!(LimitedChainArtifact::decode(&trailing).is_err());
        let mut truncated = decoded.encode();
        truncated.truncate(truncated.len() - 8);
        assert!(LimitedChainArtifact::decode(&truncated).is_err());
        let mut wrong_statement = decoded.clone();
        wrong_statement.statement.model_commitment = Digest([0xee; 32]);
        assert!(verify_limited_fold(&verify_setup, &masked, &wrong_statement).is_err());
        let mut wrong_opening = decoded.clone();
        wrong_opening.column_openings[0].symbols[0] =
            wrong_opening.column_openings[0].symbols[0] + Fp::ONE;
        assert!(verify_limited_fold(&verify_setup, &masked, &wrong_opening).is_err());
        let mut missing_opening = decoded.clone();
        missing_opening.column_openings.pop();
        assert!(verify_limited_fold(&verify_setup, &masked, &missing_opening).is_err());
        let sealed = result.sealed.expect("sealed");
        let proof = result.base_proof.expect("base proof");
        eprintln!(
            "real lm head: commit {:?} prove {:?} hwm_commit_kb {} hwm_prove_kb {} store_written {} store_read {} store_len {} base_proof {} artifact_bytes {} sealed {:?}",
            commit_wall,
            prove_wall,
            commit_hwm,
            prove_hwm,
            store_written,
            store_read,
            store_len,
            proof.len(),
            artifact_bytes.len(),
            sealed
        );
        assert_ne!(sealed, Digest::ZERO);
        assert!(!proof.is_empty());
        store.cleanup().expect("cleanup");
    }

    #[test]
    fn the_planned_stage_chain_follows_the_compiled_stages() {
        let request = zkie_models_gpt2::lm_head_plan::lm_head_plan().expect("request plan");
        let compiled =
            zkie_models_gpt2::lm_head_plan::lm_head_compiled().expect("compiled plan");
        let answers = u32::try_from(request.samples).expect("answers");
        let configuration = Digest([0x11; 32]);
        let run = Digest([0x22; 32]);
        let master = Digest([0x33; 32]);
        let last = compiled
            .once
            .folded
            .stages
            .last()
            .expect("final stage");
        let base = BaseStarBinding {
            plan: RuntimeTreePlan {
                tree_number: compiled.once_foldeq.base_tree_number,
                leaf_class: compiled.once_foldeq.base_leaf_class,
                log_n: last.main.log_n,
                log_n_zk: last.mask.log_n,
                parts: vec![RuntimeTreePart {
                    main_symbols: 3,
                    mask_symbols: 3,
                }],
                salt_bytes: compiled.once_foldeq.base_salt_bytes,
                run,
                configuration,
            },
            positions: usize::try_from(last.positions).expect("positions"),
        };
        let chain = plan_stage_chain(&compiled, answers, configuration, run, master, &base)
            .expect("stage chain")
            .expect("some chain");
        let stages = &compiled.once.folded.stages;
        let mut step = Some(&chain);
        let mut index = 0usize;
        while let Some(current) = step {
            let stage = &stages[index];
            let target = SwitchTarget::from_stage(
                stage.main,
                stage.mask,
                stage.k,
                stage.h,
                answers,
            )
            .expect("target");
            assert_eq!(current.target, target);
            assert_eq!(current.main_tails.len(), target.main_rows as usize);
            assert_eq!(current.mask_tails.len(), target.mask_rows as usize);
            assert_eq!(current.salts.len(), current.plan.columns() as usize);
            assert!(current
                .main_tails
                .iter()
                .all(|tail| tail.len() == target.code.r as usize));
            assert!(current
                .mask_tails
                .iter()
                .all(|tail| tail.len() == target.mask_code.r as usize));
            if index + 1 < stages.len() {
                let fold = current.stage_fold.as_ref().expect("fold");
                assert_eq!(fold.shape.code, target.code);
                assert_eq!(fold.shape.mask_code, target.mask_code);
                assert_eq!(fold.shape.k, target.main.bits);
                assert_eq!(fold.shape.h, target.rest.bits);
                assert_eq!(fold.layout.members[0].main_rows, target.main_rows);
                assert_eq!(fold.layout.members[0].mask_rows, target.mask_rows);
                assert_eq!(fold.randomness.salts.len(), fold.mask_tree.columns() as usize);
            } else {
                assert!(current.stage_fold.is_none());
                let base_step = current.base.as_ref().expect("base step");
                assert_eq!(base_step.salts.len(), base.plan.columns() as usize);
            }
            index += 1;
            step = current.next.as_deref();
        }
        assert_eq!(index, stages.len());

        let other = plan_stage_chain(&compiled, answers, configuration, run, master, &base)
            .expect("stage chain")
            .expect("some chain");
        assert_eq!(other.target, chain.target);
        assert_eq!(other.main_tails, chain.main_tails);
        assert_eq!(other.salts, chain.salts);
        let changed = plan_stage_chain(
            &compiled,
            answers,
            configuration,
            run,
            Digest([0x34; 32]),
            &base,
        )
        .expect("stage chain")
        .expect("some chain");
        assert_ne!(changed.main_tails, chain.main_tails);
        assert_ne!(changed.salts, chain.salts);
    }

    #[test]
    fn the_trusted_public_switch_plans_reject_edited_handoff_fields() {
        let request = zkie_models_gpt2::lm_head_plan::lm_head_plan().expect("request plan");
        let compiled =
            zkie_models_gpt2::lm_head_plan::lm_head_compiled().expect("compiled plan");
        let answers = u32::try_from(request.samples).expect("answers");
        let configuration = Digest([0x11; 32]);
        let run = Digest([0x22; 32]);
        let master = Digest([0x33; 32]);
        let last = compiled
            .once
            .folded
            .stages
            .last()
            .expect("final stage");
        let base = BaseStarBinding {
            plan: RuntimeTreePlan {
                tree_number: compiled.once_foldeq.base_tree_number,
                leaf_class: compiled.once_foldeq.base_leaf_class,
                log_n: last.main.log_n,
                log_n_zk: last.mask.log_n,
                parts: vec![RuntimeTreePart {
                    main_symbols: E_DEGREE as u64,
                    mask_symbols: E_DEGREE as u64,
                }],
                salt_bytes: compiled.once_foldeq.base_salt_bytes,
                run,
                configuration,
            },
            positions: usize::try_from(last.positions).expect("positions"),
        };
        let chain = plan_stage_chain(&compiled, answers, configuration, run, master, &base)
            .expect("stage chain")
            .expect("some chain");
        let plans = LimitedSwitchPlan::from_inputs(Some(&chain));
        let targets: Vec<SwitchTarget> = plans.iter().map(|plan| plan.target.clone()).collect();
        let expected = expected_switch_public_plans(&compiled, &targets, configuration)
            .expect("expected plans");
        check_switch_public_plans(&expected, &plans).expect("matching plans");

        let mut edited = plans.clone();
        edited[0].positions += 1;
        assert!(check_switch_public_plans(&expected, &edited).is_err());

        let mut edited = plans.clone();
        edited[0].plan.log_n += 1;
        assert!(check_switch_public_plans(&expected, &edited).is_err());

        let mut edited = plans.clone();
        edited[0].plan.leaf_class += 1;
        assert!(check_switch_public_plans(&expected, &edited).is_err());

        let mut edited = plans.clone();
        edited[0].plan.tree_number ^= 1;
        assert!(check_switch_public_plans(&expected, &edited).is_err());

        let mut edited = plans.clone();
        edited[0].plan.salt_bytes += 8;
        assert!(check_switch_public_plans(&expected, &edited).is_err());

        let mut edited = plans.clone();
        edited[0].plan.parts[0].main_symbols += E_DEGREE as u64;
        assert!(check_switch_public_plans(&expected, &edited).is_err());

        let mut edited = plans.clone();
        edited[0]
            .stage_fold
            .as_mut()
            .expect("stage fold")
            .mask_tree
            .log_n += 1;
        assert!(check_switch_public_plans(&expected, &edited).is_err());

        let mut edited = plans.clone();
        edited[0]
            .stage_fold
            .as_mut()
            .expect("stage fold")
            .mask_tree
            .tree_number ^= 1;
        assert!(check_switch_public_plans(&expected, &edited).is_err());

        let mut edited = plans.clone();
        edited[0]
            .stage_fold
            .as_mut()
            .expect("stage fold")
            .shape
            .mask_leaf_class += 1;
        assert!(check_switch_public_plans(&expected, &edited).is_err());

        let mut edited = plans.clone();
        edited[0].stage_fold = None;
        assert!(check_switch_public_plans(&expected, &edited).is_err());
    }

    #[test]
    fn the_stage_sequence_rejects_missing_or_reordered_switches() {
        use zkie_engine::pcs::request::code::build_code_plan;
        let code = build_code_plan(5, 4, 1).expect("code");
        let mask_code = build_code_plan(5, 8, 1).expect("mask code");
        let first = SwitchTarget::from_stage(code, mask_code, 2, 2, 1).expect("first");
        let second = SwitchTarget::from_stage(code, mask_code, 1, 1, 1).expect("second");
        let last = SwitchTarget::from_stage(code, mask_code, 0, 0, 1).expect("last");
        assert!(check_stage_sequence(&[first, second, last], &[first, last]).is_err());
        assert!(check_stage_sequence(&[first, second, last], &[second, first, last]).is_err());
        assert!(check_stage_sequence(&[first, second, last], &[first, second]).is_err());
        assert!(check_stage_sequence(&[first, second, last], &[first, second, last]).is_ok());
    }

    #[test]
    fn the_last_stage_rejects_a_folded_final_target() {
        use zkie_engine::pcs::request::code::build_code_plan;
        use zkie_engine::pcs::request::cs_stage::SwitchClaim;
        let code = build_code_plan(4, 4, 1).expect("code");
        let mask_code = build_code_plan(4, 8, 1).expect("mask code");
        let target = SwitchTarget::from_stage(code, mask_code, 1, 0, 1).expect("target");
        let claims = SwitchedClaims {
            positions: Vec::new(),
            old_symbols: Vec::new(),
            claims: Vec::new(),
            theta: Fp3::ZERO,
            combined: SwitchClaim {
                main: vec![Fp3::ZERO; target.code.ell as usize],
                mask: vec![Fp3::ZERO; target.mask_code.ell as usize],
                target: Fp3::ZERO,
            },
            main_len: target.code.ell,
            main_tail_len: target.code.r,
            mask_len: target.mask_code.ell,
            mask_tail_len: target.mask_code.r,
        };
        let message = SwitchedRows {
            main: vec![Fp3::ZERO; target.main_rows as usize * target.code.ell as usize],
            main_tails: vec![vec![Fp3::ZERO; target.code.r as usize]; target.main_rows as usize],
            mask: vec![Fp3::ZERO; target.mask_rows as usize * target.mask_code.ell as usize],
            mask_tails: vec![vec![Fp3::ZERO; target.mask_code.r as usize]; target.mask_rows as usize],
        };
        assert!(matches!(
            last_stage_single_rows(target, &claims, &message),
            Err(Error::UnsupportedPrimitive)
        ));
    }

    struct VecSource {
        messages: Vec<Vec<Fp>>,
        tails: Vec<Vec<Fp>>,
    }

    impl TableRowSource for VecSource {
        fn rows(&self) -> u64 {
            self.messages.len() as u64
        }

        fn row(&mut self, row: u64, message: &mut [Fp], tail: &mut [Fp]) -> Result<()> {
            let message_row = self
                .messages
                .get(row as usize)
                .ok_or(Error::InvalidIndex)?;
            let tail_row = self.tails.get(row as usize).ok_or(Error::InvalidIndex)?;
            if message.len() != message_row.len() || tail.len() != tail_row.len() {
                return Err(Error::ShapeMismatch);
            }
            message.copy_from_slice(message_row);
            tail.copy_from_slice(tail_row);
            Ok(())
        }
    }

    #[test]
    fn the_holder_fold_matches_the_key_fold() {
        let (key, gu_shape, randomness, tails, _parameters) = small_gu_fixture();
        let gamma = vec![Fp3::from_u64(2), Fp3::from_u64(3)];
        let from_key = fold_lm_head_gu(&key, &tails, &gu_shape, &gamma, &randomness)
            .expect("key fold");
        let salts: Vec<Vec<u8>> = (0..16)
            .map(|position| vec![(position as u8) ^ 0x5a; 40])
            .collect();
        let mut source = VecSource {
            messages: key.messages.clone(),
            tails: tails.clone(),
        };
        let table = zkie_engine::pcs::request::table::commit_table_streaming(
            &mut source,
            key.shape,
            key.reader.configuration,
            &salts,
            4,
            u64::MAX,
        )
        .expect("streaming table");
        assert_eq!(table.reader.root, key.reader.root);
        let mut holder = StreamingHolder::new(table, source, 8).expect("holder");
        holder.begin(1).expect("begin");
        let from_holder =
            fold_lm_head_gu_holder(&mut holder, &gu_shape, &gamma, &randomness)
                .expect("holder fold");
        assert_eq!(from_key.root(), from_holder.root());
        assert_eq!(from_key.folded(), from_holder.folded());
        assert_eq!(
            from_key.relation().main,
            from_holder.relation().main
        );
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

#[cfg(test)]
mod stage_tests {
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    use serde_json::json;

    use super::*;
    use crate::lm_head_stage::{
        claim_budget, initialize_budget, read_private, read_public, read_public_file,
        write_private, write_public, write_public_file, LmHeadPrivateState, LmHeadTableState,
    };
    use crate::s7_context::load_gpt2_lm_head_context;
    use zkie_engine::pcs::request::field::{Fp, Fp3};
    use zkie_engine::pcs::request::runtime_tree::MaskTreePlan;
    use zkie_engine::pcs::request::relation::gu_sample_pad_offsets;
    use zkie_engine::pcs::request::table::{EncodedRowStore, StreamingTable};
    use zkie_engine::storage::encoded_store::FileRowStore;
    use zkie_ops::proof::base_last_session::LastBaseRandomness;

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn random_bytes(count: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; count];
        getrandom::fill(&mut bytes).expect("entropy");
        bytes
    }

    fn random_digest() -> Digest {
        let mut bytes = [0u8; 32];
        getrandom::fill(&mut bytes).expect("entropy");
        Digest(bytes)
    }

    fn random_fp() -> Fp {
        loop {
            let mut bytes = [0u8; 8];
            getrandom::fill(&mut bytes).expect("entropy");
            let word = u64::from_le_bytes(bytes);
            if word < zkie_engine::pcs::request::field::MODULUS {
                return Fp::from_u64(word);
            }
        }
    }

    fn random_fp3() -> Fp3 {
        Fp3::new(random_fp(), random_fp(), random_fp())
    }

    fn random_fps(count: usize) -> Vec<Fp> {
        (0..count).map(|_| random_fp()).collect()
    }

    fn vm_hwm_kb() -> u64 {
        let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
            return 0;
        };
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                return rest
                    .split_whitespace()
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(0);
            }
        }
        0
    }

    fn io_bytes() -> (u64, u64) {
        let Ok(text) = std::fs::read_to_string("/proc/self/io") else {
            return (0, 0);
        };
        let mut read = 0;
        let mut written = 0;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("read_bytes:") {
                read = rest.trim().parse().unwrap_or(0);
            } else if let Some(rest) = line.strip_prefix("write_bytes:") {
                written = rest.trim().parse().unwrap_or(0);
            }
        }
        (read, written)
    }

    fn stage_dir() -> PathBuf {
        PathBuf::from(std::env::var("ZKIE_LM_HEAD_DIR").expect("stage dir"))
    }

    struct Inputs {
        context: LmHeadContext,
        request_plan: RequestPlan,
        compiled: CompiledPlan,
        configuration: Digest,
    }

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("repository root")
            .to_path_buf()
    }

    /// The pre-commit policy. Its limited table identity pins the operator's
    /// provisioning commit, not the table this run is about to commit.
    fn base_policy_path() -> PathBuf {
        repo_root().join("models/gpt2/case.policy")
    }

    /// The trusted policy copy the operator writes after commit.
    fn operator_policy_path(dir: &Path) -> PathBuf {
        dir.join("policy").join("case.policy")
    }

    fn load_inputs(policy_path: &Path) -> Result<Inputs> {
        let root = repo_root();
        let case = root.join("models/gpt2/case.toml");
        let public = root.join(
            "docs/benchmarks/20261110-0547-verify-bytes-109934bd-670a98c9/prod/gpt2-seq16/commit/public",
        );
        let statement_path = root.join(
            "docs/benchmarks/20261110-0547-verify-bytes-109934bd-670a98c9/prod/gpt2-seq16/proof/statement.bin",
        );
        let statement_bytes = std::fs::read(&statement_path).map_err(|_| Error::Io)?;
        let statement_data: crate::provider::StatementData =
            bincode::deserialize(&statement_bytes).map_err(|_| Error::InvalidEncoding)?;
        let context = load_gpt2_lm_head_context(
            &case,
            &policy_path,
            &public,
            &statement_path,
            statement_data.model_commitment,
        )?;
        let request_plan = zkie_models_gpt2::lm_head_plan::lm_head_plan()?;
        let compiled = zkie_models_gpt2::lm_head_plan::lm_head_compiled()?;
        request_plan.validate()?;
        request_plan.validate_compiled(&compiled)?;
        let parameters =
            limited_key_binding_parameters(&request_plan, context.plan_identifier)?;
        let configuration = parameters.digest()?;
        Ok(Inputs {
            context,
            request_plan,
            compiled,
            configuration,
        })
    }

    fn masked_statement(inputs: &Inputs, reader: &IndexReader) -> LimitedKeyStatement {
        LimitedKeyStatement {
            version: LIMITED_KEY_VERSION,
            model_commitment: inputs.context.model_id,
            table_root: reader.root,
            table_configuration: reader.configuration,
            table_shape: reader.shape,
            matrix_rows: 1 << inputs.request_plan.table.v,
            k_lo: inputs.request_plan.lin.k_lo,
            k_col: inputs.request_plan.lin.k_col,
            valid_min: i32::MIN as i64,
            valid_max: i32::MAX as i64,
            no_wf: true,
            trusted_i32_mapping: true,
        }
    }

    fn random_key(shape: IndexShape) -> LmHeadKeyRandomness {
        let rows = shape.rows() as usize;
        let n = shape.code.n() as usize;
        let tail_len = shape.code.r as usize;
        let pads_len = E_DEGREE * shape.answers as usize;
        LmHeadKeyRandomness {
            tails: (0..rows).map(|_| random_fps(tail_len)).collect(),
            salts: (0..n).map(|_| random_bytes(shape.salt_bytes)).collect(),
            pads: (0..rows).map(|_| random_fps(pads_len)).collect(),
        }
    }

    fn random_gu(shape: &GuShape, salt_bytes: usize) -> Result<LmHeadGuRandomness> {
        let tail_len = shape.u.code.r as usize;
        Ok(LmHeadGuRandomness {
            code_tails: (0..shape.u.main_rows)
                .map(|_| (0..tail_len).map(|_| random_fp3()).collect())
                .collect(),
            salts: (0..shape.tree.columns())
                .map(|_| random_bytes(salt_bytes))
                .collect(),
            pads: (0..gu_sample_pad_offsets(shape)?.0.len())
                .map(|_| random_fp3())
                .collect(),
        })
    }

    fn random_omega(plan: &OmegaShapePlan, salt_bytes: usize) -> LmHeadOmegaRandomness {
        LmHeadOmegaRandomness {
            code_tails: vec![
                (0..plan.u.code.r as usize)
                    .map(|_| random_fp3())
                    .collect(),
            ],
            salts: (0..plan.tree.columns())
                .map(|_| random_bytes(salt_bytes))
                .collect(),
            pad: random_fp3(),
        }
    }

    fn old_mask_plan(
        inputs: &Inputs,
        public_plan: &LimitedPublicPlan,
    ) -> MaskTreePlan {
        MaskTreePlan {
            tree_number: 0x600,
            leaf_class: inputs.compiled.once_foldeq.shared_split_leaf_class,
            log_n: public_plan.fold.mask_code.log_n,
            mask_rows: public_plan.fold.mask_rows().expect("mask rows"),
            salt_bytes: inputs.request_plan.salt_bytes,
            run: public_plan.base.plan.run,
            configuration: inputs.configuration,
        }
    }

    fn random_old_mask(
        plan: &MaskTreePlan,
        fold: FoldShape,
        salt_bytes: usize,
    ) -> LmHeadOldMaskRandomness {
        LmHeadOldMaskRandomness {
            rows: (0..plan.mask_rows)
                .map(|_| (0..fold.mask_code.ell as usize).map(|_| random_fp3()).collect())
                .collect(),
            tails: (0..plan.mask_rows)
                .map(|_| (0..fold.mask_code.r as usize).map(|_| random_fp3()).collect())
                .collect(),
            salts: (0..plan.columns())
                .map(|_| random_bytes(salt_bytes))
                .collect(),
        }
    }

    fn stage_json(
        wall: f64,
        hwm: u64,
        io: (u64, u64),
        temp: u64,
        extra: serde_json::Value,
    ) -> serde_json::Value {
        json!({
            "ok": true,
            "wall_s": wall,
            "hwm_kb": hwm,
            "io_read": io.0,
            "io_write": io.1,
            "temp_bytes": temp,
            "extra": extra,
        })
    }

    fn stage_commit(dir: &Path) -> Result<serde_json::Value> {
        let start = Instant::now();
        let inputs = load_inputs(&base_policy_path())?;
        let shape = inputs.request_plan.table_shape()?;
        let key = random_key(shape);
        let private_dir = dir.join("private");
        crate::lm_head_stage::prepare_private_dir(&private_dir)?;
        let store_dir = private_dir.join("commit-store");
        std::fs::create_dir_all(&store_dir).map_err(|_| Error::Io)?;
        let mut store = FileRowStore::new(
            &store_dir,
            random_digest(),
            shape.rows(),
            shape.code.n(),
            inputs.configuration,
            4u64 << 30,
        )?;
        let holder = build_lm_head_holder(
            &inputs.context,
            &inputs.request_plan,
            inputs.configuration,
            &key,
            &mut store,
            4096,
            2u64 << 30,
            1_700_000_000,
            1 << 20,
            1 << 30,
        )?;
        let reader = holder.reader().clone();
        let table = LmHeadTableState {
            reader: reader.clone(),
            name: holder.table().name.clone(),
            leaves: holder.table().leaves().to_vec(),
            salts: holder.table().salts().to_vec(),
            peak_bytes: holder.table().peak_bytes(),
        };
        let store_len = store.file_len()?;
        let store_written = store.bytes_written();
        let store_read = store.bytes_read();
        drop(holder);
        store.cleanup()?;
        let statement = masked_statement(&inputs, &reader);
        statement.validate()?;
        // The operator step: compute the trusted identity from the committed
        // public reader and record, then write the policy copy that prove and
        // verify consume. The table was not known before this commit.
        let committed_identity =
            crate::s7_context::committed_limited_table_identity(&inputs.context, &reader, &statement)?;
        let base_policy = std::fs::read(base_policy_path()).map_err(|_| Error::Io)?;
        let document = crate::policy::parse(&base_policy)?;
        let document = crate::policy::bind_limited_table_identity(
            &document,
            &inputs.context.policy.case_id,
            committed_identity,
        )?;
        let operator_policy = operator_policy_path(dir);
        crate::policy::write_document(&operator_policy, &document)?;
        let session_bytes = random_digest();
        let public_plan = build_limited_public_plan(
            &inputs.request_plan,
            &inputs.compiled,
            &statement,
            &reader,
            inputs.context.plan_identifier,
            inputs.context.policy.case_id.as_bytes(),
            session_bytes,
        )?;
        let gu = random_gu(&public_plan.gu, inputs.request_plan.salt_bytes)?;
        let omega = random_omega(&public_plan.omega, inputs.request_plan.salt_bytes);
        let mask = LmHeadMaskRandomness {
            seed: random_digest(),
            salts: (0..public_plan.fresh_plan.columns())
                .map(|_| random_bytes(inputs.request_plan.salt_bytes))
                .collect(),
        };
        let old_plan = old_mask_plan(&inputs, &public_plan);
        let old_mask = random_old_mask(&old_plan, public_plan.fold, inputs.request_plan.salt_bytes);
        let old_root = commit_lm_head_old_mask(&public_plan.fold, old_plan, &old_mask)?.root();
        let master = random_digest();
        let switches = plan_stage_chain(
            &inputs.compiled,
            inputs.request_plan.samples as u32,
            inputs.configuration,
            public_plan.base.plan.run,
            master,
            &public_plan.base,
        )?
        .ok_or(Error::InvalidConfig)?;
        let public_switches = LimitedSwitchPlan::from_inputs(Some(&switches));
        let handoff = crate::lm_head_stage::LmHeadPublicHandoff {
            version: crate::lm_head_stage::LM_HEAD_STAGE_VERSION,
            protocol_identity: zkie_ops::proof::session_baseline::SESSION_PROTOCOL_IDENTITY,
            statement: statement.encode(),
            reader: reader.clone(),
            plan_identifier: inputs.context.plan_identifier,
            case_id: inputs.context.policy.case_id.as_bytes().to_vec(),
            random_bytes: session_bytes,
            base: public_plan.base.clone(),
            old_mask: Some(crate::lm_head_stage::OldMaskPublic {
                plan: old_plan,
                root: old_root,
            }),
            fresh_plan: public_plan.fresh_plan,
            switches: public_switches,
        };
        handoff.validate()?;
        let digest = handoff.digest()?;
        write_public(&dir.join("public"), &handoff)?;
        let private = LmHeadPrivateState {
            version: crate::lm_head_stage::LM_HEAD_STAGE_VERSION,
            public_digest: digest,
            table,
            key,
            gu,
            omega,
            mask,
            old_mask,
            master,
            base_salts: vec![random_bytes(inputs.request_plan.salt_bytes)],
            base_randomness: LastBaseRandomness::Fixed(random_digest()),
        };
        write_private(&private_dir, &private)?;
        initialize_budget(&private_dir, digest, LM_HEAD_COLUMN_DRAWS as u64)?;
        let private_bytes = std::fs::metadata(private_dir.join("prove-state.bin"))
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        let wall = start.elapsed().as_secs_f64();
        Ok(stage_json(
            wall,
            vm_hwm_kb(),
            io_bytes(),
            store_len,
            json!({
                "store_written": store_written,
                "store_read": store_read,
                "store_len": store_len,
                "public_digest": format!("{digest:?}"),
                "switch_count": handoff.switches.len(),
                "private_state_bytes": private_bytes,
                "policy": operator_policy.to_string_lossy().into_owned(),
            }),
        ))
    }

    fn load_handoff(dir: &Path) -> Result<crate::lm_head_stage::LmHeadPublicHandoff> {
        read_public(&dir.join("public").join("public.bin"))
    }

    fn stage_prove(dir: &Path) -> Result<serde_json::Value> {
        let start = Instant::now();
        let handoff = load_handoff(dir)?;
        let digest = handoff.digest()?;
        let inputs = load_inputs(&operator_policy_path(dir))?;
        // Prove consumes the operator policy as a trusted input: the handoff's
        // reader and statement must be the table the policy pins.
        let statement = handoff.verify_trusted_identity(&inputs.context)?;
        if handoff.plan_identifier != inputs.context.plan_identifier
            || handoff.case_id != inputs.context.policy.case_id.as_bytes()
            || statement.model_commitment != inputs.context.model_id
        {
            return Err(Error::StatementMismatch);
        }
        let private_dir = dir.join("private");
        let private = read_private(&private_dir.join("prove-state.bin"), digest)?;
        let _claim = claim_budget(&private_dir, digest, LM_HEAD_COLUMN_DRAWS as u64)?;
        let public_plan = build_limited_public_plan(
            &inputs.request_plan,
            &inputs.compiled,
            &statement,
            &handoff.reader,
            inputs.context.plan_identifier,
            inputs.context.policy.case_id.as_bytes(),
            handoff.random_bytes,
        )?;
        if public_plan.base != handoff.base
            || public_plan.fresh_plan != handoff.fresh_plan
            || public_plan
                .switches
                .iter()
                .map(|target| target)
                .collect::<Vec<_>>()
                != handoff.switches.iter().map(|switch| &switch.target).collect::<Vec<_>>()
        {
            return Err(Error::StatementMismatch);
        }
        let setup_start = Instant::now();
        let table = StreamingTable::rebuild(
            private.table.reader.clone(),
            private.table.name.clone(),
            private.table.leaves.clone(),
            private.table.salts.clone(),
            private.table.peak_bytes,
        )?;
        if table.reader != handoff.reader {
            return Err(Error::StatementMismatch);
        }
        let source = LmHeadAssetSource::open(
            &inputs.context,
            &inputs.request_plan,
            handoff.reader.configuration,
            &private.key,
            1_700_000_000,
            1 << 20,
            1 << 30,
        )?;
        let holder = StreamingHolder::new(table, source, LM_HEAD_COLUMN_DRAWS as u64)?;
        let session = build_limited_shared_session_for_context(
            &inputs.context,
            &inputs.request_plan,
            &inputs.compiled,
            &statement,
            &handoff.reader,
            handoff.random_bytes,
            &private.mask,
        )?;
        let switches = plan_stage_chain(
            &inputs.compiled,
            inputs.request_plan.samples as u32,
            handoff.reader.configuration,
            public_plan.base.plan.run,
            private.master,
            &public_plan.base,
        )?
        .ok_or(Error::InvalidConfig)?;
        if LimitedSwitchPlan::from_inputs(Some(&switches)) != handoff.switches {
            return Err(Error::StatementMismatch);
        }
        let old_public = handoff.old_mask.as_ref().ok_or(Error::IncompleteArtifact)?;
        let old_mask = commit_lm_head_old_mask(&public_plan.fold, old_public.plan, &private.old_mask)?;
        if old_mask.root() != old_public.root {
            return Err(Error::StatementMismatch);
        }
        let setup_wall = setup_start.elapsed().as_secs_f64();
        let chain_start = Instant::now();
        let mut source = HolderSharedSource::new(holder, private.gu);
        let result = run_limited_shared_to_columns(
            session,
            &mut source,
            Some(&private.omega),
            Some(&old_mask),
            Some(switches),
        )?;
        let chain_wall = chain_start.elapsed().as_secs_f64();
        let artifact = LimitedChainArtifact::from_record(&statement, &result.record)?;
        let bytes = artifact.encode();
        write_public_file(&dir.join("proof"), "artifact.bin", &bytes)?;
        let sealed = result.sealed.ok_or(Error::IncompleteArtifact)?;
        // Logical draws are the trusted plan's per-switch counts; physical
        // openings are the unique positions actually carried by the artifact.
        let draws: Vec<serde_json::Value> = handoff
            .switches
            .iter()
            .zip(result.record.switches.iter())
            .map(|(plan, record)| {
                json!({
                    "positions": plan.positions,
                    "nonce": record.nonce,
                    "unique_main": record
                        .member_openings
                        .iter()
                        .map(|opening| opening.positions.len())
                        .max()
                        .unwrap_or(0),
                    "unique_mask": record.fresh_opening.positions.len(),
                })
            })
            .collect();
        let wall = start.elapsed().as_secs_f64();
        Ok(stage_json(
            wall,
            vm_hwm_kb(),
            io_bytes(),
            bytes.len() as u64,
            json!({
                "setup_wall_s": setup_wall,
                "chain_wall_s": chain_wall,
                "artifact_bytes": bytes.len(),
                "sealed": format!("{sealed:?}"),
                "column_nonce": result.record.column_nonce,
                "work_bits": {
                    "columns": inputs.request_plan.work.once.positions,
                    "switches": inputs.request_plan.work.shared.positions,
                },
                "draws": draws,
            }),
        ))
    }

    fn stage_verify(dir: &Path) -> Result<serde_json::Value> {
        let start = Instant::now();
        let handoff = load_handoff(dir)?;
        let bytes = read_public_file(&dir.join("proof").join("artifact.bin"))?;
        let artifact = LimitedChainArtifact::decode(&bytes)?;
        let inputs = load_inputs(&operator_policy_path(dir))?;
        // Verify consumes the operator policy as a trusted input. The proof's
        // own statement and roots cannot stand in for the policy identity.
        let statement = handoff.verify_trusted_identity(&inputs.context)?;
        if handoff.plan_identifier != inputs.context.plan_identifier
            || handoff.case_id != inputs.context.policy.case_id.as_bytes()
            || statement.model_commitment != inputs.context.model_id
        {
            return Err(Error::StatementMismatch);
        }
        let setup = build_limited_verify_setup(
            &inputs.request_plan,
            &inputs.compiled,
            &statement,
            &handoff.reader,
            inputs.context.plan_identifier,
            inputs.context.policy.case_id.as_bytes(),
            handoff.random_bytes,
            artifact.fold_start.root,
        )?;
        let old_public = handoff.old_mask.as_ref().ok_or(Error::IncompleteArtifact)?;
        let sealed = verify_limited_chain(
            &setup,
            &inputs.context.limited_table_identity,
            &statement,
            &handoff.switches,
            Some(old_public.plan),
            Some(old_public.root),
            &artifact,
        )?;
        if sealed != artifact.sealed {
            return Err(Error::InvalidProof);
        }
        let wall = start.elapsed().as_secs_f64();
        Ok(stage_json(
            wall,
            vm_hwm_kb(),
            io_bytes(),
            0,
            json!({
                "artifact_bytes": bytes.len(),
                "verdict": "accept",
                "sealed": format!("{sealed:?}"),
            }),
        ))
    }

    #[test]
    #[ignore]
    fn lm_head_stage_worker() {
        let Ok(stage) = std::env::var("ZKIE_LM_HEAD_STAGE") else {
            return;
        };
        let dir = stage_dir();
        let result = match stage.as_str() {
            "commit" => stage_commit(&dir),
            "prove" => stage_prove(&dir),
            "verify" => stage_verify(&dir),
            _ => Err(Error::UnsupportedPrimitive),
        };
        let stage_dir = dir.join(&stage);
        let _ = std::fs::create_dir_all(&stage_dir);
        let value = match result {
            Ok(value) => value,
            Err(error) => json!({"ok": false, "error": format!("{error:?}")}),
        };
        let _ = std::fs::write(
            stage_dir.join("result.json"),
            serde_json::to_vec_pretty(&value).unwrap_or_default(),
        );
        if !value.get("ok").and_then(|value| value.as_bool()).unwrap_or(false) {
            panic!("stage {stage} failed");
        }
    }

    fn run_stage(executable: &Path, dir: &Path, stage: &str) -> std::process::Output {
        Command::new(executable)
            .args([
                "--exact",
                "lm_head_key::stage_tests::lm_head_stage_worker",
                "--ignored",
                "--nocapture",
            ])
            .env("ZKIE_LM_HEAD_STAGE", stage)
            .env("ZKIE_LM_HEAD_DIR", dir)
            .output()
            .expect("spawn")
    }

    fn stage_result(dir: &Path, stage: &str) -> serde_json::Value {
        std::fs::read(dir.join(stage).join("result.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or(json!({"ok": false, "error": "missing result"}))
    }

    #[test]
    #[ignore]
    fn lm_head_stage_processes() {
        let dir = std::env::temp_dir().join(format!(
            "zkie-stage-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("dir");
        let executable = std::env::current_exe().expect("test executable");
        let mut positive = Vec::new();
        for stage in ["commit", "prove", "verify"] {
            let output = run_stage(&executable, &dir, stage);
            let value = stage_result(&dir, stage);
            eprintln!(
                "stage {stage}: {}",
                serde_json::to_string(&value).unwrap_or_default()
            );
            assert!(
                output.status.success(),
                "stage {stage} failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(value.get("ok").and_then(|v| v.as_bool()).unwrap_or(false));
            positive.push((stage, value));
        }

        // A new random table without an operator policy update is rejected: the
        // pre-commit policy still pins the provisioning table, not this commit.
        let operator_policy = operator_policy_path(&dir);
        let updated_policy = std::fs::read(&operator_policy).expect("operator policy");
        let stale_policy = std::fs::read(base_policy_path()).expect("base policy");
        std::fs::write(&operator_policy, &stale_policy).expect("stale policy");
        let output = run_stage(&executable, &dir, "verify");
        assert!(!output.status.success());
        std::fs::write(&operator_policy, &updated_policy).expect("restore policy");

        // A second prove must not reuse the spent budget.
        let output = run_stage(&executable, &dir, "prove");
        assert!(!output.status.success());
        assert!(!stage_result(&dir, "prove")
            .get("ok")
            .and_then(|v| v.as_bool())
            .unwrap_or(false));

        // Missing private state fails closed.
        let private = dir.join("private").join("prove-state.bin");
        let backup = dir.join("private").join("prove-state.backup");
        std::fs::rename(&private, &backup).expect("move private");
        let output = run_stage(&executable, &dir, "prove");
        assert!(!output.status.success());
        std::fs::rename(&backup, &private).expect("restore private");

        // A truncated artifact is rejected by the verifier.
        let artifact_path = dir.join("proof").join("artifact.bin");
        let artifact = std::fs::read(&artifact_path).expect("artifact");
        std::fs::write(&artifact_path, &artifact[..artifact.len() / 2]).expect("truncate");
        let output = run_stage(&executable, &dir, "verify");
        assert!(!output.status.success());
        std::fs::write(&artifact_path, &artifact).expect("restore artifact");

        // A changed public handoff is rejected by the verifier.
        let public_path = dir.join("public").join("public.bin");
        let public_bytes = std::fs::read(&public_path).expect("public");
        let mut changed = public_bytes.clone();
        changed[0] ^= 1;
        std::fs::write(&public_path, &changed).expect("change public");
        let output = run_stage(&executable, &dir, "verify");
        assert!(!output.status.success());
        std::fs::write(&public_path, &public_bytes).expect("restore public");

        // A replaced plan identity is rejected by the verifier.
        let mut handoff = read_public(&public_path).expect("handoff");
        handoff.plan_identifier = Digest([0xee; 32]);
        std::fs::write(
            &public_path,
            &handoff.encode().expect("encode handoff"),
        )
        .expect("change identity");
        let output = run_stage(&executable, &dir, "verify");
        assert!(!output.status.success());
        std::fs::write(&public_path, &public_bytes).expect("restore public identity");

        // A prover-reported new table root cannot replace the policy-pinned
        // identity: the verifier rebuilds the committed identity and rejects
        // a root that the operator policy does not name.
        let mut handoff = read_public(&public_path).expect("handoff");
        let mut replaced_statement =
            LimitedKeyStatement::decode(&handoff.statement).expect("statement");
        replaced_statement.table_root = Digest([0xef; 32]);
        handoff.statement = replaced_statement.encode();
        handoff.reader.root = replaced_statement.table_root;
        std::fs::write(&public_path, &handoff.encode().expect("encode handoff"))
            .expect("replace root");
        let output = run_stage(&executable, &dir, "verify");
        assert!(!output.status.success());
        std::fs::write(&public_path, &public_bytes).expect("restore root");

        // Replaced private bytes are rejected before any proof work.
        let private_bytes = std::fs::read(&private).expect("private");
        let mut changed_private = private_bytes.clone();
        changed_private[0] ^= 1;
        std::fs::write(&private, &changed_private).expect("change private");
        let output = run_stage(&executable, &dir, "prove");
        assert!(!output.status.success());
        std::fs::write(&private, &private_bytes).expect("restore private bytes");

        // A stale reservation lock fails closed.
        std::fs::write(dir.join("private").join("budget.lock"), b"stale").expect("lock");
        let output = run_stage(&executable, &dir, "prove");
        assert!(!output.status.success());
        let _ = std::fs::remove_file(dir.join("private").join("budget.lock"));

        let artifact_bytes = std::fs::read(dir.join("proof").join("artifact.bin"))
            .expect("artifact copy");
        let _ = std::fs::write(
            std::env::temp_dir().join(format!("zkie-stage-artifact-{}.bin", std::process::id())),
            artifact_bytes,
        );
        let _ = positive;
        let _ = std::fs::remove_dir_all(&dir);
    }
}


#[cfg(test)]
mod lin_request_tests {
    use super::*;

    fn sample() -> LinRequestDeclaration {
        LinRequestDeclaration {
            version: LIN_REQUEST_DECLARATION_VERSION,
            model_id: Digest([1; 32]),
            plan_identifier: Digest([2; 32]),
            activation_step: 4443,
            activation_output: 0,
            matmul_step: 4444,
            output_step: 4445,
            weight_asset_id: 7,
            weight_table_identity: Digest([3; 32]),
            matrix_rows: 16,
            k_col: 16,
            k_lo: 6,
            activation_digest: lin_values_digest(b"x", &[16, 64], &[1, 2, 3]),
            output_digest: lin_values_digest(b"y", &[16, 16], &[4, 5]),
            final_output_digest: lin_values_digest(b"final", &[16], &[6]),
        }
    }

    #[test]
    fn the_declaration_round_trips_and_binds_its_domain() {
        let value = sample();
        let bytes = value.encode().expect("encode");
        let decoded = LinRequestDeclaration::decode(&bytes).expect("decode");
        assert_eq!(decoded, value);
        assert_eq!(decoded.digest().expect("digest"), value.digest().expect("digest"));
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(LinRequestDeclaration::decode(&trailing).is_err());
        let mut version = value.clone();
        version.version += 1;
        assert!(version.validate().is_err());
    }

    #[test]
    fn every_bound_field_changes_the_digest_or_is_rejected() {
        let value = sample();
        let digest = value.digest().expect("digest");
        let mut step = value.clone();
        step.activation_step = 4442;
        assert_ne!(step.digest().expect("digest"), digest);
        let mut asset = value.clone();
        asset.weight_table_identity = Digest([9; 32]);
        assert_ne!(asset.digest().expect("digest"), digest);
        let mut shape = value.clone();
        shape.matrix_rows = 32;
        assert_ne!(shape.digest().expect("digest"), digest);
        let mut bad_order = value.clone();
        bad_order.output_step = 4444;
        assert!(bad_order.validate().is_err());
        let mut zero = value.clone();
        zero.model_id = Digest::ZERO;
        assert!(zero.validate().is_err());
    }

    #[test]
    fn value_digests_separate_labels_shapes_and_values() {
        let base = lin_values_digest(b"x", &[2], &[1, 2]);
        assert_ne!(base, lin_values_digest(b"y", &[2], &[1, 2]));
        assert_ne!(base, lin_values_digest(b"x", &[1, 2], &[1, 2]));
        assert_ne!(base, lin_values_digest(b"x", &[2], &[1, 3]));
    }
}
