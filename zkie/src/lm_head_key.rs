//! The output-head table's key material: the committed table, its record and
//! the mapping from the authenticated weights.
//!
//! The builder reads the authenticated weight rows through the read-only row
//! path, never the whole asset, and fills one table row at a time. The table
//! entry at row `r`, matrix row `y` and low column `j` is the source weight at
//! matrix row `y`, column `2^k_lo r + j`. Random tails, pads and salts are the
//! caller's private values; the builder checks their sizes and binds them
//! before the table's root is computed.

use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::field::{Fp, E_DEGREE};
use zkie_engine::pcs::request::chain::Chain;
use zkie_engine::pcs::request::plan::RequestPlan;
use zkie_engine::pcs::request::read::HidingState;
use zkie_engine::pcs::request::table::{
    commit_table, verify_columns, ColumnOpening, CommittedTable, IndexReader, IndexShape,
};
use zkie_ops::proof::column_session::fp_from_signed_word;

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


/// Starts the shared chain from the table's public name and record and the
/// caller's statement bytes. The chain's configuration is the table's.
pub fn start_shared_chain(
    reader: &IndexReader,
    label: u64,
    statement_bytes: &[u8],
    random_bytes: Option<&Digest>,
) -> Result<Chain> {
    if reader.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    Chain::begin(
        label,
        &reader.configuration,
        statement_bytes,
        &reader.record.encode(),
        random_bytes,
    )
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
        use zkie_engine::pcs::request::field::Fp3;
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
        use zkie_engine::pcs::request::field::Fp3;
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
        let mut chain = start_shared_chain(&table.reader, 0x1234, b"statement", None)
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
}
