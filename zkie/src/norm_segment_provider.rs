//! Production carrier and table handles of the final norm segment.
//!
//! The prover builds the output-head table from the trusted asset and the
//! plan-fixed public key material, opens the two committed parameter tables,
//! builds the norm witness from the recorded trace and runs the accepted
//! norm-hidden-output chain. The verifier pins the published table reader to
//! the operator policy, rebuilds the same plan from the trace and verifies
//! the chain.

use std::path::{Path, PathBuf};

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::field::Fp;
use zkie_engine::pcs::request::plan::RequestPlan;
use zkie_engine::pcs::request::read::HidingState;
use zkie_engine::pcs::request::table::{
    commit_table_streaming, validate_index_shape, ColumnOpening, EncodedRowStore, IndexReader,
    IndexShape, StreamingTable, TableDistances, TableRowSource,
};
use zkie_engine::primitives::codec::CanonicalWriter;
use zkie_engine::storage::encoded_store::FileRowStore;
use zkie_ops::proof::asset_table::{
    asset_row_source, asset_table_configuration, AssetRowSource, AssetTableInput,
};
use zkie_ops::proof::hidden_norm_psc::NormWitness;
use zkie_ops::proof::hidden_send_session::{
    prove_hidden_lin_hidden_with_norm_hd_rstd_table,
    verify_hidden_lin_hidden_with_rstd_table, HiddenLinMessages, HiddenLinOutcome, HiddenLinPlan,
    HiddenTableProver, NormParamTableProver,
};
use zkie_ops::proof::lookup::LookupTable;
use zkie_ops::proof::lookup_table::{PublicFileTable, RangeTable};
use zkie_ops::proof::model_commitment::{
    open_asset_reader, AssetReader, ModelAssetSpec, ModelCommitment,
};
use zkie_ops::proof::norm_param_table::NormParamKind;

use crate::lm_head_key::{
    build_lm_head_holder_from_asset, fixed_lm_head_key_randomness, limited_key_binding_parameters,
    LmHeadAssetSource, LmHeadKeyRandomness,
};
use crate::s7_context::LmHeadAsset;

/// The proof-dir carrier of the segment: the published public reader and the
/// canonical session messages. The name and the version are the reserved
/// carrier identity; the proof manifest binds the file by length and digest.
pub const NORM_SEGMENT_CARRIER_FILE: &str = "norm-segment-carrier.bin";
pub const NORM_SEGMENT_CARRIER_VERSION: u64 = 1;

const PANEL_COLUMNS: u64 = 4096;
const TABLE_MAX_BYTES: u64 = 2 << 30;
const CAP_BYTES: u64 = 1_700_000_000;
const WINDOW_BYTES: u64 = 1 << 20;
const TOTAL_BYTES: u64 = 1 << 30;
const STORE_MAX_BYTES: u64 = 4 << 30;

/// The plan-fixed identity of the bounded output-head table source. The
/// production consumer opens the source with exactly these bounds; the
/// identity rides the segment statement and the verifier compares it with
/// this derivation, so a swapped source is a statement mismatch.
pub fn segment_table_source_identity() -> Digest {
    let mut writer = CanonicalWriter::new();
    writer.put_u64(1);
    writer.put_bytes(b"bounded-asset");
    writer.put_u64(CAP_BYTES);
    writer.put_u64(WINDOW_BYTES);
    writer.put_u64(TOTAL_BYTES);
    zkie_core::artifact::digest_bytes(&writer.into_bytes())
}

/// The public carrier of the segment proof.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NormSegmentCarrier {
    pub version: u64,
    pub reader: IndexReader,
    pub messages: Vec<u8>,
}

impl NormSegmentCarrier {
    pub fn encode(&self) -> Result<Vec<u8>> {
        if self.version != NORM_SEGMENT_CARRIER_VERSION || self.messages.is_empty() {
            return Err(Error::InvalidConfig);
        }
        crate::lm_head_stage::encode_bytes(self)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let value: Self = crate::lm_head_stage::decode_bytes(bytes)?;
        if value.version != NORM_SEGMENT_CARRIER_VERSION || value.messages.is_empty() {
            return Err(Error::InvalidConfig);
        }
        Ok(value)
    }
}


/// The complete table identity the segment's own plan fixes for its output
/// head table. The consumer policy block names a different table; this
/// identity rides the statement and its digest is absorbed into the segment's
/// session derivation, so a changed identity breaks the proof.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SegmentTableIdentity {
    pub asset_id: u64,
    pub asset_table_identity: Digest,
    pub code_log_n: u32,
    pub code_ell: u64,
    pub code_r: u64,
    pub table_v: u32,
    pub answers: u32,
    pub salt_bytes: u64,
    pub distance_radius: u64,
    pub distance_later_radius: u64,
    pub distance_m_once: u64,
    pub distance_m_later: u64,
    pub matrix_rows_log: u32,
    pub k_lo: u32,
    pub k_col: u32,
    pub table_root: Digest,
    pub table_record_digest: Digest,
    pub table_configuration: Digest,
    /// The identity of the bounded table source (kind and bounds) the prover
    /// opened. The verifier compares it with the plan-fixed derivation, so a
    /// statement produced from a different source is a mismatch.
    pub source_identity: Digest,
}

impl SegmentTableIdentity {
    /// The table shape declared by the identity.
    pub fn shape(&self) -> Result<IndexShape> {
        let code = zkie_engine::pcs::request::code::build_code_plan(
            self.code_log_n,
            self.code_ell,
            self.code_r,
        )?;
        let shape = IndexShape {
            code,
            v: self.table_v,
            answers: self.answers,
            salt_bytes: usize::try_from(self.salt_bytes).map_err(|_| Error::InvalidConfig)?,
            distances: TableDistances {
                radius: self.distance_radius,
                later_radius: self.distance_later_radius,
                m_once: self.distance_m_once,
                m_later: self.distance_m_later,
            },
        };
        validate_index_shape(&shape)?;
        Ok(shape)
    }

    /// The matrix row count declared by the identity.
    pub fn matrix_rows(&self) -> Result<u64> {
        if self.table_v != self.matrix_rows_log
            || self.table_v >= 64
            || self.k_lo >= self.k_col
            || self.k_col > 63
        {
            return Err(Error::InvalidConfig);
        }
        let rows = 1u64.checked_shl(self.table_v).ok_or(Error::InvalidConfig)?;
        let factor = 1u64.checked_shl(self.k_lo).ok_or(Error::InvalidConfig)?;
        if factor.checked_mul(rows).ok_or(Error::ResourceLimit)? != 1u64 << self.k_col {
            return Err(Error::InvalidConfig);
        }
        Ok(rows)
    }

    /// Checks the identity's digests, shape and matrix mapping.
    pub fn validate(&self) -> Result<()> {
        if self.asset_id == 0
            || self.asset_table_identity == Digest::ZERO
            || self.table_root == Digest::ZERO
            || self.table_record_digest == Digest::ZERO
            || self.table_configuration == Digest::ZERO
            || self.source_identity == Digest::ZERO
        {
            return Err(Error::InvalidConfig);
        }
        self.shape()?;
        self.matrix_rows()?;
        Ok(())
    }
}

/// The canonical digest of the segment identity. It is fixed in the plan and
/// absorbed into the session derivation. The source identity is checked
/// directly against the plan-fixed derivation instead of entering this
/// digest, so the session derivation and the carrier bytes stay unchanged.
pub fn segment_identity_binding(identity: &SegmentTableIdentity) -> Digest {
    let mut writer = CanonicalWriter::new();
    writer.put_u64(1);
    writer.put_u64(identity.asset_id);
    writer.put_bytes(&identity.asset_table_identity.0);
    writer.put_u64(identity.code_log_n as u64);
    writer.put_u64(identity.code_ell);
    writer.put_u64(identity.code_r);
    writer.put_u64(identity.table_v as u64);
    writer.put_u64(identity.answers as u64);
    writer.put_u64(identity.salt_bytes);
    writer.put_u64(identity.distance_radius);
    writer.put_u64(identity.distance_later_radius);
    writer.put_u64(identity.distance_m_once);
    writer.put_u64(identity.distance_m_later);
    writer.put_u64(identity.matrix_rows_log as u64);
    writer.put_u64(identity.k_lo as u64);
    writer.put_u64(identity.k_col as u64);
    writer.put_bytes(&identity.table_root.0);
    writer.put_bytes(&identity.table_record_digest.0);
    writer.put_bytes(&identity.table_configuration.0);
    zkie_core::artifact::digest_bytes(&writer.into_bytes())
}

/// The output-head table prover: the committed table and the read-only row
/// source opened over the same asset.
pub struct SegmentTableProver {
    table: StreamingTable,
    source: LmHeadAssetSource,
    hiding: HidingState,
}

impl TableRowSource for SegmentTableProver {
    fn rows(&self) -> u64 {
        TableRowSource::rows(&self.source)
    }

    fn row(&mut self, row: u64, message: &mut [Fp], tail: &mut [Fp]) -> Result<()> {
        TableRowSource::row(&mut self.source, row, message, tail)
    }
}

impl HiddenTableProver for SegmentTableProver {
    fn reader(&self) -> &IndexReader {
        &self.table.reader
    }

    fn open_columns(&mut self, positions: &[u64]) -> Result<Vec<ColumnOpening>> {
        self.table.open_columns(&mut self.source, positions)
    }

    fn hiding_state(&mut self) -> Option<&mut HidingState> {
        Some(&mut self.hiding)
    }
}

impl SegmentTableProver {
    pub fn reader(&self) -> &IndexReader {
        &self.table.reader
    }
}

/// The plan-fixed configuration of the output-head table. It is the digest of
/// the limited-key binding parameters over the request plan and the selected
/// static plan identifier, the same derivation the operator policy binds.
pub fn configuration_for(
    request_plan: &RequestPlan,
    plan_identifier: Digest,
) -> Result<Digest> {
    limited_key_binding_parameters(request_plan, plan_identifier)?.digest()
}

/// The selected static plan identifier of the output head MatMul step.
pub fn plan_identifier(
    public: &zkie_ops::proof::model_commitment::ModelCommitmentPublic,
    trace_plan: &zkie_ops::proof::plan::TracePlan,
    matmul_index: usize,
) -> Result<Digest> {
    let selection = zkie_ops::proof::session_baseline::SelectedMatmulStep::select(
        trace_plan,
        matmul_index,
    )?;
    zkie_ops::proof::session_baseline::selected_plan_identifier(public, trace_plan, &selection)
}

/// The trusted lm_head asset identity of the committed anchor.
pub fn lm_head_asset(commitment: &ModelCommitment, path: &str) -> Result<LmHeadAsset> {
    let specs = zkie_models_gpt2::plan::asset_specs();
    let spec = specs
        .iter()
        .find(|spec| spec.path == path)
        .ok_or(Error::UnsupportedPrimitive)?;
    let meta = commitment
        .assets
        .iter()
        .find(|asset| asset.path == spec.path)
        .ok_or(Error::AssetMismatch)?;
    if meta.asset_id != spec.asset_id
        || meta.element_count != spec.element_count
        || meta.row_width != spec.row_width
    {
        return Err(Error::StatementMismatch);
    }
    Ok(LmHeadAsset {
        asset_id: meta.asset_id,
        path: meta.path.clone(),
        table_identity: zkie_ops::proof::model_commitment::asset_table_identity(spec),
        element_count: meta.element_count,
        row_width: meta.row_width as u64,
        rows: meta.rows,
        start: meta.start,
        root: meta.root,
        leaf_root: meta.leaf_record.leaf_root,
    })
}

/// Builds the committed output-head table from the trusted asset fields and
/// the plan-fixed public key material, then reopens the same table and source
/// for proving.
pub fn build_segment_table(
    asset_root: &Path,
    asset: &LmHeadAsset,
    request_plan: &RequestPlan,
    configuration: Digest,
    key: &LmHeadKeyRandomness,
    work_dir: &Path,
) -> Result<SegmentTableProver> {
    let shape = request_plan.table_shape()?;
    let store_dir = work_dir.join("segment-table-store");
    let _ = std::fs::remove_dir_all(&store_dir);
    std::fs::create_dir_all(&store_dir).map_err(|_| Error::Io)?;
    let mut store = FileRowStore::new(
        &store_dir,
        Digest([0x5A; 32]),
        shape.rows(),
        shape.code.n(),
        configuration,
        STORE_MAX_BYTES,
    )?;
    let holder = build_lm_head_holder_from_asset(
        asset_root,
        asset,
        request_plan,
        configuration,
        key,
        &mut store,
        PANEL_COLUMNS,
        TABLE_MAX_BYTES,
        CAP_BYTES,
        WINDOW_BYTES,
        TOTAL_BYTES,
    )?;
    let reader = holder.reader().clone();
    let name = holder.table().name.clone();
    let leaves = holder.table().leaves().to_vec();
    let salts = holder.table().salts().to_vec();
    let peak = holder.table().peak_bytes();
    drop(holder);
    store.cleanup()?;
    let table = StreamingTable::rebuild(reader.clone(), name, leaves, salts, peak)?;
    let source = LmHeadAssetSource::open_with_asset(
        asset_root,
        asset,
        request_plan,
        configuration,
        key,
        CAP_BYTES,
        WINDOW_BYTES,
        TOTAL_BYTES,
    )?;
    let hiding = HidingState::new(reader.root, 1, reader.shape.code.r as u64);
    Ok(SegmentTableProver {
        table,
        source,
        hiding,
    })
}

/// The plan-fixed public key material of the output-head table.
pub fn segment_table_key(shape: IndexShape) -> LmHeadKeyRandomness {
    fixed_lm_head_key_randomness(shape)
}

/// The two committed parameter tables, opened for the request claim.
pub struct SegmentParamTables {
    gain_table: StreamingTable,
    gain_source: AssetRowSource<AssetReader>,
    bias_table: StreamingTable,
    bias_source: AssetRowSource<AssetReader>,
}

fn build_param_table(
    commitment: &ModelCommitment,
    input: &AssetTableInput,
) -> Result<(StreamingTable, AssetRowSource<AssetReader>)> {
    let spec: ModelAssetSpec = zkie_models_gpt2::plan::asset_specs()
        .into_iter()
        .find(|spec| spec.path == input.params.path)
        .ok_or(Error::InvalidConfig)?;
    let identity = zkie_models_gpt2::plan::table_model_identity()?;
    let configuration = asset_table_configuration(identity, &spec, &input.params)?;
    let reader = open_asset_reader(commitment, spec.asset_id)?;
    let mut source = asset_row_source(reader, &spec, &input.params, input.randomness.tails.clone())?;
    let panel = input.params.shape.code.n().min(1024).max(1);
    let table = commit_table_streaming(
        &mut source,
        input.params.shape,
        configuration,
        &input.randomness.salts,
        panel,
        1 << 30,
    )?;
    let meta = commitment
        .tables
        .iter()
        .find(|meta| meta.path == input.params.path)
        .ok_or(Error::MissingCommitArtifact)?;
    if table.reader.root != meta.root
        || table.reader.record != meta.record
        || table.reader.configuration != meta.configuration
        || table.reader.shape != meta.shape
    {
        return Err(Error::StatementMismatch);
    }
    Ok((table, source))
}

pub fn build_param_tables(
    commitment: &ModelCommitment,
    inputs: &[AssetTableInput],
) -> Result<SegmentParamTables> {
    let gain = inputs
        .iter()
        .find(|input| input.params.path.ends_with("ln_f_w_i32.bin"))
        .ok_or(Error::InvalidConfig)?;
    let bias = inputs
        .iter()
        .find(|input| input.params.path.ends_with("ln_f_b_i32.bin"))
        .ok_or(Error::InvalidConfig)?;
    let (gain_table, gain_source) = build_param_table(commitment, gain)?;
    let (bias_table, bias_source) = build_param_table(commitment, bias)?;
    Ok(SegmentParamTables {
        gain_table,
        gain_source,
        bias_table,
        bias_source,
    })
}

impl NormParamTableProver for SegmentParamTables {
    fn source(&mut self, kind: NormParamKind) -> &mut dyn TableRowSource {
        match kind {
            NormParamKind::Gain => &mut self.gain_source,
            NormParamKind::Bias => &mut self.bias_source,
        }
    }

    fn open_columns(
        &mut self,
        kind: NormParamKind,
        positions: &[u64],
    ) -> Result<Vec<ColumnOpening>> {
        match kind {
            NormParamKind::Gain => self.gain_table.open_columns(&mut self.gain_source, positions),
            NormParamKind::Bias => self.bias_table.open_columns(&mut self.bias_source, positions),
        }
    }
}

/// Proves the norm segment and returns the canonical sidecar bytes.
pub fn prove_segment_bytes(
    plan: &HiddenLinPlan,
    table: &mut SegmentTableProver,
    seed: Digest,
    witness: &NormWitness,
    params: &mut SegmentParamTables,
    rstd_table: &dyn LookupTable,
) -> Result<Vec<u8>> {
    let (messages, _witness) = prove_hidden_lin_hidden_with_norm_hd_rstd_table(
        plan,
        table,
        seed,
        witness,
        rstd_table,
        Some(params),
    )?;
    messages.encode()
}

/// Verifies the segment sidecar over the trusted plan and table reader.
pub fn verify_segment_bytes(
    plan: &HiddenLinPlan,
    reader: &IndexReader,
    bytes: &[u8],
    rstd_table: &dyn LookupTable,
) -> Result<HiddenLinOutcome> {
    let messages = HiddenLinMessages::decode(plan, bytes)?;
    verify_hidden_lin_hidden_with_rstd_table(plan, reader, &messages, rstd_table)
}

/// Opens the frozen public rstd table view from the trusted asset file.
pub fn rstd_table_view(asset_root: &Path) -> Result<RangeTable<PublicFileTable>> {
    let config = zkie_models_gpt2::norm_profile::rstd_table_config();
    let source = PublicFileTable::new(
        asset_root.join("gpt2/weights/rsqrt_table_i32.bin"),
        config.clone(),
    )?;
    RangeTable::open(
        source,
        zkie_models_gpt2::norm_profile::RSTD_TABLE_LOG,
        &config,
    )
}

/// The work directory of one prove stage, kept next to the proof artifact so
/// the temporary store stays on the data volume.
pub fn segment_work_dir(output_dir: &Path) -> Result<PathBuf> {
    let parent = output_dir.parent().ok_or(Error::IncompleteArtifact)?;
    Ok(parent.join("norm-segment-work"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::code::build_code_plan;
    use zkie_engine::pcs::request::field::Fp3;
    use zkie_engine::pcs::request::table::{IndexRecord, TableDistances};

    fn reader() -> IndexReader {
        let shape = IndexShape {
            code: build_code_plan(12, 2049, 4).unwrap(),
            v: 0,
            answers: 1,
            salt_bytes: 40,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        };
        IndexReader {
            shape,
            configuration: Digest([0x61; 32]),
            root: Digest([0x62; 32]),
            record: IndexRecord {
                y: vec![Fp3::ONE; shape.rows() as usize],
            },
        }
    }

    #[test]
    fn the_carrier_round_trips_and_rejects_edits() {
        let carrier = NormSegmentCarrier {
            version: NORM_SEGMENT_CARRIER_VERSION,
            reader: reader(),
            messages: vec![1, 2, 3, 4],
        };
        let bytes = carrier.encode().unwrap();
        let decoded = NormSegmentCarrier::decode(&bytes).unwrap();
        assert_eq!(decoded.version, carrier.version);
        assert_eq!(decoded.reader.root, carrier.reader.root);
        assert_eq!(decoded.reader.record, carrier.reader.record);
        assert_eq!(decoded.messages, carrier.messages);
        let mut wrong_version = carrier.clone();
        wrong_version.version += 1;
        assert!(wrong_version.encode().is_err());
        let mut empty = carrier.clone();
        empty.messages.clear();
        assert!(empty.encode().is_err());
        assert!(NormSegmentCarrier::decode(&bytes[..bytes.len() - 1]).is_err());
    }
}
