//! TimesFM norm-only segment carrier and parameter table handles.
//!
//! The carrier publishes the two norm-only message stages of the L0 attention
//! RMS norm. The segment has no output-head table, no Dot/W rows and no
//! closure; the norm plan and the materialized activation identity carry the
//! relation. The parameter table is the single gain table built from the
//! committed L0_lnw asset with plan-fixed public material.

use std::path::Path;

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::table::{
    commit_table_streaming, ColumnOpening, EncodedRowStore, IndexReader, StreamingTable,
    TableRowSource,
};
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};
use zkie_engine::storage::encoded_store::FileRowStore;
use zkie_ops::proof::asset_table::{
    asset_row_source, asset_table_configuration, AssetRowSource, AssetTableInput,
};
use zkie_ops::proof::hidden_norm_psc::NormWitness;
use zkie_ops::proof::hidden_send_session::{
    prove_norm_only_first_owned, prove_norm_only_second, verify_norm_only_first,
    verify_norm_only_second, HiddenLinPlan, NormOnlyFirstMessages, NormOnlySecondMessages,
    NormParamTableProver,
};
use zkie_ops::proof::lookup::LookupTable;
use zkie_ops::proof::lookup_table::{PublicFileTable, RangeTable};
use zkie_ops::proof::model_commitment::{open_asset_reader, ModelCommitment, ModelAssetSpec};
use zkie_ops::proof::norm_param_table::NormParamKind;

pub const TIMESFM_NORM_CARRIER_FILE: &str = "timesfm-norm-segment-carrier.bin";
pub const TIMESFM_NORM_CARRIER_VERSION: u64 = 1;
const GAIN_PATH: &str = "models/timesfm/weights/L0_lnw_i32.bin";

/// The public carrier of the norm-only segment: the canonical encoded first
/// and second message stages. The proof manifest binds the file by length and
/// digest; the plan is rebuilt before either stage is decoded.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TimesfmNormCarrier {
    pub version: u64,
    pub first: Vec<u8>,
    pub second: Vec<u8>,
}

impl TimesfmNormCarrier {
    pub fn encode(&self) -> Result<Vec<u8>> {
        if self.version != TIMESFM_NORM_CARRIER_VERSION
            || self.first.is_empty()
            || self.second.is_empty()
        {
            return Err(Error::InvalidConfig);
        }
        let mut writer = CanonicalWriter::new();
        writer.put_u64(self.version);
        writer.put_bytes(&self.first);
        writer.put_bytes(&self.second);
        Ok(writer.into_bytes())
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let limits = zkie_core::common::codec::DecodeLimits {
            max_bytes: bytes.len(),
            max_elements: bytes.len(),
        };
        let mut reader = CanonicalReader::new(bytes, limits);
        let version = reader.get_u64()?;
        if version != TIMESFM_NORM_CARRIER_VERSION {
            return Err(Error::UnsupportedVersion);
        }
        let first = reader.get_bytes()?.to_vec();
        let second = reader.get_bytes()?.to_vec();
        reader.finish()?;
        if first.is_empty() || second.is_empty() {
            return Err(Error::InvalidEncoding);
        }
        Ok(Self {
            version,
            first,
            second,
        })
    }

    pub fn decode_first(&self, plan: &HiddenLinPlan) -> Result<NormOnlyFirstMessages> {
        NormOnlyFirstMessages::decode(plan, &self.first)
    }

    pub fn decode_second(&self, plan: &HiddenLinPlan) -> Result<NormOnlySecondMessages> {
        NormOnlySecondMessages::decode(plan, &self.second)
    }
}

/// The plan-fixed public configuration of the timesfm norm segment. It covers
/// the model identity and the norm profile constants; no table identity is
/// carried (there is no output-head table).
pub fn configuration(model: Digest) -> Digest {
    let mut writer = CanonicalWriter::new();
    writer.put_bytes(b"zkie/timesfm-norm-segment/v1");
    writer.put_digest(&model);
    writer.put_u64(u64::from(zkie_models_timesfm::norm_profile::HIDDEN_PAD));
    writer.put_u64(u64::from(zkie_models_timesfm::norm_profile::VALID));
    writer.put_u64(u64::from(zkie_models_timesfm::norm_profile::OUTPUT_SHIFT));
    writer.put_u64(zkie_models_timesfm::norm_profile::INPUT_BOUND);
    writer.put_u64(zkie_models_timesfm::norm_profile::RSTD_VALUE_BOUND);
    writer.put_u64(zkie_models_timesfm::norm_profile::RSTD_TABLE_REAL_LEN);
    zkie_core::artifact::digest_bytes(&writer.into_bytes())
}

/// The two norm-only stages of one request: prove and verify wrappers that
/// keep the carrier codec in one place.
pub fn prove_bytes(
    plan: &HiddenLinPlan,
    seed: Digest,
    norm: NormWitness,
    param_tables: &mut dyn NormParamTableProver,
    rstd_table: &dyn LookupTable,
) -> Result<TimesfmNormCarrier> {
    let (first, owned) = prove_norm_only_first_owned(plan, seed, norm, Some(param_tables))?;
    let (second, _binding) =
        prove_norm_only_second(plan, &first, &owned, seed, Some(rstd_table))?;
    Ok(TimesfmNormCarrier {
        version: TIMESFM_NORM_CARRIER_VERSION,
        first: first.encode()?,
        second: second.encode()?,
    })
}

pub fn verify_bytes(
    plan: &HiddenLinPlan,
    carrier: &TimesfmNormCarrier,
    rstd_table: &dyn LookupTable,
) -> Result<()> {
    let first = carrier.decode_first(plan)?;
    let _outcome = verify_norm_only_first(plan, &first)?;
    let second = carrier.decode_second(plan)?;
    let _binding = verify_norm_only_second(plan, &first, &second, Some(rstd_table))?;
    Ok(())
}

/// The gain-only parameter table, opened for the norm claim.
pub struct TimesfmParamTables {
    gain_table: StreamingTable,
    gain_source: AssetRowSource<zkie_ops::proof::model_commitment::AssetReader>,
}

fn build_param_table(
    commitment: &ModelCommitment,
    input: &AssetTableInput,
) -> Result<(StreamingTable, AssetRowSource<zkie_ops::proof::model_commitment::AssetReader>)> {
    let spec: ModelAssetSpec = zkie_models_timesfm::plan::asset_specs()
        .into_iter()
        .find(|spec| spec.path == input.params.path)
        .ok_or(Error::InvalidConfig)?;
    let identity = zkie_models_timesfm::plan::table_model_identity()?;
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
) -> Result<TimesfmParamTables> {
    let gain = inputs
        .iter()
        .find(|input| input.params.path == GAIN_PATH)
        .ok_or(Error::InvalidConfig)?;
    let (gain_table, gain_source) = build_param_table(commitment, gain)?;
    Ok(TimesfmParamTables {
        gain_table,
        gain_source,
    })
}

impl NormParamTableProver for TimesfmParamTables {
    fn source(&mut self, kind: NormParamKind) -> &mut dyn TableRowSource {
        match kind {
            NormParamKind::Gain => &mut self.gain_source,
            // A gain-only plan carries no bias claim; the drain never asks for
            // the bias source. Returning the gain source keeps the trait total
            // without adding a second table.
            NormParamKind::Bias => &mut self.gain_source,
        }
    }

    fn open_columns(
        &mut self,
        kind: NormParamKind,
        positions: &[u64],
    ) -> Result<Vec<ColumnOpening>> {
        match kind {
            NormParamKind::Gain => self.gain_table.open_columns(&mut self.gain_source, positions),
            NormParamKind::Bias => Err(Error::InvalidConfig),
        }
    }
}

/// The frozen public rsqrt table view, the same view the shared norm chain
/// checks. TimesFM uses the shared GPT-2 table with no extension.
pub fn rstd_table_view(
    asset_root: &Path,
) -> Result<RangeTable<PublicFileTable>> {
    let config = zkie_models_timesfm::norm_profile::rstd_table_config();
    let source = PublicFileTable::new(
        asset_root.join("models/gpt2/weights/rsqrt_table_i32.bin"),
        config.clone(),
    )?;
    RangeTable::open(
        source,
        zkie_models_timesfm::norm_profile::RSTD_TABLE_LOG,
        &config,
    )
}
