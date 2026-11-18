//! The one-time key qualification of the registered encoded table.
//!
//! The commit stage proves the table's key chain once over the same committed
//! instance and publishes the canonical bundle in the commit's public bundle.
//! Every request then re-verifies that one bundle through the public entry and
//! holds an in-memory accepted key; a missing bundle, a wrong instance or a
//! failed qualification is refused. Re-verifying the same proof spends the
//! plan's in-memory read ledger, never the table's durable column budget; the
//! prover builds the witness from the asset mapping one stored row at a time.

use std::sync::Arc;

use zkie_core::artifact::digest_bytes;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::memory::BudgetedMemoryGovernor;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::foldeq::reader_table_identity;
use zkie_engine::pcs::request::table::IndexReader;

use crate::proof::asset_table::{verify_asset_table_meta, AssetTableInput, AssetTableMeta, AssetTableParams};
use crate::proof::consumer_session::{consumer_table_reader, open_consumer_table_keeper};
use crate::proof::hidden_send_session::{
    production_key_first_plan, prove_key, verify_key, AcceptedKey, HiddenTableProver,
    KeyPlanInputs, KeyRunRandomness, SharedLinCheck,
};
use crate::proof::model_commitment::{AssetReader, ModelAssetSpec, ModelCommitment};
use crate::proof::table_range::{
    range_bit_values, range_multiplicities, EntryRangePlan, IntegerRangeTable,
};

/// The public bundle path of the request's one-time key proof.
pub const ASSET_KEY_FILE: &str = "public/asset-key.bin";
/// The canonical bundle path under the commit artifact root.
pub const ASSET_KEY_BUNDLE: &str = "public/asset-key.bin";

/// The trusted, model-derived parameters of one table's key stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssetKeyColumns {
    /// The key stage's own column budget (the one-time test's columns).
    pub key_columns: u64,
    /// The instance's total column budget (one-time plus the declared later
    /// batches); the durable ledger's life limit.
    pub instance_columns: u64,
    pub k_lo: u32,
    pub lin_check: SharedLinCheck,
}

/// The public plan of the key chain for one committed table.
pub fn asset_key_plan(
    meta: &AssetTableMeta,
    params: &AssetTableParams,
    table_model_identity: Digest,
    spec: &ModelAssetSpec,
    columns: AssetKeyColumns,
    range: &EntryRangePlan,
) -> Result<crate::proof::hidden_send_session::KeyFirstPlan> {
    verify_asset_table_meta(meta, table_model_identity, spec, params)?;
    let reader = consumer_table_reader(meta, params, table_model_identity, spec)?;
    let matrix_rows = meta.shape.rows();
    let full_columns = matrix_rows
        .checked_shl(columns.k_lo)
        .ok_or(Error::ResourceLimit)?;
    let plan = production_key_first_plan(&KeyPlanInputs {
        table_identity: reader_table_identity(&reader)?,
        table_configuration: meta.configuration,
        table_shape: meta.shape,
        matrix_rows,
        k_lo: columns.k_lo,
        columns: full_columns,
        total_columns: columns.key_columns,
        lin_check: columns.lin_check,
        range: range.clone(),
    })?;
    Ok(plan)
}

/// Reads the key witness of the committed matrix one stored row at a time and
/// returns the keys and the multiplicity bits. The matrix entry `(row, column)`
/// is the stored cell of record `column` at place `row`; no whole asset is
/// materialized.
fn key_witness(
    commitment: &ModelCommitment,
    params: &AssetTableParams,
    range: &EntryRangePlan,
) -> Result<(Vec<Fp3>, Vec<Fp3>)> {
    let columns = 1u64
        .checked_shl(range.column_bits)
        .ok_or(Error::ResourceLimit)?;
    let count = range
        .matrix_rows
        .checked_mul(columns)
        .ok_or(Error::ResourceLimit)?;
    let entries = usize::try_from(count).map_err(|_| Error::ResourceLimit)?;
    let rows = usize::try_from(columns).map_err(|_| Error::ResourceLimit)?;
    let width = usize::try_from(params.layout.row_width).map_err(|_| Error::ResourceLimit)?;
    if range.matrix_rows != u64::try_from(width).map_err(|_| Error::ResourceLimit)? {
        return Err(Error::InvalidConfig);
    }
    let asset = commitment
        .assets
        .iter()
        .find(|asset| asset.path == params.path)
        .ok_or(Error::AssetMismatch)?;
    let window = asset
        .element_count
        .checked_mul(8)
        .and_then(|bytes| bytes.checked_add(1 << 30))
        .ok_or(Error::ResourceLimit)?;
    let governor = BudgetedMemoryGovernor::new(window);
    let reader = Arc::new(AssetReader::open(commitment, asset.asset_id, &governor)?);
    let mut keys = vec![Fp3::ZERO; entries];
    for column in 0..rows {
        let start = u64::try_from(column)
            .map_err(|_| Error::ResourceLimit)?
            .checked_mul(u64::try_from(width).map_err(|_| Error::ResourceLimit)?)
            .ok_or(Error::ResourceLimit)?;
        let row_values = reader.read_i64(start, width)?;
        if row_values.len() != width {
            return Err(Error::ShapeMismatch);
        }
        for (place, value) in row_values.iter().enumerate() {
            let slot = place
                .checked_mul(rows)
                .and_then(|base| base.checked_add(column))
                .ok_or(Error::ResourceLimit)?;
            let entry = crate::proof::consumer_request::signed_fp3(i128::from(*value))?;
            keys[slot] = range.offset.add(&entry);
        }
    }
    let counts = range_multiplicities(range, &keys)?;
    let bits = range_bit_values(range, &counts)?;
    Ok((keys, bits))
}

/// Proves the table's one-time key bundle at commit time. The keeper is the
/// same streaming table object the request consumer stages use; the run seed is
/// fresh private material and is never published.
pub fn prove_asset_key(
    commitment: &ModelCommitment,
    meta: &AssetTableMeta,
    params: &AssetTableParams,
    table_model_identity: Digest,
    spec: &ModelAssetSpec,
    input: &AssetTableInput,
    columns: AssetKeyColumns,
    range: &EntryRangePlan,
) -> Result<Vec<u8>> {
    if input.params != *params {
        return Err(Error::StatementMismatch);
    }
    verify_asset_table_meta(meta, table_model_identity, spec, params)?;
    let plan = asset_key_plan(meta, params, table_model_identity, spec, columns, range)?;
    let (keys, bits) = key_witness(commitment, params, range)?;
    let lookup = IntegerRangeTable::new(range, 1 << 20)?;
    let reader = consumer_table_reader(meta, params, table_model_identity, spec)?;
    let mut keeper = open_consumer_table_keeper(
        commitment,
        meta,
        params,
        table_model_identity,
        input,
        columns.key_columns,
        columns.instance_columns,
        u64::MAX,
    )?;
    if keeper.reader().root != reader.root || keeper.reader().shape != reader.shape {
        return Err(Error::StatementMismatch);
    }
    let seed = KeyRunRandomness::Private.seed()?;
    let (bytes, _parts) = prove_key(&plan, &reader, &mut keeper, &lookup, &keys, &bits, seed)?;
    Ok(bytes)
}

/// Verifies one key bundle through the public entry and checks every accepted
/// field against this commit's table, record, range and plan. The key chain's
/// own messages, obligations and seals verify inside the entry; here the
/// verifier only compares what it already trusts with what the entry accepted.
#[allow(clippy::too_many_arguments)]
pub fn verify_asset_key(
    meta: &AssetTableMeta,
    params: &AssetTableParams,
    table_model_identity: Digest,
    spec: &ModelAssetSpec,
    columns: AssetKeyColumns,
    range: &EntryRangePlan,
    bytes: &[u8],
) -> Result<AcceptedKey> {
    let reader = consumer_table_reader(meta, params, table_model_identity, spec)?;
    let plan = asset_key_plan(meta, params, table_model_identity, spec, columns, range)?;
    let lookup = IntegerRangeTable::new(range, 1 << 20)?;
    let accepted = verify_key(&plan, &reader, &lookup, bytes)?;
    let record_digest = {
        let mut writer = zkie_engine::primitives::codec::CanonicalWriter::new();
        writer.put_bytes(&reader.record.encode());
        digest_bytes(&writer.into_bytes())
    };
    if accepted.plan_digest() != plan.statement_digest()
        || accepted.table_identity() != plan.table_identity
        || accepted.table_configuration() != plan.table_configuration
        || accepted.table_root() != reader.root
        || accepted.record_digest() != record_digest
        || accepted.range_digest() != plan.range.digest()
        || accepted.first_seal() == Digest::ZERO
        || accepted.shared_seal() == Digest::ZERO
        || accepted.second_seal() == Digest::ZERO
        || accepted.binding_root().fault().is_some()
    {
        return Err(Error::StatementMismatch);
    }
    Ok(accepted)
}

/// The public reader of one committed table, for callers that also want the
/// identity the key chain accepted.
pub fn asset_key_reader(
    meta: &AssetTableMeta,
    params: &AssetTableParams,
    table_model_identity: Digest,
    spec: &ModelAssetSpec,
) -> Result<IndexReader> {
    consumer_table_reader(meta, params, table_model_identity, spec)
}
