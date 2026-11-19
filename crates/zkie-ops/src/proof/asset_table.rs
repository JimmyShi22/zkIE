//! Encoded asset tables: the trusted plan parameters, the commit-side table
//! construction and the independent identity check.
//!
//! The raw row commitment in `model_commitment` stays the asset's storage
//! commitment. This module adds the column-encoded table over the same asset
//! rows. The table parameters come from the trusted model plan; the committer
//! builds the table from the read-only asset rows, and the verifier rebuilds
//! the expected identity from the plan and checks the committed table against
//! it. The table root enters the anchor through an extra leaf record, so a
//! changed table root breaks the anchor identity.

use zkie_core::error::{Error as ZkieError, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::field::{fp_from_u64, Fp, MODULUS};
use zkie_engine::pcs::request::table::{
    commit_table_streaming, validate_index_shape, IndexRecord, IndexReader, IndexShape,
    TableRowSource,
};
use zkie_engine::primitives::transcript::{hash_labeled, tag_word, STATEMENT};

use crate::proof::column_session::fp_from_signed_word;
use crate::proof::model_commitment::{
    asset_identity, expected_asset_rows, AssetReader, ModelAssetSpec,
};

/// The fixed trusted-plan layout of an asset table.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AssetTableLayout {
    pub row_width: u64,
    pub message_len: u64,
    pub tail_len: u64,
    pub pads: u64,
    pub valid_elements: u64,
    pub padding_zero: bool,
}

/// The trusted-plan parameters of one asset table.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AssetTableParams {
    pub asset_id: u64,
    pub path: String,
    pub element_count: u64,
    pub row_width: u64,
    pub encoding: String,
    pub shape: IndexShape,
    pub valid_min: i64,
    pub valid_max: i64,
    pub layout: AssetTableLayout,
}

/// The committer's private random material for one table. The salts and tails
/// never enter the public bundle; the table root commits them.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AssetTableRandomness {
    pub salts: Vec<Vec<u8>>,
    pub tails: Vec<Vec<Fp>>,
}

/// One table to build during the commit.
#[derive(Clone, Debug)]
pub struct AssetTableInput {
    pub params: AssetTableParams,
    pub randomness: AssetTableRandomness,
}

/// One table's private randomness, keyed by asset identity, for the private
/// commit artifact. The key binds the randomness to the asset whose rows it
/// salts.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AssetTableRandomnessEntry {
    pub asset_id: u64,
    pub randomness: AssetTableRandomness,
}

/// The committed table, public. The identity, shape, configuration and valid
/// range come from the trusted plan; the root and record are the committer's
/// data and are bound by the anchor.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AssetTableMeta {
    pub asset_id: u64,
    pub path: String,
    pub identity: Digest,
    pub shape: IndexShape,
    pub configuration: Digest,
    pub root: Digest,
    pub record: IndexRecord,
    pub valid_min: i64,
    pub valid_max: i64,
    pub valid_elements: u64,
}

fn push_bytes(data: &mut Vec<u8>, bytes: &[u8]) {
    data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(bytes);
}

fn push_shape(data: &mut Vec<u8>, shape: &IndexShape) {
    data.extend_from_slice(&shape.code.log_n.to_le_bytes());
    data.extend_from_slice(&shape.code.ell.to_le_bytes());
    data.extend_from_slice(&shape.code.r.to_le_bytes());
    data.extend_from_slice(&shape.v.to_le_bytes());
    data.extend_from_slice(&shape.answers.to_le_bytes());
    data.extend_from_slice(&(shape.salt_bytes as u64).to_le_bytes());
    data.extend_from_slice(&shape.distances.radius.to_le_bytes());
    data.extend_from_slice(&shape.distances.later_radius.to_le_bytes());
    data.extend_from_slice(&shape.distances.m_once.to_le_bytes());
    data.extend_from_slice(&shape.distances.m_later.to_le_bytes());
}

fn push_layout(data: &mut Vec<u8>, layout: &AssetTableLayout) {
    data.extend_from_slice(&layout.row_width.to_le_bytes());
    data.extend_from_slice(&layout.message_len.to_le_bytes());
    data.extend_from_slice(&layout.tail_len.to_le_bytes());
    data.extend_from_slice(&layout.pads.to_le_bytes());
    data.extend_from_slice(&layout.valid_elements.to_le_bytes());
    data.push(u8::from(layout.padding_zero));
}

fn table_configuration_bytes(spec: &ModelAssetSpec, params: &AssetTableParams) -> Vec<u8> {
    let mut data = Vec::new();
    push_bytes(&mut data, b"asset-table-configuration-v1");
    data.extend_from_slice(&params.asset_id.to_le_bytes());
    push_bytes(&mut data, params.path.as_bytes());
    data.extend_from_slice(&params.element_count.to_le_bytes());
    data.extend_from_slice(&params.row_width.to_le_bytes());
    push_bytes(&mut data, params.encoding.as_bytes());
    data.extend_from_slice(&(params.valid_min as u64).to_le_bytes());
    data.extend_from_slice(&(params.valid_max as u64).to_le_bytes());
    push_shape(&mut data, &params.shape);
    push_layout(&mut data, &params.layout);
    data.extend_from_slice(&asset_identity(spec).0);
    data
}

/// The configuration digest the table tree starts from. It is a function of
/// the trusted model identity and the trusted table parameters, never of the
/// committer's data.
pub fn asset_table_configuration(
    model_commitment: Digest,
    spec: &ModelAssetSpec,
    params: &AssetTableParams,
) -> Result<Digest> {
    let mut data = model_commitment.0.to_vec();
    data.extend_from_slice(&table_configuration_bytes(spec, params));
    hash_labeled(tag_word(STATEMENT, 18), 0, None, &data)
}

/// The identity of the encoded table. It covers the model identity, the raw
/// asset identity, the trusted shape/layout/range and the configuration.
pub fn asset_table_identity(
    model_commitment: Digest,
    spec: &ModelAssetSpec,
    params: &AssetTableParams,
) -> Result<Digest> {
    validate_asset_table_params(spec, params)?;
    let configuration = asset_table_configuration(model_commitment, spec, params)?;
    let mut data = model_commitment.0.to_vec();
    data.extend_from_slice(&asset_identity(spec).0);
    data.extend_from_slice(&configuration.0);
    data.extend_from_slice(&table_configuration_bytes(spec, params));
    hash_labeled(tag_word(STATEMENT, 19), 0, None, &data)
}

/// The statement identity of the committed table. It covers the plan identity,
/// the table root and the record, so the anchor binds all three.
pub fn asset_table_statement_identity(
    plan_identity: Digest,
    root: Digest,
    record: &IndexRecord,
) -> Digest {
    let record_bytes = record.encode();
    let mut data = plan_identity.0.to_vec();
    data.extend_from_slice(&root.0);
    data.extend_from_slice(&(record_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(&record_bytes);
    hash_labeled(tag_word(STATEMENT, 19), 1, None, &data).unwrap_or(Digest::ZERO)
}

/// Checks that the plan's table parameters describe this asset and are
/// internally consistent. Both the committer and the verifier run it.
pub fn validate_asset_table_params(spec: &ModelAssetSpec, params: &AssetTableParams) -> Result<()> {
    if params.asset_id != spec.asset_id
        || params.path != spec.path
        || params.element_count != spec.element_count
        || params.row_width != spec.row_width as u64
        || params.encoding != spec.encoding
    {
        return Err(ZkieError::StatementMismatch);
    }
    if params.valid_min > params.valid_max {
        return Err(ZkieError::InvalidConfig);
    }
    validate_index_shape(&params.shape)?;
    let pads = params.shape.pads() as u64;
    let layout = &params.layout;
    if params.shape.code.ell != layout.message_len
        || params.shape.code.r != layout.tail_len
        || pads != layout.pads
        || layout.valid_elements != spec.element_count
        || layout.row_width != spec.row_width as u64
        || !layout.padding_zero
    {
        return Err(ZkieError::InvalidConfig);
    }
    if layout.message_len < layout.row_width.saturating_add(pads) {
        return Err(ZkieError::InvalidConfig);
    }
    if expected_asset_rows(spec)? != params.shape.rows() {
        return Err(ZkieError::ShapeMismatch);
    }
    Ok(())
}


/// Samples fresh private randomness for one table: one salt per column and
/// one random tail element per row. The salts and tails stay private; only
/// the table root is public.
pub fn sample_asset_table_randomness(
    params: &AssetTableParams,
) -> Result<AssetTableRandomness> {
    let columns = usize::try_from(params.shape.code.n()).map_err(|_| ZkieError::ResourceLimit)?;
    let rows = usize::try_from(params.shape.rows()).map_err(|_| ZkieError::ResourceLimit)?;
    let salt_bytes = params.shape.salt_bytes;
    let tail_len = usize::try_from(params.shape.code.r).map_err(|_| ZkieError::ResourceLimit)?;
    let mut salts = Vec::with_capacity(columns);
    for _ in 0..columns {
        let mut salt = vec![0u8; salt_bytes];
        getrandom::fill(&mut salt).map_err(|_| ZkieError::EntropyUnavailable)?;
        salts.push(salt);
    }
    let mut tails = Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut tail = Vec::with_capacity(tail_len);
        for _ in 0..tail_len {
            let mut bytes = [0u8; 8];
            loop {
                getrandom::fill(&mut bytes).map_err(|_| ZkieError::EntropyUnavailable)?;
                let word = u64::from_le_bytes(bytes);
                if word < MODULUS {
                    tail.push(fp_from_u64(word));
                    break;
                }
            }
        }
        tails.push(tail);
    }
    Ok(AssetTableRandomness { salts, tails })
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::code::CodeParams;
    use zkie_engine::pcs::request::table::{IndexShape, TableDistances};

    fn params() -> AssetTableParams {
        AssetTableParams {
            asset_id: 1,
            path: "asset.bin".into(),
            element_count: 4,
            row_width: 2,
            encoding: "i32-le".into(),
            shape: IndexShape {
                code: CodeParams { log_n: 5, ell: 11, r: 2 },
                v: 2,
                answers: 1,
                salt_bytes: 40,
                distances: TableDistances { radius: 0, later_radius: 0, m_once: 0, m_later: 0 },
            },
            valid_min: -100,
            valid_max: 100,
            layout: AssetTableLayout {
                row_width: 2,
                message_len: 11,
                tail_len: 2,
                pads: 3,
                valid_elements: 4,
                padding_zero: true,
            },
        }
    }

    #[test]
    fn randomness_round_trips_and_rejects_a_mismatched_shape() {
        let params = params();
        let randomness = sample_asset_table_randomness(&params).unwrap();
        let bytes = encode_asset_table_randomness(&randomness).unwrap();
        let decoded = decode_asset_table_randomness(&bytes, &params).unwrap();
        assert_eq!(decoded.salts.len(), randomness.salts.len());
        assert_eq!(decoded.tails.len(), randomness.tails.len());
        let mut wrong = params.clone();
        wrong.shape.code.r += 1;
        assert!(decode_asset_table_randomness(&bytes, &wrong).is_err());
    }

    #[test]
    fn sampled_randomness_has_the_table_shape() {
        let params = params();
        let randomness = sample_asset_table_randomness(&params).unwrap();
        assert_eq!(randomness.salts.len(), params.shape.code.n() as usize);
        assert!(randomness.salts.iter().all(|salt| salt.len() == 40));
        assert_eq!(randomness.tails.len(), params.shape.rows() as usize);
        assert!(randomness.tails.iter().all(|tail| tail.len() == 2));
    }

    #[test]
    fn randomness_set_round_trips_and_rejects_bad_sets() {
        let mut second = params();
        second.asset_id = 2;
        second.path = "other.bin".into();
        let trusted = vec![params(), second];
        let tables = sample_asset_table_inputs(&trusted).unwrap();
        let bytes = encode_asset_table_randomness_set(&tables).unwrap();
        let decoded = decode_asset_table_randomness_set(&bytes, &trusted).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].params.asset_id, trusted[0].asset_id);
        assert_eq!(decoded[1].params.asset_id, trusted[1].asset_id);
        let mut reversed = trusted.clone();
        reversed.reverse();
        let decoded = decode_asset_table_randomness_set(&bytes, &reversed).unwrap();
        assert_eq!(decoded[0].params.asset_id, reversed[0].asset_id);
        assert_eq!(decoded[1].params.asset_id, reversed[1].asset_id);

        assert!(decode_asset_table_randomness_set(&bytes, &trusted[..1]).is_err());
        assert!(decode_asset_table_randomness_set(&bytes, &[]).is_err());
        let mut wrong_shape = trusted.clone();
        wrong_shape[1].shape.code.r += 1;
        assert!(decode_asset_table_randomness_set(&bytes, &wrong_shape).is_err());
        let duplicate = vec![tables[0].clone(), tables[0].clone()];
        let bytes = encode_asset_table_randomness_set(&duplicate).unwrap();
        assert!(decode_asset_table_randomness_set(&bytes, &trusted).is_err());
        let mut unknown = tables.clone();
        unknown[1].params.asset_id = 3;
        let bytes = encode_asset_table_randomness_set(&unknown).unwrap();
        assert!(decode_asset_table_randomness_set(&bytes, &trusted).is_err());
    }
}


/// Encodes the private table randomness for the commit artifact. The bytes
/// are never part of the public bundle.
pub fn encode_asset_table_randomness(randomness: &AssetTableRandomness) -> Result<Vec<u8>> {
    bincode::serialize(randomness).map_err(|_| ZkieError::InvalidEncoding)
}

/// Decodes private table randomness and checks it against the trusted table
/// parameters, so a truncated or mismatched private file is rejected before
/// it can produce a table root.
pub fn decode_asset_table_randomness(
    bytes: &[u8],
    params: &AssetTableParams,
) -> Result<AssetTableRandomness> {
    let randomness: AssetTableRandomness =
        bincode::deserialize(bytes).map_err(|_| ZkieError::InvalidEncoding)?;
    check_asset_table_randomness(&randomness, params)?;
    Ok(randomness)
}

fn check_asset_table_randomness(
    randomness: &AssetTableRandomness,
    params: &AssetTableParams,
) -> Result<()> {
    let columns = usize::try_from(params.shape.code.n()).map_err(|_| ZkieError::ResourceLimit)?;
    let rows = usize::try_from(params.shape.rows()).map_err(|_| ZkieError::ResourceLimit)?;
    let tail_len = usize::try_from(params.shape.code.r).map_err(|_| ZkieError::ResourceLimit)?;
    if randomness.salts.len() != columns
        || randomness
            .salts
            .iter()
            .any(|salt| salt.len() != params.shape.salt_bytes)
        || randomness.tails.len() != rows
        || randomness.tails.iter().any(|tail| tail.len() != tail_len)
    {
        return Err(ZkieError::ShapeMismatch);
    }
    Ok(())
}


/// Samples the private table material from a plan-fixed public seed. The same
/// seed and parameters give the same salts and tails, so the table root is a
/// reproducible function of the trusted asset and the public derivation. The
/// material never enters the public bundle.
pub fn sample_asset_table_randomness_public(
    params: &AssetTableParams,
    seed: zkie_core::types::Digest,
) -> Result<AssetTableRandomness> {
    use zkie_core::common::field::PrimeField64;
    use zkie_engine::pcs::request::field::Fp3;
    use zkie_engine::pcs::request::fold::{ByteMaskCoins, MaskCoins};
    if seed == zkie_core::types::Digest::ZERO {
        return Err(ZkieError::InvalidConfig);
    }
    let columns = usize::try_from(params.shape.code.n()).map_err(|_| ZkieError::ResourceLimit)?;
    let rows = usize::try_from(params.shape.rows()).map_err(|_| ZkieError::ResourceLimit)?;
    let salt_bytes = params.shape.salt_bytes;
    let tail_len = usize::try_from(params.shape.code.r).map_err(|_| ZkieError::ResourceLimit)?;
    let mut coins = ByteMaskCoins::new(seed)?;
    let mut salts = Vec::with_capacity(columns);
    for _ in 0..columns {
        let mut salt = Vec::with_capacity(salt_bytes);
        while salt.len() < salt_bytes {
            let (value, _) = coins.draw()?;
            for coefficient in value.c {
                salt.extend_from_slice(&coefficient.as_canonical_u64().to_le_bytes());
            }
        }
        salt.truncate(salt_bytes);
        salts.push(salt);
    }
    let mut tails = Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut tail = Vec::with_capacity(tail_len);
        for _ in 0..tail_len {
            let (value, _) = coins.draw()?;
            tail.push(value.c[0]);
        }
        tails.push(tail);
    }
    Ok(AssetTableRandomness { salts, tails })
}

/// Samples the private randomness of every trusted table. The result is
/// aligned with `params` and is the input of the table commit.
pub fn sample_asset_table_inputs(params: &[AssetTableParams]) -> Result<Vec<AssetTableInput>> {
    params
        .iter()
        .map(|params| {
            let randomness = sample_asset_table_randomness(params)?;
            Ok(AssetTableInput {
                params: params.clone(),
                randomness,
            })
        })
        .collect()
}

/// Encodes the private randomness of the given tables for the private commit
/// artifact. The bytes are never part of the public bundle.
pub fn encode_asset_table_randomness_set(tables: &[AssetTableInput]) -> Result<Vec<u8>> {
    let entries: Vec<AssetTableRandomnessEntry> = tables
        .iter()
        .map(|table| AssetTableRandomnessEntry {
            asset_id: table.params.asset_id,
            randomness: table.randomness.clone(),
        })
        .collect();
    bincode::serialize(&entries).map_err(|_| ZkieError::InvalidEncoding)
}

/// Decodes the private randomness set against the trusted table list and
/// returns one table input per parameter. The set must match the trusted
/// assets one to one; an unknown, duplicate or missing asset and a
/// shape-mismatched randomness are rejected.
pub fn decode_asset_table_randomness_set(
    bytes: &[u8],
    params: &[AssetTableParams],
) -> Result<Vec<AssetTableInput>> {
    let entries: Vec<AssetTableRandomnessEntry> =
        bincode::deserialize(bytes).map_err(|_| ZkieError::InvalidEncoding)?;
    if entries.len() != params.len() {
        return Err(ZkieError::ShapeMismatch);
    }
    let mut seen = std::collections::BTreeSet::new();
    for entry in &entries {
        if !seen.insert(entry.asset_id) {
            return Err(ZkieError::StatementMismatch);
        }
    }
    let mut tables = Vec::with_capacity(params.len());
    for params in params {
        let entry = entries
            .iter()
            .find(|entry| entry.asset_id == params.asset_id)
            .ok_or(ZkieError::StatementMismatch)?;
        check_asset_table_randomness(&entry.randomness, params)?;
        tables.push(AssetTableInput {
            params: params.clone(),
            randomness: entry.randomness.clone(),
        });
    }
    Ok(tables)
}

/// One read-only row access of the asset mapping, owned so a committed table
/// keeper can hold it beside the encoding table.
pub trait AssetRowReader {
    fn read_row(&self, row: u64, row_width: usize) -> Result<Vec<u64>>;
}

impl AssetRowReader for AssetReader {
    fn read_row(&self, row: u64, row_width: usize) -> Result<Vec<u64>> {
        AssetReader::read_row(self, row, row_width)
    }
}

impl<T: AssetRowReader + ?Sized> AssetRowReader for &T {
    fn read_row(&self, row: u64, row_width: usize) -> Result<Vec<u64>> {
        T::read_row(self, row, row_width)
    }
}

impl<T: AssetRowReader + ?Sized> AssetRowReader for std::sync::Arc<T> {
    fn read_row(&self, row: u64, row_width: usize) -> Result<Vec<u64>> {
        T::read_row(self, row, row_width)
    }
}

/// The streaming row source of one encoded table over the asset mapping. The
/// message keeps the plain asset columns and zero pads; the tail is the private
/// table randomness of that row.
pub struct AssetRowSource<R> {
    reader: R,
    rows: u64,
    row_width: usize,
    valid_elements: u64,
    message_len: usize,
    valid_min: i64,
    valid_max: i64,
    tails: Vec<Vec<Fp>>,
    tail_len: usize,
}

/// Builds the streaming row source of one table from the trusted parameters.
pub fn asset_row_source<R: AssetRowReader>(
    reader: R,
    spec: &ModelAssetSpec,
    params: &AssetTableParams,
    tails: Vec<Vec<Fp>>,
) -> Result<AssetRowSource<R>> {
    Ok(AssetRowSource {
        reader,
        rows: params.shape.rows(),
        row_width: spec.row_width,
        valid_elements: spec.element_count,
        message_len: usize::try_from(params.shape.code.ell).map_err(|_| ZkieError::ResourceLimit)?,
        valid_min: params.valid_min,
        valid_max: params.valid_max,
        tails,
        tail_len: usize::try_from(params.shape.code.r).map_err(|_| ZkieError::ResourceLimit)?,
    })
}

impl<R: AssetRowReader> TableRowSource for AssetRowSource<R> {
    fn rows(&self) -> u64 {
        self.rows
    }

    fn row(&mut self, row: u64, message: &mut [Fp], tail: &mut [Fp]) -> Result<()> {
        if row >= self.rows
            || message.len() != self.message_len
            || tail.len() != self.tail_len
        {
            return Err(ZkieError::ShapeMismatch);
        }
        let values = self.reader.read_row(row, self.row_width)?;
        for slot in message.iter_mut() {
            *slot = fp_from_u64(0);
        }
        let row_start = row
            .checked_mul(self.row_width as u64)
            .ok_or(ZkieError::ResourceLimit)?;
        for (column, value) in values.iter().enumerate() {
            let place = row_start
                .checked_add(column as u64)
                .ok_or(ZkieError::ResourceLimit)?;
            if place >= self.valid_elements {
                break;
            }
            let signed = *value as i64;
            if signed < self.valid_min || signed > self.valid_max {
                return Err(ZkieError::StatementMismatch);
            }
            message[column] = fp_from_signed_word(*value);
        }
        let row_tail = self
            .tails
            .get(row as usize)
            .ok_or(ZkieError::ShapeMismatch)?;
        tail.copy_from_slice(row_tail);
        Ok(())
    }
}

/// Builds one encoded table from the same read-only rows as the raw asset
/// commitment. The provider-side caller keeps `randomness` private.
pub fn build_asset_table(
    reader: &AssetReader,
    model_commitment: Digest,
    spec: &ModelAssetSpec,
    input: &AssetTableInput,
    max_bytes: u64,
) -> Result<AssetTableMeta> {
    let params = &input.params;
    validate_asset_table_params(spec, params)?;
    let shape = params.shape;
    let rows = shape.rows();
    let columns = shape.code.n() as usize;
    if input.randomness.salts.len() != columns
        || input
            .randomness
            .salts
            .iter()
            .any(|salt| salt.len() != shape.salt_bytes)
    {
        return Err(ZkieError::ShapeMismatch);
    }
    if input.randomness.tails.len() as u64 != rows
        || input
            .randomness
            .tails
            .iter()
            .any(|tail| tail.len() as u64 != shape.code.r)
    {
        return Err(ZkieError::ShapeMismatch);
    }
    let configuration = asset_table_configuration(model_commitment, spec, params)?;
    let identity = asset_table_identity(model_commitment, spec, params)?;
    let mut source = asset_row_source(reader, spec, params, input.randomness.tails.clone())?;
    let panel_columns = shape.code.n().min(1024).max(1);
    let table = commit_table_streaming(
        &mut source,
        shape,
        configuration,
        &input.randomness.salts,
        panel_columns,
        max_bytes,
    )?;
    let record = table.reader.record;
    let root = table.reader.root;
    Ok(AssetTableMeta {
        asset_id: spec.asset_id,
        path: spec.path.clone(),
        identity,
        shape,
        configuration,
        root,
        record,
        valid_min: params.valid_min,
        valid_max: params.valid_max,
        valid_elements: spec.element_count,
    })
}

/// Independently checks one committed table against the trusted plan. The
/// root and record are data; the anchor check binds them separately.
pub fn verify_asset_table_meta(
    meta: &AssetTableMeta,
    model_commitment: Digest,
    spec: &ModelAssetSpec,
    params: &AssetTableParams,
) -> Result<()> {
    validate_asset_table_params(spec, params)?;
    let identity = asset_table_identity(model_commitment, spec, params)?;
    let configuration = asset_table_configuration(model_commitment, spec, params)?;
    if meta.asset_id != spec.asset_id
        || meta.path != spec.path
        || meta.identity != identity
        || meta.shape != params.shape
        || meta.configuration != configuration
        || meta.valid_min != params.valid_min
        || meta.valid_max != params.valid_max
        || meta.valid_elements != spec.element_count
        || meta.root == Digest::ZERO
    {
        return Err(ZkieError::StatementMismatch);
    }
    let reader = IndexReader {
        shape: meta.shape,
        configuration: meta.configuration,
        root: meta.root,
        record: meta.record.clone(),
    };
    if reader.fault().is_some() {
        return Err(ZkieError::StatementMismatch);
    }
    Ok(())
}
