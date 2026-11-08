//! Shared model commitment: one anchor/leaf structure for every weight and LUT.
//!
//! Assets are committed once at commit time. A trace opens an asset interval by
//! (asset id, row range) and proves membership in the anchor registry; relation
//! steps reference the same opening instead of re-committing the asset.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use zkie_core::compute::{MemoryGovernor, MemoryRequest};
use zkie_core::error::{Error, Result};
use rayon::prelude::*;

use crate::proof::binding::oracle_leaf_i64;
use crate::proof::relations::{ExternalAssetRef, ExternalLayout};
use crate::public_lut::{self, PublicLutKind, PublicLutRef};
use zkie_core::storage::TensorReader;
use zkie_core::types::{Digest, DType, TensorSpec};
use zkie_engine::compute::cpu::{merkle_build, merkle_verify, MerkleTree};
use zkie_engine::memory::BudgetedMemoryGovernor;
use zkie_engine::storage::mmap::MmapTensorReader;
use zkie_engine::pcs::request::chain::KIND_ALGEBRAIC;
use zkie_engine::pcs::request::link::{
    make_anchor, prove_link, verify_link, Anchor, AnchorRecord, LeafRecord, LinkProof,
};
use zkie_engine::primitives::codec::CanonicalWriter;
use zkie_engine::primitives::transcript::{hash_labeled, tag_word, STATEMENT};

use crate::proof::binding::{oracle_leaf, oracle_start};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelAssetSpec {
    pub asset_id: u64,
    pub path: String,
    pub element_count: u64,
    pub row_width: usize,
    pub encoding: String,
}

#[derive(Clone, Debug)]
pub struct AssetCommit {
    pub asset_id: u64,
    pub path: String,
    pub element_count: u64,
    pub row_width: usize,
    pub encoding: String,
    pub start: Digest,
    pub root: Digest,
    pub rows: u64,
    pub leaf_record: LeafRecord,
    pub tree: Arc<MerkleTree>,
}

#[derive(Clone, Debug)]
pub struct ModelCommitment {
    pub anchor: Anchor,
    pub assets: Vec<AssetCommit>,
    pub root: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RowOpening {
    pub row: u64,
    pub values: Vec<u64>,
    pub path: Vec<Digest>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AssetOpening {
    pub asset_id: u64,
    pub link: LinkProof,
    pub rows: Vec<RowOpening>,
    pub start: Digest,
    pub root: Digest,
    pub row_count: u64,
}

fn table_identity(spec: &ModelAssetSpec) -> Digest {
    let mut data = Vec::new();
    data.extend_from_slice(&spec.asset_id.to_le_bytes());
    data.extend_from_slice(spec.path.as_bytes());
    data.extend_from_slice(&spec.element_count.to_le_bytes());
    data.extend_from_slice(&(spec.row_width as u64).to_le_bytes());
    data.extend_from_slice(spec.encoding.as_bytes());
    hash_labeled(tag_word(STATEMENT, 13), 0, None, &data).unwrap_or(Digest::ZERO)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AssetEncoding {
    I32Le,
    I64Le,
}

impl AssetEncoding {
    fn parse(encoding: &str) -> Result<Self> {
        match encoding {
            "i32-le" => Ok(Self::I32Le),
            "i64-le" => Ok(Self::I64Le),
            _ => Err(Error::UnsupportedPrimitive),
        }
    }

    const fn dtype(self) -> DType {
        match self {
            Self::I32Le => DType::I32,
            Self::I64Le => DType::I64,
        }
    }

    const fn element_size(self) -> u64 {
        match self {
            Self::I32Le => 4,
            Self::I64Le => 8,
        }
    }

    fn decode(self, bytes: &[u8]) -> Vec<u64> {
        match self {
            Self::I32Le => bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as u64)
                .collect(),
            Self::I64Le => bytes
                .chunks_exact(8)
                .map(|chunk| {
                    i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]) as u64
                })
                .collect(),
        }
    }
}

fn element_size(encoding: &str) -> Result<u64> {
    Ok(AssetEncoding::parse(encoding)?.element_size())
}

fn asset_window_bytes(asset: &AssetCommit) -> Result<u64> {
    asset
        .element_count
        .checked_mul(element_size(&asset.encoding)?)
        .ok_or(Error::ResourceLimit)
}

/// Read-only view over one committed asset. The file is mapped once; range
/// reads copy only the requested elements, and the mapping's memory window is
/// charged through the permit held by the reader until it is dropped.
pub struct AssetReader {
    reader: MmapTensorReader,
    encoding: AssetEncoding,
    element_count: u64,
}

impl AssetReader {
    fn from_mmap(reader: MmapTensorReader, encoding: AssetEncoding, element_count: u64) -> Self {
        Self {
            reader,
            encoding,
            element_count,
        }
    }

    pub fn open(
        commitment: &ModelCommitment,
        asset_id: u64,
        governor: &dyn MemoryGovernor,
    ) -> Result<Self> {
        let asset = commitment
            .assets
            .iter()
            .find(|asset| asset.asset_id == asset_id)
            .ok_or(Error::StatementMismatch)?;
        let encoding = AssetEncoding::parse(&asset.encoding)?;
        let expected = asset_window_bytes(asset)?;
        let spec = TensorSpec::new(encoding.dtype(), vec![asset.element_count]);
        let permit = governor.acquire(&MemoryRequest {
            mapped_window_bytes: expected,
            ..MemoryRequest::default()
        })?;
        let reader = MmapTensorReader::open(
            commitment.root.join(&asset.path),
            spec,
            asset.start,
            permit,
        )?;
        if reader.mapped_len() as u64 != expected {
            return Err(Error::ShapeMismatch);
        }
        Ok(Self::from_mmap(reader, encoding, asset.element_count))
    }

    /// Read `count` canonical values starting at `start`. Only this range is
    /// materialized.
    pub fn read(&self, start: u64, count: usize) -> Result<Vec<u64>> {
        let byte_len = (count as u64)
            .checked_mul(self.encoding.element_size())
            .ok_or(Error::ResourceLimit)?;
        let mut bytes = vec![0u8; usize::try_from(byte_len).map_err(|_| Error::ResourceLimit)?];
        self.reader.read_range(start, count, &mut bytes)?;
        Ok(self.encoding.decode(&bytes))
    }

    pub fn read_i64(&self, start: u64, count: usize) -> Result<Vec<i64>> {
        Ok(self
            .read(start, count)?
            .into_iter()
            .map(|value| value as i64)
            .collect())
    }

    /// Read one logical row, zero-filling the implicit padding beyond the
    /// declared element count.
    pub fn read_row(&self, row: u64, row_width: usize) -> Result<Vec<u64>> {
        let start = row
            .checked_mul(row_width as u64)
            .ok_or(Error::ResourceLimit)?;
        let mut values = vec![0u64; row_width];
        if start < self.element_count {
            let available = (self.element_count - start).min(row_width as u64) as usize;
            let read = self.read(start, available)?;
            values[..available].copy_from_slice(&read);
        }
        Ok(values)
    }
}

pub fn commit_assets(
    root: &Path,
    model_commitment: Digest,
    parameters_digest: Digest,
    specs: &[ModelAssetSpec],
) -> Result<ModelCommitment> {
    if specs.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let mut assets = Vec::with_capacity(specs.len());
    let mut records = Vec::with_capacity(specs.len());
    for (position, spec) in specs.iter().enumerate() {
        if spec.row_width == 0 || spec.element_count == 0 {
            return Err(Error::InvalidConfig);
        }
        let encoding = AssetEncoding::parse(&spec.encoding)?;
        let expected = spec
            .element_count
            .checked_mul(encoding.element_size())
            .ok_or(Error::ResourceLimit)?;
        let governor = BudgetedMemoryGovernor::new(expected.max(1));
        let permit = governor.acquire(&MemoryRequest {
            mapped_window_bytes: expected,
            ..MemoryRequest::default()
        })?;
        let reader = MmapTensorReader::open(
            root.join(&spec.path),
            TensorSpec::new(encoding.dtype(), vec![spec.element_count]),
            table_identity(spec),
            permit,
        )?;
        if reader.mapped_len() as u64 != expected {
            return Err(Error::ShapeMismatch);
        }
        let reader = AssetReader::from_mmap(reader, encoding, spec.element_count);
        let actual_rows = spec.element_count.div_ceil(spec.row_width as u64);
        let padded_rows = actual_rows.max(1).next_power_of_two();
        let mut digests = Vec::with_capacity(padded_rows as usize);
        for row in 0..padded_rows {
            let values = reader.read_row(row, spec.row_width)?;
            digests.push(oracle_leaf(row, &values));
        }
        let start = oracle_start(&table_identity(spec));
        let tree = merkle_build(&start, &digests)?;
        let leaf_record = LeafRecord {
            leaf_id: position as u64,
            generation: 0,
            table_identity: table_identity(spec),
            leaf_root: tree.root,
        };
        records.push(leaf_record.clone());
        assets.push(AssetCommit {
            asset_id: spec.asset_id,
            path: spec.path.clone(),
            element_count: spec.element_count,
            row_width: spec.row_width,
            encoding: spec.encoding.clone(),
            start,
            root: tree.root,
            rows: padded_rows,
            leaf_record,
            tree: Arc::new(tree),
        });
    }
    let anchor = make_anchor(model_commitment, parameters_digest, &records)?;
    Ok(ModelCommitment {
        anchor,
        assets,
        root: root.to_path_buf(),
    })
}

pub fn open_asset(
    commitment: &ModelCommitment,
    asset_id: u64,
    row_start: u64,
    row_count: u64,
) -> Result<AssetOpening> {
    let reader = open_asset_reader(commitment, asset_id)?;
    open_asset_from_reader(commitment, &reader, asset_id, row_start, row_count)
}

pub fn open_asset_reader(
    commitment: &ModelCommitment,
    asset_id: u64,
) -> Result<AssetReader> {
    let asset = commitment
        .assets
        .iter()
        .find(|asset| asset.asset_id == asset_id)
        .ok_or(Error::StatementMismatch)?;
    let expected = asset_window_bytes(asset)?;
    let governor = BudgetedMemoryGovernor::new(expected.max(1));
    AssetReader::open(commitment, asset_id, &governor)
}

pub fn open_asset_from_reader(
    commitment: &ModelCommitment,
    reader: &AssetReader,
    asset_id: u64,
    row_start: u64,
    row_count: u64,
) -> Result<AssetOpening> {
    let asset = commitment
        .assets
        .iter()
        .find(|asset| asset.asset_id == asset_id)
        .ok_or(Error::StatementMismatch)?;
    if row_start.checked_add(row_count).ok_or(Error::ResourceLimit)? > asset.rows || row_count == 0 {
        return Err(Error::InvalidIndex);
    }
    let link = prove_link(
        &commitment.anchor,
        &commitment
            .assets
            .iter()
            .map(|asset| asset.leaf_record.clone())
            .collect::<Vec<_>>(),
        commitment
            .assets
            .iter()
            .position(|candidate| candidate.asset_id == asset_id)
            .ok_or(Error::StatementMismatch)?,
    )?;
    let mut rows = Vec::with_capacity(row_count as usize);
    for row in row_start..row_start + row_count {
        let values = reader.read_row(row, asset.row_width)?;
        rows.push(RowOpening {
            row,
            values,
            path: asset.tree.open(row as usize)?,
        });
    }
    Ok(AssetOpening {
        asset_id,
        link,
        rows,
        start: asset.start,
        root: asset.root,
        row_count,
    })
}

/// Lazy, range-reading asset store for executors. Each asset is mapped on
/// first use; all mapped windows are charged to one governor and released when
/// the store (and its readers) are dropped.
pub struct AssetStore<'a> {
    commitment: &'a ModelCommitment,
    pool: BTreeMap<String, usize>,
    readers: RefCell<Vec<Option<AssetReader>>>,
    governor: BudgetedMemoryGovernor,
}

impl<'a> AssetStore<'a> {
    pub fn open(commitment: &'a ModelCommitment) -> Result<Self> {
        let mut total = 0u64;
        for asset in &commitment.assets {
            total = total
                .checked_add(asset_window_bytes(asset)?)
                .ok_or(Error::ResourceLimit)?;
        }
        let governor = BudgetedMemoryGovernor::new(
            total.checked_add(1 << 30).ok_or(Error::ResourceLimit)?,
        );
        let mut pool = BTreeMap::new();
        for (index, asset) in commitment.assets.iter().enumerate() {
            if pool.insert(asset.path.clone(), index).is_some() {
                return Err(Error::StatementMismatch);
            }
        }
        let readers = (0..commitment.assets.len()).map(|_| None).collect();
        Ok(Self {
            commitment,
            pool,
            readers: RefCell::new(readers),
            governor,
        })
    }

    pub fn commitment(&self) -> &'a ModelCommitment {
        self.commitment
    }

    pub fn pool(&self, path: &str) -> Result<usize> {
        self.pool.get(path).copied().ok_or(Error::AssetMismatch)
    }

    fn with_reader<T>(
        &self,
        index: usize,
        f: impl FnOnce(&AssetReader) -> Result<T>,
    ) -> Result<T> {
        let mut readers = self.readers.borrow_mut();
        if readers.get(index).map_or(true, |slot| slot.is_none()) {
            let asset = self
                .commitment
                .assets
                .get(index)
                .ok_or(Error::InvalidIndex)?;
            let reader = AssetReader::open(self.commitment, asset.asset_id, &self.governor)?;
            readers[index] = Some(reader);
        }
        f(readers[index]
            .as_ref()
            .ok_or(Error::ProverStateMissing)?)
    }

    pub fn value(&self, path: &str) -> Result<Vec<i64>> {
        let index = self.pool(path)?;
        let count = usize::try_from(self.commitment.assets[index].element_count)
            .map_err(|_| Error::ResourceLimit)?;
        self.with_reader(index, |reader| reader.read_i64(0, count))
    }

    pub fn value_range(&self, path: &str, start: u64, count: usize) -> Result<Vec<i64>> {
        let index = self.pool(path)?;
        self.with_reader(index, |reader| reader.read_i64(start, count))
    }

    pub fn columns(&self, path: &str, cols: usize, c0: usize, c1: usize) -> Result<Vec<i64>> {
        if cols == 0 || c1 <= c0 || c1 > cols {
            return Err(Error::InvalidIndex);
        }
        let index = self.pool(path)?;
        let element_count = self.commitment.assets[index].element_count;
        let rows = usize::try_from(element_count / cols as u64).map_err(|_| Error::ResourceLimit)?;
        let mut out = Vec::with_capacity(rows * (c1 - c0));
        self.with_reader(index, |reader| {
            for row in 0..rows {
                let start = (row as u64) * cols as u64 + c0 as u64;
                out.extend(reader.read_i64(start, c1 - c0)?);
            }
            Ok(())
        })?;
        Ok(out)
    }

    pub fn rows(&self, path: &str, cols: usize, r0: usize, r1: usize) -> Result<Vec<i64>> {
        if cols == 0 || r1 <= r0 {
            return Err(Error::InvalidIndex);
        }
        let start = (r0 as u64)
            .checked_mul(cols as u64)
            .ok_or(Error::ResourceLimit)?;
        let count = (r1 - r0)
            .checked_mul(cols)
            .ok_or(Error::ResourceLimit)?;
        self.value_range(path, start, count)
    }

    pub fn open_asset(&self, index: usize) -> Result<AssetOpening> {
        let asset = self
            .commitment
            .assets
            .get(index)
            .ok_or(Error::InvalidIndex)?;
        self.with_reader(index, |reader| {
            open_asset_from_reader(self.commitment, reader, asset.asset_id, 0, asset.rows)
        })
    }

    /// Open only the assets referenced by a step's external slots. The trace
    /// stores compact indices into the returned list; intervals are the full
    /// asset row range for now (P2 may narrow them).
    pub fn open_referenced(
        &self,
        steps: &mut [SharedTraceStep],
    ) -> Result<Vec<AssetOpening>> {
        for step in steps.iter() {
            if step.external.len() != step.public_luts.len() {
                return Err(Error::ShapeMismatch);
            }
            for (external, public_lut) in step.external.iter().zip(&step.public_luts) {
                if external.is_some() && public_lut.is_some() {
                    return Err(Error::StatementMismatch);
                }
                if let Some(reference) = public_lut {
                    let asset = self
                        .commitment
                        .assets
                        .iter()
                        .find(|asset| asset.asset_id == reference.asset_id)
                        .ok_or(Error::StatementMismatch)?;
                    let kind = public_lut::classify_path(&asset.path)
                        .ok_or(Error::StatementMismatch)?;
                    if kind != reference.kind
                        || reference.valid_len != asset.element_count
                        || reference.padded_len != public_lut::padded_len(reference.valid_len)
                    {
                        return Err(Error::StatementMismatch);
                    }
                }
            }
            for reference in step.external.iter().flatten() {
                let asset = self
                    .commitment
                    .assets
                    .iter()
                    .find(|asset| asset.asset_id == reference.asset_id)
                    .ok_or(Error::StatementMismatch)?;
                if reference.row_count == 0
                    || reference
                        .row_start
                        .checked_add(reference.row_count)
                        .ok_or(Error::ResourceLimit)?
                        > asset.rows
                {
                    return Err(Error::InvalidIndex);
                }
            }
        }
        let used = referenced_asset_last_use(steps)?;
        let mut openings = Vec::with_capacity(used.len());
        for asset_id in used.keys().copied() {
            let index = self
                .commitment
                .assets
                .iter()
                .position(|asset| asset.asset_id == asset_id)
                .ok_or(Error::StatementMismatch)?;
            openings.push(self.open_asset(index)?);
        }
        Ok(openings)
    }

    /// Open every committed asset, preserving the trace's protocol work: each
    /// root is checked against the anchor and each row carries a Merkle path.
    pub fn open_all(&self) -> Result<Vec<AssetOpening>> {
        let mut openings = Vec::with_capacity(self.commitment.assets.len());
        for index in 0..self.commitment.assets.len() {
            openings.push(self.open_asset(index)?);
        }
        Ok(openings)
    }
}


pub fn verify_asset_opening(
    commitment: &ModelCommitment,
    opening: &AssetOpening,
) -> Result<()> {
    let asset = commitment
        .assets
        .iter()
        .find(|asset| asset.asset_id == opening.asset_id)
        .ok_or(Error::StatementMismatch)?;
    if opening.start != asset.start || opening.root != asset.root || opening.row_count == 0 {
        return Err(Error::StatementMismatch);
    }
    verify_link(
        &commitment.anchor.record,
        &opening.link,
        &asset.leaf_record.table_identity,
        &asset.leaf_record.leaf_root,
    )?;
    opening.rows.par_iter().try_for_each(|row| {
        if row.row >= asset.rows {
            return Err(Error::InvalidIndex);
        }
        let leaf = oracle_leaf(row.row, &row.values);
        if !merkle_verify(&opening.start, &opening.root, row.row as usize, &leaf, &row.path)? {
            return Err(Error::InvalidProof);
        }
        Ok(())
    })?;
    Ok(())
}

/// Commit-time anchor record for public artifact publication.
pub fn anchor_record(commitment: &ModelCommitment) -> AnchorRecord {
    commitment.anchor.record.clone()
}

// ---------------------------------------------------------------------------
// Public model commitment and shared trace references.
//
// The public side carries only anchor/leaf metadata (no weight values). A trace
// opens each asset interval once; every relation step maps its external inputs
// to a pool entry instead of re-committing the weight/LUT.

use serde::{Deserialize, Serialize};
use zkie_engine::pcs::request::link::LinkProof as PublicLinkProof;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssetMeta {
    pub asset_id: u64,
    pub path: String,
    pub element_count: u64,
    pub row_width: usize,
    pub start: Digest,
    pub root: Digest,
    pub rows: u64,
    pub leaf_record: LeafRecord,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCommitmentPublic {
    pub anchor: AnchorRecord,
    pub assets: Vec<AssetMeta>,
    /// Trusted port-layout plan generated from the model config at commit
    /// time. The verifier compares every trace port against it.
    #[serde(default)]
    pub trace_plan: Option<TracePlan>,
}

impl ModelCommitmentPublic {
    pub fn from_commitment(commitment: &ModelCommitment) -> Self {
        Self {
            anchor: commitment.anchor.record.clone(),
            assets: commitment
                .assets
                .iter()
                .map(|asset| AssetMeta {
                    asset_id: asset.asset_id,
                    path: asset.path.clone(),
                    element_count: asset.element_count,
                    row_width: asset.row_width,
                    start: asset.start,
                    root: asset.root,
                    rows: asset.rows,
                    leaf_record: asset.leaf_record.clone(),
                })
                .collect(),
            trace_plan: None,
        }
    }

    pub fn verify_opening(&self, opening: &AssetOpening) -> Result<()> {
        let asset = self
            .assets
            .iter()
            .find(|asset| asset.asset_id == opening.asset_id)
            .ok_or(Error::StatementMismatch)?;
        if opening.start != asset.start || opening.root != asset.root || opening.row_count == 0 {
            return Err(Error::StatementMismatch);
        }
        verify_link(
            &self.anchor,
            &opening.link,
            &asset.leaf_record.table_identity,
            &asset.leaf_record.leaf_root,
        )?;
        opening.rows.par_iter().try_for_each(|row| {
            if row.row >= asset.rows {
                return Err(Error::InvalidIndex);
            }
            let leaf = oracle_leaf(row.row, &row.values);
            if !merkle_verify(&opening.start, &opening.root, row.row as usize, &leaf, &row.path)? {
                return Err(Error::InvalidProof);
            }
            Ok(())
        })?;
        Ok(())
    }

    /// Single verification entry for a shared-asset interval: checks the
    /// opening against the public commitment and that the interval is in range.
    /// Values are extracted separately so a verification pass does not retain
    /// every external interval at once. P2 may replace this with a joint
    /// opening verifier.
    pub fn verify_opening_interval(
        &self,
        opening: &AssetOpening,
        interval: &ExternalAssetRef,
    ) -> Result<()> {
        if opening.asset_id != interval.asset_id {
            return Err(Error::StatementMismatch);
        }
        self.verify_interval_layout(interval)?;
        self.verify_opening(opening)
    }

    fn verify_interval_layout(&self, interval: &ExternalAssetRef) -> Result<()> {
        let asset = self
            .assets
            .iter()
            .find(|asset| asset.asset_id == interval.asset_id)
            .ok_or(Error::StatementMismatch)?;
        let end = interval
            .row_start
            .checked_add(interval.row_count)
            .ok_or(Error::ResourceLimit)?;
        if interval.row_count == 0 || end > asset.rows {
            return Err(Error::InvalidIndex);
        }
        match interval.layout {
            ExternalLayout::Rows => {}
            ExternalLayout::Broadcast { rows } => {
                if interval.row_count != 1 || rows == 0 {
                    return Err(Error::InvalidIndex);
                }
            }
            ExternalLayout::Columns {
                source_row_width,
                col_start,
                col_count,
            } => {
                if source_row_width != asset.row_width as u64 || col_count == 0 {
                    return Err(Error::StatementMismatch);
                }
                if col_start
                    .checked_add(col_count)
                    .ok_or(Error::ResourceLimit)?
                    > source_row_width
                {
                    return Err(Error::InvalidIndex);
                }
            }
            ExternalLayout::BroadcastColumns {
                source_row_width,
                col_start,
                col_count,
                rows,
            } => {
                if interval.row_count != 1 || rows == 0 {
                    return Err(Error::InvalidIndex);
                }
                if source_row_width != asset.row_width as u64 || col_count == 0 {
                    return Err(Error::StatementMismatch);
                }
                if col_start
                    .checked_add(col_count)
                    .ok_or(Error::ResourceLimit)?
                    > source_row_width
                {
                    return Err(Error::InvalidIndex);
                }
            }
        }
        Ok(())
    }
}

/// Extract the values of an interval from an already verified opening.
pub fn extract_interval_values(
    opening: &AssetOpening,
    interval: &ExternalAssetRef,
    asset_row_width: u64,
) -> Result<Vec<i64>> {
    if opening.asset_id != interval.asset_id {
        return Err(Error::StatementMismatch);
    }
    let end = interval
        .row_start
        .checked_add(interval.row_count)
        .ok_or(Error::ResourceLimit)?;
    if interval.row_count == 0 {
        return Err(Error::InvalidIndex);
    }
    let shape = interval_shape(interval, asset_row_width)?;
    let width = usize::try_from(asset_row_width).map_err(|_| Error::ResourceLimit)?;
    let available = interval
        .row_count
        .min(opening.rows.len() as u64);
    let available = usize::try_from(available).map_err(|_| Error::ResourceLimit)?;
    let capacity = match shape {
        IntervalShape::Rows => available.checked_mul(width),
        IntervalShape::Broadcast { repeat } => repeat.checked_mul(width),
        IntervalShape::Columns { start, end } => available.checked_mul(end - start),
        IntervalShape::BroadcastColumns { start, end, repeat } => {
            repeat.checked_mul(end - start)
        }
    }
    .ok_or(Error::ResourceLimit)?;
    let mut values = Vec::with_capacity(capacity);
    let mut next = interval.row_start;
    for row in &opening.rows {
        if row.row < interval.row_start {
            continue;
        }
        if row.row >= end {
            break;
        }
        if row.row != next {
            return Err(Error::InvalidIndex);
        }
        next = next.checked_add(1).ok_or(Error::ResourceLimit)?;
        match shape {
            IntervalShape::Rows => {
                values.extend(row.values.iter().map(|value| *value as i64));
            }
            IntervalShape::Columns { start, end: window_end } => {
                if row.values.len() != width {
                    return Err(Error::ShapeMismatch);
                }
                values.extend(
                    row.values[start..window_end]
                        .iter()
                        .map(|value| *value as i64),
                );
            }
            IntervalShape::Broadcast { repeat } => {
                let total = repeat
                    .checked_mul(row.values.len())
                    .ok_or(Error::ResourceLimit)?;
                values.reserve(total);
                for _ in 0..repeat {
                    values.extend(row.values.iter().map(|value| *value as i64));
                }
            }
            IntervalShape::BroadcastColumns {
                start,
                end: window_end,
                repeat,
            } => {
                if row.values.len() != width {
                    return Err(Error::ShapeMismatch);
                }
                let total = repeat
                    .checked_mul(window_end - start)
                    .ok_or(Error::ResourceLimit)?;
                values.reserve(total);
                for _ in 0..repeat {
                    values.extend(
                        row.values[start..window_end]
                            .iter()
                            .map(|value| *value as i64),
                    );
                }
            }
        }
    }
    if next != end {
        return Err(Error::IncompleteArtifact);
    }
    Ok(values)
}

#[derive(Clone, Copy)]
enum IntervalShape {
    Rows,
    Broadcast { repeat: usize },
    Columns { start: usize, end: usize },
    BroadcastColumns { start: usize, end: usize, repeat: usize },
}

fn interval_shape(interval: &ExternalAssetRef, asset_row_width: u64) -> Result<IntervalShape> {
    match interval.layout {
        ExternalLayout::Rows => Ok(IntervalShape::Rows),
        ExternalLayout::Broadcast { rows } => {
            if interval.row_count != 1 || rows == 0 {
                return Err(Error::InvalidIndex);
            }
            Ok(IntervalShape::Broadcast {
                repeat: usize::try_from(rows).map_err(|_| Error::ResourceLimit)?,
            })
        }
        ExternalLayout::Columns {
            source_row_width,
            col_start,
            col_count,
        } => {
            let (start, end) = column_window(
                source_row_width,
                col_start,
                col_count,
                asset_row_width,
            )?;
            Ok(IntervalShape::Columns { start, end })
        }
        ExternalLayout::BroadcastColumns {
            source_row_width,
            col_start,
            col_count,
            rows,
        } => {
            if interval.row_count != 1 || rows == 0 {
                return Err(Error::InvalidIndex);
            }
            let (start, end) = column_window(
                source_row_width,
                col_start,
                col_count,
                asset_row_width,
            )?;
            Ok(IntervalShape::BroadcastColumns {
                start,
                end,
                repeat: usize::try_from(rows).map_err(|_| Error::ResourceLimit)?,
            })
        }
    }
}

fn column_window(
    source_row_width: u64,
    col_start: u64,
    col_count: u64,
    asset_row_width: u64,
) -> Result<(usize, usize)> {
    if source_row_width != asset_row_width || col_count == 0 {
        return Err(Error::StatementMismatch);
    }
    let start = usize::try_from(col_start).map_err(|_| Error::InvalidIndex)?;
    let count = usize::try_from(col_count).map_err(|_| Error::InvalidIndex)?;
    let width = usize::try_from(source_row_width).map_err(|_| Error::ResourceLimit)?;
    let end = start.checked_add(count).ok_or(Error::ResourceLimit)?;
    if end > width {
        return Err(Error::InvalidIndex);
    }
    Ok((start, end))
}

/// Reserved interface for P2 joint openings. The signature exists so P2 can
/// land without reshaping callers; the semantics are intentionally not frozen
/// and the current implementation rejects the call.
pub fn verify_joint_openings(
    _public: &ModelCommitmentPublic,
    _references: &[ExternalAssetRef],
) -> Result<Vec<Vec<i64>>> {
    Err(Error::UnsupportedPrimitive)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedTraceStep {
    pub message: crate::proof::relations::RelationReceiptMessage,
    /// For each relation input index, an explicit shared-asset interval.
    pub external: Vec<Option<ExternalAssetRef>>,
    /// For each relation input index, a public LUT reconstructed by verify.
    pub public_luts: Vec<Option<PublicLutRef>>,
    /// For each relation input index, a previous (step, output) edge.
    pub sources: Vec<Option<(usize, usize)>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedTrace {
    pub tokens: Vec<u32>,
    pub assets: Vec<AssetOpening>,
    pub steps: Vec<SharedTraceStep>,
    pub argmax: Vec<u32>,
    #[serde(default)]
    pub output: Vec<i64>,
}

/// Deterministic per-asset last consuming step. Derived only from the ordered
/// external references, with no serialized lifetime field.
pub fn referenced_asset_last_use(
    steps: &[SharedTraceStep],
) -> Result<BTreeMap<u64, u64>> {
    let mut used = BTreeMap::new();
    for (step_index, step) in steps.iter().enumerate() {
        for reference in step.external.iter().flatten() {
            let entry = used
                .entry(reference.asset_id)
                .or_insert(step_index as u64);
            if (step_index as u64) > *entry {
                *entry = step_index as u64;
            }
        }
    }
    Ok(used)
}

pub fn asset_last_use(trace: &SharedTrace) -> Result<BTreeMap<u64, u64>> {
    referenced_asset_last_use(&trace.steps)
}

/// Structural consistency check: the derived last use must equal the expected
/// reference-set maximum; an earlier declaration is rejected.
pub fn check_asset_last_use(
    trace: &SharedTrace,
    expected: &BTreeMap<u64, u64>,
) -> Result<()> {
    if asset_last_use(trace)? != *expected {
        return Err(Error::StatementMismatch);
    }
    Ok(())
}

/// Expected provenance of one relation input port.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanInput {
    Local,
    Source { step: usize, port: usize },
    External(ExternalAssetRef),
    PublicLut(PublicLutRef),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TracePlanStep {
    pub opcode: u16,
    pub config: Vec<i64>,
    pub inputs: Vec<PlanInput>,
    pub input_lengths: Vec<usize>,
    pub output_lengths: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TracePlan {
    pub steps: Vec<TracePlanStep>,
}

/// Capture the trusted port-layout plan from a trace produced under the
/// trusted model config (commit-time execution).
pub fn trace_plan_from_trace(trace: &SharedTrace) -> TracePlan {
    let steps = trace
        .steps
        .iter()
        .map(|step| {
            let input_count = step.external.len();
            let inputs = (0..input_count)
                .map(|index| {
                    if let Some(reference) = step.external[index] {
                        PlanInput::External(reference)
                    } else if let Some(reference) = step.public_luts[index] {
                        PlanInput::PublicLut(reference)
                    } else if let Some((source, port)) = step.sources[index] {
                        PlanInput::Source {
                            step: source,
                            port,
                        }
                    } else {
                        PlanInput::Local
                    }
                })
                .collect();
            TracePlanStep {
                opcode: step.message.opcode,
                config: step.message.config.clone(),
                inputs,
                input_lengths: step.message.proof.lengths.inputs.clone(),
                output_lengths: step.message.proof.lengths.outputs.clone(),
            }
        })
        .collect();
    TracePlan { steps }
}

fn verify_trace_plan(
    public: &ModelCommitmentPublic,
    trace: &SharedTrace,
) -> Result<()> {
    let plan = public
        .trace_plan
        .as_ref()
        .ok_or(Error::StatementMismatch)?;
    if plan.steps.len() != trace.steps.len() {
        return Err(Error::StatementMismatch);
    }
    for (plan_step, step) in plan.steps.iter().zip(&trace.steps) {
        if plan_step.opcode != step.message.opcode
            || plan_step.config != step.message.config
        {
            return Err(Error::StatementMismatch);
        }
        let input_count = plan_step.inputs.len();
        if step.external.len() != input_count
            || step.public_luts.len() != input_count
            || step.sources.len() != input_count
            || step.message.proof.lengths.inputs.len() != input_count
        {
            return Err(Error::ShapeMismatch);
        }
        for index in 0..input_count {
            let actual = match (
                step.external[index],
                step.public_luts[index],
                step.sources[index],
            ) {
                (Some(reference), None, None) => PlanInput::External(reference),
                (None, Some(reference), None) => PlanInput::PublicLut(reference),
                (None, None, Some((source, port))) => PlanInput::Source {
                    step: source,
                    port,
                },
                (None, None, None) => PlanInput::Local,
                _ => return Err(Error::StatementMismatch),
            };
            if actual != plan_step.inputs[index]
                || step.message.proof.lengths.inputs[index] != plan_step.input_lengths[index]
            {
                return Err(Error::StatementMismatch);
            }
        }
        if step.message.proof.lengths.outputs != plan_step.output_lengths {
            return Err(Error::StatementMismatch);
        }
    }
    Ok(())
}

const PREFIX_FORMAT_VERSION: u64 = 1;
const PREFIX_TAG: u64 = 21;

/// Canonical prefix absorbed by every per-step relation label. The order is
/// statement, model commitment, public LUT identities, graph identity, then
/// the oracle commitments. No proof field is added: the prover and verifier
/// derive the same prefix from public data and the label binds it.
pub fn trace_prefix(
    public: &ModelCommitmentPublic,
    statement_digest: &Digest,
    graph_identity: &Digest,
) -> Result<Digest> {
    let mut writer = CanonicalWriter::new();
    writer.put_u64(PREFIX_FORMAT_VERSION);
    writer.put_digest(statement_digest);
    writer.put_digest(&public.anchor.anchor_id);
    writer.put_digest(&public.anchor.model_commitment);
    writer.put_digest(&public.anchor.parameters_digest);
    writer.put_digest(&public.anchor.registry_root);
    writer.put_u64(public.anchor.leaf_count);
    let mut assets: Vec<&AssetMeta> = public.assets.iter().collect();
    assets.sort_by_key(|asset| asset.asset_id);
    let luts: Vec<(&AssetMeta, PublicLutKind)> = assets
        .iter()
        .filter_map(|asset| {
            public_lut::classify_path(&asset.path).map(|kind| (*asset, kind))
        })
        .collect();
    writer.put_u64(luts.len() as u64);
    for (asset, kind) in luts {
        writer.put_u64(asset.asset_id);
        writer.put_u64(kind.tag() as u64);
        writer.put_u64(asset.element_count);
        writer.put_u64(public_lut::padded_len(asset.element_count));
        writer.put_digest(&asset.root);
        writer.put_digest(&asset.start);
    }
    writer.put_digest(graph_identity);
    writer.put_u64(assets.len() as u64);
    for asset in assets {
        writer.put_u64(asset.asset_id);
        writer.put_digest(&asset.root);
        writer.put_digest(&asset.start);
        writer.put_u64(asset.rows);
        writer.put_u64(asset.row_width as u64);
    }
    hash_labeled(
        tag_word(STATEMENT, PREFIX_TAG),
        0,
        None,
        &writer.into_bytes(),
    )
}

pub fn verify_shared_trace(
    public: &ModelCommitmentPublic,
    trace: &SharedTrace,
    expected_tokens: &[u32],
    prefix: &Digest,
) -> Result<Vec<u32>> {
    verify_shared_trace_steps(public, trace, expected_tokens, prefix)?;
    Ok(trace.argmax.clone())
}

/// Verifies the trace and returns the recorded final output tensor, which must
/// match the output of the last relation step.
pub fn verify_shared_trace_output(
    public: &ModelCommitmentPublic,
    trace: &SharedTrace,
    expected_tokens: &[u32],
    prefix: &Digest,
) -> Result<Vec<i64>> {
    let outputs =
        verify_shared_trace_steps(public, trace, expected_tokens, prefix)?;
    if trace.output.is_empty() {
        return Err(Error::StatementMismatch);
    }
    let last = outputs.last().ok_or(Error::StatementMismatch)?;
    let mut flat = Vec::new();
    for tensor in last {
        flat.extend_from_slice(tensor);
    }
    if flat != trace.output {
        return Err(Error::StatementMismatch);
    }
    Ok(flat)
}

fn verify_shared_trace_steps(
    public: &ModelCommitmentPublic,
    trace: &SharedTrace,
    expected_tokens: &[u32],
    prefix: &Digest,
) -> Result<Vec<Vec<Vec<i64>>>> {
    if trace.tokens != expected_tokens {
        return Err(Error::StatementMismatch);
    }
    if trace.steps.is_empty() {
        return Err(Error::StatementMismatch);
    }
    // Every port layout, slice and source is checked against the plan captured
    // from the trusted model config at commit time.
    verify_trace_plan(public, trace)?;
    // StepId is implicit from the canonical ordered list: graph_id is the
    // committed model id and the ordinal is the position. Forward or self
    // references would mean the list was reordered; they are rejected.
    for (index, step) in trace.steps.iter().enumerate() {
        for source in step.sources.iter().flatten() {
            if source.0 >= index {
                return Err(Error::StatementMismatch);
            }
        }
    }
    if trace.output.is_empty() && trace.argmax.len() != expected_tokens.len() {
        return Err(Error::StatementMismatch);
    }
    let mut openings_by_id: BTreeMap<u64, &AssetOpening> = BTreeMap::new();
    for opening in &trace.assets {
        if openings_by_id.insert(opening.asset_id, opening).is_some() {
            return Err(Error::StatementMismatch);
        }
    }
    let mut referenced: BTreeSet<ExternalAssetRef> = BTreeSet::new();
    let mut public_lut_refs: BTreeMap<u64, PublicLutRef> = BTreeMap::new();
    for step in &trace.steps {
        let input_count = step.message.proof.lengths.inputs.len();
        if step.external.len() != input_count || step.public_luts.len() != input_count {
            return Err(Error::ShapeMismatch);
        }
        let mut expected: Vec<usize> = Vec::new();
        for index in 0..input_count {
            let external = step.external[index].is_some();
            let public_lut = step.public_luts[index].is_some();
            if external && public_lut {
                return Err(Error::StatementMismatch);
            }
            if external || public_lut {
                expected.push(index);
            } else if step
                .message
                .proof
                .inputs
                .get(index)
                .map(|openings| openings.is_empty())
                .unwrap_or(true)
            {
                return Err(Error::StatementMismatch);
            }
        }
        if step.message.external_inputs != expected {
            return Err(Error::StatementMismatch);
        }
        for slot in &step.external {
            if let Some(reference) = slot {
                if !openings_by_id.contains_key(&reference.asset_id) {
                    return Err(Error::StatementMismatch);
                }
                referenced.insert(*reference);
            }
        }
        for reference in step.public_luts.iter().flatten() {
            if openings_by_id.contains_key(&reference.asset_id) {
                return Err(Error::StatementMismatch);
            }
            let meta = public
                .assets
                .iter()
                .find(|asset| asset.asset_id == reference.asset_id)
                .ok_or(Error::StatementMismatch)?;
            let kind = public_lut::classify_path(&meta.path).ok_or(Error::StatementMismatch)?;
            if kind != reference.kind
                || reference.valid_len != meta.element_count
                || reference.padded_len != public_lut::padded_len(reference.valid_len)
                || meta.rows != 1
                || meta.row_width as u64 != meta.element_count
            {
                return Err(Error::StatementMismatch);
            }
            if let Some(existing) = public_lut_refs.insert(reference.asset_id, *reference) {
                if existing != *reference {
                    return Err(Error::StatementMismatch);
                }
            }
        }
        let mut expected_suffix = Vec::new();
        for reference in step.public_luts.iter().flatten() {
            let meta = public
                .assets
                .iter()
                .find(|asset| asset.asset_id == reference.asset_id)
                .ok_or(Error::StatementMismatch)?;
            expected_suffix.extend(public_lut::identity_words(reference, &meta.root));
        }
        if !expected_suffix.is_empty()
            && !step.message.config.ends_with(&expected_suffix)
        {
            return Err(Error::StatementMismatch);
        }
    }
    for (asset_id, reference) in &public_lut_refs {
        let meta = public
            .assets
            .iter()
            .find(|asset| asset.asset_id == *asset_id)
            .ok_or(Error::StatementMismatch)?;
        let values = public_lut::values(reference)?;
        let generated_root = oracle_leaf_i64(0, values.as_ref());
        if generated_root != meta.root {
            return Err(Error::StatementMismatch);
        }
    }
    let referenced_ids: BTreeSet<u64> = referenced.iter().map(|reference| reference.asset_id).collect();
    if referenced_ids.len() != openings_by_id.len()
        || openings_by_id
            .keys()
            .any(|asset_id| !referenced_ids.contains(asset_id))
    {
        return Err(Error::StatementMismatch);
    }
    // Lifetime is derived from the ordered reference structure; there is no
    // serialized declaration to trust. The map is retained for verification
    // and later bounded-lifetime use.
    let _last_use = referenced_asset_last_use(&trace.steps)?;
    openings_by_id
        .par_iter()
        .try_for_each(|(_, opening)| public.verify_opening(opening))?;
    let verified: std::sync::Mutex<BTreeSet<ExternalAssetRef>> =
        std::sync::Mutex::new(BTreeSet::new());
    referenced
        .par_iter()
        .try_for_each(|reference| -> Result<()> {
            let opening = openings_by_id
                .get(&reference.asset_id)
                .ok_or(Error::StatementMismatch)?;
            if opening.asset_id != reference.asset_id {
                return Err(Error::StatementMismatch);
            }
            public.verify_interval_layout(reference)?;
            verified
                .lock()
                .map_err(|_| Error::SynchronizationFailed)?
                .insert(*reference);
            Ok(())
        })?;
    let verified = verified
        .into_inner()
        .map_err(|_| Error::SynchronizationFailed)?;
    let mut outputs: Vec<Vec<Vec<i64>>> = Vec::with_capacity(trace.steps.len());
    for (step_index, step) in trace.steps.iter().enumerate() {
        let mut registry = crate::proof::binding::OracleRegistry::new();
        for (root, start, count) in &step.message.roots {
            registry.insert(*root, *start, *count)?;
        }
        let input_count = step.message.proof.lengths.inputs.len();
        let output_count = step.message.proof.lengths.outputs.len();
        let input_oracles = (0..input_count)
            .map(|index| {
                if step.external[index].is_some() || step.public_luts[index].is_some() {
                    Ok(None)
                } else {
                    step.message
                        .proof
                        .inputs
                        .get(index)
                        .and_then(|openings| openings.first())
                        .map(|opening| Some(opening.oracle))
                        .ok_or(Error::StatementMismatch)
                }
            })
            .collect::<Result<Vec<Option<crate::proof::binding::OracleId>>>>()?;
        let output_oracles = (0..output_count)
            .map(|index| {
                step.message
                    .proof
                    .outputs
                    .get(index)
                    .and_then(|openings| openings.first())
                    .map(|opening| opening.oracle)
                    .ok_or(Error::StatementMismatch)
            })
            .collect::<Result<Vec<_>>>()?;
        let opcode = crate::op::opcode_from_u16(step.message.opcode)?;
        let expected_start = oracle_start(&crate::proof::relations::relation_label(
            prefix,
            opcode,
            &step.message.config,
        ));
        for (_, start, _) in &step.message.roots {
            if *start != expected_start {
                return Err(Error::StatementMismatch);
            }
        }
        let public_relation = crate::proof::relations::OpRelationPublic {
            opcode,
            config: step.message.config.clone(),
            registry,
            input_oracles,
            output_oracles,
        };
        let mut external_values: Vec<Option<Vec<i64>>> = vec![None; input_count];
        for (input, reference) in step.external.iter().enumerate() {
            if let Some(reference) = reference {
                if !verified.contains(reference) {
                    return Err(Error::StatementMismatch);
                }
                let opening = openings_by_id
                    .get(&reference.asset_id)
                    .ok_or(Error::StatementMismatch)?;
                let meta = public
                    .assets
                    .iter()
                    .find(|asset| asset.asset_id == reference.asset_id)
                    .ok_or(Error::StatementMismatch)?;
                external_values[input] = Some(extract_interval_values(
                    opening,
                    reference,
                    meta.row_width as u64,
                )?);
            }
        }
        for (input, reference) in step.public_luts.iter().enumerate() {
            if let Some(reference) = reference {
                let values = public_lut::values(reference)?;
                external_values[input] = Some((*values).clone());
            }
        }
        crate::proof::relations::verify_op_relation_with_external(
            &public_relation,
            &step.message.proof,
            &step.message.config,
            &external_values,
        ).map_err(|error| { eprintln!("shared step {} op {} err {:?} ex {:?} lens {:?}", step_index, step.message.opcode, error, step.external, step.message.proof.lengths.inputs); error })?;
        let inputs = crate::proof::relations::proof_input_values(&step.message.proof);
        for (index, source) in step.sources.iter().enumerate() {
            if let Some((step_index, output_index)) = source {
                let expected = outputs
                    .get(*step_index)
                    .and_then(|values| values.get(*output_index))
                    .ok_or(Error::StatementMismatch)?;
                if inputs.get(index) != Some(expected) {
                    eprintln!("edge mismatch step {} op {} input {} source {:?} own_len {:?} exp_len {}", step_index, step.message.opcode, index, source, inputs.get(index).map(Vec::len), expected.len());
                    return Err(Error::StatementMismatch);
                }
            }
        }
        outputs.push(crate::proof::relations::proof_output_values(&step.message.proof));
    }
    Ok(outputs)
}
