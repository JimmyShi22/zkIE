//! Shared model commitment: one anchor/leaf structure for every weight and LUT.
//!
//! Assets are committed once at commit time. A trace opens an asset interval by
//! (asset id, row range) and proves membership in the anchor registry; relation
//! steps reference the same opening instead of re-committing the asset.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use zkie_core::compute::{MemoryGovernor, MemoryRequest};
use zkie_core::error::{Error, Result};
use rayon::prelude::*;
use zkie_core::storage::TensorReader;
use zkie_core::types::{Digest, DType, TensorSpec};
use zkie_engine::compute::cpu::{merkle_build, merkle_verify, MerkleTree};
use zkie_engine::memory::BudgetedMemoryGovernor;
use zkie_engine::storage::mmap::MmapTensorReader;
use zkie_engine::pcs::request::link::{
    make_anchor, prove_link, verify_link, Anchor, AnchorRecord, LeafRecord, LinkProof,
};
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
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedTraceStep {
    pub message: crate::proof::relations::RelationReceiptMessage,
    /// For each relation input index, the index into `SharedTrace::assets`.
    pub external: Vec<Option<usize>>,
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

pub fn verify_shared_trace(
    public: &ModelCommitmentPublic,
    trace: &SharedTrace,
    expected_tokens: &[u32],
) -> Result<Vec<u32>> {
    verify_shared_trace_steps(public, trace, expected_tokens)?;
    Ok(trace.argmax.clone())
}

/// Verifies the trace and returns the recorded final output tensor, which must
/// match the output of the last relation step.
pub fn verify_shared_trace_output(
    public: &ModelCommitmentPublic,
    trace: &SharedTrace,
    expected_tokens: &[u32],
) -> Result<Vec<i64>> {
    let outputs = verify_shared_trace_steps(public, trace, expected_tokens)?;
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
) -> Result<Vec<Vec<Vec<i64>>>> {
    if trace.tokens != expected_tokens {
        return Err(Error::StatementMismatch);
    }
    if trace.output.is_empty() && trace.argmax.len() != expected_tokens.len() {
        return Err(Error::StatementMismatch);
    }
    trace
        .assets
        .par_iter()
        .enumerate()
        .try_for_each(|(asset_index, opening)| {
            public.verify_opening(opening).map_err(|error| {
                eprintln!(
                    "asset opening {} id {} err {:?}",
                    asset_index, opening.asset_id, error
                );
                error
            })
        })?;
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
                step.message
                    .proof
                    .inputs
                    .get(index)
                    .and_then(|openings| openings.first())
                    .map(|opening| opening.oracle)
                    .unwrap_or(crate::proof::binding::OracleId(u64::MAX))
            })
            .collect::<Vec<_>>();
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
        let public_relation = crate::proof::relations::OpRelationPublic {
            opcode,
            config: step.message.config.clone(),
            registry,
            input_oracles,
            output_oracles,
        };
        let mut external_values: Vec<Option<Vec<i64>>> = vec![None; input_count];
        for (input, pool) in step.external.iter().enumerate() {
            if let Some(pool) = pool {
                let opening = trace.assets.get(*pool).ok_or(Error::StatementMismatch)?;
                let mut values = Vec::with_capacity(opening.rows.len() * opening.rows.first().map(|row| row.values.len()).unwrap_or(0));
                for row in &opening.rows {
                    values.extend(row.values.iter().map(|value| *value as i64));
                }
                if external_values[input].replace(values).is_some() {
                    return Err(Error::StatementMismatch);
                }
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
