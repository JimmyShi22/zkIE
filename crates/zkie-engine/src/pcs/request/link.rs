//! Anchor, leaf and link records.
//!
//! The anchor is an independent record of the model commitment and a registry
//! tree over leaf records. A link proves that a leaf record is a member of the
//! anchor's registry and that its table identity and leaf root are the expected
//! ones. It never compares two roots for equality.

use std::collections::BTreeSet;

use zkie_core::common::codec::DecodeLimits;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::compute::cpu::MerkleTree;
use crate::primitives::codec::{CanonicalReader, CanonicalWriter};
use crate::primitives::transcript::{
    hash_labeled, tag_word, BINDING_NODE, LEAF_BINDING, STATEMENT,
};

pub const LINK_FORMAT_VERSION: u16 = 1;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LeafRecord {
    pub leaf_id: u64,
    pub generation: u64,
    pub table_identity: Digest,
    pub leaf_root: Digest,
}

impl LeafRecord {
    pub fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(LINK_FORMAT_VERSION as u64);
        writer.put_u64(self.leaf_id);
        writer.put_u64(self.generation);
        writer.put_digest(&self.table_identity);
        writer.put_digest(&self.leaf_root);
        writer.into_bytes()
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        if reader.get_u64()? != LINK_FORMAT_VERSION as u64 {
            return Err(Error::UnsupportedVersion);
        }
        let record = Self {
            leaf_id: reader.get_u64()?,
            generation: reader.get_u64()?,
            table_identity: reader.get_digest()?,
            leaf_root: reader.get_digest()?,
        };
        reader.finish()?;
        Ok(record)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AnchorRecord {
    pub anchor_id: Digest,
    pub model_commitment: Digest,
    pub parameters_digest: Digest,
    pub registry_root: Digest,
    pub leaf_count: u64,
}

impl AnchorRecord {
    pub fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(LINK_FORMAT_VERSION as u64);
        writer.put_digest(&self.anchor_id);
        writer.put_digest(&self.model_commitment);
        writer.put_digest(&self.parameters_digest);
        writer.put_digest(&self.registry_root);
        writer.put_u64(self.leaf_count);
        writer.into_bytes()
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        if reader.get_u64()? != LINK_FORMAT_VERSION as u64 {
            return Err(Error::UnsupportedVersion);
        }
        let record = Self {
            anchor_id: reader.get_digest()?,
            model_commitment: reader.get_digest()?,
            parameters_digest: reader.get_digest()?,
            registry_root: reader.get_digest()?,
            leaf_count: reader.get_u64()?,
        };
        reader.finish()?;
        Ok(record)
    }

    pub fn registry_start(&self) -> Result<Digest> {
        let mut data = Vec::with_capacity(64);
        data.extend_from_slice(&self.model_commitment.0);
        data.extend_from_slice(&self.parameters_digest.0);
        hash_labeled(tag_word(STATEMENT, 0), 0, None, &data)
    }
}

pub fn leaf_binding(parameters_digest: &Digest, record: &LeafRecord) -> Result<Digest> {
    hash_labeled(
        tag_word(LEAF_BINDING, 0),
        record.leaf_id,
        Some(parameters_digest),
        &record.encode(),
    )
}

fn binding_node(start: &Digest, left: &Digest, right: &Digest) -> Result<Digest> {
    let mut data = Vec::with_capacity(64);
    data.extend_from_slice(&left.0);
    data.extend_from_slice(&right.0);
    hash_labeled(tag_word(BINDING_NODE, 0), 0, Some(start), &data)
}

fn build_registry(start: &Digest, bindings: &[Digest]) -> Result<MerkleTree> {
    if bindings.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let mut padded = 1usize;
    while padded < bindings.len() {
        padded = padded.checked_mul(2).ok_or(Error::ResourceLimit)?;
    }
    let mut level = bindings.to_vec();
    level.resize(padded, Digest::ZERO);
    let mut levels = vec![level];
    while levels.last().expect("one level").len() > 1 {
        let previous = levels.last().expect("one level");
        let mut next = Vec::with_capacity(previous.len() / 2);
        for pair in previous.chunks_exact(2) {
            next.push(binding_node(start, &pair[0], &pair[1])?);
        }
        levels.push(next);
    }
    let root = levels.last().expect("one level")[0];
    Ok(MerkleTree {
        root,
        leaf_count: bindings.len(),
        padded_count: padded,
        levels,
    })
}

#[derive(Clone, Debug)]
pub struct Anchor {
    pub record: AnchorRecord,
    pub tree: MerkleTree,
    pub start: Digest,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LinkProof {
    pub anchor_id: Digest,
    pub registry_index: u64,
    pub record: LeafRecord,
    pub binding: Digest,
    pub path: Vec<Digest>,
}

pub fn make_anchor(
    model_commitment: Digest,
    parameters_digest: Digest,
    records: &[LeafRecord],
) -> Result<Anchor> {
    if records.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let mut ids = BTreeSet::new();
    for record in records {
        if !ids.insert(record.leaf_id) {
            return Err(Error::StatementMismatch);
        }
    }
    let mut header = AnchorRecord {
        anchor_id: Digest::ZERO,
        model_commitment,
        parameters_digest,
        registry_root: Digest::ZERO,
        leaf_count: records.len() as u64,
    };
    let start = header.registry_start()?;
    let bindings: Vec<Digest> = records
        .iter()
        .map(|record| leaf_binding(&parameters_digest, record))
        .collect::<Result<_>>()?;
    let tree = build_registry(&start, &bindings)?;
    header.registry_root = tree.root;
    header.anchor_id = anchor_id(&header)?;
    Ok(Anchor {
        record: header,
        tree,
        start,
    })
}

pub fn anchor_id(anchor: &AnchorRecord) -> Result<Digest> {
    let mut data = Vec::with_capacity(128);
    data.extend_from_slice(&anchor.model_commitment.0);
    data.extend_from_slice(&anchor.parameters_digest.0);
    data.extend_from_slice(&anchor.registry_root.0);
    data.extend_from_slice(&anchor.leaf_count.to_le_bytes());
    hash_labeled(tag_word(STATEMENT, 0), 1, None, &data)
}

pub fn prove_link(anchor: &Anchor, records: &[LeafRecord], index: usize) -> Result<LinkProof> {
    if anchor.record.anchor_id != anchor_id(&anchor.record)? {
        return Err(Error::ArtifactIntegrityMismatch);
    }
    if index >= records.len() || records.len() as u64 != anchor.record.leaf_count {
        return Err(Error::InvalidIndex);
    }
    let record = records[index].clone();
    let binding = leaf_binding(&anchor.record.parameters_digest, &record)?;
    if anchor
        .tree
        .levels
        .first()
        .and_then(|level| level.get(index))
        .copied()
        != Some(binding)
    {
        return Err(Error::ArtifactIntegrityMismatch);
    }
    Ok(LinkProof {
        anchor_id: anchor.record.anchor_id,
        registry_index: index as u64,
        record,
        binding,
        path: anchor.tree.open(index)?,
    })
}

/// Verify a link against an anchor and the caller's expected table identity
/// and leaf root. Membership, record fields and the anchor identity are all
/// checked; two roots are never merely compared.
pub fn verify_link(
    anchor: &AnchorRecord,
    proof: &LinkProof,
    expected_table_identity: &Digest,
    expected_leaf_root: &Digest,
) -> Result<()> {
    if anchor.anchor_id != anchor_id(anchor)? || proof.anchor_id != anchor.anchor_id {
        return Err(Error::ArtifactIntegrityMismatch);
    }
    if proof.record.table_identity != *expected_table_identity
        || proof.record.leaf_root != *expected_leaf_root
    {
        return Err(Error::StatementMismatch);
    }
    if proof.record.leaf_id >= anchor.leaf_count
        || proof.registry_index >= anchor.leaf_count
    {
        return Err(Error::InvalidIndex);
    }
    let binding = leaf_binding(&anchor.parameters_digest, &proof.record)?;
    if binding != proof.binding {
        return Err(Error::ArtifactIntegrityMismatch);
    }
    let start = anchor.registry_start()?;
    let mut current = binding;
    let mut position = proof.registry_index as usize;
    for sibling in &proof.path {
        current = if position & 1 == 0 {
            binding_node(&start, &current, sibling)?
        } else {
            binding_node(&start, sibling, &current)?
        };
        position >>= 1;
    }
    if current != anchor.registry_root {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

impl LinkProof {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(LINK_FORMAT_VERSION as u64);
        writer.put_digest(&self.anchor_id);
        writer.put_u64(self.registry_index);
        writer.put_bytes(&self.record.encode());
        writer.put_digest(&self.binding);
        writer.put_u64(self.path.len() as u64);
        for digest in &self.path {
            writer.put_digest(digest);
        }
        Ok(writer.into_bytes())
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        if reader.get_u64()? != LINK_FORMAT_VERSION as u64 {
            return Err(Error::UnsupportedVersion);
        }
        let anchor_id = reader.get_digest()?;
        let registry_index = reader.get_u64()?;
        let record = LeafRecord::decode(reader.get_bytes()?)?;
        let binding = reader.get_digest()?;
        let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?;
        if count > 64 {
            return Err(Error::InvalidEncoding);
        }
        let mut path = Vec::with_capacity(count);
        for _ in 0..count {
            path.push(reader.get_digest()?);
        }
        reader.finish()?;
        Ok(Self {
            anchor_id,
            registry_index,
            record,
            binding,
            path,
        })
    }
}
