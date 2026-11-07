//! Canonical public commit records and their artifact persistence.

use std::fs;
use std::path::Path;

use zkie_core::artifact::{
    digest_bytes, ArtifactFile, ArtifactKind, ArtifactManifest, PrivacyMode,
};
use zkie_core::common::codec::DecodeLimits;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::artifact::{load_manifest, verify_artifact, ArtifactPublisher};
use crate::pcs::request::link::{
    verify_link, AnchorRecord, LeafRecord, LinkProof, LINK_FORMAT_VERSION,
};
use crate::pcs::request::merkle::ColumnTreeSpec;
use crate::pcs::request::wf::WfCommitment;
use crate::primitives::codec::{CanonicalReader, CanonicalWriter};

pub const COMMIT_FORMAT_VERSION: u16 = 1;
const ANCHOR_PATH: &str = "public/anchor.bin";
const LEAF_PATH: &str = "public/leaf.bin";
const LINK_PATH: &str = "public/link.proof";
const COLUMN_PATH: &str = "public/column.bin";
const WF_PATH: &str = "public/wf.bin";

#[derive(Clone, Debug)]
pub struct PublicCommit {
    pub anchor: AnchorRecord,
    pub leaf: LeafRecord,
    pub link: LinkProof,
    pub column_spec: ColumnTreeSpec,
    pub column_root: Digest,
    pub wf: WfCommitment,
}

pub fn encode_column_spec(spec: &ColumnTreeSpec, root: &Digest) -> Vec<u8> {
    let mut writer = CanonicalWriter::new();
    writer.put_u64(COMMIT_FORMAT_VERSION as u64);
    writer.put_digest(&spec.table_identity);
    writer.put_digest(&spec.parameters_digest);
    writer.put_u64(spec.tree as u64);
    writer.put_u64(spec.symbol_count as u64);
    writer.put_u64(spec.salt_bytes as u64);
    writer.put_digest(root);
    writer.into_bytes()
}

pub fn decode_column_spec(bytes: &[u8]) -> Result<(ColumnTreeSpec, Digest)> {
    let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
    if reader.get_u64()? != COMMIT_FORMAT_VERSION as u64 {
        return Err(Error::UnsupportedVersion);
    }
    let table_identity = reader.get_digest()?;
    let parameters_digest = reader.get_digest()?;
    let tree = u16::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?;
    let symbol_count =
        usize::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?;
    let salt_bytes = usize::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?;
    let root = reader.get_digest()?;
    reader.finish()?;
    Ok((
        ColumnTreeSpec::new(table_identity, parameters_digest, tree, symbol_count, salt_bytes)?,
        root,
    ))
}

pub fn encode_wf(commitment: &WfCommitment) -> Vec<u8> {
    let mut writer = CanonicalWriter::new();
    writer.put_u64(COMMIT_FORMAT_VERSION as u64);
    writer.put_digest(&commitment.root);
    writer.put_digest(&commitment.table_identity);
    writer.put_digest(&commitment.parameters_digest);
    writer.put_u64(commitment.count);
    writer.put_u64(commitment.params.bound);
    writer.put_u64(commitment.params.base_bits as u64);
    writer.into_bytes()
}

pub fn decode_wf(bytes: &[u8]) -> Result<WfCommitment> {
    let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
    if reader.get_u64()? != COMMIT_FORMAT_VERSION as u64 {
        return Err(Error::UnsupportedVersion);
    }
    let root = reader.get_digest()?;
    let table_identity = reader.get_digest()?;
    let parameters_digest = reader.get_digest()?;
    let count = reader.get_u64()?;
    let bound = reader.get_u64()?;
    let base_bits = u32::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?;
    reader.finish()?;
    Ok(WfCommitment {
        root,
        table_identity,
        parameters_digest,
        count,
        params: crate::pcs::request::wf::WfParams { bound, base_bits },
    })
}

pub fn encode_public(public: &PublicCommit) -> Result<Vec<(String, Vec<u8>)>> {
    let anchor = public.anchor.encode();
    let leaf = public.leaf.encode();
    let link = public.link.encode()?;
    let column = encode_column_spec(&public.column_spec, &public.column_root);
    let wf = encode_wf(&public.wf);
    Ok(vec![
        (ANCHOR_PATH.to_owned(), anchor),
        (LEAF_PATH.to_owned(), leaf),
        (LINK_PATH.to_owned(), link),
        (COLUMN_PATH.to_owned(), column),
        (WF_PATH.to_owned(), wf),
    ])
}

pub fn publish_commit(dir: &Path, public: &PublicCommit) -> Result<()> {
    let files = encode_public(public)?;
    let manifest_files: Vec<ArtifactFile> = files
        .iter()
        .map(|(path, bytes)| {
            ArtifactFile::new(path.as_str(), bytes.len() as u64, digest_bytes(bytes))
        })
        .collect();
    let manifest = ArtifactManifest {
        artifact_schema_version: 1,
        artifact_kind: ArtifactKind::CommitPublic,
        protocol_id: public.column_spec.parameters_digest,
        protocol_version: 1,
        backend_id: 1,
        backend_format_version: 1,
        model_commitment_id: public.anchor.model_commitment,
        parameters_digest: public.column_spec.parameters_digest,
        model_config_digest: public.leaf.table_identity,
        quantization_digest: public.wf.root,
        tables_digest: public.column_root,
        privacy_mode: PrivacyMode::Hidden,
        files: manifest_files,
    };
    let refs: Vec<(&str, &[u8])> = files
        .iter()
        .map(|(path, bytes)| (path.as_str(), bytes.as_slice()))
        .collect();
    ArtifactPublisher::publish(dir, &manifest, &refs)
}

fn read_file(dir: &Path, relative: &str) -> Result<Vec<u8>> {
    fs::read(dir.join(relative)).map_err(|_| Error::Io)
}

/// Load and verify a published commit. The manifest's file digests and the
/// link are checked; the anchor's identity is recomputed.
pub fn load_commit(dir: &Path) -> Result<PublicCommit> {
    verify_artifact(dir)?;
    let manifest = load_manifest(dir)?;
    if manifest.artifact_schema_version != 1
        || manifest.artifact_kind != ArtifactKind::CommitPublic
    {
        return Err(Error::UnsupportedVersion);
    }
    let anchor = AnchorRecord::decode(&read_file(dir, ANCHOR_PATH)?)?;
    let leaf = LeafRecord::decode(&read_file(dir, LEAF_PATH)?)?;
    let link = LinkProof::decode(&read_file(dir, LINK_PATH)?)?;
    let (column_spec, column_root) = decode_column_spec(&read_file(dir, COLUMN_PATH)?)?;
    let wf = decode_wf(&read_file(dir, WF_PATH)?)?;
    verify_link(
        &anchor,
        &link,
        &leaf.table_identity,
        &leaf.leaf_root,
    )?;
    if column_spec.table_identity != leaf.table_identity
        || column_spec.parameters_digest != anchor.parameters_digest
        || wf.table_identity != leaf.table_identity
    {
        return Err(Error::StatementMismatch);
    }
    if manifest.model_commitment_id != anchor.model_commitment {
        return Err(Error::StatementMismatch);
    }
    Ok(PublicCommit {
        anchor,
        leaf,
        link,
        column_spec,
        column_root,
        wf,
    })
}

/// Encode a link record's format version for callers that share the protocol.
pub fn link_format_version() -> u16 {
    LINK_FORMAT_VERSION
}
