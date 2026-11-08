//! Artifact schema identities.
//!
//! Manifests describe immutable files by relative path, byte length and content
//! digest. The `COMPLETE` marker contains the manifest digest and means only
//! that publishing finished, never that the mathematics verified.

use std::path::{Component, Path};

use serde::{Deserialize, Serialize};
use sha2::{Digest as Sha256Digest, Sha256};

use crate::error::{Error, Result};
use crate::types::Digest;

pub const ARTIFACT_SCHEMA_VERSION: u16 = 1;
pub const MANIFEST_FILE: &str = "manifest.json";
pub const COMPLETE_FILE: &str = "COMPLETE";
pub const PUBLIC_DIR: &str = "public";
pub const PRIVATE_DIR: &str = "private";
pub const PROOF_STATEMENT_FILE: &str = "statement.bin";
pub const PROOF_FILE: &str = "proof.bin";
pub const PROOF_PUBLIC_INPUTS_FILE: &str = "public-inputs.bin";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArtifactKind {
    Commit,
    CommitPublic,
    Proof,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrivacyMode {
    Public,
    Hidden,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactFile {
    pub relative_path: String,
    pub byte_length: u64,
    pub content_digest: Digest,
}

impl ArtifactFile {
    pub fn new(relative_path: impl Into<String>, byte_length: u64, content_digest: Digest) -> Self {
        Self {
            relative_path: relative_path.into(),
            byte_length,
            content_digest,
        }
    }

    pub fn validate(&self) -> Result<()> {
        validate_relative_path(&self.relative_path)?;
        if self.relative_path == MANIFEST_FILE || self.relative_path == COMPLETE_FILE {
            return Err(Error::StatementMismatch);
        }
        if self.content_digest == Digest::ZERO {
            return Err(Error::ArtifactIntegrityMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactManifest {
    pub artifact_schema_version: u16,
    pub artifact_kind: ArtifactKind,
    pub protocol_id: Digest,
    pub protocol_version: u16,
    pub backend_id: u32,
    pub backend_format_version: u16,
    pub model_commitment_id: Digest,
    pub parameters_digest: Digest,
    pub model_config_digest: Digest,
    pub quantization_digest: Digest,
    pub tables_digest: Digest,
    pub privacy_mode: PrivacyMode,
    pub files: Vec<ArtifactFile>,
}

impl ArtifactManifest {
    pub fn validate(&self) -> Result<()> {
        if self.artifact_schema_version != ARTIFACT_SCHEMA_VERSION {
            return Err(Error::UnsupportedVersion);
        }
        if self.protocol_version == 0 || self.backend_format_version == 0 {
            return Err(Error::UnsupportedVersion);
        }
        if self.protocol_id == Digest::ZERO
            || self.model_commitment_id == Digest::ZERO
            || self.parameters_digest == Digest::ZERO
            || self.model_config_digest == Digest::ZERO
            || self.quantization_digest == Digest::ZERO
            || self.tables_digest == Digest::ZERO
        {
            return Err(Error::StatementMismatch);
        }
        if self.files.is_empty() {
            return Err(Error::IncompleteArtifact);
        }
        let mut seen = std::collections::BTreeSet::new();
        for file in &self.files {
            file.validate()?;
            if !seen.insert(file.relative_path.as_str()) {
                return Err(Error::StatementMismatch);
            }
        }
        self.validate_layout()
    }

    pub fn validate_layout(&self) -> Result<()> {
        let under = |file: &ArtifactFile, dir: &str| file.relative_path.starts_with(dir);
        match self.artifact_kind {
            ArtifactKind::Commit => {
                let has_public = self.files.iter().any(|file| under(file, PUBLIC_DIR));
                let has_private = self.files.iter().any(|file| under(file, PRIVATE_DIR));
                if !has_public || !has_private {
                    return Err(Error::IncompleteArtifact);
                }
                if self
                    .files
                    .iter()
                    .any(|file| !under(file, PUBLIC_DIR) && !under(file, PRIVATE_DIR))
                {
                    return Err(Error::StatementMismatch);
                }
            }
            ArtifactKind::CommitPublic => {
                if self.files.iter().any(|file| !under(file, PUBLIC_DIR)) {
                    return Err(Error::StatementMismatch);
                }
            }
            ArtifactKind::Proof => {
                for required in [
                    PROOF_STATEMENT_FILE,
                    PROOF_FILE,
                    PROOF_PUBLIC_INPUTS_FILE,
                ] {
                    if !self.files.iter().any(|file| file.relative_path == required) {
                        return Err(Error::IncompleteArtifact);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn find_file(&self, relative_path: &str) -> Option<&ArtifactFile> {
        self.files
            .iter()
            .find(|file| file.relative_path == relative_path)
    }

    pub fn manifest_digest(&self) -> Result<Digest> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(|_| Error::InvalidEncoding)?;
        Ok(digest_bytes(&bytes))
    }

    pub fn to_json(&self) -> Result<String> {
        self.validate()?;
        serde_json::to_string_pretty(self).map_err(|_| Error::InvalidEncoding)
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let manifest: Self =
            serde_json::from_slice(bytes).map_err(|_| Error::InvalidEncoding)?;
        manifest.validate()?;
        Ok(manifest)
    }
}

pub fn validate_relative_path(relative_path: &str) -> Result<()> {
    if relative_path.is_empty() || relative_path.contains('\\') {
        return Err(Error::InvalidConfig);
    }
    let path = Path::new(relative_path);
    if path.is_absolute() {
        return Err(Error::InvalidConfig);
    }
    for component in path.components() {
        if !matches!(component, Component::Normal(_)) {
            return Err(Error::InvalidConfig);
        }
    }
    Ok(())
}

pub struct Sha256Stream {
    hasher: Sha256,
}

impl Sha256Stream {
    pub fn new() -> Self {
        let _profile = crate::profile::Scope::kernel(crate::profile::Kernel::Sha256);
        Self {
            hasher: Sha256::new(),
        }
    }

    pub fn update(&mut self, bytes: &[u8]) {
        self.hasher.update(bytes);
    }

    pub fn finish(self) -> Digest {
        Digest(self.hasher.finalize().into())
    }
}

impl Default for Sha256Stream {
    fn default() -> Self {
        Self::new()
    }
}

pub fn digest_bytes(bytes: &[u8]) -> Digest {
    digest_bytes_chunks(std::iter::once(bytes))
}

/// SHA-256 over a byte stream supplied in chunks, in order.
pub fn digest_bytes_chunks<I, B>(chunks: I) -> Digest
where
    I: IntoIterator<Item = B>,
    B: AsRef<[u8]>,
{
    let _profile = crate::profile::Scope::kernel(crate::profile::Kernel::Sha256);
    let mut hasher = Sha256::new();
    for chunk in chunks {
        hasher.update(chunk.as_ref());
    }
    Digest(hasher.finalize().into())
}

pub fn digest_to_hex(digest: &Digest) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(64);
    for byte in digest.as_bytes() {
        let _ = write!(out, "{:02x}", byte);
    }
    out
}

pub fn digest_from_hex(hex: &str) -> Result<Digest> {
    let hex = hex.trim();
    if hex.len() != 64 {
        return Err(Error::InvalidEncoding);
    }
    let mut bytes = [0u8; 32];
    for (index, slot) in bytes.iter_mut().enumerate() {
        let start = index * 2;
        *slot = u8::from_str_radix(&hex[start..start + 2], 16)
            .map_err(|_| Error::InvalidEncoding)?;
    }
    Ok(Digest(bytes))
}
