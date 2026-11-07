//! Static-cache identity and namespace compatibility.
//!
//! Cache keys bind implementation compatibility and math identities, never
//! thread counts or device paths. A cache entry is written once and never
//! overwritten.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use zkie_core::artifact::{digest_bytes, digest_to_hex};
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

pub const CACHE_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheKey {
    pub cache_schema_version: u16,
    pub backend_id: u32,
    pub backend_format_version: u16,
    pub implementation_compatibility_id: Digest,
    pub field_descriptor_digest: Digest,
    pub parameters_digest: Digest,
    pub weight_content_digest: Digest,
    pub shape_layout_digest: Digest,
    pub encoding_padding_digest: Digest,
    pub quantization_digest: Digest,
    pub tables_digest: Digest,
    pub privacy_mode: String,
}

impl CacheKey {
    pub fn digest(&self) -> Result<Digest> {
        let bytes = serde_json::to_vec(self).map_err(|_| Error::InvalidEncoding)?;
        Ok(digest_bytes(&bytes))
    }
}

pub struct StaticCache {
    dir: PathBuf,
}

impl StaticCache {
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir).map_err(|_| Error::Io)?;
        Ok(Self { dir })
    }

    pub fn key_path(&self, key: &CacheKey) -> Result<PathBuf> {
        Ok(self
            .dir
            .join(format!("cache-{}.json", digest_to_hex(&key.digest()?))))
    }

    pub fn lookup(&self, key: &CacheKey) -> Result<Option<PathBuf>> {
        let path = self.key_path(key)?;
        Ok(if path.exists() { Some(path) } else { None })
    }

    /// Write a new cache entry. Existing entries are never overwritten.
    pub fn store(&self, key: &CacheKey, payload: &[u8]) -> Result<PathBuf> {
        let path = self.key_path(key)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|_| Error::ArtifactAlreadyExists)?;
        file.write_all(payload).map_err(|_| Error::Io)?;
        file.sync_all().map_err(|_| Error::Io)?;
        Ok(path)
    }
}

pub fn cache_root_is_separate(root: &Path, old_root: &Path) -> bool {
    root != old_root
}
