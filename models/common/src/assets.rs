//! Asset manifests: relative path, byte length, shape, encoding and content
//! digest for every weight, LUT, input and ground-truth file.
//!
//! Resolution is always under an explicitly supplied asset root. No function
//! in this module depends on the process working directory.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use sha2::{Digest as ShaDigest, Sha256};

use zkie_core::error::{Error, Result};
use zkie_core::types::{Digest, TensorSpec};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssetSpec {
    pub relative_path: String,
    pub byte_length: u64,
    pub shape: Option<Vec<u64>>,
    pub encoding: String,
    pub role: String,
    pub shared: bool,
}

impl AssetSpec {
    pub fn new(relative_path: impl Into<String>, byte_length: u64) -> Self {
        Self {
            relative_path: relative_path.into(),
            byte_length,
            shape: None,
            encoding: "i32-le".to_owned(),
            role: String::new(),
            shared: false,
        }
    }

    pub fn with_shape(mut self, shape: Vec<u64>) -> Self {
        self.shape = Some(shape);
        self
    }

    pub fn with_encoding(mut self, encoding: impl Into<String>) -> Self {
        self.encoding = encoding.into();
        self
    }

    pub fn with_role(mut self, role: impl Into<String>) -> Self {
        self.role = role.into();
        self
    }

    pub fn shared(mut self) -> Self {
        self.shared = true;
        self
    }

    pub fn validate(&self) -> Result<()> {
        validate_relative_path(&self.relative_path)?;
        if self.encoding.is_empty() {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssetEntry {
    pub relative_path: String,
    pub byte_length: u64,
    pub digest: Digest,
    pub spec: Option<TensorSpec>,
    pub shape: Option<Vec<u64>>,
    pub encoding: String,
    pub role: String,
    pub shared: bool,
}

impl AssetEntry {
    pub fn validate(&self) -> Result<()> {
        validate_relative_path(&self.relative_path)?;
        if self.digest == Digest::ZERO {
            return Err(Error::AssetMismatch);
        }
        if let Some(spec) = &self.spec {
            spec.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssetManifest {
    pub root: PathBuf,
    pub entries: Vec<AssetEntry>,
}

impl AssetManifest {
    pub fn validate(&self) -> Result<()> {
        if self.entries.is_empty() {
            return Err(Error::AssetMismatch);
        }
        let mut seen = BTreeSet::new();
        for entry in &self.entries {
            entry.validate()?;
            if !seen.insert(entry.relative_path.as_str()) {
                return Err(Error::StatementMismatch);
            }
        }
        Ok(())
    }

    pub fn resolve(&self, relative_path: &str) -> Result<PathBuf> {
        validate_relative_path(relative_path)?;
        Ok(self.root.join(relative_path))
    }

    pub fn find(&self, relative_path: &str) -> Option<&AssetEntry> {
        self.entries
            .iter()
            .find(|entry| entry.relative_path == relative_path)
    }

    /// Read every declared file from the explicit root, check its byte length
    /// and SHA-256, and produce the immutable manifest. Nothing is copied or
    /// rewritten.
    pub fn load(root: &Path, specs: &[AssetSpec]) -> Result<Self> {
        if specs.is_empty() {
            return Err(Error::AssetMismatch);
        }
        let mut entries = Vec::with_capacity(specs.len());
        for spec in specs {
            spec.validate()?;
            let path = root.join(&spec.relative_path);
            let metadata = std::fs::metadata(&path).map_err(|_| Error::AssetMismatch)?;
            if metadata.len() != spec.byte_length {
                return Err(Error::AssetMismatch);
            }
            let digest = sha256_file(&path)?;
            entries.push(AssetEntry {
                relative_path: spec.relative_path.clone(),
                byte_length: spec.byte_length,
                digest,
                spec: None,
                shape: spec.shape.clone(),
                encoding: spec.encoding.clone(),
                role: spec.role.clone(),
                shared: spec.shared,
            });
        }
        let manifest = Self {
            root: root.to_path_buf(),
            entries,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    /// Re-read and compare every file against the manifest.
    pub fn verify(&self) -> Result<()> {
        self.validate()?;
        for entry in &self.entries {
            let path = self.resolve(&entry.relative_path)?;
            let metadata = std::fs::metadata(&path).map_err(|_| Error::AssetMismatch)?;
            if metadata.len() != entry.byte_length {
                return Err(Error::AssetMismatch);
            }
            if sha256_file(&path)? != entry.digest {
                return Err(Error::AssetMismatch);
            }
        }
        Ok(())
    }
}

pub fn sha256_file(path: &Path) -> Result<Digest> {
    let mut file = File::open(path).map_err(|_| Error::AssetMismatch)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer).map_err(|_| Error::AssetMismatch)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let result = hasher.finalize();
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&result);
    Ok(Digest(bytes))
}

pub fn validate_relative_path(relative_path: &str) -> Result<()> {
    if relative_path.is_empty() || relative_path.bytes().any(|b| b == 0x5c) {
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
