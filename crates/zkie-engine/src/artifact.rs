//! Atomic artifact publishing.
//!
//! A temporary directory is written and synced, the manifest is written and
//! checked, `COMPLETE` receives the manifest digest, the directory is synced
//! again, and only then is it renamed into place. An existing final directory
//! is never overwritten. `COMPLETE` means the write finished, not that the
//! proof is mathematically valid.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use zkie_core::artifact::{
    digest_bytes, digest_from_hex, digest_to_hex, ArtifactManifest, COMPLETE_FILE, MANIFEST_FILE,
};
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

static PUBLISH_SEQ: AtomicU64 = AtomicU64::new(0);

pub struct ArtifactPublisher;

impl ArtifactPublisher {
    pub fn publish(
        final_dir: &Path,
        manifest: &ArtifactManifest,
        files: &[(&str, &[u8])],
    ) -> Result<()> {
        manifest.validate()?;
        if final_dir.exists() {
            return Err(Error::ArtifactAlreadyExists);
        }
        if files.len() != manifest.files.len() {
            return Err(Error::IncompleteArtifact);
        }
        for file in &manifest.files {
            let provided = files
                .iter()
                .find(|(path, _)| *path == file.relative_path)
                .ok_or(Error::IncompleteArtifact)?;
            if provided.1.len() as u64 != file.byte_length {
                return Err(Error::ArtifactIntegrityMismatch);
            }
            if digest_bytes(provided.1) != file.content_digest {
                return Err(Error::ArtifactIntegrityMismatch);
            }
        }
        Self::publish_after_validation(final_dir, manifest, files)
    }

    /// Publish files whose content digests were already computed by the
    /// caller. The manifest and the supplied digests must agree; the bytes are
    /// written once without a second full-file hash. Callers that cannot prove
    /// the digests should use `publish`.
    pub fn publish_with_digests(
        final_dir: &Path,
        manifest: &ArtifactManifest,
        files: &[(&str, &[u8])],
        digests: &[(&str, Digest)],
    ) -> Result<()> {
        manifest.validate()?;
        if final_dir.exists() {
            return Err(Error::ArtifactAlreadyExists);
        }
        if files.len() != manifest.files.len() || digests.len() != manifest.files.len() {
            return Err(Error::IncompleteArtifact);
        }
        for file in &manifest.files {
            let provided = files
                .iter()
                .find(|(path, _)| *path == file.relative_path)
                .ok_or(Error::IncompleteArtifact)?;
            let digest = digests
                .iter()
                .find(|(path, _)| *path == file.relative_path)
                .ok_or(Error::IncompleteArtifact)?;
            if provided.1.len() as u64 != file.byte_length
                || digest.1 != file.content_digest
            {
                return Err(Error::ArtifactIntegrityMismatch);
            }
        }
        Self::publish_after_validation(final_dir, manifest, files)
    }

    fn publish_after_validation(
        final_dir: &Path,
        manifest: &ArtifactManifest,
        files: &[(&str, &[u8])],
    ) -> Result<()> {
        let parent = final_dir
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).map_err(|_| Error::Io)?;
        let temp = temp_dir(parent)?;
        let result = (|| {
            fs::create_dir(&temp).map_err(|_| Error::Io)?;
            for (relative_path, bytes) in files {
                let path = temp.join(relative_path);
                if let Some(dir) = path.parent() {
                    fs::create_dir_all(dir).map_err(|_| Error::Io)?;
                }
                write_and_sync(&path, bytes)?;
            }

            let manifest_json = manifest.to_json()?;
            write_and_sync(&temp.join(MANIFEST_FILE), manifest_json.as_bytes())?;

            let manifest_digest = manifest.manifest_digest()?;
            let marker = format!("{}\n", digest_to_hex(&manifest_digest));
            write_and_sync(&temp.join(COMPLETE_FILE), marker.as_bytes())?;
            sync_dir(&temp)?;

            if final_dir.exists() {
                return Err(Error::ArtifactAlreadyExists);
            }
            fs::rename(&temp, final_dir).map_err(|_| Error::Io)?;
            Ok(())
        })();

        if result.is_err() {
            let _ = fs::remove_dir_all(&temp);
        }
        result
    }
}

pub fn load_manifest(dir: &Path) -> Result<ArtifactManifest> {
    let bytes = fs::read(dir.join(MANIFEST_FILE)).map_err(|_| Error::IncompleteArtifact)?;
    let manifest = ArtifactManifest::from_json(&bytes)?;
    if read_complete_digest(dir)? != manifest.manifest_digest()? {
        return Err(Error::ArtifactIntegrityMismatch);
    }
    Ok(manifest)
}

pub fn read_complete_digest(dir: &Path) -> Result<Digest> {
    let text = fs::read_to_string(dir.join(COMPLETE_FILE))
        .map_err(|_| Error::IncompleteArtifact)?;
    digest_from_hex(text.trim())
}

pub fn verify_files(dir: &Path, manifest: &ArtifactManifest) -> Result<()> {
    manifest.validate()?;
    for file in &manifest.files {
        let bytes =
            fs::read(dir.join(&file.relative_path)).map_err(|_| Error::IncompleteArtifact)?;
        if bytes.len() as u64 != file.byte_length || digest_bytes(&bytes) != file.content_digest {
            return Err(Error::ArtifactIntegrityMismatch);
        }
    }
    Ok(())
}

pub fn verify_artifact(dir: &Path) -> Result<ArtifactManifest> {
    let manifest = load_manifest(dir)?;
    verify_files(dir, &manifest)?;
    Ok(manifest)
}

fn temp_dir(parent: &Path) -> Result<PathBuf> {
    for _ in 0..16 {
        let seq = PUBLISH_SEQ.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(".zkie-tmp-{}-{}", std::process::id(), seq));
        if !candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(Error::ResourceLimit)
}

fn write_and_sync(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = File::create(path).map_err(|_| Error::Io)?;
    file.write_all(bytes).map_err(|_| Error::Io)?;
    file.sync_all().map_err(|_| Error::Io)?;
    Ok(())
}

fn sync_dir(dir: &Path) -> Result<()> {
    let file = File::open(dir).map_err(|_| Error::Io)?;
    file.sync_all().map_err(|_| Error::Io)?;
    Ok(())
}
