//! Artifact transaction contracts.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use zkie_core::artifact::{
    digest_bytes, digest_to_hex, ArtifactFile, ArtifactKind, ArtifactManifest, PrivacyMode,
    COMPLETE_FILE,
};
use zkie_core::error::Error;
use zkie_core::types::Digest;
use zkie_engine::artifact::{load_manifest, verify_artifact, ArtifactPublisher};

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "zkie-artifact-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn digest(seed: u8) -> Digest {
    Digest([seed; 32])
}

fn commit_manifest() -> (ArtifactManifest, Vec<u8>, Vec<u8>) {
    let public = b"public".to_vec();
    let private = b"private".to_vec();
    let manifest = ArtifactManifest {
        artifact_schema_version: 1,
        artifact_kind: ArtifactKind::Commit,
        protocol_id: digest(1),
        protocol_version: 1,
        backend_id: 7,
        backend_format_version: 1,
        model_commitment_id: digest(2),
        parameters_digest: digest(3),
        model_config_digest: digest(4),
        quantization_digest: digest(5),
        tables_digest: digest(6),
        privacy_mode: PrivacyMode::Hidden,
        files: vec![
            ArtifactFile::new("public/p.bin", public.len() as u64, digest_bytes(&public)),
            ArtifactFile::new("private/s.bin", private.len() as u64, digest_bytes(&private)),
        ],
    };
    (manifest, public, private)
}

fn publish(dir: &Path, manifest: &ArtifactManifest, public: &[u8], private: &[u8]) {
    let files: Vec<(&str, &[u8])> = vec![
        ("public/p.bin", public),
        ("private/s.bin", private),
    ];
    ArtifactPublisher::publish(dir, manifest, &files).unwrap();
}

#[test]
fn publish_is_atomic_complete_and_not_overwritten() {
    let temp = TempDir::new();
    let dir = temp.path().join("commit");
    let (manifest, public, private) = commit_manifest();
    publish(&dir, &manifest, &public, &private);
    verify_artifact(&dir).unwrap();
    let loaded = load_manifest(&dir).unwrap();
    assert_eq!(loaded.manifest_digest().unwrap(), manifest.manifest_digest().unwrap());
    assert!(matches!(
        ArtifactPublisher::publish(
            &dir,
            &manifest,
            &[("public/p.bin", &public), ("private/s.bin", &private)]
        ),
        Err(Error::ArtifactAlreadyExists)
    ));
}

#[test]
fn half_written_artifact_is_rejected() {
    let temp = TempDir::new();
    let dir = temp.path().join("commit");
    let (manifest, public, private) = commit_manifest();
    publish(&dir, &manifest, &public, &private);
    fs::remove_file(dir.join(COMPLETE_FILE)).unwrap();
    assert!(matches!(
        verify_artifact(&dir),
        Err(Error::IncompleteArtifact)
    ));
}

#[test]
fn failed_publish_leaves_no_final_or_temp_directory() {
    let temp = TempDir::new();
    let dir = temp.path().join("commit");
    let (mut manifest, public, private) = commit_manifest();
    manifest.files[0].content_digest = digest(9);
    let result = ArtifactPublisher::publish(
        &dir,
        &manifest,
        &[("public/p.bin", &public), ("private/s.bin", &private)],
    );
    assert!(matches!(result, Err(Error::ArtifactIntegrityMismatch)));
    assert!(!dir.exists());
    let leftovers = fs::read_dir(temp.path())
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with(".zkie-tmp-")
        })
        .count();
    assert_eq!(leftovers, 0);
}

#[test]
fn version_and_complete_digest_mismatch_are_rejected() {
    let temp = TempDir::new();
    let dir = temp.path().join("commit");
    let (manifest, public, private) = commit_manifest();

    let mut bad_version = manifest.clone();
    bad_version.artifact_schema_version = 0;
    assert!(matches!(
        bad_version.validate(),
        Err(Error::UnsupportedVersion)
    ));

    publish(&dir, &manifest, &public, &private);
    fs::write(
        dir.join(COMPLETE_FILE),
        format!("{}\n", digest_to_hex(&digest(0))),
    )
    .unwrap();
    assert!(matches!(
        load_manifest(&dir),
        Err(Error::ArtifactIntegrityMismatch)
    ));
}
