//! Byte-consumption acceptance of the production verify entry.
//!
//! These tests need the legal artifacts of one production case: a public
//! bundle, statement, proof, public inputs, policy and model id. The proof is
//! multiple GiB and cannot be committed, so the tests are ignored by default
//! and read the artifact directory from `ZKIE_VERIFY_FIXTURE`, which must hold
//! a `gpt2-seq16` subdirectory with:
//!
//! - `public/model-commitment.bin`
//! - `statement.bin`, `proof.bin`, `public-inputs.bin`
//! - `policy.toml`, `model-id.txt`
//!
//! Each test works on a private copy and restores every mutation before the
//! next assertion, so the source fixture is never modified. The proof truncation
//! saves the removed tail byte and writes it back at the same offset; a SHA-256
//! comparison before and after the proof mutation checks that the restored file
//! is byte-identical, including when the last byte is nonzero.

use std::env;
use std::fs;
use std::io::{BufReader, Read, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use zkie_core::artifact::{digest_bytes_chunks, digest_to_hex};
use zkie_core::error::Error;
use zkie_core::progress::RunOutcome;
use zkie_core::types::Digest;
use zkie_ops::pipeline::VerifyReceipt;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir(tag: &str) -> PathBuf {
    let path = env::temp_dir().join(format!(
        "zkie-verify-bytes-{tag}-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

struct Fixture {
    root: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let source = env::var("ZKIE_VERIFY_FIXTURE").unwrap_or_else(|_| {
            panic!("set ZKIE_VERIFY_FIXTURE to a directory holding gpt2-seq16 legal artifacts")
        });
        let case = Path::new(&source).join("gpt2-seq16");
        for relative in [
            "public/model-commitment.bin",
            "statement.bin",
            "proof.bin",
            "public-inputs.bin",
            "policy.toml",
            "model-id.txt",
        ] {
            assert!(
                case.join(relative).is_file(),
                "missing fixture file {relative} under {}",
                case.display()
            );
        }
        let root = temp_dir("work");
        fs::create_dir_all(root.join("public")).unwrap();
        for relative in [
            "public/model-commitment.bin",
            "statement.bin",
            "proof.bin",
            "public-inputs.bin",
            "policy.toml",
            "model-id.txt",
        ] {
            fs::copy(case.join(relative), root.join(relative)).unwrap();
        }
        Self { root }
    }

    fn path(&self, relative: &str) -> PathBuf {
        self.root.join(relative)
    }

    fn args(&self) -> Vec<String> {
        vec![
            "--public".into(),
            self.path("public").to_string_lossy().into_owned(),
            "--statement".into(),
            self.path("statement.bin").to_string_lossy().into_owned(),
            "--proof".into(),
            self.path("proof.bin").to_string_lossy().into_owned(),
            "--public-inputs".into(),
            self.path("public-inputs.bin").to_string_lossy().into_owned(),
            "--model-id".into(),
            fs::read_to_string(self.path("model-id.txt"))
                .unwrap()
                .trim()
                .to_owned(),
            "--policy".into(),
            self.path("policy.toml").to_string_lossy().into_owned(),
            "--run-dir".into(),
            self.path("run").to_string_lossy().into_owned(),
        ]
    }

    fn verify(&self) -> Result<VerifyReceipt, Error> {
        zkie::verify::run(&self.args(), &zkie::provider::production_provider())
    }

    fn assert_accepted(&self) {
        let receipt = self.verify().unwrap_or_else(|error| {
            let report = fs::read_to_string(self.path("run/result.json")).unwrap_or_default();
            panic!("legal artifacts were rejected with {}: {} args={:?}", error.code(), report, self.args())
        });
        assert!(receipt.accepted, "legal artifacts were not accepted");
        assert_eq!(receipt.outcome, RunOutcome::Passed);
    }

    fn assert_rejected(&self, code: &str) {
        match self.verify() {
            Err(error) => assert_eq!(error.code(), code),
            Ok(receipt) => panic!(
                "expected rejection {code}, got accepted={} outcome={:?}",
                receipt.accepted, receipt.outcome
            ),
        }
    }

    fn assert_rejected_any(&self) {
        match self.verify() {
            Err(_) => {}
            Ok(receipt) => panic!(
                "expected rejection, got accepted={} outcome={:?}",
                receipt.accepted, receipt.outcome
            ),
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn append_one(path: &Path) {
    let mut file = fs::OpenOptions::new().append(true).open(path).unwrap();
    file.write_all(&[0u8]).unwrap();
}

fn append_four(path: &Path) {
    let mut file = fs::OpenOptions::new().append(true).open(path).unwrap();
    file.write_all(&[0u8; 4]).unwrap();
}

fn set_len(path: &Path, len: u64) {
    fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .set_len(len)
        .unwrap();
}

fn with_saved_bytes<F: FnOnce(&Path)>(path: &Path, action: F) {
    let original = fs::read(path).unwrap();
    action(path);
    fs::write(path, original).unwrap();
}

fn with_saved_len<F: FnOnce(&Path, u64)>(path: &Path, action: F) {
    let original = fs::metadata(path).unwrap().len();
    let mut last = [0u8; 1];
    if original > 0 {
        let read = fs::File::open(path)
            .unwrap()
            .read_at(&mut last, original - 1)
            .unwrap();
        assert_eq!(read, 1);
    }
    action(path, original);
    set_len(path, original);
    if original > 0 {
        let written = fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap()
            .write_at(&last, original - 1)
            .unwrap();
        assert_eq!(written, 1);
    }
}

struct FileChunks {
    reader: BufReader<fs::File>,
}

impl Iterator for FileChunks {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Vec<u8>> {
        let mut chunk = vec![0u8; 1 << 20];
        let mut filled = 0;
        while filled < chunk.len() {
            match self.reader.read(&mut chunk[filled..]).unwrap() {
                0 => break,
                read => filled += read,
            }
        }
        if filled == 0 {
            None
        } else {
            chunk.truncate(filled);
            Some(chunk)
        }
    }
}

fn file_digest(path: &Path) -> Digest {
    digest_bytes_chunks(FileChunks {
        reader: BufReader::new(fs::File::open(path).unwrap()),
    })
}

#[test]
fn saved_len_restores_a_nonzero_tail_byte() {
    let root = temp_dir("restore");
    let path = root.join("tail.bin");
    let original = vec![0x11u8, 0x22, 0x33, 0xab];
    fs::write(&path, &original).unwrap();
    let before = file_digest(&path);
    with_saved_len(&path, |path, original_len| {
        set_len(path, original_len - 1);
    });
    let after = file_digest(&path);
    assert_eq!(
        digest_to_hex(&before),
        digest_to_hex(&after),
        "nonzero tail byte was not restored"
    );
    assert_eq!(fs::read(&path).unwrap(), original);
    let _ = fs::remove_dir_all(&root);
}

#[test]
#[ignore = "requires ZKIE_VERIFY_FIXTURE with multi-GiB legal artifacts"]
fn production_verify_rejects_trailing_and_truncated_artifacts() {
    let fixture = Fixture::new();
    fixture.assert_accepted();

    let statement = fixture.path("statement.bin");
    with_saved_bytes(&statement, |path| {
        append_one(path);
        fixture.assert_rejected("invalid_proof");
    });
    with_saved_bytes(&statement, |path| {
        let len = fs::metadata(path).unwrap().len();
        set_len(path, len - 1);
        fixture.assert_rejected("invalid_encoding");
    });

    let commitment = fixture.path("public/model-commitment.bin");
    with_saved_bytes(&commitment, |path| {
        append_one(path);
        fixture.assert_rejected("artifact_integrity_mismatch");
    });
    with_saved_bytes(&commitment, |path| {
        let len = fs::metadata(path).unwrap().len();
        set_len(path, len - 1);
        fixture.assert_rejected("invalid_encoding");
    });

    let proof = fixture.path("proof.bin");
    let proof_before = file_digest(&proof);
    with_saved_len(&proof, |path, original| {
        append_one(path);
        fixture.assert_rejected("invalid_proof");
        set_len(path, original - 1);
        fixture.assert_rejected("invalid_encoding");
    });
    let proof_after = file_digest(&proof);
    assert_eq!(
        digest_to_hex(&proof_before),
        digest_to_hex(&proof_after),
        "proof content changed across mutation and restore"
    );
    fixture.assert_accepted();

    let backup = fixture.path("proof.bin.legal");
    fs::rename(&proof, &backup).unwrap();
    fs::create_dir(&proof).unwrap();
    fixture.assert_rejected_any();
    fs::remove_dir(&proof).unwrap();
    fs::rename(&backup, &proof).unwrap();
    fixture.assert_accepted();
}

#[test]
#[ignore = "requires ZKIE_VERIFY_FIXTURE with multi-GiB legal artifacts"]
fn production_verify_rejects_malformed_public_inputs() {
    let fixture = Fixture::new();
    fixture.assert_accepted();

    let public_inputs = fixture.path("public-inputs.bin");
    with_saved_bytes(&public_inputs, |path| {
        append_one(path);
        fixture.assert_rejected("invalid_encoding");
    });
    with_saved_bytes(&public_inputs, |path| {
        append_four(path);
        fixture.assert_rejected("invalid_encoding");
    });
    with_saved_bytes(&public_inputs, |path| {
        let len = fs::metadata(path).unwrap().len();
        set_len(path, len - 1);
        fixture.assert_rejected("invalid_encoding");
    });
    with_saved_bytes(&public_inputs, |path| {
        let mut bytes = fs::read(path).unwrap();
        let count = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        bytes[..8].copy_from_slice(&(count + 1).to_le_bytes());
        fs::write(path, bytes).unwrap();
        fixture.assert_rejected("shape_mismatch");
    });
    with_saved_bytes(&public_inputs, |path| {
        let mut bytes = fs::read(path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        fs::write(path, bytes).unwrap();
        fixture.assert_rejected("statement_mismatch");
    });

    fixture.assert_accepted();
}
