//! A private file store for one commit's encoded codewords.
//!
//! The store holds only values derived from the trusted table rows under one
//! identity, configuration and private material; it is a cache, not a trust
//! root. It lives in a private directory with a unique session name, is
//! removed when the commit finishes or fails, and makes no claim of secure
//! erasure. A truncation, a version mismatch or a row digest mismatch is
//! reported instead of returning other values.
//!
//! Session rules: every commit uses a fresh session digest, so a second store
//! with the same session fails instead of reusing the directory; a crashed
//! process leaves its own directory behind, and a later commit never reads or
//! reuses it. The owner removes its directory on completion or error with
//! cleanup(); an operator may delete stale directories. No path overwrites or
//! reuses another session's bytes.

use std::fs::{DirBuilder, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::field::{Fp, MODULUS};
use crate::pcs::request::table::EncodedRowStore;
use crate::primitives::codec::CanonicalWriter;
use crate::primitives::transcript::{hash_labeled, tag_word};

const STORE_MAGIC: u64 = 0x4b45_5953_4f54_5245;
const STORE_VERSION: u64 = 1;
const ROW_DOMAIN: u16 = 0x5753;
const DIGEST_BYTES: usize = 32;
const HEADER_BYTES: u64 = 4 * 8 + DIGEST_BYTES as u64;

pub struct FileRowStore {
    dir: PathBuf,
    path: PathBuf,
    file: File,
    rows: u64,
    columns: u64,
    configuration: Digest,
    row_bytes: u64,
    written: u64,
    read: u64,
    finished: bool,
}

fn hex(digest: &Digest) -> String {
    let mut out = String::with_capacity(64);
    for byte in digest.0 {
        out.push_str(&format!("{:02x}", byte));
    }
    out
}

impl FileRowStore {
    /// Creates a fresh store under `private_dir`. The session digest names the
    /// directory; rows and columns fix the layout; `max_temp_bytes` bounds the
    /// file before it is created. The caller passes a directory that is not
    /// world-readable; on Unix the store also creates its own directory with
    /// mode 0700 and the file with mode 0600.
    pub fn new(
        private_dir: impl Into<PathBuf>,
        session: Digest,
        rows: u64,
        columns: u64,
        configuration: Digest,
        max_temp_bytes: u64,
    ) -> Result<Self> {
        if session == Digest::ZERO
            || configuration == Digest::ZERO
            || rows == 0
            || columns == 0
        {
            return Err(Error::InvalidConfig);
        }
        let row_bytes = (DIGEST_BYTES as u64)
            .checked_add(columns.checked_mul(8).ok_or(Error::ResourceLimit)?)
            .ok_or(Error::ResourceLimit)?;
        let total = HEADER_BYTES
            .checked_add(rows.checked_mul(row_bytes).ok_or(Error::ResourceLimit)?)
            .ok_or(Error::ResourceLimit)?;
        if total > max_temp_bytes {
            return Err(Error::ResourceLimit);
        }
        let dir = private_dir
            .into()
            .join(format!("zkie-commit-{}-{}", std::process::id(), hex(&session)));
        let mut builder = DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(&dir).map_err(|_| Error::Io)?;
        let path = dir.join("rows.bin");
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&path).map_err(|_| Error::Io)?;
        let mut writer = CanonicalWriter::new();
        writer.put_u64(STORE_MAGIC);
        writer.put_u64(STORE_VERSION);
        writer.put_u64(rows);
        writer.put_u64(columns);
        writer.put_digest(&configuration);
        file.write_all(&writer.into_bytes())
            .map_err(|_| Error::Io)?;
        file.set_len(total).map_err(|_| Error::Io)?;
        Ok(Self {
            dir,
            path,
            file,
            rows,
            columns,
            configuration,
            row_bytes,
            written: 0,
            read: 0,
            finished: false,
        })
    }

    pub fn file_len(&self) -> Result<u64> {
        Ok(self
            .file
            .metadata()
            .map_err(|_| Error::Io)?
            .len())
    }

    fn row_offset(&self, row: u64) -> Result<u64> {
        if row >= self.rows {
            return Err(Error::InvalidIndex);
        }
        HEADER_BYTES
            .checked_add(row.checked_mul(self.row_bytes).ok_or(Error::ResourceLimit)?)
            .ok_or(Error::ResourceLimit)
    }

    fn codeword_bytes(&self, codeword: &[Fp]) -> Result<Vec<u8>> {
        if codeword.len() as u64 != self.columns {
            return Err(Error::ShapeMismatch);
        }
        let mut writer = CanonicalWriter::new();
        for value in codeword {
            writer.put_fp(*value);
        }
        Ok(writer.into_bytes())
    }

    fn row_digest(&self, row: u64, bytes: &[u8]) -> Result<Digest> {
        let mut data = Vec::with_capacity(DIGEST_BYTES + bytes.len());
        data.extend_from_slice(&self.configuration.0);
        data.extend_from_slice(bytes);
        hash_labeled(tag_word(ROW_DOMAIN, 0), row, None, &data)
    }

    fn read_digest(&mut self, row: u64) -> Result<Digest> {
        let offset = self.row_offset(row)?;
        self.file
            .seek(SeekFrom::Start(offset))
            .map_err(|_| Error::Io)?;
        let mut digest = [0u8; DIGEST_BYTES];
        self.file.read_exact(&mut digest).map_err(|_| Error::Io)?;
        Ok(Digest(digest))
    }

    fn read_codeword_bytes(&mut self, row: u64, bytes: &mut [u8]) -> Result<()> {
        let offset = self
            .row_offset(row)?
            .checked_add(DIGEST_BYTES as u64)
            .ok_or(Error::ResourceLimit)?;
        self.file
            .seek(SeekFrom::Start(offset))
            .map_err(|_| Error::Io)?;
        self.file.read_exact(bytes).map_err(|_| Error::Io)?;
        self.read = self
            .read
            .checked_add(bytes.len() as u64)
            .ok_or(Error::ResourceLimit)?;
        Ok(())
    }
}

impl Drop for FileRowStore {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

impl EncodedRowStore for FileRowStore {
    fn rows(&self) -> u64 {
        self.rows
    }

    fn columns(&self) -> u64 {
        self.columns
    }

    fn put_row(&mut self, row: u64, codeword: &[Fp]) -> Result<()> {
        if self.finished || row >= self.rows {
            return Err(Error::InvalidSessionState);
        }
        let bytes = self.codeword_bytes(codeword)?;
        let digest = self.row_digest(row, &bytes)?;
        let offset = self.row_offset(row)?;
        self.file
            .seek(SeekFrom::Start(offset))
            .map_err(|_| Error::Io)?;
        self.file.write_all(&digest.0).map_err(|_| Error::Io)?;
        self.file.write_all(&bytes).map_err(|_| Error::Io)?;
        self.written = self
            .written
            .checked_add(bytes.len() as u64)
            .ok_or(Error::ResourceLimit)?;
        Ok(())
    }

    fn read_columns(
        &mut self,
        first: u64,
        count: usize,
        rows: usize,
        out: &mut [Fp],
    ) -> Result<()> {
        if !self.finished || rows as u64 != self.rows {
            return Err(Error::InvalidSessionState);
        }
        if out.len() != count.saturating_mul(rows)
            || first.checked_add(count as u64).ok_or(Error::ResourceLimit)? > self.columns
        {
            return Err(Error::ShapeMismatch);
        }
        let byte_start = (first as usize)
            .checked_mul(8)
            .ok_or(Error::ResourceLimit)?;
        let byte_count = count.checked_mul(8).ok_or(Error::ResourceLimit)?;
        let mut buffer = vec![0u8; byte_count];
        for row in 0..rows {
            let offset = self
                .row_offset(row as u64)?
                .checked_add(DIGEST_BYTES as u64)
                .and_then(|value| value.checked_add(byte_start as u64))
                .ok_or(Error::ResourceLimit)?;
            self.file
                .seek(SeekFrom::Start(offset))
                .map_err(|_| Error::Io)?;
            self.file.read_exact(&mut buffer).map_err(|_| Error::Io)?;
            self.read = self
                .read
                .checked_add(buffer.len() as u64)
                .ok_or(Error::ResourceLimit)?;
            for column in 0..count {
                let word = u64::from_le_bytes(
                    buffer[column * 8..column * 8 + 8]
                        .try_into()
                        .map_err(|_| Error::InvalidEncoding)?,
                );
                if word >= MODULUS {
                    return Err(Error::InvalidEncoding);
                }
                out[column * rows + row] = Fp::from_u64(word);
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if self.finished {
            return Err(Error::InvalidSessionState);
        }
        self.file.flush().map_err(|_| Error::Io)?;
        let expected = HEADER_BYTES
            .checked_add(self.rows.checked_mul(self.row_bytes).ok_or(Error::ResourceLimit)?)
            .ok_or(Error::ResourceLimit)?;
        if self.file_len()? != expected {
            return Err(Error::IncompleteArtifact);
        }
        let mut bytes = vec![0u8; (self.columns as usize) * 8];
        for row in 0..self.rows {
            let stored = self.read_digest(row)?;
            self.read_codeword_bytes(row, &mut bytes)?;
            if stored != self.row_digest(row, &bytes)? {
                return Err(Error::InvalidProof);
            }
        }
        self.finished = true;
        Ok(())
    }

    fn cleanup(&mut self) -> Result<()> {
        let path = self.path.clone();
        let dir = self.dir.clone();
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
        Ok(())
    }

    fn bytes_written(&self) -> u64 {
        self.written
    }

    fn bytes_read(&self) -> u64 {
        self.read
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn sample_fp(value: usize) -> Fp {
        Fp::from_u64(value as u64 + 1)
    }

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn dir() -> PathBuf {
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("zkie-store-test-{}-{}", std::process::id(), id))
    }

    fn store(dir: &PathBuf) -> FileRowStore {
        std::fs::create_dir_all(dir).expect("dir");
        FileRowStore::new(
            dir,
            Digest([0x11; 32]),
            4,
            8,
            Digest([0x22; 32]),
            1 << 20,
        )
        .expect("store")
    }

    #[test]
    fn the_file_store_round_trips_and_validates() {
        let dir = dir();
        let mut store = store(&dir);
        let rows: Vec<Vec<Fp>> = (0..4)
            .map(|row| (0..8).map(|at| sample_fp(row * 8 + at)).collect())
            .collect();
        for (row, codeword) in rows.iter().enumerate() {
            store.put_row(row as u64, codeword).expect("put");
        }
        store.finish().expect("finish");
        let mut out = vec![Fp::ZERO; 4 * 8];
        store.read_columns(0, 8, 4, &mut out).expect("read");
        for column in 0..8 {
            for row in 0..4 {
                assert_eq!(out[column * 4 + row], rows[row][column]);
            }
        }
        assert_eq!(store.bytes_written(), 4 * 8 * 8);
        assert!(store.bytes_read() >= 4 * 8 * 8);
        let store_dir = store.dir.clone();
        store.cleanup().expect("cleanup");
        assert!(!store_dir.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_truncated_store_is_rejected() {
        let dir = dir();
        let mut store = store(&dir);
        for row in 0..4u64 {
            store.put_row(row, &[sample_fp(row as usize); 8]).expect("put");
        }
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(store.path.clone())
            .expect("open");
        file.set_len(HEADER_BYTES + 3).expect("truncate");
        assert!(matches!(store.finish(), Err(Error::IncompleteArtifact)));
        store.cleanup().expect("cleanup");
    }

    #[test]
    fn a_corrupted_row_is_rejected() {
        let dir = dir();
        let mut store = store(&dir);
        for row in 0..4u64 {
            store.put_row(row, &[sample_fp(row as usize); 8]).expect("put");
        }
        let at = HEADER_BYTES + DIGEST_BYTES as u64 + 8;
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(store.path.clone())
            .expect("open");
        file.seek(SeekFrom::Start(at)).expect("seek");
        let mut original = [0u8; 1];
        file.read_exact(&mut original).expect("read");
        file.seek(SeekFrom::Start(at)).expect("seek");
        file.write_all(&[original[0] ^ 1]).expect("write");
        let result = store.finish();
        assert!(
            matches!(result, Err(Error::InvalidProof)),
            "unexpected result: {:?}",
            result
        );
        store.cleanup().expect("cleanup");
    }

    #[test]
    fn a_second_store_with_the_same_session_is_rejected() {
        let dir = dir();
        std::fs::create_dir_all(&dir).expect("dir");
        let first = FileRowStore::new(
            &dir,
            Digest([0x11; 32]),
            4,
            8,
            Digest([0x22; 32]),
            1 << 20,
        );
        assert!(first.is_ok());
        let second = FileRowStore::new(
            &dir,
            Digest([0x11; 32]),
            4,
            8,
            Digest([0x22; 32]),
            1 << 20,
        );
        assert!(second.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn the_store_directory_and_file_are_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = dir();
        let mut store = store(&dir);
        let dir_mode = std::fs::metadata(&store.dir)
            .expect("dir")
            .permissions()
            .mode()
            & 0o777;
        let file_mode = std::fs::metadata(&store.path)
            .expect("file")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(dir_mode, 0o700);
        assert_eq!(file_mode, 0o600);
        for row in 0..4u64 {
            store.put_row(row, &[sample_fp(row as usize); 8]).expect("put");
        }
        store.cleanup().expect("cleanup");
    }
}
