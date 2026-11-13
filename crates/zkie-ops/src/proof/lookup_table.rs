//! Frozen public tables read by range.
//!
//! A table source fixes the trusted identity, the real (unpadded) length, the
//! row encoding and the base/extension row order. The table view adds the
//! padded power-of-two domain: rows past the real length are copies of the
//! first row, so the padding is public and the prover cannot choose it.
//!
//! The verifier evaluates a column at a point without materializing the whole
//! table: the point is split into a low half and a high half, each row chunk
//! intersecting the real table is read once by range, folded over the low
//! half, weighted by the high equality weight and accumulated. Memory is
//! `O(2^(log/2))`, and the recorded IO statistics count range reads, rows and
//! bytes. The content check streams the real rows once and compares their
//! canonical digest with the trusted content root.

use std::cell::Cell;

use zkie_core::artifact::digest_bytes_chunks;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::field::Fp3;

use crate::proof::lin_session::fp3_from_signed_word;
use crate::proof::lookup::{eq_weights, fold_adjacent, LookupTable};

/// Version of the frozen table row encoding.
pub const TABLE_ENCODING_VERSION: u64 = 1;

/// Row encoding of a frozen public table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableEncoding {
    I32Le,
    I64Le,
}

impl TableEncoding {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "i32-le" => Ok(Self::I32Le),
            "i64-le" => Ok(Self::I64Le),
            _ => Err(Error::UnsupportedPrimitive),
        }
    }

    pub const fn width(self) -> usize {
        match self {
            Self::I32Le => 4,
            Self::I64Le => 8,
        }
    }

    /// Appends one row to a canonical content buffer.
    pub fn encode_row(self, value: i64, out: &mut Vec<u8>) -> Result<()> {
        match self {
            Self::I32Le => {
                let value = i32::try_from(value).map_err(|_| Error::BoundViolation)?;
                out.extend_from_slice(&value.to_le_bytes());
            }
            Self::I64Le => out.extend_from_slice(&value.to_le_bytes()),
        }
        Ok(())
    }
}

/// The trusted configuration of a frozen table. The lookup plan fixes the
/// padded row count separately; this configuration fixes the metadata
/// identity, the content root over the concatenated real rows, the real
/// length, the encoding and the encoding version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableConfig {
    pub identity: Digest,
    pub content_root: Digest,
    pub real_len: usize,
    pub encoding: TableEncoding,
    pub version: u64,
}

/// A read-only row source over the frozen table. The adapter maps the logical
/// row index to the base and extension assets; it never exposes more than the
/// requested range.
pub trait TableRowSource {
    fn identity(&self) -> Digest;
    fn content_root(&self) -> Digest;
    fn real_len(&self) -> usize;
    fn encoding(&self) -> TableEncoding;
    /// Reads `count` real rows starting at `start`, decoded and sign-extended.
    fn read_rows(&self, start: usize, count: usize) -> Result<Vec<i64>>;
}

/// The canonical content digest of a row sequence in the given encoding.
pub fn content_digest(encoding: TableEncoding, rows: &[i64]) -> Result<Digest> {
    let mut bytes = Vec::with_capacity(rows.len().saturating_mul(encoding.width()));
    for value in rows {
        encoding.encode_row(*value, &mut bytes)?;
    }
    Ok(digest_bytes_chunks([bytes]))
}

/// Recorded table IO of one verification. `peak_rows` is the largest range
/// buffer, so the streaming property is observable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TableIoStats {
    pub reads: u64,
    pub rows: u64,
    pub bytes: u64,
    pub peak_rows: u64,
}

/// A frozen table with its padded power-of-two domain.
pub struct RangeTable<S: TableRowSource> {
    source: S,
    config: TableConfig,
    table_log: u32,
    first: i64,
    stats: Cell<TableIoStats>,
}

impl<S: TableRowSource> RangeTable<S> {
    /// Opens the table view. The source must agree with the trusted
    /// configuration on every field; the first row is read here so the padded
    /// rows are the public first-row copy.
    pub fn open(source: S, table_log: u32, config: &TableConfig) -> Result<Self> {
        if table_log == 0
            || table_log > 30
            || config.version != TABLE_ENCODING_VERSION
            || config.identity == Digest::ZERO
            || config.content_root == Digest::ZERO
            || config.real_len == 0
        {
            return Err(Error::InvalidConfig);
        }
        let rows = 1usize
            .checked_shl(table_log)
            .ok_or(Error::ResourceLimit)?;
        if config.real_len > rows
            || source.identity() != config.identity
            || source.content_root() != config.content_root
            || source.real_len() != config.real_len
            || source.encoding() != config.encoding
        {
            return Err(Error::StatementMismatch);
        }
        let mut table = Self {
            source,
            config: *config,
            table_log,
            first: 0,
            stats: Cell::default(),
        };
        let first = table.read_rows_raw(0, 1)?;
        table.first = *first.first().ok_or(Error::ShapeMismatch)?;
        Ok(table)
    }

    pub fn stats(&self) -> TableIoStats {
        self.stats.get()
    }

    pub fn config(&self) -> &TableConfig {
        &self.config
    }

    pub fn table_rows(&self) -> Result<usize> {
        1usize
            .checked_shl(self.table_log)
            .ok_or(Error::ResourceLimit)
    }

    fn read_rows_raw(&self, start: usize, count: usize) -> Result<Vec<i64>> {
        let rows = self.source.read_rows(start, count)?;
        if rows.len() != count {
            return Err(Error::ShapeMismatch);
        }
        let mut stats = self.stats.get();
        stats.reads = stats.reads.saturating_add(1);
        stats.rows = stats.rows.saturating_add(count as u64);
        stats.bytes = stats
            .bytes
            .saturating_add((count as u64).saturating_mul(self.config.encoding.width() as u64));
        stats.peak_rows = stats.peak_rows.max(count as u64);
        self.stats.set(stats);
        Ok(rows)
    }

    /// Reads `count` rows from `start` in the padded domain: the rows past the
    /// real length are the first-row copy.
    fn read_padded(&self, start: usize, count: usize) -> Result<Vec<i64>> {
        let real = self.config.real_len;
        let rows = self.table_rows()?;
        if start >= rows || count == 0 || start.checked_add(count).ok_or(Error::ResourceLimit)? > rows {
            return Err(Error::ShapeMismatch);
        }
        if start >= real {
            return Ok(vec![self.first; count]);
        }
        let take = (real - start).min(count);
        let mut values = self.read_rows_raw(start, take)?;
        values.resize(count, self.first);
        Ok(values)
    }

    fn column_chunk(&self, column: usize, start: usize, count: usize) -> Result<Vec<Fp3>> {
        match column {
            0 => {
                let rows = self.table_rows()?;
                if start.checked_add(count).ok_or(Error::ResourceLimit)? > rows {
                    return Err(Error::ShapeMismatch);
                }
                Ok((start..start + count)
                    .map(|row| {
                        if row < self.config.real_len {
                            Fp3::from_u64(row as u64)
                        } else {
                            Fp3::ZERO
                        }
                    })
                    .collect())
            }
            1 => Ok(self
                .read_padded(start, count)?
                .into_iter()
                .map(|value| fp3_from_signed_word(value as u64))
                .collect()),
            _ => Err(Error::InvalidIndex),
        }
    }

    /// Streams the real rows in bounded chunks and checks their canonical
    /// content digest against the trusted content root.
    pub fn verify_content(&self) -> Result<Digest> {
        let encoding = self.config.encoding;
        let real = self.config.real_len;
        let mut cursor = 0usize;
        let mut error: Option<Error> = None;
        let chunks = std::iter::from_fn(|| {
            if error.is_some() || cursor >= real {
                return None;
            }
            let count = (real - cursor).min(4096);
            match self.read_rows_raw(cursor, count) {
                Ok(rows) => {
                    cursor += count;
                    let mut bytes = Vec::with_capacity(count * encoding.width());
                    for value in &rows {
                        if let Err(failure) = encoding.encode_row(*value, &mut bytes) {
                            error = Some(failure);
                            return None;
                        }
                    }
                    Some(bytes)
                }
                Err(failure) => {
                    error = Some(failure);
                    None
                }
            }
        });
        let digest = digest_bytes_chunks(chunks);
        if let Some(failure) = error {
            return Err(failure);
        }
        if digest != self.config.content_root {
            return Err(Error::StatementMismatch);
        }
        Ok(digest)
    }
}

impl<S: TableRowSource> LookupTable for RangeTable<S> {
    fn columns(&self) -> usize {
        2
    }

    fn row_count(&self) -> usize {
        self.table_rows().unwrap_or(0)
    }

    fn identity(&self) -> Digest {
        self.config.identity
    }

    fn config(&self) -> TableConfig {
        self.config
    }

    fn check_content(&self) -> Result<()> {
        RangeTable::verify_content(self).map(|_| ())
    }

    fn column_at(&self, column: usize, point: &[Fp3]) -> Result<Fp3> {
        if point.len() != self.table_log as usize {
            return Err(Error::ShapeMismatch);
        }
        if column > 1 {
            return Err(Error::InvalidIndex);
        }
        let log = self.table_log as usize;
        let low_bits = (log + 1) / 2;
        let high_bits = log - low_bits;
        let chunk = 1usize
            .checked_shl(low_bits as u32)
            .ok_or(Error::ResourceLimit)?;
        let high_weights = eq_weights(&point[low_bits..]);
        if high_weights.len() != (1usize << high_bits) {
            return Err(Error::ShapeMismatch);
        }
        let mut value = Fp3::ZERO;
        for (high, weight) in high_weights.iter().enumerate() {
            let start = high
                .checked_shl(low_bits as u32)
                .ok_or(Error::ResourceLimit)?;
            let values = self.column_chunk(column, start, chunk)?;
            let folded = fold_adjacent(&values, &point[..low_bits])?;
            value = value.add(&weight.mul(&folded));
        }
        Ok(value)
    }

    fn column_rows(&self, column: usize) -> Result<Vec<Fp3>> {
        let rows = self.table_rows()?;
        self.column_chunk(column, 0, rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SplitRows {
        identity: Digest,
        root: Digest,
        base: Vec<i64>,
        extension: Vec<i64>,
        encoding: TableEncoding,
    }

    impl SplitRows {
        fn new(identity: u8, base: Vec<i64>, extension: Vec<i64>) -> Self {
            let encoding = TableEncoding::I32Le;
            let mut all = base.clone();
            all.extend_from_slice(&extension);
            let root = content_digest(encoding, &all).unwrap();
            Self {
                identity: Digest([identity; 32]),
                root,
                base,
                extension,
                encoding,
            }
        }

        fn config(&self) -> TableConfig {
            TableConfig {
                identity: self.identity,
                content_root: self.root,
                real_len: self.base.len() + self.extension.len(),
                encoding: self.encoding,
                version: TABLE_ENCODING_VERSION,
            }
        }
    }

    impl TableRowSource for SplitRows {
        fn identity(&self) -> Digest {
            self.identity
        }

        fn content_root(&self) -> Digest {
            self.root
        }

        fn real_len(&self) -> usize {
            self.base.len() + self.extension.len()
        }

        fn encoding(&self) -> TableEncoding {
            self.encoding
        }

        fn read_rows(&self, start: usize, count: usize) -> Result<Vec<i64>> {
            if start.checked_add(count).ok_or(Error::ResourceLimit)? > self.real_len() {
                return Err(Error::ShapeMismatch);
            }
            let mut rows = Vec::with_capacity(count);
            for index in start..start + count {
                rows.push(if index < self.base.len() {
                    self.base[index]
                } else {
                    self.extension[index - self.base.len()]
                });
            }
            Ok(rows)
        }
    }

    fn signed(value: i64) -> Fp3 {
        fp3_from_signed_word(value as u64)
    }

    #[test]
    fn frozen_rows_concatenate_base_and_extension_and_copy_the_first_row() {
        let source = SplitRows::new(0x81, vec![10, -3, 7], vec![0, 42]);
        let config = source.config();
        let table = RangeTable::open(source, 3, &config).unwrap();
        let index = table.column_rows(0).unwrap();
        assert_eq!(
            index,
            [0u64, 1, 2, 3, 4, 0, 0, 0]
                .iter()
                .map(|value| Fp3::from_u64(*value))
                .collect::<Vec<_>>()
        );
        let values = table.column_rows(1).unwrap();
        assert_eq!(
            values,
            [signed(10), signed(-3), signed(7), signed(0), signed(42), signed(10), signed(10), signed(10)]
        );
        assert_eq!(table.verify_content().unwrap(), table.config().content_root);
    }

    #[test]
    fn streaming_evaluation_matches_the_direct_fold() {
        let source = SplitRows::new(0x82, vec![10, -3, 7], vec![0, 42]);
        let config = source.config();
        let table = RangeTable::open(source, 3, &config).unwrap();
        let point = [Fp3::from_u64(2), Fp3::from_u64(5), Fp3::from_u64(7)];
        let direct = fold_adjacent(&table.column_rows(1).unwrap(), &point).unwrap();
        let table = RangeTable::open(SplitRows::new(0x82, vec![10, -3, 7], vec![0, 42]), 3, &config).unwrap();
        let streamed = table.column_at(1, &point).unwrap();
        assert_eq!(streamed, direct);
        let stats = table.stats();
        // The open reads row zero; the two high chunks read four and one real
        // rows, so six real rows over three reads.
        assert_eq!(stats.reads, 3);
        assert_eq!(stats.rows, 6);
        assert_eq!(stats.bytes, 24);
        assert_eq!(stats.peak_rows, 4);
        let index_direct = fold_adjacent(&table.column_rows(0).unwrap(), &point).unwrap();
        let table = RangeTable::open(SplitRows::new(0x82, vec![10, -3, 7], vec![0, 42]), 3, &config).unwrap();
        assert_eq!(table.column_at(0, &point).unwrap(), index_direct);
        assert_eq!(table.stats().reads, 1);
    }

    #[test]
    fn content_digest_binds_the_read_rows() {
        let source = SplitRows::new(0x83, vec![1, 2, 3], vec![4]);
        let config = source.config();
        let table = RangeTable::open(source, 2, &config).unwrap();
        assert_eq!(table.verify_content().unwrap(), config.content_root);
        let mut tampered = SplitRows::new(0x83, vec![1, 2, 3], vec![4]);
        tampered.extension[0] = 5;
        let table = RangeTable::open(tampered, 2, &config).unwrap();
        assert!(table.verify_content().is_err());
    }

    #[test]
    fn open_rejects_a_mismatched_identity_length_or_version() {
        let source = SplitRows::new(0x84, vec![1, 2, 3], vec![4]);
        let config = source.config();
        let mut bad = config;
        bad.identity = Digest([0x85; 32]);
        assert!(RangeTable::open(SplitRows::new(0x84, vec![1, 2, 3], vec![4]), 2, &bad).is_err());
        let mut bad = config;
        bad.content_root = Digest([0x86; 32]);
        assert!(RangeTable::open(SplitRows::new(0x84, vec![1, 2, 3], vec![4]), 2, &bad).is_err());
        let mut bad = config;
        bad.real_len = 5;
        assert!(RangeTable::open(SplitRows::new(0x84, vec![1, 2, 3], vec![4]), 2, &bad).is_err());
        let mut bad = config;
        bad.real_len = 0;
        assert!(RangeTable::open(SplitRows::new(0x84, vec![1, 2, 3], vec![4]), 2, &bad).is_err());
        let mut bad = config;
        bad.version = TABLE_ENCODING_VERSION + 1;
        assert!(RangeTable::open(SplitRows::new(0x84, vec![1, 2, 3], vec![4]), 2, &bad).is_err());
        assert!(RangeTable::open(SplitRows::new(0x84, vec![1, 2, 3], vec![4]), 1, &config).is_err());
    }

    /// A frozen asset read by range from the given path. The test that uses
    /// this source is driven by the external table path in the environment,
    /// because model assets never live in the repository.
    struct FileRows {
        path: std::path::PathBuf,
        identity: Digest,
        root: Digest,
        length: usize,
    }

    impl FileRows {
        fn open(path: std::path::PathBuf) -> Result<Self> {
            use std::io::Read;
            let mut file = std::fs::File::open(&path).map_err(|_| Error::AssetMismatch)?;
            let metadata = file.metadata().map_err(|_| Error::AssetMismatch)?;
            let bytes = usize::try_from(metadata.len()).map_err(|_| Error::ResourceLimit)?;
            if bytes == 0 || bytes % 4 != 0 {
                return Err(Error::AssetMismatch);
            }
            let mut hasher_chunks = Vec::new();
            let mut buffer = vec![0u8; 1 << 20];
            loop {
                let count = file.read(&mut buffer).map_err(|_| Error::AssetMismatch)?;
                if count == 0 {
                    break;
                }
                hasher_chunks.push(buffer[..count].to_vec());
            }
            let root = digest_bytes_chunks(hasher_chunks);
            Ok(Self {
                path,
                identity: Digest([0x91; 32]),
                root,
                length: bytes / 4,
            })
        }
    }

    impl TableRowSource for FileRows {
        fn identity(&self) -> Digest {
            self.identity
        }

        fn content_root(&self) -> Digest {
            self.root
        }

        fn real_len(&self) -> usize {
            self.length
        }

        fn encoding(&self) -> TableEncoding {
            TableEncoding::I32Le
        }

        fn read_rows(&self, start: usize, count: usize) -> Result<Vec<i64>> {
            use std::os::unix::fs::FileExt;
            if start.checked_add(count).ok_or(Error::ResourceLimit)? > self.length {
                return Err(Error::ShapeMismatch);
            }
            let mut bytes = vec![0u8; count * 4];
            let file = std::fs::File::open(&self.path).map_err(|_| Error::AssetMismatch)?;
            file.read_exact_at(&mut bytes, (start as u64) * 4)
                .map_err(|_| Error::AssetMismatch)?;
            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as i64)
                .collect())
        }
    }

    #[test]
    fn frozen_rsqrt_table_streams_the_direct_value() {
        let Ok(path) = std::env::var("ZKIE_RSQRT_TABLE") else {
            return;
        };
        let source = FileRows::open(std::path::PathBuf::from(path)).unwrap();
        let real = source.real_len();
        assert_eq!(real, 7_340_032);
        let config = TableConfig {
            identity: source.identity(),
            content_root: source.content_root(),
            real_len: real,
            encoding: TableEncoding::I32Le,
            version: TABLE_ENCODING_VERSION,
        };
        let table_log = 23;
        let table = RangeTable::open(source, table_log, &config).unwrap();
        let point = (0..table_log)
            .map(|index| Fp3::from_u64((index as u64 * 13 + 5) % 97))
            .collect::<Vec<_>>();
        let streamed = table.column_at(1, &point).unwrap();
        let stats = table.stats();
        assert_eq!(stats.rows, 1 + real as u64);
        assert_eq!(stats.bytes, (1 + real as u64) * 4);
        assert_eq!(stats.peak_rows, 1 << 12);
        let direct = fold_adjacent(&table.column_rows(1).unwrap(), &point).unwrap();
        assert_eq!(streamed, direct);
        assert_eq!(table.verify_content().unwrap(), config.content_root);
    }

    #[test]
    fn large_table_random_point_streams_the_same_value() {
        let real = 16_000usize;
        let rows = (0..real)
            .map(|index| ((index as i64 * 37) % 1_000) - 500)
            .collect::<Vec<_>>();
        let source = SplitRows::new(0x87, rows[..9_000].to_vec(), rows[9_000..].to_vec());
        let config = source.config();
        let table = RangeTable::open(source, 14, &config).unwrap();
        let point = (0..14u64)
            .map(|index| Fp3::from_u64((index * 13 + 5) % 97))
            .collect::<Vec<_>>();
        let streamed = table.column_at(1, &point).unwrap();
        let stats = table.stats();
        assert_eq!(stats.rows, 1 + real as u64);
        assert_eq!(stats.bytes, (1 + real as u64) * 4);
        assert_eq!(stats.peak_rows, 1 << 7);
        let direct = fold_adjacent(&table.column_rows(1).unwrap(), &point).unwrap();
        assert_eq!(streamed, direct);
        assert_eq!(table.verify_content().unwrap(), config.content_root);
    }
}
