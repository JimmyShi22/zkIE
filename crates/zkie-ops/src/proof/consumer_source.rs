//! Streaming matrix reads for the target asset.
//!
//! The consumer path reads one stored row of M at a time through the asset's
//! ranged reader; it never builds the whole matrix. Row `k` is the stored
//! record, and its entries are the places, which is the stored-column view
//! the first-stage W derivation and the shared reader both read.

use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::error::{Error, Result};
use zkie_engine::pcs::request::field::Fp;
use zkie_engine::pcs::request::table::TableRowSource;

use crate::proof::column_session::fp_from_signed_word;
use crate::proof::model_commitment::AssetStore;

/// One ranged reader of a matrix-shaped asset: `rows` stored rows of `width`
/// values, read on demand.
pub trait AssetRangeReader {
    fn rows(&self) -> u64;
    fn width(&self) -> usize;
    fn range(&self, start: u64, count: usize) -> Result<Vec<i64>>;
}

/// The ranged reader of one committed asset: a bounded window of the asset's
/// mmap on each call.
pub struct AssetRange<'a> {
    store: &'a AssetStore<'a>,
    path: String,
    width: usize,
    rows: u64,
}

impl<'a> AssetRange<'a> {
    /// Binds the reader to one committed asset and its matrix shape. The shape
    /// must fit the asset's element count.
    pub fn new(store: &'a AssetStore<'a>, path: &str, width: usize, rows: u64) -> Result<Self> {
        if width == 0 || rows == 0 {
            return Err(Error::InvalidConfig);
        }
        rows.checked_mul(u64::try_from(width).map_err(|_| Error::ResourceLimit)?)
            .ok_or(Error::ResourceLimit)?;
        // The path must name one of the committed assets.
        store.pool(path)?;
        Ok(Self {
            store,
            path: path.to_string(),
            width,
            rows,
        })
    }
}

impl AssetRangeReader for AssetRange<'_> {
    fn rows(&self) -> u64 {
        self.rows
    }

    fn width(&self) -> usize {
        self.width
    }

    fn range(&self, start: u64, count: usize) -> Result<Vec<i64>> {
        if count == 0 || count > self.width {
            return Err(Error::ShapeMismatch);
        }
        self.store.value_range(&self.path, start, count)
    }
}

/// The streaming table row source of the matrix asset: one stored row per
/// call, with the tail left zero. The caller's message buffer must hold at
/// least the row width; the remaining message entries are zero.
pub struct ConsumerMatrixSource<R> {
    reader: R,
}

impl<R: AssetRangeReader> ConsumerMatrixSource<R> {
    pub fn new(reader: R) -> Result<Self> {
        if reader.rows() == 0 || reader.width() == 0 {
            return Err(Error::InvalidConfig);
        }
        Ok(Self { reader })
    }

    pub fn rows(&self) -> u64 {
        self.reader.rows()
    }

    pub fn width(&self) -> usize {
        self.reader.width()
    }
}

impl<R: AssetRangeReader> TableRowSource for ConsumerMatrixSource<R> {
    fn rows(&self) -> u64 {
        self.reader.rows()
    }

    fn row(&mut self, index: u64, message: &mut [Fp], tail: &mut [Fp]) -> Result<()> {
        if index >= self.reader.rows() || message.len() < self.reader.width() {
            return Err(Error::ShapeMismatch);
        }
        let width = self.reader.width();
        let start = index
            .checked_mul(u64::try_from(width).map_err(|_| Error::ResourceLimit)?)
            .ok_or(Error::ResourceLimit)?;
        let values = self.reader.range(start, width)?;
        if values.len() != width {
            return Err(Error::ShapeMismatch);
        }
        for (slot, value) in message.iter_mut().zip(&values) {
            *slot = fp_from_signed_word(*value as u64);
        }
        for slot in message.iter_mut().skip(values.len()) {
            *slot = Fp::ZERO;
        }
        for slot in tail.iter_mut() {
            *slot = Fp::ZERO;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Flat {
        data: Vec<i64>,
        width: usize,
        rows: u64,
    }

    impl AssetRangeReader for Flat {
        fn rows(&self) -> u64 {
            self.rows
        }

        fn width(&self) -> usize {
            self.width
        }

        fn range(&self, start: u64, count: usize) -> Result<Vec<i64>> {
            let start = usize::try_from(start).map_err(|_| Error::ResourceLimit)?;
            let end = start.checked_add(count).ok_or(Error::ResourceLimit)?;
            self.data
                .get(start..end)
                .map(|values| values.to_vec())
                .ok_or(Error::ShapeMismatch)
        }
    }

    fn source() -> ConsumerMatrixSource<Flat> {
        ConsumerMatrixSource::new(Flat {
            data: vec![10, 11, 12, 20, 21, 22],
            width: 3,
            rows: 2,
        })
        .unwrap()
    }

    #[test]
    fn the_source_reads_one_stored_row_at_a_time() {
        let mut source = source();
        assert_eq!(source.rows(), 2);
        assert_eq!(source.width(), 3);
        let mut message = vec![Fp::ZERO; 5];
        let mut tail = vec![Fp::ZERO; 2];
        source.row(1, &mut message, &mut tail).unwrap();
        assert_eq!(message[0], fp_from_signed_word(20));
        assert_eq!(message[1], fp_from_signed_word(21));
        assert_eq!(message[2], fp_from_signed_word(22));
        assert_eq!(message[3], Fp::ZERO);
        assert_eq!(message[4], Fp::ZERO);
        assert_eq!(tail, vec![Fp::ZERO; 2]);
        // Out-of-range rows and a too-short message are refused.
        assert!(source.row(2, &mut message, &mut tail).is_err());
        let mut short = vec![Fp::ZERO; 2];
        assert!(source.row(0, &mut short, &mut tail).is_err());
        // A zero shape is refused at construction.
        assert!(ConsumerMatrixSource::new(Flat {
            data: vec![],
            width: 0,
            rows: 0
        })
        .is_err());
    }
}
