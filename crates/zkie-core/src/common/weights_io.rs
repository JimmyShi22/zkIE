//! Memory-mapped i32 weight loader.
//!
//! Weights are stored on disk as little-endian i32 (4 bytes). Instead of reading
//! and converting the whole file, this maps it and converts ranges to canonical
//! Goldilocks on demand, so a whole weight file never has to be resident (or
//! converted) at once. Combined with streaming this keeps peak memory near the
//! working set.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::common::field::Goldilocks;
use crate::common::fixed_point::from_i32;

pub struct WeightMmap {
    mmap: Mmap,
    len: usize,
}

impl WeightMmap {
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let len = mmap.len() / 4;
        Ok(Self { mmap, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Convert a contiguous `[start, start+len)` range of i32 to canonical
    /// Goldilocks. Only this range is materialized.
    pub fn read_goldilocks(&self, start: usize, len: usize) -> Vec<Goldilocks> {
        let end = (start + len).min(self.len);
        let mut out = Vec::with_capacity(end.saturating_sub(start));
        for i in start..end {
            let off = i * 4;
            let v = i32::from_le_bytes([
                self.mmap[off],
                self.mmap[off + 1],
                self.mmap[off + 2],
                self.mmap[off + 3],
            ]);
            out.push(from_i32(v));
        }
        out
    }

    pub fn read_all(&self) -> Vec<Goldilocks> {
        self.read_goldilocks(0, self.len)
    }
}
