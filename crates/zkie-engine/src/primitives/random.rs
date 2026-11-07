//! Domain-separated production randomness.
//!
//! The stream key is read from the operating system at construction. If the
//! entropy source fails, construction fails; there is no fixed-seed fallback.

use std::fs::File;
use std::io::Read;

use zkie_core::common::random::RandomSource;
use zkie_core::error::{Error, Result};

use crate::primitives::codec::CanonicalWriter;
use crate::primitives::transcript::Blake2b;

const KEY_BYTES: usize = 32;

pub struct OsRandomSource {
    key: [u8; KEY_BYTES],
}

impl OsRandomSource {
    pub fn new() -> Result<Self> {
        let mut file = File::open("/dev/urandom").map_err(|_| Error::EntropyUnavailable)?;
        let mut key = [0u8; KEY_BYTES];
        file.read_exact(&mut key)
            .map_err(|_| Error::EntropyUnavailable)?;
        if key.iter().all(|byte| *byte == 0) {
            return Err(Error::EntropyUnavailable);
        }
        Ok(Self { key })
    }

    fn stream_block(
        &self,
        domain: &[u8],
        logical_object: u64,
        offset: u64,
        counter: u64,
    ) -> Result<[u8; 64]> {
        let mut header = CanonicalWriter::new();
        header.put_bytes(domain);
        header.put_u64(logical_object);
        header.put_u64(offset);
        header.put_u64(counter);
        let mut hasher = Blake2b::keyed(64, &self.key, [0u8; 16], [0u8; 16])?;
        hasher.update(&header.into_bytes());
        let digest = hasher.finalize();
        let mut block = [0u8; 64];
        block.copy_from_slice(&digest);
        Ok(block)
    }
}

impl RandomSource for OsRandomSource {
    fn fill(
        &mut self,
        domain: &[u8],
        logical_object: u64,
        offset: u64,
        out: &mut [u8],
    ) -> Result<()> {
        let mut written = 0usize;
        let mut counter = 0u64;
        while written < out.len() {
            let block = self.stream_block(domain, logical_object, offset, counter)?;
            let take = core::cmp::min(block.len(), out.len() - written);
            out[written..written + take].copy_from_slice(&block[..take]);
            written += take;
            counter = counter.checked_add(1).ok_or(Error::ResourceLimit)?;
        }
        Ok(())
    }
}
