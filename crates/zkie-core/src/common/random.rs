//! Domain-separated randomness.
//!
//! Randomness is addressed by a domain plus a logical object, never by a worker
//! index or completion order. Recomputing the same logical object yields the
//! same bytes; production runs must use fresh entropy.

use crate::error::Result;

pub trait RandomSource: Send {
    fn fill(
        &mut self,
        domain: &[u8],
        logical_object: u64,
        offset: u64,
        out: &mut [u8],
    ) -> Result<()>;
}
