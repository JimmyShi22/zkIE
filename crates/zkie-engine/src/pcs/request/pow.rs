//! Proof-of-work search and verification.
//!
//! The test is on the first 64-bit word of call 0 of a round. A nonce is
//! searched in ascending order, so a fixed vector has one deterministic
//! answer; verification only applies the test to the caller's chain prefix.

use zkie_core::error::{Error, Result};

use crate::pcs::request::chain::ChainPrefix;

pub const MAX_WORK_BITS: u8 = 64;

/// Whether the low `bits` bits of `word` are zero.
pub fn work_test(word: u64, bits: u8) -> Result<bool> {
    if bits > MAX_WORK_BITS {
        return Err(Error::InvalidConfig);
    }
    Ok(bits == 0 || word.trailing_zeros() >= bits as u32)
}

/// Verify a nonce against the chain position and target stored in `prefix`.
/// A prefix whose work target differs from `bits` is rejected; the nonce is
/// bound to the prefix's chain label, round and message.
pub fn verify_nonce(prefix: &ChainPrefix, nonce: u64, bits: u8) -> Result<bool> {
    if bits > MAX_WORK_BITS {
        return Err(Error::InvalidConfig);
    }
    if prefix.work_bits != bits {
        return Ok(false);
    }
    if bits == 0 && nonce != 0 {
        return Ok(false);
    }
    let word = prefix.first_word(nonce)?;
    work_test(word, bits)
}

/// Search nonces `0, 1, 2, ...` and return the first that passes. The search
/// is bounded; exhaustion is an error, never a silent pass.
/// Searches for a nonce with a bounded tail that does not make an honest
/// prover fail with constant probability. The expected number of attempts is
/// `2^work_bits`; the cap keeps eight further bits of search headroom, so the
/// failure probability of an honest search is below `e^-256` for the work
/// sizes used by the plan, while the expected work is unchanged.
pub fn search_nonce_for_plan(prefix: &ChainPrefix) -> Result<u64> {
    let bits = prefix.work_bits as u32;
    if bits == 0 {
        return Ok(0);
    }
    let cap_bits = (bits + 8).min(40);
    search_nonce(prefix, 1u64 << cap_bits)
}

pub fn search_nonce(prefix: &ChainPrefix, max_tries: u64) -> Result<u64> {
    let bits = prefix.work_bits;
    if bits > MAX_WORK_BITS {
        return Err(Error::InvalidConfig);
    }
    if bits == 0 {
        return Ok(0);
    }
    let mut nonce = 0u64;
    for _ in 0..max_tries {
        if verify_nonce(prefix, nonce, bits)? {
            return Ok(nonce);
        }
        nonce = nonce.checked_add(1).ok_or(Error::ResourceLimit)?;
    }
    Err(Error::ResourceLimit)
}
