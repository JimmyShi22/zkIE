//! Bounded canonical byte encoding.
//!
//! Decoding never reduces a field coefficient modulo the modulus and never
//! accepts trailing bytes; lengths are checked against explicit limits.

use zkie_core::common::codec::DecodeLimits;
use zkie_core::common::field::{PrimeCharacteristicRing, PrimeField64};
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::field::{Fp, Fp3, MODULUS, FP3_BYTES, FP_BYTES};

#[derive(Clone, Debug, Default)]
pub struct CanonicalWriter {
    bytes: Vec<u8>,
}

impl CanonicalWriter {
    pub fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    pub fn put_u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub fn put_digest(&mut self, digest: &Digest) {
        self.bytes.extend_from_slice(&digest.0);
    }

    pub fn put_fp(&mut self, value: Fp) {
        self.put_u64(value.as_canonical_u64());
    }

    pub fn put_fp3(&mut self, value: &Fp3) {
        self.bytes.extend_from_slice(&value.encode());
    }

    /// A byte string prefixed by its length as one 64-bit word.
    pub fn put_bytes(&mut self, value: &[u8]) {
        self.put_u64(value.len() as u64);
        self.bytes.extend_from_slice(value);
    }

    /// Raw bytes with no length prefix, for formats that fix the length
    /// elsewhere (an opening leaf salt, for example).
    pub fn put_raw_bytes(&mut self, value: &[u8]) {
        self.bytes.extend_from_slice(value);
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

pub struct CanonicalReader<'a> {
    input: &'a [u8],
    limits: DecodeLimits,
}

impl<'a> CanonicalReader<'a> {
    pub fn new(input: &'a [u8], limits: DecodeLimits) -> Self {
        Self { input, limits }
    }

    pub fn get_u64(&mut self) -> Result<u64> {
        self.limits.check_bytes(8)?;
        if self.input.len() < 8 {
            return Err(Error::InvalidEncoding);
        }
        let word = u64::from_le_bytes(
            self.input[..8]
                .try_into()
                .map_err(|_| Error::InvalidEncoding)?,
        );
        self.input = &self.input[8..];
        Ok(word)
    }

    pub fn get_digest(&mut self) -> Result<Digest> {
        self.limits.check_bytes(32)?;
        if self.input.len() < 32 {
            return Err(Error::InvalidEncoding);
        }
        let mut digest = [0u8; 32];
        digest.copy_from_slice(&self.input[..32]);
        self.input = &self.input[32..];
        Ok(Digest(digest))
    }

    pub fn get_fp(&mut self) -> Result<Fp> {
        let word = self.get_u64()?;
        if word >= MODULUS {
            return Err(Error::InvalidEncoding);
        }
        Ok(Fp::from_u64(word))
    }

    pub fn get_fp3(&mut self) -> Result<Fp3> {
        self.limits.check_bytes(FP3_BYTES)?;
        if self.input.len() < FP3_BYTES {
            return Err(Error::InvalidEncoding);
        }
        let value = Fp3::decode(&self.input[..FP3_BYTES])?;
        self.input = &self.input[FP3_BYTES..];
        Ok(value)
    }

    /// A length-prefixed byte string, checked against the limits and against
    /// the bytes that remain.
    pub fn get_bytes(&mut self) -> Result<&'a [u8]> {
        let length = self.get_u64()?;
        let length = usize::try_from(length).map_err(|_| Error::InvalidEncoding)?;
        self.limits.check_bytes(length)?;
        if length > self.input.len() {
            return Err(Error::InvalidEncoding);
        }
        let value = &self.input[..length];
        self.input = &self.input[length..];
        Ok(value)
    }

    /// Raw bytes of a length fixed by the caller, no prefix.
    pub fn get_raw_bytes(&mut self, length: usize) -> Result<&'a [u8]> {
        self.limits.check_bytes(length)?;
        if length > self.input.len() {
            return Err(Error::InvalidEncoding);
        }
        let value = &self.input[..length];
        self.input = &self.input[length..];
        Ok(value)
    }

    pub fn remaining(&self) -> &'a [u8] {
        self.input
    }

    pub fn finish(self) -> Result<()> {
        if self.input.is_empty() {
            Ok(())
        } else {
            Err(Error::InvalidEncoding)
        }
    }
}

/// The canonical width of one base-field word.
pub const FP_WORD_BYTES: usize = FP_BYTES;
