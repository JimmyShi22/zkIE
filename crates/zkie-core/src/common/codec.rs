//! Bounded canonical encoding.
//!
//! `Canonical` is only implemented for bounded objects. Large tables are read
//! through the storage service instead of being serialized into one `Vec`.
//! The outermost decoder rejects trailing bytes via [`decode_exact`].

use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeLimits {
    pub max_bytes: usize,
    pub max_elements: usize,
}

impl DecodeLimits {
    pub const fn unlimited() -> Self {
        Self {
            max_bytes: usize::MAX,
            max_elements: usize::MAX,
        }
    }

    pub fn check_bytes(&self, bytes: usize) -> Result<()> {
        if bytes > self.max_bytes {
            Err(Error::InvalidEncoding)
        } else {
            Ok(())
        }
    }

    pub fn check_elements(&self, elements: usize) -> Result<()> {
        if elements > self.max_elements {
            Err(Error::InvalidEncoding)
        } else {
            Ok(())
        }
    }
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self::unlimited()
    }
}

pub trait Canonical: Sized {
    fn encode(&self, out: &mut Vec<u8>) -> Result<()>;
    fn decode(input: &mut &[u8], limits: &DecodeLimits) -> Result<Self>;
}

/// Decode one value and require that no trailing bytes remain.
pub fn decode_exact<T: Canonical>(bytes: &[u8], limits: &DecodeLimits) -> Result<T> {
    let mut input = bytes;
    let value = T::decode(&mut input, limits)?;
    if !input.is_empty() {
        return Err(Error::InvalidEncoding);
    }
    Ok(value)
}

impl Canonical for u32 {
    fn encode(&self, out: &mut Vec<u8>) -> Result<()> {
        out.extend_from_slice(&self.to_le_bytes());
        Ok(())
    }

    fn decode(input: &mut &[u8], limits: &DecodeLimits) -> Result<Self> {
        limits.check_bytes(4)?;
        if input.len() < 4 {
            return Err(Error::InvalidEncoding);
        }
        let bytes: [u8; 4] = input[..4]
            .try_into()
            .map_err(|_| Error::InvalidEncoding)?;
        *input = &input[4..];
        Ok(u32::from_le_bytes(bytes))
    }
}

impl Canonical for u64 {
    fn encode(&self, out: &mut Vec<u8>) -> Result<()> {
        out.extend_from_slice(&self.to_le_bytes());
        Ok(())
    }

    fn decode(input: &mut &[u8], limits: &DecodeLimits) -> Result<Self> {
        limits.check_bytes(8)?;
        if input.len() < 8 {
            return Err(Error::InvalidEncoding);
        }
        let bytes: [u8; 8] = input[..8]
            .try_into()
            .map_err(|_| Error::InvalidEncoding)?;
        *input = &input[8..];
        Ok(u64::from_le_bytes(bytes))
    }
}
