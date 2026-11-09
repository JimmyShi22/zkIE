//! Expected-output binding.
//!
//! The policy fixes the expected output digest. Verification strictly decodes
//! the public-inputs artifact, confirms it matches the authenticated computed
//! output, re-encodes the values canonically as little-endian i32 (the same
//! byte semantics as the registered ground-truth file) and compares the digest.
//! In the no-expectation mode the authenticated output and its digest are
//! returned without claiming a user expectation.

use zkie_core::artifact::digest_bytes;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::policy::{OutputExpectation, OutputKind, OutputMode};

pub struct BoundOutput {
    pub accepted: bool,
    pub mode: OutputMode,
    pub digest: Digest,
    pub canonical: Vec<u8>,
    pub values: Vec<i64>,
}

pub fn canonical_i32le(values: &[i64]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(values.len().checked_mul(4).ok_or(Error::ResourceLimit)?);
    for value in values {
        let value = i32::try_from(*value).map_err(|_| Error::StatementMismatch)?;
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(out)
}

/// Strict decode of the public-inputs artifact: u64 little-endian element
/// count followed by fixed-width little-endian elements. Truncated, trailing,
/// length-mismatched and wrong-shape inputs are rejected.
pub fn decode_public_inputs(
    bytes: &[u8],
    kind: OutputKind,
    elements: usize,
) -> Result<Vec<i64>> {
    if bytes.len() < 8 {
        return Err(Error::InvalidEncoding);
    }
    let count = u64::from_le_bytes(
        bytes[..8]
            .try_into()
            .map_err(|_| Error::InvalidEncoding)?,
    );
    let count = usize::try_from(count).map_err(|_| Error::ResourceLimit)?;
    if count != elements {
        return Err(Error::ShapeMismatch);
    }
    let body = count.checked_mul(4).ok_or(Error::ResourceLimit)?;
    if bytes.len() != 8usize.checked_add(body).ok_or(Error::ResourceLimit)? {
        return Err(Error::InvalidEncoding);
    }
    let mut values = Vec::with_capacity(count);
    for chunk in bytes[8..].chunks_exact(4) {
        let chunk: [u8; 4] = chunk.try_into().map_err(|_| Error::InvalidEncoding)?;
        let value = match kind {
            OutputKind::Argmax => u32::from_le_bytes(chunk) as i64,
            OutputKind::IntOutput => i32::from_le_bytes(chunk) as i64,
        };
        values.push(value);
    }
    Ok(values)
}

pub fn bind(
    expectation: &OutputExpectation,
    computed: &[i64],
    vocab: Option<usize>,
    public_inputs: &[u8],
) -> Result<BoundOutput> {
    if computed.is_empty() {
        return Err(Error::InvalidConfig);
    }
    if let Some(elements) = expectation.elements {
        if computed.len() != elements {
            return Err(Error::ShapeMismatch);
        }
    }
    match expectation.kind {
        OutputKind::Argmax => {
            let vocab = vocab.ok_or(Error::InvalidConfig)?;
            for value in computed {
                if *value < 0 || *value as u64 >= vocab as u64 {
                    return Err(Error::StatementMismatch);
                }
            }
        }
        OutputKind::IntOutput => {
            for value in computed {
                i32::try_from(*value).map_err(|_| Error::StatementMismatch)?;
            }
        }
    }
    let decoded = decode_public_inputs(public_inputs, expectation.kind, computed.len())?;
    if decoded != computed {
        return Err(Error::StatementMismatch);
    }
    let canonical = canonical_i32le(computed)?;
    let digest = digest_bytes(&canonical);
    let accepted = match expectation.mode {
        OutputMode::None => true,
        OutputMode::Expected => Some(digest) == expectation.sha256,
    };
    Ok(BoundOutput {
        accepted,
        mode: expectation.mode,
        digest,
        canonical,
        values: computed.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{OutputExpectation, OutputKind, OutputMode};

    fn expectation(mode: OutputMode, digest: Option<Digest>, elements: usize) -> OutputExpectation {
        OutputExpectation {
            mode,
            kind: OutputKind::Argmax,
            sha256: digest,
            elements: Some(elements),
        }
    }

    fn bincode_u32(values: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + values.len() * 4);
        out.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    fn bincode_i32(values: &[i32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + values.len() * 4);
        out.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    #[test]
    fn binds_a_matching_public_inputs_file() {
        let values = vec![1u32, 3, 0, 2];
        let bytes = bincode_u32(&values);
        let canonical = canonical_i32le(&[1, 3, 0, 2]).unwrap();
        let digest = digest_bytes(&canonical);
        let bound = bind(
            &expectation(OutputMode::Expected, Some(digest), 4),
            &[1, 3, 0, 2],
            Some(8),
            &bytes,
        )
        .unwrap();
        assert!(bound.accepted);
        assert_eq!(bound.digest, digest);
        assert_eq!(bound.canonical, canonical);
    }

    #[test]
    fn no_expectation_mode_returns_the_authenticated_output() {
        let values = vec![1u32, 3, 0, 2];
        let bytes = bincode_u32(&values);
        let bound = bind(
            &expectation(OutputMode::None, None, 4),
            &[1, 3, 0, 2],
            Some(8),
            &bytes,
        )
        .unwrap();
        assert!(bound.accepted);
        assert_eq!(bound.mode, OutputMode::None);
        assert_eq!(bound.values, vec![1, 3, 0, 2]);
    }

    #[test]
    fn rejects_digest_and_value_mismatches() {
        let bytes = bincode_u32(&[1, 3, 0, 2]);
        let other = digest_bytes(&canonical_i32le(&[2, 1, 0, 3]).unwrap());
        let bound = bind(
            &expectation(OutputMode::Expected, Some(other), 4),
            &[1, 3, 0, 2],
            Some(8),
            &bytes,
        )
        .unwrap();
        assert!(!bound.accepted);
        assert!(bind(
            &expectation(OutputMode::None, None, 4),
            &[1, 3, 0, 3],
            Some(8),
            &bytes,
        )
        .is_err());
    }

    #[test]
    fn rejects_bad_encodings_and_values() {
        let values = vec![1u32, 3, 0, 2];
        let good = bincode_u32(&values);
        let expected = expectation(OutputMode::None, None, 4);
        assert!(bind(&expected, &[1, 3, 0, 2], Some(8), &good[..good.len() - 1]).is_err());
        let mut trailing = good.clone();
        trailing.push(0);
        assert!(bind(&expected, &[1, 3, 0, 2], Some(8), &trailing).is_err());
        let mut wrong_count = good.clone();
        wrong_count[..8].copy_from_slice(&5u64.to_le_bytes());
        assert!(bind(&expected, &[1, 3, 0, 2], Some(8), &wrong_count).is_err());
        assert!(bind(&expected, &[1, 3, 0, 2], Some(3), &good).is_err());
        assert!(bind(&expected, &[1, 3, 0, 8], Some(8), &bincode_u32(&[1, 3, 0, 8])).is_err());
        assert!(bind(&expected, &[1, 3, 0, 2], None, &good).is_err());
    }

    #[test]
    fn timesfm_values_keep_i32_semantics() {
        let values = vec![-5i32, 7, 0];
        let bytes = bincode_i32(&values);
        let expected = OutputExpectation {
            mode: OutputMode::None,
            kind: OutputKind::IntOutput,
            sha256: None,
            elements: Some(3),
        };
        let bound = bind(&expected, &[-5, 7, 0], None, &bytes).unwrap();
        assert!(bound.accepted);
        assert_eq!(bound.canonical, canonical_i32le(&[-5, 7, 0]).unwrap());
        let mut tampered = bytes.clone();
        let last = tampered.len() - 1;
        tampered[last] ^= 1;
        assert!(bind(&expected, &[-5, 7, 0], None, &tampered).is_err());
    }

    #[test]
    fn argmax_and_int_output_keep_their_widths() {
        let mut bytes = Vec::with_capacity(12);
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&0x8000_0000u32.to_le_bytes());
        assert_eq!(
            decode_public_inputs(&bytes, OutputKind::Argmax, 1).unwrap(),
            vec![2_147_483_648i64]
        );
        assert_eq!(
            decode_public_inputs(&bytes, OutputKind::IntOutput, 1).unwrap(),
            vec![-2_147_483_648i64]
        );
    }
}
