//! Proof envelope decoding: version and size checks happen before allocation.

use zkie_core::common::codec::DecodeLimits;
use zkie_core::error::Error;
use zkie_core::types::Digest;

use zkie_engine::pcs::format::{
    EnvelopeHeader, ProofEnvelope, ENVELOPE_HEADER_BYTES, ENVELOPE_VERSION,
};

fn digest(byte: u8) -> Digest {
    Digest([byte; 32])
}

fn header() -> EnvelopeHeader {
    EnvelopeHeader {
        envelope_version: ENVELOPE_VERSION,
        protocol_id: digest(1),
        protocol_version: 1,
        parameter_digest: digest(2),
        statement_digest: digest(3),
        graph_digest: digest(4),
        quantization_digest: digest(5),
        tables_digest: digest(6),
        trusted_model_id: digest(7),
        coverage_digest: digest(8),
    }
}

fn envelope() -> ProofEnvelope {
    ProofEnvelope {
        header: header(),
        request_section: vec![1, 2, 3, 4],
        shared_section: vec![5, 6, 7],
        final_section: vec![8, 9],
    }
}

fn write_word(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[test]
fn envelope_round_trips_with_generous_limits() {
    let original = envelope();
    let encoded = original.encode();
    let decoded = ProofEnvelope::decode(&encoded, DecodeLimits::unlimited()).unwrap();
    assert_eq!(decoded, original);
}

#[test]
fn version_is_checked_before_sections() {
    let encoded = envelope().encode();
    let mut bumped = encoded.clone();
    write_word(&mut bumped, 0, ENVELOPE_VERSION as u64 + 1);
    assert!(matches!(
        ProofEnvelope::decode(&bumped, DecodeLimits::unlimited()),
        Err(Error::UnsupportedVersion)
    ));

    let mut zero_protocol = encoded.clone();
    write_word(&mut zero_protocol, 8 + 32, 0);
    assert!(matches!(
        ProofEnvelope::decode(&zero_protocol, DecodeLimits::unlimited()),
        Err(Error::UnsupportedVersion)
    ));
}

#[test]
fn zero_identity_digests_are_rejected() {
    let mut original = envelope();
    original.header.trusted_model_id = Digest::ZERO;
    assert!(matches!(
        ProofEnvelope::decode(&original.encode(), DecodeLimits::unlimited()),
        Err(Error::StatementMismatch)
    ));

    let mut original = envelope();
    original.header.parameter_digest = Digest::ZERO;
    assert!(matches!(
        ProofEnvelope::decode(&original.encode(), DecodeLimits::unlimited()),
        Err(Error::StatementMismatch)
    ));
}

#[test]
fn truncated_and_trailing_bytes_are_rejected() {
    let encoded = envelope().encode();
    for length in 0..encoded.len() {
        assert!(
            ProofEnvelope::decode(&encoded[..length], DecodeLimits::unlimited()).is_err(),
            "truncation to {length} bytes was accepted"
        );
    }
    let mut trailing = encoded.clone();
    trailing.push(0);
    assert!(ProofEnvelope::decode(&trailing, DecodeLimits::unlimited()).is_err());
}

#[test]
fn declared_section_length_is_checked_before_copying() {
    let encoded = envelope().encode();
    let mut huge = encoded.clone();
    write_word(&mut huge, ENVELOPE_HEADER_BYTES, u64::MAX);
    assert!(ProofEnvelope::decode(&huge, DecodeLimits::unlimited()).is_err());

    let mut beyond = encoded.clone();
    write_word(
        &mut beyond,
        ENVELOPE_HEADER_BYTES,
        (encoded.len() - ENVELOPE_HEADER_BYTES - 8) as u64,
    );
    assert!(ProofEnvelope::decode(&beyond, DecodeLimits::unlimited()).is_err());
}

#[test]
fn decode_limits_are_enforced_on_sections_and_header() {
    let encoded = envelope().encode();
    let tight_section = DecodeLimits {
        max_bytes: 3,
        max_elements: usize::MAX,
    };
    assert!(
        ProofEnvelope::decode(&encoded, tight_section).is_err(),
        "section larger than max_bytes was accepted"
    );

    let tight_header = DecodeLimits {
        max_bytes: 8,
        max_elements: usize::MAX,
    };
    assert!(
        ProofEnvelope::decode(&encoded, tight_header).is_err(),
        "header digest beyond max_bytes was accepted"
    );
}

#[test]
fn empty_sections_are_rejected() {
    let empty = ProofEnvelope {
        header: header(),
        request_section: Vec::new(),
        shared_section: vec![1],
        final_section: vec![2],
    };
    assert!(matches!(
        ProofEnvelope::decode(&empty.encode(), DecodeLimits::unlimited()),
        Err(Error::IncompleteArtifact)
    ));
}
