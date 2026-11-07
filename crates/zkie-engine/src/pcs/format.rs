//! Proof envelope header.
//!
//! The envelope binds protocol, parameters, statement, graph, quantization,
//! tables and the trusted model identity. Serialization details are added by
//! the artifact layer; this module only fixes the identity fields and the
//! header validation.
//!
//! [`ProofEnvelope::decode`] checks the format version first and validates
//! every declared section length against the caller's limits and the bytes
//! that actually remain before it copies a section. A malicious length is
//! therefore rejected without a matching allocation.

use zkie_core::common::codec::DecodeLimits;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::primitives::codec::{CanonicalReader, CanonicalWriter};

pub const ENVELOPE_VERSION: u16 = 1;

/// Bytes occupied by the fixed header before the three section lengths:
/// one 64-bit version, the protocol id, the 64-bit protocol version, and the
/// seven remaining identity digests.
pub const ENVELOPE_HEADER_BYTES: usize = 8 + 32 + 8 + 7 * 32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnvelopeHeader {
    pub envelope_version: u16,
    pub protocol_id: Digest,
    pub protocol_version: u16,
    pub parameter_digest: Digest,
    pub statement_digest: Digest,
    pub graph_digest: Digest,
    pub quantization_digest: Digest,
    pub tables_digest: Digest,
    pub trusted_model_id: Digest,
    pub coverage_digest: Digest,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofEnvelope {
    pub header: EnvelopeHeader,
    pub request_section: Vec<u8>,
    pub shared_section: Vec<u8>,
    pub final_section: Vec<u8>,
}

impl ProofEnvelope {
    pub fn validate_header(&self) -> Result<()> {
        if self.header.envelope_version != ENVELOPE_VERSION
            || self.header.protocol_version == 0
        {
            return Err(Error::UnsupportedVersion);
        }
        if self.header.protocol_id == Digest::ZERO
            || self.header.parameter_digest == Digest::ZERO
            || self.header.statement_digest == Digest::ZERO
            || self.header.graph_digest == Digest::ZERO
            || self.header.quantization_digest == Digest::ZERO
            || self.header.tables_digest == Digest::ZERO
            || self.header.trusted_model_id == Digest::ZERO
            || self.header.coverage_digest == Digest::ZERO
        {
            return Err(Error::StatementMismatch);
        }
        if self.request_section.is_empty()
            || self.shared_section.is_empty()
            || self.final_section.is_empty()
        {
            return Err(Error::IncompleteArtifact);
        }
        Ok(())
    }

    /// Canonical encoding: fixed header followed by three length-prefixed
    /// sections.
    pub fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(self.header.envelope_version as u64);
        writer.put_digest(&self.header.protocol_id);
        writer.put_u64(self.header.protocol_version as u64);
        writer.put_digest(&self.header.parameter_digest);
        writer.put_digest(&self.header.statement_digest);
        writer.put_digest(&self.header.graph_digest);
        writer.put_digest(&self.header.quantization_digest);
        writer.put_digest(&self.header.tables_digest);
        writer.put_digest(&self.header.trusted_model_id);
        writer.put_digest(&self.header.coverage_digest);
        writer.put_bytes(&self.request_section);
        writer.put_bytes(&self.shared_section);
        writer.put_bytes(&self.final_section);
        writer.into_bytes()
    }

    /// Bounded decode. Version words are checked before any section is read;
    /// each section length is checked against `limits` and against the bytes
    /// that remain before the section is copied.
    pub fn decode(bytes: &[u8], limits: DecodeLimits) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, limits);
        let envelope_version =
            u16::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?;
        if envelope_version != ENVELOPE_VERSION {
            return Err(Error::UnsupportedVersion);
        }
        let protocol_id = reader.get_digest()?;
        let protocol_version =
            u16::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?;
        if protocol_version == 0 {
            return Err(Error::UnsupportedVersion);
        }
        let parameter_digest = reader.get_digest()?;
        let statement_digest = reader.get_digest()?;
        let graph_digest = reader.get_digest()?;
        let quantization_digest = reader.get_digest()?;
        let tables_digest = reader.get_digest()?;
        let trusted_model_id = reader.get_digest()?;
        let coverage_digest = reader.get_digest()?;
        let request_section = reader.get_bytes()?.to_vec();
        let shared_section = reader.get_bytes()?.to_vec();
        let final_section = reader.get_bytes()?.to_vec();
        reader.finish()?;
        let envelope = Self {
            header: EnvelopeHeader {
                envelope_version,
                protocol_id,
                protocol_version,
                parameter_digest,
                statement_digest,
                graph_digest,
                quantization_digest,
                tables_digest,
                trusted_model_id,
                coverage_digest,
            },
            request_section,
            shared_section,
            final_section,
        };
        envelope.validate_header()?;
        Ok(envelope)
    }
}
