//! Candidate binding assembly for a request transcript.
//!
//! This is an isolated candidate, not a production protocol format. The field
//! order and the encoding below are internal to this candidate and may be
//! replaced. Values are grouped into three kinds:
//!
//! - parameters: a version word, a protocol identity, domain parameters, and a
//!   minimal plan identifier;
//! - the initial statement: case, request and table identity, plus the version
//!   word and protocol identity repeated for a consistency check;
//! - later dynamic values (messages and outputs), which this step does not
//!   handle.
//!
//! The assembly reuses the existing canonical writer, the parameter digest
//! domain, and the chain-start primitive. Reusing those primitives does not
//! make this candidate a production binding: the production verify entry does
//! not call this module, and no digest produced here has derived a real
//! challenge.
//!
//! A candidate digest only describes the local behavior of the chosen encoding
//! and inputs. It does not prove soundness, hiding or single-shot semantics,
//! and it does not authenticate the source of any preceding chain value.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::chain::Chain;
use crate::pcs::request::params::{RequestParams, REQUEST_CHAIN_LABEL};
use crate::primitives::codec::CanonicalWriter;
use crate::primitives::transcript::{hash_labeled, tag_word, PARAMETERS};

/// Candidate internal encoding marker. It is not an external format version.
const BINDING_SCHEMA_VERSION: u64 = 1;
const ROLE_PARAMETERS: u64 = 1;
const ROLE_STATEMENT: u64 = 2;
const ROLE_PLAN: u64 = 3;

/// Trusted request parameters. Missing values are represented by zero digests
/// and are rejected before any digest is produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindingParameters {
    pub version_word: u32,
    pub protocol_identity: Digest,
    pub domain_parameters: Digest,
    pub plan_identifier: Digest,
}

impl BindingParameters {
    /// Builds the trusted parameters. The domain parameters are the existing
    /// request-parameter digest; the plan identifier is supplied by the caller.
    pub fn new(
        version_word: u32,
        protocol_identity: Digest,
        request_parameters: &RequestParams,
        plan_identifier: Digest,
    ) -> Result<Self> {
        let value = Self {
            version_word,
            protocol_identity,
            domain_parameters: request_parameters.digest()?,
            plan_identifier,
        };
        value.validate()?;
        Ok(value)
    }

    /// Builds the trusted parameters with a plan identifier over an ordered
    /// list of plan-item digests. The item list and its trusted sources are
    /// outside this repository; only the ordered list is encoded here.
    pub fn with_plan_parts(
        version_word: u32,
        protocol_identity: Digest,
        request_parameters: &RequestParams,
        plan_parts: &[Digest],
    ) -> Result<Self> {
        let plan_identifier = plan_identifier(plan_parts)?;
        Self::new(version_word, protocol_identity, request_parameters, plan_identifier)
    }

    pub fn validate(&self) -> Result<()> {
        if self.version_word == 0
            || self.protocol_identity == Digest::ZERO
            || self.domain_parameters == Digest::ZERO
            || self.plan_identifier == Digest::ZERO
        {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }

    fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(BINDING_SCHEMA_VERSION);
        writer.put_u64(ROLE_PARAMETERS);
        writer.put_u64(self.version_word as u64);
        writer.put_digest(&self.protocol_identity);
        writer.put_digest(&self.domain_parameters);
        writer.put_digest(&self.plan_identifier);
        writer.into_bytes()
    }

    /// The parameter binding digest. It reuses the parameter domain and the
    /// canonical writer used by the existing request parameters.
    pub fn digest(&self) -> Result<Digest> {
        self.validate()?;
        hash_labeled(tag_word(PARAMETERS, 0), 0, None, &self.encode())
    }
}

/// The initial statement. The version word and protocol identity are repeated
/// here so that a statement from another protocol or version is rejected
/// before the binding is assembled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BindingStatement {
    pub version_word: u32,
    pub protocol_identity: Digest,
    pub case_id: Vec<u8>,
    pub request_id: Digest,
    pub table_identity: Digest,
}

impl BindingStatement {
    pub fn new(
        version_word: u32,
        protocol_identity: Digest,
        case_id: impl Into<Vec<u8>>,
        request_id: Digest,
        table_identity: Digest,
    ) -> Result<Self> {
        let value = Self {
            version_word,
            protocol_identity,
            case_id: case_id.into(),
            request_id,
            table_identity,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        if self.version_word == 0
            || self.protocol_identity == Digest::ZERO
            || self.case_id.is_empty()
            || self.request_id == Digest::ZERO
            || self.table_identity == Digest::ZERO
        {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }

    fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(BINDING_SCHEMA_VERSION);
        writer.put_u64(ROLE_STATEMENT);
        writer.put_u64(self.version_word as u64);
        writer.put_digest(&self.protocol_identity);
        writer.put_bytes(&self.case_id);
        writer.put_digest(&self.request_id);
        writer.put_digest(&self.table_identity);
        writer.into_bytes()
    }

    /// The canonical statement bytes for a session prefix. The statement is
    /// validated first; the returned bytes are the typed encoding, not a
    /// caller-supplied byte string.
    pub(crate) fn canonical_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        Ok(self.encode())
    }
}

/// The assembled candidate. It is not a production acceptance object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BindingCandidate {
    pub parameters_digest: Digest,
    pub statement_bytes: Vec<u8>,
    pub record_bytes: Vec<u8>,
    pub first_query: Digest,
}

/// A minimal plan identifier candidate: a digest over an ordered list of
/// plan-item digests. The list and the trusted sources of its items are outside
/// this repository. The encoding is candidate-internal and replaceable.
pub fn plan_identifier(parts: &[Digest]) -> Result<Digest> {
    if parts.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let mut writer = CanonicalWriter::new();
    writer.put_u64(BINDING_SCHEMA_VERSION);
    writer.put_u64(ROLE_PLAN);
    writer.put_u64(parts.len() as u64);
    for part in parts {
        if *part == Digest::ZERO {
            return Err(Error::InvalidConfig);
        }
        writer.put_digest(part);
    }
    hash_labeled(tag_word(PARAMETERS, 0), 0, None, &writer.into_bytes())
}

fn encode_record(record: Option<&[u8]>) -> Vec<u8> {
    let mut writer = CanonicalWriter::new();
    match record {
        Some(bytes) => {
            writer.put_u64(1);
            writer.put_bytes(bytes);
        }
        None => writer.put_u64(0),
    }
    writer.into_bytes()
}

/// Assembles the parameter binding, the initial statement and the record, then
/// starts a request chain. The first query is derived from the reused
/// chain-start primitive. A missing required value or a statement that
/// contradicts the parameters is rejected before any digest is returned.
pub fn assemble(
    parameters: &BindingParameters,
    statement: &BindingStatement,
    record: Option<&[u8]>,
) -> Result<BindingCandidate> {
    parameters.validate()?;
    statement.validate()?;
    if parameters.version_word != statement.version_word
        || parameters.protocol_identity != statement.protocol_identity
    {
        return Err(Error::StatementMismatch);
    }
    let parameters_digest = parameters.digest()?;
    let statement_bytes = statement.encode();
    let record_bytes = encode_record(record);
    let chain = Chain::begin(
        REQUEST_CHAIN_LABEL,
        &parameters_digest,
        &statement_bytes,
        &record_bytes,
        None,
    )?;
    Ok(BindingCandidate {
        parameters_digest,
        statement_bytes,
        record_bytes,
        first_query: *chain.value(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::transcript::{hash_labeled, tag_word, PARAMETERS, STATEMENT};

    fn request_parameters(log_n: u32) -> RequestParams {
        RequestParams::default_with_code(log_n, 4, 0).unwrap()
    }

    fn plan_parts() -> [Digest; 2] {
        [Digest([0xa1; 32]), Digest([0xa2; 32])]
    }

    fn parameters() -> BindingParameters {
        BindingParameters::with_plan_parts(
            1,
            Digest([0x11; 32]),
            &request_parameters(4),
            &plan_parts(),
        )
        .unwrap()
    }

    fn statement() -> BindingStatement {
        BindingStatement::new(
            1,
            Digest([0x11; 32]),
            b"case".to_vec(),
            Digest([0x21; 32]),
            Digest([0x22; 32]),
        )
        .unwrap()
    }

    fn first_query(
        parameters: &BindingParameters,
        statement: &BindingStatement,
        record: Option<&[u8]>,
    ) -> Digest {
        assemble(parameters, statement, record).unwrap().first_query
    }

    #[test]
    fn canonical_writer_separates_word_and_byte_string_types() {
        let mut word = CanonicalWriter::new();
        word.put_u64(0x0102_0304);
        let mut bytes = CanonicalWriter::new();
        bytes.put_bytes(&0x0102_0304u64.to_le_bytes());
        assert_ne!(word.into_bytes(), bytes.into_bytes());

        let mut zero_word = CanonicalWriter::new();
        zero_word.put_u64(0);
        let mut zero_digest = CanonicalWriter::new();
        zero_digest.put_digest(&Digest::ZERO);
        assert_ne!(zero_word.into_bytes(), zero_digest.into_bytes());
    }

    #[test]
    fn assembly_is_deterministic_and_reuses_chain_start() {
        let parameters = parameters();
        let statement = statement();
        let first = assemble(&parameters, &statement, None).unwrap();
        let second = assemble(&parameters, &statement, None).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.parameters_digest, parameters.digest().unwrap());
        let expected = Chain::begin(
            REQUEST_CHAIN_LABEL,
            &first.parameters_digest,
            &first.statement_bytes,
            &first.record_bytes,
            None,
        )
        .unwrap();
        assert_eq!(first.first_query, *expected.value());
    }

    #[test]
    fn parameter_version_changes_the_digest() {
        let mut changed = parameters();
        changed.version_word = 2;
        assert_ne!(parameters().digest().unwrap(), changed.digest().unwrap());
    }

    #[test]
    fn parameter_protocol_identity_changes_the_digest() {
        let mut changed = parameters();
        changed.protocol_identity = Digest([0x12; 32]);
        assert_ne!(parameters().digest().unwrap(), changed.digest().unwrap());
    }

    #[test]
    fn parameter_domain_parameters_change_the_digest() {
        let changed = BindingParameters::with_plan_parts(
            1,
            Digest([0x11; 32]),
            &request_parameters(5),
            &plan_parts(),
        )
        .unwrap();
        assert_ne!(parameters().digest().unwrap(), changed.digest().unwrap());
    }

    #[test]
    fn parameter_plan_identifier_changes_the_digest() {
        let changed = BindingParameters::with_plan_parts(
            1,
            Digest([0x11; 32]),
            &request_parameters(4),
            &[Digest([0xa3; 32])],
        )
        .unwrap();
        assert_ne!(parameters().digest().unwrap(), changed.digest().unwrap());
    }

    #[test]
    fn statement_case_changes_the_first_query() {
        let parameters = parameters();
        let mut changed = statement();
        changed.case_id = b"other".to_vec();
        assert_ne!(
            first_query(&parameters, &statement(), None),
            first_query(&parameters, &changed, None)
        );
    }

    #[test]
    fn statement_request_and_table_identities_change_the_first_query() {
        let parameters = parameters();
        let mut changed_request = statement();
        changed_request.request_id = Digest([0x23; 32]);
        assert_ne!(
            first_query(&parameters, &statement(), None),
            first_query(&parameters, &changed_request, None)
        );
        let mut changed_table = statement();
        changed_table.table_identity = Digest([0x24; 32]);
        assert_ne!(
            first_query(&parameters, &statement(), None),
            first_query(&parameters, &changed_table, None)
        );
    }

    #[test]
    fn statement_field_order_is_bound() {
        let parameters = parameters();
        let mut swapped = statement();
        swapped.request_id = statement().table_identity;
        swapped.table_identity = statement().request_id;
        assert_ne!(
            first_query(&parameters, &statement(), None),
            first_query(&parameters, &swapped, None)
        );
    }

    #[test]
    fn record_presence_bytes_and_length_are_bound() {
        let parameters = parameters();
        let statement = statement();
        let absent = first_query(&parameters, &statement, None);
        let empty = first_query(&parameters, &statement, Some(b""));
        let one = first_query(&parameters, &statement, Some(b"a"));
        let two = first_query(&parameters, &statement, Some(b"ab"));
        assert_ne!(absent, empty);
        assert_ne!(empty, one);
        assert_ne!(one, two);
        assert_ne!(one, first_query(&parameters, &statement, Some(b"b")));
    }

    #[test]
    fn case_id_length_and_field_framing_are_bound() {
        let parameters = parameters();
        let mut short = statement();
        short.case_id = b"a".to_vec();
        let mut long = statement();
        long.case_id = b"a\0".to_vec();
        assert_ne!(
            first_query(&parameters, &short, None),
            first_query(&parameters, &long, None)
        );
    }

    #[test]
    fn plan_identifier_binds_order_length_and_presence() {
        let [a, b] = plan_parts();
        assert_ne!(
            plan_identifier(&[a, b]).unwrap(),
            plan_identifier(&[b, a]).unwrap()
        );
        assert_ne!(
            plan_identifier(&[a]).unwrap(),
            plan_identifier(&[a, b]).unwrap()
        );
        assert_eq!(plan_identifier(&[]).unwrap_err().code(), "invalid_config");
        assert_eq!(
            plan_identifier(&[a, Digest::ZERO]).unwrap_err().code(),
            "invalid_config"
        );
    }

    #[test]
    fn domain_separation_isolates_the_domain_label() {
        let parameters = parameters();
        let encoded = parameters.encode();
        let in_parameter_domain =
            hash_labeled(tag_word(PARAMETERS, 0), 0, None, &encoded).unwrap();
        let in_other_domain = hash_labeled(tag_word(STATEMENT, 0), 0, None, &encoded).unwrap();
        assert_eq!(parameters.digest().unwrap(), in_parameter_domain);
        assert_ne!(in_parameter_domain, in_other_domain);
    }

    #[test]
    fn missing_parameters_are_rejected() {
        let mut value = parameters();
        value.version_word = 0;
        assert_eq!(value.digest().unwrap_err().code(), "invalid_config");
        let mut value = parameters();
        value.protocol_identity = Digest::ZERO;
        assert_eq!(value.digest().unwrap_err().code(), "invalid_config");
        let mut value = parameters();
        value.domain_parameters = Digest::ZERO;
        assert_eq!(value.digest().unwrap_err().code(), "invalid_config");
        let mut value = parameters();
        value.plan_identifier = Digest::ZERO;
        assert_eq!(value.digest().unwrap_err().code(), "invalid_config");
    }

    #[test]
    fn missing_statement_values_are_rejected() {
        assert_eq!(
            BindingStatement::new(0, Digest([0x11; 32]), b"case".to_vec(), Digest([0x21; 32]), Digest([0x22; 32]))
                .unwrap_err()
                .code(),
            "invalid_config"
        );
        assert_eq!(
            BindingStatement::new(1, Digest::ZERO, b"case".to_vec(), Digest([0x21; 32]), Digest([0x22; 32]))
                .unwrap_err()
                .code(),
            "invalid_config"
        );
        assert_eq!(
            BindingStatement::new(1, Digest([0x11; 32]), Vec::<u8>::new(), Digest([0x21; 32]), Digest([0x22; 32]))
                .unwrap_err()
                .code(),
            "invalid_config"
        );
        assert_eq!(
            BindingStatement::new(1, Digest([0x11; 32]), b"case".to_vec(), Digest::ZERO, Digest([0x22; 32]))
                .unwrap_err()
                .code(),
            "invalid_config"
        );
        assert_eq!(
            BindingStatement::new(1, Digest([0x11; 32]), b"case".to_vec(), Digest([0x21; 32]), Digest::ZERO)
                .unwrap_err()
                .code(),
            "invalid_config"
        );
    }

    #[test]
    fn contradictory_statement_is_rejected() {
        let parameters = parameters();
        let mut changed_version = statement();
        changed_version.version_word = 2;
        assert_eq!(
            assemble(&parameters, &changed_version, None).unwrap_err().code(),
            "statement_mismatch"
        );
        let mut changed_protocol = statement();
        changed_protocol.protocol_identity = Digest([0x12; 32]);
        assert_eq!(
            assemble(&parameters, &changed_protocol, None).unwrap_err().code(),
            "statement_mismatch"
        );
    }
}
