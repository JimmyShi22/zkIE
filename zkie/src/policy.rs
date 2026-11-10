//! Trusted verification policy.
//!
//! The policy is fixed by the operator outside the proof bundle. It selects the
//! model version and the expected request shape, and carries the expected
//! anchor and model commitment identities. For the gpt2 output head it also
//! carries the versioned limited table identity: the old model, anchor and
//! asset side, the committed encoded table's root, record, configuration,
//! shape and mapping, and the protocol security parameter identity. The
//! identity is a canonical, domain-separated digest, not a bare root.

use serde::{Deserialize, Serialize};
use zkie_core::artifact::{digest_bytes, digest_from_hex, digest_to_hex};
use zkie_core::common::codec::DecodeLimits;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::code::build_code_plan;
use zkie_engine::pcs::request::table::{validate_index_shape, IndexShape, TableDistances};
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};
use zkie_ops::proof::hidden_send_session::{ActivationCommitment, ActivationShape};

pub const POLICY_VERSION: u32 = 1;
pub const TOKEN_MAX_SEQUENCE: usize = 512;
pub const TIMESFM_RAW_POINTS: usize = 512;
pub const TIMESFM_PATCH_TOKENS: usize = 16;

/// The version of the limited output-head table identity. The version is part
/// of the identity's canonical encoding and therefore of its digest.
pub const LIMITED_TABLE_IDENTITY_VERSION: u32 = 1;
const LIMITED_TABLE_IDENTITY_DOMAIN: &[u8] = b"zkie/limited-table-identity";
const LIMITED_TABLE_IDENTITY_FIELDS: u64 = 24;

/// The version of the trusted activation commitment identity. The version is
/// part of the identity's canonical encoding and therefore of its digest.
pub const ACTIVATION_COMMITMENT_IDENTITY_VERSION: u32 = 1;
const ACTIVATION_COMMITMENT_IDENTITY_DOMAIN: &[u8] = b"zkie/activation-commitment-identity";
const ACTIVATION_COMMITMENT_IDENTITY_FIELDS: u64 = 11;

mod digest_hex {
    use serde::{Deserialize, Deserializer, Serializer};
    use zkie_core::artifact::{digest_from_hex, digest_to_hex};
    use zkie_core::types::Digest;

    pub fn serialize<S: Serializer>(
        value: &Digest,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&digest_to_hex(value))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Digest, D::Error> {
        let text = String::deserialize(deserializer)?;
        digest_from_hex(&text).map_err(serde::de::Error::custom)
    }
}

/// The trusted identity of the limited output-head table.
///
/// The old side is the public model, anchor and weight asset. The new side is
/// the committed encoded table: its root, the canonical digest of the full
/// record, its configuration, its shape and the matrix mapping. The protocol
/// identity is the security parameter set the commitments were made under.
///
/// Every field is encoded with an explicit tag in a fixed order under a fixed
/// domain and an explicit version. Decoding rejects an unknown, missing,
/// duplicate or out-of-order field, a wrong version and trailing bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LimitedTableIdentity {
    pub version: u32,
    pub model: String,
    pub model_version: u32,
    pub case_id: String,
    #[serde(with = "digest_hex")]
    pub anchor_id: Digest,
    #[serde(with = "digest_hex")]
    pub model_id: Digest,
    pub asset_id: u64,
    #[serde(with = "digest_hex")]
    pub asset_table_identity: Digest,
    #[serde(with = "digest_hex")]
    pub table_root: Digest,
    #[serde(with = "digest_hex")]
    pub table_record_digest: Digest,
    #[serde(with = "digest_hex")]
    pub table_configuration: Digest,
    pub code_log_n: u32,
    pub code_ell: u64,
    pub code_r: u64,
    pub table_v: u32,
    pub answers: u32,
    pub salt_bytes: u64,
    pub distance_radius: u64,
    pub distance_later_radius: u64,
    pub distance_m_once: u64,
    pub distance_m_later: u64,
    pub matrix_rows_log: u32,
    pub k_lo: u32,
    pub k_col: u32,
    #[serde(with = "digest_hex")]
    pub protocol_identity: Digest,
}

fn value_string(value: &[u8]) -> Result<String> {
    let text = std::str::from_utf8(value).map_err(|_| Error::InvalidEncoding)?;
    if text.is_empty() {
        return Err(Error::InvalidConfig);
    }
    Ok(text.to_owned())
}

fn value_u64(value: &[u8]) -> Result<u64> {
    let bytes: [u8; 8] = value.try_into().map_err(|_| Error::InvalidEncoding)?;
    Ok(u64::from_le_bytes(bytes))
}

fn value_u32(value: &[u8]) -> Result<u32> {
    u32::try_from(value_u64(value)?).map_err(|_| Error::InvalidConfig)
}

fn value_digest(value: &[u8]) -> Result<Digest> {
    let bytes: [u8; 32] = value.try_into().map_err(|_| Error::InvalidEncoding)?;
    Ok(Digest(bytes))
}

fn put_field(writer: &mut CanonicalWriter, tag: u64, value: &[u8]) {
    writer.put_u64(tag);
    writer.put_bytes(value);
}

fn put_number(writer: &mut CanonicalWriter, tag: u64, value: u64) {
    put_field(writer, tag, &value.to_le_bytes());
}

fn put_digest(writer: &mut CanonicalWriter, tag: u64, value: &Digest) {
    put_field(writer, tag, &value.0);
}

impl LimitedTableIdentity {
    /// The table shape declared by the identity.
    pub fn shape(&self) -> Result<IndexShape> {
        let code = build_code_plan(self.code_log_n, self.code_ell, self.code_r)?;
        let shape = IndexShape {
            code,
            v: self.table_v,
            answers: self.answers,
            salt_bytes: usize::try_from(self.salt_bytes).map_err(|_| Error::InvalidConfig)?,
            distances: TableDistances {
                radius: self.distance_radius,
                later_radius: self.distance_later_radius,
                m_once: self.distance_m_once,
                m_later: self.distance_m_later,
            },
        };
        validate_index_shape(&shape)?;
        Ok(shape)
    }

    /// The matrix row count declared by the identity. It must equal the table
    /// row count and the low-factor times the matrix rows must be the full
    /// column width.
    pub fn matrix_rows(&self) -> Result<u64> {
        if self.table_v != self.matrix_rows_log
            || self.table_v >= 64
            || self.k_lo >= self.k_col
            || self.k_col > 63
        {
            return Err(Error::InvalidConfig);
        }
        let rows = 1u64.checked_shl(self.table_v).ok_or(Error::InvalidConfig)?;
        let factor = 1u64.checked_shl(self.k_lo).ok_or(Error::InvalidConfig)?;
        if factor.checked_mul(rows).ok_or(Error::ResourceLimit)? != 1u64 << self.k_col {
            return Err(Error::InvalidConfig);
        }
        Ok(rows)
    }

    /// Checks the identity's own version, non-empty fields, digests, shape and
    /// mapping. A wrong or old version is rejected.
    pub fn validate(&self) -> Result<()> {
        if self.version != LIMITED_TABLE_IDENTITY_VERSION {
            return Err(Error::UnsupportedVersion);
        }
        if self.model.is_empty()
            || self.case_id.is_empty()
            || self.asset_id == 0
            || self.anchor_id == Digest::ZERO
            || self.model_id == Digest::ZERO
            || self.asset_table_identity == Digest::ZERO
            || self.table_root == Digest::ZERO
            || self.table_record_digest == Digest::ZERO
            || self.table_configuration == Digest::ZERO
            || self.protocol_identity == Digest::ZERO
        {
            return Err(Error::InvalidConfig);
        }
        self.shape()?;
        self.matrix_rows()?;
        Ok(())
    }

    /// The canonical tagged encoding. The fixed domain and version are written
    /// first, then one length-prefixed field per tag in ascending order.
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut writer = CanonicalWriter::new();
        writer.put_bytes(LIMITED_TABLE_IDENTITY_DOMAIN);
        writer.put_u64(self.version as u64);
        writer.put_u64(LIMITED_TABLE_IDENTITY_FIELDS);
        put_field(&mut writer, 1, self.model.as_bytes());
        put_number(&mut writer, 2, self.model_version as u64);
        put_field(&mut writer, 3, self.case_id.as_bytes());
        put_digest(&mut writer, 4, &self.anchor_id);
        put_digest(&mut writer, 5, &self.model_id);
        put_number(&mut writer, 6, self.asset_id);
        put_digest(&mut writer, 7, &self.asset_table_identity);
        put_digest(&mut writer, 8, &self.table_root);
        put_digest(&mut writer, 9, &self.table_record_digest);
        put_digest(&mut writer, 10, &self.table_configuration);
        put_number(&mut writer, 11, self.code_log_n as u64);
        put_number(&mut writer, 12, self.code_ell);
        put_number(&mut writer, 13, self.code_r);
        put_number(&mut writer, 14, self.table_v as u64);
        put_number(&mut writer, 15, self.answers as u64);
        put_number(&mut writer, 16, self.salt_bytes);
        put_number(&mut writer, 17, self.distance_radius);
        put_number(&mut writer, 18, self.distance_later_radius);
        put_number(&mut writer, 19, self.distance_m_once);
        put_number(&mut writer, 20, self.distance_m_later);
        put_number(&mut writer, 21, self.matrix_rows_log as u64);
        put_number(&mut writer, 22, self.k_lo as u64);
        put_number(&mut writer, 23, self.k_col as u64);
        put_digest(&mut writer, 24, &self.protocol_identity);
        Ok(writer.into_bytes())
    }

    /// The domain-separated, versioned digest of the canonical encoding.
    pub fn digest(&self) -> Result<Digest> {
        Ok(digest_bytes(&self.encode()?))
    }

    /// Decodes a canonical identity. Unknown, missing, duplicate or
    /// out-of-order tags, a wrong version, a wrong field width and trailing
    /// bytes are rejected.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let limits = DecodeLimits {
            max_bytes: 1 << 16,
            max_elements: 64,
        };
        let mut reader = CanonicalReader::new(bytes, limits);
        if reader.get_bytes()? != LIMITED_TABLE_IDENTITY_DOMAIN {
            return Err(Error::InvalidConfig);
        }
        if reader.get_u64()? != LIMITED_TABLE_IDENTITY_VERSION as u64 {
            return Err(Error::UnsupportedVersion);
        }
        if reader.get_u64()? != LIMITED_TABLE_IDENTITY_FIELDS {
            return Err(Error::InvalidConfig);
        }
        let mut model: Option<String> = None;
        let mut model_version: Option<u32> = None;
        let mut case_id: Option<String> = None;
        let mut anchor_id: Option<Digest> = None;
        let mut model_id: Option<Digest> = None;
        let mut asset_id: Option<u64> = None;
        let mut asset_table_identity: Option<Digest> = None;
        let mut table_root: Option<Digest> = None;
        let mut table_record_digest: Option<Digest> = None;
        let mut table_configuration: Option<Digest> = None;
        let mut code_log_n: Option<u32> = None;
        let mut code_ell: Option<u64> = None;
        let mut code_r: Option<u64> = None;
        let mut table_v: Option<u32> = None;
        let mut answers: Option<u32> = None;
        let mut salt_bytes: Option<u64> = None;
        let mut distance_radius: Option<u64> = None;
        let mut distance_later_radius: Option<u64> = None;
        let mut distance_m_once: Option<u64> = None;
        let mut distance_m_later: Option<u64> = None;
        let mut matrix_rows_log: Option<u32> = None;
        let mut k_lo: Option<u32> = None;
        let mut k_col: Option<u32> = None;
        let mut protocol_identity: Option<Digest> = None;
        let mut last_tag = 0u64;
        for _ in 0..LIMITED_TABLE_IDENTITY_FIELDS {
            let tag = reader.get_u64()?;
            if tag <= last_tag || tag > LIMITED_TABLE_IDENTITY_FIELDS {
                return Err(Error::InvalidConfig);
            }
            last_tag = tag;
            let value = reader.get_bytes()?;
            match tag {
                1 => model = Some(value_string(value)?),
                2 => model_version = Some(value_u32(value)?),
                3 => case_id = Some(value_string(value)?),
                4 => anchor_id = Some(value_digest(value)?),
                5 => model_id = Some(value_digest(value)?),
                6 => asset_id = Some(value_u64(value)?),
                7 => asset_table_identity = Some(value_digest(value)?),
                8 => table_root = Some(value_digest(value)?),
                9 => table_record_digest = Some(value_digest(value)?),
                10 => table_configuration = Some(value_digest(value)?),
                11 => code_log_n = Some(value_u32(value)?),
                12 => code_ell = Some(value_u64(value)?),
                13 => code_r = Some(value_u64(value)?),
                14 => table_v = Some(value_u32(value)?),
                15 => answers = Some(value_u32(value)?),
                16 => salt_bytes = Some(value_u64(value)?),
                17 => distance_radius = Some(value_u64(value)?),
                18 => distance_later_radius = Some(value_u64(value)?),
                19 => distance_m_once = Some(value_u64(value)?),
                20 => distance_m_later = Some(value_u64(value)?),
                21 => matrix_rows_log = Some(value_u32(value)?),
                22 => k_lo = Some(value_u32(value)?),
                23 => k_col = Some(value_u32(value)?),
                24 => protocol_identity = Some(value_digest(value)?),
                _ => return Err(Error::InvalidConfig),
            }
        }
        reader.finish()?;
        let identity = Self {
            version: LIMITED_TABLE_IDENTITY_VERSION,
            model: model.ok_or(Error::InvalidConfig)?,
            model_version: model_version.ok_or(Error::InvalidConfig)?,
            case_id: case_id.ok_or(Error::InvalidConfig)?,
            anchor_id: anchor_id.ok_or(Error::InvalidConfig)?,
            model_id: model_id.ok_or(Error::InvalidConfig)?,
            asset_id: asset_id.ok_or(Error::InvalidConfig)?,
            asset_table_identity: asset_table_identity.ok_or(Error::InvalidConfig)?,
            table_root: table_root.ok_or(Error::InvalidConfig)?,
            table_record_digest: table_record_digest.ok_or(Error::InvalidConfig)?,
            table_configuration: table_configuration.ok_or(Error::InvalidConfig)?,
            code_log_n: code_log_n.ok_or(Error::InvalidConfig)?,
            code_ell: code_ell.ok_or(Error::InvalidConfig)?,
            code_r: code_r.ok_or(Error::InvalidConfig)?,
            table_v: table_v.ok_or(Error::InvalidConfig)?,
            answers: answers.ok_or(Error::InvalidConfig)?,
            salt_bytes: salt_bytes.ok_or(Error::InvalidConfig)?,
            distance_radius: distance_radius.ok_or(Error::InvalidConfig)?,
            distance_later_radius: distance_later_radius.ok_or(Error::InvalidConfig)?,
            distance_m_once: distance_m_once.ok_or(Error::InvalidConfig)?,
            distance_m_later: distance_m_later.ok_or(Error::InvalidConfig)?,
            matrix_rows_log: matrix_rows_log.ok_or(Error::InvalidConfig)?,
            k_lo: k_lo.ok_or(Error::InvalidConfig)?,
            k_col: k_col.ok_or(Error::InvalidConfig)?,
            protocol_identity: protocol_identity.ok_or(Error::InvalidConfig)?,
        };
        identity.validate()?;
        Ok(identity)
    }

    /// Rebuilds the identity from the committed reader and statement values,
    /// keeping the trusted old-side fields. The committed side is supplied by
    /// the caller and never read from the proof's own metadata by this type.
    #[allow(clippy::too_many_arguments)]
    pub fn rebuild_committed(
        &self,
        table_root: Digest,
        table_record_digest: Digest,
        table_configuration: Digest,
        shape: IndexShape,
        matrix_rows_log: u32,
        k_lo: u32,
        k_col: u32,
        protocol_identity: Digest,
    ) -> Result<Self> {
        let value = Self {
            version: self.version,
            model: self.model.clone(),
            model_version: self.model_version,
            case_id: self.case_id.clone(),
            anchor_id: self.anchor_id,
            model_id: self.model_id,
            asset_id: self.asset_id,
            asset_table_identity: self.asset_table_identity,
            table_root,
            table_record_digest,
            table_configuration,
            code_log_n: shape.code.log_n,
            code_ell: shape.code.ell,
            code_r: shape.code.r,
            table_v: shape.v,
            answers: shape.answers,
            salt_bytes: u64::try_from(shape.salt_bytes).map_err(|_| Error::InvalidConfig)?,
            distance_radius: shape.distances.radius,
            distance_later_radius: shape.distances.later_radius,
            distance_m_once: shape.distances.m_once,
            distance_m_later: shape.distances.m_later,
            matrix_rows_log,
            k_lo,
            k_col,
            protocol_identity,
        };
        value.validate()?;
        Ok(value)
    }

    /// Checks the committed side against this trusted identity. The caller must
    /// take the committed values from the public reader, the statement and the
    /// fixed protocol identity, never from the proof's self-reported metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn check_committed(
        &self,
        table_root: Digest,
        table_record_digest: Digest,
        table_configuration: Digest,
        shape: IndexShape,
        matrix_rows_log: u32,
        k_lo: u32,
        k_col: u32,
        protocol_identity: Digest,
    ) -> Result<()> {
        let actual = self.rebuild_committed(
            table_root,
            table_record_digest,
            table_configuration,
            shape,
            matrix_rows_log,
            k_lo,
            k_col,
            protocol_identity,
        )?;
        if actual.digest()? != self.digest()? {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }
}

/// The trusted activation commitment of one request.
///
/// The identity binds the old side (model, version, case, anchor and model
/// id) to the new side (the hiding row commitment of the independent trusted
/// activation `X`): its root, the matrix shape, the salt length and the
/// configuration digest. It is a canonical, domain-separated digest. The
/// verifier compares this value with the commitment carried by the request
/// session; a prover-supplied commitment cannot substitute for it.
///
/// The field is optional in the document so an older or partial operator copy
/// still parses. The formal request verifier fails closed when the value is
/// absent; it is never defaulted or inferred from a proof.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationCommitmentIdentity {
    pub version: u32,
    pub model: String,
    pub model_version: u32,
    pub case_id: String,
    #[serde(with = "digest_hex")]
    pub anchor_id: Digest,
    #[serde(with = "digest_hex")]
    pub model_id: Digest,
    #[serde(with = "digest_hex")]
    pub activation_root: Digest,
    pub positions: u32,
    pub hidden: u32,
    pub salt_bytes: u64,
    #[serde(with = "digest_hex")]
    pub configuration: Digest,
    #[serde(with = "digest_hex")]
    pub protocol_identity: Digest,
}

impl ActivationCommitmentIdentity {
    /// The activation commitment this identity pins.
    pub fn commitment(&self) -> Result<ActivationCommitment> {
        if self.version != ACTIVATION_COMMITMENT_IDENTITY_VERSION {
            return Err(Error::UnsupportedVersion);
        }
        if self.model.is_empty()
            || self.case_id.is_empty()
            || self.anchor_id == Digest::ZERO
            || self.model_id == Digest::ZERO
            || self.activation_root == Digest::ZERO
            || self.configuration == Digest::ZERO
            || self.protocol_identity == Digest::ZERO
        {
            return Err(Error::InvalidConfig);
        }
        let shape = ActivationShape::new(
            self.positions,
            self.hidden,
            usize::try_from(self.salt_bytes).map_err(|_| Error::InvalidConfig)?,
        )?;
        let commitment = ActivationCommitment {
            root: self.activation_root,
            shape,
            configuration: self.configuration,
        };
        if commitment.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        Ok(commitment)
    }

    /// Checks the identity's own version, fields, digests and shape.
    pub fn validate(&self) -> Result<()> {
        self.commitment()?;
        Ok(())
    }

    /// The canonical tagged encoding under the fixed domain and version.
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut writer = CanonicalWriter::new();
        writer.put_bytes(ACTIVATION_COMMITMENT_IDENTITY_DOMAIN);
        writer.put_u64(self.version as u64);
        writer.put_u64(ACTIVATION_COMMITMENT_IDENTITY_FIELDS);
        put_field(&mut writer, 1, self.model.as_bytes());
        put_number(&mut writer, 2, self.model_version as u64);
        put_field(&mut writer, 3, self.case_id.as_bytes());
        put_digest(&mut writer, 4, &self.anchor_id);
        put_digest(&mut writer, 5, &self.model_id);
        put_digest(&mut writer, 6, &self.activation_root);
        put_number(&mut writer, 7, self.positions as u64);
        put_number(&mut writer, 8, self.hidden as u64);
        put_number(&mut writer, 9, self.salt_bytes);
        put_digest(&mut writer, 10, &self.configuration);
        put_digest(&mut writer, 11, &self.protocol_identity);
        Ok(writer.into_bytes())
    }

    /// The domain-separated, versioned digest of the canonical encoding.
    pub fn digest(&self) -> Result<Digest> {
        Ok(digest_bytes(&self.encode()?))
    }

    /// Decodes a canonical identity. Unknown, missing, duplicate or
    /// out-of-order tags, a wrong version and trailing bytes are rejected.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let limits = DecodeLimits {
            max_bytes: 1 << 16,
            max_elements: 32,
        };
        let mut reader = CanonicalReader::new(bytes, limits);
        if reader.get_bytes()? != ACTIVATION_COMMITMENT_IDENTITY_DOMAIN {
            return Err(Error::InvalidConfig);
        }
        if reader.get_u64()? != ACTIVATION_COMMITMENT_IDENTITY_VERSION as u64 {
            return Err(Error::UnsupportedVersion);
        }
        if reader.get_u64()? != ACTIVATION_COMMITMENT_IDENTITY_FIELDS {
            return Err(Error::InvalidConfig);
        }
        let mut model: Option<String> = None;
        let mut model_version: Option<u32> = None;
        let mut case_id: Option<String> = None;
        let mut anchor_id: Option<Digest> = None;
        let mut model_id: Option<Digest> = None;
        let mut activation_root: Option<Digest> = None;
        let mut positions: Option<u32> = None;
        let mut hidden: Option<u32> = None;
        let mut salt_bytes: Option<u64> = None;
        let mut configuration: Option<Digest> = None;
        let mut protocol_identity: Option<Digest> = None;
        let mut last_tag = 0u64;
        for _ in 0..ACTIVATION_COMMITMENT_IDENTITY_FIELDS {
            let tag = reader.get_u64()?;
            if tag <= last_tag || tag > ACTIVATION_COMMITMENT_IDENTITY_FIELDS {
                return Err(Error::InvalidConfig);
            }
            last_tag = tag;
            let value = reader.get_bytes()?;
            match tag {
                1 => model = Some(value_string(value)?),
                2 => model_version = Some(value_u32(value)?),
                3 => case_id = Some(value_string(value)?),
                4 => anchor_id = Some(value_digest(value)?),
                5 => model_id = Some(value_digest(value)?),
                6 => activation_root = Some(value_digest(value)?),
                7 => positions = Some(value_u32(value)?),
                8 => hidden = Some(value_u32(value)?),
                9 => salt_bytes = Some(value_u64(value)?),
                10 => configuration = Some(value_digest(value)?),
                11 => protocol_identity = Some(value_digest(value)?),
                _ => return Err(Error::InvalidConfig),
            }
        }
        reader.finish()?;
        let identity = Self {
            version: ACTIVATION_COMMITMENT_IDENTITY_VERSION,
            model: model.ok_or(Error::InvalidConfig)?,
            model_version: model_version.ok_or(Error::InvalidConfig)?,
            case_id: case_id.ok_or(Error::InvalidConfig)?,
            anchor_id: anchor_id.ok_or(Error::InvalidConfig)?,
            model_id: model_id.ok_or(Error::InvalidConfig)?,
            activation_root: activation_root.ok_or(Error::InvalidConfig)?,
            positions: positions.ok_or(Error::InvalidConfig)?,
            hidden: hidden.ok_or(Error::InvalidConfig)?,
            salt_bytes: salt_bytes.ok_or(Error::InvalidConfig)?,
            configuration: configuration.ok_or(Error::InvalidConfig)?,
            protocol_identity: protocol_identity.ok_or(Error::InvalidConfig)?,
        };
        identity.validate()?;
        Ok(identity)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyDocument {
    pub policy_version: u32,
    pub model: String,
    pub model_version: u32,
    pub case: Vec<PolicyCase>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyCase {
    pub case_id: String,
    pub sequence_length: Option<usize>,
    pub raw_input_length: Option<usize>,
    pub patch_tokens: Option<usize>,
    pub anchor_id: String,
    pub model_id: String,
    pub output_mode: String,
    pub expected_output_sha256: Option<String>,
    pub output_digest_algorithm: Option<String>,
    pub output_type: Option<String>,
    pub output_encoding: Option<String>,
    pub output_elements: Option<usize>,
    /// The trusted limited output-head table identity. Required for every gpt2
    /// case entry; absent and rejected for the other models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limited_table_identity: Option<LimitedTableIdentity>,
    /// The trusted activation commitment. Optional in the document so an
    /// older copy still parses; the formal request verifier requires it and
    /// fails closed when it is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation_commitment: Option<ActivationCommitmentIdentity>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputMode {
    Expected,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputKind {
    Argmax,
    IntOutput,
}

impl OutputKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Argmax => "argmax",
            Self::IntOutput => "int_output",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputExpectation {
    pub mode: OutputMode,
    pub kind: OutputKind,
    pub sha256: Option<Digest>,
    pub elements: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrustedRequest {
    pub model: String,
    pub model_version: u32,
    pub case_id: String,
    pub sequence_length: Option<usize>,
    pub raw_input_length: Option<usize>,
    pub patch_tokens: Option<usize>,
    pub anchor_id: Digest,
    pub model_id: Digest,
    pub output: OutputExpectation,
    pub limited_table_identity: Option<LimitedTableIdentity>,
    /// The trusted activation commitment of the request. The formal request
    /// verifier requires it; the presence of this field in the selected
    /// request is the operator's pin, never a proof value.
    pub activation_commitment: Option<ActivationCommitmentIdentity>,
}

impl TrustedRequest {
    /// The trusted limited table identity. It is present exactly for the gpt2
    /// output-head cases.
    pub fn trusted_limited_table_identity(&self) -> Result<&LimitedTableIdentity> {
        self.limited_table_identity
            .as_ref()
            .ok_or(Error::InvalidConfig)
    }

    /// The trusted activation commitment pin. A request without one fails
    /// closed; the verifier never substitutes a proof-reported commitment.
    pub fn trusted_activation_commitment(&self) -> Result<&ActivationCommitmentIdentity> {
        self.activation_commitment
            .as_ref()
            .ok_or(Error::InvalidConfig)
    }
}

fn select_output(
    case: &PolicyCase,
    kind: OutputKind,
    sequence: Option<usize>,
) -> Result<OutputExpectation> {
    match case.output_mode.as_str() {
        "none" => {
            if case.expected_output_sha256.is_some()
                || case.output_digest_algorithm.is_some()
                || case.output_type.is_some()
                || case.output_encoding.is_some()
                || case.output_elements.is_some()
            {
                return Err(Error::InvalidConfig);
            }
            Ok(OutputExpectation {
                mode: OutputMode::None,
                kind,
                sha256: None,
                elements: None,
            })
        }
        "expected" => {
            let sha256 = digest_from_hex(
                case.expected_output_sha256
                    .as_deref()
                    .ok_or(Error::InvalidConfig)?,
            )?;
            if case.output_digest_algorithm.as_deref() != Some("sha256")
                || case.output_encoding.as_deref() != Some("i32-le")
                || case.output_type.as_deref() != Some(kind.name())
            {
                return Err(Error::InvalidConfig);
            }
            let elements = case.output_elements.ok_or(Error::InvalidConfig)?;
            if elements == 0 {
                return Err(Error::InvalidConfig);
            }
            if let Some(sequence) = sequence {
                if elements != sequence {
                    return Err(Error::InvalidConfig);
                }
            }
            Ok(OutputExpectation {
                mode: OutputMode::Expected,
                kind,
                sha256: Some(sha256),
                elements: Some(elements),
            })
        }
        _ => Err(Error::InvalidConfig),
    }
}

/// Checks the document's own version and the binding of every case entry to its
/// limited table identity. A gpt2 case entry without an identity, an identity
/// from another case or model version, an identity on another model, and an
/// old identity version all fail here.
fn validate_document(document: &PolicyDocument) -> Result<()> {
    if document.policy_version != POLICY_VERSION {
        return Err(Error::UnsupportedVersion);
    }
    if document.model.is_empty() {
        return Err(Error::InvalidConfig);
    }
    for case in &document.case {
        match &case.limited_table_identity {
            Some(identity) => {
                identity.validate()?;
                let anchor_id = digest_from_hex(&case.anchor_id)?;
                let model_id = digest_from_hex(&case.model_id)?;
                if document.model != "gpt2"
                    || identity.model != document.model
                    || identity.model_version != document.model_version
                    || identity.case_id != case.case_id
                    || identity.anchor_id != anchor_id
                    || identity.model_id != model_id
                {
                    return Err(Error::StatementMismatch);
                }
            }
            None => {
                if document.model == "gpt2" {
                    return Err(Error::InvalidConfig);
                }
            }
        }
        match &case.activation_commitment {
            Some(identity) => {
                identity.validate()?;
                let anchor_id = digest_from_hex(&case.anchor_id)?;
                let model_id = digest_from_hex(&case.model_id)?;
                if document.model != "gpt2"
                    || identity.model != document.model
                    || identity.model_version != document.model_version
                    || identity.case_id != case.case_id
                    || identity.anchor_id != anchor_id
                    || identity.model_id != model_id
                {
                    return Err(Error::StatementMismatch);
                }
            }
            None => {}
        }
    }
    Ok(())
}

pub fn parse(bytes: &[u8]) -> Result<PolicyDocument> {
    let text = std::str::from_utf8(bytes).map_err(|_| Error::InvalidEncoding)?;
    let document: PolicyDocument = toml::from_str(text).map_err(|_| Error::InvalidEncoding)?;
    validate_document(&document)?;
    Ok(document)
}

pub fn select(
    document: &PolicyDocument,
    case_id: &str,
    model: &str,
    input_len: usize,
) -> Result<TrustedRequest> {
    if document.policy_version != POLICY_VERSION {
        return Err(Error::UnsupportedVersion);
    }
    if document.model != model {
        return Err(Error::StatementMismatch);
    }
    if !matches!(model, "gpt2" | "gemma3" | "timesfm") {
        return Err(Error::UnsupportedPrimitive);
    }
    let mut selected: Option<&PolicyCase> = None;
    let mut seen = std::collections::BTreeSet::new();
    for case in &document.case {
        if case.case_id.is_empty() || !seen.insert(case.case_id.clone()) {
            return Err(Error::StatementMismatch);
        }
        if case.case_id == case_id {
            if selected.is_some() {
                return Err(Error::StatementMismatch);
            }
            selected = Some(case);
        }
    }
    let case = selected.ok_or(Error::StatementMismatch)?;
    match model {
        "gpt2" => {
            if case.raw_input_length.is_some() || case.patch_tokens.is_some() {
                return Err(Error::InvalidConfig);
            }
            let sequence = case.sequence_length.ok_or(Error::InvalidConfig)?;
            if sequence == 0 || sequence > TOKEN_MAX_SEQUENCE || sequence != input_len {
                return Err(Error::StatementMismatch);
            }
            let anchor_id = digest_from_hex(&case.anchor_id)?;
            let model_id = digest_from_hex(&case.model_id)?;
            let identity = case
                .limited_table_identity
                .clone()
                .ok_or(Error::InvalidConfig)?;
            if identity.model != document.model
                || identity.model_version != document.model_version
                || identity.case_id != case.case_id
                || identity.anchor_id != anchor_id
                || identity.model_id != model_id
            {
                return Err(Error::StatementMismatch);
            }
            if let Some(activation) = &case.activation_commitment {
                if activation.model != document.model
                    || activation.model_version != document.model_version
                    || activation.case_id != case.case_id
                    || activation.anchor_id != anchor_id
                    || activation.model_id != model_id
                {
                    return Err(Error::StatementMismatch);
                }
            }
            let output = select_output(case, OutputKind::Argmax, Some(sequence))?;
            Ok(TrustedRequest {
                model: document.model.clone(),
                model_version: document.model_version,
                case_id: case.case_id.clone(),
                sequence_length: Some(sequence),
                raw_input_length: None,
                patch_tokens: None,
                anchor_id,
                model_id,
                output,
                limited_table_identity: Some(identity),
                activation_commitment: case.activation_commitment.clone(),
            })
        }
        "gemma3" => {
            if case.limited_table_identity.is_some() {
                return Err(Error::InvalidConfig);
            }
            if case.raw_input_length.is_some() || case.patch_tokens.is_some() {
                return Err(Error::InvalidConfig);
            }
            let sequence = case.sequence_length.ok_or(Error::InvalidConfig)?;
            if sequence == 0 || sequence > TOKEN_MAX_SEQUENCE || sequence != input_len {
                return Err(Error::StatementMismatch);
            }
            let output = select_output(case, OutputKind::Argmax, Some(sequence))?;
            Ok(TrustedRequest {
                model: document.model.clone(),
                model_version: document.model_version,
                case_id: case.case_id.clone(),
                sequence_length: Some(sequence),
                raw_input_length: None,
                patch_tokens: None,
                anchor_id: digest_from_hex(&case.anchor_id)?,
                model_id: digest_from_hex(&case.model_id)?,
                output,
                limited_table_identity: None,
                activation_commitment: None,
            })
        }
        "timesfm" => {
            if case.limited_table_identity.is_some() {
                return Err(Error::InvalidConfig);
            }
            if case.sequence_length.is_some() {
                return Err(Error::InvalidConfig);
            }
            let raw = case.raw_input_length.ok_or(Error::InvalidConfig)?;
            let patches = case.patch_tokens.ok_or(Error::InvalidConfig)?;
            if raw != TIMESFM_RAW_POINTS || patches != TIMESFM_PATCH_TOKENS || raw != input_len {
                return Err(Error::StatementMismatch);
            }
            let output = select_output(case, OutputKind::IntOutput, None)?;
            Ok(TrustedRequest {
                model: document.model.clone(),
                model_version: document.model_version,
                case_id: case.case_id.clone(),
                sequence_length: None,
                raw_input_length: Some(raw),
                patch_tokens: Some(patches),
                anchor_id: digest_from_hex(&case.anchor_id)?,
                model_id: digest_from_hex(&case.model_id)?,
                output,
                limited_table_identity: None,
                activation_commitment: None,
            })
        }
        _ => Err(Error::UnsupportedPrimitive),
    }
}

/// Replaces one case entry's limited table identity with a provisioned one and
/// validates the whole document. The operator calls this after commit, when the
/// committed public reader and record are known.
pub fn bind_limited_table_identity(
    document: &PolicyDocument,
    case_id: &str,
    identity: LimitedTableIdentity,
) -> Result<PolicyDocument> {
    if identity.case_id != case_id
        || identity.model != document.model
        || identity.model_version != document.model_version
    {
        return Err(Error::StatementMismatch);
    }
    let mut updated = document.clone();
    let mut found = false;
    for case in &mut updated.case {
        if case.case_id == case_id {
            if found {
                return Err(Error::StatementMismatch);
            }
            found = true;
            case.limited_table_identity = Some(identity.clone());
        }
    }
    if !found {
        return Err(Error::StatementMismatch);
    }
    validate_document(&updated)?;
    Ok(updated)
}

/// Replaces one case entry's activation commitment with a provisioned one and
/// validates the whole document. The operator calls this after the trusted
/// activation commitment is known; a prover does not call it on its own copy.
pub fn bind_activation_commitment(
    document: &PolicyDocument,
    case_id: &str,
    identity: ActivationCommitmentIdentity,
) -> Result<PolicyDocument> {
    if identity.case_id != case_id
        || identity.model != document.model
        || identity.model_version != document.model_version
    {
        return Err(Error::StatementMismatch);
    }
    let mut updated = document.clone();
    let mut found = false;
    for case in &mut updated.case {
        if case.case_id == case_id {
            if found {
                return Err(Error::StatementMismatch);
            }
            found = true;
            case.activation_commitment = Some(identity.clone());
        }
    }
    if !found {
        return Err(Error::StatementMismatch);
    }
    validate_document(&updated)?;
    Ok(updated)
}

pub fn encode_document(document: &PolicyDocument) -> Result<String> {
    validate_document(document)?;
    toml::to_string_pretty(document).map_err(|_| Error::InvalidEncoding)
}

/// Writes one operator policy copy. The caller decides where the trusted copy
/// lives; this function only serializes and writes it.
pub fn write_document(path: &std::path::Path, document: &PolicyDocument) -> Result<()> {
    let text = encode_document(document)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|_| Error::Io)?;
    }
    std::fs::write(path, text).map_err(|_| Error::Io)
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const B: &str = "2222222222222222222222222222222222222222222222222222222222222222";
    const C: &str = "3333333333333333333333333333333333333333333333333333333333333333";
    const D: &str = "4444444444444444444444444444444444444444444444444444444444444444";
    const E: &str = "5555555555555555555555555555555555555555555555555555555555555555";
    const F: &str = "6666666666666666666666666666666666666666666666666666666666666666";
    const G: &str = "7777777777777777777777777777777777777777777777777777777777777777";

    fn identity_block(case_id: &str, anchor: &str, model_id: &str) -> String {
        format!(
            "\n[case.limited_table_identity]\nversion = 1\nmodel = \"gpt2\"\nmodel_version = 1\ncase_id = \"{case_id}\"\nanchor_id = \"{anchor}\"\nmodel_id = \"{model_id}\"\nasset_id = 7\nasset_table_identity = \"{C}\"\ntable_root = \"{D}\"\ntable_record_digest = \"{E}\"\ntable_configuration = \"{F}\"\ncode_log_n = 3\ncode_ell = 5\ncode_r = 3\ntable_v = 1\nanswers = 1\nsalt_bytes = 40\ndistance_radius = 1\ndistance_later_radius = 1\ndistance_m_once = 0\ndistance_m_later = 0\nmatrix_rows_log = 1\nk_lo = 1\nk_col = 2\nprotocol_identity = \"{G}\"\n"
        )
    }

    fn token_policy() -> String {
        format!(
            "policy_version = 1\nmodel = \"gpt2\"\nmodel_version = 1\n\n[[case]]\ncase_id = \"gpt2-seq16\"\nsequence_length = 16\nanchor_id = \"{A}\"\nmodel_id = \"{B}\"\noutput_mode = \"none\"\n{}\n\n[[case]]\ncase_id = \"gpt2-seq512\"\nsequence_length = 512\nanchor_id = \"{A}\"\nmodel_id = \"{B}\"\noutput_mode = \"none\"\n{}\n",
            identity_block("gpt2-seq16", A, B),
            identity_block("gpt2-seq512", A, B),
        )
    }

    fn sample_identity() -> LimitedTableIdentity {
        let document = parse(token_policy().as_bytes()).expect("policy");
        document.case[0]
            .limited_table_identity
            .clone()
            .expect("identity")
    }

    #[test]
    fn parses_and_selects_token_case() {
        let document = parse(token_policy().as_bytes()).unwrap();
        let selected = select(&document, "gpt2-seq16", "gpt2", 16).unwrap();
        assert_eq!(selected.sequence_length, Some(16));
        assert_eq!(selected.anchor_id.0, [0x11; 32]);
        assert_eq!(selected.model_id.0, [0x22; 32]);
        let identity = selected.trusted_limited_table_identity().unwrap();
        assert_eq!(identity.case_id, "gpt2-seq16");
        assert_eq!(identity.table_root.0, [0x44; 32]);
    }

    #[test]
    fn parses_and_selects_timesfm_case() {
        let text = format!(
            "policy_version = 1\nmodel = \"timesfm\"\nmodel_version = 1\n\n[[case]]\ncase_id = \"timesfm-16patch\"\nraw_input_length = 512\npatch_tokens = 16\nanchor_id = \"{A}\"\nmodel_id = \"{B}\"\noutput_mode = \"none\"\n"
        );
        let document = parse(text.as_bytes()).unwrap();
        let selected = select(&document, "timesfm-16patch", "timesfm", 512).unwrap();
        assert_eq!(selected.raw_input_length, Some(512));
        assert_eq!(selected.patch_tokens, Some(16));
        assert!(selected.limited_table_identity.is_none());
    }

    #[test]
    fn rejects_unknown_fields_and_duplicate_keys() {
        let unknown = token_policy().replace("model_version = 1", "model_version = 1\nextra = 2");
        assert!(parse(unknown.as_bytes()).is_err());
        let duplicate = token_policy().replace(
            "policy_version = 1",
            "policy_version = 1\npolicy_version = 1",
        );
        assert!(parse(duplicate.as_bytes()).is_err());
    }

    #[test]
    fn rejects_bad_shapes_and_versions() {
        let document = parse(token_policy().as_bytes()).unwrap();
        assert!(select(&document, "gpt2-seq16", "gpt2", 512).is_err());
        assert!(select(&document, "gpt2-seq16", "gemma3", 16).is_err());
        assert!(select(&document, "missing", "gpt2", 16).is_err());
        let mut version = document.clone();
        version.policy_version = 2;
        assert_eq!(
            select(&version, "gpt2-seq16", "gpt2", 16),
            Err(Error::UnsupportedVersion)
        );
        let broken = token_policy().replace(A, "aa");
        assert!(parse(broken.as_bytes()).is_err());
    }

    #[test]
    fn rejects_duplicate_case_ids() {
        let duplicate = token_policy().replace("gpt2-seq512", "gpt2-seq16");
        let document = parse(duplicate.as_bytes()).unwrap();
        assert!(select(&document, "gpt2-seq16", "gpt2", 16).is_err());
    }

    #[test]
    fn selects_expected_output_mode_and_rejects_partial_declarations() {
        let identity = identity_block("gpt2-seq16", A, B);
        let text = format!(
            "policy_version = 1\nmodel = \"gpt2\"\nmodel_version = 1\n\n[[case]]\ncase_id = \"gpt2-seq16\"\nsequence_length = 16\nanchor_id = \"{A}\"\nmodel_id = \"{B}\"\noutput_mode = \"expected\"\nexpected_output_sha256 = \"{C}\"\noutput_digest_algorithm = \"sha256\"\noutput_type = \"argmax\"\noutput_encoding = \"i32-le\"\noutput_elements = 16\n{identity}"
        );
        let document = parse(text.as_bytes()).unwrap();
        let selected = select(&document, "gpt2-seq16", "gpt2", 16).unwrap();
        assert_eq!(selected.output.mode, OutputMode::Expected);
        assert_eq!(selected.output.kind, OutputKind::Argmax);
        assert_eq!(selected.output.elements, Some(16));
        assert_eq!(selected.output.sha256.unwrap().0, [0x33; 32]);

        for broken in [
            text.replace("output_type = \"argmax\"", "output_type = \"int_output\""),
            text.replace("output_encoding = \"i32-le\"", "output_encoding = \"i64-le\""),
            text.replace("output_elements = 16", "output_elements = 15"),
            text.replace(
                "expected_output_sha256 = \"3333333333333333333333333333333333333333333333333333333333333333\"\n",
                "",
            ),
        ] {
            let document = parse(broken.as_bytes()).unwrap();
            assert!(select(&document, "gpt2-seq16", "gpt2", 16).is_err());
        }

        let document = parse(token_policy().as_bytes()).unwrap();
        let selected = select(&document, "gpt2-seq16", "gpt2", 16).unwrap();
        assert_eq!(selected.output.mode, OutputMode::None);
        assert!(selected.output.sha256.is_none());
    }

    #[test]
    fn rejects_a_gpt2_case_without_an_identity() {
        let text = format!(
            "policy_version = 1\nmodel = \"gpt2\"\nmodel_version = 1\n\n[[case]]\ncase_id = \"gpt2-seq16\"\nsequence_length = 16\nanchor_id = \"{A}\"\nmodel_id = \"{B}\"\noutput_mode = \"none\"\n"
        );
        assert!(parse(text.as_bytes()).is_err());

        let mut document = parse(token_policy().as_bytes()).unwrap();
        document.case[0].limited_table_identity = None;
        assert!(encode_document(&document).is_err());
    }

    #[test]
    fn rejects_an_identity_from_another_case_or_version() {
        let mut document = parse(token_policy().as_bytes()).unwrap();
        document.case[0]
            .limited_table_identity
            .as_mut()
            .unwrap()
            .case_id = "gpt2-seq512".to_owned();
        assert!(encode_document(&document).is_err());

        let mut document = parse(token_policy().as_bytes()).unwrap();
        document.case[0]
            .limited_table_identity
            .as_mut()
            .unwrap()
            .model_version = 2;
        assert!(encode_document(&document).is_err());

        let mut document = parse(token_policy().as_bytes()).unwrap();
        document.case[0]
            .limited_table_identity
            .as_mut()
            .unwrap()
            .anchor_id = Digest([0x99; 32]);
        assert!(encode_document(&document).is_err());

        let old = token_policy().replace(
            "[case.limited_table_identity]\nversion = 1",
            "[case.limited_table_identity]\nversion = 0",
        );
        assert!(parse(old.as_bytes()).is_err());
    }

    #[test]
    fn a_timesfm_case_rejects_an_identity() {
        let text = format!(
            "policy_version = 1\nmodel = \"timesfm\"\nmodel_version = 1\n\n[[case]]\ncase_id = \"timesfm-16patch\"\nraw_input_length = 512\npatch_tokens = 16\nanchor_id = \"{A}\"\nmodel_id = \"{B}\"\noutput_mode = \"none\"\n{}",
            identity_block("timesfm-16patch", A, B)
                .replace("model = \"gpt2\"", "model = \"timesfm\"")
        );
        assert!(parse(text.as_bytes()).is_err());
    }

    #[test]
    fn the_identity_digest_is_domain_separated_and_versioned() {
        let identity = sample_identity();
        let digest = identity.digest().unwrap();
        assert_ne!(digest, digest_bytes(&identity.table_root.0));
        assert_ne!(digest, digest_bytes(&identity.encode().unwrap()[..16]));

        let mut writer = CanonicalWriter::new();
        writer.put_bytes(LIMITED_TABLE_IDENTITY_DOMAIN);
        let domain = writer.into_bytes();
        assert!(identity.encode().unwrap().starts_with(&domain));

        let mut other = identity.clone();
        other.version = LIMITED_TABLE_IDENTITY_VERSION + 1;
        assert_eq!(other.digest(), Err(Error::UnsupportedVersion));
    }

    #[test]
    fn every_identity_field_changes_the_digest_or_is_rejected() {
        let identity = sample_identity();
        let base = identity.digest().unwrap();
        let mutators: Vec<(&str, Box<dyn Fn(&mut LimitedTableIdentity)>)> = vec![
            ("model", Box::new(|value| value.model = "gpt2x".into())),
            ("model_version", Box::new(|value| value.model_version = 2)),
            ("case_id", Box::new(|value| value.case_id = "other".into())),
            (
                "anchor_id",
                Box::new(|value| value.anchor_id = Digest([0x21; 32])),
            ),
            (
                "model_id",
                Box::new(|value| value.model_id = Digest([0x28; 32])),
            ),
            ("asset_id", Box::new(|value| value.asset_id += 1)),
            (
                "asset_table_identity",
                Box::new(|value| value.asset_table_identity = Digest([0x23; 32])),
            ),
            (
                "table_root",
                Box::new(|value| value.table_root = Digest([0x24; 32])),
            ),
            (
                "table_record_digest",
                Box::new(|value| value.table_record_digest = Digest([0x25; 32])),
            ),
            (
                "table_configuration",
                Box::new(|value| value.table_configuration = Digest([0x26; 32])),
            ),
            ("code_log_n", Box::new(|value| value.code_log_n += 1)),
            ("code_ell", Box::new(|value| value.code_ell += 8)),
            ("code_r", Box::new(|value| value.code_r += 1)),
            ("table_v", Box::new(|value| value.table_v += 1)),
            ("answers", Box::new(|value| value.answers += 1)),
            ("salt_bytes", Box::new(|value| value.salt_bytes += 8)),
            (
                "distance_radius",
                Box::new(|value| value.distance_radius += 1),
            ),
            (
                "distance_later_radius",
                Box::new(|value| value.distance_later_radius += 1),
            ),
            (
                "distance_m_once",
                Box::new(|value| value.distance_m_once += 1),
            ),
            (
                "distance_m_later",
                Box::new(|value| value.distance_m_later += 1),
            ),
            (
                "matrix_rows_log",
                Box::new(|value| value.matrix_rows_log += 1),
            ),
            ("k_lo", Box::new(|value| value.k_lo += 1)),
            ("k_col", Box::new(|value| value.k_col += 1)),
            (
                "protocol_identity",
                Box::new(|value| value.protocol_identity = Digest([0x27; 32])),
            ),
        ];
        for (name, edit) in mutators {
            let mut changed = identity.clone();
            edit(&mut changed);
            match changed.digest() {
                Ok(digest) => assert_ne!(digest, base, "{name} did not change the digest"),
                Err(_) => {}
            }
        }
    }

    #[test]
    fn the_identity_encoding_round_trips_and_rejects_malformed_fields() {
        let identity = sample_identity();
        let bytes = identity.encode().unwrap();
        assert_eq!(LimitedTableIdentity::decode(&bytes).unwrap(), identity);

        let mut wrong_domain = bytes.clone();
        wrong_domain[8] ^= 1;
        assert!(LimitedTableIdentity::decode(&wrong_domain).is_err());

        let version_at = 8 + LIMITED_TABLE_IDENTITY_DOMAIN.len();
        let mut wrong_version = bytes.clone();
        wrong_version[version_at] = 2;
        assert_eq!(
            LimitedTableIdentity::decode(&wrong_version),
            Err(Error::UnsupportedVersion)
        );

        let count_at = version_at + 8;
        let mut wrong_count = bytes.clone();
        wrong_count[count_at..count_at + 8].copy_from_slice(&23u64.to_le_bytes());
        assert!(LimitedTableIdentity::decode(&wrong_count).is_err());

        let first_tag_at = count_at + 8;
        let mut unknown_tag = bytes.clone();
        unknown_tag[first_tag_at..first_tag_at + 8].copy_from_slice(&99u64.to_le_bytes());
        assert!(LimitedTableIdentity::decode(&unknown_tag).is_err());

        let second_tag_at = first_tag_at + 8 + 8 + identity.model.len();
        let mut duplicate_tag = bytes.clone();
        duplicate_tag[second_tag_at..second_tag_at + 8].copy_from_slice(&1u64.to_le_bytes());
        assert!(LimitedTableIdentity::decode(&duplicate_tag).is_err());

        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(LimitedTableIdentity::decode(&trailing).is_err());

        let mut truncated = bytes.clone();
        truncated.truncate(truncated.len() - 1);
        assert!(LimitedTableIdentity::decode(&truncated).is_err());

        let mut wide_field = bytes.clone();
        let first_length_at = first_tag_at + 8;
        wide_field[first_length_at..first_length_at + 8]
            .copy_from_slice(&(1u64 << 40).to_le_bytes());
        assert!(LimitedTableIdentity::decode(&wide_field).is_err());
    }

    #[test]
    fn a_committed_identity_must_match_the_policy_identity() {
        let identity = sample_identity();
        let shape = identity.shape().unwrap();
        identity
            .check_committed(
                identity.table_root,
                identity.table_record_digest,
                identity.table_configuration,
                shape,
                identity.matrix_rows_log,
                identity.k_lo,
                identity.k_col,
                identity.protocol_identity,
            )
            .expect("matching committed identity");

        assert!(identity
            .check_committed(
                Digest([0x91; 32]),
                identity.table_record_digest,
                identity.table_configuration,
                shape,
                identity.matrix_rows_log,
                identity.k_lo,
                identity.k_col,
                identity.protocol_identity,
            )
            .is_err());
        assert!(identity
            .check_committed(
                identity.table_root,
                Digest([0x92; 32]),
                identity.table_configuration,
                shape,
                identity.matrix_rows_log,
                identity.k_lo,
                identity.k_col,
                identity.protocol_identity,
            )
            .is_err());
        assert!(identity
            .check_committed(
                identity.table_root,
                identity.table_record_digest,
                Digest([0x93; 32]),
                shape,
                identity.matrix_rows_log,
                identity.k_lo,
                identity.k_col,
                identity.protocol_identity,
            )
            .is_err());
        assert!(identity
            .check_committed(
                identity.table_root,
                identity.table_record_digest,
                identity.table_configuration,
                shape,
                identity.matrix_rows_log,
                identity.k_lo,
                identity.k_col,
                Digest([0x94; 32]),
            )
            .is_err());

        let other_shape = IndexShape {
            code: build_code_plan(4, 8, 4).unwrap(),
            v: 2,
            answers: 1,
            salt_bytes: 40,
            distances: TableDistances {
                radius: 1,
                later_radius: 1,
                m_once: 0,
                m_later: 0,
            },
        };
        assert!(identity
            .check_committed(
                identity.table_root,
                identity.table_record_digest,
                identity.table_configuration,
                other_shape,
                2,
                1,
                3,
                identity.protocol_identity,
            )
            .is_err());
        assert!(identity
            .check_committed(
                identity.table_root,
                identity.table_record_digest,
                identity.table_configuration,
                shape,
                identity.matrix_rows_log + 1,
                identity.k_lo,
                identity.k_col,
                identity.protocol_identity,
            )
            .is_err());
        assert!(identity
            .check_committed(
                identity.table_root,
                identity.table_record_digest,
                identity.table_configuration,
                shape,
                identity.matrix_rows_log,
                identity.k_lo + 1,
                identity.k_col,
                identity.protocol_identity,
            )
            .is_err());
    }
    fn activation_block(case_id: &str, anchor: &str, model_id: &str) -> String {
        format!(
            "\n[case.activation_commitment]\nversion = 1\nmodel = \"gpt2\"\nmodel_version = 1\ncase_id = \"{case_id}\"\nanchor_id = \"{anchor}\"\nmodel_id = \"{model_id}\"\nactivation_root = \"{C}\"\npositions = 16\nhidden = 1024\nsalt_bytes = 40\nconfiguration = \"{D}\"\nprotocol_identity = \"{E}\"\n"
        )
    }

    fn policy_with_activation() -> String {
        format!(
            "policy_version = 1\nmodel = \"gpt2\"\nmodel_version = 1\n\n[[case]]\ncase_id = \"gpt2-seq16\"\nsequence_length = 16\nanchor_id = \"{A}\"\nmodel_id = \"{B}\"\noutput_mode = \"none\"\n{}{}\n\n[[case]]\ncase_id = \"gpt2-seq512\"\nsequence_length = 512\nanchor_id = \"{A}\"\nmodel_id = \"{B}\"\noutput_mode = \"none\"\n{}\n",
            identity_block("gpt2-seq16", A, B),
            activation_block("gpt2-seq16", A, B),
            identity_block("gpt2-seq512", A, B),
        )
    }

    #[test]
    fn activation_commitment_is_selected_and_bound() {
        let document = parse(policy_with_activation().as_bytes()).unwrap();
        let selected = select(&document, "gpt2-seq16", "gpt2", 16).unwrap();
        let identity = selected.trusted_activation_commitment().unwrap();
        assert_eq!(identity.activation_root.0, [0x33; 32]);
        assert_eq!(identity.positions, 16);
        assert_eq!(identity.hidden, 1024);
        assert_eq!(identity.salt_bytes, 40);
        let commitment = identity.commitment().unwrap();
        assert_eq!(commitment.root, identity.activation_root);
        assert_eq!(commitment.shape.positions, 16);
        assert_eq!(commitment.shape.hidden, 1024);
        assert_eq!(commitment.configuration, Digest([0x44; 32]));

        // The second case has no activation pin: the accessor fails closed.
        let selected = select(&document, "gpt2-seq512", "gpt2", 512).unwrap();
        assert!(selected.trusted_activation_commitment().is_err());

        // An operator bind adds the pin to the second case.
        let mut second = identity.clone();
        second.case_id = "gpt2-seq512".into();
        let bound = bind_activation_commitment(&document, "gpt2-seq512", second).unwrap();
        let selected = select(&bound, "gpt2-seq512", "gpt2", 512).unwrap();
        assert_eq!(
            selected.trusted_activation_commitment().unwrap().activation_root,
            identity.activation_root
        );
        // A bind for another case is rejected.
        assert!(bind_activation_commitment(&document, "gpt2-seq16", {
            let mut other = identity.clone();
            other.case_id = "gpt2-seq512".into();
            other
        })
        .is_err());
    }

    #[test]
    fn activation_commitment_binds_the_case_model_and_version() {
        let other_case = format!(
            "policy_version = 1\nmodel = \"gpt2\"\nmodel_version = 1\n\n[[case]]\ncase_id = \"gpt2-seq16\"\nsequence_length = 16\nanchor_id = \"{A}\"\nmodel_id = \"{B}\"\noutput_mode = \"none\"\n{}{}\n",
            identity_block("gpt2-seq16", A, B),
            activation_block("gpt2-seq512", A, B),
        );
        assert!(parse(other_case.as_bytes()).is_err());

        let other_model = format!(
            "policy_version = 1\nmodel = \"gpt2\"\nmodel_version = 1\n\n[[case]]\ncase_id = \"gpt2-seq16\"\nsequence_length = 16\nanchor_id = \"{A}\"\nmodel_id = \"{B}\"\noutput_mode = \"none\"\n{}{}\n",
            identity_block("gpt2-seq16", A, B),
            activation_block("gpt2-seq16", A, F),
        );
        assert!(parse(other_model.as_bytes()).is_err());

        let old_version = policy_with_activation().replace(
            "[case.activation_commitment]\nversion = 1",
            "[case.activation_commitment]\nversion = 2",
        );
        assert!(parse(old_version.as_bytes()).is_err());
    }

    #[test]
    fn activation_commitment_encoding_is_canonical() {
        let document = parse(policy_with_activation().as_bytes()).unwrap();
        let identity = document.case[0]
            .activation_commitment
            .clone()
            .expect("identity");
        let bytes = identity.encode().unwrap();
        let decoded = ActivationCommitmentIdentity::decode(&bytes).unwrap();
        assert_eq!(decoded, identity);
        assert_eq!(decoded.digest().unwrap(), identity.digest().unwrap());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(ActivationCommitmentIdentity::decode(&trailing).is_err());
        let mut changed = identity.clone();
        changed.activation_root = Digest([0x99; 32]);
        assert_ne!(changed.digest().unwrap(), identity.digest().unwrap());
        let mut missing = bytes.clone();
        missing.truncate(missing.len() - 1);
        assert!(ActivationCommitmentIdentity::decode(&missing).is_err());
    }

}
