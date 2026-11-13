//! Frozen interface of the norm gain/bias parameter tables.
//!
//! The norm output tail multiplies by the per-column gain and adds the
//! per-column bias. Both parameters are carried by committed encoded tables
//! and shared by every position. This module fixes the identity, the layout
//! and the plan counts that the later stages share; it neither builds a table
//! nor consumes an evaluation.
//!
//! The table identity is an external trusted input in the fixture stage: a
//! proof reports a local conclusion relative to that identity and does not
//! claim the model source is closed. The i32 mapping flag is a publication
//! assumption, not a range proof; the relation's own digit and range checks
//! cover the value ranges.

use zkie_core::artifact::digest_bytes;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::foldeq::reader_table_identity;
use zkie_engine::pcs::request::table::IndexReader;
use zkie_engine::primitives::codec::CanonicalWriter;

/// Version of the norm parameter table interface.
pub const NORM_PARAM_TABLE_VERSION: u64 = 1;

/// The norm parameter one table row carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormParamKind {
    Gain,
    Bias,
}

impl NormParamKind {
    pub fn encode(self) -> u64 {
        match self {
            Self::Gain => 0,
            Self::Bias => 1,
        }
    }
}

/// The identity of one norm parameter row in a committed encoded table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormParamTableIdentity {
    pub version: u64,
    pub kind: NormParamKind,
    pub model_identity: Digest,
    pub asset_table_identity: Digest,
    pub reader: IndexReader,
    pub row: u64,
    pub width: u64,
    pub valid_min: i64,
    pub valid_max: i64,
    pub trusted_i32_mapping: bool,
}

impl NormParamTableIdentity {
    pub fn validate(&self) -> Result<()> {
        if self.version != NORM_PARAM_TABLE_VERSION
            || self.model_identity == Digest::ZERO
            || self.asset_table_identity == Digest::ZERO
            || !self.trusted_i32_mapping
        {
            return Err(Error::InvalidConfig);
        }
        if self.reader.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        if reader_table_identity(&self.reader)? == Digest::ZERO {
            return Err(Error::InvalidConfig);
        }
        let rows = self.reader.shape.rows_checked()?;
        if self.row >= rows {
            return Err(Error::InvalidConfig);
        }
        if self.width == 0
            || !self.width.is_power_of_two()
            || self.width > self.reader.shape.plain()
        {
            return Err(Error::InvalidConfig);
        }
        if self.valid_min > self.valid_max
            || self.valid_min < i32::MIN as i64
            || self.valid_max > i32::MAX as i64
        {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }

    /// The digest of the identity, including the committed table identity.
    /// The plan validation checks the reader before this value is absorbed.
    pub fn digest(&self) -> Digest {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(self.version);
        writer.put_u64(self.kind.encode());
        writer.put_digest(&self.model_identity);
        writer.put_digest(&self.asset_table_identity);
        put_reader_identity(&mut writer, &self.reader);
        writer.put_u64(self.row);
        writer.put_u64(self.width);
        writer.put_u64(self.valid_min as u64);
        writer.put_u64(self.valid_max as u64);
        writer.put_u64(self.trusted_i32_mapping as u64);
        digest_bytes(&writer.into_bytes())
    }
}

/// Encodes the committed reader identity: root, configuration, shape numbers,
/// distances and the table record.
fn put_reader_identity(writer: &mut CanonicalWriter, reader: &IndexReader) {
    writer.put_digest(&reader.root);
    writer.put_digest(&reader.configuration);
    writer.put_u64(reader.shape.code.log_n as u64);
    writer.put_u64(reader.shape.code.ell);
    writer.put_u64(reader.shape.code.r);
    writer.put_u64(reader.shape.v as u64);
    writer.put_u64(reader.shape.answers as u64);
    writer.put_u64(reader.shape.salt_bytes as u64);
    writer.put_u64(reader.shape.distances.radius);
    writer.put_u64(reader.shape.distances.later_radius);
    writer.put_u64(reader.shape.distances.m_once);
    writer.put_u64(reader.shape.distances.m_later);
    writer.put_u64(reader.record.y.len() as u64);
    for value in &reader.record.y {
        writer.put_fp3(value);
    }
}

/// The plan values of the two parameter tables. Two identities with the same
/// reader are two rows of one committed table; two different readers are two
/// separate tables. The counts are per table and come from the trusted plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormParamTablesPlan {
    pub gain: NormParamTableIdentity,
    pub bias: NormParamTableIdentity,
    pub column_draws: usize,
    pub seed_budget: usize,
}

impl NormParamTablesPlan {
    pub fn validate(&self) -> Result<()> {
        self.gain.validate()?;
        self.bias.validate()?;
        if self.gain.kind != NormParamKind::Gain || self.bias.kind != NormParamKind::Bias {
            return Err(Error::InvalidConfig);
        }
        if self.gain.reader == self.bias.reader && self.gain.row == self.bias.row {
            return Err(Error::InvalidConfig);
        }
        if self.column_draws == 0 || self.seed_budget == 0 {
            return Err(Error::InvalidConfig);
        }
        for identity in [&self.gain, &self.bias] {
            if self.column_draws as u64 > identity.reader.shape.code.n() {
                return Err(Error::InvalidConfig);
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Digest {
        let mut writer = CanonicalWriter::new();
        writer.put_digest(&self.gain.digest());
        writer.put_digest(&self.bias.digest());
        writer.put_u64(self.column_draws as u64);
        writer.put_u64(self.seed_budget as u64);
        digest_bytes(&writer.into_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::code::build_code_plan;
    use zkie_engine::pcs::request::field::Fp3;
    use zkie_engine::pcs::request::params::LEAF_SALT_MIN_BYTES;
    use zkie_engine::pcs::request::table::{IndexRecord, IndexShape, TableDistances};

    fn shape() -> IndexShape {
        IndexShape {
            code: build_code_plan(4, 8, 1).unwrap(),
            v: 3,
            answers: 1,
            salt_bytes: LEAF_SALT_MIN_BYTES,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        }
    }

    fn reader(root: u8, configuration: u8) -> IndexReader {
        let shape = shape();
        let record = IndexRecord {
            y: vec![Fp3::ZERO; shape.rows() as usize * shape.answers as usize],
        };
        IndexReader {
            shape,
            configuration: Digest([configuration; 32]),
            root: Digest([root; 32]),
            record,
        }
    }

    fn identity(kind: NormParamKind, row: u64, reader: IndexReader) -> NormParamTableIdentity {
        NormParamTableIdentity {
            version: NORM_PARAM_TABLE_VERSION,
            kind,
            model_identity: Digest([1; 32]),
            asset_table_identity: Digest([2; 32]),
            reader,
            row,
            width: 4,
            valid_min: -128,
            valid_max: 127,
            trusted_i32_mapping: true,
        }
    }

    fn packed_plan() -> NormParamTablesPlan {
        NormParamTablesPlan {
            gain: identity(NormParamKind::Gain, 0, reader(9, 7)),
            bias: identity(NormParamKind::Bias, 1, reader(9, 7)),
            column_draws: 2,
            seed_budget: 1,
        }
    }

    #[test]
    fn identity_validates_and_binds_its_fields() {
        let identity = identity(NormParamKind::Gain, 0, reader(9, 7));
        identity.validate().unwrap();
        let digest = identity.digest();
        assert_ne!(digest, Digest::ZERO);

        let mut changed = identity.clone();
        changed.row = 3;
        assert_ne!(changed.digest(), digest);

        let mut changed = identity.clone();
        changed.width = 2;
        assert_ne!(changed.digest(), digest);

        let mut changed = identity.clone();
        changed.reader.root = Digest([8; 32]);
        assert_ne!(changed.digest(), digest);

        let mut changed = identity.clone();
        changed.valid_max = 100;
        assert_ne!(changed.digest(), digest);

        let mut changed = identity.clone();
        changed.kind = NormParamKind::Bias;
        assert_ne!(changed.digest(), digest);
    }

    #[test]
    fn identity_rejects_malformed_fields() {
        let identity = identity(NormParamKind::Gain, 0, reader(9, 7));

        let mut changed = identity.clone();
        changed.version = NORM_PARAM_TABLE_VERSION + 1;
        assert!(changed.validate().is_err());

        let mut changed = identity.clone();
        changed.model_identity = Digest::ZERO;
        assert!(changed.validate().is_err());

        let mut changed = identity.clone();
        changed.asset_table_identity = Digest::ZERO;
        assert!(changed.validate().is_err());

        let mut changed = identity.clone();
        changed.trusted_i32_mapping = false;
        assert!(changed.validate().is_err());

        let mut changed = identity.clone();
        changed.reader.configuration = Digest::ZERO;
        assert!(changed.validate().is_err());

        let mut changed = identity.clone();
        changed.row = 8;
        assert!(changed.validate().is_err());

        for width in [0u64, 3, 6, 8, 16] {
            let mut changed = identity.clone();
            changed.width = width;
            assert!(changed.validate().is_err(), "width {width} must be rejected");
        }

        let mut changed = identity.clone();
        changed.valid_min = 200;
        assert!(changed.validate().is_err());

        let mut changed = identity.clone();
        changed.valid_min = i32::MIN as i64 - 1;
        assert!(changed.validate().is_err());

        let mut changed = identity.clone();
        changed.valid_max = i32::MAX as i64 + 1;
        assert!(changed.validate().is_err());
    }

    #[test]
    fn table_plan_requires_distinct_rows_in_one_table_and_bounds_the_draws() {
        let plan = packed_plan();
        plan.validate().unwrap();
        assert_ne!(plan.digest(), Digest::ZERO);

        let mut changed = plan.clone();
        changed.bias.row = 0;
        assert!(changed.validate().is_err());

        let mut separate = plan.clone();
        separate.bias.reader = reader(8, 7);
        separate.bias.row = 0;
        separate.validate().unwrap();
        assert_ne!(separate.digest(), plan.digest());

        let mut swapped = plan.clone();
        std::mem::swap(&mut swapped.gain, &mut swapped.bias);
        assert!(swapped.validate().is_err());

        let mut changed = plan.clone();
        changed.column_draws = 0;
        assert!(changed.validate().is_err());

        let mut changed = plan.clone();
        changed.column_draws = 17;
        assert!(changed.validate().is_err());

        let mut changed = plan.clone();
        changed.seed_budget = 0;
        assert!(changed.validate().is_err());

        let mut changed = plan;
        changed.column_draws = 3;
        assert_ne!(changed.digest(), packed_plan().digest());
    }
}
