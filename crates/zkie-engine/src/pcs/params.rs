//! Trusted parameter policy.
//!
//! Parameters are matched by digest against a caller-provided trust set. A
//! missing or zero digest is rejected; there is no implicit "use whatever the
//! proof reports" path.

use zkie_core::common::field::FieldDescriptor;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParameterDescriptor {
    pub digest: Digest,
    pub protocol_version: u16,
    pub field: &'static FieldDescriptor,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParameterPolicy {
    pub trusted_digests: Vec<Digest>,
    pub require_trusted: bool,
}

impl ParameterPolicy {
    pub fn validate(&self, descriptor: &ParameterDescriptor) -> Result<()> {
        if descriptor.digest == Digest::ZERO {
            return Err(Error::InvalidConfig);
        }
        if self.require_trusted && !self.trusted_digests.contains(&descriptor.digest) {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }
}

pub trait ParameterSource {
    fn resolve(&self, digest: Digest) -> Result<ParameterDescriptor>;
}
