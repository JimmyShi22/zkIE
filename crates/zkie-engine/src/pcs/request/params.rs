//! Protocol constants and the parameter digest.
//!
//! The constants here are fixed by the protocol definition. Instance
//! parameters (code shape and per-chain work targets) are explicit fields and
//! are bound into the parameter digest together with the domain description.

use serde::{Deserialize, Serialize};
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::code::{CodeParams, GENERATOR, MOST_LOG_N, ROOT_32, SUBGROUP_LOG};
use crate::pcs::request::field::{EXTENSION_NONRESIDUE, MODULUS};
use crate::primitives::codec::CanonicalWriter;
use crate::primitives::transcript::{hash_labeled, tag_word, PARAMETERS};

pub const PARAMETERS_SCHEMA_VERSION: u64 = 1;

pub const REQUEST_CHAIN_LABEL: u64 = 0x4652_4d01;
pub const SHARED_CHAIN_LABEL: u64 = 0x4652_4d03;
pub const FINAL_CHAIN_LABEL: u64 = 0x4652_4d04;

pub const DRAW_TRIES: u32 = 6;
pub const NONZERO_TRIES: u32 = 2;

pub const FIRST_QUERY_SALT_BYTES: usize = 32;
pub const LEAF_SALT_MIN_BYTES: usize = 40;
pub const LEAF_SALT_MAX_BYTES: usize = 64;

pub const MAX_WORK_BITS: u8 = 64;
pub const MAX_CHAIN_LABEL: u64 = 1 << 48;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PowPolicy {
    pub request_work_bits: u8,
    pub shared_work_bits: u8,
    pub final_work_bits: u8,
}

impl PowPolicy {
    pub fn new(request_work_bits: u8, shared_work_bits: u8, final_work_bits: u8) -> Result<Self> {
        if request_work_bits > MAX_WORK_BITS
            || shared_work_bits > MAX_WORK_BITS
            || final_work_bits > MAX_WORK_BITS
        {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            request_work_bits,
            shared_work_bits,
            final_work_bits,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestParams {
    pub code: CodeParams,
    pub pow: PowPolicy,
}

impl RequestParams {
    pub fn new(code: CodeParams, pow: PowPolicy) -> Result<Self> {
        crate::pcs::request::code::build_code_plan(code.log_n, code.ell, code.r)?;
        Ok(Self { code, pow })
    }

    pub fn default_with_code(log_n: u32, ell: u64, r: u64) -> Result<Self> {
        Ok(Self {
            code: crate::pcs::request::code::build_code_plan(log_n, ell, r)?,
            pow: PowPolicy::new(0, 0, 0)?,
        })
    }

    /// A digest of the complete domain description and instance parameters.
    pub fn digest(&self) -> Result<Digest> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(PARAMETERS_SCHEMA_VERSION);
        writer.put_u64(MODULUS);
        writer.put_u64(3);
        writer.put_u64(EXTENSION_NONRESIDUE);
        writer.put_u64(GENERATOR);
        writer.put_u64(ROOT_32);
        writer.put_u64(SUBGROUP_LOG as u64);
        writer.put_u64(MOST_LOG_N as u64);
        writer.put_u64(DRAW_TRIES as u64);
        writer.put_u64(NONZERO_TRIES as u64);
        writer.put_u64(FIRST_QUERY_SALT_BYTES as u64);
        writer.put_u64(LEAF_SALT_MIN_BYTES as u64);
        writer.put_u64(LEAF_SALT_MAX_BYTES as u64);
        writer.put_u64(REQUEST_CHAIN_LABEL);
        writer.put_u64(SHARED_CHAIN_LABEL);
        writer.put_u64(FINAL_CHAIN_LABEL);
        writer.put_u64(self.code.log_n as u64);
        writer.put_u64(self.code.ell);
        writer.put_u64(self.code.r);
        writer.put_u64(self.pow.request_work_bits as u64);
        writer.put_u64(self.pow.shared_work_bits as u64);
        writer.put_u64(self.pow.final_work_bits as u64);
        hash_labeled(
            tag_word(PARAMETERS, 0),
            0,
            None,
            &writer.into_bytes(),
        )
    }
}
