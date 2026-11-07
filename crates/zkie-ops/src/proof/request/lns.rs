//! Range checks (LNS): every limb is a lookup into the limb table and the
//! signed value is reconstructed exactly.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::chain::Chain;

use crate::proof::binding::{
    commit_rows_with_openings, OracleId, OracleOpening, OracleRegistry,
};
use crate::proof::request::lkp::{prove_lookup, verify_lookup, LookupProof, LookupTable};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangeParams {
    /// Bits in one limb table entry.
    pub base_bits: u32,
    /// |value| must be below 2^max_value_bits.
    pub max_value_bits: u32,
}

impl RangeParams {
    pub fn new(base_bits: u32, max_value_bits: u32) -> Result<Self> {
        if base_bits == 0 || base_bits > 32 || max_value_bits == 0 || max_value_bits > 64 {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            base_bits,
            max_value_bits,
        })
    }

    pub fn limb_count(&self) -> usize {
        (64 + self.base_bits as usize - 1) / self.base_bits as usize
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeProof {
    pub values: Vec<i64>,
    pub limbs: Vec<u64>,
    pub oracle: OracleId,
    pub openings: Vec<OracleOpening>,
    pub lookup: LookupProof,
}

pub fn range_table(params: RangeParams) -> Result<LookupTable> {
    let entries = 1u64 << params.base_bits;
    let rows: Vec<Vec<u64>> = (0..entries).map(|value| vec![value]).collect();
    LookupTable::new(rows)
}

fn zigzag(value: i64) -> u64 {
    let magnitude = (value as i128).unsigned_abs() as u64;
    if value >= 0 {
        magnitude.wrapping_mul(2)
    } else {
        magnitude.wrapping_mul(2).wrapping_sub(1)
    }
}

fn unzigzag(value: u64) -> Result<i64> {
    let magnitude = value >> 1;
    let signed = if value & 1 == 0 {
        magnitude as i128
    } else {
        -(magnitude as i128) - 1
    };
    i64::try_from(signed).map_err(|_| Error::BoundViolation)
}

fn decompose(value: u64, base_bits: u32, count: usize) -> Vec<u64> {
    let mask = (1u64 << base_bits) - 1;
    let mut remaining = value;
    let mut limbs = Vec::with_capacity(count);
    for _ in 0..count {
        limbs.push(remaining & mask);
        remaining >>= base_bits;
    }
    limbs
}

fn recompose(limbs: &[u64], base_bits: u32) -> Result<u64> {
    let limit = 1u64 << base_bits;
    let mut value = 0u128;
    for limb in limbs.iter().rev() {
        if *limb >= limit {
            return Err(Error::InvalidEncoding);
        }
        value = (value << base_bits) | *limb as u128;
    }
    u64::try_from(value).map_err(|_| Error::BoundViolation)
}

pub fn limbs_for_values(params: RangeParams, values: &[i64]) -> Result<Vec<u64>> {
    if values.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let count = params.limb_count();
    let mut limbs = Vec::with_capacity(values.len() * count);
    for value in values {
        limbs.extend(decompose(zigzag(*value), params.base_bits, count));
    }
    Ok(limbs)
}

/// Prove that every value fits the declared signed range.
pub fn prove_range(
    chain: &mut Chain,
    params: RangeParams,
    values: &[i64],
    registry: &mut OracleRegistry,
    label: &Digest,
    table: &LookupTable,
) -> Result<RangeProof> {
    if values.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let limbs = limbs_for_values(params, values)?;
    let rows: Vec<Vec<u64>> = limbs.iter().map(|limb| vec![*limb]).collect();
    let real = rows.len();
    let mut padded = rows;
    padded.resize(real.next_power_of_two(), vec![0]);
    let (oracle, mut openings) = commit_rows_with_openings(registry, label, &padded)?;
    openings.truncate(real);
    let lookup = prove_lookup(chain, table, &openings, registry)?;
    Ok(RangeProof {
        values: values.to_vec(),
        limbs,
        oracle,
        openings,
        lookup,
    })
}

/// Verify the range proof. The bound is public and is not read from the proof.
pub fn verify_range(
    chain: &mut Chain,
    params: RangeParams,
    proof: &RangeProof,
    table: &LookupTable,
    registry: &OracleRegistry,
) -> Result<()> {
    let count = params.limb_count();
    if proof.values.is_empty()
        || proof.limbs.len() != proof.values.len() * count
        || proof.openings.len() != proof.limbs.len()
        || proof.oracle != proof.openings.first().map(|opening| opening.oracle).ok_or(Error::InvalidProof)?
        || proof.openings.iter().any(|opening| opening.oracle != proof.oracle)
    {
        return Err(Error::ShapeMismatch);
    }
    let bound = 1i128 << params.max_value_bits;
    for (index, value) in proof.values.iter().enumerate() {
        if (*value as i128).unsigned_abs() >= bound as u128 {
            return Err(Error::BoundViolation);
        }
        let expected = decompose(zigzag(*value), params.base_bits, count);
        let start = index * count;
        if proof.limbs[start..start + count] != expected {
            return Err(Error::InvalidProof);
        }
        let encoded = recompose(&proof.limbs[start..start + count], params.base_bits)?;
        if unzigzag(encoded)? != *value {
            return Err(Error::InvalidProof);
        }
    }
    for (index, opening) in proof.openings.iter().enumerate() {
        if opening.index != index as u64 || opening.values.len() != 1 {
            return Err(Error::InvalidProof);
        }
    }
    verify_lookup(chain, table, &proof.openings, registry, &proof.lookup)
}
