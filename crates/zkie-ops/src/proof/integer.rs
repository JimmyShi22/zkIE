//! Exact integer relations: quotient/remainder, range and limb/carry checks,
//! with authenticated openings.

use zkie_core::error::{Error, Result};
use zkie_engine::compute::cpu::{limb_decompose, limb_recompose};

use crate::proof::binding::{verify_opening, OracleOpening, OracleRegistry};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntegerVectors {
    pub numerators: Vec<i64>,
    pub denominators: Vec<i64>,
    pub quotients: Vec<i64>,
    pub remainders: Vec<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntegerProof {
    pub numerators: Vec<OracleOpening>,
    pub denominators: Vec<OracleOpening>,
    pub quotients: Vec<OracleOpening>,
    pub remainders: Vec<OracleOpening>,
    pub limbs: Vec<u64>,
    pub carries: Vec<i64>,
    pub base_bits: u32,
}

fn check_opening(
    registry: &OracleRegistry,
    opening: &OracleOpening,
    index: usize,
) -> Result<i64> {
    if opening.index != index as u64 || opening.values.len() != 1 {
        return Err(Error::InvalidProof);
    }
    let root = registry.get(opening.oracle).ok_or(Error::ProverStateMissing)?;
    verify_opening(registry, opening, root.count)?;
    Ok(opening.values[0] as i64)
}

pub fn prove_integer_relations(
    vectors: &IntegerVectors,
    openings: &IntegerProof,
    registry: &OracleRegistry,
) -> Result<IntegerProof> {
    if vectors.numerators.len() != vectors.denominators.len()
        || vectors.numerators.len() != vectors.quotients.len()
        || vectors.numerators.len() != vectors.remainders.len()
        || vectors.numerators.is_empty()
    {
        return Err(Error::ShapeMismatch);
    }
    if openings.numerators.len() != vectors.numerators.len()
        || openings.denominators.len() != vectors.denominators.len()
        || openings.quotients.len() != vectors.quotients.len()
        || openings.remainders.len() != vectors.remainders.len()
    {
        return Err(Error::ShapeMismatch);
    }
    let mut limbs = Vec::new();
    let mut carries = Vec::new();
    for index in 0..vectors.numerators.len() {
        let numerator = check_opening(registry, &openings.numerators[index], index)?;
        let denominator = check_opening(registry, &openings.denominators[index], index)?;
        let quotient = check_opening(registry, &openings.quotients[index], index)?;
        let remainder = check_opening(registry, &openings.remainders[index], index)?;
        if numerator != vectors.numerators[index]
            || denominator != vectors.denominators[index]
            || quotient != vectors.quotients[index]
            || remainder != vectors.remainders[index]
        {
            return Err(Error::StatementMismatch);
        }
        if denominator <= 0 {
            return Err(if denominator == 0 {
                Error::DivisionByZero
            } else {
                Error::InvalidConfig
            });
        }
        let product = (quotient as i128)
            .checked_mul(denominator as i128)
            .and_then(|value| value.checked_add(remainder as i128))
            .ok_or(Error::BoundViolation)?;
        if product != numerator as i128
            || remainder < 0
            || remainder >= denominator
        {
            return Err(Error::InvalidProof);
        }
        // Limb/carry: the magnitude decomposes and recomposes exactly.
        let magnitude = (numerator as i128).unsigned_abs();
        let magnitude = u64::try_from(magnitude).map_err(|_| Error::BoundViolation)?;
        let decomposition = limb_decompose(magnitude as i128, openings.base_bits)?;
        if limb_recompose(&decomposition, openings.base_bits)? != magnitude as i128 {
            return Err(Error::InvalidProof);
        }
        limbs.extend(decomposition);
        carries.push(0);
    }
    Ok(IntegerProof {
        numerators: openings.numerators.clone(),
        denominators: openings.denominators.clone(),
        quotients: openings.quotients.clone(),
        remainders: openings.remainders.clone(),
        limbs,
        carries,
        base_bits: openings.base_bits,
    })
}

pub fn verify_integer_relations(proof: &IntegerProof, registry: &OracleRegistry) -> Result<()> {
    if proof.numerators.is_empty() || proof.carries.len() != proof.numerators.len() {
        return Err(Error::ShapeMismatch);
    }
    for index in 0..proof.numerators.len() {
        let numerator = check_opening(registry, &proof.numerators[index], index)?;
        let denominator = check_opening(registry, &proof.denominators[index], index)?;
        let quotient = check_opening(registry, &proof.quotients[index], index)?;
        let remainder = check_opening(registry, &proof.remainders[index], index)?;
        if denominator <= 0 {
            return Err(Error::DivisionByZero);
        }
        let product = (quotient as i128)
            .checked_mul(denominator as i128)
            .and_then(|value| value.checked_add(remainder as i128))
            .ok_or(Error::BoundViolation)?;
        if product != numerator as i128 || remainder < 0 || remainder >= denominator {
            return Err(Error::InvalidProof);
        }
        if proof.carries[index] != 0 {
            return Err(Error::InvalidProof);
        }
    }
    let mut offset = 0usize;
    for (index, opening) in proof.numerators.iter().enumerate() {
        let magnitude = (check_opening(registry, opening, index)? as i128).unsigned_abs();
        let magnitude = u64::try_from(magnitude).map_err(|_| Error::BoundViolation)?;
        let decomposition = limb_decompose(magnitude as i128, proof.base_bits)?;
        if proof.limbs.get(offset..offset + decomposition.len()) != Some(decomposition.as_slice()) {
            return Err(Error::InvalidProof);
        }
        offset += decomposition.len();
    }
    if offset != proof.limbs.len() {
        return Err(Error::InvalidProof);
    }
    Ok(())
}
