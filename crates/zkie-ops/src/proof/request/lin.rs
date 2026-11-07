//! Linear claims: coefficients are recomputed from public inputs and the
//! transcript, and the claimed value is checked against authenticated rows.

use zkie_core::claim::{CoefficientSpec, LinearClaim, RelationId};
use zkie_core::error::{Error, Result};
use zkie_core::oracle::OracleId;
use zkie_engine::pcs::request::chain::{Chain, KIND_ALGEBRAIC};
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::primitives::codec::CanonicalWriter;
use zkie_engine::primitives::transcript::{hash_labeled, tag_word, STATEMENT};

use crate::proof::binding::{verify_opening, OracleOpening, OracleRegistry};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearProof {
    pub openings: Vec<OracleOpening>,
}

/// Resolve the claim's coefficients. `Public` coefficients are supplied by the
/// caller and are not read from the proof; challenge-derived coefficients are
/// drawn from the chain after binding label, oracle and source.
pub fn coefficients(
    chain: &mut Chain,
    spec: &CoefficientSpec,
    len: usize,
    oracle: OracleId,
    source: RelationId,
    public: &[Fp3],
) -> Result<Vec<Fp3>> {
    match spec {
        CoefficientSpec::Public => {
            if public.len() != len {
                return Err(Error::ShapeMismatch);
            }
            Ok(public.to_vec())
        }
        CoefficientSpec::ChallengeDerived { label } => {
            let mut writer = CanonicalWriter::new();
            writer.put_bytes(label);
            writer.put_u64(oracle.0);
            writer.put_u64(source.0);
            writer.put_u64(len as u64);
            let label_digest =
                hash_labeled(tag_word(STATEMENT, 9), 0, None, &writer.into_bytes())?;
            let mut message = chain.message();
            for word in label_digest.0.chunks_exact(8) {
                message.push_u64(u64::from_le_bytes(
                    word.try_into().map_err(|_| Error::InvalidEncoding)?,
                ));
            }
            let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
            if !chain.step(&prefix, 0)? {
                return Err(Error::InvalidProof);
            }
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                values.push(chain.draw_e()?);
            }
            Ok(values)
        }
    }
}

fn linear_value(coefficients: &[Fp3], openings: &[OracleOpening]) -> Result<Fp3> {
    if coefficients.len() != openings.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut sum = Fp3::ZERO;
    for (coefficient, opening) in coefficients.iter().zip(openings) {
        if opening.values.len() != 1 {
            return Err(Error::ShapeMismatch);
        }
        sum = sum.add(&coefficient.mul(&Fp3::from_u64(opening.values[0])));
    }
    Ok(sum)
}

pub fn prove_linear(
    chain: &mut Chain,
    claim: &LinearClaim<Fp3>,
    openings: &[OracleOpening],
    registry: &OracleRegistry,
    public_coefficients: &[Fp3],
) -> Result<LinearProof> {
    let resolved = coefficients(
        chain,
        &claim.coefficients,
        openings.len(),
        claim.oracle,
        claim.source,
        public_coefficients,
    )?;
    if openings.is_empty() {
        return Err(Error::InvalidConfig);
    }
    for (index, opening) in openings.iter().enumerate() {
        if opening.oracle != claim.oracle || opening.index != index as u64 {
            return Err(Error::StatementMismatch);
        }
        let root = registry.get(opening.oracle).ok_or(Error::ProverStateMissing)?;
        verify_opening(registry, opening, root.count)?;
    }
    if linear_value(&resolved, openings)? != claim.value {
        return Err(Error::StatementMismatch);
    }
    Ok(LinearProof {
        openings: openings.to_vec(),
    })
}

pub fn verify_linear(
    chain: &mut Chain,
    claim: &LinearClaim<Fp3>,
    proof: &LinearProof,
    registry: &OracleRegistry,
    public_coefficients: &[Fp3],
    expected_source: RelationId,
) -> Result<()> {
    if claim.source != expected_source {
        return Err(Error::StatementMismatch);
    }
    if proof.openings.is_empty() {
        return Err(Error::InvalidProof);
    }
    let resolved = coefficients(
        chain,
        &claim.coefficients,
        proof.openings.len(),
        claim.oracle,
        claim.source,
        public_coefficients,
    )?;
    for (index, opening) in proof.openings.iter().enumerate() {
        if opening.oracle != claim.oracle || opening.index != index as u64 {
            return Err(Error::InvalidProof);
        }
        let root = registry.get(opening.oracle).ok_or(Error::ProverStateMissing)?;
        verify_opening(registry, opening, root.count)?;
    }
    if linear_value(&resolved, &proof.openings)? != claim.value {
        return Err(Error::InvalidProof);
    }
    Ok(())
}
