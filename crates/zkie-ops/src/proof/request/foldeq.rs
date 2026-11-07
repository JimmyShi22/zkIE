//! Fold connection equations: every folded claim is tied to the two claims it
//! replaces at the round's challenge.

use std::collections::BTreeSet;

use zkie_core::claim::RelationId;
use zkie_core::error::{Error, Result};
use zkie_core::oracle::OracleId;
use zkie_engine::pcs::request::field::Fp3;

use crate::proof::binding::OracleRegistry;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldConnection {
    pub relation: RelationId,
    pub oracle: OracleId,
    pub low: Fp3,
    pub high: Fp3,
    pub challenge: Fp3,
    pub folded: Fp3,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldEqProof {
    pub connections: Vec<FoldConnection>,
}

pub fn fold_value(low: Fp3, high: Fp3, challenge: Fp3) -> Fp3 {
    low.add(&challenge.mul(&high.sub(&low)))
}

/// The prover computes each folded value and refuses to emit an inconsistent
/// connection.
pub fn prove_fold_equations(
    connections: Vec<FoldConnection>,
) -> Result<FoldEqProof> {
    for connection in &connections {
        if connection.folded != fold_value(connection.low, connection.high, connection.challenge) {
            return Err(Error::StatementMismatch);
        }
    }
    Ok(FoldEqProof { connections })
}

/// The verifier requires exactly the expected relations, each once and in the
/// agreed order, and checks every fold equation.
pub fn verify_fold_equations(
    proof: &FoldEqProof,
    required_relations: &[RelationId],
    registry: &OracleRegistry,
) -> Result<()> {
    if proof.connections.len() != required_relations.len() {
        return Err(Error::UnclosedClaims);
    }
    let mut seen = BTreeSet::new();
    for (connection, required) in proof.connections.iter().zip(required_relations) {
        if connection.relation != *required {
            return Err(Error::StatementMismatch);
        }
        if !seen.insert(connection.relation) {
            return Err(Error::StatementMismatch);
        }
        if registry.get(connection.oracle).is_none() {
            return Err(Error::ProverStateMissing);
        }
        if connection.folded != fold_value(connection.low, connection.high, connection.challenge)
        {
            return Err(Error::InvalidProof);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Connections derived from committed rows.

use crate::proof::binding::{verify_opening, OracleOpening};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommittedRow {
    pub oracle: OracleId,
    pub coefficients: Vec<Fp3>,
    pub openings: Vec<OracleOpening>,
}

pub fn committed_row_value(registry: &OracleRegistry, row: &CommittedRow) -> Result<Fp3> {
    if row.coefficients.len() != row.openings.len() || row.openings.is_empty() {
        return Err(Error::ShapeMismatch);
    }
    let root = registry.get(row.oracle).ok_or(Error::ProverStateMissing)?;
    let mut sum = Fp3::ZERO;
    for (index, (coefficient, opening)) in row.coefficients.iter().zip(&row.openings).enumerate() {
        if opening.oracle != row.oracle
            || opening.index != index as u64
            || opening.values.len() != 1
        {
            return Err(Error::StatementMismatch);
        }
        verify_opening(registry, opening, root.count)?;
        sum = sum.add(&coefficient.mul(&Fp3::from_u64(opening.values[0])));
    }
    Ok(sum)
}

/// Derive a fold connection from the committed rows and the round challenge;
/// the folded value is computed here, never supplied by a caller.
pub fn derive_connection(
    relation: RelationId,
    low: &CommittedRow,
    high: &CommittedRow,
    challenge: Fp3,
    registry: &OracleRegistry,
) -> Result<FoldConnection> {
    let low_value = committed_row_value(registry, low)?;
    let high_value = committed_row_value(registry, high)?;
    Ok(FoldConnection {
        relation,
        oracle: low.oracle,
        low: low_value,
        high: high_value,
        challenge,
        folded: fold_value(low_value, high_value, challenge),
    })
}

pub fn verify_derived_connection(
    connection: &FoldConnection,
    low: &CommittedRow,
    high: &CommittedRow,
    registry: &OracleRegistry,
) -> Result<()> {
    let derived = derive_connection(
        connection.relation,
        low,
        high,
        connection.challenge,
        registry,
    )?;
    if derived.low != connection.low
        || derived.high != connection.high
        || derived.folded != connection.folded
        || derived.oracle != connection.oracle
    {
        return Err(Error::InvalidProof);
    }
    Ok(())
}
