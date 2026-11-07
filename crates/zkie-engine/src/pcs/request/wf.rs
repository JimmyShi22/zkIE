//! Weight range proofs over committed limb decompositions.
//!
//! A weight is stored as a sign symbol followed by magnitude limbs. The
//! verifier checks every opened leaf, recomposes the magnitude exactly, and
//! rejects a value outside the declared bound. This card opens all leaves of
//! small tables; a streaming/sumcheck WF for large tables is a later card.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::merkle::{
    build_column_tree, verify_columns, ColumnOpening, ColumnTree, ColumnTreeSpec,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WfParams {
    pub bound: u64,
    pub base_bits: u32,
}

impl WfParams {
    pub fn limbs(&self) -> Result<usize> {
        if self.base_bits == 0 || self.base_bits > 32 {
            return Err(Error::InvalidConfig);
        }
        Ok((64 + self.base_bits as usize - 1) / self.base_bits as usize)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WfCommitment {
    pub root: Digest,
    pub table_identity: Digest,
    pub parameters_digest: Digest,
    pub count: u64,
    pub params: WfParams,
}

#[derive(Clone, Debug)]
pub struct WfProof {
    pub commitment: Digest,
    pub params: WfParams,
    pub leaves: ColumnOpening,
}

pub fn encode_weight(value: i64, params: WfParams) -> Result<Vec<u64>> {
    let limbs = params.limbs()?;
    let magnitude = value.unsigned_abs();
    let mut symbols = Vec::with_capacity(limbs + 1);
    symbols.push(if value < 0 { 1 } else { 0 });
    let mut remaining = magnitude;
    let mask = (1u64 << params.base_bits) - 1;
    for _ in 0..limbs {
        symbols.push(remaining & mask);
        remaining >>= params.base_bits;
    }
    if remaining != 0 {
        return Err(Error::BoundViolation);
    }
    Ok(symbols)
}

pub fn decode_weight(symbols: &[u64], params: WfParams) -> Result<i64> {
    let limbs = params.limbs()?;
    if symbols.len() != limbs + 1 {
        return Err(Error::ShapeMismatch);
    }
    if symbols[0] > 1 {
        return Err(Error::InvalidEncoding);
    }
    let limit = 1u64 << params.base_bits;
    let mut magnitude = 0u128;
    for limb in symbols[1..].iter().rev() {
        if *limb >= limit {
            return Err(Error::InvalidEncoding);
        }
        magnitude = (magnitude << params.base_bits) | *limb as u128;
    }
    let signed = if symbols[0] == 1 {
        -(i128::from(u64::try_from(magnitude).map_err(|_| Error::BoundViolation)?))
    } else {
        i128::from(u64::try_from(magnitude).map_err(|_| Error::BoundViolation)?)
    };
    i64::try_from(signed).map_err(|_| Error::BoundViolation)
}

/// Commit all weights with per-weight salts.
pub fn commit_weights(
    params: WfParams,
    table_identity: Digest,
    parameters_digest: Digest,
    weights: &[i64],
    salts: &[Vec<u8>],
) -> Result<(WfCommitment, ColumnTree)> {
    if weights.is_empty() || weights.len() != salts.len() {
        return Err(Error::InvalidConfig);
    }
    let limbs = params.limbs()?;
    let mut columns = Vec::with_capacity(weights.len());
    for (value, salt) in weights.iter().zip(salts) {
        columns.push((encode_weight(*value, params)?, salt.clone()));
    }
    let spec = ColumnTreeSpec::new(
        table_identity,
        parameters_digest,
        7,
        limbs + 1,
        salts[0].len(),
    )?;
    let tree = build_column_tree(&spec, &columns)?;
    Ok((
        WfCommitment {
            root: tree.root,
            table_identity,
            parameters_digest,
            count: weights.len() as u64,
            params,
        },
        tree,
    ))
}

/// Open every weight. Completeness is required by the verifier in this card.
pub fn prove_wf(tree: &ColumnTree, commitment: &WfCommitment) -> Result<WfProof> {
    if tree.root != commitment.root || tree.leaves.len() as u64 != commitment.count {
        return Err(Error::StatementMismatch);
    }
    let indices: Vec<u64> = (0..commitment.count).collect();
    let leaves = crate::pcs::request::merkle::open_columns(tree, &indices)?;
    Ok(WfProof {
        commitment: commitment.root,
        params: commitment.params,
        leaves,
    })
}

pub fn verify_wf(commitment: &WfCommitment, proof: &WfProof) -> Result<()> {
    if proof.commitment != commitment.root || proof.params != commitment.params {
        return Err(Error::StatementMismatch);
    }
    if proof.leaves.leaves.len() as u64 != commitment.count {
        return Err(Error::IncompleteArtifact);
    }
    verify_columns(
        &proof.leaves,
        &commitment.root,
        commitment.params.limbs()? + 1,
        proof.leaves.spec.salt_bytes,
    )?;
    let mut seen = std::collections::BTreeSet::new();
    for leaf in &proof.leaves.leaves {
        if !seen.insert(leaf.index) {
            return Err(Error::InvalidIndex);
        }
        let value = decode_weight(&leaf.symbols, commitment.params)?;
        if value.unsigned_abs() > commitment.params.bound {
            return Err(Error::BoundViolation);
        }
    }
    if seen.len() as u64 != commitment.count {
        return Err(Error::IncompleteArtifact);
    }
    Ok(())
}
