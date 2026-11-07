//! Query sampling and opening checks for one round's proximity test.
//!
//! Sampling requires an oracle that was committed in an earlier chain round;
//! positions are drawn from the chain after that commitment, never precomputed
//! per worker. The distance/list-decoding parameters are instance parameters
//! and are not invented here.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::chain::Chain;
use crate::pcs::request::merkle::{verify_columns, ColumnOpening, ColumnTree};
use crate::pcs::request::short_code::short_position;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProximityParams {
    pub queries: u32,
    pub log_short: u32,
}

impl ProximityParams {
    pub fn new(queries: u32, log_short: u32) -> Result<Self> {
        if queries == 0 || log_short >= 64 {
            return Err(Error::InvalidConfig);
        }
        Ok(Self { queries, log_short })
    }
}

/// Proof that the oracle was committed before sampling begins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommittedOracle {
    pub root: Digest,
    pub leaf_count: u64,
    pub committed_round: u64,
}

pub fn commit_oracle(tree: &ColumnTree, chain_round: u64) -> CommittedOracle {
    CommittedOracle {
        root: tree.root,
        leaf_count: tree.leaves.len() as u64,
        committed_round: chain_round,
    }
}

/// Sample positions after the commitment. A chain that has not advanced past
/// the committed round cannot sample; the function does not pre-generate a
/// whole run's positions.
pub fn sample_positions(
    chain: &mut Chain,
    oracle: &CommittedOracle,
    params: &ProximityParams,
    log_n: u32,
) -> Result<Vec<u64>> {
    if chain.rounds() <= oracle.committed_round {
        return Err(Error::InvalidSessionState);
    }
    let mut positions = Vec::with_capacity(params.queries as usize);
    for _ in 0..params.queries {
        let position = chain.draw_position(log_n)?;
        if position >= oracle.leaf_count {
            return Err(Error::InvalidIndex);
        }
        positions.push(position);
    }
    Ok(positions)
}

/// Check that every sampled position is opened and authenticates to the
/// committed root with the planned column shape.
pub fn check_openings(
    opening: &ColumnOpening,
    oracle: &CommittedOracle,
    positions: &[u64],
    symbol_count: usize,
    salt_bytes: usize,
) -> Result<()> {
    if opening.root != oracle.root {
        return Err(Error::ArtifactIntegrityMismatch);
    }
    verify_columns(opening, &oracle.root, symbol_count, salt_bytes)?;
    let opened: std::collections::BTreeSet<u64> =
        opening.leaves.iter().map(|leaf| leaf.index).collect();
    for position in positions {
        if !opened.contains(position) {
            return Err(Error::InvalidProof);
        }
    }
    Ok(())
}

/// The short position of each long position, for mask rows read at the same
/// list of columns.
pub fn short_positions(log_n: u32, params: &ProximityParams, positions: &[u64]) -> Result<Vec<u64>> {
    positions
        .iter()
        .map(|position| short_position(log_n, params.log_short, *position))
        .collect()
}
