//! WHIR commitment handles for the shard-DAG composer.
//!
//! A tensor is committed once with `commit`, then opened/verified against the
//! commitment at the prescribed points. `BatchCtx` is the batch form: one
//! commitment plus the shared prover data and opening protocol for a group of
//! same-size tables.

use crate::common::field::Goldilocks;
use crate::pcs::whir::{Commitment, OpeningProtocol, ProverData, Whir};

/// A committed tensor (commitment + prover data + opening protocol).
///
/// Prover-side handle: `prover_data` is witness data, and the only functions
/// that need it are the `open_*` calls on the prover side.
pub struct Committed {
    pub commitment: Commitment,
    pub prover_data: ProverData,
    pub protocol: OpeningProtocol,
}

/// Verifier-only view of a committed tensor: the commitment and the public
/// opening protocol. Deliberately holds no `prover_data`, so a verifier can
/// only check a transported opening proof — it can never regenerate one.
#[derive(Clone)]
pub struct CommittedPublic {
    pub commitment: Commitment,
    pub protocol: OpeningProtocol,
}

impl Committed {
    /// Public (verifier-side) view: commitment + protocol, no prover data.
    pub fn to_public(&self) -> CommittedPublic {
        CommittedPublic {
            commitment: self.commitment.clone(),
            protocol: self.protocol.clone(),
        }
    }
}

pub fn commit(whir: &Whir, values: &[Goldilocks]) -> Committed {
    let (commitment, prover_data, protocol) = whir.commit(values);
    Committed {
        commitment,
        prover_data,
        protocol,
    }
}

/// Commit a tensor with a prescribed multi-point opening protocol (one opening
/// per point), so a single WHIR opening proof can open it at many points.
pub fn commit_with_points(whir: &Whir, values: &[Goldilocks], num_points: usize) -> Committed {
    let (commitment, prover_data, protocol) = whir.commit_with_points(values, num_points);
    Committed {
        commitment,
        prover_data,
        protocol,
    }
}

/// A committed batch of same-size MLEs: one commitment plus the shared prover
/// data, opening protocol, and the batch-sized `Whir` used to open any table.
///
/// Prover-side handle; see [`BatchPublic`] for the verifier-side view.
pub struct BatchCtx {
    pub commitment: Commitment,
    pub prover_data: ProverData,
    pub protocol: OpeningProtocol,
    pub whir: Whir,
    pub num_tables: usize,
}

/// Verifier-only view of a committed batch: commitment + protocol + a fresh
/// `Whir` configured with the *same* security parameters (construction is
/// deterministic), no `prover_data`. Verify functions take this and a
/// transported opening proof; they never open.
pub struct BatchPublic {
    pub commitment: Commitment,
    pub protocol: OpeningProtocol,
    pub whir: Whir,
    pub num_tables: usize,
}

impl BatchCtx {
    /// Public (verifier-side) view. Builds a fresh batch `Whir` with identical
    /// parameters; note this allocates that instance's DFT tables.
    pub fn to_public(&self) -> BatchPublic {
        let (security_level, pow_bits) = self.whir.security_params();
        BatchPublic {
            commitment: self.commitment.clone(),
            protocol: self.protocol.clone(),
            whir: Whir::with_params(self.whir.num_variables(), security_level, pow_bits),
            num_tables: self.num_tables,
        }
    }
}
