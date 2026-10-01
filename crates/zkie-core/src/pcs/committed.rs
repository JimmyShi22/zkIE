//! WHIR commitment handles for the shard-DAG composer.
//!
//! A tensor is committed once with `commit`, then opened/verified against the
//! commitment at the prescribed points. `BatchCtx` is the batch form: one
//! commitment plus the shared prover data and opening protocol for a group of
//! same-size tables.

use crate::common::field::Goldilocks;
use crate::pcs::whir::{Commitment, OpeningProtocol, ProverData, Whir};

/// A committed tensor (commitment + prover data + opening protocol).
pub struct Committed {
    pub commitment: Commitment,
    pub prover_data: ProverData,
    pub protocol: OpeningProtocol,
}

pub fn commit(whir: &Whir, values: &[Goldilocks]) -> Committed {
    let (commitment, prover_data, protocol) = whir.commit(values);
    Committed {
        commitment,
        prover_data,
        protocol,
    }
}

/// A committed batch of same-size MLEs: one commitment plus the shared prover
/// data, opening protocol, and the batch-sized `Whir` used to open any table.
pub struct BatchCtx {
    pub commitment: Commitment,
    pub prover_data: ProverData,
    pub protocol: OpeningProtocol,
    pub whir: Whir,
    pub num_tables: usize,
}
