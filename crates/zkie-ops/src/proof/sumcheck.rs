//! Multilinear sumcheck with an explicit error budget.
//!
//! The prover sends one linear round polynomial per variable; the verifier
//! checks q(0) + q(1) against its running claim and finishes on an opened
//! value. Error accounting exposes the number of rounds and the field size.

use zkie_core::error::{Error, Result};
use zkie_engine::pcs::request::field::Fp3;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckProof {
    pub rounds: Vec<[Fp3; 3]>,
    pub final_value: Fp3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckAccounting {
    pub rounds: u32,
    pub field_bits: u32,
    /// floor of log2(rounds / |E|), as a negative bit count.
    pub error_log2_floor: i32,
}

pub fn accounting(rounds: u32) -> SumcheckAccounting {
    let field_bits = 3 * 64u32;
    let round_bits = 32 - rounds.max(1).leading_zeros();
    SumcheckAccounting {
        rounds,
        field_bits,
        error_log2_floor: (round_bits as i32) - (field_bits as i32),
    }
}

fn fold(values: &mut Vec<Fp3>, challenge: Fp3) -> Result<()> {
    if values.len() % 2 != 0 || values.is_empty() {
        return Err(Error::ShapeMismatch);
    }
    let half = values.len() / 2;
    for index in 0..half {
        let low = values[2 * index];
        let high = values[2 * index + 1];
        values[index] = low.add(&challenge.mul(&high.sub(&low)));
    }
    values.truncate(half);
    Ok(())
}

fn round_polynomial(values: &[Fp3]) -> Result<[Fp3; 3]> {
    if values.len() % 2 != 0 || values.is_empty() {
        return Err(Error::ShapeMismatch);
    }
    let mut constant = Fp3::ZERO;
    let mut linear = Fp3::ZERO;
    for pair in values.chunks_exact(2) {
        constant = constant.add(&pair[0]);
        linear = linear.add(&pair[1].sub(&pair[0]));
    }
    Ok([constant, linear, Fp3::ZERO])
}

fn evaluate(polynomial: &[Fp3; 3], point: Fp3) -> Fp3 {
    polynomial[0]
        .add(&polynomial[1].mul(&point))
        .add(&polynomial[2].mul(&point).mul(&point))
}

/// Prove the sum of `values` over the boolean hypercube equals `claimed`.
pub fn prove_sumcheck(
    values: &[Fp3],
    challenges: &[Fp3],
    claimed: Fp3,
) -> Result<SumcheckProof> {
    let _profile = zkie_core::profile::Scope::kernel(zkie_core::profile::Kernel::Sumcheck);
    if values.is_empty() || !values.len().is_power_of_two() {
        return Err(Error::InvalidConfig);
    }
    if values.len() != 1usize.checked_shl(challenges.len() as u32).unwrap_or(0) {
        return Err(Error::ShapeMismatch);
    }
    let mut current = values.to_vec();
    let mut running = claimed;
    let mut rounds = Vec::with_capacity(challenges.len());
    for challenge in challenges {
        let q = round_polynomial(&current)?;
        if q[0].add(&q[0]).add(&q[1]) != running {
            return Err(Error::InvalidProof);
        }
        running = evaluate(&q, *challenge);
        fold(&mut current, *challenge)?;
        rounds.push(q);
    }
    Ok(SumcheckProof {
        rounds,
        final_value: current[0],
    })
}

pub fn verify_sumcheck(
    proof: &SumcheckProof,
    challenges: &[Fp3],
    claimed: Fp3,
    opened_value: Fp3,
) -> Result<()> {
    let _profile = zkie_core::profile::Scope::kernel(zkie_core::profile::Kernel::Sumcheck);
    if proof.rounds.len() != challenges.len() || challenges.is_empty() {
        return Err(Error::ShapeMismatch);
    }
    let mut running = claimed;
    for (q, challenge) in proof.rounds.iter().zip(challenges) {
        if q[0].add(&q[0]).add(&q[1]) != running {
            return Err(Error::InvalidProof);
        }
        running = evaluate(q, *challenge);
    }
    if running != opened_value || proof.final_value != opened_value {
        return Err(Error::InvalidProof);
    }
    Ok(())
}
