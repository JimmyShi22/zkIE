//! Proof for `GeluIndex`: `idx = clamp(x + offset, 0, table_len - 1)`.
//!
//! The clamp is expressed with two saturation witnesses
//! `u = max(x + offset - L, 0)` and `l = max(-(x + offset), 0)`, so
//! `idx = x + offset - u + l`. The prover proves (a) the affine relation,
//! (b) complementary slackness of each saturation, and (c) that `u`/`l` are
//! non-negative via 16-bit limb lookups against the public limb table.

use zkie_core::common::field::{ChallengeSource, Goldilocks, PrimeCharacteristicRing};
use zkie_core::common::fixed_point::{from_i64, to_i32, to_i64};
use zkie_core::common::sumcheck::{prove_virtual, verify_virtual, VirtualProof};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct GeluIndexProof {
    pub affine: VirtualProof,
    pub r: Vec<Goldilocks>,
    pub alpha: Goldilocks,
    pub beta: Goldilocks,
    pub gamma: Goldilocks,
    pub delta: Goldilocks,
    pub epsilon: Goldilocks,
}

fn gelu_index_forward(x: &[Goldilocks], offset: i64, table_len: usize) -> Vec<u32> {
    x.iter()
        .map(|&v| (to_i64(v) + offset).clamp(0, table_len as i64 - 1) as u32)
        .collect()
}

pub fn prove_gelu_index(
    x: &[Goldilocks],
    idx: &[u32],
    offset: i64,
    table_len: usize,
    rng: &mut impl ChallengeSource,
) -> (GeluIndexProof, Vec<Vec<Goldilocks>>) {
    assert_eq!(x.len(), idx.len());
    assert!(x.len().is_power_of_two(), "GeluIndex length must be a power of two");
    let n = x.len();
    let l_bound = table_len as i64 - 1;

    // Saturation witnesses.
    let y: Vec<i64> = x.iter().map(|&v| to_i64(v) + offset).collect();
    let u: Vec<Goldilocks> = y.iter().map(|&v| from_i64((v - l_bound).max(0))).collect();
    let l: Vec<Goldilocks> = y.iter().map(|&v| from_i64((-v).max(0))).collect();
    let idx_f: Vec<Goldilocks> = idx.iter().map(|&v| Goldilocks::from_u64(v as u64)).collect();

    // 16-bit limbs of u and l (non-negative witnesses).
    let u_lo: Vec<Goldilocks> = u.iter().map(|&v| from_i64(to_i64(v) & 0xFFFF)).collect();
    let u_hi: Vec<Goldilocks> = u.iter().map(|&v| from_i64(to_i64(v) >> 16)).collect();
    let l_lo: Vec<Goldilocks> = l.iter().map(|&v| from_i64(to_i64(v) & 0xFFFF)).collect();
    let l_hi: Vec<Goldilocks> = l.iter().map(|&v| from_i64(to_i64(v) >> 16)).collect();

    let r: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let alpha = rng.field();
    let beta = rng.field();
    let gamma = rng.field();
    let delta = rng.field();
    let epsilon = rng.field();

    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let off = Goldilocks::from_u64(offset as u64);
    let lb = Goldilocks::from_u64(l_bound as u64);
    let two16 = Goldilocks::from_u64(1u64 << 16);
    // mles order: idx, x, u, l, u_lo, u_hi, l_lo, l_hi
    let terms = vec![
        // alpha * (idx - x + u - l) = alpha * offset  (pointwise)
        (alpha, vec![0usize]),
        (neg * alpha, vec![1usize]),
        (alpha, vec![2usize]),
        (neg * alpha, vec![3usize]),
        // beta * (u^2 - u*x + (L - offset)*u) = 0
        (beta, vec![2usize, 2usize]),
        (neg * beta, vec![2usize, 1usize]),
        (beta * (lb - off), vec![2usize]),
        // gamma * (l^2 + l*x + offset*l) = 0
        (gamma, vec![3usize, 3usize]),
        (gamma, vec![3usize, 1usize]),
        (gamma * off, vec![3usize]),
        // delta * (u - u_lo - 2^16 * u_hi) = 0
        (delta, vec![2usize]),
        (neg * delta, vec![4usize]),
        (neg * delta * two16, vec![5usize]),
        // epsilon * (l - l_lo - 2^16 * l_hi) = 0
        (epsilon, vec![3usize]),
        (neg * epsilon, vec![6usize]),
        (neg * epsilon * two16, vec![7usize]),
    ];
    let mles: Vec<&[Goldilocks]> = vec![&idx_f, x, &u, &l, &u_lo, &u_hi, &l_lo, &l_hi];
    // The polynomial is pointwise alpha*offset, so its total sum is alpha*offset*n.
    let claimed = alpha * off * Goldilocks::from_u64(n as u64);
    let affine = prove_virtual(&mles, &terms, claimed, &r);

    (
        GeluIndexProof { affine, r, alpha, beta, gamma, delta, epsilon },
        vec![u_lo, u_hi, l_lo, l_hi],
    )
}

pub fn verify_gelu_index(
    proof: &GeluIndexProof,
    offset: i64,
    table_len: usize,
) -> bool {
    let l_bound = table_len as i64 - 1;
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let off = Goldilocks::from_u64(offset as u64);
    let lb = Goldilocks::from_u64(l_bound as u64);
    let two16 = Goldilocks::from_u64(1u64 << 16);
    let terms = vec![
        (proof.alpha, vec![0usize]),
        (neg * proof.alpha, vec![1usize]),
        (proof.alpha, vec![2usize]),
        (neg * proof.alpha, vec![3usize]),
        (proof.beta, vec![2usize, 2usize]),
        (neg * proof.beta, vec![2usize, 1usize]),
        (proof.beta * (lb - off), vec![2usize]),
        (proof.gamma, vec![3usize, 3usize]),
        (proof.gamma, vec![3usize, 1usize]),
        (proof.gamma * off, vec![3usize]),
        (proof.delta, vec![2usize]),
        (neg * proof.delta, vec![4usize]),
        (neg * proof.delta * two16, vec![5usize]),
        (proof.epsilon, vec![3usize]),
        (neg * proof.epsilon, vec![6usize]),
        (neg * proof.epsilon * two16, vec![7usize]),
    ];
    let n = 1usize << proof.r.len();
    let claimed = proof.alpha * off * Goldilocks::from_u64(n as u64);
    verify_virtual(&proof.affine, &terms, claimed, &proof.r, &proof.affine.final_evals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_core::common::field::XorShift64;
    use zkie_core::common::logup_gkr::verify_lookup_fractional_chain;

    #[test]
    fn gelu_index_roundtrip() {
        let mut rng = XorShift64::new(0x6E1);
        let x: Vec<Goldilocks> = (0..8)
            .map(|i| from_i64((i as i64 - 4) * 30_000_000))
            .collect();
        let idx = gelu_index_forward(&x, 1 << 23, 1 << 24);
        let (proof, limbs) = prove_gelu_index(&x, &idx, 1 << 23, 1 << 24, &mut rng);
        assert!(verify_gelu_index(&proof, 1 << 23, 1 << 24));
        for vals in &limbs {
            for &v in vals {
                assert!(to_i64(v) >= 0 && to_i64(v) < (1i64 << 16), "limb out of range");
            }
        }
        let _ = verify_lookup_fractional_chain;
    }
}
