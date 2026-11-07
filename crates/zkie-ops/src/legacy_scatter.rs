//! Expert-major scatter: reorder tokens into dense per-expert segments and
//! prove the reordering is a correct permutation (LogUp multiset equality).

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64, ChallengeSource};
use zkie_core::common::logup_gkr::{prove_permutation, verify_permutation, PermutationProof};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ScatterProof {
    pub perm: PermutationProof,
}

/// `(token_id, expert_of)`: for each expert in order, the tokens routed to it.
pub fn scatter_forward(sel: &[Goldilocks], m: usize, n: usize) -> (Vec<u32>, Vec<u32>) {
    let mut token_id = Vec::new();
    let mut expert_of = Vec::new();
    for e in 0..n {
        for t in 0..m {
            if sel[t * n + e] == Goldilocks::ONE {
                token_id.push(t as u32);
                expert_of.push(e as u32);
            }
        }
    }
    (token_id, expert_of)
}

fn source_encodings(sel: &[Goldilocks], m: usize, n: usize) -> Vec<Goldilocks> {
    let mut a = Vec::new();
    for t in 0..m {
        for e in 0..n {
            if sel[t * n + e] == Goldilocks::ONE {
                a.push(Goldilocks::from_u64((t * n + e) as u64));
            }
        }
    }
    a
}

pub fn prove_scatter(
    sel: &[Goldilocks],
    token_id: &[u32],
    expert_of: &[u32],
    m: usize,
    n: usize,
    rng: &mut impl ChallengeSource,
) -> ScatterProof {
    let a = source_encodings(sel, m, n);
    let b: Vec<Goldilocks> = token_id
        .iter()
        .zip(expert_of.iter())
        .map(|(&t, &e)| Goldilocks::from_u64((t as usize * n + e as usize) as u64))
        .collect();
    let perm = prove_permutation(&a, &b, rng);
    ScatterProof { perm }
}

pub fn verify_scatter(proof: &ScatterProof, _m: usize, _n: usize) -> bool {
    verify_permutation(&proof.perm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scatter_roundtrip() {
        let mut rng = XorShift64::new(0x4242);
        let (m, n, k) = (4usize, 4usize, 2usize);
        let mut sel = vec![Goldilocks::ZERO; m * n];
        for t in 0..m {
            sel[t * n + (t % n)] = Goldilocks::ONE;
            sel[t * n + ((t + 1) % n)] = Goldilocks::ONE;
        }
        let (token_id, expert_of) = scatter_forward(&sel, m, n);
        assert_eq!(token_id.len(), m * k);
        let proof = prove_scatter(&sel, &token_id, &expert_of, m, n, &mut rng);
        assert!(verify_scatter(&proof, m, n));
    }
}
