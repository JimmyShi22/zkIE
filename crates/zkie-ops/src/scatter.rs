//! Expert-major scatter: reorder tokens into dense per-expert segments and
//! prove the reordering is a correct permutation (LogUp multiset equality).

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64, ChallengeSource};
use zkie_core::common::logup_gkr::{prove_permutation, verify_permutation, PermutationProof};
use zkie_core::common::transcript::Transcript;

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
    transcript: &mut Transcript,
) -> ScatterProof {
    prove_scatter_with_mode(sel, token_id, expert_of, m, n, transcript, crate::compose::FracMode::Plain)
}

pub fn prove_scatter_with_mode(
    sel: &[Goldilocks],
    token_id: &[u32],
    expert_of: &[u32],
    m: usize,
    n: usize,
    transcript: &mut Transcript,
    mode: crate::compose::FracMode,
) -> ScatterProof {
    let a = source_encodings(sel, m, n);
    let b: Vec<Goldilocks> = token_id
        .iter()
        .zip(expert_of.iter())
        .map(|(&t, &e)| Goldilocks::from_u64((t as usize * n + e as usize) as u64))
        .collect();
    let perm = match mode {
        // Committed: the committed perm section re-proves the fracs on its
        // own transcript (binding the committed column roots); here they run
        // on a scratch transcript so the shard transcript stays frac-free.
        crate::compose::FracMode::Committed => {
            let mut scratch = Transcript::new(b"zkie-scratch-perm");
            prove_permutation(&a, &b, &mut scratch)
        }
        crate::compose::FracMode::Plain => prove_permutation(&a, &b, transcript),
    };
    ScatterProof { perm }
}

pub fn verify_scatter(
    proof: &ScatterProof,
    _m: usize,
    _n: usize,
    transcript: &mut Transcript,
) -> bool {
    verify_scatter_with_mode(proof, _m, _n, transcript, crate::compose::FracMode::Plain)
}

pub fn verify_scatter_with_mode(
    proof: &ScatterProof,
    _m: usize,
    _n: usize,
    transcript: &mut Transcript,
    mode: crate::compose::FracMode,
) -> bool {
    match mode {
        crate::compose::FracMode::Plain => {
            // S3: the permutation depth comes from the (padded, proof-carried)
            // a column length (build_perm_num_den pads to at least 2^5).
            let t = proof.perm.a.len().next_power_of_two().max(32).trailing_zeros() as usize;
            verify_permutation(&proof.perm, t, transcript)
        }
        // Committed: the fracs are verified by the committed perm section on
        // the committed transcript; nothing is replayed on the shard
        // transcript.
        crate::compose::FracMode::Committed => true,
    }
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
        let mut tp = Transcript::new(b"zkie-scatter-test");
        tp.absorb_u64(0x4242);
        let proof = prove_scatter(&sel, &token_id, &expert_of, m, n, &mut tp);
        let mut tv = Transcript::new(b"zkie-scatter-test");
        tv.absorb_u64(0x4242);
        assert!(verify_scatter(&proof, m, n, &mut tv));
    }
}
