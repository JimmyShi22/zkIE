//! Scalable softmax core: exp lookup + row-sum + rescale, all O(m*n).
//!
//! `out = e / sum` (field inverse), with `sum = row_sum(e)` proven as a
//! *separate* tensor (not a virtual MLE), so the rescale `out * sum_broadcast = e`
//! is an elementwise product over (i,j) — O(m*n), not the flattened O(m*n*n)
//! product-with-virtual-row-sum. Sums-to-one is implied: row-sum gives
//! `sum_j e_ij = sum_i`, rescale gives `sum_j out_ij * sum_i = sum_j e_ij`,
//! so `sum_j out_ij = 1` (sum != 0).

use crate::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::logup_gkr::{prove_lookup_fractional, verify_lookup_fractional, FractionalProof};
use crate::mle;
use crate::sumcheck::{prove_virtual, verify_virtual, VirtualProof};

pub struct SoftmaxScaledProof {
    pub lookup: FractionalProof,
    pub row_sum: VirtualProof,
    pub rescale: VirtualProof,
    pub alpha: Goldilocks,
    pub beta: Goldilocks,
    pub r_sum: Vec<Goldilocks>,
    pub sum_ch: Vec<Goldilocks>,
    pub r_scale: Vec<Goldilocks>,
}

pub fn prove_softmax_scaled(
    indices: &[u32],
    e: &[Goldilocks],
    out: &[Goldilocks],
    table: &[Goldilocks],
    m: usize,
    n: usize,
    rng: &mut XorShift64,
) -> SoftmaxScaledProof {
    assert_eq!(indices.len(), m * n);
    assert_eq!(e.len(), m * n);
    assert_eq!(out.len(), m * n);

    let alpha = rng.field();
    let beta = rng.field();
    let lookup = prove_lookup_fractional(indices, e, table, alpha, beta, rng);

    let sum: Vec<Goldilocks> = (0..m)
        .map(|i| (0..n).fold(Goldilocks::ZERO, |acc, j| acc + e[i * n + j]))
        .collect();
    let r_sum: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let eq_i = mle::eq_evals(&r_sum);
    let eq_broadcast: Vec<Goldilocks> = (0..m * n).map(|idx| eq_i[idx / n]).collect();
    let sum_claimed = mle::eval(&sum, &r_sum);
    let sum_terms = vec![(Goldilocks::ONE, vec![0usize, 1usize])];
    let sum_mles: Vec<&[Goldilocks]> = vec![&eq_broadcast, e];
    let sum_ch: Vec<Goldilocks> = (0..(m * n).trailing_zeros() as usize).map(|_| rng.field()).collect();
    let row_sum = prove_virtual(&sum_mles, &sum_terms, sum_claimed, &sum_ch);

    let sum_broadcast: Vec<Goldilocks> = (0..m * n).map(|idx| sum[idx / n]).collect();
    let r_scale: Vec<Goldilocks> = (0..(m * n).trailing_zeros() as usize).map(|_| rng.field()).collect();
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let scale_terms = vec![(Goldilocks::ONE, vec![0usize, 2usize]), (neg, vec![1usize])];
    let scale_mles: Vec<&[Goldilocks]> = vec![out, e, &sum_broadcast];
    let rescale = prove_virtual(&scale_mles, &scale_terms, Goldilocks::ZERO, &r_scale);

    SoftmaxScaledProof {
        lookup,
        row_sum,
        rescale,
        alpha,
        beta,
        r_sum,
        sum_ch,
        r_scale,
    }
}

pub fn verify_softmax_scaled(
    proof: &SoftmaxScaledProof,
    indices: &[u32],
    e: &[Goldilocks],
    out: &[Goldilocks],
    table: &[Goldilocks],
    m: usize,
    n: usize,
) -> bool {
    assert_eq!(indices.len(), m * n);
    assert_eq!(e.len(), m * n);
    assert_eq!(out.len(), m * n);

    if !verify_lookup_fractional(&proof.lookup, indices, e, table, proof.alpha, proof.beta) {
        return false;
    }

    let sum: Vec<Goldilocks> = (0..m)
        .map(|i| (0..n).fold(Goldilocks::ZERO, |acc, j| acc + e[i * n + j]))
        .collect();
    let eq_i = mle::eq_evals(&proof.r_sum);
    let eq_broadcast: Vec<Goldilocks> = (0..m * n).map(|idx| eq_i[idx / n]).collect();
    let sum_claimed = mle::eval(&sum, &proof.r_sum);
    let sum_terms = vec![(Goldilocks::ONE, vec![0usize, 1usize])];
    let sum_fe = vec![
        mle::eval(&eq_broadcast, &proof.sum_ch),
        mle::eval(e, &proof.sum_ch),
    ];
    if !verify_virtual(&proof.row_sum, &sum_terms, sum_claimed, &proof.sum_ch, &sum_fe) {
        return false;
    }

    let sum_broadcast: Vec<Goldilocks> = (0..m * n).map(|idx| sum[idx / n]).collect();
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let scale_terms = vec![(Goldilocks::ONE, vec![0usize, 2usize]), (neg, vec![1usize])];
    let scale_fe = vec![
        mle::eval(out, &proof.r_scale),
        mle::eval(e, &proof.r_scale),
        mle::eval(&sum_broadcast, &proof.r_scale),
    ];
    verify_virtual(&proof.rescale, &scale_terms, Goldilocks::ZERO, &proof.r_scale, &scale_fe)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_scaled_roundtrip() {
        let mut rng = XorShift64::new(0x5EED);
        let (m, n) = (8usize, 8usize);
        let table_len = 1usize << 8;
        let table: Vec<Goldilocks> = (0..table_len).map(|_| rng.field()).collect();
        let indices: Vec<u32> = (0..m * n).map(|_| (rng.next_u64() % table_len as u64) as u32).collect();
        let e: Vec<Goldilocks> = indices.iter().map(|&i| table[i as usize]).collect();
        let sum: Vec<Goldilocks> = (0..m).map(|i| (0..n).fold(Goldilocks::ZERO, |a, j| a + e[i * n + j])).collect();
        let out: Vec<Goldilocks> = (0..m * n).map(|idx| e[idx] * sum[idx / n].inverse()).collect();
        let proof = prove_softmax_scaled(&indices, &e, &out, &table, m, n, &mut rng);
        assert!(verify_softmax_scaled(&proof, &indices, &e, &out, &table, m, n));
        let mut bad = out.clone();
        bad[0] = bad[0] + Goldilocks::ONE;
        assert!(!verify_softmax_scaled(&proof, &indices, &e, &bad, &table, m, n));
    }
}
