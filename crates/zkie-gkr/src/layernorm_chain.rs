//! Claim-chained RMSNorm: mean_sq = row_mean(x^2) (reduction) -> rsqrt =
//! rsqrt_lookup(mean_sq) (logUp fractional lookup) -> out = x * (rsqrt*w) + b
//! (affine). This is the last op type: a reduction + a lookup + an affine, the
//! same shape as softmax, wired as one chain with virtual intermediates.

use crate::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::fixed_point::{from_i64, to_i64};
use crate::logup_gkr::{prove_lookup_fractional, verify_lookup_fractional, FractionalProof};
use crate::mle;
use crate::sumcheck::{prove_virtual, verify_virtual, VirtualProof};

pub struct LayernormChainProof {
    pub mean_sq: VirtualProof,
    pub rsqrt: FractionalProof,
    pub out: VirtualProof,
    pub r_mean: Vec<Goldilocks>,
    pub mean_ch: Vec<Goldilocks>,
    pub r_out: Vec<Goldilocks>,
    pub alpha: Goldilocks,
    pub beta: Goldilocks,
}

#[allow(clippy::too_many_arguments)]
pub fn prove_layernorm_chain(
    x: &[Goldilocks],
    w: &[Goldilocks],
    b: &[Goldilocks],
    rsqrt_table: &[Goldilocks],
    m: usize,
    d: usize,
    rng: &mut XorShift64,
) -> LayernormChainProof {
    // forward
    let mean_sq: Vec<Goldilocks> = (0..m)
        .map(|i| (0..d).fold(Goldilocks::ZERO, |a, j| a + x[i * d + j] * x[i * d + j]))
        .collect();
    let rsqrt_idx: Vec<u32> = mean_sq
        .iter()
        .map(|&v| ((to_i64(v).max(0)) as u64 % rsqrt_table.len() as u64) as u32)
        .collect();
    let rsqrt: Vec<Goldilocks> = rsqrt_idx.iter().map(|&i| rsqrt_table[i as usize]).collect();
    let scale: Vec<Goldilocks> = (0..m * d).map(|ij| rsqrt[ij / d] * w[ij]).collect();
    let out: Vec<Goldilocks> = (0..m * d).map(|ij| x[ij] * scale[ij] + b[ij]).collect();

    // 1. mean_sq reduction: sum_{i,j} eq(r,i) * x^2 = mean_sq(r)
    let r_mean: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let eq_i = mle::eq_evals(&r_mean);
    let eq_b: Vec<Goldilocks> = (0..m * d).map(|idx| eq_i[idx / d]).collect();
    let mean_claim = mle::eval(&mean_sq, &r_mean);
    let mean_ch: Vec<Goldilocks> = (0..(m * d).trailing_zeros() as usize).map(|_| rng.field()).collect();
    let mean_sq_proof = prove_virtual(
        &[&eq_b, x],
        &[(Goldilocks::ONE, vec![0usize, 1usize, 1usize])],
        mean_claim,
        &mean_ch,
    );

    // 2. rsqrt lookup
    let alpha = rng.field();
    let beta = rng.field();
    let rsqrt_proof = prove_lookup_fractional(&rsqrt_idx, &rsqrt, rsqrt_table, alpha, beta, rng);

    // 3. out affine: out = x * scale + b  (i.e. x*scale + b - out = 0)
    let r_out: Vec<Goldilocks> = (0..(m * d).trailing_zeros() as usize).map(|_| rng.field()).collect();
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let terms = vec![
        (Goldilocks::ONE, vec![0usize, 1usize]),
        (Goldilocks::ONE, vec![2usize]),
        (neg, vec![3usize]),
    ];
    let out_proof = prove_virtual(
        &[x, &scale, b, &out],
        &terms,
        Goldilocks::ZERO,
        &r_out,
    );

    LayernormChainProof {
        mean_sq: mean_sq_proof,
        rsqrt: rsqrt_proof,
        out: out_proof,
        r_mean,
        mean_ch,
        r_out,
        alpha,
        beta,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn verify_layernorm_chain(
    proof: &LayernormChainProof,
    x: &[Goldilocks],
    w: &[Goldilocks],
    b: &[Goldilocks],
    rsqrt_table: &[Goldilocks],
    m: usize,
    d: usize,
) -> bool {
    let mean_sq: Vec<Goldilocks> = (0..m)
        .map(|i| (0..d).fold(Goldilocks::ZERO, |a, j| a + x[i * d + j] * x[i * d + j]))
        .collect();
    let rsqrt_idx: Vec<u32> = mean_sq
        .iter()
        .map(|&v| ((to_i64(v).max(0)) as u64 % rsqrt_table.len() as u64) as u32)
        .collect();
    let rsqrt: Vec<Goldilocks> = rsqrt_idx.iter().map(|&i| rsqrt_table[i as usize]).collect();
    let scale: Vec<Goldilocks> = (0..m * d).map(|ij| rsqrt[ij / d] * w[ij]).collect();
    let out: Vec<Goldilocks> = (0..m * d).map(|ij| x[ij] * scale[ij] + b[ij]).collect();

    let eq_i = mle::eq_evals(&proof.r_mean);
    let eq_b: Vec<Goldilocks> = (0..m * d).map(|idx| eq_i[idx / d]).collect();
    let mean_claim = mle::eval(&mean_sq, &proof.r_mean);
    let mean_fe = vec![
        mle::eval(&eq_b, &proof.mean_ch),
        mle::eval(x, &proof.mean_ch),
    ];
    if !verify_virtual(
        &proof.mean_sq,
        &[(Goldilocks::ONE, vec![0usize, 1usize, 1usize])],
        mean_claim,
        &proof.mean_ch,
        &mean_fe,
    ) {
        return false;
    }
    if !verify_lookup_fractional(&proof.rsqrt, &rsqrt_idx, &rsqrt, rsqrt_table, proof.alpha, proof.beta) {
        return false;
    }
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let terms = vec![
        (Goldilocks::ONE, vec![0usize, 1usize]),
        (Goldilocks::ONE, vec![2usize]),
        (neg, vec![3usize]),
    ];
    let out_fe = vec![
        mle::eval(x, &proof.r_out),
        mle::eval(&scale, &proof.r_out),
        mle::eval(b, &proof.r_out),
        mle::eval(&out, &proof.r_out),
    ];
    verify_virtual(&proof.out, &terms, Goldilocks::ZERO, &proof.r_out, &out_fe)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layernorm_chain_roundtrip() {
        let mut rng = XorShift64::new(0x9E9E);
        let (m, d) = (8usize, 8usize);
        let x: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64)).collect();
        let w: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 5) as i64 + 1)).collect();
        let b: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect();
        let rsqrt_table: Vec<Goldilocks> = (0..(1usize << 8)).map(|_| rng.field()).collect();
        let proof = prove_layernorm_chain(&x, &w, &b, &rsqrt_table, m, d, &mut rng);
        assert!(verify_layernorm_chain(&proof, &x, &w, &b, &rsqrt_table, m, d));
    }
}
