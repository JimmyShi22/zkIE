//! Scalable softmax core: exp lookup + row-sum + rescale, all O(m*n).
//!
//! `out = e / sum` (field inverse), with `sum = row_sum(e)` proven as a
//! *separate* tensor (not a virtual MLE), so the rescale `out * sum_broadcast = e`
//! is an elementwise product over (i,j) — O(m*n), not the flattened O(m*n*n)
//! product-with-virtual-row-sum. Sums-to-one is implied: row-sum gives
//! `sum_j e_ij = sum_i`, rescale gives `sum_j out_ij * sum_i = sum_j e_ij`,
//! so `sum_j out_ij = 1` (sum != 0).

use zkie_core::common::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64, ChallengeSource};
use zkie_core::common::fixed_point::from_i64;
use zkie_core::common::logup_gkr::{
    prove_lookup_fractional, verify_lookup_fractional, FractionalProof,
};
use zkie_core::common::mle;
use zkie_core::common::sumcheck::{prove_virtual, verify_virtual, virtual_eval, VirtualProof};
use zkie_core::common::transcript::Transcript;

#[derive(serde::Serialize, serde::Deserialize)]
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

/// Rounded softmax output (GPT-2 semantics): `out = round(e * 2^16 / sum)` at
/// scale 2^16, with `rem = e * 2^16 - out * sum` as a witness (not range-checked).
#[derive(serde::Serialize, serde::Deserialize)]
pub struct SoftmaxRoundedProof {
    pub row_sum: VirtualProof,
    pub rescale: VirtualProof,
    pub r_sum: Vec<Goldilocks>,
    pub sum_ch: Vec<Goldilocks>,
    pub r_scale: Vec<Goldilocks>,
    pub claimed: Goldilocks,
}

pub fn prove_softmax_rounded(
    indices: &[u32],
    e: &[Goldilocks],
    out: &[Goldilocks],
    table: &[Goldilocks],
    m: usize,
    n: usize,
    transcript: &mut Transcript,
) -> SoftmaxRoundedProof {
    assert_eq!(indices.len(), m * n);
    assert_eq!(e.len(), m * n);
    assert_eq!(out.len(), m * n);

    let sum: Vec<Goldilocks> = (0..m)
        .map(|i| (0..n).fold(Goldilocks::ZERO, |a, j| a + e[i * n + j]))
        .collect();
    // r_sum is the eq point for the row-sum reduction (squeezed upfront: the
    // eq table needs it); sum_ch comes per-round from the sumcheck (S1).
    let r_sum: Vec<Goldilocks> = transcript.field_vec(m.trailing_zeros() as usize);
    let eq_i = mle::eq_evals(&r_sum);
    let eq_broadcast: Vec<Goldilocks> = (0..m * n).map(|idx| eq_i[idx / n]).collect();
    let claimed = mle::eval(&sum, &r_sum);
    let (row_sum, sum_ch) = prove_virtual(
        &[&eq_broadcast, e],
        &[(Goldilocks::ONE, vec![0usize, 1usize])],
        claimed,
        transcript,
    );

    let sum_broadcast: Vec<Goldilocks> = (0..m * n).map(|idx| sum[idx / n]).collect();
    let rem: Vec<Goldilocks> = (0..m * n)
        .map(|ij| {
            from_i64(
                zkie_core::common::fixed_point::to_i64(e[ij]) * (1i64 << 16)
                    - zkie_core::common::fixed_point::to_i64(out[ij])
                        * zkie_core::common::fixed_point::to_i64(sum_broadcast[ij]),
            )
        })
        .collect();
    let two16 = Goldilocks::from_u64(1u64 << 16);
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    // out * sum_broadcast - e * 2^16 - rem = 0
    let (rescale, r_scale) = prove_virtual(
        &[out, e, &sum_broadcast, &rem],
        &[
            (Goldilocks::ONE, vec![0usize, 2usize]),
            (neg * two16, vec![1usize]),
            (Goldilocks::ONE, vec![3usize]),
        ],
        Goldilocks::ZERO,
        transcript,
    );

    SoftmaxRoundedProof {
        row_sum,
        rescale,
        r_sum,
        sum_ch,
        r_scale,
        claimed,
    }
}

pub fn verify_softmax_rounded(
    proof: &SoftmaxRoundedProof,
    m: usize,
    n: usize,
    transcript: &mut Transcript,
) -> bool {
    let arity_mn = (m * n).trailing_zeros() as usize;
    let r_sum: Vec<Goldilocks> = transcript.field_vec(m.trailing_zeros() as usize);
    if r_sum != proof.r_sum {
        return false;
    }
    let row_terms = [(Goldilocks::ONE, vec![0usize, 1usize])];
    let (ok, sum_ch, fc) = verify_virtual(
        &proof.row_sum,
        &row_terms,
        proof.claimed,
        arity_mn,
        transcript,
    );
    if !ok
        || sum_ch != proof.sum_ch
        || fc != virtual_eval(&row_terms, &proof.row_sum.final_evals)
    {
        return false;
    }
    let two16 = Goldilocks::from_u64(1u64 << 16);
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let scale_terms = [
        (Goldilocks::ONE, vec![0usize, 2usize]),
        (neg * two16, vec![1usize]),
        (Goldilocks::ONE, vec![3usize]),
    ];
    let (ok, r_scale, fc) = verify_virtual(
        &proof.rescale,
        &scale_terms,
        Goldilocks::ZERO,
        arity_mn,
        transcript,
    );
    ok && r_scale == proof.r_scale && fc == virtual_eval(&scale_terms, &proof.rescale.final_evals)
}

pub fn prove_softmax_scaled(
    indices: &[u32],
    e: &[Goldilocks],
    out: &[Goldilocks],
    table: &[Goldilocks],
    m: usize,
    n: usize,
    transcript: &mut Transcript,
) -> SoftmaxScaledProof {
    assert_eq!(indices.len(), m * n);
    assert_eq!(e.len(), m * n);
    assert_eq!(out.len(), m * n);

    let alpha = transcript.field();
    let beta = transcript.field();
    let lookup = prove_lookup_fractional(indices, e, table, alpha, beta, transcript);

    let sum: Vec<Goldilocks> = (0..m)
        .map(|i| (0..n).fold(Goldilocks::ZERO, |acc, j| acc + e[i * n + j]))
        .collect();
    let r_sum: Vec<Goldilocks> = transcript.field_vec(m.trailing_zeros() as usize);
    let eq_i = mle::eq_evals(&r_sum);
    let eq_broadcast: Vec<Goldilocks> = (0..m * n).map(|idx| eq_i[idx / n]).collect();
    let sum_claimed = mle::eval(&sum, &r_sum);
    let sum_terms = vec![(Goldilocks::ONE, vec![0usize, 1usize])];
    let sum_mles: Vec<&[Goldilocks]> = vec![&eq_broadcast, e];
    let (row_sum, sum_ch) = prove_virtual(&sum_mles, &sum_terms, sum_claimed, transcript);

    let sum_broadcast: Vec<Goldilocks> = (0..m * n).map(|idx| sum[idx / n]).collect();
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let scale_terms = vec![(Goldilocks::ONE, vec![0usize, 2usize]), (neg, vec![1usize])];
    let scale_mles: Vec<&[Goldilocks]> = vec![out, e, &sum_broadcast];
    let (rescale, r_scale) = prove_virtual(&scale_mles, &scale_terms, Goldilocks::ZERO, transcript);

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
    transcript: &mut Transcript,
) -> bool {
    assert_eq!(indices.len(), m * n);
    assert_eq!(e.len(), m * n);
    assert_eq!(out.len(), m * n);

    // Replay the alpha/beta preamble, then the fractional transcript.
    if transcript.field() != proof.alpha || transcript.field() != proof.beta {
        return false;
    }
    if !verify_lookup_fractional(&proof.lookup, indices, e, table, proof.alpha, proof.beta, transcript) {
        return false;
    }

    let sum: Vec<Goldilocks> = (0..m)
        .map(|i| (0..n).fold(Goldilocks::ZERO, |acc, j| acc + e[i * n + j]))
        .collect();
    let r_sum: Vec<Goldilocks> = transcript.field_vec(m.trailing_zeros() as usize);
    if r_sum != proof.r_sum {
        return false;
    }
    let eq_i = mle::eq_evals(&r_sum);
    let eq_broadcast: Vec<Goldilocks> = (0..m * n).map(|idx| eq_i[idx / n]).collect();
    let sum_claimed = mle::eval(&sum, &r_sum);
    let sum_terms = vec![(Goldilocks::ONE, vec![0usize, 1usize])];
    let (ok, sum_ch, fc) = verify_virtual(
        &proof.row_sum,
        &sum_terms,
        sum_claimed,
        (m * n).trailing_zeros() as usize,
        transcript,
    );
    if !ok || sum_ch != proof.sum_ch {
        return false;
    }
    let sum_fe = vec![
        mle::eval(&eq_broadcast, &sum_ch),
        mle::eval(e, &sum_ch),
    ];
    if fc != virtual_eval(&sum_terms, &sum_fe) {
        return false;
    }

    let sum_broadcast: Vec<Goldilocks> = (0..m * n).map(|idx| sum[idx / n]).collect();
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let scale_terms = vec![(Goldilocks::ONE, vec![0usize, 2usize]), (neg, vec![1usize])];
    let (ok, r_scale, fc) = verify_virtual(
        &proof.rescale,
        &scale_terms,
        Goldilocks::ZERO,
        (m * n).trailing_zeros() as usize,
        transcript,
    );
    if !ok || r_scale != proof.r_scale {
        return false;
    }
    let scale_fe = vec![
        mle::eval(out, &r_scale),
        mle::eval(e, &r_scale),
        mle::eval(&sum_broadcast, &r_scale),
    ];
    fc == virtual_eval(&scale_terms, &scale_fe)
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
        let indices: Vec<u32> = (0..m * n)
            .map(|_| (rng.next_u64() % table_len as u64) as u32)
            .collect();
        let e: Vec<Goldilocks> = indices.iter().map(|&i| table[i as usize]).collect();
        let sum: Vec<Goldilocks> = (0..m)
            .map(|i| (0..n).fold(Goldilocks::ZERO, |a, j| a + e[i * n + j]))
            .collect();
        let out: Vec<Goldilocks> = (0..m * n)
            .map(|idx| e[idx] * sum[idx / n].inverse())
            .collect();
        let mut tp = Transcript::new(b"zkie-softmax-test");
        tp.absorb_u64(0x5EED);
        let proof = prove_softmax_scaled(&indices, &e, &out, &table, m, n, &mut tp);
        let mut tv = Transcript::new(b"zkie-softmax-test");
        tv.absorb_u64(0x5EED);
        assert!(verify_softmax_scaled(
            &proof, &indices, &e, &out, &table, m, n, &mut tv
        ));
        let mut bad = out.clone();
        bad[0] = bad[0] + Goldilocks::ONE;
        let mut tv2 = Transcript::new(b"zkie-softmax-test");
        tv2.absorb_u64(0x5EED);
        assert!(!verify_softmax_scaled(
            &proof, &indices, &e, &bad, &table, m, n, &mut tv2
        ));
    }
}
