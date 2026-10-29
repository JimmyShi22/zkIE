//! Claim-chained RMSNorm: mean_sq = row_mean(x^2) (reduction) -> rsqrt =
//! rsqrt_lookup(mean_sq) (logUp fractional lookup) -> out = x * (rsqrt*w) + b
//! (affine). This is the last op type: a reduction + a lookup + an affine, the
//! same shape as softmax, wired as one chain with virtual intermediates.

use zkie_core::common::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64, ChallengeSource};
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_core::common::logup_gkr::{
    prove_lookup_fractional, verify_lookup_fractional, FractionalProof,
};
use zkie_core::common::mle;
use zkie_core::common::sumcheck::{prove_virtual, verify_virtual, virtual_eval, VirtualProof};
use zkie_core::common::transcript::Transcript;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct LayernormChainProof {
    pub mean_sq: VirtualProof,
    pub out: VirtualProof,
    pub r_mean: Vec<Goldilocks>,
    pub mean_ch: Vec<Goldilocks>,
    pub r_out: Vec<Goldilocks>,
    pub mean_claim: Goldilocks,
}

#[allow(clippy::too_many_arguments)]
pub fn prove_layernorm_chain(
    x: &[Goldilocks],
    w: &[Goldilocks],
    b: &[Goldilocks],
    rsqrt_table: &[Goldilocks],
    m: usize,
    d: usize,
    transcript: &mut Transcript,
) -> (LayernormChainProof, Vec<u32>, Vec<Goldilocks>) {
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

    // 1. mean_sq reduction: sum_{i,j} eq(r,i) * x^2 = mean_sq(r).
    //    r_mean is the eq point (squeezed upfront); the sumcheck challenges
    //    are derived per-round (S1).
    let r_mean: Vec<Goldilocks> = transcript.field_vec(m.trailing_zeros() as usize);
    let eq_i = mle::eq_evals(&r_mean);
    let eq_b: Vec<Goldilocks> = (0..m * d).map(|idx| eq_i[idx / d]).collect();
    let mean_claim = mle::eval(&mean_sq, &r_mean);
    let (mean_sq_proof, mean_ch) = prove_virtual(
        &[&eq_b, x],
        &[(Goldilocks::ONE, vec![0usize, 1usize, 1usize])],
        mean_claim,
        transcript,
    );

    // 3. out = x * rsqrt * w + b  (degree-3; S2: binds the weight w into the
    //    output relation instead of the prover-computed `scale` product).
    //    rsqrt comes from the lookup table, so the mles are
    //    [x, rsqrt_b, w, b, out] with x*rsqrt*w + b - out = 0.
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let rsqrt_b: Vec<Goldilocks> = (0..m * d).map(|ij| rsqrt[ij / d]).collect();
    let terms = vec![
        (Goldilocks::ONE, vec![0usize, 1usize, 2usize]),
        (Goldilocks::ONE, vec![3usize]),
        (neg, vec![4usize]),
    ];
    let (out_proof, r_out) =
        prove_virtual(&[x, &rsqrt_b, w, b, &out], &terms, Goldilocks::ZERO, transcript);

    (
        LayernormChainProof {
            mean_sq: mean_sq_proof,
            out: out_proof,
            r_mean,
            mean_ch,
            r_out,
            mean_claim,
        },
        rsqrt_idx,
        rsqrt,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn verify_layernorm_chain(
    proof: &LayernormChainProof,
    m: usize,
    d: usize,
    transcript: &mut Transcript,
) -> bool {
    let r_mean: Vec<Goldilocks> = transcript.field_vec(m.trailing_zeros() as usize);
    if r_mean != proof.r_mean {
        return false;
    }
    let mean_terms = [(Goldilocks::ONE, vec![0usize, 1usize, 1usize])];
    let (ok, mean_ch, fc) = verify_virtual(
        &proof.mean_sq,
        &mean_terms,
        proof.mean_claim,
        (m * d).trailing_zeros() as usize,
        transcript,
    );
    if !ok
        || mean_ch != proof.mean_ch
        || fc != virtual_eval(&mean_terms, &proof.mean_sq.final_evals)
    {
        return false;
    }
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let terms = vec![
        (Goldilocks::ONE, vec![0usize, 1usize, 2usize]),
        (Goldilocks::ONE, vec![3usize]),
        (neg, vec![4usize]),
    ];
    let (ok, r_out, fc) = verify_virtual(
        &proof.out,
        &terms,
        Goldilocks::ZERO,
        (m * d).trailing_zeros() as usize,
        transcript,
    );
    ok && r_out == proof.r_out && fc == virtual_eval(&terms, &proof.out.final_evals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layernorm_chain_roundtrip() {
        let mut rng = XorShift64::new(0x9E9E);
        let (m, d) = (8usize, 8usize);
        let x: Vec<Goldilocks> = (0..m * d)
            .map(|_| from_i64((rng.next_u64() % 10) as i64))
            .collect();
        let w: Vec<Goldilocks> = (0..m * d)
            .map(|_| from_i64((rng.next_u64() % 5) as i64 + 1))
            .collect();
        let b: Vec<Goldilocks> = (0..m * d)
            .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
            .collect();
        let rsqrt_table: Vec<Goldilocks> = (0..(1usize << 8)).map(|_| rng.field()).collect();
        let mut tp = Transcript::new(b"zkie-layernorm-test");
        tp.absorb_u64(0x9E9E);
        let (proof, _ridx, _rvals) = prove_layernorm_chain(&x, &w, &b, &rsqrt_table, m, d, &mut tp);
        let mut tv = Transcript::new(b"zkie-layernorm-test");
        tv.absorb_u64(0x9E9E);
        assert!(verify_layernorm_chain(&proof, m, d, &mut tv));
    }
}
