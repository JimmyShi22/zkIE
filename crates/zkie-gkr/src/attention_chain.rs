//! Claim-chained single-head attention: q/k/v projections -> scores = q@k^T ->
//! softmax -> attn = probs@v -> o projection -> residual (y = x + o), proven as
//! ONE coherent chain. The multiply-consumed input `x` (claimed by q, k, v
//! projections AND the residual) is bound via `same_poly`. Intermediates are
//! virtual (not committed) — this is the "one g per shard" shape for attention.

use crate::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::fixed_point::{from_i64, to_i64};
use crate::logup_gkr::{prove_lookup_fractional, verify_lookup_fractional, FractionalProof};
use crate::matmul::{self, MatmulProof};
use crate::mle;
use crate::projection::{prove_projection, verify_projection, ProjectionProof};
use crate::same_poly::{prove_same_poly, verify_same_poly, SamePolyProof};
use crate::softmax_scaled::{prove_softmax_scaled, verify_softmax_scaled, SoftmaxScaledProof};
use crate::sumcheck::{prove_virtual, verify_virtual, VirtualProof};

pub struct AttentionChainProof {
    pub q: ProjectionProof,
    pub k: ProjectionProof,
    pub v: ProjectionProof,
    pub scores: MatmulProof,
    pub softmax: SoftmaxScaledProof,
    pub attn: MatmulProof,
    pub o: ProjectionProof,
    pub residual: VirtualProof,
    pub same_x: SamePolyProof,
    pub r: Vec<Goldilocks>,
}

fn mm(a: &[Goldilocks], b: &[Goldilocks], m: usize, k: usize, n: usize) -> Vec<Goldilocks> {
    crate::par::mm_par(a, b, m, k, n, 64)
}

fn transpose(a: &[Goldilocks], m: usize, k: usize) -> Vec<Goldilocks> {
    let mut t = vec![Goldilocks::ZERO; k * m];
    for i in 0..m {
        for kk in 0..k {
            t[kk * m + i] = a[i * k + kk];
        }
    }
    t
}

fn round_div(a: i64, b: i64) -> i64 {
    let q = a.div_euclid(b);
    let r = a.rem_euclid(b);
    if r * 2 >= b {
        q + 1
    } else {
        q
    }
}

fn projection_fp(
    x: &[Goldilocks],
    w: &[Goldilocks],
    b: &[Goldilocks],
    m: usize,
    k: usize,
    n: usize,
    shift: u32,
) -> (Vec<Goldilocks>, Vec<Goldilocks>) {
    let h = mm(x, w, m, k, n);
    let out: Vec<Goldilocks> = (0..m * n)
        .map(|ij| from_i64(round_div(to_i64(h[ij]), 1i64 << shift) + to_i64(b[ij])))
        .collect();
    let rem: Vec<Goldilocks> = (0..m * n)
        .map(|ij| {
            from_i64(
                to_i64(h[ij]) - (to_i64(out[ij]) - to_i64(b[ij])) * (1i64 << shift)
                    + (1i64 << (shift - 1)),
            )
        })
        .collect();
    (out, rem)
}

#[allow(clippy::too_many_arguments)]
pub fn prove_attention_chain(
    x: &[Goldilocks],
    wq: &[Goldilocks],
    wk: &[Goldilocks],
    wv: &[Goldilocks],
    wo: &[Goldilocks],
    bias: &[Goldilocks],
    exp_table: &[Goldilocks],
    m: usize,
    d: usize,
    shift: u32,
    rng: &mut XorShift64,
) -> AttentionChainProof {
    let (q, q_rem) = projection_fp(x, wq, bias, m, d, d, shift);
    let (k, k_rem) = projection_fp(x, wk, bias, m, d, d, shift);
    let (v, v_rem) = projection_fp(x, wv, bias, m, d, d, shift);
    let kt = transpose(&k, m, d);
    let scores = mm(&q, &kt, m, d, m);
    let indices: Vec<u32> = scores
        .iter()
        .map(|&s| ((to_i64(s).max(0)) as u64 % exp_table.len() as u64) as u32)
        .collect();
    let e: Vec<Goldilocks> = indices.iter().map(|&i| exp_table[i as usize]).collect();
    let sum: Vec<Goldilocks> = (0..m).map(|i| (0..m).fold(Goldilocks::ZERO, |a, j| a + e[i * m + j])).collect();
    let probs: Vec<Goldilocks> = (0..m * m).map(|ij| e[ij] * sum[ij / m].inverse()).collect();
    let attn = mm(&probs, &v, m, m, d);
    let (o, o_rem) = projection_fp(&attn, wo, bias, m, d, d, shift);
    let y: Vec<Goldilocks> = (0..m * d).map(|i| x[i] + o[i]).collect();

    let q_p = prove_projection(x, wq, bias, &q, &q_rem, m, d, d, shift, rng);
    let k_p = prove_projection(x, wk, bias, &k, &k_rem, m, d, d, shift, rng);
    let v_p = prove_projection(x, wv, bias, &v, &v_rem, m, d, d, shift, rng);

    let qt = transpose(&q, m, d);
    let u: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let vs: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let ch: Vec<Goldilocks> = (0..d.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let scores_p = matmul::prove(&qt, &kt, &scores, m, d, m, &u, &vs, &ch);

    let softmax_p = prove_softmax_scaled(&indices, &e, &probs, exp_table, m, m, rng);

    let pt = transpose(&probs, m, m);
    let ch_v: Vec<Goldilocks> = (0..d.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let ch_m: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let attn_p = matmul::prove(&pt, &v, &attn, m, m, d, &u, &ch_v, &ch_m);

    let o_p = prove_projection(&attn, wo, bias, &o, &o_rem, m, d, d, shift, rng);

    let r: Vec<Goldilocks> = (0..(m * d).trailing_zeros() as usize).map(|_| rng.field()).collect();
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let terms = vec![(Goldilocks::ONE, vec![2usize]), (neg, vec![0usize]), (neg, vec![1usize])];
    let mles: Vec<&[Goldilocks]> = vec![x, &o, &y];
    let residual = prove_virtual(&mles, &terms, Goldilocks::ZERO, &r);

    let mut xq = q_p.ch.clone();
    xq.extend_from_slice(&q_p.u);
    let mut xk = k_p.ch.clone();
    xk.extend_from_slice(&k_p.u);
    let mut xv = v_p.ch.clone();
    xv.extend_from_slice(&v_p.u);
    let cq = mle::eval(x, &xq);
    let ck = mle::eval(x, &xk);
    let cv = mle::eval(x, &xv);
    let cr = mle::eval(x, &r);
    let claims = vec![(xq, cq), (xk, ck), (xv, cv), (r.clone(), cr)];
    let same_x = prove_same_poly(x, &claims, rng);

    AttentionChainProof {
        q: q_p,
        k: k_p,
        v: v_p,
        scores: scores_p,
        softmax: softmax_p,
        attn: attn_p,
        o: o_p,
        residual,
        same_x,
        r,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn verify_attention_chain(
    proof: &AttentionChainProof,
    x: &[Goldilocks],
    wq: &[Goldilocks],
    wk: &[Goldilocks],
    wv: &[Goldilocks],
    wo: &[Goldilocks],
    bias: &[Goldilocks],
    exp_table: &[Goldilocks],
    m: usize,
    d: usize,
    shift: u32,
) -> bool {
    let (q, q_rem) = projection_fp(x, wq, bias, m, d, d, shift);
    let (k, k_rem) = projection_fp(x, wk, bias, m, d, d, shift);
    let (v, v_rem) = projection_fp(x, wv, bias, m, d, d, shift);
    let kt = transpose(&k, m, d);
    let scores = mm(&q, &kt, m, d, m);
    let indices: Vec<u32> = scores
        .iter()
        .map(|&s| ((to_i64(s).max(0)) as u64 % exp_table.len() as u64) as u32)
        .collect();
    let e: Vec<Goldilocks> = indices.iter().map(|&i| exp_table[i as usize]).collect();
    let sum: Vec<Goldilocks> = (0..m).map(|i| (0..m).fold(Goldilocks::ZERO, |a, j| a + e[i * m + j])).collect();
    let probs: Vec<Goldilocks> = (0..m * m).map(|ij| e[ij] * sum[ij / m].inverse()).collect();
    let attn = mm(&probs, &v, m, m, d);
    let (o, o_rem) = projection_fp(&attn, wo, bias, m, d, d, shift);
    let y: Vec<Goldilocks> = (0..m * d).map(|i| x[i] + o[i]).collect();

    if !verify_projection(&proof.q, x, wq, bias, &q, &q_rem, m, d, d, shift)
        || !verify_projection(&proof.k, x, wk, bias, &k, &k_rem, m, d, d, shift)
        || !verify_projection(&proof.v, x, wv, bias, &v, &v_rem, m, d, d, shift)
    {
        return false;
    }
    if !verify_softmax_scaled(&proof.softmax, &indices, &e, &probs, exp_table, m, m) {
        return false;
    }
    if !verify_projection(&proof.o, &attn, wo, bias, &o, &o_rem, m, d, d, shift) {
        return false;
    }
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let terms = vec![(Goldilocks::ONE, vec![2usize]), (neg, vec![0usize]), (neg, vec![1usize])];
    let fe = vec![
        mle::eval(x, &proof.r),
        mle::eval(&o, &proof.r),
        mle::eval(&y, &proof.r),
    ];
    if !verify_virtual(&proof.residual, &terms, Goldilocks::ZERO, &proof.r, &fe) {
        return false;
    }
    // cross-shard binding on x (q/k/v + residual all open the same x)
    let claims = vec![
        {
            let mut p = proof.q.ch.clone();
            p.extend_from_slice(&proof.q.u);
            (p.clone(), mle::eval(x, &p))
        },
        {
            let mut p = proof.k.ch.clone();
            p.extend_from_slice(&proof.k.u);
            (p.clone(), mle::eval(x, &p))
        },
        {
            let mut p = proof.v.ch.clone();
            p.extend_from_slice(&proof.v.u);
            (p.clone(), mle::eval(x, &p))
        },
        (proof.r.clone(), mle::eval(x, &proof.r)),
    ];
    verify_same_poly(&proof.same_x, x, &claims).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_chain_roundtrip() {
        let mut rng = XorShift64::new(0xA7A7);
        let (m, d) = (4usize, 4usize);
        let shift = 8u32;
        let x: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let wq: Vec<Goldilocks> = (0..d * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let wk: Vec<Goldilocks> = (0..d * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let wv: Vec<Goldilocks> = (0..d * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let wo: Vec<Goldilocks> = (0..d * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let bias: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect();
        let exp_table: Vec<Goldilocks> = (0..(1usize << 8)).map(|_| rng.field()).collect();
        let proof = prove_attention_chain(&x, &wq, &wk, &wv, &wo, &bias, &exp_table, m, d, shift, &mut rng);
        assert!(verify_attention_chain(&proof, &x, &wq, &wk, &wv, &wo, &bias, &exp_table, m, d, shift));
    }
}
