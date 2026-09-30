//! Claim-chained FFN layer: `out = x + proj`, `proj = projection(act)`,
//! `act = gelu(fc)`, `fc = projection(x)` — proven as ONE coherent chain where
//! the intermediate activations are virtual and the multiply-consumed boundary
//! tensors (`x`, `proj`) are bound via `same_poly`. This is the "one g threads
//! the layer" shape (DeepProve full GKR + logUp), not independent per-op proofs.

use crate::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::fixed_point::{from_i64, to_i32, to_i64};
use crate::logup_gkr::{prove_lookup_fractional, verify_lookup_fractional, FractionalProof};
use crate::mle;
use crate::projection::{prove_projection, verify_projection, ProjectionProof};
use crate::same_poly::{prove_same_poly, verify_same_poly, SamePolyProof};
use crate::sumcheck::{prove_virtual, verify_virtual, VirtualProof};

pub struct FfnChainProof {
    pub fc: ProjectionProof,
    pub gelu: FractionalProof,
    pub proj: ProjectionProof,
    pub residual: VirtualProof,
    pub same_x: SamePolyProof,
    pub same_proj: SamePolyProof,
    pub r: Vec<Goldilocks>,
    pub gelu_alpha: Goldilocks,
    pub gelu_beta: Goldilocks,
}

fn mm(a: &[Goldilocks], b: &[Goldilocks], m: usize, k: usize, n: usize) -> Vec<Goldilocks> {
    let mut c = vec![Goldilocks::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = Goldilocks::ZERO;
            for kk in 0..k {
                acc = acc + a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
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

#[allow(clippy::too_many_arguments)]
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

type Fwd = (
    Vec<Goldilocks>,
    Vec<Goldilocks>,
    Vec<u32>,
    Vec<Goldilocks>,
    Vec<Goldilocks>,
    Vec<Goldilocks>,
    Vec<Goldilocks>,
);

#[allow(clippy::too_many_arguments)]
fn ffn_forward(
    x: &[Goldilocks],
    fc_w: &[Goldilocks],
    fc_b: &[Goldilocks],
    proj_w: &[Goldilocks],
    proj_b: &[Goldilocks],
    gelu_table: &[Goldilocks],
    m: usize,
    d: usize,
    ffn: usize,
    shift: u32,
) -> Fwd {
    let (fc, fc_rem) = projection_fp(x, fc_w, fc_b, m, d, ffn, shift);
    let act_idx: Vec<u32> = fc.iter().map(|&v| ((to_i64(v).max(0)) as u64 % 64) as u32).collect();
    let act: Vec<Goldilocks> = act_idx.iter().map(|&i| gelu_table[i as usize]).collect();
    let (proj, proj_rem) = projection_fp(&act, proj_w, proj_b, m, ffn, d, shift);
    let out: Vec<Goldilocks> = (0..m * d).map(|i| x[i] + proj[i]).collect();
    (fc, fc_rem, act_idx, act, proj, proj_rem, out)
}

#[allow(clippy::too_many_arguments)]
pub fn prove_ffn_chain(
    x: &[Goldilocks],
    fc_w: &[Goldilocks],
    fc_b: &[Goldilocks],
    proj_w: &[Goldilocks],
    proj_b: &[Goldilocks],
    gelu_table: &[Goldilocks],
    m: usize,
    d: usize,
    ffn: usize,
    shift: u32,
    rng: &mut XorShift64,
) -> FfnChainProof {
    let (fc, fc_rem, act_idx, act, proj, proj_rem, out) =
        ffn_forward(x, fc_w, fc_b, proj_w, proj_b, gelu_table, m, d, ffn, shift);

    let fc_p = prove_projection(x, fc_w, fc_b, &fc, &fc_rem, m, d, ffn, shift, rng);
    let gelu_alpha = rng.field();
    let gelu_beta = rng.field();
    let gelu = prove_lookup_fractional(&act_idx, &act, gelu_table, gelu_alpha, gelu_beta, rng);
    let proj_p = prove_projection(&act, proj_w, proj_b, &proj, &proj_rem, m, ffn, d, shift, rng);

    let r: Vec<Goldilocks> = (0..(m * d).trailing_zeros() as usize).map(|_| rng.field()).collect();
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let terms = vec![(Goldilocks::ONE, vec![2usize]), (neg, vec![0usize]), (neg, vec![1usize])];
    let mles: Vec<&[Goldilocks]> = vec![x, &proj, &out];
    let residual = prove_virtual(&mles, &terms, Goldilocks::ZERO, &r);

    let mut x_p = fc_p.ch.clone();
    x_p.extend_from_slice(&fc_p.u);
    let x_claim_fc = mle::eval(x, &x_p);
    let x_claim_r = mle::eval(x, &r);
    let same_x = prove_same_poly(x, &[(x_p, x_claim_fc), (r.clone(), x_claim_r)], rng);

    let proj_claim_pt = mle::eval(&proj, &proj_p.pt);
    let proj_claim_r = mle::eval(&proj, &r);
    let same_proj =
        prove_same_poly(&proj, &[(proj_p.pt.clone(), proj_claim_pt), (r.clone(), proj_claim_r)], rng);

    FfnChainProof {
        fc: fc_p,
        gelu,
        proj: proj_p,
        residual,
        same_x,
        same_proj,
        r,
        gelu_alpha,
        gelu_beta,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn verify_ffn_chain(
    proof: &FfnChainProof,
    x: &[Goldilocks],
    fc_w: &[Goldilocks],
    fc_b: &[Goldilocks],
    proj_w: &[Goldilocks],
    proj_b: &[Goldilocks],
    gelu_table: &[Goldilocks],
    m: usize,
    d: usize,
    ffn: usize,
    shift: u32,
) -> bool {
    let (fc, fc_rem, act_idx, act, proj, proj_rem, out) =
        ffn_forward(x, fc_w, fc_b, proj_w, proj_b, gelu_table, m, d, ffn, shift);

    if !verify_projection(&proof.fc, x, fc_w, fc_b, &fc, &fc_rem, m, d, ffn, shift) {
        return false;
    }
    if !verify_lookup_fractional(&proof.gelu, &act_idx, &act, gelu_table, proof.gelu_alpha, proof.gelu_beta) {
        return false;
    }
    if !verify_projection(&proof.proj, &act, proj_w, proj_b, &proj, &proj_rem, m, ffn, d, shift) {
        return false;
    }
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let terms = vec![(Goldilocks::ONE, vec![2usize]), (neg, vec![0usize]), (neg, vec![1usize])];
    let mles: Vec<&[Goldilocks]> = vec![x, &proj, &out];
    let fe = vec![
        mle::eval(x, &proof.r),
        mle::eval(&proj, &proof.r),
        mle::eval(&out, &proof.r),
    ];
    if !verify_virtual(&proof.residual, &terms, Goldilocks::ZERO, &proof.r, &fe) {
        return false;
    }

    let mut x_p = proof.fc.ch.clone();
    x_p.extend_from_slice(&proof.fc.u);
    let x_claim_fc = mle::eval(x, &x_p);
    let x_claim_r = mle::eval(x, &proof.r);
    if verify_same_poly(&proof.same_x, x, &[(x_p, x_claim_fc), (proof.r.clone(), x_claim_r)]).is_none() {
        return false;
    }
    let proj_claim_pt = mle::eval(&proj, &proof.proj.pt);
    let proj_claim_r = mle::eval(&proj, &proof.r);
    if verify_same_poly(
        &proof.same_proj,
        &proj,
        &[(proof.proj.pt.clone(), proj_claim_pt), (proof.r.clone(), proj_claim_r)],
    )
    .is_none()
    {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffn_chain_roundtrip() {
        let mut rng = XorShift64::new(0xF7F7);
        let (m, d, ffn) = (4usize, 8usize, 8usize);
        let shift = 8u32;
        let x: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let fc_w: Vec<Goldilocks> = (0..d * ffn).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let fc_b: Vec<Goldilocks> = (0..m * ffn).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect();
        let proj_w: Vec<Goldilocks> = (0..ffn * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let proj_b: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect();
        let gelu_table: Vec<Goldilocks> = (0..64).map(|j| from_i64((j as i64).pow(2) % 1000)).collect();
        let proof = prove_ffn_chain(
            &x, &fc_w, &fc_b, &proj_w, &proj_b, &gelu_table, m, d, ffn, shift, &mut rng,
        );
        assert!(verify_ffn_chain(
            &proof, &x, &fc_w, &fc_b, &proj_w, &proj_b, &gelu_table, m, d, ffn, shift
        ));
    }
}
