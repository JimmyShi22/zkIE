//! Reusable projection-block prover: out = round((x @ W) / 2^shift) + bias, with
//! the matmul output h = x@W never committed (chained via its GKR claim), the
//! affine rounding as an arithmetic constraint, and the range check as a logUp
//! lookup. This is the first reusable integration building block for the full
//! GPT-2 layer circuit.
use crate::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::fixed_point::{from_i64, to_i32};
use crate::logup_gkr::{prove_lookup_fractional, verify_lookup_fractional, FractionalProof};
use crate::matmul::{prove as matmul_prove, verify as matmul_verify, MatmulProof};
use crate::sumcheck::{prove_virtual, verify_virtual, VirtualProof};
pub struct ProjectionProof {
    pub matmul: MatmulProof,
    pub affine: VirtualProof,
    pub frac: FractionalProof,
    pub alpha: Goldilocks,
    pub beta: Goldilocks,
    pub pt: Vec<Goldilocks>,
    pub u: Vec<Goldilocks>,
    pub v: Vec<Goldilocks>,
    pub ch: Vec<Goldilocks>,
}
fn matmul_full(x: &[Goldilocks], w: &[Goldilocks], m: usize, k: usize, n: usize) -> Vec<Goldilocks> {
    let mut h = vec![Goldilocks::from_u64(0); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = Goldilocks::from_u64(0);
            for kk in 0..k {
                acc = acc + x[i * k + kk] * w[kk * n + j];
            }
            h[i * n + j] = acc;
        }
    }
    h
}
fn transpose(x: &[Goldilocks], m: usize, k: usize) -> Vec<Goldilocks> {
    let mut wt = vec![Goldilocks::from_u64(0); k * m];
    for kk in 0..k {
        for i in 0..m {
            wt[kk * m + i] = x[i * k + kk];
        }
    }
    wt
}
pub fn prove_projection(
    x: &[Goldilocks],
    w: &[Goldilocks],
    bias: &[Goldilocks],
    out: &[Goldilocks],
    rem_off: &[Goldilocks],
    m: usize,
    k: usize,
    n: usize,
    shift: u32,
    rng: &mut XorShift64,
) -> ProjectionProof {
    let half = Goldilocks::from_u64(1u64 << (shift - 1));
    let two_shift = Goldilocks::from_u64(1u64 << shift);
    let h = matmul_full(x, w, m, k, n);
    let wt = transpose(x, m, k);
    let u: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let ch: Vec<Goldilocks> = (0..k.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let matmul = matmul_prove(&wt, w, &h, m, k, n, &u, &v, &ch);
    let mut pt = v.clone();
    pt.extend_from_slice(&u);
    let eq = crate::mle::eq_evals(&pt);
    let ones = vec![Goldilocks::from_u64(1); m * n];
    let neg = from_i64(-1);
    let terms = vec![
        (Goldilocks::from_u64(1), vec![0usize, 1usize]),
        (neg, vec![0usize, 2usize]),
        (two_shift, vec![0usize, 3usize]),
        (neg * two_shift, vec![0usize, 4usize]),
        (neg * half, vec![0usize, 5usize]),
    ];
    let mles: Vec<&[Goldilocks]> = vec![&eq, &rem_off, &h, &out, &bias, &ones];
    let affine = prove_virtual(&mles, &terms, Goldilocks::from_u64(0), &pt);
    let table: Vec<Goldilocks> = (0..(1usize << shift)).map(|j| Goldilocks::from_u64(j as u64)).collect();
    let idx: Vec<u32> = rem_off.iter().map(|&vv| to_i32(vv) as u32).collect();
    let alpha = rng.field();
    let beta = rng.field();
    let frac = prove_lookup_fractional(&idx, rem_off, &table, alpha, beta, rng);
    ProjectionProof { matmul, affine, frac, alpha, beta, pt, u, v, ch }
}
pub fn verify_projection(
    proof: &ProjectionProof,
    x: &[Goldilocks],
    w: &[Goldilocks],
    bias: &[Goldilocks],
    out: &[Goldilocks],
    rem_off: &[Goldilocks],
    m: usize,
    k: usize,
    n: usize,
    shift: u32,
) -> bool {
    let half = Goldilocks::from_u64(1u64 << (shift - 1));
    let two_shift = Goldilocks::from_u64(1u64 << shift);
    let h = matmul_full(x, w, m, k, n);
    let eq = crate::mle::eq_evals(&proof.pt);
    let ones = vec![Goldilocks::from_u64(1); m * n];
    let neg = from_i64(-1);
    let terms = vec![
        (Goldilocks::from_u64(1), vec![0usize, 1usize]),
        (neg, vec![0usize, 2usize]),
        (two_shift, vec![0usize, 3usize]),
        (neg * two_shift, vec![0usize, 4usize]),
        (neg * half, vec![0usize, 5usize]),
    ];
    let mles: Vec<&[Goldilocks]> = vec![&eq, &rem_off, &h, &out, &bias, &ones];
    let final_evals: Vec<Goldilocks> = mles.iter().map(|mm| crate::mle::eval(mm, &proof.pt)).collect();
    let mut fe = final_evals;
    fe[2] = proof.matmul.claimed;
    if !verify_virtual(&proof.affine, &terms, Goldilocks::from_u64(0), &proof.pt, &fe) {
        return false;
    }
    let wt = transpose(x, m, k);
    let a_restricted = crate::mle::partial_eval(&wt, &proof.u);
    let b_restricted = crate::mle::partial_eval(w, &proof.v);
    let f_eval = crate::mle::eval(&a_restricted, &proof.ch);
    let h_eval = crate::mle::eval(&b_restricted, &proof.ch);
    if !matmul_verify(&proof.matmul, &proof.ch, f_eval, h_eval) {
        return false;
    }
    let table: Vec<Goldilocks> = (0..(1usize << shift)).map(|j| Goldilocks::from_u64(j as u64)).collect();
    let idx: Vec<u32> = rem_off.iter().map(|&vv| to_i32(vv) as u32).collect();
    verify_lookup_fractional(&proof.frac, &idx, rem_off, &table, proof.alpha, proof.beta)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn projection_block_roundtrip() {
        let mut rng = XorShift64::new(0xABCD);
        let (m, k, n) = (2usize, 2usize, 2usize);
        let shift = 4u32;
        let half = Goldilocks::from_u64(1u64 << (shift - 1));
        let two_shift = Goldilocks::from_u64(1u64 << shift);
        let div_round = |a: i64, b: i64| -> i64 { let q = a.div_euclid(b); let rr = a.rem_euclid(b); if rr * 2 >= b { q + 1 } else { q } };
        let x: Vec<Goldilocks> = (0..m * k).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
        let w: Vec<Goldilocks> = (0..k * n).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
        let bias: Vec<Goldilocks> = (0..m * n).map(|_| from_i64((rng.next_u64() % 20) as i64 - 10)).collect();
        let h = matmul_full(&x, &w, m, k, n);
        let out: Vec<Goldilocks> = (0..m * n).map(|ij| from_i64(div_round(crate::fixed_point::to_i64(h[ij]), 1i64 << shift) + crate::fixed_point::to_i64(bias[ij]))).collect();
        let rem_off: Vec<Goldilocks> = (0..m * n).map(|ij| {
            let h_i = crate::fixed_point::to_i64(h[ij]);
            let o_i = crate::fixed_point::to_i64(out[ij]);
            let b_i = crate::fixed_point::to_i64(bias[ij]);
            from_i64(h_i - (o_i - b_i) * (1i64 << shift) + (1i64 << (shift - 1)))
        }).collect();
        for &vv in &rem_off {
            assert!(to_i32(vv) >= 0 && to_i32(vv) < (1i32 << shift));
        }
        let _ = (half, two_shift);
        let proof = prove_projection(&x, &w, &bias, &out, &rem_off, m, k, n, shift, &mut rng);
        assert!(verify_projection(&proof, &x, &w, &bias, &out, &rem_off, m, k, n, shift));
        let mut bad_out = out.clone();
        bad_out[0] = bad_out[0] + Goldilocks::from_u64(1);
        assert!(!verify_projection(&proof, &x, &w, &bias, &bad_out, &rem_off, m, k, n, shift));
    }
}
