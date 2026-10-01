//! Single-layer matmul reduction: `C = A @ B` via one sum-check over the
//! contraction index.
//!
//! `m`, `k`, `n` must be powers of two. `at` is `A` stored transposed in
//! row-major (`k x m`), so fixing the row index reduces to `partial_eval` of
//! the low bits.

use crate::common::field::{Goldilocks, PrimeCharacteristicRing};
use crate::{common::mle, common::sumcheck, common::sumcheck::SumcheckProof};

pub struct MatmulProof {
    pub claimed: Goldilocks,
    pub sumcheck: SumcheckProof,
}

pub fn prove(
    at: &[Goldilocks],
    b: &[Goldilocks],
    c: &[Goldilocks],
    m: usize,
    k: usize,
    n: usize,
    u: &[Goldilocks],
    v: &[Goldilocks],
    challenges: &[Goldilocks],
) -> MatmulProof {
    assert_eq!(at.len(), k * m);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    assert_eq!(u.len(), m.trailing_zeros() as usize);
    assert_eq!(v.len(), n.trailing_zeros() as usize);
    assert_eq!(challenges.len(), k.trailing_zeros() as usize);

    let a_restricted = mle::partial_eval(at, u);
    let b_restricted = mle::partial_eval(b, v);
    debug_assert_eq!(a_restricted.len(), k);
    debug_assert_eq!(b_restricted.len(), k);

    let mut c_point = v.to_vec();
    c_point.extend_from_slice(u);
    let claimed = mle::eval(c, &c_point);

    let sumcheck = sumcheck::prove(&a_restricted, &b_restricted, claimed, challenges);
    MatmulProof { claimed, sumcheck }
}

pub fn verify(
    proof: &MatmulProof,
    challenges: &[Goldilocks],
    f_eval: Goldilocks,
    h_eval: Goldilocks,
) -> bool {
    sumcheck::verify(&proof.sumcheck, proof.claimed, challenges, f_eval, h_eval)
}

/// Nested matmul reduction: prove `E = (A @ B) @ D` with the intermediate
/// `C = A @ B` as a virtual MLE (never committed). Two chained
/// single-contraction sumchecks: step1 proves `E = C @ D` over `n` and leaves a
/// claim `C(u, ch_n)`; step2 proves `C = A @ B` over `k` at that same point and
/// leaves claims on `A` and `B`. Cost is O(n) + O(k) rounds, not O(m*k*n*l).
pub struct MatmulChainProof {
    pub claimed: Goldilocks,
    pub step1: MatmulProof,
    pub step2: MatmulProof,
}

pub fn prove_chain(
    at: &[Goldilocks], // A^T (k x m)
    b: &[Goldilocks],  // B (k x n)
    c: &[Goldilocks],  // C = A @ B (m x n), virtual
    d: &[Goldilocks],  // D (n x l)
    e: &[Goldilocks],  // E = C @ D (m x l)
    m: usize,
    k: usize,
    n: usize,
    l: usize,
    u: &[Goldilocks],
    l_pt: &[Goldilocks],
    ch_n: &[Goldilocks],
    ch_k: &[Goldilocks],
) -> MatmulChainProof {
    assert_eq!(at.len(), k * m);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    assert_eq!(d.len(), n * l);
    assert_eq!(e.len(), m * l);
    let mut ct = vec![Goldilocks::ZERO; n * m];
    for i in 0..m {
        for j in 0..n {
            ct[j * m + i] = c[i * n + j];
        }
    }
    let step1 = prove(&ct, d, e, m, n, l, u, l_pt, ch_n);
    let step2 = prove(at, b, c, m, k, n, u, ch_n, ch_k);
    MatmulChainProof {
        claimed: step1.claimed,
        step1,
        step2,
    }
}

pub fn verify_chain(
    proof: &MatmulChainProof,
    ch_n: &[Goldilocks],
    ch_k: &[Goldilocks],
    a_eval: Goldilocks,
    b_eval: Goldilocks,
    d_eval: Goldilocks,
) -> bool {
    let c_eval = proof.step2.claimed;
    if !verify(&proof.step2, ch_k, a_eval, b_eval) {
        return false;
    }
    verify(&proof.step1, ch_n, c_eval, d_eval)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::field::XorShift64;

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

    fn transpose(a: &[Goldilocks], m: usize, k: usize) -> Vec<Goldilocks> {
        let mut t = vec![Goldilocks::ZERO; k * m];
        for i in 0..m {
            for kk in 0..k {
                t[kk * m + i] = a[i * k + kk];
            }
        }
        t
    }

    #[test]
    fn nested_chain_roundtrip() {
        let mut rng = XorShift64::new(0xC0FFEE);
        let (m, k, n, l) = (4usize, 4usize, 4usize, 4usize);
        let a: Vec<Goldilocks> = (0..m * k).map(|_| rng.field()).collect();
        let b: Vec<Goldilocks> = (0..k * n).map(|_| rng.field()).collect();
        let d: Vec<Goldilocks> = (0..n * l).map(|_| rng.field()).collect();
        let c = mm(&a, &b, m, k, n);
        let e = mm(&c, &d, m, n, l);
        let at = transpose(&a, m, k);
        let u: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let l_pt: Vec<Goldilocks> = (0..l.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let ch_n: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let ch_k: Vec<Goldilocks> = (0..k.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let proof = prove_chain(&at, &b, &c, &d, &e, m, k, n, l, &u, &l_pt, &ch_n, &ch_k);

        let mut p_ak = ch_k.clone();
        p_ak.extend_from_slice(&u);
        let a_eval = crate::common::mle::eval(&a, &p_ak); // A[u][ch_k]
        let mut p_b = ch_n.clone();
        p_b.extend_from_slice(&ch_k);
        let b_eval = crate::common::mle::eval(&b, &p_b); // B[ch_k][ch_n]
        let mut p_d = l_pt.clone();
        p_d.extend_from_slice(&ch_n);
        let d_eval = crate::common::mle::eval(&d, &p_d); // D[ch_n][l_pt]

        assert!(verify_chain(&proof, &ch_n, &ch_k, a_eval, b_eval, d_eval));
        assert!(!verify_chain(&proof, &ch_n, &ch_k, a_eval, b_eval + Goldilocks::ONE, d_eval));
    }
}
