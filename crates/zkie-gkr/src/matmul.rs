//! Single-layer matmul reduction: `C = A @ B` via one sum-check over the
//! contraction index.
//!
//! `m`, `k`, `n` must be powers of two. `at` is `A` stored transposed in
//! row-major (`k x m`), so fixing the row index reduces to `partial_eval` of
//! the low bits.

use crate::field::Goldilocks;
use crate::{mle, sumcheck, sumcheck::SumcheckProof};

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
