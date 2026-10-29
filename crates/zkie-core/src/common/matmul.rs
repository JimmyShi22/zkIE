//! Single-layer matmul reduction: `C = A @ B` via one sum-check over the
//! contraction index.
//!
//! `m`, `k`, `n` must be powers of two. `at` is `A` stored transposed in
//! row-major (`k x m`), so fixing the row index reduces to `partial_eval` of
//! the low bits.

use crate::common::field::{Goldilocks, PrimeCharacteristicRing};
use crate::common::transcript::Transcript;
use crate::{common::mle, common::sumcheck, common::sumcheck::SumcheckProof};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct MatmulProof {
    pub claimed: Goldilocks,
    pub sumcheck: SumcheckProof,
}

/// Fiat-Shamir prove (S1): `u`/`v` are statement-bound claim points squeezed by
/// the caller; the contraction-index sumcheck challenges are derived per-round
/// from the sumcheck's own round messages and returned.
pub fn prove(
    at: &[Goldilocks],
    b: &[Goldilocks],
    c: &[Goldilocks],
    m: usize,
    k: usize,
    n: usize,
    u: &[Goldilocks],
    v: &[Goldilocks],
    transcript: &mut Transcript,
) -> (MatmulProof, Vec<Goldilocks>) {
    assert_eq!(at.len(), k * m);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    assert_eq!(u.len(), m.trailing_zeros() as usize);
    assert_eq!(v.len(), n.trailing_zeros() as usize);

    let a_restricted = mle::partial_eval(at, u);
    let b_restricted = mle::partial_eval(b, v);
    debug_assert_eq!(a_restricted.len(), k);
    debug_assert_eq!(b_restricted.len(), k);

    let mut c_point = v.to_vec();
    c_point.extend_from_slice(u);
    let claimed = mle::eval(c, &c_point);

    let (sumcheck, challenges) = sumcheck::prove(&a_restricted, &b_restricted, claimed, transcript);
    (MatmulProof { claimed, sumcheck }, challenges)
}

/// Fiat-Shamir verify (S1): replay the sumcheck rounds, deriving the
/// contraction-index challenges per-round. `num_rounds` is the statement-derived
/// `k.trailing_zeros()`.
pub fn verify(
    proof: &MatmulProof,
    num_rounds: usize,
    transcript: &mut Transcript,
) -> (bool, Vec<Goldilocks>) {
    sumcheck::verify(&proof.sumcheck, proof.claimed, num_rounds, transcript)
}

/// Nested matmul reduction: prove `E = (A @ B) @ D` with the intermediate
/// `C = A @ B` as a virtual MLE (never committed). Two chained
/// single-contraction sumchecks: step1 proves `E = C @ D` over `n` and leaves a
/// claim `C(u, ch_n)`; step2 proves `C = A @ B` over `k` at that same point and
/// leaves claims on `A` and `B`. Cost is O(n) + O(k) rounds, not O(m*k*n*l).
///
/// S1: step1 runs first on the transcript and its derived challenge vector
/// `ch_n` becomes step2's claim point on `B`; step2's challenges `ch_k` are
/// derived from step2's own rounds. Returns `(proof, ch_n, ch_k)`.
#[derive(serde::Serialize, serde::Deserialize)]
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
    transcript: &mut Transcript,
) -> (MatmulChainProof, Vec<Goldilocks>, Vec<Goldilocks>) {
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
    let (step1, ch_n) = prove(&ct, d, e, m, n, l, u, l_pt, transcript);
    let (step2, ch_k) = prove(at, b, c, m, k, n, u, &ch_n, transcript);
    (
        MatmulChainProof {
            claimed: step1.claimed,
            step1,
            step2,
        },
        ch_n,
        ch_k,
    )
}

/// Fiat-Shamir verify: replay step1 then step2 in the prover's transcript order.
pub fn verify_chain(
    proof: &MatmulChainProof,
    n_rounds: usize,
    k_rounds: usize,
    transcript: &mut Transcript,
) -> (bool, Vec<Goldilocks>, Vec<Goldilocks>) {
    let (ok1, ch_n) = verify(&proof.step1, n_rounds, transcript);
    if !ok1 {
        return (false, ch_n, Vec::new());
    }
    let (ok2, ch_k) = verify(&proof.step2, k_rounds, transcript);
    (ok1 && ok2, ch_n, ch_k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::field::{XorShift64, ChallengeSource};

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

        let mut tp = Transcript::new(b"zkie-matmul-test");
        tp.absorb_u64(1);
        let u: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| tp.field()).collect();
        let l_pt: Vec<Goldilocks> = (0..l.trailing_zeros() as usize).map(|_| tp.field()).collect();
        let (proof, ch_n, ch_k) = prove_chain(&at, &b, &c, &d, &e, m, k, n, l, &u, &l_pt, &mut tp);

        let mut tv = Transcript::new(b"zkie-matmul-test");
        tv.absorb_u64(1);
        for _ in 0..m.trailing_zeros() {
            tv.field();
        }
        for _ in 0..l.trailing_zeros() {
            tv.field();
        }
        let (ok, ch_n_v, ch_k_v) = verify_chain(
            &proof,
            n.trailing_zeros() as usize,
            k.trailing_zeros() as usize,
            &mut tv,
        );
        assert!(ok);
        assert_eq!(ch_n_v, ch_n);
        assert_eq!(ch_k_v, ch_k);

        let mut p_ak = ch_k.clone();
        p_ak.extend_from_slice(&u);
        let _a_eval = crate::common::mle::eval(&a, &p_ak); // A[u][ch_k]
        let mut p_b = ch_n.clone();
        p_b.extend_from_slice(&ch_k);
        let _b_eval = crate::common::mle::eval(&b, &p_b); // B[ch_k][ch_n]
        let mut p_d = l_pt.clone();
        p_d.extend_from_slice(&ch_n);
        let _d_eval = crate::common::mle::eval(&d, &p_d); // D[ch_n][l_pt]

        // tamper the claimed value -> reject
        let mut bad = proof;
        bad.step2.claimed = bad.step2.claimed + Goldilocks::ONE;
        let mut tv2 = Transcript::new(b"zkie-matmul-test");
        tv2.absorb_u64(1);
        for _ in 0..m.trailing_zeros() {
            tv2.field();
        }
        for _ in 0..l.trailing_zeros() {
            tv2.field();
        }
        let (ok2, _, _) = verify_chain(
            &bad,
            n.trailing_zeros() as usize,
            k.trailing_zeros() as usize,
            &mut tv2,
        );
        assert!(!ok2);
    }
}
