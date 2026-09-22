//! WHIR-committed GKR matmul for `m == 1` (a vector left operand).
//!
//! This is the core of the interpreter: commit a tensor once, then prove
//! `C = A @ B` from prescribed-point openings rather than raw evaluations. The
//! ctx32 TimesFM model runs every layer with batch/sequence `m == 1`, so the
//! left operand is a vector and needs no transpose point-swap.

use crate::field::{Goldilocks, XorShift64};
use crate::whir::{Commitment, OpeningProtocol, ProverData, Whir};
use crate::{matmul, mle};

/// A committed tensor (commitment + prover data + opening protocol).
pub struct Committed {
    pub commitment: Commitment,
    pub prover_data: ProverData,
    pub protocol: OpeningProtocol,
}

pub fn commit(whir: &Whir, values: &[Goldilocks]) -> Committed {
    let (commitment, prover_data, protocol) = whir.commit(values);
    Committed {
        commitment,
        prover_data,
        protocol,
    }
}

/// Prove `C = A @ B` with `A` of shape `1 x k`, `B` of shape `k x n`,
/// `C` of shape `1 x n`, all committed. Returns `true` iff every opening and
/// the sum-check verify.
#[allow(clippy::too_many_arguments)]
pub fn prove_matmul(
    whir_a: &Whir,
    a: &Committed,
    whir_b: &Whir,
    b: &Committed,
    whir_c: &Whir,
    c: &Committed,
    a_mat: &[Goldilocks],
    b_mat: &[Goldilocks],
    c_mat: &[Goldilocks],
    k: usize,
    n: usize,
    rng: &mut XorShift64,
) -> bool {
    let ch: Vec<Goldilocks> = (0..k.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let proof = matmul::prove(a_mat, b_mat, c_mat, 1, k, n, &[], &v, &ch);

    let (a_open, f) = whir_a.open(a.prover_data.clone(), &a.protocol, &ch);
    let mut bp = v.clone();
    bp.extend_from_slice(&ch);
    let (b_open, h) = whir_b.open(b.prover_data.clone(), &b.protocol, &bp);
    let (c_open, claimed) = whir_c.open(c.prover_data.clone(), &c.protocol, &v);

    let f_ok = whir_a.verify(&a.commitment, &a_open, &a.protocol, &ch).unwrap() == f;
    let h_ok = whir_b.verify(&b.commitment, &b_open, &b.protocol, &bp).unwrap() == h;
    let c_ok = whir_c.verify(&c.commitment, &c_open, &c.protocol, &v).unwrap() == claimed;
    let evals_ok = f == mle::eval(a_mat, &ch)
        && h == mle::eval(b_mat, &bp)
        && claimed == mle::eval(c_mat, &v);
    f_ok && h_ok && c_ok && evals_ok && claimed == proof.claimed && matmul::verify(&proof, &ch, f, h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::PrimeCharacteristicRing;

    #[test]
    fn committed_matmul_roundtrip() {
        let mut rng = XorShift64::new(0xabc);
        let (k, n) = (64usize, 64usize);
        let a: Vec<Goldilocks> = (0..k).map(|_| rng.field()).collect();
        let b: Vec<Goldilocks> = (0..k * n).map(|_| rng.field()).collect();
        let mut c = vec![Goldilocks::ZERO; n];
        for j in 0..n {
            let mut acc = Goldilocks::ZERO;
            for w in 0..k {
                acc = acc + a[w] * b[w * n + j];
            }
            c[j] = acc;
        }

        let whir_a = Whir::new_testing(6);
        let whir_b = Whir::new_testing(12);
        let whir_c = Whir::new_testing(6);
        let ca = commit(&whir_a, &a);
        let cb = commit(&whir_b, &b);
        let cc = commit(&whir_c, &c);

        assert!(prove_matmul(
            &whir_a, &ca, &whir_b, &cb, &whir_c, &cc, &a, &b, &c, k, n, &mut rng,
        ));
    }
}
