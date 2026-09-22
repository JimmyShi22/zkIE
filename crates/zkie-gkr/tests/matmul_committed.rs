//! End-to-end: bind the GKR matmul's two sum-check evaluations (and its claimed
//! output) to WHIR opening proofs instead of accepting them as trusted inputs.

use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_gkr::whir::Whir;
use zkie_gkr::{matmul, mle};

fn dense(a: &[Goldilocks], b: &[Goldilocks], m: usize, k: usize, n: usize) -> Vec<Goldilocks> {
    let mut c = vec![Goldilocks::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = Goldilocks::ZERO;
            for w in 0..k {
                acc = acc + a[i * k + w] * b[w * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn transpose(a: &[Goldilocks], m: usize, k: usize) -> Vec<Goldilocks> {
    let mut at = vec![Goldilocks::ZERO; k * m];
    for i in 0..m {
        for w in 0..k {
            at[w * m + i] = a[i * k + w];
        }
    }
    at
}

#[test]
fn matmul_committed_openings_bind_evals() {
    let mut rng = XorShift64::new(0xc0ffee);
    let (m, k, n) = (8usize, 8usize, 8usize);
    let a: Vec<Goldilocks> = (0..m * k).map(|_| rng.field()).collect();
    let b: Vec<Goldilocks> = (0..k * n).map(|_| rng.field()).collect();
    let c = dense(&a, &b, m, k, n);
    let at = transpose(&a, m, k);

    let u: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let challenges: Vec<Goldilocks> =
        (0..k.trailing_zeros() as usize).map(|_| rng.field()).collect();

    // Commit each matrix as a flat MLE.
    let a_whir = Whir::new(m.trailing_zeros() as usize + k.trailing_zeros() as usize);
    let b_whir = Whir::new(k.trailing_zeros() as usize + n.trailing_zeros() as usize);
    let c_whir = Whir::new(m.trailing_zeros() as usize + n.trailing_zeros() as usize);
    let (a_commit, a_pd, a_proto) = a_whir.commit(&at);
    let (b_commit, b_pd, b_proto) = b_whir.commit(&b);
    let (c_commit, c_pd, c_proto) = c_whir.commit(&c);

    // Existing sum-check reduction.
    let proof = matmul::prove(&at, &b, &c, m, k, n, &u, &v, &challenges);

    // Opening points: A at (u || r), B at (v || r), C at (v || u).
    let mut a_point = u.clone();
    a_point.extend_from_slice(&challenges);
    let mut b_point = v.clone();
    b_point.extend_from_slice(&challenges);
    let mut c_point = v.clone();
    c_point.extend_from_slice(&u);

    let (a_open, a_eval) = a_whir.open(a_pd, &a_proto, &a_point);
    let (b_open, b_eval) = b_whir.open(b_pd, &b_proto, &b_point);
    let (c_open, c_eval) = c_whir.open(c_pd, &c_proto, &c_point);

    // Openings reproduce the hand-computed MLE evaluations.
    assert_eq!(a_eval, mle::eval(&at, &a_point));
    assert_eq!(b_eval, mle::eval(&b, &b_point));
    assert_eq!(c_eval, mle::eval(&c, &c_point));
    assert_eq!(c_eval, proof.claimed);

    // The verifier trusts commitments + openings only, not the raw evaluations.
    let f_eval = a_whir.verify(&a_commit, &a_open, &a_proto, &a_point).unwrap();
    let h_eval = b_whir.verify(&b_commit, &b_open, &b_proto, &b_point).unwrap();
    let claimed = c_whir.verify(&c_commit, &c_open, &c_proto, &c_point).unwrap();
    assert_eq!(claimed, proof.claimed);
    assert!(matmul::verify(&proof, &challenges, f_eval, h_eval));
}
