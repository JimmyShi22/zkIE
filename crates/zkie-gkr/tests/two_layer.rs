//! Two-layer MLP with cross-layer commitment binding.
//!
//! `Y1 = X @ W1` and `Y2 = Y1 @ W2`. The intermediate activation `Y1` is
//! committed exactly once and serves both layer 1's output and layer 2's input:
//! the upstream output commitment *is* the downstream input commitment.
//!
//! The matmul stores its left operand transposed (`at = A^T`), so layer 2's input
//! is `transpose(Y1)`. For square 8x8 matrices that is just a coordinate swap:
//! `transpose(Y1)` at `(u2 || r2)` equals `Y1` at `(r2 || u2)`, which lets one
//! commitment bind both layers without committing the same data twice.

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
fn two_layer_boundary_commitment_is_shared() {
    let mut rng = XorShift64::new(0x2f0f0f);
    let (m, k, n) = (8usize, 8usize, 8usize);
    let x: Vec<Goldilocks> = (0..m * k).map(|_| rng.field()).collect();
    let w1: Vec<Goldilocks> = (0..k * n).map(|_| rng.field()).collect();
    let w2: Vec<Goldilocks> = (0..k * n).map(|_| rng.field()).collect();
    let y1 = dense(&x, &w1, m, k, n);
    let y2 = dense(&y1, &w2, m, k, n);

    // Every matrix here is 8x8 = 2^6, so one PCS instance commits them all.
    let whir = Whir::new_testing(6);

    // Commit each unique matrix once. `y1` is shared across both layers; its
    // transposed form (layer 2's left operand) is derived via the point swap.
    let (x_c, x_pd, x_proto) = whir.commit(&transpose(&x, m, k));
    let (w1_c, w1_pd, w1_proto) = whir.commit(&w1);
    let (y1_c, y1_pd, y1_proto) = whir.commit(&y1);
    let (w2_c, w2_pd, w2_proto) = whir.commit(&w2);
    let (y2_c, y2_pd, y2_proto) = whir.commit(&y2);

    // Layer 1: Y1 = X @ W1.
    let (u1, v1, r1) = sample_points(&mut rng, m, n, k);
    let p1 = matmul::prove(&transpose(&x, m, k), &w1, &y1, m, k, n, &u1, &v1, &r1);

    // Layer 2: Y2 = Y1 @ W2.
    let (u2, v2, r2) = sample_points(&mut rng, m, n, k);
    let p2 = matmul::prove(&transpose(&y1, m, k), &w2, &y2, m, k, n, &u2, &v2, &r2);

    // For square matrices the transpose is a point swap (verified inline).
    assert_eq!(
        mle::eval(&transpose(&y1, m, k), &cat(&u2, &r2)),
        mle::eval(&y1, &cat(&r2, &u2))
    );

    // Open layer 1: X^T at (u1 || r1), W1 at (v1 || r1), Y1 at (v1 || u1).
    let (x_open, f1) = whir.open(x_pd, &x_proto, &cat(&u1, &r1));
    let (w1_open, h1) = whir.open(w1_pd, &w1_proto, &cat(&v1, &r1));
    let (y1_out_open, y1_out) = whir.open(y1_pd.clone(), &y1_proto, &cat(&v1, &u1));

    // Open layer 2: Y1 (same commitment, swapped point) at (r2 || u2),
    // W2 at (v2 || r2), Y2 at (v2 || u2).
    let (y1_in_open, f2) = whir.open(y1_pd, &y1_proto, &cat(&r2, &u2));
    let (w2_open, h2) = whir.open(w2_pd, &w2_proto, &cat(&v2, &r2));
    let (y2_open, y2_out) = whir.open(y2_pd, &y2_proto, &cat(&v2, &u2));

    // The same `y1` commitment verifies at layer 1's output point and layer 2's
    // input point — the boundary binding.
    assert_eq!(y1_out, mle::eval(&y1, &cat(&v1, &u1)));
    assert_eq!(f2, mle::eval(&y1, &cat(&r2, &u2)));
    assert_eq!(y2_out, mle::eval(&y2, &cat(&v2, &u2)));
    assert_eq!(whir.verify(&y1_c, &y1_out_open, &y1_proto, &cat(&v1, &u1)).unwrap(), y1_out);
    assert_eq!(whir.verify(&y1_c, &y1_in_open, &y1_proto, &cat(&r2, &u2)).unwrap(), f2);

    // Verify both sumchecks against opened (not trusted) evaluations.
    assert_eq!(whir.verify(&x_c, &x_open, &x_proto, &cat(&u1, &r1)).unwrap(), f1);
    assert_eq!(whir.verify(&w1_c, &w1_open, &w1_proto, &cat(&v1, &r1)).unwrap(), h1);
    assert_eq!(whir.verify(&y1_c, &y1_out_open, &y1_proto, &cat(&v1, &u1)).unwrap(), p1.claimed);
    assert!(matmul::verify(&p1, &r1, f1, h1));

    assert_eq!(whir.verify(&y1_c, &y1_in_open, &y1_proto, &cat(&r2, &u2)).unwrap(), f2);
    assert_eq!(whir.verify(&w2_c, &w2_open, &w2_proto, &cat(&v2, &r2)).unwrap(), h2);
    assert_eq!(whir.verify(&y2_c, &y2_open, &y2_proto, &cat(&v2, &u2)).unwrap(), p2.claimed);
    assert!(matmul::verify(&p2, &r2, f2, h2));
}

fn sample_points(
    rng: &mut XorShift64,
    m: usize,
    n: usize,
    k: usize,
) -> (Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>) {
    let u = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let v = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let r = (0..k.trailing_zeros() as usize).map(|_| rng.field()).collect();
    (u, v, r)
}

fn cat(a: &[Goldilocks], b: &[Goldilocks]) -> Vec<Goldilocks> {
    let mut v = a.to_vec();
    v.extend_from_slice(b);
    v
}
