use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
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
fn matmul_completeness() {
    let mut rng = XorShift64::new(7);
    let (m, k, n) = (8usize, 8usize, 8usize);
    let a: Vec<Goldilocks> = (0..m * k).map(|_| rng.field()).collect();
    let b: Vec<Goldilocks> = (0..k * n).map(|_| rng.field()).collect();
    let c = dense(&a, &b, m, k, n);
    let at = transpose(&a, m, k);

    let u: Vec<Goldilocks> = (0..m.trailing_zeros()).map(|_| rng.field()).collect();
    let v: Vec<Goldilocks> = (0..n.trailing_zeros()).map(|_| rng.field()).collect();
    let challenges: Vec<Goldilocks> = (0..k.trailing_zeros()).map(|_| rng.field()).collect();

    let proof = matmul::prove(&at, &b, &c, m, k, n, &u, &v, &challenges);

    let mut f_point = u.clone();
    f_point.extend_from_slice(&challenges);
    let f_eval = mle::eval(&at, &f_point);
    let mut h_point = v.clone();
    h_point.extend_from_slice(&challenges);
    let h_eval = mle::eval(&b, &h_point);

    assert!(matmul::verify(&proof, &challenges, f_eval, h_eval));
}

#[test]
fn matmul_soundness_against_wrong_c() {
    let mut rng = XorShift64::new(8);
    let (m, k, n) = (8usize, 8usize, 8usize);
    let a: Vec<Goldilocks> = (0..m * k).map(|_| rng.field()).collect();
    let b: Vec<Goldilocks> = (0..k * n).map(|_| rng.field()).collect();
    let mut c = dense(&a, &b, m, k, n);
    c[0] = c[0] + Goldilocks::ONE;
    let at = transpose(&a, m, k);

    let u: Vec<Goldilocks> = (0..m.trailing_zeros()).map(|_| rng.field()).collect();
    let v: Vec<Goldilocks> = (0..n.trailing_zeros()).map(|_| rng.field()).collect();
    let challenges: Vec<Goldilocks> = (0..k.trailing_zeros()).map(|_| rng.field()).collect();

    let proof = matmul::prove(&at, &b, &c, m, k, n, &u, &v, &challenges);

    let mut f_point = u.clone();
    f_point.extend_from_slice(&challenges);
    let f_eval = mle::eval(&at, &f_point);
    let mut h_point = v.clone();
    h_point.extend_from_slice(&challenges);
    let h_eval = mle::eval(&b, &h_point);

    assert!(!matmul::verify(&proof, &challenges, f_eval, h_eval));
}
