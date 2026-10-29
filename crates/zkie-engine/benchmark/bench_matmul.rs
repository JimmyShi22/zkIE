//! Compares the GKR proof overhead for one matmul against the current Halo2
//! `DotProductChip` row count.
//!
//! Halo2 baseline (from `zkie-core::chips::dot_general`): a matmul is `m*n`
//! dot-product regions, each costing `k + 184` rows (184 = 64 + 60 + 60 range
//! checks). GKR overhead here is the partial-MLE folding (`m*k + k*n` field
//! ops) plus the `O(k)` sum-check.

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use {zkie_core::common::matmul, zkie_core::common::mle};

fn main() {
    let (m, k, n) = (512usize, 512usize, 512usize);
    let mut rng = XorShift64::new(0xfeed);

    let a: Vec<Goldilocks> = (0..m * k).map(|_| rng.field()).collect();
    let b: Vec<Goldilocks> = (0..k * n).map(|_| rng.field()).collect();
    let c = dense(&a, &b, m, k, n);
    let at = transpose(&a, m, k);

    let mut t = zkie_core::common::transcript::Transcript::new(b"zkie-bench-matmul");
    t.absorb_u64(0xfeed);
    let u: Vec<Goldilocks> = t.field_vec(m.trailing_zeros() as usize);
    let v: Vec<Goldilocks> = t.field_vec(n.trailing_zeros() as usize);
    
    let t0 = std::time::Instant::now();
    let (proof, challenges) = matmul::prove(&at, &b, &c, m, k, n, &u, &v, &mut t);
    let prove_secs = t0.elapsed().as_secs_f64();

    let mut f_point = u.clone();
    f_point.extend_from_slice(&challenges);
    let f_eval = mle::eval(&at, &f_point);
    let mut h_point = v.clone();
    h_point.extend_from_slice(&challenges);
    let h_eval = mle::eval(&b, &h_point);
    let mut tv = zkie_core::common::transcript::Transcript::new(b"zkie-bench-matmul");
    tv.absorb_u64(0xfeed);
    for _ in 0..m.trailing_zeros() { let _ = tv.field(); }
    for _ in 0..n.trailing_zeros() { let _ = tv.field(); }
    let (ok, _) = matmul::verify(&proof, k.trailing_zeros() as usize, &mut tv);
    assert!(ok);

    let gkr_field_ops = m * k + k * n + 6 * k;
    let halo2_rows = m * n * (k + 184);

    println!("matmul m={m} k={k} n={n}");
    println!("gkr_proof_field_ops={gkr_field_ops}");
    println!("halo2_rows={halo2_rows}");
    println!(
        "halo2_over_gkr_ratio={:.1}",
        halo2_rows as f64 / gkr_field_ops as f64
    );
    println!("gkr_prove_wall_secs={prove_secs:.4}");
}

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
