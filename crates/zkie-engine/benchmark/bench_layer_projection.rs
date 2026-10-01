use std::time::Instant;
use zkie_core::common::field::{Goldilocks, XorShift64};
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_ops::projection::prove_projection;

fn main() {
    let (m, k, n) = (512usize, 1024usize, 1024usize);
    let shift = 16u32;
    let mut rng = XorShift64::new(0x1234);
    let x: Vec<Goldilocks> = (0..m * k).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let w: Vec<Goldilocks> = (0..k * n).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let bias: Vec<Goldilocks> = (0..m * n).map(|_| from_i64((rng.next_u64() % 20) as i64 - 10)).collect();
    let mut h = vec![from_i64(0); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = from_i64(0);
            for kk in 0..k {
                acc = acc + x[i * k + kk] * w[kk * n + j];
            }
            h[i * n + j] = acc;
        }
    }
    let div_round = |a: i64, b: i64| -> i64 { let q = a.div_euclid(b); let rr = a.rem_euclid(b); if rr * 2 >= b { q + 1 } else { q } };
    let out: Vec<Goldilocks> = (0..m * n).map(|ij| from_i64(div_round(to_i64(h[ij]), 1i64 << shift) + to_i64(bias[ij]))).collect();
    let rem_off: Vec<Goldilocks> = (0..m * n).map(|ij| from_i64(to_i64(h[ij]) - (to_i64(out[ij]) - to_i64(bias[ij])) * (1i64 << shift) + (1i64 << (shift - 1)))).collect();
    let t0 = Instant::now();
    let proof = prove_projection(&x, &w, &bias, &out, &rem_off, m, k, n, shift, &mut rng);
    let dt = t0.elapsed();
    println!("projection m={} k={} n={} shift={} time={:.2}s", m, k, n, shift, dt.as_secs_f64());
    let _ = proof;
}
