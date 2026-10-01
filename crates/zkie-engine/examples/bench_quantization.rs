//! Measure projection proving time as a function of the quantization bit width
//! (`shift`). Fewer bits shrink the logUp range-check table (2^shift entries),
//! which is the dominant lookup cost; this quantifies the 16-bit -> 12-bit lever.

use std::time::Instant;

use zkie_core::common::field::{PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_ops::projection::{prove_projection, verify_projection};
use zkie_ops::par::mm_par;

fn round_div(a: i64, b: i64) -> i64 {
    let q = a.div_euclid(b);
    let r = a.rem_euclid(b);
    if r * 2 >= b { q + 1 } else { q }
}

fn main() {
    let mut rng = XorShift64::new(0xBEEF);
    let (m, k, n) = (16usize, 256usize, 256usize);
    let x: Vec<_> = (0..m * k).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let w: Vec<_> = (0..k * n).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let bias: Vec<_> = (0..m * n).map(|_| from_i64((rng.next_u64() % 20) as i64 - 10)).collect();

    println!("projection {}x{}x{} (m x k x n)", m, k, n);
    for &shift in &[8u32, 12, 16] {
        let h = mm_par(&x, &w, m, k, n, 64);
        let out: Vec<_> = (0..m * n)
            .map(|ij| from_i64(round_div(to_i64(h[ij]), 1i64 << shift) + to_i64(bias[ij])))
            .collect();
        let rem: Vec<_> = (0..m * n)
            .map(|ij| {
                from_i64(
                    to_i64(h[ij]) - (to_i64(out[ij]) - to_i64(bias[ij])) * (1i64 << shift)
                        + (1i64 << (shift - 1)),
                )
            })
            .collect();

        let t0 = Instant::now();
        let proof = prove_projection(&x, &w, &bias, &out, &rem, m, k, n, shift, &mut rng);
        let dt = t0.elapsed();
        assert!(verify_projection(&proof, &x, &w, &bias, &out, &rem, m, k, n, shift));
        println!("shift={:>2} ({}-bit): {:>8.1?}  (range table = 2^{})", shift, shift, dt, shift);
    }
}
