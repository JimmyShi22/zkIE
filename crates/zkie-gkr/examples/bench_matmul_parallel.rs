//! The forward pass is the bottleneck (~372s for GPT-2 512), and it uses a
//! single-threaded matmul. Demonstrate the win from row-parallel matmul.

use std::time::Instant;
use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_gkr::fixed_point::from_i64;

fn mm_naive(
    a: &[Goldilocks],
    b: &[Goldilocks],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Goldilocks> {
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

fn mm_parallel(
    a: &[Goldilocks],
    b: &[Goldilocks],
    m: usize,
    k: usize,
    n: usize,
    threads: usize,
) -> Vec<Goldilocks> {
    let mut c = vec![Goldilocks::ZERO; m * n];
    let chunk = (m + threads - 1) / threads;
    std::thread::scope(|s| {
        for (t, rows) in c.chunks_mut(chunk * n).enumerate() {
            s.spawn(move || {
                let start_row = t * chunk;
                for (li, cij) in rows.chunks_mut(n).enumerate() {
                    let i = start_row + li;
                    if i >= m {
                        break;
                    }
                    for j in 0..n {
                        let mut acc = Goldilocks::ZERO;
                        for kk in 0..k {
                            acc = acc + a[i * k + kk] * b[kk * n + j];
                        }
                        cij[j] = acc;
                    }
                }
            });
        }
    });
    c
}

fn main() {
    // GPT-2 FFN matmul: 512 x 1024 x 4096 (the biggest matmul in the layer).
    let (m, k, n) = (512usize, 1024usize, 4096usize);
    let mut rng = XorShift64::new(0x0DE1);
    let a: Vec<Goldilocks> = (0..m * k).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let b: Vec<Goldilocks> = (0..k * n).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();

    let t0 = Instant::now();
    let c1 = mm_naive(&a, &b, m, k, n);
    let naive = t0.elapsed();

    for threads in [8usize, 32, 64] {
        let t1 = Instant::now();
        let c2 = mm_parallel(&a, &b, m, k, n, threads);
        let par = t1.elapsed();
        assert_eq!(c1, c2);
        println!(
            "matmul {}x{}x{}: naive={:.2}s parallel({}t)={:.2}s speedup={:.2}x",
            m,
            k,
            n,
            naive.as_secs_f64(),
            threads,
            par.as_secs_f64(),
            naive.as_secs_f64() / par.as_secs_f64()
        );
    }
}
