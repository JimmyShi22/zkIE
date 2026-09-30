//! Row-parallel CPU helpers for the "layout / parallelism" knob. The forward
//! matmul (witness generation) is the bottleneck and is embarrassingly parallel
//! across rows.

use crate::field::{Goldilocks, PrimeCharacteristicRing};

/// Row-parallel matrix multiplication over Goldilocks.
pub fn mm_par(
    a: &[Goldilocks],
    b: &[Goldilocks],
    m: usize,
    k: usize,
    n: usize,
    threads: usize,
) -> Vec<Goldilocks> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
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
