//! Row-parallel CPU helpers for the "layout / parallelism" knob. Uses rayon's
//! global work-stealing pool so nested parallelism (parallel proof across layers
//! calling parallel matmul inside) does not oversubscribe.

use crate::field::{Goldilocks, PrimeCharacteristicRing};
use crate::fixed_point::{from_i64, to_i64};
use rayon::prelude::*;

/// Row-parallel matrix multiplication over the `Goldilocks` field. This is the
/// general, always-correct path (used by tests and any non-fixed-point tensor).
pub fn mm_par(
    a: &[Goldilocks],
    b: &[Goldilocks],
    m: usize,
    k: usize,
    n: usize,
    _threads: usize,
) -> Vec<Goldilocks> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut c = vec![Goldilocks::ZERO; m * n];
    c.par_chunks_mut(n).enumerate().for_each(|(i, cij)| {
        for j in 0..n {
            let mut acc = Goldilocks::ZERO;
            for kk in 0..k {
                acc = acc + a[i * k + kk] * b[kk * n + j];
            }
            cij[j] = acc;
        }
    });
    c
}

/// Fixed-point matmul for witness generation. The inputs carry `2^shift`-scaled
/// integers embedded in the field; their products (up to `2^30` for 16-bit
/// fixed point) and length-`k` dot products (up to `2^42` for the models here)
/// stay inside `i64`, so the accumulation can run in plain signed integer
/// arithmetic instead of field multiplication. Exact for the inference witness
/// value range. If any input is not a small fixed-point value, it transparently
/// falls back to the general field matmul, so this is always correct.
pub fn mm_par_fixed(
    a: &[Goldilocks],
    b: &[Goldilocks],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Goldilocks> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);

    // The i64 accumulation is exact only if the worst-case dot product fits in
    // i64: max|a| * max|b| * k < 2^63. Otherwise fall back to the field path.
    let max_a = a.iter().map(|&v| to_i64(v).unsigned_abs()).max().unwrap_or(0) as i128;
    let max_b = b.iter().map(|&v| to_i64(v).unsigned_abs()).max().unwrap_or(0) as i128;
    if max_a * max_b * (k as i128) >= (1i128 << 63) {
        eprintln!("[mm_par_fixed] overflow fallback to field (m={} k={} n={}, max_a={} max_b={})", m, k, n, max_a, max_b);
        return mm_par(a, b, m, k, n, 64);
    }

    let ai: Vec<i64> = a.iter().map(|&v| to_i64(v)).collect();
    let bi: Vec<i64> = b.iter().map(|&v| to_i64(v)).collect();
    let mut c = vec![0i64; m * n];
    c.par_chunks_mut(n).enumerate().for_each(|(i, cij)| {
        // "ikj" order: stream each row of B contiguously instead of striding
        // across it (the naive ijk order thrashes the cache for large n).
        for j in 0..n {
            cij[j] = 0;
        }
        for kk in 0..k {
            let a_val = ai[i * k + kk];
            let brow = &bi[kk * n..(kk + 1) * n];
            for j in 0..n {
                cij[j] += a_val * brow[j];
            }
        }
    });
    c.into_iter().map(from_i64).collect()
}
