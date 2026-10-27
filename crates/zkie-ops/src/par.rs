//! Row-parallel CPU helpers for the "layout / parallelism" knob. Uses rayon's
//! global work-stealing pool so nested parallelism (parallel proof across layers
//! calling parallel matmul inside) does not oversubscribe.

use rayon::prelude::*;
use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing};
use zkie_core::common::fixed_point::{from_i128, to_i128};

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
/// integers embedded in the field. Products and length-`k` dot products are
/// accumulated in `i128` (exact for the whole inference witness range, including
/// Q32 residual streams whose i64 partial sums could otherwise overflow), then
/// reduced modulo the field order so the result matches what the sumcheck
/// verifier computes over the field.
pub fn mm_par_fixed(
    a: &[Goldilocks],
    b: &[Goldilocks],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Goldilocks> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);

    let ai: Vec<i128> = a.iter().map(|&v| to_i128(v)).collect();
    let bi: Vec<i128> = b.iter().map(|&v| to_i128(v)).collect();
    let mut c = vec![0i128; m * n];
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
    c.into_iter().map(from_i128).collect()
}
