//! Row-parallel CPU helpers for the "layout / parallelism" knob. Uses rayon's
//! global work-stealing pool so nested parallelism (parallel proof across layers
//! calling parallel matmul inside) does not oversubscribe.

use crate::field::{Goldilocks, PrimeCharacteristicRing};
use rayon::prelude::*;

/// Row-parallel matrix multiplication over Goldilocks.
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
