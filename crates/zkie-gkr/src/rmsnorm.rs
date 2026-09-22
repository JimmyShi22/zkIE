//! Quantized RMSNorm: a single `rsqrt` lookup plus plain field arithmetic.
//!
//! `y_i = x_i / sqrt(mean(x^2) + eps) * w_i`. The only non-arithmetic piece is
//! `1/sqrt`, a scalar table lookup; the square, sum, mean and scale are field
//! operations the verifier recomputes. TimeFM's `input_layernorm` is exactly
//! this shape.

use crate::field::{Field, Goldilocks, PrimeCharacteristicRing};
use crate::lookup::{self, LookupProof};

pub struct RmsNormProof {
    /// `rsqrt` lookup proving `r == rsqrt_table[s_index]`.
    pub lookup: LookupProof,
    /// `mean(x^2) + eps`, the value whose rsqrt is looked up.
    pub s: Goldilocks,
    /// `rsqrt(s)`.
    pub r: Goldilocks,
}

/// Run RMSNorm over `x` and produce a proof that the output `y` and the
/// `rsqrt` value are consistent with the committed input/weight and the table.
pub fn rms_norm(
    x: &[Goldilocks],
    weight: &[Goldilocks],
    eps: Goldilocks,
    s_index: u32,
    rsqrt_table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
) -> (Vec<Goldilocks>, RmsNormProof) {
    assert_eq!(x.len(), weight.len());
    assert!(!x.is_empty());

    let sum_x2 = x.iter().fold(Goldilocks::ZERO, |acc, &v| acc + v * v);
    let n_inv = Goldilocks::from_u64(x.len() as u64).inverse();
    let s = sum_x2 * n_inv + eps;
    let r = rsqrt_table[s_index as usize];
    let y: Vec<Goldilocks> = x
        .iter()
        .zip(weight)
        .map(|(&xi, &wi)| xi * r * wi)
        .collect();
    let lookup = lookup::prove(&[s_index], &[r], rsqrt_table, alpha, beta);
    (y, RmsNormProof { lookup, s, r })
}

/// Verify that `s` matches the input, `r` matches the table, and `y` is the
/// correctly scaled output.
pub fn verify(
    x: &[Goldilocks],
    weight: &[Goldilocks],
    y: &[Goldilocks],
    eps: Goldilocks,
    s_index: u32,
    rsqrt_table: &[Goldilocks],
    proof: &RmsNormProof,
) -> bool {
    if x.len() != y.len() || x.len() != weight.len() || x.is_empty() {
        return false;
    }
    let sum_x2 = x.iter().fold(Goldilocks::ZERO, |acc, &v| acc + v * v);
    let n_inv = Goldilocks::from_u64(x.len() as u64).inverse();
    if sum_x2 * n_inv + eps != proof.s {
        return false;
    }
    if proof.r != rsqrt_table[s_index as usize] {
        return false;
    }
    if !lookup::verify(&proof.lookup) {
        return false;
    }
    x.iter()
        .zip(weight)
        .zip(y)
        .all(|((&xi, &wi), &yi)| xi * proof.r * wi == yi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::XorShift64;

    #[test]
    fn rms_norm_roundtrip() {
        let mut rng = XorShift64::new(31);
        let n = 64usize;
        let x: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let weight: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let eps = Goldilocks::from_u64(1);
        let table_size = 256usize;
        let table: Vec<Goldilocks> = (0..table_size).map(|_| rng.field()).collect();
        let s_index = (rng.next_u64() % table_size as u64) as u32;

        let (y, proof) = rms_norm(&x, &weight, eps, s_index, &table, rng.field(), rng.field());
        assert!(verify(&x, &weight, &y, eps, s_index, &table, &proof));

        let mut bad = y.clone();
        bad[0] = bad[0] + Goldilocks::ONE;
        assert!(!verify(&x, &weight, &bad, eps, s_index, &table, &proof));
    }
}
