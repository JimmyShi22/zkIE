//! Quantized GELU activation as a single LogUp lookup.
//!
//! `y_i = GELU(x_i)`. GELU is the only non-arithmetic piece of the FFN block,
//! and it is exactly a table lookup (`x_i` quantized to a table index, `y_i`
//! the corresponding precomputed GELU value), so it needs no extra field
//! arithmetic — the same shape as `softmax`'s `exp`, minus the normalization.

use crate::field::Goldilocks;
use crate::lookup::{self, LookupProof};

pub struct GeluProof {
    pub lookup: LookupProof,
}

/// Run quantized GELU over table indices and produce a lookup proof that every
/// output really came from the table.
pub fn gelu(
    indices: &[u32],
    table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
) -> (Vec<Goldilocks>, GeluProof) {
    let y: Vec<Goldilocks> = indices.iter().map(|&i| table[i as usize]).collect();
    let lookup = lookup::prove(indices, &y, table, alpha, beta);
    (y, GeluProof { lookup })
}

/// Verify that every output is exactly the table value at its index.
pub fn verify(
    indices: &[u32],
    table: &[Goldilocks],
    y: &[Goldilocks],
    proof: &GeluProof,
) -> bool {
    lookup::verify(&proof.lookup)
        && indices
            .iter()
            .zip(y)
            .all(|(&i, &v)| table[i as usize] == v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::PrimeCharacteristicRing;
    use crate::field::XorShift64;

    #[test]
    fn gelu_roundtrip() {
        let mut rng = XorShift64::new(30);
        let n = 256usize;
        let table: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let indices: Vec<u32> = (0..n).map(|_| (rng.next_u64() % n as u64) as u32).collect();
        let (y, proof) = gelu(&indices, &table, rng.field(), rng.field());
        assert!(verify(&indices, &table, &y, &proof));

        let mut bad = y.clone();
        bad[0] = bad[0] + Goldilocks::ONE;
        assert!(!verify(&indices, &table, &bad, &proof));
    }
}
