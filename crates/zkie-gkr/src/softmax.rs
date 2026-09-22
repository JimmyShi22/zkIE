//! Quantized softmax: exp lookup + reduction + division.
//!
//! `y_i = exp(x_i) / sum_j exp(x_j)`. The only non-arithmetic piece is `exp`,
//! which is a table lookup; the sum and division are plain field operations
//! (the division is a single batch inversion in a real circuit).

use crate::field::{Field, Goldilocks, PrimeCharacteristicRing};
use crate::lookup::{self, LookupProof};

pub struct SoftmaxProof {
    pub lookup: LookupProof,
    /// `sum_i e_i`.
    pub sum: Goldilocks,
}

/// Run quantized softmax over table indices and produce a lookup proof that
/// every `exp` value really came from the table.
pub fn softmax(
    indices: &[u32],
    table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
) -> (Vec<Goldilocks>, SoftmaxProof) {
    let e: Vec<Goldilocks> = indices.iter().map(|&i| table[i as usize]).collect();
    let sum = e.iter().fold(Goldilocks::ZERO, |acc, &v| acc + v);
    let inv = sum.inverse();
    let y: Vec<Goldilocks> = e.iter().map(|&v| v * inv).collect();
    let lookup = lookup::prove(indices, &e, table, alpha, beta);
    (y, SoftmaxProof { lookup, sum })
}

/// Verify that the outputs sum to one and that the exp lookup is consistent.
pub fn verify(indices: &[u32], table: &[Goldilocks], y: &[Goldilocks], proof: &SoftmaxProof) -> bool {
    // Outputs must sum to 1 (the softmax normalization).
    if y.iter().fold(Goldilocks::ZERO, |acc, &v| acc + v) != Goldilocks::ONE {
        return false;
    }
    // Recover e_i = y_i * sum, then check the lookup e_i == table[x_i].
    let e: Vec<Goldilocks> = y.iter().map(|&v| v * proof.sum).collect();
    lookup::verify(&proof.lookup)
        && e.iter().zip(indices).all(|(&ev, &i)| ev == table[i as usize])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::XorShift64;

    #[test]
    fn softmax_roundtrip() {
        let mut rng = XorShift64::new(13);
        let n = 128usize;
        let table: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let indices: Vec<u32> = (0..n).map(|_| (rng.next_u64() % n as u64) as u32).collect();
        let (y, proof) = softmax(&indices, &table, rng.field(), rng.field());
        assert!(verify(&indices, &table, &y, &proof));

        // Corrupt one output and confirm verification fails.
        let mut bad = y.clone();
        bad[0] = bad[0] + Goldilocks::ONE;
        assert!(!verify(&indices, &table, &bad, &proof));
    }
}
