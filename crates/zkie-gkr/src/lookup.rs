//! LogUp (log-derivative) lookup argument.
//!
//! Proves that `y_i = table[x_i]` for every `i`. The pairs `(x_i, y_i)` are
//! batched into single keys with a random `beta`, then the log-derivative
//! identity `sum_i 1/(alpha + key_i) == sum_j m_j/(alpha + tkey_j)` is checked
//! at a random `alpha`. The table-side sum runs over the *distinct* indices
//! hit by the lookups (not the whole table), so a single-element lookup is
//! O(1) and a batch lookup is O(number of distinct indices).

use std::collections::HashMap;

use crate::field::{Field, Goldilocks, PrimeCharacteristicRing};

#[derive(Clone, Debug)]
pub struct LookupProof {
    /// `sum_i 1/(alpha + (x_i + beta*y_i))`.
    pub lhs: Goldilocks,
    /// `sum_j m_j / (alpha + (j + beta*table[j]))`.
    pub rhs: Goldilocks,
    pub alpha: Goldilocks,
    pub beta: Goldilocks,
}

/// Build the lookup proof for `outputs[i] == table[indices[i]]`.
pub fn prove(
    indices: &[u32],
    outputs: &[Goldilocks],
    table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
) -> LookupProof {
    assert_eq!(indices.len(), outputs.len());
    // Multiplicities over distinct indices only.
    let mut m: HashMap<u32, Goldilocks> = HashMap::new();
    for &i in indices {
        assert!((i as usize) < table.len(), "lookup index out of table range");
        let e = m.entry(i).or_insert(Goldilocks::ZERO);
        *e = *e + Goldilocks::ONE;
    }
    let lhs = indices.iter().zip(outputs).fold(Goldilocks::ZERO, |acc, (&i, &y)| {
        let key = Goldilocks::from_u64(i as u64) + beta * y;
        acc + (alpha + key).inverse()
    });
    let rhs = m.iter().fold(Goldilocks::ZERO, |acc, (&j, &mult)| {
        let tkey = Goldilocks::from_u64(j as u64) + beta * table[j as usize];
        acc + mult * (alpha + tkey).inverse()
    });
    LookupProof { lhs, rhs, alpha, beta }
}

pub fn verify(proof: &LookupProof) -> bool {
    proof.lhs == proof.rhs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::XorShift64;

    #[test]
    fn lookup_completeness_and_soundness() {
        let mut rng = XorShift64::new(12);
        let n = 256usize;
        let table: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let indices: Vec<u32> = (0..n).map(|_| (rng.next_u64() % n as u64) as u32).collect();
        let outputs: Vec<Goldilocks> = indices.iter().map(|&i| table[i as usize]).collect();
        let alpha = rng.field();
        let beta = rng.field();
        let proof = prove(&indices, &outputs, &table, alpha, beta);
        assert!(verify(&proof));

        let mut bad = outputs.clone();
        bad[0] = bad[0] + Goldilocks::ONE;
        let bad_proof = prove(&indices, &bad, &table, alpha, beta);
        assert!(!verify(&bad_proof));
    }

    #[test]
    fn single_element_lookup_is_cheap() {
        let mut rng = XorShift64::new(13);
        let n = 4096usize;
        let table: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let idx = (rng.next_u64() % n as u64) as u32;
        let out = table[idx as usize];
        let proof = prove(&[idx], &[out], &table, rng.field(), rng.field());
        assert!(verify(&proof));
    }
}