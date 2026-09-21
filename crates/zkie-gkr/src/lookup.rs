//! LogUp (log-derivative) lookup argument.
//!
//! Proves that `y_i = table[x_i]` for every `i`. The pairs `(x_i, y_i)` are
//! batched into single keys with a random `beta`, then the log-derivative
//! identity `sum_i 1/(alpha + key_i) == sum_j m_j/(alpha + tkey_j)` is checked
//! at a random `alpha`. This is the same "sum of rational functions" shape our
//! sum-check already handles, so it slots into the GKR pipeline as another
//! parallel reduction rather than a sorting/permutation argument.

use crate::field::F31;

#[derive(Clone, Debug)]
pub struct LookupProof {
    /// `sum_i 1/(alpha + (x_i + beta*y_i))`.
    pub lhs: F31,
    /// `sum_j m_j / (alpha + (j + beta*table[j]))`.
    pub rhs: F31,
    pub alpha: F31,
    pub beta: F31,
}

/// Build the lookup proof for `outputs[i] == table[indices[i]]`.
pub fn prove(
    indices: &[u32],
    outputs: &[F31],
    table: &[F31],
    alpha: F31,
    beta: F31,
) -> LookupProof {
    assert_eq!(indices.len(), outputs.len());

    // Multiplicities: m[j] = number of lookups hitting table index j.
    let mut m = vec![F31::ZERO; table.len()];
    for &i in indices {
        assert!((i as usize) < table.len(), "lookup index out of table range");
        m[i as usize] = m[i as usize] + F31::ONE;
    }

    let lhs = indices.iter().zip(outputs).fold(F31::ZERO, |acc, (&i, &y)| {
        let key = F31::new(i) + beta * y;
        acc + (alpha + key).inv()
    });

    let rhs = table.iter().enumerate().fold(F31::ZERO, |acc, (j, &t)| {
        let tkey = F31::new(j as u32) + beta * t;
        acc + m[j] * (alpha + tkey).inv()
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
        let table: Vec<F31> = (0..n).map(|_| rng.field()).collect();
        let indices: Vec<u32> = (0..n).map(|_| (rng.next_u64() % n as u64) as u32).collect();
        let outputs: Vec<F31> = indices.iter().map(|&i| table[i as usize]).collect();

        let alpha = rng.field();
        let beta = rng.field();
        let proof = prove(&indices, &outputs, &table, alpha, beta);
        assert!(verify(&proof));

        // Corrupt one output -> the batched key leaves the table, so the logUp
        // identity fails with high probability over (alpha, beta).
        let mut bad = outputs.clone();
        bad[0] = bad[0] + F31::ONE;
        let bad_proof = prove(&indices, &bad, &table, alpha, beta);
        assert!(!verify(&bad_proof));
    }
}
