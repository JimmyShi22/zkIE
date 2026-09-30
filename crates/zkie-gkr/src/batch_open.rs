//! Multi-point batch opening.
//!
//! Reduces N claims `f_i(r_i) = y_i` at *different* points to one degree-2
//! sumcheck over `B(X) = sum_i gamma^i * eq(r_i, X) * f_i(X)` plus a single
//! same-point opening (`open_batch_multi`) at a fresh point, so the FRI folding
//! is paid once instead of N times.

use crate::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::mle;
use crate::sumcheck;
use crate::whir::Whir;

/// Prove + verify the batch opening. Returns true iff all `f_i(r_i) == y_i`.
#[allow(clippy::too_many_arguments)]
pub fn batch_open(
    tensors: &[Vec<Goldilocks>],
    points: &[Vec<Goldilocks>],
    claimed: &[Goldilocks],
    rng: &mut XorShift64,
) -> bool {
    let n = tensors.len();
    assert!(n > 0, "empty batch");
    assert_eq!(points.len(), n);
    assert_eq!(claimed.len(), n);
    let d = tensors[0].len().trailing_zeros() as usize;
    for t in tensors {
        assert_eq!(t.len(), 1 << d, "all tables must share one arity");
    }

    // 1. Batch-commit all tables into one witness.
    let whir = Whir::new_testing(d);
    let refs: Vec<&[Goldilocks]> = tensors.iter().map(|t| t.as_slice()).collect();
    let (commitment, prover_data, protocol, batch_whir) = whir.commit_batch(&refs);

    // 2. Sample the batching challenge gamma.
    let gamma = rng.field();

    // 3. Weights w_i = gamma^i * eq(r_i, .) and the claimed weighted sum.
    let mut weights: Vec<Vec<Goldilocks>> = Vec::with_capacity(n);
    let mut gamma_pow = Goldilocks::ONE;
    let mut claimed_sum = Goldilocks::ZERO;
    for i in 0..n {
        let eq = mle::eq_evals(&points[i]);
        let w: Vec<Goldilocks> = eq.iter().map(|&x| gamma_pow * x).collect();
        weights.push(w);
        claimed_sum = claimed_sum + gamma_pow * claimed[i];
        gamma_pow = gamma_pow * gamma;
    }

    // 4. Sum-check over B = sum_i w_i * f_i.
    let challenges: Vec<Goldilocks> = (0..d).map(|_| rng.field()).collect();
    let proof = sumcheck::prove_sum_of_products(&weights, tensors, claimed_sum, &challenges);

    // 5. Open every f_i at the single fresh point r = challenges (one FRI proof).
    let (open_proof, evals) = batch_whir.open_batch_multi(prover_data, &protocol, n, &challenges);
    let evals_ok = batch_whir
        .verify_batch_multi(&commitment, &open_proof, &protocol, n, &challenges)
        .map(|v| v == evals)
        .unwrap_or(false);

    // 6. Independent recomputation of B(r) = sum_i gamma^i * eq(r_i, r) * f_i(r).
    let mut expected = Goldilocks::ZERO;
    let mut gamma_pow = Goldilocks::ONE;
    for i in 0..n {
        let eq = mle::eq_poly(&points[i], &challenges);
        expected = expected + gamma_pow * eq * evals[i];
        gamma_pow = gamma_pow * gamma;
    }

    evals_ok && sumcheck::verify_sum_of_products(&proof, claimed_sum, &challenges, expected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::XorShift64;
    use crate::mle;

    #[test]
    fn batch_open_completeness_and_soundness() {
        let mut rng = XorShift64::new(7);
        let d = 8usize;
        let n = 8usize;
        let tensors: Vec<Vec<Goldilocks>> =
            (0..n).map(|_| (0..(1usize << d)).map(|_| rng.field()).collect()).collect();
        let points: Vec<Vec<Goldilocks>> =
            (0..n).map(|_| (0..d).map(|_| rng.field()).collect()).collect();
        let claimed: Vec<Goldilocks> =
            (0..n).map(|i| mle::eval(&tensors[i], &points[i])).collect();

        let mut r1 = XorShift64::new(99);
        assert!(batch_open(&tensors, &points, &claimed, &mut r1), "valid batch must pass");

        let mut bad = claimed.clone();
        bad[0] = bad[0] + Goldilocks::ONE;
        let mut r2 = XorShift64::new(99);
        assert!(!batch_open(&tensors, &points, &bad, &mut r2), "corrupt claim must fail");
    }
}