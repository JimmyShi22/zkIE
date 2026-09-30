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

/// General multi-point reduction on an *already committed* batch. `table_idxs[i]`
/// maps the i-th tensor to its table index in the batch (duplicates allowed, so
/// the same table can be opened at several points). Pays one FRI proof.
pub fn batch_open_committed(
    batch: &crate::committed::BatchCtx,
    table_idxs: &[usize],
    tensors: &[Vec<Goldilocks>],
    points: &[Vec<Goldilocks>],
    claimed: &[Goldilocks],
    rng: &mut XorShift64,
) -> bool {
    let n = tensors.len();
    assert_eq!(table_idxs.len(), n);
    assert_eq!(points.len(), n);
    assert_eq!(claimed.len(), n);
    let d = tensors[0].len().trailing_zeros() as usize;

    let gamma = rng.field();
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

    let challenges: Vec<Goldilocks> = (0..d).map(|_| rng.field()).collect();
    let proof = sumcheck::prove_sum_of_products(&weights, tensors, claimed_sum, &challenges);

    let (open_proof, evals) = batch.whir.open_batch_multi(batch.prover_data.clone(), &batch.protocol, batch.num_tables, &challenges);
    let evals_ok = batch
        .whir
        .verify_batch_multi(&batch.commitment, &open_proof, &batch.protocol, batch.num_tables, &challenges)
        .map(|v| v == evals)
        .unwrap_or(false);

    let mut expected = Goldilocks::ZERO;
    let mut gamma_pow = Goldilocks::ONE;
    for i in 0..n {
        let eq = mle::eq_poly(&points[i], &challenges);
        expected = expected + gamma_pow * eq * evals[table_idxs[i]];
        gamma_pow = gamma_pow * gamma;
    }
    evals_ok && sumcheck::verify_sum_of_products(&proof, claimed_sum, &challenges, expected)
}

/// Open a single committed table at multiple points via the standard
/// multi-point-to-single-point reduction, paying one FRI proof instead of one
/// per point. Returns the verified evaluations `f(points[i])` in order.
pub fn open_table_multi_point(
    batch: &crate::committed::BatchCtx,
    table_idx: usize,
    tensor: &[Goldilocks],
    points: &[Vec<Goldilocks>],
    rng: &mut XorShift64,
) -> Option<Vec<Goldilocks>> {
    let n = points.len();
    assert!(n > 0, "empty points");
    let d = tensor.len().trailing_zeros() as usize;
    assert_eq!(tensor.len(), 1 << d, "table must be a power of two");

    let gamma = rng.field();
    let mut w: Vec<Goldilocks> = vec![Goldilocks::ZERO; 1 << d];
    let mut claimed: Vec<Goldilocks> = Vec::with_capacity(n);
    let mut gamma_pow = Goldilocks::ONE;
    let mut claimed_sum = Goldilocks::ZERO;
    for p in points {
        let eq = mle::eq_evals(p);
        for (wx, &ex) in w.iter_mut().zip(&eq) {
            *wx = *wx + gamma_pow * ex;
        }
        let y = mle::eval(tensor, p);
        claimed.push(y);
        claimed_sum = claimed_sum + gamma_pow * y;
        gamma_pow = gamma_pow * gamma;
    }

    let challenges: Vec<Goldilocks> = (0..d).map(|_| rng.field()).collect();
    let proof = sumcheck::prove_sum_of_products(&[w.clone()], &[tensor.to_vec()], claimed_sum, &challenges);

    let (open_proof, evals) = batch.whir.open_batch_multi(batch.prover_data.clone(), &batch.protocol, batch.num_tables, &challenges);
    let evals_ok = batch
        .whir
        .verify_batch_multi(&batch.commitment, &open_proof, &batch.protocol, batch.num_tables, &challenges)
        .map(|v| v == evals)
        .unwrap_or(false);

    let w_r = mle::eval(&w, &challenges);
    let expected = evals[table_idx] * w_r;
    if evals_ok && sumcheck::verify_sum_of_products(&proof, claimed_sum, &challenges, expected) {
        Some(claimed)
    } else {
        None
    }
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

    #[test]
    fn open_table_multi_point_completeness() {
        use crate::committed::BatchCtx;
        use crate::whir::Whir;
        let mut rng = XorShift64::new(42);
        let d = 8usize;
        let tensor: Vec<Goldilocks> = (0..(1usize << d)).map(|_| rng.field()).collect();
        let whir = Whir::new_testing(d);
        let refs: Vec<&[Goldilocks]> = vec![tensor.as_slice()];
        let (commitment, prover_data, protocol, batch_whir) = whir.commit_batch(&refs);
        let batch = BatchCtx { commitment, prover_data, protocol, whir: batch_whir, num_tables: 1 };
        let points: Vec<Vec<Goldilocks>> = (0..3).map(|_| (0..d).map(|_| rng.field()).collect()).collect();
        let mut r2 = XorShift64::new(99);
        let evals = open_table_multi_point(&batch, 0, &tensor, &points, &mut r2).expect("valid open");
        for (i, p) in points.iter().enumerate() {
            assert_eq!(evals[i], mle::eval(&tensor, p));
        }
    }

}
