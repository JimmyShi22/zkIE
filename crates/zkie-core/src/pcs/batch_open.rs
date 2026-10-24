//! Multi-point batch opening.
//!
//! Reduces N claims `f_i(r_i) = y_i` at *different* points to one degree-2
//! sumcheck over `B(X) = sum_i gamma^i * eq(r_i, X) * f_i(X)` plus a single
//! same-point opening (`open_batch_multi`) at a fresh point, so the FRI folding
//! is paid once instead of N times.

use crate::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::common::mle;
use crate::common::sumcheck;
use crate::pcs::whir::Whir;

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
    batch: &crate::pcs::committed::BatchCtx,
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

    let (open_proof, evals) = batch.whir.open_batch_multi(
        batch.prover_data.clone(),
        &batch.protocol,
        batch.num_tables,
        &challenges,
    );
    let evals_ok = batch
        .whir
        .verify_batch_multi(
            &batch.commitment,
            &open_proof,
            &batch.protocol,
            batch.num_tables,
            &challenges,
        )
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
    batch: &crate::pcs::committed::BatchCtx,
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
    let proof =
        sumcheck::prove_sum_of_products(&[w.clone()], &[tensor.to_vec()], claimed_sum, &challenges);

    let (open_proof, evals) = batch.whir.open_batch_multi(
        batch.prover_data.clone(),
        &batch.protocol,
        batch.num_tables,
        &challenges,
    );
    let evals_ok = batch
        .whir
        .verify_batch_multi(
            &batch.commitment,
            &open_proof,
            &batch.protocol,
            batch.num_tables,
            &challenges,
        )
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

/// A single batch-opening proof: the sum-of-products sumcheck, the one FRI
/// opening proof (all tables at the fresh point), and the opened evaluations.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct BatchOpeningProof {
    pub sumcheck: sumcheck::SumcheckProofBatch,
    pub open: crate::pcs::whir::Proof,
    pub evals: Vec<Goldilocks>,
}

/// Prover side of a multi-point batch opening on an already-committed batch.
/// `gamma` and `challenges` are sampled by the caller (and carried in the
/// surrounding proof), so prove/verify stay in lockstep. Opens all `num_tables`
/// at the single fresh point `challenges` with one FRI proof.
#[allow(clippy::too_many_arguments)]
pub fn prove_batch_open_committed(
    batch: &crate::pcs::committed::BatchCtx,
    table_idxs: &[usize],
    tensors: &[Vec<Goldilocks>],
    points: &[Vec<Goldilocks>],
    claimed: &[Goldilocks],
) -> Result<BatchOpeningProof, String> {
    prove_batch_open_committed_with_data(
        &batch.whir,
        &batch.commitment,
        &batch.protocol,
        batch.num_tables,
        batch.prover_data.clone(),
        table_idxs,
        tensors,
        points,
        claimed,
    )
}

/// Owned variant of `prove_batch_open_committed` for one-shot batch openings.
/// Consumes the WHIR prover data instead of cloning its Merkle tree.
pub fn prove_batch_open_committed_owned(
    batch: crate::pcs::committed::BatchCtx,
    table_idxs: &[usize],
    tensors: &[Vec<Goldilocks>],
    points: &[Vec<Goldilocks>],
    claimed: &[Goldilocks],
) -> Result<BatchOpeningProof, String> {
    let crate::pcs::committed::BatchCtx {
        commitment,
        prover_data,
        protocol,
        whir,
        num_tables,
    } = batch;
    prove_batch_open_committed_with_data(
        &whir,
        &commitment,
        &protocol,
        num_tables,
        prover_data,
        table_idxs,
        tensors,
        points,
        claimed,
    )
}

fn prove_batch_open_committed_with_data(
    whir: &Whir,
    commitment: &crate::pcs::whir::Commitment,
    protocol: &crate::pcs::whir::OpeningProtocol,
    num_tables: usize,
    prover_data: crate::pcs::whir::ProverData,
    table_idxs: &[usize],
    tensors: &[Vec<Goldilocks>],
    points: &[Vec<Goldilocks>],
    claimed: &[Goldilocks],
) -> Result<BatchOpeningProof, String> {
    let n = tensors.len();
    if table_idxs.len() != n || points.len() != n || claimed.len() != n {
        return Err("batch-open length mismatch".to_owned());
    }
    let d = tensors[0].len().trailing_zeros() as usize;

    // Fiat-Shamir: gamma + fresh point r bound to commitment + points + claimed.
    let mut t = crate::common::transcript::Transcript::new(b"zkie-batch-open");
    t.absorb(&bincode::serialize(commitment).map_err(|e| e.to_string())?);
    for (pt, c) in points.iter().zip(claimed.iter()) {
        t.absorb_fields(pt);
        t.absorb_field(*c);
    }
    let gamma = t.field();
    let challenges = t.field_vec(d);

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

    let sumcheck = sumcheck::prove_sum_of_products(&weights, tensors, claimed_sum, &challenges);
    let (open, evals) = whir.open_batch_multi(prover_data, protocol, num_tables, &challenges);
    Ok(BatchOpeningProof {
        sumcheck,
        open,
        evals,
    })
}

/// Verifier side of a multi-point batch opening. Verifies the one FRI proof
/// against the commitment root and the sum-of-products sumcheck, reading no
/// committed tensor data. `table_idxs[i]` is the batch table index of claim i.
#[allow(clippy::too_many_arguments)]
pub fn verify_batch_open_committed(
    whir: &Whir,
    commitment: &crate::pcs::whir::Commitment,
    protocol: &crate::pcs::whir::OpeningProtocol,
    num_tables: usize,
    table_idxs: &[usize],
    points: &[Vec<Goldilocks>],
    claimed: &[Goldilocks],
    proof: &BatchOpeningProof,
) -> bool {
    let n = table_idxs.len();
    if n != points.len() || n != claimed.len() || n == 0 {
        return false;
    }
    let d = points[0].len();

    // Re-derive gamma + fresh point r (same Fiat-Shamir state as the prover).
    let mut t = crate::common::transcript::Transcript::new(b"zkie-batch-open");
    if bincode::serialize(commitment).is_err() {
        return false;
    }
    t.absorb(&bincode::serialize(commitment).unwrap());
    for (pt, c) in points.iter().zip(claimed.iter()) {
        t.absorb_fields(pt);
        t.absorb_field(*c);
    }
    let gamma = t.field();
    let challenges = t.field_vec(d);

    let mut claimed_sum = Goldilocks::ZERO;
    let mut gamma_pow = Goldilocks::ONE;
    for i in 0..n {
        claimed_sum = claimed_sum + gamma_pow * claimed[i];
        gamma_pow = gamma_pow * gamma;
    }

    let evals = match whir.verify_batch_multi(commitment, &proof.open, protocol, num_tables, &challenges) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if evals != proof.evals {
        return false;
    }

    let mut expected = Goldilocks::ZERO;
    let mut gamma_pow = Goldilocks::ONE;
    for i in 0..n {
        let eq = mle::eq_poly(&points[i], &challenges);
        expected = expected + gamma_pow * eq * evals[table_idxs[i]];
        gamma_pow = gamma_pow * gamma;
    }

    sumcheck::verify_sum_of_products(&proof.sumcheck, claimed_sum, &challenges, expected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::field::XorShift64;
    use crate::common::mle;

    #[test]
    fn batch_open_completeness_and_soundness() {
        let mut rng = XorShift64::new(7);
        let d = 8usize;
        let n = 8usize;
        let tensors: Vec<Vec<Goldilocks>> = (0..n)
            .map(|_| (0..(1usize << d)).map(|_| rng.field()).collect())
            .collect();
        let points: Vec<Vec<Goldilocks>> = (0..n)
            .map(|_| (0..d).map(|_| rng.field()).collect())
            .collect();
        let claimed: Vec<Goldilocks> = (0..n).map(|i| mle::eval(&tensors[i], &points[i])).collect();

        let mut r1 = XorShift64::new(99);
        assert!(
            batch_open(&tensors, &points, &claimed, &mut r1),
            "valid batch must pass"
        );

        let mut bad = claimed.clone();
        bad[0] = bad[0] + Goldilocks::ONE;
        let mut r2 = XorShift64::new(99);
        assert!(
            !batch_open(&tensors, &points, &bad, &mut r2),
            "corrupt claim must fail"
        );
    }

    #[test]
    fn open_table_multi_point_completeness() {
        use crate::pcs::committed::BatchCtx;
        use crate::pcs::whir::Whir;
        let mut rng = XorShift64::new(42);
        let d = 8usize;
        let tensor: Vec<Goldilocks> = (0..(1usize << d)).map(|_| rng.field()).collect();
        let whir = Whir::new_testing(d);
        let refs: Vec<&[Goldilocks]> = vec![tensor.as_slice()];
        let (commitment, prover_data, protocol, batch_whir) = whir.commit_batch(&refs);
        let batch = BatchCtx {
            commitment,
            prover_data,
            protocol,
            whir: batch_whir,
            num_tables: 1,
        };
        let points: Vec<Vec<Goldilocks>> = (0..3)
            .map(|_| (0..d).map(|_| rng.field()).collect())
            .collect();
        let mut r2 = XorShift64::new(99);
        let evals =
            open_table_multi_point(&batch, 0, &tensor, &points, &mut r2).expect("valid open");
        for (i, p) in points.iter().enumerate() {
            assert_eq!(evals[i], mle::eval(&tensor, p));
        }
    }

    #[test]
    fn split_batch_open_roundtrip() {
        use crate::pcs::committed::BatchCtx;
        let mut rng = XorShift64::new(11);
        let d = 8usize;
        let n = 4usize;
        let tensors: Vec<Vec<Goldilocks>> = (0..n)
            .map(|_| (0..(1usize << d)).map(|_| rng.field()).collect())
            .collect();
        let points: Vec<Vec<Goldilocks>> = (0..n)
            .map(|_| (0..d).map(|_| rng.field()).collect())
            .collect();
        let claimed: Vec<Goldilocks> = (0..n).map(|i| mle::eval(&tensors[i], &points[i])).collect();

        let whir = Whir::new_testing(d);
        let refs: Vec<&[Goldilocks]> = tensors.iter().map(|t| t.as_slice()).collect();
        let (commitment, prover_data, protocol, batch_whir) = whir.commit_batch(&refs);
        let batch = BatchCtx {
            commitment,
            prover_data,
            protocol,
            whir: batch_whir,
            num_tables: n,
        };

        let table_idxs: Vec<usize> = (0..n).collect();
        let proof = prove_batch_open_committed(&batch, &table_idxs, &tensors, &points, &claimed).unwrap();
        assert!(verify_batch_open_committed(
            &batch.whir,
            &batch.commitment,
            &batch.protocol,
            n,
            &table_idxs,
            &points,
            &claimed,
            &proof
        ));

        let mut bad = claimed.clone();
        bad[0] = bad[0] + Goldilocks::ONE;
        assert!(!verify_batch_open_committed(
            &batch.whir,
            &batch.commitment,
            &batch.protocol,
            n,
            &table_idxs,
            &points,
            &bad,
            &proof
        ));
    }
}
