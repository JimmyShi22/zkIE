//! KZG-style multilinear PC over BN254 (ark-poly-commit 0.5
//! `multilinear_pc::MultilinearPC`), benchmark-only.
//!
//! SRS note: `MultilinearPC::setup` is run with OS randomness, locally, for
//! THIS benchmark only. The returned parameters contain GROUP ENCODINGS of
//! trapdoor-derived values (`g^{t_i}` in `g_mask`, eq-based powers of g and
//! h), not the raw trapdoor — and dropping them does not constitute a secure
//! erasure ceremony. This is a benchmark-only setup, NOT production
//! parameters. Its contents are never printed.
//!
//! NOTE: this PCS evaluates MLEs over the BN254 scalar field. The
//! "integer-valued" tables here are integers embedded into that field; they
//! are NOT the same field elements as in the Goldilocks WHIR run, and no
//! equality between evaluations across the two fields is claimed. See README.
//!
//! Batching: the two tables are committed and opened SEPARATELY (one proof per
//! table; this crate's multilinear PC has no multi-open API).

use std::time::Instant;

use ark_bn254::{Bn254, Fr};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_poly_commit::multilinear_pc::MultilinearPC;
use ark_serialize::CanonicalSerialize;
use ark_std::UniformRand;
use rand08::rngs::OsRng;

/// One measured run of the KZG multilinear PC on `num_tables` integer-valued
/// tables of `2^n` entries, over `rounds` rounds of fresh random challenge
/// points. Setup (SRS generation) is timed and reported separately. Proof and
/// commitment byte counts are SUMMED over both tables.
pub struct KzgReport {
    pub n: usize,
    pub setup: std::time::Duration,
    pub commit: std::time::Duration,
    pub open: std::time::Duration,
    pub verify: std::time::Duration,
    pub proof_bytes: usize,
    pub commitment_bytes: usize,
    pub srs_bytes: usize,
    pub wrong_value_rejected: bool,
    pub wrong_commitment_rejected: bool,
}

/// Raw integer values of the shared generator for table `table_index`
/// (2^n entries in 0..99). Used by tests to prove both schemes share inputs.
pub fn table_values(n: usize, table_index: usize) -> Vec<u64> {
    crate::tables::values(n, table_index)
}

/// Integer-valued table of 2^n entries (shared generator, embedded into Fr).
/// The same integers are used on the Goldilocks side, but as elements of
/// DIFFERENT fields.
fn int_table(n: usize, table_index: usize) -> Vec<Fr> {
    table_values(n, table_index).into_iter().map(Fr::from).collect()
}

pub fn run(n: usize, num_tables: usize, rounds: usize) -> KzgReport {
    assert!(n > 0, "constant polynomial not supported");
    assert_eq!(num_tables, 2, "benchmark is fixed at 2 tables");
    let mut rng = OsRng;

    // ---- setup (SRS), timed separately. Trapdoor is discarded on drop. ----
    let t0 = Instant::now();
    let pp = MultilinearPC::<Bn254>::setup(n, &mut rng);
    let setup = t0.elapsed();
    let srs_bytes = pp.compressed_size();

    let (ck, vk) = MultilinearPC::trim(&pp, n);

    let tables: Vec<DenseMultilinearExtension<Fr>> = (0..num_tables)
        .map(|i| DenseMultilinearExtension::from_evaluations_vec(n, int_table(n, i)))
        .collect();

    // ---- commit (deterministic: this PCS has no hiding randomness) ----
    let t0 = Instant::now();
    let commitments: Vec<_> = tables
        .iter()
        .map(|t| MultilinearPC::commit(&ck, t))
        .collect();
    let commit = t0.elapsed();
    let commitment_bytes: usize = commitments.iter().map(|c| c.compressed_size()).sum();

    // ---- per-round: fresh random challenge vector, open + verify ----
    let mut open_elapsed = std::time::Duration::ZERO;
    let mut verify_elapsed = std::time::Duration::ZERO;
    let mut proof_bytes = 0usize;
    for _ in 0..rounds {
        let points: Vec<Vec<Fr>> = (0..num_tables)
            .map(|_| (0..n).map(|_| Fr::rand(&mut rng)).collect())
            .collect();
        // Expected values computed OUTSIDE the verify timer (the MLE fold is
        // O(2^n) field operations and belongs to the caller, not to PCS verify).
        let values: Vec<Fr> = tables
            .iter()
            .zip(points.iter())
            .map(|(t, p)| eval(t, p))
            .collect();

        let o0 = Instant::now();
        let proofs: Vec<_> = tables
            .iter()
            .zip(points.iter())
            .map(|(t, p)| MultilinearPC::open(&ck, t, p))
            .collect();
        open_elapsed += o0.elapsed();

        let v0 = Instant::now();
        for (((p, pf), v), c) in points
            .iter()
            .zip(proofs.iter())
            .zip(values.iter())
            .zip(commitments.iter())
        {
            assert!(MultilinearPC::check(&vk, c, p, *v, pf),
                "honest opening must verify");
        }
        verify_elapsed += v0.elapsed();
        proof_bytes = proofs.iter().map(|p| p.compressed_size()).sum();
    }
    let open = open_elapsed.div_f64(rounds as f64);
    let verify = verify_elapsed.div_f64(rounds as f64);

    // ---- rejection tests ----
    let points: Vec<Vec<Fr>> = (0..num_tables)
        .map(|_| (0..n).map(|_| Fr::rand(&mut rng)).collect())
        .collect();
    let proof = MultilinearPC::open(&ck, &tables[0], &points[0]);
    // wrong value: correct point, one-off value
    let wrong_value = eval(&tables[0], &points[0]) + Fr::from(1u64);
    let wrong_value_rejected =
        !MultilinearPC::check(&vk, &commitments[0], &points[0], wrong_value, &proof);
    // wrong commitment: proof for table 0 checked against table 1's commitment
    let other = MultilinearPC::commit(&ck, &tables[1]);
    let wrong_commitment_rejected =
        !MultilinearPC::check(&vk, &other, &points[0], eval(&tables[0], &points[0]), &proof);

    KzgReport {
        n,
        setup,
        commit,
        open,
        verify,
        proof_bytes,
        commitment_bytes,
        srs_bytes,
        wrong_value_rejected,
        wrong_commitment_rejected,
    }
}

/// MLE evaluation over the BN254 scalar field (plain Horner-style fold).
fn eval(poly: &DenseMultilinearExtension<Fr>, point: &[Fr]) -> Fr {
    let evals = poly.to_evaluations();
    let mut buf = evals.clone();
    let mut size = buf.len();
    for &p in point {
        let half = size / 2;
        for i in 0..half {
            buf[i] = buf[2 * i] + p * (buf[2 * i + 1] - buf[2 * i]);
        }
        size = half;
    }
    buf[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kzg_positive_and_rejection_small() {
        let r = run(6, 2, 3);
        assert!(r.setup.as_secs_f64() >= 0.0);
        assert!(r.commit.as_secs_f64() >= 0.0);
        assert!(r.proof_bytes > 0);
        assert!(r.wrong_value_rejected, "wrong value must be rejected");
        assert!(r.wrong_commitment_rejected, "wrong commitment must be rejected");
    }
}
