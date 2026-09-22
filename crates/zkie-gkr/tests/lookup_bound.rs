//! Bind a LogUp lookup to WHIR commitments.
//!
//! The bare `lookup::verify` only checks `lhs == rhs` for prover-supplied `lhs`,
//! which a cheating prover can fake. Here the verifier opens the *committed*
//! index and output columns and recomputes `lhs` itself, so the lookup identity
//! is enforced against the commitments rather than a claimed number. This is
//! O(N) opening for the PoC; the polylog form is the LogUp-sumcheck follow-up.

use zkie_gkr::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_gkr::lookup;
use zkie_gkr::whir::Whir;

#[test]
fn lookup_identity_holds_against_committed_values() {
    let mut rng = XorShift64::new(0x999);
    let n = 64usize; // 2^6 (Whir folding factor is 5, so num_vars must be >= 5)
    let table_size = 32usize;
    let table: Vec<Goldilocks> = (0..table_size).map(|_| rng.field()).collect();
    let indices: Vec<u32> = (0..n).map(|_| (rng.next_u64() % table_size as u64) as u32).collect();
    let outputs: Vec<Goldilocks> = indices.iter().map(|&i| table[i as usize]).collect();

    let whir = Whir::new_testing(6);
    let idx_vals: Vec<Goldilocks> = indices
        .iter()
        .map(|&i| Goldilocks::from_u64(i as u64))
        .collect();
    let (idx_c, idx_pd, idx_proto) = whir.commit(&idx_vals);
    let (out_c, out_pd, out_proto) = whir.commit(&outputs);

    let alpha = rng.field();
    let beta = rng.field();
    let proof = lookup::prove(&indices, &outputs, &table, alpha, beta);

    // Verifier opens the committed columns at every hypercube point and
    // recomputes the LogUp left-hand side from those opened values.
    let d = 6usize;
    let mut lhs = Goldilocks::ZERO;
    for i in 0..n {
        let point: Vec<Goldilocks> = (0..d)
            .map(|b| Goldilocks::from_bool((i >> b) & 1 == 1))
            .collect();
        let (idx_open, idx_v) = whir.open(idx_pd.clone(), &idx_proto, &point);
        let (out_open, out_v) = whir.open(out_pd.clone(), &out_proto, &point);
        // Verify each opening against its commitment, then use the bound value.
        assert_eq!(whir.verify(&idx_c, &idx_open, &idx_proto, &point).unwrap(), idx_v);
        assert_eq!(whir.verify(&out_c, &out_open, &out_proto, &point).unwrap(), out_v);
        assert_eq!(idx_v, idx_vals[i]);
        assert_eq!(out_v, outputs[i]);
        let key = idx_v + beta * out_v;
        lhs = lhs + (alpha + key).inverse();
    }
    assert_eq!(lhs, proof.rhs);
}
