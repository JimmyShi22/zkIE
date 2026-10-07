//! Claim-driven shard: linear `Add` (m = 16, k = 8) + fixed N = 8 direct
//! lookup combined under ONE shared padded output/table commitment.
//!
//! The Add op's padded output (128 = 2^7 base entries, arity 7) IS the
//! DirectN8 padded table (8 logical values, then a zero tail — the witness
//! contract enforced by `prove`). The shared tensor is committed ONCE as a
//! base tensor: `extension_linear::prove_add_shared_out` proves
//! `out = a + b` against that root (the statement's `root_out` IS the shared
//! root), and `extension_direct_lookup::PrecommittedTable` proves
//! `out[i] = table[idx[i]]` against the SAME root as the table's re
//! component (with a zero imaginary component), also never re-committed.
//!
//! Verifier = statement + proof only (`whir`, statement, proof): it checks
//! the linear relation, the lookup relation, and byte-equal shared-root
//! equality across both statements — no witness, no Store, no raw tensors,
//! no forward recomputation, no `open` calls. There is no extra Fiat–Shamir
//! cross-binding beyond the two relations' own transcripts; the shared root
//! is the binding (the linear transcript is seeded with it, and both
//! relations open it).

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, PrimeField64};
use zkie_core::pcs::whir::Whir;
use crate::extension_direct_lookup::{
    self, LookupProof, PrecommittedTable, Statement as LookupStatement,
};
use crate::extension_linear::{
    self, LinearOpKind, LinearProof, LinearStatement, PrecommittedTensor,
};

/// Add output shape: 16 x 8 = 128 = 2^7 (arity 7, matching the DirectN8
/// padded domain).
pub const SHARD_M: usize = 16;
pub const SHARD_K: usize = 8;
/// Direct lookup logical size (must match `extension_direct_lookup::N`).
pub const SHARD_N: usize = extension_direct_lookup::N;

/// Public statement: the linear statement (whose `root_out` is the shared
/// root) and the lookup statement (whose `tbl.re` is the SAME root).
#[derive(Clone, Debug)]
pub struct ClaimDrivenShardStatement {
    pub linear: LinearStatement,
    pub lookup: LookupStatement,
}

/// The transported proof: the linear proof and the direct lookup proof.
#[derive(Clone)]
pub struct ClaimDrivenShardProof {
    pub linear: LinearProof,
    pub lookup: LookupProof,
}

/// Prove the combined shard. The witness is `a`/`b` (128 base entries each;
/// their sum is the padded table — logical values 0..8, then a ZERO tail,
/// else `None`), the lookup indices (8, each < 8), and the lookup outputs
/// (8). The out/table consistency is NOT checked here (the verifier checks
/// the relation); shapes and the zero tail are. Returns `None` for malformed
/// inputs (never panics).
pub fn prove(
    whir: &Whir,
    a: &[Goldilocks],
    b: &[Goldilocks],
    indices: &[u8],
    out: &[u64],
) -> Option<(ClaimDrivenShardStatement, ClaimDrivenShardProof)> {
    if whir.num_variables() != SHARD_M.trailing_zeros() as usize + SHARD_K.trailing_zeros() as usize
    {
        return None;
    }
    let n = SHARD_M * SHARD_K;
    if a.len() != n || b.len() != n || indices.len() != SHARD_N || out.len() != SHARD_N {
        return None;
    }
    // The shared padded tensor: the Add output = the padded lookup table.
    let table_padded: Vec<Goldilocks> = (0..n).map(|i| a[i] + b[i]).collect();
    for i in SHARD_N..n {
        if table_padded[i] != Goldilocks::ZERO {
            return None; // the padded table's tail must be zero
        }
    }
    // Commit the shared tensor ONCE (base); both relations use this root.
    let (shared_root, shared_pd, shared_proto) = whir.commit(&table_padded);

    let (lin_stmt, lin_proof) = extension_linear::prove_add_shared_out(
        whir,
        SHARD_M,
        SHARD_K,
        a,
        b,
        &table_padded,
        &PrecommittedTensor {
            root: shared_root.clone(),
            prover_data: shared_pd.clone(),
            protocol: shared_proto,
        },
    )?;

    let pre_tbl = PrecommittedTable::from_committed_base(whir, shared_root.clone(), shared_pd);
    let (lk_stmt, lk_proof) = pre_tbl.prove_lookup_with_table(whir, indices, out)?;

    Some((
        ClaimDrivenShardStatement {
            linear: lin_stmt,
            lookup: lk_stmt,
        },
        ClaimDrivenShardProof {
            linear: lin_proof,
            lookup: lk_proof,
        },
    ))
}

/// Verify the combined shard from statement and proof only. Checks the
/// shared-root equality (linear output root == lookup table re root,
/// byte-equal — not a prover-supplied pairing), then the linear relation,
/// then the lookup relation. Malformed shapes return `false` (never panic).
pub fn verify(
    whir: &Whir,
    stmt: &ClaimDrivenShardStatement,
    proof: &ClaimDrivenShardProof,
) -> bool {
    if whir.num_variables() != SHARD_M.trailing_zeros() as usize + SHARD_K.trailing_zeros() as usize
    {
        return false;
    }
    // The shard relation is linear ADD over the shared output — any other
    // linear kind is out of contract even if its own proof is valid.
    if stmt.linear.kind != LinearOpKind::Add {
        return false;
    }
    if stmt.linear.m != SHARD_M || stmt.linear.k != SHARD_K {
        return false;
    }
    // Shared-root equality: the linear output and the lookup table re
    // component must be the identical commitment.
    if stmt.linear.root_out != stmt.lookup.tbl.re {
        return false;
    }
    extension_linear::verify(whir, &stmt.linear, &proof.linear)
        && extension_direct_lookup::verify(whir, &stmt.lookup, &proof.lookup)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_core::common::field::XorShift64;

    const ARITY: usize = 7;

    fn witness(
        rng: &mut XorShift64,
    ) -> (Vec<Goldilocks>, Vec<Goldilocks>, Vec<u8>, Vec<u64>) {
        let n = SHARD_M * SHARD_K;
        let mut a = vec![Goldilocks::ZERO; n];
        let mut b = vec![Goldilocks::ZERO; n];
        for i in 0..SHARD_N {
            a[i] = rng.field();
            b[i] = rng.field();
        }
        // Zero tail: the padded table has exactly 8 logical values.
        for i in SHARD_N..n {
            a[i] = rng.field();
            b[i] = Goldilocks::ZERO - a[i];
        }
        let table: Vec<u64> =
            (0..SHARD_N).map(|i| (a[i] + b[i]).as_canonical_u64()).collect();
        let indices = vec![3u8, 0, 7, 2, 5, 1, 6, 4];
        let out: Vec<u64> = indices.iter().map(|&i| table[i as usize]).collect();
        (a, b, indices, out)
    }

    fn fixture() -> (
        Whir,
        ClaimDrivenShardStatement,
        ClaimDrivenShardProof,
        Vec<Goldilocks>,
        Vec<Goldilocks>,
        Vec<u8>,
        Vec<u64>,
    ) {
        let whir = Whir::new_target(ARITY, 90, 0).expect("valid arity");
        let mut rng = XorShift64::new(0x5AD);
        let (a, b, indices, out) = witness(&mut rng);
        let (s, p) = prove(&whir, &a, &b, &indices, &out).expect("honest prove");
        (whir, s, p, a, b, indices, out)
    }

    /// Honest roundtrip: after the raw witness vectors are DROPPED, the
    /// verifier accepts from statement + proof only and never opens.
    #[test]
    fn honest_roundtrip_after_witness_dropped() {
        let (whir, s, p, a, b, indices, out) = fixture();
        drop(a);
        drop(b);
        drop(indices);
        drop(out);
        let before = whir.open_stats();
        assert!(verify(&whir, &s, &p));
        assert_eq!(whir.open_stats(), before, "verifier must never open");
    }

    /// A forged output (out[0] != table[idx[0]]) is honestly committed but
    /// the combined verifier rejects it via the lookup relation.
    #[test]
    fn forged_output_rejected() {
        let whir = Whir::new_target(ARITY, 90, 0).expect("valid arity");
        let mut rng = XorShift64::new(0x5AD);
        let (a, b, indices, mut out) = witness(&mut rng);
        out[0] += 1;
        let (s, p) = prove(&whir, &a, &b, &indices, &out).expect("honest commitments");
        assert!(!verify(&whir, &s, &p));
    }

    /// Two SEPARATELY valid shard artifacts (distinct witnesses -> distinct
    /// shared roots), each verifying standalone; a composed artifact pairing
    /// the linear relation of A with the lookup relation of B must be
    /// rejected by the shared-root equality check — the pairing is
    /// byte-structural, not prover-supplied.
    #[test]
    fn mismatched_shared_root_rejected() {
        let whir = Whir::new_target(ARITY, 90, 0).expect("valid arity");
        let mut rng_a = XorShift64::new(0x5AD);
        let (a0, b0, indices0, out0) = witness(&mut rng_a);
        let (s0, p0) = prove(&whir, &a0, &b0, &indices0, &out0).expect("artifact A");
        let mut rng_b = XorShift64::new(0xBEE);
        let (a1, b1, indices1, out1) = witness(&mut rng_b);
        let (s1, p1) = prove(&whir, &a1, &b1, &indices1, &out1).expect("artifact B");
        // Each artifact verifies standalone.
        assert!(verify(&whir, &s0, &p0));
        assert!(verify(&whir, &s1, &p1));
        // The witnesses produced distinct shared roots, each internally
        // consistent.
        assert_ne!(s0.linear.root_out, s1.linear.root_out);
        assert_eq!(s0.linear.root_out, s0.lookup.tbl.re);
        assert_eq!(s1.linear.root_out, s1.lookup.tbl.re);
        // Composed: linear from A, lookup from B -> root equality rejects.
        let composed_stmt = ClaimDrivenShardStatement {
            linear: s0.linear.clone(),
            lookup: s1.lookup.clone(),
        };
        let composed_proof = ClaimDrivenShardProof {
            linear: p0.linear.clone(),
            lookup: p1.lookup.clone(),
        };
        assert!(!verify(&whir, &composed_stmt, &composed_proof));
    }

    /// A standalone-valid NON-Add linear proof (Scale, factor 1 / shift 0)
    /// whose output IS the shared lookup table — the tensor is committed
    /// once and both statements reference the identical root — paired with a
    /// valid lookup artifact on that SAME root: `scale_stmt.root_out ==
    /// lookup_stmt.tbl.re` is asserted directly, so the combined verifier
    /// rejects SOLELY on the Add-kind gate.
    #[test]
    fn non_add_linear_paired_with_valid_lookup_rejected() {
        let whir = Whir::new_target(ARITY, 90, 0).expect("valid arity");
        let mut rng = XorShift64::new(0x5AD);
        // A padded table: 8 logical values, zero tail.
        let n = SHARD_M * SHARD_K;
        let mut table_padded = vec![Goldilocks::ZERO; n];
        for i in 0..SHARD_N {
            table_padded[i] = rng.field();
        }
        let table_vals: Vec<u64> =
            (0..SHARD_N).map(|i| table_padded[i].as_canonical_u64()).collect();
        let indices = vec![3u8, 0, 7, 2, 5, 1, 6, 4];
        let out: Vec<u64> = indices.iter().map(|&i| table_vals[i as usize]).collect();

        // Commit the shared tensor once.
        let (shared_root, shared_pd, _) = whir.commit(&table_padded);

        // Scale with factor 1, shift 0: out = x. Feeding x = table makes the
        // scale OUTPUT exactly the padded table, and the deterministic commit
        // of the same values yields the identical root.
        let (scale_stmt, scale_proof) = extension_linear::prove(
            &whir,
            LinearOpKind::Scale { factor: 1, shift: 0 },
            SHARD_M,
            SHARD_K,
            &table_padded,
            None,
            &table_padded,
        )
        .expect("valid scale proof");
        assert!(extension_linear::verify(&whir, &scale_stmt, &scale_proof));

        // The lookup shares the SAME committed tensor as its table.
        let pre_tbl =
            PrecommittedTable::from_committed_base(&whir, shared_root.clone(), shared_pd);
        let (lk_stmt, lk_proof) = pre_tbl
            .prove_lookup_with_table(&whir, &indices, &out)
            .expect("valid lookup proof");
        assert!(extension_direct_lookup::verify(&whir, &lk_stmt, &lk_proof));

        // The shared root is byte-identical across both statements — the
        // only remaining gate for the combined verifier is the Add kind.
        assert_eq!(scale_stmt.root_out, shared_root);
        assert_eq!(scale_stmt.root_out, lk_stmt.tbl.re);

        let combined_stmt =
            ClaimDrivenShardStatement { linear: scale_stmt, lookup: lk_stmt };
        let combined_proof =
            ClaimDrivenShardProof { linear: scale_proof, lookup: lk_proof };
        assert!(!verify(&whir, &combined_stmt, &combined_proof));
    }

    /// A tampered linear opening claim must be rejected.
    #[test]
    fn tampered_linear_claim_rejected() {
        let (whir, s, mut p, ..) = fixture();
        p.linear.open_a.1 = p.linear.open_a.1 + zkie_core::common::field::EF::from(Goldilocks::ONE);
        assert!(!verify(&whir, &s, &p));
    }

    /// A tampered lookup eval claim must be rejected.
    #[test]
    fn tampered_lookup_claim_rejected() {
        let (whir, s, mut p, ..) = fixture();
        p.lookup.b0.re.evals[0] =
            p.lookup.b0.re.evals[0] + zkie_core::common::field::EF::from(Goldilocks::ONE);
        assert!(!verify(&whir, &s, &p));
    }

    /// The witness boundary: a nonzero padded-tail (entries 8..128) is
    /// rejected by `prove` (`None`), and malformed shapes too.
    #[test]
    fn nonzero_tail_and_malformed_shapes_rejected() {
        let whir = Whir::new_target(ARITY, 90, 0).expect("valid arity");
        let mut rng = XorShift64::new(0x5AD);
        let (a, b, indices, out) = witness(&mut rng);
        assert!(prove(&whir, &a, &b, &indices, &out).is_some());
        // Nonzero tail.
        let mut bad_b = b.clone();
        bad_b[8] = bad_b[8] + Goldilocks::ONE;
        assert!(prove(&whir, &a, &bad_b, &indices, &out).is_none());
        // Wrong shapes.
        assert!(prove(&whir, &a[..127], &b, &indices, &out).is_none());
        assert!(prove(&whir, &a, &b, &indices[..7], &out).is_none());
        assert!(prove(&whir, &a, &b, &indices, &out[..7]).is_none());
        // Wrong whir arity.
        let whir6 = Whir::new_target(6, 90, 0).expect("valid arity");
        assert!(prove(&whir6, &a, &b, &indices, &out).is_none());
    }
}
