//! Multi-layer fractional-tree chain (stage B2a.2b) — TWO layers, N = 32 →
//! 16, built from the single-layer segment primitive of
//! `extension_lookup_fractional` (no duplicated proof logic).
//!
//! Structural chaining: ALL layers live in ONE combined tree tensor pair
//! (re/im), committed once:
//!
//!   [num_0 (32) | den_0 (32) | num_1 (16) | den_1 (16) | num_2 (8) | den_2 (8)]
//!
//! (112 entries, padded to arity 7 — above the WHIR folding-factor floor).
//! Layer 0's `next_num/next_den` sub-arrays ARE layer 1's `num_1/den_1` at
//! the SAME offsets of the SAME commitment — the cross-layer binding is
//! byte-structural by construction: there are no per-layer roots and no
//! prover-supplied root metadata to fake. Layer 1's `next` arrays (num_2/
//! den_2, length 8) terminate the chain; the chain stops here for this
//! prototype (documented floor: deeper chains extend the layout; the
//! committed tensor arity must stay ≥ the WHIR folding factor).
//!
//! Each layer runs the interactive Fiat–Shamir segment (eq point r, circuit
//! challenge c, per-round absorb-then-sample fold challenges z, terminal
//! openings at z, direct `eq(r,z)` claim) over the SHARED transcript, layer
//! 0 first — layer 1's challenges therefore depend on layer 0's messages.
//! The verifier reconstructs the identical schedule and checks every segment.
//!
//! Verifier = statement + proof only: no witness, no recomputation, no
//! `open` calls. Still NOT a lookup proof: no idx/table/multiplicity
//! relation.

use zkie_core::common::field::{BasedVectorSpace, EF, Goldilocks, PrimeCharacteristicRing, PrimeField64};
use zkie_core::common::transcript::TwoPhaseTranscript;
use zkie_core::pcs::whir::{Commitment, Whir};
use crate::extension_lookup_fractional::{
    fold_ef, prove_layer_segment, verify_layer_segment, LayerProof, LayerSpec, LayerStatement,
};

/// The canonical protocol label (its own — the multi-layer protocol).
pub const PROTOCOL: &str = "zkie/ext-fractional-chain/v1";

/// Combined layout arity (112 entries padded to 128).
pub const CHAIN_ARITY: usize = 7;
/// Number of chained layers.
pub const LAYERS: usize = 2;
/// Per-layer specs: (num_off, den_off, next_num_off, next_den_off, half).
pub const CHAIN_SPECS: [(usize, usize, usize, usize, usize); LAYERS] =
    [(0, 32, 64, 80, 16), (64, 80, 96, 104, 8)];

/// The transported proof: one `LayerProof` segment per layer, in order.
#[derive(Clone)]
pub struct ChainProof {
    pub layers: Vec<LayerProof>,
}

/// Build the honest two-layer tree arrays and the base commit tensors.
#[cfg(test)]
fn build_chain_arrays() -> (Vec<[Vec<EF>; 6]>, Vec<Goldilocks>, Vec<Goldilocks>) {
    build_chain_arrays_with_seed(0xCAFE_BABE_D00Du64)
}

/// Build the tree with an explicit witness seed (test-only; protocol
/// randomness comes entirely from the transcript).
fn build_chain_arrays_with_seed(
    seed: u64,
) -> (Vec<[Vec<EF>; 6]>, Vec<Goldilocks>, Vec<Goldilocks>) {
    // Deterministic witness generation (test-only; protocol randomness comes
    // entirely from the transcript).
    let mut x = seed;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        Goldilocks::from_u64(x % 100)
    };
    let num0: Vec<EF> = (0..32).map(|_| EF::from(next())).collect();
    let den0: Vec<EF> = (0..32).map(|_| EF::from(next())).collect();
    // Layer 0's halves.
    let n_low: Vec<EF> = num0[..16].to_vec();
    let n_high: Vec<EF> = num0[16..].to_vec();
    let d_low: Vec<EF> = den0[..16].to_vec();
    let d_high: Vec<EF> = den0[16..].to_vec();
    // Layer 1 = layer 0's fold.
    let num1: Vec<EF> = (0..16)
        .map(|s| n_low[s] * d_high[s] + n_high[s] * d_low[s])
        .collect();
    let den1: Vec<EF> = (0..16).map(|s| d_low[s] * d_high[s]).collect();
    // Layer 2 = layer 1's fold (terminates the chain).
    let num2: Vec<EF> = (0..8)
        .map(|s| num1[s] * den1[8 + s] + num1[8 + s] * den1[s])
        .collect();
    let den2: Vec<EF> = (0..8).map(|s| den1[s] * den1[8 + s]).collect();

    let seg0: [Vec<EF>; 6] = [
        num1.clone(),
        den1.clone(),
        n_low,
        n_high,
        d_low,
        d_high,
    ];
    let seg1: [Vec<EF>; 6] = [
        num2.clone(),
        den2.clone(),
        num1[..8].to_vec(),
        num1[8..].to_vec(),
        den1[..8].to_vec(),
        den1[8..].to_vec(),
    ];

    let size = 1 << CHAIN_ARITY;
    let mut re = vec![Goldilocks::ZERO; size];
    let mut im = vec![Goldilocks::ZERO; size];
    let layout: Vec<(usize, &[EF])> = vec![
        (0, &num0),
        (32, &den0),
        (64, &num1),
        (80, &den1),
        (96, &num2),
        (104, &den2),
    ];
    for (off, arr) in layout {
        for (s, &v) in arr.iter().enumerate() {
            let coeffs: &[Goldilocks] = v.as_basis_coefficients_slice();
            re[off + s] = coeffs[0];
            im[off + s] = coeffs[1];
        }
    }
    (vec![seg0, seg1], re, im)
}

fn spec_of(i: usize) -> LayerSpec {
    LayerSpec {
        arity: CHAIN_ARITY,
        num_off: CHAIN_SPECS[i].0,
        den_off: CHAIN_SPECS[i].1,
        next_num_off: CHAIN_SPECS[i].2,
        next_den_off: CHAIN_SPECS[i].3,
        half: CHAIN_SPECS[i].4,
    }
}

/// Prove the two-layer chain. Returns `(statement, proof)` or `None` for
/// malformed inputs.
pub fn prove_chain(whir: &Whir) -> Option<(LayerStatement, ChainProof)> {
    prove_chain_with_seed(whir, 0xCAFE_BABE_D00Du64)
}

/// Prove the chain with an explicit witness seed (test-only).
fn prove_chain_with_seed(whir: &Whir, seed: u64) -> Option<(LayerStatement, ChainProof)> {
    if whir.num_variables() != CHAIN_ARITY {
        return None;
    }
    let (segs, re, im) = build_chain_arrays_with_seed(seed);
    let (root_re, pd_re, proto_re) = whir.commit(&re);
    let (root_im, pd_im, proto_im) = whir.commit(&im);
    let stmt = LayerStatement {
        root_re: root_re.clone(),
        root_im: root_im.clone(),
    };

    let mut t = TwoPhaseTranscript::new(PROTOCOL, &[CHAIN_ARITY], &[&root_re, &root_im]);
    let (alpha, beta) = t.sample_derived_challenges();
    // The tree IS the statement — no derived commitments.
    t.absorb_derived_roots(&[]);

    let mut layers = Vec::with_capacity(LAYERS);
    for i in 0..LAYERS {
        let m = CHAIN_SPECS[i].4.trailing_zeros() as usize;
        let r = if i == 0 {
            // Layer 0's eq point starts with the derived challenges.
            let mut r = vec![alpha, beta];
            r.extend(t.sample_vec(m - 2));
            r
        } else {
            t.sample_vec(m)
        };
        let seg = prove_layer_segment(
            whir,
            &stmt,
            &pd_re,
            &pd_im,
            &proto_re,
            &mut t,
            &spec_of(i),
            &segs[i],
            &r,
        )?;
        layers.push(seg);
    }
    Some((stmt, ChainProof { layers }))
}

/// Verify the two-layer chain. Statement and proof only — no witness, no
/// recomputation, no `open` calls. Malformed shapes return `false` (never
/// panic).
pub fn verify_chain(whir: &Whir, stmt: &LayerStatement, proof: &ChainProof) -> bool {
    if whir.num_variables() != CHAIN_ARITY || proof.layers.len() != LAYERS {
        return false;
    }
    let mut t = TwoPhaseTranscript::new(PROTOCOL, &[CHAIN_ARITY], &[&stmt.root_re, &stmt.root_im]);
    let (alpha, beta) = t.sample_derived_challenges();
    t.absorb_derived_roots(&[]);
    for i in 0..LAYERS {
        let m = CHAIN_SPECS[i].4.trailing_zeros() as usize;
        let r = if i == 0 {
            let mut r = vec![alpha, beta];
            r.extend(t.sample_vec(m - 2));
            r
        } else {
            t.sample_vec(m)
        };
        if !verify_layer_segment(whir, stmt, &mut t, &spec_of(i), &proof.layers[i], &r) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_core::common::field::XorShift64;
    use zkie_core::pcs::whir::Whir;

    fn fixture() -> (Whir, LayerStatement, ChainProof) {
        let whir = Whir::new_target(CHAIN_ARITY, 90, 0).expect("valid arity");
        let (stmt, proof) = prove_chain(&whir).expect("honest prove");
        (whir, stmt, proof)
    }

    #[test]
    fn honest_chain_roundtrip() {
        let (whir, stmt, proof) = fixture();
        assert!(verify_chain(&whir, &stmt, &proof));
    }

    #[test]
    fn tampered_root_rejected() {
        let (whir, stmt, proof) = fixture();
        let mut rng = XorShift64::new(0xCC01);
        let fake: Vec<Goldilocks> = (0..(1 << CHAIN_ARITY)).map(|_| rng.field()).collect();
        let (fake_root, _, _) = whir.commit(&fake);
        let mut bad = stmt.clone();
        bad.root_re = fake_root;
        assert!(!verify_chain(&whir, &bad, &proof));
    }

    /// Splicing a VALID layer segment from another chain (another witness)
    /// must be rejected: its challenges/openings do not match this
    /// statement's transcript and commitment.
    #[test]
    fn spliced_layer_from_another_chain_rejected() {
        let (whir, stmt, proof) = fixture();
        // A DIFFERENT witness (different seed -> different tree/roots).
        let (_, other) = prove_chain_with_seed(&whir, 0xDEAD_BEEF_0001u64).unwrap();
        let mut bad = proof.clone();
        bad.layers[1] = other.layers[1].clone();
        assert!(!verify_chain(&whir, &stmt, &bad));
    }

    #[test]
    fn reordered_layer_proofs_rejected() {
        let (whir, stmt, mut proof) = fixture();
        proof.layers.swap(0, 1);
        assert!(!verify_chain(&whir, &stmt, &proof));
    }

    #[test]
    fn malformed_counts_rejected() {
        let (whir, stmt, proof) = fixture();
        assert!(verify_chain(&whir, &stmt, &proof));
        let mut bad = proof.clone();
        bad.layers.pop();
        assert!(!verify_chain(&whir, &stmt, &bad));
        let mut bad2 = proof.clone();
        bad2.layers[0].rounds.pop();
        assert!(!verify_chain(&whir, &stmt, &bad2));
        // Wrong arity whir.
        let whir2 = Whir::new_target(6, 90, 0).unwrap();
        assert!(prove_chain(&whir2).is_none());
    }

    #[test]
    fn verify_never_opens() {
        let (whir, stmt, proof) = fixture();
        let before = whir.open_stats();
        assert!(verify_chain(&whir, &stmt, &proof));
        assert_eq!(whir.open_stats(), before, "verifier must never open");
    }

    /// The honest chain's per-layer identities hold independently (reference
    /// evaluation of the fold relations).
    #[test]
    fn honest_segments_match_independent_folds() {
        let (whir, stmt, proof) = fixture();
        assert!(verify_chain(&whir, &stmt, &proof));
        let (segs, re, im) = build_chain_arrays();
        let read = |off: usize, s: usize| -> EF {
            let c0 = re[off + s];
            let c1 = im[off + s];
            crate::extension_lookup_fractional::open_pair(
                EF::from(c0),
                EF::from(c1),
            )
        };
        // Layer 0 fold: num1[s] == n0[s]*d0[16+s] + n0[16+s]*d0[s].
        for s in 0..16 {
            let n1 = read(64, s);
            let lhs = read(0, s) * read(32, 16 + s) + read(0, 16 + s) * read(32, s);
            assert_eq!(n1, lhs);
        }
        // Layer 0's segment terminal claims equal the independent folds at z.
        for i in 0..6 {
            let z = &proof.layers[0].z;
            let expected = fold_ef(&segs[0][i], z);
            assert_eq!(proof.layers[0].claimed[i], expected);
        }
    }
}
