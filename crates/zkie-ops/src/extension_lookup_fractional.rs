//! Fractional-tree LAYER identity primitive (stage B2a.2a) — NOT a lookup
//! proof. No index/table/output relation is claimed here; this module proves
//! exactly one thing: the eq-weighted LogUp fold identity of ONE tree layer
//! over an N = 32 segment:
//!
//!   next_num == num_low * den_high + num_high * den_low   (pointwise)
//!   next_den == den_low * den_high                        (pointwise)
//!
//! combined into the single circuit of `logup_gkr::build_terms` with a random
//! `c`:
//!   eq*next_num − eq*nL*dH − eq*nH*dL + c*eq*next_den − c*eq*dL*dH == 0.
//!
//! The six half-arrays (16 EF values each) live in TWO committed base-field
//! tensors (real and imaginary basis coefficients); every terminal claim is
//! a WHIR opening at the FINAL FOLD POINT `z`, recombined as `re + W·im`.
//!
//! INTERACTIVE FIAT–SHAMIR (soundness-critical): the eq point `r` is derived
//! once; then for EACH virtual round the prover produces the round
//! polynomial, the transcript absorbs EVERY coefficient, and only then is
//! the fresh fold challenge `z_i` sampled. The verifier follows the identical
//! sequence (rejecting any mismatch), so round polynomials cannot be
//! fabricated for known challenges without the FS grinding argument. The
//! terminal eq claim is the direct polynomial value `eq(r, z)` computed from
//! `(r, z)` by the verifier — no eq array is folded.
//!
//! Schedule (TwoPhaseTranscript, identical on both sides): initial roots
//! `[re, im]` → derived challenges `(alpha, beta)` (the first two coordinates
//! of `r`) → derived roots (none in this slice — the tree IS the statement)
//! → the remaining `r` coordinates and `c` → the interactive rounds → the
//! six claimed terminal evals (absorbed after the last fold).
//!
//! Verifier = statement + proof only: no witness, no recomputation, no
//! `open` calls. This is the algebra + schedule + FS integration; the full
//! multiset/multiplicity lookup reducer builds on this layer primitive.

use zkie_core::common::field::{
    BasedVectorSpace, EF, Goldilocks, PrimeCharacteristicRing, PrimeField64,
};
use zkie_core::common::sumcheck::{interpolate_f, virtual_round_pvals_f};
use zkie_core::common::transcript::TwoPhaseTranscript;
use zkie_core::pcs::whir::{Commitment, Point, Proof, Whir};

/// The canonical protocol label.
pub const PROTOCOL: &str = "zkie/ext-lookup-fractional/v1";

/// Segment size: N = 32 entries, half = 16 per sub-array.
pub const HALF: usize = 16;
/// Combined layout: 6 sub-arrays of HALF entries -> 96 -> padded arity 7.
pub const LAYOUT_ARITY: usize = 7;
/// Sub-array offsets in the combined tensor: next_num, next_den, num_low,
/// num_high, den_low, den_high.
pub const OFFSETS: [usize; 6] = [0, 16, 32, 48, 64, 80];
/// The layer's fold depth: log2(HALF) round challenges.
pub const ROUNDS: usize = 4;

/// Public statement: the two committed base-field tree tensors.
#[derive(Clone, Debug)]
pub struct LayerStatement {
    pub root_re: Commitment,
    pub root_im: Commitment,
}

/// One WHIR leaf pair (re/im openings) plus the recombined EF claim.
#[derive(Clone)]
pub struct LeafPair {
    pub open_re: (Proof, EF),
    pub open_im: (Proof, EF),
}

/// The transported proof.
#[derive(Clone)]
pub struct LayerProof {
    /// The eq point (first two coordinates = the derived challenges).
    pub r: Vec<EF>,
    /// The circuit combining challenge.
    pub c: EF,
    /// The four round polynomials (values at 0..=3), one per fold.
    pub rounds: Vec<Vec<EF>>,
    /// The fold challenges (checked against the verifier's reconstruction).
    pub z: Vec<EF>,
    /// The six claimed terminal evals, in layout order (absorbed after the
    /// last fold).
    pub claimed: Vec<EF>,
    /// Openings of the six sub-arrays at `z` (layout order).
    pub opens: Vec<LeafPair>,
}

pub fn to_p3_point(our_point: &[EF]) -> Point<EF> {
    Point::new(our_point.iter().rev().cloned().collect())
}

pub fn gen() -> EF {
    EF::from_basis_coefficients_slice(&[Goldilocks::ZERO, Goldilocks::ONE]).unwrap()
}

fn stored_pair(re: Goldilocks, im: Goldilocks) -> EF {
    EF::from(re) + gen() * EF::from(im)
}

pub fn open_pair(re: EF, im: EF) -> EF {
    re + gen() * im
}

/// The direct polynomial value `eq(r, z) = prod_b (z_b r_b + (1-z_b)(1-r_b))`.
fn eq_point(r: &[EF], z: &[EF]) -> EF {
    r.iter()
        .zip(z)
        .fold(EF::ONE, |acc, (&ri, &zi)| acc * (zi * ri + (EF::ONE - zi) * (EF::ONE - ri)))
}

fn eq_evals(r: &[EF]) -> Vec<EF> {
    let d = r.len();
    let n = 1 << d;
    let mut eq = vec![EF::ONE; n];
    for j in 0..d {
        let one_minus = EF::ONE - r[j];
        for i in 0..n {
            eq[i] = eq[i] * if (i >> j) & 1 == 1 { r[j] } else { one_minus };
        }
    }
    eq
}

pub fn layer_terms(c: EF) -> Vec<(EF, Vec<usize>)> {
    let neg = EF::ZERO - EF::ONE;
    vec![
        (EF::ONE, vec![0usize, 1]),
        (neg, vec![0, 3, 6]),
        (neg, vec![0, 4, 5]),
        (c, vec![0, 2]),
        (neg * c, vec![0, 5, 6]),
    ]
}

/// Opening point for a sub-array: `[local, offset bits]` (offset = high bits
/// of the flat index).
fn sub_point(offset: usize, sub_len: usize, local: &[EF], arity: usize) -> Vec<EF> {
    let local_bits = sub_len.trailing_zeros() as usize;
    let shifted = offset >> local_bits;
    let mut p = local.to_vec();
    for b in 0..(arity - local_bits) {
        p.push(if (shifted >> b) & 1 == 1 { EF::ONE } else { EF::ZERO });
    }
    p
}

/// Fold an EF array at a point (direct MLE evaluation, LSB-first).
pub(crate) fn fold_ef(values: &[EF], point: &[EF]) -> EF {
    let mut buf = values.to_vec();
    let mut size = buf.len();
    for &x in point {
        let h = size / 2;
        for i in 0..h {
            buf[i] = buf[2 * i] + x * (buf[2 * i + 1] - buf[2 * i]);
        }
        size = h;
    }
    buf[0]
}

/// Build the honest six sub-arrays (EF) and the two base commit tensors.
fn build_arrays() -> (Vec<Vec<EF>>, Vec<Goldilocks>, Vec<Goldilocks>) {
    // Deterministic "witness" values (test-only generation; the protocol's
    // randomness comes entirely from the transcript).
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        Goldilocks::from_u64(x % 100)
    };
    let num_low: Vec<EF> = (0..HALF).map(|_| EF::from(next())).collect();
    let num_high: Vec<EF> = (0..HALF).map(|_| EF::from(next())).collect();
    let den_low: Vec<EF> = (0..HALF).map(|_| EF::from(next())).collect();
    let den_high: Vec<EF> = (0..HALF).map(|_| EF::from(next())).collect();
    let next_num: Vec<EF> = (0..HALF)
        .map(|s| num_low[s] * den_high[s] + num_high[s] * den_low[s])
        .collect();
    let next_den: Vec<EF> = (0..HALF).map(|s| den_low[s] * den_high[s]).collect();
    let arrays = vec![next_num, next_den, num_low, num_high, den_low, den_high];

    let size = 1 << LAYOUT_ARITY;
    let mut re = vec![Goldilocks::ZERO; size];
    let mut im = vec![Goldilocks::ZERO; size];
    for (arr, &off) in arrays.iter().zip(&OFFSETS) {
        for (s, &v) in arr.iter().enumerate() {
            let coeffs: &[Goldilocks] = v.as_basis_coefficients_slice();
            re[off + s] = coeffs[0];
            im[off + s] = coeffs[1];
        }
    }
    (arrays, re, im)
}

/// A layer's location inside a (possibly multi-layer) combined tree tensor.
/// `half` is the sub-array length; `arity` is the committed tensor's arity.
/// The six sub-arrays are `next_num/next_den` at their offsets and
/// `num_low/num_high/den_low/den_high` as the two halves of `num/den`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerSpec {
    pub arity: usize,
    pub num_off: usize,
    pub den_off: usize,
    pub next_num_off: usize,
    pub next_den_off: usize,
    pub half: usize,
}

/// Validate a `LayerSpec` against the WHIR arity and the eq-point length
/// BEFORE any arithmetic, so an externally supplied layout can never panic or
/// silently select the wrong slices:
/// - the WHIR arity must equal `spec.arity` exactly (r and WHIR dimensions);
/// - `half` must be a nonzero power of two;
/// - every offset must be `half`-aligned with `offset + len` inside the
///   committed tensor (`num`/`den` span `2*half`, `next_*` span `half`) via
///   checked arithmetic;
/// - the four source ranges must be pairwise disjoint;
/// - the eq point must have exactly `log2(half)` coordinates.
/// The prover maps `false` to `None`, the verifier to `false`.
fn validate_spec(whir: &Whir, spec: &LayerSpec, r_len: usize) -> bool {
    if whir.num_variables() != spec.arity || !spec.half.is_power_of_two() {
        return false;
    }
    if r_len != spec.half.trailing_zeros() as usize {
        return false;
    }
    let Some(size) = 1usize.checked_shl(spec.arity as u32) else {
        return false;
    };
    let Some(num_len) = spec.half.checked_mul(2) else {
        return false;
    };
    let mut ranges = [
        (spec.num_off, num_len),
        (spec.den_off, num_len),
        (spec.next_num_off, spec.half),
        (spec.next_den_off, spec.half),
    ];
    for (off, len) in ranges {
        if off % spec.half != 0 {
            return false; // sub-point encoding requires half-aligned offsets
        }
        let Some(end) = off.checked_add(len) else {
            return false;
        };
        if end > size {
            return false;
        }
    }
    ranges.sort_unstable();
    ranges.windows(2).all(|w| w[0].0 + w[0].1 <= w[1].0)
}

/// Prove ONE layer identity segment against the given committed roots, at
/// the caller-provided eq point `r`, continuing the shared transcript.
/// Reused by both the single-layer entry point and the multi-layer chain.
/// Returns `None` for any invalid spec, eq-point length, or sub-array shape
/// (never panics, never proves a wrongly selected slice).
pub(crate) fn prove_layer_segment(
    whir: &Whir,
    stmt: &LayerStatement,
    pd_re: &zkie_core::pcs::whir::ProverData,
    pd_im: &zkie_core::pcs::whir::ProverData,
    proto: &zkie_core::pcs::whir::OpeningProtocol,
    t: &mut TwoPhaseTranscript,
    spec: &LayerSpec,
    arrays: &[Vec<EF>; 6],
    r: &[EF],
) -> Option<LayerProof> {
    if !validate_spec(whir, spec, r.len()) {
        return None;
    }
    // Every sub-array must span exactly the spec'd segment, otherwise the
    // folding below would silently operate on mismatched slices.
    if arrays.iter().any(|a| a.len() != spec.half) {
        return None;
    }
    let m = spec.half.trailing_zeros() as usize;
    let c = t.sample();
    let eq = eq_evals(r);
    let mut bufs: Vec<Vec<EF>> = vec![eq];
    bufs.extend(arrays.iter().cloned());
    let mut rounds = Vec::with_capacity(m);
    let mut z = Vec::with_capacity(m);
    for _ in 0..m {
        let pvals = virtual_round_pvals_f(&bufs, &layer_terms(c), 3);
        for &p in &pvals {
            t.absorb(p);
        }
        let zi = t.sample();
        let half = bufs[0].len() / 2;
        for buf in bufs.iter_mut() {
            for i in 0..half {
                let f0 = buf[2 * i];
                let f1 = buf[2 * i + 1];
                buf[i] = (EF::ONE - zi) * f0 + zi * f1;
            }
            buf.truncate(half);
        }
        rounds.push(pvals);
        z.push(zi);
    }
    let claimed: Vec<EF> = arrays.iter().map(|a| fold_ef(a, &z)).collect();
    for &cl in &claimed {
        t.absorb(cl);
    }
    let offsets = [
        spec.next_num_off,
        spec.next_den_off,
        spec.num_off,
        spec.num_off + spec.half,
        spec.den_off,
        spec.den_off + spec.half,
    ];
    let opens: Vec<LeafPair> = (0..6)
        .map(|i| {
            let pt = sub_point(offsets[i], spec.half, &z, spec.arity);
            let p3 = to_p3_point(&pt);
            let open_re = whir.open_ef(&stmt.root_re, pd_re.clone(), proto, &p3);
            let open_im = whir.open_ef(&stmt.root_im, pd_im.clone(), proto, &p3);
            LeafPair { open_re, open_im }
        })
        .collect();
    LayerProof {
        r: r.to_vec(),
        c,
        rounds,
        z,
        claimed,
        opens,
    }
    .into()
}

/// Verify ONE layer identity segment, continuing the shared transcript. The
/// caller supplies the expected eq point `r` (derived before the segment).
/// Malformed specs, eq points, and proof shapes return `false` (never panic).
pub(crate) fn verify_layer_segment(
    whir: &Whir,
    stmt: &LayerStatement,
    t: &mut TwoPhaseTranscript,
    spec: &LayerSpec,
    proof: &LayerProof,
    expect_r: &[EF],
) -> bool {
    if !validate_spec(whir, spec, expect_r.len()) {
        return false;
    }
    let m = spec.half.trailing_zeros() as usize;
    if proof.r != expect_r
        || proof.claimed.len() != 6
        || proof.opens.len() != 6
        || proof.rounds.len() != m
        || proof.z.len() != m
    {
        return false;
    }
    let c = t.sample();
    if proof.c != c {
        return false;
    }
    let mut prev = EF::ZERO;
    let mut z = Vec::with_capacity(proof.rounds.len());
    for pvals in &proof.rounds {
        if pvals.len() != 4 {
            return false;
        }
        for &p in pvals {
            t.absorb(p);
        }
        let zi = t.sample();
        if pvals[0] + pvals[1] != prev {
            return false;
        }
        prev = interpolate_f(pvals, zi, 3);
        z.push(zi);
    }
    if proof.z != z {
        return false;
    }
    for &cl in &proof.claimed {
        t.absorb(cl);
    }

    let proto = whir.opening_protocol(spec.arity, 1);
    let offsets = [
        spec.next_num_off,
        spec.next_den_off,
        spec.num_off,
        spec.num_off + spec.half,
        spec.den_off,
        spec.den_off + spec.half,
    ];
    let mut evals = Vec::with_capacity(6);
    for (i, leaf) in proof.opens.iter().enumerate() {
        let pt = sub_point(offsets[i], spec.half, &z, spec.arity);
        let p3 = to_p3_point(&pt);
        if whir.verify_ef(&stmt.root_re, &leaf.open_re.0, &proto, &p3).ok() != Some(leaf.open_re.1)
            || whir.verify_ef(&stmt.root_im, &leaf.open_im.0, &proto, &p3).ok() != Some(leaf.open_im.1)
        {
            return false;
        }
        let v = open_pair(leaf.open_re.1, leaf.open_im.1);
        if v != proof.claimed[i] {
            return false;
        }
        evals.push(v);
    }
    let identity = evals[0] - evals[2] * evals[5] - evals[3] * evals[4]
        + c * evals[1]
        - c * evals[4] * evals[5];
    prev == eq_point(&proof.r, &z) * identity
}

/// Prove the single-layer identity (fixed N = 32 segment). Returns
/// `(statement, proof)` or `None` for malformed inputs.
pub fn prove(whir: &Whir) -> Option<(LayerStatement, LayerProof)> {
    if whir.num_variables() != LAYOUT_ARITY {
        return None;
    }
    let (arrays, re, im) = build_arrays();
    let (root_re, pd_re, proto_re) = whir.commit(&re);
    let (root_im, pd_im, proto_im) = whir.commit(&im);
    let stmt = LayerStatement { root_re, root_im };
    let mut s = TwoPhaseTranscript::new(PROTOCOL, &[HALF], &[&stmt.root_re, &stmt.root_im]);
    let (alpha, beta) = s.sample_derived_challenges();
    // No derived commitments in this slice: the tree IS the statement.
    s.absorb_derived_roots(&[]);
    let mut r = vec![alpha, beta];
    r.extend(s.sample_vec(ROUNDS - 2));
    let spec = LayerSpec {
        arity: LAYOUT_ARITY,
        num_off: OFFSETS[2],
        den_off: OFFSETS[4],
        next_num_off: OFFSETS[0],
        next_den_off: OFFSETS[1],
        half: HALF,
    };
    let arrays_ref: &[Vec<EF>; 6] = arrays.as_slice().try_into().expect("six arrays");
    let proof = prove_layer_segment(
        whir, &stmt, &pd_re, &pd_im, &proto_re, &mut s, &spec, arrays_ref, &r,
    )?;
    Some((stmt, proof))
}

/// Verify the layer identity against the statement. Statement and proof only
/// — no witness, no recomputation, no `open` calls. Malformed shapes return
/// `false` (never panic).
pub fn verify(whir: &Whir, stmt: &LayerStatement, proof: &LayerProof) -> bool {
    if whir.num_variables() != LAYOUT_ARITY
        || proof.r.len() != ROUNDS
        || proof.z.len() != ROUNDS
        || proof.rounds.len() != ROUNDS
    {
        return false;
    }
    let mut s = TwoPhaseTranscript::new(PROTOCOL, &[HALF], &[&stmt.root_re, &stmt.root_im]);
    let (alpha, beta) = s.sample_derived_challenges();
    s.absorb_derived_roots(&[]);
    let mut r = vec![alpha, beta];
    r.extend(s.sample_vec(ROUNDS - 2));
    let spec = LayerSpec {
        arity: LAYOUT_ARITY,
        num_off: OFFSETS[2],
        den_off: OFFSETS[4],
        next_num_off: OFFSETS[0],
        next_den_off: OFFSETS[1],
        half: HALF,
    };
    verify_layer_segment(whir, stmt, &mut s, &spec, proof, &r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Whir, LayerStatement, LayerProof) {
        let whir = Whir::new_target(LAYOUT_ARITY, 90, 0).expect("valid arity");
        let (stmt, proof) = prove(&whir).expect("honest prove");
        (whir, stmt, proof)
    }

    #[test]
    fn honest_roundtrip() {
        let (whir, stmt, proof) = fixture();
        assert!(verify(&whir, &stmt, &proof));
    }

    #[test]
    fn tampered_root_rejected() {
        let (whir, stmt, proof) = fixture();
        let mut rng = zkie_core::common::field::XorShift64::new(0xF01);
        let fake: Vec<Goldilocks> = (0..(1 << LAYOUT_ARITY)).map(|_| rng.field()).collect();
        let (fake_root, _, _) = whir.commit(&fake);
        let mut bad = stmt.clone();
        bad.root_re = fake_root;
        assert!(!verify(&whir, &bad, &proof));
    }

    #[test]
    fn tampered_claimed_eval_rejected() {
        let (whir, stmt, mut proof) = fixture();
        proof.claimed[0] = proof.claimed[0] + EF::ONE;
        assert!(!verify(&whir, &stmt, &proof));
    }

    #[test]
    fn tampered_round_rejected() {
        let (whir, stmt, mut proof) = fixture();
        proof.rounds[0][0] = proof.rounds[0][0] + EF::ONE;
        assert!(!verify(&whir, &stmt, &proof));
    }

    #[test]
    fn tampered_opening_value_rejected() {
        let (whir, stmt, mut proof) = fixture();
        proof.opens[0].open_re.1 = proof.opens[0].open_re.1 + EF::ONE;
        assert!(!verify(&whir, &stmt, &proof));
    }

    #[test]
    fn reordered_eq_point_rejected() {
        let (whir, stmt, mut proof) = fixture();
        proof.r.swap(0, 1);
        assert!(!verify(&whir, &stmt, &proof));
    }

    #[test]
    fn verify_never_opens() {
        let (whir, stmt, proof) = fixture();
        let before = whir.open_stats();
        assert!(verify(&whir, &stmt, &proof));
        assert_eq!(whir.open_stats(), before, "verifier must never open");
    }

    #[test]
    fn malformed_shapes_rejected() {
        let (whir, stmt, proof) = fixture();
        assert!(verify(&whir, &stmt, &proof));
        let mut bad = proof.clone();
        bad.claimed.pop();
        assert!(!verify(&whir, &stmt, &bad));
        let mut bad2 = proof.clone();
        bad2.z.pop();
        assert!(!verify(&whir, &stmt, &bad2));
        let mut bad3 = proof.clone();
        bad3.rounds.pop();
        assert!(!verify(&whir, &stmt, &bad3));
        // Wrong arity whir instance.
        let whir2 = Whir::new_target(6, 90, 0).expect("valid arity");
        assert!(prove(&whir2).is_none());
    }

    /// The parameterized segment helpers must reject invalid external
    /// layouts: the prover returns `None`, the verifier returns `false` —
    /// never panicking and never silently selecting the wrong slices.
    #[test]
    fn segment_helpers_reject_invalid_layouts() {
        let (whir, stmt, proof) = fixture();
        let spec = LayerSpec {
            arity: LAYOUT_ARITY,
            num_off: OFFSETS[2],
            den_off: OFFSETS[4],
            next_num_off: OFFSETS[0],
            next_den_off: OFFSETS[1],
            half: HALF,
        };
        let mut ready = || {
            let mut s =
                TwoPhaseTranscript::new(PROTOCOL, &[HALF], &[&stmt.root_re, &stmt.root_im]);
            let _ = s.sample_derived_challenges();
            s.absorb_derived_roots(&[]);
            // Mirror verify()'s r derivation (alpha/beta tail) so the
            // pre-segment transcript state is identical.
            s.sample_vec(ROUNDS - 2);
            s
        };
        let mut bads = vec![];
        let mut b = spec;
        b.half = 0; // not a nonzero power of two
        bads.push(b);
        let mut b = spec;
        b.half = 12; // not a power of two
        bads.push(b);
        let mut b = spec;
        b.next_num_off = 128; // offset + half out of bounds
        bads.push(b);
        let mut b = spec;
        b.next_num_off = spec.num_off; // overlapping segments
        bads.push(b);
        let mut b = spec;
        b.arity = 6; // spec arity != WHIR arity
        bads.push(b);
        for bad in &bads {
            assert!(!verify_layer_segment(&whir, &stmt, &mut ready(), bad, &proof, &proof.r));
        }
        // Wrong eq-point length on the verifier side.
        assert!(!verify_layer_segment(&whir, &stmt, &mut ready(), &spec, &proof, &proof.r[..3]));

        // Prover side: invalid layouts yield `None` (never a proof).
        let (arrays, re, im) = build_arrays();
        let (_, pd_re, proto_re) = whir.commit(&re);
        let (_, pd_im, _) = whir.commit(&im);
        let arrays_ref: &[Vec<EF>; 6] = arrays.as_slice().try_into().expect("six arrays");
        for bad in &bads {
            let mut s = ready();
            assert!(prove_layer_segment(&whir, &stmt, &pd_re, &pd_im, &proto_re, &mut s, bad, arrays_ref, &proof.r).is_none());
        }
        // Wrong eq-point length / wrong sub-array length on the prover side.
        let mut s = ready();
        assert!(prove_layer_segment(&whir, &stmt, &pd_re, &pd_im, &proto_re, &mut s, &spec, arrays_ref, &proof.r[..3]).is_none());
        let mut short = arrays.clone();
        short[0].pop();
        let short_ref: &[Vec<EF>; 6] = short.as_slice().try_into().expect("six arrays");
        let mut s = ready();
        assert!(prove_layer_segment(&whir, &stmt, &pd_re, &pd_im, &proto_re, &mut s, &spec, short_ref, &proof.r).is_none());

        // Sanity: the honest spec still round-trips through the segment
        // helpers.
        let mut s = ready();
        assert!(verify_layer_segment(&whir, &stmt, &mut s, &spec, &proof, &proof.r));
    }

    /// Forgery regression (same-statement malicious construction): commit
    /// INVALID arrays (fold identity violated pointwise), derive `r`/`c` from
    /// THAT statement's own transcript, open the invalid arrays honestly at a
    /// freely chosen `z`, and fabricate degree-1 round polynomials
    /// `q_i(X) = T_i (X − 1/2) / (z_i − 1/2)` that satisfy the OLD
    /// known-challenge algebra exactly (q(0)+q(1) == 0 chains, final value
    /// lands on `eq(r,z)·identity(z)`). The old-style check must PASS (the
    /// old algebra is forgeable), the NEW interactive verifier must REJECT,
    /// and the rejection cause must be the transcript-derived challenge
    /// mismatch — not the roots, `r`, or `c` (which are the statement's own).
    #[test]
    fn same_statement_fabricated_known_challenge_rounds_rejected() {
        use zkie_core::common::field::Field;
        let whir = Whir::new_target(LAYOUT_ARITY, 90, 0).expect("valid arity");

        // 1. Invalid committed arrays (the statement under attack).
        let (mut arrays, _, _) = build_arrays();
        arrays[0][0] = arrays[0][0] + EF::ONE; // next_num is NOT the fold
        let (re_bad, im_bad) = {
            let size = 1 << LAYOUT_ARITY;
            let mut re = vec![Goldilocks::ZERO; size];
            let mut im = vec![Goldilocks::ZERO; size];
            for (arr, &off) in arrays.iter().zip(&OFFSETS) {
                for (s, &v) in arr.iter().enumerate() {
                    let coeffs: &[Goldilocks] = v.as_basis_coefficients_slice();
                    re[off + s] = coeffs[0];
                    im[off + s] = coeffs[1];
                }
            }
            (re, im)
        };
        let (root_re, pd_re, proto_re) = whir.commit(&re_bad);
        let (root_im, pd_im, proto_im) = whir.commit(&im_bad);
        let stmt = LayerStatement {
            root_re: root_re.clone(),
            root_im: root_im.clone(),
        };

        // 2. The statement's OWN schedule: r and c.
        let mut s = TwoPhaseTranscript::new(PROTOCOL, &[HALF], &[&root_re, &root_im]);
        let (alpha, beta) = s.sample_derived_challenges();
        s.absorb_derived_roots(&[]);
        let mut r = vec![alpha, beta];
        r.extend(s.sample_vec(ROUNDS - 2));
        let c = s.sample();

        // 3. OLD known-challenge algebra: the fold point z is FREE (not bound
        //    to the round messages) — choose the next transcript samples.
        let z: Vec<EF> = (0..ROUNDS).map(|_| s.sample()).collect();

        // 4. Honest terminal openings of the INVALID arrays at z.
        let claimed: Vec<EF> = arrays.iter().map(|a| fold_ef(a, &z)).collect();
        let opens: Vec<LeafPair> = (0..6)
            .map(|i| {
                let pt = sub_point(OFFSETS[i], HALF, &z, LAYOUT_ARITY);
                let p3 = to_p3_point(&pt);
                let open_re = whir.open_ef(&root_re, pd_re.clone(), &proto_re, &p3);
                let open_im = whir.open_ef(&root_im, pd_im.clone(), &proto_im, &p3);
                LeafPair { open_re, open_im }
            })
            .collect();
        let identity = claimed[0] - claimed[2] * claimed[5] - claimed[3] * claimed[4]
            + c * claimed[1]
            - c * claimed[4] * claimed[5];
        let target = eq_point(&r, &z) * identity;

        // 5. Fabricated degree-1 rounds q_i(X) = T_i (X - 1/2) / (z_i - 1/2),
        //    with T_0..=2 = 0 and T_3 = target.
        let half = (EF::ONE + EF::ONE).inverse();
        let rounds: Vec<Vec<EF>> = (0..ROUNDS)
            .map(|i| {
                let t = if i == ROUNDS - 1 { target } else { EF::ZERO };
                let denom = (z[i] - half).inverse();
                (0..4)
                    .map(|k| t * (EF::from(Goldilocks::from_u64(k as u64)) - half) * denom)
                    .collect()
            })
            .collect();

        // 6. The OLD known-challenge check PASSES on this forgery.
        let mut prev = EF::ZERO;
        for (pvals, zi) in rounds.iter().zip(&z) {
            assert_eq!(
                pvals[0] + pvals[1],
                prev,
                "fabricated chain must satisfy the old round checks"
            );
            prev = interpolate_f(pvals, *zi, 3);
        }
        assert_eq!(
            prev, target,
            "fabricated chain lands on the terminal check under the old known-challenge algebra"
        );

        // 7. The NEW interactive verifier rejects it.
        let forged = LayerProof {
            r: r.clone(),
            c,
            rounds,
            z: z.clone(),
            claimed,
            opens,
        };
        assert!(!verify(&whir, &stmt, &forged), "interactive verifier must reject");

        // 8. Pin the rejection cause: the interactive verifier re-samples z
        //    AFTER absorbing the fabricated coefficients, so its transcript
        //    diverges from the forged z — while r and c match exactly.
        let mut s2 = TwoPhaseTranscript::new(PROTOCOL, &[HALF], &[&root_re, &root_im]);
        let (a2, b2) = s2.sample_derived_challenges();
        s2.absorb_derived_roots(&[]);
        let mut r2 = vec![a2, b2];
        r2.extend(s2.sample_vec(ROUNDS - 2));
        let c2 = s2.sample();
        assert_eq!(r2, r, "r is the statement's own — not the rejection cause");
        assert_eq!(c2, c, "c is the statement's own — not the rejection cause");
        let mut z2 = Vec::new();
        for pvals in &forged.rounds {
            for &p in pvals {
                s2.absorb(p);
            }
            z2.push(s2.sample());
        }
        assert_ne!(z2, forged.z, "rejection is due to the transcript-derived challenge mismatch");
    }
}
