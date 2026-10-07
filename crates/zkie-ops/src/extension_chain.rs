//! Two-shard extension-field GKR chain: `Y = H @ W2`, `H = X @ W1`, with every
//! challenge, round coefficient, and terminal claim in the FULL quadratic
//! extension field `EF` while all tensor storage stays in base Goldilocks.
//!
//! Protocol (stage 1, linear matmuls only — no affine rounding, logUp, or
//! nonlinear ops):
//! - Public statement: dimensions + canonical commitment roots for
//!   `X, W1, H, W2, Y`. The verifier takes the statement and the proof only —
//!   no witness store, no forward recomputation, no prover data.
//! - All five tensors are committed (base field) before the transcript runs.
//! - A Poseidon2 Fiat–Shamir transcript is domain-separated by protocol
//!   label, dimensions, and all five roots; the output point for `Y` is a
//!   full `EF` point sampled from it; the claimed `Y` evaluation is absorbed
//!   before the rounds.
//! - Shard 2 (`Y = H @ W2`) is reduced by a product sumcheck over the
//!   contraction index; EACH round message is absorbed BEFORE that round's
//!   `EF` challenge is sampled. Its terminal claims on `H` and `W2` are full
//!   `EF` points/values, absorbed, and authenticated by WHIR openings at
//!   full `EF` points against the statement roots.
//! - Shard 1 (`H = X @ W1`) is reduced at the SAME `H` claim (identical `EF`
//!   point and value) — the chain's output claim is exactly shard 1's input
//!   claim. Its terminal claims on `X` and `W1` are likewise WHIR-authenticated.
//! - No same_poly merging is needed for this single-consumer chain; fanout
//!   merging is future work (documented).
//!
//! Soundness: WHIR openings are verified against the EXPECTED statement roots
//! (supplied by the caller), the sumcheck round checks tie the claimed `Y`
//! evaluation to the terminal products, and the terminal claims equal the
//! WHIR-verified opening values. No EF claim is ever downcast to the base
//! field; the chain's challenges are genuine extension elements.
//!
//! Security label: the WHIR instances here carry a PCS target of 90 bits with
//! a caller-chosen PoW budget. This is a correctness prototype — NO claim of
//! total 90-bit soundness is made: the sumcheck round soundness and the
//! Fiat–Shamir reduction are not yet analyzed end-to-end.
//!
//! This module does NOT remove or change forward recomputation in the existing
//! general composer (`compose.rs`) — it is a separate, self-contained path.

use zkie_core::common::field::{EF, Goldilocks, PrimeCharacteristicRing};
use zkie_core::common::mle;
use zkie_core::common::sumcheck::{fold_f, product_round_f, RoundPolyF};
use zkie_core::common::transcript::ETranscript;
use zkie_core::pcs::whir::{Commitment, Point, Proof, Whir};

/// The canonical protocol label (domain-separates the transcript).
pub const PROTOCOL: &str = "zkie/extension-chain/v1";

/// Public statement: dimensions (all powers of two, `>= 2`) and the canonical
/// commitment roots in the order `[X, W1, H, W2, Y]`.
#[derive(Clone, Debug)]
pub struct ChainStatement {
    pub m: usize,
    pub d: usize,
    pub k: usize,
    pub n: usize,
    pub root_x: Commitment,
    pub root_w1: Commitment,
    pub root_h: Commitment,
    pub root_w2: Commitment,
    pub root_y: Commitment,
}

impl ChainStatement {
    fn roots_slice(&self) -> Vec<&Commitment> {
        vec![
            &self.root_x,
            &self.root_w1,
            &self.root_h,
            &self.root_w2,
            &self.root_y,
        ]
    }
}

/// One `Whir` instance per tensor arity (a WHIR instance is bound to a single
/// number of variables).
pub struct ChainWhir {
    pub x: Whir,
    pub w1: Whir,
    pub h: Whir,
    pub w2: Whir,
    pub y: Whir,
}

fn log2_pow2(v: usize) -> Option<usize> {
    if v >= 2 && v.is_power_of_two() {
        Some(v.trailing_zeros() as usize)
    } else {
        None
    }
}

impl ChainWhir {
    /// Build the five instances for dimensions `[m, d, k, n]` (all powers of
    /// two, `>= 2`), each targeting `security_level` bits with `pow_budget`
    /// grinding budget. Returns `None` for malformed dimensions.
    pub fn new(
        m: usize,
        d: usize,
        k: usize,
        n: usize,
        security_level: usize,
        pow_budget: usize,
    ) -> Option<Self> {
        let (lm, ld, lk, ln) = (
            log2_pow2(m)?,
            log2_pow2(d)?,
            log2_pow2(k)?,
            log2_pow2(n)?,
        );
        // Each WHIR instance is fallible: tiny arities (below the folding
        // factor), PoW schedules exceeding the budget, or absurd FFT domains
        // yield `None` here rather than a panic.
        Some(ChainWhir {
            x: Whir::new_target(lm + ld, security_level, pow_budget)?,
            w1: Whir::new_target(ld + lk, security_level, pow_budget)?,
            h: Whir::new_target(lm + lk, security_level, pow_budget)?,
            w2: Whir::new_target(lk + ln, security_level, pow_budget)?,
            y: Whir::new_target(lm + ln, security_level, pow_budget)?,
        })
    }
}

/// The transported proof: two sumcheck transcripts plus the five WHIR opening
/// proofs. The terminal claim VALUES live in the opening tuples and are
/// shared between WHIR authentication and the sumcheck final checks.
#[derive(Clone)]
pub struct ExtensionChainProof {
    /// Shard 2 (`Y = H @ W2`) rounds, over `log2 k` variables.
    pub rounds2: Vec<RoundPolyF<EF>>,
    /// Shard 1 (`H = X @ W1`) rounds, over `log2 d` variables.
    pub rounds1: Vec<RoundPolyF<EF>>,
    /// WHIR opening of `Y` at the sampled output point, plus claimed value.
    pub open_y: (Proof, EF),
    /// WHIR opening of `H` at `z ++ u`, plus claimed value.
    pub open_h: (Proof, EF),
    /// WHIR opening of `X` at `r ++ u`, plus claimed value.
    pub open_x: (Proof, EF),
    /// WHIR opening of `W1` at `z ++ r`, plus claimed value.
    pub open_w1: (Proof, EF),
    /// WHIR opening of `W2` at `v ++ z`, plus claimed value.
    pub open_w2: (Proof, EF),
}

/// p3 WHIR points are MSB-first; our MLE convention is LSB-first. Convert by
/// reversing coordinates (full EF elements — no embedding, no downcast).
fn to_p3_point(our_point: &[EF]) -> Point<EF> {
    Point::new(our_point.iter().rev().cloned().collect())
}

/// `a x b` stored flat (row-major): transpose to `b x a` so the row index
/// becomes the LOW-order variables (enabling `partial_eval` to fix rows).
fn transpose(a: &[Goldilocks], m: usize, k: usize) -> Vec<Goldilocks> {
    let mut t = vec![Goldilocks::ZERO; k * m];
    for i in 0..m {
        for kk in 0..k {
            t[kk * m + i] = a[i * k + kk];
        }
    }
    t
}

/// Commit all five tensors, run the transcript, both sumchecks, and the five
/// WHIR openings. Returns the statement (for the verifier) and the proof.
pub fn prove(
    chain: &ChainWhir,
    x: &[Goldilocks],
    w1: &[Goldilocks],
    h: &[Goldilocks],
    w2: &[Goldilocks],
    y: &[Goldilocks],
    m: usize,
    d: usize,
    k: usize,
    n: usize,
) -> Option<(ChainStatement, ExtensionChainProof)> {
    let (lm, ld, lk, ln) = (log2_pow2(m)?, log2_pow2(d)?, log2_pow2(k)?, log2_pow2(n)?);
    // All five tensor sizes via checked arithmetic: huge power-of-two
    // dimensions must yield None, never an overflowing multiply.
    let (sz_x, sz_w1, sz_h, sz_w2, sz_y) = match (
        m.checked_mul(d),
        d.checked_mul(k),
        m.checked_mul(k),
        k.checked_mul(n),
        m.checked_mul(n),
    ) {
        (Some(a), Some(b), Some(c), Some(e), Some(f)) => (a, b, c, e, f),
        _ => return None,
    };
    if x.len() != sz_x || w1.len() != sz_w1 || h.len() != sz_h || w2.len() != sz_w2 || y.len() != sz_y
    {
        return None;
    }

    // Commit all base tensors BEFORE the transcript runs.
    let (root_x, pd_x, proto_x) = chain.x.commit(x);
    let (root_w1, pd_w1, proto_w1) = chain.w1.commit(w1);
    let (root_h, pd_h, proto_h) = chain.h.commit(h);
    let (root_w2, pd_w2, proto_w2) = chain.w2.commit(w2);
    let (root_y, pd_y, proto_y) = chain.y.commit(y);

    let stmt = ChainStatement {
        m,
        d,
        k,
        n,
        root_x: root_x.clone(),
        root_w1: root_w1.clone(),
        root_h: root_h.clone(),
        root_w2: root_w2.clone(),
        root_y: root_y.clone(),
    };

    let mut t = ETranscript::new(PROTOCOL, &[m, d, k, n], &stmt.roots_slice());

    // Output point for Y: full EF point in Y's own low-bit-first variable
    // order — columns (v) first, rows (u) second.
    let y_point = t.sample_vec(ln + lm);
    let v = &y_point[..ln];
    let u = &y_point[ln..];
    let y_claim = mle::eval_ef(y, &y_point);
    t.absorb(y_claim);

    // Shard 2: Y = H @ W2, sumcheck over the shared index (k variables).
    // H^T (k x m) puts H's rows in the LOW variables so `partial_eval` fixes
    // them; W2's columns are already its low variables.
    let h_t = transpose(h, m, k);
    let mut h_rest = mle::partial_eval_ef(&h_t, u); // remaining: z variables
    let mut w2_rest = mle::partial_eval_ef(w2, v); // remaining: z variables
    let mut rounds2 = Vec::with_capacity(lk);
    let mut r_w = Vec::with_capacity(lk);
    for _ in 0..lk {
        let rp = product_round_f(&h_rest, &w2_rest);
        t.absorb_round(&rp);
        let c = t.sample();
        fold_f(&mut h_rest, c);
        fold_f(&mut w2_rest, c);
        rounds2.push(rp);
        r_w.push(c);
    }
    let h_claim = h_rest[0];
    let w2_claim = w2_rest[0];
    t.absorb(h_claim);
    t.absorb(w2_claim);

    // Shard 1: H = X @ W1, sumcheck over the shared index (d variables), at
    // the SAME H claim. X^T (d x m) puts X's rows in the LOW variables; W1's
    // columns (the z variables) are already its low variables.
    let x_t = transpose(x, m, d);
    let mut x_rest = mle::partial_eval_ef(&x_t, u); // remaining: r variables
    let mut w1_rest = mle::partial_eval_ef(w1, &r_w); // fix z; remaining: r variables
    let mut rounds1 = Vec::with_capacity(ld);
    let mut r_t = Vec::with_capacity(ld);
    for _ in 0..ld {
        let rp = product_round_f(&x_rest, &w1_rest);
        t.absorb_round(&rp);
        let c = t.sample();
        fold_f(&mut x_rest, c);
        fold_f(&mut w1_rest, c);
        rounds1.push(rp);
        r_t.push(c);
    }
    let x_claim = x_rest[0];
    let w1_claim = w1_rest[0];
    t.absorb(x_claim);
    t.absorb(w1_claim);

    // Claim points (our LSB-first convention).
    // Claim points, each in the tensor's OWN low-bit-first variable order:
    // H (m x k, cols low) at z ++ u; W2 (k x n, cols low) at v ++ z;
    // X (m x d, cols low) at r ++ u; W1 (d x k, cols low) at z ++ r,
    // where z = shard-2 challenges and r = shard-1 challenges.
    let mut h_pt = r_w.clone();
    h_pt.extend_from_slice(u); // H at (z ++ u)
    let mut w2_pt = v.to_vec();
    w2_pt.extend_from_slice(&r_w); // W2 at (v ++ z)
    let mut x_pt = r_t.clone();
    x_pt.extend_from_slice(u); // X at (r ++ u)
    let mut w1_pt = r_w.clone();
    w1_pt.extend_from_slice(&r_t); // W1 at (z ++ r)

    // Independent reference evaluations BEFORE the openings, so any terminal
    // mismatch is localized to a single claim rather than a failed open.
    if y_claim != mle::eval_ef(y, &y_point)
        || h_claim != mle::eval_ef(h, &h_pt)
        || w2_claim != mle::eval_ef(w2, &w2_pt)
        || x_claim != mle::eval_ef(x, &x_pt)
        || w1_claim != mle::eval_ef(w1, &w1_pt)
    {
        return None;
    }

    // WHIR openings at full EF points; the opened values must match the
    // sumcheck terminal claims exactly.
    let open_y = chain.y.open_ef(&root_y, pd_y, &proto_y, &to_p3_point(&y_point));
    if open_y.1 != y_claim {
        return None;
    }
    let open_h = chain.h.open_ef(&root_h, pd_h, &proto_h, &to_p3_point(&h_pt));
    if open_h.1 != h_claim {
        return None;
    }
    let open_w2 = chain.w2.open_ef(&root_w2, pd_w2, &proto_w2, &to_p3_point(&w2_pt));
    if open_w2.1 != w2_claim {
        return None;
    }
    let open_x = chain.x.open_ef(&root_x, pd_x, &proto_x, &to_p3_point(&x_pt));
    if open_x.1 != x_claim {
        return None;
    }
    let open_w1 = chain.w1.open_ef(&root_w1, pd_w1, &proto_w1, &to_p3_point(&w1_pt));
    if open_w1.1 != w1_claim {
        return None;
    }

    Some((
        stmt,
        ExtensionChainProof {
            rounds2,
            rounds1,
            open_y,
            open_h,
            open_x,
            open_w1,
            open_w2,
        },
    ))
}

/// Verify the chain proof against the EXPECTED statement. Statement and proof
/// only — no witness store, no forward recomputation, no prover data.
/// Malformed shapes/proof counts return `false` (never panic).
pub fn verify(chain: &ChainWhir, stmt: &ChainStatement, proof: &ExtensionChainProof) -> bool {
    let (lm, ld, lk, ln) = match (
        log2_pow2(stmt.m),
        log2_pow2(stmt.d),
        log2_pow2(stmt.k),
        log2_pow2(stmt.n),
    ) {
        (Some(a), Some(b), Some(c), Some(e)) => (a, b, c, e),
        _ => return false,
    };
    // Tensor sizes via checked arithmetic (huge power-of-two dims must
    // reject, not overflow), and the derived per-tensor arities must match
    // the ChainWhir instances BEFORE any transcript or algebra runs.
    if stmt.m.checked_mul(stmt.d).is_none()
        || stmt.d.checked_mul(stmt.k).is_none()
        || stmt.m.checked_mul(stmt.k).is_none()
        || stmt.k.checked_mul(stmt.n).is_none()
        || stmt.m.checked_mul(stmt.n).is_none()
        || chain.x.num_variables() != lm + ld
        || chain.w1.num_variables() != ld + lk
        || chain.h.num_variables() != lm + lk
        || chain.w2.num_variables() != lk + ln
        || chain.y.num_variables() != lm + ln
    {
        return false;
    }
    if proof.rounds2.len() != lk || proof.rounds1.len() != ld {
        return false;
    }

    let mut t = ETranscript::new(PROTOCOL, &[stmt.m, stmt.d, stmt.k, stmt.n], &stmt.roots_slice());
    let y_point = t.sample_vec(ln + lm);
    let v = &y_point[..ln];
    let u = &y_point[ln..];
    let y_claim = proof.open_y.1;
    t.absorb(y_claim);

    // Shard 2 rounds: absorb each message BEFORE sampling its challenge.
    let mut prev = y_claim;
    let mut r_w = Vec::with_capacity(lk);
    for rp in &proof.rounds2 {
        t.absorb_round(rp);
        let c = t.sample();
        let p0 = rp.c0;
        let p1 = rp.c0 + rp.c1 + rp.c2;
        if p0 + p1 != prev {
            return false;
        }
        prev = rp.eval(c);
        r_w.push(c);
    }
    let h_claim = proof.open_h.1;
    let w2_claim = proof.open_w2.1;
    t.absorb(h_claim);
    t.absorb(w2_claim);
    if prev != h_claim * w2_claim {
        return false;
    }

    // Shard 1 rounds at the same H claim.
    prev = h_claim;
    let mut r_t = Vec::with_capacity(ld);
    for rp in &proof.rounds1 {
        t.absorb_round(rp);
        let c = t.sample();
        let p0 = rp.c0;
        let p1 = rp.c0 + rp.c1 + rp.c2;
        if p0 + p1 != prev {
            return false;
        }
        prev = rp.eval(c);
        r_t.push(c);
    }
    let x_claim = proof.open_x.1;
    let w1_claim = proof.open_w1.1;
    t.absorb(x_claim);
    t.absorb(w1_claim);
    if prev != x_claim * w1_claim {
        return false;
    }

    // Claim points, each in the tensor's OWN low-bit-first variable order
    // (see prove): H at z ++ u, W2 at v ++ z, X at r ++ u, W1 at z ++ r.
    let mut h_pt = r_w.clone();
    h_pt.extend_from_slice(u);
    let mut w2_pt = v.to_vec();
    w2_pt.extend_from_slice(&r_w);
    let mut x_pt = r_t.clone();
    x_pt.extend_from_slice(u);
    let mut w1_pt = r_w.clone();
    w1_pt.extend_from_slice(&r_t);

    // WHIR authentication against the EXPECTED statement roots: the verified
    // opening values must equal the claimed values used in the sumchecks.
    let proto_y = chain.y.opening_protocol(lm + ln, 1);
    let proto_h = chain.h.opening_protocol(lm + lk, 1);
    let proto_w2 = chain.w2.opening_protocol(lk + ln, 1);
    let proto_x = chain.x.opening_protocol(lm + ld, 1);
    let proto_w1 = chain.w1.opening_protocol(ld + lk, 1);

    let y_verified = match chain
        .y
        .verify_ef(&stmt.root_y, &proof.open_y.0, &proto_y, &to_p3_point(&y_point))
    {
        Ok(e) => e,
        Err(_) => return false,
    };
    if y_verified != y_claim {
        return false;
    }
    let h_verified = match chain
        .h
        .verify_ef(&stmt.root_h, &proof.open_h.0, &proto_h, &to_p3_point(&h_pt))
    {
        Ok(e) => e,
        Err(_) => return false,
    };
    if h_verified != h_claim {
        return false;
    }
    let w2_verified = match chain
        .w2
        .verify_ef(&stmt.root_w2, &proof.open_w2.0, &proto_w2, &to_p3_point(&w2_pt))
    {
        Ok(e) => e,
        Err(_) => return false,
    };
    if w2_verified != w2_claim {
        return false;
    }
    let x_verified = match chain
        .x
        .verify_ef(&stmt.root_x, &proof.open_x.0, &proto_x, &to_p3_point(&x_pt))
    {
        Ok(e) => e,
        Err(_) => return false,
    };
    if x_verified != x_claim {
        return false;
    }
    let w1_verified = match chain
        .w1
        .verify_ef(&stmt.root_w1, &proof.open_w1.0, &proto_w1, &to_p3_point(&w1_pt))
    {
        Ok(e) => e,
        Err(_) => return false,
    };
    if w1_verified != w1_claim {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_core::common::field::{BasedVectorSpace, XorShift64};

    /// Deterministic test witnesses only (test-side data generation; all
    /// protocol challenges come from the Poseidon2 transcript).
    fn matmul(a: &[Goldilocks], b: &[Goldilocks], m: usize, k: usize, n: usize) -> Vec<Goldilocks> {
        let mut c = vec![Goldilocks::ZERO; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = Goldilocks::ZERO;
                for kk in 0..k {
                    acc = acc + a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    fn fixture() -> (ChainWhir, ChainStatement, ExtensionChainProof, Vec<Vec<Goldilocks>>) {
        // Non-square power-of-two matrices; all tensor arities >= 5.
        let (m, d, k, n) = (4usize, 8usize, 8usize, 16usize);
        let mut rng = XorShift64::new(0xEC1);
        let x: Vec<Goldilocks> = (0..m * d).map(|_| rng.field()).collect();
        let w1: Vec<Goldilocks> = (0..d * k).map(|_| rng.field()).collect();
        let w2: Vec<Goldilocks> = (0..k * n).map(|_| rng.field()).collect();
        let h = matmul(&x, &w1, m, d, k);
        let y = matmul(&h, &w2, m, k, n);

        // PCS target 90 bits, PoW budget 0: correctness prototype (no
        // end-to-end 90-bit soundness claim — see module docs).
        let chain = ChainWhir::new(m, d, k, n, 90, 0).expect("dims valid");
        let (stmt, proof) = prove(&chain, &x, &w1, &h, &w2, &y, m, d, k, n).expect("honest prove");
        (chain, stmt, proof, vec![x, w1, h, w2, y])
    }

    #[test]
    fn honest_roundtrip_non_square() {
        let (chain, stmt, proof, _tensors) = fixture();
        assert!(verify(&chain, &stmt, &proof));
    }

    #[test]
    fn challenges_are_genuinely_extension() {
        let (chain, stmt, proof, _) = fixture();
        assert!(verify(&chain, &stmt, &proof));
        // Replay the transcript in the exact protocol order and check every
        // sampled round challenge has a nonzero extension coefficient.
        let mut t = ETranscript::new(PROTOCOL, &[stmt.m, stmt.d, stmt.k, stmt.n], &stmt.roots_slice());
        let lm = stmt.m.trailing_zeros() as usize;
        let ln = stmt.n.trailing_zeros() as usize;
        let _yp = t.sample_vec(ln + lm);
        t.absorb(proof.open_y.1);
        let mut count = 0usize;
        for rp in &proof.rounds2 {
            t.absorb_round(rp);
            let c = t.sample();
            let coeffs: &[Goldilocks] = c.as_basis_coefficients_slice();
            assert_ne!(coeffs[1], Goldilocks::ZERO, "round challenge must be genuinely non-base");
            count += 1;
        }
        t.absorb(proof.open_h.1);
        t.absorb(proof.open_w2.1);
        for rp in &proof.rounds1 {
            t.absorb_round(rp);
            let c = t.sample();
            let coeffs: &[Goldilocks] = c.as_basis_coefficients_slice();
            assert_ne!(coeffs[1], Goldilocks::ZERO);
            count += 1;
        }
        assert!(count > 0);
    }

    #[test]
    fn terminal_claims_match_independent_reference_ef_mle() {
        let (chain, stmt, proof, tensors) = fixture();
        assert!(verify(&chain, &stmt, &proof));
        let (x, w1, h, w2, y) = (&tensors[0], &tensors[1], &tensors[2], &tensors[3], &tensors[4]);
        let (m, d, k, n) = (stmt.m, stmt.d, stmt.k, stmt.n);
        // Replay transcript to recover the points.
        let mut t = ETranscript::new(PROTOCOL, &[m, d, k, n], &stmt.roots_slice());
        let lm = m.trailing_zeros() as usize;
        let ln = n.trailing_zeros() as usize;
        let lk = k.trailing_zeros() as usize;
        let ld = d.trailing_zeros() as usize;
        let y_point = t.sample_vec(ln + lm);
        let v = &y_point[..ln];
        let u = &y_point[ln..];
        t.absorb(proof.open_y.1);
        let mut r_w = Vec::new();
        for rp in &proof.rounds2 {
            t.absorb_round(rp);
            r_w.push(t.sample());
        }
        t.absorb(proof.open_h.1);
        t.absorb(proof.open_w2.1);
        let mut r_t = Vec::new();
        for rp in &proof.rounds1 {
            t.absorb_round(rp);
            r_t.push(t.sample());
        }
        t.absorb(proof.open_x.1);
        t.absorb(proof.open_w1.1);

        let mut h_pt = r_w.clone();
        h_pt.extend_from_slice(u);
        let mut w2_pt = v.to_vec();
        w2_pt.extend_from_slice(&r_w);
        let mut x_pt = r_t.clone();
        x_pt.extend_from_slice(u);
        let mut w1_pt = r_w.clone();
        w1_pt.extend_from_slice(&r_t);

        assert_eq!(proof.open_y.1, mle::eval_ef(y, &y_point));
        assert_eq!(proof.open_h.1, mle::eval_ef(h, &h_pt));
        assert_eq!(proof.open_w2.1, mle::eval_ef(w2, &w2_pt));
        assert_eq!(proof.open_x.1, mle::eval_ef(x, &x_pt));
        assert_eq!(proof.open_w1.1, mle::eval_ef(w1, &w1_pt));
    }

    #[test]
    fn verify_never_opens() {
        let (chain, stmt, proof, _) = fixture();
        let before = (
            chain.x.open_stats(),
            chain.w1.open_stats(),
            chain.h.open_stats(),
            chain.w2.open_stats(),
            chain.y.open_stats(),
        );
        assert!(verify(&chain, &stmt, &proof));
        let after = (
            chain.x.open_stats(),
            chain.w1.open_stats(),
            chain.h.open_stats(),
            chain.w2.open_stats(),
            chain.y.open_stats(),
        );
        assert_eq!(before, after, "verifier must never regenerate openings");
    }

    #[test]
    fn tampered_round_coeff_rejected() {
        let (chain, stmt, mut proof, _) = fixture();
        proof.rounds2[0].c0 = proof.rounds2[0].c0 + EF::ONE;
        assert!(!verify(&chain, &stmt, &proof));
    }

    #[test]
    fn tampered_terminal_claim_rejected() {
        let (chain, stmt, mut proof, _) = fixture();
        proof.open_h.1 = proof.open_h.1 + EF::ONE;
        assert!(!verify(&chain, &stmt, &proof));
    }

    #[test]
    fn tampered_y_claim_rejected() {
        let (chain, stmt, mut proof, _) = fixture();
        proof.open_y.1 = proof.open_y.1 + EF::ONE;
        assert!(!verify(&chain, &stmt, &proof));
    }

    #[test]
    fn tampered_boundary_root_rejected() {
        let (chain, stmt, proof, tensors) = fixture();
        let mut rng = XorShift64::new(0xEC2);
        let fake: Vec<Goldilocks> = (0..tensors[2].len()).map(|_| rng.field()).collect();
        let (fake_root, _, _) = chain.h.commit(&fake);
        let mut bad = stmt.clone();
        bad.root_h = fake_root;
        assert!(!verify(&chain, &bad, &proof));
    }

    #[test]
    fn swapped_openings_rejected() {
        let (chain, stmt, mut proof, _) = fixture();
        // Swap the Y and H openings (proof AND claimed value): the Y opening
        // then verifies at the H point and vice versa — must fail.
        std::mem::swap(&mut proof.open_y, &mut proof.open_h);
        assert!(!verify(&chain, &stmt, &proof));
    }

    #[test]
    fn reordered_rounds_rejected() {
        let (chain, stmt, mut proof, _) = fixture();
        proof.rounds2.swap(0, 1);
        assert!(!verify(&chain, &stmt, &proof));
    }

    #[test]
    fn truncated_proof_rejected() {
        let (chain, stmt, mut proof, _) = fixture();
        proof.rounds1.pop();
        assert!(!verify(&chain, &stmt, &proof));
    }

    #[test]
    fn wrong_statement_shape_rejected() {
        let (chain, stmt, proof, _) = fixture();
        let mut bad = stmt.clone();
        bad.n = 32; // different statement shape
        assert!(!verify(&chain, &bad, &proof));
    }

    /// Huge power-of-two dimensions (usize high bit) must be rejected with
    /// `false`/`None` — never a panicking overflow in size arithmetic.
    #[test]
    fn malformed_huge_dims_rejected() {
        let (chain, stmt, proof, _) = fixture();
        let mut bad = stmt.clone();
        bad.m = 1usize << (usize::BITS - 1); // portable usize high bit
        assert!(!verify(&chain, &bad, &proof), "huge dims must reject");

        let mut bad2 = stmt.clone();
        bad2.k = 1usize << (usize::BITS - 1);
        assert!(!verify(&chain, &bad2, &proof));

        assert!(ChainWhir::new(1usize << (usize::BITS - 1), 8, 8, 8, 90, 32).is_none());
        assert!(ChainWhir::new(1usize << (usize::BITS - 1), 1usize << (usize::BITS - 1), 8, 8, 90, 32).is_none());
    }

    /// A proof with zero rounds must be rejected by the round-count checks.
    #[test]
    fn zero_rounds_proof_rejected() {
        let (chain, stmt, mut proof, _) = fixture();
        proof.rounds2.clear();
        proof.rounds1.clear();
        assert!(!verify(&chain, &stmt, &proof));
    }

    /// `ChainWhir::new` rejects unsupported configurations (tiny arity below
    /// the folding factor, insufficient PoW budget) instead of panicking.
    #[test]
    fn chainwhir_constructor_rejects_invalid() {
        // Arity 2 per tensor: below the folding factor 5.
        assert!(ChainWhir::new(2, 2, 2, 2, 90, 0).is_none());
        // A 200-bit target with a 0-bit PoW budget cannot be met (derived
        // PoW > 0), so the instances are rejected. (90-bit/budget-0 IS valid
        // at arity 6, so this uses the higher target to exercise rejection.)
        assert!(ChainWhir::new(8, 8, 8, 8, 200, 0).is_none());
        // Same dims with a sufficient budget construct.
        assert!(ChainWhir::new(8, 8, 8, 8, 90, 32).is_some());
    }

    #[test]
    fn swapped_shard_proofs_rejected() {
        let (chain, stmt, mut proof, _) = fixture();
        // A proof for a DIFFERENT witness pair (same dims) must not verify.
        let mut rng = XorShift64::new(0xEC3);
        let (m, d, k, n) = (4usize, 8usize, 8usize, 16usize);
        let x2: Vec<Goldilocks> = (0..m * d).map(|_| rng.field()).collect();
        let w1_2: Vec<Goldilocks> = (0..d * k).map(|_| rng.field()).collect();
        let w2_2: Vec<Goldilocks> = (0..k * n).map(|_| rng.field()).collect();
        let h2 = matmul(&x2, &w1_2, m, d, k);
        let y2 = matmul(&h2, &w2_2, m, k, n);
        let (_, proof2) = prove(&chain, &x2, &w1_2, &h2, &w2_2, &y2, m, d, k, n).unwrap();
        // Splice shard-2 rounds of proof2 into proof1.
        proof.rounds2 = proof2.rounds2;
        assert!(!verify(&chain, &stmt, &proof));
    }


    #[test]
    fn whir_isolated_open_verify_full_ef_points() {
        use zkie_core::common::mle;
        for arity in [7usize, 8usize] {
            let whir = Whir::new_target(arity, 90, 0).expect("arity 7/8 valid at 90/0");
            let mut rng = XorShift64::new(0xEEE);
            let table: Vec<Goldilocks> = (0..(1usize << arity)).map(|_| rng.field()).collect();
            let (root, pd, proto) = whir.commit(&table);
            let our_pt: Vec<EF> = (0..arity).map(|_| EF::from(rng.field())).collect();
            // full-EF point with a nonzero extension part
            let mut pt = our_pt.clone();
            let g: Vec<Goldilocks> = (0..arity).map(|_| rng.field()).collect();
            for i in 0..arity {
                let b: &[Goldilocks] = pt[i].as_basis_coefficients_slice();
                let base = b[0];
                let ext = b[1] + g[i];
                pt[i] = EF::from_basis_coefficients_slice(&[base, ext]).unwrap();
            }
            let p3pt = Point::new(pt.iter().rev().cloned().collect());
            let expected = mle::eval_ef(&table, &pt);
            let (proof, opened) = whir.open_ef(&root, pd, &proto, &p3pt);
            assert_eq!(opened, expected, "open value at arity {arity}");
            let verified = whir.verify_ef(&root, &proof, &proto, &p3pt);
            match &verified {
                Ok(v) => assert_eq!(*v, expected, "verify value at arity {arity}"),
                Err(e) => panic!("arity {arity} verify Err: {:?}", e),
            }
        }
    }
}
