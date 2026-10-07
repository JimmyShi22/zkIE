//! Claim-driven extension-field reducer for the LINEAR elementwise ops:
//! `Scale` (exact, shift 0), `Add`, `AddConst` (public constant), and
//! `Transpose`. Stage B1 of the claim-driven GPT-2 integration.
//!
//! Linearity means no sumcheck is needed: given a full-EF output point and a
//! WHIR-authenticated output value, the input claim point/value are mapped
//! deterministically and authenticated via WHIR against the statement roots,
//! and the linear relation (`y = a + b`, `y = a + c`, `y = factor * a`,
//! `y[a at permuted point]`) is checked exactly. The verifier takes the
//! statement and the proof only — no witness, no forward recomputation, no
//! `open_*` calls.
//!
//! Transcript: Poseidon2 FS domain-separated by op-kind label, shapes,
//! statement roots, and public scalars (factor / constant). The output point
//! is sampled from the transcript; the claimed output value is absorbed
//! BEFORE the openings are verified. The input claims are absorbed for
//! transcript continuity (a later consumer could chain on them), but their
//! tamper rejection does NOT rely on that absorption — it derives from the
//! PCS verified-value equality (`verified == claimed`) enforced on every
//! opening.
//!
//! Scope limits (documented, not pretended):
//! - `Scale` with `shift > 0` (rounded division) is REJECTED — rounding is
//!   piecewise and belongs to the affine/Projection reducer (later stage).
//! - Only the relation on ONE op is proven; no multi-op GPT-2 chain claims.
//! - The existing composer (`compose.rs`) and its forward recomputation are
//!   untouched; this is a separate path.

use zkie_core::common::field::{EF, Goldilocks, PrimeCharacteristicRing, PrimeField64};
use zkie_core::common::mle;
use zkie_core::common::transcript::ETranscript;
use zkie_core::pcs::whir::{Commitment, Point, Proof, Whir};

/// The supported linear op kinds. Public scalars live IN the statement and are
/// bound by the transcript — no unbound caller data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearOpKind {
    /// `out = x * factor / 2^shift` with `shift == 0` REQUIRED (exact
    /// scaling; rounded scaling is rejected as unsupported).
    Scale { factor: u64, shift: u32 },
    /// `out = a + b` (both sources committed).
    Add,
    /// `out = x + c` with public constant `c`.
    AddConst { c: Goldilocks },
    /// `out` is the `k x m` transpose of the `m x k` input.
    Transpose { m: usize, k: usize },
}

impl LinearOpKind {
    fn label(&self) -> &'static str {
        match self {
            LinearOpKind::Scale { .. } => "zkie/ext-linear/v1/scale",
            LinearOpKind::Add => "zkie/ext-linear/v1/add",
            LinearOpKind::AddConst { .. } => "zkie/ext-linear/v1/add-const",
            LinearOpKind::Transpose { .. } => "zkie/ext-linear/v1/transpose",
        }
    }
}

/// Public statement: op kind, input shape (`m x k`), and the canonical roots
/// for the input, the optional second source (Add), and the output.
#[derive(Clone, Debug)]
pub struct LinearStatement {
    pub kind: LinearOpKind,
    pub m: usize,
    pub k: usize,
    pub root_a: Commitment,
    /// Second source for `Add`.
    pub root_b: Option<Commitment>,
    pub root_out: Commitment,
}

/// The transported proof: the WHIR opening of the output at the sampled
/// point, of the first source, and of the second source (Add only). The
/// claimed VALUES are shared between WHIR authentication and the linear check.
#[derive(Clone)]
pub struct LinearProof {
    pub open_out: (Proof, EF),
    pub open_a: (Proof, EF),
    pub open_b: Option<(Proof, EF)>,
}

fn log2_pow2(v: usize) -> Option<usize> {
    if v >= 2 && v.is_power_of_two() {
        Some(v.trailing_zeros() as usize)
    } else {
        None
    }
}

/// p3 WHIR points are MSB-first; our MLE convention is LSB-first.
fn to_p3_point(our_point: &[EF]) -> Point<EF> {
    Point::new(our_point.iter().rev().cloned().collect())
}

/// The output point in the output tensor's own low-bit-first variable order:
/// same as the input for Scale/Add/AddConst; for `Transpose` the output is
/// `k x m`, so its variables are `(m-vars low, k-vars high)` while the input
/// `m x k` has `(k-vars low, m-vars high)`.
fn input_point(kind: &LinearOpKind, out_point: &[EF], lm: usize) -> Vec<EF> {
    match kind {
        LinearOpKind::Transpose { .. } => {
            // out_point = [m-part (lm), k-part]; input order = [k-part, m-part].
            let mut p = out_point[lm..].to_vec();
            p.extend_from_slice(&out_point[..lm]);
            p
        }
        _ => out_point.to_vec(),
    }
}

/// Prove the linear relation. Returns the statement (for the verifier) and
/// the proof, or `None` for malformed/unsupported inputs (never panics).
pub fn prove(
    whir: &Whir,
    kind: LinearOpKind,
    m: usize,
    k: usize,
    a: &[Goldilocks],
    b: Option<&[Goldilocks]>,
    out: &[Goldilocks],
) -> Option<(LinearStatement, LinearProof)> {
    let (lm, lk) = (log2_pow2(m)?, log2_pow2(k)?);
    let arity = lm + lk;
    if whir.num_variables() != arity {
        return None;
    }
    if let Some(sz) = m.checked_mul(k) {
        if sz != a.len() || sz != out.len() {
            return None;
        }
    } else {
        return None;
    }
    if let LinearOpKind::Transpose { m: tm, k: tk } = kind {
        if tm != m || tk != k {
            return None;
        }
    }
    match (&kind, b) {
        (LinearOpKind::Add, Some(bb)) => {
            if bb.len() != a.len() {
                return None;
            }
        }
        (LinearOpKind::Add, None) => return None,
        (_, Some(_)) => return None,
        (_, None) => {}
    }
    if let LinearOpKind::Scale { shift, .. } = kind {
        if shift != 0 {
            // Rounded scaling is piecewise — explicitly unsupported here.
            return None;
        }
    }

    // Commit the sources and the output BEFORE the transcript runs.
    let (root_a, pd_a, proto_a) = whir.commit(a);
    let (root_b, pd_b, proto_b) = match b {
        Some(bb) => {
            let (r, pd, pr) = whir.commit(bb);
            (Some(r), Some(pd), Some(pr))
        }
        None => (None, None, None),
    };
    let (root_out, pd_out, proto_out) = whir.commit(out);

    let stmt = LinearStatement {
        kind,
        m,
        k,
        root_a: root_a.clone(),
        root_b: root_b.clone(),
        root_out: root_out.clone(),
    };
    let roots: Vec<&Commitment> = match &root_b {
        Some(rb) => vec![&root_a, rb, &root_out],
        None => vec![&root_a, &root_out],
    };

    let mut t = ETranscript::new(kind.label(), &[m, k], &roots);
    // Bind the public scalars.
    match kind {
        LinearOpKind::Scale { factor, shift } => {
            t.absorb_base(Goldilocks::from_u64(factor));
            t.absorb_base(Goldilocks::from_u64(shift as u64));
        }
        LinearOpKind::AddConst { c } => t.absorb_base(c),
        _ => {}
    }

    // Full-EF output point sampled from the transcript.
    let out_point = t.sample_vec(arity);
    let y_claim = mle::eval_ef(out, &out_point);
    t.absorb(y_claim);

    let a_point = input_point(&kind, &out_point, lm);
    let a_claim = mle::eval_ef(a, &a_point);
    let b_claim = b.map(|bb| mle::eval_ef(bb, &out_point));

    let open_out = whir.open_ef(&root_out, pd_out, &proto_out, &to_p3_point(&out_point));
    if open_out.1 != y_claim {
        return None;
    }
    let open_a = whir.open_ef(&root_a, pd_a, &proto_a, &to_p3_point(&a_point));
    if open_a.1 != a_claim {
        return None;
    }
    // Bind the input claims into the transcript (after the output claim), so
    // the verifier's claimed values are transcript-bound too.
    t.absorb(a_claim);
    let open_b = match (root_b, pd_b, proto_b, b_claim) {
        (Some(rb), Some(pd), Some(pr), Some(bc)) => {
            let o = whir.open_ef(&rb, pd, &pr, &to_p3_point(&out_point));
            if o.1 != bc {
                return None;
            }
            t.absorb(bc);
            Some(o)
        }
        _ => None,
    };

    Some((
        stmt,
        LinearProof {
            open_out,
            open_a,
            open_b,
        },
    ))
}

/// Verify the linear relation against the EXPECTED statement. Statement and
/// proof only — no witness, no forward recomputation, never `open`. Malformed
/// shapes/metadata return `false` (never panic).
pub fn verify(whir: &Whir, stmt: &LinearStatement, proof: &LinearProof) -> bool {
    let (lm, lk) = match (log2_pow2(stmt.m), log2_pow2(stmt.k)) {
        (Some(a), Some(b)) => (a, b),
        _ => return false,
    };
    let arity = lm + lk;
    if whir.num_variables() != arity {
        return false;
    }
    if stmt.m.checked_mul(stmt.k).is_none() {
        return false;
    }
    // Proof-shape consistency with the op kind, BEFORE the transcript:
    // Add requires the second source AND its opening; every other kind
    // requires NEITHER. Mismatches return false — never a panic.
    match stmt.kind {
        LinearOpKind::Scale { shift, .. } => {
            if shift != 0 || stmt.root_b.is_some() || proof.open_b.is_some() {
                return false;
            }
        }
        LinearOpKind::Transpose { m: tm, k: tk } => {
            if tm != stmt.m || tk != stmt.k || stmt.root_b.is_some() || proof.open_b.is_some() {
                return false;
            }
        }
        LinearOpKind::Add => {
            if stmt.root_b.is_none() || proof.open_b.is_none() {
                return false;
            }
        }
        LinearOpKind::AddConst { .. } => {
            if stmt.root_b.is_some() || proof.open_b.is_some() {
                return false;
            }
        }
    }

    let roots: Vec<&Commitment> = match &stmt.root_b {
        Some(rb) => vec![&stmt.root_a, rb, &stmt.root_out],
        None => vec![&stmt.root_a, &stmt.root_out],
    };
    let mut t = ETranscript::new(stmt.kind.label(), &[stmt.m, stmt.k], &roots);
    match stmt.kind {
        LinearOpKind::Scale { factor, shift } => {
            t.absorb_base(Goldilocks::from_u64(factor));
            t.absorb_base(Goldilocks::from_u64(shift as u64));
        }
        LinearOpKind::AddConst { c } => t.absorb_base(c),
        _ => {}
    }

    let out_point = t.sample_vec(arity);
    let y_claim = proof.open_out.1;
    t.absorb(y_claim);

    // WHIR-authenticate the output and the sources against the statement
    // roots; the verified values must equal the claimed values.
    let proto = whir.opening_protocol(arity, 1);
    let y_verified = match whir
        .verify_ef(&stmt.root_out, &proof.open_out.0, &proto, &to_p3_point(&out_point))
    {
        Ok(e) => e,
        Err(_) => return false,
    };
    if y_verified != y_claim {
        return false;
    }

    let a_point = input_point(&stmt.kind, &out_point, lm);
    let a_verified = match whir.verify_ef(&stmt.root_a, &proof.open_a.0, &proto, &to_p3_point(&a_point))
    {
        Ok(e) => e,
        Err(_) => return false,
    };
    let a_claim = proof.open_a.1;
    if a_verified != a_claim {
        return false;
    }
    t.absorb(a_claim);
    let b_verified = match &proof.open_b {
        Some(b_open) => {
            let root_b = match &stmt.root_b {
                Some(r) => r,
                // Guarded above; defense in depth, no unwrap.
                None => return false,
            };
            let bv = match whir
                .verify_ef(root_b, &b_open.0, &proto, &to_p3_point(&out_point))
            {
                Ok(e) => e,
                Err(_) => return false,
            };
            if bv != b_open.1 {
                return false;
            }
            t.absorb(b_open.1);
            Some(bv)
        }
        None => None,
    };

    // The linear relation, checked exactly in the extension field.
    match stmt.kind {
        LinearOpKind::Scale { factor, .. } => {
            let f = EF::from(Goldilocks::from_u64(factor));
            y_claim == a_verified * f
        }
        LinearOpKind::AddConst { c } => y_claim == a_verified + EF::from(c),
        LinearOpKind::Add => match b_verified {
            Some(bv) => y_claim == a_verified + bv,
            None => false,
        },
        LinearOpKind::Transpose { .. } => y_claim == a_verified,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_core::common::field::{BasedVectorSpace, XorShift64};

    fn fixture(
        kind: LinearOpKind,
        m: usize,
        k: usize,
        rng: &mut XorShift64,
    ) -> (Whir, LinearStatement, LinearProof, Vec<Vec<Goldilocks>>) {
        let arity = (m * k).trailing_zeros() as usize;
        // PCS target 90 bits, PoW budget 0 (correctness prototype).
        let whir = Whir::new_target(arity, 90, 0).expect("valid arity");
        let a: Vec<Goldilocks> = (0..m * k).map(|_| rng.field()).collect();
        let (b, out) = match kind {
            LinearOpKind::Scale { factor, .. } => {
                let out: Vec<Goldilocks> = a.iter().map(|&v| v * Goldilocks::from_u64(factor)).collect();
                (None, out)
            }
            LinearOpKind::Add => {
                let b: Vec<Goldilocks> = (0..m * k).map(|_| rng.field()).collect();
                let out: Vec<Goldilocks> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();
                (Some(b), out)
            }
            LinearOpKind::AddConst { c } => {
                let out: Vec<Goldilocks> = a.iter().map(|&v| v + c).collect();
                (None, out)
            }
            LinearOpKind::Transpose { .. } => {
                let mut out = vec![Goldilocks::ZERO; k * m];
                for i in 0..m {
                    for kk in 0..k {
                        out[kk * m + i] = a[i * k + kk];
                    }
                }
                (None, out)
            }
        };
        let (stmt, proof) =
            prove(&whir, kind, m, k, &a, b.as_deref(), &out).expect("honest prove");
        let mut tensors = vec![a, out];
        if let Some(bb) = b {
            tensors.insert(1, bb);
        }
        (whir, stmt, proof, tensors)
    }

    #[test]
    fn honest_roundtrip_all_kinds() {
        let mut rng = XorShift64::new(0xB1B1);
        for kind in [
            LinearOpKind::Scale { factor: 7, shift: 0 },
            LinearOpKind::Add,
            LinearOpKind::AddConst { c: Goldilocks::from_u64(123) },
            LinearOpKind::Transpose { m: 4, k: 8 },
        ] {
            let (whir, stmt, proof, _) = fixture(kind, 4, 8, &mut rng);
            assert!(verify(&whir, &stmt, &proof), "{:?}", kind);
        }
    }

    #[test]
    fn output_point_is_genuinely_extension() {
        let mut rng = XorShift64::new(0xB1B2);
        let (whir, stmt, proof, _) = fixture(LinearOpKind::Add, 4, 8, &mut rng);
        assert!(verify(&whir, &stmt, &proof));
        // Replay the transcript to recover the output point and check its
        // extension coordinate is nonzero.
        let roots: Vec<&Commitment> = vec![&stmt.root_a, stmt.root_b.as_ref().unwrap(), &stmt.root_out];
        let mut t = ETranscript::new(stmt.kind.label(), &[stmt.m, stmt.k], &roots);
        let arity = (stmt.m * stmt.k).trailing_zeros() as usize;
        let p = t.sample_vec(arity).pop().unwrap();
        let coeffs: &[Goldilocks] = p.as_basis_coefficients_slice();
        assert_ne!(coeffs[1], Goldilocks::ZERO);
    }

    #[test]
    fn claims_match_independent_reference_eval() {
        let mut rng = XorShift64::new(0xB1B3);
        let (whir, stmt, proof, tensors) = fixture(LinearOpKind::Transpose { m: 4, k: 8 }, 4, 8, &mut rng);
        assert!(verify(&whir, &stmt, &proof));
        let (a, out) = (&tensors[0], &tensors[1]);
        let roots: Vec<&Commitment> = vec![&stmt.root_a, &stmt.root_out];
        let mut t = ETranscript::new(stmt.kind.label(), &[stmt.m, stmt.k], &roots);
        let arity = (stmt.m * stmt.k).trailing_zeros() as usize;
        let lm = stmt.m.trailing_zeros() as usize;
        let out_point = t.sample_vec(arity);
        assert_eq!(proof.open_out.1, mle::eval_ef(out, &out_point));
        // input point = [k-part, m-part] of out_point
        let mut a_pt = out_point[lm..].to_vec();
        a_pt.extend_from_slice(&out_point[..lm]);
        assert_eq!(proof.open_a.1, mle::eval_ef(a, &a_pt));
    }

    #[test]
    fn verify_never_opens() {
        let mut rng = XorShift64::new(0xB1B4);
        let (whir, stmt, proof, _) = fixture(LinearOpKind::Add, 4, 8, &mut rng);
        let before = whir.open_stats();
        assert!(verify(&whir, &stmt, &proof));
        assert_eq!(whir.open_stats(), before, "verifier must never open");
    }

    #[test]
    fn tampered_output_claim_rejected() {
        let mut rng = XorShift64::new(0xB1B5);
        let (whir, stmt, mut proof, _) = fixture(LinearOpKind::Add, 4, 8, &mut rng);
        proof.open_out.1 = proof.open_out.1 + EF::ONE;
        assert!(!verify(&whir, &stmt, &proof));
    }

    #[test]
    fn tampered_input_claim_rejected() {
        let mut rng = XorShift64::new(0xB1B6);
        let (whir, stmt, mut proof, _) = fixture(LinearOpKind::Scale { factor: 7, shift: 0 }, 4, 8, &mut rng);
        proof.open_a.1 = proof.open_a.1 + EF::ONE;
        assert!(!verify(&whir, &stmt, &proof));
    }

    #[test]
    fn tampered_root_rejected() {
        let mut rng = XorShift64::new(0xB1B7);
        let (whir, stmt, proof, tensors) = fixture(LinearOpKind::Add, 4, 8, &mut rng);
        let fake: Vec<Goldilocks> = (0..tensors[0].len()).map(|_| rng.field()).collect();
        let (fake_root, _, _) = whir.commit(&fake);
        let mut bad = stmt.clone();
        bad.root_a = fake_root;
        assert!(!verify(&whir, &bad, &proof));
    }

    #[test]
    fn tampered_statement_scalar_rejected() {
        let mut rng = XorShift64::new(0xB1B8);
        let (whir, stmt, proof, _) = fixture(LinearOpKind::Scale { factor: 7, shift: 0 }, 4, 8, &mut rng);
        let mut bad = stmt.clone();
        bad.kind = LinearOpKind::Scale { factor: 8, shift: 0 };
        assert!(!verify(&whir, &bad, &proof));
    }

    #[test]
    fn swapped_transpose_shape_rejected() {
        let mut rng = XorShift64::new(0xB1B9);
        let (whir, stmt, proof, _) = fixture(LinearOpKind::Transpose { m: 4, k: 8 }, 4, 8, &mut rng);
        let mut bad = stmt.clone();
        bad.kind = LinearOpKind::Transpose { m: 8, k: 4 };
        assert!(!verify(&whir, &bad, &proof));
    }

    #[test]
    fn rounded_scale_rejected() {
        let mut rng = XorShift64::new(0xB1BA);
        let whir = Whir::new_target(5, 90, 0).expect("valid arity");
        let a: Vec<Goldilocks> = (0..32).map(|_| rng.field()).collect();
        let out = a.clone();
        assert!(prove(&whir, LinearOpKind::Scale { factor: 23170, shift: 16 }, 4, 8, &a, None, &out).is_none());
    }

    #[test]
    fn malformed_shapes_and_missing_source_rejected() {
        let mut rng = XorShift64::new(0xB1BB);
        let (whir, stmt, proof, _) = fixture(LinearOpKind::Add, 4, 8, &mut rng);
        // huge dims
        let mut bad = stmt.clone();
        bad.m = 1usize << (usize::BITS - 1);
        assert!(!verify(&whir, &bad, &proof));
        // missing second source for Add
        let mut bad2 = stmt.clone();
        bad2.root_b = None;
        assert!(!verify(&whir, &bad2, &proof));
        // huge dims in the constructor path (prove)
        let whir_big = Whir::new_testing(5);
        let a: Vec<Goldilocks> = (0..32).map(|_| rng.field()).collect();
        assert!(prove(&whir_big, LinearOpKind::Add, 1usize << (usize::BITS - 1), 8, &a, Some(&a), &a).is_none());
    }

    /// An attacker-supplied second opening on a NON-Add kind must return
    /// `false`, never a panic (the `root_b` unwrap is guarded).
    #[test]
    fn unexpected_second_opening_rejected_for_every_non_add_kind() {
        let mut rng = XorShift64::new(0xB1BC);
        for kind in [
            LinearOpKind::Scale { factor: 7, shift: 0 },
            LinearOpKind::AddConst { c: Goldilocks::from_u64(123) },
            LinearOpKind::Transpose { m: 4, k: 8 },
        ] {
            let (whir, stmt, mut proof, _) = fixture(kind, 4, 8, &mut rng);
            proof.open_b = Some(proof.open_a.clone());
            assert!(!verify(&whir, &stmt, &proof), "{:?}", kind);
        }
    }

    /// An Add proof missing the second opening must return `false`.
    #[test]
    fn add_missing_second_opening_rejected() {
        let mut rng = XorShift64::new(0xB1BD);
        let (whir, stmt, mut proof, _) = fixture(LinearOpKind::Add, 4, 8, &mut rng);
        proof.open_b = None;
        assert!(!verify(&whir, &stmt, &proof));
    }
}
