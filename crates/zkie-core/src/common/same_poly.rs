//! same_poly: prove a set of claims (r_i, y_i) are all evaluations of the SAME
//! MLE f (f(r_i)=y_i), and merge them into one claim at a fresh point. This is
//! the claim-merging primitive for reduction chaining (DeepProve iop/same_poly).
//!
//! S1: the batch coefficients are squeezed from the transcript (binding the
//! statement and prior messages), and the merged point is derived per-round
//! from the virtual sumcheck's own round messages.

use crate::common::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64, ChallengeSource};
use crate::common::sumcheck::{prove_virtual, verify_virtual, VirtualProof};
use crate::common::transcript::Transcript;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SamePolyProof {
    /// The batch coefficient base (S4): the coefficients are the POWERS
    /// 1, gamma, gamma^2, ... derived by the verifier from this single
    /// transcript squeeze — never a prover-chosen list.
    pub gamma: Goldilocks,
    pub proof: VirtualProof,
    pub merged_point: Vec<Goldilocks>,
}

fn absorb_claims(transcript: &mut Transcript, claims: &[(Vec<Goldilocks>, Goldilocks)]) {
    transcript.absorb(b"zkie-same-poly-claims-v1");
    transcript.absorb_u64(claims.len() as u64);
    for (point, value) in claims {
        transcript.absorb_u64(point.len() as u64);
        transcript.absorb_fields(point);
        transcript.absorb_field(*value);
    }
}

/// Prove a set of claims (r_i, y_i) are all evaluations of the SAME MLE f
/// (f(r_i) = y_i), and merge them into one claim at the sumcheck-derived
/// point r*. S4: the batch coefficients are powers of ONE transcript gamma;
/// the merged VALUE is returned alongside the proof (it is the sumcheck's
/// authenticated f(r*) — no independent stored field).
pub fn prove_same_poly(
    f: &[Goldilocks],
    claims: &[(Vec<Goldilocks>, Goldilocks)],
    transcript: &mut Transcript,
) -> (SamePolyProof, Goldilocks) {
    assert!(!claims.is_empty(), "same_poly needs at least one claim");
    let t = f.len().trailing_zeros() as usize;
    for (r_i, _) in claims {
        assert_eq!(r_i.len(), t);
    }
    absorb_claims(transcript, claims);
    let gamma = transcript.field();
    let mut coeffs = Vec::with_capacity(claims.len());
    let mut acc = Goldilocks::ONE;
    for _ in claims {
        coeffs.push(acc);
        acc = acc * gamma;
    }
    let eqs: Vec<Vec<Goldilocks>> = claims
        .iter()
        .map(|(r_i, _)| crate::common::mle::eq_evals(r_i))
        .collect();
    let mut terms: Vec<(Goldilocks, Vec<usize>)> = Vec::new();
    let mut claimed = Goldilocks::from_u64(0);
    for (i, (_, y_i)) in claims.iter().enumerate() {
        terms.push((coeffs[i], vec![0usize, i + 1]));
        claimed = claimed + coeffs[i] * *y_i;
    }
    let mut mles: Vec<&[Goldilocks]> = vec![f];
    for e in &eqs {
        mles.push(e.as_slice());
    }
    let (proof, merged_point) = prove_virtual(&mles, &terms, claimed, transcript);
    let merged_eval = proof.final_evals[0];
    (SamePolyProof { gamma, proof, merged_point }, merged_eval)
}

/// Verify the S4 merge argument and return the ONE authenticated output
/// claim `(r*, f(r*))`. The verifier derives the batch coefficients as
/// powers of the transcript gamma, checks every claim point dimension,
/// rejects same-point contradictory values (fast check), and computes the
/// endpoint W(r*) = sum_i gamma^i * eq(r_i, r*) in closed form — the merged
/// value is final_claim / W(r*), with W(r*) == 0 rejected. The output claim
/// must be consumed downstream (reduced to a producing op or opened against
/// the PCS) — the argument itself carries no independent merged_eval field.
pub fn verify_same_poly(
    proof: &SamePolyProof,
    claims: &[(Vec<Goldilocks>, Goldilocks)],
    transcript: &mut Transcript,
) -> Option<(Vec<Goldilocks>, Goldilocks)> {
    if claims.is_empty() {
        return None;
    }
    // Dimension binding: every claim point must share the tensor arity.
    let t = claims[0].0.len();
    for (r_i, _) in claims {
        if r_i.len() != t {
            return None;
        }
    }
    // The coefficients are gamma powers derived from the transcript.
    absorb_claims(transcript, claims);
    let gamma = transcript.field();
    if gamma != proof.gamma {
        return None;
    }
    if gamma == Goldilocks::ZERO || gamma == Goldilocks::ONE {
        return None; // degenerate powers: the batch equation loses soundness
    }
    let mut coeffs = Vec::with_capacity(claims.len());
    let mut acc = Goldilocks::ONE;
    for _ in claims {
        coeffs.push(acc);
        acc = acc * gamma;
    }
    // Fast check (S4): contradictory values at the SAME point can never be
    // evaluations of one MLE — reject outright.
    for i in 0..claims.len() {
        for j in (i + 1)..claims.len() {
            if claims[i].0 == claims[j].0 && claims[i].1 != claims[j].1 {
                return None;
            }
        }
    }
    let mut terms: Vec<(Goldilocks, Vec<usize>)> = Vec::new();
    let mut claimed = Goldilocks::from_u64(0);
    for (i, (_, y_i)) in claims.iter().enumerate() {
        terms.push((coeffs[i], vec![0usize, i + 1]));
        claimed = claimed + coeffs[i] * *y_i;
    }
    let (ok, r, final_claim) = verify_virtual(&proof.proof, &terms, claimed, t, transcript);
    if !ok || r != proof.merged_point {
        return None;
    }
    // The endpoint weight is computed by the verifier in closed form — the
    // proof's stored eq evals are never consulted.
    let mut w = Goldilocks::ZERO;
    for (i, (r_i, _)) in claims.iter().enumerate() {
        w = w + coeffs[i] * crate::common::mle::eq_poly(r_i, &r);
    }
    if w == Goldilocks::ZERO {
        return None;
    }
    Some((r, final_claim * w.inverse()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_distinct_point_claims_cannot_cancel() {
        let f = vec![Goldilocks::from_u64(3), Goldilocks::from_u64(7)];
        let p0 = vec![Goldilocks::ZERO];
        let p1 = vec![Goldilocks::ONE];
        let mut peek = Transcript::new(b"adaptive-claims");
        let gamma = peek.field();
        let claims = vec![(p0, f[0] + Goldilocks::ONE),
                          (p1, f[1] - gamma.inverse())];
        let mut prover = Transcript::new(b"adaptive-claims");
        let (proof, _) = prove_same_poly(&f, &claims, &mut prover);
        let mut verifier = Transcript::new(b"adaptive-claims");
        if let Some((point, value)) = verify_same_poly(&proof, &claims, &mut verifier) {
            assert_ne!(value, crate::common::mle::eval(&f, &point),
                       "forged claims passed with a genuine endpoint");
        }
    }

    #[test]
    fn same_poly_roundtrip() {
        let mut rng = XorShift64::new(0x5A1E);
        let t = 6usize;
        let n = 1usize << t;
        let f: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let m = 3usize;
        let claims: Vec<(Vec<Goldilocks>, Goldilocks)> = (0..m)
            .map(|_| {
                let r_i: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
                let y_i = crate::common::mle::eval(&f, &r_i);
                (r_i, y_i)
            })
            .collect();
        let mut tp = Transcript::new(b"zkie-same-poly-test");
        tp.absorb_u64(0x5A1E);
        let (proof, _merged_eval) = prove_same_poly(&f, &claims, &mut tp);
        let mut tv = Transcript::new(b"zkie-same-poly-test");
        tv.absorb_u64(0x5A1E);
        let merged = verify_same_poly(&proof, &claims, &mut tv);
        assert!(merged.is_some());
        let (r_star, v) = merged.unwrap();
        assert_eq!(r_star, proof.merged_point);
        assert_eq!(v, crate::common::mle::eval(&f, &r_star));
        let mut bad = claims.clone();
        bad[0].1 = bad[0].1 + Goldilocks::from_u64(1);
        let mut tv2 = Transcript::new(b"zkie-same-poly-test");
        tv2.absorb_u64(0x5A1E);
        assert!(verify_same_poly(&proof, &bad, &mut tv2).is_none());
    }

    // ------------------------------------------------------------------
    // S4 (same_poly merge argument) test-first regressions.
    // ------------------------------------------------------------------

    fn sp_transcript(seed: u64) -> Transcript {
        let mut t = Transcript::new(b"zkie-same-poly-test");
        t.absorb_u64(seed);
        t
    }

    /// S4: honest merge at distinct points roundtrips; the returned merged
    /// claim equals the tensor's MLE at the merged point.
    #[test]
    fn s4_honest_distinct_points_merge_roundtrip() {
        let mut rng = XorShift64::new(0x5401);
        let t = 6usize;
        let n = 1usize << t;
        let f: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let m = 3usize;
        let claims: Vec<(Vec<Goldilocks>, Goldilocks)> = (0..m)
            .map(|_| {
                let r_i: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
                let y_i = crate::common::mle::eval(&f, &r_i);
                (r_i, y_i)
            })
            .collect();
        let mut tp = sp_transcript(0x5401);
        let (proof, _merged_eval) = prove_same_poly(&f, &claims, &mut tp);
        let mut tv = sp_transcript(0x5401);
        let merged = verify_same_poly(&proof, &claims, &mut tv);
        assert!(merged.is_some());
        let (r_star, v) = merged.unwrap();
        assert_eq!(r_star, proof.merged_point);
        assert_eq!(v, crate::common::mle::eval(&f, &r_star));
    }

    /// S4: contradictory claims at the SAME point must be rejected. A
    /// malicious prover adapts the tensor (f'(r) = (c0*y1 + c1*y2)/(c0+c1))
    /// so the pre-fix argument passes with both claims accepted; the fix
    /// rejects the same-point contradiction outright (fast check).
    #[test]
    fn s4_contradictory_same_point_rejected() {
        let mut rng = XorShift64::new(0x5402);
        let t = 6usize;
        let n = 1usize << t;
        let f: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let r: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
        let y1 = Goldilocks::from_u64(1);
        let y2 = Goldilocks::from_u64(2);
        let claims = vec![(r.clone(), y1), (r.clone(), y2)];
        // The verifier derives the batch coefficients itself (powers of
        // one gamma); a malicious prover adapts the tensor to the SAME
        // coefficients, so the pre-fix argument passes with both
        // contradictory claims accepted. Build the proof manually with the
        // verifier's gamma (prove_same_poly would squeeze its own).
        let mut tp = sp_transcript(0x5402);
        let gamma = tp.field();
        let c0 = Goldilocks::ONE;
        let c1 = gamma;
        let blended = (c0 * y1 + c1 * y2) * (c0 + c1).inverse();
        let eqr = crate::common::mle::eq_evals(&r);
        // f' = f + delta*eq(r, .) satisfies f'(r) = blended iff
        // delta = (blended - f(r)) / eq(r, r) — note eq(r, r) != 1 for a
        // random field point.
        let eq_rr: Goldilocks = eqr.iter().map(|&e| e * e).sum();
        let delta = (blended - crate::common::mle::eval(&f, &r)) * eq_rr.inverse();
        let f_prime: Vec<Goldilocks> = f
            .iter()
            .zip(&eqr)
            .map(|(&fi, &ei)| fi + delta * ei)
            .collect();
        let terms = vec![
            (c0, vec![0usize, 1usize]),
            (c1, vec![0usize, 2usize]),
        ];
        let claimed = c0 * y1 + c1 * y2;
        let mles: Vec<&[Goldilocks]> = vec![&f_prime, &eqr, &eqr];
        let (vp, merged_point) = prove_virtual(&mles, &terms, claimed, &mut tp);
        let proof = SamePolyProof {
            gamma,
            proof: vp,
            merged_point,
        };
        // Pre-fix this was ACCEPTED (the adapted tensor satisfies the batch
        // equation). S4: the same-point contradiction must be rejected.
        let mut tv2 = sp_transcript(0x5402);
        assert!(verify_same_poly(&proof, &claims, &mut tv2).is_none());
    }

    /// S4: the merged evaluation used to be an INDEPENDENT prover field —
    /// tampering it was accepted and the forged value returned, so the merged
    /// claim would fail to match any real opening of the tensor. The field no
    /// longer exists: the returned merged claim is derived from the argument
    /// and MUST equal a real evaluation/opening of the tensor at the merged
    /// point (this assert FAILS pre-fix, where the forged value is returned).
    #[test]
    fn s4_merged_eval_forgery_rejected() {
        let mut rng = XorShift64::new(0x5403);
        let t = 6usize;
        let n = 1usize << t;
        let f: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let claims: Vec<(Vec<Goldilocks>, Goldilocks)> = (0..3)
            .map(|_| {
                let r_i: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
                let y_i = crate::common::mle::eval(&f, &r_i);
                (r_i, y_i)
            })
            .collect();
        let mut tp = sp_transcript(0x5403);
        let (proof, _merged_eval) = prove_same_poly(&f, &claims, &mut tp);
        let mut tv = sp_transcript(0x5403);
        let (r_star, v) = verify_same_poly(&proof, &claims, &mut tv).unwrap();
        assert_eq!(v, crate::common::mle::eval(&f, &r_star));
        assert_eq!(r_star, proof.merged_point);
    }

    /// S4 guard: tampered batch coefficients are rejected (they are
    /// transcript-derived and compared).
    #[test]
    fn s4_coeffs_tamper_rejected() {
        let mut rng = XorShift64::new(0x5404);
        let t = 6usize;
        let n = 1usize << t;
        let f: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let claims: Vec<(Vec<Goldilocks>, Goldilocks)> = (0..3)
            .map(|_| {
                let r_i: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
                let y_i = crate::common::mle::eval(&f, &r_i);
                (r_i, y_i)
            })
            .collect();
        let mut tp = sp_transcript(0x5404);
        let (mut proof, _merged_eval) = prove_same_poly(&f, &claims, &mut tp);
        proof.gamma = proof.gamma + Goldilocks::from_u64(1);
        let mut tv = sp_transcript(0x5404);
        assert!(verify_same_poly(&proof, &claims, &mut tv).is_none());
    }

    /// S4 guard: a tampered merged point is rejected (it must equal the
    /// sumcheck's derived challenge vector).
    #[test]
    fn s4_merged_point_forgery_rejected() {
        let mut rng = XorShift64::new(0x5405);
        let t = 6usize;
        let n = 1usize << t;
        let f: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let claims: Vec<(Vec<Goldilocks>, Goldilocks)> = (0..3)
            .map(|_| {
                let r_i: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
                let y_i = crate::common::mle::eval(&f, &r_i);
                (r_i, y_i)
            })
            .collect();
        let mut tp = sp_transcript(0x5405);
        let (mut proof, _merged_eval) = prove_same_poly(&f, &claims, &mut tp);
        proof.merged_point[0] = proof.merged_point[0] + Goldilocks::from_u64(1);
        let mut tv = sp_transcript(0x5405);
        assert!(verify_same_poly(&proof, &claims, &mut tv).is_none());
    }

    /// S4: the eq evaluations inside the proof's final_evals are NOT
    /// consulted — the merged value is derived from the sumcheck's final
    /// claim and the verifier-computed W(r*). Tampering them must not change
    /// the returned (authenticated) merged value. (Pre-fix the stored-eq
    /// check wrongly REJECTED such a tampered proof — red.)
    #[test]
    fn s4_eq_evals_not_consulted() {
        let mut rng = XorShift64::new(0x5406);
        let t = 6usize;
        let n = 1usize << t;
        let f: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let claims: Vec<(Vec<Goldilocks>, Goldilocks)> = (0..3)
            .map(|_| {
                let r_i: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
                let y_i = crate::common::mle::eval(&f, &r_i);
                (r_i, y_i)
            })
            .collect();
        let mut tp = sp_transcript(0x5406);
        let (mut proof, _merged_eval) = prove_same_poly(&f, &claims, &mut tp);
        // Corrupt every stored final eval arbitrarily: the derived-value
        // design never reads them.
        for fe in proof.proof.final_evals.iter_mut() {
            *fe = Goldilocks::ZERO;
        }
        let mut tv = sp_transcript(0x5406);
        let got = verify_same_poly(&proof, &claims, &mut tv);
        assert!(got.is_some());
        let (r_star, v) = got.unwrap();
        assert_eq!(v, crate::common::mle::eval(&f, &r_star));
    }
}
