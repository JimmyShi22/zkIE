//! Sum-check protocol for a product of two multilinear polynomials.
//!
//! Proves `H = sum_{x in {0,1}^t} f(x) * h(x)`. Each round polynomial has
//! degree at most two, so the prover sends three coefficients per round.

use rayon::prelude::*;
use crate::common::field::{Field, Goldilocks, PrimeCharacteristicRing};
use crate::common::fixed_point::from_i64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoundPoly {
    pub c0: Goldilocks,
    pub c1: Goldilocks,
    pub c2: Goldilocks,
}

impl RoundPoly {
    #[inline]
    pub fn eval(&self, x: Goldilocks) -> Goldilocks {
        self.c0 + self.c1 * x + self.c2 * x * x
    }
}

#[derive(Clone, Debug)]
pub struct SumcheckProof {
    pub rounds: Vec<RoundPoly>,
    pub f_eval: Goldilocks,
    pub h_eval: Goldilocks,
}

const INV2: Goldilocks = Goldilocks::new(crate::common::field::P / 2 + 1); // (p + 1) / 2

pub fn prove(f: &[Goldilocks], h: &[Goldilocks], _claimed_sum: Goldilocks, #[allow(unused_variables)] challenges: &[Goldilocks]) -> SumcheckProof {
    // Base-field instantiation of the shared generic round circuit
    // (`product_round_f` / `fold_f`): inline loop, no intermediate
    // `RoundPolyF` vector conversion allocated.
    let t = f.len().trailing_zeros() as usize;
    assert_eq!(f.len(), 1 << t, "f length must be a power of two");
    assert_eq!(h.len(), f.len());
    assert_eq!(challenges.len(), t);

    let mut f_buf = f.to_vec();
    let mut h_buf = h.to_vec();
    let mut rounds = Vec::with_capacity(t);
    for &r in challenges {
        let rp = product_round_f(&f_buf, &h_buf);
        rounds.push(RoundPoly { c0: rp.c0, c1: rp.c1, c2: rp.c2 });
        fold_f(&mut f_buf, r);
        fold_f(&mut h_buf, r);
    }
    SumcheckProof {
        rounds,
        f_eval: f_buf[0],
        h_eval: h_buf[0],
    }
}

fn fold(buf: &mut Vec<Goldilocks>, p: Goldilocks) {
    let half = buf.len() / 2;
    for i in 0..half {
        let a = buf[2 * i];
        let b = buf[2 * i + 1];
        buf[i] = a + p * (b - a);
    }
    buf.truncate(half);
}

pub fn verify(
    proof: &SumcheckProof,
    claimed_sum: Goldilocks, #[allow(unused_variables)]
    challenges: &[Goldilocks],
    f_eval: Goldilocks,
    h_eval: Goldilocks,
) -> bool {
    // Inline loop — no round-vector conversion allocation.
    if proof.rounds.len() != challenges.len() {
        return false;
    }
    let mut prev = claimed_sum;
    for (rp, &r) in proof.rounds.iter().zip(challenges.iter()) {
        let p0 = rp.c0;
        let p1 = rp.c0 + rp.c1 + rp.c2;
        if p0 + p1 != prev {
            return false;
        }
        prev = rp.eval(r);
    }
    prev == f_eval * h_eval
}

// ==== generic-field product-sumcheck round algebra ====
//
// The same degree-2 round polynomial for `sum_x f(x) h(x)` instantiated over
// ANY field `F`. The Goldilocks entry points above are the base instantiation;
// the EF extension chain instantiates these with `F = EF`. One circuit, two
// fields — no duplicated round algebra.

/// Degree-2 round polynomial over an arbitrary field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoundPolyF<F> {
    pub c0: F,
    pub c1: F,
    pub c2: F,
}

impl<F: Field + PrimeCharacteristicRing + Copy> RoundPolyF<F> {
    #[inline]
    pub fn eval(&self, x: F) -> F {
        self.c0 + self.c1 * x + self.c2 * x * x
    }
}

/// Compute the degree-2 round polynomial of `sum f(x) h(x)` over the current
/// (partially folded) buffers, for any field `F`.
pub fn product_round_f<F: Field + PrimeCharacteristicRing + Copy>(f: &[F], h: &[F]) -> RoundPolyF<F> {
    // Direct coefficient formula for the degree-2 round polynomial of
    // `sum f(x) h(x)` — no inverse, no per-round division:
    //   c0 = sum a0*b0
    //   c1 = sum a0*(b1-b0) + (a1-a0)*b0
    //   c2 = sum (a1-a0)*(b1-b0)
    // which is the Lagrange interpolant through q(0), q(1), q(2).
    let half = f.len() / 2;
    let mut c0 = F::ZERO;
    let mut c1 = F::ZERO;
    let mut c2 = F::ZERO;
    for s in 0..half {
        let a0 = f[2 * s];
        let a1 = f[2 * s + 1];
        let b0 = h[2 * s];
        let b1 = h[2 * s + 1];
        let da = a1 - a0;
        let db = b1 - b0;
        c0 = c0 + a0 * b0;
        c1 = c1 + a0 * db + da * b0;
        c2 = c2 + da * db;
    }
    RoundPolyF { c0, c1, c2 }
}

/// Fold a buffer by one variable at challenge `p` (any field).
pub fn fold_f<F: Field + PrimeCharacteristicRing + Copy>(buf: &mut Vec<F>, p: F) {
    let half = buf.len() / 2;
    for i in 0..half {
        let a = buf[2 * i];
        let b = buf[2 * i + 1];
        buf[i] = a + p * (b - a);
    }
    buf.truncate(half);
}

/// Run the generic product sumcheck to its terminal evals; returns
/// `(rounds, f_eval, h_eval)`.
pub fn prove_product_f<F: Field + PrimeCharacteristicRing + Copy>(
    f: &[F],
    h: &[F],
    challenges: &[F],
) -> (Vec<RoundPolyF<F>>, F, F) {
    let t = f.len().trailing_zeros() as usize;
    assert_eq!(f.len(), 1 << t, "f length must be a power of two");
    assert_eq!(h.len(), f.len());
    assert_eq!(challenges.len(), t);

    let mut f_buf = f.to_vec();
    let mut h_buf = h.to_vec();
    let mut rounds = Vec::with_capacity(t);
    for &r in challenges {
        rounds.push(product_round_f(&f_buf, &h_buf));
        fold_f(&mut f_buf, r);
        fold_f(&mut h_buf, r);
    }
    (rounds, f_buf[0], h_buf[0])
}

/// Verify the generic product sumcheck: `claimed = sum_x f(x) h(x)` with
/// terminal evals `f_eval`, `h_eval` (authenticated out-of-band, e.g. by PCS
/// openings).
pub fn verify_product_f<F: Field + PrimeCharacteristicRing + Copy>(
    rounds: &[RoundPolyF<F>],
    claimed: F,
    challenges: &[F],
    f_eval: F,
    h_eval: F,
) -> bool {
    if rounds.len() != challenges.len() {
        return false;
    }
    let mut prev = claimed;
    for (rp, &r) in rounds.iter().zip(challenges.iter()) {
        let p0 = rp.c0;
        let p1 = rp.c0 + rp.c1 + rp.c2;
        if p0 + p1 != prev {
            return false;
        }
        prev = rp.eval(r);
    }
    prev == f_eval * h_eval
}

/// A degree-3 round polynomial `c0 + c1 x + c2 x^2 + c3 x^3` for the triple
/// product sum-check `H = sum f(x) g(x) h(x)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoundPoly3 {
    pub c0: Goldilocks,
    pub c1: Goldilocks,
    pub c2: Goldilocks,
    pub c3: Goldilocks,
}

impl RoundPoly3 {
    #[inline]
    pub fn eval(&self, x: Goldilocks) -> Goldilocks {
        self.c0 + self.c1 * x + self.c2 * x * x + self.c3 * x * x * x
    }
}

#[derive(Clone, Debug)]
pub struct SumcheckProof3 {
    pub rounds: Vec<RoundPoly3>,
    pub f_eval: Goldilocks,
    pub g_eval: Goldilocks,
    pub h_eval: Goldilocks,
}

fn eval_at(f0: Goldilocks, f1: Goldilocks, t: Goldilocks) -> Goldilocks {
    f0 + (f1 - f0) * t
}

/// Interpolate the cubic `c0 + c1 t + c2 t^2 + c3 t^3` through the values at
/// `t = 0, 1, 2, 3`.
fn interpolate_cubic(p0: Goldilocks, p1: Goldilocks, p2: Goldilocks, p3: Goldilocks) -> RoundPoly3 {
    let inv2 = Goldilocks::from_u64(2).inverse();
    let inv3 = Goldilocks::from_u64(3).inverse();
    let inv6 = Goldilocks::from_u64(6).inverse();
    let d1_0 = p1 - p0;
    let d1_1 = p2 - p1;
    let d1_2 = p3 - p2;
    let d2_0 = d1_1 - d1_0;
    let d2_1 = d1_2 - d1_1;
    let d3_0 = d2_1 - d2_0;
    let c0 = p0;
    let c1 = d1_0 - d2_0 * inv2 + d3_0 * inv3;
    let c2 = d2_0 * inv2 - d3_0 * inv2;
    let c3 = d3_0 * inv6;
    RoundPoly3 { c0, c1, c2, c3 }
}

/// Sum-check for `H = sum_{x in {0,1}^t} f(x) * g(x) * h(x)`.
pub fn prove3(
    f: &[Goldilocks],
    g: &[Goldilocks],
    h: &[Goldilocks],
    _claimed_sum: Goldilocks, #[allow(unused_variables)]
    challenges: &[Goldilocks],
) -> SumcheckProof3 {
    let t = f.len().trailing_zeros() as usize;
    assert_eq!(f.len(), 1 << t, "f length must be a power of two");
    assert_eq!(g.len(), f.len());
    assert_eq!(h.len(), f.len());
    assert_eq!(challenges.len(), t);

    let mut f_buf = f.to_vec();
    let mut g_buf = g.to_vec();
    let mut h_buf = h.to_vec();
    let mut rounds = Vec::with_capacity(t);

    for r in challenges.iter().copied() {
        let half = f_buf.len() / 2;
        let mut p0 = Goldilocks::ZERO;
        let mut p1 = Goldilocks::ZERO;
        let mut p2 = Goldilocks::ZERO;
        let mut p3 = Goldilocks::ZERO;
        for s in 0..half {
            let (f0, f1) = (f_buf[2 * s], f_buf[2 * s + 1]);
            let (g0, g1) = (g_buf[2 * s], g_buf[2 * s + 1]);
            let (h0, h1) = (h_buf[2 * s], h_buf[2 * s + 1]);
            p0 = p0 + f0 * g0 * h0;
            p1 = p1 + f1 * g1 * h1;
            let two = Goldilocks::TWO;
            p2 = p2 + eval_at(f0, f1, two) * eval_at(g0, g1, two) * eval_at(h0, h1, two);
            let three = Goldilocks::from_u64(3);
            p3 = p3 + eval_at(f0, f1, three) * eval_at(g0, g1, three) * eval_at(h0, h1, three);
        }
        rounds.push(interpolate_cubic(p0, p1, p2, p3));
        fold(&mut f_buf, r);
        fold(&mut g_buf, r);
        fold(&mut h_buf, r);
    }

    SumcheckProof3 {
        rounds,
        f_eval: f_buf[0],
        g_eval: g_buf[0],
        h_eval: h_buf[0],
    }
}

pub fn verify3(
    proof: &SumcheckProof3,
    claimed_sum: Goldilocks, #[allow(unused_variables)]
    challenges: &[Goldilocks],
    f_eval: Goldilocks,
    g_eval: Goldilocks,
    h_eval: Goldilocks,
) -> bool {
    if proof.rounds.len() != challenges.len() {
        return false;
    }
    let mut prev = claimed_sum;
    for (rp, r) in proof.rounds.iter().zip(challenges.iter().copied()) {
        let p0 = rp.c0;
        let p1 = rp.c0 + rp.c1 + rp.c2 + rp.c3;
        if p0 + p1 != prev {
            return false;
        }
        prev = rp.eval(r);
    }
    prev == f_eval * g_eval * h_eval
}


/// Sum-check proof for `H = sum_x sum_i w_i(x) * v_i(x)` (a sum of products of
/// two multilinears), used by the multi-point-to-one-point batch opening.
#[derive(Clone, Debug)]
pub struct SumcheckProofBatch {
    pub rounds: Vec<RoundPoly>,
    pub b_eval: Goldilocks,
}

/// Prove `sum_x sum_i weights[i][x] * values[i][x] == claimed_sum`.
pub fn prove_sum_of_products(
    weights: &[Vec<Goldilocks>],
    values: &[Vec<Goldilocks>],
    claimed_sum: Goldilocks, #[allow(unused_variables)]
    challenges: &[Goldilocks],
) -> SumcheckProofBatch {
    let n = weights.len();
    assert!(n > 0, "empty batch");
    assert_eq!(values.len(), n);
    let t = weights[0].len().trailing_zeros() as usize;
    assert_eq!(challenges.len(), t, "challenge count mismatch");
    let mut w_bufs = weights.to_vec();
    let mut v_bufs = values.to_vec();
    let mut rounds = Vec::with_capacity(t);
    for &r in challenges {
        let half = w_bufs[0].len() / 2;
        let mut p0 = Goldilocks::ZERO;
        let mut p1 = Goldilocks::ZERO;
        let mut p2 = Goldilocks::ZERO;
        for i in 0..n {
            let w = &w_bufs[i];
            let v = &v_bufs[i];
            for s in 0..half {
                let (w0, w1) = (w[2 * s], w[2 * s + 1]);
                let (v0, v1) = (v[2 * s], v[2 * s + 1]);
                p0 = p0 + w0 * v0;
                p1 = p1 + w1 * v1;
                let w2 = Goldilocks::TWO * w1 - w0;
                let v2 = Goldilocks::TWO * v1 - v0;
                p2 = p2 + w2 * v2;
            }
        }
        let c0 = p0;
        let c1 = (Goldilocks::from_u64(4) * p1 - p2 - Goldilocks::from_u64(3) * p0) * INV2;
        let c2 = (p2 - Goldilocks::TWO * p1 + p0) * INV2;
        rounds.push(RoundPoly { c0, c1, c2 });
        for i in 0..n {
            fold(&mut w_bufs[i], r);
            fold(&mut v_bufs[i], r);
        }
    }
    let b_eval = (0..n).fold(Goldilocks::ZERO, |acc, i| acc + w_bufs[i][0] * v_bufs[i][0]);
    SumcheckProofBatch { rounds, b_eval }
}

/// Verify the sum-of-products sumcheck. `expected_b_eval` is the independent
/// recomputation of `B(r)` from the opened `f_i(r)` values.
pub fn verify_sum_of_products(
    proof: &SumcheckProofBatch,
    claimed_sum: Goldilocks, #[allow(unused_variables)]
    challenges: &[Goldilocks],
    expected_b_eval: Goldilocks,
) -> bool {
    if proof.rounds.len() != challenges.len() {
        return false;
    }
    let mut prev = claimed_sum;
    for (rp, r) in proof.rounds.iter().zip(challenges.iter().copied()) {
        let p0 = rp.c0;
        let p1 = rp.c0 + rp.c1 + rp.c2;
        if p0 + p1 != prev {
            return false;
        }
        prev = rp.eval(r);
    }
    prev == proof.b_eval && proof.b_eval == expected_b_eval
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::field::XorShift64;

    /// The direct coefficient formula must equal the Lagrange-interpolation
    /// reference `(y0, y1, y2) -> c0, c1, c2` (with explicit inverse-of-two),
    /// and the base prove/verify must roundtrip the same values.
    #[test]
    fn direct_coefficients_match_interpolation_reference() {
        let mut rng = XorShift64::new(0x5C1);
        let t = 4;
        let f: Vec<Goldilocks> = (0..(1 << t)).map(|_| rng.field()).collect();
        let h: Vec<Goldilocks> = (0..(1 << t)).map(|_| rng.field()).collect();

        let rp = product_round_f(&f, &h);
        let half = f.len() / 2;
        let two = Goldilocks::TWO;
        let inv2 = Goldilocks::TWO.inverse();
        let mut y0 = Goldilocks::ZERO;
        let mut y1 = Goldilocks::ZERO;
        let mut y2 = Goldilocks::ZERO;
        for s in 0..half {
            let f0 = f[2 * s];
            let f1 = f[2 * s + 1];
            let h0 = h[2 * s];
            let h1 = h[2 * s + 1];
            y0 = y0 + f0 * h0;
            y1 = y1 + f1 * h1;
            let f2 = two * f1 - f0;
            let h2 = two * h1 - h0;
            y2 = y2 + f2 * h2;
        }
        let three = Goldilocks::ONE + Goldilocks::TWO;
        let four = Goldilocks::TWO + Goldilocks::TWO;
        assert_eq!(rp.c0, y0);
        assert_eq!(rp.c1, (four * y1 - y2 - three * y0) * inv2);
        assert_eq!(rp.c2, (y2 - two * y1 + y0) * inv2);

        // Roundtrip through the base API with the same tables.
        let true_sum: Goldilocks = f.iter().zip(&h).fold(Goldilocks::ZERO, |acc, (&a, &b)| acc + a * b);
        let challenges: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
        let proof = prove(&f, &h, true_sum, &challenges);
        assert!(verify(&proof, true_sum, &challenges, proof.f_eval, proof.h_eval));
    }

    #[test]
    fn sumcheck_completeness_and_soundness() {
        let mut rng = XorShift64::new(3);
        let t = 8;
        let f: Vec<Goldilocks> = (0..(1 << t)).map(|_| rng.field()).collect();
        let h: Vec<Goldilocks> = (0..(1 << t)).map(|_| rng.field()).collect();
        let true_sum: Goldilocks = f.iter().zip(&h).fold(Goldilocks::ZERO, |acc, (&a, &b)| acc + a * b);
        let challenges: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();

        let proof = prove(&f, &h, true_sum, &challenges);
        let f_eval = crate::common::mle::eval(&f, &challenges);
        let h_eval = crate::common::mle::eval(&h, &challenges);
        assert!(verify(&proof, true_sum, &challenges, f_eval, h_eval));

        let wrong = true_sum + Goldilocks::ONE;
        assert!(!verify(&proof, wrong, &challenges, f_eval, h_eval));
    }

    #[test]
    fn sumcheck3_completeness_and_soundness() {
        let mut rng = XorShift64::new(4);
        let t = 8;
        let f: Vec<Goldilocks> = (0..(1 << t)).map(|_| rng.field()).collect();
        let g: Vec<Goldilocks> = (0..(1 << t)).map(|_| rng.field()).collect();
        let h: Vec<Goldilocks> = (0..(1 << t)).map(|_| rng.field()).collect();
        let true_sum: Goldilocks = f
            .iter()
            .zip(&g)
            .zip(&h)
            .fold(Goldilocks::ZERO, |acc, ((&a, &b), &c)| acc + a * b * c);
        let challenges: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();

        let proof = prove3(&f, &g, &h, true_sum, &challenges);
        let f_eval = crate::common::mle::eval(&f, &challenges);
        let g_eval = crate::common::mle::eval(&g, &challenges);
        let h_eval = crate::common::mle::eval(&h, &challenges);
        assert!(verify3(&proof, true_sum, &challenges, f_eval, g_eval, h_eval));

        let wrong = true_sum + Goldilocks::ONE;
        assert!(!verify3(&proof, wrong, &challenges, f_eval, g_eval, h_eval));
    }
}

// ==== virtual-polynomial sumcheck (sum of products of MLEs) ====
/// A sum of products of multilinear polynomials: g(x) = sum_i coeff_i * prod_{j in term_i} f_j(x).
/// Proves sum_{x in {0,1}^t} g(x) = claimed with a single sumcheck whose round
/// polynomial has degree max_i |term_i|.
pub struct VirtualProof {
    pub rounds: Vec<Vec<Goldilocks>>,
    pub final_evals: Vec<Goldilocks>,
}
pub fn prove_virtual(
    mles: &[&[Goldilocks]],
    terms: &[(Goldilocks, Vec<usize>)],
    claimed: Goldilocks,
    challenges: &[Goldilocks],
) -> VirtualProof {
    let n = mles[0].len();
    let t = n.trailing_zeros() as usize;
    assert!(n.is_power_of_two());
    for m in mles {
        assert_eq!(m.len(), n);
    }
    assert_eq!(challenges.len(), t);
    let max_deg = terms.iter().map(|(_, idxs)| idxs.len()).max().unwrap_or(0);
    let mut bufs: Vec<Vec<Goldilocks>> = mles.iter().map(|m| m.to_vec()).collect();
    let mut rounds = Vec::with_capacity(t);
    for &r in challenges {
        let half = bufs[0].len() / 2;
        let pvals: Vec<Goldilocks> = (0..half)
            .into_par_iter()
            .map(|s| {
                let mut p = vec![Goldilocks::ZERO; max_deg + 1];
                for (coeff, idxs) in terms {
                    for k in 0..=max_deg {
                        let mut prod = *coeff;
                        for &j in idxs {
                            let f0 = bufs[j][2 * s];
                            let f1 = bufs[j][2 * s + 1];
                            let fk = from_i64(k as i64) * f1 - from_i64(k as i64 - 1) * f0;
                            prod = prod * fk;
                        }
                        p[k] = p[k] + prod;
                    }
                }
                p
            })
            .reduce(
                || vec![Goldilocks::ZERO; max_deg + 1],
                |mut a, b| {
                    for i in 0..=max_deg {
                        a[i] = a[i] + b[i];
                    }
                    a
                },
            );
        rounds.push(pvals);
        bufs.par_iter_mut().for_each(|buf| {
            for s in 0..half {
                let f0 = buf[2 * s];
                let f1 = buf[2 * s + 1];
                buf[s] = (Goldilocks::ONE - r) * f0 + r * f1;
            }
            buf.truncate(half);
        });
    }
    let final_evals = bufs.iter().map(|b| b[0]).collect();
    VirtualProof { rounds, final_evals }
}
fn interpolate(vals: &[Goldilocks], x: Goldilocks, d: usize) -> Goldilocks {
    let mut out = Goldilocks::ZERO;
    for k in 0..=d {
        let mut term = vals[k];
        for j in 0..=d {
            if j != k {
                let num = x - from_i64(j as i64);
                let den = from_i64(k as i64) - from_i64(j as i64);
                term = term * num * den.inverse();
            }
        }
        out = out + term;
    }
    out
}
pub fn verify_virtual(
    proof: &VirtualProof,
    terms: &[(Goldilocks, Vec<usize>)],
    claimed: Goldilocks,
    challenges: &[Goldilocks],
    final_evals: &[Goldilocks],
) -> bool {
    let max_deg = terms.iter().map(|(_, idxs)| idxs.len()).max().unwrap_or(0);
    if proof.rounds.len() != challenges.len() {
        return false;
    }
    let mut current = claimed;
    for (pvals, &r) in proof.rounds.iter().zip(challenges) {
        if pvals.len() != max_deg + 1 {
            return false;
        }
        if pvals[0] + pvals[1] != current {
            return false;
        }
        current = interpolate(pvals, r, max_deg);
    }
    let mut final_val = Goldilocks::ZERO;
    for (coeff, idxs) in terms {
        let mut prod = *coeff;
        for &j in idxs {
            prod = prod * final_evals[j];
        }
        final_val = final_val + prod;
    }
    current == final_val
}

// ==== generic-field virtual sumcheck (sum of products of MLEs) ====
//
// The same virtual-polynomial circuit as the base `prove_virtual` /
// `verify_virtual` above, instantiated over ANY field `F` (the EF LogUp
// reducer will run it over `F = EF` with transcript-supplied challenges).
// The base entry points are untouched: no behavior change, no conversion
// allocations on the base path.

/// Generic virtual-sumcheck proof: per-round values of the round polynomial
/// at `0..=max_deg` (stored as evaluation vectors, like the base
/// `VirtualProof`), plus the final evaluations of every MLE.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VirtualProofF<F> {
    pub rounds: Vec<Vec<F>>,
    pub final_evals: Vec<F>,
}

/// Maximum supported round-polynomial degree for the generic virtual
/// sumcheck. The Lagrange interpolation denominators are `k - j` for
/// `0 <= k, j <= max_deg`; their inverses exist iff `max_deg` is below the
/// field characteristic. This bound is far below char(Goldilocks) (~2^64)
/// and char(EF) (~2^128), and comfortably covers the degree-3 LogUp circuit.
pub const MAX_VIRTUAL_DEGREE: usize = 16;

/// Small integer as a field element (0..=3 and -1 suffice for the LogUp
/// degree-3 circuit): built by repeated addition, no `from_u64` dependence.
fn small_int_f<F: PrimeCharacteristicRing + Copy>(v: i64) -> F {
    let (unit, n) = if v >= 0 { (F::ONE, v as u64) } else { (F::NEG_ONE, (-v) as u64) };
    let mut acc = F::ZERO;
    for _ in 0..n {
        acc = acc + unit;
    }
    acc
}

/// Lagrange-interpolate the degree-`d` polynomial through `vals[0..=d]` at
/// `x`. The inverse denominators depend only on `d` and are precomputed once
/// per call (fewer inversions than the base per-evaluation Lagrange).
pub fn interpolate_f<F: Field + PrimeCharacteristicRing + Copy>(vals: &[F], x: F, d: usize) -> F {
    let mut inv_dens = Vec::with_capacity(d + 1);
    for k in 0..=d {
        let mut den = F::ONE;
        for j in 0..=d {
            if j != k {
                den = den * (small_int_f::<F>(k as i64) - small_int_f::<F>(j as i64));
            }
        }
        inv_dens.push(den.inverse());
    }
    let mut out = F::ZERO;
    for k in 0..=d {
        let mut term = vals[k];
        for j in 0..=d {
            if j != k {
                term = term * (x - small_int_f::<F>(j as i64));
            }
        }
        out = out + term * inv_dens[k];
    }
    out
}

/// Compute the round-polynomial values at `0..=max_deg` for the current
/// (partially folded) buffers of a virtual sumcheck. This is the per-round
/// core shared by [`prove_virtual_f`] and interactive Fiat–Shamir loops that
/// absorb each round message before sampling the next challenge.
pub fn virtual_round_pvals_f<F: Field + PrimeCharacteristicRing + Copy>(
    bufs: &[Vec<F>],
    terms: &[(F, Vec<usize>)],
    max_deg: usize,
) -> Vec<F> {
    let half = bufs[0].len() / 2;
    let mut pvals = vec![F::ZERO; max_deg + 1];
    for s in 0..half {
        for (coeff, idxs) in terms {
            for k in 0..=max_deg {
                let mut prod = *coeff;
                for &j in idxs {
                    let f0 = bufs[j][2 * s];
                    let f1 = bufs[j][2 * s + 1];
                    let fk = small_int_f::<F>(k as i64) * f1 - small_int_f::<F>(k as i64 - 1) * f0;
                    prod = prod * fk;
                }
                pvals[k] = pvals[k] + prod;
            }
        }
    }
    pvals
}

/// Prove `sum_x sum_i coeff_i * prod_{j in term_i} f_j(x) == claimed` over any
/// field `F`. Sequential (no rayon); the base path keeps its parallel prover.
pub fn prove_virtual_f<F: Field + PrimeCharacteristicRing + Copy>(
    mles: &[&[F]],
    terms: &[(F, Vec<usize>)],
    claimed: F,
    challenges: &[F],
) -> VirtualProofF<F> {
    // API contract: at least one term is required, every term must reference
    // at least one MLE (degree >= 1) — the degree-0 case is rejected because
    // the current proof encoding stores per-round values q(0)..q(deg) and the
    // q(0)+q(1) round check has no pvals[1] slot for deg 0, not because the
    // algebra is impossible — and the degree must stay within
    // MAX_VIRTUAL_DEGREE (the interpolation denominators k-j are invertible
    // only below the field characteristic).
    assert!(!terms.is_empty(), "at least one term required");
    assert!(
        terms.iter().all(|(_, idxs)| !idxs.is_empty()),
        "every term must reference at least one MLE (degree >= 1)"
    );
    let max_deg = terms.iter().map(|(_, idxs)| idxs.len()).max().unwrap_or(0);
    assert!(
        max_deg <= MAX_VIRTUAL_DEGREE,
        "virtual sumcheck degree above MAX_VIRTUAL_DEGREE"
    );
    let n = mles[0].len();
    let t = n.trailing_zeros() as usize;
    assert!(n.is_power_of_two());
    for m in mles {
        assert_eq!(m.len(), n);
    }
    for (_, idxs) in terms {
        for &j in idxs {
            assert!(j < mles.len(), "term index out of bounds");
        }
    }
    assert_eq!(challenges.len(), t);
    let mut bufs: Vec<Vec<F>> = mles.iter().map(|m| m.to_vec()).collect();
    let mut rounds = Vec::with_capacity(t);
    for &r in challenges {
        let half = bufs[0].len() / 2;
        rounds.push(virtual_round_pvals_f(&bufs, terms, max_deg));
        for buf in bufs.iter_mut() {
            for s in 0..half {
                let f0 = buf[2 * s];
                let f1 = buf[2 * s + 1];
                buf[s] = (F::ONE - r) * f0 + r * f1;
            }
            buf.truncate(half);
        }
    }
    let final_evals = bufs.iter().map(|b| b[0]).collect();
    VirtualProofF { rounds, final_evals }
}

/// Verify the generic virtual sumcheck; `final_evals` are the terminal
/// evaluations of every MLE (authenticated out-of-band, e.g. by PCS
/// openings).
///
/// Never panics on untrusted proof/terms: malformed shapes (empty term list,
/// constant-only degree-0 terms, excessive degree, term indices beyond
/// `final_evals`, short round polynomials, wrong round/challenge counts)
/// return `false`.
///
/// API contract: every term must reference at least one MLE (degree >= 1) —
/// the degree-0 case is rejected because the current proof encoding stores
/// per-round values `q(0)..q(deg)` and the `q(0) + q(1) == current` round
/// check has no `pvals[1]` slot at degree 0, not because the algebra is
/// impossible. The degree must also stay within
/// [`MAX_VIRTUAL_DEGREE`], keeping the interpolation denominators `k - j`
/// invertible (nonzero below the field characteristic).
pub fn verify_virtual_f<F: Field + PrimeCharacteristicRing + Copy>(
    proof: &VirtualProofF<F>,
    terms: &[(F, Vec<usize>)],
    claimed: F,
    challenges: &[F],
    final_evals: &[F],
) -> bool {
    // API contract: non-empty, degree >= 1 terms only, degree bounded below
    // the field characteristic (so no interpolation denominator is zero).
    if terms.is_empty() || terms.iter().any(|(_, idxs)| idxs.is_empty()) {
        return false;
    }
    let max_deg = terms.iter().map(|(_, idxs)| idxs.len()).max().unwrap_or(0);
    if max_deg > MAX_VIRTUAL_DEGREE {
        return false;
    }
    let max_idx = terms
        .iter()
        .flat_map(|(_, idxs)| idxs.iter())
        .copied()
        .max()
        .unwrap_or(0);
    // Every term index must be indexable in `final_evals`.
    if final_evals.len() <= max_idx {
        return false;
    }
    if proof.rounds.len() != challenges.len() {
        return false;
    }
    let mut current = claimed;
    for (pvals, &r) in proof.rounds.iter().zip(challenges) {
        if pvals.len() != max_deg + 1 {
            return false;
        }
        if pvals[0] + pvals[1] != current {
            return false;
        }
        current = interpolate_f(pvals, r, max_deg);
    }
    let mut final_val = F::ZERO;
    for (coeff, idxs) in terms {
        let mut prod = *coeff;
        for &j in idxs {
            prod = prod * final_evals[j];
        }
        final_val = final_val + prod;
    }
    current == final_val
}

#[cfg(test)]
mod virtual_tests {
    use super::*;
    use crate::common::field::{EF, XorShift64};
    #[test]
    fn virtual_sumcheck_roundtrip() {
        let mut rng = XorShift64::new(0xBEEF);
        let n = 1usize << 6;
        let f: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let g: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let h: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let t = n.trailing_zeros() as usize;
        let challenges: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
        let terms: Vec<(Goldilocks, Vec<usize>)> = vec![
            (Goldilocks::from_u64(3), vec![0, 1]),
            (from_i64(-2), vec![1, 2]),
            (Goldilocks::ONE, vec![0, 1, 2]),
        ];
        let claimed: Goldilocks = (0..n).map(|i| {
            Goldilocks::from_u64(3) * f[i] * g[i] + from_i64(-2) * g[i] * h[i] + f[i] * g[i] * h[i]
        }).sum();
        let mles: Vec<&[Goldilocks]> = vec![&f, &g, &h];
        let proof = prove_virtual(&mles, &terms, claimed, &challenges);
        for (m, &ev) in mles.iter().zip(&proof.final_evals) {
            let ev2 = crate::common::mle::eval(m, &challenges);
            assert_eq!(ev, ev2);
        }
        assert!(verify_virtual(&proof, &terms, claimed, &challenges, &proof.final_evals));
        let mut bad = proof.final_evals.clone();
        bad[0] = bad[0] + Goldilocks::ONE;
        assert!(!verify_virtual(&proof, &terms, claimed, &challenges, &bad));
    }
    #[test]
    fn eq_weighted_sumcheck() {
        let mut rng = XorShift64::new(0xC0DE);
        let n = 1usize << 5;
        let f: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let t = n.trailing_zeros() as usize;
        let r: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
        let eq: Vec<Goldilocks> = crate::common::mle::eq_evals(&r);
        let claimed = crate::common::mle::eval(&f, &r);
        let mles: Vec<&[Goldilocks]> = vec![&eq, &f];
        let terms = vec![(Goldilocks::ONE, vec![0usize, 1])];
        let challenges: Vec<Goldilocks> = r.clone();
        let proof = prove_virtual(&mles, &terms, claimed, &challenges);
        assert!(verify_virtual(&proof, &terms, claimed, &challenges, &proof.final_evals));
    }

    // ==== generic-field virtual sumcheck (EF instantiation) ====

    /// Independent EF fold (direct multilinear evaluation over EF values).
    fn fold_ef(values: &[EF], point: &[EF]) -> EF {
        let mut buf = values.to_vec();
        let mut size = buf.len();
        for &p in point {
            let half = size / 2;
            for i in 0..half {
                let a = buf[2 * i];
                let b = buf[2 * i + 1];
                buf[i] = a + p * (b - a);
            }
            size = half;
        }
        buf[0]
    }

    /// eq evaluations over EF.
    fn eq_ef_evals(r: &[EF]) -> Vec<EF> {
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

    /// The 7-MLE LogUp fold-identity circuit over EF: the generic virtual
    /// sumcheck must match an independent direct evaluation of the identity,
    /// with genuinely non-base challenges, and reject tampered coefficients.
    #[test]
    fn virtual_f_ef_logup_layer_identity() {
        use crate::common::field::{BasedVectorSpace, EF};
        let mut rng = XorShift64::new(0xEF71);
        let t = 2usize; // 4 hypercube points, 7 MLEs
        let n = 1 << t;
        let mut ef = || {
            EF::from_basis_coefficients_slice(&[rng.field(), rng.field()]).unwrap()
        };        let n_low: Vec<EF> = (0..n).map(|_| ef()).collect();
        let n_high: Vec<EF> = (0..n).map(|_| ef()).collect();
        let d_low: Vec<EF> = (0..n).map(|_| ef()).collect();
        let d_high: Vec<EF> = (0..n).map(|_| ef()).collect();
        let next_num: Vec<EF> = (0..n)
            .map(|s| n_low[s] * d_high[s] + n_high[s] * d_low[s])
            .collect();
        let next_den: Vec<EF> = (0..n).map(|s| d_low[s] * d_high[s]).collect();
        let r_pt: Vec<EF> = (0..t).map(|_| ef()).collect();
        let eq: Vec<EF> = eq_ef_evals(&r_pt);
        let c = ef();
        let neg = EF::ZERO - EF::ONE;

        // Mirror of `logup_gkr::build_terms`: the pointwise identity
        //   eq*(next_num - nL*dH - nH*dL + c*next_den - c*dL*dH) == 0.
        let terms: Vec<(EF, Vec<usize>)> = vec![
            (EF::ONE, vec![0usize, 1]),
            (neg, vec![0, 3, 6]),
            (neg, vec![0, 4, 5]),
            (c, vec![0, 2]),
            (neg * c, vec![0, 5, 6]),
        ];
        let mles: Vec<&[EF]> = vec![&eq, &next_num, &next_den, &n_low, &n_high, &d_low, &d_high];
        let challenges: Vec<EF> = (0..t).map(|_| ef()).collect();
        // Genuinely non-base challenges.
        for ch in challenges.iter().chain(r_pt.iter()) {
            let coeffs: &[Goldilocks] = ch.as_basis_coefficients_slice();
            assert_ne!(coeffs[1], Goldilocks::ZERO);
        }

        // The direct evaluation of the identity sum is exactly zero.
        let claimed = EF::ZERO;

        let proof = prove_virtual_f(&mles, &terms, claimed, &challenges);
        // Independent direct reference: terminal evals = direct MLE evals.
        let ref_evals: Vec<EF> = mles.iter().map(|m| fold_ef(m, &challenges)).collect();
        assert_eq!(proof.final_evals, ref_evals, "terminal evals must match direct EF folds");
        assert!(verify_virtual_f(&proof, &terms, claimed, &challenges, &ref_evals));

        // Tampered round coefficient must be rejected.
        let mut bad = proof.clone();
        bad.rounds[0][0] = bad.rounds[0][0] + EF::ONE;
        assert!(!verify_virtual_f(&bad, &terms, claimed, &challenges, &ref_evals));
    }

    /// The generic circuit instantiated at F = Goldilocks must reproduce the
    /// base `prove_virtual` output exactly (same rounds, same final evals).
    #[test]
    fn virtual_f_base_matches_base_virtual() {
        let mut rng = XorShift64::new(0xEF72);
        let n = 1usize << 5;
        let f: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let g: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let h: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let t = n.trailing_zeros() as usize;
        let challenges: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
        let terms: Vec<(Goldilocks, Vec<usize>)> = vec![
            (Goldilocks::from_u64(3), vec![0, 1]),
            (from_i64(-2), vec![1, 2]),
            (Goldilocks::ONE, vec![0, 1, 2]),
        ];
        let claimed: Goldilocks = (0..n)
            .map(|i| Goldilocks::from_u64(3) * f[i] * g[i] + from_i64(-2) * g[i] * h[i] + f[i] * g[i] * h[i])
            .sum();
        let mles: Vec<&[Goldilocks]> = vec![&f, &g, &h];

        let base = prove_virtual(&mles, &terms, claimed, &challenges);
        let generic = prove_virtual_f(&mles, &terms, claimed, &challenges);
        assert_eq!(base.rounds, generic.rounds);
        assert_eq!(base.final_evals, generic.final_evals);
        assert!(verify_virtual_f(&generic, &terms, claimed, &challenges, &generic.final_evals));
    }

    /// `verify_virtual_f` must return `false` — never panic — for untrusted
    /// malformed inputs: empty term list, short `final_evals`, out-of-bounds
    /// term index, malformed short round, wrong challenge count.
    #[test]
    fn verify_virtual_f_rejects_malformed_inputs_without_panicking() {
        use crate::common::field::{BasedVectorSpace, EF};
        let mut rng = XorShift64::new(0xEF73);
        let t = 2usize;
        let n = 1 << t;
        let mut ef = || EF::from_basis_coefficients_slice(&[rng.field(), rng.field()]).unwrap();
        let n_low: Vec<EF> = (0..n).map(|_| ef()).collect();
        let n_high: Vec<EF> = (0..n).map(|_| ef()).collect();
        let d_low: Vec<EF> = (0..n).map(|_| ef()).collect();
        let d_high: Vec<EF> = (0..n).map(|_| ef()).collect();
        let next_num: Vec<EF> = (0..n).map(|s| n_low[s] * d_high[s] + n_high[s] * d_low[s]).collect();
        let next_den: Vec<EF> = (0..n).map(|s| d_low[s] * d_high[s]).collect();
        let r_pt: Vec<EF> = (0..t).map(|_| ef()).collect();
        let eq: Vec<EF> = eq_ef_evals(&r_pt);
        let c = ef();
        let neg = EF::ZERO - EF::ONE;
        let terms: Vec<(EF, Vec<usize>)> = vec![
            (EF::ONE, vec![0usize, 1]),
            (neg, vec![0, 3, 6]),
            (neg, vec![0, 4, 5]),
            (c, vec![0, 2]),
            (neg * c, vec![0, 5, 6]),
        ];
        let mles: Vec<&[EF]> = vec![&eq, &next_num, &next_den, &n_low, &n_high, &d_low, &d_high];
        let challenges: Vec<EF> = (0..t).map(|_| ef()).collect();
        let claimed = EF::ZERO;
        let proof = prove_virtual_f(&mles, &terms, claimed, &challenges);
        let ref_evals: Vec<EF> = mles.iter().map(|m| fold_ef(m, &challenges)).collect();
        assert!(verify_virtual_f(&proof, &terms, claimed, &challenges, &ref_evals));

        let result = std::panic::catch_unwind(|| {
            // Empty term list.
            assert!(!verify_virtual_f(&proof, &[], claimed, &challenges, &ref_evals));
            // Constant-only (degree-0) term: rejected by the API contract.
            let const_terms: Vec<(EF, Vec<usize>)> = vec![(EF::ONE, vec![])];
            assert!(!verify_virtual_f(&proof, &const_terms, claimed, &challenges, &ref_evals));
            // Excessive degree (above MAX_VIRTUAL_DEGREE): rejected before
            // any interpolation (the denominators k-j would risk zero
            // inversion at or above the field characteristic).
            let deg_terms: Vec<(EF, Vec<usize>)> =
                vec![(EF::ONE, vec![0usize; MAX_VIRTUAL_DEGREE + 1])];
            assert!(!verify_virtual_f(&proof, &deg_terms, claimed, &challenges, &ref_evals));
            // Short final_evals.
            assert!(!verify_virtual_f(&proof, &terms, claimed, &challenges, &ref_evals[..1]));
            // Out-of-bounds term index.
            let mut bad_terms = terms.clone();
            bad_terms[0].1[0] = 42;
            assert!(!verify_virtual_f(&proof, &bad_terms, claimed, &challenges, &ref_evals));
            // Malformed short round polynomial.
            let mut bad_proof = proof.clone();
            bad_proof.rounds[0] = vec![EF::ZERO]; // len 1 != max_deg + 1
            assert!(!verify_virtual_f(&bad_proof, &terms, claimed, &challenges, &ref_evals));
            // Wrong challenge count.
            assert!(!verify_virtual_f(&proof, &terms, claimed, &challenges[..1], &ref_evals));
        });
        assert!(result.is_ok(), "verify_virtual_f must never panic on malformed inputs");
    }
}
