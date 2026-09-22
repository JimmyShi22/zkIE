//! Sum-check protocol for a product of two multilinear polynomials.
//!
//! Proves `H = sum_{x in {0,1}^t} f(x) * h(x)`. Each round polynomial has
//! degree at most two, so the prover sends three coefficients per round.

use crate::field::{Field, Goldilocks, PrimeCharacteristicRing};

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

const INV2: Goldilocks = Goldilocks::new(crate::field::P / 2 + 1); // (p + 1) / 2

pub fn prove(f: &[Goldilocks], h: &[Goldilocks], _claimed_sum: Goldilocks, challenges: &[Goldilocks]) -> SumcheckProof {
    let t = f.len().trailing_zeros() as usize;
    assert_eq!(f.len(), 1 << t, "f length must be a power of two");
    assert_eq!(h.len(), f.len());
    assert_eq!(challenges.len(), t);

    let mut f_buf = f.to_vec();
    let mut h_buf = h.to_vec();
    let mut rounds = Vec::with_capacity(t);

    for r in challenges.iter().copied() {
        let half = f_buf.len() / 2;
        let mut y0 = Goldilocks::ZERO;
        let mut y1 = Goldilocks::ZERO;
        let mut y2 = Goldilocks::ZERO;
        for s in 0..half {
            let f0 = f_buf[2 * s];
            let f1 = f_buf[2 * s + 1];
            let h0 = h_buf[2 * s];
            let h1 = h_buf[2 * s + 1];
            y0 = y0 + f0 * h0;
            y1 = y1 + f1 * h1;
            let f2 = Goldilocks::TWO * f1 - f0;
            let h2 = Goldilocks::TWO * h1 - h0;
            y2 = y2 + f2 * h2;
        }

        let c0 = y0;
        let c1 = (Goldilocks::from_u64(4) * y1 - y2 - Goldilocks::from_u64(3) * y0) * INV2;
        let c2 = (y2 - Goldilocks::TWO * y1 + y0) * INV2;
        rounds.push(RoundPoly { c0, c1, c2 });

        fold(&mut f_buf, r);
        fold(&mut h_buf, r);
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
    claimed_sum: Goldilocks,
    challenges: &[Goldilocks],
    f_eval: Goldilocks,
    h_eval: Goldilocks,
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
    _claimed_sum: Goldilocks,
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
    claimed_sum: Goldilocks,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::XorShift64;

    #[test]
    fn sumcheck_completeness_and_soundness() {
        let mut rng = XorShift64::new(3);
        let t = 8;
        let f: Vec<Goldilocks> = (0..(1 << t)).map(|_| rng.field()).collect();
        let h: Vec<Goldilocks> = (0..(1 << t)).map(|_| rng.field()).collect();
        let true_sum: Goldilocks = f.iter().zip(&h).fold(Goldilocks::ZERO, |acc, (&a, &b)| acc + a * b);
        let challenges: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();

        let proof = prove(&f, &h, true_sum, &challenges);
        let f_eval = crate::mle::eval(&f, &challenges);
        let h_eval = crate::mle::eval(&h, &challenges);
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
        let f_eval = crate::mle::eval(&f, &challenges);
        let g_eval = crate::mle::eval(&g, &challenges);
        let h_eval = crate::mle::eval(&h, &challenges);
        assert!(verify3(&proof, true_sum, &challenges, f_eval, g_eval, h_eval));

        let wrong = true_sum + Goldilocks::ONE;
        assert!(!verify3(&proof, wrong, &challenges, f_eval, g_eval, h_eval));
    }
}
