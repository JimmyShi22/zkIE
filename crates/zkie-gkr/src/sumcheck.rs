//! Sum-check protocol for a product of two multilinear polynomials.
//!
//! Proves `H = sum_{x in {0,1}^t} f(x) * h(x)`. Each round polynomial has
//! degree at most two, so the prover sends three coefficients per round.

use crate::field::F64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoundPoly {
    pub c0: F64,
    pub c1: F64,
    pub c2: F64,
}

impl RoundPoly {
    #[inline]
    pub fn eval(&self, x: F64) -> F64 {
        self.c0 + self.c1 * x + self.c2 * x * x
    }
}

#[derive(Clone, Debug)]
pub struct SumcheckProof {
    pub rounds: Vec<RoundPoly>,
    pub f_eval: F64,
    pub h_eval: F64,
}

const INV2: F64 = F64(crate::field::P / 2 + 1); // (p + 1) / 2

pub fn prove(f: &[F64], h: &[F64], _claimed_sum: F64, challenges: &[F64]) -> SumcheckProof {
    let t = f.len().trailing_zeros() as usize;
    assert_eq!(f.len(), 1 << t, "f length must be a power of two");
    assert_eq!(h.len(), f.len());
    assert_eq!(challenges.len(), t);

    let mut f_buf = f.to_vec();
    let mut h_buf = h.to_vec();
    let mut rounds = Vec::with_capacity(t);

    for r in challenges.iter().copied() {
        let half = f_buf.len() / 2;
        let mut y0 = F64::ZERO;
        let mut y1 = F64::ZERO;
        let mut y2 = F64::ZERO;
        for s in 0..half {
            let f0 = f_buf[2 * s];
            let f1 = f_buf[2 * s + 1];
            let h0 = h_buf[2 * s];
            let h1 = h_buf[2 * s + 1];
            y0 = y0 + f0 * h0;
            y1 = y1 + f1 * h1;
            let f2 = F64::TWO * f1 - f0;
            let h2 = F64::TWO * h1 - h0;
            y2 = y2 + f2 * h2;
        }

        let c0 = y0;
        let c1 = (F64::new(4) * y1 - y2 - F64::new(3) * y0) * INV2;
        let c2 = (y2 - F64::TWO * y1 + y0) * INV2;
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

fn fold(buf: &mut Vec<F64>, p: F64) {
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
    claimed_sum: F64,
    challenges: &[F64],
    f_eval: F64,
    h_eval: F64,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::XorShift64;

    #[test]
    fn sumcheck_completeness_and_soundness() {
        let mut rng = XorShift64::new(3);
        let t = 8;
        let f: Vec<F64> = (0..(1 << t)).map(|_| rng.field()).collect();
        let h: Vec<F64> = (0..(1 << t)).map(|_| rng.field()).collect();
        let true_sum: F64 = f.iter().zip(&h).fold(F64::ZERO, |acc, (&a, &b)| acc + a * b);
        let challenges: Vec<F64> = (0..t).map(|_| rng.field()).collect();

        let proof = prove(&f, &h, true_sum, &challenges);
        let f_eval = crate::mle::eval(&f, &challenges);
        let h_eval = crate::mle::eval(&h, &challenges);
        assert!(verify(&proof, true_sum, &challenges, f_eval, h_eval));

        let wrong = true_sum + F64::ONE;
        assert!(!verify(&proof, wrong, &challenges, f_eval, h_eval));
    }
}
