//! FRI building blocks: multilinear-to-univariate coefficient lift, Horner
//! evaluation, low-degree-extension commitment, and the degree-halving fold.
//!
//! These are the primitives a full FRI opening (evaluation proof) is built
//! from. The opening itself is deliberately left as the next sub-step; see the
//! crate README for what remains.

use crate::field::{two_adic_root, F31};
use crate::merkle::{self, Digest};

/// Map an MLE `f` (values over `{0,1}^t`) to the univariate polynomial
/// `g(X) = f(X, X^2, X^4, ..., X^{2^{t-1}})` represented by coefficients.
///
/// This is the tensor transform `(M^T)^(⊗ t)` with `M = [[1,-1],[0,1]]`.
pub fn mle_to_coeff(f: &[F31]) -> Vec<F31> {
    let t = f.len().trailing_zeros() as usize;
    assert_eq!(f.len(), 1 << t);
    let mut coeff = f.to_vec();
    for j in 0..t {
        let step = 1 << j;
        for i in (0..f.len()).step_by(2 * step) {
            for k in 0..step {
                let a = coeff[i + k];
                coeff[i + step + k] = coeff[i + step + k] - a;
            }
        }
    }
    coeff
}

/// Evaluate a coefficient-represented polynomial at `x` via Horner.
pub fn coeff_eval(coeff: &[F31], x: F31) -> F31 {
    let mut acc = F31::ZERO;
    for &c in coeff.iter().rev() {
        acc = acc * x + c;
    }
    acc
}

/// Evaluate the LDE of `coeff` (degree `< coeff.len()`) on the full domain of
/// size `coeff.len() * 2^rate_bits`.
pub fn lde_evaluations(coeff: &[F31], rate_bits: usize) -> Vec<F31> {
    let t = coeff.len().trailing_zeros() as usize;
    let n = 1 << (t + rate_bits);
    let omega = two_adic_root(t + rate_bits);
    let mut vals = Vec::with_capacity(n);
    let mut x = F31::ONE;
    for _ in 0..n {
        vals.push(coeff_eval(coeff, x));
        x = x * omega;
    }
    vals
}

/// Commit to the MLE `f` by committing its LDE via a Merkle tree.
pub fn commit(f: &[F31], rate_bits: usize) -> Digest {
    let coeff = mle_to_coeff(f);
    let evals = lde_evaluations(&coeff, rate_bits);
    merkle::commit(&evals)
}

/// One FRI fold: `g'(X) = g_even(X) + alpha * g_odd(X)`, halving the degree.
pub fn fold(coeff: &[F31], alpha: F31) -> Vec<F31> {
    let half = coeff.len() / 2;
    let mut out = Vec::with_capacity(half);
    for i in 0..half {
        out.push(coeff[2 * i] + alpha * coeff[2 * i + 1]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::XorShift64;
    use crate::mle;

    #[test]
    fn coefficient_lift_matches_mle_eval() {
        let mut rng = XorShift64::new(10);
        let t = 6;
        let f: Vec<F31> = (0..(1 << t)).map(|_| rng.field()).collect();
        let coeff = mle_to_coeff(&f);

        let mut x = rng.field();
        let point: Vec<F31> = (0..t).map(|_| {
            let v = x;
            x = x * x;
            v
        }).collect();
        assert_eq!(coeff_eval(&coeff, point[0]), mle::eval(&f, &point));
    }

    #[test]
    fn fold_halves_degree_and_matches_eval() {
        let mut rng = XorShift64::new(11);
        let t = 6;
        let f: Vec<F31> = (0..(1 << t)).map(|_| rng.field()).collect();
        let coeff = mle_to_coeff(&f);
        let alpha = rng.field();
        let folded = fold(&coeff, alpha);
        assert_eq!(folded.len(), coeff.len() / 2);

        // g'(X^2) = g_even(X^2) + alpha * g_odd(X^2), where g(X)=g_even(X^2)+X*g_odd(X^2).
        let x = rng.field();
        let g_at_x = coeff_eval(&coeff, x);
        let g_at_negx = coeff_eval(&coeff, -x);
        let expected = (g_at_x + g_at_negx) * F31::new((crate::field::P + 1) / 2)
            + alpha * (g_at_x - g_at_negx) * F31::new((crate::field::P + 1) / 2) * x.inv();
        assert_eq!(coeff_eval(&folded, x * x), expected);
    }
}
