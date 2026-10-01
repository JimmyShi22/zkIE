//! Uniform claim type: the single currency in which op primitives,
//! `same_poly` binding, and shard boundaries all communicate.
//!
//! A `Claim` says "tensor evaluates to `eval` at MLE point `point`". Every op
//! primitive emits claims on its input/output tensors; the shard composer (and
//! cross-shard binding) collects the claims on each multiply-consumed tensor and
//! merges them via `same_poly` into one. The number of ops folded into a shard is
//! a public parameter; the claim type is what lets that folding be data-driven
//! rather than hand-wired per model structure.

use crate::field::Goldilocks;
use crate::mle;

/// A tensor evaluated at a point: `eval == MLE(tensor)(point)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claim {
    pub point: Vec<Goldilocks>,
    pub eval: Goldilocks,
}

impl Claim {
    /// Build a claim by actually evaluating `tensor` at `point`.
    pub fn new(tensor: &[Goldilocks], point: Vec<Goldilocks>) -> Self {
        let eval = mle::eval(tensor, &point);
        Claim { point, eval }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::PrimeCharacteristicRing;

    #[test]
    fn claim_evaluates_tensor_at_point() {
        let mut rng = crate::field::XorShift64::new(0x0C0C);
        let n = 1usize << 5;
        let f: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let point: Vec<Goldilocks> = (0..5).map(|_| rng.field()).collect();
        let c = Claim::new(&f, point.clone());
        assert_eq!(c.point, point);
        assert_eq!(c.eval, crate::mle::eval(&f, &point));
    }
}
