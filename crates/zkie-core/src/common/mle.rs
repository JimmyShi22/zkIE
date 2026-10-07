//! Multilinear-extension evaluation over the Goldilocks field.
//!
//! Conventions: variable `0` is the least-significant bit of the flattened
//! index; `partial_eval` fixes the *first* `fix.len()` variables.

use crate::common::field::{Goldilocks, PrimeCharacteristicRing};

pub fn eval(values: &[Goldilocks], point: &[Goldilocks]) -> Goldilocks {
    let t = point.len();
    assert_eq!(values.len(), 1 << t, "mle eval: length mismatch");
    let mut buf = values.to_vec();
    let mut size = values.len();
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

pub fn partial_eval(values: &[Goldilocks], fix: &[Goldilocks]) -> Vec<Goldilocks> {
    assert!(fix.len() <= values.len().trailing_zeros() as usize);
    let mut buf = values.to_vec();
    for &p in fix {
        let half = buf.len() / 2;
        for i in 0..half {
            let a = buf[2 * i];
            let b = buf[2 * i + 1];
            buf[i] = a + p * (b - a);
        }
        buf.truncate(half);
    }
    buf
}

/// Extension-field lift: evaluates a base-field-stored table as an MLE over
/// `EF`. Tensor storage stays base Goldilocks; only the evaluation arithmetic
/// is lifted. Same LSB-first coordinate convention as [`eval`].
pub fn eval_ef(values: &[Goldilocks], point: &[crate::common::field::EF]) -> crate::common::field::EF {
    let t = point.len();
    assert_eq!(values.len(), 1 << t, "mle eval_ef: length mismatch");
    let mut buf: Vec<crate::common::field::EF> = values.iter().map(|&v| v.into()).collect();
    let mut size = values.len();
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

/// Extension-field lift of [`partial_eval`]: fixes the first `fix.len()`
/// variables of a base-field-stored table using EF points, returning the
/// remaining evaluations as EF elements.
pub fn partial_eval_ef(values: &[Goldilocks], fix: &[crate::common::field::EF]) -> Vec<crate::common::field::EF> {
    assert!(fix.len() <= values.len().trailing_zeros() as usize);
    let mut buf: Vec<crate::common::field::EF> = values.iter().map(|&v| v.into()).collect();
    for &p in fix {
        let half = buf.len() / 2;
        for i in 0..half {
            let a = buf[2 * i];
            let b = buf[2 * i + 1];
            buf[i] = a + p * (b - a);
        }
        buf.truncate(half);
    }
    buf
}

/// Evaluate the equality (Lagrange-basis) polynomial `eq(x, r)` at every
/// hypercube point `x = i` for a fixed `r`: `eq_i(r) = prod_j (bit_j(i) * r_j +
/// (1 - bit_j(i)) * (1 - r_j))`. Used as the random selector in the
/// zero-check sumchecks.
pub fn eq_evals(r: &[Goldilocks]) -> Vec<Goldilocks> {
    let d = r.len();
    let n = 1 << d;
    let mut eq = vec![Goldilocks::ONE; n];
    for j in 0..d {
        let one_minus = Goldilocks::ONE - r[j];
        for i in 0..n {
            eq[i] = eq[i] * if (i >> j) & 1 == 1 { r[j] } else { one_minus };
        }
    }
    eq
}


/// Evaluate the equality polynomial `eq(r, x) = prod_j (x_j r_j + (1-x_j)(1-r_j))`
/// at a single point `x`. Used by the multi-point batch opening.
pub fn eq_poly(r: &[Goldilocks], x: &[Goldilocks]) -> Goldilocks {
    assert_eq!(r.len(), x.len());
    r.iter()
        .zip(x)
        .fold(Goldilocks::ONE, |acc, (&ri, &xi)| {
            acc * (xi * ri + (Goldilocks::ONE - xi) * (Goldilocks::ONE - ri))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::field::XorShift64;

    #[test]
    fn partial_then_full_matches_full() {
        let mut rng = XorShift64::new(2);
        let values: Vec<Goldilocks> = (0..16).map(|_| rng.field()).collect();
        let point: Vec<Goldilocks> = (0..4).map(|_| rng.field()).collect();

        let rest = partial_eval(&values, &point[..2]);
        assert_eq!(rest.len(), 4);
        let via_partial = eval(&rest, &point[2..]);
        let full = eval(&values, &point);
        assert_eq!(via_partial, full);
    }
}
