//! Multilinear-extension evaluation over the Goldilocks field.
//!
//! Conventions: variable `0` is the least-significant bit of the flattened
//! index; `partial_eval` fixes the *first* `fix.len()` variables.

use crate::field::F64;

pub fn eval(values: &[F64], point: &[F64]) -> F64 {
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

pub fn partial_eval(values: &[F64], fix: &[F64]) -> Vec<F64> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::XorShift64;

    #[test]
    fn partial_then_full_matches_full() {
        let mut rng = XorShift64::new(2);
        let values: Vec<F64> = (0..16).map(|_| rng.field()).collect();
        let point: Vec<F64> = (0..4).map(|_| rng.field()).collect();

        let rest = partial_eval(&values, &point[..2]);
        assert_eq!(rest.len(), 4);
        let via_partial = eval(&rest, &point[2..]);
        let full = eval(&values, &point);
        assert_eq!(via_partial, full);
    }
}
