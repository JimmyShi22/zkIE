//! int16 fixed-point layer over Goldilocks.
//!
//! This is the "define the semantics first" layer: signed int16 values are
//! embedded in the field (`x < 0` maps to `P - |x|`), and a dot product is
//! computed directly in the field. As long as the true integer result stays in
//! `(-P/2, P/2)` the field result is the integer result with no wrap, which is
//! exactly the condition Goldilocks satisfies for int16 accumulation.

use crate::field::{Goldilocks, P, PrimeCharacteristicRing, PrimeField64};

pub fn from_i16(x: i16) -> Goldilocks {
    if x >= 0 {
        Goldilocks::from_u64(x as u64)
    } else {
        Goldilocks::from_u64(P - (x.unsigned_abs() as u64))
    }
}

pub fn to_i16(x: Goldilocks) -> i16 {
    let v = x.as_canonical_u64();
    if v <= i16::MAX as u64 {
        v as i16
    } else {
        -((P - v) as i16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::XorShift64;

    #[test]
    fn int16_roundtrip() {
        let mut rng = XorShift64::new(20);
        for _ in 0..1000 {
            let x = (rng.next_u64() % 65536) as i16;
            assert_eq!(to_i16(from_i16(x)), x);
        }
    }

    #[test]
    fn int16_dot_product_no_wrap() {
        let mut rng = XorShift64::new(21);
        let k = 4096usize;
        let a: Vec<i16> = (0..k).map(|_| (rng.next_u64() % 65536) as i16).collect();
        let b: Vec<i16> = (0..k).map(|_| (rng.next_u64() % 65536) as i16).collect();

        let int_sum: i64 = a
            .iter()
            .zip(&b)
            .fold(0i64, |acc, (&x, &y)| acc + x as i64 * y as i64);

        let field_sum = a
            .iter()
            .zip(&b)
            .fold(Goldilocks::ZERO, |acc, (&x, &y)| acc + from_i16(x) * from_i16(y));

        let as_signed = if field_sum.as_canonical_u64() > i64::MAX as u64 {
            -((P - field_sum.as_canonical_u64()) as i64)
        } else {
            field_sum.as_canonical_u64() as i64
        };
        assert_eq!(as_signed, int_sum);
    }
}
