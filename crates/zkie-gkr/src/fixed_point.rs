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

/// Embed a signed `i32` fixed-point value in the field. The values must stay
/// bounded by the chosen scale so a length-K dot product does not wrap the
/// 64-bit field; TimesFM activations span ~[-3.4, 3.4], so a scale of 2^12..2^16
/// keeps K=264 accumulations well inside Goldilocks.
pub fn from_i32(x: i32) -> Goldilocks {
    if x >= 0 {
        Goldilocks::from_u64(x as u64)
    } else {
        Goldilocks::from_u64(P - (x.unsigned_abs() as u64))
    }
}

pub fn to_i32(x: Goldilocks) -> i32 {
    let v = x.as_canonical_u64();
    if v <= i32::MAX as u64 {
        v as i32
    } else {
        -((P - v) as i32)
    }
}

/// Embed a signed `i64` value in the field. Used for raw dot-product outputs at
/// scale 2^32 (a `2^16` activation dot a `2^16` weight summed over K terms),
/// which stay inside `(-2^63, 2^63)` for the TimesFM shapes and therefore never
/// wrap the 64-bit field.
pub fn from_i64(x: i64) -> Goldilocks {
    if x >= 0 {
        Goldilocks::from_u64(x as u64)
    } else {
        Goldilocks::from_u64(P - x.unsigned_abs())
    }
}

/// Recover the signed `i64` value embedded by [`from_i64`]. Values must lie in
/// `(-2^63, 2^63)` so the sign convention is unambiguous.
pub fn to_i64(x: Goldilocks) -> i64 {
    let v = x.as_canonical_u64();
    if v <= i64::MAX as u64 {
        v as i64
    } else {
        -((P - v) as i64)
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

    #[test]
    fn int32_roundtrip() {
        let mut rng = XorShift64::new(22);
        for _ in 0..1000 {
            let x = (rng.next_u64() % (1u64 << 32)) as u32 as i32;
            assert_eq!(to_i32(from_i32(x)), x);
        }
    }

    #[test]
    fn int64_roundtrip() {
        let mut rng = XorShift64::new(23);
        for _ in 0..1000 {
            let x = (rng.next_u64() % (1u64 << 60)) as i64 - (1i64 << 59);
            assert_eq!(to_i64(from_i64(x)), x);
        }
    }
}
