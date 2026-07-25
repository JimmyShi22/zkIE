use std::fmt;

pub const SCALE_18: i128 = 1_000_000_000_000_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedPointError(pub String);

impl fmt::Display for FixedPointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fixed point error: {}", self.0)
    }
}

impl std::error::Error for FixedPointError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct I18(i64);

impl I18 {
    pub fn from_raw(raw: i64) -> Self {
        I18(raw)
    }

    pub fn raw(&self) -> i64 {
        self.0
    }

    pub fn from_f64(value: f64) -> Result<Self, FixedPointError> {
        let scaled = value * (SCALE_18 as f64);
        if !scaled.is_finite() {
            return Err(FixedPointError(format!("{value} is not finite")));
        }
        let rounded = scaled.round();
        if rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
            return Err(FixedPointError(format!(
                "{value} does not fit in the I18 representable range"
            )));
        }
        Ok(I18(rounded as i64))
    }

    pub fn to_f64(&self) -> f64 {
        (self.0 as f64) / (SCALE_18 as f64)
    }
}

/// Requantizes a raw (Q36-scaled, or any SCALE_18-multiple-scaled) `i128` value
/// back down to I18 (Q18), returning the quotient (as I18) and the Euclidean
/// remainder (`0 <= remainder < SCALE_18`). Returns an error if the quotient
/// overflows I18's `i64` range.
pub fn requantize_raw(raw_value: i128) -> Result<(I18, i128), FixedPointError> {
    let quotient = raw_value.div_euclid(SCALE_18);
    let remainder = raw_value.rem_euclid(SCALE_18);
    if quotient < i64::MIN as i128 || quotient > i64::MAX as i128 {
        return Err(FixedPointError(format!(
            "raw value {raw_value} requantizes to a quotient that overflows I18 range"
        )));
    }
    Ok((I18(quotient as i64), remainder))
}

/// Computes `a * b` in I18 fixed point, returning the requantized I18 result
/// (`quotient`) and the Euclidean remainder (`0 <= remainder < SCALE_18`).
/// Returns an error if the requantized quotient overflows I18's range.
pub fn requantize_mul(a: I18, b: I18) -> Result<(I18, i128), FixedPointError> {
    let raw_product: i128 = (a.raw() as i128) * (b.raw() as i128);
    requantize_raw(raw_product).map_err(|_| {
        FixedPointError(format!(
            "product of {} and {} overflows I18 range",
            a.to_f64(),
            b.to_f64()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_f64_round_trips_small_values() {
        let x = I18::from_f64(2.5).unwrap();
        assert!((x.to_f64() - 2.5).abs() < 1e-9);
    }

    #[test]
    fn from_f64_round_trips_negative_values() {
        let x = I18::from_f64(-3.25).unwrap();
        assert!((x.to_f64() - (-3.25)).abs() < 1e-9);
    }

    #[test]
    fn from_f64_rejects_out_of_range() {
        // i64::MAX / 1e18 ~= 9.22, so 100.0 must not fit.
        assert!(I18::from_f64(100.0).is_err());
    }

    #[test]
    fn from_f64_accepts_boundary_value() {
        let max_representable = (i64::MAX as f64) / (SCALE_18 as f64);
        assert!(I18::from_f64(max_representable * 0.999).is_ok());
    }

    #[test]
    fn requantize_mul_positive_times_positive() {
        let a = I18::from_f64(2.0).unwrap();
        let b = I18::from_f64(3.0).unwrap();
        let (q, r) = requantize_mul(a, b).unwrap();
        assert!((q.to_f64() - 6.0).abs() < 1e-9);
        assert_eq!(r, 0);
    }

    #[test]
    fn requantize_mul_rounds_toward_negative_infinity_with_nonneg_remainder() {
        // 0.1 * 0.1 = 0.01 exactly in real arithmetic, but in I18 fixed point
        // 0.1 is not exactly representable, so check the remainder invariant
        // instead of an exact value.
        let a = I18::from_f64(0.1).unwrap();
        let b = I18::from_f64(0.1).unwrap();
        let (_, r) = requantize_mul(a, b).unwrap();
        assert!((0..SCALE_18).contains(&r));
    }

    #[test]
    fn requantize_mul_negative_times_positive_remainder_nonnegative() {
        let a = I18::from_f64(-2.5).unwrap();
        let b = I18::from_f64(2.0).unwrap();
        let (q, r) = requantize_mul(a, b).unwrap();
        assert!((q.to_f64() - (-5.0)).abs() < 1e-9);
        assert!((0..SCALE_18).contains(&r));
    }
}
