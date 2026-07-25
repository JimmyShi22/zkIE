use halo2_proofs::circuit::Value;
pub use halo2_proofs::halo2curves::bn256::Fr;
use halo2_proofs::halo2curves::ff::PrimeField;

pub fn i64_to_fr(v: i64) -> Fr {
    if v >= 0 {
        Fr::from(v as u64)
    } else {
        -Fr::from((-(v as i128)) as u64)
    }
}

pub fn i128_to_fr(v: i128) -> Fr {
    if v >= 0 {
        Fr::from_u128(v as u128)
    } else {
        -Fr::from_u128((-v) as u128)
    }
}

/// Shift applied to signed i64 values before range-checking them with
/// `RangeCheckChip`, which only supports unsigned ranges: adding `2^63` maps
/// the full `i64` range onto `[0, 2^64)`.
pub const SIGNED_SHIFT: i128 = 1i128 << 63;

/// Shifts a signed `i64` by `SIGNED_SHIFT` and returns both the field-element
/// witness and the raw shifted value (needed by `RangeCheckChip::assign`'s
/// bit-decomposition), so a signed value can be range-checked as unsigned.
pub fn shifted_i64_witness(v: i64) -> (Value<Fr>, Value<i128>) {
    let shifted = (v as i128) + SIGNED_SHIFT;
    (Value::known(i128_to_fr(shifted)), Value::known(shifted))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_i64_round_trips_via_u64() {
        let fr = i64_to_fr(42);
        assert_eq!(fr, Fr::from(42u64));
    }

    #[test]
    fn negative_i64_is_additive_inverse_in_field() {
        let fr = i64_to_fr(-42);
        assert_eq!(fr, -Fr::from(42u64));
    }

    #[test]
    fn zero_maps_to_field_zero() {
        assert_eq!(i64_to_fr(0), Fr::zero());
    }

    #[test]
    fn negative_i128_is_additive_inverse_in_field() {
        let fr = i128_to_fr(-123456789012345i128);
        assert_eq!(fr, -Fr::from_u128(123456789012345u128));
    }
}
