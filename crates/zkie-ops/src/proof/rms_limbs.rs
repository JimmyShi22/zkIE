//! Exact multi-limb arithmetic of the RMS mean-square statistic.
//!
//! The RMS form sums the squares of the valid cells. For a stream whose
//! cells are bounded by `X = 2^33`, the sum of squares can reach
//! `640 * (2^33)^2 < 2^75.4`, which is larger than the field modulus. The
//! statistic therefore exists only as base-256 digits and a carry chain, not
//! as one field element.
//!
//! Each stream cell is split at 16 bits, `x = lo + 2^16 hi`, with balanced
//! parts (`|lo| < 2^15`, `|hi| <= 2^17`). The square expands to
//! `x^2 = lo^2 + 2^17 lo hi + 2^32 hi^2`, so three per-position sums
//! `S_ll`, `S_lh`, `S_hh` carry the complete integer information; each is
//! below `2^44` and fits one field element. The sum of squares is then
//! combined into a ten-byte digit row with a base-256 carry chain, the
//! half-up rounding adds `valid / 2` to the digit row, and a long division
//! by `valid` produces the mean-square bytes and the division carries.
//!
//! This module is the prover-side exact arithmetic and the plan-side bounds;
//! the product-sumcheck relations, digit lookups and BAT rows are attached by
//! the norm plan that consumes these rows.

use zkie_core::error::{Error, Result};

use crate::registry::round_div_half_up;

/// The fixed positive offset of the signed cross sum `S_lh`, `2^43`. The
/// cross sum lies in `(-2^42, 2^42)`, so its offset digit row is a six-byte
/// nonnegative quantity and the combine chain subtracts the offset at byte
/// seven (`2^17 * 2^43 = 16 * 256^7`).
pub const CROSS_OFFSET: i128 = 1 << 43;

/// The split bit width of one stream cell.
pub const LIMB_BITS: u32 = 16;
/// The split radix.
pub const LIMB_BASE: i128 = 1 << LIMB_BITS;
/// The largest absolute low part: `2^15`.
pub const LOW_BOUND: i128 = 1 << (LIMB_BITS - 1);
/// The smallest accepted stream bound exponent. The plan derives the actual
/// digit counts from its own input bound.
pub const MIN_INPUT_BITS: u32 = 1;

/// The digit widths of the multi-limb RMS row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RmsLimbWidths {
    /// The first divisor of the two-stage division; `first * second = valid`.
    pub first_divisor: u32,
    pub second_divisor: u32,
    /// Base-256 digits of `S_ll`.
    pub s_ll: usize,
    /// Base-256 digits of `S_lh`.
    pub s_lh: usize,
    /// Base-256 digits of `S_hh`.
    pub s_hh: usize,
    /// Base-256 digits of the sum of squares.
    pub v: usize,
    /// Base-256 digits of the mean square.
    pub mean: usize,
}

impl RmsLimbWidths {
    /// Derives the widths from the trusted input bound and valid count. Every
    /// width covers its worst case with a strict margin: the partial sums are
    /// below `2^(8*width)` for every admitted input row.
    pub fn derive(input_bound: u64, valid: u32) -> Result<Self> {
        if input_bound == 0 || valid == 0 {
            return Err(Error::InvalidConfig);
        }
        let x = input_bound as u128;
        let d = valid as u128;
        let low = LOW_BOUND as u128;
        let high = (x + low - 1) / LIMB_BASE as u128;
        let s_ll = d
            .checked_mul(low.checked_mul(low).ok_or(Error::ResourceLimit)?)
            .ok_or(Error::ResourceLimit)?;
        let s_lh = d
            .checked_mul(low.checked_mul(high).ok_or(Error::ResourceLimit)?)
            .ok_or(Error::ResourceLimit)?;
        let s_hh = d
            .checked_mul(high.checked_mul(high).ok_or(Error::ResourceLimit)?)
            .ok_or(Error::ResourceLimit)?;
        let v = s_ll
            .checked_add(
                2u128
                    .checked_mul(1u128 << LIMB_BITS)
                    .and_then(|factor| factor.checked_mul(s_lh))
                    .ok_or(Error::ResourceLimit)?,
            )
            .and_then(|value| {
                (1u128 << (2 * LIMB_BITS))
                    .checked_mul(s_hh)
                    .and_then(|high| value.checked_add(high))
            })
            .ok_or(Error::ResourceLimit)?;
        let width = |value: u128| -> Result<usize> {
            let bits = 128 - value.leading_zeros();
            Ok(((bits + 7) / 8).max(1) as usize)
        };
        // The cross-sum offset cancels at byte seven, so the combined row
        // must cover at least eight bytes even for a tiny input bound.
        let v_width = width(v)?.max(10);
        let first_divisor = if valid == 0 {
            return Err(Error::InvalidConfig);
        } else {
            let mut divisor: u32 = 1;
            while divisor < 256 && valid % (divisor * 2) == 0 {
                divisor *= 2;
            }
            divisor
        };
        let second_divisor = valid / first_divisor;
        if first_divisor == 0 || first_divisor > 255 || second_divisor == 0 || second_divisor > 255 {
            return Err(Error::InvalidConfig);
        }
        let widths = Self {
            first_divisor,
            second_divisor,
            s_ll: width(s_ll)?.max(5),
            // The cross sum is stored with its fixed offset, so its digit
            // row must cover `S_lh + CROSS_OFFSET` even when the bound is
            // small.
            s_lh: width(s_lh)?.max(width(CROSS_OFFSET as u128)?),
            s_hh: width(s_hh)?.max(6),
            v: v_width,
            mean: v_width,
        };
        widths.validate()?;
        Ok(widths)
    }

    pub fn validate(&self) -> Result<()> {
        if self.first_divisor == 0
            || self.first_divisor > 255
            || self.second_divisor == 0
            || self.second_divisor > 255
        {
            return Err(Error::InvalidConfig);
        }
        // The three partial sums are field elements and stay within seven
        // bytes. The sum of squares and the quotient exist only as digits;
        // their rows may use up to ten bytes.
        for width in [self.s_ll, self.s_lh, self.s_hh] {
            if width == 0 || width > 7 {
                return Err(Error::InvalidConfig);
            }
        }
        for width in [self.v, self.mean] {
            if width == 0 || width > 10 {
                return Err(Error::InvalidConfig);
            }
        }
        Ok(())
    }

    fn format_fault(&self, width: usize, value: u128) -> bool {
        width == 0 || width > 10 || value >= (1u128 << (8 * width as u32))
    }
}

/// One position's exact multi-limb RMS row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RmsLimbRow {
    /// The split low and high parts of every valid cell.
    pub low: Vec<i64>,
    pub high: Vec<i64>,
    /// The three bounded sums.
    pub s_ll: i128,
    pub s_lh: i128,
    pub s_hh: i128,
    /// Their base-256 digits, least significant first.
    pub s_ll_digits: Vec<u8>,
    pub s_lh_digits: Vec<u8>,
    pub s_hh_digits: Vec<u8>,
    /// The sum-of-squares digits and their carry chain, least significant
    /// first. Carry `i` is the carry out of byte `i`.
    pub v_digits: Vec<u8>,
    pub v_carries: Vec<i16>,
    /// The digits of `V + valid/2` and its carry chain.
    pub v_plus_digits: Vec<u8>,
    pub v_plus_carries: Vec<u8>,
    /// The quotient of `V + valid/2` by 128 and its byte carries.
    pub q1_digits: Vec<u8>,
    pub shift_carries: Vec<u8>,
    pub shift_slack: Vec<u8>,
    /// The mean-square digits (the quotient by five of the shifted value) and
    /// the stage-two carries.
    pub mean_digits: Vec<u8>,
    pub division_carries: Vec<u8>,
    pub division_slack: Vec<u8>,
    /// The complement of the top mean-square byte to 31, forcing the mean
    /// below 2^61.
    pub mean_top_slack: u8,
    /// The rounded mean square and the division remainder.
    pub mean_square: i128,
    pub remainder: u16,
}

impl RmsLimbRow {
    /// Splits one cell into balanced parts. A value outside the plan bound is
    /// refused, never reduced.
    pub fn split(value: i64) -> Result<(i64, i64)> {
        let value = value as i128;
        let low = (value + LOW_BOUND).rem_euclid(LIMB_BASE) - LOW_BOUND;
        if low < -LOW_BOUND || low >= LOW_BOUND {
            return Err(Error::BoundViolation);
        }
        let high = (value - low) / LIMB_BASE;
        Ok((
            i64::try_from(low).map_err(|_| Error::BoundViolation)?,
            i64::try_from(high).map_err(|_| Error::BoundViolation)?,
        ))
    }

    /// Computes the exact row of one position. `values` holds the padded row;
    /// only the first `valid` cells enter the statistics.
    pub fn from_values(
        values: &[i64],
        valid: usize,
        input_bound: u64,
        widths: RmsLimbWidths,
    ) -> Result<Self> {
        if valid == 0 || valid > values.len() || input_bound == 0 {
            return Err(Error::InvalidConfig);
        }
        if widths.first_divisor as usize * widths.second_divisor as usize != valid {
            return Err(Error::InvalidConfig);
        }
        let bound = i64::try_from(input_bound).map_err(|_| Error::ResourceLimit)?;
        let mut low = Vec::with_capacity(valid);
        let mut high = Vec::with_capacity(valid);
        let mut s_ll: i128 = 0;
        let mut s_lh: i128 = 0;
        let mut s_hh: i128 = 0;
        for value in &values[..valid] {
            if value.abs() > bound {
                return Err(Error::BoundViolation);
            }
            let (lo, hi) = Self::split(*value)?;
            s_ll = s_ll
                .checked_add((lo as i128) * (lo as i128))
                .ok_or(Error::BoundViolation)?;
            s_lh = s_lh
                .checked_add((lo as i128) * (hi as i128))
                .ok_or(Error::BoundViolation)?;
            s_hh = s_hh
                .checked_add((hi as i128) * (hi as i128))
                .ok_or(Error::BoundViolation)?;
            low.push(lo);
            high.push(hi);
        }
        let s_ll_digits = digits(s_ll, widths.s_ll)?;
        let s_lh_digits = digits(
            s_lh
                .checked_add(CROSS_OFFSET)
                .ok_or(Error::BoundViolation)?,
            widths.s_lh,
        )?;
        let s_hh_digits = digits(s_hh, widths.s_hh)?;
        let v = (s_ll as i128)
            .checked_add(
                2i128
                    .checked_mul(LIMB_BASE)
                    .and_then(|factor| factor.checked_mul(s_lh))
                    .ok_or(Error::BoundViolation)?,
            )
            .and_then(|value| {
                (LIMB_BASE * LIMB_BASE)
                    .checked_mul(s_hh)
                    .and_then(|high| value.checked_add(high))
            })
            .ok_or(Error::BoundViolation)?;
        let (v_digits, v_carries) = combine(
            &s_ll_digits,
            &s_lh_digits,
            &s_hh_digits,
            widths.v,
        )?;
        let expected_v = digits_value(&v_digits)?;
        if expected_v != v as u128 {
            return Err(Error::InvalidProof);
        }
        let half = (valid / 2) as u16;
        let (v_plus_digits, v_plus_carries) = add_small(&v_digits, half, widths.v)?;
        let v_plus = expected_v
            .checked_add(half as u128)
            .ok_or(Error::ResourceLimit)?;
        let divisor = valid as u128;
        let mean = v_plus / divisor;
        let remainder = u16::try_from(v_plus % divisor).map_err(|_| Error::BoundViolation)?;
        // The valid count factors into two byte-sized divisors; dividing in
        // two stages keeps every carry in a byte and makes the canonical
        // remainder bound a byte slack relation.
        let first = widths.first_divisor;
        let second = widths.second_divisor;
        let (q1_digits, shift_carries) = divide(&v_plus_digits, first, widths.mean)?;
        let shift_slack = shift_carries
            .iter()
            .map(|carry| {
                (first as u8 - 1)
                    .checked_sub(*carry)
                    .ok_or(Error::BoundViolation)
            })
            .collect::<Result<Vec<_>>>()?;
        let (mean_digits, division_carries) = divide(&q1_digits, second, widths.mean)?;
        let division_slack = division_carries
            .iter()
            .map(|carry| {
                (second as u8 - 1)
                    .checked_sub(*carry)
                    .ok_or(Error::BoundViolation)
            })
            .collect::<Result<Vec<_>>>()?;
        let mean_top_slack = 31u8
            .checked_sub(*mean_digits.get(7).ok_or(Error::ShapeMismatch)?)
            .ok_or(Error::BoundViolation)?;
        if digits_value(&mean_digits)? != mean {
            return Err(Error::InvalidProof);
        }
        let mean_square = mean as i128;
        let rounded = round_div_half_up(v, valid as i128)?;
        if mean_square != rounded {
            return Err(Error::InvalidProof);
        }
        Ok(Self {
            low,
            high,
            s_ll,
            s_lh,
            s_hh,
            s_ll_digits,
            s_lh_digits,
            s_hh_digits,
            v_digits,
            v_carries,
            v_plus_digits,
            v_plus_carries,
            q1_digits,
            shift_carries,
            shift_slack,
            mean_digits,
            division_carries,
            division_slack,
            mean_top_slack,
            mean_square,
            remainder,
        })
    }

    /// Recomputes the row and checks every digit and carry byte.
    pub fn check(
        &self,
        values: &[i64],
        valid: usize,
        input_bound: u64,
        widths: RmsLimbWidths,
    ) -> Result<()> {
        let expected = Self::from_values(values, valid, input_bound, widths)?;
        if *self != expected {
            return Err(Error::InvalidProof);
        }
        Ok(())
    }
}

fn digits(value: i128, width: usize) -> Result<Vec<u8>> {
    if value < 0 {
        return Err(Error::BoundViolation);
    }
    let mut value = value as u128;
    let mut out = Vec::with_capacity(width);
    for _ in 0..width {
        out.push((value & 0xff) as u8);
        value >>= 8;
    }
    if value != 0 {
        return Err(Error::BoundViolation);
    }
    Ok(out)
}

fn digits_value(digits: &[u8]) -> Result<u128> {
    let mut value = 0u128;
    for (index, digit) in digits.iter().enumerate() {
        value |= (u128::from(*digit))
            .checked_shl(8 * index as u32)
            .ok_or(Error::ResourceLimit)?;
    }
    Ok(value)
}

/// Combines the three sums into the sum-of-squares byte row. Byte `i` of the
/// row receives `S_ll` at shift 0, `2 * S_lh` at shift 2 bytes and `S_hh` at
/// shift 4 bytes; every carry is one byte.
fn combine(
    s_ll: &[u8],
    s_lh: &[u8],
    s_hh: &[u8],
    width: usize,
) -> Result<(Vec<u8>, Vec<i16>)> {
    let mut out = vec![0u8; width];
    let mut carries = vec![0i16; width];
    let mut carry: i64 = 0;
    for index in 0..width {
        let mut sum = carry;
        if let Some(digit) = s_ll.get(index) {
            sum += i64::from(*digit);
        }
        if index >= 2 {
            if let Some(digit) = s_lh.get(index - 2) {
                sum += 2 * i64::from(*digit);
            }
        }
        if index >= 4 {
            if let Some(digit) = s_hh.get(index - 4) {
                sum += i64::from(*digit);
            }
        }
        if index == 7 {
            // The cross-sum offset: 2^17 * 2^43 = 16 * 256^7.
            sum -= 16;
        }
        let byte = sum.rem_euclid(256);
        carry = sum.div_euclid(256);
        out[index] = byte as u8;
        if index + 1 < width {
            carries[index] =
                i16::try_from(carry).map_err(|_| Error::BoundViolation)?;
        } else if carry != 0 {
            return Err(Error::BoundViolation);
        }
    }
    Ok((out, carries))
}

/// Adds a small nonnegative value to a byte row, one byte carry chain.
fn add_small(digits: &[u8], addend: u16, width: usize) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut out = vec![0u8; width];
    let mut carries = vec![0u8; width];
    let mut carry: u32 = 0;
    for index in 0..width {
        let mut sum = carry + u32::from(digits.get(index).copied().unwrap_or(0));
        if index == 0 {
            sum += u32::from(addend);
        }
        out[index] = (sum & 0xff) as u8;
        carry = sum >> 8;
        if index + 1 < width {
            if carry > u32::from(u8::MAX) {
                return Err(Error::BoundViolation);
            }
            carries[index] = carry as u8;
        } else if carry != 0 {
            return Err(Error::BoundViolation);
        }
    }
    Ok((out, carries))
}

/// Long division of a byte row by a small divisor, most significant byte
/// first. In little-endian indices the bytes and carries satisfy
/// `up_{i+1} * 256 + digit_i = divisor * q_i + down_i`, with `up_width = 0`;
/// `down_0` is the division remainder.
fn divide(digits: &[u8], divisor: u32, width: usize) -> Result<(Vec<u8>, Vec<u8>)> {
    if divisor == 0 || divisor > 256 {
        return Err(Error::InvalidConfig);
    }
    let mut out = vec![0u8; width];
    let mut carries = vec![0u8; width];
    let mut up: u32 = 0;
    for index in (0..width).rev() {
        let current = up
            .checked_mul(256)
            .and_then(|value| value.checked_add(u32::from(digits.get(index).copied().unwrap_or(0))))
            .ok_or(Error::ResourceLimit)?;
        let quotient = current / divisor;
        let down = current % divisor;
        if quotient > u32::from(u8::MAX) {
            return Err(Error::BoundViolation);
        }
        out[index] = quotient as u8;
        up = down;
        carries[index] = u8::try_from(down).map_err(|_| Error::BoundViolation)?;
    }
    Ok((out, carries))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn widths() -> RmsLimbWidths {
        RmsLimbWidths::derive(1 << 33, 640).unwrap()
    }

    #[test]
    fn the_widths_cover_the_worst_case() {
        let widths = widths();
        // (2^33)^2 * 640 < 2^75.4 and the three partial sums fit six bytes.
        assert_eq!(widths.s_ll, 5);
        assert_eq!(widths.s_lh, 6);
        assert_eq!(widths.s_hh, 6);
        assert_eq!(widths.v, 10);
    }

    #[test]
    fn the_worst_case_row_reconstructs_every_digit() {
        let values = vec![1i64 << 33; 640];
        let row = RmsLimbRow::from_values(&values, 640, 1 << 33, widths()).unwrap();
        let v = 640i128 * (1i128 << 66);
        assert_eq!(row.mean_square, round_div_half_up(v, 640).unwrap());
        assert_eq!(digits_value(&row.v_digits).unwrap(), v as u128);
        assert_eq!(row.v_carries.len(), 10);
        assert_eq!(row.division_carries.len(), 10);
        assert_eq!(row.shift_carries.len(), 10);
        row.check(&values, 640, 1 << 33, widths()).unwrap();
    }

    #[test]
    fn the_measured_case_is_admitted_and_a_tampered_row_is_refused() {
        // The measured final-norm input of the gemma3 seq16 case.
        let mut values = vec![0i64; 1024];
        values[0] = -7_859_748_442;
        values[1] = 7_859_748_442;
        for index in 0..640 {
            values[index] = values[index].wrapping_add((index as i64 % 17) - 8);
        }
        let row = RmsLimbRow::from_values(&values, 640, 1 << 33, widths()).unwrap();
        row.check(&values, 640, 1 << 33, widths()).unwrap();
        let mut changed = row.clone();
        changed.v_digits[3] ^= 1;
        assert!(changed.check(&values, 640, 1 << 33, widths()).is_err());
        let mut changed = row.clone();
        changed.division_carries[5] += 1;
        assert!(changed.check(&values, 640, 1 << 33, widths()).is_err());
        let mut changed = row;
        changed.low[0] += 1;
        assert!(changed.check(&values, 640, 1 << 33, widths()).is_err());
    }

    #[test]
    fn an_out_of_bound_cell_is_refused() {
        let mut values = vec![0i64; 640];
        values[3] = (1i64 << 33) + 1;
        assert!(RmsLimbRow::from_values(&values, 640, 1 << 33, widths()).is_err());
    }
}
