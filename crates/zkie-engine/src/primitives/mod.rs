//! Exact integer helpers and CPU primitive implementations.
//!
//! No host floating point takes part in a witness computation.

pub mod codec;
pub mod field;
pub mod transcript;

use zkie_core::error::{Error, Result};

/// Fixed-point requantization with mathematical floor:
/// `rq(a, k) = floor((a + 2^(k-1)) / 2^k)` for `k >= 1`, and `rq(a, 0) = a`.
pub fn rq(a: i128, k: u32) -> Result<i128> {
    if k == 0 {
        return Ok(a);
    }
    if k >= 128 {
        return Err(Error::BoundViolation);
    }
    let half = 1i128 << (k - 1);
    let numerator = a.checked_add(half).ok_or(Error::BoundViolation)?;
    // Arithmetic right shift is floor division by a power of two.
    Ok(numerator >> k)
}

/// Euclidean quotient and remainder. A zero divisor is rejected.
pub fn div_rem_checked(a: i128, b: i128) -> Result<(i128, i128)> {
    if b == 0 {
        return Err(Error::DivisionByZero);
    }
    let quotient = a.checked_div_euclid(b).ok_or(Error::BoundViolation)?;
    let remainder = a.checked_rem_euclid(b).ok_or(Error::BoundViolation)?;
    Ok((quotient, remainder))
}

/// `a + b + carry` as `(low, carry_out)`.
pub fn add_with_carry(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let sum = a as u128 + b as u128 + carry as u128;
    (sum as u64, (sum >> 64) as u64)
}

/// `a - b - borrow` as `(low, borrow_out)`.
pub fn sub_with_borrow(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let minuend = a as u128;
    let subtrahend = b as u128 + borrow as u128;
    if minuend >= subtrahend {
        ((minuend - subtrahend) as u64, 0)
    } else {
        ((minuend + (1u128 << 64) - subtrahend) as u64, 1)
    }
}

/// `a * b + add + carry` as `(low, carry_out)`.
pub fn mul_add_carry(a: u64, b: u64, add: u64, carry: u64) -> (u64, u64) {
    let product = a as u128 * b as u128 + add as u128 + carry as u128;
    (product as u64, (product >> 64) as u64)
}

/// A little-endian 256-bit unsigned integer with explicit limb/carry arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WideU256 {
    pub limbs: [u64; 4],
}

impl WideU256 {
    pub const ZERO: Self = Self { limbs: [0; 4] };

    pub const fn from_u64(value: u64) -> Self {
        Self {
            limbs: [value, 0, 0, 0],
        }
    }

    pub const fn from_u128(value: u128) -> Self {
        Self {
            limbs: [value as u64, (value >> 64) as u64, 0, 0],
        }
    }

    pub fn to_u128(self) -> Option<u128> {
        if self.limbs[2] != 0 || self.limbs[3] != 0 {
            return None;
        }
        Some(self.limbs[0] as u128 | ((self.limbs[1] as u128) << 64))
    }

    pub fn is_zero(self) -> bool {
        self.limbs == [0; 4]
    }

    pub fn bit(self, index: u32) -> bool {
        if index >= 256 {
            return false;
        }
        (self.limbs[(index / 64) as usize] >> (index % 64)) & 1 == 1
    }

    fn set_bit(&mut self, index: u32) {
        self.limbs[(index / 64) as usize] |= 1u64 << (index % 64);
    }

    pub fn cmp_value(self, other: Self) -> core::cmp::Ordering {
        for index in (0..4).rev() {
            match self.limbs[index].cmp(&other.limbs[index]) {
                core::cmp::Ordering::Equal => continue,
                order => return order,
            }
        }
        core::cmp::Ordering::Equal
    }

    pub fn checked_add(self, other: Self) -> Option<Self> {
        let mut out = [0u64; 4];
        let mut carry = 0u64;
        for index in 0..4 {
            let (low, next) = add_with_carry(self.limbs[index], other.limbs[index], carry);
            out[index] = low;
            carry = next;
        }
        if carry != 0 {
            None
        } else {
            Some(Self { limbs: out })
        }
    }

    pub fn checked_sub(self, other: Self) -> Option<Self> {
        if self.cmp_value(other) == core::cmp::Ordering::Less {
            return None;
        }
        Some(self.wrapping_sub(other))
    }

    fn wrapping_sub(self, other: Self) -> Self {
        let mut out = [0u64; 4];
        let mut borrow = 0u64;
        for index in 0..4 {
            let (low, next) = sub_with_borrow(self.limbs[index], other.limbs[index], borrow);
            out[index] = low;
            borrow = next;
        }
        debug_assert_eq!(borrow, 0);
        Self { limbs: out }
    }

    fn shl_one_dropping_top(self) -> Self {
        let mut out = [0u64; 4];
        let mut carry = 0u64;
        for index in 0..4 {
            out[index] = (self.limbs[index] << 1) | carry;
            carry = self.limbs[index] >> 63;
        }
        Self { limbs: out }
    }

    pub fn checked_shl(self, bits: u32) -> Option<Self> {
        if bits >= 256 {
            return if self.is_zero() { Some(Self::ZERO) } else { None };
        }
        let mut out = self;
        for _ in 0..bits {
            if out.limbs[3] >> 63 != 0 {
                return None;
            }
            out = out.shl_one_dropping_top();
        }
        Some(out)
    }

    pub fn shr(self, bits: u32) -> Self {
        if bits >= 256 {
            return Self::ZERO;
        }
        let mut out = [0u64; 4];
        let limb_shift = (bits / 64) as usize;
        let bit_shift = bits % 64;
        for index in 0..4 {
            let source = index + limb_shift;
            if source >= 4 {
                continue;
            }
            let mut value = self.limbs[source] >> bit_shift;
            if bit_shift != 0 && source + 1 < 4 {
                value |= self.limbs[source + 1] << (64 - bit_shift);
            }
            out[index] = value;
        }
        Self { limbs: out }
    }

    pub fn mul_u64(self, factor: u64) -> Option<Self> {
        let mut out = [0u64; 4];
        let mut carry = 0u64;
        for index in 0..4 {
            let (low, next) = mul_add_carry(self.limbs[index], factor, 0, carry);
            out[index] = low;
            carry = next;
        }
        if carry != 0 {
            None
        } else {
            Some(Self { limbs: out })
        }
    }

    /// Binary long division. A zero divisor is rejected.
    pub fn div_rem(self, divisor: Self) -> Result<(Self, Self)> {
        if divisor.is_zero() {
            return Err(Error::DivisionByZero);
        }
        let mut quotient = Self::ZERO;
        let mut remainder = Self::ZERO;
        for index in (0..256).rev() {
            // The true remainder before this step is below the divisor, so a set
            // top bit means the doubled value is at least the divisor.
            let top = remainder.limbs[3] >> 63;
            let mut candidate = remainder.shl_one_dropping_top();
            if self.bit(index) {
                candidate.limbs[0] |= 1;
            }
            if top != 0 || candidate.cmp_value(divisor) != core::cmp::Ordering::Less {
                // 2^256 + candidate - divisor fits in 256 bits because the
                // candidate was below the (large) divisor before doubling.
                remainder = candidate.wrapping_sub(divisor);
                quotient.set_bit(index);
            } else {
                remainder = candidate;
            }
        }
        Ok((quotient, remainder))
    }
}
