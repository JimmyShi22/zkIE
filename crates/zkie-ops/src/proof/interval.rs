//! Directed interval arithmetic for protocol parameter derivation.
//!
//! The planner compares error bounds with a target that is close to an integer
//! position count. A floating-point comparison can move that count by one, so
//! this module keeps the arithmetic exact where it can and otherwise rounds in
//! a named direction. `Big` is an arbitrary-precision non-negative integer,
//! `Real` is a mantissa times a power of two cut to 192 bits with a rounding
//! direction, and `Bound` is a pair of `Real` values that contains the true
//! value. A decision reads the safe end of a `Bound`; an operation that leaves
//! its domain returns a bad bound, which the caller treats as a rejection.

use std::cmp::Ordering;

const REAL_BITS: usize = 192;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Big {
    words: Vec<u32>,
}

impl Big {
    pub fn zero() -> Self {
        Self { words: Vec::new() }
    }

    pub fn from_u64(value: u64) -> Self {
        let mut words = Vec::new();
        let mut value = value;
        while value != 0 {
            words.push(value as u32);
            value >>= 32;
        }
        Self { words }
    }

    pub fn two_to(power: usize) -> Self {
        Self::from_u64(1).shl(power)
    }

    pub fn is_zero(&self) -> bool {
        self.words.is_empty()
    }

    pub fn bits(&self) -> usize {
        let Some(top) = self.words.last() else {
            return 0;
        };
        32 * (self.words.len() - 1) + (32 - top.leading_zeros() as usize)
    }

    pub fn any_below(&self, count: usize) -> bool {
        for (index, word) in self.words.iter().enumerate() {
            let start = 32 * index;
            if start >= count {
                break;
            }
            let keep = count - start;
            let masked = if keep < 32 {
                *word & ((1u32 << keep) - 1)
            } else {
                *word
            };
            if masked != 0 {
                return true;
            }
        }
        false
    }

    pub fn fits_u64(&self) -> bool {
        self.words.len() <= 2
    }

    pub fn to_u64(&self) -> u64 {
        assert!(self.fits_u64(), "integer does not fit u64");
        let low = self.words.first().copied().unwrap_or(0) as u64;
        let high = self.words.get(1).copied().unwrap_or(0) as u64;
        low | (high << 32)
    }

    pub fn cmp(&self, other: &Self) -> Ordering {
        if self.words.len() != other.words.len() {
            return self.words.len().cmp(&other.words.len());
        }
        for index in (0..self.words.len()).rev() {
            if self.words[index] != other.words[index] {
                return self.words[index].cmp(&other.words[index]);
            }
        }
        Ordering::Equal
    }

    pub fn shl(&self, count: usize) -> Self {
        if self.is_zero() || count == 0 {
            return self.clone();
        }
        let word_shift = count / 32;
        let bit_shift = count % 32;
        let mut words = vec![0u32; self.words.len() + word_shift + 1];
        for (index, word) in self.words.iter().enumerate() {
            let value = (*word as u64) << bit_shift;
            words[index + word_shift] |= value as u32;
            words[index + word_shift + 1] |= (value >> 32) as u32;
        }
        let mut answer = Self { words };
        answer.trim();
        answer
    }

    pub fn shr(&self, count: usize) -> Self {
        let word_shift = count / 32;
        let bit_shift = count % 32;
        if word_shift >= self.words.len() {
            return Self::zero();
        }
        let mut words = vec![0u32; self.words.len() - word_shift];
        for index in word_shift..self.words.len() {
            let mut value = self.words[index] as u64;
            if index + 1 < self.words.len() {
                value |= (self.words[index + 1] as u64) << 32;
            }
            words[index - word_shift] = (value >> bit_shift) as u32;
        }
        let mut answer = Self { words };
        answer.trim();
        answer
    }

    pub fn add(&self, other: &Self) -> Self {
        let length = self.words.len().max(other.words.len()) + 1;
        let mut words = vec![0u32; length];
        let mut carry = 0u64;
        for index in 0..length {
            let mut value = carry;
            if index < self.words.len() {
                value += self.words[index] as u64;
            }
            if index < other.words.len() {
                value += other.words[index] as u64;
            }
            words[index] = value as u32;
            carry = value >> 32;
        }
        let mut answer = Self { words };
        answer.trim();
        answer
    }

    pub fn sub(&self, other: &Self) -> Self {
        assert!(self.cmp(other) != Ordering::Less, "negative difference");
        let mut words = vec![0u32; self.words.len()];
        let mut borrow = 0i64;
        for index in 0..self.words.len() {
            let right = if index < other.words.len() {
                other.words[index] as i64
            } else {
                0
            };
            let value = self.words[index] as i64 - right + borrow;
            words[index] = value as u32;
            borrow = value >> 32;
        }
        let mut answer = Self { words };
        answer.trim();
        answer
    }

    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        let mut words = vec![0u32; self.words.len() + other.words.len()];
        for (left_index, left) in self.words.iter().enumerate() {
            let mut carry = 0u64;
            for (right_index, right) in other.words.iter().enumerate() {
                let slot = left_index + right_index;
                let value =
                    (*left as u64) * (*right as u64) + words[slot] as u64 + carry;
                words[slot] = value as u32;
                carry = value >> 32;
            }
            words[left_index + other.words.len()] = carry as u32;
        }
        let mut answer = Self { words };
        answer.trim();
        answer
    }

    pub fn divmod(&self, divisor: &Self) -> (Self, Self) {
        assert!(!divisor.is_zero(), "division by zero");
        if self.cmp(divisor) == Ordering::Less {
            return (Self::zero(), self.clone());
        }
        if divisor.words.len() == 1 {
            let divisor_word = divisor.words[0] as u64;
            let mut words = vec![0u32; self.words.len()];
            let mut carry = 0u64;
            for index in (0..self.words.len()).rev() {
                carry = (carry << 32) | self.words[index] as u64;
                words[index] = (carry / divisor_word) as u32;
                carry %= divisor_word;
            }
            let mut quotient = Self { words };
            quotient.trim();
            return (quotient, Self::from_u64(carry));
        }
        let mut shift = 0usize;
        let mut top = *divisor.words.last().expect("non-empty");
        while top & 0x8000_0000u32 == 0 {
            shift += 1;
            top <<= 1;
        }
        let shifted_divisor = divisor.shl(shift);
        let mut shifted_self = self.shl(shift);
        let divisor_len = shifted_divisor.words.len();
        shifted_self
            .words
            .resize(shifted_self.words.len().max(self.words.len() + 1), 0);
        let quotient_len = shifted_self.words.len() - divisor_len;
        let mut quotient_words = vec![0u32; quotient_len];
        for index in (0..quotient_len).rev() {
            let numerator = ((shifted_self.words[index + divisor_len] as u64) << 32)
                | shifted_self.words[index + divisor_len - 1] as u64;
            let mut trial = numerator / shifted_divisor.words[divisor_len - 1] as u64;
            let mut remainder =
                numerator % shifted_divisor.words[divisor_len - 1] as u64;
            while trial >> 32 != 0
                || trial * shifted_divisor.words[divisor_len - 2] as u64
                    > ((remainder << 32) | shifted_self.words[index + divisor_len - 2] as u64)
            {
                trial -= 1;
                remainder += shifted_divisor.words[divisor_len - 1] as u64;
                if remainder >> 32 != 0 {
                    break;
                }
            }
            let mut borrow = 0i64;
            let mut carry = 0u64;
            for inner in 0..divisor_len {
                carry += trial * shifted_divisor.words[inner] as u64;
                let value = shifted_self.words[index + inner] as i64
                    - (carry as u32) as i64
                    + borrow;
                shifted_self.words[index + inner] = value as u32;
                borrow = value >> 32;
                carry >>= 32;
            }
            let value =
                shifted_self.words[index + divisor_len] as i64 - carry as i64 + borrow;
            shifted_self.words[index + divisor_len] = value as u32;
            if value < 0 {
                trial -= 1;
                let mut carry = 0u64;
                for inner in 0..divisor_len {
                    carry += shifted_self.words[index + inner] as u64
                        + shifted_divisor.words[inner] as u64;
                    shifted_self.words[index + inner] = carry as u32;
                    carry >>= 32;
                }
                shifted_self.words[index + divisor_len] =
                    shifted_self.words[index + divisor_len].wrapping_add(carry as u32);
            }
            quotient_words[index] = trial as u32;
        }
        let mut quotient = Self {
            words: quotient_words,
        };
        quotient.trim();
        let mut remainder = shifted_self.shr(shift);
        remainder.trim();
        (quotient, remainder)
    }

    pub fn isqrt(&self) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let mut estimate = Self::two_to((self.bits() + 1) / 2);
        loop {
            let (quotient, _) = self.divmod(&estimate);
            let next = estimate.add(&quotient).shr(1);
            if next.cmp(&estimate) != Ordering::Less {
                return estimate;
            }
            estimate = next;
        }
    }

    pub fn pow(mut base: Self, mut exponent: u64) -> Self {
        let mut answer = Self::from_u64(1);
        while exponent != 0 {
            if exponent & 1 == 1 {
                answer = answer.mul(&base);
            }
            exponent >>= 1;
            if exponent != 0 {
                base = base.mul(&base);
            }
        }
        answer
    }

    fn trim(&mut self) {
        while self.words.last().is_some_and(|word| *word == 0) {
            self.words.pop();
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    Down,
    Up,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Real {
    pub mantissa: Big,
    pub exponent: i64,
}

impl Real {
    pub fn zero() -> Self {
        Self {
            mantissa: Big::zero(),
            exponent: 0,
        }
    }

    pub fn one() -> Self {
        Self::from_u64(1)
    }

    pub fn from_u64(value: u64) -> Self {
        Self {
            mantissa: Big::from_u64(value),
            exponent: 0,
        }
    }

    pub fn from_big(value: Big) -> Self {
        Self {
            mantissa: value,
            exponent: 0,
        }
    }

    pub fn two_to(power: i64) -> Self {
        Self {
            mantissa: Big::from_u64(1),
            exponent: power,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.mantissa.is_zero()
    }

    pub fn top(&self) -> i64 {
        self.mantissa.bits() as i64 + self.exponent
    }

    pub fn cut(&self, direction: Dir) -> Self {
        let bits = self.mantissa.bits();
        if bits <= REAL_BITS {
            return self.clone();
        }
        let shift = bits - REAL_BITS;
        let mut mantissa = self.mantissa.shr(shift);
        if direction == Dir::Up && self.mantissa.any_below(shift) {
            mantissa = mantissa.add(&Big::from_u64(1));
        }
        Self {
            mantissa,
            exponent: self.exponent + shift as i64,
        }
    }

    pub fn cmp(&self, other: &Self) -> Ordering {
        if self.is_zero() || other.is_zero() {
            return if self.is_zero() && other.is_zero() {
                Ordering::Equal
            } else if self.is_zero() {
                Ordering::Less
            } else {
                Ordering::Greater
            };
        }
        if self.top() != other.top() {
            return self.top().cmp(&other.top());
        }
        if self.exponent >= other.exponent {
            self.mantissa
                .shl((self.exponent - other.exponent) as usize)
                .cmp(&other.mantissa)
        } else {
            self.mantissa
                .cmp(&other.mantissa.shl((other.exponent - self.exponent) as usize))
        }
    }

    pub fn add(&self, other: &Self, direction: Dir) -> Self {
        if self.is_zero() {
            return other.clone();
        }
        if other.is_zero() {
            return self.clone();
        }
        let exponent = self.exponent.min(other.exponent);
        let left = self
            .mantissa
            .shl((self.exponent - exponent) as usize);
        let right = other
            .mantissa
            .shl((other.exponent - exponent) as usize);
        Self {
            mantissa: left.add(&right),
            exponent,
        }
        .cut(direction)
    }

    pub fn sub(&self, other: &Self, direction: Dir) -> Self {
        if other.is_zero() {
            return self.clone();
        }
        let exponent = self.exponent.min(other.exponent);
        let left = self
            .mantissa
            .shl((self.exponent - exponent) as usize);
        let right = other
            .mantissa
            .shl((other.exponent - exponent) as usize);
        Self {
            mantissa: left.sub(&right),
            exponent,
        }
        .cut(direction)
    }

    pub fn mul(&self, other: &Self, direction: Dir) -> Self {
        Self {
            mantissa: self.mantissa.mul(&other.mantissa),
            exponent: self.exponent + other.exponent,
        }
        .cut(direction)
    }

    pub fn div(&self, other: &Self, direction: Dir) -> Self {
        assert!(!other.is_zero(), "division by zero");
        if self.is_zero() {
            return Self::zero();
        }
        let shift = REAL_BITS + 2 + other.mantissa.bits();
        let (mut quotient, remainder) =
            self.mantissa.shl(shift).divmod(&other.mantissa);
        if direction == Dir::Up && !remainder.is_zero() {
            quotient = quotient.add(&Big::from_u64(1));
        }
        Self {
            mantissa: quotient,
            exponent: self.exponent - shift as i64 - other.exponent,
        }
        .cut(direction)
    }

    pub fn sqrt(&self, direction: Dir) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let top = self.top();
        let mut root_top = top / 2;
        if 2 * root_top < top {
            root_top += 1;
        }
        let unit = root_top - REAL_BITS as i64;
        let shift = self.exponent - 2 * unit;
        let whole = if shift >= 0 {
            self.mantissa.shl(shift as usize)
        } else {
            self.mantissa.shr((-shift) as usize)
        };
        let fraction = shift < 0 && self.mantissa.any_below((-shift) as usize);
        let mut root = whole.isqrt();
        if direction == Dir::Up && (fraction || root.mul(&root).cmp(&whole) != Ordering::Equal)
        {
            root = root.add(&Big::from_u64(1));
        }
        Self {
            mantissa: root,
            exponent: unit,
        }
    }

    pub fn pow(mut base: Self, mut exponent: u64, direction: Dir) -> Self {
        let mut answer = Self::one();
        while exponent != 0 {
            if exponent & 1 == 1 {
                answer = answer.mul(&base, direction);
            }
            exponent >>= 1;
            if exponent != 0 {
                base = base.mul(&base, direction);
            }
        }
        answer
    }

    pub fn floor(&self) -> Big {
        if self.exponent >= 0 {
            self.mantissa.shl(self.exponent as usize)
        } else {
            self.mantissa.shr((-self.exponent) as usize)
        }
    }

    pub fn ceil(&self) -> Big {
        if self.exponent >= 0 {
            self.mantissa.shl(self.exponent as usize)
        } else {
            let shifted = self.mantissa.shr((-self.exponent) as usize);
            if self.mantissa.any_below((-self.exponent) as usize) {
                shifted.add(&Big::from_u64(1))
            } else {
                shifted
            }
        }
    }

    pub fn log2(&self) -> f64 {
        if self.is_zero() {
            return f64::NEG_INFINITY;
        }
        let bits = self.mantissa.bits();
        let top = if bits > 64 {
            self.mantissa.shr(bits - 64)
        } else {
            self.mantissa.shl(64 - bits)
        };
        self.top() as f64 + (top.to_u64() as f64 / 18_446_744_073_709_551_616.0).log2()
    }
}

impl PartialOrd for Real {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bound {
    pub lo: Real,
    pub hi: Real,
    pub bad: bool,
}

impl Bound {
    pub fn from_u64(value: u64) -> Self {
        Self {
            lo: Real::from_u64(value),
            hi: Real::from_u64(value),
            bad: false,
        }
    }

    pub fn from_big(value: Big) -> Self {
        Self {
            lo: Real::from_big(value.clone()).cut(Dir::Down),
            hi: Real::from_big(value).cut(Dir::Up),
            bad: false,
        }
    }

    pub fn from_real(value: Real) -> Self {
        Self {
            lo: value.clone(),
            hi: value,
            bad: false,
        }
    }

    pub fn pair(lo: Real, hi: Real) -> Self {
        let bad = hi.cmp(&lo) == Ordering::Less;
        Self { lo, hi, bad }
    }

    pub fn failed() -> Self {
        Self {
            lo: Real::one(),
            hi: Real::one(),
            bad: true,
        }
    }

    pub fn ratio(numerator: &Big, denominator: &Big) -> Self {
        Self::from_big(numerator.clone()).div(&Self::from_big(denominator.clone()))
    }

    pub fn inherit(mut self, other: &Self) -> Self {
        self.bad |= other.bad;
        self
    }

    pub fn add(&self, other: &Self) -> Self {
        Self::pair(
            Real::add(&self.lo, &other.lo, Dir::Down),
            Real::add(&self.hi, &other.hi, Dir::Up),
        )
        .inherit(self)
        .inherit(other)
    }

    pub fn sub(&self, other: &Self) -> Self {
        if self.bad
            || other.bad
            || self.lo.cmp(&other.hi) != Ordering::Greater
        {
            return Self::failed();
        }
        Self::pair(
            Real::sub(&self.lo, &other.hi, Dir::Down),
            Real::sub(&self.hi, &other.lo, Dir::Up),
        )
    }

    pub fn mul(&self, other: &Self) -> Self {
        Self::pair(
            Real::mul(&self.lo, &other.lo, Dir::Down),
            Real::mul(&self.hi, &other.hi, Dir::Up),
        )
        .inherit(self)
        .inherit(other)
    }

    pub fn div(&self, other: &Self) -> Self {
        if self.bad || other.bad || other.lo.is_zero() {
            return Self::failed();
        }
        Self::pair(
            Real::div(&self.lo, &other.hi, Dir::Down),
            Real::div(&self.hi, &other.lo, Dir::Up),
        )
    }

    pub fn sqrt(&self) -> Self {
        Self::pair(
            Real::sqrt(&self.lo, Dir::Down),
            Real::sqrt(&self.hi, Dir::Up),
        )
        .inherit(self)
    }

    pub fn pow(&self, exponent: u64) -> Self {
        Self::pair(
            Real::pow(self.lo.clone(), exponent, Dir::Down),
            Real::pow(self.hi.clone(), exponent, Dir::Up),
        )
        .inherit(self)
    }

    pub fn error(&self) -> Real {
        if self.bad {
            Real::one()
        } else {
            self.hi.clone()
        }
    }
}

impl PartialOrd for Bound {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.bad || other.bad {
            return None;
        }
        Some(self.hi.cmp(&other.lo))
    }
}

pub fn surely_le(left: &Bound, right: &Bound) -> bool {
    !left.bad && !right.bad && left.hi.cmp(&right.lo) != Ordering::Greater
}

pub fn surely_lt(left: &Bound, right: &Bound) -> bool {
    !left.bad && !right.bad && left.hi.cmp(&right.lo) == Ordering::Less
}

pub fn bits_of(error: &Real) -> f64 {
    if error.is_zero() {
        f64::INFINITY
    } else {
        -error.log2()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn big_division_and_square_root_are_exact() {
        for left in [0u64, 1, 2, 17, 1 << 32, (1 << 32) + 3, u64::MAX] {
            for right in [1u64, 3, 17, 1 << 33, u64::MAX] {
                let (quotient, remainder) =
                    Big::from_u64(left).divmod(&Big::from_u64(right));
                assert_eq!(
                    quotient.mul(&Big::from_u64(right)).add(&remainder),
                    Big::from_u64(left)
                );
                assert!(remainder.cmp(&Big::from_u64(right)) == Ordering::Less);
            }
        }
        for value in [0u64, 1, 2, 3, 4, 15, 16, 17, 1 << 40, u64::MAX] {
            let root = Big::from_u64(value).isqrt();
            assert!(root.mul(&root).cmp(&Big::from_u64(value)) != Ordering::Greater);
            let next = root.add(&Big::from_u64(1));
            assert!(next.mul(&next).cmp(&Big::from_u64(value)) == Ordering::Greater);
        }
    }

    #[test]
    fn real_intervals_contain_the_true_value() {
        let one = Bound::from_u64(1);
        let three = Bound::from_u64(3);
        let third = one.div(&three);
        assert!(third.lo.log2() <= (1.0f64 / 3.0).log2() + 1e-12);
        assert!(third.hi.log2() >= (1.0f64 / 3.0).log2() - 1e-12);
        let two = Bound::from_u64(2);
        let root = two.sqrt();
        let expected = 2.0f64.sqrt().log2();
        assert!(root.lo.log2() <= expected + 1e-12);
        assert!(root.hi.log2() >= expected - 1e-12);
        let modulus = Big::two_to(64)
            .sub(&Big::two_to(32))
            .add(&Big::from_u64(1));
        let card = Bound::from_big(Big::pow(modulus, 3));
        let log2 = card.lo.log2();
        assert!(log2 > 191.99 && log2 < 192.0, "{}", log2);
        assert!(surely_le(&one, &card));
    }

    #[test]
    fn bound_power_rounds_up_on_the_safe_side() {
        let agree = Bound::from_u64(3).div(&Bound::from_u64(4));
        let error = agree.pow(254);
        assert!(bits_of(&error.hi) <= 254.0 * 0.4150375 + 1e-9);
        assert!(!error.bad);
    }
    #[test]
    fn limb_boundaries_and_directed_cuts_hold() {
        let a = Big::from_u64(u32::MAX as u64);
        let b = Big::from_u64(u32::MAX as u64);
        let product = a.mul(&b);
        let (quotient, remainder) = product.divmod(&b);
        assert_eq!(quotient, a);
        assert!(remainder.is_zero());
        let large = Big::two_to(130).add(&Big::from_u64(1));
        let (q, r) = large.divmod(&Big::from_u64(3));
        assert_eq!(q.mul(&Big::from_u64(3)).add(&r), large);
        let root = Big::two_to(192).add(&Big::from_u64(1)).isqrt();
        assert_eq!(root, Big::two_to(96));
        let root_next = root.add(&Big::from_u64(1));
        assert!(root_next.mul(&root_next).cmp(&Big::two_to(192)) == Ordering::Greater);

        let exact = Real::from_big(Big::two_to(192).add(&Big::from_u64(1)));
        let down = exact.cut(Dir::Down);
        let up = exact.cut(Dir::Up);
        assert!(down.cmp(&exact) != Ordering::Greater);
        assert!(up.cmp(&exact) != Ordering::Less);
        assert!(down.cmp(&up) != Ordering::Greater);

        let tiny = Real::two_to(-1000);
        let tiny_root = tiny.sqrt(Dir::Up);
        assert!(tiny_root.cmp(&Real::zero()) == Ordering::Greater);
        assert!(tiny_root.cmp(&Real::two_to(-499)) != Ordering::Greater);

        let bad = Bound::failed();
        assert!(bad.bad);
        assert!(bad.add(&Bound::from_u64(1)).bad);
        assert!(bad.mul(&Bound::from_u64(1)).bad);
        assert!(bad.div(&Bound::from_u64(1)).bad);
        assert!(bad.sqrt().bad);
        assert!(bad.pow(3).bad);
    }

}
