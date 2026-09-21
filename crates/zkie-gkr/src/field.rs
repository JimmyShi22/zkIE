//! Goldilocks field (`p = 2^64 - 2^32 + 1`) plus a tiny xorshift PRNG.
//!
//! Goldilocks is the "sufficient" field for large-model inference: its 64-bit
//! width comfortably holds int16 fixed-point accumulation (products are 32-bit,
//! and a length-K dot product adds `log2(K)` bits), while `p - 1 = 2^32 *
//! (2^32 - 1)` provides a large 2-adic subgroup for FRI. It stays native 64-bit
//! arithmetic with no elliptic curve, so it remains far cheaper than BN254.

use std::ops::{Add, Mul, Neg, Sub};

pub const P: u64 = 0xffff_ffff_0000_0001; // 2^64 - 2^32 + 1

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct F64(pub u64);

impl F64 {
    pub const ZERO: F64 = F64(0);
    pub const ONE: F64 = F64(1);
    pub const TWO: F64 = F64(2);

    #[inline]
    pub fn new(x: u64) -> F64 {
        F64(reduce64(x))
    }

    #[inline]
    pub fn from_u64(x: u64) -> F64 {
        F64(reduce64(x))
    }

    #[inline]
    pub fn val(self) -> u64 {
        self.0
    }

    #[inline]
    pub fn add(self, rhs: F64) -> F64 {
        let s = self.0 as u128 + rhs.0 as u128;
        F64(if s >= P as u128 { (s - P as u128) as u64 } else { s as u64 })
    }

    #[inline]
    pub fn sub(self, rhs: F64) -> F64 {
        let d = self.0 as i128 - rhs.0 as i128;
        F64(if d < 0 { (d + P as i128) as u64 } else { d as u64 })
    }

    #[inline]
    pub fn mul(self, rhs: F64) -> F64 {
        // Correct but not optimal; swap in the standard Goldilocks reduction
        // before benchmarking on GPU.
        F64(((self.0 as u128 * rhs.0 as u128) % P as u128) as u64)
    }

    #[inline]
    pub fn neg(self) -> F64 {
        if self.0 == 0 { F64(0) } else { F64(P - self.0) }
    }

    pub fn inv(self) -> F64 {
        self.pow(P - 2)
    }

    pub fn pow(self, mut e: u64) -> F64 {
        let mut base = self;
        let mut acc = F64::ONE;
        while e > 0 {
            if e & 1 == 1 {
                acc = acc.mul(base);
            }
            base = base.mul(base);
            e >>= 1;
        }
        acc
    }
}

#[inline]
fn reduce64(x: u64) -> u64 {
    if x >= P { x - P } else { x }
}

impl Add for F64 {
    type Output = F64;
    #[inline]
    fn add(self, rhs: F64) -> F64 {
        F64::add(self, rhs)
    }
}

impl Sub for F64 {
    type Output = F64;
    #[inline]
    fn sub(self, rhs: F64) -> F64 {
        F64::sub(self, rhs)
    }
}

impl Mul for F64 {
    type Output = F64;
    #[inline]
    fn mul(self, rhs: F64) -> F64 {
        F64::mul(self, rhs)
    }
}

impl Neg for F64 {
    type Output = F64;
    #[inline]
    fn neg(self) -> F64 {
        F64::neg(self)
    }
}

impl From<u32> for F64 {
    fn from(x: u32) -> F64 {
        F64(x as u64)
    }
}

impl From<u64> for F64 {
    fn from(x: u64) -> F64 {
        F64::from_u64(x)
    }
}

/// A primitive `2^log_n`-th root of unity in Goldilocks' multiplicative group.
///
/// `p - 1 = 2^32 * (2^32 - 1)`, so we find a generator of the full group and
/// raise it to `2^32 - 1` to get a primitive `2^32`-th root, then take the
/// appropriate power for the requested `log_n`.
pub fn two_adic_root(log_n: usize) -> F64 {
    assert!(log_n <= 32, "Goldilocks only supports 2-adic roots up to 2^32");
    let mut g = F64::new(2);
    loop {
        let omega = g.pow((P - 1) >> 32);
        if omega.pow(1u64 << 31).val() == P - 1 {
            return omega.pow(1u64 << (32 - log_n));
        }
        g = F64::new(g.val() + 1);
    }
}

pub struct XorShift64(u64);

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        XorShift64(seed.wrapping_add(0x9e37_79b9_7f4a_7c15))
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    pub fn field(&mut self) -> F64 {
        F64::from_u64(self.next_u64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_axioms() {
        let mut rng = XorShift64::new(1);
        for _ in 0..1000 {
            let a = rng.field();
            let b = rng.field();
            let c = rng.field();
            assert_eq!(a + b, b + a);
            assert_eq!(a * b, b * a);
            assert_eq!((a + b) + c, a + (b + c));
            assert_eq!((a * b) * c, a * (b * c));
            assert_eq!(a * (b + c), a * b + a * c);
            assert_eq!(a + F64::ZERO, a);
            assert_eq!(a * F64::ONE, a);
            assert_eq!(a + (-a), F64::ZERO);
            if a != F64::ZERO {
                assert_eq!(a * a.inv(), F64::ONE);
            }
        }
    }

    #[test]
    fn inverse_of_two() {
        assert_eq!(F64::TWO * F64::new((P + 1) / 2), F64::ONE);
    }

    #[test]
    fn two_adic_root_is_primitive() {
        let omega = two_adic_root(32);
        assert_eq!(omega.pow(1 << 32), F64::ONE);
        assert_eq!(omega.pow(1 << 31).val(), P - 1);
    }
}
