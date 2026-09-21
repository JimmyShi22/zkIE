//! Mersenne-31 field (`p = 2^31 - 1`) plus a tiny xorshift PRNG.
//!
//! This is deliberately the "small field" end of the design space: 32-bit
//! native arithmetic, no elliptic-curve MSM. In a real deployment you would
//! use BabyBear / a larger field (or an extension field) for soundness, but
//! that is orthogonal to the arithmetization comparison this crate measures.

use std::ops::{Add, Mul, Neg, Sub};

pub const P: u32 = 0x7fff_ffff;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct F31(pub u32);

impl F31 {
    pub const ZERO: F31 = F31(0);
    pub const ONE: F31 = F31(1);
    pub const TWO: F31 = F31(2);

    #[inline]
    pub fn new(x: u32) -> F31 {
        F31(x % P)
    }

    #[inline]
    pub fn from_u64(x: u64) -> F31 {
        F31(reduce(x))
    }

    #[inline]
    pub fn val(self) -> u32 {
        self.0
    }

    #[inline]
    pub fn add(self, rhs: F31) -> F31 {
        F31(reduce(self.0 as u64 + rhs.0 as u64))
    }

    #[inline]
    pub fn sub(self, rhs: F31) -> F31 {
        let d = self.0 as i64 - rhs.0 as i64;
        F31(if d < 0 { (d + P as i64) as u32 } else { d as u32 })
    }

    #[inline]
    pub fn mul(self, rhs: F31) -> F31 {
        F31(reduce(self.0 as u64 * rhs.0 as u64))
    }

    #[inline]
    pub fn neg(self) -> F31 {
        if self.0 == 0 { F31(0) } else { F31(P - self.0) }
    }

    pub fn inv(self) -> F31 {
        self.pow(P - 2)
    }

    pub fn pow(self, mut e: u32) -> F31 {
        let mut base = self;
        let mut acc = F31::ONE;
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
fn reduce(mut x: u64) -> u32 {
    x = (x >> 31) + (x & 0x7fff_ffff);
    x = (x >> 31) + (x & 0x7fff_ffff);
    let mut x = x as u32;
    if x >= P {
        x -= P;
    }
    x
}

impl Add for F31 {
    type Output = F31;
    #[inline]
    fn add(self, rhs: F31) -> F31 {
        F31::add(self, rhs)
    }
}

impl Sub for F31 {
    type Output = F31;
    #[inline]
    fn sub(self, rhs: F31) -> F31 {
        F31::sub(self, rhs)
    }
}

impl Mul for F31 {
    type Output = F31;
    #[inline]
    fn mul(self, rhs: F31) -> F31 {
        F31::mul(self, rhs)
    }
}

impl Neg for F31 {
    type Output = F31;
    #[inline]
    fn neg(self) -> F31 {
        F31::neg(self)
    }
}

impl From<u32> for F31 {
    fn from(x: u32) -> F31 {
        F31::new(x)
    }
}

impl From<u64> for F31 {
    fn from(x: u64) -> F31 {
        F31::from_u64(x)
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

    pub fn field(&mut self) -> F31 {
        F31::from_u64(self.next_u64() & 0x7fff_ffff)
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
            assert_eq!(a + F31::ZERO, a);
            assert_eq!(a * F31::ONE, a);
            assert_eq!(a + (-a), F31::ZERO);
            if a != F31::ZERO {
                assert_eq!(a * a.inv(), F31::ONE);
            }
        }
    }

    #[test]
    fn inverse_of_two() {
        assert_eq!(F31::TWO * F31::new(1 << 30), F31::ONE);
    }
}
