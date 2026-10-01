//! Goldilocks field re-export plus a tiny deterministic xorshift PRNG.
//!
//! The field arithmetic is Plonky3's audited `Goldilocks`; this module only
//! re-exports it (together with the trait plumbing the rest of the crate uses)
//! and keeps a deterministic test PRNG. Goldilocks is the "sufficient" field for
//! large-model inference: its 64-bit width holds int16 fixed-point accumulation
//! (products are 32-bit, a length-K dot product adds `log2(K)` bits), and
//! `p - 1 = 2^32 * (2^32 - 1)` gives a large 2-adic subgroup for FRI/WHIR.

pub use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
pub use p3_goldilocks::Goldilocks;

/// Field characteristic `p = 2^64 - 2^32 + 1`.
pub const P: u64 = Goldilocks::ORDER_U64;

/// A deterministic xorshift64 PRNG producing canonical `Goldilocks` elements.
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

    pub fn field(&mut self) -> Goldilocks {
        Goldilocks::from_u64(self.next_u64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_of_two() {
        assert_eq!(
            Goldilocks::TWO * Goldilocks::from_u64((P + 1) / 2),
            Goldilocks::ONE
        );
    }

    #[test]
    fn xorshift_produces_canonical_elements() {
        let mut rng = XorShift64::new(1);
        for _ in 0..1000 {
            assert!(rng.field().as_canonical_u64() < P);
        }
    }
}
