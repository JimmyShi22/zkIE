//! Goldilocks field re-export plus a tiny deterministic xorshift PRNG.
//!
//! The field arithmetic is Plonky3's audited `Goldilocks`; this module only
//! re-exports it (together with the trait plumbing the rest of the crate uses)
//! and keeps a deterministic test PRNG. Goldilocks is the "sufficient" field for
//! large-model inference: its 64-bit width holds int16 fixed-point accumulation
//! (products are 32-bit, a length-K dot product adds `log2(K)` bits), and
//! `p - 1 = 2^32 * (2^32 - 1)` gives a large 2-adic subgroup for field arithmetic and FFT-style transforms.

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
/// A deterministic challenge source (Fiat-Shamir). Both the test PRNG
/// `XorShift64` and the SHA-256 `Transcript` implement this, so prove/verify
/// can be generic over the challenge source.
pub trait ChallengeSource {
    fn field(&mut self) -> Goldilocks;
    fn next_u64(&mut self) -> u64;
    fn field_vec(&mut self, n: usize) -> Vec<Goldilocks> {
        (0..n).map(|_| self.field()).collect()
    }
}

impl ChallengeSource for XorShift64 {
    fn field(&mut self) -> Goldilocks {
        XorShift64::field(self)
    }
    fn next_u64(&mut self) -> u64 {
        XorShift64::next_u64(self)
    }
}

// ---------------------------------------------------------------------------
// Protocol-facing field descriptor and bounded embedding interface.

/// Description of a base field and its extension. Values not yet verified
/// against the fixed specification are left as `None` rather than guessed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldDescriptor {
    pub name: &'static str,
    pub modulus: u64,
    pub extension_degree: u32,
    pub irreducible_polynomial: Option<&'static [u64]>,
    pub basis: Option<&'static str>,
    pub coefficient_order: Option<&'static str>,
}

pub static GOLDILOCKS: FieldDescriptor = FieldDescriptor {
    name: "goldilocks",
    modulus: P,
    extension_degree: 1,
    irreducible_polynomial: None,
    basis: None,
    coefficient_order: None,
};

pub static GOLDILOCKS_EXT3: FieldDescriptor = FieldDescriptor {
    name: "goldilocks-ext3",
    modulus: P,
    extension_degree: 3,
    irreducible_polynomial: None,
    basis: None,
    coefficient_order: None,
};

/// A field usable by the protocol layer. Arithmetic comes from the re-exported
/// field trait; this trait adds the descriptor and the checked integer
/// embedding boundary.
pub trait ProtocolField:
    Field + crate::common::codec::Canonical + Copy + Eq + Send + Sync + 'static
{
    fn descriptor() -> &'static FieldDescriptor;

    /// Embed a signed integer only when it is inside `bound` and `bound` is
    /// inside the safe symmetric range of the field.
    fn embed_bounded_integer(value: i128, bound: u128) -> crate::error::Result<Self>;
}

impl ProtocolField for Goldilocks {
    fn descriptor() -> &'static FieldDescriptor {
        &GOLDILOCKS
    }

    fn embed_bounded_integer(value: i128, bound: u128) -> crate::error::Result<Self> {
        let safe_bound = (P as u128) / 2;
        if bound > safe_bound {
            return Err(crate::error::Error::BoundViolation);
        }
        let magnitude = value.unsigned_abs();
        if magnitude > bound {
            return Err(crate::error::Error::BoundViolation);
        }
        Ok(if value >= 0 {
            Goldilocks::from_u64(value as u64)
        } else {
            Goldilocks::from_u64(P - magnitude as u64)
        })
    }
}

impl crate::common::codec::Canonical for Goldilocks {
    fn encode(&self, out: &mut Vec<u8>) -> crate::error::Result<()> {
        out.extend_from_slice(&self.as_canonical_u64().to_le_bytes());
        Ok(())
    }

    fn decode(
        input: &mut &[u8],
        limits: &crate::common::codec::DecodeLimits,
    ) -> crate::error::Result<Self> {
        limits.check_bytes(8)?;
        if input.len() < 8 {
            return Err(crate::error::Error::InvalidEncoding);
        }
        let bytes: [u8; 8] = input[..8]
            .try_into()
            .map_err(|_| crate::error::Error::InvalidEncoding)?;
        *input = &input[8..];
        let value = u64::from_le_bytes(bytes);
        if value >= P {
            return Err(crate::error::Error::InvalidEncoding);
        }
        Ok(Goldilocks::from_u64(value))
    }
}
