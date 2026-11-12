//! The protocol base field and its cubic extension.
//!
//! The base field is Goldilocks, `p = 2^64 - 2^32 + 1`. The extension is
//! `Fp[X] / (X^3 - 2)` in the power basis `(1, X, X^2)`: an element is
//! `c0 + c1 X + c2 X^2`, and its canonical encoding is the three base-field
//! coefficients in that order, eight little-endian bytes each.

use serde::{Deserialize, Serialize};
use zkie_core::common::field::{Field, PrimeCharacteristicRing, PrimeField64};
use zkie_core::error::{Error, Result};

pub use zkie_core::common::field::Goldilocks as Fp;

pub const MODULUS: u64 = 0xFFFF_FFFF_0000_0001;
pub const EXTENSION_NONRESIDUE: u64 = 2;
pub const FP_BYTES: usize = 8;
pub const FP3_BYTES: usize = 24;
/// Degree of the extension field.
pub const E_DEGREE: usize = 3;

pub fn fp_from_u64(value: u64) -> Fp {
    Fp::from_u64(value)
}

/// Number of candidate words one base-field rejection sample may consume.
/// The rejection probability is `1 - p/2^64 < 2^-32`, so exhaustion is below
/// `2^-2048` per sample.
pub const SAMPLE_TRIES: u32 = 64;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

static SAMPLE_ACCEPTS: AtomicU64 = AtomicU64::new(0);
static SAMPLE_REJECTS: AtomicU64 = AtomicU64::new(0);

fn sample_counts_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("ZKIE_SAMPLE_COUNTS").is_some())
}

/// The accept/reject counts of the mask sampler in this process. The counters
/// are enabled by `ZKIE_SAMPLE_COUNTS`; with the variable unset they are not
/// incremented.
pub fn sampling_counts() -> (u64, u64) {
    (
        SAMPLE_ACCEPTS.load(Ordering::Relaxed),
        SAMPLE_REJECTS.load(Ordering::Relaxed),
    )
}

/// Uniform base-field sample by rejection. `word(attempt)` must return a fresh
/// 64-bit candidate; a candidate at or above the modulus is rejected and the
/// next attempt is drawn. Exhaustion returns `EntropyUnavailable`; there is no
/// modulo fallback.
pub fn uniform_fp<F>(mut word: F) -> Result<Fp>
where
    F: FnMut(u32) -> Result<u64>,
{
    let count = sample_counts_enabled();
    for attempt in 0..SAMPLE_TRIES {
        let candidate = word(attempt)?;
        if candidate < MODULUS {
            if count {
                SAMPLE_ACCEPTS.fetch_add(1, Ordering::Relaxed);
            }
            return Ok(Fp::from_u64(candidate));
        }
        if count {
            SAMPLE_REJECTS.fetch_add(1, Ordering::Relaxed);
        }
    }
    Err(Error::EntropyUnavailable)
}

pub fn fp_to_u64(value: Fp) -> u64 {
    value.as_canonical_u64()
}

pub fn fp_inverse(value: Fp) -> Result<Fp> {
    value.try_inverse().ok_or(Error::DivisionByZero)
}

pub fn fp_pow(value: Fp, mut exponent: u64) -> Fp {
    let mut base = value;
    let mut result = Fp::ONE;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = result * base;
        }
        base = base * base;
        exponent >>= 1;
    }
    result
}

/// `c0 + c1 X + c2 X^2` over the base field.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Fp3 {
    pub c: [Fp; 3],
}

impl Fp3 {
    pub const ZERO: Self = Self {
        c: [Fp::ZERO, Fp::ZERO, Fp::ZERO],
    };

    pub const ONE: Self = Self {
        c: [Fp::ONE, Fp::ZERO, Fp::ZERO],
    };

    pub const fn new(c0: Fp, c1: Fp, c2: Fp) -> Self {
        Self { c: [c0, c1, c2] }
    }

    pub const fn from_base(value: Fp) -> Self {
        Self {
            c: [value, Fp::ZERO, Fp::ZERO],
        }
    }

    pub fn from_u64(value: u64) -> Self {
        Self::from_base(Fp::from_u64(value))
    }

    pub fn is_zero(&self) -> bool {
        self.c.iter().all(|coefficient| coefficient.as_canonical_u64() == 0)
    }

    pub fn add(&self, other: &Self) -> Self {
        Self::new(
            self.c[0] + other.c[0],
            self.c[1] + other.c[1],
            self.c[2] + other.c[2],
        )
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self::new(
            self.c[0] - other.c[0],
            self.c[1] - other.c[1],
            self.c[2] - other.c[2],
        )
    }

    pub fn neg(&self) -> Self {
        Self::new(-self.c[0], -self.c[1], -self.c[2])
    }

    pub fn mul(&self, other: &Self) -> Self {
        let two = Fp::TWO;
        let a0 = self.c[0];
        let a1 = self.c[1];
        let a2 = self.c[2];
        let b0 = other.c[0];
        let b1 = other.c[1];
        let b2 = other.c[2];
        Self::new(
            a0 * b0 + two * (a1 * b2 + a2 * b1),
            a0 * b1 + a1 * b0 + two * (a2 * b2),
            a0 * b2 + a1 * b1 + a2 * b0,
        )
    }

    pub fn inverse(&self) -> Result<Self> {
        let two = Fp::TWO;
        let a = self.c[0];
        let b = self.c[1];
        let c = self.c[2];
        let aa = a * a - two * (b * c);
        let bb = two * (c * c) - a * b;
        let cc = b * b - a * c;
        let norm = a * aa + two * (b * cc + c * bb);
        let inverse_norm = fp_inverse(norm)?;
        Ok(Self::new(aa * inverse_norm, bb * inverse_norm, cc * inverse_norm))
    }

    pub fn pow_u64(&self, mut exponent: u64) -> Self {
        let mut base = *self;
        let mut result = Self::ONE;
        while exponent != 0 {
            if exponent & 1 == 1 {
                result = result.mul(&base);
            }
            base = base.mul(&base);
            exponent >>= 1;
        }
        result
    }

    /// Canonical encoding: `c0`, `c1`, `c2`, each eight little-endian bytes.
    pub fn encode(&self) -> [u8; FP3_BYTES] {
        let mut out = [0u8; FP3_BYTES];
        for (index, coefficient) in self.c.iter().enumerate() {
            out[index * FP_BYTES..(index + 1) * FP_BYTES]
                .copy_from_slice(&fp_to_u64(*coefficient).to_le_bytes());
        }
        out
    }

    /// Decode a canonical encoding. A coefficient at or above the modulus is
    /// rejected; it is never silently reduced.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != FP3_BYTES {
            return Err(Error::InvalidEncoding);
        }
        let mut coefficients = [Fp::ZERO; 3];
        for index in 0..3 {
            let word = u64::from_le_bytes(
                bytes[index * FP_BYTES..(index + 1) * FP_BYTES]
                    .try_into()
                    .map_err(|_| Error::InvalidEncoding)?,
            );
            if word >= MODULUS {
                return Err(Error::InvalidEncoding);
            }
            coefficients[index] = Fp::from_u64(word);
        }
        Ok(Self { c: coefficients })
    }
}

#[cfg(test)]
mod uniform_tests {
    use super::*;

    #[test]
    fn boundary_words_are_accepted_or_rejected_without_reduction() {
        let last = MODULUS - 1;
        assert_eq!(uniform_fp(|_| Ok(last)).unwrap(), Fp::from_u64(last));
        let first_retry = uniform_fp(|attempt| {
            if attempt == 0 {
                Ok(MODULUS)
            } else {
                Ok(7)
            }
        })
        .unwrap();
        assert_eq!(first_retry, Fp::from_u64(7));
        let top = uniform_fp(|attempt| {
            if attempt == 0 {
                Ok(u64::MAX)
            } else {
                Ok(9)
            }
        })
        .unwrap();
        assert_eq!(top, Fp::from_u64(9));
    }

    #[test]
    fn exhaustion_returns_an_error_after_the_fixed_number_of_tries() {
        let mut calls = 0u32;
        let result = uniform_fp(|_| {
            calls += 1;
            Ok(u64::MAX)
        });
        assert!(matches!(result, Err(Error::EntropyUnavailable)));
        assert_eq!(calls, SAMPLE_TRIES);
    }
}

