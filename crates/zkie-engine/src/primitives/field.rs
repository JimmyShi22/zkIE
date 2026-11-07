//! Adapter from the protocol field implementation to the frozen core interface.
//!
//! The core [`ProtocolField`] trait is implemented here; no core type depends
//! on the concrete extension representation.

use core::fmt;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{Field, Packable, PrimeCharacteristicRing, RawDataSerializable};
use zkie_core::common::codec::{Canonical, DecodeLimits};
use zkie_core::common::field::{FieldDescriptor, ProtocolField, GOLDILOCKS_EXT3, P};
use zkie_core::error::{Error, Result};

use crate::pcs::request::field::{fp_to_u64, Fp, Fp3, FP3_BYTES};

impl Add for Fp3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Fp3::add(&self, &rhs)
    }
}

impl AddAssign for Fp3 {
    fn add_assign(&mut self, rhs: Self) {
        *self = Fp3::add(self, &rhs);
    }
}

impl Sub for Fp3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Fp3::sub(&self, &rhs)
    }
}

impl SubAssign for Fp3 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = Fp3::sub(self, &rhs);
    }
}

impl Neg for Fp3 {
    type Output = Self;
    fn neg(self) -> Self {
        Fp3::neg(&self)
    }
}

impl Mul for Fp3 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Fp3::mul(&self, &rhs)
    }
}

impl MulAssign for Fp3 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = Fp3::mul(self, &rhs);
    }
}

impl Div for Fp3 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Fp3::mul(&self, &Field::inverse(&rhs))
    }
}

impl DivAssign for Fp3 {
    fn div_assign(&mut self, rhs: Self) {
        *self = Fp3::mul(self, &Field::inverse(&rhs));
    }
}

impl Sum for Fp3 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Fp3::ZERO, |accumulator, value| accumulator + value)
    }
}

impl Product for Fp3 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Fp3::ONE, |accumulator, value| accumulator * value)
    }
}

impl PrimeCharacteristicRing for Fp3 {
    type PrimeSubfield = Fp;

    const ZERO: Self = Fp3::ZERO;
    const ONE: Self = Fp3::ONE;
    const TWO: Self = Fp3::new(Fp::TWO, Fp::ZERO, Fp::ZERO);
    const NEG_ONE: Self = Fp3::new(Fp::NEG_ONE, Fp::ZERO, Fp::ZERO);

    fn from_prime_subfield(value: Self::PrimeSubfield) -> Self {
        Fp3::from_base(value)
    }
}

impl Field for Fp3 {
    type Packing = Self;

    // TODO: replace with a verified generator once one is recorded; the value is
    // not protocol-visible and no proof byte depends on it.
    const GENERATOR: Self = Fp3::new(Fp::ZERO, Fp::ONE, Fp::ZERO);

    fn try_inverse(&self) -> Option<Self> {
        Fp3::inverse(self).ok()
    }

    fn order() -> num_bigint::BigUint {
        num_bigint::BigUint::from(P).pow(3)
    }
}

impl RawDataSerializable for Fp3 {
    const NUM_BYTES: usize = FP3_BYTES;

    fn into_bytes(self) -> impl IntoIterator<Item = u8> {
        self.encode()
    }
}

impl Packable for Fp3 {}

impl fmt::Display for Fp3 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "({} + {} X + {} X^2)",
            fp_to_u64(self.c[0]),
            fp_to_u64(self.c[1]),
            fp_to_u64(self.c[2])
        )
    }
}

impl Canonical for Fp3 {
    fn encode(&self, out: &mut Vec<u8>) -> Result<()> {
        out.extend_from_slice(&Fp3::encode(self));
        Ok(())
    }

    fn decode(input: &mut &[u8], limits: &DecodeLimits) -> Result<Self> {
        limits.check_bytes(FP3_BYTES)?;
        if input.len() < FP3_BYTES {
            return Err(Error::InvalidEncoding);
        }
        let value = Fp3::decode(&input[..FP3_BYTES])?;
        *input = &input[FP3_BYTES..];
        Ok(value)
    }
}

impl ProtocolField for Fp3 {
    fn descriptor() -> &'static FieldDescriptor {
        &GOLDILOCKS_EXT3
    }

    fn embed_bounded_integer(value: i128, bound: u128) -> Result<Self> {
        let safe_bound = (P as u128) / 2;
        if bound > safe_bound {
            return Err(Error::BoundViolation);
        }
        let magnitude = value.unsigned_abs();
        if magnitude > bound {
            return Err(Error::BoundViolation);
        }
        let base = if value >= 0 {
            Fp::from_u64(value as u64)
        } else {
            Fp::from_u64(P - magnitude as u64)
        };
        Ok(Fp3::from_base(base))
    }
}
