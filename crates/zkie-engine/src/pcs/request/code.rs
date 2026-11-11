//! Reed-Solomon codes on cosets of the Goldilocks two-power subgroup.
//!
//! A code of length `2^log_n` evaluates a polynomial at positions
//! `0, 1, ..., n - 1`. Tail coefficients occupy degrees `0 .. r - 1` and
//! message coefficients `r .. r + ell - 1`. A symbol at a position is the
//! polynomial's value at that position's point.

use serde::{Deserialize, Serialize};
use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::error::{Error, Result};

use crate::pcs::request::field::{fp_from_u64, fp_pow, fp_to_u64, Fp, Fp3};

pub const GENERATOR: u64 = 7;
pub const ROOT_32: u64 = 0xE2AF_7063_5744_FD26;
pub const SUBGROUP_LOG: u32 = 32;
pub const MOST_LOG_N: u32 = 60;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeParams {
    pub log_n: u32,
    pub ell: u64,
    pub r: u64,
}

impl CodeParams {
    pub fn n(&self) -> u64 {
        1u64 << self.log_n
    }

    pub fn dim(&self) -> u64 {
        self.ell + self.r
    }

    pub fn log_dim(&self) -> u32 {
        let dimension = self.dim();
        let mut bits = 0u32;
        while (1u64 << bits) < dimension {
            bits += 1;
        }
        bits
    }

    /// The number of factors in the message's multilinear extension.
    pub fn log_ell(&self) -> u32 {
        let mut bits = 0u32;
        while (1u64 << bits) < self.ell {
            bits += 1;
        }
        bits
    }

    pub fn message_at(&self) -> u64 {
        self.r
    }

    pub fn tail_at(&self) -> u64 {
        0
    }

    pub fn hiding_positions(&self) -> u64 {
        self.r
    }
}

fn subgroup_log(log_n: u32) -> u32 {
    core::cmp::min(log_n, SUBGROUP_LOG)
}

/// Build a code plan after checking its shape.
pub fn build_code_plan(log_n: u32, ell: u64, r: u64) -> Result<CodeParams> {
    if log_n < 1 || log_n > MOST_LOG_N || ell < 1 {
        return Err(Error::InvalidConfig);
    }
    let n = 1u64 << log_n;
    if ell > n || r > n {
        return Err(Error::InvalidConfig);
    }
    let dim = ell.checked_add(r).ok_or(Error::InvalidConfig)?;
    if dim > n || dim > (1u64 << subgroup_log(log_n)) {
        return Err(Error::InvalidConfig);
    }
    Ok(CodeParams { log_n, ell, r })
}

/// The evaluation point of a position.
pub fn point(log_n: u32, position: u64) -> Result<Fp> {
    if log_n < 1 || log_n > MOST_LOG_N || position >= (1u64 << log_n) {
        return Err(Error::InvalidIndex);
    }
    let s = subgroup_log(log_n);
    let coset = position >> s;
    let mut z = fp_pow(
        fp_from_u64(GENERATOR),
        coset.checked_add(1).ok_or(Error::ResourceLimit)?,
    );
    for index in 0..s {
        if (position >> index) & 1 == 1 {
            let exponent = 1u64 << (31 - index);
            z = z * fp_pow(fp_from_u64(ROOT_32), exponent);
        }
    }
    Ok(z)
}

/// The first `count` basis factors at a position: `f_i = z^(2^i)`.
pub fn factors(log_n: u32, position: u64, count: u32) -> Result<Vec<Fp>> {
    let mut values = Vec::with_capacity(count as usize);
    if count == 0 {
        return Ok(values);
    }
    let mut factor = point(log_n, position)?;
    for _ in 0..count {
        values.push(factor);
        factor = factor * factor;
    }
    Ok(values)
}

fn coefficient<T: Copy>(message: &[T], tail: &[T], degree: u64) -> Result<T> {
    if degree < tail.len() as u64 {
        Ok(tail[degree as usize])
    } else {
        let index = degree - tail.len() as u64;
        message
            .get(index as usize)
            .copied()
            .ok_or(Error::ShapeMismatch)
    }
}

fn evaluate<T: Copy + core::ops::Mul<Output = T> + core::ops::Add<Output = T>>(
    z: T,
    message: &[T],
    tail: &[T],
) -> Result<T> {
    let dimension = message.len() as u64 + tail.len() as u64;
    let mut accumulator = None;
    for degree in (0..dimension).rev() {
        let value = coefficient(message, tail, degree)?;
        accumulator = Some(match accumulator {
            None => value,
            Some(accumulator) => accumulator * z + value,
        });
    }
    accumulator.ok_or(Error::ShapeMismatch)
}

/// Evaluate the code polynomial at an arbitrary point (the OOD answer).
pub fn evaluate_ood_fp(
    plan: &CodeParams,
    message: &[Fp],
    tail: &[Fp],
    z: Fp,
) -> Result<Fp> {
    if message.len() as u64 != plan.ell || tail.len() as u64 != plan.r {
        return Err(Error::ShapeMismatch);
    }
    evaluate(z, message, tail)
}

pub fn evaluate_ood_fp3(
    plan: &CodeParams,
    message: &[Fp3],
    tail: &[Fp3],
    z: Fp3,
) -> Result<Fp3> {
    if message.len() as u64 != plan.ell || tail.len() as u64 != plan.r {
        return Err(Error::ShapeMismatch);
    }
    evaluate(z, message, tail)
}

pub fn symbol_at_fp(
    plan: &CodeParams,
    position: u64,
    message: &[Fp],
    tail: &[Fp],
) -> Result<Fp> {
    evaluate_ood_fp(plan, message, tail, point(plan.log_n, position)?)
}


/// The transform table of the protocol's evaluation code.
///
/// The table holds the product-form factors of the coset points in the
/// reference stage order; only the first `subgroup_log(log_n)` stages are
/// built, because a code's dimension never exceeds one coset.
fn transform_table(log_n: u32) -> Result<Vec<Fp>> {
    if log_n < 1 || log_n > MOST_LOG_N {
        return Err(Error::InvalidConfig);
    }
    let n = 1usize << log_n;
    let s = subgroup_log(log_n);
    let mut table = vec![Fp::ZERO; n];
    let root = fp_from_u64(ROOT_32);
    let generator = fp_from_u64(GENERATOR);
    let half = n / 2;
    let mut points = vec![Fp::ZERO; half];
    points[0] = generator;
    for b in 0..(log_n as usize - 1) {
        let i = b as u32 + 1;
        let step = if i < s {
            fp_pow(root, 1u64 << (31 - i))
        } else {
            fp_pow(generator, 1u64 << (i - s))
        };
        let upto = 1usize << b;
        for m in 0..upto {
            points[m + upto] = points[m] * step;
        }
    }
    table[half..].copy_from_slice(&points);
    for layer in 1..s {
        let at = n >> (layer + 1);
        for h in 0..at {
            table[at + h] = table[2 * at + 2 * h] * table[2 * at + 2 * h];
        }
    }
    Ok(table)
}

fn coefficient_vector<T: Copy>(
    plan: &CodeParams,
    message: &[T],
    tail: &[T],
    zero: T,
) -> Result<Vec<T>> {
    if message.len() as u64 != plan.ell || tail.len() as u64 != plan.r {
        return Err(Error::ShapeMismatch);
    }
    let mut coefficients = vec![zero; plan.dim() as usize];
    coefficients[..tail.len()].copy_from_slice(tail);
    coefficients[plan.message_at() as usize..].copy_from_slice(message);
    Ok(coefficients)
}

fn product_form_fp(plan: &CodeParams, message: &[Fp], tail: &[Fp]) -> Result<Vec<Fp>> {
    let n = plan.n() as usize;
    let log_dim = plan.log_dim() as usize;
    let table = transform_table(plan.log_n)?;
    let coefficients = coefficient_vector(plan, message, tail, Fp::ZERO)?;
    let mut word = vec![Fp::ZERO; n];
    word[..coefficients.len()].copy_from_slice(&coefficients);
    let mut at = 1usize << log_dim;
    while at < n {
        word.copy_within(0..at, at);
        at <<= 1;
    }
    let mut layer = log_dim;
    while layer > 0 {
        layer -= 1;
        let st = 1usize << layer;
        let offset = n >> (layer + 1);
        for h in 0..offset {
            let twiddle = table[offset + h];
            for j in 0..st {
                let index = 2 * st * h + j;
                let even = word[index];
                let odd = word[index + st] * twiddle;
                word[index] = even + odd;
                word[index + st] = even - odd;
            }
        }
    }
    Ok(word)
}

fn product_form_fp3(plan: &CodeParams, message: &[Fp3], tail: &[Fp3]) -> Result<Vec<Fp3>> {
    let n = plan.n() as usize;
    let log_dim = plan.log_dim() as usize;
    let table = transform_table(plan.log_n)?;
    let coefficients = coefficient_vector(plan, message, tail, Fp3::ZERO)?;
    let mut word = vec![Fp3::ZERO; n];
    word[..coefficients.len()].copy_from_slice(&coefficients);
    let mut at = 1usize << log_dim;
    while at < n {
        word.copy_within(0..at, at);
        at <<= 1;
    }
    let mut layer = log_dim;
    while layer > 0 {
        layer -= 1;
        let st = 1usize << layer;
        let offset = n >> (layer + 1);
        for h in 0..offset {
            let twiddle = Fp3::from_base(table[offset + h]);
            for j in 0..st {
                let index = 2 * st * h + j;
                let even = word[index];
                let odd = word[index + st].mul(&twiddle);
                word[index] = even.add(&odd);
                word[index + st] = even.sub(&odd);
            }
        }
    }
    Ok(word)
}

/// The linear form of the code's message and tail entries at one position,
/// in the encoder's coefficient order: the tail first, then the message. It is
/// the transpose of the product-form network, so one position costs one pass
/// over the tree rather than one encoding per entry.
pub fn coefficients_at_fp3(plan: &CodeParams, position: u64) -> Result<Vec<Fp3>> {
    let n = plan.n() as usize;
    if position >= plan.n() {
        return Err(Error::InvalidIndex);
    }
    let log_dim = plan.log_dim() as usize;
    let block = 1usize << log_dim;
    let dim = usize::try_from(plan.dim()).map_err(|_| Error::ResourceLimit)?;
    let table = transform_table(plan.log_n)?;
    let mut row = vec![Fp3::ZERO; n];
    row[position as usize] = Fp3::ONE;
    // The forward network applies its layers from log_dim-1 down to 0; the
    // transpose applies them in the opposite order.
    for layer in 0..log_dim {
        let st = 1usize << layer;
        let offset = n >> (layer + 1);
        for h in 0..offset {
            let twiddle = Fp3::from_base(table[offset + h]);
            for j in 0..st {
                let even = 2 * st * h + j;
                let odd = even + st;
                let a = row[even];
                let b = row[odd];
                row[even] = a.add(&b);
                row[odd] = twiddle.mul(&a.sub(&b));
            }
        }
    }
    // The encoder fills the rest of the word by copying the coefficient block;
    // the form on the original entries sums those copies.
    let mut out = vec![Fp3::ZERO; dim];
    let mut at = 0usize;
    while at < n {
        for index in 0..dim {
            out[index] = out[index].add(&row[at + index]);
        }
        at += block;
    }
    Ok(out)
}

#[cfg(test)]
mod coefficients_tests {
    use super::*;
    use zkie_core::common::field::PrimeCharacteristicRing;

    #[test]
    fn the_position_form_reproduces_the_encoder_symbol() {
        let plan = build_code_plan(3, 4, 1).unwrap();
        let message = [
            Fp3::from_u64(1),
            Fp3::from_u64(2),
            Fp3::from_u64(3),
            Fp3::from_u64(4),
        ];
        let tail = [Fp3::from_u64(5)];
        let word = encode_row_fp3(&plan, &message, &tail).unwrap();
        for position in 0..plan.n() {
            let form = coefficients_at_fp3(&plan, position).unwrap();
            let mut value = Fp3::ZERO;
            for (coefficient, entry) in form.iter().zip(tail.iter().chain(message.iter())) {
                value = value.add(&coefficient.mul(entry));
            }
            assert_eq!(value, word[position as usize]);
        }
    }
}

/// Encode one row with the protocol's product-form transform. The output is
/// the codeword in position order and equals direct evaluation at the points.
pub fn encode_row_fp(plan: &CodeParams, message: &[Fp], tail: &[Fp]) -> Result<Vec<Fp>> {
    let _profile_encode = crate::profile::Scope::light("code.encode_row_fp");
    product_form_fp(plan, message, tail)
}

pub fn encode_row_fp3(plan: &CodeParams, message: &[Fp3], tail: &[Fp3]) -> Result<Vec<Fp3>> {
    let _profile_encode = crate::profile::Scope::light("code.encode_row_fp3");
    product_form_fp3(plan, message, tail)
}

/// Fingerprint used by tests and diagnostics: the position-0 coefficient
/// sum is not a code invariant, so callers must not use this as a commitment.
pub fn coefficient_at(plan: &CodeParams, position: u64, message: &[Fp], tail: &[Fp]) -> Result<Fp> {
    symbol_at_fp(plan, position, message, tail)
}

/// The canonical little-endian byte encoding of one field word, for callers
/// that stream a codeword without allocating a `Vec<Fp>`.
pub fn symbol_bytes(value: Fp) -> [u8; 8] {
    fp_to_u64(value).to_le_bytes()
}
