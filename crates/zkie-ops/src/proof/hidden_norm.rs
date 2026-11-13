//! Hidden norm core: the committed activation pair and the integer
//! digit/carry/range checks of the per-position norm statistics.
//!
//! This module is the first slice of the final norm relation. It fixes the
//! externally trusted input commitment, the output commitment interface and
//! the exact integer core that the later proof stages bind:
//!
//! * the activation rows are the final norm's input matrix `xbL`;
//! * the output rows are the interface for the norm's output `hd`;
//! * for every position the core computes the valid-column sum, the
//!   round-half-up mean, the centered squares and the round-half-up variance,
//!   and a canonical base-256 digit row for the sum magnitude, the squares and
//!   the variance; it also computes the exact offset/complement decomposition
//!   of the mean and variance rounding and the byte digits of both.
//!
//! The module also computes the exact integer output tail of every position:
//! the rstd broadcast, the rstd-weighted product, the image, the half-up shift
//! rounding decomposition and the bias/clamp output, with the non-valid
//! columns fixed to image zero and the clamped bias output. The module is
//! prover/fixture-side; the verifier-side commitment binding and the relations
//! over these rows are later slices, so this module does not claim a hidden
//! verifier for the norm yet.

use zkie_core::error::{Error, Result};

use crate::proof::hidden_send_session::{
    ActivationCommitment, ActivationShape, CommittedActivation,
};
use crate::registry::{
    round_div_half_up, RSQRT_COARSE, RSQRT_COARSE_END, RSQRT_FINE, RSQRT_HALF, RSQRT_SHIFT,
};

/// The row/column shape of the final norm's input and output matrices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormShape {
    pub positions: u32,
    pub hidden: u32,
    pub valid: u32,
}

impl NormShape {
    pub fn new(positions: u32, hidden: u32, valid: u32) -> Result<Self> {
        let shape = Self {
            positions,
            hidden,
            valid,
        };
        if shape.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        Ok(shape)
    }

    pub fn fault(&self) -> Option<&'static str> {
        if self.positions == 0 || !self.positions.is_power_of_two() || self.positions > (1 << 20)
        {
            return Some("the norm has no power-of-two row count");
        }
        if self.hidden == 0 || self.hidden > (1 << 20) {
            return Some("the norm has no bounded row width");
        }
        if self.valid == 0 || self.valid > self.hidden {
            return Some("the norm has no valid column range");
        }
        None
    }
}

/// The externally trusted input commitment and the output interface
/// commitment of one norm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormCommitments {
    pub input: ActivationCommitment,
    pub output: ActivationCommitment,
}

impl NormCommitments {
    pub fn fault(&self, shape: NormShape) -> Option<&'static str> {
        if self.input.fault().is_some() || self.output.fault().is_some() {
            return Some("a norm commitment is invalid");
        }
        if self.input.shape.positions() != shape.positions as usize
            || self.input.shape.hidden() != shape.hidden as usize
            || self.output.shape.positions() != shape.positions as usize
            || self.output.shape.hidden() != shape.hidden as usize
        {
            return Some("a norm commitment has another shape");
        }
        None
    }
}

/// The trusted plan of the norm core.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormCorePlan {
    pub shape: NormShape,
    pub commitments: NormCommitments,
}

impl NormCorePlan {
    pub fn new(shape: NormShape, commitments: NormCommitments) -> Result<Self> {
        let plan = Self { shape, commitments };
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<()> {
        if self.shape.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        if self.commitments.fault(self.shape).is_some() {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }
}

/// One position's exact integer core.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormCoreRow {
    /// `sum_{i<valid} x_i`.
    pub sum: i128,
    /// `round_half_up(sum / valid)`.
    pub mean: i64,
    /// `sum - mean * valid`.
    pub mean_remainder: i128,
    /// `sum_{i<valid} (x_i - mean)^2`.
    pub squares: i128,
    /// `round_half_up(squares / valid)`.
    pub variance: i64,
    /// `squares - variance * valid`.
    pub variance_remainder: i128,
    /// Sign of `sum` and its base-256 magnitude digits, least significant
    /// first.
    pub sum_negative: bool,
    pub sum_digits: Vec<u8>,
    pub mean_negative: bool,
    pub mean_digits: Vec<u8>,
    pub squares_digits: Vec<u8>,
    pub variance_digits: Vec<u8>,
    /// The exact rounding decomposition of the mean: `u = r - lo` and
    /// `s = valid - 1 - u`, with the byte digits of both, least significant
    /// first.
    pub mean_offset: i128,
    pub mean_complement: i128,
    pub mean_offset_digits: Vec<u8>,
    pub mean_complement_digits: Vec<u8>,
    /// The same decomposition of the variance rounding.
    pub variance_offset: i128,
    pub variance_complement: i128,
    pub variance_offset_digits: Vec<u8>,
    pub variance_complement_digits: Vec<u8>,
}

impl NormCoreRow {
    pub fn from_values(values: &[i64], valid: usize) -> Result<Self> {
        if valid == 0 || valid > values.len() {
            return Err(Error::InvalidConfig);
        }
        let mut sum: i128 = 0;
        for value in &values[..valid] {
            sum = sum
                .checked_add(*value as i128)
                .ok_or(Error::BoundViolation)?;
        }
        let divisor = valid as i128;
        let mean = round_div_half_up(sum, divisor)?;
        let mean = i64::try_from(mean).map_err(|_| Error::BoundViolation)?;
        let mean_remainder = sum
            .checked_sub((mean as i128) * divisor)
            .ok_or(Error::BoundViolation)?;
        let mut squares: i128 = 0;
        for value in &values[..valid] {
            let centered = (*value as i128) - mean as i128;
            let square = centered
                .checked_mul(centered)
                .ok_or(Error::BoundViolation)?;
            squares = squares
                .checked_add(square)
                .ok_or(Error::BoundViolation)?;
        }
        let variance = round_div_half_up(squares, divisor)?;
        let variance = i64::try_from(variance).map_err(|_| Error::BoundViolation)?;
        let variance_remainder = squares
            .checked_sub((variance as i128) * divisor)
            .ok_or(Error::BoundViolation)?;
        let sum_digits = unsigned_digits(sum.unsigned_abs())?;
        let mean_digits = unsigned_digits(mean.unsigned_abs() as u128)?;
        let squares_digits = unsigned_digits(squares.unsigned_abs() as u128)?;
        let variance_digits = unsigned_digits(variance.unsigned_abs() as u128)?;
        let half = (valid / 2) as i128;
        let mean_offset = mean_remainder
            .checked_add(half)
            .ok_or(Error::BoundViolation)?;
        let mean_complement = (valid as i128 - 1)
            .checked_sub(mean_offset)
            .ok_or(Error::BoundViolation)?;
        let variance_offset = variance_remainder
            .checked_add(half)
            .ok_or(Error::BoundViolation)?;
        let variance_complement = (valid as i128 - 1)
            .checked_sub(variance_offset)
            .ok_or(Error::BoundViolation)?;
        if mean_offset < 0
            || mean_complement < 0
            || variance_offset < 0
            || variance_complement < 0
        {
            return Err(Error::BoundViolation);
        }
        let mean_offset_digits = unsigned_digits(mean_offset as u128)?;
        let mean_complement_digits = unsigned_digits(mean_complement as u128)?;
        let variance_offset_digits = unsigned_digits(variance_offset as u128)?;
        let variance_complement_digits = unsigned_digits(variance_complement as u128)?;
        Ok(Self {
            sum,
            mean,
            mean_remainder,
            squares,
            variance,
            variance_remainder,
            sum_negative: sum < 0,
            sum_digits,
            mean_negative: mean < 0,
            mean_digits,
            squares_digits,
            variance_digits,
            mean_offset,
            mean_complement,
            mean_offset_digits,
            mean_complement_digits,
            variance_offset,
            variance_complement,
            variance_offset_digits,
            variance_complement_digits,
        })
    }

    /// Rechecks the integer equations and the canonical digit rows. The
    /// relation is exact: the remaining values are the Euclidean remainder
    /// bounds of the round-half-up division and every digit is a byte.
    pub fn check(&self, values: &[i64], valid: usize) -> Result<()> {
        let expected = Self::from_values(values, valid)?;
        if *self != expected {
            return Err(Error::InvalidProof);
        }
        let low = -((valid / 2) as i128);
        let high = low + valid as i128 - 1;
        if self.mean_remainder < low || self.mean_remainder > high {
            return Err(Error::InvalidProof);
        }
        if self.variance_remainder < low || self.variance_remainder > high {
            return Err(Error::InvalidProof);
        }
        if self.mean_offset != self.mean_remainder - low
            || self.mean_complement != valid as i128 - 1 - self.mean_offset
            || digits_value(&self.mean_offset_digits).map_err(|_| Error::InvalidProof)?
                != self.mean_offset as u128
            || digits_value(&self.mean_complement_digits).map_err(|_| Error::InvalidProof)?
                != self.mean_complement as u128
        {
            return Err(Error::InvalidProof);
        }
        if self.variance_offset != self.variance_remainder - low
            || self.variance_complement != valid as i128 - 1 - self.variance_offset
            || digits_value(&self.variance_offset_digits).map_err(|_| Error::InvalidProof)?
                != self.variance_offset as u128
            || digits_value(&self.variance_complement_digits).map_err(|_| Error::InvalidProof)?
                != self.variance_complement as u128
        {
            return Err(Error::InvalidProof);
        }
        if self.squares < 0 || self.variance < 0 {
            return Err(Error::InvalidProof);
        }
        let sum_digits = digits_value(&self.sum_digits).map_err(|_| Error::InvalidProof)?;
        let mean_digits = digits_value(&self.mean_digits).map_err(|_| Error::InvalidProof)?;
        let squares_digits =
            digits_value(&self.squares_digits).map_err(|_| Error::InvalidProof)?;
        let variance_digits =
            digits_value(&self.variance_digits).map_err(|_| Error::InvalidProof)?;
        if self.sum_negative != (self.sum < 0)
            || self.mean_negative != (self.mean < 0)
            || sum_digits != self.sum.unsigned_abs()
            || mean_digits != self.mean.unsigned_abs() as u128
            || squares_digits != self.squares as u128
            || variance_digits != self.variance.unsigned_abs() as u128
        {
            return Err(Error::InvalidProof);
        }
        Ok(())
    }
}

/// The private rsqrt index witness of one position, derived from the
/// committed variance. The variance is nonnegative here: the core computes it
/// as a round-half-up of nonnegative squares, so the runtime clamp at zero is
/// not needed in this relation.
///
/// `raw` is `round_half_up(variance, 2^18)`. Exactly one segment selector is
/// one, and the matching segment carries the nonnegative slack that proves its
/// strict bounds: `raw < FINE`, `FINE <= raw < COARSE_END`, or
/// `raw >= COARSE_END`. The segment quotient and remainder decompose the
/// shifted raw value, and `index` follows the runtime piecewise formula.
/// `index_slack` proves `index < table_real_len`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RsqrtRow {
    pub variance: i64,
    pub raw: i64,
    pub raw_digits: Vec<u8>,
    pub raw_remainder: i64,
    pub raw_remainder_digits: Vec<u8>,
    pub raw_remainder_slack: i64,
    pub raw_remainder_slack_digits: Vec<u8>,
    pub fine: bool,
    pub mid: bool,
    pub coarse: bool,
    pub fine_slack: i64,
    pub fine_slack_digits: Vec<u8>,
    pub mid_q: i64,
    pub mid_r: i64,
    pub mid_q_digits: Vec<u8>,
    pub mid_high_slack: i64,
    pub mid_high_slack_digits: Vec<u8>,
    pub coarse_q: i64,
    pub coarse_r: i64,
    pub coarse_q_digits: Vec<u8>,
    pub index: i64,
    pub index_digits: Vec<u8>,
    pub index_slack: i64,
    pub index_slack_digits: Vec<u8>,
    /// The frozen table value at `index`.
    pub rstd: i64,
}

impl RsqrtRow {
    pub fn from_variance(variance: i64, table_real_len: u64, rstd: i64) -> Result<Self> {
        if variance < 0 || table_real_len == 0 {
            return Err(Error::BoundViolation);
        }
        let var = variance as i128;
        let raw = (var + RSQRT_HALF as i128) >> 18;
        let raw = i64::try_from(raw).map_err(|_| Error::BoundViolation)?;
        let raw_remainder = var + RSQRT_HALF as i128 - (raw as i128) * (RSQRT_SHIFT as i128);
        if raw_remainder < 0 || raw_remainder >= RSQRT_SHIFT as i128 {
            return Err(Error::BoundViolation);
        }
        let raw_remainder_slack = RSQRT_SHIFT as i128 - 1 - raw_remainder;
        let fine = raw < RSQRT_FINE;
        let mid = !fine && raw < RSQRT_COARSE_END;
        let coarse = raw >= RSQRT_COARSE_END;
        let fine_slack = if fine {
            RSQRT_FINE as i128 - 1 - raw as i128
        } else {
            0
        };
        let (mid_q, mid_r, mid_high_slack) = if mid {
            let shifted = raw as i128 - RSQRT_FINE as i128;
            (
                shifted >> 8,
                shifted & 0xff,
                RSQRT_COARSE_END as i128 - 1 - raw as i128,
            )
        } else {
            (0, 0, 0)
        };
        let (coarse_q, coarse_r) = if coarse {
            let shifted = raw as i128 - RSQRT_COARSE_END as i128;
            (shifted >> 16, shifted & 0xffff)
        } else {
            (0, 0)
        };
        if fine_slack < 0 || mid_q < 0 || mid_r < 0 || mid_high_slack < 0 || coarse_q < 0 {
            return Err(Error::BoundViolation);
        }
        let index = if fine {
            raw as i128
        } else if mid {
            RSQRT_FINE as i128 + mid_q
        } else {
            RSQRT_FINE as i128 + RSQRT_COARSE as i128 + coarse_q
        };
        let index = i64::try_from(index).map_err(|_| Error::BoundViolation)?;
        let index_slack = table_real_len as i128 - 1 - index as i128;
        if index_slack < 0 {
            return Err(Error::BoundViolation);
        }
        let raw_digits = unsigned_digits(raw.unsigned_abs() as u128)?;
        let raw_remainder_digits = fixed_digits(raw_remainder as u128, 3)?;
        let raw_remainder_slack_digits = fixed_digits(raw_remainder_slack as u128, 3)?;
        let fine_slack_digits = fixed_digits(fine_slack as u128, 3)?;
        let mid_q_digits = unsigned_digits(mid_q as u128)?;
        let mid_high_slack_digits = fixed_digits(mid_high_slack as u128, 4)?;
        let coarse_q_digits = unsigned_digits(coarse_q as u128)?;
        let index_digits = fixed_digits(index as u128, 4)?;
        let index_slack_digits = fixed_digits(index_slack as u128, 4)?;
        Ok(Self {
            variance,
            raw,
            raw_digits,
            raw_remainder: i64::try_from(raw_remainder).map_err(|_| Error::BoundViolation)?,
            raw_remainder_digits,
            raw_remainder_slack: i64::try_from(raw_remainder_slack)
                .map_err(|_| Error::BoundViolation)?,
            raw_remainder_slack_digits,
            fine,
            mid,
            coarse,
            fine_slack: i64::try_from(fine_slack).map_err(|_| Error::BoundViolation)?,
            fine_slack_digits,
            mid_q: i64::try_from(mid_q).map_err(|_| Error::BoundViolation)?,
            mid_r: i64::try_from(mid_r).map_err(|_| Error::BoundViolation)?,
            mid_q_digits,
            mid_high_slack: i64::try_from(mid_high_slack)
                .map_err(|_| Error::BoundViolation)?,
            mid_high_slack_digits,
            coarse_q: i64::try_from(coarse_q).map_err(|_| Error::BoundViolation)?,
            coarse_r: i64::try_from(coarse_r).map_err(|_| Error::BoundViolation)?,
            coarse_q_digits,
            index,
            index_digits,
            index_slack: i64::try_from(index_slack).map_err(|_| Error::BoundViolation)?,
            index_slack_digits,
            rstd,
        })
    }

    /// Rechecks the exact segmentation and index against the runtime formula.
    pub fn check(&self) -> Result<()> {
        if self.variance < 0 {
            return Err(Error::InvalidProof);
        }
        let table_len = (self.index as u64)
            .checked_add(self.index_slack as u64)
            .and_then(|value| value.checked_add(1))
            .ok_or(Error::BoundViolation)?;
        let expected = Self::from_variance(self.variance, table_len, self.rstd)?;
        if *self != expected {
            return Err(Error::InvalidProof);
        }
        if crate::registry::profile_rsqrt_index(self.variance)? != self.index as u64 {
            return Err(Error::InvalidProof);
        }
        let digits = [
            (&self.raw_digits, None),
            (&self.raw_remainder_digits, Some(3usize)),
            (&self.raw_remainder_slack_digits, Some(3)),
            (&self.fine_slack_digits, Some(3)),
            (&self.mid_q_digits, None),
            (&self.mid_high_slack_digits, Some(4)),
            (&self.coarse_q_digits, None),
            (&self.index_digits, Some(4)),
            (&self.index_slack_digits, Some(4)),
        ];
        for (row, limit) in digits {
            if let Some(limit) = limit {
                if row.len() > limit {
                    return Err(Error::InvalidProof);
                }
            }
        }
        Ok(())
    }
}

/// A little-endian fixed-width byte row; leading zero bytes beyond the value
/// are allowed because the width is a plan value.
fn fixed_digits(value: u128, width: usize) -> Result<Vec<u8>> {
    let mut digits = unsigned_digits(value)?;
    if digits.len() > width {
        return Err(Error::BoundViolation);
    }
    digits.resize(width, 0);
    Ok(digits)
}

/// Reconstructs a canonical byte-digit row. Empty rows and rows with a zero
/// most significant digit are rejected.
fn digits_value(digits: &[u8]) -> Result<u128> {
    if digits.is_empty() {
        return Err(Error::InvalidProof);
    }
    if digits.len() > 1 && digits.last() == Some(&0) {
        return Err(Error::InvalidProof);
    }
    let mut value: u128 = 0;
    for (index, digit) in digits.iter().enumerate() {
        let shift = index.checked_mul(8).ok_or(Error::BoundViolation)?;
        let term = (*digit as u128)
            .checked_shl(u32::try_from(shift).map_err(|_| Error::BoundViolation)?)
            .ok_or(Error::BoundViolation)?;
        value = value.checked_add(term).ok_or(Error::BoundViolation)?;
    }
    Ok(value)
}

fn unsigned_digits(value: u128) -> Result<Vec<u8>> {
    let mut digits = Vec::new();
    let mut rest = value;
    loop {
        digits.push((rest & 0xff) as u8);
        rest >>= 8;
        if rest == 0 {
            break;
        }
    }
    if digits.is_empty() {
        digits.push(0);
    }
    Ok(digits)
}

/// The prover/fixture witness of one norm core: the committed input/output
/// pair and the exact integer rows of every position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormCoreWitness {
    pub shape: NormShape,
    pub input: CommittedActivation,
    pub output: CommittedActivation,
    /// The trusted input rows used to build the witness. They stay on the
    /// prover/fixture side; this slice does not expose them to a verifier.
    pub input_values: Vec<Vec<i64>>,
    pub rows: Vec<NormCoreRow>,
}

impl NormCoreWitness {
    /// Commits the externally supplied input rows and the output interface
    /// rows, then computes the exact integer core of every position. The
    /// input commitment is the trusted part supplied by the fixture; the
    /// output commitment is the interface the consumer will read.
    pub fn commit(
        shape: NormShape,
        input_rows: &[Vec<i64>],
        input_salts: &[Vec<u8>],
        output_rows: &[Vec<i64>],
        output_salts: &[Vec<u8>],
        input_configuration: zkie_core::types::Digest,
        output_configuration: zkie_core::types::Digest,
    ) -> Result<Self> {
        if shape.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        let positions = shape.positions as usize;
        let hidden = shape.hidden as usize;
        if input_rows.len() != positions
            || input_rows.iter().any(|row| row.len() != hidden)
            || output_rows.len() != positions
            || output_rows.iter().any(|row| row.len() != hidden)
            || input_salts.len() != positions
            || output_salts.len() != positions
        {
            return Err(Error::ShapeMismatch);
        }
        let salts = input_salts
            .first()
            .map(Vec::len)
            .ok_or(Error::InvalidConfig)?;
        if salts == 0
            || input_salts.iter().any(|salt| salt.len() != salts)
            || output_salts.iter().any(|salt| salt.len() != salts)
        {
            return Err(Error::ShapeMismatch);
        }
        let activation_shape = ActivationShape::new(shape.positions, shape.hidden, salts)?;
        let input = CommittedActivation::commit_signed(
            activation_shape,
            input_rows,
            input_salts,
            input_configuration,
        )?;
        let output = CommittedActivation::commit_signed(
            activation_shape,
            output_rows,
            output_salts,
            output_configuration,
        )?;
        let rows = input_rows
            .iter()
            .map(|row| NormCoreRow::from_values(row, shape.valid as usize))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            shape,
            input,
            output,
            input_values: input_rows.to_vec(),
            rows,
        })
    }

    pub fn plan(&self) -> NormCorePlan {
        NormCorePlan {
            shape: self.shape,
            commitments: NormCommitments {
                input: *self.input.commitment(),
                output: *self.output.commitment(),
            },
        }
    }

    /// Prover/fixture self-check: the committed pair matches the plan and
    /// every row satisfies the exact integer equations.
    pub fn check(&self, plan: &NormCorePlan) -> Result<()> {
        plan.validate()?;
        if self.shape != plan.shape
            || *self.input.commitment() != plan.commitments.input
            || *self.output.commitment() != plan.commitments.output
        {
            return Err(Error::StatementMismatch);
        }
        if self.rows.len() != plan.shape.positions as usize
            || self.input_values.len() != self.rows.len()
        {
            return Err(Error::ShapeMismatch);
        }
        if self
            .input_values
            .iter()
            .any(|row| row.len() != plan.shape.hidden as usize)
        {
            return Err(Error::ShapeMismatch);
        }
        for (row, values) in self.rows.iter().zip(&self.input_values) {
            row.check(values, plan.shape.valid as usize)?;
        }
        Ok(())
    }
}

/// The exact half-up rounding of an image by `2^shift`, decomposed for the
/// digit/selector relation: the Euclidean quotient and remainder by the
/// divisor, the selector that adds one when twice the remainder reaches the
/// divisor, and the rounded quotient.
pub fn shift_round(image: i128, shift: u32) -> Result<(i128, i128, bool, i128)> {
    if shift >= 63 {
        return Err(Error::InvalidConfig);
    }
    let divisor = 1i128 << shift;
    let quotient = image.div_euclid(divisor);
    let remainder = image.rem_euclid(divisor);
    let selector = remainder * 2 >= divisor;
    let shifted = if selector {
        quotient.checked_add(1).ok_or(Error::BoundViolation)?
    } else {
        quotient
    };
    Ok((quotient, remainder, selector, shifted))
}

/// The exact integer output tail of one position.
///
/// Every row is a column-domain row over `broadcast.len()` columns. The
/// broadcast row repeats the rstd of the position; the product row is
/// `rstd * weight` on every column; the image row is `centered * product` on
/// valid columns and zero elsewhere, which is the runtime convention when the
/// centered row is taken zero on the non-valid columns; the
/// quotient/remainder/selector rows decompose the half-up shift rounding; the
/// output row is the optional clamp of `shifted + bias`. The non-valid
/// columns therefore keep `shifted = 0` and output the clamped bias, not zero.
///
/// The witness is prover/fixture-side: the verifier relations over these rows
/// are a later slice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormOutputRow {
    pub rstd: i64,
    pub shift: u32,
    pub clamp: Option<(i64, i64)>,
    pub broadcast: Vec<i64>,
    pub product: Vec<i128>,
    pub image: Vec<i128>,
    pub quotient: Vec<i128>,
    pub remainder: Vec<i128>,
    pub selector: Vec<bool>,
    pub shifted: Vec<i128>,
    pub pre_clamp: Vec<i128>,
    pub out: Vec<i64>,
}

impl NormOutputRow {
    pub fn from_core(
        core: &NormCoreRow,
        values: &[i64],
        rstd: i64,
        weight: &[i64],
        bias: &[i64],
        valid: usize,
        shift: u32,
        clamp: Option<(i64, i64)>,
    ) -> Result<Self> {
        if valid == 0
            || valid > values.len()
            || values.len() != weight.len()
            || values.len() != bias.len()
            || shift >= 63
        {
            return Err(Error::InvalidConfig);
        }
        if let Some((low, high)) = clamp {
            if low > high {
                return Err(Error::InvalidConfig);
            }
        }
        let columns = values.len();
        let mut row = Self {
            rstd,
            shift,
            clamp,
            broadcast: Vec::with_capacity(columns),
            product: Vec::with_capacity(columns),
            image: Vec::with_capacity(columns),
            quotient: Vec::with_capacity(columns),
            remainder: Vec::with_capacity(columns),
            selector: Vec::with_capacity(columns),
            shifted: Vec::with_capacity(columns),
            pre_clamp: Vec::with_capacity(columns),
            out: Vec::with_capacity(columns),
        };
        for column in 0..columns {
            let product = (rstd as i128)
                .checked_mul(weight[column] as i128)
                .ok_or(Error::BoundViolation)?;
            let image = if column < valid {
                let centered = (values[column] as i128)
                    .checked_sub(core.mean as i128)
                    .ok_or(Error::BoundViolation)?;
                centered
                    .checked_mul(product)
                    .ok_or(Error::BoundViolation)?
            } else {
                0
            };
            let (quotient, remainder, selector, shifted) = shift_round(image, shift)?;
            let pre_clamp = shifted
                .checked_add(bias[column] as i128)
                .ok_or(Error::BoundViolation)?;
            let mut value = i64::try_from(pre_clamp).map_err(|_| Error::BoundViolation)?;
            if let Some((low, high)) = clamp {
                value = value.clamp(low, high);
            }
            row.broadcast.push(rstd);
            row.product.push(product);
            row.image.push(image);
            row.quotient.push(quotient);
            row.remainder.push(remainder);
            row.selector.push(selector);
            row.shifted.push(shifted);
            row.pre_clamp.push(pre_clamp);
            row.out.push(value);
        }
        Ok(row)
    }

    /// Rechecks every column equation of the tail and the clamp semantics.
    pub fn check(
        &self,
        core: &NormCoreRow,
        values: &[i64],
        weight: &[i64],
        bias: &[i64],
        valid: usize,
    ) -> Result<()> {
        let expected = Self::from_core(
            core,
            values,
            self.rstd,
            weight,
            bias,
            valid,
            self.shift,
            self.clamp,
        )?;
        if *self != expected {
            return Err(Error::InvalidProof);
        }
        for column in 0..self.out.len() {
            if self.broadcast[column] != self.rstd
                || self.shifted[column] != self.quotient[column] + self.selector[column] as i128
                || self.pre_clamp[column]
                    != self.shifted[column] + bias[column] as i128
            {
                return Err(Error::InvalidProof);
            }
            let expected_out = match self.clamp {
                Some((low, high)) => i64::try_from(self.pre_clamp[column])
                    .map(|value| value.clamp(low, high))
                    .map_err(|_| Error::InvalidProof)?,
                None => i64::try_from(self.pre_clamp[column]).map_err(|_| Error::InvalidProof)?,
            };
            if self.out[column] != expected_out {
                return Err(Error::InvalidProof);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_core::types::Digest;

    const SALT: usize = 40;

    fn salts(positions: usize) -> Vec<Vec<u8>> {
        (0..positions)
            .map(|position| vec![(position as u8).wrapping_mul(7) ^ 0x5A; SALT])
            .collect()
    }

    fn fixture() -> (NormCorePlan, NormCoreWitness) {
        let shape = NormShape::new(2, 4, 4).unwrap();
        let input = vec![
            vec![3, -2, 5, 1],
            vec![-4, 0, 2, 7],
        ];
        let output = vec![vec![0i64; 4]; 2];
        let witness = NormCoreWitness::commit(
            shape,
            &input,
            &salts(2),
            &output,
            &salts(2),
            Digest([1; 32]),
            Digest([2; 32]),
        )
        .unwrap();
        let plan = NormCorePlan::new(shape, witness.plan().commitments).unwrap();
        (plan, witness)
    }

    #[test]
    fn positive_core_checks_the_exact_integer_equations() {
        let (plan, witness) = fixture();
        witness.check(&plan).unwrap();
        assert_eq!(witness.rows[0].sum, 7);
        assert_eq!(witness.rows[0].mean, 2);
        assert_eq!(witness.rows[0].mean_remainder, -1);
        assert_eq!(witness.rows[0].squares, 27);
        assert_eq!(witness.rows[0].variance, 7);
        assert_eq!(witness.rows[0].variance_remainder, -1);
        assert_eq!(
            NormCoreRow::from_values(&[1, 2], 2).unwrap().variance,
            1
        );
        assert_eq!(
            NormCoreRow::from_values(&[1, 2], 2)
                .unwrap()
                .variance_remainder,
            -1
        );
    }

    #[test]
    fn wrong_input_or_output_commitment_is_rejected() {
        let (plan, _witness) = fixture();
        let shape = plan.shape;
        let output = vec![vec![0i64; 4]; 2];
        let changed_input = vec![
            vec![3, -2, 5, 2],
            vec![-4, 0, 2, 7],
        ];
        let witness = NormCoreWitness::commit(
            shape,
            &changed_input,
            &salts(2),
            &output,
            &salts(2),
            Digest([1; 32]),
            Digest([2; 32]),
        )
        .unwrap();
        assert!(witness.check(&plan).is_err());
        let witness = NormCoreWitness::commit(
            shape,
            &[
                vec![3, -2, 5, 1],
                vec![-4, 0, 2, 7],
            ],
            &salts(2),
            &vec![vec![1i64; 4]; 2],
            &salts(2),
            Digest([1; 32]),
            Digest([2; 32]),
        )
        .unwrap();
        assert!(witness.check(&plan).is_err());
    }

    #[test]
    fn tampered_intermediate_digits_or_ranges_are_rejected() {
        let (plan, witness) = fixture();
        let mut changed = witness.clone();
        changed.rows[0].mean += 1;
        assert!(changed.check(&plan).is_err());
        let mut changed = witness.clone();
        changed.rows[0].squares_digits[0] ^= 1;
        assert!(changed.check(&plan).is_err());
        let mut changed = witness.clone();
        changed.rows[0].variance_remainder += 1;
        assert!(changed.check(&plan).is_err());
        let mut changed = witness.clone();
        changed.rows[0].sum_digits[0] ^= 1;
        assert!(changed.check(&plan).is_err());
        let mut changed = witness.clone();
        changed.rows[0].mean_offset += 1;
        assert!(changed.check(&plan).is_err());
        let mut changed = witness.clone();
        changed.rows[0].variance_complement_digits[0] ^= 1;
        assert!(changed.check(&plan).is_err());
        let mut changed = witness.clone();
        changed.rows[0].mean_digits[0] ^= 1;
        assert!(changed.check(&plan).is_err());
        let mut changed = witness.clone();
        changed.rows[0].mean_negative = true;
        assert!(changed.check(&plan).is_err());
    }

    #[test]
    fn rounding_decomposition_covers_ties_negatives_and_unit_divisor() {
        let tie = NormCoreRow::from_values(&[-2, 0, 0, 0], 4).unwrap();
        assert_eq!(tie.mean_remainder, -2);
        assert_eq!(tie.mean_offset, 0);
        assert_eq!(tie.mean_complement, 3);
        tie.check(&[-2, 0, 0, 0], 4).unwrap();

        let negative = NormCoreRow::from_values(&[-3, 0, 0, 0], 4).unwrap();
        assert_eq!(negative.mean_remainder, 1);
        assert_eq!(negative.mean_offset, 3);
        assert_eq!(negative.mean_complement, 0);
        assert!(negative.mean_negative);
        assert_eq!(negative.mean_digits, vec![1]);
        negative.check(&[-3, 0, 0, 0], 4).unwrap();

        let odd = NormCoreRow::from_values(&[1, 0, 0], 3).unwrap();
        assert_eq!(odd.mean_remainder, 1);
        assert_eq!(odd.mean_offset, 2);
        assert_eq!(odd.mean_complement, 0);
        odd.check(&[1, 0, 0], 3).unwrap();

        let unit = NormCoreRow::from_values(&[5], 1).unwrap();
        assert_eq!(unit.mean_remainder, 0);
        assert_eq!(unit.mean_offset, 0);
        assert_eq!(unit.mean_complement, 0);
        assert_eq!(unit.variance_offset, 0);
        assert_eq!(unit.variance_complement, 0);
        unit.check(&[5], 1).unwrap();
    }

    #[test]
    fn rsqrt_row_matches_the_runtime_segmentation() {
        let table_len = 7_340_032u64;
        let rstd = 12_345i64;
        let zero = RsqrtRow::from_variance(0, table_len, rstd).unwrap();
        assert_eq!(zero.raw, 0);
        assert!(zero.fine);
        assert_eq!(zero.index, 0);
        zero.check().unwrap();
        let mut raw_values = vec![1i64, crate::registry::RSQRT_FINE - 1, crate::registry::RSQRT_FINE];
        for offset in 0..4i64 {
            raw_values.push(crate::registry::RSQRT_COARSE_END - offset);
            raw_values.push(crate::registry::RSQRT_COARSE_END + offset);
        }
        raw_values.push(crate::registry::RSQRT_COARSE_END + 65_536);
        for raw in raw_values {
            let variance =
                raw * crate::registry::RSQRT_SHIFT - crate::registry::RSQRT_HALF;
            let row = RsqrtRow::from_variance(variance, table_len, rstd).unwrap();
            assert_eq!(row.raw, raw);
            assert_eq!(
                row.index,
                crate::registry::profile_rsqrt_index(variance).unwrap() as i64
            );
            let selected = [row.fine, row.mid, row.coarse]
                .iter()
                .filter(|value| **value)
                .count();
            assert_eq!(selected, 1);
            if row.fine {
                assert!(row.raw < crate::registry::RSQRT_FINE);
                assert_eq!(row.fine_slack, crate::registry::RSQRT_FINE - 1 - raw);
            }
            if row.mid {
                assert!(row.raw >= crate::registry::RSQRT_FINE);
                assert!(row.raw < crate::registry::RSQRT_COARSE_END);
                assert_eq!(row.mid_q, (raw - crate::registry::RSQRT_FINE) >> 8);
                assert_eq!(row.mid_r, (raw - crate::registry::RSQRT_FINE) & 0xff);
                assert_eq!(
                    row.mid_high_slack,
                    crate::registry::RSQRT_COARSE_END - 1 - raw
                );
            }
            if row.coarse {
                assert!(row.raw >= crate::registry::RSQRT_COARSE_END);
                assert_eq!(
                    row.coarse_q,
                    (raw - crate::registry::RSQRT_COARSE_END) >> 16
                );
                assert_eq!(
                    row.coarse_r,
                    (raw - crate::registry::RSQRT_COARSE_END) & 0xffff
                );
            }
            row.check().unwrap();
        }
        for delta in [
            0i64,
            1,
            crate::registry::RSQRT_HALF - 1,
            crate::registry::RSQRT_HALF,
            crate::registry::RSQRT_HALF + 1,
            crate::registry::RSQRT_SHIFT - 1,
        ] {
            let variance = 3 * crate::registry::RSQRT_SHIFT + delta;
            let row = RsqrtRow::from_variance(variance, table_len, rstd).unwrap();
            assert_eq!(row.raw, (variance + crate::registry::RSQRT_HALF) / crate::registry::RSQRT_SHIFT);
            assert_eq!(
                row.index,
                crate::registry::profile_rsqrt_index(variance).unwrap() as i64
            );
            row.check().unwrap();
        }
        assert!(RsqrtRow::from_variance(-1, table_len, rstd).is_err());
        assert!(RsqrtRow::from_variance(0, 0, rstd).is_err());
        let beyond = (crate::registry::RSQRT_COARSE_END + 65_536)
            * crate::registry::RSQRT_SHIFT
            - crate::registry::RSQRT_HALF;
        assert!(RsqrtRow::from_variance(beyond, 1_024, rstd).is_err());
        let mut row = RsqrtRow::from_variance(1_000, table_len, rstd).unwrap();
        row.index += 1;
        assert!(row.check().is_err());
    }

    #[test]
    fn invalid_shape_or_roots_are_rejected() {
        assert!(NormShape::new(2, 4, 5).is_err());
        assert!(NormShape::new(3, 4, 4).is_err());
        let (plan, witness) = fixture();
        let mut missing = plan.commitments;
        missing.input.root = Digest::ZERO;
        assert!(NormCorePlan::new(plan.shape, missing).is_err());
        assert!(witness.check(&plan).is_ok());
    }

    #[test]
    fn shift_rounding_follows_the_half_up_discipline() {
        assert_eq!(shift_round(400, 3).unwrap(), (50, 0, false, 50));
        assert_eq!(shift_round(300, 3).unwrap(), (37, 4, true, 38));
        assert_eq!(shift_round(-500, 3).unwrap(), (-63, 4, true, -62));
        assert_eq!(shift_round(-1200, 3).unwrap(), (-150, 0, false, -150));
        assert_eq!(shift_round(4, 3).unwrap(), (0, 4, true, 1));
        assert_eq!(shift_round(-4, 3).unwrap(), (-1, 4, true, 0));
        assert_eq!(shift_round(3, 3).unwrap(), (0, 3, false, 0));
        assert_eq!(shift_round(-3, 3).unwrap(), (-1, 5, true, 0));
        assert_eq!(shift_round(7, 0).unwrap(), (7, 0, false, 7));
        assert!(shift_round(1, 63).is_err());
    }

    #[test]
    fn output_tail_matches_the_runtime_operator() {
        use crate::profile_norm::ProfileLayerNormOp;
        use crate::registry::{profile_rsqrt_index, PureForward};

        let values = vec![vec![3i64, -2], vec![-4, 7]];
        let weight = vec![2i64, -1, 3, 1];
        let bias = vec![1i64, 2, -1, 2];
        let table = vec![100i64; 8];
        let shift = 3u32;
        let clamp = Some((-20i64, 20i64));
        let valid = 2usize;
        let columns = weight.len();
        let op = ProfileLayerNormOp {
            rows: values.len(),
            cols: columns,
            valid_cols: valid,
            output_shift: shift,
            clamp,
        };
        let flat = |rows: &[Vec<i64>]| -> Vec<i64> {
            let mut flat = Vec::with_capacity(rows.len() * columns);
            for row in rows {
                flat.extend_from_slice(row);
                flat.resize(flat.len() + columns - row.len(), 0);
            }
            flat
        };
        let x = flat(&values);
        let padded = values
            .iter()
            .map(|row| {
                let mut padded = row.clone();
                padded.resize(columns, 0);
                padded
            })
            .collect::<Vec<_>>();
        let repeated = |row: &[i64]| -> Vec<i64> {
            let mut flat = Vec::with_capacity(values.len() * columns);
            for _ in 0..values.len() {
                flat.extend_from_slice(row);
            }
            flat
        };
        let record = op
            .forward_values(&[x.clone(), repeated(&weight), repeated(&bias), table.clone()])
            .unwrap();
        let outputs = &record.outputs[0];
        let images = &record.raw_accumulators[0];
        let rstds = &record.auxiliaries[2];
        for (position, row) in values.iter().enumerate() {
            let core = NormCoreRow::from_values(row, valid).unwrap();
            let rstd = table[profile_rsqrt_index(core.variance).unwrap() as usize];
            assert_eq!(rstd, rstds[position]);
            let padded_row = &padded[position];
            let tail = NormOutputRow::from_core(
                &core, padded_row, rstd, &weight, &bias, valid, shift, clamp,
            )
            .unwrap();
            tail.check(&core, padded_row, &weight, &bias, valid).unwrap();
            let start = position * columns;
            assert_eq!(&tail.image[..], &images[start..start + columns]);
            assert_eq!(&tail.out[..], &outputs[start..start + columns]);
            for column in 0..columns {
                assert_eq!(tail.broadcast[column], rstd);
                assert_eq!(tail.product[column], rstd as i128 * weight[column] as i128);
                assert_eq!(
                    tail.shifted[column],
                    tail.quotient[column] + tail.selector[column] as i128
                );
                if column >= valid {
                    assert_eq!(tail.image[column], 0);
                    assert_eq!(tail.out[column], bias[column].clamp(-20, 20));
                }
            }
        }
        assert!(NormOutputRow::from_core(
            &NormCoreRow::from_values(&values[0], valid).unwrap(),
            &padded[0],
            100,
            &weight,
            &bias,
            valid,
            shift,
            Some((5, 4)),
        )
        .is_err());
    }

    #[test]
    fn output_tail_tampering_is_rejected() {
        let values = vec![3i64, -2, 0, 0];
        let weight = vec![2i64, -1, 3, 1];
        let bias = vec![1i64, 2, -1, 2];
        let core = NormCoreRow::from_values(&values, 2).unwrap();
        let tail =
            NormOutputRow::from_core(&core, &values, 100, &weight, &bias, 2, 3, None).unwrap();
        tail.check(&core, &values, &weight, &bias, 2).unwrap();
        let mut changed = tail.clone();
        changed.image[0] += 1;
        assert!(changed.check(&core, &values, &weight, &bias, 2).is_err());
        let mut changed = tail.clone();
        changed.quotient[0] += 1;
        assert!(changed.check(&core, &values, &weight, &bias, 2).is_err());
        let mut changed = tail.clone();
        changed.selector[1] = !changed.selector[1];
        assert!(changed.check(&core, &values, &weight, &bias, 2).is_err());
        let mut changed = tail.clone();
        changed.out[3] -= 1;
        assert!(changed.check(&core, &values, &weight, &bias, 2).is_err());
        let mut changed = tail.clone();
        changed.broadcast[0] = -100;
        assert!(changed.check(&core, &values, &weight, &bias, 2).is_err());
        assert!(NormOutputRow::from_core(&core, &values, 100, &weight, &bias[..3], 2, 3, None)
            .is_err());
    }
}
