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
//!   the variance. The full carry equations of the later relation are not
//!   part of this slice.
//!
//! The module is prover/fixture-side. The verifier-side commitment binding,
//! the rsqrt table, the weight product and the output rounding are later
//! slices; this module does not claim a hidden verifier for the norm yet.

use zkie_core::error::{Error, Result};

use crate::proof::hidden_send_session::{
    ActivationCommitment, ActivationShape, CommittedActivation,
};
use crate::registry::round_div_half_up;

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
    pub squares_digits: Vec<u8>,
    pub variance_digits: Vec<u8>,
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
        let magnitude = sum.unsigned_abs();
        let mut sum_digits = Vec::new();
        let mut rest = magnitude;
        loop {
            sum_digits.push((rest & 0xff) as u8);
            rest >>= 8;
            if rest == 0 {
                break;
            }
        }
        if sum_digits.is_empty() {
            sum_digits.push(0);
        }
        let squares_digits = unsigned_digits(squares.unsigned_abs() as u128)?;
        let variance_digits = unsigned_digits(variance.unsigned_abs() as u128)?;
        Ok(Self {
            sum,
            mean,
            mean_remainder,
            squares,
            variance,
            variance_remainder,
            sum_negative: sum < 0,
            sum_digits,
            squares_digits,
            variance_digits,
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
        if self.mean_remainder <= -(valid as i128) || self.mean_remainder > valid as i128 {
            return Err(Error::InvalidProof);
        }
        if self.variance_remainder <= -(valid as i128)
            || self.variance_remainder > valid as i128
        {
            return Err(Error::InvalidProof);
        }
        if self.squares < 0 || self.variance < 0 {
            return Err(Error::InvalidProof);
        }
        Ok(())
    }
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
}
