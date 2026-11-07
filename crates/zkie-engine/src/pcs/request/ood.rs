//! Out-of-domain samples and their masked answers.
//!
//! The point is drawn only after the codeword has been committed; a degenerate
//! point is used as drawn and never redrawn. The answer is masked by the folded
//! mask row, and the verifier checks the same relation.

use zkie_core::error::{Error, Result};

use crate::pcs::request::chain::Chain;
use crate::pcs::request::code::CodeParams;
use crate::pcs::request::field::{Fp3, Fp3 as Challenge};
use crate::pcs::request::code::evaluate_ood_fp3;
use crate::compute::cpu::field_sum_fp3;

/// Draw one extension-field point from the chain. No rejection, no redraw.
pub fn sample_ood_point(chain: &mut Chain) -> Result<Challenge> {
    chain.draw_e()
}

/// The masked answer: the polynomial's value at the point plus the mask value.
pub fn prove_ood_answer(
    plan: &CodeParams,
    message: &[Fp3],
    tail: &[Fp3],
    point: Challenge,
    mask_value: Challenge,
) -> Result<Challenge> {
    Ok(evaluate_ood_fp3(plan, message, tail, point)?.add(&mask_value))
}

/// Check `masked - mask = polynomial(point)`.
pub fn verify_ood(
    plan: &CodeParams,
    message: &[Fp3],
    tail: &[Fp3],
    point: Challenge,
    mask_value: Challenge,
    masked_answer: Challenge,
) -> Result<bool> {
    let value = evaluate_ood_fp3(plan, message, tail, point)?;
    Ok(masked_answer.sub(&mask_value) == value)
}

/// Add masks to a vector of answers. The two slices must have equal lengths.
pub fn mask_ood_answers(answers: &[Fp3], masks: &[Fp3]) -> Result<Vec<Fp3>> {
    if answers.len() != masks.len() {
        return Err(Error::ShapeMismatch);
    }
    Ok(answers
        .iter()
        .zip(masks)
        .map(|(answer, mask)| answer.add(mask))
        .collect())
}

/// The masked sum of several answers, with one mask per answer.
pub fn masked_answer_sum(answers: &[Fp3], masks: &[Fp3]) -> Result<Fp3> {
    Ok(field_sum_fp3(&mask_ood_answers(answers, masks)?))
}
