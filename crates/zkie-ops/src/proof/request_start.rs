//! The start form of a request reduction's batched stack.
//!
//! The token sumcheck leaves one linear claim: the hiding form, the claim
//! slots and the kept value's coefficient against its row target. The stack's
//! start absorbs that claim: the claim's slot is subtracted, and the kept
//! map's value at the token point, solved from the token row, is added with
//! the weight batching coefficient. The form is kept as coefficients over one
//! constant, the token sumcheck's hiding mask entries and the index-mask claim
//! slots, so a masked sumcheck can take it as its target and claim masks.

use zkie_core::error::{Error, Result};
use zkie_engine::pcs::request::field::Fp3;

use crate::proof::psc_general::{GeneralOutput, MaskTerm};

pub use crate::proof::row_claim::SlotTerm;

/// The stack's start: one constant, one coefficient per hiding entry of the
/// token sumcheck (in entry order), and one coefficient per index-mask slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StartForm {
    pub constant: Fp3,
    pub hiding: Vec<Fp3>,
    pub claims: Vec<SlotTerm>,
    /// `eps` times the shown end value: the kept value's coefficient in the
    /// token row.
    pub kept_coefficient: Fp3,
}

fn combine(terms: &mut Vec<SlotTerm>, slot: usize, coefficient: Fp3) -> Result<()> {
    for term in terms.iter_mut() {
        if term.slot == slot {
            term.coefficient = term.coefficient.add(&coefficient);
            return Ok(());
        }
    }
    terms.push(SlotTerm { slot, coefficient });
    Ok(())
}

/// Builds the start form from a kept token sumcheck's output. `claim_shown`
/// and `claim_factor` are the token claim as it was sent and the factor of its
/// mask slot; `batch` is the weight batching coefficient of the panels that
/// consume the kept value.
pub fn token_start(
    output: &GeneralOutput,
    epsilon: Fp3,
    claim_slot: usize,
    claim_shown: Fp3,
    claim_factor: Fp3,
    batch: Fp3,
) -> Result<StartForm> {
    if output.ends.len() != 1 || output.row_maps.len() != 1 || output.row_maps[0].0 != 0 {
        return Err(Error::InvalidConfig);
    }
    if epsilon.is_zero() || output.ends[0].value.is_zero() {
        return Err(Error::InvalidConfig);
    }
    let kept_coefficient = epsilon.mul(&output.ends[0].value);
    let scale = kept_coefficient.inverse()?;
    let mut hiding = Vec::with_capacity(output.hiding_form.len());
    for coefficient in &output.hiding_form {
        hiding.push(Fp3::ZERO.sub(&batch.mul(&coefficient).mul(&scale)));
    }
    let mut claims = Vec::with_capacity(output.claim_masks.len() + 1);
    for MaskTerm { slot, coefficient } in &output.claim_masks {
        combine(
            &mut claims,
            *slot,
            Fp3::ZERO.sub(&batch.mul(coefficient).mul(&scale)),
        )?;
    }
    combine(&mut claims, claim_slot, claim_factor)?;
    Ok(StartForm {
        constant: batch.mul(&output.row_target).mul(&scale).sub(&claim_shown),
        hiding,
        claims,
        kept_coefficient,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::proof::psc::PscRound;
    use crate::proof::psc_general::{
        general_variables, GeneralChallenges, GeneralMaskEntries, Summand,
    };
    use crate::proof::request_reduction::{prove_token_kept, token_kept_claim, verify_token_kept};

    struct Source(u64);

    impl GeneralChallenges for Source {
        fn draw(&mut self) -> Result<Fp3> {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            Ok(Fp3::from_u64(self.0))
        }
        fn draw_nonzero(&mut self) -> Result<Fp3> {
            for _ in 0..64 {
                let value = self.draw()?;
                if value != Fp3::ZERO {
                    return Ok(value);
                }
            }
            Err(Error::EntropyUnavailable)
        }
        fn send_hiding_sum(&mut self, _value: Fp3) -> Result<()> {
            Ok(())
        }
        fn send_round(&mut self, _index: usize, _round: &PscRound) -> Result<()> {
            Ok(())
        }
        fn send_end_values(&mut self, _values: &[Fp3]) -> Result<()> {
            Ok(())
        }
    }

    fn fp(value: u64) -> Fp3 {
        Fp3::from_u64(value)
    }

    fn mask_entry(slot: usize) -> Result<Fp3> {
        Ok(fp(950 + slot as u64))
    }

    const CLAIM_SLOT: usize = 3;
    const CLAIM_FACTOR: u64 = 11;

    /// One kept token sumcheck whose claim is masked by slot three. The shown
    /// claim is the hidden claim plus the mask term, so the proven sum is the
    /// hidden claim.
    fn kept_output() -> (GeneralOutput, Fp3, Fp3, Fp3) {
        let point = [fp(2), fp(3)];
        let shown = [fp(1), fp(2), fp(3), fp(4)];
        let kept = [fp(5), fp(6), fp(7), fp(8)];
        let hidden = token_kept_claim(&point, &shown, &kept).unwrap();
        let factor = fp(CLAIM_FACTOR);
        let shown_claim = hidden.add(&factor.mul(&mask_entry(CLAIM_SLOT).unwrap()));
        let claim_masks = [MaskTerm {
            slot: CLAIM_SLOT,
            coefficient: factor,
        }];
        let variables = crate::proof::request_reduction::token_variables(&point).unwrap();
        let hiding_count =
            crate::proof::psc::hp_entries(&general_variables(&variables, &Summand::ShownKept).unwrap())
                .unwrap();
        let masks = GeneralMaskEntries {
            hiding: (0..hiding_count).map(|index| fp(700 + index as u64)).collect(),
            shown: vec![fp(800)],
        };
        let (proof, _) = prove_token_kept(
            &point,
            &shown,
            &kept,
            &claim_masks,
            &mask_entry,
            shown_claim,
            &masks,
            &mut Source(47),
        )
        .unwrap();
        let output = verify_token_kept(
            &point,
            &claim_masks,
            shown_claim,
            &proof,
            masks.hiding.len(),
            &mut Source(47),
        )
        .unwrap();
        let mut source = Source(47);
        let epsilon = source.draw().unwrap();
        (output, epsilon, hidden, shown_claim)
    }

    #[test]
    fn the_start_solves_the_token_row_for_the_kept_value() {
        let (output, epsilon, hidden, shown_claim) = kept_output();
        let batch = fp(5);
        let start = token_start(
            &output,
            epsilon,
            CLAIM_SLOT,
            shown_claim,
            fp(CLAIM_FACTOR),
            batch,
        )
        .unwrap();
        assert_eq!(
            start.kept_coefficient,
            epsilon.mul(&output.ends[0].value)
        );
        // The token row's left side without the kept term, subtracted from the
        // target and divided by the kept coefficient, is the solved value.
        let mut left = Fp3::ZERO;
        for (coefficient, entry) in output.hiding_form.iter().zip(0..) {
            left = left.add(&coefficient.mul(&fp(900 + entry as u64)));
        }
        for term in &output.claim_masks {
            left = left.add(&term.coefficient.mul(&mask_entry(term.slot).unwrap()));
        }
        let solved = output
            .row_target
            .sub(&left)
            .mul(&start.kept_coefficient.inverse().unwrap());
        // The form's terms give the same batched kept value.
        let mut form = start.constant.add(&shown_claim);
        for (coefficient, entry) in start.hiding.iter().zip(0..) {
            form = form.add(&coefficient.mul(&fp(900 + entry as u64)));
        }
        for term in &start.claims {
            form = form.add(&term.coefficient.mul(&mask_entry(term.slot).unwrap()));
        }
        // The claim slot also carries the subtracted token claim itself.
        form = form.sub(&fp(CLAIM_FACTOR).mul(&mask_entry(CLAIM_SLOT).unwrap()));
        assert_eq!(form, batch.mul(&solved));
        assert_eq!(hidden, shown_claim.sub(&fp(CLAIM_FACTOR).mul(&mask_entry(CLAIM_SLOT).unwrap())));
    }

    #[test]
    fn the_start_drives_the_stack_sumcheck() {
        let (output, epsilon, _hidden, shown_claim) = kept_output();
        let batch = fp(5);
        let start = token_start(
            &output,
            epsilon,
            CLAIM_SLOT,
            shown_claim,
            fp(CLAIM_FACTOR),
            batch,
        )
        .unwrap();
        let mut terms: Vec<MaskTerm> = start
            .claims
            .iter()
            .map(|term| MaskTerm {
                slot: term.slot,
                coefficient: term.coefficient,
            })
            .collect();
        for (index, coefficient) in start.hiding.iter().enumerate() {
            terms.push(MaskTerm {
                slot: 100 + index,
                coefficient: *coefficient,
            });
        }
        let mask_value = |slot: usize| -> Result<Fp3> {
            if slot >= 100 {
                Ok(fp(900 + (slot - 100) as u64))
            } else {
                mask_entry(slot)
            }
        };
        let mut hidden = start.constant;
        for term in &terms {
            hidden = hidden.sub(&term.coefficient.mul(&mask_value(term.slot).unwrap()));
        }
        // One two-cell panel: the first cell carries the start's hidden value,
        // the second is empty.
        let panel = crate::proof::request_stack::StackPanel {
            claim: 0,
            ec: Fp3::ONE,
            ni: 2,
            ny: 1,
            weight: vec![Fp3::ONE, Fp3::ONE],
            first_claim: None,
            w: 0,
            off: 0,
        };
        let reduction = crate::proof::request_stack::StackReduction::new(vec![panel]).unwrap();
        assert_eq!(reduction.lam, 1);
        let left = vec![vec![hidden, Fp3::ZERO]];
        let right = vec![vec![Fp3::ONE, Fp3::ONE]];
        let om = vec![Fp3::ONE];
        let variables = (0..reduction.lam)
            .map(|_| crate::proof::psc::Weight::plain(3))
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let hiding_count =
            crate::proof::psc::hp_entries(&general_variables(&variables, &Summand::Product).unwrap())
                .unwrap();
        let masks = GeneralMaskEntries {
            hiding: (0..hiding_count).map(|index| fp(600 + index as u64)).collect(),
            shown: vec![fp(801), fp(802)],
        };
        let (proof, proven) = crate::proof::request_stack::prove_stack(
            &reduction,
            &left,
            &right,
            &om,
            &terms,
            &mask_value,
            start.constant,
            &masks,
            &mut Source(49),
        )
        .unwrap();
        let verified = crate::proof::request_stack::verify_stack(
            &reduction,
            &terms,
            start.constant,
            &proof,
            masks.hiding.len(),
            &mut Source(49),
        )
        .unwrap();
        assert_eq!(proven, verified);
        // Another panel value does not sum to the start.
        assert!(crate::proof::request_stack::prove_stack(
            &reduction,
            &[vec![hidden.add(&Fp3::ONE), Fp3::ZERO]],
            &right,
            &om,
            &terms,
            &mask_value,
            start.constant,
            &masks,
            &mut Source(49),
        )
        .is_err());
    }

    #[test]
    fn a_zero_epsilon_or_end_value_is_refused() {
        let (output, epsilon, _hidden, shown_claim) = kept_output();
        assert!(token_start(&output, Fp3::ZERO, CLAIM_SLOT, shown_claim, fp(1), fp(1)).is_err());
        let mut zero = output.clone();
        zero.ends[0].value = Fp3::ZERO;
        assert!(token_start(&zero, epsilon, CLAIM_SLOT, shown_claim, fp(1), fp(1)).is_err());
    }
}
