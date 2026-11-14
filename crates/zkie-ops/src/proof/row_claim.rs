//! Linear row claims left by the masked sumchecks.
//!
//! A masked sumcheck's output leaves one linear claim on the masks: the
//! hiding form's coefficients on the sub-proof's hiding entries, the claim
//! slots' coefficients, and the kept maps' coefficients, against the row
//! target. The caller adds the claim with the sub-proof's entry values; the
//! relation chain tests it. This module keeps that shape in one place.

use zkie_core::error::{Error, Result};
use zkie_engine::pcs::request::field::Fp3;

use crate::proof::psc_general::GeneralOutput;

/// One coefficient on an index-mask slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotTerm {
    pub slot: usize,
    pub coefficient: Fp3,
}

/// The row claim a masked sumcheck leaves: `hiding . entries + slots . masks +
/// kept . values = constant`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowForm {
    pub constant: Fp3,
    pub hiding: Vec<Fp3>,
    pub slots: Vec<SlotTerm>,
    pub kept: Vec<(usize, Fp3)>,
}

impl RowForm {
    /// The claim an output leaves. The kept maps' coefficients are the
    /// output's row maps, in kept-map order.
    pub fn from_output(output: &GeneralOutput) -> Self {
        Self {
            constant: output.row_target,
            hiding: output.hiding_form.clone(),
            slots: output
                .claim_masks
                .iter()
                .map(|term| SlotTerm {
                    slot: term.slot,
                    coefficient: term.coefficient,
                })
                .collect(),
            kept: output.row_maps.clone(),
        }
    }

    /// The claim's left side at the given entries: the caller supplies the
    /// sub-proof's hiding entries in entry order, the mask values by slot and
    /// the kept maps' values in kept-map order.
    pub fn evaluate(
        &self,
        hiding: &[Fp3],
        mask_value: &dyn Fn(usize) -> Result<Fp3>,
        kept_values: &[Fp3],
    ) -> Result<Fp3> {
        if hiding.len() != self.hiding.len() {
            return Err(Error::ShapeMismatch);
        }
        if self.kept.iter().any(|(index, _)| *index >= kept_values.len()) {
            return Err(Error::ShapeMismatch);
        }
        let mut value = Fp3::ZERO;
        for (coefficient, entry) in self.hiding.iter().zip(hiding) {
            value = value.add(&coefficient.mul(entry));
        }
        for term in &self.slots {
            value = value.add(&term.coefficient.mul(&mask_value(term.slot)?));
        }
        for (index, coefficient) in &self.kept {
            value = value.add(&coefficient.mul(&kept_values[*index]));
        }
        Ok(value)
    }

    /// Checks the claim at the given entries.
    pub fn check(
        &self,
        hiding: &[Fp3],
        mask_value: &dyn Fn(usize) -> Result<Fp3>,
        kept_values: &[Fp3],
    ) -> Result<()> {
        if self.evaluate(hiding, mask_value, kept_values)? != self.constant {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::proof::psc::PscRound;
    use crate::proof::psc_general::{GeneralChallenges, GeneralMaskEntries, Summand};
    use crate::proof::request_weights::{
        prove_merge, verify_merge, ClaimFactor, MergeValue, WeightClaim, WeightGroup, WeightLayout,
    };

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

    /// One merge's output. The claim maps sum to the shown values; the mask
    /// values used by the proof are zero.
    fn merge_output() -> (GeneralOutput, GeneralMaskEntries) {
        let mut layout = WeightLayout::new(2).unwrap();
        layout
            .add(0, 0, 1, vec![WeightGroup::e("x", 4, false).unwrap()])
            .unwrap();
        let claims = vec![
            WeightClaim {
                unit: 0,
                factors: vec![ClaimFactor::e(vec![fp(2)], vec![fp(3), fp(5)])],
            },
            WeightClaim {
                unit: 0,
                factors: vec![ClaimFactor::e(vec![fp(7)], vec![fp(11), fp(13)])],
            },
        ];
        let maps = vec![
            vec![fp(1), fp(2), fp(3), fp(4)],
            vec![fp(4), fp(6), fp(4), fp(6)],
        ];
        let values = vec![
            MergeValue {
                shown: fp(10),
                mask_slot: 0,
                mask_factor: fp(3),
            },
            MergeValue {
                shown: fp(20),
                mask_slot: 1,
                mask_factor: fp(5),
            },
        ];
        let variables = (0..layout.lk)
            .map(|_| crate::proof::psc::Weight::plain(2))
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let hiding_count = crate::proof::psc::hp_entries(
            &crate::proof::psc_general::general_variables(&variables, &Summand::Map).unwrap(),
        )
        .unwrap();
        let masks = GeneralMaskEntries {
            hiding: (0..hiding_count).map(|index| fp(600 + index as u64)).collect(),
            shown: vec![fp(801)],
        };
        let zero = |_slot: usize| Ok(Fp3::ZERO);
        let outcome = prove_merge(
            &layout,
            &claims,
            &maps,
            &values,
            &zero,
            &masks,
            &mut Source(57),
        )
        .unwrap();
        let (output, _rows, _gamma) = verify_merge(
            &layout,
            &claims,
            &values,
            &outcome.proof,
            masks.hiding.len(),
            &mut Source(57),
        )
        .unwrap();
        (output, masks)
    }

    #[test]
    fn the_row_claim_holds_on_the_proof_entries() {
        let (output, masks) = merge_output();
        let row = RowForm::from_output(&output);
        assert!(row.kept.is_empty());
        let zero = |_slot: usize| Ok(Fp3::ZERO);
        row.check(&masks.hiding, &zero, &[]).unwrap();
        // Another hiding entry or mask value breaks the row.
        let mut changed = masks.hiding.clone();
        changed[0] = changed[0].add(&Fp3::ONE);
        assert!(row.check(&changed, &zero, &[]).is_err());
        let other_mask = |slot: usize| Ok(fp(950 + slot as u64));
        assert!(row.check(&masks.hiding, &other_mask, &[]).is_err());
        assert!(row.check(&masks.hiding[..1], &zero, &[]).is_err());
    }
}
