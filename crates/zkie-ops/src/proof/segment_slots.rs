//! The slot ledger of one sub-proof segment.
//!
//! A sub-proof's sumchecks send hidden values masked by the sub-proof's mask
//! row. The verifier never learns those values: it registers each sent value
//! as a shown scalar and a public factor, and the segment hands the frame one
//! linear equation over the committed arrays, the mask coefficients and a
//! public constant. A hidden slot's value is `shown - factor * mask entry`;
//! the equation's mask coefficients are the negated coefficients times the
//! factors, and the constant is the form's constant plus the coefficients
//! times the shown values. The direct evaluation of the same form on the
//! private entries stays a prover-side check.

use zkie_core::error::{Error, Result};
use zkie_engine::pcs::request::field::Fp3;

/// One term of a linear form: a slot and its coefficient.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotCoefficient {
    pub slot: usize,
    pub coefficient: Fp3,
}

/// One linear form with a public constant.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SlotForm {
    pub constant: Fp3,
    pub terms: Vec<SlotCoefficient>,
}

impl SlotForm {
    pub fn new() -> Self {
        Self {
            constant: Fp3::ZERO,
            terms: Vec::new(),
        }
    }

    pub fn add_term(&mut self, slot: usize, coefficient: Fp3) {
        self.terms.push(SlotCoefficient { slot, coefficient });
    }
}

/// The masks of one sub-proof segment: `shown[j]` and `factor[j]` fix the
/// hidden slot `j = shown[j] - factor[j] * mask entry j`, and the pushed forms
/// accumulate the coefficients and the constant of the segment's one equation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentSlots {
    shown: Vec<Option<Fp3>>,
    factor: Vec<Fp3>,
    coefficient: Vec<Fp3>,
    constant: Fp3,
}

impl SegmentSlots {
    /// A ledger of `slots` empty slots.
    pub fn new(slots: usize) -> Result<Self> {
        if slots == 0 {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            shown: vec![None; slots],
            factor: vec![Fp3::ZERO; slots],
            coefficient: vec![Fp3::ZERO; slots],
            constant: Fp3::ZERO,
        })
    }

    pub fn slots(&self) -> usize {
        self.shown.len()
    }

    /// Registers the sent value of one slot: its shown scalar and the public
    /// factor. A slot takes one value only.
    pub fn hide(&mut self, slot: usize, shown: Fp3, factor: Fp3) -> Result<()> {
        let entry = self.shown.get_mut(slot).ok_or(Error::InvalidIndex)?;
        if entry.is_some() {
            return Err(Error::StatementMismatch);
        }
        *entry = Some(shown);
        self.factor[slot] = factor;
        Ok(())
    }

    /// The value shown for one slot.
    pub fn shown(&self, slot: usize) -> Result<Fp3> {
        self.shown
            .get(slot)
            .and_then(|value| *value)
            .ok_or(Error::InvalidIndex)
    }

    /// Adds `weight` times the form to the segment's equation.
    pub fn push(&mut self, weight: Fp3, form: &SlotForm) -> Result<()> {
        self.constant = self.constant.add(&weight.mul(&form.constant));
        for term in &form.terms {
            let slot = self
                .coefficient
                .get_mut(term.slot)
                .ok_or(Error::InvalidIndex)?;
            *slot = slot.add(&weight.mul(&term.coefficient));
        }
        Ok(())
    }

    /// The form's public part: its constant plus each coefficient times the
    /// slot's shown value. Every named slot must have a value.
    pub fn public_part(&self, form: &SlotForm) -> Result<Fp3> {
        let mut value = form.constant;
        for term in &form.terms {
            value = value.add(&term.coefficient.mul(&self.shown(term.slot)?));
        }
        Ok(value)
    }

    /// Settles the segment: the mask coefficients are the negated accumulated
    /// coefficients times the factors, and the constant adds the accumulated
    /// coefficients times the shown values. A coefficient on a slot without a
    /// value is refused; no hidden value is ever recovered.
    pub fn settle(&self) -> Result<(Vec<Fp3>, Fp3)> {
        let mut mask_coefficients = Vec::with_capacity(self.coefficient.len());
        let mut constant = self.constant;
        for (slot, coefficient) in self.coefficient.iter().enumerate() {
            if coefficient.is_zero() {
                mask_coefficients.push(Fp3::ZERO);
                continue;
            }
            let shown = self
                .shown
                .get(slot)
                .and_then(|value| *value)
                .ok_or(Error::StatementMismatch)?;
            mask_coefficients.push(Fp3::ZERO.sub(&coefficient.mul(&self.factor[slot])));
            constant = constant.add(&coefficient.mul(&shown));
        }
        Ok((mask_coefficients, constant))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(value: u64) -> Fp3 {
        Fp3::from_u64(value)
    }

    #[test]
    fn the_settled_equation_equals_the_hidden_form() {
        // Three slots, hidden values registered as shown minus factor times
        // the mask entry.
        let mut slots = SegmentSlots::new(3).unwrap();
        slots.hide(0, f(11), f(2)).unwrap();
        slots.hide(1, f(23), f(3)).unwrap();
        slots.hide(2, f(37), f(5)).unwrap();
        // Two forms: 4 + 7*h0 - 2*h1 and 5*h2 + 1.
        let mut first = SlotForm::new();
        first.constant = f(4);
        first.add_term(0, f(7));
        first.add_term(1, Fp3::ZERO.sub(&f(2)));
        let mut second = SlotForm::new();
        second.constant = Fp3::ONE;
        second.add_term(2, f(5));
        slots.push(f(9), &first).unwrap();
        slots.push(f(6), &second).unwrap();
        let (mask_coefficients, constant) = slots.settle().unwrap();
        // Check the identity at the mask entries (1, 2, 4): the settled
        // equation equals the accumulated forms on the hidden values.
        let masks = [f(1), f(2), f(4)];
        let hidden: Vec<Fp3> = (0..3)
            .map(|slot| slots.shown(slot).unwrap().sub(&slots.factor[slot].mul(&masks[slot])))
            .collect();
        let mut expected = f(9)
            .mul(&first.constant.add(&f(7).mul(&hidden[0])).add(&Fp3::ZERO.sub(&f(2)).mul(&hidden[1])));
        expected = expected.add(
            &f(6).mul(&second.constant.add(&f(5).mul(&hidden[2]))),
        );
        let mut settled = constant;
        for (slot, coefficient) in mask_coefficients.iter().enumerate() {
            settled = settled.add(&coefficient.mul(&masks[slot]));
        }
        assert_eq!(settled, expected);
    }

    #[test]
    fn a_slot_takes_one_value_and_a_form_needs_its_values() {
        let mut slots = SegmentSlots::new(2).unwrap();
        slots.hide(0, f(3), f(4)).unwrap();
        assert!(slots.hide(0, f(5), f(6)).is_err());
        assert!(slots.hide(2, f(5), f(6)).is_err());
        assert_eq!(slots.shown(0).unwrap(), f(3));
        assert!(slots.shown(1).is_err());
        let mut form = SlotForm::new();
        form.add_term(1, f(8));
        slots.push(Fp3::ONE, &form).unwrap();
        // A coefficient on a slot without a value is refused at settle.
        assert!(slots.settle().is_err());
        assert!(slots.public_part(&form).is_err());
        // A form entirely on filled slots settles, with a zero coefficient on
        // a slot the equation does not name.
        slots.hide(1, f(9), f(2)).unwrap();
        let (mask_coefficients, _constant) = slots.settle().unwrap();
        assert_eq!(mask_coefficients[0], Fp3::ZERO);
        assert_eq!(mask_coefficients[1], Fp3::ZERO.sub(&f(8).mul(&f(2))));
    }
}
