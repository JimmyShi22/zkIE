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

/// One settled segment relation bound to the committed member that owns the
/// mask rows: the masked equation `constant + sum maskCoef[j]*mask[j] = 0`
/// becomes a row whose units carry the member's mask entries and whose target
/// is the negated constant. The entries are the owner layout's virtual indices,
/// one per mask slot; the caller fixes the owner and the mapping as trusted
/// plan data, so a bare slot number is never read against another object.
pub fn relation_row(
    member: u64,
    entries: &[u64],
    mask_coefficients: &[Fp3],
    constant: Fp3,
) -> Result<zkie_engine::pcs::request::bat::BatRow> {
    if entries.len() != mask_coefficients.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut row = zkie_engine::pcs::request::bat::BatRow::new(zkie_engine::pcs::request::bat::BatRowKind::Plain);
    for (entry, coefficient) in entries.iter().zip(mask_coefficients) {
        if coefficient.is_zero() {
            continue;
        }
        row.add_unit(member, zkie_engine::pcs::request::bat::BatPart::Mask, *entry, *coefficient);
    }
    row.target = Fp3::ZERO.sub(&constant);
    Ok(row)
}

/// The owner binding of one sub-proof row form: the committed member whose
/// mask rows hold the hiding entries and the counted slots, the virtual entry
/// of each hiding entry in order, the virtual entry of each mask slot, and the
/// owner and entry of every kept map value. Every number is trusted layout
/// data, so a slot or kept index is never read against another object.
#[derive(Clone, Copy, Debug)]
pub struct RowBinding<'a> {
    pub mask_member: u64,
    pub hiding_entries: &'a [u64],
    pub slot_entries: &'a [u64],
    pub kept: &'a [(u64, u64)],
}

/// Converts one masked sumcheck's row claim into a plain row on the owners the
/// binding names: the hiding coefficients and the slot coefficients become
/// mask units of the mask member, the kept-map coefficients become main units
/// of their members, and the form's constant is the row target.
pub fn row_form_row(
    binding: &RowBinding,
    form: &crate::proof::row_claim::RowForm,
) -> Result<zkie_engine::pcs::request::bat::BatRow> {
    use zkie_engine::pcs::request::bat::{BatPart, BatRow, BatRowKind};
    if binding.hiding_entries.len() != form.hiding.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut row = BatRow::new(BatRowKind::Plain);
    for (entry, coefficient) in binding.hiding_entries.iter().zip(&form.hiding) {
        row.add_unit(binding.mask_member, BatPart::Mask, *entry, *coefficient);
    }
    for term in &form.slots {
        let entry = *binding.slot_entries.get(term.slot).ok_or(Error::InvalidIndex)?;
        row.add_unit(binding.mask_member, BatPart::Mask, entry, term.coefficient);
    }
    for (index, coefficient) in &form.kept {
        let (member, entry) = *binding.kept.get(*index).ok_or(Error::InvalidIndex)?;
        row.add_unit(member, BatPart::Main, entry, *coefficient);
    }
    row.target = form.constant;
    Ok(row)
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
    fn a_settled_relation_becomes_one_owner_bound_row() {
        let mut slots = SegmentSlots::new(2).unwrap();
        slots.hide(0, f(7), f(2)).unwrap();
        slots.hide(1, f(9), f(3)).unwrap();
        let mut form = SlotForm::new();
        form.constant = f(5);
        form.add_term(0, f(4));
        form.add_term(1, f(6));
        slots.push(Fp3::ONE, &form).unwrap();
        let (coefficients, constant) = slots.settle().unwrap();
        // The owner layout maps slot 0 to entry 3 and slot 1 to entry 8.
        let row = relation_row(21, &[3, 8], &coefficients, constant).unwrap();
        assert_eq!(row.target, Fp3::ZERO.sub(&constant));
        assert_eq!(row.units.len(), 2);
        assert_eq!(row.units[0].index, 3);
        assert_eq!(row.units[1].index, 8);
        // The settled equation equals the hidden form at the slot mask
        // entries: the owner table carries slot 0's entry at index 3 and slot
        // 1's at index 8.
        let masks = [f(1), f(11)];
        let mut value = constant;
        for (slot, coefficient) in coefficients.iter().enumerate() {
            value = value.add(&coefficient.mul(&masks[slot]));
        }
        let hidden = [
            f(7).sub(&f(2).mul(&masks[0])),
            f(9).sub(&f(3).mul(&masks[1])),
        ];
        let expected = f(5)
            .add(&f(4).mul(&hidden[0]))
            .add(&f(6).mul(&hidden[1]));
        assert_eq!(value, expected);
        // A mapping of another length is refused.
        assert!(relation_row(21, &[3], &coefficients, constant).is_err());
    }

    #[test]
    fn a_row_form_lands_on_its_owners() {
        // The form hides two entries, counts two mask slots and keeps one map
        // value.
        let form = crate::proof::row_claim::RowForm {
            constant: f(31),
            hiding: vec![f(2), f(3)],
            slots: vec![
                crate::proof::row_claim::SlotTerm {
                    slot: 0,
                    coefficient: f(5),
                },
                crate::proof::row_claim::SlotTerm {
                    slot: 1,
                    coefficient: f(7),
                },
            ],
            kept: vec![(0, f(11))],
        };
        let binding = RowBinding {
            mask_member: 4,
            hiding_entries: &[0, 1],
            slot_entries: &[9, 12],
            kept: &[(2, 6)],
        };
        let row = row_form_row(&binding, &form).unwrap();
        assert_eq!(row.target, f(31));
        // Two hiding units, two slot units on the mask member, one kept unit
        // on its main member.
        assert_eq!(row.units.len(), 5);
        assert_eq!(row.units[2].index, 9);
        assert_eq!(row.units[4].oracle, 2);
        assert_eq!(row.units[4].index, 6);
        // A binding that does not cover the hiding entries or a slot is
        // refused.
        let short = RowBinding {
            hiding_entries: &[0],
            ..binding
        };
        assert!(row_form_row(&short, &form).is_err());
        let missing_slot = RowBinding {
            slot_entries: &[9],
            ..binding
        };
        assert!(row_form_row(&missing_slot, &form).is_err());
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
