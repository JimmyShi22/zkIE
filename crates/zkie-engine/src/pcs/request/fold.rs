//! FOLD hiding masks (foundation only).
//!
//! A fresh mask is an oracle message of `2^h` mask rows whose messages are
//! built from the coefficients of the hiding polynomial U. The coefficients
//! come from private randomness; the prover records each draw's number and can
//! compare the coefficients with the values that the source regenerates. No
//! production provider wires this module yet.

use serde::{Deserialize, Serialize};
use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::code::{build_code_plan, CodeParams};
use crate::pcs::request::field::{E_DEGREE, Fp, Fp3};
use crate::primitives::transcript::{hash_labeled, tag_word, TEST_COINS};

/// The draw number that no source produces.
pub const NO_DRAW: u32 = u32::MAX;

/// The mask part of a fold shape. Group layouts and members are not part of
/// this foundation step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoldShape {
    pub code: CodeParams,
    pub mask_code: CodeParams,
    pub k: u32,
    pub h: u32,
    pub mask_leaf_class: u32,
}

impl FoldShape {
    pub fn mask_rows(&self) -> u64 {
        1u64 << self.h
    }

    pub fn mask_columns(&self) -> usize {
        fold_mask_columns(self.k, self.h)
    }
}

/// The number of entries per mask row that the hiding coefficients fill.
pub fn fold_mask_columns(k: u32, h: u32) -> usize {
    (2 * k as usize) + 1 - h.min(k) as usize
}

/// A mask code is no longer than the main code.
pub fn mask_fits(main: &CodeParams, mask: &CodeParams) -> bool {
    mask.log_n <= main.log_n
}

/// What is wrong with the mask part of a fold shape.
pub fn fold_mask_shape_fault(shape: &FoldShape) -> Option<&'static str> {
    if build_code_plan(shape.code.log_n, shape.code.ell, shape.code.r).is_err()
        || build_code_plan(
            shape.mask_code.log_n,
            shape.mask_code.ell,
            shape.mask_code.r,
        )
        .is_err()
    {
        return Some("the fold shape names a code that does not exist");
    }
    if shape.k < 1 || shape.h > shape.k {
        return Some("no row variable, or more mask rows than main rows");
    }
    if shape.h >= 40 {
        return Some("the mask has too many rows");
    }
    if !mask_fits(&shape.code, &shape.mask_code) {
        return Some("the mask code is longer than the main code");
    }
    if shape.mask_code.ell < shape.mask_columns() as u64 {
        return Some("the mask rows are too short for the hiding polynomial");
    }
    None
}

pub fn validate_fold_mask_shape(shape: &FoldShape) -> Result<()> {
    if fold_mask_shape_fault(shape).is_some() {
        return Err(Error::InvalidConfig);
    }
    Ok(())
}

/// The index of a hiding coefficient: by variable, then by degree; the last
/// variable keeps its constant term.
pub fn hp_place(k: u32, variable: u32, degree: u32) -> Result<usize> {
    if k < 1 || variable >= k || degree >= E_DEGREE as u32 {
        return Err(Error::InvalidConfig);
    }
    let last = variable + 1 == k;
    if !last && degree == 0 {
        return Err(Error::InvalidConfig);
    }
    let mut at = 0usize;
    for _ in 0..variable {
        at += E_DEGREE - 1;
    }
    Ok(at + if last { degree as usize } else { degree as usize - 1 })
}

/// The hiding polynomial's `2k + 1` coefficients are the private random
/// values; this is the identity with a length check.
pub fn fold_mask_from_coins(k: u32, coins: Vec<Fp3>) -> Result<Vec<Fp3>> {
    if k < 1 || coins.len() != 2 * k as usize + 1 {
        return Err(Error::InvalidConfig);
    }
    Ok(coins)
}

/// A deterministic coin source for tests and replay. It is an engineering
/// internal source, not a production randomness source.
#[derive(Clone, Debug)]
pub struct DeterministicMaskCoins {
    seed: Digest,
    draw: u32,
}

impl DeterministicMaskCoins {
    pub fn new(seed: Digest) -> Self {
        Self { seed, draw: 0 }
    }

    pub fn draw_number(&self) -> u32 {
        self.draw
    }

    fn value_at(&self, number: u32) -> Result<Fp3> {
        let digest = hash_labeled(
            tag_word(TEST_COINS, 0),
            number as u64,
            Some(&self.seed),
            &[],
        )?;
        let mut coefficients = [Fp::ZERO; E_DEGREE];
        for (index, coefficient) in coefficients.iter_mut().enumerate() {
            let bytes = &digest.0[index * 8..(index + 1) * 8];
            let word = u64::from_le_bytes(bytes.try_into().map_err(|_| Error::InvalidEncoding)?);
            *coefficient = Fp::from_u64(word % crate::pcs::request::field::MODULUS);
        }
        Ok(Fp3::new(coefficients[0], coefficients[1], coefficients[2]))
    }

    pub fn draw(&mut self) -> Result<(Fp3, u32)> {
        let number = self.draw;
        self.draw = self.draw.checked_add(1).ok_or(Error::ResourceLimit)?;
        Ok((self.value_at(number)?, number))
    }

    pub fn redraw(&self, number: u32) -> Result<Fp3> {
        self.value_at(number)
    }
}

/// The fresh mask of one fold: coefficients, the draws' numbers and, after it
/// is sent, its oracle id.
#[derive(Debug, PartialEq, Eq)]
pub struct FoldMask {
    coefficients: Vec<Fp3>,
    draws: Vec<u32>,
    id: Option<u64>,
}

impl FoldMask {
    pub fn sent(&self) -> bool {
        self.id.is_some()
    }

    pub fn id(&self) -> Option<u64> {
        self.id
    }

    pub fn sent_as(&mut self, id: u64) -> Result<()> {
        if self.id.is_some() {
            return Err(Error::InvalidProof);
        }
        self.id = Some(id);
        Ok(())
    }

    pub fn coefficients(&self) -> &[Fp3] {
        &self.coefficients
    }

    pub fn draws(&self) -> &[u32] {
        &self.draws
    }
}

/// Draw the `2k + 1` coefficients from a numbered source and record each
/// draw's number.
pub fn fold_mask_make(
    shape: &FoldShape,
    coins: &mut DeterministicMaskCoins,
) -> Result<FoldMask> {
    validate_fold_mask_shape(shape)?;
    let mut coefficients = Vec::with_capacity(2 * shape.k as usize + 1);
    let mut draws = Vec::with_capacity(coefficients.capacity());
    for _ in 0..2 * shape.k as usize + 1 {
        let (value, number) = coins.draw()?;
        coefficients.push(value);
        draws.push(number);
    }
    Ok(FoldMask {
        coefficients: fold_mask_from_coins(shape.k, coefficients)?,
        draws,
        id: None,
    })
}

/// The prover's self-check: every coefficient equals the value that the source
/// regenerates from its recorded draw number. All coefficients are compared.
pub fn fold_mask_as_drawn(
    mask: &FoldMask,
    coins: &DeterministicMaskCoins,
) -> Result<()> {
    if mask.coefficients.len() != mask.draws.len() {
        return Err(Error::InvalidProof);
    }
    let mut differ = false;
    for (coefficient, number) in mask.coefficients.iter().zip(&mask.draws) {
        if *number == NO_DRAW || coins.redraw(*number)? != *coefficient {
            differ = true;
        }
    }
    if differ {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// The mask's oracle message: `2^h` mask rows, no main rows, no sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaskNotice {
    pub mask_rows: u64,
    pub mask_code: CodeParams,
    pub answers: u32,
    pub leaf_class: u32,
}

pub fn fold_mask_notice(shape: &FoldShape) -> MaskNotice {
    MaskNotice {
        mask_rows: shape.mask_rows(),
        mask_code: shape.mask_code,
        answers: 0,
        leaf_class: shape.mask_leaf_class,
    }
}

/// A table of `rows` rows and `len` columns, column-major: entry (row,
/// column) is `values[row + rows * column]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskTable {
    pub rows: usize,
    pub len: usize,
    pub values: Vec<Fp3>,
}

impl MaskTable {
    pub fn new(rows: usize, len: usize, values: Vec<Fp3>) -> Result<Self> {
        let entries = rows.checked_mul(len).ok_or(Error::ResourceLimit)?;
        if rows == 0 || len == 0 || values.len() != entries {
            return Err(Error::ShapeMismatch);
        }
        Ok(Self { rows, len, values })
    }

    pub fn at(&self, row: usize, column: usize) -> Result<Fp3> {
        if row >= self.rows || column >= self.len {
            return Err(Error::InvalidIndex);
        }
        Ok(self.values[row + self.rows * column])
    }
}

/// One coefficient of the hiding polynomial, indexed from 1 as in the
/// reference formula.
fn coefficient(w: &[Fp3], k: u32, variable: u32, degree: u32) -> Result<Fp3> {
    Ok(w[hp_place(k, variable - 1, degree)?])
}

/// The fresh mask rows: the hiding coefficients laid out in `2^h` rows of
/// `len` entries, in the reference's column order.
pub fn fold_mask_table(k: u32, h: u32, len: usize, w: &[Fp3]) -> Result<Vec<Fp3>> {
    if k < 1 || h > k || len < fold_mask_columns(k, h) || w.len() != 2 * k as usize + 1 {
        return Err(Error::InvalidConfig);
    }
    let rows = 1usize << h;
    let entries = rows.checked_mul(len).ok_or(Error::ResourceLimit)?;
    let mut table = vec![Fp3::ZERO; entries];
    for b in 0..rows {
        let at = |column: usize| b + rows * column;
        table[at(0)] = if h < k {
            coefficient(w, k, h + 1, 1)?
        } else {
            coefficient(w, k, k, 0)?
        };
        let mut l = h;
        while l >= 1 && (b >> (l - 1)) & 1 == 1 {
            let first = at(0);
            table[first] = table[first].add(&coefficient(w, k, l, 1)?);
            let slot = at(l as usize);
            table[slot] = coefficient(w, k, l, 2)?;
            l -= 1;
        }
        let mut column = h as usize + 1;
        for variable in (h + 1)..=k {
            let start = if variable == h + 1 { 2 } else { 1 };
            for degree in start..=2 {
                let slot = at(column);
                table[slot] = coefficient(w, k, variable, degree)?;
                column += 1;
            }
        }
        if h < k {
            let slot = at(column);
            table[slot] = coefficient(w, k, k, 0)?;
        }
    }
    Ok(table)
}

/// The public weights u(g): U(g) = <u(g), mask table folded at g_1..g_h>.
pub fn fold_mask_weights(k: u32, h: u32, len: usize, g: &[Fp3]) -> Result<Vec<Fp3>> {
    if k < 1 || h > k || g.len() != k as usize || len < fold_mask_columns(k, h) {
        return Err(Error::InvalidConfig);
    }
    let mut after = vec![Fp3::ONE; k as usize + 1];
    for l in (0..k as usize).rev() {
        after[l] = after[l + 1].mul(&g[l]);
    }
    let mut u = vec![Fp3::ZERO; len];
    u[0] = after[h as usize];
    for l in 1..=h as usize {
        u[l] = g[l - 1].mul(&after[h as usize]);
    }
    let mut column = h as usize + 1;
    for variable in (h + 1)..=k {
        let start = if variable == h + 1 { 2 } else { 1 };
        for degree in start..=2 {
            let power = if degree == 1 {
                g[variable as usize - 1]
            } else {
                g[variable as usize - 1].mul(&g[variable as usize - 1])
            };
            u[column] = power.mul(&after[variable as usize]);
            column += 1;
        }
    }
    if h < k {
        u[column] = Fp3::ONE;
    }
    Ok(u)
}

/// A placement of one member's mask rows in the group's `2^h` mask rows.
#[derive(Clone, Debug)]
pub struct MaskPlacement {
    pub first: u64,
    pub table: MaskTable,
}

/// Gather the members' mask rows into one group table of `2^h` rows.
pub fn fold_gather_mask(h: u32, len: usize, members: &[MaskPlacement]) -> Result<MaskTable> {
    let rows = 1usize << h;
    let entries = rows.checked_mul(len).ok_or(Error::ResourceLimit)?;
    let mut values = vec![Fp3::ZERO; entries];
    for member in members {
        if member.table.len != len {
            return Err(Error::ShapeMismatch);
        }
        let first = usize::try_from(member.first).map_err(|_| Error::InvalidIndex)?;
        if first
            .checked_add(member.table.rows)
            .map(|end| end > rows)
            .unwrap_or(true)
        {
            return Err(Error::InvalidIndex);
        }
        for b in 0..member.table.rows {
            for c in 0..len {
                values[(first + b) + (c << h)] = member.table.at(b, c)?;
            }
        }
    }
    MaskTable::new(rows, len, values)
}

/// The mask draft: the notice and the rows built from the coefficients.
#[derive(Clone, Debug)]
pub struct MaskDraft {
    pub notice: MaskNotice,
    pub table: MaskTable,
}

pub fn fold_mask_draft(shape: &FoldShape, mask: &FoldMask) -> Result<MaskDraft> {
    validate_fold_mask_shape(shape)?;
    let notice = fold_mask_notice(shape);
    let len = shape.mask_code.ell as usize;
    let values = fold_mask_table(shape.k, shape.h, len, mask.coefficients())?;
    let table = MaskTable::new(1usize << shape.h, len, values)?;
    Ok(MaskDraft { notice, table })
}

/// What the verifier knows of an oracle named as a fresh mask.
#[derive(Clone, Copy, Debug)]
pub struct MaskOracle {
    pub id: u64,
    pub main_rows: u64,
    pub mask_rows: u64,
    pub mask_code: CodeParams,
    pub answers: u32,
    pub sampled: bool,
    pub members: &'static [u64],
}

/// Check that the named oracle has just the shape of a fresh mask, no sample,
/// and is not a member of the group.
pub fn fold_mask_check(shape: &FoldShape, fresh: &MaskOracle) -> Result<()> {
    validate_fold_mask_shape(shape)?;
    if fresh.main_rows != 0
        || fresh.mask_rows != (1u64 << shape.h)
        || fresh.mask_code != shape.mask_code
    {
        return Err(Error::StatementMismatch);
    }
    if fresh.sampled || fresh.answers != 0 {
        return Err(Error::StatementMismatch);
    }
    if fresh.members.contains(&fresh.id) {
        return Err(Error::StatementMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> FoldShape {
        FoldShape {
            code: build_code_plan(4, 8, 1).unwrap(),
            mask_code: build_code_plan(3, 4, 1).unwrap(),
            k: 2,
            h: 1,
            mask_leaf_class: 3,
        }
    }

    fn coins() -> DeterministicMaskCoins {
        DeterministicMaskCoins::new(Digest([9; 32]))
    }

    fn values(k: u32) -> Vec<Fp3> {
        (0..2 * k + 1)
            .map(|index| Fp3::from_u64(index as u64 + 1))
            .collect()
    }

    #[test]
    fn mask_is_drawn_and_checked() {
        let shape = shape();
        let mut source = coins();
        let mask = fold_mask_make(&shape, &mut source).unwrap();
        assert_eq!(mask.coefficients().len(), 5);
        assert_eq!(mask.draws(), &[0, 1, 2, 3, 4]);
        assert!(!mask.sent());
        fold_mask_as_drawn(&mask, &source).unwrap();

        let mut tampered = FoldMask {
            coefficients: mask.coefficients().to_vec(),
            draws: mask.draws().to_vec(),
            id: None,
        };
        tampered.coefficients[0] = tampered.coefficients[0].add(&Fp3::ONE);
        assert!(fold_mask_as_drawn(&tampered, &source).is_err());

        let mut number = FoldMask {
            coefficients: mask.coefficients().to_vec(),
            draws: mask.draws().to_vec(),
            id: None,
        };
        number.draws[1] = 4;
        assert!(fold_mask_as_drawn(&number, &source).is_err());
        number.draws[1] = NO_DRAW;
        assert!(fold_mask_as_drawn(&number, &source).is_err());
    }

    #[test]
    fn mask_cannot_be_sent_twice() {
        let shape = shape();
        let mut source = coins();
        let mut mask = fold_mask_make(&shape, &mut source).unwrap();
        mask.sent_as(7).unwrap();
        assert_eq!(mask.id(), Some(7));
        assert!(mask.sent_as(8).is_err());
    }

    #[test]
    fn mask_table_matches_the_formula() {
        let w = values(2);
        let table = fold_mask_table(2, 1, 4, &w).unwrap();
        // Column-major: index = row + rows * column.
        let expected = vec![
            w[3],            // row 0, column 0
            w[3].add(&w[0]), // row 1, column 0
            Fp3::ZERO,       // row 0, column 1
            w[1],            // row 1, column 1
            w[4],            // row 0, column 2
            w[4],            // row 1, column 2
            w[2],            // row 0, column 3
            w[2],            // row 1, column 3
        ];
        assert_eq!(table, expected);
    }

    #[test]
    fn mask_table_and_weights_give_the_hiding_polynomial() {
        let k = 3u32;
        let h = 1u32;
        let len = fold_mask_columns(k, h);
        let w = values(k);
        let g = [
            Fp3::from_u64(5),
            Fp3::from_u64(7),
            Fp3::from_u64(11),
        ];
        let table = MaskTable::new(1usize << h, len, fold_mask_table(k, h, len, &w).unwrap())
            .unwrap();
        let u = fold_mask_weights(k, h, len, &g).unwrap();
        // Fold the mask rows at the first h challenges.
        let mut folded = vec![Fp3::ZERO; len];
        for b in 0..(1usize << h) {
            let mut weight = Fp3::ONE;
            for l in 0..h as usize {
                let bit = (b >> l) & 1;
                let factor = if bit == 1 {
                    g[l]
                } else {
                    Fp3::ONE.sub(&g[l])
                };
                weight = weight.mul(&factor);
            }
            for c in 0..len {
                folded[c] = folded[c].add(&weight.mul(&table.at(b, c).unwrap()));
            }
        }
        let mut inner = Fp3::ZERO;
        for c in 0..len {
            inner = inner.add(&folded[c].mul(&u[c]));
        }
        // U(g) = sum_l w_l(g_l) * prod_{j > l} g_j, with
        // w_l(X) = sum_t w[l,t] X^t (no constant for l < k).
        let mut expected = Fp3::ZERO;
        for l in 0..k as usize {
            let mut weight = Fp3::ZERO;
            for degree in 0..E_DEGREE {
                if l + 1 < k as usize && degree == 0 {
                    continue;
                }
                let place = hp_place(k, l as u32, degree as u32).unwrap();
                let mut power = Fp3::ONE;
                for _ in 0..degree {
                    power = power.mul(&g[l]);
                }
                weight = weight.add(&w[place].mul(&power));
            }
            let mut tail = Fp3::ONE;
            for j in (l + 1)..k as usize {
                tail = tail.mul(&g[j]);
            }
            expected = expected.add(&weight.mul(&tail));
        }
        assert_eq!(inner, expected);
    }

    #[test]
    fn gather_mask_rows() {
        let first = MaskTable::new(
            1,
            2,
            vec![Fp3::from_u64(1), Fp3::from_u64(2)],
        )
        .unwrap();
        let second = MaskTable::new(
            1,
            2,
            vec![Fp3::from_u64(3), Fp3::from_u64(4)],
        )
        .unwrap();
        let group = fold_gather_mask(
            1,
            2,
            &[
                MaskPlacement { first: 0, table: first },
                MaskPlacement { first: 1, table: second },
            ],
        )
        .unwrap();
        assert_eq!(group.rows, 2);
        assert_eq!(group.at(0, 0).unwrap(), Fp3::from_u64(1));
        assert_eq!(group.at(0, 1).unwrap(), Fp3::from_u64(2));
        assert_eq!(group.at(1, 0).unwrap(), Fp3::from_u64(3));
        assert_eq!(group.at(1, 1).unwrap(), Fp3::from_u64(4));
        assert!(fold_gather_mask(
            1,
            2,
            &[MaskPlacement { first: 1, table: group.clone() }]
        )
        .is_err());
    }

    #[test]
    fn notice_draft_and_check() {
        let shape = shape();
        let mut source = coins();
        let mask = fold_mask_make(&shape, &mut source).unwrap();
        let draft = fold_mask_draft(&shape, &mask).unwrap();
        assert_eq!(draft.notice.mask_rows, 2);
        assert_eq!(draft.notice.mask_code, shape.mask_code);
        assert_eq!(draft.notice.answers, 0);
        assert_eq!(draft.table.rows, 2);
        assert_eq!(draft.table.len, shape.mask_code.ell as usize);
        assert_eq!(
            draft.table.values,
            fold_mask_table(shape.k, shape.h, shape.mask_code.ell as usize, mask.coefficients())
                .unwrap()
        );

        let good = MaskOracle {
            id: 9,
            main_rows: 0,
            mask_rows: 2,
            mask_code: shape.mask_code,
            answers: 0,
            sampled: false,
            members: &[1, 2],
        };
        fold_mask_check(&shape, &good).unwrap();
        let mut bad = good;
        bad.main_rows = 1;
        assert!(fold_mask_check(&shape, &bad).is_err());
        let mut bad = good;
        bad.mask_rows = 4;
        assert!(fold_mask_check(&shape, &bad).is_err());
        let mut bad = good;
        bad.mask_code = shape.code;
        assert!(fold_mask_check(&shape, &bad).is_err());
        let mut bad = good;
        bad.sampled = true;
        assert!(fold_mask_check(&shape, &bad).is_err());
        let mut bad = good;
        bad.answers = 1;
        assert!(fold_mask_check(&shape, &bad).is_err());
        let mut bad = good;
        bad.members = &[9, 1];
        assert!(fold_mask_check(&shape, &bad).is_err());
    }

    #[test]
    fn shape_faults_are_rejected() {
        let base = shape();
        let mut bad = base;
        bad.k = 0;
        assert!(fold_mask_shape_fault(&bad).is_some());
        let mut bad = base;
        bad.h = 3;
        assert!(fold_mask_shape_fault(&bad).is_some());
        let mut bad = base;
        bad.mask_code = build_code_plan(5, 4, 1).unwrap();
        assert!(fold_mask_shape_fault(&bad).is_some());
        let mut bad = base;
        bad.mask_code.ell = 2;
        assert!(fold_mask_shape_fault(&bad).is_some());
        let mut bad = base;
        bad.k = 4;
        assert!(fold_mask_shape_fault(&bad).is_some());
        let mut bad = base;
        bad.h = 40;
        assert!(fold_mask_shape_fault(&bad).is_some());
    }
}
