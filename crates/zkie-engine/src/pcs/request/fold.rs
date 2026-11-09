//! FOLD hiding masks (foundation only).
//!
//! A fresh mask is an oracle message of `2^h` mask rows whose messages are
//! built from the coefficients of the hiding polynomial U. Each mask also has
//! a random tail with which its rows are encoded; both the coefficients and
//! the tail are drawn from a numbered source, and the prover's self-check
//! compares every value with the source's regeneration.
//!
//! The normal build provides only the `MaskCoins` interface. A production
//! implementation must draw the coefficients and the tail from an independent,
//! secret and fresh source (a CSPRNG or a key derivation) with a per-mask
//! lifecycle: one mask for one fold, never reused, and its values never
//! derived from public data. That implementation, its lifecycle and its ledger
//! are not present here, and no production provider wires this module. The
//! deterministic numbered source lives in the test module behind `cfg(test)`;
//! it is public and reproducible, provides no hiding, and must not produce a
//! proof.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::code::{build_code_plan, CodeParams};
use crate::pcs::request::field::{E_DEGREE, Fp3};
use crate::pcs::request::runtime_tree::{MaskTree, MaskTreePlan};
use crate::primitives::codec::CanonicalWriter;
use crate::primitives::transcript::{hash_labeled, tag_word, STATEMENT};

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
    /// The number of mask rows, with the shift checked. Public entries call
    /// this before using the shape; a shape whose `h` does not fit a shift is
    /// rejected instead of panicking.
    pub fn mask_rows(&self) -> Result<u64> {
        if self.h >= 64 {
            return Err(Error::InvalidConfig);
        }
        Ok(1u64 << self.h)
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

/// The random tail of the mask rows' encodings: `rows * len` values in
/// row-major order, `(row, column) -> row * len + column`, with the draw
/// number of each value. The tail is part of the sent mask message, not a
/// secret kept apart from it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskTail {
    pub rows: usize,
    pub len: usize,
    pub values: Vec<Fp3>,
    pub draws: Vec<u32>,
}

impl MaskTail {
    /// A tail with `rows * len` values and their draw numbers. `len` may be
    /// zero (a mask code with no random tail); `rows` may not.
    pub fn new(
        rows: usize,
        len: usize,
        values: Vec<Fp3>,
        draws: Vec<u32>,
    ) -> Result<Self> {
        let entries = rows.checked_mul(len).ok_or(Error::ResourceLimit)?;
        if rows == 0 || values.len() != entries || draws.len() != entries {
            return Err(Error::ShapeMismatch);
        }
        Ok(Self {
            rows,
            len,
            values,
            draws,
        })
    }

    /// The structural check. All accessors are checked as well, so a malformed
    /// tail cannot panic even if it was built without `new`.
    pub fn fault(&self) -> Option<&'static str> {
        if self.rows == 0 {
            return Some("a mask tail has no rows");
        }
        match self.rows.checked_mul(self.len) {
            Some(entries) if entries == self.values.len() && entries == self.draws.len() => None,
            Some(_) => Some("a mask tail's values do not match its shape"),
            None => Some("a mask tail's shape overflows"),
        }
    }

    pub fn at(&self, row: usize, column: usize) -> Result<Fp3> {
        if row >= self.rows || column >= self.len {
            return Err(Error::InvalidIndex);
        }
        let index = row
            .checked_mul(self.len)
            .and_then(|index| index.checked_add(column))
            .ok_or(Error::ResourceLimit)?;
        self.values.get(index).copied().ok_or(Error::ShapeMismatch)
    }
}

/// The numbered source of a fresh mask's coefficients and encoding tail.
///
/// A production implementation must draw fresh secret values from an
/// independent CSPRNG or key derivation, with one mask per fold and no reuse;
/// that implementation and its lifecycle are not present in this foundation.
pub trait MaskCoins {
    /// Draw one extension-field value and return it with its draw number.
    fn draw(&mut self) -> Result<(Fp3, u32)>;

    /// Regenerate the value drawn under `number`.
    fn redraw(&self, number: u32) -> Result<Fp3>;
}

/// The fresh mask of one fold: coefficients, the draws' numbers, the random
/// tail of the rows' encodings after it is drawn, and, after it is sent, its
/// oracle id.
#[derive(Debug, PartialEq, Eq)]
pub struct FoldMask {
    coefficients: Vec<Fp3>,
    draws: Vec<u32>,
    tail: Option<MaskTail>,
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

    pub fn tail(&self) -> Option<&MaskTail> {
        self.tail.as_ref()
    }
}

/// Draw the `2k + 1` coefficients from a numbered source and record each
/// draw's number. The encoding tail is drawn separately by
/// `fold_mask_make_tail`, as it is when the mask rows are sent.
pub fn fold_mask_make(
    shape: &FoldShape,
    coins: &mut impl MaskCoins,
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
        tail: None,
        id: None,
    })
}

/// Draw the random tail of the mask rows' encodings: `2^h` rows of the mask
/// code's `r` entries each, row-major, one numbered draw per value. The send
/// path calls this when the mask draft is encoded; the numbers stay with the
/// mask for `fold_mask_as_drawn`. A mask draws its tail once.
pub fn fold_mask_make_tail(
    shape: &FoldShape,
    mask: &mut FoldMask,
    coins: &mut impl MaskCoins,
) -> Result<()> {
    validate_fold_mask_shape(shape)?;
    if mask.tail.is_some() {
        return Err(Error::InvalidProof);
    }
    let rows = usize::try_from(shape.mask_rows()?).map_err(|_| Error::ResourceLimit)?;
    let len = usize::try_from(shape.mask_code.r).map_err(|_| Error::ResourceLimit)?;
    let entries = rows.checked_mul(len).ok_or(Error::ResourceLimit)?;
    let mut values = Vec::with_capacity(entries);
    let mut draws = Vec::with_capacity(entries);
    for _ in 0..entries {
        let (value, number) = coins.draw()?;
        values.push(value);
        draws.push(number);
    }
    mask.tail = Some(MaskTail::new(rows, len, values, draws)?);
    Ok(())
}

/// The prover's self-check for a mask of `shape`: every coefficient and every
/// value of the mask rows' encoding tail equals the value that the source
/// regenerates from its recorded draw number. The mask must have the shape's
/// `2k + 1` coefficients and a tail of the shape's `2^h` rows and mask-code
/// `r` entries. All values are compared, with no early end. A mask whose tail
/// was never drawn is rejected: the check must not silently compare nothing.
/// A `NO_DRAW` number is treated as a difference here, so a lost number fails
/// closed.
pub fn fold_mask_as_drawn(
    shape: &FoldShape,
    mask: &FoldMask,
    coins: &impl MaskCoins,
) -> Result<()> {
    validate_fold_mask_shape(shape)?;
    let coefficients = 2 * shape.k as usize + 1;
    if mask.coefficients.len() != coefficients || mask.draws.len() != coefficients {
        return Err(Error::InvalidProof);
    }
    let mut differ = false;
    for (coefficient, number) in mask.coefficients.iter().zip(&mask.draws) {
        if *number == NO_DRAW || coins.redraw(*number)? != *coefficient {
            differ = true;
        }
    }
    let tail_rows = usize::try_from(shape.mask_rows()?).map_err(|_| Error::ResourceLimit)?;
    let tail_len = usize::try_from(shape.mask_code.r).map_err(|_| Error::ResourceLimit)?;
    let tail = mask.tail.as_ref().ok_or(Error::InvalidProof)?;
    if tail.fault().is_some() || tail.rows != tail_rows || tail.len != tail_len {
        return Err(Error::InvalidProof);
    }
    for (value, number) in tail.values.iter().zip(&tail.draws) {
        if *number == NO_DRAW || coins.redraw(*number)? != *value {
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

pub fn fold_mask_notice(shape: &FoldShape) -> Result<MaskNotice> {
    validate_fold_mask_shape(shape)?;
    Ok(MaskNotice {
        mask_rows: shape.mask_rows()?,
        mask_code: shape.mask_code,
        answers: 0,
        leaf_class: shape.mask_leaf_class,
    })
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
    /// A table with `rows * len` values. `len` may be zero (an empty mask
    /// tail); `rows` may not.
    pub fn new(rows: usize, len: usize, values: Vec<Fp3>) -> Result<Self> {
        let entries = rows.checked_mul(len).ok_or(Error::ResourceLimit)?;
        if rows == 0 || values.len() != entries {
            return Err(Error::ShapeMismatch);
        }
        Ok(Self { rows, len, values })
    }

    /// The structural check. All accessors are checked as well, so a malformed
    /// table cannot panic even if it was built without `new`.
    pub fn fault(&self) -> Option<&'static str> {
        if self.rows == 0 {
            return Some("a mask table has no rows");
        }
        match self.rows.checked_mul(self.len) {
            Some(entries) if entries == self.values.len() => None,
            Some(_) => Some("a mask table's values do not match its shape"),
            None => Some("a mask table's shape overflows"),
        }
    }

    pub fn at(&self, row: usize, column: usize) -> Result<Fp3> {
        if row >= self.rows || column >= self.len {
            return Err(Error::InvalidIndex);
        }
        let index = column
            .checked_mul(self.rows)
            .and_then(|index| index.checked_add(row))
            .ok_or(Error::ResourceLimit)?;
        self.values.get(index).copied().ok_or(Error::ShapeMismatch)
    }
}

/// One coefficient of the hiding polynomial, indexed from 1 as in the layout
/// formula.
fn coefficient(w: &[Fp3], k: u32, variable: u32, degree: u32) -> Result<Fp3> {
    Ok(w[hp_place(k, variable - 1, degree)?])
}

/// The fresh mask rows: the hiding coefficients laid out in `2^h` rows of
/// `len` entries, in the layout's column order.
pub fn fold_mask_table(k: u32, h: u32, len: usize, w: &[Fp3]) -> Result<Vec<Fp3>> {
    if k < 1
        || h >= 40
        || h > k
        || len < fold_mask_columns(k, h)
        || w.len() != 2 * k as usize + 1
    {
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
    if k < 1
        || h >= 40
        || h > k
        || g.len() != k as usize
        || len < fold_mask_columns(k, h)
    {
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
///
/// The layout is a precondition: each member's rows are a power of two, its
/// first row is aligned to that number of rows, its rows lie inside the
/// group's `2^h` rows, and no two members' rows overlap. A layout builder that
/// supplies overlapping or misaligned blocks must be fixed, not silently
/// truncated, so this function rejects both cases.
pub fn fold_gather_mask(h: u32, len: usize, members: &[MaskPlacement]) -> Result<MaskTable> {
    if h >= 40 {
        return Err(Error::InvalidConfig);
    }
    let rows = 1usize << h;
    let entries = rows.checked_mul(len).ok_or(Error::ResourceLimit)?;
    let mut values = vec![Fp3::ZERO; entries];
    let mut occupied = vec![false; rows];
    for member in members {
        let member_rows = member.table.rows;
        if member.table.len != len
            || member.table.fault().is_some()
            || member_rows == 0
            || !member_rows.is_power_of_two()
        {
            return Err(Error::ShapeMismatch);
        }
        let first = usize::try_from(member.first).map_err(|_| Error::InvalidIndex)?;
        if first % member_rows != 0 {
            return Err(Error::InvalidIndex);
        }
        let end = first.checked_add(member_rows).ok_or(Error::InvalidIndex)?;
        if end > rows {
            return Err(Error::InvalidIndex);
        }
        for row in &mut occupied[first..end] {
            if *row {
                return Err(Error::InvalidIndex);
            }
            *row = true;
        }
        for b in 0..member_rows {
            for c in 0..len {
                let row = first.checked_add(b).ok_or(Error::ResourceLimit)?;
                let offset = c.checked_mul(rows).ok_or(Error::ResourceLimit)?;
                let at = row.checked_add(offset).ok_or(Error::ResourceLimit)?;
                values[at] = member.table.at(b, c)?;
            }
        }
    }
    MaskTable::new(rows, len, values)
}

/// The mask draft: the notice, the rows built from the coefficients, and the
/// random tail of the rows' encodings.
#[derive(Clone, Debug)]
pub struct MaskDraft {
    pub notice: MaskNotice,
    pub table: MaskTable,
    pub tail: MaskTail,
}

pub fn fold_mask_draft(shape: &FoldShape, mask: &FoldMask) -> Result<MaskDraft> {
    validate_fold_mask_shape(shape)?;
    let notice = fold_mask_notice(shape)?;
    let len = usize::try_from(shape.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
    let values = fold_mask_table(shape.k, shape.h, len, mask.coefficients())?;
    let rows = usize::try_from(shape.mask_rows()?).map_err(|_| Error::ResourceLimit)?;
    let table = MaskTable::new(rows, len, values)?;
    let tail = mask.tail().ok_or(Error::InvalidProof)?;
    let tail_rows = usize::try_from(shape.mask_rows()?).map_err(|_| Error::ResourceLimit)?;
    let tail_len = usize::try_from(shape.mask_code.r).map_err(|_| Error::ResourceLimit)?;
    if tail.fault().is_some() || tail.rows != tail_rows || tail.len != tail_len {
        return Err(Error::InvalidProof);
    }
    Ok(MaskDraft {
        notice,
        table,
        tail: tail.clone(),
    })
}

/// The verifier's own group entry, taken from its independently verified
/// statement/plan registry. This is the only source of members for the
/// fresh-mask check; a proof supplies a group number, never the members.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedGroup {
    pub group: u64,
    pub members: Vec<u64>,
    /// The fresh mask id the plan fixes for this group, if any.
    pub fresh: Option<u64>,
}

/// The verifier's group registry. It must be built from the verified
/// statement/plan, not from proof bytes; this foundation trusts the caller to
/// have done so, and the FOLDEQ statement binding is the gate that closes it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VerifiedGroups {
    pub groups: Vec<VerifiedGroup>,
}

impl VerifiedGroups {
    pub fn get(&self, group: u64) -> Option<&VerifiedGroup> {
        self.groups.iter().find(|entry| entry.group == group)
    }
}

impl MaskOracle {
    /// The verifier's view of a fresh-mask notice. The notice carries no
    /// members; `claimed_group` is the proof's lookup key into the verifier's
    /// registry.
    pub fn from_notice(id: u64, claimed_group: u64, notice: &MaskNotice) -> Self {
        Self {
            id,
            claimed_group,
            main_rows: 0,
            mask_rows: notice.mask_rows,
            mask_code: notice.mask_code,
            answers: notice.answers,
            sampled: false,
        }
    }
}

/// The sum of the hiding polynomial U over `{0,1}^k`: the first value FOLD
/// sends after the fresh mask. U(X) = sum_l w_l(X_l) * prod_{j>l} X_j with
/// `w_l(X) = sum_t w[l,t] X^t` and no constant term for l < k; plain `{0,1}`
/// sums give w_l(0)+w_l(1) for every variable and a factor two for every
/// variable after l.
pub fn fold_mask_sum(k: u32, w: &[Fp3]) -> Result<Fp3> {
    if k < 1 || w.len() != 2 * k as usize + 1 {
        return Err(Error::InvalidConfig);
    }
    let two = Fp3::from_u64(2);
    let mut sum = Fp3::ZERO;
    for l in 0..k {
        let last = l + 1 == k;
        let mut weight = Fp3::ZERO;
        if last {
            // The last variable keeps its constant term, and both 0 and 1
            // evaluate it.
            weight = weight.add(&w[hp_place(k, l, 0)?].mul(&two));
        }
        for degree in 1..=2 {
            weight = weight.add(&w[hp_place(k, l, degree)?]);
        }
        // Every variable before l contributes phi(1) = 2.
        let scale = two.pow_u64(l as u64);
        sum = sum.add(&weight.mul(&scale));
    }
    Ok(sum)
}

/// The public FOLD start message: the fresh mask's oracle id and claimed
/// group, its notice, its runtime-tree root and the hiding sum. A verifier
/// checks it without any coefficient, tail or coin from the prover.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldStartMessage {
    pub id: u64,
    pub claimed_group: u64,
    pub tree_number: u64,
    pub notice: MaskNotice,
    pub root: Digest,
    pub sum: Fp3,
}

/// The prover's FOLD start: the public message, the private draft and the
/// committed runtime mask tree.
#[derive(Clone, Debug)]
pub struct FoldStartProver {
    pub message: FoldStartMessage,
    pub draft: MaskDraft,
    pub tree: MaskTree,
}

/// The single-use mask ledger. The verifier's statement owns it; a second
/// start that names the same oracle id fails.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MaskSpend {
    spent: HashSet<u64>,
}

impl MaskSpend {
    pub fn spend(&mut self, id: u64) -> Result<()> {
        if !self.spent.insert(id) {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }

    pub fn is_spent(&self, id: u64) -> bool {
        self.spent.contains(&id)
    }
}

/// The verifier's statement for one run: the group registry, the runtime mask
/// tree layouts and the shared single-use mask ledger, bound by an internal
/// statement digest. The verifier constructs it from its own verified plan and
/// commitment; the type cannot prove that origin, and building it from proof
/// bytes would be a caller error. The digest is an internal binding, not a
/// public artifact identity.
#[derive(Clone, Debug)]
pub struct VerifiedStatement {
    pub id: Digest,
    pub groups: VerifiedGroups,
    pub mask_trees: Vec<MaskTreePlan>,
    spent: Arc<Mutex<MaskSpend>>,
}

impl VerifiedStatement {
    pub fn new(groups: VerifiedGroups, mask_trees: Vec<MaskTreePlan>) -> Result<Self> {
        for tree in &mask_trees {
            if tree.fault().is_some() {
                return Err(Error::InvalidConfig);
            }
        }
        let mut writer = CanonicalWriter::new();
        writer.put_u64(groups.groups.len() as u64);
        for group in &groups.groups {
            writer.put_u64(group.group);
            writer.put_u64(group.members.len() as u64);
            for member in &group.members {
                writer.put_u64(*member);
            }
            writer.put_u64(if group.fresh.is_some() { 1 } else { 0 });
            writer.put_u64(group.fresh.unwrap_or(0));
        }
        writer.put_u64(mask_trees.len() as u64);
        for tree in &mask_trees {
            writer.put_u64(tree.tree_number);
            writer.put_u64(tree.leaf_class as u64);
            writer.put_u64(tree.log_n as u64);
            writer.put_u64(tree.mask_rows);
            writer.put_u64(tree.salt_bytes as u64);
            writer.put_digest(&tree.run);
            writer.put_digest(&tree.configuration);
        }
        let id = hash_labeled(
            tag_word(STATEMENT, 0),
            0,
            None,
            &writer.into_bytes(),
        )?;
        Ok(Self {
            id,
            groups,
            mask_trees,
            spent: Arc::new(Mutex::new(MaskSpend::default())),
        })
    }

    pub fn groups(&self) -> &VerifiedGroups {
        &self.groups
    }

    pub fn mask_tree(&self, tree_number: u64) -> Option<&MaskTreePlan> {
        self.mask_trees
            .iter()
            .find(|tree| tree.tree_number == tree_number)
    }

    pub fn spend(&self, id: u64) -> Result<()> {
        self.spent
            .lock()
            .map_err(|_| Error::InvalidSessionState)?
            .spend(id)
    }

    pub fn is_spent(&self, id: u64) -> bool {
        self.spent
            .lock()
            .map(|spent| spent.is_spent(id))
            .unwrap_or(true)
    }
}

fn mask_rows(shape: &FoldShape, mask: &FoldMask) -> Result<Vec<Vec<Fp3>>> {
    let len = usize::try_from(shape.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
    let rows = usize::try_from(shape.mask_rows()?).map_err(|_| Error::ResourceLimit)?;
    let values = fold_mask_table(shape.k, shape.h, len, mask.coefficients())?;
    let mut out = Vec::new();
    out.try_reserve_exact(rows).map_err(|_| Error::ResourceLimit)?;
    for row in 0..rows {
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(len)
            .map_err(|_| Error::ResourceLimit)?;
        for column in 0..len {
            entries.push(values[row + rows * column]);
        }
        out.push(entries);
    }
    Ok(out)
}

fn mask_tail_rows(shape: &FoldShape, mask: &FoldMask) -> Result<Vec<Vec<Fp3>>> {
    let tail = mask.tail().ok_or(Error::InvalidProof)?;
    if tail.fault().is_some()
        || tail.rows != usize::try_from(shape.mask_rows()?).map_err(|_| Error::ResourceLimit)?
        || tail.len != usize::try_from(shape.mask_code.r).map_err(|_| Error::ResourceLimit)?
    {
        return Err(Error::InvalidProof);
    }
    let mut out = Vec::new();
    out.try_reserve_exact(tail.rows)
        .map_err(|_| Error::ResourceLimit)?;
    for row in 0..tail.rows {
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(tail.len)
            .map_err(|_| Error::ResourceLimit)?;
        for column in 0..tail.len {
            entries.push(tail.at(row, column)?);
        }
        out.push(entries);
    }
    Ok(out)
}

/// Prover side: check the mask against the verifier's registry and the drawn
/// values, commit the runtime mask tree under the plan, and return the public
/// message plus the private draft. The salts are fresh prover values; the
/// production source is not implemented.
pub fn fold_start_prove(
    shape: &FoldShape,
    mask: &FoldMask,
    coins: &impl MaskCoins,
    oracle: &MaskOracle,
    groups: &VerifiedGroups,
    tree_plan: MaskTreePlan,
    salts: &[Vec<u8>],
) -> Result<FoldStartProver> {
    validate_fold_mask_shape(shape)?;
    if mask.id() != Some(oracle.id) {
        return Err(Error::InvalidProof);
    }
    fold_mask_check(shape, oracle, groups)?;
    fold_mask_as_drawn(shape, mask, coins)?;
    let draft = fold_mask_draft(shape, mask)?;
    let rows = mask_rows(shape, mask)?;
    let tails = mask_tail_rows(shape, mask)?;
    let tree = MaskTree::commit(tree_plan, shape, &rows, &tails, salts)?;
    let sum = fold_mask_sum(shape.k, mask.coefficients())?;
    let tree_number = tree_plan.tree_number;
    let message = FoldStartMessage {
        id: oracle.id,
        claimed_group: oracle.claimed_group,
        tree_number,
        notice: draft.notice,
        root: tree.root,
        sum,
    };
    Ok(FoldStartProver {
        message,
        draft,
        tree,
    })
}

/// Verifier side: check the public start message against the plan's runtime
/// tree layout, the mask shape, the verifier's group registry and the
/// single-use ledger. No coefficient, tail, coin or salt is accepted.
pub fn fold_start_verify(
    shape: &FoldShape,
    message: &FoldStartMessage,
    statement: &VerifiedStatement,
) -> Result<()> {
    validate_fold_mask_shape(shape)?;
    let tree_plan = statement
        .mask_tree(message.tree_number)
        .ok_or(Error::StatementMismatch)?;
    if tree_plan.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    let rows = tree_plan.mask_rows;
    let log_n = tree_plan.log_n;
    if rows != shape.mask_rows()?
        || log_n != shape.mask_code.log_n
        || tree_plan.leaf_class != message.notice.leaf_class
        || message.notice.mask_rows != rows
        || message.notice.mask_code != shape.mask_code
        || message.notice.answers != 0
    {
        return Err(Error::StatementMismatch);
    }
    let oracle = MaskOracle::from_notice(message.id, message.claimed_group, &message.notice);
    fold_mask_check(shape, &oracle, statement.groups())?;
    if message.root == Digest::ZERO {
        return Err(Error::InvalidProof);
    }
    statement.spend(message.id)?;
    Ok(())
}

/// What the verifier knows of an oracle named as a fresh mask. `claimed_group`
/// is only a lookup key into the verifier's registry; the members come from
/// the registry entry, never from this notice.
#[derive(Clone, Copy, Debug)]
pub struct MaskOracle {
    pub id: u64,
    pub claimed_group: u64,
    pub main_rows: u64,
    pub mask_rows: u64,
    pub mask_code: CodeParams,
    pub answers: u32,
    pub sampled: bool,
}

/// Check that the named oracle has just the shape of a fresh mask, no sample,
/// and is not a member of the group the verifier's registry holds.
///
/// This is a content check against the registry entry. It does not
/// authenticate the registry; the statement/plan must be verified before the
/// registry is used, and FOLDEQ must bind the group, the fresh oracle and the
/// layout to the statement. A proof that omits its fresh id from a group it
/// invents cannot influence the check: the notice carries no members, and the
/// members are looked up by the registry's group number.
pub fn fold_mask_check(
    shape: &FoldShape,
    fresh: &MaskOracle,
    groups: &VerifiedGroups,
) -> Result<()> {
    validate_fold_mask_shape(shape)?;
    let group = groups
        .get(fresh.claimed_group)
        .ok_or(Error::StatementMismatch)?;
    if group.members.is_empty() {
        return Err(Error::StatementMismatch);
    }
    if let Some(expected) = group.fresh {
        if fresh.id != expected {
            return Err(Error::StatementMismatch);
        }
    }
    if fresh.main_rows != 0
        || fresh.mask_rows != shape.mask_rows()?
        || fresh.mask_code != shape.mask_code
    {
        return Err(Error::StatementMismatch);
    }
    if fresh.sampled || fresh.answers != 0 {
        return Err(Error::StatementMismatch);
    }
    if group.members.contains(&fresh.id) {
        return Err(Error::StatementMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::request::field::{Fp, MODULUS};
    use crate::pcs::request::params::LEAF_SALT_MIN_BYTES;
    use crate::pcs::request::runtime_tree::verify_mask_columns;
    use zkie_core::common::field::PrimeCharacteristicRing;
    use crate::primitives::transcript::{hash_labeled, tag_word, TEST_COINS};
    use zkie_core::types::Digest;

    /// A deterministic numbered source for tests only. It is compiled out of
    /// the normal build and provides no hiding.
    #[derive(Clone, Debug)]
    struct TestMaskCoins {
        seed: Digest,
        draw: u32,
    }

    impl TestMaskCoins {
        fn new(seed: Digest) -> Self {
            Self { seed, draw: 0 }
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
                let word =
                    u64::from_le_bytes(bytes.try_into().map_err(|_| Error::InvalidEncoding)?);
                *coefficient = Fp::from_u64(word % MODULUS);
            }
            Ok(Fp3::new(coefficients[0], coefficients[1], coefficients[2]))
        }
    }

    impl MaskCoins for TestMaskCoins {
        fn draw(&mut self) -> Result<(Fp3, u32)> {
            let number = self.draw;
            self.draw = self.draw.checked_add(1).ok_or(Error::ResourceLimit)?;
            Ok((self.value_at(number)?, number))
        }

        fn redraw(&self, number: u32) -> Result<Fp3> {
            self.value_at(number)
        }
    }

    fn shape() -> FoldShape {
        FoldShape {
            code: build_code_plan(4, 8, 1).unwrap(),
            mask_code: build_code_plan(3, 4, 1).unwrap(),
            k: 2,
            h: 1,
            mask_leaf_class: 3,
        }
    }

    fn coins() -> TestMaskCoins {
        TestMaskCoins::new(Digest([9; 32]))
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
        let mut mask = fold_mask_make(&shape, &mut source).unwrap();
        assert_eq!(mask.coefficients().len(), 5);
        assert_eq!(mask.draws(), &[0, 1, 2, 3, 4]);
        assert!(!mask.sent());
        // Without the encoding tail the check must not compare only one part.
        assert!(mask.tail().is_none());
        assert!(fold_mask_as_drawn(&shape, &mask, &source).is_err());

        fold_mask_make_tail(&shape, &mut mask, &mut source).unwrap();
        let tail = mask.tail().unwrap();
        assert_eq!(tail.rows, 2);
        assert_eq!(tail.len, shape.mask_code.r as usize);
        assert_eq!(tail.draws, vec![5, 6]);
        assert!(fold_mask_make_tail(&shape, &mut mask, &mut source).is_err());
        fold_mask_as_drawn(&shape, &mask, &source).unwrap();

        let mut tampered = FoldMask {
            coefficients: mask.coefficients().to_vec(),
            draws: mask.draws().to_vec(),
            tail: mask.tail().cloned(),
            id: None,
        };
        tampered.coefficients[0] = tampered.coefficients[0].add(&Fp3::ONE);
        assert!(fold_mask_as_drawn(&shape, &tampered, &source).is_err());

        let mut number = FoldMask {
            coefficients: mask.coefficients().to_vec(),
            draws: mask.draws().to_vec(),
            tail: mask.tail().cloned(),
            id: None,
        };
        number.draws[1] = 4;
        assert!(fold_mask_as_drawn(&shape, &number, &source).is_err());
        number.draws[1] = NO_DRAW;
        assert!(fold_mask_as_drawn(&shape, &number, &source).is_err());

        let mut tail_value = FoldMask {
            coefficients: mask.coefficients().to_vec(),
            draws: mask.draws().to_vec(),
            tail: mask.tail().cloned(),
            id: None,
        };
        tail_value.tail.as_mut().unwrap().values[0] =
            tail_value.tail.as_ref().unwrap().values[0].add(&Fp3::ONE);
        assert!(fold_mask_as_drawn(&shape, &tail_value, &source).is_err());

        let mut tail_number = FoldMask {
            coefficients: mask.coefficients().to_vec(),
            draws: mask.draws().to_vec(),
            tail: mask.tail().cloned(),
            id: None,
        };
        tail_number.tail.as_mut().unwrap().draws[1] = NO_DRAW;
        assert!(fold_mask_as_drawn(&shape, &tail_number, &source).is_err());

        let no_tail = FoldMask {
            coefficients: mask.coefficients().to_vec(),
            draws: mask.draws().to_vec(),
            tail: None,
            id: None,
        };
        assert!(fold_mask_as_drawn(&shape, &no_tail, &source).is_err());

        // A mask made for another shape fails the same check.
        let mut other_shape = shape;
        other_shape.k = 1;
        let mut other_mask = fold_mask_make(&other_shape, &mut source).unwrap();
        fold_mask_make_tail(&other_shape, &mut other_mask, &mut source).unwrap();
        assert!(fold_mask_as_drawn(&shape, &other_mask, &source).is_err());
        let mut taller = shape;
        taller.mask_code = build_code_plan(3, 5, 2).unwrap();
        let mut tall_mask = fold_mask_make(&taller, &mut source).unwrap();
        fold_mask_make_tail(&taller, &mut tall_mask, &mut source).unwrap();
        assert!(fold_mask_as_drawn(&shape, &tall_mask, &source).is_err());
    }

    #[test]
    fn mask_sum_matches_the_definition() {
        for k in 1..=3u32 {
            let w = values(k);
            let sum = fold_mask_sum(k, &w).unwrap();
            let mut brute = Fp3::ZERO;
            for bits in 0..(1u64 << k) {
                let mut total = Fp3::ZERO;
                for l in 0..k {
                    let last = l + 1 == k;
                    let x = if (bits >> l) & 1 == 1 {
                        Fp3::ONE
                    } else {
                        Fp3::ZERO
                    };
                    let mut weight = Fp3::ZERO;
                    for degree in if last { 0 } else { 1 }..=2 {
                        let power = x.pow_u64(degree as u64);
                        weight = weight.add(&w[hp_place(k, l, degree).unwrap()].mul(&power));
                    }
                    let mut tail = Fp3::ONE;
                    for j in (l + 1)..k {
                        let xj = if (bits >> j) & 1 == 1 {
                            Fp3::ONE
                        } else {
                            Fp3::ZERO
                        };
                        tail = tail.mul(&xj);
                    }
                    total = total.add(&weight.mul(&tail));
                }
                brute = brute.add(&total);
            }
            assert_eq!(sum, brute);
        }
    }

    fn tree_plan(shape: &FoldShape) -> MaskTreePlan {
        MaskTreePlan {
            tree_number: 5,
            leaf_class: shape.mask_leaf_class,
            log_n: shape.mask_code.log_n,
            mask_rows: shape.mask_rows().unwrap(),
            salt_bytes: LEAF_SALT_MIN_BYTES,
            run: Digest([2; 32]),
            configuration: Digest([7; 32]),
        }
    }

    fn tree_salts(plan: &MaskTreePlan) -> Vec<Vec<u8>> {
        (0..plan.columns() as usize)
            .map(|position| vec![(position as u8).wrapping_add(1); plan.salt_bytes])
            .collect()
    }

    #[test]
    fn fold_start_separates_prover_and_verifier() {
        let shape = shape();
        let mut source = coins();
        let mut mask = fold_mask_make(&shape, &mut source).unwrap();
        fold_mask_make_tail(&shape, &mut mask, &mut source).unwrap();
        let groups = VerifiedGroups {
            groups: vec![VerifiedGroup {
                group: 1,
                members: vec![1, 2],
                fresh: Some(9),
            }],
        };
        let plan = tree_plan(&shape);
        let salts = tree_salts(&plan);
        let notice = fold_mask_notice(&shape).unwrap();
        let oracle = MaskOracle::from_notice(9, 1, &notice);
        // The mask must have been sent under the id the verifier checks.
        assert!(fold_start_prove(
            &shape, &mask, &source, &oracle, &groups, plan, &salts
        )
        .is_err());
        mask.sent_as(9).unwrap();
        let start = fold_start_prove(
            &shape, &mask, &source, &oracle, &groups, plan, &salts,
        )
        .unwrap();
        assert_ne!(start.message.root, Digest::ZERO);
        assert_eq!(
            start.message.sum,
            fold_mask_sum(shape.k, mask.coefficients()).unwrap()
        );
        assert_eq!(start.draft.table.rows, 2);
        assert_eq!(start.draft.tail.draws, mask.tail().unwrap().draws);
        // The verifier accepts the public message with no prover secret.
        let statement = VerifiedStatement::new(groups.clone(), vec![plan]).unwrap();
        fold_start_verify(&shape, &start.message, &statement).unwrap();
        assert!(fold_start_verify(&shape, &start.message, &statement).is_err());
        let mut bad = start.message.clone();
        bad.tree_number = 6;
        assert!(fold_start_verify(&shape, &bad, &statement).is_err());
        // The committed root opens the mask tree at the drawn positions.
        let positions = [1u64, 4];
        let openings = start.tree.open(&positions).unwrap();
        verify_mask_columns(&plan, &start.message.root, &positions, &openings).unwrap();
        let mut tampered = openings.clone();
        tampered[0].symbols[0] = tampered[0].symbols[0] + Fp::ONE;
        assert!(
            verify_mask_columns(&plan, &start.message.root, &positions, &tampered).is_err()
        );
        // A different claimed group, a changed notice or a different tree
        // layout is rejected before the spend.
        let mut bad = start.message.clone();
        bad.claimed_group = 7;
        assert!(fold_start_verify(&shape, &bad, &statement).is_err());
        let mut bad = start.message.clone();
        bad.notice.mask_rows = 4;
        assert!(fold_start_verify(&shape, &bad, &statement).is_err());
        // A tampered prover coefficient fails the self-check.
        let mut tampered_mask = FoldMask {
            coefficients: mask.coefficients().to_vec(),
            draws: mask.draws().to_vec(),
            tail: mask.tail().cloned(),
            id: Some(9),
        };
        tampered_mask.coefficients[0] = tampered_mask.coefficients[0].add(&Fp3::ONE);
        assert!(fold_start_prove(
            &shape,
            &tampered_mask,
            &source,
            &oracle,
            &groups,
            plan,
            &salts
        )
        .is_err());
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
        // No two members may write the same row.
        let one = MaskTable::new(1, 2, vec![Fp3::from_u64(5), Fp3::from_u64(6)]).unwrap();
        assert!(fold_gather_mask(
            1,
            2,
            &[
                MaskPlacement { first: 0, table: one.clone() },
                MaskPlacement { first: 0, table: one.clone() },
            ]
        )
        .is_err());
        // A member's first row is aligned to its own row count.
        let two = MaskTable::new(
            2,
            1,
            vec![Fp3::from_u64(7), Fp3::from_u64(8)],
        )
        .unwrap();
        assert!(fold_gather_mask(
            2,
            1,
            &[MaskPlacement { first: 1, table: two.clone() }]
        )
        .is_err());
        assert!(fold_gather_mask(
            2,
            1,
            &[MaskPlacement { first: 4, table: two.clone() }]
        )
        .is_err());
        // A member's row count is a power of two.
        let three = MaskTable::new(
            3,
            1,
            vec![Fp3::from_u64(9), Fp3::from_u64(10), Fp3::from_u64(11)],
        )
        .unwrap();
        assert!(fold_gather_mask(2, 1, &[MaskPlacement { first: 0, table: three }]).is_err());
        // The shift itself is bounded; a zero-width layout is a legal empty
        // tail and keeps the group's rows.
        assert!(fold_gather_mask(40, 2, &[]).is_err());
        let empty = fold_gather_mask(1, 0, &[]).unwrap();
        assert_eq!(empty.rows, 2);
        assert_eq!(empty.len, 0);
        assert!(empty.values.is_empty());
    }

    #[test]
    fn notice_draft_and_check() {
        let shape = shape();
        let mut source = coins();
        let mut mask = fold_mask_make(&shape, &mut source).unwrap();
        assert!(fold_mask_draft(&shape, &mask).is_err());
        fold_mask_make_tail(&shape, &mut mask, &mut source).unwrap();
        let draft = fold_mask_draft(&shape, &mask).unwrap();
        assert_eq!(draft.notice.mask_rows, 2);
        assert_eq!(draft.notice.mask_code, shape.mask_code);
        assert_eq!(draft.notice.answers, 0);
        assert_eq!(draft.table.rows, 2);
        assert_eq!(draft.table.len, shape.mask_code.ell as usize);
        assert_eq!(draft.tail.rows, 2);
        assert_eq!(draft.tail.len, shape.mask_code.r as usize);
        assert_eq!(
            draft.table.values,
            fold_mask_table(shape.k, shape.h, shape.mask_code.ell as usize, mask.coefficients())
                .unwrap()
        );

        // A mask made for another legal shape cannot be drafted here: the
        // tail's rows/r and the coefficient count are checked against the
        // current shape before the draft is built.
        let mut other_shape = shape;
        other_shape.mask_code = build_code_plan(3, 5, 2).unwrap();
        let mut other_mask = fold_mask_make(&other_shape, &mut source).unwrap();
        fold_mask_make_tail(&other_shape, &mut other_mask, &mut source).unwrap();
        assert!(fold_mask_draft(&shape, &other_mask).is_err());
        let mut narrow_shape = shape;
        narrow_shape.k = 1;
        let mut narrow_mask = fold_mask_make(&narrow_shape, &mut source).unwrap();
        fold_mask_make_tail(&narrow_shape, &mut narrow_mask, &mut source).unwrap();
        assert!(fold_mask_draft(&shape, &narrow_mask).is_err());

        let groups = VerifiedGroups {
            groups: vec![VerifiedGroup {
                group: 1,
                members: vec![1, 2],
                fresh: None,
            }],
        };
        let good = MaskOracle {
            id: 9,
            claimed_group: 1,
            main_rows: 0,
            mask_rows: 2,
            mask_code: shape.mask_code,
            answers: 0,
            sampled: false,
        };
        fold_mask_check(&shape, &good, &groups).unwrap();
        let mut bad = good;
        bad.main_rows = 1;
        assert!(fold_mask_check(&shape, &bad, &groups).is_err());
        let mut bad = good;
        bad.mask_rows = 4;
        assert!(fold_mask_check(&shape, &bad, &groups).is_err());
        let mut bad = good;
        bad.mask_code = shape.code;
        assert!(fold_mask_check(&shape, &bad, &groups).is_err());
        let mut bad = good;
        bad.sampled = true;
        assert!(fold_mask_check(&shape, &bad, &groups).is_err());
        let mut bad = good;
        bad.answers = 1;
        assert!(fold_mask_check(&shape, &bad, &groups).is_err());
        // The claimed group is only a lookup key: a group the verifier's
        // registry does not hold is rejected.
        let mut bad = good;
        bad.claimed_group = 7;
        assert!(fold_mask_check(&shape, &bad, &groups).is_err());
        // The members come from the registry, not from the notice. A fresh id
        // listed as a member of the registry's group is rejected even though
        // the notice itself carries no member list.
        let member_groups = VerifiedGroups {
            groups: vec![VerifiedGroup {
                group: 1,
                members: vec![9, 1],
                fresh: None,
            }],
        };
        assert!(fold_mask_check(&shape, &good, &member_groups).is_err());
        // A fresh id the plan fixes must match.
        let fixed = VerifiedGroups {
            groups: vec![VerifiedGroup {
                group: 1,
                members: vec![1, 2],
                fresh: Some(9),
            }],
        };
        fold_mask_check(&shape, &good, &fixed).unwrap();
        let mut other = good;
        other.id = 8;
        assert!(fold_mask_check(&shape, &other, &fixed).is_err());
        // A forged registry that omits the fresh id passes this content check.
        // The recorded limitation is the registry itself: the statement/plan
        // must be independently verified before the registry is used, and
        // FOLDEQ must bind the group, fresh oracle and layout to that
        // statement. The notice cannot forge the members.
        let forged_clean = VerifiedGroups {
            groups: vec![VerifiedGroup {
                group: 1,
                members: vec![1, 2],
                fresh: None,
            }],
        };
        assert!(fold_mask_check(&shape, &good, &forged_clean).is_ok());
        // A group with no members is not a group.
        let empty = VerifiedGroups {
            groups: vec![VerifiedGroup {
                group: 1,
                members: Vec::new(),
                fresh: None,
            }],
        };
        assert!(fold_mask_check(&shape, &good, &empty).is_err());
    }

    #[test]
    fn public_table_functions_bound_their_shifts() {
        let wide = values(64);
        assert!(fold_mask_table(64, 64, 65, &wide).is_err());
        let weights: Vec<Fp3> = (0..64).map(|index| Fp3::from_u64(index + 1)).collect();
        assert!(fold_mask_weights(64, 64, 65, &weights).is_err());
        assert!(fold_mask_table(0, 0, 1, &[]).is_err());
        assert!(fold_mask_weights(0, 0, 1, &[]).is_err());
        let base = shape();
        let mut too_high = base;
        too_high.k = 64;
        too_high.h = 64;
        assert!(fold_mask_shape_fault(&too_high).is_some());
        assert!(fold_mask_notice(&too_high).is_err());
    }

    #[test]
    fn zero_length_tail_is_supported() {
        let mut shape = shape();
        shape.mask_code = build_code_plan(3, 4, 0).unwrap();
        let mut source = coins();
        let mut mask = fold_mask_make(&shape, &mut source).unwrap();
        fold_mask_make_tail(&shape, &mut mask, &mut source).unwrap();
        let tail = mask.tail().unwrap();
        assert_eq!(tail.rows, 2);
        assert_eq!(tail.len, 0);
        assert!(tail.values.is_empty());
        assert!(tail.draws.is_empty());
        assert!(tail.at(0, 0).is_err());
        fold_mask_as_drawn(&shape, &mask, &source).unwrap();
        let draft = fold_mask_draft(&shape, &mask).unwrap();
        assert_eq!(draft.tail.len, 0);
        let empty = MaskTable::new(1, 0, Vec::new()).unwrap();
        let group = fold_gather_mask(1, 0, &[MaskPlacement { first: 0, table: empty }]).unwrap();
        assert_eq!(group.rows, 2);
        assert_eq!(group.len, 0);
        assert!(group.values.is_empty());
    }

    #[test]
    fn malformed_structures_are_rejected_without_panicking() {
        let short_table = MaskTable {
            rows: 2,
            len: 2,
            values: vec![Fp3::from_u64(1)],
        };
        assert!(short_table.fault().is_some());
        assert!(short_table.at(1, 0).is_err());
        assert!(fold_gather_mask(
            1,
            2,
            &[MaskPlacement {
                first: 0,
                table: short_table.clone(),
            }]
        )
        .is_err());
        assert!(MaskTable::new(0, 2, Vec::new()).is_err());
        let empty = MaskTable::new(2, 0, Vec::new()).unwrap();
        assert!(empty.fault().is_none());
        assert!(empty.at(0, 0).is_err());

        let short_tail = MaskTail {
            rows: 2,
            len: 1,
            values: vec![Fp3::from_u64(1)],
            draws: vec![0],
        };
        assert!(short_tail.fault().is_some());
        assert!(short_tail.at(1, 0).is_err());
        assert!(MaskTail::new(0, 0, Vec::new(), Vec::new()).is_err());

        let shape = shape();
        let mut source = coins();
        let mut mask = fold_mask_make(&shape, &mut source).unwrap();
        fold_mask_make_tail(&shape, &mut mask, &mut source).unwrap();
        let broken = FoldMask {
            coefficients: mask.coefficients().to_vec(),
            draws: mask.draws().to_vec(),
            tail: Some(short_tail),
            id: None,
        };
        assert!(fold_mask_as_drawn(&shape, &broken, &source).is_err());
    }

    #[test]
    fn mask_rows_rejects_unvalidated_shifts() {
        let base = shape();
        let mut h64 = base;
        h64.h = 64;
        h64.k = 64;
        assert!(h64.mask_rows().is_err());
        assert!(fold_mask_shape_fault(&h64).is_some());
        assert!(fold_mask_notice(&h64).is_err());
        let mut source = coins();
        let mut mask = fold_mask_make(&base, &mut source).unwrap();
        assert!(fold_mask_make_tail(&h64, &mut mask, &mut source).is_err());
        assert!(fold_mask_as_drawn(&h64, &mask, &source).is_err());
        // h=63 still shifts, but the shape bound rejects it before use.
        let mut h63 = base;
        h63.h = 63;
        h63.k = 63;
        assert!(h63.mask_rows().is_ok());
        assert!(fold_mask_shape_fault(&h63).is_some());
        assert!(fold_mask_notice(&h63).is_err());
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
