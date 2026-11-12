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

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::chain::{Chain, KIND_ALGEBRAIC, KIND_EPSILON};
use crate::pcs::request::code::{build_code_plan, CodeParams};
use crate::pcs::request::field::{E_DEGREE, Fp3};
use crate::pcs::request::runtime_tree::{
    verify_mask_opening, MaskTree, MaskTreePlan, RuntimeTreeOpening,
};
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

/// One member of the fixed FOLD group layout. `main_rows` is the member's
/// real main-row count and decides the length of a kept coefficient form;
/// `kept` is the layout mark that permits that member to appear as a kept
/// form. `mask_rows` is used by the isolated mask-block space check, not only
/// as identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoldMemberLayout {
    pub oracle: u64,
    pub main_rows: u64,
    pub mask_rows: u64,
    pub kept: bool,
}

/// The fixed FOLD group layout: the group and row variables plus one entry per
/// member, ordered by oracle number. It is a plan-provided trusted input in
/// this isolated test layer; its production source is not authenticated, and a
/// proof cannot supply or change it. The checks below are only candidate
/// self-consistency checks, not a reference layout derivation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoldGroupLayout {
    pub group: u64,
    pub k: u32,
    pub h: u32,
    pub members: Vec<FoldMemberLayout>,
}

impl FoldGroupLayout {
    /// An isolated candidate self-consistency check, not a reference layout
    /// algorithm. It only checks, in the layout's oracle order, that every
    /// member has at least one main or mask row; a kept member has main rows;
    /// non-kept main and mask blocks are locally aligned by the next
    /// power-of-two boundary and fit the supplied `2^k`/`2^h` spaces; and the
    /// kept-row sum is bounded. It does not prove the reference block order,
    /// offsets, minimal k/h or the real oracle kept/main_rows/mask_rows values.
    /// Production must rebuild and compare the complete layout from
    /// authenticated statement/oracle shapes and join kept marks, not from this
    /// plan-provided test fixture or a proof row. No new production layout
    /// protocol is defined here.
    pub fn fault(&self, shape: &FoldShape) -> Option<&'static str> {
        if self.k != shape.k || self.h != shape.h || self.k >= 64 || self.h >= 64 {
            return Some("the FOLD layout does not match the fold shape");
        }
        if self.members.is_empty() {
            return Some("the FOLD layout has no members");
        }
        let main_space = 1u64 << self.k;
        let mask_space = 1u64 << self.h;
        let mut main_first = 0u64;
        let mut mask_first = 0u64;
        for (index, member) in self.members.iter().enumerate() {
            if index > 0 && self.members[index - 1].oracle >= member.oracle {
                return Some("the FOLD layout members are not strictly ordered");
            }
            if member.main_rows == 0 && member.mask_rows == 0 {
                return Some("a FOLD member is an empty oracle");
            }
            if member.kept && member.main_rows == 0 {
                return Some("a kept FOLD member has no main rows");
            }
            if !member.kept && member.main_rows > 0 {
                let block = match checked_block(member.main_rows) {
                    Some(block) => block,
                    None => return Some("the FOLD main block size overflows"),
                };
                let start = match aligned_start(main_first, block) {
                    Some(start) => start,
                    None => return Some("the FOLD main block start overflows"),
                };
                let end = match start.checked_add(member.main_rows) {
                    Some(end) => end,
                    None => return Some("the FOLD main block end overflows"),
                };
                if end > main_space {
                    return Some("the FOLD main rows do not fit the group space");
                }
                main_first = end;
            }
            if member.mask_rows > 0 {
                let block = match checked_block(member.mask_rows) {
                    Some(block) => block,
                    None => return Some("the FOLD mask block size overflows"),
                };
                let start = match aligned_start(mask_first, block) {
                    Some(start) => start,
                    None => return Some("the FOLD mask block start overflows"),
                };
                let end = match start.checked_add(member.mask_rows) {
                    Some(end) => end,
                    None => return Some("the FOLD mask block end overflows"),
                };
                if end > mask_space {
                    return Some("the FOLD mask rows do not fit the group space");
                }
                mask_first = end;
            }
        }
        if self.kept_rows_checked().is_err() {
            return Some("the FOLD kept-row count overflows");
        }
        None
    }

    pub fn member(&self, oracle: u64) -> Option<&FoldMemberLayout> {
        self.members.iter().find(|member| member.oracle == oracle)
    }

    /// The sum of kept main rows, checked at the only place that synthesizes
    /// the output count. No unchecked sum is exposed.
    pub fn kept_rows_checked(&self) -> Result<u64> {
        let mut sum = 0u64;
        for member in &self.members {
            if member.kept {
                sum = sum
                    .checked_add(member.main_rows)
                    .ok_or(Error::ResourceLimit)?;
            }
        }
        Ok(sum)
    }
}

fn checked_block(rows: u64) -> Option<u64> {
    if rows == 0 {
        return None;
    }
    let bits = 64 - (rows - 1).leading_zeros();
    1u64.checked_shl(bits)
}

/// Which row block of a fold layout a member's rows occupy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldPart {
    Main,
    Mask,
}

/// The first group row of every member's main or mask block, in layout order.
/// The derivation is the layout's own aligned-block order, not a
/// proof-supplied offset. `None` means the member has no block in that part
/// (a kept main member is not placed in the main block). The result is an
/// internal placement view; it does not authenticate the layout's source.
pub fn fold_member_firsts(
    layout: &FoldGroupLayout,
    part: FoldPart,
) -> Result<Vec<Option<u64>>> {
    let mut cursor = 0u64;
    let mut firsts = Vec::with_capacity(layout.members.len());
    for member in &layout.members {
        let rows = match part {
            FoldPart::Main => member.main_rows,
            FoldPart::Mask => member.mask_rows,
        };
        if (part == FoldPart::Main && member.kept) || rows == 0 {
            firsts.push(None);
            continue;
        }
        let block = checked_block(rows).ok_or(Error::ResourceLimit)?;
        let start = aligned_start(cursor, block).ok_or(Error::ResourceLimit)?;
        firsts.push(Some(start));
        // A member occupies its complete power-of-two block; the next member
        // starts after that block, matching the BAT group layout.
        cursor = start.checked_add(block).ok_or(Error::ResourceLimit)?;
    }
    Ok(firsts)
}

fn aligned_start(first: u64, block: u64) -> Option<u64> {
    if first == 0 {
        return Some(0);
    }
    let rest = first % block;
    if rest == 0 {
        return Some(first);
    }
    first.checked_add(block - rest)
}

/// One kept main row of the fixed group: the member identity and the compact
/// coefficient vector on that row. Kept rows are not folded; their coefficients
/// are scaled by `eps * (1 - g_1) .. (1 - g_k)` in the output. The isolated
/// representation stores only real rows, row-major as `row * ell + place`, with
/// length `main_rows * ell`. The virtual multilinear space has
/// `2^ceil_log2(main_rows)` rows at column-major index `row + 2^row_bits * place`;
/// virtual padding rows (row >= main_rows) are fixed zero and are not stored.
/// The same compact order is used by the prover dot product and the output
/// scaling. The layout and form source are plan-provided test inputs, not
/// production-authenticated values; a proof's self-report is never trusted.
/// There is no virtual-table-to-next-consumer entry yet, so the virtual
/// expansion is evidence-only and the later consumer remains unconnected.
/// Production must derive both sources from authenticated statement/oracle
/// shapes and join kept marks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldKeptForm {
    pub oracle: u64,
    pub coefficients: Vec<Fp3>,
}

impl FoldKeptForm {
    pub fn check(&self, shape: &FoldShape, layout: &FoldGroupLayout) -> Result<()> {
        let member = layout.member(self.oracle).ok_or(Error::ShapeMismatch)?;
        if !member.kept {
            return Err(Error::StatementMismatch);
        }
        let expected = member
            .main_rows
            .checked_mul(shape.code.ell)
            .ok_or(Error::ResourceLimit)?;
        if usize::try_from(expected).map_err(|_| Error::ResourceLimit)? != self.coefficients.len() {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }
}

/// An isolated FOLD input claim on one fixed group. The group/layout and the
/// coefficient vectors are an unauthenticated fixture in this step: the BAT/
/// relation layer that would produce them from sampled rows is not connected.
/// The row binds the group number, the layout's row variables and member ids,
/// the main and mask coefficients of the running claim, and its target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldInputRow {
    pub group: u64,
    pub k: u32,
    pub h: u32,
    pub members: Vec<u64>,
    pub coefficients: Vec<Fp3>,
    pub mask_coefficients: Vec<Fp3>,
    pub kept: Vec<FoldKeptForm>,
    pub target: Fp3,
}

impl FoldInputRow {
    /// The fixture must have the shape's layout and table lengths. The member
    /// list is strictly increasing and non-empty; the session separately checks
    /// it against the trusted group view.
    pub fn check(&self, shape: &FoldShape, layout: &FoldGroupLayout) -> Result<()> {
        validate_fold_mask_shape(shape)?;
        if layout.fault(shape).is_some() {
            return Err(Error::InvalidConfig);
        }
        if self.group != layout.group
            || self.k != layout.k
            || self.h != layout.h
            || self.members.is_empty()
        {
            return Err(Error::ShapeMismatch);
        }
        if self.members.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(Error::InvalidConfig);
        }
        if self
            .members
            .iter()
            .copied()
            .ne(layout.members.iter().map(|member| member.oracle))
        {
            return Err(Error::StatementMismatch);
        }
        let rows = 1usize.checked_shl(shape.k).ok_or(Error::ResourceLimit)?;
        let mask_rows = 1usize.checked_shl(shape.h).ok_or(Error::ResourceLimit)?;
        let main_len = rows
            .checked_mul(usize::try_from(shape.code.ell).map_err(|_| Error::ResourceLimit)?)
            .ok_or(Error::ResourceLimit)?;
        let mask_len = mask_rows
            .checked_mul(
                usize::try_from(shape.mask_code.ell).map_err(|_| Error::ResourceLimit)?,
            )
            .ok_or(Error::ResourceLimit)?;
        if self.coefficients.len() != main_len || self.mask_coefficients.len() != mask_len {
            return Err(Error::ShapeMismatch);
        }
        let mut kept_oracles = Vec::with_capacity(self.kept.len());
        for kept in &self.kept {
            kept.check(shape, layout)?;
            if kept_oracles.contains(&kept.oracle) {
                return Err(Error::InvalidConfig);
            }
            kept_oracles.push(kept.oracle);
        }
        Ok(())
    }
}

/// The final constraint after the FOLD rounds: the folded main coefficients,
/// the old mask's coefficients, the fresh mask's coefficients, and the new
/// target, together with the group/layout, challenges, fresh-mask id/root and
/// tree identity needed by the next stage. The next IOPP/CS stage is not
/// connected, so [`FoldOutput::next_stage`] always fails closed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldOutput {
    pub code: CodeParams,
    pub mask_code: CodeParams,
    pub group: u64,
    pub k: u32,
    pub h: u32,
    pub members: Vec<u64>,
    pub gamma: Vec<Fp3>,
    pub eps: Fp3,
    pub sum: Fp3,
    pub fresh: u64,
    pub fresh_root: Digest,
    pub tree_number: u64,
    pub coefficients: Vec<Fp3>,
    pub mask_coefficients: Vec<Fp3>,
    pub fresh_coefficients: Vec<Fp3>,
    pub kept: Vec<FoldKeptForm>,
    pub kept_factor: Fp3,
    pub layout: FoldGroupLayout,
    pub kept_rows: u64,
    pub target: Fp3,
}

impl FoldOutput {
    /// The next protocol stage (IOPP/CS) is not implemented in this isolated
    /// layer. Returning an error here makes a caller fail closed instead of
    /// treating the FOLD output as an accepted proof.
    pub fn next_stage(&self) -> Result<()> {
        Err(Error::InvalidSessionState)
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

/// One variable's coefficient of the hiding polynomial, with the constant
/// fixed to zero for every variable but the last.
fn round_coefficient(w: &[Fp3], k: u32, variable: u32, degree: u32) -> Result<Fp3> {
    let last = variable + 1 == k;
    if !last && degree == 0 {
        return Ok(Fp3::ZERO);
    }
    Ok(w[hp_place(k, variable, degree)?])
}

/// phi_l(w_l) for the plain sum over {0,1}: 2*w(0) + w(1) + w(2).
fn round_apply(w: &[Fp3], k: u32, variable: u32) -> Result<Fp3> {
    let two = Fp3::from_u64(2);
    Ok(round_coefficient(w, k, variable, 0)?
        .mul(&two)
        .add(&round_coefficient(w, k, variable, 1)?)
        .add(&round_coefficient(w, k, variable, 2)?))
}

/// A small main-row table for the round arithmetic: coefficient and message
/// rows in column-major order, folded once per variable. It is a test-sized
/// stand-in for the actual oracle rows, which a production prover reads
/// through its engine; its entries must come from that table, not from a free
/// choice of polynomial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldMainTable {
    pub coefficients: Vec<Fp3>,
    pub messages: Vec<Fp3>,
}

impl FoldMainTable {
    pub fn new(coefficients: Vec<Fp3>, messages: Vec<Fp3>) -> Result<Self> {
        if coefficients.is_empty() || coefficients.len() != messages.len() {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            coefficients,
            messages,
        })
    }

    fn fold_low_once(&mut self, gamma: Fp3) -> Result<()> {
        if self.coefficients.len() % 2 != 0 {
            return Err(Error::InvalidConfig);
        }
        let rows = self.coefficients.len() / 2;
        for index in 0..rows {
            self.coefficients[index] = fold_pair(
                self.coefficients[2 * index],
                self.coefficients[2 * index + 1],
                gamma,
            );
            self.messages[index] =
                fold_pair(self.messages[2 * index], self.messages[2 * index + 1], gamma);
        }
        self.coefficients.truncate(rows);
        self.messages.truncate(rows);
        Ok(())
    }

    fn pair_products(&self) -> Result<[Fp3; 3]> {
        pair_products(&self.coefficients, &self.messages)
    }
}

fn fold_pair(low: Fp3, high: Fp3, gamma: Fp3) -> Fp3 {
    low.add(&gamma.mul(&high.sub(&low)))
}

/// Outputs are folded in parallel from this row count upward.
const FOLD_PARALLEL_MIN_ROWS: usize = 1 << 16;
/// Number of independent outputs folded between scheduling barriers.
const FOLD_PARALLEL_BLOCK: usize = 1 << 18;
/// Pair products run in parallel from this pair count upward.
const PAIR_PARALLEL_MIN_PAIRS: usize = 1 << 15;
/// Pairs per parallel partial sum.
const PAIR_PARALLEL_CHUNK: usize = 1 << 15;

fn fold_low_once(values: &mut Vec<Fp3>, gamma: Fp3) -> Result<()> {
    fold_low_once_with_min(values, gamma, FOLD_PARALLEL_MIN_ROWS)
}

fn fold_low_once_with_min(
    values: &mut Vec<Fp3>,
    gamma: Fp3,
    parallel_min_rows: usize,
) -> Result<()> {
    if values.len() % 2 != 0 {
        return Err(Error::InvalidConfig);
    }
    let rows = values.len() / 2;
    if rows >= parallel_min_rows {
        fold_low_once_bounded(values, rows, gamma);
    } else {
        for index in 0..rows {
            values[index] = fold_pair(values[2 * index], values[2 * index + 1], gamma);
        }
    }
    values.truncate(rows);
    Ok(())
}

/// Folds every pair of `values` into its first `rows` entries. Blocks run in
/// ascending order with a barrier between them, so a block only reads source
/// entries that no earlier block has overwritten; inside a block the exact
/// independent outputs are folded in bounded parallel chunks.
fn fold_low_once_bounded(values: &mut [Fp3], rows: usize, gamma: Fp3) {
    let mut start = 0usize;
    while start < rows {
        let end = (start + FOLD_PARALLEL_BLOCK).min(rows);
        if 2 * start < end {
            // This block reads its own output range; copy the exact source
            // entries before they are overwritten.
            let scratch = values[2 * start..2 * end].to_vec();
            values[start..end]
                .par_iter_mut()
                .enumerate()
                .for_each(|(offset, slot)| {
                    *slot = fold_pair(scratch[2 * offset], scratch[2 * offset + 1], gamma);
                });
        } else {
            // The source range lies above this block, so splitting at `end`
            // gives disjoint read and write slices.
            let (low, high) = values.split_at_mut(end);
            let write = &mut low[start..end];
            let read = &high[(2 * start - end)..(2 * end - end)];
            write
                .par_iter_mut()
                .enumerate()
                .for_each(|(offset, slot)| {
                    *slot = fold_pair(read[2 * offset], read[2 * offset + 1], gamma);
                });
        }
        start = end;
    }
}

fn pair_products(a: &[Fp3], m: &[Fp3]) -> Result<[Fp3; 3]> {
    if a.len() != m.len() || a.len() % 2 != 0 {
        return Err(Error::InvalidConfig);
    }
    let pairs = a.len() / 2;
    if pairs < PAIR_PARALLEL_MIN_PAIRS {
        return Ok(pair_products_range(a, m));
    }
    // Every pair contributes an independent exact product; the chunk sums are
    // collected in order so the final accumulation is reproducible.
    let partials: Vec<[Fp3; 3]> = a
        .par_chunks(2 * PAIR_PARALLEL_CHUNK)
        .zip(m.par_chunks(2 * PAIR_PARALLEL_CHUNK))
        .map(|(left, right)| pair_products_range(left, right))
        .collect();
    let mut q = [Fp3::ZERO; 3];
    for partial in partials {
        q[0] = q[0].add(&partial[0]);
        q[1] = q[1].add(&partial[1]);
        q[2] = q[2].add(&partial[2]);
    }
    Ok(q)
}

fn pair_products_range(a: &[Fp3], m: &[Fp3]) -> [Fp3; 3] {
    debug_assert_eq!(a.len(), m.len());
    debug_assert_eq!(a.len() % 2, 0);
    let mut q = [Fp3::ZERO; 3];
    for index in 0..a.len() / 2 {
        let a0 = a[2 * index];
        let a1 = a[2 * index + 1].sub(&a0);
        let m0 = m[2 * index];
        let m1 = m[2 * index + 1].sub(&m0);
        q[0] = q[0].add(&a0.mul(&m0));
        q[1] = q[1].add(&a0.mul(&m1).add(&a1.mul(&m0)));
        q[2] = q[2].add(&a1.mul(&m1));
    }
    q
}

fn inner(a: &[Fp3], b: &[Fp3]) -> Result<Fp3> {
    if a.len() != b.len() {
        return Err(Error::InvalidConfig);
    }
    let mut sum = Fp3::ZERO;
    for (left, right) in a.iter().zip(b) {
        sum = sum.add(&left.mul(right));
    }
    Ok(sum)
}

/// What is left of the mask rows when the rounds end: the old mask's
/// coefficients (scaled), the folded old mask message and the folded fresh
/// mask message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldRoundEnd {
    pub d: Vec<Fp3>,
    pub mu: Vec<Fp3>,
    pub fresh: Vec<Fp3>,
}

/// The prover's sumcheck rounds of the hiding polynomial U with FOLD's mask
/// and kept terms. The mask term is rebuilt here from the old mask rows
/// (`d`, `mu`), the fresh mask made from the coefficients, and `later`; the
/// main term is the pair product of the actual main coefficient and message
/// rows. This is the mathematical round shell: the oracle tables, group
/// layout, statement and production channel are not wired.
#[derive(Clone, Debug)]
pub struct FoldRoundProver {
    k: u32,
    h: u32,
    coefficients: Vec<Fp3>,
    after: Vec<Fp3>,
    before: Fp3,
    next: u32,
    sum: Fp3,
    eps: Option<Fp3>,
    kept: Fp3,
    later: Fp3,
    d: Vec<Fp3>,
    mu: Vec<Fp3>,
    fresh: Vec<Fp3>,
    main: FoldMainTable,
    folded_inner: Option<Fp3>,
}

impl FoldRoundProver {
    pub fn new(
        k: u32,
        h: u32,
        coefficients: Vec<Fp3>,
        d: Vec<Fp3>,
        mu: Vec<Fp3>,
        len_zk: usize,
        main: FoldMainTable,
        kept: Fp3,
    ) -> Result<Self> {
        if k < 1 || h > k || coefficients.len() != 2 * k as usize + 1 {
            return Err(Error::InvalidConfig);
        }
        let mask_entries = (1usize << h)
            .checked_mul(len_zk)
            .ok_or(Error::ResourceLimit)?;
        if d.len() != mask_entries || mu.len() != mask_entries {
            return Err(Error::InvalidConfig);
        }
        if main.coefficients.len() % (1usize << k) != 0 {
            return Err(Error::InvalidConfig);
        }
        let fresh = fold_mask_table(k, h, len_zk, &coefficients)?;
        let two = Fp3::from_u64(2);
        let mut after = vec![Fp3::ZERO; k as usize];
        for variable in (0..(k as usize - 1)).rev() {
            let own = round_apply(&coefficients, k, variable as u32 + 1)?;
            after[variable] = own.add(&two.mul(&after[variable + 1]));
        }
        let sum = round_apply(&coefficients, k, 0)?.add(&two.mul(&after[0]));
        let folded_inner = if h == 0 {
            Some(inner(&d, &mu)?)
        } else {
            None
        };
        Ok(Self {
            k,
            h,
            coefficients,
            after,
            before: Fp3::ZERO,
            next: 0,
            sum,
            eps: None,
            kept,
            later: Fp3::ONE,
            d,
            mu,
            fresh,
            main,
            folded_inner,
        })
    }

    pub fn sum(&self) -> Fp3 {
        self.sum
    }

    pub fn start(&mut self, eps: Fp3) -> Result<()> {
        if self.eps.is_some() {
            return Err(Error::InvalidSessionState);
        }
        self.eps = Some(eps);
        Ok(())
    }

    fn u_round(&self) -> Result<[Fp3; 3]> {
        if self.next >= self.k {
            return Err(Error::InvalidSessionState);
        }
        let variable = self.next;
        let mut q = [
            round_coefficient(&self.coefficients, self.k, variable, 0)?,
            round_coefficient(&self.coefficients, self.k, variable, 1)?,
            round_coefficient(&self.coefficients, self.k, variable, 2)?,
        ];
        q[0] = q[0].add(&self.after[variable as usize]);
        q[1] = q[1].add(&self.before);
        Ok(q)
    }

    /// The round polynomial: U's round plus eps times the main and mask
    /// terms, with the kept-row term. `eps` must have been started.
    pub fn round(&self) -> Result<[Fp3; 3]> {
        let eps = self.eps.ok_or(Error::InvalidSessionState)?;
        let main = self.main.pair_products()?;
        let mask = if self.next < self.h {
            pair_products(&self.d, &self.mu)?
        } else {
            let folded_inner = self
                .folded_inner
                .ok_or(Error::InvalidSessionState)?;
            [Fp3::ZERO, folded_inner.mul(&self.later), Fp3::ZERO]
        };
        let mut q = self.u_round()?;
        for (coefficient, (main, mask)) in q.iter_mut().zip(main.iter().zip(&mask)) {
            *coefficient = coefficient.add(&eps.mul(&main.add(mask)));
        }
        let kept = eps.mul(&self.kept);
        q[0] = q[0].add(&kept);
        q[1] = q[1].sub(&kept);
        Ok(q)
    }

    /// Bind one challenge and advance to the next variable.
    pub fn bind(&mut self, gamma: Fp3) -> Result<()> {
        if self.eps.is_none() || self.next >= self.k {
            return Err(Error::InvalidSessionState);
        }
        let variable = self.next;
        let value = round_coefficient(&self.coefficients, self.k, variable, 0)?
            .add(&round_coefficient(&self.coefficients, self.k, variable, 1)?.mul(&gamma))
            .add(
                &round_coefficient(&self.coefficients, self.k, variable, 2)?
                    .mul(&gamma.mul(&gamma)),
            );
        self.before = self.before.mul(&gamma).add(&value);
        self.main.fold_low_once(gamma)?;
        if variable < self.h {
            fold_low_once(&mut self.d, gamma)?;
            fold_low_once(&mut self.mu, gamma)?;
            fold_low_once(&mut self.fresh, gamma)?;
        } else {
            self.later = self.later.mul(&gamma);
        }
        if self.next + 1 == self.h {
            self.folded_inner = Some(inner(&self.d, &self.mu)?);
        }
        self.kept = self.kept.mul(&Fp3::ONE.sub(&gamma));
        self.next += 1;
        Ok(())
    }

    /// U at the challenges, once every variable is bound.
    pub fn finish(&self) -> Result<Fp3> {
        if self.next != self.k {
            return Err(Error::InvalidSessionState);
        }
        Ok(self.before)
    }

    /// The folded mask rows and the fresh mask at the end of the rounds.
    pub fn end(&self) -> Result<FoldRoundEnd> {
        if self.next != self.k {
            return Err(Error::InvalidSessionState);
        }
        let eps = self.eps.ok_or(Error::InvalidSessionState)?;
        let scale = eps.mul(&self.later);
        Ok(FoldRoundEnd {
            d: self.d.iter().map(|value| value.mul(&scale)).collect(),
            mu: self.mu.clone(),
            fresh: self.fresh.clone(),
        })
    }

    pub fn next(&self) -> u32 {
        self.next
    }

    /// The main message row after the challenges bound so far. This is the
    /// prover's witness at the end of the rounds; it is not a verifier output.
    pub fn main_messages(&self) -> &[Fp3] {
        &self.main.messages
    }
}

/// The verifier's running claim for the same rounds. The linear coefficient
/// is the one for which q(0) + q(1) equals the claim, so it is not sent; after
/// every round the claim is q at that round's challenge.
#[derive(Clone, Copy, Debug)]
pub struct FoldRoundVerifier {
    claim: Fp3,
}

impl FoldRoundVerifier {
    pub fn new(mask_sum: Fp3, eps: Fp3, target: Fp3) -> Self {
        Self {
            claim: mask_sum.add(&eps.mul(&target)),
        }
    }

    pub fn round(&mut self, c0: Fp3, c2: Fp3, gamma: Fp3) {
        let two = Fp3::from_u64(2);
        let c1 = self.claim.sub(&two.mul(&c0)).sub(&c2);
        self.claim = c0
            .add(&c1.mul(&gamma))
            .add(&c2.mul(&gamma.mul(&gamma)));
    }

    pub fn claim(&self) -> Fp3 {
        self.claim
    }
}

fn fold_low_values(values: &[Fp3], gamma: &[Fp3], variables: u32) -> Result<Vec<Fp3>> {
    if gamma.len() < variables as usize || variables >= 64 {
        return Err(Error::InvalidConfig);
    }
    let mut out = values.to_vec();
    for challenge in gamma.iter().take(variables as usize) {
        fold_low_once(&mut out, *challenge)?;
    }
    Ok(out)
}

fn fold_mask_form_values(
    k: u32,
    h: u32,
    d: &[Fp3],
    eps: Fp3,
    gamma: &[Fp3],
) -> Result<Vec<Fp3>> {
    if gamma.len() != k as usize || h > k {
        return Err(Error::InvalidConfig);
    }
    let mut out = fold_low_values(d, gamma, h)?;
    let mut scale = eps;
    for challenge in gamma.iter().skip(h as usize) {
        scale = scale.mul(challenge);
    }
    for value in &mut out {
        *value = value.mul(&scale);
    }
    Ok(out)
}

/// Compute the final FOLD constraint from the verifier's fixed group row and
/// the challenges. The caller supplies the already-checked start identity and
/// the running claim after the last round; no new challenge or target is
/// chosen here.
pub fn fold_output(
    shape: &FoldShape,
    layout: &FoldGroupLayout,
    row: &FoldInputRow,
    fresh: u64,
    fresh_root: Digest,
    tree_number: u64,
    eps: Fp3,
    sum: Fp3,
    gamma: &[Fp3],
    claim: Fp3,
) -> Result<FoldOutput> {
    row.check(shape, layout)?;
    if gamma.len() != shape.k as usize {
        return Err(Error::ShapeMismatch);
    }
    let mut coefficients = fold_low_values(&row.coefficients, gamma, shape.k)?;
    for value in &mut coefficients {
        *value = value.mul(&eps);
    }
    let mask_coefficients =
        fold_mask_form_values(shape.k, shape.h, &row.mask_coefficients, eps, gamma)?;
    let mut kept_factor = eps;
    for challenge in gamma {
        kept_factor = kept_factor.mul(&Fp3::ONE.sub(challenge));
    }
    let mut kept = Vec::with_capacity(row.kept.len());
    for form in &row.kept {
        kept.push(FoldKeptForm {
            oracle: form.oracle,
            coefficients: form
                .coefficients
                .iter()
                .map(|value| value.mul(&kept_factor))
                .collect(),
        });
    }
    assemble_fold_output(
        shape,
        layout,
        row.group,
        row.k,
        row.h,
        row.members.clone(),
        coefficients,
        mask_coefficients,
        kept,
        kept_factor,
        claim,
        fresh,
        fresh_root,
        tree_number,
        eps,
        sum,
        gamma,
    )
}

#[allow(clippy::too_many_arguments)]
fn assemble_fold_output(
    shape: &FoldShape,
    layout: &FoldGroupLayout,
    group: u64,
    k: u32,
    h: u32,
    members: Vec<u64>,
    coefficients: Vec<Fp3>,
    mask_coefficients: Vec<Fp3>,
    kept: Vec<FoldKeptForm>,
    kept_factor: Fp3,
    target: Fp3,
    fresh: u64,
    fresh_root: Digest,
    tree_number: u64,
    eps: Fp3,
    sum: Fp3,
    gamma: &[Fp3],
) -> Result<FoldOutput> {
    let fresh_coefficients = fold_mask_weights(
        k,
        h,
        usize::try_from(shape.mask_code.ell).map_err(|_| Error::ResourceLimit)?,
        gamma,
    )?;
    Ok(FoldOutput {
        code: shape.code,
        mask_code: shape.mask_code,
        group,
        k,
        h,
        members,
        gamma: gamma.to_vec(),
        eps,
        sum,
        fresh,
        fresh_root,
        tree_number,
        coefficients,
        mask_coefficients,
        fresh_coefficients,
        kept,
        kept_factor,
        layout: layout.clone(),
        kept_rows: layout.kept_rows_checked()?,
        target,
    })
}

/// The prover's per-round self-check. The drawn side is not a caller's copy:
/// every coefficient and every encoding-tail value is regenerated from its
/// draw number through the numbered source, and the live side must agree with
/// it before a round polynomial is returned.
pub struct FoldRoundSession {
    live: FoldRoundProver,
    drawn: FoldRoundProver,
}

impl FoldRoundSession {
    pub fn new(
        shape: &FoldShape,
        mask: &FoldMask,
        coins: &impl MaskCoins,
        d: Vec<Fp3>,
        mu: Vec<Fp3>,
        main: FoldMainTable,
        kept: Fp3,
    ) -> Result<Self> {
        if mask.coefficients().len() != 2 * shape.k as usize + 1
            || mask.draws().len() != mask.coefficients().len()
        {
            return Err(Error::InvalidConfig);
        }
        let mut drawn = Vec::with_capacity(mask.coefficients().len());
        for (coefficient, number) in mask.coefficients().iter().zip(mask.draws()) {
            if *number == NO_DRAW || coins.redraw(*number)? != *coefficient {
                return Err(Error::InvalidProof);
            }
            drawn.push(coins.redraw(*number)?);
        }
        let tail = mask.tail().ok_or(Error::InvalidConfig)?;
        let tail_rows = 1usize
            .checked_shl(shape.h)
            .ok_or(Error::InvalidConfig)?;
        if tail.rows != tail_rows || tail.len != shape.mask_code.r as usize {
            return Err(Error::ShapeMismatch);
        }
        if tail.values.len() != tail.rows * tail.len || tail.draws.len() != tail.values.len() {
            return Err(Error::ShapeMismatch);
        }
        for (value, number) in tail.values.iter().zip(&tail.draws) {
            if *number == NO_DRAW || coins.redraw(*number)? != *value {
                return Err(Error::InvalidProof);
            }
        }
        let len_zk = shape.mask_code.ell as usize;
        Ok(Self {
            live: FoldRoundProver::new(
                shape.k,
                shape.h,
                mask.coefficients().to_vec(),
                d.clone(),
                mu.clone(),
                len_zk,
                main.clone(),
                kept,
            )?,
            drawn: FoldRoundProver::new(
                shape.k,
                shape.h,
                drawn,
                d,
                mu,
                len_zk,
                main,
                kept,
            )?,
        })
    }

    pub fn start(&mut self, eps: Fp3) -> Result<()> {
        self.live.start(eps)?;
        self.drawn.start(eps)
    }

    pub fn round(&mut self) -> Result<[Fp3; 3]> {
        let live = self.live.round()?;
        let drawn = self.drawn.round()?;
        if live != drawn {
            return Err(Error::InvalidProof);
        }
        Ok(live)
    }

    pub fn bind(&mut self, gamma: Fp3) -> Result<()> {
        self.live.bind(gamma)?;
        self.drawn.bind(gamma)
    }

    pub fn finish(&self) -> Result<Fp3> {
        let live = self.live.finish()?;
        let drawn = self.drawn.finish()?;
        if live != drawn {
            return Err(Error::InvalidProof);
        }
        Ok(live)
    }
}

/// The group-row metadata a FOLD output is assembled from. The coefficient
/// vectors themselves live in the live round state.
struct FoldOutputMeta {
    group: u64,
    k: u32,
    h: u32,
    members: Vec<u64>,
    kept: Vec<FoldKeptForm>,
    target: Fp3,
}

/// The prover's isolated FOLD rounds over the actual main row, the old mask
/// rows and the fresh mask. The fresh mask must already have passed the start
/// path's first as-drawn check; `start` performs the second check before the
/// first round, when the coefficients and the encoding tail are last read, and
/// the round polynomial then uses the live mask data. The q values are not
/// supplied by a verifier.
pub struct FoldProverSession {
    shape: FoldShape,
    coefficients: Vec<Fp3>,
    draws: Vec<u32>,
    tail: MaskTail,
    rounds: FoldRoundProver,
    meta: FoldOutputMeta,
    started: bool,
}

impl FoldProverSession {
    pub fn new(
        shape: &FoldShape,
        layout: &FoldGroupLayout,
        mask: &FoldMask,
        row: FoldInputRow,
        main_messages: Vec<Fp3>,
        mask_messages: Vec<Fp3>,
        kept_messages: Vec<Vec<Fp3>>,
    ) -> Result<Self> {
        row.check(shape, layout)?;
        if kept_messages.len() != row.kept.len() {
            return Err(Error::ShapeMismatch);
        }
        let mut kept = Fp3::ZERO;
        for (form, messages) in row.kept.iter().zip(&kept_messages) {
            if messages.len() != form.coefficients.len() {
                return Err(Error::ShapeMismatch);
            }
            kept = kept.add(&inner(&form.coefficients, messages)?);
        }
        let expected = 2 * shape.k as usize + 1;
        if mask.coefficients().len() != expected || mask.draws().len() != expected {
            return Err(Error::InvalidConfig);
        }
        let tail = mask.tail().cloned().ok_or(Error::InvalidProof)?;
        let FoldInputRow {
            group,
            k,
            h,
            members,
            coefficients,
            mask_coefficients,
            kept: kept_forms,
            target,
        } = row;
        let main = FoldMainTable::new(coefficients, main_messages)?;
        let len_zk = usize::try_from(shape.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        let rounds = FoldRoundProver::new(
            shape.k,
            shape.h,
            mask.coefficients().to_vec(),
            mask_coefficients,
            mask_messages,
            len_zk,
            main,
            kept,
        )?;
        Ok(Self {
            shape: *shape,
            coefficients: mask.coefficients().to_vec(),
            draws: mask.draws().to_vec(),
            tail,
            rounds,
            meta: FoldOutputMeta {
                group,
                k,
                h,
                members,
                kept: kept_forms,
                target,
            },
            started: false,
        })
    }

    pub fn sum(&self) -> Fp3 {
        self.rounds.sum()
    }

    /// The second as-drawn check, immediately before the first round.
    pub fn start(&mut self, eps: Fp3, coins: &impl MaskCoins) -> Result<()> {
        if self.started {
            return Err(Error::InvalidSessionState);
        }
        fold_mask_as_drawn_parts(
            &self.shape,
            &self.coefficients,
            &self.draws,
            &self.tail,
            coins,
        )?;
        self.rounds.start(eps)?;
        self.started = true;
        Ok(())
    }

    pub fn round(&self) -> Result<[Fp3; 3]> {
        self.rounds.round()
    }

    pub fn bind(&mut self, gamma: Fp3) -> Result<()> {
        self.rounds.bind(gamma)
    }

    pub fn end(&self) -> Result<FoldRoundEnd> {
        self.rounds.end()
    }

    pub fn main_messages(&self) -> &[Fp3] {
        self.rounds.main_messages()
    }

    /// Builds the final FOLD output from the live round state. The main and
    /// old-mask coefficients were folded in place by the rounds, so the output
    /// is derived from those vectors instead of keeping a second copy of the
    /// input row.
    #[allow(clippy::too_many_arguments)]
    pub fn output(
        &self,
        layout: &FoldGroupLayout,
        fresh: u64,
        fresh_root: Digest,
        tree_number: u64,
        eps: Fp3,
        sum: Fp3,
        gamma: &[Fp3],
        claim: Fp3,
    ) -> Result<FoldOutput> {
        if gamma.len() != self.shape.k as usize || self.rounds.next() != self.shape.k {
            return Err(Error::InvalidSessionState);
        }
        if layout.fault(&self.shape).is_some()
            || layout.group != self.meta.group
            || layout.k != self.meta.k
            || layout.h != self.meta.h
        {
            return Err(Error::InvalidConfig);
        }
        let end = self.rounds.end()?;
        let coefficients = self
            .rounds
            .main
            .coefficients
            .iter()
            .map(|value| value.mul(&eps))
            .collect::<Vec<_>>();
        let mut kept_factor = eps;
        for challenge in gamma {
            kept_factor = kept_factor.mul(&Fp3::ONE.sub(challenge));
        }
        let mut kept = Vec::with_capacity(self.meta.kept.len());
        for form in &self.meta.kept {
            kept.push(FoldKeptForm {
                oracle: form.oracle,
                coefficients: form
                    .coefficients
                    .iter()
                    .map(|value| value.mul(&kept_factor))
                    .collect(),
            });
        }
        assemble_fold_output(
            &self.shape,
            layout,
            self.meta.group,
            self.meta.k,
            self.meta.h,
            self.meta.members.clone(),
            coefficients,
            end.d,
            kept,
            kept_factor,
            claim,
            fresh,
            fresh_root,
            tree_number,
            eps,
            sum,
            gamma,
        )
    }
}

/// The challenge transcript of the rounds: epsilon after the start message,
/// then one challenge after each round's two sent coefficients. The same
/// prefix and messages reproduce the same challenges.
#[derive(Clone, Debug)]
pub struct FoldRoundTranscript {
    chain: Chain,
    k: u32,
    eps: Option<Fp3>,
    rounds: u32,
}

impl FoldRoundTranscript {
    pub fn new(k: u32, label: u64, statement: &[u8]) -> Result<Self> {
        if k < 1 {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            chain: Chain::begin(label, &Digest::ZERO, statement, &[], None)?,
            k,
            eps: None,
            rounds: 0,
        })
    }

    pub fn draw_epsilon(&mut self, start: &[u8]) -> Result<Fp3> {
        if self.eps.is_some() {
            return Err(Error::InvalidSessionState);
        }
        let mut message = self.chain.message();
        message.push_u64(start.len() as u64);
        for byte in start {
            message.push_u64(*byte as u64);
        }
        let prefix = message.prefix(KIND_EPSILON, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        let eps = self.chain.draw_e()?;
        self.eps = Some(eps);
        Ok(eps)
    }

    pub fn draw_round(&mut self, c0: &Fp3, c2: &Fp3) -> Result<Fp3> {
        if self.eps.is_none() || self.rounds >= self.k {
            return Err(Error::InvalidSessionState);
        }
        let mut message = self.chain.message();
        message.push_fp3(c0);
        message.push_fp3(c2);
        let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        let gamma = self.chain.draw_e()?;
        self.rounds += 1;
        Ok(gamma)
    }

    pub fn rounds(&self) -> u32 {
        self.rounds
    }
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

/// A numbered mask-coin source over caller-provided random bytes. Each draw
/// is a labeled hash of the bytes and its number; the source is random-access,
/// so the prover can regenerate a recorded draw. Production supplies CSPRNG
/// bytes; tests may supply fixed bytes.
pub struct ByteMaskCoins {
    seed: Digest,
    draw: u32,
    #[cfg(test)]
    forced: Option<Vec<(u32, usize, u32, u64)>>,
}

impl ByteMaskCoins {
    pub fn new(seed: Digest) -> Result<Self> {
        if seed == Digest::ZERO {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            seed,
            draw: 0,
            #[cfg(test)]
            forced: None,
        })
    }

    /// A test-only coin source that forces selected `(draw, coordinate,
    /// attempt)` candidates. Unlisted candidates fall through to the real
    /// hash path.
    #[cfg(test)]
    pub fn with_forced(
        seed: Digest,
        forced: Vec<(u32, usize, u32, u64)>,
    ) -> Result<Self> {
        let mut coins = Self::new(seed)?;
        coins.forced = Some(forced);
        Ok(coins)
    }

    /// One retry candidate in the coordinate- and attempt-specific domain.
    fn retry_candidate(&self, number: u32, coordinate: usize, attempt: u32) -> Result<u64> {
        use crate::primitives::transcript::{hash_labeled, tag_word, COIN_INPUTS};

        let mut data = Vec::with_capacity(self.seed.0.len() + 8);
        data.extend_from_slice(&self.seed.0);
        data.extend_from_slice(&(attempt as u64).to_le_bytes());
        let retry = hash_labeled(
            tag_word(COIN_INPUTS, 1 + coordinate as u64),
            number as u64,
            None,
            &data,
        )?;
        Ok(u64::from_le_bytes(
            retry.0[0..8]
                .try_into()
                .map_err(|_| Error::InvalidEncoding)?,
        ))
    }

    fn candidate<F>(
        &self,
        number: u32,
        coordinate: usize,
        attempt: u32,
        word: &mut F,
    ) -> Result<u64>
    where
        F: FnMut(usize, u32) -> Result<u64>,
    {
        #[cfg(test)]
        if let Some(forced) = &self.forced {
            if let Some(value) = forced
                .iter()
                .find(|entry| {
                    entry.0 == number && entry.1 == coordinate && entry.2 == attempt
                })
                .map(|entry| entry.3)
            {
                return Ok(value);
            }
        }
        word(coordinate, attempt)
    }

    fn value_from<F>(&self, number: u32, mut word: F) -> Result<Fp3>
    where
        F: FnMut(usize, u32) -> Result<u64>,
    {
        use zkie_core::common::field::PrimeCharacteristicRing;

        use crate::pcs::request::field::{uniform_fp, Fp, E_DEGREE};

        let mut coefficients = [Fp::ZERO; E_DEGREE];
        for (coordinate, coefficient) in coefficients.iter_mut().enumerate() {
            *coefficient = uniform_fp(|attempt| {
                self.candidate(number, coordinate, attempt, &mut word)
            })?;
        }
        Ok(Fp3::new(coefficients[0], coefficients[1], coefficients[2]))
    }

    fn value_at(&self, number: u32) -> Result<Fp3> {
        use crate::primitives::transcript::{hash_labeled, tag_word, COIN_INPUTS};

        // The base hash supplies one candidate word per coordinate. A
        // rejected candidate is retried in a coordinate- and attempt-specific
        // domain: the draw number is the hash index, the coordinate selects
        // the retry subdomain, and the attempt number is part of the data.
        // The sequential and random-access paths both call this function, so
        // `draw`, `redraw` and `value_at` agree for the same number.
        let digest = hash_labeled(
            tag_word(COIN_INPUTS, 0),
            number as u64,
            None,
            &self.seed.0,
        )?;
        let mut base = [0u64; 3];
        for (coordinate, slot) in base.iter_mut().enumerate() {
            let bytes = &digest.0[coordinate * 8..(coordinate + 1) * 8];
            *slot = u64::from_le_bytes(bytes.try_into().map_err(|_| Error::InvalidEncoding)?);
        }
        self.value_from(number, |coordinate, attempt| {
            if attempt == 0 {
                Ok(base[coordinate])
            } else {
                self.retry_candidate(number, coordinate, attempt)
            }
        })
    }
}

impl MaskCoins for ByteMaskCoins {
    fn draw(&mut self) -> Result<(Fp3, u32)> {
        let number = self.draw;
        self.draw = self.draw.checked_add(1).ok_or(Error::ResourceLimit)?;
        Ok((self.value_at(number)?, number))
    }

    fn redraw(&self, number: u32) -> Result<Fp3> {
        self.value_at(number)
    }
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
    used: bool,
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

    pub fn used(&self) -> bool {
        self.used
    }

    fn mark_used(&mut self) {
        self.used = true;
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
        used: false,
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
    let tail = mask.tail.as_ref().ok_or(Error::InvalidProof)?;
    fold_mask_as_drawn_parts(shape, &mask.coefficients, &mask.draws, tail, coins)
}

fn fold_mask_as_drawn_parts(
    shape: &FoldShape,
    coefficients: &[Fp3],
    draws: &[u32],
    tail: &MaskTail,
    coins: &impl MaskCoins,
) -> Result<()> {
    validate_fold_mask_shape(shape)?;
    let coefficient_count = 2 * shape.k as usize + 1;
    if coefficients.len() != coefficient_count || draws.len() != coefficient_count {
        return Err(Error::InvalidProof);
    }
    let mut differ = false;
    for (coefficient, number) in coefficients.iter().zip(draws) {
        if *number == NO_DRAW || coins.redraw(*number)? != *coefficient {
            differ = true;
        }
    }
    let tail_rows = usize::try_from(shape.mask_rows()?).map_err(|_| Error::ResourceLimit)?;
    let tail_len = usize::try_from(shape.mask_code.r).map_err(|_| Error::ResourceLimit)?;
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
    /// The group's base code, when the trusted plan fixes one. A G_u oracle
    /// can join only a group whose base code and mask code are its own.
    pub code: Option<CodeParams>,
    /// The group's mask code; `None` means the group names no mask code.
    pub mask_code: Option<CodeParams>,
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

/// The verifier's registered metadata for an oracle id. `sampled` is the
/// verifier's fact, never the proof's notice: a notice cannot claim that the
/// oracle has no out-of-domain sample.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OracleMetadata {
    pub id: u64,
    pub group: u64,
    pub tree_number: u64,
    pub leaf_class: u32,
    pub mask_rows: u64,
    pub mask_code: CodeParams,
}

/// The verifier's record of one oracle after its SEND message: the actual
/// root and the sampled fact. It is kept apart from the frozen plan statement
/// so the first query can be bound before the root exists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeOracle {
    pub id: u64,
    pub root: Digest,
    pub sampled: bool,
    pub opening_verified: bool,
    pub main_rows: u64,
    pub mask_rows: u64,
    pub mask_code: CodeParams,
    pub answers: u32,
    pub leaf_class: u32,
}

/// The verifier's SEND registry. Entries are private. `receive_mask` is the
/// registration phase: it binds a claimed root to the static metadata and tree
/// but does not authenticate it. `authenticate_mask` is the later phase that
/// verifies the root's openings at the verifier's allowed positions. A forged
/// root can be registered and must fail the opening phase; a directly built
/// registry is a test-only/trusted input.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RuntimeOracles {
    entries: Vec<RuntimeOracle>,
}

/// The result of a supplied mask opening. `NoOpening` is a legal explicit
/// no-op for an empty position list: it is not a successful authentication,
/// marks no opening and consumes no list. `Verified` means the supplied
/// non-empty opening passed the root check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskAuthentication {
    NoOpening,
    Verified,
}

impl RuntimeOracles {
    /// Registration phase of SEND: check the mask notice against the frozen
    /// statement's oracle metadata and the trusted runtime tree, and record
    /// the claimed root with the complete sent metadata. This binds the root
    /// to the plan; it does NOT authenticate the root, does not take
    /// positions, and does not verify openings. The later verifier phase draws
    /// or fixes positions and authenticates the opening/relations; until then
    /// the root is only a claim. A forged root can be registered here and must
    /// fail the later opening phase.
    pub fn receive_mask(
        &mut self,
        statement: &VerifiedStatement,
        tree_plan: &MaskTreePlan,
        id: u64,
        notice: &MaskNotice,
        root: Digest,
    ) -> Result<()> {
        if self.get(id).is_some() || root == Digest::ZERO {
            return Err(Error::StatementMismatch);
        }
        let metadata = statement.oracle(id).ok_or(Error::StatementMismatch)?;
        if statement.mask_tree(metadata.tree_number) != Some(tree_plan)
            || metadata.tree_number != tree_plan.tree_number
            || notice.mask_rows != metadata.mask_rows
            || notice.mask_code != metadata.mask_code
            || notice.leaf_class != metadata.leaf_class
            || notice.answers != 0
        {
            return Err(Error::StatementMismatch);
        }
        self.entries.push(RuntimeOracle {
            id,
            root,
            sampled: false,
            opening_verified: false,
            main_rows: 0,
            mask_rows: notice.mask_rows,
            mask_code: notice.mask_code,
            answers: notice.answers,
            leaf_class: notice.leaf_class,
        });
        Ok(())
    }

    /// Later verifier phase: check the mask opening at positions produced by
    /// the verifier's trusted transcript. `expected_count` is the plan's query
    /// count, a different source from the positions; the function accepts at
    /// most that many sorted unique positions (a longer or unsorted list is
    /// rejected) and verifies the openings against the registered root. The
    /// caller must supply the complete deduplicated set of the verifier's
    /// draws: this function checks the list it is given and does not prove the
    /// list's source or completeness, so a short or empty list is not evidence
    /// that the transcript drew fewer. An empty list is an explicit no-opening
    /// result, not a successful mask authentication: it marks nothing and
    /// consumes no list. This checks the opening only: it is not a complete
    /// authentication, a proximity test, a table test or a relation closure.
    ///
    /// This is a low-level entry. The count and positions are caller-supplied
    /// and are not self-authenticating: only a trusted session whose transcript
    /// binds the frozen plan may call it, and the tree's complete identity is
    /// checked by that plan, not by this function.
    pub fn authenticate_mask(
        &mut self,
        tree_plan: &MaskTreePlan,
        id: u64,
        expected_count: usize,
        positions: &[u64],
        openings: &RuntimeTreeOpening,
    ) -> Result<MaskAuthentication> {
        if positions.len() > expected_count || positions.len() != openings.columns.len() {
            return Err(Error::StatementMismatch);
        }
        if positions.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(Error::InvalidIndex);
        }
        let entry = self
            .entries
            .iter_mut()
            .find(|oracle| oracle.id == id)
            .ok_or(Error::StatementMismatch)?;
        if entry.opening_verified || entry.mask_code.log_n != tree_plan.log_n {
            return Err(Error::StatementMismatch);
        }
        if positions.is_empty() {
            return Ok(MaskAuthentication::NoOpening);
        }
        verify_mask_opening(tree_plan, &entry.root, positions, openings)?;
        entry.opening_verified = true;
        Ok(MaskAuthentication::Verified)
    }

    pub fn get(&self, id: u64) -> Option<&RuntimeOracle> {
        self.entries.iter().find(|oracle| oracle.id == id)
    }
}

impl MaskOracle {
    /// The verifier's view from its registered metadata. The notice cannot set
    /// or clear the sampled fact.
    pub fn from_metadata(metadata: &OracleMetadata, sampled: bool) -> Self {
        Self {
            id: metadata.id,
            claimed_group: metadata.group,
            main_rows: 0,
            mask_rows: metadata.mask_rows,
            mask_code: metadata.mask_code,
            answers: 0,
            sampled,
        }
    }

    /// The prover's own view of its fresh mask. It is a prover claim and must
    /// never be used by a verifier; the verifier uses the registered metadata.
    pub fn prover_view(id: u64, claimed_group: u64, notice: &MaskNotice) -> Self {
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
    id: Digest,
    groups: VerifiedGroups,
    mask_trees: Vec<MaskTreePlan>,
    oracles: Vec<OracleMetadata>,
    spent: Arc<Mutex<MaskSpend>>,
}

impl VerifiedStatement {
    /// Build the frozen plan statement. The constructor checks the complete
    /// static relations: unique group/tree/oracle numbers, a plan-fixed fresh
    /// mask outside its group's members, and each oracle's group/tree/leaf/
    /// mask-row/mask-code agreement with the registered tree. After this
    /// returns, the fields are private and cannot be changed. The actual
    /// oracle roots and sampled facts are recorded separately by SEND in
    /// `RuntimeOracles`; the type name does not prove the statement's origin,
    /// and the verifier must construct it from its own plan.
    pub fn new(
        groups: VerifiedGroups,
        mask_trees: Vec<MaskTreePlan>,
        oracles: Vec<OracleMetadata>,
    ) -> Result<Self> {
        for (index, group) in groups.groups.iter().enumerate() {
            if groups.groups[..index].iter().any(|other| other.group == group.group) {
                return Err(Error::InvalidConfig);
            }
            if let Some(fresh) = group.fresh {
                if group.members.contains(&fresh) {
                    return Err(Error::InvalidConfig);
                }
            }
            if let Some(code) = group.code {
                if build_code_plan(code.log_n, code.ell, code.r).is_err() {
                    return Err(Error::InvalidConfig);
                }
            }
            if let Some(code) = group.mask_code {
                if build_code_plan(code.log_n, code.ell, code.r).is_err() {
                    return Err(Error::InvalidConfig);
                }
            }
        }
        for (index, tree) in mask_trees.iter().enumerate() {
            if tree.fault().is_some()
                || mask_trees[..index]
                    .iter()
                    .any(|other| other.tree_number == tree.tree_number)
            {
                return Err(Error::InvalidConfig);
            }
        }
        for (index, oracle) in oracles.iter().enumerate() {
            if oracles[..index].iter().any(|other| other.id == oracle.id) {
                return Err(Error::InvalidConfig);
            }
            let group = groups
                .get(oracle.group)
                .ok_or(Error::InvalidConfig)?;
            if group.members.is_empty() {
                return Err(Error::InvalidConfig);
            }
            let tree = mask_trees
                .iter()
                .find(|tree| tree.tree_number == oracle.tree_number)
                .ok_or(Error::InvalidConfig)?;
            if oracle.mask_code.log_n != tree.log_n
                || oracle.leaf_class != tree.leaf_class
                || oracle.mask_rows != tree.mask_rows
                || build_code_plan(oracle.mask_code.log_n, oracle.mask_code.ell, oracle.mask_code.r)
                    .is_err()
            {
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
            writer.put_u64(if group.code.is_some() { 1 } else { 0 });
            if let Some(code) = group.code {
                writer.put_u64(code.log_n as u64);
                writer.put_u64(code.ell);
                writer.put_u64(code.r);
            }
            writer.put_u64(if group.mask_code.is_some() { 1 } else { 0 });
            if let Some(code) = group.mask_code {
                writer.put_u64(code.log_n as u64);
                writer.put_u64(code.ell);
                writer.put_u64(code.r);
            }
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
        writer.put_u64(oracles.len() as u64);
        for oracle in &oracles {
            writer.put_u64(oracle.id);
            writer.put_u64(oracle.group);
            writer.put_u64(oracle.tree_number);
            writer.put_u64(oracle.leaf_class as u64);
            writer.put_u64(oracle.mask_rows);
            writer.put_u64(oracle.mask_code.log_n as u64);
            writer.put_u64(oracle.mask_code.ell);
            writer.put_u64(oracle.mask_code.r);
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
            oracles,
            spent: Arc::new(Mutex::new(MaskSpend::default())),
        })
    }

    pub fn id(&self) -> &Digest {
        &self.id
    }

    pub fn groups(&self) -> &VerifiedGroups {
        &self.groups
    }

    pub fn mask_trees(&self) -> &[MaskTreePlan] {
        &self.mask_trees
    }

    pub fn mask_tree(&self, tree_number: u64) -> Option<&MaskTreePlan> {
        self.mask_trees
            .iter()
            .find(|tree| tree.tree_number == tree_number)
    }

    pub fn oracle(&self, id: u64) -> Option<&OracleMetadata> {
        self.oracles.iter().find(|oracle| oracle.id == id)
    }

    pub fn spend(&self, id: u64) -> Result<()> {
        self.spent
            .lock()
            .map_err(|_| Error::InvalidSessionState)?
            .spend(id)
    }

    /// Spend one fresh-mask oracle. The statement and the runtime registry
    /// must both show an unsampled mask-only message before the single-use
    /// ledger is touched. A sampled message, a message with main rows or a
    /// message with answers cannot be spent as a mask.
    pub fn spend_as_mask(&self, runtime: &RuntimeOracles, id: u64) -> Result<()> {
        let metadata = self.oracle(id).ok_or(Error::StatementMismatch)?;
        let entry = runtime.get(id).ok_or(Error::StatementMismatch)?;
        if entry.sampled
            || entry.answers != 0
            || entry.main_rows != 0
            || entry.mask_rows != metadata.mask_rows
            || entry.mask_code != metadata.mask_code
            || metadata.mask_rows == 0
        {
            return Err(Error::StatementMismatch);
        }
        self.spend(id)
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
    mask: &mut FoldMask,
    coins: &impl MaskCoins,
    oracle: &MaskOracle,
    groups: &VerifiedGroups,
    tree_plan: MaskTreePlan,
    salts: &[Vec<u8>],
) -> Result<FoldStartProver> {
    validate_fold_mask_shape(shape)?;
    if mask.id() != Some(oracle.id) || mask.used() {
        return Err(Error::InvalidProof);
    }
    fold_mask_check(shape, oracle, groups)?;
    fold_mask_as_drawn(shape, mask, coins)?;
    let draft = fold_mask_draft(shape, mask)?;
    let rows = mask_rows(shape, mask)?;
    let tails = mask_tail_rows(shape, mask)?;
    let tree = MaskTree::commit(tree_plan, shape, &rows, &tails, salts)?;
    let sum = fold_mask_sum(shape.k, mask.coefficients())?;
    mask.mark_used();
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
    runtime: &RuntimeOracles,
) -> Result<()> {
    fold_start_verify_with_groups(shape, message, statement, runtime, statement.groups())
}

/// The same FOLD start check with an explicit trusted group view. The overlay
/// must come from the statement plus the session's registered dynamic members;
/// it is not a proof-supplied list. This keeps a dynamically joined oracle in
/// the same group that the FOLD fresh-mask check consumes. This is a low-level
/// entry: it checks the message, registered oracle and fresh-mask rules against
/// the supplied view, but it does not authenticate the view's origin. The
/// session currently guarantees only its local overlay's source; production
/// callers must constrain the call and use an independently trusted statement.
pub fn fold_start_verify_with_groups(
    shape: &FoldShape,
    message: &FoldStartMessage,
    statement: &VerifiedStatement,
    runtime: &RuntimeOracles,
    groups: &VerifiedGroups,
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
    let metadata = statement.oracle(message.id).ok_or(Error::StatementMismatch)?;
    let runtime_entry = runtime.get(message.id).ok_or(Error::StatementMismatch)?;
    if metadata.group != message.claimed_group
        || metadata.tree_number != message.tree_number
        || metadata.leaf_class != message.notice.leaf_class
        || metadata.mask_rows != message.notice.mask_rows
        || metadata.mask_code != message.notice.mask_code
        || runtime_entry.root != message.root
        || runtime_entry.main_rows != 0
        || runtime_entry.mask_rows != message.notice.mask_rows
        || runtime_entry.mask_code != message.notice.mask_code
        || runtime_entry.answers != message.notice.answers
        || runtime_entry.leaf_class != message.notice.leaf_class
        || runtime_entry.sampled
    {
        return Err(Error::StatementMismatch);
    }
    let oracle = MaskOracle::from_metadata(metadata, runtime_entry.sampled);
    fold_mask_check(shape, &oracle, groups)?;
    if message.root == Digest::ZERO {
        return Err(Error::InvalidProof);
    }
    statement.spend_as_mask(runtime, message.id)?;
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
    use crate::pcs::request::field::{fp_from_u64, Fp, MODULUS};
    use crate::pcs::request::params::LEAF_SALT_MIN_BYTES;
    use crate::pcs::request::runtime_tree::verify_mask_opening;
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
            let mut coefficients = [Fp::ZERO; E_DEGREE];
            for (coordinate, coefficient) in coefficients.iter_mut().enumerate() {
                *coefficient = crate::pcs::request::field::uniform_fp(|attempt| {
                    let digest = hash_labeled(
                        tag_word(TEST_COINS, 1 + coordinate as u64),
                        number as u64,
                        Some(&self.seed),
                        &attempt.to_le_bytes(),
                    )?;
                    Ok(u64::from_le_bytes(
                        digest.0[0..8].try_into().map_err(|_| Error::InvalidEncoding)?,
                    ))
                })?;
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
            used: false,
        };
        tampered.coefficients[0] = tampered.coefficients[0].add(&Fp3::ONE);
        assert!(fold_mask_as_drawn(&shape, &tampered, &source).is_err());

        let mut number = FoldMask {
            coefficients: mask.coefficients().to_vec(),
            draws: mask.draws().to_vec(),
            tail: mask.tail().cloned(),
            id: None,
            used: false,
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
            used: false,
        };
        tail_value.tail.as_mut().unwrap().values[0] =
            tail_value.tail.as_ref().unwrap().values[0].add(&Fp3::ONE);
        assert!(fold_mask_as_drawn(&shape, &tail_value, &source).is_err());

        let mut tail_number = FoldMask {
            coefficients: mask.coefficients().to_vec(),
            draws: mask.draws().to_vec(),
            tail: mask.tail().cloned(),
            id: None,
            used: false,
        };
        tail_number.tail.as_mut().unwrap().draws[1] = NO_DRAW;
        assert!(fold_mask_as_drawn(&shape, &tail_number, &source).is_err());

        let no_tail = FoldMask {
            coefficients: mask.coefficients().to_vec(),
            draws: mask.draws().to_vec(),
            tail: None,
            id: None,
            used: false,
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
                code: None,
                mask_code: None,
            }],
        };
        let plan = tree_plan(&shape);
        let salts = tree_salts(&plan);
        let notice = fold_mask_notice(&shape).unwrap();
        let oracle = MaskOracle::prover_view(9, 1, &notice);
        // The mask must have been sent under the id the verifier checks.
        assert!(fold_start_prove(
            &shape, &mut mask, &source, &oracle, &groups, plan, &salts,
        )
        .is_err());
        mask.sent_as(9).unwrap();
        let start = fold_start_prove(
            &shape, &mut mask, &source, &oracle, &groups, plan, &salts,
        )
        .unwrap();
        assert!(mask.used());
        assert!(fold_start_prove(
            &shape, &mut mask, &source, &oracle, &groups, plan, &salts,
        )
        .is_err());
        assert_ne!(start.message.root, Digest::ZERO);
        assert_eq!(
            start.message.sum,
            fold_mask_sum(shape.k, mask.coefficients()).unwrap()
        );
        assert_eq!(start.draft.table.rows, 2);
        assert_eq!(start.draft.tail.draws, mask.tail().unwrap().draws);
        // The verifier accepts the public message with no prover secret.
        let metadata = OracleMetadata {
            id: 9,
            group: 1,
            tree_number: plan.tree_number,
            leaf_class: plan.leaf_class,
            mask_rows: plan.mask_rows,
            mask_code: shape.mask_code,
        };
        let statement =
            VerifiedStatement::new(groups.clone(), vec![plan], vec![metadata.clone()]).unwrap();
        let runtime = RuntimeOracles {
            entries: vec![RuntimeOracle {
                id: 9,
                root: start.message.root,
                sampled: false,
                opening_verified: false,
                main_rows: 0,
                mask_rows: start.message.notice.mask_rows,
                mask_code: start.message.notice.mask_code,
                answers: 0,
                leaf_class: start.message.notice.leaf_class,
            }],
        };
        fold_start_verify(&shape, &start.message, &statement, &runtime).unwrap();
        assert!(fold_start_verify(&shape, &start.message, &statement, &runtime).is_err());
        let mut bad = start.message.clone();
        bad.tree_number = 6;
        assert!(fold_start_verify(&shape, &bad, &statement, &runtime).is_err());
        // The verifier's SEND registry, not the notice, decides whether the
        // oracle was sampled.
        let sampled_runtime = RuntimeOracles {
            entries: vec![RuntimeOracle {
                id: 9,
                root: start.message.root,
                sampled: true,
                opening_verified: false,
                main_rows: 0,
                mask_rows: start.message.notice.mask_rows,
                mask_code: start.message.notice.mask_code,
                answers: 0,
                leaf_class: start.message.notice.leaf_class,
            }],
        };
        assert!(fold_start_verify(
            &shape,
            &start.message,
            &statement,
            &sampled_runtime
        )
        .is_err());
        // A legal non-zero but wrong root is rejected by the SEND registry.
        let wrong_runtime = RuntimeOracles {
            entries: vec![RuntimeOracle {
                id: 9,
                root: Digest([4; 32]),
                sampled: false,
                opening_verified: false,
                main_rows: 0,
                mask_rows: start.message.notice.mask_rows,
                mask_code: start.message.notice.mask_code,
                answers: 0,
                leaf_class: start.message.notice.leaf_class,
            }],
        };
        assert!(fold_start_verify(
            &shape,
            &start.message,
            &statement,
            &wrong_runtime
        )
        .is_err());
        // The committed root opens the mask tree at the drawn positions.
        let positions = [1u64, 4];
        let openings = start.tree.open(&positions).unwrap();
        verify_mask_opening(&plan, &start.message.root, &positions, &openings).unwrap();
        let mut tampered = openings.clone();
        tampered.columns[0].symbols[0] = tampered.columns[0].symbols[0] + Fp::ONE;
        assert!(
            verify_mask_opening(&plan, &start.message.root, &positions, &tampered).is_err()
        );
        // A different claimed group, a changed notice or a different tree
        // layout is rejected before the spend.
        let mut bad = start.message.clone();
        bad.claimed_group = 7;
        assert!(fold_start_verify(&shape, &bad, &statement, &runtime).is_err());
        let mut bad = start.message.clone();
        bad.notice.mask_rows = 4;
        assert!(fold_start_verify(&shape, &bad, &statement, &runtime).is_err());
        // A tampered prover coefficient fails the self-check.
        let mut tampered_mask = FoldMask {
            coefficients: mask.coefficients().to_vec(),
            draws: mask.draws().to_vec(),
            tail: mask.tail().cloned(),
            id: Some(9),
            used: false,
        };
        tampered_mask.coefficients[0] = tampered_mask.coefficients[0].add(&Fp3::ONE);
        assert!(fold_start_prove(
            &shape,
            &mut tampered_mask,
            &source,
            &oracle,
            &groups,
            plan,
            &salts,
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
                code: None,
                mask_code: None,
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
                code: None,
                mask_code: None,
            }],
        };
        assert!(fold_mask_check(&shape, &good, &member_groups).is_err());
        // A fresh id the plan fixes must match.
        let fixed = VerifiedGroups {
            groups: vec![VerifiedGroup {
                group: 1,
                members: vec![1, 2],
                fresh: Some(9),
                code: None,
                mask_code: None,
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
                code: None,
                mask_code: None,
            }],
        };
        assert!(fold_mask_check(&shape, &good, &forged_clean).is_ok());
        // A group with no members is not a group.
        let empty = VerifiedGroups {
            groups: vec![VerifiedGroup {
                group: 1,
                members: Vec::new(),
                fresh: None,
                code: None,
                mask_code: None,
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
            used: false,
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
    fn statement_validation_rejects_bad_relations() {
        let shape = shape();
        let plan = tree_plan(&shape);
        let groups = VerifiedGroups {
            groups: vec![VerifiedGroup {
                group: 1,
                members: vec![1, 2],
                fresh: Some(9),
                code: None,
                mask_code: None,
            }],
        };
        let metadata = OracleMetadata {
            id: 9,
            group: 1,
            tree_number: plan.tree_number,
            leaf_class: plan.leaf_class,
            mask_rows: plan.mask_rows,
            mask_code: shape.mask_code,
        };
        VerifiedStatement::new(groups.clone(), vec![plan], vec![metadata.clone()]).unwrap();
        let mut duplicate_group = groups.clone();
        duplicate_group.groups.push(groups.groups[0].clone());
        assert!(VerifiedStatement::new(duplicate_group, vec![plan], vec![metadata.clone()]).is_err());
        assert!(VerifiedStatement::new(groups.clone(), vec![plan, plan], vec![metadata.clone()]).is_err());
        assert!(VerifiedStatement::new(groups.clone(), vec![plan], vec![metadata.clone(), metadata.clone()]).is_err());
        let mut bad = metadata.clone();
        bad.group = 7;
        assert!(VerifiedStatement::new(groups.clone(), vec![plan], vec![bad]).is_err());
        let mut bad = metadata.clone();
        bad.tree_number = 6;
        assert!(VerifiedStatement::new(groups.clone(), vec![plan], vec![bad]).is_err());
        let mut bad = metadata.clone();
        bad.leaf_class = 4;
        assert!(VerifiedStatement::new(groups.clone(), vec![plan], vec![bad]).is_err());
        let mut fresh_member = groups;
        fresh_member.groups[0].fresh = Some(1);
        assert!(VerifiedStatement::new(fresh_member, vec![plan], vec![metadata]).is_err());
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

    fn round_shape(k: u32, h: u32) -> FoldShape {
        FoldShape {
            code: build_code_plan(4, 8, 1).unwrap(),
            mask_code: build_code_plan(3, 4, 1).unwrap(),
            k,
            h,
            mask_leaf_class: 3,
        }
    }

    fn small_mask(shape: &FoldShape) -> (FoldMask, TestMaskCoins) {
        let mut coins = TestMaskCoins::new(Digest([11; 32]));
        let mut mask = fold_mask_make(shape, &mut coins).unwrap();
        fold_mask_make_tail(shape, &mut mask, &mut coins).unwrap();
        (mask, coins)
    }

    fn zero_round_inputs(
        shape: &FoldShape,
    ) -> (Vec<Fp3>, Vec<Fp3>, FoldMainTable) {
        let mask_entries = (1usize << shape.h) * shape.mask_code.ell as usize;
        let main_entries = 1usize << shape.k;
        (
            vec![Fp3::ZERO; mask_entries],
            vec![Fp3::ZERO; mask_entries],
            FoldMainTable::new(vec![Fp3::ZERO; main_entries], vec![Fp3::ZERO; main_entries])
                .unwrap(),
        )
    }

    #[test]
    fn fold_round_polynomials_reproduce_the_hiding_sum() {
        for k in 1..=3u32 {
            let h = k;
            let shape = round_shape(k, h);
            let len_zk = shape.mask_code.ell as usize;
            let coefficients: Vec<Fp3> = (0..2 * k + 1)
                .map(|index| Fp3::from_u64(10 + index as u64))
                .collect();
            let (d, mu, main) = zero_round_inputs(&shape);
            let mut prover = FoldRoundProver::new(
                k,
                h,
                coefficients.clone(),
                d,
                mu,
                len_zk,
                main,
                Fp3::ZERO,
            )
            .unwrap();
            assert_eq!(prover.sum(), fold_mask_sum(k, &coefficients).unwrap());
            let eps = Fp3::from_u64(7);
            prover.start(eps).unwrap();
            let mut verifier = FoldRoundVerifier::new(prover.sum(), eps, Fp3::ZERO);
            let mut transcript = FoldRoundTranscript::new(k, 7, b"fold-rounds").unwrap();
            transcript.draw_epsilon(b"start").unwrap();
            for _ in 0..k {
                let q = prover.round().unwrap();
                let gamma = transcript.draw_round(&q[0], &q[2]).unwrap();
                verifier.round(q[0], q[2], gamma);
                prover.bind(gamma).unwrap();
            }
            assert_eq!(verifier.claim(), prover.finish().unwrap());
            assert!(prover.end().is_ok());
        }
    }

    #[test]
    fn fold_round_challenges_reproduce_and_enforce_order() {
        let c0 = Fp3::from_u64(1);
        let c2 = Fp3::from_u64(2);
        let mut first = FoldRoundTranscript::new(3, 7, b"same").unwrap();
        assert!(first.draw_round(&c0, &c2).is_err());
        let eps = first.draw_epsilon(b"start").unwrap();
        let mut second = FoldRoundTranscript::new(3, 7, b"same").unwrap();
        assert_eq!(eps, second.draw_epsilon(b"start").unwrap());
        let gamma = first.draw_round(&c0, &c2).unwrap();
        assert_eq!(gamma, second.draw_round(&c0, &c2).unwrap());
        let mut other = FoldRoundTranscript::new(3, 7, b"same").unwrap();
        other.draw_epsilon(b"start").unwrap();
        assert_ne!(gamma, other.draw_round(&Fp3::from_u64(3), &c2).unwrap());
        let mut changed = FoldRoundTranscript::new(3, 7, b"other").unwrap();
        assert_ne!(eps, changed.draw_epsilon(b"start").unwrap());
        let mut extra = FoldRoundTranscript::new(3, 7, b"same").unwrap();
        extra.draw_epsilon(b"start").unwrap();
        for _ in 0..3 {
            extra.draw_round(&c0, &c2).unwrap();
        }
        assert!(extra.draw_round(&c0, &c2).is_err());
    }

    #[test]
    fn fold_round_main_table_is_folded_by_the_prover() {
        let shape = round_shape(2, 2);
        let len_zk = shape.mask_code.ell as usize;
        let coefficients: Vec<Fp3> = (0..5).map(|index| Fp3::from_u64(30 + index)).collect();
        let mask_entries = (1usize << shape.h) * len_zk;
        let mut main = FoldMainTable::new(
            (0..4).map(|index| Fp3::from_u64(40 + index)).collect(),
            (0..4).map(|index| Fp3::from_u64(50 + index)).collect(),
        )
        .unwrap();
        let mut prover = FoldRoundProver::new(
            shape.k,
            shape.h,
            coefficients.clone(),
            vec![Fp3::ZERO; mask_entries],
            vec![Fp3::ZERO; mask_entries],
            len_zk,
            main.clone(),
            Fp3::ZERO,
        )
        .unwrap();
        let eps = Fp3::from_u64(7);
        prover.start(eps).unwrap();
        let target = inner(&main.coefficients, &main.messages).unwrap();
        let mut verifier = FoldRoundVerifier::new(prover.sum(), eps, target);
        let mut gammas = Vec::new();
        for round in 0..shape.k {
            let q = prover.round().unwrap();
            let gamma = Fp3::from_u64(200 + round as u64);
            verifier.round(q[0], q[2], gamma);
            prover.bind(gamma).unwrap();
            gammas.push(gamma);
        }
        for gamma in gammas {
            fold_low_once(&mut main.coefficients, gamma).unwrap();
            fold_low_once(&mut main.messages, gamma).unwrap();
        }
        let folded = inner(&main.coefficients, &main.messages).unwrap();
        assert_eq!(
            verifier.claim(),
            prover.finish().unwrap().add(&eps.mul(&folded))
        );
    }

    #[test]
    fn fold_round_as_drawn_regenerates_coefficients_and_tail() {
        let shape = round_shape(3, 3);
        let (mask, coins) = small_mask(&shape);
        let (d, mu, main) = zero_round_inputs(&shape);
        let eps = Fp3::from_u64(5);
        let draw = |round: u32| Fp3::from_u64(100 + round as u64);
        let mut session = FoldRoundSession::new(
            &shape,
            &mask,
            &coins,
            d.clone(),
            mu.clone(),
            main.clone(),
            Fp3::ZERO,
        )
        .unwrap();
        session.start(eps).unwrap();
        for round in 0..3 {
            session.round().unwrap();
            session.bind(draw(round)).unwrap();
        }
        assert!(session.finish().is_ok());

        // A coefficient lost after round 0 is caught when its variable's
        // round is computed.
        let mut session = FoldRoundSession::new(
            &shape,
            &mask,
            &coins,
            d.clone(),
            mu.clone(),
            main.clone(),
            Fp3::ZERO,
        )
        .unwrap();
        session.start(eps).unwrap();
        session.round().unwrap();
        session.bind(draw(0)).unwrap();
        session.live.coefficients[hp_place(3, 1, 2).unwrap()] = Fp3::from_u64(777);
        assert!(session.round().is_err());
        let mut session =
            FoldRoundSession::new(&shape, &mask, &coins, d.clone(), mu.clone(), main.clone(), Fp3::ZERO)
                .unwrap();
        session.start(eps).unwrap();
        session.live.coefficients[hp_place(3, 0, 1).unwrap()] = Fp3::from_u64(888);
        assert!(session.round().is_err());

        // A wrong source, an overwritten coefficient and a tampered tail are
        // each rejected before any round.
        let other = TestMaskCoins::new(Digest([12; 32]));
        assert!(FoldRoundSession::new(
            &shape,
            &mask,
            &other,
            d.clone(),
            mu.clone(),
            main.clone(),
            Fp3::ZERO,
        )
        .is_err());
        let mut same_source = TestMaskCoins::new(Digest([11; 32]));
        let mut wrong_copy = fold_mask_make(&shape, &mut same_source).unwrap();
        fold_mask_make_tail(&shape, &mut wrong_copy, &mut same_source).unwrap();
        wrong_copy.coefficients[0] = wrong_copy.coefficients[0].add(&Fp3::ONE);
        assert!(FoldRoundSession::new(
            &shape,
            &wrong_copy,
            &same_source,
            d.clone(),
            mu.clone(),
            main.clone(),
            Fp3::ZERO,
        )
        .is_err());
        let mut same_source = TestMaskCoins::new(Digest([11; 32]));
        let mut tampered = fold_mask_make(&shape, &mut same_source).unwrap();
        fold_mask_make_tail(&shape, &mut tampered, &mut same_source).unwrap();
        tampered.tail.as_mut().unwrap().values[0] =
            tampered.tail().unwrap().values[0].add(&Fp3::ONE);
        assert!(FoldRoundSession::new(
            &shape,
            &tampered,
            &same_source,
            d,
            mu,
            main,
            Fp3::ZERO,
        )
        .is_err());
    }

    #[test]
    fn the_byte_mask_coins_are_reproducible_and_seeded() {
        let mut first = ByteMaskCoins::new(Digest([0x11; 32])).unwrap();
        let (a, zero) = first.draw().unwrap();
        let (b, one) = first.draw().unwrap();
        assert_eq!((zero, one), (0, 1));
        assert_eq!(a, first.redraw(0).unwrap());
        assert_eq!(b, first.redraw(1).unwrap());
        let other = ByteMaskCoins::new(Digest([0x12; 32])).unwrap();
        assert_ne!(a, other.redraw(0).unwrap());
        assert!(ByteMaskCoins::new(Digest::ZERO).is_err());
    }

    fn fold_low_serial(values: &[Fp3], gamma: Fp3) -> Vec<Fp3> {
        let rows = values.len() / 2;
        (0..rows)
            .map(|index| fold_pair(values[2 * index], values[2 * index + 1], gamma))
            .collect()
    }

    #[test]
    fn fold_low_once_matches_serial_across_threshold_blocks_and_threads() {
        let threshold = 1usize << 16;
        let block = 1usize << 18;
        let row_counts = [
            threshold - 1,
            threshold,
            threshold + 1,
            block - 1,
            block,
            block + 1,
        ];
        for rows in row_counts {
            let len = rows * 2;
            let gamma = Fp3::from_u64(17);
            let input: Vec<Fp3> = (0..len)
                .map(|index| Fp3::from_u64(index as u64 + 1))
                .collect();
            let expected = fold_low_serial(&input, gamma);
            for threads in [1usize, 2, 64] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let mut got = input.clone();
                pool.install(|| fold_low_once(&mut got, gamma).unwrap());
                assert_eq!(got, expected, "rows {rows} threads {threads}");
            }
        }
        let mut odd = vec![Fp3::ONE; 3];
        assert!(fold_low_once(&mut odd, Fp3::from_u64(3)).is_err());
    }

}

#[cfg(test)]
mod unbiased_coin_tests {
    use super::*;
    use crate::pcs::request::field::{fp_from_u64, Fp, MODULUS};
    use crate::primitives::transcript::COIN_INPUTS;

    fn stream(seed: u8, count: usize) -> Vec<Fp3> {
        let mut coins = ByteMaskCoins::new(Digest([seed; 32])).unwrap();
        (0..count)
            .map(|_| coins.draw().unwrap().0)
            .collect()
    }

    #[test]
    fn sequential_and_random_access_agree_for_the_same_number() {
        let mut coins = ByteMaskCoins::new(Digest([0x21; 32])).unwrap();
        for number in 0..8u32 {
            let (value, drawn) = coins.draw().unwrap();
            assert_eq!(drawn, number);
            assert_eq!(coins.redraw(number).unwrap(), value);
        }
    }

    #[test]
    fn the_stream_is_deterministic_per_seed_and_distinct_across_seeds() {
        assert_eq!(stream(0x31, 8), stream(0x31, 8));
        assert_ne!(stream(0x31, 8), stream(0x32, 8));
        let values = stream(0x31, 8);
        for (index, value) in values.iter().enumerate() {
            assert!(!values[index + 1..].contains(value));
        }
    }

    fn forced_base_words(number: u32) -> Vec<(u32, usize, u32, u64)> {
        (0..3)
            .map(|coordinate| (number, coordinate, 0, MODULUS))
            .collect()
    }

    fn first_accepted_retry(coins: &ByteMaskCoins, number: u32, coordinate: usize) -> Fp {
        // The production sampler tries attempts 0..SAMPLE_TRIES (0..=63);
        // attempt 64 cannot execute. The helper range matches that bound.
        for attempt in 1..crate::pcs::request::field::SAMPLE_TRIES {
            let word = coins.retry_candidate(number, coordinate, attempt).unwrap();
            if word < MODULUS {
                return crate::pcs::request::field::fp_from_u64(word);
            }
        }
        panic!("retry exhaustion in the test seed");
    }

    #[test]
    fn forced_base_rejection_uses_the_real_retry_domain() {
        let seed = Digest([0x55; 32]);
        let number = 5u32;
        let coins = ByteMaskCoins::with_forced(seed, forced_base_words(number)).unwrap();
        let value = coins.redraw(number).unwrap();
        for coordinate in 0..3 {
            assert_eq!(
                value.c[coordinate],
                first_accepted_retry(&coins, number, coordinate)
            );
        }
    }

    #[test]
    fn forced_retries_cover_second_attempt_coordinates_and_random_access() {
        let seed = Digest([0x56; 32]);
        let number = 0u32;
        let forced = vec![
            (number, 0, 0, MODULUS),
            (number, 0, 1, MODULUS),
            (number, 0, 2, 7),
            (number, 1, 0, 9),
            (number, 2, 0, u64::MAX),
            (number, 2, 1, 11),
        ];
        let mut coins = ByteMaskCoins::with_forced(seed, forced).unwrap();
        let (drawn, drawn_number) = coins.draw().unwrap();
        assert_eq!(drawn_number, number);
        assert_eq!(
            drawn,
            Fp3::new(fp_from_u64(7), fp_from_u64(9), fp_from_u64(11))
        );
        let other = coins.redraw(1).unwrap();
        assert_ne!(other, drawn);
        assert_eq!(coins.redraw(drawn_number).unwrap(), drawn);
    }

    #[test]
    fn forced_exhaustion_returns_an_error() {
        let seed = Digest([0x57; 32]);
        let number = 3u32;
        let forced = (0..64u32)
            .map(|attempt| (number, 0, attempt, MODULUS))
            .collect::<Vec<_>>();
        let coins = ByteMaskCoins::with_forced(seed, forced).unwrap();
        assert!(matches!(
            coins.redraw(number),
            Err(Error::EntropyUnavailable)
        ));
    }

    #[test]
    fn retry_tag_uses_the_coordinate_subdomain_and_attempt_data() {
        let seed = Digest([0x58; 32]);
        let number = 4u32;
        let coins = ByteMaskCoins::new(seed).unwrap();
        for coordinate in 0..3usize {
            for attempt in 1..=3u32 {
                let mut data = Vec::new();
                data.extend_from_slice(&seed.0);
                data.extend_from_slice(&(attempt as u64).to_le_bytes());
                let digest = hash_labeled(
                    tag_word(COIN_INPUTS, 1 + coordinate as u64),
                    number as u64,
                    None,
                    &data,
                )
                .unwrap();
                let expected = u64::from_le_bytes(digest.0[0..8].try_into().unwrap());
                assert_eq!(
                    coins.retry_candidate(number, coordinate, attempt).unwrap(),
                    expected
                );
            }
        }
        assert_ne!(
            coins.retry_candidate(number, 0, 1).unwrap(),
            coins.retry_candidate(number, 1, 1).unwrap()
        );
        assert_ne!(
            coins.retry_candidate(number, 0, 1).unwrap(),
            coins.retry_candidate(number, 0, 2).unwrap()
        );
        assert_ne!(
            coins.retry_candidate(number, 0, 1).unwrap(),
            coins.retry_candidate(number + 1, 0, 1).unwrap()
        );
    }
}
