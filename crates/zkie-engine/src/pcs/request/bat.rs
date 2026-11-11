//! An isolated BAT foundation: one group's ordered claims are frozen, a
//! challenge is drawn from the session chain, and the claims are combined by
//! powers of that challenge into one group row. This is candidate development
//! code only; it is not wired into a production verifier and does not claim a
//! completed protocol round.
//!
//! The layer takes generator rows as claims. A repeated physical draw may
//! therefore appear as several rows for the same entry; they are all kept and
//! weighted by their own powers. Physical-opening deduplication belongs to the
//! caller's SEND/opening layer and is not represented or claimed here.
//!
//! This module is not connected to the FOLDEQ session. Sample rows are fixture
//! inputs shaped like SEND's registered rows; they are not authenticated by a
//! real SEND receipt, and this layer has no sample-pad registration or pad-read
//! restriction. The mask state is only a placeholder, and BAT output does not
//! unlock the isolated FOLD next stage. These are production integration gates,
//! not hidden assumptions.

use std::collections::BTreeSet;
use std::sync::Arc;

use rayon::prelude::*;

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::chain::{Chain, KIND_ALGEBRAIC};
use crate::pcs::request::code::{build_code_plan, CodeParams};
use crate::pcs::request::field::Fp3;
use crate::pcs::request::fold::{
    fold_member_firsts, mask_fits, validate_fold_mask_shape, FoldGroupLayout, FoldInputRow,
    FoldKeptForm, FoldMemberLayout, FoldPart, FoldShape,
};

/// The two tables of an oracle message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatPart {
    Main,
    Mask,
}

/// The class of a claim in the fixed order in which a group run takes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatRowKind {
    Plain,
    Sample,
    Generator,
}

/// A dense coefficient vector on one table of one oracle message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatTerm {
    pub oracle: u64,
    pub part: BatPart,
    pub values: Vec<Fp3>,
}

/// One written-out coefficient on an entry of one table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatUnit {
    pub oracle: u64,
    pub part: BatPart,
    pub index: u64,
    pub coefficient: Fp3,
}

/// Coefficients at first, first + step, ... of one table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatRun {
    pub oracle: u64,
    pub part: BatPart,
    pub first: u64,
    pub step: u64,
    pub values: Vec<Fp3>,
}

/// A whole coefficient vector on one member's main table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatForm {
    pub oracle: u64,
    pub values: Vec<Fp3>,
}

/// One isolated linear claim. The type is a candidate foundation, not a proof
/// value. The field layout mirrors the isolated claim categories: dense terms
/// and forms, written-out units and runs, and a public target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatRow {
    pub kind: BatRowKind,
    pub terms: Vec<BatTerm>,
    pub units: Vec<BatUnit>,
    pub runs: Vec<BatRun>,
    pub forms: Vec<BatForm>,
    pub target: Fp3,
}

impl BatRow {
    pub fn new(kind: BatRowKind) -> Self {
        Self {
            kind,
            terms: Vec::new(),
            units: Vec::new(),
            runs: Vec::new(),
            forms: Vec::new(),
            target: Fp3::ZERO,
        }
    }

    pub fn add_term(&mut self, oracle: u64, part: BatPart, values: Vec<Fp3>) {
        self.terms.push(BatTerm {
            oracle,
            part,
            values,
        });
    }

    pub fn add_unit(&mut self, oracle: u64, part: BatPart, index: u64, coefficient: Fp3) {
        self.units.push(BatUnit {
            oracle,
            part,
            index,
            coefficient,
        });
    }

    pub fn add_run(
        &mut self,
        oracle: u64,
        part: BatPart,
        first: u64,
        step: u64,
        values: Vec<Fp3>,
    ) {
        self.runs.push(BatRun {
            oracle,
            part,
            first,
            step,
            values,
        });
    }

    pub fn add_form(&mut self, oracle: u64, values: Vec<Fp3>) {
        self.forms.push(BatForm { oracle, values });
    }
}

/// The fixed oracle shape of one isolated group member. `main_code` is absent
/// only for a mask-only member; `mask_code` is absent only when there are no
/// mask rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatOracleShape {
    pub oracle: u64,
    pub main_rows: u64,
    pub mask_rows: u64,
    pub main_code: Option<CodeParams>,
    pub mask_code: Option<CodeParams>,
    pub answers: u32,
}

fn ceil_log2(value: u64) -> u32 {
    if value <= 1 {
        0
    } else {
        64 - (value - 1).leading_zeros()
    }
}

impl BatOracleShape {
    pub fn fault(&self) -> Option<&'static str> {
        if self.main_rows == 0 && self.mask_rows == 0 {
            return Some("a BAT member is an empty oracle");
        }
        if self.main_rows > 0 && self.main_code.is_none() {
            return Some("a BAT main table has no code");
        }
        if self.main_rows == 0 && self.main_code.is_some() {
            return Some("a mask-only BAT member names a base code");
        }
        if self.mask_rows > 0 && self.mask_code.is_none() {
            return Some("a BAT mask table has no code");
        }
        if self.mask_rows == 0 && self.mask_code.is_some() {
            return Some("a BAT member with no mask rows names a mask code");
        }
        None
    }

    pub fn has(&self, part: BatPart) -> bool {
        match part {
            BatPart::Main => self.main_rows > 0,
            BatPart::Mask => self.mask_rows > 0,
        }
    }

    pub fn rows(&self, part: BatPart) -> u64 {
        match part {
            BatPart::Main => self.main_rows,
            BatPart::Mask => self.mask_rows,
        }
    }

    pub fn row_bits(&self, part: BatPart) -> u32 {
        ceil_log2(self.rows(part))
    }

    pub fn log_ell(&self, part: BatPart) -> u32 {
        match part {
            BatPart::Main => self.main_code.map(|code| code.log_ell()).unwrap_or(0),
            BatPart::Mask => self.mask_code.map(|code| code.log_ell()).unwrap_or(0),
        }
    }

    pub fn table_bits(&self, part: BatPart) -> u32 {
        self.row_bits(part) + self.log_ell(part)
    }

    pub fn table_len(&self, part: BatPart) -> Result<usize> {
        1usize
            .checked_shl(self.table_bits(part))
            .ok_or(Error::ResourceLimit)
    }
}

/// One member of a group layout. A kept main table has no folded-main block;
/// a member without mask rows has no mask block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatMemberLayout {
    pub oracle: u64,
    pub kept: bool,
    pub main_first: Option<u64>,
    pub mask_first: Option<u64>,
}

/// The group layout: row variables, code widths, members in sending order and
/// each member's block starts. This is candidate fixture data, not a
/// proof-supplied layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatGroupLayout {
    pub group: u64,
    pub k: u32,
    pub h: u32,
    pub log_ell: u32,
    pub log_ell_zk: u32,
    pub members: Vec<BatMemberLayout>,
}

impl BatGroupLayout {
    pub fn find(&self, oracle: u64) -> Option<usize> {
        self.members
            .iter()
            .position(|member| member.oracle == oracle)
    }

    pub fn group_row_bits(&self, part: BatPart) -> u32 {
        match part {
            BatPart::Main => self.k,
            BatPart::Mask => self.h,
        }
    }

    pub fn group_len(&self, part: BatPart) -> Result<usize> {
        let bits = match part {
            BatPart::Main => self.k + self.log_ell,
            BatPart::Mask => self.h + self.log_ell_zk,
        };
        1usize.checked_shl(bits).ok_or(Error::ResourceLimit)
    }
}

#[derive(Clone, Debug)]
struct BatOracle {
    shape: BatOracleShape,
    sampled: bool,
    sample_rows: Vec<BatRow>,
    mask_open: bool,
    mask_used: BTreeSet<u64>,
}

#[derive(Clone, Debug)]
struct BatGroup {
    group: u64,
    code: CodeParams,
    mask_code: CodeParams,
    members: Vec<u64>,
    kept: Vec<bool>,
    rows: Vec<BatRow>,
    in_run: bool,
}

/// One run's frozen input: the layout and the claims in the fixed class order.
/// The rows are shared rather than copied when the session retains the frozen
/// input for the completion check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatRunInput {
    pub layout: BatGroupLayout,
    pub rows: Arc<Vec<BatRow>>,
}

impl BatRunInput {
    pub fn rows(&self) -> &[BatRow] {
        self.rows.as_slice()
    }
}

/// One combined group claim. `main` and `mask` are full group-table coefficient
/// vectors; `kept` has one full member-table vector per kept oracle that the
/// combined claim reads, ordered by oracle number. This is candidate data: it
/// does not imply FOLD/IOPP verification or unlock the FOLD next stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatGroupRow {
    pub group: u64,
    pub main: Vec<Fp3>,
    pub mask: Vec<Fp3>,
    pub kept: Vec<BatKeptForm>,
    pub target: Fp3,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatKeptForm {
    pub oracle: u64,
    pub coefficients: Vec<Fp3>,
}

/// The isolated statement and its dynamic oracle/group state.
#[derive(Clone, Debug, Default)]
pub struct BatStatement {
    oracles: Vec<BatOracle>,
    groups: Vec<BatGroup>,
}

impl BatStatement {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an oracle in the trusted sending order. Its id is the current
    /// registration count; an external id that is out of order or skips ahead
    /// is rejected. The shape's tables are checked, but the fixture itself is
    /// not authenticated as a real SEND message.
    pub fn add_oracle(&mut self, shape: BatOracleShape) -> Result<u64> {
        if shape.oracle != self.oracles.len() as u64 {
            return Err(Error::StatementMismatch);
        }
        if shape.fault().is_some()
            || self
                .oracles
                .iter()
                .any(|oracle| oracle.shape.oracle == shape.oracle)
        {
            return Err(Error::InvalidConfig);
        }
        if let Some(code) = shape.main_code {
            if !code.ell.is_power_of_two() {
                return Err(Error::InvalidConfig);
            }
            build_code_plan(code.log_n, code.ell, code.r)?;
        }
        if let Some(code) = shape.mask_code {
            if !code.ell.is_power_of_two() {
                return Err(Error::InvalidConfig);
            }
            build_code_plan(code.log_n, code.ell, code.r)?;
        }
        let oracle = shape.oracle;
        self.oracles.push(BatOracle {
            shape,
            sampled: false,
            sample_rows: Vec::new(),
            mask_open: shape.mask_rows > 0,
            mask_used: BTreeSet::new(),
        });
        Ok(oracle)
    }

    fn oracle(&self, id: u64) -> Result<&BatOracle> {
        self.oracles
            .iter()
            .find(|oracle| oracle.shape.oracle == id)
            .ok_or(Error::StatementMismatch)
    }

    fn oracle_mut(&mut self, id: u64) -> Result<&mut BatOracle> {
        self.oracles
            .iter_mut()
            .find(|oracle| oracle.shape.oracle == id)
            .ok_or(Error::StatementMismatch)
    }

    fn group(&self, group: u64) -> Result<&BatGroup> {
        self.groups
            .iter()
            .find(|entry| entry.group == group)
            .ok_or(Error::StatementMismatch)
    }

    fn group_mut(&mut self, group: u64) -> Result<&mut BatGroup> {
        self.groups
            .iter_mut()
            .find(|entry| entry.group == group)
            .ok_or(Error::StatementMismatch)
    }

    /// Register the complete ordered sample answer rows of one oracle. A
    /// sampled oracle is registered once; the row count is its answer count,
    /// every row is a SAMPLE claim, and every coefficient reference must name
    /// this oracle and fit its main/mask table. These are fixture rows standing
    /// in for SEND's registered sample rows, not a producer self-certification.
    pub fn sample(&mut self, oracle: u64, rows: Vec<BatRow>) -> Result<()> {
        let shape = {
            let entry = self.oracle(oracle)?;
            if entry.sampled || entry.shape.answers == 0 {
                return Err(Error::InvalidSessionState);
            }
            entry.shape
        };
        if rows.len() != shape.answers as usize
            || rows.iter().any(|row| row.kind != BatRowKind::Sample)
        {
            return Err(Error::ShapeMismatch);
        }
        for row in &rows {
            check_row_entries(row, |referenced| {
                if referenced != oracle {
                    return Err(Error::StatementMismatch);
                }
                Ok(shape)
            })?;
        }
        let entry = self.oracle_mut(oracle)?;
        entry.sample_rows = rows;
        entry.sampled = true;
        Ok(())
    }

    pub fn new_group(&mut self, code: CodeParams, mask_code: CodeParams) -> Result<u64> {
        if !code.ell.is_power_of_two() || !mask_code.ell.is_power_of_two() {
            return Err(Error::InvalidConfig);
        }
        build_code_plan(code.log_n, code.ell, code.r)?;
        build_code_plan(mask_code.log_n, mask_code.ell, mask_code.r)?;
        if !mask_fits(&code, &mask_code) {
            return Err(Error::InvalidConfig);
        }
        let group = self.groups.len() as u64;
        self.groups.push(BatGroup {
            group,
            code,
            mask_code,
            members: Vec::new(),
            kept: Vec::new(),
            rows: Vec::new(),
            in_run: false,
        });
        Ok(group)
    }

    /// Join a sampled oracle. The group has one base and mask code; a member is
    /// inserted in oracle order, which is sending order for numbered oracles.
    pub fn join(&mut self, group: u64, oracle: u64, kept: bool) -> Result<()> {
        let sampled = self.oracle(oracle)?.sampled;
        let shape = self.oracle(oracle)?.shape;
        let entry = self.group(group)?;
        if entry.in_run || !sampled {
            return Err(Error::InvalidSessionState);
        }
        if kept && shape.main_rows == 0 {
            return Err(Error::ShapeMismatch);
        }
        if shape.main_rows > 0 && shape.main_code != Some(entry.code) {
            return Err(Error::StatementMismatch);
        }
        if shape.mask_rows > 0 && shape.mask_code != Some(entry.mask_code) {
            return Err(Error::StatementMismatch);
        }
        if entry.members.contains(&oracle) {
            return Err(Error::StatementMismatch);
        }
        let entry = self.group_mut(group)?;
        let at = entry
            .members
            .binary_search(&oracle)
            .unwrap_or_else(|index| index);
        entry.members.insert(at, oracle);
        entry.kept.insert(at, kept);
        Ok(())
    }

    /// Add an open claim. SAMPLE rows enter only through `sample`. The claim's
    /// entries and table dimensions are checked against the member shapes.
    pub fn add_row(&mut self, group: u64, row: BatRow) -> Result<()> {
        if self.group(group)?.in_run || row.kind == BatRowKind::Sample {
            return Err(Error::InvalidSessionState);
        }
        self.check_row(group, &row)?;
        self.group_mut(group)?.rows.push(row);
        Ok(())
    }

    fn check_row(&self, group: u64, row: &BatRow) -> Result<()> {
        let entry = self.group(group)?;
        check_row_entries(row, |oracle| {
            if !entry.members.contains(&oracle) {
                return Err(Error::StatementMismatch);
            }
            Ok(self.oracle(oracle)?.shape)
        })
    }

    pub fn layout(&self, group: u64) -> Result<BatGroupLayout> {
        let entry = self.group(group)?;
        let mut main_first = vec![None; entry.members.len()];
        let mut mask_first = vec![None; entry.members.len()];
        let main_used = self.lay(entry, BatPart::Main, &mut main_first)?;
        let mask_used = self.lay(entry, BatPart::Mask, &mut mask_first)?;
        let h = ceil_log2(mask_used);
        let k = ceil_log2(main_used).max(h).max(1);
        let members = entry
            .members
            .iter()
            .enumerate()
            .map(|(index, oracle)| {
                Ok(BatMemberLayout {
                    oracle: *oracle,
                    kept: entry.kept[index],
                    main_first: main_first[index],
                    mask_first: mask_first[index],
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(BatGroupLayout {
            group,
            k,
            h,
            log_ell: entry.code.log_ell(),
            log_ell_zk: entry.mask_code.log_ell(),
            members,
        })
    }

    fn lay(&self, group: &BatGroup, part: BatPart, first: &mut [Option<u64>]) -> Result<u64> {
        let mut order = Vec::new();
        for (index, oracle) in group.members.iter().enumerate() {
            let shape = self.oracle(*oracle)?.shape;
            if shape.has(part) && !(part == BatPart::Main && group.kept[index]) {
                order.push((index, shape.row_bits(part)));
            }
        }
        order.sort_by_key(|(_, bits)| std::cmp::Reverse(*bits));
        let mut next = 0u64;
        for (index, bits) in order {
            first[index] = Some(next);
            let block = 1u64.checked_shl(bits).ok_or(Error::ResourceLimit)?;
            next = next.checked_add(block).ok_or(Error::ResourceLimit)?;
        }
        Ok(next)
    }

    /// Freeze the group and return its rows in the fixed class order. The group
    /// cannot change after this call. Open mask entries are made unavailable;
    /// this is a state placeholder only and carries no hiding guarantee.
    pub fn begin_run(&mut self, group: u64) -> Result<BatRunInput> {
        let layout = self.layout(group)?;
        let (members, mut plain, mut generator) = {
            let entry = self.group_mut(group)?;
            if entry.in_run || entry.members.is_empty() {
                return Err(Error::InvalidSessionState);
            }
            // The frozen run owns the registered rows; moving them out keeps
            // one representation of the claim set instead of duplicating it
            // while the group is closed.
            let rows = std::mem::take(&mut entry.rows);
            let mut plain = Vec::new();
            let mut generator = Vec::new();
            for row in rows {
                match row.kind {
                    BatRowKind::Plain => plain.push(row),
                    BatRowKind::Generator => generator.push(row),
                    BatRowKind::Sample => {}
                }
            }
            (entry.members.clone(), plain, generator)
        };
        for oracle in &members {
            let entry = self.oracle(*oracle)?;
            if entry.shape.answers > 0
                && entry.sample_rows.len() != entry.shape.answers as usize
            {
                return Err(Error::ShapeMismatch);
            }
        }
        let mut rows = Vec::new();
        rows.append(&mut plain);
        for oracle in &members {
            rows.extend(std::mem::take(
                &mut self.oracle_mut(*oracle)?.sample_rows,
            ));
        }
        rows.append(&mut generator);
        for oracle in &members {
            self.oracle_mut(*oracle)?.mask_open = false;
        }
        let entry = self.group_mut(group)?;
        entry.in_run = true;
        Ok(BatRunInput {
            layout,
            rows: Arc::new(rows),
        })
    }

    /// The current group mask placeholder. It has no hiding guarantee; after
    /// begin_run it is unavailable and cannot be drawn again.
    pub fn take_mask_slot(&mut self, oracle: u64) -> Result<u64> {
        let entry = self.oracle_mut(oracle)?;
        if !entry.mask_open {
            return Err(Error::InvalidSessionState);
        }
        let mask_code = entry.shape.mask_code.ok_or(Error::ShapeMismatch)?;
        let len = (entry.shape.mask_rows as usize)
            .checked_mul(mask_code.ell as usize)
            .ok_or(Error::ResourceLimit)?;
        let next = entry.mask_used.len() as u64;
        if next >= len as u64 {
            return Err(Error::ResourceLimit);
        }
        entry.mask_used.insert(next);
        Ok(next)
    }

    /// Shape-less candidate mapping to the isolated FOLD fixture. It fails
    /// closed unless the FOLD candidate checker accepts the layout and its
    /// member-order implied main/mask offsets are exactly BAT's actual
    /// main_first/mask_first; a space check alone is not enough. The public
    /// handoff entry therefore does not silently return a layout outside the
    /// supported fixture subset. The result still needs the full checked
    /// handoff for the shape code widths; it is not a general GroupLayout
    /// handoff or a production layout source.
    pub fn to_fold_layout(&self, group: u64) -> Result<FoldGroupLayout> {
        let layout = self.layout(group)?;
        let entry = self.group(group)?;
        let members = layout
            .members
            .iter()
            .map(|member| {
                let shape = self.oracle(member.oracle)?.shape;
                Ok(FoldMemberLayout {
                    oracle: member.oracle,
                    main_rows: shape.main_rows,
                    mask_rows: shape.mask_rows,
                    kept: member.kept,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let out = FoldGroupLayout {
            group,
            k: layout.k,
            h: layout.h,
            members,
        };
        let fixture_shape = FoldShape {
            code: entry.code,
            mask_code: entry.mask_code,
            k: out.k,
            h: out.h,
            mask_leaf_class: 0,
        };
        validate_fold_mask_shape(&fixture_shape)?;
        if out.fault(&fixture_shape).is_some() {
            return Err(Error::InvalidConfig);
        }
        check_fold_fixture_offsets(&layout, &out)?;
        Ok(out)
    }

    /// Convert and validate the handoff to the isolated FOLD fixture. This is
    /// a restricted fixture handoff only: it accepts a layout only when the
    /// isolated FOLD candidate checker passes and that checker's member-order
    /// implied main/mask first offsets equal BAT's actual main_first/mask_first.
    /// A legal BAT layout whose descending blocks that checker rejects is
    /// outside this S2 subset and fails closed here; it is not a general
    /// GroupLayout handoff. The only BAT layout derivation is
    /// `BatStatement::layout`; production must rebuild the same
    /// members/first offsets/k/h/kept from an authenticated statement, and must
    /// not introduce a second layout authority.
    pub fn to_fold_layout_checked(
        &self,
        group: u64,
        shape: &FoldShape,
    ) -> Result<FoldGroupLayout> {
        validate_fold_mask_shape(shape)?;
        let entry = self.group(group)?;
        if shape.code != entry.code || shape.mask_code != entry.mask_code {
            return Err(Error::StatementMismatch);
        }
        let layout = self.to_fold_layout(group)?;
        if shape.k != layout.k || shape.h != layout.h || layout.fault(shape).is_some() {
            return Err(Error::InvalidConfig);
        }
        Ok(layout)
    }

    /// Convert a combined group row to the isolated FOLD input row. The full
    /// group main/mask vectors are already in the same virtual order; kept
    /// member tables are compacted to their real rows, and every virtual
    /// padding entry must be zero.
    pub fn to_fold_input_row(
        &self,
        row: BatGroupRow,
        layout: &BatGroupLayout,
    ) -> Result<FoldInputRow> {
        if row.group != layout.group
            || self.layout(layout.group)? != *layout
            || row.main.len() != layout.group_len(BatPart::Main)?
            || row.mask.len() != layout.group_len(BatPart::Mask)?
        {
            return Err(Error::StatementMismatch);
        }
        let BatGroupRow {
            group,
            main,
            mask,
            kept,
            target,
        } = row;
        let mut compact_kept = Vec::new();
        for form in &kept {
            let member = layout
                .members
                .iter()
                .find(|member| member.oracle == form.oracle)
                .ok_or(Error::StatementMismatch)?;
            if !member.kept {
                return Err(Error::StatementMismatch);
            }
            let shape = self.oracle(form.oracle)?.shape;
            if form.coefficients.len() != shape.table_len(BatPart::Main)? {
                return Err(Error::ShapeMismatch);
            }
            let row_bits = shape.row_bits(BatPart::Main);
            let ell = 1usize
                .checked_shl(shape.log_ell(BatPart::Main))
                .ok_or(Error::ResourceLimit)?;
            let real_rows = usize::try_from(shape.main_rows).map_err(|_| Error::ResourceLimit)?;
            let mut compact = Vec::with_capacity(real_rows * ell);
            for real_row in 0..real_rows {
                for place in 0..ell {
                    let at = real_row + (place << row_bits);
                    compact.push(*form.coefficients.get(at).ok_or(Error::ShapeMismatch)?);
                }
            }
            let virtual_rows = 1usize
                .checked_shl(row_bits)
                .ok_or(Error::ResourceLimit)?;
            for real_row in real_rows..virtual_rows {
                for place in 0..ell {
                    let at = real_row + (place << row_bits);
                    if form.coefficients[at] != Fp3::ZERO {
                        return Err(Error::StatementMismatch);
                    }
                }
            }
            compact_kept.push(FoldKeptForm {
                oracle: form.oracle,
                coefficients: compact,
            });
        }
        Ok(FoldInputRow {
            group,
            k: layout.k,
            h: layout.h,
            members: layout.members.iter().map(|member| member.oracle).collect(),
            coefficients: main,
            mask_coefficients: mask,
            kept: compact_kept,
            target,
        })
    }
}

fn check_row_entries<F>(row: &BatRow, lookup: F) -> Result<()>
where
    F: Fn(u64) -> Result<BatOracleShape>,
{
    for term in &row.terms {
        let shape = lookup(term.oracle)?;
        if !shape.has(term.part) || term.values.len() != shape.table_len(term.part)? {
            return Err(Error::ShapeMismatch);
        }
    }
    for unit in &row.units {
        let shape = lookup(unit.oracle)?;
        if !shape.has(unit.part) || unit.index >= shape.table_len(unit.part)? as u64 {
            return Err(Error::InvalidIndex);
        }
    }
    for run in &row.runs {
        let shape = lookup(run.oracle)?;
        if !shape.has(run.part) || run.step == 0 || run.values.is_empty() {
            return Err(Error::ShapeMismatch);
        }
        let len = shape.table_len(run.part)? as u64;
        if run.first >= len
            || (run.values.len() as u64 - 1) > (len - 1 - run.first) / run.step
        {
            return Err(Error::InvalidIndex);
        }
    }
    for form in &row.forms {
        let shape = lookup(form.oracle)?;
        if !shape.has(BatPart::Main) || form.values.len() != shape.table_len(BatPart::Main)? {
            return Err(Error::ShapeMismatch);
        }
    }
    Ok(())
}

/// The first-row offsets that the isolated FOLD fixture's local member-order
/// alignment would imply for one table. This is a candidate checker's
/// coordinate convention, not a second production layout.
fn fold_fixture_firsts(
    layout: &FoldGroupLayout,
    part: BatPart,
) -> Result<Vec<Option<u64>>> {
    let part = match part {
        BatPart::Main => FoldPart::Main,
        BatPart::Mask => FoldPart::Mask,
    };
    fold_member_firsts(layout, part)
}

/// Require the BAT layout's actual first offsets to equal the FOLD fixture
/// checker's member-order offsets for both main and mask tables. A space check
/// alone does not establish coordinate equality.
fn check_fold_fixture_offsets(
    bat: &BatGroupLayout,
    fold: &FoldGroupLayout,
) -> Result<()> {
    if bat.members.len() != fold.members.len()
        || bat
            .members
            .iter()
            .zip(&fold.members)
            .any(|(left, right)| left.oracle != right.oracle)
    {
        return Err(Error::StatementMismatch);
    }
    let main = fold_fixture_firsts(fold, BatPart::Main)?;
    let mask = fold_fixture_firsts(fold, BatPart::Mask)?;
    for (index, member) in bat.members.iter().enumerate() {
        if main[index] != member.main_first || mask[index] != member.mask_first {
            return Err(Error::InvalidConfig);
        }
    }
    Ok(())
}

/// The isolated chain session. BAT's prover sends no message: the round is an
/// empty algebraic round, then the planned row count is checked and x is drawn
/// from the same chain. The planned row count is a trusted fixture, not proof
/// input; no PoW, real plan or production parameters are connected. The
/// supplied statement, layout, mask state and chain are unauthenticated
/// fixtures: this session is prohibited from production wiring until a
/// statement-derived plan, authentication and parameters exist. The ordinary
/// fixed order is the only path here; a code-switch caller that fixes
/// positions before x and makes opening claims after x is a separate exception
/// and is not represented by this session.
pub struct BatSession {
    statement: BatStatement,
    chain: Chain,
    planned_rows: u64,
    pending: Option<BatRunInput>,
    last_x: Option<Fp3>,
}

impl BatSession {
    pub fn new(statement: BatStatement, chain: Chain, planned_rows: u64) -> Result<Self> {
        Ok(Self {
            statement,
            chain,
            planned_rows,
            pending: None,
            last_x: None,
        })
    }

    pub fn statement(&self) -> &BatStatement {
        &self.statement
    }

    pub fn statement_mut(&mut self) -> &mut BatStatement {
        &mut self.statement
    }

    pub fn chain_value(&self) -> &Digest {
        self.chain.value()
    }

    /// The challenge drawn by the last completed run, for isolated arithmetic
    /// checks. It is not an input to the session.
    pub fn last_x(&self) -> Option<Fp3> {
        self.last_x
    }

    /// Transfers the advanced chain back to the next stage. The session is
    /// consumed, so it cannot be reused; calling before a completed run is
    /// rejected. This is an ownership transfer, not a copied digest.
    pub fn into_chain(self) -> Result<Chain> {
        if self.last_x.is_none() || self.pending.is_some() {
            return Err(Error::InvalidSessionState);
        }
        Ok(self.chain)
    }

    pub fn begin_run(&mut self, group: u64) -> Result<BatRunInput> {
        if self.pending.is_some() {
            return Err(Error::InvalidSessionState);
        }
        let input = self.statement.begin_run(group)?;
        self.pending = Some(input.clone());
        Ok(input)
    }

    pub fn add_row(&mut self, group: u64, row: BatRow) -> Result<()> {
        self.statement.add_row(group, row)
    }

    pub fn join(&mut self, group: u64, oracle: u64, kept: bool) -> Result<()> {
        self.statement.join(group, oracle, kept)
    }

    /// Close the BAT round after the input is frozen, check the planned count,
    /// draw x from the same chain and combine. `combine` does not re-check the
    /// time or source of x; the session's call order is the isolated boundary.
    pub fn finish_run(&mut self, input: BatRunInput) -> Result<BatGroupRow> {
        let pending = self.pending.take().ok_or(Error::InvalidSessionState)?;
        if pending.layout != input.layout || !Arc::ptr_eq(&pending.rows, &input.rows) {
            return Err(Error::StatementMismatch);
        }
        let prefix = self.chain.message().prefix(KIND_ALGEBRAIC, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        if input.rows.len() as u64 != self.planned_rows {
            return Err(Error::StatementMismatch);
        }
        let x = self.chain.draw_e()?;
        self.last_x = Some(x);
        combine(&self.statement, &input.layout, input.rows.as_slice(), x)
    }

    pub fn run(&mut self, group: u64) -> Result<BatGroupRow> {
        let input = self.begin_run(group)?;
        self.finish_run(input)
    }
}

pub fn group_index(index: u64, row_bits: u32, first: u64, group_bits: u32) -> Result<u64> {
    if row_bits >= 64 || group_bits >= 64 {
        return Err(Error::InvalidConfig);
    }
    let row = index & ((1u64 << row_bits) - 1);
    let place = index >> row_bits;
    first
        .checked_add(row)
        .and_then(|value| value.checked_add(place.checked_shl(group_bits)?))
        .ok_or(Error::ResourceLimit)
}

/// Combine the frozen rows by powers of x. Every occurrence of a claim is
/// accumulated. The function does not check when or from where x was drawn:
/// the ordinary session draws it from the chain, while a code-switch caller
/// may combine directly after its own x. The statement, layout and rows here
/// are unauthenticated fixtures, not a production wiring.
///
/// Folded main tables are scattered into disjoint place blocks of the group
/// row in parallel; every other term keeps the item-by-item path. Exact field
/// addition is order-independent, so the combined values are unchanged.
pub fn combine(
    statement: &BatStatement,
    layout: &BatGroupLayout,
    rows: &[BatRow],
    x: Fp3,
) -> Result<BatGroupRow> {
    combine_with_min(statement, layout, rows, x, COMBINE_PARALLEL_MIN)
}

fn combine_with_min(
    statement: &BatStatement,
    layout: &BatGroupLayout,
    rows: &[BatRow],
    x: Fp3,
    parallel_min: usize,
) -> Result<BatGroupRow> {
    if statement.layout(layout.group)? != *layout {
        return Err(Error::StatementMismatch);
    }
    let group_bits = layout.group_row_bits(BatPart::Main);
    let chunk_len = 1usize
        .checked_shl(group_bits)
        .ok_or(Error::ResourceLimit)?;
    let mut out = BatGroupRow {
        group: layout.group,
        main: vec![Fp3::ZERO; layout.group_len(BatPart::Main)?],
        mask: vec![Fp3::ZERO; layout.group_len(BatPart::Mask)?],
        kept: Vec::new(),
        target: Fp3::ZERO,
    };
    // Powers follow claim order so every row keeps the same x power.
    let mut powers = Vec::with_capacity(rows.len());
    let mut power = Fp3::ONE;
    for _ in rows {
        powers.push(power);
        power = power.mul(&x);
    }
    let mut scatter: Vec<Vec<MainScatter<'_>>> = Vec::with_capacity(rows.len());
    let mut units: Vec<Vec<MainUnitScatter>> = Vec::with_capacity(rows.len());
    for (row, scale) in rows.iter().zip(&powers) {
        let mut row_scatter = Vec::new();
        let mut row_units = Vec::new();
        for term in &row.terms {
            let shape = statement.oracle(term.oracle)?.shape;
            let member = layout
                .members
                .iter()
                .find(|member| member.oracle == term.oracle)
                .ok_or(Error::StatementMismatch)?;
            if term.part == BatPart::Main && !member.kept {
                if !shape.has(term.part) || term.values.len() != shape.table_len(term.part)? {
                    return Err(Error::ShapeMismatch);
                }
                row_scatter.push(main_scatter(
                    &term.values,
                    member.main_first.ok_or(Error::StatementMismatch)?,
                    shape.row_bits(term.part),
                    *scale,
                )?);
            } else {
                carry_table(
                    statement,
                    layout,
                    term.oracle,
                    term.part,
                    &term.values,
                    *scale,
                    &mut out,
                )?;
            }
        }
        for unit in &row.units {
            let shape = statement.oracle(unit.oracle)?.shape;
            let member = layout
                .members
                .iter()
                .find(|member| member.oracle == unit.oracle)
                .ok_or(Error::StatementMismatch)?;
            let len = shape.table_len(unit.part)?;
            if !shape.has(unit.part) || unit.index >= len as u64 {
                return Err(Error::InvalidIndex);
            }
            if unit.part == BatPart::Main && !member.kept {
                row_units.push(main_unit_scatter(
                    member.main_first.ok_or(Error::StatementMismatch)?,
                    shape.row_bits(unit.part),
                    unit.index,
                    scale.mul(&unit.coefficient),
                )?);
            } else {
                carry_entry(
                    statement,
                    layout,
                    unit.oracle,
                    unit.part,
                    unit.index,
                    unit.coefficient,
                    *scale,
                    &mut out,
                )?;
            }
        }
        for run in &row.runs {
            if run.step == 0 {
                return Err(Error::ShapeMismatch);
            }
            let shape = statement.oracle(run.oracle)?.shape;
            let member = layout
                .members
                .iter()
                .find(|member| member.oracle == run.oracle)
                .ok_or(Error::StatementMismatch)?;
            let len = shape.table_len(run.part)?;
            for (offset, value) in run.values.iter().enumerate() {
                let index = run
                    .first
                    .checked_add(
                        run.step
                            .checked_mul(offset as u64)
                            .ok_or(Error::ResourceLimit)?,
                    )
                    .ok_or(Error::ResourceLimit)?;
                if !shape.has(run.part) || index >= len as u64 {
                    return Err(Error::InvalidIndex);
                }
                if run.part == BatPart::Main && !member.kept {
                    row_units.push(main_unit_scatter(
                        member.main_first.ok_or(Error::StatementMismatch)?,
                        shape.row_bits(run.part),
                        index,
                        scale.mul(value),
                    )?);
                } else {
                    carry_entry(
                        statement,
                        layout,
                        run.oracle,
                        run.part,
                        index,
                        *value,
                        *scale,
                        &mut out,
                    )?;
                }
            }
        }
        for form in &row.forms {
            let shape = statement.oracle(form.oracle)?.shape;
            let member = layout
                .members
                .iter()
                .find(|member| member.oracle == form.oracle)
                .ok_or(Error::StatementMismatch)?;
            if !member.kept {
                if !shape.has(BatPart::Main)
                    || form.values.len() != shape.table_len(BatPart::Main)?
                {
                    return Err(Error::ShapeMismatch);
                }
                row_scatter.push(main_scatter(
                    &form.values,
                    member.main_first.ok_or(Error::StatementMismatch)?,
                    shape.row_bits(BatPart::Main),
                    *scale,
                )?);
            } else {
                carry_table(
                    statement,
                    layout,
                    form.oracle,
                    BatPart::Main,
                    &form.values,
                    *scale,
                    &mut out,
                )?;
            }
        }
        out.target = out.target.add(&scale.mul(&row.target));
        scatter.push(row_scatter);
        units.push(row_units);
    }
    let scatter_at = |place: usize, chunk: &mut [Fp3]| -> Result<()> {
        for (row_scatter, row_units) in scatter.iter().zip(&units) {
            for term in row_scatter {
                if place >= term.places {
                    continue;
                }
                let lo = place * term.rows_per_place;
                let hi = lo + term.rows_per_place;
                let target = chunk
                    .get_mut(term.first..term.first + term.rows_per_place)
                    .ok_or(Error::InvalidIndex)?;
                for (slot, value) in target.iter_mut().zip(&term.values[lo..hi]) {
                    *slot = slot.add(&term.scale.mul(value));
                }
            }
            for unit in row_units {
                if unit.place != place {
                    continue;
                }
                let slot = chunk
                    .get_mut(unit.first + unit.row)
                    .ok_or(Error::InvalidIndex)?;
                *slot = slot.add(&unit.scaled);
            }
        }
        Ok(())
    };
    if out.main.len() >= parallel_min && chunk_len <= out.main.len() {
        out.main
            .par_chunks_mut(chunk_len)
            .enumerate()
            .try_for_each(|(place, chunk)| scatter_at(place, chunk))?;
    } else {
        for (place, chunk) in out.main.chunks_mut(chunk_len).enumerate() {
            scatter_at(place, chunk)?;
        }
    }
    Ok(out)
}

/// Folded-main terms use the parallel block scatter from this group-row
/// length upward; smaller groups keep the sequential chunk loop.
const COMBINE_PARALLEL_MIN: usize = 1 << 22;

/// One folded main-table coefficient vector and its destination block.
struct MainScatter<'a> {
    values: &'a [Fp3],
    first: usize,
    rows_per_place: usize,
    places: usize,
    scale: Fp3,
}

/// One folded main-table point coefficient.
struct MainUnitScatter {
    first: usize,
    place: usize,
    row: usize,
    scaled: Fp3,
}

fn main_scatter(
    values: &[Fp3],
    first: u64,
    row_bits: u32,
    scale: Fp3,
) -> Result<MainScatter<'_>> {
    let first = usize::try_from(first).map_err(|_| Error::ResourceLimit)?;
    let rows_per_place = 1usize
        .checked_shl(row_bits)
        .ok_or(Error::ResourceLimit)?;
    if values.len() % rows_per_place != 0 {
        return Err(Error::ShapeMismatch);
    }
    Ok(MainScatter {
        values,
        first,
        rows_per_place,
        places: values.len() / rows_per_place,
        scale,
    })
}

fn main_unit_scatter(
    first: u64,
    row_bits: u32,
    index: u64,
    scaled: Fp3,
) -> Result<MainUnitScatter> {
    let first = usize::try_from(first).map_err(|_| Error::ResourceLimit)?;
    let rows_per_place = 1u64
        .checked_shl(row_bits)
        .ok_or(Error::ResourceLimit)?;
    let place = index / rows_per_place;
    let row = index % rows_per_place;
    Ok(MainUnitScatter {
        first,
        place: usize::try_from(place).map_err(|_| Error::ResourceLimit)?,
        row: usize::try_from(row).map_err(|_| Error::ResourceLimit)?,
        scaled,
    })
}

fn kept_slot<'a>(
    out: &'a mut BatGroupRow,
    oracle: u64,
    len: usize,
) -> Result<&'a mut BatKeptForm> {
    let at = match out.kept.binary_search_by_key(&oracle, |form| form.oracle) {
        Ok(index) => index,
        Err(index) => {
            out.kept.insert(
                index,
                BatKeptForm {
                    oracle,
                    coefficients: vec![Fp3::ZERO; len],
                },
            );
            index
        }
    };
    if out.kept[at].coefficients.len() != len {
        return Err(Error::ShapeMismatch);
    }
    Ok(&mut out.kept[at])
}

fn carry_table(
    statement: &BatStatement,
    layout: &BatGroupLayout,
    oracle: u64,
    part: BatPart,
    values: &[Fp3],
    scale: Fp3,
    out: &mut BatGroupRow,
) -> Result<()> {
    let shape = statement.oracle(oracle)?.shape;
    let member = layout
        .members
        .iter()
        .find(|member| member.oracle == oracle)
        .ok_or(Error::StatementMismatch)?;
    if !shape.has(part) || values.len() != shape.table_len(part)? {
        return Err(Error::ShapeMismatch);
    }
    if part == BatPart::Main && member.kept {
        let form = kept_slot(out, oracle, values.len())?;
        for (slot, value) in form.coefficients.iter_mut().zip(values) {
            *slot = slot.add(&scale.mul(value));
        }
        return Ok(());
    }
    let first = match part {
        BatPart::Main => member.main_first,
        BatPart::Mask => member.mask_first,
    }
    .ok_or(Error::StatementMismatch)?;
    let bits = layout.group_row_bits(part);
    let target = match part {
        BatPart::Main => &mut out.main,
        BatPart::Mask => &mut out.mask,
    };
    for (index, value) in values.iter().enumerate() {
        let at = group_index(index as u64, shape.row_bits(part), first, bits)?;
        let slot = target
            .get_mut(at as usize)
            .ok_or(Error::InvalidIndex)?;
        *slot = slot.add(&scale.mul(value));
    }
    Ok(())
}

/// Add one compact row-major main-table coefficient vector to a combined
/// group row. The vector is stored as `row * ell + place`, the compact table
/// layout, and is mapped into the group row one entry at a time without
/// materializing the virtual table. A kept member is rejected: its compact
/// form would need the whole virtual table, and this bounded entry is only for
/// members whose table is folded into the group row.
pub fn carry_compact_table(
    statement: &BatStatement,
    layout: &BatGroupLayout,
    oracle: u64,
    main: &[Fp3],
    main_rows: u64,
    ell: u64,
    scale: Fp3,
    out: &mut BatGroupRow,
) -> Result<()> {
    let shape = statement.oracle(oracle)?.shape;
    let member = layout
        .members
        .iter()
        .find(|member| member.oracle == oracle)
        .ok_or(Error::StatementMismatch)?;
    if !shape.has(BatPart::Main) || member.kept {
        return Err(Error::UnsupportedPrimitive);
    }
    let ell_bits = 1u64
        .checked_shl(shape.log_ell(BatPart::Main))
        .ok_or(Error::ResourceLimit)?;
    if main_rows != shape.main_rows || ell != ell_bits {
        return Err(Error::ShapeMismatch);
    }
    let len = usize::try_from(
        main_rows
            .checked_mul(ell)
            .ok_or(Error::ResourceLimit)?,
    )
    .map_err(|_| Error::ResourceLimit)?;
    if main.len() != len {
        return Err(Error::ShapeMismatch);
    }
    let row_bits = shape.row_bits(BatPart::Main);
    let bits = layout.group_row_bits(BatPart::Main);
    let first = member.main_first.ok_or(Error::StatementMismatch)?;
    for (offset, value) in main.iter().enumerate() {
        if *value == Fp3::ZERO {
            continue;
        }
        let offset = offset as u64;
        let row = offset / ell;
        let place = offset % ell;
        let virtual_index = row
            .checked_add(place.checked_shl(row_bits).ok_or(Error::ResourceLimit)?)
            .ok_or(Error::ResourceLimit)?;
        let at = group_index(virtual_index, row_bits, first, bits)?;
        let slot = out.main.get_mut(at as usize).ok_or(Error::InvalidIndex)?;
        *slot = slot.add(&scale.mul(value));
    }
    Ok(())
}

fn carry_entry(
    statement: &BatStatement,
    layout: &BatGroupLayout,
    oracle: u64,
    part: BatPart,
    index: u64,
    coefficient: Fp3,
    scale: Fp3,
    out: &mut BatGroupRow,
) -> Result<()> {
    let shape = statement.oracle(oracle)?.shape;
    let member = layout
        .members
        .iter()
        .find(|member| member.oracle == oracle)
        .ok_or(Error::StatementMismatch)?;
    let len = shape.table_len(part)?;
    if !shape.has(part) || index >= len as u64 {
        return Err(Error::InvalidIndex);
    }
    let scaled = scale.mul(&coefficient);
    if part == BatPart::Main && member.kept {
        let form = kept_slot(out, oracle, len)?;
        form.coefficients[index as usize] = form.coefficients[index as usize].add(&scaled);
        return Ok(());
    }
    let first = match part {
        BatPart::Main => member.main_first,
        BatPart::Mask => member.mask_first,
    }
    .ok_or(Error::StatementMismatch)?;
    let at = group_index(index, shape.row_bits(part), first, layout.group_row_bits(part))?;
    let target = match part {
        BatPart::Main => &mut out.main,
        BatPart::Mask => &mut out.mask,
    };
    let slot = target
        .get_mut(at as usize)
        .ok_or(Error::InvalidIndex)?;
    *slot = slot.add(&scaled);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn code() -> CodeParams {
        build_code_plan(4, 8, 1).unwrap()
    }

    fn mask_code() -> CodeParams {
        build_code_plan(4, 8, 1).unwrap()
    }

    fn chain() -> Chain {
        Chain::begin(41, &Digest::ZERO, b"isolated bat test", &[], None).unwrap()
    }

    fn shape(oracle: u64, main_rows: u64, mask_rows: u64, answers: u32) -> BatOracleShape {
        BatOracleShape {
            oracle,
            main_rows,
            mask_rows,
            main_code: if main_rows > 0 { Some(code()) } else { None },
            mask_code: if mask_rows > 0 { Some(mask_code()) } else { None },
            answers,
        }
    }

    fn sample_row(target: u64) -> BatRow {
        let mut row = BatRow::new(BatRowKind::Sample);
        row.target = Fp3::from_u64(target);
        row
    }

    fn add_shape(st: &mut BatStatement, shape: BatOracleShape) -> u64 {
        let oracle = st.add_oracle(shape).unwrap();
        if shape.answers > 0 {
            let rows = (1..=shape.answers)
                .map(|answer| sample_row(100 + oracle * 10 + answer as u64))
                .collect();
            st.sample(oracle, rows).unwrap();
        }
        oracle
    }

    fn zeros(len: usize) -> Vec<Fp3> {
        vec![Fp3::ZERO; len]
    }

    fn one_at(len: usize, index: usize, value: u64) -> Vec<Fp3> {
        let mut values = zeros(len);
        values[index] = Fp3::from_u64(value);
        values
    }

    fn group_fixture() -> (BatStatement, u64) {
        let mut st = BatStatement::new();
        add_shape(&mut st, shape(0, 2, 1, 1));
        add_shape(&mut st, shape(1, 4, 2, 1));
        add_shape(&mut st, shape(2, 3, 1, 1));
        let group = st.new_group(code(), mask_code()).unwrap();
        st.join(group, 0, false).unwrap();
        st.join(group, 1, false).unwrap();
        st.join(group, 2, true).unwrap();
        (st, group)
    }

    fn fold_fixture() -> (BatStatement, u64) {
        let mut st = BatStatement::new();
        add_shape(&mut st, shape(0, 4, 2, 1));
        add_shape(&mut st, shape(1, 2, 1, 1));
        add_shape(&mut st, shape(2, 3, 1, 1));
        let group = st.new_group(code(), mask_code()).unwrap();
        st.join(group, 0, false).unwrap();
        st.join(group, 1, false).unwrap();
        st.join(group, 2, true).unwrap();
        (st, group)
    }

    #[test]
    fn layout_is_descending_blocks_and_kept_main_is_excluded() {
        let (st, group) = group_fixture();
        let layout = st.layout(group).unwrap();
        assert_eq!(layout.k, 3);
        assert_eq!(layout.h, 2);
        assert_eq!(layout.log_ell, 3);
        assert_eq!(layout.log_ell_zk, 3);
        let members = layout.members;
        assert_eq!(members[0].oracle, 0);
        assert_eq!(members[0].main_first, Some(4));
        assert_eq!(members[0].mask_first, Some(2));
        assert_eq!(members[1].oracle, 1);
        assert_eq!(members[1].main_first, Some(0));
        assert_eq!(members[1].mask_first, Some(0));
        assert_eq!(members[2].oracle, 2);
        assert!(members[2].kept);
        assert_eq!(members[2].main_first, None);
        assert_eq!(members[2].mask_first, Some(3));
    }

    #[test]
    fn begin_run_uses_class_order_and_closes_the_group() {
        let mut st = BatStatement::new();
        add_shape(&mut st, shape(0, 2, 1, 2));
        let group = st.new_group(code(), mask_code()).unwrap();
        st.join(group, 0, false).unwrap();
        let mut plain_first = BatRow::new(BatRowKind::Plain);
        plain_first.target = Fp3::from_u64(1);
        let mut generator_first = BatRow::new(BatRowKind::Generator);
        generator_first.target = Fp3::from_u64(2);
        let mut plain_second = BatRow::new(BatRowKind::Plain);
        plain_second.target = Fp3::from_u64(3);
        let mut generator_second = BatRow::new(BatRowKind::Generator);
        generator_second.target = Fp3::from_u64(4);
        st.add_row(group, plain_first).unwrap();
        st.add_row(group, generator_first).unwrap();
        st.add_row(group, plain_second).unwrap();
        st.add_row(group, generator_second).unwrap();
        assert!(st.add_row(group, sample_row(9)).is_err());
        let mask_before = st.take_mask_slot(0).unwrap();
        assert_eq!(mask_before, 0);
        let input = st.begin_run(group).unwrap();
        assert_eq!(input.rows.len(), 6);
        let kinds = input
            .rows
            .iter()
            .map(|row| row.kind)
            .collect::<Vec<_>>();
        assert_eq!(
            kinds,
            vec![
                BatRowKind::Plain,
                BatRowKind::Plain,
                BatRowKind::Sample,
                BatRowKind::Sample,
                BatRowKind::Generator,
                BatRowKind::Generator,
            ]
        );
        let targets = input
            .rows
            .iter()
            .map(|row| row.target)
            .collect::<Vec<_>>();
        assert_eq!(
            targets,
            vec![
                Fp3::from_u64(1),
                Fp3::from_u64(3),
                Fp3::from_u64(101),
                Fp3::from_u64(102),
                Fp3::from_u64(2),
                Fp3::from_u64(4),
            ]
        );
        assert!(st.add_row(group, BatRow::new(BatRowKind::Plain)).is_err());
        assert!(st.join(group, 0, false).is_err());
        assert!(st.begin_run(group).is_err());
        assert!(st.take_mask_slot(0).is_err());
    }

    #[test]
    fn combine_uses_x_powers_per_claim_and_kept_forms() {
        let (mut st, group) = group_fixture();
        let mut first = BatRow::new(BatRowKind::Plain);
        first.target = Fp3::from_u64(5);
        first.add_term(0, BatPart::Main, one_at(16, 5, 3));
        first.add_unit(1, BatPart::Mask, 3, Fp3::from_u64(2));
        first.add_run(2, BatPart::Main, 1, 2, vec![Fp3::from_u64(7), Fp3::from_u64(9)]);
        first.add_form(0, one_at(16, 0, 1));
        let mut second = BatRow::new(BatRowKind::Plain);
        second.target = Fp3::from_u64(6);
        second.add_term(1, BatPart::Main, one_at(32, 0, 5));
        second.add_term(2, BatPart::Main, one_at(32, 9, 11));
        st.add_row(group, first).unwrap();
        st.add_row(group, second).unwrap();
        let input = st.begin_run(group).unwrap();
        let x = Fp3::from_u64(7);
        let row = combine(&st, &input.layout, &input.rows, x).unwrap();
        let layout = input.layout;
        let mut expected_main = vec![Fp3::ZERO; 64];
        expected_main[21] = Fp3::from_u64(3);
        expected_main[4] = Fp3::from_u64(1);
        expected_main[0] = x.mul(&Fp3::from_u64(5));
        let mut expected_mask = vec![Fp3::ZERO; 32];
        expected_mask[5] = Fp3::from_u64(2);
        assert_eq!(row.main, expected_main);
        assert_eq!(row.mask, expected_mask);
        assert_eq!(
            row.target,
            Fp3::from_u64(5)
                .add(&x.mul(&Fp3::from_u64(6)))
                .add(&x.mul(&x).mul(&Fp3::from_u64(101)))
                .add(&x.mul(&x).mul(&x).mul(&Fp3::from_u64(111)))
                .add(&x.mul(&x).mul(&x).mul(&x).mul(&Fp3::from_u64(121)))
        );
        let kept = row.kept.iter().find(|form| form.oracle == 2).unwrap();
        let mut expected_kept = vec![Fp3::ZERO; 32];
        expected_kept[1] = Fp3::from_u64(7);
        expected_kept[3] = Fp3::from_u64(9);
        expected_kept[9] = x.mul(&Fp3::from_u64(11));
        assert_eq!(kept.coefficients, expected_kept);
        let reversed = combine(
            &st,
            &layout,
            &input.rows.iter().cloned().rev().collect::<Vec<_>>(),
            x,
        )
        .unwrap();
        assert_ne!(row.target, reversed.target);
    }

    fn session_fixture_with_chain(planned_rows: u64, chain: Chain) -> (BatSession, u64) {
        let mut st = BatStatement::new();
        add_shape(&mut st, shape(0, 2, 1, 2));
        let group = st.new_group(code(), mask_code()).unwrap();
        st.join(group, 0, false).unwrap();
        let mut plain = BatRow::new(BatRowKind::Plain);
        plain.target = Fp3::from_u64(3);
        let mut generator_first = BatRow::new(BatRowKind::Generator);
        generator_first.target = Fp3::from_u64(5);
        let mut generator_second = BatRow::new(BatRowKind::Generator);
        generator_second.target = Fp3::from_u64(7);
        st.add_row(group, plain).unwrap();
        st.add_row(group, generator_first).unwrap();
        st.add_row(group, generator_second).unwrap();
        (BatSession::new(st, chain, planned_rows).unwrap(), group)
    }

    fn session_fixture(planned_rows: u64) -> (BatSession, u64) {
        session_fixture_with_chain(planned_rows, chain())
    }

    #[test]
    fn session_draws_x_from_its_chain_and_keeps_repeated_generator_rows() {
        let (mut session, group) = session_fixture(5);
        let row = session.run(group).unwrap();
        let x = session.last_x().unwrap();
        let mut replay = chain();
        let prefix = replay.message().prefix(KIND_ALGEBRAIC, 0).unwrap();
        assert!(replay.step(&prefix, 0).unwrap());
        assert_eq!(x, replay.draw_e().unwrap());
        let expected = Fp3::from_u64(3)
            .add(&x.mul(&Fp3::from_u64(101)))
            .add(&x.mul(&x).mul(&Fp3::from_u64(102)))
            .add(&x.mul(&x).mul(&x).mul(&Fp3::from_u64(5)))
            .add(&x.mul(&x).mul(&x).mul(&x).mul(&Fp3::from_u64(7)));
        assert_eq!(row.target, expected);
        assert!(session.run(group).is_err());
    }

    #[test]
    fn session_rejects_wrong_plan_count_and_modified_frozen_input() {
        let (mut session, group) = session_fixture(4);
        assert!(session.run(group).is_err());

        let (mut session, group) = session_fixture(5);
        let input = session.begin_run(group).unwrap();
        let mut changed = input.clone();
        changed.layout.k += 1;
        assert!(session.finish_run(changed).is_err());

        let (mut session, group) = session_fixture(5);
        let input = session.begin_run(group).unwrap();
        let mut removed = input.clone();
        Arc::make_mut(&mut removed.rows).pop();
        assert!(session.finish_run(removed).is_err());

        let (mut session, group) = session_fixture(5);
        let input = session.begin_run(group).unwrap();
        let mut replaced = input.clone();
        Arc::make_mut(&mut replaced.rows)[0].target = Fp3::from_u64(99);
        assert!(session.finish_run(replaced).is_err());
    }

    #[test]
    fn session_freezes_rows_joins_and_mask_entries() {
        let (mut session, group) = session_fixture(5);
        let input = session.begin_run(group).unwrap();
        assert_eq!(input.rows.len(), 5);
        assert!(session
            .add_row(group, BatRow::new(BatRowKind::Plain))
            .is_err());
        assert!(session.join(group, 0, false).is_err());
        assert!(session
            .statement_mut()
            .sample(0, vec![sample_row(1), sample_row(2)])
            .is_err());
        assert!(session.statement_mut().take_mask_slot(0).is_err());
        assert!(session.statement_mut().add_row(99, BatRow::new(BatRowKind::Plain)).is_err());
    }

    #[test]
    fn sample_and_membership_checks_fail_closed() {
        let mut st = BatStatement::new();
        let id = st.add_oracle(shape(0, 2, 1, 2)).unwrap();
        let group = st.new_group(code(), mask_code()).unwrap();
        assert!(st.join(group, id, false).is_err());
        assert!(st.sample(id, vec![sample_row(1)]).is_err());
        st.sample(id, vec![sample_row(1), sample_row(2)]).unwrap();
        assert!(st.sample(id, vec![sample_row(1), sample_row(2)]).is_err());
        assert!(st.join(group, id, true).is_ok());
        assert!(st.join(group, id, false).is_err());
    }

    #[test]
    fn oracle_ids_follow_registration_order() {
        let mut st = BatStatement::new();
        assert!(st.add_oracle(shape(9, 2, 1, 1)).is_err());
        assert!(st.add_oracle(shape(1, 2, 1, 1)).is_err());
        assert!(st.add_oracle(shape(0, 2, 1, 1)).is_ok());
        assert!(st.add_oracle(shape(2, 2, 1, 1)).is_err());
        assert!(st.add_oracle(shape(1, 2, 1, 1)).is_ok());
    }

    #[test]
    fn sample_rows_must_reference_their_own_oracle() {
        let mut st = BatStatement::new();
        st.add_oracle(shape(0, 2, 1, 2)).unwrap();
        st.add_oracle(shape(1, 2, 0, 1)).unwrap();
        let valid = || {
            let mut row = BatRow::new(BatRowKind::Sample);
            row.target = Fp3::from_u64(7);
            row.add_term(0, BatPart::Main, vec![Fp3::ZERO; 16]);
            row.add_unit(0, BatPart::Mask, 0, Fp3::ONE);
            row.add_run(0, BatPart::Main, 1, 2, vec![Fp3::ONE, Fp3::ZERO]);
            row.add_form(0, vec![Fp3::ZERO; 16]);
            row
        };
        let mut other_oracle = valid();
        other_oracle.add_unit(1, BatPart::Main, 0, Fp3::ONE);
        assert!(st.sample(0, vec![valid(), other_oracle]).is_err());

        let mut bad_index = valid();
        bad_index.units.clear();
        bad_index.add_unit(0, BatPart::Main, 16, Fp3::ONE);
        assert!(st.sample(0, vec![valid(), bad_index]).is_err());

        let mut no_part = valid();
        no_part.units.clear();
        no_part.add_unit(1, BatPart::Mask, 0, Fp3::ONE);
        assert!(st.sample(0, vec![valid(), no_part]).is_err());

        let mut absent_part = BatRow::new(BatRowKind::Sample);
        absent_part.add_unit(1, BatPart::Mask, 0, Fp3::ONE);
        assert!(st.sample(1, vec![absent_part]).is_err());

        let mut wrong_kind = valid();
        wrong_kind.kind = BatRowKind::Plain;
        assert!(st.sample(0, vec![valid(), wrong_kind]).is_err());

        assert!(st.sample(0, vec![valid(), valid()]).is_ok());
    }

    #[test]
    fn two_oracles_keep_send_order_and_sample_powers() {
        let mut st = BatStatement::new();
        st.add_oracle(shape(0, 2, 1, 1)).unwrap();
        st.add_oracle(shape(1, 2, 1, 1)).unwrap();
        let mut sample_first = sample_row(101);
        sample_first.add_unit(0, BatPart::Main, 0, Fp3::from_u64(7));
        let mut sample_second = sample_row(111);
        sample_second.add_unit(1, BatPart::Main, 0, Fp3::from_u64(8));
        st.sample(0, vec![sample_first]).unwrap();
        st.sample(1, vec![sample_second]).unwrap();
        let group = st.new_group(code(), mask_code()).unwrap();
        st.join(group, 1, false).unwrap();
        st.join(group, 0, false).unwrap();
        let mut first = BatRow::new(BatRowKind::Plain);
        first.target = Fp3::from_u64(5);
        first.add_term(0, BatPart::Main, one_at(16, 5, 3));
        let mut second = BatRow::new(BatRowKind::Plain);
        second.target = Fp3::from_u64(6);
        second.add_term(1, BatPart::Main, one_at(16, 5, 4));
        st.add_row(group, first).unwrap();
        st.add_row(group, second).unwrap();
        let input = st.begin_run(group).unwrap();
        assert_eq!(input.layout.members[0].oracle, 0);
        assert_eq!(input.layout.members[1].oracle, 1);
        assert_eq!(input.layout.members[0].main_first, Some(0));
        assert_eq!(input.layout.members[1].main_first, Some(2));
        let x = Fp3::from_u64(9);
        let row = combine(&st, &input.layout, &input.rows, x).unwrap();
        let mut expected_main = vec![Fp3::ZERO; 32];
        expected_main[9] = Fp3::from_u64(3);
        expected_main[11] = x.mul(&Fp3::from_u64(4));
        expected_main[0] = x.mul(&x).mul(&Fp3::from_u64(7));
        expected_main[2] = x.mul(&x).mul(&x).mul(&Fp3::from_u64(8));
        assert_eq!(row.main, expected_main);
        assert!(row.mask.iter().all(|value| *value == Fp3::ZERO));
        assert_eq!(
            row.target,
            Fp3::from_u64(5)
                .add(&x.mul(&Fp3::from_u64(6)))
                .add(&x.mul(&x).mul(&Fp3::from_u64(101)))
                .add(&x.mul(&x).mul(&x).mul(&Fp3::from_u64(111)))
        );
    }

    #[test]
    fn entry_dimensions_and_kept_mask_only_are_checked() {
        let (mut st, group) = group_fixture();
        let mut bad_term = BatRow::new(BatRowKind::Plain);
        bad_term.add_term(0, BatPart::Main, vec![Fp3::ZERO; 15]);
        assert!(st.add_row(group, bad_term).is_err());
        let mut bad_unit = BatRow::new(BatRowKind::Plain);
        bad_unit.add_unit(0, BatPart::Mask, 99, Fp3::ONE);
        assert!(st.add_row(group, bad_unit).is_err());
        let mut bad_run = BatRow::new(BatRowKind::Plain);
        bad_run.add_run(0, BatPart::Main, 15, 1, vec![Fp3::ONE, Fp3::ZERO]);
        assert!(st.add_row(group, bad_run).is_err());
        let mut bad_form = BatRow::new(BatRowKind::Plain);
        bad_form.add_form(0, vec![Fp3::ZERO; 8]);
        assert!(st.add_row(group, bad_form).is_err());

        let mut mask_only = BatStatement::new();
        add_shape(&mut mask_only, shape(0, 0, 1, 1));
        let other = mask_only.new_group(code(), mask_code()).unwrap();
        assert!(mask_only.join(other, 0, true).is_err());
        assert!(mask_only.join(other, 0, false).is_ok());
    }

    #[test]
    fn wrong_group_and_layout_are_rejected() {
        let (st, group) = group_fixture();
        let layout = st.layout(group).unwrap();
        let mut row = BatRow::new(BatRowKind::Plain);
        row.target = Fp3::from_u64(1);
        let mut changed = layout.clone();
        changed.k += 1;
        assert!(combine(&st, &changed, &[row.clone()], Fp3::from_u64(2)).is_err());
        let mut other = st.layout(group).unwrap();
        other.group = 99;
        assert!(combine(&st, &other, &[], Fp3::from_u64(2)).is_err());
        let (mut session, group) = session_fixture(5);
        assert!(session
            .add_row(group + 1, BatRow::new(BatRowKind::Plain))
            .is_err());
    }

    #[test]
    fn duplicate_units_are_kept_with_their_own_x_powers() {
        let (st, group) = group_fixture();
        let layout = st.layout(group).unwrap();
        let mut first = BatRow::new(BatRowKind::Plain);
        first.add_unit(0, BatPart::Main, 0, Fp3::from_u64(2));
        let mut second = BatRow::new(BatRowKind::Plain);
        second.add_unit(0, BatPart::Main, 0, Fp3::from_u64(3));
        let x = Fp3::from_u64(5);
        let out = combine(&st, &layout, &[first, second], x).unwrap();
        let at = layout
            .members
            .iter()
            .find(|member| member.oracle == 0)
            .unwrap()
            .main_first
            .unwrap() as usize;
        assert_eq!(
            out.main[at],
            Fp3::from_u64(2).add(&x.mul(&Fp3::from_u64(3)))
        );
    }

    #[test]
    fn direct_combine_is_the_separate_code_switch_path() {
        let (st, group) = group_fixture();
        let layout = st.layout(group).unwrap();
        let mut row = BatRow::new(BatRowKind::Plain);
        row.target = Fp3::from_u64(4);
        let combined = combine(&st, &layout, &[row], Fp3::from_u64(11)).unwrap();
        assert_eq!(combined.target, Fp3::from_u64(4));
        assert!(combined.main.iter().all(|value| *value == Fp3::ZERO));
    }

    #[test]
    fn session_uses_the_current_chain_position_and_prior_prefix() {
        let mut advanced = chain();
        let prefix = advanced.message().prefix(KIND_ALGEBRAIC, 0).unwrap();
        assert!(advanced.step(&prefix, 0).unwrap());
        let (mut session, group) = session_fixture_with_chain(5, advanced.clone());
        session.run(group).unwrap();
        let x = session.last_x().unwrap();

        let mut replay = advanced;
        let prefix = replay.message().prefix(KIND_ALGEBRAIC, 0).unwrap();
        assert!(replay.step(&prefix, 0).unwrap());
        assert_eq!(x, replay.draw_e().unwrap());

        let mut wrong_prefix = Chain::begin(41, &Digest::ZERO, b"another prefix", &[], None).unwrap();
        let prefix = wrong_prefix.message().prefix(KIND_ALGEBRAIC, 0).unwrap();
        assert!(wrong_prefix.step(&prefix, 0).unwrap());
        let (mut wrong_session, wrong_group) = session_fixture_with_chain(5, wrong_prefix);
        wrong_session.run(wrong_group).unwrap();
        assert_ne!(wrong_session.last_x(), Some(x));
    }

    #[test]
    fn legal_descending_layout_is_outside_the_fixture_handoff_subset() {
        let mut st = BatStatement::new();
        add_shape(&mut st, shape(0, 1, 1, 1));
        add_shape(&mut st, shape(1, 4, 1, 1));
        add_shape(&mut st, shape(2, 1, 1, 1));
        let group = st.new_group(code(), mask_code()).unwrap();
        st.join(group, 0, false).unwrap();
        st.join(group, 1, false).unwrap();
        st.join(group, 2, false).unwrap();
        let layout = st.layout(group).unwrap();
        assert_eq!(layout.k, 3);
        assert_eq!(layout.h, 2);
        assert!(st.to_fold_layout(group).is_err());
        let shape = crate::pcs::request::fold::FoldShape {
            code: code(),
            mask_code: mask_code(),
            k: layout.k,
            h: layout.h,
            mask_leaf_class: 3,
        };
        assert!(st.to_fold_layout_checked(group, &shape).is_err());
    }

    #[test]
    fn fold_fixture_offsets_must_match_bat_offsets() {
        fn handoff(main_rows: &[u64], mask_rows: &[u64]) -> (BatGroupLayout, Result<()>) {
            let mut st = BatStatement::new();
            for (index, (main, mask)) in main_rows.iter().zip(mask_rows).enumerate() {
                add_shape(&mut st, shape(index as u64, *main, *mask, 1));
            }
            let group = st.new_group(code(), mask_code()).unwrap();
            for index in 0..main_rows.len() {
                st.join(group, index as u64, false).unwrap();
            }
            let layout = st.layout(group).unwrap();
            (layout, st.to_fold_layout(group).map(|_| ()))
        }
        let (layout, result) = handoff(&[1, 2], &[1, 2]);
        assert_eq!(layout.members[0].main_first, Some(2));
        assert_eq!(layout.members[1].main_first, Some(0));
        assert_eq!(layout.members[0].mask_first, Some(2));
        assert_eq!(layout.members[1].mask_first, Some(0));
        assert!(result.is_err());
        assert!(handoff(&[1, 2], &[2, 1]).1.is_err());
        assert!(handoff(&[2, 1], &[1, 2]).1.is_err());
        assert!(handoff(&[2, 2], &[2, 2]).1.is_ok());
        assert!(handoff(&[2, 1], &[2, 1]).1.is_ok());
    }

    #[test]
    fn candidate_output_converts_to_the_isolated_fold_input() {
        let (mut st, group) = fold_fixture();
        let mut row = BatRow::new(BatRowKind::Plain);
        row.target = Fp3::from_u64(9);
        row.add_term(2, BatPart::Main, one_at(32, 9, 4));
        row.add_unit(2, BatPart::Mask, 0, Fp3::from_u64(6));
        st.add_row(group, row).unwrap();
        let input = st.begin_run(group).unwrap();
        let combined = combine(&st, &input.layout, &input.rows, Fp3::from_u64(3)).unwrap();
        let preview = st.to_fold_layout(group).unwrap();
        let fold_shape = crate::pcs::request::fold::FoldShape {
            code: code(),
            mask_code: mask_code(),
            k: preview.k,
            h: preview.h,
            mask_leaf_class: 3,
        };
        let fold_layout = st.to_fold_layout_checked(group, &fold_shape).unwrap();
        let mut wrong_shape = fold_shape;
        wrong_shape.k += 1;
        assert!(st.to_fold_layout_checked(group, &wrong_shape).is_err());
        let mut broken = combined.clone();
        let form = broken.kept.iter_mut().find(|form| form.oracle == 2).unwrap();
        form.coefficients[3] = Fp3::ONE;
        assert!(st.to_fold_input_row(broken, &input.layout).is_err());
        let fold_row = st.to_fold_input_row(combined, &input.layout).unwrap();
        fold_row.check(&fold_shape, &fold_layout).unwrap();
    }

    #[test]
    fn finish_run_rejects_independent_content_identical_input() {
        let (mut session, group) = session_fixture(5);
        let input = session.begin_run(group).unwrap();
        let mut independent = input.clone();
        independent.rows = Arc::new((*input.rows).clone());
        assert_eq!(independent.layout, input.layout);
        assert_eq!(independent.rows.as_slice(), input.rows.as_slice());
        assert!(!Arc::ptr_eq(&independent.rows, &input.rows));
        assert!(matches!(
            session.finish_run(independent),
            Err(Error::StatementMismatch)
        ));
    }

    #[test]
    fn combine_parallel_matches_serial_across_threads() {
        let (mut st, group) = group_fixture();
        let mut first = BatRow::new(BatRowKind::Plain);
        first.target = Fp3::from_u64(5);
        first.add_term(0, BatPart::Main, one_at(16, 5, 3));
        first.add_unit(1, BatPart::Mask, 3, Fp3::from_u64(2));
        first.add_run(2, BatPart::Main, 1, 2, vec![Fp3::from_u64(7), Fp3::from_u64(9)]);
        first.add_form(0, one_at(16, 0, 1));
        let mut second = BatRow::new(BatRowKind::Plain);
        second.target = Fp3::from_u64(6);
        second.add_term(1, BatPart::Main, one_at(32, 0, 5));
        second.add_term(2, BatPart::Main, one_at(32, 9, 11));
        st.add_row(group, first).unwrap();
        st.add_row(group, second).unwrap();
        let input = st.begin_run(group).unwrap();
        let x = Fp3::from_u64(7);
        let serial = combine_with_min(&st, &input.layout, &input.rows, x, usize::MAX).unwrap();
        for threads in [1usize, 2, 64] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let parallel = pool.install(|| {
                combine_with_min(&st, &input.layout, &input.rows, x, 0).unwrap()
            });
            assert_eq!(serial, parallel, "threads {threads}");
        }
    }
}
