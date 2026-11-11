//! Local linear relation rows for G_u claims (foundation only).
//!
//! These builders construct the coefficient rows and public targets of the
//! three G_u relations on small in-memory tables. They are local relation
//! construction and mathematical checks, not a verifier: the real verifier
//! does not hold the private G_u tables, and the chain, group, opening and
//! soundness paths are not connected. In the full protocol the index
//! commitment/name/record, public parameters and plan come from the trusted
//! statement/configuration and are chain-bound; a G_u root is a prover claim
//! received by the verifier in its round, not trusted content; gamma and draws
//! are verifier chain challenges, not caller values; and columns must be
//! opened against the table's trusted commitment. Here these are explicit
//! fixtures; no function or type name claims that the source is authenticated.
//! A changed fixture input either changes a value/target, breaks the algebraic
//! relation with a fixed witness, or (only when a real authenticated source is
//! connected) would be rejected by that source. The last category is not
//! implemented in this isolated layer and marks the integration gate; no test
//! here claims it.

use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::error::{Error, Result};

use crate::pcs::request::code::factors;
use crate::pcs::request::field::{Fp, Fp3, E_DEGREE};
use crate::pcs::request::gu::{GuOracleShape, GuRegistry, GuShape};
use crate::pcs::request::table::{
    record_seeds, IndexRecord, SampleSeed,
};

/// A small in-memory G_u witness: the main table and, when present, the mask
/// table, each in row-major order. A test holds it; a real verifier does not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuWitness {
    pub main: Vec<Fp3>,
    pub mask: Vec<Fp3>,
}

/// The kind of relation a row belongs to. The three relations are separate;
/// they are never merged into one claim set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuRelationKind {
    Record,
    Column,
    Sample,
}

/// One dense coefficient row over a small G_u table plus its public target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuRelationRow {
    pub oracle: u64,
    pub group: u64,
    pub kind: GuRelationKind,
    pub main: Vec<Fp3>,
    pub mask: Vec<Fp3>,
    pub target: Fp3,
}

impl GuRelationRow {
    /// Evaluate the row on a witness in the test. This is not a verifier
    /// entry; it exists so the tests can compare the constructed coefficients
    /// with an independent target.
    pub fn evaluate(&self, witness: &GuWitness) -> Result<Fp3> {
        if witness.main.len() != self.main.len() || witness.mask.len() != self.mask.len() {
            return Err(Error::ShapeMismatch);
        }
        let mut value = Fp3::ZERO;
        for (coefficient, entry) in self.main.iter().zip(&witness.main) {
            value = value.add(&coefficient.mul(entry));
        }
        for (coefficient, entry) in self.mask.iter().zip(&witness.mask) {
            value = value.add(&coefficient.mul(entry));
        }
        Ok(value)
    }
}

fn ceil_log2(value: u64) -> u32 {
    if value <= 1 {
        0
    } else {
        64 - (value - 1).leading_zeros()
    }
}

/// eq(index, z) over the low `bits` bits of the index.
fn eq_index(index: u64, z: &[Fp3], bits: u32) -> Result<Fp3> {
    if z.len() < bits as usize {
        return Err(Error::ShapeMismatch);
    }
    let mut value = Fp3::ONE;
    for (t, coordinate) in z.iter().take(bits as usize).enumerate() {
        let factor = if (index >> t) & 1 == 1 {
            *coordinate
        } else {
            Fp3::ONE.sub(coordinate)
        };
        value = value.mul(&factor);
    }
    Ok(value)
}

/// The complete `eq(index, z)` vector over the Boolean cube in `O(2^bits)`.
/// The index convention is the same as [`eq_index`]: the lowest bit is the
/// first seed coordinate. Callers that fill a whole sample row use this
/// vector instead of one `eq_index` evaluation per entry.
fn eq_vector(z: &[Fp3], bits: u32) -> Result<Vec<Fp3>> {
    if z.len() < bits as usize {
        return Err(Error::ShapeMismatch);
    }
    if bits >= 30 {
        return Err(Error::ResourceLimit);
    }
    let mut values = vec![Fp3::ONE];
    for coordinate in z[..bits as usize].iter().rev() {
        let mut next = vec![Fp3::ZERO; values.len() * 2];
        for (index, value) in values.iter().enumerate() {
            next[2 * index] = value.mul(&Fp3::ONE.sub(coordinate));
            next[2 * index + 1] = value.mul(coordinate);
        }
        values = next;
    }
    Ok(values)
}

fn theta(coefficient: usize) -> Fp3 {
    match coefficient {
        0 => Fp3::ONE,
        1 => Fp3::new(Fp::ZERO, Fp::ONE, Fp::ZERO),
        2 => Fp3::new(Fp::ZERO, Fp::ZERO, Fp::ONE),
        _ => Fp3::ZERO,
    }
}

/// The virtual table index of folded-row vector entry `j` under the trusted
/// G_u tree shape's row/place layout. Virtual rows are padded to a power of
/// two; they are not stored in the compact table.
fn vector_index(shape: &GuShape, j: u64) -> Result<u64> {
    let row_bits = ceil_log2(shape.u.main_rows);
    let place_bits = shape.u.code.log_ell();
    if row_bits >= 64 || place_bits >= 64 {
        return Err(Error::InvalidConfig);
    }
    if !shape.rows_high && shape.u.main_rows != (1u64 << row_bits) {
        return Err(Error::ShapeMismatch);
    }
    Ok(if shape.rows_high {
        (j >> place_bits) | ((j & ((1u64 << place_bits) - 1)) << row_bits)
    } else {
        j
    })
}

/// The compact main-table offset of one entry of the folded vector under the
/// trusted G_u layout. `None` means the entry falls in a virtual padded row or
/// a place past the code length; it is not a table slot.
pub fn gu_vector_offset(shape: &GuShape, entry: u64) -> Result<Option<usize>> {
    if shape.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    let virtual_index = vector_index(shape, entry)?;
    physical_index(shape, virtual_index)
}

/// The compact row-major offset of a virtual table index: `row = v mod
/// 2^row_bits`, `place = v div 2^row_bits`, `offset = row * ell + place`.
/// Virtual padded rows and places past the code length are not stored.
fn physical_offset(rows: u64, row_bits: u32, len: u64, virtual_index: u64) -> Result<Option<usize>> {
    if row_bits >= 64 {
        return Err(Error::InvalidConfig);
    }
    let row = virtual_index & ((1u64 << row_bits) - 1);
    let place = virtual_index >> row_bits;
    if row >= rows || place >= len {
        return Ok(None);
    }
    let offset = row
        .checked_mul(len)
        .and_then(|value| value.checked_add(place))
        .ok_or(Error::ResourceLimit)?;
    usize::try_from(offset)
        .map(Some)
        .map_err(|_| Error::ResourceLimit)
}

fn physical_index(shape: &GuShape, virtual_index: u64) -> Result<Option<usize>> {
    physical_offset(
        shape.u.main_rows,
        ceil_log2(shape.u.main_rows),
        shape.u.code.ell,
        virtual_index,
    )
}

/// Add a coefficient through the shared virtual-to-physical map. A virtual
/// padded row or a place past the code is skipped; a pad that must land in a
/// real slot fails instead of being aliased.
fn add_entry(
    shape: &GuShape,
    coefficients: &mut [Fp3],
    j: u64,
    value: Fp3,
    required: bool,
) -> Result<()> {
    let virtual_index = vector_index(shape, j)?;
    match physical_index(shape, virtual_index)? {
        Some(index) => {
            coefficients[index] = coefficients[index].add(&value);
            Ok(())
        }
        None if required => Err(Error::ShapeMismatch),
        None => Ok(()),
    }
}

/// The verifier's registry entry must be the complete trusted shape, not a
/// lookalike with another group, tree, part or layout.
fn check_binding(shape: &GuShape, registry: &GuRegistry) -> Result<()> {
    if shape.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    let oracle = registry
        .get(shape.oracle)
        .ok_or(Error::StatementMismatch)?;
    if oracle.shape != *shape || !oracle.sampled {
        return Err(Error::StatementMismatch);
    }
    Ok(())
}

fn empty_main(u: &GuOracleShape) -> Result<Vec<Fp3>> {
    let len = u
        .main_rows
        .checked_mul(u.code.ell)
        .ok_or(Error::ResourceLimit)?;
    usize::try_from(len)
        .map(|len| vec![Fp3::ZERO; len])
        .map_err(|_| Error::ResourceLimit)
}

fn empty_mask(u: &GuOracleShape) -> Result<Vec<Fp3>> {
    let len = match u.mask_code {
        Some(code) => u
            .mask_rows
            .checked_mul(code.ell)
            .ok_or(Error::ResourceLimit)?,
        None => 0,
    };
    usize::try_from(len)
        .map(|len| vec![Fp3::ZERO; len])
        .map_err(|_| Error::ResourceLimit)
}

/// The FOLDEQ row-point equation weights from the received gamma: one entry
/// per index row, `eq_table` over the index row variables. This is the
/// relation's `eq_gamma`, not a FOLD round challenge.
pub fn foldeq_weights(gamma: &[Fp3], variables: u32) -> Result<Vec<Fp3>> {
    if variables >= 64 || gamma.len() != variables as usize {
        return Err(Error::ShapeMismatch);
    }
    let rows = 1usize
        .checked_shl(variables)
        .ok_or(Error::ResourceLimit)?;
    let mut weights = vec![Fp3::ONE; rows];
    for (variable, coordinate) in gamma.iter().enumerate() {
        let half = 1usize << variable;
        for index in 0..half {
            let high = weights[index].mul(coordinate);
            weights[index + half] = high;
            weights[index] = weights[index].sub(&high);
        }
    }
    Ok(weights)
}

/// The shared layout check: every real folded-message entry and every real
/// tail entry must map through the trusted vector layout to a real compact
/// main slot. A shape whose `fault()` is empty can still fail here when a real
/// entry would fall in a virtual padded row or a place past the code length.
pub fn gu_layout_fault(shape: &GuShape) -> Option<&'static str> {
    if shape.fault().is_some() {
        return Some("the G_u shape is not valid");
    }
    let first = match tail_first(&shape.index.code) {
        Ok(first) => first,
        Err(_) => return Some("the folded row's tail position overflows"),
    };
    let end = match first.checked_add(shape.index.code.r) {
        Some(end) => end,
        None => return Some("the folded row's tail end overflows"),
    };
    for entry in 0..shape.index.code.ell {
        match vector_index(shape, entry).and_then(|value| physical_index(shape, value)) {
            Ok(Some(_)) => {}
            _ => return Some("a folded message entry has no real main slot"),
        }
    }
    for entry in first..end {
        match vector_index(shape, entry).and_then(|value| physical_index(shape, value)) {
            Ok(Some(_)) => {}
            _ => return Some("a folded tail entry has no real main slot"),
        }
    }
    None
}

pub fn gu_sample_pad_offsets(shape: &GuShape) -> Result<(Vec<usize>, Vec<usize>)> {
    let mut main = Vec::new();
    let mut mask = Vec::new();
    if shape.u.mask_code.is_none() && shape.u.answers > 0 {
        let row_bits = ceil_log2(shape.u.main_rows);
        for answer in 0..shape.u.answers {
            let place = shape.u.code.ell - shape.u.answers as u64 + answer as u64;
            let virtual_index = (shape.u.main_rows - 1)
                .checked_add(place.checked_shl(row_bits).ok_or(Error::ResourceLimit)?)
                .ok_or(Error::ResourceLimit)?;
            let offset = physical_offset(
                shape.u.main_rows,
                row_bits,
                shape.u.code.ell,
                virtual_index,
            )?
            .ok_or(Error::ShapeMismatch)?;
            main.push(offset);
        }
    }
    if let Some(code) = shape.u.mask_code {
        let row_bits = ceil_log2(shape.u.mask_rows);
        for answer in 0..shape.u.answers {
            let place = code.ell - shape.u.answers as u64 + answer as u64;
            let virtual_index = (shape.u.mask_rows - 1)
                .checked_add(place.checked_shl(row_bits).ok_or(Error::ResourceLimit)?)
                .ok_or(Error::ResourceLimit)?;
            let offset =
                physical_offset(shape.u.mask_rows, row_bits, code.ell, virtual_index)?
                    .ok_or(Error::ShapeMismatch)?;
            mask.push(offset);
        }
    }
    Ok((main, mask))
}

/// Build the actual small G_u table from the folded message and its aligned
/// tail: each real folded entry is placed through the shared layout check, the
/// rest stays zero, the index pads inside the message are retained, and the
/// G_u sample pads are a different set of slots (main without mask, mask with
/// mask). Before SEND writes them, those reserved slots must remain zero; a
/// folded entry that would land there is rejected. This is a static fixture
/// construction, not a random or hidden commitment.
pub fn gu_table(shape: &GuShape, message: &[Fp3], tail: &[Fp3]) -> Result<GuWitness> {
    if gu_layout_fault(shape).is_some()
        || message.len() != shape.index.code.ell as usize
        || tail.len() != shape.index.code.r as usize
    {
        return Err(Error::ShapeMismatch);
    }
    let (main_pads, mask_pads) = gu_sample_pad_offsets(shape)?;
    let mut main = empty_main(&shape.u)?;
    let mask = empty_mask(&shape.u)?;
    let first = tail_first(&shape.index.code)?;
    for (entry, value) in message.iter().enumerate() {
        let offset = vector_index(shape, entry as u64)
            .and_then(|virtual_index| physical_index(shape, virtual_index))?
            .ok_or(Error::ShapeMismatch)?;
        if main_pads.contains(&offset) {
            return Err(Error::ShapeMismatch);
        }
        main[offset] = *value;
    }
    for (entry, value) in tail.iter().enumerate() {
        let offset = vector_index(shape, first + entry as u64)
            .and_then(|virtual_index| physical_index(shape, virtual_index))?
            .ok_or(Error::ShapeMismatch)?;
        if main_pads.contains(&offset) {
            return Err(Error::ShapeMismatch);
        }
        main[offset] = *value;
    }
    for offset in main_pads {
        if main[offset] != Fp3::ZERO {
            return Err(Error::ShapeMismatch);
        }
    }
    // The mask table is zero here; the G_u sample pads in it are written by
    // SEND later, not by this static construction.
    for offset in mask_pads {
        if mask[offset] != Fp3::ZERO {
            return Err(Error::ShapeMismatch);
        }
    }
    Ok(GuWitness { main, mask })
}

/// Build the record relation row for one answer. `name` is the table's index
/// name bytes; the record seeds are recomputed from it and the index shape,
/// and a supplied seed list that differs from the recomputation is rejected.
/// The equation weights `eq_gamma` are the FOLDEQ row point's weights, not a
/// FOLD round challenge.
pub fn record_row(
    shape: &GuShape,
    registry: &GuRegistry,
    record: &IndexRecord,
    name: &[u8],
    supplied_seeds: &[SampleSeed],
    eq_gamma: &[Fp3],
    answer: u32,
) -> Result<GuRelationRow> {
    check_binding(shape, registry)?;
    if gu_layout_fault(shape).is_some() {
        return Err(Error::InvalidConfig);
    }
    let answers = shape.index.answers as usize;
    let rows = usize::try_from(shape.index.rows()).map_err(|_| Error::ResourceLimit)?;
    if answer as usize >= answers
        || record.y.len() != rows.checked_mul(answers).ok_or(Error::ResourceLimit)?
        || eq_gamma.len() != rows
    {
        return Err(Error::ShapeMismatch);
    }
    let seeds = record_seeds(&shape.index, name, answers)?;
    if seeds != supplied_seeds {
        return Err(Error::StatementMismatch);
    }
    let seed = &seeds[answer as usize];
    let bits = shape.index.cube_bits();
    if seed.z.len() < bits as usize || seed.beta == Fp3::ZERO {
        return Err(Error::ShapeMismatch);
    }
    let mut main = empty_main(&shape.u)?;
    let mut mask = empty_mask(&shape.u)?;
    for entry in 0..shape.index.code.ell {
        add_entry(shape, &mut main, entry, eq_index(entry, &seed.z, bits)?, true)?;
    }
    for pad_answer in 0..shape.index.answers {
        for coefficient in 0..E_DEGREE {
            let place = shape.index.pad_place(pad_answer, coefficient);
            let under = eq_index(place, &seed.z, bits)?;
            let mut value = Fp3::ZERO.sub(&under);
            if pad_answer == answer {
                value = value.add(&seed.beta.mul(&theta(coefficient)));
            }
            add_entry(shape, &mut main, place, value, true)?;
        }
    }
    let mut target = Fp3::ZERO;
    for (row, weight) in eq_gamma.iter().enumerate() {
        target = target.add(&weight.mul(&record.y[row * answers + answer as usize]));
    }
    // The record relation is on the main table; the mask coefficients stay
    // zero in this skeleton.
    mask.iter_mut().for_each(|coefficient| *coefficient = Fp3::ZERO);
    Ok(GuRelationRow {
        oracle: shape.oracle,
        group: shape.group,
        kind: GuRelationKind::Record,
        main,
        mask,
        target,
    })
}

fn product_index(f: &[Fp], count: u32, index: u64) -> Fp {
    let mut value = Fp::ONE;
    for (t, factor) in f.iter().take(count as usize).enumerate() {
        if (index >> t) & 1 == 1 {
            value = value * *factor;
        }
    }
    value
}

fn code_log_ell(code: &crate::pcs::request::code::CodeParams) -> u32 {
    ceil_log2(code.ell)
}

fn code_log_r(code: &crate::pcs::request::code::CodeParams) -> u32 {
    ceil_log2(code.r)
}

fn tail_first(code: &crate::pcs::request::code::CodeParams) -> Result<u64> {
    let step = 1u64
        .checked_shl(code_log_r(code))
        .ok_or(Error::ResourceLimit)?;
    Ok(code
        .ell
        .checked_add(step - 1)
        .ok_or(Error::ResourceLimit)?
        / step
        * step)
}

fn generator_message(
    code: &crate::pcs::request::code::CodeParams,
    position: u64,
) -> Result<Vec<Fp>> {
    let log_dim = ceil_log2(
        code.ell
            .checked_add(code.r)
            .ok_or(Error::ResourceLimit)?,
    );
    let factors = factors(code.log_n, position, log_dim)?;
    let scale = product_index(&factors, log_dim, code.r);
    let mut message = Vec::with_capacity(code.ell as usize);
    for entry in 0..code.ell {
        message.push(product_index(&factors, code_log_ell(code), entry) * scale);
    }
    Ok(message)
}

fn generator_tail(
    code: &crate::pcs::request::code::CodeParams,
    position: u64,
) -> Result<Vec<Fp>> {
    let log_dim = ceil_log2(
        code.ell
            .checked_add(code.r)
            .ok_or(Error::ResourceLimit)?,
    );
    let factors = factors(code.log_n, position, log_dim)?;
    let scale = product_index(&factors, log_dim, 0);
    let mut tail = Vec::with_capacity(code.r as usize);
    for entry in 0..code.r {
        tail.push(product_index(&factors, code_log_r(code), entry) * scale);
    }
    Ok(tail)
}

/// Validate one provided column against a column-relation shape and fold it
/// at `eq_gamma` into the target. The caller has already checked the shape's
/// binding and layout; the column length, base-field limbs and weights are
/// checked here. This is the target half of `column_row`.
pub fn column_target(shape: &GuShape, column: &[Fp3], eq_gamma: &[Fp3]) -> Result<Fp3> {
    let rows = usize::try_from(shape.index.rows()).map_err(|_| Error::ResourceLimit)?;
    if column.len() != rows || eq_gamma.len() != rows {
        return Err(Error::ShapeMismatch);
    }
    if column
        .iter()
        .any(|value| value.c[1] != Fp::ZERO || value.c[2] != Fp::ZERO)
    {
        return Err(Error::ShapeMismatch);
    }
    let mut target = Fp3::ZERO;
    for (weight, value) in eq_gamma.iter().zip(column) {
        target = target.add(&weight.mul(value));
    }
    Ok(target)
}

/// Build one column-query relation row: the generator coefficients of the
/// indexed code at the drawn position, placed on the G_u main table, with the
/// target equal to the provided index column (a fixture; the opening is not
/// connected) folded at `eq_gamma`.
pub fn column_row(
    shape: &GuShape,
    registry: &GuRegistry,
    position: u64,
    column: &[Fp3],
    eq_gamma: &[Fp3],
) -> Result<GuRelationRow> {
    check_binding(shape, registry)?;
    if gu_layout_fault(shape).is_some() {
        return Err(Error::InvalidConfig);
    }
    if position >= shape.index.code.n() {
        return Err(Error::ShapeMismatch);
    }
    let target = column_target(shape, column, eq_gamma)?;
    let message = generator_message(&shape.index.code, position)?;
    let tail = generator_tail(&shape.index.code, position)?;
    let first = tail_first(&shape.index.code)?;
    let mut main = empty_main(&shape.u)?;
    let mut mask = empty_mask(&shape.u)?;
    for (entry, coefficient) in message.iter().enumerate() {
        add_entry(
            shape,
            &mut main,
            entry as u64,
            Fp3::from_base(*coefficient),
            true,
        )?;
    }
    for (entry, coefficient) in tail.iter().enumerate() {
        add_entry(
            shape,
            &mut main,
            first + entry as u64,
            Fp3::from_base(*coefficient),
            true,
        )?;
    }
    mask.iter_mut().for_each(|coefficient| *coefficient = Fp3::ZERO);
    Ok(GuRelationRow {
        oracle: shape.oracle,
        group: shape.group,
        kind: GuRelationKind::Column,
        main,
        mask,
        target,
    })
}

/// The trusted draw sequence and the claimed positions must agree item by
/// item: a permutation of the same multiset, a missing item or a reordering is
/// not the drawn relation.
pub fn validate_draw_sequence(trusted: &[u64], claimed: &[u64]) -> Result<()> {
    if trusted.is_empty() || trusted != claimed {
        return Err(Error::StatementMismatch);
    }
    Ok(())
}

/// The unique sorted positions of a draw list, for the one physical opening
/// that may be reused by every draw at the same position. This does not
/// decide the logical draw sequence.
pub fn unique_positions(trusted: &[u64]) -> Vec<u64> {
    let mut positions = trusted.to_vec();
    positions.sort_unstable();
    positions.dedup();
    positions
}

/// Build one row per logical draw in the trusted order. `fixture_columns` is
/// the one provided column per unique position, as a sorted strictly
/// increasing list; it is a fixture input and the real column opening and
/// authentication are not connected;
/// every logical draw, repeats included, reuses the column of its position.
/// A repeated position with two different columns is rejected before any
/// target comparison, so a zero equation weight cannot hide the difference.
fn checked_column_draws<'a>(
    shape: &GuShape,
    trusted_draws: &[u64],
    fixture_columns: &'a [(u64, Vec<Fp3>)],
) -> Result<Vec<(u64, &'a [Fp3])>> {
    if trusted_draws.is_empty() || fixture_columns.is_empty() {
        return Err(Error::StatementMismatch);
    }
    let expected_positions = unique_positions(trusted_draws);
    if fixture_columns.len() != expected_positions.len()
        || fixture_columns
            .iter()
            .map(|(position, _)| *position)
            .ne(expected_positions.iter().copied())
    {
        return Err(Error::StatementMismatch);
    }
    for (position, _) in fixture_columns {
        if *position >= shape.index.code.n() {
            return Err(Error::ShapeMismatch);
        }
    }
    let mut columns = Vec::with_capacity(trusted_draws.len());
    for position in trusted_draws {
        let (_, column) = fixture_columns
            .iter()
            .find(|(at, _)| at == position)
            .ok_or(Error::StatementMismatch)?;
        columns.push((*position, column.as_slice()));
    }
    Ok(columns)
}

pub fn column_rows_from_draws(
    shape: &GuShape,
    registry: &GuRegistry,
    trusted_draws: &[u64],
    fixture_columns: &[(u64, Vec<Fp3>)],
    eq_gamma: &[Fp3],
) -> Result<Vec<GuRelationRow>> {
    let columns = checked_column_draws(shape, trusted_draws, fixture_columns)?;
    columns
        .into_iter()
        .map(|(position, column)| column_row(shape, registry, position, column, eq_gamma))
        .collect()
}

/// The isolated plan-driven column entry: the trusted plan supplies the
/// column count and the index code domain depth, and the caller supplies a
/// fixture draw sequence and one provided column per unique position. The
/// sequence length must equal the planned count, every draw must be in the
/// index code's domain, and the unique column positions must equal the draw
/// deduplication. Logical rows follow the draw order (repeats included);
/// physical openings are the unique positions; neither is a hiding-budget
/// count. The chain source of the plan, draws and columns is not connected.
pub fn column_plan_rows(
    shape: &GuShape,
    registry: &GuRegistry,
    planned_columns: usize,
    index_log_n: u32,
    draws: &[u64],
    fixture_columns: &[(u64, Vec<Fp3>)],
    eq_gamma: &[Fp3],
) -> Result<Vec<GuRelationRow>> {
    let columns = column_plan_columns(shape, planned_columns, index_log_n, draws, fixture_columns)?;
    columns
        .into_iter()
        .map(|(position, column)| column_row(shape, registry, position, column, eq_gamma))
        .collect()
}

/// Validate the plan-driven draw and fixture-column list and resolve the
/// provided column of every logical draw, without building relation rows.
/// The checks are the ones `column_plan_rows` applies: planned count, index
/// domain, and the unique-position column list in draw order.
pub fn column_plan_columns<'a>(
    shape: &GuShape,
    planned_columns: usize,
    index_log_n: u32,
    draws: &[u64],
    fixture_columns: &'a [(u64, Vec<Fp3>)],
) -> Result<Vec<(u64, &'a [Fp3])>> {
    if planned_columns == 0 || draws.len() != planned_columns {
        return Err(Error::ShapeMismatch);
    }
    if index_log_n != shape.index.code.log_n || index_log_n >= 64 {
        return Err(Error::StatementMismatch);
    }
    let domain = 1u64 << index_log_n;
    if draws.iter().any(|position| *position >= domain) {
        return Err(Error::InvalidIndex);
    }
    checked_column_draws(shape, draws, fixture_columns)
}

/// The sample cube of one G_u oracle shape: the main and mask table bit
/// counts and the coordinates of a shared seed that the form reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleCube {
    pub has_main: bool,
    pub has_mask: bool,
    pub main_row_bits: u32,
    pub main_bits: u32,
    pub mask_row_bits: u32,
    pub mask_bits: u32,
    pub bits: u32,
}

pub fn sample_cube(shape: &GuShape) -> Result<SampleCube> {
    sample_cube_of(&shape.u)
}

/// The sample cube of an ordinary oracle message: its own code, rows, mask
/// and answer count. It is the same cube the G_u path uses.
pub fn sample_cube_of(u: &GuOracleShape) -> Result<SampleCube> {
    let main_bits = ceil_log2(u.main_rows)
        .checked_add(u.code.log_ell())
        .ok_or(Error::ResourceLimit)?;
    let mask_bits = match u.mask_code {
        Some(code) => ceil_log2(u.mask_rows)
            .checked_add(code.log_ell())
            .ok_or(Error::ResourceLimit)?,
        None => 0,
    };
    let both = u.main_rows > 0 && u.mask_code.is_some();
    Ok(SampleCube {
        has_main: u.main_rows > 0,
        has_mask: u.mask_code.is_some(),
        main_row_bits: ceil_log2(u.main_rows),
        main_bits,
        mask_row_bits: ceil_log2(u.mask_rows),
        mask_bits,
        bits: if both {
            main_bits
                .max(mask_bits)
                .checked_add(1)
                .ok_or(Error::ResourceLimit)?
        } else {
            main_bits
                .checked_add(mask_bits)
                .ok_or(Error::ResourceLimit)?
        },
    })
}

fn sample_front(seed: &SampleSeed, cube: &SampleCube, table_bits: u32, mask: bool) -> Result<Fp3> {
    if !(cube.has_main && cube.has_mask) {
        return Ok(Fp3::ONE);
    }
    if seed.z.len() < cube.bits as usize {
        return Err(Error::ShapeMismatch);
    }
    let top = seed.z[cube.bits as usize - 1];
    let mut factor = if mask {
        top
    } else {
        Fp3::ONE.sub(&top)
    };
    for coordinate in seed.z.iter().take(cube.bits as usize - 1).skip(table_bits as usize) {
        factor = factor.mul(&Fp3::ONE.sub(coordinate));
    }
    Ok(factor)
}

fn sample_pad(
    coefficients: &mut [Fp3],
    seed: &SampleSeed,
    answers: u32,
    target: u32,
    rows: u64,
    row_bits: u32,
    len: u64,
    bits: u32,
    front: Fp3,
) -> Result<()> {
    let each = 1u64;
    if len < each * answers as u64 {
        return Err(Error::ShapeMismatch);
    }
    for pad_answer in 0..answers {
        let place = len - each * (answers as u64) + each * pad_answer as u64;
        let virtual_index = (rows - 1)
            .checked_add(place.checked_shl(row_bits).ok_or(Error::ResourceLimit)?)
            .ok_or(Error::ResourceLimit)?;
        let under = front.mul(&eq_index(virtual_index, &seed.z, bits)?);
        let mut value = Fp3::ZERO.sub(&under);
        if pad_answer == target {
            value = value.add(&seed.beta.mul(&theta(0)));
        }
        let slot = physical_offset(rows, row_bits, len, virtual_index)?
            .ok_or(Error::ShapeMismatch)?;
        if slot >= coefficients.len() {
            return Err(Error::ShapeMismatch);
        }
        coefficients[slot] = coefficients[slot].add(&value);
    }
    Ok(())
}

/// Build the SEND sample relation row for one oracle and one answer from the
/// shared seed set. The seed set is shared by answer number across the
/// sampled oracles of one SEND message; this function uses `seeds[answer]` and
/// the oracle's own cube prefix. The target is the answer value received from
/// SEND; the real root/seed/answer transcript is not connected.
pub fn sample_row(
    shape: &GuShape,
    registry: &GuRegistry,
    seeds: &[SampleSeed],
    answer: u32,
    target: Fp3,
) -> Result<GuRelationRow> {
    check_binding(shape, registry)?;
    let (main, mask) = sample_coefficients(&shape.u, seeds, answer, target)?;
    Ok(GuRelationRow {
        oracle: shape.oracle,
        group: shape.group,
        kind: GuRelationKind::Sample,
        main,
        mask,
        target,
    })
}

/// The SEND sample row of an ordinary oracle message, built from the same
/// cube and helpers as the G_u path. The coefficients are over the oracle's
/// own main table (and mask rows, if any); the target is the received answer.
pub fn oracle_sample_row(
    oracle: u64,
    group: u64,
    u: &GuOracleShape,
    seeds: &[SampleSeed],
    answer: u32,
    target: Fp3,
) -> Result<GuRelationRow> {
    if u.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    let (main, mask) = sample_coefficients(u, seeds, answer, target)?;
    Ok(GuRelationRow {
        oracle,
        group,
        kind: GuRelationKind::Sample,
        main,
        mask,
        target,
    })
}

fn sample_coefficients(
    u: &GuOracleShape,
    seeds: &[SampleSeed],
    answer: u32,
    target: Fp3,
) -> Result<(Vec<Fp3>, Vec<Fp3>)> {
    if answer >= u.answers || seeds.len() < u.answers as usize {
        return Err(Error::ShapeMismatch);
    }
    let cube = sample_cube_of(u)?;
    let seed = &seeds[answer as usize];
    if seed.z.len() < cube.bits as usize || seed.beta == Fp3::ZERO {
        return Err(Error::ShapeMismatch);
    }
    let mut main = empty_main(u)?;
    let mut mask = empty_mask(u)?;
    if cube.has_main {
        let front = sample_front(seed, &cube, cube.main_bits, false)?;
        let count = 1u64
            .checked_shl(cube.main_bits)
            .ok_or(Error::ResourceLimit)?;
        let eq = eq_vector(&seed.z, cube.main_bits)?;
        if eq.len() as u64 != count {
            return Err(Error::ShapeMismatch);
        }
        for virtual_index in 0..count {
            if let Some(offset) = physical_offset(
                u.main_rows,
                cube.main_row_bits,
                u.code.ell,
                virtual_index,
            )? {
                main[offset] = front.mul(&eq[virtual_index as usize]);
            }
        }
        if !cube.has_mask {
            sample_pad(
                &mut main,
                seed,
                u.answers,
                answer,
                u.main_rows,
                cube.main_row_bits,
                u.code.ell,
                cube.main_bits,
                front,
            )?;
        }
    }
    if cube.has_mask {
        let code = u.mask_code.ok_or(Error::ShapeMismatch)?;
        let front = sample_front(seed, &cube, cube.mask_bits, true)?;
        let count = 1u64
            .checked_shl(cube.mask_bits)
            .ok_or(Error::ResourceLimit)?;
        let eq = eq_vector(&seed.z, cube.mask_bits)?;
        if eq.len() as u64 != count {
            return Err(Error::ShapeMismatch);
        }
        for virtual_index in 0..count {
            if let Some(offset) = physical_offset(
                u.mask_rows,
                cube.mask_row_bits,
                code.ell,
                virtual_index,
            )? {
                mask[offset] = front.mul(&eq[virtual_index as usize]);
            }
        }
        sample_pad(
            &mut mask,
            seed,
            u.answers,
            answer,
            u.mask_rows,
            cube.mask_row_bits,
            code.ell,
            cube.mask_bits,
            front,
        )?;
    }
    Ok((main, mask))
}

/// The SEND sample row of an ordinary oracle message in the table-index
/// convention: entry `index` is row `index mod 2^row_bits` at place
/// `index div 2^row_bits`, and the sample reads `eq(index, z)` on that index.
/// It is the form the switched oracle message and the ordinary SEND path use;
/// the G_u witness has its own compact row-major layout and keeps its own
/// builder.
pub fn oracle_sample_row_index(
    oracle: u64,
    group: u64,
    u: &GuOracleShape,
    seeds: &[SampleSeed],
    answer: u32,
    target: Fp3,
) -> Result<GuRelationRow> {
    if u.fault().is_some() || answer >= u.answers || seeds.len() < u.answers as usize {
        return Err(Error::ShapeMismatch);
    }
    let cube = sample_cube_of(u)?;
    let seed = &seeds[answer as usize];
    if seed.z.len() < cube.bits as usize || seed.beta == Fp3::ZERO {
        return Err(Error::ShapeMismatch);
    }
    let main_len = usize::try_from(
        u.main_rows
            .checked_mul(u.code.ell)
            .ok_or(Error::ResourceLimit)?,
    )
    .map_err(|_| Error::ResourceLimit)?;
    let mut main = vec![Fp3::ZERO; main_len];
    if cube.has_main {
        let front = sample_front(seed, &cube, cube.main_bits, false)?;
        for (index, value) in main.iter_mut().enumerate() {
            *value = front.mul(&eq_index(index as u64, &seed.z, cube.main_bits)?);
        }
        if !cube.has_mask {
            sample_pad_index(
                &mut main,
                seed,
                u.answers,
                answer,
                u.main_rows,
                cube.main_row_bits,
                u.code.ell,
                cube.main_bits,
                front,
            )?;
        }
    }
    let mask_len = match u.mask_code {
        Some(code) => usize::try_from(
            u.mask_rows
                .checked_mul(code.ell)
                .ok_or(Error::ResourceLimit)?,
        )
        .map_err(|_| Error::ResourceLimit)?,
        None => 0,
    };
    let mut mask = vec![Fp3::ZERO; mask_len];
    if cube.has_mask {
        let code = u.mask_code.ok_or(Error::ShapeMismatch)?;
        let front = sample_front(seed, &cube, cube.mask_bits, true)?;
        for (index, value) in mask.iter_mut().enumerate() {
            *value = front.mul(&eq_index(index as u64, &seed.z, cube.mask_bits)?);
        }
        sample_pad_index(
            &mut mask,
            seed,
            u.answers,
            answer,
            u.mask_rows,
            cube.mask_row_bits,
            code.ell,
            cube.mask_bits,
            front,
        )?;
    }
    Ok(GuRelationRow {
        oracle,
        group,
        kind: GuRelationKind::Sample,
        main,
        mask,
        target,
    })
}

fn sample_pad_index(
    coefficients: &mut [Fp3],
    seed: &SampleSeed,
    answers: u32,
    target: u32,
    rows: u64,
    row_bits: u32,
    len: u64,
    bits: u32,
    front: Fp3,
) -> Result<()> {
    if len < answers as u64 {
        return Err(Error::ShapeMismatch);
    }
    for pad_answer in 0..answers {
        let place = len - answers as u64 + pad_answer as u64;
        let index = (rows - 1)
            .checked_add(place.checked_shl(row_bits).ok_or(Error::ResourceLimit)?)
            .ok_or(Error::ResourceLimit)?;
        let under = front.mul(&eq_index(index, &seed.z, bits)?);
        let mut value = Fp3::ZERO.sub(&under);
        if pad_answer == target {
            value = value.add(&seed.beta.mul(&theta(0)));
        }
        let slot = usize::try_from(index).map_err(|_| Error::ResourceLimit)?;
        let entry = coefficients.get_mut(slot).ok_or(Error::ShapeMismatch)?;
        *entry = entry.add(&value);
    }
    Ok(())
}

/// One shared seed set for the sampled oracles of one SEND message: the
/// number of seeds is the largest answer count, each seed has the largest
/// cube's coordinates, and each oracle reads the prefix its own cube needs.
pub fn validate_shared_sample_seeds(shapes: &[&GuShape], seeds: &[SampleSeed]) -> Result<u32> {
    if shapes.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let mut bits = 0u32;
    let mut answers = 0u32;
    for shape in shapes {
        if shape.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        let cube = sample_cube(shape)?;
        bits = bits.max(cube.bits);
        answers = answers.max(shape.u.answers);
    }
    if seeds.len() != answers as usize {
        return Err(Error::ShapeMismatch);
    }
    for seed in seeds {
        if seed.z.len() != bits as usize || seed.beta == Fp3::ZERO {
            return Err(Error::ShapeMismatch);
        }
    }
    Ok(bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::request::binding::{BindingParameters, BindingStatement};
    use crate::pcs::request::code::build_code_plan;
    use crate::pcs::request::fold::{
        FoldShape, OracleMetadata, RuntimeOracles, VerifiedGroup, VerifiedGroups,
        VerifiedStatement,
    };
    use crate::pcs::request::foldeq::{
        reader_table_identity, FoldeqBinding, FoldeqEdgeBinding, FoldeqPlan, GammaPath,
    };
    use crate::pcs::request::gu::{GuNotice, GuOracleShape, GuShape};
    use crate::pcs::request::read::ReadLedger;
    use crate::pcs::request::runtime_tree::MaskTreePlan;
    use crate::pcs::request::table::test_commit_table;
    use std::sync::{Arc, Mutex};
    use crate::pcs::request::params::{RequestParams, LEAF_SALT_MIN_BYTES};
    use crate::pcs::request::runtime_tree::{RuntimeTreePart, RuntimeTreePlan};
    use crate::pcs::request::table::{index_name, IndexShape, TableDistances};
    use zkie_core::proof_header::PROOF_VERSION_WORD;
    use zkie_core::types::Digest;

    fn binding_parameters(shape: &IndexShape) -> BindingParameters {
        let request_parameters =
            RequestParams::default_with_code(shape.code.log_n, shape.code.ell, shape.code.r)
                .unwrap();
        BindingParameters::new(
            PROOF_VERSION_WORD,
            Digest([0x61; 32]),
            &request_parameters,
            Digest([0x62; 32]),
        )
        .unwrap()
    }

    fn binding_for(reader: &crate::pcs::request::table::IndexReader) -> FoldeqBinding {
        let parameters = binding_parameters(&reader.shape);
        let statement = BindingStatement::new(
            PROOF_VERSION_WORD,
            Digest([0x61; 32]),
            b"case".to_vec(),
            Digest([0x63; 32]),
            reader_table_identity(reader).unwrap(),
        )
        .unwrap();
        FoldeqBinding::new(&parameters, statement, reader).unwrap()
    }

    fn index_shape() -> IndexShape {
        IndexShape {
            code: build_code_plan(3, 4, 2).unwrap(),
            v: 2,
            answers: 1,
            salt_bytes: LEAF_SALT_MIN_BYTES,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        }
    }

    fn tree(main_rows: u64, mask_rows: u64) -> RuntimeTreePlan {
        RuntimeTreePlan {
            tree_number: 5,
            leaf_class: 2,
            log_n: 3,
            log_n_zk: if mask_rows > 0 { 2 } else { 0 },
            parts: vec![RuntimeTreePart {
                main_symbols: main_rows * 3,
                mask_symbols: mask_rows * 3,
            }],
            salt_bytes: LEAF_SALT_MIN_BYTES,
            run: Digest([9; 32]),
            configuration: Digest([7; 32]),
        }
    }

    fn shape() -> GuShape {
        GuShape {
            oracle: 9,
            group: 1,
            tree: tree(2, 0),
            part: 0,
            rows_high: false,
            u: GuOracleShape {
                code: build_code_plan(3, 6, 1).unwrap(),
                main_rows: 2,
                main_deg: 3,
                mask_rows: 0,
                mask_code: None,
                answers: 1,
            },
            index: index_shape(),
        }
    }

    fn registry(shape: &GuShape) -> GuRegistry {
        let mut registry = GuRegistry::new();
        registry
            .receive(
                shape,
                &GuNotice {
                    root: Digest([4; 32]),
                    oracle: shape.oracle,
                    tree_number: shape.tree.tree_number,
                    part: shape.part,
                },
            )
            .unwrap();
        registry
    }

    fn witness(shape: &GuShape) -> GuWitness {
        let main_len = (shape.u.main_rows * shape.u.code.ell) as usize;
        let mask_len = shape
            .u
            .mask_code
            .map(|code| (shape.u.mask_rows * code.ell) as usize)
            .unwrap_or(0);
        GuWitness {
            main: (0..main_len)
                .map(|index| Fp3::from_u64(100 + index as u64))
                .collect(),
            mask: (0..mask_len)
                .map(|index| Fp3::from_u64(200 + index as u64))
                .collect(),
        }
    }

    fn physical(shape: &GuShape, entry: u64) -> Option<usize> {
        let row_bits = ceil_log2(shape.u.main_rows);
        let place_bits = shape.u.code.log_ell();
        let v = if shape.rows_high {
            (entry >> place_bits) | ((entry & ((1u64 << place_bits) - 1)) << row_bits)
        } else {
            entry
        };
        let row = v & ((1u64 << row_bits) - 1);
        let place = v >> row_bits;
        if row >= shape.u.main_rows || place >= shape.u.code.ell {
            None
        } else {
            Some((row * shape.u.code.ell + place) as usize)
        }
    }

    fn eq_manual(index: u64, z: &[Fp3], bits: u32) -> Fp3 {
        let mut value = Fp3::ONE;
        for t in 0..bits as usize {
            let factor = if (index >> t) & 1 == 1 {
                z[t]
            } else {
                Fp3::ONE.sub(&z[t])
            };
            value = value.mul(&factor);
        }
        value
    }

    #[test]
    fn record_row_matches_an_independent_small_evaluation() {
        let shape = shape();
        let registry = registry(&shape);
        let name = index_name(&Digest([4; 32]), &shape.index, &Digest([7; 32]));
        let seeds = record_seeds(&shape.index, &name, 1).unwrap();
        let record = IndexRecord {
            y: vec![
                Fp3::from_u64(11),
                Fp3::from_u64(13),
                Fp3::from_u64(17),
                Fp3::from_u64(19),
            ],
        };
        let eq_gamma = vec![
            Fp3::from_u64(2),
            Fp3::from_u64(3),
            Fp3::from_u64(5),
            Fp3::from_u64(7),
        ];
        let row = record_row(&shape, &registry, &record, &name, &seeds, &eq_gamma, 0).unwrap();
        let witness = witness(&shape);
        let expected_target = eq_gamma[0]
            .mul(&record.y[0])
            .add(&eq_gamma[1].mul(&record.y[1]))
            .add(&eq_gamma[2].mul(&record.y[2]))
            .add(&eq_gamma[3].mul(&record.y[3]));
        assert_eq!(row.target, expected_target);
        // Independent coefficients: eq(j,z) on the folded message, all pad
        // weights removed, beta*theta on the target answer's pads.
        let bits = shape.index.cube_bits();
        let mut expected = vec![Fp3::ZERO; witness.main.len()];
        for entry in 0..shape.index.code.ell {
            expected[physical(&shape, entry).unwrap()] = eq_manual(entry, &seeds[0].z, bits);
        }
        for coefficient in 0..E_DEGREE {
            let place = shape.index.pad_place(0, coefficient);
            expected[physical(&shape, place).unwrap()] = seeds[0].beta.mul(&theta(coefficient));
        }
        assert_eq!(row.main, expected);
        let evaluated = row.evaluate(&witness).unwrap();
        let mut manual = Fp3::ZERO;
        for (coefficient, entry) in expected.iter().zip(&witness.main) {
            manual = manual.add(&coefficient.mul(entry));
        }
        assert_eq!(evaluated, manual);
        // A single pad entry change with the other inputs fixed no longer
        // satisfies the target.
        let mut changed_witness = witness.clone();
        changed_witness.main[1] = changed_witness.main[1].add(&Fp3::ONE);
        assert_ne!(row.evaluate(&changed_witness).unwrap(), manual);

        // A supplied seed list that differs from the recomputation is rejected.
        let mut wrong = seeds.clone();
        wrong[0].beta = wrong[0].beta.add(&Fp3::ONE);
        assert!(record_row(&shape, &registry, &record, &name, &wrong, &eq_gamma, 0).is_err());
        // A changed record changes the target, and the row built with the
        // trusted inputs no longer matches the changed target vocabulary.
        let mut changed = record.clone();
        changed.y[2] = changed.y[2].add(&Fp3::ONE);
        let changed_row =
            record_row(&shape, &registry, &changed, &name, &seeds, &eq_gamma, 0).unwrap();
        assert_ne!(changed_row.target, row.target);
        // A different group is not the registered complete shape.
        let mut other = shape.clone();
        other.group = 2;
        assert!(record_row(&other, &registry, &record, &name, &seeds, &eq_gamma, 0).is_err());
    }

    fn column_values(shape: &GuShape) -> Vec<Fp3> {
        (0..shape.index.rows())
            .map(|row| Fp3::from_u64(300 + row))
            .collect()
    }

    fn encode_symbol(
        code: &crate::pcs::request::code::CodeParams,
        message: &[Fp],
        tail: &[Fp],
        position: u64,
    ) -> Fp {
        crate::pcs::request::code::encode_row_fp(code, message, tail).unwrap()[position as usize]
    }

    #[test]
    fn column_row_matches_an_independent_generator_evaluation() {
        let shape = shape();
        let registry = registry(&shape);
        let position = 2u64;
        let column = column_values(&shape);
        let eq_gamma = vec![
            Fp3::from_u64(2),
            Fp3::from_u64(3),
            Fp3::from_u64(5),
            Fp3::from_u64(7),
        ];
        let row = column_row(&shape, &registry, position, &column, &eq_gamma).unwrap();
        let code = &shape.index.code;
        let mut expected = vec![Fp3::ZERO; shape.u.main_rows as usize * shape.u.code.ell as usize];
        for entry in 0..code.ell {
            let mut message = vec![Fp::ZERO; code.ell as usize];
            message[entry as usize] = Fp::ONE;
            let tail = vec![Fp::ZERO; code.r as usize];
            expected[physical(&shape, entry).unwrap()] =
                Fp3::from_base(encode_symbol(code, &message, &tail, position));
        }
        let first = tail_first(code).unwrap();
        for entry in 0..code.r {
            let message = vec![Fp::ZERO; code.ell as usize];
            let mut tail = vec![Fp::ZERO; code.r as usize];
            tail[entry as usize] = Fp::ONE;
            expected[physical(&shape, first + entry).unwrap()] =
                Fp3::from_base(encode_symbol(code, &message, &tail, position));
        }
        assert_eq!(row.main, expected);
        let mut target = Fp3::ZERO;
        for (weight, value) in eq_gamma.iter().zip(&column) {
            target = target.add(&weight.mul(value));
        }
        assert_eq!(row.target, target);
        let witness = witness(&shape);
        let mut manual = Fp3::ZERO;
        for (coefficient, entry) in expected.iter().zip(&witness.main) {
            manual = manual.add(&coefficient.mul(entry));
        }
        assert_eq!(row.evaluate(&witness).unwrap(), manual);
        // A single tail entry change with the other inputs fixed no longer
        // satisfies the target.
        let tail_offset = physical(&shape, first).unwrap();
        assert_ne!(row.main[tail_offset], Fp3::ZERO);
        let mut changed = witness.clone();
        changed.main[tail_offset] = changed.main[tail_offset].add(&Fp3::ONE);
        assert_ne!(row.evaluate(&changed).unwrap(), manual);
        let mut high = column.clone();
        high[0].c[1] = Fp::ONE;
        assert!(column_row(&shape, &registry, position, &high, &eq_gamma).is_err());
        let mut high = column.clone();
        high[0].c[2] = Fp::ONE;
        assert!(column_row(&shape, &registry, position, &high, &eq_gamma).is_err());
        assert!(column_row(&shape, &registry, code.n(), &column, &eq_gamma).is_err());
        assert!(column_row(&shape, &registry, position, &column[..3], &eq_gamma).is_err());
        assert!(column_row(&shape, &registry, position, &column, &eq_gamma[..3]).is_err());
        let mut other = shape.clone();
        other.group = 2;
        assert!(column_row(&other, &registry, position, &column, &eq_gamma).is_err());
    }

    #[test]
    fn column_draw_sequence_keeps_order_and_multiplicity() {
        let shape = shape();
        let registry = registry(&shape);
        let eq_gamma = vec![Fp3::ONE; 4];
        let column = column_values(&shape);
        let trusted = [1u64, 1, 3];
        let fixture_columns = vec![(1u64, column.clone()), (3u64, column.clone())];
        let rows = column_rows_from_draws(
            &shape,
            &registry,
            &trusted,
            &fixture_columns,
            &eq_gamma,
        )
        .unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], rows[1]);
        assert_eq!(unique_positions(&trusted), vec![1, 3]);
        assert!(validate_draw_sequence(&trusted, &[1, 1, 3]).is_ok());
        assert!(validate_draw_sequence(&trusted, &[1, 3, 3]).is_err());
        assert!(validate_draw_sequence(&trusted, &[1, 1]).is_err());
        assert!(validate_draw_sequence(&trusted, &[3, 1, 1]).is_err());
        // The same position cannot carry two different fixture columns,
        // even when a zero equation weight would make the folded targets equal.
        let mut other_column = column.clone();
        other_column[0] = other_column[0].add(&Fp3::ONE);
        let mut zero_eq = eq_gamma.clone();
        zero_eq[0] = Fp3::ZERO;
        let duplicated = vec![
            (1u64, column.clone()),
            (1u64, other_column),
            (3u64, column.clone()),
        ];
        assert!(column_rows_from_draws(
            &shape,
            &registry,
            &trusted,
            &duplicated,
            &zero_eq
        )
        .is_err());
        let missing = vec![(1u64, column.clone())];
        assert!(column_rows_from_draws(
            &shape,
            &registry,
            &trusted,
            &missing,
            &eq_gamma
        )
        .is_err());
        let extra = vec![
            (1u64, column.clone()),
            (2u64, column.clone()),
            (3u64, column.clone()),
        ];
        assert!(column_rows_from_draws(
            &shape,
            &registry,
            &trusted,
            &extra,
            &eq_gamma
        )
        .is_err());
        let unsorted = vec![(3u64, column.clone()), (1u64, column.clone())];
        assert!(column_rows_from_draws(
            &shape,
            &registry,
            &trusted,
            &unsorted,
            &eq_gamma
        )
        .is_err());
        let repeated_same = vec![
            (1u64, column.clone()),
            (1u64, column.clone()),
            (3u64, column.clone()),
        ];
        assert!(column_rows_from_draws(
            &shape,
            &registry,
            &trusted,
            &repeated_same,
            &eq_gamma
        )
        .is_err());
    }

    fn sample_shape(main_rows: u64, mask_rows: u64, answers: u32) -> GuShape {
        let main_code = build_code_plan(2, 4, 0).unwrap();
        let mask_code = if mask_rows > 0 {
            Some(build_code_plan(2, 4, 0).unwrap())
        } else {
            None
        };
        GuShape {
            oracle: 9,
            group: 1,
            tree: RuntimeTreePlan {
                tree_number: 5,
                leaf_class: 2,
                log_n: 2,
                log_n_zk: if mask_rows > 0 { 2 } else { 0 },
                parts: vec![RuntimeTreePart {
                    main_symbols: main_rows * 3,
                    mask_symbols: mask_rows * 3,
                }],
                salt_bytes: LEAF_SALT_MIN_BYTES,
                run: Digest([9; 32]),
                configuration: Digest([7; 32]),
            },
            part: 0,
            rows_high: false,
            u: GuOracleShape {
                code: main_code,
                main_rows,
                main_deg: 3,
                mask_rows,
                mask_code,
                answers,
            },
            index: index_shape(),
        }
    }

    fn sample_shape_wide() -> GuShape {
        GuShape {
            oracle: 9,
            group: 1,
            tree: RuntimeTreePlan {
                tree_number: 5,
                leaf_class: 2,
                log_n: 3,
                log_n_zk: 0,
                parts: vec![RuntimeTreePart {
                    main_symbols: 6,
                    mask_symbols: 0,
                }],
                salt_bytes: LEAF_SALT_MIN_BYTES,
                run: Digest([9; 32]),
                configuration: Digest([7; 32]),
            },
            part: 0,
            rows_high: false,
            u: GuOracleShape {
                code: build_code_plan(3, 8, 0).unwrap(),
                main_rows: 2,
                main_deg: 3,
                mask_rows: 0,
                mask_code: None,
                answers: 2,
            },
            index: index_shape(),
        }
    }

    fn sample_witness(shape: &GuShape) -> GuWitness {
        let main_len = (shape.u.main_rows * shape.u.code.ell) as usize;
        let mask_len = shape
            .u
            .mask_code
            .map(|code| (shape.u.mask_rows * code.ell) as usize)
            .unwrap_or(0);
        GuWitness {
            main: (0..main_len)
                .map(|index| Fp3::from_u64(400 + index as u64))
                .collect(),
            mask: (0..mask_len)
                .map(|index| Fp3::from_u64(500 + index as u64))
                .collect(),
        }
    }

    fn shared_seed(bits: u32, answer: u64) -> SampleSeed {
        SampleSeed {
            z: (0..bits)
                .map(|coordinate| Fp3::from_u64(10 + answer * 10 + coordinate as u64))
                .collect(),
            beta: Fp3::from_u64(90 + answer),
        }
    }

    #[test]
    fn sample_rows_share_one_seed_set_by_answer() {
        let first = sample_shape(2, 0, 1);
        let second = sample_shape_wide();
        let first_registry = registry(&first);
        let second_registry = registry(&second);
        let seeds = vec![shared_seed(4, 0), shared_seed(4, 1)];
        let bits =
            validate_shared_sample_seeds(&[&first, &second], &seeds).unwrap();
        assert_eq!(bits, 4);
        assert!(validate_shared_sample_seeds(&[&first, &second], &seeds[..1]).is_err());
        let mut short = seeds.clone();
        short[0].z.pop();
        assert!(validate_shared_sample_seeds(&[&first, &second], &short).is_err());
        let mut long = seeds.clone();
        long[0].z.push(Fp3::ONE);
        assert!(validate_shared_sample_seeds(&[&first, &second], &long).is_err());
        // One oracle may consume a prefix of the shared maximum.
        assert!(sample_cube(&first).unwrap().bits < seeds[0].z.len() as u32);

        let first_witness = sample_witness(&first);
        let mut expected = vec![Fp3::ZERO; first_witness.main.len()];
        let cube = sample_cube(&first).unwrap();
        for virtual_index in 0..(1u64 << cube.main_bits) {
            if let Some(offset) = physical_offset(
                first.u.main_rows,
                cube.main_row_bits,
                first.u.code.ell,
                virtual_index,
            )
            .unwrap()
            {
                expected[offset] =
                    eq_manual(virtual_index, &seeds[0].z, cube.main_bits);
            }
        }
        let rows = first.u.main_rows;
        for pad_answer in 0..first.u.answers {
            let place = first.u.code.ell - first.u.answers as u64 + pad_answer as u64;
            let virtual_index = (rows - 1) + (place << cube.main_row_bits);
            let offset = physical_offset(
                first.u.main_rows,
                cube.main_row_bits,
                first.u.code.ell,
                virtual_index,
            )
            .unwrap()
            .unwrap();
            expected[offset] = if pad_answer == 0 {
                seeds[0].beta.mul(&theta(0))
            } else {
                Fp3::ZERO
            };
        }
        let mut target = Fp3::ZERO;
        for (coefficient, entry) in expected.iter().zip(&first_witness.main) {
            target = target.add(&coefficient.mul(entry));
        }
        let first_row =
            sample_row(&first, &first_registry, &seeds, 0, target).unwrap();
        assert_eq!(first_row.main, expected);
        assert_eq!(first_row.evaluate(&first_witness).unwrap(), target);

        // The second oracle uses the same seed of answer 0 (prefix of its own
        // cube) and the second seed of answer 1.
        let second_witness = sample_witness(&second);
        for answer in 0..second.u.answers {
            let cube = sample_cube(&second).unwrap();
            let mut expected = vec![Fp3::ZERO; second_witness.main.len()];
            for virtual_index in 0..(1u64 << cube.main_bits) {
                if let Some(offset) = physical_offset(
                    second.u.main_rows,
                    cube.main_row_bits,
                    second.u.code.ell,
                    virtual_index,
                )
                .unwrap()
                {
                    expected[offset] = eq_manual(
                        virtual_index,
                        &seeds[answer as usize].z,
                        cube.main_bits,
                    );
                }
            }
            let rows = second.u.main_rows;
            for pad_answer in 0..second.u.answers {
                let place = second.u.code.ell - second.u.answers as u64 + pad_answer as u64;
                let virtual_index = (rows - 1) + (place << cube.main_row_bits);
                let offset = physical_offset(
                    second.u.main_rows,
                    cube.main_row_bits,
                    second.u.code.ell,
                    virtual_index,
                )
                .unwrap()
                .unwrap();
                expected[offset] = if pad_answer == answer {
                    seeds[answer as usize].beta.mul(&theta(0))
                } else {
                    Fp3::ZERO
                };
            }
            let mut target = Fp3::ZERO;
            for (coefficient, entry) in expected.iter().zip(&second_witness.main) {
                target = target.add(&coefficient.mul(entry));
            }
            let row =
                sample_row(&second, &second_registry, &seeds, answer, target).unwrap();
            assert_eq!(row.main, expected);
            assert_eq!(row.evaluate(&second_witness).unwrap(), target);
        }
        // A fixed witness does not satisfy a target changed by a perturbed seed.
        let changed = sample_row(
            &first,
            &first_registry,
            &[shared_seed(4, 7), seeds[1].clone()],
            0,
            target,
        )
        .unwrap();
        assert_ne!(changed.evaluate(&first_witness).unwrap(), target);
    }

    #[test]
    fn sample_row_uses_both_main_and_mask_fronts() {
        let shape = sample_shape(2, 1, 1);
        let registry = registry(&shape);
        let seeds = vec![shared_seed(4, 0)];
        let seed = &seeds[0];
        let cube = sample_cube(&shape).unwrap();
        assert!(cube.has_main && cube.has_mask);
        assert_eq!(cube.bits, 4);
        let witness = sample_witness(&shape);
        let top = seed.z[3];
        let main_front = Fp3::ONE.sub(&top);
        let mask_front = top.mul(&Fp3::ONE.sub(&seed.z[2]));
        let mut expected_main = vec![Fp3::ZERO; witness.main.len()];
        for virtual_index in 0..(1u64 << cube.main_bits) {
            if let Some(offset) = physical_offset(
                shape.u.main_rows,
                cube.main_row_bits,
                shape.u.code.ell,
                virtual_index,
            )
            .unwrap()
            {
                expected_main[offset] =
                    main_front.mul(&eq_manual(virtual_index, &seed.z, cube.main_bits));
            }
        }
        let mask_code = shape.u.mask_code.unwrap();
        let mut expected_mask = vec![Fp3::ZERO; witness.mask.len()];
        for virtual_index in 0..(1u64 << cube.mask_bits) {
            if let Some(offset) = physical_offset(
                shape.u.mask_rows,
                cube.mask_row_bits,
                mask_code.ell,
                virtual_index,
            )
            .unwrap()
            {
                expected_mask[offset] =
                    mask_front.mul(&eq_manual(virtual_index, &seed.z, cube.mask_bits));
            }
        }
        let place = mask_code.ell - shape.u.answers as u64;
        let pad = (shape.u.mask_rows - 1) + (place << cube.mask_row_bits);
        let pad_offset = physical_offset(
            shape.u.mask_rows,
            cube.mask_row_bits,
            mask_code.ell,
            pad,
        )
        .unwrap()
        .unwrap();
        expected_mask[pad_offset] = seed.beta.mul(&theta(0));
        let mut target = Fp3::ZERO;
        for (coefficient, entry) in expected_main.iter().zip(&witness.main) {
            target = target.add(&coefficient.mul(entry));
        }
        for (coefficient, entry) in expected_mask.iter().zip(&witness.mask) {
            target = target.add(&coefficient.mul(entry));
        }
        let row = sample_row(&shape, &registry, &seeds, 0, target).unwrap();
        assert_eq!(row.main, expected_main);
        assert_eq!(row.mask, expected_mask);
        assert_eq!(row.evaluate(&witness).unwrap(), target);
        let mut changed = witness.clone();
        changed.mask[pad_offset] = changed.mask[pad_offset].add(&Fp3::ONE);
        assert_ne!(row.evaluate(&changed).unwrap(), target);
    }

    #[test]
    fn record_row_uses_the_transposed_layout_when_rows_are_high() {
        let mut shape = shape();
        shape.rows_high = true;
        let registry = registry(&shape);
        let name = index_name(&Digest([4; 32]), &shape.index, &Digest([7; 32]));
        let seeds = record_seeds(&shape.index, &name, 1).unwrap();
        let record = IndexRecord {
            y: vec![
                Fp3::from_u64(11),
                Fp3::from_u64(13),
                Fp3::from_u64(17),
                Fp3::from_u64(19),
            ],
        };
        let eq_gamma = vec![
            Fp3::from_u64(2),
            Fp3::from_u64(3),
            Fp3::from_u64(5),
            Fp3::from_u64(7),
        ];
        let row = record_row(&shape, &registry, &record, &name, &seeds, &eq_gamma, 0).unwrap();
        let mut expected = vec![Fp3::ZERO; (shape.u.main_rows * shape.u.code.ell) as usize];
        let bits = shape.index.cube_bits();
        for entry in 0..shape.index.code.ell {
            expected[physical(&shape, entry).unwrap()] = eq_manual(entry, &seeds[0].z, bits);
        }
        for coefficient in 0..E_DEGREE {
            let place = shape.index.pad_place(0, coefficient);
            expected[physical(&shape, place).unwrap()] =
                seeds[0].beta.mul(&theta(coefficient));
        }
        assert_eq!(row.main, expected);
        let witness = witness(&shape);
        let mut manual = Fp3::ZERO;
        for (coefficient, entry) in expected.iter().zip(&witness.main) {
            manual = manual.add(&coefficient.mul(entry));
        }
        assert_eq!(row.evaluate(&witness).unwrap(), manual);
    }

    #[test]
    fn record_and_column_rows_handle_three_rows_with_rows_high() {
        let mut shape = shape();
        shape.rows_high = true;
        shape.u.code = build_code_plan(4, 10, 1).unwrap();
        shape.u.main_rows = 3;
        shape.tree = RuntimeTreePlan {
            tree_number: 5,
            leaf_class: 2,
            log_n: 4,
            log_n_zk: 0,
            parts: vec![RuntimeTreePart {
                main_symbols: 9,
                mask_symbols: 0,
            }],
            salt_bytes: LEAF_SALT_MIN_BYTES,
            run: Digest([9; 32]),
            configuration: Digest([7; 32]),
        };
        let registry = registry(&shape);
        let name = index_name(&Digest([4; 32]), &shape.index, &Digest([7; 32]));
        let seeds = record_seeds(&shape.index, &name, 1).unwrap();
        let record = IndexRecord {
            y: vec![
                Fp3::from_u64(11),
                Fp3::from_u64(13),
                Fp3::from_u64(17),
                Fp3::from_u64(19),
            ],
        };
        let eq_gamma = vec![
            Fp3::from_u64(2),
            Fp3::from_u64(3),
            Fp3::from_u64(5),
            Fp3::from_u64(7),
        ];
        let row = record_row(&shape, &registry, &record, &name, &seeds, &eq_gamma, 0).unwrap();
        let mut expected = vec![Fp3::ZERO; (shape.u.main_rows * shape.u.code.ell) as usize];
        expected[physical(&shape, 0).unwrap()] =
            eq_manual(0, &seeds[0].z, shape.index.cube_bits());
        for coefficient in 0..E_DEGREE {
            let place = shape.index.pad_place(0, coefficient);
            expected[physical(&shape, place).unwrap()] =
                seeds[0].beta.mul(&theta(coefficient));
        }
        assert_eq!(row.main, expected);
        let witness = witness(&shape);
        let mut manual = Fp3::ZERO;
        for (coefficient, entry) in expected.iter().zip(&witness.main) {
            manual = manual.add(&coefficient.mul(entry));
        }
        assert_eq!(row.evaluate(&witness).unwrap(), manual);
        assert_eq!(
            physical_index(&shape, vector_index(&shape, 8).unwrap()).unwrap(),
            Some(8)
        );
        assert_eq!(
            physical_index(&shape, vector_index(&shape, 15).unwrap()).unwrap(),
            None
        );

        // A column row under the same transposed layout.
        let column = column_values(&shape);
        let column_row_value =
            column_row(&shape, &registry, 2, &column, &eq_gamma).unwrap();
        let code = &shape.index.code;
        let mut expected = vec![Fp3::ZERO; witness.main.len()];
        for entry in 0..code.ell {
            let mut message = vec![Fp::ZERO; code.ell as usize];
            message[entry as usize] = Fp::ONE;
            let tail = vec![Fp::ZERO; code.r as usize];
            expected[physical(&shape, entry).unwrap()] =
                Fp3::from_base(encode_symbol(code, &message, &tail, 2));
        }
        let first = tail_first(code).unwrap();
        for entry in 0..code.r {
            let message = vec![Fp::ZERO; code.ell as usize];
            let mut tail = vec![Fp::ZERO; code.r as usize];
            tail[entry as usize] = Fp::ONE;
            expected[physical(&shape, first + entry).unwrap()] =
                Fp3::from_base(encode_symbol(code, &message, &tail, 2));
        }
        assert_eq!(column_row_value.main, expected);
        let mut column_target = Fp3::ZERO;
        for (weight, value) in eq_gamma.iter().zip(&column) {
            column_target = column_target.add(&weight.mul(value));
        }
        assert_eq!(column_row_value.target, column_target);
        let mut manual = Fp3::ZERO;
        for (coefficient, entry) in expected.iter().zip(&witness.main) {
            manual = manual.add(&coefficient.mul(entry));
        }
        assert_eq!(column_row_value.evaluate(&witness).unwrap(), manual);
    }

    #[test]
    fn record_row_handles_two_answers_and_target_answer_one() {
        let mut shape = shape();
        shape.index.code = build_code_plan(4, 8, 2).unwrap();
        shape.index.answers = 2;
        let registry = registry(&shape);
        let name = index_name(&Digest([4; 32]), &shape.index, &Digest([7; 32]));
        let seeds = record_seeds(&shape.index, &name, 2).unwrap();
        let record = IndexRecord {
            y: (0..8).map(|index| Fp3::from_u64(20 + index)).collect(),
        };
        let eq_gamma = vec![
            Fp3::from_u64(2),
            Fp3::from_u64(3),
            Fp3::from_u64(5),
            Fp3::from_u64(7),
        ];
        let row = record_row(&shape, &registry, &record, &name, &seeds, &eq_gamma, 1).unwrap();
        let bits = shape.index.cube_bits();
        let mut expected = vec![Fp3::ZERO; (shape.u.main_rows * shape.u.code.ell) as usize];
        for entry in 0..shape.index.code.ell {
            expected[physical(&shape, entry).unwrap()] =
                eq_manual(entry, &seeds[1].z, bits);
        }
        for coefficient in 0..E_DEGREE {
            let answer_zero = shape.index.pad_place(0, coefficient);
            let answer_one = shape.index.pad_place(1, coefficient);
            expected[physical(&shape, answer_zero).unwrap()] = Fp3::ZERO;
            expected[physical(&shape, answer_one).unwrap()] =
                seeds[1].beta.mul(&theta(coefficient));
        }
        assert_eq!(row.main, expected);
        let witness = witness(&shape);
        let mut manual = Fp3::ZERO;
        for (coefficient, entry) in expected.iter().zip(&witness.main) {
            manual = manual.add(&coefficient.mul(entry));
        }
        assert_eq!(row.evaluate(&witness).unwrap(), manual);
        let mut target = Fp3::ZERO;
        for (row_index, weight) in eq_gamma.iter().enumerate() {
            target = target.add(&weight.mul(&record.y[row_index * 2 + 1]));
        }
        assert_eq!(row.target, target);
        assert!(record_row(&shape, &registry, &record, &name, &seeds, &eq_gamma, 2).is_err());
    }

    fn sample_shape_three(mask_rows: u64) -> GuShape {
        let main_code = build_code_plan(4, 10, 0).unwrap();
        let mask_code = if mask_rows > 0 {
            Some(build_code_plan(4, 10, 0).unwrap())
        } else {
            None
        };
        GuShape {
            oracle: 9,
            group: 1,
            tree: RuntimeTreePlan {
                tree_number: 5,
                leaf_class: 2,
                log_n: 4,
                log_n_zk: if mask_rows > 0 { 4 } else { 0 },
                parts: vec![RuntimeTreePart {
                    main_symbols: 9,
                    mask_symbols: mask_rows * 3,
                }],
                salt_bytes: LEAF_SALT_MIN_BYTES,
                run: Digest([9; 32]),
                configuration: Digest([7; 32]),
            },
            part: 0,
            rows_high: true,
            u: GuOracleShape {
                code: main_code,
                main_rows: 3,
                main_deg: 3,
                mask_rows,
                mask_code,
                answers: 2,
            },
            index: index_shape(),
        }
    }

    #[test]
    fn sample_rows_handle_three_rows_and_mask_rows() {
        for mask_rows in [0u64, 3] {
            let shape = sample_shape_three(mask_rows);
            let registry = registry(&shape);
            let cube = sample_cube(&shape).unwrap();
            let bits = cube.bits;
            let manual_offset = |rows: u64, row_bits: u32, len: u64, value: u64| {
                let row = value & ((1u64 << row_bits) - 1);
                let place = value >> row_bits;
                if row >= rows || place >= len {
                    None
                } else {
                    Some((row * len + place) as usize)
                }
            };
            let seeds: Vec<SampleSeed> = (0..2)
                .map(|answer| SampleSeed {
                    z: (0..bits)
                        .map(|coordinate| {
                            Fp3::from_u64(600 + answer * 10 + coordinate as u64)
                        })
                        .collect(),
                    beta: Fp3::from_u64(700 + answer),
                })
                .collect();
            assert_eq!(
                validate_shared_sample_seeds(&[&shape], &seeds).unwrap(),
                bits
            );
            let answer = 1u32;
            let witness = sample_witness(&shape);
            let (main_front, mask_front) = if mask_rows > 0 {
                let top = seeds[answer as usize].z[bits as usize - 1];
                (Fp3::ONE.sub(&top), top)
            } else {
                (Fp3::ONE, Fp3::ONE)
            };
            let mut expected_main = vec![Fp3::ZERO; witness.main.len()];
            for virtual_index in 0..(1u64 << cube.main_bits) {
                if let Some(offset) = manual_offset(
                    shape.u.main_rows,
                    cube.main_row_bits,
                    shape.u.code.ell,
                    virtual_index,
                ) {
                    expected_main[offset] = main_front.mul(&eq_manual(
                        virtual_index,
                        &seeds[answer as usize].z,
                        cube.main_bits,
                    ));
                }
            }
            let mut expected_mask = vec![Fp3::ZERO; witness.mask.len()];
            if mask_rows > 0 {
                let code = shape.u.mask_code.unwrap();
                for virtual_index in 0..(1u64 << cube.mask_bits) {
                    if let Some(offset) = manual_offset(
                        shape.u.mask_rows,
                        cube.mask_row_bits,
                        code.ell,
                        virtual_index,
                    ) {
                        expected_mask[offset] = mask_front.mul(&eq_manual(
                            virtual_index,
                            &seeds[answer as usize].z,
                            cube.mask_bits,
                        ));
                    }
                }
            }
            let len = shape.u.code.ell;
            for pad_answer in 0..shape.u.answers {
                let place = len - shape.u.answers as u64 + pad_answer as u64;
                let main_virtual = (shape.u.main_rows - 1) + (place << cube.main_row_bits);
                if mask_rows == 0 {
                    let offset = manual_offset(
                        shape.u.main_rows,
                        cube.main_row_bits,
                        len,
                        main_virtual,
                    )
                    .unwrap();
                    expected_main[offset] = if pad_answer == answer {
                        seeds[answer as usize].beta.mul(&theta(0))
                    } else {
                        Fp3::ZERO
                    };
                } else {
                    let code = shape.u.mask_code.unwrap();
                    let mask_virtual =
                        (shape.u.mask_rows - 1) + (place << cube.mask_row_bits);
                    let offset = manual_offset(
                        shape.u.mask_rows,
                        cube.mask_row_bits,
                        code.ell,
                        mask_virtual,
                    )
                    .unwrap();
                    expected_mask[offset] = if pad_answer == answer {
                        seeds[answer as usize].beta.mul(&theta(0))
                    } else {
                        Fp3::ZERO
                    };
                }
            }
            let mut target = Fp3::ZERO;
            for (coefficient, entry) in expected_main.iter().zip(&witness.main) {
                target = target.add(&coefficient.mul(entry));
            }
            for (coefficient, entry) in expected_mask.iter().zip(&witness.mask) {
                target = target.add(&coefficient.mul(entry));
            }
            let row = sample_row(&shape, &registry, &seeds, answer, target).unwrap();
            assert_eq!(row.main, expected_main);
            assert_eq!(row.mask, expected_mask);
            assert_eq!(row.evaluate(&witness).unwrap(), target);
            let non_pad = (0..expected_main.len())
                .find(|index| expected_main[*index] != Fp3::ZERO)
                .unwrap();
            let mut changed = witness.clone();
            changed.main[non_pad] = changed.main[non_pad].add(&Fp3::ONE);
            assert_ne!(row.evaluate(&changed).unwrap(), target);
            let place = len - shape.u.answers as u64 + answer as u64;
            if mask_rows == 0 {
                let pad_virtual = (shape.u.main_rows - 1) + (place << cube.main_row_bits);
                let pad_offset = manual_offset(
                    shape.u.main_rows,
                    cube.main_row_bits,
                    len,
                    pad_virtual,
                )
                .unwrap();
                let mut changed = witness.clone();
                changed.main[pad_offset] = changed.main[pad_offset].add(&Fp3::ONE);
                assert_ne!(row.evaluate(&changed).unwrap(), target);
            } else {
                let code = shape.u.mask_code.unwrap();
                let pad_virtual = (shape.u.mask_rows - 1) + (place << cube.mask_row_bits);
                let pad_offset = manual_offset(
                    shape.u.mask_rows,
                    cube.mask_row_bits,
                    code.ell,
                    pad_virtual,
                )
                .unwrap();
                let mut changed = witness.clone();
                changed.mask[pad_offset] = changed.mask[pad_offset].add(&Fp3::ONE);
                assert_ne!(row.evaluate(&changed).unwrap(), target);
            }
        }
    }

    fn tail_layout_shape(index_r: u64, mask_rows: u64) -> GuShape {
        let index = IndexShape {
            code: build_code_plan(4, 8, index_r).unwrap(),
            v: 2,
            answers: 1,
            salt_bytes: LEAF_SALT_MIN_BYTES,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        };
        let mask_code = if mask_rows > 0 {
            Some(build_code_plan(4, 10, 1).unwrap())
        } else {
            None
        };
        GuShape {
            oracle: 9,
            group: 1,
            tree: RuntimeTreePlan {
                tree_number: 5,
                leaf_class: 2,
                log_n: 4,
                log_n_zk: if mask_rows > 0 { 4 } else { 0 },
                parts: vec![RuntimeTreePart {
                    main_symbols: 9,
                    mask_symbols: mask_rows * 3,
                }],
                salt_bytes: LEAF_SALT_MIN_BYTES,
                run: Digest([9; 32]),
                configuration: Digest([7; 32]),
            },
            part: 0,
            rows_high: true,
            u: GuOracleShape {
                code: build_code_plan(4, 10, 1).unwrap(),
                main_rows: 3,
                main_deg: 3,
                mask_rows,
                mask_code,
                answers: 1,
            },
            index,
        }
    }

    #[test]
    fn column_row_rejects_a_real_tail_without_a_physical_slot() {
        let shape = tail_layout_shape(4, 1);
        assert!(shape.fault().is_none());
        let registry = registry(&shape);
        let column = vec![
            Fp3::from_u64(1),
            Fp3::from_u64(2),
            Fp3::from_u64(3),
            Fp3::from_u64(4),
        ];
        let eq_gamma = vec![Fp3::ONE; 4];
        // index ell 8, r 4: the tail entries 8..12 are real; entries 10 and 11
        // map to place 10/11, beyond the G_u row length 10, and must fail
        // closed rather than be silently dropped.
        assert!(column_row(&shape, &registry, 0, &column, &eq_gamma).is_err());
    }

    #[test]
    fn column_row_places_a_non_power_width_tail_that_fits() {
        let shape = tail_layout_shape(2, 0);
        assert!(shape.fault().is_none());
        let registry = registry(&shape);
        let column = vec![
            Fp3::from_u64(1),
            Fp3::from_u64(2),
            Fp3::from_u64(3),
            Fp3::from_u64(4),
        ];
        let eq_gamma = vec![Fp3::from_u64(2), Fp3::from_u64(3), Fp3::from_u64(5), Fp3::from_u64(7)];
        let row = column_row(&shape, &registry, 0, &column, &eq_gamma).unwrap();
        let manual_offset = |rows: u64, row_bits: u32, len: u64, value: u64| {
            let row = value & ((1u64 << row_bits) - 1);
            let place = value >> row_bits;
            if row >= rows || place >= len {
                None
            } else {
                Some((row * len + place) as usize)
            }
        };
        let code = &shape.index.code;
        let mut expected = vec![Fp3::ZERO; (shape.u.main_rows * shape.u.code.ell) as usize];
        for entry in 0..code.ell {
            let mut message = vec![Fp::ZERO; code.ell as usize];
            message[entry as usize] = Fp::ONE;
            let tail = vec![Fp::ZERO; code.r as usize];
            let v = (entry >> shape.u.code.log_ell())
                | ((entry & ((1u64 << shape.u.code.log_ell()) - 1)) << 2);
            expected[manual_offset(
                shape.u.main_rows,
                2,
                shape.u.code.ell,
                v,
            )
            .unwrap()] = Fp3::from_base(encode_symbol(code, &message, &tail, 0));
        }
        let first = tail_first(code).unwrap();
        for entry in 0..code.r {
            let message = vec![Fp::ZERO; code.ell as usize];
            let mut tail = vec![Fp::ZERO; code.r as usize];
            tail[entry as usize] = Fp::ONE;
            let v = ((first + entry) >> shape.u.code.log_ell())
                | (((first + entry) & ((1u64 << shape.u.code.log_ell()) - 1)) << 2);
            expected[manual_offset(
                shape.u.main_rows,
                2,
                shape.u.code.ell,
                v,
            )
            .unwrap()] = Fp3::from_base(encode_symbol(code, &message, &tail, 0));
        }
        assert_eq!(row.main, expected);
        let mut target = Fp3::ZERO;
        for (weight, value) in eq_gamma.iter().zip(&column) {
            target = target.add(&weight.mul(value));
        }
        assert_eq!(row.target, target);
    }

    fn manual_slot(rows: u64, row_bits: u32, len: u64, value: u64) -> Option<usize> {
        let row = value & ((1u64 << row_bits) - 1);
        let place = value >> row_bits;
        if row >= rows || place >= len {
            None
        } else {
            Some((row * len + place) as usize)
        }
    }

    #[test]
    fn gu_table_keeps_index_pads_and_reserves_send_pads() {
        // rows_high, main_rows 3, non-power ell 10, no mask.
        let wide = tail_layout_shape(2, 0);
        assert!(wide.fault().is_none());
        assert!(gu_layout_fault(&wide).is_none());
        let message: Vec<Fp3> = (0..8).map(|index| Fp3::from_u64(100 + index)).collect();
        let tail = vec![Fp3::from_u64(200), Fp3::from_u64(201)];
        let table = gu_table(&wide, &message, &tail).unwrap();
        let mut expected = vec![Fp3::ZERO; (wide.u.main_rows * wide.u.code.ell) as usize];
        let row_bits = 2u32;
        let place_bits = wide.u.code.log_ell();
        for (entry, value) in message.iter().enumerate() {
            let value = *value;
            let v = ((entry as u64) >> place_bits)
                | (((entry as u64) & ((1u64 << place_bits) - 1)) << row_bits);
            expected[manual_slot(wide.u.main_rows, row_bits, wide.u.code.ell, v).unwrap()] =
                value;
        }
        let first = tail_first(&wide.index.code).unwrap();
        for (entry, value) in tail.iter().enumerate() {
            let v = ((first + entry as u64) >> place_bits)
                | (((first + entry as u64) & ((1u64 << place_bits) - 1)) << row_bits);
            expected[manual_slot(wide.u.main_rows, row_bits, wide.u.code.ell, v).unwrap()] =
                *value;
        }
        // The index pads inside the message are retained.
        for pad in 5..8 {
            let v = ((pad as u64) >> place_bits)
                | (((pad as u64) & ((1u64 << place_bits) - 1)) << row_bits);
            expected[manual_slot(wide.u.main_rows, row_bits, wide.u.code.ell, v).unwrap()] =
                message[pad];
        }
        let send_pad = (wide.u.main_rows - 1)
            + ((wide.u.code.ell - wide.u.answers as u64) << row_bits);
        let send_pad_offset =
            manual_slot(wide.u.main_rows, row_bits, wide.u.code.ell, send_pad).unwrap();
        expected[send_pad_offset] = Fp3::ZERO;
        assert_eq!(table.main, expected);
        assert!(table.mask.is_empty());

        // With a mask, the G_u sample pad is in the mask table and the main
        // folded data (including the index pads) stays.
        let masked = tail_layout_shape(2, 1);
        assert!(masked.fault().is_none());
        let masked_table = gu_table(&masked, &message, &tail).unwrap();
        let mut masked_expected = expected.clone();
        masked_expected[send_pad_offset] = Fp3::ZERO;
        assert_eq!(masked_table.main, masked_expected);
        let mask_pad = (masked.u.mask_rows - 1)
            + ((masked.u.mask_code.unwrap().ell - masked.u.answers as u64)
                << ceil_log2(masked.u.mask_rows));
        let mask_pad_offset = manual_slot(
            masked.u.mask_rows,
            ceil_log2(masked.u.mask_rows),
            masked.u.mask_code.unwrap().ell,
            mask_pad,
        )
        .unwrap();
        assert_eq!(masked_table.mask[mask_pad_offset], Fp3::ZERO);
        assert!(masked_table.mask.iter().all(|value| *value == Fp3::ZERO));

        // The rows_low layout: main_rows 2, ell 6, index ell 4, tail 2.
        let low = shape();
        let message = vec![
            Fp3::from_u64(10),
            Fp3::from_u64(11),
            Fp3::from_u64(12),
            Fp3::from_u64(13),
        ];
        let tail = vec![Fp3::from_u64(20), Fp3::from_u64(21)];
        let table = gu_table(&low, &message, &tail).unwrap();
        let mut expected = vec![Fp3::ZERO; 12];
        for entry in 0..4u64 {
            let row = entry & 1;
            let place = entry >> 1;
            expected[(row * low.u.code.ell + place) as usize] = message[entry as usize];
        }
        let first = tail_first(&low.index.code).unwrap();
        for entry in 0..2u64 {
            let v = first + entry;
            let row = v & 1;
            let place = v >> 1;
            expected[(row * low.u.code.ell + place) as usize] = tail[entry as usize];
        }
        assert_eq!(table.main, expected);

        // A shape that passes fault() but has a real tail without a slot is
        // rejected by the same layout check and by the table builder.
        let bad = tail_layout_shape(4, 1);
        assert!(bad.fault().is_none());
        assert!(gu_layout_fault(&bad).is_some());
        assert!(gu_table(&bad, &[Fp3::ONE; 8], &[Fp3::ONE; 4]).is_err());
    }

    #[test]
    fn column_plan_rows_checks_the_plan_and_fixture_list() {
        let shape = shape();
        let registry = registry(&shape);
        let column = column_values(&shape);
        let eq_gamma = vec![Fp3::ONE; 4];
        let fixture = vec![(1u64, column.clone()), (3u64, column.clone())];
        let draws = [1u64, 1, 3];
        let log_n = shape.index.code.log_n;
        let rows = column_plan_rows(
            &shape, &registry, 3, log_n, &draws, &fixture, &eq_gamma,
        )
        .unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], rows[1]);
        assert_eq!(unique_positions(&draws), vec![1, 3]);
        assert!(column_plan_rows(
            &shape, &registry, 2, log_n, &draws, &fixture, &eq_gamma
        )
        .is_err());
        assert!(column_plan_rows(
            &shape, &registry, 4, log_n, &draws, &fixture, &eq_gamma
        )
        .is_err());
        assert!(column_plan_rows(
            &shape, &registry, 3, log_n + 1, &draws, &fixture, &eq_gamma
        )
        .is_err());
        assert!(column_plan_rows(
            &shape,
            &registry,
            3,
            log_n,
            &[1, 1, (1u64 << log_n)],
            &fixture,
            &eq_gamma
        )
        .is_err());
        let extra = vec![
            (1u64, column.clone()),
            (2u64, column.clone()),
            (3u64, column.clone()),
        ];
        assert!(column_plan_rows(
            &shape, &registry, 3, log_n, &draws, &extra, &eq_gamma
        )
        .is_err());
    }

    #[test]
    fn fixture_source_replacements_fail_at_the_matching_boundary() {
        let shape = shape();
        let registry = registry(&shape);
        let name = index_name(&Digest([4; 32]), &shape.index, &Digest([7; 32]));
        let seeds = record_seeds(&shape.index, &name, 1).unwrap();
        let record = IndexRecord {
            y: vec![
                Fp3::from_u64(11),
                Fp3::from_u64(13),
                Fp3::from_u64(17),
                Fp3::from_u64(19),
            ],
        };
        let eq_gamma = vec![
            Fp3::from_u64(2),
            Fp3::from_u64(3),
            Fp3::from_u64(5),
            Fp3::from_u64(7),
        ];
        let row = record_row(&shape, &registry, &record, &name, &seeds, &eq_gamma, 0).unwrap();
        // Value/relationship change: a different table name changes the
        // recomputed seeds, so the supplied fixture seeds no longer match.
        // This is a structural recomputation, not an authenticated-source
        // rejection.
        let other_name = index_name(&Digest([6; 32]), &shape.index, &Digest([7; 32]));
        assert!(record_row(
            &shape, &registry, &record, &other_name, &seeds, &eq_gamma, 0
        )
        .is_err());
        // Value/relationship change: a different row-point weight changes the
        // target. With a fixed witness the old row no longer matches it.
        let mut other_gamma = eq_gamma.clone();
        other_gamma[0] = other_gamma[0].add(&Fp3::ONE);
        let other_row =
            record_row(&shape, &registry, &record, &name, &seeds, &other_gamma, 0).unwrap();
        assert_ne!(other_row.target, row.target);
        // Value/relationship change: a different provided column changes the
        // column target; it does not prove that a real opening would accept or
        // reject either column.
        let column = column_values(&shape);
        let column_row_value = column_row(&shape, &registry, 1, &column, &eq_gamma).unwrap();
        let mut other_column = column.clone();
        other_column[0] = other_column[0].add(&Fp3::ONE);
        let other_column_row =
            column_row(&shape, &registry, 1, &other_column, &eq_gamma).unwrap();
        assert_ne!(other_column_row.target, column_row_value.target);
        // Structure change: a changed draw sequence does not match the
        // provided unique columns. The real chain/opening rejection is not
        // connected and is an integration gate.
        let fixture = vec![(1u64, column.clone()), (3u64, column.clone())];
        assert!(column_plan_rows(
            &shape,
            &registry,
            2,
            shape.index.code.log_n,
            &[1, 2],
            &fixture,
            &eq_gamma
        )
        .is_err());
    }

    #[test]
    fn one_actual_gu_table_satisfies_record_column_and_sample() {
        let shape = shape();
        let registry = registry(&shape);
        let rows: Vec<Vec<Fp>> = vec![
            vec![Fp::from_u64(3), Fp::from_u64(5), Fp::from_u64(7), Fp::from_u64(11)],
            vec![Fp::from_u64(13), Fp::from_u64(17), Fp::from_u64(19), Fp::from_u64(23)],
            vec![Fp::from_u64(29), Fp::from_u64(31), Fp::from_u64(37), Fp::from_u64(41)],
            vec![Fp::from_u64(43), Fp::from_u64(47), Fp::from_u64(53), Fp::from_u64(59)],
        ];
        let tails: Vec<Vec<Fp>> = vec![
            vec![Fp::from_u64(2), Fp::from_u64(3)],
            vec![Fp::from_u64(5), Fp::from_u64(7)],
            vec![Fp::from_u64(11), Fp::from_u64(13)],
            vec![Fp::from_u64(17), Fp::from_u64(19)],
        ];
        let eq_gamma = vec![
            Fp3::from_u64(2),
            Fp3::from_u64(3),
            Fp3::from_u64(5),
            Fp3::from_u64(7),
        ];
        let mut folded_message = vec![Fp3::ZERO; shape.index.code.ell as usize];
        for (row, weight) in eq_gamma.iter().enumerate() {
            for (entry, value) in rows[row].iter().enumerate() {
                folded_message[entry] = folded_message[entry]
                    .add(&weight.mul(&Fp3::from_base(*value)));
            }
        }
        let mut folded_tail = vec![Fp3::ZERO; shape.index.code.r as usize];
        for (row, weight) in eq_gamma.iter().enumerate() {
            for (entry, value) in tails[row].iter().enumerate() {
                folded_tail[entry] =
                    folded_tail[entry].add(&weight.mul(&Fp3::from_base(*value)));
            }
        }
        let mut witness = gu_table(&shape, &folded_message, &folded_tail).unwrap();
        // The G_u sample pad is a separate slot; fill it with a fixture value.
        let send_pad_virtual = (shape.u.main_rows - 1)
            + ((shape.u.code.ell - shape.u.answers as u64) << ceil_log2(shape.u.main_rows));
        let row_bits = ceil_log2(shape.u.main_rows);
        let send_pad = ((send_pad_virtual & ((1u64 << row_bits) - 1)) * shape.u.code.ell
            + (send_pad_virtual >> row_bits)) as usize;
        assert_eq!(witness.main[send_pad], Fp3::ZERO);
        let send_value = Fp3::from_u64(999);
        witness.main[send_pad] = send_value;

        let name = index_name(&Digest([4; 32]), &shape.index, &Digest([7; 32]));
        let record_seed = record_seeds(&shape.index, &name, 1).unwrap();
        let bits = shape.index.cube_bits();
        let plain = shape.index.code.ell - shape.index.pads() as u64;
        let mut record = IndexRecord { y: Vec::new() };
        for row in &rows {
            let mut value = Fp3::ZERO;
            for entry in 0..plain {
                value = value.add(
                    &eq_manual(entry, &record_seed[0].z, bits)
                        .mul(&Fp3::from_base(row[entry as usize])),
                );
            }
            for coefficient in 0..E_DEGREE {
                let place = shape.index.pad_place(0, coefficient) as usize;
                value = value.add(
                    &record_seed[0]
                        .beta
                        .mul(&theta(coefficient))
                        .mul(&Fp3::from_base(row[place])),
                );
            }
            record.y.push(value);
        }
        let record_row_value =
            record_row(&shape, &registry, &record, &name, &record_seed, &eq_gamma, 0).unwrap();
        assert_eq!(record_row_value.evaluate(&witness).unwrap(), record_row_value.target);

        let x = 2u64;
        let column: Vec<Fp3> = rows
            .iter()
            .zip(&tails)
            .map(|(row, tail)| {
                Fp3::from_base(
                    crate::pcs::request::code::encode_row_fp(&shape.index.code, row, tail)
                        .unwrap()[x as usize],
                )
            })
            .collect();
        let column_row_value = column_row(&shape, &registry, x, &column, &eq_gamma).unwrap();
        let mut column_target = Fp3::ZERO;
        for (weight, value) in eq_gamma.iter().zip(&column) {
            column_target = column_target.add(&weight.mul(value));
        }
        assert_eq!(column_row_value.target, column_target);
        assert_eq!(
            column_row_value.evaluate(&witness).unwrap(),
            column_target
        );

        let sample_seed = SampleSeed {
            z: (0..4).map(|index| Fp3::from_u64(70 + index)).collect(),
            beta: Fp3::from_u64(80),
        };
        let sample_seeds = vec![sample_seed];
        let cube = sample_cube(&shape).unwrap();
        let mut expected_sample = vec![Fp3::ZERO; witness.main.len()];
        for virtual_index in 0..(1u64 << cube.main_bits) {
            let row = virtual_index & ((1u64 << cube.main_row_bits) - 1);
            let place = virtual_index >> cube.main_row_bits;
            if row < shape.u.main_rows && place < shape.u.code.ell {
                expected_sample[(row * shape.u.code.ell + place) as usize] =
                    eq_manual(virtual_index, &sample_seeds[0].z, cube.main_bits);
            }
        }
        let pad_virtual = send_pad_virtual;
        expected_sample[send_pad] =
            expected_sample[send_pad]
                .sub(&eq_manual(pad_virtual, &sample_seeds[0].z, cube.main_bits))
                .add(&sample_seeds[0].beta.mul(&theta(0)));
        let mut sample_target = Fp3::ZERO;
        for (coefficient, entry) in expected_sample.iter().zip(&witness.main) {
            sample_target = sample_target.add(&coefficient.mul(entry));
        }
        let sample_row_value =
            sample_row(&shape, &registry, &sample_seeds, 0, sample_target).unwrap();
        assert_eq!(sample_row_value.main, expected_sample);
        assert_eq!(
            sample_row_value.evaluate(&witness).unwrap(),
            sample_target
        );

        // A G_u SEND pad changes only the sample relation.
        let mut changed = witness.clone();
        changed.main[send_pad] = changed.main[send_pad].add(&Fp3::ONE);
        assert_eq!(
            record_row_value.evaluate(&changed).unwrap(),
            record_row_value.target
        );
        assert_eq!(
            column_row_value.evaluate(&changed).unwrap(),
            column_target
        );
        assert_ne!(
            sample_row_value.evaluate(&changed).unwrap(),
            sample_target
        );
        // A tail entry changes the column and sample relations, not the record.
        let tail_virtual = shape.index.code.ell; // tail_first == ell in this fixture
        let tail_row = tail_virtual & ((1u64 << ceil_log2(shape.u.main_rows)) - 1);
        let tail_place = tail_virtual >> ceil_log2(shape.u.main_rows);
        let tail_offset = (tail_row * shape.u.code.ell + tail_place) as usize;
        assert_ne!(column_row_value.main[tail_offset], Fp3::ZERO);
        let mut changed = witness.clone();
        changed.main[tail_offset] = changed.main[tail_offset].add(&Fp3::ONE);
        assert_eq!(
            record_row_value.evaluate(&changed).unwrap(),
            record_row_value.target
        );
        assert_ne!(
            column_row_value.evaluate(&changed).unwrap(),
            column_target
        );
        assert_ne!(
            sample_row_value.evaluate(&changed).unwrap(),
            sample_target
        );
        // An index pad changes the record relation at least.
        let pad_place = shape.index.pad_place(0, 0);
        let pad_row = pad_place & ((1u64 << ceil_log2(shape.u.main_rows)) - 1);
        let pad_column = pad_place >> ceil_log2(shape.u.main_rows);
        let pad_offset = (pad_row * shape.u.code.ell + pad_column) as usize;
        assert_ne!(record_row_value.main[pad_offset], Fp3::ZERO);
        let mut changed = witness.clone();
        changed.main[pad_offset] = changed.main[pad_offset].add(&Fp3::ONE);
        assert_ne!(
            record_row_value.evaluate(&changed).unwrap(),
            record_row_value.target
        );
        // An ordinary entry changes all three.
        let ordinary = witness.main[0];
        let mut changed = witness.clone();
        changed.main[0] = changed.main[0].add(&Fp3::ONE);
        assert_ne!(ordinary, changed.main[0]);
        assert_ne!(
            record_row_value.evaluate(&changed).unwrap(),
            record_row_value.target
        );
        assert_ne!(
            column_row_value.evaluate(&changed).unwrap(),
            column_target
        );
        assert_ne!(
            sample_row_value.evaluate(&changed).unwrap(),
            sample_target
        );
    }

    #[test]
    fn gu_table_keeps_the_gap_when_tail_first_exceeds_ell() {
        let index = IndexShape {
            code: build_code_plan(4, 5, 3).unwrap(),
            v: 2,
            answers: 1,
            salt_bytes: LEAF_SALT_MIN_BYTES,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        };
        let shape = GuShape {
            oracle: 9,
            group: 1,
            tree: RuntimeTreePlan {
                tree_number: 5,
                leaf_class: 2,
                log_n: 4,
                log_n_zk: 0,
                parts: vec![RuntimeTreePart {
                    main_symbols: 6,
                    mask_symbols: 0,
                }],
                salt_bytes: LEAF_SALT_MIN_BYTES,
                run: Digest([9; 32]),
                configuration: Digest([7; 32]),
            },
            part: 0,
            rows_high: false,
            u: GuOracleShape {
                code: build_code_plan(4, 12, 0).unwrap(),
                main_rows: 2,
                main_deg: 3,
                mask_rows: 0,
                mask_code: None,
                answers: 1,
            },
            index,
        };
        assert!(shape.fault().is_none());
        assert!(gu_layout_fault(&shape).is_none());
        let message: Vec<Fp3> = (0..5).map(|index| Fp3::from_u64(10 + index)).collect();
        let tail: Vec<Fp3> = (0..3).map(|index| Fp3::from_u64(30 + index)).collect();
        let table = gu_table(&shape, &message, &tail).unwrap();
        // Independent gap/tail positions: step = 2^ceil_log2(3) = 4,
        // first = ceil(5/4)*4 = 8, so vector positions 5,6,7 are the gap.
        let step = 4u64;
        let first = 8u64;
        let row_bits = 1u32;
        let offset = |value: u64| {
            let row = value & ((1u64 << row_bits) - 1);
            let place = value >> row_bits;
            (row * shape.u.code.ell + place) as usize
        };
        for entry in 5..first {
            assert_eq!(table.main[offset(entry)], Fp3::ZERO);
        }
        for (entry, value) in tail.iter().enumerate() {
            assert_eq!(table.main[offset(first + entry as u64)], *value);
        }
        for (entry, value) in message.iter().enumerate() {
            assert_eq!(table.main[offset(entry as u64)], *value);
        }
        assert_eq!(step, first - 4);
        // A shape whose G_u pad would overlap the folded row is rejected at a
        // prerequisite; it is not claimed legal.
        let mut bad = shape.clone();
        bad.u.main_rows = 1;
        bad.u.code = build_code_plan(2, 4, 0).unwrap();
        bad.tree.log_n = 2;
        bad.tree.parts = vec![RuntimeTreePart {
            main_symbols: 3,
            mask_symbols: 0,
        }];
        assert!(bad.fault().is_some() || gu_layout_fault(&bad).is_some());
    }

    fn session_plan(gu: GuShape) -> FoldeqPlan {
        let index = gu.index;
        let rows = index.rows() as usize;
        let messages: Vec<Vec<Fp>> = (0..rows)
            .map(|row| {
                (0..index.code.ell)
                    .map(|entry| Fp::from_u64((row * 10 + entry as usize + 1) as u64))
                    .collect()
            })
            .collect();
        let tails: Vec<Vec<Fp>> = (0..rows)
            .map(|row| {
                (0..index.code.r)
                    .map(|entry| Fp::from_u64((row + entry as usize + 2) as u64))
                    .collect()
            })
            .collect();
        let salts: Vec<Vec<u8>> = (0..index.code.n() as usize)
            .map(|position| vec![(position as u8).wrapping_add(1); index.salt_bytes])
            .collect();
        let configuration = binding_parameters(&index).digest().unwrap();
        let table = test_commit_table(index, configuration, &messages, &tails, &salts).unwrap();
        let fold = FoldShape {
            code: build_code_plan(3, 4, 1).unwrap(),
            mask_code: build_code_plan(3, 4, 1).unwrap(),
            k: 2,
            h: 1,
            mask_leaf_class: 3,
        };
        let tree = MaskTreePlan {
            tree_number: 5,
            leaf_class: 3,
            log_n: fold.mask_code.log_n,
            mask_rows: fold.mask_rows().unwrap(),
            salt_bytes: LEAF_SALT_MIN_BYTES,
            run: Digest([2; 32]),
            configuration: Digest([7; 32]),
        };
        let statement = Arc::new(
            VerifiedStatement::new(
                VerifiedGroups {
                    groups: vec![VerifiedGroup {
                        group: 1,
                        members: vec![10],
                        fresh: Some(9),
                        code: None,
                        mask_code: None,
                    }],
                },
                vec![tree],
                vec![OracleMetadata {
                    id: 9,
                    group: 1,
                    tree_number: tree.tree_number,
                    leaf_class: tree.leaf_class,
                    mask_rows: tree.mask_rows,
                    mask_code: build_code_plan(3, 4, 1).unwrap(),
                }],
            )
            .unwrap(),
        );
        FoldeqPlan {
            reader: table.reader.clone(),
            binding: binding_for(&table.reader),
            edge: FoldeqEdgeBinding::NotAnEdge,
            fold,
            statement,
            oracles: Arc::new(Mutex::new(RuntimeOracles::default())),
            mask_tree_number: tree.tree_number,
            group_id: 1,
            leaf_class: 3,
            seed_budget: gu.index.answers as usize,
            ledger: Arc::new(Mutex::new(ReadLedger::new(2))),
            column_draws: 2,
            mask_query_count: 2,
            gamma_path: GammaPath::Direct {
                columns: gu.index.v,
            },
            gu: Some(gu),
            omega: None,
            fold_layout: None,
        }
    }

    fn session_gu() -> GuShape {
        let mut gu = shape();
        gu.oracle = 10;
        gu
    }






    #[test]
    fn plan_first_query_binds_the_trusted_gu_shape() {
        let gu = session_gu();
        let plan = session_plan(gu.clone());
        assert!(plan.fault().is_none());
        let base = plan.first_query();
        let mut other = plan.clone();
        if let Some(shape) = other.gu.as_mut() {
            shape.tree.tree_number += 1;
        }
        assert_ne!(base, other.first_query());
        let mut other = plan.clone();
        if let Some(shape) = other.gu.as_mut() {
            shape.oracle += 1;
        }
        assert_ne!(base, other.first_query());
        let mut other = plan.clone();
        if let Some(shape) = other.gu.as_mut() {
            shape.tree.leaf_class = 9;
        }
        assert_ne!(base, other.first_query());
        let mut other = plan.clone();
        other.reader.record.y[0] = other.reader.record.y[0].add(&Fp3::ONE);
        assert_ne!(base, other.first_query());
        let mut other = plan;
        if let Some(shape) = other.gu.as_mut() {
            shape.group = 2;
        }
        assert!(other.fault().is_some());
    }
}
