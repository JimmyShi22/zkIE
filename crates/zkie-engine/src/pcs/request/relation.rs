//! Local linear relation rows for G_u claims (foundation only).
//!
//! These builders construct the coefficient rows and public targets of the
//! three G_u relations on small in-memory tables. They are local relation
//! construction and mathematical checks, not a verifier: the real verifier
//! does not hold the private G_u tables, and the chain, group, opening and
//! soundness paths are not connected. Fixtures used by the tests supply the
//! seeds and witnesses; their source is not authenticated.

use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::error::{Error, Result};

use crate::pcs::request::code::factors;
use crate::pcs::request::field::{Fp, Fp3, E_DEGREE};
use crate::pcs::request::gu::{GuRegistry, GuShape};
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

fn theta(coefficient: usize) -> Fp3 {
    match coefficient {
        0 => Fp3::ONE,
        1 => Fp3::new(Fp::ZERO, Fp::ONE, Fp::ZERO),
        2 => Fp3::new(Fp::ZERO, Fp::ZERO, Fp::ONE),
        _ => Fp3::ZERO,
    }
}

/// The table index of folded-row vector entry `j` under the trusted G_u tree
/// shape's row/place layout.
fn vector_index(shape: &GuShape, j: u64) -> Result<usize> {
    let row_bits = ceil_log2(shape.u.main_rows);
    let place_bits = shape.u.code.log_ell();
    if !shape.rows_high && shape.u.main_rows != (1u64 << row_bits) {
        return Err(Error::ShapeMismatch);
    }
    let index = if shape.rows_high {
        (j >> place_bits)
            | ((j & ((1u64 << place_bits) - 1)) << row_bits)
    } else {
        j
    };
    let entries = shape
        .u
        .main_rows
        .checked_mul(shape.u.code.ell)
        .ok_or(Error::ResourceLimit)?;
    if index >= entries {
        return Err(Error::ShapeMismatch);
    }
    usize::try_from(index).map_err(|_| Error::ResourceLimit)
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

fn empty_main(shape: &GuShape) -> Result<Vec<Fp3>> {
    let len = shape
        .u
        .main_rows
        .checked_mul(shape.u.code.ell)
        .ok_or(Error::ResourceLimit)?;
    usize::try_from(len)
        .map(|len| vec![Fp3::ZERO; len])
        .map_err(|_| Error::ResourceLimit)
}

fn empty_mask(shape: &GuShape) -> Result<Vec<Fp3>> {
    let len = match shape.u.mask_code {
        Some(code) => shape
            .u
            .mask_rows
            .checked_mul(code.ell)
            .ok_or(Error::ResourceLimit)?,
        None => 0,
    };
    usize::try_from(len)
        .map(|len| vec![Fp3::ZERO; len])
        .map_err(|_| Error::ResourceLimit)
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
    let mut main = empty_main(shape)?;
    let mut mask = empty_mask(shape)?;
    for entry in 0..shape.index.code.ell {
        let index = vector_index(shape, entry)?;
        main[index] = main[index].add(&eq_index(entry, &seed.z, bits)?);
    }
    for pad_answer in 0..shape.index.answers {
        for coefficient in 0..E_DEGREE {
            let place = shape.index.pad_place(pad_answer, coefficient);
            let index = vector_index(shape, place)?;
            let under = eq_index(place, &seed.z, bits)?;
            let mut value = Fp3::ZERO.sub(&under);
            if pad_answer == answer {
                value = value.add(&seed.beta.mul(&theta(coefficient)));
            }
            main[index] = main[index].add(&value);
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

/// Build one column-query relation row: the generator coefficients of the
/// indexed code at the drawn position, placed on the G_u main table, with the
/// target equal to the authenticated index column folded at `eq_gamma`.
pub fn column_row(
    shape: &GuShape,
    registry: &GuRegistry,
    position: u64,
    column: &[Fp3],
    eq_gamma: &[Fp3],
) -> Result<GuRelationRow> {
    check_binding(shape, registry)?;
    let rows = usize::try_from(shape.index.rows()).map_err(|_| Error::ResourceLimit)?;
    if position >= shape.index.code.n() || column.len() != rows || eq_gamma.len() != rows {
        return Err(Error::ShapeMismatch);
    }
    let message = generator_message(&shape.index.code, position)?;
    let tail = generator_tail(&shape.index.code, position)?;
    let first = tail_first(&shape.index.code)?;
    let mut main = empty_main(shape)?;
    let mut mask = empty_mask(shape)?;
    for (entry, coefficient) in message.iter().enumerate() {
        let index = vector_index(shape, entry as u64)?;
        main[index] = main[index].add(&Fp3::from_base(*coefficient));
    }
    for (entry, coefficient) in tail.iter().enumerate() {
        let index = vector_index(shape, first + entry as u64)?;
        main[index] = main[index].add(&Fp3::from_base(*coefficient));
    }
    mask.iter_mut().for_each(|coefficient| *coefficient = Fp3::ZERO);
    let mut target = Fp3::ZERO;
    for (weight, value) in eq_gamma.iter().zip(column) {
        target = target.add(&weight.mul(value));
    }
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

/// Build one row per logical draw in the trusted order. Each claim carries its
/// own position and authenticated column; the positions must equal the trusted
/// draw list exactly (multiplicity and order included). Physical openings may
/// be deduplicated separately with `unique_positions`.
pub fn column_rows_from_draws(
    shape: &GuShape,
    registry: &GuRegistry,
    trusted_draws: &[u64],
    claimed: &[(u64, Vec<Fp3>)],
    eq_gamma: &[Fp3],
) -> Result<Vec<GuRelationRow>> {
    if trusted_draws.is_empty() || trusted_draws.len() != claimed.len() {
        return Err(Error::StatementMismatch);
    }
    let positions: Vec<u64> = claimed.iter().map(|(position, _)| *position).collect();
    validate_draw_sequence(trusted_draws, &positions)?;
    let mut rows = Vec::with_capacity(claimed.len());
    for (position, column) in claimed {
        rows.push(column_row(shape, registry, *position, column, eq_gamma)?);
    }
    Ok(rows)
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
    let u = &shape.u;
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
        let index = (rows - 1)
            .checked_add(place.checked_shl(row_bits).ok_or(Error::ResourceLimit)?)
            .ok_or(Error::ResourceLimit)?;
        let under = front.mul(&eq_index(index, &seed.z, bits)?);
        let mut value = Fp3::ZERO.sub(&under);
        if pad_answer == target {
            value = value.add(&seed.beta.mul(&theta(0)));
        }
        let slot = usize::try_from(index).map_err(|_| Error::ResourceLimit)?;
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
    if answer >= shape.u.answers || seeds.len() < shape.u.answers as usize {
        return Err(Error::ShapeMismatch);
    }
    let cube = sample_cube(shape)?;
    let seed = &seeds[answer as usize];
    if seed.z.len() < cube.bits as usize || seed.beta == Fp3::ZERO {
        return Err(Error::ShapeMismatch);
    }
    let mut main = empty_main(shape)?;
    let mut mask = empty_mask(shape)?;
    if cube.has_main {
        let front = sample_front(seed, &cube, cube.main_bits, false)?;
        for index in 0..main.len() {
            main[index] = front.mul(&eq_index(index as u64, &seed.z, cube.main_bits)?);
        }
        if !cube.has_mask {
            sample_pad(
                &mut main,
                seed,
                shape.u.answers,
                answer,
                shape.u.main_rows,
                cube.main_row_bits,
                shape.u.code.ell,
                cube.main_bits,
                front,
            )?;
        }
    }
    if cube.has_mask {
        let code = shape.u.mask_code.ok_or(Error::ShapeMismatch)?;
        let front = sample_front(seed, &cube, cube.mask_bits, true)?;
        for index in 0..mask.len() {
            mask[index] = front.mul(&eq_index(index as u64, &seed.z, cube.mask_bits)?);
        }
        sample_pad(
            &mut mask,
            seed,
            shape.u.answers,
            answer,
            shape.u.mask_rows,
            cube.mask_row_bits,
            code.ell,
            cube.mask_bits,
            front,
        )?;
    }
    Ok(GuRelationRow {
        oracle: shape.oracle,
        group: shape.group,
        kind: GuRelationKind::Sample,
        main,
        mask,
        target,
    })
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
        if seed.z.len() < bits as usize || seed.beta == Fp3::ZERO {
            return Err(Error::ShapeMismatch);
        }
    }
    Ok(bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::request::code::build_code_plan;
    use crate::pcs::request::gu::{GuNotice, GuOracleShape, GuShape};
    use crate::pcs::request::params::LEAF_SALT_MIN_BYTES;
    use crate::pcs::request::runtime_tree::{RuntimeTreePart, RuntimeTreePlan};
    use crate::pcs::request::table::{index_name, IndexShape, TableDistances};
    use zkie_core::types::Digest;

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
            expected[entry as usize] = eq_manual(entry, &seeds[0].z, bits);
        }
        for coefficient in 0..E_DEGREE {
            let place = shape.index.pad_place(0, coefficient);
            expected[place as usize] = seeds[0].beta.mul(&theta(coefficient));
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
            expected[entry as usize] =
                Fp3::from_base(encode_symbol(code, &message, &tail, position));
        }
        let first = tail_first(code).unwrap();
        for entry in 0..code.r {
            let message = vec![Fp::ZERO; code.ell as usize];
            let mut tail = vec![Fp::ZERO; code.r as usize];
            tail[entry as usize] = Fp::ONE;
            expected[(first + entry) as usize] =
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
        assert_ne!(row.main[first as usize], Fp3::ZERO);
        let mut changed = witness.clone();
        changed.main[first as usize] = changed.main[first as usize].add(&Fp3::ONE);
        assert_ne!(row.evaluate(&changed).unwrap(), manual);
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
        let claimed = vec![
            (1u64, column.clone()),
            (1u64, column.clone()),
            (3u64, column.clone()),
        ];
        let rows =
            column_rows_from_draws(&shape, &registry, &trusted, &claimed, &eq_gamma).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], rows[1]);
        let mut duplicate_permuted = claimed.clone();
        duplicate_permuted[1].0 = 3;
        duplicate_permuted[2].0 = 1;
        assert!(column_rows_from_draws(
            &shape,
            &registry,
            &trusted,
            &duplicate_permuted,
            &eq_gamma
        )
        .is_err());
        assert!(column_rows_from_draws(
            &shape,
            &registry,
            &trusted,
            &claimed[..2],
            &eq_gamma
        )
        .is_err());
        let reordered = vec![
            (3u64, column.clone()),
            (1u64, column.clone()),
            (1u64, column.clone()),
        ];
        assert!(column_rows_from_draws(&shape, &registry, &trusted, &reordered, &eq_gamma).is_err());
        assert_eq!(unique_positions(&trusted), vec![1, 3]);
        assert!(validate_draw_sequence(&trusted, &[1, 1, 3]).is_ok());
        assert!(validate_draw_sequence(&trusted, &[1, 3, 3]).is_err());
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

        let first_witness = sample_witness(&first);
        let mut expected = vec![Fp3::ZERO; first_witness.main.len()];
        let cube = sample_cube(&first).unwrap();
        for index in 0..expected.len() {
            expected[index] = Fp3::ONE
                .mul(&eq_manual(index as u64, &seeds[0].z, cube.main_bits));
        }
        let rows = first.u.main_rows;
        for pad_answer in 0..first.u.answers {
            let place = first.u.code.ell - first.u.answers as u64 + pad_answer as u64;
            let index = (rows - 1) + (place << cube.main_row_bits);
            expected[index as usize] = if pad_answer == 0 {
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
            for index in 0..expected.len() {
                expected[index] = eq_manual(
                    index as u64,
                    &seeds[answer as usize].z,
                    cube.main_bits,
                );
            }
            let rows = second.u.main_rows;
            for pad_answer in 0..second.u.answers {
                let place = second.u.code.ell - second.u.answers as u64 + pad_answer as u64;
                let index = (rows - 1) + (place << cube.main_row_bits);
                expected[index as usize] = if pad_answer == answer {
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
        for index in 0..expected_main.len() {
            expected_main[index] =
                main_front.mul(&eq_manual(index as u64, &seed.z, cube.main_bits));
        }
        let mask_code = shape.u.mask_code.unwrap();
        let mut expected_mask = vec![Fp3::ZERO; witness.mask.len()];
        for index in 0..expected_mask.len() {
            expected_mask[index] =
                mask_front.mul(&eq_manual(index as u64, &seed.z, cube.mask_bits));
        }
        let place = mask_code.ell - shape.u.answers as u64;
        let pad = (shape.u.mask_rows - 1) + (place << cube.mask_row_bits);
        expected_mask[pad as usize] = seed.beta.mul(&theta(0));
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
        changed.mask[pad as usize] = changed.mask[pad as usize].add(&Fp3::ONE);
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
        let place_bits = shape.u.code.log_ell();
        let row_bits = 1u32;
        let index_of = |entry: u64| (entry >> place_bits) | ((entry & ((1u64 << place_bits) - 1)) << row_bits);
        for entry in 0..shape.index.code.ell {
            expected[index_of(entry) as usize] = eq_manual(entry, &seeds[0].z, bits);
        }
        for coefficient in 0..E_DEGREE {
            let place = shape.index.pad_place(0, coefficient);
            expected[index_of(place) as usize] = seeds[0].beta.mul(&theta(coefficient));
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
        let place_bits = shape.u.code.log_ell();
        let row_bits = 2u32;
        let index_of = |entry: u64| (entry >> place_bits) | ((entry & ((1u64 << place_bits) - 1)) << row_bits);
        let mut expected = vec![Fp3::ZERO; (shape.u.main_rows * shape.u.code.ell) as usize];
        expected[index_of(0) as usize] = eq_manual(0, &seeds[0].z, shape.index.cube_bits());
        for coefficient in 0..E_DEGREE {
            let place = shape.index.pad_place(0, coefficient);
            expected[index_of(place) as usize] = seeds[0].beta.mul(&theta(coefficient));
        }
        assert_eq!(row.main, expected);
        let witness = witness(&shape);
        let mut manual = Fp3::ZERO;
        for (coefficient, entry) in expected.iter().zip(&witness.main) {
            manual = manual.add(&coefficient.mul(entry));
        }
        assert_eq!(row.evaluate(&witness).unwrap(), manual);
        assert!(vector_index(&shape, 8).is_err());

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
            expected[index_of(entry) as usize] =
                Fp3::from_base(encode_symbol(code, &message, &tail, 2));
        }
        let first = tail_first(code).unwrap();
        for entry in 0..code.r {
            let message = vec![Fp::ZERO; code.ell as usize];
            let mut tail = vec![Fp::ZERO; code.r as usize];
            tail[entry as usize] = Fp::ONE;
            expected[index_of(first + entry) as usize] =
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
            expected[entry as usize] = eq_manual(entry, &seeds[1].z, bits);
        }
        for coefficient in 0..E_DEGREE {
            let answer_zero = shape.index.pad_place(0, coefficient);
            let answer_one = shape.index.pad_place(1, coefficient);
            expected[answer_zero as usize] = Fp3::ZERO;
            expected[answer_one as usize] = seeds[1].beta.mul(&theta(coefficient));
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
}
