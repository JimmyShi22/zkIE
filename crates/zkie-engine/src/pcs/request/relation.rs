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
}
