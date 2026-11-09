//! Runtime mask trees: the tree that carries a fresh mask's rows (foundation
//! only).
//!
//! A runtime tree is laid out by the verifier's plan: tree number, leaf class,
//! depth, mask-row count, salt length and the run/configuration starts are
//! plan values, never proof values. Its leaves use the runtime leaf domain and
//! the run's value as the hash start; internal nodes use the tree node domain,
//! the same tree number and the configuration digest. Stored table leaves stay
//! under their own domain and configuration. The byte layout is the
//! repository's internal candidate: domain separation is kept, but no
//! interoperability with another representation is claimed.

use serde::{Deserialize, Serialize};
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::compute::cpu::merkle_node;
use crate::pcs::request::code::encode_row_fp3;
use crate::pcs::request::field::{Fp, Fp3, E_DEGREE};
use crate::pcs::request::fold::FoldShape;
use crate::pcs::request::params::{LEAF_SALT_MAX_BYTES, LEAF_SALT_MIN_BYTES};
use crate::primitives::codec::CanonicalWriter;
use crate::primitives::transcript::{hash_labeled, tag_word, LEAF_MAIN, TREE_NODE};

/// The verifier's layout of one runtime tree that carries mask rows.
///
/// The tree has `2^log_n` columns; every column holds the mask rows' symbols
/// (`mask_rows * E_DEGREE` base-field elements) and one salt. The tree number
/// and leaf class are plan values; the run value binds the leaves to one run
/// and the configuration digest starts the internal nodes. `mask_rows` is the
/// number of mask rows (a power of two); it need not equal the column count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaskTreePlan {
    pub tree_number: u64,
    pub leaf_class: u32,
    pub log_n: u32,
    pub mask_rows: u64,
    pub salt_bytes: usize,
    pub run: Digest,
    pub configuration: Digest,
}

impl MaskTreePlan {
    pub fn width(&self) -> Result<u64> {
        self.mask_rows
            .checked_mul(E_DEGREE as u64)
            .ok_or(Error::ResourceLimit)
    }

    pub fn fault(&self) -> Option<&'static str> {
        if self.tree_number >= (1u64 << 16) {
            return Some("the runtime tree number does not fit the tag");
        }
        if self.log_n < 1 || self.log_n > 60 {
            return Some("the runtime tree depth is not a tree");
        }
        if self.mask_rows == 0 || !self.mask_rows.is_power_of_two() {
            return Some("the runtime tree has no power-of-two mask rows");
        }
        let width = match self.width() {
            Ok(width) => width,
            Err(_) => return Some("the runtime tree width overflows"),
        };
        if width >> 24 != 0 {
            return Some("the runtime tree column is too wide");
        }
        if self.salt_bytes < LEAF_SALT_MIN_BYTES || self.salt_bytes > LEAF_SALT_MAX_BYTES {
            return Some("the runtime tree salt length is not admitted");
        }
        if self.salt_bytes % 8 != 0 {
            return Some("the runtime tree salt length is not a multiple of eight");
        }
        if self.run == Digest::ZERO || self.configuration == Digest::ZERO {
            return Some("the runtime tree has no run or configuration start");
        }
        None
    }

    pub fn columns(&self) -> u64 {
        1u64 << self.log_n
    }
}

/// One opening of a runtime mask column.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaskColumnOpening {
    pub position: u64,
    pub salt: Vec<u8>,
    pub symbols: Vec<Fp>,
    pub path: Vec<Digest>,
}

fn mask_leaf(
    plan: &MaskTreePlan,
    position: u64,
    salt: &[u8],
    symbols: &[Fp],
) -> Result<Digest> {
    let width = plan.width()?;
    if symbols.len() as u64 != width {
        return Err(Error::ShapeMismatch);
    }
    let index = (1u64 << plan.log_n)
        .checked_add(position)
        .ok_or(Error::ResourceLimit)?;
    let mut writer = CanonicalWriter::new();
    writer.put_u64(width);
    writer.put_u64(salt.len() as u64);
    for symbol in symbols {
        writer.put_fp(*symbol);
    }
    writer.put_bytes(salt);
    hash_labeled(
        tag_word(LEAF_MAIN, plan.tree_number),
        index,
        Some(&plan.run),
        &writer.into_bytes(),
    )
}

fn mask_node(plan: &MaskTreePlan, left: &Digest, right: &Digest) -> Result<Digest> {
    merkle_node(
        tag_word(TREE_NODE, plan.tree_number),
        0,
        &plan.configuration,
        left,
        right,
    )
}

/// A committed runtime mask tree with the private rows needed to open columns.
#[derive(Clone, Debug)]
pub struct MaskTree {
    pub plan: MaskTreePlan,
    pub root: Digest,
    levels: Vec<Vec<Digest>>,
    salts: Vec<Vec<u8>>,
    codewords: Vec<Vec<Fp3>>,
}

impl MaskTree {
    /// Commit `mask_rows` mask messages (`ell` entries each) with their
    /// encoding tails (`r` entries each) and one salt per column.
    pub fn commit(
        plan: MaskTreePlan,
        shape: &FoldShape,
        rows: &[Vec<Fp3>],
        tails: &[Vec<Fp3>],
        salts: &[Vec<u8>],
    ) -> Result<Self> {
        if plan.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        if plan.log_n != shape.mask_code.log_n {
            return Err(Error::ShapeMismatch);
        }
        let expected_rows = shape.mask_rows()?;
        if plan.mask_rows != expected_rows {
            return Err(Error::ShapeMismatch);
        }
        let columns = plan.columns() as usize;
        if rows.len() as u64 != expected_rows
            || tails.len() as u64 != expected_rows
            || salts.len() != columns
        {
            return Err(Error::ShapeMismatch);
        }
        for row in rows {
            if row.len() as u64 != shape.mask_code.ell {
                return Err(Error::ShapeMismatch);
            }
        }
        for tail in tails {
            if tail.len() as u64 != shape.mask_code.r {
                return Err(Error::ShapeMismatch);
            }
        }
        for salt in salts {
            if salt.len() != plan.salt_bytes {
                return Err(Error::ShapeMismatch);
            }
        }
        let mut codewords = Vec::with_capacity(rows.len());
        for (row, tail) in rows.iter().zip(tails) {
            codewords.push(encode_row_fp3(&shape.mask_code, row, tail)?);
        }
        let width = plan.width()?;
        let mut leaves = Vec::with_capacity(columns);
        for position in 0..columns {
            let mut symbols = Vec::with_capacity(width as usize);
            for codeword in &codewords {
                for coefficient in codeword[position].c {
                    symbols.push(coefficient);
                }
            }
            leaves.push(mask_leaf(
                &plan,
                position as u64,
                &salts[position],
                &symbols,
            )?);
        }
        let mut levels = vec![leaves];
        while levels.last().expect("one level").len() > 1 {
            let previous = levels.last().expect("one level");
            let mut next = Vec::with_capacity(previous.len() / 2);
            for pair in previous.chunks_exact(2) {
                next.push(mask_node(&plan, &pair[0], &pair[1])?);
            }
            levels.push(next);
        }
        let root = levels.last().expect("one level")[0];
        Ok(Self {
            plan,
            root,
            levels,
            salts: salts.to_vec(),
            codewords,
        })
    }

    /// Open the columns at `positions` (sorted, without repeats).
    pub fn open(&self, positions: &[u64]) -> Result<Vec<MaskColumnOpening>> {
        let columns = self.plan.columns();
        let width = self.plan.width()? as usize;
        let mut openings = Vec::with_capacity(positions.len());
        let mut previous = None;
        for position in positions {
            if *position >= columns {
                return Err(Error::InvalidIndex);
            }
            if let Some(previous) = previous {
                if *position <= previous {
                    return Err(Error::InvalidIndex);
                }
            }
            previous = Some(*position);
            let mut symbols = Vec::with_capacity(width);
            for codeword in &self.codewords {
                for coefficient in codeword[*position as usize].c {
                    symbols.push(coefficient);
                }
            }
            openings.push(MaskColumnOpening {
                position: *position,
                salt: self.salts[*position as usize].clone(),
                symbols,
                path: self.path(*position as usize)?,
            });
        }
        Ok(openings)
    }

    fn path(&self, index: usize) -> Result<Vec<Digest>> {
        if index >= self.plan.columns() as usize {
            return Err(Error::InvalidIndex);
        }
        let mut path = Vec::with_capacity(self.levels.len().saturating_sub(1));
        let mut position = index;
        for level in &self.levels[..self.levels.len() - 1] {
            path.push(level[position ^ 1]);
            position >>= 1;
        }
        Ok(path)
    }
}

/// Verify runtime mask openings against the plan and root. The positions must
/// be the verifier's drawn list, sorted and without repeats; the caller checks
/// that list against its plan.
pub fn verify_mask_columns(
    plan: &MaskTreePlan,
    root: &Digest,
    positions: &[u64],
    openings: &[MaskColumnOpening],
) -> Result<Vec<Fp>> {
    if plan.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    if positions.len() != openings.len() {
        return Err(Error::ShapeMismatch);
    }
    let columns = plan.columns();
    let width = plan.width()? as usize;
    let mut symbols = Vec::with_capacity(
        positions
            .len()
            .checked_mul(width)
            .ok_or(Error::ResourceLimit)?,
    );
    let mut previous = None;
    for (position, opening) in positions.iter().zip(openings) {
        if *position >= columns {
            return Err(Error::InvalidIndex);
        }
        if let Some(previous) = previous {
            if *position <= previous {
                return Err(Error::InvalidIndex);
            }
        }
        previous = Some(*position);
        if opening.position != *position
            || opening.salt.len() != plan.salt_bytes
            || opening.symbols.len() != width
            || opening.path.len() != plan.log_n as usize
        {
            return Err(Error::ShapeMismatch);
        }
        let leaf = mask_leaf(plan, *position, &opening.salt, &opening.symbols)?;
        let mut current = leaf;
        let mut at = *position as usize;
        for sibling in &opening.path {
            current = if at & 1 == 0 {
                mask_node(plan, &current, sibling)?
            } else {
                mask_node(plan, sibling, &current)?
            };
            at >>= 1;
        }
        if current != *root {
            return Err(Error::InvalidProof);
        }
        symbols.extend_from_slice(&opening.symbols);
    }
    Ok(symbols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::request::code::build_code_plan;
    use crate::pcs::request::field::Fp;
    use crate::pcs::request::fold::fold_mask_columns;
    use zkie_core::common::field::PrimeCharacteristicRing;

    fn shape() -> FoldShape {
        FoldShape {
            code: build_code_plan(4, 8, 1).unwrap(),
            mask_code: build_code_plan(3, 4, 1).unwrap(),
            k: 2,
            h: 1,
            mask_leaf_class: 3,
        }
    }

    fn plan() -> MaskTreePlan {
        let shape = shape();
        MaskTreePlan {
            tree_number: 5,
            leaf_class: 3,
            log_n: shape.mask_code.log_n,
            mask_rows: shape.mask_rows().unwrap(),
            salt_bytes: LEAF_SALT_MIN_BYTES,
            run: Digest([9; 32]),
            configuration: Digest([7; 32]),
        }
    }

    fn rows(shape: &FoldShape) -> Vec<Vec<Fp3>> {
        let columns = fold_mask_columns(shape.k, shape.h);
        (0..shape.mask_rows().unwrap())
            .map(|row| {
                let mut entries = vec![Fp3::ZERO; shape.mask_code.ell as usize];
                for (index, entry) in entries.iter_mut().take(columns).enumerate() {
                    *entry = Fp3::from_u64(row * 100 + index as u64 + 1);
                }
                entries
            })
            .collect()
    }

    fn tails(shape: &FoldShape) -> Vec<Vec<Fp3>> {
        (0..shape.mask_rows().unwrap())
            .map(|row| {
                (0..shape.mask_code.r)
                    .map(|index| Fp3::from_u64(row + index + 1))
                    .collect()
            })
            .collect()
    }

    fn salts(_shape: &FoldShape, plan: &MaskTreePlan) -> Vec<Vec<u8>> {
        (0..plan.columns() as usize)
            .map(|position| vec![(position as u8).wrapping_add(1); plan.salt_bytes])
            .collect()
    }

    fn tree() -> MaskTree {
        let shape = shape();
        let plan = plan();
        let rows = rows(&shape);
        let tails = tails(&shape);
        let salts = salts(&shape, &plan);
        MaskTree::commit(plan, &shape, &rows, &tails, &salts).unwrap()
    }

    #[test]
    fn runtime_mask_tree_round_trip() {
        let shape = shape();
        let tree = tree();
        assert_ne!(tree.root, Digest::ZERO);
        let positions = [1u64, 4, 6];
        let openings = tree.open(&positions).unwrap();
        let symbols = verify_mask_columns(&tree.plan, &tree.root, &positions, &openings).unwrap();
        assert_eq!(symbols.len(), positions.len() * tree.plan.width().unwrap() as usize);
        let _ = shape;
    }

    #[test]
    fn tampered_mask_openings_are_rejected() {
        let tree = tree();
        let positions = [1u64, 4];
        let openings = tree.open(&positions).unwrap();
        let mut tampered = openings.clone();
        tampered[0].symbols[0] = tampered[0].symbols[0] + Fp::ONE;
        assert!(verify_mask_columns(&tree.plan, &tree.root, &positions, &tampered).is_err());
        let mut tampered = openings.clone();
        tampered[0].salt[0] ^= 1;
        assert!(verify_mask_columns(&tree.plan, &tree.root, &positions, &tampered).is_err());
        let mut tampered = openings.clone();
        tampered[0].path[0] = Digest([3; 32]);
        assert!(verify_mask_columns(&tree.plan, &tree.root, &positions, &tampered).is_err());
        let mut tampered = openings;
        tampered[0].position = 2;
        assert!(verify_mask_columns(&tree.plan, &tree.root, &positions, &tampered).is_err());
    }

    #[test]
    fn runtime_plan_values_are_bound() {
        let tree = tree();
        let positions = [1u64, 4];
        let openings = tree.open(&positions).unwrap();
        let mut other = tree.plan;
        other.tree_number += 1;
        assert!(verify_mask_columns(&other, &tree.root, &positions, &openings).is_err());
        let mut other = tree.plan;
        other.run = Digest([8; 32]);
        assert!(verify_mask_columns(&other, &tree.root, &positions, &openings).is_err());
        let mut other = tree.plan;
        other.configuration = Digest([6; 32]);
        assert!(verify_mask_columns(&other, &tree.root, &positions, &openings).is_err());
    }

    #[test]
    fn mask_tree_plan_faults_are_rejected() {
        let base = plan();
        let mut bad = base;
        bad.tree_number = 1 << 16;
        assert!(bad.fault().is_some());
        let mut bad = base;
        bad.mask_rows = 3;
        assert!(bad.fault().is_some());
        let mut bad = base;
        bad.salt_bytes = LEAF_SALT_MIN_BYTES - 1;
        assert!(bad.fault().is_some());
        let mut bad = base;
        bad.run = Digest::ZERO;
        assert!(bad.fault().is_some());
        let mut bad = base;
        bad.log_n = 64;
        assert!(bad.fault().is_some());
    }
}
