//! Runtime co-trees: the trees that carry a prover message's oracle parts
//! (foundation only).
//!
//! A runtime tree is laid out by the verifier's plan: tree number, leaf class,
//! long and short depths, part widths, salt length and the run/configuration
//! starts are plan values, never proof values. A long-column leaf holds every
//! part's main symbols in part order, then every part's mask symbols at their
//! short positions in part order, then the leaf's own salt. Leaves use the
//! runtime leaf domain and the run's value as the hash start; internal nodes
//! use the tree node domain, the same tree number and the configuration
//! digest. Stored table leaves stay under their own domain and configuration
//! and are never laid out here. A mask-only tree is the one-part special case;
//! commit, leaf hashing and the whole-tree opening are shared with every other
//! part count. The tree is an in-memory test foundation: it materializes every
//! column and salt, is not the production committer, and the verifier cannot
//! see whether the private salts are random. The root is a tree root, not a
//! value that a session has authenticated; the session/group wiring is open.
//!
//! The leaf hash input is the current isolated candidate format: a `u64`
//! column width, a `u64` salt length, every symbol in canonical little-endian
//! form, then `CanonicalWriter::put_bytes(salt)`, which writes the salt length
//! **again** followed by the salt bytes. The tag is the runtime leaf domain
//! with the tree number, the index is `2^log_n + x`, and the hash starts from
//! `run`. The stored `column_leaf` encoding has a similar shape; that alone
//! does not establish that this runtime form is frozen. The reference leaf
//! hash has no length fields; this repository makes no interoperability claim
//! and the global format is not changed.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::compute::cpu::merkle_node;
use crate::pcs::request::code::encode_row_fp3;
use crate::pcs::request::field::{Fp, Fp3, E_DEGREE};
use crate::pcs::request::fold::FoldShape;
use crate::pcs::request::params::{LEAF_SALT_MAX_BYTES, LEAF_SALT_MIN_BYTES};
use crate::primitives::codec::{CanonicalReader, CanonicalWriter};
use zkie_core::common::codec::DecodeLimits;
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

impl MaskTreePlan {
    /// The same tree as a single mask-only part of a general runtime tree:
    /// the mask-only tree is the one-part special case, not another encoding.
    pub fn runtime_plan(&self) -> Result<RuntimeTreePlan> {
        let mask_symbols = self
            .mask_rows
            .checked_mul(E_DEGREE as u64)
            .ok_or(Error::ResourceLimit)?;
        Ok(RuntimeTreePlan {
            tree_number: self.tree_number,
            leaf_class: self.leaf_class,
            log_n: self.log_n,
            log_n_zk: self.log_n,
            parts: vec![RuntimeTreePart {
                main_symbols: 0,
                mask_symbols,
            }],
            salt_bytes: self.salt_bytes,
            run: self.run,
            configuration: self.configuration,
        })
    }
}

/// A committed runtime mask tree: the one-part mask-only special case of the
/// general runtime tree. It keeps its plan and root; commit and open use the
/// same leaf encoding and whole-tree opening as every other runtime tree.
#[derive(Clone, Debug)]
pub struct MaskTree {
    pub plan: MaskTreePlan,
    pub root: Digest,
    inner: RuntimeTree,
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
        let _profile_commit = crate::profile::Scope::new("mask_tree.commit");
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
        let mut codewords = Vec::with_capacity(rows.len());
        for (row, tail) in rows.iter().zip(tails) {
            codewords.push(encode_row_fp3(&shape.mask_code, row, tail)?);
        }
        let inner = RuntimeTree::commit(
            plan.runtime_plan()?,
            &[RuntimePartRows {
                main: Vec::new(),
                mask: codewords,
                main_deg: 1,
            }],
            salts,
        )?;
        Ok(Self {
            plan,
            root: *inner.root(),
            inner,
        })
    }

    /// Open the complete leaves at `positions` (strictly increasing, unique).
    pub fn open(&self, positions: &[u64]) -> Result<RuntimeTreeOpening> {
        self.inner.open(positions)
    }
}

/// Check a whole-tree opening of a mask-only tree. The mask-only tree is the
/// one-part special case: the same full leaves and shared-sibling walk as the
/// general runtime tree.
pub fn verify_mask_opening(
    plan: &MaskTreePlan,
    root: &Digest,
    positions: &[u64],
    opening: &RuntimeTreeOpening,
) -> Result<Vec<Fp>> {
    let runtime = plan.runtime_plan()?;
    verify_runtime_opening(&runtime, root, positions, opening)?;
    let width = plan.width()? as usize;
    let mut symbols = Vec::with_capacity(
        positions
            .len()
            .checked_mul(width)
            .ok_or(Error::ResourceLimit)?,
    );
    for column in &opening.columns {
        symbols.extend_from_slice(&column.symbols);
    }
    Ok(symbols)
}

/// One oracle message's planned shape for a runtime tree layout. The plan
/// supplies the numbers; nothing here comes from the proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeOracleShape {
    pub leaf_class: u32,
    pub base_log_n: u32,
    pub main_rows: u64,
    pub main_deg: u32,
    pub mask_rows: u64,
    pub mask_log_n: u32,
}

/// One part's column widths in a runtime tree: its main rows and mask rows in
/// base-field symbols. Parts are appended in sending order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeTreePart {
    pub main_symbols: u64,
    pub mask_symbols: u64,
}

/// A trusted plan's layout of one runtime tree. A leaf's column holds every
/// part's main symbols in part order, then every part's mask symbols in part
/// order, then one salt. The tree is runtime-only: leaves start from `run`
/// under the runtime leaf domain and inner nodes from `configuration`; a
/// stored table keeps its own domain and configuration and is never laid out
/// here.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeTreePlan {
    pub tree_number: u64,
    pub leaf_class: u32,
    pub log_n: u32,
    pub log_n_zk: u32,
    pub parts: Vec<RuntimeTreePart>,
    pub salt_bytes: usize,
    pub run: Digest,
    pub configuration: Digest,
}

impl RuntimeTreePlan {
    pub fn main_width(&self) -> Result<u64> {
        let mut width = 0u64;
        for part in &self.parts {
            width = width
                .checked_add(part.main_symbols)
                .ok_or(Error::ResourceLimit)?;
        }
        Ok(width)
    }

    pub fn mask_width(&self) -> Result<u64> {
        let mut width = 0u64;
        for part in &self.parts {
            width = width
                .checked_add(part.mask_symbols)
                .ok_or(Error::ResourceLimit)?;
        }
        Ok(width)
    }

    pub fn width(&self) -> Result<u64> {
        self.main_width()?
            .checked_add(self.mask_width()?)
            .ok_or(Error::ResourceLimit)
    }

    pub fn has_main(&self) -> bool {
        self.parts.iter().any(|part| part.main_symbols > 0)
    }

    pub fn has_mask(&self) -> bool {
        self.parts.iter().any(|part| part.mask_symbols > 0)
    }

    pub fn main_offset(&self, part: usize) -> Result<u64> {
        if part >= self.parts.len() {
            return Err(Error::InvalidIndex);
        }
        let mut offset = 0u64;
        for entry in &self.parts[..part] {
            offset = offset
                .checked_add(entry.main_symbols)
                .ok_or(Error::ResourceLimit)?;
        }
        Ok(offset)
    }

    pub fn mask_offset(&self, part: usize) -> Result<u64> {
        if part >= self.parts.len() {
            return Err(Error::InvalidIndex);
        }
        let mut offset = 0u64;
        for entry in &self.parts[..part] {
            offset = offset
                .checked_add(entry.mask_symbols)
                .ok_or(Error::ResourceLimit)?;
        }
        Ok(offset)
    }

    pub fn fault(&self) -> Option<&'static str> {
        if self.tree_number >= (1u64 << 16) {
            return Some("the runtime tree number does not fit the tag");
        }
        if self.log_n < 1 || self.log_n > 60 {
            return Some("the runtime tree depth is not a tree");
        }
        if self.parts.is_empty() || (!self.has_main() && !self.has_mask()) {
            return Some("the runtime tree has no part or no symbol");
        }
        if self.has_mask() && (self.log_n_zk < 1 || self.log_n_zk > self.log_n) {
            return Some("the runtime tree mask rows do not fit under its main rows");
        }
        if !self.has_main() && self.log_n != self.log_n_zk {
            return Some("a mask-only runtime tree has two depths");
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

    pub fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(self.tree_number);
        writer.put_u64(self.leaf_class as u64);
        writer.put_u64(self.log_n as u64);
        writer.put_u64(self.log_n_zk as u64);
        writer.put_u64(self.parts.len() as u64);
        for part in &self.parts {
            writer.put_u64(part.main_symbols);
            writer.put_u64(part.mask_symbols);
        }
        writer.put_u64(self.salt_bytes as u64);
        writer.put_bytes(&self.run.0);
        writer.put_bytes(&self.configuration.0);
        writer.into_bytes()
    }
}

/// The verifier's layout of a prover message's runtime oracles.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeTreeLay {
    pub tree: Vec<u32>,
    pub part: Vec<u32>,
    pub oracles: Vec<RuntimeOracleShape>,
    pub plans: Vec<RuntimeTreePlan>,
}

impl RuntimeTreeLay {
    pub fn fault(&self) -> Option<&'static str> {
        if self.tree.len() != self.part.len() || self.tree.len() != self.oracles.len() {
            return Some("the runtime layout has one tree and part for each oracle only");
        }
        if self.plans.is_empty() != self.oracles.is_empty() {
            return Some("a runtime layout without oracles has no trees");
        }
        for (index, plan) in self.plans.iter().enumerate() {
            if plan.fault().is_some() {
                return Some("a runtime layout tree is not a tree");
            }
            if self.plans[..index]
                .iter()
                .any(|other| other.tree_number == plan.tree_number)
            {
                return Some("the runtime layout repeats a tree number");
            }
        }
        let mut mask_only_seen = false;
        for plan in &self.plans {
            if plan.has_main() {
                if mask_only_seen {
                    return Some("a main tree follows a mask-only tree");
                }
            } else {
                mask_only_seen = true;
            }
        }
        let mut used: Vec<Vec<bool>> = self
            .plans
            .iter()
            .map(|plan| vec![false; plan.parts.len()])
            .collect();
        for (index, oracle) in self.oracles.iter().enumerate() {
            if oracle_fault(oracle).is_some() {
                return Some("a runtime layout oracle is not an oracle shape");
            }
            let tree = self.tree[index] as usize;
            let part = self.part[index] as usize;
            let plan = match self.plans.get(tree) {
                Some(plan) => plan,
                None => return Some("the runtime layout names a tree it does not hold"),
            };
            if part >= plan.parts.len() {
                return Some("the runtime layout names a part the tree does not hold");
            }
            if used[tree][part] {
                return Some("the runtime layout names a part twice");
            }
            used[tree][part] = true;
            if plan.leaf_class != oracle.leaf_class {
                return Some("a runtime layout tree has another leaf class than its oracle");
            }
            let main_symbols = match oracle.main_rows.checked_mul(oracle.main_deg as u64) {
                Some(value) => value,
                None => return Some("a runtime layout oracle's main width overflows"),
            };
            let mask_symbols = match oracle.mask_rows.checked_mul(E_DEGREE as u64) {
                Some(value) => value,
                None => return Some("a runtime layout oracle's mask width overflows"),
            };
            if plan.parts[part].main_symbols != main_symbols
                || plan.parts[part].mask_symbols != mask_symbols
            {
                return Some("a runtime layout part has other widths than its oracle");
            }
            if oracle.main_rows > 0 && plan.log_n != oracle.base_log_n {
                return Some("a runtime layout main tree has another depth than its oracle");
            }
            if oracle.mask_rows > 0 && plan.log_n_zk != oracle.mask_log_n {
                return Some("a runtime layout mask tree has another depth than its oracle");
            }
        }
        for (tree, plan) in self.plans.iter().enumerate() {
            if used[tree].iter().any(|used| !used) || plan.parts.len() != used[tree].len() {
                return Some("a runtime layout leaves a part unused");
            }
        }
        None
    }

    pub fn check_against(&self, oracles: &[RuntimeOracleShape]) -> Option<&'static str> {
        if self.oracles != oracles {
            return Some("the runtime layout was not built from these trusted oracle shapes");
        }
        self.fault()
    }

    pub fn check_rebuilt(
        &self,
        oracles: &[RuntimeOracleShape],
        first_tree_number: u64,
        salt_bytes: usize,
        run: Digest,
        configuration: Digest,
    ) -> Option<&'static str> {
        let trusted =
            match lay_runtime_trees(oracles, first_tree_number, salt_bytes, run, configuration) {
                Ok(layout) => layout,
                Err(_) => return Some("the trusted oracle shapes do not lay out"),
            };
        if *self != trusted {
            return Some("the runtime layout differs from the independently built layout");
        }
        None
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(self.tree.len() as u64);
        for (tree, part) in self.tree.iter().zip(&self.part) {
            writer.put_u64(*tree as u64);
            writer.put_u64(*part as u64);
        }
        writer.put_u64(self.oracles.len() as u64);
        for oracle in &self.oracles {
            writer.put_u64(oracle.leaf_class as u64);
            writer.put_u64(oracle.base_log_n as u64);
            writer.put_u64(oracle.main_rows);
            writer.put_u64(oracle.main_deg as u64);
            writer.put_u64(oracle.mask_rows);
            writer.put_u64(oracle.mask_log_n as u64);
        }
        writer.put_u64(self.plans.len() as u64);
        for plan in &self.plans {
            let bytes = plan.encode();
            writer.put_u64(bytes.len() as u64);
            writer.put_bytes(&bytes);
        }
        writer.into_bytes()
    }
}

fn oracle_fault(oracle: &RuntimeOracleShape) -> Option<&'static str> {
    if oracle.main_rows == 0 && oracle.mask_rows == 0 {
        return Some("a runtime oracle has no row");
    }
    if oracle.main_rows > 0 {
        if oracle.main_deg != 1 && oracle.main_deg != E_DEGREE as u32 {
            return Some("a runtime oracle's main rows are over the wrong field");
        }
        if oracle.base_log_n < 1 || oracle.base_log_n > 60 {
            return Some("a runtime oracle's base depth is not a tree");
        }
    }
    if oracle.mask_rows > 0 && (oracle.mask_log_n < 1 || oracle.mask_log_n > 60) {
        return Some("a runtime oracle's mask depth is not a tree");
    }
    None
}

pub fn lay_runtime_trees(
    oracles: &[RuntimeOracleShape],
    first_tree_number: u64,
    salt_bytes: usize,
    run: Digest,
    configuration: Digest,
) -> Result<RuntimeTreeLay> {
    for oracle in oracles {
        if oracle_fault(oracle).is_some() {
            return Err(Error::InvalidConfig);
        }
    }
    if first_tree_number >= (1u64 << 16) {
        return Err(Error::InvalidConfig);
    }
    let mut tree = vec![0u32; oracles.len()];
    let mut part = vec![0u32; oracles.len()];
    let mut plans: Vec<RuntimeTreePlan> = Vec::new();
    let new_plan = |number: u64, leaf_class: u32, log_n: u32| RuntimeTreePlan {
        tree_number: number,
        leaf_class,
        log_n,
        log_n_zk: 0,
        parts: Vec::new(),
        salt_bytes,
        run,
        configuration,
    };
    for (index, oracle) in oracles.iter().enumerate() {
        if oracle.main_rows == 0 {
            continue;
        }
        let mut at = 0usize;
        while at < plans.len()
            && !(plans[at].leaf_class == oracle.leaf_class
                && plans[at].log_n == oracle.base_log_n)
        {
            at += 1;
        }
        if at == plans.len() {
            let number = first_tree_number
                .checked_add(plans.len() as u64)
                .ok_or(Error::ResourceLimit)?;
            if number >= (1u64 << 16) {
                return Err(Error::ResourceLimit);
            }
            plans.push(new_plan(number, oracle.leaf_class, oracle.base_log_n));
        }
        tree[index] = at as u32;
    }
    for (index, oracle) in oracles.iter().enumerate() {
        if oracle.main_rows != 0 {
            continue;
        }
        let mut at = 0usize;
        while at < plans.len() && plans[at].leaf_class != oracle.leaf_class {
            at += 1;
        }
        if at == plans.len() {
            let number = first_tree_number
                .checked_add(plans.len() as u64)
                .ok_or(Error::ResourceLimit)?;
            if number >= (1u64 << 16) {
                return Err(Error::ResourceLimit);
            }
            plans.push(new_plan(number, oracle.leaf_class, 0));
        }
        tree[index] = at as u32;
    }
    for (index, oracle) in oracles.iter().enumerate() {
        let at = tree[index] as usize;
        let plan = &mut plans[at];
        if oracle.mask_rows > 0 {
            if plan.has_mask() && plan.log_n_zk != oracle.mask_log_n {
                return Err(Error::InvalidConfig);
            }
            plan.log_n_zk = oracle.mask_log_n;
        }
        part[index] = u32::try_from(plan.parts.len()).map_err(|_| Error::ResourceLimit)?;
        let main_symbols = oracle
            .main_rows
            .checked_mul(oracle.main_deg as u64)
            .ok_or(Error::ResourceLimit)?;
        let mask_symbols = oracle
            .mask_rows
            .checked_mul(E_DEGREE as u64)
            .ok_or(Error::ResourceLimit)?;
        plan.parts.push(RuntimeTreePart {
            main_symbols,
            mask_symbols,
        });
    }
    for plan in &mut plans {
        if !plan.has_main() {
            plan.log_n = plan.log_n_zk;
        }
        if plan.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
    }
    let lay = RuntimeTreeLay {
        tree,
        part,
        oracles: oracles.to_vec(),
        plans,
    };
    if lay.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    Ok(lay)
}

/// One part's rows for committing a runtime tree. `main` rows are codewords
/// over the long columns (`2^log_n` symbols, each an extension-field element);
/// `mask` rows are codewords over the short columns (`2^log_n_zk` symbols).
/// A mask row is read at `x mod 2^log_n_zk` in the leaf of the long position x.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimePartRows {
    pub main: Vec<Vec<Fp3>>,
    pub mask: Vec<Vec<Fp3>>,
    pub main_deg: u32,
}

/// One opened long-column leaf: its position, its own salt and the complete
/// column (every part's main symbols, then every part's mask symbols, in part
/// order).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeLeafColumn {
    pub position: u64,
    pub salt: Vec<u8>,
    pub symbols: Vec<Fp>,
}

/// The opening of a whole runtime co-tree at a list of long positions.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RuntimeTreeOpening {
    pub positions: Vec<u64>,
    pub columns: Vec<RuntimeLeafColumn>,
    pub given: Vec<Digest>,
}

/// The position at which a short codeword is read for the leaf of long
/// position `x`: `x mod 2^log_n_zk`.
pub fn short_position(log_n: u32, log_short: u32, x: u64) -> Result<u64> {
    if log_n < 1 || log_short > log_n || log_n >= 64 || log_short >= 64 {
        return Err(Error::InvalidConfig);
    }
    if x >> log_n != 0 {
        return Err(Error::InvalidIndex);
    }
    Ok(x & ((1u64 << log_short) - 1))
}

/// Shrink a query list drawn over a longer column domain to a tree's depth:
/// each position is read at `x mod 2^tree_log_n`, the result sorted and
/// deduplicated. The original query positions are not lost: the caller keeps
/// them and maps each to its short position when extracting from the leaf.
pub fn reduce_positions(
    positions: &[u64],
    source_log_n: u32,
    tree_log_n: u32,
) -> Result<Vec<u64>> {
    if tree_log_n > source_log_n || source_log_n >= 64 || tree_log_n >= 64 {
        return Err(Error::InvalidConfig);
    }
    if positions.iter().any(|position| position >> source_log_n != 0) {
        return Err(Error::InvalidIndex);
    }
    let mask = (1u64 << tree_log_n) - 1;
    let mut reduced: Vec<u64> = positions.iter().map(|position| position & mask).collect();
    reduced.sort_unstable();
    reduced.dedup();
    Ok(reduced)
}

fn runtime_leaf(
    plan: &RuntimeTreePlan,
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

fn runtime_node(plan: &RuntimeTreePlan, left: &Digest, right: &Digest) -> Result<Digest> {
    merkle_node(
        tag_word(TREE_NODE, plan.tree_number),
        0,
        &plan.configuration,
        left,
        right,
    )
}

fn walk_given(
    plan: &RuntimeTreePlan,
    list: &[u64],
    leaves: &[Digest],
    levels: &[Vec<Digest>],
) -> Result<Vec<Digest>> {
    if list.len() != leaves.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut given = Vec::new();
    let mut now: Vec<(u64, Digest)> = list.iter().copied().zip(leaves.iter().copied()).collect();
    for step in 0..plan.log_n {
        let level = &levels[step as usize];
        let mut next = Vec::with_capacity((now.len() + 1) / 2);
        let mut index = 0usize;
        while index < now.len() {
            let place = now[index].0;
            if place & 1 == 0 && index + 1 < now.len() && now[index + 1].0 == place + 1 {
                next.push((place >> 1, runtime_node(plan, &now[index].1, &now[index + 1].1)?));
                index += 2;
            } else {
                let sibling = level[(place ^ 1) as usize];
                given.push(sibling);
                let parent = if place & 1 == 0 {
                    runtime_node(plan, &now[index].1, &sibling)?
                } else {
                    runtime_node(plan, &sibling, &now[index].1)?
                };
                next.push((place >> 1, parent));
                index += 1;
            }
        }
        now = next;
    }
    if now.len() != 1 {
        return Err(Error::InvalidProof);
    }
    Ok(given)
}

fn opening_given_count(plan: &RuntimeTreePlan, list: &[u64]) -> Result<usize> {
    let mut now: Vec<u64> = list.to_vec();
    let mut given = 0usize;
    for _ in 0..plan.log_n {
        let mut next = Vec::with_capacity((now.len() + 1) / 2);
        let mut index = 0usize;
        while index < now.len() {
            let place = now[index];
            if place & 1 == 0 && index + 1 < now.len() && now[index + 1] == place + 1 {
                next.push(place >> 1);
                index += 2;
            } else {
                given += 1;
                next.push(place >> 1);
                index += 1;
            }
        }
        now = next;
    }
    Ok(given)
}


/// A committed runtime co-tree with the private columns and salts needed to
/// open complete leaves. This is an in-memory test foundation; it materializes
/// every column and is not the production committer.
#[derive(Clone, Debug)]
pub struct RuntimeTree {
    plan: RuntimeTreePlan,
    root: Digest,
    levels: Vec<Vec<Digest>>,
    columns: Vec<Vec<Fp>>,
    salts: Vec<Vec<u8>>,
}

impl RuntimeTree {
    pub fn commit(
        plan: RuntimeTreePlan,
        parts: &[RuntimePartRows],
        salts: &[Vec<u8>],
    ) -> Result<Self> {
        let _profile_commit = crate::profile::Scope::new("runtime_tree.commit");
        if plan.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        if parts.len() != plan.parts.len() || salts.len() as u64 != plan.columns() {
            return Err(Error::ShapeMismatch);
        }
        let columns = plan.columns();
        let short_columns = 1u64 << plan.log_n_zk;
        for (part, rows) in plan.parts.iter().zip(parts) {
            if rows.main_deg != 1 && rows.main_deg != E_DEGREE as u32 {
                return Err(Error::InvalidConfig);
            }
            if part.main_symbols == 0 {
                if !rows.main.is_empty() {
                    return Err(Error::ShapeMismatch);
                }
            } else {
                if part.main_symbols % rows.main_deg as u64 != 0
                    || rows.main.len() as u64 != part.main_symbols / rows.main_deg as u64
                    || rows.main.iter().any(|row| row.len() as u64 != columns)
                {
                    return Err(Error::ShapeMismatch);
                }
                // A base-field row carries one coordinate per element. Two
                // different extension-field inputs must not collapse to the
                // same commitment, so a non-zero high coordinate is rejected
                // rather than dropped.
                if rows.main_deg == 1
                    && rows
                        .main
                        .iter()
                        .flatten()
                        .any(|element| element.c[1] != Fp::ZERO || element.c[2] != Fp::ZERO)
                {
                    return Err(Error::ShapeMismatch);
                }
            }
            if part.mask_symbols == 0 {
                if !rows.mask.is_empty() {
                    return Err(Error::ShapeMismatch);
                }
            } else {
                if part.mask_symbols % E_DEGREE as u64 != 0
                    || rows.mask.len() as u64 != part.mask_symbols / E_DEGREE as u64
                    || rows
                        .mask
                        .iter()
                        .any(|row| row.len() as u64 != short_columns)
                {
                    return Err(Error::ShapeMismatch);
                }
            }
        }
        for salt in salts {
            if salt.len() != plan.salt_bytes {
                return Err(Error::ShapeMismatch);
            }
        }
        let width = plan.width()? as usize;
        let mut columns_data = Vec::with_capacity(columns as usize);
        let mut leaves = Vec::with_capacity(columns as usize);
        for position in 0..columns {
            let short = short_position(plan.log_n, plan.log_n_zk, position)? as usize;
            let mut symbols = Vec::with_capacity(width);
            for rows in parts {
                for row in &rows.main {
                    for coefficient in 0..rows.main_deg as usize {
                        symbols.push(row[position as usize].c[coefficient]);
                    }
                }
            }
            for rows in parts {
                for row in &rows.mask {
                    for coefficient in 0..E_DEGREE {
                        symbols.push(row[short].c[coefficient]);
                    }
                }
            }
            leaves.push(runtime_leaf(&plan, position, &salts[position as usize], &symbols)?);
            columns_data.push(symbols);
        }
        let mut levels = vec![leaves];
        while levels.last().expect("one level").len() > 1 {
            let previous = levels.last().expect("one level");
            let mut next = Vec::with_capacity(previous.len() / 2);
            for pair in previous.chunks_exact(2) {
                next.push(runtime_node(&plan, &pair[0], &pair[1])?);
            }
            levels.push(next);
        }
        let root = levels.last().expect("one level")[0];
        Ok(Self {
            plan,
            root,
            levels,
            columns: columns_data,
            salts: salts.to_vec(),
        })
    }

    pub fn plan(&self) -> &RuntimeTreePlan {
        &self.plan
    }

    pub fn root(&self) -> &Digest {
        &self.root
    }

    pub fn open(&self, positions: &[u64]) -> Result<RuntimeTreeOpening> {
        if positions.is_empty() {
            return Err(Error::InvalidIndex);
        }
        let columns = self.plan.columns();
        let mut previous = None;
        let mut leaves = Vec::with_capacity(positions.len());
        let mut opened = Vec::with_capacity(positions.len());
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
            let symbols = self.columns[*position as usize].clone();
            leaves.push(runtime_leaf(
                &self.plan,
                *position,
                &self.salts[*position as usize],
                &symbols,
            )?);
            opened.push(RuntimeLeafColumn {
                position: *position,
                salt: self.salts[*position as usize].clone(),
                symbols,
            });
        }
        let given = walk_given(&self.plan, positions, &leaves, &self.levels)?;
        Ok(RuntimeTreeOpening {
            positions: positions.to_vec(),
            columns: opened,
            given,
        })
    }
}

/// Nodes already seen under one tree root. The key is
/// `(root, depth, index)`; a different value at the same vertex is a conflict.
/// This is an isolated candidate store, not a reference vertex record and not
/// a production session object.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OpeningNodeStore {
    nodes: BTreeMap<(Digest, u32, u64), Digest>,
}

impl OpeningNodeStore {
    pub fn get(&self, root: &Digest, depth: u32, index: u64) -> Option<&Digest> {
        self.nodes.get(&(*root, depth, index))
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    #[cfg(test)]
    pub fn insert_for_test(&mut self, root: Digest, depth: u32, index: u64, digest: Digest) {
        self.nodes.insert((root, depth, index), digest);
    }

    fn record(&mut self, root: &Digest, depth: u32, index: u64, digest: Digest) -> Result<()> {
        match self.nodes.get(&(*root, depth, index)) {
            Some(known) if *known == digest => Ok(()),
            Some(_) => Err(Error::InvalidProof),
            None => {
                self.nodes.insert((*root, depth, index), digest);
                Ok(())
            }
        }
    }
}

/// Check one opening and return a node store extended with every leaf, given
/// sibling and computed parent visited by the walk. The input store is not
/// changed on failure; callers commit the returned store only after success.
/// The root is part of every key, so values from another root or tree do not
/// collide.
pub fn verify_runtime_opening_shared(
    plan: &RuntimeTreePlan,
    root: &Digest,
    list: &[u64],
    opening: &RuntimeTreeOpening,
    store: &OpeningNodeStore,
) -> Result<OpeningNodeStore> {
    if plan.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    if opening.positions != list || opening.columns.len() != list.len() {
        return Err(Error::ShapeMismatch);
    }
    if list.is_empty() || list.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(Error::InvalidIndex);
    }
    let width = plan.width()? as usize;
    let columns = plan.columns();
    let mut staged = store.clone();
    let mut now = Vec::with_capacity(list.len());
    for (position, column) in list.iter().zip(&opening.columns) {
        if *position >= columns
            || column.position != *position
            || column.salt.len() != plan.salt_bytes
            || column.symbols.len() != width
        {
            return Err(Error::ShapeMismatch);
        }
        let leaf = runtime_leaf(plan, *position, &column.salt, &column.symbols)?;
        staged.record(root, 0, *position, leaf)?;
        now.push((*position, leaf));
    }
    let mut used = 0usize;
    for depth in 0..plan.log_n {
        let mut next = Vec::with_capacity((now.len() + 1) / 2);
        let mut index = 0usize;
        while index < now.len() {
            let place = now[index].0;
            if place & 1 == 0 && index + 1 < now.len() && now[index + 1].0 == place + 1 {
                let parent = runtime_node(plan, &now[index].1, &now[index + 1].1)?;
                staged.record(root, depth + 1, place >> 1, parent)?;
                next.push((place >> 1, parent));
                index += 2;
            } else {
                let sibling = *opening.given.get(used).ok_or(Error::ShapeMismatch)?;
                used += 1;
                staged.record(root, depth, place ^ 1, sibling)?;
                let parent = if place & 1 == 0 {
                    runtime_node(plan, &now[index].1, &sibling)?
                } else {
                    runtime_node(plan, &sibling, &now[index].1)?
                };
                staged.record(root, depth + 1, place >> 1, parent)?;
                next.push((place >> 1, parent));
                index += 1;
            }
        }
        now = next;
    }
    if used != opening.given.len() || now.len() != 1 || now[0].1 != *root {
        return Err(Error::InvalidProof);
    }
    staged.record(root, plan.log_n, 0, *root)?;
    Ok(staged)
}

/// Check a runtime opening without retaining its vertex values.
pub fn verify_runtime_opening(
    plan: &RuntimeTreePlan,
    root: &Digest,
    list: &[u64],
    opening: &RuntimeTreeOpening,
) -> Result<()> {
    verify_runtime_opening_shared(
        plan,
        root,
        list,
        opening,
        &OpeningNodeStore::default(),
    )
    .map(|_| ())
}

pub fn encode_runtime_opening(plan: &RuntimeTreePlan, opening: &RuntimeTreeOpening) -> Result<Vec<u8>> {
    if plan.fault().is_some() || opening.positions.len() != opening.columns.len() {
        return Err(Error::InvalidConfig);
    }
    let width = plan.width()? as usize;
    let mut writer = CanonicalWriter::new();
    for column in &opening.columns {
        if column.symbols.len() != width || column.salt.len() != plan.salt_bytes {
            return Err(Error::ShapeMismatch);
        }
        for symbol in &column.symbols {
            writer.put_fp(*symbol);
        }
        // The salt is a fixed-length field of the opening, not a
        // length-prefixed string.
        writer.put_raw_bytes(&column.salt);
    }
    for digest in &opening.given {
        writer.put_digest(digest);
    }
    Ok(writer.into_bytes())
}

pub fn decode_runtime_opening(
    plan: &RuntimeTreePlan,
    list: &[u64],
    bytes: &[u8],
) -> Result<RuntimeTreeOpening> {
    if plan.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    if list.is_empty()
        || list.windows(2).any(|pair| pair[0] >= pair[1])
        || list.iter().any(|x| *x >= plan.columns())
    {
        return Err(Error::InvalidIndex);
    }
    let width = plan.width()? as usize;
    let given_count = opening_given_count(plan, list)?;
    let mut reader = CanonicalReader::new(
        bytes,
        DecodeLimits {
            max_bytes: bytes.len(),
            max_elements: given_count.saturating_add(list.len()),
        },
    );
    let mut columns = Vec::with_capacity(list.len());
    for position in list {
        let mut symbols = Vec::with_capacity(width);
        for _ in 0..width {
            symbols.push(reader.get_fp()?);
        }
        let salt = reader.get_raw_bytes(plan.salt_bytes)?.to_vec();
        columns.push(RuntimeLeafColumn {
            position: *position,
            salt,
            symbols,
        });
    }
    let mut given = Vec::with_capacity(given_count);
    for _ in 0..given_count {
        given.push(reader.get_digest()?);
    }
    reader.finish()?;
    Ok(RuntimeTreeOpening {
        positions: list.to_vec(),
        columns,
        given,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::request::code::{build_code_plan, encode_row_fp3};
    use crate::pcs::request::field::{Fp, MODULUS};
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
        let symbols = verify_mask_opening(&tree.plan, &tree.root, &positions, &openings).unwrap();
        assert_eq!(symbols.len(), positions.len() * tree.plan.width().unwrap() as usize);
        let _ = shape;
    }

    #[test]
    fn tampered_mask_openings_are_rejected() {
        let tree = tree();
        let positions = [1u64, 4];
        let openings = tree.open(&positions).unwrap();
        let mut tampered = openings.clone();
        tampered.columns[0].symbols[0] = tampered.columns[0].symbols[0] + Fp::ONE;
        assert!(verify_mask_opening(&tree.plan, &tree.root, &positions, &tampered).is_err());
        let mut tampered = openings.clone();
        tampered.columns[0].salt[0] ^= 1;
        assert!(verify_mask_opening(&tree.plan, &tree.root, &positions, &tampered).is_err());
        let mut tampered = openings.clone();
        tampered.given[0] = Digest([3; 32]);
        assert!(verify_mask_opening(&tree.plan, &tree.root, &positions, &tampered).is_err());
        let mut tampered = openings;
        tampered.columns[0].position = 2;
        assert!(verify_mask_opening(&tree.plan, &tree.root, &positions, &tampered).is_err());
    }

    #[test]
    fn runtime_plan_values_are_bound() {
        let tree = tree();
        let positions = [1u64, 4];
        let openings = tree.open(&positions).unwrap();
        let mut other = tree.plan;
        other.tree_number += 1;
        assert!(verify_mask_opening(&other, &tree.root, &positions, &openings).is_err());
        let mut other = tree.plan;
        other.run = Digest([8; 32]);
        assert!(verify_mask_opening(&other, &tree.root, &positions, &openings).is_err());
        let mut other = tree.plan;
        other.configuration = Digest([6; 32]);
        assert!(verify_mask_opening(&other, &tree.root, &positions, &openings).is_err());
    }

    fn co_plan() -> RuntimeTreePlan {
        RuntimeTreePlan {
            tree_number: 7,
            leaf_class: 3,
            log_n: 2,
            log_n_zk: 1,
            parts: vec![part(6, 3), part(0, 6)],
            salt_bytes: LEAF_SALT_MIN_BYTES,
            run: Digest([9; 32]),
            configuration: Digest([7; 32]),
        }
    }

    fn co_parts() -> Vec<RuntimePartRows> {
        let main = (0..2)
            .map(|row| {
                (0..4)
                    .map(|column| Fp3::from_u64(100 + row * 10 + column + 1))
                    .collect()
            })
            .collect();
        let mask0 = (0..1)
            .map(|row| {
                (0..2)
                    .map(|column| Fp3::from_u64(200 + row * 10 + column + 1))
                    .collect()
            })
            .collect();
        let mask1 = (0..2)
            .map(|row| {
                (0..2)
                    .map(|column| Fp3::from_u64(300 + row * 10 + column + 1))
                    .collect()
            })
            .collect();
        vec![
            RuntimePartRows {
                main,
                mask: mask0,
                main_deg: 3,
            },
            RuntimePartRows {
                main: Vec::new(),
                mask: mask1,
                main_deg: 1,
            },
        ]
    }

    fn co_salts() -> Vec<Vec<u8>> {
        (0..4)
            .map(|position| vec![(position as u8) + 1; LEAF_SALT_MIN_BYTES])
            .collect()
    }

    fn oracle(
        leaf_class: u32,
        base_log_n: u32,
        main_rows: u64,
        main_deg: u32,
        mask_rows: u64,
        mask_log_n: u32,
    ) -> RuntimeOracleShape {
        RuntimeOracleShape {
            leaf_class,
            base_log_n,
            main_rows,
            main_deg,
            mask_rows,
            mask_log_n,
        }
    }

    fn part(main_symbols: u64, mask_symbols: u64) -> RuntimeTreePart {
        RuntimeTreePart {
            main_symbols,
            mask_symbols,
        }
    }

    fn layout(oracles: &[RuntimeOracleShape], first: u64) -> RuntimeTreeLay {
        lay_runtime_trees(
            oracles,
            first,
            LEAF_SALT_MIN_BYTES,
            Digest([9; 32]),
            Digest([7; 32]),
        )
        .unwrap()
    }

    #[test]
    fn runtime_layout_groups_same_class_and_depth() {
        let oracles = [
            oracle(1, 4, 4, 1, 2, 3),
            oracle(1, 4, 8, 3, 0, 0),
            oracle(1, 0, 0, 1, 4, 3),
            oracle(2, 0, 0, 1, 8, 5),
        ];
        let lay = layout(&oracles, 5);
        assert_eq!(lay.tree, vec![0, 0, 0, 1]);
        assert_eq!(lay.part, vec![0, 1, 2, 0]);
        assert_eq!(lay.plans.len(), 2);
        assert_eq!(lay.plans[0].tree_number, 5);
        assert_eq!(lay.plans[0].log_n, 4);
        assert_eq!(lay.plans[0].log_n_zk, 3);
        assert_eq!(
            lay.plans[0].parts,
            vec![part(4, 6), part(24, 0), part(0, 12)]
        );
        assert_eq!(lay.plans[0].main_width().unwrap(), 28);
        assert_eq!(lay.plans[0].mask_width().unwrap(), 18);
        assert_eq!(lay.plans[0].width().unwrap(), 46);
        assert_eq!(lay.plans[0].main_offset(1).unwrap(), 4);
        assert_eq!(lay.plans[0].mask_offset(2).unwrap(), 6);
        assert_eq!(lay.plans[1].tree_number, 6);
        assert_eq!(lay.plans[1].log_n, 5);
        assert_eq!(lay.plans[1].log_n_zk, 5);
        assert_eq!(lay.plans[1].parts, vec![part(0, 24)]);
        assert!(lay.fault().is_none());
    }

    #[test]
    fn runtime_layout_separates_depth_and_class() {
        let oracles = [
            oracle(1, 4, 4, 1, 0, 0),
            oracle(1, 5, 4, 1, 0, 0),
            oracle(2, 4, 4, 1, 0, 0),
        ];
        let lay = layout(&oracles, 0);
        assert_eq!(lay.tree, vec![0, 1, 2]);
        assert_eq!(lay.part, vec![0, 0, 0]);
        assert_eq!(lay.plans.len(), 3);
        assert_eq!(lay.plans[0].log_n, 4);
        assert_eq!(lay.plans[1].log_n, 5);
        assert_eq!(lay.plans[2].log_n, 4);
    }

    #[test]
    fn runtime_layout_faults_are_rejected() {
        assert!(lay_runtime_trees(
            &[oracle(1, 4, 0, 1, 0, 0)],
            0,
            LEAF_SALT_MIN_BYTES,
            Digest([9; 32]),
            Digest([7; 32])
        )
        .is_err());
        assert!(lay_runtime_trees(
            &[oracle(1, 4, 4, 2, 0, 0)],
            0,
            LEAF_SALT_MIN_BYTES,
            Digest([9; 32]),
            Digest([7; 32])
        )
        .is_err());
        assert!(lay_runtime_trees(
            &[oracle(1, 0, 4, 1, 0, 0)],
            0,
            LEAF_SALT_MIN_BYTES,
            Digest([9; 32]),
            Digest([7; 32])
        )
        .is_err());
        // Mask rows of two lengths cannot share a tree.
        assert!(lay_runtime_trees(
            &[oracle(1, 4, 4, 1, 2, 3), oracle(1, 4, 4, 1, 2, 4)],
            0,
            LEAF_SALT_MIN_BYTES,
            Digest([9; 32]),
            Digest([7; 32])
        )
        .is_err());
        // Mask rows cannot be deeper than the main rows.
        assert!(lay_runtime_trees(
            &[oracle(1, 3, 4, 1, 2, 4)],
            0,
            LEAF_SALT_MIN_BYTES,
            Digest([9; 32]),
            Digest([7; 32])
        )
        .is_err());
        // The next tree number must fit the tag.
        assert!(lay_runtime_trees(
            &[oracle(1, 4, 4, 1, 0, 0), oracle(2, 4, 4, 1, 0, 0)],
            (1u64 << 16) - 1,
            LEAF_SALT_MIN_BYTES,
            Digest([9; 32]),
            Digest([7; 32])
        )
        .is_err());
    }

    #[test]
    fn runtime_layout_tampers_are_rejected() {
        let oracles = [
            oracle(1, 4, 4, 1, 2, 3),
            oracle(1, 0, 0, 1, 4, 3),
            oracle(2, 4, 4, 1, 0, 0),
        ];
        let lay = layout(&oracles, 0);
        assert!(lay.fault().is_none());
        let mut bad = lay.clone();
        bad.tree[1] = 0;
        bad.part[1] = 0;
        assert!(bad.fault().is_some());
        let mut bad = lay.clone();
        bad.plans[1].tree_number = bad.plans[0].tree_number;
        assert!(bad.fault().is_some());
        let mut bad = lay.clone();
        bad.oracles[0].leaf_class = 9;
        assert!(bad.fault().is_some());
        let mut bad = lay.clone();
        bad.oracles[0].main_rows = 5;
        assert!(bad.fault().is_some());
        let mut bad = lay.clone();
        bad.oracles[1].mask_log_n = 4;
        assert!(bad.fault().is_some());
        assert!(lay.check_against(&oracles).is_none());
        assert!(lay.check_against(&[oracles[0]]).is_some());
    }

    #[test]
    fn runtime_layout_overflowing_widths_are_rejected() {
        let base = oracle(1, 4, 4, 3, 0, 0);
        let mut lay = layout(&[base], 0);
        assert!(lay.check_against(&[base]).is_none());
        let mut huge = lay.oracles.clone();
        huge[0].main_rows = u64::MAX / 3 + 1;
        lay.oracles = huge.clone();
        assert_eq!(
            lay.check_against(&huge),
            Some("a runtime layout oracle's main width overflows")
        );
        let base = oracle(1, 4, 4, 1, 2, 3);
        let mut lay = layout(&[base], 0);
        let mut huge = lay.oracles.clone();
        huge[0].mask_rows = u64::MAX / 3 + 1;
        lay.oracles = huge.clone();
        assert_eq!(
            lay.check_against(&huge),
            Some("a runtime layout oracle's mask width overflows")
        );
        // The legal shape remains usable.
        let legal = layout(&[base], 0);
        assert!(legal.check_against(&[base]).is_none());
    }

    #[test]
    fn runtime_layout_rebuild_gate_rejects_reorder_split_and_context() {
        let a = oracle(1, 4, 4, 1, 0, 0);
        let b = oracle(1, 4, 8, 1, 0, 0);
        let oracles = [a, b];
        let external = layout(&oracles, 0);
        let salt = LEAF_SALT_MIN_BYTES;
        let run = Digest([9; 32]);
        let configuration = Digest([7; 32]);
        assert!(external
            .check_rebuilt(&oracles, 0, salt, run, configuration)
            .is_none());
        // Reordered oracle shapes.
        assert!(external
            .check_rebuilt(&[b, a], 0, salt, run, configuration)
            .is_some());
        // A split layout that is locally valid but differs from the rebuilt one.
        let one_a = layout(&[a], 0);
        let one_b = layout(&[b], 1);
        let split = RuntimeTreeLay {
            tree: vec![0, 1],
            part: vec![0, 0],
            oracles: vec![a, b],
            plans: vec![one_a.plans[0].clone(), one_b.plans[0].clone()],
        };
        assert!(split.fault().is_none());
        assert!(split
            .check_rebuilt(&oracles, 0, salt, run, configuration)
            .is_some());
        // Another tree context: first number, run or configuration.
        assert!(external
            .check_rebuilt(&oracles, 1, salt, run, configuration)
            .is_some());
        assert!(external
            .check_rebuilt(&oracles, 0, salt, Digest([8; 32]), configuration)
            .is_some());
        assert!(external
            .check_rebuilt(&oracles, 0, salt, run, Digest([6; 32]))
            .is_some());
    }

    #[test]
    fn runtime_layout_encoding_binds_classes_depths_and_parts() {
        let oracles = [oracle(1, 4, 4, 1, 2, 3), oracle(2, 0, 0, 1, 4, 3)];
        let first = layout(&oracles, 0);
        let again = layout(&oracles, 0);
        assert_eq!(first.encode(), again.encode());
        let mut changed = oracles;
        changed[0].leaf_class = 3;
        assert_ne!(first.encode(), layout(&changed, 0).encode());
        let mut changed = oracles;
        changed[0].main_rows = 5;
        assert_ne!(first.encode(), layout(&changed, 0).encode());
        let mut changed = oracles;
        changed[1].mask_log_n = 4;
        assert_ne!(first.encode(), layout(&changed, 0).encode());
    }

    #[test]
    fn runtime_tree_plan_shape_faults_are_rejected() {
        let lay = layout(&[oracle(1, 4, 4, 1, 2, 3)], 0);
        let base = &lay.plans[0];
        assert!(base.fault().is_none());
        let mut bad = base.clone();
        bad.log_n_zk = 5;
        assert!(bad.fault().is_some());
        let mut bad = base.clone();
        bad.parts[0].main_symbols = 1u64 << 24;
        assert!(bad.fault().is_some());
        let mut bad = base.clone();
        bad.salt_bytes = LEAF_SALT_MIN_BYTES - 1;
        assert!(bad.fault().is_some());
        let mut bad = base.clone();
        bad.run = Digest::ZERO;
        assert!(bad.fault().is_some());
        let mut mask_only = layout(&[oracle(1, 0, 0, 1, 2, 3)], 0)
            .plans[0]
            .clone();
        assert!(mask_only.fault().is_none());
        mask_only.log_n = 4;
        assert!(mask_only.fault().is_some());
        assert!(base.main_offset(1).is_err());
        assert!(base.mask_offset(1).is_err());
    }

    #[test]
    fn runtime_co_tree_round_trip_and_opening_units() {
        let plan = co_plan();
        let tree = RuntimeTree::commit(plan.clone(), &co_parts(), &co_salts()).unwrap();
        let positions = [0u64, 1, 2, 3];
        let opening = tree.open(&positions).unwrap();
        verify_runtime_opening(&plan, tree.root(), &positions, &opening).unwrap();
        assert_eq!(opening.columns.len(), 4);
        assert_eq!(opening.columns[0].symbols.len(), 15);
        // Mask sections: x=0 and x=2 share short position 0; x=1 and x=3 share 1.
        assert_eq!(&opening.columns[0].symbols[6..9], &opening.columns[2].symbols[6..9]);
        assert_eq!(&opening.columns[1].symbols[6..9], &opening.columns[3].symbols[6..9]);
        assert_ne!(&opening.columns[0].symbols[..6], &opening.columns[2].symbols[..6]);
        assert_ne!(opening.columns[0].salt, opening.columns[2].salt);
        assert_ne!(
            runtime_leaf(&plan, 0, &opening.columns[0].salt, &opening.columns[0].symbols).unwrap(),
            runtime_leaf(&plan, 2, &opening.columns[2].salt, &opening.columns[2].symbols).unwrap()
        );
        let mut bad = opening.clone();
        bad.columns[0].symbols[0] = bad.columns[0].symbols[0] + Fp::ONE;
        assert!(verify_runtime_opening(&plan, tree.root(), &positions, &bad).is_err());
        // A symbol in another part's mask section is bound too.
        let mut bad = opening.clone();
        bad.columns[0].symbols[10] = bad.columns[0].symbols[10] + Fp::ONE;
        assert!(verify_runtime_opening(&plan, tree.root(), &positions, &bad).is_err());
        // A leaf with a missing part symbol is not the full leaf.
        let mut bad = opening.clone();
        bad.columns[0].symbols.pop();
        assert!(verify_runtime_opening(&plan, tree.root(), &positions, &bad).is_err());
        let mut bad = opening.clone();
        bad.columns[0].salt[0] ^= 1;
        assert!(verify_runtime_opening(&plan, tree.root(), &positions, &bad).is_err());
        // A sparse list needs siblings; its given data is exact.
        let sparse_positions = [0u64, 2];
        let sparse = tree.open(&sparse_positions).unwrap();
        verify_runtime_opening(&plan, tree.root(), &sparse_positions, &sparse).unwrap();
        let mut bad = sparse.clone();
        bad.given.pop();
        assert!(verify_runtime_opening(&plan, tree.root(), &sparse_positions, &bad).is_err());
        let mut bad = sparse.clone();
        bad.given.push(Digest([3; 32]));
        assert!(verify_runtime_opening(&plan, tree.root(), &sparse_positions, &bad).is_err());
        let mut bad = opening.clone();
        bad.positions = vec![0, 1, 3, 2];
        assert!(verify_runtime_opening(&plan, tree.root(), &positions, &bad).is_err());
        assert!(verify_runtime_opening(&plan, &Digest([4; 32]), &positions, &opening).is_err());
        let mut other = plan.clone();
        other.run = Digest([8; 32]);
        assert!(verify_runtime_opening(&other, tree.root(), &positions, &opening).is_err());
        let mut other = plan.clone();
        other.configuration = Digest([6; 32]);
        assert!(verify_runtime_opening(&other, tree.root(), &positions, &opening).is_err());
        assert!(tree.open(&[1, 0]).is_err());
        assert!(tree.open(&[0, 0]).is_err());
        assert!(tree.open(&[0, 4]).is_err());
        assert!(tree.open(&[]).is_err());

        let bytes = encode_runtime_opening(&plan, &opening).unwrap();
        assert_eq!(
            bytes.len(),
            positions.len() * (15 * 8 + plan.salt_bytes) + opening.given.len() * 32
        );
        let decoded = decode_runtime_opening(&plan, &positions, &bytes).unwrap();
        assert_eq!(decoded, opening);
        let mut noncanonical = bytes.clone();
        noncanonical[..8].copy_from_slice(&MODULUS.to_le_bytes());
        assert!(decode_runtime_opening(&plan, &positions, &noncanonical).is_err());
        assert!(decode_runtime_opening(&plan, &positions, &bytes[..bytes.len() - 1]).is_err());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(decode_runtime_opening(&plan, &positions, &trailing).is_err());
    }

    #[test]
    fn mask_only_tree_is_the_one_part_special_case() {
        let shape = shape();
        let plan = plan();
        let rows = rows(&shape);
        let tails = tails(&shape);
        let salts = salts(&shape, &plan);
        let mask_tree = MaskTree::commit(plan, &shape, &rows, &tails, &salts).unwrap();
        let codewords: Vec<Vec<Fp3>> = rows
            .iter()
            .zip(&tails)
            .map(|(row, tail)| encode_row_fp3(&shape.mask_code, row, tail).unwrap())
            .collect();
        let generic = RuntimeTree::commit(
            plan.runtime_plan().unwrap(),
            &[RuntimePartRows {
                main: Vec::new(),
                mask: codewords,
                main_deg: 1,
            }],
            &salts,
        )
        .unwrap();
        assert_eq!(mask_tree.root, *generic.root());
        assert_eq!(
            short_position(plan.log_n, plan.log_n, 5).unwrap(),
            5 & ((1u64 << plan.log_n) - 1)
        );
    }

    #[test]
    fn reduced_query_positions_keep_their_long_identity() {
        let plan = co_plan();
        let tree = RuntimeTree::commit(plan.clone(), &co_parts(), &co_salts()).unwrap();
        let queries = [1u64, 9, 3, 11];
        let reduced = reduce_positions(&queries, 4, 2).unwrap();
        assert_eq!(reduced, vec![1, 3]);
        let opening = tree.open(&reduced).unwrap();
        verify_runtime_opening(&plan, tree.root(), &reduced, &opening).unwrap();
        for query in queries {
            let tree_position = short_position(4, plan.log_n, query).unwrap();
            let mask_position = short_position(plan.log_n, plan.log_n_zk, tree_position).unwrap();
            assert_eq!(mask_position, short_position(4, plan.log_n_zk, query).unwrap());
            assert!(reduced.contains(&tree_position));
        }
        assert_eq!(short_position(4, 2, 1).unwrap(), 1);
        assert_eq!(short_position(4, 2, 9).unwrap(), 1);
        assert_eq!(short_position(2, 1, 0).unwrap(), 0);
        assert_eq!(short_position(2, 1, 2).unwrap(), 0);
    }

    #[test]
    fn short_position_checks_the_source_domain() {
        assert_eq!(short_position(2, 1, 3).unwrap(), 1);
        assert!(short_position(2, 1, 4).is_err());
        assert!(short_position(2, 1, u64::MAX).is_err());
        assert_eq!(short_position(60, 1, (1u64 << 60) - 1).unwrap(), 1);
        assert!(short_position(60, 1, 1u64 << 60).is_err());
        assert_eq!(short_position(60, 60, 12_345).unwrap(), 12_345);
        assert!(short_position(60, 60, 1u64 << 60).is_err());
        assert_eq!(short_position(2, 0, 3).unwrap(), 0);
        assert!(short_position(2, 0, 4).is_err());
        assert!(short_position(0, 0, 0).is_err());
        assert!(short_position(1, 2, 0).is_err());
        assert!(short_position(64, 1, 0).is_err());
        assert!(short_position(1, 64, 0).is_err());
    }

    #[test]
    fn opening_bytes_are_an_independent_fixed_format() {
        let plan = RuntimeTreePlan {
            tree_number: 5,
            leaf_class: 2,
            log_n: 1,
            log_n_zk: 1,
            parts: vec![part(0, 3)],
            salt_bytes: LEAF_SALT_MIN_BYTES,
            run: Digest([9; 32]),
            configuration: Digest([7; 32]),
        };
        let row = vec![
            Fp3::new(Fp::from_u64(1), Fp::from_u64(2), Fp::from_u64(3)),
            Fp3::new(Fp::from_u64(4), Fp::from_u64(5), Fp::from_u64(6)),
        ];
        let salts = vec![
            vec![0xA0; LEAF_SALT_MIN_BYTES],
            vec![0xB0; LEAF_SALT_MIN_BYTES],
        ];
        let tree = RuntimeTree::commit(
            plan.clone(),
            &[RuntimePartRows {
                main: Vec::new(),
                mask: vec![row],
                main_deg: 1,
            }],
            &salts,
        )
        .unwrap();
        let opening = tree.open(&[0]).unwrap();
        let mut expected = Vec::new();
        for value in [1u64, 2, 3] {
            expected.extend_from_slice(&value.to_le_bytes());
        }
        expected.extend_from_slice(&salts[0]);
        for digest in &opening.given {
            expected.extend_from_slice(&digest.0);
        }
        assert_eq!(
            expected.len(),
            3 * 8 + plan.salt_bytes + opening.given.len() * 32
        );
        assert_eq!(encode_runtime_opening(&plan, &opening).unwrap(), expected);
        assert_eq!(
            decode_runtime_opening(&plan, &[0], &expected).unwrap(),
            opening
        );
        assert!(decode_runtime_opening(&plan, &[0], &expected[..expected.len() - 1]).is_err());
        let mut trailing = expected;
        trailing.push(0);
        assert!(decode_runtime_opening(&plan, &[0], &trailing).is_err());
    }

    #[test]
    fn runtime_leaf_hash_input_is_the_current_candidate_format() {
        use crate::primitives::transcript::{hash_labeled, tag_word, LEAF_MAIN};
        let plan = RuntimeTreePlan {
            tree_number: 5,
            leaf_class: 2,
            log_n: 1,
            log_n_zk: 1,
            parts: vec![part(0, 3)],
            salt_bytes: LEAF_SALT_MIN_BYTES,
            run: Digest([9; 32]),
            configuration: Digest([7; 32]),
        };
        let salt = vec![0xAB; LEAF_SALT_MIN_BYTES];
        let symbols = vec![
            Fp::from_u64(11),
            Fp::from_u64(12),
            Fp::from_u64(13),
        ];
        let mut input = Vec::new();
        input.extend_from_slice(&3u64.to_le_bytes());
        input.extend_from_slice(&(salt.len() as u64).to_le_bytes());
        for value in [11u64, 12, 13] {
            input.extend_from_slice(&value.to_le_bytes());
        }
        input.extend_from_slice(&(salt.len() as u64).to_le_bytes());
        input.extend_from_slice(&salt);
        let expected = hash_labeled(
            tag_word(LEAF_MAIN, plan.tree_number),
            (1u64 << plan.log_n) + 1,
            Some(&plan.run),
            &input,
        )
        .unwrap();
        assert_eq!(runtime_leaf(&plan, 1, &salt, &symbols).unwrap(), expected);
    }

    #[test]
    fn base_rows_reject_nonzero_high_components() {
        let plan = RuntimeTreePlan {
            tree_number: 11,
            leaf_class: 1,
            log_n: 1,
            log_n_zk: 0,
            parts: vec![part(1, 0)],
            salt_bytes: LEAF_SALT_MIN_BYTES,
            run: Digest([9; 32]),
            configuration: Digest([7; 32]),
        };
        let salts = vec![vec![1; LEAF_SALT_MIN_BYTES]; 2];
        let base = |value: u64| vec![vec![Fp3::from_base(Fp::from_u64(value)); 2]];
        let tree = RuntimeTree::commit(
            plan.clone(),
            &[RuntimePartRows {
                main: base(7),
                mask: Vec::new(),
                main_deg: 1,
            }],
            &salts,
        )
        .unwrap();
        let opening = tree.open(&[0]).unwrap();
        verify_runtime_opening(&plan, tree.root(), &[0], &opening).unwrap();
        assert_eq!(opening.columns[0].symbols, vec![Fp::from_u64(7)]);
        for high in 1..=2usize {
            let mut element = Fp3::from_base(Fp::ONE);
            element.c[high] = Fp::ONE;
            assert!(RuntimeTree::commit(
                plan.clone(),
                &[RuntimePartRows {
                    main: vec![vec![element; 2]],
                    mask: Vec::new(),
                    main_deg: 1,
                }],
                &salts,
            )
            .is_err());
        }
        let extended_plan = RuntimeTreePlan {
            parts: vec![part(3, 0)],
            ..plan
        };
        let extended = Fp3::new(Fp::from_u64(2), Fp::from_u64(3), Fp::from_u64(4));
        let tree = RuntimeTree::commit(
            extended_plan.clone(),
            &[RuntimePartRows {
                main: vec![vec![extended; 2]],
                mask: Vec::new(),
                main_deg: 3,
            }],
            &salts,
        )
        .unwrap();
        let opening = tree.open(&[0]).unwrap();
        assert_eq!(
            opening.columns[0].symbols,
            vec![Fp::from_u64(2), Fp::from_u64(3), Fp::from_u64(4)]
        );
        verify_runtime_opening(&extended_plan, tree.root(), &[0], &opening).unwrap();
    }

    #[test]
    fn illegal_co_tree_shapes_fail_closed() {
        let mut bad = co_plan();
        bad.log_n = 64;
        assert!(bad.fault().is_some());
        assert!(RuntimeTree::commit(bad, &co_parts(), &co_salts()).is_err());
        let mut bad = co_plan();
        bad.parts[0].mask_symbols = u64::MAX;
        assert!(bad.fault().is_some());
        assert!(RuntimeTree::commit(bad, &co_parts(), &co_salts()).is_err());
        let mut bad = co_plan();
        bad.parts[0].main_symbols = 0;
        bad.parts[0].mask_symbols = 0;
        assert!(bad.fault().is_some());
        let mut parts = co_parts();
        parts[0].main.pop();
        assert!(RuntimeTree::commit(co_plan(), &parts, &co_salts()).is_err());
        let mut salts = co_salts();
        salts[0].pop();
        assert!(RuntimeTree::commit(co_plan(), &co_parts(), &salts).is_err());
        let bytes = vec![0u8; 8];
        assert!(decode_runtime_opening(&co_plan(), &[0], &bytes).is_err());
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
