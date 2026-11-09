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

/// One oracle message's planned shape for a runtime tree layout. The plan
/// supplies the numbers; nothing here comes from the proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeOracleShape {
    pub leaf_class: u32,
    /// The base code's depth; used only when the message has main rows.
    pub base_log_n: u32,
    pub main_rows: u64,
    /// The main rows are over the base field (1) or its cubic extension (3).
    pub main_deg: u32,
    pub mask_rows: u64,
    /// The mask code's depth; used only when the message has mask rows.
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
    /// The mask rows' depth, at most the main depth; a mask-only tree has
    /// `log_n == log_n_zk`.
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

    /// The offset of a part's main symbols from the start of the main section.
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

    /// The offset of a part's mask symbols from the start of the mask section
    /// (the mask section starts after every main symbol).
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

    /// Canonical bytes for binding this tree into a plan or first query.
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

/// The verifier's layout of a prover message's runtime oracles: each oracle's
/// tree and part, and the trees in creation order. Trees with main rows are
/// numbered first; mask-only trees follow. The proof never supplies these
/// numbers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeTreeLay {
    pub tree: Vec<u32>,
    pub part: Vec<u32>,
    pub oracles: Vec<RuntimeOracleShape>,
    pub plans: Vec<RuntimeTreePlan>,
}

impl RuntimeTreeLay {
    /// The layout is complete and consistent with the trusted oracle shapes
    /// it was built from: every (tree, part) appears exactly once, every part
    /// is used, tree numbers are unique, main trees come before mask-only
    /// trees, and each part's class, depth and widths equal its oracle's.
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
            let main_symbols = oracle
                .main_rows
                .checked_mul(oracle.main_deg as u64)
                .ok_or(Error::ResourceLimit)
                .ok()?;
            let mask_symbols = oracle
                .mask_rows
                .checked_mul(E_DEGREE as u64)
                .ok_or(Error::ResourceLimit)
                .ok()?;
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
            if !plan.has_main() && self.oracles.iter().enumerate().any(|(index, oracle)| {
                self.tree[index] as usize == tree && oracle.main_rows > 0
            }) {
                return Some("a runtime mask-only tree holds a main oracle");
            }
        }
        None
    }

    /// A layout received from elsewhere must equal the verifier's own trusted
    /// oracle shapes before it is used; the verifier rebuilds or checks it.
    pub fn check_against(&self, oracles: &[RuntimeOracleShape]) -> Option<&'static str> {
        if self.oracles != oracles {
            return Some("the runtime layout was not built from these trusted oracle shapes");
        }
        self.fault()
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

/// Lay out one prover message's runtime oracles exactly by class and depth:
/// main-row oracles of one leaf class and one base depth share a tree; a
/// mask-only oracle joins the first tree of its class, or forms a mask-only
/// tree of its own; parts follow the sending order. All mask rows in one tree
/// must have one depth, and mask rows cannot be deeper than the main rows.
/// Runtime-only: the caller must not pass a stored table's shape.
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
    // Main-row oracles first: one tree per (leaf class, base depth), in the
    // order the first such oracle is sent.
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
    // Mask-only oracles join the first tree of their class, or make a
    // mask-only tree (depth set after the parts are known).
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
    // Parts in sending order, one entry per oracle in its tree.
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
