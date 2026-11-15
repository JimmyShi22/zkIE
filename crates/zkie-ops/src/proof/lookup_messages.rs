//! Committed messages and roots for the byte lookup.
//!
//! The lookup plan fixes the multiplicity-bit-message shape and the mask-only
//! message shape. This module commits one multiplicity-bit message and one mask
//! message from private encoding material, returns their roots before any
//! lookup challenge is drawn, and exposes the mask entries by slot. The key
//! vector is not committed here: it is read from already committed rounding
//! rows through the plan's key map. Opening sampling and the ledger entries are
//! the next step; this module is the commitment side only.

use zkie_core::common::field::PrimeField64;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::code::{encode_row_fp3, CodeParams};
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::fold::{ByteMaskCoins, MaskCoins};
use zkie_engine::pcs::request::runtime_tree::{RuntimePartRows, RuntimeTree, RuntimeTreePlan};

use crate::proof::hidden_send_session::WShape;
use crate::proof::output_closure::{commit_value_group, CommittedValueGroup};

/// The shapes, identifiers and tree plans of the lookup messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupCommitPlan {
    pub bits_shape: WShape,
    pub bits_oracle: u64,
    pub group: u64,
    /// The number of opened positions of the multiplicity-bit tree.
    pub bits_openings: usize,
    pub mask_oracle: u64,
    pub mask_code: CodeParams,
    pub mask_rows: u64,
    /// The number of opened positions of the mask tree.
    pub mask_openings: usize,
    pub mask_tree: RuntimeTreePlan,
}

impl LookupCommitPlan {
    pub fn validate(&self, bit_count: usize, mask_count: usize) -> Result<()> {
        if self.bits_oracle == 0
            || self.mask_oracle == 0
            || self.group == 0
            || self.bits_oracle == self.mask_oracle
        {
            return Err(Error::InvalidConfig);
        }
        if self.bits_openings == 0
            || self.bits_openings as u64 > self.bits_shape.code.r
            || self.mask_openings == 0
            || self.mask_openings as u64 > self.mask_code.r
        {
            return Err(Error::InvalidConfig);
        }
        if self.bits_shape.fault(bit_count as u64, 1).is_some() {
            return Err(Error::InvalidConfig);
        }
        if self.bits_shape.answers != 1 {
            return Err(Error::InvalidConfig);
        }
        if zkie_engine::pcs::request::code::build_code_plan(
            self.mask_code.log_n,
            self.mask_code.ell,
            self.mask_code.r,
        )
        .is_err()
        {
            return Err(Error::InvalidConfig);
        }
        if self.mask_rows == 0 || !self.mask_rows.is_power_of_two() {
            return Err(Error::InvalidConfig);
        }
        if self.mask_tree.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        if self.mask_tree.log_n != self.mask_code.log_n
            || self.mask_tree.log_n_zk != self.mask_code.log_n
            || self.mask_tree.parts.len() != 1
        {
            return Err(Error::InvalidConfig);
        }
        let symbols = self
            .mask_rows
            .checked_mul(zkie_engine::pcs::request::field::E_DEGREE as u64)
            .ok_or(Error::ResourceLimit)?;
        if self.mask_tree.parts[0].main_symbols != 0
            || self.mask_tree.parts[0].mask_symbols != symbols
        {
            return Err(Error::InvalidConfig);
        }
        if self.bits_shape.tree_number == self.mask_tree.tree_number {
            return Err(Error::InvalidConfig);
        }
        let ell = usize::try_from(self.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        let rows = usize::try_from(self.mask_rows).map_err(|_| Error::ResourceLimit)?;
        let capacity = rows
            .checked_mul(ell)
            .and_then(|value| value.checked_sub(1))
            .ok_or(Error::ResourceLimit)?;
        if mask_count > capacity {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }
}

/// The public roots of the lookup messages, fixed before the lookup alpha.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LookupRoots {
    pub bits: Digest,
    pub mask: Digest,
}

/// The committed lookup messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupMessages {
    pub bits: CommittedValueGroup,
    /// The flat encoding tails of the multiplicity-bit message rows:
    /// `tails[row * r + i]`.
    pub bits_tails: Vec<Fp3>,
    pub mask_root: Digest,
    pub mask_rows: Vec<Vec<Fp3>>,
    pub mask_tails: Vec<Vec<Fp3>>,
    pub mask_salts: Vec<Vec<u8>>,
}

impl LookupMessages {
    pub fn bits_root(&self) -> Digest {
        self.bits.roots.first().copied().unwrap_or(Digest::ZERO)
    }

    pub fn roots(&self) -> LookupRoots {
        LookupRoots {
            bits: self.bits_root(),
            mask: self.mask_root,
        }
    }

    pub fn mask_entry(&self, plan: &LookupCommitPlan, slot: usize) -> Result<Fp3> {
        let ell = usize::try_from(plan.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        let row = slot / ell;
        let place = slot % ell;
        self.mask_rows
            .get(row)
            .and_then(|row| row.get(place))
            .copied()
            .ok_or(Error::ShapeMismatch)
    }

    pub fn mask_tree(&self, plan: &LookupCommitPlan) -> Result<RuntimeTree> {
        let mut codewords = Vec::with_capacity(self.mask_rows.len());
        for (row, tail) in self.mask_rows.iter().zip(&self.mask_tails) {
            codewords.push(encode_row_fp3(&plan.mask_code, row, tail)?);
        }
        RuntimeTree::commit(
            plan.mask_tree.clone(),
            &[RuntimePartRows {
                main: Vec::new(),
                mask: codewords,
                main_deg: 1,
            }],
            &self.mask_salts,
        )
    }
}

/// A deterministic byte stream for the standalone lookup.
pub struct LookupRandomSource {
    state: u64,
}

impl LookupRandomSource {
    pub fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    pub fn fp3(&mut self) -> Fp3 {
        Fp3::from_u64(self.next())
    }

    pub fn bytes(&mut self, count: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(count);
        while bytes.len() < count {
            bytes.extend_from_slice(&self.next().to_le_bytes());
        }
        bytes.truncate(count);
        bytes
    }
}

/// The random material of one lookup message commit: the encoding tails, the
/// pads, the salts and the mask rows. The key chain supplies it from the run's
/// private coins; a source derived from public identifiers is for standalone
/// tests only.
pub trait LookupRandomMaterial {
    fn fp3(&mut self) -> Result<Fp3>;
    fn bytes(&mut self, count: usize) -> Result<Vec<u8>>;
}

impl LookupRandomMaterial for LookupRandomSource {
    fn fp3(&mut self) -> Result<Fp3> {
        Ok(LookupRandomSource::fp3(self))
    }

    fn bytes(&mut self, count: usize) -> Result<Vec<u8>> {
        Ok(LookupRandomSource::bytes(self, count))
    }
}

/// The key lookup's private random material: the run's coin source, keyed by a
/// domain-separated private seed. Each draw is a labelled hash of that seed and
/// the draw number, so nothing is predictable from the public plan.
pub struct LookupCoins {
    coins: ByteMaskCoins,
}

impl LookupCoins {
    pub fn new(seed: Digest) -> Result<Self> {
        Ok(Self {
            coins: ByteMaskCoins::new(seed)?,
        })
    }
}

impl LookupRandomMaterial for LookupCoins {
    fn fp3(&mut self) -> Result<Fp3> {
        let (value, _) = self.coins.draw()?;
        Ok(value)
    }

    fn bytes(&mut self, count: usize) -> Result<Vec<u8>> {
        let words = count.div_ceil(24);
        let mut bytes = Vec::with_capacity(words.saturating_mul(24));
        for _ in 0..words {
            let (value, _) = self.coins.draw()?;
            for coefficient in value.c {
                bytes.extend_from_slice(&coefficient.as_canonical_u64().to_le_bytes());
            }
        }
        bytes.truncate(count);
        Ok(bytes)
    }
}

fn shape_randomness(
    shape: &WShape,
    source: &mut dyn LookupRandomMaterial,
) -> Result<(Vec<Fp3>, Vec<Fp3>, Vec<Vec<u8>>)> {
    let rows = shape.main_rows as usize;
    let tail = shape.code.r as usize;
    let mut tails = Vec::with_capacity(rows * tail);
    for _ in 0..rows * tail {
        tails.push(source.fp3()?);
    }
    let mut pad = Fp3::ZERO;
    while pad == Fp3::ZERO {
        pad = source.fp3()?;
    }
    let pads = vec![pad];
    let columns = 1usize << shape.code.log_n;
    let salts = (0..columns)
        .map(|_| source.bytes(shape.salt_bytes))
        .collect::<Result<Vec<_>>>()?;
    Ok((tails, pads, salts))
}

/// Draws the private encoding material and commits the three lookup messages.
/// The committed mask entries are the prover's hiding values; the opening
/// sampling and ledger entries are added when the lookup joins the session.
pub fn commit_lookup(
    plan: &LookupCommitPlan,
    bit_values: &[Fp3],
    mask_count: usize,
    source: &mut dyn LookupRandomMaterial,
) -> Result<LookupMessages> {
    plan.validate(bit_values.len(), mask_count)?;
    let (bits_tails, bits_pads, bits_salts) = shape_randomness(&plan.bits_shape, source)?;
    let bits = commit_value_group(
        &plan.bits_shape,
        bit_values.len() as u64,
        plan.bits_oracle,
        plan.group,
        &[bit_values.to_vec()],
        &bits_pads,
        std::slice::from_ref(&bits_tails),
        &[bits_salts],
        1,
    )?;
    let len = usize::try_from(plan.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
    let rows = usize::try_from(plan.mask_rows).map_err(|_| Error::ResourceLimit)?;
    let mut mask_rows = Vec::with_capacity(rows);
    for _ in 0..rows {
        mask_rows.push((0..len).map(|_| source.fp3()).collect::<Result<Vec<_>>>()?);
    }
    let mut pad = Fp3::ZERO;
    while pad == Fp3::ZERO {
        pad = source.fp3()?;
    }
    mask_rows[rows - 1][len - 1] = pad;
    let tail_len = usize::try_from(plan.mask_code.r).map_err(|_| Error::ResourceLimit)?;
    let mask_tails = (0..rows)
        .map(|_| (0..tail_len).map(|_| source.fp3()).collect::<Result<Vec<_>>>())
        .collect::<Result<Vec<_>>>()?;
    let columns = usize::try_from(plan.mask_tree.columns()).map_err(|_| Error::ResourceLimit)?;
    let mask_salts = (0..columns)
        .map(|_| source.bytes(plan.mask_tree.salt_bytes))
        .collect::<Result<Vec<_>>>()?;
    let messages = LookupMessages {
        bits,
        bits_tails,
        mask_root: Digest::ZERO,
        mask_rows,
        mask_tails,
        mask_salts,
    };
    let mask_root = *messages.mask_tree(plan)?.root();
    let messages = LookupMessages {
        mask_root,
        ..messages
    };
    Ok(messages)
}

#[cfg(test)]
mod private_source_tests {
    use super::*;
    use zkie_engine::pcs::request::code::build_code_plan;
    use zkie_engine::pcs::request::runtime_tree::RuntimeTreePart;

    fn commit_plan() -> LookupCommitPlan {
        let code = build_code_plan(8, 64, 4).unwrap();
        let bits_shape =
            WShape::new(16, code, 1, 1, 40, 3, 1, Digest([1; 32]), Digest([2; 32])).unwrap();
        LookupCommitPlan {
            bits_shape,
            bits_oracle: 1,
            group: 5,
            bits_openings: 1,
            mask_oracle: 2,
            mask_code: build_code_plan(8, 16, 4).unwrap(),
            mask_rows: 1,
            mask_openings: 1,
            mask_tree: RuntimeTreePlan {
                tree_number: 4,
                leaf_class: 3,
                log_n: 8,
                log_n_zk: 8,
                parts: vec![RuntimeTreePart {
                    main_symbols: 0,
                    mask_symbols: 3,
                }],
                salt_bytes: 40,
                run: Digest([3; 32]),
                configuration: Digest([4; 32]),
            },
        }
    }

    fn bit_values() -> Vec<Fp3> {
        (0..16u64).map(Fp3::from_u64).collect()
    }

    fn commit_with_seed(seed: Digest) -> LookupMessages {
        let plan = commit_plan();
        let mut source = LookupCoins::new(seed).unwrap();
        commit_lookup(&plan, &bit_values(), 1, &mut source).unwrap()
    }

    #[test]
    fn the_private_source_is_deterministic() {
        let first = commit_with_seed(Digest([9; 32]));
        let second = commit_with_seed(Digest([9; 32]));
        assert_eq!(first.bits_root(), second.bits_root());
        assert_eq!(first.mask_root, second.mask_root);
        assert_eq!(first.mask_rows, second.mask_rows);
    }

    #[test]
    fn another_private_seed_commits_other_messages() {
        let first = commit_with_seed(Digest([9; 32]));
        let other = commit_with_seed(Digest([10; 32]));
        assert_ne!(first.bits_root(), other.bits_root());
        assert_ne!(first.mask_root, other.mask_root);
        assert_ne!(first.mask_rows, other.mask_rows);
    }

    #[test]
    fn the_public_linear_source_still_serves_its_tests() {
        let plan = commit_plan();
        let mut source = LookupRandomSource::new(7);
        let messages = commit_lookup(&plan, &bit_values(), 1, &mut source).unwrap();
        let private = commit_with_seed(Digest([9; 32]));
        assert_ne!(messages.mask_root, private.mask_root);
    }
}
