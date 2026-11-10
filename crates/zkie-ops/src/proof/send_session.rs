//! Single-oracle fixture SEND data-flow demonstration with a typed witness.
//!
//! A typed fixture witness stores the complete G_u main/mask message rows,
//! their encoding tails and the per-column salts. Each row is encoded with the
//! oracle code and committed through the existing `RuntimeTree::commit` path;
//! the resulting root is fixed before the SEND seed. The provider then derives
//! the sample answers from the same shared witness object after the seed.
//!
//! The verifier registers the answers as pending sample claims; it has no
//! standard-answer comparison path. The provider's plaintext witness and
//! arithmetic are an auxiliary test oracle, not the verification path. The
//! deterministic public pads are test-only values and do not imply hiding.
//!
//! The witness fields are private and the canonical constructors derive the
//! relation view, codewords and root together. A caller cannot mutate one
//! field independently. `from_committed` rechecks the encoding-to-root and
//! relation consistency, so a forged relation with the same stored root is
//! rejected. This is a test-fixture guarantee only; it must not be extended to
//! production, where the oracle root and witness come from the protocol and
//! authenticated sources.
//!
//! Source table:
//!
//! | item | source and layout |
//! | --- | --- |
//! | main rows | `u.main_rows` fixture messages of `u.code.ell` entries, each with `u.code.r` tail entries; encoded by `u.code` into `2^tree.log_n` symbols |
//! | mask rows | `u.mask_rows` fixture messages of the mask code's `ell`, each with its `r` tail; encoded by the mask code into `2^tree.log_n_zk` symbols |
//! | salts | one deterministic test salt per tree column (`tree.columns()`, `tree.salt_bytes`) |
//! | pads | reserved values inside the main/mask message tables, encoded and committed before the seed; the seed's `beta` only supplies the linear coefficient; deterministic test values, not hiding |
//! | relation view | row-major flattening of the same main/mask message rows; tails are committed but not part of the relation view |
//! | tree part | a single part at index 0 is required; the part widths must equal `main_rows * 3` and `mask_rows * 3`, and the code depths must match the tree |
//!
//! Unsupported multi-part or mismatched layouts are rejected; no mismatched
//! root is constructed.
//!
//! Order: fixed committed oracle root, independent seed, complete ordered
//! answers, sample registration, record/column joins, then the same continuous
//! chain transferred to BAT. A missing or unsupported G_u shape/tree layout is
//! rejected; no mismatched root is constructed.
//!
//! Still unclosed: real G_u/asset authentication, real SEND, multi-oracle SEND,
//! random pads/hiding, complete FOLD/proximity, single-shot and production.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::code::encode_row_fp3;
use zkie_engine::pcs::request::field::{Fp3, E_DEGREE};
use zkie_engine::pcs::request::gu::{GuNotice, GuRegistry, GuShape};
use zkie_engine::pcs::request::relation::{sample_row, GuWitness};
use zkie_engine::pcs::request::runtime_tree::{RuntimePartRows, RuntimeTree};
use zkie_engine::pcs::request::table::SampleSeed;

use crate::proof::column_session::{
    build_authenticated_encoding_table_with_configuration, demo_fold_shape, demo_gu_shape,
    AssetEncodingTable, ColumnBindingInput,
};
use crate::proof::gu_combine_session::{
    verify_gu_combine_session_with_table_and_provider, GuCombineOutcome,
};
use crate::proof::model_commitment::{AssetOpening, ModelCommitmentPublic};
use crate::proof::session_baseline::TrustedModel;

/// A typed fixture witness for one G_u oracle: complete main/mask message rows,
/// encoding tails and per-column salts. The root is produced by encoding these
/// rows with the oracle code and committing the resulting codewords through the
/// existing runtime tree.
pub struct CommittedGuWitness {
    shape: GuShape,
    main_messages: Vec<Vec<Fp3>>,
    main_tails: Vec<Vec<Fp3>>,
    mask_messages: Vec<Vec<Fp3>>,
    mask_tails: Vec<Vec<Fp3>>,
    salts: Vec<Vec<u8>>,
    main_codewords: Vec<Vec<Fp3>>,
    mask_codewords: Vec<Vec<Fp3>>,
    root: Digest,
    relation: GuWitness,
}

impl CommittedGuWitness {
    pub fn shape(&self) -> &GuShape { &self.shape }
    pub fn root(&self) -> Digest { self.root }
    pub fn main_messages(&self) -> &[Vec<Fp3>] { &self.main_messages }
    pub fn main_tails(&self) -> &[Vec<Fp3>] { &self.main_tails }
    pub fn mask_messages(&self) -> &[Vec<Fp3>] { &self.mask_messages }
    pub fn mask_tails(&self) -> &[Vec<Fp3>] { &self.mask_tails }
    pub fn main_codewords(&self) -> &[Vec<Fp3>] { &self.main_codewords }
    pub fn mask_codewords(&self) -> &[Vec<Fp3>] { &self.mask_codewords }
    pub fn relation(&self) -> &GuWitness { &self.relation }
}

impl CommittedGuWitness {
    pub fn new(shape: &GuShape) -> Result<Self> {
        Self::new_with_tag(shape, 0)
    }

    /// Builds the deterministic fixture with a tag that separates independent
    /// witnesses for replacement tests.
    pub fn new_with_tag(shape: &GuShape, tag: u64) -> Result<Self> {
        if shape.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        if shape.tree.parts.len() != 1 || shape.part != 0 {
            return Err(Error::UnsupportedPrimitive);
        }
        let part = shape.tree.parts.first().ok_or(Error::InvalidConfig)?;
        let main_rows = usize::try_from(shape.u.main_rows).map_err(|_| Error::ResourceLimit)?;
        let mask_rows = usize::try_from(shape.u.mask_rows).map_err(|_| Error::ResourceLimit)?;
        let mask_code = shape.u.mask_code.ok_or(Error::UnsupportedPrimitive)?;
        let expected_main = shape
            .u
            .main_rows
            .checked_mul(shape.u.main_deg as u64)
            .ok_or(Error::ResourceLimit)?;
        let expected_mask = shape
            .u
            .mask_rows
            .checked_mul(E_DEGREE as u64)
            .ok_or(Error::ResourceLimit)?;
        if shape.u.main_deg != E_DEGREE as u32
            || part.main_symbols != expected_main
            || part.mask_symbols != expected_mask
            || shape.tree.log_n != shape.u.code.log_n
            || shape.tree.log_n_zk != mask_code.log_n
        {
            return Err(Error::UnsupportedPrimitive);
        }
        let main_messages = (0..main_rows)
            .map(|row| {
                (0..shape.u.code.ell)
                    .map(|column| {
                        Fp3::from_u64(
                            0x700 + tag * 0x1000 + (row as u64 * shape.u.code.ell + column),
                        )
                    })
                    .collect()
            })
            .collect::<Vec<Vec<Fp3>>>();
        let main_tails = (0..main_rows)
            .map(|row| {
                (0..shape.u.code.r)
                    .map(|column| Fp3::from_u64(0x800 + tag * 0x1000 + row as u64 * 0x10 + column))
                    .collect()
            })
            .collect::<Vec<Vec<Fp3>>>();
        let mask_messages = (0..mask_rows)
            .map(|row| {
                (0..mask_code.ell)
                    .map(|column| {
                        Fp3::from_u64(
                            0x900 + tag * 0x1000 + (row as u64 * mask_code.ell + column),
                        )
                    })
                    .collect()
            })
            .collect::<Vec<Vec<Fp3>>>();
        let mask_tails = (0..mask_rows)
            .map(|row| {
                (0..mask_code.r)
                    .map(|column| Fp3::from_u64(0xa00 + tag * 0x1000 + row as u64 * 0x10 + column))
                    .collect()
            })
            .collect::<Vec<Vec<Fp3>>>();
        let columns = usize::try_from(shape.tree.columns()).map_err(|_| Error::ResourceLimit)?;
        let salts = (0..columns)
            .map(|position| vec![(position as u8) ^ 0xA5; shape.tree.salt_bytes])
            .collect::<Vec<Vec<u8>>>();
        let mut main_codewords = Vec::with_capacity(main_rows);
        for (message, tail) in main_messages.iter().zip(&main_tails) {
            main_codewords.push(encode_row_fp3(&shape.u.code, message, tail)?);
        }
        let mut mask_codewords = Vec::with_capacity(mask_rows);
        for (message, tail) in mask_messages.iter().zip(&mask_tails) {
            mask_codewords.push(encode_row_fp3(&mask_code, message, tail)?);
        }
        let tree = RuntimeTree::commit(
            shape.tree.clone(),
            &[RuntimePartRows {
                main: main_codewords.clone(),
                mask: mask_codewords.clone(),
                main_deg: shape.u.main_deg,
            }],
            &salts,
        )?;
        let root = *tree.root();
        let mut relation_main = Vec::new();
        for message in &main_messages {
            relation_main.extend_from_slice(message);
        }
        let mut relation_mask = Vec::new();
        for message in &mask_messages {
            relation_mask.extend_from_slice(message);
        }
        Ok(Self {
            shape: shape.clone(),
            main_messages,
            main_tails,
            mask_messages,
            mask_tails,
            salts,
            main_codewords,
            mask_codewords,
            root,
            relation: GuWitness {
                main: relation_main,
                mask: relation_mask,
            },
        })
    }

    /// Canonical constructor with a specific value in the reserved mask sample
    /// pad slot. The pad value is part of the committed mask message; the seed
    /// only supplies its linear coefficient. This is a test-only fixture entry.
    pub fn new_with_mask_pad(shape: &GuShape, pad: Fp3) -> Result<Self> {
        let mut witness = Self::new_with_tag(shape, 0)?;
        let mask_code = shape.u.mask_code.ok_or(Error::UnsupportedPrimitive)?;
        if shape.u.mask_rows != 1 || shape.u.answers == 0 {
            return Err(Error::UnsupportedPrimitive);
        }
        let pad_index = usize::try_from(mask_code.ell - shape.u.answers as u64)
            .map_err(|_| Error::ResourceLimit)?;
        if pad_index >= witness.mask_messages[0].len() {
            return Err(Error::ShapeMismatch);
        }
        witness.mask_messages[0][pad_index] = pad;
        let (root, main_codewords, mask_codewords, relation) =
            witness.recompute_parts()?;
        witness.root = root;
        witness.main_codewords = main_codewords;
        witness.mask_codewords = mask_codewords;
        witness.relation = relation;
        Ok(witness)
    }

    fn recompute_parts(&self) -> Result<(Digest, Vec<Vec<Fp3>>, Vec<Vec<Fp3>>, GuWitness)> {
        let mut main_codewords = Vec::with_capacity(self.main_messages.len());
        for (message, tail) in self.main_messages.iter().zip(&self.main_tails) {
            main_codewords.push(encode_row_fp3(&self.shape.u.code, message, tail)?);
        }
        let mask_code = self.shape.u.mask_code.ok_or(Error::UnsupportedPrimitive)?;
        let mut mask_codewords = Vec::with_capacity(self.mask_messages.len());
        for (message, tail) in self.mask_messages.iter().zip(&self.mask_tails) {
            mask_codewords.push(encode_row_fp3(&mask_code, message, tail)?);
        }
        let tree = RuntimeTree::commit(
            self.shape.tree.clone(),
            &[RuntimePartRows {
                main: main_codewords.clone(),
                mask: mask_codewords.clone(),
                main_deg: self.shape.u.main_deg,
            }],
            &self.salts,
        )?;
        let mut relation_main = Vec::new();
        for message in &self.main_messages {
            relation_main.extend_from_slice(message);
        }
        let mut relation_mask = Vec::new();
        for message in &self.mask_messages {
            relation_mask.extend_from_slice(message);
        }
        Ok((
            *tree.root(),
            main_codewords,
            mask_codewords,
            GuWitness {
                main: relation_main,
                mask: relation_mask,
            },
        ))
    }

    /// Re-encodes the stored rows/tails and re-commits the codewords. This is
    /// the independent encoding-to-root consistency check.
    pub fn recompute_root(&self) -> Result<Digest> {
        Ok(self.recompute_parts()?.0)
    }
}

/// The provider holds the same shared witness object as the fixed root.
pub struct FixtureSendProvider {
    committed: Arc<CommittedGuWitness>,
    registry: GuRegistry,
}

impl FixtureSendProvider {
    pub fn from_committed(committed: Arc<CommittedGuWitness>) -> Result<Self> {
        Self::from_committed_with_root(committed, None)
    }

    /// Builds a provider and optionally requires the caller's fixed root to
    /// equal the witness root. A different root is rejected before sampling.
    pub fn from_committed_with_root(
        committed: Arc<CommittedGuWitness>,
        fixed_root: Option<Digest>,
    ) -> Result<Self> {
        if let Some(root) = fixed_root {
            if root != committed.root() {
                return Err(Error::StatementMismatch);
            }
        }
        if committed.recompute_root()? != committed.root()
            || committed.recompute_parts()?.3 != committed.relation
        {
            return Err(Error::StatementMismatch);
        }
        let mut registry = GuRegistry::new();
        registry.receive(
            committed.shape(),
            &GuNotice {
                root: committed.root(),
                oracle: committed.shape().oracle,
                tree_number: committed.shape().tree.tree_number,
                part: committed.shape().part,
            },
        )?;
        Ok(Self {
            committed,
            registry,
        })
    }

    pub fn committed(&self) -> &Arc<CommittedGuWitness> {
        &self.committed
    }

    /// Computes the complete ordered answers from the sample-form rows and the
    /// shared relation view. The plaintext evaluation is an auxiliary test
    /// oracle, not a verifier comparison.
    pub fn answers(&self, seeds: &[SampleSeed]) -> Result<Vec<Fp3>> {
        let mut answers = Vec::with_capacity(self.committed.shape().u.answers as usize);
        for answer in 0..self.committed.shape().u.answers {
            let row = sample_row(
                self.committed.shape(),
                &self.registry,
                seeds,
                answer,
                Fp3::ZERO,
            )?;
            answers.push(row.evaluate(self.committed.relation())?);
        }
        Ok(answers)
    }
}

/// The verifier outcome plus the recorded seed/answer list and the fixed
/// witness object.
pub struct SendOutcome {
    pub combine: GuCombineOutcome,
    pub committed: Arc<CommittedGuWitness>,
    pub gu_shape: GuShape,
    pub gu_root: Digest,
    pub seeds: Vec<SampleSeed>,
    pub provider_answers: Vec<Fp3>,
}

fn run_send_table<F>(
    table: &AssetEncodingTable,
    committed: Arc<CommittedGuWitness>,
    input: &ColumnBindingInput,
    provider: F,
) -> Result<SendOutcome>
where
    F: FnOnce(Arc<CommittedGuWitness>, &GuShape, &[SampleSeed]) -> Result<Vec<Fp3>>,
{
    let seeds = Rc::new(RefCell::new(Vec::new()));
    let answers = Rc::new(RefCell::new(Vec::new()));
    let shape_cell = Rc::new(RefCell::new(None));
    let recorded_seeds = Rc::clone(&seeds);
    let recorded_answers = Rc::clone(&answers);
    let recorded_shape = Rc::clone(&shape_cell);
    let committed_for_provider = Arc::clone(&committed);
    let combine = verify_gu_combine_session_with_table_and_provider(
        input,
        table,
        committed.root(),
        move |shape, seed_set| {
            if shape != committed_for_provider.shape() {
                return Err(Error::StatementMismatch);
            }
            recorded_shape.replace(Some(shape.clone()));
            recorded_seeds.replace(seed_set.to_vec());
            let values = provider(Arc::clone(&committed_for_provider), shape, seed_set)?;
            recorded_answers.replace(values.clone());
            Ok(values)
        },
    )?;
    Ok(SendOutcome {
        combine,
        gu_shape: shape_cell.take().ok_or(Error::InvalidSessionState)?,
        gu_root: committed.root(),
        committed,
        seeds: seeds.take(),
        provider_answers: answers.take(),
    })
}

fn build_table(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    input: &ColumnBindingInput,
    asset_path: &str,
    opening: &AssetOpening,
) -> Result<AssetEncodingTable> {
    build_authenticated_encoding_table_with_configuration(
        public,
        trusted,
        asset_path,
        opening,
        input.parameters.digest()?,
    )
}

/// Runs the fixture-provider SEND demonstration. The provider and the root use
/// the same committed witness object.
pub fn verify_send_session(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    input: &ColumnBindingInput,
    asset_path: &str,
    opening: &AssetOpening,
) -> Result<SendOutcome> {
    let table = build_table(public, trusted, input, asset_path, opening)?;
    let fold_shape = demo_fold_shape(table.shape)?;
    let gu_shape = demo_gu_shape(table.shape, fold_shape.mask_code)?;
    let committed = Arc::new(CommittedGuWitness::new(&gu_shape)?);
    run_send_table(&table, committed, input, |committed, _, seeds| {
        FixtureSendProvider::from_committed(committed)?.answers(seeds)
    })
}

/// Runs the flow with a caller-supplied committed witness and provider. The
/// witness shape must match the session plan; a mismatched witness is rejected
/// before the seed.
pub fn verify_send_session_with_committed<F>(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    input: &ColumnBindingInput,
    asset_path: &str,
    opening: &AssetOpening,
    committed: Arc<CommittedGuWitness>,
    provider: F,
) -> Result<SendOutcome>
where
    F: FnOnce(Arc<CommittedGuWitness>, &GuShape, &[SampleSeed]) -> Result<Vec<Fp3>>,
{
    let table = build_table(public, trusted, input, asset_path, opening)?;
    let fold_shape = demo_fold_shape(table.shape)?;
    let gu_shape = demo_gu_shape(table.shape, fold_shape.mask_code)?;
    if committed.shape() != &gu_shape {
        return Err(Error::StatementMismatch);
    }
    run_send_table(&table, committed, input, provider)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::code::build_code_plan;
    use zkie_engine::pcs::request::params::LEAF_SALT_MIN_BYTES;
    use zkie_engine::pcs::request::table::{IndexShape, TableDistances};

    fn shape() -> GuShape {
        let index = IndexShape {
            code: build_code_plan(3, 4, 1).unwrap(),
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
        let fold = crate::proof::column_session::demo_fold_shape(index).unwrap();
        demo_gu_shape(index, fold.mask_code).unwrap()
    }

    #[test]
    fn a_forged_relation_with_the_same_root_is_rejected() {
        let shape = shape();
        let canonical = CommittedGuWitness::new(&shape).unwrap();
        let mut forged = CommittedGuWitness::new(&shape).unwrap();
        forged.relation.mask[0] = forged.relation.mask[0].add(&Fp3::ONE);
        assert_eq!(forged.root(), canonical.root());
        assert!(matches!(
            FixtureSendProvider::from_committed(Arc::new(forged)),
            Err(Error::StatementMismatch)
        ));
    }
}
