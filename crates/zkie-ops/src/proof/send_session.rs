//! Single-oracle fixture SEND data-flow demonstration.
//!
//! A fixture provider constructs sample answers from the actual plan's G_u
//! oracle shape, the independent seed set and a fixture witness, using the
//! sample-form coefficient row and its pad positions over the extension field.
//! The verifier receives those answers as pending sample claims; it has no
//! standard-answer comparison path. The provider is not a real SEND provider.
//!
//! The fixed order is: fixed oracle commitment, independent seed draw, the
//! complete ordered answer list, sample registration, record/column joins, then
//! the same continuous chain is transferred to BAT for the frozen group and
//! challenge. No challenge is drawn between the seed and the answers.
//!
//! The deterministic public pads are test-only values and do not imply hiding.
//! The provider's plaintext witness and its arithmetic are an auxiliary test
//! oracle; they are not the verification path. The provider cannot replace the
//! fixed oracle commitment after the seed is drawn; the session stage rejects
//! it. A hash or challenge-prefix change is not source authentication and no
//! collision guarantee is claimed. BAT only combines the pending declarations
//! and does not judge truth; no later proximity is connected.

use std::cell::RefCell;
use std::rc::Rc;

use zkie_core::error::Result;
use zkie_core::types::Digest;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::gu::{GuNotice, GuRegistry, GuShape};
use zkie_engine::pcs::request::relation::{gu_table, sample_row, GuWitness};
use zkie_engine::pcs::request::table::SampleSeed;

use crate::proof::column_session::{ColumnBindingInput, GU_NOTICE_ROOT};
use crate::proof::gu_combine_session::{
    verify_gu_combine_session_with_provider, GuCombineOutcome,
};
use crate::proof::model_commitment::{AssetOpening, ModelCommitmentPublic};
use crate::proof::session_baseline::TrustedModel;

/// The fixed, self-reported G_u root used by the fixture SEND demonstration.
pub const FIXTURE_GU_ROOT: Digest = GU_NOTICE_ROOT;

/// A fixture provider for one G_u oracle. The witness and pads are local test
/// data; the provider is not a real SEND provider and the witness is not
/// authenticated from the G_u commitment.
pub struct FixtureSendProvider {
    shape: GuShape,
    witness: GuWitness,
    registry: GuRegistry,
}

impl FixtureSendProvider {
    pub fn new(shape: &GuShape, root: Digest) -> Result<Self> {
        let message: Vec<Fp3> = (0..shape.index.code.ell)
            .map(|index| Fp3::from_u64(0x500 + index))
            .collect();
        let tail: Vec<Fp3> = (0..shape.index.code.r)
            .map(|index| Fp3::from_u64(0x600 + index))
            .collect();
        let witness = gu_table(shape, &message, &tail)?;
        let mut registry = GuRegistry::new();
        registry.receive(
            shape,
            &GuNotice {
                root,
                oracle: shape.oracle,
                tree_number: shape.tree.tree_number,
                part: shape.part,
            },
        )?;
        Ok(Self {
            shape: shape.clone(),
            witness,
            registry,
        })
    }

    /// Computes the complete ordered answers from the sample-form rows and the
    /// fixture witness. This plaintext evaluation is an auxiliary test oracle,
    /// not a verifier comparison.
    pub fn answers(&self, seeds: &[SampleSeed]) -> Result<Vec<Fp3>> {
        let mut answers = Vec::with_capacity(self.shape.u.answers as usize);
        for answer in 0..self.shape.u.answers {
            let row = sample_row(&self.shape, &self.registry, seeds, answer, Fp3::ZERO)?;
            answers.push(row.evaluate(&self.witness)?);
        }
        Ok(answers)
    }
}

/// The verifier outcome plus the recorded seed/answer list and the provider
/// answers. The recorded values are evidence for the test oracle; the verifier
/// did not compare them with a standard answer.
pub struct SendOutcome {
    pub combine: GuCombineOutcome,
    pub gu_shape: GuShape,
    pub gu_root: Digest,
    pub seeds: Vec<SampleSeed>,
    pub provider_answers: Vec<Fp3>,
}

fn run_send<F>(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    input: &ColumnBindingInput,
    asset_path: &str,
    opening: &AssetOpening,
    gu_root: Digest,
    provider: F,
) -> Result<SendOutcome>
where
    F: FnOnce(&GuShape, &[SampleSeed]) -> Result<Vec<Fp3>>,
{
    let seeds = Rc::new(RefCell::new(Vec::new()));
    let answers = Rc::new(RefCell::new(Vec::new()));
    let shape_cell = Rc::new(RefCell::new(None));
    let recorded_seeds = Rc::clone(&seeds);
    let recorded_answers = Rc::clone(&answers);
    let recorded_shape = Rc::clone(&shape_cell);
    let combine = verify_gu_combine_session_with_provider(
        public,
        trusted,
        input,
        asset_path,
        opening,
        gu_root,
        move |shape, seed_set| {
            recorded_shape.replace(Some(shape.clone()));
            recorded_seeds.replace(seed_set.to_vec());
            let values = provider(shape, seed_set)?;
            recorded_answers.replace(values.clone());
            Ok(values)
        },
    )?;
    Ok(SendOutcome {
        combine,
        gu_shape: shape_cell.take().ok_or(zkie_core::error::Error::InvalidSessionState)?,
        gu_root,
        seeds: seeds.take(),
        provider_answers: answers.take(),
    })
}

/// Runs the fixture-provider SEND demonstration. The provider constructs the
/// answers; the verifier only registers them.
pub fn verify_send_session(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    input: &ColumnBindingInput,
    asset_path: &str,
    opening: &AssetOpening,
) -> Result<SendOutcome> {
    run_send(
        public,
        trusted,
        input,
        asset_path,
        opening,
        FIXTURE_GU_ROOT,
        |shape, seeds| FixtureSendProvider::new(shape, FIXTURE_GU_ROOT)?.answers(seeds),
    )
}

/// Runs the same flow with a caller-supplied provider. This is used by tests to
/// show that a syntactically valid wrong answer is registered as pending, not
/// rejected by the verifier.
pub fn verify_send_session_with_provider<F>(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    input: &ColumnBindingInput,
    asset_path: &str,
    opening: &AssetOpening,
    gu_root: Digest,
    provider: F,
) -> Result<SendOutcome>
where
    F: FnOnce(&GuShape, &[SampleSeed]) -> Result<Vec<Fp3>>,
{
    run_send(
        public,
        trusted,
        input,
        asset_path,
        opening,
        gu_root,
        provider,
    )
}

