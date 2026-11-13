//! Hidden SEND-sample interface for the activation matrix and the reduced
//! weight rows.
//!
//! The output-head relation needs two private inputs before it can draw any
//! challenge:
//!
//! - `X`, one activation matrix with one row for each of the 16 positions and
//!   one column for each hidden coordinate (1024 in the output head). Its rows
//!   are the row-weight vectors of the linear claim.
//! - `W`, one reduced row for each position, `W_t = g_t^T M`, where `g_t` is
//!   the position's row-weight vector and `M` is the committed encoded weight
//!   table. `W_t` is exactly the row the Eval relation reads at the position's
//!   challenge point.
//!
//! This module is the first unit of that hidden interface. It fixes `X` under a
//! hiding row commitment before the session draws anything, derives the `W`
//! rows from the committed encoded table through the table's own row source
//! (never through a plaintext matrix), commits each `W` row with the
//! product-form codeword encoding and a runtime tree, runs the SEND sample
//! phase over the committed rows, registers the sample relation rows and the
//! group, and builds the Eval relation rows that the first, shared and second
//! stages consume later.
//!
//! # X: a hiding row commitment
//!
//! A row commitment is a Merkle tree over the rows of `X`. One leaf binds the
//! position, the row width, the row values, one random salt of 40 to 64 bytes
//! and the configuration digest. The salt of every row is private and fresh;
//! the tree root is the only public value. Without a salt an activation row of
//! a small range could be enumerated from its leaf. With the salt the leaf is
//! hiding in the standard model of a salted commitment; the commitment is
//! binding through collision resistance of the labelled hash.
//!
//! The public commitment carries the root, the shape and the configuration.
//! Openings are available for later stages that must read a row at a
//! challenge-derived point, but the SEND phase itself opens no activation row:
//! the verifier never receives `X`, only the root. The prover keeps the rows
//! and salts in [`CommittedActivation`].
//!
//! This commitment uses the repository's labelled hash and Merkle primitives;
//! it does not invent a second hash or a second tree format. The activation
//! leaf domain is the runtime leaf domain with a distinct tree number, so an
//! activation leaf cannot be read as a runtime-tree leaf or the reverse.
//!
//! # W: derived from the committed encoded table
//!
//! The encoded table is the streaming commitment of the table layer: a Merkle
//! tree over the encoded columns, a record of out-of-domain sample answers, and
//! a reader that names the root, shape, configuration and record. The prover
//! holds the table's row source. [`derive_w_rows`] makes one pass over that
//! source and, for every table row `r` and low place `j`, computes
//!
//! ```text
//! W_t[j + 2^k_lo r] = sum_y g_t[y] * row_r[j + 2^k_lo y]
//! ```
//!
//! which is the matrix mapping of the committed table, without materializing
//! the complete weight matrix and without building a plaintext table object.
//! The row source is the same read path the table committer used; the complete
//! table never leaves the prover.
//!
//! Each `W_t` is then placed in the main rows of an oracle message, encoded
//! with the product-form codeword encoding, given a private random tail and one
//! salt per tree column, and committed under a runtime tree. The per-position
//! trees form the SEND group; the ordered roots are the group registration.
//!
//! # SEND samples and Eval rows
//!
//! After the roots are fixed the session draws one shared seed for each answer
//! number. Each `W_t` message gets one answer per seed, computed with the
//! existing sample form (the `eq` weighting over the message cube, plus the
//! reserved pad entry under the seed's `beta`). The answers are public; the
//! sampled row values are not opened. The sample relation rows are built from
//! the drawn seed and the received answer and stay pending, exactly as the
//! existing sample interface leaves them: the later proximity tests decide
//! them. A production verifier does not compare them with a plaintext
//! evaluation.
//!
//! The unit's in-memory verifier entry additionally holds the committed `W`
//! witness, as the existing fixture SEND sessions do, and uses it only to
//! evaluate the same linear forms: it never receives the rows through the
//! proof bytes. This makes the unit's negative tests meaningful (a changed
//! answer is rejected here) while the protocol-shaped entry
//! [`verify_hidden_send_public`] keeps the sample claims pending. The witness
//! check is an auxiliary fixture check, not a production verification path.
//!
//! After the answers the verifier draws the opening positions of every `W`
//! tree and checks the runtime openings against the registered roots, then
//! draws one Eval point per position. When the plan carries the sampled
//! mask-only message, the Eval row has the masked form
//!
//! ```text
//! W_t at psi_t + e_t  =  tau_t
//! ```
//!
//! with the `eq(psi_t, .)` coefficients on the `W_t` message, one unit
//! coefficient on the position's `e_t` entry of the committed mask message and
//! the public masked target `tau_t`. The mask entry is fixed before any
//! challenge, so `tau_t` is uniform and the row is a hiding binding, not a
//! public evaluation. `X`'s row-weight vector is what produced `W_t`; it is
//! not revealed. The sample rows, the Eval rows and the masked Dot right-hand
//! side rows belong to the same group and are checked by its proximity test.
//!
//! # Order
//!
//! 1. the activation commitment, the `W` roots and the mask message root are
//!    fixed;
//! 2. the roots and the group declaration are absorbed and the shared seeds
//!    are drawn;
//! 3. the complete ordered answers are absorbed;
//! 4. the `W` and mask opening positions are drawn; the openings are checked;
//! 5. one Eval point per position is absorbed and drawn;
//! 6. the masked Eval targets are absorbed and the first message value is
//!    fixed.
//!
//! No position is drawn between the seed and the answers. The seed is drawn by
//! the verifier from the chain; the message copy of the seed is checked against
//! the drawn value and is never used as a challenge source.
//!
//! # Scope
//!
//! This unit fixes and samples the hidden inputs; it does not run the
//! proximity tests, the code switch or a soundness bound. The relation between
//! the `W` roots and the encoded table root is established by the later
//! Eval/Dot/Fold relations, not by the SEND phase. The encoded table identity
//! is checked against the plan, and the `W` derivation reads the table's row
//! source and checks its shape, but no function here proves to the verifier
//! that the `W` message is `g^T M`: the verifier holds no `M` and no `X`. The
//! sample answers are public and padded with a private random value; the pads
//! and tails of this unit are supplied by the caller and the tests use
//! deterministic values, which are not hiding evidence.
//!
//! The unit is deliberately in-memory and uses bounded vectors; the production
//! memory path and the durable table read are not part of it.

//! # The merged three-stage session
//!
//! The `HiddenLin*` types below extend this unit into one session that runs the
//! first, shared and second parts of the linear request without handing data
//! to a table-based component. The first part fixes the activation commitment,
//! the public claim items, the reduced weight roots and the Eval rows and
//! seals. The shared part resumes from that seal, folds the committed encoded
//! table through its row source, authenticates the record and the drawn
//! columns, sends the folded row and `Omega`, registers the Fold row and runs
//! a proximity test on the folded-row group. The second part starts from the
//! first query and the shared seal, adopts `Omega`, registers the masked Dot
//! right-hand side rows, draws the batching challenge, registers the committed
//! product-sumcheck rows of the hidden Dot identity, and runs the proximity
//! test on the request's own group, ending in the BASE consumption and the
//! final seal.
//!
//! The public claim items are the activation commitment root, the output `Y`,
//! the public row weights and the table identity. `Y` is bound through the
//! Eval target (a function of `Y` at the drawn point), not by a digest. The
//! verifier never receives activation rows, table message rows or reduced
//! weight rows; it checks the committed roots, sample answers, drawn positions,
//! openings and the two BAT/FOLD/BASE proximity tests. The prover keeps the
//! table row source, the activation rows and every private tail, salt and pad.
//!
//! Limits: both proximity groups must share one main and one mask code, so a
//! split for differing codes is rejected by the plan rather than silently
//! dropped. The proximity tests use the engine's BAT/FOLD/BASE components with
//! no code-switch stage, and the session gives no formal soundness bound,
//! single-shot decode or production CLI wiring. The public entry keeps the
//! Eval/Dot/Fold claims pending until the proximity tests accept them; the
//! auxiliary witness entry checks the prover's samples in unit tests only.
//!
//! # Hidden-input mode
//!
//! A plan with an empty public weights list selects the hidden-input mode. The
//! verifier then holds no activation row: the first statement binds the
//! activation commitment (which is already part of the SEND statement) and the
//! sampled index-mask message. The Eval targets are the masked values
//! `W_t(psi_t) + e_t`, each bound by a BAT row to the committed reduced row and
//! the committed `e_t` entry; the masked Dot right-hand side
//! `W_t(psi*) + g_t` is bound the same way. The reduced rows are the MatMul
//! output and are committed under the same hiding product-form encoding, so
//! the output never needs a separate plaintext input. The diagnostic mode
//! keeps the public-weight Dot row and has no mask message.
//!
//! The hidden-input mode also commits a shown-map message: one main table
//! holding the activation rows, with its own encoding tail, salts and sample
//! pad. After the tilde values and the batching challenge, the prover sends
//! the public residual `R = sum_t chi^t (g_t - c_t)` and the masked
//! product-sumcheck transcript. The verifier absorbs the residual before the
//! product-sumcheck challenges and rebuilds the adjusted target
//! `sum_t chi^t tilde_t - R`, the `(U)` row, the two `(E_j)` end-value rows
//! and one difference row `sum_t chi^t g_t - sum_t chi^t c_t = R` from it.
//! Those rows and the existing tilde rows are registered in the own group, so
//! the BAT/FOLD/BASE test checks `<X_t, Omega> = W_t(psi*)` against the
//! committed shown maps and the committed reduced rows. The shown-map
//! commitment is pinned like the activation commitment: the equality of the
//! two commitments of `X` is an operator warranty, not a proof. The mask
//! entries come from the separate one-time schema: the hiding coefficients,
//! the two end masks, the claim masks `c_t`, the tilde masks `g_t` and the
//! Eval masks `e_t` occupy disjoint slots and none is reused. The claim and
//! tilde masks are independent draws from the same stage-1 stream; the one
//! difference row is their only consistency use and does not force equality.

use std::sync::Arc;

use rayon::prelude::*;
use zkie_core::common::codec::DecodeLimits;
use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use zkie_core::common::field::PrimeField64;
use zkie_engine::compute::cpu::{merkle_build, merkle_verify};
use zkie_engine::pcs::request::bat::{
    BatGroupLayout, BatOracleShape, BatPart, BatRow, BatRowKind, BatSession,
    BatStatement,
};
use zkie_engine::pcs::request::chain::{
    Chain, ChainMessage, KIND_ALGEBRAIC, KIND_EMPTY, KIND_EPSILON, KIND_POSITIONS,
    KIND_SEED,
};
use zkie_engine::pcs::request::code::{build_code_plan, encode_row_fp3, CodeParams};
use zkie_engine::pcs::request::field::{Fp, Fp3, E_DEGREE};
use zkie_engine::pcs::request::fold::{
    fold_mask_make, fold_mask_make_tail, fold_mask_notice, fold_mask_shape_fault,
    fold_member_firsts, fold_output, fold_start_prove, fold_start_verify, ByteMaskCoins,
    FoldGroupLayout, FoldPart, FoldProverSession,
    FoldRoundVerifier, FoldShape, FoldStartMessage, MaskCoins, MaskDraft, MaskOracle,
    OracleMetadata, RuntimeOracles, VerifiedGroup, VerifiedGroups, VerifiedStatement,
};
use zkie_engine::pcs::request::foldeq::{reader_table_identity, OmegaShapePlan};
use zkie_engine::pcs::request::gu::{
    GuNotice, GuOracleShape, GuRegistry, GuShape,
};
use zkie_engine::pcs::request::lin::LinMask;
use zkie_engine::pcs::request::params::{
    FINAL_CHAIN_LABEL, LEAF_SALT_MAX_BYTES, LEAF_SALT_MIN_BYTES, SHARED_CHAIN_LABEL,
};
use zkie_engine::pcs::request::pow::{search_nonce_for_plan, verify_nonce};
use zkie_engine::pcs::request::proximity::SingleRows;
use zkie_engine::pcs::request::read::unique_sorted;
use zkie_engine::pcs::request::relation::{
    column_row, foldeq_weights, gu_sample_pad_offsets, gu_table, oracle_sample_row, record_row,
    sample_cube_of, sample_row, GuRelationKind, GuRelationRow, GuWitness,
};
use zkie_engine::pcs::request::runtime_tree::{
    decode_runtime_opening, encode_runtime_opening, verify_runtime_opening, MaskTree,
    MaskTreePlan, RuntimePartRows, RuntimeTree, RuntimeLeafColumn, RuntimeTreeOpening, RuntimeTreePart, RuntimeTreePlan,
};
use zkie_engine::pcs::request::table::{
    record_seeds, validate_index_shape, verify_columns, ColumnOpening, IndexReader, IndexShape,
    SampleSeed, TableRowSource,
};
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};
use zkie_engine::primitives::transcript::{
    hash_labeled, tag_word, LEAF_MAIN, PARAMETERS,
};

use crate::proof::psc::{
    check_product_claims, hp_entries, psc_variables, ClaimMask, DotClaim, DotMaskEntries,
    DotOutput, DotProof, PscChallenges, PscRound, DOT_VARIABLES, PLAIN_DEGREE, SHOWN_MAPS,
};
use crate::proof::output_closure::{
    commit_closure_mask, commit_value_group, closure_values_from_rows, prove_closure_second,
    verify_closure_second, ClosureFirstMessages, ClosureMaskMessage, ClosureMaskPlan, ClosurePlan,
    ClosureRandomness, ClosureRowBinding, ClosureSecondMessages, ClosureWitness,
    CommittedClosureMask, CommittedGroupMessage, CommittedValueGroup, RetainedTrees,
};
use crate::registry::{RSQRT_COARSE, RSQRT_COARSE_END, RSQRT_FINE, RSQRT_HALF, RSQRT_SHIFT};
use crate::proof::hidden_norm_psc::{
    NormFirstMessages, NormGroupPlan, NormGroupWitness, NormLookupMessage, NormLookupSecond,
    NormMaskMessage, NormMaskPlan, NormOutputMessage, NormParamClaimMessage,
    NormParamMessage, NormPlan, NormPositionBinding, NormPscSecond,
    NormRowBinding, NormWitness, RSQRT_FINE_SLACK_WIDTH, RSQRT_INDEX_SLACK_WIDTH,
    RSQRT_MID_HIGH_WIDTH, RSQRT_RAW_REMAINDER_WIDTH,
};
use crate::proof::committed_psc::CommittedPscGroup;
use crate::proof::norm_param_table::{NormParamClaimPlan, NormParamKind, NormParamTableIdentity};
use crate::proof::base_final_session::{
    fold_identity, BaseFinalCheck, BaseFinalOutcome, BaseFinalRows, BaseFinalSession, BaseMessages,
    BaseCommitment, BaseOpening, BaseOpenings, BaseProver, FoldBasis, FoldMember,
};
use crate::proof::base_last_session::LastBaseRandomness;
use crate::proof::fold_handoff::{FoldHandoff, FoldHandoffAnchor};
use crate::proof::multi_stage_iopp::{
    prove_multi_stage_iopp, verify_multi_stage_iopp_with_last_positions, IoppBasePrivate,
    IoppChain, IoppProof, IoppStageFoldPrivate, IoppStageZero, IoppSwitchPrivate,
    IoppVerifyStageZero,
};
use crate::proof::proximity_iopp::{
    proximity_iopp_plan, ProximityGroupRole, ProximityIoppPlan,
};

/// Version of the canonical messages of this interface. Version 4 carries the
/// public Dot residual, the independent claim/tilde mask draws, the one
/// difference row that replaces the per-position equality rows and, when the
/// trusted plan selects it, the presence-flagged output-closure block (the
/// slack, packed-bit, product and mask messages and their samples).
const HIDDEN_SEND_SCHEMA_VERSION: u64 = 4;

/// The plan tag appended to the send statement when the Dot activation is the
/// norm hidden-output object. The tag is absent on the independently
/// committed activation path, whose statement bytes stay unchanged.
const HD_ACTIVATION_MODE_TAG: u64 = 0x4E4F_524D_4844;

/// The plan statement root of the norm hidden-output mode. The committed
/// object itself is bound by the ordered roots in the first-stage norm
/// message; this value only marks the mode and is never a second commitment.
pub const HD_ACTIVATION_ROOT: Digest = Digest([0x48; 32]);

/// The plan statement configuration of the norm hidden-output mode.
pub const HD_ACTIVATION_CONFIGURATION: Digest = Digest([0x47; 32]);

/// Version of the derived hidden-output object identity.
const HD_ACTIVATION_IDENTITY_VERSION: u64 = 1;
/// Domain tag of the public Dot residual in the second-stage transcript.
const DOT_RESIDUAL_MESSAGE_TAG: u64 = 19;
/// Version of the Dot residual relation. It fixes the sign convention
/// `R = sum_t chi^t (g_t - c_t)` and the directed adjusted target
/// `T = sum_t chi^t tilde_t - R`; a different convention is a different
/// relation and must use a new version.
const DOT_RESIDUAL_RELATION: u64 = 1;
/// Chain label of the hidden SEND-sample message.
const HIDDEN_SEND_CHAIN_LABEL: u64 = 0x4853_4e44;
/// The activation row tree's number in the runtime leaf domain. The value is
/// distinct from every runtime tree number the plan uses.
const ACTIVATION_LEAF_TREE: u64 = 0xE001;
/// Upper bound on a decoded activation row width or position count.
const ACTIVATION_DECODE_BOUND: usize = 1 << 20;
/// Upper bound on the number of sample coordinates a seed may carry.
const SEED_COORDINATE_BOUND: usize = 1 << 12;
/// Upper bound on the number of group entries.
const GROUP_ENTRY_BOUND: usize = 1 << 12;
/// Upper bound on a decoded proximity loop proof. The bound is on the encoded
/// artifact; the proof's internal counts are bounded again by its own decoder.
const PROXIMITY_LOOP_DECODE_BOUND: usize = 1 << 28;

// ---------------------------------------------------------------------------
// The activation matrix and its hiding row commitment
// ---------------------------------------------------------------------------

/// The shape of one activation matrix: one row per position and `hidden`
/// values per row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActivationShape {
    pub positions: u32,
    pub hidden: u32,
    pub salt_bytes: usize,
}

impl ActivationShape {
    pub fn new(positions: u32, hidden: u32, salt_bytes: usize) -> Result<Self> {
        let shape = Self {
            positions,
            hidden,
            salt_bytes,
        };
        if shape.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        Ok(shape)
    }

    /// What is wrong with the shape by itself. The row count is a power of two
    /// because the row commitment is a perfect binary tree.
    pub fn fault(&self) -> Option<&'static str> {
        if self.positions == 0 || !self.positions.is_power_of_two() || self.positions > (1 << 30) {
            return Some("the activation matrix has no power-of-two row count");
        }
        if self.hidden == 0 || self.hidden > (1 << 30) {
            return Some("an activation row has no bounded width");
        }
        if self.salt_bytes < LEAF_SALT_MIN_BYTES || self.salt_bytes > LEAF_SALT_MAX_BYTES {
            return Some("the activation leaves have no admitted salt");
        }
        if self.salt_bytes % 8 != 0 {
            return Some("the activation salt length is not a multiple of eight");
        }
        None
    }

    pub fn positions(&self) -> usize {
        self.positions as usize
    }

    pub fn hidden(&self) -> usize {
        self.hidden as usize
    }

    fn depth(&self) -> u32 {
        self.positions.trailing_zeros()
    }

    fn encode_into(&self, writer: &mut CanonicalWriter) {
        writer.put_u64(self.positions as u64);
        writer.put_u64(self.hidden as u64);
        writer.put_u64(self.salt_bytes as u64);
    }

    fn decode_from(reader: &mut CanonicalReader<'_>) -> Result<Self> {
        let positions = u32::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        let hidden = u32::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        let salt_bytes = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        let shape = Self {
            positions,
            hidden,
            salt_bytes,
        };
        if shape.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        Ok(shape)
    }
}

/// The committed object that supplies the hidden Dot activation rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivationSource {
    /// The activation is committed independently and pinned by the plan.
    Commitment,
    /// The activation is the norm hidden-output object. The Dot reads the
    /// committed norm output messages; the plan carries no separate
    /// activation commitment.
    NormHd,
}

impl ActivationSource {
    /// Appends the mode tag to a statement or parameter digest. The default
    /// path appends nothing so its statement bytes are unchanged.
    fn append(self, writer: &mut CanonicalWriter) {
        if self == Self::NormHd {
            writer.put_u64(HD_ACTIVATION_MODE_TAG);
        }
    }

    pub fn is_norm_hd(self) -> bool {
        self == Self::NormHd
    }
}

/// The public commitment of the activation matrix: the row-tree root, the
/// shape and the configuration digest the leaves start from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActivationCommitment {
    pub root: Digest,
    pub shape: ActivationShape,
    pub configuration: Digest,
}

impl ActivationCommitment {
    pub fn fault(&self) -> Option<&'static str> {
        if self.shape.fault().is_some() {
            return Some("the activation commitment has no valid shape");
        }
        if self.root == Digest::ZERO {
            return Some("the activation commitment has no root");
        }
        if self.configuration == Digest::ZERO {
            return Some("the activation commitment has no configuration");
        }
        None
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(HIDDEN_SEND_SCHEMA_VERSION);
        self.shape.encode_into(&mut writer);
        writer.put_digest(&self.configuration);
        writer.put_digest(&self.root);
        writer.into_bytes()
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        if reader.get_u64()? != HIDDEN_SEND_SCHEMA_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let shape = ActivationShape::decode_from(&mut reader)?;
        let configuration = reader.get_digest()?;
        let root = reader.get_digest()?;
        reader.finish()?;
        let commitment = Self {
            root,
            shape,
            configuration,
        };
        if commitment.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        Ok(commitment)
    }
}

/// One opened activation row: the row values, its salt and the Merkle path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActivationOpening {
    pub position: u32,
    pub values: Vec<Fp3>,
    pub salt: Vec<u8>,
    pub path: Vec<Digest>,
}

impl ActivationOpening {
    pub fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(HIDDEN_SEND_SCHEMA_VERSION);
        writer.put_u64(self.position as u64);
        writer.put_u64(self.values.len() as u64);
        for value in &self.values {
            writer.put_fp3(value);
        }
        writer.put_bytes(&self.salt);
        writer.put_u64(self.path.len() as u64);
        for digest in &self.path {
            writer.put_digest(digest);
        }
        writer.into_bytes()
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        if reader.get_u64()? != HIDDEN_SEND_SCHEMA_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let position = u32::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        let values = read_fp3_vec(&mut reader, ACTIVATION_DECODE_BOUND)?;
        let salt = reader.get_bytes()?.to_vec();
        if salt.len() < LEAF_SALT_MIN_BYTES || salt.len() > LEAF_SALT_MAX_BYTES {
            return Err(Error::InvalidEncoding);
        }
        let path = read_digest_vec(&mut reader, ACTIVATION_DECODE_BOUND)?;
        reader.finish()?;
        Ok(Self {
            position,
            values,
            salt,
            path,
        })
    }
}

/// The prover's activation witness: the private rows, their salts and the
/// public commitment. The rows are never serialized into a proof message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommittedActivation {
    commitment: ActivationCommitment,
    rows: Vec<Vec<Fp3>>,
    salts: Vec<Vec<u8>>,
    leaves: Vec<Digest>,
}

impl CommittedActivation {
    /// Commits field-valued rows. Every row needs its own salt; the caller
    /// supplies fresh random salts in a production run.
    pub fn commit(
        shape: ActivationShape,
        rows: &[Vec<Fp3>],
        salts: &[Vec<u8>],
        configuration: Digest,
    ) -> Result<Self> {
        if shape.fault().is_some() || configuration == Digest::ZERO {
            return Err(Error::InvalidConfig);
        }
        let positions = shape.positions();
        let hidden = shape.hidden();
        if rows.len() != positions
            || rows.iter().any(|row| row.len() != hidden)
            || salts.len() != positions
            || salts.iter().any(|salt| salt.len() != shape.salt_bytes)
        {
            return Err(Error::ShapeMismatch);
        }
        let mut leaves = Vec::with_capacity(positions);
        for (position, (row, salt)) in rows.iter().zip(salts).enumerate() {
            leaves.push(activation_leaf(
                &shape,
                &configuration,
                position as u32,
                row,
                salt,
            )?);
        }
        let tree = merkle_build(&configuration, &leaves)?;
        let commitment = ActivationCommitment {
            root: tree.root,
            shape,
            configuration,
        };
        if commitment.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            commitment,
            rows: rows.to_vec(),
            salts: salts.to_vec(),
            leaves,
        })
    }

    /// Commits signed words. A negative word is the negative field element,
    /// never its unsigned reduction.
    pub fn commit_signed(
        shape: ActivationShape,
        rows: &[Vec<i64>],
        salts: &[Vec<u8>],
        configuration: Digest,
    ) -> Result<Self> {
        let mut field_rows = Vec::with_capacity(rows.len());
        for row in rows {
            let mut field_row = Vec::with_capacity(row.len());
            for value in row {
                field_row.push(Fp3::from_base(crate::proof::column_session::fp_from_signed_word(*value as u64)));
            }
            field_rows.push(field_row);
        }
        Self::commit(shape, &field_rows, salts, configuration)
    }

    pub fn commitment(&self) -> &ActivationCommitment {
        &self.commitment
    }

    pub fn root(&self) -> Digest {
        self.commitment.root
    }

    pub fn shape(&self) -> ActivationShape {
        self.commitment.shape
    }

    pub fn rows(&self) -> &[Vec<Fp3>] {
        &self.rows
    }

    pub fn salts(&self) -> &[Vec<u8>] {
        &self.salts
    }

    pub fn leaves(&self) -> &[Digest] {
        &self.leaves
    }

    /// One activation row. A caller that reads it must first decide whether the
    /// point at which it reads is public.
    pub fn row(&self, position: u32) -> Result<&[Fp3]> {
        if position >= self.commitment.shape.positions {
            return Err(Error::InvalidIndex);
        }
        self.rows
            .get(position as usize)
            .map(Vec::as_slice)
            .ok_or(Error::InvalidIndex)
    }

    /// The row-weight vector of one position, cut or padded to the matrix row
    /// count. Entries at or above the activation width are zero.
    pub fn weights(&self, position: u32, matrix_rows: u64) -> Result<Vec<Fp3>> {
        let matrix_rows = usize::try_from(matrix_rows).map_err(|_| Error::ResourceLimit)?;
        let row = self.row(position)?;
        if row.len() > matrix_rows {
            return Err(Error::ShapeMismatch);
        }
        let mut weights = vec![Fp3::ZERO; matrix_rows];
        weights[..row.len()].copy_from_slice(row);
        Ok(weights)
    }

    /// Opens one row with its path. The opening is public and reveals the row;
    /// it must not be used to bind the row at a point drawn after the
    /// commitment unless the protocol calls for that reading.
    pub fn open(&self, position: u32) -> Result<ActivationOpening> {
        let row = self.row(position)?;
        let tree = merkle_build(&self.commitment.configuration, &self.leaves)?;
        let path = tree.open(position as usize)?;
        Ok(ActivationOpening {
            position,
            values: row.to_vec(),
            salt: self
                .salts
                .get(position as usize)
                .cloned()
                .ok_or(Error::InvalidIndex)?,
            path,
        })
    }

    pub fn recompute_root(&self) -> Result<Digest> {
        let mut leaves = Vec::with_capacity(self.rows.len());
        for (position, (row, salt)) in self.rows.iter().zip(&self.salts).enumerate() {
            leaves.push(activation_leaf(
                &self.commitment.shape,
                &self.commitment.configuration,
                position as u32,
                row,
                salt,
            )?);
        }
        Ok(merkle_build(&self.commitment.configuration, &leaves)?.root)
    }
}

/// Checks an activation opening against its commitment. A changed value, salt
/// or path is rejected; the salt length and the row width must match the
/// trusted shape.
pub fn verify_activation_opening(
    commitment: &ActivationCommitment,
    opening: &ActivationOpening,
) -> Result<()> {
    if commitment.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    let shape = commitment.shape;
    if opening.position >= shape.positions
        || opening.values.len() != shape.hidden()
        || opening.salt.len() != shape.salt_bytes
        || opening.path.len() != shape.depth() as usize
    {
        return Err(Error::ShapeMismatch);
    }
    let leaf = activation_leaf(
        &shape,
        &commitment.configuration,
        opening.position,
        &opening.values,
        &opening.salt,
    )?;
    if !merkle_verify(
        &commitment.configuration,
        &commitment.root,
        opening.position as usize,
        &leaf,
        &opening.path,
    )? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn activation_leaf(
    shape: &ActivationShape,
    configuration: &Digest,
    position: u32,
    values: &[Fp3],
    salt: &[u8],
) -> Result<Digest> {
    if shape.fault().is_some()
        || position >= shape.positions
        || values.len() != shape.hidden()
        || salt.len() != shape.salt_bytes
    {
        return Err(Error::ShapeMismatch);
    }
    let mut writer = CanonicalWriter::new();
    writer.put_u64(shape.positions as u64);
    writer.put_u64(shape.hidden as u64);
    writer.put_u64(position as u64);
    for value in values {
        writer.put_fp3(value);
    }
    writer.put_bytes(salt);
    hash_labeled(
        tag_word(LEAF_MAIN, ACTIVATION_LEAF_TREE),
        position as u64,
        Some(configuration),
        &writer.into_bytes(),
    )
}

// ---------------------------------------------------------------------------
// The W message shape and the group
// ---------------------------------------------------------------------------

/// The shape of one `W` oracle message. The message holds `columns` entries in
/// `main_rows` rows of the code's `ell` places, with the sample pads reserved at
/// the end of the last row. The entry-to-place map is the sample cube's map:
/// entry `j` is row `j mod main_rows` at place `j div main_rows`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WShape {
    pub code: CodeParams,
    pub main_rows: u64,
    pub answers: u32,
    pub salt_bytes: usize,
    pub tree_number: u64,
    pub leaf_class: u32,
    pub run: Digest,
    pub configuration: Digest,
}

impl WShape {
    /// Derives a shape for a message of `columns` entries. The row count is the
    /// smallest power of two that leaves the last row's sample pads free.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        columns: u64,
        code: CodeParams,
        answers: u32,
        positions: u32,
        salt_bytes: usize,
        tree_number: u64,
        leaf_class: u32,
        run: Digest,
        configuration: Digest,
    ) -> Result<Self> {
        if columns == 0 || !columns.is_power_of_two() {
            return Err(Error::InvalidConfig);
        }
        build_code_plan(code.log_n, code.ell, code.r)?;
        if answers == 0 || answers as u64 > code.ell {
            return Err(Error::InvalidConfig);
        }
        let capacity = code.ell - answers as u64;
        if capacity == 0 {
            return Err(Error::InvalidConfig);
        }
        let rows_needed = columns
            .checked_add(capacity - 1)
            .ok_or(Error::ResourceLimit)?
            / capacity;
        let main_rows = rows_needed
            .checked_next_power_of_two()
            .ok_or(Error::ResourceLimit)?;
        let shape = Self {
            code,
            main_rows,
            answers,
            salt_bytes,
            tree_number,
            leaf_class,
            run,
            configuration,
        };
        if shape.fault(columns, positions).is_some() {
            return Err(Error::InvalidConfig);
        }
        Ok(shape)
    }

    /// What is wrong with the shape for a message of `columns` entries and
    /// `positions` trees.
    pub fn fault(&self, columns: u64, positions: u32) -> Option<&'static str> {
        if build_code_plan(self.code.log_n, self.code.ell, self.code.r).is_err() {
            return Some("the W oracle code is not a code");
        }
        if columns == 0 || !columns.is_power_of_two() {
            return Some("the W message has no power-of-two width");
        }
        if self.main_rows == 0 || !self.main_rows.is_power_of_two() || self.main_rows > (1 << 30) {
            return Some("the W message has no power-of-two row count");
        }
        if self.answers == 0 || self.answers as u64 > self.code.ell {
            return Some("the W message has no room for its sample answers");
        }
        let capacity = self.code.ell - self.answers as u64;
        let room = self
            .main_rows
            .checked_mul(self.code.ell)
            .ok_or(Error::ResourceLimit);
        match room {
            Ok(room) if room >= columns => {}
            Ok(_) => return Some("the W message is shorter than its column count"),
            Err(_) => return Some("the W message width overflows"),
        }
        let last_place = (columns - 1) / self.main_rows;
        if last_place >= capacity {
            return Some("a W entry falls on a sample pad");
        }
        if self.salt_bytes < LEAF_SALT_MIN_BYTES || self.salt_bytes > LEAF_SALT_MAX_BYTES {
            return Some("the W tree leaves have no admitted salt");
        }
        if self.salt_bytes % 8 != 0 {
            return Some("the W tree salt length is not a multiple of eight");
        }
        if self.run == Digest::ZERO || self.configuration == Digest::ZERO {
            return Some("the W tree has no run or configuration start");
        }
        if positions == 0 {
            return Some("the W group has no position");
        }
        match self.tree_number.checked_add(positions as u64) {
            Some(end) if end <= (1u64 << 16) => {}
            _ => return Some("a W tree number does not fit the tag"),
        }
        None
    }

    /// The oracle shape of one `W` message: main rows over the extension field,
    /// no mask rows, one answer per shared seed.
    pub fn oracle_shape(&self) -> GuOracleShape {
        GuOracleShape {
            code: self.code,
            main_rows: self.main_rows,
            main_deg: E_DEGREE as u32,
            mask_rows: 0,
            mask_code: None,
            answers: self.answers,
        }
    }

    /// The runtime tree of one position's `W` message.
    pub fn tree_plan(&self, position: u32) -> Result<RuntimeTreePlan> {
        let main_symbols = self
            .main_rows
            .checked_mul(E_DEGREE as u64)
            .ok_or(Error::ResourceLimit)?;
        let tree_number = self
            .tree_number
            .checked_add(position as u64)
            .ok_or(Error::ResourceLimit)?;
        if tree_number >= (1u64 << 16) {
            return Err(Error::InvalidConfig);
        }
        Ok(RuntimeTreePlan {
            tree_number,
            leaf_class: self.leaf_class,
            log_n: self.code.log_n,
            log_n_zk: self.code.log_n,
            parts: vec![RuntimeTreePart {
                main_symbols,
                mask_symbols: 0,
            }],
            salt_bytes: self.salt_bytes,
            run: self.run,
            configuration: self.configuration,
        })
    }

    /// The number of seed coordinates the sample form reads.
    pub fn sample_bits(&self) -> Result<u32> {
        let shape = self.oracle_shape();
        if shape.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        Ok(sample_cube_of(&shape)?.bits)
    }

    pub(crate) fn encode_into(&self, writer: &mut CanonicalWriter) {
        writer.put_u64(self.code.log_n as u64);
        writer.put_u64(self.code.ell);
        writer.put_u64(self.code.r);
        writer.put_u64(self.main_rows);
        writer.put_u64(self.answers as u64);
        writer.put_u64(self.salt_bytes as u64);
        writer.put_u64(self.tree_number);
        writer.put_u64(self.leaf_class as u64);
        writer.put_digest(&self.run);
        writer.put_digest(&self.configuration);
    }
}

/// The trusted plan of the committed shown-map message of the hidden Dot
/// identity. One main row block holds the activation rows: entry
/// `t * hidden + j` is value `j` of position `t`, placed with the same
/// entry-to-place map as a `W` message. The message is maskless, has one
/// sample answer and lives in the own proximity group, so the `(E_j)` row on
/// the batched activation map is checked by the same BAT/FOLD/BASE test as
/// the reduced rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DotMapPlan {
    /// The oracle id of the shown-map message.
    pub oracle: u64,
    /// The SEND group the message belongs to.
    pub group: u64,
    /// The message shape: one row block over the group's main code.
    pub shape: WShape,
    /// Entries per position.
    pub hidden: u32,
}

impl DotMapPlan {
    /// The number of message entries: one row of `hidden` values per position.
    pub fn columns(&self, positions: u32) -> Result<u64> {
        if positions == 0 || self.hidden == 0 {
            return Err(Error::InvalidConfig);
        }
        (positions as u64)
            .checked_mul(self.hidden as u64)
            .ok_or(Error::ResourceLimit)
    }

    /// The runtime tree of the shown-map message. The message uses the first
    /// tree number of the shape; the caller keeps that number distinct.
    pub fn tree_plan(&self) -> Result<RuntimeTreePlan> {
        self.shape.tree_plan(0)
    }

    /// The oracle shape of the shown-map message.
    pub fn oracle_shape(&self) -> GuOracleShape {
        self.shape.oracle_shape()
    }

    /// The number of seed coordinates the sample form reads.
    pub fn sample_bits(&self) -> Result<u32> {
        self.shape.sample_bits()
    }

    /// What is wrong with the shape for the `positions` positions and the
    /// `matrix_rows` hidden entries of one position.
    pub fn fault(&self, positions: u32, matrix_rows: u64) -> Option<&'static str> {
        if self.oracle == 0 || self.group == 0 {
            return Some("the shown-map message has no identity");
        }
        if self.hidden == 0 || self.hidden as u64 != matrix_rows {
            return Some("the shown-map message does not carry the hidden entry count");
        }
        if self.shape.answers != 1 {
            return Some("the shown-map message does not have one answer");
        }
        let columns = match self.columns(positions) {
            Ok(columns) => columns,
            Err(_) => return Some("the shown-map message width overflows"),
        };
        if self.shape.fault(columns, positions).is_some() {
            return Some("the shown-map message shape is not a message shape");
        }
        None
    }

    fn encode_into(&self, writer: &mut CanonicalWriter) {
        writer.put_u64(self.oracle);
        writer.put_u64(self.group);
        writer.put_u64(self.hidden as u64);
        self.shape.encode_into(writer);
    }
}

/// The prover's private shown-map message: the unencoded rows, their encoding
/// tails, the tree salts and the committed root. The rows and salts are never
/// serialized.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DotMapWitness {
    pub message: Vec<Vec<Fp3>>,
    pub tails: Vec<Vec<Fp3>>,
    pub salts: Vec<Vec<u8>>,
    pub pad: Fp3,
    pub root: Digest,
}

impl DotMapWitness {
    fn check(&self, plan: &DotMapPlan) -> Result<()> {
        let rows = usize::try_from(plan.shape.main_rows).map_err(|_| Error::ResourceLimit)?;
        let places = usize::try_from(plan.shape.code.ell).map_err(|_| Error::ResourceLimit)?;
        let tail = usize::try_from(plan.shape.code.r).map_err(|_| Error::ResourceLimit)?;
        let columns = usize::try_from(plan.shape.code.n()).map_err(|_| Error::ResourceLimit)?;
        if self.message.len() != rows
            || self.message.iter().any(|row| row.len() != places)
            || self.tails.len() != rows
            || self.tails.iter().any(|row| row.len() != tail)
            || self.salts.len() != columns
            || self.salts.iter().any(|salt| salt.len() != plan.shape.salt_bytes)
            || self.pad == Fp3::ZERO
            || self.root == Digest::ZERO
        {
            return Err(Error::ShapeMismatch);
        }
        let last = self
            .message
            .get(rows - 1)
            .and_then(|row| row.get(places - 1))
            .ok_or(Error::ShapeMismatch)?;
        if *last != self.pad {
            return Err(Error::StatementMismatch);
        }
        if *self.tree(plan)?.root() != self.root {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }

    fn tree(&self, plan: &DotMapPlan) -> Result<RuntimeTree> {
        commit_rows_tree(
            plan.tree_plan()?,
            plan.shape.code,
            &self.message,
            &self.tails,
            &self.salts,
        )
    }
}

/// One registered `W` oracle: its position, identity, shape, tree and root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SendGroupEntry {
    pub position: u32,
    pub oracle: u64,
    pub shape: GuOracleShape,
    pub tree: RuntimeTreePlan,
    pub root: Digest,
}

/// The group registration of the `W` oracle messages. The group joins the
/// SEND samples and the Eval rows; the ordered roots are the SEND statement of
/// this message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SendGroup {
    pub group: u64,
    pub entries: Vec<SendGroupEntry>,
}

impl SendGroup {
    pub fn fault(&self, positions: u32) -> Option<&'static str> {
        if self.entries.len() != positions as usize || self.entries.is_empty() {
            return Some("the W group does not have one entry per position");
        }
        let mut last_oracle = None;
        for (index, entry) in self.entries.iter().enumerate() {
            if entry.position != index as u32 {
                return Some("the W group positions are not in order");
            }
            if entry.shape.fault().is_some() {
                return Some("a W group member is not an oracle shape");
            }
            if entry.root == Digest::ZERO {
                return Some("a W group member has no root");
            }
            if entry.tree.fault().is_some() {
                return Some("a W group member has no runtime tree");
            }
            if let Some(previous) = last_oracle {
                if entry.oracle <= previous {
                    return Some("the W group oracle ids are not increasing");
                }
            }
            last_oracle = Some(entry.oracle);
        }
        None
    }

    pub fn root(&self, position: u32) -> Result<Digest> {
        self.entries
            .get(position as usize)
            .map(|entry| entry.root)
            .ok_or(Error::InvalidIndex)
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(HIDDEN_SEND_SCHEMA_VERSION);
        writer.put_u64(self.group);
        writer.put_u64(self.entries.len() as u64);
        for entry in &self.entries {
            writer.put_u64(entry.position as u64);
            writer.put_u64(entry.oracle);
            put_oracle_shape(&mut writer, &entry.shape);
            put_tree_plan(&mut writer, &entry.tree);
            writer.put_digest(&entry.root);
        }
        writer.into_bytes()
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        if reader.get_u64()? != HIDDEN_SEND_SCHEMA_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let group = reader.get_u64()?;
        let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if count == 0 || count > GROUP_ENTRY_BOUND {
            return Err(Error::InvalidEncoding);
        }
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            let position = u32::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
            let oracle = reader.get_u64()?;
            let shape = get_oracle_shape(&mut reader)?;
            let tree = get_tree_plan(&mut reader)?;
            let root = reader.get_digest()?;
            entries.push(SendGroupEntry {
                position,
                oracle,
                shape,
                tree,
                root,
            });
        }
        reader.finish()?;
        Ok(Self { group, entries })
    }
}

/// One SEND sample message: the shared seed set drawn after the roots and the
/// complete ordered answers. The answers are public; the sampled rows are not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SendSamples {
    pub seed: Vec<SampleSeed>,
    pub answers: Vec<Fp3>,
}

impl SendSamples {
    pub fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(HIDDEN_SEND_SCHEMA_VERSION);
        writer.put_u64(self.seed.len() as u64);
        for seed in &self.seed {
            writer.put_u64(seed.z.len() as u64);
            for coordinate in &seed.z {
                writer.put_fp3(coordinate);
            }
            writer.put_fp3(&seed.beta);
        }
        writer.put_u64(self.answers.len() as u64);
        for answer in &self.answers {
            writer.put_fp3(answer);
        }
        writer.into_bytes()
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        if reader.get_u64()? != HIDDEN_SEND_SCHEMA_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let seed_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if seed_count == 0 || seed_count > SEED_COORDINATE_BOUND {
            return Err(Error::InvalidEncoding);
        }
        let mut seed = Vec::with_capacity(seed_count);
        for _ in 0..seed_count {
            let coordinate_count =
                usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
            if coordinate_count > SEED_COORDINATE_BOUND {
                return Err(Error::InvalidEncoding);
            }
            let mut z = Vec::with_capacity(coordinate_count);
            for _ in 0..coordinate_count {
                z.push(reader.get_fp3()?);
            }
            let beta = reader.get_fp3()?;
            if beta == Fp3::ZERO {
                return Err(Error::InvalidEncoding);
            }
            seed.push(SampleSeed { z, beta });
        }
        let answer_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if answer_count > SEED_COORDINATE_BOUND {
            return Err(Error::InvalidEncoding);
        }
        let mut answers = Vec::with_capacity(answer_count);
        for _ in 0..answer_count {
            answers.push(reader.get_fp3()?);
        }
        reader.finish()?;
        Ok(Self { seed, answers })
    }
}

// ---------------------------------------------------------------------------
// The one-time mask schema
// ---------------------------------------------------------------------------

fn ceil_log2_u64(value: u64) -> u32 {
    if value <= 1 {
        0
    } else {
        64 - (value - 1).leading_zeros()
    }
}

/// The fixed allocation of the one-time mask entries in the index-mask
/// message. The five classes occupy disjoint ranges in this order: the PSC
/// hiding coefficients, the two shown-value masks, the claim masks, the tilde
/// masks and the Eval masks. The claim and tilde entries are never merged and
/// carry independent draws: `tilde_t` reads a `tilde` entry and the claim
/// reads a `claim` entry; one public difference row ties their chi-weighted
/// sums to the published residual without forcing the values to be equal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskEntryAllocation {
    /// The number of positions; the claim, tilde and Eval classes have one
    /// entry per position.
    pub positions: u32,
    /// The number of sumcheck variables of the masked product sumcheck. The
    /// production shape fixes it to [`DOT_VARIABLES`]; the entry count of the
    /// hiding class follows from it, so a smaller test shape carries a
    /// smaller allocation and the 73-entry production schema is unchanged.
    pub variables: usize,
    /// The PSC hiding-polynomial coefficients.
    pub hiding: usize,
    /// The two shown-map end masks.
    pub end: usize,
    /// The claim masks `c_t`.
    pub claim: usize,
    /// The tilde masks `g_t`.
    pub tilde: usize,
    /// The Eval masks `e_t`.
    pub eval: usize,
}

impl MaskEntryAllocation {
    /// The 73-entry fallback allocation for `positions` positions.
    pub fn new(positions: u32) -> Result<Self> {
        Self::for_variables(DOT_VARIABLES, positions)
    }

    /// The separate-mask allocation for a sumcheck of `variables` variables
    /// and `positions` positions. Production uses [`Self::new`], whose
    /// variable count is [`DOT_VARIABLES`] and whose entry count is 73 for 16
    /// positions.
    pub fn for_variables(variables: usize, positions: u32) -> Result<Self> {
        if positions == 0 || variables == 0 {
            return Err(Error::InvalidConfig);
        }
        let variables_weights = psc_variables(variables)?;
        let count = positions as usize;
        let allocation = Self {
            positions,
            variables,
            hiding: hp_entries(&variables_weights)?,
            end: SHOWN_MAPS,
            claim: count,
            tilde: count,
            eval: count,
        };
        allocation.check()?;
        Ok(allocation)
    }

    /// Checks the class counts against the PSC sizes and the position count.
    pub fn check(&self) -> Result<()> {
        let variables = psc_variables(self.variables)?;
        if self.positions == 0
            || self.variables == 0
            || self.hiding != hp_entries(&variables)?
            || self.end != SHOWN_MAPS
            || self.claim != self.positions as usize
            || self.tilde != self.positions as usize
            || self.eval != self.positions as usize
        {
            return Err(Error::InvalidConfig);
        }
        if self.total()? == 0 {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }

    /// The total number of allocated entries.
    pub fn total(&self) -> Result<usize> {
        let mut total = self.hiding;
        for count in [self.end, self.claim, self.tilde, self.eval] {
            total = total.checked_add(count).ok_or(Error::ResourceLimit)?;
        }
        Ok(total)
    }

    fn class_range(&self, before: &[usize], count: usize) -> Result<core::ops::Range<usize>> {
        let mut first = 0usize;
        for value in before {
            first = first.checked_add(*value).ok_or(Error::ResourceLimit)?;
        }
        let end = first.checked_add(count).ok_or(Error::ResourceLimit)?;
        Ok(first..end)
    }

    pub fn hiding_range(&self) -> Result<core::ops::Range<usize>> {
        self.class_range(&[], self.hiding)
    }

    pub fn end_range(&self) -> Result<core::ops::Range<usize>> {
        self.class_range(&[self.hiding], self.end)
    }

    pub fn claim_range(&self) -> Result<core::ops::Range<usize>> {
        self.class_range(&[self.hiding, self.end], self.claim)
    }

    pub fn tilde_range(&self) -> Result<core::ops::Range<usize>> {
        self.class_range(&[self.hiding, self.end, self.claim], self.tilde)
    }

    pub fn eval_range(&self) -> Result<core::ops::Range<usize>> {
        self.class_range(&[self.hiding, self.end, self.claim, self.tilde], self.eval)
    }

    fn position_slot(&self, first: usize, count: usize, position: u32) -> Result<usize> {
        if position >= self.positions || position as usize >= count {
            return Err(Error::InvalidIndex);
        }
        first
            .checked_add(position as usize)
            .ok_or(Error::ResourceLimit)
    }

    pub fn claim_slot(&self, position: u32) -> Result<usize> {
        self.position_slot(self.claim_range()?.start, self.claim, position)
    }

    pub fn tilde_slot(&self, position: u32) -> Result<usize> {
        self.position_slot(self.tilde_range()?.start, self.tilde, position)
    }

    pub fn eval_slot(&self, position: u32) -> Result<usize> {
        self.position_slot(self.eval_range()?.start, self.eval, position)
    }

    /// The exact entry count of the separate-mask schema.
    pub fn is_separate_mask(&self) -> bool {
        self.claim == self.tilde && self.tilde == self.eval
    }
}

/// The trusted plan of the sampled index-mask message. It is a mask-only
/// message: no main rows, the group's mask code and one answer. Its mask rows
/// carry the one-time entries, the SEND pad sits at the last entry of the last
/// real row, and the runtime tree commits the encoded rows before any
/// challenge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexMaskPlan {
    pub oracle: u64,
    pub group: u64,
    pub mask_code: CodeParams,
    pub mask_rows: u64,
    pub answers: u32,
    pub openings: usize,
    pub tree: RuntimeTreePlan,
    pub entries: MaskEntryAllocation,
}

impl IndexMaskPlan {
    /// The number of seed coordinates the mask sample form reads.
    pub fn sample_bits(&self) -> Result<u32> {
        ceil_log2_u64(self.mask_rows)
            .checked_add(self.mask_code.log_ell())
            .ok_or(Error::ResourceLimit)
    }

    /// The number of rows of the fresh fold-mask table: the next power of two
    /// at or above the group's mask rows.
    pub fn tree_rows(&self) -> Result<u64> {
        let shift = ceil_log2_u64(self.mask_rows);
        1u64.checked_shl(shift).ok_or(Error::ResourceLimit)
    }

    /// The free entries of the mask rows, excluding the SEND pads.
    pub fn free_entries(&self) -> Result<usize> {
        let rows = usize::try_from(self.mask_rows).map_err(|_| Error::ResourceLimit)?;
        let ell = usize::try_from(self.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        let answers = self.answers as usize;
        if answers == 0 || answers > ell {
            return Err(Error::InvalidConfig);
        }
        let per_row = ell.checked_sub(answers).ok_or(Error::ResourceLimit)?;
        rows.checked_mul(per_row)
            .and_then(|value| value.checked_add(rows.saturating_sub(1)))
            .ok_or(Error::ResourceLimit)
    }

    /// The compact row-major offset of an allocated entry.
    pub fn entry_offset(&self, slot: usize) -> Result<usize> {
        let rows = usize::try_from(self.mask_rows).map_err(|_| Error::ResourceLimit)?;
        let ell = usize::try_from(self.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        let length = rows.checked_mul(ell).ok_or(Error::ResourceLimit)?;
        if slot >= length {
            return Err(Error::InvalidIndex);
        }
        Ok(slot)
    }

    /// The BAT virtual table index of an allocated entry: `row + place *
    /// 2^row_bits`, which is the layout the group mask table uses.
    pub fn entry_virtual_index(&self, slot: usize) -> Result<u64> {
        let offset = self.entry_offset(slot)?;
        let ell = usize::try_from(self.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        let row = offset / ell;
        let place = offset % ell;
        let row_bits = ceil_log2_u64(self.mask_rows);
        (row as u64)
            .checked_add((place as u64) << row_bits)
            .ok_or(Error::ResourceLimit)
    }

    /// The compact offset of the SEND pad: the last entry of the last real mask
    /// row. The pad is not an allocated entry.
    pub fn pad_offset(&self) -> Result<usize> {
        let rows = usize::try_from(self.mask_rows).map_err(|_| Error::ResourceLimit)?;
        let ell = usize::try_from(self.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        rows.checked_mul(ell)
            .and_then(|value| value.checked_sub(1))
            .ok_or(Error::ShapeMismatch)
    }

    /// The structural checks of the mask-only message plan.
    pub fn fault(&self) -> Option<&'static str> {
        if self.oracle == 0 || self.group == 0 {
            return Some("the index-mask message has no identity");
        }
        if build_code_plan(
            self.mask_code.log_n,
            self.mask_code.ell,
            self.mask_code.r,
        )
        .is_err()
        {
            return Some("the index-mask code is not a code");
        }
        if self.mask_rows == 0 {
            return Some("the index-mask message has no mask row");
        }
        if self.answers != 1 {
            return Some("the index-mask message does not have one answer");
        }
        if self.openings == 0 || self.openings as u64 > self.mask_code.r {
            return Some("the index-mask opening count exceeds its tail");
        }
        if self.tree.fault().is_some() {
            return Some("the index-mask tree is not a runtime tree");
        }
        if self.tree.log_n != self.mask_code.log_n
            || self.tree.log_n_zk != self.mask_code.log_n
            || self.tree.parts.len() != 1
        {
            return Some("the index-mask tree does not match its code");
        }
        let mask_symbols = match self.mask_rows.checked_mul(E_DEGREE as u64) {
            Some(value) => value,
            None => return Some("the index-mask row width overflows"),
        };
        if self.tree.parts[0].main_symbols != 0 || self.tree.parts[0].mask_symbols != mask_symbols
        {
            return Some("the index-mask tree does not carry its mask rows");
        }
        if self.entries.check().is_err() {
            return Some("the index-mask entry allocation is not the separate-mask schema");
        }
        match self.free_entries() {
            Ok(free) if free >= self.entries.total().unwrap_or(usize::MAX) => {}
            _ => return Some("the index-mask rows are shorter than the entry allocation"),
        }
        None
    }

    fn encode_into(&self, writer: &mut CanonicalWriter) {
        writer.put_u64(self.oracle);
        writer.put_u64(self.group);
        writer.put_u64(self.mask_code.log_n as u64);
        writer.put_u64(self.mask_code.ell);
        writer.put_u64(self.mask_code.r);
        writer.put_u64(self.mask_rows);
        writer.put_u64(self.answers as u64);
        writer.put_u64(self.openings as u64);
        writer.put_u64(self.tree.tree_number);
        writer.put_u64(self.tree.leaf_class as u64);
        writer.put_u64(self.tree.log_n as u64);
        writer.put_u64(self.tree.log_n_zk as u64);
        writer.put_u64(self.tree.parts.len() as u64);
        for part in &self.tree.parts {
            writer.put_u64(part.main_symbols);
            writer.put_u64(part.mask_symbols);
        }
        writer.put_u64(self.tree.salt_bytes as u64);
        writer.put_digest(&self.tree.run);
        writer.put_digest(&self.tree.configuration);
        writer.put_u64(self.entries.positions as u64);
        writer.put_u64(self.entries.variables as u64);
        writer.put_u64(self.entries.hiding as u64);
        writer.put_u64(self.entries.end as u64);
        writer.put_u64(self.entries.claim as u64);
        writer.put_u64(self.entries.tilde as u64);
        writer.put_u64(self.entries.eval as u64);
    }
}

/// The prover's private randomness of the index-mask message: the mask rows,
/// their encoding tails and one salt per tree column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexMaskRandomness {
    pub rows: Vec<Vec<Fp3>>,
    pub tails: Vec<Vec<Fp3>>,
    pub salts: Vec<Vec<u8>>,
}

impl IndexMaskRandomness {
    fn from_coins(coins: &mut ByteMaskCoins, plan: &IndexMaskPlan) -> Result<Self> {
        let rows = usize::try_from(plan.mask_rows).map_err(|_| Error::ResourceLimit)?;
        let ell = usize::try_from(plan.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        let tail = usize::try_from(plan.mask_code.r).map_err(|_| Error::ResourceLimit)?;
        let mut values = Vec::with_capacity(rows);
        for _ in 0..rows {
            values.push(draw_values(coins, ell)?);
        }
        // The SEND pad is the last entry of the last real mask row. It is
        // private, uniform and non-zero; a drawn zero is rejected and redrawn.
        let pad_row = rows.checked_sub(1).ok_or(Error::ShapeMismatch)?;
        let pad_place = ell.checked_sub(1).ok_or(Error::ShapeMismatch)?;
        for _ in 0..64 {
            if values[pad_row][pad_place] != Fp3::ZERO {
                break;
            }
            values[pad_row][pad_place] = coins.draw()?.0;
        }
        if values[pad_row][pad_place] == Fp3::ZERO {
            return Err(Error::EntropyUnavailable);
        }
        // The PSC claim mask `c_t` and the tilde mask `g_t` are distinct
        // one-time entries and keep independent draws from the stage-1
        // stream. The Dot layer publishes their chi-weighted residual and
        // checks it with one difference row; it does not copy the tilde value
        // into the claim slot and does not force the two values to be equal.
        // The allocation's disjoint slot ranges are checked by `fault`.
        let mut tails = Vec::with_capacity(rows);
        for _ in 0..rows {
            tails.push(draw_values(coins, tail)?);
        }
        let salts = draw_salt_bytes(
            coins,
            usize::try_from(plan.tree.columns()).map_err(|_| Error::ResourceLimit)?,
            plan.tree.salt_bytes,
        )?;
        let randomness = Self {
            rows: values,
            tails,
            salts,
        };
        randomness.check(plan)?;
        Ok(randomness)
    }

    fn check(&self, plan: &IndexMaskPlan) -> Result<()> {
        if plan.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        let rows = usize::try_from(plan.mask_rows).map_err(|_| Error::ResourceLimit)?;
        let ell = usize::try_from(plan.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        let tail = usize::try_from(plan.mask_code.r).map_err(|_| Error::ResourceLimit)?;
        let columns = usize::try_from(plan.tree.columns()).map_err(|_| Error::ResourceLimit)?;
        if self.rows.len() != rows
            || self.rows.iter().any(|row| row.len() != ell)
            || self.tails.len() != rows
            || self.tails.iter().any(|row| row.len() != tail)
            || self.salts.len() != columns
            || self.salts.iter().any(|salt| salt.len() != plan.tree.salt_bytes)
            || self.rows[rows - 1][ell - 1] == Fp3::ZERO
        {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }
}

/// The prover's private index-mask message: the compact mask rows, their
/// encoding tails, the tree salts and the committed root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexMaskWitness {
    pub rows: Vec<Vec<Fp3>>,
    pub tails: Vec<Vec<Fp3>>,
    pub salts: Vec<Vec<u8>>,
    pub root: Digest,
}

impl IndexMaskWitness {
    fn check(&self, plan: &IndexMaskPlan) -> Result<()> {
        if plan.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        let randomness = IndexMaskRandomness {
            rows: self.rows.clone(),
            tails: self.tails.clone(),
            salts: self.salts.clone(),
        };
        randomness.check(plan)?;
        if *self.tree(plan)?.root() != self.root {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }

    fn tree(&self, plan: &IndexMaskPlan) -> Result<RuntimeTree> {
        let mut codewords = Vec::with_capacity(self.rows.len());
        for (row, tail) in self.rows.iter().zip(&self.tails) {
            codewords.push(encode_row_fp3(&plan.mask_code, row, tail)?);
        }
        RuntimeTree::commit(
            plan.tree.clone(),
            &[RuntimePartRows {
                main: Vec::new(),
                mask: codewords,
                main_deg: 1,
            }],
            &self.salts,
        )
    }

    /// The value of one allocated mask entry.
    pub fn entry(&self, plan: &IndexMaskPlan, slot: usize) -> Result<Fp3> {
        let offset = plan.entry_offset(slot)?;
        let ell = usize::try_from(plan.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
        let row = self.rows.get(offset / ell).ok_or(Error::ShapeMismatch)?;
        row.get(offset % ell).copied().ok_or(Error::ShapeMismatch)
    }
}

/// One sampled index-mask message. The tree plan is carried and checked
/// against the trusted plan so the opening decodes without a second authority.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexMaskMessage {
    pub tree: RuntimeTreePlan,
    pub root: Digest,
    pub sample: SendSamples,
    pub positions: Vec<u64>,
    pub opening: RuntimeTreeOpening,
}

// ---------------------------------------------------------------------------
// The hidden SEND plan
// ---------------------------------------------------------------------------

/// The trusted plan of one hidden SEND-sample message. It carries the public
/// shapes, the encoded table identity, the activation commitment pin and, for
/// a verifier plan, the expected `W` roots. The expected roots are a fixture
/// expectation in this unit; in a production run they would be statement
/// values received under their own binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HiddenSendPlan {
    pub activation: ActivationShape,
    pub activation_configuration: Digest,
    pub activation_root: Digest,
    /// The committed object the Dot reads its activation rows from.
    pub activation_source: ActivationSource,
    pub table_shape: IndexShape,
    pub table_identity: Digest,
    pub matrix_rows: u64,
    pub k_lo: u32,
    pub columns: u64,
    pub group: u64,
    pub oracle_base: u64,
    pub openings_per_oracle: usize,
    pub w: WShape,
    /// The sampled mask-only message that carries the one-time mask entries.
    /// The hidden-input mode requires it; the diagnostic mode leaves it empty.
    pub index_mask: Option<IndexMaskPlan>,
    /// The committed shown-map message of the hidden Dot identity. The
    /// hidden-input mode requires it; the diagnostic mode leaves it empty.
    pub dot: Option<DotMapPlan>,
    /// A fixture expectation for the shown-map message root. It is a test
    /// hook, not a policy pin: a production verifier pins the message through
    /// the operator policy that also pins the activation commitment.
    pub expected_dot_root: Option<Digest>,
    pub expected_roots: Option<Vec<Digest>>,
}

impl HiddenSendPlan {
    pub fn validate(&self) -> Result<()> {
        if self.activation.fault().is_some()
            || self.activation_configuration == Digest::ZERO
            || self.activation_root == Digest::ZERO
        {
            return Err(Error::InvalidConfig);
        }
        if self.activation_source.is_norm_hd() && self.dot.is_none() {
            return Err(Error::InvalidConfig);
        }
        validate_index_shape(&self.table_shape)?;
        if self.table_identity == Digest::ZERO || self.group == 0 {
            return Err(Error::InvalidConfig);
        }
        let table_rows = self.table_shape.rows_checked()?;
        if self.matrix_rows == 0
            || !self.matrix_rows.is_power_of_two()
            || self.matrix_rows != table_rows
        {
            return Err(Error::InvalidConfig);
        }
        let columns = self
            .matrix_rows
            .checked_shl(self.k_lo)
            .ok_or(Error::ResourceLimit)?;
        if columns == 0
            || !columns.is_power_of_two()
            || columns != self.columns
            || self.k_lo.checked_add(self.table_shape.v) != Some(columns.trailing_zeros())
        {
            return Err(Error::InvalidConfig);
        }
        if self.table_shape.plain() < self.columns {
            return Err(Error::ShapeMismatch);
        }
        if self.openings_per_oracle == 0 || self.openings_per_oracle as u64 > self.w.code.n() {
            return Err(Error::InvalidConfig);
        }
        if self.w.fault(self.columns, self.activation.positions).is_some() {
            return Err(Error::InvalidConfig);
        }
        if self.activation.hidden as u64 > self.matrix_rows {
            return Err(Error::ShapeMismatch);
        }
        for position in 0..self.activation.positions {
            self.oracle(position)?;
        }
        match (&self.index_mask, &self.dot) {
            (Some(_), Some(_)) | (None, None) => {}
            _ => return Err(Error::InvalidConfig),
        }
        if let Some(index) = &self.index_mask {
            if index.fault().is_some() || index.group != self.group {
                return Err(Error::InvalidConfig);
            }
            if index.oracle >= self.oracle_base
                && index.oracle
                    < self
                        .oracle_base
                        .checked_add(self.activation.positions as u64)
                        .ok_or(Error::ResourceLimit)?
            {
                return Err(Error::InvalidConfig);
            }
            if index.entries.variables != self.matrix_rows.trailing_zeros() as usize {
                return Err(Error::InvalidConfig);
            }
        }
        if let Some(dot) = &self.dot {
            if dot.fault(self.activation.positions, self.matrix_rows).is_some()
                || self.activation.hidden as u64 != self.matrix_rows
                || dot.group != self.group
                || dot.shape.code != self.w.code
            {
                return Err(Error::InvalidConfig);
            }
            let in_w_oracles = dot.oracle >= self.oracle_base
                && dot.oracle
                    < self
                        .oracle_base
                        .checked_add(self.activation.positions as u64)
                        .ok_or(Error::ResourceLimit)?;
            let in_w_trees = dot.shape.tree_number >= self.w.tree_number
                && dot.shape.tree_number
                    < self
                        .w
                        .tree_number
                        .checked_add(self.activation.positions as u64)
                        .ok_or(Error::ResourceLimit)?;
            if in_w_oracles || in_w_trees {
                return Err(Error::InvalidConfig);
            }
            if let Some(index) = &self.index_mask {
                if dot.oracle == index.oracle
                    || dot.shape.tree_number == index.tree.tree_number
                {
                    return Err(Error::InvalidConfig);
                }
            }
            if let Some(root) = self.expected_dot_root {
                if root == Digest::ZERO {
                    return Err(Error::ShapeMismatch);
                }
            }
        } else if self.expected_dot_root.is_some() {
            return Err(Error::InvalidConfig);
        }
        if let Some(roots) = &self.expected_roots {
            if roots.len() != self.activation.positions() || roots.iter().any(|root| *root == Digest::ZERO)
            {
                return Err(Error::ShapeMismatch);
            }
        }
        Ok(())
    }

    pub fn positions(&self) -> u32 {
        self.activation.positions
    }

    pub fn oracle(&self, position: u32) -> Result<u64> {
        if position >= self.activation.positions {
            return Err(Error::InvalidIndex);
        }
        self.oracle_base
            .checked_add(position as u64)
            .ok_or(Error::ResourceLimit)
    }

    /// The plan's parameter digest. It fixes every public shape of the run
    /// before any challenge.
    pub fn parameters_digest(&self) -> Digest {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(HIDDEN_SEND_SCHEMA_VERSION);
        self.activation.encode_into(&mut writer);
        writer.put_digest(&self.activation_configuration);
        writer.put_digest(&self.activation_root);
        self.activation_source.append(&mut writer);
        put_index_shape(&mut writer, &self.table_shape);
        writer.put_digest(&self.table_identity);
        writer.put_u64(self.matrix_rows);
        writer.put_u64(self.k_lo as u64);
        writer.put_u64(self.columns);
        writer.put_u64(self.group);
        writer.put_u64(self.oracle_base);
        writer.put_u64(self.openings_per_oracle as u64);
        match &self.index_mask {
            Some(index) => {
                writer.put_u64(1);
                index.encode_into(&mut writer);
            }
            None => writer.put_u64(0),
        }
        match &self.dot {
            Some(dot) => {
                writer.put_u64(1);
                dot.encode_into(&mut writer);
            }
            None => writer.put_u64(0),
        }
        self.w.encode_into(&mut writer);
        hash_labeled(tag_word(PARAMETERS, 0), 0, None, &writer.into_bytes())
            .unwrap_or(Digest::ZERO)
    }

    /// The statement bytes of the chain's first query. The `W` roots are proof
    /// values and are not part of it.
    pub fn statement_bytes(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(HIDDEN_SEND_SCHEMA_VERSION);
        self.activation.encode_into(&mut writer);
        writer.put_digest(&self.activation_configuration);
        writer.put_digest(&self.activation_root);
        self.activation_source.append(&mut writer);
        put_index_shape(&mut writer, &self.table_shape);
        writer.put_digest(&self.table_identity);
        writer.put_u64(self.matrix_rows);
        writer.put_u64(self.k_lo as u64);
        writer.put_u64(self.columns);
        writer.put_u64(self.group);
        writer.put_u64(self.oracle_base);
        writer.put_u64(self.openings_per_oracle as u64);
        match &self.index_mask {
            Some(index) => {
                writer.put_u64(1);
                index.encode_into(&mut writer);
            }
            None => writer.put_u64(0),
        }
        match &self.dot {
            Some(dot) => {
                writer.put_u64(1);
                dot.encode_into(&mut writer);
            }
            None => writer.put_u64(0),
        }
        self.w.encode_into(&mut writer);
        writer.into_bytes()
    }
}

// ---------------------------------------------------------------------------
// Prover-side W derivation and commitment
// ---------------------------------------------------------------------------

/// The private random values of the committed shown-map message: one encoding
/// tail per message row, one salt per tree column and one non-zero sample pad.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DotMapRandomness {
    /// `tail[row * r + i]`.
    pub tail: Vec<Fp3>,
    /// `salts[column]`.
    pub salts: Vec<Vec<u8>>,
    pub pad: Fp3,
}

/// The private random values of one `W` commitment: one tail per message row,
/// one salt per tree column and one pad per answer. Every pad must be non-zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WCommitRandomness {
    /// `tails[position][row * r + i]`, one entry per message row and tail
    /// coefficient.
    pub tails: Vec<Vec<Fp3>>,
    /// The private randomness of the closure messages, when the plan carries
    /// them.
    pub closure: Option<ClosureRandomness>,
    /// `salts[position][column]`, one salt per tree column.
    pub salts: Vec<Vec<Vec<u8>>>,
    /// `pads[position][answer]`, non-zero.
    pub pads: Vec<Vec<Fp3>>,
    /// The private randomness of the sampled index-mask message, when the plan
    /// carries one. It is drawn from the same private source after the `W`
    /// values, so all masks of one run are independent of the challenges.
    pub index_mask: Option<IndexMaskRandomness>,
    /// The private randomness of the shown-map message, when the plan carries
    /// one. It is drawn from the same private source after the mask values.
    pub dot: Option<DotMapRandomness>,
}

impl WCommitRandomness {
    fn check(&self, plan: &HiddenSendPlan) -> Result<()> {
        let positions = plan.positions() as usize;
        let rows = usize::try_from(plan.w.main_rows).map_err(|_| Error::ResourceLimit)?;
        let tail = usize::try_from(plan.w.code.r).map_err(|_| Error::ResourceLimit)?;
        let columns = usize::try_from(plan.w.code.n()).map_err(|_| Error::ResourceLimit)?;
        let answers = plan.w.answers as usize;
        if self.tails.len() != positions
            || self.tails.iter().any(|values| values.len() != rows * tail)
            || self.salts.len() != positions
            || self.salts.iter().any(|salts| {
                salts.len() != columns || salts.iter().any(|s| s.len() != plan.w.salt_bytes)
            })
            || self.pads.len() != positions
            || self.pads.iter().any(|pads| pads.len() != answers)
            || self.pads.iter().flatten().any(|pad| *pad == Fp3::ZERO)
        {
            return Err(Error::ShapeMismatch);
        }
        match (&plan.index_mask, &self.index_mask) {
            (Some(index), Some(randomness)) => randomness.check(index)?,
            (None, None) => {}
            _ => return Err(Error::ShapeMismatch),
        }
        match (&plan.dot, &self.dot) {
            (Some(dot), Some(randomness)) => {
                let rows = usize::try_from(dot.shape.main_rows).map_err(|_| Error::ResourceLimit)?;
                let tail = usize::try_from(dot.shape.code.r).map_err(|_| Error::ResourceLimit)?;
                let columns = usize::try_from(dot.shape.code.n()).map_err(|_| Error::ResourceLimit)?;
                if randomness.tail.len() != rows * tail
                    || randomness.salts.len() != columns
                    || randomness
                        .salts
                        .iter()
                        .any(|salt| salt.len() != dot.shape.salt_bytes)
                    || randomness.pad == Fp3::ZERO
                {
                    return Err(Error::ShapeMismatch);
                }
            }
            (Some(_), None) if plan.activation_source.is_norm_hd() => {}
            (None, None) => {}
            _ => return Err(Error::ShapeMismatch),
        }
        Ok(())
    }
}

/// One committed `W` oracle plus the private witness of the sample and Eval
/// linear forms. The messages and pads are never serialized into the proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HiddenSendWitness {
    pub group: u64,
    pub w: WShape,
    pub positions: u32,
    pub oracle_base: u64,
    pub columns: u64,
    /// `messages[position][row][place]`, the unencoded message rows.
    pub messages: Vec<Vec<Vec<Fp3>>>,
    /// `tails[position][row * r + i]`.
    pub tails: Vec<Vec<Fp3>>,
    /// `salts[position][column]`.
    pub salts: Vec<Vec<Vec<u8>>>,
    /// `pads[position][answer]`.
    pub pads: Vec<Vec<Fp3>>,
    pub roots: Vec<Digest>,
    /// The runtime trees built by the same W commit, in position order.
    pub trees: RetainedTrees,
    /// The private index-mask message, when the plan carries one.
    pub index_mask: Option<IndexMaskWitness>,
    /// The private shown-map message, when the plan carries one.
    pub dot: Option<DotMapWitness>,
    /// The private closure messages, when the plan carries them.
    pub closure: Option<ClosureWitness>,
    /// The private norm group messages, when the plan carries the norm.
    pub norm: Option<NormWitness>,
}

impl HiddenSendWitness {
    fn check(&self, plan: &HiddenSendPlan) -> Result<()> {
        let positions = plan.positions() as usize;
        let rows = usize::try_from(plan.w.main_rows).map_err(|_| Error::ResourceLimit)?;
        let places = usize::try_from(plan.w.code.ell).map_err(|_| Error::ResourceLimit)?;
        if self.group != plan.group
            || self.w != plan.w
            || self.positions != plan.activation.positions
            || self.oracle_base != plan.oracle_base
            || self.columns != plan.columns
            || self.messages.len() != positions
            || self.messages.iter().any(|message| {
                message.len() != rows || message.iter().any(|row| row.len() != places)
            })
            || self.pads.len() != positions
            || self.pads.iter().any(|pads| pads.len() != plan.w.answers as usize)
            || self.roots.len() != positions
        {
            return Err(Error::ShapeMismatch);
        }
        let randomness = WCommitRandomness {
            tails: self.tails.clone(),
            salts: self.salts.clone(),
            pads: self.pads.clone(),
            index_mask: match (&plan.index_mask, &self.index_mask) {
                (Some(_index), Some(witness)) => {
                    Some(IndexMaskRandomness {
                        rows: witness.rows.clone(),
                        tails: witness.tails.clone(),
                        salts: witness.salts.clone(),
                    })
                }
                (None, None) => None,
                _ => return Err(Error::ShapeMismatch),
            },
            dot: match (&plan.dot, &self.dot) {
                (Some(_dot), Some(witness)) => Some(DotMapRandomness {
                    tail: witness.tails.concat(),
                    salts: witness.salts.clone(),
                    pad: witness.pad,
                }),
                (Some(_), None) if plan.activation_source.is_norm_hd() => None,
                (None, None) => None,
                _ => return Err(Error::ShapeMismatch),
            },
            closure: None,
        };
        randomness.check(plan)?;
        if let Some(index) = &self.index_mask {
            index.check(
                plan.index_mask
                    .as_ref()
                    .ok_or(Error::ShapeMismatch)?,
            )?;
        }
        if let Some(dot) = &self.dot {
            dot.check(plan.dot.as_ref().ok_or(Error::ShapeMismatch)?)?;
            if let Some(root) = plan.expected_dot_root {
                if dot.root != root {
                    return Err(Error::StatementMismatch);
                }
            }
        }
        if let Some(roots) = &plan.expected_roots {
            if self.roots != *roots {
                return Err(Error::StatementMismatch);
            }
        }
        for position in 0..plan.positions() {
            if *self.tree(position)?.root() != self.roots[position as usize] {
                return Err(Error::StatementMismatch);
            }
        }
        Ok(())
    }

    /// Returns the runtime tree built by the W commit for one position. The
    /// root must equal the registered root.
    pub fn tree(&self, position: u32) -> Result<&RuntimeTree> {
        if position >= self.positions {
            return Err(Error::InvalidIndex);
        }
        self.trees
            .get(position as usize)
            .map(|tree| tree.as_ref())
            .ok_or(Error::InvalidIndex)
    }

    pub fn oracle(&self, position: u32) -> Result<u64> {
        if position >= self.positions {
            return Err(Error::InvalidIndex);
        }
        self.oracle_base
            .checked_add(position as u64)
            .ok_or(Error::ResourceLimit)
    }

    /// The compact row-major witness view the sample and Eval relations read.
    pub fn relation(&self, position: u32) -> Result<GuWitness> {
        let message = self
            .messages
            .get(position as usize)
            .ok_or(Error::InvalidIndex)?;
        let mut main = Vec::new();
        for row in message {
            main.extend_from_slice(row);
        }
        Ok(GuWitness {
            main,
            mask: Vec::new(),
        })
    }

    /// The complete ordered answers of the SEND sample message: by position,
    /// then by answer number.
    pub fn answers(&self, seeds: &[SampleSeed]) -> Result<Vec<Fp3>> {
        let shape = self.w.oracle_shape();
        if shape.fault().is_some() || seeds.len() != self.w.answers as usize {
            return Err(Error::ShapeMismatch);
        }
        let mut answers = Vec::with_capacity(self.positions as usize * self.w.answers as usize);
        for position in 0..self.positions {
            let witness = self.relation(position)?;
            let oracle = self.oracle(position)?;
            for answer in 0..self.w.answers {
                let row = oracle_sample_row(oracle, self.group, &shape, seeds, answer, Fp3::ZERO)?;
                answers.push(row.evaluate(&witness)?);
            }
        }
        Ok(answers)
    }

    /// One Eval value per position: `W_t at psi_t`.
    pub fn eval_values(&self, points: &[Vec<Fp3>]) -> Result<Vec<Fp3>> {
        if points.len() != self.positions as usize {
            return Err(Error::ShapeMismatch);
        }
        let mut values = Vec::with_capacity(points.len());
        for (position, point) in points.iter().enumerate() {
            let message = self.messages.get(position).ok_or(Error::InvalidIndex)?;
            values.push(eval_message(
                point,
                message,
                self.w.main_rows,
                self.columns,
            )?);
        }
        Ok(values)
    }
}

/// Derives the `W` rows from the committed encoded table. One pass over the
/// table's row source; the complete weight matrix is never materialized. The
/// table identity, shape and matrix mapping are checked against the plan.
pub fn derive_w_rows(
    plan: &HiddenSendPlan,
    activation: &CommittedActivation,
    reader: &IndexReader,
    source: &mut dyn TableRowSource,
) -> Result<Vec<Vec<Fp3>>> {
    plan.validate()?;
    check_table(plan, reader)?;
    check_activation(plan, activation)?;
    if plan.activation_source.is_norm_hd() {
        return Err(Error::InvalidConfig);
    }
    let positions = plan.positions() as usize;
    let mut weights = Vec::with_capacity(positions);
    for position in 0..positions {
        weights.push(activation.weights(position as u32, plan.matrix_rows)?);
    }
    derive_w_rows_from_weights(plan, &weights, reader, source)
}

/// Derives the `W` rows from explicit private row-weight vectors. The caller
/// owns the source of the rows: the committed activation, or the committed
/// norm hidden-output messages.
fn derive_w_rows_from_weights(
    plan: &HiddenSendPlan,
    weights: &[Vec<Fp3>],
    reader: &IndexReader,
    source: &mut dyn TableRowSource,
) -> Result<Vec<Vec<Fp3>>> {
    if source.rows() != plan.table_shape.rows() {
        return Err(Error::ShapeMismatch);
    }
    let positions = plan.positions() as usize;
    let matrix_rows = usize::try_from(plan.matrix_rows).map_err(|_| Error::ResourceLimit)?;
    let columns = usize::try_from(plan.columns).map_err(|_| Error::ResourceLimit)?;
    let table_rows = usize::try_from(plan.table_shape.rows()).map_err(|_| Error::ResourceLimit)?;
    let factor = 1usize << plan.k_lo;
    if !columns.is_power_of_two() || columns != (matrix_rows << plan.k_lo) {
        return Err(Error::InvalidConfig);
    }
    if weights.len() != positions || weights.iter().any(|row| row.len() != matrix_rows) {
        return Err(Error::ShapeMismatch);
    }
    let mut w = vec![vec![Fp3::ZERO; columns]; positions];
    let ell = usize::try_from(plan.table_shape.code.ell).map_err(|_| Error::ResourceLimit)?;
    let tail_len = usize::try_from(plan.table_shape.code.r).map_err(|_| Error::ResourceLimit)?;
    let mut message = vec![Fp::ZERO; ell];
    let mut tail = vec![Fp::ZERO; tail_len];
    for table_row in 0..table_rows {
        source.row(table_row as u64, &mut message, &mut tail)?;
        let base = table_row << plan.k_lo;
        if base + factor > columns {
            return Err(Error::ShapeMismatch);
        }
        // The table row is read in order; the independent position rows are
        // derived in parallel. Each output cell keeps its serial sum order.
        if positions > 1 && (factor as u64).saturating_mul(plan.matrix_rows) >= 4096 {
            w.par_iter_mut()
                .enumerate()
                .try_for_each(|(position, w_row)| -> Result<()> {
                    let g = &weights[position];
                    for low in 0..factor {
                        let mut sum = Fp3::ZERO;
                        for (row, weight) in g.iter().enumerate() {
                            let entry = low + (row << plan.k_lo);
                            let value = *message.get(entry).ok_or(Error::ShapeMismatch)?;
                            if !weight.is_zero() {
                                sum = sum.add(&weight.mul(&Fp3::from_base(value)));
                            }
                        }
                        w_row[base + low] = sum;
                    }
                    Ok(())
                })?;
        } else {
            for (position, g) in weights.iter().enumerate() {
                for low in 0..factor {
                    let mut sum = Fp3::ZERO;
                    for (row, weight) in g.iter().enumerate() {
                        let entry = low + (row << plan.k_lo);
                        let value = *message.get(entry).ok_or(Error::ShapeMismatch)?;
                        if !weight.is_zero() {
                            sum = sum.add(&weight.mul(&Fp3::from_base(value)));
                        }
                    }
                    w[position][base + low] = sum;
                }
            }
        }
    }
    Ok(w)
}

/// Commits one `W` message per position and returns the group and the private
/// witness. Each message is placed in its main rows, encoded with the product
/// form and committed under its runtime tree. The sample pads are reserved at
/// the end of the last row and filled from the supplied private randomness.
pub fn commit_w_rows(
    plan: &HiddenSendPlan,
    rows: &[Vec<Fp3>],
    randomness: &WCommitRandomness,
) -> Result<(SendGroup, HiddenSendWitness)> {
    plan.validate()?;
    randomness.check(plan)?;
    let positions = plan.positions() as usize;
    if rows.len() != positions || rows.iter().any(|row| row.len() as u64 != plan.columns) {
        return Err(Error::ShapeMismatch);
    }
    let main_rows = usize::try_from(plan.w.main_rows).map_err(|_| Error::ResourceLimit)?;
    let ell = usize::try_from(plan.w.code.ell).map_err(|_| Error::ResourceLimit)?;
    let tail_len = usize::try_from(plan.w.code.r).map_err(|_| Error::ResourceLimit)?;
    let answers = plan.w.answers as usize;
    let mut entries = Vec::with_capacity(positions);
    let mut messages = Vec::with_capacity(positions);
    let mut roots = Vec::with_capacity(positions);
    let mut trees = Vec::with_capacity(positions);
    for (position, values) in rows.iter().enumerate() {
        let position = position as u32;
        let mut message = vec![vec![Fp3::ZERO; ell]; main_rows];
        for (column, value) in values.iter().enumerate() {
            let row = column % main_rows;
            let place = column / main_rows;
            let entry = message
                .get_mut(row)
                .and_then(|row| row.get_mut(place))
                .ok_or(Error::ShapeMismatch)?;
            *entry = *value;
        }
        let pads = randomness
            .pads
            .get(position as usize)
            .ok_or(Error::ShapeMismatch)?;
        for (answer, pad) in pads.iter().enumerate() {
            let place = ell
                .checked_sub(answers)
                .and_then(|first| first.checked_add(answer))
                .ok_or(Error::ResourceLimit)?;
            let entry = message
                .get_mut(main_rows - 1)
                .and_then(|row| row.get_mut(place))
                .ok_or(Error::ShapeMismatch)?;
            if !entry.is_zero() {
                return Err(Error::StatementMismatch);
            }
            *entry = *pad;
        }
        let tails = randomness
            .tails
            .get(position as usize)
            .ok_or(Error::ShapeMismatch)?;
        let mut codewords = Vec::with_capacity(main_rows);
        for row in 0..main_rows {
            let message_row = message.get(row).ok_or(Error::ShapeMismatch)?;
            let tail = tails
                .get(row * tail_len..(row + 1) * tail_len)
                .ok_or(Error::ShapeMismatch)?;
            codewords.push(encode_row_fp3(&plan.w.code, message_row, tail)?);
        }
        let salts = randomness
            .salts
            .get(position as usize)
            .ok_or(Error::ShapeMismatch)?;
        let tree_plan = plan.w.tree_plan(position)?;
        let tree = RuntimeTree::commit(
            tree_plan.clone(),
            &[RuntimePartRows {
                main: codewords,
                mask: Vec::new(),
                main_deg: E_DEGREE as u32,
            }],
            salts,
        )?;
        let root = *tree.root();
        entries.push(SendGroupEntry {
            position,
            oracle: plan.oracle(position)?,
            shape: plan.w.oracle_shape(),
            tree: tree_plan,
            root,
        });
        roots.push(root);
        trees.push(Arc::new(tree));
        messages.push(message);
    }
    if let Some(expected) = &plan.expected_roots {
        if roots != *expected {
            return Err(Error::StatementMismatch);
        }
    }
    let group = SendGroup {
        group: plan.group,
        entries,
    };
    if group.fault(plan.activation.positions).is_some() {
        return Err(Error::InvalidConfig);
    }
    let index_mask = match (&plan.index_mask, &randomness.index_mask) {
        (Some(index), Some(randomness)) => {
            let mut codewords = Vec::with_capacity(randomness.rows.len());
            for (row, tail) in randomness.rows.iter().zip(&randomness.tails) {
                codewords.push(encode_row_fp3(&index.mask_code, row, tail)?);
            }
            let tree = RuntimeTree::commit(
                index.tree.clone(),
                &[RuntimePartRows {
                    main: Vec::new(),
                    mask: codewords,
                    main_deg: 1,
                }],
                &randomness.salts,
            )?;
            let witness = IndexMaskWitness {
                rows: randomness.rows.clone(),
                tails: randomness.tails.clone(),
                salts: randomness.salts.clone(),
                root: *tree.root(),
            };
            witness.check(index)?;
            Some(witness)
        }
        (None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    let witness = HiddenSendWitness {
        group: plan.group,
        w: plan.w,
        positions: plan.activation.positions,
        oracle_base: plan.oracle_base,
        columns: plan.columns,
        messages,
        tails: randomness.tails.clone(),
        salts: randomness.salts.clone(),
        pads: randomness.pads.clone(),
        roots,
        trees: RetainedTrees(trees),
        index_mask,
        dot: None,
        closure: None,
        norm: None,
    };
    Ok((group, witness))
}

// ---------------------------------------------------------------------------
// The shown-map message of the hidden Dot identity
// ---------------------------------------------------------------------------

/// Commits the shown-map message from the activation rows. The entry
/// `t * hidden + j` holds value `j` of position `t`; the sample pad is the
/// last entry of the last message row. The rows, tails and salts stay with the
/// prover; only the root and the sample answer are public.
fn commit_dot_map(
    plan: &HiddenSendPlan,
    activation: &CommittedActivation,
    randomness: &WCommitRandomness,
) -> Result<(Digest, DotMapWitness)> {
    plan.validate()?;
    check_activation(plan, activation)?;
    let dot = plan.dot.as_ref().ok_or(Error::InvalidConfig)?;
    let positions = plan.positions() as usize;
    let hidden = dot.hidden as usize;
    let rows = usize::try_from(dot.shape.main_rows).map_err(|_| Error::ResourceLimit)?;
    let places = usize::try_from(dot.shape.code.ell).map_err(|_| Error::ResourceLimit)?;
    let tail_len = usize::try_from(dot.shape.code.r).map_err(|_| Error::ResourceLimit)?;
    let random = randomness.dot.as_ref().ok_or(Error::ShapeMismatch)?;
    if activation.rows().len() != positions
        || activation.rows().iter().any(|row| row.len() != hidden)
    {
        return Err(Error::ShapeMismatch);
    }
    let mut message = vec![vec![Fp3::ZERO; places]; rows];
    for (position, values) in activation.rows().iter().enumerate() {
        for (value_at, value) in values.iter().enumerate() {
            let entry = position
                .checked_mul(hidden)
                .and_then(|base| base.checked_add(value_at))
                .ok_or(Error::ResourceLimit)?;
            let row = entry % rows;
            let place = entry / rows;
            let slot = message
                .get_mut(row)
                .and_then(|row| row.get_mut(place))
                .ok_or(Error::ShapeMismatch)?;
            *slot = *value;
        }
    }
    let pad_slot = message
        .get_mut(rows - 1)
        .and_then(|row| row.get_mut(places - 1))
        .ok_or(Error::ShapeMismatch)?;
    if !pad_slot.is_zero() {
        return Err(Error::StatementMismatch);
    }
    *pad_slot = random.pad;
    let mut tails = Vec::with_capacity(rows);
    for row in 0..rows {
        tails.push(
            random
                .tail
                .get(row * tail_len..(row + 1) * tail_len)
                .ok_or(Error::ShapeMismatch)?
                .to_vec(),
        );
    }
    let tree = commit_rows_tree(
        dot.tree_plan()?,
        dot.shape.code,
        &message,
        &tails,
        &random.salts,
    )?;
    let root = *tree.root();
    let witness = DotMapWitness {
        message,
        tails,
        salts: random.salts.clone(),
        pad: random.pad,
        root,
    };
    witness.check(dot)?;
    Ok((root, witness))
}

// ---------------------------------------------------------------------------
// The Eval row
// ---------------------------------------------------------------------------

/// One Eval relation row on a `W` message: the `eq(psi, .)` coefficients over
/// the message entries and the public value the row must take.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvalSampleRow {
    pub oracle: u64,
    pub group: u64,
    pub position: u32,
    pub main: Vec<Fp3>,
    pub target: Fp3,
}

impl EvalSampleRow {
    /// Evaluates the row on a witness. This is a unit-test and auxiliary
    /// helper; the verifier does not hold the witness in the protocol path.
    pub fn evaluate(&self, witness: &GuWitness) -> Result<Fp3> {
        if witness.main.len() != self.main.len() {
            return Err(Error::ShapeMismatch);
        }
        let mut value = Fp3::ZERO;
        for (coefficient, entry) in self.main.iter().zip(&witness.main) {
            value = value.add(&coefficient.mul(entry));
        }
        Ok(value)
    }
}

fn eval_row(
    plan: &HiddenSendPlan,
    position: u32,
    point: &[Fp3],
    target: Fp3,
) -> Result<EvalSampleRow> {
    let k_col = plan.columns.trailing_zeros();
    if point.len() != k_col as usize {
        return Err(Error::ShapeMismatch);
    }
    let main_rows = usize::try_from(plan.w.main_rows).map_err(|_| Error::ResourceLimit)?;
    let ell = usize::try_from(plan.w.code.ell).map_err(|_| Error::ResourceLimit)?;
    let columns = usize::try_from(plan.columns).map_err(|_| Error::ResourceLimit)?;
    let mut main = vec![Fp3::ZERO; main_rows * ell];
    for j in 0..columns {
        let row = j % main_rows;
        let place = j / main_rows;
        let coefficient = eq_index(point, j as u64, k_col)?;
        let entry = main.get_mut(row * ell + place).ok_or(Error::ShapeMismatch)?;
        *entry = entry.add(&coefficient);
    }
    Ok(EvalSampleRow {
        oracle: plan.oracle(position)?,
        group: plan.group,
        position,
        main,
        target,
    })
}

fn eval_message(
    point: &[Fp3],
    message: &[Vec<Fp3>],
    main_rows: u64,
    columns: u64,
) -> Result<Fp3> {
    if !columns.is_power_of_two() || point.len() != columns.trailing_zeros() as usize {
        return Err(Error::ShapeMismatch);
    }
    if main_rows == 0 || !main_rows.is_power_of_two() {
        return Err(Error::InvalidConfig);
    }
    let k_col = columns.trailing_zeros();
    let mut sum = Fp3::ZERO;
    for j in 0..columns {
        let row = (j % main_rows) as usize;
        let place = (j / main_rows) as usize;
        let value = message
            .get(row)
            .and_then(|row| row.get(place))
            .ok_or(Error::ShapeMismatch)?;
        sum = sum.add(&eq_index(point, j, k_col)?.mul(value));
    }
    Ok(sum)
}

fn eq_index(point: &[Fp3], index: u64, bits: u32) -> Result<Fp3> {
    if point.len() != bits as usize {
        return Err(Error::ShapeMismatch);
    }
    let mut value = Fp3::ONE;
    for (bit, coordinate) in point.iter().enumerate() {
        let factor = if (index >> bit) & 1 == 1 {
            *coordinate
        } else {
            Fp3::ONE.sub(coordinate)
        };
        value = value.mul(&factor);
    }
    Ok(value)
}

// ---------------------------------------------------------------------------
// The message and the session
// ---------------------------------------------------------------------------

/// The public shown-map message of the hidden Dot identity: the trusted tree
/// plan, the committed root and the one SEND sample answer that the own group
/// checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DotMapMessage {
    pub tree: RuntimeTreePlan,
    pub root: Digest,
    pub sample: SendSamples,
}

/// The complete public message of one hidden SEND-sample run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HiddenSendMessages {
    pub activation: ActivationCommitment,
    pub group: SendGroup,
    pub samples: SendSamples,
    /// The opening positions per oracle, drawn after the answers.
    pub positions: Vec<Vec<u64>>,
    pub openings: Vec<RuntimeTreeOpening>,
    /// One Eval point per position, drawn after the openings.
    pub psi: Vec<Vec<Fp3>>,
    /// The masked Eval targets `tau_t = W_t(psi_t) + e_t` in the hidden-input
    /// mode; the plain evaluations in the diagnostic mode.
    pub eval_targets: Vec<Fp3>,
    /// The sampled mask-only message. The hidden-input mode requires it.
    pub index_mask: Option<IndexMaskMessage>,
    /// The committed shown-map message. The hidden-input mode requires it.
    pub dot: Option<DotMapMessage>,
    /// The committed closure messages. The formal output-relation mode
    /// requires them.
    pub closure: Option<ClosureFirstMessages>,
    /// The integrated norm first message, when the plan carries the norm.
    pub norm: Option<NormFirstMessages>,
    pub chain_value: Digest,
}

impl HiddenSendMessages {
    pub fn encode(&self) -> Result<Vec<u8>> {
        if self.group.entries.len() != self.openings.len()
            || self.positions.len() != self.openings.len()
        {
            return Err(Error::ShapeMismatch);
        }
        for (positions, opening) in self.positions.iter().zip(&self.openings) {
            if opening.positions != *positions {
                return Err(Error::ShapeMismatch);
            }
        }
        let mut writer = CanonicalWriter::new();
        writer.put_u64(HIDDEN_SEND_SCHEMA_VERSION);
        writer.put_bytes(&self.activation.encode());
        writer.put_bytes(&self.group.encode());
        writer.put_bytes(&self.samples.encode());
        writer.put_u64(self.positions.len() as u64);
        for positions in &self.positions {
            writer.put_u64(positions.len() as u64);
            for position in positions {
                writer.put_u64(*position);
            }
        }
        writer.put_u64(self.openings.len() as u64);
        for (entry, opening) in self.group.entries.iter().zip(&self.openings) {
            writer.put_bytes(&encode_runtime_opening(&entry.tree, opening)?);
        }
        writer.put_u64(self.psi.len() as u64);
        for point in &self.psi {
            writer.put_u64(point.len() as u64);
            for coordinate in point {
                writer.put_fp3(coordinate);
            }
        }
        writer.put_u64(self.eval_targets.len() as u64);
        for target in &self.eval_targets {
            writer.put_fp3(target);
        }
        match &self.index_mask {
            Some(message) => {
                writer.put_u64(1);
                put_tree_plan(&mut writer, &message.tree);
                writer.put_digest(&message.root);
                writer.put_bytes(&message.sample.encode());
                writer.put_u64(message.positions.len() as u64);
                for position in &message.positions {
                    writer.put_u64(*position);
                }
                writer.put_bytes(&encode_runtime_opening(&message.tree, &message.opening)?);
            }
            None => writer.put_u64(0),
        }
        match &self.dot {
            Some(message) => {
                writer.put_u64(1);
                put_tree_plan(&mut writer, &message.tree);
                writer.put_digest(&message.root);
                writer.put_bytes(&message.sample.encode());
            }
            None => writer.put_u64(0),
        }
        if let Some(closure) = &self.closure {
            writer.put_u64(1);
            writer.put_bytes(&closure.encode()?);
        }
        if let Some(norm) = &self.norm {
            writer.put_bytes(&norm.encode()?);
        }
        writer.put_digest(&self.chain_value);
        Ok(writer.into_bytes())
    }

    /// Decodes a message against the plan. The plan supplies the trusted tree
    /// shapes of the openings; the decoded group is checked by the verifier.
    pub fn decode(
        plan: &HiddenSendPlan,
        closure: Option<&ClosurePlan>,
        norm: Option<&NormPlan>,
        bytes: &[u8],
    ) -> Result<Self> {
        plan.validate()?;
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        if reader.get_u64()? != HIDDEN_SEND_SCHEMA_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let activation = ActivationCommitment::decode(reader.get_bytes()?)?;
        let group = SendGroup::decode(reader.get_bytes()?)?;
        let samples = SendSamples::decode(reader.get_bytes()?)?;
        let position_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if position_count != plan.positions() as usize {
            return Err(Error::ShapeMismatch);
        }
        let mut positions = Vec::with_capacity(position_count);
        for _ in 0..position_count {
            let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
            if count == 0 || count > plan.w.code.n() as usize {
                return Err(Error::InvalidEncoding);
            }
            let mut list = Vec::with_capacity(count);
            for _ in 0..count {
                list.push(reader.get_u64()?);
            }
            positions.push(list);
        }
        let opening_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if opening_count != position_count {
            return Err(Error::ShapeMismatch);
        }
        let mut openings = Vec::with_capacity(opening_count);
        for (position, list) in positions.iter().enumerate() {
            let tree = plan.w.tree_plan(position as u32)?;
            let opening = decode_runtime_opening(&tree, list, reader.get_bytes()?)?;
            if opening.positions != *list {
                return Err(Error::InvalidEncoding);
            }
            openings.push(opening);
        }
        let point_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if point_count != position_count {
            return Err(Error::ShapeMismatch);
        }
        let k_col = plan.columns.trailing_zeros();
        let mut psi = Vec::with_capacity(point_count);
        for _ in 0..point_count {
            let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
            if count != k_col as usize {
                return Err(Error::ShapeMismatch);
            }
            let mut point = Vec::with_capacity(count);
            for _ in 0..count {
                point.push(reader.get_fp3()?);
            }
            psi.push(point);
        }
        let target_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if target_count != position_count {
            return Err(Error::ShapeMismatch);
        }
        let mut eval_targets = Vec::with_capacity(target_count);
        for _ in 0..target_count {
            eval_targets.push(reader.get_fp3()?);
        }
        let index_mask = match reader.get_u64()? {
            0 => None,
            1 => {
                let index = plan.index_mask.as_ref().ok_or(Error::InvalidEncoding)?;
                let tree = get_tree_plan(&mut reader)?;
                if tree != index.tree {
                    return Err(Error::StatementMismatch);
                }
                let root = reader.get_digest()?;
                if root == Digest::ZERO {
                    return Err(Error::InvalidEncoding);
                }
                let sample = SendSamples::decode(reader.get_bytes()?)?;
                if sample.seed.len() != index.answers as usize
                    || sample.answers.len() != index.answers as usize
                {
                    return Err(Error::ShapeMismatch);
                }
                let position_count =
                    usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
                if position_count != index.openings {
                    return Err(Error::ShapeMismatch);
                }
                let mut mask_positions = Vec::with_capacity(position_count);
                for _ in 0..position_count {
                    mask_positions.push(reader.get_u64()?);
                }
                let opening =
                    decode_runtime_opening(&tree, &mask_positions, reader.get_bytes()?)?;
                Some(IndexMaskMessage {
                    tree,
                    root,
                    sample,
                    positions: mask_positions,
                    opening,
                })
            }
            _ => return Err(Error::InvalidEncoding),
        };
        let dot = match reader.get_u64()? {
            0 => None,
            1 => {
                let map = plan.dot.as_ref().ok_or(Error::InvalidEncoding)?;
                let tree = get_tree_plan(&mut reader)?;
                if tree != map.tree_plan()? {
                    return Err(Error::StatementMismatch);
                }
                let root = reader.get_digest()?;
                if root == Digest::ZERO {
                    return Err(Error::InvalidEncoding);
                }
                let sample = SendSamples::decode(reader.get_bytes()?)?;
                if sample.seed.len() != map.shape.answers as usize
                    || sample.answers.len() != map.shape.answers as usize
                {
                    return Err(Error::ShapeMismatch);
                }
                Some(DotMapMessage {
                    tree,
                    root,
                    sample,
                })
            }
            _ => return Err(Error::InvalidEncoding),
        };
        let closure = match closure {
            Some(closure_plan) => match reader.get_u64()? {
                0 => return Err(Error::ShapeMismatch),
                1 => Some(ClosureFirstMessages::decode(
                    closure_plan,
                    plan.positions(),
                    reader.get_bytes()?,
                )?),
                _ => return Err(Error::InvalidEncoding),
            },
            None => None,
        };
        let norm = match norm {
            Some(norm_plan) => Some(NormFirstMessages::decode(norm_plan, reader.get_bytes()?)?),
            None => None,
        };
        let chain_value = reader.get_digest()?;
        reader.finish()?;
        Ok(Self {
            activation,
            group,
            samples,
            positions,
            openings,
            psi,
            eval_targets,
            index_mask,
            dot,
            closure,
            norm,
            chain_value,
        })
    }
}

/// The verifier outcome: every public value plus the pending sample rows and
/// the Eval rows the later stages consume.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HiddenSendOutcome {
    pub activation: ActivationCommitment,
    pub group: SendGroup,
    pub seeds: Vec<SampleSeed>,
    pub answers: Vec<Fp3>,
    pub sample_rows: Vec<GuRelationRow>,
    pub eval_rows: Vec<EvalSampleRow>,
    pub positions: Vec<Vec<u64>>,
    pub psi: Vec<Vec<Fp3>>,
    pub eval_targets: Vec<Fp3>,
    /// The committed index-mask root, when the plan carries the mask message.
    pub index_mask_root: Option<Digest>,
    /// The committed shown-map root, when the plan carries the Dot message.
    pub dot_root: Option<Digest>,
    pub chain_value: Digest,
}

/// Prover entry. The activation commitment must equal the plan's pin; the
/// encoded table identity must equal the plan's; the `W` rows are derived from
/// the table's row source. The returned messages are public, the witness is
/// private.
pub fn prove_hidden_send(
    plan: &HiddenSendPlan,
    activation: &CommittedActivation,
    table_reader: &IndexReader,
    table_source: &mut dyn TableRowSource,
    randomness: &WCommitRandomness,
) -> Result<(HiddenSendMessages, HiddenSendWitness)> {
    plan.validate()?;
    check_activation(plan, activation)?;
    check_table(plan, table_reader)?;
    let rows = derive_w_rows(plan, activation, table_reader, table_source)?;
    let (group, mut witness) = commit_w_rows(plan, &rows, randomness)?;
    let dot_map = match &plan.dot {
        Some(_) => {
            let (root, map) = commit_dot_map(plan, activation, randomness)?;
            witness.dot = Some(map);
            Some(root)
        }
        None => None,
    };
    witness.check(plan)?;

    let mut chain = Chain::begin(
        HIDDEN_SEND_CHAIN_LABEL,
        &plan.parameters_digest(),
        &plan.statement_bytes(),
        &plan.table_identity.0,
        None,
    )?;
    absorb_group(&mut chain, &group)?;
    let index_mask = match (&plan.index_mask, &witness.index_mask) {
        (Some(index), Some(mask_witness)) => {
            absorb_index_mask_root(&mut chain, index, &mask_witness.root)?;
            Some((index, mask_witness))
        }
        (None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    let dot_root = match (&plan.dot, &dot_map) {
        (Some(_), Some(root)) => {
            absorb_dot_root(&mut chain, plan, root)?;
            Some(*root)
        }
        (None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    let seeds = draw_seeds(&mut chain, plan)?;
    let dot_seeds = match &plan.dot {
        Some(dot) => Some(draw_seed_set(&mut chain, dot.sample_bits()?, 1)?),
        None => None,
    };
    let answers = witness.answers(&seeds)?;
    absorb_answers(&mut chain, &answers)?;
    let dot_sample = match (&plan.dot, &dot_seeds, &witness.dot) {
        (Some(_dot), Some(seed_set), Some(map)) => {
            let answer = dot_sample_answer(plan, map, &seed_set[0])?;
            absorb_dot_answer(&mut chain, answer)?;
            Some(SendSamples {
                seed: seed_set.clone(),
                answers: vec![answer],
            })
        }
        (None, None, None) => None,
        _ => return Err(Error::InvalidProof),
    };
    let mut index_mask_seed = None;    let mut index_mask_answer_value = None;
    let mut index_mask_positions = None;
    let mut index_mask_opening = None;
    if let Some((index, mask_witness)) = index_mask {
        let seed = seeds.first().ok_or(Error::ShapeMismatch)?.clone();
        let answer = index_mask_answer(index, &mask_witness.rows, &seed)?;
        absorb_index_mask_answer(&mut chain, answer)?;
        let positions = draw_index_mask_positions(&mut chain, index)?;
        let opening = mask_witness.tree(index)?.open(&positions)?;
        index_mask_seed = Some(seed);
        index_mask_answer_value = Some(answer);
        index_mask_positions = Some(positions);
        index_mask_opening = Some(opening);
    }
    let positions = draw_positions(&mut chain, plan)?;
    let openings = zkie_engine::scheduling::try_map_range_bounded(
        positions.len(),
        zkie_engine::scheduling::DEFAULT_UNITS_IN_FLIGHT,
        |position| witness.tree(position as u32)?.open(&positions[position]),
    )?;
    let psi = draw_psi(&mut chain, plan)?;
    let mut eval_targets = witness.eval_values(&psi)?;
    if let (Some(index), Some(mask_witness)) = (&plan.index_mask, &witness.index_mask) {
        for position in 0..plan.positions() {
            let mask = mask_witness.entry(index, index.entries.eval_slot(position)?)?;
            eval_targets[position as usize] = eval_targets[position as usize].add(&mask);
        }
    }
    absorb_eval(&mut chain, plan, &eval_targets)?;
    let chain_value = *chain.value();

    let index_mask = match (index_mask_seed, index_mask_answer_value, index_mask_positions, index_mask_opening) {
        (Some(seed), Some(answer), Some(positions), Some(opening)) => Some(IndexMaskMessage {
            tree: plan
                .index_mask
                .as_ref()
                .ok_or(Error::ShapeMismatch)?
                .tree
                .clone(),
            root: witness
                .index_mask
                .as_ref()
                .ok_or(Error::ShapeMismatch)?
                .root,
            sample: SendSamples {
                seed: vec![seed],
                answers: vec![answer],
            },
            positions,
            opening,
        }),
        (None, None, None, None) => None,
        _ => return Err(Error::InvalidProof),
    };
    let dot = match (plan.dot.as_ref(), dot_root, dot_sample) {
        (Some(map), Some(root), Some(sample)) => Some(DotMapMessage {
            tree: map.tree_plan()?,
            root,
            sample,
        }),
        (None, None, None) => None,
        _ => return Err(Error::InvalidProof),
    };
    let messages = HiddenSendMessages {
        activation: *activation.commitment(),
        group,
        samples: SendSamples { seed: seeds, answers },
        positions,
        openings,
        psi,
        eval_targets,
        index_mask,
        dot,
        chain_value,
        closure: None,
        norm: None,
    };
    Ok((messages, witness))
}

/// Verifier entry with the unit's auxiliary committed witness. The proof bytes
/// never carry the witness; the check evaluates the same sample and Eval linear
/// forms and therefore rejects a changed answer or Eval target here. The
/// protocol-shaped entry is [`verify_hidden_send_public`].
pub fn verify_hidden_send(
    plan: &HiddenSendPlan,
    messages: &HiddenSendMessages,
    witness: &HiddenSendWitness,
) -> Result<HiddenSendOutcome> {
    witness.check(plan)?;
    let outcome = run_verify(plan, messages)?;
    let expected = witness.answers(&outcome.seeds)?;
    if expected != outcome.answers {
        return Err(Error::InvalidProof);
    }
    let expected = witness.eval_values(&outcome.psi)?;
    match (&plan.index_mask, &witness.index_mask) {
        (Some(index), Some(mask_witness)) => {
            let message = messages.index_mask.as_ref().ok_or(Error::ShapeMismatch)?;
            let seed = message
                .sample
                .seed
                .first()
                .ok_or(Error::ShapeMismatch)?;
            let answer = index_mask_answer(index, &mask_witness.rows, seed)?;
            if message.sample.answers.first() != Some(&answer) {
                return Err(Error::InvalidProof);
            }
            for position in 0..plan.positions() {
                let mask = mask_witness.entry(index, index.entries.eval_slot(position)?)?;
                let target = *messages
                    .eval_targets
                    .get(position as usize)
                    .ok_or(Error::ShapeMismatch)?;
                if target != expected[position as usize].add(&mask) {
                    return Err(Error::InvalidProof);
                }
            }
        }
        (None, None) => {
            if expected != outcome.eval_targets {
                return Err(Error::InvalidProof);
            }
        }
        _ => return Err(Error::ShapeMismatch),
    }
    if plan.index_mask.is_none() {
        for (position, row) in outcome.eval_rows.iter().enumerate() {
            let relation = witness.relation(position as u32)?;
            if row.evaluate(&relation)? != row.target {
                return Err(Error::InvalidProof);
            }
        }
    }
    Ok(outcome)
}

/// Protocol-shaped verifier entry. It checks the plan, the group, the drawn
/// seed, the answers' order and count, the openings and the Eval points; the
/// sample and Eval rows stay pending, because this unit does not run the
/// proximity test that decides them. A well-formed wrong answer is not
/// rejected here.
pub fn verify_hidden_send_public(
    plan: &HiddenSendPlan,
    messages: &HiddenSendMessages,
) -> Result<HiddenSendOutcome> {
    run_verify(plan, messages)
}

fn run_verify(plan: &HiddenSendPlan, messages: &HiddenSendMessages) -> Result<HiddenSendOutcome> {
    plan.validate()?;
    let expected_roots = plan
        .expected_roots
        .as_ref()
        .ok_or(Error::StatementMismatch)?;
    if messages.activation.fault().is_some()
        || messages.activation.root != plan.activation_root
        || messages.activation.shape != plan.activation
        || messages.activation.configuration != plan.activation_configuration
    {
        return Err(Error::StatementMismatch);
    }
    if messages.group.fault(plan.activation.positions).is_some() {
        return Err(Error::InvalidConfig);
    }
    if messages.group.entries.len() != plan.positions() as usize {
        return Err(Error::ShapeMismatch);
    }
    for (position, entry) in messages.group.entries.iter().enumerate() {
        if entry.position != position as u32
            || entry.oracle != plan.oracle(position as u32)?
            || entry.shape != plan.w.oracle_shape()
            || entry.tree != plan.w.tree_plan(position as u32)?
            || entry.root != expected_roots[position]
        {
            return Err(Error::StatementMismatch);
        }
    }

    let mut chain = Chain::begin(
        HIDDEN_SEND_CHAIN_LABEL,
        &plan.parameters_digest(),
        &plan.statement_bytes(),
        &plan.table_identity.0,
        None,
    )?;
    absorb_group(&mut chain, &messages.group)?;
    match (&plan.index_mask, &messages.index_mask) {
        (Some(index), Some(message)) => {
            if message.tree != index.tree
                || message.root == Digest::ZERO
                || message.sample.seed.len() != index.answers as usize
                || message.sample.answers.len() != index.answers as usize
            {
                return Err(Error::StatementMismatch);
            }
            absorb_index_mask_root(&mut chain, index, &message.root)?;
        }
        (None, None) => {}
        _ => return Err(Error::ShapeMismatch),
    }
    match (&plan.dot, &messages.dot) {
        (Some(map), Some(message)) => {
            if message.tree != map.tree_plan()?
                || message.root == Digest::ZERO
                || message.sample.seed.len() != map.shape.answers as usize
                || message.sample.answers.len() != map.shape.answers as usize
            {
                return Err(Error::StatementMismatch);
            }
            if let Some(root) = plan.expected_dot_root {
                if message.root != root {
                    return Err(Error::StatementMismatch);
                }
            }
            absorb_dot_root(&mut chain, plan, &message.root)?;
        }
        (None, None) => {}
        _ => return Err(Error::ShapeMismatch),
    }
    let seeds = draw_seeds(&mut chain, plan)?;
    if seeds != messages.samples.seed {
        return Err(Error::InvalidProof);
    }
    let dot_seeds = match &plan.dot {
        Some(dot) => {
            let drawn = draw_seed_set(&mut chain, dot.sample_bits()?, 1)?;
            let message = messages.dot.as_ref().ok_or(Error::ShapeMismatch)?;
            if drawn != message.sample.seed {
                return Err(Error::InvalidProof);
            }
            Some(drawn)
        }
        None => None,
    };
    let expected_answers = plan.positions() as usize * plan.w.answers as usize;
    if messages.samples.answers.len() != expected_answers {
        return Err(Error::ShapeMismatch);
    }
    absorb_answers(&mut chain, &messages.samples.answers)?;
    if let (Some(_seed), Some(message)) = (&dot_seeds, &messages.dot) {
        absorb_dot_answer(
            &mut chain,
            *message
                .sample
                .answers
                .first()
                .ok_or(Error::ShapeMismatch)?,
        )?;
    }
    if let (Some(index), Some(message)) = (&plan.index_mask, &messages.index_mask) {
        if message.sample.seed != seeds {
            return Err(Error::InvalidProof);
        }
        absorb_index_mask_answer(
            &mut chain,
            *message
                .sample
                .answers
                .first()
                .ok_or(Error::ShapeMismatch)?,
        )?;
        let mask_positions = draw_index_mask_positions(&mut chain, index)?;
        if mask_positions != message.positions
            || message.opening.positions != message.positions
        {
            return Err(Error::InvalidProof);
        }
        verify_runtime_opening(
            &index.tree,
            &message.root,
            &message.positions,
            &message.opening,
        )?;
    }
    let positions = draw_positions(&mut chain, plan)?;
    if positions != messages.positions {
        return Err(Error::InvalidProof);
    }
    if messages.openings.len() != positions.len() {
        return Err(Error::ShapeMismatch);
    }
    for (position, (list, opening)) in positions.iter().zip(&messages.openings).enumerate() {
        if opening.positions != *list {
            return Err(Error::InvalidIndex);
        }
        let tree = plan.w.tree_plan(position as u32)?;
        verify_runtime_opening(&tree, &expected_roots[position], list, opening)?;
    }
    let psi = draw_psi(&mut chain, plan)?;
    if psi != messages.psi {
        return Err(Error::InvalidProof);
    }
    if messages.eval_targets.len() != plan.positions() as usize {
        return Err(Error::ShapeMismatch);
    }
    absorb_eval(&mut chain, plan, &messages.eval_targets)?;
    if *chain.value() != messages.chain_value {
        return Err(Error::InvalidProof);
    }

    let shape = plan.w.oracle_shape();
    let mut sample_rows = Vec::with_capacity(expected_answers);
    for position in 0..plan.positions() {
        for answer in 0..plan.w.answers {
            let index = position as usize * plan.w.answers as usize + answer as usize;
            let target = *messages
                .samples
                .answers
                .get(index)
                .ok_or(Error::ShapeMismatch)?;
            sample_rows.push(oracle_sample_row(
                plan.oracle(position)?,
                plan.group,
                &shape,
                &seeds,
                answer,
                target,
            )?);
        }
    }
    if let (Some(index), Some(message)) = (&plan.index_mask, &messages.index_mask) {
        let target = *message
            .sample
            .answers
            .first()
            .ok_or(Error::ShapeMismatch)?;
        sample_rows.push(index_mask_sample_row(index, message, target)?);
    }
    if let (Some(dot), Some(seed_set), Some(message)) = (&plan.dot, &dot_seeds, &messages.dot) {
        let target = *message
            .sample
            .answers
            .first()
            .ok_or(Error::ShapeMismatch)?;
        sample_rows.push(oracle_sample_row(
            dot.oracle,
            dot.group,
            &dot.oracle_shape(),
            seed_set,
            0,
            target,
        )?);
    }
    let mut eval_rows = Vec::with_capacity(plan.positions() as usize);
    for (position, point) in psi.iter().enumerate() {
        let target = *messages
            .eval_targets
            .get(position)
            .ok_or(Error::ShapeMismatch)?;
        eval_rows.push(eval_row(plan, position as u32, point, target)?);
    }

    Ok(HiddenSendOutcome {
        activation: messages.activation,
        group: messages.group.clone(),
        seeds,
        answers: messages.samples.answers.clone(),
        sample_rows,
        eval_rows,
        positions,
        psi,
        eval_targets: messages.eval_targets.clone(),
        index_mask_root: messages.index_mask.as_ref().map(|message| message.root),
        dot_root: messages.dot.as_ref().map(|message| message.root),
        chain_value: *chain.value(),
    })
}

fn check_activation(plan: &HiddenSendPlan, activation: &CommittedActivation) -> Result<()> {
    if activation.shape() != plan.activation
        || activation.root() != plan.activation_root
        || activation.commitment().configuration != plan.activation_configuration
    {
        return Err(Error::StatementMismatch);
    }
    if activation.recompute_root()? != plan.activation_root {
        return Err(Error::StatementMismatch);
    }
    Ok(())
}

/// The activation rows of the norm hidden-output object. The rows are the
/// first `matrix_rows` committed entries of every position's norm output
/// message: the same committed cells the norm relation proves and the Dot
/// consumes, not a second commitment.
fn hd_activation_rows(
    plan: &HiddenLinPlan,
    norm: Option<&NormWitness>,
) -> Result<Vec<Vec<Fp3>>> {
    if !plan.send.activation_source.is_norm_hd() {
        return Err(Error::InvalidConfig);
    }
    let norm_plan = plan.norm.as_ref().ok_or(Error::InvalidConfig)?;
    let output_plan = norm_plan.output.as_ref().ok_or(Error::InvalidConfig)?;
    let group = &output_plan.out;
    let norm = norm.ok_or(Error::ShapeMismatch)?;
    let output = norm.output.as_ref().ok_or(Error::ShapeMismatch)?;
    let rows = usize::try_from(group.shape.main_rows).map_err(|_| Error::ResourceLimit)?;
    let ell = usize::try_from(group.shape.code.ell).map_err(|_| Error::ResourceLimit)?;
    let hidden = usize::try_from(plan.send.matrix_rows).map_err(|_| Error::ResourceLimit)?;
    let positions = plan.send.positions() as usize;
    if hidden == 0
        || hidden > usize::try_from(group.columns).map_err(|_| Error::ResourceLimit)?
        || output.out.messages.len() != positions
    {
        return Err(Error::ShapeMismatch);
    }
    let mut values = Vec::with_capacity(positions);
    for message in &output.out.messages {
        if message.len() != rows || message.iter().any(|row| row.len() != ell) {
            return Err(Error::ShapeMismatch);
        }
        let mut row = Vec::with_capacity(hidden);
        for value_at in 0..hidden {
            let entry = value_at;
            row.push(message[entry % rows][entry / rows]);
        }
        values.push(row);
    }
    Ok(values)
}

/// The statement identity of the norm hidden-output object: the recorded plan
/// numbers and the ordered committed roots of the norm output group. The
/// verifier recomputes it from the first-stage norm message, so the Dot is
/// bound to the object the norm relation proves.
fn hd_activation_identity(
    plan: &HiddenLinPlan,
    entries: &[SendGroupEntry],
) -> Result<Digest> {
    let norm = plan.norm.as_ref().ok_or(Error::InvalidConfig)?;
    let output = norm.output.as_ref().ok_or(Error::InvalidConfig)?;
    let group = &output.out;
    if entries.len() != plan.send.positions() as usize {
        return Err(Error::ShapeMismatch);
    }
    for (position, entry) in entries.iter().enumerate() {
        if entry.position != position as u32
            || entry.oracle != group.oracle_base + position as u64
            || entry.shape != group.shape.oracle_shape()
        {
            return Err(Error::StatementMismatch);
        }
    }
    let mut writer = CanonicalWriter::new();
    writer.put_u64(HD_ACTIVATION_IDENTITY_VERSION);
    writer.put_u64(group.oracle_base);
    writer.put_u64(group.group);
    writer.put_u64(group.columns);
    group.shape.encode_into(&mut writer);
    writer.put_u64(entries.len() as u64);
    for entry in entries {
        writer.put_u64(entry.position as u64);
        writer.put_u64(entry.oracle);
        writer.put_digest(&entry.root);
    }
    hash_labeled(tag_word(PARAMETERS, 7), 0, None, &writer.into_bytes())
}

fn check_table(plan: &HiddenSendPlan, reader: &IndexReader) -> Result<()> {
    if reader.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    validate_index_shape(&reader.shape)?;
    if reader.shape != plan.table_shape {
        return Err(Error::StatementMismatch);
    }
    if reader_table_identity(reader)? != plan.table_identity {
        return Err(Error::StatementMismatch);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Chain rounds
// ---------------------------------------------------------------------------

fn push_digest(message: &mut ChainMessage, digest: &Digest) {
    for word in digest.0.chunks_exact(8) {
        message.push_u64(u64::from_le_bytes(
            word.try_into().expect("eight-byte word"),
        ));
    }
}

fn push_oracle_shape(message: &mut ChainMessage, shape: &GuOracleShape) {
    message.push_u64(shape.code.log_n as u64);
    message.push_u64(shape.code.ell);
    message.push_u64(shape.code.r);
    message.push_u64(shape.main_rows);
    message.push_u64(shape.main_deg as u64);
    message.push_u64(shape.mask_rows);
    message.push_u64(if shape.mask_code.is_some() { 1 } else { 0 });
    if let Some(code) = shape.mask_code {
        message.push_u64(code.log_n as u64);
        message.push_u64(code.ell);
        message.push_u64(code.r);
    }
    message.push_u64(shape.answers as u64);
}

fn push_tree_plan(message: &mut ChainMessage, tree: &RuntimeTreePlan) {
    message.push_u64(tree.tree_number);
    message.push_u64(tree.leaf_class as u64);
    message.push_u64(tree.log_n as u64);
    message.push_u64(tree.log_n_zk as u64);
    message.push_u64(tree.parts.len() as u64);
    for part in &tree.parts {
        message.push_u64(part.main_symbols);
        message.push_u64(part.mask_symbols);
    }
    message.push_u64(tree.salt_bytes as u64);
    push_digest(message, &tree.run);
    push_digest(message, &tree.configuration);
}

fn absorb_group(chain: &mut Chain, group: &SendGroup) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(group.group);
    message.push_u64(group.entries.len() as u64);
    for entry in &group.entries {
        message.push_u64(entry.position as u64);
        message.push_u64(entry.oracle);
        push_oracle_shape(&mut message, &entry.shape);
        push_tree_plan(&mut message, &entry.tree);
        push_digest(&mut message, &entry.root);
    }
    let prefix = message.prefix(KIND_SEED, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn draw_seeds(chain: &mut Chain, plan: &HiddenSendPlan) -> Result<Vec<SampleSeed>> {
    let mut bits = plan.w.sample_bits()?;
    if let Some(index) = &plan.index_mask {
        bits = bits.max(index.sample_bits()?);
    }
    if bits == 0 {
        return Err(Error::InvalidConfig);
    }
    let mut seeds = Vec::with_capacity(plan.w.answers as usize);
    for _ in 0..plan.w.answers {
        let mut z = Vec::with_capacity(bits as usize);
        for _ in 0..bits {
            z.push(chain.draw_e()?);
        }
        let beta = chain.draw_e_nonzero()?;
        seeds.push(SampleSeed { z, beta });
    }
    Ok(seeds)
}

fn absorb_answers(chain: &mut Chain, answers: &[Fp3]) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(answers.len() as u64);
    for answer in answers {
        message.push_fp3(answer);
    }
    let prefix = message.prefix(KIND_EMPTY, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// Absorbs the shown-map message root before its sample seed is drawn.
fn absorb_dot_root(chain: &mut Chain, plan: &HiddenSendPlan, root: &Digest) -> Result<()> {
    let dot = plan.dot.as_ref().ok_or(Error::InvalidConfig)?;
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(14);
    message.push_u64(dot.oracle);
    message.push_u64(dot.group);
    push_tree_plan(&mut message, &dot.tree_plan()?);
    push_digest(&mut message, root);
    let prefix = message.prefix(KIND_SEED, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// Absorbs the shown-map sample answer after the reduced-row answers.
fn absorb_dot_answer(chain: &mut Chain, answer: Fp3) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(15);
    message.push_fp3(&answer);
    let prefix = message.prefix(KIND_EMPTY, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// The sample answer of the shown-map message: the `eq` form over the message
/// cube plus the reserved pad under the seed's `beta`.
fn dot_sample_answer(
    plan: &HiddenSendPlan,
    witness: &DotMapWitness,
    seed: &SampleSeed,
) -> Result<Fp3> {
    let dot = plan.dot.as_ref().ok_or(Error::InvalidConfig)?;
    let gu = GuWitness {
        main: witness.message.concat(),
        mask: Vec::new(),
    };
    let row = oracle_sample_row(
        dot.oracle,
        dot.group,
        &dot.oracle_shape(),
        std::slice::from_ref(seed),
        0,
        Fp3::ZERO,
    )?;
    row.evaluate(&gu)
}

fn draw_positions(chain: &mut Chain, plan: &HiddenSendPlan) -> Result<Vec<Vec<u64>>> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(plan.positions() as u64);
    message.push_u64(plan.openings_per_oracle as u64);
    let prefix = message.prefix(KIND_POSITIONS, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    let mut positions = Vec::with_capacity(plan.positions() as usize);
    for _ in 0..plan.positions() {
        let mut drawn = Vec::with_capacity(plan.openings_per_oracle);
        for _ in 0..plan.openings_per_oracle {
            drawn.push(chain.draw_position(plan.w.code.log_n)?);
        }
        positions.push(unique_sorted(&drawn));
    }
    Ok(positions)
}

fn draw_psi(chain: &mut Chain, plan: &HiddenSendPlan) -> Result<Vec<Vec<Fp3>>> {
    let k_col = plan.columns.trailing_zeros();
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(plan.positions() as u64);
    message.push_u64(k_col as u64);
    let prefix = message.prefix(KIND_SEED, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    let mut points = Vec::with_capacity(plan.positions() as usize);
    for _ in 0..plan.positions() {
        let mut point = Vec::with_capacity(k_col as usize);
        for _ in 0..k_col {
            point.push(chain.draw_e()?);
        }
        points.push(point);
    }
    Ok(points)
}

fn absorb_eval(chain: &mut Chain, plan: &HiddenSendPlan, targets: &[Fp3]) -> Result<()> {
    if targets.len() != plan.positions() as usize {
        return Err(Error::ShapeMismatch);
    }
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(targets.len() as u64);
    for (position, target) in targets.iter().enumerate() {
        message.push_u64(position as u64);
        message.push_fp3(target);
    }
    let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn absorb_index_mask_root(chain: &mut Chain, plan: &IndexMaskPlan, root: &Digest) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(plan.oracle);
    message.push_u64(plan.group);
    message.push_u64(plan.mask_rows);
    message.push_u64(plan.answers as u64);
    message.push_u64(plan.mask_code.log_n as u64);
    message.push_u64(plan.mask_code.ell);
    message.push_u64(plan.mask_code.r);
    push_tree_plan(&mut message, &plan.tree);
    push_digest(&mut message, root);
    let prefix = message.prefix(KIND_SEED, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn absorb_index_mask_answer(chain: &mut Chain, answer: Fp3) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(1);
    message.push_fp3(&answer);
    let prefix = message.prefix(KIND_EMPTY, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn draw_index_mask_positions(
    chain: &mut Chain,
    plan: &IndexMaskPlan,
) -> Result<Vec<u64>> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(1);
    message.push_u64(plan.openings as u64);
    let prefix = message.prefix(KIND_POSITIONS, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    let mut positions = Vec::with_capacity(plan.openings);
    for _ in 0..plan.openings {
        positions.push(chain.draw_position(plan.mask_code.log_n)?);
    }
    Ok(unique_sorted(&positions))
}

/// The SEND sample answer of the mask-only message: `eq(z, .)` over every
/// non-pad mask entry plus the public non-zero `beta` on the private pad.
fn index_mask_answer(
    plan: &IndexMaskPlan,
    rows: &[Vec<Fp3>],
    seed: &SampleSeed,
) -> Result<Fp3> {
    if plan.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    let bits = plan.sample_bits()?;
    if seed.z.len() < bits as usize || seed.beta == Fp3::ZERO {
        return Err(Error::ShapeMismatch);
    }
    let ell = usize::try_from(plan.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
    let mask_rows = usize::try_from(plan.mask_rows).map_err(|_| Error::ResourceLimit)?;
    if rows.len() != mask_rows || rows.iter().any(|row| row.len() != ell) {
        return Err(Error::ShapeMismatch);
    }
    let row_bits = ceil_log2_u64(plan.mask_rows);
    let point = &seed.z[..bits as usize];
    let count = 1u64
        .checked_shl(bits)
        .ok_or(Error::ResourceLimit)?;
    let mut answer = Fp3::ZERO;
    for index in 0..count {
        let row = (index & ((1u64 << row_bits) - 1)) as usize;
        let place = (index >> row_bits) as usize;
        if row >= mask_rows || place >= ell {
            continue;
        }
        let coefficient = if row + 1 == mask_rows && place + 1 == ell {
            seed.beta
        } else {
            eq_index(point, index, bits)?
        };
        answer = answer.add(&coefficient.mul(&rows[row][place]));
    }
    Ok(answer)
}

/// The SEND sample row of the mask-only message. The target is the received
/// answer; the coefficients are on the mask table in its compact row-major
/// order, which the group adapter maps to the BAT virtual order.
fn index_mask_sample_row(
    plan: &IndexMaskPlan,
    message: &IndexMaskMessage,
    target: Fp3,
) -> Result<GuRelationRow> {
    if plan.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    if message.sample.answers.len() != 1 || message.sample.seed.len() < 1 {
        return Err(Error::ShapeMismatch);
    }
    let seed = &message.sample.seed[0];
    let bits = plan.sample_bits()?;
    if seed.z.len() < bits as usize || seed.beta == Fp3::ZERO {
        return Err(Error::ShapeMismatch);
    }
    let ell = usize::try_from(plan.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
    let mask_rows = usize::try_from(plan.mask_rows).map_err(|_| Error::ResourceLimit)?;
    let mut mask = vec![Fp3::ZERO; mask_rows * ell];
    let row_bits = ceil_log2_u64(plan.mask_rows);
    let stride = 1usize
        .checked_shl(row_bits)
        .ok_or(Error::ResourceLimit)?;
    let point = &seed.z[..bits as usize];
    let count = 1u64
        .checked_shl(bits)
        .ok_or(Error::ResourceLimit)?;
    for index in 0..count {
        let row = (index & (stride as u64 - 1)) as usize;
        let place = (index >> row_bits) as usize;
        if row >= mask_rows || place >= ell {
            continue;
        }
        let coefficient = if row + 1 == mask_rows && place + 1 == ell {
            seed.beta
        } else {
            eq_index(point, index, bits)?
        };
        mask[row * ell + place] = coefficient;
    }
    Ok(GuRelationRow {
        oracle: plan.oracle,
        group: plan.group,
        kind: GuRelationKind::Sample,
        main: Vec::new(),
        mask,
        target,
    })
}

// ---------------------------------------------------------------------------
// Output-closure first stage
// ---------------------------------------------------------------------------

/// The SEND sample answer of the closure mask-only message.
fn closure_mask_answer(
    plan: &ClosureMaskPlan,
    rows: &[Vec<Fp3>],
    seed: &SampleSeed,
) -> Result<Fp3> {
    if plan.oracle == 0 || plan.group == 0 || plan.answers != 1 {
        return Err(Error::InvalidConfig);
    }
    let bits = plan.sample_bits()?;
    if seed.z.len() < bits as usize || seed.beta == Fp3::ZERO {
        return Err(Error::ShapeMismatch);
    }
    let ell = usize::try_from(plan.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
    let mask_rows = usize::try_from(plan.mask_rows).map_err(|_| Error::ResourceLimit)?;
    if rows.len() != mask_rows || rows.iter().any(|row| row.len() != ell) {
        return Err(Error::ShapeMismatch);
    }
    let row_bits = ceil_log2_u64(plan.mask_rows);
    let point = &seed.z[..bits as usize];
    let count = 1u64.checked_shl(bits).ok_or(Error::ResourceLimit)?;
    let mut answer = Fp3::ZERO;
    for index in 0..count {
        let row = (index & ((1u64 << row_bits) - 1)) as usize;
        let place = (index >> row_bits) as usize;
        if row >= mask_rows || place >= ell {
            continue;
        }
        let coefficient = if row + 1 == mask_rows && place + 1 == ell {
            seed.beta
        } else {
            eq_index(point, index, bits)?
        };
        answer = answer.add(&coefficient.mul(&rows[row][place]));
    }
    Ok(answer)
}

/// The SEND sample row of the closure mask-only message.
fn closure_mask_sample_row(
    plan: &ClosureMaskPlan,
    message: &ClosureMaskMessage,
    target: Fp3,
) -> Result<GuRelationRow> {
    if message.sample.answers.len() != 1 || message.sample.seed.len() < 1 {
        return Err(Error::ShapeMismatch);
    }
    let seed = &message.sample.seed[0];
    let bits = plan.sample_bits()?;
    if seed.z.len() < bits as usize || seed.beta == Fp3::ZERO {
        return Err(Error::ShapeMismatch);
    }
    let ell = usize::try_from(plan.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
    let mask_rows = usize::try_from(plan.mask_rows).map_err(|_| Error::ResourceLimit)?;
    let mut mask = vec![Fp3::ZERO; mask_rows * ell];
    let row_bits = ceil_log2_u64(plan.mask_rows);
    let stride = 1usize.checked_shl(row_bits).ok_or(Error::ResourceLimit)?;
    let point = &seed.z[..bits as usize];
    let count = 1u64.checked_shl(bits).ok_or(Error::ResourceLimit)?;
    for index in 0..count {
        let row = (index & (stride as u64 - 1)) as usize;
        let place = (index >> row_bits) as usize;
        if row >= mask_rows || place >= ell {
            continue;
        }
        let coefficient = if row + 1 == mask_rows && place + 1 == ell {
            seed.beta
        } else {
            eq_index(point, index, bits)?
        };
        mask[row * ell + place] = coefficient;
    }
    Ok(GuRelationRow {
        oracle: plan.oracle,
        group: plan.group,
        kind: GuRelationKind::Sample,
        main: Vec::new(),
        mask,
        target,
    })
}

/// One answer per position of one committed value group.
fn value_group_answers(
    shape: &WShape,
    oracle_base: u64,
    group: u64,
    messages: &[Vec<Vec<Fp3>>],
    seeds: &[SampleSeed],
) -> Result<Vec<Fp3>> {
    if seeds.len() != shape.answers as usize || messages.len() == 0 {
        return Err(Error::ShapeMismatch);
    }
    let oracle_shape = shape.oracle_shape();
    if oracle_shape.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    let mut answers = Vec::with_capacity(messages.len());
    for (position, message) in messages.iter().enumerate() {
        let oracle = oracle_base
            .checked_add(position as u64)
            .ok_or(Error::ResourceLimit)?;
        let row = oracle_sample_row(oracle, group, &oracle_shape, seeds, 0, Fp3::ZERO)?;
        let witness = GuWitness {
            main: message.concat(),
            mask: Vec::new(),
        };
        answers.push(row.evaluate(&witness)?);
    }
    Ok(answers)
}

/// Absorbs one closure group before its sample seed is drawn.
fn absorb_closure_group(chain: &mut Chain, tag: u64, group: &SendGroup) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(tag);
    message.push_u64(group.group);
    message.push_u64(group.entries.len() as u64);
    for entry in &group.entries {
        message.push_u64(entry.position as u64);
        message.push_u64(entry.oracle);
        push_oracle_shape(&mut message, &entry.shape);
        push_tree_plan(&mut message, &entry.tree);
        push_digest(&mut message, &entry.root);
    }
    let prefix = message.prefix(KIND_SEED, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// Absorbs the closure mask root before its sample seed is drawn.
fn absorb_closure_mask_root(chain: &mut Chain, plan: &ClosureMaskPlan, root: &Digest) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(25);
    message.push_u64(plan.oracle);
    message.push_u64(plan.group);
    message.push_u64(plan.mask_rows);
    message.push_u64(plan.answers as u64);
    message.push_u64(plan.mask_code.log_n as u64);
    message.push_u64(plan.mask_code.ell);
    message.push_u64(plan.mask_code.r);
    push_tree_plan(&mut message, &plan.tree);
    push_digest(&mut message, root);
    let prefix = message.prefix(KIND_SEED, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// Draws one closure seed set after a domain-separated round.
fn draw_closure_seed_set(
    chain: &mut Chain,
    tag: u64,
    bits: u32,
    answers: u32,
) -> Result<Vec<SampleSeed>> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(tag);
    message.push_u64(bits as u64);
    message.push_u64(answers as u64);
    let prefix = message.prefix(KIND_SEED, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    draw_seed_set(chain, bits, answers)
}

/// Absorbs the closure sample answers in a fixed order.
fn absorb_closure_answers(
    chain: &mut Chain,
    slack: &[Fp3],
    bits: &[Fp3],
    mask: &[Fp3],
) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(30);
    for values in [slack, bits, mask] {
        message.push_u64(values.len() as u64);
        for value in values {
            message.push_fp3(value);
        }
    }
    let prefix = message.prefix(KIND_EMPTY, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// Absorbs one closure answer list under a domain tag.
fn absorb_tagged_answers(chain: &mut Chain, tag: u64, values: &[Fp3]) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(tag);
    message.push_u64(values.len() as u64);
    for value in values {
        message.push_fp3(value);
    }
    let prefix = message.prefix(KIND_EMPTY, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// Draws the first-stage booleanity point after the bit message has been
/// committed and sampled.
fn draw_closure_r(chain: &mut Chain, variables: u32) -> Result<Vec<Fp3>> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(34);
    message.push_u64(variables as u64);
    let prefix = message.prefix(KIND_SEED, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    let mut point = Vec::with_capacity(variables as usize);
    for _ in 0..variables {
        point.push(chain.draw_e()?);
    }
    Ok(point)
}

/// Draws the opening positions of one closure value group.
fn draw_closure_group_positions(
    chain: &mut Chain,
    tag: u64,
    positions: u32,
    openings: usize,
    log_n: u32,
) -> Result<Vec<Vec<u64>>> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(tag);
    message.push_u64(positions as u64);
    message.push_u64(openings as u64);
    let prefix = message.prefix(KIND_POSITIONS, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    let mut drawn = Vec::with_capacity(positions as usize);
    for _ in 0..positions {
        let mut list = Vec::with_capacity(openings);
        for _ in 0..openings {
            list.push(chain.draw_position(log_n)?);
        }
        drawn.push(unique_sorted(&list));
    }
    Ok(drawn)
}

/// Opens the drawn positions of one committed value group from its witness.
fn open_value_group_openings(
    shape: &WShape,
    messages: &[Vec<Vec<Fp3>>],
    tails: &[Vec<Fp3>],
    salts: &[Vec<Vec<u8>>],
    positions: &[Vec<u64>],
    trees: &RetainedTrees,
) -> Result<Vec<RuntimeTreeOpening>> {
    if messages.len() != positions.len()
        || tails.len() != positions.len()
        || salts.len() != positions.len()
        || trees.len() != positions.len()
    {
        return Err(Error::ShapeMismatch);
    }
    zkie_engine::scheduling::try_map_range_bounded(
        positions.len(),
        zkie_engine::scheduling::DEFAULT_UNITS_IN_FLIGHT,
        |position| {
            let tree = trees.get(position).ok_or(Error::ShapeMismatch)?;
            if tree.plan() != &shape.tree_plan(position as u32)? {
                return Err(Error::StatementMismatch);
            }
            tree.open(&positions[position])
        },
    )
}

/// The compact mask entries of every position, hiding entries first and the
/// two shown masks last.
fn closure_mask_entry_rows(
    plan: &ClosurePlan,
    mask: &CommittedClosureMask,
) -> Result<Vec<Vec<Fp3>>> {
    let positions = plan.winner.len();
    let per_position = plan.mask.per_position()?;
    let mut rows = Vec::with_capacity(positions);
    for position in 0..positions {
        let mut entries = Vec::with_capacity(per_position);
        for index in 0..plan.mask.hiding {
            entries.push(mask.entry(&plan.mask, plan.mask.hiding_slot(position as u32, index)?)?);
        }
        for index in 0..plan.mask.shown {
            entries.push(mask.entry(&plan.mask, plan.mask.shown_slot(position as u32, index)?)?);
        }
        rows.push(entries);
    }
    Ok(rows)
}

/// Draws the closure mask opening positions.
fn draw_closure_mask_positions(chain: &mut Chain, plan: &ClosureMaskPlan) -> Result<Vec<u64>> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(33);
    message.push_u64(1);
    message.push_u64(plan.openings as u64);
    let prefix = message.prefix(KIND_POSITIONS, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    let mut positions = Vec::with_capacity(plan.openings);
    for _ in 0..plan.openings {
        positions.push(chain.draw_position(plan.mask_code.log_n)?);
    }
    Ok(unique_sorted(&positions))
}

// ---------------------------------------------------------------------------
// Canonical helpers
// ---------------------------------------------------------------------------

fn put_index_shape(writer: &mut CanonicalWriter, shape: &IndexShape) {
    writer.put_u64(shape.code.log_n as u64);
    writer.put_u64(shape.code.ell);
    writer.put_u64(shape.code.r);
    writer.put_u64(shape.v as u64);
    writer.put_u64(shape.answers as u64);
    writer.put_u64(shape.salt_bytes as u64);
}

fn put_oracle_shape(writer: &mut CanonicalWriter, shape: &GuOracleShape) {
    writer.put_u64(shape.code.log_n as u64);
    writer.put_u64(shape.code.ell);
    writer.put_u64(shape.code.r);
    writer.put_u64(shape.main_rows);
    writer.put_u64(shape.main_deg as u64);
    writer.put_u64(shape.mask_rows);
    writer.put_u64(if shape.mask_code.is_some() { 1 } else { 0 });
    if let Some(code) = shape.mask_code {
        writer.put_u64(code.log_n as u64);
        writer.put_u64(code.ell);
        writer.put_u64(code.r);
    }
    writer.put_u64(shape.answers as u64);
}

fn get_oracle_shape(reader: &mut CanonicalReader<'_>) -> Result<GuOracleShape> {
    let log_n = u32::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    let ell = reader.get_u64()?;
    let r = reader.get_u64()?;
    let code = build_code_plan(log_n, ell, r)?;
    let main_rows = reader.get_u64()?;
    let main_deg = u32::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    let mask_rows = reader.get_u64()?;
    let has_mask = reader.get_u64()?;
    if has_mask > 1 {
        return Err(Error::InvalidEncoding);
    }
    let mask_code = if has_mask == 1 {
        let log_n = u32::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        let ell = reader.get_u64()?;
        let r = reader.get_u64()?;
        Some(build_code_plan(log_n, ell, r)?)
    } else {
        None
    };
    let answers = u32::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    let shape = GuOracleShape {
        code,
        main_rows,
        main_deg,
        mask_rows,
        mask_code,
        answers,
    };
    if shape.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    Ok(shape)
}

fn put_tree_plan(writer: &mut CanonicalWriter, tree: &RuntimeTreePlan) {
    writer.put_u64(tree.tree_number);
    writer.put_u64(tree.leaf_class as u64);
    writer.put_u64(tree.log_n as u64);
    writer.put_u64(tree.log_n_zk as u64);
    writer.put_u64(tree.parts.len() as u64);
    for part in &tree.parts {
        writer.put_u64(part.main_symbols);
        writer.put_u64(part.mask_symbols);
    }
    writer.put_u64(tree.salt_bytes as u64);
    writer.put_digest(&tree.run);
    writer.put_digest(&tree.configuration);
}

fn get_tree_plan(reader: &mut CanonicalReader<'_>) -> Result<RuntimeTreePlan> {
    let tree_number = reader.get_u64()?;
    let leaf_class = u32::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    let log_n = u32::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    let log_n_zk = u32::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    let part_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if part_count == 0 || part_count > 16 {
        return Err(Error::InvalidEncoding);
    }
    let mut parts = Vec::with_capacity(part_count);
    for _ in 0..part_count {
        parts.push(RuntimeTreePart {
            main_symbols: reader.get_u64()?,
            mask_symbols: reader.get_u64()?,
        });
    }
    let salt_bytes = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    let run = reader.get_digest()?;
    let configuration = reader.get_digest()?;
    let tree = RuntimeTreePlan {
        tree_number,
        leaf_class,
        log_n,
        log_n_zk,
        parts,
        salt_bytes,
        run,
        configuration,
    };
    if tree.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    Ok(tree)
}

fn read_fp3_vec(reader: &mut CanonicalReader<'_>, max: usize) -> Result<Vec<Fp3>> {
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if count > max {
        return Err(Error::ResourceLimit);
    }
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(reader.get_fp3()?);
    }
    Ok(values)
}

fn read_digest_vec(reader: &mut CanonicalReader<'_>, max: usize) -> Result<Vec<Digest>> {
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if count > max {
        return Err(Error::ResourceLimit);
    }
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(reader.get_digest()?);
    }
    Ok(values)
}


// ===========================================================================
// The merged three-stage LIN session
// ===========================================================================
//
// The first stage fixes the hidden activation commitment, the public claim
// items and the committed `W` oracles, samples them, registers the Eval rows
// and seals. The shared stage resumes from that seal, folds the committed
// encoded table at the shared point, authenticates the table record and the
// drawn columns, sends the folded row and `Omega`, registers the Fold row and
// runs the proximity test on the folded-row group. The second stage starts
// from the first query and the shared seal, adopts `Omega`, registers the Dot
// rows and runs the proximity test on the request's own group (the `W` oracles
// and `Omega`), ending in the final BASE consumption and seal.
//
// The public claim items are the activation commitment, the public output `Y`,
// the public row weights `g_t` and the table identity. The prover keeps the
// activation rows, the table row source and every private tail, salt and pad.
// The verifier holds the committed table reader (root, record, configuration
// and shape), the public weights and `Y`; it never receives an activation row,
// a table message row or a `W` message row in the clear.

/// The public output matrix shape. `columns` is the complete padded column
/// domain of the MatMul relation, not the valid vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HiddenOutputShape {
    pub positions: u32,
    pub columns: u64,
}

impl HiddenOutputShape {
    pub fn fault(&self) -> Option<&'static str> {
        if self.positions == 0 || self.columns == 0 || !self.columns.is_power_of_two() {
            return Some("the output has no power-of-two column domain");
        }
        None
    }

    fn encode_into(&self, writer: &mut CanonicalWriter) {
        writer.put_u64(self.positions as u64);
        writer.put_u64(self.columns);
    }
}

/// One stage of the trusted multi-stage IOPP chain. The first stage is the
/// caller's group; each later stage is made by a code switch. The fields are
/// plan data, never proof bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IoppStagePlan {
    pub main: CodeParams,
    pub mask: CodeParams,
    pub k: u32,
    pub h: u32,
    pub positions: usize,
    pub samples: u32,
    pub work_bits: u8,
}

pub(crate) fn iopp_stage_fault(
    stages: &[IoppStagePlan],
    first_main: CodeParams,
    first_mask: CodeParams,
    base_positions: usize,
) -> Option<&'static str> {
    if stages.len() < 2 {
        return Some("a multi-stage IOPP needs at least two stages");
    }
    for (index, stage) in stages.iter().enumerate() {
        if build_code_plan(stage.main.log_n, stage.main.ell, stage.main.r).is_err()
            || build_code_plan(stage.mask.log_n, stage.mask.ell, stage.mask.r).is_err()
        {
            return Some("an IOPP stage code is not a code");
        }
        if !stage.main.ell.is_power_of_two() || !stage.mask.ell.is_power_of_two() {
            return Some("an IOPP stage message length is not a power of two");
        }
        if stage.mask.log_n > stage.main.log_n {
            return Some("an IOPP mask code is longer than its main code");
        }
        if stage.samples != 1 {
            return Some("an IOPP stage sample count is not one");
        }
        if stage.positions as u64 > stage.main.r || stage.positions as u64 > stage.mask.r {
            return Some("an IOPP stage position count exceeds a code tail");
        }
        let last = index + 1 == stages.len();
        if last {
            if stage.k != 0 || stage.h != 0 {
                return Some("the last IOPP stage is not k=h=0");
            }
            if stage.positions != base_positions {
                return Some("the last IOPP stage positions differ from the base positions");
            }
        } else {
            if stage.k < 1 || stage.h > stage.k {
                return Some("an IOPP non-last stage has k/h out of range");
            }
            if stage.mask.ell < 2 * stage.k as u64 + 1 - stage.h as u64 {
                return Some("an IOPP mask message is too short for the fold hiding polynomial");
            }
        }
        if index == 0 {
            if stage.main != first_main || stage.mask != first_mask {
                return Some("the first IOPP stage codes differ from the group codes");
            }
        } else {
            let prev = &stages[index - 1];
            let folded = match stage.main.ell.checked_shl(stage.k) {
                Some(value) => value,
                None => return Some("an IOPP stage shift overflows"),
            };
            if folded != prev.main.ell {
                return Some("the IOPP folded main message does not fill the next rows");
            }
            let room = match stage.mask.ell.checked_shl(stage.h) {
                Some(value) => value,
                None => return Some("an IOPP mask shift overflows"),
            };
            let carried = 2 * prev.mask.ell + prev.main.r + 2 * prev.mask.r + 1;
            if room < carried {
                return Some("the IOPP mask rows have no room for the fold masks and randomness");
            }
        }
    }
    None
}

/// The trusted shapes of the two proximity tests and of the folded table
/// message. All members of both groups share one main code and one mask code so
/// that the split path is not needed; a shape that names other codes
/// is rejected instead of silently changing the relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HiddenProximityPlan {
    /// The folded table message of the shared stage.
    pub gu: GuShape,
    /// The shared `Omega` message.
    pub omega: OmegaShapePlan,
    /// The group main code, shared by `G_u`, `Omega` and every `W` oracle.
    pub code: CodeParams,
    /// The fold mask code, shared by both fresh masks.
    pub mask_code: CodeParams,
    pub shared_mask_tree: MaskTreePlan,
    pub own_mask_tree: MaskTreePlan,
    pub shared_fresh: u64,
    pub own_fresh: u64,
    pub column_draws: usize,
    /// The maximum number of table columns one run may read. Every run charges
    /// the same count it draws; a plan whose draw count exceeds the budget is
    /// rejected before any read.
    pub column_budget: usize,
    pub column_work_bits: u8,
    pub shared_base_positions: usize,
    pub own_base_positions: usize,
    /// The trusted multi-stage IOPP stage lists for the two groups. Empty
    /// keeps the current single-stage path.
    pub shared_stages: Vec<IoppStagePlan>,
    pub own_stages: Vec<IoppStagePlan>,
    /// When true, the own two-stage list is re-derived by the trusted planner
    /// and must match this plan. The toy fixtures leave it false.
    pub planner_checked: bool,
}

impl HiddenProximityPlan {
    pub fn fault(
        &self,
        send: &HiddenSendPlan,
        closure: Option<&ClosurePlan>,
        norm: Option<&NormPlan>,
    ) -> Option<&'static str> {
        if self.gu.fault().is_some() || self.omega.fault().is_some() {
            return Some("a proximity message shape is not valid");
        }
        if build_code_plan(self.code.log_n, self.code.ell, self.code.r).is_err()
            || build_code_plan(
                self.mask_code.log_n,
                self.mask_code.ell,
                self.mask_code.r,
            )
            .is_err()
        {
            return Some("a proximity code is not a code");
        }
        if self.gu.u.code != self.code || self.omega.u.code != self.code || send.w.code != self.code
        {
            return Some("the proximity members do not share one main code");
        }
        if let Some(norm) = norm {
            for group in [&norm.x, &norm.sum, &norm.squares, &norm.mean, &norm.variance] {
                if group.shape.code != self.code || group.group != send.group {
                    return Some("a norm group does not share the proximity group");
                }
            }
            if let Some(mask) = &norm.mask {
                if mask.mask_code != self.mask_code || mask.group != send.group {
                    return Some("the norm mask does not share the proximity mask group");
                }
            }
            if let Some(lookup) = &norm.lookup {
                if lookup.commit.bits_shape.code != self.code
                    || lookup.commit.mask_code != self.mask_code
                    || lookup.commit.group != send.group
                {
                    return Some("the lookup messages do not share the proximity groups");
                }
            }
        }
        if self.gu.tree.log_n_zk != self.mask_code.log_n
            || self.omega.tree.log_n_zk != self.mask_code.log_n
            || self.mask_code.log_n != self.code.log_n
        {
            return Some("a proximity member tree does not match the fold mask code");
        }
        if self.gu.tree.log_n != self.code.log_n
            || self.omega.tree.log_n != self.code.log_n
        {
            return Some("a proximity member tree has another depth than its code");
        }
        if self.gu.u.main_rows == 0
            || !self.gu.u.main_rows.is_power_of_two()
            || self.omega.u.main_rows == 0
            || !self.omega.u.main_rows.is_power_of_two()
            || send.w.main_rows == 0
            || !send.w.main_rows.is_power_of_two()
        {
            return Some("a proximity member does not have power-of-two main rows");
        }
        if self.gu.u.mask_rows != 0
            || self.omega.u.mask_rows != 0
            || self.gu.u.answers != 1
            || self.omega.u.answers != 1
            || send.w.answers != 1
        {
            return Some("a proximity member is not a maskless single-answer message");
        }
        // The hiding budget of every member is the union of its distinct
        // opened positions against the code tail rank. The declared counts are
        // summed here as a conservative pre-check; the runtime ledger checks
        // the exact union of SEND, BASE and any later opening.
        let own_base = self.own_base_positions as u64;
        let shared_base = self.shared_base_positions as u64;
        // A staged group opens its original members at the stage-0 switch,
        // not at BASE; the last stage's own codes bound the BASE openings.
        // The single-stage checks below therefore apply only to an empty stage
        // list. The staged form checks the stage-0 opening against the member
        // and fresh-mask tails, and `iopp_stage_fault` checks the last stage's
        // positions against the last stage codes.
        let own_switch = self
            .own_stages
            .first()
            .map(|stage| stage.positions as u64)
            .unwrap_or(own_base);
        let shared_switch = self
            .shared_stages
            .first()
            .map(|stage| stage.positions as u64)
            .unwrap_or(shared_base);
        if self.own_stages.is_empty() {
            if send.openings_per_oracle as u64 + own_base > send.w.code.r {
                return Some("the W SEND and own-BASE openings exceed the code tail");
            }
            if let Some(index) = &send.index_mask {
                if index.openings as u64 + own_base > index.mask_code.r {
                    return Some("the index-mask SEND and own-BASE openings exceed the mask tail");
                }
            }
            if let Some(dot) = &send.dot {
                if own_base > dot.shape.code.r {
                    return Some("the shown-map own-BASE openings exceed the code tail");
                }
            }
            if own_base > self.mask_code.r {
                return Some("the fresh-mask BASE openings exceed the mask tail");
            }
        } else {
            if send.openings_per_oracle as u64 + own_switch > send.w.code.r
                || own_switch > self.code.r
                || own_switch > self.mask_code.r
            {
                return Some("the own stage-0 switch openings exceed a group tail");
            }
            if let Some(index) = &send.index_mask {
                if index.openings as u64 + own_switch > index.mask_code.r {
                    return Some("the index-mask stage-0 openings exceed the mask tail");
                }
            }
            if let Some(dot) = &send.dot {
                if own_switch > dot.shape.code.r {
                    return Some("the shown-map stage-0 openings exceed the code tail");
                }
            }
        }
        if self.shared_stages.is_empty() {
            if shared_base + own_base > self.code.r {
                return Some("the Omega shared and own-BASE openings exceed the code tail");
            }
            if shared_base > self.code.r {
                return Some("the G_u shared-BASE openings exceed the code tail");
            }
            if shared_base > self.mask_code.r {
                return Some("the fresh-mask BASE openings exceed the mask tail");
            }
        } else if shared_switch > self.code.r || shared_switch > self.mask_code.r {
            return Some("the shared stage-0 switch openings exceed a group tail");
        }
        if self.column_budget as u64 > send.table_shape.code.r {
            return Some("the table column budget exceeds the table code tail");
        }
        if let Some(closure) = closure {
            let own_open = if self.own_stages.is_empty() {
                own_base
            } else {
                own_switch
            };
            if closure.slack_openings as u64 + own_open > closure.slack.code.r {
                return Some("the slack SEND and own openings exceed the code tail");
            }
            if closure.bits_openings as u64 + own_open > closure.bits.code.r {
                return Some("the packed-bit SEND and own openings exceed the code tail");
            }
            if closure.map_openings as u64 + own_open > closure.map.code.r {
                return Some("the product-map SEND and own openings exceed the code tail");
            }
            if closure.mask.openings as u64 + own_open > closure.mask.mask_code.r {
                return Some("the closure-mask SEND and own openings exceed the mask tail");
            }
        }
        if !self.shared_stages.is_empty() || !self.own_stages.is_empty() {
            if proximity_iopp_plan(self, send, closure, norm, ProximityGroupRole::Shared).is_err() {
                return Some("the shared IOPP chain cannot be derived from the trusted plan");
            }
            if proximity_iopp_plan(self, send, closure, norm, ProximityGroupRole::Own).is_err() {
                return Some("the own IOPP chain cannot be derived from the trusted plan");
            }
        }
        if self.planner_checked {
            let own = &self.own_stages;
            if own.len() != 2
                || own[0].main != self.code
                || own[0].mask != self.mask_code
                || own[0].k == 0
                || own[0].h > own[0].k
                || own[0].samples != 1
                || own[1].k != 0
                || own[1].h != 0
                || own[1].samples != 1
            {
                return Some("the planner-checked own stage list is not the two-stage shape");
            }
            let mut max_zero = self.code.r.min(self.mask_code.r);
            max_zero = max_zero
                .checked_sub(send.openings_per_oracle as u64)
                .unwrap_or(0);
            if let Some(index) = &send.index_mask {
                let room = index.mask_code.r.saturating_sub(index.openings as u64);
                max_zero = max_zero.min(room);
            }
            if let Some(closure) = closure {
                let room = closure
                    .mask
                    .mask_code
                    .r
                    .saturating_sub(closure.mask.openings as u64);
                max_zero = max_zero.min(room);
            }
            let planned = match crate::proof::proximity_planner::plan_own_two_stage(
                self.code,
                self.mask_code,
                own[1].main,
                own[1].mask,
                own[0].k,
                own[0].h,
                max_zero as usize,
            ) {
                Ok(planned) => planned,
                Err(_) => return Some("the trusted planner rejects the own two-stage stage list"),
            };
            if own[0].positions != planned.stage_zero_positions
                || own[0].work_bits != planned.stage_zero_work_bits
                || own[1].positions != planned.last_positions
                || own[1].main.r != planned.last_main_r
                || own[1].mask.r != planned.last_mask_r
                || self.own_base_positions != planned.last_positions
            {
                return Some("the own stage list differs from the trusted planner");
            }
        }
        // The shared group is a join over table-derived rows that the table
        // column openings already authenticate. Its two positions and zero
        // work are a fixed join count, not a proximity parameter: the
        // proximity guarantee comes from the table column draws. Any other
        // count changes the plan semantics and is rejected here.
        let shared = &self.shared_stages;
        if self.planner_checked
            && !shared.is_empty()
            && (shared.len() != 2
            || self.shared_base_positions != 2
            || shared[0].main != self.code
            || shared[0].mask != self.mask_code
            || shared[0].k != 2
            || shared[0].h != 0
            || shared[0].samples != 1
            || shared[0].positions != 2
            || shared[0].work_bits != 0
            || shared[1].k != 0
            || shared[1].h != 0
            || shared[1].samples != 1
            || shared[1].positions != 2
                || shared[1].work_bits != 0)
        {
            return Some("the shared join stage list differs from the trusted plan");
        }
        if self.shared_mask_tree.fault().is_some() || self.own_mask_tree.fault().is_some() {
            return Some("a fresh mask tree is not valid");
        }
        let mut own_mask_rows = match &send.index_mask {
            Some(index) => match index.tree_rows() {
                Ok(rows) => rows,
                Err(_) => return Some("the index-mask row count overflows"),
            },
            None => 0,
        };
        if let Some(closure) = closure {
            own_mask_rows = match own_mask_rows.checked_add(match closure.mask.tree_rows() {
                Ok(rows) => rows,
                Err(_) => return Some("the closure-mask row count overflows"),
            }) {
                Some(rows) => rows,
                None => return Some("the own mask row count overflows"),
            };
        }
        let mut norm_rows = match norm.and_then(|norm| norm.mask.as_ref()) {
            Some(mask) => mask.mask_rows,
            None => 0,
        };
        if let Some(lookup) = norm.and_then(|norm| norm.lookup.as_ref()) {
            norm_rows = match norm_rows.checked_add(lookup.commit.mask_rows) {
                Some(rows) => rows,
                None => return Some("the lookup mask row count overflows"),
            };
        }
        if let Some(rstd) = norm.and_then(|norm| norm.rstd.as_ref()) {
            norm_rows = match norm_rows.checked_add(rstd.commit.mask_rows) {
                Some(rows) => rows,
                None => return Some("the rstd mask row count overflows"),
            };
        }
        let own_mask_rows = match crate::proof::hidden_norm_psc::norm_own_mask_rows(
            own_mask_rows,
            norm_rows,
        ) {
            Ok(rows) => rows,
            Err(_) => return Some("the own mask row count overflows"),
        };
        if self.shared_mask_tree.mask_rows != 1
            || self.own_mask_tree.mask_rows != own_mask_rows
            || self.shared_mask_tree.log_n != self.mask_code.log_n
            || self.own_mask_tree.log_n != self.mask_code.log_n
        {
            return Some("a fresh mask tree does not match the fold mask code");
        }
        if let Some(index) = &send.index_mask {
            if index.mask_code != self.mask_code
                || index.group != send.group
                || index.oracle == self.gu.oracle
                || index.oracle == self.omega.oracle
                || index.tree.tree_number == self.gu.tree.tree_number
                || index.tree.tree_number == self.omega.tree.tree_number
                || index.tree.tree_number == self.shared_mask_tree.tree_number
                || index.tree.tree_number == self.own_mask_tree.tree_number
            {
                return Some("the index-mask member does not match the own group");
            }
            match send.w.tree_number.checked_add(send.activation.positions as u64) {
                Some(end)
                    if index.tree.tree_number >= send.w.tree_number
                        && index.tree.tree_number < end =>
                {
                    return Some("the index-mask tree collides with a W tree");
                }
                Some(_) => {}
                None => return Some("the W tree range overflows"),
            }
        }
        if self.shared_mask_tree.tree_number == self.own_mask_tree.tree_number {
            return Some("the two fresh masks share one tree number");
        }
        if let Some(dot) = &send.dot {
            if dot.shape.code != self.code
                || dot.oracle == self.gu.oracle
                || dot.oracle == self.omega.oracle
                || dot.shape.tree_number == self.gu.tree.tree_number
                || dot.shape.tree_number == self.omega.tree.tree_number
                || dot.shape.tree_number == self.shared_mask_tree.tree_number
                || dot.shape.tree_number == self.own_mask_tree.tree_number
            {
                return Some("the shown-map message does not match the own group");
            }
        }
        if let Some(closure) = closure {
            if closure.mask.mask_code != self.mask_code
                || closure.mask.group != send.group
                || closure.slack.code != self.code
                || closure.bits.code != self.code
            {
                return Some("the closure messages do not match the own group");
            }
            let positions = send.positions() as u64;
            let mut oracle_ranges = vec![
                (send.oracle_base, send.oracle_base.saturating_add(positions)),
                (
                    closure.slack_oracle_base,
                    closure.slack_oracle_base.saturating_add(positions),
                ),
                (
                    closure.bits_oracle_base,
                    closure.bits_oracle_base.saturating_add(positions),
                ),
                (
                    closure.map_oracle_base,
                    closure.map_oracle_base.saturating_add(positions),
                ),
                (self.gu.oracle, self.gu.oracle.saturating_add(1)),
                (self.omega.oracle, self.omega.oracle.saturating_add(1)),
                (closure.mask.oracle, closure.mask.oracle.saturating_add(1)),
            ];
            if let Some(index) = &send.index_mask {
                oracle_ranges.push((index.oracle, index.oracle.saturating_add(1)));
            }
            if let Some(dot) = &send.dot {
                oracle_ranges.push((dot.oracle, dot.oracle.saturating_add(1)));
            }
            for first in 0..oracle_ranges.len() {
                for second in first + 1..oracle_ranges.len() {
                    let (a0, a1) = oracle_ranges[first];
                    let (b0, b1) = oracle_ranges[second];
                    if a1 == 0 || b1 == 0 || (a0 < b1 && b0 < a1) {
                        return Some("a closure oracle collides with another oracle");
                    }
                }
            }
            let mut tree_ranges = vec![
                (send.w.tree_number, send.w.tree_number.saturating_add(positions)),
                (
                    closure.slack.tree_number,
                    closure.slack.tree_number.saturating_add(positions),
                ),
                (
                    closure.bits.tree_number,
                    closure.bits.tree_number.saturating_add(positions),
                ),
                (
                    closure.map.tree_number,
                    closure.map.tree_number.saturating_add(positions),
                ),
                (
                    self.gu.tree.tree_number,
                    self.gu.tree.tree_number.saturating_add(1),
                ),
                (
                    self.omega.tree.tree_number,
                    self.omega.tree.tree_number.saturating_add(1),
                ),
                (
                    self.shared_mask_tree.tree_number,
                    self.shared_mask_tree.tree_number.saturating_add(1),
                ),
                (
                    self.own_mask_tree.tree_number,
                    self.own_mask_tree.tree_number.saturating_add(1),
                ),
                (
                    closure.mask.tree.tree_number,
                    closure.mask.tree.tree_number.saturating_add(1),
                ),
            ];
            if let Some(index) = &send.index_mask {
                tree_ranges.push((index.tree.tree_number, index.tree.tree_number.saturating_add(1)));
            }
            if let Some(dot) = &send.dot {
                tree_ranges.push((dot.shape.tree_number, dot.shape.tree_number.saturating_add(1)));
            }
            for first in 0..tree_ranges.len() {
                for second in first + 1..tree_ranges.len() {
                    let (a0, a1) = tree_ranges[first];
                    let (b0, b1) = tree_ranges[second];
                    if a1 == 0 || b1 == 0 || (a0 < b1 && b0 < a1) {
                        return Some("a closure tree collides with another tree");
                    }
                }
            }
        }
        if self.shared_fresh == self.own_fresh {
            return Some("the two fresh masks share one oracle id");
        }
        if self.shared_fresh < 2 {
            return Some("the shared fresh mask id collides with a group member id");
        }
        // Upper bound of the own BAT member set: one W member per position,
        // one Omega member, optional Dot and index-mask members, three
        // closure members per position and one closure-mask member.
        let own_member_bound = (send.positions() as u64)
            .checked_mul(4)
            .and_then(|value| value.checked_add(4))
            .unwrap_or(u64::MAX);
        if self.own_fresh < own_member_bound {
            return Some("the own fresh mask id collides with a group member id");
        }
        if self.column_draws == 0 || self.column_budget == 0 {
            return Some("the shared stage has no table column budget");
        }
        if self.column_draws > self.column_budget {
            return Some("the shared stage draw count exceeds its column budget");
        }
        if self.column_work_bits > 64 {
            return Some("the column work target does not fit");
        }
        if self.shared_base_positions == 0 || self.own_base_positions == 0 {
            return Some("a proximity test has no BASE position");
        }
        None
    }
}

/// The complete trusted plan of one merged three-stage run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HiddenLinPlan {
    pub send: HiddenSendPlan,
    pub output: HiddenOutputShape,
    /// The public row weights of every position, one vector of `matrix_rows`
    /// entries per position. They are plan values, never proof bytes. The list
    /// is empty in the hidden-input mode: the verifier does not receive the
    /// activation rows, and the first statement binds the activation
    /// commitment instead of a weights digest.
    pub weights: Vec<Vec<Fp3>>,
    /// The public claim masks. The hidden path supports the empty list; a
    /// non-empty list needs a mask oracle message that this session does not
    /// have and is rejected.
    pub masks: Vec<Vec<LinMask>>,
    pub proximity: HiddenProximityPlan,
    /// The canonical digest of the trusted output relation declaration. It is
    /// absorbed into the first statement so a swapped mask, validity range,
    /// argmax rule or expected output changes the first query and the seals.
    /// The diagnostic path leaves it empty; the formal hidden-input path
    /// requires it.
    pub output_relation: Option<Digest>,
    /// The exact integer range and comparison closure. The formal
    /// hidden-input path requires it; the diagnostic path leaves it empty.
    pub closure: Option<ClosurePlan>,
    /// The integrated norm relation. The trusted fixture input is the
    /// product-form x group; the sum/squares/mean/variance groups are
    /// proof-fixed witness messages. `None` keeps the existing layout.
    pub norm: Option<NormPlan>,
}

impl HiddenLinPlan {
    pub fn validate(&self) -> Result<()> {
        self.send.validate()?;
        if self.output.fault().is_some()
            || self.output.positions != self.send.activation.positions
            || self.output.columns != self.send.columns
        {
            return Err(Error::InvalidConfig);
        }
        if self.masks.len() != self.send.activation.positions as usize {
            return Err(Error::ShapeMismatch);
        }
        if let Some(norm) = &self.norm {
            norm.validate()?;
            if norm.shape.positions != self.send.activation.positions {
                return Err(Error::ShapeMismatch);
            }
        }
        if self.send.activation_source.is_norm_hd() {
            let norm = self.norm.as_ref().ok_or(Error::InvalidConfig)?;
            let output = norm.output.as_ref().ok_or(Error::InvalidConfig)?;
            let out = &output.out;
            if u64::from(norm.shape.valid) != self.send.matrix_rows
                || norm.columns < self.send.matrix_rows
                || out.columns != norm.columns
                || out.oracle_base == 0
                || out.shape.fault(out.columns, norm.shape.positions).is_some()
                || self.send.activation.hidden as u64 != self.send.matrix_rows
            {
                return Err(Error::InvalidConfig);
            }
        }
        // An empty weights list selects the hidden-input mode. The verifier
        // then has no plaintext row weights and takes the Eval targets from
        // the proof message; the committed activation root, which is already
        // part of the first statement, is the only public binding of `X`.
        if !self.weights.is_empty()
            && self.weights.len() != self.send.activation.positions as usize
        {
            return Err(Error::ShapeMismatch);
        }
        let rows = usize::try_from(self.send.matrix_rows).map_err(|_| Error::ResourceLimit)?;
        if self.weights.iter().any(|weights| weights.len() != rows) {
            return Err(Error::ShapeMismatch);
        }
        if self.masks.iter().any(|masks| !masks.is_empty()) {
            return Err(Error::UnsupportedPrimitive);
        }
        match (&self.send.index_mask, self.weights.is_empty()) {
            (Some(_), true) => {}
            (None, false) => {}
            (Some(_), false) => return Err(Error::InvalidConfig),
            (None, true) => return Err(Error::InvalidConfig),
        }
        if let Some(reason) = self
            .proximity
            .fault(&self.send, self.closure.as_ref(), self.norm.as_ref())
        {
            let _ = reason;
            return Err(Error::InvalidConfig);
        }
        if self.output_relation == Some(Digest::ZERO) {
            return Err(Error::InvalidConfig);
        }
        match (&self.closure, self.weights.is_empty(), &self.output_relation) {
            (Some(closure), true, Some(_)) => {
                if closure.fault(&self.send).is_some() {
                    return Err(Error::InvalidConfig);
                }
                if closure.mask.mask_code != self.proximity.mask_code {
                    return Err(Error::InvalidConfig);
                }
                if closure.winner.len() != self.send.activation.positions as usize {
                    return Err(Error::ShapeMismatch);
                }
            }
            (None, false, _) => {}
            (None, true, None) => {}
            _ => return Err(Error::InvalidConfig),
        }
        Ok(())
    }

    fn parameters_digest(&self) -> Digest {
        let base = match &self.closure {
            Some(closure) => {
                let mut writer = CanonicalWriter::new();
                writer.put_bytes(&self.send.parameters_digest().0);
                closure.encode_into(&mut writer);
                writer.put_digest(&closure.parameters_digest());
                hash_labeled(tag_word(PARAMETERS, 5), 0, None, &writer.into_bytes())
                    .unwrap_or(Digest::ZERO)
            }
            None => self.send.parameters_digest(),
        };
        match &self.norm {
            Some(norm) => {
                let mut writer = CanonicalWriter::new();
                writer.put_bytes(&base.0);
                writer.put_digest(&norm.digest());
                hash_labeled(tag_word(PARAMETERS, 6), 0, None, &writer.into_bytes())
                    .unwrap_or(Digest::ZERO)
            }
            None => base,
        }
    }

    fn weights_digest(&self) -> Result<Digest> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(self.weights.len() as u64);
        for weights in &self.weights {
            writer.put_u64(weights.len() as u64);
            for value in weights {
                writer.put_fp3(value);
            }
        }
        writer.put_u64(self.masks.len() as u64);
        for masks in &self.masks {
            writer.put_u64(masks.len() as u64);
            for mask in masks {
                writer.put_fp3(&mask.value);
                writer.put_fp3(&mask.coefficient);
            }
        }
        hash_labeled(
            tag_word(PARAMETERS, 1),
            0,
            None,
            &writer.into_bytes(),
        )
    }

    fn first_statement(&self) -> Result<Vec<u8>> {
        let mut writer = CanonicalWriter::new();
        writer.put_bytes(&self.send.statement_bytes());
        self.output.encode_into(&mut writer);
        writer.put_digest(&self.weights_digest()?);
        writer.put_u64(self.proximity.gu.oracle);
        writer.put_u64(self.proximity.omega.oracle);
        let gu = self.proximity.gu.encode();
        writer.put_bytes(&gu);
        writer.put_u64(self.proximity.omega.oracle);
        writer.put_u64(self.proximity.omega.group);
        write_tree_plan_public(&mut writer, &self.proximity.omega.tree);
        writer.put_u64(self.proximity.omega.matrix_rows);
        writer.put_u64(self.proximity.omega.k_lo as u64);
        put_code_public(&mut writer, self.proximity.code);
        put_code_public(&mut writer, self.proximity.mask_code);
        writer.put_u64(self.proximity.shared_mask_tree.tree_number);
        writer.put_u64(self.proximity.own_mask_tree.tree_number);
        writer.put_u64(self.proximity.shared_fresh);
        writer.put_u64(self.proximity.own_fresh);
        writer.put_u64(self.proximity.column_draws as u64);
        writer.put_u64(self.proximity.column_work_bits as u64);
        writer.put_u64(self.proximity.shared_base_positions as u64);
        writer.put_u64(self.proximity.own_base_positions as u64);
        match &self.output_relation {
            Some(digest) => {
                writer.put_u64(1);
                writer.put_digest(digest);
            }
            None => writer.put_u64(0),
        }
        if let Some(closure) = &self.closure {
            writer.put_u64(1);
            closure.encode_into(&mut writer);
        }
        Ok(writer.into_bytes())
    }
}

pub(crate) fn put_code_public(writer: &mut CanonicalWriter, code: CodeParams) {
    writer.put_u64(code.log_n as u64);
    writer.put_u64(code.ell);
    writer.put_u64(code.r);
}

fn write_tree_plan_public(writer: &mut CanonicalWriter, plan: &RuntimeTreePlan) {
    writer.put_u64(plan.tree_number);
    writer.put_u64(plan.leaf_class as u64);
    writer.put_u64(plan.log_n as u64);
    writer.put_u64(plan.log_n_zk as u64);
    writer.put_u64(plan.parts.len() as u64);
    for part in &plan.parts {
        writer.put_u64(part.main_symbols);
        writer.put_u64(part.mask_symbols);
    }
    writer.put_u64(plan.salt_bytes as u64);
    writer.put_digest(&plan.run);
    writer.put_digest(&plan.configuration);
}

/// The messages of the shared stage. The record and column rows are derived
/// from public data and the session transcript; they are not carried.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HiddenSharedMessages {
    pub gu_root: Digest,
    pub gu_seeds: Vec<SampleSeed>,
    pub gu_answers: Vec<Fp3>,
    pub column_nonce: u64,
    pub column_positions: Vec<u64>,
    pub column_openings: Vec<ColumnOpening>,
    pub omega_root: Digest,
    pub omega_seeds: Vec<SampleSeed>,
    pub omega_answers: Vec<Fp3>,
    pub zeta: Vec<Fp3>,
    pub fold_start: FoldStartMessage,
    pub fold_rounds: Vec<(Fp3, Fp3)>,
    /// The group's end: the single-stage BASE artifact, or the multi-stage
    /// loop proof whose final segment is the LastBase artifact.
    pub tail: ProximityTail,
    pub seal: Digest,
}

/// The messages of the second stage. The adopted `Omega` root and its samples
/// are taken from the verifier's shared-stage state; the message names the
/// values the prover used so the verifier can reject a mismatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HiddenSecondMessages {
    pub omega_root: Digest,
    pub first_query: Digest,
    pub shared_seal: Digest,
    /// The masked Dot right-hand side `tilde_t = W_t(psi*) + g_t` in the
    /// hidden-input mode; empty in the diagnostic mode.
    pub tilde: Vec<Fp3>,
    /// The public residual `R = sum_t chi^t (g_t - c_t)` of the independent
    /// claim and tilde masks. It is absorbed after `chi` and before the
    /// product-sumcheck challenges; the adjusted target is
    /// `sum_t chi^t tilde_t - R`. `None` in the diagnostic mode, where the
    /// equality rows bind the masks instead.
    pub residual: Option<Fp3>,
    /// The masked product-sumcheck transcript of the hidden Dot identity in
    /// the hidden-input mode; empty in the diagnostic mode.
    pub dot: Option<DotProof>,
    /// The booleanity product-sumcheck transcripts of the output closure,
    /// one per position, when the plan carries the closure.
    pub closure: Option<ClosureSecondMessages>,
    /// The norm product-sumcheck transcripts when the plan carries the norm.
    pub norm: Option<NormPscSecond>,
    pub fold_start: FoldStartMessage,
    pub fold_rounds: Vec<(Fp3, Fp3)>,
    /// The group's end: the single-stage BASE artifact, or the multi-stage
    /// loop proof whose final segment is the LastBase artifact.
    pub tail: ProximityTail,
    pub seal: Digest,
}

/// The complete public message of one merged run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HiddenLinMessages {
    pub first: HiddenSendMessages,
    pub first_seal: Digest,
    pub shared: HiddenSharedMessages,
    pub second: HiddenSecondMessages,
}

/// The verifier outcome of one merged run. Every relation row is registered;
/// the proximity outcome records the local BASE result. This is not a
/// production soundness claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HiddenLinOutcome {
    pub activation: ActivationCommitment,
    pub group: SendGroup,
    pub eval_rows: Vec<EvalSampleRow>,
    pub shared_claims: Vec<GuRelationRow>,
    pub own_claims: Vec<GuRelationRow>,
    pub psi: Vec<Vec<Fp3>>,
    pub shared_point: Vec<Fp3>,
    pub zeta: Vec<Fp3>,
    pub first_seal: Digest,
    pub shared_seal: Digest,
    pub seal: Digest,
    pub column_positions: Vec<u64>,
    pub shared_base: ProximityBaseOutcome,
    pub own_base: ProximityBaseOutcome,
    /// The checked Dot transcript binding when the hidden Dot mode is active:
    /// the batching challenge, the public residual and the product-sumcheck
    /// output claims. `None` in the diagnostic mode.
    pub dot: Option<DotRowBinding>,
    /// The checked output-closure bindings when the plan carries the closure.
    pub closure: Option<ClosureRowBinding>,
    /// The checked output product bindings when the plan carries the norm
    /// output: the original column point and the masked claim the verifier
    /// accepted. The claim is the masked value, never the private product
    /// evaluation.
    pub norm_output: Option<Vec<crate::proof::hidden_norm_psc::NormOutputBinding>>,
}

/// The prover's private material for the auxiliary unit checks. It is never
/// serialized and never given to the public verification entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HiddenLinWitness {
    pub send: HiddenSendWitness,
    pub gu_message: Vec<Fp3>,
    pub gu_tail: Vec<Fp3>,
    pub omega_message: Vec<Fp3>,
    pub omega_tail: Vec<Fp3>,
}

/// The prover-side table handle: the committed reader, the row source and the
/// charged column opener. The row source and the opener are the same table
/// object in a real prover; keeping one trait makes a second read path
/// impossible.
pub trait HiddenTableProver: TableRowSource {
    fn reader(&self) -> &IndexReader;
    fn open_columns(&mut self, positions: &[u64]) -> Result<Vec<ColumnOpening>>;
}

/// The prover-side handle of the norm gain/bias parameter tables. Each table
/// is committed once and shared across positions; the verifier only holds the
/// plan readers and checks the drawn openings against them.
pub trait NormParamTableProver {
    /// The read-only row source of one parameter table, folded to its single
    /// row message.
    fn source(&mut self, kind: NormParamKind) -> &mut dyn TableRowSource;
    fn open_columns(
        &mut self,
        kind: NormParamKind,
        positions: &[u64],
    ) -> Result<Vec<ColumnOpening>>;
}

/// One committed member of a proximity group.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ProximityMember {
    bat_oracle: u64,
    source_oracle: u64,
    main_code: CodeParams,
    mask_code: CodeParams,
    main_rows: u64,
    mask_rows: u64,
    answers: u32,
    root: Digest,
    tree_plan: RuntimeTreePlan,
    /// Prover side only: the message rows and the encoding tails.
    message: Vec<Vec<Fp3>>,
    tails: Vec<Vec<Fp3>>,
    /// Prover side only: the mask-table rows and their encoding tails.
    mask_message: Vec<Vec<Fp3>>,
    mask_tails: Vec<Vec<Fp3>>,
}

/// A committed proximity member built by the prover.
struct ProximityMemberProver {
    member: ProximityMember,
    tree: Arc<RuntimeTree>,
}

impl ProximityMemberProver {
    fn sealed(member: ProximityMember, tree: Arc<RuntimeTree>) -> Result<Self> {
        if *tree.root() != member.root {
            return Err(Error::StatementMismatch);
        }
        Ok(Self { member, tree })
    }
}

/// The public messages of one proximity test.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ProximityMessages {
    fold_start: FoldStartMessage,
    fold_rounds: Vec<(Fp3, Fp3)>,
    tail: ProximityTail,
}

/// The end of one proximity group: the single-stage BASE artifact, or the
/// multi-stage loop proof. Which one is legal is decided by the trusted plan;
/// the decoder rejects a tail that disagrees with the plan's stage list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProximityTail {
    /// The group ended in the single-stage BASE test, which uses an empty
    /// stage list.
    Single {
        base: BaseMessages,
        base_rows: BaseFinalRows,
        base_openings: BaseOpenings,
    },
    /// The group ended in the multi-stage loop; the loop proof's last segment
    /// is its canonical LastBase artifact.
    Stages(IoppProof),
}

/// The verifier-side result of one proximity test.
struct ProximityVerification {
    seal: Digest,
    base: ProximityBaseOutcome,
}

/// The verified end of one proximity group. `Single` carries the single-stage
/// BASE verdict; `Stages` records that the loop replayed every switch, every
/// stage FOLD and the final LastBase artifact and returned its seal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProximityBaseOutcome {
    Single(BaseFinalOutcome),
    Stages { last_positions: Vec<u64> },
}

impl ProximityBaseOutcome {
    /// The single-stage BASE verdict, when the group ended in BASE.
    pub fn single(&self) -> Option<&BaseFinalOutcome> {
        match self {
            Self::Single(base) => Some(base),
            Self::Stages { .. } => None,
        }
    }

    /// True when the group ended in the verified multi-stage loop.
    pub fn is_stages(&self) -> bool {
        matches!(self, Self::Stages { .. })
    }

    /// The last-stage BASE positions, when the group ended in the loop.
    pub fn last_positions(&self) -> Option<&[u64]> {
        match self {
            Self::Single(_) => None,
            Self::Stages { last_positions } => Some(last_positions),
        }
    }
}

/// The prover-side result of one proximity test.
struct ProximityProof {
    messages: ProximityMessages,
    seal: Digest,
}

fn member_fold_shape(
    code: CodeParams,
    mask_code: CodeParams,
    members: &[ProximityMember],
    mask_leaf_class: u32,
) -> Result<FoldShape> {
    if members.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let mut main_rows = 0u64;
    let mut mask_rows = 0u64;
    let mut last = None;
    for member in members {
        if let Some(previous) = last {
            if member.bat_oracle <= previous {
                return Err(Error::InvalidConfig);
            }
        }
        last = Some(member.bat_oracle);
        if member.main_rows > 0 && !member.main_rows.is_power_of_two() {
            return Err(Error::InvalidConfig);
        }
        main_rows = main_rows
            .checked_add(member.main_rows)
            .ok_or(Error::ResourceLimit)?;
        // BAT and FOLD place every member in a power-of-two block, so the
        // group's mask space is the sum of the padded member blocks.
        let block = if member.mask_rows == 0 {
            0
        } else {
            1u64.checked_shl(ceil_log2_u64(member.mask_rows))
                .ok_or(Error::ResourceLimit)?
        };
        mask_rows = mask_rows.checked_add(block).ok_or(Error::ResourceLimit)?;
    }
    let k = if main_rows <= 1 {
        1
    } else {
        64 - (main_rows - 1).leading_zeros()
    };
    let h = ceil_log2_u64(mask_rows.max(1));
    let k = k.max(h).max(1);
    let shape = FoldShape {
        code,
        mask_code,
        k,
        h,
        mask_leaf_class,
    };
    if fold_mask_shape_fault(&shape).is_some() {
        return Err(Error::InvalidConfig);
    }
    Ok(shape)
}

fn member_bat_shape(member: &ProximityMember) -> BatOracleShape {
    BatOracleShape {
        oracle: member.bat_oracle,
        main_rows: member.main_rows,
        mask_rows: member.mask_rows,
        main_code: if member.main_rows > 0 {
            Some(member.main_code)
        } else {
            None
        },
        mask_code: if member.mask_rows > 0 {
            Some(member.mask_code)
        } else {
            None
        },
        answers: member.answers,
    }
}

/// Reorders a row-major member table (`row * ell + place`) into BAT's virtual
/// order (`row + place * 2^row_bits`), zero-padding virtual rows past the real
/// rows. Sample rows and relation rows are built row-major; BAT terms are
/// column-major.
pub(crate) fn to_bat_table(values: &[Fp3], rows: u64, ell: u64) -> Result<Vec<Fp3>> {
    let bits = if rows <= 1 {
        0
    } else {
        64 - (rows - 1).leading_zeros()
    };
    let length = 1usize
        .checked_shl(bits.checked_add(ell.trailing_zeros()).ok_or(Error::ResourceLimit)?)
        .ok_or(Error::ResourceLimit)?;
    let expected = usize::try_from(rows.checked_mul(ell).ok_or(Error::ResourceLimit)?)
        .map_err(|_| Error::ResourceLimit)?;
    if values.len() != expected {
        return Err(Error::ShapeMismatch);
    }
    let mut out = vec![Fp3::ZERO; length];
    let stride = 1usize << bits;
    let ell = usize::try_from(ell).map_err(|_| Error::ResourceLimit)?;
    let real_rows = usize::try_from(rows).map_err(|_| Error::ResourceLimit)?;
    for row in 0..real_rows {
        for place in 0..ell {
            let at = row
                .checked_add(place.checked_mul(stride).ok_or(Error::ResourceLimit)?)
                .ok_or(Error::ResourceLimit)?;
            let slot = out.get_mut(at).ok_or(Error::ShapeMismatch)?;
            *slot = values[row * ell + place];
        }
    }
    Ok(out)
}

fn adapt_sample_row(
    row: &GuRelationRow,
    member: &ProximityMember,
) -> Result<BatRow> {
    if row.oracle != member.source_oracle {
        return Err(Error::StatementMismatch);
    }
    if row.kind != GuRelationKind::Sample
        || row.main.len() as u64 != member.main_rows * member.main_code.ell
        || row.mask.len() as u64 != member.mask_rows * member.mask_code.ell
    {
        return Err(Error::ShapeMismatch);
    }
    let mut out = BatRow::new(BatRowKind::Sample);
    if member.main_rows > 0 {
        out.add_term(
            member.bat_oracle,
            BatPart::Main,
            to_bat_table(&row.main, member.main_rows, member.main_code.ell)?,
        );
    } else if !row.main.is_empty() {
        return Err(Error::ShapeMismatch);
    }
    if member.mask_rows > 0 {
        out.add_term(
            member.bat_oracle,
            BatPart::Mask,
            to_bat_table(&row.mask, member.mask_rows, member.mask_code.ell)?,
        );
    } else if !row.mask.is_empty() {
        return Err(Error::ShapeMismatch);
    }
    out.target = row.target;
    Ok(out)
}

/// Builds the group main table in the FOLD layout order from the committed
/// member messages.
fn build_group_table(
    layout: &FoldGroupLayout,
    bat_layout: &BatGroupLayout,
    members: &[ProximityMember],
) -> Result<Vec<Fp3>> {
    build_group_table_with_min(layout, bat_layout, members, GROUP_TABLE_PARALLEL_MIN)
}

fn build_group_table_with_min(
    layout: &FoldGroupLayout,
    bat_layout: &BatGroupLayout,
    members: &[ProximityMember],
    parallel_min: usize,
) -> Result<Vec<Fp3>> {
    if members.is_empty() {
        return Err(Error::ShapeMismatch);
    }
    let group_bits = bat_layout.group_row_bits(BatPart::Main);
    let group_len = bat_layout.group_len(BatPart::Main)?;
    let mut values = vec![Fp3::ZERO; group_len];
    if layout.members.len() != bat_layout.members.len() || layout.members.len() != members.len() {
        return Err(Error::StatementMismatch);
    }
    let firsts = fold_member_firsts(layout, FoldPart::Main)?;
    let mut placed = Vec::new();
    for (index, (fold_member, bat_member)) in
        layout.members.iter().zip(&bat_layout.members).enumerate()
    {
        if fold_member.oracle != bat_member.oracle || fold_member.kept {
            return Err(Error::UnsupportedPrimitive);
        }
        let member = members.get(index).ok_or(Error::StatementMismatch)?;
        if member.bat_oracle != fold_member.oracle {
            return Err(Error::StatementMismatch);
        }
        if member.message.len() as u64 != fold_member.main_rows
            || member
                .message
                .iter()
                .any(|row| row.len() as u64 != member.main_code.ell)
        {
            return Err(Error::ShapeMismatch);
        }
        if fold_member.main_rows == 0 {
            if !member.message.is_empty() || firsts[index].is_some() {
                return Err(Error::ShapeMismatch);
            }
            continue;
        }
        let first = usize::try_from(firsts[index].ok_or(Error::StatementMismatch)?)
            .map_err(|_| Error::ResourceLimit)?;
        placed.push((first, member.message.as_slice()));
    }
    let chunk_len = 1usize
        .checked_shl(group_bits)
        .ok_or(Error::ResourceLimit)?;
    if chunk_len == 0 || group_len % chunk_len != 0 {
        return Err(Error::ShapeMismatch);
    }
    let write_chunk = |place: usize, chunk: &mut [Fp3]| -> Result<()> {
        for (first, message) in &placed {
            for (row, message_row) in message.iter().enumerate() {
                if let Some(value) = message_row.get(place) {
                    let slot = chunk.get_mut(first + row).ok_or(Error::InvalidIndex)?;
                    *slot = *value;
                }
            }
        }
        Ok(())
    };
    if group_len >= parallel_min {
        values
            .par_chunks_mut(chunk_len)
            .enumerate()
            .try_for_each(|(place, chunk)| write_chunk(place, chunk))?;
    } else {
        for (place, chunk) in values.chunks_mut(chunk_len).enumerate() {
            write_chunk(place, chunk)?;
        }
    }
    Ok(values)
}

/// Group tables switch to the place-block parallel fill at this length.
const GROUP_TABLE_PARALLEL_MIN: usize = 1 << 22;

/// Builds the group mask table in the FOLD layout order from the committed
/// member mask rows. Exactly the members with mask rows occupy mask blocks.
fn build_group_mask(
    layout: &FoldGroupLayout,
    bat_layout: &BatGroupLayout,
    members: &[ProximityMember],
) -> Result<Vec<Fp3>> {
    let group_bits = bat_layout.group_row_bits(BatPart::Mask);
    let group_len = bat_layout.group_len(BatPart::Mask)?;
    let mut values = vec![Fp3::ZERO; group_len];
    if layout.members.len() != bat_layout.members.len() || layout.members.len() != members.len() {
        return Err(Error::StatementMismatch);
    }
    let firsts = fold_member_firsts(layout, FoldPart::Mask)?;
    for (index, (fold_member, bat_member)) in
        layout.members.iter().zip(&bat_layout.members).enumerate()
    {
        if fold_member.oracle != bat_member.oracle || fold_member.kept {
            return Err(Error::UnsupportedPrimitive);
        }
        let member = members.get(index).ok_or(Error::StatementMismatch)?;
        if member.bat_oracle != fold_member.oracle {
            return Err(Error::StatementMismatch);
        }
        if member.mask_message.len() as u64 != fold_member.mask_rows
            || member
                .mask_message
                .iter()
                .any(|row| row.len() as u64 != member.mask_code.ell)
        {
            return Err(Error::ShapeMismatch);
        }
        if fold_member.mask_rows == 0 {
            if !member.mask_message.is_empty() || firsts[index].is_some() {
                return Err(Error::ShapeMismatch);
            }
            continue;
        }
        let first = firsts[index].ok_or(Error::StatementMismatch)?;
        for (row, message) in member.mask_message.iter().enumerate() {
            for (place, value) in message.iter().enumerate() {
                let at = first
                    .checked_add(row as u64)
                    .and_then(|value| value.checked_add((place as u64).checked_shl(group_bits)?))
                    .ok_or(Error::ResourceLimit)?;
                let slot = values.get_mut(at as usize).ok_or(Error::InvalidIndex)?;
                *slot = *value;
            }
        }
    }
    Ok(values)
}

/// The fold coefficients of one variable count, in the layout's row order.
fn fold_coefficients_public(variables: u32, gamma: &[Fp3]) -> Result<Vec<Fp3>> {
    if variables >= 64 || gamma.len() < variables as usize {
        return Err(Error::InvalidConfig);
    }
    let rows = 1usize
        .checked_shl(variables)
        .ok_or(Error::ResourceLimit)?;
    let mut out = Vec::with_capacity(rows);
    for group in 0..rows {
        let mut coefficient = Fp3::ONE;
        for variable in 0..variables as usize {
            let weight = if (group >> variable) & 1 == 1 {
                gamma[variable]
            } else {
                Fp3::ONE.sub(&gamma[variable])
            };
            coefficient = coefficient.mul(&weight);
        }
        out.push(coefficient);
    }
    Ok(out)
}

/// The folded encoding tails: position by position, fold every member's tail
/// row with the transcript's fold coefficients. The fresh mask's tail rows are
/// folded with the mask coefficients.
fn folded_tails(
    handoff: &FoldHandoff,
    members: &[ProximityMember],
    fresh_tails: &[Vec<Fp3>],
) -> Result<(Vec<Fp3>, Vec<Fp3>, Vec<Fp3>)> {
    let layout = &handoff.layout;
    let main_coeff = fold_coefficients_public(handoff.k, &handoff.gamma)?;
    let mask_gamma = &handoff.gamma[..handoff.h as usize];
    let mask_coeff = fold_coefficients_public(handoff.h, mask_gamma)?;
    let main_firsts = fold_member_firsts(layout, FoldPart::Main)?;
    let main_len = usize::try_from(handoff.code.r).map_err(|_| Error::ResourceLimit)?;
    let mask_len = usize::try_from(handoff.mask_code.r).map_err(|_| Error::ResourceLimit)?;
    let mut main_tail = vec![Fp3::ZERO; main_len];
    let mut mask_tail = vec![Fp3::ZERO; mask_len];
    let mask_firsts = fold_member_firsts(layout, FoldPart::Mask)?;
    if members.len() != layout.members.len() {
        return Err(Error::ShapeMismatch);
    }
    for (index, (member, layout_member)) in members.iter().zip(&layout.members).enumerate() {
        if member.bat_oracle != layout_member.oracle {
            return Err(Error::StatementMismatch);
        }
        if member.tails.len() as u64 != layout_member.main_rows {
            return Err(Error::ShapeMismatch);
        }
        if layout_member.main_rows > 0 {
            let main_first = main_firsts[index].ok_or(Error::StatementMismatch)?;
            for (row, tail) in member.tails.iter().enumerate() {
                if tail.len() != main_len {
                    return Err(Error::ShapeMismatch);
                }
                let group = main_first
                    .checked_add(row as u64)
                    .ok_or(Error::ResourceLimit)?;
                let coefficient = *main_coeff
                    .get(group as usize)
                    .ok_or(Error::InvalidIndex)?;
                for (value, tail) in main_tail.iter_mut().zip(tail) {
                    *value = value.add(&coefficient.mul(tail));
                }
            }
        }
        if member.mask_tails.len() as u64 != layout_member.mask_rows {
            return Err(Error::ShapeMismatch);
        }
        if layout_member.mask_rows > 0 {
            let mask_first = mask_firsts[index].ok_or(Error::StatementMismatch)?;
            for (row, tail) in member.mask_tails.iter().enumerate() {
                if tail.len() != mask_len {
                    return Err(Error::ShapeMismatch);
                }
                let group = mask_first
                    .checked_add(row as u64)
                    .ok_or(Error::ResourceLimit)?;
                let coefficient = *mask_coeff
                    .get(group as usize)
                    .ok_or(Error::InvalidIndex)?;
                for (value, tail) in mask_tail.iter_mut().zip(tail) {
                    *value = value.add(&coefficient.mul(tail));
                }
            }
        }
    }
    let fresh_rows = 1usize
        .checked_shl(handoff.h)
        .ok_or(Error::ResourceLimit)?;
    if fresh_tails.len() != fresh_rows {
        return Err(Error::ShapeMismatch);
    }
    let mut fresh_tail = vec![Fp3::ZERO; mask_len];
    for (row, tail) in fresh_tails.iter().enumerate() {
        if tail.len() != mask_len {
            return Err(Error::ShapeMismatch);
        }
        let coefficient = *mask_coeff.get(row).ok_or(Error::InvalidIndex)?;
        for (value, tail) in fresh_tail.iter_mut().zip(tail) {
            *value = value.add(&coefficient.mul(tail));
        }
    }
    Ok((main_tail, mask_tail, fresh_tail))
}

fn draw_salt_bytes(coins: &mut ByteMaskCoins, count: usize, salt_bytes: usize) -> Result<Vec<Vec<u8>>> {
    let mut salts = Vec::with_capacity(count);
    let words = salt_bytes.div_ceil(24);
    for _ in 0..count {
        let mut bytes = Vec::with_capacity(salt_bytes);
        for _ in 0..words {
            let (value, _) = coins.draw()?;
            for coefficient in value.c {
                bytes.extend_from_slice(&coefficient.as_canonical_u64().to_le_bytes());
            }
        }
        bytes.truncate(salt_bytes);
        salts.push(bytes);
    }
    Ok(salts)
}

fn draw_values(coins: &mut ByteMaskCoins, count: usize) -> Result<Vec<Fp3>> {
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(coins.draw()?.0);
    }
    Ok(values)
}

fn mask_rows_of_draft(draft: &MaskDraft, shape: &FoldShape) -> Result<(Vec<Vec<Fp3>>, Vec<Vec<Fp3>>)> {
    let rows = usize::try_from(shape.mask_rows()?).map_err(|_| Error::ResourceLimit)?;
    let ell = usize::try_from(shape.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
    let tail_len = usize::try_from(shape.mask_code.r).map_err(|_| Error::ResourceLimit)?;
    if draft.tail.rows != rows || draft.tail.len != tail_len {
        return Err(Error::ShapeMismatch);
    }
    let mut messages = Vec::with_capacity(rows);
    let mut tails = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut message = Vec::with_capacity(ell);
        for place in 0..ell {
            message.push(draft.table.at(row, place)?);
        }
        messages.push(message);
        let mut tail = Vec::with_capacity(tail_len);
        for place in 0..tail_len {
            tail.push(draft.tail.at(row, place)?);
        }
        tails.push(tail);
    }
    Ok((messages, tails))
}

/// Where the proximity prover takes a member's committed runtime tree. The
/// value groups and W retain the trees built by their commits, so the retained
/// source returns a shared handle without rebuilding; the mask, index-mask and
/// Dot trees are built from their witnesses when they are first needed.
enum ProximityTreeSource<'a> {
    Ready(Vec<(u64, Arc<RuntimeTree>)>),
    Retained {
        witness: &'a HiddenSendWitness,
        shared_witness: &'a SharedWitness,
        first: &'a HiddenSendMessages,
        plan: &'a HiddenLinPlan,
    },
}

impl ProximityTreeSource<'_> {
    fn tree_for(&self, member: &ProximityMember) -> Result<Arc<RuntimeTree>> {
        match self {
            Self::Ready(trees) => trees
                .iter()
                .find(|(oracle, _)| *oracle == member.source_oracle)
                .map(|(_, tree)| Arc::clone(tree))
                .ok_or(Error::ShapeMismatch),
            Self::Retained {
                witness,
                shared_witness,
                first,
                plan,
            } => witness_tree_for(member, witness, shared_witness, first, plan),
        }
    }
}

/// The prover side of one proximity test: BAT over the registered claims,
/// FOLD rounds with a fresh mask, then BASE on the same chain. Every value the
/// verifier needs is in the returned messages; the member rows never leave the
/// prover.
#[allow(clippy::too_many_arguments)]
fn proximity_prove(
    chain: Chain,
    fold: FoldShape,
    members: Vec<ProximityMember>,
    trees: ProximityTreeSource<'_>,
    sample_rows: &[Vec<GuRelationRow>],
    plain_rows: Vec<BatRow>,
    fresh_id: u64,
    mask_tree: MaskTreePlan,
    seed: Digest,
    base_positions: usize,
    iopp: Option<&ProximityIoppPlan>,
) -> Result<ProximityProof> {
    let _profile_proximity = zkie_engine::profile::Scope::new("prox.proximity_prove");
    if members.is_empty() || sample_rows.len() != members.len() {
        return Err(Error::ShapeMismatch);
    }
    let _profile_bat = zkie_engine::profile::Scope::new("prox.bat");
    let mut statement = BatStatement::new();
    for member in &members {
        statement.add_oracle(member_bat_shape(member))?;
    }
    for (index, member) in members.iter().enumerate() {
        let rows = sample_rows[index]
            .iter()
            .map(|row| adapt_sample_row(row, member))
            .collect::<Result<Vec<_>>>()?;
        if rows.len() != member.answers as usize {
            return Err(Error::ShapeMismatch);
        }
        statement.sample(member.bat_oracle, rows)?;
    }
    let group = statement.new_group(fold.code, fold.mask_code)?;
    for member in &members {
        statement.join(group, member.bat_oracle, false)?;
    }
    let planned = plain_rows.len() as u64
        + members.iter().map(|member| member.answers as u64).sum::<u64>();
    for row in plain_rows {
        statement.add_row(group, row)?;
    }
    let layout = statement.layout(group)?;
    let fold_layout = statement.to_fold_layout_checked(group, &fold)?;
    let mut bat = BatSession::new(statement, chain, planned)?;
    let input = bat.begin_run(group)?;
    let combined = bat.finish_run(input)?;
    let fold_row = bat.statement().to_fold_input_row(combined, &layout)?;
    let chain = bat.into_chain()?;
    drop(_profile_bat);

    let _profile_group_tables = zkie_engine::profile::Scope::new("prox.group_tables");
    let group_main = build_group_table(&fold_layout, &layout, &members)?;
    let group_mask = build_group_mask(&fold_layout, &layout, &members)?;
    drop(_profile_group_tables);

    let _profile_fold_start = zkie_engine::profile::Scope::new("prox.fold_start");
    let mut coins = ByteMaskCoins::new(seed)?;
    let mut mask = fold_mask_make(&fold, &mut coins)?;
    fold_mask_make_tail(&fold, &mut mask, &mut coins)?;
    mask.sent_as(fresh_id)?;
    let groups = VerifiedGroups {
        groups: vec![VerifiedGroup {
            group,
            members: members.iter().map(|m| m.bat_oracle).collect(),
            fresh: Some(fresh_id),
            code: Some(fold.code),
            mask_code: Some(fold.mask_code),
        }],
    };
    let notice = fold_mask_notice(&fold)?;
    let oracle = MaskOracle::prover_view(fresh_id, group, &notice);
    zkie_engine::pcs::request::fold::fold_mask_check(&fold, &oracle, &groups)?;
    let salts = draw_salt_bytes(
        &mut coins,
        usize::try_from(mask_tree.columns()).map_err(|_| Error::ResourceLimit)?,
        mask_tree.salt_bytes,
    )?;
    let start = fold_start_prove(
        &fold,
        &mut mask,
        &coins,
        &oracle,
        &groups,
        mask_tree,
        &salts,
    )?;
    let start_message = start.message.clone();
    let (fresh_rows, fresh_tails) = mask_rows_of_draft(&start.draft, &fold)?;
    drop(_profile_fold_start);
    let mut chain = chain;
    absorb_fold_start(&mut chain, &start_message)?;
    let epsilon = chain.draw_e()?;

    let mut claim = FoldRoundVerifier::new(start_message.sum, epsilon, fold_row.target);
    let mut prover = FoldProverSession::new(
        &fold,
        &fold_layout,
        &mask,
        fold_row,
        group_main,
        group_mask,
        Vec::new(),
    )?;
    if prover.sum() != start_message.sum {
        return Err(Error::StatementMismatch);
    }
    prover.start(epsilon, &coins)?;
    let mut gamma = Vec::with_capacity(fold.k as usize);
    let mut rounds = Vec::with_capacity(fold.k as usize);
    let _profile_fold_rounds = zkie_engine::profile::Scope::new("prox.fold_rounds");
    for _ in 0..fold.k {
        let round = prover.round()?;
        absorb_fold_round(&mut chain, round[0], round[2])?;
        let challenge = chain.draw_e()?;
        claim.round(round[0], round[2], challenge);
        prover.bind(challenge)?;
        gamma.push(challenge);
        rounds.push((round[0], round[2]));
    }
    drop(_profile_fold_rounds);
    let _profile_fold_output = zkie_engine::profile::Scope::new("prox.fold_output");
    let output = prover.output(
        &fold_layout,
        fresh_id,
        start_message.root,
        mask_tree.tree_number,
        epsilon,
        start_message.sum,
        &gamma,
        claim.claim(),
    )?;
    let end = prover.end()?;
    let anchor = FoldHandoffAnchor {
        shape: fold,
        group,
        members: members.iter().map(|m| m.bat_oracle).collect(),
        fresh: fresh_id,
        fresh_root: start_message.root,
        tree_number: mask_tree.tree_number,
        layout: fold_layout.clone(),
        fold_gamma: output.gamma.clone(),
        fold_epsilon: output.eps,
        fold_sum: output.sum,
    };
    let handoff = FoldHandoff::from_output(&output, &anchor)?;
    let (main_tail, mask_tail, fresh_tail) =
        folded_tails(&handoff, &members, &fresh_tails)?;
    let basis = FoldBasis {
        layout: handoff.layout.clone(),
        fresh_root: handoff.fresh_root,
        fresh_tree: mask_tree,
        members: members
            .iter()
            .map(|member| FoldMember {
                oracle: member.bat_oracle,
                main_code: fold.code,
                mask_code: fold.mask_code,
                root: member.root,
                tree: member.tree_plan.clone(),
            })
            .collect(),
    };
    if let Some(iopp) = iopp {
        let stage_zero_tree = MaskTree::commit(
            mask_tree,
            &fold,
            &fresh_rows,
            &fresh_tails,
            &salts,
        )?;
        let member_trees = members
            .iter()
            .map(|member| trees.tree_for(member))
            .collect::<Result<Vec<_>>>()?;
        let stage_zero_rows = SingleRows {
            code: handoff.code,
            mask_code: handoff.mask_code,
            main: prover.main_messages().to_vec(),
            main_tail,
            masks: vec![end.mu.clone(), end.fresh.clone()],
            mask_tails: vec![mask_tail, fresh_tail],
            claim: handoff.main_claim.clone(),
            mask_claims: vec![handoff.mask_claim.clone(), handoff.fresh_claim.clone()],
            target: handoff.target,
        };
        drop(prover);
        let private = loop_private_material(&seed, &iopp.chain)?;
        drop(_profile_fold_output);
        let (proof, seal) = prove_multi_stage_iopp(
            chain,
            &iopp.chain,
            IoppStageZero {
                handoff,
                basis,
                rows: stage_zero_rows,
                member_trees,
                old_mask_tree: None,
                fresh_tree: stage_zero_tree,
            },
            &private.switches,
            &private.stage_folds,
            &private.base,
        )?;
        return Ok(ProximityProof {
            messages: ProximityMessages {
                fold_start: start_message,
                fold_rounds: rounds,
                tail: ProximityTail::Stages(proof),
            },
            seal,
        });
    }
    let rows = handoff.single_rows(
        prover.main_messages().to_vec(),
        main_tail,
        end.mu.clone(),
        mask_tail,
    )?;
    let identity = fold_identity(&handoff);
    drop(_profile_fold_output);
    let _profile_base_prover = zkie_engine::profile::Scope::new("prox.base_prover");
    let star_main = draw_values(
        &mut coins,
        usize::try_from(fold.code.ell).map_err(|_| Error::ResourceLimit)?,
    )?;
    let star_tail = draw_values(
        &mut coins,
        usize::try_from(fold.code.r).map_err(|_| Error::ResourceLimit)?,
    )?;
    let star_mask = draw_values(
        &mut coins,
        usize::try_from(fold.mask_code.ell).map_err(|_| Error::ResourceLimit)?,
    )?;
    let star_mask_tail = draw_values(
        &mut coins,
        usize::try_from(fold.mask_code.r).map_err(|_| Error::ResourceLimit)?,
    )?;
    let base_prover = BaseProver::new(
        rows,
        identity,
        end.fresh.clone(),
        fresh_tail,
        star_main,
        star_tail,
        star_mask,
        star_mask_tail,
        Fp3::ZERO,
    )?;
    let mut verifier = BaseFinalSession::new(chain, handoff, basis)?;
    let base_messages = base_prover.messages()?;
    verifier.receive_commitments(base_messages)?;
    let lambda = verifier.draw_lambda()?;
    let base_rows = base_prover.base_rows(lambda)?;
    verifier.receive_base_rows(base_rows.clone())?;
    drop(_profile_base_prover);
    let _profile_base_openings = zkie_engine::profile::Scope::new("prox.base_openings");
    let positions = verifier.draw_base_positions(base_positions)?;
    let unique = unique_sorted(&positions);
    let mut provers = Vec::with_capacity(members.len());
    for mut member in members {
        let tree = trees.tree_for(&member)?;
        member.message = Vec::new();
        member.tails = Vec::new();
        member.mask_message = Vec::new();
        member.mask_tails = Vec::new();
        let prover = ProximityMemberProver::sealed(member, tree)?;
        provers.push(prover);
    }
    provers.sort_by_key(|member| member.member.bat_oracle);
    let member_openings = zkie_engine::scheduling::try_map_bounded(
        provers.as_slice(),
        zkie_engine::scheduling::DEFAULT_UNITS_IN_FLIGHT,
        |member| member.tree.open(&unique),
    )?;
    let mask_positions: Vec<u64> = positions
        .iter()
        .map(|position| position % fold.mask_code.n())
        .collect();
    let fresh_tree = MaskTree::commit(
        mask_tree,
        &fold,
        &fresh_rows,
        &fresh_tails,
        &salts,
    )?;
    let fresh_opening = fresh_tree.open(&unique_sorted(&mask_positions))?;
    let star_openings = base_prover.openings(&positions)?;
    let openings = BaseOpenings {
        star: star_openings.star,
        star_mask: star_openings.star_mask,
        members: member_openings,
        fresh: fresh_opening,
    };
    verifier.verify_openings(&openings)?;
    drop(_profile_base_openings);
    let _profile_base_finish = zkie_engine::profile::Scope::new("prox.base_finish");
    let (seal, _base) = match verifier.finish()? {
        BaseFinalCheck::Passed(base) => (base.check_value, base),
        BaseFinalCheck::Rejected(_) => return Err(Error::InvalidProof),
    };
    Ok(ProximityProof {
        messages: ProximityMessages {
            fold_start: start_message,
            fold_rounds: rounds,
            tail: ProximityTail::Single {
                base: base_messages,
                base_rows,
                base_openings: openings,
            },
        },
        seal,
    })
}

/// The verifier side of one proximity test. It rebuilds the BAT statement from
/// the registered sample and plain rows, reproduces every challenge, checks the
/// FOLD start and rounds, and runs BASE against its own member roots. It never
/// reads a member message row.
#[allow(clippy::too_many_arguments)]
fn proximity_verify(
    chain: Chain,
    fold: FoldShape,
    members: &[ProximityMember],
    sample_rows: &[Vec<GuRelationRow>],
    plain_rows: Vec<BatRow>,
    messages: &ProximityMessages,
    fresh_id: u64,
    mask_tree: MaskTreePlan,
    base_positions: usize,
    iopp: Option<&ProximityIoppPlan>,
) -> Result<ProximityVerification> {
    let _profile = zkie_engine::profile::Scope::new("verify.proximity");
    if members.is_empty() || sample_rows.len() != members.len() {
        return Err(Error::ShapeMismatch);
    }
    let _profile_statement = zkie_engine::profile::Scope::light("verify.proximity.statement");
    let mut statement = BatStatement::new();
    for member in members {
        statement.add_oracle(member_bat_shape(member))?;
    }
    for (index, member) in members.iter().enumerate() {
        let rows = sample_rows[index]
            .iter()
            .map(|row| adapt_sample_row(row, member))
            .collect::<Result<Vec<_>>>()?;
        if rows.len() != member.answers as usize {
            return Err(Error::ShapeMismatch);
        }
        statement.sample(member.bat_oracle, rows)?;
    }
    let group = statement.new_group(fold.code, fold.mask_code)?;
    for member in members {
        statement.join(group, member.bat_oracle, false)?;
    }
    let planned = plain_rows.len() as u64
        + members.iter().map(|member| member.answers as u64).sum::<u64>();
    for row in plain_rows {
        statement.add_row(group, row)?;
    }
    let layout = statement.layout(group)?;
    let fold_layout = statement.to_fold_layout_checked(group, &fold)?;
    drop(_profile_statement);
    let _profile_bat = zkie_engine::profile::Scope::new("verify.proximity.bat");
    let mut bat = BatSession::new(statement, chain, planned)?;
    let input = bat.begin_run(group)?;
    let combined = bat.finish_run(input)?;
    let fold_row = bat.statement().to_fold_input_row(combined, &layout)?;
    let chain = bat.into_chain()?;
    drop(_profile_bat);

    let _profile_fold = zkie_engine::profile::Scope::new("verify.proximity.fold");
    let verified_groups = VerifiedGroups {
        groups: vec![VerifiedGroup {
            group,
            members: members.iter().map(|member| member.bat_oracle).collect(),
            fresh: Some(fresh_id),
            code: Some(fold.code),
            mask_code: Some(fold.mask_code),
        }],
    };
    let metadata = OracleMetadata {
        id: fresh_id,
        group,
        tree_number: mask_tree.tree_number,
        leaf_class: fold.mask_leaf_class,
        mask_rows: mask_tree.mask_rows,
        mask_code: fold.mask_code,
    };
    let verified_statement = VerifiedStatement::new(
        verified_groups,
        vec![mask_tree],
        vec![metadata],
    )?;
    let notice = fold_mask_notice(&fold)?;
    let mut runtime = RuntimeOracles::default();
    runtime.receive_mask(
        &verified_statement,
        &mask_tree,
        fresh_id,
        &notice,
        messages.fold_start.root,
    )?;
    fold_start_verify(
        &fold,
        &messages.fold_start,
        &verified_statement,
        &runtime,
    )?;
    let mut chain = chain;
    absorb_fold_start(&mut chain, &messages.fold_start)?;
    let epsilon = chain.draw_e()?;
    let mut claim = FoldRoundVerifier::new(messages.fold_start.sum, epsilon, fold_row.target);
    let mut gamma = Vec::with_capacity(fold.k as usize);
    if messages.fold_rounds.len() != fold.k as usize {
        return Err(Error::ShapeMismatch);
    }
    for (c0, c2) in &messages.fold_rounds {
        absorb_fold_round(&mut chain, *c0, *c2)?;
        let challenge = chain.draw_e()?;
        claim.round(*c0, *c2, challenge);
        gamma.push(challenge);
    }
    let output = fold_output(
        &fold,
        &fold_layout,
        &fold_row,
        fresh_id,
        messages.fold_start.root,
        mask_tree.tree_number,
        epsilon,
        messages.fold_start.sum,
        &gamma,
        claim.claim(),
    )?;
    let anchor = FoldHandoffAnchor {
        shape: fold,
        group,
        members: members.iter().map(|member| member.bat_oracle).collect(),
        fresh: fresh_id,
        fresh_root: messages.fold_start.root,
        tree_number: mask_tree.tree_number,
        layout: fold_layout.clone(),
        fold_gamma: output.gamma.clone(),
        fold_epsilon: output.eps,
        fold_sum: output.sum,
    };
    let handoff = FoldHandoff::from_output(&output, &anchor)?;
    drop(_profile_fold);
    let _profile_base = zkie_engine::profile::Scope::new("verify.proximity.base");
    let basis = FoldBasis {
        layout: handoff.layout.clone(),
        fresh_root: handoff.fresh_root,
        fresh_tree: mask_tree,
        members: members
            .iter()
            .map(|member| FoldMember {
                oracle: member.bat_oracle,
                main_code: fold.code,
                mask_code: fold.mask_code,
                root: member.root,
                tree: member.tree_plan.clone(),
            })
            .collect(),
    };
    if let Some(iopp) = iopp {
        if iopp.stage_zero_fresh != fresh_id
            || iopp.chain.config.base_positions != base_positions
        {
            return Err(Error::InvalidConfig);
        }
        let proof = match &messages.tail {
            ProximityTail::Stages(proof) => proof,
            ProximityTail::Single { .. } => return Err(Error::StatementMismatch),
        };
        let (seal, last_positions) = verify_multi_stage_iopp_with_last_positions(
            chain,
            &iopp.chain,
            IoppVerifyStageZero {
                handoff,
                basis,
                first_old_mask: None,
            },
            proof,
        )?;
        return Ok(ProximityVerification {
            seal,
            base: ProximityBaseOutcome::Stages { last_positions },
        });
    }
    let ProximityTail::Single {
        base: base_messages,
        base_rows,
        base_openings,
    } = &messages.tail
    else {
        return Err(Error::StatementMismatch);
    };
    let mut verifier = BaseFinalSession::new(chain, handoff, basis)?;
    verifier.receive_commitments(*base_messages)?;
    let lambda = verifier.draw_lambda()?;
    if base_rows.base.lambda != lambda {
        return Err(Error::InvalidProof);
    }
    verifier.receive_base_rows(base_rows.clone())?;
    verifier.draw_base_positions(base_positions)?;
    verifier.verify_openings(base_openings)?;
    let (seal, base) = match verifier.finish()? {
        BaseFinalCheck::Passed(base) => (base.check_value, base),
        BaseFinalCheck::Rejected(_) => return Err(Error::InvalidProof),
    };
    Ok(ProximityVerification {
        seal,
        base: ProximityBaseOutcome::Single(base),
    })
}

fn absorb_fold_start(chain: &mut Chain, message: &FoldStartMessage) -> Result<()> {
    let mut out = chain.message();
    out.push_u64(message.id);
    out.push_u64(message.claimed_group);
    out.push_u64(message.tree_number);
    out.push_u64(message.notice.mask_rows);
    put_code_chain(&mut out, message.notice.mask_code);
    out.push_u64(message.notice.answers as u64);
    out.push_u64(message.notice.leaf_class as u64);
    push_digest_chain(&mut out, &message.root);
    out.push_fp3(&message.sum);
    let prefix = out.prefix(KIND_EPSILON, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn absorb_fold_round(chain: &mut Chain, c0: Fp3, c2: Fp3) -> Result<()> {
    let mut out = chain.message();
    out.push_fp3(&c0);
    out.push_fp3(&c2);
    let prefix = out.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn push_digest_chain(message: &mut ChainMessage, digest: &Digest) {
    for word in digest.0.chunks_exact(8) {
        message.push_u64(u64::from_le_bytes(
            word.try_into().expect("eight-byte word"),
        ));
    }
}

fn put_code_chain(message: &mut ChainMessage, code: CodeParams) {
    message.push_u64(code.log_n as u64);
    message.push_u64(code.ell);
    message.push_u64(code.r);
}


// ---------------------------------------------------------------------------
// Stage helpers
// ---------------------------------------------------------------------------

/// A deterministic sub-seed of the run's private randomness, domain
/// separated by the stage and the purpose.
fn derive_run_seed(master: &Digest, stage: u64, purpose: u64) -> Result<Digest> {
    if *master == Digest::ZERO {
        return Err(Error::InvalidConfig);
    }
    let mut bytes = Vec::with_capacity(56);
    bytes.extend_from_slice(&master.0);
    for value in [stage, purpose, 0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    hash_labeled(tag_word(PARAMETERS, 2), stage, None, &bytes)
}

/// The loop's private one-time material for one proximity group: every
/// switch's row tails, pad and tree salts, every stage FOLD's fresh-mask seed
/// and salts, and the BASE salts and randomness.
struct LoopPrivateMaterial {
    switches: Vec<IoppSwitchPrivate>,
    stage_folds: Vec<IoppStageFoldPrivate>,
    base: IoppBasePrivate,
}

/// Draws the loop's private material from the group's message seed. Every
/// switch, stage FOLD and the BASE star uses a domain-separated stream keyed
/// by its purpose and stage ordinal, so no stream is reused; the draw order
/// inside one stream is fixed by the trusted chain.
fn loop_private_material(seed: &Digest, plan: &IoppChain) -> Result<LoopPrivateMaterial> {
    let mut switches = Vec::with_capacity(plan.stages.len() - 1);
    for (index, stage) in plan.stages.iter().enumerate().skip(1) {
        let target = stage.target.ok_or(Error::InvalidConfig)?;
        let tree = stage.tree.as_ref().ok_or(Error::InvalidConfig)?;
        let mut coins = ByteMaskCoins::new(derive_run_seed(seed, 6, index as u64)?)?;
        let mut main_tails = Vec::with_capacity(
            usize::try_from(target.main_rows).map_err(|_| Error::ResourceLimit)?,
        );
        for _ in 0..target.main_rows {
            let main_tail = draw_values(
                &mut coins,
                usize::try_from(target.code.r).map_err(|_| Error::ResourceLimit)?,
            )?;
            main_tails.push(main_tail);
        }
        let mut mask_tails = Vec::with_capacity(
            usize::try_from(target.mask_rows).map_err(|_| Error::ResourceLimit)?,
        );
        for _ in 0..target.mask_rows {
            let mask_tail = draw_values(
                &mut coins,
                usize::try_from(target.mask_code.r).map_err(|_| Error::ResourceLimit)?,
            )?;
            mask_tails.push(mask_tail);
        }
        let mut pad = Fp3::ZERO;
        for _ in 0..64 {
            pad = coins.draw()?.0;
            if pad != Fp3::ZERO {
                break;
            }
        }
        if pad == Fp3::ZERO {
            return Err(Error::EntropyUnavailable);
        }
        let salts = draw_salt_bytes(
            &mut coins,
            usize::try_from(tree.columns()).map_err(|_| Error::ResourceLimit)?,
            tree.salt_bytes,
        )?;
        switches.push(IoppSwitchPrivate {
            main_tails,
            mask_tails,
            pad,
            salts,
        });
    }
    let mut stage_folds = Vec::with_capacity(plan.stages.len().saturating_sub(2));
    for (index, stage) in plan.stages.iter().enumerate().skip(1) {
        let Some(fold) = stage.fold.as_ref() else {
            continue;
        };
        let fold_seed = derive_run_seed(seed, 7, index as u64)?;
        let mut coins = ByteMaskCoins::new(derive_run_seed(&fold_seed, 8, index as u64)?)?;
        let salts = draw_salt_bytes(
            &mut coins,
            usize::try_from(fold.mask_tree.columns()).map_err(|_| Error::ResourceLimit)?,
            fold.mask_tree.salt_bytes,
        )?;
        stage_folds.push(IoppStageFoldPrivate {
            seed: fold_seed,
            salts,
        });
    }
    let star = &plan.config.base_star;
    let mut coins = ByteMaskCoins::new(derive_run_seed(seed, 9, 0)?)?;
    let salts = draw_salt_bytes(
        &mut coins,
        usize::try_from(star.columns()).map_err(|_| Error::ResourceLimit)?,
        star.salt_bytes,
    )?;
    let randomness = LastBaseRandomness::Fixed(derive_run_seed(seed, 9, 1)?);
    Ok(LoopPrivateMaterial {
        switches,
        stage_folds,
        base: IoppBasePrivate { salts, randomness },
    })
}

impl WCommitRandomness {
    /// Draws the complete `W` randomness from the existing numbered mask-coin
    /// ledger. The draw order is fixed: tails by position and row, one non-zero
    /// pad per answer, then one salt per tree column. The caller keeps the
    /// seed; the values are never serialized.
    pub fn from_coins(
        coins: &mut ByteMaskCoins,
        plan: &HiddenSendPlan,
        closure: Option<&ClosurePlan>,
    ) -> Result<Self> {
        plan.validate()?;
        let positions = plan.positions() as usize;
        let rows = usize::try_from(plan.w.main_rows).map_err(|_| Error::ResourceLimit)?;
        let tail_len = usize::try_from(plan.w.code.r).map_err(|_| Error::ResourceLimit)?;
        let columns = usize::try_from(plan.w.code.n()).map_err(|_| Error::ResourceLimit)?;
        let answers = plan.w.answers as usize;
        let mut tails = Vec::with_capacity(positions);
        for _ in 0..positions {
            let mut values = Vec::with_capacity(rows * tail_len);
            for _ in 0..rows * tail_len {
                values.push(coins.draw()?.0);
            }
            tails.push(values);
        }
        let mut pads = Vec::with_capacity(positions);
        for _ in 0..positions {
            let mut values = Vec::with_capacity(answers);
            for _ in 0..answers {
                let mut pad = Fp3::ZERO;
                for _ in 0..64 {
                    pad = coins.draw()?.0;
                    if pad != Fp3::ZERO {
                        break;
                    }
                }
                if pad == Fp3::ZERO {
                    return Err(Error::InvalidConfig);
                }
                values.push(pad);
            }
            pads.push(values);
        }
        let mut salts = Vec::with_capacity(positions);
        for _ in 0..positions {
            salts.push(draw_salt_bytes(coins, columns, plan.w.salt_bytes)?);
        }
        let index_mask = match &plan.index_mask {
            Some(index) => Some(IndexMaskRandomness::from_coins(coins, index)?),
            None => None,
        };
        let dot = match (&plan.dot, plan.activation_source) {
            (Some(dot), ActivationSource::Commitment) => {
                let rows = usize::try_from(dot.shape.main_rows).map_err(|_| Error::ResourceLimit)?;
                let tail_len = usize::try_from(dot.shape.code.r).map_err(|_| Error::ResourceLimit)?;
                let columns = usize::try_from(dot.shape.code.n()).map_err(|_| Error::ResourceLimit)?;
                let mut tail = Vec::with_capacity(rows * tail_len);
                for _ in 0..rows * tail_len {
                    tail.push(coins.draw()?.0);
                }
                let salts = draw_salt_bytes(coins, columns, dot.shape.salt_bytes)?;
                let mut pad = Fp3::ZERO;
                for _ in 0..64 {
                    pad = coins.draw()?.0;
                    if pad != Fp3::ZERO {
                        break;
                    }
                }
                if pad == Fp3::ZERO {
                    return Err(Error::EntropyUnavailable);
                }
                Some(DotMapRandomness { tail, salts, pad })
            }
            (Some(_), ActivationSource::NormHd) => None,
            (None, _) => None,
        };
        let closure = match closure {
            Some(closure_plan) => Some(ClosureRandomness::from_coins(
                coins,
                closure_plan,
                plan,
            )?),
            None => None,
        };
        let randomness = Self {
            tails,
            salts,
            pads,
            index_mask,
            dot,
            closure,
        };
        randomness.check(plan)?;
        Ok(randomness)
    }
}

fn eval_fp3_vector(values: &[Fp3], point: &[Fp3]) -> Result<Fp3> {
    let bits = u32::try_from(point.len()).map_err(|_| Error::ResourceLimit)?;
    let expected = 1usize
        .checked_shl(bits)
        .ok_or(Error::ResourceLimit)?;
    if values.len() != expected {
        return Err(Error::ShapeMismatch);
    }
    let mut sum = Fp3::ZERO;
    for (index, value) in values.iter().enumerate() {
        sum = sum.add(&eq_index(point, index as u64, bits)?.mul(value));
    }
    Ok(sum)
}

fn first_chain(plan: &HiddenLinPlan, reader: &IndexReader) -> Result<Chain> {
    Chain::begin(
        HIDDEN_SEND_CHAIN_LABEL,
        &plan.parameters_digest(),
        &plan.first_statement()?,
        &reader.name(),
        None,
    )
}

fn absorb_claim(chain: &mut Chain, plan: &HiddenLinPlan) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(7);
    push_digest_chain(&mut message, &plan.weights_digest()?);
    message.push_u64(plan.output.positions as u64);
    message.push_u64(plan.output.columns);
    message.push_u64(plan.masks.iter().map(|masks| masks.len() as u64).sum());
    let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// Draws `count` extension-field challenges from a fresh round. The round's
/// message is the caller's fixed bytes, so the challenge cannot be chosen after
/// the values it tests.
fn draw_point_round(
    chain: &mut Chain,
    kind: u32,
    words: &[u64],
    count: u32,
) -> Result<Vec<Fp3>> {
    let mut message = chain.message();
    message.push_u64(words.len() as u64);
    for word in words {
        message.push_u64(*word);
    }
    let prefix = message.prefix(kind, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    let mut point = Vec::with_capacity(count as usize);
    for _ in 0..count {
        point.push(chain.draw_e()?);
    }
    Ok(point)
}

fn absorb_digest_round(chain: &mut Chain, stage: u64, digests: &[Digest], words: &[u64]) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(stage);
    message.push_u64(digests.len() as u64);
    for digest in digests {
        push_digest_chain(&mut message, digest);
    }
    message.push_u64(words.len() as u64);
    for word in words {
        message.push_u64(*word);
    }
    let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// The first stage: the hidden activation commitment, the public claim items,
/// Computes one Eval target per position. In the diagnostic mode the target is
/// the multilinear evaluation of the public output rows. In the hidden-input
/// mode the same evaluation is taken from the prover's derived reduced rows,
/// which are the private MatMul output and are committed as the group members;
/// the verifier never receives those rows.
fn eval_targets(
    plan: &HiddenLinPlan,
    rows: &[Vec<Fp3>],
    output: Option<&[Vec<Fp3>]>,
    psi: &[Vec<Fp3>],
) -> Result<Vec<Fp3>> {
    let positions = plan.send.positions() as usize;
    let values: &[Vec<Fp3>] = match (output, plan.weights.is_empty()) {
        (None, true) => rows,
        (Some(output), false) => output,
        _ => return Err(Error::InvalidConfig),
    };
    if values.len() != positions
        || values
            .iter()
            .any(|row| row.len() as u64 != plan.output.columns)
        || psi.len() != positions
    {
        return Err(Error::ShapeMismatch);
    }
    let mut targets = Vec::with_capacity(positions);
    for (position, point) in psi.iter().enumerate() {
        let mut target = eval_fp3_vector(&values[position], point)?;
        for mask in &plan.masks[position] {
            target = target.add(&mask.value.mul(&mask.coefficient));
        }
        targets.push(target);
    }
    Ok(targets)
}

/// the committed `W` oracles, the SEND samples, the drawn points and the Eval
/// targets. Returns the public messages, the first seal and the private
/// witness.
fn prove_first(
    plan: &HiddenLinPlan,
    activation: Option<&CommittedActivation>,
    table: &mut dyn HiddenTableProver,
    output: Option<&[Vec<Fp3>]>,
    randomness: &WCommitRandomness,
    norm: Option<&NormWitness>,
    seed: &Digest,
    param_tables: Option<&mut dyn NormParamTableProver>,
) -> Result<(HiddenSendMessages, Digest, HiddenSendWitness)> {
    let _profile_first = zkie_engine::profile::Scope::new("first.prove_first");
    plan.validate()?;
    let reader = table.reader().clone();
    check_table(&plan.send, &reader)?;
    let weights = match plan.send.activation_source {
        ActivationSource::Commitment => {
            let activation = activation.ok_or(Error::InvalidConfig)?;
            check_activation(&plan.send, activation)?;
            let positions = plan.send.positions() as usize;
            let mut weights = Vec::with_capacity(positions);
            for position in 0..positions {
                weights.push(activation.weights(position as u32, plan.send.matrix_rows)?);
            }
            weights
        }
        ActivationSource::NormHd => {
            if activation.is_some() {
                return Err(Error::InvalidConfig);
            }
            hd_activation_rows(plan, norm)?
        }
    };
    let _profile_derive = zkie_engine::profile::Scope::new("first.derive_w_rows");
    let rows = derive_w_rows_from_weights(&plan.send, &weights, &reader, table)?;
    drop(_profile_derive);
    let _profile_commit_w = zkie_engine::profile::Scope::new("first.commit_w_rows");
    let (group, mut witness) = commit_w_rows(&plan.send, &rows, randomness)?;
    drop(_profile_commit_w);
    let dot_root = match (&plan.send.dot, plan.send.activation_source) {
        (Some(_), ActivationSource::Commitment) => {
            let activation = activation.ok_or(Error::InvalidConfig)?;
            let _profile_dot = zkie_engine::profile::Scope::new("first.commit_dot_map");
            let (root, map) = commit_dot_map(&plan.send, activation, randomness)?;
            drop(_profile_dot);
            witness.dot = Some(map);
            Some(root)
        }
        (Some(_), ActivationSource::NormHd) => None,
        (None, _) => None,
    };
    let closure_commit = match (&plan.closure, &randomness.closure) {
        (Some(closure_plan), Some(closure_randomness)) => {
            let _profile_values = zkie_engine::profile::Scope::new("first.closure_values");
            let (slack_values, bit_values) = closure_values_from_rows(closure_plan, &rows)?;
            drop(_profile_values);
            let _profile_slack = zkie_engine::profile::Scope::new("first.closure_slack_commit");
            let slack = commit_value_group(
                &closure_plan.slack,
                closure_plan.vocab as u64,
                closure_plan.slack_oracle_base,
                plan.send.group,
                &slack_values,
                &closure_randomness.slack_pads,
                &closure_randomness.slack_tails,
                &closure_randomness.slack_salts,
                plan.send.positions(),
            )?;
            drop(_profile_slack);
            let _profile_bits = zkie_engine::profile::Scope::new("first.closure_bits_commit");
            let bits = commit_value_group(
                &closure_plan.bits,
                closure_plan.bits_columns,
                closure_plan.bits_oracle_base,
                plan.send.group,
                &bit_values,
                &closure_randomness.bits_pads,
                &closure_randomness.bits_tails,
                &closure_randomness.bits_salts,
                plan.send.positions(),
            )?;
            drop(_profile_bits);
            let _profile_mask = zkie_engine::profile::Scope::new("first.closure_mask_commit");
            let mask = commit_closure_mask(&closure_plan.mask, closure_randomness)?;
            drop(_profile_mask);
            let _profile_entries = zkie_engine::profile::Scope::light("first.closure_mask_entries");
            let mask_entries = closure_mask_entry_rows(closure_plan, &mask)?;
            drop(_profile_entries);
            Some(ClosureWitness {
                slack,
                slack_tails: closure_randomness.slack_tails.clone(),
                slack_salts: closure_randomness.slack_salts.clone(),
                slack_pads: closure_randomness.slack_pads.clone(),
                bits,
                bits_tails: closure_randomness.bits_tails.clone(),
                bits_salts: closure_randomness.bits_salts.clone(),
                bits_pads: closure_randomness.bits_pads.clone(),
                bit_values,
                map: CommittedValueGroup {
                    group: SendGroup {
                        group: plan.send.group,
                        entries: Vec::new(),
                    },
                    messages: Vec::new(),
                    roots: Vec::new(),
                    trees: RetainedTrees::default(),
                },
                map_tails: closure_randomness.map_tails.clone(),
                map_salts: closure_randomness.map_salts.clone(),
                map_pads: closure_randomness.map_pads.clone(),
                map_values: Vec::new(),
                r_bool: Vec::new(),
                mask,
                mask_entries,
            })
        }
        (None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    witness.closure = closure_commit;
    let _profile_check = zkie_engine::profile::Scope::light("first.witness_check");
    witness.check(&plan.send)?;
    drop(_profile_check);

    let mut chain = first_chain(plan, &reader)?;
    absorb_group(&mut chain, &group)?;
    absorb_claim(&mut chain, plan)?;
    let index_mask = match (&plan.send.index_mask, &witness.index_mask) {
        (Some(index), Some(mask_witness)) => {
            absorb_index_mask_root(&mut chain, index, &mask_witness.root)?;
            Some((index, mask_witness))
        }
        (None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    match (&plan.send.dot, dot_root, plan.send.activation_source) {
        (Some(_), Some(root), ActivationSource::Commitment) => {
            absorb_dot_root(&mut chain, &plan.send, &root)?
        }
        (Some(_), None, ActivationSource::NormHd) => {}
        (None, None, _) => {}
        _ => return Err(Error::ShapeMismatch),
    }
    if let Some(closure) = &witness.closure {
        let closure_plan = plan.closure.as_ref().ok_or(Error::InvalidConfig)?;
        absorb_closure_group(&mut chain, 23, &closure.slack.group)?;
        absorb_closure_group(&mut chain, 24, &closure.bits.group)?;
        absorb_closure_mask_root(&mut chain, &closure_plan.mask, &closure.mask.root)?;
    }
    let norm_first = match (&plan.norm, norm) {
        (Some(_), Some(norm_witness)) => {
            let (message, lookup_messages, rstd_messages, param_claims) =
                prove_norm_first(plan, &mut chain, norm_witness, seed, param_tables)?;
            let mut owned = norm_witness.clone();
            if let Some(lookup) = owned.lookup.as_mut() {
                lookup.messages = lookup_messages;
            }
            if let Some(rstd) = owned.rstd.as_mut() {
                rstd.messages = rstd_messages;
            }
            if let Some((gain_claim, bias_claim)) = param_claims {
                let params = owned.params.as_mut().ok_or(Error::ShapeMismatch)?;
                params.gain_claim = Some(gain_claim);
                params.bias_claim = bias_claim;
            }
            witness.norm = Some(owned);
            Some(message)
        }
        (None, None) => {
            if param_tables.is_some() {
                return Err(Error::ShapeMismatch);
            }
            None
        }
        _ => return Err(Error::ShapeMismatch),
    };
    let seeds = draw_seeds(&mut chain, &plan.send)?;
    let dot_seeds = match (&plan.send.dot, plan.send.activation_source) {
        (Some(dot), ActivationSource::Commitment) => {
            Some(draw_seed_set(&mut chain, dot.sample_bits()?, 1)?)
        }
        _ => None,
    };
    let _profile_answers = zkie_engine::profile::Scope::new("first.w_answers");
    let answers = witness.answers(&seeds)?;
    drop(_profile_answers);
    absorb_answers(&mut chain, &answers)?;
    let dot_sample = match (&plan.send.dot, &dot_seeds, &witness.dot) {
        (Some(_dot), Some(seed_set), Some(map)) => {
            let answer = dot_sample_answer(&plan.send, map, &seed_set[0])?;
            absorb_dot_answer(&mut chain, answer)?;
            Some(SendSamples {
                seed: seed_set.clone(),
                answers: vec![answer],
            })
        }
        (Some(_), None, None) if plan.send.activation_source.is_norm_hd() => None,
        (None, None, None) => None,
        _ => return Err(Error::InvalidProof),
    };
    let closure_samples = match (&plan.closure, &witness.closure) {
        (Some(closure_plan), Some(closure)) => {
            let slack_seeds = draw_closure_seed_set(
                &mut chain,
                27,
                closure_plan.slack.sample_bits()?,
                closure_plan.slack.answers,
            )?;
            let bits_seeds = draw_closure_seed_set(
                &mut chain,
                28,
                closure_plan.bits.sample_bits()?,
                closure_plan.bits.answers,
            )?;
            let mask_seeds = draw_closure_seed_set(
                &mut chain,
                29,
                closure_plan.mask.sample_bits()?,
                closure_plan.mask.answers,
            )?;
            let _profile_closure_answers =
                zkie_engine::profile::Scope::new("first.closure_answers");
            let slack_answers = value_group_answers(
                &closure_plan.slack,
                closure_plan.slack_oracle_base,
                plan.send.group,
                &closure.slack.messages,
                &slack_seeds,
            )?;
            let bits_answers = value_group_answers(
                &closure_plan.bits,
                closure_plan.bits_oracle_base,
                plan.send.group,
                &closure.bits.messages,
                &bits_seeds,
            )?;
            let mask_answer =
                closure_mask_answer(&closure_plan.mask, &closure.mask.rows, &mask_seeds[0])?;
            drop(_profile_closure_answers);
            absorb_closure_answers(
                &mut chain,
                &slack_answers,
                &bits_answers,
                &[mask_answer],
            )?;
            Some((
                slack_seeds,
                slack_answers,
                bits_seeds,
                bits_answers,
                mask_seeds,
                mask_answer,
            ))
        }
        (None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    let closure_map = match (plan.closure.as_ref(), &mut witness.closure, &randomness.closure) {
        (Some(closure_plan), Some(closure), Some(closure_randomness)) => {
            let r_bool = draw_closure_r(&mut chain, closure_plan.boolean_variables)?;
            let _profile_map_values =
                zkie_engine::profile::Scope::new("first.closure_map_values");
            let eq_bool = crate::proof::output_closure::eq_vector(&r_bool)?;
            let map_values = closure
                .bit_values
                .iter()
                .map(|bits| {
                    crate::proof::output_closure::closure_map_values_with_eq(bits, &eq_bool)
                })
                .collect::<Result<Vec<_>>>()?;
            drop(_profile_map_values);
            let _profile_map_commit =
                zkie_engine::profile::Scope::new("first.closure_map_commit");
            let map = commit_value_group(
                &closure_plan.map,
                closure_plan.bits_columns,
                closure_plan.map_oracle_base,
                plan.send.group,
                &map_values,
                &closure_randomness.map_pads,
                &closure_randomness.map_tails,
                &closure_randomness.map_salts,
                plan.send.positions(),
            )?;
            drop(_profile_map_commit);
            absorb_closure_group(&mut chain, 26, &map.group)?;
            let map_seeds = draw_closure_seed_set(
                &mut chain,
                35,
                closure_plan.map.sample_bits()?,
                closure_plan.map.answers,
            )?;
            let _profile_map_answers =
                zkie_engine::profile::Scope::new("first.closure_map_answers");
            let map_answers = value_group_answers(
                &closure_plan.map,
                closure_plan.map_oracle_base,
                plan.send.group,
                &map.messages,
                &map_seeds,
            )?;
            drop(_profile_map_answers);
            absorb_tagged_answers(&mut chain, 36, &map_answers)?;
            let _profile_closure_check =
                zkie_engine::profile::Scope::new("first.closure_check");
            closure.map = map.clone();
            closure.map_values = map_values;
            closure.r_bool = r_bool.clone();
            closure.check(closure_plan, &plan.send)?;
            drop(_profile_closure_check);
            Some((map, map_seeds, map_answers, r_bool))
        }
        (None, None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    let mut index_mask_seed = None;
    let mut index_mask_answer_value = None;
    let mut index_mask_positions = None;
    let mut index_mask_opening = None;
    if let Some((index, mask_witness)) = index_mask {
        let _profile_index_openings =
            zkie_engine::profile::Scope::new("first.index_mask_openings");
        let seed = seeds.first().ok_or(Error::ShapeMismatch)?.clone();
        let answer = index_mask_answer(index, &mask_witness.rows, &seed)?;
        absorb_index_mask_answer(&mut chain, answer)?;
        let positions = draw_index_mask_positions(&mut chain, index)?;
        let opening = mask_witness.tree(index)?.open(&positions)?;
        index_mask_seed = Some(seed);
        index_mask_answer_value = Some(answer);
        index_mask_positions = Some(positions);
        index_mask_opening = Some(opening);
    }
    let _profile_w_openings = zkie_engine::profile::Scope::new("first.w_openings");
    let positions = draw_positions(&mut chain, &plan.send)?;
    let openings = zkie_engine::scheduling::try_map_range_bounded(
        positions.len(),
        zkie_engine::scheduling::DEFAULT_UNITS_IN_FLIGHT,
        |position| witness.tree(position as u32)?.open(&positions[position]),
    )?;
    drop(_profile_w_openings);
    let closure = match (
        plan.closure.as_ref(),
        &witness.closure,
        closure_samples,
        closure_map,
    ) {
        (
            Some(closure_plan),
            Some(closure),
            Some((
                slack_seeds,
                slack_answers,
                bits_seeds,
                bits_answers,
                mask_seeds,
                mask_answer,
            )),
            Some((map, map_seeds, map_answers, _r_bool)),
        ) => {
            let _profile_closure_openings =
                zkie_engine::profile::Scope::new("first.closure_openings");
            let slack_positions = draw_closure_group_positions(
                &mut chain,
                31,
                plan.send.positions(),
                closure_plan.slack_openings,
                closure_plan.slack.code.log_n,
            )?;
            let _profile_slack_open =
                zkie_engine::profile::Scope::new("first.closure_slack_openings");
            let slack_openings = open_value_group_openings(
                &closure_plan.slack,
                &closure.slack.messages,
                &closure.slack_tails,
                &closure.slack_salts,
                &slack_positions,
                &closure.slack.trees,
            )?;
            drop(_profile_slack_open);
            let bits_positions = draw_closure_group_positions(
                &mut chain,
                32,
                plan.send.positions(),
                closure_plan.bits_openings,
                closure_plan.bits.code.log_n,
            )?;
            let _profile_bits_open =
                zkie_engine::profile::Scope::new("first.closure_bits_openings");
            let bits_openings = open_value_group_openings(
                &closure_plan.bits,
                &closure.bits.messages,
                &closure.bits_tails,
                &closure.bits_salts,
                &bits_positions,
                &closure.bits.trees,
            )?;
            drop(_profile_bits_open);
            let mask_positions = draw_closure_mask_positions(&mut chain, &closure_plan.mask)?;
            let _profile_mask_open =
                zkie_engine::profile::Scope::new("first.closure_mask_openings");
            let mask_opening = closure.mask.tree(&closure_plan.mask)?.open(&mask_positions)?;
            drop(_profile_mask_open);
            let map_positions = draw_closure_group_positions(
                &mut chain,
                37,
                plan.send.positions(),
                closure_plan.map_openings,
                closure_plan.map.code.log_n,
            )?;
            let _profile_map_open =
                zkie_engine::profile::Scope::new("first.closure_map_openings");
            let map_openings = open_value_group_openings(
                &closure_plan.map,
                &map.messages,
                &closure.map_tails,
                &closure.map_salts,
                &map_positions,
                &map.trees,
            )?;
            drop(_profile_map_open);
            Some(ClosureFirstMessages {
                slack: CommittedGroupMessage::new(closure.slack.group.clone()),
                slack_sample: SendSamples {
                    seed: slack_seeds,
                    answers: slack_answers,
                },
                slack_positions,
                slack_openings,
                bits: CommittedGroupMessage::new(closure.bits.group.clone()),
                bits_sample: SendSamples {
                    seed: bits_seeds,
                    answers: bits_answers,
                },
                bits_positions,
                bits_openings,
                map: CommittedGroupMessage::new(map.group.clone()),
                map_sample: SendSamples {
                    seed: map_seeds,
                    answers: map_answers,
                },
                map_positions,
                map_openings,
                mask: ClosureMaskMessage {
                    tree: closure_plan.mask.tree.clone(),
                    root: closure.mask.root,
                    sample: SendSamples {
                        seed: mask_seeds,
                        answers: vec![mask_answer],
                    },
                    positions: mask_positions,
                    opening: mask_opening,
                },
            })
        }
        (None, None, None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    let _profile_eval = zkie_engine::profile::Scope::new("first.eval_targets");
    let psi = draw_psi(&mut chain, &plan.send)?;
    let mut targets = eval_targets(plan, &rows, output, &psi)?;
    let expected = witness.eval_values(&psi)?;
    drop(_profile_eval);
    if expected != targets {
        return Err(Error::InvalidProof);
    }
    if let (Some(index), Some(mask_witness)) = (&plan.send.index_mask, &witness.index_mask) {
        for position in 0..plan.send.positions() {
            let mask = mask_witness.entry(index, index.entries.eval_slot(position)?)?;
            targets[position as usize] = targets[position as usize].add(&mask);
        }
    }
    absorb_eval(&mut chain, &plan.send, &targets)?;
    let first_seal = *chain.value();
    let index_mask = match (index_mask_seed, index_mask_answer_value, index_mask_positions, index_mask_opening) {
        (Some(seed), Some(answer), Some(positions), Some(opening)) => Some(IndexMaskMessage {
            tree: plan
                .send
                .index_mask
                .as_ref()
                .ok_or(Error::ShapeMismatch)?
                .tree
                .clone(),
            root: witness
                .index_mask
                .as_ref()
                .ok_or(Error::ShapeMismatch)?
                .root,
            sample: SendSamples {
                seed: vec![seed],
                answers: vec![answer],
            },
            positions,
            opening,
        }),
        (None, None, None, None) => None,
        _ => return Err(Error::InvalidProof),
    };
    let dot = match (plan.send.dot.as_ref(), dot_root, dot_sample) {
        (Some(map), Some(root), Some(sample)) => Some(DotMapMessage {
            tree: map.tree_plan()?,
            root,
            sample,
        }),
        (Some(_), None, None) if plan.send.activation_source.is_norm_hd() => None,
        (None, None, None) => None,
        _ => return Err(Error::InvalidProof),
    };
    let activation_commitment = match plan.send.activation_source {
        ActivationSource::Commitment => {
            *activation.ok_or(Error::InvalidConfig)?.commitment()
        }
        ActivationSource::NormHd => {
            let first_norm = norm_first.as_ref().ok_or(Error::InvalidConfig)?;
            let output = first_norm.output.as_ref().ok_or(Error::InvalidConfig)?;
            ActivationCommitment {
                shape: plan.send.activation,
                root: hd_activation_identity(plan, &output.out.message.group.entries)?,
                configuration: plan.send.activation_configuration,
            }
        }
    };
    let messages = HiddenSendMessages {
        activation: activation_commitment,
        group,
        samples: SendSamples { seed: seeds, answers },
        positions,
        openings,
        psi,
        eval_targets: targets,
        index_mask,
        dot,
        chain_value: first_seal,
        closure,
        norm: norm_first,
    };
    Ok((messages, first_seal, witness))
}

fn verify_first(
    plan: &HiddenLinPlan,
    reader: &IndexReader,
    output: Option<&[Vec<Fp3>]>,
    messages: &HiddenSendMessages,
) -> Result<(Vec<EvalSampleRow>, Digest, Option<Vec<Fp3>>)> {
    let _profile = zkie_engine::profile::Scope::new("verify.first");
    plan.validate()?;
    // A fixture may pin the reduced roots as a negative-test expectation. The
    // hidden-input entry leaves them unpinned: the roots are proof values
    // absorbed into the first chain and checked through the openings below.
    let expected_roots = plan.send.expected_roots.as_ref();
    if let Some(roots) = expected_roots {
        if roots.len() != plan.send.positions() as usize {
            return Err(Error::ShapeMismatch);
        }
    }
    if messages.activation.fault().is_some()
        || messages.activation.shape != plan.send.activation
        || messages.activation.configuration != plan.send.activation_configuration
        || (!plan.send.activation_source.is_norm_hd()
            && messages.activation.root != plan.send.activation_root)
    {
        return Err(Error::StatementMismatch);
    }
    if messages.group.fault(plan.send.activation.positions).is_some()
        || messages.group.group != plan.send.group
        || messages.group.entries.len() != plan.send.positions() as usize
    {
        return Err(Error::InvalidConfig);
    }
    for (position, entry) in messages.group.entries.iter().enumerate() {
        if entry.position != position as u32
            || entry.oracle != plan.send.oracle(position as u32)?
            || entry.shape != plan.send.w.oracle_shape()
            || entry.tree != plan.send.w.tree_plan(position as u32)?
            || entry.root == Digest::ZERO
        {
            return Err(Error::StatementMismatch);
        }
        if let Some(roots) = expected_roots {
            if entry.root != roots[position] {
                return Err(Error::StatementMismatch);
            }
        }
    }
    let mut chain = first_chain(plan, reader)?;
    absorb_group(&mut chain, &messages.group)?;
    absorb_claim(&mut chain, plan)?;
    match (&plan.send.index_mask, &messages.index_mask) {
        (Some(index), Some(message)) => {
            if message.tree != index.tree
                || message.root == Digest::ZERO
                || message.sample.seed.len() != index.answers as usize
                || message.sample.answers.len() != index.answers as usize
            {
                return Err(Error::StatementMismatch);
            }
            absorb_index_mask_root(&mut chain, index, &message.root)?;
        }
        (None, None) => {}
        _ => return Err(Error::ShapeMismatch),
    }
    match (&plan.send.dot, &messages.dot) {
        (Some(map), Some(message)) => {
            if message.tree != map.tree_plan()?
                || message.root == Digest::ZERO
                || message.sample.seed.len() != map.shape.answers as usize
                || message.sample.answers.len() != map.shape.answers as usize
            {
                return Err(Error::StatementMismatch);
            }
            if let Some(root) = plan.send.expected_dot_root {
                if message.root != root {
                    return Err(Error::StatementMismatch);
                }
            }
            absorb_dot_root(&mut chain, &plan.send, &message.root)?;
        }
        (Some(_), None) if plan.send.activation_source.is_norm_hd() => {}
        (None, None) => {}
        _ => return Err(Error::ShapeMismatch),
    }
    match (&plan.closure, &messages.closure) {
        (Some(closure_plan), Some(closure)) => {
            if closure.slack.group.fault(plan.send.positions()).is_some()
                || closure.bits.group.fault(plan.send.positions()).is_some()
                || closure.slack.group.group != plan.send.group
                || closure.bits.group.group != plan.send.group
                || closure.mask.tree != closure_plan.mask.tree
                || closure.mask.root == Digest::ZERO
                || closure.slack_sample.seed.len() != closure_plan.slack.answers as usize
                || closure.slack_sample.answers.len() != plan.send.positions() as usize
                || closure.bits_sample.seed.len() != closure_plan.bits.answers as usize
                || closure.bits_sample.answers.len() != plan.send.positions() as usize
                || closure.mask.sample.seed.len() != closure_plan.mask.answers as usize
                || closure.mask.sample.answers.len() != closure_plan.mask.answers as usize
                || closure.map.group.fault(plan.send.positions()).is_some()
                || closure.map.group.group != plan.send.group
                || closure.map_sample.seed.len() != closure_plan.map.answers as usize
                || closure.map_sample.answers.len() != plan.send.positions() as usize
            {
                return Err(Error::StatementMismatch);
            }
            for (position, entry) in closure.slack.group.entries.iter().enumerate() {
                if entry.position != position as u32
                    || entry.oracle
                        != closure_plan
                            .slack_oracle_base
                            .checked_add(position as u64)
                            .ok_or(Error::ResourceLimit)?
                    || entry.shape != closure_plan.slack.oracle_shape()
                    || entry.tree != closure_plan.slack.tree_plan(position as u32)?
                    || entry.root == Digest::ZERO
                {
                    return Err(Error::StatementMismatch);
                }
            }
            for (position, entry) in closure.bits.group.entries.iter().enumerate() {
                if entry.position != position as u32
                    || entry.oracle
                        != closure_plan
                            .bits_oracle_base
                            .checked_add(position as u64)
                            .ok_or(Error::ResourceLimit)?
                    || entry.shape != closure_plan.bits.oracle_shape()
                    || entry.tree != closure_plan.bits.tree_plan(position as u32)?
                    || entry.root == Digest::ZERO
                {
                    return Err(Error::StatementMismatch);
                }
            }
            for (position, entry) in closure.map.group.entries.iter().enumerate() {
                if entry.position != position as u32
                    || entry.oracle
                        != closure_plan
                            .map_oracle_base
                            .checked_add(position as u64)
                            .ok_or(Error::ResourceLimit)?
                    || entry.shape != closure_plan.map.oracle_shape()
                    || entry.tree != closure_plan.map.tree_plan(position as u32)?
                    || entry.root == Digest::ZERO
                {
                    return Err(Error::StatementMismatch);
                }
            }
            absorb_closure_group(&mut chain, 23, &closure.slack.group)?;
            absorb_closure_group(&mut chain, 24, &closure.bits.group)?;
            absorb_closure_mask_root(&mut chain, &closure_plan.mask, &closure.mask.root)?;
        }
        (None, None) => {}
        _ => return Err(Error::ShapeMismatch),
    }
    match (&plan.norm, &messages.norm) {
        (Some(_), Some(message)) => verify_norm_first(&mut chain, plan, message)?,
        (None, None) => {}
        _ => return Err(Error::ShapeMismatch),
    }
    if plan.send.activation_source.is_norm_hd() {
        let norm_first = messages.norm.as_ref().ok_or(Error::ShapeMismatch)?;
        let output = norm_first.output.as_ref().ok_or(Error::ShapeMismatch)?;
        let identity = hd_activation_identity(plan, &output.out.message.group.entries)?;
        if identity != messages.activation.root {
            return Err(Error::StatementMismatch);
        }
    }
    let seeds = draw_seeds(&mut chain, &plan.send)?;
    if seeds != messages.samples.seed {
        return Err(Error::InvalidProof);
    }
    let dot_seeds = match (&plan.send.dot, plan.send.activation_source) {
        (Some(dot), ActivationSource::Commitment) => {
            let drawn = draw_seed_set(&mut chain, dot.sample_bits()?, 1)?;
            let message = messages.dot.as_ref().ok_or(Error::ShapeMismatch)?;
            if drawn != message.sample.seed {
                return Err(Error::InvalidProof);
            }
            Some(drawn)
        }
        _ => None,
    };
    let answer_count = plan.send.positions() as usize * plan.send.w.answers as usize;
    if messages.samples.answers.len() != answer_count {
        return Err(Error::ShapeMismatch);
    }
    absorb_answers(&mut chain, &messages.samples.answers)?;
    if let (Some(_seed), Some(message)) = (&dot_seeds, &messages.dot) {
        absorb_dot_answer(
            &mut chain,
            *message
                .sample
                .answers
                .first()
                .ok_or(Error::ShapeMismatch)?,
        )?;
    }
    if let (Some(closure_plan), Some(closure)) = (&plan.closure, &messages.closure) {
        let slack_seeds = draw_closure_seed_set(
            &mut chain,
            27,
            closure_plan.slack.sample_bits()?,
            closure_plan.slack.answers,
        )?;
        if slack_seeds != closure.slack_sample.seed {
            return Err(Error::InvalidProof);
        }
        let bits_seeds = draw_closure_seed_set(
            &mut chain,
            28,
            closure_plan.bits.sample_bits()?,
            closure_plan.bits.answers,
        )?;
        if bits_seeds != closure.bits_sample.seed {
            return Err(Error::InvalidProof);
        }
        let mask_seeds = draw_closure_seed_set(
            &mut chain,
            29,
            closure_plan.mask.sample_bits()?,
            closure_plan.mask.answers,
        )?;
        if mask_seeds != closure.mask.sample.seed {
            return Err(Error::InvalidProof);
        }
        absorb_closure_answers(
            &mut chain,
            &closure.slack_sample.answers,
            &closure.bits_sample.answers,
            &closure.mask.sample.answers,
        )?;
    }
    let closure_r = match (&plan.closure, &messages.closure) {
        (Some(closure_plan), Some(closure)) => {
            let r_bool = draw_closure_r(&mut chain, closure_plan.boolean_variables)?;
            absorb_closure_group(&mut chain, 26, &closure.map.group)?;
            let map_seeds = draw_closure_seed_set(
                &mut chain,
                35,
                closure_plan.map.sample_bits()?,
                closure_plan.map.answers,
            )?;
            if map_seeds != closure.map_sample.seed {
                return Err(Error::InvalidProof);
            }
            absorb_tagged_answers(&mut chain, 36, &closure.map_sample.answers)?;
            Some(r_bool)
        }
        (None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    if let (Some(index), Some(message)) = (&plan.send.index_mask, &messages.index_mask) {
        if message.sample.seed != seeds {
            return Err(Error::InvalidProof);
        }
        absorb_index_mask_answer(
            &mut chain,
            *message
                .sample
                .answers
                .first()
                .ok_or(Error::ShapeMismatch)?,
        )?;
        let mask_positions = draw_index_mask_positions(&mut chain, index)?;
        if mask_positions != message.positions
            || message.opening.positions != message.positions
        {
            return Err(Error::InvalidProof);
        }
        verify_runtime_opening(
            &index.tree,
            &message.root,
            &message.positions,
            &message.opening,
        )?;
    }
    let positions = draw_positions(&mut chain, &plan.send)?;
    if positions != messages.positions || messages.openings.len() != positions.len() {
        return Err(Error::InvalidProof);
    }
    let _profile_openings = zkie_engine::profile::Scope::light("verify.first.openings");
    for (position, (list, opening)) in positions.iter().zip(&messages.openings).enumerate() {
        if opening.positions != *list {
            return Err(Error::InvalidIndex);
        }
        let tree = plan.send.w.tree_plan(position as u32)?;
        let root = messages
            .group
            .entries
            .get(position)
            .map(|entry| entry.root)
            .ok_or(Error::ShapeMismatch)?;
        verify_runtime_opening(&tree, &root, list, opening)?;
    }
    if let (Some(closure_plan), Some(closure)) = (&plan.closure, &messages.closure) {
        let slack_positions = draw_closure_group_positions(
            &mut chain,
            31,
            plan.send.positions(),
            closure_plan.slack_openings,
            closure_plan.slack.code.log_n,
        )?;
        if slack_positions != closure.slack_positions
            || closure.slack_openings.len() != plan.send.positions() as usize
        {
            return Err(Error::InvalidProof);
        }
        for (position, (list, opening)) in slack_positions
            .iter()
            .zip(&closure.slack_openings)
            .enumerate()
        {
            if opening.positions != *list {
                return Err(Error::InvalidIndex);
            }
            let tree = closure_plan.slack.tree_plan(position as u32)?;
            let root = closure
                .slack
                .group
                .entries
                .get(position)
                .map(|entry| entry.root)
                .ok_or(Error::ShapeMismatch)?;
            verify_runtime_opening(&tree, &root, list, opening)?;
        }
        let bits_positions = draw_closure_group_positions(
            &mut chain,
            32,
            plan.send.positions(),
            closure_plan.bits_openings,
            closure_plan.bits.code.log_n,
        )?;
        if bits_positions != closure.bits_positions
            || closure.bits_openings.len() != plan.send.positions() as usize
        {
            return Err(Error::InvalidProof);
        }
        for (position, (list, opening)) in bits_positions
            .iter()
            .zip(&closure.bits_openings)
            .enumerate()
        {
            if opening.positions != *list {
                return Err(Error::InvalidIndex);
            }
            let tree = closure_plan.bits.tree_plan(position as u32)?;
            let root = closure
                .bits
                .group
                .entries
                .get(position)
                .map(|entry| entry.root)
                .ok_or(Error::ShapeMismatch)?;
            verify_runtime_opening(&tree, &root, list, opening)?;
        }
        let mask_positions = draw_closure_mask_positions(&mut chain, &closure_plan.mask)?;
        if mask_positions != closure.mask.positions
            || closure.mask.opening.positions != closure.mask.positions
        {
            return Err(Error::InvalidProof);
        }
        verify_runtime_opening(
            &closure_plan.mask.tree,
            &closure.mask.root,
            &closure.mask.positions,
            &closure.mask.opening,
        )?;
        let map_positions = draw_closure_group_positions(
            &mut chain,
            37,
            plan.send.positions(),
            closure_plan.map_openings,
            closure_plan.map.code.log_n,
        )?;
        if map_positions != closure.map_positions
            || closure.map_openings.len() != plan.send.positions() as usize
        {
            return Err(Error::InvalidProof);
        }
        for (position, (list, opening)) in map_positions
            .iter()
            .zip(&closure.map_openings)
            .enumerate()
        {
            if opening.positions != *list {
                return Err(Error::InvalidIndex);
            }
            let tree = closure_plan.map.tree_plan(position as u32)?;
            let root = closure
                .map
                .group
                .entries
                .get(position)
                .map(|entry| entry.root)
                .ok_or(Error::ShapeMismatch)?;
            verify_runtime_opening(&tree, &root, list, opening)?;
        }
    }
    drop(_profile_openings);
    let psi = draw_psi(&mut chain, &plan.send)?;
    if psi != messages.psi {
        return Err(Error::InvalidProof);
    }
    if messages.eval_targets.len() != plan.send.positions() as usize {
        return Err(Error::ShapeMismatch);
    }
    if plan.weights.is_empty() {
        // Hidden-input mode: the Eval targets are message values. No plaintext
        // output row is accepted here; the BAT rows built from the messages
        // bind every target to the committed reduced rows.
        if output.is_some() {
            return Err(Error::InvalidConfig);
        }
        if plan.send.index_mask.is_none() {
            return Err(Error::InvalidConfig);
        }
    } else {
        if plan.send.index_mask.is_some() {
            return Err(Error::InvalidConfig);
        }
        let expected = eval_targets(plan, &[], output, &psi)?;
        if expected != messages.eval_targets {
            return Err(Error::InvalidProof);
        }
    }
    absorb_eval(&mut chain, &plan.send, &messages.eval_targets)?;
    if *chain.value() != messages.chain_value {
        return Err(Error::InvalidProof);
    }
    let mut eval_rows = Vec::with_capacity(psi.len());
    for (position, point) in psi.iter().enumerate() {
        eval_rows.push(eval_row(
            &plan.send,
            position as u32,
            point,
            messages.eval_targets[position],
        )?);
    }
    Ok((eval_rows, messages.chain_value, closure_r))
}

/// Folds the committed table by the shared point's high column coordinates.
/// Only the table's row source is read; no plaintext table object is built.
fn fold_table(
    source: &mut dyn TableRowSource,
    shape: IndexShape,
    gamma: &[Fp3],
) -> Result<(Vec<Fp3>, Vec<Fp3>)> {
    let rows = usize::try_from(shape.rows()).map_err(|_| Error::ResourceLimit)?;
    let ell = usize::try_from(shape.code.ell).map_err(|_| Error::ResourceLimit)?;
    let tail_len = usize::try_from(shape.code.r).map_err(|_| Error::ResourceLimit)?;
    if gamma.len() != shape.v as usize || source.rows() != shape.rows() {
        return Err(Error::ShapeMismatch);
    }
    let weights = foldeq_weights(gamma, shape.v)?;
    let mut folded = vec![Fp3::ZERO; ell];
    let mut folded_tail = vec![Fp3::ZERO; tail_len];
    let mut message = vec![Fp::ZERO; ell];
    let mut tail = vec![Fp::ZERO; tail_len];
    for row in 0..rows {
        source.row(row as u64, &mut message, &mut tail)?;
        let weight = weights[row];
        for (index, value) in message.iter().enumerate() {
            if *value != Fp::ZERO {
                folded[index] = folded[index].add(&weight.mul(&Fp3::from_base(*value)));
            }
        }
        for (index, value) in tail.iter().enumerate() {
            if *value != Fp::ZERO {
                folded_tail[index] =
                    folded_tail[index].add(&weight.mul(&Fp3::from_base(*value)));
            }
        }
    }
    Ok((folded, folded_tail))
}

/// Places a vector in one oracle message and commits the runtime tree.
fn gu_answers(
    gu: &GuShape,
    registry: &GuRegistry,
    witness: &GuWitness,
    seeds: &[SampleSeed],
) -> Result<Vec<Fp3>> {
    let mut answers = Vec::with_capacity(gu.u.answers as usize);
    for answer in 0..gu.u.answers {
        let row = sample_row(gu, registry, seeds, answer, Fp3::ZERO)?;
        answers.push(row.evaluate(witness)?);
    }
    Ok(answers)
}

fn omega_answers(
    omega: &OmegaShapePlan,
    witness: &GuWitness,
    seeds: &[SampleSeed],
) -> Result<Vec<Fp3>> {
    let mut answers = Vec::with_capacity(omega.u.answers as usize);
    for answer in 0..omega.u.answers {
        let row = oracle_sample_row(
            omega.oracle,
            omega.group,
            &omega.u,
            seeds,
            answer,
            Fp3::ZERO,
        )?;
        answers.push(row.evaluate(witness)?);
    }
    Ok(answers)
}

fn absorb_root_round(chain: &mut Chain, root: &Digest) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(8);
    push_digest_chain(&mut message, root);
    let prefix = message.prefix(KIND_SEED, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn draw_seed_set(chain: &mut Chain, bits: u32, answers: u32) -> Result<Vec<SampleSeed>> {
    let mut seeds = Vec::with_capacity(answers as usize);
    for _ in 0..answers {
        let mut z = Vec::with_capacity(bits as usize);
        for _ in 0..bits {
            z.push(chain.draw_e()?);
        }
        let beta = chain.draw_e_nonzero()?;
        seeds.push(SampleSeed { z, beta });
    }
    Ok(seeds)
}

fn absorb_answer_round(chain: &mut Chain, answers: &[Fp3]) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(9);
    message.push_u64(answers.len() as u64);
    for answer in answers {
        message.push_fp3(answer);
    }
    let prefix = message.prefix(KIND_EMPTY, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn draw_columns(
    chain: &mut Chain,
    draws: usize,
    work_bits: u8,
    log_n: u32,
    nonce: Option<u64>,
) -> Result<(Vec<u64>, u64)> {
    let prefix = chain.message().prefix(KIND_POSITIONS, work_bits)?;
    let nonce = match nonce {
        Some(value) => {
            if !verify_nonce(&prefix, value, work_bits)? {
                return Err(Error::InvalidProof);
            }
            value
        }
        None => search_nonce_for_plan(&prefix)?,
    };
    if !chain.step(&prefix, nonce)? {
        return Err(Error::InvalidProof);
    }
    let mut positions = Vec::with_capacity(draws);
    for _ in 0..draws {
        positions.push(chain.draw_position(log_n)?);
    }
    Ok((positions, nonce))
}

fn table_column_map(
    reader: &IndexReader,
    positions: &[u64],
    openings: &[ColumnOpening],
) -> Result<Vec<(u64, Vec<Fp3>)>> {
    let unique = unique_sorted(positions);
    let flat = verify_columns(reader, &unique, openings)?;
    let rows = reader.shape.rows() as usize;
    let mut map = Vec::with_capacity(unique.len());
    for (index, position) in unique.iter().enumerate() {
        let start = index * rows;
        let end = start.checked_add(rows).ok_or(Error::ResourceLimit)?;
        let column = flat
            .get(start..end)
            .ok_or(Error::ShapeMismatch)?
            .iter()
            .map(|value| Fp3::from_base(*value))
            .collect();
        map.push((*position, column));
    }
    Ok(map)
}

fn column_for(map: &[(u64, Vec<Fp3>)], position: u64) -> Result<Vec<Fp3>> {
    map.iter()
        .find(|(at, _)| *at == position)
        .map(|(_, column)| column.clone())
        .ok_or(Error::InvalidIndex)
}

/// Builds the Fold row of the shared stage: `Omega` at `zeta` against the
/// folded row `u` at the shared point and `zeta`.
fn fold_plain_row(
    gu: &GuShape,
    omega: &OmegaShapePlan,
    gu_bat: u64,
    omega_bat: u64,
    psi: &[Fp3],
    zeta: &[Fp3],
    matrix_rows: u64,
    k_lo: u32,
) -> Result<BatRow> {
    if psi.len() != k_lo as usize + gu.index.v as usize {
        return Err(Error::ShapeMismatch);
    }
    let row_bits = matrix_rows.trailing_zeros();
    if zeta.len() != row_bits as usize || matrix_rows < 1 || !matrix_rows.is_power_of_two() {
        return Err(Error::ShapeMismatch);
    }
    let gu_len = usize::try_from(gu.u.main_rows.checked_mul(gu.u.code.ell).ok_or(Error::ResourceLimit)?)
        .map_err(|_| Error::ResourceLimit)?;
    let omega_len = usize::try_from(
        omega
            .u
            .main_rows
            .checked_mul(omega.u.code.ell)
            .ok_or(Error::ResourceLimit)?,
    )
    .map_err(|_| Error::ResourceLimit)?;
    let mut gu_coefficients = vec![Fp3::ZERO; gu_len];
    let mut omega_coefficients = vec![Fp3::ZERO; omega_len];
    let low_count = 1u64 << k_lo;
    let mut low = Fp3::ZERO;
    for low_index in 0..low_count {
        low = eq_index(&psi[..k_lo as usize], low_index, k_lo)?;
        for y in 0..matrix_rows {
            let zeta_weight = eq_index(zeta, y, row_bits)?;
            let entry = low_index
                .checked_add(y.checked_shl(k_lo).ok_or(Error::ResourceLimit)?)
                .ok_or(Error::ResourceLimit)?;
            let at = gu_vector_bat_index(gu, entry)?;
            let slot = gu_coefficients.get_mut(at).ok_or(Error::ShapeMismatch)?;
            *slot = slot.sub(&low.mul(&zeta_weight));
        }
    }
    let _ = low;
    for y in 0..matrix_rows {
        let weight = eq_index(zeta, y, row_bits)?;
        let slot = omega_coefficients
            .get_mut(usize::try_from(y).map_err(|_| Error::ResourceLimit)?)
            .ok_or(Error::ShapeMismatch)?;
        *slot = weight;
    }
    let mut row = BatRow::new(BatRowKind::Plain);
    row.add_term(gu_bat, BatPart::Main, gu_coefficients);
    row.add_term(omega_bat, BatPart::Main, omega_coefficients);
    Ok(row)
}

/// The BAT table index of one folded-row vector entry under the trusted `G_u`
/// layout. The layout swaps row and place bits at the top; only real entries
/// are admitted.
fn gu_vector_bat_index(gu: &GuShape, entry: u64) -> Result<usize> {
    let row_bits = gu.u.main_rows.trailing_zeros();
    let place_bits = gu.u.code.log_ell();
    if !gu.rows_high || row_bits >= 64 || place_bits >= 64 {
        return Err(Error::UnsupportedPrimitive);
    }
    let places = 1u64 << place_bits;
    let virtual_index = (entry >> place_bits) | ((entry & (places - 1)) << row_bits);
    let row = virtual_index & (gu.u.main_rows - 1);
    let place = virtual_index >> row_bits;
    if place >= gu.u.code.ell {
        return Err(Error::ShapeMismatch);
    }
    usize::try_from(
        row.checked_add(place.checked_shl(row_bits).ok_or(Error::ResourceLimit)?)
            .ok_or(Error::ResourceLimit)?,
    )
    .map_err(|_| Error::ResourceLimit)
}

// ---------------------------------------------------------------------------
// Shared stage
// ---------------------------------------------------------------------------

struct SharedProveResult {
    messages: HiddenSharedMessages,
    witness: SharedWitness,
    points: SharedPoints,
}

struct SharedWitness {
    gu_message: Vec<Vec<Fp3>>,
    gu_tails: Vec<Vec<Fp3>>,
    omega_message: Vec<Fp3>,
    omega_tails: Vec<Vec<Fp3>>,
    omega_salts: Vec<Vec<u8>>,
}

struct SharedPoints {
    psi: Vec<Fp3>,
    zeta: Vec<Fp3>,
}

fn shared_start_chain(
    plan: &HiddenLinPlan,
    reader: &IndexReader,
    first_seal: &Digest,
) -> Result<Chain> {
    let statement = plan.first_statement()?;
    let mut statement_input = Vec::with_capacity(statement.len() + 32);
    statement_input.extend_from_slice(&first_seal.0);
    statement_input.extend_from_slice(&statement);
    let statement_digest = hash_labeled(
        tag_word(PARAMETERS, 3),
        1,
        None,
        &statement_input,
    )?;
    let mut chain = Chain::resume(SHARED_CHAIN_LABEL, *first_seal);
    absorb_digest_round(
        &mut chain,
        1,
        &[plan.parameters_digest(), statement_digest, *first_seal],
        &[reader.shape.v as u64, plan.send.k_lo as u64],
    )?;
    Ok(chain)
}

fn shared_point_bits(plan: &HiddenLinPlan) -> Result<u32> {
    let k_col = plan.send.columns.trailing_zeros();
    if plan
        .send
        .k_lo
        .checked_add(plan.send.table_shape.v)
        != Some(k_col)
    {
        return Err(Error::InvalidConfig);
    }
    Ok(k_col)
}

/// The inputs of the parameter claim members that join the shared proximity
/// group. The prover also supplies the committed folded message material.
struct NormParamClaimInput {
    identity: NormParamTableIdentity,
    claim: NormParamClaimPlan,
    message: NormParamClaimMessage,
    material: Option<crate::proof::hidden_norm_psc::NormParamClaimWitness>,
}

fn norm_param_claim_inputs(
    plan: &HiddenLinPlan,
    first: &HiddenSendMessages,
    witness: Option<&NormWitness>,
) -> Result<Vec<NormParamClaimInput>> {
    let norm = match &plan.norm {
        Some(norm) => norm,
        None => return Ok(Vec::new()),
    };
    let params_plan = match &norm.params {
        Some(params) => params,
        None => return Ok(Vec::new()),
    };
    let norm_first = first.norm.as_ref().ok_or(Error::ShapeMismatch)?;
    let message = norm_first.params.as_ref().ok_or(Error::ShapeMismatch)?;
    let (gain_material, bias_material) = match witness {
        Some(witness) => {
            let params = witness.params.as_ref().ok_or(Error::ShapeMismatch)?;
            (
                Some(params.gain_claim.clone().ok_or(Error::ShapeMismatch)?),
                params.bias_claim.clone(),
            )
        }
        None => (None, None),
    };
    let mut inputs = Vec::new();
    inputs.push(NormParamClaimInput {
        identity: params_plan.gain.clone(),
        claim: params_plan
            .gain_claim
            .clone()
            .ok_or(Error::ShapeMismatch)?,
        message: message.gain.clone(),
        material: gain_material,
    });
    if let Some(bias) = &message.bias {
        inputs.push(NormParamClaimInput {
            identity: params_plan.bias.clone(),
            claim: params_plan
                .bias_claim
                .clone()
                .ok_or(Error::ShapeMismatch)?,
            message: bias.clone(),
            material: bias_material,
        });
    } else if bias_material.is_some() {
        return Err(Error::ShapeMismatch);
    }
    Ok(inputs)
}

fn norm_param_registry(
    claim: &NormParamClaimPlan,
    message: &NormParamClaimMessage,
) -> Result<GuRegistry> {
    let gu = &claim.gu;
    let mut registry = GuRegistry::new();
    registry.receive(
        gu,
        &GuNotice {
            root: message.root,
            oracle: gu.oracle,
            tree_number: gu.tree.tree_number,
            part: gu.part,
        },
    )?;
    Ok(registry)
}

/// Prover side of the shared stage.
fn prove_shared(
    plan: &HiddenLinPlan,
    reader: &IndexReader,
    table: &mut dyn HiddenTableProver,
    first: &HiddenSendMessages,
    first_seal: &Digest,
    seed: &Digest,
    norm_witness: Option<&NormWitness>,
) -> Result<SharedProveResult> {
    let _profile_shared = zkie_engine::profile::Scope::new("shared.prove_shared");
    let k_col = shared_point_bits(plan)?;
    let mut chain = shared_start_chain(plan, reader, first_seal)?;
    let psi = draw_point_round(
        &mut chain,
        KIND_SEED,
        &[HIDDEN_SEND_SCHEMA_VERSION, 10],
        k_col,
    )?;
    let gamma = psi[plan.send.k_lo as usize..].to_vec();
    let _profile_fold_table = zkie_engine::profile::Scope::new("shared.fold_table");
    let (folded, folded_tail) = fold_table(table, reader.shape, &gamma)?;
    drop(_profile_fold_table);
    let gu = plan.proximity.gu.clone();
    let mut gu_coins = ByteMaskCoins::new(derive_run_seed(seed, 2, 0)?)?;
    let gu_pads = draw_values(&mut gu_coins, gu.u.answers as usize)?;
    let gu_tails = {
        let rows = usize::try_from(gu.u.main_rows).map_err(|_| Error::ResourceLimit)?;
        let tail_len = usize::try_from(gu.u.code.r).map_err(|_| Error::ResourceLimit)?;
        let mut tails = Vec::with_capacity(rows);
        for _ in 0..rows {
            tails.push(draw_values(&mut gu_coins, tail_len)?);
        }
        tails
    };
    let gu_salts = draw_salt_bytes(
        &mut gu_coins,
        usize::try_from(gu.tree.columns()).map_err(|_| Error::ResourceLimit)?,
        gu.tree.salt_bytes,
    )?;
    let _profile_gu_table = zkie_engine::profile::Scope::new("shared.gu_table");
    let mut gu_witness = gu_table(&gu, &folded, &folded_tail)?;
    drop(_profile_gu_table);
    let (main_pads, mask_pads) = gu_sample_pad_offsets(&gu)?;
    if !mask_pads.is_empty() || main_pads.len() != gu_pads.len() {
        return Err(Error::ShapeMismatch);
    }
    for (offset, pad) in main_pads.iter().zip(&gu_pads) {
        let slot = gu_witness.main.get_mut(*offset).ok_or(Error::ShapeMismatch)?;
        *slot = *pad;
    }
    let gu_rows = usize::try_from(gu.u.main_rows).map_err(|_| Error::ResourceLimit)?;
    let gu_ell = usize::try_from(gu.u.code.ell).map_err(|_| Error::ResourceLimit)?;
    let mut gu_message = Vec::with_capacity(gu_rows);
    for row in 0..gu_rows {
        let start = row.checked_mul(gu_ell).ok_or(Error::ResourceLimit)?;
        let end = start.checked_add(gu_ell).ok_or(Error::ResourceLimit)?;
        gu_message.push(
            gu_witness
                .main
                .get(start..end)
                .ok_or(Error::ShapeMismatch)?
                .to_vec(),
        );
    }
    let _profile_gu_commit = zkie_engine::profile::Scope::new("shared.gu_commit");
    let gu_tree = commit_rows_tree(gu.tree.clone(), gu.u.code, &gu_message, &gu_tails, &gu_salts)?;
    drop(_profile_gu_commit);
    let gu_root = *gu_tree.root();
    let mut registry = GuRegistry::new();
    registry.receive(
        &gu,
        &GuNotice {
            root: gu_root,
            oracle: gu.oracle,
            tree_number: gu.tree.tree_number,
            part: gu.part,
        },
    )?;
    absorb_root_round(&mut chain, &gu_root)?;
    let gu_bits = gu.sample_bits()?;
    let gu_seeds = draw_seed_set(&mut chain, gu_bits, gu.u.answers)?;
    let gu_answers = gu_answers(&gu, &registry, &gu_witness, &gu_seeds)?;
    absorb_answer_round(&mut chain, &gu_answers)?;

    let (positions, nonce) = draw_columns(
        &mut chain,
        plan.proximity.column_draws,
        plan.proximity.column_work_bits,
        reader.shape.code.log_n,
        None,
    )?;
    let unique = unique_sorted(&positions);
    let _profile_open_columns = zkie_engine::profile::Scope::new("shared.table_open_columns");
    let gu_openings = table.open_columns(&unique)?;
    drop(_profile_open_columns);
    let _profile_column_rows = zkie_engine::profile::Scope::new("shared.column_rows");
    let column_map = table_column_map(reader, &positions, &gu_openings)?;
    let eq_gamma = foldeq_weights(&gamma, reader.shape.v)?;
    let record_seeds = record_seeds(&reader.shape, &reader.name(), reader.shape.answers as usize)?;
    let mut record_rows = Vec::with_capacity(reader.shape.answers as usize);
    for answer in 0..reader.shape.answers {
        record_rows.push(record_row(
            &gu,
            &registry,
            &reader.record,
            &reader.name(),
            &record_seeds,
            &eq_gamma,
            answer,
        )?);
    }
    let mut column_rows = Vec::with_capacity(positions.len());
    for position in &positions {
        let column = column_for(&column_map, *position)?;
        column_rows.push(column_row(&gu, &registry, *position, &column, &eq_gamma)?);
    }
    drop(_profile_column_rows);

    let omega = plan.proximity.omega.clone();
    let matrix_rows = omega.matrix_rows;
    let k_lo = omega.k_lo;
    if plan.send.k_lo != k_lo || plan.send.matrix_rows != matrix_rows {
        return Err(Error::StatementMismatch);
    }
    let low_count = 1u64 << k_lo;
    let mut omega_vector = vec![Fp3::ZERO; usize::try_from(matrix_rows).map_err(|_| Error::ResourceLimit)?];
    let _profile_omega_vector = zkie_engine::profile::Scope::new("shared.omega_vector");
    for y in 0..matrix_rows {
        let mut sum = Fp3::ZERO;
        for low in 0..low_count {
            let entry = low
                .checked_add(y.checked_shl(k_lo).ok_or(Error::ResourceLimit)?)
                .ok_or(Error::ResourceLimit)?;
            let weight = eq_index(&psi[..k_lo as usize], low, k_lo)?;
            let value = folded
                .get(usize::try_from(entry).map_err(|_| Error::ResourceLimit)?)
                .ok_or(Error::ShapeMismatch)?;
            sum = sum.add(&weight.mul(value));
        }
        omega_vector[usize::try_from(y).map_err(|_| Error::ResourceLimit)?] = sum;
    }
    drop(_profile_omega_vector);
    let mut omega_coins = ByteMaskCoins::new(derive_run_seed(seed, 3, 0)?)?;
    let omega_pads = draw_values(&mut omega_coins, omega.u.answers as usize)?;
    let omega_tail_len = usize::try_from(omega.u.code.r).map_err(|_| Error::ResourceLimit)?;
    let omega_rows = usize::try_from(omega.u.main_rows).map_err(|_| Error::ResourceLimit)?;
    let mut omega_tails = Vec::with_capacity(omega_rows);
    for _ in 0..omega_rows {
        omega_tails.push(draw_values(&mut omega_coins, omega_tail_len)?);
    }
    let omega_salts = draw_salt_bytes(
        &mut omega_coins,
        usize::try_from(omega.tree.columns()).map_err(|_| Error::ResourceLimit)?,
        omega.tree.salt_bytes,
    )?;
    let omega_ell = usize::try_from(omega.u.code.ell).map_err(|_| Error::ResourceLimit)?;
    let _profile_omega_commit = zkie_engine::profile::Scope::new("shared.omega_commit");
    let mut omega_message = vec![vec![Fp3::ZERO; omega_ell]; omega_rows];
    for (index, value) in omega_vector.iter().enumerate() {
        let row = index % omega_rows;
        let place = index / omega_rows;
        let slot = omega_message
            .get_mut(row)
            .and_then(|row| row.get_mut(place))
            .ok_or(Error::ShapeMismatch)?;
        *slot = *value;
    }
    for (answer, pad) in omega_pads.iter().enumerate() {
        let place = omega_ell
            .checked_sub(omega_pads.len())
            .and_then(|first| first.checked_add(answer))
            .ok_or(Error::ResourceLimit)?;
        let slot = omega_message
            .get_mut(omega_rows - 1)
            .and_then(|row| row.get_mut(place))
            .ok_or(Error::ShapeMismatch)?;
        if !slot.is_zero() {
            return Err(Error::StatementMismatch);
        }
        *slot = *pad;
    }
    let omega_tree = commit_rows_tree(
        omega.tree.clone(),
        omega.u.code,
        &omega_message,
        &omega_tails,
        &omega_salts,
    )?;
    drop(_profile_omega_commit);
    let omega_root = *omega_tree.root();
    absorb_root_round(&mut chain, &omega_root)?;
    let omega_bits = sample_cube_of(&omega.u)?.bits;
    let omega_seeds = draw_seed_set(&mut chain, omega_bits, omega.u.answers)?;
    let omega_witness = GuWitness {
        main: omega_message.concat(),
        mask: Vec::new(),
    };
    let omega_answers = omega_answers(&omega, &omega_witness, &omega_seeds)?;
    absorb_answer_round(&mut chain, &omega_answers)?;
    let zeta = draw_point_round(
        &mut chain,
        KIND_ALGEBRAIC,
        &[HIDDEN_SEND_SCHEMA_VERSION, 11],
        matrix_rows.trailing_zeros(),
    )?;

    // The two members of the shared proximity group.
    let mut members = vec![
        ProximityMember {
            bat_oracle: 0,
            source_oracle: gu.oracle,
            main_code: gu.u.code,
            mask_code: plan.proximity.mask_code,
            main_rows: gu.u.main_rows,
            mask_rows: 0,
            answers: gu.u.answers,
            root: gu_root,
            tree_plan: gu.tree.clone(),
            message: gu_message.clone(),
            tails: gu_tails.clone(),
            mask_message: Vec::new(),
            mask_tails: Vec::new(),
        },
        ProximityMember {
            bat_oracle: 1,
            source_oracle: omega.oracle,
            main_code: omega.u.code,
            mask_code: plan.proximity.mask_code,
            main_rows: omega.u.main_rows,
            mask_rows: 0,
            answers: omega.u.answers,
            root: omega_root,
            tree_plan: omega.tree.clone(),
            message: omega_message.clone(),
            tails: omega_tails.clone(),
            mask_message: Vec::new(),
            mask_tails: Vec::new(),
        },
    ];
    let param_inputs = norm_param_claim_inputs(plan, first, norm_witness)?;
    let mut param_trees = Vec::new();
    for input in &param_inputs {
        let gu = &input.claim.gu;
        let material = input.material.as_ref().ok_or(Error::ShapeMismatch)?;
        let tree = commit_rows_tree(
            gu.tree.clone(),
            gu.u.code,
            &material.message,
            &material.tails,
            &material.salts,
        )?;
        if *tree.root() != input.message.root {
            return Err(Error::StatementMismatch);
        }
        param_trees.push((gu.oracle, Arc::new(tree)));
        members.push(ProximityMember {
            bat_oracle: 0,
            source_oracle: gu.oracle,
            main_code: gu.u.code,
            mask_code: plan.proximity.mask_code,
            main_rows: gu.u.main_rows,
            mask_rows: 0,
            answers: gu.u.answers,
            root: input.message.root,
            tree_plan: gu.tree.clone(),
            message: material.message.clone(),
            tails: material.tails.clone(),
            mask_message: Vec::new(),
            mask_tails: Vec::new(),
        });
    }
    members.sort_by_key(|member| std::cmp::Reverse((member.main_rows, member.source_oracle)));
    let mut gu_bat = None;
    let mut omega_bat = None;
    for (index, member) in members.iter_mut().enumerate() {
        member.bat_oracle = index as u64;
        if member.source_oracle == gu.oracle {
            gu_bat = Some(member.bat_oracle);
        }
        if member.source_oracle == omega.oracle {
            omega_bat = Some(member.bat_oracle);
        }
    }
    let gu_bat = gu_bat.ok_or(Error::StatementMismatch)?;
    let omega_bat = omega_bat.ok_or(Error::StatementMismatch)?;
    let mut param_registries = Vec::with_capacity(param_inputs.len());
    for input in &param_inputs {
        param_registries.push(norm_param_registry(&input.claim, &input.message)?);
    }
    let param_of = |oracle: u64| {
        param_inputs
            .iter()
            .position(|input| input.claim.gu.oracle == oracle)
    };
    let mut sample_rows = Vec::with_capacity(members.len());
    for member in &members {
        if member.source_oracle == gu.oracle {
            sample_rows.push(vec![sample_row(
                &gu,
                &registry,
                &gu_seeds,
                0,
                gu_answers[0],
            )?]);
        } else if let Some(index) = param_of(member.source_oracle) {
            let input = &param_inputs[index];
            sample_rows.push(sample_rows_of(
                &input.claim.gu,
                &param_registries[index],
                &input.message.seeds,
                &input.message.answers,
            )?);
        } else {
            sample_rows.push(vec![oracle_sample_row(
                omega.oracle,
                omega.group,
                &omega.u,
                &omega_seeds,
                0,
                omega_answers[0],
            )?]);
        }
    }
    let _profile_plain_rows = zkie_engine::profile::Scope::new("shared.plain_rows");
    let mut plain_rows = Vec::new();
    for row in &record_rows {
        plain_rows.push(relation_row_bat(row, &members[gu_bat as usize])?);
    }
    for row in &column_rows {
        plain_rows.push(relation_row_bat(row, &members[gu_bat as usize])?);
    }
    plain_rows.push(fold_plain_row(
        &gu,
        &omega,
        gu_bat,
        omega_bat,
        &psi,
        &zeta,
        matrix_rows,
        k_lo,
    )?);
    for (index, input) in param_inputs.iter().enumerate() {
        let member = members
            .iter()
            .find(|member| member.source_oracle == input.claim.gu.oracle)
            .ok_or(Error::StatementMismatch)?;
        let rows = norm_param_claim_rows(&input.identity, &input.claim, &input.message)?;
        for row in rows.record.iter().chain(rows.column.iter()) {
            plain_rows.push(relation_row_bat(row, member)?);
        }
        let _ = index;
    }
    drop(_profile_plain_rows);
    let mut trees = vec![
        (members[gu_bat as usize].source_oracle, Arc::new(gu_tree)),
        (
            members[omega_bat as usize].source_oracle,
            Arc::new(omega_tree),
        ),
    ];
    trees.extend(param_trees);
    let trees = ProximityTreeSource::Ready(trees);
    let fold = member_fold_shape(
        plan.proximity.code,
        plan.proximity.mask_code,
        &members,
        plan.proximity.shared_mask_tree.leaf_class,
    )?;
    let _profile_proximity = zkie_engine::profile::Scope::new("shared.proximity");
    let iopp = proximity_iopp_plan(
        &plan.proximity,
        &plan.send,
        plan.closure.as_ref(),
        plan.norm.as_ref(),
        ProximityGroupRole::Shared,
    )?;
    let proximity = proximity_prove(
        chain,
        fold,
        members,
        trees,
        &sample_rows,
        plain_rows,
        plan.proximity.shared_fresh,
        plan.proximity.shared_mask_tree,
        derive_run_seed(seed, 4, 0)?,
        plan.proximity.shared_base_positions,
        iopp.as_ref(),
    )?;
    drop(_profile_proximity);
    let messages = HiddenSharedMessages {
        gu_root,
        gu_seeds: gu_seeds.clone(),
        gu_answers: gu_answers.clone(),
        column_nonce: nonce,
        column_positions: positions.clone(),
        column_openings: gu_openings,
        omega_root,
        omega_seeds: omega_seeds.clone(),
        omega_answers: omega_answers.clone(),
        zeta: zeta.clone(),
        fold_start: proximity.messages.fold_start.clone(),
        fold_rounds: proximity.messages.fold_rounds.clone(),
        tail: proximity.messages.tail,
        seal: proximity.seal,
    };
    Ok(SharedProveResult {
        messages,
        witness: SharedWitness {
            gu_message,
            gu_tails,
            omega_message: omega_message.concat(),
            omega_tails,
            omega_salts,
        },
        points: SharedPoints { psi, zeta },
    })
}

fn relation_row_bat(row: &GuRelationRow, member: &ProximityMember) -> Result<BatRow> {
    if row.oracle != member.source_oracle || !row.mask.is_empty() {
        return Err(Error::StatementMismatch);
    }
    let mut out = BatRow::new(match row.kind {
        GuRelationKind::Sample => BatRowKind::Sample,
        GuRelationKind::Record | GuRelationKind::Column => BatRowKind::Plain,
    });
    out.add_term(
        member.bat_oracle,
        BatPart::Main,
        to_bat_table(&row.main, member.main_rows, member.main_code.ell)?,
    );
    out.target = row.target;
    Ok(out)
}

fn commit_rows_tree(
    tree_plan: RuntimeTreePlan,
    code: CodeParams,
    message_rows: &[Vec<Fp3>],
    tails: &[Vec<Fp3>],
    salts: &[Vec<u8>],
) -> Result<RuntimeTree> {
    if message_rows.len() != tails.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut codewords = Vec::with_capacity(message_rows.len());
    let _profile_encode = zkie_engine::profile::Scope::light("commit_rows_tree.encode");
    for (message, tail) in message_rows.iter().zip(tails) {
        codewords.push(encode_row_fp3(&code, message, tail)?);
    }
    drop(_profile_encode);
    RuntimeTree::commit(
        tree_plan,
        &[RuntimePartRows {
            main: codewords,
            mask: Vec::new(),
            main_deg: E_DEGREE as u32,
        }],
        salts,
    )
}

/// Verifier side of the shared stage.
fn verify_shared(
    plan: &HiddenLinPlan,
    reader: &IndexReader,
    first: &HiddenSendMessages,
    first_seal: &Digest,
    messages: &HiddenSharedMessages,
) -> Result<(Digest, Vec<GuRelationRow>, SharedPoints, ProximityBaseOutcome)> {
    let _profile = zkie_engine::profile::Scope::new("verify.shared");
    let k_col = shared_point_bits(plan)?;
    let mut chain = shared_start_chain(plan, reader, first_seal)?;
    let psi = draw_point_round(
        &mut chain,
        KIND_SEED,
        &[HIDDEN_SEND_SCHEMA_VERSION, 10],
        k_col,
    )?;
    let gamma = psi[plan.send.k_lo as usize..].to_vec();
    let eq_gamma = foldeq_weights(&gamma, reader.shape.v)?;
    let gu = plan.proximity.gu.clone();
    absorb_root_round(&mut chain, &messages.gu_root)?;
    let gu_bits = gu.sample_bits()?;
    let gu_seeds = draw_seed_set(&mut chain, gu_bits, gu.u.answers)?;
    if gu_seeds != messages.gu_seeds {
        return Err(Error::InvalidProof);
    }
    if messages.gu_answers.len() != gu.u.answers as usize {
        return Err(Error::ShapeMismatch);
    }
    absorb_answer_round(&mut chain, &messages.gu_answers)?;
    let (positions, _nonce) = draw_columns(
        &mut chain,
        plan.proximity.column_draws,
        plan.proximity.column_work_bits,
        reader.shape.code.log_n,
        Some(messages.column_nonce),
    )?;
    if positions != messages.column_positions {
        return Err(Error::InvalidProof);
    }
    let unique = unique_sorted(&positions);
    let _profile_columns = zkie_engine::profile::Scope::new("verify.shared.columns");
    let column_map = table_column_map(reader, &positions, &messages.column_openings)?;
    drop(_profile_columns);
    let _ = unique;
    let mut registry = GuRegistry::new();
    registry.receive(
        &gu,
        &GuNotice {
            root: messages.gu_root,
            oracle: gu.oracle,
            tree_number: gu.tree.tree_number,
            part: gu.part,
        },
    )?;
    let _profile_record_rows = zkie_engine::profile::Scope::new("verify.shared.record_rows");
    let record_seeds = record_seeds(&reader.shape, &reader.name(), reader.shape.answers as usize)?;
    let mut record_rows = Vec::with_capacity(reader.shape.answers as usize);
    for answer in 0..reader.shape.answers {
        record_rows.push(record_row(
            &gu,
            &registry,
            &reader.record,
            &reader.name(),
            &record_seeds,
            &eq_gamma,
            answer,
        )?);
    }
    drop(_profile_record_rows);
    let _profile_column_rows = zkie_engine::profile::Scope::new("verify.shared.column_rows");
    let mut column_rows = Vec::with_capacity(positions.len());
    for position in &positions {
        let column = column_for(&column_map, *position)?;
        column_rows.push(column_row(&gu, &registry, *position, &column, &eq_gamma)?);
    }
    drop(_profile_column_rows);

    let omega = plan.proximity.omega.clone();
    absorb_root_round(&mut chain, &messages.omega_root)?;
    let omega_bits = sample_cube_of(&omega.u)?.bits;
    let omega_seeds = draw_seed_set(&mut chain, omega_bits, omega.u.answers)?;
    if omega_seeds != messages.omega_seeds {
        return Err(Error::InvalidProof);
    }
    if messages.omega_answers.len() != omega.u.answers as usize {
        return Err(Error::ShapeMismatch);
    }
    absorb_answer_round(&mut chain, &messages.omega_answers)?;
    let zeta = draw_point_round(
        &mut chain,
        KIND_ALGEBRAIC,
        &[HIDDEN_SEND_SCHEMA_VERSION, 11],
        omega.matrix_rows.trailing_zeros(),
    )?;
    if zeta != messages.zeta {
        return Err(Error::InvalidProof);
    }
    let mut members = vec![
        ProximityMember {
            bat_oracle: 0,
            source_oracle: gu.oracle,
            main_code: gu.u.code,
            mask_code: plan.proximity.mask_code,
            main_rows: gu.u.main_rows,
            mask_rows: 0,
            answers: gu.u.answers,
            root: messages.gu_root,
            tree_plan: gu.tree.clone(),
            message: Vec::new(),
            tails: Vec::new(),
            mask_message: Vec::new(),
            mask_tails: Vec::new(),
        },
        ProximityMember {
            bat_oracle: 1,
            source_oracle: omega.oracle,
            main_code: omega.u.code,
            mask_code: plan.proximity.mask_code,
            main_rows: omega.u.main_rows,
            mask_rows: 0,
            answers: omega.u.answers,
            root: messages.omega_root,
            tree_plan: omega.tree.clone(),
            message: Vec::new(),
            tails: Vec::new(),
            mask_message: Vec::new(),
            mask_tails: Vec::new(),
        },
    ];
    let param_inputs = norm_param_claim_inputs(plan, first, None)?;
    for input in &param_inputs {
        let gu = &input.claim.gu;
        members.push(ProximityMember {
            bat_oracle: 0,
            source_oracle: gu.oracle,
            main_code: gu.u.code,
            mask_code: plan.proximity.mask_code,
            main_rows: gu.u.main_rows,
            mask_rows: 0,
            answers: gu.u.answers,
            root: input.message.root,
            tree_plan: gu.tree.clone(),
            message: Vec::new(),
            tails: Vec::new(),
            mask_message: Vec::new(),
            mask_tails: Vec::new(),
        });
    }
    members.sort_by_key(|member| std::cmp::Reverse((member.main_rows, member.source_oracle)));
    let mut gu_bat = None;
    let mut omega_bat = None;
    for (index, member) in members.iter_mut().enumerate() {
        member.bat_oracle = index as u64;
        if member.source_oracle == gu.oracle {
            gu_bat = Some(member.bat_oracle);
        }
        if member.source_oracle == omega.oracle {
            omega_bat = Some(member.bat_oracle);
        }
    }
    let gu_bat = gu_bat.ok_or(Error::StatementMismatch)?;
    let omega_bat = omega_bat.ok_or(Error::StatementMismatch)?;
    let mut param_registries = Vec::with_capacity(param_inputs.len());
    for input in &param_inputs {
        param_registries.push(norm_param_registry(&input.claim, &input.message)?);
    }
    let param_of = |oracle: u64| {
        param_inputs
            .iter()
            .position(|input| input.claim.gu.oracle == oracle)
    };
    let mut sample_rows = Vec::with_capacity(members.len());
    for member in &members {
        if member.source_oracle == gu.oracle {
            sample_rows.push(sample_rows_of(&gu, &registry, &gu_seeds, &messages.gu_answers)?);
        } else if let Some(index) = param_of(member.source_oracle) {
            let input = &param_inputs[index];
            sample_rows.push(sample_rows_of(
                &input.claim.gu,
                &param_registries[index],
                &input.message.seeds,
                &input.message.answers,
            )?);
        } else {
            let mut rows = Vec::with_capacity(messages.omega_answers.len());
            for (answer, target) in messages.omega_answers.iter().enumerate() {
                rows.push(oracle_sample_row(
                    omega.oracle,
                    omega.group,
                    &omega.u,
                    &omega_seeds,
                    answer as u32,
                    *target,
                )?);
            }
            sample_rows.push(rows);
        }
    }
    let _profile_plain_rows = zkie_engine::profile::Scope::new("verify.shared.plain_rows");
    let mut plain_rows = Vec::new();
    for row in &record_rows {
        plain_rows.push(relation_row_bat(row, &members[gu_bat as usize])?);
    }
    for row in &column_rows {
        plain_rows.push(relation_row_bat(row, &members[gu_bat as usize])?);
    }
    plain_rows.push(fold_plain_row(
        &gu,
        &omega,
        gu_bat,
        omega_bat,
        &psi,
        &zeta,
        omega.matrix_rows,
        omega.k_lo,
    )?);
    let mut param_claim_rows = Vec::new();
    for input in &param_inputs {
        let member = members
            .iter()
            .find(|member| member.source_oracle == input.claim.gu.oracle)
            .ok_or(Error::StatementMismatch)?;
        let rows = norm_param_claim_rows(&input.identity, &input.claim, &input.message)?;
        for row in rows.record.iter().chain(rows.column.iter()) {
            plain_rows.push(relation_row_bat(row, member)?);
        }
        param_claim_rows.push(rows);
    }
    drop(_profile_plain_rows);
    let fold = member_fold_shape(
        plan.proximity.code,
        plan.proximity.mask_code,
        &members,
        plan.proximity.shared_mask_tree.leaf_class,
    )?;
    let iopp = proximity_iopp_plan(
        &plan.proximity,
        &plan.send,
        plan.closure.as_ref(),
        plan.norm.as_ref(),
        ProximityGroupRole::Shared,
    )?;
    let _profile_proximity = zkie_engine::profile::Scope::new("verify.shared.proximity");
    let proximity = proximity_verify(
        chain,
        fold,
        &members,
        &sample_rows,
        plain_rows,
        &ProximityMessages {
            fold_start: messages.fold_start.clone(),
            fold_rounds: messages.fold_rounds.clone(),
            tail: messages.tail.clone(),
        },
        plan.proximity.shared_fresh,
        plan.proximity.shared_mask_tree,
        plan.proximity.shared_base_positions,
        iopp.as_ref(),
    )?;
    drop(_profile_proximity);
    if proximity.seal != messages.seal {
        return Err(Error::InvalidProof);
    }
    let mut claims = sample_rows.into_iter().flatten().collect::<Vec<_>>();
    claims.extend(record_rows);
    claims.extend(column_rows);
    for rows in param_claim_rows {
        claims.extend(rows.record);
        claims.extend(rows.column);
    }
    Ok((
        proximity.seal,
        claims,
        SharedPoints { psi, zeta },
        proximity.base,
    ))
}

fn sample_rows_of(
    gu: &GuShape,
    registry: &GuRegistry,
    seeds: &[SampleSeed],
    answers: &[Fp3],
) -> Result<Vec<GuRelationRow>> {
    let mut rows = Vec::with_capacity(answers.len());
    for (answer, target) in answers.iter().enumerate() {
        rows.push(sample_row(gu, registry, seeds, answer as u32, *target)?);
    }
    Ok(rows)
}
// ---------------------------------------------------------------------------
// Second stage
// ---------------------------------------------------------------------------

fn second_chain(
    plan: &HiddenLinPlan,
    reader: &IndexReader,
    first_query: &Digest,
    shared_seal: &Digest,
    omega_root: &Digest,
    shared_point: &[Fp3],
    zeta: &[Fp3],
) -> Result<Chain> {
    let mut writer = CanonicalWriter::new();
    writer.put_u64(HIDDEN_SEND_SCHEMA_VERSION);
    writer.put_digest(first_query);
    writer.put_digest(shared_seal);
    writer.put_digest(omega_root);
    writer.put_u64(shared_point.len() as u64);
    for value in shared_point {
        writer.put_fp3(value);
    }
    writer.put_u64(zeta.len() as u64);
    for value in zeta {
        writer.put_fp3(value);
    }
    let statement = writer.into_bytes();
    let mut chain = Chain::begin(
        FINAL_CHAIN_LABEL,
        &plan.parameters_digest(),
        &statement,
        &reader.name(),
        None,
    )?;
    absorb_digest_round(
        &mut chain,
        2,
        &[plan.parameters_digest(), *first_query, *shared_seal, *omega_root],
        &[shared_point.len() as u64, zeta.len() as u64],
    )?;
    Ok(chain)
}

struct OwnProveResult {
    messages: HiddenSecondMessages,
    claims: Vec<GuRelationRow>,
}

/// The own-group member list, sorted by descending main rows so that BAT's
/// block order and the FOLD layout agree. `bat_of` maps a source oracle to its
/// BAT registration id.
fn own_members(
    plan: &HiddenLinPlan,
    first: &HiddenSendMessages,
    shared: &HiddenSharedMessages,
    witness: Option<&HiddenSendWitness>,
    omega_witness: Option<(&[Fp3], &[Vec<Fp3>])>,
) -> Result<(Vec<ProximityMember>, Vec<u64>)> {
    let mut members = Vec::with_capacity(plan.send.positions() as usize + 1);
    for position in 0..plan.send.positions() {
        let root = first
            .group
            .entries
            .get(position as usize)
            .ok_or(Error::ShapeMismatch)?
            .root;
        let (message, tails) = match witness {
            Some(witness) => {
                let message = witness
                    .messages
                    .get(position as usize)
                    .ok_or(Error::ShapeMismatch)?
                    .clone();
                let tail_len = usize::try_from(plan.send.w.code.r).map_err(|_| Error::ResourceLimit)?;
                let tail = witness
                    .tails
                    .get(position as usize)
                    .ok_or(Error::ShapeMismatch)?;
                let rows = usize::try_from(plan.send.w.main_rows).map_err(|_| Error::ResourceLimit)?;
                let mut tails = Vec::with_capacity(rows);
                for row in 0..rows {
                    tails.push(
                        tail.get(row * tail_len..(row + 1) * tail_len)
                            .ok_or(Error::ShapeMismatch)?
                            .to_vec(),
                    );
                }
                (message, tails)
            }
            None => (Vec::new(), Vec::new()),
        };
        members.push(ProximityMember {
            bat_oracle: 0,
            source_oracle: plan.send.oracle(position)?,
            main_code: plan.send.w.code,
            mask_code: plan.proximity.mask_code,
            main_rows: plan.send.w.main_rows,
            mask_rows: 0,
            answers: plan.send.w.answers,
            root,
            tree_plan: plan.send.w.tree_plan(position)?,
            message,
            tails,
            mask_message: Vec::new(),
            mask_tails: Vec::new(),
        });
    }
    let omega = plan.proximity.omega.clone();
    let (message, tails) = match omega_witness {
        Some((message, tails)) => {
            let ell = usize::try_from(omega.u.code.ell).map_err(|_| Error::ResourceLimit)?;
            let rows = usize::try_from(omega.u.main_rows).map_err(|_| Error::ResourceLimit)?;
            if message.len() != ell
                || tails.len() != rows
            {
                return Err(Error::ShapeMismatch);
            }
            let mut message_rows = Vec::with_capacity(rows);
            for row in 0..rows {
                let start = row.checked_mul(ell).ok_or(Error::ResourceLimit)?;
                let end = start.checked_add(ell).ok_or(Error::ResourceLimit)?;
                message_rows.push(
                    message
                        .get(start..end)
                        .ok_or(Error::ShapeMismatch)?
                        .to_vec(),
                );
            }
            (message_rows, tails.to_vec())
        }
        None => (Vec::new(), Vec::new()),
    };
    members.push(ProximityMember {
        bat_oracle: 0,
        source_oracle: omega.oracle,
        main_code: omega.u.code,
        mask_code: plan.proximity.mask_code,
        main_rows: omega.u.main_rows,
        mask_rows: 0,
        answers: omega.u.answers,
        root: shared.omega_root,
        tree_plan: omega.tree.clone(),
        message,
        tails,
        mask_message: Vec::new(),
        mask_tails: Vec::new(),
    });
    if let Some(dot) = plan.send.dot.as_ref().filter(|_| !plan.send.activation_source.is_norm_hd()) {
        let message = first.dot.as_ref().ok_or(Error::ShapeMismatch)?;
        if message.tree != dot.tree_plan()? {
            return Err(Error::StatementMismatch);
        }
        let (main_message, main_tails) = match witness {
            Some(witness) => {
                let map = witness.dot.as_ref().ok_or(Error::ShapeMismatch)?;
                map.check(dot)?;
                (map.message.clone(), map.tails.clone())
            }
            None => (Vec::new(), Vec::new()),
        };
        members.push(ProximityMember {
            bat_oracle: 0,
            source_oracle: dot.oracle,
            main_code: dot.shape.code,
            mask_code: plan.proximity.mask_code,
            main_rows: dot.shape.main_rows,
            mask_rows: 0,
            answers: dot.shape.answers,
            root: message.root,
            tree_plan: message.tree.clone(),
            message: main_message,
            tails: main_tails,
            mask_message: Vec::new(),
            mask_tails: Vec::new(),
        });
    }
    if let (Some(index), Some(message)) = (&plan.send.index_mask, &first.index_mask) {
        let (mask_message, mask_tails) = match witness {
            Some(witness) => {
                let mask_witness = witness
                    .index_mask
                    .as_ref()
                    .ok_or(Error::ShapeMismatch)?;
                mask_witness.check(index)?;
                (mask_witness.rows.clone(), mask_witness.tails.clone())
            }
            None => (Vec::new(), Vec::new()),
        };
        members.push(ProximityMember {
            bat_oracle: 0,
            source_oracle: index.oracle,
            main_code: plan.proximity.code,
            mask_code: index.mask_code,
            main_rows: 0,
            mask_rows: index.mask_rows,
            answers: index.answers,
            root: message.root,
            tree_plan: index.tree.clone(),
            message: Vec::new(),
            tails: Vec::new(),
            mask_message,
            mask_tails,
        });
    }
    if let (Some(closure_plan), Some(closure_first)) = (&plan.closure, &first.closure) {
        let positions = plan.send.positions();
        for position in 0..positions {
            let root = closure_first
                .slack
                .group
                .entries
                .get(position as usize)
                .ok_or(Error::ShapeMismatch)?
                .root;
            let (message, tails) = match witness {
                Some(witness) => {
                    let closure = witness.closure.as_ref().ok_or(Error::ShapeMismatch)?;
                    let message = closure
                        .slack
                        .messages
                        .get(position as usize)
                        .ok_or(Error::ShapeMismatch)?
                        .clone();
                    let rows = usize::try_from(closure_plan.slack.main_rows)
                        .map_err(|_| Error::ResourceLimit)?;
                    let tail_len = usize::try_from(closure_plan.slack.code.r)
                        .map_err(|_| Error::ResourceLimit)?;
                    let flat = closure
                        .slack_tails
                        .get(position as usize)
                        .ok_or(Error::ShapeMismatch)?;
                    let mut tails = Vec::with_capacity(rows);
                    for row in 0..rows {
                        tails.push(
                            flat.get(row * tail_len..(row + 1) * tail_len)
                                .ok_or(Error::ShapeMismatch)?
                                .to_vec(),
                        );
                    }
                    (message, tails)
                }
                None => (Vec::new(), Vec::new()),
            };
            members.push(ProximityMember {
                bat_oracle: 0,
                source_oracle: closure_plan
                    .slack_oracle_base
                    .checked_add(position as u64)
                    .ok_or(Error::ResourceLimit)?,
                main_code: closure_plan.slack.code,
                mask_code: plan.proximity.mask_code,
                main_rows: closure_plan.slack.main_rows,
                mask_rows: 0,
                answers: closure_plan.slack.answers,
                root,
                tree_plan: closure_plan.slack.tree_plan(position)?,
                message,
                tails,
                mask_message: Vec::new(),
                mask_tails: Vec::new(),
            });
        }
        for position in 0..positions {
            let root = closure_first
                .bits
                .group
                .entries
                .get(position as usize)
                .ok_or(Error::ShapeMismatch)?
                .root;
            let (message, tails) = match witness {
                Some(witness) => {
                    let closure = witness.closure.as_ref().ok_or(Error::ShapeMismatch)?;
                    let message = closure
                        .bits
                        .messages
                        .get(position as usize)
                        .ok_or(Error::ShapeMismatch)?
                        .clone();
                    let rows = usize::try_from(closure_plan.bits.main_rows)
                        .map_err(|_| Error::ResourceLimit)?;
                    let tail_len = usize::try_from(closure_plan.bits.code.r)
                        .map_err(|_| Error::ResourceLimit)?;
                    let flat = closure
                        .bits_tails
                        .get(position as usize)
                        .ok_or(Error::ShapeMismatch)?;
                    let mut tails = Vec::with_capacity(rows);
                    for row in 0..rows {
                        tails.push(
                            flat.get(row * tail_len..(row + 1) * tail_len)
                                .ok_or(Error::ShapeMismatch)?
                                .to_vec(),
                        );
                    }
                    (message, tails)
                }
                None => (Vec::new(), Vec::new()),
            };
            members.push(ProximityMember {
                bat_oracle: 0,
                source_oracle: closure_plan
                    .bits_oracle_base
                    .checked_add(position as u64)
                    .ok_or(Error::ResourceLimit)?,
                main_code: closure_plan.bits.code,
                mask_code: plan.proximity.mask_code,
                main_rows: closure_plan.bits.main_rows,
                mask_rows: 0,
                answers: closure_plan.bits.answers,
                root,
                tree_plan: closure_plan.bits.tree_plan(position)?,
                message,
                tails,
                mask_message: Vec::new(),
                mask_tails: Vec::new(),
            });
        }
        for position in 0..positions {
            let root = closure_first
                .map
                .group
                .entries
                .get(position as usize)
                .ok_or(Error::ShapeMismatch)?
                .root;
            let (message, tails) = match witness {
                Some(witness) => {
                    let closure = witness.closure.as_ref().ok_or(Error::ShapeMismatch)?;
                    let message = closure
                        .map
                        .messages
                        .get(position as usize)
                        .ok_or(Error::ShapeMismatch)?
                        .clone();
                    let rows = usize::try_from(closure_plan.map.main_rows)
                        .map_err(|_| Error::ResourceLimit)?;
                    let tail_len = usize::try_from(closure_plan.map.code.r)
                        .map_err(|_| Error::ResourceLimit)?;
                    let flat = closure
                        .map_tails
                        .get(position as usize)
                        .ok_or(Error::ShapeMismatch)?;
                    let mut tails = Vec::with_capacity(rows);
                    for row in 0..rows {
                        tails.push(
                            flat.get(row * tail_len..(row + 1) * tail_len)
                                .ok_or(Error::ShapeMismatch)?
                                .to_vec(),
                        );
                    }
                    (message, tails)
                }
                None => (Vec::new(), Vec::new()),
            };
            members.push(ProximityMember {
                bat_oracle: 0,
                source_oracle: closure_plan
                    .map_oracle_base
                    .checked_add(position as u64)
                    .ok_or(Error::ResourceLimit)?,
                main_code: closure_plan.map.code,
                mask_code: plan.proximity.mask_code,
                main_rows: closure_plan.map.main_rows,
                mask_rows: 0,
                answers: closure_plan.map.answers,
                root,
                tree_plan: closure_plan.map.tree_plan(position)?,
                message,
                tails,
                mask_message: Vec::new(),
                mask_tails: Vec::new(),
            });
        }
        let (mask_message, mask_tails) = match witness {
            Some(witness) => {
                let closure = witness.closure.as_ref().ok_or(Error::ShapeMismatch)?;
                (closure.mask.rows.clone(), closure.mask.tails.clone())
            }
            None => (Vec::new(), Vec::new()),
        };
        members.push(ProximityMember {
            bat_oracle: 0,
            source_oracle: closure_plan.mask.oracle,
            main_code: plan.proximity.code,
            mask_code: closure_plan.mask.mask_code,
            main_rows: 0,
            mask_rows: closure_plan.mask.mask_rows,
            answers: closure_plan.mask.answers,
            root: closure_first.mask.root,
            tree_plan: closure_plan.mask.tree.clone(),
            message: Vec::new(),
            tails: Vec::new(),
            mask_message,
            mask_tails,
        });
    }
    if let Some(norm_plan) = &plan.norm {
        let norm_first = first.norm.as_ref().ok_or(Error::ShapeMismatch)?;
        let norm_witness = witness.and_then(|witness| witness.norm.as_ref());
        let positions = plan.send.positions();
        let mut groups = vec![
            (&norm_first.x, &norm_plan.x, norm_witness.map(|w| &w.x)),
            (&norm_first.sum, &norm_plan.sum, norm_witness.map(|w| &w.sum)),
            (&norm_first.squares, &norm_plan.squares, norm_witness.map(|w| &w.squares)),
            (&norm_first.mean, &norm_plan.mean, norm_witness.map(|w| &w.mean)),
            (&norm_first.variance, &norm_plan.variance, norm_witness.map(|w| &w.variance)),
        ];
        if let (Some(round_message), Some(round_plan)) =
            (&norm_first.round, &norm_plan.round)
        {
            groups.push((
                round_message,
                &round_plan.group,
                norm_witness.and_then(|witness| witness.round.as_ref()),
            ));
        }
        if let (Some(output_message), Some(output_plan)) =
            (&norm_first.output, &norm_plan.output)
        {
            let output_witness = norm_witness.and_then(|witness| witness.output.as_ref());
            groups.push((
                &output_message.broadcast,
                &output_plan.broadcast,
                output_witness.map(|witness| &witness.broadcast),
            ));
            groups.push((
                &output_message.product,
                &output_plan.product,
                output_witness.map(|witness| &witness.product),
            ));
            groups.push((
                &output_message.image,
                &output_plan.image,
                output_witness.map(|witness| &witness.image),
            ));
            groups.push((
                &output_message.out,
                &output_plan.out,
                output_witness.map(|witness| &witness.out),
            ));
        }
        if let (Some(tail_message), Some(output_plan)) = (&norm_first.tail, &norm_plan.output) {
            if let Some(tail_plan) = &output_plan.tail {
                let tail_witness = norm_witness
                    .and_then(|witness| witness.output.as_ref())
                    .and_then(|witness| witness.tail.as_ref());
                groups.push((
                    &tail_message.group,
                    &tail_plan.group,
                    tail_witness.map(|witness| &witness.committed),
                ));
            }
        }
        for (message, group_plan, group_witness) in groups {
            if message.message.group.group != group_plan.group
                || message.message.roots.len() != positions as usize
            {
                return Err(Error::StatementMismatch);
            }
            for position in 0..positions {
                let root = message
                    .message
                    .group
                    .entries
                    .get(position as usize)
                    .ok_or(Error::ShapeMismatch)?
                    .root;
                let (message_rows, tails) = match group_witness {
                    Some(group_witness) => {
                        let rows = group_witness
                            .messages
                            .get(position as usize)
                            .ok_or(Error::ShapeMismatch)?
                            .clone();
                        let row_count = usize::try_from(group_plan.shape.main_rows)
                            .map_err(|_| Error::ResourceLimit)?;
                        let tail_len = usize::try_from(group_plan.shape.code.r)
                            .map_err(|_| Error::ResourceLimit)?;
                        let flat = group_witness
                            .tails
                            .get(position as usize)
                            .ok_or(Error::ShapeMismatch)?;
                        let mut tails = Vec::with_capacity(row_count);
                        for row in 0..row_count {
                            tails.push(
                                flat.get(row * tail_len..(row + 1) * tail_len)
                                    .ok_or(Error::ShapeMismatch)?
                                    .to_vec(),
                            );
                        }
                        (rows, tails)
                    }
                    None => (Vec::new(), Vec::new()),
                };
                members.push(ProximityMember {
                    bat_oracle: 0,
                    source_oracle: group_plan
                        .oracle_base
                        .checked_add(position as u64)
                        .ok_or(Error::ResourceLimit)?,
                    main_code: group_plan.shape.code,
                    mask_code: plan.proximity.mask_code,
                    main_rows: group_plan.shape.main_rows,
                    mask_rows: 0,
                    answers: group_plan.shape.answers,
                    root,
                    tree_plan: group_plan.shape.tree_plan(position)?,
                    message: message_rows,
                    tails,
                    mask_message: Vec::new(),
                    mask_tails: Vec::new(),
                });
            }
        }
        if let Some(mask_plan) = &norm_plan.mask {
            let mask_first = norm_first.mask.as_ref().ok_or(Error::ShapeMismatch)?;
            if mask_first.tree != mask_plan.tree {
                return Err(Error::StatementMismatch);
            }
            let (mask_rows, mask_tails) = match norm_witness {
                Some(witness) => match &witness.mask {
                    Some(mask) => (mask.committed.rows.clone(), mask.committed.tails.clone()),
                    None => (Vec::new(), Vec::new()),
                },
                None => (Vec::new(), Vec::new()),
            };
            members.push(ProximityMember {
                bat_oracle: 0,
                source_oracle: mask_plan.oracle,
                main_code: plan.proximity.code,
                mask_code: mask_plan.mask_code,
                main_rows: 0,
                mask_rows: mask_plan.mask_rows,
                answers: mask_plan.answers,
                root: mask_first.root,
                tree_plan: mask_plan.tree.clone(),
                message: Vec::new(),
                tails: Vec::new(),
                mask_message: mask_rows,
                mask_tails,
            });
        }
        if let Some(lookup_plan) = &norm_plan.lookup {
            let lookup_first = norm_first.lookup.as_ref().ok_or(Error::ShapeMismatch)?;
            let commit = &lookup_plan.commit;
            let bits_entry = lookup_first
                .bits
                .message
                .group
                .entries
                .first()
                .ok_or(Error::ShapeMismatch)?;
            if bits_entry.tree != commit.bits_shape.tree_plan(0)?
                || lookup_first.mask.tree != commit.mask_tree
            {
                return Err(Error::StatementMismatch);
            }
            let lookup_messages = norm_witness
                .and_then(|witness| witness.lookup.as_ref())
                .and_then(|lookup| lookup.messages.as_ref());
            let row_count = usize::try_from(commit.bits_shape.main_rows)
                .map_err(|_| Error::ResourceLimit)?;
            let tail_len = usize::try_from(commit.bits_shape.code.r)
                .map_err(|_| Error::ResourceLimit)?;
            let (bits_message, bits_tails) = match lookup_messages {
                Some(messages) => {
                    let message = messages
                        .bits
                        .messages
                        .first()
                        .ok_or(Error::ShapeMismatch)?
                        .clone();
                    let mut tails = Vec::with_capacity(row_count);
                    for row in 0..row_count {
                        tails.push(
                            messages
                                .bits_tails
                                .get(row * tail_len..(row + 1) * tail_len)
                                .ok_or(Error::ShapeMismatch)?
                                .to_vec(),
                        );
                    }
                    (message, tails)
                }
                None => (Vec::new(), Vec::new()),
            };
            let (lookup_mask_rows, lookup_mask_tails) = match lookup_messages {
                Some(messages) => (messages.mask_rows.clone(), messages.mask_tails.clone()),
                None => (Vec::new(), Vec::new()),
            };
            members.push(ProximityMember {
                bat_oracle: 0,
                source_oracle: commit.bits_oracle,
                main_code: commit.bits_shape.code,
                mask_code: plan.proximity.mask_code,
                main_rows: commit.bits_shape.main_rows,
                mask_rows: 0,
                answers: commit.bits_shape.answers,
                root: lookup_first
                    .bits
                    .message
                    .roots
                    .first()
                    .copied()
                    .ok_or(Error::ShapeMismatch)?,
                tree_plan: commit.bits_shape.tree_plan(0)?,
                message: bits_message,
                tails: bits_tails,
                mask_message: Vec::new(),
                mask_tails: Vec::new(),
            });
            members.push(ProximityMember {
                bat_oracle: 0,
                source_oracle: commit.mask_oracle,
                main_code: plan.proximity.code,
                mask_code: commit.mask_code,
                main_rows: 0,
                mask_rows: commit.mask_rows,
                answers: 1,
                root: lookup_first.mask.root,
                tree_plan: commit.mask_tree.clone(),
                message: Vec::new(),
                tails: Vec::new(),
                mask_message: lookup_mask_rows,
                mask_tails: lookup_mask_tails,
            });
        }

        // frozen-table lookup: own members
        if let Some(lookup_plan) = &norm_plan.rstd {
            let lookup_first = norm_first.rstd.as_ref().ok_or(Error::ShapeMismatch)?;
            let commit = &lookup_plan.commit;
            let bits_entry = lookup_first
                .bits
                .message
                .group
                .entries
                .first()
                .ok_or(Error::ShapeMismatch)?;
            if bits_entry.tree != commit.bits_shape.tree_plan(0)?
                || lookup_first.mask.tree != commit.mask_tree
            {
                return Err(Error::StatementMismatch);
            }
            let lookup_messages = norm_witness
                .and_then(|witness| witness.rstd.as_ref())
                .and_then(|lookup| lookup.messages.as_ref());
            let row_count = usize::try_from(commit.bits_shape.main_rows)
                .map_err(|_| Error::ResourceLimit)?;
            let tail_len = usize::try_from(commit.bits_shape.code.r)
                .map_err(|_| Error::ResourceLimit)?;
            let (bits_message, bits_tails) = match lookup_messages {
                Some(messages) => {
                    let message = messages
                        .bits
                        .messages
                        .first()
                        .ok_or(Error::ShapeMismatch)?
                        .clone();
                    let mut tails = Vec::with_capacity(row_count);
                    for row in 0..row_count {
                        tails.push(
                            messages
                                .bits_tails
                                .get(row * tail_len..(row + 1) * tail_len)
                                .ok_or(Error::ShapeMismatch)?
                                .to_vec(),
                        );
                    }
                    (message, tails)
                }
                None => (Vec::new(), Vec::new()),
            };
            let (lookup_mask_rows, lookup_mask_tails) = match lookup_messages {
                Some(messages) => (messages.mask_rows.clone(), messages.mask_tails.clone()),
                None => (Vec::new(), Vec::new()),
            };
            members.push(ProximityMember {
                bat_oracle: 0,
                source_oracle: commit.bits_oracle,
                main_code: commit.bits_shape.code,
                mask_code: plan.proximity.mask_code,
                main_rows: commit.bits_shape.main_rows,
                mask_rows: 0,
                answers: commit.bits_shape.answers,
                root: lookup_first
                    .bits
                    .message
                    .roots
                    .first()
                    .copied()
                    .ok_or(Error::ShapeMismatch)?,
                tree_plan: commit.bits_shape.tree_plan(0)?,
                message: bits_message,
                tails: bits_tails,
                mask_message: Vec::new(),
                mask_tails: Vec::new(),
            });
            members.push(ProximityMember {
                bat_oracle: 0,
                source_oracle: commit.mask_oracle,
                main_code: plan.proximity.code,
                mask_code: commit.mask_code,
                main_rows: 0,
                mask_rows: commit.mask_rows,
                answers: 1,
                root: lookup_first.mask.root,
                tree_plan: commit.mask_tree.clone(),
                message: Vec::new(),
                tails: Vec::new(),
                mask_message: lookup_mask_rows,
                mask_tails: lookup_mask_tails,
            });
        }
    }
    for input in norm_param_claim_inputs(plan, first, witness.and_then(|w| w.norm.as_ref()))? {
        let gu = &input.claim.gu;
        let (message, tails) = match &input.material {
            Some(material) => (material.message.clone(), material.tails.clone()),
            None => (Vec::new(), Vec::new()),
        };
        members.push(ProximityMember {
            bat_oracle: 0,
            source_oracle: gu.oracle,
            main_code: gu.u.code,
            mask_code: plan.proximity.mask_code,
            main_rows: gu.u.main_rows,
            mask_rows: 0,
            answers: gu.u.answers,
            root: input.message.root,
            tree_plan: gu.tree.clone(),
            message,
            tails,
            mask_message: Vec::new(),
            mask_tails: Vec::new(),
        });
    }
    // The isolated FOLD fixture places blocks in registration order, so the
    // member order must follow the BAT block order: larger main blocks first,
    // and among mask-only members the larger mask block first.
    members.sort_by_key(|member| {
        std::cmp::Reverse((member.main_rows, member.mask_rows, member.source_oracle))
    });
    let mut order = Vec::with_capacity(members.len());
    for (index, member) in members.iter_mut().enumerate() {
        member.bat_oracle = index as u64;
        order.push(member.source_oracle);
    }
    Ok((members, order))
}

fn own_sample_rows(
    plan: &HiddenLinPlan,
    members: &[ProximityMember],
    first: &HiddenSendMessages,
    shared: &HiddenSharedMessages,
) -> Result<Vec<Vec<GuRelationRow>>> {
    let shape = plan.send.w.oracle_shape();
    zkie_engine::scheduling::try_map_bounded(
        members,
        zkie_engine::scheduling::DEFAULT_UNITS_IN_FLIGHT,
        |member| -> Result<Vec<GuRelationRow>> {
        if let Some(norm_plan) = &plan.norm {
            if let Some(params_plan) = &norm_plan.params {
                let norm_first = first.norm.as_ref().ok_or(Error::ShapeMismatch)?;
                let message_params = norm_first.params.as_ref().ok_or(Error::ShapeMismatch)?;
                let claims = [
                    (
                        params_plan.gain_claim.as_ref(),
                        Some(&message_params.gain),
                    ),
                    (
                        params_plan.bias_claim.as_ref(),
                        message_params.bias.as_ref(),
                    ),
                ];
                for (claim, message) in claims {
                    let (claim, message) = match (claim, message) {
                        (Some(claim), Some(message)) => (claim, message),
                        _ => continue,
                    };
                    if member.source_oracle == claim.gu.oracle {
                        let registry = norm_param_registry(claim, message)?;
                        return Ok(sample_rows_of(
                            &claim.gu,
                            &registry,
                            &message.seeds,
                            &message.answers,
                        )?);
                    }
                }
            }
        }
        if let Some(norm_plan) = &plan.norm {
            let norm_first = first.norm.as_ref().ok_or(Error::ShapeMismatch)?;
            let groups = norm_group_plans(norm_plan);
            let mut messages = vec![
                &norm_first.x,
                &norm_first.sum,
                &norm_first.squares,
                &norm_first.mean,
                &norm_first.variance,
            ];
            if let Some(round) = &norm_first.round {
                messages.push(round);
            }
            let positions = plan.send.positions() as u64;
            for (group_plan, message) in groups.iter().zip(messages.iter()) {
                let base = group_plan.oracle_base;
                if member.source_oracle >= base
                    && member.source_oracle < base.saturating_add(positions)
                {
                    let position = (member.source_oracle - base) as usize;
                    let target = *message
                        .sample
                        .answers
                        .get(position)
                        .ok_or(Error::ShapeMismatch)?;
                    return Ok(vec![oracle_sample_row(
                        member.source_oracle,
                        plan.send.group,
                        &group_plan.shape.oracle_shape(),
                        &message.sample.seed,
                        0,
                        target,
                    )?]);
                }
            }
            if let (Some(output_message), Some(output_plan)) =
                (&norm_first.output, &norm_plan.output)
            {
                let outputs = [
                    (&output_message.broadcast, &output_plan.broadcast),
                    (&output_message.product, &output_plan.product),
                    (&output_message.image, &output_plan.image),
                    (&output_message.out, &output_plan.out),
                ];
                let positions = plan.send.positions() as u64;
                for (message, group_plan) in outputs {
                    let base = group_plan.oracle_base;
                    if member.source_oracle >= base
                        && member.source_oracle < base.saturating_add(positions)
                    {
                        let position = (member.source_oracle - base) as usize;
                        let target = *message
                            .sample
                            .answers
                            .get(position)
                            .ok_or(Error::ShapeMismatch)?;
                        return Ok(vec![oracle_sample_row(
                            member.source_oracle,
                            plan.send.group,
                            &group_plan.shape.oracle_shape(),
                            &message.sample.seed,
                            0,
                            target,
                        )?]);
                    }
                }
            }
            if let (Some(tail_message), Some(output_plan)) =
                (&norm_first.tail, &norm_plan.output)
            {
                if let Some(tail_plan) = &output_plan.tail {
                    let base = tail_plan.group.oracle_base;
                    let positions = plan.send.positions() as u64;
                    if member.source_oracle >= base
                        && member.source_oracle < base.saturating_add(positions)
                    {
                        let position = (member.source_oracle - base) as usize;
                        let target = *tail_message
                            .group
                            .sample
                            .answers
                            .get(position)
                            .ok_or(Error::ShapeMismatch)?;
                        return Ok(vec![oracle_sample_row(
                            member.source_oracle,
                            plan.send.group,
                            &tail_plan.group.shape.oracle_shape(),
                            &tail_message.group.sample.seed,
                            0,
                            target,
                        )?]);
                    }
                }
            }
            if let Some(mask_plan) = &norm_plan.mask {
                if member.source_oracle == mask_plan.oracle {
                    let message = norm_first.mask.as_ref().ok_or(Error::ShapeMismatch)?;
                    let target = *message
                        .sample
                        .answers
                        .first()
                        .ok_or(Error::ShapeMismatch)?;
                    let mut row = norm_mask_sample_row(mask_plan, message, target)?;
                    row.group = plan.send.group;
                    return Ok(vec![row]);
                }
            }
            if let Some(lookup_plan) = &norm_plan.lookup {
                let lookup_first = norm_first.lookup.as_ref().ok_or(Error::ShapeMismatch)?;
                let commit = &lookup_plan.commit;
                if member.source_oracle == commit.bits_oracle {
                    let target = *lookup_first
                        .bits
                        .sample
                        .answers
                        .first()
                        .ok_or(Error::ShapeMismatch)?;
                    return Ok(vec![oracle_sample_row(
                        member.source_oracle,
                        lookup_first.bits.message.group.group,
                        &commit.bits_shape.oracle_shape(),
                        &lookup_first.bits.sample.seed,
                        0,
                        target,
                    )?]);
                }
                if member.source_oracle == commit.mask_oracle {
                    let target = *lookup_first
                        .mask
                        .sample
                        .answers
                        .first()
                        .ok_or(Error::ShapeMismatch)?;
                    let row =
                        lookup_mask_sample_row(commit, &lookup_first.mask, target)?;
                    return Ok(vec![row]);
                }
            }
            // frozen-table lookup: own sample rows
            if let Some(lookup_plan) = &norm_plan.rstd {
                let lookup_first = norm_first.rstd.as_ref().ok_or(Error::ShapeMismatch)?;
                let commit = &lookup_plan.commit;
                if member.source_oracle == commit.bits_oracle {
                    let target = *lookup_first
                        .bits
                        .sample
                        .answers
                        .first()
                        .ok_or(Error::ShapeMismatch)?;
                    return Ok(vec![oracle_sample_row(
                        member.source_oracle,
                        lookup_first.bits.message.group.group,
                        &commit.bits_shape.oracle_shape(),
                        &lookup_first.bits.sample.seed,
                        0,
                        target,
                    )?]);
                }
                if member.source_oracle == commit.mask_oracle {
                    let target = *lookup_first
                        .mask
                        .sample
                        .answers
                        .first()
                        .ok_or(Error::ShapeMismatch)?;
                    let row =
                        lookup_mask_sample_row(commit, &lookup_first.mask, target)?;
                    return Ok(vec![row]);
                }
            }
        }
        if let (Some(closure_plan), Some(closure_first)) = (&plan.closure, &first.closure) {
            let positions = plan.send.positions() as u64;
            let slack_base = closure_plan.slack_oracle_base;
            let bits_base = closure_plan.bits_oracle_base;
            if member.source_oracle >= slack_base
                && member.source_oracle < slack_base.saturating_add(positions)
            {
                let position = (member.source_oracle - slack_base) as usize;
                let target = *closure_first
                    .slack_sample
                    .answers
                    .get(position)
                    .ok_or(Error::ShapeMismatch)?;
                return Ok(vec![oracle_sample_row(
                    member.source_oracle,
                    plan.send.group,
                    &closure_plan.slack.oracle_shape(),
                    &closure_first.slack_sample.seed,
                    0,
                    target,
                )?]);
            }
            if member.source_oracle >= bits_base
                && member.source_oracle < bits_base.saturating_add(positions)
            {
                let position = (member.source_oracle - bits_base) as usize;
                let target = *closure_first
                    .bits_sample
                    .answers
                    .get(position)
                    .ok_or(Error::ShapeMismatch)?;
                return Ok(vec![oracle_sample_row(
                    member.source_oracle,
                    plan.send.group,
                    &closure_plan.bits.oracle_shape(),
                    &closure_first.bits_sample.seed,
                    0,
                    target,
                )?]);
            }
            let map_base = closure_plan.map_oracle_base;
            if member.source_oracle >= map_base
                && member.source_oracle < map_base.saturating_add(positions)
            {
                let position = (member.source_oracle - map_base) as usize;
                let target = *closure_first
                    .map_sample
                    .answers
                    .get(position)
                    .ok_or(Error::ShapeMismatch)?;
                return Ok(vec![oracle_sample_row(
                    member.source_oracle,
                    plan.send.group,
                    &closure_plan.map.oracle_shape(),
                    &closure_first.map_sample.seed,
                    0,
                    target,
                )?]);
            }
            if member.source_oracle == closure_plan.mask.oracle {
                let target = *closure_first
                    .mask
                    .sample
                    .answers
                    .first()
                    .ok_or(Error::ShapeMismatch)?;
                return Ok(vec![closure_mask_sample_row(
                    &closure_plan.mask,
                    &closure_first.mask,
                    target,
                )?]);
            }
        }
        if let Some(index) = &plan.send.index_mask {
            if member.source_oracle == index.oracle {
                let message = first.index_mask.as_ref().ok_or(Error::ShapeMismatch)?;
                let target = *message
                    .sample
                    .answers
                    .first()
                    .ok_or(Error::ShapeMismatch)?;
                return Ok(vec![index_mask_sample_row(index, message, target)?]);
            }
        }
        if let Some(dot) = plan.send.dot.as_ref().filter(|_| !plan.send.activation_source.is_norm_hd()) {
            if member.source_oracle == dot.oracle {
                let message = first.dot.as_ref().ok_or(Error::ShapeMismatch)?;
                let target = *message
                    .sample
                    .answers
                    .first()
                    .ok_or(Error::ShapeMismatch)?;
                return Ok(vec![oracle_sample_row(
                    dot.oracle,
                    dot.group,
                    &dot.oracle_shape(),
                    &message.sample.seed,
                    0,
                    target,
                )?]);
            }
        }
        if member.source_oracle == plan.proximity.omega.oracle {
            let mut list = Vec::with_capacity(shared.omega_answers.len());
            for (answer, target) in shared.omega_answers.iter().enumerate() {
                list.push(oracle_sample_row(
                    plan.proximity.omega.oracle,
                    plan.proximity.omega.group,
                    &plan.proximity.omega.u,
                    &shared.omega_seeds,
                    answer as u32,
                    *target,
                )?);
            }
            return Ok(list);
        } else {
            let position = (member.source_oracle - plan.send.oracle_base) as usize;
            if position >= plan.send.positions() as usize {
                return Err(Error::StatementMismatch);
            }
            let index = position * plan.send.w.answers as usize;
            let target = *first
                .samples
                .answers
                .get(index)
                .ok_or(Error::ShapeMismatch)?;
            return Ok(vec![oracle_sample_row(
                member.source_oracle,
                plan.send.group,
                &shape,
                &first.samples.seed,
                0,
                target,
            )?]);
        }
        },
    )
}


/// The verifier-built output-closure rows: one comparison row, three
/// reconstruction rows, one bit end-value row, the `(U)` row and the virtual
/// Omega row per position. Every row names the same committed members the
/// existing Eval/tilde/Dot rows use for the reduced rows.
// ---------------------------------------------------------------------------
// Integrated norm relation
// ---------------------------------------------------------------------------

const NORM_ROOT_TAGS: [u64; 5] = [60, 61, 62, 63, 64];
const NORM_MASK_ROOT_TAG: u64 = 65;
const NORM_SEED_TAGS: [u64; 5] = [66, 67, 68, 69, 70];
const NORM_MASK_SEED_TAG: u64 = 71;
const NORM_ANSWER_TAGS: [u64; 5] = [72, 73, 74, 75, 76];
const NORM_MASK_ANSWER_TAG: u64 = 77;
const NORM_POSITION_TAGS: [u64; 5] = [78, 79, 80, 81, 82];
const NORM_MASK_POSITION_TAG: u64 = 83;
const NORM_ROUND_ROOT_TAG: u64 = 84;
const NORM_ROUND_SEED_TAG: u64 = 85;
const NORM_ROUND_ANSWER_TAG: u64 = 86;
const NORM_ROUND_POSITION_TAG: u64 = 87;
const NORM_LOOKUP_BITS_ROOT_TAG: u64 = 88;
const NORM_LOOKUP_MASK_ROOT_TAG: u64 = 89;
const NORM_LOOKUP_BITS_SEED_TAG: u64 = 90;
const NORM_LOOKUP_MASK_SEED_TAG: u64 = 91;
const NORM_LOOKUP_BITS_ANSWER_TAG: u64 = 92;
const NORM_LOOKUP_MASK_ANSWER_TAG: u64 = 93;
const NORM_LOOKUP_BITS_POSITION_TAG: u64 = 94;
const NORM_LOOKUP_MASK_POSITION_TAG: u64 = 95;
const NORM_LOOKUP_SEED_TAG: u64 = 96;
const NORM_RSTD_BITS_ROOT_TAG: u64 = 97;
const NORM_RSTD_MASK_ROOT_TAG: u64 = 98;
const NORM_RSTD_BITS_SEED_TAG: u64 = 99;
const NORM_RSTD_MASK_SEED_TAG: u64 = 100;
const NORM_RSTD_BITS_ANSWER_TAG: u64 = 101;
const NORM_RSTD_MASK_ANSWER_TAG: u64 = 102;
const NORM_RSTD_BITS_POSITION_TAG: u64 = 103;
const NORM_RSTD_MASK_POSITION_TAG: u64 = 104;
const NORM_RSTD_SEED_TAG: u64 = 105;
const NORM_PARAM_GAIN_POSITION_TAG: u64 = 106;
const NORM_PARAM_BIAS_POSITION_TAG: u64 = 107;
const NORM_PARAM_GAIN_ROOT_TAG: u64 = 108;
const NORM_PARAM_BIAS_ROOT_TAG: u64 = 109;
const NORM_PARAM_GAIN_ANSWER_TAG: u64 = 110;
const NORM_PARAM_BIAS_ANSWER_TAG: u64 = 111;
const NORM_OUTPUT_BROADCAST_ROOT_TAG: u64 = 112;
const NORM_OUTPUT_PRODUCT_ROOT_TAG: u64 = 113;
const NORM_OUTPUT_IMAGE_ROOT_TAG: u64 = 114;
const NORM_OUTPUT_OUT_ROOT_TAG: u64 = 115;
const NORM_OUTPUT_BROADCAST_SEED_TAG: u64 = 116;
const NORM_OUTPUT_PRODUCT_SEED_TAG: u64 = 117;
const NORM_OUTPUT_IMAGE_SEED_TAG: u64 = 118;
const NORM_OUTPUT_OUT_SEED_TAG: u64 = 119;
const NORM_OUTPUT_BROADCAST_ANSWER_TAG: u64 = 120;
const NORM_OUTPUT_PRODUCT_ANSWER_TAG: u64 = 121;
const NORM_OUTPUT_IMAGE_ANSWER_TAG: u64 = 122;
const NORM_OUTPUT_OUT_ANSWER_TAG: u64 = 123;
const NORM_OUTPUT_BROADCAST_POSITION_TAG: u64 = 124;
const NORM_OUTPUT_PRODUCT_POSITION_TAG: u64 = 125;
const NORM_OUTPUT_IMAGE_POSITION_TAG: u64 = 126;
const NORM_OUTPUT_OUT_POSITION_TAG: u64 = 127;
const NORM_OUTPUT_TAIL_ROOT_TAG: u64 = 128;
const NORM_OUTPUT_TAIL_SEED_TAG: u64 = 129;
const NORM_OUTPUT_TAIL_ANSWER_TAG: u64 = 130;
const NORM_OUTPUT_TAIL_POSITION_TAG: u64 = 131;

const NORM_OUTPUT_ROOT_TAGS: [u64; 4] = [
    NORM_OUTPUT_BROADCAST_ROOT_TAG,
    NORM_OUTPUT_PRODUCT_ROOT_TAG,
    NORM_OUTPUT_IMAGE_ROOT_TAG,
    NORM_OUTPUT_OUT_ROOT_TAG,
];
const NORM_OUTPUT_SEED_TAGS: [u64; 4] = [
    NORM_OUTPUT_BROADCAST_SEED_TAG,
    NORM_OUTPUT_PRODUCT_SEED_TAG,
    NORM_OUTPUT_IMAGE_SEED_TAG,
    NORM_OUTPUT_OUT_SEED_TAG,
];
const NORM_OUTPUT_ANSWER_TAGS: [u64; 4] = [
    NORM_OUTPUT_BROADCAST_ANSWER_TAG,
    NORM_OUTPUT_PRODUCT_ANSWER_TAG,
    NORM_OUTPUT_IMAGE_ANSWER_TAG,
    NORM_OUTPUT_OUT_ANSWER_TAG,
];
const NORM_OUTPUT_POSITION_TAGS: [u64; 4] = [
    NORM_OUTPUT_BROADCAST_POSITION_TAG,
    NORM_OUTPUT_PRODUCT_POSITION_TAG,
    NORM_OUTPUT_IMAGE_POSITION_TAG,
    NORM_OUTPUT_OUT_POSITION_TAG,
];

fn norm_group_plans(plan: &NormPlan) -> Vec<&NormGroupPlan> {
    let mut groups = vec![
        &plan.x,
        &plan.sum,
        &plan.squares,
        &plan.mean,
        &plan.variance,
    ];
    if let Some(round) = &plan.round {
        groups.push(&round.group);
    }
    groups
}

fn norm_group_witnesses(witness: &NormWitness) -> Vec<&NormGroupWitness> {
    let mut groups = vec![
        &witness.x,
        &witness.sum,
        &witness.squares,
        &witness.mean,
        &witness.variance,
    ];
    if let Some(round) = &witness.round {
        groups.push(round);
    }
    groups
}

fn absorb_norm_mask_root(
    chain: &mut Chain,
    plan: &NormMaskPlan,
    root: &Digest,
) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(NORM_MASK_ROOT_TAG);
    message.push_u64(plan.oracle);
    message.push_u64(plan.group);
    message.push_u64(plan.mask_rows);
    message.push_u64(plan.answers as u64);
    message.push_u64(plan.mask_code.log_n as u64);
    message.push_u64(plan.mask_code.ell);
    message.push_u64(plan.mask_code.r);
    push_tree_plan(&mut message, &plan.tree);
    push_digest(&mut message, root);
    let prefix = message.prefix(KIND_SEED, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn draw_norm_mask_positions(chain: &mut Chain, plan: &NormMaskPlan) -> Result<Vec<u64>> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(NORM_MASK_POSITION_TAG);
    message.push_u64(1);
    message.push_u64(plan.openings as u64);
    let prefix = message.prefix(KIND_POSITIONS, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    let mut positions = Vec::with_capacity(plan.openings);
    for _ in 0..plan.openings {
        positions.push(chain.draw_position(plan.mask_code.log_n)?);
    }
    Ok(unique_sorted(&positions))
}

/// The transcript tags of one norm lookup section.
#[derive(Clone, Copy)]
struct NormLookupTags {
    bits_root: u64,
    mask_root: u64,
    bits_seed: u64,
    mask_seed: u64,
    bits_answer: u64,
    mask_answer: u64,
    bits_position: u64,
    mask_position: u64,
}

const NORM_LOOKUP_TAGS: NormLookupTags = NormLookupTags {
    bits_root: NORM_LOOKUP_BITS_ROOT_TAG,
    mask_root: NORM_LOOKUP_MASK_ROOT_TAG,
    bits_seed: NORM_LOOKUP_BITS_SEED_TAG,
    mask_seed: NORM_LOOKUP_MASK_SEED_TAG,
    bits_answer: NORM_LOOKUP_BITS_ANSWER_TAG,
    mask_answer: NORM_LOOKUP_MASK_ANSWER_TAG,
    bits_position: NORM_LOOKUP_BITS_POSITION_TAG,
    mask_position: NORM_LOOKUP_MASK_POSITION_TAG,
};

/// Prover side of one norm lookup first-stage section: commit the multiplicity
/// bit and mask messages, absorb their roots and answers, draw the positions
/// and open them. The tag bundle names the section in the transcript.
fn prove_norm_lookup_first(
    chain: &mut Chain,
    lookup_plan: &crate::proof::hidden_norm_psc::NormLookupPlan,
    lookup_witness: &crate::proof::hidden_norm_psc::NormLookupWitness,
    commit_seed: u64,
    tags: &NormLookupTags,
) -> Result<(
    NormLookupMessage,
    crate::proof::lookup_messages::LookupMessages,
)> {
    let mask_entries = crate::proof::lookup::mask_count(&lookup_plan.lookup)?;
    let mut source =
        crate::proof::lookup_messages::LookupRandomSource::new(commit_seed);
    let messages = crate::proof::lookup_messages::commit_lookup(
        &lookup_plan.commit,
        &lookup_witness.bits,
        mask_entries,
        &mut source,
    )?;
    absorb_closure_group(chain, tags.bits_root, &messages.bits.group)?;
    absorb_lookup_mask_root(chain, lookup_plan, &messages.mask_root, tags.mask_root)?;
    let bits_shape = &lookup_plan.commit.bits_shape;
    let bits_seed = draw_closure_seed_set(
        chain,
        tags.bits_seed,
        bits_shape.sample_bits()?,
        bits_shape.answers,
    )?;
    let bits_answers = value_group_answers(
        bits_shape,
        lookup_plan.commit.bits_oracle,
        lookup_plan.commit.group,
        &messages.bits.messages,
        &bits_seed,
    )?;
    absorb_tagged_answers(chain, tags.bits_answer, &bits_answers)?;
    let mask_seed = draw_closure_seed_set(
        chain,
        tags.mask_seed,
        lookup_mask_sample_bits(&lookup_plan.commit)?,
        1,
    )?;
    let mask_answer = lookup_mask_answer(
        &lookup_plan.commit,
        &messages.mask_rows,
        &mask_seed[0],
    )?;
    absorb_tagged_answers(chain, tags.mask_answer, &[mask_answer])?;
    let bits_positions = draw_closure_group_positions(
        chain,
        tags.bits_position,
        1,
        lookup_plan.commit.bits_openings,
        bits_shape.code.log_n,
    )?;
    let bits_tree = messages
        .bits
        .trees
        .get(0)
        .ok_or(Error::ShapeMismatch)?;
    if bits_tree.plan() != &bits_shape.tree_plan(0)? {
        return Err(Error::StatementMismatch);
    }
    let bits_opening = bits_tree.open(&bits_positions[0])?;
    let mask_positions = draw_lookup_mask_positions(chain, lookup_plan, tags.mask_position)?;
    let mask_tree = messages.mask_tree(&lookup_plan.commit)?;
    if *mask_tree.root() != messages.mask_root {
        return Err(Error::StatementMismatch);
    }
    let mask_opening = mask_tree.open(&mask_positions)?;
    let message = NormLookupMessage {
        bits: CommittedPscGroup {
            message: CommittedGroupMessage::new(messages.bits.group.clone()),
            sample: SendSamples {
                seed: bits_seed,
                answers: bits_answers,
            },
            positions: bits_positions,
            openings: vec![bits_opening],
        },
        mask: NormMaskMessage {
            tree: lookup_plan.commit.mask_tree.clone(),
            root: messages.mask_root,
            sample: SendSamples {
                seed: mask_seed,
                answers: vec![mask_answer],
            },
            positions: mask_positions,
            opening: mask_opening,
        },
    };
    Ok((message, messages))
}

/// Verifier side of one norm lookup first-stage section.
fn verify_norm_lookup_first(
    chain: &mut Chain,
    lookup_plan: &crate::proof::hidden_norm_psc::NormLookupPlan,
    lookup: &NormLookupMessage,
    tags: &NormLookupTags,
) -> Result<()> {
    absorb_closure_group(chain, tags.bits_root, &lookup.bits.message.group)?;
    absorb_lookup_mask_root(chain, lookup_plan, &lookup.mask.root, tags.mask_root)?;
    let bits_shape = &lookup_plan.commit.bits_shape;
    let bits_seed = draw_closure_seed_set(
        chain,
        tags.bits_seed,
        bits_shape.sample_bits()?,
        bits_shape.answers,
    )?;
    if bits_seed != lookup.bits.sample.seed {
        return Err(Error::InvalidProof);
    }
    absorb_tagged_answers(chain, tags.bits_answer, &lookup.bits.sample.answers)?;
    let mask_seed = draw_closure_seed_set(
        chain,
        tags.mask_seed,
        lookup_mask_sample_bits(&lookup_plan.commit)?,
        1,
    )?;
    if mask_seed != lookup.mask.sample.seed {
        return Err(Error::InvalidProof);
    }
    absorb_tagged_answers(chain, tags.mask_answer, &lookup.mask.sample.answers)?;
    let bits_positions = draw_closure_group_positions(
        chain,
        tags.bits_position,
        1,
        lookup_plan.commit.bits_openings,
        bits_shape.code.log_n,
    )?;
    if bits_positions != lookup.bits.positions {
        return Err(Error::InvalidProof);
    }
    let bits_entry = lookup
        .bits
        .message
        .group
        .entries
        .first()
        .ok_or(Error::ShapeMismatch)?;
    verify_runtime_opening(
        &bits_entry.tree,
        &bits_entry.root,
        &bits_positions[0],
        &lookup.bits.openings[0],
    )?;
    let mask_positions = draw_lookup_mask_positions(chain, lookup_plan, tags.mask_position)?;
    if mask_positions != lookup.mask.positions {
        return Err(Error::InvalidProof);
    }
    verify_runtime_opening(
        &lookup_plan.commit.mask_tree,
        &lookup.mask.root,
        &mask_positions,
        &lookup.mask.opening,
    )?;
    Ok(())
}

/// Derives the private lookup commitment seed from the run seed.
fn lookup_commit_seed(seed: &Digest) -> Result<u64> {
    let digest = derive_run_seed(seed, 10, 0)?;
    Ok(u64::from_le_bytes(
        digest.0[..8]
            .try_into()
            .map_err(|_| Error::InvalidConfig)?,
    ))
}

fn rstd_commit_seed(seed: &Digest) -> Result<u64> {
    let digest = derive_run_seed(seed, 11, 0)?;
    Ok(u64::from_le_bytes(
        digest.0[..8]
            .try_into()
            .map_err(|_| Error::InvalidConfig)?,
    ))
}

/// Absorbs the lookup mask root before its sample seed is drawn.
fn absorb_lookup_mask_root(
    chain: &mut Chain,
    plan: &crate::proof::hidden_norm_psc::NormLookupPlan,
    root: &Digest,
    tag: u64,
) -> Result<()> {
    let commit = &plan.commit;
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(tag);
    message.push_u64(commit.mask_oracle);
    message.push_u64(commit.group);
    message.push_u64(commit.mask_rows);
    message.push_u64(1);
    message.push_u64(commit.mask_code.log_n as u64);
    message.push_u64(commit.mask_code.ell);
    message.push_u64(commit.mask_code.r);
    push_tree_plan(&mut message, &commit.mask_tree);
    push_digest(&mut message, root);
    let prefix = message.prefix(KIND_SEED, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// The number of seed coordinates the lookup mask sample form reads.
fn absorb_rstd_mask_root(
    chain: &mut Chain,
    plan: &crate::proof::hidden_norm_psc::NormRstdPlan,
    root: &Digest,
) -> Result<()> {
    let commit = &plan.commit;
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(NORM_RSTD_MASK_ROOT_TAG);
    message.push_u64(commit.mask_oracle);
    message.push_u64(commit.group);
    message.push_u64(commit.mask_rows);
    message.push_u64(1);
    message.push_u64(commit.mask_code.log_n as u64);
    message.push_u64(commit.mask_code.ell);
    message.push_u64(commit.mask_code.r);
    push_tree_plan(&mut message, &commit.mask_tree);
    push_digest(&mut message, root);
    let prefix = message.prefix(KIND_SEED, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn draw_rstd_mask_positions(
    chain: &mut Chain,
    plan: &crate::proof::hidden_norm_psc::NormRstdPlan,
) -> Result<Vec<u64>> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(NORM_RSTD_MASK_POSITION_TAG);
    message.push_u64(1);
    message.push_u64(plan.commit.mask_openings as u64);
    let prefix = message.prefix(KIND_POSITIONS, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    let mut positions = Vec::with_capacity(plan.commit.mask_openings);
    for _ in 0..plan.commit.mask_openings {
        positions.push(chain.draw_position(plan.commit.mask_code.log_n)?);
    }
    Ok(unique_sorted(&positions))
}

fn lookup_mask_sample_bits(
    commit: &crate::proof::lookup_messages::LookupCommitPlan,
) -> Result<u32> {
    ceil_log2_u64(commit.mask_rows)
        .checked_add(commit.mask_code.log_ell())
        .ok_or(Error::ResourceLimit)
}

/// The sample answer of the lookup mask message: the `eq` form over the mask
/// cube plus the reserved pad under the seed's `beta`.
fn lookup_mask_answer(
    commit: &crate::proof::lookup_messages::LookupCommitPlan,
    rows: &[Vec<Fp3>],
    seed: &SampleSeed,
) -> Result<Fp3> {
    if commit.mask_oracle == 0 || commit.group == 0 {
        return Err(Error::InvalidConfig);
    }
    let bits = lookup_mask_sample_bits(commit)?;
    if seed.z.len() < bits as usize || seed.beta == Fp3::ZERO {
        return Err(Error::ShapeMismatch);
    }
    let ell = usize::try_from(commit.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
    let mask_rows = usize::try_from(commit.mask_rows).map_err(|_| Error::ResourceLimit)?;
    if rows.len() != mask_rows || rows.iter().any(|row| row.len() != ell) {
        return Err(Error::ShapeMismatch);
    }
    let row_bits = ceil_log2_u64(commit.mask_rows);
    let point = &seed.z[..bits as usize];
    let count = 1u64.checked_shl(bits).ok_or(Error::ResourceLimit)?;
    let mut answer = Fp3::ZERO;
    for index in 0..count {
        let row = (index & ((1u64 << row_bits) - 1)) as usize;
        let place = (index >> row_bits) as usize;
        if row >= mask_rows || place >= ell {
            continue;
        }
        let coefficient = if row + 1 == mask_rows && place + 1 == ell {
            seed.beta
        } else {
            eq_index(point, index, bits)?
        };
        answer = answer.add(&coefficient.mul(&rows[row][place]));
    }
    Ok(answer)
}

/// Draws the lookup mask opening positions.
fn draw_lookup_mask_positions(
    chain: &mut Chain,
    plan: &crate::proof::hidden_norm_psc::NormLookupPlan,
    tag: u64,
) -> Result<Vec<u64>> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(tag);
    message.push_u64(1);
    message.push_u64(plan.commit.mask_openings as u64);
    let prefix = message.prefix(KIND_POSITIONS, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    let mut positions = Vec::with_capacity(plan.commit.mask_openings);
    for _ in 0..plan.commit.mask_openings {
        positions.push(chain.draw_position(plan.commit.mask_code.log_n)?);
    }
    Ok(unique_sorted(&positions))
}

/// Draws the lookup sub-transcript seed after the first message is fixed.
fn draw_lookup_seed(chain: &mut Chain, tag: u64) -> Result<u64> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(tag);
    let prefix = message.prefix(KIND_SEED, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(chain.draw_f()?.as_canonical_u64())
}

fn norm_mask_answer(
    plan: &NormMaskPlan,
    rows: &[Vec<Fp3>],
    seed: &SampleSeed,
) -> Result<Fp3> {
    if plan.oracle == 0 || plan.group == 0 || plan.answers != 1 {
        return Err(Error::InvalidConfig);
    }
    let bits = plan.sample_bits()?;
    if seed.z.len() < bits as usize || seed.beta == Fp3::ZERO {
        return Err(Error::ShapeMismatch);
    }
    let ell = usize::try_from(plan.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
    let mask_rows = usize::try_from(plan.mask_rows).map_err(|_| Error::ResourceLimit)?;
    if rows.len() != mask_rows || rows.iter().any(|row| row.len() != ell) {
        return Err(Error::ShapeMismatch);
    }
    let row_bits = ceil_log2_u64(plan.mask_rows);
    let point = &seed.z[..bits as usize];
    let count = 1u64.checked_shl(bits).ok_or(Error::ResourceLimit)?;
    let mut answer = Fp3::ZERO;
    for index in 0..count {
        let row = (index & ((1u64 << row_bits) - 1)) as usize;
        let place = (index >> row_bits) as usize;
        if row >= mask_rows || place >= ell {
            continue;
        }
        let coefficient = if row + 1 == mask_rows && place + 1 == ell {
            seed.beta
        } else {
            eq_index(point, index, bits)?
        };
        answer = answer.add(&coefficient.mul(&rows[row][place]));
    }
    Ok(answer)
}

fn mask_sample_row(
    oracle: u64,
    group: u64,
    mask_code: zkie_engine::pcs::request::code::CodeParams,
    mask_rows: u64,
    answers: u32,
    bits: u32,
    message: &NormMaskMessage,
    target: Fp3,
) -> Result<GuRelationRow> {
    if message.sample.answers.len() != answers as usize
        || message.sample.seed.is_empty()
    {
        return Err(Error::ShapeMismatch);
    }
    let seed = &message.sample.seed[0];
    if seed.z.len() < bits as usize || seed.beta == Fp3::ZERO {
        return Err(Error::ShapeMismatch);
    }
    let ell = usize::try_from(mask_code.ell).map_err(|_| Error::ResourceLimit)?;
    let mask_rows = usize::try_from(mask_rows).map_err(|_| Error::ResourceLimit)?;
    let mut mask = vec![Fp3::ZERO; mask_rows * ell];
    let row_bits = ceil_log2_u64(mask_rows as u64);
    let stride = 1usize.checked_shl(row_bits).ok_or(Error::ResourceLimit)?;
    let point = &seed.z[..bits as usize];
    let count = 1u64.checked_shl(bits).ok_or(Error::ResourceLimit)?;
    for index in 0..count {
        let row = (index & (stride as u64 - 1)) as usize;
        let place = (index >> row_bits) as usize;
        if row >= mask_rows || place >= ell {
            continue;
        }
        let coefficient = if row + 1 == mask_rows && place + 1 == ell {
            seed.beta
        } else {
            eq_index(point, index, bits)?
        };
        mask[row * ell + place] = coefficient;
    }
    Ok(GuRelationRow {
        oracle,
        group,
        kind: GuRelationKind::Sample,
        main: Vec::new(),
        mask,
        target,
    })
}

fn norm_mask_sample_row(
    plan: &NormMaskPlan,
    message: &NormMaskMessage,
    target: Fp3,
) -> Result<GuRelationRow> {
    mask_sample_row(
        plan.oracle,
        plan.group,
        plan.mask_code,
        plan.mask_rows,
        plan.answers,
        plan.sample_bits()?,
        message,
        target,
    )
}

fn lookup_mask_sample_row(
    commit: &crate::proof::lookup_messages::LookupCommitPlan,
    message: &NormMaskMessage,
    target: Fp3,
) -> Result<GuRelationRow> {
    mask_sample_row(
        commit.mask_oracle,
        commit.group,
        commit.mask_code,
        commit.mask_rows,
        1,
        lookup_mask_sample_bits(commit)?,
        message,
        target,
    )
}

/// The four output tail groups of the first-stage commitment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NormOutputGroup {
    Broadcast,
    Product,
    Image,
    Out,
}

/// The committed column values of one output tail group, one row per
/// position. The signed integers are encoded exactly like the other norm
/// group values.
fn norm_output_values(
    rows: &[crate::proof::hidden_norm::NormOutputRow],
    columns: u64,
    group: NormOutputGroup,
) -> Result<Vec<Vec<Fp3>>> {
    use crate::proof::output_closure::fp3_signed;
    let width = usize::try_from(columns).map_err(|_| Error::ResourceLimit)?;
    let mut values = Vec::with_capacity(rows.len());
    for row in rows {
        let signed: Vec<i128> = match group {
            NormOutputGroup::Broadcast => row.broadcast.iter().map(|value| *value as i128).collect(),
            NormOutputGroup::Product => row.product.clone(),
            NormOutputGroup::Image => row.image.clone(),
            NormOutputGroup::Out => row.out.iter().map(|value| *value as i128).collect(),
        };
        if signed.len() != width {
            return Err(Error::ShapeMismatch);
        }
        let mut message = Vec::with_capacity(width);
        for value in signed {
            message.push(fp3_signed(value)?);
        }
        values.push(message);
    }
    Ok(values)
}

fn prove_norm_first(
    plan: &HiddenLinPlan,
    chain: &mut Chain,
    witness: &NormWitness,
    seed: &Digest,
    param_tables: Option<&mut dyn NormParamTableProver>,
    ) -> Result<(
        NormFirstMessages,
        Option<crate::proof::lookup_messages::LookupMessages>,
        Option<crate::proof::lookup_messages::LookupMessages>,
        Option<(
            crate::proof::hidden_norm_psc::NormParamClaimWitness,
            Option<crate::proof::hidden_norm_psc::NormParamClaimWitness>,
        )>,
    )> {
    let norm = plan.norm.as_ref().ok_or(Error::InvalidConfig)?;
    let positions = norm.shape.positions;
    let groups = norm_group_plans(norm);
    let group_witnesses = norm_group_witnesses(witness);
    if witness.rows.len() != positions as usize
        || witness.core.len() != positions as usize
        || witness.x.committed.roots != norm.x_roots
    {
        return Err(Error::StatementMismatch);
    }
    for (group_plan, group_witness) in groups.iter().zip(group_witnesses.iter()) {
        if group_witness.committed.group.group != group_plan.group
            || group_witness.committed.group.fault(positions).is_some()
            || group_witness.committed.roots.len() != positions as usize
            || group_witness.messages.len() != positions as usize
            || group_witness.tails.len() != positions as usize
            || group_witness.salts.len() != positions as usize
            || group_witness.pads.len() != positions as usize
        {
            return Err(Error::ShapeMismatch);
        }
        for (position, entry) in group_witness
            .committed
            .group
            .entries
            .iter()
            .enumerate()
        {
            if entry.position != position as u32
                || entry.oracle
                    != group_plan
                        .oracle_base
                        .checked_add(position as u64)
                        .ok_or(Error::ResourceLimit)?
                || entry.shape != group_plan.shape.oracle_shape()
                || entry.tree != group_plan.shape.tree_plan(position as u32)?
                || entry.root == Digest::ZERO
            {
                return Err(Error::StatementMismatch);
            }
        }
    }
    let mask_witness = match (&norm.mask, &witness.mask) {
        (Some(mask_plan), Some(mask_witness)) => {
            let ell = usize::try_from(mask_plan.mask_code.ell)
                .map_err(|_| Error::ResourceLimit)?;
            let tail = usize::try_from(mask_plan.mask_code.r)
                .map_err(|_| Error::ResourceLimit)?;
            let rows = usize::try_from(mask_plan.mask_rows)
                .map_err(|_| Error::ResourceLimit)?;
            let columns = usize::try_from(mask_plan.tree.columns())
                .map_err(|_| Error::ResourceLimit)?;
            if mask_witness.committed.root == Digest::ZERO
                || mask_witness.committed.rows.len() != rows
                || mask_witness.committed
                    .rows
                    .iter()
                    .any(|row| row.len() != ell)
                || mask_witness.committed.tails.len() != rows
                || mask_witness
                    .committed
                    .tails
                    .iter()
                    .any(|row| row.len() != tail)
                || mask_witness.committed.salts.len() != columns
                || mask_witness.committed.salts.iter().any(|salt| {
                    salt.len() != mask_plan.tree.salt_bytes
                })
            {
                return Err(Error::ShapeMismatch);
            }
            Some(mask_witness)
        }
        (None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    for (tag, group_witness) in NORM_ROOT_TAGS.iter().zip(group_witnesses.iter()) {
        absorb_closure_group(chain, *tag, &group_witness.committed.group)?;
    }
    let round_group = match (&norm.round, &witness.round) {
        (Some(round_plan), Some(round_witness)) => {
            if round_witness.committed.group.group != round_plan.group.group
                || round_witness
                    .committed
                    .group
                    .fault(positions)
                    .is_some()
                || round_witness.committed.roots.len() != positions as usize
                || round_witness.messages.len() != positions as usize
                || round_witness.tails.len() != positions as usize
                || round_witness.salts.len() != positions as usize
                || round_witness.pads.len() != positions as usize
            {
                return Err(Error::ShapeMismatch);
            }
            for (position, entry) in round_witness.committed.group.entries.iter().enumerate() {
                if entry.position != position as u32
                    || entry.oracle
                        != round_plan
                            .group
                            .oracle_base
                            .checked_add(position as u64)
                            .ok_or(Error::ResourceLimit)?
                    || entry.shape != round_plan.group.shape.oracle_shape()
                    || entry.tree != round_plan.group.shape.tree_plan(position as u32)?
                    || entry.root == Digest::ZERO
                {
                    return Err(Error::StatementMismatch);
                }
            }
            let rows = usize::try_from(round_plan.group.shape.main_rows)
                .map_err(|_| Error::ResourceLimit)?;
            let ell = usize::try_from(round_plan.group.shape.code.ell)
                .map_err(|_| Error::ResourceLimit)?;
            let columns = usize::try_from(round_plan.group.columns)
                .map_err(|_| Error::ResourceLimit)?;
            for (position, message) in round_witness.messages.iter().enumerate() {
                if message.len() != rows || message.iter().any(|row| row.len() != ell) {
                    return Err(Error::ShapeMismatch);
                }
                let expected = match witness
                    .rsqrt
                    .as_ref()
                    .and_then(|rows| rows.get(position))
                {
                    Some(rsqrt) => round_plan.layout.values_with_rsqrt(
                        &witness.core[position],
                        rsqrt,
                        round_plan.group.columns,
                    )?,
                    None => round_plan
                        .layout
                        .values(&witness.core[position], round_plan.group.columns)?,
                };
                for slot in 0..columns {
                    let row = slot % rows;
                    let place = slot / rows;
                    if place >= ell
                        || message
                            .get(row)
                            .and_then(|row| row.get(place))
                            .copied()
                            .ok_or(Error::ShapeMismatch)?
                            != expected[slot]
                    {
                        return Err(Error::StatementMismatch);
                    }
                }
            }
            absorb_closure_group(chain, NORM_ROUND_ROOT_TAG, &round_witness.committed.group)?;
            Some(round_witness)
        }
        (None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    if let (Some(mask_plan), Some(mask_witness)) = (&norm.mask, mask_witness) {
        absorb_norm_mask_root(chain, mask_plan, &mask_witness.committed.root)?;
    }
    let mut seeds = Vec::with_capacity(groups.len());
    for (tag, group_plan) in NORM_SEED_TAGS.iter().zip(groups.iter()) {
        seeds.push(draw_closure_seed_set(
            chain,
            *tag,
            group_plan.shape.sample_bits()?,
            group_plan.shape.answers,
        )?);
    }
    let round_seed = match (&norm.round, round_group) {
        (Some(round_plan), Some(_)) => Some(draw_closure_seed_set(
            chain,
            NORM_ROUND_SEED_TAG,
            round_plan.group.shape.sample_bits()?,
            round_plan.group.shape.answers,
        )?),
        (None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    let mask_seeds = match norm.mask.as_ref() {
        Some(mask_plan) => Some(draw_closure_seed_set(
            chain,
            NORM_MASK_SEED_TAG,
            mask_plan.sample_bits()?,
            mask_plan.answers,
        )?),
        None => None,
    };
    let mut answers = Vec::with_capacity(groups.len());
    for (index, ((group_plan, group_witness), seed_set)) in groups
        .iter()
        .zip(group_witnesses.iter())
        .zip(seeds.iter())
        .enumerate()
    {
        let values = value_group_answers(
            &group_plan.shape,
            group_plan.oracle_base,
            group_plan.group,
            &group_witness.committed.messages,
            seed_set,
        )?;
        absorb_tagged_answers(chain, NORM_ANSWER_TAGS[index], &values)?;
        answers.push(values);
    }
    let round_answers = match (&norm.round, round_group, round_seed.as_ref()) {
        (Some(round_plan), Some(round_witness), Some(seed_set)) => {
            let values = value_group_answers(
                &round_plan.group.shape,
                round_plan.group.oracle_base,
                round_plan.group.group,
                &round_witness.committed.messages,
                seed_set,
            )?;
            absorb_tagged_answers(chain, NORM_ROUND_ANSWER_TAG, &values)?;
            Some(values)
        }
        (None, None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    let mask_sample = match (
        norm.mask.as_ref(),
        mask_witness,
        mask_seeds,
    ) {
        (Some(mask_plan), Some(mask_witness), Some(seed_set)) => {
            let answer =
                norm_mask_answer(mask_plan, &mask_witness.committed.rows, &seed_set[0])?;
            absorb_tagged_answers(chain, NORM_MASK_ANSWER_TAG, &[answer])?;
            Some((seed_set, answer))
        }
        (None, None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    let mut group_messages = Vec::with_capacity(groups.len());
    for (index, ((group_plan, group_witness), seed_set)) in groups
        .iter()
        .zip(group_witnesses.iter())
        .zip(seeds.iter())
        .enumerate()
    {
        let positions = draw_closure_group_positions(
            chain,
            NORM_POSITION_TAGS[index],
            positions,
            group_plan.openings,
            group_plan.shape.code.log_n,
        )?;
        let openings = open_value_group_openings(
            &group_plan.shape,
            &group_witness.committed.messages,
            &group_witness.tails,
            &group_witness.salts,
            &positions,
            &group_witness.committed.trees,
        )?;
        group_messages.push(CommittedPscGroup {
            message: CommittedGroupMessage::new(group_witness.committed.group.clone()),
            sample: SendSamples {
                seed: seed_set.clone(),
                answers: answers[index].clone(),
            },
            positions,
            openings,
        });
    }
    let round_message = match (
        &norm.round,
        round_group,
        round_seed.as_ref(),
        round_answers.as_ref(),
    ) {
        (Some(round_plan), Some(round_witness), Some(seed_set), Some(answers)) => {
            let round_positions = draw_closure_group_positions(
                chain,
                NORM_ROUND_POSITION_TAG,
                positions,
                round_plan.group.openings,
                round_plan.group.shape.code.log_n,
            )?;
            let openings = open_value_group_openings(
                &round_plan.group.shape,
                &round_witness.committed.messages,
                &round_witness.tails,
                &round_witness.salts,
                &round_positions,
                &round_witness.committed.trees,
            )?;
            Some(CommittedPscGroup {
                message: CommittedGroupMessage::new(round_witness.committed.group.clone()),
                sample: SendSamples {
                    seed: seed_set.clone(),
                    answers: answers.clone(),
                },
                positions: round_positions,
                openings,
            })
        }
        (None, None, None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    let (mask_positions, mask_opening) = match (norm.mask.as_ref(), mask_witness) {
        (Some(mask_plan), Some(mask_witness)) => {
            let positions = draw_norm_mask_positions(chain, mask_plan)?;
            let opening = mask_witness.committed.tree(mask_plan)?.open(&positions)?;
            (positions, opening)
        }
        (None, None) => (Vec::new(), zkie_engine::pcs::request::runtime_tree::RuntimeTreeOpening {
            positions: Vec::new(),
            columns: Vec::new(),
            given: Vec::new(),
        }),
        _ => return Err(Error::ShapeMismatch),
    };
    let mask = match (norm.mask.as_ref(), mask_sample) {
        (Some(mask_plan), Some((seed_set, answer))) => Some(NormMaskMessage {
            tree: mask_plan.tree.clone(),
            root: mask_witness
                .ok_or(Error::ShapeMismatch)?
                .committed
                .root,
            sample: SendSamples {
                seed: seed_set,
                answers: vec![answer],
            },
            positions: mask_positions,
            opening: mask_opening,
        }),
        (None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    let (lookup, lookup_messages) = match &norm.lookup {
        Some(lookup_plan) => {
            let lookup_witness = witness.lookup.as_ref().ok_or(Error::ShapeMismatch)?;
            let (message, messages) = prove_norm_lookup_first(
                chain,
                lookup_plan,
                lookup_witness,
                lookup_commit_seed(seed)?,
                &NORM_LOOKUP_TAGS,
            )?;
            (Some(message), Some(messages))
        }
        None => (None, None),
    };
    let (rstd, rstd_messages) = match &norm.rstd {
        Some(rstd_plan) => {
            let rstd_witness = witness.rstd.as_ref().ok_or(Error::ShapeMismatch)?;
            let mask_entries = crate::proof::lookup::mask_count(&rstd_plan.lookup)?;
            let mut source = crate::proof::lookup_messages::LookupRandomSource::new(
                rstd_commit_seed(seed)?,
            );
            let messages = crate::proof::lookup_messages::commit_lookup(
                &rstd_plan.commit,
                &rstd_witness.bits,
                mask_entries,
                &mut source,
            )?;
            absorb_closure_group(chain, NORM_RSTD_BITS_ROOT_TAG, &messages.bits.group)?;
            absorb_rstd_mask_root(chain, rstd_plan, &messages.mask_root)?;
            let bits_shape = &rstd_plan.commit.bits_shape;
            let bits_seed = draw_closure_seed_set(
                chain,
                NORM_RSTD_BITS_SEED_TAG,
                bits_shape.sample_bits()?,
                bits_shape.answers,
            )?;
            let bits_answers = value_group_answers(
                bits_shape,
                rstd_plan.commit.bits_oracle,
                rstd_plan.commit.group,
                &messages.bits.messages,
                &bits_seed,
            )?;
            absorb_tagged_answers(chain, NORM_RSTD_BITS_ANSWER_TAG, &bits_answers)?;
            let mask_seed = draw_closure_seed_set(
                chain,
                NORM_RSTD_MASK_SEED_TAG,
                lookup_mask_sample_bits(&rstd_plan.commit)?,
                1,
            )?;
            let mask_answer = lookup_mask_answer(
                &rstd_plan.commit,
                &messages.mask_rows,
                &mask_seed[0],
            )?;
            absorb_tagged_answers(chain, NORM_RSTD_MASK_ANSWER_TAG, &[mask_answer])?;
            let bits_positions = draw_closure_group_positions(
                chain,
                NORM_RSTD_BITS_POSITION_TAG,
                1,
                rstd_plan.commit.bits_openings,
                bits_shape.code.log_n,
            )?;
            let bits_tree = messages
                .bits
                .trees
                .get(0)
                .ok_or(Error::ShapeMismatch)?;
            if bits_tree.plan() != &bits_shape.tree_plan(0)? {
                return Err(Error::StatementMismatch);
            }
            let bits_opening = bits_tree.open(&bits_positions[0])?;
            let mask_positions = draw_rstd_mask_positions(chain, rstd_plan)?;
            let mask_tree = messages.mask_tree(&rstd_plan.commit)?;
            if *mask_tree.root() != messages.mask_root {
                return Err(Error::StatementMismatch);
            }
            let mask_opening = mask_tree.open(&mask_positions)?;
            let message = NormLookupMessage {
                bits: CommittedPscGroup {
                    message: CommittedGroupMessage::new(messages.bits.group.clone()),
                    sample: SendSamples {
                        seed: bits_seed,
                        answers: bits_answers,
                    },
                    positions: bits_positions,
                    openings: vec![bits_opening],
                },
                mask: NormMaskMessage {
                    tree: rstd_plan.commit.mask_tree.clone(),
                    root: messages.mask_root,
                    sample: SendSamples {
                        seed: mask_seed,
                        answers: vec![mask_answer],
                    },
                    positions: mask_positions,
                    opening: mask_opening,
                },
            };
            (Some(message), Some(messages))
        }
        None => (None, None),
    };
    let mut param_claims = None;
    let params = match (&norm.params, param_tables) {
        (Some(params_plan), Some(tables)) => {
            let width = usize::try_from(norm.columns).map_err(|_| Error::ResourceLimit)?;
            let values = witness.params.as_ref().ok_or(Error::ShapeMismatch)?;
            if values.gain.len() != width || values.bias.len() != width {
                return Err(Error::ShapeMismatch);
            }
            let gain_claim = params_plan
                .gain_claim
                .as_ref()
                .ok_or(Error::ShapeMismatch)?;
            let (gain, gain_material) = prove_norm_param_claim(
                chain,
                &params_plan.gain,
                gain_claim,
                NormParamKind::Gain,
                &mut *tables,
                seed,
                NORM_PARAM_GAIN_ROOT_TAG,
                NORM_PARAM_GAIN_ANSWER_TAG,
                NORM_PARAM_GAIN_POSITION_TAG,
            )?;
            let bias = if params_plan.bias.reader == params_plan.gain.reader {
                None
            } else {
                let bias_claim = params_plan
                    .bias_claim
                    .as_ref()
                    .ok_or(Error::ShapeMismatch)?;
                let (bias, bias_material) = prove_norm_param_claim(
                    chain,
                    &params_plan.bias,
                    bias_claim,
                    NormParamKind::Bias,
                    &mut *tables,
                    seed,
                    NORM_PARAM_BIAS_ROOT_TAG,
                    NORM_PARAM_BIAS_ANSWER_TAG,
                    NORM_PARAM_BIAS_POSITION_TAG,
                )?;
                param_claims = Some((gain_material.clone(), Some(bias_material)));
                Some(bias)
            };
            if param_claims.is_none() {
                param_claims = Some((gain_material, None));
            }
            Some(NormParamMessage {
                gain: gain.clone(),
                bias,
            })
        }
        (None, None) => {
            if witness.params.is_some() {
                return Err(Error::ShapeMismatch);
            }
            None
        }
        _ => return Err(Error::ShapeMismatch),
    };
    let output = match (&norm.output, &witness.output) {
        (Some(output_plan), Some(output_witness)) => {
            let output_groups = [
                (
                    &output_plan.broadcast,
                    &output_witness.broadcast,
                    NormOutputGroup::Broadcast,
                ),
                (
                    &output_plan.product,
                    &output_witness.product,
                    NormOutputGroup::Product,
                ),
                (
                    &output_plan.image,
                    &output_witness.image,
                    NormOutputGroup::Image,
                ),
                (&output_plan.out, &output_witness.out, NormOutputGroup::Out),
            ];
            if output_witness.rows.len() != positions as usize {
                return Err(Error::ShapeMismatch);
            }
            for (group_plan, group_witness, kind) in &output_groups {
                if group_witness.committed.group.group != group_plan.group
                    || group_witness.committed.group.fault(positions).is_some()
                    || group_witness.committed.roots.len() != positions as usize
                    || group_witness.messages.len() != positions as usize
                    || group_witness.tails.len() != positions as usize
                    || group_witness.salts.len() != positions as usize
                    || group_witness.pads.len() != positions as usize
                {
                    return Err(Error::ShapeMismatch);
                }
                for (position, entry) in group_witness.committed.group.entries.iter().enumerate() {
                    if entry.position != position as u32
                        || entry.oracle
                            != group_plan
                                .oracle_base
                                .checked_add(position as u64)
                                .ok_or(Error::ResourceLimit)?
                        || entry.shape != group_plan.shape.oracle_shape()
                        || entry.tree != group_plan.shape.tree_plan(position as u32)?
                        || entry.root == Digest::ZERO
                    {
                        return Err(Error::StatementMismatch);
                    }
                }
                let expected =
                    norm_output_values(&output_witness.rows, group_plan.columns, *kind)?;
                let row_count = usize::try_from(group_plan.shape.main_rows)
                    .map_err(|_| Error::ResourceLimit)?;
                let ell = usize::try_from(group_plan.shape.code.ell)
                    .map_err(|_| Error::ResourceLimit)?;
                let columns = usize::try_from(group_plan.columns)
                    .map_err(|_| Error::ResourceLimit)?;
                for (position, message) in group_witness.messages.iter().enumerate() {
                    if message.len() != row_count || message.iter().any(|row| row.len() != ell) {
                        return Err(Error::ShapeMismatch);
                    }
                    for slot in 0..columns {
                        let row = slot % row_count;
                        let place = slot / row_count;
                        if place >= ell
                            || message
                                .get(row)
                                .and_then(|row| row.get(place))
                                .copied()
                                .ok_or(Error::ShapeMismatch)?
                                != expected[position][slot]
                        {
                            return Err(Error::StatementMismatch);
                        }
                    }
                }
            }
            for (tag, (_, group_witness, _)) in
                NORM_OUTPUT_ROOT_TAGS.iter().zip(output_groups.iter())
            {
                absorb_closure_group(chain, *tag, &group_witness.committed.group)?;
            }
            let mut output_seeds = Vec::with_capacity(output_groups.len());
            for (tag, (group_plan, _, _)) in
                NORM_OUTPUT_SEED_TAGS.iter().zip(output_groups.iter())
            {
                output_seeds.push(draw_closure_seed_set(
                    chain,
                    *tag,
                    group_plan.shape.sample_bits()?,
                    group_plan.shape.answers,
                )?);
            }
            let mut output_answers = Vec::with_capacity(output_groups.len());
            for ((group_plan, group_witness, _), seed_set) in
                output_groups.iter().zip(output_seeds.iter())
            {
                output_answers.push(value_group_answers(
                    &group_plan.shape,
                    group_plan.oracle_base,
                    group_plan.group,
                    &group_witness.committed.messages,
                    seed_set,
                )?);
            }
            for (tag, answers) in NORM_OUTPUT_ANSWER_TAGS.iter().zip(output_answers.iter()) {
                absorb_tagged_answers(chain, *tag, answers)?;
            }
            let mut output_messages = Vec::with_capacity(output_groups.len());
            for (index, ((group_plan, group_witness, _), seed_set)) in output_groups
                .iter()
                .zip(output_seeds.iter())
                .enumerate()
            {
                let drawn = draw_closure_group_positions(
                    chain,
                    NORM_OUTPUT_POSITION_TAGS[index],
                    positions,
                    group_plan.openings,
                    group_plan.shape.code.log_n,
                )?;
                let openings = open_value_group_openings(
                    &group_plan.shape,
                    &group_witness.committed.messages,
                    &group_witness.tails,
                    &group_witness.salts,
                    &drawn,
                    &group_witness.committed.trees,
                )?;
                output_messages.push(CommittedPscGroup {
                    message: CommittedGroupMessage::new(group_witness.committed.group.clone()),
                    sample: SendSamples {
                        seed: seed_set.clone(),
                        answers: output_answers[index].clone(),
                    },
                    positions: drawn,
                    openings,
                });
            }
            let mut iter = output_messages.into_iter();
            Some(NormOutputMessage {
                broadcast: iter.next().ok_or(Error::ShapeMismatch)?,
                product: iter.next().ok_or(Error::ShapeMismatch)?,
                image: iter.next().ok_or(Error::ShapeMismatch)?,
                out: iter.next().ok_or(Error::ShapeMismatch)?,
            })
        }
        (None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    let tail = match (
        norm.output.as_ref().and_then(|output| output.tail.as_ref()),
        &witness.output,
    ) {
        (Some(tail_plan), Some(output_witness)) => {
            let tail_witness = output_witness.tail.as_ref().ok_or(Error::ShapeMismatch)?;
            let expected = output_witness
                .rows
                .iter()
                .map(|row| tail_plan.values(row, norm.columns))
                .collect::<Result<Vec<_>>>()?;
            let group_witness = &tail_witness.committed;
            if group_witness.committed.group.group != tail_plan.group.group
                || group_witness.committed.group.fault(positions).is_some()
                || group_witness.committed.roots.len() != positions as usize
                || group_witness.messages.len() != positions as usize
                || group_witness.tails.len() != positions as usize
                || group_witness.salts.len() != positions as usize
                || group_witness.pads.len() != positions as usize
            {
                return Err(Error::ShapeMismatch);
            }
            for (position, entry) in group_witness.committed.group.entries.iter().enumerate() {
                if entry.position != position as u32
                    || entry.oracle
                        != tail_plan
                            .group
                            .oracle_base
                            .checked_add(position as u64)
                            .ok_or(Error::ResourceLimit)?
                    || entry.shape != tail_plan.group.shape.oracle_shape()
                    || entry.tree != tail_plan.group.shape.tree_plan(position as u32)?
                    || entry.root == Digest::ZERO
                {
                    return Err(Error::StatementMismatch);
                }
            }
            let row_count = usize::try_from(tail_plan.group.shape.main_rows)
                .map_err(|_| Error::ResourceLimit)?;
            let ell = usize::try_from(tail_plan.group.shape.code.ell)
                .map_err(|_| Error::ResourceLimit)?;
            let columns = usize::try_from(tail_plan.group.columns)
                .map_err(|_| Error::ResourceLimit)?;
            for (position, message) in group_witness.messages.iter().enumerate() {
                if message.len() != row_count || message.iter().any(|row| row.len() != ell) {
                    return Err(Error::ShapeMismatch);
                }
                for slot in 0..columns {
                    let row = slot % row_count;
                    let place = slot / row_count;
                    if place >= ell
                        || message
                            .get(row)
                            .and_then(|row| row.get(place))
                            .copied()
                            .ok_or(Error::ShapeMismatch)?
                            != expected[position][slot]
                    {
                        return Err(Error::StatementMismatch);
                    }
                }
            }
            absorb_closure_group(chain, NORM_OUTPUT_TAIL_ROOT_TAG, &group_witness.committed.group)?;
            let seed_set = draw_closure_seed_set(
                chain,
                NORM_OUTPUT_TAIL_SEED_TAG,
                tail_plan.group.shape.sample_bits()?,
                tail_plan.group.shape.answers,
            )?;
            let answers = value_group_answers(
                &tail_plan.group.shape,
                tail_plan.group.oracle_base,
                tail_plan.group.group,
                &group_witness.committed.messages,
                &seed_set,
            )?;
            absorb_tagged_answers(chain, NORM_OUTPUT_TAIL_ANSWER_TAG, &answers)?;
            let drawn = draw_closure_group_positions(
                chain,
                NORM_OUTPUT_TAIL_POSITION_TAG,
                positions,
                tail_plan.group.openings,
                tail_plan.group.shape.code.log_n,
            )?;
            let openings = open_value_group_openings(
                &tail_plan.group.shape,
                &group_witness.committed.messages,
                &group_witness.tails,
                &group_witness.salts,
                &drawn,
                &group_witness.committed.trees,
            )?;
            Some(crate::proof::hidden_norm_psc::NormOutputTailMessage {
                group: CommittedPscGroup {
                    message: CommittedGroupMessage::new(group_witness.committed.group.clone()),
                    sample: SendSamples {
                        seed: seed_set,
                        answers,
                    },
                    positions: drawn,
                    openings,
                },
            })
        }
        (None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    let mut iter = group_messages.into_iter();
    Ok((
        NormFirstMessages {
            x: iter.next().ok_or(Error::ShapeMismatch)?,
            sum: iter.next().ok_or(Error::ShapeMismatch)?,
            squares: iter.next().ok_or(Error::ShapeMismatch)?,
            mean: iter.next().ok_or(Error::ShapeMismatch)?,
            variance: iter.next().ok_or(Error::ShapeMismatch)?,
            round: round_message,
            mask,
            lookup,
            rstd,
            params,
            output,
            tail,
        },
        lookup_messages,
        rstd_messages,
        param_claims,
    ))
}

fn verify_norm_first(
    chain: &mut Chain,
    plan: &HiddenLinPlan,
    message: &NormFirstMessages,
) -> Result<()> {
    let norm = plan.norm.as_ref().ok_or(Error::InvalidConfig)?;
    let positions = norm.shape.positions;
    let groups = norm_group_plans(norm);
    let messages = [
        &message.x,
        &message.sum,
        &message.squares,
        &message.mean,
        &message.variance,
    ];
    for (group_plan, message) in groups.iter().zip(messages.iter()) {
        if message.message.group.group != group_plan.group
            || message.fault(&group_plan.shape, positions).is_some()
            || message.message.roots.len() != positions as usize
        {
            return Err(Error::StatementMismatch);
        }
        for (position, entry) in message.message.group.entries.iter().enumerate() {
            if entry.position != position as u32
                || entry.oracle
                    != group_plan
                        .oracle_base
                        .checked_add(position as u64)
                        .ok_or(Error::ResourceLimit)?
                || entry.shape != group_plan.shape.oracle_shape()
                || entry.tree != group_plan.shape.tree_plan(position as u32)?
                || entry.root == Digest::ZERO
            {
                return Err(Error::StatementMismatch);
            }
        }
    }
    if message.x.message.roots != norm.x_roots {
        return Err(Error::StatementMismatch);
    }
    let round = match (&norm.round, &message.round) {
        (Some(round_plan), Some(message)) => {
            if message.message.group.group != round_plan.group.group
                || message
                    .fault(&round_plan.group.shape, positions)
                    .is_some()
                || message.message.roots.len() != positions as usize
            {
                return Err(Error::StatementMismatch);
            }
            for (position, entry) in message.message.group.entries.iter().enumerate() {
                if entry.position != position as u32
                    || entry.oracle
                        != round_plan
                            .group
                            .oracle_base
                            .checked_add(position as u64)
                            .ok_or(Error::ResourceLimit)?
                    || entry.shape != round_plan.group.shape.oracle_shape()
                    || entry.tree != round_plan.group.shape.tree_plan(position as u32)?
                    || entry.root == Digest::ZERO
                {
                    return Err(Error::StatementMismatch);
                }
            }
            Some(message)
        }
        (None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    match (&norm.mask, &message.mask) {
        (Some(mask_plan), Some(mask)) => {
            if mask.tree != mask_plan.tree
                || mask.root == Digest::ZERO
                || mask.sample.seed.len() != mask_plan.answers as usize
                || mask.sample.answers.len() != mask_plan.answers as usize
                || mask.positions.len() != mask_plan.openings
                || mask.opening.positions != mask.positions
            {
                return Err(Error::StatementMismatch);
            }
        }
        (None, None) => {}
        _ => return Err(Error::ShapeMismatch),
    }
    for (tag, message) in NORM_ROOT_TAGS.iter().zip(messages.iter()) {
        absorb_closure_group(chain, *tag, &message.message.group)?;
    }
    if let (Some(round_plan), Some(message)) = (&norm.round, round) {
        absorb_closure_group(chain, NORM_ROUND_ROOT_TAG, &message.message.group)?;
        let _ = round_plan;
    }
    if let (Some(mask_plan), Some(mask)) = (&norm.mask, &message.mask) {
        absorb_norm_mask_root(chain, mask_plan, &mask.root)?;
    }
    for (tag, (group_plan, message)) in NORM_SEED_TAGS
        .iter()
        .zip(groups.iter().zip(messages.iter()))
    {
        let seed_set = draw_closure_seed_set(
            chain,
            *tag,
            group_plan.shape.sample_bits()?,
            group_plan.shape.answers,
        )?;
        if seed_set != message.sample.seed {
            return Err(Error::InvalidProof);
        }
    }
    if let (Some(round_plan), Some(message)) = (&norm.round, round) {
        let seed_set = draw_closure_seed_set(
            chain,
            NORM_ROUND_SEED_TAG,
            round_plan.group.shape.sample_bits()?,
            round_plan.group.shape.answers,
        )?;
        if seed_set != message.sample.seed {
            return Err(Error::InvalidProof);
        }
    }
    if let Some(mask_plan) = &norm.mask {
        let seed_set = draw_closure_seed_set(
            chain,
            NORM_MASK_SEED_TAG,
            mask_plan.sample_bits()?,
            mask_plan.answers,
        )?;
        if seed_set != message.mask.as_ref().ok_or(Error::ShapeMismatch)?.sample.seed {
            return Err(Error::InvalidProof);
        }
    }
    for (index, message) in messages.iter().enumerate() {
        if message.sample.answers.len() != positions as usize {
            return Err(Error::ShapeMismatch);
        }
        absorb_tagged_answers(chain, NORM_ANSWER_TAGS[index], &message.sample.answers)?;
    }
    if let Some(round) = round {
        absorb_tagged_answers(chain, NORM_ROUND_ANSWER_TAG, &round.sample.answers)?;
    }
    if let Some(mask) = &message.mask {
        absorb_tagged_answers(chain, NORM_MASK_ANSWER_TAG, &mask.sample.answers)?;
    }
    for (index, (group_plan, message)) in groups.iter().zip(messages.iter()).enumerate() {
        let positions = draw_closure_group_positions(
            chain,
            NORM_POSITION_TAGS[index],
            positions,
            group_plan.openings,
            group_plan.shape.code.log_n,
        )?;
        if positions != message.positions
            || message.openings.len() != message.positions.len()
        {
            return Err(Error::InvalidProof);
        }
        for (position, (list, opening)) in positions.iter().zip(&message.openings).enumerate() {
            if opening.positions != *list {
                return Err(Error::InvalidIndex);
            }
            let tree = group_plan.shape.tree_plan(position as u32)?;
            let root = message
                .message
                .group
                .entries
                .get(position)
                .map(|entry| entry.root)
                .ok_or(Error::ShapeMismatch)?;
            verify_runtime_opening(&tree, &root, list, opening)?;
        }
    }
    if let (Some(round_plan), Some(message)) = (&norm.round, round) {
        let round_positions = draw_closure_group_positions(
            chain,
            NORM_ROUND_POSITION_TAG,
            positions,
            round_plan.group.openings,
            round_plan.group.shape.code.log_n,
        )?;
        if round_positions != message.positions
            || message.openings.len() != message.positions.len()
        {
            return Err(Error::InvalidProof);
        }
        for (position, (list, opening)) in
            round_positions.iter().zip(&message.openings).enumerate()
        {
            if opening.positions != *list {
                return Err(Error::InvalidIndex);
            }
            let tree = round_plan.group.shape.tree_plan(position as u32)?;
            let root = message
                .message
                .group
                .entries
                .get(position)
                .map(|entry| entry.root)
                .ok_or(Error::ShapeMismatch)?;
            verify_runtime_opening(&tree, &root, list, opening)?;
        }
    }
    if let (Some(mask_plan), Some(mask)) = (&norm.mask, &message.mask) {
        let positions = draw_norm_mask_positions(chain, mask_plan)?;
        if positions != mask.positions || mask.opening.positions != mask.positions {
            return Err(Error::InvalidProof);
        }
        verify_runtime_opening(
            &mask_plan.tree,
            &mask.root,
            &mask.positions,
            &mask.opening,
        )?;
    }
    match (&norm.lookup, &message.lookup) {
        (Some(lookup_plan), Some(lookup)) => {
            verify_norm_lookup_first(chain, lookup_plan, lookup, &NORM_LOOKUP_TAGS)?
        }
        (None, None) => {}
        _ => return Err(Error::ShapeMismatch),
    }
    match (&norm.rstd, &message.rstd) {
        (Some(rstd_plan), Some(rstd)) => {
            absorb_closure_group(chain, NORM_RSTD_BITS_ROOT_TAG, &rstd.bits.message.group)?;
            absorb_rstd_mask_root(chain, rstd_plan, &rstd.mask.root)?;
            let bits_shape = &rstd_plan.commit.bits_shape;
            let bits_seed = draw_closure_seed_set(
                chain,
                NORM_RSTD_BITS_SEED_TAG,
                bits_shape.sample_bits()?,
                bits_shape.answers,
            )?;
            if bits_seed != rstd.bits.sample.seed {
                return Err(Error::InvalidProof);
            }
            absorb_tagged_answers(chain, NORM_RSTD_BITS_ANSWER_TAG, &rstd.bits.sample.answers)?;
            let mask_seed = draw_closure_seed_set(
                chain,
                NORM_RSTD_MASK_SEED_TAG,
                lookup_mask_sample_bits(&rstd_plan.commit)?,
                1,
            )?;
            if mask_seed != rstd.mask.sample.seed {
                return Err(Error::InvalidProof);
            }
            absorb_tagged_answers(chain, NORM_RSTD_MASK_ANSWER_TAG, &rstd.mask.sample.answers)?;
            let bits_positions = draw_closure_group_positions(
                chain,
                NORM_RSTD_BITS_POSITION_TAG,
                1,
                rstd_plan.commit.bits_openings,
                bits_shape.code.log_n,
            )?;
            if bits_positions != rstd.bits.positions {
                return Err(Error::InvalidProof);
            }
            let bits_entry = rstd
                .bits
                .message
                .group
                .entries
                .first()
                .ok_or(Error::ShapeMismatch)?;
            verify_runtime_opening(
                &bits_entry.tree,
                &bits_entry.root,
                &bits_positions[0],
                &rstd.bits.openings[0],
            )?;
            let mask_positions = draw_rstd_mask_positions(chain, rstd_plan)?;
            if mask_positions != rstd.mask.positions {
                return Err(Error::InvalidProof);
            }
            verify_runtime_opening(
                &rstd_plan.commit.mask_tree,
                &rstd.mask.root,
                &mask_positions,
                &rstd.mask.opening,
            )?;
        }
        (None, None) => {}
        _ => return Err(Error::ShapeMismatch),
    }
    match (&norm.params, &message.params) {
        (Some(params_plan), Some(params)) => {
            params.check(params_plan)?;
            let gain_claim = params_plan
                .gain_claim
                .as_ref()
                .ok_or(Error::ShapeMismatch)?;
            verify_norm_param_claim(
                chain,
                &params_plan.gain,
                gain_claim,
                &params.gain,
                NORM_PARAM_GAIN_ROOT_TAG,
                NORM_PARAM_GAIN_ANSWER_TAG,
                NORM_PARAM_GAIN_POSITION_TAG,
            )?;
            if let Some(bias) = &params.bias {
                let bias_claim = params_plan
                    .bias_claim
                    .as_ref()
                    .ok_or(Error::ShapeMismatch)?;
                verify_norm_param_claim(
                    chain,
                    &params_plan.bias,
                    bias_claim,
                    bias,
                    NORM_PARAM_BIAS_ROOT_TAG,
                    NORM_PARAM_BIAS_ANSWER_TAG,
                    NORM_PARAM_BIAS_POSITION_TAG,
                )?;
            }
        }
        (None, None) => {}
        _ => return Err(Error::ShapeMismatch),
    }
    match (&norm.output, &message.output) {
        (Some(output_plan), Some(output)) => {
            output.check(output_plan, positions)?;
            let output_groups = [
                (&output.broadcast, &output_plan.broadcast),
                (&output.product, &output_plan.product),
                (&output.image, &output_plan.image),
                (&output.out, &output_plan.out),
            ];
            for (tag, (group_message, _)) in
                NORM_OUTPUT_ROOT_TAGS.iter().zip(output_groups.iter())
            {
                absorb_closure_group(chain, *tag, &group_message.message.group)?;
            }
            for (tag, (group_message, group_plan)) in
                NORM_OUTPUT_SEED_TAGS.iter().zip(output_groups.iter())
            {
                let seed_set = draw_closure_seed_set(
                    chain,
                    *tag,
                    group_plan.shape.sample_bits()?,
                    group_plan.shape.answers,
                )?;
                if seed_set != group_message.sample.seed {
                    return Err(Error::InvalidProof);
                }
            }
            for (index, (group_message, _)) in output_groups.iter().enumerate() {
                if group_message.sample.answers.len() != positions as usize {
                    return Err(Error::ShapeMismatch);
                }
                absorb_tagged_answers(
                    chain,
                    NORM_OUTPUT_ANSWER_TAGS[index],
                    &group_message.sample.answers,
                )?;
            }
            for (index, (group_message, group_plan)) in output_groups.iter().enumerate() {
                let drawn = draw_closure_group_positions(
                    chain,
                    NORM_OUTPUT_POSITION_TAGS[index],
                    positions,
                    group_plan.openings,
                    group_plan.shape.code.log_n,
                )?;
                if drawn != group_message.positions
                    || group_message.openings.len() != group_message.positions.len()
                {
                    return Err(Error::InvalidProof);
                }
                for (position, (list, opening)) in
                    drawn.iter().zip(&group_message.openings).enumerate()
                {
                    if opening.positions != *list {
                        return Err(Error::InvalidIndex);
                    }
                    let tree = group_plan.shape.tree_plan(position as u32)?;
                    let root = group_message
                        .message
                        .group
                        .entries
                        .get(position)
                        .map(|entry| entry.root)
                        .ok_or(Error::ShapeMismatch)?;
                    verify_runtime_opening(&tree, &root, list, opening)?;
                }
            }
        }
        (None, None) => {}
        _ => return Err(Error::ShapeMismatch),
    }
    let tail_plan = norm.output.as_ref().and_then(|output| output.tail.as_ref());
    match (tail_plan, &message.tail) {
        (Some(tail_plan), Some(tail)) => {
            tail.check(tail_plan, positions)?;
            absorb_closure_group(chain, NORM_OUTPUT_TAIL_ROOT_TAG, &tail.group.message.group)?;
            let seed_set = draw_closure_seed_set(
                chain,
                NORM_OUTPUT_TAIL_SEED_TAG,
                tail_plan.group.shape.sample_bits()?,
                tail_plan.group.shape.answers,
            )?;
            if seed_set != tail.group.sample.seed {
                return Err(Error::InvalidProof);
            }
            if tail.group.sample.answers.len() != positions as usize {
                return Err(Error::ShapeMismatch);
            }
            absorb_tagged_answers(
                chain,
                NORM_OUTPUT_TAIL_ANSWER_TAG,
                &tail.group.sample.answers,
            )?;
            let drawn = draw_closure_group_positions(
                chain,
                NORM_OUTPUT_TAIL_POSITION_TAG,
                positions,
                tail_plan.group.openings,
                tail_plan.group.shape.code.log_n,
            )?;
            if drawn != tail.group.positions
                || tail.group.openings.len() != tail.group.positions.len()
            {
                return Err(Error::InvalidProof);
            }
            for (position, (list, opening)) in drawn.iter().zip(&tail.group.openings).enumerate() {
                if opening.positions != *list {
                    return Err(Error::InvalidIndex);
                }
                let tree = tail_plan.group.shape.tree_plan(position as u32)?;
                let root = tail
                    .group
                    .message
                    .group
                    .entries
                    .get(position)
                    .map(|entry| entry.root)
                    .ok_or(Error::ShapeMismatch)?;
                verify_runtime_opening(&tree, &root, list, opening)?;
            }
        }
        (None, None) => {}
        _ => return Err(Error::ShapeMismatch),
    }
    Ok(())
}

fn draw_norm_param_positions(
    chain: &mut Chain,
    tag: u64,
    count: usize,
    log_n: u32,
) -> Result<Vec<u64>> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(tag);
    message.push_u64(count as u64);
    let prefix = message.prefix(KIND_POSITIONS, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    let mut drawn = Vec::with_capacity(count);
    for _ in 0..count {
        drawn.push(chain.draw_position(log_n)?);
    }
    Ok(unique_sorted(&drawn))
}

fn absorb_norm_param_root(chain: &mut Chain, tag: u64, root: &Digest) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(tag);
    push_digest_chain(&mut message, root);
    let prefix = message.prefix(KIND_SEED, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn absorb_norm_param_answers(chain: &mut Chain, tag: u64, answers: &[Fp3]) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(tag);
    message.push_u64(answers.len() as u64);
    for answer in answers {
        message.push_fp3(answer);
    }
    let prefix = message.prefix(KIND_EMPTY, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// Proves one parameter table claim: fold the single table row, commit the
/// folded message as G_u, absorb its root, sample and absorb its answers, then
/// draw and open the table columns.
fn prove_norm_param_claim(
    chain: &mut Chain,
    identity: &NormParamTableIdentity,
    claim: &NormParamClaimPlan,
    kind: NormParamKind,
    table: &mut dyn NormParamTableProver,
    seed: &Digest,
    root_tag: u64,
    answer_tag: u64,
    column_tag: u64,
) -> Result<(
    NormParamClaimMessage,
    crate::proof::hidden_norm_psc::NormParamClaimWitness,
)> {
    let gu = &claim.gu;
    let mut coins = ByteMaskCoins::new(derive_run_seed(seed, 2, kind.encode())?)?;
    let gu_pads = draw_values(&mut coins, gu.u.answers as usize)?;
    let gu_tails = {
        let rows = usize::try_from(gu.u.main_rows).map_err(|_| Error::ResourceLimit)?;
        let tail_len = usize::try_from(gu.u.code.r).map_err(|_| Error::ResourceLimit)?;
        let mut tails = Vec::with_capacity(rows);
        for _ in 0..rows {
            tails.push(draw_values(&mut coins, tail_len)?);
        }
        tails
    };
    let gu_salts = draw_salt_bytes(
        &mut coins,
        usize::try_from(gu.tree.columns()).map_err(|_| Error::ResourceLimit)?,
        gu.tree.salt_bytes,
    )?;
    let (folded, folded_tail) = fold_table(table.source(kind), identity.reader.shape, &[])?;
    let mut gu_witness = gu_table(gu, &folded, &folded_tail)?;
    let (main_pads, mask_pads) = gu_sample_pad_offsets(gu)?;
    if !mask_pads.is_empty() || main_pads.len() != gu_pads.len() {
        return Err(Error::ShapeMismatch);
    }
    for (offset, pad) in main_pads.iter().zip(&gu_pads) {
        let slot = gu_witness
            .main
            .get_mut(*offset)
            .ok_or(Error::ShapeMismatch)?;
        *slot = *pad;
    }
    let gu_rows = usize::try_from(gu.u.main_rows).map_err(|_| Error::ResourceLimit)?;
    let gu_ell = usize::try_from(gu.u.code.ell).map_err(|_| Error::ResourceLimit)?;
    let mut gu_message = Vec::with_capacity(gu_rows);
    for row in 0..gu_rows {
        let start = row.checked_mul(gu_ell).ok_or(Error::ResourceLimit)?;
        let end = start.checked_add(gu_ell).ok_or(Error::ResourceLimit)?;
        gu_message.push(
            gu_witness
                .main
                .get(start..end)
                .ok_or(Error::ShapeMismatch)?
                .to_vec(),
        );
    }
    let tree = commit_rows_tree(gu.tree.clone(), gu.u.code, &gu_message, &gu_tails, &gu_salts)?;
    let root = *tree.root();
    let mut registry = GuRegistry::new();
    registry.receive(
        gu,
        &GuNotice {
            root,
            oracle: gu.oracle,
            tree_number: gu.tree.tree_number,
            part: gu.part,
        },
    )?;
    absorb_norm_param_root(chain, root_tag, &root)?;
    let seeds = draw_seed_set(chain, gu.sample_bits()?, gu.u.answers)?;
    let answers = gu_answers(gu, &registry, &gu_witness, &seeds)?;
    absorb_norm_param_answers(chain, answer_tag, &answers)?;
    let positions = draw_norm_param_positions(
        chain,
        column_tag,
        claim.column_draws,
        gu.index.code.log_n,
    )?;
    let openings = table.open_columns(kind, &positions)?;
    Ok((
        NormParamClaimMessage {
            root,
            seeds,
            answers,
            openings,
        },
        crate::proof::hidden_norm_psc::NormParamClaimWitness {
            message: gu_message,
            tails: gu_tails,
            salts: gu_salts,
        },
    ))
}

/// Verifies one parameter table claim in the fixed order: the root, the
/// sample seeds, the answers, then the drawn columns and their openings.
fn verify_norm_param_claim(
    chain: &mut Chain,
    identity: &NormParamTableIdentity,
    claim: &NormParamClaimPlan,
    message: &NormParamClaimMessage,
    root_tag: u64,
    answer_tag: u64,
    column_tag: u64,
) -> Result<()> {
    let gu = &claim.gu;
    absorb_norm_param_root(chain, root_tag, &message.root)?;
    let seeds = draw_seed_set(chain, gu.sample_bits()?, gu.u.answers)?;
    if seeds != message.seeds {
        return Err(Error::InvalidProof);
    }
    absorb_norm_param_answers(chain, answer_tag, &message.answers)?;
    let positions =
        draw_norm_param_positions(chain, column_tag, claim.column_draws, gu.index.code.log_n)?;
    if message
        .openings
        .iter()
        .map(|opening| opening.position)
        .collect::<Vec<_>>()
        != positions
    {
        return Err(Error::StatementMismatch);
    }
    verify_columns(&identity.reader, &positions, &message.openings)?;
    Ok(())
}

/// Scatters the equality weights of a column point into the BAT table layout of
/// one parameter message. The coefficients bind the parameter vector
/// evaluation to the committed folded message; the caller adds the term to a
/// relation row. Nothing here reads a prover-supplied value.
fn norm_param_eval_coefficients(
    gu: &GuShape,
    width: u64,
    point: &[Fp3],
) -> Result<Vec<Fp3>> {
    if width == 0
        || !width.is_power_of_two()
        || point.len() != width.trailing_zeros() as usize
    {
        return Err(Error::ShapeMismatch);
    }
    if !gu.rows_high {
        return Err(Error::UnsupportedPrimitive);
    }
    let room = gu
        .u
        .main_rows
        .checked_mul(gu.u.code.ell)
        .and_then(|value| value.checked_sub(gu.u.answers as u64))
        .ok_or(Error::ResourceLimit)?;
    if width > room {
        return Err(Error::ShapeMismatch);
    }
    let rows = usize::try_from(gu.u.main_rows).map_err(|_| Error::ResourceLimit)?;
    let ell = usize::try_from(gu.u.code.ell).map_err(|_| Error::ResourceLimit)?;
    let eq = crate::proof::output_closure::eq_vector(point)?;
    if eq.len() != width as usize {
        return Err(Error::ShapeMismatch);
    }
    let mut coefficients = vec![Fp3::ZERO; rows * ell];
    for column in 0..width as usize {
        let index = gu_vector_bat_index(gu, column as u64)?;
        let slot = coefficients.get_mut(index).ok_or(Error::ShapeMismatch)?;
        *slot = slot.add(&eq[column]);
    }
    Ok(coefficients)
}

/// The record and column declarations of one parameter table claim. Both the
/// prover and the verifier build them from the same claim message and plan
/// reader; the rows tie the folded message to the committed table through the
/// record answers and the opened columns.
pub(crate) struct NormParamClaimRows {
    pub record: Vec<GuRelationRow>,
    pub column: Vec<GuRelationRow>,
}

fn norm_param_claim_rows(
    identity: &NormParamTableIdentity,
    claim: &NormParamClaimPlan,
    message: &NormParamClaimMessage,
) -> Result<NormParamClaimRows> {
    let gu = &claim.gu;
    let reader = &identity.reader;
    if gu.index != reader.shape {
        return Err(Error::StatementMismatch);
    }
    let positions: Vec<u64> = message
        .openings
        .iter()
        .map(|opening| opening.position)
        .collect();
    if positions.is_empty() {
        return Err(Error::ShapeMismatch);
    }
    let column_map = table_column_map(reader, &positions, &message.openings)?;
    let mut registry = GuRegistry::new();
    registry.receive(
        gu,
        &GuNotice {
            root: message.root,
            oracle: gu.oracle,
            tree_number: gu.tree.tree_number,
            part: gu.part,
        },
    )?;
    let eq_gamma = foldeq_weights(&[], reader.shape.v)?;
    let name = reader.name();
    let record_seeds =
        record_seeds(&reader.shape, &name, reader.shape.answers as usize)?;
    let mut record = Vec::with_capacity(reader.shape.answers as usize);
    for answer in 0..reader.shape.answers {
        record.push(record_row(
            gu,
            &registry,
            &reader.record,
            &name,
            &record_seeds,
            &eq_gamma,
            answer,
        )?);
    }
    let mut column = Vec::with_capacity(positions.len());
    for position in &positions {
        let values = column_for(&column_map, *position)?;
        column.push(column_row(gu, &registry, *position, &values, &eq_gamma)?);
    }
    Ok(NormParamClaimRows { record, column })
}

fn draw_rstd_seed(chain: &mut Chain) -> Result<u64> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(NORM_RSTD_SEED_TAG);
    let prefix = message.prefix(KIND_SEED, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(chain.draw_f()?.as_canonical_u64())
}

/// Checks the trusted table view against the plan and the public first-row
/// padding pair. The all-zero point evaluates each column at row zero.
fn check_rstd_table_view(
    plan: &crate::proof::hidden_norm_psc::NormRstdPlan,
    table: &dyn crate::proof::lookup::LookupTable,
) -> Result<()> {
    if table.config() != plan.table
        || table.columns() != plan.lookup.columns as usize
        || table.row_count() != plan.lookup.table_rows()?
    {
        return Err(Error::StatementMismatch);
    }
    table.check_content()?;
    let point = vec![Fp3::ZERO; plan.lookup.table_log as usize];
    let first_index = table.column_at(0, &point)?;
    let first_value = table.column_at(1, &point)?;
    if first_index != plan.key.padding_values[0]
        || first_value != plan.key.padding_values[1]
    {
        return Err(Error::StatementMismatch);
    }
    Ok(())
}

/// Draws the column point of one pointwise product instance.
fn draw_norm_output_point(chain: &mut Chain, instance: u64, bits: usize) -> Result<Vec<Fp3>> {
    let mut message = chain.message();
    message.push_u64(crate::proof::hidden_norm_psc::NORM_PSC_VERSION);
    message.push_u64(crate::proof::hidden_norm_psc::NORM_OUTPUT_POINT_TAG);
    message.push_u64(instance);
    let prefix = message.prefix(KIND_SEED, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    let mut point = Vec::with_capacity(bits);
    for _ in 0..bits {
        point.push(chain.draw_e()?);
    }
    Ok(point)
}

/// Draws a tail-section point with its own prefix tag.
fn draw_norm_tail_point(chain: &mut Chain, tag: u64, bits: usize) -> Result<Vec<Fp3>> {
    let mut message = chain.message();
    message.push_u64(crate::proof::hidden_norm_psc::NORM_PSC_VERSION);
    message.push_u64(tag);
    message.push_u64(0);
    let prefix = message.prefix(KIND_SEED, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    let mut point = Vec::with_capacity(bits);
    for _ in 0..bits {
        point.push(chain.draw_e()?);
    }
    Ok(point)
}

/// The two committed column maps of one pointwise product instance: the
/// broadcast row against the gain vector, or the centered row against the
/// product row. The centered row is zero on non-valid columns.
fn norm_pointwise_maps(
    plan: &NormPlan,
    row: &[i64],
    core: &crate::proof::hidden_norm::NormCoreRow,
    gain: &[i64],
    output_row: &crate::proof::hidden_norm::NormOutputRow,
    kind: u8,
) -> Result<(Vec<Fp3>, Vec<Fp3>)> {
    use crate::proof::output_closure::fp3_signed;

    let columns = usize::try_from(plan.columns).map_err(|_| Error::ResourceLimit)?;
    let valid = plan.shape.valid as usize;
    if row.len() != plan.shape.hidden as usize
        || gain.len() != columns
        || output_row.broadcast.len() != columns
        || output_row.product.len() != columns
        || output_row.out.len() != columns
    {
        return Err(Error::ShapeMismatch);
    }
    let mut left = Vec::with_capacity(columns);
    let mut right = Vec::with_capacity(columns);
    match kind {
        0 => {
            for index in 0..columns {
                left.push(fp3_signed(output_row.broadcast[index] as i128)?);
                right.push(fp3_signed(gain[index] as i128)?);
            }
        }
        1 => {
            for index in 0..columns {
                let centered = if index < valid {
                    (row[index] as i128)
                        .checked_sub(core.mean as i128)
                        .ok_or(Error::BoundViolation)?
                } else {
                    0
                };
                left.push(fp3_signed(centered)?);
                right.push(fp3_signed(output_row.product[index])?);
            }
        }
        _ => return Err(Error::InvalidConfig),
    }
    Ok((left, right))
}

/// Resolves the key cells of the norm digit lookup across the committed
/// rounding and tail messages. The key groups name the message each source
/// addresses; the caller supplies the unencoded message rows of both groups.
fn norm_lookup_key_values(
    plan: &NormPlan,
    lookup: &crate::proof::hidden_norm_psc::NormLookupPlan,
    round_rows: &[Vec<Vec<Fp3>>],
    tail_rows: &[Vec<Vec<Fp3>>],
) -> Result<Vec<Fp3>> {
    if lookup.lookup.columns != 1 {
        return Err(Error::InvalidConfig);
    }
    let positions = plan.shape.positions as u64;
    let round_base = plan.round.as_ref().map(|round| round.group.oracle_base);
    let tail_base = plan
        .output
        .as_ref()
        .and_then(|output| output.tail.as_ref())
        .map(|tail| tail.group.oracle_base);
    let resolve = |source: &crate::proof::lookup::LookupKeySource| -> Result<Fp3> {
        let in_round = round_base
            .map(|base| source.oracle >= base && source.oracle < base.saturating_add(positions))
            .unwrap_or(false);
        let in_tail = tail_base
            .map(|base| source.oracle >= base && source.oracle < base.saturating_add(positions))
            .unwrap_or(false);
        let rows = match (in_round, in_tail) {
            (true, false) => round_rows,
            (false, true) => tail_rows,
            _ => return Err(Error::StatementMismatch),
        };
        rows.get(source.position as usize)
            .and_then(|message| message.get(source.row as usize))
            .and_then(|row| row.get(source.place as usize))
            .copied()
            .ok_or(Error::ShapeMismatch)
    };
    lookup.key.read_with(&lookup.lookup, Fp3::ONE, &resolve)
}

fn prove_norm_second(
    chain: &mut Chain,
    plan: &NormPlan,
    first: &NormFirstMessages,
    witness: &NormWitness,
    rstd_table: Option<&dyn crate::proof::lookup::LookupTable>,
) -> Result<(NormPscSecond, NormRowBinding)> {
    let positions = plan.shape.positions as usize;
    let relations = plan.relations.clone();
    let valid = plan.shape.valid as usize;
    let columns = usize::try_from(plan.columns).map_err(|_| Error::ResourceLimit)?;
    let mask_plan = plan.mask.as_ref().ok_or(Error::InvalidConfig)?;
    let mask_witness = witness.mask.as_ref().ok_or(Error::ShapeMismatch)?;
    if first.mask.is_none()
        || witness.rows.len() != positions
        || witness.core.len() != positions
        || witness.rows.iter().any(|row| row.len() != plan.shape.hidden as usize)
    {
        return Err(Error::ShapeMismatch);
    }
    let instances = positions
        .checked_mul(relations.len())
        .ok_or(Error::ResourceLimit)?;
    if mask_witness.entries.len() != instances
        || mask_plan.hiding != mask_witness.entries[0].hiding.len()
    {
        return Err(Error::ShapeMismatch);
    }
    let lookup = match &plan.lookup {
        Some(lookup_plan) => {
            let lookup_witness = witness.lookup.as_ref().ok_or(Error::ShapeMismatch)?;
            let messages = lookup_witness
                .messages
                .as_ref()
                .ok_or(Error::InvalidSessionState)?;
            let round_witness = witness.round.as_ref().ok_or(Error::ShapeMismatch)?;
            let first_lookup = first.lookup.as_ref().ok_or(Error::ShapeMismatch)?;
            if first_lookup.bits.message.roots.len() != 1
                || first_lookup.bits.message.roots[0] != messages.bits_root()
                || first_lookup.mask.root != messages.mask_root
            {
                return Err(Error::StatementMismatch);
            }
            let seed = draw_lookup_seed(chain, NORM_LOOKUP_SEED_TAG)?;
            let tail_rows: &[Vec<Vec<Fp3>>] = witness
                .output
                .as_ref()
                .and_then(|output| output.tail.as_ref())
                .map(|tail| tail.committed.messages.as_slice())
                .unwrap_or(&[]);
            let key_values = norm_lookup_key_values(
                plan,
                lookup_plan,
                &round_witness.committed.messages,
                tail_rows,
            )?;
            let (proof, claims) = crate::proof::lookup::prove_lookup_with_key_values(
                &lookup_plan.lookup,
                &lookup_plan.commit,
                &lookup_plan.key,
                &key_values,
                &crate::proof::lookup::LookupWitness {
                    bits: lookup_witness.bits.clone(),
                },
                messages,
                &crate::proof::lookup::ByteTable,
                seed,
            )?;
            Some(NormLookupSecond { proof, claims })
        }
        None => None,
    };
    let rstd = match (&plan.rstd, rstd_table) {
        (Some(rstd_plan), Some(table)) => {
            let rstd_witness = witness.rstd.as_ref().ok_or(Error::ShapeMismatch)?;
            let messages = rstd_witness
                .messages
                .as_ref()
                .ok_or(Error::InvalidSessionState)?;
            let round_witness = witness.round.as_ref().ok_or(Error::ShapeMismatch)?;
            let first_rstd = first.rstd.as_ref().ok_or(Error::ShapeMismatch)?;
            if first_rstd.bits.message.roots.len() != 1
                || first_rstd.bits.message.roots[0] != messages.bits_root()
                || first_rstd.mask.root != messages.mask_root
            {
                return Err(Error::StatementMismatch);
            }
            check_rstd_table_view(rstd_plan, table)?;
            let seed = draw_rstd_seed(chain)?;
            let (proof, claims) = crate::proof::lookup::prove_lookup_with_messages(
                &rstd_plan.lookup,
                &rstd_plan.commit,
                &rstd_plan.key,
                &round_witness.messages,
                &crate::proof::lookup::LookupWitness {
                    bits: rstd_witness.bits.clone(),
                },
                messages,
                table,
                seed,
            )?;
            Some(NormLookupSecond { proof, claims })
        }
        (None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    let claim = DotClaim {
        masks: Vec::new(),
        target: Fp3::ZERO,
    };
    let mut psc = Vec::with_capacity(instances);
    let mut point_values = Vec::with_capacity(instances);
    let mut bindings = Vec::with_capacity(instances);
    for position in 0..positions {
        for (relation_index, relation) in relations.iter().enumerate() {
            let (left, right) = crate::proof::hidden_norm_psc::rel_map(
                &witness.core[position],
                &witness.rows[position],
                valid,
                columns,
                *relation,
                plan.round.as_ref(),
                witness
                    .rsqrt
                    .as_ref()
                    .and_then(|rows| rows.get(position)),
            )?;
            let instance = position * relations.len() + relation_index;
            let entries = &mask_witness.entries[instance];
            let mut challenges = crate::proof::committed_psc::PositionPscChallenges::new(
                chain,
                crate::proof::hidden_norm_psc::NORM_PSC_TAGS,
                instance as u64,
            );
            let (proof, output) = crate::proof::psc::prove_dot_sumcheck(
                &left,
                &right,
                &claim,
                entries,
                &mut challenges,
            )?;
            psc.push(proof);
            point_values.push([output.ends[0].value, output.ends[1].value]);
            bindings.push(NormPositionBinding {
                position: position as u32,
                relation: *relation,
                output,
            });
        }
    }
    let (output_message, output_binding) = match (&plan.output, &witness.output) {
        (Some(output_plan), Some(output_witness)) => {
            let params = witness.params.as_ref().ok_or(Error::ShapeMismatch)?;
            let width = usize::try_from(plan.columns).map_err(|_| Error::ResourceLimit)?;
            let bits = usize::try_from(plan.columns.trailing_zeros())
                .map_err(|_| Error::ResourceLimit)?;
            if params.gain.len() != width
                || output_witness.rows.len() != positions
                || output_witness
                    .rows
                    .iter()
                    .any(|row| row.broadcast.len() != width)
            {
                return Err(Error::ShapeMismatch);
            }
            let pointwise_entries = crate::proof::hidden_norm_psc::norm_pointwise_entries(
                mask_plan,
                &mask_witness.committed,
                instances,
            )?;
            let mut claims = Vec::with_capacity(positions * 2);
            let mut proofs = Vec::with_capacity(positions * 2);
            let mut output_bindings = Vec::with_capacity(positions * 2);
            for position in 0..positions {
                for kind in 0..2u8 {
                    let instance = position * 2 + kind as usize;
                    let point = draw_norm_output_point(chain, instance as u64, bits)?;
                    let (left, right) = norm_pointwise_maps(
                        plan,
                        &witness.rows[position],
                        &witness.core[position],
                        &params.gain,
                        &output_witness.rows[position],
                        kind,
                    )?;
                    let eq = crate::proof::output_closure::eq_vector(&point)?;
                    if eq.len() != width {
                        return Err(Error::ShapeMismatch);
                    }
                    let mut target = Fp3::ZERO;
                    for index in 0..width {
                        target = target.add(&eq[index].mul(&left[index]).mul(&right[index]));
                    }
                    let variables = (0..bits)
                        .map(|index| crate::proof::psc::Weight::equality(point[index], 3))
                        .collect::<Result<Vec<_>>>()?;
                    let mut inner = crate::proof::psc_general::WeightedProductInner::new(
                        &left, &right, &point,
                    )?;
                    let entries = pointwise_entries
                        .get(instance)
                        .ok_or(Error::ShapeMismatch)?;
                    let claim_slot = mask_plan.claim_slot(instances, instance)?;
                    let claim_mask =
                        mask_witness.committed.entry(mask_plan, claim_slot)?;
                    let sent = target.add(&claim_mask);
                    let claim_masks = [crate::proof::psc_general::MaskTerm {
                        slot: claim_slot,
                        coefficient: Fp3::ONE,
                    }];
                    let mask_entry = |slot: usize| -> Result<Fp3> {
                        mask_witness.committed.entry(mask_plan, slot)
                    };
                    let mut challenges =
                        crate::proof::committed_psc::PositionPscChallenges::new(
                            chain,
                            crate::proof::hidden_norm_psc::NORM_OUTPUT_PSC_TAGS,
                            instance as u64,
                        );
                    challenges.send_claim(sent)?;
                    let (proof, checked) = crate::proof::psc_general::prove_general(
                        &variables,
                        &claim_masks,
                        &mask_entry,
                        sent,
                        &crate::proof::psc_general::Summand::Product,
                        None,
                        &mut inner,
                        entries,
                        &mut challenges,
                    )?;
                    claims.push(sent);
                    proofs.push(proof);
                    output_bindings.push(crate::proof::hidden_norm_psc::NormOutputBinding {
                        position: position as u32,
                        kind,
                        point: point.clone(),
                        claim: sent,
                        output: checked,
                    });
                }
            }
            let _ = output_plan;
            (
                Some(crate::proof::hidden_norm_psc::NormOutputSecond { claims, proofs }),
                Some(output_bindings),
            )
        }
        (None, None) => (None, None),
        _ => return Err(Error::ShapeMismatch),
    };
    let (tail_message, tail_bindings) = match (
        plan.output.as_ref().and_then(|output| output.tail.as_ref()),
        &witness.output,
    ) {
        (Some(tail_plan), Some(output_witness)) => {
            let entries = crate::proof::hidden_norm_psc::norm_tail_entries(
                mask_plan,
                &mask_witness.committed,
                instances,
            )?;
            let variables = crate::proof::hidden_norm_psc::output_tail_variables(plan.columns)?;
            let claim = DotClaim {
                masks: Vec::new(),
                target: Fp3::ZERO,
            };
            let mut psc = Vec::with_capacity(positions * crate::proof::hidden_norm_psc::NORM_OUTPUT_TAIL_RELATIONS);
            let mut point_values = Vec::with_capacity(positions * crate::proof::hidden_norm_psc::NORM_OUTPUT_TAIL_RELATIONS);
            let mut tail_bindings = Vec::with_capacity(positions * crate::proof::hidden_norm_psc::NORM_OUTPUT_TAIL_RELATIONS);
            for position in 0..positions {
                for relation in 0..crate::proof::hidden_norm_psc::NORM_OUTPUT_TAIL_RELATIONS {
                    let instance = position * crate::proof::hidden_norm_psc::NORM_OUTPUT_TAIL_RELATIONS + relation;
                    let (left, right) = crate::proof::hidden_norm_psc::output_rel_map(
                        tail_plan,
                        &output_witness.rows[position],
                        plan.columns,
                        plan.output.as_ref().ok_or(Error::InvalidConfig)?.shift,
                        relation,
                    )?;
                    if left.len() != right.len() || left.len() != 1usize << variables {
                        return Err(Error::ShapeMismatch);
                    }
                    let entry = entries.get(instance).ok_or(Error::ShapeMismatch)?;
                    let mut challenges =
                        crate::proof::committed_psc::PositionPscChallenges::new(
                            chain,
                            crate::proof::hidden_norm_psc::NORM_TAIL_PSC_TAGS,
                            instance as u64,
                        );
                    let (proof, output) = crate::proof::psc::prove_dot_sumcheck(
                        &left,
                        &right,
                        &claim,
                        entry,
                        &mut challenges,
                    )?;
                    psc.push(proof);
                    point_values.push([output.ends[0].value, output.ends[1].value]);
                    tail_bindings.push(crate::proof::hidden_norm_psc::NormTailBinding {
                        position: position as u32,
                        relation,
                        output,
                    });
                }
            }
            (
                Some(crate::proof::hidden_norm_psc::NormTailPscSecond { psc, point_values }),
                Some(tail_bindings),
            )
        }
        (None, None) => (None, None),
        _ => return Err(Error::ShapeMismatch),
    };
    let bias_point = match &tail_message {
        Some(_) => {
            let bits = usize::try_from(plan.columns.trailing_zeros())
                .map_err(|_| Error::ResourceLimit)?;
            Some(draw_norm_tail_point(
                chain,
                crate::proof::hidden_norm_psc::NORM_BIAS_POINT_TAG,
                bits,
            )?)
        }
        None => None,
    };
    Ok((
        NormPscSecond {
            psc,
            point_values,
            lookup: lookup.clone(),
            rstd: rstd.clone(),
            output: output_message,
            tail: tail_message,
        },
        NormRowBinding {
            positions: bindings,
            lookup,
            rstd,
            output: output_binding,
            tail: tail_bindings,
            bias_point,
        },
    ))
}

fn verify_norm_second(
    chain: &mut Chain,
    plan: &NormPlan,
    first: &NormFirstMessages,
    messages: &NormPscSecond,
    rstd_table: Option<&dyn crate::proof::lookup::LookupTable>,
) -> Result<NormRowBinding> {
    let positions = plan.shape.positions as usize;
    let relations = plan.relations.clone();
    let variables = plan.variables;
    let mask_plan = plan.mask.as_ref().ok_or(Error::InvalidConfig)?;
    if first.mask.is_none() {
        return Err(Error::ShapeMismatch);
    }
    messages.check_for(plan)?;
    let lookup_binding = match (&plan.lookup, &messages.lookup) {
        (Some(lookup_plan), Some(lookup)) => {
            let first_lookup = first.lookup.as_ref().ok_or(Error::ShapeMismatch)?;
            let roots = crate::proof::lookup_messages::LookupRoots {
                bits: first_lookup
                    .bits
                    .message
                    .roots
                    .first()
                    .copied()
                    .ok_or(Error::ShapeMismatch)?,
                mask: first_lookup.mask.root,
            };
            let seed = draw_lookup_seed(chain, NORM_LOOKUP_SEED_TAG)?;
            let claims = crate::proof::lookup::verify_lookup(
                &lookup_plan.lookup,
                &lookup_plan.commit,
                &lookup_plan.key,
                &roots,
                &lookup.proof,
                &crate::proof::lookup::ByteTable,
                seed,
            )?;
            if claims != lookup.claims {
                return Err(Error::InvalidProof);
            }
            Some(NormLookupSecond {
                proof: lookup.proof.clone(),
                claims,
            })
        }
        (None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    let rstd_binding = match (&plan.rstd, &messages.rstd, rstd_table) {
        (Some(rstd_plan), Some(rstd), Some(table)) => {
            let first_rstd = first.rstd.as_ref().ok_or(Error::ShapeMismatch)?;
            let roots = crate::proof::lookup_messages::LookupRoots {
                bits: first_rstd
                    .bits
                    .message
                    .roots
                    .first()
                    .copied()
                    .ok_or(Error::ShapeMismatch)?,
                mask: first_rstd.mask.root,
            };
            check_rstd_table_view(rstd_plan, table)?;
            let seed = draw_rstd_seed(chain)?;
            let claims = crate::proof::lookup::verify_lookup(
                &rstd_plan.lookup,
                &rstd_plan.commit,
                &rstd_plan.key,
                &roots,
                &rstd.proof,
                table,
                seed,
            )?;
            if claims != rstd.claims {
                return Err(Error::InvalidProof);
            }
            Some(NormLookupSecond {
                proof: rstd.proof.clone(),
                claims,
            })
        }
        (None, None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    let instances = positions
        .checked_mul(relations.len())
        .ok_or(Error::ResourceLimit)?;
    let weights = psc_variables(variables).map_err(|_| Error::InvalidConfig)?;
    let hiding = hp_entries(&weights).map_err(|_| Error::InvalidConfig)?;
    if hiding != mask_plan.hiding {
        return Err(Error::InvalidConfig);
    }
    let entries = DotMaskEntries {
        hiding: vec![Fp3::ZERO; hiding],
        shown: [Fp3::ZERO; SHOWN_MAPS],
    };
    let claim = DotClaim {
        masks: Vec::new(),
        target: Fp3::ZERO,
    };
    let mut bindings = Vec::with_capacity(instances);
    for position in 0..positions {
        for (relation_index, relation) in relations.iter().enumerate() {
            let instance = position * relations.len() + relation_index;
            let proof = messages
                .psc
                .get(instance)
                .ok_or(Error::ShapeMismatch)?;
            let mut challenges = crate::proof::committed_psc::PositionPscChallenges::new(
                chain,
                crate::proof::hidden_norm_psc::NORM_PSC_TAGS,
                instance as u64,
            );
            let output = crate::proof::psc::verify_psc_transcript(
                &claim,
                &entries,
                proof,
                variables,
                &mut challenges,
            )?;
            let expected = [
                output.ends[0].value,
                output.ends[1].value,
            ];
            if messages
                .point_values
                .get(instance)
                .copied()
                .ok_or(Error::ShapeMismatch)?
                != expected
            {
                return Err(Error::InvalidProof);
            }
            bindings.push(NormPositionBinding {
                position: position as u32,
                relation: *relation,
                output,
            });
        }
    }
    let output_binding = match (&plan.output, &messages.output) {
        (Some(_output_plan), Some(message)) => {
            let bits = usize::try_from(plan.columns.trailing_zeros())
                .map_err(|_| Error::ResourceLimit)?;
            let hiding = crate::proof::hidden_norm_psc::norm_pointwise_masks(plan.variables)?.0;
            let mut output_bindings = Vec::with_capacity(positions * 2);
            for position in 0..positions {
                for kind in 0..2u8 {
                    let instance = position * 2 + kind as usize;
                    let point = draw_norm_output_point(chain, instance as u64, bits)?;
                    let variables = (0..bits)
                        .map(|index| crate::proof::psc::Weight::equality(point[index], 3))
                        .collect::<Result<Vec<_>>>()?;
                    let claim = *message
                        .claims
                        .get(instance)
                        .ok_or(Error::ShapeMismatch)?;
                    let proof = message
                        .proofs
                        .get(instance)
                        .ok_or(Error::ShapeMismatch)?;
                    let mut challenges =
                        crate::proof::committed_psc::PositionPscChallenges::new(
                            chain,
                            crate::proof::hidden_norm_psc::NORM_OUTPUT_PSC_TAGS,
                            instance as u64,
                        );
                    challenges.send_claim(claim)?;
                    let claim_slot = mask_plan.claim_slot(instances, instance)?;
                    let claim_masks = [crate::proof::psc_general::MaskTerm {
                        slot: claim_slot,
                        coefficient: Fp3::ONE,
                    }];
                    let checked = crate::proof::psc_general::verify_general(
                        &variables,
                        &claim_masks,
                        claim,
                        &crate::proof::psc_general::Summand::Product,
                        None,
                        proof,
                        hiding,
                        &mut challenges,
                    )?;
                    output_bindings.push(crate::proof::hidden_norm_psc::NormOutputBinding {
                        position: position as u32,
                        kind,
                        point: point.clone(),
                        claim,
                        output: checked,
                    });
                }
            }
            Some(output_bindings)
        }
        (None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    let tail_binding = match (
        plan.output.as_ref().and_then(|output| output.tail.as_ref()),
        &messages.tail,
    ) {
        (Some(tail_plan), Some(message)) => {
            let variables = crate::proof::hidden_norm_psc::output_tail_variables(plan.columns)?;
            let entries = DotMaskEntries {
                hiding: vec![Fp3::ZERO; mask_plan.tail_hiding],
                shown: [Fp3::ZERO; SHOWN_MAPS],
            };
            let claim = DotClaim {
                masks: Vec::new(),
                target: Fp3::ZERO,
            };
            let mut tail_bindings = Vec::with_capacity(positions * crate::proof::hidden_norm_psc::NORM_OUTPUT_TAIL_RELATIONS);
            for instance in 0..positions * crate::proof::hidden_norm_psc::NORM_OUTPUT_TAIL_RELATIONS {
                let proof = message.psc.get(instance).ok_or(Error::ShapeMismatch)?;
                let mut challenges =
                    crate::proof::committed_psc::PositionPscChallenges::new(
                        chain,
                        crate::proof::hidden_norm_psc::NORM_TAIL_PSC_TAGS,
                        instance as u64,
                    );
                let output = crate::proof::psc::verify_psc_transcript(
                    &claim,
                    &entries,
                    proof,
                    variables,
                    &mut challenges,
                )?;
                let expected = [output.ends[0].value, output.ends[1].value];
                if message
                    .point_values
                    .get(instance)
                    .copied()
                    .ok_or(Error::ShapeMismatch)?
                    != expected
                {
                    return Err(Error::InvalidProof);
                }
                tail_bindings.push(crate::proof::hidden_norm_psc::NormTailBinding {
                    position: (instance / crate::proof::hidden_norm_psc::NORM_OUTPUT_TAIL_RELATIONS) as u32,
                    relation: instance % crate::proof::hidden_norm_psc::NORM_OUTPUT_TAIL_RELATIONS,
                    output,
                });
            }
            let _ = tail_plan;
            Some(tail_bindings)
        }
        (None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    let bias_point = match &tail_binding {
        Some(_) => {
            let bits = usize::try_from(plan.columns.trailing_zeros())
                .map_err(|_| Error::ResourceLimit)?;
            Some(draw_norm_tail_point(
                chain,
                crate::proof::hidden_norm_psc::NORM_BIAS_POINT_TAG,
                bits,
            )?)
        }
        None => None,
    };
    Ok(NormRowBinding {
        positions: bindings,
        lookup: lookup_binding,
        rstd: rstd_binding,
        output: output_binding,
        tail: tail_binding,
        bias_point,
    })
}

/// The BAT virtual index of one lookup mask entry.
fn lookup_mask_virtual_index(
    commit: &crate::proof::lookup_messages::LookupCommitPlan,
    slot: usize,
) -> Result<u64> {
    let ell = usize::try_from(commit.mask_code.ell).map_err(|_| Error::ResourceLimit)?;
    let rows = usize::try_from(commit.mask_rows).map_err(|_| Error::ResourceLimit)?;
    if ell == 0 || slot >= rows.checked_mul(ell).ok_or(Error::ResourceLimit)? {
        return Err(Error::InvalidIndex);
    }
    let row = slot / ell;
    let place = slot % ell;
    let row_bits = ceil_log2_u64(commit.mask_rows);
    (row as u64)
        .checked_add((place as u64) << row_bits)
        .ok_or(Error::ResourceLimit)
}

/// The BAT rows of the lookup claims: the key-leaf claim on its committed
/// rounding cells, the multiplicity row on the multiplicity bits, the leaf
/// kept-map row and the bit summand's hiding and shown-end claims. Every mask
/// term becomes a unit on the committed lookup mask member.
fn lookup_plain_rows(
    plan: &HiddenLinPlan,
    lookup_plan: &dyn crate::proof::hidden_norm_psc::NormLookupView,
    members: &[ProximityMember],
    binding: &NormLookupSecond,
) -> Result<Vec<BatRow>> {
    let norm = plan.norm.as_ref().ok_or(Error::InvalidConfig)?;
    let round_plan = norm.round.as_ref().ok_or(Error::UnsupportedPrimitive)?;
    let lookup = lookup_plan.lookup_plan();
    let commit = lookup_plan.commit_plan();
    let claims = &binding.claims;
    let planes = lookup.planes()?;
    let table_rows = lookup.table_rows()?;
    let bits_columns = planes
        .checked_mul(table_rows)
        .ok_or(Error::ResourceLimit)?;
    let mut bat_of = std::collections::HashMap::new();
    for member in members {
        bat_of.insert(member.source_oracle, member);
    }
    let bits_member = bat_of
        .get(&commit.bits_oracle)
        .ok_or(Error::StatementMismatch)?;
    if bits_member.main_rows != commit.bits_shape.main_rows
        || bits_member.main_code != commit.bits_shape.code
        || bits_member.mask_rows != 0
    {
        return Err(Error::StatementMismatch);
    }
    let mask_member = bat_of
        .get(&commit.mask_oracle)
        .ok_or(Error::StatementMismatch)?;
    if mask_member.mask_rows != commit.mask_rows
        || mask_member.mask_code != commit.mask_code
        || mask_member.main_rows != 0
    {
        return Err(Error::StatementMismatch);
    }
    let mut rows = Vec::new();

    // The key-leaf claim expanded on the committed key group cells. The
    // frozen-table view keeps no group of its own and uses the rounding group;
    // the digit lookup names the rounding and tail groups.
    let groups = if lookup_plan.key_groups().is_empty() {
        vec![crate::proof::hidden_norm_psc::LookupKeyGroup {
            oracle_base: round_plan.group.oracle_base,
            shape: round_plan.group.shape,
            columns: round_plan.group.columns,
            positions: norm.shape.positions,
        }]
    } else {
        lookup_plan.key_groups().to_vec()
    };
    let mut key_members = Vec::new();
    for source in &lookup_plan.key_plan().sources {
        let group = groups
            .iter()
            .find(|group| group.covers(source.oracle))
            .ok_or(Error::StatementMismatch)?;
        let member = bat_of
            .get(&source.oracle)
            .ok_or(Error::StatementMismatch)?;
        if key_members
            .iter()
            .any(|entry: &crate::proof::lookup::KeyClaimMember| {
                entry.source_oracle == member.source_oracle
            })
        {
            continue;
        }
        key_members.push(crate::proof::lookup::KeyClaimMember {
            source_oracle: member.source_oracle,
            bat_oracle: member.bat_oracle,
            shape: group.shape.clone(),
            columns: group.columns,
        });
    }
    rows.push(lookup_plan.key_plan().registration_row(
        lookup,
        &claims.key,
        claims.theta,
        &key_members,
        mask_member.bat_oracle,
        &|slot| lookup_mask_virtual_index(commit, slot),
    )?);

    // The multiplicity row: the binary counts sum to the key count.
    let mut multiplicity = BatRow::new(BatRowKind::Plain);
    multiplicity.add_term(
        bits_member.bat_oracle,
        BatPart::Main,
        scatter_message_coefficients(
            &commit.bits_shape,
            bits_columns as u64,
            &claims.multiplicity.0,
        )?,
    );
    multiplicity.target = claims.multiplicity.1;
    rows.push(multiplicity);

    // The leaf kept-map row and the bit summand's hiding and shown-end claims
    // all land on the committed multiplicity-bit message.
    let leaf_index = (planes - 1)
        .checked_add(lookup.key_bits as usize)
        .and_then(|value| value.checked_add(lookup.table_log as usize))
        .ok_or(Error::ResourceLimit)?;
    let bits_index = leaf_index.checked_add(1).ok_or(Error::ResourceLimit)?;
    if claims.rows.len() != bits_index + 1 {
        return Err(Error::ShapeMismatch);
    }
    let leaf = &claims.rows[leaf_index];
    if leaf.row_maps.len() != 1 {
        return Err(Error::ShapeMismatch);
    }
    let (map_index, map_coefficient) = leaf.row_maps[0];
    if map_index != 0 {
        return Err(Error::ShapeMismatch);
    }
    let eq = crate::proof::output_closure::eq_vector(&leaf.point)?;
    if eq.len() != bits_columns {
        return Err(Error::ShapeMismatch);
    }
    let dense = eq
        .iter()
        .map(|value| map_coefficient.mul(value))
        .collect::<Vec<_>>();
    let mut row = BatRow::new(BatRowKind::Plain);
    row.add_term(
        bits_member.bat_oracle,
        BatPart::Main,
        scatter_message_coefficients(&commit.bits_shape, bits_columns as u64, &dense)?,
    );
    let leaf_hiding = claims
        .hiding_slots
        .get(leaf_index)
        .ok_or(Error::ShapeMismatch)?;
    if leaf_hiding.len() != leaf.hiding_form.len() {
        return Err(Error::ShapeMismatch);
    }
    for (slot, coefficient) in leaf_hiding.iter().zip(&leaf.hiding_form) {
        row.add_unit(
            mask_member.bat_oracle,
            BatPart::Mask,
            lookup_mask_virtual_index(commit, *slot)?,
            *coefficient,
        );
    }
    for term in &leaf.claim_masks {
        row.add_unit(
            mask_member.bat_oracle,
            BatPart::Mask,
            lookup_mask_virtual_index(commit, term.slot)?,
            term.coefficient,
        );
    }
    row.target = leaf.row_target;
    rows.push(row);

    let bits_output = &claims.rows[bits_index];
    let bits_hiding = claims
        .hiding_slots
        .get(bits_index)
        .ok_or(Error::ShapeMismatch)?;
    if bits_hiding.len() != bits_output.hiding_form.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut row = BatRow::new(BatRowKind::Plain);
    for (slot, coefficient) in bits_hiding.iter().zip(&bits_output.hiding_form) {
        row.add_unit(
            mask_member.bat_oracle,
            BatPart::Mask,
            lookup_mask_virtual_index(commit, *slot)?,
            *coefficient,
        );
    }
    for term in &bits_output.claim_masks {
        row.add_unit(
            mask_member.bat_oracle,
            BatPart::Mask,
            lookup_mask_virtual_index(commit, term.slot)?,
            term.coefficient,
        );
    }
    row.target = bits_output.row_target;
    rows.push(row);

    if bits_output.ends.len() != 1 {
        return Err(Error::ShapeMismatch);
    }
    let end = &bits_output.ends[0];
    let eq = crate::proof::output_closure::eq_vector(&end.point)?;
    if eq.len() != bits_columns {
        return Err(Error::ShapeMismatch);
    }
    let shown_slot = claims
        .end_slots
        .get(bits_index)
        .and_then(|slots| slots.first())
        .copied()
        .ok_or(Error::ShapeMismatch)?;
    let mut row = BatRow::new(BatRowKind::Plain);
    row.add_term(
        bits_member.bat_oracle,
        BatPart::Main,
        scatter_message_coefficients(&commit.bits_shape, bits_columns as u64, &eq)?,
    );
    row.add_unit(
        mask_member.bat_oracle,
        BatPart::Mask,
        lookup_mask_virtual_index(commit, shown_slot)?,
        end.mask_scale,
    );
    row.target = end.value;
    rows.push(row);
    Ok(rows)
}

fn scatter_message_coefficients(
    shape: &WShape,
    columns: u64,
    coefficients: &[Fp3],
) -> Result<Vec<Fp3>> {
    let rows = usize::try_from(shape.main_rows).map_err(|_| Error::ResourceLimit)?;
    let ell = usize::try_from(shape.code.ell).map_err(|_| Error::ResourceLimit)?;
    let mut values = vec![Fp3::ZERO; rows.checked_mul(ell).ok_or(Error::ResourceLimit)?];
    for (column, coefficient) in coefficients
        .iter()
        .enumerate()
        .take(usize::try_from(columns).map_err(|_| Error::ResourceLimit)?)
    {
        if coefficient.is_zero() {
            continue;
        }
        let row = column % rows;
        let place = column / rows;
        let slot = values
            .get_mut(row * ell + place)
            .ok_or(Error::ShapeMismatch)?;
        *slot = *coefficient;
    }
    to_bat_table(&values, shape.main_rows, shape.code.ell)
}

fn scalar_message_coefficients(shape: &WShape, coefficient: Fp3) -> Result<Vec<Fp3>> {
    let rows = usize::try_from(shape.main_rows).map_err(|_| Error::ResourceLimit)?;
    let ell = usize::try_from(shape.code.ell).map_err(|_| Error::ResourceLimit)?;
    let mut values = vec![Fp3::ZERO; rows.checked_mul(ell).ok_or(Error::ResourceLimit)?];
    let slot = values.first_mut().ok_or(Error::ShapeMismatch)?;
    *slot = coefficient;
    to_bat_table(&values, shape.main_rows, shape.code.ell)
}

fn push_digit_coefficients(
    start: usize,
    width: usize,
    coefficient: Fp3,
    slots: &mut Vec<(usize, Fp3)>,
) {
    let mut power = Fp3::ONE;
    for index in 0..width {
        slots.push((start + index, coefficient.mul(&power)));
        power = power.mul(&Fp3::from_u64(256));
    }
}

fn norm_plain_rows(
    plan: &HiddenLinPlan,
    members: &[ProximityMember],
    first: &HiddenSendMessages,
    binding: &NormRowBinding,
) -> Result<Vec<BatRow>> {
    let norm = plan.norm.as_ref().ok_or(Error::InvalidConfig)?;
    let mask_plan = norm.mask.as_ref().ok_or(Error::InvalidConfig)?;
    let norm_first = first.norm.as_ref().ok_or(Error::ShapeMismatch)?;
    if norm_first.mask.is_none() {
        return Err(Error::ShapeMismatch);
    }
    let positions = plan.send.positions() as usize;
    let relations = norm.relations.len();
    if binding.positions.len() != positions * relations {
        return Err(Error::ShapeMismatch);
    }
    let mut bat_of = std::collections::HashMap::new();
    for member in members {
        bat_of.insert(member.source_oracle, member);
    }
    let mask_member = bat_of
        .get(&mask_plan.oracle)
        .ok_or(Error::StatementMismatch)?;
    if mask_member.mask_rows != mask_plan.mask_rows
        || mask_member.mask_code != mask_plan.mask_code
        || mask_member.answers != mask_plan.answers
    {
        return Err(Error::StatementMismatch);
    }
    let valid = norm.shape.valid as usize;
    let columns = usize::try_from(norm.columns).map_err(|_| Error::ResourceLimit)?;
    let mut rows = Vec::with_capacity(binding.positions.len());
    for (index, word) in binding.positions.iter().enumerate() {
        let position = index / relations;
        let relation_index = index % relations;
        if word.position as usize != position
            || word.relation != norm.relations[relation_index]
            || word.output.point.len() != norm.variables
            || word
                .output
                .ends
                .iter()
                .any(|end| end.point != word.output.point)
        {
            return Err(Error::InvalidProof);
        }
        let eq = crate::proof::output_closure::eq_vector(&word.output.point)?;
        if eq.len() != 2 * columns {
            return Err(Error::ShapeMismatch);
        }
        let x_member = bat_of
            .get(
                &norm
                    .x
                    .oracle_base
                    .checked_add(position as u64)
                    .ok_or(Error::ResourceLimit)?,
            )
            .ok_or(Error::StatementMismatch)?;
        let instance = index;
        let shown0 = mask_plan.shown_slot(instance, 0)?;
        let shown1 = mask_plan.shown_slot(instance, 1)?;
        let mut x_coefficients = vec![Fp3::ZERO; 2 * columns];
        for entry in 0..valid.min(columns) {
            x_coefficients[entry] = eq[entry];
        }
        let eq_sum = eq[..valid.min(columns)]
            .iter()
            .fold(Fp3::ZERO, |sum, value| sum.add(value));
        let round_plan = norm.round.as_ref();
        let round_member = match round_plan {
            Some(plan) => Some(
                bat_of
                    .get(
                        &plan
                            .group
                            .oracle_base
                            .checked_add(position as u64)
                            .ok_or(Error::ResourceLimit)?,
                    )
                    .ok_or(Error::StatementMismatch)?,
            ),
            None => None,
        };
        let round_coefficients = |slots: &[(usize, Fp3)]| -> Result<Vec<Fp3>> {
            let plan = round_plan.ok_or(Error::UnsupportedPrimitive)?;
            let round_columns = usize::try_from(plan.group.columns)
                .map_err(|_| Error::ResourceLimit)?;
            let mut dense = vec![Fp3::ZERO; round_columns];
            for (slot, coefficient) in slots {
                let entry = dense.get_mut(*slot).ok_or(Error::ShapeMismatch)?;
                *entry = coefficient.add(entry);
            }
            scatter_message_coefficients(
                &plan.group.shape,
                plan.group.columns,
                &dense,
            )
        };
        match word.relation {
            0 => {
                let sum_member = bat_of
                    .get(
                        &norm
                            .sum
                            .oracle_base
                            .checked_add(position as u64)
                            .ok_or(Error::ResourceLimit)?,
                    )
                    .ok_or(Error::StatementMismatch)?;
                let mut row = BatRow::new(BatRowKind::Plain);
                row.add_term(
                    x_member.bat_oracle,
                    BatPart::Main,
                    scatter_message_coefficients(&norm.x.shape, norm.columns, &x_coefficients)?,
                );
                row.add_term(
                    sum_member.bat_oracle,
                    BatPart::Main,
                    scalar_message_coefficients(
                        &norm.sum.shape,
                        Fp3::ZERO.sub(&eq[columns]),
                    )?,
                );
                row.add_unit(
                    mask_member.bat_oracle,
                    BatPart::Mask,
                    mask_plan.entry_virtual_index(shown0)?,
                    word.output.ends[0].mask_scale,
                );
                row.target = word.output.ends[0].value;
                rows.push(row);

                let constant = eq_sum.add(&eq[columns]);
                let mut row = BatRow::new(BatRowKind::Plain);
                row.add_unit(
                    mask_member.bat_oracle,
                    BatPart::Mask,
                    mask_plan.entry_virtual_index(shown1)?,
                    word.output.ends[1].mask_scale,
                );
                row.target = word.output.ends[1].value.sub(&constant);
                rows.push(row);
            }
            1 => {
                let mean_member = bat_of
                    .get(
                        &norm
                            .mean
                            .oracle_base
                            .checked_add(position as u64)
                            .ok_or(Error::ResourceLimit)?,
                    )
                    .ok_or(Error::StatementMismatch)?;
                let squares_member = bat_of
                    .get(
                        &norm
                            .squares
                            .oracle_base
                            .checked_add(position as u64)
                            .ok_or(Error::ResourceLimit)?,
                    )
                    .ok_or(Error::StatementMismatch)?;
                let mut left = BatRow::new(BatRowKind::Plain);
                left.add_term(
                    x_member.bat_oracle,
                    BatPart::Main,
                    scatter_message_coefficients(&norm.x.shape, norm.columns, &x_coefficients)?,
                );
                left.add_term(
                    mean_member.bat_oracle,
                    BatPart::Main,
                    scalar_message_coefficients(
                        &norm.mean.shape,
                        Fp3::ZERO.sub(&eq_sum),
                    )?,
                );
                left.add_term(
                    squares_member.bat_oracle,
                    BatPart::Main,
                    scalar_message_coefficients(
                        &norm.squares.shape,
                        Fp3::ZERO.sub(&eq[columns]),
                    )?,
                );
                left.add_unit(
                    mask_member.bat_oracle,
                    BatPart::Mask,
                    mask_plan.entry_virtual_index(shown0)?,
                    word.output.ends[0].mask_scale,
                );
                left.target = word.output.ends[0].value;
                rows.push(left);

                let mut right = BatRow::new(BatRowKind::Plain);
                right.add_term(
                    x_member.bat_oracle,
                    BatPart::Main,
                    scatter_message_coefficients(&norm.x.shape, norm.columns, &x_coefficients)?,
                );
                right.add_term(
                    mean_member.bat_oracle,
                    BatPart::Main,
                    scalar_message_coefficients(
                        &norm.mean.shape,
                        Fp3::ZERO.sub(&eq_sum),
                    )?,
                );
                right.add_unit(
                    mask_member.bat_oracle,
                    BatPart::Mask,
                    mask_plan.entry_virtual_index(shown1)?,
                    word.output.ends[1].mask_scale,
                );
                right.target = word.output.ends[1].value.sub(&eq[columns]);
                rows.push(right);
            }
            2 => {
                let sum_member = bat_of
                    .get(
                        &norm
                            .sum
                            .oracle_base
                            .checked_add(position as u64)
                            .ok_or(Error::ResourceLimit)?,
                    )
                    .ok_or(Error::StatementMismatch)?;
                let mean_member = bat_of
                    .get(
                        &norm
                            .mean
                            .oracle_base
                            .checked_add(position as u64)
                            .ok_or(Error::ResourceLimit)?,
                    )
                    .ok_or(Error::StatementMismatch)?;
                let sum_coefficient = eq[0].sub(&eq[2]);
                let mean_coefficient = crate::proof::output_closure::fp3_signed(
                    valid as i128,
                )?
                .mul(&eq[2].sub(&eq[1]));
                let mut left = BatRow::new(BatRowKind::Plain);
                left.add_term(
                    sum_member.bat_oracle,
                    BatPart::Main,
                    scalar_message_coefficients(&norm.sum.shape, sum_coefficient)?,
                );
                left.add_term(
                    mean_member.bat_oracle,
                    BatPart::Main,
                    scalar_message_coefficients(&norm.mean.shape, mean_coefficient)?,
                );
                left.add_unit(
                    mask_member.bat_oracle,
                    BatPart::Mask,
                    mask_plan.entry_virtual_index(shown0)?,
                    word.output.ends[0].mask_scale,
                );
                left.target = word.output.ends[0].value;
                rows.push(left);

                let constant = eq[0].add(&eq[1]).add(&eq[2]);
                let mut right = BatRow::new(BatRowKind::Plain);
                right.add_unit(
                    mask_member.bat_oracle,
                    BatPart::Mask,
                    mask_plan.entry_virtual_index(shown1)?,
                    word.output.ends[1].mask_scale,
                );
                right.target = word.output.ends[1].value.sub(&constant);
                rows.push(right);
            }
            3 => {
                let squares_member = bat_of
                    .get(
                        &norm
                            .squares
                            .oracle_base
                            .checked_add(position as u64)
                            .ok_or(Error::ResourceLimit)?,
                    )
                    .ok_or(Error::StatementMismatch)?;
                let variance_member = bat_of
                    .get(
                        &norm
                            .variance
                            .oracle_base
                            .checked_add(position as u64)
                            .ok_or(Error::ResourceLimit)?,
                    )
                    .ok_or(Error::StatementMismatch)?;
                let squares_coefficient = eq[0].sub(&eq[2]);
                let variance_coefficient = crate::proof::output_closure::fp3_signed(
                    valid as i128,
                )?
                .mul(&eq[2].sub(&eq[1]));
                let mut left = BatRow::new(BatRowKind::Plain);
                left.add_term(
                    squares_member.bat_oracle,
                    BatPart::Main,
                    scalar_message_coefficients(
                        &norm.squares.shape,
                        squares_coefficient,
                    )?,
                );
                left.add_term(
                    variance_member.bat_oracle,
                    BatPart::Main,
                    scalar_message_coefficients(
                        &norm.variance.shape,
                        variance_coefficient,
                    )?,
                );
                left.add_unit(
                    mask_member.bat_oracle,
                    BatPart::Mask,
                    mask_plan.entry_virtual_index(shown0)?,
                    word.output.ends[0].mask_scale,
                );
                left.target = word.output.ends[0].value;
                rows.push(left);

                let constant = eq[0].add(&eq[1]).add(&eq[2]);
                let mut right = BatRow::new(BatRowKind::Plain);
                right.add_unit(
                    mask_member.bat_oracle,
                    BatPart::Mask,
                    mask_plan.entry_virtual_index(shown1)?,
                    word.output.ends[1].mask_scale,
                );
                right.target = word.output.ends[1].value.sub(&constant);
                rows.push(right);
            }
            4 | 5 => {
                let round_plan = round_plan.ok_or(Error::UnsupportedPrimitive)?;
                let round_member = round_member.ok_or(Error::StatementMismatch)?;
                let (offset_slot, complement_slot) = if word.relation == 4 {
                    (
                        round_plan.layout.slots.mean_offset,
                        round_plan.layout.slots.mean_complement,
                    )
                } else {
                    (
                        round_plan.layout.slots.variance_offset,
                        round_plan.layout.slots.variance_complement,
                    )
                };
                let coefficients = round_coefficients(&[
                    (offset_slot, eq[0]),
                    (complement_slot, eq[1]),
                ])?;
                let constant = crate::proof::output_closure::fp3_signed(
                    valid as i128 - 1,
                )?
                .mul(&eq[2]);
                let mut left = BatRow::new(BatRowKind::Plain);
                left.add_term(round_member.bat_oracle, BatPart::Main, coefficients);
                left.add_unit(
                    mask_member.bat_oracle,
                    BatPart::Mask,
                    mask_plan.entry_virtual_index(shown0)?,
                    word.output.ends[0].mask_scale,
                );
                left.target = word.output.ends[0].value.add(&constant);
                rows.push(left);

                let constant = eq[0].add(&eq[1]).add(&eq[2]);
                let mut right = BatRow::new(BatRowKind::Plain);
                right.add_unit(
                    mask_member.bat_oracle,
                    BatPart::Mask,
                    mask_plan.entry_virtual_index(shown1)?,
                    word.output.ends[1].mask_scale,
                );
                right.target = word.output.ends[1].value.sub(&constant);
                rows.push(right);
            }
            6 | 7 => {
                let round_plan = round_plan.ok_or(Error::UnsupportedPrimitive)?;
                let round_member = round_member.ok_or(Error::StatementMismatch)?;
                let (remainder_slot, offset_slot) = if word.relation == 6 {
                    (
                        round_plan.layout.slots.mean_remainder,
                        round_plan.layout.slots.mean_offset,
                    )
                } else {
                    (
                        round_plan.layout.slots.variance_remainder,
                        round_plan.layout.slots.variance_offset,
                    )
                };
                let coefficients = round_coefficients(&[
                    (remainder_slot, eq[0]),
                    (offset_slot, Fp3::ZERO.sub(&eq[1])),
                ])?;
                let low = -((valid / 2) as i128);
                let constant = crate::proof::output_closure::fp3_signed(low)?
                    .mul(&eq[2]);
                let mut left = BatRow::new(BatRowKind::Plain);
                left.add_term(round_member.bat_oracle, BatPart::Main, coefficients);
                left.add_unit(
                    mask_member.bat_oracle,
                    BatPart::Mask,
                    mask_plan.entry_virtual_index(shown0)?,
                    word.output.ends[0].mask_scale,
                );
                left.target = word.output.ends[0].value.add(&constant);
                rows.push(left);

                let constant = eq[0].add(&eq[1]).add(&eq[2]);
                let mut right = BatRow::new(BatRowKind::Plain);
                right.add_unit(
                    mask_member.bat_oracle,
                    BatPart::Mask,
                    mask_plan.entry_virtual_index(shown1)?,
                    word.output.ends[1].mask_scale,
                );
                right.target = word.output.ends[1].value.sub(&constant);
                rows.push(right);
            }
            8..=15 => {
                let round_plan = round_plan.ok_or(Error::UnsupportedPrimitive)?;
                let round_member = round_member.ok_or(Error::StatementMismatch)?;
                let layout = &round_plan.layout;
                let (start, width, value_slot, value_member, offset) =
                    match word.relation {
                        8 => (
                            layout.slots.sum_digits,
                            layout.widths.sum,
                            None,
                            Some((
                                bat_of
                                    .get(
                                        &norm
                                            .sum
                                            .oracle_base
                                            .checked_add(position as u64)
                                            .ok_or(Error::ResourceLimit)?,
                                    )
                                    .ok_or(Error::StatementMismatch)?,
                                &norm.sum.shape,
                            )),
                            layout.sum_offset,
                        ),
                        9 => (
                            layout.slots.squares_digits,
                            layout.widths.squares,
                            None,
                            Some((
                                bat_of
                                    .get(
                                        &norm
                                            .squares
                                            .oracle_base
                                            .checked_add(position as u64)
                                            .ok_or(Error::ResourceLimit)?,
                                    )
                                    .ok_or(Error::StatementMismatch)?,
                                &norm.squares.shape,
                            )),
                            0,
                        ),
                        10 => (
                            layout.slots.variance_digits,
                            layout.widths.variance,
                            None,
                            Some((
                                bat_of
                                    .get(
                                        &norm
                                            .variance
                                            .oracle_base
                                            .checked_add(position as u64)
                                            .ok_or(Error::ResourceLimit)?,
                                    )
                                    .ok_or(Error::StatementMismatch)?,
                                &norm.variance.shape,
                            )),
                            0,
                        ),
                        11 => (
                            layout.slots.mean_digits,
                            layout.widths.mean,
                            None,
                            Some((
                                bat_of
                                    .get(
                                        &norm
                                            .mean
                                            .oracle_base
                                            .checked_add(position as u64)
                                            .ok_or(Error::ResourceLimit)?,
                                    )
                                    .ok_or(Error::StatementMismatch)?,
                                &norm.mean.shape,
                            )),
                            layout.mean_offset,
                        ),
                        12 => (
                            layout.slots.mean_offset_digits,
                            layout.widths.offset,
                            Some(layout.slots.mean_offset),
                            None,
                            0,
                        ),
                        13 => (
                            layout.slots.mean_complement_digits,
                            layout.widths.offset,
                            Some(layout.slots.mean_complement),
                            None,
                            0,
                        ),
                        14 => (
                            layout.slots.variance_offset_digits,
                            layout.widths.offset,
                            Some(layout.slots.variance_offset),
                            None,
                            0,
                        ),
                        _ => (
                            layout.slots.variance_complement_digits,
                            layout.widths.offset,
                            Some(layout.slots.variance_complement),
                            None,
                            0,
                        ),
                    };
                if width >= columns {
                    return Err(Error::ShapeMismatch);
                }
                let round_columns = usize::try_from(round_plan.group.columns)
                    .map_err(|_| Error::ResourceLimit)?;
                let mut dense = vec![Fp3::ZERO; round_columns];
                let mut power = Fp3::ONE;
                for index in 0..width {
                    let entry = dense.get_mut(start + index).ok_or(Error::ShapeMismatch)?;
                    *entry = power.mul(&eq[index]);
                    power = power.add(&power);
                }
                if let Some(slot) = value_slot {
                    let entry = dense.get_mut(slot).ok_or(Error::ShapeMismatch)?;
                    *entry = entry.sub(&eq[columns]);
                }
                let coefficients = scatter_message_coefficients(
                    &round_plan.group.shape,
                    round_plan.group.columns,
                    &dense,
                )?;
                let mut constant = eq[..width]
                    .iter()
                    .fold(Fp3::ZERO, |sum, value| sum.add(value))
                    .add(&eq[columns]);
                let mut left = BatRow::new(BatRowKind::Plain);
                left.add_term(round_member.bat_oracle, BatPart::Main, coefficients);
                if let Some((member, shape)) = value_member {
                    left.add_term(
                        member.bat_oracle,
                        BatPart::Main,
                        scalar_message_coefficients(
                            shape,
                            Fp3::ZERO.sub(&eq[columns]),
                        )?,
                    );
                }
                if offset != 0 {
                    left.target = word.output.ends[0].value.add(
                        &crate::proof::output_closure::fp3_signed(offset as i128)?
                            .mul(&eq[width]),
                    );
                    constant = constant.add(&eq[width]);
                } else {
                    left.target = word.output.ends[0].value;
                }
                left.add_unit(
                    mask_member.bat_oracle,
                    BatPart::Mask,
                    mask_plan.entry_virtual_index(shown0)?,
                    word.output.ends[0].mask_scale,
                );
                rows.push(left);

                let mut right = BatRow::new(BatRowKind::Plain);
                right.add_unit(
                    mask_member.bat_oracle,
                    BatPart::Mask,
                    mask_plan.entry_virtual_index(shown1)?,
                    word.output.ends[1].mask_scale,
                );
                right.target = word.output.ends[1].value.sub(&constant);
                rows.push(right);
            }
            16..=25 => {
                let round_plan = round_plan.ok_or(Error::UnsupportedPrimitive)?;
                let round_member = round_member.ok_or(Error::StatementMismatch)?;
                let layout = &round_plan.layout;
                let widths = &layout.widths;
                let signed = crate::proof::output_closure::fp3_signed;
                let mut left_slots: Vec<(usize, Fp3)> = Vec::new();
                let mut right_slots: Vec<(usize, Fp3)> = Vec::new();
                let mut left_constant = Fp3::ZERO;
                let mut right_constant = Fp3::ZERO;
                let mut variance_term = None;
                match word.relation {
                    16 => {
                        let variance_member = bat_of
                            .get(
                                &norm
                                    .variance
                                    .oracle_base
                                    .checked_add(position as u64)
                                    .ok_or(Error::ResourceLimit)?,
                            )
                            .ok_or(Error::StatementMismatch)?;
                        variance_term = Some(variance_member);
                        left_constant = left_constant.add(&eq[1]);
                        push_digit_coefficients(
                            layout.slots.raw_digits,
                            widths.raw(),
                            eq[2],
                            &mut left_slots,
                        );
                        push_digit_coefficients(
                            layout.slots.raw_remainder,
                            RSQRT_RAW_REMAINDER_WIDTH,
                            eq[3],
                            &mut left_slots,
                        );
                        right_constant = right_constant
                            .add(&eq[0])
                            .add(&eq[1].mul(&signed(RSQRT_HALF as i128)?))
                            .add(&eq[2].mul(&signed(-(RSQRT_SHIFT as i128))?))
                            .add(&eq[3].mul(&signed(-1)?));
                    }
                    17 => {
                        push_digit_coefficients(
                            layout.slots.raw_remainder,
                            RSQRT_RAW_REMAINDER_WIDTH,
                            eq[0],
                            &mut left_slots,
                        );
                        push_digit_coefficients(
                            layout.slots.raw_remainder_slack,
                            RSQRT_RAW_REMAINDER_WIDTH,
                            eq[1],
                            &mut left_slots,
                        );
                        left_constant = left_constant.add(&eq[2]);
                        right_constant = eq[0]
                            .add(&eq[1])
                            .add(&eq[2].mul(&signed(-(RSQRT_SHIFT as i128 - 1))?));
                    }
                    18 => {
                        for (index, slot) in [
                            layout.slots.fine_selector,
                            layout.slots.mid_selector,
                            layout.slots.coarse_selector,
                        ]
                        .iter()
                        .enumerate()
                        {
                            left_slots.push((*slot, eq[index]));
                            right_slots.push((*slot, Fp3::ZERO.sub(&eq[index])));
                            right_constant = right_constant.add(&eq[index]);
                        }
                    }
                    19 => {
                        for (index, slot) in [
                            layout.slots.fine_selector,
                            layout.slots.mid_selector,
                            layout.slots.coarse_selector,
                        ]
                        .iter()
                        .enumerate()
                        {
                            left_slots.push((*slot, eq[index]));
                            right_constant = right_constant.add(&eq[index]);
                        }
                        left_constant = left_constant.add(&eq[3]);
                        right_constant = right_constant.sub(&eq[3]);
                    }
                    20 => {
                        let fine = layout.slots.fine_selector;
                        left_slots.push((fine, eq[0]));
                        left_slots.push((fine, eq[1]));
                        push_digit_coefficients(
                            layout.slots.fine_slack,
                            RSQRT_FINE_SLACK_WIDTH,
                            eq[2],
                            &mut left_slots,
                        );
                        push_digit_coefficients(
                            layout.slots.raw_digits,
                            widths.raw(),
                            Fp3::ZERO.sub(&eq[1]),
                            &mut right_slots,
                        );
                        right_constant = right_constant
                            .add(&eq[0].mul(&signed(RSQRT_FINE as i128 - 1)?))
                            .sub(&eq[2]);
                    }
                    21 => {
                        let middle = layout.slots.mid_selector;
                        left_slots.push((middle, eq[0]));
                        left_slots.push((middle, eq[1]));
                        push_digit_coefficients(
                            layout.slots.mid_q_digits,
                            widths.mid_q(),
                            eq[2],
                            &mut left_slots,
                        );
                        left_slots.push((layout.slots.mid_r, eq[3]));
                        push_digit_coefficients(
                            layout.slots.raw_digits,
                            widths.raw(),
                            eq[0],
                            &mut right_slots,
                        );
                        right_constant = right_constant
                            .add(&eq[1].mul(&signed(-(RSQRT_FINE as i128))?))
                            .add(&eq[2].mul(&signed(-256)?))
                            .sub(&eq[3]);
                    }
                    22 => {
                        let middle = layout.slots.mid_selector;
                        left_slots.push((middle, eq[0]));
                        left_slots.push((middle, eq[1]));
                        push_digit_coefficients(
                            layout.slots.mid_high_slack,
                            RSQRT_MID_HIGH_WIDTH,
                            eq[2],
                            &mut left_slots,
                        );
                        push_digit_coefficients(
                            layout.slots.raw_digits,
                            widths.raw(),
                            Fp3::ZERO.sub(&eq[1]),
                            &mut right_slots,
                        );
                        right_constant = right_constant
                            .add(&eq[0].mul(&signed(RSQRT_COARSE_END as i128 - 1)?))
                            .sub(&eq[2]);
                    }
                    23 => {
                        let coarse = layout.slots.coarse_selector;
                        left_slots.push((coarse, eq[0]));
                        left_slots.push((coarse, eq[1]));
                        push_digit_coefficients(
                            layout.slots.coarse_q_digits,
                            widths.coarse_q(),
                            eq[2],
                            &mut left_slots,
                        );
                        left_slots.push((layout.slots.coarse_r + 1, eq[3]));
                        left_slots.push((layout.slots.coarse_r, eq[4]));
                        push_digit_coefficients(
                            layout.slots.raw_digits,
                            widths.raw(),
                            eq[0],
                            &mut right_slots,
                        );
                        right_constant = right_constant
                            .add(&eq[1].mul(&signed(-(RSQRT_COARSE_END as i128))?))
                            .add(&eq[2].mul(&signed(-65_536)?))
                            .add(&eq[3].mul(&signed(-256)?))
                            .sub(&eq[4]);
                    }
                    24 => {
                        left_slots.push((layout.slots.index_value, eq[0]));
                        left_slots.push((layout.slots.fine_selector, eq[1]));
                        left_slots.push((layout.slots.mid_selector, eq[2]));
                        left_slots.push((layout.slots.coarse_selector, eq[3]));
                        left_slots.push((layout.slots.mid_selector, eq[4]));
                        left_slots.push((layout.slots.coarse_selector, eq[5]));
                        push_digit_coefficients(
                            layout.slots.raw_digits,
                            widths.raw(),
                            Fp3::ZERO.sub(&eq[1]),
                            &mut right_slots,
                        );
                        push_digit_coefficients(
                            layout.slots.mid_q_digits,
                            widths.mid_q(),
                            Fp3::ZERO.sub(&eq[2]),
                            &mut right_slots,
                        );
                        push_digit_coefficients(
                            layout.slots.coarse_q_digits,
                            widths.coarse_q(),
                            Fp3::ZERO.sub(&eq[3]),
                            &mut right_slots,
                        );
                        right_constant = right_constant
                            .add(&eq[0])
                            .add(&eq[4].mul(&signed(-(RSQRT_FINE as i128))?))
                            .add(&eq[5].mul(&signed(
                                -((RSQRT_FINE + RSQRT_COARSE) as i128),
                            )?));
                    }
                    _ => {
                        left_slots.push((layout.slots.index_value, eq[0]));
                        push_digit_coefficients(
                            layout.slots.index_slack,
                            RSQRT_INDEX_SLACK_WIDTH,
                            eq[1],
                            &mut left_slots,
                        );
                        left_constant = left_constant.add(&eq[2]);
                        right_constant = eq[0]
                            .add(&eq[1])
                            .add(&eq[2].mul(&signed(
                                -((round_plan.table_real_len - 1) as i128),
                            )?));
                    }
                }
                let mut left = BatRow::new(BatRowKind::Plain);
                if !left_slots.is_empty() {
                    left.add_term(
                        round_member.bat_oracle,
                        BatPart::Main,
                        round_coefficients(&left_slots)?,
                    );
                }
                if let Some(member) = variance_term {
                    if eq[0] != Fp3::ZERO {
                        left.add_term(
                            member.bat_oracle,
                            BatPart::Main,
                            scalar_message_coefficients(&norm.variance.shape, eq[0])?,
                        );
                    }
                }
                left.target = word.output.ends[0].value.sub(&left_constant);
                left.add_unit(
                    mask_member.bat_oracle,
                    BatPart::Mask,
                    mask_plan.entry_virtual_index(shown0)?,
                    word.output.ends[0].mask_scale,
                );
                rows.push(left);

                let mut right = BatRow::new(BatRowKind::Plain);
                if !right_slots.is_empty() {
                    right.add_term(
                        round_member.bat_oracle,
                        BatPart::Main,
                        round_coefficients(&right_slots)?,
                    );
                }
                right.target = word.output.ends[1].value.sub(&right_constant);
                right.add_unit(
                    mask_member.bat_oracle,
                    BatPart::Mask,
                    mask_plan.entry_virtual_index(shown1)?,
                    word.output.ends[1].mask_scale,
                );
                rows.push(right);
            }
            _ => return Err(Error::UnsupportedPrimitive),
        }
        let mut row = BatRow::new(BatRowKind::Plain);
        for (entry, coefficient) in word.output.hiding_form.iter().enumerate() {
            let slot = mask_plan.hiding_slot(instance, entry)?;
            row.add_unit(
                mask_member.bat_oracle,
                BatPart::Mask,
                mask_plan.entry_virtual_index(slot)?,
                *coefficient,
            );
        }
        row.target = word.output.row_target;
        rows.push(row);
    }
    Ok(rows)
}

fn closure_plain_rows(
    plan: &HiddenLinPlan,
    members: &[ProximityMember],
    first: &HiddenSendMessages,
    binding: &ClosureRowBinding,
) -> Result<Vec<BatRow>> {
    let closure_plan = plan.closure.as_ref().ok_or(Error::InvalidConfig)?;
    let positions = plan.send.positions() as usize;
    if binding.positions.len() != positions
        || closure_plan.winner.len() != positions
        || closure_plan.valid == 0
        || closure_plan.valid >= closure_plan.vocab
    {
        return Err(Error::ShapeMismatch);
    }
    let mut bat_of = std::collections::HashMap::new();
    for member in members {
        bat_of.insert(member.source_oracle, member);
    }
    let w_rows = usize::try_from(plan.send.w.main_rows).map_err(|_| Error::ResourceLimit)?;
    let w_ell = usize::try_from(plan.send.w.code.ell).map_err(|_| Error::ResourceLimit)?;
    let slack_rows =
        usize::try_from(closure_plan.slack.main_rows).map_err(|_| Error::ResourceLimit)?;
    let slack_ell =
        usize::try_from(closure_plan.slack.code.ell).map_err(|_| Error::ResourceLimit)?;
    let bits_rows =
        usize::try_from(closure_plan.bits.main_rows).map_err(|_| Error::ResourceLimit)?;
    let bits_ell =
        usize::try_from(closure_plan.bits.code.ell).map_err(|_| Error::ResourceLimit)?;
    let bits_columns = usize::try_from(closure_plan.bits_columns)
        .map_err(|_| Error::ResourceLimit)?;
    let k = closure_plan.boolean_variables as usize;
    let k_col = closure_plan.vocab.trailing_zeros() as usize;
    let offset = closure_plan.offset()? as i128;
    let (u_off, v_off, s_off) = closure_plan.bit_offsets()?;
    let mask = bat_of
        .get(&closure_plan.mask.oracle)
        .ok_or(Error::StatementMismatch)?;
    if mask.mask_rows != closure_plan.mask.mask_rows
        || mask.mask_code != closure_plan.mask.mask_code
        || mask.answers != closure_plan.mask.answers
    {
        return Err(Error::StatementMismatch);
    }
    let shared_r_bool = binding
        .positions
        .first()
        .map(|word| word.r_bool.as_slice())
        .unwrap_or(&[]);
    let eq_r = crate::proof::output_closure::eq_vector(shared_r_bool)?;
    let mut rows = Vec::with_capacity(positions * 7);
    for position in 0..positions {
        let word = &binding.positions[position];
        if word.r_bool.len() != k
            || word.r_bool.as_slice() != shared_r_bool
            || word.r_diff.len() != k_col
            || word.r_recon.len() != k_col
            || word.output.point.len() != k
            || word.output.ends.iter().any(|end| end.point != word.output.point)
        {
            return Err(Error::InvalidProof);
        }
        let w_member = bat_of
            .get(&plan.send.oracle(position as u32)?)
            .ok_or(Error::StatementMismatch)?;
        let slack_member = bat_of
            .get(
                &closure_plan
                    .slack_oracle_base
                    .checked_add(position as u64)
                    .ok_or(Error::ResourceLimit)?,
            )
            .ok_or(Error::StatementMismatch)?;
        let bits_member = bat_of
            .get(
                &closure_plan
                    .bits_oracle_base
                    .checked_add(position as u64)
                    .ok_or(Error::ResourceLimit)?,
            )
            .ok_or(Error::StatementMismatch)?;
        let map_member = bat_of
            .get(
                &closure_plan
                    .map_oracle_base
                    .checked_add(position as u64)
                    .ok_or(Error::ResourceLimit)?,
            )
            .ok_or(Error::StatementMismatch)?;
        let winner = closure_plan.winner[position] as usize;

        // R_diff(t): sum_j c_j S_j + sum_j c_j Y_j - C Y_a
        //   = -(M sum_{j>=V} c_j + sum_{j<a} c_j).
        let mut slack_coefficients = vec![Fp3::ZERO; slack_rows * slack_ell];
        let mut w_coefficients = vec![Fp3::ZERO; w_rows * w_ell];
        let eq_diff = crate::proof::output_closure::eq_vector(
            word.r_diff.get(..k_col).ok_or(Error::ShapeMismatch)?,
        )?;
        if eq_diff.len() != closure_plan.vocab {
            return Err(Error::ShapeMismatch);
        }
        let mut c_sum = Fp3::ZERO;
        let mut masked_sum = Fp3::ZERO;
        let mut low_sum = Fp3::ZERO;
        for column in 0..closure_plan.vocab {
            let coefficient = eq_diff[column];
            c_sum = c_sum.add(&coefficient);
            if column >= closure_plan.valid {
                masked_sum = masked_sum.add(&coefficient);
            }
            if column < winner {
                low_sum = low_sum.add(&coefficient);
            }
            slack_coefficients[column] = coefficient;
            w_coefficients[column] = coefficient;
        }
        w_coefficients[winner] = w_coefficients[winner].sub(&c_sum);
        let mut row = BatRow::new(BatRowKind::Plain);
        row.add_term(slack_member.bat_oracle, BatPart::Main, slack_coefficients);
        row.add_term(w_member.bat_oracle, BatPart::Main, w_coefficients);
        row.target = Fp3::ZERO.sub(
            &crate::proof::output_closure::fp3_signed(closure_plan.mask_value as i128)?
                .mul(&masked_sum)
                .add(&low_sum),
        );
        rows.push(row);

        // Reconstruction at one random point: u = Y + A, v = A - Y and the
        // slack bit vector equal the committed slack message.
        let eq_recon = crate::proof::output_closure::eq_vector(
            word.r_recon.get(..k_col).ok_or(Error::ShapeMismatch)?,
        )?;
        if eq_recon.len() != closure_plan.vocab {
            return Err(Error::ShapeMismatch);
        }
        for (which, target) in [(0u8, offset), (1u8, offset), (2u8, 0i128)] {
            let mut bit_coefficients = vec![Fp3::ZERO; bits_rows * bits_ell];
            let mut main_coefficients = if which < 2 {
                vec![Fp3::ZERO; w_rows * w_ell]
            } else {
                vec![Fp3::ZERO; slack_rows * slack_ell]
            };
            let (bit_offset, weight) = match which {
                0 => (u_off, -1i128),
                1 => (v_off, 1i128),
                _ => (s_off, 0i128),
            };
            let bit_count = if which < 2 {
                closure_plan.range_bits as usize + 1
            } else {
                closure_plan.slack_bits as usize
            };
            for column in 0..closure_plan.vocab {
                let eq = eq_recon[column];
                main_coefficients[column] = if which < 2 {
                    crate::proof::output_closure::fp3_signed(weight)?.mul(&eq)
                } else {
                    Fp3::ZERO.sub(&eq)
                };
                let mut power = Fp3::ONE;
                for bit in 0..bit_count {
                    let entry = closure_plan.bit_entry(column, bit_offset + bit)?;
                    bit_coefficients[entry] = eq.mul(&power);
                    power = power.add(&power);
                }
            }
            let mut row = BatRow::new(BatRowKind::Plain);
            row.add_term(bits_member.bat_oracle, BatPart::Main, bit_coefficients);
            if which < 2 {
                row.add_term(w_member.bat_oracle, BatPart::Main, main_coefficients);
            } else {
                row.add_term(slack_member.bat_oracle, BatPart::Main, main_coefficients);
            }
            row.target = crate::proof::output_closure::fp3_signed(target)?;
            rows.push(row);
        }

        let shown0 = closure_plan.mask.shown_slot(position as u32, 0)?;
        let shown1 = closure_plan.mask.shown_slot(position as u32, 1)?;
        let shown2 = closure_plan.mask.shown_slot(position as u32, 2)?;
        let shown3 = closure_plan.mask.shown_slot(position as u32, 3)?;
        let shown4 = closure_plan.mask.shown_slot(position as u32, 4)?;

        // (E_0): the committed product end value plus its shown mask.
        let eq_point = crate::proof::output_closure::eq_vector(&word.output.point)?;
        if eq_point.len() != bits_columns {
            return Err(Error::InvalidProof);
        }
        let mut coefficients = vec![Fp3::ZERO; bits_rows * bits_ell];
        coefficients[..bits_columns].copy_from_slice(&eq_point);
        let mut row = BatRow::new(BatRowKind::Plain);
        row.add_term(map_member.bat_oracle, BatPart::Main, coefficients);
        row.add_unit(
            mask.bat_oracle,
            BatPart::Mask,
            closure_plan.mask.entry_virtual_index(shown0)?,
            word.output.ends[0].mask_scale,
        );
        row.target = word.output.ends[0].value;
        rows.push(row);

        // The committed bit map at the booleanity point, one-time masked.
        let mut coefficients = vec![Fp3::ZERO; bits_rows * bits_ell];
        coefficients[..bits_columns].copy_from_slice(&eq_point);
        let mut row = BatRow::new(BatRowKind::Plain);
        row.add_term(bits_member.bat_oracle, BatPart::Main, coefficients);
        row.add_unit(
            mask.bat_oracle,
            BatPart::Mask,
            closure_plan.mask.entry_virtual_index(shown2)?,
            Fp3::ONE,
        );
        row.target = word.b_point;
        rows.push(row);

        // The virtual second map is affine in the committed bits:
        // Omega(pt) = 1 - B(pt).
        let mut row = BatRow::new(BatRowKind::Plain);
        row.add_unit(
            mask.bat_oracle,
            BatPart::Mask,
            closure_plan.mask.entry_virtual_index(shown2)?,
            Fp3::ONE,
        );
        row.add_unit(
            mask.bat_oracle,
            BatPart::Mask,
            closure_plan.mask.entry_virtual_index(shown1)?,
            word.output.ends[1].mask_scale,
        );
        row.target = word
            .output
            .ends[1]
            .value
            .sub(&Fp3::ONE)
            .add(&word.b_point);
        rows.push(row);

        // (U): the hiding coefficients against the last-round target.
        let mut row = BatRow::new(BatRowKind::Plain);
        for (index, coefficient) in word.output.hiding_form.iter().enumerate() {
            let slot = closure_plan.mask.hiding_slot(position as u32, index)?;
            row.add_unit(
                mask.bat_oracle,
                BatPart::Mask,
                closure_plan.mask.entry_virtual_index(slot)?,
                *coefficient,
            );
        }
        row.target = word.output.row_target;
        rows.push(row);

        // D = eq(r, .) B at a fresh point, checked by two public linear
        // functionals of the committed messages and one equality row.
        if word.q.len() != k {
            return Err(Error::InvalidProof);
        }
        let eq_q = crate::proof::output_closure::eq_vector(&word.q)?;
        if eq_q.len() != bits_columns || eq_r.len() != bits_columns {
            return Err(Error::InvalidProof);
        }
        let mut d_coefficients = vec![Fp3::ZERO; bits_rows * bits_ell];
        d_coefficients[..bits_columns].copy_from_slice(&eq_q);
        let mut b_coefficients = vec![Fp3::ZERO; bits_rows * bits_ell];
        for entry in 0..bits_columns {
            b_coefficients[entry] = eq_q[entry].mul(&eq_r[entry]);
        }
        let mut row = BatRow::new(BatRowKind::Plain);
        row.add_term(map_member.bat_oracle, BatPart::Main, d_coefficients);
        row.add_unit(
            mask.bat_oracle,
            BatPart::Mask,
            closure_plan.mask.entry_virtual_index(shown3)?,
            Fp3::ONE,
        );
        row.target = word.d_value;
        rows.push(row);
        let mut row = BatRow::new(BatRowKind::Plain);
        row.add_term(bits_member.bat_oracle, BatPart::Main, b_coefficients);
        row.add_unit(
            mask.bat_oracle,
            BatPart::Mask,
            closure_plan.mask.entry_virtual_index(shown4)?,
            Fp3::ONE,
        );
        row.target = word.bq_value;
        rows.push(row);
        let mut row = BatRow::new(BatRowKind::Plain);
        row.add_unit(
            mask.bat_oracle,
            BatPart::Mask,
            closure_plan.mask.entry_virtual_index(shown3)?,
            Fp3::ONE,
        );
        row.add_unit(
            mask.bat_oracle,
            BatPart::Mask,
            closure_plan.mask.entry_virtual_index(shown4)?,
            Fp3::ZERO.sub(&Fp3::ONE),
        );
        row.target = word.d_value.sub(&word.bq_value);
        rows.push(row);
    }
    Ok(rows)
}

/// The committed column message of one output group as a BAT coefficient
/// vector. Under the message layout the virtual BAT index of a column equals
/// the column number, so the coefficients are the leading entries.
fn output_message_coefficients(
    shape: &WShape,
    values: &[Fp3],
    columns: usize,
) -> Result<Vec<Fp3>> {
    let rows = usize::try_from(shape.main_rows).map_err(|_| Error::ResourceLimit)?;
    let ell = usize::try_from(shape.code.ell).map_err(|_| Error::ResourceLimit)?;
    let room = rows.checked_mul(ell).ok_or(Error::ResourceLimit)?;
    if values.len() != columns || columns > room {
        return Err(Error::ShapeMismatch);
    }
    let mut coefficients = vec![Fp3::ZERO; room];
    coefficients[..columns].copy_from_slice(values);
    Ok(coefficients)
}

/// The BAT rows of the pointwise product instances: the claim value on the
/// committed product or image row at the original column point, the two shown
/// end values on their committed maps at the transcript point (the centered
/// end on the x and mean rows), and the hiding form of every instance against
/// the committed mask entries.
fn output_plain_rows(
    plan: &HiddenLinPlan,
    members: &[ProximityMember],
    first: &HiddenSendMessages,
    binding: &[crate::proof::hidden_norm_psc::NormOutputBinding],
) -> Result<Vec<BatRow>> {
    let norm = plan.norm.as_ref().ok_or(Error::InvalidConfig)?;
    let output_plan = norm.output.as_ref().ok_or(Error::InvalidConfig)?;
    let params_plan = norm.params.as_ref().ok_or(Error::InvalidConfig)?;
    let norm_first = first.norm.as_ref().ok_or(Error::ShapeMismatch)?;
    if norm_first.output.is_none() {
        return Err(Error::ShapeMismatch);
    }
    let mask_plan = norm.mask.as_ref().ok_or(Error::InvalidConfig)?;
    let positions = norm.shape.positions as usize;
    let relations = norm.relations.len();
    let instances = positions
        .checked_mul(relations)
        .ok_or(Error::ResourceLimit)?;
    let columns = usize::try_from(norm.columns).map_err(|_| Error::ResourceLimit)?;
    let valid = norm.shape.valid as usize;
    let bits = usize::try_from(norm.columns.trailing_zeros())
        .map_err(|_| Error::ResourceLimit)?;
    if binding.len() != positions.checked_mul(2).ok_or(Error::ResourceLimit)? {
        return Err(Error::ShapeMismatch);
    }
    let mut bat_of = std::collections::HashMap::new();
    for member in members {
        bat_of.insert(member.source_oracle, member);
    }
    let mask_member = bat_of
        .get(&mask_plan.oracle)
        .ok_or(Error::StatementMismatch)?;
    if mask_member.mask_rows != mask_plan.mask_rows
        || mask_member.mask_code != mask_plan.mask_code
        || mask_member.answers != mask_plan.answers
    {
        return Err(Error::StatementMismatch);
    }
    let member_for = |oracle: u64| -> Result<&ProximityMember> {
        bat_of
            .get(&oracle)
            .copied()
            .ok_or(Error::StatementMismatch)
    };
    let gain_claim = params_plan
        .gain_claim
        .as_ref()
        .ok_or(Error::InvalidConfig)?;
    let mut rows = Vec::with_capacity(binding.len() * 4);
    for word in binding {
        let position = word.position as usize;
        if position >= positions || word.kind > 1 {
            return Err(Error::InvalidProof);
        }
        let instance = position * 2 + word.kind as usize;
        if word.point.len() != bits || word.output.point.len() != bits {
            return Err(Error::InvalidProof);
        }
        let eq_claim = crate::proof::output_closure::eq_vector(&word.point)?;
        if eq_claim.len() != columns {
            return Err(Error::ShapeMismatch);
        }
        let (claim_group, claim_oracle) = match word.kind {
            0 => (
                &output_plan.product,
                output_plan.product.oracle_base + position as u64,
            ),
            _ => (
                &output_plan.image,
                output_plan.image.oracle_base + position as u64,
            ),
        };
        let claim_member = member_for(claim_oracle)?;
        let claim_slot = mask_plan.claim_slot(instances, instance)?;
        let mut row = BatRow::new(BatRowKind::Plain);
        row.add_term(
            claim_member.bat_oracle,
            BatPart::Main,
            output_message_coefficients(&claim_group.shape, &eq_claim, columns)?,
        );
        row.add_unit(
            mask_member.bat_oracle,
            BatPart::Mask,
            mask_plan.entry_virtual_index(claim_slot)?,
            Fp3::ONE,
        );
        row.target = word.claim;
        rows.push(row);

        let shown0 = mask_plan.pointwise_shown_slot(instances, instance, 0)?;
        let shown1 = mask_plan.pointwise_shown_slot(instances, instance, 1)?;
        let end0 = &word.output.ends[0];
        let end1 = &word.output.ends[1];
        if end0.point != word.output.point
            || end1.point != word.output.point
            || end0.point.len() != bits
        {
            return Err(Error::InvalidProof);
        }
        let eq_end = crate::proof::output_closure::eq_vector(&end0.point)?;
        if eq_end.len() != columns {
            return Err(Error::ShapeMismatch);
        }
        let mut row = BatRow::new(BatRowKind::Plain);
        match word.kind {
            0 => {
                let oracle = output_plan.broadcast.oracle_base + position as u64;
                let member = member_for(oracle)?;
                row.add_term(
                    member.bat_oracle,
                    BatPart::Main,
                    output_message_coefficients(
                        &output_plan.broadcast.shape,
                        &eq_end,
                        columns,
                    )?,
                );
            }
            _ => {
                let x_oracle = norm.x.oracle_base + position as u64;
                let x_member = member_for(x_oracle)?;
                let x_rows = usize::try_from(norm.x.shape.main_rows)
                    .map_err(|_| Error::ResourceLimit)?;
                let x_ell = usize::try_from(norm.x.shape.code.ell)
                    .map_err(|_| Error::ResourceLimit)?;
                if x_rows * x_ell < columns {
                    return Err(Error::ShapeMismatch);
                }
                let mut coefficients = vec![Fp3::ZERO; x_rows * x_ell];
                let mut eq_sum = Fp3::ZERO;
                for index in 0..valid.min(columns) {
                    let value = eq_end[index];
                    eq_sum = eq_sum.add(&value);
                    coefficients[index] = value;
                }
                row.add_term(x_member.bat_oracle, BatPart::Main, coefficients);
                let mean_oracle = norm.mean.oracle_base + position as u64;
                let mean_member = member_for(mean_oracle)?;
                let mean_rows = usize::try_from(norm.mean.shape.main_rows)
                    .map_err(|_| Error::ResourceLimit)?;
                let mean_ell = usize::try_from(norm.mean.shape.code.ell)
                    .map_err(|_| Error::ResourceLimit)?;
                let mut coefficients = vec![Fp3::ZERO; mean_rows * mean_ell];
                coefficients[0] = Fp3::ZERO.sub(&eq_sum);
                row.add_term(mean_member.bat_oracle, BatPart::Main, coefficients);
            }
        }
        row.target = end0.value;
        row.add_unit(
            mask_member.bat_oracle,
            BatPart::Mask,
            mask_plan.entry_virtual_index(shown0)?,
            end0.mask_scale,
        );
        rows.push(row);

        let mut row = BatRow::new(BatRowKind::Plain);
        match word.kind {
            0 => {
                let coefficients = norm_param_eval_coefficients(
                    &gain_claim.gu,
                    norm.columns,
                    &end1.point,
                )?;
                let gain_member = member_for(gain_claim.gu.oracle)?;
                row.add_term(gain_member.bat_oracle, BatPart::Main, coefficients);
            }
            _ => {
                let oracle = output_plan.product.oracle_base + position as u64;
                let member = member_for(oracle)?;
                row.add_term(
                    member.bat_oracle,
                    BatPart::Main,
                    output_message_coefficients(&output_plan.product.shape, &eq_end, columns)?,
                );
            }
        }
        row.target = end1.value;
        row.add_unit(
            mask_member.bat_oracle,
            BatPart::Mask,
            mask_plan.entry_virtual_index(shown1)?,
            end1.mask_scale,
        );
        rows.push(row);

        let mut row = BatRow::new(BatRowKind::Plain);
        for (index, coefficient) in word.output.hiding_form.iter().enumerate() {
            let slot = mask_plan.pointwise_hiding_slot(instances, instance, index)?;
            row.add_unit(
                mask_member.bat_oracle,
                BatPart::Mask,
                mask_plan.entry_virtual_index(slot)?,
                *coefficient,
            );
        }
        for term in &word.output.claim_masks {
            row.add_unit(
                mask_member.bat_oracle,
                BatPart::Mask,
                mask_plan.entry_virtual_index(term.slot)?,
                term.coefficient,
            );
        }
        row.target = word.output.row_target;
        rows.push(row);
    }
    Ok(rows)
}

/// Adds one tail relation factor to its coefficient vector: a committed tail
/// slot with its scale, the committed image value of the term's column, or a
/// public constant.
fn add_tail_factor(
    weight: Fp3,
    factor: &crate::proof::hidden_norm_psc::TailFactor,
    tail: &mut [Fp3],
    image: &mut [Fp3],
    out: &mut [Fp3],
    column: usize,
    constant: &mut Fp3,
) -> Result<()> {
    use crate::proof::hidden_norm_psc::TailFactor;

    match factor {
        TailFactor::Slot { index, scale } => {
            let slot = tail.get_mut(*index).ok_or(Error::ShapeMismatch)?;
            *slot = slot.add(&weight.mul(scale));
        }
        TailFactor::Image => {
            let slot = image.get_mut(column).ok_or(Error::ShapeMismatch)?;
            *slot = slot.add(&weight);
        }
        TailFactor::Out => {
            let slot = out.get_mut(column).ok_or(Error::ShapeMismatch)?;
            *slot = slot.add(&weight);
        }
        TailFactor::Constant(value) => {
            *constant = constant.add(&weight.mul(value));
        }
    }
    Ok(())
}

/// The BAT rows of the tail relation instances: the two shown maps at the
/// transcript point against the committed tail slots and image rows, and the
/// hiding form against the committed tail mask entries.
fn output_tail_plain_rows(
    plan: &HiddenLinPlan,
    members: &[ProximityMember],
    first: &HiddenSendMessages,
    binding: &[crate::proof::hidden_norm_psc::NormTailBinding],
) -> Result<Vec<BatRow>> {
    use crate::proof::hidden_norm_psc::{
        output_rel_terms, output_tail_variables, NORM_OUTPUT_TAIL_PAIRS,
    };

    let norm = plan.norm.as_ref().ok_or(Error::InvalidConfig)?;
    let output_plan = norm.output.as_ref().ok_or(Error::InvalidConfig)?;
    let tail_plan = output_plan.tail.as_ref().ok_or(Error::InvalidConfig)?;
    let mask_plan = norm.mask.as_ref().ok_or(Error::InvalidConfig)?;
    let norm_first = first.norm.as_ref().ok_or(Error::ShapeMismatch)?;
    if norm_first.tail.is_none() {
        return Err(Error::ShapeMismatch);
    }
    let positions = norm.shape.positions as usize;
    let relations = norm.relations.len();
    let instances = positions
        .checked_mul(relations)
        .ok_or(Error::ResourceLimit)?;
    let columns = usize::try_from(norm.columns).map_err(|_| Error::ResourceLimit)?;
    let variables = output_tail_variables(norm.columns)?;
    let tail_rows = usize::try_from(tail_plan.group.shape.main_rows)
        .map_err(|_| Error::ResourceLimit)?;
    let tail_ell = usize::try_from(tail_plan.group.shape.code.ell)
        .map_err(|_| Error::ResourceLimit)?;
    let image_rows = usize::try_from(output_plan.image.shape.main_rows)
        .map_err(|_| Error::ResourceLimit)?;
    let image_ell = usize::try_from(output_plan.image.shape.code.ell)
        .map_err(|_| Error::ResourceLimit)?;
    let out_rows = usize::try_from(output_plan.out.shape.main_rows)
        .map_err(|_| Error::ResourceLimit)?;
    let out_ell = usize::try_from(output_plan.out.shape.code.ell)
        .map_err(|_| Error::ResourceLimit)?;
    if binding.len() != positions.checked_mul(crate::proof::hidden_norm_psc::NORM_OUTPUT_TAIL_RELATIONS).ok_or(Error::ResourceLimit)? {
        return Err(Error::ShapeMismatch);
    }
    let mut bat_of = std::collections::HashMap::new();
    for member in members {
        bat_of.insert(member.source_oracle, member);
    }
    let mask_member = bat_of
        .get(&mask_plan.oracle)
        .ok_or(Error::StatementMismatch)?;
    if mask_member.mask_rows != mask_plan.mask_rows
        || mask_member.mask_code != mask_plan.mask_code
        || mask_member.answers != mask_plan.answers
    {
        return Err(Error::StatementMismatch);
    }
    let member_for = |oracle: u64| -> Result<&ProximityMember> {
        bat_of
            .get(&oracle)
            .copied()
            .ok_or(Error::StatementMismatch)
    };
    let mut rows = Vec::with_capacity(binding.len() * 3);
    for word in binding {
        let position = word.position as usize;
        if position >= positions || word.relation >= crate::proof::hidden_norm_psc::NORM_OUTPUT_TAIL_RELATIONS {
            return Err(Error::InvalidProof);
        }
        let instance = position * crate::proof::hidden_norm_psc::NORM_OUTPUT_TAIL_RELATIONS + word.relation;
        let output = &word.output;
        if output.point.len() != variables || output.ends.len() != 2 {
            return Err(Error::InvalidProof);
        }
        let eq = crate::proof::output_closure::eq_vector(&output.point)?;
        if eq.len() != columns * NORM_OUTPUT_TAIL_PAIRS {
            return Err(Error::ShapeMismatch);
        }
        let terms = output_rel_terms(tail_plan, norm.columns, output_plan.shift, word.relation)?;
        let tail_member = member_for(tail_plan.group.oracle_base + position as u64)?;
        let image_member = member_for(output_plan.image.oracle_base + position as u64)?;
        let out_member = member_for(output_plan.out.oracle_base + position as u64)?;
        let mut tail_left = vec![Fp3::ZERO; tail_rows * tail_ell];
        let mut image_left = vec![Fp3::ZERO; image_rows * image_ell];
        let mut tail_right = vec![Fp3::ZERO; tail_rows * tail_ell];
        let mut image_right = vec![Fp3::ZERO; image_rows * image_ell];
        let mut out_left = vec![Fp3::ZERO; out_rows * out_ell];
        let mut out_right = vec![Fp3::ZERO; out_rows * out_ell];
        let mut left_constant = Fp3::ZERO;
        let mut right_constant = Fp3::ZERO;
        let mut used = vec![0usize; columns];
        for term in terms {
            if term.column >= columns {
                return Err(Error::ShapeMismatch);
            }
            let at = term.column * NORM_OUTPUT_TAIL_PAIRS + used[term.column];
            used[term.column] += 1;
            if used[term.column] > NORM_OUTPUT_TAIL_PAIRS || at >= eq.len() {
                return Err(Error::ShapeMismatch);
            }
            let weight = eq[at];
            add_tail_factor(
                weight,
                &term.left,
                &mut tail_left,
                &mut image_left,
                &mut out_left,
                term.column,
                &mut left_constant,
            )?;
            add_tail_factor(
                weight,
                &term.right,
                &mut tail_right,
                &mut image_right,
                &mut out_right,
                term.column,
                &mut right_constant,
            )?;
        }
        let shown0 = mask_plan.tail_shown_slot(instances, instance, 0)?;
        let shown1 = mask_plan.tail_shown_slot(instances, instance, 1)?;
        let end0 = &output.ends[0];
        let end1 = &output.ends[1];
        if end0.point != output.point || end1.point != output.point {
            return Err(Error::InvalidProof);
        }
        let mut row = BatRow::new(BatRowKind::Plain);
        row.add_term(tail_member.bat_oracle, BatPart::Main, tail_left);
        row.add_term(image_member.bat_oracle, BatPart::Main, image_left);
        row.add_term(out_member.bat_oracle, BatPart::Main, out_left);
        row.add_unit(
            mask_member.bat_oracle,
            BatPart::Mask,
            mask_plan.entry_virtual_index(shown0)?,
            end0.mask_scale,
        );
        row.target = end0.value.sub(&left_constant);
        rows.push(row);
        let mut row = BatRow::new(BatRowKind::Plain);
        row.add_term(tail_member.bat_oracle, BatPart::Main, tail_right);
        row.add_term(image_member.bat_oracle, BatPart::Main, image_right);
        row.add_term(out_member.bat_oracle, BatPart::Main, out_right);
        row.add_unit(
            mask_member.bat_oracle,
            BatPart::Mask,
            mask_plan.entry_virtual_index(shown1)?,
            end1.mask_scale,
        );
        row.target = end1.value.sub(&right_constant);
        rows.push(row);
        let mut row = BatRow::new(BatRowKind::Plain);
        for (index, coefficient) in output.hiding_form.iter().enumerate() {
            let slot = mask_plan.tail_hiding_slot(instances, instance, index)?;
            row.add_unit(
                mask_member.bat_oracle,
                BatPart::Mask,
                mask_plan.entry_virtual_index(slot)?,
                *coefficient,
            );
        }
        row.target = output.row_target;
        rows.push(row);
    }
    // The clamp distance rows: one per position and column binds each
    // committed distance row to pre_clamp minus low (and the high side) over
    // the same committed tail cells. The half-range constants are public.
    if let Some(clamp) = &tail_plan.clamp {
        let widths = &tail_plan.widths;
        let slots = tail_plan.slots()?;
        let pre_at = widths.quotient
            + widths.remainder
            + widths.doubled_remainder
            + widths.tie_slack
            + 1
            + widths.shifted;
        let low_at = pre_at + widths.pre_clamp;
        let high_at = low_at + clamp.low;
        let pre_half = 1u128 << (8 * widths.pre_clamp as u32 - 1);
        let low_half = 1u128 << (8 * clamp.low as u32 - 1);
        let high_half = 1u128 << (8 * clamp.high as u32 - 1);
        let (low, high) = output_plan.clamp.ok_or(Error::InvalidConfig)?;
        for position in 0..positions {
            let tail_member = member_for(tail_plan.group.oracle_base + position as u64)?;
            for column in 0..columns {
                let base = column * slots;
                for (bound_at, bound_width, bound_half, bound) in [
                    (low_at, clamp.low, low_half, low as i128),
                    (high_at, clamp.high, high_half, high as i128),
                ] {
                    let mut coefficients = vec![Fp3::ZERO; tail_rows * tail_ell];
                    let mut power = Fp3::ONE;
                    for digit in 0..bound_width {
                        let at = base + bound_at + digit;
                        coefficients[at] = coefficients[at].add(&power);
                        power = power.mul(&Fp3::from_u64(256));
                    }
                    let mut power = Fp3::ONE;
                    for digit in 0..widths.pre_clamp {
                        let at = base + pre_at + digit;
                        coefficients[at] = coefficients[at].sub(&power);
                        power = power.mul(&Fp3::from_u64(256));
                    }
                    let mut row = BatRow::new(BatRowKind::Plain);
                    row.add_term(tail_member.bat_oracle, BatPart::Main, coefficients);
                    row.target = crate::proof::output_closure::fp3_signed(
                        bound_half as i128 - pre_half as i128 - bound,
                    )?;
                    rows.push(row);
                }
            }
        }
    }
    Ok(rows)
}

/// The BAT rows of the bias addition: `pre_clamp = shifted + bias` at one
/// fresh point per position, with the bias evaluation taken from the
/// parameter member and the public half-range offsets folded into the target.
fn output_tail_bias_rows(
    plan: &HiddenLinPlan,
    members: &[ProximityMember],
    point: &[Fp3],
) -> Result<Vec<BatRow>> {
    let norm = plan.norm.as_ref().ok_or(Error::InvalidConfig)?;
    let output_plan = norm.output.as_ref().ok_or(Error::InvalidConfig)?;
    let tail_plan = output_plan.tail.as_ref().ok_or(Error::InvalidConfig)?;
    let params_plan = norm.params.as_ref().ok_or(Error::InvalidConfig)?;
    let bias_claim = params_plan
        .bias_claim
        .as_ref()
        .ok_or(Error::InvalidConfig)?;
    let positions = norm.shape.positions as usize;
    let columns = usize::try_from(norm.columns).map_err(|_| Error::ResourceLimit)?;
    let slots = tail_plan.slots()?;
    let q_at = 0usize;
    let r_at = tail_plan.widths.quotient;
    let u_at = r_at + tail_plan.widths.remainder;
    let w_at = u_at + tail_plan.widths.doubled_remainder;
    let s_at = w_at + tail_plan.widths.tie_slack;
    let sh_at = s_at + 1;
    let pre_at = sh_at + tail_plan.widths.shifted;
    let tail_rows = usize::try_from(tail_plan.group.shape.main_rows)
        .map_err(|_| Error::ResourceLimit)?;
    let tail_ell = usize::try_from(tail_plan.group.shape.code.ell)
        .map_err(|_| Error::ResourceLimit)?;
    let eq = crate::proof::output_closure::eq_vector(point)?;
    if eq.len() != columns {
        return Err(Error::ShapeMismatch);
    }
    let half_of = |width: usize| -> Result<u128> {
        if width == 0 || width > 7 {
            return Err(Error::InvalidConfig);
        }
        Ok(1u128 << (8 * width as u32 - 1))
    };
    let field = |value: u128| -> Result<Fp3> {
        Ok(Fp3::from_u64(
            u64::try_from(value).map_err(|_| Error::BoundViolation)?,
        ))
    };
    let target = field(half_of(tail_plan.widths.pre_clamp)?)?
        .sub(&field(half_of(tail_plan.widths.shifted)?)?);
    let bias_coefficients =
        norm_param_eval_coefficients(&bias_claim.gu, norm.columns, point)?;
    let negated: Vec<Fp3> = bias_coefficients
        .iter()
        .map(|value| Fp3::ZERO.sub(value))
        .collect();
    let mut bat_of = std::collections::HashMap::new();
    for member in members {
        bat_of.insert(member.source_oracle, member);
    }
    let member_for = |oracle: u64| -> Result<&ProximityMember> {
        bat_of
            .get(&oracle)
            .copied()
            .ok_or(Error::StatementMismatch)
    };
    let bias_member = member_for(bias_claim.gu.oracle)?;
    let mut rows = Vec::with_capacity(positions);
    for position in 0..positions {
        let tail_member = member_for(tail_plan.group.oracle_base + position as u64)?;
        let mut coefficients = vec![Fp3::ZERO; tail_rows * tail_ell];
        for column in 0..columns {
            let weight = eq[column];
            let base = column * slots;
            let mut power = weight;
            for digit in 0..tail_plan.widths.pre_clamp {
                let at = base + pre_at + digit;
                coefficients[at] = coefficients[at].add(&power);
                power = power.mul(&Fp3::from_u64(256));
            }
            let mut power = weight;
            for digit in 0..tail_plan.widths.shifted {
                let at = base + sh_at + digit;
                coefficients[at] = coefficients[at].sub(&power);
                power = power.mul(&Fp3::from_u64(256));
            }
        }
        let mut row = BatRow::new(BatRowKind::Plain);
        row.add_term(tail_member.bat_oracle, BatPart::Main, coefficients);
        row.add_term(bias_member.bat_oracle, BatPart::Main, negated.clone());
        row.target = target;
        rows.push(row);
    }
    Ok(rows)
}

/// The BAT rows of the padding rule: every non-valid column's pre-clamp
/// value equals the bias entry of the same column read from the committed
/// G_u message at the unit point of that column. The clamp value relation
/// then yields `out = clamp(bias)` on those columns. The valid count is the
/// trusted plan value; the rows enumerate exactly the non-valid columns.
fn output_tail_padding_rows(
    plan: &HiddenLinPlan,
    members: &[ProximityMember],
) -> Result<Vec<BatRow>> {
    let norm = plan.norm.as_ref().ok_or(Error::InvalidConfig)?;
    let output_plan = norm.output.as_ref().ok_or(Error::InvalidConfig)?;
    let tail_plan = output_plan.tail.as_ref().ok_or(Error::InvalidConfig)?;
    let params_plan = norm.params.as_ref().ok_or(Error::InvalidConfig)?;
    let bias_claim = params_plan
        .bias_claim
        .as_ref()
        .ok_or(Error::InvalidConfig)?;
    let positions = norm.shape.positions as usize;
    let columns = usize::try_from(norm.columns).map_err(|_| Error::ResourceLimit)?;
    let valid = norm.shape.valid as usize;
    if valid >= columns {
        return Ok(Vec::new());
    }
    let bits = usize::try_from(norm.columns.trailing_zeros())
        .map_err(|_| Error::ResourceLimit)?;
    let slots = tail_plan.slots()?;
    let q_at = 0usize;
    let r_at = tail_plan.widths.quotient;
    let u_at = r_at + tail_plan.widths.remainder;
    let w_at = u_at + tail_plan.widths.doubled_remainder;
    let s_at = w_at + tail_plan.widths.tie_slack;
    let sh_at = s_at + 1;
    let pre_at = sh_at + tail_plan.widths.shifted;
    let tail_rows = usize::try_from(tail_plan.group.shape.main_rows)
        .map_err(|_| Error::ResourceLimit)?;
    let tail_ell = usize::try_from(tail_plan.group.shape.code.ell)
        .map_err(|_| Error::ResourceLimit)?;
    let half_p = 1u128 << (8 * tail_plan.widths.pre_clamp as u32 - 1);
    // pre_off - bias = half_p: the digit reconstruction minus the unit-point
    // bias read equals the public half-range offset.
    let target = Fp3::from_u64(u64::try_from(half_p).map_err(|_| Error::BoundViolation)?);
    let mut bat_of = std::collections::HashMap::new();
    for member in members {
        bat_of.insert(member.source_oracle, member);
    }
    let member_for = |oracle: u64| -> Result<&ProximityMember> {
        bat_of
            .get(&oracle)
            .copied()
            .ok_or(Error::StatementMismatch)
    };
    let bias_member = member_for(bias_claim.gu.oracle)?;
    let mut rows = Vec::with_capacity(positions * (columns - valid));
    for position in 0..positions {
        let tail_member = member_for(tail_plan.group.oracle_base + position as u64)?;
        for column in valid..columns {
            if column * slots + pre_at + tail_plan.widths.pre_clamp > tail_rows * tail_ell {
                return Err(Error::ShapeMismatch);
            }
            let mut coefficients = vec![Fp3::ZERO; tail_rows * tail_ell];
            let base = column * slots + pre_at;
            let mut power = Fp3::ONE;
            for digit in 0..tail_plan.widths.pre_clamp {
                coefficients[base + digit] = power;
                power = power.mul(&Fp3::from_u64(256));
            }
            let mut unit = vec![Fp3::ZERO; bits];
            for bit in 0..bits {
                if (column >> bit) & 1 == 1 {
                    unit[bit] = Fp3::ONE;
                }
            }
            let bias_coefficients =
                norm_param_eval_coefficients(&bias_claim.gu, norm.columns, &unit)?;
            let negated: Vec<Fp3> = bias_coefficients
                .iter()
                .map(|value| Fp3::ZERO.sub(value))
                .collect();
            let mut row = BatRow::new(BatRowKind::Plain);
            row.add_term(tail_member.bat_oracle, BatPart::Main, coefficients);
            row.add_term(bias_member.bat_oracle, BatPart::Main, negated);
            row.target = target;
            rows.push(row);
        }
    }
    Ok(rows)
}

fn own_plain_rows(
    plan: &HiddenLinPlan,
    members: &[ProximityMember],
    first: &HiddenSendMessages,
    shared_point: &[Fp3],
    tilde: &[Fp3],
    dot: Option<&DotRowBinding>,
    closure: Option<&ClosureRowBinding>,
    norm: Option<&NormRowBinding>,
) -> Result<Vec<BatRow>> {
    let mut bat_of = std::collections::HashMap::new();
    for member in members {
        bat_of.insert(member.source_oracle, member);
    }
    let w_rows = usize::try_from(plan.send.w.main_rows).map_err(|_| Error::ResourceLimit)?;
    let w_ell = usize::try_from(plan.send.w.code.ell).map_err(|_| Error::ResourceLimit)?;
    let omega_rows =
        usize::try_from(plan.proximity.omega.u.main_rows).map_err(|_| Error::ResourceLimit)?;
    let omega_ell =
        usize::try_from(plan.proximity.omega.u.code.ell).map_err(|_| Error::ResourceLimit)?;
    let mut rows = Vec::with_capacity(2 * plan.send.positions() as usize);
    let eq_bits = plan.send.columns.trailing_zeros() as usize;
    for position in 0..plan.send.positions() {
        let member = bat_of
            .get(&plan.send.oracle(position)?)
            .ok_or(Error::StatementMismatch)?;
        let point = first
            .psi
            .get(position as usize)
            .and_then(|point| point.get(..eq_bits))
            .ok_or(Error::ShapeMismatch)?;
        let eq = crate::proof::output_closure::eq_vector(point)?;
        let mut coefficients = vec![Fp3::ZERO; w_rows * w_ell];
        coefficients
            .get_mut(..eq.len())
            .ok_or(Error::ShapeMismatch)?
            .copy_from_slice(&eq);
        let mut row = BatRow::new(BatRowKind::Plain);
        row.add_term(member.bat_oracle, BatPart::Main, coefficients);
        if let Some(index) = &plan.send.index_mask {
            let mask_member = bat_of.get(&index.oracle).ok_or(Error::StatementMismatch)?;
            let slot = index.entries.eval_slot(position)?;
            let virtual_index = index.entry_virtual_index(slot)?;
            row.add_unit(
                mask_member.bat_oracle,
                BatPart::Mask,
                virtual_index,
                Fp3::ONE,
            );
        }
        row.target = first.eval_targets[position as usize];
        rows.push(row);
    }
    if let Some(index) = &plan.send.index_mask {
        if tilde.len() != plan.send.positions() as usize {
            return Err(Error::ShapeMismatch);
        }
        let mask_member = bat_of.get(&index.oracle).ok_or(Error::StatementMismatch)?;
        let eq_shared = crate::proof::output_closure::eq_vector(
            shared_point.get(..eq_bits).ok_or(Error::ShapeMismatch)?,
        )?;
        for position in 0..plan.send.positions() {
            let w_member = bat_of
                .get(&plan.send.oracle(position)?)
                .ok_or(Error::StatementMismatch)?;
            let mut coefficients = vec![Fp3::ZERO; w_rows * w_ell];
            coefficients
                .get_mut(..eq_shared.len())
                .ok_or(Error::ShapeMismatch)?
                .copy_from_slice(&eq_shared);
            let mut row = BatRow::new(BatRowKind::Plain);
            row.add_term(w_member.bat_oracle, BatPart::Main, coefficients);
            let slot = index.entries.tilde_slot(position)?;
            let virtual_index = index.entry_virtual_index(slot)?;
            row.add_unit(
                mask_member.bat_oracle,
                BatPart::Mask,
                virtual_index,
                Fp3::ONE,
            );
            row.target = tilde[position as usize];
            rows.push(row);
        }
    }
    // The diagnostic Dot row is linear in the reduced row with the public
    // activation row as its coefficients. The hidden-input mode has no public
    // activation row; its masked Dot right-hand side rows above carry the
    // `tilde_t` binding instead, and the committed sumcheck rows below close
    // the identity against the committed shown maps.
    if !plan.weights.is_empty() {
        let omega = bat_of
            .get(&plan.proximity.omega.oracle)
            .ok_or(Error::StatementMismatch)?;
        let omega_len = omega_rows * omega_ell;
        let eq_shared = crate::proof::output_closure::eq_vector(
            shared_point.get(..eq_bits).ok_or(Error::ShapeMismatch)?,
        )?;
        for position in 0..plan.send.positions() {
            let w_member = bat_of
                .get(&plan.send.oracle(position)?)
                .ok_or(Error::StatementMismatch)?;
            let weights = plan
                .weights
                .get(position as usize)
                .ok_or(Error::ShapeMismatch)?;
            let mut omega_coefficients = vec![Fp3::ZERO; omega_len];
            for (row, weight) in weights.iter().enumerate() {
                let slot = omega_coefficients.get_mut(row).ok_or(Error::ShapeMismatch)?;
                *slot = *weight;
            }
            let mut w_coefficients = vec![Fp3::ZERO; w_rows * w_ell];
            let span = w_coefficients
                .get_mut(..eq_shared.len())
                .ok_or(Error::ShapeMismatch)?;
            for (slot, coefficient) in span.iter_mut().zip(&eq_shared) {
                *slot = Fp3::ZERO.sub(coefficient);
            }
            let mut row = BatRow::new(BatRowKind::Plain);
            row.add_term(omega.bat_oracle, BatPart::Main, omega_coefficients);
            row.add_term(w_member.bat_oracle, BatPart::Main, w_coefficients);
            rows.push(row);
        }
    }
    if let Some(binding) = dot {
        rows.extend(dot_plain_rows(plan, members, binding)?);
    }
    if let Some(binding) = closure {
        rows.extend(closure_plain_rows(plan, members, first, binding)?);
    }
    if let Some(binding) = norm {
        rows.extend(norm_plain_rows(plan, members, first, binding)?);
        if let Some(output) = &binding.output {
            rows.extend(output_plain_rows(plan, members, first, output)?);
        }
        if let Some(tail) = &binding.tail {
            rows.extend(output_tail_plain_rows(plan, members, first, tail)?);
        }
        if let Some(point) = &binding.bias_point {
            rows.extend(output_tail_bias_rows(plan, members, point)?);
            rows.extend(output_tail_padding_rows(plan, members)?);
        }
        let norm_plan = plan.norm.as_ref().ok_or(Error::InvalidConfig)?;
        if let Some(lookup) = &binding.lookup {
            rows.extend(lookup_plain_rows(
                plan,
                norm_plan.lookup.as_ref().ok_or(Error::InvalidConfig)?,
                members,
                lookup,
            )?);
        }
        if let Some(rstd) = &binding.rstd {
            rows.extend(lookup_plain_rows(
                plan,
                norm_plan.rstd.as_ref().ok_or(Error::InvalidConfig)?,
                members,
                rstd,
            )?);
        }
    }
    Ok(rows)
}

/// The masked Dot right-hand side: `W_t(psi*) + g_t` from the committed `W`
/// rows and the tilde mask entries. The mask makes the sent value uniform; the
/// Dot row and the own proximity test bind it to the committed messages.
fn tilde_values(
    plan: &HiddenLinPlan,
    witness: &HiddenSendWitness,
    shared_point: &[Fp3],
) -> Result<Vec<Fp3>> {
    let index = plan
        .send
        .index_mask
        .as_ref()
        .ok_or(Error::InvalidConfig)?;
    let mask_witness = witness
        .index_mask
        .as_ref()
        .ok_or(Error::ShapeMismatch)?;
    let mut tilde = Vec::with_capacity(plan.send.positions() as usize);
    for position in 0..plan.send.positions() {
        let message = witness
            .messages
            .get(position as usize)
            .ok_or(Error::ShapeMismatch)?;
        let value = eval_message(
            shared_point,
            message,
            plan.send.w.main_rows,
            plan.send.columns,
        )?;
        let mask = mask_witness.entry(index, index.entries.tilde_slot(position)?)?;
        tilde.push(value.add(&mask));
    }
    Ok(tilde)
}

fn absorb_tilde(chain: &mut Chain, tilde: &[Fp3]) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(13);
    message.push_u64(tilde.len() as u64);
    for value in tilde {
        message.push_fp3(value);
    }
    let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// Absorbs the public residual and its relation description. The binding
/// happens after `chi` and before the product-sumcheck challenges: the next
/// challenge drawn from the chain is `eps`, which multiplies the adjusted
/// target `sum_t chi^t tilde_t - R`.
fn absorb_dot_residual(chain: &mut Chain, residual: Fp3) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(DOT_RESIDUAL_MESSAGE_TAG);
    message.push_u64(DOT_RESIDUAL_RELATION);
    message.push_fp3(&residual);
    let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The committed product-sumcheck rows of the hidden Dot identity
// ---------------------------------------------------------------------------

/// The checked transcript output, the batching challenge and the public
/// residual. Both sides derive the committed `(U)`, `(E_j)` and difference
/// rows from this binding; the arithmetic module never binds a row by itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DotRowBinding {
    pub output: DotOutput,
    pub chi: Fp3,
    /// `R = sum_t chi^t (g_t - c_t)`, the target of the difference row and
    /// the value absorbed into the second-stage transcript after `chi`.
    pub residual: Fp3,
}

/// The number of product-sumcheck variables of `matrix_rows` entries.
fn dot_variables(matrix_rows: u64) -> Result<usize> {
    if matrix_rows < 2 || !matrix_rows.is_power_of_two() {
        return Err(Error::InvalidConfig);
    }
    Ok(matrix_rows.trailing_zeros() as usize)
}

/// `sum_t chi^t tilde_t`, the public target of the batched Dot claim.
fn dot_claim_target(tilde: &[Fp3], chi: Fp3) -> Result<Fp3> {
    if tilde.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let mut target = Fp3::ZERO;
    let mut power = Fp3::ONE;
    for value in tilde {
        target = target.add(&power.mul(value));
        power = power.mul(&chi);
    }
    Ok(target)
}

/// `R = sum_t chi^t (g_t - c_t)`, computed from the committed claim and tilde
/// entries. The prover reads it from the same mask witness that committed the
/// message; the verifier receives the same field element and binds it through
/// the difference row instead of recomputing it from hidden values.
fn dot_residual(
    index: &IndexMaskPlan,
    mask_witness: &IndexMaskWitness,
    chi: Fp3,
) -> Result<Fp3> {
    let mut residual = Fp3::ZERO;
    let mut power = Fp3::ONE;
    for position in 0..index.entries.positions {
        let tilde = mask_witness.entry(index, index.entries.tilde_slot(position)?)?;
        let claim = mask_witness.entry(index, index.entries.claim_slot(position)?)?;
        residual = residual.add(&power.mul(&tilde.sub(&claim)));
        power = power.mul(&chi);
    }
    Ok(residual)
}

/// `T = sum_t chi^t tilde_t - R`, the target of the masked Dot claim. This is
/// the only place the residual enters the product sumcheck: it cancels the
/// chi-weighted mask difference exactly, so the PSC still proves
/// `sum_t chi^t <X_t, Omega> = sum_t chi^t W_t(psi*)`.
fn dot_adjusted_target(tilde: &[Fp3], chi: Fp3, residual: Fp3) -> Result<Fp3> {
    Ok(dot_claim_target(tilde, chi)?.sub(&residual))
}

/// The claim masks of the Dot claim, read from the committed claim entries.
fn dot_claim_from_witness(
    index: &IndexMaskPlan,
    mask_witness: &IndexMaskWitness,
    chi: Fp3,
    target: Fp3,
) -> Result<DotClaim> {
    let mut masks = Vec::with_capacity(index.entries.positions as usize);
    let mut power = Fp3::ONE;
    for position in 0..index.entries.positions {
        let value = mask_witness.entry(index, index.entries.claim_slot(position)?)?;
        masks.push(ClaimMask {
            coefficient: power,
            value,
        });
        power = power.mul(&chi);
    }
    Ok(DotClaim { masks, target })
}

/// The hiding coefficients and end masks of the Dot claim, read from the
/// committed mask entries.
fn dot_entries_from_witness(
    index: &IndexMaskPlan,
    mask_witness: &IndexMaskWitness,
) -> Result<DotMaskEntries> {
    let hiding = index
        .entries
        .hiding_range()?
        .map(|slot| mask_witness.entry(index, slot))
        .collect::<Result<Vec<_>>>()?;
    let end_start = index.entries.end_range()?.start;
    let shown = [
        mask_witness.entry(index, end_start)?,
        mask_witness.entry(index, end_start + 1)?,
    ];
    Ok(DotMaskEntries { hiding, shown })
}

/// The chi-batched hidden values `X_chi` of the committed shown-map message.
fn dot_batched_x(
    plan: &HiddenLinPlan,
    witness: &HiddenSendWitness,
    chi: Fp3,
) -> Result<Vec<Fp3>> {
    let values = dot_witness_values(plan, witness)?;
    let hidden = values.first().map_or(0, Vec::len);
    if hidden == 0 {
        return Err(Error::ShapeMismatch);
    }
    let mut x_chi = vec![Fp3::ZERO; hidden];
    let mut power = Fp3::ONE;
    for position_values in &values {
        for (slot, value) in x_chi.iter_mut().zip(position_values) {
            *slot = slot.add(&power.mul(value));
        }
        power = power.mul(&chi);
    }
    Ok(x_chi)
}

/// The `hidden` values of every position, read out of the committed shown-map
/// message with the same entry-to-place map the message uses.
fn dot_witness_values(
    plan: &HiddenLinPlan,
    witness: &HiddenSendWitness,
) -> Result<Vec<Vec<Fp3>>> {
    if plan.send.activation_source.is_norm_hd() {
        return hd_activation_rows(plan, witness.norm.as_ref());
    }
    let dot = plan.send.dot.as_ref().ok_or(Error::InvalidConfig)?;
    let map = witness.dot.as_ref().ok_or(Error::ShapeMismatch)?;
    let rows = usize::try_from(dot.shape.main_rows).map_err(|_| Error::ResourceLimit)?;
    let places = usize::try_from(dot.shape.code.ell).map_err(|_| Error::ResourceLimit)?;
    let hidden = dot.hidden as usize;
    let positions = plan.send.positions() as usize;
    if hidden == 0
        || map.message.len() != rows
        || map.message.iter().any(|row| row.len() != places)
    {
        return Err(Error::ShapeMismatch);
    }
    let mut values = vec![vec![Fp3::ZERO; hidden]; positions];
    for (position, row_values) in values.iter_mut().enumerate() {
        for (value_at, slot) in row_values.iter_mut().enumerate() {
            let entry = position
                .checked_mul(hidden)
                .and_then(|base| base.checked_add(value_at))
                .ok_or(Error::ResourceLimit)?;
            let row = entry % rows;
            let place = entry / rows;
            *slot = *map
                .message
                .get(row)
                .and_then(|row| row.get(place))
                .ok_or(Error::ShapeMismatch)?;
        }
    }
    Ok(values)
}

/// The committed `Omega` entries, `matrix_rows` of them, read out of the flat
/// message with the message's entry-to-place map.
fn omega_message_values(plan: &HiddenLinPlan, message: &[Fp3]) -> Result<Vec<Fp3>> {
    let omega = &plan.proximity.omega;
    let rows = usize::try_from(omega.u.main_rows).map_err(|_| Error::ResourceLimit)?;
    let places = usize::try_from(omega.u.code.ell).map_err(|_| Error::ResourceLimit)?;
    let count = usize::try_from(plan.send.matrix_rows).map_err(|_| Error::ResourceLimit)?;
    if message.len() != rows * places {
        return Err(Error::ShapeMismatch);
    }
    let mut values = Vec::with_capacity(count);
    for value_at in 0..count {
        let entry = (value_at % rows)
            .checked_mul(places)
            .and_then(|base| base.checked_add(value_at / rows))
            .ok_or(Error::ResourceLimit)?;
        values.push(*message.get(entry).ok_or(Error::ShapeMismatch)?);
    }
    Ok(values)
}

/// The chain-backed challenge source of one product sumcheck. Every prover
/// message is absorbed before the challenge that tests it is drawn.
struct ChainChallenges<'a> {
    chain: &'a mut Chain,
}

impl PscChallenges for ChainChallenges<'_> {
    fn draw(&mut self) -> Result<Fp3> {
        self.chain.draw_e()
    }

    fn draw_nonzero(&mut self) -> Result<Fp3> {
        self.chain.draw_e_nonzero()
    }

    fn send_hiding_sum(&mut self, value: Fp3) -> Result<()> {
        absorb_psc_hiding_sum(self.chain, value)
    }

    fn send_round(&mut self, index: usize, round: &PscRound) -> Result<()> {
        absorb_psc_round(self.chain, index, round)
    }

    fn send_end_values(&mut self, values: &[Fp3; SHOWN_MAPS]) -> Result<()> {
        absorb_psc_end_values(self.chain, values)
    }
}

fn absorb_psc_hiding_sum(chain: &mut Chain, value: Fp3) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(16);
    message.push_fp3(&value);
    let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn absorb_psc_round(chain: &mut Chain, index: usize, round: &PscRound) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(17);
    message.push_u64(index as u64);
    message.push_u64(round.coefficients.len() as u64);
    for coefficient in &round.coefficients {
        message.push_fp3(coefficient);
    }
    let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn absorb_psc_end_values(
    chain: &mut Chain,
    values: &[Fp3; SHOWN_MAPS],
) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(HIDDEN_SEND_SCHEMA_VERSION);
    message.push_u64(18);
    message.push_u64(values.len() as u64);
    for value in values {
        message.push_fp3(value);
    }
    let prefix = message.prefix(KIND_EMPTY, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// Prover side of the hidden Dot sumcheck: draws the batching challenge after
/// the tilde values are fixed, runs the masked product sumcheck with the
/// separate mask entries and returns the transcript and the row binding.
fn prove_dot_identity(
    plan: &HiddenLinPlan,
    chain: &mut Chain,
    witness: &HiddenSendWitness,
    omega_message: &[Fp3],
    tilde: &[Fp3],
    bind_residual: bool,
) -> Result<(DotProof, DotRowBinding)> {
    let dot = plan.send.dot.as_ref().ok_or(Error::InvalidConfig)?;
    let index = plan.send.index_mask.as_ref().ok_or(Error::InvalidConfig)?;
    let mask_witness = witness.index_mask.as_ref().ok_or(Error::ShapeMismatch)?;
    let positions = plan.send.positions() as usize;
    let hidden = dot.hidden as usize;
    if hidden != plan.send.matrix_rows as usize || tilde.len() != positions {
        return Err(Error::ShapeMismatch);
    }
    let chi = chain.draw_e()?;
    let residual = dot_residual(index, mask_witness, chi)?;
    if bind_residual {
        absorb_dot_residual(chain, residual)?;
    }
    let x_chi = dot_batched_x(plan, witness, chi)?;
    let omega = omega_message_values(plan, omega_message)?;

    let target = dot_adjusted_target(tilde, chi, residual)?;
    let claim = dot_claim_from_witness(index, mask_witness, chi, target)?;
    let entries = dot_entries_from_witness(index, mask_witness)?;

    let mut challenges = ChainChallenges { chain };
    let (proof, output) =
        crate::proof::psc::prove_dot_sumcheck(&x_chi, &omega, &claim, &entries, &mut challenges)?;
    check_product_claims(&output, &claim, &entries, &x_chi, &omega)?;
    Ok((
        proof,
        DotRowBinding {
            output,
            chi,
            residual,
        },
    ))
}

/// Verifier side of the hidden Dot sumcheck: draws the same batching challenge,
/// absorbs the public residual with its relation description, and checks the
/// transcript against `sum_t chi^t tilde_t - R`. The mask entries are not
/// needed: the committed `(U)`, `(E_j)` and difference rows below carry them.
fn verify_dot_identity(
    plan: &HiddenLinPlan,
    chain: &mut Chain,
    messages: &HiddenSecondMessages,
    tilde: &[Fp3],
) -> Result<DotRowBinding> {
    let dot = plan.send.dot.as_ref().ok_or(Error::InvalidConfig)?;
    let index = plan.send.index_mask.as_ref().ok_or(Error::InvalidConfig)?;
    let positions = plan.send.positions() as usize;
    let hidden = dot.hidden as usize;
    let variables = dot_variables(plan.send.matrix_rows)?;
    if hidden != plan.send.matrix_rows as usize || tilde.len() != positions {
        return Err(Error::ShapeMismatch);
    }
    let proof = messages.dot.as_ref().ok_or(Error::ShapeMismatch)?;
    let residual = messages.residual.ok_or(Error::ShapeMismatch)?;
    let chi = chain.draw_e()?;
    absorb_dot_residual(chain, residual)?;
    let mut masks = Vec::with_capacity(positions);
    let mut power = Fp3::ONE;
    for _ in 0..positions {
        masks.push(ClaimMask {
            coefficient: power,
            value: Fp3::ZERO,
        });
        power = power.mul(&chi);
    }
    let claim = DotClaim {
        masks,
        target: dot_adjusted_target(tilde, chi, residual)?,
    };
    // The transcript check needs only the entry counts; the mask values are
    // committed and checked through the rows.
    let entries = DotMaskEntries {
        hiding: vec![Fp3::ZERO; index.entries.hiding],
        shown: [Fp3::ZERO; SHOWN_MAPS],
    };
    let mut challenges = ChainChallenges { chain };
    let output = crate::proof::psc::verify_psc_transcript(
        &claim,
        &entries,
        proof,
        variables,
        &mut challenges,
    )?;
    Ok(DotRowBinding {
        output,
        chi,
        residual,
    })
}

/// The committed rows of the hidden Dot identity: one public difference row,
/// the `(U)` row, and the two `(E_j)` rows on the committed shown maps. The
/// difference row ties the chi-weighted claim and tilde entries to the public
/// residual; it does not force the two mask values to be equal. The rows are
/// built from the checked transcript, so the arithmetic module does not
/// decide a row on its own.
fn dot_plain_rows(
    plan: &HiddenLinPlan,
    members: &[ProximityMember],
    binding: &DotRowBinding,
) -> Result<Vec<BatRow>> {
    let dot = plan.send.dot.as_ref().ok_or(Error::InvalidConfig)?;
    let index = plan.send.index_mask.as_ref().ok_or(Error::InvalidConfig)?;
    let output = &binding.output;
    let variables = dot_variables(plan.send.matrix_rows)?;
    let hidden = dot.hidden as usize;
    let positions = plan.send.positions();
    if output.point.len() != variables
        || output.ends.iter().any(|end| end.point != output.point)
        || hidden != plan.send.matrix_rows as usize
    {
        return Err(Error::InvalidProof);
    }
    let mut bat_of = std::collections::HashMap::new();
    for member in members {
        bat_of.insert(member.source_oracle, member);
    }
    let mask_member = bat_of.get(&index.oracle).ok_or(Error::StatementMismatch)?;
    // The difference row must name the mask member with the planned mask
    // batch, not a swapped object or a member with a different mask shape.
    if mask_member.mask_rows != index.mask_rows
        || mask_member.mask_code != index.mask_code
        || mask_member.answers != index.answers
    {
        return Err(Error::StatementMismatch);
    }
    let dot_member = match plan.send.activation_source {
        ActivationSource::Commitment => {
            Some(bat_of.get(&dot.oracle).ok_or(Error::StatementMismatch)?)
        }
        ActivationSource::NormHd => None,
    };
    let omega_member = bat_of
        .get(&plan.proximity.omega.oracle)
        .ok_or(Error::StatementMismatch)?;
    let mut rows = Vec::with_capacity(4);

    // One difference row: sum_t chi^t g_t - sum_t chi^t c_t = R. The claim and
    // tilde entries are independent; this row binds only their chi-weighted
    // difference, which is exactly the residual removed from the PSC target.
    // Both units name the same committed mask member and the coefficient
    // powers are recomputed here from the checked transcript challenge.
    let tilde_range = index.entries.tilde_range()?;
    let claim_range = index.entries.claim_range()?;
    if tilde_range.len() != positions as usize || claim_range.len() != positions as usize {
        return Err(Error::InvalidConfig);
    }
    let mut row = BatRow::new(BatRowKind::Plain);
    let mut power = Fp3::ONE;
    for position in 0..positions {
        let tilde_slot = index.entries.tilde_slot(position)?;
        let claim_slot = index.entries.claim_slot(position)?;
        if !tilde_range.contains(&tilde_slot)
            || !claim_range.contains(&claim_slot)
            || tilde_slot == claim_slot
        {
            return Err(Error::InvalidConfig);
        }
        row.add_unit(
            mask_member.bat_oracle,
            BatPart::Mask,
            index.entry_virtual_index(tilde_slot)?,
            power,
        );
        row.add_unit(
            mask_member.bat_oracle,
            BatPart::Mask,
            index.entry_virtual_index(claim_slot)?,
            Fp3::ZERO.sub(&power),
        );
        power = power.mul(&binding.chi);
    }
    if row.terms.len() != 0 || row.units.len() != 2 * positions as usize {
        return Err(Error::InvalidConfig);
    }
    row.target = binding.residual;
    rows.push(row);

    // (U): <hiding_form, hiding entries> + claim_scale <chi^t, c_t> = row_target.
    let hiding = index.entries.hiding_range()?;
    if output.hiding_form.len() != hiding.len() {
        return Err(Error::InvalidProof);
    }
    let mut row = BatRow::new(BatRowKind::Plain);
    for (slot, coefficient) in hiding.zip(&output.hiding_form) {
        row.add_unit(
            mask_member.bat_oracle,
            BatPart::Mask,
            index.entry_virtual_index(slot)?,
            *coefficient,
        );
    }
    let mut power = Fp3::ONE;
    for position in 0..positions {
        let coefficient = output.claim_scale.mul(&power);
        row.add_unit(
            mask_member.bat_oracle,
            BatPart::Mask,
            index
                .entry_virtual_index(index.entries.claim_slot(position)?)?,
            coefficient,
        );
        power = power.mul(&binding.chi);
    }
    row.target = output.row_target;
    rows.push(row);

    // (E_0): the batched activation map at the sumcheck point plus the first
    // end mask, against the first masked end value. A BAT term is indexed by
    // the message's virtual entry `row + place * 2^row_bits`, which is the
    // entry index of the shown-map message.
    let end_start = index.entries.end_range()?.start;
    let mut row = BatRow::new(BatRowKind::Plain);
    match plan.send.activation_source {
        ActivationSource::Commitment => {
            let dot_member = dot_member.ok_or(Error::InvalidConfig)?;
            let dot_rows =
                usize::try_from(dot.shape.main_rows).map_err(|_| Error::ResourceLimit)?;
            let dot_ell = usize::try_from(dot.shape.code.ell).map_err(|_| Error::ResourceLimit)?;
            let mut coefficients = vec![Fp3::ZERO; dot_rows * dot_ell];
            let mut power = Fp3::ONE;
            for position in 0..positions as usize {
                for value_at in 0..hidden {
                    let coefficient = power.mul(&eq_index(
                        &output.point,
                        value_at as u64,
                        variables as u32,
                    )?);
                    let entry = position
                        .checked_mul(hidden)
                        .and_then(|base| base.checked_add(value_at))
                        .ok_or(Error::ResourceLimit)?;
                    let slot = coefficients.get_mut(entry).ok_or(Error::ShapeMismatch)?;
                    *slot = slot.add(&coefficient);
                }
                power = power.mul(&binding.chi);
            }
            row.add_term(dot_member.bat_oracle, BatPart::Main, coefficients);
        }
        ActivationSource::NormHd => {
            let norm_plan = plan.norm.as_ref().ok_or(Error::InvalidConfig)?;
            let output_plan = norm_plan.output.as_ref().ok_or(Error::InvalidConfig)?;
            let group = &output_plan.out;
            let out_rows =
                usize::try_from(group.shape.main_rows).map_err(|_| Error::ResourceLimit)?;
            let out_ell = usize::try_from(group.shape.code.ell).map_err(|_| Error::ResourceLimit)?;
            let mut power = Fp3::ONE;
            for position in 0..positions as usize {
                let member = bat_of
                    .get(
                        &group
                            .oracle_base
                            .checked_add(position as u64)
                            .ok_or(Error::ResourceLimit)?,
                    )
                    .ok_or(Error::StatementMismatch)?;
                let mut coefficients = vec![Fp3::ZERO; out_rows * out_ell];
                for value_at in 0..hidden {
                    let coefficient = power.mul(&eq_index(
                        &output.point,
                        value_at as u64,
                        variables as u32,
                    )?);
                    let slot = coefficients
                        .get_mut(value_at)
                        .ok_or(Error::ShapeMismatch)?;
                    *slot = slot.add(&coefficient);
                }
                row.add_term(member.bat_oracle, BatPart::Main, coefficients);
                power = power.mul(&binding.chi);
            }
        }
    }
    row.add_unit(
        mask_member.bat_oracle,
        BatPart::Mask,
        index.entry_virtual_index(end_start)?,
        output.ends[0].mask_scale,
    );
    row.target = output.ends[0].value;
    rows.push(row);

    // (E_1): the committed Omega map at the sumcheck point plus the second
    // end mask, against the second masked end value.
    let omega_rows = usize::try_from(plan.proximity.omega.u.main_rows)
        .map_err(|_| Error::ResourceLimit)?;
    let omega_ell = usize::try_from(plan.proximity.omega.u.code.ell)
        .map_err(|_| Error::ResourceLimit)?;
    let mut coefficients = vec![Fp3::ZERO; omega_rows * omega_ell];
    for value_at in 0..hidden {
        let coefficient = eq_index(&output.point, value_at as u64, variables as u32)?;
        let slot = coefficients
            .get_mut(value_at)
            .ok_or(Error::ShapeMismatch)?;
        *slot = slot.add(&coefficient);
    }
    let mut row = BatRow::new(BatRowKind::Plain);
    row.add_term(omega_member.bat_oracle, BatPart::Main, coefficients);
    row.add_unit(
        mask_member.bat_oracle,
        BatPart::Mask,
        index.entry_virtual_index(end_start + 1)?,
        output.ends[1].mask_scale,
    );
    row.target = output.ends[1].value;
    rows.push(row);
    // Exactly the difference row, the (U) row and the two (E_j) rows. The
    // complete-row check rejects a derived-row plan that drops, duplicates or
    // reorders a row before the proximity test sees it.
    if rows.len() != 4 {
        return Err(Error::InvalidConfig);
    }
    Ok(rows)
}

fn prove_second(
    plan: &HiddenLinPlan,
    reader: &IndexReader,
    first: &HiddenSendMessages,
    first_query: &Digest,
    shared: &HiddenSharedMessages,
    shared_witness: &SharedWitness,
    witness: &HiddenSendWitness,
    shared_point: &[Fp3],
    seed: &Digest,
    bind_residual: bool,
    rstd_table: Option<&dyn crate::proof::lookup::LookupTable>,
) -> Result<OwnProveResult> {
    let _profile_second = zkie_engine::profile::Scope::new("second.prove_second");
    let _profile_own_members = zkie_engine::profile::Scope::new("second.own_members");
    let (members, order) = own_members(
        plan,
        first,
        shared,
        Some(witness),
        Some((&shared_witness.omega_message, &shared_witness.omega_tails)),
    )?;
    drop(_profile_own_members);
    let _profile_tilde = zkie_engine::profile::Scope::light("second.tilde");
    let tilde = match &plan.send.index_mask {
        Some(_) => tilde_values(plan, witness, shared_point)?,
        None => Vec::new(),
    };
    drop(_profile_tilde);
    let mut chain = second_chain(
        plan,
        reader,
        first_query,
        &shared.seal,
        &shared.omega_root,
        shared_point,
        &shared.zeta,
    )?;
    if !tilde.is_empty() {
        absorb_tilde(&mut chain, &tilde)?;
    }
    let (dot_proof, dot_binding) = match &plan.send.dot {
        Some(_) => {
            let _profile_dot = zkie_engine::profile::Scope::new("second.dot_identity");
            let (proof, binding) = prove_dot_identity(
                plan,
                &mut chain,
                witness,
                &shared_witness.omega_message,
                &tilde,
                bind_residual,
            )?;
            drop(_profile_dot);
            (Some(proof), Some(binding))
        }
        None => (None, None),
    };
    let (closure_proof, closure_binding) = match (&plan.closure, &witness.closure) {
        (Some(closure_plan), Some(closure)) => {
            let (proofs, binding) = prove_closure_second(
                &mut chain,
                closure_plan,
                &closure.bit_values,
                &closure.map_values,
                &closure.mask_entries,
                &closure.r_bool,
            )?;
            (Some(proofs), Some(binding))
        }
        (None, None) => (None, None),
        _ => return Err(Error::ShapeMismatch),
    };
    let (norm_proof, norm_binding) = match (
        plan.norm.as_ref(),
        witness.norm.as_ref(),
        first.norm.as_ref(),
    ) {
        (Some(norm_plan), Some(norm_witness), Some(first_norm)) => {
            let (proof, binding) =
                prove_norm_second(&mut chain, norm_plan, first_norm, norm_witness, rstd_table)?;
            (Some(proof), Some(binding))
        }
        (None, None, None) => (None, None),
        _ => return Err(Error::ShapeMismatch),
    };
    let _profile_sample_rows = zkie_engine::profile::Scope::new("second.sample_rows");
    let sample_rows = own_sample_rows(plan, &members, first, shared)?;
    drop(_profile_sample_rows);
    let _profile_plain_rows = zkie_engine::profile::Scope::new("second.plain_rows");
    let plain_rows = own_plain_rows(
        plan,
        &members,
        first,
        shared_point,
        &tilde,
        dot_binding.as_ref(),
        closure_binding.as_ref(),
        norm_binding.as_ref(),
    )?;
    drop(_profile_plain_rows);
    let fold = member_fold_shape(
        plan.proximity.code,
        plan.proximity.mask_code,
        &members,
        plan.proximity.own_mask_tree.leaf_class,
    )?;
    let iopp = proximity_iopp_plan(
        &plan.proximity,
        &plan.send,
        plan.closure.as_ref(),
        plan.norm.as_ref(),
        ProximityGroupRole::Own,
    )?;
    for (i, m) in members.iter().enumerate() {
    }
    let _profile_proximity = zkie_engine::profile::Scope::new("second.proximity");
    let proximity = proximity_prove(
        chain,
        fold,
        members,
        ProximityTreeSource::Retained {
            witness,
            shared_witness,
            first,
            plan,
        },
        &sample_rows,
        plain_rows,
        plan.proximity.own_fresh,
        plan.proximity.own_mask_tree,
        derive_run_seed(seed, 5, 0)?,
        plan.proximity.own_base_positions,
        iopp.as_ref(),
    )?;
    drop(_profile_proximity);
    let mut claims = sample_rows.into_iter().flatten().collect::<Vec<_>>();
    let _ = order;
    Ok(OwnProveResult {
        messages: HiddenSecondMessages {
            omega_root: shared.omega_root,
            first_query: *first_query,
            shared_seal: shared.seal,
            tilde,
            residual: dot_binding.as_ref().map(|binding| binding.residual),
            dot: dot_proof,
            closure: closure_proof,
            norm: norm_proof,
            fold_start: proximity.messages.fold_start.clone(),
            fold_rounds: proximity.messages.fold_rounds.clone(),
            tail: proximity.messages.tail,
            seal: proximity.seal,
        },
        claims: std::mem::take(&mut claims),
    })
}

fn witness_tree_for(
    member: &ProximityMember,
    witness: &HiddenSendWitness,
    shared_witness: &SharedWitness,
    first: &HiddenSendMessages,
    plan: &HiddenLinPlan,
) -> Result<Arc<RuntimeTree>> {
    if let Some(norm_plan) = &plan.norm {
        let norm_witness = witness.norm.as_ref().ok_or(Error::ShapeMismatch)?;
        let groups = norm_group_plans(norm_plan);
        let group_witnesses = norm_group_witnesses(norm_witness);
        let positions = plan.send.positions() as u64;
        for (group_plan, group_witness) in groups.iter().zip(group_witnesses.iter()) {
            let base = group_plan.oracle_base;
            if member.source_oracle >= base
                && member.source_oracle < base.saturating_add(positions)
            {
                let position = (member.source_oracle - base) as usize;
                return retained_member_tree(
                    &group_witness.committed.trees,
                    position,
                    member,
                );
            }
        }
        if let Some(output_plan) = &norm_plan.output {
            let output_witness = norm_witness.output.as_ref().ok_or(Error::ShapeMismatch)?;
            let outputs = [
                (&output_plan.broadcast, &output_witness.broadcast),
                (&output_plan.product, &output_witness.product),
                (&output_plan.image, &output_witness.image),
                (&output_plan.out, &output_witness.out),
            ];
            for (group_plan, group_witness) in outputs {
                let base = group_plan.oracle_base;
                if member.source_oracle >= base
                    && member.source_oracle < base.saturating_add(positions)
                {
                    let position = (member.source_oracle - base) as usize;
                    return retained_member_tree(
                        &group_witness.committed.trees,
                        position,
                        member,
                    );
                }
            }
            if let Some(tail_plan) = norm_plan
                .output
                .as_ref()
                .and_then(|output| output.tail.as_ref())
            {
                let tail_witness = output_witness.tail.as_ref().ok_or(Error::ShapeMismatch)?;
                let base = tail_plan.group.oracle_base;
                if member.source_oracle >= base
                    && member.source_oracle < base.saturating_add(positions)
                {
                    let position = (member.source_oracle - base) as usize;
                    return retained_member_tree(
                        &tail_witness.committed.committed.trees,
                        position,
                        member,
                    );
                }
            }
        }
        if let Some(mask_plan) = &norm_plan.mask {
            if member.source_oracle == mask_plan.oracle {
                let mask = norm_witness.mask.as_ref().ok_or(Error::ShapeMismatch)?;
                return Ok(Arc::new(mask.committed.tree(mask_plan)?));
            }
        }
        if let Some(lookup_plan) = &norm_plan.lookup {
            let commit = &lookup_plan.commit;
            if member.source_oracle == commit.bits_oracle
                || member.source_oracle == commit.mask_oracle
            {
                let messages = norm_witness
                    .lookup
                    .as_ref()
                    .and_then(|lookup| lookup.messages.as_ref())
                    .ok_or(Error::ShapeMismatch)?;
                if member.source_oracle == commit.bits_oracle {
                    let tree = messages.bits.trees.get(0).ok_or(Error::ShapeMismatch)?;
                    if tree.plan() != &member.tree_plan || *tree.root() != member.root {
                        return Err(Error::StatementMismatch);
                    }
                    return Ok(tree.clone());
                }
                let tree = messages.mask_tree(commit)?;
                if tree.plan() != &member.tree_plan || *tree.root() != member.root {
                    return Err(Error::StatementMismatch);
                }
                return Ok(Arc::new(tree));
            }
        }
        // frozen-table lookup: retained trees
        if let Some(lookup_plan) = &norm_plan.rstd {
            let commit = &lookup_plan.commit;
            if member.source_oracle == commit.bits_oracle
                || member.source_oracle == commit.mask_oracle
            {
                let messages = norm_witness
                    .rstd
                    .as_ref()
                    .and_then(|lookup| lookup.messages.as_ref())
                    .ok_or(Error::ShapeMismatch)?;
                if member.source_oracle == commit.bits_oracle {
                    let tree = messages.bits.trees.get(0).ok_or(Error::ShapeMismatch)?;
                    if tree.plan() != &member.tree_plan || *tree.root() != member.root {
                        return Err(Error::StatementMismatch);
                    }
                    return Ok(tree.clone());
                }
                let tree = messages.mask_tree(commit)?;
                if tree.plan() != &member.tree_plan || *tree.root() != member.root {
                    return Err(Error::StatementMismatch);
                }
                return Ok(Arc::new(tree));
            }
        }
        for input in norm_param_claim_inputs(plan, first, Some(norm_witness))? {
            if member.source_oracle == input.claim.gu.oracle {
                let gu = &input.claim.gu;
                let material = input.material.as_ref().ok_or(Error::ShapeMismatch)?;
                let tree = commit_rows_tree(
                    gu.tree.clone(),
                    gu.u.code,
                    &material.message,
                    &material.tails,
                    &material.salts,
                )?;
                if tree.plan() != &member.tree_plan || *tree.root() != member.root {
                    return Err(Error::StatementMismatch);
                }
                return Ok(Arc::new(tree));
            }
        }
    }
    if let Some(closure_plan) = &plan.closure {
        let positions = plan.send.positions() as u64;
        let closure = witness.closure.as_ref().ok_or(Error::ShapeMismatch)?;
        if member.source_oracle >= closure_plan.slack_oracle_base
            && member.source_oracle
                < closure_plan.slack_oracle_base.saturating_add(positions)
        {
            let position = (member.source_oracle - closure_plan.slack_oracle_base) as usize;
            return retained_member_tree(&closure.slack.trees, position, member);
        }
        if member.source_oracle >= closure_plan.bits_oracle_base
            && member.source_oracle
                < closure_plan.bits_oracle_base.saturating_add(positions)
        {
            let position = (member.source_oracle - closure_plan.bits_oracle_base) as usize;
            return retained_member_tree(&closure.bits.trees, position, member);
        }
        if member.source_oracle >= closure_plan.map_oracle_base
            && member.source_oracle
                < closure_plan.map_oracle_base.saturating_add(positions)
        {
            let position = (member.source_oracle - closure_plan.map_oracle_base) as usize;
            return retained_member_tree(&closure.map.trees, position, member);
        }
        if member.source_oracle == closure_plan.mask.oracle {
            let _profile_tree = zkie_engine::profile::Scope::light("witness_tree.mask");
            return Ok(Arc::new(closure.mask.tree(&closure_plan.mask)?));
        }
    }
    if let Some(index) = &plan.send.index_mask {
        if member.source_oracle == index.oracle {
            let mask_witness = witness
                .index_mask
                .as_ref()
                .ok_or(Error::ShapeMismatch)?;
            let _profile_tree = zkie_engine::profile::Scope::light("witness_tree.index_mask");
            return Ok(Arc::new(mask_witness.tree(index)?));
        }
    }
    if let Some(dot) = plan.send.dot.as_ref().filter(|_| !plan.send.activation_source.is_norm_hd()) {
        if member.source_oracle == dot.oracle {
            let map = witness.dot.as_ref().ok_or(Error::ShapeMismatch)?;
            let _profile_tree = zkie_engine::profile::Scope::light("witness_tree.dot");
            return Ok(Arc::new(map.tree(dot)?));
        }
    }
    if member.source_oracle == plan.proximity.omega.oracle {
        let _profile_tree = zkie_engine::profile::Scope::light("witness_tree.omega");
        return Ok(Arc::new(commit_rows_tree(
            member.tree_plan.clone(),
            member.main_code,
            &member.message,
            &member.tails,
            &omega_salts_of(shared_witness)?,
        )?));
    }
    let position = (member.source_oracle - plan.send.oracle_base) as usize;
    if position >= plan.send.positions() as usize {
        return Err(Error::StatementMismatch);
    }
    retained_member_tree(&witness.trees, position, member)
}

fn retained_member_tree(
    trees: &RetainedTrees,
    position: usize,
    member: &ProximityMember,
) -> Result<Arc<RuntimeTree>> {
    let tree = trees.get(position).ok_or(Error::ShapeMismatch)?;
    if tree.plan() != &member.tree_plan {
        return Err(Error::StatementMismatch);
    }
    Ok(Arc::clone(tree))
}

fn omega_salts_of(witness: &SharedWitness) -> Result<Vec<Vec<u8>>> {
    if witness.omega_salts.is_empty() {
        return Err(Error::UnsupportedPrimitive);
    }
    Ok(witness.omega_salts.clone())
}

fn verify_second(
    plan: &HiddenLinPlan,
    reader: &IndexReader,
    first: &HiddenSendMessages,
    first_query: &Digest,
    shared: &HiddenSharedMessages,
    shared_point: &[Fp3],
    messages: &HiddenSecondMessages,
    closure_r: Option<&[Fp3]>,
    rstd_table: Option<&dyn crate::proof::lookup::LookupTable>,
) -> Result<(
    Digest,
    Vec<GuRelationRow>,
    ProximityBaseOutcome,
    Option<DotRowBinding>,
    Option<ClosureRowBinding>,
    Option<Vec<crate::proof::hidden_norm_psc::NormOutputBinding>>,
)> {
    let _profile = zkie_engine::profile::Scope::new("verify.second");
    if messages.omega_root != shared.omega_root
        || messages.first_query != *first_query
        || messages.shared_seal != shared.seal
    {
        return Err(Error::StatementMismatch);
    }
    match (&plan.send.index_mask, &plan.send.dot) {
        (Some(_), Some(_)) => {
            if messages.tilde.len() != plan.send.positions() as usize {
                return Err(Error::ShapeMismatch);
            }
            if messages.dot.is_none() || messages.residual.is_none() {
                return Err(Error::ShapeMismatch);
            }
        }
        (None, None) => {
            if !messages.tilde.is_empty()
                || messages.dot.is_some()
                || messages.residual.is_some()
            {
                return Err(Error::ShapeMismatch);
            }
        }
        _ => return Err(Error::InvalidConfig),
    }
    let _profile_members = zkie_engine::profile::Scope::new("verify.second.members");
    let (members, _order) = own_members(plan, first, shared, None, None)?;
    drop(_profile_members);
    let _profile_sample_rows = zkie_engine::profile::Scope::new("verify.second.sample_rows");
    let sample_rows = own_sample_rows(plan, &members, first, shared)?;
    drop(_profile_sample_rows);
    let mut chain = second_chain(
        plan,
        reader,
        first_query,
        &shared.seal,
        &shared.omega_root,
        shared_point,
        &shared.zeta,
    )?;
    if !messages.tilde.is_empty() {
        absorb_tilde(&mut chain, &messages.tilde)?;
    }
    let _profile_dot = zkie_engine::profile::Scope::new("verify.second.dot");
    let dot_binding = match &plan.send.dot {
        Some(_) => Some(verify_dot_identity(
            plan,
            &mut chain,
            messages,
            &messages.tilde,
        )?),
        None => None,
    };
    drop(_profile_dot);
    let _profile_closure = zkie_engine::profile::Scope::new("verify.second.closure");
    let closure_binding = match (&plan.closure, &messages.closure, closure_r) {
        (Some(closure_plan), Some(closure_messages), Some(r_bool)) => {
            Some(verify_closure_second(
                &mut chain,
                closure_plan,
                closure_messages,
                r_bool,
            )?)
        }
        (None, None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    drop(_profile_closure);
    let norm_binding = match (&plan.norm, &messages.norm, &first.norm) {
        (Some(norm_plan), Some(norm_messages), Some(first_norm)) => Some(
            verify_norm_second(
                &mut chain,
                norm_plan,
                first_norm,
                norm_messages,
                rstd_table,
            )?,
        ),
        (None, None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    let _profile_plain_rows = zkie_engine::profile::Scope::new("verify.second.plain_rows");
    let plain_rows = own_plain_rows(
        plan,
        &members,
        first,
        shared_point,
        &messages.tilde,
        dot_binding.as_ref(),
        closure_binding.as_ref(),
        norm_binding.as_ref(),
    )?;
    drop(_profile_plain_rows);
    let fold = member_fold_shape(
        plan.proximity.code,
        plan.proximity.mask_code,
        &members,
        plan.proximity.own_mask_tree.leaf_class,
    )?;
    let iopp = proximity_iopp_plan(
        &plan.proximity,
        &plan.send,
        plan.closure.as_ref(),
        plan.norm.as_ref(),
        ProximityGroupRole::Own,
    )?;
    let _profile_proximity = zkie_engine::profile::Scope::new("verify.second.proximity");
    let proximity = proximity_verify(
        chain,
        fold,
        &members,
        &sample_rows,
        plain_rows,
        &ProximityMessages {
            fold_start: messages.fold_start.clone(),
            fold_rounds: messages.fold_rounds.clone(),
            tail: messages.tail.clone(),
        },
        plan.proximity.own_fresh,
        plan.proximity.own_mask_tree,
        plan.proximity.own_base_positions,
        iopp.as_ref(),
    )?;
    drop(_profile_proximity);
    if proximity.seal != messages.seal {
        return Err(Error::InvalidProof);
    }
    Ok((
        proximity.seal,
        sample_rows.into_iter().flatten().collect(),
        proximity.base,
        dot_binding,
        closure_binding,
        norm_binding.as_ref().and_then(|binding| binding.output.clone()),
    ))
}


// ---------------------------------------------------------------------------
// Public entries
// ---------------------------------------------------------------------------

fn prove_hidden_lin_inner(
    plan: &HiddenLinPlan,
    activation: Option<&CommittedActivation>,
    table: &mut dyn HiddenTableProver,
    output: Option<&[Vec<Fp3>]>,
    seed: Digest,
    bind_residual: bool,
    norm: Option<&NormWitness>,
    rstd_table: Option<&dyn crate::proof::lookup::LookupTable>,
    param_tables: Option<&mut dyn NormParamTableProver>,
) -> Result<(HiddenLinMessages, HiddenLinWitness)> {
    let _profile_session = zkie_engine::profile::Scope::new("session.prove_hidden_lin_inner");
    plan.validate()?;
    if seed == Digest::ZERO {
        return Err(Error::InvalidConfig);
    }
    let reader = table.reader().clone();
    let mut coins = ByteMaskCoins::new(derive_run_seed(&seed, 1, 0)?)?;
    let _profile_randomness = zkie_engine::profile::Scope::light("session.w_commit_randomness");
    let randomness = WCommitRandomness::from_coins(
        &mut coins,
        &plan.send,
        plan.closure.as_ref(),
    )?;
    drop(_profile_randomness);
    let (first, first_seal, send_witness) =
        prove_first(
            plan,
            activation,
            table,
            output,
            &randomness,
            norm,
            &seed,
            param_tables,
        )?;
    let first_query = *first_chain(plan, &reader)?.value();
    let shared = prove_shared(
        plan,
        &reader,
        table,
        &first,
        &first_seal,
        &seed,
        send_witness.norm.as_ref(),
    )?;
    let own = prove_second(
        plan,
        &reader,
        &first,
        &first_query,
        &shared.messages,
        &shared.witness,
        &send_witness,
        &shared.points.psi,
        &seed,
        bind_residual,
        rstd_table,
    )?;
    let messages = HiddenLinMessages {
        first: first.clone(),
        first_seal,
        shared: shared.messages,
        second: own.messages,
    };
    let witness = HiddenLinWitness {
        send: send_witness,
        gu_message: shared.witness.gu_message.concat(),
        gu_tail: shared.witness.gu_tails.concat(),
        omega_message: shared.witness.omega_message,
        omega_tail: shared.witness.omega_tails.concat(),
    };
    let _ = (&mut coins, own.claims);
    Ok((messages, witness))
}

/// Prover entry of the merged three-stage run with the public output rows as a
/// diagnostic plan value. The activation commitment is an in-memory witness
/// only; the public message carries its root. This entry is the diagnostic
/// path: the output rows are plaintext caller values. The hidden-input entry
/// below is the formal path.
pub fn prove_hidden_lin(
    plan: &HiddenLinPlan,
    activation: &CommittedActivation,
    table: &mut dyn HiddenTableProver,
    output: &[Vec<Fp3>],
    seed: Digest,
) -> Result<(HiddenLinMessages, HiddenLinWitness)> {
    if plan.weights.is_empty() {
        return Err(Error::InvalidConfig);
    }
    prove_hidden_lin_inner(plan, Some(activation), table, Some(output), seed, true, None, None, None)
}

/// Prover entry of the merged three-stage run in the hidden-input mode. The
/// activation rows stay private; the derived reduced rows are the MatMul
/// output and are committed as the group members. The Eval targets are the
/// evaluations of those committed rows, so no plaintext output is supplied.
pub fn prove_hidden_lin_hidden(
    plan: &HiddenLinPlan,
    activation: &CommittedActivation,
    table: &mut dyn HiddenTableProver,
    seed: Digest,
) -> Result<(HiddenLinMessages, HiddenLinWitness)> {
    if !plan.weights.is_empty() || plan.norm.is_some() {
        return Err(Error::InvalidConfig);
    }
    prove_hidden_lin_inner(plan, Some(activation), table, None, seed, true, None, None, None)
}

/// Prover entry of the hidden-input mode with the private norm witness. The
/// norm groups are committed and sampled before the second stage draws its
/// challenges; the witness never enters the proof bytes.
pub fn prove_hidden_lin_hidden_with_norm(
    plan: &HiddenLinPlan,
    activation: &CommittedActivation,
    table: &mut dyn HiddenTableProver,
    seed: Digest,
    norm: &NormWitness,
) -> Result<(HiddenLinMessages, HiddenLinWitness)> {
    if !plan.weights.is_empty() || plan.norm.is_none() {
        return Err(Error::InvalidConfig);
    }
    prove_hidden_lin_inner(plan, Some(activation), table, None, seed, true, Some(norm), None, None)
}

/// Prover entry of the hidden-input mode where the Dot activation is the norm
/// hidden-output object. No activation matrix is supplied: the rows are the
/// committed norm output messages, so the MatMul consumes the same object the
/// norm relation proves. The norm witness carries those messages.
pub fn prove_hidden_lin_hidden_with_norm_hd(
    plan: &HiddenLinPlan,
    table: &mut dyn HiddenTableProver,
    seed: Digest,
    norm: &NormWitness,
    param_tables: Option<&mut dyn NormParamTableProver>,
) -> Result<(HiddenLinMessages, HiddenLinWitness)> {
    if !plan.weights.is_empty()
        || plan.norm.is_none()
        || !plan.send.activation_source.is_norm_hd()
    {
        return Err(Error::InvalidConfig);
    }
    prove_hidden_lin_inner(plan, None, table, None, seed, true, Some(norm), None, param_tables)
}

/// Test-only prover entry that omits the residual absorption before the
/// product-sumcheck challenges. The canonical verifier must reject its
/// messages; the flag exists so the challenge-order negative can be built
/// from the real prover path instead of a hand-assembled transcript.
#[cfg(test)]
fn prove_hidden_lin_hidden_unbound_residual(
    plan: &HiddenLinPlan,
    activation: &CommittedActivation,
    table: &mut dyn HiddenTableProver,
    seed: Digest,
) -> Result<(HiddenLinMessages, HiddenLinWitness)> {
    if !plan.weights.is_empty() {
        return Err(Error::InvalidConfig);
    }
    prove_hidden_lin_inner(plan, Some(activation), table, None, seed, false, None, None, None)
}

fn verify_hidden_lin_inner(
    plan: &HiddenLinPlan,
    reader: &IndexReader,
    output: Option<&[Vec<Fp3>]>,
    messages: &HiddenLinMessages,
    rstd_table: Option<&dyn crate::proof::lookup::LookupTable>,
) -> Result<HiddenLinOutcome> {
    let _profile_verify = zkie_engine::profile::Scope::new("verify.hidden_lin");
    plan.validate()?;
    let _profile_first = zkie_engine::profile::Scope::new("verify.hidden_lin.first");
    let (eval_rows, first_seal, closure_r) =
        verify_first(plan, reader, output, &messages.first)?;
    drop(_profile_first);
    if first_seal != messages.first_seal || messages.first.chain_value != first_seal {
        return Err(Error::InvalidProof);
    }
    let _profile_first_query = zkie_engine::profile::Scope::light("verify.hidden_lin.first_query");
    let first_query = *first_chain(plan, reader)?.value();
    drop(_profile_first_query);
    let _profile_shared = zkie_engine::profile::Scope::new("verify.hidden_lin.shared");
    let (shared_seal, shared_claims, points, shared_base) =
        verify_shared(plan, reader, &messages.first, &first_seal, &messages.shared)?;
    drop(_profile_shared);
    if shared_seal != messages.shared.seal {
        return Err(Error::InvalidProof);
    }
    let _profile_second = zkie_engine::profile::Scope::new("verify.hidden_lin.second");
    let (seal, own_claims, own_base, dot_binding, closure_binding, norm_output) = verify_second(
        plan,
        reader,
        &messages.first,
        &first_query,
        &messages.shared,
        &points.psi,
        &messages.second,
        closure_r.as_deref(),
        rstd_table,
    )?;
    drop(_profile_second);
    Ok(HiddenLinOutcome {
        activation: messages.first.activation,
        group: messages.first.group.clone(),
        eval_rows,
        shared_claims,
        own_claims,
        psi: messages.first.psi.clone(),
        shared_point: points.psi,
        zeta: points.zeta,
        first_seal,
        shared_seal,
        seal,
        column_positions: messages.shared.column_positions.clone(),
        shared_base,
        own_base,
        dot: dot_binding,
        closure: closure_binding,
        norm_output,
    })
}

/// Verifier entry of the merged three-stage run with the public output rows as
/// a diagnostic plan value. The activation rows, the table message rows and
/// the reduced rows never enter this entry; the output rows do, so this entry
/// is the diagnostic path, not the formal hidden-input path.
pub fn verify_hidden_lin(
    plan: &HiddenLinPlan,
    reader: &IndexReader,
    output: &[Vec<Fp3>],
    messages: &HiddenLinMessages,
) -> Result<HiddenLinOutcome> {
    if plan.weights.is_empty() {
        return Err(Error::InvalidConfig);
    }
    verify_hidden_lin_inner(plan, reader, Some(output), messages, None)
}

/// Verifier entry of the merged three-stage run in the hidden-input mode. It
/// replays all three chains, checks every commitment, sample answer, drawn
/// position, opening and seal, and runs the two proximity tests against its
/// own table reader and member roots. Neither the activation rows nor any
/// output rows enter this entry; the Eval targets are message values bound to
/// the committed reduced rows by the BAT rows.
pub fn verify_hidden_lin_hidden(
    plan: &HiddenLinPlan,
    reader: &IndexReader,
    messages: &HiddenLinMessages,
) -> Result<HiddenLinOutcome> {
    if !plan.weights.is_empty() {
        return Err(Error::InvalidConfig);
    }
    verify_hidden_lin_inner(plan, reader, None, messages, None)
}

/// Proves the hidden chain with the frozen-table lookup witness and the
/// trusted table view.
pub fn prove_hidden_lin_hidden_with_norm_and_rstd_table(
    plan: &HiddenLinPlan,
    activation: &CommittedActivation,
    table: &mut dyn HiddenTableProver,
    seed: Digest,
    norm: &NormWitness,
    rstd_table: &dyn crate::proof::lookup::LookupTable,
) -> Result<(HiddenLinMessages, HiddenLinWitness)> {
    if !plan.weights.is_empty() || plan.norm.is_none() {
        return Err(Error::InvalidConfig);
    }
    prove_hidden_lin_inner(
        plan,
        Some(activation),
        table,
        None,
        seed,
        true,
        Some(norm),
        Some(rstd_table),
        None,
    )
}

/// Proves the hidden chain with the norm parameter tables. The tables are the
/// fixture's external trusted encoded tables; the verifier only holds the plan
/// readers and checks the drawn openings against them.
pub fn prove_hidden_lin_hidden_with_norm_and_params(
    plan: &HiddenLinPlan,
    activation: &CommittedActivation,
    table: &mut dyn HiddenTableProver,
    seed: Digest,
    norm: &NormWitness,
    param_tables: &mut dyn NormParamTableProver,
) -> Result<(HiddenLinMessages, HiddenLinWitness)> {
    if !plan.weights.is_empty() || plan.norm.is_none() {
        return Err(Error::InvalidConfig);
    }
    prove_hidden_lin_inner(
        plan,
        Some(activation),
        table,
        None,
        seed,
        true,
        Some(norm),
        None,
        Some(param_tables),
    )
}

/// Verifies the hidden chain with the trusted frozen-table view.
pub fn verify_hidden_lin_hidden_with_rstd_table(
    plan: &HiddenLinPlan,
    reader: &IndexReader,
    messages: &HiddenLinMessages,
    rstd_table: &dyn crate::proof::lookup::LookupTable,
) -> Result<HiddenLinOutcome> {
    if !plan.weights.is_empty() {
        return Err(Error::InvalidConfig);
    }
    verify_hidden_lin_inner(plan, reader, None, messages, Some(rstd_table))
}

/// Auxiliary verifier for unit and fixture runs. It runs the public entry and
/// then checks the prover's private samples against the public answers. The
/// witness is never serialized and this entry is not a production path.
pub fn verify_hidden_lin_with_witness(
    plan: &HiddenLinPlan,
    reader: &IndexReader,
    output: &[Vec<Fp3>],
    messages: &HiddenLinMessages,
    witness: &HiddenLinWitness,
) -> Result<HiddenLinOutcome> {
    let outcome = verify_hidden_lin(plan, reader, output, messages)?;
    verify_hidden_lin_witness_agreement(plan, messages, witness, &outcome)?;
    Ok(outcome)
}

/// Auxiliary verifier for the hidden-input mode. It runs the public entry and
/// then checks the prover's private samples and mask entries against the
/// public values, including the tilde row, the public residual, the difference
/// row target and the product-sumcheck output rows. The witness is never
/// serialized and this entry is not a production path.
pub fn verify_hidden_lin_hidden_with_witness(
    plan: &HiddenLinPlan,
    reader: &IndexReader,
    messages: &HiddenLinMessages,
    witness: &HiddenLinWitness,
) -> Result<HiddenLinOutcome> {
    let outcome = verify_hidden_lin_hidden(plan, reader, messages)?;
    verify_hidden_lin_witness_agreement(plan, messages, witness, &outcome)?;
    Ok(outcome)
}

/// Checks the prover's private samples and mask entries against an outcome
/// that the public verifier already produced. The auxiliary witness verifier
/// calls this after its own verification; a caller that already holds the
/// outcome (for example a fixture that verified once) can call it directly
/// without replaying the verifier. The witness is never serialized and this
/// entry is not a production path.
pub fn verify_hidden_lin_witness_agreement(
    plan: &HiddenLinPlan,
    messages: &HiddenLinMessages,
    witness: &HiddenLinWitness,
    outcome: &HiddenLinOutcome,
) -> Result<()> {
    witness.send.check(&plan.send)?;
    let expected = witness.send.answers(&messages.first.samples.seed)?;
    if expected != messages.first.samples.answers {
        return Err(Error::InvalidProof);
    }
    let eval = witness.send.eval_values(&messages.first.psi)?;
    match (&plan.send.index_mask, &witness.send.index_mask) {
        (Some(index), Some(mask_witness)) => {
            for position in 0..plan.send.positions() {
                let mask = mask_witness.entry(index, index.entries.eval_slot(position)?)?;
                if messages.first.eval_targets[position as usize]
                    != eval[position as usize].add(&mask)
                {
                    return Err(Error::InvalidProof);
                }
            }
            // The plaintext mask relation: `tilde_t = W_t(psi*) + g_t`, the
            // published residual is the chi-weighted difference of the
            // committed claim and tilde entries, and the product-sumcheck
            // output rows hold for the same entries and the adjusted target.
            check_hidden_dot_mask_relation(
                plan,
                messages,
                &witness.send,
                &witness.omega_message,
                outcome,
            )?;
        }
        (None, None) => {
            if eval != messages.first.eval_targets {
                return Err(Error::InvalidProof);
            }
        }
        _ => return Err(Error::ShapeMismatch),
    }
    if witness.gu_message.len() as u64 != plan.proximity.gu.u.main_rows * plan.proximity.gu.u.code.ell
    {
        return Err(Error::ShapeMismatch);
    }
    let gu_witness = GuWitness {
        main: witness.gu_message.clone(),
        mask: Vec::new(),
    };
    let mut registry = GuRegistry::new();
    registry.receive(
        &plan.proximity.gu,
        &GuNotice {
            root: messages.shared.gu_root,
            oracle: plan.proximity.gu.oracle,
            tree_number: plan.proximity.gu.tree.tree_number,
            part: plan.proximity.gu.part,
        },
    )?;
    let gu_expected = gu_answers(
        &plan.proximity.gu,
        &registry,
        &gu_witness,
        &messages.shared.gu_seeds,
    )?;
    if gu_expected != messages.shared.gu_answers {
        return Err(Error::InvalidProof);
    }
    let omega_witness = GuWitness {
        main: witness.omega_message.clone(),
        mask: Vec::new(),
    };
    let omega_expected = omega_answers(
        &plan.proximity.omega,
        &omega_witness,
        &messages.shared.omega_seeds,
    )?;
    if omega_expected != messages.shared.omega_answers {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// The exact plaintext form of the hidden Dot mask relation, used by the
/// auxiliary witness verifier and by the negatives. The production verifier
/// does not see these values; it checks the same equations through the
/// committed own-group rows.
fn check_hidden_dot_mask_relation(
    plan: &HiddenLinPlan,
    messages: &HiddenLinMessages,
    witness: &HiddenSendWitness,
    omega_message: &[Fp3],
    outcome: &HiddenLinOutcome,
) -> Result<()> {
    let index = plan.send.index_mask.as_ref().ok_or(Error::ShapeMismatch)?;
    let mask_witness = witness.index_mask.as_ref().ok_or(Error::ShapeMismatch)?;
    let binding = outcome.dot.as_ref().ok_or(Error::ShapeMismatch)?;
    let chi = binding.chi;
    let residual = binding.residual;
    if messages.second.residual != Some(residual) {
        return Err(Error::InvalidProof);
    }
    let expected_tilde = tilde_values(plan, witness, &outcome.shared_point)?;
    if expected_tilde != messages.second.tilde {
        return Err(Error::InvalidProof);
    }
    if dot_residual(index, mask_witness, chi)? != residual {
        return Err(Error::InvalidProof);
    }
    let target = dot_adjusted_target(&messages.second.tilde, chi, residual)?;
    let claim = dot_claim_from_witness(index, mask_witness, chi, target)?;
    let entries = dot_entries_from_witness(index, mask_witness)?;
    let x_chi = dot_batched_x(plan, witness, chi)?;
    let omega = omega_message_values(plan, omega_message)?;
    check_product_claims(&binding.output, &claim, &entries, &x_chi, &omega)
}


// ---------------------------------------------------------------------------
// Canonical encoding of the merged messages
// ---------------------------------------------------------------------------

fn put_seed(writer: &mut CanonicalWriter, seed: &SampleSeed) {
    writer.put_u64(seed.z.len() as u64);
    for value in &seed.z {
        writer.put_fp3(value);
    }
    writer.put_fp3(&seed.beta);
}

fn get_seed(reader: &mut CanonicalReader<'_>) -> Result<SampleSeed> {
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if count > SEED_COORDINATE_BOUND {
        return Err(Error::ResourceLimit);
    }
    let mut z = Vec::with_capacity(count);
    for _ in 0..count {
        z.push(reader.get_fp3()?);
    }
    let beta = reader.get_fp3()?;
    if beta == Fp3::ZERO {
        return Err(Error::InvalidEncoding);
    }
    Ok(SampleSeed { z, beta })
}

fn put_seed_vec(writer: &mut CanonicalWriter, seeds: &[SampleSeed]) {
    writer.put_u64(seeds.len() as u64);
    for seed in seeds {
        put_seed(writer, seed);
    }
}

fn get_seed_vec(reader: &mut CanonicalReader<'_>) -> Result<Vec<SampleSeed>> {
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if count > SEED_COORDINATE_BOUND {
        return Err(Error::ResourceLimit);
    }
    let mut seeds = Vec::with_capacity(count);
    for _ in 0..count {
        seeds.push(get_seed(reader)?);
    }
    Ok(seeds)
}

fn put_fp3_vec(writer: &mut CanonicalWriter, values: &[Fp3]) {
    writer.put_u64(values.len() as u64);
    for value in values {
        writer.put_fp3(value);
    }
}

fn get_fp3_vec(reader: &mut CanonicalReader<'_>, max: usize) -> Result<Vec<Fp3>> {
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if count > max {
        return Err(Error::ResourceLimit);
    }
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(reader.get_fp3()?);
    }
    Ok(values)
}

fn put_u64_vec(writer: &mut CanonicalWriter, values: &[u64]) {
    writer.put_u64(values.len() as u64);
    for value in values {
        writer.put_u64(*value);
    }
}

fn get_u64_vec(reader: &mut CanonicalReader<'_>, max: usize) -> Result<Vec<u64>> {
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if count > max {
        return Err(Error::ResourceLimit);
    }
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(reader.get_u64()?);
    }
    Ok(values)
}

pub(crate) fn put_column_opening(writer: &mut CanonicalWriter, opening: &ColumnOpening) {
    writer.put_u64(opening.position);
    writer.put_bytes(&opening.salt);
    writer.put_u64(opening.symbols.len() as u64);
    for symbol in &opening.symbols {
        writer.put_fp(*symbol);
    }
    writer.put_u64(opening.path.len() as u64);
    for digest in &opening.path {
        writer.put_digest(digest);
    }
}

pub(crate) fn get_column_opening(
    reader: &mut CanonicalReader<'_>,
    max: usize,
) -> Result<ColumnOpening> {
    let position = reader.get_u64()?;
    let salt = reader.get_bytes()?.to_vec();
    if salt.len() < LEAF_SALT_MIN_BYTES || salt.len() > LEAF_SALT_MAX_BYTES {
        return Err(Error::InvalidEncoding);
    }
    let symbols = get_fp_vec(reader, max)?;
    let path = get_digest_vec_public(reader, max)?;
    Ok(ColumnOpening {
        position,
        salt,
        symbols,
        path,
    })
}

fn get_fp_vec(reader: &mut CanonicalReader<'_>, max: usize) -> Result<Vec<Fp>> {
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if count > max {
        return Err(Error::ResourceLimit);
    }
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(reader.get_fp()?);
    }
    Ok(values)
}

fn get_digest_vec_public(reader: &mut CanonicalReader<'_>, max: usize) -> Result<Vec<Digest>> {
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if count > max {
        return Err(Error::ResourceLimit);
    }
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(reader.get_digest()?);
    }
    Ok(values)
}

pub(crate) fn put_runtime_opening(writer: &mut CanonicalWriter, opening: &RuntimeTreeOpening) {
    writer.put_u64(opening.positions.len() as u64);
    for position in &opening.positions {
        writer.put_u64(*position);
    }
    writer.put_u64(opening.columns.len() as u64);
    for column in &opening.columns {
        writer.put_u64(column.position);
        writer.put_bytes(&column.salt);
        writer.put_u64(column.symbols.len() as u64);
        for symbol in &column.symbols {
            writer.put_fp(*symbol);
        }
    }
    writer.put_u64(opening.given.len() as u64);
    for digest in &opening.given {
        writer.put_digest(digest);
    }
}

pub(crate) fn get_runtime_opening(
    reader: &mut CanonicalReader<'_>,
    max: usize,
) -> Result<RuntimeTreeOpening> {
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if count > max {
        return Err(Error::ResourceLimit);
    }
    let mut positions = Vec::with_capacity(count);
    for _ in 0..count {
        positions.push(reader.get_u64()?);
    }
    let column_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if column_count != count {
        return Err(Error::InvalidEncoding);
    }
    let mut columns = Vec::with_capacity(column_count);
    for _ in 0..column_count {
        let position = reader.get_u64()?;
        let salt = reader.get_bytes()?.to_vec();
        let symbols = get_fp_vec(reader, max)?;
        columns.push(RuntimeLeafColumn {
            position,
            salt,
            symbols,
        });
    }
    let given = get_digest_vec_public(reader, max)?;
    Ok(RuntimeTreeOpening {
        positions,
        columns,
        given,
    })
}

pub(crate) fn put_fold_start(writer: &mut CanonicalWriter, message: &FoldStartMessage) {
    writer.put_u64(message.id);
    writer.put_u64(message.claimed_group);
    writer.put_u64(message.tree_number);
    writer.put_u64(message.notice.mask_rows);
    put_code_public(writer, message.notice.mask_code);
    writer.put_u64(message.notice.answers as u64);
    writer.put_u64(message.notice.leaf_class as u64);
    writer.put_digest(&message.root);
    writer.put_fp3(&message.sum);
}

pub(crate) fn get_fold_start(reader: &mut CanonicalReader<'_>) -> Result<FoldStartMessage> {
    let id = reader.get_u64()?;
    let claimed_group = reader.get_u64()?;
    let tree_number = reader.get_u64()?;
    let mask_rows = reader.get_u64()?;
    let code = get_code_public(reader)?;
    let answers = u32::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    let leaf_class = u32::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    let root = reader.get_digest()?;
    let sum = reader.get_fp3()?;
    Ok(FoldStartMessage {
        id,
        claimed_group,
        tree_number,
        notice: zkie_engine::pcs::request::fold::MaskNotice {
            mask_rows,
            mask_code: code,
            answers,
            leaf_class,
        },
        root,
        sum,
    })
}

pub(crate) fn get_code_public(reader: &mut CanonicalReader<'_>) -> Result<CodeParams> {
    let log_n = u32::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    let ell = reader.get_u64()?;
    let r = reader.get_u64()?;
    build_code_plan(log_n, ell, r)
}

fn put_base_commitment(writer: &mut CanonicalWriter, value: &BaseCommitment) {
    writer.put_digest(&value.root);
    writer.put_digest(&value.start);
    writer.put_u64(value.count);
}

fn get_base_commitment(reader: &mut CanonicalReader<'_>) -> Result<BaseCommitment> {
    Ok(BaseCommitment {
        root: reader.get_digest()?,
        start: reader.get_digest()?,
        count: reader.get_u64()?,
    })
}

fn put_base_messages(writer: &mut CanonicalWriter, value: &BaseMessages) {
    writer.put_digest(&value.fold_identity);
    put_base_commitment(writer, &value.star);
    put_base_commitment(writer, &value.star_mask);
    writer.put_fp3(&value.target_star);
}

fn get_base_messages(reader: &mut CanonicalReader<'_>) -> Result<BaseMessages> {
    Ok(BaseMessages {
        fold_identity: reader.get_digest()?,
        star: get_base_commitment(reader)?,
        star_mask: get_base_commitment(reader)?,
        target_star: reader.get_fp3()?,
    })
}

fn put_base_rows(writer: &mut CanonicalWriter, value: &BaseFinalRows) {
    writer.put_fp3(&value.base.target_star);
    writer.put_fp3(&value.base.lambda);
    put_fp3_vec(writer, &value.base.main_prime);
    put_fp3_vec(writer, &value.base.main_tail_prime);
    put_fp3_vec(writer, &value.base.mask_prime);
    put_fp3_vec(writer, &value.base.mask_tail_prime);
    put_fp3_vec(writer, &value.fresh_row);
    put_fp3_vec(writer, &value.fresh_tail);
}

fn get_base_rows(reader: &mut CanonicalReader<'_>, max: usize) -> Result<BaseFinalRows> {
    Ok(BaseFinalRows {
        base: zkie_engine::pcs::request::base_stage::BaseRows {
            target_star: reader.get_fp3()?,
            lambda: reader.get_fp3()?,
            main_prime: get_fp3_vec(reader, max)?,
            main_tail_prime: get_fp3_vec(reader, max)?,
            mask_prime: get_fp3_vec(reader, max)?,
            mask_tail_prime: get_fp3_vec(reader, max)?,
        },
        fresh_row: get_fp3_vec(reader, max)?,
        fresh_tail: get_fp3_vec(reader, max)?,
    })
}

fn put_base_openings(writer: &mut CanonicalWriter, value: &BaseOpenings) {
    writer.put_u64(value.star.len() as u64);
    for opening in &value.star {
        writer.put_u64(opening.index);
        put_u64_vec(writer, &opening.values);
        writer.put_u64(opening.path.len() as u64);
        for digest in &opening.path {
            writer.put_digest(digest);
        }
    }
    writer.put_u64(value.star_mask.len() as u64);
    for opening in &value.star_mask {
        writer.put_u64(opening.index);
        put_u64_vec(writer, &opening.values);
        writer.put_u64(opening.path.len() as u64);
        for digest in &opening.path {
            writer.put_digest(digest);
        }
    }
    writer.put_u64(value.members.len() as u64);
    for opening in &value.members {
        put_runtime_opening(writer, opening);
    }
    put_runtime_opening(writer, &value.fresh);
}

fn get_base_openings(reader: &mut CanonicalReader<'_>, max: usize) -> Result<BaseOpenings> {
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if count > max {
        return Err(Error::ResourceLimit);
    }
    let mut star = Vec::with_capacity(count);
    for _ in 0..count {
        let index = reader.get_u64()?;
        let values = get_u64_vec(reader, max)?;
        let path = get_digest_vec_public(reader, max)?;
        star.push(BaseOpening {
            index,
            values,
            path,
        });
    }
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if count > max {
        return Err(Error::ResourceLimit);
    }
    let mut star_mask = Vec::with_capacity(count);
    for _ in 0..count {
        let index = reader.get_u64()?;
        let values = get_u64_vec(reader, max)?;
        let path = get_digest_vec_public(reader, max)?;
        star_mask.push(BaseOpening {
            index,
            values,
            path,
        });
    }
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if count > max {
        return Err(Error::ResourceLimit);
    }
    let mut members = Vec::with_capacity(count);
    for _ in 0..count {
        members.push(get_runtime_opening(reader, max)?);
    }
    let fresh = get_runtime_opening(reader, max)?;
    Ok(BaseOpenings {
        star: star.into_iter().map(Into::into).collect(),
        star_mask: star_mask.into_iter().map(Into::into).collect(),
        members,
        fresh,
    })
}

fn put_shared(writer: &mut CanonicalWriter, message: &HiddenSharedMessages) {
    writer.put_digest(&message.gu_root);
    put_seed_vec(writer, &message.gu_seeds);
    put_fp3_vec(writer, &message.gu_answers);
    writer.put_u64(message.column_nonce);
    put_u64_vec(writer, &message.column_positions);
    writer.put_u64(message.column_openings.len() as u64);
    for opening in &message.column_openings {
        put_column_opening(writer, opening);
    }
    writer.put_digest(&message.omega_root);
    put_seed_vec(writer, &message.omega_seeds);
    put_fp3_vec(writer, &message.omega_answers);
    put_fp3_vec(writer, &message.zeta);
    put_fold_start(writer, &message.fold_start);
    writer.put_u64(message.fold_rounds.len() as u64);
    for (c0, c2) in &message.fold_rounds {
        writer.put_fp3(c0);
        writer.put_fp3(c2);
    }
    put_tail(writer, &message.tail);
    writer.put_digest(&message.seal);
}

/// Encodes the group's end. The tag names which end the plan produced; the
/// decoder derives the same choice from the trusted plan and rejects the
/// other tag.
fn put_tail(writer: &mut CanonicalWriter, tail: &ProximityTail) {
    match tail {
        ProximityTail::Single {
            base,
            base_rows,
            base_openings,
        } => {
            writer.put_u64(0);
            put_base_messages(writer, base);
            put_base_rows(writer, base_rows);
            put_base_openings(writer, base_openings);
        }
        ProximityTail::Stages(proof) => {
            writer.put_u64(1);
            writer.put_bytes(&proof.encode());
        }
    }
}

/// Decodes the group's end against the trusted plan. A group with an empty
/// stage list carries the single-stage BASE artifact; a group with stages
/// carries a loop proof that is decoded against the plan-derived chain.
fn get_tail(
    reader: &mut CanonicalReader<'_>,
    iopp: Option<&ProximityIoppPlan>,
) -> Result<ProximityTail> {
    match (reader.get_u64()?, iopp) {
        (0, None) => {
            let base = get_base_messages(reader)?;
            let base_rows = get_base_rows(reader, ACTIVATION_DECODE_BOUND)?;
            let base_openings = get_base_openings(reader, ACTIVATION_DECODE_BOUND)?;
            Ok(ProximityTail::Single {
                base,
                base_rows,
                base_openings,
            })
        }
        (1, Some(iopp)) => {
            let bytes = reader.get_bytes()?;
            if bytes.len() > PROXIMITY_LOOP_DECODE_BOUND {
                return Err(Error::ResourceLimit);
            }
            Ok(ProximityTail::Stages(IoppProof::decode(
                &iopp.chain,
                bytes,
            )?))
        }
        _ => Err(Error::ShapeMismatch),
    }
}

fn get_shared(
    reader: &mut CanonicalReader<'_>,
    plan: &HiddenLinPlan,
) -> Result<HiddenSharedMessages> {
    let gu_root = reader.get_digest()?;
    let gu_seeds = get_seed_vec(reader)?;
    let gu_answers = get_fp3_vec(reader, SEED_COORDINATE_BOUND)?;
    let column_nonce = reader.get_u64()?;
    let column_positions = get_u64_vec(reader, GROUP_ENTRY_BOUND)?;
    let opening_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if opening_count > GROUP_ENTRY_BOUND {
        return Err(Error::ResourceLimit);
    }
    let mut column_openings = Vec::with_capacity(opening_count);
    for _ in 0..opening_count {
        column_openings.push(get_column_opening(reader, ACTIVATION_DECODE_BOUND)?);
    }
    let omega_root = reader.get_digest()?;
    let omega_seeds = get_seed_vec(reader)?;
    let omega_answers = get_fp3_vec(reader, SEED_COORDINATE_BOUND)?;
    let zeta = get_fp3_vec(reader, SEED_COORDINATE_BOUND)?;
    let fold_start = get_fold_start(reader)?;
    let round_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if round_count > 64 {
        return Err(Error::ResourceLimit);
    }
    let mut fold_rounds = Vec::with_capacity(round_count);
    for _ in 0..round_count {
        fold_rounds.push((reader.get_fp3()?, reader.get_fp3()?));
    }
    let iopp = proximity_iopp_plan(
        &plan.proximity,
        &plan.send,
        plan.closure.as_ref(),
        plan.norm.as_ref(),
        ProximityGroupRole::Shared,
    )?;
    let tail = get_tail(reader, iopp.as_ref())?;
    let seal = reader.get_digest()?;
    Ok(HiddenSharedMessages {
        gu_root,
        gu_seeds,
        gu_answers,
        column_nonce,
        column_positions,
        column_openings,
        omega_root,
        omega_seeds,
        omega_answers,
        zeta,
        fold_start,
        fold_rounds,
        tail,
        seal,
    })
}

fn put_psc_round(writer: &mut CanonicalWriter, round: &PscRound) {
    writer.put_u64(round.coefficients.len() as u64);
    for coefficient in &round.coefficients {
        writer.put_fp3(coefficient);
    }
}

fn get_psc_round(reader: &mut CanonicalReader<'_>, max: usize) -> Result<PscRound> {
    let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if count > max {
        return Err(Error::ResourceLimit);
    }
    let mut coefficients = Vec::with_capacity(count);
    for _ in 0..count {
        coefficients.push(reader.get_fp3()?);
    }
    Ok(PscRound { coefficients })
}

fn put_dot_proof(writer: &mut CanonicalWriter, proof: &DotProof) {
    writer.put_fp3(&proof.hiding_sum);
    writer.put_u64(proof.rounds.len() as u64);
    for round in &proof.rounds {
        put_psc_round(writer, round);
    }
    put_psc_round(writer, &proof.y_round);
    writer.put_u64(proof.end_values.len() as u64);
    for value in &proof.end_values {
        writer.put_fp3(value);
    }
}

fn get_dot_proof(reader: &mut CanonicalReader<'_>, plan: &HiddenLinPlan) -> Result<DotProof> {
    let hidden = plan
        .send
        .dot
        .as_ref()
        .ok_or(Error::InvalidEncoding)?
        .hidden as u64;
    let variables = psc_variables(dot_variables(plan.send.matrix_rows)?)?;
    let k = variables.len() - 1;
    let hiding_sum = reader.get_fp3()?;
    let round_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if round_count != k || hidden != plan.send.matrix_rows {
        return Err(Error::ShapeMismatch);
    }
    let mut rounds = Vec::with_capacity(k);
    for weight in &variables[..k] {
        let round = get_psc_round(reader, PLAIN_DEGREE)?;
        if round.coefficients.len() != weight.degree() - 1 {
            return Err(Error::ShapeMismatch);
        }
        rounds.push(round);
    }
    let y_round = get_psc_round(reader, PLAIN_DEGREE)?;
    if y_round.coefficients.len() != variables[k].degree() - 1 {
        return Err(Error::ShapeMismatch);
    }
    let end_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if end_count != SHOWN_MAPS {
        return Err(Error::ShapeMismatch);
    }
    let mut end_values = [Fp3::ZERO; SHOWN_MAPS];
    for value in end_values.iter_mut() {
        *value = reader.get_fp3()?;
    }
    Ok(DotProof {
        hiding_sum,
        rounds,
        y_round,
        end_values,
    })
}

fn put_second(writer: &mut CanonicalWriter, message: &HiddenSecondMessages) -> Result<()> {
    writer.put_digest(&message.omega_root);
    writer.put_digest(&message.first_query);
    writer.put_digest(&message.shared_seal);
    writer.put_u64(message.tilde.len() as u64);
    for value in &message.tilde {
        writer.put_fp3(value);
    }
    match &message.dot {
        Some(proof) => {
            writer.put_u64(1);
            writer.put_fp3(&message.residual.ok_or(Error::ShapeMismatch)?);
            put_dot_proof(writer, proof);
        }
        None => {
            if message.residual.is_some() {
                return Err(Error::ShapeMismatch);
            }
            writer.put_u64(0);
        }
    }
    if let Some(closure) = &message.closure {
        writer.put_u64(1);
        writer.put_bytes(&closure.encode()?);
    }
    if let Some(norm) = &message.norm {
        writer.put_bytes(&norm.encode()?);
    }
    put_fold_start(writer, &message.fold_start);
    writer.put_u64(message.fold_rounds.len() as u64);
    for (c0, c2) in &message.fold_rounds {
        writer.put_fp3(c0);
        writer.put_fp3(c2);
    }
    put_tail(writer, &message.tail);
    writer.put_digest(&message.seal);
    Ok(())
}

fn get_second(reader: &mut CanonicalReader<'_>, plan: &HiddenLinPlan) -> Result<HiddenSecondMessages> {
    let omega_root = reader.get_digest()?;
    let first_query = reader.get_digest()?;
    let shared_seal = reader.get_digest()?;
    let tilde_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if tilde_count > ACTIVATION_DECODE_BOUND {
        return Err(Error::ResourceLimit);
    }
    let mut tilde = Vec::with_capacity(tilde_count);
    for _ in 0..tilde_count {
        tilde.push(reader.get_fp3()?);
    }
    let (residual, dot) = match reader.get_u64()? {
        0 => {
            if plan.send.dot.is_some() {
                return Err(Error::ShapeMismatch);
            }
            (None, None)
        }
        1 => {
            if plan.send.dot.is_none() {
                return Err(Error::ShapeMismatch);
            }
            let residual = reader.get_fp3()?;
            (Some(residual), Some(get_dot_proof(reader, plan)?))
        }
        _ => return Err(Error::InvalidEncoding),
    };
    let closure = match plan.closure.as_ref() {
        Some(closure_plan) => match reader.get_u64()? {
            0 => return Err(Error::ShapeMismatch),
            1 => Some(ClosureSecondMessages::decode(
                closure_plan,
                plan.send.positions(),
                reader.get_bytes()?,
            )?),
            _ => return Err(Error::InvalidEncoding),
        },
        None => None,
    };
    let norm = match plan.norm.as_ref() {
        Some(norm_plan) => Some(NormPscSecond::decode_for(norm_plan, reader.get_bytes()?)?),
        None => None,
    };
    let fold_start = get_fold_start(reader)?;
    let round_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if round_count > 64 {
        return Err(Error::ResourceLimit);
    }
    let mut fold_rounds = Vec::with_capacity(round_count);
    for _ in 0..round_count {
        fold_rounds.push((reader.get_fp3()?, reader.get_fp3()?));
    }
    let iopp = proximity_iopp_plan(
        &plan.proximity,
        &plan.send,
        plan.closure.as_ref(),
        plan.norm.as_ref(),
        ProximityGroupRole::Own,
    )?;
    let tail = get_tail(reader, iopp.as_ref())?;
    let seal = reader.get_digest()?;
    Ok(HiddenSecondMessages {
        omega_root,
        first_query,
        shared_seal,
        tilde,
        residual,
        dot,
        closure,
        norm,
        fold_start,
        fold_rounds,
        tail,
        seal,
    })
}

impl HiddenLinMessages {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(HIDDEN_SEND_SCHEMA_VERSION);
        writer.put_u64(14);
        writer.put_bytes(&self.first.encode()?);
        writer.put_digest(&self.first_seal);
        let mut shared = CanonicalWriter::new();
        put_shared(&mut shared, &self.shared);
        writer.put_bytes(&shared.into_bytes());
        let mut second = CanonicalWriter::new();
        put_second(&mut second, &self.second)?;
        writer.put_bytes(&second.into_bytes());
        Ok(writer.into_bytes())
    }

    pub fn decode(plan: &HiddenLinPlan, bytes: &[u8]) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        if reader.get_u64()? != HIDDEN_SEND_SCHEMA_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let layout = reader.get_u64()?;
        if layout < 14 {
            return Err(Error::UnsupportedVersion);
        }
        if layout != 14 {
            return Err(Error::InvalidEncoding);
        }
        let first = HiddenSendMessages::decode(
            &plan.send,
            plan.closure.as_ref(),
            plan.norm.as_ref(),
            reader.get_bytes()?,
        )?;
        let first_seal = reader.get_digest()?;
        let shared_bytes = reader.get_bytes()?.to_vec();
        let second_bytes = reader.get_bytes()?.to_vec();
        reader.finish()?;
        let mut shared_reader = CanonicalReader::new(&shared_bytes, DecodeLimits::unlimited());
        let shared = get_shared(&mut shared_reader, plan)?;
        shared_reader.finish()?;
        let mut second_reader = CanonicalReader::new(&second_bytes, DecodeLimits::unlimited());
        let second = get_second(&mut second_reader, plan)?;
        second_reader.finish()?;
        Ok(Self {
            first,
            first_seal,
            shared,
            second,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::column_session::fp_from_signed_word;
    use zkie_engine::pcs::request::bat::BatMemberLayout;
    use zkie_engine::pcs::request::field::Fp;
    use zkie_engine::pcs::request::fold::FoldMemberLayout;
    use zkie_engine::pcs::request::relation::GuRelationKind;
    use zkie_engine::pcs::request::table::{
        commit_table_streaming, IndexShape, TableDistances, VecTableSource,
    };

    const POSITIONS: u32 = 4;
    const HIDDEN: u32 = 4;
    const TABLE_ROWS: u64 = 4;
    const K_LO: u32 = 1;
    const COLUMNS: u64 = 8;
    const SALT: usize = LEAF_SALT_MIN_BYTES;
    const TABLE_CODE_LOG_N: u32 = 4;
    const TABLE_CODE_ELL: u64 = 12;
    const TABLE_CODE_R: u64 = 1;
    const W_CODE_LOG_N: u32 = 3;
    const W_CODE_ELL: u64 = 4;
    const W_CODE_R: u64 = 4;
    const TABLE_CONFIGURATION: Digest = Digest([0x21; 32]);
    const ACTIVATION_CONFIGURATION: Digest = Digest([0x31; 32]);
    const W_CONFIGURATION: Digest = Digest([0x51; 32]);
    const W_RUN: Digest = Digest([0x41; 32]);
    const GROUP: u64 = 7;
    const ORACLE_BASE: u64 = 20;
    const TREE_NUMBER: u64 = 100;
    const LEAF_CLASS: u32 = 1;

    fn table_shape() -> IndexShape {
        IndexShape {
            code: build_code_plan(TABLE_CODE_LOG_N, TABLE_CODE_ELL, TABLE_CODE_R).unwrap(),
            v: 2,
            answers: 1,
            salt_bytes: SALT,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        }
    }

    fn table_messages() -> Vec<Vec<Fp>> {
        let shape = table_shape();
        let rows = shape.rows() as usize;
        let ell = shape.code.ell as usize;
        let mut messages = vec![vec![Fp::ZERO; ell]; rows];
        for (row, message) in messages.iter_mut().enumerate() {
            for entry in 0..COLUMNS as usize {
                let word = ((row as i64 * 13 + entry as i64 * 5) % 11) - 5;
                message[entry] = fp_from_signed_word(word as u64);
            }
            for entry in shape.plain() as usize..ell {
                message[entry] = Fp::ZERO;
            }
        }
        messages
    }

    fn table_tails() -> Vec<Vec<Fp>> {
        vec![vec![Fp::ZERO; TABLE_CODE_R as usize]; TABLE_ROWS as usize]
    }

    fn table_salts() -> Vec<Vec<u8>> {
        (0..(1usize << TABLE_CODE_LOG_N))
            .map(|position| vec![(position as u8) ^ 0x5A; SALT])
            .collect()
    }

    fn activation_rows() -> Vec<Vec<i64>> {
        vec![
            vec![1, 2, -1, 0],
            vec![0, -2, 3, 1],
            vec![-1, 1, 1, 2],
            vec![2, 0, -3, 1],
        ]
    }

    fn activation_salts() -> Vec<Vec<u8>> {
        (0..POSITIONS)
            .map(|position| vec![(position as u8).wrapping_mul(7) ^ 0x3C; SALT])
            .collect()
    }

    fn activation() -> CommittedActivation {
        CommittedActivation::commit_signed(
            ActivationShape::new(POSITIONS, HIDDEN, SALT).unwrap(),
            &activation_rows(),
            &activation_salts(),
            ACTIVATION_CONFIGURATION,
        )
        .unwrap()
    }

    fn w_shape() -> WShape {
        WShape::new(
            COLUMNS,
            build_code_plan(W_CODE_LOG_N, W_CODE_ELL, W_CODE_R).unwrap(),
            1,
            POSITIONS,
            SALT,
            TREE_NUMBER,
            LEAF_CLASS,
            W_RUN,
            W_CONFIGURATION,
        )
        .unwrap()
    }

    fn randomness() -> WCommitRandomness {
        let main_rows = w_shape().main_rows as usize;
        let tail_len = W_CODE_R as usize;
        let columns = 1usize << W_CODE_LOG_N;
        WCommitRandomness {
            tails: (0..POSITIONS as usize)
                .map(|position| {
                    (0..main_rows * tail_len)
                        .map(|index| {
                            Fp3::from_u64(0x200 + position as u64 * 0x40 + index as u64)
                        })
                        .collect()
                })
                .collect(),
            salts: (0..POSITIONS as usize)
                .map(|position| {
                    (0..columns)
                        .map(|column| vec![(position as u8).wrapping_mul(11) ^ column as u8; SALT])
                        .collect()
                })
                .collect(),
            index_mask: None,
            dot: None,
            pads: (0..POSITIONS as usize)
                .map(|position| vec![Fp3::from_u64(0x900 + position as u64)])
                .collect(),
            closure: None,
        }
    }

    struct Fixture {
        plan: HiddenSendPlan,
        verifier_plan: HiddenSendPlan,
        activation: CommittedActivation,
        reader: IndexReader,
        randomness: WCommitRandomness,
        messages: HiddenSendMessages,
        witness: HiddenSendWitness,
    }

    fn fixture() -> Fixture {
        let messages = table_messages();
        let tails = table_tails();
        let salts = table_salts();
        let mut source = VecTableSource::new(messages.clone(), tails.clone()).unwrap();
        let table = commit_table_streaming(
            &mut source,
            table_shape(),
            TABLE_CONFIGURATION,
            &salts,
            64,
            u64::MAX,
        )
        .unwrap();
        let reader = table.reader.clone();
        let activation = activation();
        let plan = HiddenSendPlan {
            activation: activation.shape(),
            activation_configuration: ACTIVATION_CONFIGURATION,
            activation_root: activation.root(),
            activation_source: ActivationSource::Commitment,
            table_shape: table_shape(),
            table_identity: reader_table_identity(&reader).unwrap(),
            matrix_rows: TABLE_ROWS,
            k_lo: K_LO,
            columns: COLUMNS,
            group: GROUP,
            oracle_base: ORACLE_BASE,
            openings_per_oracle: 2,
            w: w_shape(),
            index_mask: None,
            dot: None,
            expected_dot_root: None,
            expected_roots: None,
        };
        let mut source = VecTableSource::new(messages, tails).unwrap();
        let randomness = randomness();
        let (messages, witness) =
            prove_hidden_send(&plan, &activation, &reader, &mut source, &randomness).unwrap();
        let mut verifier_plan = plan.clone();
        verifier_plan.expected_roots = Some(witness.roots.clone());
        Fixture {
            plan,
            verifier_plan,
            activation,
            reader,
            randomness,
            messages,
            witness,
        }
    }

    fn assert_reject(result: Result<HiddenSendOutcome>) {
        assert!(result.is_err(), "the altered message was accepted");
    }

    fn assert_prove_reject(result: Result<(HiddenSendMessages, HiddenSendWitness)>) {
        assert!(result.is_err(), "the altered witness was accepted");
    }

    #[test]
    fn honest_run_registers_samples_and_eval_rows() {
        let fixture = fixture();
        let outcome =
            verify_hidden_send(&fixture.verifier_plan, &fixture.messages, &fixture.witness)
                .unwrap();
        assert_eq!(outcome.sample_rows.len(), POSITIONS as usize);
        assert_eq!(outcome.eval_rows.len(), POSITIONS as usize);
        assert_eq!(outcome.answers.len(), POSITIONS as usize);
        assert_eq!(outcome.seeds.len(), 1);
        assert_eq!(outcome.seeds[0].z.len(), 4);
        assert!(outcome.seeds[0].beta != Fp3::ZERO);
        for row in &outcome.sample_rows {
            assert_eq!(row.kind, GuRelationKind::Sample);
            assert_eq!(row.group, GROUP);
        }
        for (position, row) in outcome.eval_rows.iter().enumerate() {
            assert_eq!(row.position, position as u32);
            assert_eq!(row.oracle, ORACLE_BASE + position as u64);
            assert_eq!(row.group, GROUP);
        }
        // The protocol-shaped public entry accepts the same honest message.
        verify_hidden_send_public(&fixture.verifier_plan, &fixture.messages).unwrap();
    }

    #[test]
    fn x_commitment_is_hiding_and_bound() {
        let fixture = fixture();
        let activation = fixture.activation.clone();
        let commitment = *activation.commitment();
        let encoded = commitment.encode();
        assert_eq!(ActivationCommitment::decode(&encoded).unwrap(), commitment);

        // A changed row under the same salt changes the root.
        let mut rows = activation.rows().to_vec();
        rows[0][0] = rows[0][0].add(&Fp3::ONE);
        let changed = CommittedActivation::commit(
            activation.shape(),
            &rows,
            activation.salts(),
            ACTIVATION_CONFIGURATION,
        )
        .unwrap();
        assert!(changed.root() != commitment.root);
        let mut source = VecTableSource::new(table_messages(), table_tails()).unwrap();
        assert_prove_reject(prove_hidden_send(
            &fixture.plan,
            &changed,
            &fixture.reader,
            &mut source,
            &fixture.randomness,
        ));

        // A tampered opening is rejected.
        let mut opening = activation.open(0).unwrap();
        assert!(verify_activation_opening(&commitment, &opening).is_ok());
        opening.values[0] = opening.values[0].add(&Fp3::ONE);
        assert!(verify_activation_opening(&commitment, &opening).is_err());
        let mut opening = activation.open(1).unwrap();
        opening.salt[0] ^= 1;
        assert!(verify_activation_opening(&commitment, &opening).is_err());

        // The commitment message carries no row values: its fixed-size
        // encoding is just the shape, the configuration and the root.
        assert_eq!(encoded.len(), 8 + 8 * 3 + 32 + 32);
    }

    #[test]
    fn changed_activation_commitment_or_value_is_rejected() {
        let fixture = fixture();
        let mut messages = fixture.messages.clone();
        messages.activation.root = Digest([0x77; 32]);
        assert_reject(verify_hidden_send(
            &fixture.verifier_plan,
            &messages,
            &fixture.witness,
        ));

        // The prover cannot commit other rows under the pinned root.
        let mut rows = fixture.activation.rows().to_vec();
        rows[2][1] = rows[2][1].add(&Fp3::ONE);
        let changed = CommittedActivation::commit(
            fixture.activation.shape(),
            &rows,
            fixture.activation.salts(),
            ACTIVATION_CONFIGURATION,
        )
        .unwrap();
        let mut source = VecTableSource::new(table_messages(), table_tails()).unwrap();
        assert_prove_reject(prove_hidden_send(
            &fixture.plan,
            &changed,
            &fixture.reader,
            &mut source,
            &fixture.randomness,
        ));
    }

    #[test]
    fn changed_sample_answer_is_rejected_by_the_witness_check() {
        let fixture = fixture();
        let mut messages = fixture.messages.clone();
        let honest = messages.samples.answers[0];
        messages.samples.answers[0] = messages.samples.answers[0].add(&Fp3::ONE);
        // The witness check rejects the changed answer.
        assert_reject(verify_hidden_send(
            &fixture.verifier_plan,
            &messages,
            &fixture.witness,
        ));
        // The protocol-shaped entry also rejects this message: the changed
        // answer moves the chain value every later challenge hangs on, while
        // the message still carries the honest transcript value. A prover that
        // recomputed the complete post-answer transcript would produce a
        // well-formed message whose sample claim stays pending; only the later
        // proximity test would decide it, and that path is out of this unit.
        assert_reject(verify_hidden_send_public(&fixture.verifier_plan, &messages));
        assert!(messages.samples.answers[0] != honest);
    }

    #[test]
    fn changed_seed_or_position_is_rejected() {
        let fixture = fixture();
        let mut messages = fixture.messages.clone();
        messages.samples.seed[0].z[0] = messages.samples.seed[0].z[0].add(&Fp3::ONE);
        assert_reject(verify_hidden_send(
            &fixture.verifier_plan,
            &messages,
            &fixture.witness,
        ));

        let mut messages = fixture.messages.clone();
        messages.positions[0][0] = messages.positions[0][0].wrapping_add(1);
        assert_reject(verify_hidden_send(
            &fixture.verifier_plan,
            &messages,
            &fixture.witness,
        ));

        // A swapped group position is not the planned position.
        let mut messages = fixture.messages.clone();
        messages.group.entries[0].position = 1;
        assert_reject(verify_hidden_send(
            &fixture.verifier_plan,
            &messages,
            &fixture.witness,
        ));
    }

    #[test]
    fn missing_sample_is_rejected() {
        let fixture = fixture();
        let mut messages = fixture.messages.clone();
        messages.samples.answers.pop();
        assert_reject(verify_hidden_send(
            &fixture.verifier_plan,
            &messages,
            &fixture.witness,
        ));

        let mut messages = fixture.messages.clone();
        messages.samples.seed.clear();
        assert_reject(verify_hidden_send(
            &fixture.verifier_plan,
            &messages,
            &fixture.witness,
        ));
    }

    #[test]
    fn wrong_row_or_root_is_rejected() {
        let fixture = fixture();
        let mut messages = fixture.messages.clone();
        messages.openings[0].columns[0].symbols[0] = Fp::from_u64(0xDEAD);
        assert_reject(verify_hidden_send(
            &fixture.verifier_plan,
            &messages,
            &fixture.witness,
        ));

        let mut messages = fixture.messages.clone();
        messages.group.entries[0].root = Digest([0x66; 32]);
        assert_reject(verify_hidden_send(
            &fixture.verifier_plan,
            &messages,
            &fixture.witness,
        ));

        // A witness row that does not agree with the public answer is a wrong
        // row for the sample form.
        let mut witness = fixture.witness.clone();
        witness.messages[0][0][0] = witness.messages[0][0][0].add(&Fp3::ONE);
        assert_reject(verify_hidden_send(
            &fixture.verifier_plan,
            &fixture.messages,
            &witness,
        ));
    }

    #[test]
    fn canonical_messages_round_trip_and_reject_tampering() {
        let fixture = fixture();
        let bytes = fixture.messages.encode().unwrap();
        let decoded = HiddenSendMessages::decode(&fixture.verifier_plan, None, None, &bytes).unwrap();
        assert_eq!(decoded, fixture.messages);
        verify_hidden_send(&fixture.verifier_plan, &decoded, &fixture.witness).unwrap();

        // A tampered byte is rejected by decoding or by verification.
        let mut tampered = bytes.clone();
        let at = tampered.len() / 2;
        tampered[at] ^= 1;
        let accepted = HiddenSendMessages::decode(&fixture.verifier_plan, None, None, &tampered)
            .and_then(|message| {
                verify_hidden_send(&fixture.verifier_plan, &message, &fixture.witness)
            })
            .is_ok();
        assert!(!accepted, "a tampered canonical message was accepted");

        // The group and the sample message have their own canonical bytes.
        let group = fixture.messages.group.encode();
        assert_eq!(SendGroup::decode(&group).unwrap(), fixture.messages.group);
        let samples = fixture.messages.samples.encode();
        assert_eq!(
            SendSamples::decode(&samples).unwrap(),
            fixture.messages.samples
        );
        let mut tampered = samples.clone();
        tampered[8] = 0;
        tampered[9] = 0;
        tampered[10] = 0;
        tampered[11] = 0;
        tampered[12] = 0;
        tampered[13] = 0;
        tampered[14] = 0;
        tampered[15] = 0;
        assert!(SendSamples::decode(&tampered).is_err());
    }

    #[test]
    fn changed_encoded_table_identity_is_rejected() {
        let fixture = fixture();
        let mut plan = fixture.plan.clone();
        plan.table_identity = Digest([0x55; 32]);
        plan.expected_roots = Some(fixture.witness.roots.clone());
        assert_reject(verify_hidden_send(
            &plan,
            &fixture.messages,
            &fixture.witness,
        ));
        // The prover also rejects an encoded table that is not the planned one.
        let mut source = VecTableSource::new(table_messages(), table_tails()).unwrap();
        assert_prove_reject(prove_hidden_send(
            &plan,
            &fixture.activation,
            &fixture.reader,
            &mut source,
            &fixture.randomness,
        ));
    }

    #[test]
    fn derived_rows_follow_the_matrix_mapping() {
        let fixture = fixture();
        let mut source = VecTableSource::new(table_messages(), table_tails()).unwrap();
        let rows = derive_w_rows(
            &fixture.plan,
            &fixture.activation,
            &fixture.reader,
            &mut source,
        )
        .unwrap();
        assert_eq!(rows.len(), POSITIONS as usize);
        let messages = table_messages();
        let signed = activation_rows();
        let main_rows = w_shape().main_rows as usize;
        for position in 0..POSITIONS as usize {
            for table_row in 0..TABLE_ROWS as usize {
                for low in 0..(1usize << K_LO) {
                    let mut expected = Fp3::ZERO;
                    for y in 0..TABLE_ROWS as usize {
                        let weight =
                            Fp3::from_base(fp_from_signed_word(signed[position][y] as u64));
                        let value = Fp3::from_base(messages[table_row][low + (y << K_LO)]);
                        expected = expected.add(&weight.mul(&value));
                    }
                    let column = low + (table_row << K_LO);
                    assert_eq!(rows[position][column], expected);
                    // The committed W message holds the same entry at its
                    // layout place.
                    assert_eq!(
                        fixture.witness.messages[position][column % main_rows]
                            [column / main_rows],
                        expected
                    );
                }
            }
        }
        // X contributes the row-weight vector directly.
        let weights = fixture.activation.weights(1, TABLE_ROWS).unwrap();
        assert_eq!(weights.len(), TABLE_ROWS as usize);
        assert_eq!(
            weights[0],
            Fp3::from_base(fp_from_signed_word(activation_rows()[1][0] as u64))
        );
    }

    #[test]
    fn changed_eval_target_or_chain_value_is_rejected() {
        let fixture = fixture();
        let mut messages = fixture.messages.clone();
        messages.eval_targets[0] = messages.eval_targets[0].add(&Fp3::ONE);
        assert_reject(verify_hidden_send(
            &fixture.verifier_plan,
            &messages,
            &fixture.witness,
        ));

        let mut messages = fixture.messages.clone();
        messages.chain_value = Digest([0x44; 32]);
        assert_reject(verify_hidden_send(
            &fixture.verifier_plan,
            &messages,
            &fixture.witness,
        ));
    }

    #[test]
    fn full_matrix_w_shape_fits_the_large_message() {
        let code = build_code_plan(17, 65539, 8192).unwrap();
        let shape = WShape::new(
            1u64 << 16,
            code,
            1,
            16,
            SALT,
            TREE_NUMBER,
            LEAF_CLASS,
            W_RUN,
            W_CONFIGURATION,
        )
        .unwrap();
        assert_eq!(shape.main_rows, 1);
        assert!(shape.fault(1u64 << 16, 16).is_none());
        assert_eq!(shape.code.ell - 1, 65538);
    }


    fn group_table_tree_plan(tree_number: u64, main_rows: u64) -> RuntimeTreePlan {
        RuntimeTreePlan {
            tree_number,
            leaf_class: 1,
            log_n: W_CODE_LOG_N,
            log_n_zk: W_CODE_LOG_N,
            parts: vec![RuntimeTreePart {
                main_symbols: main_rows * E_DEGREE as u64,
                mask_symbols: 0,
            }],
            salt_bytes: SALT,
            run: W_RUN,
            configuration: W_CONFIGURATION,
        }
    }

    fn group_table_member(main_rows: u64, ell: u64) -> ProximityMember {
        let code = build_code_plan(3, ell, 4).unwrap();
        let message = (0..main_rows)
            .map(|row| {
                (0..ell)
                    .map(|place| Fp3::from_u64(row * ell + place + 1))
                    .collect()
            })
            .collect::<Vec<Vec<Fp3>>>();
        ProximityMember {
            bat_oracle: 1,
            source_oracle: 1,
            main_code: code,
            mask_code: code,
            main_rows,
            mask_rows: 0,
            answers: 1,
            root: Digest::ZERO,
            tree_plan: group_table_tree_plan(500, main_rows),
            message,
            tails: Vec::new(),
            mask_message: Vec::new(),
            mask_tails: Vec::new(),
        }
    }

    fn group_table_layout(
        main_rows: u64,
        log_ell: u32,
    ) -> (FoldGroupLayout, BatGroupLayout) {
        let k = 1;
        (
            FoldGroupLayout {
                group: 7,
                k,
                h: 1,
                members: vec![FoldMemberLayout {
                    oracle: 1,
                    main_rows,
                    mask_rows: 0,
                    kept: false,
                }],
            },
            BatGroupLayout {
                group: 7,
                k,
                h: 1,
                log_ell,
                log_ell_zk: 2,
                members: vec![BatMemberLayout {
                    oracle: 1,
                    kept: false,
                    main_first: Some(0),
                    mask_first: None,
                }],
            },
        )
    }

    #[test]
    fn group_table_rejects_malformed_members_and_cross_block_placement() {
        let member = group_table_member(2, 4);
        let (layout, bat_layout) = group_table_layout(2, 2);
        let serial = build_group_table_with_min(
            &layout,
            &bat_layout,
            std::slice::from_ref(&member),
            usize::MAX,
        )
        .unwrap();
        for threads in [1usize, 2, 64] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let parallel = pool.install(|| {
                build_group_table_with_min(&layout, &bat_layout, std::slice::from_ref(&member), 0)
                    .unwrap()
            });
            assert_eq!(serial, parallel, "threads {threads}");
        }
        assert_eq!(serial.len(), 8);
        assert_eq!(serial[0], member.message[0][0]);
        assert_eq!(serial[1], member.message[1][0]);
        assert_eq!(serial[6], member.message[0][3]);
        assert_eq!(serial[7], member.message[1][3]);

        let layout_second = FoldGroupLayout {
            group: 7,
            k: 2,
            h: 1,
            members: vec![FoldMemberLayout {
                oracle: 1,
                main_rows: 2,
                mask_rows: 0,
                kept: false,
            }],
        };
        let bat_second = BatGroupLayout {
            group: 7,
            k: 2,
            h: 1,
            log_ell: 1,
            log_ell_zk: 2,
            members: vec![BatMemberLayout {
                oracle: 1,
                kept: false,
                main_first: Some(0),
                mask_first: None,
            }],
        };
        let serial_second = build_group_table_with_min(
            &layout_second,
            &bat_second,
            std::slice::from_ref(&member),
            usize::MAX,
        )
        .unwrap();
        for threads in [1usize, 2, 64] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let parallel_second = pool.install(|| {
                build_group_table_with_min(
                    &layout_second,
                    &bat_second,
                    std::slice::from_ref(&member),
                    0,
                )
                .unwrap()
            });
            assert_eq!(serial_second, parallel_second, "second shape threads {threads}");
        }

        let mut too_long = member.clone();
        too_long.message[0].push(Fp3::from_u64(99));
        assert!(matches!(
            build_group_table_with_min(&layout, &bat_layout, &[too_long], 0),
            Err(Error::ShapeMismatch)
        ));

        let mut short = member.clone();
        short.message.pop();
        assert!(matches!(
            build_group_table_with_min(&layout, &bat_layout, &[short], 0),
            Err(Error::ShapeMismatch)
        ));

        let empty_layout = FoldGroupLayout {
            group: 7,
            k: 1,
            h: 1,
            members: Vec::new(),
        };
        let empty_bat = BatGroupLayout {
            group: 7,
            k: 1,
            h: 1,
            log_ell: 2,
            log_ell_zk: 2,
            members: Vec::new(),
        };
        assert!(matches!(
            build_group_table_with_min(&empty_layout, &empty_bat, &[], 0),
            Err(Error::ShapeMismatch)
        ));

        let crossing = group_table_member(3, 4);
        let (layout3, bat3) = group_table_layout(3, 2);
        assert!(matches!(
            build_group_table_with_min(&layout3, &bat3, &[crossing], 0),
            Err(Error::InvalidIndex)
        ));

        let mut overflow = bat_layout.clone();
        overflow.k = 64;
        assert!(matches!(
            build_group_table_with_min(&layout, &overflow, &[member], 0),
            Err(Error::ResourceLimit)
        ));
    }

    #[test]
    fn changed_message_with_original_tree_changes_the_group_table() {
        let mut member = group_table_member(2, 4);
        let plan = group_table_tree_plan(500, 2);
        let tails = vec![vec![Fp3::ZERO; 4]; 2];
        let salts = vec![vec![0u8; SALT]; 1 << W_CODE_LOG_N];
        let tree = commit_rows_tree(
            plan,
            member.main_code,
            &member.message,
            &tails,
            &salts,
        )
        .unwrap();
        member.root = *tree.root();
        let trees = RetainedTrees(vec![Arc::new(tree)]);
        let (layout, bat_layout) = group_table_layout(2, 2);
        let original = build_group_table_with_min(
            &layout,
            &bat_layout,
            std::slice::from_ref(&member),
            usize::MAX,
        )
        .unwrap();
        assert!(retained_member_tree(&trees, 0, &member).is_ok());
        assert!(ProximityMemberProver::sealed(
            member.clone(),
            Arc::clone(&trees.0[0])
        )
        .is_ok());

        // Keep the original tree and root; change only the prover-side message.
        let mut changed = member;
        changed.message[0][0] = changed.message[0][0].add(&Fp3::ONE);
        assert!(retained_member_tree(&trees, 0, &changed).is_ok());
        let changed_table = build_group_table_with_min(
            &layout,
            &bat_layout,
            std::slice::from_ref(&changed),
            usize::MAX,
        )
        .unwrap();
        assert_ne!(original, changed_table);
    }

    #[test]
    fn retained_tree_checks_plan_and_sealed_root() {
        let mut member = group_table_member(2, 4);
        let plan = group_table_tree_plan(500, 2);
        let tails = vec![vec![Fp3::ZERO; 4]; 2];
        let salts = vec![vec![0u8; SALT]; 1 << W_CODE_LOG_N];
        let tree = commit_rows_tree(
            plan.clone(),
            member.main_code,
            &member.message,
            &tails,
            &salts,
        )
        .unwrap();
        member.root = *tree.root();
        let trees = RetainedTrees(vec![Arc::new(tree)]);
        assert!(retained_member_tree(&trees, 0, &member).is_ok());

        let mut wrong_plan = member.clone();
        wrong_plan.tree_plan.tree_number += 1;
        assert!(matches!(
            retained_member_tree(&trees, 0, &wrong_plan),
            Err(Error::StatementMismatch)
        ));

        let mut wrong_root = member.clone();
        wrong_root.root.0[0] ^= 1;
        assert!(matches!(
            ProximityMemberProver::sealed(wrong_root, Arc::clone(&trees.0[0])),
            Err(Error::StatementMismatch)
        ));

        let mut wrong_message = member.clone();
        wrong_message.message[0][0] = Fp3::from_u64(999);
        let other_tree = commit_rows_tree(
            plan,
            wrong_message.main_code,
            &wrong_message.message,
            &tails,
            &salts,
        )
        .unwrap();
        assert!(matches!(
            ProximityMemberProver::sealed(wrong_message, Arc::new(other_tree)),
            Err(Error::StatementMismatch)
        ));

        let other_plan = group_table_tree_plan(501, 2);
        let other_tree = commit_rows_tree(
            other_plan,
            member.main_code,
            &member.message,
            &tails,
            &salts,
        )
        .unwrap();
        let other_trees = RetainedTrees(vec![Arc::new(other_tree)]);
        assert!(matches!(
            retained_member_tree(&other_trees, 0, &member),
            Err(Error::StatementMismatch)
        ));
    }
}


#[cfg(test)]
mod merged_tests {
    use super::*;
    use std::cell::RefCell;

    use crate::proof::column_session::fp_from_signed_word;

    use zkie_engine::pcs::request::table::{
        commit_table_streaming, StreamingTable, TableDistances, VecTableSource,
    };

    const POSITIONS: u32 = 2;
    const HIDDEN: u32 = 2;
    const MATRIX_ROWS: u64 = 2;
    const K_LO: u32 = 1;
    const COLUMNS: u64 = 4;
    const TABLE_ROWS: u64 = 2;
    const TABLE_LOG_N: u32 = 4;
    const TABLE_ELL: u64 = 8;
    const TABLE_R: u64 = 8;
    const CODE_LOG_N: u32 = 8;
    const CODE_ELL: u64 = 32;
    const CODE_R: u64 = 4;
    const INDEX_MASK_ORACLE: u64 = 30;
    const INDEX_MASK_TREE: u64 = 202;
    const INDEX_MASK_RUN: Digest = Digest([0x66; 32]);
    const SALT: usize = LEAF_SALT_MIN_BYTES;
    const GROUP: u64 = 7;
    const ORACLE_BASE: u64 = 20;
    const TABLE_CONFIGURATION: Digest = Digest([0x21; 32]);
    const ACTIVATION_CONFIGURATION: Digest = Digest([0x31; 32]);
    const W_CONFIGURATION: Digest = Digest([0x51; 32]);
    const W_RUN: Digest = Digest([0x41; 32]);
    const GU_ORACLE: u64 = 10;
    const OMEGA_ORACLE: u64 = 11;
    const GU_GROUP: u64 = 5;

    fn table_code() -> CodeParams {
        build_code_plan(TABLE_LOG_N, TABLE_ELL, TABLE_R).unwrap()
    }

    fn code() -> CodeParams {
        build_code_plan(CODE_LOG_N, CODE_ELL, CODE_R).unwrap()
    }

    fn mask_code() -> CodeParams {
        build_code_plan(CODE_LOG_N, CODE_ELL, CODE_R).unwrap()
    }

    fn table_shape() -> IndexShape {
        IndexShape {
            code: table_code(),
            v: 1,
            answers: 1,
            salt_bytes: SALT,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        }
    }

    fn w_shape() -> WShape {
        WShape::new(
            COLUMNS,
            code(),
            1,
            POSITIONS,
            SALT,
            100,
            1,
            W_RUN,
            W_CONFIGURATION,
        )
        .unwrap()
    }

    fn tree_plan(tree_number: u64, leaf_class: u32, main_rows: u64) -> RuntimeTreePlan {
        RuntimeTreePlan {
            tree_number,
            leaf_class,
            log_n: CODE_LOG_N,
            log_n_zk: CODE_LOG_N,
            parts: vec![RuntimeTreePart {
                main_symbols: main_rows * E_DEGREE as u64,
                mask_symbols: 0,
            }],
            salt_bytes: SALT,
            run: Digest([0x61; 32]),
            configuration: Digest([0x62; 32]),
        }
    }

    fn gu_shape() -> GuShape {
        GuShape {
            oracle: GU_ORACLE,
            group: GU_GROUP,
            tree: tree_plan(300, 2, 1),
            part: 0,
            rows_high: true,
            u: GuOracleShape {
                code: code(),
                main_rows: 1,
                main_deg: E_DEGREE as u32,
                mask_rows: 0,
                mask_code: None,
                answers: 1,
            },
            index: table_shape(),
        }
    }

    fn omega_shape() -> OmegaShapePlan {
        OmegaShapePlan {
            oracle: OMEGA_ORACLE,
            group: GU_GROUP,
            tree: tree_plan(301, 2, 1),
            u: GuOracleShape {
                code: code(),
                main_rows: 1,
                main_deg: E_DEGREE as u32,
                mask_rows: 0,
                mask_code: None,
                answers: 1,
            },
            rows_high: true,
            matrix_rows: MATRIX_ROWS,
            k_lo: K_LO,
        }
    }

    fn mask_tree(tree_number: u64) -> MaskTreePlan {
        MaskTreePlan {
            tree_number,
            leaf_class: 3,
            log_n: CODE_LOG_N,
            mask_rows: 1,
            salt_bytes: SALT,
            run: Digest([0x63; 32]),
            configuration: Digest([0x64; 32]),
        }
    }

    fn index_mask_plan() -> IndexMaskPlan {
        IndexMaskPlan {
            oracle: INDEX_MASK_ORACLE,
            group: GROUP,
            mask_code: mask_code(),
            mask_rows: 4,
            answers: 1,
            openings: 1,
            tree: RuntimeTreePlan {
                tree_number: INDEX_MASK_TREE,
                leaf_class: 1,
                log_n: CODE_LOG_N,
                log_n_zk: CODE_LOG_N,
                parts: vec![RuntimeTreePart {
                    main_symbols: 0,
                    mask_symbols: 4 * E_DEGREE as u64,
                }],
                salt_bytes: SALT,
                run: INDEX_MASK_RUN,
                configuration: W_CONFIGURATION,
            },
            entries: MaskEntryAllocation::for_variables(
                MATRIX_ROWS.trailing_zeros() as usize,
                POSITIONS,
            )
            .unwrap(),
        }
    }

    fn dot_plan() -> DotMapPlan {
        DotMapPlan {
            oracle: 12,
            group: GROUP,
            shape: WShape::new(
                POSITIONS as u64 * HIDDEN as u64,
                code(),
                1,
                POSITIONS,
                SALT,
                400,
                2,
                Digest([0x67; 32]),
                W_CONFIGURATION,
            )
            .unwrap(),
            hidden: HIDDEN,
        }
    }

    fn activation_rows() -> Vec<Vec<i64>> {
        vec![vec![1, 2], vec![0, -2]]
    }

    fn activation_salts() -> Vec<Vec<u8>> {
        (0..POSITIONS)
            .map(|position| vec![(position as u8).wrapping_mul(7) ^ 0x3C; SALT])
            .collect()
    }

    fn activation() -> CommittedActivation {
        CommittedActivation::commit_signed(
            ActivationShape::new(POSITIONS, HIDDEN, SALT).unwrap(),
            &activation_rows(),
            &activation_salts(),
            ACTIVATION_CONFIGURATION,
        )
        .unwrap()
    }

    fn table_messages() -> Vec<Vec<Fp>> {
        let rows = table_shape().rows() as usize;
        let ell = TABLE_ELL as usize;
        let mut messages = vec![vec![Fp::ZERO; ell]; rows];
        for (row, message) in messages.iter_mut().enumerate() {
            for entry in 0..COLUMNS as usize {
                let word = ((row as i64 * 13 + entry as i64 * 5) % 11) - 5;
                message[entry] = fp_from_signed_word(word as u64);
            }
        }
        messages
    }

    fn table_tails() -> Vec<Vec<Fp>> {
        vec![vec![Fp::ZERO; TABLE_R as usize]; TABLE_ROWS as usize]
    }

    struct Table {
        table: StreamingTable,
        source: RefCell<VecTableSource>,
        reader: IndexReader,
    }

    impl TableRowSource for Table {
        fn rows(&self) -> u64 {
            self.source.borrow().rows()
        }

        fn row(&mut self, index: u64, message: &mut [Fp], tail: &mut [Fp]) -> Result<()> {
            self.source.borrow_mut().row(index, message, tail)
        }
    }

    impl HiddenTableProver for Table {
        fn reader(&self) -> &IndexReader {
            &self.reader
        }

        fn open_columns(&mut self, positions: &[u64]) -> Result<Vec<ColumnOpening>> {
            let mut source = self.source.borrow_mut();
            self.table.open_columns(&mut *source, positions)
        }
    }

    fn table() -> Table {
        let messages = table_messages();
        let tails = table_tails();
        let mut source = VecTableSource::new(messages, tails).unwrap();
        let salts: Vec<Vec<u8>> = (0..table_code().n())
            .map(|position| vec![(position as u8) ^ 0x5A; SALT])
            .collect();
        let table = commit_table_streaming(
            &mut source,
            table_shape(),
            TABLE_CONFIGURATION,
            &salts,
            64,
            u64::MAX,
        )
        .unwrap();
        let reader = table.reader.clone();
        Table {
            table,
            source: RefCell::new(source),
            reader,
        }
    }

    fn output_values(activation: &CommittedActivation) -> Vec<Vec<Fp3>> {
        let messages = table_messages();
        let mut output = vec![vec![Fp3::ZERO; COLUMNS as usize]; POSITIONS as usize];
        let weights: Vec<Vec<Fp3>> = (0..POSITIONS)
            .map(|position| activation.weights(position, MATRIX_ROWS).unwrap())
            .collect();
        for position in 0..POSITIONS as usize {
            for column in 0..COLUMNS as usize {
                let table_row = column as u64 >> K_LO;
                let low = column as u64 & ((1u64 << K_LO) - 1);
                let mut sum = Fp3::ZERO;
                for y in 0..MATRIX_ROWS as usize {
                    let entry = (low + (y as u64) * (1u64 << K_LO)) as usize;
                    let value = Fp3::from_base(messages[table_row as usize][entry]);
                    sum = sum.add(&weights[position][y].mul(&value));
                }
                output[position][column] = sum;
            }
        }
        output
    }

    fn plan(activation: &CommittedActivation) -> HiddenLinPlan {
        let table = table();
        let w = w_shape();
        let send = HiddenSendPlan {
            activation: activation.shape(),
            activation_configuration: ACTIVATION_CONFIGURATION,
            activation_root: activation.root(),
            activation_source: ActivationSource::Commitment,
            table_shape: table_shape(),
            table_identity: reader_table_identity(&table.reader).unwrap(),
            matrix_rows: MATRIX_ROWS,
            k_lo: K_LO,
            columns: COLUMNS,
            group: GROUP,
            oracle_base: ORACLE_BASE,
            openings_per_oracle: 2,
            w,
            index_mask: None,
            dot: None,
            expected_dot_root: None,
            expected_roots: None,
        };
        let weights: Vec<Vec<Fp3>> = (0..POSITIONS)
            .map(|position| activation.weights(position, MATRIX_ROWS).unwrap())
            .collect();
        HiddenLinPlan {
        norm: None,
            send,
            output: HiddenOutputShape {
                positions: POSITIONS,
                columns: COLUMNS,
            },
            masks: vec![Vec::new(); POSITIONS as usize],
            weights,
            proximity: HiddenProximityPlan {
                gu: gu_shape(),
                omega: omega_shape(),
                code: code(),
                mask_code: mask_code(),
                shared_mask_tree: mask_tree(200),
                own_mask_tree: mask_tree(201),
                shared_fresh: 50,
                own_fresh: 51,
                column_draws: 2,
                column_budget: 4,
                column_work_bits: 0,
                shared_base_positions: 2,
                own_base_positions: 2,
                shared_stages: Vec::new(),
                own_stages: Vec::new(),
                planner_checked: false,
            },
            output_relation: None,
            closure: None,
        }
    }

    fn run() -> (HiddenLinPlan, HiddenLinMessages, HiddenLinWitness) {
        let activation = activation();
        let output = output_values(&activation);
        let mut table = table();
        let plan = plan(&activation);
        let (messages, witness) =
            prove_hidden_lin(&plan, &activation, &mut table, &output, Digest([0x71; 32])).unwrap();
        let mut verifier_plan = plan;
        verifier_plan.send.expected_roots = Some(witness.send.roots.clone());
        (verifier_plan, messages, witness)
    }

    fn hidden_plan(activation: &CommittedActivation) -> HiddenLinPlan {
        let mut plan = plan(activation);
        plan.weights = Vec::new();
        plan.send.index_mask = Some(index_mask_plan());
        plan.send.dot = Some(dot_plan());
        plan.proximity.own_mask_tree.mask_rows = 4;
        plan
    }

    /// One two-stage chain on the toy group's own codes. The stage list is
    /// plan data: it derives a loop chain but no proof is made against it.
    fn norm_group(columns: u64, group: u64, oracle: u64, tree: u64) -> crate::proof::hidden_norm_psc::NormGroupPlan {
        crate::proof::hidden_norm_psc::NormGroupPlan {
            shape: WShape::new(
                columns,
                code(),
                1,
                POSITIONS,
                SALT,
                tree,
                1,
                W_RUN,
                W_CONFIGURATION,
            )
            .unwrap(),
            columns,
            group,
            oracle_base: oracle,
            openings: 1,
        }
    }

    fn norm_plan() -> crate::proof::hidden_norm_psc::NormPlan {
        let columns = COLUMNS;
        crate::proof::hidden_norm_psc::NormPlan {
            shape: crate::proof::hidden_norm::NormShape::new(POSITIONS, HIDDEN, HIDDEN)
                .unwrap(),
            columns,
            variables: columns.trailing_zeros() as usize + 1,
            relations: vec![0, 1, 2, 3],
            x: norm_group(columns, 80, 9000, 60000),
            x_roots: (0..POSITIONS)
                .map(|position| Digest([0x90 + position as u8; 32]))
                .collect(),
            sum: norm_group(1, 81, 9100, 60100),
            squares: norm_group(1, 82, 9200, 60200),
            mean: norm_group(1, 83, 9300, 60300),
            variance: norm_group(1, 84, 9400, 60400),
            round: None,
            lookup: None,
            rstd: None,
            params: None,
            output: None,
            mask: None,
        }
    }

    fn norm_mask_plan() -> crate::proof::hidden_norm_psc::NormMaskPlan {
        crate::proof::hidden_norm_psc::NormMaskPlan {
            oracle: 9500,
            group: 85,
            mask_code: mask_code(),
            mask_rows: 4,
            answers: 1,
            openings: 1,
            tree: zkie_engine::pcs::request::runtime_tree::RuntimeTreePlan {
                tree_number: 60500,
                leaf_class: 1,
                log_n: CODE_LOG_N,
                log_n_zk: CODE_LOG_N,
                parts: vec![zkie_engine::pcs::request::runtime_tree::RuntimeTreePart {
                    main_symbols: 0,
                    mask_symbols: 12,
                }],
                salt_bytes: SALT,
                run: W_RUN,
                configuration: W_CONFIGURATION,
            },
            positions: POSITIONS,
            hiding: 9,
            shown: 2,
            claims: 1,
            pointwise: 0,
            pointwise_hiding: crate::proof::hidden_norm_psc::norm_pointwise_masks(3)
                .unwrap()
                .0,
            pointwise_shown: crate::proof::hidden_norm_psc::norm_pointwise_masks(3)
                .unwrap()
                .1,
            tail: 0,
            tail_hiding: crate::proof::hidden_norm_psc::norm_tail_mask_masks(4)
                .unwrap()
                .0,
            tail_shown: crate::proof::hidden_norm_psc::norm_tail_mask_masks(4)
                .unwrap()
                .1,
        }
    }

    #[test]
    fn norm_groups_are_counted_in_the_proximity_bound() {
        use crate::proof::proximity_iopp::proximity_bound_ids;
        let activation = activation();
        let plan = plan(&activation);
        let positions = plan.send.positions() as u64;
        let without = proximity_bound_ids(&plan.proximity, &plan.send, None, None).unwrap();
        let mut norm = norm_plan();
        for group in [
            &mut norm.x,
            &mut norm.sum,
            &mut norm.squares,
            &mut norm.mean,
            &mut norm.variance,
        ] {
            group.group = plan.send.group;
        }
        norm.validate().unwrap();
        let mut integrated = plan.clone();
        integrated.norm = Some(norm.clone());
        integrated.validate().unwrap();
        let with =
            proximity_bound_ids(&plan.proximity, &plan.send, None, Some(&norm)).unwrap();

        let groups = [&norm.x, &norm.sum, &norm.squares, &norm.mean, &norm.variance];
        for group in groups {
            for position in 0..positions {
                let oracle = group.oracle_base + position;
                assert!(!without.oracles.contains(&oracle));
                assert!(with.oracles.contains(&oracle));
                let tree = group.shape.tree_number + position;
                assert!(!without.trees.contains(&tree));
                assert!(with.trees.contains(&tree));
            }
        }
        let last_member = positions + 1 + 5 * positions - 1;
        assert!(!without.oracles.contains(&last_member));
        assert!(with.oracles.contains(&last_member));

        let mut masked = norm.clone();
        masked.mask = Some(norm_mask_plan());
        masked.validate().unwrap();
        let with_mask =
            proximity_bound_ids(&plan.proximity, &plan.send, None, Some(&masked)).unwrap();
        assert!(with_mask.oracles.contains(&(last_member + 1)));
        assert!(with_mask.oracles.contains(&9500));
        assert!(with_mask.trees.contains(&60500));
        assert!(!with.oracles.contains(&9500));
        assert!(!with.trees.contains(&60500));

        let mut round_planned = norm.clone();
        let widths = crate::proof::hidden_norm_psc::NormRoundWidths {
            sum: 1,
            squares: 1,
            variance: 1,
            mean: 1,
            offset: 1,
        };
        let layout =
            crate::proof::hidden_norm_psc::NormRoundLayout::new(widths, HIDDEN, 8, 4)
                .unwrap();
        let columns = widths.columns().unwrap();
        round_planned.round = Some(crate::proof::hidden_norm_psc::NormRoundPlan {
            group: norm_group(columns, GROUP, 9600, 60600),
            layout,
            input_bound: 2,
            table_real_len: 8,
        });
        round_planned.validate().unwrap();
        let with_round =
            proximity_bound_ids(&plan.proximity, &plan.send, None, Some(&round_planned))
                .unwrap();
        assert!(with_round.oracles.contains(&9600));
        assert!(with_round.trees.contains(&60600));
        assert!(!with.oracles.contains(&9600));
        assert!(!with.trees.contains(&60600));
    }

    fn norm_values(row: &[i64], columns: u64) -> Vec<Fp3> {
        let mut values = vec![Fp3::ZERO; columns as usize];
        for (slot, value) in values.iter_mut().zip(row.iter()) {
            *slot = crate::proof::output_closure::fp3_signed(*value as i128).unwrap();
        }
        values
    }

    fn draw_norm_tails(shape: &WShape, positions: usize, coins: &mut ByteMaskCoins) -> Vec<Vec<Fp3>> {
        let rows = shape.main_rows as usize;
        let tail = shape.code.r as usize;
        (0..positions)
            .map(|_| (0..rows * tail).map(|_| coins.draw().unwrap().0).collect())
            .collect()
    }

    fn draw_norm_pads(shape: &WShape, positions: usize, coins: &mut ByteMaskCoins) -> Vec<Fp3> {
        let answers = shape.answers as usize;
        let mut pads = Vec::with_capacity(positions * answers);
        for _ in 0..positions * answers {
            let mut pad = Fp3::ZERO;
            for _ in 0..64 {
                pad = coins.draw().unwrap().0;
                if pad != Fp3::ZERO {
                    break;
                }
            }
            assert_ne!(pad, Fp3::ZERO);
            pads.push(pad);
        }
        pads
    }

    fn draw_norm_salts(
        shape: &WShape,
        positions: usize,
        coins: &mut ByteMaskCoins,
    ) -> Vec<Vec<Vec<u8>>> {
        use zkie_core::common::field::PrimeField64;
        let columns = shape.code.n() as usize;
        (0..positions)
            .map(|_| {
                (0..columns)
                    .map(|_| {
                        let mut salt = Vec::with_capacity(shape.salt_bytes);
                        while salt.len() < shape.salt_bytes {
                            let (value, _) = coins.draw().unwrap();
                            for coefficient in value.c {
                                salt.extend_from_slice(
                                    &coefficient.as_canonical_u64().to_le_bytes(),
                                );
                            }
                        }
                        salt.truncate(shape.salt_bytes);
                        salt
                    })
                    .collect()
            })
            .collect()
    }

    fn commit_norm_group(
        group: &crate::proof::hidden_norm_psc::NormGroupPlan,
        values: &[Vec<Fp3>],
        coins: &mut ByteMaskCoins,
    ) -> crate::proof::hidden_norm_psc::NormGroupWitness {
        let positions = values.len();
        let pads = draw_norm_pads(&group.shape, positions, coins);
        let tails = draw_norm_tails(&group.shape, positions, coins);
        let salts = draw_norm_salts(&group.shape, positions, coins);
        let committed = commit_value_group(
            &group.shape,
            group.columns,
            group.oracle_base,
            group.group,
            values,
            &pads,
            &tails,
            &salts,
            positions as u32,
        )
        .unwrap();
        crate::proof::hidden_norm_psc::NormGroupWitness {
            messages: committed.messages.clone(),
            tails,
            salts,
            pads,
            committed,
        }
    }

    fn norm_fixture(
        _activation: &CommittedActivation,
    ) -> (
        HiddenLinPlan,
        crate::proof::hidden_norm_psc::NormWitness,
    ) {
        use crate::proof::hidden_norm_psc::{
            commit_norm_mask, norm_mask_entries, NormGroupPlan, NormMaskPlan, NormMaskWitness,
            NormPlan, NormRoundLayout, NormRoundPlan, NormRoundWidths, NormWitness,
        };
        use crate::proof::hidden_norm::NormCoreRow;

        let rows = activation_rows();
        let core = rows
            .iter()
            .map(|row| NormCoreRow::from_values(row, HIDDEN as usize).unwrap())
            .collect::<Vec<_>>();
        let columns = COLUMNS;
        let group = |columns: u64, id: u64, oracle: u64, tree: u64| NormGroupPlan {
            shape: WShape::new(
                columns,
                code(),
                1,
                POSITIONS,
                SALT,
                tree,
                1,
                W_RUN,
                W_CONFIGURATION,
            )
            .unwrap(),
            columns,
            group: id,
            oracle_base: oracle,
            openings: 1,
        };
        let mask_plan = NormMaskPlan {
            oracle: 9500,
            group: GROUP,
            mask_code: mask_code(),
            mask_rows: 64,
            answers: 1,
            openings: 1,
            tree: RuntimeTreePlan {
                tree_number: 60500,
                leaf_class: 1,
                log_n: CODE_LOG_N,
                log_n_zk: CODE_LOG_N,
                parts: vec![RuntimeTreePart {
                    main_symbols: 0,
                    mask_symbols: 192,
                }],
                salt_bytes: SALT,
                run: W_RUN,
                configuration: W_CONFIGURATION,
            },
            positions: POSITIONS,
            hiding: 9,
            shown: 2,
            claims: 1,
            pointwise: 0,
            pointwise_hiding: crate::proof::hidden_norm_psc::norm_pointwise_masks(3)
                .unwrap()
                .0,
            pointwise_shown: crate::proof::hidden_norm_psc::norm_pointwise_masks(3)
                .unwrap()
                .1,
            tail: 0,
            tail_hiding: crate::proof::hidden_norm_psc::norm_tail_mask_masks(4)
                .unwrap()
                .0,
            tail_shown: crate::proof::hidden_norm_psc::norm_tail_mask_masks(4)
                .unwrap()
                .1,
        };
        let mut norm = NormPlan {
            shape: crate::proof::hidden_norm::NormShape::new(POSITIONS, HIDDEN, HIDDEN)
                .unwrap(),
            columns,
            variables: columns.trailing_zeros() as usize + 1,
            relations: vec![
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25,
            ],
            x: group(columns, GROUP, 9000, 60000),
            x_roots: Vec::new(),
            sum: group(1, GROUP, 9100, 60100),
            squares: group(1, GROUP, 9200, 60200),
            mean: group(1, GROUP, 9300, 60300),
            variance: group(1, GROUP, 9400, 60400),
            round: None,
            lookup: None,
            rstd: None,
            params: None,
            output: None,
            mask: Some(mask_plan.clone()),
        };
        let round_widths = NormRoundWidths {
            sum: 1,
            squares: 1,
            variance: 1,
            mean: 1,
            offset: 1,
        };
        let round_layout = NormRoundLayout::new(round_widths, HIDDEN, 8, 4).unwrap();
        let round_columns = round_widths.columns().unwrap();
        norm.round = Some(NormRoundPlan {
            group: group(round_columns, GROUP, 9600, 60600),
            layout: round_layout,
            input_bound: 2,
            table_real_len: 8,
        });
        let mut coins = ByteMaskCoins::new(Digest([0x91; 32])).unwrap();
        let x_values = rows
            .iter()
            .map(|row| norm_values(row, columns))
            .collect::<Vec<_>>();
        let x = commit_norm_group(&norm.x, &x_values, &mut coins);
        norm.x_roots = x.committed.roots.clone();
        let scalar_values = |value: &dyn Fn(&NormCoreRow) -> i128| {
            core.iter()
                .map(|row| {
                    vec![crate::proof::output_closure::fp3_signed(value(row)).unwrap()]
                })
                .collect::<Vec<_>>()
        };
        let sum = commit_norm_group(
            &norm.sum,
            &scalar_values(&|row| row.sum),
            &mut coins,
        );
        let squares = commit_norm_group(
            &norm.squares,
            &scalar_values(&|row| row.squares),
            &mut coins,
        );
        let mean = commit_norm_group(
            &norm.mean,
            &scalar_values(&|row| row.mean as i128),
            &mut coins,
        );
        let variance = commit_norm_group(
            &norm.variance,
            &scalar_values(&|row| row.variance as i128),
            &mut coins,
        );
        let table_real_len = norm
            .round
            .as_ref()
            .ok_or(Error::InvalidConfig)
            .unwrap()
            .table_real_len;
        let rsqrt_rows = core
            .iter()
            .map(|row| {
                crate::proof::hidden_norm::RsqrtRow::from_variance(
                    row.variance,
                    table_real_len,
                    (row.variance % 97) + 3,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let round = match &norm.round {
            Some(round_plan) => {
                let values = core
                    .iter()
                    .zip(&rsqrt_rows)
                    .map(|(row, rsqrt)| {
                        round_plan
                            .layout
                            .values_with_rsqrt(row, rsqrt, round_plan.group.columns)
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                Some(commit_norm_group(&round_plan.group, &values, &mut coins))
            }
            None => None,
        };
        let mask_rows = mask_plan.mask_rows as usize;
        let ell = mask_plan.mask_code.ell as usize;
        let mut mask_values = Vec::with_capacity(mask_rows);
        for _ in 0..mask_rows {
            mask_values.push(
                (0..ell)
                    .map(|_| coins.draw().unwrap().0)
                    .collect::<Vec<_>>(),
            );
        }
        let mut pad = Fp3::ZERO;
        while pad == Fp3::ZERO {
            pad = coins.draw().unwrap().0;
        }
        mask_values[mask_rows - 1][ell - 1] = pad;
        let mask_tails = (0..mask_rows)
            .map(|_| {
                (0..mask_plan.mask_code.r as usize)
                    .map(|_| coins.draw().unwrap().0)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mask_salts = (0..mask_plan.tree.columns() as usize)
            .map(|_| {
                let mut salt = Vec::with_capacity(mask_plan.tree.salt_bytes);
                while salt.len() < mask_plan.tree.salt_bytes {
                    let (value, _) = coins.draw().unwrap();
                    for coefficient in value.c {
                        salt.extend_from_slice(
                            &zkie_core::common::field::PrimeField64::as_canonical_u64(
                                &coefficient,
                            )
                            .to_le_bytes(),
                        );
                    }
                }
                salt.truncate(mask_plan.tree.salt_bytes);
                salt
            })
            .collect::<Vec<_>>();
        let committed_mask =
            commit_norm_mask(&mask_plan, mask_values, mask_tails, mask_salts).unwrap();
        let entries =
            norm_mask_entries(&mask_plan, &committed_mask, norm.relations.len()).unwrap();
        let mask_witness = NormMaskWitness {
            committed: committed_mask,
            entries,
        };
        let witness = NormWitness {
            rows,
            core,
            x,
            sum,
            squares,
            mean,
            variance,
            round,
            rsqrt: Some(rsqrt_rows),
            mask: Some(mask_witness),
            lookup: None,
            rstd: None,
            params: None,
            output: None,
        };
        let mut plan = hidden_plan(_activation);
        plan.norm = Some(norm);
        let index_rows = plan
            .send
            .index_mask
            .as_ref()
            .map(|index| index.tree_rows().unwrap())
            .unwrap_or(0);
        plan.proximity.own_mask_tree.mask_rows = crate::proof::hidden_norm_psc::norm_own_mask_rows(
            index_rows,
            mask_plan.mask_rows,
        )
        .unwrap();
        plan.validate().unwrap();
        (plan, witness)
    }

    struct RstdRows {
        identity: Digest,
        root: Digest,
        rows: Vec<i64>,
    }

    impl RstdRows {
        fn new(identity: u8, rows: Vec<i64>) -> Self {
            use crate::proof::lookup_table::{content_digest, TableEncoding};
            let root = content_digest(TableEncoding::I32Le, &rows).unwrap();
            Self {
                identity: Digest([identity; 32]),
                root,
                rows,
            }
        }

        fn config(&self) -> crate::proof::lookup_table::TableConfig {
            use crate::proof::lookup_table::{TableConfig, TableEncoding, TABLE_ENCODING_VERSION};
            TableConfig {
                identity: self.identity,
                content_root: self.root,
                real_len: self.rows.len(),
                encoding: TableEncoding::I32Le,
                version: TABLE_ENCODING_VERSION,
            }
        }
    }

    impl crate::proof::lookup_table::TableRowSource for RstdRows {
        fn identity(&self) -> Digest {
            self.identity
        }

        fn content_root(&self) -> Digest {
            self.root
        }

        fn real_len(&self) -> usize {
            self.rows.len()
        }

        fn encoding(&self) -> crate::proof::lookup_table::TableEncoding {
            crate::proof::lookup_table::TableEncoding::I32Le
        }

        fn read_rows(&self, start: usize, count: usize) -> Result<Vec<i64>> {
            let end = start.checked_add(count).ok_or(Error::ResourceLimit)?;
            if end > self.rows.len() {
                return Err(Error::ShapeMismatch);
            }
            Ok(self.rows[start..end].to_vec())
        }
    }

    struct ParamTables {
        gain: StreamingTable,
        gain_source: VecTableSource,
        bias: StreamingTable,
        bias_source: VecTableSource,
    }

    impl NormParamTableProver for ParamTables {
        fn source(&mut self, kind: NormParamKind) -> &mut dyn TableRowSource {
            match kind {
                NormParamKind::Gain => &mut self.gain_source,
                NormParamKind::Bias => &mut self.bias_source,
            }
        }

        fn open_columns(
            &mut self,
            kind: NormParamKind,
            positions: &[u64],
        ) -> Result<Vec<ColumnOpening>> {
            match kind {
                NormParamKind::Gain => self.gain.open_columns(&mut self.gain_source, positions),
                NormParamKind::Bias => self.bias.open_columns(&mut self.bias_source, positions),
            }
        }
    }

    fn param_table(
        values: &[i64],
        configuration: u8,
    ) -> (StreamingTable, VecTableSource, IndexReader) {
        use zkie_engine::pcs::request::code::build_code_plan;

        let shape = IndexShape {
            code: build_code_plan(4, 7, 2).unwrap(),
            v: 0,
            answers: 1,
            salt_bytes: SALT,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        };
        let ell = shape.code.ell as usize;
        let mut message = vec![fp_from_signed_word(0); ell];
        for (index, value) in values.iter().enumerate() {
            message[index] = fp_from_signed_word(*value as u64);
        }
        let tails = vec![vec![fp_from_signed_word(0); shape.code.r as usize]];
        let mut source = VecTableSource::new(vec![message], tails).unwrap();
        let salts = vec![vec![0u8; shape.salt_bytes]; shape.code.n() as usize];
        let table = commit_table_streaming(
            &mut source,
            shape,
            Digest([configuration; 32]),
            &salts,
            8,
            1 << 20,
        )
        .unwrap();
        let reader = table.reader.clone();
        (table, source, reader)
    }

    fn param_identity(
        kind: NormParamKind,
        reader: IndexReader,
        width: u64,
    ) -> crate::proof::norm_param_table::NormParamTableIdentity {
        use crate::proof::norm_param_table::NORM_PARAM_TABLE_VERSION;

        crate::proof::norm_param_table::NormParamTableIdentity {
            version: NORM_PARAM_TABLE_VERSION,
            kind,
            model_identity: Digest([0x63; 32]),
            asset_table_identity: Digest([0x64; 32]),
            reader,
            row: 0,
            width,
            valid_min: -128,
            valid_max: 127,
            trusted_i32_mapping: true,
        }
    }

    fn param_claim(
        kind: NormParamKind,
        shape: IndexShape,
    ) -> crate::proof::norm_param_table::NormParamClaimPlan {
        use crate::proof::norm_param_table::NormParamClaimPlan;
        use zkie_engine::pcs::request::gu::GuOracleShape;
        use zkie_engine::pcs::request::runtime_tree::{RuntimeTreePart, RuntimeTreePlan};

        let main_rows = 2u64;
        NormParamClaimPlan {
            kind,
            gu: GuShape {
                oracle: 0x8800 + kind.encode(),
                group: 0x42,
                tree: RuntimeTreePlan {
                    tree_number: 0x1234,
                    leaf_class: 1,
                    log_n: code().log_n,
                    log_n_zk: code().log_n,
                    parts: vec![RuntimeTreePart {
                        main_symbols: main_rows * E_DEGREE as u64,
                        mask_symbols: 0,
                    }],
                    salt_bytes: SALT,
                    run: Digest([0x71; 32]),
                    configuration: Digest([0x72; 32]),
                },
                part: 0,
                rows_high: true,
                u: GuOracleShape {
                    code: code(),
                    main_rows,
                    main_deg: E_DEGREE as u32,
                    mask_rows: 0,
                    mask_code: None,
                    answers: 1,
                },
                index: shape,
            },
            column_draws: 2,
        }
    }

    fn norm_params_fixture(
        activation: &CommittedActivation,
    ) -> (HiddenLinPlan, NormWitness, ParamTables) {
        use crate::proof::hidden_norm_psc::NormParamWitness;
        use crate::proof::norm_param_table::NormParamTablesPlan;

        let (mut plan, mut witness) = norm_fixture(activation);
        let gain_values = vec![2i64, -1, 3, 1];
        let bias_values = vec![1i64, 2, -1, 2];
        let (gain_table, gain_source, gain_reader) = param_table(&gain_values, 0x61);
        let (bias_table, bias_source, bias_reader) = param_table(&bias_values, 0x62);
        let width = plan.norm.as_ref().unwrap().columns;
        let gain_identity = param_identity(NormParamKind::Gain, gain_reader, width);
        let bias_identity = param_identity(NormParamKind::Bias, bias_reader, width);
        plan.norm.as_mut().unwrap().params = Some(NormParamTablesPlan {
            gain_claim: Some(param_claim(
                NormParamKind::Gain,
                gain_identity.reader.shape,
            )),
            bias_claim: Some(param_claim(
                NormParamKind::Bias,
                bias_identity.reader.shape,
            )),
            gain: gain_identity,
            bias: bias_identity,
            column_draws: 2,
            seed_budget: 1,
        });
        witness.params = Some(NormParamWitness {
            gain: gain_values,
            bias: bias_values,
            gain_claim: None,
            bias_claim: None,
        });
        plan.validate().unwrap();
        (
            plan,
            witness,
            ParamTables {
                gain: gain_table,
                gain_source,
                bias: bias_table,
                bias_source,
            },
        )
    }

    #[test]
    fn norm_parameter_openings_prove_and_verify() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = norm_params_fixture(&activation);
        let (messages, hidden_witness) = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &witness,
            &mut params,
        )
        .unwrap();
        let mut verifier_plan = plan.clone();
        verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        let outcome = verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &messages).unwrap();
        assert_eq!(outcome.first_seal, messages.first_seal);
        let plan_params = plan.norm.as_ref().unwrap().params.as_ref().unwrap();
        let openings = messages
            .first
            .norm
            .as_ref()
            .unwrap()
            .params
            .as_ref()
            .unwrap()
            .gain
            .openings
            .len();
        let gain_rows = outcome
            .shared_claims
            .iter()
            .filter(|row| row.oracle == plan_params.gain_claim.as_ref().unwrap().gu.oracle)
            .count();
        assert_eq!(gain_rows, 2 + openings);
    }

    #[test]
    fn norm_parameter_opening_tampering_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = norm_params_fixture(&activation);
        let (messages, hidden_witness) = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &witness,
            &mut params,
        )
        .unwrap();
        let mut verifier_plan = plan.clone();
        verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        let bytes = messages.encode().unwrap();
        let base = HiddenLinMessages::decode(&plan, &bytes).unwrap();
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &base).is_ok());

        let mut changed = base.clone();
        changed.first.norm.as_mut().unwrap().params.as_mut().unwrap().gain.openings[0].position += 1;
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        let symbol = &mut changed
            .first
            .norm
            .as_mut()
            .unwrap()
            .params
            .as_mut()
            .unwrap()
            .gain
            .openings[0]
            .symbols[0];
        *symbol = *symbol + fp_from_signed_word(1);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        changed.first.norm.as_mut().unwrap().params.as_mut().unwrap().bias = None;
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut wrong_plan = plan.clone();
        wrong_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        wrong_plan
            .norm
            .as_mut()
            .unwrap()
            .params
            .as_mut()
            .unwrap()
            .gain
            .reader
            .root = Digest([0x99; 32]);
        assert!(verify_hidden_lin_hidden(&wrong_plan, &table.reader(), &base).is_err());
    }

    #[test]
    fn norm_parameter_eval_coefficients_follow_the_message_layout() {
        let activation = activation();
        let (plan, _, _) = norm_params_fixture(&activation);
        let params_plan = plan.norm.as_ref().unwrap().params.as_ref().unwrap();
        let width = params_plan.gain.width;
        let point = vec![Fp3::ONE; width.trailing_zeros() as usize];
        for claim in [params_plan.gain_claim.as_ref().unwrap(), params_plan.bias_claim.as_ref().unwrap()] {
            let coefficients =
                norm_param_eval_coefficients(&claim.gu, width, &point).unwrap();
            let eq = crate::proof::output_closure::eq_vector(&point).unwrap();
            let rows = usize::try_from(claim.gu.u.main_rows).unwrap();
            let ell = usize::try_from(claim.gu.u.code.ell).unwrap();
            assert_eq!(coefficients.len(), rows * ell);
            for column in 0..width as usize {
                let index = gu_vector_bat_index(&claim.gu, column as u64).unwrap();
                assert_eq!(coefficients[index], eq[column]);
            }
            assert!(norm_param_eval_coefficients(
                &claim.gu,
                width,
                &point[..point.len().saturating_sub(1)]
            )
            .is_err());
        }
        let too_wide = 128u64;
        assert!(norm_param_eval_coefficients(
            &params_plan.gain_claim.as_ref().unwrap().gu,
            too_wide,
            &vec![Fp3::ONE; too_wide.trailing_zeros() as usize]
        )
        .is_err());
    }

    #[test]
    fn norm_parameter_claim_rows_are_shared_by_both_sides() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = norm_params_fixture(&activation);
        let (messages, _) = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &witness,
            &mut params,
        )
        .unwrap();
        let plan_params = plan.norm.as_ref().unwrap().params.as_ref().unwrap();
        let message_params = messages.first.norm.as_ref().unwrap().params.as_ref().unwrap();
        let rows = norm_param_claim_rows(
            &plan_params.gain,
            plan_params.gain_claim.as_ref().unwrap(),
            &message_params.gain,
        )
        .unwrap();
        assert_eq!(rows.record.len(), 1);
        assert_eq!(rows.column.len(), message_params.gain.openings.len());
        let again = norm_param_claim_rows(
            &plan_params.gain,
            plan_params.gain_claim.as_ref().unwrap(),
            &message_params.gain,
        )
        .unwrap();
        assert_eq!(rows.record, again.record);
        assert_eq!(rows.column, again.column);

        let mut swapped = message_params.gain.clone();
        swapped.openings.swap(0, 1);
        assert!(norm_param_claim_rows(
            &plan_params.gain,
            plan_params.gain_claim.as_ref().unwrap(),
            &swapped,
        )
        .is_err());

        let mut tampered = message_params.gain.clone();
        tampered.openings[0].symbols[0] = tampered.openings[0].symbols[0] + fp_from_signed_word(1);
        assert!(norm_param_claim_rows(
            &plan_params.gain,
            plan_params.gain_claim.as_ref().unwrap(),
            &tampered,
        )
        .is_err());
    }

    fn rstd_bits_from_counts(
        plan: &crate::proof::lookup::LookupPlan,
        counts: &[u64],
    ) -> Vec<Fp3> {
        let planes = plan.planes().unwrap();
        let rows = plan.table_rows().unwrap();
        assert_eq!(counts.len(), rows);
        let mut bits = vec![Fp3::ZERO; planes * rows];
        for row in 0..rows {
            for plane in 0..planes {
                if (counts[row] >> plane) & 1 == 1 {
                    bits[plane + planes * row] = Fp3::ONE;
                }
            }
        }
        bits
    }

    fn rstd_fixture(
        activation: &CommittedActivation,
    ) -> (
        HiddenLinPlan,
        crate::proof::hidden_norm_psc::NormWitness,
        crate::proof::lookup_table::RangeTable<RstdRows>,
    ) {
        use crate::proof::hidden_norm_psc::{NormRstdParams, NormRstdPlan};
        use crate::proof::lookup_table::RangeTable;

        let (mut plan, mut witness) = norm_fixture(activation);
        let values: Vec<i64> = (0..8).map(|index| 100 + 7 * index).collect();
        let source = RstdRows::new(0x81, values.clone());
        let config = source.config();
        let table = RangeTable::open(source, 3, &config).unwrap();
        let mut rows = witness.rsqrt.clone().unwrap();
        for row in rows.iter_mut() {
            assert_eq!(row.index, 0);
            row.rstd = values[0];
        }
        witness.rsqrt = Some(rows.clone());
        let round_plan = plan
            .norm
            .as_ref()
            .unwrap()
            .round
            .as_ref()
            .unwrap()
            .clone();
        let round_values = witness
            .core
            .iter()
            .zip(&rows)
            .map(|(core, rsqrt)| {
                round_plan
                    .layout
                    .values_with_rsqrt(core, rsqrt, round_plan.group.columns)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut coins = ByteMaskCoins::new(Digest([0x94; 32])).unwrap();
        witness.round = Some(commit_norm_group(&round_plan.group, &round_values, &mut coins));
        let params = NormRstdParams {
            code: code(),
            mask_code: mask_code(),
            group: GROUP,
            bits_oracle: 9800,
            mask_oracle: 9801,
            bits_tree: 60900,
            mask_tree: 60901,
            leaf_class: 1,
            salt_bytes: SALT,
            run: W_RUN,
            configuration: W_CONFIGURATION,
        };
        let rstd_plan = NormRstdPlan::from_round(
            &round_plan,
            POSITIONS,
            config,
            Fp3::from_u64(values[0] as u64),
            &params,
        )
        .unwrap();
        let messages = &witness.round.as_ref().unwrap().messages;
        let columns = rstd_plan
            .key
            .read_columns(&rstd_plan.lookup, messages)
            .unwrap();
        let keys = rstd_plan.lookup.key_count().unwrap();
        let entries = rstd_plan.key.sources.len() / 2;
        let mut counts = vec![0u64; rstd_plan.lookup.table_rows().unwrap()];
        for key in 0..entries {
            assert_eq!(columns[0][key], Fp3::ZERO);
            assert_eq!(columns[1][key], Fp3::from_u64(values[0] as u64));
            counts[0] += 1;
        }
        for _ in entries..keys {
            counts[0] += 1;
        }
        let bits = rstd_bits_from_counts(&rstd_plan.lookup, &counts);
        witness.rstd = Some(crate::proof::hidden_norm_psc::NormLookupWitness {
            bits,
            messages: None,
        });
        plan.norm.as_mut().unwrap().rstd = Some(rstd_plan);
        let norm_mask_rows = plan
            .norm
            .as_ref()
            .unwrap()
            .mask
            .as_ref()
            .unwrap()
            .mask_rows;
        let rstd_mask_rows = plan
            .norm
            .as_ref()
            .unwrap()
            .rstd
            .as_ref()
            .unwrap()
            .commit
            .mask_rows;
        let index_rows = plan
            .send
            .index_mask
            .as_ref()
            .map(|index| index.tree_rows().unwrap())
            .unwrap_or(0);
        plan.proximity.own_mask_tree.mask_rows =
            crate::proof::hidden_norm_psc::norm_own_mask_rows(
                index_rows,
                norm_mask_rows + rstd_mask_rows,
            )
            .unwrap();
        plan.validate().unwrap();
        (plan, witness, table)
    }

    #[test]
    fn rstd_lookup_proves_and_verifies() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, rstd_table) = rstd_fixture(&activation);
        let (messages, hidden_witness) = prove_hidden_lin_hidden_with_norm_and_rstd_table(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &witness,
            &rstd_table,
        )
        .unwrap();
        let mut verifier_plan = plan.clone();
        verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        let outcome = verify_hidden_lin_hidden_with_rstd_table(
            &verifier_plan,
            &table.reader(),
            &messages,
            &rstd_table,
        )
        .unwrap();
        assert_eq!(outcome.first_seal, messages.first_seal);
    }

    fn recommit_round(
        round_plan: &crate::proof::hidden_norm_psc::NormRoundPlan,
        core: &[crate::proof::hidden_norm::NormCoreRow],
        rows: &[crate::proof::hidden_norm::RsqrtRow],
    ) -> crate::proof::hidden_norm_psc::NormGroupWitness {
        use zkie_engine::pcs::request::fold::ByteMaskCoins;

        let values = core
            .iter()
            .zip(rows)
            .map(|(core, rsqrt)| {
                round_plan
                    .layout
                    .values_with_rsqrt(core, rsqrt, round_plan.group.columns)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut coins = ByteMaskCoins::new(Digest([0x95; 32])).unwrap();
        commit_norm_group(&round_plan.group, &values, &mut coins)
    }

    /// Re-runs the prove entry with one internally consistent but dishonest
    /// committed round row and asserts that the chain rejects it.
    fn prove_with_tampered_rstd(
        activation: &CommittedActivation,
        plan: &HiddenLinPlan,
        witness: &crate::proof::hidden_norm_psc::NormWitness,
        rstd_table: &dyn crate::proof::lookup::LookupTable,
        edit: impl FnOnce(&mut crate::proof::hidden_norm::RsqrtRow),
    ) -> Result<()> {
        let mut table = table();
        let round_plan = plan.norm.as_ref().unwrap().round.as_ref().unwrap();
        let mut changed = witness.clone();
        let mut rows = changed.rsqrt.clone().unwrap();
        edit(&mut rows[0]);
        changed.rsqrt = Some(rows.clone());
        changed.round = Some(recommit_round(round_plan, &changed.core, &rows));
        let proved = prove_hidden_lin_hidden_with_norm_and_rstd_table(
            plan,
            activation,
            &mut table,
            Digest([0x71; 32]),
            &changed,
            rstd_table,
        );
        proved.map(|_| ())
    }

    #[test]
    fn rstd_committed_value_tampering_is_rejected() {
        let activation = activation();
        let (plan, witness, rstd_table) = rstd_fixture(&activation);
        // 107 is a real table value, but it belongs to index 1, not index 0.
        let proved = prove_with_tampered_rstd(&activation, &plan, &witness, &rstd_table, |row| {
            row.rstd = 107;
        });
        assert!(
            proved.is_err(),
            "a committed value that is not the row of the committed index must not produce a proof"
        );
    }

    #[test]
    fn rstd_committed_index_tampering_is_rejected() {
        let activation = activation();
        let (plan, witness, rstd_table) = rstd_fixture(&activation);
        let proved = prove_with_tampered_rstd(&activation, &plan, &witness, &rstd_table, |row| {
            row.index = 1;
        });
        assert!(
            proved.is_err(),
            "a committed index that breaks the index relation must not produce a proof"
        );
    }

    #[test]
    fn rstd_index_past_the_real_length_is_rejected() {
        let activation = activation();
        let (plan, witness, rstd_table) = rstd_fixture(&activation);
        // The trusted table has eight real rows, so index 8 is out of range.
        let proved = prove_with_tampered_rstd(&activation, &plan, &witness, &rstd_table, |row| {
            row.index = 8;
        });
        assert!(
            proved.is_err(),
            "an index past the real table length must not produce a proof"
        );
    }

    fn rstd_values() -> Vec<i64> {
        (0..8).map(|index| 100 + 7 * index).collect()
    }

    fn verify_with_rstd_table(
        activation: &CommittedActivation,
        table_log: u32,
        source: impl crate::proof::lookup_table::TableRowSource,
    ) -> Result<()> {
        use crate::proof::lookup_table::RangeTable;

        let mut table = table();
        let (plan, witness, rstd_table) = rstd_fixture(activation);
        let (messages, hidden_witness) = prove_hidden_lin_hidden_with_norm_and_rstd_table(
            &plan,
            activation,
            &mut table,
            Digest([0x71; 32]),
            &witness,
            &rstd_table,
        )?;
        let config = source_config(&source);
        let wrong = RangeTable::open(source, table_log, &config)?;
        let mut verifier_plan = plan.clone();
        verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        let verified = verify_hidden_lin_hidden_with_rstd_table(
            &verifier_plan,
            &table.reader(),
            &messages,
            &wrong,
        );
        verified.map(|_| ())
    }

    fn source_config(
        source: &impl crate::proof::lookup_table::TableRowSource,
    ) -> crate::proof::lookup_table::TableConfig {
        use crate::proof::lookup_table::{TableConfig, TABLE_ENCODING_VERSION};

        TableConfig {
            identity: source.identity(),
            content_root: source.content_root(),
            real_len: source.real_len(),
            encoding: source.encoding(),
            version: TABLE_ENCODING_VERSION,
        }
    }

    #[test]
    fn rstd_table_identity_tampering_is_rejected() {
        let activation = activation();
        let (plan, _, _) = rstd_fixture(&activation);
        let table_log = plan
            .norm
            .as_ref()
            .unwrap()
            .rstd
            .as_ref()
            .unwrap()
            .lookup
            .table_log;
        let source = RstdRows::new(0x82, rstd_values());
        let verified = verify_with_rstd_table(&activation, table_log, source);
        assert!(
            verified.is_err(),
            "a table whose identity differs from the trusted plan must be rejected"
        );
    }

    #[test]
    fn rstd_table_content_tampering_is_rejected() {
        let activation = activation();
        let (plan, _, _) = rstd_fixture(&activation);
        let table_log = plan
            .norm
            .as_ref()
            .unwrap()
            .rstd
            .as_ref()
            .unwrap()
            .lookup
            .table_log;
        let mut values = rstd_values();
        values[3] += 1;
        let source = RstdRows::new(0x81, values);
        let verified = verify_with_rstd_table(&activation, table_log, source);
        assert!(
            verified.is_err(),
            "a table whose content root differs from the trusted plan must be rejected"
        );
    }

    #[test]
    fn rstd_table_length_tampering_is_rejected() {
        let activation = activation();
        let (plan, _, _) = rstd_fixture(&activation);
        let table_log = plan
            .norm
            .as_ref()
            .unwrap()
            .rstd
            .as_ref()
            .unwrap()
            .lookup
            .table_log;
        let source = RstdRows::new(0x81, rstd_values()[..7].to_vec());
        let verified = verify_with_rstd_table(&activation, table_log, source);
        assert!(
            verified.is_err(),
            "a table whose real length differs from the trusted plan must be rejected"
        );
    }

    /// A source with explicit metadata. With `lie_first` it claims the trusted
    /// metadata but returns a different first row, so only the content check
    /// can catch it; with a different encoding it opens but the verifier's
    /// configuration compare must reject it.
    struct RowsView {
        identity: Digest,
        root: Digest,
        rows: Vec<i64>,
        encoding: crate::proof::lookup_table::TableEncoding,
        lie_first: bool,
    }

    impl crate::proof::lookup_table::TableRowSource for RowsView {
        fn identity(&self) -> Digest {
            self.identity
        }

        fn content_root(&self) -> Digest {
            self.root
        }

        fn real_len(&self) -> usize {
            self.rows.len()
        }

        fn encoding(&self) -> crate::proof::lookup_table::TableEncoding {
            self.encoding
        }

        fn read_rows(&self, start: usize, count: usize) -> Result<Vec<i64>> {
            let end = start.checked_add(count).ok_or(Error::ResourceLimit)?;
            if end > self.rows.len() {
                return Err(Error::ShapeMismatch);
            }
            let mut rows = self.rows[start..end].to_vec();
            if self.lie_first && start == 0 {
                rows[0] += 1;
            }
            Ok(rows)
        }
    }

    #[test]
    fn rstd_table_padding_tampering_is_rejected() {
        use crate::proof::lookup_table::{content_digest, TableEncoding};

        let activation = activation();
        let (plan, _, _) = rstd_fixture(&activation);
        let table_log = plan
            .norm
            .as_ref()
            .unwrap()
            .rstd
            .as_ref()
            .unwrap()
            .lookup
            .table_log;
        let values = rstd_values();
        let root = content_digest(TableEncoding::I64Le, &values).unwrap();
        let source = RowsView {
            identity: Digest([0x81; 32]),
            root,
            rows: values,
            encoding: TableEncoding::I64Le,
            lie_first: true,
        };
        let verified = verify_with_rstd_table(&activation, table_log, source);
        assert!(
            verified.is_err(),
            "a padded table whose read rows do not match the trusted content must be rejected"
        );
    }

    #[test]
    fn rstd_table_encoding_tampering_is_rejected() {
        use crate::proof::lookup_table::{content_digest, TableEncoding};

        let activation = activation();
        let (plan, _, _) = rstd_fixture(&activation);
        let table_log = plan
            .norm
            .as_ref()
            .unwrap()
            .rstd
            .as_ref()
            .unwrap()
            .lookup
            .table_log;
        let values = rstd_values();
        // The trusted fixture table is an i32-le source; i64-le is the wrong
        // encoding that still encodes the same numbers.
        let root = content_digest(TableEncoding::I64Le, &values).unwrap();
        let source = RowsView {
            identity: Digest([0x81; 32]),
            root,
            rows: values,
            encoding: TableEncoding::I64Le,
            lie_first: false,
        };
        let verified = verify_with_rstd_table(&activation, table_log, source);
        assert!(
            verified.is_err(),
            "a table whose encoding differs from the trusted plan must be rejected"
        );
    }

    #[test]
    fn rstd_proof_does_not_carry_plaintext_values() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, rstd_table) = rstd_fixture(&activation);
        let (messages, _) = prove_hidden_lin_hidden_with_norm_and_rstd_table(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &witness,
            &rstd_table,
        )
        .unwrap();
        let norm = messages.second.norm.as_ref().unwrap();
        let rstd = norm.rstd.as_ref().unwrap();
        let first_rstd = messages
            .first
            .norm
            .as_ref()
            .unwrap()
            .rstd
            .as_ref()
            .unwrap();
        // The lookup proof and its outcome contain only fixed-width fields
        // (u64 and canonical field elements), and the sample messages contain
        // only fixed-width fields too, so their eight-byte alignment is kept:
        // a plaintext field value would appear there as its little-endian
        // bytes followed by sixteen zero bytes at an eight-byte boundary.
        // Digests, salts, Merkle paths, public positions and slot lists are
        // not field values and are not scanned.
        let buffers = [
            ("rstd proof", rstd.proof.encode().unwrap()),
            ("rstd claims", rstd.claims.encode_unchecked()),
            ("rstd bits sample", first_rstd.bits.sample.encode()),
            ("rstd mask sample", first_rstd.mask.sample.encode()),
        ];
        for value in rstd_values() {
            let mut needle = Vec::with_capacity(24);
            needle.extend_from_slice(&(value as u64).to_le_bytes());
            needle.extend_from_slice(&[0u8; 16]);
            for (name, buffer) in &buffers {
                let hit = buffer
                    .windows(needle.len())
                    .enumerate()
                    .any(|(offset, window)| offset % 8 == 0 && window == needle.as_slice());
                assert!(
                    !hit,
                    "the {name} must not carry the plaintext rstd value {value}"
                );
            }
        }
    }

    #[test]
    fn rstd_key_padding_tampering_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (mut plan, witness, rstd_table) = rstd_fixture(&activation);
        plan.norm
            .as_mut()
            .unwrap()
            .rstd
            .as_mut()
            .unwrap()
            .key
            .padding_values[1] = Fp3::from_u64(101);
        let proved = prove_hidden_lin_hidden_with_norm_and_rstd_table(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &witness,
            &rstd_table,
        );
        assert!(
            proved.is_err(),
            "a key-side padding pair that differs from the trusted first table row must be rejected"
        );
    }

    #[test]
    fn rstd_message_tampering_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, rstd_table) = rstd_fixture(&activation);
        let (messages, hidden_witness) = prove_hidden_lin_hidden_with_norm_and_rstd_table(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &witness,
            &rstd_table,
        )
        .unwrap();
        let bytes = messages.encode().unwrap();
        let base = HiddenLinMessages::decode(&plan, &bytes).unwrap();
        let mut verifier_plan = plan.clone();
        verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        assert!(
            verify_hidden_lin_hidden_with_rstd_table(
                &verifier_plan,
                &table.reader(),
                &base,
                &rstd_table
            )
            .is_ok()
        );

        let mut changed = base.clone();
        changed
            .first
            .norm
            .as_mut()
            .unwrap()
            .rstd
            .as_mut()
            .unwrap()
            .bits
            .message
            .roots[0] = Digest::ZERO;
        assert!(
            verify_hidden_lin_hidden_with_rstd_table(
                &verifier_plan,
                &table.reader(),
                &changed,
                &rstd_table
            )
            .is_err(),
            "a changed rstd bits root must be rejected"
        );

        let mut changed = base.clone();
        changed
            .first
            .norm
            .as_mut()
            .unwrap()
            .rstd
            .as_mut()
            .unwrap()
            .mask
            .root = Digest::ZERO;
        assert!(
            verify_hidden_lin_hidden_with_rstd_table(
                &verifier_plan,
                &table.reader(),
                &changed,
                &rstd_table
            )
            .is_err(),
            "a changed rstd mask root must be rejected"
        );

        let mut changed = base.clone();
        let value = &mut changed
            .second
            .norm
            .as_mut()
            .unwrap()
            .rstd
            .as_mut()
            .unwrap()
            .claims
            .key
            .value;
        *value = value.add(&Fp3::ONE);
        assert!(
            verify_hidden_lin_hidden_with_rstd_table(
                &verifier_plan,
                &table.reader(),
                &changed,
                &rstd_table
            )
            .is_err(),
            "a changed rstd key claim must be rejected"
        );

        let mut changed = base.clone();
        let answers = &mut changed
            .first
            .norm
            .as_mut()
            .unwrap()
            .rstd
            .as_mut()
            .unwrap()
            .bits
            .sample
            .answers;
        assert!(!answers.is_empty());
        answers[0] = answers[0].add(&Fp3::ONE);
        assert!(
            verify_hidden_lin_hidden_with_rstd_table(
                &verifier_plan,
                &table.reader(),
                &changed,
                &rstd_table
            )
            .is_err(),
            "a changed rstd sample answer must be rejected"
        );

        let mut changed = base.clone();
        let tie = &mut changed
            .second
            .norm
            .as_mut()
            .unwrap()
            .rstd
            .as_mut()
            .unwrap()
            .proof
            .tie;
        assert!(!tie.is_empty() && !tie[0].end_values.is_empty());
        tie[0].end_values[0] = tie[0].end_values[0].add(&Fp3::ONE);
        assert!(
            verify_hidden_lin_hidden_with_rstd_table(
                &verifier_plan,
                &table.reader(),
                &changed,
                &rstd_table
            )
            .is_err(),
            "a changed rstd lookup proof value must be rejected"
        );
    }

    #[test]
    fn rstd_selector_tampering_is_rejected() {
        let activation = activation();
        let (plan, witness, rstd_table) = rstd_fixture(&activation);
        let two_segments =
            prove_with_tampered_rstd(&activation, &plan, &witness, &rstd_table, |row| {
                row.mid = true;
            });
        assert!(
            two_segments.is_err(),
            "two active rsqrt segments must not produce a proof"
        );
        let no_segment =
            prove_with_tampered_rstd(&activation, &plan, &witness, &rstd_table, |row| {
                row.fine = false;
            });
        assert!(
            no_segment.is_err(),
            "no active rsqrt segment must not produce a proof"
        );
    }

    #[test]
    fn rstd_rounding_remainder_tampering_is_rejected() {
        let activation = activation();
        let (plan, witness, rstd_table) = rstd_fixture(&activation);
        let shifted = prove_with_tampered_rstd(&activation, &plan, &witness, &rstd_table, |row| {
            row.raw += 1;
        });
        assert!(
            shifted.is_err(),
            "a shifted rounding quotient must not produce a proof"
        );
        let remainder = prove_with_tampered_rstd(&activation, &plan, &witness, &rstd_table, |row| {
            row.raw_remainder += 1;
        });
        assert!(
            remainder.is_err(),
            "a shifted rounding remainder must not produce a proof"
        );
    }

    #[test]
    fn rstd_index_quotient_tampering_is_rejected() {
        let activation = activation();
        let (plan, witness, rstd_table) = rstd_fixture(&activation);
        let proved = prove_with_tampered_rstd(&activation, &plan, &witness, &rstd_table, |row| {
            row.mid_q += 1;
        });
        assert!(
            proved.is_err(),
            "a shifted segment quotient must not produce a proof"
        );
    }

    #[test]
    fn norm_inconsistent_sum_witness_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (plan, witness) = norm_fixture(&activation);
        let norm = plan.norm.as_ref().unwrap();
        let mut coins = ByteMaskCoins::new(Digest([0x92; 32])).unwrap();
        let altered = witness
            .core
            .iter()
            .map(|row| {
                vec![crate::proof::output_closure::fp3_signed(row.sum + 1).unwrap()]
            })
            .collect::<Vec<_>>();
        let mut changed = witness.clone();
        changed.sum = commit_norm_group(&norm.sum, &altered, &mut coins);
        let proved = prove_hidden_lin_hidden_with_norm(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &changed,
        );
        assert!(
            proved.is_err(),
            "an inconsistent committed sum must not produce a proof"
        );
    }

    #[test]
    fn rsqrt_witness_value_tampering_is_rejected() {
        use zkie_engine::pcs::request::fold::ByteMaskCoins;

        let activation = activation();
        let mut table = table();
        let (plan, witness) = norm_fixture(&activation);
        let norm = plan.norm.as_ref().unwrap();
        let round_plan = norm.round.as_ref().unwrap();
        let mut changed = witness.clone();
        let mut rows = changed.rsqrt.clone().unwrap();
        rows[0].fine_slack += 1;
        changed.rsqrt = Some(rows.clone());
        let values = changed
            .core
            .iter()
            .zip(&rows)
            .map(|(core, rsqrt)| {
                round_plan
                    .layout
                    .values_with_rsqrt(core, rsqrt, round_plan.group.columns)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut coins = ByteMaskCoins::new(Digest([0x93; 32])).unwrap();
        changed.round = Some(commit_norm_group(&round_plan.group, &values, &mut coins));
        let proved = prove_hidden_lin_hidden_with_norm(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &changed,
        );
        assert!(
            proved.is_err(),
            "a tampered rsqrt slack must not produce a proof"
        );
    }

    #[test]
    fn rsqrt_plan_table_length_tampering_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (mut plan, witness) = norm_fixture(&activation);
        plan.norm
            .as_mut()
            .unwrap()
            .round
            .as_mut()
            .unwrap()
            .table_real_len += 1;
        let proved = prove_hidden_lin_hidden_with_norm(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &witness,
        );
        assert!(
            proved.is_err(),
            "a changed rsqrt table length must not produce a proof"
        );
    }

    #[test]
    fn rsqrt_mask_entry_tampering_is_rejected() {
        use zkie_engine::pcs::request::field::Fp3;

        let activation = activation();
        let mut table = table();
        let (plan, witness) = norm_fixture(&activation);
        let mut changed = witness.clone();
        let relations = plan.norm.as_ref().unwrap().relations.len();
        let instance = relations + 24;
        let mask = changed.mask.as_mut().unwrap();
        mask.entries[instance].shown[0] =
            mask.entries[instance].shown[0].add(&Fp3::ONE);
        let proved = prove_hidden_lin_hidden_with_norm(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &changed,
        );
        assert!(
            proved.is_err(),
            "a tampered rsqrt instance mask must not produce a proof"
        );
    }

    #[test]
    fn norm_message_tampering_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (plan, witness) = norm_fixture(&activation);
        let (messages, hidden_witness) =
            prove_hidden_lin_hidden_with_norm(
                &plan,
                &activation,
                &mut table,
                Digest([0x71; 32]),
                &witness,
            )
            .unwrap();
        let bytes = messages.encode().unwrap();
        let base = HiddenLinMessages::decode(&plan, &bytes).unwrap();
        let mut verifier_plan = plan.clone();
        verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &base).is_ok());

        let mut changed = base.clone();
        let value = &mut changed.second.norm.as_mut().unwrap().point_values[0][0];
        *value = value.add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        changed.first.norm.as_mut().unwrap().mask.as_mut().unwrap().root = Digest::ZERO;
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        changed.first.norm.as_mut().unwrap().round.as_mut().unwrap().message.group.entries[0].root =
            Digest::ZERO;
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        let answer = &mut changed
            .first
            .norm
            .as_mut()
            .unwrap()
            .round
            .as_mut()
            .unwrap()
            .sample
            .answers[0];
        *answer = answer.add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        changed.first.norm.as_mut().unwrap().round.as_mut().unwrap().positions[0][0] ^= 1;
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        let answer = &mut changed.first.norm.as_mut().unwrap().sum.sample.answers[0];
        *answer = answer.add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        let end = &mut changed.second.norm.as_mut().unwrap().psc[0].end_values[0];
        *end = end.add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut other = verifier_plan;
        other.norm.as_mut().unwrap().relations = vec![0];
        assert!(verify_hidden_lin_hidden(&other, &table.reader(), &base).is_err());
    }

    #[test]
    fn norm_four_relation_negative_matrix() {
        let activation = activation();
        let mut table = table();
        let (plan, witness) = norm_fixture(&activation);
        let (messages, hidden_witness) = prove_hidden_lin_hidden_with_norm(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &witness,
        )
        .unwrap();
        let base =
            HiddenLinMessages::decode(&plan, &messages.encode().unwrap()).unwrap();
        let mut verifier_plan = plan.clone();
        verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &base).is_ok());
        let instances = base.second.norm.as_ref().unwrap().psc.len();
        assert_eq!(
            instances,
            POSITIONS as usize * plan.norm.as_ref().unwrap().relations.len()
        );

        // Endpoint values of every relation instance.
        for instance in 0..instances {
            let mut changed = base.clone();
            let value = &mut changed.second.norm.as_mut().unwrap().point_values[instance][0];
            *value = value.add(&Fp3::ONE);
            assert!(
                verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err(),
                "a changed endpoint of instance {instance} was accepted"
            );
        }

        // Product-sumcheck end values of every relation instance.
        for instance in 0..instances {
            let mut changed = base.clone();
            let value = &mut changed.second.norm.as_mut().unwrap().psc[instance].end_values[0];
            *value = value.add(&Fp3::ONE);
            assert!(
                verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err(),
                "a changed transcript end of instance {instance} was accepted"
            );
        }

        // Challenge order: the first two instances exchange transcripts.
        let mut changed = base.clone();
        changed.second.norm.as_mut().unwrap().psc.swap(0, 1);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        // Mask root and mask sample answer.
        let mut changed = base.clone();
        changed.first.norm.as_mut().unwrap().mask.as_mut().unwrap().root = Digest::ZERO;
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());
        let mut changed = base.clone();
        let answer = &mut changed
            .first
            .norm
            .as_mut()
            .unwrap()
            .mask
            .as_mut()
            .unwrap()
            .sample
            .answers[0];
        *answer = answer.add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        // Wrong sample row target of every group.
        for group in 0..5 {
            let mut changed = base.clone();
            let norm = changed.first.norm.as_mut().unwrap();
            let target = match group {
                0 => &mut norm.x.sample.answers[0],
                1 => &mut norm.sum.sample.answers[0],
                2 => &mut norm.squares.sample.answers[0],
                3 => &mut norm.mean.sample.answers[0],
                _ => &mut norm.variance.sample.answers[0],
            };
            *target = target.add(&Fp3::ONE);
            assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());
        }

        // Source mapping: a committed mean root and its entry root move together.
        let mut changed = base.clone();
        let mean = &mut changed.first.norm.as_mut().unwrap().mean;
        mean.message.group.entries[0].root = Digest([0x99; 32]);
        mean.message.roots[0] = Digest([0x99; 32]);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        // Source mapping: a committed group oracle is renumbered.
        let mut changed = base.clone();
        changed
            .first
            .norm
            .as_mut()
            .unwrap()
            .variance
            .message
            .group
            .entries[0]
            .oracle += 1;
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        // Coordinate domain and relation domain changes.
        let mut other = verifier_plan.clone();
        other.norm.as_mut().unwrap().shape.valid -= 1;
        assert!(verify_hidden_lin_hidden(&other, &table.reader(), &base).is_err());
        let mut other = verifier_plan.clone();
        other.norm.as_mut().unwrap().relations = vec![0, 1, 2];
        assert!(verify_hidden_lin_hidden(&other, &table.reader(), &base).is_err());
    }

    #[test]
    fn norm_inconsistent_remainder_witness_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (plan, witness) = norm_fixture(&activation);
        let mut changed = witness.clone();
        changed.core[0].mean_remainder += 1;
        changed.core[0].variance_remainder -= 1;
        let proved = prove_hidden_lin_hidden_with_norm(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &changed,
        );
        assert!(
            proved.is_err(),
            "an inconsistent quotient remainder must not produce a proof"
        );
    }

    #[test]
    fn norm_bound_relations_accept_the_hidden_fixture() {
        let activation = activation();
        let mut table = table();
        let (plan, witness) = norm_fixture(&activation);
        let (messages, hidden_witness) = prove_hidden_lin_hidden_with_norm(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &witness,
        )
        .unwrap();
        let mut verifier_plan = plan;
        verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        let outcome =
            verify_hidden_lin_hidden_with_witness(
                &verifier_plan,
                &table.reader(),
                &messages,
                &hidden_witness,
            )
            .unwrap();
        assert!(outcome
            .own_base
            .single()
            .expect("the fixture ends in BASE")
            .fold_main);
    }

    fn lookup_bits(plan: &crate::proof::lookup::LookupPlan, keys: &[Fp3]) -> Vec<Fp3> {
        let planes = plan.planes().unwrap();
        let rows = plan.table_rows().unwrap();
        let mut counts = vec![0u64; rows];
        for key in keys {
            let mut found = false;
            for (row, count) in counts.iter_mut().enumerate() {
                if *key == Fp3::from_u64(row as u64) {
                    *count += 1;
                    found = true;
                    break;
                }
            }
            assert!(found, "a lookup key is not a byte");
        }
        let mut bits = vec![Fp3::ZERO; planes * rows];
        for row in 0..rows {
            for plane in 0..planes {
                if (counts[row] >> plane) & 1 == 1 {
                    bits[plane + planes * row] = Fp3::ONE;
                }
            }
        }
        bits
    }

    #[test]
    fn norm_lookup_carrier_proves_and_verifies() {
        use crate::proof::hidden_norm_psc::{
            norm_own_mask_rows, NormLookupParams, NormLookupPlan, NormLookupWitness,
        };

        let activation = activation();
        let mut table = table();
        let (mut plan, mut witness) = norm_fixture(&activation);
        let round = plan
            .norm
            .as_ref()
            .unwrap()
            .round
            .clone()
            .expect("the fixture carries the rounding rows");
        let params = NormLookupParams {
            code: code(),
            mask_code: mask_code(),
            group: GROUP,
            bits_oracle: 9700,
            mask_oracle: 9701,
            bits_tree: 60701,
            mask_tree: 60700,
            leaf_class: 1,
            salt_bytes: SALT,
            run: W_RUN,
            configuration: W_CONFIGURATION,
            padding_byte: 0,
            table_identity: Digest([0x63; 32]),
        };
        let lookup_plan =
            NormLookupPlan::from_round(&round, POSITIONS, &params).unwrap();
        let keys = lookup_plan
            .key
            .read(
                &lookup_plan.lookup,
                &witness.round.as_ref().unwrap().messages,
                Fp3::ONE,
            )
            .unwrap();
        let bits = lookup_bits(&lookup_plan.lookup, &keys);
        let mask_rows = lookup_plan.commit.mask_rows;
        plan.norm.as_mut().unwrap().lookup = Some(lookup_plan);
        let norm_mask_rows = plan
            .norm
            .as_ref()
            .unwrap()
            .mask
            .as_ref()
            .unwrap()
            .mask_rows;
        let index_rows = plan
            .send
            .index_mask
            .as_ref()
            .map(|index| index.tree_rows().unwrap())
            .unwrap_or(0);
        plan.proximity.own_mask_tree.mask_rows =
            norm_own_mask_rows(index_rows, norm_mask_rows + mask_rows).unwrap();
        witness.lookup = Some(NormLookupWitness {
            bits,
            messages: None,
        });
        plan.validate().unwrap();

        let (messages, hidden_witness) = prove_hidden_lin_hidden_with_norm(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &witness,
        )
        .unwrap();
        let bytes = messages.encode().unwrap();
        let decoded = HiddenLinMessages::decode(&plan, &bytes).unwrap();
        assert!(decoded
            .second
            .norm
            .as_ref()
            .unwrap()
            .lookup
            .is_some());
        let mut verifier_plan = plan.clone();
        verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &decoded).is_ok());

        let mut changed = decoded.clone();
        {
            let claims = &mut changed
                .second
                .norm
                .as_mut()
                .unwrap()
                .lookup
                .as_mut()
                .unwrap()
                .claims;
            claims.key.value = claims.key.value.add(&Fp3::ONE);
        }
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = decoded.clone();
        {
            let lookup = changed
                .second
                .norm
                .as_mut()
                .unwrap()
                .lookup
                .as_mut()
                .unwrap();
            lookup.proof.bits.hiding_sum = lookup.proof.bits.hiding_sum.add(&Fp3::ONE);
        }
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let (planes, key_bits, table_log) = {
            let lookup_plan = plan.norm.as_ref().unwrap().lookup.as_ref().unwrap();
            (
                lookup_plan.lookup.planes().unwrap(),
                lookup_plan.lookup.key_bits as usize,
                lookup_plan.lookup.table_log as usize,
            )
        };
        let leaf_index = (planes - 1) + key_bits + table_log;
        let bits_index = leaf_index + 1;

        let mut changed = decoded.clone();
        {
            let claims = &mut changed
                .second
                .norm
                .as_mut()
                .unwrap()
                .lookup
                .as_mut()
                .unwrap()
                .claims;
            let leaf = &mut claims.rows[leaf_index];
            leaf.row_target = leaf.row_target.add(&Fp3::ONE);
        }
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = decoded.clone();
        {
            let claims = &mut changed
                .second
                .norm
                .as_mut()
                .unwrap()
                .lookup
                .as_mut()
                .unwrap()
                .claims;
            let leaf = &mut claims.rows[leaf_index];
            leaf.point[0] = leaf.point[0].add(&Fp3::ONE);
        }
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = decoded.clone();
        {
            let claims = &mut changed
                .second
                .norm
                .as_mut()
                .unwrap()
                .lookup
                .as_mut()
                .unwrap()
                .claims;
            claims.multiplicity.1 = claims.multiplicity.1.add(&Fp3::ONE);
        }
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = decoded.clone();
        {
            let claims = &mut changed
                .second
                .norm
                .as_mut()
                .unwrap()
                .lookup
                .as_mut()
                .unwrap()
                .claims;
            let shown = &mut claims.rows[bits_index].ends[0];
            shown.value = shown.value.add(&Fp3::ONE);
        }
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = decoded.clone();
        {
            let lookup = changed
                .second
                .norm
                .as_mut()
                .unwrap()
                .lookup
                .as_mut()
                .unwrap();
            lookup.proof.leaves.rounds[0].coefficients[0] =
                lookup.proof.leaves.rounds[0].coefficients[0].add(&Fp3::ONE);
        }
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        // First-message lookup material: a changed bit root, mask root,
        // sample answer or opening position is rejected before the second
        // stage.
        let mut changed = decoded.clone();
        {
            let entry = &mut changed
                .first
                .norm
                .as_mut()
                .unwrap()
                .lookup
                .as_mut()
                .unwrap()
                .bits
                .message
                .group
                .entries[0];
            entry.root = Digest([0x77; 32]);
        }
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = decoded.clone();
        {
            changed
                .first
                .norm
                .as_mut()
                .unwrap()
                .lookup
                .as_mut()
                .unwrap()
                .mask
                .root = Digest([0x78; 32]);
        }
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = decoded.clone();
        {
            let sample = &mut changed
                .first
                .norm
                .as_mut()
                .unwrap()
                .lookup
                .as_mut()
                .unwrap()
                .bits
                .sample;
            sample.answers[0] = sample.answers[0].add(&Fp3::ONE);
        }
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = decoded.clone();
        {
            let mask = &mut changed
                .first
                .norm
                .as_mut()
                .unwrap()
                .lookup
                .as_mut()
                .unwrap()
                .mask;
            mask.positions[0] = mask.positions[0].wrapping_add(1);
        }
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        // A missing second-stage lookup carrier is rejected.
        let mut changed = decoded.clone();
        changed.second.norm.as_mut().unwrap().lookup = None;
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        // The verifier-visible proof does not carry the private key vector.
        let key_pattern = {
            let lookup_plan = plan.norm.as_ref().unwrap().lookup.as_ref().unwrap();
            let keys = lookup_plan
                .key
                .read(
                    &lookup_plan.lookup,
                    &witness.round.as_ref().unwrap().messages,
                    Fp3::ONE,
                )
                .unwrap();
            let mut pattern = Vec::new();
            for key in keys.iter().take(4) {
                for coefficient in key.c {
                    pattern.extend_from_slice(&coefficient.as_canonical_u64().to_le_bytes());
                }
            }
            pattern
        };
        assert!(!key_pattern.is_empty());
        assert!(!bytes
            .windows(key_pattern.len())
            .any(|window| window == key_pattern.as_slice()));
    }

    #[test]
    fn production_key_map_reads_the_round_digits() {
        use crate::proof::hidden_norm_psc::{lookup_parameters, round_digit_lookup_key};
        use crate::proof::lookup::LookupPlan;

        let activation = activation();
        let (plan, witness) = norm_fixture(&activation);
        let norm = plan.norm.as_ref().expect("the fixture carries the norm");
        let round = norm.round.as_ref().expect("the fixture carries the round rows");
        let digits = round.layout.digit_slots().unwrap();
        let positions = plan.send.positions();
        let (key_bits, mu) = lookup_parameters(digits.len() * positions as usize).unwrap();
        let key = round_digit_lookup_key(
            round,
            positions,
            key_bits,
            0,
            Digest([0x63; 32]),
        )
        .unwrap();
        let lookup = LookupPlan {
            key_bits,
            mu,
            table_log: 8,
            columns: 1,
        };
        lookup.validate().unwrap();
        let round_messages = &witness.round.as_ref().unwrap().messages;
        let keys = key.read(&lookup, round_messages, Fp3::ONE).unwrap();
        assert_eq!(keys.len(), lookup.key_count().unwrap());
        let rows = usize::try_from(round.group.shape.main_rows).unwrap();
        for position in 0..positions as usize {
            for (index, slot) in digits.iter().enumerate() {
                let value = round_messages[position][slot % rows][slot / rows];
                assert_eq!(keys[position * digits.len() + index], value);
            }
        }
        for extra in keys.iter().skip(key.sources.len()) {
            assert_eq!(*extra, Fp3::ZERO);
        }
    }

    fn staged_plan(plan: &HiddenLinPlan) -> HiddenLinPlan {
        let mut staged = plan.clone();
        let main = plan.proximity.code;
        let mask = plan.proximity.mask_code;
        let next_main = build_code_plan(main.log_n + 2, main.ell, main.r).unwrap();
        let next_mask = build_code_plan(main.log_n + 2, mask.ell * 4, mask.r).unwrap();
        let stage_zero = IoppStagePlan {
            main,
            mask,
            k: 2,
            h: 0,
            positions: plan.proximity.shared_base_positions,
            samples: 1,
            work_bits: 0,
        };
        let last = IoppStagePlan {
            main: next_main,
            mask: next_mask,
            k: 0,
            h: 0,
            positions: plan.proximity.shared_base_positions,
            samples: 1,
            work_bits: 0,
        };
        staged.proximity.shared_stages = vec![stage_zero.clone(), last.clone()];
        staged.proximity.own_stages = vec![stage_zero, last];
        staged
    }

    /// One legal two-stage chain for the hidden toy fixture. The stage-0
    /// k/h values are the fixture group's actual fold shape; the last stage
    /// uses a wider mask code so the switch mask room holds every old tail.
    fn staged_two_stage(
        plan: &HiddenLinPlan,
        shared_k: u32,
        shared_h: u32,
        own_k: u32,
        own_h: u32,
    ) -> HiddenLinPlan {
        let mut staged = plan.clone();
        let main = plan.proximity.code;
        let mask = plan.proximity.mask_code;
        let next_main =
            build_code_plan(main.log_n + 2, main.ell, main.r).unwrap();
        let next_mask =
            build_code_plan(mask.log_n + 2, mask.ell * 4, mask.r).unwrap();
        let stage_zero = |k: u32, h: u32| IoppStagePlan {
            main,
            mask,
            k,
            h,
            positions: plan.proximity.shared_base_positions,
            samples: 1,
            work_bits: 0,
        };
        let last = IoppStagePlan {
            main: next_main,
            mask: next_mask,
            k: 0,
            h: 0,
            positions: plan.proximity.own_base_positions,
            samples: 1,
            work_bits: 0,
        };
        staged.proximity.shared_stages =
            vec![stage_zero(shared_k, shared_h), last.clone()];
        staged.proximity.own_stages = vec![stage_zero(own_k, own_h), last];
        staged
    }

    #[test]
    fn staged_final_verifier_accepts_and_rejects_tampering() {
        let activation = activation();
        let mut table = table();
        let base = hidden_plan(&activation);
        let staged = staged_two_stage(&base, 1, 0, 2, 2);
        let (messages, witness) =
            prove_hidden_lin_hidden(&staged, &activation, &mut table, Digest([0x71; 32]))
                .unwrap();
        let mut verifier_plan = staged;
        verifier_plan.send.expected_roots = Some(witness.send.roots.clone());
        let outcome =
            verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &messages).unwrap();
        assert!(outcome.shared_base.is_stages());
        assert!(outcome.own_base.is_stages());
        let mut tampered = messages.clone();
        match &mut tampered.shared.tail {
            ProximityTail::Stages(proof) => proof.switches[0].declared_root.0[0] ^= 1,
            ProximityTail::Single { .. } => panic!("the fixture is staged"),
        }
        assert!(
            verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &tampered).is_err()
        );
        let mut tampered = messages.clone();
        match &mut tampered.second.tail {
            ProximityTail::Stages(proof) => proof.base[0] ^= 1,
            ProximityTail::Single { .. } => panic!("the fixture is staged"),
        }
        assert!(
            verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &tampered).is_err()
        );
        // A switch target that differs from the trusted plan is rejected
        // before its openings are used.
        let mut tampered = messages.clone();
        match &mut tampered.second.tail {
            ProximityTail::Stages(proof) => proof.switches[0].target.main_rows += 1,
            ProximityTail::Single { .. } => panic!("the fixture is staged"),
        }
        assert!(
            verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &tampered).is_err()
        );
        // A changed old-member opening reaches the final verifier and is
        // rejected by the switch authentication.
        let mut tampered = messages.clone();
        match &mut tampered.second.tail {
            ProximityTail::Stages(proof) => {
                proof.switches[0].member_openings[0].columns[0].symbols[0] =
                    fp_from_signed_word(0x1234_5678);
            }
            ProximityTail::Single { .. } => panic!("the fixture is staged"),
        }
        assert!(
            verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &tampered).is_err()
        );
        // A missing switch is a sequence mismatch at the final verifier.
        let mut tampered = messages.clone();
        match &mut tampered.second.tail {
            ProximityTail::Stages(proof) => {
                proof.switches.pop();
            }
            ProximityTail::Single { .. } => panic!("the fixture is staged"),
        }
        assert!(
            verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &tampered).is_err()
        );
        // The shared group's old mask is the trusted layout's zero: an
        // old-mask opening is rejected at the final verifier.
        let mut tampered = messages.clone();
        match &mut tampered.shared.tail {
            ProximityTail::Stages(proof) => {
                proof.switches[0].old_mask_opening =
                    Some(proof.switches[0].fresh_opening.clone());
            }
            ProximityTail::Single { .. } => panic!("the fixture is staged"),
        }
        assert!(
            verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &tampered).is_err()
        );
        // The own group's old mask comes from its committed members: an
        // old-mask opening is rejected at the final verifier.
        let mut tampered = messages.clone();
        match &mut tampered.second.tail {
            ProximityTail::Stages(proof) => {
                proof.switches[0].old_mask_opening =
                    Some(proof.switches[0].fresh_opening.clone());
            }
            ProximityTail::Single { .. } => panic!("the fixture is staged"),
        }
        assert!(
            verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &tampered).is_err()
        );
        // The stage-0 fresh mask is a separate object: a changed fresh
        // opening is rejected at the final verifier.
        let mut tampered = messages.clone();
        match &mut tampered.second.tail {
            ProximityTail::Stages(proof) => {
                proof.switches[0].fresh_opening.columns[0].symbols[0] =
                    fp_from_signed_word(0x2345_6789);
            }
            ProximityTail::Single { .. } => panic!("the fixture is staged"),
        }
        assert!(
            verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &tampered).is_err()
        );
    }

    fn hidden_run() -> (HiddenLinPlan, HiddenLinMessages, HiddenLinWitness) {
        let activation = activation();
        let mut plan = hidden_plan(&activation);
        let mut table = table();
        let (messages, witness) =
            prove_hidden_lin_hidden(&plan, &activation, &mut table, Digest([0x71; 32])).unwrap();
        plan.send.expected_dot_root = witness.send.dot.as_ref().map(|map| map.root);
        (plan, messages, witness)
    }

    fn hidden_run_unbound_residual() -> (HiddenLinPlan, HiddenLinMessages, HiddenLinWitness) {
        let activation = activation();
        let mut plan = hidden_plan(&activation);
        let mut table = table();
        let (messages, witness) = prove_hidden_lin_hidden_unbound_residual(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
        )
        .unwrap();
        plan.send.expected_dot_root = witness.send.dot.as_ref().map(|map| map.root);
        (plan, messages, witness)
    }

    /// Gaussian-elimination rank over `Fp3`.
    fn rank_over_fp3(matrix: &[Vec<Fp3>]) -> usize {
        let rows = matrix.len();
        let cols = matrix.first().map_or(0, Vec::len);
        let mut work = matrix.to_vec();
        let mut rank = 0usize;
        for column in 0..cols {
            let pivot = (rank..rows).find(|&row| work[row][column] != Fp3::ZERO);
            let Some(pivot) = pivot else {
                continue;
            };
            work.swap(rank, pivot);
            let inverse = work[rank][column].inverse().unwrap();
            for value in work[rank].iter_mut().skip(column) {
                *value = value.mul(&inverse);
            }
            for row in 0..rows {
                if row == rank || work[row][column] == Fp3::ZERO {
                    continue;
                }
                let factor = work[row][column];
                for column_after in column..cols {
                    work[row][column_after] =
                        work[row][column_after].sub(&factor.mul(&work[rank][column_after]));
                }
            }
            rank += 1;
        }
        rank
    }

    #[test]
    fn hidden_masked_eval_and_dot_rows_are_bound() {
        let (plan, messages, witness) = hidden_run();
        let table = table();
        let outcome = verify_hidden_lin_hidden(&plan, &table.reader, &messages).unwrap();
        assert_eq!(outcome.eval_rows.len(), POSITIONS as usize);
        let own_base = outcome.own_base.single().expect("the fixture ends in BASE");
        assert!(own_base.fold_main);
        assert!(own_base.base_linear);

        // The verifier rebuilds both masked rows from the trusted plan.
        let (members, _) = own_members(&plan, &messages.first, &messages.shared, None, None).unwrap();
        let rows = own_plain_rows(
            &plan,
            &members,
            &messages.first,
            &outcome.shared_point,
            &messages.second.tilde,
            None,
            None,
            None,
        )
        .unwrap();
        let index = plan.send.index_mask.as_ref().unwrap();
        let positions = POSITIONS as usize;
        assert_eq!(rows.len(), 2 * positions);
        for position in 0..positions {
            let eval = &rows[position];
            assert_eq!(eval.kind, BatRowKind::Plain);
            assert_eq!(eval.target, messages.first.eval_targets[position]);
            assert_eq!(eval.terms.len(), 1);
            assert_eq!(eval.terms[0].part, BatPart::Main);
            let eval_slot = index.entries.eval_slot(position as u32).unwrap();
            assert_eq!(eval.units.len(), 1);
            assert_eq!(eval.units[0].part, BatPart::Mask);
            assert_eq!(
                eval.units[0].index,
                index.entry_virtual_index(eval_slot).unwrap()
            );
            assert_eq!(eval.units[0].coefficient, Fp3::ONE);
            let dot = &rows[positions + position];
            assert_eq!(dot.kind, BatRowKind::Plain);
            assert_eq!(dot.target, messages.second.tilde[position]);
            assert_eq!(dot.terms.len(), 1);
            assert_eq!(dot.terms[0].part, BatPart::Main);
            let tilde_slot = index.entries.tilde_slot(position as u32).unwrap();
            assert_eq!(
                dot.units[0].part,
                BatPart::Mask
            );
            assert_eq!(
                dot.units[0].index,
                index.entry_virtual_index(tilde_slot).unwrap()
            );
            assert_ne!(eval_slot, tilde_slot);
        }
        // The Eval target is the masked value, not the plain committed row
        // evaluation.
        let eval = witness.send.eval_values(&messages.first.psi).unwrap();
        let mask_witness = witness.send.index_mask.as_ref().unwrap();
        for position in 0..positions {
            let mask = mask_witness
                .entry(index, index.entries.eval_slot(position as u32).unwrap())
                .unwrap();
            assert_eq!(
                messages.first.eval_targets[position],
                eval[position].add(&mask)
            );
        }
    }

    #[test]
    fn hidden_masked_negatives_reach_the_final_verifier() {
        // Changed W root.
        let (plan, mut messages, _) = hidden_run();
        let table = table();
        messages.first.group.entries[0].root = Digest([0x66; 32]);
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_err());

        // Changed W opening row symbol.
        let (plan, mut messages, _) = hidden_run();
        messages.first.openings[0].columns[0].symbols[0] = Fp::from_u64(0xDEAD);
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_err());

        // Changed Eval target `tau_t`.
        let (plan, mut messages, _) = hidden_run();
        messages.first.eval_targets[0] = messages.first.eval_targets[0].add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_err());

        // Changed Eval point `psi_t`.
        let (plan, mut messages, _) = hidden_run();
        messages.first.psi[0][0] = messages.first.psi[0][0].add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_err());

        // Changed mask root, seed, answer and opening symbol.
        let (plan, mut messages, _) = hidden_run();
        messages
            .first
            .index_mask
            .as_mut()
            .unwrap()
            .root = Digest([0x67; 32]);
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_err());

        let (plan, mut messages, _) = hidden_run();
        messages
            .first
            .index_mask
            .as_mut()
            .unwrap()
            .sample
            .seed[0]
            .z[0] = Fp3::ONE;
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_err());

        let (plan, mut messages, _) = hidden_run();
        messages
            .first
            .index_mask
            .as_mut()
            .unwrap()
            .sample
            .answers[0] = messages.first.index_mask.as_ref().unwrap().sample.answers[0]
            .add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_err());

        let (plan, mut messages, _) = hidden_run();
        messages.first.index_mask.as_mut().unwrap().opening.columns[0].symbols[0] =
            Fp::from_u64(0xBEEF);
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_err());

        // Missing mask message or Eval row allocation.
        let (mut plan, _messages, _) = hidden_run();
        plan.send.index_mask = None;
        plan.proximity.own_mask_tree.mask_rows = 1;
        assert!(plan.validate().is_err());
        let (mut plan, _messages, _) = hidden_run();
        plan.send.index_mask.as_mut().unwrap().entries.eval = 0;
        assert!(plan.validate().is_err());
        // A message without the committed mask message is rejected.
        let (plan, mut messages, _) = hidden_run();
        messages.first.index_mask = None;
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_err());
    }

    #[test]
    fn hidden_dot_row_negatives_reach_the_final_verifier() {
        let table = table();

        // The honest proof, including the canonical encoding of the new
        // transcript, is accepted first.
        let (plan, messages, _) = hidden_run();
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_ok());
        let bytes = messages.encode().unwrap();
        let decoded = HiddenLinMessages::decode(&plan, &bytes).unwrap();
        assert_eq!(decoded, messages);

        // Changed U row: one round coefficient.
        let (plan, mut messages, _) = hidden_run();
        messages.second.dot.as_mut().unwrap().rounds[0].coefficients[0] =
            messages.second.dot.as_ref().unwrap().rounds[0].coefficients[0].add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_err());

        // Changed hiding sum, the first value of the `(U)` row.
        let (plan, mut messages, _) = hidden_run();
        messages.second.dot.as_mut().unwrap().hiding_sum =
            messages.second.dot.as_ref().unwrap().hiding_sum.add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_err());

        // Changed end value of the first `(E_j)` row.
        let (plan, mut messages, _) = hidden_run();
        messages.second.dot.as_mut().unwrap().end_values[0] =
            messages.second.dot.as_ref().unwrap().end_values[0].add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_err());

        // Changed end value of the second `(E_j)` row.
        let (plan, mut messages, _) = hidden_run();
        messages.second.dot.as_mut().unwrap().end_values[1] =
            messages.second.dot.as_ref().unwrap().end_values[1].add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_err());

        // Wrong challenge order: the last (Y) round is sent in the first
        // position, so the challenge drawn after it differs.
        let (plan, mut messages, _) = hidden_run();
        if let Some(dot) = messages.second.dot.as_mut() {
            let y_round = dot.y_round.clone();
            dot.rounds[0] = y_round;
        }
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_err());

        // Changed shown-map commitment.
        let (plan, mut messages, _) = hidden_run();
        messages.first.dot.as_mut().unwrap().root = Digest([0x68; 32]);
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_err());

        // Changed Omega commitment.
        let (plan, mut messages, _) = hidden_run();
        messages.shared.omega_root = Digest([0x69; 32]);
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_err());

        // Missing shown-map message: the U and E rows cannot be built.
        let (plan, mut messages, _) = hidden_run();
        messages.first.dot = None;
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_err());
        let (plan, mut messages, _) = hidden_run();
        messages.second.dot = None;
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_err());
        let (mut plan, _messages, _) = hidden_run();
        plan.send.dot = None;
        assert!(plan.validate().is_err());

        // A changed or reused mask slot is rejected before proof work: every
        // class keeps its own entries and the claim and tilde slots are never
        // the same slot.
        let allocation = MaskEntryAllocation::new(16).unwrap();
        for position in 0..16u32 {
            assert_ne!(
                allocation.claim_slot(position).unwrap(),
                allocation.tilde_slot(position).unwrap()
            );
            assert_ne!(
                allocation.claim_slot(position).unwrap(),
                allocation.eval_slot(position).unwrap()
            );
            assert_ne!(
                allocation.tilde_slot(position).unwrap(),
                allocation.eval_slot(position).unwrap()
            );
        }
        let (mut plan, _messages, _) = hidden_run();
        plan.send.index_mask.as_mut().unwrap().entries.claim = 0;
        assert!(plan.validate().is_err());
        let (mut plan, _messages, _) = hidden_run();
        plan.send.index_mask.as_mut().unwrap().entries.tilde = 0;
        assert!(plan.validate().is_err());
        let (mut plan, _messages, _) = hidden_run();
        plan.send.index_mask.as_mut().unwrap().entries.end = 0;
        assert!(plan.validate().is_err());
        let (mut plan, _messages, _) = hidden_run();
        plan.send.index_mask.as_mut().unwrap().entries.hiding = 0;
        assert!(plan.validate().is_err());
        let (mut plan, _messages, _) = hidden_run();
        plan.send.index_mask.as_mut().unwrap().entries.variables = 0;
        assert!(plan.validate().is_err());

        // The shown-map plan must carry the hidden entry count and a tree that
        // does not collide with the reduced-row trees.
        let (mut plan, _messages, _) = hidden_run();
        plan.send.dot.as_mut().unwrap().hidden = 1;
        assert!(plan.validate().is_err());
        let (mut plan, _messages, _) = hidden_run();
        plan.send.dot.as_mut().unwrap().shape.tree_number = plan.send.w.tree_number;
        assert!(plan.validate().is_err());
        let (mut plan, _messages, _) = hidden_run();
        plan.send.dot.as_mut().unwrap().shape.answers = 2;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn hidden_dot_residual_is_independent_and_batched() {
        let (plan, messages, witness) = hidden_run();
        let table = table();
        let outcome = verify_hidden_lin_hidden(&plan, &table.reader, &messages).unwrap();
        let binding = outcome.dot.as_ref().expect("the hidden Dot binding");
        let index = plan.send.index_mask.as_ref().unwrap();
        let mask_witness = witness.send.index_mask.as_ref().unwrap();
        let positions = POSITIONS as usize;

        // The claim and tilde entries keep independent draws: at least one
        // position of the deterministic fixture differs.
        let mut independent = false;
        let mut expected_residual = Fp3::ZERO;
        let mut power = Fp3::ONE;
        for position in 0..POSITIONS {
            let claim = mask_witness
                .entry(index, index.entries.claim_slot(position).unwrap())
                .unwrap();
            let tilde = mask_witness
                .entry(index, index.entries.tilde_slot(position).unwrap())
                .unwrap();
            if claim != tilde {
                independent = true;
            }
            expected_residual = expected_residual.add(&power.mul(&tilde.sub(&claim)));
            power = power.mul(&binding.chi);
        }
        assert!(independent, "the claim and tilde draws must be independent");
        assert_eq!(binding.residual, expected_residual);
        assert_eq!(messages.second.residual, Some(expected_residual));
        assert_ne!(expected_residual, Fp3::ZERO);

        // The verifier rebuilds the Eval and tilde rows followed by exactly
        // four Dot rows: one difference row with both entries for every
        // position, then the (U) and (E_j) rows. No per-position tie row
        // survives.
        let (members, _) =
            own_members(&plan, &messages.first, &messages.shared, None, None).unwrap();
        let rows = own_plain_rows(
            &plan,
            &members,
            &messages.first,
            &outcome.shared_point,
            &messages.second.tilde,
            Some(binding),
            None,
            None,
        )
        .unwrap();
        assert_eq!(rows.len(), 2 * positions + 4);
        let dot_rows = &rows[2 * positions..];
        assert_eq!(dot_rows.len(), 4);
        let difference = &dot_rows[0];
        assert_eq!(difference.kind, BatRowKind::Plain);
        assert_eq!(difference.target, binding.residual);
        assert!(difference.terms.is_empty());
        assert_eq!(difference.units.len(), 2 * positions);
        let mut power = Fp3::ONE;
        for position in 0..positions {
            let tilde_slot = index.entries.tilde_slot(position as u32).unwrap();
            let claim_slot = index.entries.claim_slot(position as u32).unwrap();
            assert_eq!(difference.units[2 * position].part, BatPart::Mask);
            assert_eq!(
                difference.units[2 * position].index,
                index.entry_virtual_index(tilde_slot).unwrap()
            );
            assert_eq!(difference.units[2 * position].coefficient, power);
            assert_eq!(difference.units[2 * position + 1].part, BatPart::Mask);
            assert_eq!(
                difference.units[2 * position + 1].index,
                index.entry_virtual_index(claim_slot).unwrap()
            );
            assert_eq!(
                difference.units[2 * position + 1].coefficient,
                Fp3::ZERO.sub(&power)
            );
            power = power.mul(&binding.chi);
        }
        assert_eq!(dot_rows[1].target, binding.output.row_target);
        assert_eq!(dot_rows[2].target, binding.output.ends[0].value);
        assert_eq!(dot_rows[3].target, binding.output.ends[1].value);
    }

    #[test]
    fn public_mask_view_rank_leaves_the_expected_kernel() {
        // The public mask-dependent linear forms on (g, c) are the sixteen
        // tilde values and the residual. For P positions they have rank P + 1
        // and kernel P - 1, the claim-mask directions orthogonal to chi.
        let positions = 16usize;
        let chi = Fp3::from_u64(0x1234_5678);
        let mut matrix = vec![vec![Fp3::ZERO; 2 * positions]; positions + 1];
        let mut power = Fp3::ONE;
        for position in 0..positions {
            matrix[position][position] = Fp3::ONE;
            matrix[positions][position] = power;
            matrix[positions][positions + position] = Fp3::ZERO.sub(&power);
            power = power.mul(&chi);
        }
        assert_eq!(rank_over_fp3(&matrix), positions + 1);
    }

    #[test]
    fn hidden_dot_residual_negatives_reach_the_verifier() {
        let table = table();
        let (plan, messages, _) = hidden_run();
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_ok());

        // A changed residual moves both the absorbed transcript value and the
        // adjusted target.
        let (plan, mut messages, _) = hidden_run();
        let residual = messages.second.residual.unwrap();
        messages.second.residual = Some(residual.add(&Fp3::ONE));
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_err());

        // The opposite sign convention is a different relation.
        let (plan, mut messages, _) = hidden_run();
        let residual = messages.second.residual.unwrap();
        messages.second.residual = Some(Fp3::ZERO.sub(&residual));
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_err());

        // The residual must be present exactly when the Dot proof is present.
        let (plan, mut messages, _) = hidden_run();
        messages.second.residual = None;
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_err());
        let (plan, mut messages, _) = hidden_run();
        messages.second.dot = None;
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_err());

        // A residual that is not absorbed before the product-sumcheck
        // challenges leaves the prover and verifier on different challenges.
        let (plan, messages, _) = hidden_run_unbound_residual();
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &messages).is_err());
    }

    #[test]
    fn hidden_dot_changed_mask_entries_are_rejected_by_the_plaintext_check() {
        let (plan, messages, witness) = hidden_run();
        let table = table();
        let outcome = verify_hidden_lin_hidden(&plan, &table.reader, &messages).unwrap();
        assert!(
            verify_hidden_lin_hidden_with_witness(&plan, &table.reader, &messages, &witness)
                .is_ok()
        );
        let index = plan.send.index_mask.as_ref().unwrap();
        let ell = index.mask_code.ell as usize;

        let changed_entry = |slot: usize| {
            let mut changed = witness.clone();
            let rows = &mut changed.send.index_mask.as_mut().unwrap().rows;
            rows[slot / ell][slot % ell] = rows[slot / ell][slot % ell].add(&Fp3::ONE);
            changed
        };

        // A changed tilde mask breaks the tilde row and the residual.
        let changed = changed_entry(index.entries.tilde_slot(0).unwrap());
        assert!(
            check_hidden_dot_mask_relation(
                &plan,
                &messages,
                &changed.send,
                &changed.omega_message,
                &outcome,
            )
            .is_err()
        );
        // A changed claim mask breaks the residual and the (U) row.
        let changed = changed_entry(index.entries.claim_slot(0).unwrap());
        assert!(
            check_hidden_dot_mask_relation(
                &plan,
                &messages,
                &changed.send,
                &changed.omega_message,
                &outcome,
            )
            .is_err()
        );
        // A changed hiding coefficient breaks the (U) row.
        let changed = changed_entry(index.entries.hiding_range().unwrap().start);
        assert!(
            check_hidden_dot_mask_relation(
                &plan,
                &messages,
                &changed.send,
                &changed.omega_message,
                &outcome,
            )
            .is_err()
        );
        // A changed end mask breaks an (E_j) row.
        let changed = changed_entry(index.entries.end_range().unwrap().start);
        assert!(
            check_hidden_dot_mask_relation(
                &plan,
                &messages,
                &changed.send,
                &changed.omega_message,
                &outcome,
            )
            .is_err()
        );

        // A wrong residual in the checked binding changes the difference-row
        // target and the adjusted PSC target; the plaintext check rejects it.
        let mut wrong_target = outcome.clone();
        wrong_target.dot.as_mut().unwrap().residual = outcome
            .dot
            .as_ref()
            .unwrap()
            .residual
            .add(&Fp3::ONE);
        assert!(
            check_hidden_dot_mask_relation(
                &plan,
                &messages,
                &witness.send,
                &witness.omega_message,
                &wrong_target,
            )
            .is_err()
        );
    }

    #[test]
    fn hidden_dot_residual_encoding_is_canonical() {
        let (plan, messages, witness) = hidden_run();
        let bytes = messages.encode().unwrap();
        let decoded = HiddenLinMessages::decode(&plan, &bytes).unwrap();
        assert_eq!(decoded, messages);
        assert_eq!(decoded.second.residual, messages.second.residual);

        // No trailing bytes are accepted.
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(HiddenLinMessages::decode(&plan, &trailing).is_err());

        // An individual per-position difference is never a valid extra field:
        // appending one rejects the canonical encoding.
        let index = plan.send.index_mask.as_ref().unwrap();
        let mask_witness = witness.send.index_mask.as_ref().unwrap();
        let mut leaked = bytes;
        for position in 0..POSITIONS {
            let claim = mask_witness
                .entry(index, index.entries.claim_slot(position).unwrap())
                .unwrap();
            let tilde = mask_witness
                .entry(index, index.entries.tilde_slot(position).unwrap())
                .unwrap();
            let mut writer = CanonicalWriter::new();
            writer.put_fp3(&tilde.sub(&claim));
            leaked.extend_from_slice(&writer.into_bytes());
        }
        assert!(HiddenLinMessages::decode(&plan, &leaked).is_err());
    }

    #[test]
    fn mask_slot_schema_is_separate_and_one_time() {
        let allocation = MaskEntryAllocation::new(POSITIONS).unwrap();
        assert_eq!(
            allocation.total().unwrap(),
            23 + 2 + 3 * POSITIONS as usize
        );
        // The production position count fixes the 73-entry fallback schema.
        assert_eq!(MaskEntryAllocation::new(16).unwrap().total().unwrap(), 73);
        assert!(allocation.is_separate_mask());
        let hiding = allocation.hiding_range().unwrap();
        let end = allocation.end_range().unwrap();
        let claim = allocation.claim_range().unwrap();
        let tilde = allocation.tilde_range().unwrap();
        let eval = allocation.eval_range().unwrap();
        assert_eq!(hiding.len(), 23);
        assert_eq!(end.len(), 2);
        assert_eq!(claim.len(), POSITIONS as usize);
        assert_eq!(tilde.len(), POSITIONS as usize);
        assert_eq!(eval.len(), POSITIONS as usize);
        assert!(hiding.end <= end.start && end.end <= claim.start);
        assert!(claim.end <= tilde.start && tilde.end <= eval.start);
        for left in [&hiding, &end, &claim, &tilde, &eval] {
            for right in [&hiding, &end, &claim, &tilde, &eval] {
                if !std::ptr::eq(left, right) {
                    assert!(left.end <= right.start || right.end <= left.start);
                }
            }
        }
        // Each position has its own Eval entry; a claim or tilde entry can
        // never be the Eval entry.
        let mut seen = std::collections::BTreeSet::new();
        for position in 0..POSITIONS {
            let eval_slot = allocation.eval_slot(position).unwrap();
            assert!(seen.insert(eval_slot));
            assert!(!claim.contains(&eval_slot));
            assert!(!tilde.contains(&eval_slot));
            assert!(!hiding.contains(&eval_slot));
            assert!(!end.contains(&eval_slot));
        }
        assert!(allocation.eval_slot(POSITIONS).is_err());

        // A short or overlapping schema is rejected before any proof work.
        let mut short = allocation;
        short.eval = 0;
        assert!(short.check().is_err());
        assert!(short.eval_slot(0).is_err());
        let mut merged = allocation;
        merged.tilde = 0;
        assert!(merged.check().is_err());
        let mut wrong_hiding = allocation;
        wrong_hiding.hiding = 21;
        assert!(wrong_hiding.check().is_err());
        // The plan carries the same checked allocation.
        let (mut plan, _, _) = hidden_run();
        plan.send.index_mask.as_mut().unwrap().entries = short;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn merged_toy_run_accepts() {
        let (plan, messages, witness) = run();
        let activation = activation();
        let output = output_values(&activation);
        let table = table();
        let outcome =
            verify_hidden_lin_with_witness(&plan, &table.reader, &output, &messages, &witness)
                .unwrap();
        assert_eq!(outcome.eval_rows.len(), POSITIONS as usize);
        let shared_base = outcome.shared_base.single().expect("the fixture ends in BASE");
        let own_base = outcome.own_base.single().expect("the fixture ends in BASE");
        assert!(shared_base.fold_main);
        assert!(own_base.fold_main);
        assert!(shared_base.base_linear);
        assert!(own_base.base_linear);
        assert_eq!(outcome.first_seal, messages.first_seal);
        assert_eq!(outcome.shared_seal, messages.shared.seal);
        assert_eq!(outcome.seal, messages.second.seal);
        assert_ne!(outcome.first_seal, outcome.shared_seal);
        assert_ne!(outcome.shared_seal, outcome.seal);
        let bytes = messages.encode().unwrap();
        let decoded = HiddenLinMessages::decode(&plan, &bytes).unwrap();
        assert_eq!(decoded, messages);
    }

    #[test]
    fn old_or_unknown_session_layout_is_rejected() {
        let (plan, messages, _witness) = run();
        // The current schema with the previous single-stage layout is an
        // unsupported version, not a silent fallback.
        let mut old = CanonicalWriter::new();
        old.put_u64(HIDDEN_SEND_SCHEMA_VERSION);
        old.put_u64(13);
        assert_eq!(
            HiddenLinMessages::decode(&plan, &old.into_bytes()).unwrap_err(),
            Error::UnsupportedVersion
        );
        // A newer unknown layout cannot decode either.
        let mut newer = CanonicalWriter::new();
        newer.put_u64(HIDDEN_SEND_SCHEMA_VERSION);
        newer.put_u64(15);
        assert_eq!(
            HiddenLinMessages::decode(&plan, &newer.into_bytes()).unwrap_err(),
            Error::InvalidEncoding
        );
        // The current layout still round trips.
        let bytes = messages.encode().unwrap();
        assert_eq!(HiddenLinMessages::decode(&plan, &bytes).unwrap(), messages);
    }

    #[test]
    fn a_staged_plan_rejects_the_single_stage_tail() {
        let (plan, messages, _witness) = run();
        let staged = staged_plan(&plan);
        // The trusted plan derives a loop chain for both groups.
        assert!(proximity_iopp_plan(
            &staged.proximity,
            &staged.send,
            staged.closure.as_ref(),
            staged.norm.as_ref(),
            ProximityGroupRole::Shared,
        )
        .unwrap()
        .is_some());
        assert!(proximity_iopp_plan(
            &staged.proximity,
            &staged.send,
            staged.closure.as_ref(),
            staged.norm.as_ref(),
            ProximityGroupRole::Own,
        )
        .unwrap()
        .is_some());
        // An artifact whose tails are the single-stage artifacts cannot
        // decode against a plan that requires the loop: there is no silent
        // fallback to the old delivery.
        let bytes = messages.encode().unwrap();
        assert_eq!(
            HiddenLinMessages::decode(&staged, &bytes).unwrap_err(),
            Error::ShapeMismatch
        );
    }

    #[test]
    fn the_two_groups_share_one_registry() {
        let (plan, _messages, _witness) = run();
        let staged = staged_plan(&plan);
        let shared = proximity_iopp_plan(
            &staged.proximity,
            &staged.send,
            staged.closure.as_ref(),
            staged.norm.as_ref(),
            ProximityGroupRole::Shared,
        )
        .unwrap()
        .expect("the shared group has stages");
        let own = proximity_iopp_plan(
            &staged.proximity,
            &staged.send,
            staged.closure.as_ref(),
            staged.norm.as_ref(),
            ProximityGroupRole::Own,
        )
        .unwrap()
        .expect("the own group has stages");
        // The two groups and stage 0 allocate from one registry: a tree
        // number or a stage FOLD identifier is never bound twice.
        for shared_tree in &shared.chain.config.switch_tree_numbers {
            assert!(!own.chain.config.switch_tree_numbers.contains(shared_tree));
        }
        for shared_tree in &shared.chain.config.fold_tree_numbers {
            assert!(!own.chain.config.fold_tree_numbers.contains(shared_tree));
        }
        assert_ne!(
            shared.chain.config.base_star.tree_number,
            own.chain.config.base_star.tree_number
        );
        let mut shared_ids: Vec<u64> = Vec::new();
        for ids in &shared.chain.config.fold_ids {
            shared_ids.extend([ids.group, ids.member, ids.fresh]);
        }
        for ids in &own.chain.config.fold_ids {
            for id in [ids.group, ids.member, ids.fresh] {
                assert!(!shared_ids.contains(&id));
            }
        }
    }

    #[test]
    fn hidden_input_mode_accepts_without_output_rows() {
        let activation = activation();
        let diagnostic_plan = plan(&activation);
        let mut plan = diagnostic_plan.clone();
        plan.weights = Vec::new();
        plan.send.index_mask = Some(index_mask_plan());
        plan.send.dot = Some(dot_plan());
        plan.proximity.own_mask_tree.mask_rows = 4;
        let mut table = table();
        let (messages, witness) =
            prove_hidden_lin_hidden(&plan, &activation, &mut table, Digest([0x71; 32])).unwrap();
        // The Eval targets are `W_t(psi_t) + e_t`; the mask entry is not the
        // plain evaluation.
        let expected = witness.send.eval_values(&messages.first.psi).unwrap();
        let index = plan.send.index_mask.as_ref().unwrap();
        let mask_witness = witness.send.index_mask.as_ref().unwrap();
        for position in 0..POSITIONS {
            let mask = mask_witness
                .entry(index, index.entries.eval_slot(position).unwrap())
                .unwrap();
            assert_eq!(
                messages.first.eval_targets[position as usize],
                expected[position as usize].add(&mask)
            );
            assert_ne!(
                messages.first.eval_targets[position as usize],
                expected[position as usize]
            );
        }
        let outcome = verify_hidden_lin_hidden(&plan, &table.reader, &messages).unwrap();
        assert_eq!(outcome.eval_rows.len(), POSITIONS as usize);
        let own_base = outcome.own_base.single().expect("the fixture ends in BASE");
        assert!(own_base.fold_main);
        assert!(own_base.base_linear);
        // A changed target no longer matches the committed row.
        let mut changed = messages.clone();
        changed.first.eval_targets[0] = changed.first.eval_targets[0].add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&plan, &table.reader, &changed).is_err());
        // The diagnostic entries require the other mode.
        let output = output_values(&activation);
        assert!(verify_hidden_lin(&plan, &table.reader, &output, &messages).is_err());
        assert!(verify_hidden_lin_hidden(&diagnostic_plan, &table.reader, &messages).is_err());
    }

    fn assert_reject(result: Result<HiddenLinOutcome>) {
        assert!(result.is_err(), "the altered message was accepted");
    }

    #[test]
    fn changed_activation_commitment_is_rejected() {
        let (plan, mut messages, witness) = run();
        let activation = activation();
        let output = output_values(&activation);
        let table = table();
        messages.first.activation.root = Digest([0x77; 32]);
        assert_reject(verify_hidden_lin_with_witness(
            &plan,
            &table.reader,
            &output,
            &messages,
            &witness,
        ));
    }

    #[test]
    fn changed_w_root_or_row_is_rejected() {
        let (plan, mut messages, witness) = run();
        let activation = activation();
        let output = output_values(&activation);
        let table = table();
        messages.first.group.entries[0].root = Digest([0x66; 32]);
        assert_reject(verify_hidden_lin_with_witness(
            &plan,
            &table.reader,
            &output,
            &messages,
            &witness,
        ));
        let (plan, mut messages, witness) = run();
        messages.first.openings[0].columns[0].symbols[0] = Fp::from_u64(0xDEAD);
        assert_reject(verify_hidden_lin_with_witness(
            &plan,
            &table.reader,
            &output,
            &messages,
            &witness,
        ));
    }

    #[test]
    fn changed_table_root_or_record_is_rejected() {
        let (plan, messages, witness) = run();
        let activation = activation();
        let output = output_values(&activation);
        let mut table = table();
        table.reader.root.0[0] ^= 1;
        assert_reject(verify_hidden_lin_with_witness(
            &plan,
            &table.reader,
            &output,
            &messages,
            &witness,
        ));
    }

    #[test]
    fn changed_plan_or_statement_is_rejected() {
        let (plan, messages, witness) = run();
        let activation = activation();
        let output = output_values(&activation);
        let table_handle = table();
        let mut changed = plan.clone();
        changed.weights[0][0] = changed.weights[0][0].add(&Fp3::ONE);
        assert_reject(verify_hidden_lin_with_witness(
            &changed,
            &table_handle.reader,
            &output,
            &messages,
            &witness,
        ));
        let mut changed = plan.clone();
        changed.send.activation_root = Digest([0x78; 32]);
        assert_reject(verify_hidden_lin_with_witness(
            &changed,
            &table_handle.reader,
            &output,
            &messages,
            &witness,
        ));
        let mut changed = plan.clone();
        changed.proximity.column_draws += 1;
        assert_reject(verify_hidden_lin_with_witness(
            &changed,
            &table_handle.reader,
            &output,
            &messages,
            &witness,
        ));
        // The prover also rejects a plan whose public weights do not match the
        // committed witness: the Dot row no longer holds on its own group.
        let mut changed = plan.clone();
        changed.weights[0][0] = changed.weights[0][0].add(&Fp3::ONE);
        changed.send.expected_roots = None;
        let mut table2 = table_handle;
        assert!(prove_hidden_lin(
            &changed,
            &activation,
            &mut table2,
            &output,
            Digest([0x71; 32]),
        )
        .is_err());
    }

    #[test]
    fn changed_shared_values_are_rejected() {
        let (plan, mut messages, witness) = run();
        let activation = activation();
        let output = output_values(&activation);
        let table = table();
        messages.shared.gu_root.0[0] ^= 1;
        assert_reject(verify_hidden_lin_with_witness(
            &plan,
            &table.reader,
            &output,
            &messages,
            &witness,
        ));
        let (plan, mut messages, witness) = run();
        messages.shared.omega_answers[0] =
            messages.shared.omega_answers[0].add(&Fp3::ONE);
        assert_reject(verify_hidden_lin_with_witness(
            &plan,
            &table.reader,
            &output,
            &messages,
            &witness,
        ));
        let (plan, mut messages, witness) = run();
        messages.shared.zeta[0] = messages.shared.zeta[0].add(&Fp3::ONE);
        assert_reject(verify_hidden_lin_with_witness(
            &plan,
            &table.reader,
            &output,
            &messages,
            &witness,
        ));
        let (plan, mut messages, witness) = run();
        messages.first.psi[0][0] = messages.first.psi[0][0].add(&Fp3::ONE);
        assert_reject(verify_hidden_lin_with_witness(
            &plan,
            &table.reader,
            &output,
            &messages,
            &witness,
        ));
        let (plan, mut messages, witness) = run();
        messages.shared.seal.0[0] ^= 1;
        assert_reject(verify_hidden_lin_with_witness(
            &plan,
            &table.reader,
            &output,
            &messages,
            &witness,
        ));
    }

    #[test]
    fn changed_second_values_are_rejected() {
        let (plan, mut messages, witness) = run();
        let activation = activation();
        let output = output_values(&activation);
        let table = table();
        messages.second.omega_root.0[0] ^= 1;
        assert_reject(verify_hidden_lin_with_witness(
            &plan,
            &table.reader,
            &output,
            &messages,
            &witness,
        ));
        let (plan, mut messages, witness) = run();
        messages.second.seal.0[0] ^= 1;
        assert_reject(verify_hidden_lin_with_witness(
            &plan,
            &table.reader,
            &output,
            &messages,
            &witness,
        ));
        let (plan, mut messages, witness) = run();
        messages.second.fold_rounds[0].0 =
            messages.second.fold_rounds[0].0.add(&Fp3::ONE);
        assert_reject(verify_hidden_lin_with_witness(
            &plan,
            &table.reader,
            &output,
            &messages,
            &witness,
        ));
        let (plan, mut messages, witness) = run();
        match &mut messages.second.tail {
            ProximityTail::Single { base_rows, .. } => {
                base_rows.base.lambda = base_rows.base.lambda.add(&Fp3::ONE);
            }
            ProximityTail::Stages(_) => panic!("the fixture has no stages"),
        }
        assert_reject(verify_hidden_lin_with_witness(
            &plan,
            &table.reader,
            &output,
            &messages,
            &witness,
        ));
    }

    #[test]
    fn changed_output_is_rejected() {
        let (plan, messages, witness) = run();
        let activation = activation();
        let mut output = output_values(&activation);
        let table = table();
        output[0][0] = output[0][0].add(&Fp3::ONE);
        assert_reject(verify_hidden_lin_with_witness(
            &plan,
            &table.reader,
            &output,
            &messages,
            &witness,
        ));
    }

    #[test]
    fn changed_column_position_or_nonce_is_rejected() {
        let (plan, mut messages, witness) = run();
        let activation = activation();
        let output = output_values(&activation);
        let table = table();
        messages.shared.column_positions[0] ^= 1;
        assert_reject(verify_hidden_lin_with_witness(
            &plan,
            &table.reader,
            &output,
            &messages,
            &witness,
        ));
        let (plan, mut messages, witness) = run();
        messages.shared.column_nonce ^= 1;
        assert_reject(verify_hidden_lin_with_witness(
            &plan,
            &table.reader,
            &output,
            &messages,
            &witness,
        ));
    }

    #[test]
    fn column_budget_is_enforced() {
        let (mut plan, messages, witness) = run();
        plan.proximity.column_draws = plan.proximity.column_budget + 1;
        assert!(plan.validate().is_err());
        let activation = activation();
        let output = output_values(&activation);
        let table = table();
        assert_reject(verify_hidden_lin_with_witness(
            &plan,
            &table.reader,
            &output,
            &messages,
            &witness,
        ));
    }

    #[test]
    fn missing_stage_or_seal_is_rejected() {
        let (plan, messages, witness) = run();
        let activation = activation();
        let output = output_values(&activation);
        let table = table();
        let mut missing = messages.clone();
        missing.second.fold_rounds.clear();
        assert_reject(verify_hidden_lin_with_witness(
            &plan,
            &table.reader,
            &output,
            &missing,
            &witness,
        ));
        let mut missing = messages.clone();
        missing.shared.fold_rounds.clear();
        assert_reject(verify_hidden_lin_with_witness(
            &plan,
            &table.reader,
            &output,
            &missing,
            &witness,
        ));
        let mut changed = messages.clone();
        changed.first_seal.0[0] ^= 1;
        assert_reject(verify_hidden_lin_with_witness(
            &plan,
            &table.reader,
            &output,
            &changed,
            &witness,
        ));
    }

    #[test]
    fn proximity_tail_union_is_enforced_by_the_plan() {
        let activation = activation();
        let base = hidden_plan(&activation);
        base.validate().expect("baseline plan");
        let proximity = base.proximity.clone();
        let own = proximity.own_base_positions as u64;

        let mut send = base.send.clone();
        send.openings_per_oracle = (send.w.code.r + 1 - own) as usize;
        assert_eq!(
            proximity.fault(&send, None, None),
            Some("the W SEND and own-BASE openings exceed the code tail")
        );

        let mut send = base.send.clone();
        let index = send.index_mask.as_mut().expect("index mask");
        index.openings = (index.mask_code.r + 1 - own) as usize;
        assert_eq!(
            proximity.fault(&send, None, None),
            Some("the index-mask SEND and own-BASE openings exceed the mask tail")
        );

        let mut wide = proximity.clone();
        wide.shared_base_positions = (base.proximity.code.r + 1 - own) as usize;
        assert_eq!(
            wide.fault(&base.send, None, None),
            Some("the Omega shared and own-BASE openings exceed the code tail")
        );

        let mut budget = proximity.clone();
        budget.column_draws = base.send.table_shape.code.r as usize + 1;
        budget.column_budget = budget.column_draws;
        assert_eq!(
            budget.fault(&base.send, None, None),
            Some("the table column budget exceeds the table code tail")
        );

        let mut shown = base.send.clone();
        shown.dot.as_mut().expect("shown map").shape.code.r = own - 1;
        assert!(proximity.fault(&shown, None, None).is_some());
        assert!(base.validate().is_ok());
    }

    fn norm_output_plan() -> crate::proof::hidden_norm_psc::NormOutputPlan {
        use crate::proof::hidden_norm_psc::NormGroupPlan;

        let group = |id: u64, oracle: u64, tree: u64| NormGroupPlan {
            shape: WShape::new(
                COLUMNS,
                code(),
                1,
                POSITIONS,
                SALT,
                tree,
                1,
                W_RUN,
                W_CONFIGURATION,
            )
            .unwrap(),
            columns: COLUMNS,
            group: id,
            oracle_base: oracle,
            openings: 1,
        };
        crate::proof::hidden_norm_psc::NormOutputPlan {
            broadcast: group(90, 20_000, 61_000),
            product: group(91, 20_100, 61_100),
            image: group(92, 20_200, 61_200),
            out: group(93, 20_300, 61_300),
            shift: 3,
            clamp: Some((-1_000, 1_000)),
            tail: Some(crate::proof::hidden_norm_psc::NormOutputTailPlan {
                group: NormGroupPlan {
                    shape: WShape::new(
                        256,
                        code(),
                        1,
                        POSITIONS,
                        SALT,
                        61_400,
                        1,
                        W_RUN,
                        W_CONFIGURATION,
                    )
                    .unwrap(),
                    columns: 256,
                    group: 94,
                    oracle_base: 20_400,
                    openings: 1,
                },
                widths: crate::proof::hidden_norm_psc::NormOutputTailWidths {
                    quotient: 3,
                    remainder: 2,
                    doubled_remainder: 2,
                    tie_slack: 2,
                    shifted: 3,
                    pre_clamp: 3,
                },
                clamp: Some(crate::proof::hidden_norm_psc::NormClampWidths {
                    low: 4,
                    high: 4,
                    low_strict: 4,
                    low_nonstrict: 4,
                    high_strict: 4,
                    high_nonstrict: 4,
                }),
            }),
        }
    }

    fn norm_output_fixture(
        activation: &CommittedActivation,
    ) -> (HiddenLinPlan, NormWitness, ParamTables) {
        let clamp = norm_output_plan().clamp;
        norm_output_fixture_with_clamp(activation, clamp)
    }

    fn norm_output_fixture_with_clamp(
        activation: &CommittedActivation,
        clamp: Option<(i64, i64)>,
    ) -> (HiddenLinPlan, NormWitness, ParamTables) {
        use crate::proof::hidden_norm::{NormCoreRow, NormOutputRow};
        use crate::proof::hidden_norm_psc::{NormOutputTailWitness, NormOutputWitness};

        let (mut plan, mut witness, params) = norm_params_fixture(activation);
        let mut output_plan = norm_output_plan();
        output_plan.clamp = clamp;
        {
            let norm = plan.norm.as_mut().unwrap();
            norm.output = Some(output_plan.clone());
            let mask = norm.mask.as_mut().unwrap();
            mask.pointwise = 2;
            mask.tail = crate::proof::hidden_norm_psc::NORM_OUTPUT_TAIL_RELATIONS;
            mask.claims = usize::try_from(norm.shape.positions)
                .unwrap()
                .checked_mul(2)
                .unwrap();
        }
        let gain = vec![2i64, -1, 3, 1];
        let bias = vec![1i64, 2, -1, 2];
        let shift = output_plan.shift;
        let clamp = output_plan.clamp;
        let mut rows = Vec::new();
        for row in activation_rows() {
            let core = NormCoreRow::from_values(&row, HIDDEN as usize).unwrap();
            let mut padded = row.clone();
            padded.resize(COLUMNS as usize, 0);
            rows.push(
                NormOutputRow::from_core(
                    &core,
                    &padded,
                    100,
                    &gain,
                    &bias,
                    HIDDEN as usize,
                    shift,
                    clamp,
                )
                .unwrap(),
            );
        }
        let mut coins = ByteMaskCoins::new(Digest([0x92; 32])).unwrap();
        let signed = |values: Vec<i128>| -> Vec<Fp3> {
            values
                .into_iter()
                .map(|value| crate::proof::output_closure::fp3_signed(value).unwrap())
                .collect()
        };
        let column = |values: &[i128]| signed(values.to_vec());
        let broadcast_values = rows
            .iter()
            .map(|row| column(&row.broadcast.iter().map(|value| *value as i128).collect::<Vec<_>>()))
            .collect::<Vec<_>>();
        let product_values = rows
            .iter()
            .map(|row| column(&row.product))
            .collect::<Vec<_>>();
        let image_values = rows
            .iter()
            .map(|row| column(&row.image))
            .collect::<Vec<_>>();
        let out_values = rows
            .iter()
            .map(|row| column(&row.out.iter().map(|value| *value as i128).collect::<Vec<_>>()))
            .collect::<Vec<_>>();
        let tail_plan = output_plan.tail.as_ref().unwrap();
        let tail_values = rows
            .iter()
            .map(|row| tail_plan.values(row, COLUMNS).unwrap())
            .collect::<Vec<_>>();
        let tail = NormOutputTailWitness {
            committed: commit_norm_group(&tail_plan.group, &tail_values, &mut coins),
        };
        let output = NormOutputWitness {
            tail: Some(tail),
            broadcast: commit_norm_group(&output_plan.broadcast, &broadcast_values, &mut coins),
            product: commit_norm_group(&output_plan.product, &product_values, &mut coins),
            image: commit_norm_group(&output_plan.image, &image_values, &mut coins),
            out: commit_norm_group(&output_plan.out, &out_values, &mut coins),
            rows,
        };
        witness.output = Some(output);
        plan.validate().unwrap();
        (plan, witness, params)
    }

    fn slack_lookup_bits(
        plan: &crate::proof::lookup::LookupPlan,
        keys: &[Fp3],
    ) -> Vec<Fp3> {
        let planes = plan.planes().unwrap();
        let rows = plan.table_rows().unwrap();
        let mut counts = vec![0u64; rows];
        for key in keys {
            let mut found = false;
            for (row, count) in counts.iter_mut().enumerate() {
                if *key == Fp3::from_u64(row as u64) {
                    *count += 1;
                    found = true;
                    break;
                }
            }
            assert!(found, "a committed slack digit is not a byte");
        }
        let mut bits = vec![Fp3::ZERO; planes * rows];
        for row in 0..rows {
            for plane in 0..planes {
                if (counts[row] >> plane) & 1 == 1 {
                    bits[plane + planes * row] = Fp3::ONE;
                }
            }
        }
        bits
    }

    fn slack_lookup_params() -> crate::proof::hidden_norm_psc::NormLookupParams {
        crate::proof::hidden_norm_psc::NormLookupParams {
            code: code(),
            mask_code: mask_code(),
            group: GROUP,
            bits_oracle: 9900,
            mask_oracle: 9901,
            bits_tree: 60980,
            mask_tree: 60981,
            leaf_class: 1,
            salt_bytes: SALT,
            run: W_RUN,
            configuration: W_CONFIGURATION,
            padding_byte: 0,
            table_identity: crate::proof::lookup::BYTE_TABLE_IDENTITY,
        }
    }

    /// Attaches the byte-table lookup over the committed rounding digits and
    /// every committed tail digit cell. The multiplicity bits are counted
    /// from the committed messages, so the honest fixture commits the same
    /// keys the lookup argument proves.
    fn attach_slack_lookup(plan: &mut HiddenLinPlan, witness: &mut NormWitness) {
        use crate::proof::hidden_norm_psc::{
            norm_own_mask_rows, NormLookupPlan, NormLookupWitness,
        };

        let norm = plan.norm.as_ref().unwrap();
        let round_plan = norm
            .round
            .clone()
            .expect("the fixture carries the rounding rows");
        let tail_plan = norm
            .output
            .as_ref()
            .and_then(|output| output.tail.as_ref())
            .expect("the output fixture carries the tail")
            .clone();
        let lookup_plan = NormLookupPlan::from_round_and_tail(
            &round_plan,
            &tail_plan,
            norm.columns,
            POSITIONS,
            &slack_lookup_params(),
        )
        .unwrap();
        let keys = {
            let round_rows = &witness.round.as_ref().unwrap().messages;
            let tail_rows = &witness
                .output
                .as_ref()
                .unwrap()
                .tail
                .as_ref()
                .unwrap()
                .committed
                .messages;
            norm_lookup_key_values(plan.norm.as_ref().unwrap(), &lookup_plan, round_rows, tail_rows)
                .unwrap()
        };
        let bits = slack_lookup_bits(&lookup_plan.lookup, &keys);
        let lookup_mask_rows = lookup_plan.commit.mask_rows;
        plan.norm.as_mut().unwrap().lookup = Some(lookup_plan);
        let norm_mask_rows = plan
            .norm
            .as_ref()
            .unwrap()
            .mask
            .as_ref()
            .unwrap()
            .mask_rows;
        let index_rows = plan
            .send
            .index_mask
            .as_ref()
            .map(|index| index.tree_rows().unwrap())
            .unwrap_or(0);
        plan.proximity.own_mask_tree.mask_rows =
            norm_own_mask_rows(index_rows, norm_mask_rows + lookup_mask_rows).unwrap();
        witness.lookup = Some(NormLookupWitness {
            bits,
            messages: None,
        });
        plan.validate().unwrap();
    }

    fn slack_lookup_fixture(
        activation: &CommittedActivation,
    ) -> (HiddenLinPlan, NormWitness, ParamTables) {
        let (mut plan, mut witness, params) = norm_output_fixture(activation);
        attach_slack_lookup(&mut plan, &mut witness);
        (plan, witness, params)
    }

    fn clamp_boundary_fixture(
        activation: &CommittedActivation,
        low: i64,
        high: i64,
    ) -> (HiddenLinPlan, NormWitness, ParamTables) {
        let (mut plan, mut witness, params) =
            norm_output_fixture_with_clamp(activation, Some((low, high)));
        attach_slack_lookup(&mut plan, &mut witness);
        (plan, witness, params)
    }

    #[test]
    fn norm_tail_slack_lookup_proves_and_verifies() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = slack_lookup_fixture(&activation);
        let lookup_plan = plan.norm.as_ref().unwrap().lookup.as_ref().unwrap();
        let tail_group = &plan
            .norm
            .as_ref()
            .unwrap()
            .output
            .as_ref()
            .unwrap()
            .tail
            .as_ref()
            .unwrap()
            .group;
        let round_group = &plan.norm.as_ref().unwrap().round.as_ref().unwrap().group;
        assert_eq!(lookup_plan.key_groups.len(), 2);
        assert_eq!(lookup_plan.key_groups[0].oracle_base, round_group.oracle_base);
        assert_eq!(lookup_plan.key_groups[0].shape, round_group.shape);
        assert_eq!(lookup_plan.key_groups[0].columns, round_group.columns);
        assert_eq!(lookup_plan.key_groups[1].oracle_base, tail_group.oracle_base);
        assert_eq!(lookup_plan.key_groups[1].shape, tail_group.shape);
        assert_eq!(lookup_plan.key_groups[1].columns, tail_group.columns);
        let tail_plan = plan
            .norm
            .as_ref()
            .unwrap()
            .output
            .as_ref()
            .unwrap()
            .tail
            .as_ref()
            .unwrap()
            .clone();
        let digit_groups = crate::proof::hidden_norm_psc::tail_digit_groups(&tail_plan).unwrap();
        assert_eq!(digit_groups.len(), 12);
        let tail_width: usize = digit_groups.iter().map(|(_, width)| *width).sum();
        let norm_columns = usize::try_from(plan.norm.as_ref().unwrap().columns).unwrap();
        let positions = plan.norm.as_ref().unwrap().shape.positions as u64;
        let within = |source: &crate::proof::lookup::LookupKeySource, base: u64| {
            source.oracle >= base && source.oracle < base.saturating_add(positions)
        };
        assert!(
            lookup_plan
                .key
                .sources
                .iter()
                .any(|source| within(source, round_group.oracle_base)),
            "the lookup keys must cover the rounding digits"
        );
        let tail_sources = lookup_plan
            .key
            .sources
            .iter()
            .filter(|source| within(source, tail_group.oracle_base))
            .count();
        assert_eq!(
            tail_sources,
            positions as usize * norm_columns * tail_width,
            "the lookup keys must cover every tail digit cell"
        );

        let (messages, hidden_witness) = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &witness,
            &mut params,
        )
        .unwrap();
        assert!(messages.first.norm.as_ref().unwrap().lookup.is_some());
        let second = messages
            .second
            .norm
            .as_ref()
            .unwrap()
            .lookup
            .as_ref()
            .unwrap();
        assert!(!second.claims.rows.is_empty());

        let mut verifier_plan = plan.clone();
        verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        let outcome = verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &messages).unwrap();
        assert_eq!(outcome.first_seal, messages.first_seal);

        let bytes = messages.encode().unwrap();
        let decoded = HiddenLinMessages::decode(&plan, &bytes).unwrap();
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &decoded).is_ok());
    }

    #[test]
    fn norm_tail_slack_lookup_claim_and_message_tampering_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = slack_lookup_fixture(&activation);
        let (messages, hidden_witness) = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &witness,
            &mut params,
        )
        .unwrap();
        let mut verifier_plan = plan.clone();
        verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        let base = HiddenLinMessages::decode(&plan, &messages.encode().unwrap()).unwrap();
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &base).is_ok());

        let mut changed = base.clone();
        let claim = &mut changed
            .second
            .norm
            .as_mut()
            .unwrap()
            .lookup
            .as_mut()
            .unwrap()
            .claims
            .key;
        claim.value = claim.value.add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        let multiplicity = &mut changed
            .second
            .norm
            .as_mut()
            .unwrap()
            .lookup
            .as_mut()
            .unwrap()
            .claims
            .multiplicity;
        multiplicity.1 = multiplicity.1.add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let (planes, key_bits, table_log) = {
            let lookup_plan = plan.norm.as_ref().unwrap().lookup.as_ref().unwrap();
            (
                lookup_plan.lookup.planes().unwrap(),
                lookup_plan.lookup.key_bits as usize,
                lookup_plan.lookup.table_log as usize,
            )
        };
        let leaf_index = (planes - 1) + key_bits + table_log;
        let mut changed = base.clone();
        let leaf = &mut changed
            .second
            .norm
            .as_mut()
            .unwrap()
            .lookup
            .as_mut()
            .unwrap()
            .claims
            .rows[leaf_index];
        leaf.row_target = leaf.row_target.add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        changed
            .first
            .norm
            .as_mut()
            .unwrap()
            .lookup
            .as_mut()
            .unwrap()
            .bits
            .message
            .group
            .entries[0]
            .root = Digest([0x97; 32]);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        let answer = &mut changed
            .first
            .norm
            .as_mut()
            .unwrap()
            .lookup
            .as_mut()
            .unwrap()
            .bits
            .sample
            .answers[0];
        *answer = answer.add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        changed
            .first
            .norm
            .as_mut()
            .unwrap()
            .lookup
            .as_mut()
            .unwrap()
            .mask
            .root = Digest([0x96; 32]);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        changed.second.norm.as_mut().unwrap().lookup = None;
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());
    }

    #[test]
    fn norm_tail_slack_lookup_plan_gates() {
        use crate::proof::hidden_norm_psc::NormLookupPlan;

        let activation = activation();
        let (plan, _, _) = slack_lookup_fixture(&activation);
        let tail = plan
            .norm
            .as_ref()
            .unwrap()
            .output
            .as_ref()
            .unwrap()
            .tail
            .as_ref()
            .unwrap()
            .clone();

        let round = plan
            .norm
            .as_ref()
            .unwrap()
            .round
            .as_ref()
            .unwrap()
            .clone();
        let mut params = slack_lookup_params();
        params.table_identity = Digest([0x11; 32]);
        assert!(
            NormLookupPlan::from_round_and_tail(&round, &tail, COLUMNS, POSITIONS, &params)
                .is_err()
        );

        let mut params = slack_lookup_params();
        params.mask_oracle = params.bits_oracle;
        assert!(
            NormLookupPlan::from_round_and_tail(&round, &tail, COLUMNS, POSITIONS, &params)
                .is_err()
        );

        let mut params = slack_lookup_params();
        params.group = 0;
        assert!(
            NormLookupPlan::from_round_and_tail(&round, &tail, COLUMNS, POSITIONS, &params)
                .is_err()
        );

        let mut without_clamp = tail.clone();
        without_clamp.clamp = None;
        assert!(NormLookupPlan::from_round_and_tail(
            &round,
            &without_clamp,
            COLUMNS,
            POSITIONS,
            &slack_lookup_params()
        )
        .is_err());

        let mut changed = plan.clone();
        let out_base = changed
            .norm
            .as_ref()
            .unwrap()
            .output
            .as_ref()
            .unwrap()
            .out
            .oracle_base;
        changed
            .norm
            .as_mut()
            .unwrap()
            .lookup
            .as_mut()
            .unwrap()
            .key_groups[0]
            .oracle_base = out_base;
        assert!(changed.validate().is_err());

        // The key map must address every digit cell of both groups in the
        // canonical order; dropping or redirecting one cell is a plan fault.
        let mut changed = plan.clone();
        changed
            .norm
            .as_mut()
            .unwrap()
            .lookup
            .as_mut()
            .unwrap()
            .key
            .sources
            .pop();
        assert!(changed.validate().is_err());

        let mut changed = plan.clone();
        changed
            .norm
            .as_mut()
            .unwrap()
            .lookup
            .as_mut()
            .unwrap()
            .key
            .sources[0]
            .place ^= 1;
        assert!(changed.validate().is_err());

        let mut changed = plan.clone();
        let sources = &mut changed
            .norm
            .as_mut()
            .unwrap()
            .lookup
            .as_mut()
            .unwrap()
            .key
            .sources;
        let last = sources.len() - 1;
        sources[last].row ^= 1;
        assert!(changed.validate().is_err());
    }

    #[test]
    fn norm_tail_slack_lookup_committed_cell_tampering_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = slack_lookup_fixture(&activation);
        let lookup_plan = plan.norm.as_ref().unwrap().lookup.as_ref().unwrap();
        let source = lookup_plan.key.sources[0];
        let tail_plan = plan
            .norm
            .as_ref()
            .unwrap()
            .output
            .as_ref()
            .unwrap()
            .tail
            .as_ref()
            .unwrap()
            .clone();
        let mut changed = witness.clone();
        let messages = changed
            .output
            .as_ref()
            .unwrap()
            .tail
            .as_ref()
            .unwrap()
            .committed
            .messages
            .clone();
        let mut coins = ByteMaskCoins::new(Digest([0x93; 32])).unwrap();
        let rows = usize::try_from(tail_plan.group.shape.main_rows).unwrap();
        let ell = usize::try_from(tail_plan.group.shape.code.ell).unwrap();
        let columns = usize::try_from(tail_plan.group.columns).unwrap();
        let slots_of = |message: &[Vec<Fp3>]| {
            let mut slots = vec![Fp3::ZERO; columns];
            for (slot, entry) in slots.iter_mut().enumerate() {
                let row = slot % rows;
                let place = slot / rows;
                if place >= ell {
                    break;
                }
                *entry = message[row][place];
            }
            slots
        };
        let mut values = messages.iter().map(|message| slots_of(message)).collect::<Vec<_>>();
        let mut tampered_message = messages[source.position as usize].clone();
        tampered_message[source.row as usize][source.place as usize] =
            tampered_message[source.row as usize][source.place as usize].add(&Fp3::from_u64(200));
        values[source.position as usize] = slots_of(&tampered_message);
        let tampered = commit_norm_group(&tail_plan.group, &values, &mut coins);
        changed
            .output
            .as_mut()
            .unwrap()
            .tail
            .as_mut()
            .unwrap()
            .committed
            .committed = tampered.committed;
        let proved = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &changed,
            &mut params,
        );
        match proved {
            Err(_) => {}
            Ok((messages, hidden_witness)) => {
                let mut verifier_plan = plan.clone();
                verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
                assert!(
                    verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &messages).is_err(),
                    "a committed slack cell that differs from the proven key must not verify"
                );
            }
        }
    }

    #[test]
    fn norm_tail_slack_lookup_committed_digit_tampering_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = slack_lookup_fixture(&activation);
        let lookup_plan = plan.norm.as_ref().unwrap().lookup.as_ref().unwrap();
        let source = lookup_plan.key.sources[0];
        let mut changed = witness.clone();
        {
            let tail = changed
                .output
                .as_mut()
                .unwrap()
                .tail
                .as_mut()
                .unwrap();
            let value = &mut tail.committed.messages[source.position as usize]
                [source.row as usize][source.place as usize];
            *value = value.add(&Fp3::ONE);
        }
        let proved = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &changed,
            &mut params,
        );
        assert!(
            proved.is_err(),
            "a committed slack digit that differs from the exact tail row must not prove"
        );
    }

    #[test]
    fn norm_tail_slack_lookup_out_of_range_digit_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = slack_lookup_fixture(&activation);
        let mut changed = witness.clone();
        {
            let high = plan
                .norm
                .as_ref()
                .unwrap()
                .output
                .as_ref()
                .unwrap()
                .clamp
                .expect("the fixture clamps")
                .1;
            let row = &mut changed.output.as_mut().unwrap().rows[0];
            row.pre_clamp[0] = high as i128 + (1i128 << 32) + 5;
        }
        let proved = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &changed,
            &mut params,
        );
        assert!(
            proved.is_err(),
            "a slack value that does not fit its byte width must not prove"
        );
    }

    fn prove_and_verify_with_slack(
        activation: &CommittedActivation,
        plan: &HiddenLinPlan,
        witness: &crate::proof::hidden_norm_psc::NormWitness,
        params: &mut ParamTables,
    ) -> Result<()> {
        let mut table = table();
        let (messages, hidden_witness) = prove_hidden_lin_hidden_with_norm_and_params(
            plan,
            activation,
            &mut table,
            Digest([0x71; 32]),
            witness,
            params,
        )?;
        let mut verifier_plan = plan.clone();
        verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &messages).map(|_| ())
    }

    #[test]
    fn norm_tail_clamp_boundaries_prove_and_verify() {
        let activation = activation();
        let (_, base, _) = norm_output_fixture(&activation);
        let pre = base
            .output
            .as_ref()
            .unwrap()
            .rows[0]
            .pre_clamp[0];
        assert!(pre.abs() < i64::MAX as i128);
        let pre = pre as i64;
        let cases = [
            (pre + 1, pre + 2),
            (pre, pre + 2),
            (pre - 1, pre + 2),
            (pre - 2, pre + 1),
            (pre - 2, pre),
            (pre - 2, pre - 1),
            (pre, pre),
        ];
        for (low, high) in cases {
            let (plan, witness, mut params) = clamp_boundary_fixture(&activation, low, high);
            prove_and_verify_with_slack(&activation, &plan, &witness, &mut params).unwrap_or_else(
                |error| panic!("the clamp boundary {low}..{high} must prove and verify: {error:?}"),
            );
        }
    }

    #[test]
    fn norm_tail_clamp_selector_flip_is_rejected() {
        let activation = activation();
        let (plan, witness, mut params) = clamp_boundary_fixture(&activation, -2, 2);
        let tail_plan = plan
            .norm
            .as_ref()
            .unwrap()
            .output
            .as_ref()
            .unwrap()
            .tail
            .as_ref()
            .unwrap()
            .clone();
        let paths = &tail_plan.widths;
        let clamp = tail_plan.clamp.as_ref().unwrap();
        let pre_at = paths.quotient
            + paths.remainder
            + paths.doubled_remainder
            + paths.tie_slack
            + 1
            + paths.shifted
            + paths.pre_clamp;
        let low_at = pre_at + paths.pre_clamp;
        let low_selector = low_at + clamp.low + clamp.high;
        let columns = usize::try_from(tail_plan.group.columns).unwrap();
        let rows = usize::try_from(tail_plan.group.shape.main_rows).unwrap();
        let ell = usize::try_from(tail_plan.group.shape.code.ell).unwrap();
        let messages = witness
            .output
            .as_ref()
            .unwrap()
            .tail
            .as_ref()
            .unwrap()
            .committed
            .messages
            .clone();
        let mut values = Vec::with_capacity(messages.len());
        for message in &messages {
            let mut slots = vec![Fp3::ZERO; columns];
            for (slot, value) in slots.iter_mut().enumerate() {
                if slot / rows < ell {
                    *value = message[slot % rows][slot / rows];
                }
            }
            values.push(slots);
        }
        // Flip the low selector of column zero while the distances stay.
        values[0][low_selector] = Fp3::ONE.sub(&values[0][low_selector]);
        let mut coins = ByteMaskCoins::new(Digest([0x97; 32])).unwrap();
        let tail = commit_norm_group(&tail_plan.group, &values, &mut coins);
        let mut changed = witness.clone();
        changed.output.as_mut().unwrap().tail.as_mut().unwrap().committed = tail;
        let proved = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table(),
            Digest([0x71; 32]),
            &changed,
            &mut params,
        );
        assert!(
            proved.is_err(),
            "a flipped clamp selector must not prove"
        );
    }

    #[test]
    fn norm_tail_clamp_out_tampering_is_rejected() {
        let activation = activation();
        let (mut plan, mut witness, mut params) = slack_lookup_fixture(&activation);
        let output_plan = plan
            .norm
            .as_ref()
            .unwrap()
            .output
            .as_ref()
            .unwrap()
            .clone();
        {
            let row = &mut witness.output.as_mut().unwrap().rows[0];
            row.out[0] += 1;
        }
        let signed = |values: Vec<i128>| -> Vec<Fp3> {
            values
                .into_iter()
                .map(|value| crate::proof::output_closure::fp3_signed(value).unwrap())
                .collect()
        };
        let rows = &witness.output.as_ref().unwrap().rows;
        let out_values = rows
            .iter()
            .map(|row| signed(row.out.iter().map(|value| *value as i128).collect()))
            .collect::<Vec<_>>();
        let mut coins = ByteMaskCoins::new(Digest([0x95; 32])).unwrap();
        witness.output.as_mut().unwrap().out =
            commit_norm_group(&output_plan.out, &out_values, &mut coins);
        attach_slack_lookup(&mut plan, &mut witness);
        let proved = prove_and_verify_with_slack(&activation, &plan, &witness, &mut params);
        assert!(
            proved.is_err(),
            "a changed clamped output must not prove and verify"
        );
    }

    #[test]
    fn norm_tail_clamp_distance_slot_tampering_is_rejected() {
        let activation = activation();
        let (plan, witness, mut params) = clamp_boundary_fixture(&activation, -2, 2);
        let tail_plan = plan
            .norm
            .as_ref()
            .unwrap()
            .output
            .as_ref()
            .unwrap()
            .tail
            .as_ref()
            .unwrap()
            .clone();
        let widths = tail_plan.widths;
        let pre_at = widths.quotient
            + widths.remainder
            + widths.doubled_remainder
            + widths.tie_slack
            + 1
            + widths.shifted
            + widths.pre_clamp;
        let low_at = pre_at + widths.pre_clamp;
        let columns = usize::try_from(tail_plan.group.columns).unwrap();
        let rows = usize::try_from(tail_plan.group.shape.main_rows).unwrap();
        let ell = usize::try_from(tail_plan.group.shape.code.ell).unwrap();
        let messages = witness
            .output
            .as_ref()
            .unwrap()
            .tail
            .as_ref()
            .unwrap()
            .committed
            .messages
            .clone();
        let mut values = Vec::with_capacity(messages.len());
        for message in &messages {
            let mut slots = vec![Fp3::ZERO; columns];
            for (slot, value) in slots.iter_mut().enumerate() {
                if slot / rows < ell {
                    *value = message[slot % rows][slot / rows];
                }
            }
            values.push(slots);
        }
        // Change one byte of the low distance of column zero.
        values[0][low_at] = values[0][low_at].add(&Fp3::ONE);
        let mut coins = ByteMaskCoins::new(Digest([0x96; 32])).unwrap();
        let tail = commit_norm_group(&tail_plan.group, &values, &mut coins);
        let mut changed = witness.clone();
        changed.output.as_mut().unwrap().tail.as_mut().unwrap().committed = tail;
        let proved = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table(),
            Digest([0x71; 32]),
            &changed,
            &mut params,
        );
        assert!(
            proved.is_err(),
            "a changed clamp distance slot must not prove"
        );
    }

    #[test]
    fn norm_tail_clamp_distance_width_gate_is_enforced() {
        let activation = activation();
        let (plan, _, _) = clamp_boundary_fixture(&activation, -2, 2);
        let mut changed = plan.clone();
        let tail = changed
            .norm
            .as_mut()
            .unwrap()
            .output
            .as_mut()
            .unwrap()
            .tail
            .as_mut()
            .unwrap();
        tail.clamp.as_mut().unwrap().low = 1;
        assert!(
            changed.validate().is_err(),
            "a clamp distance width that cannot hold the span must be rejected"
        );
    }

    fn recommit_tail_from_rows(
        plan: &HiddenLinPlan,
        witness: &mut NormWitness,
    ) -> Result<()> {
        let tail_plan = plan
            .norm
            .as_ref()
            .ok_or(Error::InvalidConfig)?
            .output
            .as_ref()
            .ok_or(Error::InvalidConfig)?
            .tail
            .as_ref()
            .ok_or(Error::InvalidConfig)?
            .clone();
        let rows = &witness
            .output
            .as_ref()
            .ok_or(Error::ShapeMismatch)?
            .rows;
        let values = rows
            .iter()
            .map(|row| tail_plan.values(row, COLUMNS))
            .collect::<Result<Vec<_>>>()?;
        let mut coins = ByteMaskCoins::new(Digest([0x98; 32]))?;
        let tail = commit_norm_group(&tail_plan.group, &values, &mut coins);
        witness
            .output
            .as_mut()
            .ok_or(Error::ShapeMismatch)?
            .tail
            .as_mut()
            .ok_or(Error::ShapeMismatch)?
            .committed = tail;
        Ok(())
    }

    #[test]
    fn norm_tail_padding_partition_tampering_is_rejected() {
        let activation = activation();
        let (mut plan, witness, mut params) = slack_lookup_fixture(&activation);
        {
            let norm = plan.norm.as_mut().unwrap();
            assert!(u64::from(norm.shape.valid) < norm.columns);
            norm.shape.valid -= 1;
        }
        let proved = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table(),
            Digest([0x71; 32]),
            &witness,
            &mut params,
        );
        assert!(
            proved.is_err(),
            "a valid/padding partition that leaves a padded column inside the valid set must not prove"
        );
    }

    #[test]
    fn norm_tail_padding_out_tampering_is_rejected() {
        let activation = activation();
        let (mut plan, mut witness, mut params) = slack_lookup_fixture(&activation);
        let output_plan = plan
            .norm
            .as_ref()
            .unwrap()
            .output
            .as_ref()
            .unwrap()
            .clone();
        let valid = plan.norm.as_ref().unwrap().shape.valid as usize;
        {
            let row = &mut witness.output.as_mut().unwrap().rows[0];
            row.out[valid] += 1;
        }
        let signed = |values: Vec<i128>| -> Vec<Fp3> {
            values
                .into_iter()
                .map(|value| crate::proof::output_closure::fp3_signed(value).unwrap())
                .collect()
        };
        let rows = &witness.output.as_ref().unwrap().rows;
        let out_values = rows
            .iter()
            .map(|row| signed(row.out.iter().map(|value| *value as i128).collect()))
            .collect::<Vec<_>>();
        let mut coins = ByteMaskCoins::new(Digest([0x99; 32])).unwrap();
        witness.output.as_mut().unwrap().out =
            commit_norm_group(&output_plan.out, &out_values, &mut coins);
        attach_slack_lookup(&mut plan, &mut witness);
        let proved = prove_and_verify_with_slack(&activation, &plan, &witness, &mut params);
        assert!(
            proved.is_err(),
            "a padded column whose output differs from clamp(bias) must not prove and verify"
        );
    }

    #[test]
    fn norm_tail_padding_pre_tampering_is_rejected() {
        let activation = activation();
        let (mut plan, mut witness, mut params) = slack_lookup_fixture(&activation);
        let valid = plan.norm.as_ref().unwrap().shape.valid as usize;
        {
            let row = &mut witness.output.as_mut().unwrap().rows[0];
            row.pre_clamp[valid] += 1;
        }
        recommit_tail_from_rows(&plan, &mut witness).unwrap();
        attach_slack_lookup(&mut plan, &mut witness);
        let proved = prove_and_verify_with_slack(&activation, &plan, &witness, &mut params);
        assert!(
            proved.is_err(),
            "a padded column whose pre-clamp value differs from the committed bias must not prove and verify"
        );
    }

    #[test]
    fn norm_tail_padding_claim_swap_is_rejected() {
        let activation = activation();
        let (mut plan, _, _) = slack_lookup_fixture(&activation);
        let gain = plan
            .norm
            .as_ref()
            .unwrap()
            .params
            .as_ref()
            .unwrap()
            .gain_claim
            .clone()
            .unwrap();
        plan.norm
            .as_mut()
            .unwrap()
            .params
            .as_mut()
            .unwrap()
            .bias_claim = Some(gain);
        assert!(
            plan.validate().is_err(),
            "a bias claim that points at the gain table row must be rejected"
        );
    }

    #[test]
    fn norm_output_hd_is_the_committed_out_object() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = slack_lookup_fixture(&activation);
        let output_plan = plan.norm.as_ref().unwrap().output.as_ref().unwrap();
        assert_eq!(output_plan.hd_group(), &output_plan.out);
        let (messages, hidden_witness) = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &witness,
            &mut params,
        )
        .unwrap();
        let mut verifier_plan = plan.clone();
        verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &messages).is_ok());
        let output = messages.first.norm.as_ref().unwrap().output.as_ref().unwrap();
        let committed = &witness.output.as_ref().unwrap().out;
        for entry in &output.out.message.group.entries {
            assert_eq!(
                entry.oracle,
                output_plan.out.oracle_base + entry.position as u64
            );
            assert_eq!(entry.shape, output_plan.out.shape.oracle_shape());
        }
        assert_eq!(output.out.message.roots, committed.committed.roots);
        // The consumer reads the same committed object: the out value at a
        // column is the committed cell, not a second root.
        let rows = usize::try_from(output_plan.out.shape.main_rows).unwrap();
        let ell = usize::try_from(output_plan.out.shape.code.ell).unwrap();
        let op_rows = &witness.output.as_ref().unwrap().rows;
        for position in 0..POSITIONS as usize {
            for column in 0..COLUMNS as usize {
                let row = column % rows;
                let place = column / rows;
                if place >= ell {
                    break;
                }
                assert_eq!(
                    committed.messages[position][row][place],
                    crate::proof::output_closure::fp3_signed(op_rows[position].out[column] as i128)
                        .unwrap()
                );
            }
        }
    }

    #[test]
    fn norm_output_hd_cross_position_substitution_is_rejected() {
        let activation = activation();
        let (plan, witness, mut params) = slack_lookup_fixture(&activation);
        let output_plan = plan.norm.as_ref().unwrap().output.as_ref().unwrap().clone();
        let columns = usize::try_from(output_plan.out.columns).unwrap();
        let rows = usize::try_from(output_plan.out.shape.main_rows).unwrap();
        let ell = usize::try_from(output_plan.out.shape.code.ell).unwrap();
        let messages = witness
            .output
            .as_ref()
            .unwrap()
            .out
            .committed
            .messages
            .clone();
        let mut values = Vec::with_capacity(messages.len());
        for (position, message) in messages.iter().enumerate() {
            let source = if position == 1 { &messages[0] } else { message };
            let mut slots = vec![Fp3::ZERO; columns];
            for (slot, value) in slots.iter_mut().enumerate() {
                if slot / rows < ell {
                    *value = source[slot % rows][slot / rows];
                }
            }
            values.push(slots);
        }
        let mut coins = ByteMaskCoins::new(Digest([0x9a; 32])).unwrap();
        let substituted = commit_norm_group(&output_plan.out, &values, &mut coins);
        let mut changed = witness.clone();
        changed.output.as_mut().unwrap().out.committed = substituted.committed;
        let mut table = table();
        let proved = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &changed,
            &mut params,
        );
        match proved {
            Err(_) => {}
            Ok((messages, hidden_witness)) => {
                let mut verifier_plan = plan.clone();
                verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
                assert!(
                    verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &messages).is_err(),
                    "another position's output object must not verify in this position"
                );
            }
        }
    }

    #[test]
    fn norm_output_hd_root_tampering_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = slack_lookup_fixture(&activation);
        let (messages, hidden_witness) = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &witness,
            &mut params,
        )
        .unwrap();
        let mut verifier_plan = plan.clone();
        verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        let base = HiddenLinMessages::decode(&plan, &messages.encode().unwrap()).unwrap();
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &base).is_ok());
        let mut changed = base.clone();
        changed
            .first
            .norm
            .as_mut()
            .unwrap()
            .output
            .as_mut()
            .unwrap()
            .out
            .message
            .group
            .entries[0]
            .root = Digest([0x9b; 32]);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());
        let mut changed = base.clone();
        let answer = &mut changed
            .first
            .norm
            .as_mut()
            .unwrap()
            .output
            .as_mut()
            .unwrap()
            .out
            .sample
            .answers[0];
        *answer = answer.add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());
    }

    #[test]
    fn norm_output_hd_shift_mismatch_is_rejected() {
        let activation = activation();
        let (mut plan, witness, mut params) = slack_lookup_fixture(&activation);
        plan.norm
            .as_mut()
            .unwrap()
            .output
            .as_mut()
            .unwrap()
            .shift += 1;
        let proved = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table(),
            Digest([0x71; 32]),
            &witness,
            &mut params,
        );
        assert!(
            proved.is_err(),
            "an output shift that differs from the committed tail rows must not prove"
        );
    }

    fn norm_hd_fixture(
        activation: &CommittedActivation,
    ) -> (HiddenLinPlan, NormWitness, ParamTables) {
        let (mut plan, witness, params) = slack_lookup_fixture(activation);
        let norm = plan.norm.as_ref().unwrap();
        assert_eq!(u64::from(norm.shape.valid), plan.send.matrix_rows);
        assert!(norm.columns >= plan.send.matrix_rows);
        plan.send.activation_source = ActivationSource::NormHd;
        plan.send.activation =
            ActivationShape::new(POSITIONS, plan.send.matrix_rows as u32, SALT).unwrap();
        plan.send.activation_root = HD_ACTIVATION_ROOT;
        plan.send.activation_configuration = HD_ACTIVATION_CONFIGURATION;
        plan.validate().unwrap();
        (plan, witness, params)
    }

    #[test]
    fn norm_hd_feeds_the_hidden_dot() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = norm_hd_fixture(&activation);
        let (messages, hidden_witness) = prove_hidden_lin_hidden_with_norm_hd(
            &plan,
            &mut table,
            Digest([0x71; 32]),
            &witness,
            Some(&mut params),
        )
        .unwrap();
        // The Dot has no shown-map commitment of its own; the activation
        // statement is derived from the committed norm output object.
        assert!(messages.first.dot.is_none());
        let norm_first = messages.first.norm.as_ref().unwrap();
        let entries = norm_first
            .output
            .as_ref()
            .unwrap()
            .out
            .message
            .group
            .entries
            .clone();
        let identity = hd_activation_identity(&plan, &entries).unwrap();
        assert_eq!(messages.first.activation.root, identity);
        assert_eq!(messages.first.activation.shape, plan.send.activation);
        let mut verifier_plan = plan.clone();
        verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        let outcome = verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &messages).unwrap();
        assert_eq!(outcome.first_seal, messages.first_seal);
        assert_eq!(outcome.activation.root, identity);
        assert!(outcome.dot.is_some());

        let bytes = messages.encode().unwrap();
        let decoded = HiddenLinMessages::decode(&plan, &bytes).unwrap();
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &decoded).is_ok());
    }

    #[test]
    fn norm_hd_consumer_object_swap_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = norm_hd_fixture(&activation);
        let (messages, hidden_witness) = prove_hidden_lin_hidden_with_norm_hd(
            &plan,
            &mut table,
            Digest([0x71; 32]),
            &witness,
            Some(&mut params),
        )
        .unwrap();
        let mut verifier_plan = plan.clone();
        verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        let mut changed = HiddenLinMessages::decode(&plan, &messages.encode().unwrap()).unwrap();
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_ok());
        // Another committed object in the same position: the derived identity
        // no longer matches the statement, so the consumer reads a different
        // object than the producer proved.
        changed
            .first
            .norm
            .as_mut()
            .unwrap()
            .output
            .as_mut()
            .unwrap()
            .out
            .message
            .group
            .entries[0]
            .root = Digest([0x9c; 32]);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = HiddenLinMessages::decode(&plan, &messages.encode().unwrap()).unwrap();
        changed.first.norm.as_mut().unwrap().output = None;
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());
    }

    #[test]
    fn norm_hd_old_or_foreign_plan_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = norm_hd_fixture(&activation);
        let (messages, hidden_witness) = prove_hidden_lin_hidden_with_norm_hd(
            &plan,
            &mut table,
            Digest([0x71; 32]),
            &witness,
            Some(&mut params),
        )
        .unwrap();
        let mut verifier_plan = plan.clone();
        verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &messages).is_ok());

        // The independent-activation statement cannot accept an object-derived
        // activation root, and an old proof cannot satisfy the norm mode.
        let mut old_plan = plan.clone();
        old_plan.send.activation_source = ActivationSource::Commitment;
        old_plan.send.activation_root = Digest([0x71; 32]);
        assert!(old_plan.validate().is_ok());
        assert!(verify_hidden_lin_hidden(&old_plan, &table.reader(), &messages).is_err());

        let mut foreign = plan.clone();
        foreign
            .norm
            .as_mut()
            .unwrap()
            .output
            .as_mut()
            .unwrap()
            .out
            .oracle_base += 1;
        let mut foreign_verifier = foreign.clone();
        foreign_verifier.send.expected_roots = Some(hidden_witness.send.roots.clone());
        assert!(
            verify_hidden_lin_hidden(&foreign_verifier, &table.reader(), &messages).is_err(),
            "a foreign hidden-output object must not verify"
        );
    }

    #[test]
    fn norm_hd_missing_dot_plan_is_rejected() {
        let activation = activation();
        let (mut plan, _, _) = norm_hd_fixture(&activation);
        plan.send.dot = None;
        assert!(plan.validate().is_err());

        let (mut plan, _, _) = norm_hd_fixture(&activation);
        plan.send.matrix_rows += 1;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn norm_hd_committed_row_tampering_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = norm_hd_fixture(&activation);
        let mut changed = witness.clone();
        {
            let output = changed.output.as_mut().unwrap();
            let out = output.out.messages.clone();
            let group = &plan.norm.as_ref().unwrap().output.as_ref().unwrap().out;
            let rows = usize::try_from(group.shape.main_rows).unwrap();
            let ell = usize::try_from(group.shape.code.ell).unwrap();
            let columns = usize::try_from(group.columns).unwrap();
            let mut values = Vec::with_capacity(out.len());
            for message in &out {
                let mut slots = vec![Fp3::ZERO; columns];
                for (slot, value) in slots.iter_mut().enumerate() {
                    if slot / rows < ell {
                        *value = message[slot % rows][slot / rows];
                    }
                }
                values.push(slots);
            }
            values[0][0] = values[0][0].add(&Fp3::ONE);
            let mut coins = ByteMaskCoins::new(Digest([0x9d; 32])).unwrap();
            let tampered = commit_norm_group(&group.clone(), &values, &mut coins);
            output.out.committed = tampered.committed;
        }
        let proved = prove_hidden_lin_hidden_with_norm_hd(
            &plan,
            &mut table,
            Digest([0x71; 32]),
            &changed,
            Some(&mut params),
        );
        assert!(
            proved.is_err(),
            "a committed hidden-output row that differs from the norm relation must not prove"
        );
    }

    fn signed_value(value: &Fp3) -> Result<i128> {
        use zkie_engine::pcs::request::field::MODULUS;
        if value.c[1].as_canonical_u64() != 0 || value.c[2].as_canonical_u64() != 0 {
            return Err(Error::BoundViolation);
        }
        let canonical = value.c[0].as_canonical_u64();
        if canonical <= i64::MAX as u64 {
            Ok(canonical as i128)
        } else {
            let magnitude = MODULUS.checked_sub(canonical).ok_or(Error::BoundViolation)?;
            Ok(-(magnitude as i128))
        }
    }

    fn norm_hd_closure_plan(
        plan: &HiddenLinPlan,
        winner: Vec<u32>,
    ) -> crate::proof::output_closure::ClosurePlan {
        use crate::proof::psc::psc_variables;
        use zkie_engine::pcs::request::runtime_tree::{RuntimeTreePart, RuntimeTreePlan};

        let range_bits = 16u32;
        let slack_bits = 18u32;
        let width = (range_bits as u64 + 1) * 2 + slack_bits as u64;
        let bits_columns = (plan.send.columns * width).next_power_of_two();
        let boolean_variables = bits_columns.trailing_zeros() as usize;
        let variables = psc_variables(boolean_variables).unwrap();
        let mask_code = plan.proximity.mask_code;
        let mask_rows = 16u64;
        let wshape = |columns: u64, tree: u64| {
            WShape::new(
                columns,
                plan.send.w.code,
                1,
                POSITIONS,
                SALT,
                tree,
                1,
                W_RUN,
                W_CONFIGURATION,
            )
            .unwrap()
        };
        crate::proof::output_closure::ClosurePlan {
            range_bits,
            slack_bits,
            valid: 2,
            vocab: plan.send.columns as usize,
            mask_value: -65_536,
            winner,
            slack: wshape(plan.send.columns, 63_000),
            slack_oracle_base: 42_000,
            slack_openings: 1,
            bits: wshape(bits_columns, 63_010),
            bits_oracle_base: 42_010,
            bits_openings: 1,
            map: wshape(bits_columns, 63_020),
            map_oracle_base: 42_020,
            map_openings: 1,
            bits_columns,
            bits_per_cell: width as u32,
            boolean_variables: boolean_variables as u32,
            mask: crate::proof::output_closure::ClosureMaskPlan {
                oracle: 42_030,
                group: plan.send.group,
                mask_code,
                mask_rows,
                answers: 1,
                openings: 1,
                tree: RuntimeTreePlan {
                    tree_number: 63_030,
                    leaf_class: 1,
                    log_n: mask_code.log_n,
                    log_n_zk: mask_code.log_n,
                    parts: vec![RuntimeTreePart {
                        main_symbols: 0,
                        mask_symbols: mask_rows * E_DEGREE as u64,
                    }],
                    salt_bytes: SALT,
                    run: W_RUN,
                    configuration: W_CONFIGURATION,
                },
                positions: POSITIONS,
                hiding: hp_entries(&variables).unwrap(),
                shown: 5,
            },
        }
    }

    fn norm_hd_logits(
        plan: &HiddenLinPlan,
        witness: &NormWitness,
        table: &mut Table,
    ) -> Vec<Vec<i128>> {
        let weights = hd_activation_rows(plan, Some(witness)).unwrap();
        let reader = table.reader().clone();
        let rows = derive_w_rows_from_weights(&plan.send, &weights, &reader, table).unwrap();
        rows.iter()
            .map(|row| {
                row.iter()
                    .map(|value| signed_value(value).unwrap())
                    .collect()
            })
            .collect()
    }

    fn first_maximum(rows: &[Vec<i128>], valid: usize, mask: i128) -> Vec<u32> {
        rows.iter()
            .map(|row| {
                let mut best = 0usize;
                let mut best_value = i128::MIN;
                for (column, value) in row.iter().enumerate() {
                    let masked = if column >= valid { *value + mask } else { *value };
                    if column == 0 || masked > best_value {
                        best = column;
                        best_value = masked;
                    }
                }
                assert!(best < valid);
                best as u32
            })
            .collect()
    }

    #[test]
    fn norm_hd_logits_close_to_argmax() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = norm_hd_fixture(&activation);
        let logits = norm_hd_logits(&plan, &witness, &mut table);
        let winner = first_maximum(&logits, 2, -65_536);
        let mut plan = plan;
        plan.closure = Some(norm_hd_closure_plan(&plan, winner.clone()));
        plan.output_relation = Some(Digest([0x5a; 32]));
        plan.validate().unwrap();
        let (messages, hidden_witness) = prove_hidden_lin_hidden_with_norm_hd(
            &plan,
            &mut table,
            Digest([0x71; 32]),
            &witness,
            Some(&mut params),
        )
        .unwrap();
        let mut verifier_plan = plan.clone();
        verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        let outcome = verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &messages).unwrap();
        assert!(outcome.closure.is_some());
        let rows = crate::proof::output_head_lin::witness_output_rows(&hidden_witness).unwrap();
        let closed = first_maximum(
            &rows
                .iter()
                .map(|row| row.iter().map(|value| *value as i128).collect())
                .collect::<Vec<_>>(),
            2,
            -65_536,
        );
        assert_eq!(closed, winner);
        assert!(messages.first.dot.is_none());
        assert!(messages.first.norm.as_ref().unwrap().output.is_some());
    }

    #[test]
    fn norm_hd_closure_wrong_winner_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = norm_hd_fixture(&activation);
        let logits = norm_hd_logits(&plan, &witness, &mut table);
        let winner = first_maximum(&logits, 2, -65_536);
        let wrong = winner
            .iter()
            .map(|index| if *index == 0 { 1 } else { 0 })
            .collect::<Vec<_>>();
        let mut plan = plan;
        plan.closure = Some(norm_hd_closure_plan(&plan, wrong));
        plan.output_relation = Some(Digest([0x5a; 32]));
        plan.validate().unwrap();
        let proved = prove_hidden_lin_hidden_with_norm_hd(
            &plan,
            &mut table,
            Digest([0x71; 32]),
            &witness,
            Some(&mut params),
        );
        assert!(
            proved.is_err(),
            "a closure winner that differs from the logits must not prove"
        );
    }

    #[test]
    fn norm_output_claims_are_masked() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = slack_lookup_fixture(&activation);
        let (messages, hidden_witness) = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &witness,
            &mut params,
        )
        .unwrap();
        let mut verifier_plan = plan.clone();
        verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        let outcome = verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &messages).unwrap();
        let bindings = outcome.norm_output.as_ref().unwrap();
        assert_eq!(bindings.len(), POSITIONS as usize * 2);
        let output_witness = witness.output.as_ref().unwrap();
        let params_values = witness.params.as_ref().unwrap();
        let proof_bytes = messages.encode().unwrap();
        for word in bindings {
            let position = word.position as usize;
            let (left, right) = norm_pointwise_maps(
                plan.norm.as_ref().unwrap(),
                &witness.rows[position],
                &witness.core[position],
                &params_values.gain,
                &output_witness.rows[position],
                word.kind,
            )
            .unwrap();
            let eq = crate::proof::output_closure::eq_vector(&word.point).unwrap();
            let mut plain = Fp3::ZERO;
            for index in 0..left.len() {
                plain = plain.add(&eq[index].mul(&left[index]).mul(&right[index]));
            }
            assert_ne!(
                word.claim, plain,
                "the sent product claim must be masked, not the private evaluation"
            );
            let mut writer = CanonicalWriter::new();
            writer.put_fp3(&plain);
            let needle = writer.into_bytes();
            assert!(
                !proof_bytes
                    .windows(needle.len())
                    .any(|window| window == needle.as_slice()),
                "the private product evaluation must not appear in the proof bytes"
            );
        }
    }

    #[test]
    fn norm_output_claim_tampering_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = slack_lookup_fixture(&activation);
        let (messages, hidden_witness) = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &witness,
            &mut params,
        )
        .unwrap();
        let mut verifier_plan = plan.clone();
        verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        let mut changed = HiddenLinMessages::decode(&plan, &messages.encode().unwrap()).unwrap();
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_ok());
        let claim = &mut changed
            .second
            .norm
            .as_mut()
            .unwrap()
            .output
            .as_mut()
            .unwrap()
            .claims[0];
        *claim = claim.add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());
    }

    #[test]
    fn norm_output_claim_mask_tampering_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = slack_lookup_fixture(&activation);
        let (messages, hidden_witness) = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &witness,
            &mut params,
        )
        .unwrap();
        let mut verifier_plan = plan.clone();
        verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        let mut changed = HiddenLinMessages::decode(&plan, &messages.encode().unwrap()).unwrap();
        let mask = changed
            .first
            .norm
            .as_mut()
            .unwrap()
            .mask
            .as_mut()
            .unwrap();
        mask.root = Digest([0x9e; 32]);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = HiddenLinMessages::decode(&plan, &messages.encode().unwrap()).unwrap();
        let answer = &mut changed
            .first
            .norm
            .as_mut()
            .unwrap()
            .mask
            .as_mut()
            .unwrap()
            .sample
            .answers[0];
        *answer = answer.add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());
    }

    #[test]
    fn norm_output_claim_mask_entry_tampering_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = slack_lookup_fixture(&activation);
        let norm_plan = plan.norm.as_ref().unwrap();
        let mask_plan = norm_plan.mask.as_ref().unwrap();
        let instances = norm_plan.shape.positions as usize * norm_plan.relations.len();
        let slot = mask_plan.claim_slot(instances, 0).unwrap();
        let ell = usize::try_from(mask_plan.mask_code.ell).unwrap();
        let mut changed = witness.clone();
        {
            let mask = changed.mask.as_mut().unwrap();
            mask.committed.rows[slot / ell][slot % ell] =
                mask.committed.rows[slot / ell][slot % ell].add(&Fp3::ONE);
        }
        let proved = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &changed,
            &mut params,
        );
        assert!(
            proved.is_err(),
            "a claim mask entry that differs from the committed mask message must not prove"
        );
    }

    fn group_slots(
        shape: &WShape,
        messages: &[Vec<Vec<Fp3>>],
        columns: u64,
    ) -> Vec<Vec<Fp3>> {
        let rows = usize::try_from(shape.main_rows).unwrap();
        let ell = usize::try_from(shape.code.ell).unwrap();
        let columns = usize::try_from(columns).unwrap();
        messages
            .iter()
            .map(|message| {
                let mut slots = vec![Fp3::ZERO; columns];
                for (slot, value) in slots.iter_mut().enumerate() {
                    if slot / rows < ell {
                        *value = message[slot % rows][slot / rows];
                    }
                }
                slots
            })
            .collect()
    }

    #[test]
    fn norm_malicious_tail_digit_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = slack_lookup_fixture(&activation);
        let tail_plan = plan
            .norm
            .as_ref()
            .unwrap()
            .output
            .as_ref()
            .unwrap()
            .tail
            .as_ref()
            .unwrap()
            .clone();
        let mut changed = witness.clone();
        let mut values = group_slots(
            &tail_plan.group.shape,
            &changed
                .output
                .as_ref()
                .unwrap()
                .tail
                .as_ref()
                .unwrap()
                .committed
                .messages,
            tail_plan.group.columns,
        );
        values[0][0] = Fp3::from_u64(300);
        let mut coins = ByteMaskCoins::new(Digest([0xaa; 32])).unwrap();
        let tampered = commit_norm_group(&tail_plan.group, &values, &mut coins);
        changed
            .output
            .as_mut()
            .unwrap()
            .tail
            .as_mut()
            .unwrap()
            .committed = tampered;
        let proved = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &changed,
            &mut params,
        );
        assert!(
            proved.is_err(),
            "a tail digit that is not a byte must not prove"
        );
    }

    #[test]
    fn norm_malicious_round_digit_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = slack_lookup_fixture(&activation);
        let round_plan = plan
            .norm
            .as_ref()
            .unwrap()
            .round
            .as_ref()
            .unwrap()
            .clone();
        let mut changed = witness.clone();
        let mut values = group_slots(
            &round_plan.group.shape,
            &changed
                .round
                .as_ref()
                .unwrap()
                .committed
                .messages,
            round_plan.group.columns,
        );
        values[0][round_plan.layout.slots.mean_digits] = Fp3::from_u64(300);
        let mut coins = ByteMaskCoins::new(Digest([0xab; 32])).unwrap();
        let tampered = commit_norm_group(&round_plan.group, &values, &mut coins);
        changed.round.as_mut().unwrap().committed = tampered.committed;
        let proved = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &changed,
            &mut params,
        );
        assert!(
            proved.is_err(),
            "a rounding digit that is not a byte must not prove"
        );
    }

    #[test]
    fn norm_malicious_distance_slot_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = slack_lookup_fixture(&activation);
        let tail_plan = plan
            .norm
            .as_ref()
            .unwrap()
            .output
            .as_ref()
            .unwrap()
            .tail
            .as_ref()
            .unwrap()
            .clone();
        let groups = crate::proof::hidden_norm_psc::tail_digit_groups(&tail_plan).unwrap();
        let low_at = groups[6].0;
        let mut changed = witness.clone();
        let mut values = group_slots(
            &tail_plan.group.shape,
            &changed
                .output
                .as_ref()
                .unwrap()
                .tail
                .as_ref()
                .unwrap()
                .committed
                .messages,
            tail_plan.group.columns,
        );
        values[0][low_at] = values[0][low_at].add(&Fp3::ONE);
        let mut coins = ByteMaskCoins::new(Digest([0xac; 32])).unwrap();
        let tampered = commit_norm_group(&tail_plan.group, &values, &mut coins);
        changed
            .output
            .as_mut()
            .unwrap()
            .tail
            .as_mut()
            .unwrap()
            .committed = tampered;
        let proved = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &changed,
            &mut params,
        );
        assert!(
            proved.is_err(),
            "a distance slot that differs from pre_clamp minus low must not prove"
        );
    }

    #[test]
    fn norm_malicious_out_shift_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = slack_lookup_fixture(&activation);
        let output_plan = plan
            .norm
            .as_ref()
            .unwrap()
            .output
            .as_ref()
            .unwrap()
            .clone();
        let mut changed = witness.clone();
        let mut values = group_slots(
            &output_plan.out.shape,
            &changed.output.as_ref().unwrap().out.committed.messages,
            output_plan.out.columns,
        );
        values[0][0] = values[0][0].add(&Fp3::ONE);
        let mut coins = ByteMaskCoins::new(Digest([0xad; 32])).unwrap();
        let tampered = commit_norm_group(&output_plan.out, &values, &mut coins);
        changed.output.as_mut().unwrap().out.committed = tampered.committed;
        let proved = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &changed,
            &mut params,
        );
        match proved {
            Err(_) => {}
            Ok((messages, hidden_witness)) => {
                let mut verifier_plan = plan.clone();
                verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
                assert!(
                    verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &messages).is_err(),
                    "a shifted hidden output must not verify"
                );
            }
        }
    }

    fn malicious_out_shift_fixture(
        activation: &CommittedActivation,
    ) -> (HiddenLinPlan, NormWitness, ParamTables) {
        let (plan, witness, params) = clamp_boundary_fixture(activation, -2, 2);
        let output_plan = plan
            .norm
            .as_ref()
            .unwrap()
            .output
            .as_ref()
            .unwrap()
            .clone();
        let mut changed = witness;
        let mut values = group_slots(
            &output_plan.out.shape,
            &changed.output.as_ref().unwrap().out.committed.messages,
            output_plan.out.columns,
        );
        values[0][0] = values[0][0].add(&Fp3::ONE);
        let mut coins = ByteMaskCoins::new(Digest([0xae; 32])).unwrap();
        let tampered = commit_norm_group(&output_plan.out, &values, &mut coins);
        changed.output.as_mut().unwrap().out.committed = tampered.committed;
        (plan, changed, params)
    }

    #[test]
    fn norm_malicious_out_shift_at_the_collapsed_interval_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = malicious_out_shift_fixture(&activation);
        let proved = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &witness,
            &mut params,
        );
        match proved {
            Err(_) => {}
            Ok((messages, hidden_witness)) => {
                let mut verifier_plan = plan.clone();
                verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
                assert!(
                    verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &messages).is_err(),
                    "a shifted hidden output at the collapsed interval must not verify"
                );
            }
        }
    }

    #[test]
    fn norm_malicious_variance_digit_is_rejected() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = slack_lookup_fixture(&activation);
        let round_plan = plan
            .norm
            .as_ref()
            .unwrap()
            .round
            .as_ref()
            .unwrap()
            .clone();
        let mut changed = witness.clone();
        let mut values = group_slots(
            &round_plan.group.shape,
            &changed.round.as_ref().unwrap().committed.messages,
            round_plan.group.columns,
        );
        values[0][round_plan.layout.slots.variance_digits] = Fp3::from_u64(300);
        let mut coins = ByteMaskCoins::new(Digest([0xaf; 32])).unwrap();
        let tampered = commit_norm_group(&round_plan.group, &values, &mut coins);
        changed.round.as_mut().unwrap().committed = tampered.committed;
        let proved = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &changed,
            &mut params,
        );
        assert!(
            proved.is_err(),
            "a variance digit that is not a byte must not prove"
        );
    }

    fn rejects_malicious_witness(
        activation: &CommittedActivation,
        plan: &HiddenLinPlan,
        witness: &NormWitness,
        params: &mut ParamTables,
        context: &str,
    ) {
        let mut table = table();
        let proved = prove_hidden_lin_hidden_with_norm_and_params(
            plan,
            activation,
            &mut table,
            Digest([0x71; 32]),
            witness,
            params,
        );
        match proved {
            Err(_) => {}
            Ok((messages, hidden_witness)) => {
                let mut verifier_plan = plan.clone();
                verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
                assert!(
                    verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &messages).is_err(),
                    "{context}"
                );
            }
        }
    }

    fn tampered_tail_digits_fixture(
        activation: &CommittedActivation,
        at: usize,
        delta: u64,
        coin: u8,
    ) -> (HiddenLinPlan, NormWitness, ParamTables) {
        let (plan, witness, params) = slack_lookup_fixture(activation);
        let tail_plan = plan
            .norm
            .as_ref()
            .unwrap()
            .output
            .as_ref()
            .unwrap()
            .tail
            .as_ref()
            .unwrap()
            .clone();
        let mut changed = witness;
        let mut values = group_slots(
            &tail_plan.group.shape,
            &changed
                .output
                .as_ref()
                .unwrap()
                .tail
                .as_ref()
                .unwrap()
                .committed
                .messages,
            tail_plan.group.columns,
        );
        values[0][at] = values[0][at].add(&Fp3::from_u64(delta));
        let mut coins = ByteMaskCoins::new(Digest([coin; 32])).unwrap();
        let tampered = commit_norm_group(&tail_plan.group, &values, &mut coins);
        changed
            .output
            .as_mut()
            .unwrap()
            .tail
            .as_mut()
            .unwrap()
            .committed = tampered;
        (plan, changed, params)
    }

    #[test]
    fn norm_malicious_tail_digit_classes_are_rejected() {
        let activation = activation();
        let (base_plan, _, _) = slack_lookup_fixture(&activation);
        let tail_plan = base_plan
            .norm
            .as_ref()
            .unwrap()
            .output
            .as_ref()
            .unwrap()
            .tail
            .as_ref()
            .unwrap()
            .clone();
        let groups = crate::proof::hidden_norm_psc::tail_digit_groups(&tail_plan).unwrap();
        let cases = [
            ("a quotient digit that is not a byte", groups[0].0),
            ("a remainder digit that is not a byte", groups[1].0),
            ("a pre clamp digit that is not a byte", groups[5].0),
        ];
        for (index, (context, at)) in cases.iter().enumerate() {
            let (plan, witness, mut params) =
                tampered_tail_digits_fixture(&activation, *at, 300, 0xb0 + index as u8);
            rejects_malicious_witness(&activation, &plan, &witness, &mut params, context);
        }
    }

    #[test]
    fn norm_malicious_tie_slack_digit_is_rejected() {
        let activation = activation();
        let (base_plan, _, _) = slack_lookup_fixture(&activation);
        let tail_plan = base_plan
            .norm
            .as_ref()
            .unwrap()
            .output
            .as_ref()
            .unwrap()
            .tail
            .as_ref()
            .unwrap()
            .clone();
        let groups = crate::proof::hidden_norm_psc::tail_digit_groups(&tail_plan).unwrap();
        let (plan, witness, mut params) =
            tampered_tail_digits_fixture(&activation, groups[3].0, 1, 0xb8);
        rejects_malicious_witness(
            &activation,
            &plan,
            &witness,
            &mut params,
            "a tie slack digit that breaks the doubled remainder relation",
        );
    }

    fn tampered_out_fixture(
        activation: &CommittedActivation,
        low: i64,
        high: i64,
        delta: i64,
        coin: u8,
    ) -> (HiddenLinPlan, NormWitness, ParamTables) {
        let (plan, witness, params) = clamp_boundary_fixture(activation, low, high);
        let output_plan = plan
            .norm
            .as_ref()
            .unwrap()
            .output
            .as_ref()
            .unwrap()
            .clone();
        let mut changed = witness;
        let mut values = group_slots(
            &output_plan.out.shape,
            &changed.output.as_ref().unwrap().out.committed.messages,
            output_plan.out.columns,
        );
        values[0][0] = if delta >= 0 {
            values[0][0].add(&Fp3::from_u64(delta as u64))
        } else {
            values[0][0].sub(&Fp3::from_u64(delta.unsigned_abs()))
        };
        let mut coins = ByteMaskCoins::new(Digest([coin; 32])).unwrap();
        let tampered = commit_norm_group(&output_plan.out, &values, &mut coins);
        changed.output.as_mut().unwrap().out.committed = tampered.committed;
        (plan, changed, params)
    }

    #[test]
    fn norm_malicious_boundary_and_collapsed_clamp_witnesses_are_rejected() {
        let activation = activation();
        let (_, base, _) = norm_output_fixture(&activation);
        let pre = base.output.as_ref().unwrap().rows[0].pre_clamp[0];
        let pre = i64::try_from(pre).unwrap();
        let cases = [
            ("a collapsed interval admits only the bound", (pre, pre, 1i64)),
            ("a value one below the low bound must clamp", (pre + 1, pre + 2, -1)),
            ("a value one above the high bound must clamp", (pre - 2, pre - 1, 1)),
        ];
        for (index, (context, (low, high, delta))) in cases.iter().enumerate() {
            let (plan, witness, mut params) =
                tampered_out_fixture(&activation, *low, *high, *delta, 0xbc + index as u8);
            rejects_malicious_witness(&activation, &plan, &witness, &mut params, context);
        }
    }

    fn clamp_attack_fixture(
        activation: &CommittedActivation,
        low: i64,
        high: i64,
        column: usize,
        coin: u8,
    ) -> (HiddenLinPlan, NormWitness, ParamTables) {
        use crate::proof::output_closure::fp3_signed;

        let (plan, witness, params) = clamp_boundary_fixture(activation, low, high);
        let output_plan = plan
            .norm
            .as_ref()
            .unwrap()
            .output
            .as_ref()
            .unwrap()
            .clone();
        let tail_plan = output_plan.tail.clone().unwrap();
        let clamp = tail_plan.clamp.clone().unwrap();
        let groups = crate::proof::hidden_norm_psc::tail_digit_groups(&tail_plan).unwrap();
        let high_at = groups[7].0;
        let low_selector = high_at + clamp.high;
        let low_strict_at = groups[8].0;
        let low_nonstrict_at = groups[9].0;
        let high_strict_at = groups[10].0;
        let high_nonstrict_at = groups[11].0;
        let slots = tail_plan.slots().unwrap();
        let mut changed = witness;
        let mut values = group_slots(
            &tail_plan.group.shape,
            &changed
                .output
                .as_ref()
                .unwrap()
                .tail
                .as_ref()
                .unwrap()
                .committed
                .messages,
            tail_plan.group.columns,
        );
        let base = column * slots;
        let pre = changed.output.as_ref().unwrap().rows[0].pre_clamp[column];
        let u = pre - low as i128;
        let v = pre - high as i128;
        let selector = if u < 0 { low_selector } else { low_selector + 1 };
        values[0][base + selector] = Fp3::from_u64(2);
        if u < 0 {
            // Relation 7: alpha = -u - 1 must hold; relation 8: beta = u as a
            // field element, which is not a byte when u is negative.
            for digit in 0..clamp.low_strict {
                values[0][base + low_strict_at + digit] = Fp3::ZERO;
            }
            values[0][base + low_strict_at] = fp3_signed(-u - 1).unwrap();
            for digit in 0..clamp.low_nonstrict {
                values[0][base + low_nonstrict_at + digit] = Fp3::ZERO;
            }
            values[0][base + low_nonstrict_at] = fp3_signed(u).unwrap();
        } else {
            // Relation 9: alpha = v - 1; relation 10: beta = -v as a field
            // element, which is not a byte when v is positive.
            for digit in 0..clamp.high_strict {
                values[0][base + high_strict_at + digit] = Fp3::ZERO;
            }
            values[0][base + high_strict_at] = fp3_signed(v - 1).unwrap();
            for digit in 0..clamp.high_nonstrict {
                values[0][base + high_nonstrict_at + digit] = Fp3::ZERO;
            }
            values[0][base + high_nonstrict_at] = fp3_signed(-v).unwrap();
        }
        let mut coins = ByteMaskCoins::new(Digest([coin; 32])).unwrap();
        let tampered = commit_norm_group(&tail_plan.group, &values, &mut coins);
        changed
            .output
            .as_mut()
            .unwrap()
            .tail
            .as_mut()
            .unwrap()
            .committed = tampered;
        // Relation 5 with the flipped selector: out = pre - selector * distance.
        let out_star = if u < 0 {
            pre - 2 * u
        } else {
            pre - 2 * v
        };
        let mut out_values = group_slots(
            &output_plan.out.shape,
            &changed.output.as_ref().unwrap().out.committed.messages,
            output_plan.out.columns,
        );
        out_values[0][column] = fp3_signed(out_star).unwrap();
        let mut coins = ByteMaskCoins::new(Digest([coin.wrapping_add(1); 32])).unwrap();
        let tampered = commit_norm_group(&output_plan.out, &out_values, &mut coins);
        changed.output.as_mut().unwrap().out.committed = tampered.committed;
        changed.output.as_mut().unwrap().rows[0].out[column] =
            i64::try_from(out_star).unwrap();
        (plan, changed, params)
    }

    #[test]
    fn norm_malicious_clamp_selector_attack_is_rejected() {
        let activation = activation();
        let (_, base, _) = norm_output_fixture(&activation);
        let pre = base.output.as_ref().unwrap().rows[0].pre_clamp[0];
        let pre = i64::try_from(pre).unwrap();
        let cases = [
            ("the low selector attack", (pre + 2, pre + 3)),
            ("the high selector attack", (pre - 3, pre - 2)),
        ];
        for (index, (context, (low, high))) in cases.iter().enumerate() {
            let (plan, witness, mut params) = clamp_attack_fixture(
                &activation,
                *low,
                *high,
                0,
                0xd4 + index as u8,
            );
            let mut table = table();
            let proved = prove_hidden_lin_hidden_with_norm_and_params(
                &plan,
                &activation,
                &mut table,
                Digest([0x71; 32]),
                &witness,
                &mut params,
            );
            match proved {
                Err(_) => {}
                Ok((messages, hidden_witness)) => {
                    let mut verifier_plan = plan.clone();
                    verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
                    assert!(
                        verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &messages)
                            .is_err(),
                        "{context} must not verify"
                    );
                }
            }
        }
    }

    #[test]
    fn norm_malicious_padding_selector_attack_is_rejected() {
        let activation = activation();
        let (plan, _, _) = slack_lookup_fixture(&activation);
        let valid = usize::try_from(plan.norm.as_ref().unwrap().shape.valid).unwrap();
        let low = plan
            .norm
            .as_ref()
            .unwrap()
            .output
            .as_ref()
            .unwrap()
            .clamp
            .unwrap()
            .0;
        let high = plan
            .norm
            .as_ref()
            .unwrap()
            .output
            .as_ref()
            .unwrap()
            .clamp
            .unwrap()
            .1;
        let (plan, witness, mut params) =
            clamp_attack_fixture(&activation, low, high, valid, 0xd8);
        let mut table = table();
        let proved = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &witness,
            &mut params,
        );
        match proved {
            Err(_) => {}
            Ok((messages, hidden_witness)) => {
                let mut verifier_plan = plan.clone();
                verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
                assert!(
                    verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &messages).is_err(),
                    "a flipped clamp selector on a padding column must not verify"
                );
            }
        }
    }

    #[test]
    fn norm_malicious_slack_digit_out_of_byte_is_rejected() {
        let activation = activation();
        let (base_plan, _, _) = slack_lookup_fixture(&activation);
        let tail_plan = base_plan
            .norm
            .as_ref()
            .unwrap()
            .output
            .as_ref()
            .unwrap()
            .tail
            .as_ref()
            .unwrap()
            .clone();
        let groups = crate::proof::hidden_norm_psc::tail_digit_groups(&tail_plan).unwrap();
        for (index, (context, at)) in [
            ("a low strict slack digit that is not a byte", groups[8].0),
            ("a low non strict slack digit that is not a byte", groups[9].0),
            ("a high strict slack digit that is not a byte", groups[10].0),
            ("a high non strict slack digit that is not a byte", groups[11].0),
        ]
        .iter()
        .enumerate()
        {
            let (plan, witness, mut params) =
                tampered_tail_digits_fixture(&activation, *at, 300, 0xdc + index as u8);
            rejects_malicious_witness(&activation, &plan, &witness, &mut params, context);
        }
    }
    fn lookup_bits_skipping_foreign_keys(
        plan: &crate::proof::lookup::LookupPlan,
        keys: &[Fp3],
    ) -> Vec<Fp3> {
        let planes = plan.planes().unwrap();
        let rows = plan.table_rows().unwrap();
        let mut counts = vec![0u64; rows];
        for key in keys {
            for (row, count) in counts.iter_mut().enumerate() {
                if *key == Fp3::from_u64(row as u64) {
                    *count += 1;
                    break;
                }
            }
        }
        let mut bits = vec![Fp3::ZERO; planes * rows];
        for row in 0..rows {
            for plane in 0..planes {
                if (counts[row] >> plane) & 1 == 1 {
                    bits[plane + planes * row] = Fp3::ONE;
                }
            }
        }
        bits
    }

    #[test]
    fn norm_malicious_slack_key_cannot_pass_the_byte_lookup() {
        use crate::proof::lookup::{
            prove_lookup_with_key_values, verify_lookup, ByteTable, LookupWitness,
        };
        use crate::proof::lookup_messages::{
            commit_lookup, LookupRandomSource, LookupRoots,
        };

        let activation = activation();
        let (plan, witness, _) = slack_lookup_fixture(&activation);
        let norm = plan.norm.as_ref().unwrap();
        let lookup_plan = norm.lookup.as_ref().unwrap();
        let tail_plan = norm
            .output
            .as_ref()
            .unwrap()
            .tail
            .as_ref()
            .unwrap()
            .clone();
        let groups = crate::proof::hidden_norm_psc::tail_digit_groups(&tail_plan).unwrap();
        let low_nonstrict_at = groups[9].0;
        let round_rows = witness.round.as_ref().unwrap().committed.messages.clone();
        let mut tail_rows = witness
            .output
            .as_ref()
            .unwrap()
            .tail
            .as_ref()
            .unwrap()
            .committed
            .messages
            .clone();
        let tail_rows_in_group = usize::try_from(tail_plan.group.shape.main_rows).unwrap();
        let ell = usize::try_from(tail_plan.group.shape.code.ell).unwrap();
        let slot = low_nonstrict_at;
        let row = slot % tail_rows_in_group;
        let place = slot / tail_rows_in_group;
        assert!(place < ell);
        tail_rows[0][row][place] =
            crate::proof::output_closure::fp3_signed(-2).unwrap();
        let keys = norm_lookup_key_values(norm, lookup_plan, &round_rows, &tail_rows).unwrap();
        let bits = lookup_bits_skipping_foreign_keys(&lookup_plan.lookup, &keys);
        let mask_entries = crate::proof::lookup::mask_count(&lookup_plan.lookup).unwrap();
        let mut source = LookupRandomSource::new(11);
        let messages = commit_lookup(
            &lookup_plan.commit,
            &bits,
            mask_entries,
            &mut source,
        )
        .unwrap();
        let seed = 23u64;
        let proved = prove_lookup_with_key_values(
            &lookup_plan.lookup,
            &lookup_plan.commit,
            &lookup_plan.key,
            &keys,
            &LookupWitness { bits },
            &messages,
            &ByteTable,
            seed,
        );
        match proved {
            Err(_) => {}
            Ok((proof, _)) => {
                let roots = LookupRoots {
                    bits: messages.bits_root(),
                    mask: messages.mask_root,
                };
                assert!(
                    verify_lookup(
                        &lookup_plan.lookup,
                        &lookup_plan.commit,
                        &lookup_plan.key,
                        &roots,
                        &proof,
                        &ByteTable,
                        seed,
                    )
                    .is_err(),
                    "a non-byte slack key must not pass the byte lookup"
                );
            }
        }
    }
    #[test]
    fn norm_output_groups_prove_and_verify() {
        let activation = activation();
        let mut table = table();
        let (plan, witness, mut params) = norm_output_fixture(&activation);
        let (messages, hidden_witness) = prove_hidden_lin_hidden_with_norm_and_params(
            &plan,
            &activation,
            &mut table,
            Digest([0x71; 32]),
            &witness,
            &mut params,
        )
        .unwrap();
        let mut verifier_plan = plan.clone();
        verifier_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        let outcome = verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &messages).unwrap();
        assert_eq!(outcome.first_seal, messages.first_seal);
        let output_plan = plan.norm.as_ref().unwrap().output.as_ref().unwrap();
        let output = messages.first.norm.as_ref().unwrap().output.as_ref().unwrap();
        assert_eq!(
            output.broadcast.message.group.group,
            output_plan.broadcast.group
        );
        assert_eq!(output.out.message.group.group, output_plan.out.group);

        let bytes = messages.encode().unwrap();
        let base = HiddenLinMessages::decode(&plan, &bytes).unwrap();
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &base).is_ok());

        let mut changed = base.clone();
        changed
            .first
            .norm
            .as_mut()
            .unwrap()
            .output
            .as_mut()
            .unwrap()
            .product
            .message
            .group
            .entries[0]
            .root = Digest([0x99; 32]);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        let sample = &mut changed
            .first
            .norm
            .as_mut()
            .unwrap()
            .output
            .as_mut()
            .unwrap()
            .image
            .sample
            .answers[0];
        *sample = sample.add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        changed
            .first
            .norm
            .as_mut()
            .unwrap()
            .output
            .as_mut()
            .unwrap()
            .out
            .positions[0][0] += 1;
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        changed.first.norm.as_mut().unwrap().output = None;
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut wrong_plan = plan.clone();
        wrong_plan.send.expected_roots = Some(hidden_witness.send.roots.clone());
        wrong_plan.norm.as_mut().unwrap().output.as_mut().unwrap().out.group = 94;
        assert!(verify_hidden_lin_hidden(&wrong_plan, &table.reader(), &base).is_err());

        let second = base.second.norm.as_ref().unwrap().output.as_ref().unwrap();
        assert_eq!(second.claims.len(), POSITIONS as usize * 2);
        assert_eq!(second.proofs.len(), POSITIONS as usize * 2);
        let mut changed = base.clone();
        changed
            .second
            .norm
            .as_mut()
            .unwrap()
            .output
            .as_mut()
            .unwrap()
            .proofs[0]
            .rounds
            .pop();
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        changed
            .second
            .norm
            .as_mut()
            .unwrap()
            .output
            .as_mut()
            .unwrap()
            .proofs[1]
            .end_values
            .pop();
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        changed
            .second
            .norm
            .as_mut()
            .unwrap()
            .output
            .as_mut()
            .unwrap()
            .claims
            .pop();
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        changed.second.norm.as_mut().unwrap().output = None;
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        // The claim value is now bound to the committed product/image row at
        // the original column point; a changed claim is rejected.
        let mut changed = base.clone();
        let claim = &mut changed
            .second
            .norm
            .as_mut()
            .unwrap()
            .output
            .as_mut()
            .unwrap()
            .claims[0];
        *claim = claim.add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        // The shown end values are bound to their committed maps; changed
        // end values are rejected.
        let mut changed = base.clone();
        let end = &mut changed
            .second
            .norm
            .as_mut()
            .unwrap()
            .output
            .as_mut()
            .unwrap()
            .proofs[0]
            .end_values[0];
        *end = end.add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        let end = &mut changed
            .second
            .norm
            .as_mut()
            .unwrap()
            .output
            .as_mut()
            .unwrap()
            .proofs[1]
            .end_values[1];
        *end = end.add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        // A changed round coefficient changes the masked rounds and the
        // hiding form; the bound rows reject it.
        let mut changed = base.clone();
        let round = &mut changed
            .second
            .norm
            .as_mut()
            .unwrap()
            .output
            .as_mut()
            .unwrap()
            .proofs[2]
            .rounds[0]
            .coefficients[0];
        *round = round.add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        // A changed hiding sum is rejected by the hiding-form row.
        let mut changed = base.clone();
        let hiding_sum = &mut changed
            .second
            .norm
            .as_mut()
            .unwrap()
            .output
            .as_mut()
            .unwrap()
            .proofs[3]
            .hiding_sum;
        *hiding_sum = hiding_sum.add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        // The committed tail group: a changed root, sample answer, opening
        // symbol or a missing section is rejected.
        let mut changed = base.clone();
        changed
            .first
            .norm
            .as_mut()
            .unwrap()
            .tail
            .as_mut()
            .unwrap()
            .group
            .message
            .group
            .entries[0]
            .root = Digest([0x98; 32]);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        let answer = &mut changed
            .first
            .norm
            .as_mut()
            .unwrap()
            .tail
            .as_mut()
            .unwrap()
            .group
            .sample
            .answers[0];
        *answer = answer.add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        let symbol = &mut changed
            .first
            .norm
            .as_mut()
            .unwrap()
            .tail
            .as_mut()
            .unwrap()
            .group
            .openings[0]
            .columns[0]
            .symbols[0];
        *symbol = *symbol + fp_from_signed_word(1);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        changed.first.norm.as_mut().unwrap().tail = None;
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        // The tail relation transcripts are sectioned and structural: a
        // popped proof, a popped point value or a missing section is rejected.
        assert_eq!(
            base.second.norm.as_ref().unwrap().tail.as_ref().unwrap().psc.len(),
            POSITIONS as usize * crate::proof::hidden_norm_psc::NORM_OUTPUT_TAIL_RELATIONS
        );
        let mut changed = base.clone();
        changed
            .second
            .norm
            .as_mut()
            .unwrap()
            .tail
            .as_mut()
            .unwrap()
            .psc
            .pop();
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        changed
            .second
            .norm
            .as_mut()
            .unwrap()
            .tail
            .as_mut()
            .unwrap()
            .point_values
            .pop();
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        let mut changed = base.clone();
        changed.second.norm.as_mut().unwrap().tail = None;
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());

        // The tail shown values are bound to the committed slots by the rows:
        // a changed point value is rejected.
        let mut changed = base.clone();
        let value = &mut changed
            .second
            .norm
            .as_mut()
            .unwrap()
            .tail
            .as_mut()
            .unwrap()
            .point_values[0][0];
        *value = value.add(&Fp3::ONE);
        assert!(verify_hidden_lin_hidden(&verifier_plan, &table.reader(), &changed).is_err());
    }

    #[test]
    fn norm_pointwise_masks_are_allocated_after_the_relation_blocks() {
        use crate::proof::hidden_norm_psc::{norm_pointwise_entries, norm_pointwise_masks};

        let activation = activation();
        let (plan, witness, _) = norm_output_fixture(&activation);
        let norm = plan.norm.as_ref().unwrap();
        let mask_plan = norm.mask.as_ref().unwrap();
        let mask_witness = witness.mask.as_ref().unwrap();
        let relations = norm.relations.len();
        let instances = POSITIONS as usize * relations;
        let (hiding, shown) = norm_pointwise_masks(norm.variables).unwrap();
        assert_eq!(mask_plan.pointwise, 2);
        assert_eq!(mask_plan.pointwise_hiding, hiding);
        assert_eq!(mask_plan.pointwise_shown, shown);
        let entries = norm_pointwise_entries(mask_plan, &mask_witness.committed, instances).unwrap();
        assert_eq!(entries.len(), POSITIONS as usize * 2);
        for entry in &entries {
            assert_eq!(entry.hiding.len(), hiding);
            assert_eq!(entry.shown.len(), shown);
        }
        let first = mask_plan.pointwise_hiding_slot(instances, 0, 0).unwrap();
        let last = mask_plan
            .pointwise_shown_slot(instances, entries.len() - 1, shown - 1)
            .unwrap();
        assert!(first >= instances * mask_plan.per_instance().unwrap());
        assert!(last + 1 <= mask_plan.total_entries(relations).unwrap());

        let mut without_output = plan.clone();
        without_output.norm.as_mut().unwrap().output = None;
        without_output.norm.as_mut().unwrap().mask.as_mut().unwrap().pointwise = 2;
        assert!(without_output.validate().is_err());

        let mut without_allocation = plan.clone();
        without_allocation.norm.as_mut().unwrap().mask.as_mut().unwrap().pointwise = 0;
        assert!(without_allocation.validate().is_err());

        let mut wrong_counts = plan.clone();
        wrong_counts
            .norm
            .as_mut()
            .unwrap()
            .mask
            .as_mut()
            .unwrap()
            .pointwise_hiding = hiding + 1;
        assert!(wrong_counts.validate().is_err());
    }
}

#[cfg(test)]
mod rework7_joint_tests {
    use super::*;

    #[test]
    fn eval_target_is_the_real_multilinear_evaluation_plus_a_one_time_mask() {
        let point = [Fp3::from_u64(3), Fp3::from_u64(5)];
        let main_rows = 2u64;
        let columns = 4u64;
        let mut coefficients = Vec::new();
        for index in 0..4u64 {
            let mut message = vec![vec![Fp3::ZERO; 2]; 2];
            message[(index % main_rows) as usize][(index / main_rows) as usize] = Fp3::ONE;
            coefficients.push(eval_message(&point, &message, main_rows, columns).unwrap());
        }
        for (index, coefficient) in coefficients.iter().enumerate() {
            assert_eq!(*coefficient, eq_index(&point, index as u64, 2).unwrap());
        }
        assert_eq!(
            coefficients.iter().filter(|value| !value.is_zero()).count(),
            4
        );
        let base = eval_message(
            &point,
            &vec![vec![Fp3::from_u64(7); 2]; 2],
            main_rows,
            columns,
        )
        .unwrap();
        let mask = Fp3::from_u64(11);
        assert_eq!(base.add(&mask).sub(&base), mask);
    }

    #[test]
    fn send_answer_is_the_real_index_mask_sample_plus_its_pad() {
        let mask_code = build_code_plan(6, 32, 1).unwrap();
        let plan = ClosureMaskPlan {
            oracle: 9,
            group: 1,
            mask_code,
            mask_rows: 1,
            answers: 1,
            openings: 1,
            tree: RuntimeTreePlan {
                tree_number: 9,
                leaf_class: 1,
                log_n: 6,
                log_n_zk: 4,
                parts: vec![RuntimeTreePart {
                    main_symbols: 0,
                    mask_symbols: 3,
                }],
                salt_bytes: 40,
                run: Digest([0x61; 32]),
                configuration: Digest([0x62; 32]),
            },
            positions: 1,
            hiding: hp_entries(&psc_variables(4).unwrap()).unwrap(),
            shown: 5,
        };
        let seed = SampleSeed {
            z: vec![
                Fp3::from_u64(3),
                Fp3::from_u64(5),
                Fp3::from_u64(7),
                Fp3::from_u64(9),
                Fp3::from_u64(11),
            ],
            beta: Fp3::from_u64(13),
        };
        let mut coefficients = Vec::new();
        for column in 0..32usize {
            let mut row = vec![Fp3::ZERO; 32];
            row[column] = Fp3::ONE;
            coefficients.push(closure_mask_answer(&plan, &[row], &seed).unwrap());
        }
        for (index, coefficient) in coefficients.iter().enumerate() {
            if index + 1 == 32 {
                assert_eq!(*coefficient, seed.beta);
            } else {
                assert_eq!(*coefficient, eq_index(&seed.z, index as u64, 5).unwrap());
            }
        }
        assert!(!coefficients[31].is_zero());
    }
}



#[cfg(test)]
mod iopp_stage_plan_tests {
    use super::*;

    fn stage(main: CodeParams, mask: CodeParams, k: u32, h: u32, positions: usize) -> IoppStagePlan {
        IoppStagePlan {
            main,
            mask,
            k,
            h,
            positions,
            samples: 1,
            work_bits: 0,
        }
    }

    #[test]
    fn actual_shared_and_own_stage_lists_are_legal() {
        let main = build_code_plan(17, 65_536, 8_192).unwrap();
        let mask = build_code_plan(17, 512, 256).unwrap();
        let shared = vec![
            stage(main, mask, 2, 0, 2),
            stage(
                build_code_plan(17, 65_536, 256).unwrap(),
                build_code_plan(17, 16_384, 256).unwrap(),
                0,
                0,
                2,
            ),
        ];
        assert!(iopp_stage_fault(&shared, main, mask, 2).is_none());
        let own = vec![
            stage(main, mask, 14, 2, 2),
            stage(
                build_code_plan(17, 8, 256).unwrap(),
                build_code_plan(17, 16, 256).unwrap(),
                13,
                13,
                2,
            ),
            stage(
                build_code_plan(17, 8, 256).unwrap(),
                build_code_plan(17, 1_024, 256).unwrap(),
                0,
                0,
                2,
            ),
        ];
        assert!(iopp_stage_fault(&own, main, mask, 2).is_none());
    }

    #[test]
    fn bad_stage_chains_are_rejected() {
        let main = build_code_plan(17, 65_536, 8_192).unwrap();
        let mask = build_code_plan(17, 512, 256).unwrap();
        // Wrong first code.
        let wrong_first = vec![
            stage(build_code_plan(17, 32_768, 8_192).unwrap(), mask, 2, 0, 2),
            stage(main, build_code_plan(17, 16_384, 256).unwrap(), 0, 0, 2),
        ];
        assert!(iopp_stage_fault(&wrong_first, main, mask, 2).is_some());
        // Wrong adjacent folded length.
        let wrong_fold = vec![
            stage(main, mask, 2, 0, 2),
            stage(build_code_plan(17, 8_192, 256).unwrap(), build_code_plan(17, 16_384, 256).unwrap(), 0, 0, 2),
        ];
        assert!(iopp_stage_fault(&wrong_fold, main, mask, 2).is_some());
        // Last stage not k=h=0.
        let wrong_last = vec![
            stage(main, mask, 2, 0, 2),
            stage(build_code_plan(17, 65_536, 256).unwrap(), build_code_plan(17, 16_384, 256).unwrap(), 1, 0, 2),
        ];
        assert!(iopp_stage_fault(&wrong_last, main, mask, 2).is_some());
    }

}
