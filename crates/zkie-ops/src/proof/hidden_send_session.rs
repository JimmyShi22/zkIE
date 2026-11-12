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
use crate::proof::hidden_norm_psc::{
    NormFirstMessages, NormGroupPlan, NormGroupWitness, NormMaskMessage, NormMaskPlan,
    NormMaskWitness, NormPlan, NormPositionBinding, NormPscSecond, NormRowBinding, NormWitness,
};
use crate::proof::committed_psc::CommittedPscGroup;
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
    // The row-weight vectors are fixed before the read; they stay private.
    let mut weights = Vec::with_capacity(positions);
    for position in 0..positions {
        weights.push(activation.weights(position as u32, plan.matrix_rows)?);
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
        if let Some(norm) = norm {
            if let Some(mask) = &norm.mask {
                let rows = mask.mask_rows.next_power_of_two();
                own_mask_rows = match own_mask_rows.checked_add(rows) {
                    Some(rows) => rows,
                    None => return Some("the own mask row count overflows"),
                };
            }
        }
        if own_mask_rows == 0 {
            own_mask_rows = 1;
        }
        let own_mask_rows = own_mask_rows.next_power_of_two();
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
        if self
            .proximity
            .fault(&self.send, self.closure.as_ref(), self.norm.as_ref())
            .is_some()
        {
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
fn to_bat_table(values: &[Fp3], rows: u64, ell: u64) -> Result<Vec<Fp3>> {
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
        provers.push(ProximityMemberProver::sealed(member, tree)?);
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
        let dot = match &plan.dot {
            Some(dot) => {
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
            None => None,
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
    activation: &CommittedActivation,
    table: &mut dyn HiddenTableProver,
    output: Option<&[Vec<Fp3>]>,
    randomness: &WCommitRandomness,
    norm: Option<&NormWitness>,
) -> Result<(HiddenSendMessages, Digest, HiddenSendWitness)> {
    let _profile_first = zkie_engine::profile::Scope::new("first.prove_first");
    plan.validate()?;
    let reader = table.reader().clone();
    check_activation(&plan.send, activation)?;
    check_table(&plan.send, &reader)?;
    let _profile_derive = zkie_engine::profile::Scope::new("first.derive_w_rows");
    let rows = derive_w_rows(&plan.send, activation, &reader, table)?;
    drop(_profile_derive);
    let _profile_commit_w = zkie_engine::profile::Scope::new("first.commit_w_rows");
    let (group, mut witness) = commit_w_rows(&plan.send, &rows, randomness)?;
    drop(_profile_commit_w);
    let dot_root = match &plan.send.dot {
        Some(_) => {
            let _profile_dot = zkie_engine::profile::Scope::new("first.commit_dot_map");
            let (root, map) = commit_dot_map(&plan.send, activation, randomness)?;
            drop(_profile_dot);
            witness.dot = Some(map);
            Some(root)
        }
        None => None,
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
    match (&plan.send.dot, dot_root) {
        (Some(_), Some(root)) => absorb_dot_root(&mut chain, &plan.send, &root)?,
        (None, None) => {}
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
            let message = prove_norm_first(plan, &mut chain, norm_witness)?;
            witness.norm = Some(norm_witness.clone());
            Some(message)
        }
        (None, None) => None,
        _ => return Err(Error::ShapeMismatch),
    };
    let seeds = draw_seeds(&mut chain, &plan.send)?;
    let dot_seeds = match &plan.send.dot {
        Some(dot) => Some(draw_seed_set(&mut chain, dot.sample_bits()?, 1)?),
        None => None,
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
        || messages.activation.root != plan.send.activation_root
        || messages.activation.shape != plan.send.activation
        || messages.activation.configuration != plan.send.activation_configuration
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
    let seeds = draw_seeds(&mut chain, &plan.send)?;
    if seeds != messages.samples.seed {
        return Err(Error::InvalidProof);
    }
    let dot_seeds = match &plan.send.dot {
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

/// Prover side of the shared stage.
fn prove_shared(
    plan: &HiddenLinPlan,
    reader: &IndexReader,
    table: &mut dyn HiddenTableProver,
    first_seal: &Digest,
    seed: &Digest,
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
    drop(_profile_plain_rows);
    let trees = ProximityTreeSource::Ready(vec![
        (members[gu_bat as usize].source_oracle, Arc::new(gu_tree)),
        (
            members[omega_bat as usize].source_oracle,
            Arc::new(omega_tree),
        ),
    ]);
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
    let mut sample_rows = Vec::with_capacity(members.len());
    for member in &members {
        if member.source_oracle == gu.oracle {
            sample_rows.push(sample_rows_of(&gu, &registry, &gu_seeds, &messages.gu_answers)?);
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
    if let Some(dot) = &plan.send.dot {
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
        let groups = [
            (&norm_first.x, &norm_plan.x, norm_witness.map(|w| &w.x)),
            (&norm_first.sum, &norm_plan.sum, norm_witness.map(|w| &w.sum)),
            (&norm_first.squares, &norm_plan.squares, norm_witness.map(|w| &w.squares)),
            (&norm_first.mean, &norm_plan.mean, norm_witness.map(|w| &w.mean)),
            (&norm_first.variance, &norm_plan.variance, norm_witness.map(|w| &w.variance)),
        ];
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
    }
    members.sort_by_key(|member| std::cmp::Reverse((member.main_rows, member.source_oracle)));
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
            let norm_first = first.norm.as_ref().ok_or(Error::ShapeMismatch)?;
            let groups = norm_group_plans(norm_plan);
            let messages = [
                &norm_first.x,
                &norm_first.sum,
                &norm_first.squares,
                &norm_first.mean,
                &norm_first.variance,
            ];
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
        if let Some(dot) = &plan.send.dot {
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

fn norm_group_plans(plan: &NormPlan) -> [&NormGroupPlan; 5] {
    [&plan.x, &plan.sum, &plan.squares, &plan.mean, &plan.variance]
}

fn norm_group_witnesses(witness: &NormWitness) -> [&NormGroupWitness; 5] {
    [
        &witness.x,
        &witness.sum,
        &witness.squares,
        &witness.mean,
        &witness.variance,
    ]
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

fn norm_mask_sample_row(
    plan: &NormMaskPlan,
    message: &NormMaskMessage,
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

fn prove_norm_first(
    plan: &HiddenLinPlan,
    chain: &mut Chain,
    witness: &NormWitness,
) -> Result<NormFirstMessages> {
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
    let mut iter = group_messages.into_iter();
    Ok(NormFirstMessages {
        x: iter.next().ok_or(Error::ShapeMismatch)?,
        sum: iter.next().ok_or(Error::ShapeMismatch)?,
        squares: iter.next().ok_or(Error::ShapeMismatch)?,
        mean: iter.next().ok_or(Error::ShapeMismatch)?,
        variance: iter.next().ok_or(Error::ShapeMismatch)?,
        mask,
    })
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
    Ok(())
}

fn prove_norm_second(
    chain: &mut Chain,
    plan: &NormPlan,
    first: &NormFirstMessages,
    witness: &NormWitness,
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
    Ok((
        NormPscSecond { psc, point_values },
        NormRowBinding {
            positions: bindings,
        },
    ))
}

fn verify_norm_second(
    chain: &mut Chain,
    plan: &NormPlan,
    first: &NormFirstMessages,
    messages: &NormPscSecond,
) -> Result<NormRowBinding> {
    let positions = plan.shape.positions as usize;
    let relations = plan.relations.clone();
    let variables = plan.variables;
    let mask_plan = plan.mask.as_ref().ok_or(Error::InvalidConfig)?;
    if first.mask.is_none() {
        return Err(Error::ShapeMismatch);
    }
    messages.check_for(plan)?;
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
    Ok(NormRowBinding {
        positions: bindings,
    })
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
    let dot_member = bat_of.get(&dot.oracle).ok_or(Error::StatementMismatch)?;
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
    let dot_rows = usize::try_from(dot.shape.main_rows).map_err(|_| Error::ResourceLimit)?;
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
    let end_start = index.entries.end_range()?.start;
    let mut row = BatRow::new(BatRowKind::Plain);
    row.add_term(dot_member.bat_oracle, BatPart::Main, coefficients);
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
                prove_norm_second(&mut chain, norm_plan, first_norm, norm_witness)?;
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
    let _ = first;
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
        if let Some(mask_plan) = &norm_plan.mask {
            if member.source_oracle == mask_plan.oracle {
                let mask = norm_witness.mask.as_ref().ok_or(Error::ShapeMismatch)?;
                return Ok(Arc::new(mask.committed.tree(mask_plan)?));
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
    if let Some(dot) = &plan.send.dot {
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
) -> Result<(
    Digest,
    Vec<GuRelationRow>,
    ProximityBaseOutcome,
    Option<DotRowBinding>,
    Option<ClosureRowBinding>,
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
            verify_norm_second(&mut chain, norm_plan, first_norm, norm_messages)?,
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
    ))
}


// ---------------------------------------------------------------------------
// Public entries
// ---------------------------------------------------------------------------

fn prove_hidden_lin_inner(
    plan: &HiddenLinPlan,
    activation: &CommittedActivation,
    table: &mut dyn HiddenTableProver,
    output: Option<&[Vec<Fp3>]>,
    seed: Digest,
    bind_residual: bool,
    norm: Option<&NormWitness>,
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
        prove_first(plan, activation, table, output, &randomness, norm)?;
    let first_query = *first_chain(plan, &reader)?.value();
    let shared = prove_shared(plan, &reader, table, &first_seal, &seed)?;
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
    prove_hidden_lin_inner(plan, activation, table, Some(output), seed, true, None)
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
    prove_hidden_lin_inner(plan, activation, table, None, seed, true, None)
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
    prove_hidden_lin_inner(plan, activation, table, None, seed, true, Some(norm))
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
    prove_hidden_lin_inner(plan, activation, table, None, seed, false, None)
}

fn verify_hidden_lin_inner(
    plan: &HiddenLinPlan,
    reader: &IndexReader,
    output: Option<&[Vec<Fp3>]>,
    messages: &HiddenLinMessages,
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
        verify_shared(plan, reader, &first_seal, &messages.shared)?;
    drop(_profile_shared);
    if shared_seal != messages.shared.seal {
        return Err(Error::InvalidProof);
    }
    let _profile_second = zkie_engine::profile::Scope::new("verify.hidden_lin.second");
    let (seal, own_claims, own_base, dot_binding, closure_binding) = verify_second(
        plan,
        reader,
        &messages.first,
        &first_query,
        &messages.shared,
        &points.psi,
        &messages.second,
        closure_r.as_deref(),
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
    verify_hidden_lin_inner(plan, reader, Some(output), messages)
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
    verify_hidden_lin_inner(plan, reader, None, messages)
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

fn put_column_opening(writer: &mut CanonicalWriter, opening: &ColumnOpening) {
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

fn get_column_opening(reader: &mut CanonicalReader<'_>, max: usize) -> Result<ColumnOpening> {
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
    const CODE_LOG_N: u32 = 6;
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
            mask_rows: 3,
            answers: 1,
            openings: 1,
            tree: RuntimeTreePlan {
                tree_number: INDEX_MASK_TREE,
                leaf_class: 1,
                log_n: CODE_LOG_N,
                log_n_zk: CODE_LOG_N,
                parts: vec![RuntimeTreePart {
                    main_symbols: 0,
                    mask_symbols: 3 * E_DEGREE as u64,
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
                log_n: 6,
                log_n_zk: 6,
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
            NormPlan, NormWitness,
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
            mask_rows: 4,
            answers: 1,
            openings: 1,
            tree: RuntimeTreePlan {
                tree_number: 60500,
                leaf_class: 1,
                log_n: CODE_LOG_N,
                log_n_zk: CODE_LOG_N,
                parts: vec![RuntimeTreePart {
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
        };
        let mut norm = NormPlan {
            shape: crate::proof::hidden_norm::NormShape::new(POSITIONS, HIDDEN, HIDDEN)
                .unwrap(),
            columns,
            variables: columns.trailing_zeros() as usize + 1,
            relations: vec![0, 1, 2, 3],
            x: group(columns, GROUP, 9000, 60000),
            x_roots: Vec::new(),
            sum: group(1, GROUP, 9100, 60100),
            squares: group(1, GROUP, 9200, 60200),
            mean: group(1, GROUP, 9300, 60300),
            variance: group(1, GROUP, 9400, 60400),
            mask: Some(mask_plan.clone()),
        };
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
            mask: Some(mask_witness),
        };
        let mut plan = hidden_plan(_activation);
        plan.proximity.own_mask_tree.mask_rows = 8;
        plan.norm = Some(norm);
        plan.validate().unwrap();
        (plan, witness)
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
        assert_eq!(instances, POSITIONS as usize * 4);

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
