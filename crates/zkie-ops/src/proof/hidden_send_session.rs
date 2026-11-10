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
//! draws one Eval point per position. The Eval row has the form
//!
//! ```text
//! W_t at psi_t  =  target_t
//! ```
//!
//! with the `eq(psi_t, .)` coefficients on the `W_t` message, and the target is
//! public. `X`'s row-weight vector is what produced `W_t`; it is not revealed.
//! The sample rows and the Eval rows belong to the same group and are returned
//! together for the later stages.
//!
//! # Order
//!
//! 1. the activation commitment and the `W` roots are fixed;
//! 2. the roots and the group declaration are absorbed and the shared seeds
//!    are drawn;
//! 3. the complete ordered answers are absorbed;
//! 4. the `W` opening positions are absorbed and drawn; the openings are
//!    checked;
//! 5. one Eval point per position is absorbed and drawn;
//! 6. the Eval targets are absorbed and the first message value is fixed.
//!
//! No position is drawn between the seed and the answers. The seed is drawn by
//! the verifier from the chain; the message copy of the seed is checked against
//! the drawn value and is never used as a challenge source.
//!
//! # Scope
//!
//! This unit fixes and samples the hidden inputs; it does not run the proximity
//! tests, the Dot/Fold relations, the code switch or a soundness bound. The
//! relation between the `W` roots and the encoded table root is established by
//! the later Eval/Dot/Fold relations, not by the SEND phase. The encoded table
//! identity is checked against the plan, and the `W` derivation reads the
//! table's row source and checks its shape, but no function here proves to the
//! verifier that the `W` message is `g^T M`: the verifier holds no `M` and no
//! `X`. The sample answers are public and padded with a private random value;
//! the pads and tails of this unit are supplied by the caller and the tests use
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
//! first query and the shared seal, adopts `Omega`, registers the Dot rows and
//! runs the proximity test on the request's own group, ending in the BASE
//! consumption and the final seal.
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
//! Eval targets are message values, each bound to its committed reduced row by
//! a BAT row. The reduced rows are the MatMul output and are committed under
//! the same hiding product-form encoding, so the output never needs a separate
//! plaintext input. The Dot row, whose coefficients are the activation row, is
//! not part of this mode because the verifier has no public coefficient
//! source; the diagnostic mode keeps it. Closing the Dot relation with a
//! hidden activation requires a committed inner-product mechanism that this
//! session does not have.

use zkie_core::common::codec::DecodeLimits;
use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use zkie_core::common::field::PrimeField64;
use zkie_engine::compute::cpu::{merkle_build, merkle_verify};
use zkie_engine::pcs::request::bat::{
    BatGroupLayout, BatOracleShape, BatPart, BatRow, BatRowKind, BatSession, BatStatement,
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
use zkie_engine::pcs::request::pow::{search_nonce, verify_nonce};
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

use crate::proof::base_final_session::{
    fold_identity, BaseFinalCheck, BaseFinalOutcome, BaseFinalRows, BaseFinalSession, BaseMessages,
    BaseCommitment, BaseOpening, BaseOpenings, BaseProver, FoldBasis, FoldMember,
};
use crate::proof::fold_handoff::{FoldHandoff, FoldHandoffAnchor};

/// Version of the canonical messages of this interface.
const HIDDEN_SEND_SCHEMA_VERSION: u64 = 1;
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

    fn encode_into(&self, writer: &mut CanonicalWriter) {
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
        if self.activation.hidden as u64 > self.matrix_rows {
            return Err(Error::ShapeMismatch);
        }
        if self.openings_per_oracle == 0 || self.openings_per_oracle as u64 > self.w.code.n() {
            return Err(Error::InvalidConfig);
        }
        if self.w.fault(self.columns, self.activation.positions).is_some() {
            return Err(Error::InvalidConfig);
        }
        for position in 0..self.activation.positions {
            self.oracle(position)?;
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
        self.w.encode_into(&mut writer);
        writer.into_bytes()
    }
}

// ---------------------------------------------------------------------------
// Prover-side W derivation and commitment
// ---------------------------------------------------------------------------

/// The private random values of one `W` commitment: one tail per message row,
/// one salt per tree column and one pad per answer. Every pad must be non-zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WCommitRandomness {
    /// `tails[position][row * r + i]`, one entry per message row and tail
    /// coefficient.
    pub tails: Vec<Vec<Fp3>>,
    /// `salts[position][column]`, one salt per tree column.
    pub salts: Vec<Vec<Vec<u8>>>,
    /// `pads[position][answer]`, non-zero.
    pub pads: Vec<Vec<Fp3>>,
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
        };
        randomness.check(plan)?;
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

    /// Rebuilds the committed runtime tree of one position from the retained
    /// messages, tails and salts. The root must equal the registered root.
    pub fn tree(&self, position: u32) -> Result<RuntimeTree> {
        if position >= self.positions {
            return Err(Error::InvalidIndex);
        }
        let message = self
            .messages
            .get(position as usize)
            .ok_or(Error::InvalidIndex)?;
        let tails = self
            .tails
            .get(position as usize)
            .ok_or(Error::InvalidIndex)?;
        let salts = self
            .salts
            .get(position as usize)
            .ok_or(Error::InvalidIndex)?;
        let main_rows = usize::try_from(self.w.main_rows).map_err(|_| Error::ResourceLimit)?;
        let tail_len = usize::try_from(self.w.code.r).map_err(|_| Error::ResourceLimit)?;
        if message.len() != main_rows || tails.len() != main_rows * tail_len {
            return Err(Error::ShapeMismatch);
        }
        let mut codewords = Vec::with_capacity(main_rows);
        for (row, message_row) in message.iter().enumerate() {
            let tail = tails
                .get(row * tail_len..(row + 1) * tail_len)
                .ok_or(Error::ShapeMismatch)?;
            codewords.push(encode_row_fp3(&self.w.code, message_row, tail)?);
        }
        RuntimeTree::commit(
            self.w.tree_plan(position)?,
            &[RuntimePartRows {
                main: codewords,
                mask: Vec::new(),
                main_deg: E_DEGREE as u32,
            }],
            salts,
        )
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
    };
    Ok((group, witness))
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
    pub eval_targets: Vec<Fp3>,
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
        writer.put_digest(&self.chain_value);
        Ok(writer.into_bytes())
    }

    /// Decodes a message against the plan. The plan supplies the trusted tree
    /// shapes of the openings; the decoded group is checked by the verifier.
    pub fn decode(plan: &HiddenSendPlan, bytes: &[u8]) -> Result<Self> {
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
    let (group, witness) = commit_w_rows(plan, &rows, randomness)?;
    witness.check(plan)?;

    let mut chain = Chain::begin(
        HIDDEN_SEND_CHAIN_LABEL,
        &plan.parameters_digest(),
        &plan.statement_bytes(),
        &plan.table_identity.0,
        None,
    )?;
    absorb_group(&mut chain, &group)?;
    let seeds = draw_seeds(&mut chain, plan)?;
    let answers = witness.answers(&seeds)?;
    absorb_answers(&mut chain, &answers)?;
    let positions = draw_positions(&mut chain, plan)?;
    let mut openings = Vec::with_capacity(positions.len());
    for (position, list) in positions.iter().enumerate() {
        openings.push(witness.tree(position as u32)?.open(list)?);
    }
    let psi = draw_psi(&mut chain, plan)?;
    let eval_targets = witness.eval_values(&psi)?;
    absorb_eval(&mut chain, plan, &eval_targets)?;
    let chain_value = *chain.value();

    let messages = HiddenSendMessages {
        activation: *activation.commitment(),
        group,
        samples: SendSamples { seed: seeds, answers },
        positions,
        openings,
        psi,
        eval_targets,
        chain_value,
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
    if expected != outcome.eval_targets {
        return Err(Error::InvalidProof);
    }
    for (position, row) in outcome.eval_rows.iter().enumerate() {
        let relation = witness.relation(position as u32)?;
        if row.evaluate(&relation)? != row.target {
            return Err(Error::InvalidProof);
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
    let seeds = draw_seeds(&mut chain, plan)?;
    if seeds != messages.samples.seed {
        return Err(Error::InvalidProof);
    }
    let expected_answers = plan.positions() as usize * plan.w.answers as usize;
    if messages.samples.answers.len() != expected_answers {
        return Err(Error::ShapeMismatch);
    }
    absorb_answers(&mut chain, &messages.samples.answers)?;
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
    let bits = plan.w.sample_bits()?;
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
}

impl HiddenProximityPlan {
    pub fn fault(&self, send: &HiddenSendPlan) -> Option<&'static str> {
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
        if self.shared_mask_tree.fault().is_some() || self.own_mask_tree.fault().is_some() {
            return Some("a fresh mask tree is not valid");
        }
        if self.shared_mask_tree.mask_rows != 1
            || self.own_mask_tree.mask_rows != 1
            || self.shared_mask_tree.log_n != self.mask_code.log_n
            || self.own_mask_tree.log_n != self.mask_code.log_n
        {
            return Some("a fresh mask tree does not match the fold mask code");
        }
        if self.shared_mask_tree.tree_number == self.own_mask_tree.tree_number {
            return Some("the two fresh masks share one tree number");
        }
        if self.shared_fresh == self.own_fresh {
            return Some("the two fresh masks share one oracle id");
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
        if self.proximity.fault(&self.send).is_some() {
            return Err(Error::InvalidConfig);
        }
        if self.output_relation == Some(Digest::ZERO) {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }

    fn parameters_digest(&self) -> Digest {
        self.send.parameters_digest()
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
        Ok(writer.into_bytes())
    }
}

fn put_code_public(writer: &mut CanonicalWriter, code: CodeParams) {
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
    pub base: BaseMessages,
    pub base_rows: BaseFinalRows,
    pub base_openings: BaseOpenings,
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
    pub fold_start: FoldStartMessage,
    pub fold_rounds: Vec<(Fp3, Fp3)>,
    pub base: BaseMessages,
    pub base_rows: BaseFinalRows,
    pub base_openings: BaseOpenings,
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
    pub shared_base: BaseFinalOutcome,
    pub own_base: BaseFinalOutcome,
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
    answers: u32,
    root: Digest,
    tree_plan: RuntimeTreePlan,
    /// Prover side only: the message rows and the encoding tails.
    message: Vec<Vec<Fp3>>,
    tails: Vec<Vec<Fp3>>,
}

/// A committed proximity member built by the prover.
struct ProximityMemberProver {
    member: ProximityMember,
    tree: RuntimeTree,
}

impl ProximityMemberProver {
    fn sealed(member: ProximityMember, tree: RuntimeTree) -> Result<Self> {
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
    base: BaseMessages,
    base_rows: BaseFinalRows,
    base_openings: BaseOpenings,
}

/// The verifier-side result of one proximity test.
struct ProximityVerification {
    seal: Digest,
    base: BaseFinalOutcome,
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
    let mut last = None;
    for member in members {
        if let Some(previous) = last {
            if member.bat_oracle <= previous {
                return Err(Error::InvalidConfig);
            }
        }
        last = Some(member.bat_oracle);
        if member.main_rows == 0 || !member.main_rows.is_power_of_two() {
            return Err(Error::InvalidConfig);
        }
        main_rows = main_rows
            .checked_add(member.main_rows)
            .ok_or(Error::ResourceLimit)?;
    }
    let k = if main_rows <= 1 {
        1
    } else {
        64 - (main_rows - 1).leading_zeros()
    };
    let shape = FoldShape {
        code,
        mask_code,
        k,
        h: 0,
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
        mask_rows: 0,
        main_code: Some(member.main_code),
        mask_code: None,
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
        || !row.mask.is_empty()
    {
        return Err(Error::ShapeMismatch);
    }
    let mut out = BatRow::new(BatRowKind::Sample);
    out.add_term(
        member.bat_oracle,
        BatPart::Main,
        to_bat_table(&row.main, member.main_rows, member.main_code.ell)?,
    );
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
    let group_bits = bat_layout.group_row_bits(BatPart::Main);
    let group_len = bat_layout.group_len(BatPart::Main)?;
    let mut values = vec![Fp3::ZERO; group_len];
    if layout.members.len() != bat_layout.members.len() || layout.members.len() != members.len() {
        return Err(Error::StatementMismatch);
    }
    let firsts = fold_member_firsts(layout, FoldPart::Main)?;
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
        let first = firsts[index].ok_or(Error::StatementMismatch)?;
        for (row, message) in member.message.iter().enumerate() {
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
    let mask_tail = vec![Fp3::ZERO; mask_len];
    if members.len() != layout.members.len() {
        return Err(Error::ShapeMismatch);
    }
    for (index, (member, layout_member)) in members.iter().zip(&layout.members).enumerate() {
        if member.bat_oracle != layout_member.oracle {
            return Err(Error::StatementMismatch);
        }
        let main_first = main_firsts[index].ok_or(Error::StatementMismatch)?;
        if member.tails.len() as u64 != layout_member.main_rows {
            return Err(Error::ShapeMismatch);
        }
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

/// The prover side of one proximity test: BAT over the registered claims,
/// FOLD rounds with a fresh mask, then BASE on the same chain. Every value the
/// verifier needs is in the returned messages; the member rows never leave the
/// prover.
#[allow(clippy::too_many_arguments)]
fn proximity_prove(
    chain: Chain,
    fold: FoldShape,
    members: &[ProximityMemberProver],
    sample_rows: &[Vec<GuRelationRow>],
    plain_rows: &[BatRow],
    fresh_id: u64,
    mask_tree: MaskTreePlan,
    seed: Digest,
    base_positions: usize,
) -> Result<ProximityProof> {
    if members.is_empty() || sample_rows.len() != members.len() {
        return Err(Error::ShapeMismatch);
    }
    let plain_members: Vec<ProximityMember> = members.iter().map(|m| m.member.clone()).collect();
    let mut statement = BatStatement::new();
    for member in &plain_members {
        statement.add_oracle(member_bat_shape(member))?;
    }
    for (index, member) in plain_members.iter().enumerate() {
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
    for member in &plain_members {
        statement.join(group, member.bat_oracle, false)?;
    }
    for row in plain_rows {
        statement.add_row(group, row.clone())?;
    }
    let layout = statement.layout(group)?;
    let fold_layout = statement.to_fold_layout_checked(group, &fold)?;
    let planned = plain_rows.len() as u64
        + plain_members
            .iter()
            .map(|member| member.answers as u64)
            .sum::<u64>();
    let mut bat = BatSession::new(statement, chain, planned)?;
    let input = bat.begin_run(group)?;
    let combined = bat.finish_run(input)?;
    let fold_row = bat.statement().to_fold_input_row(&combined, &layout)?;
    let chain = bat.into_chain()?;

    let group_main = build_group_table(&fold_layout, &layout, &plain_members)?;
    let group_mask = vec![Fp3::ZERO; layout.group_len(BatPart::Mask)?];

    let mut coins = ByteMaskCoins::new(seed)?;
    let mut mask = fold_mask_make(&fold, &mut coins)?;
    fold_mask_make_tail(&fold, &mut mask, &mut coins)?;
    mask.sent_as(fresh_id)?;
    let groups = VerifiedGroups {
        groups: vec![VerifiedGroup {
            group,
            members: plain_members.iter().map(|m| m.bat_oracle).collect(),
            fresh: Some(fresh_id),
            code: Some(fold.code),
            mask_code: Some(fold.mask_code),
        }],
    };
    let notice = fold_mask_notice(&fold)?;
    let oracle = MaskOracle::prover_view(fresh_id, group, &notice);
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
    let mut chain = chain;
    absorb_fold_start(&mut chain, &start_message)?;
    let epsilon = chain.draw_e()?;

    let mut claim = FoldRoundVerifier::new(start_message.sum, epsilon, fold_row.target);
    let mut prover = FoldProverSession::new(
        &fold,
        &fold_layout,
        &mask,
        &fold_row,
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
    for _ in 0..fold.k {
        let round = prover.round()?;
        absorb_fold_round(&mut chain, round[0], round[2])?;
        let challenge = chain.draw_e()?;
        claim.round(round[0], round[2], challenge);
        prover.bind(challenge)?;
        gamma.push(challenge);
        rounds.push((round[0], round[2]));
    }
    let output = fold_output(
        &fold,
        &fold_layout,
        &fold_row,
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
        members: plain_members.iter().map(|m| m.bat_oracle).collect(),
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
        folded_tails(&handoff, &plain_members, &fresh_tails)?;
    let rows = handoff.single_rows(
        prover.main_messages().to_vec(),
        main_tail,
        end.mu.clone(),
        mask_tail,
    )?;
    let identity = fold_identity(&handoff);
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
    let basis = FoldBasis {
        layout: handoff.layout.clone(),
        fresh_root: handoff.fresh_root,
        fresh_tree: mask_tree,
        members: plain_members
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
    let mut verifier = BaseFinalSession::new(chain, handoff, basis)?;
    let base_messages = base_prover.messages()?;
    verifier.receive_commitments(base_messages)?;
    let lambda = verifier.draw_lambda()?;
    let base_rows = base_prover.base_rows(lambda)?;
    verifier.receive_base_rows(base_rows.clone())?;
    let positions = verifier.draw_base_positions(base_positions)?;
    let unique = unique_sorted(&positions);
    let member_openings = members
        .iter()
        .map(|member| member.tree.open(&unique))
        .collect::<Result<Vec<_>>>()?;
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
    let (seal, _base) = match verifier.finish()? {
        BaseFinalCheck::Passed(base) => (base.check_value, base),
        BaseFinalCheck::Rejected(_) => return Err(Error::InvalidProof),
    };
    Ok(ProximityProof {
        messages: ProximityMessages {
            fold_start: start_message,
            fold_rounds: rounds,
            base: base_messages,
            base_rows,
            base_openings: openings,
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
    plain_rows: &[BatRow],
    messages: &ProximityMessages,
    fresh_id: u64,
    mask_tree: MaskTreePlan,
    base_positions: usize,
) -> Result<ProximityVerification> {
    if members.is_empty() || sample_rows.len() != members.len() {
        return Err(Error::ShapeMismatch);
    }
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
    for row in plain_rows {
        statement.add_row(group, row.clone())?;
    }
    let layout = statement.layout(group)?;
    let fold_layout = statement.to_fold_layout_checked(group, &fold)?;
    let planned = plain_rows.len() as u64
        + members.iter().map(|member| member.answers as u64).sum::<u64>();
    let mut bat = BatSession::new(statement, chain, planned)?;
    let input = bat.begin_run(group)?;
    let combined = bat.finish_run(input)?;
    let fold_row = bat.statement().to_fold_input_row(&combined, &layout)?;
    let chain = bat.into_chain()?;

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
    let mut verifier = BaseFinalSession::new(chain, handoff, basis)?;
    verifier.receive_commitments(messages.base)?;
    let lambda = verifier.draw_lambda()?;
    if messages.base_rows.base.lambda != lambda {
        return Err(Error::InvalidProof);
    }
    verifier.receive_base_rows(messages.base_rows.clone())?;
    verifier.draw_base_positions(base_positions)?;
    verifier.verify_openings(&messages.base_openings)?;
    let (seal, _base) = match verifier.finish()? {
        BaseFinalCheck::Passed(base) => (base.check_value, base),
        BaseFinalCheck::Rejected(_) => return Err(Error::InvalidProof),
    };
    Ok(ProximityVerification { seal, base: _base })
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

impl WCommitRandomness {
    /// Draws the complete `W` randomness from the existing numbered mask-coin
    /// ledger. The draw order is fixed: tails by position and row, one non-zero
    /// pad per answer, then one salt per tree column. The caller keeps the
    /// seed; the values are never serialized.
    pub fn from_coins(
        coins: &mut ByteMaskCoins,
        plan: &HiddenSendPlan,
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
        let randomness = Self {
            tails,
            salts,
            pads,
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
) -> Result<(HiddenSendMessages, Digest, HiddenSendWitness)> {
    plan.validate()?;
    let reader = table.reader().clone();
    check_activation(&plan.send, activation)?;
    check_table(&plan.send, &reader)?;
    let rows = derive_w_rows(&plan.send, activation, &reader, table)?;
    let (group, witness) = commit_w_rows(&plan.send, &rows, randomness)?;
    witness.check(&plan.send)?;

    let mut chain = first_chain(plan, &reader)?;
    absorb_group(&mut chain, &group)?;
    absorb_claim(&mut chain, plan)?;
    let seeds = draw_seeds(&mut chain, &plan.send)?;
    let answers = witness.answers(&seeds)?;
    absorb_answers(&mut chain, &answers)?;
    let positions = draw_positions(&mut chain, &plan.send)?;
    let mut openings = Vec::with_capacity(positions.len());
    for (position, list) in positions.iter().enumerate() {
        openings.push(witness.tree(position as u32)?.open(list)?);
    }
    let psi = draw_psi(&mut chain, &plan.send)?;
    let targets = eval_targets(plan, &rows, output, &psi)?;
    let expected = witness.eval_values(&psi)?;
    if expected != targets {
        return Err(Error::InvalidProof);
    }
    absorb_eval(&mut chain, &plan.send, &targets)?;
    let first_seal = *chain.value();
    let messages = HiddenSendMessages {
        activation: *activation.commitment(),
        group,
        samples: SendSamples { seed: seeds, answers },
        positions,
        openings,
        psi,
        eval_targets: targets,
        chain_value: first_seal,
    };
    Ok((messages, first_seal, witness))
}

fn verify_first(
    plan: &HiddenLinPlan,
    reader: &IndexReader,
    output: Option<&[Vec<Fp3>]>,
    messages: &HiddenSendMessages,
) -> Result<(Vec<EvalSampleRow>, Digest)> {
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
    let seeds = draw_seeds(&mut chain, &plan.send)?;
    if seeds != messages.samples.seed {
        return Err(Error::InvalidProof);
    }
    let answer_count = plan.send.positions() as usize * plan.send.w.answers as usize;
    if messages.samples.answers.len() != answer_count {
        return Err(Error::ShapeMismatch);
    }
    absorb_answers(&mut chain, &messages.samples.answers)?;
    let positions = draw_positions(&mut chain, &plan.send)?;
    if positions != messages.positions || messages.openings.len() != positions.len() {
        return Err(Error::InvalidProof);
    }
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
    } else {
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
    Ok((eval_rows, messages.chain_value))
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
        None => search_nonce(&prefix, 1u64 << 24)?,
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
    let k_col = shared_point_bits(plan)?;
    let mut chain = shared_start_chain(plan, reader, first_seal)?;
    let psi = draw_point_round(
        &mut chain,
        KIND_SEED,
        &[HIDDEN_SEND_SCHEMA_VERSION, 10],
        k_col,
    )?;
    let gamma = psi[plan.send.k_lo as usize..].to_vec();
    let (folded, folded_tail) = fold_table(table, reader.shape, &gamma)?;
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
    let mut gu_witness = gu_table(&gu, &folded, &folded_tail)?;
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
    let gu_tree = commit_rows_tree(gu.tree.clone(), gu.u.code, &gu_message, &gu_tails, &gu_salts)?;
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
    let gu_openings = table.open_columns(&unique)?;
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

    let omega = plan.proximity.omega.clone();
    let matrix_rows = omega.matrix_rows;
    let k_lo = omega.k_lo;
    if plan.send.k_lo != k_lo || plan.send.matrix_rows != matrix_rows {
        return Err(Error::StatementMismatch);
    }
    let low_count = 1u64 << k_lo;
    let mut omega_vector = vec![Fp3::ZERO; usize::try_from(matrix_rows).map_err(|_| Error::ResourceLimit)?];
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
            answers: gu.u.answers,
            root: gu_root,
            tree_plan: gu.tree.clone(),
            message: gu_message.clone(),
            tails: gu_tails.clone(),
        },
        ProximityMember {
            bat_oracle: 1,
            source_oracle: omega.oracle,
            main_code: omega.u.code,
            mask_code: plan.proximity.mask_code,
            main_rows: omega.u.main_rows,
            answers: omega.u.answers,
            root: omega_root,
            tree_plan: omega.tree.clone(),
            message: omega_message.clone(),
            tails: omega_tails.clone(),
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
    let gu_prover = ProximityMemberProver::sealed(members[gu_bat as usize].clone(), gu_tree)?;
    let omega_prover =
        ProximityMemberProver::sealed(members[omega_bat as usize].clone(), omega_tree)?;
    let mut provers = vec![gu_prover, omega_prover];
    provers.sort_by_key(|member| member.member.bat_oracle);
    let fold = member_fold_shape(
        plan.proximity.code,
        plan.proximity.mask_code,
        &members,
        plan.proximity.shared_mask_tree.leaf_class,
    )?;
    let proximity = proximity_prove(
        chain,
        fold,
        &provers,
        &sample_rows,
        &plain_rows,
        plan.proximity.shared_fresh,
        plan.proximity.shared_mask_tree,
        derive_run_seed(seed, 4, 0)?,
        plan.proximity.shared_base_positions,
    )?;
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
        base: proximity.messages.base,
        base_rows: proximity.messages.base_rows,
        base_openings: proximity.messages.base_openings,
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
    for (message, tail) in message_rows.iter().zip(tails) {
        codewords.push(encode_row_fp3(&code, message, tail)?);
    }
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
) -> Result<(Digest, Vec<GuRelationRow>, SharedPoints, BaseFinalOutcome)> {
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
    let column_map = table_column_map(reader, &positions, &messages.column_openings)?;
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
    let mut sample_rows = Vec::with_capacity(gu.u.answers as usize);
    for (answer, target) in messages.gu_answers.iter().enumerate() {
        sample_rows.push(sample_row(
            &gu,
            &registry,
            &gu_seeds,
            answer as u32,
            *target,
        )?);
    }
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
            answers: gu.u.answers,
            root: messages.gu_root,
            tree_plan: gu.tree.clone(),
            message: Vec::new(),
            tails: Vec::new(),
        },
        ProximityMember {
            bat_oracle: 1,
            source_oracle: omega.oracle,
            main_code: omega.u.code,
            mask_code: plan.proximity.mask_code,
            main_rows: omega.u.main_rows,
            answers: omega.u.answers,
            root: messages.omega_root,
            tree_plan: omega.tree.clone(),
            message: Vec::new(),
            tails: Vec::new(),
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
    let fold = member_fold_shape(
        plan.proximity.code,
        plan.proximity.mask_code,
        &members,
        plan.proximity.shared_mask_tree.leaf_class,
    )?;
    let proximity = proximity_verify(
        chain,
        fold,
        &members,
        &sample_rows,
        &plain_rows,
        &ProximityMessages {
            fold_start: messages.fold_start.clone(),
            fold_rounds: messages.fold_rounds.clone(),
            base: messages.base,
            base_rows: messages.base_rows.clone(),
            base_openings: messages.base_openings.clone(),
        },
        plan.proximity.shared_fresh,
        plan.proximity.shared_mask_tree,
        plan.proximity.shared_base_positions,
    )?;
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
            answers: plan.send.w.answers,
            root,
            tree_plan: plan.send.w.tree_plan(position)?,
            message,
            tails,
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
        answers: omega.u.answers,
        root: shared.omega_root,
        tree_plan: omega.tree.clone(),
        message,
        tails,
    });
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
    let mut rows = Vec::with_capacity(members.len());
    for member in members {
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
            rows.push(list);
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
            rows.push(vec![oracle_sample_row(
                member.source_oracle,
                plan.send.group,
                &shape,
                &first.samples.seed,
                0,
                target,
            )?]);
        }
    }
    Ok(rows)
}

fn own_plain_rows(
    plan: &HiddenLinPlan,
    members: &[ProximityMember],
    first: &HiddenSendMessages,
    shared_point: &[Fp3],
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
    for position in 0..plan.send.positions() {
        let member = bat_of
            .get(&plan.send.oracle(position)?)
            .ok_or(Error::StatementMismatch)?;
        let mut coefficients = vec![Fp3::ZERO; w_rows * w_ell];
        for column in 0..plan.send.columns {
            let coefficient = eq_index(
                &first.psi[position as usize],
                column,
                plan.send.columns.trailing_zeros(),
            )?;
            let slot = coefficients
                .get_mut(usize::try_from(column).map_err(|_| Error::ResourceLimit)?)
                .ok_or(Error::ShapeMismatch)?;
            *slot = coefficient;
        }
        let mut row = BatRow::new(BatRowKind::Plain);
        row.add_term(member.bat_oracle, BatPart::Main, coefficients);
        row.target = first.eval_targets[position as usize];
        rows.push(row);
    }
    // The Dot row is linear in the reduced row with the activation row as its
    // public coefficients. In the hidden-input mode the verifier holds no
    // activation row, so there is no public coefficient source and the Dot
    // row is not part of this session; the diagnostic mode keeps it. See the
    // module documentation for the open relation this leaves.
    if !plan.weights.is_empty() {
        let omega = bat_of
            .get(&plan.proximity.omega.oracle)
            .ok_or(Error::StatementMismatch)?;
        let omega_len = omega_rows * omega_ell;
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
            for column in 0..plan.send.columns {
                let coefficient = eq_index(
                    shared_point,
                    column,
                    plan.send.columns.trailing_zeros(),
                )?;
                let slot = w_coefficients
                    .get_mut(usize::try_from(column).map_err(|_| Error::ResourceLimit)?)
                    .ok_or(Error::ShapeMismatch)?;
                *slot = Fp3::ZERO.sub(&coefficient);
            }
            let mut row = BatRow::new(BatRowKind::Plain);
            row.add_term(omega.bat_oracle, BatPart::Main, omega_coefficients);
            row.add_term(w_member.bat_oracle, BatPart::Main, w_coefficients);
            rows.push(row);
        }
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
) -> Result<OwnProveResult> {
    let (members, order) = own_members(
        plan,
        first,
        shared,
        Some(witness),
        Some((&shared_witness.omega_message, &shared_witness.omega_tails)),
    )?;
    let sample_rows = own_sample_rows(plan, &members, first, shared)?;
    let plain_rows = own_plain_rows(plan, &members, first, shared_point)?;
    let chain = second_chain(
        plan,
        reader,
        first_query,
        &shared.seal,
        &shared.omega_root,
        shared_point,
        &shared.zeta,
    )?;
    let mut provers = Vec::with_capacity(members.len());
    for member in &members {
        let tree = witness_tree_for(
            member,
            witness,
            shared_witness,
            first,
            plan,
        )?;
        provers.push(ProximityMemberProver::sealed(member.clone(), tree)?);
    }
    provers.sort_by_key(|member| member.member.bat_oracle);
    let fold = member_fold_shape(
        plan.proximity.code,
        plan.proximity.mask_code,
        &members,
        plan.proximity.own_mask_tree.leaf_class,
    )?;
    let proximity = proximity_prove(
        chain,
        fold,
        &provers,
        &sample_rows,
        &plain_rows,
        plan.proximity.own_fresh,
        plan.proximity.own_mask_tree,
        derive_run_seed(seed, 5, 0)?,
        plan.proximity.own_base_positions,
    )?;
    let mut claims = sample_rows.into_iter().flatten().collect::<Vec<_>>();
    let _ = order;
    Ok(OwnProveResult {
        messages: HiddenSecondMessages {
            omega_root: shared.omega_root,
            first_query: *first_query,
            shared_seal: shared.seal,
            fold_start: proximity.messages.fold_start.clone(),
            fold_rounds: proximity.messages.fold_rounds.clone(),
            base: proximity.messages.base,
            base_rows: proximity.messages.base_rows,
            base_openings: proximity.messages.base_openings,
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
) -> Result<RuntimeTree> {
    if member.source_oracle == plan.proximity.omega.oracle {
        commit_rows_tree(
            member.tree_plan.clone(),
            member.main_code,
            &member.message,
            &member.tails,
            &omega_salts_of(shared_witness)?,
        )
    } else {
        let position = (member.source_oracle - plan.send.oracle_base) as usize;
        if position >= plan.send.positions() as usize {
            return Err(Error::StatementMismatch);
        }
        let _ = first;
        let tails_len = usize::try_from(plan.send.w.code.r).map_err(|_| Error::ResourceLimit)?;
        let rows = usize::try_from(plan.send.w.main_rows).map_err(|_| Error::ResourceLimit)?;
        let tail = witness.tails.get(position).ok_or(Error::ShapeMismatch)?;
        let mut tails = Vec::with_capacity(rows);
        for row in 0..rows {
            tails.push(
                tail.get(row * tails_len..(row + 1) * tails_len)
                    .ok_or(Error::ShapeMismatch)?
                    .to_vec(),
            );
        }
        let salts = witness
            .salts
            .get(position)
            .ok_or(Error::ShapeMismatch)?
            .clone();
        commit_rows_tree(
            member.tree_plan.clone(),
            member.main_code,
            &member.message,
            &tails,
            &salts,
        )
    }
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
) -> Result<(Digest, Vec<GuRelationRow>, BaseFinalOutcome)> {
    if messages.omega_root != shared.omega_root
        || messages.first_query != *first_query
        || messages.shared_seal != shared.seal
    {
        return Err(Error::StatementMismatch);
    }
    let (members, _order) = own_members(plan, first, shared, None, None)?;
    let sample_rows = own_sample_rows(plan, &members, first, shared)?;
    let plain_rows = own_plain_rows(plan, &members, first, shared_point)?;
    let chain = second_chain(
        plan,
        reader,
        first_query,
        &shared.seal,
        &shared.omega_root,
        shared_point,
        &shared.zeta,
    )?;
    let fold = member_fold_shape(
        plan.proximity.code,
        plan.proximity.mask_code,
        &members,
        plan.proximity.own_mask_tree.leaf_class,
    )?;
    let proximity = proximity_verify(
        chain,
        fold,
        &members,
        &sample_rows,
        &plain_rows,
        &ProximityMessages {
            fold_start: messages.fold_start.clone(),
            fold_rounds: messages.fold_rounds.clone(),
            base: messages.base,
            base_rows: messages.base_rows.clone(),
            base_openings: messages.base_openings.clone(),
        },
        plan.proximity.own_fresh,
        plan.proximity.own_mask_tree,
        plan.proximity.own_base_positions,
    )?;
    if proximity.seal != messages.seal {
        return Err(Error::InvalidProof);
    }
    Ok((
        proximity.seal,
        sample_rows.into_iter().flatten().collect(),
        proximity.base,
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
) -> Result<(HiddenLinMessages, HiddenLinWitness)> {
    plan.validate()?;
    if seed == Digest::ZERO {
        return Err(Error::InvalidConfig);
    }
    let reader = table.reader().clone();
    let mut coins = ByteMaskCoins::new(derive_run_seed(&seed, 1, 0)?)?;
    let randomness = WCommitRandomness::from_coins(&mut coins, &plan.send)?;
    let (first, first_seal, send_witness) =
        prove_first(plan, activation, table, output, &randomness)?;
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
    prove_hidden_lin_inner(plan, activation, table, Some(output), seed)
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
    if !plan.weights.is_empty() {
        return Err(Error::InvalidConfig);
    }
    prove_hidden_lin_inner(plan, activation, table, None, seed)
}

fn verify_hidden_lin_inner(
    plan: &HiddenLinPlan,
    reader: &IndexReader,
    output: Option<&[Vec<Fp3>]>,
    messages: &HiddenLinMessages,
) -> Result<HiddenLinOutcome> {
    plan.validate()?;
    let (eval_rows, first_seal) = verify_first(plan, reader, output, &messages.first)?;
    if first_seal != messages.first_seal || messages.first.chain_value != first_seal {
        return Err(Error::InvalidProof);
    }
    let first_query = *first_chain(plan, reader)?.value();
    let (shared_seal, shared_claims, points, shared_base) =
        verify_shared(plan, reader, &first_seal, &messages.shared)?;
    if shared_seal != messages.shared.seal {
        return Err(Error::InvalidProof);
    }
    let (seal, own_claims, own_base) = verify_second(
        plan,
        reader,
        &messages.first,
        &first_query,
        &messages.shared,
        &points.psi,
        &messages.second,
    )?;
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
    witness.send.check(&plan.send)?;
    let outcome = verify_hidden_lin(plan, reader, output, messages)?;
    let expected = witness.send.answers(&messages.first.samples.seed)?;
    if expected != messages.first.samples.answers {
        return Err(Error::InvalidProof);
    }
    let eval = witness.send.eval_values(&messages.first.psi)?;
    if eval != messages.first.eval_targets {
        return Err(Error::InvalidProof);
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
    Ok(outcome)
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

fn put_runtime_opening(writer: &mut CanonicalWriter, opening: &RuntimeTreeOpening) {
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

fn get_runtime_opening(
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

fn put_fold_start(writer: &mut CanonicalWriter, message: &FoldStartMessage) {
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

fn get_fold_start(reader: &mut CanonicalReader<'_>) -> Result<FoldStartMessage> {
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

fn get_code_public(reader: &mut CanonicalReader<'_>) -> Result<CodeParams> {
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
    put_base_messages(writer, &message.base);
    put_base_rows(writer, &message.base_rows);
    put_base_openings(writer, &message.base_openings);
    writer.put_digest(&message.seal);
}

fn get_shared(reader: &mut CanonicalReader<'_>) -> Result<HiddenSharedMessages> {
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
    let base = get_base_messages(reader)?;
    let base_rows = get_base_rows(reader, ACTIVATION_DECODE_BOUND)?;
    let base_openings = get_base_openings(reader, ACTIVATION_DECODE_BOUND)?;
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
        base,
        base_rows,
        base_openings,
        seal,
    })
}

fn put_second(writer: &mut CanonicalWriter, message: &HiddenSecondMessages) {
    writer.put_digest(&message.omega_root);
    writer.put_digest(&message.first_query);
    writer.put_digest(&message.shared_seal);
    put_fold_start(writer, &message.fold_start);
    writer.put_u64(message.fold_rounds.len() as u64);
    for (c0, c2) in &message.fold_rounds {
        writer.put_fp3(c0);
        writer.put_fp3(c2);
    }
    put_base_messages(writer, &message.base);
    put_base_rows(writer, &message.base_rows);
    put_base_openings(writer, &message.base_openings);
    writer.put_digest(&message.seal);
}

fn get_second(reader: &mut CanonicalReader<'_>) -> Result<HiddenSecondMessages> {
    let omega_root = reader.get_digest()?;
    let first_query = reader.get_digest()?;
    let shared_seal = reader.get_digest()?;
    let fold_start = get_fold_start(reader)?;
    let round_count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
    if round_count > 64 {
        return Err(Error::ResourceLimit);
    }
    let mut fold_rounds = Vec::with_capacity(round_count);
    for _ in 0..round_count {
        fold_rounds.push((reader.get_fp3()?, reader.get_fp3()?));
    }
    let base = get_base_messages(reader)?;
    let base_rows = get_base_rows(reader, ACTIVATION_DECODE_BOUND)?;
    let base_openings = get_base_openings(reader, ACTIVATION_DECODE_BOUND)?;
    let seal = reader.get_digest()?;
    Ok(HiddenSecondMessages {
        omega_root,
        first_query,
        shared_seal,
        fold_start,
        fold_rounds,
        base,
        base_rows,
        base_openings,
        seal,
    })
}

impl HiddenLinMessages {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(HIDDEN_SEND_SCHEMA_VERSION);
        writer.put_u64(12);
        writer.put_bytes(&self.first.encode()?);
        writer.put_digest(&self.first_seal);
        let mut shared = CanonicalWriter::new();
        put_shared(&mut shared, &self.shared);
        writer.put_bytes(&shared.into_bytes());
        let mut second = CanonicalWriter::new();
        put_second(&mut second, &self.second);
        writer.put_bytes(&second.into_bytes());
        Ok(writer.into_bytes())
    }

    pub fn decode(plan: &HiddenLinPlan, bytes: &[u8]) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        if reader.get_u64()? != HIDDEN_SEND_SCHEMA_VERSION || reader.get_u64()? != 12 {
            return Err(Error::InvalidEncoding);
        }
        let first = HiddenSendMessages::decode(&plan.send, reader.get_bytes()?)?;
        let first_seal = reader.get_digest()?;
        let shared_bytes = reader.get_bytes()?.to_vec();
        let second_bytes = reader.get_bytes()?.to_vec();
        reader.finish()?;
        let mut shared_reader = CanonicalReader::new(&shared_bytes, DecodeLimits::unlimited());
        let shared = get_shared(&mut shared_reader)?;
        shared_reader.finish()?;
        let mut second_reader = CanonicalReader::new(&second_bytes, DecodeLimits::unlimited());
        let second = get_second(&mut second_reader)?;
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
    use zkie_engine::pcs::request::field::Fp;
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
    const W_CODE_R: u64 = 1;
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
            pads: (0..POSITIONS as usize)
                .map(|position| vec![Fp3::from_u64(0x900 + position as u64)])
                .collect(),
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
        let decoded = HiddenSendMessages::decode(&fixture.verifier_plan, &bytes).unwrap();
        assert_eq!(decoded, fixture.messages);
        verify_hidden_send(&fixture.verifier_plan, &decoded, &fixture.witness).unwrap();

        // A tampered byte is rejected by decoding or by verification.
        let mut tampered = bytes.clone();
        let at = tampered.len() / 2;
        tampered[at] ^= 1;
        let accepted = HiddenSendMessages::decode(&fixture.verifier_plan, &tampered)
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
    const TABLE_R: u64 = 1;
    const CODE_LOG_N: u32 = 5;
    const CODE_ELL: u64 = 16;
    const CODE_R: u64 = 1;
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
            expected_roots: None,
        };
        let weights: Vec<Vec<Fp3>> = (0..POSITIONS)
            .map(|position| activation.weights(position, MATRIX_ROWS).unwrap())
            .collect();
        HiddenLinPlan {
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
            },
            output_relation: None,
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
        assert!(outcome.shared_base.fold_main);
        assert!(outcome.own_base.fold_main);
        assert!(outcome.shared_base.base_linear);
        assert!(outcome.own_base.base_linear);
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
    fn hidden_input_mode_accepts_without_output_rows() {
        let activation = activation();
        let diagnostic_plan = plan(&activation);
        let mut plan = diagnostic_plan.clone();
        plan.weights = Vec::new();
        let mut table = table();
        let (messages, witness) =
            prove_hidden_lin_hidden(&plan, &activation, &mut table, Digest([0x71; 32])).unwrap();
        // The Eval targets are the evaluations of the committed reduced rows.
        let expected = witness.send.eval_values(&messages.first.psi).unwrap();
        assert_eq!(expected, messages.first.eval_targets);
        let outcome = verify_hidden_lin_hidden(&plan, &table.reader, &messages).unwrap();
        assert_eq!(outcome.eval_rows.len(), POSITIONS as usize);
        assert!(outcome.own_base.fold_main);
        assert!(outcome.own_base.base_linear);
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
        messages.second.base_rows.base.lambda =
            messages.second.base_rows.base.lambda.add(&Fp3::ONE);
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
}
