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

use zkie_core::common::codec::DecodeLimits;
use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use zkie_engine::compute::cpu::{merkle_build, merkle_verify};
use zkie_engine::pcs::request::chain::{
    Chain, ChainMessage, KIND_ALGEBRAIC, KIND_EMPTY, KIND_POSITIONS, KIND_SEED,
};
use zkie_engine::pcs::request::code::{build_code_plan, encode_row_fp3, CodeParams};
use zkie_engine::pcs::request::field::{Fp, Fp3, E_DEGREE};
use zkie_engine::pcs::request::foldeq::reader_table_identity;
use zkie_engine::pcs::request::gu::GuOracleShape;
use zkie_engine::pcs::request::params::{LEAF_SALT_MAX_BYTES, LEAF_SALT_MIN_BYTES};
use zkie_engine::pcs::request::read::unique_sorted;
use zkie_engine::pcs::request::relation::{
    oracle_sample_row, sample_cube_of, GuRelationRow, GuWitness,
};
use zkie_engine::pcs::request::runtime_tree::{
    decode_runtime_opening, encode_runtime_opening, verify_runtime_opening, RuntimePartRows,
    RuntimeTree, RuntimeTreeOpening, RuntimeTreePart, RuntimeTreePlan,
};
use zkie_engine::pcs::request::table::{
    validate_index_shape, IndexReader, IndexShape, SampleSeed, TableRowSource,
};
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};
use zkie_engine::primitives::transcript::{
    hash_labeled, tag_word, LEAF_MAIN, PARAMETERS,
};

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
