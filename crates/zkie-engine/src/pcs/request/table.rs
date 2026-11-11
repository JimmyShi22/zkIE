//! Committed table: shape, encoded columns, record and column openings.
//!
//! A table has `2^v` rows, each a codeword of length `2^log_n` over the base
//! field. The tree runs over the columns: a leaf holds one salt and one symbol
//! per row. The record stores the committer's answers to out-of-domain samples
//! of each row's message. The table's name binds its root, the configuration
//! digest and the code/row/answer/salt numbers; the sampling distances are a
//! plan value and are deliberately absent from the name (they are supplied by
//! the trusted plan, not by the table).

use serde::{Deserialize, Serialize};
use zkie_core::common::codec::DecodeLimits;
use zkie_core::common::field::PrimeCharacteristicRing;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::compute::cpu::{merkle_build, merkle_verify, MerkleTree};
use crate::pcs::request::chain::{Chain, KIND_SEED};
use crate::pcs::request::code::{encode_row_fp, CodeParams};
use crate::pcs::request::field::{E_DEGREE, Fp, Fp3, FP3_BYTES};
use crate::pcs::request::params::{LEAF_SALT_MAX_BYTES, LEAF_SALT_MIN_BYTES};
use crate::pcs::request::read::{HidingCharge, HidingState};
use crate::pcs::request::relation::foldeq_weights;
use crate::primitives::codec::{CanonicalReader, CanonicalWriter};
use crate::primitives::transcript::{hash_labeled, tag_word, LEAF_STORED_MAIN};

/// The label of the chain that derives a table's record seeds.
pub const RECORD_SEED_LABEL: u64 = 1 << 32;

/// The sampling distances a plan fixes for a table. They are not part of the
/// table's name or of any first query; the trusted plan supplies them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableDistances {
    pub radius: u64,
    pub later_radius: u64,
    pub m_once: u64,
    pub m_later: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexShape {
    pub code: CodeParams,
    pub v: u32,
    pub answers: u32,
    pub salt_bytes: usize,
    pub distances: TableDistances,
}

impl IndexShape {
    /// The number of rows. The shape must have been validated first; a `v`
    /// that does not fit a `u64` shift is rejected by `rows_checked`.
    pub fn rows(&self) -> u64 {
        1u64 << self.v
    }

    /// The number of rows with the shift checked. Public entries call this
    /// before any accessor that shifts by `v`.
    pub fn rows_checked(&self) -> Result<u64> {
        if self.v >= 64 {
            return Err(Error::InvalidConfig);
        }
        Ok(1u64 << self.v)
    }

    pub fn pads(&self) -> usize {
        E_DEGREE * self.answers as usize
    }

    pub fn plain(&self) -> u64 {
        self.code.ell - self.pads() as u64
    }

    pub fn pad_place(&self, answer: u32, coefficient: usize) -> u64 {
        self.plain() + (E_DEGREE * answer as usize + coefficient) as u64
    }

    pub fn cube_bits(&self) -> u32 {
        self.code.log_ell()
    }
}

/// What is wrong with a table shape, judged by itself.
pub fn index_fault(shape: &IndexShape) -> Option<&'static str> {
    if crate::pcs::request::code::build_code_plan(
        shape.code.log_n,
        shape.code.ell,
        shape.code.r,
    )
    .is_err()
    {
        return Some("the table's numbers are not a code");
    }
    if shape.v > 40 {
        return Some("the table has too many rows");
    }
    if shape.answers < 1 {
        return Some("the table's number of answers is below 1");
    }
    if shape.pads() as u64 >= shape.code.ell {
        return Some("a row has no room beside its pads");
    }
    if shape.salt_bytes < LEAF_SALT_MIN_BYTES || shape.salt_bytes > LEAF_SALT_MAX_BYTES {
        return Some("the leaves of the table's tree have no admitted salt");
    }
    None
}

/// The tree-shape checks that index_fault() does not cover: the row count
/// fits a shift, the salt is a multiple of eight bytes, and a column (one
/// symbol per row) is narrower than 2^24. The salt-length range remains
/// index_fault()'s check.
pub fn tree_shape_fault(shape: &IndexShape) -> Option<&'static str> {
    if shape.v >= 64 {
        return Some("the table's row count does not fit a shift");
    }
    if shape.salt_bytes % 8 != 0 {
        return Some("a leaf salt is not a multiple of eight bytes");
    }
    if shape.rows() >> 24 != 0 {
        return Some("a column is wider than the tree tag admits");
    }
    None
}

pub fn validate_index_shape(shape: &IndexShape) -> Result<()> {
    if index_fault(shape).is_some() || tree_shape_fault(shape).is_some() {
        return Err(Error::InvalidConfig);
    }
    Ok(())
}

/// The committer's answers to out-of-domain samples of each row, by row and
/// then by answer. The seeds are derived from the table's name and are not
/// stored.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexRecord {
    pub y: Vec<Fp3>,
}

impl IndexRecord {
    pub fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        for value in &self.y {
            writer.put_fp3(value);
        }
        writer.into_bytes()
    }

    /// Decode at most `max_elements` answers; the caller passes a bound
    /// derived from the verified shape.
    pub fn decode(bytes: &[u8], max_elements: usize) -> Result<Self> {
        if bytes.len() % FP3_BYTES != 0 {
            return Err(Error::InvalidEncoding);
        }
        let count = bytes.len() / FP3_BYTES;
        if count > max_elements {
            return Err(Error::ResourceLimit);
        }
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        let mut y = Vec::with_capacity(count);
        while !reader.remaining().is_empty() {
            y.push(reader.get_fp3()?);
        }
        reader.finish()?;
        Ok(Self { y })
    }

    /// Decode a record for a shape: the shape is validated first, then the
    /// length must equal rows * answers, with checked arithmetic.
    pub fn decode_for(shape: &IndexShape, bytes: &[u8]) -> Result<Self> {
        validate_index_shape(shape)?;
        let expected = shape
            .rows_checked()?
            .checked_mul(shape.answers as u64)
            .ok_or(Error::ResourceLimit)?;
        let expected = usize::try_from(expected).map_err(|_| Error::ResourceLimit)?;
        let record = Self::decode(bytes, expected)?;
        if record_length_fault(shape, &record).is_some() {
            return Err(Error::ShapeMismatch);
        }
        Ok(record)
    }
}

/// The record-length check against a shape that is already validated. The
/// public entry validates the shape before calling this.
fn record_length_fault(shape: &IndexShape, record: &IndexRecord) -> Option<&'static str> {
    let rows = match shape.rows_checked() {
        Ok(rows) => rows,
        Err(_) => return Some("the table's row count does not fit"),
    };
    let expected = rows.checked_mul(shape.answers as u64);
    match expected {
        Some(expected) if record.y.len() as u64 == expected => None,
        Some(_) => Some("the record has not one answer for each row and seed"),
        None => Some("the record length overflows the verified shape"),
    }
}

/// What is wrong with the shape or the record. The shape is validated first;
/// a `v` that does not fit a shift is a fault and is never shifted.
pub fn record_fault(shape: &IndexShape, record: &IndexRecord) -> Option<&'static str> {
    if index_fault(shape).is_some() {
        return Some("the shape is not a table shape");
    }
    if tree_shape_fault(shape).is_some() {
        return Some("the shape is not a tree shape");
    }
    record_length_fault(shape, record)
}

/// A table's statement item: its commitment bytes and record answers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexName {
    pub commitment: Vec<u8>,
    pub record: IndexRecord,
}

impl IndexName {
    /// The bytes a first query binds: the commitment (length-prefixed) and
    /// the complete record, before any challenge that reads it.
    pub fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_bytes(&self.commitment);
        for value in &self.record.y {
            writer.put_fp3(value);
        }
        writer.into_bytes()
    }
}

/// One out-of-domain sample: a point with one extension coordinate per message
/// factor and a non-zero factor on the answer's pad.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampleSeed {
    pub z: Vec<Fp3>,
    pub beta: Fp3,
}

fn first_query_bytes(shape: &IndexShape, configuration: &Digest, root: &Digest) -> Vec<u8> {
    let mut writer = CanonicalWriter::new();
    writer.put_digest(root);
    writer.put_digest(configuration);
    for value in [
        shape.code.log_n as u64,
        shape.code.ell,
        shape.code.r,
        shape.v as u64,
        shape.answers as u64,
        shape.salt_bytes as u64,
    ] {
        writer.put_u64(value);
    }
    writer.into_bytes()
}

/// The table's name: the bytes a first query binds. The distances are absent
/// by design: the plan supplies them.
pub fn index_name(root: &Digest, shape: &IndexShape, configuration: &Digest) -> Vec<u8> {
    first_query_bytes(shape, configuration, root)
}

fn record_eq(index: u64, z: &[Fp3], bits: u32) -> Fp3 {
    let mut value = Fp3::ONE;
    for t in 0..bits {
        let coordinate = &z[t as usize];
        let factor = if (index >> t) & 1 == 1 {
            *coordinate
        } else {
            Fp3::ONE.sub(coordinate)
        };
        value = value.mul(&factor);
    }
    value
}

fn record_mle(message: &[Fp], z: &[Fp3], bits: u32) -> Fp3 {
    let mut sum = Fp3::ZERO;
    for (index, value) in message.iter().enumerate() {
        let term = record_eq(index as u64, z, bits).mul(&Fp3::from_base(*value));
        sum = sum.add(&term);
    }
    sum
}

fn theta(coefficient: usize) -> Fp3 {
    match coefficient {
        0 => Fp3::ONE,
        1 => Fp3::new(Fp::ZERO, Fp::ONE, Fp::ZERO),
        2 => Fp3::new(Fp::ZERO, Fp::ZERO, Fp::ONE),
        _ => Fp3::ZERO,
    }
}

/// One answer of the record: the multilinear extension of the message before
/// the pads at the seed's point, plus the seed's factor times the answer's pad.
fn record_value(shape: &IndexShape, seed: &SampleSeed, answer: u32, message: &[Fp]) -> Result<Fp3> {
    let bits = shape.cube_bits();
    let mut sum = record_mle(message, &seed.z, bits);
    for position in shape.plain()..shape.code.ell {
        let eq = record_eq(position, &seed.z, bits);
        sum = sum.sub(&eq.mul(&Fp3::from_base(message[position as usize])));
    }
    for coefficient in 0..E_DEGREE {
        let pad = Fp3::from_base(message[shape.pad_place(answer, coefficient) as usize]);
        let term = seed
            .beta
            .mul(&theta(coefficient))
            .mul(&pad);
        sum = sum.add(&term);
    }
    Ok(sum)
}

/// The seeds of a table's record, from its name. The label and the empty seed
/// round follow the chain layout; the seed values are drawn from the
/// repository's chain rules.
///
/// The shape is validated first. `max_seeds` is the caller's trusted budget on
/// the number of seeds (the plan or priced table supplies it); a shape that
/// asks for more is rejected before the vector is reserved.
pub fn record_seeds(
    shape: &IndexShape,
    name: &[u8],
    max_seeds: usize,
) -> Result<Vec<SampleSeed>> {
    validate_index_shape(shape)?;
    let answers = usize::try_from(shape.answers).map_err(|_| Error::ResourceLimit)?;
    if answers > max_seeds {
        return Err(Error::ResourceLimit);
    }
    let mut chain = Chain::begin(
        RECORD_SEED_LABEL,
        &Digest::ZERO,
        name,
        &[],
        None,
    )?;
    let prefix = chain.message().prefix(KIND_SEED, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    let mut seeds = Vec::new();
    seeds
        .try_reserve_exact(answers)
        .map_err(|_| Error::ResourceLimit)?;
    for _ in 0..answers {
        let mut z = Vec::with_capacity(shape.cube_bits() as usize);
        for _ in 0..shape.cube_bits() {
            z.push(chain.draw_e()?);
        }
        let beta = chain.draw_e_nonzero()?;
        seeds.push(SampleSeed { z, beta });
    }
    Ok(seeds)
}

/// What a verifier holds of a committed table.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexReader {
    pub shape: IndexShape,
    pub configuration: Digest,
    pub root: Digest,
    pub record: IndexRecord,
}

impl IndexReader {
    pub fn commitment(&self) -> Vec<u8> {
        self.root.0.to_vec()
    }

    pub fn name(&self) -> Vec<u8> {
        index_name(&self.root, &self.shape, &self.configuration)
    }

    pub fn fault(&self) -> Option<&'static str> {
        if let Some(fault) = index_fault(&self.shape) {
            return Some(fault);
        }
        if let Some(fault) = tree_shape_fault(&self.shape) {
            return Some(fault);
        }
        if self.configuration == Digest::ZERO {
            return Some("the key names no configuration");
        }
        record_length_fault(&self.shape, &self.record)
    }
}

/// One opening of a column: the drawn position, its salt, one symbol per row
/// and the Merkle path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnOpening {
    pub position: u64,
    pub salt: Vec<u8>,
    pub symbols: Vec<Fp>,
    pub path: Vec<Digest>,
}

/// The leaf of one stored column.
///
/// The leaf index is `2^log_n + position` (so a leaf index is at least
/// `2^log_n` and a bare position cannot name a leaf); the tree is a perfect
/// binary tree with exactly `2^log_n` leaves and no padding.
///
/// The leaf data, in order: the column width and the salt length as two
/// `u64` little-endian words, one canonical base-field symbol per row as
/// eight-byte little-endian words below the modulus, then the salt as a
/// length-prefixed byte string (a `u64` length followed by the bytes). The
/// salt length therefore appears twice; a parser admits the leaf only when
/// both equal the trusted shape's salt length, the symbol count equals the
/// trusted shape's row count, and no bytes follow the salt.
///
/// The tag is `tag_word(LEAF_STORED_MAIN, 0)`: the stored-leaf domain with
/// tree number 0. The configuration digest is the hash start, and it must be
/// nonzero. The hash is the protocol's labelled 32-byte BLAKE2b: a sixteen
/// byte label (tag and index, both little-endian), and, because the stored
/// leaf domain takes a start, the start's first sixteen bytes as BLAKE2b's
/// salt and its next sixteen as personalization.
///
/// Domain separation: a stored leaf uses domain 174 and index
/// `2^log_n + position`; a tree node uses domain 166 and index 0 with the two
/// child digests as data; binding leaves, binding nodes and chain rounds use
/// their own domains and index spaces. Two different objects therefore never
/// share a domain, an index and a data layout, so a digest of one object
/// cannot be read as another without a hash collision. This is an integrity
/// argument: the salt and the configuration digest are public binding values,
/// not secret keys; collision resistance is the assumption.
fn column_leaf(
    configuration: &Digest,
    log_n: u32,
    position: u64,
    salt: &[u8],
    symbols: &[Fp],
) -> Result<Digest> {
    let mut writer = CanonicalWriter::new();
    writer.put_u64(symbols.len() as u64);
    writer.put_u64(salt.len() as u64);
    for value in symbols {
        writer.put_fp(*value);
    }
    writer.put_bytes(salt);
    let index = (1u64 << log_n)
        .checked_add(position)
        .ok_or(Error::ResourceLimit)?;
    hash_labeled(
        tag_word(LEAF_STORED_MAIN, 0),
        index,
        Some(configuration),
        &writer.into_bytes(),
    )
}

/// A bounded source of a table's row messages and random tails. The models
/// layer maps the read-only asset into these rows and the committer reads one
/// row at a time; no caller hands over the whole table.
pub trait TableRowSource {
    fn rows(&self) -> u64;
    fn row(&mut self, row: u64, message: &mut [Fp], tail: &mut [Fp]) -> Result<()>;
}

/// A table committed by bounded column panels. It keeps the column leaves and
/// the Merkle tree, not the codewords; an opening recomputes the drawn columns
/// from the row source and checks them against this tree.
pub struct StreamingTable {
    pub reader: IndexReader,
    pub name: Vec<u8>,
    tree: MerkleTree,
    salts: Vec<Vec<u8>>,
    leaves: Vec<Digest>,
    peak_bytes: u64,
}

impl StreamingTable {
    pub fn peak_bytes(&self) -> u64 {
        self.peak_bytes
    }

    /// The stored column leaves, in position order.
    pub fn leaves(&self) -> &[Digest] {
        &self.leaves
    }

    /// The private salts of the stored columns, in position order.
    pub fn salts(&self) -> &[Vec<u8>] {
        &self.salts
    }

    /// Rebuilds a committed table from its persisted public reader, name,
    /// leaves, salts and peak byte count. The leaves are re-hashed into the
    /// tree and the rebuilt root must equal the reader's root; no codeword is
    /// recomputed and no asset row is read.
    pub fn rebuild(
        reader: IndexReader,
        name: Vec<u8>,
        leaves: Vec<Digest>,
        salts: Vec<Vec<u8>>,
        peak_bytes: u64,
    ) -> Result<Self> {
        if reader.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        let expected = 1usize
            .checked_shl(reader.shape.code.log_n)
            .ok_or(Error::ResourceLimit)?;
        if leaves.len() != expected || salts.len() != expected {
            return Err(Error::ShapeMismatch);
        }
        let tree = merkle_build(&reader.configuration, &leaves)?;
        if tree.root != reader.root {
            return Err(Error::InvalidProof);
        }
        let derived_name = index_name(&tree.root, &reader.shape, &reader.configuration);
        if derived_name != name {
            return Err(Error::StatementMismatch);
        }
        Ok(Self {
            reader,
            name,
            tree,
            salts,
            leaves,
            peak_bytes,
        })
    }

    /// Recomputes the drawn columns from the row source and opens them against
    /// the stored tree. One row's message, one tail and one codeword are held
    /// at a time; no column is stored.
    pub fn open_columns<S: TableRowSource>(
        &self,
        source: &mut S,
        positions: &[u64],
    ) -> Result<Vec<ColumnOpening>> {
        if self.reader.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        let shape = self.reader.shape;
        if source.rows() != shape.rows() {
            return Err(Error::ShapeMismatch);
        }
        let n = shape.code.n();
        let rows = shape.rows() as usize;
        let ell = usize::try_from(shape.code.ell).map_err(|_| Error::ResourceLimit)?;
        let tail_len = usize::try_from(shape.code.r).map_err(|_| Error::ResourceLimit)?;
        let mut message = vec![Fp::ZERO; ell];
        let mut tail = vec![Fp::ZERO; tail_len];
        let mut symbols = vec![Fp::ZERO; positions.len() * rows];
        for row in 0..rows {
            source.row(row as u64, &mut message, &mut tail)?;
            let codeword = encode_row_fp(&shape.code, &message, &tail)?;
            for (index, position) in positions.iter().enumerate() {
                symbols[index * rows + row] = codeword[*position as usize];
            }
        }
        let mut openings = Vec::with_capacity(positions.len());
        let mut previous = None;
        for (index, position) in positions.iter().enumerate() {
            if *position >= n {
                return Err(Error::InvalidIndex);
            }
            if let Some(previous) = previous {
                if *position <= previous {
                    return Err(Error::InvalidIndex);
                }
            }
            previous = Some(*position);
            openings.push(ColumnOpening {
                position: *position,
                salt: self.salts[*position as usize].clone(),
                symbols: symbols[index * rows..(index + 1) * rows].to_vec(),
                path: self.tree.open(*position as usize)?,
            });
        }
        Ok(openings)
    }
}

/// Commits a table in bounded column panels. The source is read once for the
/// leaves and once for the record; the peak buffers are one panel of symbols,
/// one codeword, one message, one tail and the column leaves, bounded by
/// `max_bytes`. The root, name, record and openings are the same as the
/// in-memory test committer's.
pub fn commit_table_streaming<S: TableRowSource>(
    source: &mut S,
    shape: IndexShape,
    configuration: Digest,
    salts: &[Vec<u8>],
    panel_columns: u64,
    max_bytes: u64,
) -> Result<StreamingTable> {
    validate_index_shape(&shape)?;
    if configuration == Digest::ZERO || panel_columns == 0 {
        return Err(Error::InvalidConfig);
    }
    if source.rows() != shape.rows() {
        return Err(Error::ShapeMismatch);
    }
    let rows = shape.rows() as usize;
    let n = shape.code.n() as usize;
    if salts.len() != n || salts.iter().any(|salt| salt.len() != shape.salt_bytes) {
        return Err(Error::ShapeMismatch);
    }
    let ell = usize::try_from(shape.code.ell).map_err(|_| Error::ResourceLimit)?;
    let tail_len = usize::try_from(shape.code.r).map_err(|_| Error::ResourceLimit)?;
    let panel = usize::try_from(panel_columns.min(n as u64)).map_err(|_| Error::ResourceLimit)?;
    let panel_bytes = rows
        .checked_mul(panel)
        .and_then(|value| value.checked_mul(8))
        .ok_or(Error::ResourceLimit)?;
    let codeword_bytes = n.checked_mul(8).ok_or(Error::ResourceLimit)?;
    let message_bytes = ell
        .checked_add(tail_len)
        .and_then(|value| value.checked_mul(8))
        .ok_or(Error::ResourceLimit)?;
    let leaf_bytes = n.checked_mul(32).ok_or(Error::ResourceLimit)?;
    let tree_bytes = n
        .checked_mul(32)
        .and_then(|value| value.checked_mul(2))
        .ok_or(Error::ResourceLimit)?;
    let needed = panel_bytes
        .checked_add(codeword_bytes)
        .and_then(|value| value.checked_add(message_bytes))
        .and_then(|value| value.checked_add(leaf_bytes))
        .and_then(|value| value.checked_add(tree_bytes))
        .ok_or(Error::ResourceLimit)?;
    if (needed as u64) > max_bytes {
        return Err(Error::ResourceLimit);
    }
    let mut message = vec![Fp::ZERO; ell];
    let mut tail = vec![Fp::ZERO; tail_len];
    let mut symbols = vec![Fp::ZERO; rows * panel];
    let mut leaves = Vec::with_capacity(n);
    let mut start = 0usize;
    while start < n {
        let count = panel.min(n - start);
        for row in 0..rows {
            source.row(row as u64, &mut message, &mut tail)?;
            let codeword = encode_row_fp(&shape.code, &message, &tail)?;
            for column in 0..count {
                symbols[column * rows + row] = codeword[start + column];
            }
        }
        for column in 0..count {
            let position = start + column;
            leaves.push(column_leaf(
                &configuration,
                shape.code.log_n,
                position as u64,
                &salts[position],
                &symbols[column * rows..(column + 1) * rows],
            )?);
        }
        start += count;
    }
    let tree = merkle_build(&configuration, &leaves)?;
    let root = tree.root;
    let name = index_name(&root, &shape, &configuration);
    let answers = usize::try_from(shape.answers).map_err(|_| Error::ResourceLimit)?;
    let seeds = record_seeds(&shape, &name, answers)?;
    let mut record = IndexRecord {
        y: Vec::with_capacity(rows.saturating_mul(answers)),
    };
    for row in 0..rows {
        source.row(row as u64, &mut message, &mut tail)?;
        for answer in 0..answers {
            record.y.push(record_value(
                &shape,
                &seeds[answer],
                answer as u32,
                &message,
            )?);
        }
    }
    let reader = IndexReader {
        shape,
        configuration,
        root,
        record,
    };
    if reader.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    Ok(StreamingTable {
        reader,
        name,
        tree,
        salts: salts.to_vec(),
        leaves,
        peak_bytes: needed as u64,
    })
}

/// Where fold_table() places the folded row: `rows` rows of `len` entries,
/// the folded message from entry 0 and the folded tail from `tail_first`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldedWant {
    pub rows: usize,
    pub len: usize,
    pub tail_first: usize,
}

/// A folded row in the caller's table layout. The values are the index space
/// of `FoldedWant`; zeros fill the unused entries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldedTable {
    pub rows: usize,
    pub len: usize,
    pub values: Vec<Fp3>,
}

/// A row source over fixture messages and tails. It carries no asset; tests
/// and fixture sessions use it to exercise the one streaming committer
/// without a second table implementation.
pub struct VecTableSource {
    messages: Vec<Vec<Fp>>,
    tails: Vec<Vec<Fp>>,
}

impl VecTableSource {
    pub fn new(messages: Vec<Vec<Fp>>, tails: Vec<Vec<Fp>>) -> Result<Self> {
        if messages.is_empty() || messages.len() != tails.len() {
            return Err(Error::ShapeMismatch);
        }
        Ok(Self { messages, tails })
    }
}

impl TableRowSource for VecTableSource {
    fn rows(&self) -> u64 {
        self.messages.len() as u64
    }

    fn row(&mut self, row: u64, message: &mut [Fp], tail: &mut [Fp]) -> Result<()> {
        let message_row = self
            .messages
            .get(row as usize)
            .ok_or(Error::InvalidIndex)?;
        let tail_row = self.tails.get(row as usize).ok_or(Error::InvalidIndex)?;
        if message.len() != message_row.len() || tail.len() != tail_row.len() {
            return Err(Error::ShapeMismatch);
        }
        message.copy_from_slice(message_row);
        tail.copy_from_slice(tail_row);
        Ok(())
    }
}

/// A committed table and its row source. FOLDEQ and its callers fold and open
/// through this holder: the fold is one pass over the rows and the opening one
/// pass over the drawn columns, both with the source's bounded buffers; the
/// complete codewords are never kept. `forget` releases fold state; the stored
/// tree and salts stay with the table.
pub struct StreamingHolder<S: TableRowSource> {
    table: StreamingTable,
    source: Option<S>,
    peak_bytes: u64,
    hiding: HidingState,
    charge: Option<HidingCharge>,
}

impl<S: TableRowSource> StreamingHolder<S> {
    /// `max_columns` is the table-life column budget of the hiding tails: the
    /// number of columns that may be opened over this table's lifetime.
    pub fn new(table: StreamingTable, source: S, max_columns: u64) -> Result<Self> {
        if source.rows() != table.reader.shape.rows() || max_columns == 0 {
            return Err(Error::ShapeMismatch);
        }
        let peak = table.peak_bytes();
        let hiding = HidingState::new(table.reader.root, 1, max_columns);
        Ok(Self {
            table,
            source: Some(source),
            peak_bytes: peak,
            hiding,
            charge: None,
        })
    }

    /// Reserves `columns` opened columns before any fold. The reservation is
    /// spent whether or not the proof is finished; repeated reservations fail
    /// once the table-life budget is exhausted. The count is in-memory in this
    /// layer; a production caller persists it before folding.
    pub fn begin(&mut self, columns: usize) -> Result<()> {
        if columns == 0 {
            return Err(Error::InvalidConfig);
        }
        let root = self.table.reader.root;
        self.charge = Some(self.hiding.begin_charge(root, 1, columns)?);
        Ok(())
    }

    pub fn hiding_used(&self) -> u64 {
        self.hiding.used()
    }

    pub fn reader(&self) -> &IndexReader {
        &self.table.reader
    }

    pub fn table(&self) -> &StreamingTable {
        &self.table
    }

    pub fn peak_bytes(&self) -> u64 {
        self.peak_bytes
    }

    /// message[j] = sum_r eq(r, gamma) * message entry j of row r, pads
    /// included; tail likewise. One row's message and tail are held at a time.
    pub fn fold(&mut self, gamma: &[Fp3]) -> Result<(Vec<Fp3>, Vec<Fp3>)> {
        if self.charge.is_none() {
            return Err(Error::InvalidSessionState);
        }
        let shape = self.table.reader.shape;
        if validate_index_shape(&shape).is_err() || gamma.len() != shape.v as usize {
            return Err(Error::ShapeMismatch);
        }
        let eq_gamma = foldeq_weights(gamma, shape.v)?;
        let rows = shape.rows() as usize;
        let ell = usize::try_from(shape.code.ell).map_err(|_| Error::ResourceLimit)?;
        let tail_len = usize::try_from(shape.code.r).map_err(|_| Error::ResourceLimit)?;
        let mut message = vec![Fp::ZERO; ell];
        let mut tail = vec![Fp::ZERO; tail_len];
        let mut folded = vec![Fp3::ZERO; ell];
        let mut folded_tail = vec![Fp3::ZERO; tail_len];
        for row in 0..rows {
            self.source
                .as_mut()
                .ok_or(Error::InvalidSessionState)?
                .row(row as u64, &mut message, &mut tail)?;
            let weight = eq_gamma[row];
            for (place, value) in message.iter().enumerate() {
                folded[place] = folded[place].add(&weight.mul(&Fp3::from_base(*value)));
            }
            for (place, value) in tail.iter().enumerate() {
                folded_tail[place] =
                    folded_tail[place].add(&weight.mul(&Fp3::from_base(*value)));
            }
        }
        let fold_bytes = ((ell + tail_len) * 8 + (ell + tail_len) * 24 + rows * 24) as u64;
        self.peak_bytes = self.peak_bytes.max(fold_bytes);
        Ok((folded, folded_tail))
    }

    /// The same fold placed in the caller's table layout, without a second
    /// pass over the rows.
    pub fn fold_table(
        &mut self,
        gamma: &[Fp3],
        want: &FoldedWant,
    ) -> Result<FoldedTable> {
        if want.rows == 0 || want.len == 0 {
            return Err(Error::InvalidConfig);
        }
        let (message, tail) = self.fold(gamma)?;
        let total = want
            .rows
            .checked_mul(want.len)
            .ok_or(Error::ResourceLimit)?;
        if want.tail_first < message.len()
            || want
                .tail_first
                .checked_add(tail.len())
                .ok_or(Error::ResourceLimit)?
                > total
            || message.len() > total
        {
            return Err(Error::ShapeMismatch);
        }
        let mut values = vec![Fp3::ZERO; total];
        values[..message.len()].copy_from_slice(&message);
        values[want.tail_first..want.tail_first + tail.len()].copy_from_slice(&tail);
        Ok(FoldedTable {
            rows: want.rows,
            len: want.len,
            values,
        })
    }

    /// Recomputes the drawn columns from the source and checks them against
    /// the stored tree.
    pub fn open(&mut self, positions: &[u64]) -> Result<Vec<ColumnOpening>> {
        {
            let charge = self.charge.as_ref().ok_or(Error::InvalidSessionState)?;
            charge.consume(positions)?;
        }
        let source = self
            .source
            .as_mut()
            .ok_or(Error::InvalidSessionState)?;
        self.table.open_columns(source, positions)
    }

    /// Releases fold state. This holder keeps no folded row between calls; a
    /// caller that caches the returned rows drops them itself.
    pub fn forget(&mut self) -> Result<()> {
        self.charge = None;
        Ok(())
    }

    /// Releases the row source after its last fold or opening. The table's
    /// reader, name, tree and salts stay; the source can no longer fold or
    /// open. A caller uses this once every drawn column was opened.
    pub fn release_source(&mut self) -> Result<()> {
        self.source = None;
        self.charge = None;
        Ok(())
    }
}

/// A private store of the encoded codewords of one commit. It is a cache of
/// values derived from the trusted table rows under one identity, parameters
/// and private material; it is not a trust root. A store must be fresh for one
/// commit, keep the codewords at their positions, and report truncation or a
/// version mismatch instead of returning other values.
pub trait EncodedRowStore {
    fn rows(&self) -> u64;
    fn columns(&self) -> u64;
    fn put_row(&mut self, row: u64, codeword: &[Fp]) -> Result<()>;
    /// Reads a column panel: `out[column * rows + row]` is the codeword
    /// symbol at `first + column` of `row`.
    fn read_columns(
        &mut self,
        first: u64,
        count: usize,
        rows: usize,
        out: &mut [Fp],
    ) -> Result<()>;
    /// Flushes and checks what was written (length, per-row digests, version).
    fn finish(&mut self) -> Result<()>;
    /// Removes the stored bytes. A best-effort removal, not a claim of secure
    /// erasure.
    fn cleanup(&mut self) -> Result<()>;
    fn bytes_written(&self) -> u64;
    fn bytes_read(&self) -> u64;
}

/// An in-memory row store for small tables and tests.
pub struct MemoryRowStore {
    rows: u64,
    columns: u64,
    codewords: Vec<Vec<Fp>>,
    written: u64,
    read: u64,
    finished: bool,
}

impl MemoryRowStore {
    pub fn new(rows: u64, columns: u64) -> Result<Self> {
        let rows_usize = usize::try_from(rows).map_err(|_| Error::ResourceLimit)?;
        let columns_usize = usize::try_from(columns).map_err(|_| Error::ResourceLimit)?;
        Ok(Self {
            rows,
            columns,
            codewords: vec![Vec::new(); rows_usize],
            written: 0,
            read: 0,
            finished: false,
        })
    }
}

impl EncodedRowStore for MemoryRowStore {
    fn rows(&self) -> u64 {
        self.rows
    }

    fn columns(&self) -> u64 {
        self.columns
    }

    fn put_row(&mut self, row: u64, codeword: &[Fp]) -> Result<()> {
        if self.finished || row >= self.rows || codeword.len() as u64 != self.columns {
            return Err(Error::InvalidSessionState);
        }
        let at = self
            .codewords
            .get_mut(row as usize)
            .ok_or(Error::InvalidIndex)?;
        if !at.is_empty() {
            return Err(Error::InvalidSessionState);
        }
        *at = codeword.to_vec();
        self.written = self
            .written
            .checked_add((codeword.len() as u64).saturating_mul(8))
            .ok_or(Error::ResourceLimit)?;
        Ok(())
    }

    fn read_columns(
        &mut self,
        first: u64,
        count: usize,
        rows: usize,
        out: &mut [Fp],
    ) -> Result<()> {
        if !self.finished || rows as u64 != self.rows {
            return Err(Error::InvalidSessionState);
        }
        if out.len() != count.saturating_mul(rows)
            || first.checked_add(count as u64).ok_or(Error::ResourceLimit)? > self.columns
        {
            return Err(Error::ShapeMismatch);
        }
        for row in 0..rows {
            let codeword = self.codewords.get(row).ok_or(Error::InvalidIndex)?;
            if codeword.is_empty() {
                return Err(Error::IncompleteArtifact);
            }
            for column in 0..count {
                out[column * rows + row] = codeword[first as usize + column];
            }
        }
        self.read = self
            .read
            .checked_add((count.saturating_mul(rows) as u64).saturating_mul(8))
            .ok_or(Error::ResourceLimit)?;
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if self.codewords.iter().any(Vec::is_empty) {
            return Err(Error::IncompleteArtifact);
        }
        self.finished = true;
        Ok(())
    }

    fn cleanup(&mut self) -> Result<()> {
        self.codewords.clear();
        Ok(())
    }

    fn bytes_written(&self) -> u64 {
        self.written
    }

    fn bytes_read(&self) -> u64 {
        self.read
    }
}

/// Commits a table through a bounded row source and a private store of the
/// encoded rows. The first pass encodes each row once into the store; the
/// second pass reads column panels from the store and builds the leaves. The
/// root, record and openings are the same as the plain streaming committer's;
/// the store holds no plaintext and is not trusted.
pub fn commit_table_stored<S: TableRowSource, T: EncodedRowStore + ?Sized>(
    source: &mut S,
    store: &mut T,
    shape: IndexShape,
    configuration: Digest,
    salts: &[Vec<u8>],
    panel_columns: u64,
    max_bytes: u64,
) -> Result<StreamingTable> {
    validate_index_shape(&shape)?;
    if configuration == Digest::ZERO || panel_columns == 0 {
        return Err(Error::InvalidConfig);
    }
    let rows = shape.rows();
    let n = shape.code.n();
    if source.rows() != rows
        || salts.len() as u64 != n
        || salts.iter().any(|salt| salt.len() != shape.salt_bytes)
        || store.rows() != rows
        || store.columns() != n
    {
        return Err(Error::ShapeMismatch);
    }
    let rows_usize = usize::try_from(rows).map_err(|_| Error::ResourceLimit)?;
    let n_usize = usize::try_from(n).map_err(|_| Error::ResourceLimit)?;
    let ell = usize::try_from(shape.code.ell).map_err(|_| Error::ResourceLimit)?;
    let tail_len = usize::try_from(shape.code.r).map_err(|_| Error::ResourceLimit)?;
    let panel = usize::try_from(panel_columns.min(n)).map_err(|_| Error::ResourceLimit)?;
    let panel_bytes = rows_usize
        .checked_mul(panel)
        .and_then(|value| value.checked_mul(8))
        .ok_or(Error::ResourceLimit)?;
    let message_bytes = ell
        .checked_add(tail_len)
        .and_then(|value| value.checked_mul(8))
        .ok_or(Error::ResourceLimit)?;
    let leaf_bytes = n_usize.checked_mul(32).ok_or(Error::ResourceLimit)?;
    let tree_bytes = n_usize
        .checked_mul(64)
        .ok_or(Error::ResourceLimit)?;
    let codeword_bytes = n_usize.checked_mul(8).ok_or(Error::ResourceLimit)?;
    let needed = panel_bytes
        .checked_add(message_bytes)
        .and_then(|value| value.checked_add(leaf_bytes))
        .and_then(|value| value.checked_add(tree_bytes))
        .and_then(|value| value.checked_add(codeword_bytes))
        .ok_or(Error::ResourceLimit)?;
    if (needed as u64) > max_bytes {
        return Err(Error::ResourceLimit);
    }
    let mut message = vec![Fp::ZERO; ell];
    let mut tail = vec![Fp::ZERO; tail_len];
    let _profile_encode_rows = crate::profile::Scope::new("table.commit.encode_rows");
    for row in 0..rows_usize {
        source.row(row as u64, &mut message, &mut tail)?;
        let codeword = encode_row_fp(&shape.code, &message, &tail)?;
        store.put_row(row as u64, &codeword)?;
    }
    drop(_profile_encode_rows);
    let _profile_store_finish = crate::profile::Scope::new("table.commit.store_finish");
    store.finish()?;
    drop(_profile_store_finish);
    let mut symbols = vec![Fp::ZERO; rows_usize * panel];
    let mut leaves = Vec::with_capacity(n_usize);
    let mut start = 0usize;
    let _profile_leaves = crate::profile::Scope::new("table.commit.leaves");
    while start < n_usize {
        let count = panel.min(n_usize - start);
        store.read_columns(start as u64, count, rows_usize, &mut symbols)?;
        for column in 0..count {
            leaves.push(column_leaf(
                &configuration,
                shape.code.log_n,
                (start + column) as u64,
                &salts[start + column],
                &symbols[column * rows_usize..(column + 1) * rows_usize],
            )?);
        }
        start += count;
    }
    drop(_profile_leaves);
    let _profile_merkle = crate::profile::Scope::new("table.commit.merkle");
    let tree = merkle_build(&configuration, &leaves)?;
    drop(_profile_merkle);
    let root = tree.root;
    let name = index_name(&root, &shape, &configuration);
    let answers = usize::try_from(shape.answers).map_err(|_| Error::ResourceLimit)?;
    let seeds = record_seeds(&shape, &name, answers)?;
    let mut record = IndexRecord {
        y: Vec::with_capacity(rows_usize.saturating_mul(answers)),
    };
    let _profile_record = crate::profile::Scope::new("table.commit.record");
    for row in 0..rows_usize {
        source.row(row as u64, &mut message, &mut tail)?;
        for answer in 0..answers {
            record.y.push(record_value(
                &shape,
                &seeds[answer],
                answer as u32,
                &message,
            )?);
        }
    }
    drop(_profile_record);
    let reader = IndexReader {
        shape,
        configuration,
        root,
        record,
    };
    if reader.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    Ok(StreamingTable {
        reader,
        name,
        tree,
        salts: salts.to_vec(),
        leaves,
        peak_bytes: needed as u64,
    })
}

/// The test-only table entry. It has the same reader, name and opening
/// behaviour the unit fixtures used, but it is built by the production
/// streaming committer over an in-memory row source; it never materializes a
/// codeword matrix and is not a production constructor.
#[cfg(test)]
pub struct TestTable {
    pub reader: IndexReader,
    pub name: Vec<u8>,
    pub codewords: Vec<Vec<Fp>>,
    pub salts: Vec<Vec<u8>>,
    pub tree: MerkleTree,
    table: StreamingTable,
    source: std::sync::Mutex<VecTableSource>,
}

#[cfg(test)]
impl TestTable {
    pub fn open_columns(&self, positions: &[u64]) -> Result<Vec<ColumnOpening>> {
        let mut source = self
            .source
            .lock()
            .map_err(|_| Error::InvalidSessionState)?;
        self.table.open_columns(&mut *source, positions)
    }

    pub fn open_columns_charged(
        &self,
        charge: &HidingCharge,
        positions: &[u64],
    ) -> Result<Vec<ColumnOpening>> {
        if charge.table() != self.reader.root {
            return Err(Error::StatementMismatch);
        }
        charge.consume(positions)?;
        self.open_columns(positions)
    }
}

/// Builds the test-only table entry through the streaming committer.
#[cfg(test)]
pub fn test_commit_table(
    shape: IndexShape,
    configuration: Digest,
    messages: &[Vec<Fp>],
    tails: &[Vec<Fp>],
    salts: &[Vec<u8>],
) -> Result<TestTable> {
    let codewords = messages
        .iter()
        .zip(tails)
        .map(|(message, tail)| encode_row_fp(&shape.code, message, tail))
        .collect::<Result<Vec<_>>>()?;
    let mut source = VecTableSource::new(messages.to_vec(), tails.to_vec())?;
    let table = commit_table_streaming(&mut source, shape, configuration, salts, 64, u64::MAX)?;
    let reader = table.reader.clone();
    let name = table.name.clone();
    Ok(TestTable {
        reader,
        name,
        codewords,
        salts: salts.to_vec(),
        tree: table.tree.clone(),
        table,
        source: std::sync::Mutex::new(source),
    })
}

/// Check column openings against a reader and return the columns as
/// `columns[i * rows + r]`.
pub fn verify_columns(
    reader: &IndexReader,
    positions: &[u64],
    openings: &[ColumnOpening],
) -> Result<Vec<Fp>> {
    if reader.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    if openings.len() != positions.len() {
        return Err(Error::ShapeMismatch);
    }
    let rows = reader.shape.rows() as usize;
    let n = reader.shape.code.n();
    let capacity = positions
        .len()
        .checked_mul(rows)
        .ok_or(Error::ResourceLimit)?;
    let mut columns = Vec::with_capacity(capacity);
    let mut previous = None;
    for (position, opening) in positions.iter().zip(openings) {
        if *position >= n {
            return Err(Error::InvalidIndex);
        }
        if let Some(previous) = previous {
            if *position <= previous {
                return Err(Error::InvalidIndex);
            }
        }
        previous = Some(*position);
        if opening.position != *position {
            return Err(Error::InvalidIndex);
        }
        if opening.salt.len() != reader.shape.salt_bytes {
            return Err(Error::ShapeMismatch);
        }
        if opening.symbols.len() != rows {
            return Err(Error::ShapeMismatch);
        }
        if opening.path.len() != reader.shape.code.log_n as usize {
            return Err(Error::InvalidProof);
        }
        let leaf = column_leaf(
            &reader.configuration,
            reader.shape.code.log_n,
            opening.position,
            &opening.salt,
            &opening.symbols,
        )?;
        if !merkle_verify(
            &reader.configuration,
            &reader.root,
            opening.position as usize,
            &leaf,
            &opening.path,
        )? {
            return Err(Error::InvalidProof);
        }
        columns.extend_from_slice(&opening.symbols);
    }
    Ok(columns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::request::chain::KIND_SEED;
    use crate::pcs::request::code::{build_code_plan, symbol_at_fp};
    use crate::primitives::transcript::{
        hash_labeled, hash_labeled_64, tag_word, CHAIN_QUERY, CHAIN_START,
    };
    use zkie_core::common::field::{PrimeCharacteristicRing, PrimeField64};

    fn shape() -> IndexShape {
        IndexShape {
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
        }
    }

    fn messages(shape: &IndexShape) -> Vec<Vec<Fp>> {
        let rows = shape.rows() as usize;
        (0..rows)
            .map(|row| {
                (0..shape.code.ell)
                    .map(|index| Fp::from_u64((row * 10 + index as usize + 1) as u64))
                    .collect()
            })
            .collect()
    }

    fn tails(shape: &IndexShape) -> Vec<Vec<Fp>> {
        let rows = shape.rows() as usize;
        (0..rows)
            .map(|row| {
                (0..shape.code.r)
                    .map(|index| Fp::from_u64((row + index as usize + 1) as u64))
                    .collect()
            })
            .collect()
    }

    fn salts(shape: &IndexShape) -> Vec<Vec<u8>> {
        (0..shape.code.n() as usize)
            .map(|position| vec![(position as u8).wrapping_add(1); shape.salt_bytes])
            .collect()
    }

    fn committed() -> TestTable {
        let shape = shape();
        let configuration = Digest([7; 32]);
        test_commit_table(
            shape,
            configuration,
            &messages(&shape),
            &tails(&shape),
            &salts(&shape),
        )
        .unwrap()
    }

    #[test]
    fn positive_commit_and_open() {
        let table = committed();
        assert_eq!(
            table.reader.record.y.len(),
            (table.reader.shape.rows() * table.reader.shape.answers as u64) as usize
        );
        let positions = [1u64, 4, 6];
        let openings = table.open_columns(&positions).unwrap();
        let columns = verify_columns(&table.reader, &positions, &openings).unwrap();
        assert_eq!(
            columns.len(),
            positions.len() * table.reader.shape.rows() as usize
        );
        // One symbol equals the direct evaluation of the row's message and tail.
        let messages = messages(&table.reader.shape);
        let tails = tails(&table.reader.shape);
        let expected = symbol_at_fp(
            &table.reader.shape.code,
            positions[0],
            &messages[0],
            &tails[0],
        )
        .unwrap();
        assert_eq!(columns[0], expected);
    }

    #[test]
    fn tampered_columns_are_rejected() {
        let table = committed();
        let positions = [1u64, 4];
        let openings = table.open_columns(&positions).unwrap();

        let mut symbol = openings.clone();
        symbol[0].symbols[0] = symbol[0].symbols[0] + Fp::ONE;
        assert!(verify_columns(&table.reader, &positions, &symbol).is_err());

        let mut salt = openings.clone();
        salt[0].salt[0] ^= 1;
        assert!(verify_columns(&table.reader, &positions, &salt).is_err());

        let mut path = openings.clone();
        path[0].path[0] = Digest([9; 32]);
        assert!(verify_columns(&table.reader, &positions, &path).is_err());

        let mut index = openings.clone();
        index[0].position += 1;
        assert!(verify_columns(&table.reader, &positions, &index).is_err());

        let mut short = openings.clone();
        short[0].symbols.pop();
        assert!(verify_columns(&table.reader, &positions, &short).is_err());
    }

    #[test]
    fn order_and_length_are_checked() {
        let table = committed();
        let openings = table.open_columns(&[1, 4]).unwrap();
        assert!(verify_columns(&table.reader, &[4, 1], &openings).is_err());
        assert!(verify_columns(&table.reader, &[1, 1], &openings).is_err());
        assert!(verify_columns(&table.reader, &[1], &openings).is_err());
        assert!(verify_columns(&table.reader, &[1, 4, 6], &openings).is_err());
    }

    #[test]
    fn configuration_and_distances_binding() {
        let shape = shape();
        let messages = messages(&shape);
        let tails = tails(&shape);
        let salts = salts(&shape);
        let first = test_commit_table(
            shape,
            Digest([7; 32]),
            &messages,
            &tails,
            &salts,
        )
        .unwrap();
        let second = test_commit_table(
            shape,
            Digest([8; 32]),
            &messages,
            &tails,
            &salts,
        )
        .unwrap();
        assert_ne!(first.reader.root, second.reader.root);
        assert_ne!(first.name, second.name);

        // Distances are not bound by the name or the commitment.
        let mut changed = shape;
        changed.distances.radius += 1;
        changed.distances.later_radius += 2;
        changed.distances.m_once += 3;
        changed.distances.m_later += 4;
        assert_eq!(
            index_name(&first.reader.root, &changed, &first.reader.configuration),
            first.name
        );
        let renamed = test_commit_table(
            changed,
            Digest([7; 32]),
            &messages,
            &tails,
            &salts,
        )
        .unwrap();
        assert_eq!(renamed.reader.root, first.reader.root);
        assert_eq!(renamed.name, first.name);
    }

    #[test]
    fn record_seeds_are_deterministic_and_bound() {
        let shape = shape();
        let name = b"table-name";
        let first = record_seeds(&shape, name, shape.answers as usize).unwrap();
        let second = record_seeds(&shape, name, shape.answers as usize).unwrap();
        assert_eq!(first, second);
        let other = record_seeds(&shape, b"other-name", shape.answers as usize).unwrap();
        assert_ne!(first, other);
        assert!(!first[0].beta.is_zero());

        // One sequential draw per answer: with the same code, the first seeds
        // do not change when more answers are added.
        let mut two_answers = shape;
        two_answers.answers = 2;
        two_answers.code = crate::pcs::request::code::build_code_plan(5, 16, 1).unwrap();
        let ordered =
            record_seeds(&two_answers, b"table-name", two_answers.answers as usize).unwrap();
        assert_eq!(ordered.len(), 2);
        assert_ne!(ordered[0], ordered[1]);
        let mut three_answers = two_answers;
        three_answers.answers = 3;
        let extended =
            record_seeds(&three_answers, b"table-name", three_answers.answers as usize).unwrap();
        assert_eq!(extended.len(), 3);
        assert_eq!(extended[..2], ordered[..]);

        // The entry validates the shape and checks the caller's seed budget.
        assert!(record_seeds(&shape, name, 0).is_err());
        assert!(record_seeds(&two_answers, b"table-name", 1).is_err());
        let mut bad = shape;
        bad.v = 64;
        assert!(record_seeds(&bad, name, 8).is_err());
        let mut bad_code = shape;
        bad_code.code.log_n = 64;
        assert!(record_seeds(&bad_code, name, 8).is_err());
    }

    #[test]
    fn index_name_binds_the_complete_record() {
        let table = committed();
        let mut name = IndexName {
            commitment: table.reader.commitment(),
            record: table.reader.record.clone(),
        };
        let original = name.encode();
        name.record.y[0] = name.record.y[0].add(&Fp3::ONE);
        assert_ne!(original, name.encode());
        let mut other = IndexName {
            commitment: vec![1, 2, 3],
            record: table.reader.record.clone(),
        };
        let with_other_commitment = other.encode();
        other.commitment = table.reader.commitment();
        assert_ne!(with_other_commitment, other.encode());
    }

    #[test]
    fn record_round_trip_and_length_check() {
        let table = committed();
        let bytes = table.reader.record.encode();
        let decoded = IndexRecord::decode(&bytes, table.reader.record.y.len()).unwrap();
        assert_eq!(decoded, table.reader.record);
        let mut short = table.reader.record.clone();
        short.y.pop();
        assert!(record_fault(&table.reader.shape, &short).is_some());
    }

    #[test]
    fn shape_faults_are_rejected() {
        let base = shape();
        let mut bad = base;
        bad.answers = 0;
        assert!(index_fault(&bad).is_some());
        let mut bad = base;
        bad.salt_bytes = LEAF_SALT_MIN_BYTES - 1;
        assert!(index_fault(&bad).is_some());
        let mut bad = base;
        bad.salt_bytes = LEAF_SALT_MAX_BYTES + 1;
        assert!(index_fault(&bad).is_some());
        let mut bad = base;
        bad.v = 41;
        assert!(index_fault(&bad).is_some());
        let mut bad = base;
        bad.code.ell = 3;
        assert!(index_fault(&bad).is_some());
        let mut bad = base;
        bad.code.ell = 0;
        assert!(index_fault(&bad).is_some());
    }

    #[test]
    fn commit_rejects_wrong_inputs() {
        let shape = shape();
        let messages = messages(&shape);
        let tails = tails(&shape);
        let salts = salts(&shape);
        assert!(test_commit_table(shape, Digest::ZERO, &messages, &tails, &salts).is_err());
        assert!(test_commit_table(
            shape,
            Digest([7; 32]),
            &messages[..3],
            &tails,
            &salts
        )
        .is_err());
        assert!(test_commit_table(
            shape,
            Digest([7; 32]),
            &messages,
            &tails,
            &salts[..7]
        )
        .is_err());
        let mut short_message = messages.clone();
        short_message[0].pop();
        assert!(test_commit_table(shape, Digest([7; 32]), &short_message, &tails, &salts).is_err());
        let mut short_salt = salts.clone();
        short_salt[0].pop();
        assert!(test_commit_table(shape, Digest([7; 32]), &messages, &tails, &short_salt).is_err());
    }

    #[test]
    fn leaf_root_and_depth_are_checked() {
        let table = committed();
        let rows = table.reader.shape.rows() as usize;
        let shape = table.reader.shape;
        // A malicious reader whose root is one column's leaf and whose opening
        // has an empty path must be rejected: the declared tree is larger.
        let position = 0u64;
        let mut symbols = Vec::with_capacity(rows);
        for codeword in &table.codewords {
            symbols.push(codeword[position as usize]);
        }
        let leaf = column_leaf(
            &table.reader.configuration,
            shape.code.log_n,
            position,
            &table.salts[position as usize],
            &symbols,
        )
        .unwrap();
        let malicious = IndexReader {
            shape,
            configuration: table.reader.configuration,
            root: leaf,
            record: table.reader.record.clone(),
        };
        let opening = ColumnOpening {
            position,
            salt: table.salts[position as usize].clone(),
            symbols,
            path: Vec::new(),
        };
        assert!(verify_columns(&malicious, &[position], &[opening]).is_err());
    }

    #[test]
    fn decode_is_bounded_and_checked() {
        let table = committed();
        let bytes = table.reader.record.encode();
        let count = table.reader.record.y.len();
        assert!(IndexRecord::decode(&bytes, count - 1).is_err());
        assert!(IndexRecord::decode(&bytes, count).is_ok());
        let mut trailing = bytes.clone();
        trailing.push(1);
        assert!(IndexRecord::decode(&trailing, count + 1).is_err());
        assert!(IndexRecord::decode_for(&table.reader.shape, &bytes).is_ok());
        let mut short = bytes;
        short.truncate(short.len() - crate::pcs::request::field::FP3_BYTES);
        assert!(IndexRecord::decode_for(&table.reader.shape, &short).is_err());
    }

    fn push_length_prefixed(data: &mut Vec<u8>, value: &[u8]) {
        data.extend_from_slice(&(value.len() as u64).to_le_bytes());
        data.extend_from_slice(value);
    }

    /// The seeds derived by an independent local reconstruction of the
    /// documented chain bytes and draws (no Chain method is called).
    struct ManualSeeds {
        rho: Digest,
        round: u64,
        call_index: u64,
        words: [u64; 8],
        next: usize,
        have: usize,
    }

    impl ManualSeeds {
        fn new(name: &[u8]) -> Self {
            let mut data = Vec::new();
            data.extend_from_slice(&Digest::ZERO.0);
            push_length_prefixed(&mut data, name);
            push_length_prefixed(&mut data, &[]);
            data.extend_from_slice(&0u64.to_le_bytes());
            let rho = hash_labeled(
                tag_word(CHAIN_START, RECORD_SEED_LABEL),
                RECORD_SEED_LABEL,
                None,
                &data,
            )
            .unwrap();
            let mut manual = Self {
                rho: Digest::ZERO,
                round: 0,
                call_index: 1,
                words: [0u64; 8],
                next: 0,
                have: 0,
            };
            let first = manual.call(0, &rho);
            // Every call of this round hashes the round's rho_prev, the chain
            // start value; the call's response is not that prefix.
            manual.rho = rho;
            for index in 0..4 {
                manual.words[index] = u64::from_le_bytes(
                    first[32 + index * 8..40 + index * 8].try_into().unwrap(),
                );
            }
            manual.have = 4;
            // Later calls in the same round keep the prefix's round number.
            manual
        }

        fn call(&self, j: u64, rho: &Digest) -> [u8; 64] {
            let closing = (KIND_SEED as u64) | (j << 16);
            let mut message = Vec::new();
            message.extend_from_slice(&closing.to_le_bytes());
            message.extend_from_slice(&0u64.to_le_bytes());
            hash_labeled_64(
                tag_word(CHAIN_QUERY, RECORD_SEED_LABEL),
                self.round,
                rho,
                &message,
            )
            .unwrap()
        }

        fn next_word(&mut self) -> u64 {
            if self.next == self.have {
                let response = self.call(self.call_index, &self.rho);
                self.call_index += 1;
                for index in 0..8 {
                    self.words[index] = u64::from_le_bytes(
                        response[index * 8..(index + 1) * 8].try_into().unwrap(),
                    );
                }
                self.next = 0;
                self.have = 8;
            }
            let word = self.words[self.next];
            self.next += 1;
            word
        }

        fn draw_f(&mut self) -> Fp {
            for _ in 0..6 {
                let word = self.next_word();
                if word < crate::pcs::request::field::MODULUS {
                    return Fp::from_u64(word);
                }
            }
            Fp::ZERO
        }

        fn draw_e(&mut self) -> Fp3 {
            Fp3::new(self.draw_f(), self.draw_f(), self.draw_f())
        }

        fn seeds(&mut self, shape: &IndexShape) -> Vec<(Vec<Fp3>, Fp3)> {
            let mut seeds = Vec::new();
            for _ in 0..shape.answers {
                let mut z = Vec::new();
                for _ in 0..shape.cube_bits() {
                    z.push(self.draw_e());
                }
                let mut beta = Fp3::ONE;
                for attempt in 0..2 {
                    let value = self.draw_e();
                    if !value.is_zero() || attempt == 1 {
                        beta = if value.is_zero() { Fp3::ONE } else { value };
                        break;
                    }
                }
                seeds.push((z, beta));
            }
            seeds
        }
    }

    fn manual_seeds(shape: &IndexShape, name: &[u8]) -> Vec<(Vec<Fp3>, Fp3)> {
        ManualSeeds::new(name).seeds(shape)
    }

    #[test]
    fn record_seeds_match_an_independent_derivation() {
        let shape = shape();
        let name = b"independent-name";
        let seeds = record_seeds(&shape, name, shape.answers as usize).unwrap();
        let manual = manual_seeds(&shape, name);
        assert_eq!(seeds.len(), manual.len());
        for (left, right) in seeds.iter().zip(&manual) {
            assert_eq!(&left.z, &right.0);
            assert_eq!(left.beta, right.1);
        }
    }

    #[test]
    fn noncanonical_leaf_variants_are_rejected() {
        let table = committed();
        let shape = table.reader.shape;
        let position = 0u64;
        let rows = shape.rows() as usize;
        let mut symbols = Vec::with_capacity(rows);
        for codeword in &table.codewords {
            symbols.push(codeword[position as usize]);
        }
        let salt = &table.salts[position as usize];
        let mut data = Vec::new();
        data.extend_from_slice(&(symbols.len() as u64).to_le_bytes());
        data.extend_from_slice(&(salt.len() as u64).to_le_bytes());
        for value in &symbols {
            data.extend_from_slice(&value.as_canonical_u64().to_le_bytes());
        }
        data.extend_from_slice(&(salt.len() as u64).to_le_bytes());
        data.extend_from_slice(salt);
        let index = (1u64 << shape.code.log_n) + position;
        let current = hash_labeled(
            tag_word(crate::primitives::transcript::LEAF_STORED_MAIN, 0),
            index,
            Some(&table.reader.configuration),
            &data,
        )
        .unwrap();
        // The leaf as if the index were the bare position.
        let bare = hash_labeled(
            tag_word(crate::primitives::transcript::LEAF_STORED_MAIN, 0),
            position,
            Some(&table.reader.configuration),
            &data,
        )
        .unwrap();
        assert_ne!(current, bare);
        // A salt-first data order.
        let mut salt_first = Vec::new();
        salt_first.extend_from_slice(&(symbols.len() as u64).to_le_bytes());
        salt_first.extend_from_slice(&(salt.len() as u64).to_le_bytes());
        salt_first.extend_from_slice(&(salt.len() as u64).to_le_bytes());
        salt_first.extend_from_slice(salt);
        for value in &symbols {
            salt_first.extend_from_slice(&value.as_canonical_u64().to_le_bytes());
        }
        let other = hash_labeled(
            tag_word(crate::primitives::transcript::LEAF_STORED_MAIN, 0),
            index,
            Some(&table.reader.configuration),
            &salt_first,
        )
        .unwrap();
        assert_ne!(current, other);
        // A reader with the bare-index leaf as root rejects a full-depth path.
        let malicious = IndexReader {
            shape,
            configuration: table.reader.configuration,
            root: bare,
            record: table.reader.record.clone(),
        };
        let opening = ColumnOpening {
            position,
            salt: salt.clone(),
            symbols,
            path: table.tree.open(position as usize).unwrap(),
        };
        assert!(verify_columns(&malicious, &[position], &[opening]).is_err());
    }

    #[test]
    fn tree_shape_checks_are_separate() {
        let base = shape();
        let mut bad = base;
        bad.salt_bytes = LEAF_SALT_MIN_BYTES + 1;
        assert!(index_fault(&bad).is_none());
        assert!(tree_shape_fault(&bad).is_some());
        assert!(validate_index_shape(&bad).is_err());
        let mut wide = base;
        wide.v = 24;
        assert!(tree_shape_fault(&wide).is_some());
    }

    #[test]
    fn index_name_record_encoding_is_stable() {
        let table = committed();
        let name = IndexName {
            commitment: table.reader.commitment(),
            record: table.reader.record.clone(),
        };
        let first = name.encode();
        let second = name.encode();
        assert_eq!(first, second);
        assert!(first.len() > table.reader.record.encode().len());
    }

    #[test]
    fn oversized_shapes_are_rejected_without_a_shift_panic() {
        let base = shape();
        for v in [64u32, 65, u32::MAX] {
            let mut bad = base;
            bad.v = v;
            assert!(tree_shape_fault(&bad).is_some());
            assert!(record_fault(&bad, &IndexRecord { y: Vec::new() }).is_some());
            assert!(IndexRecord::decode_for(&bad, &[]).is_err());
            assert!(validate_index_shape(&bad).is_err());
            let reader = IndexReader {
                shape: bad,
                configuration: Digest([7; 32]),
                root: Digest([1; 32]),
                record: IndexRecord { y: Vec::new() },
            };
            assert!(reader.fault().is_some());
            assert!(verify_columns(&reader, &[], &[]).is_err());
        }
        // A code number that cannot describe a code is rejected before the
        // code's length is shifted.
        let mut bad_code = base;
        bad_code.code.log_n = 64;
        assert!(record_fault(&bad_code, &IndexRecord { y: Vec::new() }).is_some());
        assert!(IndexRecord::decode_for(&bad_code, &[]).is_err());
        assert!(tree_shape_fault(&bad_code).is_none());
    }

    #[test]
    fn opening_paths_are_bound_to_position_and_configuration() {
        let table = committed();
        let positions = [1u64, 4];
        let openings = table.open_columns(&positions).unwrap();
        // Swap the two paths: neither leaf matches the other's path.
        let mut swapped = openings.clone();
        let first_path = swapped[0].path.clone();
        swapped[0].path = swapped[1].path.clone();
        swapped[1].path = first_path;
        assert!(verify_columns(&table.reader, &positions, &swapped).is_err());
        // A different configuration changes the hash start of every leaf.
        let mut other = table.reader.clone();
        other.configuration = Digest([8; 32]);
        assert!(verify_columns(&other, &positions, &openings).is_err());
        // A different root rejects every path.
        let mut other = table.reader.clone();
        other.root = Digest([1; 32]);
        assert!(verify_columns(&other, &positions, &openings).is_err());
    }
}


#[cfg(test)]
mod streaming_tests {
    use super::*;
    use crate::pcs::request::code::build_code_plan;
    use zkie_core::common::field::PrimeCharacteristicRing;

    struct VecSource {
        messages: Vec<Vec<Fp>>,
        tails: Vec<Vec<Fp>>,
    }

    impl TableRowSource for VecSource {
        fn rows(&self) -> u64 {
            self.messages.len() as u64
        }

        fn row(&mut self, row: u64, message: &mut [Fp], tail: &mut [Fp]) -> Result<()> {
            let message_row = self
                .messages
                .get(row as usize)
                .ok_or(Error::InvalidIndex)?;
            let tail_row = self
                .tails
                .get(row as usize)
                .ok_or(Error::InvalidIndex)?;
            if message.len() != message_row.len() || tail.len() != tail_row.len() {
                return Err(Error::ShapeMismatch);
            }
            message.copy_from_slice(message_row);
            tail.copy_from_slice(tail_row);
            Ok(())
        }
    }

    fn shape() -> IndexShape {
        IndexShape {
            code: build_code_plan(3, 4, 1).expect("code"),
            v: 2,
            answers: 1,
            salt_bytes: LEAF_SALT_MIN_BYTES,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        }
    }

    fn fixture() -> (IndexShape, Digest, Vec<Vec<Fp>>, Vec<Vec<Fp>>, Vec<Vec<u8>>) {
        let shape = shape();
        let messages: Vec<Vec<Fp>> = (0..shape.rows() as usize)
            .map(|row| {
                (0..shape.code.ell as usize)
                    .map(|index| Fp::from_u64((row * 10 + index + 1) as u64))
                    .collect()
            })
            .collect();
        let tails: Vec<Vec<Fp>> = (0..shape.rows() as usize)
            .map(|row| {
                (0..shape.code.r as usize)
                    .map(|index| Fp::from_u64((row + index + 1) as u64))
                    .collect()
            })
            .collect();
        let salts: Vec<Vec<u8>> = (0..shape.code.n() as usize)
            .map(|position| vec![(position as u8).wrapping_add(1); shape.salt_bytes])
            .collect();
        (shape, Digest([7; 32]), messages, tails, salts)
    }

    #[test]
    fn streaming_matches_the_in_memory_committer() {
        let (shape, configuration, messages, tails, salts) = fixture();
        let table = test_commit_table(
            shape,
            configuration,
            &messages,
            &tails,
            &salts,
        )
        .expect("in-memory");
        for panel in [1u64, 2, 8] {
            let mut source = VecSource {
                messages: messages.clone(),
                tails: tails.clone(),
            };
            let streamed = commit_table_streaming(
                &mut source,
                shape,
                configuration,
                &salts,
                panel,
                u64::MAX,
            )
            .expect("streaming");
            assert_eq!(streamed.reader.root, table.reader.root);
            assert_eq!(streamed.name, table.name);
            assert_eq!(streamed.reader.record.y, table.reader.record.y);
            assert_eq!(streamed.reader.shape, table.reader.shape);
            assert_eq!(streamed.reader.configuration, table.reader.configuration);
            assert!(streamed.peak_bytes() > 0);
            let positions = [1u64, 4, 6];
            let expected = table.open_columns(&positions).expect("open");
            let opened = streamed
                .open_columns(&mut source, &positions)
                .expect("stream open");
            assert_eq!(opened, expected);
        }
    }

    #[test]
    fn the_holder_folds_opens_and_places_the_folded_row() {
        let (shape, configuration, messages, tails, salts) = fixture();
        let expected_table =
            test_commit_table(shape, configuration, &messages, &tails, &salts).expect("in-memory");
        let mut source = VecSource {
            messages: messages.clone(),
            tails: tails.clone(),
        };
        let streamed =
            commit_table_streaming(&mut source, shape, configuration, &salts, 2, u64::MAX)
                .expect("streaming");
        let mut holder = StreamingHolder::new(streamed, source, 8).expect("holder");
        holder.begin(3).expect("begin");
        let gamma = vec![Fp3::from_u64(2), Fp3::from_u64(3)];
        let (folded, folded_tail) = holder.fold(&gamma).expect("fold");
        let eq = foldeq_weights(&gamma, shape.v).expect("eq");
        let mut expected = vec![Fp3::ZERO; shape.code.ell as usize];
        let mut expected_tail = vec![Fp3::ZERO; shape.code.r as usize];
        for (row, weight) in eq.iter().enumerate() {
            for (place, value) in messages[row].iter().enumerate() {
                expected[place] = expected[place].add(&weight.mul(&Fp3::from_base(*value)));
            }
            for (place, value) in tails[row].iter().enumerate() {
                expected_tail[place] =
                    expected_tail[place].add(&weight.mul(&Fp3::from_base(*value)));
            }
        }
        assert_eq!(folded, expected);
        assert_eq!(folded_tail, expected_tail);
        let want = FoldedWant {
            rows: 5,
            len: 8,
            tail_first: 8,
        };
        let placed = holder.fold_table(&gamma, &want).expect("fold table");
        assert_eq!(&placed.values[..folded.len()], folded.as_slice());
        assert_eq!(
            &placed.values[want.tail_first..want.tail_first + folded_tail.len()],
            folded_tail.as_slice()
        );
        let positions = [1u64, 4, 6];
        let opened = holder.open(&positions).expect("open");
        let expected_open = expected_table.open_columns(&positions).expect("in-memory open");
        assert_eq!(opened, expected_open);
        holder.forget().expect("forget");
        holder.begin(2).expect("begin again");
        let (again, _) = holder.fold(&gamma).expect("fold again");
        assert_eq!(again, folded);
        assert!(holder.peak_bytes() > 0);
    }

    struct TamperedStore {
        inner: MemoryRowStore,
        tamper: bool,
    }

    impl EncodedRowStore for TamperedStore {
        fn rows(&self) -> u64 {
            self.inner.rows()
        }
        fn columns(&self) -> u64 {
            self.inner.columns()
        }
        fn put_row(&mut self, row: u64, codeword: &[Fp]) -> Result<()> {
            self.inner.put_row(row, codeword)
        }
        fn read_columns(
            &mut self,
            first: u64,
            count: usize,
            rows: usize,
            out: &mut [Fp],
        ) -> Result<()> {
            self.inner.read_columns(first, count, rows, out)?;
            if self.tamper && !out.is_empty() {
                out[0] = out[0] + Fp::from_u64(1);
            }
            Ok(())
        }
        fn finish(&mut self) -> Result<()> {
            self.inner.finish()
        }
        fn cleanup(&mut self) -> Result<()> {
            self.inner.cleanup()
        }
        fn bytes_written(&self) -> u64 {
            self.inner.bytes_written()
        }
        fn bytes_read(&self) -> u64 {
            self.inner.bytes_read()
        }
    }

    #[test]
    fn the_stored_commit_matches_the_plain_stream_and_rejects_corruption() {
        let (shape, configuration, messages, tails, salts) = fixture();
        let expected = commit_table_streaming(
            &mut VecSource {
                messages: messages.clone(),
                tails: tails.clone(),
            },
            shape,
            configuration,
            &salts,
            2,
            u64::MAX,
        )
        .expect("plain");
        let mut source = VecSource {
            messages: messages.clone(),
            tails: tails.clone(),
        };
        let mut store = MemoryRowStore::new(shape.rows(), shape.code.n()).expect("store");
        let stored = commit_table_stored(
            &mut source,
            &mut store,
            shape,
            configuration,
            &salts,
            2,
            u64::MAX,
        )
        .expect("stored");
        assert_eq!(stored.reader.root, expected.reader.root);
        assert_eq!(stored.name, expected.name);
        assert_eq!(stored.reader.record.y, expected.reader.record.y);
        assert_eq!(
            store.bytes_written(),
            shape.rows() * shape.code.n() * 8
        );
        assert_eq!(store.bytes_read(), shape.rows() * shape.code.n() * 8);
        let positions = [1u64, 4, 6];
        let opened = stored.open_columns(&mut source, &positions).expect("open");
        let want = expected
            .open_columns(
                &mut VecSource {
                    messages: messages.clone(),
                    tails: tails.clone(),
                },
                &positions,
            )
            .expect("plain open");
        assert_eq!(opened, want);
        store.cleanup().expect("cleanup");
        assert!(store.bytes_written() > 0);

        let mut tampered = TamperedStore {
            inner: MemoryRowStore::new(shape.rows(), shape.code.n()).expect("store"),
            tamper: true,
        };
        let mut source = VecSource { messages, tails };
        let changed = commit_table_stored(
            &mut source,
            &mut tampered,
            shape,
            configuration,
            &salts,
            2,
            u64::MAX,
        )
        .expect("tampered commit");
        assert_ne!(changed.reader.root, expected.reader.root);
    }

    #[test]
    fn the_holder_requires_a_reservation_and_enforces_its_budget() {
        let (shape, configuration, messages, tails, salts) = fixture();
        let mut source = VecSource {
            messages: messages.clone(),
            tails: tails.clone(),
        };
        let streamed =
            commit_table_streaming(&mut source, shape, configuration, &salts, 2, u64::MAX)
                .expect("streaming");
        let mut holder = StreamingHolder::new(streamed, source, 4).expect("holder");
        let gamma = vec![Fp3::from_u64(2), Fp3::from_u64(3)];
        assert!(matches!(holder.fold(&gamma), Err(Error::InvalidSessionState)));
        assert!(matches!(
            holder.open(&[1, 4, 6]),
            Err(Error::InvalidSessionState)
        ));
        holder.begin(3).expect("first reservation");
        assert_eq!(holder.hiding_used(), 3);
        assert!(matches!(
            holder.begin(3),
            Err(Error::ResourceLimit)
        ));
        let (folded, _) = holder.fold(&gamma).expect("fold");
        assert!(!folded.is_empty());
        let opened = holder.open(&[1, 4, 6]).expect("open");
        assert_eq!(opened.len(), 3);
        assert!(matches!(
            holder.open(&[1, 4, 6, 7]),
            Err(Error::ResourceLimit)
        ));
        holder.forget().expect("forget");
        assert!(matches!(
            holder.open(&[1]),
            Err(Error::InvalidSessionState)
        ));
    }

    #[test]
    fn a_tampered_source_changes_the_root() {
        let (shape, configuration, messages, tails, salts) = fixture();
        let mut source = VecSource {
            messages: messages.clone(),
            tails: tails.clone(),
        };
        let honest =
            commit_table_streaming(&mut source, shape, configuration, &salts, 2, u64::MAX)
                .expect("honest");
        let mut tampered_messages = messages;
        tampered_messages[1][0] = tampered_messages[1][0] + Fp::from_u64(1);
        let mut tampered = VecSource {
            messages: tampered_messages,
            tails,
        };
        let changed =
            commit_table_streaming(&mut tampered, shape, configuration, &salts, 2, u64::MAX)
                .expect("tampered");
        assert_ne!(changed.reader.root, honest.reader.root);
    }

    #[test]
    fn a_source_with_the_wrong_row_count_is_rejected() {
        let (shape, configuration, messages, tails, salts) = fixture();
        let mut source = VecSource {
            messages: messages[..2].to_vec(),
            tails: tails[..2].to_vec(),
        };
        assert!(matches!(
            commit_table_streaming(&mut source, shape, configuration, &salts, 2, u64::MAX),
            Err(Error::ShapeMismatch)
        ));
    }

    #[test]
    fn a_memory_cap_below_the_buffers_is_rejected() {
        let (shape, configuration, messages, tails, salts) = fixture();
        let mut source = VecSource { messages, tails };
        assert!(matches!(
            commit_table_streaming(&mut source, shape, configuration, &salts, 2, 1),
            Err(Error::ResourceLimit)
        ));
    }
}
