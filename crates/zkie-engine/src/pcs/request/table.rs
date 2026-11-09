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
use crate::pcs::request::field::{Fp, Fp3};
use crate::pcs::request::params::{LEAF_SALT_MAX_BYTES, LEAF_SALT_MIN_BYTES};
use crate::primitives::codec::{CanonicalReader, CanonicalWriter};
use crate::primitives::transcript::{hash_labeled, tag_word, LEAF_STORED_MAIN};

/// The label of the chain that derives a table's record seeds.
pub const RECORD_SEED_LABEL: u64 = 1 << 32;
/// Degree of the extension field: a pad is one element of the base field per
/// coefficient.
pub const E_DEGREE: usize = 3;

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
    pub fn rows(&self) -> u64 {
        1u64 << self.v
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

pub fn validate_index_shape(shape: &IndexShape) -> Result<()> {
    if index_fault(shape).is_some() {
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

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        let mut y = Vec::new();
        while !reader.remaining().is_empty() {
            y.push(reader.get_fp3()?);
        }
        Ok(Self { y })
    }
}

pub fn record_fault(shape: &IndexShape, record: &IndexRecord) -> Option<&'static str> {
    if record.y.len() as u64 != shape.rows() * shape.answers as u64 {
        return Some("the record has not one answer for each row and seed");
    }
    None
}

/// A table's statement item: its commitment bytes and record answers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexName {
    pub commitment: Vec<u8>,
    pub record: IndexRecord,
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
/// round match the reference chain; the seed values are drawn from the
/// repository's chain layout.
pub fn record_seeds(shape: &IndexShape, name: &[u8]) -> Result<Vec<SampleSeed>> {
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
    let mut seeds = Vec::with_capacity(shape.answers as usize);
    for _ in 0..shape.answers {
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
        if self.configuration == Digest::ZERO {
            return Some("the key names no configuration");
        }
        record_fault(&self.shape, &self.record)
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

fn column_leaf(
    configuration: &Digest,
    position: u64,
    salt: &[u8],
    symbols: &[Fp],
) -> Result<Digest> {
    let mut writer = CanonicalWriter::new();
    writer.put_bytes(salt);
    for value in symbols {
        writer.put_fp(*value);
    }
    hash_labeled(
        tag_word(LEAF_STORED_MAIN, 0),
        position,
        Some(configuration),
        &writer.into_bytes(),
    )
}

/// A committed table, with the private rows and salts needed to open columns.
pub struct CommittedTable {
    pub reader: IndexReader,
    pub name: Vec<u8>,
    tree: MerkleTree,
    salts: Vec<Vec<u8>>,
    codewords: Vec<Vec<Fp>>,
}

impl CommittedTable {
    /// Open the columns at `positions` (drawn by the chain; the caller keeps
    /// them sorted and without repeats).
    pub fn open_columns(&self, positions: &[u64]) -> Result<Vec<ColumnOpening>> {
        let n = self.reader.shape.code.n();
        let rows = self.reader.shape.rows() as usize;
        let mut openings = Vec::with_capacity(positions.len());
        let mut previous = None;
        for position in positions {
            if *position >= n {
                return Err(Error::InvalidIndex);
            }
            if let Some(previous) = previous {
                if *position <= previous {
                    return Err(Error::InvalidIndex);
                }
            }
            previous = Some(*position);
            let mut symbols = Vec::with_capacity(rows);
            for codeword in &self.codewords {
                symbols.push(codeword[*position as usize]);
            }
            openings.push(ColumnOpening {
                position: *position,
                salt: self.salts[*position as usize].clone(),
                symbols,
                path: self.tree.open(*position as usize)?,
            });
        }
        Ok(openings)
    }
}

/// Commit a table from the rows' messages (each `ell` entries: plain entries,
/// then the answers' pads) and random tails (each `r` entries). `salts` holds
/// one salt per column, `salt_bytes` long.
pub fn commit_table(
    shape: IndexShape,
    configuration: Digest,
    messages: &[Vec<Fp>],
    tails: &[Vec<Fp>],
    salts: &[Vec<u8>],
) -> Result<CommittedTable> {
    validate_index_shape(&shape)?;
    if configuration == Digest::ZERO {
        return Err(Error::InvalidConfig);
    }
    let rows = shape.rows() as usize;
    let n = shape.code.n() as usize;
    if messages.len() != rows || tails.len() != rows || salts.len() != n {
        return Err(Error::ShapeMismatch);
    }
    for message in messages {
        if message.len() as u64 != shape.code.ell {
            return Err(Error::ShapeMismatch);
        }
    }
    for tail in tails {
        if tail.len() as u64 != shape.code.r {
            return Err(Error::ShapeMismatch);
        }
    }
    for salt in salts {
        if salt.len() != shape.salt_bytes {
            return Err(Error::ShapeMismatch);
        }
    }

    let mut codewords = Vec::with_capacity(rows);
    for (message, tail) in messages.iter().zip(tails) {
        codewords.push(encode_row_fp(&shape.code, message, tail)?);
    }

    let mut leaves = Vec::with_capacity(n);
    for position in 0..n {
        let mut symbols = Vec::with_capacity(rows);
        for codeword in &codewords {
            symbols.push(codeword[position]);
        }
        leaves.push(column_leaf(
            &configuration,
            position as u64,
            &salts[position],
            &symbols,
        )?);
    }
    let tree = merkle_build(&configuration, &leaves)?;
    let root = tree.root;
    let name = index_name(&root, &shape, &configuration);
    let seeds = record_seeds(&shape, &name)?;
    let mut record = IndexRecord {
        y: Vec::with_capacity(rows * shape.answers as usize),
    };
    for message in messages {
        for answer in 0..shape.answers {
            record.y.push(record_value(
                &shape,
                &seeds[answer as usize],
                answer,
                message,
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
    Ok(CommittedTable {
        reader,
        name,
        tree,
        salts: salts.to_vec(),
        codewords,
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
    let mut columns = Vec::with_capacity(positions.len() * rows);
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
        let leaf = column_leaf(
            &reader.configuration,
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
    use crate::pcs::request::code::{build_code_plan, symbol_at_fp};

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

    fn committed() -> CommittedTable {
        let shape = shape();
        let configuration = Digest([7; 32]);
        commit_table(
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
        let first = commit_table(
            shape,
            Digest([7; 32]),
            &messages,
            &tails,
            &salts,
        )
        .unwrap();
        let second = commit_table(
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
        let renamed = commit_table(
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
        let first = record_seeds(&shape, name).unwrap();
        let second = record_seeds(&shape, name).unwrap();
        assert_eq!(first, second);
        let other = record_seeds(&shape, b"other-name").unwrap();
        assert_ne!(first, other);
        assert!(!first[0].beta.is_zero());
    }

    #[test]
    fn record_round_trip_and_length_check() {
        let table = committed();
        let bytes = table.reader.record.encode();
        let decoded = IndexRecord::decode(&bytes).unwrap();
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
        assert!(commit_table(shape, Digest::ZERO, &messages, &tails, &salts).is_err());
        assert!(commit_table(
            shape,
            Digest([7; 32]),
            &messages[..3],
            &tails,
            &salts
        )
        .is_err());
        assert!(commit_table(
            shape,
            Digest([7; 32]),
            &messages,
            &tails,
            &salts[..7]
        )
        .is_err());
        let mut short_message = messages.clone();
        short_message[0].pop();
        assert!(commit_table(shape, Digest([7; 32]), &short_message, &tails, &salts).is_err());
        let mut short_salt = salts.clone();
        short_salt[0].pop();
        assert!(commit_table(shape, Digest([7; 32]), &messages, &tails, &short_salt).is_err());
    }
}
