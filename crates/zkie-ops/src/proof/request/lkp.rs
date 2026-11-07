//! Product-forest lookup.
//!
//! Query factors and table factors are multiplied into two product forests.
//! Their roots must agree; a repeated table entry is handled by its exponent.
//! Unit leaves pad each forest to a power of two.

use std::collections::BTreeMap;

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::chain::{Chain, KIND_ALGEBRAIC};
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::primitives::codec::CanonicalWriter;
use zkie_engine::primitives::transcript::{hash_labeled, tag_word, STATEMENT};

use crate::proof::binding::{
    verify_opening, OracleId, OracleOpening, OracleRegistry,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupTable {
    pub id: Digest,
    pub rows: Vec<Vec<u64>>,
}

impl LookupTable {
    pub fn new(rows: Vec<Vec<u64>>) -> Result<Self> {
        if rows.is_empty() {
            return Err(Error::InvalidConfig);
        }
        let width = rows[0].len();
        if width == 0 || rows.iter().any(|row| row.len() != width) {
            return Err(Error::ShapeMismatch);
        }
        let mut writer = CanonicalWriter::new();
        writer.put_u64(width as u64);
        writer.put_u64(rows.len() as u64);
        for row in &rows {
            for value in row {
                writer.put_u64(*value);
            }
        }
        let id = hash_labeled(tag_word(STATEMENT, 7), 0, None, &writer.into_bytes())?;
        Ok(Self { id, rows })
    }

    pub fn width(&self) -> usize {
        self.rows.first().map(Vec::len).unwrap_or(0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductForest {
    pub levels: Vec<Vec<Fp3>>,
    pub root: Fp3,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupProof {
    pub gamma: Fp3,
    pub padding_factor: Fp3,
    pub query: ProductForest,
    pub table: ProductForest,
}

fn compress(tuple: &[u64], gamma: Fp3) -> Fp3 {
    let mut value = Fp3::ZERO;
    for word in tuple.iter().rev() {
        value = value.mul(&gamma).add(&Fp3::from_u64(*word));
    }
    value
}

fn product_forest(mut leaves: Vec<Fp3>) -> ProductForest {
    if leaves.is_empty() {
        leaves.push(Fp3::ONE);
    }
    let mut padded = 1usize;
    while padded < leaves.len() {
        padded *= 2;
    }
    leaves.resize(padded, Fp3::ONE);
    let mut levels = vec![leaves];
    while levels.last().expect("one level").len() > 1 {
        let previous = levels.last().expect("one level");
        let mut next = Vec::with_capacity(previous.len() / 2);
        for pair in previous.chunks_exact(2) {
            next.push(pair[0].mul(&pair[1]));
        }
        levels.push(next);
    }
    let root = levels.last().expect("one level")[0];
    ProductForest { levels, root }
}

fn challenge(
    chain: &mut Chain,
    table_id: &Digest,
    registry_digest: &Digest,
    query_count: u64,
    padding_count: u64,
) -> Result<Fp3> {
    let mut message = chain.message();
    for word in table_id.0.chunks_exact(8) {
        message.push_u64(u64::from_le_bytes(
            word.try_into().map_err(|_| Error::InvalidEncoding)?,
        ));
    }
    for word in registry_digest.0.chunks_exact(8) {
        message.push_u64(u64::from_le_bytes(
            word.try_into().map_err(|_| Error::InvalidEncoding)?,
        ));
    }
    message.push_u64(query_count);
    message.push_u64(padding_count);
    let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    chain.draw_e()
}

/// The multiplicities of the queried tuples. Every queried tuple must occur in
/// the table; the table side of the forest uses these multiplicities, so the
/// two roots are the same multiset product.
fn query_multiplicities(
    table: &LookupTable,
    queries: &[OracleOpening],
    padding_keys: &[Vec<u64>],
    gamma: Fp3,
) -> Result<BTreeMap<Vec<u64>, u64>> {
    let mut counts: BTreeMap<Vec<u64>, u64> = BTreeMap::new();
    for query in queries {
        if !table.rows.iter().any(|row| row == &query.values) {
            return Err(Error::InvalidProof);
        }
        if padding_keys.iter().any(|key| key == &query.values) {
            continue;
        }
        let entry = counts.entry(query.values.clone()).or_insert(0);
        *entry = entry.checked_add(1).ok_or(Error::ResourceLimit)?;
    }
    let _ = gamma;
    Ok(counts)
}

fn check_openings(
    queries: &[OracleOpening],
    registry: &OracleRegistry,
    width: usize,
) -> Result<()> {
    for query in queries {
        if query.values.len() != width {
            return Err(Error::ShapeMismatch);
        }
        let root = registry.get(query.oracle).ok_or(Error::ProverStateMissing)?;
        verify_opening(registry, query, root.count)?;
    }
    Ok(())
}

pub fn prove_lookup(
    chain: &mut Chain,
    table: &LookupTable,
    queries: &[OracleOpening],
    registry: &OracleRegistry,
) -> Result<LookupProof> {
    prove_lookup_with_padding(chain, table, queries, &[], registry)
}

pub fn prove_lookup_with_padding(
    chain: &mut Chain,
    table: &LookupTable,
    queries: &[OracleOpening],
    padding_keys: &[Vec<u64>],
    registry: &OracleRegistry,
) -> Result<LookupProof> {
    check_openings(queries, registry, table.width())?;
    for key in padding_keys {
        if key.len() != table.width() || !table.rows.iter().any(|row| row == key) {
            return Err(Error::InvalidProof);
        }
    }
    let gamma = challenge(
        chain,
        &table.id,
        &registry.digest(),
        queries.len() as u64,
        padding_keys.len() as u64,
    )?;
    let mut query_leaves = Vec::with_capacity(queries.len());
    for query in queries {
        let factor = gamma.sub(&compress(&query.values, gamma));
        if factor.is_zero() {
            return Err(Error::InvalidProof);
        }
        query_leaves.push(factor);
    }
    let mut padding_factor = Fp3::ONE;
    for key in padding_keys {
        padding_factor = padding_factor.mul(&gamma.sub(&compress(key, gamma)));
    }
    let counts = query_multiplicities(table, queries, padding_keys, gamma)?;
    let mut table_leaves = Vec::with_capacity(counts.len());
    for (row, count) in counts {
        let base = gamma.sub(&compress(&row, gamma));
        if base.is_zero() {
            return Err(Error::InvalidProof);
        }
        let mut factor = Fp3::ONE;
        for _ in 0..count {
            factor = factor.mul(&base);
        }
        table_leaves.push(factor);
    }
    let query = product_forest(query_leaves);
    let table_forest = product_forest(table_leaves);
    if query.root != padding_factor.mul(&table_forest.root) {
        return Err(Error::InvalidProof);
    }
    Ok(LookupProof {
        gamma,
        padding_factor,
        query,
        table: table_forest,
    })
}

fn same_forests(expected: &ProductForest, proof: &ProductForest) -> bool {
    expected.levels == proof.levels && expected.root == proof.root
}

pub fn verify_lookup(
    chain: &mut Chain,
    table: &LookupTable,
    queries: &[OracleOpening],
    registry: &OracleRegistry,
    proof: &LookupProof,
) -> Result<()> {
    verify_lookup_with_padding(chain, table, queries, &[], registry, proof)
}

pub fn verify_lookup_with_padding(
    chain: &mut Chain,
    table: &LookupTable,
    queries: &[OracleOpening],
    padding_keys: &[Vec<u64>],
    registry: &OracleRegistry,
    proof: &LookupProof,
) -> Result<()> {
    check_openings(queries, registry, table.width())?;
    for key in padding_keys {
        if key.len() != table.width() || !table.rows.iter().any(|row| row == key) {
            return Err(Error::InvalidProof);
        }
    }
    let gamma = challenge(
        chain,
        &table.id,
        &registry.digest(),
        queries.len() as u64,
        padding_keys.len() as u64,
    )?;
    if gamma != proof.gamma {
        return Err(Error::InvalidProof);
    }
    let mut query_leaves = Vec::with_capacity(queries.len());
    for query in queries {
        let factor = gamma.sub(&compress(&query.values, gamma));
        if factor.is_zero() {
            return Err(Error::InvalidProof);
        }
        query_leaves.push(factor);
    }
    let mut padding_factor = Fp3::ONE;
    for key in padding_keys {
        padding_factor = padding_factor.mul(&gamma.sub(&compress(key, gamma)));
    }
    if padding_factor != proof.padding_factor {
        return Err(Error::InvalidProof);
    }
    let counts = query_multiplicities(table, queries, padding_keys, gamma)?;
    let mut table_leaves = Vec::with_capacity(counts.len());
    for (row, count) in counts {
        let base = gamma.sub(&compress(&row, gamma));
        if base.is_zero() {
            return Err(Error::InvalidProof);
        }
        let mut factor = Fp3::ONE;
        for _ in 0..count {
            factor = factor.mul(&base);
        }
        table_leaves.push(factor);
    }
    let expected_query = product_forest(query_leaves);
    let expected_table = product_forest(table_leaves);
    if !same_forests(&expected_query, &proof.query) || !same_forests(&expected_table, &proof.table)
    {
        return Err(Error::InvalidProof);
    }
    if expected_query.root != padding_factor.mul(&expected_table.root) {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// Open one row of the table for use as a verification key or public input.
pub fn table_id(table: &LookupTable) -> Digest {
    table.id
}

/// A helper for relation ids that bind a table into a statement.
pub fn oracle_label(prefix: u64) -> Digest {
    let mut writer = CanonicalWriter::new();
    writer.put_u64(prefix);
    hash_labeled(tag_word(STATEMENT, 8), prefix, None, &writer.into_bytes())
        .unwrap_or(Digest::ZERO)
}

#[allow(dead_code)]
fn _oracle_id_marker(_: OracleId) {}

/// Multiplicity rows `(tuple, count)` in canonical sorted order, excluding
/// padding keys. Each row is encoded as the tuple words followed by the count.
pub fn multiplicity_rows(
    table: &LookupTable,
    queries: &[OracleOpening],
    padding_keys: &[Vec<u64>],
) -> Result<Vec<Vec<u64>>> {
    let counts = query_multiplicities(table, queries, padding_keys, Fp3::ONE)?;
    let mut rows = Vec::with_capacity(counts.len());
    for (key, count) in counts {
        let mut row = key;
        row.push(count);
        rows.push(row);
    }
    Ok(rows)
}
