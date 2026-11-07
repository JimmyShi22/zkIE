//! Column commitments: one leaf per codeword position, one tree per table.
//!
//! A leaf binds the table through the tree's start digest, the column position
//! through the leaf index, and the column content through the canonical symbol
//! encoding. Openings check the column length, the index, the salt shape and
//! the authentication path.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::compute::cpu::{leaf_tag, merkle_build, merkle_leaf, merkle_verify, MerkleTree};
use crate::pcs::request::params::REQUEST_CHAIN_LABEL;
use crate::primitives::transcript::{hash_labeled, run_value, tag_word, LEAF_MAIN};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnTreeSpec {
    pub table_identity: Digest,
    pub parameters_digest: Digest,
    pub tree: u16,
    pub symbol_count: usize,
    pub salt_bytes: usize,
}

impl ColumnTreeSpec {
    pub fn new(
        table_identity: Digest,
        parameters_digest: Digest,
        tree: u16,
        symbol_count: usize,
        salt_bytes: usize,
    ) -> Result<Self> {
        if symbol_count == 0 || symbol_count >= (1 << 24) {
            return Err(Error::InvalidConfig);
        }
        if salt_bytes < 40 || salt_bytes > 64 || salt_bytes % 8 != 0 {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            table_identity,
            parameters_digest,
            tree,
            symbol_count,
            salt_bytes,
        })
    }

    /// The leaves of one table start from the same digest. The table identity
    /// is bound here, not added to the leaf bytes.
    pub fn start(&self) -> Result<Digest> {
        run_value(
            REQUEST_CHAIN_LABEL,
            &self.parameters_digest,
            &self.table_identity,
        )
    }

    pub fn leaf_tag(&self) -> Result<u64> {
        leaf_tag(
            LEAF_MAIN,
            self.tree,
            self.symbol_count as u64,
            self.salt_bytes,
        )
    }
}

#[derive(Clone, Debug)]
pub struct ColumnLeaf {
    pub index: u64,
    pub symbols: Vec<u64>,
    pub salt: Vec<u8>,
    pub digest: Digest,
    pub path: Vec<Digest>,
}

#[derive(Clone, Debug)]
pub struct ColumnTree {
    pub spec: ColumnTreeSpec,
    pub leaves: Vec<ColumnLeaf>,
    pub root: Digest,
    pub start: Digest,
    pub structure: MerkleTree,
}

#[derive(Clone, Debug)]
pub struct ColumnOpening {
    pub spec: ColumnTreeSpec,
    pub root: Digest,
    pub leaves: Vec<ColumnLeaf>,
}

pub fn hash_column(
    spec: &ColumnTreeSpec,
    index: u64,
    symbols: &[u64],
    salt: &[u8],
) -> Result<Digest> {
    if symbols.len() != spec.symbol_count || salt.len() != spec.salt_bytes {
        return Err(Error::ShapeMismatch);
    }
    merkle_leaf(spec.leaf_tag()?, index, &spec.start()?, symbols, salt)
}

/// Build the column tree. All leaves must have the planned shape.
pub fn build_column_tree(
    spec: &ColumnTreeSpec,
    columns: &[(Vec<u64>, Vec<u8>)],
) -> Result<ColumnTree> {
    if columns.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let start = spec.start()?;
    let tag = spec.leaf_tag()?;
    let mut digests = Vec::with_capacity(columns.len());
    for (index, (symbols, salt)) in columns.iter().enumerate() {
        if symbols.len() != spec.symbol_count || salt.len() != spec.salt_bytes {
            return Err(Error::ShapeMismatch);
        }
        digests.push(merkle_leaf(tag, index as u64, &start, symbols, salt)?);
    }
    let structure = merkle_build(&start, &digests)?;
    let mut leaves = Vec::with_capacity(columns.len());
    for (index, digest) in digests.iter().enumerate() {
        leaves.push(ColumnLeaf {
            index: index as u64,
            symbols: columns[index].0.clone(),
            salt: columns[index].1.clone(),
            digest: *digest,
            path: structure.open(index)?,
        });
    }
    Ok(ColumnTree {
        spec: spec.clone(),
        leaves,
        root: structure.root,
        start,
        structure,
    })
}

pub fn open_columns(tree: &ColumnTree, indices: &[u64]) -> Result<ColumnOpening> {
    let mut selected = Vec::with_capacity(indices.len());
    let mut seen = std::collections::BTreeSet::new();
    for index in indices {
        if *index >= tree.leaves.len() as u64 {
            return Err(Error::InvalidIndex);
        }
        if !seen.insert(*index) {
            return Err(Error::InvalidIndex);
        }
        selected.push(tree.leaves[*index as usize].clone());
    }
    Ok(ColumnOpening {
        spec: tree.spec.clone(),
        root: tree.root,
        leaves: selected,
    })
}

/// Verify a set of column openings against an expected root and shape.
pub fn verify_columns(
    opening: &ColumnOpening,
    expected_root: &Digest,
    expected_symbols: usize,
    expected_salt_bytes: usize,
) -> Result<()> {
    if &opening.root != expected_root {
        return Err(Error::ArtifactIntegrityMismatch);
    }
    if opening.spec.symbol_count != expected_symbols
        || opening.spec.salt_bytes != expected_salt_bytes
    {
        return Err(Error::ShapeMismatch);
    }
    let start = opening.spec.start()?;
    let tag = opening.spec.leaf_tag()?;
    let mut seen = std::collections::BTreeSet::new();
    for leaf in &opening.leaves {
        if leaf.symbols.len() != expected_symbols || leaf.salt.len() != expected_salt_bytes {
            return Err(Error::ShapeMismatch);
        }
        if !seen.insert(leaf.index) {
            return Err(Error::InvalidIndex);
        }
        let digest = merkle_leaf(tag, leaf.index, &start, &leaf.symbols, &leaf.salt)?;
        if digest != leaf.digest {
            return Err(Error::ArtifactIntegrityMismatch);
        }
        if !merkle_verify(&start, expected_root, leaf.index as usize, &digest, &leaf.path)? {
            return Err(Error::InvalidProof);
        }
    }
    Ok(())
}

/// The digest of a table's column root, used inside commit records.
pub fn column_root_digest(spec: &ColumnTreeSpec, root: &Digest) -> Result<Digest> {
    let mut data = Vec::with_capacity(64);
    data.extend_from_slice(&spec.table_identity.0);
    data.extend_from_slice(&root.0);
    hash_labeled(tag_word(crate::primitives::transcript::STATEMENT, 0), 0, None, &data)
}

impl ColumnOpening {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut writer = crate::primitives::codec::CanonicalWriter::new();
        writer.put_u64(1);
        writer.put_digest(&self.spec.table_identity);
        writer.put_digest(&self.spec.parameters_digest);
        writer.put_u64(self.spec.tree as u64);
        writer.put_u64(self.spec.symbol_count as u64);
        writer.put_u64(self.spec.salt_bytes as u64);
        writer.put_digest(&self.root);
        writer.put_u64(self.leaves.len() as u64);
        for leaf in &self.leaves {
            writer.put_u64(leaf.index);
            writer.put_u64(leaf.symbols.len() as u64);
            for symbol in &leaf.symbols {
                writer.put_u64(*symbol);
            }
            writer.put_bytes(&leaf.salt);
            writer.put_digest(&leaf.digest);
            writer.put_u64(leaf.path.len() as u64);
            for digest in &leaf.path {
                writer.put_digest(digest);
            }
        }
        Ok(writer.into_bytes())
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut reader = crate::primitives::codec::CanonicalReader::new(
            bytes,
            zkie_core::common::codec::DecodeLimits::unlimited(),
        );
        if reader.get_u64()? != 1 {
            return Err(Error::UnsupportedVersion);
        }
        let table_identity = reader.get_digest()?;
        let parameters_digest = reader.get_digest()?;
        let tree = u16::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?;
        let symbol_count =
            usize::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?;
        let salt_bytes = usize::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?;
        let root = reader.get_digest()?;
        let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?;
        if count > 4096 {
            return Err(Error::InvalidEncoding);
        }
        let mut leaves = Vec::with_capacity(count);
        for _ in 0..count {
            let index = reader.get_u64()?;
            let symbols_len =
                usize::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?;
            if symbols_len != symbol_count {
                return Err(Error::ShapeMismatch);
            }
            let mut symbols = Vec::with_capacity(symbols_len);
            for _ in 0..symbols_len {
                symbols.push(reader.get_u64()?);
            }
            let salt = reader.get_bytes()?.to_vec();
            if salt.len() != salt_bytes {
                return Err(Error::ShapeMismatch);
            }
            let digest = reader.get_digest()?;
            let path_len =
                usize::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?;
            if path_len > 64 {
                return Err(Error::InvalidEncoding);
            }
            let mut path = Vec::with_capacity(path_len);
            for _ in 0..path_len {
                path.push(reader.get_digest()?);
            }
            leaves.push(ColumnLeaf {
                index,
                symbols,
                salt,
                digest,
                path,
            });
        }
        reader.finish()?;
        Ok(Self {
            spec: ColumnTreeSpec::new(
                table_identity,
                parameters_digest,
                tree,
                symbol_count,
                salt_bytes,
            )?,
            root,
            leaves,
        })
    }
}
