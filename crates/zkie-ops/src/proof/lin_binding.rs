//! The batch binding tree over first-chain seals.
//!
//! A first chain ends in a seal. The binding tree hashes the seal with the
//! plan's binding configuration, the request's place and a per-place salt; its
//! root and place count start the shared chain's parameters. The verifier
//! never reads a root from the proof: it recomputes the leaf from its own
//! verified seal and walks the supplied path to the root, so a wrong seal,
//! place, salt or depth leads to another shared chain.
//!
//! A leaf's index is the place count plus the place, so a path of another
//! depth cannot lead to the same root. This round binds one first chain: one
//! leaf, place zero, an empty sibling path.

use zkie_core::common::codec::DecodeLimits;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::params::{LEAF_SALT_MAX_BYTES, LEAF_SALT_MIN_BYTES};
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};
use zkie_engine::primitives::transcript::{hash_labeled, tag_word, BINDING_NODE, LEAF_BINDING};

pub const LIN_BINDING_FORMAT_VERSION: u64 = 1;
pub const LIN_BINDING_MAX_LOG: u32 = 48;

/// The public shape of one binding tree: its depth and one leaf's salt length.
/// The depth fixes the number of places (`2^log_leaves`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinBindingShape {
    pub log_leaves: u32,
    pub salt_bytes: usize,
}

impl LinBindingShape {
    pub fn places(&self) -> Result<u64> {
        if self.log_leaves > LIN_BINDING_MAX_LOG {
            return Err(Error::InvalidConfig);
        }
        1u64
            .checked_shl(self.log_leaves)
            .ok_or(Error::ResourceLimit)
    }

    pub fn fault(&self) -> Option<&'static str> {
        if self.log_leaves > LIN_BINDING_MAX_LOG {
            return Some("the binding tree is too deep");
        }
        if self.salt_bytes < LEAF_SALT_MIN_BYTES || self.salt_bytes > LEAF_SALT_MAX_BYTES {
            return Some("the binding leaf's salt length is not admitted");
        }
        if self.salt_bytes % 8 != 0 {
            return Some("the binding leaf's salt length is not a multiple of eight");
        }
        None
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(LIN_BINDING_FORMAT_VERSION);
        writer.put_u64(self.log_leaves as u64);
        writer.put_u64(self.salt_bytes as u64);
        writer.into_bytes()
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        if reader.get_u64()? != LIN_BINDING_FORMAT_VERSION {
            return Err(Error::UnsupportedVersion);
        }
        let log_leaves = u32::try_from(reader.get_u64()?).map_err(|_| Error::InvalidEncoding)?;
        let salt_bytes = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        reader.finish()?;
        let shape = Self {
            log_leaves,
            salt_bytes,
        };
        if shape.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        Ok(shape)
    }
}

/// The root of one binding tree with the number of its places.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinBindingRoot {
    pub digest: Digest,
    pub places: u64,
}

impl LinBindingRoot {
    pub fn fault(&self) -> Option<&'static str> {
        if self.digest == Digest::ZERO {
            return Some("the binding root is empty");
        }
        if self.places == 0 || !self.places.is_power_of_two() {
            return Some("the binding root has no power-of-two places");
        }
        if self.places.trailing_zeros() > LIN_BINDING_MAX_LOG {
            return Some("the binding root is deeper than a tree");
        }
        None
    }
}

/// One request's path: its place, the leaf's salt and the siblings from the
/// leaf up. One leaf has an empty sibling list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinBindingPath {
    pub place: u64,
    pub salt: Vec<u8>,
    pub siblings: Vec<Digest>,
}

impl LinBindingPath {
    pub fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(LIN_BINDING_FORMAT_VERSION);
        writer.put_u64(self.place);
        writer.put_bytes(&self.salt);
        writer.put_u64(self.siblings.len() as u64);
        for sibling in &self.siblings {
            writer.put_digest(sibling);
        }
        writer.into_bytes()
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, DecodeLimits::unlimited());
        if reader.get_u64()? != LIN_BINDING_FORMAT_VERSION {
            return Err(Error::UnsupportedVersion);
        }
        let place = reader.get_u64()?;
        let salt = reader.get_bytes()?.to_vec();
        let count = usize::try_from(reader.get_u64()?).map_err(|_| Error::ResourceLimit)?;
        if count > LIN_BINDING_MAX_LOG as usize {
            return Err(Error::InvalidEncoding);
        }
        let mut siblings = Vec::with_capacity(count);
        for _ in 0..count {
            siblings.push(reader.get_digest()?);
        }
        reader.finish()?;
        Ok(Self {
            place,
            salt,
            siblings,
        })
    }
}

/// A leaf hashes its index, the seal and the salt under the binding
/// configuration. The index is the place count plus the place, which binds the
/// tree's depth.
pub fn binding_leaf(
    shape: &LinBindingShape,
    configuration: &Digest,
    place: u64,
    seal: &Digest,
    salt: &[u8],
) -> Result<Digest> {
    if shape.fault().is_some()
        || place >= shape.places()?
        || salt.len() != shape.salt_bytes
    {
        return Err(Error::InvalidConfig);
    }
    let index = shape
        .places()?
        .checked_add(place)
        .ok_or(Error::ResourceLimit)?;
    let mut data = Vec::with_capacity(16 + 32 + salt.len());
    data.extend_from_slice(&(shape.log_leaves as u64).to_le_bytes());
    data.extend_from_slice(&(shape.salt_bytes as u64).to_le_bytes());
    data.extend_from_slice(&seal.0);
    data.extend_from_slice(salt);
    hash_labeled(
        tag_word(LEAF_BINDING, 0),
        index,
        Some(configuration),
        &data,
    )
}

/// An inner vertex hashes its two children under the binding configuration.
pub fn binding_node(
    configuration: &Digest,
    left: &Digest,
    right: &Digest,
) -> Result<Digest> {
    let mut data = Vec::with_capacity(64);
    data.extend_from_slice(&left.0);
    data.extend_from_slice(&right.0);
    hash_labeled(
        tag_word(BINDING_NODE, 0),
        0,
        Some(configuration),
        &data,
    )
}

/// The prover's binding tree: salt and leaves are fixed by the caller, one
/// place is taken by each request's seal, then the root and a request's path
/// are read off. Every place must be taken before the root is read; a batch
/// with fewer requests fills its free places from the same random material
/// (this round binds exactly one request).
pub struct LinBindingTree {
    shape: LinBindingShape,
    configuration: Digest,
    salts: Vec<u8>,
    leaves: Vec<Option<Digest>>,
    levels: Vec<Vec<Digest>>,
    sealed: bool,
}

impl LinBindingTree {
    /// `salts` holds one salt per place, in place order.
    pub fn new(
        shape: LinBindingShape,
        configuration: Digest,
        salts: Vec<u8>,
    ) -> Result<Self> {
        if shape.fault().is_some() || configuration == Digest::ZERO {
            return Err(Error::InvalidConfig);
        }
        let places = usize::try_from(shape.places()?).map_err(|_| Error::ResourceLimit)?;
        if salts.len() != places.checked_mul(shape.salt_bytes).ok_or(Error::ResourceLimit)? {
            return Err(Error::ShapeMismatch);
        }
        Ok(Self {
            shape,
            configuration,
            salts,
            leaves: vec![None; places],
            levels: Vec::new(),
            sealed: false,
        })
    }

    /// Takes one place with the request's seal.
    pub fn place(&mut self, place: u64, seal: &Digest) -> Result<()> {
        if self.sealed {
            return Err(Error::InvalidSessionState);
        }
        let places = self.shape.places()?;
        if place >= places {
            return Err(Error::InvalidIndex);
        }
        let at = usize::try_from(place).map_err(|_| Error::ResourceLimit)?;
        if self.leaves[at].is_some() {
            return Err(Error::InvalidIndex);
        }
        let start = at
            .checked_mul(self.shape.salt_bytes)
            .ok_or(Error::ResourceLimit)?;
        let salt = self
            .salts
            .get(start..start + self.shape.salt_bytes)
            .ok_or(Error::ShapeMismatch)?;
        self.leaves[at] = Some(binding_leaf(
            &self.shape,
            &self.configuration,
            place,
            seal,
            salt,
        )?);
        Ok(())
    }

    /// Builds the tree and returns its root with the place count. Every place
    /// must be taken.
    pub fn root(&mut self) -> Result<LinBindingRoot> {
        if self.levels.is_empty() {
            if self.leaves.iter().any(|leaf| leaf.is_none()) {
                return Err(Error::InvalidConfig);
            }
            let mut level: Vec<Digest> = self
                .leaves
                .iter()
                .map(|leaf| leaf.ok_or(Error::InvalidConfig))
                .collect::<Result<Vec<_>>>()?;
            let mut levels = vec![level.clone()];
            while level.len() > 1 {
                let mut next = Vec::with_capacity(level.len() / 2);
                for pair in level.chunks_exact(2) {
                    next.push(binding_node(
                        &self.configuration,
                        &pair[0],
                        &pair[1],
                    )?);
                }
                levels.push(next.clone());
                level = next;
            }
            self.levels = levels;
        }
        self.sealed = true;
        let digest = self
            .levels
            .last()
            .and_then(|level| level.first())
            .copied()
            .ok_or(Error::InvalidConfig)?;
        Ok(LinBindingRoot {
            digest,
            places: self.shape.places()?,
        })
    }

    /// One request's path: its place, its salt and the siblings from the leaf
    /// up.
    pub fn path(&self, place: u64) -> Result<LinBindingPath> {
        if self.levels.is_empty() {
            return Err(Error::InvalidSessionState);
        }
        let places = self.shape.places()?;
        if place >= places {
            return Err(Error::InvalidIndex);
        }
        let at = usize::try_from(place).map_err(|_| Error::ResourceLimit)?;
        if self.leaves[at].is_none() {
            return Err(Error::InvalidIndex);
        }
        let start = at
            .checked_mul(self.shape.salt_bytes)
            .ok_or(Error::ResourceLimit)?;
        let salt = self
            .salts
            .get(start..start + self.shape.salt_bytes)
            .ok_or(Error::ShapeMismatch)?
            .to_vec();
        let depth = usize::try_from(self.shape.log_leaves).map_err(|_| Error::ResourceLimit)?;
        let mut siblings = Vec::with_capacity(depth);
        let mut at_place = at;
        for level in self.levels.iter().take(depth) {
            let sibling = at_place ^ 1;
            siblings.push(*level.get(sibling).ok_or(Error::ShapeMismatch)?);
            at_place >>= 1;
        }
        Ok(LinBindingPath {
            place,
            salt,
            siblings,
        })
    }
}

/// The root that one request's verified seal and its path lead to. The caller
/// supplies the shape the first chain is bound to; the place must be inside it
/// and the path must have one sibling per level.
pub fn binding_root_from_path(
    shape: &LinBindingShape,
    configuration: &Digest,
    seal: &Digest,
    path: &LinBindingPath,
) -> Result<LinBindingRoot> {
    if shape.fault().is_some() || path.siblings.len() != shape.log_leaves as usize {
        return Err(Error::InvalidConfig);
    }
    let places = shape.places()?;
    if path.place >= places || path.salt.len() != shape.salt_bytes {
        return Err(Error::InvalidEncoding);
    }
    let mut root = binding_leaf(shape, configuration, path.place, seal, &path.salt)?;
    let mut place = path.place;
    for sibling in &path.siblings {
        root = if place & 1 == 1 {
            binding_node(configuration, sibling, &root)?
        } else {
            binding_node(configuration, &root, sibling)?
        };
        place >>= 1;
    }
    Ok(LinBindingRoot {
        digest: root,
        places,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(log_leaves: u32) -> LinBindingShape {
        LinBindingShape {
            log_leaves,
            salt_bytes: 40,
        }
    }

    fn seal(byte: u8) -> Digest {
        Digest([byte; 32])
    }

    fn salt(byte: u8) -> Vec<u8> {
        vec![byte; 40]
    }

    fn configuration() -> Digest {
        Digest([0x11; 32])
    }

    #[test]
    fn one_leaf_binds_its_seal() {
        let shape = shape(0);
        let mut tree =
            LinBindingTree::new(shape, configuration(), salt(7)).unwrap();
        tree.place(0, &seal(1)).unwrap();
        let root = tree.root().unwrap();
        assert_eq!(root.places, 1);
        let path = tree.path(0).unwrap();
        assert_eq!(path.place, 0);
        assert!(path.siblings.is_empty());
        let read = binding_root_from_path(&shape, &configuration(), &seal(1), &path).unwrap();
        assert_eq!(read, root);
        let other = binding_root_from_path(&shape, &configuration(), &seal(2), &path).unwrap();
        assert_ne!(other, root);
        let mut changed = path.clone();
        changed.salt[0] ^= 1;
        let ground = binding_root_from_path(&shape, &configuration(), &seal(1), &changed).unwrap();
        assert_ne!(ground, root);
    }

    #[test]
    fn a_path_reaches_the_root_of_its_tree() {
        let shape = shape(2);
        let mut salts = Vec::new();
        for place in 0..4u8 {
            salts.extend_from_slice(&salt(0x20 + place));
        }
        let mut tree = LinBindingTree::new(shape, configuration(), salts).unwrap();
        for place in 0..4u64 {
            tree.place(place, &seal(0x40 + place as u8)).unwrap();
        }
        let root = tree.root().unwrap();
        assert_eq!(root.places, 4);
        for place in 0..4u64 {
            let path = tree.path(place).unwrap();
            assert_eq!(path.siblings.len(), 2);
            let read = binding_root_from_path(
                &shape,
                &configuration(),
                &seal(0x40 + place as u8),
                &path,
            )
            .unwrap();
            assert_eq!(read, root);
        }
        let mut changed = tree.path(2).unwrap();
        changed.siblings[0] = seal(0x55);
        let ground = binding_root_from_path(&shape, &configuration(), &seal(0x42), &changed).unwrap();
        assert_ne!(ground, root);
    }

    #[test]
    fn depth_and_place_are_bound_into_the_leaf() {
        let shallow = binding_leaf(&shape(0), &configuration(), 0, &seal(1), &salt(7)).unwrap();
        let deep = binding_leaf(&shape(1), &configuration(), 0, &seal(1), &salt(7)).unwrap();
        let another_place =
            binding_leaf(&shape(1), &configuration(), 1, &seal(1), &salt(7)).unwrap();
        assert_ne!(shallow, deep);
        assert_ne!(deep, another_place);
        assert!(binding_leaf(&shape(1), &configuration(), 2, &seal(1), &salt(7)).is_err());
        assert!(binding_leaf(&shape(0), &configuration(), 0, &seal(1), &salt(7)[..32]).is_err());
    }

    #[test]
    fn a_tree_refuses_an_unfilled_place_and_a_second_place() {
        let shape = shape(1);
        let mut salts = salt(1);
        salts.extend_from_slice(&salt(2));
        let mut tree = LinBindingTree::new(shape, configuration(), salts).unwrap();
        tree.place(0, &seal(1)).unwrap();
        assert!(tree.place(0, &seal(3)).is_err());
        assert!(tree.root().is_err());
        tree.place(1, &seal(2)).unwrap();
        assert!(tree.root().is_ok());
    }

    #[test]
    fn shape_and_path_round_trip() {
        let shape = shape(1);
        let encoded = shape.encode();
        assert_eq!(LinBindingShape::decode(&encoded).unwrap(), shape);
        let path = LinBindingPath {
            place: 1,
            salt: salt(3),
            siblings: vec![seal(4)],
        };
        let encoded = path.encode();
        assert_eq!(LinBindingPath::decode(&encoded).unwrap(), path);
        let mut bad = shape;
        bad.salt_bytes = 24;
        assert!(bad.fault().is_some());
        let mut deep = shape;
        deep.log_leaves = 49;
        assert!(deep.fault().is_some());
        assert!(deep.places().is_err());
    }
}
