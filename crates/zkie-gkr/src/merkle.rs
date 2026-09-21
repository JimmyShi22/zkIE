//! Merkle tree over field elements.
//!
//! The hash here is `std`'s `DefaultHasher` (SipHash) purely as a deterministic
//! placeholder for the PoC. Production must use a collision-resistant field
//! hash (Poseidon / Rescue) or a native hash (SHA-256 / Blake3); the tree
//! structure and opening logic are identical either way.

use crate::field::F64;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub type Digest = u64;

pub fn hash_leaf(v: F64) -> Digest {
    let mut h = DefaultHasher::new();
    v.val().hash(&mut h);
    h.finish()
}

pub fn hash_pair(a: Digest, b: Digest) -> Digest {
    let mut h = DefaultHasher::new();
    a.hash(&mut h);
    b.hash(&mut h);
    h.finish()
}

pub fn commit(values: &[F64]) -> Digest {
    let n = values.len();
    assert!(n.is_power_of_two(), "merkle commit needs a power-of-two length");
    let mut level: Vec<Digest> = values.iter().map(|&v| hash_leaf(v)).collect();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len() / 2);
        for i in 0..level.len() / 2 {
            next.push(hash_pair(level[2 * i], level[2 * i + 1]));
        }
        level = next;
    }
    level[0]
}

#[derive(Clone, Debug)]
pub struct Opening {
    pub index: usize,
    pub value: F64,
    pub siblings: Vec<Digest>,
}

pub fn open(values: &[F64], index: usize) -> Opening {
    let n = values.len();
    assert!(index < n);
    assert!(n.is_power_of_two());
    let mut level: Vec<Digest> = values.iter().map(|&v| hash_leaf(v)).collect();
    let mut siblings = Vec::new();
    let mut idx = index;
    while level.len() > 1 {
        siblings.push(level[idx ^ 1]);
        let mut next = Vec::with_capacity(level.len() / 2);
        for i in 0..level.len() / 2 {
            next.push(hash_pair(level[2 * i], level[2 * i + 1]));
        }
        level = next;
        idx >>= 1;
    }
    Opening {
        index,
        value: values[index],
        siblings,
    }
}

pub fn verify(root: Digest, opening: &Opening) -> bool {
    let mut h = hash_leaf(opening.value);
    let mut idx = opening.index;
    for &sib in &opening.siblings {
        h = if idx & 1 == 0 {
            hash_pair(h, sib)
        } else {
            hash_pair(sib, h)
        };
        idx >>= 1;
    }
    h == root
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::XorShift64;

    #[test]
    fn merkle_open_verify() {
        let mut rng = XorShift64::new(9);
        let values: Vec<F64> = (0..64).map(|_| rng.field()).collect();
        let root = commit(&values);
        for index in [0usize, 1, 31, 32, 63] {
            let opening = open(&values, index);
            assert!(verify(root, &opening));
            let mut bad = opening.clone();
            bad.value = bad.value + F64::ONE;
            assert!(!verify(root, &bad));
        }
    }
}
