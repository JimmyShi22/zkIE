//! Batch sharing: freeze members and their order, then combine their claims
//! under one challenge drawn from the shared chain.

use std::collections::BTreeSet;

use zkie_core::claim::RelationId;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::chain::{Chain, KIND_ALGEBRAIC};
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::primitives::codec::CanonicalWriter;
use zkie_engine::primitives::transcript::{hash_labeled, tag_word, STATEMENT};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealedMember {
    pub request_id: Digest,
    pub seal: Digest,
    pub claims: Vec<(RelationId, Fp3)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrozenBatch {
    pub members: Vec<SealedMember>,
    pub root: Digest,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedReceipt {
    pub root: Digest,
    pub challenge: Fp3,
    pub combined: Vec<(RelationId, Fp3)>,
}

pub fn batch_root(members: &[SealedMember]) -> Result<Digest> {
    if members.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let mut writer = CanonicalWriter::new();
    writer.put_u64(members.len() as u64);
    for member in members {
        writer.put_digest(&member.request_id);
        writer.put_digest(&member.seal);
        writer.put_u64(member.claims.len() as u64);
        for (relation, value) in &member.claims {
            writer.put_u64(relation.0);
            writer.put_u64(value.encode().len() as u64);
            writer.put_bytes(&value.encode());
        }
    }
    hash_labeled(tag_word(STATEMENT, 10), 0, None, &writer.into_bytes())
}

pub fn freeze_batch(members: Vec<SealedMember>) -> Result<FrozenBatch> {
    if members.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let mut request_ids = BTreeSet::new();
    let mut seals = BTreeSet::new();
    for member in &members {
        if !request_ids.insert(member.request_id) || !seals.insert(member.seal) {
            return Err(Error::StatementMismatch);
        }
    }
    let root = batch_root(&members)?;
    Ok(FrozenBatch { members, root })
}

pub fn shared_challenge(chain: &mut Chain, batch: &FrozenBatch) -> Result<Fp3> {
    let mut message = chain.message();
    for word in batch.root.0.chunks_exact(8) {
        message.push_u64(u64::from_le_bytes(
            word.try_into().map_err(|_| Error::InvalidEncoding)?,
        ));
    }
    message.push_u64(batch.members.len() as u64);
    message.push_u64(batch.root.0[0] as u64);
    let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    chain.draw_e()
}

fn relation_list(members: &[SealedMember]) -> Result<Vec<RelationId>> {
    let first = members.first().ok_or(Error::InvalidConfig)?;
    let relations: Vec<RelationId> = first.claims.iter().map(|(relation, _)| *relation).collect();
    let mut seen = BTreeSet::new();
    if relations.is_empty() || !relations.iter().all(|relation| seen.insert(*relation)) {
        return Err(Error::StatementMismatch);
    }
    for member in members.iter().skip(1) {
        let ids: Vec<RelationId> = member.claims.iter().map(|(relation, _)| *relation).collect();
        if ids != relations {
            return Err(Error::StatementMismatch);
        }
    }
    Ok(relations)
}

pub fn combine_claims(
    members: &[SealedMember],
    challenge: Fp3,
) -> Result<Vec<(RelationId, Fp3)>> {
    let relations = relation_list(members)?;
    let mut combined = Vec::with_capacity(relations.len());
    for (position, relation) in relations.into_iter().enumerate() {
        let mut power = Fp3::ONE;
        let mut sum = Fp3::ZERO;
        for member in members {
            sum = sum.add(&power.mul(&member.claims[position].1));
            power = power.mul(&challenge);
        }
        combined.push((relation, sum));
    }
    Ok(combined)
}

pub fn prove_shared(chain: &mut Chain, batch: &FrozenBatch) -> Result<SharedReceipt> {
    if batch_root(&batch.members)? != batch.root {
        return Err(Error::StatementMismatch);
    }
    let challenge = shared_challenge(chain, batch)?;
    let combined = combine_claims(&batch.members, challenge)?;
    Ok(SharedReceipt {
        root: batch.root,
        challenge,
        combined,
    })
}

/// The verifier receives the frozen root from an earlier public input and
/// rebuilds every relation from the public member list.
pub fn verify_shared(
    chain: &mut Chain,
    expected_root: &Digest,
    batch: &FrozenBatch,
    receipt: &SharedReceipt,
    required_relations: &[RelationId],
) -> Result<()> {
    if batch.root != *expected_root || batch_root(&batch.members)? != *expected_root {
        return Err(Error::StatementMismatch);
    }
    if receipt.root != *expected_root {
        return Err(Error::InvalidProof);
    }
    let challenge = shared_challenge(chain, batch)?;
    if challenge != receipt.challenge {
        return Err(Error::InvalidProof);
    }
    let combined = combine_claims(&batch.members, challenge)?;
    if combined.len() != required_relations.len() {
        return Err(Error::UnclosedClaims);
    }
    for ((relation, value), required) in combined.iter().zip(required_relations) {
        if relation != required {
            return Err(Error::StatementMismatch);
        }
        let expected = receipt
            .combined
            .iter()
            .find(|(candidate, _)| candidate == relation)
            .map(|(_, value)| *value)
            .ok_or(Error::UnclosedClaims)?;
        if expected != *value {
            return Err(Error::InvalidProof);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Batch binding tree and slots.
//
// Leaves are request seals (four words) plus a salt; vertices start from the
// configuration digest. Unused slots hold 32 random bytes chosen by the
// prover and are never read by a verifier.

use zkie_engine::compute::cpu::{leaf_tag, merkle_leaf};
use zkie_engine::primitives::transcript::{BINDING_NODE, LEAF_BINDING};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BindingTree {
    pub log_n: u32,
    pub configuration_digest: Digest,
    pub root: Digest,
    pub leaves: Vec<Digest>,
    pub levels: Vec<Vec<Digest>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotProof {
    pub slot: u64,
    pub seal: Digest,
    pub salt: Vec<u8>,
    pub path: Vec<Digest>,
    pub root: Digest,
}

fn binding_node(start: &Digest, left: &Digest, right: &Digest) -> Result<Digest> {
    let mut data = Vec::with_capacity(64);
    data.extend_from_slice(&left.0);
    data.extend_from_slice(&right.0);
    hash_labeled(tag_word(BINDING_NODE, 0), 0, Some(start), &data)
}

pub fn binding_leaf(
    configuration_digest: &Digest,
    slot: u64,
    seal: &Digest,
    salt: &[u8],
) -> Result<Digest> {
    if salt.len() < 40 || salt.len() > 64 || salt.len() % 8 != 0 {
        return Err(Error::InvalidConfig);
    }
    let mut symbols = Vec::with_capacity(4);
    for word in seal.0.chunks_exact(8) {
        symbols.push(u64::from_le_bytes(
            word.try_into().map_err(|_| Error::InvalidEncoding)?,
        ));
    }
    merkle_leaf(
        leaf_tag(LEAF_BINDING, 0, 4, salt.len())?,
        slot,
        configuration_digest,
        &symbols,
        salt,
    )
}

/// Build a perfect binding tree. `occupied` lists (slot, seal, salt); every
/// other slot receives the caller's 32 random bytes. Slots must be unique and
/// below `2^log_n`.
pub fn build_binding_tree(
    log_n: u32,
    configuration_digest: Digest,
    occupied: &[(u64, Digest, Vec<u8>)],
    padding_digests: &[Digest],
) -> Result<BindingTree> {
    let count = 1usize
        .checked_shl(log_n)
        .ok_or(Error::ResourceLimit)?;
    if padding_digests.len() != count {
        return Err(Error::ShapeMismatch);
    }
    let mut leaves = padding_digests.to_vec();
    let mut seen = BTreeSet::new();
    for (slot, seal, salt) in occupied {
        if *slot >= count as u64 || !seen.insert(*slot) {
            return Err(Error::InvalidIndex);
        }
        leaves[*slot as usize] = binding_leaf(&configuration_digest, *slot, seal, salt)?;
    }
    let mut levels = vec![leaves.clone()];
    while levels.last().expect("one level").len() > 1 {
        let previous = levels.last().expect("one level");
        let mut next = Vec::with_capacity(previous.len() / 2);
        for pair in previous.chunks_exact(2) {
            next.push(binding_node(&configuration_digest, &pair[0], &pair[1])?);
        }
        levels.push(next);
    }
    let root = levels.last().expect("one level")[0];
    Ok(BindingTree {
        log_n,
        configuration_digest,
        root,
        leaves,
        levels,
    })
}

pub fn open_slot(
    tree: &BindingTree,
    slot: u64,
    seal: &Digest,
    salt: Vec<u8>,
) -> Result<SlotProof> {
    if slot >= tree.leaves.len() as u64 {
        return Err(Error::InvalidIndex);
    }
    let leaf = binding_leaf(&tree.configuration_digest, slot, seal, &salt)?;
    if leaf != tree.leaves[slot as usize] {
        return Err(Error::StatementMismatch);
    }
    let mut path = Vec::with_capacity(tree.levels.len().saturating_sub(1));
    let mut position = slot as usize;
    for level in &tree.levels[..tree.levels.len() - 1] {
        path.push(level[position ^ 1]);
        position >>= 1;
    }
    Ok(SlotProof {
        slot,
        seal: *seal,
        salt,
        path,
        root: tree.root,
    })
}

pub fn verify_slot(
    expected_root: &Digest,
    configuration_digest: &Digest,
    proof: &SlotProof,
) -> Result<()> {
    if proof.root != *expected_root {
        return Err(Error::ArtifactIntegrityMismatch);
    }
    let leaf = binding_leaf(configuration_digest, proof.slot, &proof.seal, &proof.salt)?;
    let mut current = leaf;
    let mut position = proof.slot as usize;
    for sibling in &proof.path {
        current = if position & 1 == 0 {
            binding_node(configuration_digest, &current, sibling)?
        } else {
            binding_node(configuration_digest, sibling, &current)?
        };
        position >>= 1;
    }
    if current == *expected_root {
        Ok(())
    } else {
        Err(Error::InvalidProof)
    }
}

/// Sample a slot from a chain that has already absorbed the batch root.
pub fn sample_slot(chain: &mut Chain, log_n: u32) -> Result<u64> {
    chain.draw_position(log_n)
}

#[derive(Clone, Debug)]
pub struct BindingBatch {
    pub members: Vec<SealedMember>,
    pub slots: Vec<u64>,
    pub tree: BindingTree,
    pub root: Digest,
}

pub fn freeze_binding_batch(
    members: Vec<SealedMember>,
    slots: Vec<u64>,
    tree: BindingTree,
) -> Result<BindingBatch> {
    if members.is_empty() || members.len() != slots.len() {
        return Err(Error::InvalidConfig);
    }
    let mut ids = BTreeSet::new();
    let mut seen_slots = BTreeSet::new();
    for (member, slot) in members.iter().zip(&slots) {
        if !ids.insert(member.request_id) || !seen_slots.insert(*slot) {
            return Err(Error::StatementMismatch);
        }
    }
    Ok(BindingBatch {
        root: tree.root,
        members,
        slots,
        tree,
    })
}

/// Use a binding tree root as the frozen batch root. The shared challenge then
/// binds the protocol's slot tree instead of the seal-list hash.
pub fn freeze_batch_with_binding(
    members: Vec<SealedMember>,
    tree: &BindingTree,
    slots: &[u64],
) -> Result<FrozenBatch> {
    if members.is_empty() || members.len() != slots.len() {
        return Err(Error::InvalidConfig);
    }
    let mut ids = BTreeSet::new();
    let mut seen_slots = BTreeSet::new();
    for (member, slot) in members.iter().zip(slots) {
        if !ids.insert(member.request_id) || !seen_slots.insert(*slot) {
            return Err(Error::StatementMismatch);
        }
        if *slot >= tree.leaves.len() as u64 {
            return Err(Error::InvalidIndex);
        }
    }
    Ok(FrozenBatch {
        members,
        root: tree.root,
    })
}
