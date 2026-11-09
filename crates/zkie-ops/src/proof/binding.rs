//! Relation prover/verifier ports and authenticated oracle commitments.
//!
//! Operators submit oracle bindings and pending claims to the session. They do
//! not own a transcript, choose a commitment backend, or register their own
//! proof-completion state.

use std::collections::BTreeMap;

use zkie_core::claim::{ClaimKind, PendingClaims, RelationId};
use zkie_core::common::field::ProtocolField;
use zkie_core::compute::ExecutionPort;
use zkie_core::error::{Error, Result};
pub use zkie_core::oracle::OracleId;
use zkie_core::oracle::OracleBinding;
use zkie_core::types::{Digest, TensorView};
use zkie_engine::compute::cpu::{merkle_build, merkle_verify};
use zkie_engine::primitives::codec::CanonicalWriter;
use zkie_engine::primitives::transcript::{hash_labeled, tag_word, STATEMENT};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationSpec {
    pub id: RelationId,
    pub kind: ClaimKind,
    pub oracles: Vec<OracleId>,
    pub parameters_digest: Digest,
}

#[derive(Clone, Debug, Default)]
pub struct RelationWitness {
    pub values: Vec<TensorView>,
    pub auxiliary: Vec<TensorView>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationReceipt {
    pub relation: RelationId,
    pub message: Vec<u8>,
}

pub trait RelationProver<F: ProtocolField> {
    fn register_oracle(&mut self, binding: &OracleBinding, values: &TensorView)
        -> Result<OracleId>;

    fn prove_relation(
        &mut self,
        ctx: &mut dyn ExecutionPort<F>,
        relation: &RelationSpec,
        witness: &RelationWitness,
    ) -> Result<RelationReceipt>;

    fn register_claims(&mut self, claims: PendingClaims<F>) -> Result<()>;

    /// Optional request-chain sink. Sessions backed by a coordinator submit the
    /// receipt digest as an ALGEBRAIC message; stateless test sessions ignore it.
    fn submit_receipt(&mut self, _receipt: &RelationReceipt) -> Result<()> {
        Ok(())
    }
}

pub trait RelationVerifier<F: ProtocolField> {
    fn verify_relation(
        &mut self,
        relation: &RelationSpec,
        receipt: &RelationReceipt,
    ) -> Result<PendingClaims<F>>;

    fn register_claims(&mut self, claims: PendingClaims<F>) -> Result<()>;
}

pub fn register_oracle_and_claims<F: ProtocolField>(
    session: &mut dyn RelationProver<F>,
    binding: &OracleBinding,
    values: &TensorView,
    claims: PendingClaims<F>,
) -> Result<OracleId> {
    let oracle = session.register_oracle(binding, values)?;
    session.register_claims(claims)?;
    Ok(oracle)
}

// ---------------------------------------------------------------------------
// Authenticated oracle registry and vector openings.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OracleRoot {
    pub root: Digest,
    pub start: Digest,
    pub count: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OracleRegistry {
    roots: BTreeMap<OracleId, OracleRoot>,
    next: u64,
}

impl OracleRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, root: Digest, start: Digest, count: u64) -> Result<OracleId> {
        if count == 0 || !count.is_power_of_two() {
            return Err(Error::InvalidConfig);
        }
        let id = OracleId(self.next);
        self.next = self.next.checked_add(1).ok_or(Error::ResourceLimit)?;
        self.roots.insert(id, OracleRoot { root, start, count });
        Ok(id)
    }

    pub fn get(&self, id: OracleId) -> Option<&OracleRoot> {
        self.roots.get(&id)
    }

    pub fn len(&self) -> usize {
        self.roots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }

    pub fn ids(&self) -> impl Iterator<Item = OracleId> + '_ {
        self.roots.keys().copied()
    }

    /// All roots in oracle-id order. A challenge that binds this digest binds
    /// every registered oracle.
    pub fn digest(&self) -> Digest {
        let mut writer = CanonicalWriter::new();
        for (id, root) in &self.roots {
            writer.put_u64(id.0);
            writer.put_digest(&root.root);
            writer.put_digest(&root.start);
            writer.put_u64(root.count);
        }
        hash_labeled(tag_word(STATEMENT, 5), 0, None, &writer.into_bytes())
            .unwrap_or(Digest::ZERO)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OracleOpening {
    pub oracle: OracleId,
    pub index: u64,
    pub values: Vec<u64>,
    pub path: Vec<Digest>,
}

pub fn oracle_start(label: &Digest) -> Digest {
    hash_labeled(tag_word(STATEMENT, 6), 0, None, &label.0).unwrap_or(Digest::ZERO)
}

pub fn oracle_leaf(index: u64, values: &[u64]) -> Digest {
    let _profile = zkie_core::profile::Scope::kernel(zkie_core::profile::Kernel::Merkle);
    let mut writer = CanonicalWriter::new();
    for value in values {
        writer.put_u64(*value);
    }
    hash_labeled(tag_word(STATEMENT, 0), index, None, &writer.into_bytes())
        .unwrap_or(Digest::ZERO)
}

/// Commit a vector of single-word rows and return the registry entry.
pub fn commit_vector(
    registry: &mut OracleRegistry,
    label: &Digest,
    values: &[u64],
) -> Result<OracleId> {
    let rows: Vec<Vec<u64>> = values.iter().map(|value| vec![*value]).collect();
    commit_rows(registry, label, &rows)
}

/// Commit rows and return every leaf opening. The caller uses the openings as
/// proof material; the verifier only needs the registry root.
fn oracle_tree(
    label: &Digest,
    rows: &[Vec<u64>],
) -> Result<(Digest, Digest, zkie_engine::compute::cpu::MerkleTree, Vec<Digest>)> {
    if rows.is_empty() || !rows.len().is_power_of_two() {
        return Err(Error::InvalidConfig);
    }
    let start = oracle_start(label);
    let digests: Vec<Digest> = rows
        .iter()
        .enumerate()
        .map(|(index, values)| oracle_leaf(index as u64, values))
        .collect();
    let tree = merkle_build(&start, &digests)?;
    Ok((tree.root, start, tree, digests))
}

pub fn commit_rows(
    registry: &mut OracleRegistry,
    label: &Digest,
    rows: &[Vec<u64>],
) -> Result<OracleId> {
    let (root, start, _, _) = oracle_tree(label, rows)?;
    registry.insert(root, start, rows.len() as u64)
}

pub fn commit_rows_with_openings(
    registry: &mut OracleRegistry,
    label: &Digest,
    rows: &[Vec<u64>],
) -> Result<(OracleId, Vec<OracleOpening>)> {
    let (root, start, tree, digests) = oracle_tree(label, rows)?;
    let oracle = registry.insert(root, start, rows.len() as u64)?;
    let mut openings = Vec::with_capacity(rows.len());
    for (index, values) in rows.iter().enumerate() {
        openings.push(OracleOpening {
            oracle,
            index: index as u64,
            values: values.clone(),
            path: tree.open(index)?,
        });
    }
    let _ = digests;
    Ok((oracle, openings))
}

pub fn verify_opening(
    registry: &OracleRegistry,
    opening: &OracleOpening,
    expected_count: u64,
) -> Result<()> {
    let root = registry.get(opening.oracle).ok_or(Error::ProverStateMissing)?;
    if root.count != expected_count {
        return Err(Error::StatementMismatch);
    }
    if opening.index >= root.count {
        return Err(Error::InvalidIndex);
    }
    if root.count == 0 || !root.count.is_power_of_two() {
        return Err(Error::InvalidConfig);
    }
    let depth = root.count.trailing_zeros() as usize;
    if opening.path.len() != depth {
        return Err(Error::InvalidProof);
    }
    let leaf = oracle_leaf(opening.index, &opening.values);
    if merkle_verify(&root.start, &root.root, opening.index as usize, &leaf, &opening.path)? {
        Ok(())
    } else {
        Err(Error::InvalidProof)
    }
}

pub fn commit_rows_paths(
    registry: &mut OracleRegistry,
    label: &Digest,
    rows: &[Vec<u64>],
) -> Result<(OracleId, Vec<OracleOpening>)> {
    commit_rows_with_openings(registry, label, rows)
}

// ---------------------------------------------------------------------------
// Graph edge binding.
//
// An edge is identified by its own id and by the exact tensor ids at its two
// ends; identical content under two different ids is not the same edge.
// Layout and padding digests are part of the binding.

use zkie_core::types::{TensorId, TensorSpec};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EdgeBinding {
    pub edge_id: u64,
    pub from: TensorId,
    pub to: TensorId,
    pub from_spec_digest: Digest,
    pub to_spec_digest: Digest,
    pub layout_digest: Digest,
    pub padding_digest: Digest,
}

fn spec_digest(spec: &TensorSpec) -> Digest {
    let mut writer = CanonicalWriter::new();
    writer.put_u64(spec.dtype.element_size());
    writer.put_u64(spec.shape.len() as u64);
    for dim in &spec.shape {
        writer.put_u64(*dim);
    }
    match &spec.layout {
        zkie_core::types::LayoutSpec::RowMajor => writer.put_u64(0),
        zkie_core::types::LayoutSpec::ColumnMajor => writer.put_u64(1),
        zkie_core::types::LayoutSpec::Strided(strides) => {
            writer.put_u64(2);
            writer.put_u64(strides.len() as u64);
            for stride in strides {
                writer.put_u64(*stride);
            }
        }
    }
    match spec.fixed {
        Some(fixed) => {
            writer.put_u64(1);
            writer.put_u64(fixed.scale as u64);
            writer.put_u64(fixed.signed_bound as u64);
            writer.put_u64((fixed.signed_bound >> 64) as u64);
        }
        None => writer.put_u64(0),
    }
    writer.put_u64(spec.padding.value as u64);
    match spec.padding.valid_len {
        Some(length) => {
            writer.put_u64(1);
            writer.put_u64(length);
        }
        None => writer.put_u64(0),
    }
    if let Some(digest) = spec.padding.mask_digest {
        writer.put_digest(&digest);
    }
    hash_labeled(tag_word(STATEMENT, 14), 0, None, &writer.into_bytes())
        .unwrap_or(Digest::ZERO)
}

pub fn bind_edge(
    edge_id: u64,
    from: TensorId,
    to: TensorId,
    from_spec: &TensorSpec,
    to_spec: &TensorSpec,
) -> Result<EdgeBinding> {
    from_spec.validate()?;
    to_spec.validate()?;
    let layout_digest = hash_labeled(
        tag_word(STATEMENT, 15),
        edge_id,
        None,
        &spec_digest(from_spec).0,
    )?;
    Ok(EdgeBinding {
        edge_id,
        from,
        to,
        from_spec_digest: spec_digest(from_spec),
        to_spec_digest: spec_digest(to_spec),
        layout_digest,
        padding_digest: hash_labeled(
            tag_word(STATEMENT, 16),
            edge_id,
            None,
            &spec_digest(to_spec).0,
        )?,
    })
}

pub fn verify_edge(
    binding: &EdgeBinding,
    edge_id: u64,
    from: TensorId,
    to: TensorId,
    from_spec: &TensorSpec,
    to_spec: &TensorSpec,
) -> Result<()> {
    if binding.edge_id != edge_id || binding.from != from || binding.to != to {
        return Err(Error::StatementMismatch);
    }
    let expected = bind_edge(edge_id, from, to, from_spec, to_spec)?;
    if binding.from_spec_digest != expected.from_spec_digest
        || binding.to_spec_digest != expected.to_spec_digest
        || binding.layout_digest != expected.layout_digest
        || binding.padding_digest != expected.padding_digest
    {
        return Err(Error::StatementMismatch);
    }
    Ok(())
}

pub fn graph_binding_digest(edges: &[EdgeBinding]) -> Digest {
    let mut writer = CanonicalWriter::new();
    writer.put_u64(edges.len() as u64);
    for edge in edges {
        writer.put_u64(edge.edge_id);
        writer.put_u64(edge.from.0);
        writer.put_u64(edge.to.0);
        writer.put_digest(&edge.from_spec_digest);
        writer.put_digest(&edge.to_spec_digest);
        writer.put_digest(&edge.layout_digest);
        writer.put_digest(&edge.padding_digest);
    }
    hash_labeled(tag_word(STATEMENT, 17), 0, None, &writer.into_bytes())
        .unwrap_or(Digest::ZERO)
}
