//! Isolated tied-use candidate for the model asset layouts.
//!
//! This module checks candidate bindings between verified asset identities and
//! the index lists a model operator reads. It does not read assets, verify an
//! actual opening proof, execute a model, build a plan, or run a LIN proof.
//! `opening_root` is only the root a caller's opening check reports; the real
//! opening verification remains an unimplemented production gate. No file path,
//! label, or graph name is treated as a trusted anchor.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::field::Fp3;

/// Candidate allocation bound for the model adapter. The dimension and
/// vocabulary are plan values, never proof values; this cap keeps a malformed
/// or untrusted plan from forcing a huge index vector before a model proof can
/// start. It is an adapter limit, not a protocol constant.
pub const MAX_GPT2_EMBEDDING_DIMENSION: u64 = 1 << 20;

fn check_gpt2_dimensions(hidden: u64, vocab: u64) -> Result<()> {
    if hidden == 0 || vocab == 0 || hidden > MAX_GPT2_EMBEDDING_DIMENSION || vocab > MAX_GPT2_EMBEDDING_DIMENSION {
        return Err(Error::ResourceLimit);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssetShape {
    pub rows: u64,
    pub columns: u64,
}

impl AssetShape {
    pub fn count(&self) -> Result<u64> {
        self.rows
            .checked_mul(self.columns)
            .filter(|count| *count > 0)
            .ok_or(Error::ResourceLimit)
    }

    pub fn contains(&self, index: u64) -> Result<bool> {
        Ok(index < self.count()?)
    }
}

/// One use of a verified asset: the root and shape come from a trusted plan,
/// `indices` are the entries this use opens, and `opening_root` is the root the
/// caller's opening check reports. The fields are public so a verifier can
/// re-check a decoded use; every check revalidates them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssetUse {
    pub root: Digest,
    pub opening_root: Digest,
    pub shape: AssetShape,
    pub indices: Vec<u64>,
}

impl AssetUse {
    pub fn new(
        root: Digest,
        opening_root: Digest,
        shape: AssetShape,
        indices: Vec<u64>,
    ) -> Result<Self> {
        let use_ = Self {
            root,
            opening_root,
            shape,
            indices,
        };
        check_asset_use(&use_)?;
        Ok(use_)
    }
}

fn check_asset_use(use_: &AssetUse) -> Result<()> {
    if use_.root == Digest::ZERO
        || use_.opening_root == Digest::ZERO
        || use_.indices.is_empty()
        || use_.shape.count()? == 0
    {
        return Err(Error::InvalidConfig);
    }
    if use_.opening_root != use_.root {
        return Err(Error::StatementMismatch);
    }
    for index in &use_.indices {
        if !use_.shape.contains(*index)? {
            return Err(Error::InvalidIndex);
        }
    }
    Ok(())
}

/// The real opening proof is not implemented in this candidate. A production
/// caller must not treat the structural `opening_root` equality above as proof;
/// this entry stays fail-closed until a separately approved verifier exists.
pub fn verify_candidate_opening(_use: &AssetUse) -> Result<()> {
    Err(Error::UnsupportedPrimitive)
}

/// A candidate GPT-2 tied use. The embedding lookup and the head matrix refer
/// to one verified root and one trusted `[hidden, vocab]` layout, while each
/// side keeps its own opened indices. The embedding indices are checked against
/// the lookup mapping; no intersection or containment with the head's opening
/// query is required, because LIN random queries and lookup queries may differ.
/// Each side still needs its own proof; identity alone is not a value or graph
/// edge proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gpt2Tied {
    pub hidden: u64,
    pub vocab: u64,
    pub token: u64,
    pub embedding: AssetUse,
    pub head: AssetUse,
    pub configuration: Digest,
}

impl Gpt2Tied {
    pub fn check(&self) -> Result<()> {
        if self.hidden == 0 || self.vocab == 0 || self.token >= self.vocab {
            return Err(Error::InvalidIndex);
        }
        check_gpt2_dimensions(self.hidden, self.vocab)?;
        if self.configuration == Digest::ZERO {
            return Err(Error::InvalidConfig);
        }
        let shape = AssetShape {
            rows: self.hidden,
            columns: self.vocab,
        };
        for use_ in [&self.embedding, &self.head] {
            if use_.shape != shape {
                return Err(Error::ShapeMismatch);
            }
            check_asset_use(use_)?;
        }
        if self.embedding.root != self.head.root {
            return Err(Error::StatementMismatch);
        }
        let hidden = usize::try_from(self.hidden).map_err(|_| Error::ResourceLimit)?;
        if self.embedding.indices.len() != hidden {
            return Err(Error::ShapeMismatch);
        }
        for (dimension, index) in self.embedding.indices.iter().enumerate() {
            let expected = (dimension as u64)
                .checked_mul(self.vocab)
                .and_then(|value| value.checked_add(self.token))
                .ok_or(Error::ResourceLimit)?;
            if *index != expected {
                return Err(Error::StatementMismatch);
            }
        }
        Ok(())
    }
}

/// The embedding lookup's flat indices for one token, in dimension order.
pub fn gpt2_embedding_indices(hidden: u64, vocab: u64, token: u64) -> Result<Vec<u64>> {
    if hidden == 0 || vocab == 0 || token >= vocab {
        return Err(Error::InvalidIndex);
    }
    check_gpt2_dimensions(hidden, vocab)?;
    let hidden_len = usize::try_from(hidden).map_err(|_| Error::ResourceLimit)?;
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(hidden_len)
        .map_err(|_| Error::ResourceLimit)?;
    for dimension in 0..hidden {
        indices.push(
            dimension
                .checked_mul(vocab)
                .and_then(|value| value.checked_add(token))
                .ok_or(Error::ResourceLimit)?,
        );
    }
    Ok(indices)
}

/// Small lookup relation: read the embedding vector through the embedding
/// indices. This is arithmetic evidence only, not an opening or LIN proof.
pub fn gpt2_lookup(values: &[Fp3], hidden: u64, vocab: u64, token: u64) -> Result<Vec<Fp3>> {
    let indices = gpt2_embedding_indices(hidden, vocab, token)?;
    let mut out = Vec::new();
    out.try_reserve_exact(indices.len())
        .map_err(|_| Error::ResourceLimit)?;
    for index in indices {
        out.push(
            *values
                .get(usize::try_from(index).map_err(|_| Error::ResourceLimit)?)
                .ok_or(Error::InvalidIndex)?,
        );
    }
    Ok(out)
}

/// Small head relation: the token's logit contribution is the dot product of
/// the activation with the same asset column, read at `dimension * vocab +
/// token`. This is the linear dual-use arithmetic, not a proof.
pub fn gpt2_head_token(
    values: &[Fp3],
    activation: &[Fp3],
    hidden: u64,
    vocab: u64,
    token: u64,
) -> Result<Fp3> {
    if hidden == 0
        || vocab == 0
        || token >= vocab
        || activation.len() != usize::try_from(hidden).map_err(|_| Error::ResourceLimit)?
    {
        return Err(Error::ShapeMismatch);
    }
    check_gpt2_dimensions(hidden, vocab)?;
    let mut sum = Fp3::ZERO;
    for (dimension, activation) in activation.iter().enumerate() {
        let index = (dimension as u64)
            .checked_mul(vocab)
            .and_then(|value| value.checked_add(token))
            .ok_or(Error::ResourceLimit)?;
        let value = *values
            .get(usize::try_from(index).map_err(|_| Error::ResourceLimit)?)
            .ok_or(Error::InvalidIndex)?;
        sum = sum.add(&activation.mul(&value));
    }
    Ok(sum)
}

/// A candidate self-consistency check for the GPT-2 embedding/head label
/// mismatch: the graph names an external embedding table while the executor
/// reads the head asset. Equal caller-supplied roots are only a local
/// consistency condition; the trusted plan, root source and graph edge are not
/// connected here. Names and paths are never anchors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gpt2PlanCandidate {
    pub graph_root: Digest,
    pub executor_root: Digest,
    pub shape: AssetShape,
    pub configuration: Digest,
}

impl Gpt2PlanCandidate {
    pub fn check_candidate(&self) -> Result<()> {
        if self.graph_root == Digest::ZERO
            || self.executor_root == Digest::ZERO
            || self.configuration == Digest::ZERO
            || self.shape.count()? == 0
        {
            return Err(Error::InvalidConfig);
        }
        if self.graph_root != self.executor_root {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }
}

/// The Gemma3 candidate's two assets have different commitments in the current
/// model, and even one root would not establish the runtime transpose. This
/// gate stays unsupported: a production relation needs a separately adjudicated
/// proof with both roots, the full index domain, quantization, padding and
/// layout. No script or generic link substitutes for it.
pub fn check_gemma3_tied(_embedding: &AssetUse, _head: &AssetUse) -> Result<()> {
    Err(Error::UnsupportedPrimitive)
}

/// The TimesFM embedding is a one-dimensional vector broadcast into a sum, not
/// a matrix row or a tied head. The indices must cover the full vector in file
/// order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimesfmEmbeddingUse {
    pub root: Digest,
    pub opening_root: Digest,
    pub length: u64,
    pub indices: Vec<u64>,
}

impl TimesfmEmbeddingUse {
    pub fn check(&self) -> Result<()> {
        if self.root == Digest::ZERO
            || self.opening_root == Digest::ZERO
            || self.length == 0
            || self.opening_root != self.root
        {
            return Err(Error::InvalidConfig);
        }
        let length = usize::try_from(self.length).map_err(|_| Error::ResourceLimit)?;
        if self.indices.len() != length {
            return Err(Error::ShapeMismatch);
        }
        for (position, index) in self.indices.iter().enumerate() {
            if *index != position as u64 {
                return Err(Error::StatementMismatch);
            }
        }
        Ok(())
    }
}

/// TimesFM has no tied embedding/head mapping; a tied request stays unsupported
/// rather than forcing the GPT-2 head shape onto it.
pub fn check_timesfm_tied(_embedding: &TimesfmEmbeddingUse) -> Result<()> {
    Err(Error::UnsupportedPrimitive)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(count: u64) -> Vec<Fp3> {
        (0..count).map(|value| Fp3::from_u64(value + 1)).collect()
    }

    fn use_(
        root: u8,
        opening_root: u8,
        shape: AssetShape,
        indices: Vec<u64>,
    ) -> AssetUse {
        AssetUse::new(Digest([root; 32]), Digest([opening_root; 32]), shape, indices).unwrap()
    }

    fn gpt2() -> Gpt2Tied {
        let shape = AssetShape {
            rows: 3,
            columns: 5,
        };
        Gpt2Tied {
            hidden: 3,
            vocab: 5,
            token: 2,
            embedding: use_(1, 1, shape, vec![2, 7, 12]),
            head: use_(1, 1, shape, (0..15).collect()),
            configuration: Digest([9; 32]),
        }
    }

    #[test]
    fn gpt2_tied_binds_same_root_shape_and_own_indices() {
        let candidate = gpt2();
        candidate.check().unwrap();
        let mut disjoint_head = gpt2();
        disjoint_head.head.indices = vec![0, 1, 3];
        disjoint_head.check().unwrap();
        let values = values(15);
        let lookup = gpt2_lookup(&values, 3, 5, 2).unwrap();
        assert_eq!(lookup, vec![values[2], values[7], values[12]]);
        let activation = vec![Fp3::from_u64(2), Fp3::from_u64(3), Fp3::from_u64(5)];
        let head = gpt2_head_token(&values, &activation, 3, 5, 2).unwrap();
        assert_eq!(
            head,
            Fp3::from_u64(2)
                .mul(&lookup[0])
                .add(&Fp3::from_u64(3).mul(&lookup[1]))
                .add(&Fp3::from_u64(5).mul(&lookup[2]))
        );
        assert!(matches!(
            verify_candidate_opening(&candidate.embedding),
            Err(Error::UnsupportedPrimitive)
        ));
        assert!(gpt2_head_token(&values, &activation[..2], 3, 5, 2).is_err());
        assert!(gpt2_head_token(&values[..12], &activation, 3, 5, 2).is_err());
        assert!(gpt2_lookup(&values, 3, 5, 5).is_err());
    }

    #[test]
    fn gpt2_identity_alone_and_wrong_mappings_are_rejected() {
        let mut different_root = gpt2();
        different_root.head.root = Digest([2; 32]);
        different_root.head.opening_root = Digest([2; 32]);
        assert!(different_root.check().is_err());

        let mut wrong_shape = gpt2();
        wrong_shape.embedding.shape.columns = 4;
        assert!(wrong_shape.check().is_err());

        let mut head_shape = gpt2();
        head_shape.head.shape.columns = 6;
        assert!(head_shape.check().is_err());

        let mut token_major = gpt2();
        token_major.embedding.indices = vec![6, 7, 8];
        assert!(token_major.check().is_err());

        let mut head_out_of_range = gpt2();
        head_out_of_range.head.indices.push(99);
        assert!(head_out_of_range.check().is_err());

        let mut empty_head = gpt2();
        empty_head.head.indices.clear();
        assert!(empty_head.check().is_err());

        let mut out_of_range = gpt2();
        out_of_range.embedding.indices[0] = 99;
        assert!(out_of_range.check().is_err());

        let mut wrong_opening = gpt2();
        wrong_opening.head.opening_root = Digest([3; 32]);
        assert!(wrong_opening.check().is_err());

        let mut wrong_token = gpt2();
        wrong_token.token = 5;
        assert!(wrong_token.check().is_err());

        let mut zero_configuration = gpt2();
        zero_configuration.configuration = Digest::ZERO;
        assert!(zero_configuration.check().is_err());

        let mut short_embedding = gpt2();
        short_embedding.embedding.indices.pop();
        assert!(short_embedding.check().is_err());

        let mut too_large = gpt2();
        too_large.hidden = MAX_GPT2_EMBEDDING_DIMENSION + 1;
        assert!(too_large.check().is_err());
        assert!(gpt2_embedding_indices(MAX_GPT2_EMBEDDING_DIMENSION + 1, 5, 2).is_err());
        assert!(gpt2_embedding_indices(3, MAX_GPT2_EMBEDDING_DIMENSION + 1, 2).is_err());
    }

    #[test]
    fn plan_candidate_self_consistency_uses_roots_only() {
        let binding = Gpt2PlanCandidate {
            graph_root: Digest([4; 32]),
            executor_root: Digest([4; 32]),
            shape: AssetShape {
                rows: 3,
                columns: 5,
            },
            configuration: Digest([9; 32]),
        };
        binding.check_candidate().unwrap();
        let mut other_root = binding;
        other_root.executor_root = Digest([5; 32]);
        assert!(other_root.check_candidate().is_err());
        let mut zero = binding;
        zero.graph_root = Digest::ZERO;
        assert!(zero.check_candidate().is_err());
        let mut empty = binding;
        empty.shape.rows = 0;
        assert!(empty.check_candidate().is_err());
    }

    #[test]
    fn gemma3_tied_stays_unsupported() {
        let embed_shape = AssetShape {
            rows: 5,
            columns: 3,
        };
        let head_shape = AssetShape {
            rows: 3,
            columns: 5,
        };
        let embed = use_(1, 1, embed_shape, (0..15).collect());
        let head = use_(2, 2, head_shape, (0..15).collect());
        assert!(matches!(
            check_gemma3_tied(&embed, &head),
            Err(Error::UnsupportedPrimitive)
        ));
        let same = use_(1, 1, head_shape, (0..15).collect());
        assert!(matches!(
            check_gemma3_tied(&embed, &same),
            Err(Error::UnsupportedPrimitive)
        ));
    }

    #[test]
    fn timesfm_maps_a_vector_and_has_no_tied_head() {
        let use_ = TimesfmEmbeddingUse {
            root: Digest([6; 32]),
            opening_root: Digest([6; 32]),
            length: 4,
            indices: vec![0, 1, 2, 3],
        };
        use_.check().unwrap();
        let mut missing = use_.clone();
        missing.indices.pop();
        assert!(missing.check().is_err());
        let mut reordered = use_.clone();
        reordered.indices = vec![0, 2, 1, 3];
        assert!(reordered.check().is_err());
        let mut wrong_opening = use_;
        wrong_opening.opening_root = Digest([7; 32]);
        assert!(wrong_opening.check().is_err());
        assert!(matches!(
            check_timesfm_tied(&wrong_opening),
            Err(Error::UnsupportedPrimitive)
        ));
    }
}
