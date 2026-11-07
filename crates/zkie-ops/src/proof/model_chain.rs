//! Nonlinear stages chained to the model logits.
//!
//! The first stage input is bound to the model logits oracle root. Every later
//! stage input must equal the previous stage output, and each stage is checked
//! by its algebraic relation without invoking the operator forward routine.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::proof::relations::{
    prove_op_relation, proof_input_values, proof_output_values, verify_op_relation,
    OpRelationProof, OpRelationPublic,
};
use crate::registry::PureForward;

pub struct OpChainStage {
    pub op: Box<dyn PureForward>,
    pub config: Vec<i64>,
    pub inputs: Vec<Vec<i64>>,
}

#[derive(Clone, Debug)]
pub struct OpChainPublic {
    pub initial_root: Digest,
    pub stages: Vec<OpRelationPublic>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpChainProof {
    pub stages: Vec<OpRelationProof>,
}

pub fn prove_op_chain(
    initial_binding: Digest,
    stages: &[OpChainStage],
) -> Result<(OpChainPublic, OpChainProof)> {
    if stages.is_empty() {
        return Err(Error::InvalidConfig);
    }
    if value_binding(&stages[0].inputs[..1]) != initial_binding {
        return Err(Error::StatementMismatch);
    }
    let mut publics = Vec::new();
    let mut proofs = Vec::new();
    let mut previous: Option<Vec<Vec<i64>>> = None;
    for (index, stage) in stages.iter().enumerate() {
        if let Some(values) = &previous {
            if stage.inputs.first() != values.first() {
                return Err(Error::StatementMismatch);
            }
        }
        let (public, proof) = prove_op_relation(stage.op.as_ref(), &stage.inputs, &stage.config)?;
        previous = Some(proof_output_values(&proof));
        publics.push(public);
        proofs.push(proof);
    }
    Ok((
        OpChainPublic {
            initial_root: initial_binding,
            stages: publics,
        },
        OpChainProof { stages: proofs },
    ))
}

/// Verify every nonlinear stage and the value-level edge between stages.
/// Returns the final stage outputs.
pub fn verify_op_chain(
    public: &OpChainPublic,
    proof: &OpChainProof,
    expected_initial_root: Digest,
) -> Result<Vec<Vec<i64>>> {
    if public.initial_root != expected_initial_root || public.stages.len() != proof.stages.len() {
        return Err(Error::StatementMismatch);
    }
    let mut previous: Option<Vec<Vec<i64>>> = None;
    for (index, (stage_public, stage_proof)) in public.stages.iter().zip(&proof.stages).enumerate() {
        verify_op_relation(stage_public, stage_proof)?;
        let inputs = proof_input_values(stage_proof);
        if index == 0 && value_binding(&inputs[..1]) != expected_initial_root {
            return Err(Error::StatementMismatch);
        }
        if let Some(values) = &previous {
            if inputs.first() != values.first() {
                return Err(Error::StatementMismatch);
            }
        }
        previous = Some(proof_output_values(stage_proof));
    }
    previous.ok_or(Error::InvalidConfig)
}


/// Content binding of a stage input vector set. The model proof opens the
/// logits row; the verifier recomputes this digest from that opening.
pub fn value_binding(values: &[Vec<i64>]) -> Digest {
    let mut writer = zkie_engine::primitives::codec::CanonicalWriter::new();
    writer.put_u64(values.len() as u64);
    for vector in values {
        writer.put_u64(vector.len() as u64);
        for value in vector {
            writer.put_u64(*value as u64);
        }
    }
    zkie_engine::primitives::transcript::hash_labeled(
        zkie_engine::primitives::transcript::tag_word(
            zkie_engine::primitives::transcript::STATEMENT,
            10,
        ),
        0,
        None,
        &writer.into_bytes(),
    )
    .unwrap_or(Digest::ZERO)
}
