//! Single-shot statement construction and coverage checks.
//!
//! One request statement covers the complete decode sequence. The statement
//! records every covered position, including the last predicted token.

use zkie_core::error::{Error, Result};
use zkie_core::session::{DecodeCoverage, EosPolicy, OutputPolicy, Statement};
use zkie_core::types::Digest;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeRequest {
    pub prompt_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub eos_policy: EosPolicy,
    pub output_policy: OutputPolicy,
}

impl DecodeRequest {
    pub fn validate(&self) -> Result<()> {
        if self.prompt_tokens.is_empty() {
            return Err(Error::StatementMismatch);
        }
        if self.output_tokens.is_empty() {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }
}

pub fn build_decode_coverage(decode_len: u32) -> Result<DecodeCoverage> {
    if decode_len == 0 {
        return Err(Error::StatementMismatch);
    }
    let coverage = DecodeCoverage::full(decode_len);
    coverage.validate()?;
    Ok(coverage)
}

#[allow(clippy::too_many_arguments)]
pub fn build_single_shot_statement(
    protocol_id: Digest,
    protocol_version: u16,
    parameter_digest: Digest,
    model_id: Digest,
    graph_digest: Digest,
    quantization_digest: Digest,
    tables_digest: Digest,
    request: &DecodeRequest,
) -> Result<Statement> {
    request.validate()?;
    let decode_len =
        u32::try_from(request.output_tokens.len()).map_err(|_| Error::StatementMismatch)?;
    let coverage = build_decode_coverage(decode_len)?;
    let statement = Statement {
        protocol_id,
        protocol_version,
        parameter_digest,
        model_id,
        graph_digest,
        quantization_digest,
        tables_digest,
        prompt_tokens: request.prompt_tokens.clone(),
        output_tokens: request.output_tokens.clone(),
        coverage,
        eos_policy: request.eos_policy,
        output_policy: request.output_policy.clone(),
    };
    statement.validate_single_shot()?;
    Ok(statement)
}

pub fn validate_decode_coverage(statement: &Statement) -> Result<()> {
    statement.validate_single_shot()
}

// ---------------------------------------------------------------------------
// Single-shot decode proof: one proof covers all D next-token relations and
// the KV/causal/RoPE state they read.

use std::collections::BTreeSet;

use zkie_core::oracle::OracleId;
use zkie_core::session::VerificationResult;

use crate::proof::binding::{
    commit_rows_with_openings, verify_opening, OracleOpening, OracleRegistry,
};
use crate::proof::request::output_policy::{
    statement_digest, validate_statement_policy, verify_output, OutputProof,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KvKind {
    Key,
    Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KvAddress {
    pub layer: u32,
    pub head: u32,
    pub position: u32,
    pub channel: u32,
    pub kind: KvKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepWitness {
    pub token: u32,
    pub kv_appends: Vec<KvAddress>,
    pub kv_reads: Vec<KvAddress>,
    pub mask: Vec<u8>,
    pub rope_offset: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepProof {
    pub position: u32,
    pub token: u32,
    pub logits: OracleOpening,
    pub kv_appends: Vec<KvAddress>,
    pub kv_reads: Vec<KvAddress>,
    pub mask: Vec<u8>,
    pub rope_offset: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SingleShotProof {
    pub statement_digest: Digest,
    pub steps: Vec<StepProof>,
    pub gaps: Vec<i64>,
}

#[derive(Clone, Debug)]
pub struct SingleShotPublic {
    pub statement: Statement,
    pub registry: OracleRegistry,
    pub logits_oracle: OracleId,
    pub layers: u32,
    pub heads: u32,
    pub width: u32,
}

pub fn build_single_shot_public(
    statement: Statement,
    logits_rows: &[Vec<i64>],
    layers: u32,
    heads: u32,
    width: u32,
) -> Result<(SingleShotPublic, Vec<OracleOpening>)> {
    statement.validate_single_shot()?;
    if logits_rows.len() != statement.coverage.decode_len as usize {
        return Err(Error::ShapeMismatch);
    }
    if layers == 0 || heads == 0 || width == 0 {
        return Err(Error::InvalidConfig);
    }
    let label = statement_digest(&statement)?;
    let rows: Vec<Vec<u64>> = logits_rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| u64::try_from(*value)
                    .map_err(|_| Error::BoundViolation))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<_>>()?;
    let mut registry = OracleRegistry::new();
    let (logits_oracle, openings) =
        commit_rows_with_openings(&mut registry, &label, &rows)?;
    Ok((
        SingleShotPublic {
            statement,
            registry,
            logits_oracle,
            layers,
            heads,
            width,
        },
        openings,
    ))
}

fn argmax(logits: &[u64]) -> Result<u32> {
    if logits.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let mut best = 0usize;
    for index in 1..logits.len() {
        if logits[index] > logits[best] {
            best = index;
        }
    }
    u32::try_from(best).map_err(|_| Error::ResourceLimit)
}

fn expected_position(public: &SingleShotPublic, step: usize) -> Result<u32> {
    let prompt = u32::try_from(public.statement.prompt_tokens.len())
        .map_err(|_| Error::ResourceLimit)?;
    let step = u32::try_from(step).map_err(|_| Error::ResourceLimit)?;
    prompt
        .checked_sub(1)
        .and_then(|value| value.checked_add(step))
        .ok_or(Error::InvalidConfig)
}

fn check_kv(public: &SingleShotPublic, steps: &[StepProof]) -> Result<()> {
    let mut history: BTreeSet<KvAddress> = BTreeSet::new();
    for step in steps {
        for address in &step.kv_appends {
            if address.layer >= public.layers
                || address.head >= public.heads
                || address.channel >= public.width
                || address.position != step.position
            {
                return Err(Error::InvalidIndex);
            }
            if !history.insert(*address) {
                return Err(Error::StatementMismatch);
            }
        }
        for address in &step.kv_reads {
            if address.layer >= public.layers
                || address.head >= public.heads
                || address.channel >= public.width
                || address.position > step.position
                || !history.contains(address)
            {
                return Err(Error::InvalidIndex);
            }
        }
        let expected_mask = (step.position as usize) + 1;
        if step.mask.len() != expected_mask || step.mask.iter().any(|value| *value != 1) {
            return Err(Error::StatementMismatch);
        }
        if step.rope_offset != step.position {
            return Err(Error::StatementMismatch);
        }
    }
    Ok(())
}

pub fn prove_single_shot(
    public: &SingleShotPublic,
    witnesses: &[StepWitness],
    openings: &[OracleOpening],
    gaps: Vec<i64>,
) -> Result<SingleShotProof> {
    if witnesses.len() != public.statement.coverage.decode_len as usize
        || openings.len() != witnesses.len()
    {
        return Err(Error::ShapeMismatch);
    }
    let mut steps = Vec::with_capacity(witnesses.len());
    for (index, witness) in witnesses.iter().enumerate() {
        let position = expected_position(public, index)?;
        let opening = &openings[index];
        if opening.oracle != public.logits_oracle || opening.index != index as u64 {
            return Err(Error::StatementMismatch);
        }
        let root = public
            .registry
            .get(public.logits_oracle)
            .ok_or(Error::ProverStateMissing)?;
        verify_opening(&public.registry, opening, root.count)?;
        if argmax(&opening.values)? != witness.token
            || witness.token != public.statement.output_tokens[index]
        {
            return Err(Error::InvalidProof);
        }
        steps.push(StepProof {
            position,
            token: witness.token,
            logits: opening.clone(),
            kv_appends: witness.kv_appends.clone(),
            kv_reads: witness.kv_reads.clone(),
            mask: witness.mask.clone(),
            rope_offset: witness.rope_offset,
        });
    }
    check_kv(public, &steps)?;
    Ok(SingleShotProof {
        statement_digest: statement_digest(&public.statement)?,
        steps,
        gaps,
    })
}

pub fn verify_single_shot(
    public: &SingleShotPublic,
    proof: &SingleShotProof,
) -> Result<VerificationResult> {
    validate_statement_policy(&public.statement)?;
    let expected = statement_digest(&public.statement)?;
    if proof.statement_digest != expected
        || proof.steps.len() != public.statement.coverage.decode_len as usize
    {
        return Err(Error::StatementMismatch);
    }
    for (index, step) in proof.steps.iter().enumerate() {
        if step.position != expected_position(public, index)?
            || step.token != public.statement.output_tokens[index]
        {
            return Err(Error::InvalidProof);
        }
        let opening = &step.logits;
        if opening.oracle != public.logits_oracle || opening.index != index as u64 {
            return Err(Error::InvalidProof);
        }
        let root = public
            .registry
            .get(public.logits_oracle)
            .ok_or(Error::ProverStateMissing)?;
        verify_opening(&public.registry, opening, root.count)?;
        if argmax(&opening.values)? != step.token {
            return Err(Error::InvalidProof);
        }
    }
    check_kv(public, &proof.steps)?;
    // The final output token has a real predicted position: the causal input
    // covers P + D - 1 positions and the last step reads the last one.
    let prompt = public.statement.prompt_tokens.len() as u32;
    let decode = public.statement.coverage.decode_len;
    let last = proof.steps.last().ok_or(Error::IncompleteArtifact)?;
    if last.position != prompt + decode - 2 || prompt + decode - 1 <= last.position {
        return Err(Error::InvalidProof);
    }
    verify_output(
        &public.statement,
        &OutputProof {
            tokens: public.statement.output_tokens.clone(),
            gaps: proof.gaps.clone(),
        },
    )
}
