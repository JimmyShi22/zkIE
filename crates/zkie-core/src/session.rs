//! Session state, statements, public bundles and private state leases.
//!
//! The state machine is the protocol lifecycle. It is intentionally independent
//! of any proof backend: illegal transitions fail even when every component is
//! individually available.

use std::collections::BTreeSet;

use crate::common::codec::DecodeLimits;
use crate::compute::VerifierPolicy;
use crate::error::{Error, Result};
use crate::progress::ProgressSink;
use crate::types::Digest;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionState {
    Prepared,
    RequestChainRunning,
    RequestChainSealed,
    BatchFrozen,
    SharedChainRunning,
    SharedChainSealed,
    FinalChainRunning,
    Finalized,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionStateMachine {
    state: SessionState,
    claims_closed: bool,
}

impl Default for SessionStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionStateMachine {
    pub fn new() -> Self {
        Self {
            state: SessionState::Prepared,
            claims_closed: false,
        }
    }

    pub fn state(&self) -> SessionState {
        self.state
    }

    pub fn claims_closed(&self) -> bool {
        self.claims_closed
    }

    /// Advance one legal edge. Entering `Finalized` requires that all pending
    /// claims have already been closed.
    pub fn advance(&mut self, next: SessionState) -> Result<()> {
        let allowed = matches!(
            (self.state, next),
            (SessionState::Prepared, SessionState::RequestChainRunning)
                | (
                    SessionState::RequestChainRunning,
                    SessionState::RequestChainSealed
                )
                | (SessionState::RequestChainSealed, SessionState::BatchFrozen)
                | (SessionState::BatchFrozen, SessionState::SharedChainRunning)
                | (
                    SessionState::SharedChainRunning,
                    SessionState::SharedChainSealed
                )
                | (
                    SessionState::SharedChainSealed,
                    SessionState::FinalChainRunning
                )
                | (SessionState::FinalChainRunning, SessionState::Finalized)
        );
        if !allowed {
            return Err(Error::InvalidSessionState);
        }
        if next == SessionState::Finalized && !self.claims_closed {
            return Err(Error::UnclosedClaims);
        }
        self.state = next;
        Ok(())
    }

    /// Claim closure is only accepted during the final chain, after every
    /// request and shared relation has produced its pending claims.
    pub fn close_claims(&mut self) -> Result<()> {
        if self.state != SessionState::FinalChainRunning {
            return Err(Error::InvalidSessionState);
        }
        self.claims_closed = true;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeCoverage {
    pub decode_len: u32,
    pub covered_positions: Vec<u32>,
}

impl DecodeCoverage {
    pub fn full(decode_len: u32) -> Self {
        Self {
            decode_len,
            covered_positions: (0..decode_len).collect(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.decode_len == 0 {
            return Err(Error::StatementMismatch);
        }
        let mut seen = BTreeSet::new();
        for &position in &self.covered_positions {
            if position >= self.decode_len {
                return Err(Error::InvalidIndex);
            }
            if !seen.insert(position) {
                return Err(Error::StatementMismatch);
            }
        }
        if seen.len() != self.decode_len as usize {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EosPolicy {
    FixedLength,
    StopAtFirstEos,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClippedGapPolicy {
    pub parameters: Vec<u8>,
    pub parameters_digest: Digest,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutputPolicy {
    ExactMatch,
    ClippedGap(ClippedGapPolicy),
}

/// One request statement. A single statement covers the complete decode
/// sequence; changing the number of steps changes `coverage`, not the number of
/// statements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Statement {
    pub protocol_id: Digest,
    pub protocol_version: u16,
    pub parameter_digest: Digest,
    pub model_id: Digest,
    pub graph_digest: Digest,
    pub quantization_digest: Digest,
    pub tables_digest: Digest,
    pub prompt_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub coverage: DecodeCoverage,
    pub eos_policy: EosPolicy,
    pub output_policy: OutputPolicy,
}

impl Statement {
    pub fn decode_len(&self) -> u32 {
        self.coverage.decode_len
    }

    pub fn validate(&self) -> Result<()> {
        if self.protocol_version == 0 {
            return Err(Error::UnsupportedVersion);
        }
        if self.protocol_id == Digest::ZERO
            || self.parameter_digest == Digest::ZERO
            || self.model_id == Digest::ZERO
            || self.graph_digest == Digest::ZERO
            || self.quantization_digest == Digest::ZERO
            || self.tables_digest == Digest::ZERO
        {
            return Err(Error::StatementMismatch);
        }
        if self.prompt_tokens.is_empty() {
            return Err(Error::StatementMismatch);
        }
        if self.output_tokens.len() != self.coverage.decode_len as usize {
            return Err(Error::StatementMismatch);
        }
        self.coverage.validate()
    }

    pub fn validate_single_shot(&self) -> Result<()> {
        self.validate()
    }
}

/// Public bundle as read by the verifier. It contains only public identities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicBundle {
    pub manifest_digest: Digest,
    pub protocol_id: Digest,
    pub protocol_version: u16,
    pub model_commitment_id: Digest,
    pub parameters_digest: Digest,
    pub complete: bool,
}

/// Constructible only by the public-bundle verifier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedPublicBundle {
    manifest_digest: Digest,
    protocol_id: Digest,
    protocol_version: u16,
    model_commitment_id: Digest,
    parameters_digest: Digest,
}

impl VerifiedPublicBundle {
    pub fn manifest_digest(&self) -> Digest {
        self.manifest_digest
    }

    pub fn protocol_id(&self) -> Digest {
        self.protocol_id
    }

    pub fn protocol_version(&self) -> u16 {
        self.protocol_version
    }

    pub fn model_commitment_id(&self) -> Digest {
        self.model_commitment_id
    }

    pub fn parameters_digest(&self) -> Digest {
        self.parameters_digest
    }
}

pub fn verify_public_bundle(bundle: &PublicBundle) -> Result<VerifiedPublicBundle> {
    if !bundle.complete {
        return Err(Error::IncompleteArtifact);
    }
    if bundle.protocol_version == 0 {
        return Err(Error::UnsupportedVersion);
    }
    if bundle.manifest_digest == Digest::ZERO
        || bundle.protocol_id == Digest::ZERO
        || bundle.model_commitment_id == Digest::ZERO
        || bundle.parameters_digest == Digest::ZERO
    {
        return Err(Error::ArtifactIntegrityMismatch);
    }
    Ok(VerifiedPublicBundle {
        manifest_digest: bundle.manifest_digest,
        protocol_id: bundle.protocol_id,
        protocol_version: bundle.protocol_version,
        model_commitment_id: bundle.model_commitment_id,
        parameters_digest: bundle.parameters_digest,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerificationResult {
    pub accepted: bool,
    pub statement_digest: Digest,
    pub output_policy: OutputPolicy,
}

/// Verifier context: policy, limits, progress and metrics only. No weight
/// reader and no prover state are reachable from here.
pub trait VerifierSessionContext {
    fn policy(&self) -> &VerifierPolicy;
    fn limits(&self) -> &DecodeLimits;
    fn progress(&self) -> &dyn ProgressSink;
    fn record_metric(&mut self, name: &str, value: u64);
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BudgetItem {
    pub kind: String,
    pub amount: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BudgetReservation {
    pub commitment: Digest,
    pub leaf: u64,
    pub generation: u64,
    pub run_id: String,
    pub items: Vec<BudgetItem>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExposureRecord {
    pub items: Vec<BudgetItem>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeaseOutcome {
    Completed,
    Failed,
    Cancelled,
}

/// A one-shot lease. It is not `Clone`; `finish` consumes it.
#[derive(Debug, PartialEq, Eq)]
pub struct StateLease {
    id: u64,
    commitment: Digest,
    leaf: u64,
    generation: u64,
    run_id: String,
}

impl StateLease {
    /// Only a state store should issue leases.
    pub fn new(
        id: u64,
        commitment: Digest,
        leaf: u64,
        generation: u64,
        run_id: impl Into<String>,
    ) -> Self {
        Self {
            id,
            commitment,
            leaf,
            generation,
            run_id: run_id.into(),
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn commitment(&self) -> Digest {
        self.commitment
    }

    pub fn leaf(&self) -> u64 {
        self.leaf
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }
}

pub trait ProverStateStore: Send {
    fn reserve(&mut self, request: &BudgetReservation) -> Result<StateLease>;
    fn mark_exposed(&mut self, lease: &StateLease, exposure: &ExposureRecord) -> Result<()>;
    fn finish(&mut self, lease: StateLease, outcome: LeaseOutcome) -> Result<()>;
}
