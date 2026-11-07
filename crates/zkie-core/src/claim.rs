//! Evaluation and linear claims, pending sets, and closure.
//!
//! Operators only submit pending claims. `ClosedClaims` is constructed by the
//! closure path, after the session has checked each claim against its oracle and
//! source relation.

use crate::common::field::ProtocolField;
use crate::error::Result;
use crate::oracle::OracleId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RelationId(pub u64);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvalClaim<F: ProtocolField> {
    pub oracle: OracleId,
    pub point: Vec<F>,
    pub value: F,
    pub source: RelationId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoefficientSpec {
    Public,
    ChallengeDerived { label: Vec<u8> },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearClaim<F: ProtocolField> {
    pub oracle: OracleId,
    pub coefficients: CoefficientSpec,
    pub value: F,
    pub source: RelationId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClaimKind {
    Evaluation,
    Linear,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Claim<F: ProtocolField> {
    Eval(EvalClaim<F>),
    Linear(LinearClaim<F>),
}

impl<F: ProtocolField> Claim<F> {
    pub fn oracle(&self) -> OracleId {
        match self {
            Claim::Eval(claim) => claim.oracle,
            Claim::Linear(claim) => claim.oracle,
        }
    }

    pub fn source(&self) -> RelationId {
        match self {
            Claim::Eval(claim) => claim.source,
            Claim::Linear(claim) => claim.source,
        }
    }

    pub fn kind(&self) -> ClaimKind {
        match self {
            Claim::Eval(_) => ClaimKind::Evaluation,
            Claim::Linear(_) => ClaimKind::Linear,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingClaims<F: ProtocolField> {
    claims: Vec<Claim<F>>,
}

impl<F: ProtocolField> Default for PendingClaims<F> {
    fn default() -> Self {
        Self { claims: Vec::new() }
    }
}

impl<F: ProtocolField> PendingClaims<F> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_claims(claims: Vec<Claim<F>>) -> Self {
        Self { claims }
    }

    pub fn push(&mut self, claim: Claim<F>) {
        self.claims.push(claim);
    }

    pub fn len(&self) -> usize {
        self.claims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }

    pub fn claims(&self) -> &[Claim<F>] {
        &self.claims
    }
}

/// Implemented by the protocol session. A claim is closed only after its oracle
/// and source relation have been checked.
pub trait ClaimCloser<F: ProtocolField> {
    fn close(&mut self, claim: &Claim<F>) -> Result<()>;
}

/// Closed claims. The fields are private and there is no public constructor;
/// the only path is [`close_claims`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosedClaims<F: ProtocolField> {
    claims: Vec<Claim<F>>,
}

impl<F: ProtocolField> ClosedClaims<F> {
    pub fn len(&self) -> usize {
        self.claims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }

    pub fn claims(&self) -> &[Claim<F>] {
        &self.claims
    }
}

pub fn close_claims<F: ProtocolField>(
    pending: PendingClaims<F>,
    closer: &mut dyn ClaimCloser<F>,
) -> Result<ClosedClaims<F>> {
    let PendingClaims { claims } = pending;
    let mut closed = Vec::with_capacity(claims.len());
    for claim in claims {
        closer.close(&claim)?;
        closed.push(claim);
    }
    Ok(ClosedClaims { claims: closed })
}
