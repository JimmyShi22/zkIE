//! Request-session lifecycle and delayed sampling contracts.
//!
//! The three chains are represented by one explicit state machine. A request
//! cannot enter shared challenges before its request chain is sealed, and the
//! session cannot reach its final state while claims remain open.

use zkie_core::common::field::ProtocolField;
use zkie_core::common::random::RandomSource;
use zkie_core::compute::ExecutionPort;
use zkie_core::error::{Error, Result};
use zkie_core::oracle::OracleId;
use zkie_core::session::{
    SessionState, SessionStateMachine, StateLease, Statement, VerificationResult,
    VerifiedPublicBundle, VerifierSessionContext,
};
use zkie_core::types::{Digest, TensorId};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SessionLifecycle {
    state: SessionStateMachine,
}

impl SessionLifecycle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn state(&self) -> SessionState {
        self.state.state()
    }

    pub fn claims_closed(&self) -> bool {
        self.state.claims_closed()
    }

    pub fn start_request(&mut self) -> Result<()> {
        self.state.advance(SessionState::RequestChainRunning)
    }

    pub fn seal_request(&mut self) -> Result<()> {
        self.state.advance(SessionState::RequestChainSealed)
    }

    pub fn freeze_batch(&mut self) -> Result<()> {
        self.state.advance(SessionState::BatchFrozen)
    }

    pub fn begin_shared(&mut self) -> Result<()> {
        self.state.advance(SessionState::SharedChainRunning)
    }

    pub fn seal_shared(&mut self) -> Result<()> {
        self.state.advance(SessionState::SharedChainSealed)
    }

    pub fn begin_final(&mut self) -> Result<()> {
        self.state.advance(SessionState::FinalChainRunning)
    }

    pub fn close_claims(&mut self) -> Result<()> {
        self.state.close_claims()
    }

    pub fn finalize(&mut self) -> Result<()> {
        self.state.advance(SessionState::Finalized)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightPass {
    First,
    Second,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightPassRequest {
    pub pass: WeightPass,
    pub layer_start: u32,
    pub layer_end: u32,
    pub tensor_ids: Vec<TensorId>,
    pub depends_on_seals: Vec<Digest>,
}

/// Two-pass weight input. The first pass runs before shared challenges; the
/// second pass may only read after its seals are in the chain.
pub trait TwoPassWeightInput {
    fn first_pass(&self) -> Result<Vec<WeightPassRequest>>;
    fn second_pass(&self, gate: &dyn SealGate) -> Result<Vec<WeightPassRequest>>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnQuery {
    pub oracle: OracleId,
    pub depends_on_seals: Vec<Digest>,
    pub count: usize,
}

pub trait SealGate {
    fn is_sealed(&self, seal: Digest) -> bool;
}

pub fn check_weight_pass_preconditions(
    request: &WeightPassRequest,
    gate: &dyn SealGate,
) -> Result<()> {
    if request.pass == WeightPass::Second
        && !request
            .depends_on_seals
            .iter()
            .all(|seal| gate.is_sealed(*seal))
    {
        return Err(Error::InvalidSessionState);
    }
    Ok(())
}

/// Check delayed-sampling preconditions without sampling anything. The oracle
/// must already be committed and every seal it depends on must be in the chain.
pub fn check_sampling_preconditions(
    query: &ColumnQuery,
    gate: &dyn SealGate,
    oracle_committed: bool,
) -> Result<()> {
    if !query
        .depends_on_seals
        .iter()
        .all(|seal| gate.is_sealed(*seal))
    {
        return Err(Error::InvalidSessionState);
    }
    if !oracle_committed {
        return Err(Error::InvalidSessionState);
    }
    Ok(())
}

/// The request-session protocol owns the transcript, randomness and private
/// state lease. Concrete messages are supplied by the request implementation.
pub trait RequestSessionProtocol<F: ProtocolField>: Send + Sync {
    type Session;
    type Sealed;
    type Shared;
    type Proof;

    fn begin_request(
        &self,
        public: &VerifiedPublicBundle,
        statement: &Statement,
        lease: StateLease,
        rng: Box<dyn RandomSource>,
    ) -> Result<Self::Session>;

    fn seal_request(&self, session: Self::Session) -> Result<Self::Sealed>;

    fn prove_shared(
        &self,
        requests: &mut [Self::Sealed],
        ctx: &mut dyn ExecutionPort<F>,
    ) -> Result<Self::Shared>;

    fn finalize_request(
        &self,
        request: Self::Sealed,
        shared: &Self::Shared,
        ctx: &mut dyn ExecutionPort<F>,
    ) -> Result<Self::Proof>;

    fn verify_request(
        &self,
        ctx: &mut dyn VerifierSessionContext,
        public: &VerifiedPublicBundle,
        statement: &Statement,
        proof: &Self::Proof,
    ) -> Result<VerificationResult>;
}

// ---------------------------------------------------------------------------
// Three-chain coordinator.
//
// The coordinator is the only writer of the three transcripts. Workers hand it
// messages; they never hold a chain and cannot advance the transcript.

use zkie_engine::pcs::request::chain::{Chain, KIND_SEED};
use zkie_engine::pcs::request::params::{
    FINAL_CHAIN_LABEL, REQUEST_CHAIN_LABEL, SHARED_CHAIN_LABEL,
};

use crate::proof::request::output_policy::statement_digest;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainRole {
    Request,
    Shared,
    Final,
}

pub struct ThreeChainCoordinator {
    lifecycle: SessionLifecycle,
    request: Chain,
    shared: Chain,
    final_chain: Chain,
    statement_digest: Digest,
    batch_root: Option<Digest>,
    shared_messages: u64,
    final_messages: u64,
    policy: ChainMessagePolicy,
}

impl ThreeChainCoordinator {
    pub fn new(statement: &Statement, parameters: &Digest) -> Result<Self> {
        let digest = statement_digest(statement)?;
        let request = Chain::begin(REQUEST_CHAIN_LABEL, parameters, &digest.0, &[], None)?;
        let shared = Chain::begin(SHARED_CHAIN_LABEL, parameters, &[], &[], None)?;
        let final_chain = Chain::begin(FINAL_CHAIN_LABEL, parameters, &[], &[], None)?;
        Ok(Self {
            lifecycle: SessionLifecycle::new(),
            request,
            shared,
            final_chain,
            statement_digest: digest,
            batch_root: None,
            shared_messages: 0,
            final_messages: 0,
            policy: ChainMessagePolicy::default(),
        })
    }

    pub fn state(&self) -> SessionState {
        self.lifecycle.state()
    }

    pub fn statement_digest(&self) -> Digest {
        self.statement_digest
    }

    pub fn request_value(&self) -> Digest {
        *self.request.value()
    }

    pub fn shared_value(&self) -> Digest {
        *self.shared.value()
    }

    pub fn final_value(&self) -> Digest {
        *self.final_chain.value()
    }

    pub fn batch_root(&self) -> Option<Digest> {
        self.batch_root
    }

    pub fn start_request(&mut self) -> Result<()> {
        self.lifecycle.start_request()
    }

    pub fn seal_request(&mut self) -> Result<()> {
        self.lifecycle.seal_request()
    }

    pub fn freeze_batch(&mut self, root: Digest) -> Result<()> {
        self.lifecycle.freeze_batch()?;
        self.batch_root = Some(root);
        Ok(())
    }

    pub fn begin_shared(&mut self) -> Result<()> {
        let root = self.batch_root.ok_or(Error::InvalidSessionState)?;
        self.lifecycle.begin_shared()?;
        let mut words = Vec::with_capacity(4);
        for chunk in root.0.chunks_exact(8) {
            words.push(u64::from_le_bytes(
                chunk.try_into().map_err(|_| Error::InvalidEncoding)?,
            ));
        }
        self.submit(ChainRole::Shared, KIND_SEED, &words)?;
        Ok(())
    }

    pub fn seal_shared(&mut self) -> Result<()> {
        if self.shared_messages == 0 {
            return Err(Error::InvalidSessionState);
        }
        self.lifecycle.seal_shared()
    }

    pub fn begin_final(&mut self) -> Result<()> {
        self.lifecycle.begin_final()
    }

    pub fn close_claims(&mut self) -> Result<()> {
        self.lifecycle.close_claims()
    }

    pub fn finalize(&mut self) -> Result<()> {
        self.lifecycle.finalize()
    }

    /// Submit one worker message to the chain that the current phase allows.
    pub fn submit(&mut self, role: ChainRole, kind: u32, words: &[u64]) -> Result<()> {
        let state = self.lifecycle.state();
        let chain = match role {
            ChainRole::Request if state == SessionState::RequestChainRunning => &mut self.request,
            ChainRole::Shared if state == SessionState::SharedChainRunning => &mut self.shared,
            ChainRole::Final if state == SessionState::FinalChainRunning => &mut self.final_chain,
            _ => return Err(Error::InvalidSessionState),
        };
        let mut message = chain.message();
        for word in words {
            message.push_u64(*word);
        }
        let prefix = message.prefix(kind, 0)?;
        if !chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        match role {
            ChainRole::Shared => self.shared_messages += 1,
            ChainRole::Final => self.final_messages += 1,
            ChainRole::Request => {}
        }
        Ok(())
    }

    pub fn shared_messages(&self) -> u64 {
        self.shared_messages
    }

    pub fn final_messages(&self) -> u64 {
        self.final_messages
    }
}

// ---------------------------------------------------------------------------
// Small relation graph entry points.

use zkie_core::claim::{LinearClaim, RelationId};
use zkie_engine::pcs::request::field::Fp3;

use crate::proof::binding::{OracleOpening, OracleRegistry};
use crate::proof::request::foldeq::{prove_fold_equations, verify_fold_equations, FoldConnection, FoldEqProof};
use crate::proof::request::lin::{prove_linear, verify_linear, LinearProof};
use crate::proof::request::lkp::{prove_lookup, verify_lookup, LookupProof, LookupTable};
use crate::proof::request::lns::{verify_range, limbs_for_values, RangeParams, RangeProof};
use crate::proof::request::output_policy::validate_statement_policy;

#[derive(Clone)]
pub struct RelationGraphPublic {
    pub statement: Statement,
    pub registry: OracleRegistry,
    pub lookup_table: LookupTable,
    pub lookup_queries: Vec<OracleOpening>,
    pub range_table: LookupTable,
    pub range_params: RangeParams,
    pub range_values: Vec<i64>,
    pub range_oracle: OracleId,
    pub range_openings: Vec<OracleOpening>,
    pub linear_claim: LinearClaim<Fp3>,
    pub linear_openings: Vec<OracleOpening>,
    pub public_coefficients: Vec<Fp3>,
    pub required_relations: Vec<RelationId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationGraphProof {
    pub lookup: LookupProof,
    pub range: RangeProof,
    pub linear: LinearProof,
    pub foldeq: FoldEqProof,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationGraphWitness {
    pub range_values: Vec<i64>,
    pub fold_connections: Vec<FoldConnection>,
}

/// Prove the small relation graph: table lookup, range check, linear claim and
/// fold connections.
pub fn prove_relation_graph(
    chain: &mut Chain,
    public: &RelationGraphPublic,
    witness: &RelationGraphWitness,
) -> Result<RelationGraphProof> {
    if witness.range_values != public.range_values {
        return Err(Error::StatementMismatch);
    }
    let lookup = prove_lookup(
        chain,
        &public.lookup_table,
        &public.lookup_queries,
        &public.registry,
    )?;
    let limbs = limbs_for_values(public.range_params, &public.range_values)?;
    let range_lookup = prove_lookup(
        chain,
        &public.range_table,
        &public.range_openings,
        &public.registry,
    )?;
    let range = RangeProof {
        values: public.range_values.clone(),
        limbs,
        oracle: public.range_oracle,
        openings: public.range_openings.clone(),
        lookup: range_lookup,
    };
    let linear = prove_linear(
        chain,
        &public.linear_claim,
        &public.linear_openings,
        &public.registry,
        &public.public_coefficients,
    )?;
    let foldeq = prove_fold_equations(witness.fold_connections.clone())?;
    Ok(RelationGraphProof {
        lookup,
        range,
        linear,
        foldeq,
    })
}

/// Verify the graph from public data and the proof only.
pub fn verify_relation_graph(
    chain: &mut Chain,
    public: &RelationGraphPublic,
    proof: &RelationGraphProof,
) -> Result<VerificationResult> {
    validate_statement_policy(&public.statement)?;
    verify_lookup(
        chain,
        &public.lookup_table,
        &public.lookup_queries,
        &public.registry,
        &proof.lookup,
    )?;
    if proof.range.values != public.range_values
        || proof.range.oracle != public.range_oracle
    {
        return Err(Error::StatementMismatch);
    }
    verify_range(
        chain,
        public.range_params,
        &proof.range,
        &public.range_table,
        &public.registry,
    )?;
    verify_linear(
        chain,
        &public.linear_claim,
        &proof.linear,
        &public.registry,
        &public.public_coefficients,
        public.linear_claim.source,
    )?;
    verify_fold_equations(
        &proof.foldeq,
        &public.required_relations,
        &public.registry,
    )?;
    Ok(VerificationResult {
        accepted: true,
        statement_digest: statement_digest(&public.statement)?,
        output_policy: public.statement.output_policy.clone(),
    })
}

// ---------------------------------------------------------------------------
// Per-message policy: work targets, OOD placement and two-pass scheduling.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainMessagePolicy {
    pub request_work_bits: u8,
    pub shared_work_bits: u8,
    pub final_work_bits: u8,
    pub ood_after_messages: u64,
}

impl Default for ChainMessagePolicy {
    fn default() -> Self {
        Self {
            request_work_bits: 0,
            shared_work_bits: 0,
            final_work_bits: 0,
            ood_after_messages: 1,
        }
    }
}

fn kind_is_allowed(role: ChainRole, kind: u32) -> bool {
    match role {
        ChainRole::Request => matches!(kind, 1 | 2 | 3 | 6 | 7),
        ChainRole::Shared => matches!(kind, 1 | 3 | 6 | 7),
        ChainRole::Final => matches!(kind, 1 | 2 | 4 | 5 | 6 | 7),
    }
}

impl ThreeChainCoordinator {
    pub fn new_with_policy(
        statement: &Statement,
        parameters: &Digest,
        policy: ChainMessagePolicy,
    ) -> Result<Self> {
        let mut coordinator = Self::new(statement, parameters)?;
        coordinator.policy = policy;
        Ok(coordinator)
    }

    pub fn message_policy(&self) -> ChainMessagePolicy {
        self.policy
    }

    /// Submit a message with the policy example work target for SEAL rounds.
    /// A failed work test rejects the nonce; no redraw happens here.
    pub fn submit_with_nonce(
        &mut self,
        role: ChainRole,
        kind: u32,
        words: &[u64],
        nonce: u64,
    ) -> Result<()> {
        let work_bits = if kind == 6 {
            match role {
                ChainRole::Request => self.policy.request_work_bits,
                ChainRole::Shared => self.policy.shared_work_bits,
                ChainRole::Final => self.policy.final_work_bits,
            }
        } else {
            0
        };
        self.submit_round(role, kind, words, work_bits, nonce)
    }

    /// Submit one round with its own instance work target. Every non-EMPTY
    /// round may carry work; an EMPTY round must have work_bits == 0.
    pub fn submit_round(
        &mut self,
        role: ChainRole,
        kind: u32,
        words: &[u64],
        work_bits: u8,
        nonce: u64,
    ) -> Result<()> {
        if !kind_is_allowed(role, kind) {
            return Err(Error::InvalidEncoding);
        }
        if kind == 7 && work_bits != 0 {
            return Err(Error::InvalidConfig);
        }
        let state = self.lifecycle.state();
        let chain = match role {
            ChainRole::Request if state == SessionState::RequestChainRunning => &mut self.request,
            ChainRole::Shared if state == SessionState::SharedChainRunning => &mut self.shared,
            ChainRole::Final if state == SessionState::FinalChainRunning => &mut self.final_chain,
            _ => return Err(Error::InvalidSessionState),
        };
        if work_bits > 64 {
            return Err(Error::InvalidConfig);
        }
        let mut message = chain.message();
        for word in words {
            message.push_u64(*word);
        }
        let prefix = message.prefix(kind, work_bits)?;
        if !chain.step(&prefix, nonce)? {
            return Err(Error::InvalidProof);
        }
        match role {
            ChainRole::Shared => self.shared_messages += 1,
            ChainRole::Final => self.final_messages += 1,
            ChainRole::Request => {}
        }
        Ok(())
    }

    /// OOD draws are only available on the final chain after the configured
    /// number of messages and commitments.
    pub fn draw_ood_point(&mut self) -> Result<Fp3> {
        if self.lifecycle.state() != SessionState::FinalChainRunning
            || self.final_messages < self.policy.ood_after_messages
        {
            return Err(Error::InvalidSessionState);
        }
        self.final_chain.draw_e()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TwoPassSchedule {
    first_done: bool,
    second_started: bool,
}

impl Default for TwoPassSchedule {
    fn default() -> Self {
        Self::new()
    }
}

impl TwoPassSchedule {
    pub fn new() -> Self {
        Self {
            first_done: false,
            second_started: false,
        }
    }

    pub fn complete_first(&mut self) -> Result<()> {
        if self.first_done {
            return Err(Error::InvalidSessionState);
        }
        self.first_done = true;
        Ok(())
    }

    /// The second weight pass may start only after the first pass finished and
    /// the shared chain is sealed.
    pub fn begin_second(&mut self, state: SessionState) -> Result<()> {
        if !self.first_done || self.second_started {
            return Err(Error::InvalidSessionState);
        }
        if !matches!(
            state,
            SessionState::SharedChainSealed
                | SessionState::FinalChainRunning
                | SessionState::Finalized
        ) {
            return Err(Error::InvalidSessionState);
        }
        self.second_started = true;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Enumerated message plan for the three chains.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MessageStep {
    pub role: ChainRole,
    pub kind: u32,
    pub work_bits: u8,
    pub ood: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MessagePlan {
    pub steps: Vec<MessageStep>,
}

pub fn default_message_plan(policy: ChainMessagePolicy) -> MessagePlan {
    MessagePlan {
        steps: vec![
            MessageStep { role: ChainRole::Request, kind: 1, work_bits: 0, ood: false },
            MessageStep { role: ChainRole::Request, kind: 6, work_bits: policy.request_work_bits, ood: false },
            MessageStep { role: ChainRole::Shared, kind: 3, work_bits: 0, ood: false },
            MessageStep { role: ChainRole::Shared, kind: 1, work_bits: 0, ood: false },
            MessageStep { role: ChainRole::Shared, kind: 6, work_bits: policy.shared_work_bits, ood: false },
            MessageStep { role: ChainRole::Final, kind: 4, work_bits: 0, ood: false },
            MessageStep { role: ChainRole::Final, kind: 1, work_bits: 0, ood: true },
            MessageStep { role: ChainRole::Final, kind: 1, work_bits: 0, ood: false },
            MessageStep { role: ChainRole::Final, kind: 6, work_bits: policy.final_work_bits, ood: false },
        ],
    }
}

/// The roles must appear in request, shared, final order; OOD occurs exactly
/// once, on the final chain, after at least one final message; work is only on
/// SEAL messages and is at most 64 bits.
pub fn validate_message_plan(plan: &MessagePlan) -> Result<()> {
    if plan.steps.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let mut last_role = 0u8;
    let mut ood_seen = false;
    let mut final_messages_before_ood = 0u64;
    for step in &plan.steps {
        let role = match step.role {
            ChainRole::Request => 1,
            ChainRole::Shared => 2,
            ChainRole::Final => 3,
        };
        if role < last_role {
            return Err(Error::StatementMismatch);
        }
        last_role = role;
        if !kind_is_allowed(step.role, step.kind) {
            return Err(Error::InvalidEncoding);
        }
        if step.work_bits > 64 {
            return Err(Error::InvalidConfig);
        }
        if step.kind == 7 && step.work_bits != 0 {
            return Err(Error::InvalidConfig);
        }
        if step.ood {
            if ood_seen || step.role != ChainRole::Final || final_messages_before_ood == 0 {
                return Err(Error::StatementMismatch);
            }
            ood_seen = true;
        }
        if step.role == ChainRole::Final && !step.ood {
            final_messages_before_ood += 1;
        }
    }
    if !ood_seen {
        return Err(Error::StatementMismatch);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Subproof plans and item-by-item message-plan binding.
//
// Each subproof owns a contiguous run of messages on exactly one chain role.
// Plans are compared item by item: number of rounds, kind, work placement and
// OOD placement must all agree. The resulting digest is the value a receipt
// binds to the plan.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubproofPlan {
    pub proof_id: u32,
    pub role: ChainRole,
    pub rounds: Vec<MessageStep>,
}

pub fn split_message_plan(plan: &MessagePlan) -> Result<Vec<SubproofPlan>> {
    validate_message_plan(plan)?;
    let mut groups: Vec<SubproofPlan> = Vec::new();
    for step in &plan.steps {
        match groups.last_mut() {
            Some(group) if group.role == step.role => group.rounds.push(*step),
            _ => groups.push(SubproofPlan {
                proof_id: groups.len() as u32,
                role: step.role,
                rounds: vec![*step],
            }),
        }
    }
    Ok(groups)
}

pub fn bind_subproof_plans(plan: &MessagePlan, subproofs: &[SubproofPlan]) -> Result<()> {
    let split = split_message_plan(plan)?;
    if split.len() != subproofs.len() {
        return Err(Error::StatementMismatch);
    }
    for (expected, actual) in split.iter().zip(subproofs) {
        if expected.proof_id != actual.proof_id
            || expected.role != actual.role
            || expected.rounds != actual.rounds
        {
            return Err(Error::StatementMismatch);
        }
    }
    Ok(())
}

pub fn message_plan_digest(plan: &MessagePlan) -> Result<Digest> {
    validate_message_plan(plan)?;
    let mut writer = zkie_engine::primitives::codec::CanonicalWriter::new();
    writer.put_u64(plan.steps.len() as u64);
    for step in &plan.steps {
        writer.put_u64(match step.role {
            ChainRole::Request => 1,
            ChainRole::Shared => 2,
            ChainRole::Final => 3,
        });
        writer.put_u64(step.kind as u64);
        writer.put_u64(step.work_bits as u64);
        writer.put_u64(u64::from(step.ood));
    }
    zkie_engine::primitives::transcript::hash_labeled(
        zkie_engine::primitives::transcript::tag_word(
            zkie_engine::primitives::transcript::STATEMENT,
            7,
        ),
        0,
        None,
        &writer.into_bytes(),
    )
}

pub fn subproof_plans_digest(subproofs: &[SubproofPlan]) -> Result<Digest> {
    let mut writer = zkie_engine::primitives::codec::CanonicalWriter::new();
    writer.put_u64(subproofs.len() as u64);
    for subproof in subproofs {
        writer.put_u64(subproof.proof_id as u64);
        writer.put_u64(match subproof.role {
            ChainRole::Request => 1,
            ChainRole::Shared => 2,
            ChainRole::Final => 3,
        });
        writer.put_u64(subproof.rounds.len() as u64);
        for round in &subproof.rounds {
            writer.put_u64(round.kind as u64);
            writer.put_u64(round.work_bits as u64);
            writer.put_u64(u64::from(round.ood));
        }
    }
    zkie_engine::primitives::transcript::hash_labeled(
        zkie_engine::primitives::transcript::tag_word(
            zkie_engine::primitives::transcript::STATEMENT,
            8,
        ),
        0,
        None,
        &writer.into_bytes(),
    )
}

// ---------------------------------------------------------------------------
// Relation receipt binding on the request chain.
//
// The receipt (relation id + message) is hashed into canonical words and
// submitted as an ALGEBRAIC round. A verifier replaying the same receipt on a
// coordinator in the same state obtains the same chain value; any byte change
// changes the submitted message.

use crate::proof::binding::RelationReceipt;

pub fn relation_receipt_words(receipt: &RelationReceipt) -> Result<Vec<u64>> {
    let mut writer = zkie_engine::primitives::codec::CanonicalWriter::new();
    writer.put_u64(receipt.relation.0);
    writer.put_bytes(&receipt.message);
    let digest = zkie_engine::primitives::transcript::hash_labeled(
        zkie_engine::primitives::transcript::tag_word(
            zkie_engine::primitives::transcript::STATEMENT,
            12,
        ),
        0,
        None,
        &writer.into_bytes(),
    )?;
    Ok(digest.0.chunks_exact(8).map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap_or([0; 8]))).collect())
}

pub fn submit_relation_receipt(
    coordinator: &mut ThreeChainCoordinator,
    role: ChainRole,
    receipt: &RelationReceipt,
    nonce: u64,
) -> Result<()> {
    let words = relation_receipt_words(receipt)?;
    coordinator.submit_round(role, 1, &words, 0, nonce)
}
