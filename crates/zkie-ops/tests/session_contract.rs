//! Session-layer contract tests.
//!
//! All test doubles in this file are private to the test binary; none of them is
//! reachable as a production provider.

use std::collections::BTreeMap;

use zkie_core::claim::{
    close_claims, Claim, ClaimCloser, ClaimKind, ClosedClaims, CoefficientSpec, EvalClaim,
    LinearClaim, PendingClaims, RelationId,
};
use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing};
use zkie_core::error::{Error, Result};
use zkie_core::oracle::OracleId;
use zkie_core::session::{
    verify_public_bundle, BudgetItem, BudgetReservation, EosPolicy, ExposureRecord, LeaseOutcome,
    OutputPolicy, ProverStateStore, PublicBundle, SessionState, StateLease,
};
use zkie_core::types::{Digest, TensorId};
use zkie_ops::proof::request::request::{
    check_sampling_preconditions, check_weight_pass_preconditions, ColumnQuery, SealGate,
    SessionLifecycle, WeightPass, WeightPassRequest,
};
use zkie_ops::single_shot::{
    build_decode_coverage, build_single_shot_statement, validate_decode_coverage, DecodeRequest,
};

struct AcceptAll;

impl ClaimCloser<Goldilocks> for AcceptAll {
    fn close(&mut self, _claim: &Claim<Goldilocks>) -> Result<()> {
        Ok(())
    }
}

struct RejectAll;

impl ClaimCloser<Goldilocks> for RejectAll {
    fn close(&mut self, _claim: &Claim<Goldilocks>) -> Result<()> {
        Err(Error::InvalidProof)
    }
}

fn sample_eval_claim() -> Claim<Goldilocks> {
    Claim::Eval(EvalClaim {
        oracle: OracleId(1),
        point: vec![Goldilocks::from_u64(2)],
        value: Goldilocks::from_u64(3),
        source: RelationId(1),
    })
}

fn sample_linear_claim() -> Claim<Goldilocks> {
    Claim::Linear(LinearClaim {
        oracle: OracleId(2),
        coefficients: CoefficientSpec::Public,
        value: Goldilocks::from_u64(4),
        source: RelationId(2),
    })
}

#[test]
fn claims_close_only_through_the_closure_path() {
    let mut pending = PendingClaims::new();
    pending.push(sample_eval_claim());
    pending.push(sample_linear_claim());

    let closed: ClosedClaims<Goldilocks> = close_claims(pending, &mut AcceptAll).unwrap();
    assert_eq!(closed.len(), 2);
    assert_eq!(closed.claims()[0].kind(), ClaimKind::Evaluation);
    assert_eq!(closed.claims()[1].kind(), ClaimKind::Linear);
    assert_eq!(closed.claims()[0].oracle(), OracleId(1));
    assert_eq!(closed.claims()[1].source(), RelationId(2));
}

#[test]
fn closure_failure_does_not_yield_closed_claims() {
    let pending = PendingClaims::with_claims(vec![sample_eval_claim()]);
    assert!(matches!(
        close_claims(pending, &mut RejectAll),
        Err(Error::InvalidProof)
    ));
}

#[test]
fn three_chain_state_machine_rejects_illegal_order() {
    let mut lifecycle = SessionLifecycle::new();
    assert_eq!(lifecycle.state(), SessionState::Prepared);
    assert!(matches!(
        lifecycle.seal_request(),
        Err(Error::InvalidSessionState)
    ));

    lifecycle.start_request().unwrap();
    assert!(matches!(
        lifecycle.begin_shared(),
        Err(Error::InvalidSessionState)
    ));

    lifecycle.seal_request().unwrap();
    assert!(matches!(
        lifecycle.begin_shared(),
        Err(Error::InvalidSessionState)
    ));

    lifecycle.freeze_batch().unwrap();
    lifecycle.begin_shared().unwrap();
    lifecycle.seal_shared().unwrap();
    lifecycle.begin_final().unwrap();

    assert!(matches!(lifecycle.finalize(), Err(Error::UnclosedClaims)));
    lifecycle.close_claims().unwrap();
    lifecycle.finalize().unwrap();
    assert_eq!(lifecycle.state(), SessionState::Finalized);
}

#[test]
fn claims_can_only_be_closed_during_the_final_chain() {
    let mut lifecycle = SessionLifecycle::new();
    lifecycle.start_request().unwrap();
    assert!(matches!(
        lifecycle.close_claims(),
        Err(Error::InvalidSessionState)
    ));
}

#[derive(Default)]
struct TestGate {
    sealed: BTreeMap<Digest, bool>,
}

impl SealGate for TestGate {
    fn is_sealed(&self, seal: Digest) -> bool {
        self.sealed.get(&seal).copied().unwrap_or(false)
    }
}

#[test]
fn second_weight_pass_requires_its_seals() {
    let seal = Digest([2u8; 32]);
    let request = WeightPassRequest {
        pass: WeightPass::Second,
        layer_start: 0,
        layer_end: 1,
        tensor_ids: vec![TensorId(1)],
        depends_on_seals: vec![seal],
    };
    let mut gate = TestGate::default();
    assert!(matches!(
        check_weight_pass_preconditions(&request, &gate),
        Err(Error::InvalidSessionState)
    ));
    gate.sealed.insert(seal, true);
    check_weight_pass_preconditions(&request, &gate).unwrap();

    let first_pass = WeightPassRequest {
        pass: WeightPass::First,
        ..request
    };
    check_weight_pass_preconditions(&first_pass, &TestGate::default()).unwrap();
}

#[test]
fn sampling_requires_all_seals_and_a_committed_oracle() {
    let seal = Digest([1u8; 32]);
    let query = ColumnQuery {
        oracle: OracleId(1),
        depends_on_seals: vec![seal],
        count: 4,
    };
    let mut gate = TestGate::default();

    assert!(matches!(
        check_sampling_preconditions(&query, &gate, true),
        Err(Error::InvalidSessionState)
    ));

    gate.sealed.insert(seal, true);
    assert!(matches!(
        check_sampling_preconditions(&query, &gate, false),
        Err(Error::InvalidSessionState)
    ));
    check_sampling_preconditions(&query, &gate, true).unwrap();
}

#[derive(Clone, Debug)]
struct LeaseRecord {
    commitment: Digest,
    finished: bool,
}

#[derive(Clone, Debug, Default)]
struct TestStateStore {
    remaining: BTreeMap<String, u64>,
    leases: BTreeMap<u64, LeaseRecord>,
    next_id: u64,
}

impl TestStateStore {
    fn new(budget: Vec<BudgetItem>) -> Self {
        let mut remaining = BTreeMap::new();
        for item in budget {
            let entry = remaining.entry(item.kind).or_insert(0u64);
            *entry = entry.saturating_add(item.amount);
        }
        Self {
            remaining,
            leases: BTreeMap::new(),
            next_id: 0,
        }
    }

    fn remaining(&self, kind: &str) -> u64 {
        self.remaining.get(kind).copied().unwrap_or(0)
    }
}

impl ProverStateStore for TestStateStore {
    fn reserve(&mut self, request: &BudgetReservation) -> Result<StateLease> {
        let mut requested: BTreeMap<String, u64> = BTreeMap::new();
        for item in &request.items {
            let entry = requested.entry(item.kind.clone()).or_insert(0u64);
            *entry = entry
                .checked_add(item.amount)
                .ok_or(Error::ResourceLimit)?;
        }
        for (kind, amount) in &requested {
            if *amount > self.remaining(kind) {
                return Err(Error::LeafBudgetExhausted);
            }
        }
        for (kind, amount) in &requested {
            if let Some(entry) = self.remaining.get_mut(kind) {
                *entry = entry
                    .checked_sub(*amount)
                    .ok_or(Error::LeafBudgetExhausted)?;
            }
        }
        self.next_id = self.next_id.checked_add(1).ok_or(Error::ResourceLimit)?;
        let lease = StateLease::new(
            self.next_id,
            request.commitment,
            request.leaf,
            request.generation,
            request.run_id.clone(),
        );
        self.leases.insert(
            lease.id(),
            LeaseRecord {
                commitment: request.commitment,
                finished: false,
            },
        );
        Ok(lease)
    }

    fn mark_exposed(&mut self, lease: &StateLease, _exposure: &ExposureRecord) -> Result<()> {
        let record = self
            .leases
            .get_mut(&lease.id())
            .ok_or(Error::ProverStateMissing)?;
        if record.finished {
            return Err(Error::InvalidSessionState);
        }
        if record.commitment != lease.commitment() {
            return Err(Error::StateRollbackSuspected);
        }
        Ok(())
    }

    fn finish(&mut self, lease: StateLease, _outcome: LeaseOutcome) -> Result<()> {
        let record = self
            .leases
            .get_mut(&lease.id())
            .ok_or(Error::ProverStateMissing)?;
        if record.finished {
            return Err(Error::InvalidSessionState);
        }
        if record.commitment != lease.commitment() {
            return Err(Error::StateRollbackSuspected);
        }
        record.finished = true;
        Ok(())
    }
}

// A StateLease deliberately does not implement Clone; it is moved into one
// request (or consumed by finish). The forged-lease check below covers the
// store-side identity check as well.
#[test]
fn lease_is_single_use_and_budget_is_conserved() {
    let mut store = TestStateStore::new(vec![BudgetItem {
        kind: "query".into(),
        amount: 10,
    }]);
    let reservation = BudgetReservation {
        commitment: Digest([7u8; 32]),
        leaf: 1,
        generation: 0,
        run_id: "run-1".into(),
        items: vec![BudgetItem {
            kind: "query".into(),
            amount: 6,
        }],
    };
    let exposure = ExposureRecord {
        items: vec![BudgetItem {
            kind: "query".into(),
            amount: 2,
        }],
    };

    let lease = store.reserve(&reservation).unwrap();
    let lease_id = lease.id();
    store.mark_exposed(&lease, &exposure).unwrap();

    let forged = StateLease::new(
        lease_id,
        Digest([9u8; 32]),
        reservation.leaf,
        reservation.generation,
        reservation.run_id.clone(),
    );
    assert!(matches!(
        store.mark_exposed(&forged, &exposure),
        Err(Error::StateRollbackSuspected)
    ));

    store.finish(lease, LeaseOutcome::Failed).unwrap();
    assert_eq!(store.remaining("query"), 4);

    let finished_lease = StateLease::new(
        lease_id,
        reservation.commitment,
        reservation.leaf,
        reservation.generation,
        reservation.run_id.clone(),
    );
    assert!(matches!(
        store.mark_exposed(&finished_lease, &exposure),
        Err(Error::InvalidSessionState)
    ));

    let too_large = BudgetReservation {
        items: vec![BudgetItem {
            kind: "query".into(),
            amount: 5,
        }],
        ..reservation
    };
    assert!(matches!(
        store.reserve(&too_large),
        Err(Error::LeafBudgetExhausted)
    ));
}

#[test]
fn verified_public_bundle_requires_complete_public_materials() {
    let bundle = PublicBundle {
        manifest_digest: Digest([1u8; 32]),
        protocol_id: Digest([2u8; 32]),
        protocol_version: 1,
        model_commitment_id: Digest([3u8; 32]),
        parameters_digest: Digest([4u8; 32]),
        complete: true,
    };
    let verified = verify_public_bundle(&bundle).unwrap();
    assert_eq!(verified.manifest_digest(), bundle.manifest_digest);
    assert_eq!(verified.model_commitment_id(), bundle.model_commitment_id);

    let incomplete = PublicBundle {
        complete: false,
        ..bundle.clone()
    };
    assert!(matches!(
        verify_public_bundle(&incomplete),
        Err(Error::IncompleteArtifact)
    ));

    let zero_model = PublicBundle {
        model_commitment_id: Digest::ZERO,
        ..bundle
    };
    assert!(matches!(
        verify_public_bundle(&zero_model),
        Err(Error::ArtifactIntegrityMismatch)
    ));
}

fn decode_request(decode_len: u32) -> DecodeRequest {
    DecodeRequest {
        prompt_tokens: vec![1, 2, 3],
        output_tokens: (0..decode_len).map(|index| index + 10).collect(),
        eos_policy: EosPolicy::FixedLength,
        output_policy: OutputPolicy::ExactMatch,
    }
}

#[test]
fn one_statement_covers_every_decode_step() {
    for decode_len in [1u32, 2, 4, 8, 16, 512] {
        let request = decode_request(decode_len);
        let statement = build_single_shot_statement(
            Digest([1u8; 32]),
            1,
            Digest([2u8; 32]),
            Digest([3u8; 32]),
            Digest([4u8; 32]),
            Digest([5u8; 32]),
            Digest([6u8; 32]),
            &request,
        )
        .unwrap();
        assert_eq!(statement.decode_len(), decode_len);
        assert_eq!(
            statement.coverage.covered_positions.len(),
            decode_len as usize
        );
        validate_decode_coverage(&statement).unwrap();
    }
}

#[test]
fn zero_or_incomplete_coverage_is_rejected() {
    assert!(build_decode_coverage(0).is_err());

    let request = decode_request(16);
    let mut statement = build_single_shot_statement(
        Digest([1u8; 32]),
        1,
        Digest([2u8; 32]),
        Digest([3u8; 32]),
        Digest([4u8; 32]),
        Digest([5u8; 32]),
        Digest([6u8; 32]),
        &request,
    )
    .unwrap();
    statement.coverage.covered_positions.pop();
    assert!(matches!(
        validate_decode_coverage(&statement),
        Err(Error::StatementMismatch)
    ));

    let mut duplicate = statement.clone();
    duplicate.coverage.covered_positions.push(14);
    assert!(matches!(
        validate_decode_coverage(&duplicate),
        Err(Error::StatementMismatch)
    ));
}
