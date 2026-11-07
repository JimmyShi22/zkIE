//! Message-plan ordering, sumcheck accounting and execution-order determinism.

#[path = "support/single_shot_fixture.rs"]
mod fixture;

use zkie_core::error::Error;
use zkie_core::types::Digest;

use zkie_engine::parallel::{run_deterministic, LogicalTaskKey, TaskPhase, ThreadPool};
use zkie_engine::pcs::request::chain::Chain;
use zkie_engine::pcs::request::params::SHARED_CHAIN_LABEL;

use zkie_ops::proof::request::batt::{
    build_binding_tree, freeze_batch, freeze_batch_with_binding, shared_challenge, SealedMember,
};
use zkie_ops::proof::request::request::{
    default_message_plan, validate_message_plan, ChainMessagePolicy, ChainRole, MessagePlan, MessageStep,
};
use zkie_ops::proof::sumcheck::accounting;
use zkie_ops::single_shot::{prove_single_shot, StepWitness};

fn shared_chain() -> Chain {
    Chain::begin(SHARED_CHAIN_LABEL, &Digest([2u8; 32]), b"shared", &[], None).unwrap()
}

#[test]
fn message_plan_enforces_roles_ood_and_work() {
    let policy = ChainMessagePolicy::default();
    let plan = default_message_plan(policy);
    validate_message_plan(&plan).unwrap();

    let mut wrong_order = plan.clone();
    wrong_order.steps.swap(0, 2);
    assert!(validate_message_plan(&wrong_order).is_err());

    let mut early_ood = plan.clone();
    early_ood.steps[0].ood = true;
    assert!(validate_message_plan(&early_ood).is_err());

    // Work is a per-round instance parameter: any non-EMPTY round may carry it.
    let mut work_any_round = plan.clone();
    work_any_round.steps[0].work_bits = 8;
    validate_message_plan(&work_any_round).unwrap();

    // EMPTY rounds must have zero work.
    let mut empty_with_work = plan.clone();
    empty_with_work.steps[0].kind = 7;
    empty_with_work.steps[0].work_bits = 8;
    assert!(validate_message_plan(&empty_with_work).is_err());
    let mut empty_zero = plan.clone();
    empty_zero.steps[0].kind = 7;
    empty_zero.steps[0].work_bits = 0;
    validate_message_plan(&empty_zero).unwrap();

    let mut no_ood = plan.clone();
    for step in &mut no_ood.steps {
        step.ood = false;
    }
    assert!(validate_message_plan(&no_ood).is_err());
    assert!(matches!(
        validate_message_plan(&MessagePlan { steps: Vec::new() }),
        Err(Error::InvalidConfig)
    ));
}

#[test]
fn execution_order_does_not_change_the_transcript() {
    let decode_len = 4;
    let (public, openings, witnesses) = fixture::fixture(decode_len);
    let sequential = prove_single_shot(&public, &witnesses, &openings, vec![0; decode_len]).unwrap();

    let pool = ThreadPool::new(3).unwrap();
    let mut tasks = Vec::new();
    for (index, witness) in witnesses.iter().cloned().enumerate().rev() {
        tasks.push((
            LogicalTaskKey::new(0, TaskPhase::Prove, 0, 0, 0, index as u32),
            move || Ok(witness),
        ));
    }
    let results = run_deterministic(&pool, tasks).unwrap();
    let ordered: Vec<StepWitness> = results.into_iter().map(|result| result.value).collect();
    let reordered = prove_single_shot(&public, &ordered, &openings, vec![0; decode_len]).unwrap();
    assert_eq!(sequential, reordered);
    assert_eq!(accounting(decode_len as u32).rounds, 4);
}

#[test]
fn batch_root_can_be_the_binding_tree_root() {
    let configuration = Digest([3u8; 32]);
    let members = vec![
        SealedMember {
            request_id: Digest([1u8; 32]),
            seal: Digest([4u8; 32]),
            claims: Vec::new(),
        },
        SealedMember {
            request_id: Digest([2u8; 32]),
            seal: Digest([5u8; 32]),
            claims: Vec::new(),
        },
    ];
    let padding: Vec<Digest> = (0..2).map(|index| Digest([0xB0 + index; 32])).collect();
    let tree = build_binding_tree(
        1,
        configuration,
        &[
            (0, members[0].seal, vec![1u8; 40]),
            (1, members[1].seal, vec![2u8; 40]),
        ],
        &padding,
    )
    .unwrap();
    let seal_batch = freeze_batch(members.clone()).unwrap();
    let binding_batch = freeze_batch_with_binding(members, &tree, &[0, 1]).unwrap();
    assert_eq!(binding_batch.root, tree.root);
    assert_ne!(binding_batch.root, seal_batch.root);

    let mut first = shared_chain();
    let mut second = shared_chain();
    let first_challenge = shared_challenge(&mut first, &seal_batch).unwrap();
    let second_challenge = shared_challenge(&mut second, &binding_batch).unwrap();
    assert_ne!(first_challenge, second_challenge);
}

#[test]
fn message_step_is_copyable_and_ordered() {
    let step = MessageStep {
        role: ChainRole::Final,
        kind: 1,
        work_bits: 0,
        ood: false,
    };
    let copy = step;
    assert_eq!(step, copy);
}

#[test]
fn subproof_plans_bind_to_message_plan_item_by_item() {
    use zkie_ops::proof::request::request::{
        bind_subproof_plans, message_plan_digest, split_message_plan, subproof_plans_digest,
    };

    let plan = default_message_plan(ChainMessagePolicy::default());
    let subproofs = split_message_plan(&plan).unwrap();
    assert_eq!(subproofs.len(), 3);
    assert_eq!(subproofs[0].rounds.len(), 2);
    assert_eq!(subproofs[1].rounds.len(), 3);
    assert_eq!(subproofs[2].rounds.len(), 4);
    bind_subproof_plans(&plan, &subproofs).unwrap();
    assert!(subproofs.iter().flat_map(|subproof| &subproof.rounds).all(|round| {
        round.work_bits == 0 || round.kind == 6
    }));

    let mut wrong_ood = subproofs.clone();
    wrong_ood[2].rounds[1].ood = false;
    assert!(bind_subproof_plans(&plan, &wrong_ood).is_err());

    let mut wrong_kind = subproofs.clone();
    wrong_kind[1].rounds[0].kind = 4;
    assert!(bind_subproof_plans(&plan, &wrong_kind).is_err());

    let mut extra_round = subproofs.clone();
    let extra = extra_round[0].rounds[0]; extra_round[0].rounds.push(extra);
    assert!(bind_subproof_plans(&plan, &extra_round).is_err());

    let mut reordered = subproofs.clone();
    reordered.swap(0, 1);
    assert!(bind_subproof_plans(&plan, &reordered).is_err());

    let base_digest = message_plan_digest(&plan).unwrap();
    let mut changed_plan = plan.clone();
    changed_plan.steps[1].work_bits = 9;
    assert_ne!(message_plan_digest(&changed_plan).unwrap(), base_digest);
    let mut changed_rounds = subproofs.clone();
    changed_rounds[0].rounds[0].ood = false;
    changed_rounds[0].rounds[1].work_bits = 9;
    assert_ne!(
        subproof_plans_digest(&changed_rounds).unwrap(),
        subproof_plans_digest(&subproofs).unwrap()
    );
}

#[test]
fn every_round_carries_its_own_work_and_empty_rounds_are_zero() {
    use zkie_ops::proof::request::request::ThreeChainCoordinator;

    let statement = fixture::statement(1);
    let mut coordinator =
        ThreeChainCoordinator::new(&statement, &Digest([9u8; 32])).unwrap();
    coordinator.start_request().unwrap();
    assert!(matches!(
        coordinator.submit_round(ChainRole::Request, 7, &[1], 3, 0),
        Err(Error::InvalidConfig)
    ));
    let mut found = None;
    for nonce in 0..4096u64 {
        if coordinator
            .submit_round(ChainRole::Request, 1, &[1, 2], 4, nonce)
            .is_ok()
        {
            found = Some(nonce);
            break;
        }
    }
    assert!(found.is_some(), "a work nonce must exist on an ALGEBRAIC round");
    coordinator
        .submit_round(ChainRole::Request, 7, &[3], 0, 0)
        .unwrap();
}


#[test]
fn relation_receipts_are_bound_to_request_chain_messages() {
    use zkie_core::claim::RelationId;
    use zkie_ops::proof::binding::RelationReceipt;
    use zkie_ops::proof::request::request::{
        relation_receipt_words, submit_relation_receipt, ThreeChainCoordinator,
    };

    let receipt = RelationReceipt {
        relation: RelationId(7),
        message: b"relation-message".to_vec(),
    };
    let statement = fixture::statement(1);
    let mut first = ThreeChainCoordinator::new(&statement, &Digest([9u8; 32])).unwrap();
    first.start_request().unwrap();
    submit_relation_receipt(&mut first, ChainRole::Request, &receipt, 0).unwrap();

    let mut replay = ThreeChainCoordinator::new(&statement, &Digest([9u8; 32])).unwrap();
    replay.start_request().unwrap();
    submit_relation_receipt(&mut replay, ChainRole::Request, &receipt, 0).unwrap();
    assert_eq!(first.request_value(), replay.request_value());

    let mut tampered = receipt.clone();
    tampered.message[0] ^= 1;
    assert_ne!(
        relation_receipt_words(&receipt).unwrap(),
        relation_receipt_words(&tampered).unwrap()
    );
    let mut bad = ThreeChainCoordinator::new(&statement, &Digest([9u8; 32])).unwrap();
    bad.start_request().unwrap();
    submit_relation_receipt(&mut bad, ChainRole::Request, &tampered, 0).unwrap();
    assert_ne!(first.request_value(), bad.request_value());
}
