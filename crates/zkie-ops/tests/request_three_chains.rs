//! Three-chain ordering, batch sharing and verifier replay.

use std::collections::BTreeSet;

use serde_json::Value;
use zkie_core::artifact::digest_from_hex;
use zkie_core::claim::RelationId;
use zkie_core::error::Error;
use zkie_core::session::{
    DecodeCoverage, EosPolicy, OutputPolicy, Statement,
};
use zkie_core::types::Digest;

use zkie_engine::pcs::request::chain::Chain;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::params::SHARED_CHAIN_LABEL;

use zkie_ops::proof::request::batt::{
    build_binding_tree, combine_claims, freeze_batch, open_slot, prove_shared, verify_shared,
    verify_slot, SealedMember,
};
use zkie_ops::proof::request::request::{
    ChainMessagePolicy, ChainRole, ThreeChainCoordinator, TwoPassSchedule,
};
use zkie_ops::proof::request::request::{
    check_weight_pass_preconditions, SealGate, WeightPass, WeightPassRequest,
};

fn vectors() -> Value {
    serde_json::from_str(include_str!("vectors/request/chains.json")).unwrap()
}

fn statement() -> Statement {
    Statement {
        protocol_id: Digest([1u8; 32]),
        protocol_version: 1,
        parameter_digest: Digest([2u8; 32]),
        model_id: Digest([3u8; 32]),
        graph_digest: Digest([4u8; 32]),
        quantization_digest: Digest([5u8; 32]),
        tables_digest: Digest([6u8; 32]),
        prompt_tokens: vec![1, 2],
        output_tokens: vec![7],
        coverage: DecodeCoverage::full(1),
        eos_policy: EosPolicy::FixedLength,
        output_policy: OutputPolicy::ExactMatch,
    }
}

fn words(value: &Value) -> Vec<u64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry.as_u64().unwrap())
        .collect()
}

fn seals(vectors: &Value) -> Vec<Digest> {
    vectors["batch_seals_hex"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| digest_from_hex(value.as_str().unwrap()).unwrap())
        .collect()
}

fn member(seal: Digest, index: usize) -> SealedMember {
    SealedMember {
        request_id: Digest([index as u8 + 1; 32]),
        seal,
        claims: vec![
            (RelationId(1), Fp3::from_u64(index as u64 + 1)),
            (RelationId(2), Fp3::from_u64(index as u64 + 10)),
        ],
    }
}

fn shared_chain() -> Chain {
    Chain::begin(SHARED_CHAIN_LABEL, &Digest([2u8; 32]), b"shared", &[], None).unwrap()
}

#[test]
fn coordinator_enforces_chain_order_and_replay() {
    let vectors = vectors();
    let request_message = words(&vectors["request_messages"][0]);
    let shared_message = words(&vectors["shared_messages"][0]);
    let final_message = words(&vectors["final_messages"][0]);
    let parameters = Digest([2u8; 32]);
    let statement = statement();

    let mut coordinator = ThreeChainCoordinator::new(&statement, &parameters).unwrap();
    assert!(matches!(
        coordinator.submit(ChainRole::Request, 1, &[1]),
        Err(Error::InvalidSessionState)
    ));
    coordinator.start_request().unwrap();
    assert!(matches!(
        coordinator.submit(ChainRole::Shared, 1, &[1]),
        Err(Error::InvalidSessionState)
    ));
    coordinator.submit(ChainRole::Request, 1, &request_message).unwrap();
    coordinator.seal_request().unwrap();
    assert!(matches!(
        coordinator.submit(ChainRole::Request, 1, &[1]),
        Err(Error::InvalidSessionState)
    ));

    let batch = freeze_batch(vec![member(seals(&vectors)[0], 0), member(seals(&vectors)[1], 1)]).unwrap();
    coordinator.freeze_batch(batch.root).unwrap();
    coordinator.begin_shared().unwrap();
    coordinator.submit(ChainRole::Shared, 1, &shared_message).unwrap();
    coordinator.seal_shared().unwrap();
    coordinator.begin_final().unwrap();
    coordinator.submit(ChainRole::Final, 1, &final_message).unwrap();
    assert!(matches!(coordinator.finalize(), Err(Error::UnclosedClaims)));
    coordinator.close_claims().unwrap();
    coordinator.finalize().unwrap();

    // A verifier replays the public message log on its own coordinator.
    let mut replay = ThreeChainCoordinator::new(&statement, &parameters).unwrap();
    replay.start_request().unwrap();
    replay.submit(ChainRole::Request, 1, &request_message).unwrap();
    replay.seal_request().unwrap();
    replay.freeze_batch(batch.root).unwrap();
    replay.begin_shared().unwrap();
    replay.submit(ChainRole::Shared, 1, &shared_message).unwrap();
    replay.seal_shared().unwrap();
    replay.begin_final().unwrap();
    replay.submit(ChainRole::Final, 1, &final_message).unwrap();
    replay.close_claims().unwrap();
    replay.finalize().unwrap();
    assert_eq!(coordinator.request_value(), replay.request_value());
    assert_eq!(coordinator.shared_value(), replay.shared_value());
    assert_eq!(coordinator.final_value(), replay.final_value());
    assert_eq!(coordinator.statement_digest(), replay.statement_digest());

    let mut early = ThreeChainCoordinator::new(&statement, &parameters).unwrap();
    early.start_request().unwrap();
    early.seal_request().unwrap();
    early.freeze_batch(batch.root).unwrap();
    early.begin_shared().unwrap();
    early.seal_shared().unwrap();
    early.begin_final().unwrap();
    assert!(matches!(early.finalize(), Err(Error::UnclosedClaims)));
}

#[test]
fn batches_of_one_two_and_four_share_claims() {
    let vectors = vectors();
    let seals = seals(&vectors);
    for size in [1usize, 2, 4] {
        let members: Vec<SealedMember> = (0..size).map(|index| member(seals[index], index)).collect();
        let batch = freeze_batch(members).unwrap();
        let expected_root = batch.root;
        let mut prover_chain = shared_chain();
        let receipt = prove_shared(&mut prover_chain, &batch).unwrap();
        let mut verifier_chain = shared_chain();
        verify_shared(
            &mut verifier_chain,
            &expected_root,
            &batch,
            &receipt,
            &[RelationId(1), RelationId(2)],
        )
        .unwrap();

        // The combined value is the expected power combination.
        let mut power = Fp3::ONE;
        let mut expected = Fp3::ZERO;
        for member in &batch.members {
            expected = expected.add(&power.mul(&member.claims[0].1));
            power = power.mul(&receipt.challenge);
        }
        assert_eq!(receipt.combined[0].1, expected);
        assert_eq!(combine_claims(&batch.members, receipt.challenge).unwrap(), receipt.combined);

        // Reordering changes the frozen root.
        let mut reordered = batch.members.clone();
        reordered.reverse();
        let reordered_batch = freeze_batch(reordered).unwrap();
        if size > 1 {
            assert_ne!(reordered_batch.root, expected_root);
            let mut chain = shared_chain();
            assert!(verify_shared(
                &mut chain,
                &expected_root,
                &reordered_batch,
                &receipt,
                &[RelationId(1), RelationId(2)],
            )
            .is_err());
        }

        // A changed seal changes the root.
        let mut changed = batch.members.clone();
        changed[0].seal = Digest([0xEE; 32]);
        let changed_batch = freeze_batch(changed).unwrap();
        let mut chain = shared_chain();
        assert!(verify_shared(
            &mut chain,
            &expected_root,
            &changed_batch,
            &receipt,
            &[RelationId(1), RelationId(2)],
        )
        .is_err());

        // A tampered combined value is rejected.
        let mut tampered = receipt.clone();
        tampered.combined[0].1 = tampered.combined[0].1.add(&Fp3::ONE);
        let mut chain = shared_chain();
        assert!(verify_shared(
            &mut chain,
            &expected_root,
            &batch,
            &tampered,
            &[RelationId(1), RelationId(2)],
        )
        .is_err());

        // A different shared chain gives a different challenge.
        let mut other_chain = Chain::begin(
            SHARED_CHAIN_LABEL,
            &Digest([9u8; 32]),
            b"shared",
            &[],
            None,
        )
        .unwrap();
        assert!(verify_shared(
            &mut other_chain,
            &expected_root,
            &batch,
            &receipt,
            &[RelationId(1), RelationId(2)],
        )
        .is_err());

        // Missing a required relation is rejected.
        let mut chain = shared_chain();
        assert!(verify_shared(
            &mut chain,
            &expected_root,
            &batch,
            &receipt,
            &[RelationId(1)],
        )
        .is_err());
    }

    let mut mismatched = vec![member(seals[0], 0), member(seals[1], 1)];
    mismatched[1].claims.pop();
    assert!(freeze_batch(mismatched.clone()).is_ok());
    let batch = freeze_batch(mismatched).unwrap();
    let mut chain = shared_chain();
    assert!(prove_shared(&mut chain, &batch).is_err());
}

struct SealedSet(BTreeSet<Digest>);

impl SealGate for SealedSet {
    fn is_sealed(&self, seal: Digest) -> bool {
        self.0.contains(&seal)
    }
}

#[test]
fn two_pass_weight_reads_respect_seals() {
    let seal = Digest([4u8; 32]);
    let request = WeightPassRequest {
        pass: WeightPass::Second,
        layer_start: 0,
        layer_end: 4,
        tensor_ids: Vec::new(),
        depends_on_seals: vec![seal],
    };
    let empty = SealedSet(BTreeSet::new());
    assert!(matches!(
        check_weight_pass_preconditions(&request, &empty),
        Err(Error::InvalidSessionState)
    ));
    let sealed = SealedSet(BTreeSet::from([seal]));
    check_weight_pass_preconditions(&request, &sealed).unwrap();
    let first = WeightPassRequest {
        pass: WeightPass::First,
        ..request
    };
    check_weight_pass_preconditions(&first, &empty).unwrap();
}

#[test]
fn binding_tree_slots_and_padding_are_checked() {
    let vectors = vectors();
    let seals = seals(&vectors);
    let configuration = Digest([3u8; 32]);
    let salt0 = vec![1u8; 40];
    let salt1 = vec![2u8; 40];
    let occupied = vec![
        (0u64, seals[0], salt0.clone()),
        (2u64, seals[1], salt1.clone()),
    ];
    let padding: Vec<Digest> = (0..4).map(|index| Digest([0xA0 + index as u8; 32])).collect();
    let tree = build_binding_tree(2, configuration, &occupied, &padding).unwrap();

    let proof0 = open_slot(&tree, 0, &seals[0], salt0).unwrap();
    verify_slot(&tree.root, &configuration, &proof0).unwrap();
    let proof1 = open_slot(&tree, 2, &seals[1], salt1.clone()).unwrap();
    verify_slot(&tree.root, &configuration, &proof1).unwrap();

    let mut tampered_salt = proof0.clone();
    tampered_salt.salt[0] ^= 1;
    assert!(verify_slot(&tree.root, &configuration, &tampered_salt).is_err());
    let mut tampered_path = proof1.clone();
    tampered_path.path[0] = Digest([9u8; 32]);
    assert!(verify_slot(&tree.root, &configuration, &tampered_path).is_err());
    assert!(verify_slot(&Digest([8u8; 32]), &configuration, &proof0).is_err());
    assert!(build_binding_tree(
        2,
        configuration,
        &[(0, seals[0], vec![1u8; 40]), (0, seals[1], vec![2u8; 40])],
        &padding,
    )
    .is_err());
    assert!(build_binding_tree(2, configuration, &occupied, &padding[..2]).is_err());
}

#[test]
fn coordinator_policy_covers_work_ood_and_two_pass() {
    let vectors = vectors();
    let request_message = words(&vectors["request_messages"][0]);
    let parameters = Digest([2u8; 32]);
    let statement = statement();
    let policy = ChainMessagePolicy {
        request_work_bits: 0,
        shared_work_bits: 4,
        final_work_bits: 0,
        ood_after_messages: 1,
    };
    let mut coordinator = ThreeChainCoordinator::new_with_policy(&statement, &parameters, policy).unwrap();
    coordinator.start_request().unwrap();
    coordinator
        .submit_with_nonce(ChainRole::Request, 1, &request_message, 0)
        .unwrap();
    coordinator.seal_request().unwrap();
    let batch = freeze_batch(vec![member(seals(&vectors)[0], 0)]).unwrap();
    coordinator.freeze_batch(batch.root).unwrap();
    coordinator.begin_shared().unwrap();
    let mut found = None;
    for nonce in 0..4096u64 {
        if coordinator
            .submit_with_nonce(ChainRole::Shared, 6, &[1, 2], nonce)
            .is_ok()
        {
            found = Some(nonce);
            break;
        }
    }
    let nonce = found.expect("a work nonce must exist");
    coordinator.seal_shared().unwrap();
    coordinator.begin_final().unwrap();
    assert!(matches!(
        coordinator.draw_ood_point(),
        Err(Error::InvalidSessionState)
    ));
    assert!(matches!(
        coordinator.submit_with_nonce(ChainRole::Final, 3, &[1], 0),
        Err(Error::InvalidEncoding)
    ));
    coordinator
        .submit_with_nonce(ChainRole::Final, 1, &[7], 0)
        .unwrap();
    let ood = coordinator.draw_ood_point().unwrap();

    let mut replay = ThreeChainCoordinator::new_with_policy(&statement, &parameters, policy).unwrap();
    replay.start_request().unwrap();
    replay
        .submit_with_nonce(ChainRole::Request, 1, &request_message, 0)
        .unwrap();
    replay.seal_request().unwrap();
    replay.freeze_batch(batch.root).unwrap();
    replay.begin_shared().unwrap();
    replay
        .submit_with_nonce(ChainRole::Shared, 6, &[1, 2], nonce)
        .unwrap();
    replay.seal_shared().unwrap();
    replay.begin_final().unwrap();
    replay
        .submit_with_nonce(ChainRole::Final, 1, &[7], 0)
        .unwrap();
    assert_eq!(ood, replay.draw_ood_point().unwrap());

    let mut schedule = TwoPassSchedule::new();
    assert!(schedule.begin_second(zkie_core::session::SessionState::SharedChainSealed).is_err());
    schedule.complete_first().unwrap();
    assert!(schedule.begin_second(zkie_core::session::SessionState::SharedChainRunning).is_err());
    schedule
        .begin_second(zkie_core::session::SessionState::SharedChainSealed)
        .unwrap();
    assert!(schedule
        .begin_second(zkie_core::session::SessionState::SharedChainSealed)
        .is_err());
}
