//! Fold-side contracts: FOLD rounds, OOD, proximity and hiding budgets.

use serde_json::Value;
use zkie_core::error::Error;
use zkie_core::types::Digest;

use zkie_engine::pcs::request::chain::{Chain, KIND_SEED};
use zkie_engine::pcs::request::code::CodeParams;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::fold::{
    fold_mask_table, fold_mask_weights, fresh_mask_tree, prove_fold, verify_fold, FoldClaim,
    FoldShape,
};
use zkie_engine::pcs::request::hiding::{ExposureKind, HidingLimits, HidingState};
use zkie_engine::pcs::request::merkle::open_columns;
use zkie_engine::pcs::request::ood::{
    mask_ood_answers, prove_ood_answer, sample_ood_point, verify_ood,
};
use zkie_engine::pcs::request::params::REQUEST_CHAIN_LABEL;
use zkie_engine::pcs::request::proximity::{
    check_openings, commit_oracle, sample_positions, ProximityParams,
};

fn vectors() -> Value {
    serde_json::from_str(include_str!("vectors/request/fold.json")).unwrap()
}

fn fp3(value: &Value) -> Fp3 {
    let entries = value.as_array().unwrap();
    let mut words = [0u64; 3];
    for (index, entry) in entries.iter().enumerate() {
        words[index] = entry.as_u64().unwrap();
    }
    Fp3::new(
        zkie_engine::pcs::request::field::fp_from_u64(words[0]),
        zkie_engine::pcs::request::field::fp_from_u64(words[1]),
        zkie_engine::pcs::request::field::fp_from_u64(words[2]),
    )
}

fn fp3s(value: &Value) -> Vec<Fp3> {
    value.as_array().unwrap().iter().map(fp3).collect()
}

fn challenge_chain() -> Chain {
    Chain::begin(
        REQUEST_CHAIN_LABEL,
        &Digest([1u8; 32]),
        b"fold-test",
        &[],
        None,
    )
    .unwrap()
}

fn inner(a: &[Fp3], b: &[Fp3]) -> Fp3 {
    let mut sum = Fp3::ZERO;
    for (left, right) in a.iter().zip(b) {
        sum = sum.add(&left.mul(right));
    }
    sum
}

fn claim() -> (FoldShape, FoldClaim, Fp3) {
    let shape = FoldShape::new(2, 1, 4).unwrap();
    let main: Vec<Fp3> = (1..=8u64)
        .map(|value| Fp3::from_u64(value))
        .collect();
    let coefficients_main: Vec<Fp3> = (1..=8u64)
        .map(|value| Fp3::from_u64(value * 2 + 1))
        .collect();
    let mask: Vec<Fp3> = (1..=8u64)
        .map(|value| Fp3::from_u64(value + 20))
        .collect();
    let coefficients_mask: Vec<Fp3> = (1..=8u64)
        .map(|value| Fp3::from_u64(value + 40))
        .collect();
    let kept = Fp3::from_u64(3);
    let target = inner(&coefficients_main, &main)
        .add(&inner(&coefficients_mask, &mask))
        .add(&kept);
    (
        shape,
        FoldClaim {
            main_rows: 4,
            main_cols: 2,
            main,
            coefficients_main,
            mask,
            coefficients_mask,
            kept,
            target,
        },
        target,
    )
}

#[test]
fn mask_polynomial_matches_vectors() {
    let vectors = vectors();
    let shape = FoldShape::new(2, 1, 4).unwrap();
    let coins = fp3s(&vectors["coins"]);
    let table = fold_mask_table(&shape, &coins).unwrap();
    let expected_rows = vectors["mask_table"].as_array().unwrap();
    let rows = shape.mask_rows();
    assert_eq!(table.len(), rows * shape.len_zk);
    for row in 0..rows {
        for column in 0..shape.len_zk {
            assert_eq!(
                table[row + rows * column],
                fp3(&expected_rows[row][column])
            );
        }
    }
    let challenges = fp3s(&vectors["weights_g"][0]);
    let weights = fold_mask_weights(&shape, &challenges).unwrap();
    let expected = fp3s(&vectors["weights_g"][1]);
    assert_eq!(weights, expected);
}

#[test]
fn fold_accepts_and_rejects_tampered_receipts() {
    let vectors = vectors();
    let (shape, claim, target) = claim();
    let coins = fp3s(&vectors["coins"]);
    let mut prover_chain = challenge_chain();
    let receipt = prove_fold(&shape, &claim, &coins, &mut prover_chain).unwrap();
    assert_eq!(receipt.mask_sum, fp3(&vectors["mask_sum"]));
    let mut verifier_chain = challenge_chain();
    verify_fold(&shape, target, &receipt, &mut verifier_chain).unwrap();

    let mut wrong_target = target;
    wrong_target = wrong_target.add(&Fp3::ONE);
    let mut chain = challenge_chain();
    assert!(matches!(
        verify_fold(&shape, wrong_target, &receipt, &mut chain),
        Err(Error::InvalidProof)
    ));

    let mut tampered = receipt.clone();
    tampered.rounds[0].polynomial[0] = tampered.rounds[0].polynomial[0].add(&Fp3::ONE);
    let mut chain = challenge_chain();
    assert!(verify_fold(&shape, target, &tampered, &mut chain).is_err());

    let mut tampered = receipt.clone();
    tampered.output.main[0] = tampered.output.main[0].add(&Fp3::ONE);
    let mut chain = challenge_chain();
    assert!(verify_fold(&shape, target, &tampered, &mut chain).is_err());

    let mut tampered = receipt.clone();
    tampered.output.target = tampered.output.target.add(&Fp3::ONE);
    let mut chain = challenge_chain();
    assert!(verify_fold(&shape, target, &tampered, &mut chain).is_err());

    let mut tampered = receipt.clone();
    tampered.fresh_root = Digest([9u8; 32]);
    let mut chain = challenge_chain();
    assert!(verify_fold(&shape, target, &tampered, &mut chain).is_err());

    let mut tampered = receipt.clone();
    tampered.output.coefficients_mask[0] = tampered.output.coefficients_mask[0].add(&Fp3::ONE);
    let mut chain = challenge_chain();
    assert!(verify_fold(&shape, target, &tampered, &mut chain).is_err());

    let mut chain = challenge_chain();
    assert!(matches!(
        verify_fold(&FoldShape::new(1, 1, 4).unwrap(), target, &receipt, &mut chain),
        Err(Error::ShapeMismatch)
    ));
}

#[test]
fn fresh_mask_commitment_opens_and_tampering_is_rejected() {
    let vectors = vectors();
    let shape = FoldShape::new(2, 1, 4).unwrap();
    let coins = fp3s(&vectors["coins"]);
    let tree = fresh_mask_tree(&shape, &coins).unwrap();
    let indices: Vec<u64> = (0..tree.leaves.len() as u64).collect();
    let opening = open_columns(&tree, &indices).unwrap();
    zkie_engine::pcs::request::merkle::verify_columns(
        &opening,
        &tree.root,
        tree.spec.symbol_count,
        tree.spec.salt_bytes,
    )
    .unwrap();

    let mut tampered = opening.clone();
    tampered.leaves[0].symbols[0] ^= 1;
    assert!(zkie_engine::pcs::request::merkle::verify_columns(
        &tampered,
        &tree.root,
        tree.spec.symbol_count,
        tree.spec.salt_bytes,
    )
    .is_err());
}

#[test]
fn ood_answers_are_masked_and_degenerate_points_are_not_redrawn() {
    let vectors = vectors();
    let case = &vectors["ood"];
    let plan = CodeParams {
        log_n: 2,
        ell: 2,
        r: 1,
    };
    let message = fp3s(&case["message"]);
    let tail = fp3s(&case["tail"]);
    let point = fp3(&case["point"]);
    let mask = fp3(&case["mask"]);
    let masked = prove_ood_answer(&plan, &message, &tail, point, mask).unwrap();
    assert_eq!(masked, fp3(&case["masked_answer"]));
    assert!(verify_ood(&plan, &message, &tail, point, mask, masked).unwrap());

    let mut wrong = masked;
    wrong = wrong.add(&Fp3::ONE);
    assert!(!verify_ood(&plan, &message, &tail, point, mask, wrong).unwrap());
    let other_point = point.add(&Fp3::ONE);
    assert!(!verify_ood(&plan, &message, &tail, other_point, mask, masked).unwrap());

    // A zero challenge is used as drawn: no redraw path exists.
    let zero = Fp3::ZERO;
    let zero_answer = prove_ood_answer(&plan, &message, &tail, zero, mask).unwrap();
    assert!(verify_ood(&plan, &message, &tail, zero, mask, zero_answer).unwrap());

    let several = mask_ood_answers(&[point, zero], &[mask, mask]).unwrap();
    assert_eq!(several.len(), 2);
    assert_eq!(several[1], mask);

    let mut chain = challenge_chain();
    let mut message = chain.message();
    message.push_u64(1);
    let prefix = message.prefix(KIND_SEED, 0).unwrap();
    assert!(chain.step(&prefix, 0).unwrap());
    let drawn = sample_ood_point(&mut chain).unwrap();
    let mut chain2 = challenge_chain();
    let mut message = chain2.message();
    message.push_u64(1);
    let prefix = message.prefix(KIND_SEED, 0).unwrap();
    assert!(chain2.step(&prefix, 0).unwrap());
    assert_eq!(drawn, sample_ood_point(&mut chain2).unwrap());
}

#[test]
fn sampling_requires_a_prior_commitment_and_checks_openings() {
    let vectors = vectors();
    let shape = FoldShape::new(2, 1, 4).unwrap();
    let coins = fp3s(&vectors["coins"]);
    let tree = fresh_mask_tree(&shape, &coins).unwrap();
    let params = ProximityParams::new(2, 1).unwrap();
    let mut chain = challenge_chain();

    let oracle = commit_oracle(&tree, chain.rounds());
    let log_n = tree.structure.leaf_count.trailing_zeros();
    assert!(matches!(
        sample_positions(&mut chain, &oracle, &params, log_n),
        Err(Error::InvalidSessionState)
    ));

    let mut message = chain.message();
    message.push_u64(1);
    let prefix = message.prefix(KIND_SEED, 0).unwrap();
    assert!(chain.step(&prefix, 0).unwrap());
    let positions = sample_positions(&mut chain, &oracle, &params, log_n).unwrap();
    assert_eq!(positions.len(), 2);
    let mut unique = positions.clone();
    unique.sort_unstable();
    unique.dedup();

    let opening = open_columns(&tree, &unique).unwrap();
    check_openings(
        &opening,
        &oracle,
        &positions,
        tree.spec.symbol_count,
        tree.spec.salt_bytes,
    )
    .unwrap();

    let mut wrong_root = opening.clone();
    wrong_root.root = Digest([2u8; 32]);
    assert!(check_openings(
        &wrong_root,
        &oracle,
        &positions,
        tree.spec.symbol_count,
        tree.spec.salt_bytes,
    )
    .is_err());
}

#[test]
fn fold_and_ood_reserve_budgets_per_exposed_object() {
    let limits = HidingLimits::new()
        .with(ExposureKind::Query, 2)
        .with(ExposureKind::Ood, 1);
    let mut state = HidingState::new(limits);
    let queries = state.reserve(ExposureKind::Query, 2).unwrap();
    state.record_exposure(queries, 2).unwrap();
    assert!(matches!(
        state.reserve(ExposureKind::Query, 1),
        Err(Error::LeafBudgetExhausted)
    ));
    let ood = state.reserve(ExposureKind::Ood, 1).unwrap();
    state.record_exposure(ood, 1).unwrap();
    assert!(matches!(
        state.reserve(ExposureKind::Ood, 1),
        Err(Error::LeafBudgetExhausted)
    ));
    state.validate().unwrap();
}
