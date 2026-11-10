//! Isolated fixture BAT-to-FOLD algebraic/chain interface demonstration. Each
//! group member has its own fixture runtime-tree root; the ordered roots are
//! bound before the BAT challenge and the member openings are checked. The
//! roots are fixture expectations, not SEND receipts or asset roots.

use zkie_core::types::Digest;
use zkie_engine::pcs::request::chain::Chain;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::params::REQUEST_CHAIN_LABEL;
use zkie_ops::proof::base_final_session::BaseRejection;
use zkie_ops::proof::bat_fold_session::{
    run_bat_fold, run_bat_fold_with_base, run_bat_fold_with_base_and_provider,
    run_bat_fold_with_base_and_target_shift, BaseFinalRunError, BatFoldFixture,
};

fn chain(seed: &[u8]) -> Chain {
    Chain::begin(REQUEST_CHAIN_LABEL, &Digest::ZERO, seed, &[], None).unwrap()
}

#[test]
fn the_fixture_bat_row_enters_fold_natively() {
    let fixture = BatFoldFixture::new(7).unwrap();
    assert_eq!(
        fixture.recompute_root().unwrap(),
        fixture.start_message().root
    );
    let run = run_bat_fold(&fixture, chain(b"fixture bat fold")).unwrap();
    assert!(run.pending);
    assert!(!run.fold_decides_truth);
    assert!(!run.adapter_used);
    assert!(run.native_shared_codes);
    assert!(run.group_roots_bound);
    assert!(!run.single_group_root_claimed);
    assert!(run.openings_verified);
    assert!(run.member_roots_from_fixture);
    assert!(!run.real_send_receipts);
    assert!(!run.group_roots_authenticated);
    assert!(run.fold_messages_from_committed_members);
    assert!(run.sample_claims_registered);
    assert!(run.sample_claims_pending);
    assert!(run.sample_claims_from_received_answers);
    assert!(run.sample_answers_provider_constructed);
    assert!(!run.verifier_standard_answer_comparison);
    assert!(!run.send_judges_truth);
    assert!(run.sample_pads_committed);
    assert!(!run.sample_pad_hiding);
    assert_eq!(run.group_roots.len(), 3);
    assert_eq!(run.group_positions.len(), 3);
    assert_eq!(run.group_roots, fixture.recompute_member_roots().unwrap());
    assert_eq!(run.group_rounds, 3);
    assert_eq!(run.bat_chain_rounds, run.group_rounds + 1);
    assert_eq!(run.sample_claims_in_bat, 3);
    assert_eq!(run.sample_rows.len(), 3);
    assert_eq!(run.seeds.len(), 1);
    assert_eq!(run.answer_order.len(), 3);
    assert_eq!(run.provider_answers, run.received_answers);
    assert_eq!(&run.bat_input_targets[2..5], run.received_answers.as_slice());
    assert!(run.combined_matches_witness);
    assert!(run.fold_matches_witness);
    assert_eq!(run.fold_row.coefficients, run.combined.main);
    assert_eq!(run.fold_row.mask_coefficients, run.combined.mask);
    assert_eq!(run.fold_row.target, run.combined.target);
    assert_eq!(run.fold_entry_rounds, run.bat_chain_rounds);
    assert_eq!(run.fold_entry_value, run.bat_chain_value);
    assert_eq!(
        run.final_rounds,
        run.fold_entry_rounds + 1 + fixture.shape().k as u64
    );
    assert!(run.output.next_stage().is_err());
    println!(
        "roots={:?} positions={:?} seeds={:?} answers={:?} root={:?} sum={:?} x={:?} eps={:?} target={:?} auxiliary={:?}",
        run.group_roots,
        run.group_positions,
        run.seeds,
        run.received_answers,
        run.start.root,
        run.start.sum,
        run.x,
        run.epsilon,
        run.output.target,
        run.auxiliary_value
    );
}

#[test]
fn the_fixture_base_final_wiring_passes_but_stays_pending() {
    let fixture = BatFoldFixture::new(9).unwrap();
    let run = run_bat_fold_with_base(&fixture, chain(b"fixture base final")).unwrap();
    assert!(run.base.base_linear);
    assert!(run.base.fold_main);
    assert!(run.base.fold_mask);
    assert!(run.base.fold_fresh);
    assert!(run.base.openings_verified);
    assert!(run.base.member_openings_verified);
    assert!(run.base.fresh_opening_verified);
    assert!(run.base.rejection.is_none());
    assert!(run.base.pending);
    assert!(!run.base.complete_base);
    assert!(run.base.sampled_fold_relation_only);
    assert!(!run.base.complete_fold_binding);
    assert!(run.base.fresh_term_checked_against_fold_root);
    assert!(run.base.conditional_local_experiment);
    assert!(!run.base.cs_connected);
    assert!(!run.base.reference_iopp_path);
    assert!(run.base.chain_handoff);
    assert_eq!(run.base.entry_rounds, run.fold_exit_rounds);
    assert_eq!(run.base.entry_value, run.fold_exit_value);
    assert!(run.base.commit_rounds > run.base.entry_rounds);
    assert!(run.base.rows_rounds > run.base.commit_rounds);
    assert_eq!(run.base.check_rounds, run.base.rows_rounds);
    assert_eq!(run.base.check_value, run.base.rows_value);
    assert_eq!(run.base.base_positions.len(), 2);
    println!(
        "base fold_identity={:?} star_roots=[{:?},{:?}] member_roots={:?} fresh_root={:?} lambda={:?} target_star={:?} positions={:?}",
        run.base.fold_identity,
        run.base.star_root,
        run.base.star_mask_root,
        run.base.member_roots,
        run.base.fresh_root,
        run.base.lambda,
        run.base.target_star,
        run.base.base_positions
    );
}

#[test]
fn a_fixed_base_target_shift_is_rejected_at_the_target_layer() {
    let fixture = BatFoldFixture::new(10).unwrap();
    let rejection = match run_bat_fold_with_base_and_target_shift(
        &fixture,
        chain(b"fixture base target"),
        Fp3::ONE,
    ) {
        Err(BaseFinalRunError::Rejected(rejection)) => rejection,
        Err(BaseFinalRunError::Stage(error)) => panic!("stage error: {error:?}"),
        Ok(_) => panic!("the fixed target shift was not rejected"),
    };
    assert_eq!(rejection.layer, BaseRejection::TargetLinear);
    assert!(!rejection.base_linear);
    assert_eq!(rejection.base_positions.len(), 2);
    println!(
        "rejected base lambda={:?} target_star={:?} positions={:?} layer=TargetLinear",
        rejection.lambda, rejection.target_star, rejection.base_positions
    );
}

#[test]
fn a_wrong_upstream_answer_is_rejected_at_base() {
    let fixture = BatFoldFixture::new(11).unwrap();
    let rejection = match run_bat_fold_with_base_and_provider(
        &fixture,
        chain(b"fixture base wrong upstream"),
        |_order, _seeds, true_answers| {
            let mut answers = true_answers.to_vec();
            answers[0] = answers[0].add(&Fp3::ONE);
            Ok(answers)
        },
    ) {
        Err(BaseFinalRunError::Rejected(rejection)) => rejection,
        Err(BaseFinalRunError::Stage(error)) => panic!("stage error: {error:?}"),
        Ok(_) => panic!("the wrong upstream answer was not rejected"),
    };
    assert_eq!(rejection.layer, BaseRejection::TargetLinear);
    assert!(!rejection.base_linear);
    assert_eq!(rejection.base_positions.len(), 2);
    println!(
        "rejected upstream answer lambda={:?} target_star={:?} positions={:?} layer=TargetLinear path=wrong SEND answer -> BAT target -> FOLD output -> BASE",
        rejection.lambda, rejection.target_star, rejection.base_positions
    );
}

#[test]
fn a_wrong_bat_declaration_stays_pending_after_fold() {
    let fixture = BatFoldFixture::new_with_target_shift(8, Fp3::from_u64(5)).unwrap();
    let run = run_bat_fold(&fixture, chain(b"fixture pending declaration")).unwrap();
    assert!(run.target_shifted);
    assert!(!run.combined_matches_witness);
    assert!(!run.fold_matches_witness);
    assert!(run.pending);
    assert!(run.output.next_stage().is_err());
    assert_ne!(run.output.target, run.auxiliary_value);
}
