//! Isolated fixture BAT-to-FOLD algebraic/chain interface demonstration. Each
//! group member has its own fixture runtime-tree root; the ordered roots are
//! bound before the BAT challenge and the member openings are checked. The
//! roots are fixture expectations, not SEND receipts or asset roots.

use zkie_core::types::Digest;
use zkie_engine::pcs::request::chain::Chain;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::params::REQUEST_CHAIN_LABEL;
use zkie_ops::proof::bat_fold_session::{run_bat_fold, BatFoldFixture};

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
    assert!(!run.sample_rows_are_send_receipts);
    assert!(!run.declarations_from_bound_messages);
    assert_eq!(run.group_roots.len(), 3);
    assert_eq!(run.group_positions.len(), 3);
    assert_eq!(run.group_roots, fixture.recompute_member_roots().unwrap());
    assert_eq!(run.group_rounds, 2);
    assert_eq!(run.bat_chain_rounds, run.group_rounds + 1);
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
        "roots={:?} positions={:?} root={:?} sum={:?} x={:?} eps={:?} target={:?} auxiliary={:?}",
        run.group_roots,
        run.group_positions,
        run.start.root,
        run.start.sum,
        run.x,
        run.epsilon,
        run.output.target,
        run.auxiliary_value
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
