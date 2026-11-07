//! KV addresses, append-only history, causal mask and RoPE offset.

#[path = "support/single_shot_fixture.rs"]
mod fixture;

use zkie_ops::single_shot::{prove_single_shot, verify_single_shot};

fn proof() -> (
    zkie_ops::single_shot::SingleShotPublic,
    zkie_ops::single_shot::SingleShotProof,
) {
    let decode_len = 2;
    let (public, openings, witnesses) = fixture::fixture(decode_len);
    let proof = prove_single_shot(&public, &witnesses, &openings, vec![0; decode_len]).unwrap();
    (public, proof)
}

#[test]
fn changing_a_kv_address_is_rejected() {
    let (public, base) = proof();
    for mutate in [0usize, 1, 2, 3, 4] {
        let mut tampered = base.clone();
        let address = &mut tampered.steps[1].kv_appends[0];
        match mutate {
            0 => address.layer = 1,
            1 => address.head = 1,
            2 => address.channel = 1,
            3 => address.position += 1,
            _ => address.kind = match address.kind {
                zkie_ops::single_shot::KvKind::Key => zkie_ops::single_shot::KvKind::Value,
                zkie_ops::single_shot::KvKind::Value => zkie_ops::single_shot::KvKind::Key,
            },
        }
        assert!(verify_single_shot(&public, &tampered).is_err());
    }
}

#[test]
fn appends_are_append_only_and_reads_must_exist() {
    let (public, base) = proof();

    let mut duplicate = base.clone();
    let previous = duplicate.steps[0].kv_appends[0];
    duplicate.steps[1].kv_appends.push(previous);
    assert!(verify_single_shot(&public, &duplicate).is_err());

    let mut missing_append = base.clone();
    missing_append.steps[1].kv_appends.pop();
    assert!(verify_single_shot(&public, &missing_append).is_err());

    let mut future_read = base.clone();
    future_read.steps[1].kv_reads[0].position += 10;
    assert!(verify_single_shot(&public, &future_read).is_err());
}

#[test]
fn mask_and_rope_offset_must_match_the_position() {
    let (public, base) = proof();

    let mut bad_mask = base.clone();
    bad_mask.steps[1].mask[0] = 0;
    assert!(verify_single_shot(&public, &bad_mask).is_err());

    let mut short_mask = base.clone();
    short_mask.steps[1].mask.pop();
    assert!(verify_single_shot(&public, &short_mask).is_err());

    let mut bad_rope = base.clone();
    bad_rope.steps[1].rope_offset += 1;
    assert!(verify_single_shot(&public, &bad_rope).is_err());
}
