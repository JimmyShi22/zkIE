use zkie_core::common::field::XorShift64;
use zkie_ops::compose::{prove_shard, verify_shard, Op, Store};

#[test]
fn engram_hash_roundtrip() {
    let mut store = Store::new();
    let tokens = store.push_idx(vec![10, 20, 30]);
    let out = store.push_idx(vec![]);
    let ops = vec![Op::EngramHash { tokens, out, seq: 3, max_ngram: 3, n_heads: 2 }];
    let mut rng = XorShift64::new(6);
    let proof = prove_shard(&mut store, &ops, &[], &mut rng);
    assert!(verify_shard(&[], &ops, &proof));
    assert_eq!(store.idx[out].len(), 3 * 2 * 2);
}
