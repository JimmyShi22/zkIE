use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_ops::compose::{prove_shard, verify_shard, Op, Store};

#[test]
fn sparse_indexer_topk_roundtrip() {
    let m = 4usize;
    let n = 4usize;
    let k = 2usize;
    let mut store = Store::new();
    let x = store.push((0..m * n).map(|i| Goldilocks::from_u64((i * 7 % 13) as u64)).collect::<Vec<_>>());
    let sel = store.push(vec![]);
    let thr = store.push(vec![]);
    let gate = store.push(vec![]);
    let d1 = store.push(vec![]);
    let d2 = store.push(vec![]);
    let ops = vec![Op::SparseIndexer { x, sel, thr, gate, d1, d2, m, n, k }];
    let mut rng = XorShift64::new(9);
    let proof = prove_shard(&mut store, &ops, &[], &mut rng);
    assert!(verify_shard(&[], &ops, &proof));
}
