use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_ops::compose::{prove_shard, verify_shard, Op, Store};

#[test]
fn sort_selected_roundtrip() {
    let m = 1usize;
    let n = 4usize;
    let k = 2usize;
    let mut store = Store::new();
    let sel = store.push(vec![
        Goldilocks::from_u64(1),
        Goldilocks::ZERO,
        Goldilocks::from_u64(1),
        Goldilocks::ZERO,
    ]);
    let out = store.push_idx(vec![]);
    let ops = vec![Op::SortSelected { sel, out, m, n, k }];
    let mut rng = XorShift64::new(14);
    let proof = prove_shard(&mut store, &ops, &[], &mut rng);
    assert!(verify_shard(&[], &ops, &proof));
    assert_eq!(store.idx[out], vec![0, 2]);
}
