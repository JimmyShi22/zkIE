use zkie_core::common::field::XorShift64;
use zkie_ops::compose::{prove_shard, verify_shard, Op, Store};

#[test]
fn compress_mask_and_sort_indices_roundtrip() {
    let mut store = Store::new();
    let mask = store.push(vec![]);
    let idx = store.push_idx(vec![3, 1, 2, 0, 2, 3, 1, 0]);
    let sorted = store.push_idx(vec![]);
    let ops = vec![
        Op::CompressMask { out: mask, m: 4, n: 4, ratio: 2 },
        Op::SortIndices { idx, out: sorted, m: 2, n: 4 },
    ];
    let mut rng = XorShift64::new(2);
    let proof = prove_shard(&mut store, &ops, &[], &mut rng);
    assert!(verify_shard(&[], &ops, &proof));
    assert_eq!(store.idx[sorted], vec![0, 1, 2, 3, 0, 1, 2, 3]);
}
