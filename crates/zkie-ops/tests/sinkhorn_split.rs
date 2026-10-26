use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::from_i64;
use zkie_ops::compose::{prove_shard, verify_shard, Op, Store};

#[test]
fn sinkhorn_split_roundtrip() {
    let m = 1usize;
    let hc = 4usize;
    let mix_hc = (2 + hc) * hc;
    let mut store = Store::new();
    let mixes = store.push((0..m * mix_hc).map(|i| from_i64((i as i64 + 1) * 100)).collect::<Vec<_>>());
    let scale = store.push((0..3).map(|i| from_i64(65536 / (i as i64 + 1))).collect::<Vec<_>>());
    let base = store.push((0..mix_hc).map(|i| from_i64(i as i64 * 10)).collect::<Vec<_>>());
    let pre = store.push(vec![]);
    let post = store.push(vec![]);
    let comb = store.push(vec![]);
    let ops = vec![Op::SinkhornSplit {
        mixes,
        scale,
        base,
        pre,
        post,
        comb,
        m,
        hc,
        iters: 20,
        eps: 1,
    }];
    let mut rng = XorShift64::new(3);
    let proof = prove_shard(&mut store, &ops, &[], &mut rng);
    assert!(verify_shard(&[], &ops, &proof));
    assert_eq!(store.get(pre).len(), m * hc);
    assert_eq!(store.get(post).len(), m * hc);
    assert_eq!(store.get(comb).len(), m * hc * hc);
}
