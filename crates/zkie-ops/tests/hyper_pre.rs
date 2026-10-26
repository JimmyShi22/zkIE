use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::from_i64;
use zkie_ops::compose::{prove_shard, verify_shard, Op, Store};

#[test]
fn hyper_pre_roundtrip() {
    let m = 1usize;
    let hc = 4usize;
    let d = 2usize;
    let mut store = Store::new();
    let x = store.push((0..m * hc * d).map(|i| from_i64((i as i64 + 1) << 16)).collect::<Vec<_>>());
    let pre = store.push((0..m * hc).map(|i| from_i64(16384)).collect::<Vec<_>>());
    let out = store.push(vec![]);
    let ops = vec![Op::HyperPre { x, pre, out, m, hc, d }];
    let mut rng = XorShift64::new(4);
    let proof = prove_shard(&mut store, &ops, &[], &mut rng);
    assert!(verify_shard(&[], &ops, &proof));
    assert_eq!(store.get(out).len(), m * d);
}
