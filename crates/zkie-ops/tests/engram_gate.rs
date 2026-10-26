use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::from_i64;
use zkie_ops::compose::{prove_shard, verify_shard, Op, Store};

#[test]
fn engram_gate_roundtrip() {
    let m = 1usize;
    let hc = 4usize;
    let d = 2usize;
    let mut store = Store::new();
    let x = store.push((0..m * hc * d).map(|i| from_i64((i as i64 + 1) << 16)).collect::<Vec<_>>());
    let key = store.push((0..m * hc * d).map(|i| from_i64((i as i64 + 2) << 16)).collect::<Vec<_>>());
    let weight = store.push((0..hc * d).map(|_| from_i64(65536)).collect::<Vec<_>>());
    let value = store.push((0..m * d).map(|i| from_i64((i as i64 + 1) << 16)).collect::<Vec<_>>());
    let out = store.push(vec![]);
    let ops = vec![Op::EngramGate { x, key, weight, value, out, m, hc, d, eps: 1, clamp: 1 }];
    let mut rng = XorShift64::new(10);
    let proof = prove_shard(&mut store, &ops, &[], &mut rng);
    assert!(verify_shard(&[], &ops, &proof));
    assert_eq!(store.get(out).len(), m * hc * d);
}
