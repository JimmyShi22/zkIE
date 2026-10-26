use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_ops::compose::{prove_shard, verify_shard, Op, Store};

#[test]
fn slice_concat_roundtrip() {
    let mut store = Store::new();
    let x = store.push((0..8).map(|i| Goldilocks::from_u64(i)).collect::<Vec<_>>());
    let a = store.push(vec![]);
    let b = store.push(vec![]);
    let c = store.push(vec![]);
    let ops = vec![
        Op::Slice { x, out: a, start: 0, len: 4 },
        Op::Slice { x, out: b, start: 4, len: 4 },
        Op::Concat { parts: vec![a, b], out: c },
    ];
    let mut rng = XorShift64::new(1);
    let proof = prove_shard(&mut store, &ops, &[], &mut rng);
    assert!(verify_shard(&[], &ops, &proof));
    assert_eq!(store.get(c), store.get(x));
}
