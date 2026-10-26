use zkie_core::common::field::{PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::from_i64;
use zkie_ops::compose::{prove_shard, verify_shard, Op, Store};

#[test]
fn sum_norm_roundtrip() {
    let mut store = Store::new();
    let x = store.push(vec![
        from_i64(1 << 16),
        from_i64(2 << 16),
        from_i64(3 << 16),
        from_i64(4 << 16),
    ]);
    let out = store.push(vec![]);
    let ops = vec![Op::SumNorm { x, out, m: 2, n: 2 }];
    let mut rng = XorShift64::new(0x51a);
    let proof = prove_shard(&mut store, &ops, &[], &mut rng);
    assert!(verify_shard(&[], &ops, &proof));

    // Row sums must each land within +-1 of 2^16 after rounding.
    let got = store.get(out);
    for row in 0..2 {
        let sum: i64 = (0..2)
            .map(|j| zkie_core::common::fixed_point::to_i64(got[row * 2 + j]))
            .sum();
        assert!(
            (sum - (1 << 16)).abs() <= 2,
            "row {row} sum {sum} not normalized to 2^16"
        );
    }
}
