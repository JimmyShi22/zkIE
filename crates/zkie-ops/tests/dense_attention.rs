use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::from_i64;
use zkie_ops::compose::{prove_shard, verify_shard, Op, Store};

#[test]
fn shared_kv_dense_attention_chain_roundtrip() {
    let m = 2usize;
    let d = 2usize;
    let table_len = 8usize;

    let mut store = Store::new();
    let q = store.push(vec![from_i64(1), from_i64(2), from_i64(3), from_i64(4)]);
    let kv = store.push(vec![from_i64(5), from_i64(6), from_i64(7), from_i64(8)]);
    let kt = store.push(vec![]);
    let scores = store.push(vec![]);
    let idx = store.push_idx(vec![]);
    let e = store.push(vec![]);
    let probs = store.push(vec![]);
    let out = store.push(vec![]);
    // Small exp table: index 0..7 maps to 1,2,4,8,16,32,64,128.
    let exp_table = store.push(
        (0..table_len)
            .map(|i| from_i64(1i64 << i))
            .collect::<Vec<_>>(),
    );

    let ops = vec![
        Op::Transpose {
            x: kv,
            out: kt,
            m,
            k: d,
        },
        Op::MatMul {
            a: q,
            b: kt,
            c: scores,
            m,
            k: d,
            n: m,
        },
        Op::SoftmaxIndex {
            x: scores,
            out: idx,
            table_len,
        },
        Op::Softmax {
            idx,
            e,
            out: probs,
            table: exp_table,
            m,
            n: m,
        },
        Op::MatMul {
            a: probs,
            b: kv,
            c: out,
            m,
            k: m,
            n: d,
        },
    ];

    let mut rng = XorShift64::new(0x5151);
    let proof = prove_shard(&mut store, &ops, &[], &mut rng);
    assert!(verify_shard(&store.owned_slices(), &ops, &proof));
    assert_eq!(store.get(out).len(), m * d);
}
