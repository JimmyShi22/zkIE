use zkie_core::common::field::Goldilocks;
use zkie_core::common::fixed_point::from_i64;
use zkie_models_deepseek_v41_flash::{build_deepseek_v41, HEAD_DIM, N_HEADS};
use zkie_ops::compose::{causal_mask, prove_shard, verify_shard, Op, Store};

fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = std::fs::read(path).unwrap();
    bytes
        .chunks_exact(4)
        .map(|c| from_i64(i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64))
        .collect()
}

fn slice_head(q: &[Goldilocks], head: usize, m: usize) -> Vec<Goldilocks> {
    let d = HEAD_DIM;
    (0..m)
        .flat_map(|i| q[(i * N_HEADS + head) * d..(i * N_HEADS + head + 1) * d].iter().copied())
        .collect()
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/zkIE/models/deepseek-v4.1-flash/raw/model-00003-of-00048.safetensors".to_string());
    let dir = std::path::Path::new(&path)
        .parent()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    let m = 16usize;
    let d = HEAD_DIM;
    let mut store = Store::new();
    let mut ops: Vec<Op> = Vec::new();
    let (q, kv) = build_deepseek_v41(&mut store, &mut ops, &dir, m, 16);

    let exp_table = store.push(load_i32("/data/zkIE/models/gpt2/weights/exp_table_i32.bin"));
    let mask = store.push(causal_mask(m));
    let kt = store.push(vec![]);
    ops.push(Op::Transpose {
        x: kv,
        out: kt,
        m,
        k: d,
    });

    // Materialize q/kv before slicing per-head query tensors.
    let mut pre_rng = zkie_core::common::field::XorShift64::new(0xACE);
    let _pre = prove_shard(&mut store, &ops, &[], &mut pre_rng);

    let mut head_outs = Vec::new();
    for head in 0..N_HEADS {
        let qh = store.push(slice_head(store.get(q), head, m));
        let scores = store.push(vec![]);
        let idx = store.push_idx(vec![]);
        let e = store.push(vec![]);
        let probs = store.push(vec![]);
        let attn = store.push(vec![]);
        ops.push(Op::MatMul {
            a: qh,
            b: kt,
            c: scores,
            m,
            k: d,
            n: m,
        });
        ops.push(Op::StableSoftmaxIndex {
            x: scores,
            mask,
            out: idx,
            offset: 1 << 21,
            table_len: 1 << 21,
            m,
            n: m,
        });
        ops.push(Op::Softmax {
            idx,
            e,
            out: probs,
            table: exp_table,
            m,
            n: m,
        });
        ops.push(Op::MatMul {
            a: probs,
            b: kv,
            c: attn,
            m,
            k: m,
            n: d,
        });
        head_outs.push(attn);
    }

    let mut rng = zkie_core::common::field::XorShift64::new(0xACE);
    let proof = prove_shard(&mut store, &ops, &[], &mut rng);
    assert!(verify_shard(&store.owned_slices(), &ops, &proof));
    println!("ok heads={N_HEADS} m={m} ops={}", ops.len());
}
