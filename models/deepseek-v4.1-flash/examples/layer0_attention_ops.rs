use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing};
use zkie_core::common::fixed_point::from_i64;
use zkie_models_deepseek_v41_flash::{build_deepseek_v41, open_shard, HEAD_DIM, N_HEADS};
use zkie_ops::compose::{causal_mask, prove_shard, verify_shard, Op, Store};

fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = std::fs::read(path).unwrap();
    bytes
        .chunks_exact(4)
        .map(|c| from_i64(i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64))
        .collect()
}

fn slice_head_part(q: &[Goldilocks], head: usize, m: usize, start: usize, len: usize) -> Vec<Goldilocks> {
    (0..m)
        .flat_map(|i| {
            let base = (i * N_HEADS + head) * HEAD_DIM + start;
            q[base..base + len].iter().copied()
        })
        .collect()
}

fn slice_kv_part(kv: &[Goldilocks], m: usize, start: usize, len: usize) -> Vec<Goldilocks> {
    (0..m)
        .flat_map(|i| kv[i * HEAD_DIM + start..i * HEAD_DIM + start + len].iter().copied())
        .collect()
}

fn gen_rope(m: usize, qk: usize) -> (Vec<Goldilocks>, Vec<Goldilocks>) {
    let half = qk / 2;
    let mut cos = Vec::with_capacity(m * half);
    let mut sin = Vec::with_capacity(m * half);
    for p in 0..m {
        for i in 0..half {
            let theta = p as f64 * (10000f64).powf(-2.0 * i as f64 / qk as f64);
            cos.push(from_i64((theta.cos() * 65536.0).round() as i64));
            sin.push(from_i64((theta.sin() * 65536.0).round() as i64));
        }
    }
    (cos, sin)
}

fn pad_vec(mut v: Vec<Goldilocks>, len: usize) -> Vec<Goldilocks> {
    v.resize(len, Goldilocks::ZERO);
    v
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
    let (q, kv) = build_deepseek_v41(&mut store, &mut ops, &dir, 0, m, 16);
    let file = open_shard(&path);
    let sink_raw = file.read_f32("layers.0.attn.attn_sink");

    let exp_table = store.push(load_i32("/data/zkIE/models/gpt2/weights/exp_table_i32.bin"));
    let mask = store.push(causal_mask(m));
    // Materialize q/kv before slicing per-head query tensors.
    let mut pre_rng = zkie_core::common::field::XorShift64::new(0xACE);
    let _pre = prove_shard(&mut store, &ops, &[], &mut pre_rng);

    let qk = 64usize;
    let nope = HEAD_DIM - qk;
    let nope_pad = nope.next_power_of_two();
    let (cos, sin) = gen_rope(m, qk);
    let cos_t = store.push(cos);
    let sin_t = store.push(sin);
    let limb_table = store.push((0..(1usize << 16)).map(|j| Goldilocks::from_u64(j as u64)).collect::<Vec<_>>());
    let kvn = store.push(pad_vec(slice_kv_part(store.get(kv), m, 0, nope), m * nope_pad));
    let kvr = store.push(slice_kv_part(store.get(kv), m, nope, qk));
    let kt_nope = store.push(vec![]);
    let kt_rope = store.push(vec![]);
    ops.push(Op::Transpose { x: kvn, out: kt_nope, m, k: nope_pad });
    ops.push(Op::Transpose { x: kvr, out: kt_rope, m, k: qk });

    let mut head_outs = Vec::new();
    for head in 0..N_HEADS {
        let qn = store.push(pad_vec(slice_head_part(store.get(q), head, m, 0, nope), m * nope_pad));
        let qr = store.push(slice_head_part(store.get(q), head, m, nope, qk));
        let qr_rot = store.push(vec![]);
        ops.push(Op::RoPE {
            x: qr,
            cos: cos_t,
            sin: sin_t,
            out: qr_rot,
            range_table: limb_table,
            m,
            d: qk,
            shift: 16,
        });
        let scores_n = store.push(vec![]);
        let scores_r = store.push(vec![]);
        let scores = store.push(vec![]);
        ops.push(Op::MatMul { a: qn, b: kt_nope, c: scores_n, m, k: nope_pad, n: m });
        ops.push(Op::MatMul { a: qr_rot, b: kt_rope, c: scores_r, m, k: qk, n: m });
        ops.push(Op::Add { a: scores_n, b: scores_r, c: scores });
        let idx = store.push_idx(vec![]);
        let e = store.push(vec![]);
        let probs = store.push(vec![]);
        let probs_bias = store.push(vec![]);
        let sink_fixed = {
            let v = zkie_core::common::fixed_point::to_i64(sink_raw[head]);
            ((v as f64 / 65536.0).exp() * 65536.0).round() as i64
        };
        let sink_bias = store.push(vec![from_i64(sink_fixed); m]);
        let attn_n = store.push(vec![]);
        let attn_r = store.push(vec![]);
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
            prec: 16,
        });
        ops.push(Op::SoftmaxBias {
            e,
            bias: sink_bias,
            out: probs_bias,
            m,
            n: m,
        });
        ops.push(Op::MatMul {
            a: probs_bias,
            b: kvn,
            c: attn_n,
            m,
            k: m,
            n: nope_pad,
        });
        let attn_n16 = store.push(vec![]);
        ops.push(Op::Scale { x: attn_n, out: attn_n16, factor: 1, shift: 16 });
        ops.push(Op::MatMul {
            a: probs_bias,
            b: kvr,
            c: attn_r,
            m,
            k: m,
            n: qk,
        });
        let attn_r16 = store.push(vec![]);
        ops.push(Op::Scale { x: attn_r, out: attn_r16, factor: 1, shift: 16 });
        head_outs.push(attn_n16);
    }

    let mut rng = zkie_core::common::field::XorShift64::new(0xACE);
    let proof = prove_shard(&mut store, &ops, &[], &mut rng);
    assert!(verify_shard(&store.owned_slices(), &ops, &proof));
    println!("ok heads={N_HEADS} m={m} ops={}", ops.len());
    if let Some(&h0) = head_outs.first() {
        let vals = store.get(h0);
        let mut out = Vec::new();
        for v in vals.iter().take(8) {
            out.push(zkiE_i64(*v));
        }
        println!("attn_head0_first8={out:?}");
    }
}

fn zkiE_i64(v: Goldilocks) -> i64 {
    use zkie_core::common::fixed_point::to_i64;
    to_i64(v)
}
