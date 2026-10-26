use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_models_deepseek_v41_flash::{
    build_deepseek_v41, build_quantized_matmul, open_shard, push_quantized_fp8, DIM_PAD,
    HEAD_DIM, N_HEADS, O_GROUPS, O_LORA_RANK,
};
use zkie_ops::compose::{causal_mask, prove_shard, verify_shard, Op, Store};

fn load_i32(path: &str) -> Vec<Goldilocks> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| from_i64(i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64))
        .collect()
}

fn gen_rope(qk: usize) -> (Vec<Goldilocks>, Vec<Goldilocks>) {
    let half = qk / 2;
    let mut cos = Vec::with_capacity(half);
    let mut sin = Vec::with_capacity(half);
    for i in 0..half {
        let theta = (10000f64).powf(-2.0 * i as f64 / qk as f64);
        cos.push(from_i64((theta.cos() * 65536.0).round() as i64));
        sin.push(from_i64((theta.sin() * 65536.0).round() as i64));
    }
    (cos, sin)
}

fn slice_rows(mat: &[i64], cols: usize, r0: usize, r1: usize) -> Vec<Goldilocks> {
    (r0..r1)
        .flat_map(|r| mat[r * cols..(r + 1) * cols].iter().map(|&v| from_i64(v)))
        .collect()
}

fn slice_cols(mat: &[i64], cols: usize, c0: usize, c1: usize) -> Vec<Goldilocks> {
    let rows = mat.len() / cols;
    (0..rows)
        .flat_map(|r| mat[r * cols + c0..r * cols + c1].iter().map(|&v| from_i64(v)))
        .collect()
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/zkIE/models/deepseek-v4.1-flash/raw/model-00003-of-00048.safetensors".to_string());
    let dir = std::path::Path::new(&path).parent().unwrap().to_str().unwrap().to_string();
    let m = 1usize;
    let qk = 64usize;
    let nope = HEAD_DIM - qk;
    let nope_pad = nope.next_power_of_two();
    let mut store = Store::new();
    let mut ops: Vec<Op> = Vec::new();
    let (q, kv) = build_deepseek_v41(&mut store, &mut ops, &dir, m, 16);

    let file = open_shard(&path);
    let exp_table = store.push(load_i32("/data/zkIE/models/gpt2/weights/exp_table_i32.bin"));
    let mask = store.push(causal_mask(m));
    let limb_table = store.push((0..(1usize << 16)).map(|j| Goldilocks::from_u64(j as u64)).collect::<Vec<_>>());
    let (cos, sin) = gen_rope(qk);
    let cos_t = store.push(cos);
    let sin_t = store.push(sin);

    let kvn = store.push(vec![]);
    let kvr = store.push(vec![]);
    ops.push(Op::Slice { x: kv, out: kvn, start: 0, len: m * nope });
    ops.push(Op::Slice { x: kv, out: kvr, start: m * nope, len: m * qk });
    let zero_n = store.push(vec![Goldilocks::ZERO; m * (nope_pad - nope)]);
    let kvn_pad = store.push(vec![]);
    ops.push(Op::Concat { parts: vec![kvn, zero_n], out: kvn_pad });
    let ktn = store.push(vec![]);
    let ktr = store.push(vec![]);
    ops.push(Op::Transpose { x: kvn_pad, out: ktn, m, k: nope_pad });
    ops.push(Op::Transpose { x: kvr, out: ktr, m, k: qk });

    let mut head_outs = Vec::new();
    for head in 0..N_HEADS {
        let qn = store.push(vec![]);
        let qr = store.push(vec![]);
        ops.push(Op::Slice { x: q, out: qn, start: head * HEAD_DIM, len: nope });
        ops.push(Op::Slice { x: q, out: qr, start: head * HEAD_DIM + nope, len: qk });
        let qn_pad = store.push(vec![]);
        ops.push(Op::Concat { parts: vec![qn, zero_n], out: qn_pad });
        let qr_rot = store.push(vec![]);
        ops.push(Op::RoPE { x: qr, cos: cos_t, sin: sin_t, out: qr_rot, range_table: limb_table, m, d: qk, shift: 16 });
        let scores_n = store.push(vec![]);
        let scores_r = store.push(vec![]);
        let scores = store.push(vec![]);
        ops.push(Op::MatMul { a: qn_pad, b: ktn, c: scores_n, m, k: nope_pad, n: m });
        ops.push(Op::MatMul { a: qr_rot, b: ktr, c: scores_r, m, k: qk, n: m });
        ops.push(Op::Add { a: scores_n, b: scores_r, c: scores });
        let idx = store.push_idx(vec![]);
        let e = store.push(vec![]);
        let probs = store.push(vec![]);
        let probs_bias = store.push(vec![]);
        ops.push(Op::StableSoftmaxIndex { x: scores, mask, out: idx, offset: 1 << 21, table_len: 1 << 21, m, n: m });
        ops.push(Op::Softmax { idx, e, out: probs, table: exp_table, m, n: m });
        let sink_fixed = {
            let v = to_i64(file.read_f32("layers.0.attn.attn_sink")[head]);
            ((v as f64 / 65536.0).exp() * 65536.0).round() as i64
        };
        let sink_bias = store.push(vec![from_i64(sink_fixed); m]);
        ops.push(Op::SoftmaxBias { e, bias: sink_bias, out: probs_bias, m, n: m });
        let attn_n = store.push(vec![]);
        let attn_r = store.push(vec![]);
        ops.push(Op::MatMul { a: probs_bias, b: kvn_pad, c: attn_n, m, k: m, n: nope_pad });
        ops.push(Op::MatMul { a: probs_bias, b: kvr, c: attn_r, m, k: m, n: qk });
        let attn_n16 = store.push(vec![]);
        let attn_r16 = store.push(vec![]);
        ops.push(Op::Scale { x: attn_n, out: attn_n16, factor: 1, shift: 16 });
        ops.push(Op::Scale { x: attn_r, out: attn_r16, factor: 1, shift: 16 });
        let attn_n_sl = store.push(vec![]);
        ops.push(Op::Slice { x: attn_n16, out: attn_n_sl, start: 0, len: m * nope });
        let head_out = store.push(vec![]);
        ops.push(Op::Concat { parts: vec![attn_n_sl, attn_r16], out: head_out });
        head_outs.push(head_out);
    }

    // Concatenate all heads into o [m, N_HEADS*HEAD_DIM].
    let mut o = head_outs[0];
    for &h in &head_outs[1..] {
        let cat = store.push(vec![]);
        ops.push(Op::Concat { parts: vec![o, h], out: cat });
        o = cat;
    }

    let wa = push_quantized_fp8(&mut store, file.clone(), "layers.0.attn.wo_a.weight", "layers.0.attn.wo_a.scale");
    let wb = push_quantized_fp8(&mut store, file.clone(), "layers.0.attn.wo_b.weight", "layers.0.attn.wo_b.scale");
    let wa_mat: Vec<i64> = store.materialize(wa).iter().map(|&v| to_i64(v)).collect();
    let wb_mat: Vec<i64> = store.materialize(wb).iter().map(|&v| to_i64(v)).collect();

    let mut outs = Vec::new();
    for g in 0..O_GROUPS {
        let o_g = store.push(vec![]);
        ops.push(Op::Slice { x: o, out: o_g, start: g * (N_HEADS * HEAD_DIM / O_GROUPS), len: N_HEADS * HEAD_DIM / O_GROUPS });
        let wa_g = store.push(slice_rows(&wa_mat, N_HEADS * HEAD_DIM / O_GROUPS, g * O_LORA_RANK, (g + 1) * O_LORA_RANK));
        let oa_raw = build_quantized_matmul(&mut store, &mut ops, o_g, wa_g, m, N_HEADS * HEAD_DIM / O_GROUPS, O_LORA_RANK);
        let oa = store.push(vec![]);
        ops.push(Op::Scale { x: oa_raw, out: oa, factor: 1, shift: 16 });
        let wb_g = store.push(slice_cols(&wb_mat, O_GROUPS * O_LORA_RANK, g * O_LORA_RANK, (g + 1) * O_LORA_RANK));
        let out_raw = build_quantized_matmul(&mut store, &mut ops, oa, wb_g, m, O_LORA_RANK, DIM_PAD);
        let out = store.push(vec![]);
        ops.push(Op::Scale { x: out_raw, out, factor: 1, shift: 16 });
        outs.push(out);
    }
    let mut acc = outs[0];
    for &out in &outs[1..] {
        let s = store.push(vec![]);
        ops.push(Op::Add { a: acc, b: out, c: s });
        acc = s;
    }

    let mut rng = XorShift64::new(0xACE);
    let proof = prove_shard(&mut store, &ops, &[], &mut rng);
    assert!(verify_shard(&store.owned_slices(), &ops, &proof));
    println!("ok full_attention ops={} first4={:?}", ops.len(), store.get(acc).iter().take(4).map(|&v| to_i64(v)).collect::<Vec<_>>());
}
