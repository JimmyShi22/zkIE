use zkie_core::common::field::{Goldilocks, XorShift64};
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_models_deepseek_v41_flash::{
    build_quantized_matmul, open_shard, push_quantized_fp8, DIM_PAD, O_GROUPS, O_LORA_RANK,
};
use zkie_ops::compose::{prove_shard, verify_shard, Op, Store};

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
    let file = open_shard(&path);
    let m = 1usize;
    let mut store = Store::new();
    let mut ops: Vec<Op> = Vec::new();

    let wa = push_quantized_fp8(
        &mut store,
        file.clone(),
        "layers.0.attn.wo_a.weight",
        "layers.0.attn.wo_a.scale",
    );
    let wb = push_quantized_fp8(
        &mut store,
        file.clone(),
        "layers.0.attn.wo_b.weight",
        "layers.0.attn.wo_b.scale",
    );
    let wa_mat: Vec<i64> = store.materialize(wa).iter().map(|&v| to_i64(v)).collect();
    let wb_mat: Vec<i64> = store.materialize(wb).iter().map(|&v| to_i64(v)).collect();

    let mut outs = Vec::new();
    for g in 0..O_GROUPS {
        let o_g = store.push(vec![from_i64(1 << 16); m * 4096]);
        let wa_g = store.push(slice_rows(&wa_mat, 4096, g * O_LORA_RANK, (g + 1) * O_LORA_RANK));
        let oa_raw = build_quantized_matmul(&mut store, &mut ops, o_g, wa_g, m, 4096, O_LORA_RANK);
        let oa = store.push(vec![]);
        ops.push(Op::Scale { x: oa_raw, out: oa, factor: 1, shift: 16 });

        let wb_g = store.push(slice_cols(&wb_mat, O_GROUPS * O_LORA_RANK, g * O_LORA_RANK, (g + 1) * O_LORA_RANK));
        let out_raw = build_quantized_matmul(&mut store, &mut ops, oa, wb_g, m, O_LORA_RANK, DIM_PAD);
        let out = store.push(vec![]);
        ops.push(Op::Scale { x: out_raw, out, factor: 1, shift: 16 });
        outs.push(out);
    }

    let mut acc = outs[0];
    for &o in &outs[1..] {
        let s = store.push(vec![]);
        ops.push(Op::Add { a: acc, b: o, c: s });
        acc = s;
    }

    let mut rng = XorShift64::new(0xABCD);
    let proof = prove_shard(&mut store, &ops, &[], &mut rng);
    assert!(verify_shard(&store.owned_slices(), &ops, &proof));
    println!("ok groups={O_GROUPS} ops={} first4={:?}", ops.len(), store.get(acc).iter().take(4).map(|&v| to_i64(v)).collect::<Vec<_>>());
}
