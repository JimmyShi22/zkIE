use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_models_deepseek_v41_flash::{
    build_quantized_matmul, open_shard, push_quantized_fp4, DIM, DIM_PAD, MOE_INTER_DIM,
    MOE_INTER_DIM_PAD,
};
use zkie_ops::compose::{prove_shard, verify_shard, Op, Store};

fn load_i32(path: &str) -> Vec<Goldilocks> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| from_i64(i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64))
        .collect()
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/data/zkIE/models/deepseek-v4.1-flash/raw/model-00003-of-00048.safetensors".to_string()
    });
    let eid = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(240usize);
    let m = 1usize;
    let shift = 16u32;
    let file = open_shard(&path);

    let mut store = Store::new();
    let silu = store.push(load_i32("/data/zkIE/models/silu_clipped_table_i32.bin"));
    let clip = store.push(load_i32("/data/zkIE/models/clip_m10_p10_table_i32.bin"));
    let limb_table = store.push(
        (0..(1usize << 16))
            .map(|j| Goldilocks::from_u64(j as u64))
            .collect::<Vec<_>>(),
    );
    let x = store.push(vec![from_i64(1 << 16); m * DIM_PAD]);
    let mut ops: Vec<Op> = Vec::new();

    let base = format!("layers.0.ffn.experts.{eid}");
    let w1 = push_quantized_fp4(
        &mut store,
        file.clone(),
        &format!("{base}.w1.weight"),
        &format!("{base}.w1.scale"),
        DIM,
    );
    let w2 = push_quantized_fp4(
        &mut store,
        file.clone(),
        &format!("{base}.w2.weight"),
        &format!("{base}.w2.scale"),
        MOE_INTER_DIM,
    );
    let w3 = push_quantized_fp4(
        &mut store,
        file.clone(),
        &format!("{base}.w3.weight"),
        &format!("{base}.w3.scale"),
        DIM,
    );

    let gate_raw = build_quantized_matmul(&mut store, &mut ops, x, w1, m, DIM_PAD, MOE_INTER_DIM_PAD);
    let gate = store.push(vec![]);
    ops.push(Op::Scale { x: gate_raw, out: gate, factor: 1, shift });
    let gate_idx = store.push_idx(vec![]);
    ops.push(Op::GeluIndex { x: gate, out: gate_idx, offset: 1 << 23, table_len: 1 << 24, limb_table });
    let gate_act = store.push(vec![]);
    ops.push(Op::Lookup { idx: gate_idx, out: gate_act, table: silu });

    let up_raw = build_quantized_matmul(&mut store, &mut ops, x, w3, m, DIM_PAD, MOE_INTER_DIM_PAD);
    let up = store.push(vec![]);
    ops.push(Op::Scale { x: up_raw, out: up, factor: 1, shift });
    let up_idx = store.push_idx(vec![]);
    ops.push(Op::GeluIndex { x: up, out: up_idx, offset: 1 << 23, table_len: 1 << 24, limb_table });
    let up_clip = store.push(vec![]);
    ops.push(Op::Lookup { idx: up_idx, out: up_clip, table: clip });

    let hidden = store.push(vec![]);
    ops.push(Op::ScaleVec { x: gate_act, scale: up_clip, out: hidden, shift: 16 });
    let out_raw = build_quantized_matmul(&mut store, &mut ops, hidden, w2, m, MOE_INTER_DIM_PAD, DIM_PAD);
    let out = store.push(vec![]);
    ops.push(Op::Scale { x: out_raw, out, factor: 1, shift });

    let mut rng = XorShift64::new(0xE1D1);
    let proof = prove_shard(&mut store, &ops, &[], &mut rng);
    assert!(verify_shard(&[], &ops, &proof));
    println!(
        "ok routed_expert_{eid} first4={:?}",
        store.get(out).iter().take(4).map(|&v| to_i64(v)).collect::<Vec<_>>()
    );
}
