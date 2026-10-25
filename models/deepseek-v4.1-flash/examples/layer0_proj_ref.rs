//! Print the layer-0 MLA projection intermediates for an all-ones input, for
//! comparison against `scripts/v41_ref_cpu.py --mode attention`.

use zkie_core::common::field::XorShift64;
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_models_deepseek_v41_flash::{
    build_quantized_matmul, open_shard, push_quantized_fp8, DIM_PAD, HEAD_DIM, N_HEADS,
    Q_LORA_RANK_PAD,
};
use zkie_ops::compose::{prove_shard, Op, Store};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/zkIE/models/deepseek-v4.1-flash/raw/model-00003-of-00048.safetensors".to_string());
    let file = open_shard(&path);
    let m = 16usize;

    let mut store = Store::new();
    let x = store.push(vec![from_i64(1 << 16); m * DIM_PAD]);
    let wq_a = push_quantized_fp8(
        &mut store,
        file.clone(),
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.wq_a.scale",
    );
    let wq_b = push_quantized_fp8(
        &mut store,
        file.clone(),
        "layers.0.attn.wq_b.weight",
        "layers.0.attn.wq_b.scale",
    );
    let wkv = push_quantized_fp8(
        &mut store,
        file.clone(),
        "layers.0.attn.wkv.weight",
        "layers.0.attn.wkv.scale",
    );

    let mut ops: Vec<Op> = Vec::new();
    let qr_raw = build_quantized_matmul(&mut store, &mut ops, x, wq_a, m, DIM_PAD, Q_LORA_RANK_PAD);
    let qr = store.push(vec![]);
    ops.push(Op::Scale {
        x: qr_raw,
        out: qr,
        factor: 1,
        shift: 16,
    });
    let q_raw = build_quantized_matmul(
        &mut store,
        &mut ops,
        qr,
        wq_b,
        m,
        Q_LORA_RANK_PAD,
        N_HEADS * HEAD_DIM,
    );
    let q = store.push(vec![]);
    ops.push(Op::Scale {
        x: q_raw,
        out: q,
        factor: 1,
        shift: 16,
    });

    let kv_raw = build_quantized_matmul(&mut store, &mut ops, x, wkv, m, DIM_PAD, HEAD_DIM);
    let kv = store.push(vec![]);
    ops.push(Op::Scale {
        x: kv_raw,
        out: kv,
        factor: 1,
        shift: 16,
    });
    let mut rng = XorShift64::new(0x1234);
    let _proof = prove_shard(&mut store, &ops, &[], &mut rng);
    println!("qr[0..4]: {:?}", store.get(qr).iter().take(4).map(|&v| to_i64(v)).collect::<Vec<_>>());
    println!("q[0..4]: {:?}", store.get(q).iter().take(4).map(|&v| to_i64(v)).collect::<Vec<_>>());
    println!("kv[0..4]: {:?}", store.get(kv).iter().take(4).map(|&v| to_i64(v)).collect::<Vec<_>>());
}
