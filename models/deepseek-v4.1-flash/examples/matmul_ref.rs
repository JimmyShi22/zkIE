//! Print the first block-quantized matmul outputs for an all-ones input, so the
//! result can be compared directly against `scripts/v41_ref_cpu.py`.

use zkie_core::common::field::XorShift64;
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_models_deepseek_v41_flash::{build_quantized_matmul, open_shard, push_quantized_fp8};
use zkie_ops::compose::{prove_shard, Op, Store};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/zkIE/models/deepseek-v4.1-flash/raw/model-00003-of-00048.safetensors".to_string());
    let file = open_shard(&path);
    let m = 1usize;
    let k = 4096usize;
    let n = 8192usize;

    let mut store = Store::new();
    let x = store.push(vec![from_i64(1); m * k]);
    let w = push_quantized_fp8(
        &mut store,
        file,
        "layers.0.attn.wo_a.weight",
        "layers.0.attn.wo_a.scale",
    );
    let mut ops: Vec<Op> = Vec::new();
    let c = build_quantized_matmul(&mut store, &mut ops, x, w, m, k, n);
    let mut rng = XorShift64::new(0x1234);
    let _proof = prove_shard(&mut store, &ops, &[], &mut rng);
    let vals = store.get(c);
    for (i, v) in vals.iter().take(8).enumerate() {
        println!("[{i}] {}", to_i64(*v));
    }
}
