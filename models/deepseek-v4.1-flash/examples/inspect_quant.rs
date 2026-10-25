//! Inspect the dequantized values of a layer-0 FP8 weight. This is a quick
//! codec smoke test against the real checkpoint, useful before running proofs.

use zkie_core::common::fixed_point::to_i64;
use zkie_models_deepseek_v41_flash::{open_shard, push_quantized_fp8};
use zkie_ops::compose::Store;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/zkIE/models/deepseek-v4.1-flash/raw/model-00003-of-00048.safetensors".to_string());
    let file = open_shard(&path);
    let mut store = Store::new();
    let w = push_quantized_fp8(
        &mut store,
        file,
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.wq_a.scale",
    );
    let vals = store.materialize(w);
    println!("layers.0.attn.wq_a dequantized len: {}", vals.len());
    for (i, v) in vals.iter().take(8).enumerate() {
        println!("[{i}] {v:?} ({})", to_i64(*v));
    }
}
