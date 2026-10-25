//! Prove and verify a single layer-0 block-quantized matmul against the real
//! checkpoint. This is the first vertical slice: one native quantized matmul,
//! end-to-end through `prove_shard`/`verify_shard`.

use zkie_core::common::field::{Goldilocks, XorShift64};
use zkie_core::common::fixed_point::from_i64;
use zkie_models_deepseek_v41_flash::{build_quantized_matmul, open_shard, push_quantized_fp8};
use zkie_ops::compose::{prove_shard, verify_shard, Op, Store};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/zkIE/models/deepseek-v4.1-flash/raw/model-00003-of-00048.safetensors".to_string());
    let file = open_shard(&path);

    // `wo_a` is a power-of-two FP8 weight: [8192, 4096]. The field-level
    // matmul prover requires power-of-two dims, so it is the natural first
    // smoke test from the real checkpoint.
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
    let _ = c;

    let mut rng = XorShift64::new(0x0F41A5);
    let proof = prove_shard(&mut store, &ops, &[], &mut rng);
    assert!(verify_shard(&[], &ops, &proof));
    println!(
        "ok: layer0 wo_a block-quantized matmul proven and verified (m={m}, k={k}, n={n})"
    );
}
