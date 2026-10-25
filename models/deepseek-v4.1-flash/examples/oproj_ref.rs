//! Direct fixed-point computation of the layer-0 `wo_a`/`wo_b` output
//! projection for an all-ones attention input, compared against
//! `scripts/v41_ref_cpu.py --mode oproj`.

use zkie_core::common::fixed_point::to_i64;
use zkie_models_deepseek_v41_flash::{open_shard, push_quantized_fp8};
use zkie_ops::compose::Store;

fn round_div(a: i64, b: i64) -> i64 {
    let q = a.div_euclid(b);
    let r = a.rem_euclid(b);
    if r * 2 >= b {
        q + 1
    } else {
        q
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/zkIE/models/deepseek-v4.1-flash/raw/model-00003-of-00048.safetensors".to_string());
    let file = open_shard(&path);
    let mut store = Store::new();
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
    let wa: Vec<i64> = store.materialize(wa).iter().map(|&v| to_i64(v)).collect();
    let wb: Vec<i64> = store.materialize(wb).iter().map(|&v| to_i64(v)).collect();

    let groups = 8usize;
    let o_lora = 1024usize;
    let d = 4096usize;
    let hidden_pad = 8192usize;

    // Attention input all ones at fixed scale 2^16. After the first projection
    // + shift, each wo_a output is exactly the integer sum of the corresponding
    // quantized weight row.
    let mut oa = vec![0i64; groups * o_lora];
    for g in 0..groups {
        for r in 0..o_lora {
            let mut s = 0i64;
            for j in 0..d {
                s += wa[(g * o_lora + r) * d + j];
            }
            oa[g * o_lora + r] = s;
        }
    }

    // wo_b is stored padded [8192, 8192]; the first 5120 output rows are real.
    for out in 0..4 {
        let mut s = 0i64;
        for k in 0..hidden_pad {
            s += oa[k] * wb[out * hidden_pad + k];
        }
        println!("[{out}] {}", round_div(s, 1 << 16));
    }
}
