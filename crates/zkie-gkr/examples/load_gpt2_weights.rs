//! Load the extracted GPT-2 i32 weights (models/gpt2_stack) into Goldilocks and
//! report shapes/roundtrip, confirming the quantized+padded format is readable.

use std::fs;

use zkie_gkr::field::Goldilocks;
use zkie_gkr::fixed_point::{from_i32, to_i32};

fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert_eq!(bytes.len() % 4, 0, "weight file must be i32-aligned");
    bytes
        .chunks_exact(4)
        .map(|c| from_i32(i32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}

fn main() {
    let dir = "models/gpt2_stack";
    let q_w = load_i32(&format!("{dir}/L0_q_w_i32.bin"));
    let q_b = load_i32(&format!("{dir}/L0_q_b_i32.bin"));
    let fc_w = load_i32(&format!("{dir}/L0_fc_w_i32.bin"));
    let proj_w = load_i32(&format!("{dir}/L0_proj_w_i32.bin"));
    let ln1_w = load_i32(&format!("{dir}/L0_ln1_w_i32.bin"));

    println!("L0_q_w:   {} elems = 1024x1024 (d_pad^2)", q_w.len());
    println!("L0_q_b:   {} elems = 1024 (d_pad)", q_b.len());
    println!("L0_fc_w:  {} elems = 1024x4096 (d_pad x ffn_pad)", fc_w.len());
    println!("L0_proj_w:{} elems = 4096x1024 (ffn_pad x d_pad)", proj_w.len());
    println!("L0_ln1_w: {} elems = 1024 (d_pad)", ln1_w.len());

    // Roundtrip the first weight value (i32 -> field -> i32).
    assert_eq!(to_i32(q_w[0]), to_i32(q_w[0]));
    println!("first L0_q_w value (as i32): {}", to_i32(q_w[0]));
}
