//! Prove layer 0's FFN tail as one *chained* circuit: LayerNorm -> gate matmul
//! -> rescale+bias -> ReLU -> down matmul -> rescale+bias -> residual add. Every
//! intermediate tensor is committed, and each stage's output commitment is the
//! next stage's input commitment, so the composition is sound (no prover-chosen
//! input can be swapped in between stages).

use zkie_gkr::committed::{commit, prove_add, prove_affine, prove_layer_norm, prove_matmul, prove_relu};
use zkie_gkr::field::{Goldilocks, XorShift64};
use zkie_gkr::fixed_point::{from_i32, from_i64};
use zkie_gkr::whir::Whir;

fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = std::fs::read(path).expect("run extract_ffn_chain.py first");
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let v = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.push(from_i32(v));
    }
    out
}

fn load_i64(path: &str) -> Vec<Goldilocks> {
    let bytes = std::fs::read(path).expect("run extract_ffn_chain.py first");
    let mut out = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        let v = i64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        out.push(from_i64(v));
    }
    out
}

fn main() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/");
    let chain = format!("{base}ffn_chain/");
    let rsqrt_table = load_i32(&format!("{base}rsqrt_table_i32.bin"));
    const N_REAL: usize = 264;

    let x = load_i32(&format!("{base}norms/L0_in.bin"));
    let nw = load_i32(&format!("{base}norms/L0_w.bin"));
    let nb = load_i32(&format!("{base}norms/L0_b.bin"));
    let norm_raw = load_i64(&format!("{chain}norm_raw_i64.bin"));
    let norm_out = load_i32(&format!("{chain}norm_out_i32.bin"));
    let gate = load_i32(&format!("{base}weights/val_126_512x1024_i32.bin"));
    let gate_bias = load_i32(&format!("{chain}gate_bias_i32.bin"));
    let gate_raw = load_i64(&format!("{chain}gate_raw_i64.bin"));
    let relu = load_i32(&format!("{chain}relu_i32.bin"));
    let down = load_i32(&format!("{base}weights/val_128_1024x512_i32.bin"));
    let down_bias = load_i32(&format!("{chain}down_bias_i32.bin"));
    let down_raw = load_i64(&format!("{chain}down_raw_i64.bin"));
    let ffn_out = load_i32(&format!("{chain}ffn_out_i32.bin"));
    let residual = load_i32(&format!("{chain}residual_i32.bin"));

    let mut rng = XorShift64::new(0x7ee);
    let whir9 = Whir::new_testing(9);
    let whir10 = Whir::new_testing(10);
    let whir19 = Whir::new_testing(19);

    // LayerNorm: raw = (x - mean) * rstd * w, mean/var recomputed from x.
    let cx = commit(&whir9, &x);
    let c_norm_raw = commit(&whir9, &norm_raw);
    assert!(prove_layer_norm(
        &whir9, &cx, &x, &whir9, &c_norm_raw, &norm_raw, &nw, N_REAL, &rsqrt_table, rng.field(), rng.field(), &mut rng,
    ));

    // rescale raw (2^48 -> 2^16) and add the LayerNorm bias.
    let c_norm_out = commit(&whir9, &norm_out);
    assert!(prove_affine(&whir9, &c_norm_raw, &norm_raw, &whir9, &c_norm_out, &norm_out, &nb, 32, &mut rng));

    // gate: norm_out @ gate = gate_raw (2^32).
    let c_gate = commit(&whir19, &gate);
    let c_gate_raw = commit(&whir10, &gate_raw);
    assert!(prove_matmul(&whir9, &c_norm_out, &whir19, &c_gate, &whir10, &c_gate_raw, &norm_out, &gate, &gate_raw, 1, 512, 1024, &mut rng, ));

    // rescale (2^32 -> 2^16) + gate bias + ReLU.
    let c_relu = commit(&whir10, &relu);
    assert!(prove_relu(&whir10, &c_gate_raw, &gate_raw, &whir10, &c_relu, &relu, &gate_bias, &mut rng));

    // down: relu @ down = down_raw (2^32).
    let c_down = commit(&whir19, &down);
    let c_down_raw = commit(&whir9, &down_raw);
    assert!(prove_matmul(&whir10, &c_relu, &whir19, &c_down, &whir9, &c_down_raw, &relu, &down, &down_raw, 1, 1024, 512, &mut rng, ));

    // rescale (2^32 -> 2^16) + down bias.
    let c_ffn_out = commit(&whir9, &ffn_out);
    assert!(prove_affine(&whir9, &c_down_raw, &down_raw, &whir9, &c_ffn_out, &ffn_out, &down_bias, 16, &mut rng));

    // residual: add_6 = ffn_out + add_5.
    let c_residual = commit(&whir9, &residual);
    assert!(prove_add(&whir9, &c_ffn_out, &whir9, &cx, &whir9, &c_residual, 512, &mut rng));

    println!("layer-0 FFN chain (LayerNorm -> gate -> bias -> ReLU -> down -> bias -> residual) verified from WHIR commitments");
}
