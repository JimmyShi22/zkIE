//! Prove TimesFM layer 0's full matmul chain from WHIR commitments.
//!
//! Reads the real weights (scale 2^16) and activations (scale 2^16), computes
//! each matmul's raw output, commits every tensor once, and verifies the four
//! weight matmuls (QKV, o_proj, gate, down) from prescribed-point openings.
//! The attention QK^T/PV is degenerate at seq=1 (softmax of one value is 1), so
//! it is folded into the o_proj input (the V slice of the QKV output).

use zkie_gkr::committed::{commit, prove_matmul};
use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_gkr::fixed_point::from_i32;
use zkie_gkr::whir::Whir;

fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = std::fs::read(path).expect("run the extraction scripts first");
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let v = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.push(from_i32(v));
    }
    out
}

fn dense(a: &[Goldilocks], b: &[Goldilocks], k: usize, n: usize) -> Vec<Goldilocks> {
    let mut c = vec![Goldilocks::ZERO; n];
    for j in 0..n {
        let mut acc = Goldilocks::ZERO;
        for w in 0..k {
            acc = acc + a[w] * b[w * n + j];
        }
        c[j] = acc;
    }
    c
}

fn main() {
    let w = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/weights/");
    let a = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/activations/");

    // Weights (padded int32, scale 2^16).
    let qkv = load_i32(&format!("{w}val_94_512x1024_i32.bin"));
    let o_proj = load_i32(&format!("{w}val_122_512x512_i32.bin"));
    let gate = load_i32(&format!("{w}val_126_512x1024_i32.bin"));
    let down = load_i32(&format!("{w}val_128_1024x512_i32.bin"));

    // Activations (padded int32, scale 2^16).
    let input = load_i32(&format!("{a}mul_9_512_i32.bin"));
    let o_in = load_i32(&format!("{a}view_5_512_i32.bin")); // attention output (o_proj input)
    let ffn_in = load_i32(&format!("{a}layer_norm_512_i32.bin"));
    let relu = load_i32(&format!("{a}relu_1024_i32.bin"));

    let mut rng = XorShift64::new(0x7ee);
    let whir9 = Whir::new_testing(9);
    let whir10 = Whir::new_testing(10);
    let whir18 = Whir::new_testing(18);
    let whir19 = Whir::new_testing(19);

    // QKV: input [512] @ qkv [512,1024] = qkv_out_raw [1024].
    let qkv_raw = dense(&input, &qkv, 512, 1024);
    let c_in = commit(&whir9, &input);
    let c_qkv = commit(&whir19, &qkv);
    let c_qkv_raw = commit(&whir10, &qkv_raw);
    assert!(prove_matmul(&whir9, &c_in, &whir19, &c_qkv, &whir10, &c_qkv_raw, &input, &qkv, &qkv_raw, 1, 512, 1024, &mut rng, ));

    // gate: ffn_in [512] @ gate [512,1024] = gate_raw [1024].
    let gate_raw = dense(&ffn_in, &gate, 512, 1024);
    let c_ffn_in = commit(&whir9, &ffn_in);
    let c_gate = commit(&whir19, &gate);
    let c_gate_raw = commit(&whir10, &gate_raw);
    assert!(prove_matmul(&whir9, &c_ffn_in, &whir19, &c_gate, &whir10, &c_gate_raw, &ffn_in, &gate, &gate_raw, 1, 512, 1024, &mut rng, ));

    // down: relu [1024] @ down [1024,512] = down_raw [512].
    let down_raw = dense(&relu, &down, 1024, 512);
    let c_relu = commit(&whir10, &relu);
    let c_down = commit(&whir19, &down);
    let c_down_raw = commit(&whir9, &down_raw);
    assert!(prove_matmul(&whir10, &c_relu, &whir19, &c_down, &whir9, &c_down_raw, &relu, &down, &down_raw, 1, 1024, 512, &mut rng, ));

    // o_proj: attention output [512] @ o_proj [512,512] = o_raw [512].
    let o_raw = dense(&o_in, &o_proj, 512, 512);
    let c_o_in = commit(&whir9, &o_in);
    let c_o_proj = commit(&whir18, &o_proj);
    let c_o_raw = commit(&whir9, &o_raw);
    assert!(prove_matmul(&whir9, &c_o_in, &whir18, &c_o_proj, &whir9, &c_o_raw, &o_in, &o_proj, &o_raw, 1, 512, 512, &mut rng, ));

    println!("TimesFM layer 0: QKV + o_proj + gate + down matmuls verified from WHIR commitments");
}
