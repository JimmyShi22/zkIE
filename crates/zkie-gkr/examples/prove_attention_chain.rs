//! Prove layer 0's attention block as one *chained* circuit at seq=1 (softmax
//! degenerates to 1, so `softmax @ V = V`): RMSNorm -> V matmul -> rescale+bias
//! -> o_proj matmul -> rescale+bias -> residual add. Every intermediate tensor
//! is committed and each stage's output commitment is the next stage's input
//! commitment, so the composition is sound.

use zkie_gkr::committed::{commit, prove_add, prove_affine, prove_matmul, prove_rms_norm};
use zkie_gkr::field::{Goldilocks, XorShift64};
use zkie_gkr::fixed_point::{from_i32, from_i64};
use zkie_gkr::whir::Whir;

fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = std::fs::read(path).expect("run extract_attention_chain.py first");
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let v = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.push(from_i32(v));
    }
    out
}

fn load_i64(path: &str) -> Vec<Goldilocks> {
    let bytes = std::fs::read(path).expect("run extract_attention_chain.py first");
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
    let chain = format!("{base}attention_chain/");
    let rsqrt_table = load_i32(&format!("{base}rsqrt_table_i32.bin"));
    const N_REAL: usize = 264;

    let add2 = load_i32(&format!("{chain}add2_i32.bin"));
    let ln_w = load_i32(&format!("{chain}ln_w_i32.bin"));
    let rms_raw = load_i64(&format!("{chain}rms_raw_i64.bin"));
    let mul9 = load_i32(&format!("{chain}mul9_i32.bin"));
    let v_w = load_i32(&format!("{chain}v_w_i32.bin"));
    let v_b = load_i32(&format!("{chain}v_b_i32.bin"));
    let v_raw = load_i64(&format!("{chain}v_raw_i64.bin"));
    let lin3v = load_i32(&format!("{chain}lin3v_i32.bin"));
    let op_w = load_i32(&format!("{chain}op_w_i32.bin"));
    let op_b = load_i32(&format!("{chain}op_b_i32.bin"));
    let op_raw = load_i64(&format!("{chain}op_raw_i64.bin"));
    let lin4 = load_i32(&format!("{chain}lin4_i32.bin"));
    let add5 = load_i32(&format!("{chain}add5_i32.bin"));

    let mut rng = XorShift64::new(0x7ee);
    let whir9 = Whir::new_testing(9);
    let whir18 = Whir::new_testing(18);

    // RMSNorm: rms_raw = add2 * rstd * ln_w, rstd bound to mean(add2^2) by lookup.
    let c_add2 = commit(&whir9, &add2);
    let c_rms_raw = commit(&whir9, &rms_raw);
    assert!(prove_rms_norm(
        &whir9, &c_add2, &add2, &whir9, &c_rms_raw, &rms_raw, &ln_w, N_REAL, &rsqrt_table, rng.field(), rng.field(), &mut rng,
    ));

    // rescale raw (2^48 -> 2^16); RMSNorm has no bias.
    let c_mul9 = commit(&whir9, &mul9);
    let zero_bias = vec![from_i32(0); 512];
    assert!(prove_affine(&whir9, &c_rms_raw, &rms_raw, &whir9, &c_mul9, &mul9, &zero_bias, 32, &mut rng));

    // V matmul: v_raw = mul9 @ v_w (2^32).
    let c_vw = commit(&whir18, &v_w);
    let c_v_raw = commit(&whir9, &v_raw);
    assert!(prove_matmul(&whir9, &c_mul9, &whir18, &c_vw, &whir9, &c_v_raw, &mul9, &v_w, &v_raw, 1, 512, 512, &mut rng, ));

    // rescale (2^32 -> 2^16) + V bias.
    let c_lin3v = commit(&whir9, &lin3v);
    assert!(prove_affine(&whir9, &c_v_raw, &v_raw, &whir9, &c_lin3v, &lin3v, &v_b, 16, &mut rng));

    // o_proj matmul: op_raw = lin3v @ op_w (2^32).
    let c_opw = commit(&whir18, &op_w);
    let c_op_raw = commit(&whir9, &op_raw);
    assert!(prove_matmul(&whir9, &c_lin3v, &whir18, &c_opw, &whir9, &c_op_raw, &lin3v, &op_w, &op_raw, 1, 512, 512, &mut rng, ));

    // rescale (2^32 -> 2^16) + o_proj bias.
    let c_lin4 = commit(&whir9, &lin4);
    assert!(prove_affine(&whir9, &c_op_raw, &op_raw, &whir9, &c_lin4, &lin4, &op_b, 16, &mut rng));

    // residual: add_5 = add_2 + linear_4.
    let c_add5 = commit(&whir9, &add5);
    assert!(prove_add(&whir9, &c_add2, &whir9, &c_lin4, &whir9, &c_add5, 512, &mut rng));

    println!("layer-0 attention chain (RMSNorm -> V -> o_proj -> residual) verified from WHIR commitments");
}
