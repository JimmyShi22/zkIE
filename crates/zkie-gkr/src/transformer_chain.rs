//! The full transformer layer as ONE claim chain (single head, no layernorm yet):
//! attention (x -> y = x + o) then FFN (y -> x_new = y + proj). The intermediate
//! residual stream `y` is claimed by both halves at different points and bound via
//! `same_poly`, so the composed proof is sound iff both halves are sound AND y is
//! consistent. This is the "one g per layer" shape (layernorm still to be wired).

use crate::attention_chain::{
    prove_attention_chain, verify_attention_chain, AttentionChainProof,
};
use crate::ffn_chain::{prove_ffn_chain, verify_ffn_chain, FfnChainProof};
use crate::layernorm_chain::{
    prove_layernorm_chain, verify_layernorm_chain, LayernormChainProof,
};
use crate::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::fixed_point::{from_i64, to_i64};
use crate::mle;
use crate::same_poly::{prove_same_poly, verify_same_poly, SamePolyProof};

pub struct TransformerChainProof {
    pub attention: AttentionChainProof,
    pub ffn: FfnChainProof,
    pub same_y: SamePolyProof,
}

/// Full post-norm transformer layer: h = layernorm(x) -> attention+FFN on h.
/// The layernorm output `h` is claimed by the layernorm (out) and by the
/// attention (its merged same_x point), bound via `same_poly`.
pub struct TransformerLayerProof {
    pub layernorm: LayernormChainProof,
    pub block: TransformerChainProof,
    pub same_h: SamePolyProof,
}

fn mm(a: &[Goldilocks], b: &[Goldilocks], m: usize, k: usize, n: usize) -> Vec<Goldilocks> {
    let mut c = vec![Goldilocks::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = Goldilocks::ZERO;
            for kk in 0..k {
                acc = acc + a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn transpose(a: &[Goldilocks], m: usize, k: usize) -> Vec<Goldilocks> {
    let mut t = vec![Goldilocks::ZERO; k * m];
    for i in 0..m {
        for kk in 0..k {
            t[kk * m + i] = a[i * k + kk];
        }
    }
    t
}

fn round_div(a: i64, b: i64) -> i64 {
    let q = a.div_euclid(b);
    let r = a.rem_euclid(b);
    if r * 2 >= b {
        q + 1
    } else {
        q
    }
}

fn projection_fp(
    x: &[Goldilocks],
    w: &[Goldilocks],
    b: &[Goldilocks],
    m: usize,
    k: usize,
    n: usize,
    shift: u32,
) -> Vec<Goldilocks> {
    let h = mm(x, w, m, k, n);
    (0..m * n)
        .map(|ij| from_i64(round_div(to_i64(h[ij]), 1i64 << shift) + to_i64(b[ij])))
        .collect()
}

/// Attention forward: x -> y = x + o.
fn attention_forward(
    x: &[Goldilocks],
    wq: &[Goldilocks],
    wk: &[Goldilocks],
    wv: &[Goldilocks],
    wo: &[Goldilocks],
    bias: &[Goldilocks],
    exp_table: &[Goldilocks],
    m: usize,
    d: usize,
    shift: u32,
) -> Vec<Goldilocks> {
    let q = projection_fp(x, wq, bias, m, d, d, shift);
    let k = projection_fp(x, wk, bias, m, d, d, shift);
    let v = projection_fp(x, wv, bias, m, d, d, shift);
    let kt = transpose(&k, m, d);
    let scores = mm(&q, &kt, m, d, m);
    let indices: Vec<u32> = scores
        .iter()
        .map(|&s| ((to_i64(s).max(0)) as u64 % exp_table.len() as u64) as u32)
        .collect();
    let e: Vec<Goldilocks> = indices.iter().map(|&i| exp_table[i as usize]).collect();
    let sum: Vec<Goldilocks> = (0..m).map(|i| (0..m).fold(Goldilocks::ZERO, |a, j| a + e[i * m + j])).collect();
    let probs: Vec<Goldilocks> = (0..m * m).map(|ij| e[ij] * sum[ij / m].inverse()).collect();
    let attn = mm(&probs, &v, m, m, d);
    let o = projection_fp(&attn, wo, bias, m, d, d, shift);
    (0..m * d).map(|i| x[i] + o[i]).collect()
}

#[allow(clippy::too_many_arguments)]
pub fn prove_transformer_chain(
    x: &[Goldilocks],
    wq: &[Goldilocks],
    wk: &[Goldilocks],
    wv: &[Goldilocks],
    wo: &[Goldilocks],
    bias: &[Goldilocks],
    exp_table: &[Goldilocks],
    fc_w: &[Goldilocks],
    fc_b: &[Goldilocks],
    proj_w: &[Goldilocks],
    proj_b: &[Goldilocks],
    gelu_table: &[Goldilocks],
    m: usize,
    d: usize,
    ffn: usize,
    shift: u32,
    rng: &mut XorShift64,
) -> TransformerChainProof {
    let y = attention_forward(x, wq, wk, wv, wo, bias, exp_table, m, d, shift);
    let attention = prove_attention_chain(x, wq, wk, wv, wo, bias, exp_table, m, d, shift, rng);
    let ffn = prove_ffn_chain(&y, fc_w, fc_b, proj_w, proj_b, gelu_table, m, d, ffn, shift, rng);

    // y is claimed by: attention residual (r), ffn's fc matmul (ch ++ u), ffn residual (r).
    let mut y_fc = ffn.fc.ch.clone();
    y_fc.extend_from_slice(&ffn.fc.u);
    let claims = vec![
        (attention.r.clone(), mle::eval(&y, &attention.r)),
        (y_fc.clone(), mle::eval(&y, &y_fc)),
        (ffn.r.clone(), mle::eval(&y, &ffn.r)),
    ];
    let same_y = prove_same_poly(&y, &claims, rng);
    TransformerChainProof { attention, ffn, same_y }
}

#[allow(clippy::too_many_arguments)]
pub fn verify_transformer_chain(
    proof: &TransformerChainProof,
    x: &[Goldilocks],
    wq: &[Goldilocks],
    wk: &[Goldilocks],
    wv: &[Goldilocks],
    wo: &[Goldilocks],
    bias: &[Goldilocks],
    exp_table: &[Goldilocks],
    fc_w: &[Goldilocks],
    fc_b: &[Goldilocks],
    proj_w: &[Goldilocks],
    proj_b: &[Goldilocks],
    gelu_table: &[Goldilocks],
    m: usize,
    d: usize,
    ffn: usize,
    shift: u32,
) -> bool {
    let y = attention_forward(x, wq, wk, wv, wo, bias, exp_table, m, d, shift);
    if !verify_attention_chain(&proof.attention, x, wq, wk, wv, wo, bias, exp_table, m, d, shift) {
        return false;
    }
    if !verify_ffn_chain(&proof.ffn, &y, fc_w, fc_b, proj_w, proj_b, gelu_table, m, d, ffn, shift) {
        return false;
    }
    let mut y_fc = proof.ffn.fc.ch.clone();
    y_fc.extend_from_slice(&proof.ffn.fc.u);
    let claims = vec![
        (proof.attention.r.clone(), mle::eval(&y, &proof.attention.r)),
        (y_fc.clone(), mle::eval(&y, &y_fc)),
        (proof.ffn.r.clone(), mle::eval(&y, &proof.ffn.r)),
    ];
    verify_same_poly(&proof.same_y, &y, &claims).is_some()
}

/// RMSNorm forward (witness) matching `layernorm_chain`.
#[allow(clippy::too_many_arguments)]
fn layernorm_forward(
    x: &[Goldilocks],
    w: &[Goldilocks],
    b: &[Goldilocks],
    rsqrt_table: &[Goldilocks],
    m: usize,
    d: usize,
) -> Vec<Goldilocks> {
    let mean_sq: Vec<Goldilocks> = (0..m)
        .map(|i| (0..d).fold(Goldilocks::ZERO, |a, j| a + x[i * d + j] * x[i * d + j]))
        .collect();
    let rsqrt: Vec<Goldilocks> = mean_sq
        .iter()
        .map(|&v| rsqrt_table[((to_i64(v).max(0)) as u64 % rsqrt_table.len() as u64) as usize])
        .collect();
    let scale: Vec<Goldilocks> = (0..m * d).map(|ij| rsqrt[ij / d] * w[ij]).collect();
    (0..m * d).map(|ij| x[ij] * scale[ij] + b[ij]).collect()
}

/// Full post-norm transformer layer: h = layernorm(x), then attention + FFN on h,
/// with `same_poly` binding the layernorm output `h` to the attention's input.
#[allow(clippy::too_many_arguments)]
pub fn prove_transformer_layer(
    x: &[Goldilocks],
    wq: &[Goldilocks],
    wk: &[Goldilocks],
    wv: &[Goldilocks],
    wo: &[Goldilocks],
    bias: &[Goldilocks],
    exp_table: &[Goldilocks],
    fc_w: &[Goldilocks],
    fc_b: &[Goldilocks],
    proj_w: &[Goldilocks],
    proj_b: &[Goldilocks],
    gelu_table: &[Goldilocks],
    ln_w: &[Goldilocks],
    ln_b: &[Goldilocks],
    rsqrt_table: &[Goldilocks],
    m: usize,
    d: usize,
    ffn: usize,
    shift: u32,
    rng: &mut XorShift64,
) -> TransformerLayerProof {
    let h = layernorm_forward(x, ln_w, ln_b, rsqrt_table, m, d);
    let layernorm = prove_layernorm_chain(x, ln_w, ln_b, rsqrt_table, m, d, rng);
    let block = prove_transformer_chain(
        &h, wq, wk, wv, wo, bias, exp_table, fc_w, fc_b, proj_w, proj_b, gelu_table, m, d, ffn, shift, rng,
    );
    let claims = vec![
        (layernorm.r_out.clone(), mle::eval(&h, &layernorm.r_out)),
        (
            block.attention.same_x.merged_point.clone(),
            block.attention.same_x.merged_eval,
        ),
    ];
    let same_h = prove_same_poly(&h, &claims, rng);
    TransformerLayerProof { layernorm, block, same_h }
}

#[allow(clippy::too_many_arguments)]
pub fn verify_transformer_layer(
    proof: &TransformerLayerProof,
    x: &[Goldilocks],
    wq: &[Goldilocks],
    wk: &[Goldilocks],
    wv: &[Goldilocks],
    wo: &[Goldilocks],
    bias: &[Goldilocks],
    exp_table: &[Goldilocks],
    fc_w: &[Goldilocks],
    fc_b: &[Goldilocks],
    proj_w: &[Goldilocks],
    proj_b: &[Goldilocks],
    gelu_table: &[Goldilocks],
    ln_w: &[Goldilocks],
    ln_b: &[Goldilocks],
    rsqrt_table: &[Goldilocks],
    m: usize,
    d: usize,
    ffn: usize,
    shift: u32,
) -> bool {
    let h = layernorm_forward(x, ln_w, ln_b, rsqrt_table, m, d);
    if !verify_layernorm_chain(&proof.layernorm, x, ln_w, ln_b, rsqrt_table, m, d) {
        return false;
    }
    if !verify_transformer_chain(
        &proof.block, &h, wq, wk, wv, wo, bias, exp_table, fc_w, fc_b, proj_w, proj_b, gelu_table,
        m, d, ffn, shift,
    ) {
        return false;
    }
    let claims = vec![
        (proof.layernorm.r_out.clone(), mle::eval(&h, &proof.layernorm.r_out)),
        (
            proof.block.attention.same_x.merged_point.clone(),
            proof.block.attention.same_x.merged_eval,
        ),
    ];
    verify_same_poly(&proof.same_h, &h, &claims).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transformer_chain_roundtrip() {
        let mut rng = XorShift64::new(0x7A7A);
        let (m, d, ffn) = (4usize, 4usize, 4usize);
        let shift = 8u32;
        let x: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let wq: Vec<Goldilocks> = (0..d * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let wk: Vec<Goldilocks> = (0..d * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let wv: Vec<Goldilocks> = (0..d * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let wo: Vec<Goldilocks> = (0..d * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let bias: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect();
        let exp_table: Vec<Goldilocks> = (0..(1usize << 8)).map(|_| rng.field()).collect();
        let fc_w: Vec<Goldilocks> = (0..d * ffn).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let fc_b: Vec<Goldilocks> = (0..m * ffn).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect();
        let proj_w: Vec<Goldilocks> = (0..ffn * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let proj_b: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect();
        let gelu_table: Vec<Goldilocks> = (0..64).map(|j| from_i64((j as i64).pow(2) % 1000)).collect();
        let proof = prove_transformer_chain(
            &x, &wq, &wk, &wv, &wo, &bias, &exp_table, &fc_w, &fc_b, &proj_w, &proj_b,
            &gelu_table, m, d, ffn, shift, &mut rng,
        );
        assert!(verify_transformer_chain(
            &proof, &x, &wq, &wk, &wv, &wo, &bias, &exp_table, &fc_w, &fc_b, &proj_w, &proj_b,
            &gelu_table, m, d, ffn, shift
        ));
    }

    #[test]
    fn transformer_layer_roundtrip() {
        let mut rng = XorShift64::new(0xB7B7);
        let (m, d, ffn) = (4usize, 4usize, 4usize);
        let shift = 8u32;
        let x: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let wq: Vec<Goldilocks> = (0..d * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let wk: Vec<Goldilocks> = (0..d * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let wv: Vec<Goldilocks> = (0..d * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let wo: Vec<Goldilocks> = (0..d * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let bias: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect();
        let exp_table: Vec<Goldilocks> = (0..(1usize << 8)).map(|_| rng.field()).collect();
        let fc_w: Vec<Goldilocks> = (0..d * ffn).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let fc_b: Vec<Goldilocks> = (0..m * ffn).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect();
        let proj_w: Vec<Goldilocks> = (0..ffn * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let proj_b: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect();
        let gelu_table: Vec<Goldilocks> = (0..64).map(|j| from_i64((j as i64).pow(2) % 1000)).collect();
        let ln_w: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 5) as i64 + 1)).collect();
        let ln_b: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect();
        let rsqrt_table: Vec<Goldilocks> = (0..(1usize << 8)).map(|_| rng.field()).collect();
        let proof = prove_transformer_layer(
            &x, &wq, &wk, &wv, &wo, &bias, &exp_table, &fc_w, &fc_b, &proj_w, &proj_b,
            &gelu_table, &ln_w, &ln_b, &rsqrt_table, m, d, ffn, shift, &mut rng,
        );
        assert!(verify_transformer_layer(
            &proof, &x, &wq, &wk, &wv, &wo, &bias, &exp_table, &fc_w, &fc_b, &proj_w, &proj_b,
            &gelu_table, &ln_w, &ln_b, &rsqrt_table, m, d, ffn, shift
        ));
    }
}
