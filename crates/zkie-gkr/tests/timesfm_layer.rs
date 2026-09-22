//! Full TimeFM decoder layer composed out of the GKR primitives.
//!
//! `input_layernorm -> QKV projection -> attention (QK^T, softmax, PV) ->
//! pre_ffn_layernorm -> FFN (matmul, GELU, matmul)`. Every matmul is a GKR
//! sum-check; every nonlinearity (RMSNorm's rsqrt, softmax's exp, GELU) is a
//! LogUp lookup. This demonstrates that the arithmetization now covers every op
//! a real TimesFM layer needs, not just matmul + softmax.

use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing, PrimeField64, XorShift64};
use zkie_gkr::{gelu, matmul, mle, rmsnorm, softmax};

fn dense(a: &[Goldilocks], b: &[Goldilocks], m: usize, k: usize, n: usize) -> Vec<Goldilocks> {
    let mut c = vec![Goldilocks::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = Goldilocks::ZERO;
            for w in 0..k {
                acc = acc + a[i * k + w] * b[w * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn transpose(a: &[Goldilocks], m: usize, k: usize) -> Vec<Goldilocks> {
    let mut at = vec![Goldilocks::ZERO; k * m];
    for i in 0..m {
        for w in 0..k {
            at[w * m + i] = a[i * k + w];
        }
    }
    at
}

fn matmul_ok(
    at: &[Goldilocks],
    b: &[Goldilocks],
    c: &[Goldilocks],
    m: usize,
    k: usize,
    n: usize,
    rng: &mut XorShift64,
) -> bool {
    let u: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let ch: Vec<Goldilocks> = (0..k.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let proof = matmul::prove(at, b, c, m, k, n, &u, &v, &ch);
    let mut fp = u.clone();
    fp.extend_from_slice(&ch);
    let f = mle::eval(at, &fp);
    let mut hp = v.clone();
    hp.extend_from_slice(&ch);
    let h = mle::eval(b, &hp);
    matmul::verify(&proof, &ch, f, h)
}

/// Apply RMSNorm row-wise (TimeFM normalizes over the hidden dim, sharing the
/// per-hidden-dim weight across rows).
fn rms_norm_batched(
    x: &[Goldilocks],
    weight: &[Goldilocks],
    seq: usize,
    hidden: usize,
    eps: Goldilocks,
    rsqrt_table: &[Goldilocks],
    rng: &mut XorShift64,
) -> Vec<Goldilocks> {
    let mut y = vec![Goldilocks::ZERO; seq * hidden];
    for r in 0..seq {
        let s_idx = (rng.next_u64() % rsqrt_table.len() as u64) as u32;
        let row = &x[r * hidden..(r + 1) * hidden];
        let (y_row, proof) = rmsnorm::rms_norm(
            row, weight, eps, s_idx, rsqrt_table, rng.field(), rng.field(),
        );
        assert!(rmsnorm::verify(row, weight, &y_row, eps, s_idx, rsqrt_table, &proof));
        y[r * hidden..(r + 1) * hidden].copy_from_slice(&y_row);
    }
    y
}

#[test]
fn timesfm_layer_arithmetization_covers_all_ops() {
    let mut rng = XorShift64::new(0x7f4d);
    let (seq, hidden) = (4usize, 4usize);
    let table_size = 64usize;

    let w_rms1: Vec<Goldilocks> = (0..hidden).map(|_| rng.field()).collect();
    let w_rms2: Vec<Goldilocks> = (0..hidden).map(|_| rng.field()).collect();
    let w_q: Vec<Goldilocks> = (0..hidden * hidden).map(|_| rng.field()).collect();
    let w_k: Vec<Goldilocks> = (0..hidden * hidden).map(|_| rng.field()).collect();
    let w_v: Vec<Goldilocks> = (0..hidden * hidden).map(|_| rng.field()).collect();
    let w_f1: Vec<Goldilocks> = (0..hidden * hidden).map(|_| rng.field()).collect();
    let w_f2: Vec<Goldilocks> = (0..hidden * hidden).map(|_| rng.field()).collect();

    let rsqrt_table: Vec<Goldilocks> = (0..table_size).map(|_| rng.field()).collect();
    let exp_table: Vec<Goldilocks> = (0..table_size).map(|_| rng.field()).collect();
    let gelu_table: Vec<Goldilocks> = (0..table_size).map(|_| rng.field()).collect();

    let x: Vec<Goldilocks> = (0..seq * hidden).map(|_| rng.field()).collect();

    let eps = Goldilocks::from_u64(1);
    let x_norm = rms_norm_batched(&x, &w_rms1, seq, hidden, eps, &rsqrt_table, &mut rng);

    let q = dense(&x_norm, &w_q, seq, hidden, hidden);
    let k = dense(&x_norm, &w_k, seq, hidden, hidden);
    let v = dense(&x_norm, &w_v, seq, hidden, hidden);
    assert!(matmul_ok(&transpose(&x_norm, seq, hidden), &w_q, &q, seq, hidden, hidden, &mut rng));
    assert!(matmul_ok(&transpose(&x_norm, seq, hidden), &w_k, &k, seq, hidden, hidden, &mut rng));
    assert!(matmul_ok(&transpose(&x_norm, seq, hidden), &w_v, &v, seq, hidden, hidden, &mut rng));

    let kt = transpose(&k, seq, hidden);
    let s = dense(&q, &kt, seq, hidden, seq);
    assert!(matmul_ok(&transpose(&q, seq, hidden), &kt, &s, seq, hidden, seq, &mut rng));
    let s_indices: Vec<u32> = s
        .iter()
        .map(|z| (z.as_canonical_u64() % table_size as u64) as u32)
        .collect();
    let (p, softmax_proof) = softmax::softmax(&s_indices, &exp_table, rng.field(), rng.field());
    assert!(softmax::verify(&s_indices, &exp_table, &p, &softmax_proof));

    let o = dense(&p, &v, seq, seq, hidden);
    assert!(matmul_ok(&transpose(&p, seq, seq), &v, &o, seq, seq, hidden, &mut rng));

    let o_norm = rms_norm_batched(&o, &w_rms2, seq, hidden, eps, &rsqrt_table, &mut rng);

    let f1 = dense(&o_norm, &w_f1, seq, hidden, hidden);
    assert!(matmul_ok(&transpose(&o_norm, seq, hidden), &w_f1, &f1, seq, hidden, hidden, &mut rng));
    let f1_indices: Vec<u32> = f1
        .iter()
        .map(|z| (z.as_canonical_u64() % table_size as u64) as u32)
        .collect();
    let (g, gelu_proof) = gelu::gelu(&f1_indices, &gelu_table, rng.field(), rng.field());
    assert!(gelu::verify(&f1_indices, &gelu_table, &g, &gelu_proof));
    let f2 = dense(&g, &w_f2, seq, hidden, hidden);
    assert!(matmul_ok(&transpose(&g, seq, hidden), &w_f2, &f2, seq, hidden, hidden, &mut rng));
}
