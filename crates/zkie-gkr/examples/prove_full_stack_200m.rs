//! Prove the whole TimesFM 200M stack (20 layers, real 16-token attention) as
//! one chained circuit.
//!
//! Per layer: RMSNorm -> Q/K/V projection -> QK^T -> softmax -> PV -> o_proj ->
//! residual -> LayerNorm -> gate -> ReLU -> down -> residual. Unlike the 8M
//! stack, seq = 16, so the attention is non-degenerate and every matmul has
//! m = seq = 16. The row-wise norms normalize each of the 16 tokens
//! independently; the per-head_dim Q scale is folded into the Q projection at
//! extraction time (see `extract_200m_attn.py`).

use zkie_gkr::committed::{
    affine_raw, commit, layer_norm_raw, prove_add, prove_affine, prove_layer_norm, prove_lookup,
    prove_matmul, prove_relu, prove_rms_norm, prove_scale, prove_softmax_rows, rms_norm_raw,
    scale_raw, silu_raw,
};
use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_gkr::fixed_point::{from_i32, to_i32};
use zkie_gkr::whir::Whir;

const H_PAD: usize = 2048;
const SEQ: usize = 16;
const HEADS: usize = 16;
const HDIM: usize = 80;
const HDIM_PAD: usize = 128;
const N_REAL: usize = 1280;
const N_LAYERS: usize = 20;
const SILU_OFFSET: u32 = 1 << 19;

fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = std::fs::read(path).expect("run the extract scripts first");
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let v = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.push(from_i32(v));
    }
    out
}

fn dense_m(a: &[Goldilocks], b: &[Goldilocks], m: usize, k: usize, n: usize) -> Vec<Goldilocks> {
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

fn add_vec(a: &[Goldilocks], b: &[Goldilocks]) -> Vec<Goldilocks> {
    a.iter().zip(b).map(|(&x, &y)| x + y).collect()
}

/// Tile a per-column bias `[h]` across the `seq` rows to `[seq, h]`, matching
/// the ONNX broadcasting of a 1-D bias over the batch/sequence dimension.
fn broadcast_bias(bias: &[Goldilocks], seq: usize) -> Vec<Goldilocks> {
    let h = bias.len();
    let mut out = Vec::with_capacity(seq * h);
    for _ in 0..seq {
        out.extend_from_slice(bias);
    }
    out
}

/// Round-half-up integer division (mirrors `committed::div_round`).
fn dr(a: i64, b: i64) -> i64 {
    let q = a.div_euclid(b);
    let r = a.rem_euclid(b);
    if r * 2 >= b { q + 1 } else { q }
}

/// Reshape `q` / `v` from `[SEQ, H_PAD]` into `[HEADS, SEQ, HDIM_PAD]`,
/// zero-padding each head's head_dim to the next power of two.
fn split_qv(x: &[Goldilocks]) -> Vec<Goldilocks> {
    let mut out = vec![Goldilocks::ZERO; HEADS * SEQ * HDIM_PAD];
    for h in 0..HEADS {
        for s in 0..SEQ {
            for j in 0..HDIM {
                out[(h * SEQ + s) * HDIM_PAD + j] = x[s * H_PAD + h * HDIM + j];
            }
        }
    }
    out
}

/// Reshape `k` from `[SEQ, H_PAD]` into `[HEADS, HDIM_PAD, SEQ]`.
fn split_k(x: &[Goldilocks]) -> Vec<Goldilocks> {
    let mut out = vec![Goldilocks::ZERO; HEADS * HDIM_PAD * SEQ];
    for h in 0..HEADS {
        for j in 0..HDIM {
            for s in 0..SEQ {
                out[(h * HDIM_PAD + j) * SEQ + s] = x[s * H_PAD + h * HDIM + j];
            }
        }
    }
    out
}

/// Collapse `attn` from `[HEADS, SEQ, HDIM_PAD]` back into `[SEQ, H_PAD]`,
/// dropping each head's head_dim padding tail.
fn concat_attn(x: &[Goldilocks]) -> Vec<Goldilocks> {
    let mut out = vec![Goldilocks::ZERO; SEQ * H_PAD];
    for h in 0..HEADS {
        for s in 0..SEQ {
            for j in 0..HDIM {
                out[s * H_PAD + h * HDIM + j] = x[(h * SEQ + s) * HDIM_PAD + j];
            }
        }
    }
    out
}

/// Row-wise RMSNorm: normalize each of the `SEQ` tokens independently.
/// Returns `(raw, norm)` where `norm` is the rescaled `[SEQ, H_PAD]` tensor.
fn rms_norm_rows(
    x: &[Goldilocks],
    weight: &[Goldilocks],
    n_real: usize,
    rsqrt: &[Goldilocks],
) -> (Vec<Goldilocks>, Vec<Goldilocks>) {
    let h = x.len() / SEQ;
    let zero = vec![from_i32(0); h];
    let mut raw = Vec::with_capacity(x.len());
    let mut norm = Vec::with_capacity(x.len());
    for s in 0..SEQ {
        let row = &x[s * h..(s + 1) * h];
        let (r, _, _) = rms_norm_raw(row, weight, n_real, rsqrt);
        let n = affine_raw(&r, &zero, 32, false);
        raw.extend_from_slice(&r);
        norm.extend_from_slice(&n);
    }
    (raw, norm)
}

/// Row-wise LayerNorm. Returns `(raw, out)` where `out` is the biased
/// `[SEQ, H_PAD]` tensor.
fn layer_norm_rows(
    x: &[Goldilocks],
    weight: &[Goldilocks],
    bias: &[Goldilocks],
    n_real: usize,
    rsqrt: &[Goldilocks],
) -> (Vec<Goldilocks>, Vec<Goldilocks>) {
    let h = x.len() / SEQ;
    let mut raw = Vec::with_capacity(x.len());
    let mut out = Vec::with_capacity(x.len());
    for s in 0..SEQ {
        let row = &x[s * h..(s + 1) * h];
        let (r, _, _, _) = layer_norm_raw(row, weight, n_real, rsqrt);
        let o = affine_raw(&r, bias, 32, false);
        raw.extend_from_slice(&r);
        out.extend_from_slice(&o);
    }
    (raw, out)
}

/// Prove the row-wise RMSNorm against per-row `[H_PAD]` commitments: for each
/// token, bind `raw = x * rstd * w` (rstd derived from that row) and then
/// `norm = round(raw / 2^32)`.
#[allow(clippy::too_many_arguments)]
fn prove_rms_norm_rows(
    whir_h: &Whir,
    x_plain: &[Goldilocks],
    raw_plain: &[Goldilocks],
    norm_plain: &[Goldilocks],
    weight: &[Goldilocks],
    n_real: usize,
    rsqrt: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
    rng: &mut XorShift64,
) -> bool {
    let h = x_plain.len() / SEQ;
    let zero = vec![from_i32(0); h];
    for s in 0..SEQ {
        let xr = &x_plain[s * h..(s + 1) * h];
        let rr = &raw_plain[s * h..(s + 1) * h];
        let nr = &norm_plain[s * h..(s + 1) * h];
        let cx = commit(whir_h, xr);
        let cr = commit(whir_h, rr);
        let cn = commit(whir_h, nr);
        if !prove_rms_norm(whir_h, &cx, xr, whir_h, &cr, rr, weight, n_real, rsqrt, alpha, beta, rng)
            || !prove_affine(whir_h, &cr, rr, whir_h, &cn, nr, &zero, 32, rng)
        {
            return false;
        }
    }
    true
}

/// Prove the row-wise LayerNorm against per-row `[H_PAD]` commitments.
#[allow(clippy::too_many_arguments)]
fn prove_layer_norm_rows(
    whir_h: &Whir,
    x_plain: &[Goldilocks],
    raw_plain: &[Goldilocks],
    out_plain: &[Goldilocks],
    weight: &[Goldilocks],
    bias: &[Goldilocks],
    n_real: usize,
    rsqrt: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
    rng: &mut XorShift64,
) -> bool {
    let h = x_plain.len() / SEQ;
    for s in 0..SEQ {
        let xr = &x_plain[s * h..(s + 1) * h];
        let rr = &raw_plain[s * h..(s + 1) * h];
        let or = &out_plain[s * h..(s + 1) * h];
        let cx = commit(whir_h, xr);
        let cr = commit(whir_h, rr);
        let co = commit(whir_h, or);
        if !prove_layer_norm(whir_h, &cx, xr, whir_h, &cr, rr, weight, n_real, rsqrt, alpha, beta, rng)
            || !prove_affine(whir_h, &cr, rr, whir_h, &co, or, bias, 32, rng)
        {
            return false;
        }
    }
    true
}

fn main() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/");
    let stack = format!("{base}full_stack_200m/");
    let rsqrt_table = load_i32(&format!("{base}rsqrt_table_i32.bin"));
    let silu_table = load_i32(&format!("{base}silu_table_i32.bin"));
    let exp_table = load_i32(&format!("{base}exp_table_i32.bin"));
    const EXP_OFFSET: u32 = 1 << 21;

    let mut rng = XorShift64::new(0x200);
    let whir8 = Whir::new_testing(8);
    let whir10 = Whir::new_testing(10);
    let whir11 = Whir::new_testing(11);
    let whir12 = Whir::new_testing(12);
    let whir15 = Whir::new_testing(15);
    let whir17 = Whir::new_testing(17);
    let whir22 = Whir::new_testing(22);

    // --- Prologue: the RevIN input embedding is deterministic public-input
    // preprocessing (mean/rstd over the first 32 timestamps, applied globally),
    // precomputed by `extract_200m_full.py`; the proof starts from `cat`.
    let cat = load_i32(&format!("{stack}cat_i32.bin"));

    // --- Input FFN: cat [SEQ, 64] -> SiLU FFN -> add.
    let pro_hid_w = load_i32(&format!("{stack}pro_hid_w_i32.bin"));
    let pro_hid_b = broadcast_bias(&load_i32(&format!("{stack}pro_hid_b_i32.bin")), SEQ);
    let pro_out_w = load_i32(&format!("{stack}pro_out_w_i32.bin"));
    let pro_out_b = broadcast_bias(&load_i32(&format!("{stack}pro_out_b_i32.bin")), SEQ);
    let pro_res_w = load_i32(&format!("{stack}pro_res_w_i32.bin"));
    let pro_res_b = broadcast_bias(&load_i32(&format!("{stack}pro_res_b_i32.bin")), SEQ);
    let gather = load_i32(&format!("{stack}gather_i32.bin"));
    let embedding = broadcast_bias(&load_i32(&format!("{stack}embedding_i32.bin")), SEQ);

    let hid_raw = dense_m(&cat, &pro_hid_w, SEQ, 64, H_PAD);
    let linear = affine_raw(&hid_raw, &pro_hid_b, 16, false);
    let (silu_idx, silu) = silu_raw(&linear, &silu_table, SILU_OFFSET);
    let out_raw = dense_m(&silu, &pro_out_w, SEQ, H_PAD, H_PAD);
    let linear_1 = affine_raw(&out_raw, &pro_out_b, 16, false);
    let res_raw = dense_m(&cat, &pro_res_w, SEQ, 64, H_PAD);
    let linear_2 = affine_raw(&res_raw, &pro_res_b, 16, false);
    let add = add_vec(&linear_1, &linear_2);
    let add_1 = add_vec(&add, &gather);
    let x = add_vec(&add_1, &embedding);

    let c_cat = commit(&whir10, &cat);
    let c_hidw = commit(&whir17, &pro_hid_w);
    let c_hidraw = commit(&whir15, &hid_raw);
    assert!(prove_matmul(
        &whir10, &c_cat, &whir17, &c_hidw, &whir15, &c_hidraw, &cat, &pro_hid_w, &hid_raw,
        SEQ, 64, H_PAD, &mut rng,
    ));
    let c_linear = commit(&whir15, &linear);
    assert!(prove_affine(
        &whir15, &c_hidraw, &hid_raw, &whir15, &c_linear, &linear, &pro_hid_b, 16, &mut rng,
    ));
    let idx_field: Vec<Goldilocks> = silu_idx.iter().map(|&i| from_i32(i as i32)).collect();
    let c_idx = commit(&whir15, &idx_field);
    let c_silu = commit(&whir15, &silu);
    assert!(prove_lookup(
        &whir15, &c_idx, &idx_field, &whir15, &c_silu, &silu, &silu_idx, &silu_table,
        rng.field(), rng.field(), &mut rng,
    ));
    let c_outw = commit(&whir22, &pro_out_w);
    let c_outraw = commit(&whir15, &out_raw);
    assert!(prove_matmul(
        &whir15, &c_silu, &whir22, &c_outw, &whir15, &c_outraw, &silu, &pro_out_w, &out_raw,
        SEQ, H_PAD, H_PAD, &mut rng,
    ));
    let c_lin1 = commit(&whir15, &linear_1);
    assert!(prove_affine(
        &whir15, &c_outraw, &out_raw, &whir15, &c_lin1, &linear_1, &pro_out_b, 16, &mut rng,
    ));
    let c_resw = commit(&whir17, &pro_res_w);
    let c_resraw = commit(&whir15, &res_raw);
    assert!(prove_matmul(
        &whir10, &c_cat, &whir17, &c_resw, &whir15, &c_resraw, &cat, &pro_res_w, &res_raw,
        SEQ, 64, H_PAD, &mut rng,
    ));
    let c_lin2 = commit(&whir15, &linear_2);
    assert!(prove_affine(
        &whir15, &c_resraw, &res_raw, &whir15, &c_lin2, &linear_2, &pro_res_b, 16, &mut rng,
    ));
    let c_add = commit(&whir15, &add);
    assert!(prove_add(&whir15, &c_lin1, &whir15, &c_lin2, &whir15, &c_add, SEQ * H_PAD, &mut rng));
    let c_gather = commit(&whir15, &gather);
    let c_add1 = commit(&whir15, &add_1);
    assert!(prove_add(&whir15, &c_add, &whir15, &c_gather, &whir15, &c_add1, SEQ * H_PAD, &mut rng));
    let c_emb = commit(&whir15, &embedding);
    let c_x = commit(&whir15, &x);
    assert!(prove_add(&whir15, &c_add1, &whir15, &c_emb, &whir15, &c_x, SEQ * H_PAD, &mut rng));

    println!("TimesFM 200M: prologue (RevIN + input FFN + freq) verified");

    let mask = load_i32(&format!("{stack}mask_q_i32.bin"));
    let zero_hd = vec![from_i32(0); SEQ * HDIM_PAD];
    let mut x = x;

    let t_stack = std::time::Instant::now();
    let stats_before = || {
        [
            ("whir8", whir8.commit_stats()),
            ("whir10", whir10.commit_stats()),
            ("whir11", whir11.commit_stats()),
            ("whir12", whir12.commit_stats()),
            ("whir15", whir15.commit_stats()),
            ("whir17", whir17.commit_stats()),
            ("whir22", whir22.commit_stats()),
        ]
    };
    let fmt_stats = |s: [(&str, (u64, f64)); 7]| {
        let total: f64 = s.iter().map(|(_, (_, t))| t).sum();
        let per: Vec<String> = s
            .iter()
            .map(|(name, (n, t))| format!("{name}={t:.1}s/{n}"))
            .collect();
        format!("whir commits: {total:.1}s [{}]", per.join(", "))
    };
    let mut prev_stats = stats_before();

    for li in 0..N_LAYERS {
        let t_layer = std::time::Instant::now();
        let lnw = load_i32(&format!("{stack}L{li}_lnw_i32.bin"));
        let q_w = load_i32(&format!("{stack}L{li}_q_w_i32.bin"));
        let k_w = load_i32(&format!("{stack}L{li}_k_w_i32.bin"));
        let v_w = load_i32(&format!("{stack}L{li}_v_w_i32.bin"));
        let q_b = broadcast_bias(&load_i32(&format!("{stack}L{li}_q_b_i32.bin")), SEQ);
        let k_b = broadcast_bias(&load_i32(&format!("{stack}L{li}_k_b_i32.bin")), SEQ);
        let v_b = broadcast_bias(&load_i32(&format!("{stack}L{li}_v_b_i32.bin")), SEQ);
        let op_w = load_i32(&format!("{stack}L{li}_o_proj_w_i32.bin"));
        let op_b = broadcast_bias(&load_i32(&format!("{stack}L{li}_o_proj_b_i32.bin")), SEQ);
        let mlp_w = load_i32(&format!("{stack}L{li}_mlp_w_i32.bin"));
        let mlp_b = load_i32(&format!("{stack}L{li}_mlp_b_i32.bin"));
        let gate_w = load_i32(&format!("{stack}L{li}_gate_w_i32.bin"));
        let gate_b = broadcast_bias(&load_i32(&format!("{stack}L{li}_gate_b_i32.bin")), SEQ);
        let down_w = load_i32(&format!("{stack}L{li}_down_w_i32.bin"));
        let down_b = broadcast_bias(&load_i32(&format!("{stack}L{li}_down_b_i32.bin")), SEQ);

        // --- Host-side forward pass.
        let (rms_raw, mul9) = rms_norm_rows(&x, &lnw, N_REAL, &rsqrt_table);
        let q_raw = dense_m(&mul9, &q_w, SEQ, H_PAD, H_PAD);
        let q = affine_raw(&q_raw, &q_b, 16, false);
        let k_raw = dense_m(&mul9, &k_w, SEQ, H_PAD, H_PAD);
        let k = affine_raw(&k_raw, &k_b, 16, false);
        let v_raw = dense_m(&mul9, &v_w, SEQ, H_PAD, H_PAD);
        let v = affine_raw(&v_raw, &v_b, 16, false);
        let qh = split_qv(&q);
        let kh = split_k(&k);
        let vh = split_qv(&v);

        let mut scores = vec![Goldilocks::ZERO; HEADS * SEQ * SEQ];
        let mut scores_raw_all = vec![Goldilocks::ZERO; HEADS * SEQ * SEQ];
        for h in 0..HEADS {
            let qh_h = &qh[h * SEQ * HDIM_PAD..(h + 1) * SEQ * HDIM_PAD];
            let kh_h = &kh[h * HDIM_PAD * SEQ..(h + 1) * HDIM_PAD * SEQ];
            let sr = dense_m(qh_h, kh_h, SEQ, HDIM_PAD, SEQ);
            let sc = affine_raw(&sr, &mask, 16, false);
            scores_raw_all[h * SEQ * SEQ..(h + 1) * SEQ * SEQ].copy_from_slice(&sr);
            scores[h * SEQ * SEQ..(h + 1) * SEQ * SEQ].copy_from_slice(&sc);
        }

        let n_rows = HEADS * SEQ;
        let c: Vec<Goldilocks> = (0..n_rows)
            .map(|r| from_i32((0..SEQ).map(|k| to_i32(scores[r * SEQ + k])).max().unwrap()))
            .collect();
        let shifted: Vec<Goldilocks> = scores
            .iter()
            .enumerate()
            .map(|(i, &s)| s - c[i / SEQ])
            .collect();
        let e: Vec<Goldilocks> = shifted
            .iter()
            .map(|&s| {
                let idx = (to_i32(s) as i64 + EXP_OFFSET as i64)
                    .clamp(0, exp_table.len() as i64 - 1) as usize;
                exp_table[idx]
            })
            .collect();
        let sum: Vec<Goldilocks> = (0..n_rows)
            .map(|r| (0..SEQ).fold(Goldilocks::ZERO, |a, k| a + e[r * SEQ + k]))
            .collect();
        let sum_broadcast: Vec<Goldilocks> = (0..n_rows * SEQ).map(|i| sum[i / SEQ]).collect();
        let sm: Vec<Goldilocks> = e
            .iter()
            .enumerate()
            .map(|(i, &v)| from_i32(dr(to_i32(v) as i64 * 65536, to_i32(sum[i / SEQ]) as i64) as i32))
            .collect();

        let mut attn_raw_all = vec![Goldilocks::ZERO; HEADS * SEQ * HDIM_PAD];
        let mut attn_all = vec![Goldilocks::ZERO; HEADS * SEQ * HDIM_PAD];
        for h in 0..HEADS {
            let sm_h = &sm[h * SEQ * SEQ..(h + 1) * SEQ * SEQ];
            let vh_h = &vh[h * SEQ * HDIM_PAD..(h + 1) * SEQ * HDIM_PAD];
            let ar = dense_m(sm_h, vh_h, SEQ, SEQ, HDIM_PAD);
            let at = affine_raw(&ar, &zero_hd, 16, false);
            attn_raw_all[h * SEQ * HDIM_PAD..(h + 1) * SEQ * HDIM_PAD].copy_from_slice(&ar);
            attn_all[h * SEQ * HDIM_PAD..(h + 1) * SEQ * HDIM_PAD].copy_from_slice(&at);
        }
        let attn_full = concat_attn(&attn_all);
        let op_raw = dense_m(&attn_full, &op_w, SEQ, H_PAD, H_PAD);
        let op = affine_raw(&op_raw, &op_b, 16, false);
        let add5 = add_vec(&x, &op);
        let (ln_raw, ln_out) = layer_norm_rows(&add5, &mlp_w, &mlp_b, N_REAL, &rsqrt_table);
        let gate_raw = dense_m(&ln_out, &gate_w, SEQ, H_PAD, H_PAD);
        let relu = affine_raw(&gate_raw, &gate_b, 16, true);
        let down_raw = dense_m(&relu, &down_w, SEQ, H_PAD, H_PAD);
        let ffn_out = affine_raw(&down_raw, &down_b, 16, false);
        let next_x = add_vec(&add5, &ffn_out);

        // --- Proofs.
        let alpha = rng.field();
        let beta = rng.field();
        assert!(prove_rms_norm_rows(
            &whir11, &x, &rms_raw, &mul9, &lnw, N_REAL, &rsqrt_table, alpha, beta, &mut rng,
        ));
        let c_mul9 = commit(&whir15, &mul9);
        let c_qw = commit(&whir22, &q_w);
        let c_qraw = commit(&whir15, &q_raw);
        assert!(prove_matmul(&whir15, &c_mul9, &whir22, &c_qw, &whir15, &c_qraw, &mul9, &q_w, &q_raw, SEQ, H_PAD, H_PAD, &mut rng));
        let c_q = commit(&whir15, &q);
        assert!(prove_affine(&whir15, &c_qraw, &q_raw, &whir15, &c_q, &q, &q_b, 16, &mut rng));
        let c_kw = commit(&whir22, &k_w);
        let c_kraw = commit(&whir15, &k_raw);
        assert!(prove_matmul(&whir15, &c_mul9, &whir22, &c_kw, &whir15, &c_kraw, &mul9, &k_w, &k_raw, SEQ, H_PAD, H_PAD, &mut rng));
        let c_k = commit(&whir15, &k);
        assert!(prove_affine(&whir15, &c_kraw, &k_raw, &whir15, &c_k, &k, &k_b, 16, &mut rng));
        let c_vw = commit(&whir22, &v_w);
        let c_vraw = commit(&whir15, &v_raw);
        assert!(prove_matmul(&whir15, &c_mul9, &whir22, &c_vw, &whir15, &c_vraw, &mul9, &v_w, &v_raw, SEQ, H_PAD, H_PAD, &mut rng));
        let c_v = commit(&whir15, &v);
        assert!(prove_affine(&whir15, &c_vraw, &v_raw, &whir15, &c_v, &v, &v_b, 16, &mut rng));

        for h in 0..HEADS {
            let qh_h = &qh[h * SEQ * HDIM_PAD..(h + 1) * SEQ * HDIM_PAD];
            let kh_h = &kh[h * HDIM_PAD * SEQ..(h + 1) * HDIM_PAD * SEQ];
            let sr = &scores_raw_all[h * SEQ * SEQ..(h + 1) * SEQ * SEQ];
            let sc = &scores[h * SEQ * SEQ..(h + 1) * SEQ * SEQ];
            let c_qh = commit(&whir11, qh_h);
            let c_kh = commit(&whir11, kh_h);
            let c_sr = commit(&whir8, sr);
            assert!(prove_matmul(&whir11, &c_qh, &whir11, &c_kh, &whir8, &c_sr, qh_h, kh_h, sr, SEQ, HDIM_PAD, SEQ, &mut rng));
            let c_sc = commit(&whir8, sc);
            assert!(prove_affine(&whir8, &c_sr, sr, &whir8, &c_sc, sc, &mask, 16, &mut rng));
        }

        let cs_all = commit(&whir12, &scores);
        let cc = commit(&whir8, &c);
        let csh = commit(&whir12, &shifted);
        let ce = commit(&whir12, &e);
        let csum = commit(&whir8, &sum);
        let csb = commit(&whir12, &sum_broadcast);
        let csm = commit(&whir12, &sm);
        assert!(prove_softmax_rows(
            &whir12, &whir8, &cs_all, &scores, &cc, &c, &csh, &shifted, &ce, &e, &csum, &sum,
            &csb, &sum_broadcast, &csm, &sm, &exp_table, EXP_OFFSET, n_rows, SEQ, alpha, beta, &mut rng,
        ));

        for h in 0..HEADS {
            let sm_h = &sm[h * SEQ * SEQ..(h + 1) * SEQ * SEQ];
            let vh_h = &vh[h * SEQ * HDIM_PAD..(h + 1) * SEQ * HDIM_PAD];
            let ar = &attn_raw_all[h * SEQ * HDIM_PAD..(h + 1) * SEQ * HDIM_PAD];
            let at = &attn_all[h * SEQ * HDIM_PAD..(h + 1) * SEQ * HDIM_PAD];
            let c_sm = commit(&whir8, sm_h);
            let c_vh = commit(&whir11, vh_h);
            let c_ar = commit(&whir11, ar);
            assert!(prove_matmul(&whir8, &c_sm, &whir11, &c_vh, &whir11, &c_ar, sm_h, vh_h, ar, SEQ, SEQ, HDIM_PAD, &mut rng));
            let c_at = commit(&whir11, at);
            assert!(prove_affine(&whir11, &c_ar, ar, &whir11, &c_at, at, &zero_hd, 16, &mut rng));
        }

        let c_attn = commit(&whir15, &attn_full);
        let c_opw = commit(&whir22, &op_w);
        let c_opraw = commit(&whir15, &op_raw);
        assert!(prove_matmul(&whir15, &c_attn, &whir22, &c_opw, &whir15, &c_opraw, &attn_full, &op_w, &op_raw, SEQ, H_PAD, H_PAD, &mut rng));
        let c_op = commit(&whir15, &op);
        assert!(prove_affine(&whir15, &c_opraw, &op_raw, &whir15, &c_op, &op, &op_b, 16, &mut rng));
        let c_x = commit(&whir15, &x);
        let c_add5 = commit(&whir15, &add5);
        assert!(prove_add(&whir15, &c_x, &whir15, &c_op, &whir15, &c_add5, SEQ * H_PAD, &mut rng));
        assert!(prove_layer_norm_rows(
            &whir11, &add5, &ln_raw, &ln_out, &mlp_w, &mlp_b, N_REAL, &rsqrt_table, alpha, beta, &mut rng,
        ));
        let c_lnout = commit(&whir15, &ln_out);
        let c_gatew = commit(&whir22, &gate_w);
        let c_gateraw = commit(&whir15, &gate_raw);
        assert!(prove_matmul(&whir15, &c_lnout, &whir22, &c_gatew, &whir15, &c_gateraw, &ln_out, &gate_w, &gate_raw, SEQ, H_PAD, H_PAD, &mut rng));
        let c_relu = commit(&whir15, &relu);
        assert!(prove_relu(&whir15, &c_gateraw, &gate_raw, &whir15, &c_relu, &relu, &gate_b, &mut rng));
        let c_downw = commit(&whir22, &down_w);
        let c_downraw = commit(&whir15, &down_raw);
        assert!(prove_matmul(&whir15, &c_relu, &whir22, &c_downw, &whir15, &c_downraw, &relu, &down_w, &down_raw, SEQ, H_PAD, H_PAD, &mut rng));
        let c_ffnout = commit(&whir15, &ffn_out);
        assert!(prove_affine(&whir15, &c_downraw, &down_raw, &whir15, &c_ffnout, &ffn_out, &down_b, 16, &mut rng));
        let c_next = commit(&whir15, &next_x);
        assert!(prove_add(&whir15, &c_add5, &whir15, &c_ffnout, &whir15, &c_next, SEQ * H_PAD, &mut rng));

        x = next_x;
        let now = stats_before();
        println!(
            "layer {li} verified in {:.1}s | {}",
            t_layer.elapsed().as_secs_f64(),
            fmt_stats(now)
        );
        prev_stats = now;
    }
    println!(
        "20-layer stack done in {:.1}s | {}",
        t_stack.elapsed().as_secs_f64(),
        fmt_stats(prev_stats)
    );

    // --- Epilogue: horizon FFN output head -> rescale.
    let hid_w = load_i32(&format!("{stack}head_hid_w_i32.bin"));
    let hid_b = broadcast_bias(&load_i32(&format!("{stack}head_hid_b_i32.bin")), SEQ);
    let out_w = load_i32(&format!("{stack}head_out_w_i32.bin"));
    let out_b = broadcast_bias(&load_i32(&format!("{stack}head_out_b_i32.bin")), SEQ);
    let res_w = load_i32(&format!("{stack}head_res_w_i32.bin"));
    let res_b = broadcast_bias(&load_i32(&format!("{stack}head_res_b_i32.bin")), SEQ);
    let scale_bytes = std::fs::read(format!("{stack}scale_i64.bin")).expect("scale file");
    let scale_q = i64::from_le_bytes(scale_bytes[0..8].try_into().unwrap());
    let bias_q = i64::from_le_bytes(scale_bytes[8..16].try_into().unwrap());

    let hid_raw = dense_m(&x, &hid_w, SEQ, H_PAD, H_PAD);
    let lin31 = affine_raw(&hid_raw, &hid_b, 16, false);
    let (silu_idx, silu1) = silu_raw(&lin31, &silu_table, SILU_OFFSET);
    let out_raw = dense_m(&silu1, &out_w, SEQ, H_PAD, H_PAD);
    let lin32 = affine_raw(&out_raw, &out_b, 16, false);
    let res_raw = dense_m(&x, &res_w, SEQ, H_PAD, H_PAD);
    let lin33 = affine_raw(&res_raw, &res_b, 16, false);
    let add31 = add_vec(&lin32, &lin33);
    let bias_bcast = vec![from_i32(bias_q as i32); SEQ * H_PAD];
    let output_ts = scale_raw(&add31, scale_q, &bias_bcast);

    let c_x = commit(&whir15, &x);
    let c_hidw = commit(&whir22, &hid_w);
    let c_hidraw = commit(&whir15, &hid_raw);
    assert!(prove_matmul(&whir15, &c_x, &whir22, &c_hidw, &whir15, &c_hidraw, &x, &hid_w, &hid_raw, SEQ, H_PAD, H_PAD, &mut rng));
    let c_lin31 = commit(&whir15, &lin31);
    assert!(prove_affine(&whir15, &c_hidraw, &hid_raw, &whir15, &c_lin31, &lin31, &hid_b, 16, &mut rng));
    let idx_field: Vec<Goldilocks> = silu_idx.iter().map(|&i| from_i32(i as i32)).collect();
    let c_idx = commit(&whir15, &idx_field);
    let c_silu = commit(&whir15, &silu1);
    assert!(prove_lookup(&whir15, &c_idx, &idx_field, &whir15, &c_silu, &silu1, &silu_idx, &silu_table, rng.field(), rng.field(), &mut rng));
    let c_outw = commit(&whir22, &out_w);
    let c_outraw = commit(&whir15, &out_raw);
    assert!(prove_matmul(&whir15, &c_silu, &whir22, &c_outw, &whir15, &c_outraw, &silu1, &out_w, &out_raw, SEQ, H_PAD, H_PAD, &mut rng));
    let c_lin32 = commit(&whir15, &lin32);
    assert!(prove_affine(&whir15, &c_outraw, &out_raw, &whir15, &c_lin32, &lin32, &out_b, 16, &mut rng));
    let c_resw = commit(&whir22, &res_w);
    let c_resraw = commit(&whir15, &res_raw);
    assert!(prove_matmul(&whir15, &c_x, &whir22, &c_resw, &whir15, &c_resraw, &x, &res_w, &res_raw, SEQ, H_PAD, H_PAD, &mut rng));
    let c_lin33 = commit(&whir15, &lin33);
    assert!(prove_affine(&whir15, &c_resraw, &res_raw, &whir15, &c_lin33, &lin33, &res_b, 16, &mut rng));
    let c_add31 = commit(&whir15, &add31);
    assert!(prove_add(&whir15, &c_lin32, &whir15, &c_lin33, &whir15, &c_add31, SEQ * H_PAD, &mut rng));
    let c_out = commit(&whir15, &output_ts);
    assert!(prove_scale(&whir15, &c_add31, &add31, &whir15, &c_out, &output_ts, scale_q, &bias_bcast, &mut rng));

    println!("TimesFM 200M: prologue + 20-layer stack + output head verified from WHIR commitments");
}
