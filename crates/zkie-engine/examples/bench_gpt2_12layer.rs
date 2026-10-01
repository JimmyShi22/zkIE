use std::time::Instant;
use zkie_core::common::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_core::common::logup_gkr::prove_lookup_fractional;
use zkie_core::common::matmul;
use zkie_ops::projection::prove_projection;
use zkie_ops::softmax_scaled::prove_softmax_scaled;
use zkie_core::common::sumcheck::prove_virtual;

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
) -> (Vec<Goldilocks>, Vec<Goldilocks>) {
    let h = mm(x, w, m, k, n);
    let out: Vec<Goldilocks> = (0..m * n)
        .map(|ij| from_i64(round_div(to_i64(h[ij]), 1i64 << shift) + to_i64(b[ij])))
        .collect();
    let rem: Vec<Goldilocks> = (0..m * n)
        .map(|ij| {
            from_i64(
                to_i64(h[ij]) - (to_i64(out[ij]) - to_i64(b[ij])) * (1i64 << shift)
                    + (1i64 << (shift - 1)),
            )
        })
        .collect();
    (out, rem)
}

struct LayerWeights {
    wq: Vec<Goldilocks>,
    wk: Vec<Goldilocks>,
    wv: Vec<Goldilocks>,
    wo: Vec<Goldilocks>,
    wfc: Vec<Goldilocks>,
    wproj: Vec<Goldilocks>,
    bd: Vec<Goldilocks>,
    bffn: Vec<Goldilocks>,
}

fn prove_layer(
    x: &[Goldilocks],
    lw: &LayerWeights,
    exp_table: &[Goldilocks],
    gelu_table: &[Goldilocks],
    m: usize,
    d: usize,
    ffn: usize,
    shift: u32,
    rng: &mut XorShift64,
) -> Vec<Goldilocks> {
    let (q, q_rem) = projection_fp(x, &lw.wq, &lw.bd, m, d, d, shift);
    let (k, k_rem) = projection_fp(x, &lw.wk, &lw.bd, m, d, d, shift);
    let (v, v_rem) = projection_fp(x, &lw.wv, &lw.bd, m, d, d, shift);
    let kt = transpose(&k, m, d);
    let scores = mm(&q, &kt, m, d, m);
    let table_len = exp_table.len();
    let indices: Vec<u32> = scores
        .iter()
        .map(|&v| ((to_i64(v).max(0)) as u64 % table_len as u64) as u32)
        .collect();
    let e: Vec<Goldilocks> = indices.iter().map(|&i| exp_table[i as usize]).collect();
    let sum: Vec<Goldilocks> = (0..m).map(|i| (0..m).fold(Goldilocks::ZERO, |a, j| a + e[i * m + j])).collect();
    let probs: Vec<Goldilocks> = (0..m * m).map(|ij| e[ij] * sum[ij / m].inverse()).collect();
    let attn = mm(&probs, &v, m, m, d);
    let (o, o_rem) = projection_fp(&attn, &lw.wo, &lw.bd, m, d, d, shift);
    let y: Vec<Goldilocks> = (0..m * d).map(|i| x[i] + o[i]).collect();
    let (fc, fc_rem) = projection_fp(&y, &lw.wfc, &lw.bffn, m, d, ffn, shift);
    let gelu_idx: Vec<u32> = fc.iter().map(|&v| ((to_i64(v).max(0)) as u64 % 64) as u32).collect();
    let act: Vec<Goldilocks> = gelu_idx.iter().map(|&i| gelu_table[i as usize]).collect();
    let (proj, proj_rem) = projection_fp(&act, &lw.wproj, &lw.bd, m, ffn, d, shift);
    let out: Vec<Goldilocks> = (0..m * d).map(|i| y[i] + proj[i]).collect();

    let _ = prove_projection(x, &lw.wq, &lw.bd, &q, &q_rem, m, d, d, shift, rng);
    let _ = prove_projection(x, &lw.wk, &lw.bd, &k, &k_rem, m, d, d, shift, rng);
    let _ = prove_projection(x, &lw.wv, &lw.bd, &v, &v_rem, m, d, d, shift, rng);
    let qt = transpose(&q, m, d);
    let u: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let vs: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let ch: Vec<Goldilocks> = (0..d.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let _ = matmul::prove(&qt, &kt, &scores, m, d, m, &u, &vs, &ch);
    let _ = prove_softmax_scaled(&indices, &e, &probs, exp_table, m, m, rng);
    let pt = transpose(&probs, m, m);
    let ch_v: Vec<Goldilocks> = (0..d.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let ch_m: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let _ = matmul::prove(&pt, &v, &attn, m, m, d, &u, &ch_v, &ch_m);
    let _ = prove_projection(&attn, &lw.wo, &lw.bd, &o, &o_rem, m, d, d, shift, rng);
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let terms = vec![(Goldilocks::ONE, vec![2usize]), (neg, vec![0usize]), (neg, vec![1usize])];
    let r_add1: Vec<Goldilocks> = (0..(m * d).trailing_zeros() as usize).map(|_| rng.field()).collect();
    let mles: Vec<&[Goldilocks]> = vec![x, &o, &y];
    let _ = prove_virtual(&mles, &terms, Goldilocks::ZERO, &r_add1);
    let _ = prove_projection(&y, &lw.wfc, &lw.bffn, &fc, &fc_rem, m, d, ffn, shift, rng);
    let _ = prove_lookup_fractional(&gelu_idx, &act, gelu_table, rng.field(), rng.field(), rng);
    let _ = prove_projection(&act, &lw.wproj, &lw.bd, &proj, &proj_rem, m, ffn, d, shift, rng);
    let r_add2: Vec<Goldilocks> = (0..(m * d).trailing_zeros() as usize).map(|_| rng.field()).collect();
    let mles2: Vec<&[Goldilocks]> = vec![&y, &proj, &out];
    let _ = prove_virtual(&mles2, &terms, Goldilocks::ZERO, &r_add2);

    let _ = (q_rem, k_rem, v_rem, o_rem, fc_rem, proj_rem, scores, attn);
    out
}

fn main() {
    let (m, d, ffn) = (512usize, 1024usize, 4096usize);
    let shift = 16u32;
    let layers = 12usize;
    let mut rng = XorShift64::new(0x7A11);
    let x: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let lw = LayerWeights {
        wq: (0..d * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect(),
        wk: (0..d * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect(),
        wv: (0..d * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect(),
        wo: (0..d * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect(),
        wfc: (0..d * ffn).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect(),
        wproj: (0..ffn * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect(),
        bd: (0..m * d).map(|_| from_i64((rng.next_u64() % 20) as i64 - 10)).collect(),
        bffn: (0..m * ffn).map(|_| from_i64((rng.next_u64() % 20) as i64 - 10)).collect(),
    };
    let exp_table: Vec<Goldilocks> = (0..(1usize << 18)).map(|_| rng.field()).collect();
    let gelu_table: Vec<Goldilocks> = (0..64).map(|j| from_i64((j as i64).pow(2) % 1000)).collect();

    let mut x_cur = x;
    let t0 = Instant::now();
    for li in 0..layers {
        x_cur = prove_layer(&x_cur, &lw, &exp_table, &gelu_table, m, d, ffn, shift, &mut rng);
        eprintln!("layer {} proved at {:.2}s", li, t0.elapsed().as_secs_f64());
    }
    let total = t0.elapsed();
    println!(
        "gpt2-12layer m={} d={} ffn={} shift={} layers={} total={:.2}s (plain, single-head, no layernorm)",
        m,
        d,
        ffn,
        shift,
        layers,
        total.as_secs_f64()
    );
    let _ = x_cur;
}
