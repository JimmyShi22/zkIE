//! A tiny single-head attention block composed out of the GKR matmul and the
//! logUp softmax, to show the primitives compose into an end-to-end layer:
//!
//!     S = Q @ K^T        (matmul, sum-check)
//!     P = softmax(S)     (exp lookup, logUp)
//!     O = P @ V          (matmul, sum-check)

use zkie_gkr::field::{F64, XorShift64};
use zkie_gkr::{matmul, mle, softmax};

fn dense(a: &[F64], b: &[F64], m: usize, k: usize, n: usize) -> Vec<F64> {
    let mut c = vec![F64::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = F64::ZERO;
            for w in 0..k {
                acc = acc + a[i * k + w] * b[w * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn transpose(a: &[F64], m: usize, k: usize) -> Vec<F64> {
    let mut at = vec![F64::ZERO; k * m];
    for i in 0..m {
        for w in 0..k {
            at[w * m + i] = a[i * k + w];
        }
    }
    at
}

fn main() {
    // seq = 4, head_dim = 4, exp table = 64 entries.
    let (seq, dim) = (4usize, 4usize);
    let table_size = 64usize;
    let mut rng = XorShift64::new(0xadd);

    let q: Vec<F64> = (0..seq * dim).map(|_| rng.field()).collect();
    let k: Vec<F64> = (0..seq * dim).map(|_| rng.field()).collect();
    let v: Vec<F64> = (0..seq * dim).map(|_| rng.field()).collect();
    let table: Vec<F64> = (0..table_size).map(|_| rng.field()).collect();

    // S = Q @ K^T.
    let kt = transpose(&k, seq, dim);
    let s = dense(&q, &kt, seq, dim, seq);
    let s_indices: Vec<u32> = s.iter().map(|x| (x.val() % table_size as u64) as u32).collect();

    // P = softmax(S) (per row, flattened here for brevity).
    let (p, softmax_proof) = softmax::softmax(&s_indices, &table, rng.field(), rng.field());

    // O = P @ V.
    let o = dense(&p, &v, seq, seq, dim);

    // Prove both matmuls with the GKR sum-check.
    let (s_at, s_kt) = (transpose(&q, seq, dim), kt);
    let s_u: Vec<F64> = (0..seq.trailing_zeros()).map(|_| rng.field()).collect();
    let s_v: Vec<F64> = (0..seq.trailing_zeros()).map(|_| rng.field()).collect();
    let s_ch: Vec<F64> = (0..dim.trailing_zeros()).map(|_| rng.field()).collect();
    let s_proof = matmul::prove(&s_at, &s_kt, &s, seq, dim, seq, &s_u, &s_v, &s_ch);

    let o_at = transpose(&p, seq, seq);
    let o_u: Vec<F64> = (0..seq.trailing_zeros()).map(|_| rng.field()).collect();
    let o_v: Vec<F64> = (0..dim.trailing_zeros()).map(|_| rng.field()).collect();
    let o_ch: Vec<F64> = (0..seq.trailing_zeros()).map(|_| rng.field()).collect();
    let o_proof = matmul::prove(&o_at, &v, &o, seq, seq, dim, &o_u, &o_v, &o_ch);

    // Independent openings.
    let mut sf = s_u.clone();
    sf.extend_from_slice(&s_ch);
    let mut sh = s_v.clone();
    sh.extend_from_slice(&s_ch);
    let s_ok = matmul::verify(&s_proof, &s_ch, mle::eval(&s_at, &sf), mle::eval(&s_kt, &sh));

    let mut of = o_u.clone();
    of.extend_from_slice(&o_ch);
    let mut oh = o_v.clone();
    oh.extend_from_slice(&o_ch);
    let o_ok = matmul::verify(&o_proof, &o_ch, mle::eval(&o_at, &of), mle::eval(&v, &oh));

    let softmax_ok = softmax::verify(&s_indices, &table, &p, &softmax_proof);
    assert!(s_ok && o_ok && softmax_ok);
    println!("attention seq={seq} dim={dim} -> matmul(2) + softmax(1) all verified");
}
