//! A full TimesFM decoder layer whose matmul chain is cryptographically bound.
//!
//! Every 64-element tensor (activations + matmul weights) is committed once via
//! WHIR; each matmul is a GKR sum-check whose two terminal evaluations and its
//! claimed output are reproduced from WHIR prescribed-point openings, so the
//! verifier trusts commitments + openings, not raw evaluations. The upstream
//! matmul's output commitment *is* the downstream matmul's input commitment
//! (cross-layer binding). Nonlinearities (RMSNorm/softmax/GELU) keep their LogUp
//! proofs; binding their input/output commitments is the next step.

use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing, PrimeField64, XorShift64};
use zkie_gkr::whir::{Commitment, ProverData, OpeningProtocol, Whir};
use zkie_gkr::{gelu, matmul, mle, rmsnorm, softmax};

/// A committed tensor (commitment + prover data + opening protocol).
struct Ct {
    commitment: Commitment,
    pd: ProverData,
    proto: OpeningProtocol,
}

fn commit(whir: &Whir, m: &[Goldilocks]) -> Ct {
    let (commitment, pd, proto) = whir.commit(m);
    Ct {
        commitment,
        pd,
        proto,
    }
}

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

fn sample(rng: &mut XorShift64, bits: usize) -> Vec<Goldilocks> {
    (0..bits).map(|_| rng.field()).collect()
}

fn cat(a: &[Goldilocks], b: &[Goldilocks]) -> Vec<Goldilocks> {
    let mut v = a.to_vec();
    v.extend_from_slice(b);
    v
}

/// Prove `C = A @ B` and bind it to the three committed tensors via WHIR
/// prescribed-point openings.
#[allow(clippy::too_many_arguments)]
fn committed_matmul(
    whir: &Whir,
    a: &Ct,
    b: &Ct,
    c: &Ct,
    a_mat: &[Goldilocks],
    b_mat: &[Goldilocks],
    c_mat: &[Goldilocks],
    m: usize,
    k: usize,
    n: usize,
    b_transposed: bool,
    rng: &mut XorShift64,
) -> bool {
    let u = sample(rng, m.trailing_zeros() as usize);
    let v = sample(rng, n.trailing_zeros() as usize);
    let ch = sample(rng, k.trailing_zeros() as usize);
    let at = transpose(a_mat, m, k);
    let b_prove = if b_transposed { transpose(b_mat, k, n) } else { b_mat.to_vec() };
    let proof = matmul::prove(&at, &b_prove, c_mat, m, k, n, &u, &v, &ch);

    // A and (optionally) B are committed in normal form; a transpose is a
    // coordinate swap for square matrices.
    let ap = cat(&ch, &u);
    let (a_open, f) = whir.open(a.pd.clone(), &a.proto, &ap);
    let bp = if b_transposed { cat(&ch, &v) } else { cat(&v, &ch) };
    let (b_open, h) = whir.open(b.pd.clone(), &b.proto, &bp);
    let cp = cat(&v, &u);
    let (c_open, claimed) = whir.open(c.pd.clone(), &c.proto, &cp);

    let f_ok = whir.verify(&a.commitment, &a_open, &a.proto, &ap).unwrap() == f;
    let h_ok = whir.verify(&b.commitment, &b_open, &b.proto, &bp).unwrap() == h;
    let c_ok = whir.verify(&c.commitment, &c_open, &c.proto, &cp).unwrap() == claimed;
    let evals_ok = f == mle::eval(a_mat, &ap)
        && h == mle::eval(b_mat, &bp)
        && claimed == mle::eval(c_mat, &cp);
    f_ok && h_ok && c_ok && evals_ok && claimed == proof.claimed && matmul::verify(&proof, &ch, f, h)
}

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
fn timesfm_layer_matmul_chain_is_whir_bound() {
    let mut rng = XorShift64::new(0x8f5d);
    let (seq, hidden) = (8usize, 8usize);
    let table_size = 64usize;
    let whir = Whir::new_testing(6); // every 8x8 tensor is 2^6 evaluations

    // Weights.
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

    // input_layernorm.
    let x_norm = rms_norm_batched(&x, &w_rms1, seq, hidden, eps, &rsqrt_table, &mut rng);

    // Commit every 64-element tensor once.
    let c_x_norm = commit(&whir, &x_norm);
    let c_wq = commit(&whir, &w_q);
    let c_wk = commit(&whir, &w_k);
    let c_wv = commit(&whir, &w_v);
    let c_wf1 = commit(&whir, &w_f1);
    let c_wf2 = commit(&whir, &w_f2);

    // QKV projection.
    let q = dense(&x_norm, &w_q, seq, hidden, hidden);
    let k = dense(&x_norm, &w_k, seq, hidden, hidden);
    let v = dense(&x_norm, &w_v, seq, hidden, hidden);
    let c_q = commit(&whir, &q);
    let c_k = commit(&whir, &k);
    let c_v = commit(&whir, &v);
    assert!(committed_matmul(&whir, &c_x_norm, &c_wq, &c_q, &x_norm, &w_q, &q, seq, hidden, hidden, false, &mut rng));
    assert!(committed_matmul(&whir, &c_x_norm, &c_wk, &c_k, &x_norm, &w_k, &k, seq, hidden, hidden, false, &mut rng));
    assert!(committed_matmul(&whir, &c_x_norm, &c_wv, &c_v, &x_norm, &w_v, &v, seq, hidden, hidden, false, &mut rng));

    // Attention scores + softmax.
    let s = dense(&q, &transpose(&k, seq, hidden), seq, hidden, seq);
    let c_s = commit(&whir, &s);
    assert!(committed_matmul(&whir, &c_q, &c_k, &c_s, &q, &k, &s, seq, hidden, seq, true, &mut rng));
    let s_indices: Vec<u32> = s.iter().map(|z| (z.as_canonical_u64() % table_size as u64) as u32).collect();
    let (p, softmax_proof) = softmax::softmax(&s_indices, &exp_table, rng.field(), rng.field());
    assert!(softmax::verify(&s_indices, &exp_table, &p, &softmax_proof));
    let c_p = commit(&whir, &p);

    // Output projection.
    let o = dense(&p, &v, seq, seq, hidden);
    let c_o = commit(&whir, &o);
    assert!(committed_matmul(&whir, &c_p, &c_v, &c_o, &p, &v, &o, seq, seq, hidden, false, &mut rng));

    // pre_ffn_layernorm.
    let o_norm = rms_norm_batched(&o, &w_rms2, seq, hidden, eps, &rsqrt_table, &mut rng);
    let c_o_norm = commit(&whir, &o_norm);

    // FFN.
    let f1 = dense(&o_norm, &w_f1, seq, hidden, hidden);
    let c_f1 = commit(&whir, &f1);
    assert!(committed_matmul(&whir, &c_o_norm, &c_wf1, &c_f1, &o_norm, &w_f1, &f1, seq, hidden, hidden, false, &mut rng));
    let f1_indices: Vec<u32> = f1.iter().map(|z| (z.as_canonical_u64() % table_size as u64) as u32).collect();
    let (g, gelu_proof) = gelu::gelu(&f1_indices, &gelu_table, rng.field(), rng.field());
    assert!(gelu::verify(&f1_indices, &gelu_table, &g, &gelu_proof));
    let c_g = commit(&whir, &g);
    let f2 = dense(&g, &w_f2, seq, hidden, hidden);
    let c_f2 = commit(&whir, &f2);
    assert!(committed_matmul(&whir, &c_g, &c_wf2, &c_f2, &g, &w_f2, &f2, seq, hidden, hidden, false, &mut rng));
}
