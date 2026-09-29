//! Unified op interface: one clear call site per numeric op.
//!
//! Each op computes its output natively, commits the involved tensors through
//! the shared [Ctx], proves the relationship with the underlying WHIR/GKR
//! primitives, and returns the committed output [Tensor]. This is the layer an
//! AI-generated per-model program compiles against.

use std::collections::HashMap;

use crate::committed::{
    affine_raw, commit, prove_add, prove_affine, prove_lookup, prove_matmul, prove_relu,
    prove_scale, scale_raw, layer_norm_raw, rms_norm_raw, prove_layer_norm, prove_rms_norm, prove_softmax_rows, BatchCtx, Committed,
};
use crate::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::fixed_point::{from_i32, to_i32};
use crate::whir::Whir;
use std::sync::Arc;

/// A committed tensor: plain field values plus their WHIR commitment.
pub struct Tensor {
    pub plain: Vec<Goldilocks>,
    pub committed: Committed,
}

impl Tensor {
    pub fn len(&self) -> usize {
        self.plain.len()
    }
    pub fn log2_len(&self) -> usize {
        self.plain.len().trailing_zeros() as usize
    }
}

/// Shared proving context: WHIR instances sized by log2 length, plus the RNG.
pub struct Ctx {
    whirs: HashMap<usize, Whir>,
    pub rng: XorShift64,
}

impl Ctx {
    pub fn new(seed: u64) -> Self {
        Ctx {
            whirs: HashMap::new(),
            rng: XorShift64::new(seed),
        }
    }

    fn ensure_whir(&mut self, log2_len: usize) {
        self.whirs
            .entry(log2_len)
            .or_insert_with(|| Whir::new_testing(log2_len));
    }

    /// Commit raw field values into a [Tensor].
    pub fn commit(&mut self, plain: Vec<Goldilocks>) -> Tensor {
        let log2 = plain.len().trailing_zeros() as usize;
        self.ensure_whir(log2);
        let committed = commit(&self.whirs[&log2], &plain);
        Tensor { plain, committed }
    }
}

/// Collects plain tensors during the forward pass and commits them in
/// (size, group) buckets, so the proving phase can open each tensor by
/// (size, group, table index). The group lets callers separate same-size
/// tensors of different kinds (weights, activations, bit columns).
pub struct BatchBuilder {
    pending: HashMap<(usize, usize), Vec<Arc<Vec<Goldilocks>>>>,
}

impl BatchBuilder {
    pub fn new() -> Self {
        BatchBuilder { pending: HashMap::new() }
    }

    /// Record a plain tensor into group 0; returns (size, group, index).
    pub fn push(&mut self, plain: Vec<Goldilocks>) -> (usize, usize, usize) {
        self.push_group(Arc::new(plain), 0)
    }

    /// Record a plain tensor into an explicit group.
    pub fn push_group(&mut self, plain: Arc<Vec<Goldilocks>>, group: usize) -> (usize, usize, usize) {
        let size = plain.len();
        let bucket = self.pending.entry((size, group)).or_default();
        bucket.push(plain);
        (size, group, bucket.len() - 1)
    }

    /// Commit every collected tensor, grouped by (size, group).
    pub fn commit(&self, whir: &Whir) -> HashMap<(usize, usize), BatchCtx> {
        let mut batches = HashMap::new();
        for (&(size, group), tensors) in &self.pending {
            let refs: Vec<&[Goldilocks]> = tensors.iter().map(|t| t.as_slice()).collect();
            let (commitment, prover_data, protocol, w) = whir.commit_batch(&refs);
            batches.insert(
                (size, group),
                BatchCtx { commitment, prover_data, protocol, whir: w, num_tables: refs.len() },
            );
        }
        batches
    }
}

/// Row-major dense matmul C[m,n] = A[m,k] @ B[k,n] in the field.
pub fn dense_m(a: &[Goldilocks], b: &[Goldilocks], m: usize, k: usize, n: usize) -> Vec<Goldilocks> {
    use p3_maybe_rayon::prelude::*;
    (0..m * n)
        .into_par_iter()
        .map(|idx| {
            let i = idx / n;
            let j = idx % n;
            let mut acc = Goldilocks::ZERO;
            for w in 0..k {
                acc = acc + a[i * k + w] * b[w * n + j];
            }
            acc
        })
        .collect()
}

/// Elementwise add, equal length.
pub fn add_vec(a: &[Goldilocks], b: &[Goldilocks]) -> Vec<Goldilocks> {
    a.iter().zip(b).map(|(&x, &y)| x + y).collect()
}

/// MatMul op: C = A @ B.
pub fn matmul(ctx: &mut Ctx, a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Tensor {
    let c_plain = dense_m(&a.plain, &b.plain, m, k, n);
    let al = a.log2_len();
    let bl = b.log2_len();
    let cl = c_plain.len().trailing_zeros() as usize;
    ctx.ensure_whir(cl);
    let c_committed = commit(&ctx.whirs[&cl], &c_plain);
    let ok = prove_matmul(
        &ctx.whirs[&al],
        &a.committed,
        &ctx.whirs[&bl],
        &b.committed,
        &ctx.whirs[&cl],
        &c_committed,
        &a.plain,
        &b.plain,
        &c_plain,
        m,
        k,
        n,
        &mut ctx.rng,
    );
    assert!(ok);
    Tensor {
        plain: c_plain,
        committed: c_committed,
    }
}

/// Add op: C = A + B, equal length, scale unchanged.
pub fn add(ctx: &mut Ctx, a: &Tensor, b: &Tensor) -> Tensor {
    let c_plain = add_vec(&a.plain, &b.plain);
    let al = a.log2_len();
    let bl = b.log2_len();
    let cl = c_plain.len().trailing_zeros() as usize;
    ctx.ensure_whir(cl);
    let c_committed = commit(&ctx.whirs[&cl], &c_plain);
    let ok = prove_add(
        &ctx.whirs[&al],
        &a.committed,
        &ctx.whirs[&bl],
        &b.committed,
        &ctx.whirs[&cl],
        &c_committed,
        c_plain.len(),
        &mut ctx.rng,
    );
    assert!(ok);
    Tensor {
        plain: c_plain,
        committed: c_committed,
    }
}

/// Affine op: out = round(in / 2^shift) + bias, optional ReLU.
pub fn affine(ctx: &mut Ctx, x: &Tensor, bias: &[Goldilocks], shift: u32, relu: bool) -> Tensor {
    let out_plain = affine_raw(&x.plain, bias, shift, relu);
    let xl = x.log2_len();
    let ol = out_plain.len().trailing_zeros() as usize;
    ctx.ensure_whir(ol);
    let out_committed = commit(&ctx.whirs[&ol], &out_plain);
    let ok = prove_affine(
        &ctx.whirs[&xl],
        &x.committed,
        &x.plain,
        &ctx.whirs[&ol],
        &out_committed,
        &out_plain,
        bias,
        shift,
        &mut ctx.rng,
    );
    assert!(ok);
    Tensor {
        plain: out_plain,
        committed: out_committed,
    }
}

/// Scale op: out = round(in * scale / 2^16) + bias.
pub fn scale(ctx: &mut Ctx, x: &Tensor, scale: i64, bias: &[Goldilocks]) -> Tensor {
    let out_plain = scale_raw(&x.plain, scale, bias);
    let xl = x.log2_len();
    let ol = out_plain.len().trailing_zeros() as usize;
    ctx.ensure_whir(ol);
    let out_committed = commit(&ctx.whirs[&ol], &out_plain);
    let ok = prove_scale(
        &ctx.whirs[&xl],
        &x.committed,
        &x.plain,
        &ctx.whirs[&ol],
        &out_committed,
        &out_plain,
        scale,
        bias,
        &mut ctx.rng,
    );
    assert!(ok);
    Tensor {
        plain: out_plain,
        committed: out_committed,
    }
}

/// ReLU op: out = max(round(in / 2^16) + bias, 0).
pub fn relu(ctx: &mut Ctx, x: &Tensor, bias: &[Goldilocks]) -> Tensor {
    let out_plain = affine_raw(&x.plain, bias, 16, true);
    let xl = x.log2_len();
    let ol = out_plain.len().trailing_zeros() as usize;
    ctx.ensure_whir(ol);
    let out_committed = commit(&ctx.whirs[&ol], &out_plain);
    let ok = prove_relu(
        &ctx.whirs[&xl],
        &x.committed,
        &x.plain,
        &ctx.whirs[&ol],
        &out_committed,
        &out_plain,
        bias,
        &mut ctx.rng,
    );
    assert!(ok);
    Tensor {
        plain: out_plain,
        committed: out_committed,
    }
}

/// Lookup op: out[i] = table[indices[i]] via LogUp.
pub fn lookup(
    ctx: &mut Ctx,
    indices: &[u32],
    outputs: &[Goldilocks],
    table: &[Goldilocks],
) -> Tensor {
    let idx_field: Vec<Goldilocks> = indices.iter().map(|&i| from_i32(i as i32)).collect();
    let il = idx_field.len().trailing_zeros() as usize;
    let ol = outputs.len().trailing_zeros() as usize;
    ctx.ensure_whir(il);
    ctx.ensure_whir(ol);
    let c_idx = commit(&ctx.whirs[&il], &idx_field);
    let c_out = commit(&ctx.whirs[&ol], outputs);
    let alpha = ctx.rng.field();
    let beta = ctx.rng.field();
    let ok = prove_lookup(
        &ctx.whirs[&il],
        &c_idx,
        &idx_field,
        &ctx.whirs[&ol],
        &c_out,
        outputs,
        indices,
        table,
        alpha,
        beta,
        &mut ctx.rng,
    );
    assert!(ok);
    Tensor {
        plain: outputs.to_vec(),
        committed: c_out,
    }
}

/// Round half-up (round .5 up).
fn round_half_up(a: i64, b: i64) -> i64 {
    let q = a.div_euclid(b);
    let r = a.rem_euclid(b);
    if r * 2 >= b {
        q + 1
    } else {
        q
    }
}

/// Row-wise RMSNorm op: per token, raw = x * rstd * w, then norm = round(raw / 2^32).
pub fn rms_norm_rows(
    ctx: &mut Ctx,
    x: &Tensor,
    weight: &[Goldilocks],
    n_real: usize,
    rsqrt: &[Goldilocks],
    n_rows: usize,
) -> Tensor {
    let h = x.len() / n_rows;
    let zero = vec![Goldilocks::ZERO; h];
    let mut raw = Vec::with_capacity(x.len());
    let mut norm = Vec::with_capacity(x.len());
    for s in 0..n_rows {
        let row = &x.plain[s * h..(s + 1) * h];
        let (r, _, _) = rms_norm_raw(row, weight, n_real, rsqrt);
        let n = affine_raw(&r, &zero, 32, false);
        raw.extend_from_slice(&r);
        norm.extend_from_slice(&n);
    }
    let hl = h.trailing_zeros() as usize;
    ctx.ensure_whir(hl);
    let alpha = ctx.rng.field();
    let beta = ctx.rng.field();
    for s in 0..n_rows {
        let xr = &x.plain[s * h..(s + 1) * h];
        let rr = &raw[s * h..(s + 1) * h];
        let nr = &norm[s * h..(s + 1) * h];
        let cx = commit(&ctx.whirs[&hl], xr);
        let cr = commit(&ctx.whirs[&hl], rr);
        let cn = commit(&ctx.whirs[&hl], nr);
        assert!(prove_rms_norm(
            &ctx.whirs[&hl], &cx, xr, &ctx.whirs[&hl], &cr, rr, weight, n_real, rsqrt,
            alpha, beta, &mut ctx.rng,
        ));
        assert!(prove_affine(
            &ctx.whirs[&hl], &cr, rr, &ctx.whirs[&hl], &cn, nr, &zero, 32, &mut ctx.rng,
        ));
    }
    ctx.commit(norm)
}

/// Row-wise LayerNorm op: per token, raw = (x - mean) * rstd * w, then
/// out = round(raw / 2^32) + bias.
pub fn layer_norm_rows(
    ctx: &mut Ctx,
    x: &Tensor,
    weight: &[Goldilocks],
    bias: &[Goldilocks],
    n_real: usize,
    rsqrt: &[Goldilocks],
    n_rows: usize,
) -> Tensor {
    let h = x.len() / n_rows;
    let mut raw = Vec::with_capacity(x.len());
    let mut out = Vec::with_capacity(x.len());
    for s in 0..n_rows {
        let row = &x.plain[s * h..(s + 1) * h];
        let (r, _, _, _) = layer_norm_raw(row, weight, n_real, rsqrt);
        let o = affine_raw(&r, bias, 32, false);
        raw.extend_from_slice(&r);
        out.extend_from_slice(&o);
    }
    let hl = h.trailing_zeros() as usize;
    ctx.ensure_whir(hl);
    let alpha = ctx.rng.field();
    let beta = ctx.rng.field();
    for s in 0..n_rows {
        let xr = &x.plain[s * h..(s + 1) * h];
        let rr = &raw[s * h..(s + 1) * h];
        let or = &out[s * h..(s + 1) * h];
        let cx = commit(&ctx.whirs[&hl], xr);
        let cr = commit(&ctx.whirs[&hl], rr);
        let co = commit(&ctx.whirs[&hl], or);
        assert!(prove_layer_norm(
            &ctx.whirs[&hl], &cx, xr, &ctx.whirs[&hl], &cr, rr, weight, n_real, rsqrt,
            alpha, beta, &mut ctx.rng,
        ));
        assert!(prove_affine(
            &ctx.whirs[&hl], &cr, rr, &ctx.whirs[&hl], &co, or, bias, 32, &mut ctx.rng,
        ));
    }
    ctx.commit(out)
}

/// Row-wise Softmax op: out = exp(shifted) / row_sum, numerically stable.
pub fn softmax_rows(
    ctx: &mut Ctx,
    scores: &Tensor,
    exp_table: &[Goldilocks],
    offset: u32,
    n_rows: usize,
    n_cols: usize,
) -> Tensor {
    let n = n_rows * n_cols;
    let scores_plain = &scores.plain;
    let c_plain: Vec<Goldilocks> = (0..n_rows)
        .map(|r| {
            from_i32(
                (0..n_cols)
                    .map(|k| to_i32(scores_plain[r * n_cols + k]))
                    .max()
                    .unwrap(),
            )
        })
        .collect();
    let shifted_plain: Vec<Goldilocks> = scores_plain
        .iter()
        .enumerate()
        .map(|(i, &s)| s - c_plain[i / n_cols])
        .collect();
    let e_plain: Vec<Goldilocks> = shifted_plain
        .iter()
        .map(|&s| {
            let idx =
                (to_i32(s) as i64 + offset as i64).clamp(0, exp_table.len() as i64 - 1) as usize;
            exp_table[idx]
        })
        .collect();
    let sum_plain: Vec<Goldilocks> = (0..n_rows)
        .map(|r| {
            (0..n_cols).fold(Goldilocks::ZERO, |a, k| a + e_plain[r * n_cols + k])
        })
        .collect();
    let sum_broadcast_plain: Vec<Goldilocks> =
        (0..n).map(|i| sum_plain[i / n_cols]).collect();
    let out_plain: Vec<Goldilocks> = e_plain
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            from_i32(round_half_up(
                to_i32(v) as i64 * 65536,
                to_i32(sum_plain[i / n_cols]) as i64,
            ) as i32)
        })
        .collect();

    let c = ctx.commit(c_plain);
    let shifted = ctx.commit(shifted_plain);
    let e = ctx.commit(e_plain);
    let sum = ctx.commit(sum_plain);
    let sum_broadcast = ctx.commit(sum_broadcast_plain);
    let out = ctx.commit(out_plain);

    let alpha = ctx.rng.field();
    let beta = ctx.rng.field();
    let ok = prove_softmax_rows(
        &ctx.whirs[&scores.log2_len()],
        &ctx.whirs[&c.log2_len()],
        &scores.committed,
        scores_plain,
        &c.committed,
        &c.plain,
        &shifted.committed,
        &shifted.plain,
        &e.committed,
        &e.plain,
        &sum.committed,
        &sum.plain,
        &sum_broadcast.committed,
        &sum_broadcast.plain,
        &out.committed,
        &out.plain,
        exp_table,
        offset,
        n_rows,
        n_cols,
        alpha,
        beta,
        &mut ctx.rng,
    );
    assert!(ok);
    out
}
