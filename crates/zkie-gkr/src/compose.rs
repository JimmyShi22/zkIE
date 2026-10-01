//! Generic linear shard composer: fold a contiguous run of op primitives into
//! ONE `g` (one set of sumchecks + `same_poly` bindings). The number of ops per
//! shard is a public parameter (the autotuning knob), NOT hardcoded to a layer.
//!
//! First slice: a linear chain of `projection` blocks (matmul + affine + round),
//! no branching. Step `i` proves `y_i = round(y_{i-1} @ W_i / 2^shift_i) + b_i`,
//! with every intermediate `y_1..y_{N-1}` virtual (never committed) and bound
//! between adjacent projections via `same_poly`. A shard of `N` projections is
//! the degenerate "whole model folded into one g" when `N` is the full depth, and
//! the "one op per shard" case when `N == 1` (no internal binding).

use crate::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use rayon::prelude::*;
use crate::fixed_point::{from_i64, to_i64};
use crate::matmul::{prove as matmul_prove, verify as matmul_verify, MatmulProof};
use crate::mle;
use crate::projection::{prove_projection, verify_projection, ProjectionProof};
use crate::same_poly::{prove_same_poly, verify_same_poly, SamePolyProof};
use crate::sumcheck::{prove_virtual, verify_virtual, VirtualProof};
use crate::logup_gkr::{prove_lookup_fractional, verify_lookup_fractional, FractionalProof};
use crate::softmax_scaled::{prove_softmax_rounded, verify_softmax_rounded, SoftmaxRoundedProof};
use crate::layernorm_chain::{prove_layernorm_chain, verify_layernorm_chain, LayernormChainProof};
use crate::layer_norm_centered::{
    layer_norm_forward as layer_norm_centered_forward,
    prove_layer_norm_centered, verify_layer_norm_centered, LayerNormCenteredProof,
};

fn transpose(a: &[Goldilocks], m: usize, k: usize) -> Vec<Goldilocks> {
    let mut t = vec![Goldilocks::ZERO; k * m];
    for i in 0..m {
        for kk in 0..k {
            t[kk * m + i] = a[i * k + kk];
        }
    }
    t
}

/// One projection step's shape + weights. `w` is `k x n`, `bias` is `m x n`.
#[derive(Clone, Debug)]
pub struct ProjectionStep {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub shift: u32,
    pub w: Vec<Goldilocks>,
    pub bias: Vec<Goldilocks>,
}

impl ProjectionStep {
    pub fn new(
        m: usize,
        k: usize,
        n: usize,
        shift: u32,
        w: Vec<Goldilocks>,
        bias: Vec<Goldilocks>,
    ) -> Self {
        assert_eq!(w.len(), k * n, "weight shape k*n");
        assert_eq!(bias.len(), m * n, "bias shape m*n");
        ProjectionStep { m, k, n, shift, w, bias }
    }
}

/// A chain of `N` projections folded into one g.
pub struct ProjectionChainProof {
    pub steps: Vec<ProjectionProof>,
    /// `binds[i]` links `steps[i]`'s output to `steps[i+1]`'s input (same tensor,
    /// two different claim points) for `i in 0..steps.len()-1`.
    pub binds: Vec<SamePolyProof>,
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

fn projection_fwd(
    x: &[Goldilocks],
    step: &ProjectionStep,
) -> (Vec<Goldilocks>, Vec<Goldilocks>) {
    let h = crate::par::mm_par(x, &step.w, step.m, step.k, step.n, 64);
    let out: Vec<Goldilocks> = (0..step.m * step.n)
        .map(|ij| from_i64(round_div(to_i64(h[ij]), 1i64 << step.shift) + to_i64(step.bias[ij])))
        .collect();
    let rem: Vec<Goldilocks> = (0..step.m * step.n)
        .map(|ij| {
            from_i64(
                to_i64(h[ij]) - (to_i64(out[ij]) - to_i64(step.bias[ij])) * (1i64 << step.shift)
                    + (1i64 << (step.shift - 1)),
            )
        })
        .collect();
    (out, rem)
}

/// Run the whole chain forward; returns `ys[1..=N]` (the per-step outputs) and
/// `rems[0..N]` (each step's rounding remainder). `ys[0]` is the caller's `x`.
fn forward(
    x: &[Goldilocks],
    steps: &[ProjectionStep],
) -> (Vec<Vec<Goldilocks>>, Vec<Vec<Goldilocks>>) {
    let mut cur = x.to_vec();
    let mut ys = Vec::with_capacity(steps.len());
    let mut rems = Vec::with_capacity(steps.len());
    for step in steps {
        assert_eq!(cur.len(), step.m * step.k, "chain shape mismatch");
        let (out, rem) = projection_fwd(&cur, step);
        rems.push(rem);
        cur = out.clone();
        ys.push(out);
    }
    (ys, rems)
}

/// Prove a chain of `steps.len()` projections as one g. `x` is the shard input
/// (shape `steps[0].m x steps[0].k`).
pub fn prove_projection_chain(
    x: &[Goldilocks],
    steps: &[ProjectionStep],
    rng: &mut XorShift64,
) -> ProjectionChainProof {
    assert!(!steps.is_empty(), "chain needs at least one projection");
    let (ys, rems) = forward(x, steps);

    let mut step_proofs = Vec::with_capacity(steps.len());
    let mut prev = x;
    for (i, step) in steps.iter().enumerate() {
        let p = prove_projection(
            prev,
            &step.w,
            &step.bias,
            &ys[i],
            &rems[i],
            step.m,
            step.k,
            step.n,
            step.shift,
            rng,
        );
        step_proofs.push(p);
        prev = &ys[i];
    }

    let mut binds = Vec::with_capacity(steps.len().saturating_sub(1));
    for i in 1..steps.len() {
        // y_i: output of step i-1 (claimed at step_proofs[i-1].pt) and input of
        // step i (claimed at ch_i ++ u_i).
        let out_pt = step_proofs[i - 1].pt.clone();
        let mut in_pt = step_proofs[i].ch.clone();
        in_pt.extend_from_slice(&step_proofs[i].u);
        let claims = vec![
            (out_pt.clone(), mle::eval(&ys[i], &out_pt)),
            (in_pt.clone(), mle::eval(&ys[i], &in_pt)),
        ];
        binds.push(prove_same_poly(&ys[i], &claims, rng));
    }

    ProjectionChainProof { steps: step_proofs, binds }
}

/// Verify a chain proof. Returns true iff every projection is sound AND every
/// adjacent intermediate binding is consistent.
pub fn verify_projection_chain(
    proof: &ProjectionChainProof,
    x: &[Goldilocks],
    steps: &[ProjectionStep],
) -> bool {
    assert_eq!(proof.steps.len(), steps.len());
    assert_eq!(proof.binds.len(), steps.len().saturating_sub(1));
    let (ys, rems) = forward(x, steps);

    let mut prev = x;
    for (i, step) in steps.iter().enumerate() {
        if !verify_projection(
            &proof.steps[i],
            prev,
            &step.w,
            &step.bias,
            &ys[i],
            &rems[i],
            step.m,
            step.k,
            step.n,
            step.shift,
        ) {
            return false;
        }
        prev = &ys[i];
    }

    for i in 1..steps.len() {
        let out_pt = proof.steps[i - 1].pt.clone();
        let mut in_pt = proof.steps[i].ch.clone();
        in_pt.extend_from_slice(&proof.steps[i].u);
        let claims = vec![
            (out_pt.clone(), mle::eval(&ys[i], &out_pt)),
            (in_pt.clone(), mle::eval(&ys[i], &in_pt)),
        ];
        if verify_same_poly(&proof.binds[i - 1], &ys[i], &claims).is_none() {
            return false;
        }
    }
    true
}

/// A sharded projection chain: split a linear chain of projections into shards
/// of `ops_per_shard` ops, prove each shard independently (folding its ops into
/// one g), and `same_poly`-bind every cross-shard boundary tensor. This is the
/// "shard granularity drives the proof" mechanism: `ops_per_shard` is a public
/// parameter, and the cross-shard bindings enforce composition soundness.
pub struct ProjectionShardedProof {
    pub shards: Vec<ProjectionChainProof>,
    pub cross_binds: Vec<SamePolyProof>,
    /// Global step index of each cross-shard boundary tensor `ys[b]`.
    pub boundary_steps: Vec<usize>,
}

pub fn prove_projection_sharded(
    x: &[Goldilocks],
    steps: &[ProjectionStep],
    ops_per_shard: usize,
    rng: &mut XorShift64,
) -> ProjectionShardedProof {
    assert!(!steps.is_empty(), "chain needs at least one projection");
    let ops_per_shard = ops_per_shard.max(1);
    let (ys, _) = forward(x, steps);

    let mut shards = Vec::new();
    let mut boundary_steps = Vec::new();
    let mut start = 0;
    while start < steps.len() {
        let end = (start + ops_per_shard).min(steps.len());
        let input: &[Goldilocks] = if start == 0 { x } else { &ys[start - 1] };
        shards.push(prove_projection_chain(input, &steps[start..end], rng));
        start = end;
        if start < steps.len() {
            boundary_steps.push(start - 1);
        }
    }

    let mut cross_binds = Vec::with_capacity(boundary_steps.len());
    for &b in &boundary_steps {
        let prev_shard = b / ops_per_shard;
        let next_shard = prev_shard + 1;
        let prev = shards[prev_shard].steps.last().unwrap();
        let next = shards[next_shard].steps.first().unwrap();
        let out_pt = prev.pt.clone();
        let mut in_pt = next.ch.clone();
        in_pt.extend_from_slice(&next.u);
        let claims = vec![
            (out_pt.clone(), mle::eval(&ys[b], &out_pt)),
            (in_pt.clone(), mle::eval(&ys[b], &in_pt)),
        ];
        cross_binds.push(prove_same_poly(&ys[b], &claims, rng));
    }

    ProjectionShardedProof { shards, cross_binds, boundary_steps }
}

pub fn verify_projection_sharded(
    proof: &ProjectionShardedProof,
    x: &[Goldilocks],
    steps: &[ProjectionStep],
    ops_per_shard: usize,
) -> bool {
    let ops_per_shard = ops_per_shard.max(1);
    let (ys, _) = forward(x, steps);

    let mut start = 0;
    let mut si = 0;
    while start < steps.len() {
        let end = (start + ops_per_shard).min(steps.len());
        let input: &[Goldilocks] = if start == 0 { x } else { &ys[start - 1] };
        if !verify_projection_chain(&proof.shards[si], input, &steps[start..end]) {
            return false;
        }
        si += 1;
        start = end;
    }

    if proof.cross_binds.len() != proof.boundary_steps.len() {
        return false;
    }
    for (ci, &b) in proof.boundary_steps.iter().enumerate() {
        let prev_shard = b / ops_per_shard;
        let next_shard = prev_shard + 1;
        let prev = proof.shards[prev_shard].steps.last().unwrap();
        let next = proof.shards[next_shard].steps.first().unwrap();
        let out_pt = prev.pt.clone();
        let mut in_pt = next.ch.clone();
        in_pt.extend_from_slice(&next.u);
        let claims = vec![
            (out_pt.clone(), mle::eval(&ys[b], &out_pt)),
            (in_pt.clone(), mle::eval(&ys[b], &in_pt)),
        ];
        if verify_same_poly(&proof.cross_binds[ci], &ys[b], &claims).is_none() {
            return false;
        }
    }
    true
}

/// A chain of residual projection blocks, folded into one g:
/// `p_i = projection(x_{i-1})`, `x_i = x_{i-1} + p_i`. This is the first
/// *branching* shape: each `x_{i-1}` is consumed by both the projection and the
/// residual add, and each `p_i` is produced by the projection and consumed by
/// the add — both bound via `same_poly`. A shard of `N` blocks is `2N` ops folded
/// into one g with every intermediate `x_1..x_{N-1}` virtual.
pub struct ResidualChainProof {
    pub proj: Vec<ProjectionProof>,
    /// `adds[i]` is the `(proof, r)` for `x_i = x_{i-1} + p_i`.
    pub adds: Vec<(VirtualProof, Vec<Goldilocks>)>,
    /// `same_x[i]` binds `x_i` across its consumers (projection in / residual in /
    /// previous residual out), for `i in 0..N-1` (the final `x_N` is the boundary).
    pub same_x: Vec<SamePolyProof>,
    /// `same_p[i]` binds `p_i` across projection out and residual in.
    pub same_p: Vec<SamePolyProof>,
}

fn residual_forward(
    x: &[Goldilocks],
    steps: &[ProjectionStep],
) -> (Vec<Vec<Goldilocks>>, Vec<Vec<Goldilocks>>, Vec<Vec<Goldilocks>>) {
    let mut cur = x.to_vec();
    let mut xs = vec![x.to_vec()];
    let mut ps = Vec::with_capacity(steps.len());
    let mut rems = Vec::with_capacity(steps.len());
    for step in steps {
        assert_eq!(cur.len(), step.m * step.k, "chain shape mismatch");
        let (p, rem) = projection_fwd(&cur, step);
        let next: Vec<Goldilocks> = cur.iter().zip(&p).map(|(a, b)| *a + *b).collect();
        ps.push(p);
        rems.push(rem);
        cur = next.clone();
        xs.push(next);
    }
    (xs, ps, rems)
}

pub fn prove_residual_chain(
    x: &[Goldilocks],
    steps: &[ProjectionStep],
    rng: &mut XorShift64,
) -> ResidualChainProof {
    assert!(!steps.is_empty(), "chain needs at least one block");
    let (xs, ps, rems) = residual_forward(x, steps);
    let n = steps.len();

    let mut proj = Vec::with_capacity(n);
    for (i, step) in steps.iter().enumerate() {
        proj.push(prove_projection(
            &xs[i],
            &step.w,
            &step.bias,
            &ps[i],
            &rems[i],
            step.m,
            step.k,
            step.n,
            step.shift,
            rng,
        ));
    }

    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let terms = vec![
        (Goldilocks::ONE, vec![2usize]),
        (neg, vec![0usize]),
        (neg, vec![1usize]),
    ];
    let mut adds = Vec::with_capacity(n);
    for i in 0..n {
        let r: Vec<Goldilocks> = (0..xs[i + 1].len().trailing_zeros() as usize)
            .map(|_| rng.field())
            .collect();
        let mles: Vec<&[Goldilocks]> = vec![&xs[i], &ps[i], &xs[i + 1]];
        let proof = prove_virtual(&mles, &terms, Goldilocks::ZERO, &r);
        adds.push((proof, r));
    }

    // Bind each x_i across its consumers. x_0 has 2 claims; x_i (0<i<N) has 3.
    let mut same_x = Vec::with_capacity(n);
    for i in 0..n {
        let mut claims: Vec<(Vec<Goldilocks>, Goldilocks)> = Vec::new();
        if i > 0 {
            // x_i is the output of add_{i-1} (claimed at r_{i-1}).
            let pt = adds[i - 1].1.clone();
            claims.push((pt.clone(), mle::eval(&xs[i], &pt)));
        }
        // x_i is the input of proj_i (claimed at ch_i ++ u_i).
        let mut in_pt = proj[i].ch.clone();
        in_pt.extend_from_slice(&proj[i].u);
        claims.push((in_pt.clone(), mle::eval(&xs[i], &in_pt)));
        // x_i is the residual input of add_i (claimed at r_i).
        let r_i = adds[i].1.clone();
        claims.push((r_i.clone(), mle::eval(&xs[i], &r_i)));
        same_x.push(prove_same_poly(&xs[i], &claims, rng));
    }

    // Bind each p_i across projection out (pt_i) and residual in (r_i).
    let mut same_p = Vec::with_capacity(n);
    for i in 0..n {
        let pt_i = proj[i].pt.clone();
        let r_i = adds[i].1.clone();
        let claims = vec![
            (pt_i.clone(), mle::eval(&ps[i], &pt_i)),
            (r_i.clone(), mle::eval(&ps[i], &r_i)),
        ];
        same_p.push(prove_same_poly(&ps[i], &claims, rng));
    }

    ResidualChainProof { proj, adds, same_x, same_p }
}

pub fn verify_residual_chain(
    proof: &ResidualChainProof,
    x: &[Goldilocks],
    steps: &[ProjectionStep],
) -> bool {
    assert_eq!(proof.proj.len(), steps.len());
    assert_eq!(proof.adds.len(), steps.len());
    assert_eq!(proof.same_x.len(), steps.len());
    assert_eq!(proof.same_p.len(), steps.len());
    let (xs, ps, rems) = residual_forward(x, steps);
    let n = steps.len();

    for (i, step) in steps.iter().enumerate() {
        if !verify_projection(
            &proof.proj[i],
            &xs[i],
            &step.w,
            &step.bias,
            &ps[i],
            &rems[i],
            step.m,
            step.k,
            step.n,
            step.shift,
        ) {
            return false;
        }
    }

    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let terms = vec![
        (Goldilocks::ONE, vec![2usize]),
        (neg, vec![0usize]),
        (neg, vec![1usize]),
    ];
    for i in 0..n {
        let r = &proof.adds[i].1;
        let fe = vec![
            mle::eval(&xs[i], r),
            mle::eval(&ps[i], r),
            mle::eval(&xs[i + 1], r),
        ];
        if !verify_virtual(&proof.adds[i].0, &terms, Goldilocks::ZERO, r, &fe) {
            return false;
        }
    }

    for i in 0..n {
        let mut claims: Vec<(Vec<Goldilocks>, Goldilocks)> = Vec::new();
        if i > 0 {
            let pt = proof.adds[i - 1].1.clone();
            claims.push((pt.clone(), mle::eval(&xs[i], &pt)));
        }
        let mut in_pt = proof.proj[i].ch.clone();
        in_pt.extend_from_slice(&proof.proj[i].u);
        claims.push((in_pt.clone(), mle::eval(&xs[i], &in_pt)));
        let r_i = proof.adds[i].1.clone();
        claims.push((r_i.clone(), mle::eval(&xs[i], &r_i)));
        if verify_same_poly(&proof.same_x[i], &xs[i], &claims).is_none() {
            return false;
        }
    }

    for i in 0..n {
        let pt_i = proof.proj[i].pt.clone();
        let r_i = proof.adds[i].1.clone();
        let claims = vec![
            (pt_i.clone(), mle::eval(&ps[i], &pt_i)),
            (r_i.clone(), mle::eval(&ps[i], &r_i)),
        ];
        if verify_same_poly(&proof.same_p[i], &ps[i], &claims).is_none() {
            return false;
        }
    }
    true
}

// ===========================================================================
// General data-driven shard composer: fold a list of op primitives into ONE g.
// The op list (and therefore the shard granularity) is a public parameter; this
// is what makes "shard = a group of ops" tunable instead of hand-wired per model.
// ===========================================================================

/// Tensor id in the shard's tensor store.
pub type T = usize;

/// A shard's tensor store: `v[id]` is a Goldilocks tensor; `idx[id]` is an
/// integer index tensor (used by `Lookup`).
#[derive(Default)]
pub struct Store {
    pub v: Vec<Vec<Goldilocks>>,
    pub idx: Vec<Vec<u32>>,
}

impl Store {
    pub fn new() -> Self {
        Store::default()
    }
    pub fn push(&mut self, t: Vec<Goldilocks>) -> T {
        let id = self.v.len();
        self.v.push(t);
        id
    }
    pub fn push_idx(&mut self, t: Vec<u32>) -> T {
        let id = self.idx.len();
        self.idx.push(t);
        id
    }
    pub fn get(&self, id: T) -> &[Goldilocks] {
        &self.v[id]
    }
}

/// One op primitive. `Projection` = matmul + affine + round + range check;
/// `Add` = elementwise `c = a + b`; `Lookup` = `out[i] = table[idx[i]]`.
#[derive(Clone, Debug)]
pub enum Op {
    Transpose {
        x: T,
        out: T,
        m: usize,
        k: usize,
    },
    Scale {
        x: T,
        out: T,
        factor: i64,
        shift: u32,
    },
    MatMul {
        a: T,
        b: T,
        c: T,
        m: usize,
        k: usize,
        n: usize,
    },
    Projection {
        x: T,
        w: T,
        bias: T,
        out: T,
        rem: T,
        m: usize,
        k: usize,
        n: usize,
        shift: u32,
    },
    Add {
        a: T,
        b: T,
        c: T,
    },
    Lookup {
        idx: T,
        out: T,
        table: T,
    },
    Softmax {
        idx: T,
        e: T,
        out: T,
        table: T,
        m: usize,
        n: usize,
    },
    SoftmaxIndex {
        x: T,
        out: T,
        table_len: usize,
    },
    /// Correct GPT-2 gelu index: `idx = (to_i64(x) + offset).clamp(0, table_len-1)`.
    GeluIndex {
        x: T,
        out: T,
        offset: i64,
        table_len: usize,
    },
    /// Numerically-stable softmax index with a causal mask:
    /// `idx = (to_i64(x + mask) - row_max + offset).clamp(0, table_len-1)`.
    StableSoftmaxIndex {
        x: T,
        mask: T,
        out: T,
        offset: i64,
        table_len: usize,
        m: usize,
        n: usize,
    },
    Layernorm {
        x: T,
        w: T,
        b: T,
        out: T,
        rsqrt_table: T,
        m: usize,
        d: usize,
    },
    LayerNormCentered {
        x: T,
        w: T,
        b: T,
        out: T,
        rsqrt_table: T,
        m: usize,
        d: usize,
        n_real: usize,
    },
}

/// A proof for one op, kept heterogeneous because the primitives have different
/// proof shapes.
pub enum OpProof {
    Transpose,
    Scale,
    MatMul(MatmulProof, Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>),
    Projection(ProjectionProof),
    Add(VirtualProof, Vec<Goldilocks>),
    Lookup(FractionalProof, Goldilocks, Goldilocks),
    Softmax(SoftmaxRoundedProof),
    SoftmaxIndex,
    GeluIndex,
    StableSoftmaxIndex,
    Layernorm(LayernormChainProof),
    LayerNormCentered(LayerNormCenteredProof),
}

/// Upper-triangular causal mask as field values (`-(1<<30)` for `j > i`).
pub fn causal_mask(m: usize) -> Vec<Goldilocks> {
    (0..m * m)
        .map(|ij| {
            let (i, j) = (ij / m, ij % m);
            if j > i {
                from_i64(-(1i64 << 30))
            } else {
                Goldilocks::ZERO
            }
        })
        .collect()
}

/// Piecewise rsqrt-table index for a 2^32-scale LayerNorm variance: the raw
/// index is `round(var / 2^18)` (2^14 scale); indices below 2^20 are full
/// resolution, larger ones are quantized with step 2^8.
pub fn rsqrt_index(var: i64) -> u32 {
    const FINE: i64 = 1 << 20;
    let raw = round_div(var, 1 << 18);
    if raw < FINE {
        raw.max(0) as u32
    } else {
        (FINE + ((raw - FINE) >> 8)) as u32
    }
}

/// A shard proof: per-op proofs plus the `same_poly` bindings on every
/// multiply-consumed activation tensor.
pub struct OpShardProof {
    pub ops: Vec<OpProof>,
    /// `binds[j]` merges the claims of tensor `bound[j]`.
    pub bound: Vec<T>,
    pub binds: Vec<SamePolyProof>,
    /// Raw `(tensor, point, eval)` claims emitted by this shard's ops, used by
    /// the shard-DAG composer for cross-shard binding.
    pub claims: Vec<(T, Vec<Goldilocks>, Goldilocks)>,
}

/// Run the forward pass for all ops; materializes every `out`/`rem`/`c` into the
/// store. Input tensors (`x`, `w`, `bias`, `idx`, `table`) must already exist.
fn forward_ops(store: &mut Store, ops: &[Op]) {
    for op in ops.iter().cloned() {
        match op {
            Op::Transpose { x, out, m, k } => {
                store.v[out] = transpose(store.get(x), m, k);
            }
            Op::Scale { x, out, factor, shift } => {
                let o: Vec<Goldilocks> = store.get(x)
                    .iter()
                    .map(|&v| from_i64(round_div(to_i64(v) * factor, 1i64 << shift)))
                    .collect();
                store.v[out] = o;
            }
            Op::MatMul { a, b, c, m, k, n } => {
                let cval = crate::par::mm_par_fixed(store.get(a), store.get(b), m, k, n);
                store.v[c] = cval;
            }
            Op::Projection { x, w, bias, out, rem, m, k, n, shift } => {
                let h = crate::par::mm_par_fixed(store.get(x), store.get(w), m, k, n);
                let o: Vec<Goldilocks> = (0..m * n)
                    .map(|ij| {
                        from_i64(round_div(to_i64(h[ij]), 1i64 << shift) + to_i64(store.get(bias)[ij]))
                    })
                    .collect();
                let r: Vec<Goldilocks> = (0..m * n)
                    .map(|ij| {
                        from_i64(
                            to_i64(h[ij]) - (to_i64(o[ij]) - to_i64(store.get(bias)[ij])) * (1i64 << shift)
                                + (1i64 << (shift - 1)),
                        )
                    })
                    .collect();
                store.v[out] = o;
                store.v[rem] = r;
            }
            Op::Add { a, b, c } => {
                let s: Vec<Goldilocks> = store.get(a)
                    .iter()
                    .zip(store.get(b))
                    .map(|(x, y)| *x + *y)
                    .collect();
                store.v[c] = s;
            }
            Op::Lookup { idx, out, table } => {
                let o: Vec<Goldilocks> = store.idx[idx]
                    .iter()
                    .map(|&i| store.get(table)[i as usize])
                    .collect();
                store.v[out] = o;
            }
            Op::Softmax { idx, e, out, table, m, n } => {
                let ev: Vec<Goldilocks> = store.idx[idx]
                    .iter()
                    .map(|&i| store.get(table)[i as usize])
                    .collect();
                let sum: Vec<Goldilocks> = (0..m)
                    .map(|i| (0..n).fold(Goldilocks::ZERO, |acc, j| acc + ev[i * n + j]))
                    .collect();
                let o: Vec<Goldilocks> = (0..m * n)
                    .map(|ij| from_i64(round_div(to_i64(ev[ij]) * (1i64 << 16), to_i64(sum[ij / n]))))
                    .collect();
                store.v[e] = ev;
                store.v[out] = o;
            }
            Op::SoftmaxIndex { x, out, table_len } => {
                let indices: Vec<u32> = store.get(x)
                    .iter()
                    .map(|&s| ((to_i64(s).max(0)) as u64 % table_len as u64) as u32)
                    .collect();
                store.idx[out] = indices;
            }
            Op::GeluIndex { x, out, offset, table_len } => {
                let indices: Vec<u32> = store.get(x)
                    .iter()
                    .map(|&v| (to_i64(v) + offset).clamp(0, table_len as i64 - 1) as u32)
                    .collect();
                store.idx[out] = indices;
            }
            Op::StableSoftmaxIndex { x, mask, out, offset, table_len, m, n } => {
                let xv = store.get(x);
                let mv = store.get(mask);
                let masked: Vec<Goldilocks> = xv.iter().zip(mv).map(|(a, b)| *a + *b).collect();
                let row_max: Vec<Goldilocks> = (0..m)
                    .map(|i| {
                        (0..n).fold(masked[i * n], |acc, j| {
                            let v = masked[i * n + j];
                            if to_i64(v) > to_i64(acc) { v } else { acc }
                        })
                    })
                    .collect();
                let indices: Vec<u32> = (0..m * n)
                    .map(|ij| {
                        let shifted = to_i64(masked[ij]) - to_i64(row_max[ij / n]);
                        (shifted + offset).clamp(0, table_len as i64 - 1) as u32
                    })
                    .collect();
                store.idx[out] = indices;
            }
            Op::Layernorm { x, w, b, out, rsqrt_table, m, d } => {
                let xv = store.get(x);
                let wv = store.get(w);
                let bv = store.get(b);
                let table = store.get(rsqrt_table);
                let mean_sq: Vec<Goldilocks> = (0..m)
                    .map(|i| (0..d).fold(Goldilocks::ZERO, |acc, j| acc + xv[i * d + j] * xv[i * d + j]))
                    .collect();
                let rsqrt_idx: Vec<u32> = mean_sq
                    .iter()
                    .map(|&v| ((to_i64(v).max(0)) as u64 % table.len() as u64) as u32)
                    .collect();
                let rsqrt: Vec<Goldilocks> = rsqrt_idx.iter().map(|&i| table[i as usize]).collect();
                let scale: Vec<Goldilocks> = (0..m * d).map(|ij| rsqrt[ij / d] * wv[ij]).collect();
                let o: Vec<Goldilocks> = (0..m * d).map(|ij| xv[ij] * scale[ij] + bv[ij]).collect();
                store.v[out] = o;
            }
            Op::LayerNormCentered { x, w, b, out, rsqrt_table, m, d, n_real } => {
                let (_, _, _, _, _, o, _, _, _, _, _) =
                    layer_norm_centered_forward(store.get(x), store.get(w), store.get(b), store.get(rsqrt_table), m, d, n_real);
                store.v[out] = o;
            }
        }
    }
}

/// Prove a shard: run forward, prove each op, collect the claims each op makes
/// on activation tensors, and `same_poly`-bind every tensor claimed more than
/// once (at different points). `boundary` marks the shard output tensor(s) that
/// are committed at the boundary and therefore not bound internally.
pub fn prove_shard(store: &mut Store, ops: &[Op], boundary: &[T], rng: &mut XorShift64) -> OpShardProof {
    forward_ops(store, ops);
    prove_shard_precomputed(store, ops, boundary, rng)
}

/// Prove a shard whose witness is already materialized in `store` (the forward
/// pass has already run). Read-only on `store`, so shards can be proven in
/// parallel by the shard-DAG composer.
pub fn prove_shard_precomputed(
    store: &Store,
    ops: &[Op],
    boundary: &[T],
    rng: &mut XorShift64,
) -> OpShardProof {

    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let add_terms = vec![
        (Goldilocks::ONE, vec![2usize]),
        (neg, vec![0usize]),
        (neg, vec![1usize]),
    ];

    // (tensor_id, point, eval) claims emitted by each op's proof.
    let mut claims: Vec<(T, Vec<Goldilocks>, Goldilocks)> = Vec::new();
    let mut op_proofs = Vec::with_capacity(ops.len());

    for op in ops {
        match *op {
            Op::Transpose { .. } => {
                op_proofs.push(OpProof::Transpose);
            }
            Op::Scale { .. } => {
                op_proofs.push(OpProof::Scale);
            }
            Op::MatMul { a, b, c, m, k, n } => {
                let at = transpose(store.get(a), m, k);
                let u: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
                let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
                let ch: Vec<Goldilocks> = (0..k.trailing_zeros() as usize).map(|_| rng.field()).collect();
                let p = matmul_prove(&at, store.get(b), store.get(c), m, k, n, &u, &v, &ch);
                let mut ap = ch.clone();
                ap.extend_from_slice(&u);
                claims.push((a, ap.clone(), mle::eval(store.get(a), &ap)));
                let mut bp = v.clone();
                bp.extend_from_slice(&ch);
                claims.push((b, bp.clone(), mle::eval(store.get(b), &bp)));
                let mut cp = v.clone();
                cp.extend_from_slice(&u);
                claims.push((c, cp.clone(), mle::eval(store.get(c), &cp)));
                op_proofs.push(OpProof::MatMul(p, u, v, ch));
            }
            Op::Projection { x, w, bias, out, rem, m, k, n, shift } => {
                let p = prove_projection(
                    store.get(x),
                    store.get(w),
                    store.get(bias),
                    store.get(out),
                    store.get(rem),
                    m, k, n, shift, rng,
                );
                let mut in_pt = p.ch.clone();
                in_pt.extend_from_slice(&p.u);
                claims.push((x, in_pt.clone(), mle::eval(store.get(x), &in_pt)));
                claims.push((out, p.pt.clone(), mle::eval(store.get(out), &p.pt)));
                op_proofs.push(OpProof::Projection(p));
            }
            Op::Add { a, b, c } => {
                let r: Vec<Goldilocks> = (0..store.get(c).len().trailing_zeros() as usize)
                    .map(|_| rng.field())
                    .collect();
                let mles: Vec<&[Goldilocks]> = vec![store.get(a), store.get(b), store.get(c)];
                let proof = prove_virtual(&mles, &add_terms, Goldilocks::ZERO, &r);
                claims.push((a, r.clone(), mle::eval(store.get(a), &r)));
                claims.push((b, r.clone(), mle::eval(store.get(b), &r)));
                claims.push((c, r.clone(), mle::eval(store.get(c), &r)));
                op_proofs.push(OpProof::Add(proof, r));
            }
            Op::Lookup { idx, out, table } => {
                let alpha = rng.field();
                let beta = rng.field();
                let p = prove_lookup_fractional(
                    &store.idx[idx],
                    store.get(out),
                    store.get(table),
                    alpha,
                    beta,
                    rng,
                );
                op_proofs.push(OpProof::Lookup(p, alpha, beta));
            }
            Op::Softmax { idx, e, out, table, m, n } => {
                let p = prove_softmax_rounded(
                    &store.idx[idx],
                    store.get(e),
                    store.get(out),
                    store.get(table),
                    m,
                    n,
                    rng,
                );
                claims.push((out, p.r_scale.clone(), mle::eval(store.get(out), &p.r_scale)));
                op_proofs.push(OpProof::Softmax(p));
            }
            Op::SoftmaxIndex { .. } => {
                op_proofs.push(OpProof::SoftmaxIndex);
            }
            Op::GeluIndex { .. } => {
                op_proofs.push(OpProof::GeluIndex);
            }
            Op::StableSoftmaxIndex { .. } => {
                op_proofs.push(OpProof::StableSoftmaxIndex);
            }
            Op::Layernorm { x, w, b, out, rsqrt_table, m, d } => {
                let p = prove_layernorm_chain(
                    store.get(x),
                    store.get(w),
                    store.get(b),
                    store.get(rsqrt_table),
                    m,
                    d,
                    rng,
                );
                claims.push((out, p.r_out.clone(), mle::eval(store.get(out), &p.r_out)));
                op_proofs.push(OpProof::Layernorm(p));
            }
            Op::LayerNormCentered { x, w, b, out, rsqrt_table, m, d, n_real } => {
                let p = prove_layer_norm_centered(
                    store.get(x),
                    store.get(w),
                    store.get(b),
                    store.get(rsqrt_table),
                    m,
                    d,
                    n_real,
                    rng,
                );
                claims.push((x, p.centered_r.clone(), mle::eval(store.get(x), &p.centered_r)));
                claims.push((out, p.r_out.clone(), mle::eval(store.get(out), &p.r_out)));
                op_proofs.push(OpProof::LayerNormCentered(p));
            }
        }
    }

    // Group claims by tensor and bind those with >1 distinct point (and not a
    // committed boundary tensor).
    let raw_claims = claims.clone();
    let boundary_set: std::collections::HashSet<T> = boundary.iter().copied().collect();
    let mut by_tensor: std::collections::BTreeMap<T, Vec<(Vec<Goldilocks>, Goldilocks)>> =
        std::collections::BTreeMap::new();
    for (t, pt, ev) in claims {
        by_tensor.entry(t).or_default().push((pt, ev));
    }

    let mut bound = Vec::new();
    let mut binds = Vec::new();
    for (t, cs) in by_tensor {
        if boundary_set.contains(&t) || cs.len() < 2 {
            continue;
        }
        let sp = prove_same_poly(store.get(t), &cs, rng);
        bound.push(t);
        binds.push(sp);
    }

    OpShardProof { ops: op_proofs, bound, binds, claims: raw_claims }
}

/// Verify a shard proof (recomputes the witness first).
pub fn verify_shard(store: &Store, ops: &[Op], proof: &OpShardProof) -> bool {
    let mut ws = Store { v: store.v.clone(), idx: store.idx.clone() };
    forward_ops(&mut ws, ops);
    verify_shard_precomputed(&ws, ops, proof)
}

/// Verify a shard proof against an already-materialized witness store. Read-only
/// on `store`, so shards can be verified in parallel.
pub fn verify_shard_precomputed(store: &Store, ops: &[Op], proof: &OpShardProof) -> bool {
    let ws = store;
    assert_eq!(proof.ops.len(), ops.len());
    assert_eq!(proof.bound.len(), proof.binds.len());

    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let add_terms = vec![
        (Goldilocks::ONE, vec![2usize]),
        (neg, vec![0usize]),
        (neg, vec![1usize]),
    ];

    let mut claims: Vec<(T, Vec<Goldilocks>, Goldilocks)> = Vec::new();
    for (op, p) in ops.iter().zip(&proof.ops) {
        match (op, p) {
            (Op::Transpose { .. }, OpProof::Transpose) => {
                // No proof: the verifier recomputes `out = transpose(x)` in
                // forward_ops, and the downstream op's claim on `out` is checked
                // against that recomputed value.
            }
            (Op::Scale { .. }, OpProof::Scale) => {}
            (Op::MatMul { a, b, c, m, k, n: _ }, OpProof::MatMul(mp, u, v, ch)) => {
                let at = transpose(ws.get(*a), *m, *k);
                let a_restricted = mle::partial_eval(&at, u);
                let b_restricted = mle::partial_eval(ws.get(*b), v);
                let f_eval = mle::eval(&a_restricted, ch);
                let h_eval = mle::eval(&b_restricted, ch);
                if !matmul_verify(mp, ch, f_eval, h_eval) {
                    return false;
                }
                let mut ap = ch.clone();
                ap.extend_from_slice(u);
                claims.push((*a, ap.clone(), mle::eval(ws.get(*a), &ap)));
                let mut bp = v.clone();
                bp.extend_from_slice(ch);
                claims.push((*b, bp.clone(), mle::eval(ws.get(*b), &bp)));
                let mut cp = v.clone();
                cp.extend_from_slice(u);
                claims.push((*c, cp.clone(), mle::eval(ws.get(*c), &cp)));
            }
            (
                Op::Projection { x, w, bias, out, rem, m, k, n, shift },
                OpProof::Projection(pp),
            ) => {
                if !verify_projection(
                    pp,
                    ws.get(*x),
                    ws.get(*w),
                    ws.get(*bias),
                    ws.get(*out),
                    ws.get(*rem),
                    *m, *k, *n, *shift,
                ) {
                    return false;
                }
                let mut in_pt = pp.ch.clone();
                in_pt.extend_from_slice(&pp.u);
                claims.push((*x, in_pt.clone(), mle::eval(ws.get(*x), &in_pt)));
                claims.push((*out, pp.pt.clone(), mle::eval(ws.get(*out), &pp.pt)));
            }
            (Op::Add { a, b, c }, OpProof::Add(vp, r)) => {
                let fe = vec![
                    mle::eval(ws.get(*a), r),
                    mle::eval(ws.get(*b), r),
                    mle::eval(ws.get(*c), r),
                ];
                if !verify_virtual(vp, &add_terms, Goldilocks::ZERO, r, &fe) {
                    return false;
                }
                claims.push((*a, r.clone(), mle::eval(ws.get(*a), r)));
                claims.push((*b, r.clone(), mle::eval(ws.get(*b), r)));
                claims.push((*c, r.clone(), mle::eval(ws.get(*c), r)));
            }
            (Op::Lookup { idx, out, table }, OpProof::Lookup(fp, alpha, beta)) => {
                if !verify_lookup_fractional(
                    fp,
                    &ws.idx[*idx],
                    ws.get(*out),
                    ws.get(*table),
                    *alpha,
                    *beta,
                ) {
                    return false;
                }
            }
            (Op::Softmax { idx, e, out, table, m, n }, OpProof::Softmax(sp)) => {
                if !verify_softmax_rounded(
                    sp,
                    &ws.idx[*idx],
                    ws.get(*e),
                    ws.get(*out),
                    ws.get(*table),
                    *m,
                    *n,
                ) {
                    return false;
                }
                claims.push((*out, sp.r_scale.clone(), mle::eval(ws.get(*out), &sp.r_scale)));
            }
            (Op::SoftmaxIndex { .. }, OpProof::SoftmaxIndex) => {
                // No proof: the verifier recomputes `idx = softmax_index(scores)`.
            }
            (Op::GeluIndex { .. }, OpProof::GeluIndex) => {}
            (Op::StableSoftmaxIndex { .. }, OpProof::StableSoftmaxIndex) => {}
            (Op::Layernorm { x, w, b, out, rsqrt_table, m, d }, OpProof::Layernorm(lp)) => {
                if !verify_layernorm_chain(
                    lp,
                    ws.get(*x),
                    ws.get(*w),
                    ws.get(*b),
                    ws.get(*rsqrt_table),
                    *m,
                    *d,
                ) {
                    return false;
                }
                claims.push((*out, lp.r_out.clone(), mle::eval(ws.get(*out), &lp.r_out)));
            }
            (Op::LayerNormCentered { x, w, b, out, rsqrt_table, m, d, n_real }, OpProof::LayerNormCentered(lp)) => {
                if !verify_layer_norm_centered(
                    lp,
                    ws.get(*x),
                    ws.get(*w),
                    ws.get(*b),
                    ws.get(*rsqrt_table),
                    *m,
                    *d,
                    *n_real,
                ) {
                    return false;
                }
                claims.push((*x, lp.centered_r.clone(), mle::eval(ws.get(*x), &lp.centered_r)));
                claims.push((*out, lp.r_out.clone(), mle::eval(ws.get(*out), &lp.r_out)));
            }
            _ => return false,
        }
    }

    // Rebuild the same grouping as prove_shard and verify each binding.
    let mut by_tensor: std::collections::BTreeMap<T, Vec<(Vec<Goldilocks>, Goldilocks)>> =
        std::collections::BTreeMap::new();
    for (t, pt, ev) in claims {
        by_tensor.entry(t).or_default().push((pt, ev));
    }
    // Recompute which tensors are bound (same rule: >1 claim). We rely on the
    // proof's `bound` list to be the canonical order.
    let mut bound_claims: Vec<Vec<(Vec<Goldilocks>, Goldilocks)>> = Vec::new();
    for &t in &proof.bound {
        match by_tensor.get(&t) {
            Some(cs) if cs.len() >= 2 => bound_claims.push(cs.clone()),
            _ => return false,
        }
    }
    if bound_claims.len() != proof.binds.len() {
        return false;
    }
    for (i, &t) in proof.bound.iter().enumerate() {
        if verify_same_poly(&proof.binds[i], ws.get(t), &bound_claims[i]).is_none() {
            return false;
        }
    }
    true
}

/// A whole-model shard-DAG proof: every shard folded into one g, plus
/// `same_poly` cross-shard bindings on every tensor claimed by more than one
/// shard. `ops_per_shard` is the public granularity parameter.
pub struct ShardDagProof {
    pub shards: Vec<OpShardProof>,
    pub cross_tensors: Vec<T>,
    pub cross_binds: Vec<SamePolyProof>,
}

/// Split `op_count` into contiguous `[start, end)` ranges of `ops_per_shard`.
pub fn shard_ranges(op_count: usize, ops_per_shard: usize) -> Vec<(usize, usize)> {
    let ops_per_shard = ops_per_shard.max(1);
    let mut out = Vec::new();
    let mut s = 0;
    while s < op_count {
        let e = (s + ops_per_shard).min(op_count);
        out.push((s, e));
        s = e;
    }
    out
}

pub fn prove_shard_dag(
    store: &mut Store,
    ops: &[Op],
    ops_per_shard: usize,
    rng: &mut XorShift64,
) -> ShardDagProof {
    let ranges = shard_ranges(ops.len(), ops_per_shard);

    // Materialize every op output once, then prove each shard's sumchecks in
    // parallel (read-only on the store). Per-shard RNGs are seeded
    // deterministically from the caller's RNG, so the proof stays reproducible.
    forward_ops(store, ops);
    let store_ref: &Store = store;
    let seeds: Vec<u64> = (0..ranges.len()).map(|_| rng.next_u64()).collect();
    let shards: Vec<OpShardProof> = ranges
        .par_iter()
        .zip(seeds.par_iter())
        .map(|(&(s, e), &seed)| {
            let mut srng = XorShift64::new(seed);
            prove_shard_precomputed(store_ref, &ops[s..e], &[], &mut srng)
        })
        .collect();

    // Group raw claims by tensor across shards; bind tensors claimed by >1 shard.
    let mut by_tensor: std::collections::BTreeMap<T, Vec<(Vec<Goldilocks>, Goldilocks)>> =
        std::collections::BTreeMap::new();
    for shard in shards.iter() {
        for (t, pt, ev) in &shard.claims {
            by_tensor.entry(*t).or_default().push((pt.clone(), *ev));
        }
    }
    // Recompute per-tensor shard count with a set (a tensor may be claimed twice
    // by one shard, so a plain counter would overcount).
    let mut tensor_shards: std::collections::BTreeMap<T, std::collections::HashSet<usize>> =
        std::collections::BTreeMap::new();
    for (si, shard) in shards.iter().enumerate() {
        for (t, _, _) in &shard.claims {
            tensor_shards.entry(*t).or_default().insert(si);
        }
    }

    let mut cross_tensors = Vec::new();
    let mut cross_binds = Vec::new();
    for (t, claims) in by_tensor {
        if tensor_shards.get(&t).map(|s| s.len()).unwrap_or(0) > 1 {
            let sp = prove_same_poly(store.get(t), &claims, rng);
            cross_tensors.push(t);
            cross_binds.push(sp);
        }
    }
    ShardDagProof { shards, cross_tensors, cross_binds }
}

pub fn verify_shard_dag(
    store: &Store,
    ops: &[Op],
    ops_per_shard: usize,
    proof: &ShardDagProof,
) -> bool {
    let ranges = shard_ranges(ops.len(), ops_per_shard);
    if proof.shards.len() != ranges.len() || proof.cross_tensors.len() != proof.cross_binds.len() {
        return false;
    }

    // Recompute the witness to get fresh tensor values for binding evals.
    let mut ws = Store { v: store.v.clone(), idx: store.idx.clone() };
    forward_ops(&mut ws, ops);

    // Verify each shard in parallel against the shared fresh witness.
    let all_ok = ranges
        .par_iter()
        .enumerate()
        .map(|(i, &(s, e))| verify_shard_precomputed(&ws, &ops[s..e], &proof.shards[i]))
        .all(|ok| ok);
    if !all_ok {
        return false;
    }

    // Rebuild the same cross-shard grouping (points only; evals recomputed).
    let mut by_tensor: std::collections::BTreeMap<T, Vec<Vec<Goldilocks>>> =
        std::collections::BTreeMap::new();
    let mut tensor_shards: std::collections::BTreeMap<T, std::collections::HashSet<usize>> =
        std::collections::BTreeMap::new();
    for (si, shard) in proof.shards.iter().enumerate() {
        for (t, pt, _) in &shard.claims {
            by_tensor.entry(*t).or_default().push(pt.clone());
            tensor_shards.entry(*t).or_default().insert(si);
        }
    }

    let mut expected_cross: Vec<T> = Vec::new();
    for (t, _) in &by_tensor {
        if tensor_shards.get(t).map(|s| s.len()).unwrap_or(0) > 1 {
            expected_cross.push(*t);
        }
    }
    if expected_cross != proof.cross_tensors {
        return false;
    }
    for (ci, &t) in proof.cross_tensors.iter().enumerate() {
        let claims: Vec<(Vec<Goldilocks>, Goldilocks)> = by_tensor[&t]
            .iter()
            .map(|pt| (pt.clone(), mle::eval(ws.get(t), pt)))
            .collect();
        if verify_same_poly(&proof.cross_binds[ci], ws.get(t), &claims).is_none() {
            return false;
        }
    }
    true
}

/// Committed cross-shard binding: open a committed boundary tensor at every
/// claim point, verify each opening against the commitment, and merge the claims
/// with `same_poly`. The commitment itself ties the two shards' claims to one
/// tensor; the merge collapses them toward a single future opening. This is the
/// "boundary commit" form of cross-shard binding (the Commit/Open stages) that
/// the plain `prove_shard_dag` defers.
pub fn committed_cross_bind(
    whir: &crate::whir::Whir,
    committed: &crate::committed::Committed,
    tensor: &[Goldilocks],
    claims: &[(Vec<Goldilocks>, Goldilocks)],
    rng: &mut XorShift64,
) -> Option<SamePolyProof> {
    for (pt, expected) in claims {
        let (open, opened) = whir.open(committed.prover_data.clone(), &committed.protocol, pt);
        if whir
            .verify(&committed.commitment, &open, &committed.protocol, pt)
            .ok()?
            != opened
        {
            return None;
        }
        if opened != *expected || opened != mle::eval(tensor, pt) {
            return None;
        }
    }
    Some(prove_same_poly(tensor, claims, rng))
}

/// Verify a [`committed_cross_bind`] proof: open each claim against the
/// commitment, then verify the merged `same_poly`.
pub fn verify_committed_cross_bind(
    whir: &crate::whir::Whir,
    committed: &crate::committed::Committed,
    tensor: &[Goldilocks],
    claims: &[(Vec<Goldilocks>, Goldilocks)],
    proof: &SamePolyProof,
) -> bool {
    for (pt, expected) in claims {
        let (open, opened) = whir.open(committed.prover_data.clone(), &committed.protocol, pt);
        if whir
            .verify(&committed.commitment, &open, &committed.protocol, pt)
            .ok()
            != Some(opened)
        {
            return false;
        }
        if opened != *expected || opened != mle::eval(tensor, pt) {
            return false;
        }
    }
    verify_same_poly(proof, tensor, claims).is_some()
}

/// A committed shard-DAG proof: every shard proven as before (plain sumchecks,
/// virtual intermediates), plus a WHIR commitment per cross-shard boundary tensor
/// and a committed cross-shard binding. Only boundary tensors are committed; this
/// is the "only shard boundaries + global weights commit" model.
pub struct CommittedShardDagProof {
    pub shards: Vec<OpShardProof>,
    pub boundary_tensors: Vec<T>,
    pub boundary_commitments: Vec<crate::committed::Committed>,
    pub cross_binds: Vec<SamePolyProof>,
}

pub fn prove_committed_shard_dag(
    store: &mut Store,
    ops: &[Op],
    ops_per_shard: usize,
    whir: &crate::whir::Whir,
    rng: &mut XorShift64,
) -> CommittedShardDagProof {
    let plain = prove_shard_dag(store, ops, ops_per_shard, rng);
    let boundary_tensors = plain.cross_tensors.clone();

    let mut boundary_commitments = Vec::with_capacity(boundary_tensors.len());
    for &t in &boundary_tensors {
        boundary_commitments.push(crate::committed::commit(whir, store.get(t)));
    }

    let mut cross_binds = Vec::with_capacity(boundary_tensors.len());
    for (i, &t) in boundary_tensors.iter().enumerate() {
        let mut claims = Vec::new();
        for shard in &plain.shards {
            for (tt, pt, ev) in &shard.claims {
                if *tt == t {
                    claims.push((pt.clone(), *ev));
                }
            }
        }
        let sp = committed_cross_bind(whir, &boundary_commitments[i], store.get(t), &claims, rng)
            .expect("honest committed bind");
        cross_binds.push(sp);
    }

    CommittedShardDagProof {
        shards: plain.shards,
        boundary_tensors,
        boundary_commitments,
        cross_binds,
    }
}

pub fn verify_committed_shard_dag(
    store: &Store,
    ops: &[Op],
    ops_per_shard: usize,
    whir: &crate::whir::Whir,
    proof: &CommittedShardDagProof,
) -> bool {
    let ranges = shard_ranges(ops.len(), ops_per_shard);
    if proof.shards.len() != ranges.len()
        || proof.boundary_tensors.len() != proof.boundary_commitments.len()
        || proof.boundary_tensors.len() != proof.cross_binds.len()
    {
        return false;
    }
    for (i, (s, e)) in ranges.iter().enumerate() {
        if !verify_shard(store, &ops[*s..*e], &proof.shards[i]) {
            return false;
        }
    }

    let mut ws = Store { v: store.v.clone(), idx: store.idx.clone() };
    forward_ops(&mut ws, ops);

    for (i, &t) in proof.boundary_tensors.iter().enumerate() {
        let mut claims = Vec::new();
        for shard in &proof.shards {
            for (tt, pt, ev) in &shard.claims {
                if *tt == t {
                    claims.push((pt.clone(), *ev));
                }
            }
        }
        if !verify_committed_cross_bind(
            whir,
            &proof.boundary_commitments[i],
            ws.get(t),
            &claims,
            &proof.cross_binds[i],
        ) {
            return false;
        }
    }
    true
}

/// Batch-commit same-size weight/bias tensors into ONE commitment — the "global
/// weights commit": commit once, open per-use. The verifier opens a weight
/// against this single commitment instead of trusting raw values.
pub fn commit_weights_batch(
    whir: &crate::whir::Whir,
    tensors: &[&[Goldilocks]],
) -> crate::committed::BatchCtx {
    let (commitment, prover_data, protocol, w) = whir.commit_batch(tensors);
    crate::committed::BatchCtx {
        commitment,
        prover_data,
        protocol,
        whir: w,
        num_tables: tensors.len(),
    }
}

/// Open the `idx`-th weight of a batch commitment at `point` and verify it
/// equals `tensor`'s MLE evaluation there.
pub fn verify_weight_batch(
    batch: &crate::committed::BatchCtx,
    idx: usize,
    tensor: &[Goldilocks],
    point: &[Goldilocks],
) -> bool {
    let (open, ev) = batch
        .whir
        .open_batch(batch.prover_data.clone(), &batch.protocol, idx, batch.num_tables, point);
    if batch
        .whir
        .verify_batch(&batch.commitment, &open, &batch.protocol, idx, batch.num_tables, point)
        .ok()
        != Some(ev)
    {
        return false;
    }
    ev == mle::eval(tensor, point)
}

/// A projection block proven with committed weights: the matmul keeps the
/// intermediate `h = x @ w` virtual (plain GKR claim), but `w` and `bias` are
/// opened against their global batch commitments at the prescribed claim points.
pub struct CommittedProjectionProof {
    pub plain: ProjectionProof,
    pub w_open: (crate::whir::Proof, Goldilocks),
    pub bias_open: (crate::whir::Proof, Goldilocks),
}

#[allow(clippy::too_many_arguments)]
pub fn prove_projection_committed(
    x: &[Goldilocks],
    w_batch: &crate::committed::BatchCtx,
    w_idx: usize,
    bias_batch: &crate::committed::BatchCtx,
    bias_idx: usize,
    w: &[Goldilocks],
    bias: &[Goldilocks],
    out: &[Goldilocks],
    rem: &[Goldilocks],
    m: usize,
    k: usize,
    n: usize,
    shift: u32,
    rng: &mut XorShift64,
) -> CommittedProjectionProof {
    let plain = prove_projection(x, w, bias, out, rem, m, k, n, shift, rng);
    // The matmul opens the weight `w` (k x n) at `v ++ ch`.
    let mut w_pt = plain.v.clone();
    w_pt.extend_from_slice(&plain.ch);
    let w_open = w_batch.whir.open_batch(
        w_batch.prover_data.clone(),
        &w_batch.protocol,
        w_idx,
        w_batch.num_tables,
        &w_pt,
    );
    // The affine opens `bias` (m x n) at `pt = v ++ u`.
    let bias_open = bias_batch.whir.open_batch(
        bias_batch.prover_data.clone(),
        &bias_batch.protocol,
        bias_idx,
        bias_batch.num_tables,
        &plain.pt,
    );
    CommittedProjectionProof { plain, w_open, bias_open }
}

#[allow(clippy::too_many_arguments)]
pub fn verify_projection_committed(
    proof: &CommittedProjectionProof,
    x: &[Goldilocks],
    w_batch: &crate::committed::BatchCtx,
    w_idx: usize,
    bias_batch: &crate::committed::BatchCtx,
    bias_idx: usize,
    w: &[Goldilocks],
    bias: &[Goldilocks],
    out: &[Goldilocks],
    rem: &[Goldilocks],
    m: usize,
    k: usize,
    n: usize,
    shift: u32,
) -> bool {
    if !verify_projection(&proof.plain, x, w, bias, out, rem, m, k, n, shift) {
        return false;
    }
    let mut w_pt = proof.plain.v.clone();
    w_pt.extend_from_slice(&proof.plain.ch);
    if w_batch
        .whir
        .verify_batch(&w_batch.commitment, &proof.w_open.0, &w_batch.protocol, w_idx, w_batch.num_tables, &w_pt)
        .ok()
        != Some(proof.w_open.1)
        || proof.w_open.1 != mle::eval(w, &w_pt)
    {
        return false;
    }
    if bias_batch
        .whir
        .verify_batch(&bias_batch.commitment, &proof.bias_open.0, &bias_batch.protocol, bias_idx, bias_batch.num_tables, &proof.plain.pt)
        .ok()
        != Some(proof.bias_open.1)
        || proof.bias_open.1 != mle::eval(bias, &proof.plain.pt)
    {
        return false;
    }
    true
}

/// Build a full pre-norm GPT-2 transformer layer as an op list:
/// `h = layernorm(x)` -> multi-head attention -> `x2 = x + attn` ->
/// `h2 = layernorm(x2)` -> FFN (fc -> gelu -> proj) -> `out = x2 + ffn_out`.
/// Returns the op list and the output tensor id. `heads` is the number of
/// attention heads (`d` must be divisible by `heads`).
#[allow(clippy::too_many_arguments)]
pub fn build_gpt2_layer(
    store: &mut Store,
    x: T,
    heads: usize,
    m: usize,
    d: usize,
    ffn: usize,
    shift: u32,
    exp_table: T,
    gelu_table: T,
    rsqrt_table: T,
    rng: &mut XorShift64,
) -> (Vec<Op>, T) {
    let dh = d / heads;
    let mut ops = Vec::new();

    // pre-norm for attention
    let ln1_w = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 5) as i64 + 1)).collect());
    let ln1_b = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect());
    let h = store.push(vec![]);
    ops.push(Op::Layernorm { x, w: ln1_w, b: ln1_b, out: h, rsqrt_table, m, d });

    // multi-head attention on h
    let mut head_outs = Vec::new();
    for _ in 0..heads {
        let wq = store.push((0..d * dh).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
        let wk = store.push((0..d * dh).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
        let wv = store.push((0..d * dh).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
        let wo = store.push((0..dh * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
        let bq = store.push((0..m * dh).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect());
        let bk = store.push((0..m * dh).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect());
        let bv = store.push((0..m * dh).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect());
        let bo = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect());
        let q = store.push(vec![]);
        let k = store.push(vec![]);
        let v = store.push(vec![]);
        let qr = store.push(vec![]);
        let kr = store.push(vec![]);
        let vr = store.push(vec![]);
        let kt = store.push(vec![]);
        let scores = store.push(vec![]);
        let idx = store.push_idx(vec![]);
        let e = store.push(vec![]);
        let probs = store.push(vec![]);
        let attn = store.push(vec![]);
        let out_h = store.push(vec![]);
        let out_h_rem = store.push(vec![]);
        ops.push(Op::Projection { x: h, w: wq, bias: bq, out: q, rem: qr, m, k: d, n: dh, shift });
        ops.push(Op::Projection { x: h, w: wk, bias: bk, out: k, rem: kr, m, k: d, n: dh, shift });
        ops.push(Op::Projection { x: h, w: wv, bias: bv, out: v, rem: vr, m, k: d, n: dh, shift });
        ops.push(Op::Transpose { x: k, out: kt, m, k: dh });
        ops.push(Op::MatMul { a: q, b: kt, c: scores, m, k: dh, n: m });
        ops.push(Op::SoftmaxIndex { x: scores, out: idx, table_len: 1 << 8 });
        ops.push(Op::Softmax { idx, e, out: probs, table: exp_table, m, n: m });
        ops.push(Op::MatMul { a: probs, b: v, c: attn, m, k: m, n: dh });
        ops.push(Op::Projection { x: attn, w: wo, bias: bo, out: out_h, rem: out_h_rem, m, k: dh, n: d, shift });
        head_outs.push(out_h);
    }
    let mut attn_acc = head_outs[0];
    for &ho in &head_outs[1..] {
        let sum = store.push(vec![]);
        ops.push(Op::Add { a: attn_acc, b: ho, c: sum });
        attn_acc = sum;
    }
    // residual
    let x2 = store.push(vec![]);
    ops.push(Op::Add { a: x, b: attn_acc, c: x2 });

    // pre-norm for FFN
    let ln2_w = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 5) as i64 + 1)).collect());
    let ln2_b = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect());
    let h2 = store.push(vec![]);
    ops.push(Op::Layernorm { x: x2, w: ln2_w, b: ln2_b, out: h2, rsqrt_table, m, d });

    // FFN
    let fc_w = store.push((0..d * ffn).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
    let fc_b = store.push((0..m * ffn).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect());
    let proj_w = store.push((0..ffn * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
    let proj_b = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect());
    let fc = store.push(vec![]);
    let fc_rem = store.push(vec![]);
    let gelu_idx = store.push_idx(vec![]);
    let act = store.push(vec![]);
    let proj2 = store.push(vec![]);
    let proj2_rem = store.push(vec![]);
    ops.push(Op::Projection { x: h2, w: fc_w, bias: fc_b, out: fc, rem: fc_rem, m, k: d, n: ffn, shift });
    ops.push(Op::SoftmaxIndex { x: fc, out: gelu_idx, table_len: 64 });
    ops.push(Op::Lookup { idx: gelu_idx, out: act, table: gelu_table });
    ops.push(Op::Projection { x: act, w: proj_w, bias: proj_b, out: proj2, rem: proj2_rem, m, k: ffn, n: d, shift });

    // residual
    let out = store.push(vec![]);
    ops.push(Op::Add { a: x2, b: proj2, c: out });
    (ops, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::PrimeCharacteristicRing;

    fn step(m: usize, k: usize, n: usize, shift: u32, rng: &mut XorShift64) -> ProjectionStep {
        let w: Vec<Goldilocks> = (0..k * n).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let bias: Vec<Goldilocks> = (0..m * n).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect();
        ProjectionStep::new(m, k, n, shift, w, bias)
    }

    #[test]
    fn single_projection_is_op_granularity() {
        let mut rng = XorShift64::new(0x0A0A);
        let (m, k, n, shift) = (4usize, 8usize, 8usize, 8u32);
        let x: Vec<Goldilocks> = (0..m * k).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let steps = vec![step(m, k, n, shift, &mut rng)];
        let proof = prove_projection_chain(&x, &steps, &mut rng);
        assert!(proof.binds.is_empty(), "no internal binding for one op");
        assert!(verify_projection_chain(&proof, &x, &steps));
    }

    #[test]
    fn chain_folds_n_projections_into_one_g() {
        let mut rng = XorShift64::new(0x0B0B);
        let (m, d, shift) = (4usize, 8usize, 8u32);
        let x: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let steps = vec![
            step(m, d, d, shift, &mut rng),
            step(m, d, d, shift, &mut rng),
            step(m, d, d, shift, &mut rng),
            step(m, d, d, shift, &mut rng),
        ];
        let proof = prove_projection_chain(&x, &steps, &mut rng);
        assert_eq!(proof.binds.len(), 3, "N-1 internal bindings");
        assert!(verify_projection_chain(&proof, &x, &steps));

        // Tampering with an intermediate weight must break the whole chain: the
        // verifier recomputes the forward pass from `steps`, so a single altered
        // weight propagates a wrong witness and the affected projection fails.
        let mut bad_steps = steps.clone();
        bad_steps[1].w[0] = bad_steps[1].w[0] + Goldilocks::ONE;
        assert!(!verify_projection_chain(&proof, &x, &bad_steps));
    }

    #[test]
    fn projection_sharded_granularity() {
        let mut rng = XorShift64::new(0x1111);
        let (m, d, shift) = (4usize, 8usize, 8u32);
        let x: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let steps: Vec<ProjectionStep> = (0..8).map(|_| step(m, d, d, shift, &mut rng)).collect();

        let ops_per_shard = 2;
        let proof = prove_projection_sharded(&x, &steps, ops_per_shard, &mut rng);
        assert_eq!(proof.shards.len(), 4, "8 ops / 2 per shard = 4 shards");
        assert_eq!(proof.cross_binds.len(), 3, "3 cross-shard boundaries");
        assert!(verify_projection_sharded(&proof, &x, &steps, ops_per_shard));

        // A single altered weight in the middle breaks the affected shard.
        let mut bad_steps = steps.clone();
        bad_steps[3].w[0] = bad_steps[3].w[0] + Goldilocks::ONE;
        assert!(!verify_projection_sharded(&proof, &x, &bad_steps, ops_per_shard));

        // Whole-model granularity = one shard, no cross-bindings.
        let whole = prove_projection_sharded(&x, &steps, steps.len(), &mut rng);
        assert_eq!(whole.shards.len(), 1);
        assert!(whole.cross_binds.is_empty());
        assert!(verify_projection_sharded(&whole, &x, &steps, steps.len()));
    }

    #[test]
    fn residual_chain_roundtrip() {
        let mut rng = XorShift64::new(0x0D0D);
        let (m, d, shift) = (4usize, 8usize, 8u32);
        let x: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let steps = vec![
            step(m, d, d, shift, &mut rng),
            step(m, d, d, shift, &mut rng),
            step(m, d, d, shift, &mut rng),
        ];
        let proof = prove_residual_chain(&x, &steps, &mut rng);
        assert_eq!(proof.same_x.len(), 3);
        assert_eq!(proof.same_p.len(), 3);
        assert!(verify_residual_chain(&proof, &x, &steps));

        // Tampering with a residual output must break the chain.
        let mut bad_steps = steps.clone();
        bad_steps[1].w[0] = bad_steps[1].w[0] + Goldilocks::ONE;
        assert!(!verify_residual_chain(&proof, &x, &bad_steps));
    }

    #[test]
    fn op_shard_ffn_roundtrip() {
        let mut rng = XorShift64::new(0x0E0E);
        let (m, d, ffn, shift) = (4usize, 8usize, 16usize, 8u32);
        let mut store = Store::new();
        let x = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
        let fc_w = store.push((0..d * ffn).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
        let fc_b = store.push((0..m * ffn).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect());
        let proj_w = store.push((0..ffn * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
        let proj_b = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect());
        let gelu_table = store.push((0..64).map(|j| from_i64((j as i64).pow(2) % 1000)).collect());
        // Reserve output/witness slots (values overwritten by forward_ops).
        let fc = store.push(vec![]);
        let fc_rem = store.push(vec![]);
        let act_idx = store.push_idx(vec![]);
        let act = store.push(vec![]);
        let proj2 = store.push(vec![]);
        let proj2_rem = store.push(vec![]);
        let out = store.push(vec![]);

        // fc = projection(x); act = gelu(fc); proj2 = projection(act); out = x + proj2.
        // The gelu index is derived from fc in the forward (same simplification as
        // ffn_chain: index = fc % 64, wired via the idx tensor).
        let ops = vec![
            Op::Projection { x, w: fc_w, bias: fc_b, out: fc, rem: fc_rem, m, k: d, n: ffn, shift },
            Op::Lookup { idx: act_idx, out: act, table: gelu_table },
            Op::Projection { x: act, w: proj_w, bias: proj_b, out: proj2, rem: proj2_rem, m, k: ffn, n: d, shift },
            Op::Add { a: x, b: proj2, c: out },
        ];

        // Fill the gelu idx tensor (derived from fc) before proving.
        let fc_val = {
            let h = crate::par::mm_par(store.get(x), store.get(fc_w), m, d, ffn, 64);
            (0..m * ffn)
                .map(|ij| from_i64(round_div(to_i64(h[ij]), 1i64 << shift) + to_i64(store.get(fc_b)[ij])))
                .collect::<Vec<_>>()
        };
        let idx: Vec<u32> = fc_val
            .iter()
            .map(|&v| ((to_i64(v).max(0)) as u64 % 64) as u32)
            .collect();
        store.idx[act_idx] = idx;

        let proof = prove_shard(&mut store, &ops, &[out], &mut rng);
        assert!(!proof.bound.is_empty(), "residual x must be bound");
        assert!(verify_shard(&store, &ops, &proof));

        // Tamper with an intermediate weight -> must fail.
        let mut bad = Store { v: store.v.clone(), idx: store.idx.clone() };
        bad.v[fc_w][0] = bad.v[fc_w][0] + Goldilocks::ONE;
        assert!(!verify_shard(&bad, &ops, &proof));
    }

    #[test]
    fn op_shard_softmax_roundtrip() {
        let mut rng = XorShift64::new(0x0F0F);
        let (m, n) = (4usize, 8usize);
        let table_len = 1usize << 8;
        let mut store = Store::new();
        let table = store.push((0..table_len).map(|j| from_i64((j % 255 + 1) as i64)).collect());
        let idx = store.push_idx((0..m * n).map(|_| (rng.next_u64() % table_len as u64) as u32).collect());
        let e = store.push(vec![]);
        let out = store.push(vec![]);
        let ops = vec![Op::Softmax { idx, e, out, table, m, n }];
        let proof = prove_shard(&mut store, &ops, &[out], &mut rng);
        assert!(verify_shard(&store, &ops, &proof));

        let mut bad = Store { v: store.v.clone(), idx: store.idx.clone() };
        bad.idx[idx][0] ^= 1;
        assert!(!verify_shard(&bad, &ops, &proof));
    }

    #[test]
    fn op_shard_layernorm_roundtrip() {
        let mut rng = XorShift64::new(0x1010);
        let (m, d) = (4usize, 8usize);
        let mut store = Store::new();
        let x = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64)).collect());
        let w = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 5) as i64 + 1)).collect());
        let b = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect());
        let rsqrt_table = store.push((0..(1usize << 8)).map(|j| from_i64((j % 255 + 1) as i64)).collect());
        let out = store.push(vec![]);
        let ops = vec![Op::Layernorm { x, w, b, out, rsqrt_table, m, d }];
        let proof = prove_shard(&mut store, &ops, &[out], &mut rng);
        assert!(verify_shard(&store, &ops, &proof));

        let mut bad = Store { v: store.v.clone(), idx: store.idx.clone() };
        bad.v[x][0] = bad.v[x][0] + Goldilocks::ONE;
        assert!(!verify_shard(&bad, &ops, &proof));
    }

    #[test]
    fn op_shard_dag_projection_chain() {
        let mut rng = XorShift64::new(0x1212);
        let (m, d, shift) = (4usize, 8usize, 8u32);
        let mut store = Store::new();
        let x = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        for _ in 0..4 {
            weights.push(store.push((0..d * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect()));
            biases.push(store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect()));
        }
        let mut outs = Vec::new();
        let mut rems = Vec::new();
        for _ in 0..4 {
            outs.push(store.push(vec![]));
            rems.push(store.push(vec![]));
        }
        let ops = vec![
            Op::Projection { x, w: weights[0], bias: biases[0], out: outs[0], rem: rems[0], m, k: d, n: d, shift },
            Op::Projection { x: outs[0], w: weights[1], bias: biases[1], out: outs[1], rem: rems[1], m, k: d, n: d, shift },
            Op::Projection { x: outs[1], w: weights[2], bias: biases[2], out: outs[2], rem: rems[2], m, k: d, n: d, shift },
            Op::Projection { x: outs[2], w: weights[3], bias: biases[3], out: outs[3], rem: rems[3], m, k: d, n: d, shift },
        ];

        let proof = prove_shard_dag(&mut store, &ops, 2, &mut rng);
        assert_eq!(proof.shards.len(), 2);
        assert_eq!(proof.cross_tensors.len(), 1, "only y2 spans the shard boundary");
        assert!(verify_shard_dag(&store, &ops, 2, &proof));

        let mut bad = Store { v: store.v.clone(), idx: store.idx.clone() };
        bad.v[weights[1]][0] = bad.v[weights[1]][0] + Goldilocks::ONE;
        assert!(!verify_shard_dag(&bad, &ops, 2, &proof));
    }

    #[test]
    fn committed_cross_bind_roundtrip() {
        use crate::committed::commit;
        use crate::field::PrimeCharacteristicRing;
        use crate::whir::Whir;

        let mut rng = XorShift64::new(0x1414);
        let n = 1usize << 6;
        let f: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let whir = Whir::new_testing(6);
        let c = commit(&whir, &f);

        let p0: Vec<Goldilocks> = (0..6).map(|_| rng.field()).collect();
        let p1: Vec<Goldilocks> = (0..6).map(|_| rng.field()).collect();
        let claims = vec![
            (p0.clone(), mle::eval(&f, &p0)),
            (p1.clone(), mle::eval(&f, &p1)),
        ];

        let proof = committed_cross_bind(&whir, &c, &f, &claims, &mut rng).expect("honest bind");
        assert_eq!(proof.coeffs.len(), 2);

        // A wrong claimed eval must fail the commitment check.
        let bad = vec![
            (p0.clone(), mle::eval(&f, &p0) + Goldilocks::ONE),
            (p1.clone(), mle::eval(&f, &p1)),
        ];
        assert!(committed_cross_bind(&whir, &c, &f, &bad, &mut rng).is_none());
    }

    #[test]
    fn committed_shard_dag_roundtrip() {
        use crate::field::PrimeCharacteristicRing;
        use crate::whir::Whir;

        let mut rng = XorShift64::new(0x1515);
        let (m, d, shift) = (4usize, 8usize, 8u32);
        let mut store = Store::new();
        let x = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
        let mut ws = Vec::new();
        let mut bs = Vec::new();
        for _ in 0..4 {
            ws.push(store.push((0..d * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect()));
            bs.push(store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect()));
        }
        let mut outs = Vec::new();
        let mut rems = Vec::new();
        for _ in 0..4 {
            outs.push(store.push(vec![]));
            rems.push(store.push(vec![]));
        }
        let ops = vec![
            Op::Projection { x, w: ws[0], bias: bs[0], out: outs[0], rem: rems[0], m, k: d, n: d, shift },
            Op::Projection { x: outs[0], w: ws[1], bias: bs[1], out: outs[1], rem: rems[1], m, k: d, n: d, shift },
            Op::Projection { x: outs[1], w: ws[2], bias: bs[2], out: outs[2], rem: rems[2], m, k: d, n: d, shift },
            Op::Projection { x: outs[2], w: ws[3], bias: bs[3], out: outs[3], rem: rems[3], m, k: d, n: d, shift },
        ];

        let whir = Whir::new_testing(5); // m*d = 32 = 2^5
        let proof = prove_committed_shard_dag(&mut store, &ops, 2, &whir, &mut rng);
        assert_eq!(proof.boundary_tensors.len(), 1);
        assert_eq!(proof.boundary_commitments.len(), 1);
        assert_eq!(proof.cross_binds.len(), 1);
        assert!(verify_committed_shard_dag(&store, &ops, 2, &whir, &proof));

        let mut bad = Store { v: store.v.clone(), idx: store.idx.clone() };
        bad.v[ws[1]][0] = bad.v[ws[1]][0] + Goldilocks::ONE;
        assert!(!verify_committed_shard_dag(&bad, &ops, 2, &whir, &proof));
    }

    #[test]
    fn global_weights_batch_commit_roundtrip() {
        use crate::field::PrimeCharacteristicRing;
        use crate::whir::Whir;

        let mut rng = XorShift64::new(0x1616);
        let n = 1usize << 6;
        let weights: Vec<Vec<Goldilocks>> = (0..4).map(|_| (0..n).map(|_| rng.field()).collect()).collect();
        let refs: Vec<&[Goldilocks]> = weights.iter().map(|w| w.as_slice()).collect();

        let whir = Whir::new_testing(6);
        let batch = commit_weights_batch(&whir, &refs);
        assert_eq!(batch.num_tables, 4);

        let point: Vec<Goldilocks> = (0..6).map(|_| rng.field()).collect();
        for (i, w) in weights.iter().enumerate() {
            assert!(verify_weight_batch(&batch, i, w, &point));
        }

        // A wrong tensor at a table index must fail.
        let mut bad = weights[0].clone();
        bad[0] = bad[0] + Goldilocks::ONE;
        assert!(!verify_weight_batch(&batch, 0, &bad, &point));
    }

    #[test]
    fn committed_projection_roundtrip() {
        use crate::field::PrimeCharacteristicRing;
        use crate::whir::Whir;

        let mut rng = XorShift64::new(0x1717);
        let (m, d, shift) = (4usize, 8usize, 8u32);
        let x: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let w: Vec<Goldilocks> = (0..d * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let bias: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect();
        let step = ProjectionStep::new(m, d, d, shift, w.clone(), bias.clone());
        let (out, rem) = projection_fwd(&x, &step);

        // Weights (d*d) and biases (m*d) are different sizes -> two batches.
        let whir_w = Whir::new_testing((d * d).trailing_zeros() as usize);
        let w_batch = commit_weights_batch(&whir_w, &[&w]);
        let whir_b = Whir::new_testing((m * d).trailing_zeros() as usize);
        let bias_batch = commit_weights_batch(&whir_b, &[&bias]);

        let proof = prove_projection_committed(
            &x, &w_batch, 0, &bias_batch, 0, &w, &bias, &out, &rem, m, d, d, shift, &mut rng,
        );
        assert!(verify_projection_committed(
            &proof, &x, &w_batch, 0, &bias_batch, 0, &w, &bias, &out, &rem, m, d, d, shift,
        ));

        // Wrong weight at the committed index must fail.
        let mut bad_w = w.clone();
        bad_w[0] = bad_w[0] + Goldilocks::ONE;
        assert!(!verify_projection_committed(
            &proof, &x, &w_batch, 0, &bias_batch, 0, &bad_w, &bias, &out, &rem, m, d, d, shift,
        ));
    }

    #[test]
    fn op_shard_matmul_roundtrip() {
        let mut rng = XorShift64::new(0x1818);
        let (m, k, n) = (4usize, 8usize, 8usize);
        let mut store = Store::new();
        let a = store.push((0..m * k).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
        let b = store.push((0..k * n).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
        let c = store.push(vec![]);
        let ops = vec![Op::MatMul { a, b, c, m, k, n }];
        let proof = prove_shard(&mut store, &ops, &[c], &mut rng);
        assert!(verify_shard(&store, &ops, &proof));

        let mut bad = Store { v: store.v.clone(), idx: store.idx.clone() };
        bad.v[a][0] = bad.v[a][0] + Goldilocks::ONE;
        assert!(!verify_shard(&bad, &ops, &proof));
    }

    #[test]
    fn op_shard_transpose_matmul_roundtrip() {
        // scores = Q @ K^T  (the core of attention): transpose K, then matmul.
        let mut rng = XorShift64::new(0x1919);
        let (m, kd) = (4usize, 8usize);
        let mut store = Store::new();
        let q_t = store.push((0..m * kd).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
        let k_t = store.push((0..m * kd).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
        let kt_t = store.push(vec![]);
        let scores_t = store.push(vec![]);
        let ops = vec![
            Op::Transpose { x: k_t, out: kt_t, m, k: kd },
            Op::MatMul { a: q_t, b: kt_t, c: scores_t, m, k: kd, n: m },
        ];
        let proof = prove_shard(&mut store, &ops, &[scores_t], &mut rng);
        assert!(verify_shard(&store, &ops, &proof));

        let mut bad = Store { v: store.v.clone(), idx: store.idx.clone() };
        bad.v[q_t][0] = bad.v[q_t][0] + Goldilocks::ONE;
        assert!(!verify_shard(&bad, &ops, &proof));
    }

    #[test]
    fn op_shard_attention_block_roundtrip() {
        // Single-head attention: Q/K/V projections -> scores = Q@K^T -> softmax
        // -> attn = probs@V, all folded into one g.
        use crate::field::PrimeCharacteristicRing;
        let mut rng = XorShift64::new(0x1A1A);
        let (m, d, dh, shift) = (4usize, 8usize, 4usize, 8u32);
        let table_len = 1usize << 8;
        let mut store = Store::new();

        let x = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
        let exp_table = store.push((0..table_len).map(|j| from_i64((j % 255 + 1) as i64)).collect());
        let mut ws = Vec::new();
        let mut bs = Vec::new();
        for _ in 0..3 {
            ws.push(store.push((0..d * dh).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect()));
            bs.push(store.push((0..m * dh).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect()));
        }
        let mut outs = Vec::new();
        let mut rems = Vec::new();
        for _ in 0..3 {
            outs.push(store.push(vec![]));
            rems.push(store.push(vec![]));
        }
        let kt = store.push(vec![]);
        let scores = store.push(vec![]);
        let idx = store.push_idx(vec![]);
        let e = store.push(vec![]);
        let probs = store.push(vec![]);
        let attn = store.push(vec![]);

        let ops = vec![
            Op::Projection { x, w: ws[0], bias: bs[0], out: outs[0], rem: rems[0], m, k: d, n: dh, shift },
            Op::Projection { x, w: ws[1], bias: bs[1], out: outs[1], rem: rems[1], m, k: d, n: dh, shift },
            Op::Projection { x, w: ws[2], bias: bs[2], out: outs[2], rem: rems[2], m, k: d, n: dh, shift },
            Op::Transpose { x: outs[1], out: kt, m, k: dh },
            Op::MatMul { a: outs[0], b: kt, c: scores, m, k: dh, n: m },
            Op::SoftmaxIndex { x: scores, out: idx, table_len },
            Op::Softmax { idx, e, out: probs, table: exp_table, m, n: m },
            Op::MatMul { a: probs, b: outs[2], c: attn, m, k: m, n: dh },
        ];

        let proof = prove_shard(&mut store, &ops, &[attn], &mut rng);
        assert!(verify_shard(&store, &ops, &proof));

        let mut bad = Store { v: store.v.clone(), idx: store.idx.clone() };
        bad.v[ws[0]][0] = bad.v[ws[0]][0] + Goldilocks::ONE;
        assert!(!verify_shard(&bad, &ops, &proof));
    }

    #[test]
    fn op_shard_multihead_attention_roundtrip() {
        // 2-head attention: per-head Q/K/V -> Q@K^T -> softmax -> probs@V ->
        // per-head output projection, then sum heads.
        use crate::field::PrimeCharacteristicRing;
        let mut rng = XorShift64::new(0x1B1B);
        let (heads, m, d, shift) = (2usize, 4usize, 8usize, 8u32);
        let dh = d / heads;
        let table_len = 1usize << 8;
        let mut store = Store::new();
        let x = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
        let exp_table = store.push((0..table_len).map(|j| from_i64((j % 255 + 1) as i64)).collect());

        let mut ops = Vec::new();
        let mut head_outs = Vec::new();
        for _h in 0..heads {
            let wq = store.push((0..d * dh).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
            let wk = store.push((0..d * dh).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
            let wv = store.push((0..d * dh).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
            let wo = store.push((0..dh * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
            let bq = store.push((0..m * dh).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect());
            let bk = store.push((0..m * dh).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect());
            let bv = store.push((0..m * dh).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect());
            let bo = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect());
            let q = store.push(vec![]);
            let k = store.push(vec![]);
            let v = store.push(vec![]);
            let qr = store.push(vec![]);
            let kr = store.push(vec![]);
            let vr = store.push(vec![]);
            let kt = store.push(vec![]);
            let scores = store.push(vec![]);
            let idx = store.push_idx(vec![]);
            let e = store.push(vec![]);
            let probs = store.push(vec![]);
            let attn = store.push(vec![]);
            let out_h = store.push(vec![]);
            let out_h_rem = store.push(vec![]);

            ops.push(Op::Projection { x, w: wq, bias: bq, out: q, rem: qr, m, k: d, n: dh, shift });
            ops.push(Op::Projection { x, w: wk, bias: bk, out: k, rem: kr, m, k: d, n: dh, shift });
            ops.push(Op::Projection { x, w: wv, bias: bv, out: v, rem: vr, m, k: d, n: dh, shift });
            ops.push(Op::Transpose { x: k, out: kt, m, k: dh });
            ops.push(Op::MatMul { a: q, b: kt, c: scores, m, k: dh, n: m });
            ops.push(Op::SoftmaxIndex { x: scores, out: idx, table_len });
            ops.push(Op::Softmax { idx, e, out: probs, table: exp_table, m, n: m });
            ops.push(Op::MatMul { a: probs, b: v, c: attn, m, k: m, n: dh });
            ops.push(Op::Projection { x: attn, w: wo, bias: bo, out: out_h, rem: out_h_rem, m, k: dh, n: d, shift });
            head_outs.push(out_h);
        }

        let mut acc = head_outs[0];
        for &h in &head_outs[1..] {
            let sum = store.push(vec![]);
            ops.push(Op::Add { a: acc, b: h, c: sum });
            acc = sum;
        }

        let proof = prove_shard(&mut store, &ops, &[acc], &mut rng);
        assert!(verify_shard(&store, &ops, &proof));

        let mut bad = Store { v: store.v.clone(), idx: store.idx.clone() };
        bad.v[x][0] = bad.v[x][0] + Goldilocks::ONE;
        assert!(!verify_shard(&bad, &ops, &proof));
    }

    #[test]
    fn op_shard_prenorm_ffn_roundtrip() {
        // GPT-2 pre-norm FFN block: h = layernorm(x) -> fc = proj(h) ->
        // act = gelu(fc) -> proj2 = proj(act) -> out = x + proj2.
        use crate::field::PrimeCharacteristicRing;
        let mut rng = XorShift64::new(0x1C1C);
        let (m, d, ffn, shift) = (4usize, 8usize, 16usize, 8u32);
        let mut store = Store::new();
        let x = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
        let ln_w = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 5) as i64 + 1)).collect());
        let ln_b = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect());
        let rsqrt_table = store.push((0..(1usize << 8)).map(|j| from_i64((j % 255 + 1) as i64)).collect());
        let fc_w = store.push((0..d * ffn).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
        let fc_b = store.push((0..m * ffn).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect());
        let proj_w = store.push((0..ffn * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
        let proj_b = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect());
        let gelu_table = store.push((0..64).map(|j| from_i64((j as i64).pow(2) % 1000)).collect());

        let h = store.push(vec![]);
        let fc = store.push(vec![]);
        let fc_rem = store.push(vec![]);
        let gelu_idx = store.push_idx(vec![]);
        let act = store.push(vec![]);
        let proj2 = store.push(vec![]);
        let proj2_rem = store.push(vec![]);
        let out = store.push(vec![]);

        let ops = vec![
            Op::Layernorm { x, w: ln_w, b: ln_b, out: h, rsqrt_table, m, d },
            Op::Projection { x: h, w: fc_w, bias: fc_b, out: fc, rem: fc_rem, m, k: d, n: ffn, shift },
            Op::SoftmaxIndex { x: fc, out: gelu_idx, table_len: 64 },
            Op::Lookup { idx: gelu_idx, out: act, table: gelu_table },
            Op::Projection { x: act, w: proj_w, bias: proj_b, out: proj2, rem: proj2_rem, m, k: ffn, n: d, shift },
            Op::Add { a: x, b: proj2, c: out },
        ];

        let proof = prove_shard(&mut store, &ops, &[out], &mut rng);
        assert!(verify_shard(&store, &ops, &proof));

        let mut bad = Store { v: store.v.clone(), idx: store.idx.clone() };
        bad.v[ln_w][0] = bad.v[ln_w][0] + Goldilocks::ONE;
        assert!(!verify_shard(&bad, &ops, &proof));
    }

    #[test]
    fn build_gpt2_layer_roundtrip() {
        use crate::field::PrimeCharacteristicRing;
        let mut rng = XorShift64::new(0x1D1D);
        let (m, d, ffn, heads, shift) = (4usize, 8usize, 16usize, 2usize, 8u32);
        let table_len = 1usize << 8;
        let mut store = Store::new();
        let x = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
        let exp_table = store.push((0..table_len).map(|j| from_i64((j % 255 + 1) as i64)).collect());
        let gelu_table = store.push((0..64).map(|j| from_i64((j as i64).pow(2) % 1000)).collect());
        let rsqrt_table = store.push((0..table_len).map(|j| from_i64((j % 255 + 1) as i64)).collect());

        let (ops, out) = build_gpt2_layer(
            &mut store, x, heads, m, d, ffn, shift, exp_table, gelu_table, rsqrt_table, &mut rng,
        );
        let proof = prove_shard(&mut store, &ops, &[out], &mut rng);
        assert!(verify_shard(&store, &ops, &proof));

        let mut bad = Store { v: store.v.clone(), idx: store.idx.clone() };
        bad.v[x][0] = bad.v[x][0] + Goldilocks::ONE;
        assert!(!verify_shard(&bad, &ops, &proof));
    }

    #[test]
    fn rsqrt_index_piecewise() {
        // Below the fine boundary: raw index = round(var / 2^18).
        assert_eq!(rsqrt_index(0), 0);
        assert_eq!(rsqrt_index((1 << 18) / 2), 1);
        assert_eq!(rsqrt_index((1 << 20) << 18), 1 << 20);
        // Above the boundary: step 2^8.
        let var_large = ((1 << 20) + 256) << 18;
        assert_eq!(rsqrt_index(var_large), (1 << 20) + 1);
    }
}
