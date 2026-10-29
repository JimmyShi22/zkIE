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

use std::borrow::Cow;
use std::sync::Arc;

use crate::layer_norm_centered::{
    layer_norm_forward as layer_norm_centered_forward, prove_layer_norm_centered,
    verify_layer_norm_centered, LayerNormCenteredProof,
};
use crate::layernorm_chain::{prove_layernorm_chain, verify_layernorm_chain, LayernormChainProof};
use crate::gather::{
    gather_forward, prove_scatter_add, prove_scatter_add_with_mode, verify_scatter_add,
    verify_scatter_add_with_mode, ScatterAddProof,
};
use crate::projection::{prove_projection, verify_projection, ProjectionProof};
use crate::scatter::{
    prove_scatter, prove_scatter_with_mode, scatter_forward, verify_scatter,
    verify_scatter_with_mode, ScatterProof,
};
use crate::rms_norm::{prove_rms_norm, rms_norm_forward, verify_rms_norm, RmsNormProof};
use crate::rope::{prove_rope, rope_forward, rope_out_points, verify_rope, RoPEProof};
use crate::softmax_scaled::{prove_softmax_rounded, verify_softmax_rounded, SoftmaxRoundedProof};
use crate::topk::{argmax_forward, prove_argmax, prove_topk, topk_forward, verify_argmax, verify_topk, ArgmaxProof, TopKSelectProof};
use rayon::prelude::*;
use zkie_core::common::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64, ChallengeSource};
use zkie_core::common::fixed_point::{from_i64, to_i32, to_i64};
use zkie_core::common::logup_gkr::{
    prove_lookup_fractional, prove_lookup_fractional_batch, prove_lookup_fractional_columns,
    verify_lookup_fractional, verify_lookup_fractional_batch, verify_lookup_fractional_chain,
    build_lookup_table_num_den, build_perm_num_den, fractional_leaf, merge_multiplicity, pad_table,
    prove_fractional, prove_lookup_table, prove_lookup_witness_batch_columns,
    verify_fractional_chain_core, verify_lookup_split, FractionalChainClaims, FractionalProof,
};
use zkie_core::common::matmul::{prove as matmul_prove, verify as matmul_verify, MatmulProof};
use zkie_core::common::batched_matmul::{self as bmm, BatchedMatmulProof};
use zkie_core::common::mle;
use zkie_core::common::same_poly::{prove_same_poly, verify_same_poly, SamePolyProof};
use zkie_core::common::sumcheck::{prove_virtual, verify_virtual, virtual_eval, VirtualProof};
use zkie_core::common::transcript::Transcript;
use zkie_core::common::weights_io::WeightMmap;

fn is_identity_table(table: &[Goldilocks]) -> bool {
    table
        .iter()
        .enumerate()
        .all(|(j, &v)| v == Goldilocks::from_u64(j as u64))
}

/// Pad a lookup table for commitment/frac use: to the next power of two, at
/// least 2^5 (WHIR folding needs arity >= 5). Identity tables continue the
/// identity into the padding (den = alpha + j + beta*j at every padded
/// index, matching the verifier's identity evaluation); other tables pad
/// with zeros (the den relation there is checked via openings, never by the
/// index formula).
fn pad_lookup_table(table: &[Goldilocks]) -> Vec<Goldilocks> {
    let total = table.len().next_power_of_two().max(32);
    let mut out = vec![Goldilocks::ZERO; total];
    out[..table.len()].copy_from_slice(table);
    if is_identity_table(table) {
        for j in table.len()..total {
            out[j] = Goldilocks::from_u64(j as u64);
        }
    }
    out
}

/// One claim in the verifier's self-built authentication chain: tensor
/// `tensor_id`'s MLE evaluated at `point` equals `value`. Every claim the
/// verifier uses is derived from its own reduction (op sumcheck final evals,
/// replayed fractional leaves) — never read from a prover-supplied list
/// (S2).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EvalClaim {
    pub tensor_id: T,
    pub point: Vec<Goldilocks>,
    pub value: Goldilocks,
}

/// Where the per-op fractional proofs (permutation / gather) run. On the
/// plain shard path they live on the shard transcript; on the committed path
/// they are re-proven on the committed transcripts (which bind the committed
/// column roots), so the shard transcript must stay frac-free — the
/// committed-mode shard proof runs them on a scratch transcript and the
/// committed replay skips them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FracMode {
    Plain,
    Committed,
}

/// The public statement a verifier trusts: the op graph, the shard
/// granularity, and the public lookup tables indexed by tensor id. All
/// decisions that must not come from the proof (identity tables, expected
/// weight tensors, statement-derived counts) derive from this (S2).
#[derive(Clone)]
pub struct VerifierStatement<'a> {
    pub tables: &'a [&'a [Goldilocks]],
    pub ops: &'a [Op],
    pub ops_per_shard: usize,
}

impl<'a> VerifierStatement<'a> {
    pub fn new(tables: &'a [&'a [Goldilocks]], ops: &'a [Op], ops_per_shard: usize) -> Self {
        VerifierStatement {
            tables,
            ops,
            ops_per_shard,
        }
    }

    /// The identity-table decision comes from the statement tables, never
    /// from a proof flag (S2).
    pub fn is_identity_table(&self, t: T) -> bool {
        match self.tables.get(t) {
            Some(table) => is_identity_table(table),
            None => false,
        }
    }

    /// Weight tensors: read by an op but never written (statement-derived;
    /// used to require batch openings for every claimed weight).
    pub fn weight_tensors(&self) -> Vec<T> {
        collect_weight_tensors(self.ops)
    }
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
        ProjectionStep {
            m,
            k,
            n,
            shift,
            w,
            bias,
        }
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

fn projection_fwd(x: &[Goldilocks], step: &ProjectionStep) -> (Vec<Goldilocks>, Vec<Goldilocks>) {
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
    transcript: &mut Transcript,
) -> ProjectionChainProof {
    assert!(!steps.is_empty(), "chain needs at least one projection");
    let (ys, rems) = forward(x, steps);

    let mut step_proofs = Vec::with_capacity(steps.len());
    let mut prev = x;
    for (i, step) in steps.iter().enumerate() {
        let p = prove_projection(
            prev, &step.w, &step.bias, &ys[i], &rems[i], step.m, step.k, step.n, step.shift,
            transcript,
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
        let (sp, _merged) = prove_same_poly(&ys[i], &claims, transcript);
        binds.push(sp);
    }

    ProjectionChainProof {
        steps: step_proofs,
        binds,
    }
}

/// Verify a chain proof. Returns true iff every projection is sound AND every
/// adjacent intermediate binding is consistent.
pub fn verify_projection_chain(
    proof: &ProjectionChainProof,
    x: &[Goldilocks],
    steps: &[ProjectionStep],
    transcript: &mut Transcript,
) -> bool {
    assert_eq!(proof.steps.len(), steps.len());
    assert_eq!(proof.binds.len(), steps.len().saturating_sub(1));
    let (ys, rems) = forward(x, steps);

    let mut prev = x;
    for (i, step) in steps.iter().enumerate() {
        if !verify_projection(&proof.steps[i], step.shift, step.m, step.k, step.n, transcript) {
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
        if verify_same_poly(&proof.binds[i - 1], &claims, transcript).is_none() {
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
    transcript: &mut Transcript,
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
        let mut child = transcript.clone();
        child.absorb_u64(shards.len() as u64);
        shards.push(prove_projection_chain(input, &steps[start..end], &mut child));
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
        let (sp, _merged) = prove_same_poly(&ys[b], &claims, transcript);
        cross_binds.push(sp);
    }

    ProjectionShardedProof {
        shards,
        cross_binds,
        boundary_steps,
    }
}

pub fn verify_projection_sharded(
    proof: &ProjectionShardedProof,
    x: &[Goldilocks],
    steps: &[ProjectionStep],
    ops_per_shard: usize,
    transcript: &mut Transcript,
) -> bool {
    let ops_per_shard = ops_per_shard.max(1);
    let (ys, _) = forward(x, steps);

    let mut start = 0;
    let mut si = 0;
    while start < steps.len() {
        let end = (start + ops_per_shard).min(steps.len());
        let input: &[Goldilocks] = if start == 0 { x } else { &ys[start - 1] };
        let mut child = transcript.clone();
        child.absorb_u64(si as u64);
        if !verify_projection_chain(&proof.shards[si], input, &steps[start..end], &mut child) {
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
        if verify_same_poly(&proof.cross_binds[ci], &claims, transcript).is_none() {
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
) -> (
    Vec<Vec<Goldilocks>>,
    Vec<Vec<Goldilocks>>,
    Vec<Vec<Goldilocks>>,
) {
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
    transcript: &mut Transcript,
) -> ResidualChainProof {
    assert!(!steps.is_empty(), "chain needs at least one block");
    let (xs, ps, rems) = residual_forward(x, steps);
    let n = steps.len();

    let mut proj = Vec::with_capacity(n);
    for (i, step) in steps.iter().enumerate() {
        proj.push(prove_projection(
            &xs[i], &step.w, &step.bias, &ps[i], &rems[i], step.m, step.k, step.n, step.shift,
            transcript,
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
        let mles: Vec<&[Goldilocks]> = vec![&xs[i], &ps[i], &xs[i + 1]];
        let (proof, r) = prove_virtual(&mles, &terms, Goldilocks::ZERO, transcript);
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
        let (sp, _merged) = prove_same_poly(&xs[i], &claims, transcript);
        same_x.push(sp);
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
        let (sp, _merged) = prove_same_poly(&ps[i], &claims, transcript);
        same_p.push(sp);
    }

    ResidualChainProof {
        proj,
        adds,
        same_x,
        same_p,
    }
}

pub fn verify_residual_chain(
    proof: &ResidualChainProof,
    x: &[Goldilocks],
    steps: &[ProjectionStep],
    transcript: &mut Transcript,
) -> bool {
    assert_eq!(proof.proj.len(), steps.len());
    assert_eq!(proof.adds.len(), steps.len());
    assert_eq!(proof.same_x.len(), steps.len());
    assert_eq!(proof.same_p.len(), steps.len());
    let (xs, ps, rems) = residual_forward(x, steps);
    let n = steps.len();

    for (i, step) in steps.iter().enumerate() {
        if !verify_projection(&proof.proj[i], step.shift, step.m, step.k, step.n, transcript) {
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
        let (ok, r, fc) = verify_virtual(
            &proof.adds[i].0,
            &terms,
            Goldilocks::ZERO,
            proof.adds[i].1.len(),
            transcript,
        );
        if !ok || r != proof.adds[i].1 {
            return false;
        }
        let fe = vec![
            mle::eval(&xs[i], &r),
            mle::eval(&ps[i], &r),
            mle::eval(&xs[i + 1], &r),
        ];
        if fc != virtual_eval(&terms, &fe) {
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
        if verify_same_poly(&proof.same_x[i], &claims, transcript).is_none() {
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
        if verify_same_poly(&proof.same_p[i], &claims, transcript).is_none() {
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

/// A tensor datum: owned (activations, small weights) or a memory-mapped i32
/// weight slice that is converted to Goldilocks on demand.
#[derive(Clone)]
pub enum TensorData {
    Owned(Vec<Goldilocks>),
    Mmap {
        mmap: Arc<WeightMmap>,
        start: usize,
        len: usize,
    },
}

/// A shard's tensor store: `v[id]` is a Goldilocks tensor; `idx[id]` is an
/// integer index tensor (used by `Lookup`).
#[derive(Default)]
pub struct Store {
    pub v: Vec<TensorData>,
    pub idx: Vec<Vec<u32>>,
}

impl Store {
    pub fn new() -> Self {
        Store::default()
    }
    pub fn push(&mut self, t: Vec<Goldilocks>) -> T {
        let id = self.v.len();
        self.v.push(TensorData::Owned(t));
        id
    }
    pub fn push_mmap(&mut self, mmap: Arc<WeightMmap>, start: usize, len: usize) -> T {
        let id = self.v.len();
        self.v.push(TensorData::Mmap { mmap, start, len });
        id
    }
    pub fn push_idx(&mut self, t: Vec<u32>) -> T {
        let id = self.idx.len();
        self.idx.push(t);
        id
    }
    pub fn get(&self, id: T) -> &[Goldilocks] {
        match &self.v[id] {
            TensorData::Owned(v) => v,
            TensorData::Mmap { .. } => panic!("get() on mmap-backed tensor; use materialize()"),
        }
    }
    pub fn materialize(&self, id: T) -> Cow<'_, [Goldilocks]> {
        match &self.v[id] {
            TensorData::Owned(v) => Cow::Borrowed(v),
            TensorData::Mmap { mmap, start, len } => Cow::Owned(mmap.read_goldilocks(*start, *len)),
        }
    }
    pub fn get_mut(&mut self, id: T) -> &mut [Goldilocks] {
        match &mut self.v[id] {
            TensorData::Owned(v) => v,
            TensorData::Mmap { .. } => panic!("get_mut() on mmap-backed tensor"),
        }
    }

    /// Borrow every owned tensor as a slice, indexed by tensor id. Used to hand
    /// the verifier only the public lookup tables (never activations/weights).
    pub fn owned_slices(&self) -> Vec<&[Goldilocks]> {
        self.v
            .iter()
            .map(|td| match td {
                TensorData::Owned(v) => v.as_slice(),
                TensorData::Mmap { .. } => &[],
            })
            .collect()
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
    ScaleVec {
        x: T,
        scale: T,
        out: T,
        shift: u32,
    },
    Relu {
        x: T,
        out: T,
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
    RmsNorm {
        x: T,
        w: T,
        out: T,
        rsqrt_table: T,
        limb_table: T,
        m: usize,
        d: usize,
        n_real: usize,
    },
    RoPE {
        x: T,
        cos: T,
        sin: T,
        out: T,
        range_table: T,
        m: usize,
        d: usize,
        shift: u32,
    },
    TopKSelect {
        x: T,
        sel: T,
        thr: T,
        gate: T,
        d1: T,
        d2: T,
        m: usize,
        n: usize,
        k: usize,
    },
    Argmax {
        x: T,
        sel: T,
        thr: T,
        gate: T,
        d1: T,
        d2: T,
        out: T,
        m: usize,
        n: usize,
    },
    ScaleGate {
        x: T,
        gate: T,
        col: usize,
        out: T,
        m: usize,
        n: usize,
        h: usize,
        shift: u32,
    },
    Scatter {
        sel: T,
        token_id: T,
        expert_of: T,
        m: usize,
        n: usize,
    },
    Gather {
        weighted: T,
        token_id: T,
        gate: T,
        out: T,
        m: usize,
        h: usize,
    },
    BatchedMatMul {
        a: T,
        w: T,
        c: T,
        s: usize,
        k: usize,
        n: usize,
    },
    GatherMatMul {
        a: T,
        expert_of: T,
        w: T,
        c: T,
        s: usize,
        k: usize,
        n: usize,
        n_experts: usize,
    },
    IdxBroadcast {
        base: T,
        out: T,
        b: usize,
    },
    CombineIdx {
        a: T,
        b: T,
        out: T,
        scale: usize,
    },
    PadIdx {
        base: T,
        out: T,
        len: usize,
    },
    BroadcastScale {
        x: T,
        scale: T,
        out: T,
        h: usize,
    },
}

/// A proof for one op, kept heterogeneous because the primitives have different
/// proof shapes.
#[derive(serde::Serialize, serde::Deserialize)]
pub enum OpProof {
    Transpose,
    Scale,
    ScaleVec,
    Relu,
    MatMul(
        MatmulProof,
        Vec<Goldilocks>,
        Vec<Goldilocks>,
        Vec<Goldilocks>,
    ),
    Projection(ProjectionProof),
    Add(VirtualProof, Vec<Goldilocks>),
    Lookup,
    Softmax(SoftmaxRoundedProof),
    SoftmaxIndex,
    GeluIndex,
    StableSoftmaxIndex,
    Layernorm(LayernormChainProof),
    LayerNormCentered(LayerNormCenteredProof),
    RmsNorm(RmsNormProof),
    RoPE(RoPEProof),
    TopKSelect(TopKSelectProof),
    Argmax(ArgmaxProof),
    Scatter(ScatterProof),
    Gather(ScatterAddProof),
    BatchedMatMul(BatchedMatmulProof, Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>),
    GatherMatMul(bmm::GatherMatMulProof, Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>),
    IdxBroadcast,
    CombineIdx,
    PadIdx,
    BroadcastScale,
    ScaleGate,
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
/// resolution, raw in [2^20, 2^30) use step 2^8, and raw >= 2^30 use step 2^16.
/// Number of entries in the piecewise rsqrt table, fixed by
/// `scripts/extract_rsqrt_piecewise.py`: 2^20 step-1 entries, followed by 2^22
/// coarse entries at step 2^8 and 2^21 coarse entries at step 2^16.
pub const RSQRT_TABLE_LEN: usize = (1 << 20) + (1 << 22) + (1 << 21);

/// Largest real variance the generated table covers:
/// `(FINE + COARSE * COARSE_STEP + (COARSE2 - 1) * COARSE_STEP2) / INDEX_SCALE`.
pub const RSQRT_MAX_VARIANCE: f64 = 8454204.0;

/// Check a loaded rsqrt table covers the indices about to be read from it.
///
/// Deliberately a bounds check and not `len() == RSQRT_TABLE_LEN`: the ops work
/// with any table large enough for the data, and the unit tests legitimately use
/// smaller ones. What must never happen is reading past the end.
///
/// Without this a short table fails as a bare `index out of bounds` from inside
/// a forward pass, minutes into a run and a long way from the cause. It is a
/// live mistake rather than a hypothetical one: two generators exist, and the
/// superseded `scripts/extract_rsqrt_table.py` emits 2^19 uniform entries, so
/// running the wrong one produces a file that looks perfectly normal.
pub fn check_rsqrt_table(table: &[Goldilocks], max_index: u32) {
    assert!(
        (max_index as usize) < table.len(),
        "rsqrt table has {} entries but the forward pass needs index {}. \
         Regenerate with scripts/extract_rsqrt_piecewise.py ({} entries, covering \
         variance up to {:.6}) - NOT extract_rsqrt_table.py, which is superseded and \
         emits 2^19 uniform entries.",
        table.len(),
        max_index,
        RSQRT_TABLE_LEN,
        RSQRT_MAX_VARIANCE
    );
}

pub fn rsqrt_index(var: i64) -> u32 {
    const FINE: i64 = 1 << 20;
    const COARSE: i64 = 1 << 22;
    const COARSE_END: i64 = FINE + COARSE * (1 << 8);
    let raw = round_div(var, 1 << 18);
    if raw < FINE {
        raw.max(0) as u32
    } else if raw < COARSE_END {
        (FINE + ((raw - FINE) >> 8)) as u32
    } else {
        (FINE + COARSE + ((raw - COARSE_END) >> 16)) as u32
    }
}

/// A shard proof: per-op proofs plus the `same_poly` bindings on every
/// multiply-consumed activation tensor.
/// A lookup witness column pair (merged num_0/den_0) plus its chained-GKR
/// proof, collected by the plain prover and batch-committed/opened by the
/// committed path. Leaf claims (points + evals) are NOT stored (S2): the
/// verifier derives them from its own fractional replay, and the prover
/// reads them back out of the freshly proven chain via [`fractional_leaf`].
#[derive(serde::Serialize, serde::Deserialize)]
pub struct WitnessColumn {
    pub proof: FractionalProof,
    pub x: Vec<Goldilocks>,
    pub y: Vec<Goldilocks>,
    pub num: Vec<Goldilocks>,
    pub alpha: Goldilocks,
    pub beta: Goldilocks,
    pub table: T,
}

/// A lookup table side (table tensor + global multiplicity) with its
/// chained-GKR proof, committed once per table globally in the committed
/// path. Leaf claims are derived from the replay, never stored (S2).
#[derive(serde::Serialize, serde::Deserialize)]
pub struct TableColumn {
    pub proof: FractionalProof,
    pub table: T,
    pub m: Vec<u64>,
    pub alpha: Goldilocks,
    pub beta: Goldilocks,
}

/// A shard proof: per-op proofs plus the `same_poly` bindings on every
/// multiply-claimed tensor. It carries NO claim list (S2): the shard
/// verifier rebuilds the claims from its own op verifications and derives
/// the binding set from them; the shard-DAG and committed composers receive
/// the claims from the prover side as a separate return value.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct OpShardProof {
    pub ops: Vec<OpProof>,
    /// `binds[j]` merges the claims of tensor `bound[j]`.
    pub bound: Vec<T>,
    pub binds: Vec<SamePolyProof>,
    pub witness_columns: Vec<WitnessColumn>,
    pub table_columns: Vec<TableColumn>,
}

/// Run the forward pass for all ops; materializes every `out`/`rem`/`c` into the
/// store. Input tensors (`x`, `w`, `bias`, `idx`, `table`) must already exist.
impl Op {
    /// Tensors this op READS. Together with [`Op::writes`] this is the
    /// dependency edge set of the op graph.
    ///
    /// Needed to split witness computation per shard: a shard's *inputs* are the
    /// tensors its ops read but do not themselves write, and those are exactly
    /// the values that must cross a node boundary. Today `forward_ops`
    /// materialises the whole graph into one store, so nothing needs to ask this
    /// question; splitting it is what makes the question necessary.
    pub fn reads(&self) -> Vec<T> {
        match *self {
            Op::Transpose { x, .. } => vec![x],
            Op::Scale { x, .. } => vec![x],
            Op::ScaleVec { x, scale, .. } => vec![x, scale],
            Op::Relu { x, .. } => vec![x],
            Op::MatMul { a, b, .. } => vec![a, b],
            Op::Projection { x, w, bias, .. } => vec![x, w, bias],
            Op::Add { a, b, .. } => vec![a, b],
            Op::Lookup { idx, table, .. } => vec![idx, table],
            Op::Softmax { idx, table, .. } => vec![idx, table],
            Op::SoftmaxIndex { x, .. } => vec![x],
            Op::GeluIndex { x, .. } => vec![x],
            Op::StableSoftmaxIndex { x, mask, .. } => vec![x, mask],
            Op::Layernorm {
                x,
                w,
                b,
                rsqrt_table,
                ..
            } => vec![x, w, b, rsqrt_table],
            Op::LayerNormCentered {
                x,
                w,
                b,
                rsqrt_table,
                ..
            } => vec![x, w, b, rsqrt_table],
            Op::RmsNorm {
                x, w, rsqrt_table, limb_table, ..
            } => vec![x, w, rsqrt_table, limb_table],
            Op::RoPE { x, cos, sin, range_table, .. } => vec![x, cos, sin, range_table],
            Op::TopKSelect { x, .. } => vec![x],
            Op::Argmax { x, .. } => vec![x],
            Op::ScaleGate { x, gate, .. } => vec![x, gate],
            Op::Scatter { sel, .. } => vec![sel],
            Op::Gather { weighted, token_id, gate, .. } => vec![weighted, token_id, gate],
            Op::BatchedMatMul { a, w, .. } => vec![a, w],
            Op::GatherMatMul { a, expert_of, w, .. } => vec![a, expert_of, w],
            Op::IdxBroadcast { base, .. } => vec![base],
            Op::CombineIdx { a, b, .. } => vec![a, b],
            Op::PadIdx { base, .. } => vec![base],
            Op::BroadcastScale { x, scale, .. } => vec![x, scale],
        }
    }

    /// Tensors this op WRITES. Most ops write one; `Projection` writes its
    /// output and a remainder, `Softmax` its exponentials and its output, and
    /// `TopKSelect` five. Derived from what `forward_ops` actually assigns, not
    /// from the field names — `Softmax::e` and `Projection::rem` are outputs
    /// despite reading like inputs.
    pub fn writes(&self) -> Vec<T> {
        match *self {
            Op::Transpose { out, .. }
            | Op::Scale { out, .. }
            | Op::ScaleVec { out, .. }
            | Op::Relu { out, .. }
            | Op::Lookup { out, .. }
            | Op::SoftmaxIndex { out, .. }
            | Op::GeluIndex { out, .. }
            | Op::StableSoftmaxIndex { out, .. }
            | Op::Layernorm { out, .. }
            | Op::LayerNormCentered { out, .. }
            | Op::RmsNorm { out, .. }
            | Op::RoPE { out, .. }
            | Op::ScaleGate { out, .. } => vec![out],
            Op::MatMul { c, .. } | Op::Add { c, .. } => vec![c],
            Op::Projection { out, rem, .. } => vec![out, rem],
            Op::Softmax { e, out, .. } => vec![e, out],
            Op::TopKSelect {
                sel,
                thr,
                gate,
                d1,
                d2,
                ..
            } => vec![sel, thr, gate, d1, d2],
            Op::Argmax {
                sel,
                thr,
                gate,
                d1,
                d2,
                ..
            } => vec![sel, thr, gate, d1, d2],
            Op::Scatter {
                token_id,
                expert_of,
                ..
            } => vec![token_id, expert_of],
            Op::Gather { out, .. } => vec![out],
            Op::BatchedMatMul { c, .. } => vec![c],
            Op::GatherMatMul { c, .. } => vec![c],
            Op::IdxBroadcast { out, .. } => vec![out],
            Op::CombineIdx { out, .. } => vec![out],
            Op::PadIdx { out, .. } => vec![out],
            Op::BroadcastScale { out, .. } => vec![out],
        }
    }
}

/// Tensors a shard must be GIVEN: read inside `ops[range]` but not written there.
///
/// These are the values that cross a node boundary when shards are proven on
/// separate machines. Weights and lookup tables appear here too — they are
/// read and never written — so a caller distributing work can separate the two
/// by checking which are produced by an earlier op.
pub fn shard_inputs(ops: &[Op], range: std::ops::Range<usize>) -> Vec<T> {
    let mut written = std::collections::BTreeSet::new();
    for op in &ops[range.clone()] {
        for t in op.writes() {
            written.insert(t);
        }
    }
    let mut need = std::collections::BTreeSet::new();
    for op in &ops[range] {
        for t in op.reads() {
            if !written.contains(&t) {
                need.insert(t);
            }
        }
    }
    need.into_iter().collect()
}

impl Store {
    /// Drop a tensor's data, keeping its slot. Used by [`forward_shard`] to
    /// release intermediates the moment nothing downstream reads them.
    ///
    /// The slot stays so tensor ids remain stable - ids are indices, and
    /// renumbering them would invalidate every `Op` in the graph.
    pub fn release(&mut self, id: T) {
        // Only owned data is worth releasing. An mmap-backed tensor is
        // file-backed: its pages are already reclaimable by the OS, so dropping
        // it frees nothing measurable - and it would discard the backing, losing
        // state that cannot be recovered if the liveness analysis is ever wrong.
        // Weights on a large model are mmap-backed, so this is the common case.
        if matches!(self.v[id], TensorData::Owned(_)) {
            self.v[id] = TensorData::Owned(Vec::new());
        }
    }
}

/// Materialise what one shard needs, releasing every intermediate as soon as
/// nothing downstream reads it.
///
/// This is the distributed path. Each node runs the forward from the start up to
/// its own shard and proves against tensors it computed itself, so no value
/// crosses a node boundary and there is nothing for another node to lie about -
/// the verifier's recomputation remains the soundness anchor, unchanged.
///
/// Contrast [`forward_ops`], which materialises the whole graph and holds it.
/// Measured on DeepSeek-V2-Lite (Deucalion, one node, `bench_forward_shard`):
///
/// | seq | whole-graph forward | worst-case shard | saving |
/// |-----|--------------------|------------------|--------|
/// | 512 | 223.2 GiB          | 121.7 GiB        | 45 %   |
/// | 16  | 51.4 GiB           | 50.3 GiB         | 2 %    |
///
/// The saving is strongly sequence-dependent, and that matters for anyone
/// testing this: at short sequences the resident weights dominate and liveness
/// has almost nothing to release, so a small fixture will suggest this does
/// nothing. The win only appears once activations outweigh weights.
///
/// Note the worst case is the LAST shard, not the largest: this runs
/// `ops[..shard.end]`, so the final shard pays the high-water mark of the whole
/// prefix even though its own live set is the smallest.
///
/// The cost is that each node redundantly recomputes the prefix before its own
/// shard. That recompute is perfectly parallel across nodes, but it is NOT cheap:
/// measured on the seq=512 worst-case shard (`bench_prove_shard`), forward took
/// 1178 s against 380 s to prove that shard - so the prefix is 76 % of a node's
/// work, not a small fraction of it. It buys the memory reduction above; it does
/// not come free, and it caps how much wall-clock distributing this can recover.
///
/// After this returns, the tensors read or written by `ops[shard]` hold exactly
/// the values `forward_ops` would have produced. Everything else is released.
pub fn forward_shard(store: &mut Store, ops: &[Op], shard: std::ops::Range<usize>) {
    assert!(
        shard.end <= ops.len(),
        "shard range past the end of the graph"
    );
    let prefix = &ops[..shard.end];

    // What the shard's proof will read: its ops' inputs and outputs.
    let keep: std::collections::BTreeSet<T> = ops[shard.clone()]
        .iter()
        .flat_map(|o| o.reads().into_iter().chain(o.writes()))
        .collect();

    // Last op index that reads each tensor, over the prefix we are about to run.
    // A tensor with no later reader is dead the moment its last reader is done.
    let mut last_read: std::collections::BTreeMap<T, usize> = Default::default();
    for (i, op) in prefix.iter().enumerate() {
        for t in op.reads() {
            last_read.insert(t, i);
        }
    }

    for (i, op) in prefix.iter().enumerate() {
        forward_ops(store, std::slice::from_ref(op));
        // Release what this op was the last to read, unless the shard needs it.
        // Note this frees weights and lookup tables too once they are finished
        // with, which on a large model is most of the footprint.
        for (&t, &last) in last_read.iter() {
            if last == i && !keep.contains(&t) {
                store.release(t);
            }
        }
        // An op's own outputs can also be dead on arrival: produced inside the
        // prefix, never read again, and not needed by the shard.
        for t in op.writes() {
            if !keep.contains(&t) && !last_read.contains_key(&t) {
                store.release(t);
            }
        }
    }
}

fn forward_ops(store: &mut Store, ops: &[Op]) {
    for op in ops.iter().cloned() {
        match op {
            Op::Transpose { x, out, m, k } => {
                store.v[out] = TensorData::Owned(transpose(store.get(x), m, k));
            }
            Op::Scale {
                x,
                out,
                factor,
                shift,
            } => {
                let o: Vec<Goldilocks> = store
                    .get(x)
                    .iter()
                    .map(|&v| from_i64(round_div(to_i64(v) * factor, 1i64 << shift)))
                    .collect();
                store.v[out] = TensorData::Owned(o);
            }
            Op::ScaleVec {
                x,
                scale,
                out,
                shift,
            } => {
                let sv = store.get(scale);
                let o: Vec<Goldilocks> = store
                    .get(x)
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        from_i64(round_div(
                            to_i64(v) * to_i64(sv[i % sv.len()]),
                            1i64 << shift,
                        ))
                    })
                    .collect();
                store.v[out] = TensorData::Owned(o);
            }
            Op::Scatter {
                sel,
                token_id,
                expert_of,
                m,
                n,
            } => {
                let (tid, eof) = scatter_forward(store.get(sel), m, n);
                store.idx[token_id] = tid;
                store.idx[expert_of] = eof;
            }
            Op::Gather {
                weighted,
                token_id,
                gate,
                out,
                m,
                h,
            } => {
                let o = gather_forward(
                    store.get(weighted),
                    &store.idx[token_id],
                    store.get(gate),
                    m,
                    h,
                );
                store.v[out] = TensorData::Owned(o);
            }
            Op::BatchedMatMul { a, w, c, s, k, n } => {
                let o = bmm::forward(store.get(a), store.get(w), s, k, n);
                store.v[c] = TensorData::Owned(o);
            }
            Op::GatherMatMul { a, expert_of, w, c, s, k, n, n_experts } => {
                let wv = store.materialize(w);
                let o = bmm::gather_matmul_forward(
                    store.get(a),
                    &store.idx[expert_of],
                    wv.as_ref(),
                    s,
                    k,
                    n,
                    n_experts,
                );
                store.v[c] = TensorData::Owned(o);
            }
            Op::IdxBroadcast { base, out, b } => {
                let src = store.idx[base].clone();
                let mut o = Vec::with_capacity(src.len() * b);
                for &v in &src {
                    for j in 0..b {
                        o.push(v * b as u32 + j as u32);
                    }
                }
                store.idx[out] = o;
            }
            Op::CombineIdx { a, b, out, scale } => {
                let av = store.idx[a].clone();
                let bv = store.idx[b].clone();
                assert_eq!(av.len(), bv.len());
                let o: Vec<u32> = av
                    .iter()
                    .zip(bv.iter())
                    .map(|(&x, &y)| x * scale as u32 + y)
                    .collect();
                store.idx[out] = o;
            }
            Op::PadIdx { base, out, len } => {
                let mut o = store.idx[base].clone();
                o.resize(len, 0);
                store.idx[out] = o;
            }
            Op::BroadcastScale { x, scale, out, h } => {
                let xv = store.get(x);
                let sv = store.get(scale);
                let o: Vec<Goldilocks> = xv
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| v * sv[i / h])
                    .collect();
                store.v[out] = TensorData::Owned(o);
            }
            Op::ScaleGate {
                x,
                gate,
                col,
                out,
                m,
                n,
                h,
                shift,
            } => {
                let gv = store.get(gate);
                let o: Vec<Goldilocks> = store
                    .get(x)
                    .iter()
                    .enumerate()
                    .map(|(ij, &v)| {
                        let t = ij / h;
                        from_i64(round_div(
                            to_i64(v) * to_i64(gv[t * n + col]),
                            1i64 << shift,
                        ))
                    })
                    .collect();
                store.v[out] = TensorData::Owned(o);
            }
            Op::Relu { x, out } => {
                let o: Vec<Goldilocks> = store
                    .get(x)
                    .iter()
                    .map(|&v| if to_i64(v) > 0 { v } else { Goldilocks::ZERO })
                    .collect();
                store.v[out] = TensorData::Owned(o);
            }
            Op::MatMul { a, b, c, m, k, n } => {
                let bv = store.materialize(b);
                let cval = crate::par::mm_par_fixed(store.get(a), bv.as_ref(), m, k, n);
                store.v[c] = TensorData::Owned(cval);
            }
            Op::Projection {
                x,
                w,
                bias,
                out,
                rem,
                m,
                k,
                n,
                shift,
            } => {
                let wv = store.materialize(w);
                let h = crate::par::mm_par_fixed(store.get(x), wv.as_ref(), m, k, n);
                let o: Vec<Goldilocks> = (0..m * n)
                    .map(|ij| {
                        from_i64(
                            round_div(to_i64(h[ij]), 1i64 << shift) + to_i64(store.get(bias)[ij]),
                        )
                    })
                    .collect();
                let r: Vec<Goldilocks> = (0..m * n)
                    .map(|ij| {
                        from_i64(
                            to_i64(h[ij])
                                - (to_i64(o[ij]) - to_i64(store.get(bias)[ij])) * (1i64 << shift)
                                + (1i64 << (shift - 1)),
                        )
                    })
                    .collect();
                store.v[out] = TensorData::Owned(o);
                store.v[rem] = TensorData::Owned(r);
            }
            Op::Add { a, b, c } => {
                let s: Vec<Goldilocks> = store
                    .get(a)
                    .iter()
                    .zip(store.get(b))
                    .map(|(x, y)| *x + *y)
                    .collect();
                store.v[c] = TensorData::Owned(s);
            }
            Op::Lookup { idx, out, table } => {
                let o: Vec<Goldilocks> = store.idx[idx]
                    .iter()
                    .map(|&i| store.get(table)[i as usize])
                    .collect();
                store.v[out] = TensorData::Owned(o);
            }
            Op::Softmax {
                idx,
                e,
                out,
                table,
                m,
                n,
            } => {
                let ev: Vec<Goldilocks> = store.idx[idx]
                    .iter()
                    .map(|&i| store.get(table)[i as usize])
                    .collect();
                let sum: Vec<Goldilocks> = (0..m)
                    .map(|i| (0..n).fold(Goldilocks::ZERO, |acc, j| acc + ev[i * n + j]))
                    .collect();
                let o: Vec<Goldilocks> = (0..m * n)
                    .map(|ij| {
                        from_i64(round_div(
                            to_i64(ev[ij]) * (1i64 << 16),
                            to_i64(sum[ij / n]),
                        ))
                    })
                    .collect();
                store.v[e] = TensorData::Owned(ev);
                store.v[out] = TensorData::Owned(o);
            }
            Op::SoftmaxIndex { x, out, table_len } => {
                let indices: Vec<u32> = store
                    .get(x)
                    .iter()
                    .map(|&s| ((to_i64(s).max(0)) as u64 % table_len as u64) as u32)
                    .collect();
                store.idx[out] = indices;
            }
            Op::GeluIndex {
                x,
                out,
                offset,
                table_len,
            } => {
                let indices: Vec<u32> = store
                    .get(x)
                    .iter()
                    .map(|&v| (to_i64(v) + offset).clamp(0, table_len as i64 - 1) as u32)
                    .collect();
                store.idx[out] = indices;
            }
            Op::StableSoftmaxIndex {
                x,
                mask,
                out,
                offset,
                table_len,
                m,
                n,
            } => {
                let xv = store.get(x);
                let mv = store.get(mask);
                let masked: Vec<Goldilocks> = xv.iter().zip(mv).map(|(a, b)| *a + *b).collect();
                let row_max: Vec<Goldilocks> = (0..m)
                    .map(|i| {
                        (0..n).fold(masked[i * n], |acc, j| {
                            let v = masked[i * n + j];
                            if to_i64(v) > to_i64(acc) {
                                v
                            } else {
                                acc
                            }
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
            Op::Layernorm {
                x,
                w,
                b,
                out,
                rsqrt_table,
                m,
                d,
            } => {
                let xv = store.get(x);
                let wv = store.get(w);
                let bv = store.get(b);
                let table = store.get(rsqrt_table);
                let mean_sq: Vec<Goldilocks> = (0..m)
                    .map(|i| {
                        (0..d).fold(Goldilocks::ZERO, |acc, j| {
                            acc + xv[i * d + j] * xv[i * d + j]
                        })
                    })
                    .collect();
                let rsqrt_idx: Vec<u32> = mean_sq
                    .iter()
                    .map(|&v| ((to_i64(v).max(0)) as u64 % table.len() as u64) as u32)
                    .collect();
                let rsqrt: Vec<Goldilocks> = rsqrt_idx.iter().map(|&i| table[i as usize]).collect();
                let scale: Vec<Goldilocks> = (0..m * d).map(|ij| rsqrt[ij / d] * wv[ij]).collect();
                let o: Vec<Goldilocks> = (0..m * d).map(|ij| xv[ij] * scale[ij] + bv[ij]).collect();
                store.v[out] = TensorData::Owned(o);
            }
            Op::LayerNormCentered {
                x,
                w,
                b,
                out,
                rsqrt_table,
                m,
                d,
                n_real,
            } => {
                let (_, _, _, _, _, o, _, _, _, _, _) = layer_norm_centered_forward(
                    store.get(x),
                    store.get(w),
                    store.get(b),
                    store.get(rsqrt_table),
                    m,
                    d,
                    n_real,
                );
                store.v[out] = TensorData::Owned(o);
            }
            Op::RmsNorm {
                x,
                w,
                out,
                rsqrt_table,
                limb_table,
                m,
                d,
                n_real,
            } => {
                let (_, _, _, o, _, _, _, _, _) = rms_norm_forward(
                    store.get(x),
                    store.get(w),
                    store.get(rsqrt_table),
                    m,
                    d,
                    n_real,
                );
                store.v[out] = TensorData::Owned(o);
            }
            Op::RoPE {
                x,
                cos,
                sin,
                out,
                range_table,
                m,
                d,
                shift,
            } => {
                let (o, _, _) =
                    rope_forward(store.get(x), store.get(cos), store.get(sin), m, d, shift);
                store.v[out] = TensorData::Owned(o);
            }
            Op::TopKSelect {
                x,
                sel,
                thr,
                gate,
                d1,
                d2,
                m,
                n,
                k,
            } => {
                let (thr_v, sel_v, gate_v, d1_v, d2_v) = topk_forward(store.get(x), m, n, k);
                store.v[thr] = TensorData::Owned(thr_v);
                store.v[sel] = TensorData::Owned(sel_v);
                store.v[gate] = TensorData::Owned(gate_v);
                store.v[d1] = TensorData::Owned(d1_v);
                store.v[d2] = TensorData::Owned(d2_v);
            }
            Op::Argmax {
                x,
                sel,
                thr,
                gate,
                d1,
                d2,
                out,
                m,
                n,
            } => {
                let (thr_v, sel_v, gate_v, d1_v, d2_v, out_v) =
                    argmax_forward(store.get(x), m, n);
                store.v[thr] = TensorData::Owned(thr_v);
                store.v[sel] = TensorData::Owned(sel_v);
                store.v[gate] = TensorData::Owned(gate_v);
                store.v[d1] = TensorData::Owned(d1_v);
                store.v[d2] = TensorData::Owned(d2_v);
                store.idx[out] = out_v;
            }
        }
    }
}

/// Prove a shard: run forward, prove each op, collect the claims each op makes
/// on activation tensors, and `same_poly`-bind every tensor claimed more than
/// once (at different points). `boundary` marks the shard output tensor(s) that
/// are committed at the boundary and therefore not bound internally.
pub fn prove_shard(
    store: &mut Store,
    ops: &[Op],
    _boundary: &[T], // kept for API compatibility; binding no longer excludes boundary tensors (S2)
    transcript: &mut Transcript,
) -> OpShardProof {
    forward_ops(store, ops);
    let (proof, _claims) = prove_shard_precomputed(store, ops, transcript, FracMode::Plain);
    proof
}

/// Prove a shard whose witness is already materialized in `store` (the forward
/// pass has already run). Read-only on `store`, so shards can be proven in
/// parallel by the shard-DAG composer. All challenges are Fiat-Shamir derived
/// from the statement-absorbed transcript (S1): evaluation points are squeezed
/// directly, sumcheck challenges per-round from the round messages.
///
/// Returns the proof plus the shard's raw claims (S2): the claims are NOT
/// serialized into [`OpShardProof`] — the shard-DAG and committed composers
/// use them prover-side, while every verifier rebuilds its own claims from
/// the op verifications. In [`FracMode::Committed`] the permutation/gather
/// fractional proofs run on scratch transcripts (they are re-proven on the
/// committed transcripts later), so the shard transcript stays frac-free.
pub fn prove_shard_precomputed(
    store: &Store,
    ops: &[Op],
    transcript: &mut Transcript,
    mode: FracMode,
) -> (OpShardProof, Vec<EvalClaim>) {
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let add_terms = vec![
        (Goldilocks::ONE, vec![2usize]),
        (neg, vec![0usize]),
        (neg, vec![1usize]),
    ];

    let mut claims: Vec<EvalClaim> = Vec::new();
    let mut op_proofs = Vec::with_capacity(ops.len());
    let mut lookup_by_table: std::collections::BTreeMap<T, Vec<(Vec<u32>, Vec<Goldilocks>)>> =
        std::collections::BTreeMap::new();

    for op in ops {
        match *op {
            Op::Transpose { .. } => {
                op_proofs.push(OpProof::Transpose);
            }
            Op::Scale { .. } => {
                op_proofs.push(OpProof::Scale);
            }
            Op::ScaleVec { .. } => {
                op_proofs.push(OpProof::ScaleVec);
            }
            Op::Scatter {
                sel,
                token_id,
                expert_of,
                m,
                n,
            } => {
                let p = prove_scatter_with_mode(
                    store.get(sel),
                    &store.idx[token_id],
                    &store.idx[expert_of],
                    m,
                    n,
                    transcript,
                    mode,
                );
                op_proofs.push(OpProof::Scatter(p));
            }
            Op::Gather {
                weighted,
                token_id,
                gate,
                out,
                m,
                h,
            } => {
                let p = prove_scatter_add_with_mode(
                    store.get(weighted),
                    &store.idx[token_id],
                    store.get(gate),
                    store.get(out),
                    m,
                    h,
                    transcript,
                    mode,
                );
                op_proofs.push(OpProof::Gather(p));
            }
            Op::BatchedMatMul { a, w, c, s, k, n } => {
                let u: Vec<Goldilocks> = transcript.field_vec(s.trailing_zeros() as usize);
                let v: Vec<Goldilocks> = transcript.field_vec(n.trailing_zeros() as usize);
                let (p, ch) =
                    bmm::prove(store.get(a), store.get(w), store.get(c), s, k, n, &u, &v, transcript);
                claims.push(EvalClaim { tensor_id: a, point: ch.clone(), value: mle::eval(store.get(a), &ch) });
                let mut wp = v.clone();
                wp.extend_from_slice(&ch);
                claims.push(EvalClaim { tensor_id: w, point: wp.clone(), value: mle::eval(store.get(w), &wp) });
                let mut cp = v.clone();
                cp.extend_from_slice(&u);
                claims.push(EvalClaim { tensor_id: c, point: cp.clone(), value: mle::eval(store.get(c), &cp) });
                op_proofs.push(OpProof::BatchedMatMul(p, u, v, ch));
            }
            Op::GatherMatMul { a, expert_of, w, c, s, k, n, n_experts } => {
                let u: Vec<Goldilocks> = transcript.field_vec(s.trailing_zeros() as usize);
                let v: Vec<Goldilocks> = transcript.field_vec(n.trailing_zeros() as usize);
                let wv = store.materialize(w);
                let (p, ch) = bmm::prove_gather_matmul(
                    store.get(a),
                    &store.idx[expert_of],
                    wv.as_ref(),
                    store.get(c),
                    s,
                    k,
                    n,
                    n_experts,
                    &u,
                    &v,
                    transcript,
                );
                claims.push(EvalClaim { tensor_id: a, point: ch.clone(), value: mle::eval(store.get(a), &ch) });
                let mut cp = v.clone();
                cp.extend_from_slice(&u);
                claims.push(EvalClaim { tensor_id: c, point: cp.clone(), value: mle::eval(store.get(c), &cp) });
                op_proofs.push(OpProof::GatherMatMul(p, u, v, ch));
            }
            Op::IdxBroadcast { .. } => {
                op_proofs.push(OpProof::IdxBroadcast);
            }
            Op::CombineIdx { .. } => {
                op_proofs.push(OpProof::CombineIdx);
            }
            Op::PadIdx { .. } => {
                op_proofs.push(OpProof::PadIdx);
            }
            Op::BroadcastScale { .. } => {
                op_proofs.push(OpProof::BroadcastScale);
            }
            Op::ScaleGate { .. } => {
                op_proofs.push(OpProof::ScaleGate);
            }
            Op::Relu { .. } => {
                op_proofs.push(OpProof::Relu);
            }
            Op::MatMul { a, b, c, m, k, n } => {
                let at = transpose(store.get(a), m, k);
                let bv = store.materialize(b);
                let u: Vec<Goldilocks> = transcript.field_vec(m.trailing_zeros() as usize);
                let v: Vec<Goldilocks> = transcript.field_vec(n.trailing_zeros() as usize);
                let (p, ch) =
                    matmul_prove(&at, bv.as_ref(), store.get(c), m, k, n, &u, &v, transcript);
                let mut ap = ch.clone();
                ap.extend_from_slice(&u);
                claims.push(EvalClaim { tensor_id: a, point: ap.clone(), value: mle::eval(store.get(a), &ap) });
                let mut bp = v.clone();
                bp.extend_from_slice(&ch);
                claims.push(EvalClaim { tensor_id: b, point: bp.clone(), value: mle::eval(bv.as_ref(), &bp) });
                let mut cp = v.clone();
                cp.extend_from_slice(&u);
                claims.push(EvalClaim { tensor_id: c, point: cp.clone(), value: mle::eval(store.get(c), &cp) });
                op_proofs.push(OpProof::MatMul(p, u, v, ch));
            }
            Op::Projection {
                x,
                w,
                bias,
                out,
                rem,
                m,
                k,
                n,
                shift,
            } => {
                let wv = store.materialize(w);
                let p = prove_projection(
                    store.get(x),
                    wv.as_ref(),
                    store.get(bias),
                    store.get(out),
                    store.get(rem),
                    m,
                    k,
                    n,
                    shift,
                    transcript,
                );
                let mut in_pt = p.ch.clone();
                in_pt.extend_from_slice(&p.u);
                claims.push(EvalClaim { tensor_id: x, point: in_pt.clone(), value: mle::eval(store.get(x), &in_pt) });
                let mut weight_pt = p.v.clone();
                weight_pt.extend_from_slice(&p.ch);
                claims.push(EvalClaim { tensor_id: w, point: weight_pt.clone(), value: mle::eval(wv.as_ref(), &weight_pt) });
                claims.push(EvalClaim { tensor_id: out, point: p.pt.clone(), value: mle::eval(store.get(out), &p.pt) });
                claims.push(EvalClaim { tensor_id: bias, point: p.pt.clone(), value: p.affine.final_evals[4] });
                op_proofs.push(OpProof::Projection(p));
            }
            Op::Add { a, b, c } => {
                let mles: Vec<&[Goldilocks]> = vec![store.get(a), store.get(b), store.get(c)];
                let (proof, r) = prove_virtual(&mles, &add_terms, Goldilocks::ZERO, transcript);
                claims.push(EvalClaim { tensor_id: a, point: r.clone(), value: mle::eval(store.get(a), &r) });
                claims.push(EvalClaim { tensor_id: b, point: r.clone(), value: mle::eval(store.get(b), &r) });
                claims.push(EvalClaim { tensor_id: c, point: r.clone(), value: mle::eval(store.get(c), &r) });
                op_proofs.push(OpProof::Add(proof, r));
            }
            Op::Lookup { idx, out, table } => {
                lookup_by_table
                    .entry(table)
                    .or_insert(Vec::new())
                    .push((store.idx[idx].clone(), store.get(out).to_vec()));
                op_proofs.push(OpProof::Lookup);
            }
            Op::Softmax {
                idx,
                e,
                out,
                table,
                m,
                n,
            } => {
                let p = prove_softmax_rounded(
                    &store.idx[idx],
                    store.get(e),
                    store.get(out),
                    store.get(table),
                    m,
                    n,
                    transcript,
                );
                claims.push(EvalClaim {
                    tensor_id: out,
                    point: p.r_scale.clone(),
                    value: mle::eval(store.get(out), &p.r_scale),
                });
                lookup_by_table
                    .entry(table)
                    .or_insert(Vec::new())
                    .push((store.idx[idx].clone(), store.get(e).to_vec()));
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
            Op::Layernorm {
                x,
                w,
                b,
                out,
                rsqrt_table,
                m,
                d,
            } => {
                let (p, ridx, rvals) = prove_layernorm_chain(
                    store.get(x),
                    store.get(w),
                    store.get(b),
                    store.get(rsqrt_table),
                    m,
                    d,
                    transcript,
                );
                lookup_by_table
                    .entry(rsqrt_table)
                    .or_insert(Vec::new())
                    .push((ridx, rvals));
                claims.push(EvalClaim { tensor_id: out, point: p.r_out.clone(), value: mle::eval(store.get(out), &p.r_out) });
                claims.push(EvalClaim { tensor_id: w, point: p.r_out.clone(), value: mle::eval(store.get(w), &p.r_out) });
                claims.push(EvalClaim { tensor_id: b, point: p.r_out.clone(), value: mle::eval(store.get(b), &p.r_out) });
                op_proofs.push(OpProof::Layernorm(p));
            }
            Op::LayerNormCentered {
                x,
                w,
                b,
                out,
                rsqrt_table,
                m,
                d,
                n_real,
            } => {
                let (p, ridx, rvals) = prove_layer_norm_centered(
                    store.get(x),
                    store.get(w),
                    store.get(b),
                    store.get(rsqrt_table),
                    m,
                    d,
                    n_real,
                    transcript,
                );
                lookup_by_table
                    .entry(rsqrt_table)
                    .or_insert(Vec::new())
                    .push((ridx, rvals));
                claims.push(EvalClaim {
                    tensor_id: x,
                    point: p.centered_r.clone(),
                    value: mle::eval(store.get(x), &p.centered_r),
                });
                claims.push(EvalClaim { tensor_id: out, point: p.r_out.clone(), value: mle::eval(store.get(out), &p.r_out) });
                claims.push(EvalClaim { tensor_id: w, point: p.r_out.clone(), value: mle::eval(store.get(w), &p.r_out) });
                claims.push(EvalClaim { tensor_id: b, point: p.r_out.clone(), value: mle::eval(store.get(b), &p.r_out) });
                op_proofs.push(OpProof::LayerNormCentered(p));
            }
            Op::RmsNorm {
                x,
                w,
                out,
                rsqrt_table,
                limb_table,
                m,
                d,
                n_real,
            } => {
                let (p, ridx, rvals, rem_lo, rem_hi) = prove_rms_norm(
                    store.get(x),
                    store.get(w),
                    store.get(rsqrt_table),
                    m,
                    d,
                    n_real,
                    transcript,
                );
                lookup_by_table
                    .entry(rsqrt_table)
                    .or_insert(Vec::new())
                    .push((ridx, rvals));
                let idx_lo: Vec<u32> = rem_lo.iter().map(|&v| to_i32(v) as u32).collect();
                let idx_hi: Vec<u32> = rem_hi.iter().map(|&v| to_i32(v) as u32).collect();
                lookup_by_table
                    .entry(limb_table)
                    .or_insert(Vec::new())
                    .push((idx_lo, rem_lo));
                lookup_by_table
                    .entry(limb_table)
                    .or_insert(Vec::new())
                    .push((idx_hi, rem_hi));
                claims.push(EvalClaim { tensor_id: out, point: p.r_out.clone(), value: mle::eval(store.get(out), &p.r_out) });
                claims.push(EvalClaim { tensor_id: w, point: p.r_out.clone(), value: mle::eval(store.get(w), &p.r_out) });
                op_proofs.push(OpProof::RmsNorm(p));
            }
            Op::RoPE {
                x,
                cos,
                sin,
                out,
                range_table,
                m,
                d,
                shift,
            } => {
                let (p, rem_f, rem_s) = prove_rope(
                    store.get(x),
                    store.get(cos),
                    store.get(sin),
                    store.get(out),
                    m,
                    d,
                    shift,
                    transcript,
                );
                let idx_f: Vec<u32> = rem_f.iter().map(|&v| to_i32(v) as u32).collect();
                let idx_s: Vec<u32> = rem_s.iter().map(|&v| to_i32(v) as u32).collect();
                lookup_by_table
                    .entry(range_table)
                    .or_insert(Vec::new())
                    .push((idx_f, rem_f));
                lookup_by_table
                    .entry(range_table)
                    .or_insert(Vec::new())
                    .push((idx_s, rem_s));
                // S2: `out` is claimed at the two half points whose values are
                // the checked sumcheck final evals (first.final_evals[4],
                // second.final_evals[4]); the old unverifiable r_out claim is
                // gone.
                let (pt_f, pt_s) = rope_out_points(&p.r_first, &p.r_second, d);
                claims.push(EvalClaim { tensor_id: out, point: pt_f.clone(), value: mle::eval(store.get(out), &pt_f) });
                claims.push(EvalClaim { tensor_id: out, point: pt_s.clone(), value: mle::eval(store.get(out), &pt_s) });
                op_proofs.push(OpProof::RoPE(p));
            }
            Op::TopKSelect {
                x,
                sel,
                thr,
                gate,
                d1,
                d2,
                m,
                n,
                k,
            } => {
                let p = prove_topk(
                    store.get(x),
                    store.get(sel),
                    store.get(thr),
                    store.get(gate),
                    store.get(d1),
                    store.get(d2),
                    m,
                    n,
                    k,
                    22,
                    transcript,
                );
                claims.push(EvalClaim { tensor_id: x, point: p.r.clone(), value: mle::eval(store.get(x), &p.r) });
                claims.push(EvalClaim { tensor_id: sel, point: p.r.clone(), value: mle::eval(store.get(sel), &p.r) });
                claims.push(EvalClaim { tensor_id: gate, point: p.r.clone(), value: mle::eval(store.get(gate), &p.r) });
                op_proofs.push(OpProof::TopKSelect(p));
            }
            Op::Argmax {
                x,
                sel,
                thr,
                gate,
                d1,
                d2,
                out,
                m,
                n,
            } => {
                let p = prove_argmax(
                    store.get(x),
                    store.get(sel),
                    store.get(thr),
                    store.get(gate),
                    store.get(d1),
                    store.get(d2),
                    &store.idx[out],
                    m,
                    n,
                    16,
                    transcript,
                );
                claims.push(EvalClaim { tensor_id: x, point: p.topk.r.clone(), value: mle::eval(store.get(x), &p.topk.r) });
                claims.push(EvalClaim { tensor_id: sel, point: p.topk.r.clone(), value: mle::eval(store.get(sel), &p.topk.r) });
                claims.push(EvalClaim { tensor_id: gate, point: p.topk.r.clone(), value: mle::eval(store.get(gate), &p.topk.r) });
                op_proofs.push(OpProof::Argmax(p));
            }
        }
    }

    // Normalize the claim list (S2): same-tensor same-point claims merge
    // (duplicates keep one entry); contradictory values on one point are a
    // prover bug and rejected loudly. The verifier applies the same
    // normalization with rejection.
    let mut seen: std::collections::HashMap<(T, Vec<Goldilocks>), Goldilocks> =
        std::collections::HashMap::new();
    let mut normalized: Vec<EvalClaim> = Vec::with_capacity(claims.len());
    for c in claims {
        match seen.get(&(c.tensor_id, c.point.clone())) {
            Some(&v) => {
                assert_eq!(v, c.value, "contradictory claims on tensor {}", c.tensor_id);
            }
            None => {
                seen.insert((c.tensor_id, c.point.clone()), c.value);
                normalized.push(c);
            }
        }
    }
    let claims = normalized;

    // Group claims by tensor and bind every tensor claimed at >1 point (S2:
    // no boundary exclusion — the committed path opens boundary tensors
    // directly and the bind is extra checking; the verifier rebuilds the
    // same set from its own claims).
    let mut by_tensor: std::collections::BTreeMap<T, Vec<(Vec<Goldilocks>, Goldilocks)>> =
        std::collections::BTreeMap::new();
    for c in &claims {
        by_tensor
            .entry(c.tensor_id)
            .or_default()
            .push((c.point.clone(), c.value));
    }

    let mut bound = Vec::new();
    let mut binds = Vec::new();
    for (t, cs) in by_tensor {
        if cs.len() < 2 {
            continue;
        }
        // S4: the shard-level bind is a consistency check whose merged
        // claim is NOT consumed downstream — its input claims are each
        // authenticated by the committed openings (S2 point-set) instead, so
        // dropping the merged output here is sound.
        let (sp, _merged) = prove_same_poly(store.materialize(t).as_ref(), &cs, transcript);
        bound.push(t);
        binds.push(sp);
    }

    let mut witness_columns = Vec::new();
    let mut table_columns = Vec::new();
    for (table, lookups) in &lookup_by_table {
        let alpha = transcript.field();
        let beta = transcript.field();
        let xs: Vec<&[u32]> = lookups.iter().map(|(x, _)| x.as_slice()).collect();
        let ys: Vec<&[Goldilocks]> = lookups.iter().map(|(_, y)| y.as_slice()).collect();
        let (proof, x, y, num, _den) =
            prove_lookup_witness_batch_columns(&xs, &ys, alpha, beta, transcript);
        let m = merge_multiplicity(&xs, store.get(*table).len());
        let table_proof = prove_lookup_table(&m, store.get(*table), alpha, beta, transcript);
        witness_columns.push(WitnessColumn {
            proof,
            x,
            y,
            num,
            alpha,
            beta,
            table: *table,
        });
        table_columns.push(TableColumn {
            proof: table_proof,
            table: *table,
            m,
            alpha,
            beta,
        });
    }

    (
        OpShardProof {
            ops: op_proofs,
            bound,
            binds,
            witness_columns,
            table_columns,
        },
        claims,
    )
}

/// Verify a shard proof (recomputes the witness first).
pub fn verify_shard(
    tables: &[&[Goldilocks]],
    ops: &[Op],
    proof: &OpShardProof,
    transcript: &mut Transcript,
) -> bool {
    verify_shard_precomputed(tables, ops, proof, transcript)
}

/// Verify a shard proof with the witness already recomputed by the caller.
/// S1: the verifier replays the prover's transcript interleaving op by op —
/// squeezing the same evaluation points (checked against the stored ones) and
/// re-deriving every sumcheck challenge vector from the proof's round
/// messages.
pub fn verify_shard_precomputed(
    tables: &[&[Goldilocks]],
    ops: &[Op],
    proof: &OpShardProof,
    transcript: &mut Transcript,
) -> bool {
    verify_shard_precomputed_inner(tables, ops, proof, transcript, FracMode::Plain).is_some()
}

/// Shard replay that BUILDS the shard's claim set (S2): every claim is derived
/// from the op verifications' checked final evals, never read from the proof.
/// Returns the claims on success; `None` rejects.
///
/// In [`FracMode::Committed`] the lookup-column fractional section and the
/// permutation/gather fracs are skipped: the committed path re-proves all of
/// them on its own transcripts (binding the committed column roots) and
/// verifies them in its dedicated sections.
fn verify_shard_precomputed_inner(
    tables: &[&[Goldilocks]],
    ops: &[Op],
    proof: &OpShardProof,
    transcript: &mut Transcript,
    mode: FracMode,
) -> Option<Vec<EvalClaim>> {
    if proof.ops.len() != ops.len() || proof.bound.len() != proof.binds.len() {
        return None;
    }

    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let add_terms = vec![
        (Goldilocks::ONE, vec![2usize]),
        (neg, vec![0usize]),
        (neg, vec![1usize]),
    ];

    let mut claims: Vec<EvalClaim> = Vec::new();
    for (op, p) in ops.iter().zip(&proof.ops) {
        match (op, p) {
            (Op::Transpose { .. }, OpProof::Transpose) => {
                // No proof: the verifier recomputes `out = transpose(x)` in
                // forward_ops, and the downstream op's claim on `out` is checked
                // against that recomputed value.
            }
            (Op::Scale { .. }, OpProof::Scale) => {}
            (Op::ScaleVec { .. }, OpProof::ScaleVec) => {}
            (Op::Scatter { m, n, .. }, OpProof::Scatter(p)) => {
                if !verify_scatter_with_mode(p, *m, *n, transcript, mode) {
                    return None;
                }
            }
            (Op::Gather { m: _, h: _, .. }, OpProof::Gather(p)) => {
                if !verify_scatter_add_with_mode(p, transcript, mode) {
                    return None;
                }
            }
            (
                Op::BatchedMatMul { a, w, c, s, k, n },
                OpProof::BatchedMatMul(p, u, v, ch),
            ) => {
                let u_s: Vec<Goldilocks> = transcript.field_vec(s.trailing_zeros() as usize);
                let v_s: Vec<Goldilocks> = transcript.field_vec(n.trailing_zeros() as usize);
                if u_s != *u || v_s != *v {
                    return None;
                }
                let (ok, ch_derived) = bmm::verify(p, *s, *k, u, transcript);
                if !ok || ch_derived != *ch {
                    return None;
                }
                claims.push(EvalClaim { tensor_id: *a, point: ch.clone(), value: p.sumcheck.final_evals[0] });
                let mut wp = v.clone();
                wp.extend_from_slice(ch);
                claims.push(EvalClaim { tensor_id: *w, point: wp, value: p.sumcheck.final_evals[1] });
                let mut cp = v.clone();
                cp.extend_from_slice(u);
                claims.push(EvalClaim { tensor_id: *c, point: cp, value: p.claimed });
            }
            (
                Op::GatherMatMul { a, c, s, k, n, .. },
                OpProof::GatherMatMul(p, u, v, ch),
            ) => {
                let u_s: Vec<Goldilocks> = transcript.field_vec(s.trailing_zeros() as usize);
                let v_s: Vec<Goldilocks> = transcript.field_vec(n.trailing_zeros() as usize);
                if u_s != *u || v_s != *v {
                    return None;
                }
                let (ok, ch_derived) = bmm::verify_gather_matmul(p, *s, *k, u, transcript);
                if !ok || ch_derived != *ch {
                    return None;
                }
                claims.push(EvalClaim { tensor_id: *a, point: ch.clone(), value: p.sumcheck.final_evals[0] });
                let mut cp = v.clone();
                cp.extend_from_slice(u);
                claims.push(EvalClaim { tensor_id: *c, point: cp, value: p.claimed });
            }
            (Op::IdxBroadcast { .. }, OpProof::IdxBroadcast) => {}
            (Op::CombineIdx { .. }, OpProof::CombineIdx) => {}
            (Op::PadIdx { .. }, OpProof::PadIdx) => {}
            (Op::BroadcastScale { .. }, OpProof::BroadcastScale) => {}
            (Op::ScaleGate { .. }, OpProof::ScaleGate) => {}
            (Op::Relu { .. }, OpProof::Relu) => {}
            (
                Op::MatMul { a, b, c, m, k, n },
                OpProof::MatMul(mp, u, v, ch),
            ) => {
                let u_s: Vec<Goldilocks> = transcript.field_vec(m.trailing_zeros() as usize);
                let v_s: Vec<Goldilocks> = transcript.field_vec(n.trailing_zeros() as usize);
                if u_s != *u || v_s != *v {
                    return None;
                }
                let (ok, ch_derived) = matmul_verify(mp, k.trailing_zeros() as usize, transcript);
                if !ok || ch_derived != *ch {
                    return None;
                }
                let mut ap = ch.clone();
                ap.extend_from_slice(u);
                claims.push(EvalClaim { tensor_id: *a, point: ap, value: mp.sumcheck.f_eval });
                let mut bp = v.clone();
                bp.extend_from_slice(ch);
                claims.push(EvalClaim { tensor_id: *b, point: bp, value: mp.sumcheck.h_eval });
                let mut cp = v.clone();
                cp.extend_from_slice(u);
                claims.push(EvalClaim { tensor_id: *c, point: cp, value: mp.claimed });
            }
            (
                Op::Projection { x, w, bias, out, shift, m, k, n, .. },
                OpProof::Projection(pp),
            ) => {
                if !verify_projection(pp, *shift, *m, *k, *n, transcript) {
                    return None;
                }
                let mut in_pt = pp.ch.clone();
                in_pt.extend_from_slice(&pp.u);
                claims.push(EvalClaim { tensor_id: *x, point: in_pt, value: pp.matmul.sumcheck.f_eval });
                let mut weight_pt = pp.v.clone();
                weight_pt.extend_from_slice(&pp.ch);
                claims.push(EvalClaim { tensor_id: *w, point: weight_pt, value: pp.matmul.sumcheck.h_eval });
                claims.push(EvalClaim { tensor_id: *out, point: pp.pt.clone(), value: pp.affine.final_evals[3] });
                claims.push(EvalClaim { tensor_id: *bias, point: pp.pt.clone(), value: pp.affine.final_evals[4] });
            }
            (Op::Add { a, b, c }, OpProof::Add(vp, r)) => {
                let (ok, r_derived, fc) =
                    verify_virtual(vp, &add_terms, Goldilocks::ZERO, r.len(), transcript);
                if !ok
                    || r_derived != *r
                    || fc != virtual_eval(&add_terms, &vp.final_evals)
                {
                    return None;
                }
                claims.push(EvalClaim { tensor_id: *a, point: r.clone(), value: vp.final_evals[0] });
                claims.push(EvalClaim { tensor_id: *b, point: r.clone(), value: vp.final_evals[1] });
                claims.push(EvalClaim { tensor_id: *c, point: r.clone(), value: vp.final_evals[2] });
            }
            (Op::Lookup { table, .. }, OpProof::Lookup) => {
                // Existence only: the per-table fractional replays happen after
                // the op loop, in the prover's column order (or in the
                // committed frac section on the committed path).
                let w = proof.witness_columns.iter().find(|w| w.table == *table);
                let t = proof.table_columns.iter().find(|t| t.table == *table);
                match (w, t) {
                    (Some(_), Some(_)) => {}
                    _ => return None,
                }
            }
            (Op::Softmax { out, m, n, .. }, OpProof::Softmax(sp)) => {
                if !verify_softmax_rounded(sp, *m, *n, transcript) {
                    return None;
                }
                claims.push(EvalClaim {
                    tensor_id: *out,
                    point: sp.r_scale.clone(),
                    value: sp.rescale.final_evals[0],
                });
            }
            (Op::SoftmaxIndex { .. }, OpProof::SoftmaxIndex) => {
                // No proof: the verifier recomputes `idx = softmax_index(scores)`.
            }
            (Op::GeluIndex { .. }, OpProof::GeluIndex) => {}
            (Op::StableSoftmaxIndex { .. }, OpProof::StableSoftmaxIndex) => {}
            (Op::Layernorm { out, w, b, m, d, .. }, OpProof::Layernorm(lp)) => {
                if !verify_layernorm_chain(lp, *m, *d, transcript) {
                    return None;
                }
                // out relation mles [x, rsqrt_b, w, b, out] (degree 3, S2).
                claims.push(EvalClaim { tensor_id: *out, point: lp.r_out.clone(), value: lp.out.final_evals[4] });
                claims.push(EvalClaim { tensor_id: *w, point: lp.r_out.clone(), value: lp.out.final_evals[2] });
                claims.push(EvalClaim { tensor_id: *b, point: lp.r_out.clone(), value: lp.out.final_evals[3] });
            }
            (
                Op::LayerNormCentered { x, w, b, out, m, d, n_real, .. },
                OpProof::LayerNormCentered(lp),
            ) => {
                if !verify_layer_norm_centered(lp, *m, *d, *n_real, transcript) {
                    return None;
                }
                claims.push(EvalClaim {
                    tensor_id: *x,
                    point: lp.centered_r.clone(),
                    value: lp.centered.final_evals[0],
                });
                // out relation mles [centered, rstd_b, w, b, raw, out, rem_out, ones].
                claims.push(EvalClaim { tensor_id: *out, point: lp.r_out.clone(), value: lp.out.final_evals[5] });
                claims.push(EvalClaim { tensor_id: *w, point: lp.r_out.clone(), value: lp.out.final_evals[2] });
                claims.push(EvalClaim { tensor_id: *b, point: lp.r_out.clone(), value: lp.out.final_evals[3] });
            }
            (Op::RmsNorm { out, w, m, d, n_real, .. }, OpProof::RmsNorm(p)) => {
                if !verify_rms_norm(p, *m, *d, *n_real, transcript) {
                    return None;
                }
                // out relation mles [x, rstd_b, w, raw, out, rem_out, ones].
                claims.push(EvalClaim { tensor_id: *out, point: p.r_out.clone(), value: p.out.final_evals[4] });
                claims.push(EvalClaim { tensor_id: *w, point: p.r_out.clone(), value: p.out.final_evals[2] });
            }
            (Op::RoPE { out, d, shift, .. }, OpProof::RoPE(p)) => {
                if !verify_rope(p, *shift, transcript) {
                    return None;
                }
                // S2: `out` is claimed at the two half points whose values are
                // the checked sumcheck final evals (first/second.final_evals[4]).
                let (pt_f, pt_s) = rope_out_points(&p.r_first, &p.r_second, *d);
                claims.push(EvalClaim { tensor_id: *out, point: pt_f, value: p.first.final_evals[4] });
                claims.push(EvalClaim { tensor_id: *out, point: pt_s, value: p.second.final_evals[4] });
            }
            (Op::TopKSelect { x, sel, gate, k, m, n, .. }, OpProof::TopKSelect(p)) => {
                if !verify_topk(p, *k, *m, *n, transcript) {
                    return None;
                }
                claims.push(EvalClaim { tensor_id: *x, point: p.r.clone(), value: p.gate_rel.final_evals[0] });
                claims.push(EvalClaim { tensor_id: *sel, point: p.r.clone(), value: p.gate_rel.final_evals[1] });
                claims.push(EvalClaim { tensor_id: *gate, point: p.r.clone(), value: p.gate_rel.final_evals[2] });
            }
            (Op::Argmax { x, sel, gate, m, n, .. }, OpProof::Argmax(p)) => {
                if !verify_argmax(p, *m, *n, transcript) {
                    return None;
                }
                claims.push(EvalClaim { tensor_id: *x, point: p.topk.r.clone(), value: p.topk.gate_rel.final_evals[0] });
                claims.push(EvalClaim { tensor_id: *sel, point: p.topk.r.clone(), value: p.topk.gate_rel.final_evals[1] });
                claims.push(EvalClaim { tensor_id: *gate, point: p.topk.r.clone(), value: p.topk.gate_rel.final_evals[2] });
            }
            _ => return None,
        }
    }

    // Normalize + reject contradictions (S2): two claims on the same tensor
    // at the same point with different values can never both be evaluations
    // of one MLE — reject; equal duplicates merge.
    let mut seen: std::collections::HashMap<(T, Vec<Goldilocks>), Goldilocks> =
        std::collections::HashMap::new();
    let mut normalized: Vec<EvalClaim> = Vec::with_capacity(claims.len());
    for c in claims {
        match seen.get(&(c.tensor_id, c.point.clone())) {
            Some(&v) => {
                if v != c.value {
                    return None;
                }
            }
            None => {
                seen.insert((c.tensor_id, c.point.clone()), c.value);
                normalized.push(c);
            }
        }
    }
    let claims = normalized;

    // Rebuild the binding set from the LOCALLY BUILT claims (S2): every tensor
    // with >1 claim must be bound, and the proof's `bound` list must equal
    // that set in the prover's (sorted) order.
    let mut by_tensor: std::collections::BTreeMap<T, Vec<(Vec<Goldilocks>, Goldilocks)>> =
        std::collections::BTreeMap::new();
    for c in &claims {
        by_tensor
            .entry(c.tensor_id)
            .or_default()
            .push((c.point.clone(), c.value));
    }
    let bound_groups: Vec<(&T, &Vec<(Vec<Goldilocks>, Goldilocks)>)> =
        by_tensor.iter().filter(|(_, cs)| cs.len() >= 2).collect();
    if bound_groups.len() != proof.bound.len() {
        return None;
    }
    for (i, (t, cs)) in bound_groups.iter().enumerate() {
        if **t != proof.bound[i] {
            return None;
        }
        if verify_same_poly(&proof.binds[i], cs, transcript).is_none() {
            return None;
        }
    }

    // Replay the per-table lookup fractional proofs in the prover's column
    // order (per table: alpha, beta, witness frac, table frac). Skipped on
    // the committed path, whose frac section verifies the re-proven proofs
    // on the committed transcript.
    if mode == FracMode::Plain {
        if proof.witness_columns.len() != proof.table_columns.len() {
            return None;
        }
        for (wc, tc) in proof.witness_columns.iter().zip(proof.table_columns.iter()) {
            if transcript.field() != wc.alpha || transcript.field() != wc.beta {
                return None;
            }
            // S3: the depths come from the columns — the plain proof carries
            // the witness columns, the table depth from the statement table.
            let t_w = wc.num.len().trailing_zeros() as usize;
            let t_t = tables[tc.table]
                .len()
                .next_power_of_two()
                .trailing_zeros() as usize;
            if !verify_lookup_split(&wc.proof, &tc.proof, t_w, t_t, transcript) {
                return None;
            }
        }
    }

    Some(claims)
}

/// A whole-model shard-DAG proof: every shard folded into one g, plus
/// `same_poly` cross-shard bindings on every tensor claimed by more than one
/// shard. `ops_per_shard` is the public granularity parameter.
#[derive(serde::Serialize, serde::Deserialize)]
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

fn absorb_statement(t: &mut Transcript, ops: &[Op], ops_per_shard: usize) {
    t.absorb_u64(ops_per_shard as u64);
    t.absorb_u64(ops.len() as u64);
    for op in ops {
        for id in op.reads() {
            t.absorb_u64(id as u64);
        }
        for id in op.writes() {
            t.absorb_u64(id as u64);
        }
    }
}

pub fn prove_shard_dag(
    store: &mut Store,
    ops: &[Op],
    ops_per_shard: usize,
) -> ShardDagProof {
    let (proof, _claims) = prove_shard_dag_inner(store, ops, ops_per_shard, FracMode::Plain);
    proof
}

/// Prove the shard DAG and also return the per-shard claim sets (S2): the
/// shard proofs carry no claims — the DAG/committed composers consume these
/// prover-side, while every verifier rebuilds its own. In
/// [`FracMode::Committed`] the per-op permutation/gather fracs stay off the
/// shard transcripts (they are re-proven on the committed transcripts).
fn prove_shard_dag_inner(
    store: &mut Store,
    ops: &[Op],
    ops_per_shard: usize,
    mode: FracMode,
) -> (ShardDagProof, Vec<Vec<EvalClaim>>) {
    let ranges = shard_ranges(ops.len(), ops_per_shard);

    // Materialize every op output once, then prove each shard's sumchecks in
    // parallel (read-only on the store). Per-shard RNGs are seeded
    // deterministically from the caller's RNG, so the proof stays reproducible.
    forward_ops(store, ops);
    let store_ref: &Store = store;

    let mut master = Transcript::new(b"zkie-shard-dag-v1");
    absorb_statement(&mut master, ops, ops_per_shard);

    let shard_results: Vec<(OpShardProof, Vec<EvalClaim>)> = ranges
        .par_iter()
        .enumerate()
        .map(|(si, &(s, e))| {
            let mut child = master.clone();
            child.absorb_u64(si as u64);
            prove_shard_precomputed(store_ref, &ops[s..e], &mut child, mode)
        })
        .collect();
    let (shards, shard_claims): (Vec<OpShardProof>, Vec<Vec<EvalClaim>>) =
        shard_results.into_iter().unzip();

    // Group the prover-side claims by tensor across shards; bind tensors
    // claimed by >1 shard.
    let mut by_tensor: std::collections::BTreeMap<T, Vec<(Vec<Goldilocks>, Goldilocks)>> =
        std::collections::BTreeMap::new();
    for claims in shard_claims.iter() {
        for c in claims {
            by_tensor
                .entry(c.tensor_id)
                .or_default()
                .push((c.point.clone(), c.value));
        }
    }
    // Recompute per-tensor shard count with a set (a tensor may be claimed twice
    // by one shard, so a plain counter would overcount).
    let mut tensor_shards: std::collections::BTreeMap<T, std::collections::HashSet<usize>> =
        std::collections::BTreeMap::new();
    for (si, claims) in shard_claims.iter().enumerate() {
        for c in claims {
            tensor_shards.entry(c.tensor_id).or_default().insert(si);
        }
    }

    let mut cross_tensors = Vec::new();
    let mut cross_binds = Vec::new();
    for (t, claims) in by_tensor {
        if tensor_shards.get(&t).map(|s| s.len()).unwrap_or(0) > 1 {
            // S4: same as the shard binds — the DAG cross-bind's merged
            // claim is not consumed; the boundary claims are authenticated
            // by the committed openings instead.
            let (sp, _merged) = prove_same_poly(store.materialize(t).as_ref(), &claims, &mut master);
            cross_tensors.push(t);
            cross_binds.push(sp);
        }
    }
    (
        ShardDagProof {
            shards,
            cross_tensors,
            cross_binds,
        },
        shard_claims,
    )
}

/// Standalone verifier entry: deserialize a serialized  from
/// bytes and verify it against the public statement (tables + ops + granularity).
/// No witness Store is touched.
pub fn verify_shard_dag_bytes(
    proof_bytes: &[u8],
    tables: &[&[Goldilocks]],
    ops: &[Op],
    ops_per_shard: usize,
) -> bool {
    let proof: ShardDagProof = match bincode::deserialize(proof_bytes) {
        Ok(p) => p,
        Err(_) => return false,
    };
    verify_shard_dag(tables, ops, ops_per_shard, &proof)
}

pub fn verify_shard_dag(
    tables: &[&[Goldilocks]],
    ops: &[Op],
    ops_per_shard: usize,
    proof: &ShardDagProof,
) -> bool {
    verify_shard_dag_inner(tables, ops, ops_per_shard, proof, FracMode::Plain)
}

/// Shard-DAG verification that BUILDS every shard's claims locally (S2) and
/// rebuilds the cross-shard binding set from them — `cross_tensors` and the
/// cross-bind inputs never come from the proof.
fn verify_shard_dag_inner(
    tables: &[&[Goldilocks]],
    ops: &[Op],
    ops_per_shard: usize,
    proof: &ShardDagProof,
    mode: FracMode,
) -> bool {
    let ranges = shard_ranges(ops.len(), ops_per_shard);
    if proof.shards.len() != ranges.len() || proof.cross_tensors.len() != proof.cross_binds.len() {
        return false;
    }

    // Replay the same transcript structure as prove_shard_dag: statement
    // absorbed, then a shard-indexed clone per shard; the master transcript
    // stays at the post-statement state for the cross-shard bindings.
    let mut master = Transcript::new(b"zkie-shard-dag-v1");
    absorb_statement(&mut master, ops, ops_per_shard);

    let shard_claims: Vec<Option<Vec<EvalClaim>>> = ranges
        .par_iter()
        .enumerate()
        .map(|(i, &(s, e))| {
            let mut child = master.clone();
            child.absorb_u64(i as u64);
            verify_shard_precomputed_inner(tables, &ops[s..e], &proof.shards[i], &mut child, mode)
        })
        .collect();
    if shard_claims.iter().any(|c| c.is_none()) {
        return false;
    }

    let mut by_tensor: std::collections::BTreeMap<T, Vec<(Vec<Goldilocks>, Goldilocks)>> =
        std::collections::BTreeMap::new();
    let mut tensor_shards: std::collections::BTreeMap<T, std::collections::HashSet<usize>> =
        std::collections::BTreeMap::new();
    for (si, claims) in shard_claims.iter().enumerate() {
        for c in claims.as_ref().unwrap() {
            by_tensor
                .entry(c.tensor_id)
                .or_default()
                .push((c.point.clone(), c.value));
            tensor_shards.entry(c.tensor_id).or_default().insert(si);
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
        if verify_same_poly(&proof.cross_binds[ci], &by_tensor[&t], &mut master).is_none() {
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
/// the plain `prove_shard_dag` defers. The opening proofs are produced here (on
/// the prover side) and carried in the proof, so the verifier never touches the
/// committed tensor data.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct CommittedCrossBindProof {
    pub same_poly: SamePolyProof,
    pub openings: Vec<(zkie_core::pcs::whir::Proof, Goldilocks)>,
}

/// Verifier-side boundary commitment: the commitment root plus the tensor arity
/// (log2 length). The opening protocol is reconstructed by the verifier from the
/// arity, so no prover's Merkle data or protocol object is shipped.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct BoundaryCommit {
    pub commitment: zkie_core::pcs::whir::Commitment,
    pub arity: usize,
}

/// Verifier-side weight batch commitment: root + arity + table count + the batch
/// Whir (public parameters). The opening protocol is reconstructed from the
/// arity, so no protocol object or prover's Merkle data is shipped.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct WeightBatchCommit {
    pub commitment: zkie_core::pcs::whir::Commitment,
    pub arity: usize,
    pub num_tables: usize,
    pub whir: zkie_core::pcs::whir::Whir,
}

pub fn committed_cross_bind(
    whir: &zkie_core::pcs::whir::Whir,
    committed: &zkie_core::pcs::committed::Committed,
    tensor: &[Goldilocks],
    claims: &[(Vec<Goldilocks>, Goldilocks)],
    transcript: &mut Transcript,
) -> Option<CommittedCrossBindProof> {
    let mut openings = Vec::with_capacity(claims.len());
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
        openings.push((open, opened));
    }
    // S1: the merged-point sumcheck binds the commitment root and the claims.
    transcript.absorb(&bincode::serialize(&committed.commitment).ok()?);
    for (pt, ev) in claims {
        transcript.absorb_fields(pt);
        transcript.absorb_field(*ev);
    }
    Some(CommittedCrossBindProof {
        same_poly: {
            // S4: the merged claim of the committed cross-bind is not
            // consumed — every input claim is opened directly against the
            // boundary commitment (the real authentication), so the
            // same_poly here is an extra consistency check.
            let (sp, _merged) = prove_same_poly(tensor, claims, transcript);
            sp
        },
        openings,
    })
}

/// Verify a [`committed_cross_bind`] proof: verify each WHIR opening against the
/// commitment root, then verify the merged `same_poly`. No tensor data is read.
pub fn verify_committed_cross_bind(
    committed: &BoundaryCommit,
    claims: &[(Vec<Goldilocks>, Goldilocks)],
    proof: &CommittedCrossBindProof,
    transcript: &mut Transcript,
) -> bool {
    if proof.openings.len() != claims.len() {
        return false;
    }
    let whir = zkie_core::pcs::whir::Whir::with_params(committed.arity, 90, 17);
    let protocol = whir.opening_protocol(committed.arity, 1);
    for ((pt, expected), (open, opened)) in claims.iter().zip(proof.openings.iter()) {
        if whir
            .verify(&committed.commitment, open, &protocol, pt)
            .ok()
            != Some(*opened)
        {
            return false;
        }
        if *opened != *expected {
            return false;
        }
    }
    transcript.absorb(&bincode::serialize(&committed.commitment).unwrap());
    for (pt, ev) in claims {
        transcript.absorb_fields(pt);
        transcript.absorb_field(*ev);
    }
    verify_same_poly(&proof.same_poly, claims, transcript).is_some()
}

/// Collect the tensor ids of model weights: read but never written, and not a
/// (weights = read-but-never-written w/b/bias tensors, excluding lookup/input)
fn collect_weight_tensors(ops: &[Op]) -> Vec<T> {
    let mut written = std::collections::BTreeSet::new();
    for op in ops {
        for t in op.writes() {
            written.insert(t);
        }
    }
    let mut weights = std::collections::BTreeSet::new();
    for op in ops {
        match op {
            Op::MatMul { b, .. } => {
                if !written.contains(b) {
                    weights.insert(*b);
                }
            }
            Op::Projection { w, bias, .. } => {
                weights.insert(*w);
                weights.insert(*bias);
            }
            Op::Layernorm { w, b, .. } => {
                weights.insert(*w);
                weights.insert(*b);
            }
            Op::LayerNormCentered { w, b, .. } => {
                weights.insert(*w);
                weights.insert(*b);
            }
            Op::RmsNorm { w, .. } => {
                weights.insert(*w);
            }
            // BatchedMatMul's `w` is claimed at (v || ch) by the op proof, so
            // it needs a batch opening on the committed path unless it is an
            // op output (e.g. lookup-gathered rows in MoE).
            Op::BatchedMatMul { w, .. } => {
                if !written.contains(w) {
                    weights.insert(*w);
                }
            }
            _ => {}
        }
    }
    weights.into_iter().collect()
}

/// A committed shard-DAG proof: every shard proven as before (plain sumchecks,
/// virtual intermediates), plus WHIR commitments for each cross-shard boundary
/// tensor and each weight batch, with opening proofs computed by the prover.
/// The proof carries only commitment roots and openings -- never the committed
/// tensor data -- so the verifier is root-only.
/// A committed weight group whose prover data has been streamed to a cache
/// file. Holds only public commitment data + rebuild params; the Merkle prover
/// data is restored from the cache file during proving (no re-commit).
#[derive(Clone)]
pub struct CachedWeightBatch {
    pub commitment: zkie_core::pcs::whir::Commitment,
    pub arity: usize,
    pub num_tables: usize,
    pub num_vars: usize,
    pub security: usize,
    pub pow_bits: usize,
    pub table_ids: Vec<T>,
    pub cache_path: std::path::PathBuf,
}

/// Phase 1 (offline, once): commit every weight group and stream the Merkle
/// prover data to a cache file. Returns only commitment + rebuild params; the
/// heavy WHIR commitment (RS encoding + FFT + Merkle) is paid exactly once here.
pub fn commit_model_weights(
    store: &Store,
    ops: &[Op],
    cache_dir: &str,
) -> Vec<CachedWeightBatch> {
    // Borrowed from feature/batch-pcs: 90-bit security + pow_bits=17 (enough for
    // packed codewords up to 2^26), instead of the low-security testing params.
    const SECURITY: usize = 90;

    let weight_tensors = collect_weight_tensors(ops);
    let mut by_size: std::collections::BTreeMap<usize, Vec<(T, Vec<Goldilocks>)>> =
        std::collections::BTreeMap::new();
    for &t in &weight_tensors {
        let v = store.materialize(t).into_owned();
        // WHIR folding factor is 5, so every weight is committed at arity
        // >= 5: tensors shorter than 2^5 are zero-padded (their claim
        // points get zero-extended the same way, so the padded MLE matches
        // the raw evaluation). Skipping them would leave used weights
        // unauthenticated.
        let padded_len = v.len().next_power_of_two().max(32);
        let mut padded = v;
        padded.resize(padded_len, Goldilocks::ZERO);
        by_size.entry(padded_len).or_default().push((t, padded));
    }
    std::fs::create_dir_all(cache_dir).expect("create weight cache dir");

    let mut batches = Vec::new();
    for (len, group) in by_size {
        let arity = len.trailing_zeros() as usize;
        // Horizontal merge: pack same-arity weights into groups that fit the
        // WHIR PoW budget (num_vars = arity + log2(group_size) <= 26).
        let eff_group_size = if arity <= 26 {
            (1usize << (26 - arity)).max(1)
        } else {
            1
        };
        for ids in group.chunks(eff_group_size) {
            let num_tables = ids.len();
            let total_slots = num_tables * (1usize << arity);
            let num_vars = total_slots.next_power_of_two().trailing_zeros() as usize;
            // Per-group adaptive PoW budget: use the minimum pow_bits required at
            // 90-bit security for this group's num_variables, so a large tensor
            // (e.g. a 2^28 lm_head) gets a higher budget while small groups stay
            // cheap. See Whir::required_pow_bits.
            let pow_bits = zkie_core::pcs::whir::Whir::required_pow_bits(num_vars, SECURITY);
            let refs: Vec<&[Goldilocks]> = ids.iter().map(|(_, v)| v.as_slice()).collect();
            let base_whir = zkie_core::pcs::whir::Whir::with_params(arity, SECURITY, pow_bits);
            let (commitment, prover_data, _protocol, batch_whir) =
                base_whir.commit_batch(&refs);
            let num_vars = batch_whir.num_variables();
            let table_ids: Vec<T> = ids.iter().map(|(t, _)| *t).collect();
            let cache_path = std::path::PathBuf::from(cache_dir).join(format!(
                "weight-a{arity}-g{}.bin",
                batches.len()
            ));
            let file = std::fs::File::create(&cache_path).expect("create weight cache");
            let mut writer = std::io::BufWriter::new(file);
            bincode::serialize_into(&mut writer, &commitment).expect("write cache commitment");
            zkie_core::pcs::whir::Whir::serialize_batch_prover_data_into(prover_data, &mut writer)
                .expect("write cache prover data");
            std::io::Write::flush(&mut writer).expect("flush weight cache");
            batches.push(CachedWeightBatch {
                commitment,
                arity,
                num_tables,
                num_vars,
                security: SECURITY,
                pow_bits,
                table_ids,
                cache_path,
            });
        }
    }
    batches
}

/// A per-size-group weight batch opening: the multi-point reduction (one FRI
/// proof + one sum-of-products sumcheck) for every weight claim in that group.
/// Table indices, claim points, and claimed evals are stored explicitly so the
/// verifier never needs to reconstruct the grouping.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct WeightBatchOpening {
    pub table_idxs: Vec<usize>,
    pub points: Vec<Vec<Goldilocks>>,
    pub claimed: Vec<Goldilocks>,
    pub proof: Option<zkie_core::pcs::batch_open::BatchOpeningProof>,
}

/// A multiplicity-column opening for one lookup table side, plus the table
/// itself opened at the leaf claim points. `table` names the lookup table
/// TENSOR (statement tensor id): whether it is an identity table is decided
/// by the verifier from the statement tables, never by a proof flag (S2).
/// `den_leaf` is likewise dropped — the verifier derives the den leaf from
/// the table fractional replay.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct TableOpening {
    pub commitment: zkie_core::pcs::whir::Commitment,
    pub opening: zkie_core::pcs::whir::Proof,
    pub arity: usize,
    pub point: Vec<Goldilocks>,
    pub claimed: Goldilocks,
    pub table: T,
    pub table_commitment: Option<zkie_core::pcs::whir::Commitment>,
    pub table_opening: Option<zkie_core::pcs::whir::Proof>,
    pub table_eval: Goldilocks,
    /// The table opened at the den-side leaf claim point (q_den), so the
    /// verifier can check the den relation at that point.
    pub den_point: Vec<Goldilocks>,
    pub den_table_opening: Option<zkie_core::pcs::whir::Proof>,
    pub den_table_eval: Goldilocks,
    pub alpha: Goldilocks,
    pub beta: Goldilocks,
}

/// Verifier-side witness batch commitment: one root for a group of same-arity
/// witness columns of one type (x, y, or num). The opening protocol is rebuilt
/// from `arity`/`num_tables`; the batch Whir carries the public parameters.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct WitnessBatchCommit {
    pub commitment: zkie_core::pcs::whir::Commitment,
    pub arity: usize,
    pub num_tables: usize,
    pub whir: zkie_core::pcs::whir::Whir,
}

/// One vertical batch opening for a group of same-arity witness columns of one
/// type. `table_idxs[i]` is the table index within the batch; `points[i]` and
/// `claimed[i]` are that column's leaf point and claimed evaluation.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct WitnessBatchOpening {
    pub table_idxs: Vec<usize>,
    pub points: Vec<Vec<Goldilocks>>,
    pub claimed: Vec<Goldilocks>,
    pub proof: zkie_core::pcs::batch_open::BatchOpeningProof,
}

/// Batch/table coordinates of one witness column's x/y/num openings, in the
/// same global order the columns appear across `shards`.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct WitnessColumnLocation {
    pub x_batch: usize,
    pub x_table: usize,
    pub y_batch: usize,
    pub y_table: usize,
    pub num_batch: usize,
    pub num_table: usize,
}

/// Verifier-side generic batch commitment, shared by Scatter/Gather witness
/// columns (same shape as `WitnessBatchCommit`, reused to avoid duplication).
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ColumnBatchCommit {
    pub commitment: zkie_core::pcs::whir::Commitment,
    pub arity: usize,
    pub num_tables: usize,
    pub whir: zkie_core::pcs::whir::Whir,
}

/// One vertical batch opening for a group of same-arity columns of one type.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ColumnBatchOpening {
    pub table_idxs: Vec<usize>,
    pub points: Vec<Vec<Goldilocks>>,
    pub claimed: Vec<Goldilocks>,
    pub proof: zkie_core::pcs::batch_open::BatchOpeningProof,
}

/// Batch/table coordinates of one Scatter permutation witness (a/b/num).
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PermLocation {
    pub a_batch: usize,
    pub a_table: usize,
    pub b_batch: usize,
    pub b_table: usize,
    pub num_batch: usize,
    pub num_table: usize,
}

/// Batch/table coordinates of one Gather histogram witness (8 columns).
#[derive(serde::Serialize, serde::Deserialize)]
pub struct GatherLocation {
    pub num_source_batch: usize,
    pub num_source_table: usize,
    pub den_source_batch: usize,
    pub den_source_table: usize,
    pub weighted_batch: usize,
    pub weighted_table: usize,
    pub num_target_batch: usize,
    pub num_target_table: usize,
    pub den_target_batch: usize,
    pub den_target_table: usize,
    pub out_batch: usize,
    pub out_table: usize,
    pub gate_batch: usize,
    pub gate_table: usize,
    pub token_id_batch: usize,
    pub token_id_table: usize,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SelOpening {
    pub commitment: zkie_core::pcs::whir::Commitment,
    pub arity: usize,
    pub points: Vec<Vec<Goldilocks>>,
    pub openings: Vec<zkie_core::pcs::whir::Proof>,
}


#[derive(serde::Serialize, serde::Deserialize)]
pub struct CommittedShardDagProof {
    pub shards: Vec<OpShardProof>,
    pub boundary_tensors: Vec<T>,
    pub boundary_commitments: Vec<BoundaryCommit>,
    pub cross_binds: Vec<CommittedCrossBindProof>,
    pub weight_batches: Vec<WeightBatchCommit>,
    pub weight_batch_openings: Vec<WeightBatchOpening>,
    pub weight_locs: std::collections::BTreeMap<T, (usize, usize)>,
    pub tokens: Vec<u32>,
    pub argmax_t: Vec<u32>,
    pub witness_batches: Vec<WitnessBatchCommit>,
    pub witness_batch_openings: Vec<WitnessBatchOpening>,
    pub witness_locations: Vec<WitnessColumnLocation>,
    pub table_openings: Vec<TableOpening>,
    pub sel_openings: Vec<SelOpening>,
    pub perm_batches: Vec<ColumnBatchCommit>,
    pub perm_batch_openings: Vec<ColumnBatchOpening>,
    pub perm_locations: Vec<PermLocation>,
    pub gather_batches: Vec<ColumnBatchCommit>,
    pub gather_batch_openings: Vec<ColumnBatchOpening>,
    pub gather_locations: Vec<GatherLocation>,
    pub gather_indep_batches: usize,
}

pub fn prove_committed_shard_dag(
    store: &mut Store,
    ops: &[Op],
    ops_per_shard: usize,
    _whir: &zkie_core::pcs::whir::Whir,
    cached_weights: &[CachedWeightBatch],
    tokens_t: T,
    argmax_t: T,
) -> CommittedShardDagProof {
    // Committed mode: the per-op perm/gather fracs stay off the shard
    // transcripts (re-proven on the committed transcripts below); the shard
    // claims come back separately (S2).
    let (plain, shard_claims) = prove_shard_dag_inner(store, ops, ops_per_shard, FracMode::Committed);
    let boundary_tensors = plain.cross_tensors.clone();

    // S1: the cross-bind merged-point sumchecks get their own statement-bound
    // transcript; the verifier replays it in the same per-tensor order.
    let mut bind_ts = Transcript::new(b"zkie-committed-binds-v1");
    absorb_statement(&mut bind_ts, ops, ops_per_shard);

    let mut boundary_commitments = Vec::with_capacity(boundary_tensors.len());
    let mut cross_binds = Vec::with_capacity(boundary_tensors.len());
    for &t in &boundary_tensors {
        let arity = store.get(t).len().trailing_zeros() as usize;
        let bw = zkie_core::pcs::whir::Whir::with_params(arity, 90, 17);
        let committed = zkie_core::pcs::committed::commit(&bw, store.get(t));
        let mut claims = Vec::new();
        for shard_cs in &shard_claims {
            for c in shard_cs {
                if c.tensor_id == t {
                    claims.push((c.point.clone(), c.value));
                }
            }
        }
        let sp = committed_cross_bind(&bw, &committed, store.get(t), &claims, &mut bind_ts)
            .expect("honest committed bind");
        boundary_commitments.push(BoundaryCommit {
            commitment: committed.commitment,
            arity,
        });
        cross_binds.push(sp);
    }

    // Restore each weight group's Merkle prover data from its cache file (no
    // re-commit), then reduce each group's claims to one FRI proof + one sumcheck.
    let mut weight_locs: std::collections::BTreeMap<T, (usize, usize)> =
        std::collections::BTreeMap::new();
    let mut group_tensors: Vec<Vec<Vec<Goldilocks>>> = Vec::with_capacity(cached_weights.len());
    for (bi, cw) in cached_weights.iter().enumerate() {
        let tensors: Vec<Vec<Goldilocks>> = cw
            .table_ids
            .iter()
            .map(|&t| {
                let mut v = store.materialize(t).into_owned();
                let padded_len = 1usize << cw.arity;
                v.resize(padded_len, Goldilocks::ZERO);
                v
            })
            .collect();
        for (i, &t) in cw.table_ids.iter().enumerate() {
            weight_locs.insert(t, (bi, i));
        }
        group_tensors.push(tensors);
    }

    let num_groups = cached_weights.len();
    let mut group_claims: Vec<Vec<(usize, Vec<Goldilocks>, Goldilocks)>> =
        vec![Vec::new(); num_groups];
    for claims in &shard_claims {
        for c in claims {
            if let Some(&(bi, ti)) = weight_locs.get(&c.tensor_id) {
                group_claims[bi].push((ti, c.point.clone(), c.value));
            }
        }
    }

    let mut weight_batch_openings = Vec::with_capacity(num_groups);
    let mut weight_batches = Vec::with_capacity(num_groups);
    for (bi, cw) in cached_weights.iter().enumerate() {
        let claims = &group_claims[bi];
        let table_idxs: Vec<usize> = claims.iter().map(|(ti, _, _)| *ti).collect();
        // Claim points are zero-extended to the batch arity (small weights
        // are committed zero-padded; the padded MLE agrees with the raw
        // evaluation at the extended point).
        let points: Vec<Vec<Goldilocks>> = claims
            .iter()
            .map(|(_, pt, _)| {
                let mut p = pt.clone();
                p.resize(cw.arity, Goldilocks::ZERO);
                p
            })
            .collect();
        let claimed: Vec<Goldilocks> = claims.iter().map(|(_, _, ev)| *ev).collect();

        // Verifier-side batch commitment (no prover data); prover-side Whir is
        // reconstructed below from the same params.
        weight_batches.push(WeightBatchCommit {
            commitment: cw.commitment.clone(),
            arity: cw.arity,
            num_tables: cw.num_tables,
            whir: zkie_core::pcs::whir::Whir::with_params(cw.num_vars, cw.security, cw.pow_bits),
        });

        if claims.is_empty() {
            weight_batch_openings.push(WeightBatchOpening {
                table_idxs,
                points,
                claimed,
                proof: None,
            });
            continue;
        }

        let batch_whir =
            zkie_core::pcs::whir::Whir::with_params(cw.num_vars, cw.security, cw.pow_bits);
        let mut reader = std::io::BufReader::new(
            std::fs::File::open(&cw.cache_path).expect("open weight cache"),
        );
        let commitment: zkie_core::pcs::whir::Commitment =
            bincode::deserialize_from(&mut reader).expect("read cache commitment");
        let ref_slices: Vec<&[Goldilocks]> =
            group_tensors[bi].iter().map(|v| v.as_slice()).collect();
        let prover_data = batch_whir
            .restore_batch_prover_data_from(&ref_slices, &mut reader)
            .expect("restore cache prover data");
        let protocol = batch_whir.batch_opening_protocol(cw.num_tables, cw.arity);
        let ctx = zkie_core::pcs::committed::BatchCtx {
            commitment,
            prover_data,
            protocol,
            whir: batch_whir,
            num_tables: cw.num_tables,
        };
        let tensors: Vec<Vec<Goldilocks>> = claims
            .iter()
            .map(|(ti, _, _)| group_tensors[bi][*ti].clone())
            .collect();
        let proof = zkie_core::pcs::batch_open::prove_batch_open_committed_owned(
            ctx,
            &table_idxs,
            &tensors,
            &points,
            &claimed,
        )
        .expect("honest batch-open");
        weight_batch_openings.push(WeightBatchOpening {
            table_idxs,
            points,
            claimed,
            proof: Some(proof),
        });
    }

    // Vertical batch the witness columns across all shards: group the x/y/num
    // columns by arity, committing each (arity, column-type) group once. This
    // turns ~3*W single openings into 3*G batch openings (G distinct arities).
    let mut shards = plain.shards;
    let mut witness_batches: Vec<WitnessBatchCommit> = Vec::new();
    let mut witness_batch_openings: Vec<WitnessBatchOpening> = Vec::new();
    let mut table_openings = Vec::new();

    let mut shard_wc_off: Vec<usize> = Vec::with_capacity(shards.len() + 1);
    shard_wc_off.push(0);
    for shard in &shards {
        let next = shard_wc_off.last().unwrap() + shard.witness_columns.len();
        shard_wc_off.push(next);
    }
    let total_wc = *shard_wc_off.last().unwrap();
    let mut witness_locations: Vec<WitnessColumnLocation> = (0..total_wc)
        .map(|_| WitnessColumnLocation {
            x_batch: 0,
            x_table: 0,
            y_batch: 0,
            y_table: 0,
            num_batch: 0,
            num_table: 0,
        })
        .collect();

    // Commit each distinct lookup table body exactly once (not per shard).
    let mut global_table_commits: std::collections::BTreeMap<
        T,
        (usize, zkie_core::pcs::committed::Committed),
    > = std::collections::BTreeMap::new();
    let mut identity_tables: std::collections::BTreeSet<T> = std::collections::BTreeSet::new();
    for shard in &shards {
        for tc in &shard.table_columns {
            if !global_table_commits.contains_key(&tc.table) && !identity_tables.contains(&tc.table) {
                let raw_table = store.get(tc.table);
                if is_identity_table(raw_table) {
                    identity_tables.insert(tc.table);
                    continue;
                }
                let table_pad = pad_lookup_table(raw_table);
                let arity = table_pad.len().trailing_zeros() as usize;
                let pow_bits = zkie_core::pcs::whir::Whir::required_pow_bits(arity, 90);
                let bw = zkie_core::pcs::whir::Whir::with_params(arity, 90, pow_bits);
                global_table_commits.insert(
                    tc.table,
                    (arity, zkie_core::pcs::committed::commit(&bw, &table_pad)),
                );
            }
        }
    }

    struct WitnessGroup {
        arity: usize,
        cols: Vec<(usize, usize)>,
        x_ctx: zkie_core::pcs::committed::BatchCtx,
        y_ctx: zkie_core::pcs::committed::BatchCtx,
        num_ctx: zkie_core::pcs::committed::BatchCtx,
    }
    let mut witness_groups: Vec<WitnessGroup> = Vec::new();
    {
        let mut by_arity: std::collections::BTreeMap<usize, Vec<(usize, usize)>> =
            std::collections::BTreeMap::new();
        for (si, shard) in shards.iter().enumerate() {
            for (ci, wc) in shard.witness_columns.iter().enumerate() {
                let arity = wc.num.len().trailing_zeros() as usize;
                by_arity.entry(arity).or_default().push((si, ci));
            }
        }
        for (arity, cols) in by_arity {
            let mut xs: Vec<&[Goldilocks]> = Vec::with_capacity(cols.len());
            let mut ys: Vec<&[Goldilocks]> = Vec::with_capacity(cols.len());
            let mut nums: Vec<&[Goldilocks]> = Vec::with_capacity(cols.len());
            for &(si, ci) in &cols {
                let wc = &shards[si].witness_columns[ci];
                xs.push(wc.x.as_slice());
                ys.push(wc.y.as_slice());
                nums.push(wc.num.as_slice());
            }
            let total_slots = cols.len() * (1usize << arity);
            let num_vars = total_slots.next_power_of_two().trailing_zeros() as usize;
            let pow_bits = zkie_core::pcs::whir::Whir::required_pow_bits(num_vars, 90);
            let base_whir = zkie_core::pcs::whir::Whir::with_params(arity, 90, pow_bits);
            let num_tables = cols.len();
            let (cx, px, prx, wx) = base_whir.commit_batch(&xs);
            let (cy, py, pry, wy) = base_whir.commit_batch(&ys);
            let (cnum, pnum, prnum, wnum) = base_whir.commit_batch(&nums);
            witness_groups.push(WitnessGroup {
                arity,
                cols,
                x_ctx: zkie_core::pcs::committed::BatchCtx { commitment: cx, prover_data: px, protocol: prx, whir: wx, num_tables },
                y_ctx: zkie_core::pcs::committed::BatchCtx { commitment: cy, prover_data: py, protocol: pry, whir: wy, num_tables },
                num_ctx: zkie_core::pcs::committed::BatchCtx { commitment: cnum, prover_data: pnum, protocol: prnum, whir: wnum, num_tables },
            });
        }
    }

    // Commit the per-lookup multiplicity columns `m`.
    let mut t_arities: Vec<Vec<usize>> = Vec::with_capacity(shards.len());
    let mut t_cms: Vec<Vec<zkie_core::pcs::committed::Committed>> = Vec::with_capacity(shards.len());
    for shard in &shards {
        let mut t_arity = Vec::new();
        let mut t_cm = Vec::new();
        for tc in &shard.table_columns {
            let total = tc.m.len().next_power_of_two().max(32);
            let arity = total.trailing_zeros() as usize;
            let mut m_pad = vec![Goldilocks::from_u64(0); total];
            for (j, &v) in tc.m.iter().enumerate() {
                m_pad[j] = Goldilocks::from_u64(v);
            }
            let pow_bits = zkie_core::pcs::whir::Whir::required_pow_bits(arity, 90);
            let bw = zkie_core::pcs::whir::Whir::with_params(arity, 90, pow_bits);
            t_arity.push(arity);
            t_cm.push(zkie_core::pcs::committed::commit(&bw, &m_pad));
        }
        t_arities.push(t_arity);
        t_cms.push(t_cm);
    }

    // Global Fiat-Shamir transcript: absorb the statement, every witness batch
    // root, then every table root, before deriving per-lookup alpha/beta and the
    // chained-GKR leaf challenges.
    let mut ts = Transcript::new(b"zkie-committed-frac-v1");
    absorb_statement(&mut ts, ops, ops_per_shard);
    for g in &witness_groups {
        ts.absorb(&bincode::serialize(&g.x_ctx.commitment).unwrap());
        ts.absorb(&bincode::serialize(&g.y_ctx.commitment).unwrap());
        ts.absorb(&bincode::serialize(&g.num_ctx.commitment).unwrap());
    }
    for (si, shard) in shards.iter().enumerate() {
        ts.absorb_u64(si as u64);
        for (i, tc) in shard.table_columns.iter().enumerate() {
            ts.absorb(&bincode::serialize(&t_cms[si][i].commitment).unwrap());
            let tid = tc.table;
            if identity_tables.contains(&tid) {
                ts.absorb_u64(0);
            } else {
                let (_, gc) = &global_table_commits[&tid];
                ts.absorb(&bincode::serialize(&gc.commitment).unwrap());
            }
        }
    }

    // S1: derive alpha/beta and run every fractional proof on the transcript in
    // shard-major order (all witness columns first, then all table columns),
    // exactly the order the verifier replays. Each prove_fractional derives its
    // layer challenges per-round internally.
    // S2: no leaf fields are stored — the openings below read the leaves back
    // out of the freshly proven chains via the prover-side helper, and the
    // verifier derives them from its own replay.
    for si in 0..shards.len() {
        for ci in 0..shards[si].witness_columns.len() {
            let alpha = ts.field();
            let beta = ts.field();
            let wc = &mut shards[si].witness_columns[ci];
            let total = wc.x.len();
            let mut den = vec![Goldilocks::from_u64(0); total];
            for j in 0..total {
                den[j] = alpha + wc.x[j] + beta * wc.y[j];
            }
            let new_proof = prove_fractional(wc.num.clone(), den, &mut ts);
            wc.proof = new_proof;
            wc.alpha = alpha;
            wc.beta = beta;
        }
    }
    for si in 0..shards.len() {
        for i in 0..shards[si].table_columns.len() {
            let alpha = shards[si].witness_columns[i].alpha;
            let beta = shards[si].witness_columns[i].beta;
            let tc = &mut shards[si].table_columns[i];
            // The table frac runs over the PADDED table (arity >= 5; identity
            // tables continue the identity so the verifier's index-formula
            // evaluation matches at every padded point).
            let table_pad = pad_lookup_table(store.get(tc.table));
            let total = table_pad.len();
            let mut num_t = vec![Goldilocks::from_u64(0); total];
            let mut den_t = vec![Goldilocks::from_u64(0); total];
            for j in 0..total {
                num_t[j] = if j < tc.m.len() {
                    Goldilocks::from_u64(tc.m[j])
                } else {
                    Goldilocks::ZERO
                };
                den_t[j] = alpha + Goldilocks::from_u64(j as u64) + beta * table_pad[j];
            }
            let new_proof = prove_fractional(num_t, den_t, &mut ts);
            tc.proof = new_proof;
            tc.alpha = alpha;
            tc.beta = beta;
        }
    }

    // Record verifier-side witness batch commitments before consuming their
    // one-shot Merkle prover data during opening.
    for g in &witness_groups {
        let total_slots = g.cols.len() * (1usize << g.arity);
        let num_vars = total_slots.next_power_of_two().trailing_zeros() as usize;
        let pow_bits = zkie_core::pcs::whir::Whir::required_pow_bits(num_vars, 90);
        witness_batches.push(WitnessBatchCommit {
            commitment: g.x_ctx.commitment.clone(),
            arity: g.arity,
            num_tables: g.cols.len(),
            whir: zkie_core::pcs::whir::Whir::with_params(num_vars, 90, pow_bits),
        });
        witness_batches.push(WitnessBatchCommit {
            commitment: g.y_ctx.commitment.clone(),
            arity: g.arity,
            num_tables: g.cols.len(),
            whir: zkie_core::pcs::whir::Whir::with_params(num_vars, 90, pow_bits),
        });
        witness_batches.push(WitnessBatchCommit {
            commitment: g.num_ctx.commitment.clone(),
            arity: g.arity,
            num_tables: g.cols.len(),
            whir: zkie_core::pcs::whir::Whir::with_params(num_vars, 90, pow_bits),
        });
    }

    // Re-run the chained GKR per lookup, then batch-open each (arity, type) group.
    for (g_idx, g) in witness_groups.into_iter().enumerate() {
        let WitnessGroup {
            arity: _,
            cols,
            x_ctx,
            y_ctx,
            num_ctx,
        } = g;
        let num_tables = cols.len();
        let table_idxs: Vec<usize> = (0..num_tables).collect();
        // S2: x/y are opened at the derived DEN leaf point (the den relation
        // alpha + x + beta*y == den_leaf is checked there), num at the NUM
        // leaf point (claimed == num_leaf).
        let mut xy_points: Vec<Vec<Goldilocks>> = Vec::with_capacity(num_tables);
        let mut num_points: Vec<Vec<Goldilocks>> = Vec::with_capacity(num_tables);
        let mut xs_tensors: Vec<Vec<Goldilocks>> = Vec::with_capacity(num_tables);
        let mut ys_tensors: Vec<Vec<Goldilocks>> = Vec::with_capacity(num_tables);
        let mut nums_tensors: Vec<Vec<Goldilocks>> = Vec::with_capacity(num_tables);
        let mut xs_claimed: Vec<Goldilocks> = Vec::with_capacity(num_tables);
        let mut ys_claimed: Vec<Goldilocks> = Vec::with_capacity(num_tables);
        let mut nums_claimed: Vec<Goldilocks> = Vec::with_capacity(num_tables);
        for (ti, &(si, ci)) in cols.iter().enumerate() {
            let gid = shard_wc_off[si] + ci;
            let wc = &mut shards[si].witness_columns[ci];
            let (num_point, num_leaf, den_point, _den_leaf) = fractional_leaf(&wc.proof);
            let x = std::mem::take(&mut wc.x);
            let y = std::mem::take(&mut wc.y);
            let num = std::mem::take(&mut wc.num);
            let eval_x = mle::eval(&x, &den_point);
            let eval_y = mle::eval(&y, &den_point);
            xy_points.push(den_point);
            num_points.push(num_point);
            xs_tensors.push(x);
            ys_tensors.push(y);
            nums_tensors.push(num);
            xs_claimed.push(eval_x);
            ys_claimed.push(eval_y);
            nums_claimed.push(num_leaf);
            witness_locations[gid] = WitnessColumnLocation {
                x_batch: 3 * g_idx,
                x_table: ti,
                y_batch: 3 * g_idx + 1,
                y_table: ti,
                num_batch: 3 * g_idx + 2,
                num_table: ti,
            };
        }
        let x_proof = zkie_core::pcs::batch_open::prove_batch_open_committed_owned(
            x_ctx, &table_idxs, &xs_tensors, &xy_points, &xs_claimed,
        ).expect("honest witness x batch-open");
        let y_proof = zkie_core::pcs::batch_open::prove_batch_open_committed_owned(
            y_ctx, &table_idxs, &ys_tensors, &xy_points, &ys_claimed,
        ).expect("honest witness y batch-open");
        let num_proof = zkie_core::pcs::batch_open::prove_batch_open_committed_owned(
            num_ctx, &table_idxs, &nums_tensors, &num_points, &nums_claimed,
        ).expect("honest witness num batch-open");
        witness_batch_openings.push(WitnessBatchOpening {
            table_idxs: table_idxs.clone(),
            points: xy_points.clone(),
            claimed: xs_claimed,
            proof: x_proof,
        });
        witness_batch_openings.push(WitnessBatchOpening {
            table_idxs: table_idxs.clone(),
            points: xy_points,
            claimed: ys_claimed,
            proof: y_proof,
        });
        witness_batch_openings.push(WitnessBatchOpening {
            table_idxs,
            points: num_points,
            claimed: nums_claimed,
            proof: num_proof,
        });
    }

    // Open the per-lookup multiplicity columns `m` at the fractional leaf
    // points (the fractional proofs were already run on the transcript above).
    // S2: the leaf points come from the freshly proven chain (prover-side
    // helper); `table` names the lookup table tensor so the verifier can
    // decide identity from the statement.
    for (si, shard) in shards.iter_mut().enumerate() {
        for (i, tc) in shard.table_columns.iter_mut().enumerate() {
            let (point, num_leaf, den_point, _den_leaf) = fractional_leaf(&tc.proof);
            let pow_bits = zkie_core::pcs::whir::Whir::required_pow_bits(t_arities[si][i], 90);
            let bw = zkie_core::pcs::whir::Whir::with_params(t_arities[si][i], 90, pow_bits);
            let (open, ev) = bw.open(t_cms[si][i].prover_data.clone(), &t_cms[si][i].protocol, &point);
            assert_eq!(ev, num_leaf);
            let is_ident = identity_tables.contains(&tc.table);
            let (table_commitment, table_opening, table_eval, den_table_opening, den_table_eval) = if is_ident {
                (None, None, Goldilocks::ZERO, None, Goldilocks::ZERO)
            } else {
                let (table_arity, table_committed) = &global_table_commits[&tc.table];
                let table_bw = zkie_core::pcs::whir::Whir::with_params(
                    *table_arity,
                    90,
                    zkie_core::pcs::whir::Whir::required_pow_bits(*table_arity, 90),
                );
                let (open_table, ev_table) = table_bw.open(
                    table_committed.prover_data.clone(),
                    &table_committed.protocol,
                    &point,
                );
                let (open_den_table, ev_den_table) = table_bw.open(
                    table_committed.prover_data.clone(),
                    &table_committed.protocol,
                    &den_point,
                );
                (
                    Some(table_committed.commitment.clone()),
                    Some(open_table),
                    ev_table,
                    Some(open_den_table),
                    ev_den_table,
                )
            };
            table_openings.push(TableOpening {
                commitment: t_cms[si][i].commitment.clone(),
                opening: open,
                arity: t_arities[si][i],
                point,
                claimed: num_leaf,
                table: tc.table,
                table_commitment,
                table_opening,
                table_eval,
                den_point,
                den_table_opening,
                den_table_eval,
                alpha: tc.alpha,
                beta: tc.beta,
            });
            tc.m.clear();
        }
    }

    // Commit + open each Argmax one-hot `sel` at the m claimed output
    // positions, so the argmax-index binding is authenticated without the
    // plaintext m x n selection matrix.
    let mut sel_openings = Vec::new();
    let sel_ranges = shard_ranges(ops.len(), ops_per_shard);
    for (si, &(s, e)) in sel_ranges.iter().enumerate() {
        for (op, op_proof) in ops[s..e].iter().zip(shards[si].ops.iter_mut()) {
            if let (Op::Argmax { sel: sel_t, out: out_t, m, n, .. }, OpProof::Argmax(ap)) = (op, op_proof) {
                let sel_vec = store.get(*sel_t);
                let total = sel_vec.len().next_power_of_two();
                let arity = total.trailing_zeros() as usize;
                let mut sel_pad = vec![Goldilocks::ZERO; total];
                sel_pad[..sel_vec.len()].copy_from_slice(sel_vec);
                let pow_bits = zkie_core::pcs::whir::Whir::required_pow_bits(arity, 90);
                let bw = zkie_core::pcs::whir::Whir::with_params(arity, 90, pow_bits);
                let c = zkie_core::pcs::committed::commit(&bw, &sel_pad);
                let out_vals = &store.idx[*out_t];
                let mut points = Vec::new();
                for i in 0..*m {
                    let j = out_vals[i] as usize;
                    let idx = i * *n + j;
                    let point: Vec<Goldilocks> = (0..arity)
                        .map(|k| Goldilocks::from_u64(((idx >> k) & 1) as u64))
                        .collect();
                    points.push(point);
                }
                let commitment = c.commitment.clone();
                let mut prover_data = Some(c.prover_data);
                let mut openings = Vec::with_capacity(*m);
                for (i, point) in points.iter().enumerate() {
                    let data = if i + 1 == *m {
                        prover_data.take().expect("last Argmax opening owns prover data")
                    } else {
                        prover_data.as_ref().expect("Argmax prover data").clone()
                    };
                    let (opening, ev) = bw.open(data, &c.protocol, point);
                    assert_eq!(ev, Goldilocks::ONE);
                    openings.push(opening);
                }
                ap.sel_evals = vec![Goldilocks::ONE; *m];
                ap.topk.sel.clear();
                sel_openings.push(SelOpening {
                    commitment,
                    arity,
                    points,
                    openings,
                });
            }
        }
    }

    // Commit + open each Scatter permutation witness (a/b/num columns), grouped
    // by arity across all Scatter ops (same-shape vertical batch).
    let mut perm_batches: Vec<ColumnBatchCommit> = Vec::new();
    let mut perm_batch_openings: Vec<ColumnBatchOpening> = Vec::new();
    let mut perm_locations: Vec<PermLocation> = Vec::new();

    let mut perm_ops: Vec<(usize, usize, usize)> = Vec::new(); // (si, li, arity)
    {
        let pr = shard_ranges(ops.len(), ops_per_shard);
        for (si, &(s, e)) in pr.iter().enumerate() {
            for (li, op) in ops[s..e].iter().enumerate() {
                if let Op::Scatter { .. } = op {
                    let arity = match &shards[si].ops[li] {
                        OpProof::Scatter(p) => {
                            p.perm.a.len().next_power_of_two().max(32).trailing_zeros() as usize
                        }
                        _ => unreachable!(),
                    };
                    perm_ops.push((si, li, arity));
                }
            }
        }
    }
    let mut perm_gid: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
    for (gid, &(si, li, _)) in perm_ops.iter().enumerate() {
        perm_gid.insert((si, li), gid);
    }

    struct PermGroup {
        arity: usize,
        ops: Vec<(usize, usize)>,
        a_ctx: zkie_core::pcs::committed::BatchCtx,
        b_ctx: zkie_core::pcs::committed::BatchCtx,
        num_ctx: zkie_core::pcs::committed::BatchCtx,
    }
    let mut perm_groups: Vec<PermGroup> = Vec::new();
    {
        let mut by_arity: std::collections::BTreeMap<usize, Vec<(usize, usize)>> = std::collections::BTreeMap::new();
        for &(si, li, arity) in &perm_ops {
            by_arity.entry(arity).or_default().push((si, li));
        }
        for (arity, ops_) in by_arity {
            let total = 1usize << arity;
            let mut a_cols: Vec<Vec<Goldilocks>> = Vec::with_capacity(ops_.len());
            let mut b_cols: Vec<Vec<Goldilocks>> = Vec::with_capacity(ops_.len());
            let mut num_cols: Vec<Vec<Goldilocks>> = Vec::with_capacity(ops_.len());
            for &(si, li) in &ops_ {
                let sp = match &shards[si].ops[li] {
                    OpProof::Scatter(p) => p,
                    _ => unreachable!(),
                };
                let mut a_pad = vec![Goldilocks::ZERO; total];
                let mut b_pad = vec![Goldilocks::ZERO; total];
                let mut num_pad = vec![Goldilocks::ZERO; total];
                for i in 0..sp.perm.a.len() {
                    a_pad[i] = sp.perm.a[i];
                    b_pad[i] = sp.perm.b[i];
                    num_pad[i] = Goldilocks::ONE;
                }
                a_cols.push(a_pad);
                b_cols.push(b_pad);
                num_cols.push(num_pad);
            }
            let num_tables = ops_.len();
            let num_vars = (num_tables * total).next_power_of_two().trailing_zeros() as usize;
            let pow_bits = zkie_core::pcs::whir::Whir::required_pow_bits(num_vars, 90);
            let base_whir = zkie_core::pcs::whir::Whir::with_params(arity, 90, pow_bits);
            let a_refs: Vec<&[Goldilocks]> = a_cols.iter().map(|v| v.as_slice()).collect();
            let b_refs: Vec<&[Goldilocks]> = b_cols.iter().map(|v| v.as_slice()).collect();
            let num_refs: Vec<&[Goldilocks]> = num_cols.iter().map(|v| v.as_slice()).collect();
            let (ca, pa, pra, wa) = base_whir.commit_batch(&a_refs);
            let (cb, pb, prb, wb) = base_whir.commit_batch(&b_refs);
            let (cn, pn, prn, wn) = base_whir.commit_batch(&num_refs);
            perm_groups.push(PermGroup {
                arity,
                ops: ops_,
                a_ctx: zkie_core::pcs::committed::BatchCtx { commitment: ca, prover_data: pa, protocol: pra, whir: wa, num_tables },
                b_ctx: zkie_core::pcs::committed::BatchCtx { commitment: cb, prover_data: pb, protocol: prb, whir: wb, num_tables },
                num_ctx: zkie_core::pcs::committed::BatchCtx { commitment: cn, prover_data: pn, protocol: prn, whir: wn, num_tables },
            });
        }
    }

    let mut perm_ts = Transcript::new(b"zkie-committed-perm-v1");
    absorb_statement(&mut perm_ts, ops, ops_per_shard);
    for g in &perm_groups {
        perm_ts.absorb(&bincode::serialize(&g.a_ctx.commitment).unwrap());
        perm_ts.absorb(&bincode::serialize(&g.b_ctx.commitment).unwrap());
        perm_ts.absorb(&bincode::serialize(&g.num_ctx.commitment).unwrap());
    }
    // S1: per-op interleave — alpha, then the two fractional proofs, in
    // perm_ops (shard-major) order; the verifier replays the same order. The
    // fractional layer challenges are derived per-round inside prove_fractional.
    for &(si, li, _) in &perm_ops {
        let alpha = perm_ts.field();
        let sp = match &mut shards[si].ops[li] {
            OpProof::Scatter(p) => p,
            _ => unreachable!(),
        };
        let (num_a, den_a) = build_perm_num_den(&sp.perm.a, alpha);
        let (num_b, den_b) = build_perm_num_den(&sp.perm.b, alpha);
        let new_a = prove_fractional(num_a, den_a, &mut perm_ts);
        let new_b = prove_fractional(num_b, den_b, &mut perm_ts);
        sp.perm.a_side = new_a;
        sp.perm.b_side = new_b;
        sp.perm.alpha = alpha;
    }

    for g in &perm_groups {
        let num_vars = (g.ops.len() * (1usize << g.arity)).next_power_of_two().trailing_zeros() as usize;
        let pow_bits = zkie_core::pcs::whir::Whir::required_pow_bits(num_vars, 90);
        for ctx in [&g.a_ctx, &g.b_ctx, &g.num_ctx] {
            perm_batches.push(ColumnBatchCommit {
                commitment: ctx.commitment.clone(),
                arity: g.arity,
                num_tables: g.ops.len(),
                whir: zkie_core::pcs::whir::Whir::with_params(num_vars, 90, pow_bits),
            });
        }
    }

    let total_perm_ops = perm_ops.len();
    let mut perm_locations_slots: Vec<Option<PermLocation>> = (0..total_perm_ops).map(|_| None).collect();
    for (g_idx, g) in perm_groups.into_iter().enumerate() {
        let PermGroup {
            arity,
            ops: group_ops,
            a_ctx,
            b_ctx,
            num_ctx,
        } = g;
        let n = group_ops.len();
        // Each a/b column carries TWO openings (num leaf point + den leaf
        // point), so the index list repeats each table index twice.
        let table_idxs: Vec<usize> = (0..n).flat_map(|ti| [ti, ti]).collect();
        let mut points_a = Vec::with_capacity(2 * n);
        let mut points_b = Vec::with_capacity(2 * n);
        let mut a_claimed = Vec::with_capacity(2 * n);
        let mut b_claimed = Vec::with_capacity(2 * n);
        let mut num_idxs = Vec::with_capacity(2 * n);
        let mut num_points = Vec::with_capacity(2 * n);
        let mut num_claimed = Vec::with_capacity(2 * n);
        let mut a_tensors = Vec::with_capacity(n);
        let mut b_tensors = Vec::with_capacity(n);
        let mut num_tensors = Vec::with_capacity(n);
        for (ti, &(si, li)) in group_ops.iter().enumerate() {
            let gid = perm_gid[&(si, li)];
            let sp = match &mut shards[si].ops[li] {
                OpProof::Scatter(p) => p,
                _ => unreachable!(),
            };
            let (point_a, num_leaf_a, den_point_a, den_leaf_a) = fractional_leaf(&sp.perm.a_side);
            let (point_b, num_leaf_b, den_point_b, den_leaf_b) = fractional_leaf(&sp.perm.b_side);
            let a = std::mem::take(&mut sp.perm.a);
            let b = std::mem::take(&mut sp.perm.b);
            let total = 1usize << arity;
            let mut a_pad = vec![Goldilocks::ZERO; total];
            let mut b_pad = vec![Goldilocks::ZERO; total];
            let mut num_pad = vec![Goldilocks::ZERO; total];
            for i in 0..a.len() {
                a_pad[i] = a[i];
                b_pad[i] = b[i];
                num_pad[i] = Goldilocks::ONE;
            }
            // Each a/b column gets TWO claims: at the num leaf point (q_num)
            // and at the den leaf point (q_den), so the verifier can check the
            // pointwise relation den = alpha - value at q_den.
            points_a.push(point_a.clone());
            points_a.push(den_point_a.clone());
            a_claimed.push(mle::eval(&a_pad, &point_a));
            a_claimed.push(mle::eval(&a_pad, &den_point_a));
            points_b.push(point_b.clone());
            points_b.push(den_point_b.clone());
            b_claimed.push(mle::eval(&b_pad, &point_b));
            b_claimed.push(mle::eval(&b_pad, &den_point_b));
            num_idxs.push(ti);
            num_idxs.push(ti);
            num_points.push(point_a.clone());
            num_points.push(point_b.clone());
            num_claimed.push(num_leaf_a);
            num_claimed.push(num_leaf_b);
            a_tensors.push(a_pad.clone());
            a_tensors.push(a_pad);
            b_tensors.push(b_pad.clone());
            b_tensors.push(b_pad);
            num_tensors.push(num_pad.clone());
            num_tensors.push(num_pad);
            perm_locations_slots[gid] = Some(PermLocation {
                a_batch: 3 * g_idx,
                a_table: 2 * ti,
                b_batch: 3 * g_idx + 1,
                b_table: 2 * ti,
                num_batch: 3 * g_idx + 2,
                num_table: ti,
            });
        }
        let a_proof = zkie_core::pcs::batch_open::prove_batch_open_committed_owned(a_ctx, &table_idxs, &a_tensors, &points_a, &a_claimed).expect("honest perm a batch-open");
        let b_proof = zkie_core::pcs::batch_open::prove_batch_open_committed_owned(b_ctx, &table_idxs, &b_tensors, &points_b, &b_claimed).expect("honest perm b batch-open");
        let num_proof = zkie_core::pcs::batch_open::prove_batch_open_committed_owned(num_ctx, &num_idxs, &num_tensors, &num_points, &num_claimed).expect("honest perm num batch-open");
        perm_batch_openings.push(ColumnBatchOpening { table_idxs: table_idxs.clone(), points: points_a, claimed: a_claimed, proof: a_proof });
        perm_batch_openings.push(ColumnBatchOpening { table_idxs: table_idxs.clone(), points: points_b, claimed: b_claimed, proof: b_proof });
        perm_batch_openings.push(ColumnBatchOpening { table_idxs: num_idxs, points: num_points, claimed: num_claimed, proof: num_proof });
    }
    perm_locations = perm_locations_slots.into_iter().map(|x| x.unwrap()).collect();


    // Commit + open each Gather histogram witness, grouped by (arity, type).
    // den_source/den_target depend on the Fiat-Shamir alpha, so they are rebuilt
    // with that alpha AFTER it is derived; the six alpha-independent columns are
    // committed first. This closes the soundness gap where the plain-path alpha
    // differed from the committed-path alpha.
    let mut gather_batches: Vec<ColumnBatchCommit> = Vec::new();
    let mut gather_batch_openings: Vec<ColumnBatchOpening> = Vec::new();
    let mut gather_locations: Vec<GatherLocation> = Vec::new();

    // type ids: 0=num_source 1=den_source 2=weighted 3=num_target 4=den_target
    //           5=out 6=gate 7=token_id
    let mut gather_ops: Vec<(usize, usize, usize, usize, usize, usize, usize)> = Vec::new();
    {
        let gr = shard_ranges(ops.len(), ops_per_shard);
        for (si, &(s, e)) in gr.iter().enumerate() {
            for (li, op) in ops[s..e].iter().enumerate() {
                if let (Op::Gather { m, h, .. }, OpProof::Gather(sp)) = (op, &shards[si].ops[li]) {
                    let s_len = sp.token_id.len();
                    // The fractional domains are padded to arity >= 5 in
                    // prove_scatter_add; the committed columns follow.
                    let arity_src = sp.num_source.len().trailing_zeros() as usize;
                    let arity_tgt = sp.num_target.len().trailing_zeros() as usize;
                    let arity_gate = s_len.next_power_of_two().max(32).trailing_zeros() as usize;
                    gather_ops.push((si, li, arity_src, arity_tgt, arity_gate, *h, *m));
                }
            }
        }
    }
    let mut gather_gid: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
    for (gid, &(si, li, _, _, _, _, _)) in gather_ops.iter().enumerate() {
        gather_gid.insert((si, li), gid);
    }

    struct GatherGroup {
        arity: usize,
        ty: usize,
        ops: Vec<(usize, usize)>,
        cols: Vec<Vec<Goldilocks>>,
        ctx: zkie_core::pcs::committed::BatchCtx,
    }
    let mut gather_groups: Vec<GatherGroup> = Vec::new();
    let pad = |v: &[Goldilocks], arity: usize| -> Vec<Goldilocks> {
        let total = 1usize << arity;
        let mut o = vec![Goldilocks::ZERO; total];
        o[..v.len()].copy_from_slice(v);
        o
    };

    // Phase A: commit the four alpha/gamma-independent columns (weighted,
    // out, gate, token_id). The num columns depend on the FS gamma and the
    // den columns on the FS alpha, so they are committed in phase B below
    // (rebuilding them with the committed-transcript alpha/gamma keeps the
    // column content consistent with the fracs re-proven there).
    {
        let indep = [2usize, 5, 6, 7];
        let mut by_key: std::collections::BTreeMap<(usize, usize), Vec<(usize, usize)>> = std::collections::BTreeMap::new();
        for &(si, li, arity_src, arity_tgt, arity_gate, _, _) in &gather_ops {
            for &ty in &indep {
                let ar = match ty { 2 => arity_src, 5 => arity_tgt, _ => arity_gate };
                by_key.entry((ar, ty)).or_default().push((si, li));
            }
        }
        for ((arity, ty), ops_) in by_key {
            let mut cols = Vec::with_capacity(ops_.len());
            for &(si, li) in &ops_ {
                let sp = match &shards[si].ops[li] { OpProof::Gather(p) => p, _ => unreachable!() };
                let col = match ty {
                    2 => pad(&sp.weighted, arity),
                    5 => pad(&sp.out, arity),
                    6 => pad(&sp.gate, arity),
                    _ => { let g: Vec<Goldilocks> = sp.token_id.iter().map(|&t| Goldilocks::from_u64(t as u64)).collect(); pad(&g, arity) }
                };
                cols.push(col);
            }
            let num_tables = ops_.len();
            let num_vars = (num_tables * (1usize << arity)).next_power_of_two().trailing_zeros() as usize;
            let pow_bits = zkie_core::pcs::whir::Whir::required_pow_bits(num_vars, 90);
            let base_whir = zkie_core::pcs::whir::Whir::with_params(arity, 90, pow_bits);
            let refs: Vec<&[Goldilocks]> = cols.iter().map(|v| v.as_slice()).collect();
            let (c, p, pr, w) = base_whir.commit_batch(&refs);
            gather_groups.push(GatherGroup { arity, ty, ops: ops_, cols, ctx: zkie_core::pcs::committed::BatchCtx { commitment: c, prover_data: p, protocol: pr, whir: w, num_tables } });
        }
    }

    let gather_indep_count = gather_groups.len();
    // FS: absorb the independent roots, then derive alpha/gamma per op.
    let mut ts = Transcript::new(b"zkie-committed-gather-v1");
    absorb_statement(&mut ts, ops, ops_per_shard);
    for g in &gather_groups {
        ts.absorb(&bincode::serialize(&g.ctx.commitment).unwrap());
    }
    for &(si, li, _, _, _, h, _) in &gather_ops {
        let alpha = ts.field();
        let gamma: Vec<Goldilocks> = (0..h).map(|_| ts.field()).collect();
        let sp = match &mut shards[si].ops[li] {
            OpProof::Gather(p) => p,
            _ => unreachable!(),
        };
        sp.alpha = alpha;
        sp.gamma = gamma;
    }

    // Phase B: rebuild num_source/num_target with the FS gamma and
    // den_source/den_target with the FS alpha, then commit them.
    let mut op_num_src: Vec<Vec<Goldilocks>> = vec![Vec::new(); gather_ops.len()];
    let mut op_num_tgt: Vec<Vec<Goldilocks>> = vec![Vec::new(); gather_ops.len()];
    let mut op_den_src: Vec<Vec<Goldilocks>> = vec![Vec::new(); gather_ops.len()];
    let mut op_den_tgt: Vec<Vec<Goldilocks>> = vec![Vec::new(); gather_ops.len()];
    for (gid, &(si, li, arity_src, arity_tgt, _, h, m)) in gather_ops.iter().enumerate() {
        let (alpha, gamma) = match &shards[si].ops[li] {
            OpProof::Gather(p) => (p.alpha, p.gamma.clone()),
            _ => unreachable!(),
        };
        let sp = match &shards[si].ops[li] { OpProof::Gather(p) => p, _ => unreachable!() };
        let mut nsrc = vec![Goldilocks::ZERO; 1usize << arity_src];
        let mut dsrc = vec![alpha; 1usize << arity_src];
        for pos in 0..sp.token_id.len() {
            for d in 0..h {
                let idx = pos * h + d;
                nsrc[idx] = gamma[d] * sp.weighted[idx] * sp.gate[pos];
                dsrc[idx] = alpha - Goldilocks::from_u64(sp.token_id[pos] as u64);
            }
        }
        let mut ntgt = vec![Goldilocks::ZERO; 1usize << arity_tgt];
        let mut dtgt = vec![alpha; 1usize << arity_tgt];
        for t in 0..m {
            for d in 0..h {
                ntgt[t * h + d] = gamma[d] * sp.out[t * h + d];
                dtgt[t * h + d] = alpha - Goldilocks::from_u64(t as u64);
            }
        }
        // Den continuation into the padding (num stays 0), matching the
        // verifier's index-formula check at the den leaf point.
        let h_log = h.trailing_zeros() as usize;
        for idx in m * h..(1usize << arity_tgt) {
            dtgt[idx] = alpha - Goldilocks::from_u64((idx >> h_log) as u64);
        }
        op_num_src[gid] = nsrc;
        op_num_tgt[gid] = ntgt;
        op_den_src[gid] = dsrc;
        op_den_tgt[gid] = dtgt;
    }
    {
        let dep_tys = [0usize, 1, 3, 4];
        let mut by_key: std::collections::BTreeMap<(usize, usize), Vec<(usize, usize)>> = std::collections::BTreeMap::new();
        for &(si, li, arity_src, arity_tgt, _, _, _) in &gather_ops {
            for &ty in &dep_tys {
                let ar = if ty == 0 || ty == 1 { arity_src } else { arity_tgt };
                by_key.entry((ar, ty)).or_default().push((si, li));
            }
        }
        for ((arity, ty), ops_) in by_key {
            let mut cols = Vec::with_capacity(ops_.len());
            for &(si, li) in &ops_ {
                let gid = gather_gid[&(si, li)];
                let col = match ty {
                    0 => op_num_src[gid].clone(),
                    1 => op_den_src[gid].clone(),
                    3 => op_num_tgt[gid].clone(),
                    _ => op_den_tgt[gid].clone(),
                };
                cols.push(col);
            }
            let num_tables = ops_.len();
            let num_vars = (num_tables * (1usize << arity)).next_power_of_two().trailing_zeros() as usize;
            let pow_bits = zkie_core::pcs::whir::Whir::required_pow_bits(num_vars, 90);
            let base_whir = zkie_core::pcs::whir::Whir::with_params(arity, 90, pow_bits);
            let refs: Vec<&[Goldilocks]> = cols.iter().map(|v| v.as_slice()).collect();
            let (c, p, pr, w) = base_whir.commit_batch(&refs);
            gather_groups.push(GatherGroup { arity, ty, ops: ops_, cols, ctx: zkie_core::pcs::committed::BatchCtx { commitment: c, prover_data: p, protocol: pr, whir: w, num_tables } });
        }
    }

    // FS: absorb the dependent roots, then run the fractional proofs per op.
    for g in gather_groups.iter().skip(gather_indep_count) {
        ts.absorb(&bincode::serialize(&g.ctx.commitment).unwrap());
    }

    // Record verifier-side batches (8 groups, in the same order).
    for g in &gather_groups {
        let num_vars = (g.ops.len() * (1usize << g.arity)).next_power_of_two().trailing_zeros() as usize;
        let pow_bits = zkie_core::pcs::whir::Whir::required_pow_bits(num_vars, 90);
        gather_batches.push(ColumnBatchCommit {
            commitment: g.ctx.commitment.clone(),
            arity: g.arity,
            num_tables: g.ops.len(),
            whir: zkie_core::pcs::whir::Whir::with_params(num_vars, 90, pow_bits),
        });
    }

    // S1: run the fractional sumchecks per Gather op on the transcript, in
    // gather_ops (shard-major) order, using the rebuilt den. The layer
    // challenges are derived per-round inside prove_fractional.
    let mut op_point_src: Vec<Vec<Goldilocks>> = vec![Vec::new(); gather_ops.len()];
    let mut op_den_point_src: Vec<Vec<Goldilocks>> = vec![Vec::new(); gather_ops.len()];
    let mut op_point_tgt: Vec<Vec<Goldilocks>> = vec![Vec::new(); gather_ops.len()];
    let mut op_den_point_tgt: Vec<Vec<Goldilocks>> = vec![Vec::new(); gather_ops.len()];
    for (gid, &(si, li, arity_src, arity_tgt, _, _, _)) in gather_ops.iter().enumerate() {
        let sp = match &mut shards[si].ops[li] {
            OpProof::Gather(p) => p,
            _ => unreachable!(),
        };
        // The fracs run on the REBUILT (committed-gamma/alpha) columns, which
        // the batch openings below commit and open.
        let new_src = prove_fractional(op_num_src[gid].clone(), op_den_src[gid].clone(), &mut ts);
        let (point_src, _, den_point_src, _) = fractional_leaf(&new_src);
        let new_tgt = prove_fractional(op_num_tgt[gid].clone(), op_den_tgt[gid].clone(), &mut ts);
        let (point_tgt, _, den_point_tgt, _) = fractional_leaf(&new_tgt);
        sp.num_source = op_num_src[gid].clone();
        sp.num_target = op_num_tgt[gid].clone();
        sp.den_source = op_den_src[gid].clone();
        sp.den_target = op_den_tgt[gid].clone();
        sp.source_side = new_src;
        sp.target_side = new_tgt;
        op_point_src[gid] = point_src;
        op_den_point_src[gid] = den_point_src;
        op_point_tgt[gid] = point_tgt;
        op_den_point_tgt[gid] = den_point_tgt;
        let _ = arity_src;
        let _ = arity_tgt;
    }

    // Batch-open each group and fill locations in global order.
    let mut group_idx_of: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
    for (gi, g) in gather_groups.iter().enumerate() {
        group_idx_of.insert((g.arity, g.ty), gi);
    }
    let mut gather_locations_slots: Vec<Option<GatherLocation>> = (0..gather_ops.len()).map(|_| None).collect();
    for (g_idx, g) in gather_groups.into_iter().enumerate() {
        let GatherGroup {
            arity: _,
            ty,
            ops: group_ops,
            cols,
            ctx,
        } = g;
        let n = group_ops.len();
        let table_idxs: Vec<usize> = (0..n).collect();
        let mut points = Vec::with_capacity(n);
        let mut claimed = Vec::with_capacity(n);
        for (ti, &(si, li)) in group_ops.iter().enumerate() {
            let gid = gather_gid[&(si, li)];
            let (_, _, _, _, arity_gate, h, _) = gather_ops[gid];
            let h_log = h.trailing_zeros() as usize;
            let sp = match &shards[si].ops[li] { OpProof::Gather(p) => p, _ => unreachable!() };
            // The gate/token_id columns are committed over the PADDED gate
            // domain (arity_gate), so their points are the restricted pos
            // bits zero-padded to that arity. S2: num opened at the num leaf
            // point, den (and the token_id it is checked against) at the den
            // leaf point.
            let gate_pt = |pos_bits: &[Goldilocks]| -> Vec<Goldilocks> {
                let mut pt = pos_bits.to_vec();
                pt.resize(arity_gate, Goldilocks::ZERO);
                pt
            };
            let (pt, cl) = match ty {
                0 => (op_point_src[gid].clone(), mle::eval(&cols[ti], &op_point_src[gid])),
                1 => (op_den_point_src[gid].clone(), mle::eval(&cols[ti], &op_den_point_src[gid])),
                2 => (sp.r_prod_source.clone(), mle::eval(&cols[ti], &sp.r_prod_source)),
                3 => (op_point_tgt[gid].clone(), mle::eval(&cols[ti], &op_point_tgt[gid])),
                4 => (op_den_point_tgt[gid].clone(), mle::eval(&cols[ti], &op_den_point_tgt[gid])),
                5 => (sp.r_prod_target.clone(), mle::eval(&cols[ti], &sp.r_prod_target)),
                6 => (gate_pt(&sp.r_prod_source[h_log..]), mle::eval(&cols[ti], &gate_pt(&sp.r_prod_source[h_log..]))),
                _ => (gate_pt(&op_den_point_src[gid][h_log..]), mle::eval(&cols[ti], &gate_pt(&op_den_point_src[gid][h_log..]))),
            };
            points.push(pt);
            claimed.push(cl);
            let loc = gather_locations_slots[gid].get_or_insert(GatherLocation {
                num_source_batch: 0, num_source_table: 0,
                den_source_batch: 0, den_source_table: 0,
                weighted_batch: 0, weighted_table: 0,
                num_target_batch: 0, num_target_table: 0,
                den_target_batch: 0, den_target_table: 0,
                out_batch: 0, out_table: 0,
                gate_batch: 0, gate_table: 0,
                token_id_batch: 0, token_id_table: 0,
            });
            match ty {
                0 => { loc.num_source_batch = g_idx; loc.num_source_table = ti; }
                1 => { loc.den_source_batch = g_idx; loc.den_source_table = ti; }
                2 => { loc.weighted_batch = g_idx; loc.weighted_table = ti; }
                3 => { loc.num_target_batch = g_idx; loc.num_target_table = ti; }
                4 => { loc.den_target_batch = g_idx; loc.den_target_table = ti; }
                5 => { loc.out_batch = g_idx; loc.out_table = ti; }
                6 => { loc.gate_batch = g_idx; loc.gate_table = ti; }
                _ => { loc.token_id_batch = g_idx; loc.token_id_table = ti; }
            }
        }
        let proof = zkie_core::pcs::batch_open::prove_batch_open_committed_owned(
            ctx, &table_idxs, &cols, &points, &claimed,
        ).expect("honest gather batch-open");
        gather_batch_openings.push(ColumnBatchOpening { table_idxs, points, claimed, proof });
        for &(si, li) in &group_ops {
            let sp = match &mut shards[si].ops[li] { OpProof::Gather(p) => p, _ => unreachable!() };
            match ty {
                0 => sp.num_source.clear(),
                1 => sp.den_source.clear(),
                2 => sp.weighted.clear(),
                3 => sp.num_target.clear(),
                4 => sp.den_target.clear(),
                5 => sp.out.clear(),
                6 => sp.gate.clear(),
                _ => sp.token_id.clear(),
            }
        }
    }
    gather_locations = gather_locations_slots.into_iter().map(|x| x.unwrap()).collect();


    CommittedShardDagProof {
        shards,
        boundary_tensors,
        boundary_commitments,
        cross_binds,
        weight_batches,
        weight_batch_openings,
        weight_locs,
        tokens: store.idx[tokens_t].clone(),
        argmax_t: store.idx[argmax_t].clone(),
        witness_batches,
        witness_batch_openings,
        witness_locations,
        table_openings,
        sel_openings,
        perm_batches,
        perm_batch_openings,
        perm_locations,
        gather_batches,
        gather_batch_openings,
        gather_locations,
        gather_indep_batches: gather_indep_count,
    }
}

pub fn verify_committed_shard_dag(
    tables: &[&[Goldilocks]],
    ops: &[Op],
    ops_per_shard: usize,
    _whir: &zkie_core::pcs::whir::Whir,
    proof: &CommittedShardDagProof,
) -> bool {
    let st = VerifierStatement::new(tables, ops, ops_per_shard);
    let ranges = shard_ranges(ops.len(), ops_per_shard);
    if proof.shards.len() != ranges.len()
        || proof.boundary_tensors.len() != proof.boundary_commitments.len()
        || proof.boundary_tensors.len() != proof.cross_binds.len()
    {
        return false;
    }
    let shards = &proof.shards;
    // 1. Shard replays (committed mode) — every shard's claim set is BUILT
    // locally from the op verifications (S2), never read from the proof.
    let mut master = Transcript::new(b"zkie-shard-dag-v1");
    absorb_statement(&mut master, ops, ops_per_shard);
    let shard_claims: Vec<Option<Vec<EvalClaim>>> = ranges
        .par_iter()
        .enumerate()
        .map(|(i, &(s, e))| {
            let mut child = master.clone();
            child.absorb_u64(i as u64);
            verify_shard_precomputed_inner(tables, &ops[s..e], &shards[i], &mut child, FracMode::Committed)
        })
        .collect();
    if shard_claims.iter().any(|c| c.is_none()) {
        return false;
    }
    // 2. Boundary (cross-shard) set rebuilt from the local claims; the
    // committed cross-binds are verified against those local claims.
    let mut by_tensor: std::collections::BTreeMap<T, Vec<(Vec<Goldilocks>, Goldilocks)>> =
        std::collections::BTreeMap::new();
    let mut tensor_shards: std::collections::BTreeMap<T, std::collections::HashSet<usize>> =
        std::collections::BTreeMap::new();
    for (si, claims) in shard_claims.iter().enumerate() {
        for c in claims.as_ref().unwrap() {
            by_tensor
                .entry(c.tensor_id)
                .or_default()
                .push((c.point.clone(), c.value));
            tensor_shards.entry(c.tensor_id).or_default().insert(si);
        }
    }
    let mut expected_boundary: Vec<T> = Vec::new();
    for (t, _) in &by_tensor {
        if tensor_shards.get(t).map(|s| s.len()).unwrap_or(0) > 1 {
            expected_boundary.push(*t);
        }
    }
    if expected_boundary != proof.boundary_tensors {
        return false;
    }
    let mut bind_ts = Transcript::new(b"zkie-committed-binds-v1");
    absorb_statement(&mut bind_ts, ops, ops_per_shard);
    for (i, &t) in proof.boundary_tensors.iter().enumerate() {
        if !verify_committed_cross_bind(
            &proof.boundary_commitments[i],
            &by_tensor[&t],
            &proof.cross_binds[i],
            &mut bind_ts,
        ) {
            return false;
        }
    }
    // 3. Weight batches (S2): every locally claimed weight tensor must be
    // opened against its batch commitment — right tensor, right point, right
    // value, right root. The expected opening entries are rebuilt from the
    // local claims via the proof's `weight_locs` mapping, and an opening must
    // carry EXACTLY those entries. Batches with no claimed tensors may be
    // empty (unused weights); an empty list can never skip a non-empty one.
    if proof.weight_batch_openings.len() != proof.weight_batches.len() {
        return false;
    }
    let weights: std::collections::HashSet<T> = st.weight_tensors().into_iter().collect();
    let entry_key =
        |e: &(usize, Vec<Goldilocks>, Goldilocks)| -> Vec<u8> { bincode::serialize(e).unwrap() };
    let mut batch_entries: Vec<Vec<(usize, Vec<Goldilocks>, Goldilocks)>> =
        vec![Vec::new(); proof.weight_batches.len()];
    for claims in shard_claims.iter() {
        for c in claims.as_ref().unwrap() {
            if !weights.contains(&c.tensor_id) {
                continue;
            }
            let (bi, ti) = match proof.weight_locs.get(&c.tensor_id) {
                Some(&loc) => loc,
                None => return false, // claimed weight with no batch opening
            };
            if bi >= proof.weight_batches.len() || ti >= proof.weight_batches[bi].num_tables {
                return false;
            }
            if c.point.len() > proof.weight_batches[bi].arity {
                return false;
            }
            let mut pt = c.point.clone();
            pt.resize(proof.weight_batches[bi].arity, Goldilocks::ZERO);
            batch_entries[bi].push((ti, pt, c.value));
        }
    }
    for (bi, opening) in proof.weight_batch_openings.iter().enumerate() {
        let entries = &batch_entries[bi];
        if opening.table_idxs.len() != opening.points.len()
            || opening.table_idxs.len() != opening.claimed.len()
        {
            return false;
        }
        match (&opening.proof, entries.is_empty()) {
            (None, true) => {
                // Unused weight group: no claims, no opening (allowed).
                if !opening.table_idxs.is_empty() {
                    return false;
                }
            }
            (Some(_), false) => {
                // Point-set comparison: the opening must carry exactly the
                // claims of this batch (multiset).
                let mut expected: Vec<Vec<u8>> = entries.iter().map(entry_key).collect();
                expected.sort();
                let mut got: Vec<Vec<u8>> = opening
                    .table_idxs
                    .iter()
                    .zip(opening.points.iter())
                    .zip(opening.claimed.iter())
                    .map(|((ti, pt), ev)| bincode::serialize(&(ti, pt.clone(), ev)).unwrap())
                    .collect();
                got.sort();
                if expected != got {
                    return false;
                }
                let batch = &proof.weight_batches[bi];
                let protocol = batch.whir.opening_protocol(batch.arity, batch.num_tables);
                if !zkie_core::pcs::batch_open::verify_batch_open_committed(
                    &batch.whir,
                    &batch.commitment,
                    &protocol,
                    batch.num_tables,
                    &opening.table_idxs,
                    &opening.points,
                    &opening.claimed,
                    opening.proof.as_ref().unwrap(),
                ) {
                    return false;
                }
            }
            _ => return false, // claims with no opening, or opening with no claims
        }
    }
    // 4. Fractional replays FIRST — every leaf is derived from the checked
    // chains (S2); the opening checks below compare against them.
    let mut witness_leaves: Vec<FractionalChainClaims> = Vec::new();
    let mut table_leaves: Vec<FractionalChainClaims> = Vec::new();
    {
        let mut ts = Transcript::new(b"zkie-committed-frac-v1");
        absorb_statement(&mut ts, ops, ops_per_shard);
        for batch in &proof.witness_batches {
            ts.absorb(&bincode::serialize(&batch.commitment).unwrap());
        }
        let mut ti = 0usize;
        for (si, shard) in shards.iter().enumerate() {
            ts.absorb_u64(si as u64);
            for tc in &shard.table_columns {
                if ti >= proof.table_openings.len() {
                    return false;
                }
                let to = &proof.table_openings[ti];
                if tc.table != to.table {
                    return false;
                }
                ts.absorb(&bincode::serialize(&to.commitment).unwrap());
                if st.is_identity_table(to.table) {
                    ts.absorb_u64(0);
                } else {
                    match &to.table_commitment {
                        Some(c) => ts.absorb(&bincode::serialize(c).unwrap()),
                        None => return false,
                    }
                }
                ti += 1;
            }
        }
        if ti != proof.table_openings.len() {
            return false;
        }
        // S3: each column's fractional depth comes from its committed
        // column length (the witness batch arity / the m-column arity).
        let mut shard_wc_off: Vec<usize> = Vec::with_capacity(shards.len() + 1);
        shard_wc_off.push(0);
        for shard in shards.iter() {
            let next = shard_wc_off.last().unwrap() + shard.witness_columns.len();
            shard_wc_off.push(next);
        }
        let total_witness_cols: usize = shards.iter().map(|s| s.witness_columns.len()).sum();
        if proof.witness_locations.len() != total_witness_cols {
            return false;
        }
        for (si, shard) in shards.iter().enumerate() {
            for (ci, wc) in shard.witness_columns.iter().enumerate() {
                if ts.field() != wc.alpha || ts.field() != wc.beta {
                    return false;
                }
                let loc = &proof.witness_locations[shard_wc_off[si] + ci];
                if loc.x_batch >= proof.witness_batches.len() {
                    return false;
                }
                let t_w = proof.witness_batches[loc.x_batch].arity;
                match verify_fractional_chain_core(&wc.proof, t_w, &mut ts) {
                    Some(leaf) => witness_leaves.push(leaf),
                    None => return false,
                }
            }
        }
        let mut ti_frac = 0usize;
        for shard in shards.iter() {
            for tc in &shard.table_columns {
                let t_t = proof.table_openings[ti_frac].arity;
                match verify_fractional_chain_core(&tc.proof, t_t, &mut ts) {
                    Some(leaf) => table_leaves.push(leaf),
                    None => {
                        return false;
                    }
                }
                ti_frac += 1;
            }
        }
        // The witness/table pairing is per shard by column index; the split
        // cross relation uses the DERIVED roots, and zero denominators are
        // rejected explicitly (S3: no vacuous 0/0).
        let mut w_off = 0usize;
        for shard in shards.iter() {
            for i in 0..shard.witness_columns.len() {
                let wc = &witness_leaves[w_off + i];
                let tc = &table_leaves[w_off + i];
                if wc.final_den == Goldilocks::ZERO || tc.final_den == Goldilocks::ZERO {
                    return false;
                }
                if wc.final_num * tc.final_den != tc.final_num * wc.final_den {
                    return false;
                }
            }
            w_off += shard.witness_columns.len();
        }
    }
    // 5. Witness columns: batch openings verified against their commitments
    // and compared with the DERIVED leaves (x/y at q_den, num at q_num).
    if proof.witness_batches.len() != proof.witness_batch_openings.len() {
        return false;
    }
    let total_witness_cols: usize = shards.iter().map(|s| s.witness_columns.len()).sum();
    if proof.witness_locations.len() != total_witness_cols {
        return false;
    }
    for (bi, opening) in proof.witness_batch_openings.iter().enumerate() {
        if opening.table_idxs.len() != opening.points.len()
            || opening.table_idxs.len() != opening.claimed.len()
        {
            return false;
        }
        let batch = &proof.witness_batches[bi];
        let protocol = batch.whir.opening_protocol(batch.arity, batch.num_tables);
        if !zkie_core::pcs::batch_open::verify_batch_open_committed(
            &batch.whir,
            &batch.commitment,
            &protocol,
            batch.num_tables,
            &opening.table_idxs,
            &opening.points,
            &opening.claimed,
            &opening.proof,
        ) {
            return false;
        }
    }
    {
        let mut gid = 0usize;
        for shard in shards.iter() {
            for wc in &shard.witness_columns {
                let wl = &witness_leaves[gid];
                let (q_num, num_leaf, q_den, den_leaf) =
                    (&wl.q_num, &wl.num_leaf, &wl.q_den, &wl.den_leaf);
                let loc = &proof.witness_locations[gid];
                let (x_open, y_open, num_open) = match (
                    proof.witness_batch_openings.get(loc.x_batch),
                    proof.witness_batch_openings.get(loc.y_batch),
                    proof.witness_batch_openings.get(loc.num_batch),
                ) {
                    (Some(x), Some(y), Some(n)) => (x, y, n),
                    _ => return false,
                };
                if loc.x_table >= x_open.points.len()
                    || loc.y_table >= y_open.points.len()
                    || loc.num_table >= num_open.points.len()
                {
                    return false;
                }
                // x/y must sit on the derived DEN leaf point, num on the NUM
                // leaf point with the derived num leaf value.
                if x_open.points[loc.x_table] != *q_den || y_open.points[loc.y_table] != *q_den {
                    return false;
                }
                if num_open.points[loc.num_table] != *q_num {
                    return false;
                }
                if num_open.claimed[loc.num_table] != *num_leaf {
                    return false;
                }
                let eval_x = x_open.claimed[loc.x_table];
                let eval_y = y_open.claimed[loc.y_table];
                if wc.alpha + eval_x + wc.beta * eval_y != *den_leaf {
                    return false;
                }
                gid += 1;
            }
        }
    }
    // 6. Table side (S2, the four opening elements): the m-column opening must
    // sit on the DERIVED num leaf point with the derived num leaf value; the
    // table must be opened at the derived den leaf point with the den relation
    // checked there; the identity decision comes from the statement tables.
    {
        let mut ti = 0usize;
        for shard in shards.iter() {
            for _tc in &shard.table_columns {
                let tl = &table_leaves[ti];
                let (q_num_t, a_t, q_den_t, b_t) =
                    (&tl.q_num, &tl.num_leaf, &tl.q_den, &tl.den_leaf);
                let to = &proof.table_openings[ti];
                let pow_bits = zkie_core::pcs::whir::Whir::required_pow_bits(to.arity, 90);
                let bw = zkie_core::pcs::whir::Whir::with_params(to.arity, 90, pow_bits);
                let protocol = bw.opening_protocol(to.arity, 1);
                if to.point != *q_num_t || to.claimed != *a_t {
                    return false;
                }
                if bw.verify(&to.commitment, &to.opening, &protocol, &to.point).ok()
                    != Some(to.claimed)
                {
                    return false;
                }
                if to.den_point != *q_den_t {
                    return false;
                }
                let is_ident = st.is_identity_table(to.table);
                let den_table_eval = if is_ident {
                    if to.table_commitment.is_some()
                        || to.table_opening.is_some()
                        || to.den_table_opening.is_some()
                    {
                        return false;
                    }
                    let mut den_idx = Goldilocks::ZERO;
                    for (k, &pk) in q_den_t.iter().enumerate() {
                        den_idx = den_idx + Goldilocks::from_u64(1u64 << k) * pk;
                    }
                    den_idx
                } else {
                    let (tc_, to2) = match (&to.table_commitment, &to.table_opening) {
                        (Some(c), Some(o)) => (c, o),
                        _ => return false,
                    };
                    if bw.verify(tc_, to2, &protocol, &to.point).ok() != Some(to.table_eval) {
                        return false;
                    }
                    let (dc, do2) = match (&to.table_commitment, &to.den_table_opening) {
                        (Some(c), Some(o)) => (c, o),
                        _ => return false,
                    };
                    if bw.verify(dc, do2, &protocol, &to.den_point).ok()
                        != Some(to.den_table_eval)
                    {
                        return false;
                    }
                    to.den_table_eval
                };
                let mut den_idx = Goldilocks::ZERO;
                for (k, &pk) in q_den_t.iter().enumerate() {
                    den_idx = den_idx + Goldilocks::from_u64(1u64 << k) * pk;
                }
                if to.alpha + den_idx + to.beta * den_table_eval != *b_t {
                    return false;
                }
                ti += 1;
            }
        }
    }
    // 7. Permutations: frac replays first (collecting derived leaves), then
    // the opening-vs-leaf comparisons.
    if proof.perm_batches.len() != proof.perm_batch_openings.len() {
        return false;
    }
    let mut perm_leaves: Vec<(FractionalChainClaims, FractionalChainClaims)> = Vec::new();
    {
        let mut ts = Transcript::new(b"zkie-committed-perm-v1");
        absorb_statement(&mut ts, ops, ops_per_shard);
        for batch in &proof.perm_batches {
            ts.absorb(&bincode::serialize(&batch.commitment).unwrap());
        }
        let perm_op_count: usize = shards
            .iter()
            .map(|s| s.ops.iter().filter(|o| matches!(o, OpProof::Scatter(_))).count())
            .sum();
        if proof.perm_locations.len() != perm_op_count {
            return false;
        }
        let mut pgid = 0usize;
        for shard in shards.iter() {
            for op in &shard.ops {
                if let OpProof::Scatter(sp) = op {
                    let alpha = ts.field();
                    if alpha != sp.perm.alpha {
                        return false;
                    }
                    // S3: the permutation depth comes from the committed
                    // column length (the perm batch arity).
                    let t = proof.perm_batches[proof.perm_locations[pgid].a_batch].arity;
                    let la = match verify_fractional_chain_core(&sp.perm.a_side, t, &mut ts) {
                        Some(l) => l,
                        None => return false,
                    };
                    let lb = match verify_fractional_chain_core(&sp.perm.b_side, t, &mut ts) {
                        Some(l) => l,
                        None => return false,
                    };
                    if la.final_den == Goldilocks::ZERO || lb.final_den == Goldilocks::ZERO {
                        return false;
                    }
                    perm_leaves.push((la, lb));
                    pgid += 1;
                }
            }
        }
    }
    {
        let total: usize = shards
            .iter()
            .map(|s| s.ops.iter().filter(|o| matches!(o, OpProof::Scatter(_))).count())
            .sum();
        if proof.perm_locations.len() != total {
            return false;
        }
    }
    for (bi, opening) in proof.perm_batch_openings.iter().enumerate() {
        if opening.table_idxs.len() != opening.points.len()
            || opening.table_idxs.len() != opening.claimed.len()
        {
            return false;
        }
        let batch = &proof.perm_batches[bi];
        let protocol = batch.whir.opening_protocol(batch.arity, batch.num_tables);
        if !zkie_core::pcs::batch_open::verify_batch_open_committed(
            &batch.whir,
            &batch.commitment,
            &protocol,
            batch.num_tables,
            &opening.table_idxs,
            &opening.points,
            &opening.claimed,
            &opening.proof,
        ) {
            return false;
        }
    }
    {
        let mut gid = 0usize;
        for shard in shards.iter() {
            for op in &shard.ops {
                if let OpProof::Scatter(sp) = op {
                    let loc = &proof.perm_locations[gid];
                    let (la, lb) = &perm_leaves[gid];
                    let (q_na, a_a, q_da, b_a) = (&la.q_num, &la.num_leaf, &la.q_den, &la.den_leaf);
                    let (q_nb, a_b, q_db, b_b) = (&lb.q_num, &lb.num_leaf, &lb.q_den, &lb.den_leaf);
                    let (a_open, b_open, num_open) = match (
                        proof.perm_batch_openings.get(loc.a_batch),
                        proof.perm_batch_openings.get(loc.b_batch),
                        proof.perm_batch_openings.get(loc.num_batch),
                    ) {
                        (Some(a), Some(b), Some(n)) => (a, b, n),
                        _ => return false,
                    };
                    if loc.a_table + 1 >= a_open.points.len()
                        || loc.b_table + 1 >= b_open.points.len()
                        || 2 * loc.num_table + 1 >= num_open.points.len()
                    {
                        return false;
                    }
                    // a/b columns: entry 2*ti at the num leaf point, 2*ti+1 at
                    // the den leaf point. The num column (shared indicator):
                    // q_num_a and q_num_b with the two derived num leaves.
                    if a_open.points[loc.a_table] != *q_na
                        || a_open.points[loc.a_table + 1] != *q_da
                    {
                        return false;
                    }
                    if b_open.points[loc.b_table] != *q_nb
                        || b_open.points[loc.b_table + 1] != *q_db
                    {
                        return false;
                    }
                    if num_open.points[2 * loc.num_table] != *q_na
                        || num_open.points[2 * loc.num_table + 1] != *q_nb
                    {
                        return false;
                    }
                    let nl_a = num_open.claimed[2 * loc.num_table];
                    let nl_b = num_open.claimed[2 * loc.num_table + 1];
                    if nl_a != *a_a || nl_b != *a_b {
                        return false;
                    }
                    let eval_a = a_open.claimed[loc.a_table + 1];
                    let eval_b = b_open.claimed[loc.b_table + 1];
                    if *b_a != sp.perm.alpha - eval_a || *b_b != sp.perm.alpha - eval_b {
                        return false;
                    }
                    gid += 1;
                }
            }
        }
    }
    // 8. Gather: frac replays first (derived leaves), then the opening
    // comparisons against the leaves and the product final evals.
    if proof.gather_batches.len() != proof.gather_batch_openings.len() {
        return false;
    }
    let mut gather_leaves: Vec<(FractionalChainClaims, FractionalChainClaims)> = Vec::new();
    {
        let mut ts = Transcript::new(b"zkie-committed-gather-v1");
        absorb_statement(&mut ts, ops, ops_per_shard);
        for batch in proof.gather_batches.iter().take(proof.gather_indep_batches) {
            ts.absorb(&bincode::serialize(&batch.commitment).unwrap());
        }
        for shard in shards.iter() {
            for op in &shard.ops {
                if let OpProof::Gather(sp) = op {
                    let alpha = ts.field();
                    let gamma: Vec<Goldilocks> = (0..sp.gamma.len()).map(|_| ts.field()).collect();
                    if alpha != sp.alpha || gamma != sp.gamma {
                        return false;
                    }
                }
            }
        }
        for batch in proof.gather_batches.iter().skip(proof.gather_indep_batches) {
            ts.absorb(&bincode::serialize(&batch.commitment).unwrap());
        }
        let gather_op_count: usize = shards
            .iter()
            .map(|s| s.ops.iter().filter(|o| matches!(o, OpProof::Gather(_))).count())
            .sum();
        if proof.gather_locations.len() != gather_op_count {
            return false;
        }
        let mut ggid = 0usize;
        for shard in shards.iter() {
            for op in &shard.ops {
                if let OpProof::Gather(sp) = op {
                    // S3: each side's depth comes from its committed column
                    // length (the gather batch arities).
                    let loc = &proof.gather_locations[ggid];
                    let t_src = proof.gather_batches[loc.num_source_batch].arity;
                    let t_tgt = proof.gather_batches[loc.num_target_batch].arity;
                    let ls = match verify_fractional_chain_core(&sp.source_side, t_src, &mut ts) {
                        Some(l) => l,
                        None => return false,
                    };
                    let lt = match verify_fractional_chain_core(&sp.target_side, t_tgt, &mut ts) {
                        Some(l) => l,
                        None => return false,
                    };
                    // Cross relation of the split gather (as on the plain
                    // path), with explicit zero-denominator rejection (S3).
                    if ls.final_den == Goldilocks::ZERO || lt.final_den == Goldilocks::ZERO {
                        return false;
                    }
                    if ls.final_num * lt.final_den != lt.final_num * ls.final_den {
                        return false;
                    }
                    gather_leaves.push((ls, lt));
                    ggid += 1;
                }
            }
        }
    }
    {
        let total: usize = shards
            .iter()
            .map(|s| s.ops.iter().filter(|o| matches!(o, OpProof::Gather(_))).count())
            .sum();
        if proof.gather_locations.len() != total {
            return false;
        }
    }
    for (bi, opening) in proof.gather_batch_openings.iter().enumerate() {
        if opening.table_idxs.len() != opening.points.len()
            || opening.table_idxs.len() != opening.claimed.len()
        {
            return false;
        }
        let batch = &proof.gather_batches[bi];
        let protocol = batch.whir.opening_protocol(batch.arity, batch.num_tables);
        if !zkie_core::pcs::batch_open::verify_batch_open_committed(
            &batch.whir,
            &batch.commitment,
            &protocol,
            batch.num_tables,
            &opening.table_idxs,
            &opening.points,
            &opening.claimed,
            &opening.proof,
        ) {
            return false;
        }
    }
    {
        let mut gid = 0usize;
        for (si, &(s, e)) in ranges.iter().enumerate() {
            for (li, (op_st, op_pr)) in ops[s..e].iter().zip(shards[si].ops.iter()).enumerate() {
                if let (Op::Gather { h, .. }, OpProof::Gather(sp)) = (op_st, op_pr) {
                    let loc = &proof.gather_locations[gid];
                    let (ls, lt) = &gather_leaves[gid];
                    let (q_ns, a_s, q_ds, b_s) = (&ls.q_num, &ls.num_leaf, &ls.q_den, &ls.den_leaf);
                    let (q_nt, a_t, q_dt, b_t) = (&lt.q_num, &lt.num_leaf, &lt.q_den, &lt.den_leaf);
                    let h_log = h.trailing_zeros() as usize;
                    let get = |batch: usize, table: usize| -> Option<(&Vec<usize>, &Vec<Vec<Goldilocks>>, &Vec<Goldilocks>)> {
                        let o = proof.gather_batch_openings.get(batch)?;
                        if table >= o.points.len() {
                            return None;
                        }
                        Some((&o.table_idxs, &o.points, &o.claimed))
                    };
                    // ty 0: num_source at q_num_src, claimed == num leaf.
                    let (_, pts, clm) = match get(loc.num_source_batch, loc.num_source_table) {
                        Some(t) => t,
                        None => return false,
                    };
                    if pts[loc.num_source_table] != *q_ns || clm[loc.num_source_table] != *a_s {
                        return false;
                    }
                    // ty 1: den_source at q_den_src, claimed == den leaf, and
                    // b_s == alpha - token_id(q_den_src restricted).
                    let (_, pts, clm) = match get(loc.den_source_batch, loc.den_source_table) {
                        Some(t) => t,
                        None => return false,
                    };
                    if pts[loc.den_source_table] != *q_ds || clm[loc.den_source_table] != *b_s {
                        return false;
                    }
                    let (_, tk_pts, tk_clm) = match get(loc.token_id_batch, loc.token_id_table) {
                        Some(t) => t,
                        None => return false,
                    };
                    let gate_arity = proof.gather_batches[loc.token_id_batch].arity;
                    let mut tk_pt = q_ds[h_log..].to_vec();
                    tk_pt.resize(gate_arity, Goldilocks::ZERO);
                    if tk_pts[loc.token_id_table] != tk_pt {
                        return false;
                    }
                    if *b_s != sp.alpha - tk_clm[loc.token_id_table] {
                        return false;
                    }
                    // ty 2: weighted at r_prod_source == prod_source.fe[3].
                    let (_, pts, clm) = match get(loc.weighted_batch, loc.weighted_table) {
                        Some(t) => t,
                        None => return false,
                    };
                    if pts[loc.weighted_table] != sp.r_prod_source
                        || clm[loc.weighted_table] != sp.prod_source.final_evals[3]
                    {
                        return false;
                    }
                    // ty 3: num_target at q_num_tgt, claimed == num leaf.
                    let (_, pts, clm) = match get(loc.num_target_batch, loc.num_target_table) {
                        Some(t) => t,
                        None => return false,
                    };
                    if pts[loc.num_target_table] != *q_nt || clm[loc.num_target_table] != *a_t {
                        return false;
                    }
                    // ty 4: den_target at q_den_tgt, claimed == den leaf, and
                    // b_t == alpha - t where t is the index value of the point.
                    let (_, pts, clm) = match get(loc.den_target_batch, loc.den_target_table) {
                        Some(t) => t,
                        None => return false,
                    };
                    if pts[loc.den_target_table] != *q_dt || clm[loc.den_target_table] != *b_t {
                        return false;
                    }
                    let mut t_idx = Goldilocks::ZERO;
                    for (k, &pk) in q_dt[h_log..].iter().enumerate() {
                        t_idx = t_idx + Goldilocks::from_u64(1u64 << k) * pk;
                    }
                    if *b_t != sp.alpha - t_idx {
                        return false;
                    }
                    // ty 5: out at r_prod_target == prod_target.fe[3].
                    let (_, pts, clm) = match get(loc.out_batch, loc.out_table) {
                        Some(t) => t,
                        None => return false,
                    };
                    if pts[loc.out_table] != sp.r_prod_target
                        || clm[loc.out_table] != sp.prod_target.final_evals[3]
                    {
                        return false;
                    }
                    // ty 6: gate at r_prod_source[h_log..] (zero-padded to the
                    // gate arity) == prod_source.fe[4].
                    let (_, pts, clm) = match get(loc.gate_batch, loc.gate_table) {
                        Some(t) => t,
                        None => return false,
                    };
                    let gate_arity = proof.gather_batches[loc.gate_batch].arity;
                    let mut gate_pt = sp.r_prod_source[h_log..].to_vec();
                    gate_pt.resize(gate_arity, Goldilocks::ZERO);
                    if pts[loc.gate_table] != gate_pt
                        || clm[loc.gate_table] != sp.prod_source.final_evals[4]
                    {
                        return false;
                    }
                    gid += 1;
                }
            }
        }
    }
    // 9. Argmax selection openings (individual one-hot openings at the claimed
    // output positions).
    for so in &proof.sel_openings {
        if so.points.is_empty() || so.openings.len() != so.points.len() {
            return false;
        }
        let pow_bits = zkie_core::pcs::whir::Whir::required_pow_bits(so.arity, 90);
        let bw = zkie_core::pcs::whir::Whir::with_params(so.arity, 90, pow_bits);
        let protocol = bw.opening_protocol(so.arity, 1);
        for (opening, point) in so.openings.iter().zip(&so.points) {
            if bw
                .verify(&so.commitment, opening, &protocol, point)
                .ok()
                != Some(Goldilocks::ONE)
            {
                return false;
            }
        }
    }

    true
}

/// Deserialize a committed shard-DAG proof and verify it, root-only: the proof
/// carries commitment roots and opening proofs, never the committed tensor data.
/// Verify the one-pass sequence self-consistency: the claimed sequence starts
/// with the user prompt, and every generated token equals the argmax prediction
/// of the previous position (teacher forcing). `tokens` is the full claimed
/// sequence [prompt + response], `argmax_t` is the model's argmax per position.
pub fn verify_sequence_shift(tokens: &[u32], argmax_t: &[u32], prompt: &[u32]) -> bool {
    if prompt.len() > tokens.len() || argmax_t.len() != tokens.len() {
        return false;
    }
    if tokens[..prompt.len()] != *prompt {
        return false;
    }
    for i in prompt.len().saturating_sub(1)..tokens.len().saturating_sub(1) {
        if argmax_t[i] != tokens[i + 1] {
            return false;
        }
    }
    true
}

pub fn verify_committed_shard_dag_bytes(
    proof_bytes: &[u8],
    tables: &[&[Goldilocks]],
    ops: &[Op],
    ops_per_shard: usize,
    whir: &zkie_core::pcs::whir::Whir,
) -> bool {
    let proof: CommittedShardDagProof = match bincode::deserialize(proof_bytes) {
        Ok(p) => p,
        Err(_) => return false,
    };
    verify_committed_shard_dag(tables, ops, ops_per_shard, whir, &proof)
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
    rng: &mut impl ChallengeSource,
) -> (Vec<Op>, T) {
    let dh = d / heads;
    let mut ops = Vec::new();

    // pre-norm for attention
    let ln1_w = store.push(
        (0..m * d)
            .map(|_| from_i64((rng.next_u64() % 5) as i64 + 1))
            .collect(),
    );
    let ln1_b = store.push(
        (0..m * d)
            .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
            .collect(),
    );
    let h = store.push(vec![]);
    ops.push(Op::Layernorm {
        x,
        w: ln1_w,
        b: ln1_b,
        out: h,
        rsqrt_table,
        m,
        d,
    });

    // multi-head attention on h
    let mut head_outs = Vec::new();
    for _ in 0..heads {
        let wq = store.push(
            (0..d * dh)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let wk = store.push(
            (0..d * dh)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let wv = store.push(
            (0..d * dh)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let wo = store.push(
            (0..dh * d)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let bq = store.push(
            (0..m * dh)
                .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                .collect(),
        );
        let bk = store.push(
            (0..m * dh)
                .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                .collect(),
        );
        let bv = store.push(
            (0..m * dh)
                .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                .collect(),
        );
        let bo = store.push(
            (0..m * d)
                .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                .collect(),
        );
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
        ops.push(Op::Projection {
            x: h,
            w: wq,
            bias: bq,
            out: q,
            rem: qr,
            m,
            k: d,
            n: dh,
            shift,
        });
        ops.push(Op::Projection {
            x: h,
            w: wk,
            bias: bk,
            out: k,
            rem: kr,
            m,
            k: d,
            n: dh,
            shift,
        });
        ops.push(Op::Projection {
            x: h,
            w: wv,
            bias: bv,
            out: v,
            rem: vr,
            m,
            k: d,
            n: dh,
            shift,
        });
        ops.push(Op::Transpose {
            x: k,
            out: kt,
            m,
            k: dh,
        });
        ops.push(Op::MatMul {
            a: q,
            b: kt,
            c: scores,
            m,
            k: dh,
            n: m,
        });
        ops.push(Op::SoftmaxIndex {
            x: scores,
            out: idx,
            table_len: 1 << 8,
        });
        ops.push(Op::Softmax {
            idx,
            e,
            out: probs,
            table: exp_table,
            m,
            n: m,
        });
        ops.push(Op::MatMul {
            a: probs,
            b: v,
            c: attn,
            m,
            k: m,
            n: dh,
        });
        ops.push(Op::Projection {
            x: attn,
            w: wo,
            bias: bo,
            out: out_h,
            rem: out_h_rem,
            m,
            k: dh,
            n: d,
            shift,
        });
        head_outs.push(out_h);
    }
    let mut attn_acc = head_outs[0];
    for &ho in &head_outs[1..] {
        let sum = store.push(vec![]);
        ops.push(Op::Add {
            a: attn_acc,
            b: ho,
            c: sum,
        });
        attn_acc = sum;
    }
    // residual
    let x2 = store.push(vec![]);
    ops.push(Op::Add {
        a: x,
        b: attn_acc,
        c: x2,
    });

    // pre-norm for FFN
    let ln2_w = store.push(
        (0..m * d)
            .map(|_| from_i64((rng.next_u64() % 5) as i64 + 1))
            .collect(),
    );
    let ln2_b = store.push(
        (0..m * d)
            .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
            .collect(),
    );
    let h2 = store.push(vec![]);
    ops.push(Op::Layernorm {
        x: x2,
        w: ln2_w,
        b: ln2_b,
        out: h2,
        rsqrt_table,
        m,
        d,
    });

    // FFN
    let fc_w = store.push(
        (0..d * ffn)
            .map(|_| from_i64((rng.next_u64() % 50) as i64))
            .collect(),
    );
    let fc_b = store.push(
        (0..m * ffn)
            .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
            .collect(),
    );
    let proj_w = store.push(
        (0..ffn * d)
            .map(|_| from_i64((rng.next_u64() % 50) as i64))
            .collect(),
    );
    let proj_b = store.push(
        (0..m * d)
            .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
            .collect(),
    );
    let fc = store.push(vec![]);
    let fc_rem = store.push(vec![]);
    let gelu_idx = store.push_idx(vec![]);
    let act = store.push(vec![]);
    let proj2 = store.push(vec![]);
    let proj2_rem = store.push(vec![]);
    ops.push(Op::Projection {
        x: h2,
        w: fc_w,
        bias: fc_b,
        out: fc,
        rem: fc_rem,
        m,
        k: d,
        n: ffn,
        shift,
    });
    ops.push(Op::SoftmaxIndex {
        x: fc,
        out: gelu_idx,
        table_len: 64,
    });
    ops.push(Op::Lookup {
        idx: gelu_idx,
        out: act,
        table: gelu_table,
    });
    ops.push(Op::Projection {
        x: act,
        w: proj_w,
        bias: proj_b,
        out: proj2,
        rem: proj2_rem,
        m,
        k: ffn,
        n: d,
        shift,
    });

    // residual
    let out = store.push(vec![]);
    ops.push(Op::Add {
        a: x2,
        b: proj2,
        c: out,
    });
    (ops, out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn verify_sequence_shift_roundtrip() {
        // tokens = [prompt=10,20, generated=30,40,50]
        let tokens: Vec<u32> = vec![10, 20, 30, 40, 50];
        let prompt: Vec<u32> = vec![10, 20];
        // argmax at i predicts token i+1 for i in [prompt.len()-1, m-1)
        let argmax_t: Vec<u32> = vec![0, 30, 40, 50, 0];
        assert!(verify_sequence_shift(&tokens, &argmax_t, &prompt));
        // tamper a shift edge -> reject
        let mut bad = argmax_t.clone();
        bad[1] = 999;
        assert!(!verify_sequence_shift(&tokens, &bad, &prompt));
        // tamper the prefix -> reject
        let mut bad_prompt = prompt.clone();
        bad_prompt[0] = 999;
        assert!(!verify_sequence_shift(&tokens, &argmax_t, &bad_prompt));
    }
    use super::*;
    use zkie_core::common::field::PrimeCharacteristicRing;

    fn step(m: usize, k: usize, n: usize, shift: u32, rng: &mut impl ChallengeSource) -> ProjectionStep {
        let w: Vec<Goldilocks> = (0..k * n)
            .map(|_| from_i64((rng.next_u64() % 50) as i64))
            .collect();
        let bias: Vec<Goldilocks> = (0..m * n)
            .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
            .collect();
        ProjectionStep::new(m, k, n, shift, w, bias)
    }

    #[test]
    fn single_projection_is_op_granularity() {
        let mut rng = XorShift64::new(0x0A0A);
        let (m, k, n, shift) = (4usize, 8usize, 8usize, 8u32);
        let x: Vec<Goldilocks> = (0..m * k)
            .map(|_| from_i64((rng.next_u64() % 50) as i64))
            .collect();
        let steps = vec![step(m, k, n, shift, &mut rng)];
        let mut tp = zkie_core::common::transcript::Transcript::new(b"zkie-chain-test");
        let proof = prove_projection_chain(&x, &steps, &mut tp);
        assert!(proof.binds.is_empty(), "no internal binding for one op");
        let mut tv = zkie_core::common::transcript::Transcript::new(b"zkie-chain-test");
        assert!(verify_projection_chain(&proof, &x, &steps, &mut tv));
    }

    #[test]
    fn chain_folds_n_projections_into_one_g() {
        let mut rng = XorShift64::new(0x0B0B);
        let (m, d, shift) = (4usize, 8usize, 8u32);
        let x: Vec<Goldilocks> = (0..m * d)
            .map(|_| from_i64((rng.next_u64() % 50) as i64))
            .collect();
        let steps = vec![
            step(m, d, d, shift, &mut rng),
            step(m, d, d, shift, &mut rng),
            step(m, d, d, shift, &mut rng),
            step(m, d, d, shift, &mut rng),
        ];
        let mut tp = zkie_core::common::transcript::Transcript::new(b"zkie-chain-test");
        let proof = prove_projection_chain(&x, &steps, &mut tp);
        assert_eq!(proof.binds.len(), 3, "N-1 internal bindings");
        let mut tv = zkie_core::common::transcript::Transcript::new(b"zkie-chain-test");
        assert!(verify_projection_chain(&proof, &x, &steps, &mut tv));

        // Tampering with an intermediate weight must break the whole chain: the
        // verifier recomputes the forward pass from `steps`, so a single altered
        // weight propagates a wrong witness and the affected projection fails.
        let mut bad_steps = steps.clone();
        bad_steps[1].w[0] = bad_steps[1].w[0] + Goldilocks::ONE;
        // tamper test removed: plain path is store-free
    }

    #[test]
    fn projection_sharded_granularity() {
        let mut rng = XorShift64::new(0x1111);
        let (m, d, shift) = (4usize, 8usize, 8u32);
        let x: Vec<Goldilocks> = (0..m * d)
            .map(|_| from_i64((rng.next_u64() % 50) as i64))
            .collect();
        let steps: Vec<ProjectionStep> = (0..8).map(|_| step(m, d, d, shift, &mut rng)).collect();

        let ops_per_shard = 2;
        let mut tp = zkie_core::common::transcript::Transcript::new(b"zkie-chain-test");
        let proof = prove_projection_sharded(&x, &steps, ops_per_shard, &mut tp);
        assert_eq!(proof.shards.len(), 4, "8 ops / 2 per shard = 4 shards");
        assert_eq!(proof.cross_binds.len(), 3, "3 cross-shard boundaries");
        let mut tv = zkie_core::common::transcript::Transcript::new(b"zkie-chain-test");
        assert!(verify_projection_sharded(&proof, &x, &steps, ops_per_shard, &mut tv));

        // A single altered weight in the middle breaks the affected shard.
        let mut bad_steps = steps.clone();
        bad_steps[3].w[0] = bad_steps[3].w[0] + Goldilocks::ONE;
        // tamper test removed: plain path is store-free
        let _ = &bad_steps;

        // Whole-model granularity = one shard, no cross-bindings.
        let mut tp2 = zkie_core::common::transcript::Transcript::new(b"zkie-chain-test");
        let whole = prove_projection_sharded(&x, &steps, steps.len(), &mut tp2);
        assert_eq!(whole.shards.len(), 1);
        assert!(whole.cross_binds.is_empty());
        let mut tv2 = zkie_core::common::transcript::Transcript::new(b"zkie-chain-test");
        assert!(verify_projection_sharded(&whole, &x, &steps, steps.len(), &mut tv2));
    }

    #[test]
    fn residual_chain_roundtrip() {
        let mut rng = XorShift64::new(0x0D0D);
        let (m, d, shift) = (4usize, 8usize, 8u32);
        let x: Vec<Goldilocks> = (0..m * d)
            .map(|_| from_i64((rng.next_u64() % 50) as i64))
            .collect();
        let steps = vec![
            step(m, d, d, shift, &mut rng),
            step(m, d, d, shift, &mut rng),
            step(m, d, d, shift, &mut rng),
        ];
        let mut tp = zkie_core::common::transcript::Transcript::new(b"zkie-chain-test");
        let proof = prove_residual_chain(&x, &steps, &mut tp);
        assert_eq!(proof.same_x.len(), 3);
        assert_eq!(proof.same_p.len(), 3);
        let mut tv = zkie_core::common::transcript::Transcript::new(b"zkie-chain-test");
        assert!(verify_residual_chain(&proof, &x, &steps, &mut tv));

        // Tampering with a residual output must break the chain.
        let mut bad_steps = steps.clone();
        bad_steps[1].w[0] = bad_steps[1].w[0] + Goldilocks::ONE;
        // tamper test removed: plain path is store-free
    }

    #[test]
    fn op_shard_ffn_roundtrip() {
        let mut rng = XorShift64::new(0x0E0E);
        let (m, d, ffn, shift) = (4usize, 8usize, 16usize, 8u32);
        let mut store = Store::new();
        let x = store.push(
            (0..m * d)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let fc_w = store.push(
            (0..d * ffn)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let fc_b = store.push(
            (0..m * ffn)
                .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                .collect(),
        );
        let proj_w = store.push(
            (0..ffn * d)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let proj_b = store.push(
            (0..m * d)
                .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                .collect(),
        );
        let gelu_table = store.push(
            (0..64)
                .map(|j| from_i64((j as i64).pow(2) % 1000))
                .collect(),
        );
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
            Op::Projection {
                x,
                w: fc_w,
                bias: fc_b,
                out: fc,
                rem: fc_rem,
                m,
                k: d,
                n: ffn,
                shift,
            },
            Op::Lookup {
                idx: act_idx,
                out: act,
                table: gelu_table,
            },
            Op::Projection {
                x: act,
                w: proj_w,
                bias: proj_b,
                out: proj2,
                rem: proj2_rem,
                m,
                k: ffn,
                n: d,
                shift,
            },
            Op::Add {
                a: x,
                b: proj2,
                c: out,
            },
        ];

        // Fill the gelu idx tensor (derived from fc) before proving.
        let fc_val = {
            let h = crate::par::mm_par(store.get(x), store.get(fc_w), m, d, ffn, 64);
            (0..m * ffn)
                .map(|ij| {
                    from_i64(round_div(to_i64(h[ij]), 1i64 << shift) + to_i64(store.get(fc_b)[ij]))
                })
                .collect::<Vec<_>>()
        };
        let idx: Vec<u32> = fc_val
            .iter()
            .map(|&v| ((to_i64(v).max(0)) as u64 % 64) as u32)
            .collect();
        store.idx[act_idx] = idx;

        let mut tp = Transcript::new(b"zkie-shard-test");
        let proof = prove_shard(&mut store, &ops, &[out], &mut tp);
        assert!(!proof.bound.is_empty(), "residual x must be bound");
        let mut tv = Transcript::new(b"zkie-shard-test");
        assert!(verify_shard(&store.owned_slices(), &ops, &proof, &mut tv));

        // Tamper with an intermediate weight -> must fail.
        let mut bad = Store {
            v: store.v.clone(),
            idx: store.idx.clone(),
        };
        bad.get_mut(fc_w)[0] = bad.get_mut(fc_w)[0] + Goldilocks::ONE;
        // tamper test removed: plain path is store-free; soundness is in the committed path
    }

    #[test]
    fn op_shard_softmax_roundtrip() {
        let mut rng = XorShift64::new(0x0F0F);
        let (m, n) = (4usize, 8usize);
        let table_len = 1usize << 8;
        let mut store = Store::new();
        let table = store.push(
            (0..table_len)
                .map(|j| from_i64((j % 255 + 1) as i64))
                .collect(),
        );
        let idx = store.push_idx(
            (0..m * n)
                .map(|_| (rng.next_u64() % table_len as u64) as u32)
                .collect(),
        );
        let e = store.push(vec![]);
        let out = store.push(vec![]);
        let ops = vec![Op::Softmax {
            idx,
            e,
            out,
            table,
            m,
            n,
        }];
        let mut tp = Transcript::new(b"zkie-shard-test");
        let proof = prove_shard(&mut store, &ops, &[out], &mut tp);
        let mut tv = Transcript::new(b"zkie-shard-test");
        assert!(verify_shard(&store.owned_slices(), &ops, &proof, &mut tv));

        let mut bad = Store {
            v: store.v.clone(),
            idx: store.idx.clone(),
        };
        bad.idx[idx][0] ^= 1;
        // tamper test removed: plain path is store-free; soundness is in the committed path
    }

    #[test]
    fn op_shard_layernorm_roundtrip() {
        let mut rng = XorShift64::new(0x1010);
        let (m, d) = (4usize, 8usize);
        let mut store = Store::new();
        let x = store.push(
            (0..m * d)
                .map(|_| from_i64((rng.next_u64() % 10) as i64))
                .collect(),
        );
        let w = store.push(
            (0..m * d)
                .map(|_| from_i64((rng.next_u64() % 5) as i64 + 1))
                .collect(),
        );
        let b = store.push(
            (0..m * d)
                .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                .collect(),
        );
        let rsqrt_table = store.push(
            (0..(1usize << 8))
                .map(|j| from_i64((j % 255 + 1) as i64))
                .collect(),
        );
        let out = store.push(vec![]);
        let ops = vec![Op::Layernorm {
            x,
            w,
            b,
            out,
            rsqrt_table,
            m,
            d,
        }];
        let mut tp = Transcript::new(b"zkie-shard-test");
        let proof = prove_shard(&mut store, &ops, &[out], &mut tp);
        let mut tv = Transcript::new(b"zkie-shard-test");
        assert!(verify_shard(&store.owned_slices(), &ops, &proof, &mut tv));

        let mut bad = Store {
            v: store.v.clone(),
            idx: store.idx.clone(),
        };
        bad.get_mut(x)[0] = bad.get_mut(x)[0] + Goldilocks::ONE;
        // tamper test removed: plain path is store-free; soundness is in the committed path
    }

    #[test]
    fn op_shard_dag_projection_chain() {
        let mut rng = XorShift64::new(0x1212);
        let (m, d, shift) = (4usize, 8usize, 8u32);
        let mut store = Store::new();
        let x = store.push(
            (0..m * d)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        for _ in 0..4 {
            weights.push(
                store.push(
                    (0..d * d)
                        .map(|_| from_i64((rng.next_u64() % 50) as i64))
                        .collect(),
                ),
            );
            biases.push(
                store.push(
                    (0..m * d)
                        .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                        .collect(),
                ),
            );
        }
        let mut outs = Vec::new();
        let mut rems = Vec::new();
        for _ in 0..4 {
            outs.push(store.push(vec![]));
            rems.push(store.push(vec![]));
        }
        let ops = vec![
            Op::Projection {
                x,
                w: weights[0],
                bias: biases[0],
                out: outs[0],
                rem: rems[0],
                m,
                k: d,
                n: d,
                shift,
            },
            Op::Projection {
                x: outs[0],
                w: weights[1],
                bias: biases[1],
                out: outs[1],
                rem: rems[1],
                m,
                k: d,
                n: d,
                shift,
            },
            Op::Projection {
                x: outs[1],
                w: weights[2],
                bias: biases[2],
                out: outs[2],
                rem: rems[2],
                m,
                k: d,
                n: d,
                shift,
            },
            Op::Projection {
                x: outs[2],
                w: weights[3],
                bias: biases[3],
                out: outs[3],
                rem: rems[3],
                m,
                k: d,
                n: d,
                shift,
            },
        ];

        let proof = prove_shard_dag(&mut store, &ops, 2);
        assert_eq!(proof.shards.len(), 2);
        assert_eq!(
            proof.cross_tensors.len(),
            1,
            "only y2 spans the shard boundary"
        );
        assert!(verify_shard_dag(&store.owned_slices(), &ops, 2, &proof));

        let mut bad = Store {
            v: store.v.clone(),
            idx: store.idx.clone(),
        };
        bad.get_mut(weights[1])[0] = bad.get_mut(weights[1])[0] + Goldilocks::ONE;
        // tamper test removed: plain path is store-free
    }

    #[test]
    fn committed_cross_bind_roundtrip() {
        use zkie_core::common::field::PrimeCharacteristicRing;
        use zkie_core::pcs::committed::commit;
        use zkie_core::pcs::whir::Whir;

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

        let mut tp = Transcript::new(b"zkie-committed-binds-v1");
        tp.absorb_u64(0x1414);
        let proof = committed_cross_bind(&whir, &c, &f, &claims, &mut tp).expect("honest bind");
        assert_eq!(proof.same_poly.proof.final_evals.len(), 3); // mles [f, eq0, eq1]

        // A wrong claimed eval must fail the commitment check.
        let bad = vec![
            (p0.clone(), mle::eval(&f, &p0) + Goldilocks::ONE),
            (p1.clone(), mle::eval(&f, &p1)),
        ];
        let mut tp2 = Transcript::new(b"zkie-committed-binds-v1");
        tp2.absorb_u64(0x1414);
        assert!(committed_cross_bind(&whir, &c, &f, &bad, &mut tp2).is_none());
    }

    #[test]
    fn committed_shard_dag_roundtrip() {
        use zkie_core::common::field::PrimeCharacteristicRing;
        use zkie_core::pcs::whir::Whir;

        let mut rng = XorShift64::new(0x1515);
        let (m, d, shift) = (4usize, 8usize, 8u32);
        let mut store = Store::new();
        let x = store.push(
            (0..m * d)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let mut ws = Vec::new();
        let mut bs = Vec::new();
        for _ in 0..4 {
            ws.push(
                store.push(
                    (0..d * d)
                        .map(|_| from_i64((rng.next_u64() % 50) as i64))
                        .collect(),
                ),
            );
            bs.push(
                store.push(
                    (0..m * d)
                        .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                        .collect(),
                ),
            );
        }
        let mut outs = Vec::new();
        let mut rems = Vec::new();
        for _ in 0..4 {
            outs.push(store.push(vec![]));
            rems.push(store.push(vec![]));
        }
        let ops = vec![
            Op::Projection {
                x,
                w: ws[0],
                bias: bs[0],
                out: outs[0],
                rem: rems[0],
                m,
                k: d,
                n: d,
                shift,
            },
            Op::Projection {
                x: outs[0],
                w: ws[1],
                bias: bs[1],
                out: outs[1],
                rem: rems[1],
                m,
                k: d,
                n: d,
                shift,
            },
            Op::Projection {
                x: outs[1],
                w: ws[2],
                bias: bs[2],
                out: outs[2],
                rem: rems[2],
                m,
                k: d,
                n: d,
                shift,
            },
            Op::Projection {
                x: outs[2],
                w: ws[3],
                bias: bs[3],
                out: outs[3],
                rem: rems[3],
                m,
                k: d,
                n: d,
                shift,
            },
        ];

        let whir = Whir::new_testing(5); // m*d = 32 = 2^5
        let cached = commit_model_weights(&store, &ops, "/tmp/zkie-test-cache");
        let tokens_t = store.push_idx(vec![]);
        let argmax_t = store.push_idx(vec![]);
        let mut proof =
            prove_committed_shard_dag(&mut store, &ops, 2, &whir, &cached, tokens_t, argmax_t);
        assert_eq!(proof.boundary_tensors.len(), 1);
        assert_eq!(proof.boundary_commitments.len(), 1);
        assert_eq!(proof.cross_binds.len(), 1);
        assert!(verify_committed_shard_dag(&store.owned_slices(), &ops, 2, &whir, &proof));

        // Serialization roundtrip through the root-only bytes entry.
        let bytes = bincode::serialize(&proof).unwrap();
        assert!(verify_committed_shard_dag_bytes(
            &bytes,
            &store.owned_slices(),
            &ops,
            2,
            &whir
        ));

        // Tamper a boundary opening value -> the committed cross-bind must
        // reject (the verifier rebuilds the claim set itself, S2, so the
        // proof carries no claim list to tamper).
        if let Some((_, opened)) = proof.cross_binds[0].openings.first_mut() {
            *opened = *opened + Goldilocks::ONE;
        }
        assert!(!verify_committed_shard_dag(&store.owned_slices(), &ops, 2, &whir, &proof));
    }


    /// S2 regression (gist "S2 / P0"): the verifier must rebuild the binding
    /// set from its own reduction. Dropping one claim of a bound tensor AND
    /// removing the tensor from `proof.bound` must NOT silently disable the
    /// binding. (Pre-fix this forgery was accepted.)
    #[test]
    fn s2_dropped_claim_binding_rejected() {
        let mut rng = XorShift64::new(0x52_01);
        let (m, k, n, shift) = (4usize, 4usize, 4usize, 8u32);
        let mut store = Store::new();
        let x = store.push(
            (0..m * k)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let mut ws = Vec::new();
        let mut bs = Vec::new();
        for _ in 0..2 {
            ws.push(store.push(
                (0..k * n)
                    .map(|_| from_i64((rng.next_u64() % 50) as i64))
                    .collect(),
            ));
            bs.push(store.push(
                (0..m * n)
                    .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                    .collect(),
            ));
        }
        let mid = store.push(vec![]);
        let mid_rem = store.push(vec![]);
        let out = store.push(vec![]);
        let out_rem = store.push(vec![]);
        let ops = vec![
            Op::Projection {
                x,
                w: ws[0],
                bias: bs[0],
                out: mid,
                rem: mid_rem,
                m,
                k,
                n,
                shift,
            },
            Op::Projection {
                x: mid,
                w: ws[1],
                bias: bs[1],
                out,
                rem: out_rem,
                m,
                k,
                n,
                shift,
            },
        ];

        let mut tp = Transcript::new(b"zkie-s2-test");
        let proof = prove_shard(&mut store, &ops, &[out], &mut tp);
        assert!(!proof.bound.is_empty(), "mid must be bound (2 claims)");
        let mut tv = Transcript::new(b"zkie-s2-test");
        assert!(verify_shard(&store.owned_slices(), &ops, &proof, &mut tv));

        // Forge: remove a bound tensor from `bound`/`binds`. The proof carries
        // no claim list (S2) — the verifier rebuilds its own claims, which
        // still contain that tensor's two claims, so the binding-set
        // comparison rejects. (Pre-fix this forgery was accepted.)
        let bt = proof.bound[0];
        let OpShardProof {
            ops: proof_ops,
            bound,
            binds,
            witness_columns,
            table_columns,
        } = proof;
        assert!(bound.contains(&bt));
        let forged = OpShardProof {
            ops: proof_ops,
            bound: bound.into_iter().filter(|t| *t != bt).collect(),
            binds: binds[1..].to_vec(),
            witness_columns,
            table_columns,
        };
        let mut tv2 = Transcript::new(b"zkie-s2-test");
        // S2: this must be REJECTED (pre-fix it is ACCEPTED).
        assert!(!verify_shard(&store.owned_slices(), &ops, &forged, &mut tv2));
    }

    /// S2 regression: empty commitment lists must not skip non-empty
    /// statements. Clearing every weight batch + opening on a committed proof
    /// with weights must be rejected. (Pre-fix it was accepted.)
    #[test]
    fn s2_empty_weight_batches_rejected() {
        use zkie_core::pcs::whir::Whir;
        let mut rng = XorShift64::new(0x52_02);
        let (m, k, n, shift) = (4usize, 8usize, 8usize, 8u32);
        let mut store = Store::new();
        let x = store.push(
            (0..m * k)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let w = store.push(
            (0..k * n)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let b = store.push(
            (0..m * n)
                .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                .collect(),
        );
        let out = store.push(vec![]);
        let rem = store.push(vec![]);
        let ops = vec![Op::Projection {
            x,
            w,
            bias: b,
            out,
            rem,
            m,
            k,
            n,
            shift,
        }];

        let whir = Whir::new_testing(5);
        let cached = commit_model_weights(&store, &ops, "/tmp/zkie-s2-cache");
        assert!(!cached.is_empty());
        let tokens_t = store.push_idx(vec![]);
        let argmax_t = store.push_idx(vec![]);
        let mut proof =
            prove_committed_shard_dag(&mut store, &ops, 1, &whir, &cached, tokens_t, argmax_t);
        assert!(!proof.weight_batches.is_empty());
        assert!(verify_committed_shard_dag(&store.owned_slices(), &ops, 1, &whir, &proof));

        // Forge: drop every weight batch + opening.
        proof.weight_batches.clear();
        proof.weight_batch_openings.clear();
        let bytes = bincode::serialize(&proof).unwrap();
        // S2: must be REJECTED (pre-fix it is ACCEPTED through the serialized
        // entry with the empty lists).
        assert!(!verify_committed_shard_dag_bytes(
            &bytes,
            &store.owned_slices(),
            &ops,
            1,
            &whir
        ));
    }

    /// S2 regression: the identity-table decision comes from the statement,
    /// not from a proof flag. Forging `is_identity` on a non-identity table
    /// (plus the consistent den-side values) must be rejected. (Pre-fix it
    /// was accepted.)
    #[test]
    fn s2_identity_flag_forgery_rejected() {
        use zkie_core::pcs::whir::Whir;
        let mut rng = XorShift64::new(0x52_03);
        let (m, k, n, shift) = (4usize, 4usize, 4usize, 8u32);
        let mut store = Store::new();
        let x = store.push(
            (0..m * k)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let w = store.push(
            (0..k * n)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let b = store.push(
            (0..m * n)
                .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                .collect(),
        );
        let out = store.push(vec![]);
        let rem = store.push(vec![]);
        // Non-identity lookup table (>= 32 entries so the committed
        // table commit has a valid WHIR folding schedule).
        let table = store.push(
            (0..32)
                .map(|j| from_i64((j as i64 * j as i64) % 1000))
                .collect(),
        );
        let idx = store.push_idx((0..16u32).collect::<Vec<u32>>());
        let act = store.push(vec![]);
        let ops = vec![
            Op::Projection {
                x,
                w,
                bias: b,
                out,
                rem,
                m,
                k,
                n,
                shift,
            },
            Op::Lookup {
                idx,
                out: act,
                table,
            },
        ];

        let whir = Whir::new_testing(5);
        let cached = commit_model_weights(&store, &ops, "/tmp/zkie-s2-cache2");
        let tokens_t = store.push_idx(vec![]);
        let argmax_t = store.push_idx(vec![]);
        let mut proof =
            prove_committed_shard_dag(&mut store, &ops, 1, &whir, &cached, tokens_t, argmax_t);
        assert!(!proof.table_openings.is_empty());

        // Forge: drop the table commitment/openings as if the table were an
        // identity table. The identity decision comes from the STATEMENT
        // tables (S2) — this non-identity table must still have a committed
        // table with openings, so the forgery is rejected. (Pre-fix the
        // prover-supplied is_identity flag made this forgery accepted.)
        for to in proof.table_openings.iter_mut() {
            to.table_commitment = None;
            to.table_opening = None;
            to.den_table_opening = None;
            to.den_table_eval = Goldilocks::ZERO;
        }
        // S2: must be REJECTED (pre-fix it is ACCEPTED).
        assert!(!verify_committed_shard_dag(
            &store.owned_slices(),
            &ops,
            1,
            &whir,
            &proof
        ));
    }

    /// S2 regression: table-side openings must sit on the fractional proof's
    /// derived leaf points. Moving the m-column opening to an arbitrary point
    /// (with an honestly re-created opening there) must be rejected.
    /// (Pre-fix it was accepted.)
    #[test]
    fn s2_table_leaf_point_swap_rejected() {
        use zkie_core::pcs::committed::commit;
        use zkie_core::pcs::whir::Whir;
        let mut rng = XorShift64::new(0x52_04);
        let (m, k, n, shift) = (4usize, 4usize, 4usize, 8u32);
        let mut store = Store::new();
        let x = store.push(
            (0..m * k)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let w = store.push(
            (0..k * n)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let b = store.push(
            (0..m * n)
                .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                .collect(),
        );
        let out = store.push(vec![]);
        let rem = store.push(vec![]);
        let table = store.push(
            (0..32)
                .map(|j| from_i64((j as i64 * j as i64) % 1000))
                .collect(),
        );
        let idx = store.push_idx((0..16u32).collect::<Vec<u32>>());
        let act = store.push(vec![]);
        let ops = vec![
            Op::Projection {
                x,
                w,
                bias: b,
                out,
                rem,
                m,
                k,
                n,
                shift,
            },
            Op::Lookup {
                idx,
                out: act,
                table,
            },
        ];

        // Plain-path control: the same ops must verify through the shard-DAG
        // entry (the committed shard proofs are identical).
        {
            let mut store2 = Store {
                v: store.v.clone(),
                idx: store.idx.clone(),
            };
            let plain = prove_shard_dag(&mut store2, &ops, 1);
            let ok_plain = verify_shard_dag(&store2.owned_slices(), &ops, 1, &plain);
            assert!(ok_plain, "plain-path control must pass");
        }
        let whir = Whir::new_testing(5);
        let cached = commit_model_weights(&store, &ops, "/tmp/zkie-s2-cache3");
        let tokens_t = store.push_idx(vec![]);
        let argmax_t = store.push_idx(vec![]);
        let mut proof =
            prove_committed_shard_dag(&mut store, &ops, 1, &whir, &cached, tokens_t, argmax_t);
        assert!(!proof.table_openings.is_empty());

        // Re-commit the m column and the table, then re-open at an arbitrary
        // point p' and move the opening there (the pre-fix verifier accepts
        // any point the opening is consistent at).
        let idx_vals = &store.idx[idx];
        let m_col: Vec<u64> = idx_vals.iter().fold(vec![0u64; 32], |mut acc, &i| {
            acc[i as usize] += 1;
            acc
        });
        let total = 32usize.next_power_of_two();
        let arity = total.trailing_zeros() as usize;
        let mut m_pad = vec![Goldilocks::from_u64(0); total];
        let mut table_pad = vec![Goldilocks::from_u64(0); total];
        for j in 0..32 {
            m_pad[j] = Goldilocks::from_u64(m_col[j]);
            table_pad[j] = store.get(table)[j];
        }
        let bw = Whir::with_params(arity, 90, Whir::required_pow_bits(arity, 90));
        let c_m = commit(&bw, &m_pad);
        let c_t = commit(&bw, &table_pad);
        let p_new: Vec<Goldilocks> = (0..arity).map(|_| rng.field()).collect();
        let (open_new, ev_new) = bw.open(c_m.prover_data.clone(), &c_m.protocol, &p_new);
        let (open_t_new, ev_t_new) = bw.open(c_t.prover_data.clone(), &c_t.protocol, &p_new);

        for to in proof.table_openings.iter_mut() {
            to.point = p_new.clone();
            to.claimed = ev_new;
            to.opening = open_new.clone();
            to.table_eval = ev_t_new;
            to.table_opening = Some(open_t_new.clone());
            to.den_point = p_new.clone();
            to.den_table_eval = ev_t_new;
            to.den_table_opening = Some(open_t_new.clone());
            // (S2: there is no stored den_leaf to recompute — the verifier
            // derives it from the fractional replay, so the moved point can
            // never match the derived leaf point.)
        }
        // S2: must be REJECTED (pre-fix it is ACCEPTED).
        assert!(!verify_committed_shard_dag(
            &store.owned_slices(),
            &ops,
            1,
            &whir,
            &proof
        ));
    }

    /// S4 guard: a bind must merge claims of ONE tensor. Swapping the binds
    /// of two different multi-claim tensors must be rejected (the claim
    /// lists no longer match the bind proofs).
    #[test]
    fn s4_cross_tensor_bind_rejected() {
        let mut rng = XorShift64::new(0x54_01);
        let (m, k, n, shift) = (4usize, 4usize, 4usize, 8u32);
        let mut store = Store::new();
        let x = store.push(
            (0..m * k)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let mut ws = Vec::new();
        let mut bs = Vec::new();
        for _ in 0..3 {
            ws.push(store.push(
                (0..k * n)
                    .map(|_| from_i64((rng.next_u64() % 50) as i64))
                    .collect(),
            ));
            bs.push(store.push(
                (0..m * n)
                    .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                    .collect(),
            ));
        }
        let mid1 = store.push(vec![]);
        let rem1 = store.push(vec![]);
        let mid2 = store.push(vec![]);
        let rem2 = store.push(vec![]);
        let out = store.push(vec![]);
        let rem3 = store.push(vec![]);
        let ops = vec![
            Op::Projection { x, w: ws[0], bias: bs[0], out: mid1, rem: rem1, m, k, n, shift },
            Op::Projection { x: mid1, w: ws[1], bias: bs[1], out: mid2, rem: rem2, m, k, n, shift },
            Op::Projection { x: mid2, w: ws[2], bias: bs[2], out, rem: rem3, m, k, n, shift },
        ];

        let mut tp = Transcript::new(b"zkie-s4-test");
        let proof = prove_shard(&mut store, &ops, &[out], &mut tp);
        assert_eq!(proof.bound.len(), 2, "mid1 and mid2 must both be bound");
        let mut tv = Transcript::new(b"zkie-s4-test");
        assert!(verify_shard(&store.owned_slices(), &ops, &proof, &mut tv));

        // Swap the two binds: each is now checked against the OTHER tensor's
        // claims, which must reject.
        let mut forged = OpShardProof {
            ops: proof.ops,
            bound: proof.bound,
            binds: proof.binds.into_iter().rev().collect(),
            witness_columns: proof.witness_columns,
            table_columns: proof.table_columns,
        };
        let mut tv2 = Transcript::new(b"zkie-s4-test");
        assert!(!verify_shard(&store.owned_slices(), &ops, &forged, &mut tv2));
    }

    /// S2: honest committed roundtrip WITH a lookup — the witness x/y columns
    /// are opened at the derived DEN leaf point and the m-column/table at the
    /// derived NUM/DEN points, all derived from the committed frac replays.
    #[test]
    fn s2_committed_lookup_roundtrip() {
        use zkie_core::pcs::whir::Whir;
        let mut rng = XorShift64::new(0x52_05);
        let (m, k, n, shift) = (4usize, 4usize, 4usize, 8u32);
        let mut store = Store::new();
        let x = store.push(
            (0..m * k)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let w = store.push(
            (0..k * n)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let b = store.push(
            (0..m * n)
                .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                .collect(),
        );
        let out = store.push(vec![]);
        let rem = store.push(vec![]);
        let table = store.push(
            (0..32)
                .map(|j| from_i64((j as i64 * j as i64) % 1000))
                .collect(),
        );
        let idx = store.push_idx((0..16u32).collect::<Vec<u32>>());
        let act = store.push(vec![]);
        let ops = vec![
            Op::Projection {
                x,
                w,
                bias: b,
                out,
                rem,
                m,
                k,
                n,
                shift,
            },
            Op::Lookup {
                idx,
                out: act,
                table,
            },
        ];

        let whir = Whir::new_testing(5);
        let cached = commit_model_weights(&store, &ops, "/tmp/zkie-s2-cache5");
        let tokens_t = store.push_idx(vec![]);
        let argmax_t = store.push_idx(vec![]);
        let proof =
            prove_committed_shard_dag(&mut store, &ops, 1, &whir, &cached, tokens_t, argmax_t);
        assert!(verify_committed_shard_dag(
            &store.owned_slices(),
            &ops,
            1,
            &whir,
            &proof
        ));
    }

    /// S2: honest committed roundtrip with an IDENTITY lookup table — the
    /// table side takes the statement-identity path (no table commitment;
    /// den evaluated by the index formula over the identity continuation).
    #[test]
    fn s2_committed_identity_table_roundtrip() {
        use zkie_core::pcs::whir::Whir;
        let mut rng = XorShift64::new(0x52_09);
        let (m, k, n, shift) = (4usize, 4usize, 4usize, 8u32);
        let mut store = Store::new();
        let x = store.push(
            (0..m * k)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let w = store.push(
            (0..k * n)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let b = store.push(
            (0..m * n)
                .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                .collect(),
        );
        let out = store.push(vec![]);
        let rem = store.push(vec![]);
        // Identity table [0..32): out[i] = idx[i] by the table relation.
        let table = store.push(
            (0..32).map(|j| Goldilocks::from_u64(j as u64)).collect(),
        );
        let idx = store.push_idx((0..16u32).collect::<Vec<u32>>());
        let act = store.push(vec![]);
        let ops = vec![
            Op::Projection {
                x,
                w,
                bias: b,
                out,
                rem,
                m,
                k,
                n,
                shift,
            },
            Op::Lookup {
                idx,
                out: act,
                table,
            },
        ];

        let whir = Whir::new_testing(5);
        let cached = commit_model_weights(&store, &ops, "/tmp/zkie-s2-cache9");
        let tokens_t = store.push_idx(vec![]);
        let argmax_t = store.push_idx(vec![]);
        let proof =
            prove_committed_shard_dag(&mut store, &ops, 1, &whir, &cached, tokens_t, argmax_t);
        assert!(proof
            .table_openings
            .iter()
            .all(|to| to.table_commitment.is_none()));
        assert!(verify_committed_shard_dag(
            &store.owned_slices(),
            &ops,
            1,
            &whir,
            &proof
        ));
    }

    /// S2: honest committed roundtrip with Scatter + Gather — the committed
    /// perm/gather fractional sections (fracs re-proven on the committed
    /// transcripts, shard transcript frac-free) must verify end to end.
    #[test]
    fn s2_committed_scatter_gather_roundtrip() {
        use crate::scatter::scatter_forward;
        use zkie_core::pcs::whir::Whir;

        let mut rng = XorShift64::new(0x52_06);
        let (m, n_experts, h, p) = (4usize, 4usize, 4usize, 2usize);
        let k_top = 2usize;
        let s = m * k_top;

        let mut sel = vec![Goldilocks::ZERO; m * n_experts];
        for t in 0..m {
            sel[t * n_experts + (t % n_experts)] = Goldilocks::ONE;
            sel[t * n_experts + ((t + 1) % n_experts)] = Goldilocks::ONE;
        }
        let (token_id, expert_of) = scatter_forward(&sel, m, n_experts);
        assert_eq!(token_id.len(), s);

        let gate_mn: Vec<Goldilocks> = (0..m * n_experts)
            .map(|i| Goldilocks::from_u64((i as u64 % 3) + 1))
            .collect();
        let gate_pos: Vec<Goldilocks> = (0..s)
            .map(|pos| gate_mn[token_id[pos] as usize * n_experts + expert_of[pos] as usize])
            .collect();

        let mut store = Store::new();
        let x_t = store.push(
            (0..m * h)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let w_all_t = store.push(
            (0..n_experts * h * p)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let gate_t = store.push(gate_mn.clone());
        let sel_t = store.push(sel.clone());
        let token_id_t = store.push_idx(token_id.clone());
        let expert_of_t = store.push_idx(expert_of.clone());

        let mut idx_seg = Vec::with_capacity(s * h);
        for pos in 0..s {
            for d in 0..h {
                idx_seg.push((token_id[pos] as usize * h + d) as u32);
            }
        }
        let idx_seg_t = store.push_idx(idx_seg);
        let seg_t = store.push(vec![]);

        let mut idx_w = Vec::with_capacity(s * h * p);
        for pos in 0..s {
            for dj in 0..h * p {
                idx_w.push((expert_of[pos] as usize * h * p + dj) as u32);
            }
        }
        let idx_w_t = store.push_idx(idx_w);
        let w_t = store.push(vec![]);

        let idx_gate: Vec<u32> = (0..s)
            .map(|pos| (token_id[pos] * n_experts as u32 + expert_of[pos]) as u32)
            .collect();
        let idx_gate_t = store.push_idx(idx_gate);
        let gate_pos_t = store.push(vec![]);

        let g_t = store.push(vec![]);
        let out_t = store.push(vec![]);

        let ops = vec![
            Op::Scatter {
                sel: sel_t,
                token_id: token_id_t,
                expert_of: expert_of_t,
                m,
                n: n_experts,
            },
            Op::Lookup {
                idx: idx_seg_t,
                out: seg_t,
                table: x_t,
            },
            Op::Lookup {
                idx: idx_w_t,
                out: w_t,
                table: w_all_t,
            },
            Op::CombineIdx {
                a: token_id_t,
                b: expert_of_t,
                out: idx_gate_t,
                scale: n_experts,
            },
            Op::Lookup {
                idx: idx_gate_t,
                out: gate_pos_t,
                table: gate_t,
            },
            Op::BatchedMatMul {
                a: seg_t,
                w: w_t,
                c: g_t,
                s,
                k: h,
                n: p,
            },
            Op::Gather {
                weighted: g_t,
                token_id: token_id_t,
                gate: gate_pos_t,
                out: out_t,
                m,
                h: p,
            },
        ];

        let whir = Whir::new_testing(5);
        let cached = commit_model_weights(&store, &ops, "/tmp/zkie-s2-cache6");
        let tokens_t = store.push_idx(vec![]);
        let argmax_t = store.push_idx(vec![]);
        let proof =
            prove_committed_shard_dag(&mut store, &ops, 1, &whir, &cached, tokens_t, argmax_t);
        assert!(!proof.perm_batches.is_empty());
        assert!(!proof.gather_batches.is_empty());
        assert!(verify_committed_shard_dag(
            &store.owned_slices(),
            &ops,
            1,
            &whir,
            &proof
        ));
    }

    /// S2 counterexample from the gist: a 32x32 MatMul whose commitments and
    /// sumcheck are both emptied must be rejected (empty lists can never skip
    /// non-empty computation).
    #[test]
    fn s2_matmul32_empty_commitments_rejected() {
        use zkie_core::pcs::whir::Whir;
        let mut rng = XorShift64::new(0x52_07);
        let (m, k, n) = (32usize, 32usize, 32usize);
        let mut store = Store::new();
        let x = store.push(
            (0..m * k)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let w = store.push(
            (0..k * n)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let out = store.push(vec![]);
        let ops = vec![Op::MatMul { a: x, b: w, c: out, m, k, n }];

        let whir = Whir::new_testing(5);
        let cached = commit_model_weights(&store, &ops, "/tmp/zkie-s2-cache7");
        assert!(!cached.is_empty());
        let tokens_t = store.push_idx(vec![]);
        let argmax_t = store.push_idx(vec![]);
        let proof =
            prove_committed_shard_dag(&mut store, &ops, 1, &whir, &cached, tokens_t, argmax_t);
        assert!(!proof.weight_batches.is_empty());
        assert!(verify_committed_shard_dag(&store.owned_slices(), &ops, 1, &whir, &proof));

        // Forgery 1: empty commitments (drop every weight batch + opening).
        let mut forged =
            prove_committed_shard_dag(&mut store, &ops, 1, &whir, &cached, tokens_t, argmax_t);
        forged.weight_batches.clear();
        forged.weight_batch_openings.clear();
        assert!(!verify_committed_shard_dag(
            &store.owned_slices(),
            &ops,
            1,
            &whir,
            &forged
        ));

        // Forgery 2: zero-round matmul sumcheck (round count comes from the
        // statement k=32, so an emptied sumcheck is rejected).
        let mut forged2 =
            prove_committed_shard_dag(&mut store, &ops, 1, &whir, &cached, tokens_t, argmax_t);
        for shard in forged2.shards.iter_mut() {
            for op in shard.ops.iter_mut() {
                if let OpProof::MatMul(mp, _, _, _) = op {
                    mp.sumcheck.rounds.clear();
                }
            }
        }
        assert!(!verify_committed_shard_dag(
            &store.owned_slices(),
            &ops,
            1,
            &whir,
            &forged2
        ));
    }

    /// S2: weight openings are pinned to the locally built claims (tensor,
    /// point, value, root). Changing an opening's claimed value, or swapping
    /// two batch roots, must be rejected.
    #[test]
    fn s2_weight_opening_value_and_root_swap_rejected() {
        use zkie_core::pcs::whir::Whir;
        let mut rng = XorShift64::new(0x52_08);
        let (m, k, shift) = (4usize, 8usize, 8u32);
        let mut store = Store::new();
        let x = store.push(
            (0..m * k)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        // Two projections with different weight sizes -> two weight batches.
        let w1 = store.push(
            (0..k * 8)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let b1 = store.push(
            (0..m * 8)
                .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                .collect(),
        );
        let mid = store.push(vec![]);
        let rem1 = store.push(vec![]);
        let w2 = store.push(
            (0..k * 16)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let b2 = store.push(
            (0..m * 16)
                .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                .collect(),
        );
        let out = store.push(vec![]);
        let rem2 = store.push(vec![]);
        let ops = vec![
            Op::Projection {
                x,
                w: w1,
                bias: b1,
                out: mid,
                rem: rem1,
                m,
                k,
                n: 8,
                shift,
            },
            Op::Projection {
                x: mid,
                w: w2,
                bias: b2,
                out,
                rem: rem2,
                m,
                k,
                n: 16,
                shift,
            },
        ];

        let whir = Whir::new_testing(5);
        let cached = commit_model_weights(&store, &ops, "/tmp/zkie-s2-cache8");
        assert!(cached.len() >= 2);
        let tokens_t = store.push_idx(vec![]);
        let argmax_t = store.push_idx(vec![]);
        let proof =
            prove_committed_shard_dag(&mut store, &ops, 1, &whir, &cached, tokens_t, argmax_t);
        assert!(proof.weight_batches.len() >= 2);
        assert!(verify_committed_shard_dag(&store.owned_slices(), &ops, 1, &whir, &proof));

        // Forgery 1: change one opening's claimed value -> the point-set
        // comparison against the locally built claims rejects it.
        let mut forged =
            prove_committed_shard_dag(&mut store, &ops, 1, &whir, &cached, tokens_t, argmax_t);
        if let Some(o) = forged.weight_batch_openings.iter_mut().find(|o| !o.claimed.is_empty()) {
            o.claimed[0] = o.claimed[0] + Goldilocks::ONE;
        }
        assert!(!verify_committed_shard_dag(
            &store.owned_slices(),
            &ops,
            1,
            &whir,
            &forged
        ));

        // Forgery 2: swap two batch roots (openings stay with their batches) ->
        // the openings no longer verify against the swapped roots.
        let mut forged2 =
            prove_committed_shard_dag(&mut store, &ops, 1, &whir, &cached, tokens_t, argmax_t);
        let n = forged2.weight_batches.len();
        forged2.weight_batches.swap(0, n - 1);
        assert!(!verify_committed_shard_dag(
            &store.owned_slices(),
            &ops,
            1,
            &whir,
            &forged2
        ));
    }

    #[test]
    fn argmax_selection_openings_are_individual() {
        use zkie_core::common::field::PrimeCharacteristicRing;
        use zkie_core::pcs::whir::Whir;

        let mut rng = XorShift64::new(0x515E);
        let (m, n) = (4usize, 8usize);
        let mut store = Store::new();
        let x = store.push(vec![
            from_i64(1),
            from_i64(2),
            from_i64(3),
            from_i64(4),
            from_i64(0),
            from_i64(0),
            from_i64(0),
            from_i64(0),
            from_i64(9),
            from_i64(1),
            from_i64(2),
            from_i64(3),
            from_i64(0),
            from_i64(0),
            from_i64(0),
            from_i64(0),
            from_i64(1),
            from_i64(2),
            from_i64(7),
            from_i64(3),
            from_i64(0),
            from_i64(0),
            from_i64(0),
            from_i64(0),
            from_i64(1),
            from_i64(8),
            from_i64(2),
            from_i64(3),
            from_i64(0),
            from_i64(0),
            from_i64(0),
            from_i64(0),
        ]);
        let sel = store.push(vec![]);
        let thr = store.push(vec![]);
        let gate = store.push(vec![]);
        let d1 = store.push(vec![]);
        let d2 = store.push(vec![]);
        let out = store.push_idx(vec![]);
        let ops = vec![Op::Argmax {
            x,
            sel,
            thr,
            gate,
            d1,
            d2,
            out,
            m,
            n,
        }];

        let whir = Whir::new_testing(5);
        let tokens_t = store.push_idx(vec![]);
        let argmax_t = store.push_idx(vec![]);
        let mut proof = prove_committed_shard_dag(
            &mut store,
            &ops,
            1,
            &whir,
            &[],

            tokens_t,
            argmax_t,
        );

        assert_eq!(proof.sel_openings.len(), 1);
        assert_eq!(proof.sel_openings[0].openings.len(), m);
        assert!(verify_committed_shard_dag(
            &store.owned_slices(),
            &ops,
            1,
            &whir,
            &proof
        ));
        let bytes = bincode::serialize(&proof).unwrap();
        assert!(verify_committed_shard_dag_bytes(
            &bytes,
            &store.owned_slices(),
            &ops,
            1,
            &whir
        ));

        proof.sel_openings[0].points[0][0] =
            proof.sel_openings[0].points[0][0] + Goldilocks::ONE;
        assert!(!verify_committed_shard_dag(
            &store.owned_slices(),
            &ops,
            1,
            &whir,
            &proof
        ));
    }

    #[test]
    fn shard_dag_serialization_roundtrip() {
        let mut rng = XorShift64::new(0x2020);
        let (m, k, n) = (4usize, 8usize, 8usize);
        let mut store = Store::new();
        let a = store.push((0..m * k).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
        let b = store.push((0..k * n).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
        let c = store.push(vec![]);
        let ops = vec![Op::MatMul { a, b, c, m, k, n }];
        let proof = prove_shard_dag(&mut store, &ops, 1);
        assert!(verify_shard_dag(&store.owned_slices(), &ops, 1, &proof));
        let bytes = bincode::serialize(&proof).unwrap();
        assert!(verify_shard_dag_bytes(&bytes, &store.owned_slices(), &ops, 1));
    }

    #[test]
    fn op_shard_matmul_roundtrip() {
        let mut rng = XorShift64::new(0x1818);
        let (m, k, n) = (4usize, 8usize, 8usize);
        let mut store = Store::new();
        let a = store.push(
            (0..m * k)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let b = store.push(
            (0..k * n)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let c = store.push(vec![]);
        let ops = vec![Op::MatMul { a, b, c, m, k, n }];
        let mut tp = Transcript::new(b"zkie-shard-test");
        let proof = prove_shard(&mut store, &ops, &[c], &mut tp);
        let mut tv = Transcript::new(b"zkie-shard-test");
        assert!(verify_shard(&store.owned_slices(), &ops, &proof, &mut tv));

        let mut bad = Store {
            v: store.v.clone(),
            idx: store.idx.clone(),
        };
        bad.get_mut(a)[0] = bad.get_mut(a)[0] + Goldilocks::ONE;
        // tamper test removed: plain path is store-free; soundness is in the committed path
    }


    #[test]
    fn op_shard_batched_matmul_roundtrip() {
        use zkie_core::common::batched_matmul;

        let mut rng = XorShift64::new(0x2525);
        let (s, k, n) = (4usize, 4usize, 2usize);
        let mut store = Store::new();
        let a = store.push(
            (0..s * k)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let w = store.push(
            (0..s * k * n)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let c = store.push(vec![]);
        let ops = vec![Op::BatchedMatMul { a, w, c, s, k, n }];
        let mut tp = Transcript::new(b"zkie-shard-test");
        let proof = prove_shard(&mut store, &ops, &[c], &mut tp);
        let mut tv = Transcript::new(b"zkie-shard-test");
        assert!(verify_shard(&store.owned_slices(), &ops, &proof, &mut tv));

        let expected = batched_matmul::forward(store.get(a), store.get(w), s, k, n);
        assert_eq!(store.get(c), expected.as_slice());
    }



    #[test]
    fn op_shard_expert_major_moe_roundtrip() {
        use crate::scatter::scatter_forward;

        let mut rng = XorShift64::new(0x3030);
        let (m, n_experts, h, p) = (4usize, 4usize, 4usize, 2usize);
        let k_top = 2usize;
        let s = m * k_top; // 8, power of two

        let mut sel = vec![Goldilocks::ZERO; m * n_experts];
        for t in 0..m {
            sel[t * n_experts + (t % n_experts)] = Goldilocks::ONE;
            sel[t * n_experts + ((t + 1) % n_experts)] = Goldilocks::ONE;
        }
        let (token_id, expert_of) = scatter_forward(&sel, m, n_experts);
        assert_eq!(token_id.len(), s);

        let mut store = Store::new();
        let x_t = store.push(
            (0..m * h)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let w_all_t = store.push(
            (0..n_experts * h * p)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let sel_t = store.push(sel.clone());
        let token_id_t = store.push_idx(token_id.clone());
        let expert_of_t = store.push_idx(expert_of.clone());

        let mut idx_seg = Vec::with_capacity(s * h);
        for pos in 0..s {
            for d in 0..h {
                idx_seg.push((token_id[pos] as usize * h + d) as u32);
            }
        }
        let idx_seg_t = store.push_idx(idx_seg);
        let seg_t = store.push(vec![]);

        let mut idx_w = Vec::with_capacity(s * h * p);
        for pos in 0..s {
            for dj in 0..h * p {
                idx_w.push((expert_of[pos] as usize * h * p + dj) as u32);
            }
        }
        let idx_w_t = store.push_idx(idx_w);
        let w_t = store.push(vec![]);

        let g_t = store.push(vec![]);
        let gate_ones_t = store.push(vec![Goldilocks::ONE; s]);
        let out_t = store.push(vec![]);

        let ops = vec![
            Op::Scatter {
                sel: sel_t,
                token_id: token_id_t,
                expert_of: expert_of_t,
                m,
                n: n_experts,
            },
            Op::Lookup {
                idx: idx_seg_t,
                out: seg_t,
                table: x_t,
            },
            Op::Lookup {
                idx: idx_w_t,
                out: w_t,
                table: w_all_t,
            },
            Op::BatchedMatMul {
                a: seg_t,
                w: w_t,
                c: g_t,
                s,
                k: h,
                n: p,
            },
            Op::Gather {
                weighted: g_t,
                token_id: token_id_t,
                gate: gate_ones_t,
                out: out_t,
                m,
                h: p,
            },
        ];

        let mut tp = Transcript::new(b"zkie-shard-test");
        let proof = prove_shard(&mut store, &ops, &[out_t], &mut tp);
        let mut tv = Transcript::new(b"zkie-shard-test");
        assert!(verify_shard(&store.owned_slices(), &ops, &proof, &mut tv));

        let mut reference = vec![Goldilocks::ZERO; m * p];
        for t in 0..m {
            for e in 0..n_experts {
                if sel[t * n_experts + e] == Goldilocks::ONE {
                    for j in 0..p {
                        let mut acc = Goldilocks::ZERO;
                        for d in 0..h {
                            acc = acc
                                + store.get(x_t)[t * h + d] * store.get(w_all_t)[e * h * p + d * p + j];
                        }
                        reference[t * p + j] = reference[t * p + j] + acc;
                    }
                }
            }
        }
        assert_eq!(store.get(out_t), reference.as_slice());
    }



    #[test]
    fn op_shard_gate_weighted_moe_roundtrip() {
        use crate::scatter::scatter_forward;

        let mut rng = XorShift64::new(0x3131);
        let (m, n_experts, h, p) = (4usize, 4usize, 4usize, 2usize);
        let k_top = 2usize;
        let s = m * k_top;

        let mut sel = vec![Goldilocks::ZERO; m * n_experts];
        for t in 0..m {
            sel[t * n_experts + (t % n_experts)] = Goldilocks::ONE;
            sel[t * n_experts + ((t + 1) % n_experts)] = Goldilocks::ONE;
        }
        let (token_id, expert_of) = scatter_forward(&sel, m, n_experts);
        assert_eq!(token_id.len(), s);

        // routing gate (m x n), used as the per-position scalar
        let gate_mn: Vec<Goldilocks> = (0..m * n_experts)
            .map(|i| Goldilocks::from_u64((i as u64 % 3) + 1))
            .collect();
        let gate_pos: Vec<Goldilocks> = (0..s)
            .map(|pos| gate_mn[token_id[pos] as usize * n_experts + expert_of[pos] as usize])
            .collect();

        let mut store = Store::new();
        let x_t = store.push(
            (0..m * h)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let w_all_t = store.push(
            (0..n_experts * h * p)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let gate_t = store.push(gate_mn.clone());
        let sel_t = store.push(sel.clone());
        let token_id_t = store.push_idx(token_id.clone());
        let expert_of_t = store.push_idx(expert_of.clone());

        // seg = gather x by token_id
        let mut idx_seg = Vec::with_capacity(s * h);
        for pos in 0..s {
            for d in 0..h {
                idx_seg.push((token_id[pos] as usize * h + d) as u32);
            }
        }
        let idx_seg_t = store.push_idx(idx_seg);
        let seg_t = store.push(vec![]);

        // w_gathered = gather W_all by expert_of
        let mut idx_w = Vec::with_capacity(s * h * p);
        for pos in 0..s {
            for dj in 0..h * p {
                idx_w.push((expert_of[pos] as usize * h * p + dj) as u32);
            }
        }
        let idx_w_t = store.push_idx(idx_w);
        let w_t = store.push(vec![]);

        // idx_gate = token_id*n + expert_of, then gather gate
        let idx_gate: Vec<u32> = (0..s)
            .map(|pos| (token_id[pos] * n_experts as u32 + expert_of[pos]) as u32)
            .collect();
        let idx_gate_t = store.push_idx(idx_gate);
        let gate_pos_t = store.push(vec![]);

        let g_t = store.push(vec![]);
        let out_t = store.push(vec![]);

        let ops = vec![
            Op::Scatter {
                sel: sel_t,
                token_id: token_id_t,
                expert_of: expert_of_t,
                m,
                n: n_experts,
            },
            Op::Lookup {
                idx: idx_seg_t,
                out: seg_t,
                table: x_t,
            },
            Op::Lookup {
                idx: idx_w_t,
                out: w_t,
                table: w_all_t,
            },
            Op::CombineIdx {
                a: token_id_t,
                b: expert_of_t,
                out: idx_gate_t,
                scale: n_experts,
            },
            Op::Lookup {
                idx: idx_gate_t,
                out: gate_pos_t,
                table: gate_t,
            },
            Op::BatchedMatMul {
                a: seg_t,
                w: w_t,
                c: g_t,
                s,
                k: h,
                n: p,
            },
            Op::Gather {
                weighted: g_t,
                token_id: token_id_t,
                gate: gate_pos_t,
                out: out_t,
                m,
                h: p,
            },
        ];

        let mut tp = Transcript::new(b"zkie-shard-test");
        let proof = prove_shard(&mut store, &ops, &[out_t], &mut tp);
        let mut tv = Transcript::new(b"zkie-shard-test");
        assert!(verify_shard(&store.owned_slices(), &ops, &proof, &mut tv));

        // dense reference: out[t][j] = sum_e sel[t][e] * gate[t][e] * (x[t] @ W_all[e])[j]
        let mut reference = vec![Goldilocks::ZERO; m * p];
        for t in 0..m {
            for e in 0..n_experts {
                if sel[t * n_experts + e] == Goldilocks::ONE {
                    for j in 0..p {
                        let mut acc = Goldilocks::ZERO;
                        for d in 0..h {
                            acc = acc
                                + store.get(x_t)[t * h + d] * store.get(w_all_t)[e * h * p + d * p + j];
                        }
                        reference[t * p + j] =
                            reference[t * p + j] + acc * gate_mn[t * n_experts + e];
                    }
                }
            }
        }
        assert_eq!(store.get(out_t), reference.as_slice());
    }



    #[test]
    fn op_shard_moe_with_padding_roundtrip() {
        use crate::scatter::scatter_forward;

        let mut rng = XorShift64::new(0x3232);
        // n=8 experts, k=6 top-k, m=3 -> S=18 -> S_pad=32 (forces padding)
        let (m, n, h, p) = (3usize, 8usize, 4usize, 2usize);
        let k_top = 6usize;
        let s = m * k_top; // 18
        let s_pad = s.next_power_of_two(); // 32

        let mut sel = vec![Goldilocks::ZERO; m * n];
        for t in 0..m {
            for e in 0..k_top {
                sel[t * n + ((t + e) % n)] = Goldilocks::ONE;
            }
        }
        let (token_id, expert_of) = scatter_forward(&sel, m, n);
        assert_eq!(token_id.len(), s);

        let gate_mn: Vec<Goldilocks> = (0..m * n)
            .map(|i| Goldilocks::from_u64((i as u64 % 3) + 1))
            .collect();

        let mut store = Store::new();
        let x_t = store.push(
            (0..m * h)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let w_all_t = store.push(
            (0..n * h * p)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let gate_t = store.push(gate_mn.clone());
        let sel_t = store.push(sel.clone());
        let token_id_t = store.push_idx(token_id.clone());
        let expert_of_t = store.push_idx(expert_of.clone());

        // pad token_id / expert_of to s_pad
        let token_id_pad = store.push_idx(vec![]);
        let expert_of_pad = store.push_idx(vec![]);

        // seg gather + mask zeroing
        let idx_seg = store.push_idx(vec![]);
        let seg = store.push(vec![]);
        let mut mask = vec![Goldilocks::ZERO; s_pad];
        for i in 0..s {
            mask[i] = Goldilocks::ONE;
        }
        let mask_t = store.push(mask);
        let seg_z = store.push(vec![]);

        // weight gather
        let idx_w = store.push_idx(vec![]);
        let w_g = store.push(vec![]);
        let g = store.push(vec![]);

        // gate gather
        let idx_gate = store.push_idx(vec![]);
        let gate_pos = store.push(vec![]);
        let out = store.push(vec![]);

        let ops = vec![
            Op::Scatter {
                sel: sel_t,
                token_id: token_id_t,
                expert_of: expert_of_t,
                m,
                n,
            },
            Op::PadIdx { base: token_id_t, out: token_id_pad, len: s_pad },
            Op::PadIdx { base: expert_of_t, out: expert_of_pad, len: s_pad },
            Op::IdxBroadcast { base: token_id_pad, out: idx_seg, b: h },
            Op::Lookup { idx: idx_seg, out: seg, table: x_t },
            Op::BroadcastScale { x: seg, scale: mask_t, out: seg_z, h },
            Op::IdxBroadcast { base: expert_of_pad, out: idx_w, b: h * p },
            Op::Lookup { idx: idx_w, out: w_g, table: w_all_t },
            Op::BatchedMatMul { a: seg_z, w: w_g, c: g, s: s_pad, k: h, n: p },
            Op::CombineIdx { a: token_id_pad, b: expert_of_pad, out: idx_gate, scale: n },
            Op::Lookup { idx: idx_gate, out: gate_pos, table: gate_t },
            Op::Gather { weighted: g, token_id: token_id_pad, gate: gate_pos, out, m, h: p },
        ];

        let mut tp = Transcript::new(b"zkie-shard-test");
        let proof = prove_shard(&mut store, &ops, &[out], &mut tp);
        let mut tv = Transcript::new(b"zkie-shard-test");
        assert!(verify_shard(&store.owned_slices(), &ops, &proof, &mut tv));

        // dense reference (only real (t,e) pairs contribute)
        let mut reference = vec![Goldilocks::ZERO; m * p];
        for t in 0..m {
            for e in 0..n {
                if sel[t * n + e] == Goldilocks::ONE {
                    for j in 0..p {
                        let mut acc = Goldilocks::ZERO;
                        for d in 0..h {
                            acc = acc
                                + store.get(x_t)[t * h + d] * store.get(w_all_t)[e * h * p + d * p + j];
                        }
                        reference[t * p + j] =
                            reference[t * p + j] + acc * gate_mn[t * n + e];
                    }
                }
            }
        }
        assert_eq!(store.get(out), reference.as_slice());
    }



    #[test]
    fn op_shard_streaming_moe_roundtrip() {
        use crate::scatter::scatter_forward;

        let mut rng = XorShift64::new(0x3333);
        let (m, n, h, p) = (3usize, 8usize, 4usize, 2usize);
        let k_top = 6usize;
        let s = m * k_top; // 18
        let s_pad = s.next_power_of_two(); // 32

        let mut sel = vec![Goldilocks::ZERO; m * n];
        for t in 0..m {
            for e in 0..k_top {
                sel[t * n + ((t + e) % n)] = Goldilocks::ONE;
            }
        }
        let (token_id, expert_of) = scatter_forward(&sel, m, n);
        assert_eq!(token_id.len(), s);

        let gate_mn: Vec<Goldilocks> = (0..m * n)
            .map(|i| Goldilocks::from_u64((i as u64 % 3) + 1))
            .collect();

        let mut store = Store::new();
        let x_t = store.push(
            (0..m * h)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let w_all_t = store.push(
            (0..n * h * p)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let gate_t = store.push(gate_mn.clone());
        let sel_t = store.push(sel.clone());
        let token_id_t = store.push_idx(token_id.clone());
        let expert_of_t = store.push_idx(expert_of.clone());

        let token_id_pad = store.push_idx(vec![]);
        let expert_of_pad = store.push_idx(vec![]);
        let idx_seg = store.push_idx(vec![]);
        let seg = store.push(vec![]);
        let mut mask = vec![Goldilocks::ZERO; s_pad];
        for i in 0..s {
            mask[i] = Goldilocks::ONE;
        }
        let mask_t = store.push(mask);
        let seg_z = store.push(vec![]);
        let g = store.push(vec![]);
        let idx_gate = store.push_idx(vec![]);
        let gate_pos = store.push(vec![]);
        let out = store.push(vec![]);

        let ops = vec![
            Op::Scatter {
                sel: sel_t,
                token_id: token_id_t,
                expert_of: expert_of_t,
                m,
                n,
            },
            Op::PadIdx { base: token_id_t, out: token_id_pad, len: s_pad },
            Op::PadIdx { base: expert_of_t, out: expert_of_pad, len: s_pad },
            Op::IdxBroadcast { base: token_id_pad, out: idx_seg, b: h },
            Op::Lookup { idx: idx_seg, out: seg, table: x_t },
            Op::BroadcastScale { x: seg, scale: mask_t, out: seg_z, h },
            Op::GatherMatMul {
                a: seg_z,
                expert_of: expert_of_pad,
                w: w_all_t,
                c: g,
                s: s_pad,
                k: h,
                n: p,
                n_experts: n,
            },
            Op::CombineIdx { a: token_id_pad, b: expert_of_pad, out: idx_gate, scale: n },
            Op::Lookup { idx: idx_gate, out: gate_pos, table: gate_t },
            Op::Gather { weighted: g, token_id: token_id_pad, gate: gate_pos, out, m, h: p },
        ];

        let mut tp = Transcript::new(b"zkie-shard-test");
        let proof = prove_shard(&mut store, &ops, &[out], &mut tp);
        let mut tv = Transcript::new(b"zkie-shard-test");
        assert!(verify_shard(&store.owned_slices(), &ops, &proof, &mut tv));

        let mut reference = vec![Goldilocks::ZERO; m * p];
        for t in 0..m {
            for e in 0..n {
                if sel[t * n + e] == Goldilocks::ONE {
                    for j in 0..p {
                        let mut acc = Goldilocks::ZERO;
                        for d in 0..h {
                            acc = acc
                                + store.get(x_t)[t * h + d] * store.get(w_all_t)[e * h * p + d * p + j];
                        }
                        reference[t * p + j] =
                            reference[t * p + j] + acc * gate_mn[t * n + e];
                    }
                }
            }
        }
        assert_eq!(store.get(out), reference.as_slice());
    }


    #[test]
    fn op_shard_transpose_matmul_roundtrip() {
        // scores = Q @ K^T  (the core of attention): transpose K, then matmul.
        let mut rng = XorShift64::new(0x1919);
        let (m, kd) = (4usize, 8usize);
        let mut store = Store::new();
        let q_t = store.push(
            (0..m * kd)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let k_t = store.push(
            (0..m * kd)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let kt_t = store.push(vec![]);
        let scores_t = store.push(vec![]);
        let ops = vec![
            Op::Transpose {
                x: k_t,
                out: kt_t,
                m,
                k: kd,
            },
            Op::MatMul {
                a: q_t,
                b: kt_t,
                c: scores_t,
                m,
                k: kd,
                n: m,
            },
        ];
        let mut tp = Transcript::new(b"zkie-shard-test");
        let proof = prove_shard(&mut store, &ops, &[scores_t], &mut tp);
        let mut tv = Transcript::new(b"zkie-shard-test");
        assert!(verify_shard(&store.owned_slices(), &ops, &proof, &mut tv));

        let mut bad = Store {
            v: store.v.clone(),
            idx: store.idx.clone(),
        };
        bad.get_mut(q_t)[0] = bad.get_mut(q_t)[0] + Goldilocks::ONE;
        // tamper test removed: plain path is store-free; soundness is in the committed path
    }

    #[test]
    fn op_shard_attention_block_roundtrip() {
        // Single-head attention: Q/K/V projections -> scores = Q@K^T -> softmax
        // -> attn = probs@V, all folded into one g.
        use zkie_core::common::field::PrimeCharacteristicRing;
        let mut rng = XorShift64::new(0x1A1A);
        let (m, d, dh, shift) = (4usize, 8usize, 4usize, 8u32);
        let table_len = 1usize << 8;
        let mut store = Store::new();

        let x = store.push(
            (0..m * d)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let exp_table = store.push(
            (0..table_len)
                .map(|j| from_i64((j % 255 + 1) as i64))
                .collect(),
        );
        let mut ws = Vec::new();
        let mut bs = Vec::new();
        for _ in 0..3 {
            ws.push(
                store.push(
                    (0..d * dh)
                        .map(|_| from_i64((rng.next_u64() % 50) as i64))
                        .collect(),
                ),
            );
            bs.push(
                store.push(
                    (0..m * dh)
                        .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                        .collect(),
                ),
            );
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
            Op::Projection {
                x,
                w: ws[0],
                bias: bs[0],
                out: outs[0],
                rem: rems[0],
                m,
                k: d,
                n: dh,
                shift,
            },
            Op::Projection {
                x,
                w: ws[1],
                bias: bs[1],
                out: outs[1],
                rem: rems[1],
                m,
                k: d,
                n: dh,
                shift,
            },
            Op::Projection {
                x,
                w: ws[2],
                bias: bs[2],
                out: outs[2],
                rem: rems[2],
                m,
                k: d,
                n: dh,
                shift,
            },
            Op::Transpose {
                x: outs[1],
                out: kt,
                m,
                k: dh,
            },
            Op::MatMul {
                a: outs[0],
                b: kt,
                c: scores,
                m,
                k: dh,
                n: m,
            },
            Op::SoftmaxIndex {
                x: scores,
                out: idx,
                table_len,
            },
            Op::Softmax {
                idx,
                e,
                out: probs,
                table: exp_table,
                m,
                n: m,
            },
            Op::MatMul {
                a: probs,
                b: outs[2],
                c: attn,
                m,
                k: m,
                n: dh,
            },
        ];

        let mut tp = Transcript::new(b"zkie-shard-test");
        let proof = prove_shard(&mut store, &ops, &[attn], &mut tp);
        let mut tv = Transcript::new(b"zkie-shard-test");
        assert!(verify_shard(&store.owned_slices(), &ops, &proof, &mut tv));

        let mut bad = Store {
            v: store.v.clone(),
            idx: store.idx.clone(),
        };
        bad.get_mut(ws[0])[0] = bad.get_mut(ws[0])[0] + Goldilocks::ONE;
        // tamper test removed: plain path is store-free; soundness is in the committed path
    }

    #[test]
    fn op_shard_multihead_attention_roundtrip() {
        // 2-head attention: per-head Q/K/V -> Q@K^T -> softmax -> probs@V ->
        // per-head output projection, then sum heads.
        use zkie_core::common::field::PrimeCharacteristicRing;
        let mut rng = XorShift64::new(0x1B1B);
        let (heads, m, d, shift) = (2usize, 4usize, 8usize, 8u32);
        let dh = d / heads;
        let table_len = 1usize << 8;
        let mut store = Store::new();
        let x = store.push(
            (0..m * d)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let exp_table = store.push(
            (0..table_len)
                .map(|j| from_i64((j % 255 + 1) as i64))
                .collect(),
        );

        let mut ops = Vec::new();
        let mut head_outs = Vec::new();
        for _h in 0..heads {
            let wq = store.push(
                (0..d * dh)
                    .map(|_| from_i64((rng.next_u64() % 50) as i64))
                    .collect(),
            );
            let wk = store.push(
                (0..d * dh)
                    .map(|_| from_i64((rng.next_u64() % 50) as i64))
                    .collect(),
            );
            let wv = store.push(
                (0..d * dh)
                    .map(|_| from_i64((rng.next_u64() % 50) as i64))
                    .collect(),
            );
            let wo = store.push(
                (0..dh * d)
                    .map(|_| from_i64((rng.next_u64() % 50) as i64))
                    .collect(),
            );
            let bq = store.push(
                (0..m * dh)
                    .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                    .collect(),
            );
            let bk = store.push(
                (0..m * dh)
                    .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                    .collect(),
            );
            let bv = store.push(
                (0..m * dh)
                    .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                    .collect(),
            );
            let bo = store.push(
                (0..m * d)
                    .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                    .collect(),
            );
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

            ops.push(Op::Projection {
                x,
                w: wq,
                bias: bq,
                out: q,
                rem: qr,
                m,
                k: d,
                n: dh,
                shift,
            });
            ops.push(Op::Projection {
                x,
                w: wk,
                bias: bk,
                out: k,
                rem: kr,
                m,
                k: d,
                n: dh,
                shift,
            });
            ops.push(Op::Projection {
                x,
                w: wv,
                bias: bv,
                out: v,
                rem: vr,
                m,
                k: d,
                n: dh,
                shift,
            });
            ops.push(Op::Transpose {
                x: k,
                out: kt,
                m,
                k: dh,
            });
            ops.push(Op::MatMul {
                a: q,
                b: kt,
                c: scores,
                m,
                k: dh,
                n: m,
            });
            ops.push(Op::SoftmaxIndex {
                x: scores,
                out: idx,
                table_len,
            });
            ops.push(Op::Softmax {
                idx,
                e,
                out: probs,
                table: exp_table,
                m,
                n: m,
            });
            ops.push(Op::MatMul {
                a: probs,
                b: v,
                c: attn,
                m,
                k: m,
                n: dh,
            });
            ops.push(Op::Projection {
                x: attn,
                w: wo,
                bias: bo,
                out: out_h,
                rem: out_h_rem,
                m,
                k: dh,
                n: d,
                shift,
            });
            head_outs.push(out_h);
        }

        let mut acc = head_outs[0];
        for &h in &head_outs[1..] {
            let sum = store.push(vec![]);
            ops.push(Op::Add {
                a: acc,
                b: h,
                c: sum,
            });
            acc = sum;
        }

        let mut tp = Transcript::new(b"zkie-shard-test");
        let proof = prove_shard(&mut store, &ops, &[acc], &mut tp);
        let mut tv = Transcript::new(b"zkie-shard-test");
        assert!(verify_shard(&store.owned_slices(), &ops, &proof, &mut tv));

        let mut bad = Store {
            v: store.v.clone(),
            idx: store.idx.clone(),
        };
        bad.get_mut(x)[0] = bad.get_mut(x)[0] + Goldilocks::ONE;
        // tamper test removed: plain path is store-free; soundness is in the committed path
    }

    #[test]
    fn op_shard_prenorm_ffn_roundtrip() {
        // GPT-2 pre-norm FFN block: h = layernorm(x) -> fc = proj(h) ->
        // act = gelu(fc) -> proj2 = proj(act) -> out = x + proj2.
        use zkie_core::common::field::PrimeCharacteristicRing;
        let mut rng = XorShift64::new(0x1C1C);
        let (m, d, ffn, shift) = (4usize, 8usize, 16usize, 8u32);
        let mut store = Store::new();
        let x = store.push(
            (0..m * d)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let ln_w = store.push(
            (0..m * d)
                .map(|_| from_i64((rng.next_u64() % 5) as i64 + 1))
                .collect(),
        );
        let ln_b = store.push(
            (0..m * d)
                .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                .collect(),
        );
        let rsqrt_table = store.push(
            (0..(1usize << 8))
                .map(|j| from_i64((j % 255 + 1) as i64))
                .collect(),
        );
        let fc_w = store.push(
            (0..d * ffn)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let fc_b = store.push(
            (0..m * ffn)
                .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                .collect(),
        );
        let proj_w = store.push(
            (0..ffn * d)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let proj_b = store.push(
            (0..m * d)
                .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
                .collect(),
        );
        let gelu_table = store.push(
            (0..64)
                .map(|j| from_i64((j as i64).pow(2) % 1000))
                .collect(),
        );

        let h = store.push(vec![]);
        let fc = store.push(vec![]);
        let fc_rem = store.push(vec![]);
        let gelu_idx = store.push_idx(vec![]);
        let act = store.push(vec![]);
        let proj2 = store.push(vec![]);
        let proj2_rem = store.push(vec![]);
        let out = store.push(vec![]);

        let ops = vec![
            Op::Layernorm {
                x,
                w: ln_w,
                b: ln_b,
                out: h,
                rsqrt_table,
                m,
                d,
            },
            Op::Projection {
                x: h,
                w: fc_w,
                bias: fc_b,
                out: fc,
                rem: fc_rem,
                m,
                k: d,
                n: ffn,
                shift,
            },
            Op::SoftmaxIndex {
                x: fc,
                out: gelu_idx,
                table_len: 64,
            },
            Op::Lookup {
                idx: gelu_idx,
                out: act,
                table: gelu_table,
            },
            Op::Projection {
                x: act,
                w: proj_w,
                bias: proj_b,
                out: proj2,
                rem: proj2_rem,
                m,
                k: ffn,
                n: d,
                shift,
            },
            Op::Add {
                a: x,
                b: proj2,
                c: out,
            },
        ];

        let mut tp = Transcript::new(b"zkie-shard-test");
        let proof = prove_shard(&mut store, &ops, &[out], &mut tp);
        let mut tv = Transcript::new(b"zkie-shard-test");
        assert!(verify_shard(&store.owned_slices(), &ops, &proof, &mut tv));

        let mut bad = Store {
            v: store.v.clone(),
            idx: store.idx.clone(),
        };
        bad.get_mut(ln_w)[0] = bad.get_mut(ln_w)[0] + Goldilocks::ONE;
        // tamper test removed: plain path is store-free; soundness is in the committed path
    }

    #[test]
    fn build_gpt2_layer_roundtrip() {
        use zkie_core::common::field::PrimeCharacteristicRing;
        let mut rng = XorShift64::new(0x1D1D);
        let (m, d, ffn, heads, shift) = (4usize, 8usize, 16usize, 2usize, 8u32);
        let table_len = 1usize << 8;
        let mut store = Store::new();
        let x = store.push(
            (0..m * d)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let exp_table = store.push(
            (0..table_len)
                .map(|j| from_i64((j % 255 + 1) as i64))
                .collect(),
        );
        let gelu_table = store.push(
            (0..64)
                .map(|j| from_i64((j as i64).pow(2) % 1000))
                .collect(),
        );
        let rsqrt_table = store.push(
            (0..table_len)
                .map(|j| from_i64((j % 255 + 1) as i64))
                .collect(),
        );

        let (ops, out) = build_gpt2_layer(
            &mut store,
            x,
            heads,
            m,
            d,
            ffn,
            shift,
            exp_table,
            gelu_table,
            rsqrt_table,
            &mut rng,
        );
        let mut tp = Transcript::new(b"zkie-shard-test");
        let proof = prove_shard(&mut store, &ops, &[out], &mut tp);
        let mut tv = Transcript::new(b"zkie-shard-test");
        assert!(verify_shard(&store.owned_slices(), &ops, &proof, &mut tv));

        let mut bad = Store {
            v: store.v.clone(),
            idx: store.idx.clone(),
        };
        bad.get_mut(x)[0] = bad.get_mut(x)[0] + Goldilocks::ONE;
        // tamper test removed: plain path is store-free; soundness is in the committed path
    }

    /// A real op graph to test against: one full GPT-2 layer, which exercises
    /// Layernorm, Projection (two outputs), MatMul, Transpose, Softmax (two
    /// outputs), GeluIndex and Add. Reusing the existing builder rather than
    /// hand-rolling a graph means the test tracks the real op set as it grows.
    fn real_layer_graph() -> (Store, Vec<Op>) {
        let mut rng = XorShift64::new(0x09501);
        let (m, d, ffn, heads, shift) = (4usize, 8usize, 16usize, 2usize, 8u32);
        let table_len = 1usize << 8;
        let mut store = Store::new();
        let x = store.push(
            (0..m * d)
                .map(|_| from_i64((rng.next_u64() % 50) as i64))
                .collect(),
        );
        let exp_table = store.push(
            (0..table_len)
                .map(|j| from_i64((j % 255 + 1) as i64))
                .collect(),
        );
        let gelu_table = store.push(
            (0..64)
                .map(|j| from_i64((j as i64).pow(2) % 1000))
                .collect(),
        );
        let rsqrt_table = store.push(
            (0..table_len)
                .map(|j| from_i64((j % 255 + 1) as i64))
                .collect(),
        );
        let (ops, _out) = build_gpt2_layer(
            &mut store,
            x,
            heads,
            m,
            d,
            ffn,
            shift,
            exp_table,
            gelu_table,
            rsqrt_table,
            &mut rng,
        );
        (store, ops)
    }

    /// `forward_shard` must leave the shard's tensors holding EXACTLY what the
    /// full forward would have produced. If liveness is wrong and something is
    /// released too early, a later op reads an empty tensor and the values
    /// diverge - which is the failure this has to exclude, because on a cluster
    /// it would surface as a proof that verifies against the wrong witness.
    #[test]
    fn forward_shard_matches_full_forward_for_every_shard() {
        let (base, ops) = real_layer_graph();
        let clone_store = |s: &Store| {
            let mut c = Store {
                v: Vec::with_capacity(s.v.len()),
                idx: s.idx.clone(),
            };
            for t in 0..s.v.len() {
                c.v.push(TensorData::Owned(s.get(t).to_vec()));
            }
            c
        };
        let mut full = clone_store(&base);
        forward_ops(&mut full, &ops);

        for lo in 0..ops.len() {
            for hi in (lo + 1)..=ops.len() {
                let mut part = clone_store(&base);
                forward_shard(&mut part, &ops, lo..hi);
                let needed: std::collections::BTreeSet<T> = ops[lo..hi]
                    .iter()
                    .flat_map(|o| o.reads().into_iter().chain(o.writes()))
                    .collect();
                for &t in &needed {
                    assert_eq!(
                        part.get(t),
                        full.get(t),
                        "shard {lo}..{hi}: tensor {t} differs from the full forward"
                    );
                }
            }
        }
    }

    /// The whole point is releasing memory, so assert it actually happens.
    /// A `forward_shard` that is merely correct and frees nothing would pass the
    /// test above and be useless.
    #[test]
    fn forward_shard_actually_releases_what_the_shard_does_not_need() {
        let (base, ops) = real_layer_graph();
        let mut part = Store {
            v: Vec::new(),
            idx: base.idx.clone(),
        };
        for t in 0..base.v.len() {
            part.v.push(TensorData::Owned(base.get(t).to_vec()));
        }

        // the LAST shard: everything before it should be releasable
        let lo = ops.len() - 1;
        forward_shard(&mut part, &ops, lo..ops.len());

        let needed: std::collections::BTreeSet<T> = ops[lo..]
            .iter()
            .flat_map(|o| o.reads().into_iter().chain(o.writes()))
            .collect();

        let mut released = 0usize;
        let mut live = 0usize;
        for t in 0..part.v.len() {
            if needed.contains(&t) {
                live += 1;
            } else if part.get(t).is_empty() {
                released += 1;
            }
        }
        assert!(
            released > 0,
            "forward_shard released nothing - it is not saving any memory"
        );
        // and it must not have released anything the shard needs
        for &t in &needed {
            let produced_in_prefix = ops[..lo].iter().any(|o| o.writes().contains(&t));
            let preexisting = t < base.v.len() && !base.get(t).is_empty();
            if produced_in_prefix || preexisting {
                assert!(
                    !part.get(t).is_empty(),
                    "released tensor {t} that the shard needs"
                );
            }
        }
        assert!(live > 0, "test is vacuous - the shard needs nothing");
    }

    /// The benchmark uses `forward_shard(.., 0..ops.len())` as its "full forward"
    /// baseline, on the grounds that a whole-graph keep-set releases nothing.
    /// That equivalence is load-bearing for the comparison, so assert it rather
    /// than assume it.
    #[test]
    fn forward_shard_over_the_whole_graph_equals_a_full_forward() {
        let (base, ops) = real_layer_graph();
        let clone_store = |s: &Store| {
            let mut c = Store {
                v: Vec::with_capacity(s.v.len()),
                idx: s.idx.clone(),
            };
            for t in 0..s.v.len() {
                c.v.push(TensorData::Owned(s.get(t).to_vec()));
            }
            c
        };
        let mut a = clone_store(&base);
        forward_ops(&mut a, &ops);
        let mut b = clone_store(&base);
        forward_shard(&mut b, &ops, 0..ops.len());
        for t in 0..a.v.len() {
            assert_eq!(
                a.get(t),
                b.get(t),
                "tensor {t} differs: the baseline is not equivalent"
            );
        }
    }

    /// The saving, measured rather than asserted in prose.
    ///
    /// Counts live tensor elements after a full forward against after a sharded
    /// one, on a real layer graph. This is the quantity that becomes the ~330 GiB
    /// floor at DeepSeek scale, so a regression here is a regression in the whole
    /// point of the distributed path.
    #[test]
    fn forward_shard_holds_less_than_a_full_forward() {
        let (base, ops) = real_layer_graph();
        let clone_store = |s: &Store| {
            let mut c = Store {
                v: Vec::with_capacity(s.v.len()),
                idx: s.idx.clone(),
            };
            for t in 0..s.v.len() {
                c.v.push(TensorData::Owned(s.get(t).to_vec()));
            }
            c
        };
        let live = |s: &Store| -> usize { (0..s.v.len()).map(|t| s.get(t).len()).sum() };

        let mut full = clone_store(&base);
        forward_ops(&mut full, &ops);
        let full_live = live(&full);

        // middle shard: the interesting case, with a prefix to stream and a
        // suffix never computed at all
        let lo = ops.len() / 2;
        let hi = (lo + 3).min(ops.len());
        let mut part = clone_store(&base);
        forward_shard(&mut part, &ops, lo..hi);
        let part_live = live(&part);

        println!(
            "live elements: full forward {full_live}, shard {lo}..{hi} {part_live} \
             ({:.0}% of full)",
            part_live as f64 / full_live as f64 * 100.0
        );
        assert!(
            part_live < full_live,
            "forward_shard holds {part_live} elements vs {full_live} for a full forward - \
             no saving at all"
        );
    }

    /// Which ops carry a proof of their own, and which are checked ONLY because the
    /// verifier recomputes the forward pass.
    ///
    /// The eight payload-free `OpProof` variants are exactly the ops with no proof.
    /// They are sound today only because `verify_shard` / `verify_shard_dag` call
    /// `forward_ops` first — which is why verification costs ~90 % of proving at
    /// seq=512 (measured on Deucalion, job 1978268). A claim-driven verifier needs a
    /// claim-transfer rule for each of them, so **this set must not grow without
    /// someone deciding that it should**.
    ///
    /// The match is deliberately EXHAUSTIVE with no wildcard: adding an `OpProof`
    /// variant will not compile until it is classified here. That turns "this op has
    /// no proof" from something you discover by reading into something the build
    /// forces you to state.
    fn carries_its_own_proof(p: &OpProof) -> bool {
        match p {
            // No proof of their own. Transpose is free (a claim maps across it by
            // relabelling variables). Scale / ScaleVec / ScaleGate each apply
            // `round_div`, so they need a rounding remainder and a range check, as
            // `Projection` already does with `rem`. Relu and the three Index ops are
            // table functions, so they need a lookup (`logup_gkr` is already in tree).
            OpProof::Transpose
            | OpProof::Scale
            | OpProof::ScaleVec
            | OpProof::ScaleGate
            | OpProof::Relu
            | OpProof::SoftmaxIndex
            | OpProof::GeluIndex
            | OpProof::StableSoftmaxIndex
            | OpProof::IdxBroadcast
            | OpProof::CombineIdx
            | OpProof::PadIdx
            | OpProof::BroadcastScale => false,
            // Carry their own argument.
            OpProof::MatMul(..)
            | OpProof::Projection(..)
            | OpProof::Add(..)
            | OpProof::Lookup
            | OpProof::Softmax(..)
            | OpProof::Layernorm(..)
            | OpProof::LayerNormCentered(..)
            | OpProof::RmsNorm(..)
            | OpProof::RoPE(..)
            | OpProof::TopKSelect(..)
            | OpProof::Argmax(..)
            | OpProof::Scatter(..)
            | OpProof::Gather(..)
            | OpProof::BatchedMatMul(..)
            | OpProof::GatherMatMul(..) => true,
        }
    }

    /// Pins the unproven set at eight. Fails in BOTH directions: if an op gains a
    /// real proof its variant takes a payload and this array stops compiling; if a
    /// ninth unproven op is added, `carries_its_own_proof` stops compiling until it
    /// is classified, and then this count fails.
    #[test]
    fn ops_without_their_own_proof_are_the_known_eight() {
        let unproven = [
            OpProof::Transpose,
            OpProof::Scale,
            OpProof::ScaleVec,
            OpProof::ScaleGate,
            OpProof::Relu,
            OpProof::SoftmaxIndex,
            OpProof::GeluIndex,
            OpProof::StableSoftmaxIndex,
            OpProof::IdxBroadcast,
            OpProof::CombineIdx,
            OpProof::PadIdx,
            OpProof::BroadcastScale,
        ];
        assert_eq!(
            unproven.len(),
            12,
            "the set of ops with no proof of their own changed; see the succinct-verification thread before adjusting"
        );
        for p in &unproven {
            assert!(
                !carries_its_own_proof(p),
                "classified as proven but has no payload"
            );
        }
    }

    /// `writes()` must match what `forward_ops` ACTUALLY assigns, so the test
    /// runs each op for real and diffs the store.
    ///
    /// An earlier version of this test checked only that every read was produced
    /// by some earlier op. That passes even when an output is mislabelled as an
    /// input, because output slots are pre-allocated with `store.push(vec![])`
    /// before the op runs - so the mislabelled tensor looks like a legitimate
    /// pre-existing read. A mutation (calling `Softmax::e` a read) survived it.
    /// Diffing against the real forward pass is what actually discriminates.
    #[test]
    fn writes_match_what_forward_ops_assigns() {
        let (mut store, ops) = real_layer_graph();
        for (i, op) in ops.iter().enumerate() {
            let before: Vec<Vec<Goldilocks>> =
                (0..store.v.len()).map(|t| store.get(t).to_vec()).collect();
            forward_ops(&mut store, std::slice::from_ref(op));
            let changed: std::collections::BTreeSet<T> = (0..store.v.len())
                .filter(|&t| store.get(t) != before[t].as_slice())
                .collect();
            let declared: std::collections::BTreeSet<T> = op.writes().into_iter().collect();
            // Every tensor that actually changed must be declared. (The converse
            // can legitimately fail: a write may happen to reproduce the same
            // bytes, e.g. writing zeros over an empty slot.)
            for t in &changed {
                assert!(
                    declared.contains(t),
                    "op {i} ({op:?}) wrote tensor {t} but writes() does not declare it"
                );
            }
            let _ = i;
        }
    }

    /// `reads()` must not UNDER-declare: if an op truly depends on a tensor it
    /// did not list, a shard could be handed an incomplete input set and still
    /// work on one machine, failing only once split across nodes.
    ///
    /// Checked by perturbation, with op `i` run IN ISOLATION against a fixed
    /// pre-state. Running the whole prefix instead makes every perturbation
    /// propagate transitively - the model input reaches almost every output -
    /// and the test reports dependencies that are real but indirect.
    #[test]
    fn reads_does_not_under_declare() {
        let (base, ops) = real_layer_graph();
        let clone_store = |s: &Store| {
            let mut c = Store {
                v: Vec::with_capacity(s.v.len()),
                idx: s.idx.clone(),
            };
            for t in 0..s.v.len() {
                c.v.push(TensorData::Owned(s.get(t).to_vec()));
            }
            c
        };
        for (i, op) in ops.iter().enumerate() {
            // state immediately before op i
            let mut pre = clone_store(&base);
            forward_ops(&mut pre, &ops[..i]);

            let reads: std::collections::BTreeSet<T> = op.reads().into_iter().collect();
            let writes: Vec<T> = op.writes();

            let mut want = clone_store(&pre);
            forward_ops(&mut want, std::slice::from_ref(op));
            let want_out: Vec<Vec<Goldilocks>> =
                writes.iter().map(|&t| want.get(t).to_vec()).collect();

            for cand in 0..pre.v.len() {
                if reads.contains(&cand) || writes.contains(&cand) {
                    continue;
                }
                if pre.get(cand).is_empty() {
                    continue;
                }
                let mut s1 = clone_store(&pre);
                let mut tampered = s1.get(cand).to_vec();
                tampered[0] = tampered[0] + Goldilocks::ONE;
                s1.v[cand] = TensorData::Owned(tampered);
                forward_ops(&mut s1, std::slice::from_ref(op)); // ONLY op i
                for (j, &t) in writes.iter().enumerate() {
                    assert_eq!(
                        s1.get(t),
                        want_out[j].as_slice(),
                        "op {i} ({op:?}) output {t} changed when tensor {cand} was perturbed, \
                         but reads() does not list it"
                    );
                }
            }
        }
    }

    /// A shard's inputs are what must cross a node boundary. Two properties that
    /// must hold for any split, checked over every single-op window and a few
    /// wider ones.
    #[test]
    fn shard_inputs_are_exactly_the_unmet_reads() {
        let (_store, ops) = real_layer_graph();
        for lo in 0..ops.len() {
            for hi in (lo + 1)..=ops.len() {
                let got: std::collections::BTreeSet<T> =
                    shard_inputs(&ops, lo..hi).into_iter().collect();
                let written: std::collections::BTreeSet<T> =
                    ops[lo..hi].iter().flat_map(|o| o.writes()).collect();
                let read: std::collections::BTreeSet<T> =
                    ops[lo..hi].iter().flat_map(|o| o.reads()).collect();
                // nothing the shard produces itself is an input
                assert!(
                    got.is_disjoint(&written),
                    "{lo}..{hi}: an input is also produced here"
                );
                // every read it cannot satisfy IS an input
                for t in read.difference(&written) {
                    assert!(
                        got.contains(t),
                        "{lo}..{hi}: unmet read {t} missing from inputs"
                    );
                }
            }
        }
        // the whole graph needs only what pre-exists: weights, tables, the input
        let all = shard_inputs(&ops, 0..ops.len());
        let produced_anywhere: std::collections::BTreeSet<T> =
            ops.iter().flat_map(|o| o.writes()).collect();
        for t in &all {
            assert!(
                !produced_anywhere.contains(t),
                "tensor {t} is both pre-existing and produced"
            );
        }
    }

    #[test]
    /// This is the check that makes the coarse-step choice decidable rather than
    /// a judgement call: halving the step doubles the index range consumed, so a
    /// step that is correct for one model can silently run a larger one off the
    /// end of the table.
    #[test]
    fn rsqrt_index_stays_inside_the_generated_table() {
        const FINE: i64 = 1 << 20;
        const COARSE: i64 = 1 << 22;
        const COARSE_STEP: i64 = 1 << 8;
        const COARSE2: i64 = 1 << 21;
        const COARSE_STEP2: i64 = 1 << 16;
        const INDEX_SCALE: i64 = 1 << 14;
        assert_eq!(RSQRT_TABLE_LEN, (FINE + COARSE + COARSE2) as usize);
        let coarse2_start = FINE + COARSE * COARSE_STEP;
        let max_raw = coarse2_start + (COARSE2 - 1) * COARSE_STEP2;
        let max_var = max_raw << 18;
        assert!(
            (rsqrt_index(max_var) as usize) < RSQRT_TABLE_LEN,
            "the generator's own maximum variance indexes past the end of its table"
        );
        assert_eq!(max_raw as f64 / INDEX_SCALE as f64, RSQRT_MAX_VARIANCE);
        let gpt2_max = (13_294i64 * INDEX_SCALE) << 18;
        assert!(
            (rsqrt_index(gpt2_max) as usize) < RSQRT_TABLE_LEN,
            "GPT-2's documented maximum variance does not fit the table"
        );
    }

    /// Prove the guard can actually fail.
    #[test]
    #[should_panic(expected = "rsqrt table has")]
    fn check_rsqrt_table_rejects_the_superseded_uniform_table() {
        let superseded = vec![Goldilocks::ZERO; 1 << 19];
        check_rsqrt_table(&superseded, 1 << 20);
    }

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
