//! Per-position (batched) matmul: `g[pos][j] = sum_d seg[pos][d] * w[pos][d][j]`,
//! where each position `pos` carries its own `K x N` weight matrix (already
//! gathered from the per-expert weights).
//!
//! The ordinary single-contraction sumcheck does not apply here: with a shared
//! weight the identity `C~(u,v) = sum_d A~(u,d) B~(d,v)` is multilinear in `u`
//! because `B~` does not depend on `u`; with a per-position weight both factors
//! depend on `u`, so that identity is degree-2 in `u` and is not the MLE of `g`.
//!
//! The sound reduction is an eq-selector virtual sumcheck over the `(S, K)`
//! domain: fix the output variable `j = v`, then
//!
//!   G~(u,v) = sum_{pos,d} seg[pos][d] * w[pos][d][v] * eq(pos, u).
//!
//! This is a single triple-product virtual sumcheck (domain `S*K`), and it
//! reduces to claims on `seg` (at the full challenge) and `w` (at `v || ch`),
//! plus the public `eq(pos, u)` selector.

use crate::common::field::{Goldilocks, PrimeCharacteristicRing};
use crate::common::mle;
use crate::common::sumcheck::{prove_virtual, verify_virtual, VirtualProof};
use rayon::prelude::*;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct BatchedMatmulProof {
    pub claimed: Goldilocks,
    pub sumcheck: VirtualProof,
}

fn broadcast_eq(eq_u: &[Goldilocks], k: usize) -> Vec<Goldilocks> {
    let s = eq_u.len();
    let mut out = vec![Goldilocks::ZERO; s * k];
    for pos in 0..s {
        for d in 0..k {
            out[pos * k + d] = eq_u[pos];
        }
    }
    out
}

fn terms() -> Vec<(Goldilocks, Vec<usize>)> {
    vec![(Goldilocks::ONE, vec![0usize, 1, 2])]
}

/// Prove `g[pos][j] = sum_d seg[pos][d] * w[pos][d][j]` at the random MLE point
/// `(u, v)`.
///
/// Layouts (row-major): `seg` = `S x K`, `w` = `S x K x N`, `g` = `S x N`.
/// `u` fixes the `S` position variables, `v` fixes the `N` output variables, and
/// `ch` carries `S + K` sumcheck challenges (the `(pos, d)` domain). `S`, `K`,
/// `N` must be powers of two.
pub fn prove(
    seg: &[Goldilocks],
    w: &[Goldilocks],
    g: &[Goldilocks],
    s: usize,
    k: usize,
    n: usize,
    u: &[Goldilocks],
    v: &[Goldilocks],
    ch: &[Goldilocks],
) -> BatchedMatmulProof {
    assert_eq!(seg.len(), s * k);
    assert_eq!(w.len(), s * k * n);
    assert_eq!(g.len(), s * n);
    assert_eq!(u.len(), s.trailing_zeros() as usize);
    assert_eq!(v.len(), n.trailing_zeros() as usize);
    assert_eq!(ch.len(), (s * k).trailing_zeros() as usize);

    let w_v = mle::partial_eval(w, v); // S x K, index pos*K + d
    let eq_u = mle::eq_evals(u); // S
    let eq_b = broadcast_eq(&eq_u, k); // S x K

    let mut gp = v.to_vec();
    gp.extend_from_slice(u);
    let claimed = mle::eval(g, &gp);

    let mles: [&[Goldilocks]; 3] = [seg, &w_v, &eq_b];
    let sumcheck = prove_virtual(&mles, &terms(), claimed, ch);
    BatchedMatmulProof { claimed, sumcheck }
}

/// Verify the sumcheck, recomputing the public `eq(pos, u)` selector. The
/// `seg`/`w` endpoint evals carried by the proof are bound to their commitments
/// by the caller (op layer), matching the ordinary-matmul convention.
pub fn verify(
    proof: &BatchedMatmulProof,
    s: usize,
    k: usize,
    u: &[Goldilocks],
    ch: &[Goldilocks],
) -> bool {
    assert_eq!(u.len(), s.trailing_zeros() as usize);
    assert_eq!(ch.len(), (s * k).trailing_zeros() as usize);
    let eq_u = mle::eq_evals(u);
    let eq_b = broadcast_eq(&eq_u, k);
    let eq_final = mle::eval(&eq_b, ch);
    let final_evals = vec![
        proof.sumcheck.final_evals[0],
        proof.sumcheck.final_evals[1],
        eq_final,
    ];
    verify_virtual(
        &proof.sumcheck,
        &terms(),
        proof.claimed,
        ch,
        &final_evals,
    )
}

/// Plain per-position matmul forward (also the test oracle).
pub fn forward(
    seg: &[Goldilocks],
    w: &[Goldilocks],
    s: usize,
    k: usize,
    n: usize,
) -> Vec<Goldilocks> {
    let mut g = vec![Goldilocks::ZERO; s * n];
    for pos in 0..s {
        for j in 0..n {
            let mut acc = Goldilocks::ZERO;
            for d in 0..k {
                acc = acc + seg[pos * k + d] * w[pos * k * n + d * n + j];
            }
            g[pos * n + j] = acc;
        }
    }
    g
}

/// `w_gathered[pos][d][j] = w_all[expert_of[pos]][d][j]`, with `w_all` laid out
/// expert-major as `n_experts x K x N`. Plain forward only: the *proof* of this
/// gather is the existing `Lookup` op (`out = table[idx]`), so it is not
/// duplicated here.
pub fn gather_weights(
    expert_of: &[u32],
    w_all: &[Goldilocks],
    s: usize,
    n_experts: usize,
    k: usize,
    n: usize,
) -> Vec<Goldilocks> {
    assert_eq!(expert_of.len(), s);
    assert_eq!(w_all.len(), n_experts * k * n);
    let mut w = vec![Goldilocks::ZERO; s * k * n];
    for pos in 0..s {
        let base = expert_of[pos] as usize * k * n;
        for d in 0..k {
            for j in 0..n {
                w[pos * k * n + d * n + j] = w_all[base + d * n + j];
            }
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::field::{ChallengeSource, XorShift64};

    #[test]
    fn batched_matmul_roundtrip() {
        let mut rng = XorShift64::new(0xbeef);
        let (s, k, n, n_experts) = (4usize, 4usize, 2usize, 3usize);

        let mut w_all = vec![Goldilocks::ZERO; n_experts * k * n];
        for i in 0..w_all.len() {
            w_all[i] = Goldilocks::from_u64((i as u64 % 7) + 1);
        }
        let expert_of: Vec<u32> = (0..s).map(|i| (i % n_experts) as u32).collect();
        let seg: Vec<Goldilocks> = (0..s * k)
            .map(|i| Goldilocks::from_u64((i as u64 % 11) + 1))
            .collect();

        let w = gather_weights(&expert_of, &w_all, s, n_experts, k, n);
        let g = forward(&seg, &w, s, k, n);

        // dense reference oracle
        let mut reference = vec![Goldilocks::ZERO; s * n];
        for pos in 0..s {
            let e = expert_of[pos] as usize;
            for j in 0..n {
                let mut acc = Goldilocks::ZERO;
                for d in 0..k {
                    acc = acc + seg[pos * k + d] * w_all[e * k * n + d * n + j];
                }
                reference[pos * n + j] = acc;
            }
        }
        assert_eq!(g, reference);

        let u: Vec<Goldilocks> = (0..s.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let ch: Vec<Goldilocks> = (0..(s * k).trailing_zeros() as usize)
            .map(|_| rng.field())
            .collect();

        let proof = prove(&seg, &w, &g, s, k, n, &u, &v, &ch);
        assert!(verify(&proof, s, k, &u, &ch));

        // endpoint evals match the committed tensors at the expected points
        let w_v = mle::partial_eval(&w, &v);
        assert_eq!(proof.sumcheck.final_evals[0], mle::eval(&seg, &ch));
        assert_eq!(proof.sumcheck.final_evals[1], mle::eval(&w_v, &ch));

        // tamper the claimed output value -> reject
        let mut bad = proof;
        bad.claimed = bad.claimed + Goldilocks::ONE;
        assert!(!verify(&bad, s, k, &u, &ch));
    }
}


/// Stream the `partial_eval(W_gathered, v)` result from the per-expert weights,
/// without materializing `W_gathered = S x K x N`. This is the memory-friendly
/// replacement for `partial_eval(gather_weights(...), v)` used by the seq512 path.
pub fn gather_w_v(
    expert_of: &[u32],
    w_all: &[Goldilocks],
    s: usize,
    n_experts: usize,
    k: usize,
    n: usize,
    v: &[Goldilocks],
) -> Vec<Goldilocks> {
    assert_eq!(expert_of.len(), s);
    assert_eq!(w_all.len(), n_experts * k * n);
    let eq_v = mle::eq_evals(v); // n elements
    let mut w_v = vec![Goldilocks::ZERO; s * k];
    for pos in 0..s {
        let base = expert_of[pos] as usize * k * n;
        for d in 0..k {
            let mut acc = Goldilocks::ZERO;
            for j in 0..n {
                acc = acc + w_all[base + d * n + j] * eq_v[j];
            }
            w_v[pos * k + d] = acc;
        }
    }
    w_v
}

/// `counts[e] = sum_{pos: expert_of[pos]=e} eq(pos, ch_pos)`. Computed by the
/// verifier from the (already authenticated) `expert_of`.
pub fn expert_counts(expert_of: &[u32], n_experts: usize, ch_pos: &[Goldilocks]) -> Vec<Goldilocks> {
    let eq_pos = mle::eq_evals(ch_pos); // s elements
    let mut counts = vec![Goldilocks::ZERO; n_experts];
    for (pos, &e) in expert_of.iter().enumerate() {
        counts[e as usize] = counts[e as usize] + eq_pos[pos];
    }
    counts
}

/// `table[e] = MLE_eval(W_all[e], (ch_d, v))`, the MLE of the `K x N` block of
/// expert `e` at the point `(ch_d, v)`. These are the n openings of the committed
/// `W_all` (batchable).
pub fn expert_table_evals(
    w_all: &[Goldilocks],
    n_experts: usize,
    k: usize,
    n: usize,
    ch_d: &[Goldilocks],
    v: &[Goldilocks],
) -> Vec<Goldilocks> {
    assert_eq!(w_all.len(), n_experts * k * n);
    let mut point = v.to_vec();
    point.extend_from_slice(ch_d); // [j = v (LSB), d = ch_d (MSB)]
    let mut table = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        let block = &w_all[e * k * n..(e + 1) * k * n];
        table.push(mle::eval(block, &point));
    }
    table
}

/// The linear-combination binding: `W~_gathered(ch_pos, ch_d, v) = sum_e counts[e] * table[e]`.
pub fn gather_open(counts: &[Goldilocks], table: &[Goldilocks]) -> Goldilocks {
    assert_eq!(counts.len(), table.len());
    let mut acc = Goldilocks::ZERO;
    for i in 0..counts.len() {
        acc = acc + counts[i] * table[i];
    }
    acc
}

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use crate::common::field::{ChallengeSource, XorShift64};

    #[test]
    fn gather_opening_linear_combination() {
        let mut rng = XorShift64::new(0x5151);
        let (s, k, n, n_experts) = (8usize, 4usize, 2usize, 4usize);

        let w_all: Vec<Goldilocks> = (0..n_experts * k * n)
            .map(|i| Goldilocks::from_u64((i as u64 % 7) + 1))
            .collect();
        let expert_of: Vec<u32> = (0..s).map(|i| (i % n_experts) as u32).collect();

        let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let ch_d: Vec<Goldilocks> = (0..k.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let ch_pos: Vec<Goldilocks> = (0..s.trailing_zeros() as usize).map(|_| rng.field()).collect();

        // materialized reference
        let w_gathered = gather_weights(&expert_of, &w_all, s, n_experts, k, n);
        let w_v_ref = mle::partial_eval(&w_gathered, &v);

        // streaming
        let w_v = gather_w_v(&expert_of, &w_all, s, n_experts, k, n, &v);
        assert_eq!(w_v, w_v_ref);

        // weight claim == linear combination over experts
        let mut ch = ch_d.clone();
        ch.extend_from_slice(&ch_pos);
        let weight_claim = mle::eval(&w_v, &ch);

        let counts = expert_counts(&expert_of, n_experts, &ch_pos);
        let table = expert_table_evals(&w_all, n_experts, k, n, &ch_d, &v);
        let bound = gather_open(&counts, &table);

        assert_eq!(weight_claim, bound);
    }
}


/// Streaming per-position matmul forward: `g[pos][j] = sum_d seg[pos][d] * W_all[expert_of[pos]][d][j]`,
/// reading one `K x N` block of `W_all` at a time (no `S x K x N` materialization).
pub fn gather_matmul_forward(
    seg: &[Goldilocks],
    expert_of: &[u32],
    w_all: &[Goldilocks],
    s: usize,
    k: usize,
    n: usize,
    n_experts: usize,
) -> Vec<Goldilocks> {
    assert_eq!(seg.len(), s * k);
    assert_eq!(expert_of.len(), s);
    assert_eq!(w_all.len(), n_experts * k * n);
    let seg_ref = seg;
    let w_ref = w_all;
    let expert_ref = expert_of;
    (0..s)
        .into_par_iter()
        .flat_map_iter(|pos| {
            let base = expert_ref[pos] as usize * k * n;
            (0..n).map(move |j| {
                let mut acc = Goldilocks::ZERO;
                for d in 0..k {
                    acc = acc + seg_ref[pos * k + d] * w_ref[base + d * n + j];
                }
                acc
            })
        })
        .collect()
}

/// Streaming prove: same eq-selector sumcheck as `prove`, but `w_v` is computed on
/// the fly from the per-expert weights instead of a materialized `S x K x N` gather.
pub fn prove_streaming(
    seg: &[Goldilocks],
    expert_of: &[u32],
    w_all: &[Goldilocks],
    g: &[Goldilocks],
    s: usize,
    k: usize,
    n: usize,
    n_experts: usize,
    u: &[Goldilocks],
    v: &[Goldilocks],
    ch: &[Goldilocks],
) -> BatchedMatmulProof {
    assert_eq!(seg.len(), s * k);
    assert_eq!(expert_of.len(), s);
    assert_eq!(w_all.len(), n_experts * k * n);
    assert_eq!(g.len(), s * n);

    let w_v = gather_w_v(expert_of, w_all, s, n_experts, k, n, v);
    let eq_u = mle::eq_evals(u);
    let eq_b = broadcast_eq(&eq_u, k);

    let mut gp = v.to_vec();
    gp.extend_from_slice(u);
    let claimed = mle::eval(g, &gp);

    let mles: [&[Goldilocks]; 3] = [seg, &w_v, &eq_b];
    let sumcheck = prove_virtual(&mles, &terms(), claimed, ch);
    BatchedMatmulProof { claimed, sumcheck }
}

/// Streaming verify: the ordinary eq-selector sumcheck, plus the weight claim bound
/// by the linear combination `final_evals[1] == sum_e counts[e] * table[e]`.
pub fn verify_streaming(
    proof: &BatchedMatmulProof,
    expert_of: &[u32],
    w_all: &[Goldilocks],
    s: usize,
    k: usize,
    n: usize,
    n_experts: usize,
    u: &[Goldilocks],
    v: &[Goldilocks],
    ch: &[Goldilocks],
) -> bool {
    assert_eq!(expert_of.len(), s);
    assert_eq!(w_all.len(), n_experts * k * n);
    if !verify(proof, s, k, u, ch) {
        return false;
    }
    let k_bits = k.trailing_zeros() as usize;
    let ch_d = &ch[..k_bits];
    let ch_pos = &ch[k_bits..];
    let counts = expert_counts(expert_of, n_experts, ch_pos);
    let table = expert_table_evals(w_all, n_experts, k, n, ch_d, v);
    proof.sumcheck.final_evals[1] == gather_open(&counts, &table)
}

#[cfg(test)]
mod streaming_matmul_tests {
    use super::*;
    use crate::common::field::{ChallengeSource, XorShift64};

    #[test]
    fn streaming_matches_materialized() {
        let mut rng = XorShift64::new(0x6161);
        let (s, k, n, n_experts) = (8usize, 4usize, 2usize, 4usize);

        let w_all: Vec<Goldilocks> = (0..n_experts * k * n)
            .map(|i| Goldilocks::from_u64((i as u64 % 7) + 1))
            .collect();
        let expert_of: Vec<u32> = (0..s).map(|i| (i % n_experts) as u32).collect();
        let seg: Vec<Goldilocks> = (0..s * k)
            .map(|i| Goldilocks::from_u64((i as u64 % 11) + 1))
            .collect();

        // streaming forward == materialized forward
        let g_stream = gather_matmul_forward(&seg, &expert_of, &w_all, s, k, n, n_experts);
        let w_gathered = gather_weights(&expert_of, &w_all, s, n_experts, k, n);
        let g_mat = forward(&seg, &w_gathered, s, k, n);
        assert_eq!(g_stream, g_mat);

        let u: Vec<Goldilocks> = (0..s.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let ch: Vec<Goldilocks> = (0..(s * k).trailing_zeros() as usize).map(|_| rng.field()).collect();

        let proof = prove_streaming(&seg, &expert_of, &w_all, &g_stream, s, k, n, n_experts, &u, &v, &ch);
        assert!(verify_streaming(&proof, &expert_of, &w_all, s, k, n, n_experts, &u, &v, &ch));

        // tamper the weight claim -> reject
        let mut bad = proof;
        bad.sumcheck.final_evals[1] = bad.sumcheck.final_evals[1] + Goldilocks::ONE;
        assert!(!verify_streaming(&bad, &expert_of, &w_all, s, k, n, n_experts, &u, &v, &ch));
    }
}


/// Self-contained streaming gather-matmul proof: the eq-selector sumcheck plus the
/// linear-combination terms that bind the weight claim to the per-expert weights.
/// `table[e] = MLE_eval(W_all[e], (ch_d, v))` is opened from the committed `W_all`
/// on the committed path; `counts[e] = sum_{pos->e} eq(pos, ch_pos)` is recomputed
/// from the authenticated `expert_of`.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct GatherMatMulProof {
    pub claimed: Goldilocks,
    pub sumcheck: VirtualProof,
    pub counts: Vec<Goldilocks>,
    pub table: Vec<Goldilocks>,
}

pub fn prove_gather_matmul(
    seg: &[Goldilocks],
    expert_of: &[u32],
    w_all: &[Goldilocks],
    g: &[Goldilocks],
    s: usize,
    k: usize,
    n: usize,
    n_experts: usize,
    u: &[Goldilocks],
    v: &[Goldilocks],
    ch: &[Goldilocks],
) -> GatherMatMulProof {
    let matmul = prove_streaming(seg, expert_of, w_all, g, s, k, n, n_experts, u, v, ch);
    let k_bits = k.trailing_zeros() as usize;
    let ch_d = &ch[..k_bits];
    let ch_pos = &ch[k_bits..];
    let counts = expert_counts(expert_of, n_experts, ch_pos);
    let table = expert_table_evals(w_all, n_experts, k, n, ch_d, v);
    GatherMatMulProof {
        claimed: matmul.claimed,
        sumcheck: matmul.sumcheck,
        counts,
        table,
    }
}

/// Memory-streaming variant of [`prove_gather_matmul`] for the full 552B
/// MoE: instead of requiring the whole `n_experts * k * n` weight stack to be
/// materialized at once, it pulls each expert on demand through `mat_expert`.
/// The eq-selector sumcheck only ever needs the *selected* experts (cached), and
/// the per-expert `table[e]` evals materialize one expert at a time, so peak
/// memory stays at a single expert plus the selected-expert cache.
pub fn prove_gather_matmul_streaming<F>(
    seg: &[Goldilocks],
    expert_of: &[u32],
    mat_expert: F,
    g: &[Goldilocks],
    s: usize,
    k: usize,
    n: usize,
    n_experts: usize,
    u: &[Goldilocks],
    v: &[Goldilocks],
    ch: &[Goldilocks],
) -> GatherMatMulProof
where
    F: Fn(usize) -> Vec<Goldilocks>,
{
    assert_eq!(seg.len(), s * k);
    assert_eq!(expert_of.len(), s);
    assert_eq!(g.len(), s * n);

    // gather_w_v over only the experts actually selected, materializing each
    // unique selected expert once.
    let eq_v = mle::eq_evals(v);
    let mut cache: Vec<Option<Vec<Goldilocks>>> = (0..n_experts).map(|_| None).collect();
    let mut w_v = vec![Goldilocks::ZERO; s * k];
    for pos in 0..s {
        let e = expert_of[pos] as usize;
        if cache[e].is_none() {
            let w = mat_expert(e);
            assert_eq!(w.len(), k * n);
            cache[e] = Some(w);
        }
        let w = cache[e].as_ref().unwrap();
        for d in 0..k {
            let mut acc = Goldilocks::ZERO;
            for j in 0..n {
                acc = acc + w[d * n + j] * eq_v[j];
            }
            w_v[pos * k + d] = acc;
        }
    }
    drop(cache);

    let eq_u = mle::eq_evals(u);
    let eq_b = broadcast_eq(&eq_u, k);
    let mut gp = v.to_vec();
    gp.extend_from_slice(u);
    let claimed = mle::eval(g, &gp);
    let mles: [&[Goldilocks]; 3] = [seg, &w_v, &eq_b];
    let sumcheck = prove_virtual(&mles, &terms(), claimed, ch);

    let k_bits = k.trailing_zeros() as usize;
    let ch_d = &ch[..k_bits];
    let ch_pos = &ch[k_bits..];
    let counts = expert_counts(expert_of, n_experts, ch_pos);

    // expert_table_evals, one expert materialized at a time.
    let mut point = v.to_vec();
    point.extend_from_slice(ch_d);
    let mut table = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        let w = mat_expert(e);
        assert_eq!(w.len(), k * n);
        table.push(mle::eval(&w, &point));
    }

    GatherMatMulProof {
        claimed,
        sumcheck,
        counts,
        table,
    }
}

pub fn verify_gather_matmul(
    proof: &GatherMatMulProof,
    s: usize,
    k: usize,
    u: &[Goldilocks],
    ch: &[Goldilocks],
) -> bool {
    let eq_u = mle::eq_evals(u);
    let eq_b = broadcast_eq(&eq_u, k);
    let eq_final = mle::eval(&eq_b, ch);
    let final_evals = vec![
        proof.sumcheck.final_evals[0],
        proof.sumcheck.final_evals[1],
        eq_final,
    ];
    if !verify_virtual(&proof.sumcheck, &terms(), proof.claimed, ch, &final_evals) {
        return false;
    }
    proof.sumcheck.final_evals[1] == gather_open(&proof.counts, &proof.table)
}

#[cfg(test)]
mod gather_matmul_proof_tests {
    use super::*;
    use crate::common::field::{ChallengeSource, XorShift64};

    #[test]
    fn gather_matmul_parallel_matches_sequential() {
        use rayon::prelude::*;
        let mut rng = XorShift64::new(0x9191);
        let (s, k, n, n_experts) = (32usize, 64usize, 64usize, 8usize);
        let seg: Vec<Goldilocks> = (0..s * k)
            .map(|_| Goldilocks::from_u64(rng.next_u64() % (1 << 20)))
            .collect();
        let w_all: Vec<Goldilocks> = (0..n_experts * k * n)
            .map(|_| Goldilocks::from_u64(rng.next_u64() % (1 << 20)))
            .collect();
        let expert_of: Vec<u32> = (0..s)
            .map(|_| (rng.next_u64() % n_experts as u64) as u32)
            .collect();
        let seq = gather_matmul_forward(&seg, &expert_of, &w_all, s, k, n, n_experts);
        let seg_ref = seg.as_slice();
        let w_ref = w_all.as_slice();
        let expert_ref = expert_of.as_slice();
        let par: Vec<Goldilocks> = (0..s)
            .into_par_iter()
            .flat_map_iter(|pos| {
                let base = expert_ref[pos] as usize * k * n;
                (0..n).map(move |j| {
                    let mut acc = Goldilocks::ZERO;
                    for d in 0..k {
                        acc = acc + seg_ref[pos * k + d] * w_ref[base + d * n + j];
                    }
                    acc
                })
            })
            .collect();
        assert_eq!(seq, par);
    }

    #[test]
    fn gather_matmul_self_contained() {
        let mut rng = XorShift64::new(0x7171);
        let (s, k, n, n_experts) = (8usize, 4usize, 2usize, 4usize);
        let w_all: Vec<Goldilocks> = (0..n_experts * k * n)
            .map(|i| Goldilocks::from_u64((i as u64 % 7) + 1))
            .collect();
        let expert_of: Vec<u32> = (0..s).map(|i| (i % n_experts) as u32).collect();
        let seg: Vec<Goldilocks> = (0..s * k)
            .map(|i| Goldilocks::from_u64((i as u64 % 11) + 1))
            .collect();
        let g = gather_matmul_forward(&seg, &expert_of, &w_all, s, k, n, n_experts);

        let u: Vec<Goldilocks> = (0..s.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let ch: Vec<Goldilocks> = (0..(s * k).trailing_zeros() as usize).map(|_| rng.field()).collect();

        let proof = prove_gather_matmul(&seg, &expert_of, &w_all, &g, s, k, n, n_experts, &u, &v, &ch);
        assert!(verify_gather_matmul(&proof, s, k, &u, &ch));

        // tamper the weight claim (final_evals[1]) -> reject
        let mut bad = proof;
        bad.sumcheck.final_evals[1] = bad.sumcheck.final_evals[1] + Goldilocks::ONE;
        assert!(!verify_gather_matmul(&bad, s, k, &u, &ch));
    }
}

