//! Expert-major gather (scatter-add): sum the weighted per-expert-segment
//! outputs back into token-major order, proved via the fractional histogram
//! identity `sum_pos weighted[pos]/(alpha - token_id[pos]) = sum_t out[t]/(alpha - t)`
//! (the reverse of a lookup).

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64, ChallengeSource};
use zkie_core::common::fixed_point::from_i64;
use zkie_core::common::logup_gkr::{prove_fractional, verify_fractional, verify_lookup_split, FractionalProof};
use zkie_core::common::mle;
use zkie_core::common::sumcheck::{prove_virtual, verify_virtual, virtual_eval, VirtualProof};
use zkie_core::common::transcript::Transcript;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ScatterAddProof {
    pub source_side: FractionalProof,
    pub target_side: FractionalProof,
    pub alpha: Goldilocks,
    pub gamma: Vec<Goldilocks>,
    pub num_source: Vec<Goldilocks>,
    pub den_source: Vec<Goldilocks>,
    pub num_target: Vec<Goldilocks>,
    pub den_target: Vec<Goldilocks>,
    pub prod_source: VirtualProof,
    pub prod_target: VirtualProof,
    pub r_prod_source: Vec<Goldilocks>,
    pub r_prod_target: Vec<Goldilocks>,
    pub weighted: Vec<Goldilocks>,
    pub gate: Vec<Goldilocks>,
    pub out: Vec<Goldilocks>,
    pub token_id: Vec<u32>,
}

/// `out[t][d] = sum_{pos: token_id[pos] == t} weighted[pos][d] * gate[pos]`.
pub fn gather_forward(
    weighted: &[Goldilocks],
    token_id: &[u32],
    gate: &[Goldilocks],
    m: usize,
    h: usize,
) -> Vec<Goldilocks> {
    assert_eq!(gate.len(), token_id.len());
    let mut out = vec![Goldilocks::ZERO; m * h];
    for (pos, &t) in token_id.iter().enumerate() {
        for d in 0..h {
            out[t as usize * h + d] = out[t as usize * h + d] + weighted[pos * h + d] * gate[pos];
        }
    }
    out
}

pub fn prove_scatter_add(
    weighted: &[Goldilocks],
    token_id: &[u32],
    gate: &[Goldilocks],
    out: &[Goldilocks],
    m: usize,
    h: usize,
    transcript: &mut Transcript,
) -> ScatterAddProof {
    prove_scatter_add_with_mode(weighted, token_id, gate, out, m, h, transcript, crate::compose::FracMode::Plain)
}

#[allow(clippy::too_many_arguments)]
pub fn prove_scatter_add_with_mode(
    weighted: &[Goldilocks],
    token_id: &[u32],
    gate: &[Goldilocks],
    out: &[Goldilocks],
    m: usize,
    h: usize,
    transcript: &mut Transcript,
    mode: crate::compose::FracMode,
) -> ScatterAddProof {
    let s = token_id.len();
    assert_eq!(weighted.len(), s * h);
    assert_eq!(gate.len(), s);
    assert_eq!(out.len(), m * h);
    // Committed mode: alpha/gamma and the two fractional proofs are derived
    // on a scratch transcript — the committed gather section re-proves them
    // on its own transcript (binding the committed column roots), and the
    // shard transcript must stay frac-free for the committed replay. The
    // product sumchecks below always run on the caller's transcript.
    let mut scratch = Transcript::new(b"zkie-scratch-gather");
    let frac_ts: &mut Transcript = match mode {
        crate::compose::FracMode::Plain => transcript,
        crate::compose::FracMode::Committed => &mut scratch,
    };
    let alpha = frac_ts.field();
    let gamma: Vec<Goldilocks> = frac_ts.field_vec(h);
    // WHIR folding needs arity >= 5: pad the fractional domains (num=0 /
    // den=alpha padding contributes 0 to the logUp sum, so the roots
    // relation is unaffected).
    let src_total = (s * h).next_power_of_two().max(32);
    let tgt_total = (m * h).next_power_of_two().max(32);
    let mut num_source = vec![Goldilocks::ZERO; src_total];
    let mut den_source = vec![alpha; src_total];
    let mut num_target = vec![Goldilocks::ZERO; tgt_total];
    let mut den_target = vec![alpha; tgt_total];
    for pos in 0..s {
        for d in 0..h {
            num_source[pos * h + d] = gamma[d] * weighted[pos * h + d] * gate[pos];
            den_source[pos * h + d] = alpha - Goldilocks::from_u64(token_id[pos] as u64);
        }
    }
    for t in 0..m {
        for d in 0..h {
            num_target[t * h + d] = gamma[d] * out[t * h + d];
            den_target[t * h + d] = alpha - Goldilocks::from_u64(t as u64);
        }
    }
    // Den continuation into the padding: the verifier checks
    // den_target = alpha - (index at the point), so padded entries keep the
    // same formula (index = idx >> h_log); num stays 0 there.
    let h_log = h.trailing_zeros() as usize;
    for idx in m * h..tgt_total {
        den_target[idx] = alpha - Goldilocks::from_u64((idx >> h_log) as u64);
    }
    let source_side = prove_fractional(num_source.clone(), den_source.clone(), frac_ts);
    let target_side = prove_fractional(num_target.clone(), den_target.clone(), frac_ts);

    // Product check (source): num_source[pos*h+d] == gamma[d]*weighted[pos*h+d]*gate[pos].
    // The eq point is squeezed upfront (the eq table needs it); the sumcheck
    // challenges are derived per-round (S1) and stored as r_prod_source.
    let src_arity = src_total.trailing_zeros() as usize;
    let r_prod_eq_src: Vec<Goldilocks> = transcript.field_vec(src_arity);
    let eq_src = mle::eq_evals(&r_prod_eq_src);
    let mut gamma_b = vec![Goldilocks::ZERO; src_total];
    let mut gate_b = vec![Goldilocks::ZERO; src_total];
    let mut num_src = vec![Goldilocks::ZERO; src_total];
    let mut weighted_pad = vec![Goldilocks::ZERO; src_total];
    for pos in 0..s {
        for d in 0..h {
            let idx = pos * h + d;
            gamma_b[idx] = gamma[d];
            gate_b[idx] = gate[pos];
            num_src[idx] = num_source[idx];
            weighted_pad[idx] = weighted[idx];
        }
    }
    let (prod_source, r_prod_source) = prove_virtual(
        &[&eq_src, &num_src, &gamma_b, &weighted_pad, &gate_b],
        &[
            (Goldilocks::ONE, vec![0usize, 1usize]),
            (Goldilocks::ZERO - Goldilocks::ONE, vec![0usize, 2usize, 3usize, 4usize]),
        ],
        Goldilocks::ZERO,
        transcript,
    );

    // Product check (target): num_target[t*h+d] == gamma[d]*out[t*h+d]
    let tgt_arity = tgt_total.trailing_zeros() as usize;
    let r_prod_eq_tgt: Vec<Goldilocks> = transcript.field_vec(tgt_arity);
    let eq_tgt = mle::eq_evals(&r_prod_eq_tgt);
    let mut gamma_b2 = vec![Goldilocks::ZERO; tgt_total];
    let mut num_tgt = vec![Goldilocks::ZERO; tgt_total];
    let mut out_pad = vec![Goldilocks::ZERO; tgt_total];
    for t in 0..m {
        for d in 0..h {
            let idx = t * h + d;
            gamma_b2[idx] = gamma[d];
            num_tgt[idx] = num_target[idx];
            out_pad[idx] = out[idx];
        }
    }
    let (prod_target, r_prod_target) = prove_virtual(
        &[&eq_tgt, &num_tgt, &gamma_b2, &out_pad],
        &[
            (Goldilocks::ONE, vec![0usize, 1usize]),
            (Goldilocks::ZERO - Goldilocks::ONE, vec![0usize, 2usize, 3usize]),
        ],
        Goldilocks::ZERO,
        transcript,
    );

    ScatterAddProof {
        source_side,
        target_side,
        alpha,
        gamma,
        num_source,
        den_source,
        num_target,
        den_target,
        prod_source,
        prod_target,
        r_prod_source,
        r_prod_target,
        weighted: weighted.to_vec(),
        gate: gate.to_vec(),
        out: out.to_vec(),
        token_id: token_id.to_vec(),
    }
}

pub fn verify_scatter_add(proof: &ScatterAddProof, transcript: &mut Transcript) -> bool {
    verify_scatter_add_with_mode(proof, transcript, crate::compose::FracMode::Plain)
}

pub fn verify_scatter_add_with_mode(
    proof: &ScatterAddProof,
    transcript: &mut Transcript,
    mode: crate::compose::FracMode,
) -> bool {
    // Plain: replay the alpha/gamma preamble, then the two fractional proofs.
    // Committed: alpha/gamma and the fracs live on the committed gather
    // transcript (never on this shard transcript), so nothing is replayed
    // here; the committed gather section verifies them and compares the
    // product final evals against the openings.
    if mode == crate::compose::FracMode::Plain {
        if transcript.field() != proof.alpha {
            return false;
        }
        let gamma: Vec<Goldilocks> = transcript.field_vec(proof.gamma.len());
        if gamma != proof.gamma {
            return false;
        }
        // S3: each side's depth comes from its (padded, proof-carried)
        // column length.
        let t_src = proof.num_source.len().trailing_zeros() as usize;
        let t_tgt = proof.num_target.len().trailing_zeros() as usize;
        if !verify_lookup_split(
            &proof.source_side,
            &proof.target_side,
            t_src,
            t_tgt,
            transcript,
        ) {
            return false;
        }
    }
    // Product checks: sumcheck self-consistency on the stored final_evals.
    // The final_evals are authenticated by the committed openings. The eq
    // point squeezes only advance the transcript (the eq evals are stored).
    let src_terms = [
        (Goldilocks::ONE, vec![0usize, 1usize]),
        (Goldilocks::ZERO - Goldilocks::ONE, vec![0usize, 2usize, 3usize, 4usize]),
    ];
    let src_arity = proof.r_prod_source.len();
    let _: Vec<Goldilocks> = transcript.field_vec(src_arity);
    let (ok, r, fc) = verify_virtual(
        &proof.prod_source,
        &src_terms,
        Goldilocks::ZERO,
        src_arity,
        transcript,
    );
    if !ok || r != proof.r_prod_source || fc != virtual_eval(&src_terms, &proof.prod_source.final_evals) {
        return false;
    }
    let tgt_terms = [
        (Goldilocks::ONE, vec![0usize, 1usize]),
        (Goldilocks::ZERO - Goldilocks::ONE, vec![0usize, 2usize, 3usize]),
    ];
    let tgt_arity = proof.r_prod_target.len();
    let _: Vec<Goldilocks> = transcript.field_vec(tgt_arity);
    let (ok, r, fc) = verify_virtual(
        &proof.prod_target,
        &tgt_terms,
        Goldilocks::ZERO,
        tgt_arity,
        transcript,
    );
    ok && r == proof.r_prod_target && fc == virtual_eval(&tgt_terms, &proof.prod_target.final_evals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scatter_add_roundtrip() {
        let mut rng = XorShift64::new(0x777);
        let (m, h) = (4usize, 2usize);
        // token_id: 4 positions routed to tokens [0, 2, 1, 2]
        let token_id = vec![0u32, 2, 1, 2];
    let ones: Vec<Goldilocks> = vec![Goldilocks::ONE; token_id.len()];
        let weighted: Vec<Goldilocks> = (0..token_id.len() * h)
            .map(|i| Goldilocks::from_u64((i as u64 % 5) + 1))
            .collect();
        let out = gather_forward(&weighted, &token_id, &ones, m, h);
        // sanity: out[2] = weighted[1] + weighted[3]
        let mut expect = vec![Goldilocks::ZERO; m * h];
        for (pos, &t) in token_id.iter().enumerate() {
            for d in 0..h {
                expect[t as usize * h + d] = expect[t as usize * h + d] + weighted[pos * h + d];
            }
        }
        assert_eq!(out, expect);
        let mut tp = Transcript::new(b"zkie-gather-test");
        tp.absorb_u64(0x777);
        let proof = prove_scatter_add(&weighted, &token_id, &ones, &out, m, h, &mut tp);
        let mut tv = Transcript::new(b"zkie-gather-test");
        tv.absorb_u64(0x777);
        assert!(verify_scatter_add(&proof, &mut tv));
    }
}



    #[test]
    fn scatter_add_weighted_roundtrip() {
        let mut rng = XorShift64::new(0x888);
        let (m, h) = (4usize, 2usize);
        let token_id = vec![0u32, 2, 1, 2];
        let gate: Vec<Goldilocks> = vec![
            Goldilocks::from_u64(2),
            Goldilocks::from_u64(3),
            Goldilocks::from_u64(5),
            Goldilocks::from_u64(7),
        ];
        let weighted: Vec<Goldilocks> = (0..token_id.len() * h)
            .map(|i| Goldilocks::from_u64((i as u64 % 5) + 1))
            .collect();
        let out = gather_forward(&weighted, &token_id, &gate, m, h);
        let mut expect = vec![Goldilocks::ZERO; m * h];
        for (pos, &t) in token_id.iter().enumerate() {
            for d in 0..h {
                expect[t as usize * h + d] =
                    expect[t as usize * h + d] + weighted[pos * h + d] * gate[pos];
            }
        }
        assert_eq!(out, expect);
        let mut tp = Transcript::new(b"zkie-gather-test");
        tp.absorb_u64(0x888);
        let proof = prove_scatter_add(&weighted, &token_id, &gate, &out, m, h, &mut tp);
        let mut tv = Transcript::new(b"zkie-gather-test");
        tv.absorb_u64(0x888);
        assert!(verify_scatter_add(&proof, &mut tv));
    }


#[cfg(test)]
mod moe_minimal {
    use super::*;
    use zkie_core::common::field::XorShift64;

    #[test]
    fn moe_minimal_closure() {
        // synthetic MoE: n=4 experts, k=2 top-k, m=4 tokens, h=2 hidden dims
        let mut rng = XorShift64::new(0x999);
        let (m, n, h) = (4usize, 4usize, 2usize);
        let mut sel = vec![Goldilocks::ZERO; m * n];
        for t in 0..m {
            sel[t * n + (t % n)] = Goldilocks::ONE;
            sel[t * n + ((t + 1) % n)] = Goldilocks::ONE;
        }
        let gate: Vec<Goldilocks> = (0..m * n)
            .map(|i| Goldilocks::from_u64((i as u64 % 3) + 1))
            .collect();
        let x: Vec<Goldilocks> = (0..m * h)
            .map(|i| Goldilocks::from_u64((i as u64 % 7) + 1))
            .collect();

        let (token_id, expert_of) = crate::scatter::scatter_forward(&sel, m, n);
    let ones: Vec<Goldilocks> = vec![Goldilocks::ONE; token_id.len()];
        let mut seg = Vec::with_capacity(token_id.len() * h);
        for &t in &token_id {
            let base = t as usize * h;
            for d in 0..h {
                seg.push(x[base + d]);
            }
        }
        let ffn_out: Vec<Goldilocks> = (0..token_id.len() * h)
            .map(|i| {
                let pos = i / h;
                seg[i] * Goldilocks::from_u64(expert_of[pos] as u64 + 1)
            })
            .collect();
        let weighted: Vec<Goldilocks> = (0..token_id.len() * h)
            .map(|i| {
                let pos = i / h;
                ffn_out[i] * gate[token_id[pos] as usize * n + expert_of[pos] as usize]
            })
            .collect();
        let out = gather_forward(&weighted, &token_id, &ones, m, h);

        // dense reference
        let mut reference = vec![Goldilocks::ZERO; m * h];
        for t in 0..m {
            for e in 0..n {
                if sel[t * n + e] == Goldilocks::ONE {
                    for d in 0..h {
                        reference[t * h + d] = reference[t * h + d]
                            + x[t * h + d] * Goldilocks::from_u64(e as u64 + 1) * gate[t * n + e];
                    }
                }
            }
        }
        assert_eq!(out, reference);

        let mut tsp = Transcript::new(b"zkie-gather-scatter-test");
        tsp.absorb_u64(0x999);
        let sp = crate::scatter::prove_scatter(&sel, &token_id, &expert_of, m, n, &mut tsp);
        let mut tsv = Transcript::new(b"zkie-gather-scatter-test");
        tsv.absorb_u64(0x999);
        assert!(crate::scatter::verify_scatter(&sp, m, n, &mut tsv));
        let mut tp = Transcript::new(b"zkie-gather-test");
        tp.absorb_u64(0x999);
        let gp = prove_scatter_add(&weighted, &token_id, &ones, &out, m, h, &mut tp);
        let mut tv = Transcript::new(b"zkie-gather-test");
        tv.absorb_u64(0x999);
        assert!(verify_scatter_add(&gp, &mut tv));

    }
}
#[cfg(test)]
mod moe_bmm {
    use super::*;
    use crate::scatter::{prove_scatter, scatter_forward, verify_scatter};
    use zkie_core::common::batched_matmul;
    use zkie_core::common::field::{ChallengeSource, Goldilocks, PrimeCharacteristicRing, XorShift64};

    #[test]
    fn expert_major_bmm_closure() {
        let mut rng = XorShift64::new(0x1234);
        let (m, n, h, p) = (4usize, 4usize, 4usize, 2usize);
        let k_top = 2usize;
        let s = m * k_top;

        let mut sel = vec![Goldilocks::ZERO; m * n];
        for t in 0..m {
            sel[t * n + (t % n)] = Goldilocks::ONE;
            sel[t * n + ((t + 1) % n)] = Goldilocks::ONE;
        }
        let gate: Vec<Goldilocks> = (0..m * n)
            .map(|i| Goldilocks::from_u64((i as u64 % 3) + 1))
            .collect();
        let x: Vec<Goldilocks> = (0..m * h)
            .map(|i| Goldilocks::from_u64((i as u64 % 7) + 1))
            .collect();
        let eg_all: Vec<Goldilocks> = (0..n * h * p)
            .map(|i| Goldilocks::from_u64((i as u64 % 5) + 1))
            .collect();

        let (token_id, expert_of) = scatter_forward(&sel, m, n);
    let ones: Vec<Goldilocks> = vec![Goldilocks::ONE; token_id.len()];
        assert_eq!(token_id.len(), s);

        let mut seg = vec![Goldilocks::ZERO; s * h];
        for pos in 0..s {
            let t = token_id[pos] as usize;
            for d in 0..h {
                seg[pos * h + d] = x[t * h + d];
            }
        }

        let eg = batched_matmul::gather_weights(&expert_of, &eg_all, s, n, h, p);
        let g = batched_matmul::forward(&seg, &eg, s, h, p);

        let weighted: Vec<Goldilocks> = (0..s * p)
            .map(|i| {
                let pos = i / p;
                g[i] * gate[token_id[pos] as usize * n + expert_of[pos] as usize]
            })
            .collect();
        let out = gather_forward(&weighted, &token_id, &ones, m, p);

        let mut reference = vec![Goldilocks::ZERO; m * p];
        for t in 0..m {
            for e in 0..n {
                if sel[t * n + e] == Goldilocks::ONE {
                    for j in 0..p {
                        let mut acc = Goldilocks::ZERO;
                        for d in 0..h {
                            acc = acc + x[t * h + d] * eg_all[e * h * p + d * p + j];
                        }
                        reference[t * p + j] = reference[t * p + j] + acc * gate[t * n + e];
                    }
                }
            }
        }
        assert_eq!(out, reference);

        let mut tsp = Transcript::new(b"zkie-gather-scatter-test");
        tsp.absorb_u64(0x1234);
        let sp = prove_scatter(&sel, &token_id, &expert_of, m, n, &mut tsp);
        let mut tsv = Transcript::new(b"zkie-gather-scatter-test");
        tsv.absorb_u64(0x1234);
        assert!(verify_scatter(&sp, m, n, &mut tsv));

        let mut tp_bmm = Transcript::new(b"zkie-bmm-test");
        tp_bmm.absorb_u64(0x1234);
        let u: Vec<Goldilocks> = tp_bmm.field_vec(s.trailing_zeros() as usize);
        let v: Vec<Goldilocks> = tp_bmm.field_vec(p.trailing_zeros() as usize);
        let (bp, ch) = batched_matmul::prove(&seg, &eg, &g, s, h, p, &u, &v, &mut tp_bmm);
        let mut tv_bmm = Transcript::new(b"zkie-bmm-test");
        tv_bmm.absorb_u64(0x1234);
        for _ in 0..s.trailing_zeros() {
            tv_bmm.field();
        }
        for _ in 0..p.trailing_zeros() {
            tv_bmm.field();
        }
        let (ok, ch_v) = batched_matmul::verify(&bp, s, h, &u, &mut tv_bmm);
        assert!(ok);
        assert_eq!(ch_v, ch);

        let mut tp = Transcript::new(b"zkie-gather-test");
        tp.absorb_u64(0x1234);
        let gp = prove_scatter_add(&weighted, &token_id, &ones, &out, m, p, &mut tp);
        let mut tv = Transcript::new(b"zkie-gather-test");
        tv.absorb_u64(0x1234);
        assert!(verify_scatter_add(&gp, &mut tv));

        let mut bad_tid = token_id.clone();
        bad_tid[0] = (bad_tid[0] + 1) % m as u32;
        let mut tsp2 = Transcript::new(b"zkie-gather-scatter-test");
        tsp2.absorb_u64(0x1234);
        let bad_sp = prove_scatter(&sel, &bad_tid, &expert_of, m, n, &mut tsp2);
        let mut tsv2 = Transcript::new(b"zkie-gather-scatter-test");
        tsv2.absorb_u64(0x1234);
        assert!(!verify_scatter(&bad_sp, m, n, &mut tsv2));
    }
}

