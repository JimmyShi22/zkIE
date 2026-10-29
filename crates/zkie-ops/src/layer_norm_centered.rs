//! GPT-2 LayerNorm (centered): `out = trunc((x - mean) * rstd * w / 2^32) + b`,
//! with `mean = trunc(mean(x))`, `var = trunc(mean((x-mean)^2))` and
//! `rstd = rsqrt_table[rsqrt_index(var)]`. Proven as one chain; the truncation
//! remainders are witness values (not yet range-checked — soundness TODO).

use crate::compose::{check_rsqrt_table, rsqrt_index};
use zkie_core::common::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64, ChallengeSource};
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_core::common::logup_gkr::{
    prove_lookup_fractional, verify_lookup_fractional, FractionalProof,
};
use zkie_core::common::mle;
use zkie_core::common::sumcheck::{prove_virtual, verify_virtual, virtual_eval, VirtualProof};
use zkie_core::common::transcript::Transcript;

fn round_div(a: i64, b: i64) -> i64 {
    let q = a.div_euclid(b);
    let r = a.rem_euclid(b);
    if r * 2 >= b {
        q + 1
    } else {
        q
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct LayerNormCenteredProof {
    pub mean: VirtualProof,
    pub centered: VirtualProof,
    pub var: VirtualProof,
    pub out: VirtualProof,
    pub r_mean: Vec<Goldilocks>,
    pub mean_ch: Vec<Goldilocks>,
    pub centered_r: Vec<Goldilocks>,
    pub r_var: Vec<Goldilocks>,
    pub var_ch: Vec<Goldilocks>,
    pub r_out: Vec<Goldilocks>,
}

#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_forward(
    x: &[Goldilocks],
    w: &[Goldilocks],
    b: &[Goldilocks],
    rsqrt_table: &[Goldilocks],
    m: usize,
    d: usize,
    n_real: usize,
) -> (
    Vec<Goldilocks>, // mean
    Vec<Goldilocks>, // centered
    Vec<Goldilocks>, // var
    Vec<u32>,        // s_index
    Vec<Goldilocks>, // rstd
    Vec<Goldilocks>, // out
    Vec<Goldilocks>, // rem_out
    Vec<Goldilocks>, // sum_mean
    Vec<Goldilocks>, // rem_mean
    Vec<Goldilocks>, // sum_var
    Vec<Goldilocks>, // rem_var
) {
    let mean: Vec<Goldilocks> = (0..m)
        .map(|r| {
            let s: i64 = (0..n_real).map(|j| to_i64(x[r * d + j])).sum();
            from_i64(round_div(s, n_real as i64))
        })
        .collect();
    let centered: Vec<Goldilocks> = (0..m * d).map(|ij| x[ij] - mean[ij / d]).collect();
    let var: Vec<Goldilocks> = (0..m)
        .map(|r| {
            let s: i64 = (0..n_real)
                .map(|j| {
                    let c = to_i64(centered[r * d + j]);
                    c * c
                })
                .sum();
            from_i64(round_div(s, n_real as i64))
        })
        .collect();
    let s_index: Vec<u32> = var.iter().map(|&v| rsqrt_index(to_i64(v))).collect();
    check_rsqrt_table(rsqrt_table, s_index.iter().copied().max().unwrap_or(0));
    let rstd: Vec<Goldilocks> = s_index.iter().map(|&i| rsqrt_table[i as usize]).collect();
    let raw: Vec<Goldilocks> = (0..m * d)
        .map(|ij| centered[ij] * rstd[ij / d] * w[ij])
        .collect();
    let out: Vec<Goldilocks> = (0..m * d)
        .map(|ij| from_i64(round_div(to_i64(raw[ij]), 1 << 32) + to_i64(b[ij])))
        .collect();
    let rem_out: Vec<Goldilocks> = (0..m * d)
        .map(|ij| {
            from_i64(
                to_i64(raw[ij]) - (to_i64(out[ij]) - to_i64(b[ij])) * (1i64 << 32) + (1i64 << 31),
            )
        })
        .collect();
    let sum_mean: Vec<Goldilocks> = (0..m)
        .map(|r| (0..n_real).fold(Goldilocks::ZERO, |a, j| a + x[r * d + j]))
        .collect();
    let rem_mean: Vec<Goldilocks> = (0..m)
        .map(|r| {
            from_i64(to_i64(sum_mean[r]) - to_i64(mean[r]) * n_real as i64 + (n_real as i64 / 2))
        })
        .collect();
    let sum_var: Vec<Goldilocks> = (0..m)
        .map(|r| {
            (0..n_real).fold(Goldilocks::ZERO, |a, j| {
                a + centered[r * d + j] * centered[r * d + j]
            })
        })
        .collect();
    let rem_var: Vec<Goldilocks> = (0..m)
        .map(|r| {
            from_i64(to_i64(sum_var[r]) - to_i64(var[r]) * n_real as i64 + (n_real as i64 / 2))
        })
        .collect();
    (
        mean, centered, var, s_index, rstd, out, rem_out, sum_mean, rem_mean, sum_var, rem_var,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn prove_layer_norm_centered(
    x: &[Goldilocks],
    w: &[Goldilocks],
    b: &[Goldilocks],
    rsqrt_table: &[Goldilocks],
    m: usize,
    d: usize,
    n_real: usize,
    transcript: &mut Transcript,
) -> (LayerNormCenteredProof, Vec<u32>, Vec<Goldilocks>) {
    let (mean, centered, var, s_index, rstd, out, rem_out, sum_mean, rem_mean, sum_var, rem_var) =
        layer_norm_forward(x, w, b, rsqrt_table, m, d, n_real);
    let d_f = Goldilocks::from_u64(n_real as u64);
    let half_d = Goldilocks::from_u64((n_real / 2) as u64);
    let half32 = Goldilocks::from_u64(1u64 << 31);
    let neg = Goldilocks::ZERO - Goldilocks::ONE;

    // r_mean is kept as a statement-bound point (the struct field survives for
    // proof-format compatibility); all sumcheck challenges below are derived
    // per-round from their own round messages (S1).
    let r_mean: Vec<Goldilocks> = transcript.field_vec(m.trailing_zeros() as usize);

    // mean truncation: mean*d + rem_mean = sum_mean (elementwise on [m]).
    let ones_m: Vec<Goldilocks> = vec![Goldilocks::ONE; m];
    let trunc_terms = [
        (d_f, vec![0usize]),
        (Goldilocks::ONE, vec![1usize]),
        (neg, vec![2usize]),
        (neg * half_d, vec![3usize]),
    ];
    let (mean_proof, m_ch) = prove_virtual(
        &[&mean, &rem_mean, &sum_mean, &ones_m],
        &trunc_terms,
        Goldilocks::ZERO,
        transcript,
    );

    // centered = x - mean.
    let mean_b: Vec<Goldilocks> = (0..m * d).map(|ij| mean[ij / d]).collect();
    let centered_terms = [
        (Goldilocks::ONE, vec![2usize]),
        (neg, vec![0usize]),
        (Goldilocks::ONE, vec![1usize]),
    ];
    let (centered_proof, centered_r) = prove_virtual(
        &[x, &mean_b, &centered],
        &centered_terms,
        Goldilocks::ZERO,
        transcript,
    );

    // var truncation: var*d + rem_var = sum_var (elementwise on [m]).
    let r_var: Vec<Goldilocks> = transcript.field_vec(m.trailing_zeros() as usize);
    let (var_proof, v_ch) = prove_virtual(
        &[&var, &rem_var, &sum_var, &ones_m],
        &trunc_terms,
        Goldilocks::ZERO,
        transcript,
    );

    // out: raw = centered*rstd*w ; raw = (out-b)*2^32 + rem_out.
    let rstd_b: Vec<Goldilocks> = (0..m * d).map(|ij| rstd[ij / d]).collect();
    let raw: Vec<Goldilocks> = (0..m * d)
        .map(|ij| centered[ij] * rstd[ij / d] * w[ij])
        .collect();
    let two32 = Goldilocks::from_u64(1u64 << 32);
    let ones_md: Vec<Goldilocks> = vec![Goldilocks::ONE; m * d];
    let out_terms = [
        (Goldilocks::ONE, vec![0usize, 1usize, 2usize]),
        (neg, vec![4usize]),
        (Goldilocks::ONE, vec![4usize]),
        (neg * two32, vec![5usize]),
        (two32, vec![3usize]),
        (neg, vec![6usize]),
        (half32, vec![7usize]),
    ];
    let (out_proof, r_out) = prove_virtual(
        &[&centered, &rstd_b, w, b, &raw, &out, &rem_out, &ones_md],
        &out_terms,
        Goldilocks::ZERO,
        transcript,
    );

    (
        LayerNormCenteredProof {
            mean: mean_proof,
            centered: centered_proof,
            var: var_proof,
            out: out_proof,
            r_mean,
            mean_ch: m_ch,
            centered_r,
            r_var,
            var_ch: v_ch,
            r_out,
        },
        s_index,
        rstd,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn verify_layer_norm_centered(
    proof: &LayerNormCenteredProof,
    m: usize,
    d: usize,
    n_real: usize,
    transcript: &mut Transcript,
) -> bool {
    let d_f = Goldilocks::from_u64(n_real as u64);
    let half_d = Goldilocks::from_u64((n_real / 2) as u64);
    let half32 = Goldilocks::from_u64(1u64 << 31);
    let two32 = Goldilocks::from_u64(1u64 << 32);
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let arity_m = m.trailing_zeros() as usize;
    let arity_md = (m * d).trailing_zeros() as usize;

    let r_mean: Vec<Goldilocks> = transcript.field_vec(arity_m);
    if r_mean != proof.r_mean {
        return false;
    }

    let trunc_terms = [
        (d_f, vec![0usize]),
        (Goldilocks::ONE, vec![1usize]),
        (neg, vec![2usize]),
        (neg * half_d, vec![3usize]),
    ];
    let (ok, mean_ch, fc) = verify_virtual(
        &proof.mean,
        &trunc_terms,
        Goldilocks::ZERO,
        arity_m,
        transcript,
    );
    if !ok || mean_ch != proof.mean_ch || fc != virtual_eval(&trunc_terms, &proof.mean.final_evals) {
        return false;
    }

    let centered_terms = [
        (Goldilocks::ONE, vec![2usize]),
        (neg, vec![0usize]),
        (Goldilocks::ONE, vec![1usize]),
    ];
    let (ok, centered_r, fc) = verify_virtual(
        &proof.centered,
        &centered_terms,
        Goldilocks::ZERO,
        arity_md,
        transcript,
    );
    if !ok
        || centered_r != proof.centered_r
        || fc != virtual_eval(&centered_terms, &proof.centered.final_evals)
    {
        return false;
    }

    let r_var: Vec<Goldilocks> = transcript.field_vec(arity_m);
    if r_var != proof.r_var {
        return false;
    }
    let (ok, var_ch, fc) = verify_virtual(
        &proof.var,
        &trunc_terms,
        Goldilocks::ZERO,
        arity_m,
        transcript,
    );
    if !ok || var_ch != proof.var_ch || fc != virtual_eval(&trunc_terms, &proof.var.final_evals) {
        return false;
    }

    let out_terms = [
        (Goldilocks::ONE, vec![0usize, 1usize, 2usize]),
        (neg, vec![4usize]),
        (Goldilocks::ONE, vec![4usize]),
        (neg * two32, vec![5usize]),
        (two32, vec![3usize]),
        (neg, vec![6usize]),
        (half32, vec![7usize]),
    ];
    let (ok, r_out, fc) = verify_virtual(
        &proof.out,
        &out_terms,
        Goldilocks::ZERO,
        arity_md,
        transcript,
    );
    ok && r_out == proof.r_out && fc == virtual_eval(&out_terms, &proof.out.final_evals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_norm_centered_roundtrip() {
        let mut rng = XorShift64::new(0x9E9E);
        let (m, d) = (4usize, 8usize);
        // Small int16-scale inputs keep var inside the (tiny) test table.
        let x: Vec<Goldilocks> = (0..m * d)
            .map(|_| from_i64((rng.next_u64() % 100) as i64 - 50))
            .collect();
        let w: Vec<Goldilocks> = (0..m * d)
            .map(|_| from_i64((rng.next_u64() % 5) as i64 + 1))
            .collect();
        let b: Vec<Goldilocks> = (0..m * d)
            .map(|_| from_i64((rng.next_u64() % 10) as i64 - 5))
            .collect();
        let rsqrt_table: Vec<Goldilocks> = (0..(1usize << 20))
            .map(|j| from_i64((j % 1000 + 1) as i64))
            .collect();
        let mut tp = Transcript::new(b"zkie-lnc-test");
        tp.absorb_u64(0x9E9E);
        let (proof, _ridx, _rvals) =
            prove_layer_norm_centered(&x, &w, &b, &rsqrt_table, m, d, d, &mut tp);
        let mut tv = Transcript::new(b"zkie-lnc-test");
        tv.absorb_u64(0x9E9E);
        assert!(verify_layer_norm_centered(&proof, m, d, d, &mut tv));
    }
}
