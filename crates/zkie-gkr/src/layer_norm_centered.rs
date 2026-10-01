//! GPT-2 LayerNorm (centered): `out = trunc((x - mean) * rstd * w / 2^32) + b`,
//! with `mean = trunc(mean(x))`, `var = trunc(mean((x-mean)^2))` and
//! `rstd = rsqrt_table[rsqrt_index(var)]`. Proven as one chain; the truncation
//! remainders are witness values (not yet range-checked — soundness TODO).

use crate::compose::rsqrt_index;
use crate::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::fixed_point::{from_i64, to_i64};
use crate::logup_gkr::{prove_lookup_fractional, verify_lookup_fractional, FractionalProof};
use crate::mle;
use crate::sumcheck::{prove_virtual, verify_virtual, VirtualProof};

fn round_div(a: i64, b: i64) -> i64 {
    let q = a.div_euclid(b);
    let r = a.rem_euclid(b);
    if r * 2 >= b { q + 1 } else { q }
}

pub struct LayerNormCenteredProof {
    pub mean: VirtualProof,
    pub centered: VirtualProof,
    pub var: VirtualProof,
    pub rstd: FractionalProof,
    pub out: VirtualProof,
    pub r_mean: Vec<Goldilocks>,
    pub mean_ch: Vec<Goldilocks>,
    pub centered_r: Vec<Goldilocks>,
    pub r_var: Vec<Goldilocks>,
    pub var_ch: Vec<Goldilocks>,
    pub r_out: Vec<Goldilocks>,
    pub alpha: Goldilocks,
    pub beta: Goldilocks,
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
            let s: i64 = (0..n_real).map(|j| {
                let c = to_i64(centered[r * d + j]);
                c * c
            }).sum();
            from_i64(round_div(s, n_real as i64))
        })
        .collect();
    let s_index: Vec<u32> = var.iter().map(|&v| rsqrt_index(to_i64(v))).collect();
    let rstd: Vec<Goldilocks> = s_index.iter().map(|&i| rsqrt_table[i as usize]).collect();
    let raw: Vec<Goldilocks> = (0..m * d).map(|ij| centered[ij] * rstd[ij / d] * w[ij]).collect();
    let out: Vec<Goldilocks> = (0..m * d)
        .map(|ij| from_i64(round_div(to_i64(raw[ij]), 1 << 32) + to_i64(b[ij])))
        .collect();
    let rem_out: Vec<Goldilocks> = (0..m * d)
        .map(|ij| from_i64(to_i64(raw[ij]) - (to_i64(out[ij]) - to_i64(b[ij])) * (1i64 << 32) + (1i64 << 31)))
        .collect();
    let sum_mean: Vec<Goldilocks> = (0..m).map(|r| (0..n_real).fold(Goldilocks::ZERO, |a, j| a + x[r * d + j])).collect();
    let rem_mean: Vec<Goldilocks> = (0..m).map(|r| from_i64(to_i64(sum_mean[r]) - to_i64(mean[r]) * n_real as i64 + (n_real as i64 / 2))).collect();
    let sum_var: Vec<Goldilocks> = (0..m).map(|r| (0..n_real).fold(Goldilocks::ZERO, |a, j| a + centered[r * d + j] * centered[r * d + j])).collect();
    let rem_var: Vec<Goldilocks> = (0..m).map(|r| from_i64(to_i64(sum_var[r]) - to_i64(var[r]) * n_real as i64 + (n_real as i64 / 2))).collect();
    (mean, centered, var, s_index, rstd, out, rem_out, sum_mean, rem_mean, sum_var, rem_var)
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
    rng: &mut XorShift64,
) -> LayerNormCenteredProof {
    let (mean, centered, var, s_index, rstd, out, rem_out, sum_mean, rem_mean, sum_var, rem_var) =
        layer_norm_forward(x, w, b, rsqrt_table, m, d, n_real);
    let d_f = Goldilocks::from_u64(n_real as u64);
    let half_d = Goldilocks::from_u64((n_real / 2) as u64);
    let half32 = Goldilocks::from_u64(1u64 << 31);
    let neg = Goldilocks::ZERO - Goldilocks::ONE;

    // mean reduction: sum_mean(r) = sum_j x(r,j).
    let r_mean: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let eq_m = mle::eq_evals(&r_mean);
    let eq_b: Vec<Goldilocks> = (0..m * d).map(|idx| eq_m[idx / d]).collect();
    let mean_ch: Vec<Goldilocks> = (0..(m * d).trailing_zeros() as usize).map(|_| rng.field()).collect();
    prove_virtual(
        &[&eq_b, x],
        &[(Goldilocks::ONE, vec![0usize, 1usize])],
        mle::eval(&sum_mean, &r_mean),
        &mean_ch,
    );
    // mean truncation: mean*d + rem_mean = sum_mean (elementwise on [m]).
    let m_ch: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let ones_m: Vec<Goldilocks> = vec![Goldilocks::ONE; m];
    let mean_proof = prove_virtual(
        &[&mean, &rem_mean, &sum_mean, &ones_m],
        &[
            (d_f, vec![0usize]),
            (Goldilocks::ONE, vec![1usize]),
            (neg, vec![2usize]),
            (neg * half_d, vec![3usize]),
        ],
        Goldilocks::ZERO,
        &m_ch,
    );

    // centered = x - mean.
    let mean_b: Vec<Goldilocks> = (0..m * d).map(|ij| mean[ij / d]).collect();
    let centered_r: Vec<Goldilocks> = (0..(m * d).trailing_zeros() as usize).map(|_| rng.field()).collect();
    let centered_proof = prove_virtual(
        &[x, &mean_b, &centered],
        &[(Goldilocks::ONE, vec![2usize]), (neg, vec![0usize]), (Goldilocks::ONE, vec![1usize])],
        Goldilocks::ZERO,
        &centered_r,
    );

    // var reduction: sum_var(r) = sum_j centered^2.
    let r_var: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let eq_m2 = mle::eq_evals(&r_var);
    let eq_b2: Vec<Goldilocks> = (0..m * d).map(|idx| eq_m2[idx / d]).collect();
    let var_ch: Vec<Goldilocks> = (0..(m * d).trailing_zeros() as usize).map(|_| rng.field()).collect();
    prove_virtual(
        &[&eq_b2, &centered],
        &[(Goldilocks::ONE, vec![0usize, 1usize, 1usize])],
        mle::eval(&sum_var, &r_var),
        &var_ch,
    );
    let v_ch: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let var_proof = prove_virtual(
        &[&var, &rem_var, &sum_var, &ones_m],
        &[
            (d_f, vec![0usize]),
            (Goldilocks::ONE, vec![1usize]),
            (neg, vec![2usize]),
            (neg * half_d, vec![3usize]),
        ],
        Goldilocks::ZERO,
        &v_ch,
    );

    // rstd lookup.
    let alpha = rng.field();
    let beta = rng.field();
    let rstd_proof = prove_lookup_fractional(&s_index, &rstd, rsqrt_table, alpha, beta, rng);

    // out: raw = centered*rstd*w ; raw = (out-b)*2^32 + rem_out.
    let rstd_b: Vec<Goldilocks> = (0..m * d).map(|ij| rstd[ij / d]).collect();
    let raw: Vec<Goldilocks> = (0..m * d).map(|ij| centered[ij] * rstd[ij / d] * w[ij]).collect();
    let r_out: Vec<Goldilocks> = (0..(m * d).trailing_zeros() as usize).map(|_| rng.field()).collect();
    let two32 = Goldilocks::from_u64(1u64 << 32);
    let ones_md: Vec<Goldilocks> = vec![Goldilocks::ONE; m * d];
    let out_proof = prove_virtual(
        &[&centered, &rstd_b, w, b, &raw, &out, &rem_out, &ones_md],
        &[
            (Goldilocks::ONE, vec![0usize, 1usize, 2usize]),
            (neg, vec![4usize]),
            (Goldilocks::ONE, vec![4usize]),
            (neg * two32, vec![5usize]),
            (two32, vec![3usize]),
            (neg, vec![6usize]),
            (half32, vec![7usize]),
        ],
        Goldilocks::ZERO,
        &r_out,
    );

    LayerNormCenteredProof {
        mean: mean_proof,
        centered: centered_proof,
        var: var_proof,
        rstd: rstd_proof,
        out: out_proof,
        r_mean,
        mean_ch: m_ch,
        centered_r,
        r_var,
        var_ch: v_ch,
        r_out,
        alpha,
        beta,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn verify_layer_norm_centered(
    proof: &LayerNormCenteredProof,
    x: &[Goldilocks],
    w: &[Goldilocks],
    b: &[Goldilocks],
    rsqrt_table: &[Goldilocks],
    m: usize,
    d: usize,
    n_real: usize,
) -> bool {
    let (mean, centered, var, s_index, rstd, out, rem_out, sum_mean, rem_mean, sum_var, rem_var) =
        layer_norm_forward(x, w, b, rsqrt_table, m, d, n_real);
    let d_f = Goldilocks::from_u64(n_real as u64);
    let half_d = Goldilocks::from_u64((n_real / 2) as u64);
    let half32 = Goldilocks::from_u64(1u64 << 31);
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let ones_m: Vec<Goldilocks> = vec![Goldilocks::ONE; m];

    // mean truncation.
    if !verify_virtual(
        &proof.mean,
        &[
            (d_f, vec![0usize]),
            (Goldilocks::ONE, vec![1usize]),
            (neg, vec![2usize]),
            (neg * half_d, vec![3usize]),
        ],
        Goldilocks::ZERO,
        &proof.mean_ch,
        &[mle::eval(&mean, &proof.mean_ch), mle::eval(&rem_mean, &proof.mean_ch), mle::eval(&sum_mean, &proof.mean_ch), mle::eval(&ones_m, &proof.mean_ch)],
    ) {
        return false;
    }

    let mean_b: Vec<Goldilocks> = (0..m * d).map(|ij| mean[ij / d]).collect();
    if !verify_virtual(
        &proof.centered,
        &[(Goldilocks::ONE, vec![2usize]), (neg, vec![0usize]), (Goldilocks::ONE, vec![1usize])],
        Goldilocks::ZERO,
        &proof.centered_r,
        &[mle::eval(x, &proof.centered_r), mle::eval(&mean_b, &proof.centered_r), mle::eval(&centered, &proof.centered_r)],
    ) {
        return false;
    }

    if !verify_virtual(
        &proof.var,
        &[
            (d_f, vec![0usize]),
            (Goldilocks::ONE, vec![1usize]),
            (neg, vec![2usize]),
            (neg * half_d, vec![3usize]),
        ],
        Goldilocks::ZERO,
        &proof.var_ch,
        &[mle::eval(&var, &proof.var_ch), mle::eval(&rem_var, &proof.var_ch), mle::eval(&sum_var, &proof.var_ch), mle::eval(&ones_m, &proof.var_ch)],
    ) {
        return false;
    }

    if !verify_lookup_fractional(&proof.rstd, &s_index, &rstd, rsqrt_table, proof.alpha, proof.beta) {
        return false;
    }

    let rstd_b: Vec<Goldilocks> = (0..m * d).map(|ij| rstd[ij / d]).collect();
    let raw: Vec<Goldilocks> = (0..m * d).map(|ij| centered[ij] * rstd[ij / d] * w[ij]).collect();
    let two32 = Goldilocks::from_u64(1u64 << 32);
    let ones_md: Vec<Goldilocks> = vec![Goldilocks::ONE; m * d];
    verify_virtual(
        &proof.out,
        &[
            (Goldilocks::ONE, vec![0usize, 1usize, 2usize]),
            (neg, vec![4usize]),
            (Goldilocks::ONE, vec![4usize]),
            (neg * two32, vec![5usize]),
            (two32, vec![3usize]),
            (neg, vec![6usize]),
            (half32, vec![7usize]),
        ],
        Goldilocks::ZERO,
        &proof.r_out,
        &[
            mle::eval(&centered, &proof.r_out),
            mle::eval(&rstd_b, &proof.r_out),
            mle::eval(w, &proof.r_out),
            mle::eval(b, &proof.r_out),
            mle::eval(&raw, &proof.r_out),
            mle::eval(&out, &proof.r_out),
            mle::eval(&rem_out, &proof.r_out),
            mle::eval(&ones_md, &proof.r_out),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_norm_centered_roundtrip() {
        let mut rng = XorShift64::new(0x9E9E);
        let (m, d) = (4usize, 8usize);
        // Small int16-scale inputs keep var inside the (tiny) test table.
        let x: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 100) as i64 - 50)).collect();
        let w: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 5) as i64 + 1)).collect();
        let b: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect();
        let rsqrt_table: Vec<Goldilocks> = (0..(1usize << 20)).map(|j| from_i64((j % 1000 + 1) as i64)).collect();
        let proof = prove_layer_norm_centered(&x, &w, &b, &rsqrt_table, m, d, d, &mut rng);
        assert!(verify_layer_norm_centered(&proof, &x, &w, &b, &rsqrt_table, m, d, d));
    }
}
