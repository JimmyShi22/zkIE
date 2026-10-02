//! RMSNorm (no centering): `out = trunc(x * rstd * w / 2^32)`, with
//! `ms = trunc(mean(x^2))` over `n_real`, and
//! `rstd = rsqrt_table[rsqrt_index(ms)]`. Mirrors `LayerNormCentered`'s
//! actually-verified chain: ms truncation + rsqrt lookup + out affine.

use crate::compose::rsqrt_index;
use zkie_core::common::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::{from_i64, to_i32, to_i64};
use zkie_core::common::logup_gkr::{
    prove_lookup_fractional, verify_lookup_fractional, FractionalProof,
};
use zkie_core::common::mle;
use zkie_core::common::sumcheck::{prove_virtual, verify_virtual, VirtualProof};

fn round_div(a: i64, b: i64) -> i64 {
    let q = a.div_euclid(b);
    let r = a.rem_euclid(b);
    if r * 2 >= b { q + 1 } else { q }
}

pub struct RmsNormProof {
    pub trunc: VirtualProof,
    pub rstd: FractionalProof,
    pub out: VirtualProof,
    pub rem_split: VirtualProof,
    pub rem_lo_range: FractionalProof,
    pub rem_hi_range: FractionalProof,
    pub trunc_ch: Vec<Goldilocks>,
    pub r_out: Vec<Goldilocks>,
    pub rem_split_ch: Vec<Goldilocks>,
    pub alpha: Goldilocks,
    pub beta: Goldilocks,
    pub alpha_lo: Goldilocks,
    pub beta_lo: Goldilocks,
    pub alpha_hi: Goldilocks,
    pub beta_hi: Goldilocks,
}

#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_forward(
    x: &[Goldilocks],
    w: &[Goldilocks],
    rsqrt_table: &[Goldilocks],
    m: usize,
    d: usize,
    n_real: usize,
) -> (
    Vec<Goldilocks>, // ms
    Vec<u32>,        // s_index
    Vec<Goldilocks>, // rstd
    Vec<Goldilocks>, // out
    Vec<Goldilocks>, // rem_out
    Vec<Goldilocks>, // rem_lo
    Vec<Goldilocks>, // rem_hi
    Vec<Goldilocks>, // sum_sq
    Vec<Goldilocks>, // rem_s
) {
    let ms: Vec<Goldilocks> = (0..m)
        .map(|r| {
            let s: i64 = (0..n_real)
                .map(|j| {
                    let v = to_i64(x[r * d + j]);
                    v * v
                })
                .sum();
            from_i64(round_div(s, n_real as i64))
        })
        .collect();
    let s_index: Vec<u32> = ms.iter().map(|&v| rsqrt_index(to_i64(v))).collect();
    let rstd: Vec<Goldilocks> = s_index.iter().map(|&i| rsqrt_table[i as usize]).collect();
    let raw: Vec<Goldilocks> = (0..m * d).map(|ij| x[ij] * rstd[ij / d] * w[ij]).collect();
    let out: Vec<Goldilocks> = (0..m * d)
        .map(|ij| from_i64(round_div(to_i64(raw[ij]), 1i64 << 32)))
        .collect();
    let rem_out: Vec<Goldilocks> = (0..m * d)
        .map(|ij| from_i64(to_i64(raw[ij]) - to_i64(out[ij]) * (1i64 << 32) + (1i64 << 31)))
        .collect();
    let rem_lo: Vec<Goldilocks> = rem_out.iter().map(|&v| from_i64(to_i64(v) & 0xFFFF)).collect();
    let rem_hi: Vec<Goldilocks> = rem_out.iter().map(|&v| from_i64(to_i64(v) >> 16)).collect();
    let sum_sq: Vec<Goldilocks> = (0..m)
        .map(|r| (0..n_real).fold(Goldilocks::ZERO, |a, j| a + x[r * d + j] * x[r * d + j]))
        .collect();
    let rem_s: Vec<Goldilocks> = (0..m)
        .map(|r| from_i64(to_i64(sum_sq[r]) - to_i64(ms[r]) * n_real as i64 + (n_real as i64 / 2)))
        .collect();
    (ms, s_index, rstd, out, rem_out, rem_lo, rem_hi, sum_sq, rem_s)
}

#[allow(clippy::too_many_arguments)]
pub fn prove_rms_norm(
    x: &[Goldilocks],
    w: &[Goldilocks],
    rsqrt_table: &[Goldilocks],
    m: usize,
    d: usize,
    n_real: usize,
    rng: &mut XorShift64,
) -> RmsNormProof {
    let (ms, s_index, rstd, out, rem_out, rem_lo, rem_hi, sum_sq, rem_s) =
        rms_norm_forward(x, w, rsqrt_table, m, d, n_real);

    let d_f = Goldilocks::from_u64(n_real as u64);
    let half_d = Goldilocks::from_u64((n_real / 2) as u64);
    let half32 = Goldilocks::from_u64(1u64 << 31);
    let two32 = Goldilocks::from_u64(1u64 << 32);
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let ones_m: Vec<Goldilocks> = vec![Goldilocks::ONE; m];
    let ones_md: Vec<Goldilocks> = vec![Goldilocks::ONE; m * d];

    let trunc_ch: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let trunc_proof = prove_virtual(
        &[&ms, &rem_s, &sum_sq, &ones_m],
        &[
            (d_f, vec![0usize]),
            (Goldilocks::ONE, vec![1usize]),
            (neg, vec![2usize]),
            (neg * half_d, vec![3usize]),
        ],
        Goldilocks::ZERO,
        &trunc_ch,
    );

    let alpha = rng.field();
    let beta = rng.field();
    let rstd_proof = prove_lookup_fractional(&s_index, &rstd, rsqrt_table, alpha, beta, rng);

    let rstd_b: Vec<Goldilocks> = (0..m * d).map(|ij| rstd[ij / d]).collect();
    let raw: Vec<Goldilocks> = (0..m * d).map(|ij| x[ij] * rstd[ij / d] * w[ij]).collect();
    let r_out: Vec<Goldilocks> = (0..(m * d).trailing_zeros() as usize).map(|_| rng.field()).collect();
    let out_proof = prove_virtual(
        &[x, &rstd_b, w, &raw, &out, &rem_out, &ones_md],
        &[
            (Goldilocks::ONE, vec![0usize, 1usize, 2usize]),
            (neg, vec![3usize]),
            (Goldilocks::ONE, vec![3usize]),
            (neg * two32, vec![4usize]),
            (neg, vec![5usize]),
            (half32, vec![6usize]),
        ],
        Goldilocks::ZERO,
        &r_out,
    );

    let rem_split_ch: Vec<Goldilocks> = (0..(m * d).trailing_zeros() as usize).map(|_| rng.field()).collect();
    let rem_split = prove_virtual(
        &[&rem_out, &rem_lo, &rem_hi],
        &[
            (Goldilocks::ONE, vec![0usize]),
            (neg, vec![1usize]),
            (neg * Goldilocks::from_u64(1u64 << 16), vec![2usize]),
        ],
        Goldilocks::ZERO,
        &rem_split_ch,
    );
    let limb_table: Vec<Goldilocks> = (0..(1usize << 16)).map(|j| Goldilocks::from_u64(j as u64)).collect();
    let idx_lo: Vec<u32> = rem_lo.iter().map(|&v| to_i32(v) as u32).collect();
    let idx_hi: Vec<u32> = rem_hi.iter().map(|&v| to_i32(v) as u32).collect();
    let alpha_lo = rng.field();
    let beta_lo = rng.field();
    let rem_lo_range = prove_lookup_fractional(&idx_lo, &rem_lo, &limb_table, alpha_lo, beta_lo, rng);
    let alpha_hi = rng.field();
    let beta_hi = rng.field();
    let rem_hi_range = prove_lookup_fractional(&idx_hi, &rem_hi, &limb_table, alpha_hi, beta_hi, rng);

    RmsNormProof {
        trunc: trunc_proof,
        rstd: rstd_proof,
        out: out_proof,
        rem_split,
        rem_lo_range,
        rem_hi_range,
        trunc_ch,
        r_out,
        rem_split_ch,
        alpha,
        beta,
        alpha_lo,
        beta_lo,
        alpha_hi,
        beta_hi,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn verify_rms_norm(
    proof: &RmsNormProof,
    x: &[Goldilocks],
    w: &[Goldilocks],
    rsqrt_table: &[Goldilocks],
    m: usize,
    d: usize,
    n_real: usize,
) -> bool {
    let (ms, s_index, rstd, out, rem_out, rem_lo, rem_hi, sum_sq, rem_s) =
        rms_norm_forward(x, w, rsqrt_table, m, d, n_real);
    let d_f = Goldilocks::from_u64(n_real as u64);
    let half_d = Goldilocks::from_u64((n_real / 2) as u64);
    let half32 = Goldilocks::from_u64(1u64 << 31);
    let two32 = Goldilocks::from_u64(1u64 << 32);
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let ones_m: Vec<Goldilocks> = vec![Goldilocks::ONE; m];
    let ones_md: Vec<Goldilocks> = vec![Goldilocks::ONE; m * d];

    if !verify_virtual(
        &proof.trunc,
        &[
            (d_f, vec![0usize]),
            (Goldilocks::ONE, vec![1usize]),
            (neg, vec![2usize]),
            (neg * half_d, vec![3usize]),
        ],
        Goldilocks::ZERO,
        &proof.trunc_ch,
        &[
            mle::eval(&ms, &proof.trunc_ch),
            mle::eval(&rem_s, &proof.trunc_ch),
            mle::eval(&sum_sq, &proof.trunc_ch),
            mle::eval(&ones_m, &proof.trunc_ch),
        ],
    ) {
        return false;
    }
    if !verify_lookup_fractional(&proof.rstd, &s_index, &rstd, rsqrt_table, proof.alpha, proof.beta) {
        return false;
    }
    let rstd_b: Vec<Goldilocks> = (0..m * d).map(|ij| rstd[ij / d]).collect();
    let raw: Vec<Goldilocks> = (0..m * d).map(|ij| x[ij] * rstd[ij / d] * w[ij]).collect();
    if !verify_virtual(
        &proof.out,
        &[
            (Goldilocks::ONE, vec![0usize, 1usize, 2usize]),
            (neg, vec![3usize]),
            (Goldilocks::ONE, vec![3usize]),
            (neg * two32, vec![4usize]),
            (neg, vec![5usize]),
            (half32, vec![6usize]),
        ],
        Goldilocks::ZERO,
        &proof.r_out,
        &[
            mle::eval(x, &proof.r_out),
            mle::eval(&rstd_b, &proof.r_out),
            mle::eval(w, &proof.r_out),
            mle::eval(&raw, &proof.r_out),
            mle::eval(&out, &proof.r_out),
            mle::eval(&rem_out, &proof.r_out),
            mle::eval(&ones_md, &proof.r_out),
        ],
    ) {
        return false;
    }
    if !verify_virtual(
        &proof.rem_split,
        &[
            (Goldilocks::ONE, vec![0usize]),
            (neg, vec![1usize]),
            (neg * Goldilocks::from_u64(1u64 << 16), vec![2usize]),
        ],
        Goldilocks::ZERO,
        &proof.rem_split_ch,
        &[
            mle::eval(&rem_out, &proof.rem_split_ch),
            mle::eval(&rem_lo, &proof.rem_split_ch),
            mle::eval(&rem_hi, &proof.rem_split_ch),
        ],
    ) {
        return false;
    }
    let limb_table: Vec<Goldilocks> = (0..(1usize << 16)).map(|j| Goldilocks::from_u64(j as u64)).collect();
    let idx_lo: Vec<u32> = rem_lo.iter().map(|&v| to_i32(v) as u32).collect();
    let idx_hi: Vec<u32> = rem_hi.iter().map(|&v| to_i32(v) as u32).collect();
    verify_lookup_fractional(&proof.rem_lo_range, &idx_lo, &rem_lo, &limb_table, proof.alpha_lo, proof.beta_lo)
        && verify_lookup_fractional(&proof.rem_hi_range, &idx_hi, &rem_hi, &limb_table, proof.alpha_hi, proof.beta_hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_roundtrip() {
        let mut rng = XorShift64::new(0x1234_5678);
        let m = 4usize;
        let d = 8usize;
        let n_real = 5usize;
        let x: Vec<Goldilocks> = (0..m * d).map(|i| from_i64((i as i64 % 97) - 48)).collect();
        let w: Vec<Goldilocks> = (0..m * d).map(|i| from_i64((i as i64 % 13) + 1)).collect();
        let rsqrt_table: Vec<Goldilocks> = (0..64).map(|_| from_i64(65536)).collect();
        let (ms, _s_index, _rstd, out, _rem_out, _rem_lo, _rem_hi, _sum_sq, _rem_s) =
            rms_norm_forward(&x, &w, &rsqrt_table, m, d, n_real);
        let proof = prove_rms_norm(&x, &w, &rsqrt_table, m, d, n_real, &mut rng);
        assert!(verify_rms_norm(&proof, &x, &w, &rsqrt_table, m, d, n_real));
        assert_eq!(ms.len(), m);
        assert_eq!(out.len(), m * d);
    }
}
