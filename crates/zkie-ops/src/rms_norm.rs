//! RMSNorm (no centering): `out = trunc(x * rstd * w / 2^32)`, with
//! `ms = trunc(mean(x^2))` over `n_real`, and
//! `rstd = rsqrt_table[rsqrt_index(ms)]`. Mirrors `LayerNormCentered`'s
//! actually-verified chain: ms truncation + rsqrt lookup + out affine.

use crate::compose::{check_rsqrt_table, rsqrt_index};
use zkie_core::common::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64, ChallengeSource};
use zkie_core::common::fixed_point::{from_i64, to_i32, to_i64};
use zkie_core::common::logup_gkr::{
    prove_lookup_fractional_batch, verify_lookup_fractional_batch, FractionalProof,
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
pub struct RmsNormProof {
    pub trunc: VirtualProof,
    pub out: VirtualProof,
    pub rem_split: VirtualProof,
    pub trunc_ch: Vec<Goldilocks>,
    pub r_out: Vec<Goldilocks>,
    pub rem_split_ch: Vec<Goldilocks>,
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
    check_rsqrt_table(rsqrt_table, s_index.iter().copied().max().unwrap_or(0));
    let rstd: Vec<Goldilocks> = s_index.iter().map(|&i| rsqrt_table[i as usize]).collect();
    let raw: Vec<Goldilocks> = (0..m * d).map(|ij| x[ij] * rstd[ij / d] * w[ij]).collect();
    let out: Vec<Goldilocks> = (0..m * d)
        .map(|ij| from_i64(round_div(to_i64(raw[ij]), 1i64 << 32)))
        .collect();
    let rem_out: Vec<Goldilocks> = (0..m * d)
        .map(|ij| from_i64(to_i64(raw[ij]) - to_i64(out[ij]) * (1i64 << 32) + (1i64 << 31)))
        .collect();
    let rem_lo: Vec<Goldilocks> = rem_out
        .iter()
        .map(|&v| from_i64(to_i64(v) & 0xFFFF))
        .collect();
    let rem_hi: Vec<Goldilocks> = rem_out.iter().map(|&v| from_i64(to_i64(v) >> 16)).collect();
    let sum_sq: Vec<Goldilocks> = (0..m)
        .map(|r| (0..n_real).fold(Goldilocks::ZERO, |a, j| a + x[r * d + j] * x[r * d + j]))
        .collect();
    let rem_s: Vec<Goldilocks> = (0..m)
        .map(|r| from_i64(to_i64(sum_sq[r]) - to_i64(ms[r]) * n_real as i64 + (n_real as i64 / 2)))
        .collect();
    (
        ms, s_index, rstd, out, rem_out, rem_lo, rem_hi, sum_sq, rem_s,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn prove_rms_norm(
    x: &[Goldilocks],
    w: &[Goldilocks],
    rsqrt_table: &[Goldilocks],
    m: usize,
    d: usize,
    n_real: usize,
    transcript: &mut Transcript,
) -> (RmsNormProof, Vec<u32>, Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>) {
    let (ms, s_index, rstd, out, rem_out, rem_lo, rem_hi, sum_sq, rem_s) =
        rms_norm_forward(x, w, rsqrt_table, m, d, n_real);

    let d_f = Goldilocks::from_u64(n_real as u64);
    let half_d = Goldilocks::from_u64((n_real / 2) as u64);
    let half32 = Goldilocks::from_u64(1u64 << 31);
    let two32 = Goldilocks::from_u64(1u64 << 32);
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let ones_m: Vec<Goldilocks> = vec![Goldilocks::ONE; m];
    let ones_md: Vec<Goldilocks> = vec![Goldilocks::ONE; m * d];

    let trunc_terms = [
        (d_f, vec![0usize]),
        (Goldilocks::ONE, vec![1usize]),
        (neg, vec![2usize]),
        (neg * half_d, vec![3usize]),
    ];
    let (trunc_proof, trunc_ch) = prove_virtual(
        &[&ms, &rem_s, &sum_sq, &ones_m],
        &trunc_terms,
        Goldilocks::ZERO,
        transcript,
    );

    let rstd_b: Vec<Goldilocks> = (0..m * d).map(|ij| rstd[ij / d]).collect();
    let raw: Vec<Goldilocks> = (0..m * d).map(|ij| x[ij] * rstd[ij / d] * w[ij]).collect();
    // S5: EQ-WEIGHTED out relation (pointwise at the squeezed eq point;
    // the claim point r_out stays the sumcheck's derived vector).
    let q: Vec<Goldilocks> =
        transcript.field_vec((m * d).trailing_zeros() as usize);
    let eq = zkie_core::common::mle::eq_evals(&q);
    let out_terms = [
        (Goldilocks::ONE, vec![0usize, 1usize, 2usize, 3usize]),
        (neg, vec![0usize, 4usize]),
        (Goldilocks::ONE, vec![0usize, 4usize]),
        (neg * two32, vec![0usize, 5usize]),
        (neg, vec![0usize, 6usize]),
        (half32, vec![0usize, 7usize]),
    ];
    let (out_proof, r_out) = prove_virtual(
        &[&eq, x, &rstd_b, w, &raw, &out, &rem_out, &ones_md],
        &out_terms,
        Goldilocks::ZERO,
        transcript,
    );

    let rem_split_terms = [
        (Goldilocks::ONE, vec![0usize]),
        (neg, vec![1usize]),
        (neg * Goldilocks::from_u64(1u64 << 16), vec![2usize]),
    ];
    let (rem_split, rem_split_ch) = prove_virtual(
        &[&rem_out, &rem_lo, &rem_hi],
        &rem_split_terms,
        Goldilocks::ZERO,
        transcript,
    );
    (
        RmsNormProof {
            trunc: trunc_proof,
            out: out_proof,
            rem_split,
            trunc_ch,
            r_out,
            rem_split_ch,
        },
        s_index,
        rstd,
        rem_lo,
        rem_hi,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn verify_rms_norm(
    proof: &RmsNormProof,
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

    let trunc_terms = [
        (d_f, vec![0usize]),
        (Goldilocks::ONE, vec![1usize]),
        (neg, vec![2usize]),
        (neg * half_d, vec![3usize]),
    ];
    let (ok, r, fc) = verify_virtual(
        &proof.trunc,
        &trunc_terms,
        Goldilocks::ZERO,
        m.trailing_zeros() as usize,
        transcript,
    );
    if !ok || r != proof.trunc_ch || fc != virtual_eval(&trunc_terms, &proof.trunc.final_evals) {
        return false;
    }

    let arity_md = (m * d).trailing_zeros() as usize;
    let q_s: Vec<Goldilocks> = transcript.field_vec(arity_md);
    let out_terms = [
        (Goldilocks::ONE, vec![0usize, 1usize, 2usize, 3usize]),
        (neg, vec![0usize, 4usize]),
        (Goldilocks::ONE, vec![0usize, 4usize]),
        (neg * two32, vec![0usize, 5usize]),
        (neg, vec![0usize, 6usize]),
        (half32, vec![0usize, 7usize]),
    ];
    let (ok, r, fc) = verify_virtual(
        &proof.out,
        &out_terms,
        Goldilocks::ZERO,
        arity_md,
        transcript,
    );
    if !ok
        || r != proof.r_out
        || proof.out.final_evals[0] != zkie_core::common::mle::eq_poly(&q_s, &r)
        || fc != virtual_eval(&out_terms, &proof.out.final_evals)
    {
        return false;
    }

    let rem_split_terms = [
        (Goldilocks::ONE, vec![0usize]),
        (neg, vec![1usize]),
        (neg * Goldilocks::from_u64(1u64 << 16), vec![2usize]),
    ];
    let (ok, r, fc) = verify_virtual(
        &proof.rem_split,
        &rem_split_terms,
        Goldilocks::ZERO,
        arity_md,
        transcript,
    );
    ok && r == proof.rem_split_ch && fc == virtual_eval(&rem_split_terms, &proof.rem_split.final_evals)
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
        let mut tp = zkie_core::common::transcript::Transcript::new(b"zkie-rmsnorm-test");
        tp.absorb_u64(0x1234);
        let (proof, _ridx, _rvals, _rem_lo, _rem_hi) =
            prove_rms_norm(&x, &w, &rsqrt_table, m, d, n_real, &mut tp);
        let mut tv = zkie_core::common::transcript::Transcript::new(b"zkie-rmsnorm-test");
        tv.absorb_u64(0x1234);
        assert!(verify_rms_norm(&proof, m, d, n_real, &mut tv));
        assert_eq!(ms.len(), m);
        assert_eq!(out.len(), m * d);
    }
}
