//! Proof for the fixed-point `Scale` op: `out = round(x * factor / 2^shift)`.
//!
//! Mirrors `Projection`'s remainder handling: the prover commits to the offset
//! rounding remainder `rem_off = x*factor - out*2^shift + 2^(shift-1)`, proves
//! the affine relation with one virtual sumcheck, and proves `rem_off` lies in
//! `[0, 2^shift)` with a fractional lookup against the limb table.

use zkie_core::common::field::{ChallengeSource, Goldilocks, PrimeCharacteristicRing};
use zkie_core::common::fixed_point::{from_i64, to_i32, to_i64};
use zkie_core::common::logup_gkr::{
    prove_lookup_fractional, verify_lookup_fractional_chain, FractionalProof,
};
use zkie_core::common::sumcheck::{prove_virtual, verify_virtual, VirtualProof};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ScaleProof {
    pub affine: VirtualProof,
    pub frac: FractionalProof,
    pub alpha: Goldilocks,
    pub beta: Goldilocks,
    pub r: Vec<Goldilocks>,
}

/// Proof for elementwise `ScaleVec`: `out[i] = round(x[i] * scale[i] / 2^shift)`.
/// Same affine + range-check structure as `Scale`, but the multiplier is a
/// tensor, so the affine relation is a degree-2 product.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ScaleVecProof {
    pub affine: VirtualProof,
    pub frac: FractionalProof,
    pub alpha: Goldilocks,
    pub beta: Goldilocks,
    pub r: Vec<Goldilocks>,
}

pub fn scale_forward(x: &[Goldilocks], factor: i64, shift: u32) -> Vec<Goldilocks> {
    x.iter()
        .map(|&v| from_i64(round_div(to_i64(v) * factor, 1i64 << shift)))
        .collect()
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

pub fn prove_scale(
    x: &[Goldilocks],
    out: &[Goldilocks],
    factor: i64,
    shift: u32,
    rng: &mut impl ChallengeSource,
) -> ScaleProof {
    assert_eq!(x.len(), out.len());
    assert!(x.len().is_power_of_two(), "Scale length must be a power of two");
    let half = Goldilocks::from_u64(1u64 << (shift - 1));
    let two_shift = Goldilocks::from_u64(1u64 << shift);
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let f = from_i64(factor);

    // Offset remainder: `x*factor - out*2^shift + 2^(shift-1)` in [0, 2^shift).
    let rem_off: Vec<Goldilocks> = x
        .iter()
        .zip(out.iter())
        .map(|(&xv, &ov)| {
            let rem = to_i64(xv) * factor - to_i64(ov) * (1i64 << shift);
            from_i64(rem + (1i64 << (shift - 1)))
        })
        .collect();

    let r: Vec<Goldilocks> = (0..x.len().trailing_zeros() as usize)
        .map(|_| rng.field())
        .collect();
    let terms = vec![
        (f, vec![0usize]),
        (neg * two_shift, vec![1usize]),
        (neg, vec![2usize]),
    ];
    let mles: Vec<&[Goldilocks]> = vec![x, out, &rem_off];
    // The combined polynomial `f*x - 2^shift*out - rem_off` is pointwise
    // `-half`, so its total sum is `-half * len`.
    let claimed = (Goldilocks::ZERO - half) * Goldilocks::from_u64(x.len() as u64);
    let affine = prove_virtual(&mles, &terms, claimed, &r);

    let table: Vec<Goldilocks> = (0..(1usize << shift))
        .map(|j| Goldilocks::from_u64(j as u64))
        .collect();
    let idx: Vec<u32> = rem_off.iter().map(|&v| to_i32(v) as u32).collect();
    let alpha = rng.field();
    let beta = rng.field();
    let frac = prove_lookup_fractional(&idx, &rem_off, &table, alpha, beta, rng);
    ScaleProof { affine, frac, alpha, beta, r }
}

pub fn verify_scale(proof: &ScaleProof, factor: i64, shift: u32) -> bool {
    let half = Goldilocks::from_u64(1u64 << (shift - 1));
    let two_shift = Goldilocks::from_u64(1u64 << shift);
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let f = from_i64(factor);
    let terms = vec![
        (f, vec![0usize]),
        (neg * two_shift, vec![1usize]),
        (neg, vec![2usize]),
    ];
    if !verify_virtual(
        &proof.affine,
        &terms,
        (Goldilocks::ZERO - half) * Goldilocks::from_u64(1u64 << proof.r.len()),
        &proof.r,
        &proof.affine.final_evals,
    ) {
        return false;
    }
    verify_lookup_fractional_chain(&proof.frac)
}

pub fn prove_scale_vec(
    x: &[Goldilocks],
    scale: &[Goldilocks],
    out: &[Goldilocks],
    shift: u32,
    rng: &mut impl ChallengeSource,
) -> ScaleVecProof {
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len(), scale.len(), "ScaleVec proof requires same-length scale");
    assert!(x.len().is_power_of_two(), "ScaleVec length must be a power of two");
    let half = Goldilocks::from_u64(1u64 << (shift - 1));
    let two_shift = Goldilocks::from_u64(1u64 << shift);
    let neg = Goldilocks::ZERO - Goldilocks::ONE;

    let rem_off: Vec<Goldilocks> = x
        .iter()
        .zip(scale.iter())
        .zip(out.iter())
        .map(|((&xv, &sv), &ov)| {
            let rem = to_i64(xv) * to_i64(sv) - to_i64(ov) * (1i64 << shift);
            from_i64(rem + (1i64 << (shift - 1)))
        })
        .collect();

    let r: Vec<Goldilocks> = (0..x.len().trailing_zeros() as usize)
        .map(|_| rng.field())
        .collect();
    let terms = vec![
        (Goldilocks::ONE, vec![0usize, 1usize]),
        (neg * two_shift, vec![2usize]),
        (neg, vec![3usize]),
    ];
    let mles: Vec<&[Goldilocks]> = vec![scale, x, out, &rem_off];
    let claimed = (Goldilocks::ZERO - half) * Goldilocks::from_u64(x.len() as u64);
    let affine = prove_virtual(&mles, &terms, claimed, &r);

    let table: Vec<Goldilocks> = (0..(1usize << shift))
        .map(|j| Goldilocks::from_u64(j as u64))
        .collect();
    let idx: Vec<u32> = rem_off.iter().map(|&v| to_i32(v) as u32).collect();
    let alpha = rng.field();
    let beta = rng.field();
    let frac = prove_lookup_fractional(&idx, &rem_off, &table, alpha, beta, rng);
    ScaleVecProof { affine, frac, alpha, beta, r }
}

pub fn verify_scale_vec(proof: &ScaleVecProof, shift: u32) -> bool {
    let half = Goldilocks::from_u64(1u64 << (shift - 1));
    let two_shift = Goldilocks::from_u64(1u64 << shift);
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let terms = vec![
        (Goldilocks::ONE, vec![0usize, 1usize]),
        (neg * two_shift, vec![2usize]),
        (neg, vec![3usize]),
    ];
    if !verify_virtual(
        &proof.affine,
        &terms,
        (Goldilocks::ZERO - half) * Goldilocks::from_u64(1u64 << proof.r.len()),
        &proof.r,
        &proof.affine.final_evals,
    ) {
        return false;
    }
    verify_lookup_fractional_chain(&proof.frac)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_core::common::field::XorShift64;

    #[test]
    fn scale_roundtrip() {
        let mut rng = XorShift64::new(0x5CA1E);
        let x: Vec<Goldilocks> = (0..8).map(|i| from_i64((i as i64 - 4) * 12345)).collect();
        let out = scale_forward(&x, 98304, 16);
        let proof = prove_scale(&x, &out, 98304, 16, &mut rng);
        assert!(verify_scale(&proof, 98304, 16));
    }

    #[test]
    fn scale_vec_roundtrip() {
        let mut rng = XorShift64::new(0x5CA1F);
        let x: Vec<Goldilocks> = (0..8).map(|i| from_i64((i as i64 - 4) * 7000)).collect();
        let scale: Vec<Goldilocks> = (0..8).map(|i| from_i64((i as i64 + 1) * 1000)).collect();
        let out: Vec<Goldilocks> = x
            .iter()
            .zip(scale.iter())
            .map(|(&xv, &sv)| from_i64(round_div(to_i64(xv) * to_i64(sv), 1i64 << 16)))
            .collect();
        let proof = prove_scale_vec(&x, &scale, &out, 16, &mut rng);
        assert!(verify_scale_vec(&proof, 16));
    }
}
