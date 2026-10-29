//! Rotary position embedding (RoPE) for a multi-head query/key tensor, using
//! the HuggingFace "rotate half" convention (as used by Gemma 3).
//!
//! The input `x` is laid out as `[m, heads, d]` (row-major, head-major, `d`
//! even). For position `p`, head `h` and `i = 0..d/2` the elements
//! `x[p,h,i]` and `x[p,h,i + d/2]` are rotated by the angle encoded in
//! `cos[p,i]` / `sin[p,i]`:
//!
//!   out_first  = x_first*cos - x_second*sin
//!   out_second = x_second*cos + x_first*sin
//!
//! `cos`/`sin` are `[m, d/2]` fixed-point tables shared across heads. The
//! output keeps the natural layout, so it can feed the attention matmul.

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64, ChallengeSource};
use zkie_core::common::fixed_point::{from_i64, to_i32, to_i64};
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
pub struct RoPEProof {
    pub first: VirtualProof,
    pub second: VirtualProof,
    pub r_first: Vec<Goldilocks>,
    pub r_second: Vec<Goldilocks>,
}

/// The two authenticated claim points on the full `out` tensor (S2): the
/// half-selector bit (position `log2(d/2)`) is 0 resp. 1, every other bit
/// comes from the first resp. second half-sumcheck's challenge vector. `out`
/// at these points equals `first.final_evals[4]` resp. `second.final_evals[4]`
/// — this replaces the old unverifiable `r_out` claim, whose point was
/// squeezed but never tied to either sumcheck.
pub fn rope_out_points(
    r_f: &[Goldilocks],
    r_s: &[Goldilocks],
    d: usize,
) -> (Vec<Goldilocks>, Vec<Goldilocks>) {
    let half_log = (d / 2).trailing_zeros() as usize;
    assert!(half_log <= r_f.len());
    let insert = |r: &[Goldilocks], bit: Goldilocks| -> Vec<Goldilocks> {
        let mut p = Vec::with_capacity(r.len() + 1);
        p.extend_from_slice(&r[..half_log]);
        p.push(bit);
        p.extend_from_slice(&r[half_log..]);
        p
    };
    (insert(r_f, Goldilocks::ZERO), insert(r_s, Goldilocks::ONE))
}

fn heads_of(x_len: usize, m: usize, d: usize) -> usize {
    assert!(d % 2 == 0, "head dim must be even");
    assert_eq!(x_len % (m * d), 0, "tensor not divisible into heads");
    x_len / (m * d)
}

fn extract_first(x: &[Goldilocks], m: usize, d: usize, heads: usize) -> Vec<Goldilocks> {
    let half = d / 2;
    let mut out = Vec::with_capacity(m * heads * half);
    for p in 0..m {
        for h in 0..heads {
            let base = p * heads * d + h * d;
            for i in 0..half {
                out.push(x[base + i]);
            }
        }
    }
    out
}

fn extract_second(x: &[Goldilocks], m: usize, d: usize, heads: usize) -> Vec<Goldilocks> {
    let half = d / 2;
    let mut out = Vec::with_capacity(m * heads * half);
    for p in 0..m {
        for h in 0..heads {
            let base = p * heads * d + h * d + half;
            for i in 0..half {
                out.push(x[base + i]);
            }
        }
    }
    out
}

fn broadcast_pair(table: &[Goldilocks], m: usize, d: usize, heads: usize) -> Vec<Goldilocks> {
    let half = d / 2;
    let mut out = Vec::with_capacity(m * heads * half);
    for p in 0..m {
        for _h in 0..heads {
            for i in 0..half {
                out.push(table[p * half + i]);
            }
        }
    }
    out
}

#[allow(clippy::type_complexity)]
pub fn rope_forward(
    x: &[Goldilocks],
    cos: &[Goldilocks],
    sin: &[Goldilocks],
    m: usize,
    d: usize,
    shift: u32,
) -> (Vec<Goldilocks>, Vec<Goldilocks>, Vec<Goldilocks>) {
    let heads = heads_of(x.len(), m, d);
    let half = d / 2;
    let two_shift = 1i64 << shift;
    let half_shift = 1i64 << (shift - 1);
    let mut out = vec![Goldilocks::ZERO; x.len()];
    let mut rem_f = vec![Goldilocks::ZERO; m * heads * half];
    let mut rem_s = vec![Goldilocks::ZERO; m * heads * half];
    for p in 0..m {
        for h in 0..heads {
            let base = p * heads * d + h * d;
            for i in 0..half {
                let xf = to_i64(x[base + i]);
                let xs = to_i64(x[base + half + i]);
                let c = to_i64(cos[p * half + i]);
                let s = to_i64(sin[p * half + i]);
                let raw_f = xf * c - xs * s;
                let raw_s = xs * c + xf * s;
                let of = round_div(raw_f, two_shift);
                let os = round_div(raw_s, two_shift);
                out[base + i] = from_i64(of);
                out[base + half + i] = from_i64(os);
                let k = (p * heads + h) * half + i;
                rem_f[k] = from_i64(raw_f - of * two_shift + half_shift);
                rem_s[k] = from_i64(raw_s - os * two_shift + half_shift);
            }
        }
    }
    (out, rem_f, rem_s)
}

#[allow(clippy::too_many_arguments)]
pub fn prove_rope(
    x: &[Goldilocks],
    cos: &[Goldilocks],
    sin: &[Goldilocks],
    out: &[Goldilocks],
    m: usize,
    d: usize,
    shift: u32,
    transcript: &mut Transcript,
) -> (RoPEProof, Vec<Goldilocks>, Vec<Goldilocks>) {
    let heads = heads_of(x.len(), m, d);
    let (_out, rem_f, rem_s) = rope_forward(x, cos, sin, m, d, shift);
    let x_f = extract_first(x, m, d, heads);
    let x_s = extract_second(x, m, d, heads);
    let cos_b = broadcast_pair(cos, m, d, heads);
    let sin_b = broadcast_pair(sin, m, d, heads);
    let out_f = extract_first(out, m, d, heads);
    let out_s = extract_second(out, m, d, heads);

    let n = m * heads * (d / 2);
    let ones = vec![Goldilocks::ONE; n];
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let two_shift = Goldilocks::from_u64(1u64 << shift);
    let half = Goldilocks::from_u64(1u64 << (shift - 1));

    let first_terms = vec![
        (Goldilocks::ONE, vec![0usize, 1usize]),
        (neg, vec![2usize, 3usize]),
        (neg * two_shift, vec![4usize]),
        (neg, vec![5usize]),
        (half, vec![6usize]),
    ];
    let first_mles: Vec<&[Goldilocks]> = vec![&x_f, &cos_b, &x_s, &sin_b, &out_f, &rem_f, &ones];
    let (first, r_f) = prove_virtual(&first_mles, &first_terms, Goldilocks::ZERO, transcript);

    let second_terms = vec![
        (Goldilocks::ONE, vec![0usize, 1usize]),
        (Goldilocks::ONE, vec![2usize, 3usize]),
        (neg * two_shift, vec![4usize]),
        (neg, vec![5usize]),
        (half, vec![6usize]),
    ];
    let second_mles: Vec<&[Goldilocks]> = vec![&x_s, &cos_b, &x_f, &sin_b, &out_s, &rem_s, &ones];
    let (second, r_s) = prove_virtual(&second_mles, &second_terms, Goldilocks::ZERO, transcript);

    // S2: no r_out squeeze — the out tensor is claimed at the two half points
    // built by [`rope_out_points`], whose values are the checked sumcheck
    // final evals (first.final_evals[4], second.final_evals[4]).
    (
        RoPEProof {
            first,
            second,
            r_first: r_f,
            r_second: r_s,
        },
        rem_f,
        rem_s,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn verify_rope(proof: &RoPEProof, shift: u32, transcript: &mut Transcript) -> bool {
    let neg = Goldilocks::ZERO - Goldilocks::ONE;
    let two_shift = Goldilocks::from_u64(1u64 << shift);
    let half = Goldilocks::from_u64(1u64 << (shift - 1));

    // The half-tensor arity is the length of the stored challenge vectors
    // (checked against the proof's round count below). The claim points on
    // `out` are built by the caller from the statement's `d` via
    // [`rope_out_points`], and their length is pinned against the other
    // claims on `out` by the same_poly binding.
    let arity = proof.r_first.len();

    let first_terms = vec![
        (Goldilocks::ONE, vec![0usize, 1usize]),
        (neg, vec![2usize, 3usize]),
        (neg * two_shift, vec![4usize]),
        (neg, vec![5usize]),
        (half, vec![6usize]),
    ];
    let (ok, r_f, fc) = verify_virtual(
        &proof.first,
        &first_terms,
        Goldilocks::ZERO,
        arity,
        transcript,
    );
    if !ok || r_f != proof.r_first || fc != virtual_eval(&first_terms, &proof.first.final_evals) {
        return false;
    }

    let second_terms = vec![
        (Goldilocks::ONE, vec![0usize, 1usize]),
        (Goldilocks::ONE, vec![2usize, 3usize]),
        (neg * two_shift, vec![4usize]),
        (neg, vec![5usize]),
        (half, vec![6usize]),
    ];
    let (ok, r_s, fc) = verify_virtual(
        &proof.second,
        &second_terms,
        Goldilocks::ZERO,
        arity,
        transcript,
    );
    ok && r_s == proof.r_second && fc == virtual_eval(&second_terms, &proof.second.final_evals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_roundtrip() {
        let mut rng = XorShift64::new(0x0D0D_0001);
        let m = 4usize;
        let heads = 2usize;
        let d = 8usize;
        let shift = 16u32;
        let x: Vec<Goldilocks> = (0..m * heads * d)
            .map(|i| from_i64((i as i64 % 97) - 48))
            .collect();
        let half = d / 2;
        let cos: Vec<Goldilocks> = (0..m * half).map(|_| from_i64(65536)).collect();
        let sin: Vec<Goldilocks> = (0..m * half).map(|_| from_i64(0)).collect();
        let (out, _rf, _rs) = rope_forward(&x, &cos, &sin, m, d, shift);
        let mut tp = Transcript::new(b"zkie-rope-test");
        tp.absorb_u64(0x0D0D);
        let (proof, _rem_f, _rem_s) = prove_rope(&x, &cos, &sin, &out, m, d, shift, &mut tp);
        let mut tv = Transcript::new(b"zkie-rope-test");
        tv.absorb_u64(0x0D0D);
        assert!(verify_rope(&proof, shift, &mut tv));
        assert_eq!(out.len(), m * heads * d);
    }
}
