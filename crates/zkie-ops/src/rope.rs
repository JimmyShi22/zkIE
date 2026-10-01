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

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_core::common::mle;
use zkie_core::common::sumcheck::{prove_virtual, verify_virtual, VirtualProof};

fn round_div(a: i64, b: i64) -> i64 {
    let q = a.div_euclid(b);
    let r = a.rem_euclid(b);
    if r * 2 >= b { q + 1 } else { q }
}

pub struct RoPEProof {
    pub first: VirtualProof,
    pub second: VirtualProof,
    pub r_first: Vec<Goldilocks>,
    pub r_second: Vec<Goldilocks>,
    pub r_out: Vec<Goldilocks>,
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
    rng: &mut XorShift64,
) -> RoPEProof {
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

    let r_f: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let first_terms = vec![
        (Goldilocks::ONE, vec![0usize, 1usize]),
        (neg, vec![2usize, 3usize]),
        (neg * two_shift, vec![4usize]),
        (neg, vec![5usize]),
        (half, vec![6usize]),
    ];
    let first_mles: Vec<&[Goldilocks]> = vec![&x_f, &cos_b, &x_s, &sin_b, &out_f, &rem_f, &ones];
    let first = prove_virtual(&first_mles, &first_terms, Goldilocks::ZERO, &r_f);

    let r_s: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let second_terms = vec![
        (Goldilocks::ONE, vec![0usize, 1usize]),
        (Goldilocks::ONE, vec![2usize, 3usize]),
        (neg * two_shift, vec![4usize]),
        (neg, vec![5usize]),
        (half, vec![6usize]),
    ];
    let second_mles: Vec<&[Goldilocks]> = vec![&x_s, &cos_b, &x_f, &sin_b, &out_s, &rem_s, &ones];
    let second = prove_virtual(&second_mles, &second_terms, Goldilocks::ZERO, &r_s);

    let r_out: Vec<Goldilocks> =
        (0..out.len().trailing_zeros() as usize).map(|_| rng.field()).collect();

    RoPEProof { first, second, r_first: r_f, r_second: r_s, r_out }
}

#[allow(clippy::too_many_arguments)]
pub fn verify_rope(
    proof: &RoPEProof,
    x: &[Goldilocks],
    cos: &[Goldilocks],
    sin: &[Goldilocks],
    out: &[Goldilocks],
    m: usize,
    d: usize,
    shift: u32,
) -> bool {
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
    let first_fe = vec![
        mle::eval(&x_f, &proof.r_first),
        mle::eval(&cos_b, &proof.r_first),
        mle::eval(&x_s, &proof.r_first),
        mle::eval(&sin_b, &proof.r_first),
        mle::eval(&out_f, &proof.r_first),
        mle::eval(&rem_f, &proof.r_first),
        mle::eval(&ones, &proof.r_first),
    ];
    if !verify_virtual(&proof.first, &first_terms, Goldilocks::ZERO, &proof.r_first, &first_fe) {
        return false;
    }

    let second_terms = vec![
        (Goldilocks::ONE, vec![0usize, 1usize]),
        (Goldilocks::ONE, vec![2usize, 3usize]),
        (neg * two_shift, vec![4usize]),
        (neg, vec![5usize]),
        (half, vec![6usize]),
    ];
    let second_fe = vec![
        mle::eval(&x_s, &proof.r_second),
        mle::eval(&cos_b, &proof.r_second),
        mle::eval(&x_f, &proof.r_second),
        mle::eval(&sin_b, &proof.r_second),
        mle::eval(&out_s, &proof.r_second),
        mle::eval(&rem_s, &proof.r_second),
        mle::eval(&ones, &proof.r_second),
    ];
    verify_virtual(&proof.second, &second_terms, Goldilocks::ZERO, &proof.r_second, &second_fe)
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
        let proof = prove_rope(&x, &cos, &sin, &out, m, d, shift, &mut rng);
        assert!(verify_rope(&proof, &x, &cos, &sin, &out, m, d, shift));
        assert_eq!(out.len(), m * heads * d);
    }
}
