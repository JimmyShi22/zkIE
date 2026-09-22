//! WHIR-committed GKR matmul for `m == 1` (a vector left operand).
//!
//! This is the core of the interpreter: commit a tensor once, then prove
//! `C = A @ B` from prescribed-point openings rather than raw evaluations. The
//! ctx32 TimesFM model runs every layer with batch/sequence `m == 1`, so the
//! left operand is a vector and needs no transpose point-swap.

use crate::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::fixed_point::{from_i32, from_i64, to_i32, to_i64};
use crate::lookup;
use crate::whir::{Commitment, OpeningProtocol, ProverData, Whir};
use crate::{matmul, mle, sumcheck};

/// A committed tensor (commitment + prover data + opening protocol).
pub struct Committed {
    pub commitment: Commitment,
    pub prover_data: ProverData,
    pub protocol: OpeningProtocol,
}

pub fn commit(whir: &Whir, values: &[Goldilocks]) -> Committed {
    let (commitment, prover_data, protocol) = whir.commit(values);
    Committed {
        commitment,
        prover_data,
        protocol,
    }
}

/// Prove `C = A @ B` with `A` of shape `1 x k`, `B` of shape `k x n`,
/// `C` of shape `1 x n`, all committed. Returns `true` iff every opening and
/// the sum-check verify.
#[allow(clippy::too_many_arguments)]
pub fn prove_matmul(
    whir_a: &Whir,
    a: &Committed,
    whir_b: &Whir,
    b: &Committed,
    whir_c: &Whir,
    c: &Committed,
    a_mat: &[Goldilocks],
    b_mat: &[Goldilocks],
    c_mat: &[Goldilocks],
    k: usize,
    n: usize,
    rng: &mut XorShift64,
) -> bool {
    let ch: Vec<Goldilocks> = (0..k.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let proof = matmul::prove(a_mat, b_mat, c_mat, 1, k, n, &[], &v, &ch);

    let (a_open, f) = whir_a.open(a.prover_data.clone(), &a.protocol, &ch);
    let mut bp = v.clone();
    bp.extend_from_slice(&ch);
    let (b_open, h) = whir_b.open(b.prover_data.clone(), &b.protocol, &bp);
    let (c_open, claimed) = whir_c.open(c.prover_data.clone(), &c.protocol, &v);

    let f_ok = whir_a.verify(&a.commitment, &a_open, &a.protocol, &ch).unwrap() == f;
    let h_ok = whir_b.verify(&b.commitment, &b_open, &b.protocol, &bp).unwrap() == h;
    let c_ok = whir_c.verify(&c.commitment, &c_open, &c.protocol, &v).unwrap() == claimed;
    let evals_ok = f == mle::eval(a_mat, &ch)
        && h == mle::eval(b_mat, &bp)
        && claimed == mle::eval(c_mat, &v);
    f_ok && h_ok && c_ok && evals_ok && claimed == proof.claimed && matmul::verify(&proof, &ch, f, h)
}

/// Prove `outputs[i] == table[indices[i]]` against WHIR commitments, in the
/// O(N)-opening PoC form: open the committed index and output columns at every
/// hypercube point and recompute the LogUp left-hand side from those bound
/// values. `x` holds the indices embedded as field values, `y` the outputs.
pub fn prove_lookup(
    whir_x: &Whir,
    x: &Committed,
    whir_y: &Whir,
    y: &Committed,
    indices: &[u32],
    table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
) -> bool {
    let n = indices.len();
    let d = n.trailing_zeros() as usize;
    let mut lhs = Goldilocks::ZERO;
    for i in 0..n {
        let point: Vec<Goldilocks> = (0..d)
            .map(|b| Goldilocks::from_bool((i >> b) & 1 == 1))
            .collect();
        let (x_open, xv) = whir_x.open(x.prover_data.clone(), &x.protocol, &point);
        let (y_open, yv) = whir_y.open(y.prover_data.clone(), &y.protocol, &point);
        if whir_x.verify(&x.commitment, &x_open, &x.protocol, &point).unwrap() != xv {
            return false;
        }
        if whir_y.verify(&y.commitment, &y_open, &y.protocol, &point).unwrap() != yv {
            return false;
        }
        let key = xv + beta * yv;
        lhs = lhs + (alpha + key).inverse();
    }

    let mut m = vec![Goldilocks::ZERO; table.len()];
    for &i in indices {
        m[i as usize] = m[i as usize] + Goldilocks::ONE;
    }
    let rhs = table
        .iter()
        .enumerate()
        .fold(Goldilocks::ZERO, |acc, (j, &t)| {
            let tkey = Goldilocks::from_u64(j as u64) + beta * t;
            acc + m[j] * (alpha + tkey).inverse()
        });
    lhs == rhs
}

/// Round `a / b` to the nearest integer, ties toward +inf (round half up), for
/// the signed integer reductions below. `b` must be positive. This convention
/// gives a remainder in `[-b/2, b/2)`, which the sub-linear round range-check
/// relies on.
fn div_round(a: i64, b: i64) -> i64 {
    debug_assert!(b > 0);
    let q = a.div_euclid(b);
    let r = a.rem_euclid(b);
    if r * 2 >= b {
        q + 1
    } else {
        q
    }
}

/// Derive the quantized LayerNorm scalars from the true `n_real` signed i32
/// inputs (ignoring the power-of-two padding tail): `mean` at scale 2^16 and the
/// `rsqrt_table` index for `1/sqrt(var + eps)`. The variance is accumulated at
/// scale 2^32, then the index is taken at scale 2^14 (`var_q >> 18`).
fn layer_norm_scalars_i32(x_i32: &[i32], n_real: usize) -> (i32, u32) {
    let sum: i64 = x_i32[..n_real].iter().map(|&v| v as i64).sum();
    let mean = div_round(sum, n_real as i64) as i32;
    let sqsum: i64 = x_i32[..n_real]
        .iter()
        .map(|&v| {
            let d = v as i64 - mean as i64;
            d * d
        })
        .sum();
    let var = div_round(sqsum, n_real as i64);
    let s_index = div_round(var, 1 << 18) as u32;
    (mean, s_index)
}

/// Compute the raw LayerNorm product `raw = (x - mean) * rstd * w` (scale 2^48,
/// before the two `2^16` rescales) together with the scalars, so a caller can
/// commit `raw` and then run [`prove_layer_norm`] against it. `mean` is derived
/// from `x` (not trusted), and `rstd` is read from `rsqrt_table` at the index of
/// `var + eps`, so the caller's raw and the proof share one consistent protocol.
pub fn layer_norm_raw(
    x: &[Goldilocks],
    weight: &[Goldilocks],
    n_real: usize,
    rsqrt_table: &[Goldilocks],
) -> (Vec<Goldilocks>, Goldilocks, Goldilocks, u32) {
    assert_eq!(x.len(), weight.len());
    let x_i32: Vec<i32> = x.iter().map(|&v| to_i32(v)).collect();
    let (mean, s_index) = layer_norm_scalars_i32(&x_i32, n_real);
    assert!(
        (s_index as usize) < rsqrt_table.len(),
        "var out of rsqrt table range"
    );
    let mean_f = from_i32(mean);
    let rstd = rsqrt_table[s_index as usize];
    let raw = x
        .iter()
        .zip(weight)
        .map(|(&xv, &w)| (xv - mean_f) * rstd * w)
        .collect();
    (raw, mean_f, rstd, s_index)
}

/// Prove the raw LayerNorm product `raw = (x - mean) * rstd * w` (scale 2^48,
/// before the two `2^16` rescales) against WHIR commitments, in the O(N)-opening
/// PoC form. `mean` and `var` are recomputed from the *committed* `x` (integer
/// reduction over the first `n_real` entries, ignoring padding) via a pair of
/// degree-2 sum-checks, `rstd = 1/sqrt(var + eps)` is bound by a LogUp lookup,
/// and the pointwise product `raw = rstd (x - mean) w` is proven by a degree-3
/// zero-check sum-check — the prover runs O(N) field work but the verifier opens
/// `x`/`raw` at only a constant number of random points.
#[allow(clippy::too_many_arguments)]
pub fn prove_layer_norm(
    whir_x: &Whir,
    x: &Committed,
    x_plain: &[Goldilocks],
    whir_raw: &Whir,
    raw: &Committed,
    raw_plain: &[Goldilocks],
    weight: &[Goldilocks],
    n_real: usize,
    rsqrt_table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
    rng: &mut XorShift64,
) -> bool {
    let n = weight.len();
    let d = n.trailing_zeros() as usize;
    assert_eq!(x_plain.len(), n);
    assert_eq!(raw_plain.len(), n);
    let ones: Vec<Goldilocks> = vec![Goldilocks::ONE; n];

    // mean: S_x = sum x(i) via degree-2 sum-check (f = x, h = 1).
    let s_x: i64 = x_plain[..n_real].iter().map(|&v| to_i32(v) as i64).sum();
    let s_x_f = from_i64(s_x);
    let r1: Vec<Goldilocks> = (0..d).map(|_| rng.field()).collect();
    let proof_x = sumcheck::prove(x_plain, &ones, s_x_f, &r1);
    let (x_open1, x_r1) = whir_x.open(x.prover_data.clone(), &x.protocol, &r1);
    if whir_x.verify(&x.commitment, &x_open1, &x.protocol, &r1).unwrap() != x_r1 {
        return false;
    }
    if !sumcheck::verify(&proof_x, s_x_f, &r1, x_r1, Goldilocks::ONE) {
        return false;
    }
    let mean = div_round(s_x, n_real as i64) as i32;
    let mean_f = from_i32(mean);

    // var: S_x2 = sum x(i)^2 via degree-2 sum-check (f = x, h = x); then
    // var = E[x^2] - mean^2 (padding is zero, so the full sum equals the real sum).
    let s_x2: i64 = x_plain[..n_real]
        .iter()
        .map(|&v| {
            let d = to_i32(v) as i64;
            d * d
        })
        .sum();
    let s_x2_f = from_i64(s_x2);
    let r2: Vec<Goldilocks> = (0..d).map(|_| rng.field()).collect();
    let proof_x2 = sumcheck::prove(x_plain, x_plain, s_x2_f, &r2);
    let (x_open2, x_r2) = whir_x.open(x.prover_data.clone(), &x.protocol, &r2);
    if whir_x.verify(&x.commitment, &x_open2, &x.protocol, &r2).unwrap() != x_r2 {
        return false;
    }
    if !sumcheck::verify(&proof_x2, s_x2_f, &r2, x_r2, x_r2) {
        return false;
    }
    // sqsum = sum (x_i - mean)^2 = S_x2 - 2 mean S_x + n_real mean^2.
    let m = mean as i64;
    let sqsum = s_x2 - 2 * m * s_x + (n_real as i64) * m * m;
    let var = div_round(sqsum, n_real as i64);
    let s_index = div_round(var, 1 << 18) as u32;
    if (s_index as usize) >= rsqrt_table.len() {
        return false;
    }
    let rstd = rsqrt_table[s_index as usize];
    let lk = lookup::prove(&[s_index], &[rstd], rsqrt_table, alpha, beta);
    if !lookup::verify(&lk) {
        return false;
    }

    // raw = rstd (x - mean) w via a degree-3 zero-check:
    //   sum eq(x,r) raw(x) == rstd (sum eq(x,r) x(x) w(x) - mean sum eq(x,r) w(x)).
    let r: Vec<Goldilocks> = (0..d).map(|_| rng.field()).collect();
    let s = mle::eq_evals(&r);
    let c_raw: Goldilocks = s.iter().zip(raw_plain).fold(Goldilocks::ZERO, |a, (&si, &ri)| a + si * ri);
    let c_xw: Goldilocks = s
        .iter()
        .zip(x_plain)
        .zip(weight)
        .fold(Goldilocks::ZERO, |a, ((&si, &xi), &wi)| a + si * xi * wi);
    let c_w: Goldilocks = s.iter().zip(weight).fold(Goldilocks::ZERO, |a, (&si, &wi)| a + si * wi);

    let proof_raw = sumcheck::prove(&s, raw_plain, c_raw, &r);
    let proof_xw = sumcheck::prove3(&s, x_plain, weight, c_xw, &r);
    let proof_w = sumcheck::prove(&s, weight, c_w, &r);

    let (x_open, xr) = whir_x.open(x.prover_data.clone(), &x.protocol, &r);
    let (raw_open, rv) = whir_raw.open(raw.prover_data.clone(), &raw.protocol, &r);
    if whir_x.verify(&x.commitment, &x_open, &x.protocol, &r).unwrap() != xr {
        return false;
    }
    if whir_raw.verify(&raw.commitment, &raw_open, &raw.protocol, &r).unwrap() != rv {
        return false;
    }
    let s_r = mle::eval(&s, &r);
    let w_r = mle::eval(weight, &r);
    if !sumcheck::verify(&proof_raw, c_raw, &r, s_r, rv) {
        return false;
    }
    if !sumcheck::verify3(&proof_xw, c_xw, &r, s_r, xr, w_r) {
        return false;
    }
    if !sumcheck::verify(&proof_w, c_w, &r, s_r, w_r) {
        return false;
    }
    if c_raw != rstd * (c_xw - mean_f * c_w) {
        return false;
    }
    true
}

/// Derive the quantized RMSNorm rsqrt table index from the true `n_real` signed
/// i32 inputs (ignoring padding): `s = mean(x^2)` at scale 2^32, index at scale
/// 2^14 (`s >> 18`). RMSNorm has no mean subtraction, so this is a pure
/// sum-of-squares reduction — the same single non-arithmetic lookup as LayerNorm.
fn rms_norm_scalars_i32(x_i32: &[i32], n_real: usize) -> u32 {
    let sqsum: i64 = x_i32[..n_real]
        .iter()
        .map(|&v| {
            let d = v as i64;
            d * d
        })
        .sum();
    let s = div_round(sqsum, n_real as i64);
    div_round(s, 1 << 18) as u32
}

/// Compute the raw RMSNorm product `raw = x * rstd * w` (scale 2^48) together
/// with the rsqrt value and index, so a caller can commit `raw` and then run
/// [`prove_rms_norm`]. `rstd` is read from `rsqrt_table` at the index of
/// `mean(x^2) + eps` (not trusted).
pub fn rms_norm_raw(
    x: &[Goldilocks],
    weight: &[Goldilocks],
    n_real: usize,
    rsqrt_table: &[Goldilocks],
) -> (Vec<Goldilocks>, Goldilocks, u32) {
    assert_eq!(x.len(), weight.len());
    let x_i32: Vec<i32> = x.iter().map(|&v| to_i32(v)).collect();
    let s_index = rms_norm_scalars_i32(&x_i32, n_real);
    assert!((s_index as usize) < rsqrt_table.len(), "mean(x^2) out of rsqrt table range");
    let rstd = rsqrt_table[s_index as usize];
    let raw = x
        .iter()
        .zip(weight)
        .map(|(&xv, &w)| xv * rstd * w)
        .collect();
    (raw, rstd, s_index)
}

/// Compute `out = f(round(in / 2^shift) + bias)` with the same rounding rules
/// [`prove_affine`] checks, so a caller can commit the output and then run
/// `prove_affine` against it. `f` is the identity (`relu == false`) or ReLU.
pub fn affine_raw(
    input: &[Goldilocks],
    bias: &[Goldilocks],
    shift: u32,
    relu: bool,
) -> Vec<Goldilocks> {
    assert_eq!(input.len(), bias.len());
    input
        .iter()
        .zip(bias)
        .map(|(&iv, &bv)| {
            let v = div_round(to_i64(iv), 1i64 << shift) + to_i32(bv) as i64;
            let v = if relu { v.max(0) } else { v };
            from_i32(v as i32)
        })
        .collect()
}

/// Compute the SiLU lookup indices (`x + offset`, non-negative) and the
/// corresponding table outputs, so a caller can commit them and run
/// [`prove_lookup`] against the quantized SiLU table. `SiLU(x) = x * sigmoid(x)`
/// is the one non-arithmetic activation of the input/horizon FFNs.
pub fn silu_raw(
    x: &[Goldilocks],
    table: &[Goldilocks],
    offset: u32,
) -> (Vec<u32>, Vec<Goldilocks>) {
    let indices: Vec<u32> = x
        .iter()
        .map(|&v| (to_i32(v) as i64 + offset as i64) as u32)
        .collect();
    let outputs: Vec<Goldilocks> = indices.iter().map(|&i| table[i as usize]).collect();
    (indices, outputs)
}

/// Compute `out = round(in * scale / 2^16) + bias` (the output-head rescale of
/// the horizon FFN), matching [`prove_scale`]. `in` and `bias` are at scale
/// 2^16, `scale` is an integer at scale 2^16 (e.g. `round(1.0487 * 2^16)`).
pub fn scale_raw(input: &[Goldilocks], scale: i64, bias: &[Goldilocks]) -> Vec<Goldilocks> {
    assert_eq!(input.len(), bias.len());
    input
        .iter()
        .zip(bias)
        .map(|(&iv, &bv)| {
            let v = div_round(to_i64(iv) * scale, 1i64 << 16) + to_i32(bv) as i64;
            from_i32(v as i32)
        })
        .collect()
}

/// Prove `out = round(in * scale / 2^16) + bias` against WHIR commitments, in
/// the O(N)-opening PoC form. The scalar multiply is a deterministic host-side
/// integer operation (no lookup), exactly like the rescale in [`prove_affine`].
/// Prove `bits` are all in {0,1} and that `value_at_r = sum_j bits_j(r) * 2^j`,
/// i.e. the value whose bits are given is in `[0, 2^bits.len())`. The bit-check
/// uses `sum eq b (b-1) = 0 <=> sum eq b^2 == sum eq b`, and the reconstruction
/// is a single random-point linear check.
fn prove_bits_range(
    whir: &Whir,
    bits: &[Vec<Goldilocks>],
    c_bits: &[Committed],
    value_at_r: Goldilocks,
    r: &[Goldilocks],
    s: &[Goldilocks],
    s_r: Goldilocks,
) -> bool {
    for (j, bj) in bits.iter().enumerate() {
        let c_bb: Goldilocks = s.iter().zip(bj).zip(bj).fold(Goldilocks::ZERO, |a, ((&si, &xi), &yi)| a + si * xi * yi);
        let c_b: Goldilocks = s.iter().zip(bj).fold(Goldilocks::ZERO, |a, (&si, &xi)| a + si * xi);
        let proof_bb = sumcheck::prove3(s, bj, bj, c_bb, r);
        let proof_b = sumcheck::prove(s, bj, c_b, r);
        let (b_open, b_r) = whir.open(c_bits[j].prover_data.clone(), &c_bits[j].protocol, r);
        if whir.verify(&c_bits[j].commitment, &b_open, &c_bits[j].protocol, r).unwrap() != b_r {
            return false;
        }
        if !sumcheck::verify3(&proof_bb, c_bb, r, s_r, b_r, b_r)
            || !sumcheck::verify(&proof_b, c_b, r, s_r, b_r)
            || c_bb != c_b
        {
            return false;
        }
    }
    let mut rhs = Goldilocks::ZERO;
    for (j, c_b) in c_bits.iter().enumerate() {
        let (b_open, b_r) = whir.open(c_b.prover_data.clone(), &c_b.protocol, r);
        if whir.verify(&c_b.commitment, &b_open, &c_b.protocol, r).unwrap() != b_r {
            return false;
        }
        rhs = rhs + b_r * from_i64(1i64 << j);
    }
    value_at_r == rhs
}

#[allow(clippy::too_many_arguments)]
fn prove_round(
    whir_in: &Whir,
    input: &Committed,
    in_plain: &[Goldilocks],
    whir_out: &Whir,
    output: &Committed,
    out_plain: &[Goldilocks],
    scale: i64,
    bias: &[Goldilocks],
    shift: usize,
    rng: &mut XorShift64,
) -> bool {
    let n = bias.len();
    let d = n.trailing_zeros() as usize;
    let scale_f = from_i64(scale);
    let two_shift = from_i64(1i64 << shift);
    let half = from_i64(1i64 << (shift - 1));

    // Host-side: decompose rem_off = in*scale - (out-bias)*2^shift + 2^(shift-1).
    let mut bits: Vec<Vec<Goldilocks>> = vec![vec![Goldilocks::ZERO; n]; shift];
    for i in 0..n {
        let in_i = to_i64(in_plain[i]);
        let out_i = to_i32(out_plain[i]) as i64;
        let b_i = to_i32(bias[i]) as i64;
        let rem = in_i * scale - (out_i - b_i) * (1i64 << shift);
        let rem_off = rem + (1i64 << (shift - 1));
        for (j, bj) in bits.iter_mut().enumerate() {
            bj[i] = from_i32(((rem_off >> j) & 1) as i32);
        }
    }

    let c_bits: Vec<Committed> = bits.iter().map(|b| commit(whir_out, b)).collect();
    let r: Vec<Goldilocks> = (0..d).map(|_| rng.field()).collect();
    let s = mle::eq_evals(&r);
    let s_r = mle::eval(&s, &r);

    let (in_open, in_r) = whir_in.open(input.prover_data.clone(), &input.protocol, &r);
    let (out_open, out_r) = whir_out.open(output.prover_data.clone(), &output.protocol, &r);
    if whir_in.verify(&input.commitment, &in_open, &input.protocol, &r).unwrap() != in_r {
        return false;
    }
    if whir_out.verify(&output.commitment, &out_open, &output.protocol, &r).unwrap() != out_r {
        return false;
    }
    let bias_r = mle::eval(bias, &r);
    let value_at_r = in_r * scale_f - (out_r - bias_r) * two_shift + half;
    prove_bits_range(whir_out, &bits, &c_bits, value_at_r, &r, &s, s_r)
}

#[allow(clippy::too_many_arguments)]
pub fn prove_scale(
    whir_in: &Whir,
    input: &Committed,
    in_plain: &[Goldilocks],
    whir_out: &Whir,
    output: &Committed,
    out_plain: &[Goldilocks],
    scale: i64,
    bias: &[Goldilocks],
    rng: &mut XorShift64,
) -> bool {
    prove_round(whir_in, input, in_plain, whir_out, output, out_plain, scale, bias, 16, rng)
}

/// Prove the raw RMSNorm product `raw = x * rstd * w` (scale 2^48) against WHIR
/// commitments, in the O(N)-opening PoC form. `mean(x^2)` is recomputed from the
/// *committed* `x`, and `rstd = 1/sqrt(mean(x^2) + eps)` is bound to it by a
/// LogUp lookup into `rsqrt_table` — the single non-arithmetic scalar of the
/// input RMSNorm (TimesFM's `input_layernorm`), not a trusted input.
#[allow(clippy::too_many_arguments)]
pub fn prove_rms_norm(
    whir_x: &Whir,
    x: &Committed,
    x_plain: &[Goldilocks],
    whir_raw: &Whir,
    raw: &Committed,
    raw_plain: &[Goldilocks],
    weight: &[Goldilocks],
    n_real: usize,
    rsqrt_table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
    rng: &mut XorShift64,
) -> bool {
    let n = weight.len();
    let d = n.trailing_zeros() as usize;
    assert_eq!(x_plain.len(), n);
    assert_eq!(raw_plain.len(), n);

    // mean(x^2): S_x2 = sum x(i)^2 via degree-2 sum-check (f = x, h = x).
    let s_x2: i64 = x_plain[..n_real]
        .iter()
        .map(|&v| {
            let d = to_i32(v) as i64;
            d * d
        })
        .sum();
    let s_x2_f = from_i64(s_x2);
    let r2: Vec<Goldilocks> = (0..d).map(|_| rng.field()).collect();
    let proof_x2 = sumcheck::prove(x_plain, x_plain, s_x2_f, &r2);
    let (x_open2, x_r2) = whir_x.open(x.prover_data.clone(), &x.protocol, &r2);
    if whir_x.verify(&x.commitment, &x_open2, &x.protocol, &r2).unwrap() != x_r2 {
        return false;
    }
    if !sumcheck::verify(&proof_x2, s_x2_f, &r2, x_r2, x_r2) {
        return false;
    }
    let s_index = div_round(div_round(s_x2, n_real as i64), 1 << 18) as u32;
    if (s_index as usize) >= rsqrt_table.len() {
        return false;
    }
    let rstd = rsqrt_table[s_index as usize];
    let lk = lookup::prove(&[s_index], &[rstd], rsqrt_table, alpha, beta);
    if !lookup::verify(&lk) {
        return false;
    }

    // raw = rstd x w via a degree-3 zero-check:
    //   sum eq(x,r) raw(x) == rstd sum eq(x,r) x(x) w(x).
    let r: Vec<Goldilocks> = (0..d).map(|_| rng.field()).collect();
    let s = mle::eq_evals(&r);
    let c_raw: Goldilocks = s.iter().zip(raw_plain).fold(Goldilocks::ZERO, |a, (&si, &ri)| a + si * ri);
    let c_xw: Goldilocks = s
        .iter()
        .zip(x_plain)
        .zip(weight)
        .fold(Goldilocks::ZERO, |a, ((&si, &xi), &wi)| a + si * xi * wi);

    let proof_raw = sumcheck::prove(&s, raw_plain, c_raw, &r);
    let proof_xw = sumcheck::prove3(&s, x_plain, weight, c_xw, &r);

    let (x_open, xr) = whir_x.open(x.prover_data.clone(), &x.protocol, &r);
    let (raw_open, rv) = whir_raw.open(raw.prover_data.clone(), &raw.protocol, &r);
    if whir_x.verify(&x.commitment, &x_open, &x.protocol, &r).unwrap() != xr {
        return false;
    }
    if whir_raw.verify(&raw.commitment, &raw_open, &raw.protocol, &r).unwrap() != rv {
        return false;
    }
    let s_r = mle::eval(&s, &r);
    let w_r = mle::eval(weight, &r);
    if !sumcheck::verify(&proof_raw, c_raw, &r, s_r, rv) {
        return false;
    }
    if !sumcheck::verify3(&proof_xw, c_xw, &r, s_r, xr, w_r) {
        return false;
    }
    if c_raw != rstd * c_xw {
        return false;
    }
    true
}

/// Prove `out = round(in / 2^shift) + bias` (the non-ReLU affine/rescale step)
/// with a sub-linear bit-decomposition range check, exactly like [`prove_round`].
#[allow(clippy::too_many_arguments)]
pub fn prove_affine(
    whir_in: &Whir,
    input: &Committed,
    in_plain: &[Goldilocks],
    whir_out: &Whir,
    output: &Committed,
    out_plain: &[Goldilocks],
    bias: &[Goldilocks],
    shift: u32,
    rng: &mut XorShift64,
) -> bool {
    prove_round(
        whir_in, input, in_plain, whir_out, output, out_plain, 1, bias, shift as usize, rng,
    )
}

/// Prove `out = ReLU(round(in / 2^16) + bias)` (the gate ReLU step) against WHIR
/// commitments, sub-linearly. The round is proven by [`prove_round`], then
/// `pre = rescaled + bias` and `abs = |pre|` are committed, `abs` is range-
/// checked to `[0, 2^31)`, `abs^2 = pre^2` is proven by a degree-3 zero-check
/// (so `abs = ±pre`, hence `|pre|` given the range), and `out = (pre + abs)/2`.
#[allow(clippy::too_many_arguments)]
pub fn prove_relu(
    whir_in: &Whir,
    input: &Committed,
    in_plain: &[Goldilocks],
    whir_out: &Whir,
    output: &Committed,
    _out_plain: &[Goldilocks],
    bias: &[Goldilocks],
    rng: &mut XorShift64,
) -> bool {
    let n = bias.len();
    let d = n.trailing_zeros() as usize;
    const SHIFT: usize = 16;
    const BITS: usize = 31;

    let rescaled: Vec<Goldilocks> = in_plain
        .iter()
        .map(|&iv| from_i32(div_round(to_i64(iv), 1i64 << SHIFT) as i32))
        .collect();
    let pre: Vec<Goldilocks> = rescaled.iter().zip(bias).map(|(&r, &b)| r + b).collect();
    let abs_pre: Vec<Goldilocks> = pre
        .iter()
        .map(|&p| from_i32((to_i32(p) as i64).abs() as i32))
        .collect();
    let mut bits: Vec<Vec<Goldilocks>> = vec![vec![Goldilocks::ZERO; n]; BITS];
    for i in 0..n {
        let a = (to_i32(pre[i]) as i64).abs();
        for (j, bj) in bits.iter_mut().enumerate() {
            bj[i] = from_i32(((a >> j) & 1) as i32);
        }
    }

    let c_rescaled = commit(whir_out, &rescaled);
    let c_pre = commit(whir_out, &pre);
    let c_abs = commit(whir_out, &abs_pre);
    let c_bits: Vec<Committed> = bits.iter().map(|b| commit(whir_out, b)).collect();

    // 1. rescaled = round(in / 2^16).
    let zero_bias = vec![from_i32(0); n];
    if !prove_round(
        whir_in, input, in_plain, whir_out, &c_rescaled, &rescaled, 1, &zero_bias, SHIFT, rng,
    ) {
        return false;
    }

    let r: Vec<Goldilocks> = (0..d).map(|_| rng.field()).collect();
    let s = mle::eq_evals(&r);
    let s_r = mle::eval(&s, &r);

    let (res_open, res_r) = whir_out.open(c_rescaled.prover_data.clone(), &c_rescaled.protocol, &r);
    let (pre_open, pre_r) = whir_out.open(c_pre.prover_data.clone(), &c_pre.protocol, &r);
    let (abs_open, abs_r) = whir_out.open(c_abs.prover_data.clone(), &c_abs.protocol, &r);
    let (out_open, out_r) = whir_out.open(output.prover_data.clone(), &output.protocol, &r);
    if whir_out.verify(&c_rescaled.commitment, &res_open, &c_rescaled.protocol, &r).unwrap() != res_r
        || whir_out.verify(&c_pre.commitment, &pre_open, &c_pre.protocol, &r).unwrap() != pre_r
        || whir_out.verify(&c_abs.commitment, &abs_open, &c_abs.protocol, &r).unwrap() != abs_r
        || whir_out.verify(&output.commitment, &out_open, &output.protocol, &r).unwrap() != out_r
    {
        return false;
    }
    let bias_r = mle::eval(bias, &r);

    // 2. pre = rescaled + bias.
    if pre_r != res_r + bias_r {
        return false;
    }
    // 3. abs_pre ∈ [0, 2^31).
    if !prove_bits_range(whir_out, &bits, &c_bits, abs_r, &r, &s, s_r) {
        return false;
    }
    // 4. abs^2 = pre^2 (degree-3 zero-check).
    let c_aa: Goldilocks = s.iter().zip(&abs_pre).zip(&abs_pre).fold(Goldilocks::ZERO, |a, ((&si, &xi), &yi)| a + si * xi * yi);
    let c_pp: Goldilocks = s.iter().zip(&pre).zip(&pre).fold(Goldilocks::ZERO, |a, ((&si, &xi), &yi)| a + si * xi * yi);
    let proof_aa = sumcheck::prove3(&s, &abs_pre, &abs_pre, c_aa, &r);
    let proof_pp = sumcheck::prove3(&s, &pre, &pre, c_pp, &r);
    if !sumcheck::verify3(&proof_aa, c_aa, &r, s_r, abs_r, abs_r)
        || !sumcheck::verify3(&proof_pp, c_pp, &r, s_r, pre_r, pre_r)
        || c_aa != c_pp
    {
        return false;
    }
    // 5. out = (pre + abs)/2  <=>  2 out = pre + abs.
    out_r * from_i64(2) == pre_r + abs_r
}

/// Prove the element-wise field addition `c = a + b` against WHIR commitments
/// with a single random-point opening. Because `a`, `b`, `c` are multilinear
/// extensions and `c(x) = a(x) + b(x)` as polynomials iff `c_i = a_i + b_i` at
/// every hypercube point, checking `c(r) = a(r) + b(r)` at one random `r` is
/// sound by Schwartz–Zippel (error ~ n / |F|). This replaces the O(N)-opening
/// PoC with a sub-linear check — the same shape `prove_matmul` uses.
#[allow(clippy::too_many_arguments)]
pub fn prove_add(
    whir_a: &Whir,
    a: &Committed,
    whir_b: &Whir,
    b: &Committed,
    whir_c: &Whir,
    c: &Committed,
    n: usize,
    rng: &mut XorShift64,
) -> bool {
    let d = n.trailing_zeros() as usize;
    let r: Vec<Goldilocks> = (0..d).map(|_| rng.field()).collect();
    let (a_open, av) = whir_a.open(a.prover_data.clone(), &a.protocol, &r);
    let (b_open, bv) = whir_b.open(b.prover_data.clone(), &b.protocol, &r);
    let (c_open, cv) = whir_c.open(c.prover_data.clone(), &c.protocol, &r);
    let a_ok = whir_a.verify(&a.commitment, &a_open, &a.protocol, &r).unwrap() == av;
    let b_ok = whir_b.verify(&b.commitment, &b_open, &b.protocol, &r).unwrap() == bv;
    let c_ok = whir_c.verify(&c.commitment, &c_open, &c.protocol, &r).unwrap() == cv;
    a_ok && b_ok && c_ok && cv == av + bv
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::PrimeCharacteristicRing;
    use crate::fixed_point::from_i64;

    #[test]
    fn committed_matmul_roundtrip() {
        let mut rng = XorShift64::new(0xabc);
        let (k, n) = (64usize, 64usize);
        let a: Vec<Goldilocks> = (0..k).map(|_| rng.field()).collect();
        let b: Vec<Goldilocks> = (0..k * n).map(|_| rng.field()).collect();
        let mut c = vec![Goldilocks::ZERO; n];
        for j in 0..n {
            let mut acc = Goldilocks::ZERO;
            for w in 0..k {
                acc = acc + a[w] * b[w * n + j];
            }
            c[j] = acc;
        }

        let whir_a = Whir::new_testing(6);
        let whir_b = Whir::new_testing(12);
        let whir_c = Whir::new_testing(6);
        let ca = commit(&whir_a, &a);
        let cb = commit(&whir_b, &b);
        let cc = commit(&whir_c, &c);

        assert!(prove_matmul(
            &whir_a, &ca, &whir_b, &cb, &whir_c, &cc, &a, &b, &c, k, n, &mut rng,
        ));
    }

    #[test]
    fn committed_lookup_roundtrip() {
        let mut rng = XorShift64::new(0xdef);
        let n = 64usize;
        let table_size = 64usize;
        let table: Vec<Goldilocks> = (0..table_size).map(|_| rng.field()).collect();
        let indices: Vec<u32> = (0..n).map(|_| (rng.next_u64() % table_size as u64) as u32).collect();
        let outputs: Vec<Goldilocks> = indices.iter().map(|&i| table[i as usize]).collect();
        let idx_vals: Vec<Goldilocks> = indices.iter().map(|&i| Goldilocks::from_u64(i as u64)).collect();

        let whir = Whir::new_testing(6);
        let cx = commit(&whir, &idx_vals);
        let cy = commit(&whir, &outputs);
        let alpha = rng.field();
        let beta = rng.field();
        assert!(prove_lookup(&whir, &cx, &whir, &cy, &indices, &table, alpha, beta));
    }

    #[test]
    fn committed_layer_norm_roundtrip() {
        let mut rng = XorShift64::new(0x123);
        let n = 64usize;
        // x as small i32 fixed-point values (scale 2^16); mean is derived from
        // these, and rstd is bound by an rsqrt lookup instead of being trusted.
        let x: Vec<Goldilocks> = (0..n)
            .map(|_| from_i32((rng.next_u64() % 65536) as i32))
            .collect();
        let weight: Vec<Goldilocks> = (0..n)
            .map(|_| from_i32((rng.next_u64() % 65536) as i32))
            .collect();
        // rsqrt table: index at scale 2^14, rstd at scale 2^16.
        let table_size = 1 << 17;
        let rsqrt_table: Vec<Goldilocks> = (0..table_size)
            .map(|j| {
                let s = j as f64 / 16384.0 + 1e-6;
                from_i32((1.0 / s.sqrt() * 65536.0).round() as i32)
            })
            .collect();
        let n_real = n;
        let (raw, _mean, _rstd, _s_index) = layer_norm_raw(&x, &weight, n_real, &rsqrt_table);

        let whir = Whir::new_testing(6);
        let cx = commit(&whir, &x);
        let c_raw = commit(&whir, &raw);
        let alpha = rng.field();
        let beta = rng.field();
        assert!(prove_layer_norm(
            &whir, &cx, &x, &whir, &c_raw, &raw, &weight, n_real, &rsqrt_table, alpha, beta, &mut rng,
        ));

        // Corrupt the committed raw -> the pointwise check must fail.
        let mut bad = raw.clone();
        bad[0] += Goldilocks::ONE;
        let c_bad = commit(&whir, &bad);
        assert!(!prove_layer_norm(
            &whir, &cx, &x, &whir, &c_bad, &bad, &weight, n_real, &rsqrt_table, alpha, beta, &mut rng,
        ));
    }

    #[test]
    fn committed_rms_norm_roundtrip() {
        let mut rng = XorShift64::new(0x567);
        let n = 64usize;
        let x: Vec<Goldilocks> = (0..n)
            .map(|_| from_i32((rng.next_u64() % 65536) as i32))
            .collect();
        let weight: Vec<Goldilocks> = (0..n)
            .map(|_| from_i32((rng.next_u64() % 65536) as i32))
            .collect();
        let table_size = 1 << 17;
        let rsqrt_table: Vec<Goldilocks> = (0..table_size)
            .map(|j| {
                let s = j as f64 / 16384.0 + 1e-6;
                from_i32((1.0 / s.sqrt() * 65536.0).round() as i32)
            })
            .collect();
        let n_real = n;
        let (raw, _rstd, _s_index) = rms_norm_raw(&x, &weight, n_real, &rsqrt_table);

        let whir = Whir::new_testing(6);
        let cx = commit(&whir, &x);
        let c_raw = commit(&whir, &raw);
        let alpha = rng.field();
        let beta = rng.field();
        assert!(prove_rms_norm(
            &whir, &cx, &x, &whir, &c_raw, &raw, &weight, n_real, &rsqrt_table, alpha, beta, &mut rng,
        ));

        let mut bad = raw.clone();
        bad[0] += Goldilocks::ONE;
        let c_bad = commit(&whir, &bad);
        assert!(!prove_rms_norm(
            &whir, &cx, &x, &whir, &c_bad, &bad, &weight, n_real, &rsqrt_table, alpha, beta, &mut rng,
        ));
    }

    #[test]
    fn committed_silu_lookup_roundtrip() {
        let mut rng = XorShift64::new(0x678);
        let n = 64usize;
        // Small synthetic SiLU table: x in [-2, 2) at scale 2^16, offset 2^17.
        let offset = 1u32 << 17;
        let table_size = 1usize << 18;
        let table: Vec<Goldilocks> = (0..table_size)
            .map(|j| {
                let xf = (j as f64 - offset as f64) / 65536.0;
                let s = xf / (1.0 + (-xf).exp());
                from_i32((s * 65536.0).round() as i32)
            })
            .collect();
        // x_q in [-1.5, 1.5) at scale 2^16.
        let x: Vec<Goldilocks> = (0..n)
            .map(|_| from_i32((rng.next_u64() % 196608) as i32 - 98304))
            .collect();
        let (indices, outputs) = silu_raw(&x, &table, offset);
        let idx_field: Vec<Goldilocks> = indices.iter().map(|&i| from_i32(i as i32)).collect();

        let whir = Whir::new_testing(6);
        let c_idx = commit(&whir, &idx_field);
        let c_out = commit(&whir, &outputs);
        let alpha = rng.field();
        let beta = rng.field();
        assert!(prove_lookup(&whir, &c_idx, &whir, &c_out, &indices, &table, alpha, beta));

        let mut bad = outputs.clone();
        bad[0] += Goldilocks::ONE;
        let c_bad = commit(&whir, &bad);
        assert!(!prove_lookup(&whir, &c_idx, &whir, &c_bad, &indices, &table, alpha, beta));
    }

    #[test]
    fn committed_scale_roundtrip() {
        let mut rng = XorShift64::new(0x789);
        let n = 64usize;
        let input: Vec<Goldilocks> = (0..n)
            .map(|_| from_i32((rng.next_u64() % 200000) as i32 - 100000))
            .collect();
        let bias: Vec<Goldilocks> = (0..n)
            .map(|_| from_i32((rng.next_u64() % 65536) as i32 - 32768))
            .collect();
        let scale = 68729i64; // round(1.0487 * 2^16)
        let out = scale_raw(&input, scale, &bias);

        let whir = Whir::new_testing(6);
        let c_in = commit(&whir, &input);
        let c_out = commit(&whir, &out);
        assert!(prove_scale(&whir, &c_in, &input, &whir, &c_out, &out, scale, &bias, &mut rng));

        let mut bad = out.clone();
        bad[0] += Goldilocks::ONE;
        let c_bad = commit(&whir, &bad);
        assert!(!prove_scale(&whir, &c_in, &input, &whir, &c_bad, &bad, scale, &bias, &mut rng));
    }

    #[test]
    fn committed_relu_roundtrip() {
        let mut rng = XorShift64::new(0x234);
        let n = 64usize;
        // Raw matmul output at scale 2^32 (signed i64), bias at scale 2^16.
        let raw: Vec<Goldilocks> = (0..n)
            .map(|_| {
                let v = (rng.next_u64() % (1u64 << 34)) as i64 - (1i64 << 33);
                from_i64(v)
            })
            .collect();
        let bias: Vec<Goldilocks> = (0..n)
            .map(|_| from_i32((rng.next_u64() % 65536) as i32 - 32768))
            .collect();
        let out: Vec<Goldilocks> = raw
            .iter()
            .zip(&bias)
            .map(|(&r, &b)| {
                let linear = div_round(to_i64(r), 1 << 16) + to_i32(b) as i64;
                from_i32(linear.max(0) as i32)
            })
            .collect();

        let whir = Whir::new_testing(6);
        let c_in = commit(&whir, &raw);
        let c_out = commit(&whir, &out);
        assert!(prove_relu(&whir, &c_in, &raw, &whir, &c_out, &out, &bias, &mut rng));

        let mut bad = out.clone();
        bad[0] += Goldilocks::ONE;
        let c_bad = commit(&whir, &bad);
        assert!(!prove_relu(&whir, &c_in, &raw, &whir, &c_bad, &bad, &bias, &mut rng));
    }

    #[test]
    fn committed_affine_rescale_roundtrip() {
        let mut rng = XorShift64::new(0x456);
        let n = 64usize;
        // Raw LayerNorm product at scale 2^48, rescale by 2^32, then bias.
        let raw: Vec<Goldilocks> = (0..n)
            .map(|_| {
                let v = (rng.next_u64() % (1u64 << 50)) as i64 - (1i64 << 49);
                from_i64(v)
            })
            .collect();
        let bias: Vec<Goldilocks> = (0..n)
            .map(|_| from_i32((rng.next_u64() % 65536) as i32 - 32768))
            .collect();
        let out: Vec<Goldilocks> = raw
            .iter()
            .zip(&bias)
            .map(|(&r, &b)| {
                let v = div_round(to_i64(r), 1i64 << 32) + to_i32(b) as i64;
                from_i32(v as i32)
            })
            .collect();

        let whir = Whir::new_testing(6);
        let c_in = commit(&whir, &raw);
        let c_out = commit(&whir, &out);
        assert!(prove_affine(&whir, &c_in, &raw, &whir, &c_out, &out, &bias, 32, &mut rng));
    }

    #[test]
    fn committed_add_roundtrip() {
        let mut rng = XorShift64::new(0x345);
        let n = 64usize;
        let a: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let b: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let c: Vec<Goldilocks> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();

        let whir = Whir::new_testing(6);
        let ca = commit(&whir, &a);
        let cb = commit(&whir, &b);
        let cc = commit(&whir, &c);
        assert!(prove_add(&whir, &ca, &whir, &cb, &whir, &cc, n, &mut rng));

        let mut bad = c.clone();
        bad[0] += Goldilocks::ONE;
        let c_bad = commit(&whir, &bad);
        assert!(!prove_add(&whir, &ca, &whir, &cb, &whir, &c_bad, n, &mut rng));
    }
}
