//! WHIR-committed GKR matmul for `m == 1` (a vector left operand).
//!
//! This is the core of the interpreter: commit a tensor once, then prove
//! `C = A @ B` from prescribed-point openings rather than raw evaluations. The
//! ctx32 TimesFM model runs every layer with batch/sequence `m == 1`, so the
//! left operand is a vector and needs no transpose point-swap.

use crate::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::fixed_point::{from_i32, to_i32, to_i64};
use crate::lookup;
use crate::whir::{Commitment, OpeningProtocol, ProverData, Whir};
use crate::{matmul, mle};

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

/// Round `a / b` to the nearest integer, ties to even (matching Python's
/// `round`), for the signed integer reductions below. `b` must be positive.
fn div_round(a: i64, b: i64) -> i64 {
    debug_assert!(b > 0);
    let q = a.div_euclid(b);
    let r = a.rem_euclid(b);
    let twice = r * 2;
    if twice > b || (twice == b && q % 2 != 0) {
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
/// reduction over the first `n_real` entries, ignoring padding), and `rstd =
/// 1/sqrt(var + eps)` is bound to that variance by a LogUp lookup into
/// `rsqrt_table` — the single non-arithmetic scalar is no longer a trusted input.
#[allow(clippy::too_many_arguments)]
pub fn prove_layer_norm(
    whir_x: &Whir,
    x: &Committed,
    whir_raw: &Whir,
    raw: &Committed,
    weight: &[Goldilocks],
    n_real: usize,
    rsqrt_table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
) -> bool {
    let n = weight.len();
    let d = n.trailing_zeros() as usize;
    let mut xv: Vec<Goldilocks> = Vec::with_capacity(n);
    for i in 0..n {
        let point: Vec<Goldilocks> = (0..d)
            .map(|b| Goldilocks::from_bool((i >> b) & 1 == 1))
            .collect();
        let (x_open, xv_i) = whir_x.open(x.prover_data.clone(), &x.protocol, &point);
        if whir_x.verify(&x.commitment, &x_open, &x.protocol, &point).unwrap() != xv_i {
            return false;
        }
        xv.push(xv_i);
    }

    let x_i32: Vec<i32> = xv.iter().map(|&v| to_i32(v)).collect();
    let (mean, s_index) = layer_norm_scalars_i32(&x_i32, n_real);
    if (s_index as usize) >= rsqrt_table.len() {
        return false;
    }
    let mean_f = from_i32(mean);
    let rstd = rsqrt_table[s_index as usize];
    let lk = lookup::prove(&[s_index], &[rstd], rsqrt_table, alpha, beta);
    if !lookup::verify(&lk) {
        return false;
    }

    for i in 0..n {
        let point: Vec<Goldilocks> = (0..d)
            .map(|b| Goldilocks::from_bool((i >> b) & 1 == 1))
            .collect();
        let (raw_open, rv) = whir_raw.open(raw.prover_data.clone(), &raw.protocol, &point);
        if whir_raw.verify(&raw.commitment, &raw_open, &raw.protocol, &point).unwrap() != rv {
            return false;
        }
        let expected = (xv[i] - mean_f) * rstd * weight[i];
        if rv != expected {
            return false;
        }
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

/// Prove the raw RMSNorm product `raw = x * rstd * w` (scale 2^48) against WHIR
/// commitments, in the O(N)-opening PoC form. `mean(x^2)` is recomputed from the
/// *committed* `x`, and `rstd = 1/sqrt(mean(x^2) + eps)` is bound to it by a
/// LogUp lookup into `rsqrt_table` — the single non-arithmetic scalar of the
/// input RMSNorm (TimesFM's `input_layernorm`), not a trusted input.
#[allow(clippy::too_many_arguments)]
pub fn prove_rms_norm(
    whir_x: &Whir,
    x: &Committed,
    whir_raw: &Whir,
    raw: &Committed,
    weight: &[Goldilocks],
    n_real: usize,
    rsqrt_table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
) -> bool {
    let n = weight.len();
    let d = n.trailing_zeros() as usize;
    let mut xv: Vec<Goldilocks> = Vec::with_capacity(n);
    for i in 0..n {
        let point: Vec<Goldilocks> = (0..d)
            .map(|b| Goldilocks::from_bool((i >> b) & 1 == 1))
            .collect();
        let (x_open, xv_i) = whir_x.open(x.prover_data.clone(), &x.protocol, &point);
        if whir_x.verify(&x.commitment, &x_open, &x.protocol, &point).unwrap() != xv_i {
            return false;
        }
        xv.push(xv_i);
    }

    let x_i32: Vec<i32> = xv.iter().map(|&v| to_i32(v)).collect();
    let s_index = rms_norm_scalars_i32(&x_i32, n_real);
    if (s_index as usize) >= rsqrt_table.len() {
        return false;
    }
    let rstd = rsqrt_table[s_index as usize];
    let lk = lookup::prove(&[s_index], &[rstd], rsqrt_table, alpha, beta);
    if !lookup::verify(&lk) {
        return false;
    }

    for i in 0..n {
        let point: Vec<Goldilocks> = (0..d)
            .map(|b| Goldilocks::from_bool((i >> b) & 1 == 1))
            .collect();
        let (raw_open, rv) = whir_raw.open(raw.prover_data.clone(), &raw.protocol, &point);
        if whir_raw.verify(&raw.commitment, &raw_open, &raw.protocol, &point).unwrap() != rv {
            return false;
        }
        if rv != xv[i] * rstd * weight[i] {
            return false;
        }
    }
    true
}

/// Prove `out = f(round(in / 2^shift) + bias)` against WHIR commitments, in the
/// O(N)-opening PoC form, where `f` is either the identity (`relu == false`) or
/// ReLU (`relu == true`). `in` is a raw dot-product output embedded as signed
/// i64 (scale 2^32 for matmuls, 2^48 for the LayerNorm raw product), `bias` and
/// `out` are at scale 2^16. The rescale is ties-to-even (matching the Python
/// reference), and the ReLU is a deterministic host-side sign check — no lookup,
/// exactly like the mean/variance reduction in `prove_layer_norm`.
#[allow(clippy::too_many_arguments)]
pub fn prove_affine(
    whir_in: &Whir,
    input: &Committed,
    whir_out: &Whir,
    output: &Committed,
    bias: &[Goldilocks],
    shift: u32,
    relu: bool,
) -> bool {
    let n = bias.len();
    let d = n.trailing_zeros() as usize;
    for (i, &bias_i) in bias.iter().enumerate() {
        let point: Vec<Goldilocks> = (0..d)
            .map(|b| Goldilocks::from_bool((i >> b) & 1 == 1))
            .collect();
        let (in_open, inv) = whir_in.open(input.prover_data.clone(), &input.protocol, &point);
        let (out_open, ov) = whir_out.open(output.prover_data.clone(), &output.protocol, &point);
        if whir_in
            .verify(&input.commitment, &in_open, &input.protocol, &point)
            .unwrap()
            != inv
        {
            return false;
        }
        if whir_out
            .verify(&output.commitment, &out_open, &output.protocol, &point)
            .unwrap()
            != ov
        {
            return false;
        }
        let val_q = div_round(to_i64(inv), 1i64 << shift);
        let linear = val_q + to_i32(bias_i) as i64;
        let expected = if relu { linear.max(0) } else { linear };
        if to_i32(ov) as i64 != expected {
            return false;
        }
    }
    true
}

/// Convenience wrapper over [`prove_affine`] for the ReLU step: `out =
/// ReLU(round(in / 2^16) + bias)` with `in` a raw matmul output at scale 2^32.
#[allow(clippy::too_many_arguments)]
pub fn prove_relu(
    whir_in: &Whir,
    input: &Committed,
    whir_out: &Whir,
    output: &Committed,
    bias: &[Goldilocks],
) -> bool {
    prove_affine(whir_in, input, whir_out, output, bias, 16, true)
}

/// Prove the element-wise field addition `c = a + b` against WHIR commitments,
/// in the O(N)-opening PoC form. Used for bias additions and residual
/// connections once the intermediate tensors are chained: both operands are
/// committed, and the verifier recomputes the sum at every hypercube point.
#[allow(clippy::too_many_arguments)]
pub fn prove_add(
    whir_a: &Whir,
    a: &Committed,
    whir_b: &Whir,
    b: &Committed,
    whir_c: &Whir,
    c: &Committed,
    n: usize,
) -> bool {
    let d = n.trailing_zeros() as usize;
    for i in 0..n {
        let point: Vec<Goldilocks> = (0..d)
            .map(|b| Goldilocks::from_bool((i >> b) & 1 == 1))
            .collect();
        let (a_open, av) = whir_a.open(a.prover_data.clone(), &a.protocol, &point);
        let (b_open, bv) = whir_b.open(b.prover_data.clone(), &b.protocol, &point);
        let (c_open, cv) = whir_c.open(c.prover_data.clone(), &c.protocol, &point);
        if whir_a.verify(&a.commitment, &a_open, &a.protocol, &point).unwrap() != av {
            return false;
        }
        if whir_b.verify(&b.commitment, &b_open, &b.protocol, &point).unwrap() != bv {
            return false;
        }
        if whir_c.verify(&c.commitment, &c_open, &c.protocol, &point).unwrap() != cv {
            return false;
        }
        if cv != av + bv {
            return false;
        }
    }
    true
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
            &whir, &cx, &whir, &c_raw, &weight, n_real, &rsqrt_table, alpha, beta,
        ));

        // Corrupt the committed raw -> the pointwise check must fail.
        let mut bad = raw.clone();
        bad[0] += Goldilocks::ONE;
        let c_bad = commit(&whir, &bad);
        assert!(!prove_layer_norm(
            &whir, &cx, &whir, &c_bad, &weight, n_real, &rsqrt_table, alpha, beta,
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
            &whir, &cx, &whir, &c_raw, &weight, n_real, &rsqrt_table, alpha, beta,
        ));

        let mut bad = raw.clone();
        bad[0] += Goldilocks::ONE;
        let c_bad = commit(&whir, &bad);
        assert!(!prove_rms_norm(
            &whir, &cx, &whir, &c_bad, &weight, n_real, &rsqrt_table, alpha, beta,
        ));
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
        assert!(prove_relu(&whir, &c_in, &whir, &c_out, &bias));

        let mut bad = out.clone();
        bad[0] += Goldilocks::ONE;
        let c_bad = commit(&whir, &bad);
        assert!(!prove_relu(&whir, &c_in, &whir, &c_bad, &bias));
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
        assert!(prove_affine(&whir, &c_in, &whir, &c_out, &bias, 32, false));
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
        assert!(prove_add(&whir, &ca, &whir, &cb, &whir, &cc, n));

        let mut bad = c.clone();
        bad[0] += Goldilocks::ONE;
        let c_bad = commit(&whir, &bad);
        assert!(!prove_add(&whir, &ca, &whir, &cb, &whir, &c_bad, n));
    }
}
