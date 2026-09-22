//! WHIR-committed GKR matmul for `m == 1` (a vector left operand).
//!
//! This is the core of the interpreter: commit a tensor once, then prove
//! `C = A @ B` from prescribed-point openings rather than raw evaluations. The
//! ctx32 TimesFM model runs every layer with batch/sequence `m == 1`, so the
//! left operand is a vector and needs no transpose point-swap.

use crate::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::PrimeCharacteristicRing;

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
}
