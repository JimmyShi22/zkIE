//! Minimal elementwise layer compiler: translate a small op list (add / mul /
//! linear affine) into `prove_layer_circuit` constraints. This is the first,
//! arithmetic-only slice of the "Op -> one g per layer" compiler; the GPT-2
//! rounding/lookup/matmul ops build on the same constraint-folding pattern.
use crate::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::fixed_point::from_i64;
use crate::layer_circuit::{prove_layer_circuit, verify_layer_circuit, LayerCircuitProof};
#[derive(Clone, Debug)]
pub enum ElemOp {
    Add { a: usize, b: usize, c: usize },
    Mul { a: usize, b: usize, c: usize },
    Affine { x: usize, y: usize, scale: Goldilocks, bias: usize },
}
pub fn compile(ops: &[ElemOp]) -> Vec<Vec<(Goldilocks, Vec<usize>)>> {
    let mut out = Vec::new();
    for op in ops {
        match *op {
            ElemOp::Add { a, b, c } => out.push(vec![
                (Goldilocks::from_u64(1), vec![c]),
                (from_i64(-1), vec![a]),
                (from_i64(-1), vec![b]),
            ]),
            ElemOp::Mul { a, b, c } => out.push(vec![
                (Goldilocks::from_u64(1), vec![c]),
                (from_i64(-1), vec![a, b]),
            ]),
            ElemOp::Affine { x, y, scale, bias } => out.push(vec![
                (Goldilocks::from_u64(1), vec![y]),
                (from_i64(-1) * scale, vec![x]),
                (from_i64(-1), vec![bias]),
            ]),
        }
    }
    out
}
pub fn prove_layer(
    tensors: &[&[Goldilocks]],
    ops: &[ElemOp],
    r: &[Goldilocks],
    rng: &mut XorShift64,
) -> LayerCircuitProof {
    prove_layer_circuit(tensors, &compile(ops), r, rng)
}
pub fn verify_layer(
    proof: &LayerCircuitProof,
    tensors: &[&[Goldilocks]],
    ops: &[ElemOp],
    r: &[Goldilocks],
) -> bool {
    verify_layer_circuit(proof, tensors, &compile(ops), r)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compile_add_mul_affine() {
        let mut rng = XorShift64::new(0xF00D);
        let n = 1usize << 5;
        let x: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let b: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let w: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let scale = rng.field();
        let y: Vec<Goldilocks> = (0..n).map(|i| x[i] * scale + b[i]).collect();
        let z: Vec<Goldilocks> = (0..n).map(|i| y[i] * w[i]).collect();
        let t = n.trailing_zeros() as usize;
        let r: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
        let ops = vec![
            ElemOp::Affine { x: 0, y: 3, scale, bias: 1 },
            ElemOp::Mul { a: 3, b: 2, c: 4 },
        ];
        let tensors: Vec<&[Goldilocks]> = vec![&x, &b, &w, &y, &z];
        let proof = prove_layer(&tensors, &ops, &r, &mut rng);
        assert!(verify_layer(&proof, &tensors, &ops, &r));
        let mut bad_z = z.clone();
        bad_z[0] = bad_z[0] + Goldilocks::from_u64(1);
        let tensors_bad: Vec<&[Goldilocks]> = vec![&x, &b, &w, &y, &bad_z];
        assert!(!verify_layer(&proof, &tensors_bad, &ops, &r));
    }
    #[test]
    fn affine_round_layer() {
        let mut rng = XorShift64::new(0xAFFE);
        let n = 1usize << 5;
        let shift = 16u32;
        let half = Goldilocks::from_u64(1u64 << (shift - 1));
        let two_shift = Goldilocks::from_u64(1u64 << shift);
        let div_round = |a: i64, b: i64| -> i64 { let q = a.div_euclid(b); let rr = a.rem_euclid(b); if rr * 2 >= b { q + 1 } else { q } };
        let input: Vec<Goldilocks> = (0..n).map(|_| from_i64((rng.next_u64() % (1u64 << 30)) as i64)).collect();
        let bias: Vec<Goldilocks> = (0..n).map(|_| from_i64((rng.next_u64() % 2000) as i64 - 1000)).collect();
        let out: Vec<Goldilocks> = (0..n).map(|i| from_i64(div_round(crate::fixed_point::to_i64(input[i]), 1i64 << shift) + crate::fixed_point::to_i64(bias[i]))).collect();
        let rem_off: Vec<Goldilocks> = (0..n).map(|i| {
            let in_i = crate::fixed_point::to_i64(input[i]);
            let out_i = crate::fixed_point::to_i64(out[i]);
            let b_i = crate::fixed_point::to_i64(bias[i]);
            from_i64(in_i - (out_i - b_i) * (1i64 << shift) + (1i64 << (shift - 1)))
        }).collect();
        for &v in &rem_off {
            let iv = crate::fixed_point::to_i32(v);
            assert!(iv >= 0 && iv < (1i32 << shift));
        }
        let t = n.trailing_zeros() as usize;
        let r: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
        let neg = from_i64(-1);
        let constraint = vec![
            (Goldilocks::from_u64(1), vec![3usize]),
            (neg, vec![0usize]),
            (two_shift, vec![2usize]),
            (neg * two_shift, vec![1usize]),
            (neg * half, vec![4usize]),
        ];
        let ones = vec![Goldilocks::from_u64(1); n];
        let tensors: Vec<&[Goldilocks]> = vec![&input, &bias, &out, &rem_off, &ones];
        let lc = prove_layer_circuit(&tensors, std::slice::from_ref(&constraint), &r, &mut rng);
        assert!(verify_layer_circuit(&lc, &tensors, std::slice::from_ref(&constraint), &r));
        let table: Vec<Goldilocks> = (0..(1usize << shift)).map(|j| Goldilocks::from_u64(j as u64)).collect();
        let idx: Vec<u32> = rem_off.iter().map(|&v| crate::fixed_point::to_i32(v) as u32).collect();
        let alpha = rng.field();
        let beta = rng.field();
        let frac = crate::logup_gkr::prove_lookup_fractional(&idx, &rem_off, &table, alpha, beta, &mut rng);
        assert!(crate::logup_gkr::verify_lookup_fractional(&frac, &idx, &rem_off, &table, alpha, beta));
        let mut bad_out = out.clone();
        bad_out[0] = bad_out[0] + Goldilocks::from_u64(1);
        let tensors_bad: Vec<&[Goldilocks]> = vec![&input, &bias, &bad_out, &rem_off, &ones];
        assert!(!verify_layer_circuit(&lc, &tensors_bad, std::slice::from_ref(&constraint), &r));
    }
    #[test]
    fn matmul_add_chain_no_commit_c() {
        let mut rng = XorShift64::new(0x1234);
        let (m, k, n) = (2usize, 2usize, 2usize);
        let a: Vec<Goldilocks> = (0..m * k).map(|_| rng.field()).collect();
        let b: Vec<Goldilocks> = (0..k * n).map(|_| rng.field()).collect();
        let bias: Vec<Goldilocks> = (0..m * n).map(|_| rng.field()).collect();
        let mut c = vec![Goldilocks::from_u64(0); m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = Goldilocks::from_u64(0);
                for kk in 0..k {
                    acc = acc + a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        let d: Vec<Goldilocks> = c.iter().zip(&bias).map(|(&cv, &bv)| cv + bv).collect();
        let mut at = vec![Goldilocks::from_u64(0); k * m];
        for kk in 0..k {
            for i in 0..m {
                at[kk * m + i] = a[i * k + kk];
            }
        }
        let u = vec![rng.field()];
        let v = vec![rng.field()];
        let ch = vec![rng.field()];
        let mat = crate::matmul::prove(&at, &b, &c, m, k, n, &u, &v, &ch);
        let c_claim = mat.claimed;
        let pt = vec![v[0], u[0]];
        let eq = crate::mle::eq_evals(&pt);
        let neg = from_i64(-1);
        let terms = vec![
            (Goldilocks::from_u64(1), vec![0usize, 1usize]),
            (neg, vec![0usize, 2usize]),
            (neg, vec![0usize, 3usize]),
        ];
        let mles: Vec<&[Goldilocks]> = vec![&eq, &d, &c, &bias];
        let add_proof = crate::sumcheck::prove_virtual(&mles, &terms, Goldilocks::from_u64(0), &pt);
        let final_evals = vec![
            crate::mle::eval(&eq, &pt),
            crate::mle::eval(&d, &pt),
            c_claim,
            crate::mle::eval(&bias, &pt),
        ];
        assert!(crate::sumcheck::verify_virtual(&add_proof, &terms, Goldilocks::from_u64(0), &pt, &final_evals));
        let a_restricted = crate::mle::partial_eval(&at, &u);
        let b_restricted = crate::mle::partial_eval(&b, &v);
        let f_eval = crate::mle::eval(&a_restricted, &ch);
        let h_eval = crate::mle::eval(&b_restricted, &ch);
        assert!(crate::matmul::verify(&mat, &ch, f_eval, h_eval));
        let mut bad_d = d.clone();
        bad_d[0] = bad_d[0] + Goldilocks::from_u64(1);
        let bad_evals = vec![crate::mle::eval(&eq, &pt), crate::mle::eval(&bad_d, &pt), c_claim, crate::mle::eval(&bias, &pt)];
        assert!(!crate::sumcheck::verify_virtual(&add_proof, &terms, Goldilocks::from_u64(0), &pt, &bad_evals));
    }
}
