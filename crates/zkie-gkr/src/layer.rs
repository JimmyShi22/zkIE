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
    #[test]
    fn matmul_affine_chain_no_commit_c() {
        let mut rng = XorShift64::new(0x5678);
        let (m, k, n) = (2usize, 2usize, 2usize);
        let shift = 4u32;
        let half = Goldilocks::from_u64(1u64 << (shift - 1));
        let two_shift = Goldilocks::from_u64(1u64 << shift);
        let div_round = |x: i64, b: i64| -> i64 { let q = x.div_euclid(b); let rr = x.rem_euclid(b); if rr * 2 >= b { q + 1 } else { q } };
        let a: Vec<Goldilocks> = (0..m * k).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
        let b: Vec<Goldilocks> = (0..k * n).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
        let bias: Vec<Goldilocks> = (0..m * n).map(|_| from_i64((rng.next_u64() % 20) as i64 - 10)).collect();
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
        let out: Vec<Goldilocks> = (0..m * n).map(|ij| from_i64(div_round(crate::fixed_point::to_i64(c[ij]), 1i64 << shift) + crate::fixed_point::to_i64(bias[ij]))).collect();
        let rem_off: Vec<Goldilocks> = (0..m * n).map(|ij| {
            let c_i = crate::fixed_point::to_i64(c[ij]);
            let o_i = crate::fixed_point::to_i64(out[ij]);
            let b_i = crate::fixed_point::to_i64(bias[ij]);
            from_i64(c_i - (o_i - b_i) * (1i64 << shift) + (1i64 << (shift - 1)))
        }).collect();
        for &vv in &rem_off {
            assert!(crate::fixed_point::to_i32(vv) >= 0 && crate::fixed_point::to_i32(vv) < (1i32 << shift));
        }
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
        let ones = vec![Goldilocks::from_u64(1); m * n];
        let neg = from_i64(-1);
        let terms = vec![
            (Goldilocks::from_u64(1), vec![0usize, 1usize]),
            (neg, vec![0usize, 2usize]),
            (two_shift, vec![0usize, 3usize]),
            (neg * two_shift, vec![0usize, 4usize]),
            (neg * half, vec![0usize, 5usize]),
        ];
        let mles: Vec<&[Goldilocks]> = vec![&eq, &rem_off, &c, &out, &bias, &ones];
        let affine_proof = crate::sumcheck::prove_virtual(&mles, &terms, Goldilocks::from_u64(0), &pt);
        let final_evals = vec![
            crate::mle::eval(&eq, &pt),
            crate::mle::eval(&rem_off, &pt),
            c_claim,
            crate::mle::eval(&out, &pt),
            crate::mle::eval(&bias, &pt),
            crate::mle::eval(&ones, &pt),
        ];
        assert!(crate::sumcheck::verify_virtual(&affine_proof, &terms, Goldilocks::from_u64(0), &pt, &final_evals));
        let a_restricted = crate::mle::partial_eval(&at, &u);
        let b_restricted = crate::mle::partial_eval(&b, &v);
        let f_eval = crate::mle::eval(&a_restricted, &ch);
        let h_eval = crate::mle::eval(&b_restricted, &ch);
        assert!(crate::matmul::verify(&mat, &ch, f_eval, h_eval));
        let table: Vec<Goldilocks> = (0..(1usize << shift)).map(|j| Goldilocks::from_u64(j as u64)).collect();
        let idx: Vec<u32> = rem_off.iter().map(|&vv| crate::fixed_point::to_i32(vv) as u32).collect();
        let alpha = rng.field();
        let beta = rng.field();
        let frac = crate::logup_gkr::prove_lookup_fractional(&idx, &rem_off, &table, alpha, beta, &mut rng);
        assert!(crate::logup_gkr::verify_lookup_fractional(&frac, &idx, &rem_off, &table, alpha, beta));
    }
    #[test]
    fn row_sum_reduction_virtual() {
        let mut rng = XorShift64::new(0x9ABC);
        let (m, n) = (4usize, 4usize);
        let e: Vec<Goldilocks> = (0..m * n).map(|_| rng.field()).collect();
        let sum: Vec<Goldilocks> = (0..m).map(|i| (0..n).fold(Goldilocks::from_u64(0), |acc, j| acc + e[i * n + j])).collect();
        let r: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let eq_i = crate::mle::eq_evals(&r);
        let eq_broadcast: Vec<Goldilocks> = (0..m * n).map(|idx| eq_i[idx / n]).collect();
        let terms = vec![(Goldilocks::from_u64(1), vec![0usize, 1usize])];
        let mles: Vec<&[Goldilocks]> = vec![&eq_broadcast, &e];
        let claimed = crate::mle::eval(&sum, &r);
        let challenges: Vec<Goldilocks> = (0..(m * n).trailing_zeros() as usize).map(|_| rng.field()).collect();
        let proof = crate::sumcheck::prove_virtual(&mles, &terms, claimed, &challenges);
        let final_evals = vec![crate::mle::eval(&eq_broadcast, &challenges), crate::mle::eval(&e, &challenges)];
        assert!(crate::sumcheck::verify_virtual(&proof, &terms, claimed, &challenges, &final_evals));
    }
    #[test]
    fn product_with_virtual_row_sum() {
        let mut rng = XorShift64::new(0x7777);
        let (m, n) = (4usize, 4usize);
        let e: Vec<Goldilocks> = (0..m * n).map(|_| rng.field()).collect();
        let out: Vec<Goldilocks> = (0..m * n).map(|_| rng.field()).collect();
        let sum: Vec<Goldilocks> = (0..m).map(|i| (0..n).fold(Goldilocks::from_u64(0), |acc, j| acc + e[i * n + j])).collect();
        let check: Vec<Goldilocks> = (0..m * n).map(|idx| out[idx] * sum[idx / n]).collect();
        let t2 = (m * n).trailing_zeros() as usize;
        let r: Vec<Goldilocks> = (0..t2).map(|_| rng.field()).collect();
        let eq2 = crate::mle::eq_evals(&r);
        let n3 = m * n * n;
        let mut eq_b: Vec<Goldilocks> = vec![Goldilocks::from_u64(0); n3];
        let mut out_b: Vec<Goldilocks> = vec![Goldilocks::from_u64(0); n3];
        let mut e_b: Vec<Goldilocks> = vec![Goldilocks::from_u64(0); n3];
        for i in 0..m {
            for j in 0..n {
                for jp in 0..n {
                    let idx = (i * n + j) * n + jp;
                    eq_b[idx] = eq2[i * n + j];
                    out_b[idx] = out[i * n + j];
                    e_b[idx] = e[i * n + jp];
                }
            }
        }
        let terms = vec![(Goldilocks::from_u64(1), vec![0usize, 1usize, 2usize])];
        let mles: Vec<&[Goldilocks]> = vec![&eq_b, &out_b, &e_b];
        let claimed = crate::mle::eval(&check, &r);
        let challenges: Vec<Goldilocks> = (0..n3.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let proof = crate::sumcheck::prove_virtual(&mles, &terms, claimed, &challenges);
        let final_evals = vec![crate::mle::eval(&eq_b, &challenges), crate::mle::eval(&out_b, &challenges), crate::mle::eval(&e_b, &challenges)];
        assert!(crate::sumcheck::verify_virtual(&proof, &terms, claimed, &challenges, &final_evals));
    }
    #[test]
    fn matmul_matmul_chain_no_commit_c() {
        let mut rng = XorShift64::new(0x8888);
        let (m, k, n, l) = (2usize, 2usize, 2usize, 2usize);
        let a: Vec<Goldilocks> = (0..m * k).map(|_| rng.field()).collect();
        let b: Vec<Goldilocks> = (0..k * n).map(|_| rng.field()).collect();
        let d: Vec<Goldilocks> = (0..n * l).map(|_| rng.field()).collect();
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
        let mut e = vec![Goldilocks::from_u64(0); m * l];
        for i in 0..m {
            for ll in 0..l {
                let mut acc = Goldilocks::from_u64(0);
                for j in 0..n {
                    acc = acc + c[i * n + j] * d[j * l + ll];
                }
                e[i * l + ll] = acc;
            }
        }
        let r: Vec<Goldilocks> = (0..(m * l).trailing_zeros() as usize).map(|_| rng.field()).collect();
        let eq2 = crate::mle::eq_evals(&r);
        let dom = m * k * n * l;
        let mut eq_b = vec![Goldilocks::from_u64(0); dom];
        let mut a_b = vec![Goldilocks::from_u64(0); dom];
        let mut b_b = vec![Goldilocks::from_u64(0); dom];
        let mut d_b = vec![Goldilocks::from_u64(0); dom];
        for i in 0..m {
            for kk in 0..k {
                for j in 0..n {
                    for ll in 0..l {
                        let idx = ((i * k + kk) * n + j) * l + ll;
                        eq_b[idx] = eq2[i * l + ll];
                        a_b[idx] = a[i * k + kk];
                        b_b[idx] = b[kk * n + j];
                        d_b[idx] = d[j * l + ll];
                    }
                }
            }
        }
        let terms = vec![(Goldilocks::from_u64(1), vec![0usize, 1usize, 2usize, 3usize])];
        let mles: Vec<&[Goldilocks]> = vec![&eq_b, &a_b, &b_b, &d_b];
        let claimed = crate::mle::eval(&e, &r);
        let challenges: Vec<Goldilocks> = (0..dom.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let proof = crate::sumcheck::prove_virtual(&mles, &terms, claimed, &challenges);
        let final_evals = vec![crate::mle::eval(&eq_b, &challenges), crate::mle::eval(&a_b, &challenges), crate::mle::eval(&b_b, &challenges), crate::mle::eval(&d_b, &challenges)];
        assert!(crate::sumcheck::verify_virtual(&proof, &terms, claimed, &challenges, &final_evals));
    }
}
