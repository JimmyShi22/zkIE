//! FFN subgraph test: fc = proj(x,fc_w,fc_b), act = gelu(fc), proj = proj(act,proj_w,proj_b),
//! out = x + proj. Composes the reusable blocks (projection + logUp lookup + residual
//! add) into a real GPT-2 FFN subgraph. Intermediate matmul outputs are never committed.
use crate::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::fixed_point::{from_i64, to_i32, to_i64};
use crate::logup_gkr::{prove_lookup_fractional, verify_lookup_fractional};
use crate::projection::{prove_projection, verify_projection};
fn matmul_full(a: &[Goldilocks], b: &[Goldilocks], m: usize, k: usize, n: usize) -> Vec<Goldilocks> {
    let mut h = vec![Goldilocks::from_u64(0); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = Goldilocks::from_u64(0);
            for kk in 0..k {
                acc = acc + a[i * k + kk] * b[kk * n + j];
            }
            h[i * n + j] = acc;
        }
    }
    h
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ffn_block_composition() {
        let mut rng = XorShift64::new(0xFF00);
        let (m, k, n) = (2usize, 2usize, 2usize);
        let shift = 4u32;
        let half = Goldilocks::from_u64(1u64 << (shift - 1));
        let two_shift = Goldilocks::from_u64(1u64 << shift);
        let div_round = |a: i64, b: i64| -> i64 { let q = a.div_euclid(b); let rr = a.rem_euclid(b); if rr * 2 >= b { q + 1 } else { q } };
        let x: Vec<Goldilocks> = (0..m * k).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let fc_w: Vec<Goldilocks> = (0..k * n).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let fc_b: Vec<Goldilocks> = (0..m * n).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect();
        let proj_w: Vec<Goldilocks> = (0..n * n).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
        let proj_b: Vec<Goldilocks> = (0..m * n).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect();
        let gelu_table: Vec<Goldilocks> = (0..64).map(|j| from_i64((j as i64).pow(2) % 1000)).collect();
        let offset = 0i64;
        let h1 = matmul_full(&x, &fc_w, m, k, n);
        let fc: Vec<Goldilocks> = (0..m * n).map(|ij| from_i64(div_round(to_i64(h1[ij]), 1i64 << shift) + to_i64(fc_b[ij]))).collect();
        let fc_rem: Vec<Goldilocks> = (0..m * n).map(|ij| from_i64(to_i64(h1[ij]) - (to_i64(fc[ij]) - to_i64(fc_b[ij])) * (1i64 << shift) + (1i64 << (shift - 1)))).collect();
        let act_idx: Vec<u32> = fc.iter().map(|&v| ((to_i64(v) + offset).max(0) as u64 % 64) as u32).collect();
        let act: Vec<Goldilocks> = act_idx.iter().map(|&idx| gelu_table[idx as usize]).collect();
        let h2 = matmul_full(&act, &proj_w, m, n, n);
        let proj: Vec<Goldilocks> = (0..m * n).map(|ij| from_i64(div_round(to_i64(h2[ij]), 1i64 << shift) + to_i64(proj_b[ij]))).collect();
        let proj_rem: Vec<Goldilocks> = (0..m * n).map(|ij| from_i64(to_i64(h2[ij]) - (to_i64(proj[ij]) - to_i64(proj_b[ij])) * (1i64 << shift) + (1i64 << (shift - 1)))).collect();
        let out: Vec<Goldilocks> = (0..m * n).map(|ij| x[ij] + proj[ij]).collect();
        let _ = (half, two_shift);
        for &vv in &fc_rem {
            assert!(to_i32(vv) >= 0 && to_i32(vv) < (1i32 << shift));
        }
        for &vv in &proj_rem {
            assert!(to_i32(vv) >= 0 && to_i32(vv) < (1i32 << shift));
        }
        let fc_proof = prove_projection(&x, &fc_w, &fc_b, &fc, &fc_rem, m, k, n, shift, &mut rng);
        assert!(verify_projection(&fc_proof, &x, &fc_w, &fc_b, &fc, &fc_rem, m, k, n, shift));
        let a_alpha = rng.field();
        let a_beta = rng.field();
        let act_proof = prove_lookup_fractional(&act_idx, &act, &gelu_table, a_alpha, a_beta, &mut rng);
        assert!(verify_lookup_fractional(&act_proof, &act_idx, &act, &gelu_table, a_alpha, a_beta));
        let proj_proof = prove_projection(&act, &proj_w, &proj_b, &proj, &proj_rem, m, n, n, shift, &mut rng);
        assert!(verify_projection(&proj_proof, &act, &proj_w, &proj_b, &proj, &proj_rem, m, n, n, shift));
        for ij in 0..m * n {
            assert_eq!(out[ij], x[ij] + proj[ij]);
        }
    }
}
