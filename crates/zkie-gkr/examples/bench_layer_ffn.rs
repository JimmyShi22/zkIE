use std::time::Instant;
use zkie_gkr::field::{Goldilocks, XorShift64};
use zkie_gkr::fixed_point::{from_i64, to_i64};
use zkie_gkr::projection::prove_projection;
use zkie_gkr::logup_gkr::prove_lookup_fractional;

fn matmul(a: &[Goldilocks], b: &[Goldilocks], m: usize, k: usize, n: usize) -> Vec<Goldilocks> {
    let mut h = vec![from_i64(0); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = from_i64(0);
            for kk in 0..k {
                acc = acc + a[i * k + kk] * b[kk * n + j];
            }
            h[i * n + j] = acc;
        }
    }
    h
}

fn main() {
    let (m, hdim, ffn) = (512usize, 1024usize, 4096usize);
    let shift = 16u32;
    let mut rng = XorShift64::new(0x1234);
    let x: Vec<Goldilocks> = (0..m * hdim).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let fc_w: Vec<Goldilocks> = (0..hdim * ffn).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let fc_b: Vec<Goldilocks> = (0..m * ffn).map(|_| from_i64((rng.next_u64() % 20) as i64 - 10)).collect();
    let proj_w: Vec<Goldilocks> = (0..ffn * hdim).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let proj_b: Vec<Goldilocks> = (0..m * hdim).map(|_| from_i64((rng.next_u64() % 20) as i64 - 10)).collect();
    let gelu_table: Vec<Goldilocks> = (0..64).map(|j| from_i64((j as i64).pow(2) % 1000)).collect();
    let div_round = |a: i64, b: i64| -> i64 { let q = a.div_euclid(b); let rr = a.rem_euclid(b); if rr * 2 >= b { q + 1 } else { q } };
    let h1 = matmul(&x, &fc_w, m, hdim, ffn);
    let fc: Vec<Goldilocks> = (0..m * ffn).map(|ij| from_i64(div_round(to_i64(h1[ij]), 1i64 << shift) + to_i64(fc_b[ij]))).collect();
    let fc_rem: Vec<Goldilocks> = (0..m * ffn).map(|ij| from_i64(to_i64(h1[ij]) - (to_i64(fc[ij]) - to_i64(fc_b[ij])) * (1i64 << shift) + (1i64 << (shift - 1)))).collect();
    let act_idx: Vec<u32> = fc.iter().map(|&v| ((to_i64(v).max(0)) as u64 % 64) as u32).collect();
    let act: Vec<Goldilocks> = act_idx.iter().map(|&idx| gelu_table[idx as usize]).collect();
    let h2 = matmul(&act, &proj_w, m, ffn, hdim);
    let proj: Vec<Goldilocks> = (0..m * hdim).map(|ij| from_i64(div_round(to_i64(h2[ij]), 1i64 << shift) + to_i64(proj_b[ij]))).collect();
    let proj_rem: Vec<Goldilocks> = (0..m * hdim).map(|ij| from_i64(to_i64(h2[ij]) - (to_i64(proj[ij]) - to_i64(proj_b[ij])) * (1i64 << shift) + (1i64 << (shift - 1)))).collect();
    let t0 = Instant::now();
    let _fc_p = prove_projection(&x, &fc_w, &fc_b, &fc, &fc_rem, m, hdim, ffn, shift, &mut rng);
    let a_alpha = rng.field();
    let a_beta = rng.field();
    let _act_p = prove_lookup_fractional(&act_idx, &act, &gelu_table, a_alpha, a_beta, &mut rng);
    let _proj_p = prove_projection(&act, &proj_w, &proj_b, &proj, &proj_rem, m, ffn, hdim, shift, &mut rng);
    let dt = t0.elapsed();
    println!("ffn m={} hdim={} ffn={} shift={} time={:.2}s", m, hdim, ffn, shift, dt.as_secs_f64());
}
