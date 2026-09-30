use std::time::Instant;
use zkie_gkr::field::{Goldilocks, XorShift64};
use zkie_gkr::fixed_point::{from_i64, to_i64};
use zkie_gkr::matmul::prove as matmul_prove;
use zkie_gkr::sumcheck::prove_virtual;

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

fn transpose(a: &[Goldilocks], m: usize, k: usize) -> Vec<Goldilocks> {
    let mut at = vec![from_i64(0); k * m];
    for kk in 0..k {
        for i in 0..m {
            at[kk * m + i] = a[i * k + kk];
        }
    }
    at
}

fn chs(rng: &mut XorShift64, n: usize) -> Vec<Goldilocks> {
    (0..n).map(|_| rng.field()).collect()
}

fn main() {
    let (m, d) = (512usize, 1024usize);
    let shift = 16u32;
    let mut rng = XorShift64::new(0x5678);
    let q: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let k: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let v: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let kt = transpose(&k, m, d);
    let div_round = |a: i64, b: i64| -> i64 { let q = a.div_euclid(b); let rr = a.rem_euclid(b); if rr * 2 >= b { q + 1 } else { q } };
    let scores = matmul(&q, &kt, m, d, m);
    let sum: Vec<Goldilocks> = (0..m).map(|i| (0..m).fold(from_i64(0), |acc, j| acc + scores[i * m + j])).collect();
    let probs: Vec<Goldilocks> = (0..m * m).map(|ij| from_i64(div_round(to_i64(scores[ij]) * (1i64 << shift), to_i64(sum[ij / m])))).collect();
    let rem: Vec<Goldilocks> = (0..m * m).map(|ij| scores[ij] * from_i64(1i64 << shift) - probs[ij] * sum[ij / m]).collect();
    let attn = matmul(&probs, &v, m, m, d);
    let qt = transpose(&q, m, d);
    let pt = transpose(&probs, m, m);
    let ch9 = chs(&mut rng, m.trailing_zeros() as usize);
    let ch10 = chs(&mut rng, d.trailing_zeros() as usize);
    let t0 = Instant::now();
    let _scores_p = matmul_prove(&qt, &kt, &scores, m, d, m, &ch9, &ch9, &ch10);
    let _attn_p = matmul_prove(&pt, &v, &attn, m, m, d, &ch9, &ch10, &ch9);
    let eq_i = zkie_gkr::mle::eq_evals(&ch9);
    let eq_b: Vec<Goldilocks> = (0..m * m).map(|idx| eq_i[idx / m]).collect();
    let sum_claim = zkie_gkr::mle::eval(&sum, &ch9);
    let sum_mles: Vec<&[Goldilocks]> = vec![&eq_b, &scores];
    let sum_terms = vec![(from_i64(1), vec![0usize, 1usize])];
    let sum_ch: Vec<Goldilocks> = (0..(m * m).trailing_zeros() as usize).map(|_| rng.field()).collect();
    let _sum_p = prove_virtual(&sum_mles, &sum_terms, sum_claim, &sum_ch);
    let dt = t0.elapsed();
    println!("attention m={} d={} shift={} proving={:.2}s (2 matmul GKR + softmax row-sum)", m, d, shift, dt.as_secs_f64());
    let _ = (scores, probs, rem, attn);
}
