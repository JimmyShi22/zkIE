use std::time::Instant;
use zkie_core::common::field::{Goldilocks, XorShift64};
use zkie_core::common::fixed_point::{from_i64, to_i64, to_i32};
use zkie_core::common::logup_gkr::prove_lookup_fractional;
use zkie_core::common::sumcheck::prove_virtual;

fn main() {
    let seq = 512usize;
    let table_log = 18u32;
    let table_size = 1usize << table_log;
    let offset = table_size / 2;
    let shift = 16u32;
    let mut rng = XorShift64::new(0x9ABC);
    let scores: Vec<Goldilocks> = (0..seq * seq).map(|_| from_i64((rng.next_u64() % 200000) as i64 - 100000)).collect();
    let exp_table: Vec<Goldilocks> = (0..table_size).map(|j| {
        let x = (j as f64 - offset as f64) / (1i64 << shift) as f64;
        from_i64((x.exp() * (1i64 << shift) as f64).round() as i64)
    }).collect();
    let indices: Vec<u32> = scores.iter().map(|&v| ((to_i32(v) as i64 + offset as i64).clamp(0, table_size as i64 - 1) as u32)).collect();
    let e: Vec<Goldilocks> = indices.iter().map(|&i| exp_table[i as usize]).collect();
    let sum: Vec<Goldilocks> = (0..seq).map(|i| (0..seq).fold(from_i64(0), |acc, j| acc + e[i * seq + j])).collect();
    let div_round = |a: i64, b: i64| -> i64 { let q = a.div_euclid(b); let rr = a.rem_euclid(b); if rr * 2 >= b { q + 1 } else { q } };
    let probs: Vec<Goldilocks> = (0..seq * seq).map(|ij| from_i64(div_round(to_i64(e[ij]) * (1i64 << shift), to_i64(sum[ij / seq])))).collect();
    let rem: Vec<Goldilocks> = (0..seq * seq).map(|ij| e[ij] * from_i64(1i64 << shift) - probs[ij] * sum[ij / seq]).collect();
    let alpha = rng.field();
    let beta = rng.field();
    let t0 = Instant::now();
    let _lookup = prove_lookup_fractional(&indices, &e, &exp_table, alpha, beta, &mut rng);
    let r_i: Vec<Goldilocks> = (0..seq.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let eq_i = zkie_core::common::mle::eq_evals(&r_i);
    let eq_b: Vec<Goldilocks> = (0..seq * seq).map(|idx| eq_i[idx / seq]).collect();
    let sum_claim = zkie_core::common::mle::eval(&sum, &r_i);
    let sum_mles: Vec<&[Goldilocks]> = vec![&eq_b, &e];
    let sum_terms = vec![(from_i64(1), vec![0usize, 1usize])];
    let sum_ch: Vec<Goldilocks> = (0..(seq * seq).trailing_zeros() as usize).map(|_| rng.field()).collect();
    let _sum_p = prove_virtual(&sum_mles, &sum_terms, sum_claim, &sum_ch);
    let dt = t0.elapsed();
    println!("softmax seq={} table=2^{} shift={} time={:.2}s (exp lookup + row-sum; rescale omitted)", seq, table_log, shift, dt.as_secs_f64());
    let _ = (scores, probs, rem);
}
