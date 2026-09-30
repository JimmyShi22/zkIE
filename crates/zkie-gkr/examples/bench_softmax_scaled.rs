use std::time::Instant;
use zkie_gkr::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_gkr::softmax_scaled::prove_softmax_scaled;

fn main() {
    let (m, n) = (512usize, 512usize);
    let table_len = 1usize << 18;
    let mut rng = XorShift64::new(0xBEEF);
    let table: Vec<Goldilocks> = (0..table_len).map(|_| rng.field()).collect();
    let indices: Vec<u32> = (0..m * n).map(|_| (rng.next_u64() % table_len as u64) as u32).collect();
    let e: Vec<Goldilocks> = indices.iter().map(|&i| table[i as usize]).collect();
    let sum: Vec<Goldilocks> = (0..m).map(|i| (0..n).fold(Goldilocks::ZERO, |a, j| a + e[i * n + j])).collect();
    let out: Vec<Goldilocks> = (0..m * n).map(|idx| e[idx] * sum[idx / n].inverse()).collect();
    let t0 = Instant::now();
    let _proof = prove_softmax_scaled(&indices, &e, &out, &table, m, n, &mut rng);
    let dt = t0.elapsed();
    println!(
        "softmax-scaled m={} n={} table=2^18 time={:.2}s (exp lookup + row-sum + rescale, O(m*n))",
        m,
        n,
        dt.as_secs_f64()
    );
}
