use std::time::Instant;
use zkie_core::pcs::committed::commit;
use zkie_core::common::field::{Goldilocks, XorShift64};
use zkie_core::common::fixed_point::from_i64;
use zkie_core::pcs::whir::Whir;

fn main() {
    let (m, k, n) = (512usize, 1024usize, 1024usize);
    let shift = 16u32;
    let mut rng = XorShift64::new(0xABCD);
    let x: Vec<Goldilocks> = (0..m * k).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let w: Vec<Goldilocks> = (0..k * n).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let bias: Vec<Goldilocks> = (0..m * n).map(|_| from_i64((rng.next_u64() % 20) as i64 - 10)).collect();
    let out: Vec<Goldilocks> = (0..m * n).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let rem: Vec<Goldilocks> = (0..m * n).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();

    let whir_km = Whir::new_testing((k * m).trailing_zeros() as usize);
    let whir_kn = Whir::new_testing((k * n).trailing_zeros() as usize);
    let whir_mn = Whir::new_testing((m * n).trailing_zeros() as usize);

    let t0 = Instant::now();
    let c_x = commit(&whir_km, &x);
    let c_w = commit(&whir_kn, &w);
    let c_bias = commit(&whir_mn, &bias);
    let c_out = commit(&whir_mn, &out);
    let c_rem = commit(&whir_mn, &rem);
    let commit_time = t0.elapsed();

    let u: Vec<Goldilocks> = (0..m.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let ch: Vec<Goldilocks> = (0..k.trailing_zeros() as usize).map(|_| rng.field()).collect();
    let mut ap = ch.clone();
    ap.extend_from_slice(&u);
    let mut bp = v.clone();
    bp.extend_from_slice(&ch);
    let mut pt = v.clone();
    pt.extend_from_slice(&u);

    let t1 = Instant::now();
    let _ = whir_km.open(c_x.prover_data.clone(), &c_x.protocol, &ap);
    let _ = whir_kn.open(c_w.prover_data.clone(), &c_w.protocol, &bp);
    let _ = whir_mn.open(c_bias.prover_data.clone(), &c_bias.protocol, &pt);
    let _ = whir_mn.open(c_out.prover_data.clone(), &c_out.protocol, &pt);
    let _ = whir_mn.open(c_rem.prover_data.clone(), &c_rem.protocol, &pt);
    let open_time = t1.elapsed();

    println!(
        "committed projection m={} k={} n={} shift={}: commit={:.2}s open={:.2}s total={:.2}s",
        m,
        k,
        n,
        shift,
        commit_time.as_secs_f64(),
        open_time.as_secs_f64(),
        (commit_time + open_time).as_secs_f64()
    );
    let _ = (bias, out, rem, x, w);
}
