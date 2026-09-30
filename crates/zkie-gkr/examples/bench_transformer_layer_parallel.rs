//! Layer parallelism for the claim-chained full layer: witness generation is
//! sequential (layers chain), but the proofs run across layers in threads.

use std::time::Instant;
use zkie_gkr::field::{Field, Goldilocks, XorShift64};
use zkie_gkr::fixed_point::from_i64;
use zkie_gkr::transformer_chain::{prove_transformer_layer, transformer_layer_forward};

fn main() {
    let (m, d, ffn) = (128usize, 256usize, 512usize);
    let shift = 16u32;
    let layers = 8usize;
    let mut rng = XorShift64::new(0xABCD);
    let x0: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let wq: Vec<Goldilocks> = (0..d * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let wk: Vec<Goldilocks> = (0..d * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let wv: Vec<Goldilocks> = (0..d * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let wo: Vec<Goldilocks> = (0..d * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let bias: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 20) as i64 - 10)).collect();
    let exp_table: Vec<Goldilocks> = (0..(1usize << 16)).map(|_| rng.field()).collect();
    let fc_w: Vec<Goldilocks> = (0..d * ffn).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let fc_b: Vec<Goldilocks> = (0..m * ffn).map(|_| from_i64((rng.next_u64() % 20) as i64 - 10)).collect();
    let proj_w: Vec<Goldilocks> = (0..ffn * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect();
    let proj_b: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 20) as i64 - 10)).collect();
    let gelu_table: Vec<Goldilocks> = (0..64).map(|j| from_i64((j as i64).pow(2) % 1000)).collect();
    let ln_w: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 5) as i64 + 1)).collect();
    let ln_b: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 20) as i64 - 10)).collect();
    let rsqrt_table: Vec<Goldilocks> = (0..(1usize << 16)).map(|_| rng.field()).collect();

    // Sequential witness generation.
    let mut xs: Vec<Vec<Goldilocks>> = Vec::with_capacity(layers + 1);
    xs.push(x0.clone());
    let mut x_cur = x0;
    for _ in 0..layers {
        x_cur = transformer_layer_forward(
            &x_cur, &wq, &wk, &wv, &wo, &bias, &exp_table, &fc_w, &fc_b, &proj_w, &proj_b,
            &gelu_table, &ln_w, &ln_b, &rsqrt_table, m, d, ffn, shift,
        );
        xs.push(x_cur.clone());
    }

    // Single-threaded baseline: prove all layers sequentially.
    let t0 = Instant::now();
    for i in 0..layers {
        let _ = prove_transformer_layer(
            &xs[i], &wq, &wk, &wv, &wo, &bias, &exp_table, &fc_w, &fc_b, &proj_w, &proj_b,
            &gelu_table, &ln_w, &ln_b, &rsqrt_table, m, d, ffn, shift, &mut rng,
        );
    }
    let single = t0.elapsed();

    // Parallel proof across layers.
    let xs_ref = &xs;
    let wq_ref = &wq;
    let wk_ref = &wk;
    let wv_ref = &wv;
    let wo_ref = &wo;
    let bias_ref = &bias;
    let exp_ref = &exp_table;
    let fc_w_ref = &fc_w;
    let fc_b_ref = &fc_b;
    let proj_w_ref = &proj_w;
    let proj_b_ref = &proj_b;
    let gelu_ref = &gelu_table;
    let ln_w_ref = &ln_w;
    let ln_b_ref = &ln_b;
    let rsqrt_ref = &rsqrt_table;
    let t1 = Instant::now();
    std::thread::scope(|s| {
        for i in 0..layers {
            s.spawn(move || {
                let mut r = XorShift64::new(0xBEEF + i as u64);
                let _ = prove_transformer_layer(
                    &xs_ref[i], wq_ref, wk_ref, wv_ref, wo_ref, bias_ref, exp_ref, fc_w_ref,
                    fc_b_ref, proj_w_ref, proj_b_ref, gelu_ref, ln_w_ref, ln_b_ref, rsqrt_ref,
                    m, d, ffn, shift, &mut r,
                );
            });
        }
    });
    let parallel = t1.elapsed();

    println!(
        "transformer-layer-parallel m={} d={} ffn={} layers={}: single={:.2}s parallel={:.2}s speedup={:.2}x",
        m,
        d,
        ffn,
        layers,
        single.as_secs_f64(),
        parallel.as_secs_f64(),
        single.as_secs_f64() / parallel.as_secs_f64()
    );
}
