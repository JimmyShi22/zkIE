//! Bridge the engine's shard DAG compiler to the real claim-chained FFN prover:
//! compile a model (N FFN blocks) into shards at a tunable granularity, then
//! actually prove each shard with `ffn_chain` and measure wall time. This turns
//! the autotuner's "granularity" knob from a cost model into a real measurement.

use std::time::Instant;
use zkie_gkr::engine::{compile_shard_dag, Backend, Granularity, StageSchedule};
use zkie_gkr::ffn_chain::prove_ffn_chain;
use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_gkr::fixed_point::{from_i64, to_i64};

fn mm(a: &[Goldilocks], b: &[Goldilocks], m: usize, k: usize, n: usize) -> Vec<Goldilocks> {
    let mut c = vec![Goldilocks::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = Goldilocks::ZERO;
            for kk in 0..k {
                acc = acc + a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn round_div(a: i64, b: i64) -> i64 {
    let q = a.div_euclid(b);
    let r = a.rem_euclid(b);
    if r * 2 >= b {
        q + 1
    } else {
        q
    }
}

// One FFN forward step (witness only), so we can chain blocks.
fn ffn_step(
    x: &[Goldilocks],
    fc_w: &[Goldilocks],
    fc_b: &[Goldilocks],
    proj_w: &[Goldilocks],
    proj_b: &[Goldilocks],
    gelu_table: &[Goldilocks],
    m: usize,
    d: usize,
    ffn: usize,
    shift: u32,
) -> Vec<Goldilocks> {
    let h1 = mm(x, fc_w, m, d, ffn);
    let fc: Vec<Goldilocks> = (0..m * ffn)
        .map(|ij| from_i64(round_div(to_i64(h1[ij]), 1i64 << shift) + to_i64(fc_b[ij])))
        .collect();
    let act_idx: Vec<u32> = fc.iter().map(|&v| ((to_i64(v).max(0)) as u64 % 64) as u32).collect();
    let act: Vec<Goldilocks> = act_idx.iter().map(|&i| gelu_table[i as usize]).collect();
    let h2 = mm(&act, proj_w, m, ffn, d);
    let proj: Vec<Goldilocks> = (0..m * d)
        .map(|ij| from_i64(round_div(to_i64(h2[ij]), 1i64 << shift) + to_i64(proj_b[ij])))
        .collect();
    (0..m * d).map(|i| x[i] + proj[i]).collect()
}

fn main() {
    let (m, d, ffn) = (64usize, 128usize, 256usize);
    let shift = 8u32;
    let blocks = 8usize;
    let ops_per_block = 4usize; // fc, gelu, proj, residual
    let op_count = blocks * ops_per_block;
    let mut rng = XorShift64::new(0x1E1E);
    let x0: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
    let fc_w: Vec<Goldilocks> = (0..d * ffn).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
    let fc_b: Vec<Goldilocks> = (0..m * ffn).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect();
    let proj_w: Vec<Goldilocks> = (0..ffn * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
    let proj_b: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect();
    let gelu_table: Vec<Goldilocks> = (0..64).map(|j| from_i64((j as i64).pow(2) % 1000)).collect();

    let granularities = [
        Granularity::Ops(4),
        Granularity::Layers(1),
        Granularity::Layers(2),
        Granularity::WholeModel,
    ];

    println!("engine->real FFN shards (m={} d={} ffn={} blocks={}):", m, d, ffn, blocks);
    for g in &granularities {
        let dag = compile_shard_dag(op_count, blocks, *g, StageSchedule::uniform(Backend::Cpu));
        let t0 = Instant::now();
        let mut x_cur = x0.clone();
        for shard in &dag.shards {
            let blocks_in_shard = (shard.op_end - shard.op_start) / ops_per_block;
            for _ in 0..blocks_in_shard {
                let _ = prove_ffn_chain(
                    &x_cur, &fc_w, &fc_b, &proj_w, &proj_b, &gelu_table, m, d, ffn, shift, &mut rng,
                );
                x_cur = ffn_step(&x_cur, &fc_w, &fc_b, &proj_w, &proj_b, &gelu_table, m, d, ffn, shift);
            }
        }
        println!(
            "  {:?} -> {} shard(s) ({} blocks/shard), proved in {:.2}s",
            g,
            dag.shards.len(),
            op_count / dag.shards.len() / ops_per_block,
            t0.elapsed().as_secs_f64()
        );
    }
}
