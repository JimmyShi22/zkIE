//! The autotuner on real measurements: for each granularity, run the actual
//! boundary commit + parallel proof and measure the wall time, then let
//! `autotune_with` pick the lowest-total config. The schedule dimension is still
//! uniform-CPU (GPU dispatch is a later step); granularity is fully real.

use std::time::Instant;
use zkie_gkr::committed::commit;
use zkie_gkr::engine::{
    autotune_with, compile_shard_dag, Backend, Granularity, StageSchedule, TuningResult,
};
use zkie_gkr::ffn_chain::{ffn_forward, prove_ffn_chain_witness, Fwd};
use zkie_gkr::field::{Goldilocks, XorShift64};
use zkie_gkr::fixed_point::from_i64;
use zkie_gkr::whir::Whir;

fn main() {
    let (m, d, ffn) = (256usize, 512usize, 1024usize);
    let shift = 16u32;
    let blocks = 12usize;
    let ops_per_block = 4usize;
    let op_count = blocks * ops_per_block;
    let mut rng = XorShift64::new(0x2E2E);
    let x0: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
    let fc_w: Vec<Goldilocks> = (0..d * ffn).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
    let fc_b: Vec<Goldilocks> = (0..m * ffn).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect();
    let proj_w: Vec<Goldilocks> = (0..ffn * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect();
    let proj_b: Vec<Goldilocks> = (0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect();
    let gelu_table: Vec<Goldilocks> = (0..64).map(|j| from_i64((j as i64).pow(2) % 1000)).collect();

    let mut witnesses: Vec<(Vec<Goldilocks>, Fwd)> = Vec::with_capacity(blocks);
    let mut xs: Vec<Vec<Goldilocks>> = Vec::with_capacity(blocks + 1);
    xs.push(x0.clone());
    let mut x_cur = x0;
    for _ in 0..blocks {
        let fwd = ffn_forward(&x_cur, &fc_w, &fc_b, &proj_w, &proj_b, &gelu_table, m, d, ffn, shift);
        witnesses.push((x_cur.clone(), fwd.clone()));
        x_cur = fwd.6.clone();
        xs.push(x_cur.clone());
    }
    let whir = Whir::new_testing((m * d).trailing_zeros() as usize);

    let granularities = [
        Granularity::Ops(4),
        Granularity::Layers(1),
        Granularity::Layers(2),
        Granularity::Layers(3),
        Granularity::WholeModel,
    ];

    let measure = |g: Granularity, sched: StageSchedule| -> TuningResult {
        let dag = compile_shard_dag(op_count, blocks, g, sched);
        let blocks_per_shard = op_count / dag.shards.len() / ops_per_block;

        let t_c = Instant::now();
        for i in 0..dag.shards.len() {
            let _ = commit(&whir, &xs[i * blocks_per_shard]);
        }
        let _ = commit(&whir, &xs[blocks]);
        let commit_s = t_c.elapsed().as_secs_f64();

        let w_ref = &witnesses;
        let fc_w_ref = &fc_w;
        let fc_b_ref = &fc_b;
        let proj_w_ref = &proj_w;
        let proj_b_ref = &proj_b;
        let gelu_ref = &gelu_table;
        let t_p = Instant::now();
        std::thread::scope(|s| {
            for shard in &dag.shards {
                let start_block = shard.op_start / ops_per_block;
                let count = (shard.op_end - shard.op_start) / ops_per_block;
                s.spawn(move || {
                    let mut r = XorShift64::new(0xBEEF + start_block as u64);
                    for b in start_block..start_block + count {
                        let _ = prove_ffn_chain_witness(
                            &w_ref[b].0, &w_ref[b].1, fc_w_ref, fc_b_ref, proj_w_ref, proj_b_ref,
                            gelu_ref, m, d, ffn, shift, &mut r,
                        );
                    }
                });
            }
        });
        let proof_s = t_p.elapsed().as_secs_f64();

        TuningResult {
            granularity: g,
            schedule: sched,
            shards: dag.shards.len(),
            boundaries: dag.boundaries.len(),
            forward_s: 0.0,
            commit_s,
            sumcheck_s: proof_s,
            open_s: 0.0,
            total_s: commit_s + proof_s,
        }
    };

    let best = autotune_with(&granularities, &[StageSchedule::uniform(Backend::Cpu)], measure);
    println!("autotuned (real measurement) -> {}", best);
}
