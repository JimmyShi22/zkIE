//! Demonstrate the autotuner's core lever: finer shard granularity = more shards
//! = more parallelism. Forward pass is sequential (witness); the proofs are run
//! across shards in threads, so finer granularity finishes the proof faster.

use std::time::Instant;
use zkie_gkr::engine::{compile_shard_dag, Backend, Granularity, StageSchedule};
use zkie_gkr::ffn_chain::{ffn_forward, prove_ffn_chain_witness, Fwd};
use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_gkr::fixed_point::from_i64;

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

    // Sequential witness generation (blocks chain, so this cannot be parallelized).
    let mut witnesses: Vec<(Vec<Goldilocks>, Fwd)> = Vec::with_capacity(blocks);
    let mut x_cur = x0;
    for _ in 0..blocks {
        let fwd = ffn_forward(&x_cur, &fc_w, &fc_b, &proj_w, &proj_b, &gelu_table, m, d, ffn, shift);
        witnesses.push((x_cur.clone(), fwd.clone()));
        x_cur = fwd.6.clone();
    }

    // Single-threaded baseline (WholeModel = one shard, all sequential).
    let single_dag = compile_shard_dag(op_count, blocks, Granularity::WholeModel, StageSchedule::uniform(Backend::Cpu));
    let t0 = Instant::now();
    for (x, fwd) in &witnesses {
        let _ = prove_ffn_chain_witness(x, fwd, &fc_w, &fc_b, &proj_w, &proj_b, &gelu_table, m, d, ffn, shift, &mut rng);
    }
    let single = t0.elapsed();
    println!("single-threaded ({} shard): {:.2}s", single_dag.shards.len(), single.as_secs_f64());

    let granularities = [
        Granularity::Ops(4),
        Granularity::Layers(1),
        Granularity::Layers(2),
        Granularity::Layers(3),
        Granularity::WholeModel,
    ];
    println!("parallel proof across shards (blocks={}):", blocks);
    for g in &granularities {
        let dag = compile_shard_dag(op_count, blocks, *g, StageSchedule::uniform(Backend::Cpu));
        let blocks_per_shard = op_count / dag.shards.len() / ops_per_block;
        let t1 = Instant::now();
        let w_ref = &witnesses;
        let fc_w_ref = &fc_w;
        let fc_b_ref = &fc_b;
        let proj_w_ref = &proj_w;
        let proj_b_ref = &proj_b;
        let gelu_ref = &gelu_table;
        std::thread::scope(|s| {
            for shard in &dag.shards {
                let start_block = shard.op_start / ops_per_block;
                let count = (shard.op_end - shard.op_start) / ops_per_block;
                s.spawn(move || {
                    let mut r = XorShift64::new(0xBEEF + start_block as u64);
                    for b in start_block..start_block + count {
                        let _ = prove_ffn_chain_witness(
                            &w_ref[b].0,
                            &w_ref[b].1,
                            fc_w_ref,
                            fc_b_ref,
                            proj_w_ref,
                            proj_b_ref,
                            gelu_ref,
                            m,
                            d,
                            ffn,
                            shift,
                            &mut r,
                        );
                    }
                });
            }
        });
        println!(
            "  {:?} -> {} shard(s), {} block(s)/shard: {:.2}s",
            g,
            dag.shards.len(),
            blocks_per_shard,
            t1.elapsed().as_secs_f64()
        );
    }
}
