//! Real-measurement autotune over shard granularity: prove a synthetic model at
//! several `ops_per_shard` values, time each with the actual proof, and report
//! the fastest. This is the "tune once, reuse" IE loop — the granularity knob is
//! the real shard-DAG composer, not a cost model.

use std::time::Instant;

use zkie_ops::compose::{Op, Store};
use zkie_engine::engine::{prove_model, verify_model, Granularity};
use zkie_core::common::field::XorShift64;
use zkie_core::common::fixed_point::from_i64;

fn main() {
    let mut rng = XorShift64::new(0xBEEF);
    let (m, d, shift, n) = (16usize, 128usize, 8u32, 20usize);

    let mut store = Store::new();
    let x = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect());
    let mut ws = Vec::new();
    let mut bs = Vec::new();
    for _ in 0..n {
        ws.push(store.push((0..d * d).map(|_| from_i64((rng.next_u64() % 100) as i64)).collect()));
        bs.push(store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 20) as i64 - 10)).collect()));
    }
    let mut outs = Vec::new();
    let mut rems = Vec::new();
    for _ in 0..n {
        outs.push(store.push(vec![]));
        rems.push(store.push(vec![]));
    }
    let mut ops = Vec::with_capacity(n);
    let mut prev = x;
    for i in 0..n {
        ops.push(Op::Projection { x: prev, w: ws[i], bias: bs[i], out: outs[i], rem: rems[i], m, k: d, n: d, shift });
        prev = outs[i];
    }

    println!("model: {} projections of {}x{} (shift={})", n, m, d, shift);
    let granularities = [1usize, 2, 4, 5, 10, 20];
    let mut best = (usize::MAX, std::time::Duration::MAX);
    for &g in &granularities {
        let t0 = Instant::now();
        let proof = prove_model(&mut store, &ops, Granularity::Ops(g), 1, &mut rng);
        let dt = t0.elapsed();
        assert!(verify_model(&mut store, &ops, Granularity::Ops(g), 1, &proof), "verify failed at {g}");
        println!("ops_per_shard={:>2}: {:>8.1?}  ({} shards)", g, dt, proof.shards.len());
        if dt < best.1 {
            best = (g, dt);
        }
    }
    println!("best: ops_per_shard={} at {:?}", best.0, best.1);
}
