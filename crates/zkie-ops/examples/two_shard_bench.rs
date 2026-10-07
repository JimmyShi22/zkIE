//! Tiny reproducible two-shard committed-DAG benchmark.
//!
//! Same fixture as `compose::tests::committed_two_shard_verify_never_opens`
//! (4 projection ops, 2 shards, one boundary tensor, fixed seed), with
//! explicitly-labelled TESTING WHIR parameters (`Whir::new_testing` — 32-bit
//! security, 10 PoW-bit budget). These are NOT production parameters: the run
//! must stay fast and deterministic, with no real-security grinding.
//!
//! Reports prove/verify wall time, per-instance open/verify stats, and asserts
//! the verifier produced no openings.
//!
//! Run: cargo run --release -p zkie-ops --example two_shard_bench

use zkie_core::common::field::{PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::from_i64;
use zkie_core::pcs::whir::Whir;
use zkie_ops::compose::{
    prove_committed_shard_dag, verify_committed_shard_dag, Op, Store,
};

fn main() {
    let mut rng = XorShift64::new(0x2525);
    let (m, d, shift) = (4usize, 8usize, 8u32);
    let mut store = Store::new();
    let x = store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect());
    let mut ws = Vec::new();
    let mut bs = Vec::new();
    for _ in 0..4 {
        ws.push(store.push((0..d * d).map(|_| from_i64((rng.next_u64() % 50) as i64)).collect()));
        bs.push(store.push((0..m * d).map(|_| from_i64((rng.next_u64() % 10) as i64 - 5)).collect()));
    }
    let mut outs = Vec::new();
    let mut rems = Vec::new();
    for _ in 0..4 {
        outs.push(store.push(vec![]));
        rems.push(store.push(vec![]));
    }
    let ops = vec![
        Op::Projection { x, w: ws[0], bias: bs[0], out: outs[0], rem: rems[0], m, k: d, n: d, shift },
        Op::Projection { x: outs[0], w: ws[1], bias: bs[1], out: outs[1], rem: rems[1], m, k: d, n: d, shift },
        Op::Projection { x: outs[1], w: ws[2], bias: bs[2], out: outs[2], rem: rems[2], m, k: d, n: d, shift },
        Op::Projection { x: outs[2], w: ws[3], bias: bs[3], out: outs[3], rem: rems[3], m, k: d, n: d, shift },
    ];

    // FIXED TESTING PARAMETERS: 32-bit security, 10 PoW-bit budget. Not
    // production — `Whir::new` (90-bit) would grind real proof-of-work.
    let whir = Whir::new_testing(5);

    let t0 = std::time::Instant::now();
    let proof = prove_committed_shard_dag(&mut store, &ops, 2, &whir, &mut rng);
    let prove_time = t0.elapsed();
    let (open_n, open_s) = whir.open_stats();
    let (verify_n, _) = whir.verify_stats();

    let t1 = std::time::Instant::now();
    let ok = verify_committed_shard_dag(&store, &ops, 2, &whir, &proof);
    let verify_time = t1.elapsed();
    let (open_n2, open_s2) = whir.open_stats();
    let (verify_n2, verify_s2) = whir.verify_stats();

    println!("two_shard_bench (4 projections, 2 shards, 1 boundary tensor, m=4, d=8, Whir::new_testing(5))");
    println!("prove: {:?}  verify: {:?}  ok={ok}", prove_time, verify_time);
    println!("shards={} boundary_tensors={} cross_open_proofs={}",
        proof.shards.len(), proof.boundary_tensors.len(), proof.cross_open_proofs.len());
    println!("after prove : open_stats=({open_n}, {open_s:.6}s) verify_stats=({verify_n}, _)");
    println!("after verify: open_stats=({open_n2}, {open_s2:.6}s) verify_stats=({verify_n2}, {verify_s2:.6}s)");
    assert!(ok, "honest proof must verify");
    assert_eq!((open_n, open_s), (open_n2, open_s2),
        "verifier produced openings — proof transport is broken");
    assert!(verify_n2 > verify_n, "verifier made no verify calls");
    println!("verifier produced no openings: OK (open_stats unchanged across verify)");
}
