//! Prove the full TimesFM 1.0 200M (20 layers, 16 heads) end to end.

use zkie_core::common::field::{PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::to_i64;
use zkie_models_timesfm::build_timesfm;
use zkie_ops::compose::{prove_shard_dag, verify_shard_dag, Store};

fn main() {
    let dir = "models/timesfm/weights";
    let mut store = Store::new();
    let mut ops = Vec::new();
    let out = build_timesfm(&mut store, &mut ops, dir, 16);

    println!("total ops: {}", ops.len());
    let ops_per_shard = 214; // start with whole-model (1 shard), then tune
    let mut rng = XorShift64::new(0xBEEF);
    let t0 = std::time::Instant::now();
    let proof = prove_shard_dag(&mut store, &ops, ops_per_shard, &mut rng);
    let prove_t = t0.elapsed();
    let t1 = std::time::Instant::now();
    assert!(verify_shard_dag(&store, &ops, ops_per_shard, &proof));
    let verify_t = t1.elapsed();
    println!("prove {:?}, verify {:?}", prove_t, verify_t);

    let outv = store.get(out);
    println!("out len {}", outv.len());
    for i in 0..8.min(outv.len()) {
        println!("  out[{}] = {}", i, to_i64(outv[i]));
    }
}
