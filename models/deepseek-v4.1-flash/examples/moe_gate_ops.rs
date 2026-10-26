use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_models_deepseek_v41_flash::{
    build_moe_gate, open_shard, DIM_PAD, N_ROUTED_EXPERTS_PAD,
};
use zkie_ops::compose::{prove_shard, verify_shard, Op, Store};

fn load_i32(path: &str) -> Vec<Goldilocks> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| from_i64(i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64))
        .collect()
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/data/zkIE/models/deepseek-v4.1-flash/raw/model-00003-of-00048.safetensors".to_string()
    });
    let m = 16usize;
    let shift = 16u32;
    let file = open_shard(&path);

    let mut store = Store::new();
    let sqrt_sp = store.push(load_i32("/data/zkIE/models/sqrt_softplus_table_i32.bin"));
    let limb_table = store.push(
        (0..(1usize << 16)).map(|j| Goldilocks::from_u64(j as u64)).collect::<Vec<_>>(),
    );
    let x = store.push(vec![from_i64(1 << 16); m * DIM_PAD]);
    let mut ops: Vec<Op> = Vec::new();
    let (sel, _thr, weights, _d1, _d2) =
        build_moe_gate(&mut store, &mut ops, file, 0, x, m, shift, sqrt_sp, limb_table);

    let mut rng = XorShift64::new(0xBEEF);
    let proof = prove_shard(&mut store, &ops, &[], &mut rng);
    assert!(verify_shard(&[], &ops, &proof));

    let n = N_ROUTED_EXPERTS_PAD;
    let selv = store.get(sel);
    let wv = store.get(weights);
    let mut idx = Vec::new();
    let mut wvals = Vec::new();
    for j in 0..n {
        if to_i64(selv[j]) == 1 {
            idx.push(j);
            wvals.push(to_i64(wv[j]));
        }
    }
    println!(
        "ok moe_gate ops={} topk={:?} weights={:?}",
        ops.len(),
        idx,
        wvals
    );
}
