use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_models_deepseek_v41_flash::{
    build_deepseek_v41, build_mla_attention, build_oproj, open_shard, HEAD_QK,
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
    let dir = std::path::Path::new(&path)
        .parent()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    let m: usize = std::env::var("ZKIE_M")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let shift = 16u32;
    let mut store = Store::new();
    let mut ops: Vec<Op> = Vec::new();

    let (q, kv) = build_deepseek_v41(&mut store, &mut ops, &dir, 0, m, shift);

    let file = open_shard(&path);
    let exp_table = store.push(load_i32("/data/zkIE/models/gpt2/weights/exp_table_i32.bin"));
    let limb_table = store.push(
        (0..(1usize << 16))
            .map(|j| Goldilocks::from_u64(j as u64))
            .collect::<Vec<_>>(),
    );
    let (cos, sin) = zkie_models_deepseek_v41_flash::build_rope_tables(&mut store, m, HEAD_QK);

    let o = build_mla_attention(
        &mut store,
        &mut ops,
        file.clone(),
        0,
        m,
        shift,
        q,
        kv,
        cos,
        sin,
        exp_table,
        limb_table,
    );
    let acc = build_oproj(&mut store, &mut ops, file, 0, m, shift, o);

    let mut rng = XorShift64::new(0xACE);
    let proof = prove_shard(&mut store, &ops, &[], &mut rng);
    assert!(verify_shard(&store.owned_slices(), &ops, &proof));
    println!(
        "ok full_attention m={} ops={} first8={:?}",
        m,
        ops.len(),
        store.get(acc).iter().take(8).map(|&v| to_i64(v)).collect::<Vec<_>>()
    );
}
