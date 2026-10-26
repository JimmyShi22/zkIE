use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_models_deepseek_v41_flash::{build_compressor, open_shard, DIM_PAD};
use zkie_ops::compose::{forward_ops, prove_shard, verify_shard, Op, Store};

fn load_i32(path: &str) -> Vec<Goldilocks> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| from_i64(i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64))
        .collect()
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/data/zkIE/models/deepseek-v4.1-flash/raw/model-00005-of-00048.safetensors".to_string()
    });
    let m: usize = std::env::var("ZKIE_M")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let ratio: usize = std::env::var("ZKIE_RATIO")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let file = open_shard(&path);
    let mut store = Store::new();
    let mut ops: Vec<Op> = Vec::new();

    let rsqrt_table = store.push(load_i32("/data/zkIE/models/gpt2/weights/rsqrt_table_i32.bin"));
    let limb_table = store.push(
        (0..(1usize << 16))
            .map(|j| Goldilocks::from_u64(j as u64))
            .collect::<Vec<_>>(),
    );
    let exp_table = store.push(load_i32("/data/zkIE/models/gpt2/weights/exp_table_i32.bin"));
    let x = store.push(vec![from_i64(1 << 16); m * DIM_PAD]);

    let latent = build_compressor(
        &mut store,
        &mut ops,
        file,
        2,
        x,
        m,
        ratio,
        16,
        rsqrt_table,
        limb_table,
        exp_table,
    );

    forward_ops(&mut store, &ops);
    println!(
        "ok compressor m={} ratio={} ops={} first8={:?}",
        m,
        ratio,
        ops.len(),
        store.get(latent).iter().take(8).map(|&v| to_i64(v)).collect::<Vec<_>>()
    );

    let mut rng = XorShift64::new(0xC0FF);
    let proof = prove_shard(&mut store, &ops, &[], &mut rng);
    assert!(verify_shard(&store.owned_slices(), &ops, &proof));
    println!("ok compressor verify=true");
}
