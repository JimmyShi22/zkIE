use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_models_deepseek_v41_flash::{build_engram, open_shard, DIM_PAD, ENGRAM_N_HASH, HC_MULT};
use zkie_ops::compose::{forward_ops, prove_shard, verify_shard, Op, Store};

fn load_i32(path: &str) -> Vec<Goldilocks> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| from_i64(i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64))
        .collect()
}

fn load_i32_raw(path: &str) -> Vec<i32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/data/zkIE/models/deepseek-v4.1-flash/raw/model-00047-of-00048.safetensors".to_string()
    });
    let hash_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/data/zkIE/models/deepseek-v4.1-flash/cache/engram/hash_ids_i32.bin".to_string());
    let layer: usize = std::env::var("ZKIE_LAYER").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let m: usize = std::env::var("ZKIE_M").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let file = open_shard(&path);
    let mut store = Store::new();
    let mut ops: Vec<Op> = Vec::new();

    let rsqrt_table = store.push(load_i32("/data/zkIE/models/gpt2/weights/rsqrt_table_i32.bin"));
    let limb_table = store.push(
        (0..(1usize << 16))
            .map(|j| Goldilocks::from_u64(j as u64))
            .collect::<Vec<_>>(),
    );
    let gate_table = store.push(load_i32("/data/zkIE/models/engram_gate_table_i32.bin"));
    let x4 = store.push(vec![from_i64(1 << 16); m * HC_MULT * DIM_PAD]);

    // hash ids are [m, 2 layers, 24]; pick this layer's 24 columns.
    let all = load_i32_raw(&hash_path);
    let col0 = if layer == 1 { 0 } else { ENGRAM_N_HASH };
    let mut hash_ids = Vec::with_capacity(m * ENGRAM_N_HASH);
    for i in 0..m {
        for j in 0..ENGRAM_N_HASH {
            hash_ids.push(all[i * (2 * ENGRAM_N_HASH) + col0 + j] as u32);
        }
    }

    let out = build_engram(
        &mut store,
        &mut ops,
        file,
        layer,
        x4,
        m,
        &hash_ids,
        16,
        rsqrt_table,
        limb_table,
        gate_table,
    );

    forward_ops(&mut store, &ops);
    let ov = store.get(out);
    println!(
        "ok engram layer={} m={} ops={} first8={:?}",
        layer,
        m,
        ops.len(),
        ov.iter().take(8).map(|&v| to_i64(v)).collect::<Vec<_>>()
    );
    let last = (m - 1) * HC_MULT * DIM_PAD;
    println!(
        "ok engram last8={:?}",
        ov[last..last + 8].iter().map(|&v| to_i64(v)).collect::<Vec<_>>()
    );

    let mut rng = XorShift64::new(0xE6E6);
    let proof = prove_shard(&mut store, &ops, &[], &mut rng);
    assert!(verify_shard(&store.owned_slices(), &ops, &proof));
    println!("ok engram verify=true");
}
