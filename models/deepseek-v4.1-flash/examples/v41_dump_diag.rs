//! Diagnostic: dump full `x4` (Q16 i64) after specific layers so layer-24 can
//! be re-computed in float64 and diffed against the Rust builder one layer at
//! a time. Usage:
//!   ZKIE_DUMP_LAYERS=23,24 ZKIE_LAYERS=25 ZKIE_M=16 ./v41_dump_diag <raw-dir>

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing};
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_models_deepseek_v41_flash::{
    build_engram, build_identity_pre_mix, build_rope_tables, build_rope_tables_at,
    build_v41_layer, compress_ratio, expand_to_hc, is_kv_source, layer_shard, open_shard,
    push_embedding, COMPRESS_ROPE_THETA, DIM_PAD, ENGRAM_N_HASH, HC_MULT, HEAD_QK, ROPE_THETA,
};
use zkie_ops::compose::{forward_ops, Op, Store};

fn load_i32(path: &str) -> Vec<Goldilocks> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| from_i64(i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64))
        .collect()
}

fn load_i64(path: &str) -> Vec<Goldilocks> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(8)
        .map(|c| from_i64(i64::from_le_bytes(c.try_into().unwrap())))
        .collect()
}

fn load_i32_raw(path: &str) -> Vec<i32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn dump_i64(path: &str, vals: &[Goldilocks]) {
    let out: Vec<i64> = vals.iter().map(|&v| to_i64(v)).collect();
    let bytes: Vec<u8> = out
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(path, bytes).unwrap();
    eprintln!("wrote {} ({} i64)", path, out.len());
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/zkIE/models/deepseek-v4.1-flash/raw".to_string());
    let m: usize = std::env::var("ZKIE_M").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let n_layers: usize = std::env::var("ZKIE_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(25);
    let routed = std::env::var("ZKIE_NO_ROUTED").is_err();
    let dump_layers: Vec<usize> = std::env::var("ZKIE_DUMP_LAYERS")
        .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
        .unwrap_or_default();
    let dump_dir = std::env::var("ZKIE_DUMP_DIR").unwrap_or_else(|_| "/tmp".to_string());

    let mut store = Store::new();
    let mut ops: Vec<Op> = Vec::new();
    let exp_table = store.push(load_i32("/data/zkIE/models/gpt2/weights/exp_table_i32.bin"));
    let exp_q32 = store.push(load_i64("/data/zkIE/models/exp_table_q32_i64.bin"));
    let rsqrt_table = store.push(load_i32("/data/zkIE/models/gpt2/weights/rsqrt_table_i32.bin"));
    let limb_table = store.push(
        (0..(1usize << 16)).map(|j| Goldilocks::from_u64(j as u64)).collect::<Vec<_>>(),
    );
    let sqrt_sp = store.push(load_i32("/data/zkIE/models/sqrt_softplus_table_i32.bin"));
    let silu = store.push(load_i32("/data/zkIE/models/silu_clipped_table_i32.bin"));
    let clip = store.push(load_i32("/data/zkIE/models/clip_m10_p10_table_i32.bin"));
    let sigmoid = store.push(load_i32("/data/zkIE/models/sigmoid_table_i32.bin"));
    let gate_table = store.push(load_i32("/data/zkIE/models/engram_gate_table_i32.bin"));
    let hash_ids = load_i32_raw("/data/zkIE/models/deepseek-v4.1-flash/cache/engram/hash_ids_i32.bin");
    let hash_ids: Vec<u32> = hash_ids.into_iter().map(|v| v as u32).collect();
    let x = push_embedding(
        &mut store,
        "/data/zkIE/models/deepseek-v4.1-flash/cache/engram/embed_prompt_i32.bin",
        m,
    );

    let x4 = expand_to_hc(&mut store, &mut ops, x, m, HC_MULT, DIM_PAD);
    let mut pre_mix = build_identity_pre_mix(&mut store, m);
    let mut x4 = x4;
    let mut compress_kv: Option<usize> = None;
    let shift = 16u32;
    let mut done = 0usize;

    for layer in 0..n_layers.min(40) {
        if layer == 1 || layer == 14 {
            let eng_shard = open_shard(&format!(
                "{dir}/model-{:05}-of-00048.safetensors",
                if layer == 1 { 47 } else { 48 }
            ));
            let col0 = if layer == 1 { 0 } else { ENGRAM_N_HASH };
            let mut layer_hash = Vec::with_capacity(m * ENGRAM_N_HASH);
            for i in 0..m {
                for j in 0..ENGRAM_N_HASH {
                    layer_hash.push(hash_ids[i * (2 * ENGRAM_N_HASH) + col0 + j]);
                }
            }
            x4 = build_engram(
                &mut store, &mut ops, eng_shard, layer, x4, m, &layer_hash, shift,
                rsqrt_table, limb_table, gate_table,
            );
        }
        let ratio = compress_ratio(layer);
        let file = open_shard(layer_shard(dir.as_str(), layer).as_str());
        let rope_theta = if ratio > 0 { COMPRESS_ROPE_THETA } else { ROPE_THETA };
        let (cos, sin) = build_rope_tables(&mut store, m, HEAD_QK, rope_theta);
        let groups = if ratio > 0 { m / ratio } else { 0 };
        let cpos: Vec<usize> = (0..groups).map(|g| g * ratio).collect();
        let (ccos, csin) = build_rope_tables_at(&mut store, &cpos, HEAD_QK, rope_theta);
        let (x4o, pre_mixo, new_ckv) = build_v41_layer(
            &mut store, &mut ops, file, layer, m, ratio, routed, shift, x4, pre_mix,
            compress_kv, cos, sin, ccos, csin, exp_table, exp_q32, rsqrt_table, limb_table,
            sqrt_sp, silu, clip, sigmoid,
        );
        x4 = x4o;
        pre_mix = pre_mixo;
        if new_ckv.is_some() {
            compress_kv = new_ckv;
        }
        forward_ops(&mut store, &ops[done..]);
        done = ops.len();
        let out = store.get(x4);
        println!(
            "L{} first8 {:?}",
            layer,
            out.iter().take(8).map(|&v| to_i64(v)).collect::<Vec<_>>()
        );
        {
            let mut tokmax = Vec::with_capacity(m);
            for i in 0..m {
                let mut mx: i64 = 0;
                for h in 0..HC_MULT {
                    for j in 0..DIM_PAD {
                        let v = to_i64(out[i * HC_MULT * DIM_PAD + h * DIM_PAD + j]).unsigned_abs() as i64;
                        if v > mx {
                            mx = v;
                        }
                    }
                }
                tokmax.push(mx as f64 / 65536.0);
            }
            println!("L{} tokmax {:?}", layer, tokmax);
        }
        if dump_layers.contains(&layer) {
            dump_i64(
                &format!("{}/x4_L{:02}.bin", dump_dir, layer),
                store.get(x4),
            );
            dump_i64(
                &format!("{}/pre_mix_L{:02}.bin", dump_dir, layer),
                store.get(pre_mix),
            );
            if let Some(ckv) = compress_kv {
                dump_i64(
                    &format!("{}/compress_kv_L{:02}.bin", dump_dir, layer),
                    store.get(ckv),
                );
            }
        }
    }
}
