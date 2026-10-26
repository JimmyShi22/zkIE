use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing};
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_models_deepseek_v41_flash::{
    build_final_logits, build_v41_forward, push_embedding, DIM_PAD, HC_MULT,
};
use zkie_ops::compose::{forward_ops, Op, Store};

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
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/zkIE/models/deepseek-v4.1-flash/raw".to_string());
    let m: usize = std::env::var("ZKIE_M")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let n_layers: usize = std::env::var("ZKIE_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let routed = std::env::var("ZKIE_NO_ROUTED").is_err();

    let mut store = Store::new();
    let mut ops: Vec<Op> = Vec::new();
    let exp_table = store.push(load_i32("/data/zkIE/models/gpt2/weights/exp_table_i32.bin"));
    let rsqrt_table = store.push(load_i32("/data/zkIE/models/gpt2/weights/rsqrt_table_i32.bin"));
    let limb_table = store.push(
        (0..(1usize << 16))
            .map(|j| Goldilocks::from_u64(j as u64))
            .collect::<Vec<_>>(),
    );
    let sqrt_sp = store.push(load_i32("/data/zkIE/models/sqrt_softplus_table_i32.bin"));
    let silu = store.push(load_i32("/data/zkIE/models/silu_clipped_table_i32.bin"));
    let clip = store.push(load_i32("/data/zkIE/models/clip_m10_p10_table_i32.bin"));
    let gate_table = store.push(load_i32("/data/zkIE/models/engram_gate_table_i32.bin"));
    let hash_ids = load_i32_raw("/data/zkIE/models/deepseek-v4.1-flash/cache/engram/hash_ids_i32.bin");
    let hash_ids: Vec<u32> = hash_ids.into_iter().map(|v| v as u32).collect();
    let x = if std::env::var("ZKIE_REAL").is_ok() {
        push_embedding(
            &mut store,
            "/data/zkIE/models/deepseek-v4.1-flash/cache/engram/embed_prompt_i32.bin",
            m,
        )
    } else {
        store.push(vec![from_i64(1 << 16); m * DIM_PAD])
    };

    let (x4, _pre_mix) = build_v41_forward(
        &mut store,
        &mut ops,
        &dir,
        m,
        16,
        routed,
        n_layers,
        x,
        exp_table,
        rsqrt_table,
        limb_table,
        sqrt_sp,
        silu,
        clip,
        gate_table,
        &hash_ids,
    );

    forward_ops(&mut store, &ops);
    let out = store.get(x4);
    println!(
        "ok v41_forward layers={} m={} routed={} ops={} first8={:?}",
        n_layers,
        m,
        routed,
        ops.len(),
        out.iter().take(8).map(|&v| to_i64(v)).collect::<Vec<_>>()
    );
    let last = (m - 1) * HC_MULT * DIM_PAD;
    println!(
        "ok v41_forward last8={:?}",
        out[last..last + 8].iter().map(|&v| to_i64(v)).collect::<Vec<_>>()
    );

    if n_layers >= 40 {
        let (_logits, argmax) = build_final_logits(
            &mut store,
            &mut ops,
            &dir,
            m,
            16,
            x4,
            _pre_mix,
            rsqrt_table,
            limb_table,
        );
        forward_ops(&mut store, &ops);
        let am = &store.idx[argmax];
        println!("argmax={:?}", &am[..m.min(am.len())]);
    }
}
