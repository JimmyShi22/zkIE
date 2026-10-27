use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_models_deepseek_v41_flash::{
    build_block, build_identity_pre_mix, expand_to_hc, open_shard, DIM_PAD, HC_MULT,
};
use zkie_ops::compose::{forward_ops, prove_shard, verify_shard, Op, Store};

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

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/data/zkIE/models/deepseek-v4.1-flash/raw/model-00003-of-00048.safetensors".to_string()
    });
    let m: usize = std::env::var("ZKIE_M")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let shift = 16u32;
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
    let exp_q32 = store.push(load_i64("/data/zkIE/models/exp_table_q32_i64.bin"));
    let sqrt_sp = store.push(load_i32("/data/zkIE/models/sqrt_softplus_table_i32.bin"));
    let silu = store.push(load_i32("/data/zkIE/models/silu_clipped_table_i32.bin"));
    let clip = store.push(load_i32("/data/zkIE/models/clip_m10_p10_table_i32.bin"));
    let sigmoid = store.push(load_i32("/data/zkIE/models/sigmoid_table_i32.bin"));

    // All-ones embedding input, expanded to the four HC residual copies.
    let x = store.push(vec![from_i64(1 << 16); m * DIM_PAD]);
    let x4 = expand_to_hc(&mut store, &mut ops, x, m, HC_MULT, DIM_PAD);
    let pre_mix = build_identity_pre_mix(&mut store, m);
    let routed = std::env::var("ZKIE_NO_ROUTED").is_err();

    let (x4_out, ffn_pre, _x4_attn, ffn_out) = build_block(
        &mut store,
        &mut ops,
        file,
        0,
        m,
        shift,
        routed,
        x4,
        pre_mix,
        rsqrt_table,
        limb_table,
        exp_table,
        exp_q32,
        sqrt_sp,
        silu,
        clip,
        sigmoid,
    );

    forward_ops(&mut store, &ops);
    let out = store.get(x4_out);
    println!(
        "ok block m={} ops={} first8={:?}",
        m,
        ops.len(),
        out.iter().take(8).map(|&v| to_i64(v)).collect::<Vec<_>>()
    );
    println!(
        "ok block last_token_first8={:?}",
        out[(m - 1) * HC_MULT * DIM_PAD..(m - 1) * HC_MULT * DIM_PAD + 8]
            .iter()
            .map(|&v| to_i64(v))
            .collect::<Vec<_>>()
    );
    println!(
        "ffn_pre={:?}",
        store.get(ffn_pre).iter().take(4).map(|&v| to_i64(v)).collect::<Vec<_>>()
    );
    println!(
        "ffn_out_first8={:?}",
        store.get(ffn_out).iter().take(8).map(|&v| to_i64(v)).collect::<Vec<_>>()
    );

    if std::env::var("ZKIE_FORWARD_ONLY").is_ok() {
        return;
    }
    let mut rng = XorShift64::new(0xB10C);
    let proof = prove_shard(&mut store, &ops, &[], &mut rng);
    assert!(verify_shard(&store.owned_slices(), &ops, &proof));
    println!("ok block verify=true");
}
