use zkie_core::common::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::{from_i64, to_i64};
use zkie_models_deepseek_v41_flash::{build_hc_mixes, open_shard, HC_MULT};
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
    let m = 1usize;
    let hc = HC_MULT;
    let d = 5120usize;
    let d_real = hc * d; // 20480
    let d_pad = d_real.next_power_of_two(); // 32768
    let mix_hc = (2 + hc) * hc; // 24
    let mix_hc_pad = mix_hc.next_power_of_two(); // 32
    let file = open_shard(&path);

    let mut store = Store::new();
    let rsqrt_table = store.push(load_i32("/data/zkIE/models/gpt2/weights/rsqrt_table_i32.bin"));
    let limb_table = store.push(
        (0..(1usize << 16))
            .map(|j| Goldilocks::from_u64(j as u64))
            .collect::<Vec<_>>(),
    );
    // x4 = ones over the real 4-stream flatten, row-padded to d_pad.
    let mut x4 = Vec::with_capacity(m * d_pad);
    for _ in 0..m {
        x4.extend(std::iter::repeat(from_i64(1 << 16)).take(d_real));
        x4.extend(std::iter::repeat(Goldilocks::ZERO).take(d_pad - d_real));
    }
    let x4_pad = store.push(x4);
    let hc_fn_t = {
        // hc_attn_fn is [24, 20480]; transpose to [d_pad, mix_hc_pad].
        let mut out = vec![Goldilocks::ZERO; d_pad * mix_hc_pad];
        let v: Vec<i64> = file
            .read_f32("layers.0.hc_attn_fn")
            .iter()
            .map(|&x| to_i64(x))
            .collect();
        for r in 0..mix_hc {
            for c in 0..d_real {
                out[c * mix_hc_pad + r] = from_i64(v[r * d_real + c]);
            }
        }
        store.push(out)
    };
    let hc_base = store.push(file.read_f32("layers.0.hc_attn_base"));
    let hc_scale = store.push(file.read_f32("layers.0.hc_attn_scale"));

    let mut ops: Vec<Op> = Vec::new();
    let mix = build_hc_mixes(
        &mut store,
        &mut ops,
        x4_pad,
        hc_fn_t,
        m,
        d_real,
        d_pad,
        mix_hc_pad,
        rsqrt_table,
        limb_table,
    );
    let pre = store.push(vec![]);
    let post = store.push(vec![]);
    let comb = store.push(vec![]);
    ops.push(Op::SinkhornSplit {
        mixes: mix,
        scale: hc_scale,
        base: hc_base,
        pre,
        post,
        comb,
        m,
        hc,
        iters: 20,
        eps: 1,
    });

    let mut rng = XorShift64::new(0xABC1);
    let proof = prove_shard(&mut store, &ops, &[], &mut rng);
    assert!(verify_shard(&[], &ops, &proof));
    println!(
        "mix[0,:4]={:?}",
        store.get(mix).iter().take(4).map(|&v| to_i64(v)).collect::<Vec<_>>()
    );
    println!(
        "pre={:?}",
        store.get(pre).iter().take(4).map(|&v| to_i64(v)).collect::<Vec<_>>()
    );
    println!(
        "post={:?}",
        store.get(post).iter().take(4).map(|&v| to_i64(v)).collect::<Vec<_>>()
    );
    println!(
        "comb[0,:4]={:?}",
        store.get(comb).iter().take(4).map(|&v| to_i64(v)).collect::<Vec<_>>()
    );
}
