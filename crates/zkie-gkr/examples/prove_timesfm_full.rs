//! Prove the complete TimesFM 8M: 7 layers, each with a LayerNorm and four
//! weight matmuls, all bound to WHIR commitments using real quantized data.

use zkie_gkr::committed::{commit, layer_norm_raw, prove_layer_norm, prove_matmul};
use zkie_gkr::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_gkr::fixed_point::from_i32;
use zkie_gkr::whir::Whir;

fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = std::fs::read(path).expect("run the extraction scripts first");
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let v = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.push(from_i32(v));
    }
    out
}

fn dense(a: &[Goldilocks], b: &[Goldilocks], k: usize, n: usize) -> Vec<Goldilocks> {
    let mut c = vec![Goldilocks::ZERO; n];
    for j in 0..n {
        let mut acc = Goldilocks::ZERO;
        for w in 0..k {
            acc = acc + a[w] * b[w * n + j];
        }
        c[j] = acc;
    }
    c
}

fn main() {
    const WEIGHTS: [[&str; 4]; 7] = [
        ["val_94", "val_122", "val_126", "val_128"],
        ["val_134", "val_159", "val_163", "val_165"],
        ["val_171", "val_196", "val_200", "val_202"],
        ["val_208", "val_233", "val_237", "val_239"],
        ["val_245", "val_270", "val_274", "val_276"],
        ["val_282", "val_307", "val_311", "val_313"],
        ["val_319", "val_344", "val_348", "val_350"],
    ];
    const SHAPES: [(usize, usize); 4] = [(512, 1024), (512, 512), (512, 1024), (1024, 512)];
    const IN_N: [usize; 4] = [512, 512, 512, 1024];
    const OUT_N: [usize; 4] = [1024, 512, 1024, 512];
    const KEYS: [&str; 4] = ["qkv", "o_proj", "gate", "down"];

    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/");
    // rsqrt lookup table: index at scale 2^14, rstd at scale 2^16.
    let rsqrt_table = load_i32(&format!("{base}rsqrt_table_i32.bin"));
    // TimesFM hidden dim is 264, padded to 512 for the WHIR power-of-two length.
    const N_REAL: usize = 264;
    let mut rng = XorShift64::new(0x7ee);
    let whir9 = Whir::new_testing(9);
    let whir10 = Whir::new_testing(10);
    let whir18 = Whir::new_testing(18);
    let whir19 = Whir::new_testing(19);

    for (li, layer) in WEIGHTS.iter().enumerate() {
        // LayerNorm: mean is derived from committed x, rstd is bound to var+eps
        // by an rsqrt lookup (not a trusted scalar).
        let x = load_i32(&format!("{base}norms/L{li}_in.bin"));
        let nw = load_i32(&format!("{base}norms/L{li}_w.bin"));
        let (raw, _mean, _rstd, _s_index) = layer_norm_raw(&x, &nw, N_REAL, &rsqrt_table);
        let cx = commit(&whir9, &x);
        let c_raw = commit(&whir9, &raw);
        let alpha = rng.field();
        let beta = rng.field();
        assert!(prove_layer_norm(
            &whir9,
            &cx,
            &x,
            &whir9,
            &c_raw,
            &raw,
            &nw,
            N_REAL,
            &rsqrt_table,
            alpha,
            beta,
            &mut rng,
        ));

        // Four weight matmuls.
        for (mi, key) in KEYS.iter().enumerate() {
            let (h, w) = SHAPES[mi];
            let k = IN_N[mi];
            let n = OUT_N[mi];
            let weight = load_i32(&format!("{base}weights/{}_{}x{}_i32.bin", layer[mi], h, w));
            let input = load_i32(&format!("{base}layers/L{li}_{key}_in_{k}_i32.bin"));
            let out = dense(&input, &weight, k, n);
            let (win, ww, wo) = match k {
                512 if n == 1024 => (&whir9, &whir19, &whir10),
                512 => (&whir9, &whir18, &whir9),
                1024 => (&whir10, &whir19, &whir9),
                _ => unreachable!(),
            };
            let ci = commit(win, &input);
            let cw = commit(ww, &weight);
            let co = commit(wo, &out);
            assert!(prove_matmul(win, &ci, ww, &cw, wo, &co, &input, &weight, &out, 1, k, n, &mut rng));
        }
    }
    println!("TimesFM 8M full proof: 7 layers x (LayerNorm + 4 matmuls) verified from WHIR commitments");
}
