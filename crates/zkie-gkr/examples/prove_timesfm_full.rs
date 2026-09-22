//! Prove the complete TimesFM 8M: 7 layers, each with a LayerNorm and four
//! weight matmuls, all bound to WHIR commitments using real quantized data.

use zkie_gkr::committed::{commit, prove_layer_norm, prove_matmul};
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

fn load_scalars(path: &str) -> (Goldilocks, Goldilocks) {
    let bytes = std::fs::read(path).unwrap();
    let mean = f64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let rstd = f64::from_le_bytes(bytes[8..16].try_into().unwrap());
    (
        from_i32((mean * 65536.0).round() as i32),
        from_i32((rstd * 65536.0).round() as i32),
    )
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
    let mut rng = XorShift64::new(0x7ee);
    let whir9 = Whir::new_testing(9);
    let whir10 = Whir::new_testing(10);
    let whir18 = Whir::new_testing(18);
    let whir19 = Whir::new_testing(19);

    for (li, layer) in WEIGHTS.iter().enumerate() {
        // LayerNorm: raw = (x - mean) * rstd * w.
        let x = load_i32(&format!("{base}norms/L{li}_in.bin"));
        let nw = load_i32(&format!("{base}norms/L{li}_w.bin"));
        let (mean, rstd) = load_scalars(&format!("{base}norms/L{li}_scalars_f64.bin"));
        let raw: Vec<Goldilocks> = x.iter().zip(&nw).map(|(&xv, &w)| (xv - mean) * rstd * w).collect();
        let cx = commit(&whir9, &x);
        let c_raw = commit(&whir9, &raw);
        assert!(prove_layer_norm(&whir9, &cx, &whir9, &c_raw, mean, rstd, &nw));

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
            assert!(prove_matmul(win, &ci, ww, &cw, wo, &co, &input, &weight, &out, k, n, &mut rng));
        }
    }
    println!("TimesFM 8M full proof: 7 layers x (LayerNorm + 4 matmuls) verified from WHIR commitments");
}
