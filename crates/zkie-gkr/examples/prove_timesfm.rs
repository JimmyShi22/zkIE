//! Prove the full TimesFM 8M matmul chain (7 decoder layers, 28 weight matmuls)
//! from WHIR commitments, reading real quantized weights and activations.

use zkie_gkr::committed::{commit, prove_matmul};
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
    // (qkv, o_proj, gate, down) weight names per layer, and padded shapes.
    const WEIGHTS: [[&str; 4]; 7] = [
        ["val_94", "val_122", "val_126", "val_128"],
        ["val_134", "val_159", "val_163", "val_165"],
        ["val_171", "val_196", "val_200", "val_202"],
        ["val_208", "val_233", "val_237", "val_239"],
        ["val_245", "val_270", "val_274", "val_276"],
        ["val_282", "val_307", "val_311", "val_313"],
        ["val_319", "val_344", "val_348", "val_350"],
    ];
    const SHAPES: [(usize, usize); 4] = [
        (512, 1024), // qkv
        (512, 512),  // o_proj
        (512, 1024), // gate
        (1024, 512), // down
    ];
    const IN_N: [usize; 4] = [512, 512, 512, 1024];
    const OUT_N: [usize; 4] = [1024, 512, 1024, 512];
    const KEYS: [&str; 4] = ["qkv", "o_proj", "gate", "down"];

    let wdir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/weights/");
    let ldir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/layers/");
    let mut rng = XorShift64::new(0x7ee);
    let whir9 = Whir::new_testing(9);
    let whir10 = Whir::new_testing(10);
    let whir18 = Whir::new_testing(18);
    let whir19 = Whir::new_testing(19);

    for (li, layer) in WEIGHTS.iter().enumerate() {
        for (mi, key) in KEYS.iter().enumerate() {
            let (h, w) = SHAPES[mi];
            let k = IN_N[mi];
            let n = OUT_N[mi];
            let wpath = format!("{wdir}{}_{}x{}_i32.bin", layer[mi], h, w);
            let weight = load_i32(&wpath);
            let input = load_i32(&format!("{ldir}L{li}_{key}_in_{k}_i32.bin"));
            let raw = dense(&input, &weight, k, n);

            let (whir_in, whir_w, whir_out) = match k {
                512 if n == 1024 => (&whir9, &whir19, &whir10),
                512 => (&whir9, &whir18, &whir9),
                1024 => (&whir10, &whir19, &whir9),
                _ => unreachable!(),
            };
            let c_in = commit(whir_in, &input);
            let c_w = commit(whir_w, &weight);
            let c_raw = commit(whir_out, &raw);
            assert!(prove_matmul(whir_in, &c_in, whir_w, &c_w, whir_out, &c_raw, &input, &weight, &raw, 1, k, n, &mut rng, ));
        }
    }
    println!("TimesFM 8M: 7 layers x 4 matmuls verified from WHIR commitments");
}
