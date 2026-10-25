//! Direct f64 reference for layer-0 MLA attention. Compares the first few
//! fixed-point logits against `scripts/v41_ref_cpu.py --mode attention`.

use zkie_core::common::fixed_point::to_i64;
use zkie_core::common::quantized_io::SafetensorFile;

fn f64s(v: &[zkie_core::common::field::Goldilocks]) -> Vec<f64> {
    v.iter().map(|&x| to_i64(x) as f64 / 65536.0).collect()
}

fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    // b is n x k; result is m x n.
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for kk in 0..k {
                s += a[i * k + kk] * b[j * k + kk];
            }
            out[i * n + j] = s;
        }
    }
    out
}

fn rms_norm(x: &[f64], w: &[f64], m: usize, d: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * d];
    for i in 0..m {
        let mean_sq = (0..d).map(|j| x[i * d + j] * x[i * d + j]).sum::<f64>() / d as f64;
        let rsqrt = 1.0 / (mean_sq + 1e-20).sqrt();
        for j in 0..d {
            out[i * d + j] = x[i * d + j] * rsqrt * w[j];
        }
    }
    out
}

fn rope(x: &mut [f64], m: usize, heads: usize, d: usize, qk_rope: usize) {
    for p in 0..m {
        for h in 0..heads {
            let base = p * heads * d + h * d;
            for i in 0..qk_rope / 2 {
                let theta = p as f64 * (10000f64).powf(-2.0 * i as f64 / qk_rope as f64);
                let (c, s) = (theta.cos(), theta.sin());
                let off = d - qk_rope;
                let a = x[base + off + i];
                let b = x[base + off + i + qk_rope / 2];
                x[base + off + i] = a * c - b * s;
                x[base + off + i + qk_rope / 2] = a * s + b * c;
            }
        }
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/zkIE/models/deepseek-v4.1-flash/raw/model-00003-of-00048.safetensors".to_string());
    let file = SafetensorFile::open(&path).unwrap();
    let m = 16usize;
    let dim = 5120usize;
    let dim_pad = 8192usize;
    let heads = 64usize;
    let head_dim = 512usize;
    let q_lora = 1280usize;
    let q_lora_pad = 2048usize;
    let qk_rope = 64usize;

    let wq_a = f64s(&file.read_fp8_weight("layers.0.attn.wq_a.weight", "layers.0.attn.wq_a.scale", 32));
    let wq_b = f64s(&file.read_fp8_weight("layers.0.attn.wq_b.weight", "layers.0.attn.wq_b.scale", 32));
    let wkv = f64s(&file.read_fp8_weight("layers.0.attn.wkv.weight", "layers.0.attn.wkv.scale", 32));
    let wo_a = f64s(&file.read_fp8_weight("layers.0.attn.wo_a.weight", "layers.0.attn.wo_a.scale", 32));
    let wo_b = f64s(&file.read_fp8_weight("layers.0.attn.wo_b.weight", "layers.0.attn.wo_b.scale", 32));
    let q_norm_real = f64s(&file.read_bf16("layers.0.attn.q_norm.weight"));
    let kv_norm = f64s(&file.read_bf16("layers.0.attn.kv_norm.weight"));
    let sink_raw = file.read_f32("layers.0.attn.attn_sink");
    let sink: Vec<f64> = sink_raw.iter().map(|&x| to_i64(x) as f64 / 65536.0).collect();

    let mut q_norm = vec![0.0; q_lora_pad];
    q_norm[..q_lora].copy_from_slice(&q_norm_real);

    let mut x = vec![0.0; m * dim_pad];
    for i in 0..m {
        for j in 0..dim {
            x[i * dim_pad + j] = 1.0;
        }
    }
    let qr = rms_norm(&matmul(&x, &wq_a, m, dim_pad, q_lora_pad), &q_norm, m, q_lora_pad);
    let mut q = matmul(&qr, &wq_b, m, q_lora_pad, heads * head_dim);
    rope(&mut q, m, heads, head_dim, qk_rope);
    let mut kv = rms_norm(&matmul(&x, &wkv, m, dim_pad, head_dim), &kv_norm, m, head_dim);
    rope(&mut kv, m, 1, head_dim, qk_rope);

    // Dense causal attention; for seq 16 this equals the layer-0 sliding window.
    let mut o = vec![0.0; m * heads * head_dim];
    let scale = (head_dim as f64).powf(-0.5);
    for p in 0..m {
        for h in 0..heads {
            let mut scores = vec![0.0; p + 2];
            for j in 0..=p {
                let mut s = 0.0;
                for k in 0..head_dim {
                    s += q[(p * heads + h) * head_dim + k] * kv[j * head_dim + k];
                }
                scores[j] = s * scale;
            }
            scores[p + 1] = sink[h];
            let max = scores.iter().cloned().fold(f64::MIN, f64::max);
            let probs: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f64 = probs.iter().sum();
            for k in 0..head_dim {
                let mut acc = 0.0;
                for j in 0..=p {
                    acc += probs[j] * kv[j * head_dim + k];
                }
                o[(p * heads + h) * head_dim + k] = acc / sum;
            }
        }
    }

    // wo_a grouped, then wo_b.
    let groups = 8usize;
    let o_lora = 1024usize;
    let mut oa = vec![0.0; m * groups * o_lora];
    for p in 0..m {
        for g in 0..groups {
            for r in 0..o_lora {
                let mut s = 0.0;
                let group_dim = heads * head_dim / groups;
                let base = p * heads * head_dim + g * group_dim;
                for d in 0..group_dim {
                    s += o[base + d] * wo_a[(g * o_lora + r) * group_dim + d];
                }
                oa[(p * groups + g) * o_lora + r] = s;
            }
        }
    }
    let out = matmul(&oa, &wo_b, m, groups * o_lora, dim_pad);
    for i in 0..4 {
        println!("[{i}] {}", (out[i] * 65536.0).round() as i64);
    }
}
