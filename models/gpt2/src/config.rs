//! GPT-2 124M model configuration, graph and authoritative asset bindings.
//!
//! All paths are relative to an explicitly supplied asset root. The graph
//! declares every stage, its opcode, explicit scale and the asset role it
//! consumes.

use std::path::Path;

use zkie_core::error::{Error, Result};
use zkie_models_common::assets::AssetSpec;
use zkie_models_common::gt::GtStatus;
use zkie_models_common::{GraphNodeSpec, ModelGraph};
use zkie_ops::op::OpCode;

pub const LAYERS: usize = 12;
pub const HEADS: usize = 12;
pub const HEAD_DIM: usize = 64;
pub const HIDDEN: usize = 768;
pub const HIDDEN_PAD: usize = 1024;
pub const FFN: usize = 4096;
pub const VOCAB: usize = 50257;
pub const VOCAB_PAD: usize = 65536;
pub const MAX_SEQUENCE: usize = 512;
pub const BASE_SHIFT: u32 = 16;
pub const QK_SCALE_FACTOR: i64 = 23170;
pub const QK_SCALE_SHIFT: u32 = 16;
pub const EXP_OFFSET: i64 = 1 << 21;
pub const EXP_TABLE_LEN: usize = 1 << 21;
pub const GELU_OFFSET: i64 = 1 << 23;
pub const GELU_TABLE_LEN: usize = 1 << 24;
pub const RSQRT_INDEX_SHIFT: u32 = 18;
pub const NORM_OUTPUT_SHIFT: u32 = 32;
pub const CAUSAL_MASK: i64 = -(1 << 30);
pub const VOCAB_MASK: i64 = -(1 << 41);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gpt2Config {
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub hidden: usize,
    pub hidden_padded: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub vocab_padded: usize,
    pub max_sequence: usize,
}

impl Default for Gpt2Config {
    fn default() -> Self {
        Self {
            layers: LAYERS,
            heads: HEADS,
            head_dim: HEAD_DIM,
            hidden: HIDDEN,
            hidden_padded: HIDDEN_PAD,
            ffn: FFN,
            vocab: VOCAB,
            vocab_padded: VOCAB_PAD,
            max_sequence: MAX_SEQUENCE,
        }
    }
}

impl Gpt2Config {
    pub fn validate(&self) -> Result<()> {
        if self.layers != LAYERS
            || self.heads * self.head_dim != self.hidden
            || self.hidden > self.hidden_padded
            || self.vocab > self.vocab_padded
            || self.max_sequence != MAX_SEQUENCE
        {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }

    pub fn asset_specs(&self, root: &Path) -> Result<Vec<AssetSpec>> {
        self.validate()?;
        let mut specs = Vec::new();
        let layer_shapes: [(&str, Vec<u64>, &str); 16] = [
            ("q_w", vec![HIDDEN_PAD as u64, HIDDEN_PAD as u64], "i32-le"),
            ("q_b", vec![HIDDEN_PAD as u64], "i32-le"),
            ("k_w", vec![HIDDEN_PAD as u64, HIDDEN_PAD as u64], "i32-le"),
            ("k_b", vec![HIDDEN_PAD as u64], "i32-le"),
            ("v_w", vec![HIDDEN_PAD as u64, HIDDEN_PAD as u64], "i32-le"),
            ("v_b", vec![HIDDEN_PAD as u64], "i32-le"),
            ("o_proj_w", vec![HIDDEN_PAD as u64, HIDDEN_PAD as u64], "i32-le"),
            ("o_proj_b", vec![HIDDEN_PAD as u64], "i32-le"),
            ("ln1_w", vec![HIDDEN_PAD as u64], "i32-le"),
            ("ln1_b", vec![HIDDEN_PAD as u64], "i32-le"),
            ("ln2_w", vec![HIDDEN_PAD as u64], "i32-le"),
            ("ln2_b", vec![HIDDEN_PAD as u64], "i32-le"),
            ("fc_w", vec![HIDDEN_PAD as u64, FFN as u64], "i32-le"),
            ("fc_b", vec![FFN as u64], "i32-le"),
            ("proj_w", vec![FFN as u64, HIDDEN_PAD as u64], "i32-le"),
            ("proj_b", vec![HIDDEN_PAD as u64], "i32-le"),
        ];
        for layer in 0..LAYERS {
            for (suffix, shape, encoding) in &layer_shapes {
                let relative = format!("models/gpt2/weights/L{layer}_{suffix}_i32.bin");
                specs.push(spec(root, &relative, Some(shape.clone()), encoding)?);
            }
        }
        for (name, shape, encoding) in [
            ("lm_head_w_i32", vec![HIDDEN_PAD as u64, VOCAB_PAD as u64], "i32-le"),
            ("pos_emb_i32", vec![MAX_SEQUENCE as u64, HIDDEN_PAD as u64], "i32-le"),
            ("ln_f_w_i32", vec![HIDDEN_PAD as u64], "i32-le"),
            ("ln_f_b_i32", vec![HIDDEN_PAD as u64], "i32-le"),
            ("mask_i32", vec![MAX_SEQUENCE as u64, MAX_SEQUENCE as u64], "i32-le"),
            ("embedding_i32", vec![MAX_SEQUENCE as u64, HIDDEN_PAD as u64], "i32-le"),
            ("exp_table_i32", vec![2_097_152], "i32-le"),
            ("rsqrt_table_i32", vec![7_340_032], "i32-le"),
            ("gelu_table_i32", vec![16_777_216], "i32-le"),
        ] {
            specs.push(spec(root, &format!("models/gpt2/weights/{name}.bin"), Some(shape), encoding)?);
        }
        for name in ["tokenizer/tokenizer.json", "tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer/tokenizer_config.json"] {
            let relative = format!("models/gpt2/{name}");
            specs.push(spec(root, &relative, None, "utf8")?);
        }
        for (name, encoding) in [("tokens_16_i32.bin", "i32-le"), ("tokens_512_i32.bin", "i32-le"), ("input_text.txt", "utf8"), ("case_meta.json", "utf8")] {
            specs.push(spec(root, &format!("cases/gpt2/{name}"), None, encoding)?);
        }
        for (name, shape, encoding) in [
            ("gt_logits_f32.bin", vec![1, 512, 50257], "f32-le"),
            ("gt_argmax_16_i32.bin", vec![16], "i32-le"),
            ("gt_argmax_512_i32.bin", vec![512], "i32-le"),
            ("gt_int_argmax_16_i32.bin", vec![16], "i32-le"),
            ("gt_int_argmax_512_i32.bin", vec![512], "i32-le"),
        ] {
            specs.push(spec(root, &format!("cases/gpt2/{name}"), Some(shape), encoding)?);
        }
        Ok(specs)
    }

    pub fn manifest(&self, root: &Path) -> Result<zkie_models_common::assets::AssetManifest> {
        zkie_models_common::assets::AssetManifest::load(root, &self.asset_specs(root)?)
    }

    pub fn gt_status(&self) -> (GtStatus, GtStatus) {
        (GtStatus::Available, GtStatus::Available)
    }

    pub fn graph(&self) -> ModelGraph {
        let mut nodes = Vec::new();
        nodes.push(node("embedding", OpCode::TableLookup, vec!["prompt_tokens".into(), "embedding_table".into()], vec!["h0".into()], None, "lm_head_w (tied embedding)"));
        nodes.push(node("position_add", OpCode::Add, vec!["h0".into(), "pos_emb".into()], vec!["x0".into()], Some(16), "pos_emb"));
        let mut current = "x0".to_owned();
        for layer in 0..LAYERS {
            let ln1 = format!("L{layer}.ln1");
            let normed = format!("L{layer}.n1");
            push(&mut nodes, node(&ln1, OpCode::LayerNorm, vec![current.clone()], vec![normed.clone()], Some(16), "ln1_w/ln1_b"));
            let q = format!("L{layer}.q");
            let k = format!("L{layer}.k");
            let v = format!("L{layer}.v");
            push(&mut nodes, node(&q, OpCode::Projection, vec![normed.clone()], vec![format!("L{layer}.q_out")], Some(16), "q_w/q_b"));
            push(&mut nodes, node(&k, OpCode::Projection, vec![normed.clone()], vec![format!("L{layer}.k_out")], Some(16), "k_w/k_b"));
            push(&mut nodes, node(&v, OpCode::Projection, vec![normed.clone()], vec![format!("L{layer}.v_out")], Some(16), "v_w/v_b"));
            let attn = format!("L{layer}.attn");
            push(&mut nodes, node(&attn, OpCode::CausalAttention, vec![format!("L{layer}.q_out"), format!("L{layer}.k_out"), format!("L{layer}.v_out"), "exp_table".into(), "mask".into()], vec![format!("L{layer}.attn_out")], Some(16), "exp/rsqrt tables"));
            let o = format!("L{layer}.o");
            push(&mut nodes, node(&o, OpCode::Projection, vec![format!("L{layer}.attn_out")], vec![format!("L{layer}.o_out")], Some(16), "o_proj_w/o_proj_b"));
            let residual = format!("L{layer}.residual_attn");
            push(&mut nodes, node(&residual, OpCode::Add, vec![current.clone(), format!("L{layer}.o_out")], vec![format!("L{layer}.x1")], None, "residual"));
            let ln2 = format!("L{layer}.ln2");
            let normed2 = format!("L{layer}.n2");
            push(&mut nodes, node(&ln2, OpCode::LayerNorm, vec![format!("L{layer}.x1")], vec![normed2.clone()], Some(16), "ln2_w/ln2_b"));
            let fc = format!("L{layer}.fc");
            push(&mut nodes, node(&fc, OpCode::Projection, vec![normed2], vec![format!("L{layer}.fc_out")], Some(16), "fc_w/fc_b"));
            let gelu = format!("L{layer}.gelu");
            push(&mut nodes, node(&gelu, OpCode::TableLookup, vec![format!("L{layer}.fc_out"), "gelu_idx".into(), "gelu_table".into()], vec![format!("L{layer}.gelu_out")], Some(16), "gelu_table_i32"));
            let proj = format!("L{layer}.proj");
            push(&mut nodes, node(&proj, OpCode::Projection, vec![format!("L{layer}.gelu_out")], vec![format!("L{layer}.proj_out")], Some(16), "proj_w/proj_b"));
            let residual2 = format!("L{layer}.residual_ffn");
            push(&mut nodes, node(&residual2, OpCode::Add, vec![format!("L{layer}.x1"), format!("L{layer}.proj_out")], vec![format!("L{layer}.x2")], None, "residual"));
            current = format!("L{layer}.x2");
        }
        push(&mut nodes, node("final_norm", OpCode::LayerNorm, vec![current], vec!["final_normed".into()], Some(16), "ln_f_w/ln_f_b"));
        push(&mut nodes, node("lm_head", OpCode::MatMul, vec!["final_normed".into(), "lm_head_w".into()], vec!["logits".into()], Some(16), "lm_head_w"));
        push(&mut nodes, node("argmax", OpCode::Argmax, vec!["logits".into()], vec!["next_token".into()], None, "smallest-index tie break"));
        ModelGraph {
            model_id: "gpt2-124m".into(),
            external_inputs: vec!["prompt_tokens".into(), "pos_emb".into(), "gelu_idx".into(), "embedding_table".into(), "exp_table".into(), "mask".into(), "gelu_table".into(), "lm_head_w".into()],
            nodes,
        }
    }
}

fn spec(root: &Path, relative: &str, shape: Option<Vec<u64>>, encoding: &str) -> Result<AssetSpec> {
    let path = root.join(relative);
    let metadata = std::fs::metadata(&path).map_err(|_| Error::AssetMismatch)?;
    if let Some(shape) = &shape {
        let elements: u64 = shape.iter().try_fold(1u64, |acc, value| acc.checked_mul(*value).ok_or(Error::ResourceLimit))?;
        let width = match encoding {
            "i32-le" | "f32-le" => 4u64,
            "i64-le" | "f64-le" => 8u64,
            _ => 1u64,
        };
        if elements.checked_mul(width).ok_or(Error::ResourceLimit)? != metadata.len() {
            return Err(Error::AssetMismatch);
        }
    }
    Ok(AssetSpec::new(relative, metadata.len())
        .with_shape(shape.unwrap_or_default())
        .with_encoding(encoding)
        .with_role(relative))
}

fn node(name: &str, opcode: OpCode, inputs: Vec<String>, outputs: Vec<String>, scale: Option<u32>, asset: &str) -> GraphNodeSpec {
    let mut spec = GraphNodeSpec::new(name, opcode, inputs, outputs);
    if let Some(scale) = scale {
        spec = spec.with_scale(scale);
    }
    spec.with_attr("asset", asset)
}

fn push(nodes: &mut Vec<GraphNodeSpec>, spec: GraphNodeSpec) {
    nodes.push(spec);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_is_topologically_valid() {
        let config = crate::config::Gpt2Config::default();
        let graph = config.graph();
        let mut available: std::collections::BTreeSet<String> = graph.external_inputs.iter().cloned().collect();
        for node in &graph.nodes {
            for input in &node.inputs {
                assert!(available.contains(input), "node {} missing input {}", node.name, input);
            }
            for output in &node.outputs {
                available.insert(output.clone());
            }
        }
        graph.validate().unwrap();
    }

    #[test]
    fn fixed_point_profile_is_frozen() {
        assert_eq!(BASE_SHIFT, 16);
        assert_eq!(QK_SCALE_FACTOR, 23170);
        assert_eq!(EXP_OFFSET + EXP_TABLE_LEN as i64, 1 << 22);
        assert_eq!(GELU_OFFSET, 1 << 23);
        assert_eq!(RSQRT_INDEX_SHIFT, 18);
        assert_eq!(CAUSAL_MASK, -(1 << 30));
        assert_eq!(VOCAB_MASK, -(1 << 41));
    }

    #[test]
    fn asset_specs_have_declared_shapes() {
        let config = crate::config::Gpt2Config::default();
        let specs = config.asset_specs(Path::new("/data/zkie-assets")).unwrap();
        assert!(!specs.is_empty());
    }
}
