//! Gemma3 270M configuration, graph and authoritative asset bindings.
//!
//! Global and sliding-window layers, local/global RoPE tables, RMSNorm and the
//! gated FFN are explicit graph attributes. Shared LUT paths are registered,
//! not resolved relative to the process working directory.

use std::path::Path;

use zkie_core::error::{Error, Result};
use zkie_models_common::assets::{AssetManifest, AssetSpec};
use zkie_models_common::gt::GtStatus;
use zkie_models_common::{GraphNodeSpec, ModelGraph};
use zkie_ops::op::OpCode;

pub const LAYERS: usize = 18;
pub const HEADS: usize = 4;
pub const HEAD_DIM: usize = 256;
pub const HIDDEN: usize = 640;
pub const HIDDEN_PAD: usize = 1024;
pub const FFN: usize = 2048;
pub const VOCAB: usize = 262144;
pub const MAX_SEQUENCE: usize = 512;
pub const BASE_SHIFT: u32 = 16;
pub const FULL_LAYERS: [usize; 3] = [5, 11, 17];
pub const Q_SCALE_FACTOR: i64 = 4096;
pub const Q_SCALE_SHIFT: u32 = 16;
pub const ROPE_SHIFT: u32 = 16;
pub const LOCAL_ROPE_BASE: u64 = 10_000;
pub const GLOBAL_ROPE_BASE: u64 = 1_000_000;
pub const SLIDING_WINDOW: usize = 512;
pub const RMS_NORM_OUTPUT_SHIFT: u32 = 32;
pub const EXP_OFFSET: i64 = 1 << 21;
pub const EXP_TABLE_LEN: usize = 1 << 21;
pub const GELU_OFFSET: i64 = 1 << 23;
pub const GELU_TABLE_LEN: usize = 1 << 24;
pub const CAUSAL_MASK: i64 = -(1 << 30);
pub const RSQRT_BASE_PATH: &str = "models/gpt2/weights/rsqrt_table_i32.bin";
pub const RSQRT_BASE_ENTRIES: usize = 7_340_032;
pub const RSQRT_EXTENSION_PATH: &str = "cases/gemma3/rsqrt_ext_i32.bin";
pub const RSQRT_EXTENSION_ENTRIES: usize = 14_849_263;
pub const RSQRT_EXTENSION_MAX_VARIANCE: u64 = 67_851_256;
pub const RSQRT_INDEX_SCALE: u32 = 14;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gemma3Config {
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub hidden: usize,
    pub hidden_padded: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub max_sequence: usize,
    pub full_attention_layers: Vec<usize>,
}

impl Default for Gemma3Config {
    fn default() -> Self {
        Self {
            layers: LAYERS,
            heads: HEADS,
            head_dim: HEAD_DIM,
            hidden: HIDDEN,
            hidden_padded: HIDDEN_PAD,
            ffn: FFN,
            vocab: VOCAB,
            max_sequence: MAX_SEQUENCE,
            full_attention_layers: FULL_LAYERS.to_vec(),
        }
    }
}

impl Gemma3Config {
    pub fn validate(&self) -> Result<()> {
        if self.layers != LAYERS
            || self.heads * self.head_dim != self.hidden_padded
            || self.hidden > self.hidden_padded
            || self.max_sequence != MAX_SEQUENCE
            || self.full_attention_layers.iter().any(|layer| *layer >= LAYERS)
            || self.full_attention_layers != FULL_LAYERS.to_vec()
        {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }

    pub fn is_full_attention(&self, layer: usize) -> bool {
        self.full_attention_layers.contains(&layer)
    }

    pub fn asset_specs(&self, root: &Path) -> Result<Vec<AssetSpec>> {
        self.validate()?;
        let mut specs = Vec::new();
        let layer_shapes: [(&str, Vec<u64>, &str); 13] = [
            ("q_w", vec![HIDDEN_PAD as u64, HIDDEN_PAD as u64], "i32-le"),
            ("k_w", vec![HIDDEN_PAD as u64, HEAD_DIM as u64], "i32-le"),
            ("v_w", vec![HIDDEN_PAD as u64, HEAD_DIM as u64], "i32-le"),
            ("o_w", vec![HIDDEN_PAD as u64, HIDDEN_PAD as u64], "i32-le"),
            ("gate_w", vec![HIDDEN_PAD as u64, FFN as u64], "i32-le"),
            ("up_w", vec![HIDDEN_PAD as u64, FFN as u64], "i32-le"),
            ("down_w", vec![FFN as u64, HIDDEN_PAD as u64], "i32-le"),
            ("in_norm", vec![HIDDEN_PAD as u64], "i32-le"),
            ("post_attn_norm", vec![HIDDEN_PAD as u64], "i32-le"),
            ("pre_ffn_norm", vec![HIDDEN_PAD as u64], "i32-le"),
            ("post_ffn_norm", vec![HIDDEN_PAD as u64], "i32-le"),
            ("q_norm", vec![HEAD_DIM as u64], "i32-le"),
            ("k_norm", vec![HEAD_DIM as u64], "i32-le"),
        ];
        for layer in 0..LAYERS {
            for (suffix, shape, encoding) in &layer_shapes {
                let relative = format!("models/gemma3/weights/L{layer}_{suffix}_i32.bin");
                specs.push(spec(root, &relative, Some(shape.clone()), encoding)?);
            }
        }
        for (name, shape, encoding) in [
            ("embed_tokens_i32", vec![VOCAB as u64, HIDDEN_PAD as u64], "i32-le"),
            ("lm_head_i32", vec![HIDDEN_PAD as u64, VOCAB as u64], "i32-le"),
            ("final_norm_i32", vec![HIDDEN_PAD as u64], "i32-le"),
            ("rope_cos_global_i32", vec![MAX_SEQUENCE as u64, (HEAD_DIM / 2) as u64], "i32-le"),
            ("rope_cos_local_i32", vec![MAX_SEQUENCE as u64, (HEAD_DIM / 2) as u64], "i32-le"),
            ("rope_sin_global_i32", vec![MAX_SEQUENCE as u64, (HEAD_DIM / 2) as u64], "i32-le"),
            ("rope_sin_local_i32", vec![MAX_SEQUENCE as u64, (HEAD_DIM / 2) as u64], "i32-le"),
            ("x0_i32", vec![16, HIDDEN_PAD as u64], "i32-le"),
        ] {
            specs.push(spec(root, &format!("models/gemma3/weights/{name}.bin"), Some(shape), encoding)?);
        }
        for (name, shape) in [
            ("models/gpt2/weights/exp_table_i32.bin", vec![2_097_152u64]),
            ("models/gpt2/weights/rsqrt_table_i32.bin", vec![7_340_032u64]),
            ("models/gpt2/weights/gelu_table_i32.bin", vec![16_777_216u64]),
        ] {
            specs.push(spec(root, name, Some(shape), "i32-le")?.shared());
        }
        for name in ["hf/tokenizer.json", "hf/tokenizer.model", "hf/tokenizer_config.json", "hf/config.json", "hf/special_tokens_map.json"] {
            specs.push(spec(root, &format!("models/gemma3/{name}"), None, "utf8")?);
        }
        for (name, encoding) in [("tokens_16_i32.bin", "i32-le"), ("tokens_512_i32.bin", "i32-le"), ("input_text.txt", "utf8"), ("case_meta.json", "utf8"), ("rsqrt_ext_meta.json", "utf8")] {
            specs.push(spec(root, &format!("cases/gemma3/{name}"), None, encoding)?);
        }
        for (name, shape, encoding) in [
            ("rsqrt_ext_i32.bin", vec![RSQRT_EXTENSION_ENTRIES as u64], "i32-le"),
            ("gt_logits_16_f32.bin", vec![16, 262144], "f32-le"),
            ("gt_argmax_16_i32.bin", vec![16], "i32-le"),
            ("gt_argmax_512_i32.bin", vec![512], "i32-le"),
            ("gt_int_argmax_16_i32.bin", vec![16], "i32-le"),
            ("gt_int_argmax_512_i32.bin", vec![512], "i32-le"),
        ] {
            specs.push(spec(root, &format!("cases/gemma3/{name}"), Some(shape), encoding)?);
        }
        Ok(specs)
    }

    pub fn manifest(&self, root: &Path) -> Result<AssetManifest> {
        AssetManifest::load(root, &self.asset_specs(root)?)
    }

    /// RMSNorm table profile: the shared prefix is byte-identical to the base
    /// table; the extension continues the same piecewise formula and covers the
    /// observed Gemma3 variance with margin.
    pub fn rsqrt_profile(&self) -> (&'static str, &'static str, usize, u64) {
        (
            RSQRT_BASE_PATH,
            RSQRT_EXTENSION_PATH,
            RSQRT_EXTENSION_ENTRIES,
            RSQRT_EXTENSION_MAX_VARIANCE,
        )
    }

    pub fn gt_status(&self) -> (GtStatus, GtStatus) {
        (GtStatus::Available, GtStatus::Available)
    }

    pub fn graph(&self) -> ModelGraph {
        let mut nodes = Vec::new();
        nodes.push(node("embedding", OpCode::TableLookup, vec!["prompt_tokens".into()], vec!["h0".into()], None, "embed_tokens_i32"));
        nodes.push(node("position_add", OpCode::Add, vec!["h0".into(), "positional".into()], vec!["x0".into()], Some(16), "x0_i32"));
        let mut current = "x0".to_owned();
        for layer in 0..LAYERS {
            let norm = format!("L{layer}.in_norm");
            let normed = format!("L{layer}.n0");
            push(&mut nodes, node(&norm, OpCode::RmsNorm, vec![current.clone()], vec![normed.clone()], Some(16), "in_norm"));
            push(&mut nodes, node(&format!("L{layer}.q"), OpCode::Projection, vec![normed.clone()], vec![format!("L{layer}.q_out")], Some(16), "q_w"));
            push(&mut nodes, node(&format!("L{layer}.k"), OpCode::Projection, vec![normed.clone()], vec![format!("L{layer}.k_out")], Some(16), "k_w"));
            push(&mut nodes, node(&format!("L{layer}.v"), OpCode::Projection, vec![normed], vec![format!("L{layer}.v_out")], Some(16), "v_w"));
            push(&mut nodes, node(&format!("L{layer}.q_norm"), OpCode::RmsNorm, vec![format!("L{layer}.q_out")], vec![format!("L{layer}.q_normed")], Some(16), "q_norm"));
            push(&mut nodes, node(&format!("L{layer}.k_norm"), OpCode::RmsNorm, vec![format!("L{layer}.k_out")], vec![format!("L{layer}.k_normed")], Some(16), "k_norm"));
            let rope_asset = if self.is_full_attention(layer) { "rope_cos_global/rope_sin_global" } else { "rope_cos_local/rope_sin_local" };
            push(&mut nodes, node(&format!("L{layer}.rope_q"), OpCode::RopeProjection, vec![format!("L{layer}.q_normed"), if self.is_full_attention(layer) { "rope_cos_global" } else { "rope_cos_local" }.into(), if self.is_full_attention(layer) { "rope_sin_global" } else { "rope_sin_local" }.into()], vec![format!("L{layer}.q_rope")], Some(16), rope_asset));
            push(&mut nodes, node(&format!("L{layer}.rope_k"), OpCode::RopeProjection, vec![format!("L{layer}.k_normed"), if self.is_full_attention(layer) { "rope_cos_global" } else { "rope_cos_local" }.into(), if self.is_full_attention(layer) { "rope_sin_global" } else { "rope_sin_local" }.into()], vec![format!("L{layer}.k_rope")], Some(16), rope_asset));
            let attention = format!("L{layer}.attn");
            let mut attn = node(&attention, OpCode::CausalAttention, vec![format!("L{layer}.q_rope"), format!("L{layer}.k_rope"), format!("L{layer}.v_out"), "exp_table".into(), "mask".into()], vec![format!("L{layer}.attn_out")], Some(16), "exp/rsqrt tables");
            attn = attn.with_attr("window", if self.is_full_attention(layer) { "full" } else { "sliding" });
            push(&mut nodes, attn);
            push(&mut nodes, node(&format!("L{layer}.o"), OpCode::Projection, vec![format!("L{layer}.attn_out")], vec![format!("L{layer}.o_out")], Some(16), "o_w"));
            push(&mut nodes, node(&format!("L{layer}.post_attn_norm"), OpCode::RmsNorm, vec![format!("L{layer}.o_out")], vec![format!("L{layer}.post_attn")], Some(16), "post_attn_norm"));
            push(&mut nodes, node(&format!("L{layer}.residual_attn"), OpCode::Add, vec![current.clone(), format!("L{layer}.post_attn")], vec![format!("L{layer}.x1")], None, "residual"));
            push(&mut nodes, node(&format!("L{layer}.pre_ffn_norm"), OpCode::RmsNorm, vec![format!("L{layer}.x1")], vec![format!("L{layer}.n1")], Some(16), "pre_ffn_norm"));
            push(&mut nodes, node(&format!("L{layer}.gate"), OpCode::Projection, vec![format!("L{layer}.n1")], vec![format!("L{layer}.gate_out")], Some(16), "gate_w"));
            push(&mut nodes, node(&format!("L{layer}.up"), OpCode::Projection, vec![format!("L{layer}.n1")], vec![format!("L{layer}.up_out")], Some(16), "up_w"));
            push(&mut nodes, node(&format!("L{layer}.gated_ffn"), OpCode::GatedFfn, vec![format!("L{layer}.gate_out"), format!("L{layer}.up_out")], vec![format!("L{layer}.ffn_out")], Some(16), "gelu_table_i32"));
            push(&mut nodes, node(&format!("L{layer}.down"), OpCode::Projection, vec![format!("L{layer}.ffn_out")], vec![format!("L{layer}.down_out")], Some(16), "down_w"));
            push(&mut nodes, node(&format!("L{layer}.post_ffn_norm"), OpCode::RmsNorm, vec![format!("L{layer}.down_out")], vec![format!("L{layer}.post_ffn")], Some(16), "post_ffn_norm"));
            push(&mut nodes, node(&format!("L{layer}.residual_ffn"), OpCode::Add, vec![format!("L{layer}.x1"), format!("L{layer}.post_ffn")], vec![format!("L{layer}.x2")], None, "residual"));
            current = format!("L{layer}.x2");
        }
        push(&mut nodes, node("final_norm", OpCode::RmsNorm, vec![current], vec!["final_normed".into()], Some(16), "final_norm_i32"));
        push(&mut nodes, node("lm_head", OpCode::MatMul, vec!["final_normed".into(), "lm_head_w".into()], vec!["logits".into()], Some(16), "lm_head_i32"));
        push(&mut nodes, node("argmax", OpCode::Argmax, vec!["logits".into()], vec!["next_token".into()], None, "smallest-index tie break"));
        ModelGraph {
            model_id: "gemma3-270m".into(),
            external_inputs: vec!["prompt_tokens".into(), "positional".into(), "rope_cos_global".into(), "rope_sin_global".into(), "rope_cos_local".into(), "rope_sin_local".into(), "exp_table".into(), "mask".into(), "lm_head_w".into()],
            nodes,
        }
    }
}

fn spec(root: &Path, relative: &str, shape: Option<Vec<u64>>, encoding: &str) -> Result<AssetSpec> {
    let metadata = std::fs::metadata(root.join(relative)).map_err(|_| Error::AssetMismatch)?;
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
        let config = crate::config::Gemma3Config::default();
        let graph = config.graph();
        let mut available: std::collections::BTreeSet<String> = graph.external_inputs.iter().cloned().collect();
        for node in &graph.nodes {
            for input in &node.inputs {
                assert!(available.contains(input), "node {} missing input {}", node.name, input);
            }
            for output in &node.outputs {
                assert!(available.insert(output.clone()), "duplicate output {}", output);
            }
        }
        graph.validate().unwrap();
    }

    #[test]
    fn fixed_point_profile_is_frozen() {
        assert_eq!(Q_SCALE_FACTOR, 4096);
        assert_eq!(ROPE_SHIFT, 16);
        assert_eq!(FULL_LAYERS, [5, 11, 17]);
        assert_eq!(SLIDING_WINDOW, 512);
        assert_eq!(EXP_OFFSET, 1 << 21);
        assert_eq!(GELU_TABLE_LEN, 1 << 24);
        assert_eq!(CAUSAL_MASK, -(1 << 30));
    }

    #[test]
    fn rsqrt_profile_references_extension() {
        let config = Gemma3Config::default();
        let (base, extension, entries, max_variance) = config.rsqrt_profile();
        assert!(base.ends_with("rsqrt_table_i32.bin"));
        assert!(extension.ends_with("rsqrt_ext_i32.bin"));
        assert!(entries > 0);
        assert!(max_variance >= 33_925_628);
    }

    #[test]
    fn asset_specs_have_declared_shapes() {
        let config = crate::config::Gemma3Config::default();
        let specs = config.asset_specs(Path::new("/data/zkie-assets")).unwrap();
        assert!(!specs.is_empty());
    }
}
