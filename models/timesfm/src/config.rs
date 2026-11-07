//! TimesFM 200M configuration, graph and authoritative asset bindings.
//!
//! Patch tokens are the unit of the sequence: exactly 16 patch tokens, each
//! covering 32 raw points. Any other patch count is rejected by configuration.

use std::path::Path;

use zkie_core::error::{Error, Result};
use zkie_models_common::assets::{AssetManifest, AssetSpec};
use zkie_models_common::gt::GtStatus;
use zkie_models_common::{GraphNodeSpec, ModelGraph};
use zkie_ops::op::OpCode;

pub const LAYERS: usize = 20;
pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = 80;
pub const HEAD_DIM_PAD: usize = 128;
pub const HIDDEN: usize = 1280;
pub const HIDDEN_PAD: usize = 2048;
pub const PATCH_TOKENS: usize = 16;
pub const RAW_POINTS_PER_PATCH: usize = 32;
pub const RAW_POINTS: usize = 512;
pub const HORIZON: usize = 128;
pub const BASE_SHIFT: u32 = 16;
pub const SILU_OFFSET: i64 = 1 << 19;
pub const SILU_TABLE_LEN: usize = 1 << 20;
pub const EXP_OFFSET: i64 = 1 << 21;
pub const EXP_TABLE_LEN: usize = 1 << 21;
pub const MASK_VALUE: i64 = -(1 << 21);
pub const OUTPUT_SCALE_SHIFT: u32 = 16;
pub const NORM_OUTPUT_SHIFT: u32 = 32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimesFmConfig {
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub head_dim_padded: usize,
    pub hidden: usize,
    pub hidden_padded: usize,
    pub patch_tokens: usize,
    pub raw_points_per_patch: usize,
    pub raw_points: usize,
}

impl Default for TimesFmConfig {
    fn default() -> Self {
        Self {
            layers: LAYERS,
            heads: HEADS,
            head_dim: HEAD_DIM,
            head_dim_padded: HEAD_DIM_PAD,
            hidden: HIDDEN,
            hidden_padded: HIDDEN_PAD,
            patch_tokens: PATCH_TOKENS,
            raw_points_per_patch: RAW_POINTS_PER_PATCH,
            raw_points: RAW_POINTS,
        }
    }
}

impl TimesFmConfig {
    /// Exactly 16 patch tokens are accepted. 512 is the raw point count, not a
    /// patch count, and is rejected here.
    pub fn validate_patch_tokens(&self, patch_tokens: usize) -> Result<()> {
        if patch_tokens != self.patch_tokens || patch_tokens == self.raw_points {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        if self.layers != LAYERS
            || self.heads * self.head_dim_padded != self.hidden_padded
            || self.patch_tokens != PATCH_TOKENS
            || self.raw_points_per_patch != RAW_POINTS_PER_PATCH
            || self.raw_points != PATCH_TOKENS * RAW_POINTS_PER_PATCH
        {
            return Err(Error::InvalidConfig);
        }
        self.validate_patch_tokens(self.patch_tokens)
    }

    pub fn asset_specs(&self, root: &Path) -> Result<Vec<AssetSpec>> {
        self.validate()?;
        let mut specs = Vec::new();
        let layer_shapes: [(&str, Vec<u64>, &str); 11] = [
            ("qkv_w", vec![HIDDEN_PAD as u64, 4096], "i32-le"),
            ("qkv_b", vec![3 * HIDDEN as u64], "i32-le"),
            ("o_proj_w", vec![HIDDEN_PAD as u64, HIDDEN_PAD as u64], "i32-le"),
            ("o_proj_b", vec![HIDDEN_PAD as u64], "i32-le"),
            ("gate_w", vec![HIDDEN_PAD as u64, HIDDEN_PAD as u64], "i32-le"),
            ("gate_b", vec![HIDDEN_PAD as u64], "i32-le"),
            ("down_w", vec![HIDDEN_PAD as u64, HIDDEN_PAD as u64], "i32-le"),
            ("down_b", vec![HIDDEN_PAD as u64], "i32-le"),
            ("lnw", vec![HIDDEN_PAD as u64], "i32-le"),
            ("mlp_w", vec![HIDDEN_PAD as u64], "i32-le"),
            ("mlp_b", vec![HIDDEN_PAD as u64], "i32-le"),
        ];
        for layer in 0..LAYERS {
            for (suffix, shape, encoding) in &layer_shapes {
                specs.push(spec(root, &format!("models/timesfm/weights/L{layer}_{suffix}_i32.bin"), Some(shape.clone()), encoding)?);
            }
        }
        for (name, shape, encoding) in [
            ("embedding_i32", vec![HIDDEN_PAD as u64], "i32-le"),
            ("cat_i32", vec![1024], "i32-le"),
            ("gather_i32", vec![32768], "i32-le"),
            ("q_scale_i32", vec![128], "i32-le"),
            ("mask_q_i32", vec![256], "i32-le"),
            ("scale_i64", vec![2], "i64-le"),
            ("pro_hid_w_i32", vec![64, HIDDEN_PAD as u64], "i32-le"),
            ("pro_hid_b_i32", vec![HIDDEN_PAD as u64], "i32-le"),
            ("pro_out_w_i32", vec![HIDDEN_PAD as u64, HIDDEN_PAD as u64], "i32-le"),
            ("pro_out_b_i32", vec![HIDDEN_PAD as u64], "i32-le"),
            ("pro_res_w_i32", vec![64, HIDDEN_PAD as u64], "i32-le"),
            ("pro_res_b_i32", vec![HIDDEN_PAD as u64], "i32-le"),
            ("head_hid_w_i32", vec![HIDDEN_PAD as u64, HIDDEN_PAD as u64], "i32-le"),
            ("head_hid_b_i32", vec![HIDDEN_PAD as u64], "i32-le"),
            ("head_out_w_i32", vec![HIDDEN_PAD as u64, HIDDEN_PAD as u64], "i32-le"),
            ("head_out_b_i32", vec![HIDDEN_PAD as u64], "i32-le"),
            ("head_res_w_i32", vec![HIDDEN_PAD as u64, HIDDEN_PAD as u64], "i32-le"),
            ("head_res_b_i32", vec![HIDDEN_PAD as u64], "i32-le"),
            ("silu_table_i32", vec![1_048_576], "i32-le"),
        ] {
            specs.push(spec(root, &format!("models/timesfm/weights/{name}.bin"), Some(shape), encoding)?);
        }
        for (name, shape) in [
            ("models/gpt2/weights/exp_table_i32.bin", vec![2_097_152u64]),
            ("models/gpt2/weights/rsqrt_table_i32.bin", vec![7_340_032u64]),
        ] {
            specs.push(spec(root, name, Some(shape), "i32-le")?.shared());
        }
        for (name, encoding) in [("input_ts_i32.bin", "i32-le"), ("case_meta.json", "utf8")] {
            specs.push(spec(root, &format!("cases/timesfm/{name}"), None, encoding)?);
        }
        for (name, shape, encoding) in [
            ("gt_output_f32.bin", vec![1, 16, 128, 10], "f32-le"),
            ("gt_int_output_i32.bin", vec![16, 2048], "i32-le"),
        ] {
            specs.push(spec(root, &format!("cases/timesfm/{name}"), Some(shape), encoding)?);
        }
        Ok(specs)
    }

    pub fn manifest(&self, root: &Path) -> Result<AssetManifest> {
        AssetManifest::load(root, &self.asset_specs(root)?)
    }

    pub fn gt_status(&self) -> (GtStatus, GtStatus) {
        (GtStatus::Available, GtStatus::Available)
    }

    pub fn graph(&self) -> ModelGraph {
        let mut nodes = Vec::new();
        nodes.push(node("patchify", OpCode::Patchify, vec!["raw_points".into()], vec!["patches".into()], None, "patch=32"));
        nodes.push(node("input_norm", OpCode::LayerNorm, vec!["patches".into(), "norm_weight".into()], vec!["normed".into()], Some(16), "normalization profile"));
        nodes.push(node("embedding", OpCode::TableLookup, vec!["patch_ids".into()], vec!["embedded".into()], None, "embedding_i32"));
        nodes.push(node("pro_hid", OpCode::Projection, vec!["normed".into()], vec!["pro_hid".into()], Some(16), "pro_hid_w/pro_hid_b"));
        nodes.push(node("pro_silu", OpCode::TableLookup, vec!["pro_hid".into(), "silu_idx".into()], vec!["pro_silu".into()], Some(16), "silu_table_i32"));
        nodes.push(node("pro_out", OpCode::Projection, vec!["pro_silu".into()], vec!["pro_out".into()], Some(16), "pro_out_w/pro_out_b"));
        nodes.push(node("pro_res", OpCode::Projection, vec!["normed".into()], vec!["pro_res".into()], Some(16), "pro_res_w/pro_res_b"));
        nodes.push(node("pro_add", OpCode::Add, vec!["pro_out".into(), "pro_res".into()], vec!["pro_residual".into()], None, "residual"));
        nodes.push(node("pro_embed_add", OpCode::Add, vec!["pro_residual".into(), "embedded".into()], vec!["x0".into()], None, "embedding add"));
        let mut current = "x0".to_owned();
        for layer in 0..LAYERS {
            push(&mut nodes, node(&format!("L{layer}.qkv"), OpCode::Projection, vec![current.clone()], vec![format!("L{layer}.qkv")], Some(16), "qkv_w/qkv_b"));
            push(&mut nodes, node(&format!("L{layer}.attn"), OpCode::CausalAttention, vec![format!("L{layer}.qkv"), "mask".into(), "exp_table".into()], vec![format!("L{layer}.attn_out")], Some(16), "exp/rsqrt tables"));
            push(&mut nodes, node(&format!("L{layer}.o"), OpCode::Projection, vec![format!("L{layer}.attn_out")], vec![format!("L{layer}.o_out")], Some(16), "o_proj_w/o_proj_b"));
            push(&mut nodes, node(&format!("L{layer}.residual_attn"), OpCode::Add, vec![current.clone(), format!("L{layer}.o_out")], vec![format!("L{layer}.x1")], None, "residual"));
            push(&mut nodes, node(&format!("L{layer}.ln"), OpCode::RmsNorm, vec![format!("L{layer}.x1")], vec![format!("L{layer}.n1")], Some(16), "lnw"));
            push(&mut nodes, node(&format!("L{layer}.gate"), OpCode::Projection, vec![format!("L{layer}.n1")], vec![format!("L{layer}.gate_out")], Some(16), "gate_w/gate_b"));
            push(&mut nodes, node(&format!("L{layer}.silu"), OpCode::TableLookup, vec![format!("L{layer}.gate_out"), "silu_idx".into()], vec![format!("L{layer}.silu_out")], Some(16), "silu_table_i32"));
            push(&mut nodes, node(&format!("L{layer}.down"), OpCode::Projection, vec![format!("L{layer}.silu_out")], vec![format!("L{layer}.down_out")], Some(16), "down_w/down_b"));
            push(&mut nodes, node(&format!("L{layer}.residual_ffn"), OpCode::Add, vec![format!("L{layer}.x1"), format!("L{layer}.down_out")], vec![format!("L{layer}.x2")], None, "residual"));
            current = format!("L{layer}.x2");
        }
        push(&mut nodes, node("head_hid", OpCode::Projection, vec![current.clone()], vec!["head_hid".into()], Some(16), "head_hid_w/head_hid_b"));
        push(&mut nodes, node("head_silu", OpCode::TableLookup, vec!["head_hid".into(), "silu_idx".into()], vec!["head_silu".into()], Some(16), "silu_table_i32"));
        push(&mut nodes, node("head_out", OpCode::Projection, vec!["head_silu".into()], vec!["head_out".into()], Some(16), "head_out_w/head_out_b"));
        push(&mut nodes, node("head_res", OpCode::Projection, vec![current], vec!["head_res".into()], Some(16), "head_res_w/head_res_b"));
        push(&mut nodes, node("head_add", OpCode::Add, vec!["head_out".into(), "head_res".into()], vec!["head_residual".into()], None, "residual"));
        push(&mut nodes, node("regression_head", OpCode::RegressionHead, vec!["head_residual".into()], vec!["predictions".into()], Some(16), "regression head"));
        push(&mut nodes, node("denormalize", OpCode::AffineScale, vec!["predictions".into()], vec!["denormalized".into()], None, "scale_i64"));
        ModelGraph {
            model_id: "timesfm-200m".into(),
            external_inputs: vec![
                "raw_points".into(),
                "norm_weight".into(),
                "patch_ids".into(),
                "silu_idx".into(),
                "mask".into(),
                "exp_table".into(),
            ],
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
        let config = crate::config::TimesFmConfig::default();
        config.graph().validate().unwrap();
    }

    #[test]
    fn fixed_point_profile_is_frozen() {
        assert_eq!(BASE_SHIFT, 16);
        assert_eq!(SILU_OFFSET, 1 << 19);
        assert_eq!(SILU_TABLE_LEN, 1 << 20);
        assert_eq!(EXP_OFFSET, 1 << 21);
        assert_eq!(MASK_VALUE, -(1 << 21));
        assert_eq!(OUTPUT_SCALE_SHIFT, 16);
    }

    #[test]
    fn rejects_512_patch_tokens() {
        let config = TimesFmConfig::default();
        config.validate_patch_tokens(16).unwrap();
        assert!(config.validate_patch_tokens(512).is_err());
        assert!(config.validate_patch_tokens(8).is_err());
    }

    #[test]
    fn asset_specs_have_declared_shapes() {
        let config = crate::config::TimesFmConfig::default();
        let specs = config.asset_specs(Path::new("/data/zkie-assets")).unwrap();
        assert!(!specs.is_empty());
    }
}
