//! Per-operator algebraic relations and `Operation::prove/verify` material.
//!
//! The prover commits input, output, raw, quotient, helper and auxiliary
//! vectors as authenticated oracles. The verifier checks the op-specific
//! equations from the opened vectors and never invokes the operator's forward
//! routine.

use std::borrow::Cow;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::op::OpCode;
use crate::proof::binding::{
    commit_rows_with_openings, verify_opening, OracleId, OracleOpening, OracleRegistry,
};
use crate::registry::{exp_profile_value, isqrt_u128, profile_rsqrt_index, round_div_half_up, PureForward};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationLengths {
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    pub raw: Vec<usize>,
    pub quotients: Vec<usize>,
    pub helpers: Vec<usize>,
    pub aux: Vec<usize>,
}

/// Layout applied when a shared asset interval is supplied to a step input.
/// `Columns` covers row-major column slices; `Broadcast` repeats one row;
/// `BroadcastColumns` selects a column window of one row and repeats it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ExternalLayout {
    Rows,
    Broadcast { rows: u64 },
    Columns {
        source_row_width: u64,
        col_start: u64,
        col_count: u64,
    },
    BroadcastColumns {
        source_row_width: u64,
        col_start: u64,
        col_count: u64,
        rows: u64,
    },
}

impl Default for ExternalLayout {
    fn default() -> Self {
        Self::Rows
    }
}

/// Reference to a row interval of a committed model asset plus an explicit
/// layout. The interval is explicit so a step never relies on a positional or
/// sentinel convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExternalAssetRef {
    pub asset_id: u64,
    pub row_start: u64,
    pub row_count: u64,
    #[serde(default)]
    pub layout: ExternalLayout,
}

impl ExternalAssetRef {
    pub const fn rows(asset_id: u64, row_start: u64, row_count: u64) -> Self {
        Self {
            asset_id,
            row_start,
            row_count,
            layout: ExternalLayout::Rows,
        }
    }
}

/// Executor-side request to bind an input to a shared asset interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExternalSpec {
    pub asset_id: u64,
    pub row_start: u64,
    pub row_count: u64,
    pub layout: ExternalLayout,
}

#[derive(Clone, Debug)]
pub struct OpRelationPublic {
    pub opcode: OpCode,
    pub config: Vec<i64>,
    pub registry: OracleRegistry,
    /// `None` means the input is supplied by a shared commitment interval;
    /// `Some(oracle)` means a local oracle in this relation proof. There is no
    /// magic oracle id for external inputs.
    pub input_oracles: Vec<Option<OracleId>>,
    pub output_oracles: Vec<OracleId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpRelationProof {
    pub lengths: RelationLengths,
    pub inputs: Vec<Vec<OracleOpening>>,
    pub outputs: Vec<Vec<OracleOpening>>,
    pub raw: Vec<Vec<OracleOpening>>,
    pub quotients: Vec<Vec<OracleOpening>>,
    pub helpers: Vec<Vec<OracleOpening>>,
    pub aux: Vec<Vec<OracleOpening>>,
}

/// Wire form carried in `RelationReceipt::message`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationReceiptMessage {
    pub opcode: u16,
    pub config: Vec<i64>,
    pub roots: Vec<(Digest, Digest, u64)>,
    pub external_inputs: Vec<usize>,
    pub proof: OpRelationProof,
}

pub trait RelationConfig {
    fn relation_config(&self, inputs: &[Vec<i64>]) -> Result<Vec<i64>>;
}

fn clamp_config(clamp: Option<(i64, i64)>) -> Vec<i64> {
    match clamp {
        Some((low, high)) => vec![1, low, high],
        None => vec![0, 0, 0],
    }
}

impl RelationConfig for crate::layout::LayoutOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![
            if matches!(self.kind, crate::layout::LayoutKind::Reshape) { 0 } else { 1 },
            self.from.first().copied().unwrap_or(0) as i64,
            self.from.get(1).copied().unwrap_or(1) as i64,
            self.to.first().copied().unwrap_or(0) as i64,
            self.to.get(1).copied().unwrap_or(1) as i64,
        ])
    }
}

impl RelationConfig for crate::requantize::RequantizeOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![self.shift as i64])
    }
}

impl RelationConfig for crate::affine_scale::AffineScaleOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        let mut config = vec![self.multiplier as i64, self.addend as i64, self.shift as i64];
        config.extend(clamp_config(self.clamp));
        Ok(config)
    }
}

impl RelationConfig for crate::relu::ReluOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(Vec::new())
    }
}

fn matrix_config(m: usize, n: usize, k: usize, shift: u32, bias: Option<&[i64]>) -> Vec<i64> {
    let mut config = vec![m as i64, n as i64, k as i64, shift as i64];
    match bias {
        Some(values) => {
            config.push(values.len() as i64);
            config.extend(values.iter().copied());
        }
        None => config.push(0),
    }
    config
}

impl RelationConfig for crate::matmul::MatMulOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(matrix_config(self.m, self.n, self.k, self.shift, None))
    }
}

impl RelationConfig for crate::projection::ProjectionOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(matrix_config(self.m, self.n, self.k, self.shift, self.bias.as_deref()))
    }
}

impl RelationConfig for crate::regression_head::RegressionHeadOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(matrix_config(self.m, self.n, self.k, self.shift, self.bias.as_deref()))
    }
}

impl RelationConfig for crate::add::AddOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(Vec::new())
    }
}

impl RelationConfig for crate::table_lookup::TableLookupOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(Vec::new())
    }
}

impl RelationConfig for crate::softmax_rows::SoftmaxRowsOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![self.rows as i64, self.cols as i64, self.shift as i64])
    }
}

impl RelationConfig for crate::softmax::SoftmaxOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![self.shift as i64])
    }
}

fn norm_config(rows: usize, cols: usize, valid: usize, shift: u32, clamp: Option<(i64, i64)>) -> Vec<i64> {
    let mut config = vec![rows as i64, cols as i64, valid as i64, shift as i64];
    config.extend(clamp_config(clamp));
    config
}

impl RelationConfig for crate::layer_norm::LayerNormOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(norm_config(self.rows, self.cols, self.valid_cols, self.shift, self.clamp))
    }
}

impl RelationConfig for crate::rms_norm::RmsNormOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(norm_config(self.rows, self.cols, self.valid_cols, self.shift, self.clamp))
    }
}

impl RelationConfig for crate::rope_projection::RopeProjectionOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![self.shift as i64])
    }
}

impl RelationConfig for crate::topk::TopKOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![self.k as i64])
    }
}

impl RelationConfig for crate::argmax::ArgmaxOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(Vec::new())
    }
}

impl RelationConfig for crate::route_gate::RouteGateOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![self.top_k as i64, self.threshold])
    }
}

impl RelationConfig for crate::scatter::ScatterOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![self.size as i64])
    }
}

impl RelationConfig for crate::gather::GatherOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![self.cols as i64])
    }
}

impl RelationConfig for crate::batched_matmul::BatchedMatMulOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![
            self.batch as i64,
            self.m as i64,
            self.n as i64,
            self.k as i64,
            self.shift as i64,
        ])
    }
}

impl RelationConfig for crate::gather_matmul::GatherMatMulOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![self.cols as i64, self.n as i64, self.shift as i64])
    }
}

impl RelationConfig for crate::index_compose::IndexComposeOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(Vec::new())
    }
}

impl RelationConfig for crate::pad::PadOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![self.valid_len as i64, self.value])
    }
}

impl RelationConfig for crate::hadamard::HadamardOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![self.shift as i64])
    }
}

impl RelationConfig for crate::range::RangeOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![self.low, self.high])
    }
}

impl RelationConfig for crate::div_round::DivRoundOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(Vec::new())
    }
}

impl RelationConfig for crate::int_sqrt::IntSqrtOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(Vec::new())
    }
}

impl RelationConfig for crate::gated_ffn::GatedFfnOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![self.shift as i64])
    }
}

impl RelationConfig for crate::moe_down::MoeDownOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        let mut config = vec![self.width as i64, self.out_cols as i64, self.shift as i64];
        config.extend(self.shared_weights.iter().copied());
        config.extend(self.routed_weights.iter().copied());
        Ok(config)
    }
}

impl RelationConfig for crate::kv_cache::KvCacheOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![self.capacity as i64, self.width as i64, self.position as i64])
    }
}

impl RelationConfig for crate::patchify::PatchifyOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![self.patch as i64])
    }
}

impl RelationConfig for crate::profile_norm::ProfileLayerNormOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        let mut config = vec![
            self.rows as i64,
            self.cols as i64,
            self.valid_cols as i64,
            self.output_shift as i64,
        ];
        config.extend(clamp_config(self.clamp));
        Ok(config)
    }
}

impl RelationConfig for crate::profile_rms::ProfileRmsNormOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        let mut config = vec![
            self.rows as i64,
            self.cols as i64,
            self.valid_cols as i64,
            self.output_shift as i64,
        ];
        config.extend(clamp_config(self.clamp));
        Ok(config)
    }
}

impl RelationConfig for crate::profile_rope::ProfileRopeOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![self.head_dim as i64, self.shift as i64])
    }
}

impl RelationConfig for crate::index_gen::RowEmbeddingIndexOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![self.hidden as i64, self.vocab as i64])
    }
}

impl RelationConfig for crate::profile_standardize::ProfileStandardizeOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        let mut config = vec![
            self.rows as i64,
            self.cols as i64,
            self.output_cols as i64,
            self.valid_cols as i64,
            self.source_row as i64,
            self.output_shift as i64,
        ];
        config.extend(clamp_config(self.clamp));
        Ok(config)
    }
}

impl RelationConfig for crate::index_gen::TableIndexOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![self.offset, self.table_len as i64])
    }
}

impl RelationConfig for crate::index_gen::EmbeddingIndexOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![self.hidden as i64, self.vocab as i64])
    }
}

impl RelationConfig for crate::causal_attention::CausalAttentionOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![self.seq as i64, self.head_dim as i64, self.shift as i64])
    }
}

fn pad(rows: Vec<Vec<u64>>) -> Vec<Vec<u64>> {
    let width = rows.first().map(Vec::len).unwrap_or(1);
    let mut padded = rows;
    let target = padded.len().max(1).next_power_of_two();
    padded.resize(target, vec![0; width]);
    padded
}

fn commit_group(
    registry: &mut OracleRegistry,
    label: &Digest,
    values: &[i64],
) -> Result<Vec<OracleOpening>> {
    commit_group_width(registry, label, values, 1)
}

/// Commit as rows of `width` values. Matrix weights are committed row-wise so
/// openings scale with row count, not element count.
fn commit_group_width(
    registry: &mut OracleRegistry,
    label: &Digest,
    values: &[i64],
    width: usize,
) -> Result<Vec<OracleOpening>> {
    if width == 0 || values.len() % width != 0 {
        return Err(Error::ShapeMismatch);
    }
    let rows = pad(
        values
            .chunks_exact(width)
            .map(|chunk| chunk.iter().map(|value| *value as u64).collect())
            .collect(),
    );
    let (_, openings) = commit_rows_with_openings(registry, label, &rows)?;
    Ok(openings)
}

fn commit_group_wide(
    registry: &mut OracleRegistry,
    label: &Digest,
    values: &[i128],
) -> Result<Vec<OracleOpening>> {
    let mut words = Vec::with_capacity(values.len() * 2);
    for value in values {
        words.push(*value as u64);
        words.push((*value >> 64) as u64);
    }
    let (_, openings) = commit_rows_with_openings(registry, label, &[words])?;
    Ok(openings)
}

/// Per-step relation label. The trace prefix is absorbed before the step's
/// opcode/config, so the step commitments are bound to the fixed prefix.
pub fn relation_label(prefix: &Digest, opcode: OpCode, config: &[i64]) -> Digest {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&prefix.0);
    bytes.extend_from_slice(&(opcode as u16).to_le_bytes());
    for value in config {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    zkie_engine::primitives::transcript::hash_labeled(
        zkie_engine::primitives::transcript::tag_word(
            zkie_engine::primitives::transcript::STATEMENT,
            9,
        ),
        0,
        None,
        &bytes,
    )
    .unwrap_or(Digest::ZERO)
}

fn input_widths(opcode: OpCode, config: &[i64], inputs: &[Vec<i64>]) -> Vec<usize> {
    let count = inputs.len();
    let mut widths = vec![1usize; count];
    match opcode {
        OpCode::MatMul | OpCode::Projection | OpCode::RegressionHead => {
            if let Some(n) = config.get(1) {
                if *n > 0 && widths.len() > 1 {
                    widths[1] = *n as usize;
                }
            }
            if let Some(k) = config.get(2) {
                if *k > 0 && !widths.is_empty() {
                    widths[0] = *k as usize;
                }
            }
        }
        OpCode::BatchedMatMul => {
            if let Some(n) = config.get(2) {
                if *n > 0 && widths.len() > 1 {
                    widths[1] = *n as usize;
                }
            }
            if let Some(k) = config.get(3) {
                if *k > 0 && !widths.is_empty() {
                    widths[0] = *k as usize;
                }
            }
        }
        OpCode::GatherMatMul => {
            if let Some(n) = config.get(1) {
                if *n > 0 && widths.len() > 2 {
                    widths[2] = *n as usize;
                }
            }
        }
        OpCode::TableLookup => {
            if !inputs.is_empty() && !inputs[0].is_empty() {
                widths[0] = inputs[0].len();
            }
            if inputs.len() > 1 && !inputs[1].is_empty() {
                widths[1] = inputs[1].len();
            }
        }
        OpCode::Softmax => {
            if inputs.len() > 1 && !inputs[1].is_empty() {
                widths[1] = inputs[1].len();
            }
        }
        OpCode::SoftmaxRows => {
            if let Some(cols) = config.get(1) {
                if *cols > 0 {
                    widths[0] = *cols as usize;
                }
            }
            if inputs.len() > 1 && !inputs[1].is_empty() {
                widths[1] = inputs[1].len();
            }
        }
        OpCode::CausalAttention => {
            if inputs.len() > 3 && !inputs[3].is_empty() {
                widths[3] = inputs[3].len();
            }
        }
        OpCode::ProfileLayerNorm => {
            if inputs.len() > 3 && !inputs[3].is_empty() {
                widths[3] = inputs[3].len();
            }
            if inputs.len() > 2 {
                widths[1] = inputs[1].len();
                widths[2] = inputs[2].len();
            }
        }
        OpCode::ProfileRmsNorm => {
            if inputs.len() > 2 {
                widths[1] = inputs[1].len();
                widths[2] = inputs[2].len();
            }
            if inputs.len() > 3 {
                widths[3] = inputs[3].len();
            }
        }
        OpCode::ProfileRope => {
            for (index, values) in inputs.iter().enumerate() {
                widths[index] = values.len();
            }
        }
        OpCode::RowEmbeddingIndex => {
            if !inputs.is_empty() {
                widths[0] = inputs[0].len();
            }
        }
        OpCode::ProfileStandardize => {
            if inputs.len() > 1 {
                widths[1] = inputs[1].len();
            }
            if inputs.len() > 2 {
                widths[2] = inputs[2].len();
            }
        }
        OpCode::Add => {
            for (index, values) in inputs.iter().enumerate() {
                widths[index] = values.len();
            }
        }
        OpCode::Argmax | OpCode::TableIndex => {
            if !inputs.is_empty() {
                widths[0] = inputs[0].len();
            }
        }
        _ => {}
    }
    widths
}

fn output_widths(opcode: OpCode, config: &[i64], count: usize, outputs: &[Vec<i64>]) -> Vec<usize> {
    let mut widths = vec![1usize; count];
    match opcode {
        OpCode::SoftmaxRows => {
            if let Some(cols) = config.get(1) {
                if *cols > 0 && !widths.is_empty() {
                    widths[0] = *cols as usize;
                }
            }
            return widths;
        }
        OpCode::Add | OpCode::ProfileLayerNorm | OpCode::ProfileRmsNorm | OpCode::ProfileRope | OpCode::ProfileStandardize | OpCode::TableIndex | OpCode::Argmax | OpCode::EmbeddingIndex | OpCode::RowEmbeddingIndex | OpCode::TableLookup => {
            for (index, values) in outputs.iter().enumerate() {
                widths[index] = values.len();
            }
            return widths;
        }
        _ => {}
    }
    let _ = outputs;
    match opcode {
        OpCode::MatMul | OpCode::Projection | OpCode::RegressionHead | OpCode::BatchedMatMul => {
            if let Some(n) = config.get(1) {
                if *n > 0 && !widths.is_empty() {
                    widths[0] = *n as usize;
                }
            }
        }
        OpCode::GatherMatMul => {
            if let Some(n) = config.get(1) {
                if *n > 0 && !widths.is_empty() {
                    widths[0] = *n as usize;
                }
            }
        }
        _ => {}
    }
    widths
}

pub fn prove_op_relation(
    op: &dyn PureForward,
    inputs: &[Vec<i64>],
    config: &[i64],
) -> Result<(OpRelationPublic, OpRelationProof)> {
    prove_op_relation_shared(op, inputs, config, &[], &Digest::ZERO)
}

/// Prove a relation where selected inputs are supplied by a shared model
/// commitment instead of a local oracle. External inputs carry no openings.
pub fn prove_op_relation_shared(
    op: &dyn PureForward,
    inputs: &[Vec<i64>],
    config: &[i64],
    external: &[usize],
    prefix: &Digest,
) -> Result<(OpRelationPublic, OpRelationProof)> {
    let _profile = zkie_core::profile::Scope::operator(op.opcode() as u16);
    let record = op.forward_values(inputs)?;
    let opcode = op.opcode();
    let label = relation_label(prefix, opcode, config);
    let mut registry = OracleRegistry::new();
    let input_widths = input_widths(opcode, config, inputs);
    let mut input_openings = Vec::new();
    let mut input_oracles = Vec::new();
    for (index, values) in inputs.iter().enumerate() {
        if external.contains(&index) {
            input_oracles.push(None);
            input_openings.push(Vec::new());
            continue;
        }
        let openings = commit_group_width(
            &mut registry,
            &label,
            values,
            *input_widths.get(index).unwrap_or(&1),
        )?;
        input_oracles.push(Some(openings[0].oracle));
        input_openings.push(openings);
    }
    let output_widths = output_widths(opcode, config, record.outputs.len(), &record.outputs);
    let mut output_openings = Vec::new();
    let mut output_oracles = Vec::new();
    for (index, values) in record.outputs.iter().enumerate() {
        let openings = commit_group_width(
            &mut registry,
            &label,
            values,
            *output_widths.get(index).unwrap_or(&1),
        )?;
        output_oracles.push(openings[0].oracle);
        output_openings.push(openings);
    }
    let mut raw = Vec::new();
    for values in &record.raw_accumulators {
        raw.push(commit_group_wide(&mut registry, &label, values)?);
    }
    let mut quotients = Vec::new();
    for values in &record.quotients {
        quotients.push(commit_group(&mut registry, &label, values)?);
    }
    let mut helpers = Vec::new();
    for values in &record.range_helpers {
        helpers.push(commit_group(&mut registry, &label, values)?);
    }
    let mut aux = Vec::new();
    for values in &record.auxiliaries {
        aux.push(commit_group(&mut registry, &label, values)?);
    }
    let lengths = RelationLengths {
        inputs: inputs.iter().map(Vec::len).collect(),
        outputs: record.outputs.iter().map(Vec::len).collect(),
        raw: record.raw_accumulators.iter().map(Vec::len).collect(),
        quotients: record.quotients.iter().map(Vec::len).collect(),
        helpers: record.range_helpers.iter().map(Vec::len).collect(),
        aux: record.auxiliaries.iter().map(Vec::len).collect(),
    };
    Ok((
        OpRelationPublic {
            opcode,
            config: config.to_vec(),
            registry,
            input_oracles,
            output_oracles,
        },
        OpRelationProof {
            lengths,
            inputs: input_openings,
            outputs: output_openings,
            raw,
            quotients,
            helpers,
            aux,
        },
    ))
}

fn verify_group(registry: &OracleRegistry, openings: &[OracleOpening]) -> Result<()> {
    let oracle = openings.first().map(|opening| opening.oracle).ok_or(Error::ShapeMismatch)?;
    let root = registry.get(oracle).ok_or(Error::ProverStateMissing)?;
    for opening in openings {
        if opening.oracle != oracle {
            return Err(Error::StatementMismatch);
        }
        verify_opening(registry, opening, root.count)?;
    }
    Ok(())
}

fn decode(openings: &[OracleOpening], len: usize) -> Vec<i64> {
    let mut values = Vec::with_capacity(len);
    for opening in openings {
        values.extend(opening.values.iter().map(|value| *value as i64));
        if values.len() >= len {
            break;
        }
    }
    values.truncate(len);
    values
}

fn decode_wide(openings: &[OracleOpening], len: usize) -> Vec<i128> {
    let mut values = Vec::with_capacity(len);
    for opening in openings {
        for pair in opening.values.chunks_exact(2) {
            values.push(((pair[1] as i128) << 64) | pair[0] as i128);
        }
        if values.len() >= len {
            break;
        }
    }
    values.truncate(len);
    values
}

fn rq(value: i128, shift: u32) -> Result<i128> {
    if shift == 0 {
        return Ok(value);
    }
    if shift >= 128 {
        return Err(Error::BoundViolation);
    }
    let half = 1i128 << (shift - 1);
    let numerator = value.checked_add(half).ok_or(Error::BoundViolation)?;
    Ok(numerator >> shift)
}

fn clamp_value(value: i64, config: &[i64], base: usize) -> Result<i64> {
    if config.get(base).copied().unwrap_or(0) == 0 {
        return Ok(value);
    }
    let low = *config.get(base + 1).ok_or(Error::InvalidConfig)?;
    let high = *config.get(base + 2).ok_or(Error::InvalidConfig)?;
    if low > high {
        return Err(Error::InvalidConfig);
    }
    Ok(value.clamp(low, high))
}

fn isqrt(value: u128) -> u128 {
    isqrt_u128(value)
}

pub fn verify_op_relation(public: &OpRelationPublic, proof: &OpRelationProof) -> Result<()> {
    verify_op_relation_with_config(public, proof, &public.config)
}

pub fn verify_op_relation_with_config(
    public: &OpRelationPublic,
    proof: &OpRelationProof,
    expected_config: &[i64],
) -> Result<()> {
    verify_op_relation_with_external(public, proof, expected_config, &[])
}

/// Verify a relation where selected inputs must be supplied from an already
/// authenticated shared model opening.
pub fn verify_op_relation_with_external(
    public: &OpRelationPublic,
    proof: &OpRelationProof,
    expected_config: &[i64],
    external_values: &[Option<Vec<i64>>],
) -> Result<()> {
    let _profile = zkie_core::profile::Scope::operator(public.opcode as u16);
    if public.config != expected_config {
        return Err(Error::StatementMismatch);
    }
    if public.input_oracles.len() != proof.inputs.len()
        || public.output_oracles.len() != proof.outputs.len()
        || public.input_oracles.len() != proof.lengths.inputs.len()
        || public.output_oracles.len() != proof.lengths.outputs.len()
    {
        return Err(Error::ShapeMismatch);
    }
    for (index, openings) in proof.inputs.iter().enumerate() {
        let supplied = external_values.get(index).and_then(Option::as_ref);
        match (public.input_oracles[index], supplied) {
            (None, Some(values)) => {
                if !openings.is_empty() || values.len() != proof.lengths.inputs[index] {
                    return Err(Error::StatementMismatch);
                }
            }
            (None, None) => return Err(Error::StatementMismatch),
            (Some(oracle), None) => {
                if openings.first().map(|opening| opening.oracle) != Some(oracle) {
                    return Err(Error::StatementMismatch);
                }
                verify_group(&public.registry, openings)?;
            }
            (Some(_), Some(_)) => return Err(Error::StatementMismatch),
        }
    }
    for (index, openings) in proof.outputs.iter().enumerate() {
        if openings.first().map(|opening| opening.oracle) != Some(public.output_oracles[index]) {
            return Err(Error::StatementMismatch);
        }
        verify_group(&public.registry, openings)?;
    }
    for openings in &proof.raw {
        verify_group(&public.registry, openings)?;
    }
    for openings in &proof.quotients {
        verify_group(&public.registry, openings)?;
    }
    for openings in &proof.helpers {
        verify_group(&public.registry, openings)?;
    }
    for openings in &proof.aux {
        verify_group(&public.registry, openings)?;
    }
    let inputs: Vec<Cow<'_, [i64]>> = proof
        .inputs
        .iter()
        .zip(&proof.lengths.inputs)
        .enumerate()
        .map(|(index, (openings, len))| match public.input_oracles.get(index) {
            Some(None) => external_values
                .get(index)
                .and_then(Option::as_ref)
                .map(|values| Cow::Borrowed(values.as_slice()))
                .ok_or(Error::StatementMismatch),
            _ => Ok(Cow::Owned(decode(openings, *len))),
        })
        .collect::<Result<Vec<_>>>()?;
    let outputs: Vec<Vec<i64>> = proof
        .outputs
        .iter()
        .zip(&proof.lengths.outputs)
        .map(|(openings, len)| decode(openings, *len))
        .collect();
    let raw: Vec<Vec<i128>> = proof
        .raw
        .iter()
        .zip(&proof.lengths.raw)
        .map(|(openings, len)| decode_wide(openings, *len))
        .collect();
    let quotients: Vec<Vec<i64>> = proof
        .quotients
        .iter()
        .zip(&proof.lengths.quotients)
        .map(|(openings, len)| decode(openings, *len))
        .collect();
    let helpers: Vec<Vec<i64>> = proof
        .helpers
        .iter()
        .zip(&proof.lengths.helpers)
        .map(|(openings, len)| decode(openings, *len))
        .collect();
    let aux: Vec<Vec<i64>> = proof
        .aux
        .iter()
        .zip(&proof.lengths.aux)
        .map(|(openings, len)| decode(openings, *len))
        .collect();
    let config = &public.config;
    match public.opcode {
        OpCode::Layout => {
            let kind = config.first().copied().ok_or(Error::InvalidConfig)?;
            let from_x = *config.get(1).ok_or(Error::InvalidConfig)? as usize;
            let from_y = *config.get(2).ok_or(Error::InvalidConfig)? as usize;
            let to_x = *config.get(3).ok_or(Error::InvalidConfig)? as usize;
            let to_y = *config.get(4).ok_or(Error::InvalidConfig)? as usize;
            if from_x * from_y != to_x * to_y || inputs.len() != 1 || outputs[0].len() != inputs[0].len() {
                return Err(Error::ShapeMismatch);
            }
            for row in 0..from_x {
                for column in 0..from_y {
                    let source = row * from_y + column;
                    let target = if kind == 0 { source } else { column * from_x + row };
                    if inputs[0][source] != outputs[0][target] {
                        return Err(Error::InvalidProof);
                    }
                }
            }
        }
        OpCode::Requantize => {
            let shift = *config.first().ok_or(Error::InvalidConfig)? as u32;
            if inputs.len() != 1 || outputs[0].len() != inputs[0].len() {
                return Err(Error::ShapeMismatch);
            }
            for (input, output) in inputs[0].iter().zip(&outputs[0]) {
                if rq(*input as i128, shift)? != *output as i128 {
                    return Err(Error::InvalidProof);
                }
            }
        }
        OpCode::AffineScale => {
            let multiplier = *config.first().ok_or(Error::InvalidConfig)? as i128;
            let addend = *config.get(1).ok_or(Error::InvalidConfig)? as i128;
            let shift = *config.get(2).ok_or(Error::InvalidConfig)? as u32;
            if inputs.len() != 1 || outputs[0].len() != inputs[0].len() || raw.is_empty() {
                return Err(Error::ShapeMismatch);
            }
            for index in 0..inputs[0].len() {
                let expected = multiplier * inputs[0][index] as i128 + addend;
                if raw[0].get(index) != Some(&expected) {
                    return Err(Error::InvalidProof);
                }
                let quantized = i64::try_from(rq(expected, shift)?).map_err(|_| Error::BoundViolation)?;
                if outputs[0][index] != clamp_value(quantized, config, 3)? {
                    return Err(Error::InvalidProof);
                }
            }
        }
        OpCode::Relu => {
            if inputs.len() != 1 || outputs[0].len() != inputs[0].len() {
                return Err(Error::ShapeMismatch);
            }
            for (input, output) in inputs[0].iter().zip(&outputs[0]) {
                if *output != (*input).max(0) {
                    return Err(Error::InvalidProof);
                }
            }
        }
        OpCode::MatMul | OpCode::Projection | OpCode::RegressionHead => {
            let m = *config.first().ok_or(Error::InvalidConfig)? as usize;
            let n = *config.get(1).ok_or(Error::InvalidConfig)? as usize;
            let k = *config.get(2).ok_or(Error::InvalidConfig)? as usize;
            let shift = *config.get(3).ok_or(Error::InvalidConfig)? as u32;
            let bias_len = *config.get(4).ok_or(Error::InvalidConfig)? as usize;
            if config.len() != 5 + bias_len
                || inputs.len() != 2
                || inputs[0].len() != m * k
                || inputs[1].len() != k * n
                || outputs[0].len() != m * n
                || raw[0].len() != m * n
            {
                return Err(Error::ShapeMismatch);
            }
            let bias = &config[5..5 + bias_len];
            let computed = crate::registry::matmul_raw(&inputs[0], &inputs[1], m, n, k)?;
            (0..m * n).into_par_iter().try_for_each(|slot| -> Result<()> {
                let expected = computed[slot]
                    .checked_add(if bias.is_empty() {
                        0i128
                    } else {
                        bias[slot] as i128
                    })
                    .ok_or(Error::BoundViolation)?;
                if raw[0][slot] != expected
                    || rq(expected, shift)? != outputs[0][slot] as i128
                {
                    return Err(Error::InvalidProof);
                }
                Ok(())
            })?;
        }
        OpCode::Add => {
            if inputs.len() != 2 || outputs[0].len() != inputs[0].len() || inputs[0].len() != inputs[1].len() {
                return Err(Error::ShapeMismatch);
            }
            for index in 0..inputs[0].len() {
                if outputs[0][index] as i128
                    != inputs[0][index] as i128 + inputs[1][index] as i128
                {
                    return Err(Error::InvalidProof);
                }
            }
        }
        OpCode::TableLookup => {
            if inputs.len() != 2 || outputs[0].len() != inputs[1].len() {
                return Err(Error::ShapeMismatch);
            }
            for (index, output) in inputs[1].iter().zip(&outputs[0]) {
                if *index < 0 || *index as usize >= inputs[0].len() || inputs[0][*index as usize] != *output {
                    return Err(Error::InvalidProof);
                }
            }
        }
        OpCode::Softmax => {
            let shift = *config.first().ok_or(Error::InvalidConfig)? as u32;
            if inputs.len() != 2 || inputs[0].is_empty() || outputs[0].len() != inputs[0].len() {
                return Err(Error::ShapeMismatch);
            }
            let length = inputs[0].len();
            let max = *inputs[0].iter().max().ok_or(Error::InvalidProof)?;
            if aux.len() != 3 || aux[0] != vec![max] || aux[1].len() != length || aux[2].len() != 1 {
                return Err(Error::InvalidProof);
            }
            let mut denominator = 0i128;
            for index in 0..length {
                let distance = max.checked_sub(inputs[0][index]).ok_or(Error::BoundViolation)?;
                let entry = exp_profile_value(&inputs[1], distance)?;
                if entry < 0 || aux[1][index] != entry {
                    return Err(Error::InvalidProof);
                }
                denominator += entry as i128;
            }
            if denominator == 0 || aux[2][0] as i128 != denominator {
                return Err(Error::InvalidProof);
            }
            for index in 0..length {
                let numerator = (aux[1][index] as i128)
                    .checked_mul(1i128 << shift)
                    .ok_or(Error::BoundViolation)?;
                if raw[0][index] != numerator
                    || outputs[0][index] as i128 != crate::registry::round_div_half_up(numerator, denominator)?
                {
                    return Err(Error::InvalidProof);
                }
            }
        }
        OpCode::SoftmaxRows => {
            let rows = *config.first().ok_or(Error::InvalidConfig)? as usize;
            let cols = *config.get(1).ok_or(Error::InvalidConfig)? as usize;
            let shift = *config.get(2).ok_or(Error::InvalidConfig)? as u32;
            let count = rows.checked_mul(cols).ok_or(Error::ResourceLimit)?;
            if inputs.len() != 2
                || rows == 0
                || cols == 0
                || inputs[0].len() != count
                || inputs[1].is_empty()
                || outputs[0].len() != count
                || raw[0].len() != count
                || aux.len() != 2
                || aux[0].len() != rows
                || aux[1].len() != rows
            {
                return Err(Error::ShapeMismatch);
            }
            let scale = 1i128.checked_shl(shift).ok_or(Error::BoundViolation)?;
            for row in 0..rows {
                let start = row * cols;
                let max = *inputs[0][start..start + cols]
                    .iter()
                    .max()
                    .ok_or(Error::InvalidProof)?;
                if aux[0][row] != max {
                    return Err(Error::InvalidProof);
                }
                let mut entries = Vec::with_capacity(cols);
                let mut denominator = 0i128;
                for index in 0..cols {
                    let distance = max
                        .checked_sub(inputs[0][start + index])
                        .ok_or(Error::BoundViolation)?;
                    let entry = exp_profile_value(&inputs[1], distance)?;
                    if entry < 0 {
                        return Err(Error::InvalidProof);
                    }
                    denominator = denominator
                        .checked_add(entry as i128)
                        .ok_or(Error::BoundViolation)?;
                    entries.push(entry);
                }
                if denominator == 0 || aux[1][row] as i128 != denominator {
                    return Err(Error::InvalidProof);
                }
                for (index, entry) in entries.iter().enumerate() {
                    let numerator = (*entry as i128)
                        .checked_mul(scale)
                        .ok_or(Error::BoundViolation)?;
                    let slot = start + index;
                    if raw[0][slot] != numerator
                        || outputs[0][slot] as i128
                            != crate::registry::round_div_half_up(numerator, denominator)?
                    {
                        return Err(Error::InvalidProof);
                    }
                }
            }
        }
        OpCode::LayerNorm | OpCode::RmsNorm => {
            let rows = *config.first().ok_or(Error::InvalidConfig)? as usize;
            let cols = *config.get(1).ok_or(Error::InvalidConfig)? as usize;
            let valid = *config.get(2).ok_or(Error::InvalidConfig)? as usize;
            let shift = *config.get(3).ok_or(Error::InvalidConfig)? as u32;
            if valid == 0 || valid > cols || inputs.len() != 2
                || inputs[0].len() != rows * cols
                || inputs[1].len() != rows * cols
                || outputs[0].len() != rows * cols
                || raw[0].len() != rows * cols
            {
                return Err(Error::ShapeMismatch);
            }
            let layer_norm = public.opcode == OpCode::LayerNorm;
            if layer_norm {
                if aux.len() != 3 || aux[0].len() != rows || aux[1].len() != rows || aux[2].len() != rows {
                    return Err(Error::InvalidProof);
                }
            } else if aux.len() != 2 || aux[0].len() != rows || aux[1].len() != rows {
                return Err(Error::InvalidProof);
            }
            let scale_squared = 1u128
                .checked_shl(2 * shift)
                .ok_or(Error::BoundViolation)?;
            for row in 0..rows {
                let start = row * cols;
                let mut mean = 0i128;
                if layer_norm {
                    for index in 0..valid {
                        mean += inputs[0][start + index] as i128;
                    }
                    mean = mean.div_euclid(valid as i128);
                } else {
                    for index in 0..valid {
                        let value = inputs[0][start + index] as i128;
                        mean += value * value;
                    }
                    mean = mean.div_euclid(valid as i128);
                }
                if aux[0][row] as i128 != mean {
                    return Err(Error::InvalidProof);
                }
                let mut variance = 0i128;
                if layer_norm {
                    for index in 0..valid {
                        let centered = inputs[0][start + index] as i128 - mean;
                        variance += centered * centered;
                    }
                    variance = variance.div_euclid(valid as i128).max(1);
                    if aux[1][row] as i128 != variance {
                        return Err(Error::InvalidProof);
                    }
                }
                let denom = u128::try_from(if layer_norm { variance } else { mean.max(1) })
                    .map_err(|_| Error::BoundViolation)?;
                let rstd = isqrt(scale_squared / denom);
                let rstd_index = if layer_norm { 2 } else { 1 };
                if aux[rstd_index][row] as u128 != rstd {
                    return Err(Error::InvalidProof);
                }
                for index in 0..cols {
                    let image = if index < valid {
                        let base = if layer_norm {
                            inputs[0][start + index] as i128 - mean
                        } else {
                            inputs[0][start + index] as i128
                        };
                        base.checked_mul(rstd as i128)
                            .and_then(|value| value.checked_mul(inputs[1][start + index] as i128))
                            .ok_or(Error::BoundViolation)?
                    } else {
                        0
                    };
                    if raw[0][start + index] != image {
                        return Err(Error::InvalidProof);
                    }
                    let quantized = i64::try_from(rq(image, shift)?).map_err(|_| Error::BoundViolation)?;
                    if outputs[0][start + index] != clamp_value(quantized, config, 4)? {
                        return Err(Error::InvalidProof);
                    }
                }
            }
        }
        OpCode::RopeProjection => {
            let shift = *config.first().ok_or(Error::InvalidConfig)? as u32;
            if inputs.len() != 3 || inputs[0].len() % 2 != 0 {
                return Err(Error::ShapeMismatch);
            }
            let pairs = inputs[0].len() / 2;
            if inputs[1].len() != pairs || inputs[2].len() != pairs || outputs[0].len() != inputs[0].len() {
                return Err(Error::ShapeMismatch);
            }
            for pair in 0..pairs {
                let even = inputs[0][2 * pair] as i128;
                let odd = inputs[0][2 * pair + 1] as i128;
                let c = inputs[1][pair] as i128;
                let s = inputs[2][pair] as i128;
                let rotated_even = even.checked_mul(c).and_then(|v| v.checked_sub(odd.checked_mul(s)?)).ok_or(Error::BoundViolation)?;
                let rotated_odd = even.checked_mul(s).and_then(|v| v.checked_add(odd.checked_mul(c)?)).ok_or(Error::BoundViolation)?;
                if raw[0][2 * pair] != rotated_even
                    || raw[0][2 * pair + 1] != rotated_odd
                    || rq(rotated_even, shift)? != outputs[0][2 * pair] as i128
                    || rq(rotated_odd, shift)? != outputs[0][2 * pair + 1] as i128
                {
                    return Err(Error::InvalidProof);
                }
            }
        }
        OpCode::TopK => {
            let k = *config.first().ok_or(Error::InvalidConfig)? as usize;
            if inputs.len() != 1 || k == 0 || k > inputs[0].len() || outputs[0].len() != 2 * k {
                return Err(Error::ShapeMismatch);
            }
            let mut order: Vec<usize> = (0..inputs[0].len()).collect();
            order.sort_by(|left, right| inputs[0][*right].cmp(&inputs[0][*left]).then_with(|| left.cmp(right)));
            for slot in 0..k {
                let index = order[slot];
                if outputs[0][slot] != index as i64 || outputs[0][k + slot] != inputs[0][index] {
                    return Err(Error::InvalidProof);
                }
            }
        }
        OpCode::Argmax => {
            if inputs.len() != 1 || outputs[0].len() != 1 || inputs[0].is_empty() {
                return Err(Error::ShapeMismatch);
            }
            let best = outputs[0][0];
            if best < 0 || best as usize >= inputs[0].len() {
                return Err(Error::InvalidIndex);
            }
            let best = best as usize;
            for index in 0..inputs[0].len() {
                if index < best {
                    if inputs[0][index] >= inputs[0][best] {
                        return Err(Error::InvalidProof);
                    }
                } else if index > best && inputs[0][index] > inputs[0][best] {
                    return Err(Error::InvalidProof);
                }
            }
        }
        OpCode::RouteGate => {
            let top_k = *config.first().ok_or(Error::InvalidConfig)? as usize;
            let threshold = *config.get(1).ok_or(Error::InvalidConfig)?;
            if inputs.len() != 1 || top_k == 0 || top_k > inputs[0].len() || outputs[0].len() != 1 + 2 * top_k {
                return Err(Error::ShapeMismatch);
            }
            let mut order: Vec<usize> = (0..inputs[0].len()).collect();
            order.sort_by(|left, right| inputs[0][*right].cmp(&inputs[0][*left]).then_with(|| left.cmp(right)));
            let selected: Vec<usize> = order
                .into_iter()
                .filter(|index| inputs[0][*index] >= threshold)
                .take(top_k)
                .collect();
            if outputs[0][0] != selected.len() as i64 {
                return Err(Error::InvalidProof);
            }
            for slot in 0..top_k {
                let expected_index = selected.get(slot).copied().map(|index| index as i64).unwrap_or(-1);
                let expected_value = selected.get(slot).map(|index| inputs[0][*index]).unwrap_or(0);
                if outputs[0][1 + slot] != expected_index || outputs[0][1 + top_k + slot] != expected_value {
                    return Err(Error::InvalidProof);
                }
            }
        }
        OpCode::Scatter => {
            let size = *config.first().ok_or(Error::InvalidConfig)? as usize;
            if inputs.len() != 2 || inputs[0].len() != inputs[1].len() || outputs[0].len() != size {
                return Err(Error::ShapeMismatch);
            }
            let mut expected = vec![0i64; size];
            let mut seen = vec![false; size];
            for (index, value) in inputs[0].iter().zip(inputs[1].iter()) {
                if *index < 0 || *index as usize >= size || seen[*index as usize] {
                    return Err(Error::InvalidIndex);
                }
                seen[*index as usize] = true;
                expected[*index as usize] = *value;
            }
            if expected != outputs[0] {
                return Err(Error::InvalidProof);
            }
        }
        OpCode::Gather => {
            let cols = *config.first().ok_or(Error::InvalidConfig)? as usize;
            if inputs.len() != 2 || cols == 0 || inputs[0].len() % cols != 0 || outputs[0].len() != inputs[1].len() * cols {
                return Err(Error::ShapeMismatch);
            }
            let rows = inputs[0].len() / cols;
            for (slot, index) in inputs[1].iter().enumerate() {
                if *index < 0 || *index as usize >= rows {
                    return Err(Error::InvalidIndex);
                }
                let start = *index as usize * cols;
                for column in 0..cols {
                    if outputs[0][slot * cols + column] != inputs[0][start + column] {
                        return Err(Error::InvalidProof);
                    }
                }
            }
        }
        OpCode::BatchedMatMul => {
            let batch = *config.first().ok_or(Error::InvalidConfig)? as usize;
            let m = *config.get(1).ok_or(Error::InvalidConfig)? as usize;
            let n = *config.get(2).ok_or(Error::InvalidConfig)? as usize;
            let k = *config.get(3).ok_or(Error::InvalidConfig)? as usize;
            let shift = *config.get(4).ok_or(Error::InvalidConfig)? as u32;
            if inputs.len() != 2
                || inputs[0].len() != batch * m * k
                || inputs[1].len() != batch * k * n
                || outputs[0].len() != batch * m * n
                || raw[0].len() != batch * m * n
            {
                return Err(Error::ShapeMismatch);
            }
            for batch_index in 0..batch {
                for row in 0..m {
                    for column in 0..n {
                        let mut expected = 0i128;
                        for inner in 0..k {
                            expected += inputs[0][batch_index * m * k + row * k + inner] as i128
                                * inputs[1][batch_index * k * n + inner * n + column] as i128;
                        }
                        let slot = batch_index * m * n + row * n + column;
                        if raw[0][slot] != expected || rq(expected, shift)? != outputs[0][slot] as i128 {
                            return Err(Error::InvalidProof);
                        }
                    }
                }
            }
        }
        OpCode::GatherMatMul => {
            let cols = *config.first().ok_or(Error::InvalidConfig)? as usize;
            let n = *config.get(1).ok_or(Error::InvalidConfig)? as usize;
            let shift = *config.get(2).ok_or(Error::InvalidConfig)? as u32;
            if inputs.len() != 3 || cols == 0 || inputs[0].len() % cols != 0 || inputs[2].len() != cols * n {
                return Err(Error::ShapeMismatch);
            }
            let rows = inputs[0].len() / cols;
            if outputs[0].len() != inputs[1].len() * n || raw[0].len() != inputs[1].len() * n {
                return Err(Error::ShapeMismatch);
            }
            for (slot, index) in inputs[1].iter().enumerate() {
                if *index < 0 || *index as usize >= rows {
                    return Err(Error::InvalidIndex);
                }
                for column in 0..n {
                    let mut expected = 0i128;
                    for inner in 0..cols {
                        expected += inputs[0][*index as usize * cols + inner] as i128
                            * inputs[2][inner * n + column] as i128;
                    }
                    let position = slot * n + column;
                    if raw[0][position] != expected || rq(expected, shift)? != outputs[0][position] as i128 {
                        return Err(Error::InvalidProof);
                    }
                }
            }
        }
        OpCode::IndexCompose => {
            if inputs.len() != 2 || outputs[0].len() != inputs[0].len() {
                return Err(Error::ShapeMismatch);
            }
            for (slot, index) in inputs[0].iter().enumerate() {
                if *index < 0 || *index as usize >= inputs[1].len() || outputs[0][slot] != inputs[1][*index as usize] {
                    return Err(Error::InvalidProof);
                }
            }
        }
        OpCode::Pad => {
            let valid = *config.first().ok_or(Error::InvalidConfig)? as usize;
            let value = *config.get(1).ok_or(Error::InvalidConfig)?;
            if inputs.len() != 1 || valid > inputs[0].len() || outputs[0].len() != inputs[0].len() {
                return Err(Error::ShapeMismatch);
            }
            for index in 0..inputs[0].len() {
                let expected = if index < valid { inputs[0][index] } else { value };
                if outputs[0][index] != expected {
                    return Err(Error::InvalidProof);
                }
            }
        }
        OpCode::Hadamard | OpCode::GatedFfn => {
            let shift = *config.first().ok_or(Error::InvalidConfig)? as u32;
            if inputs.len() != 2 || inputs[0].len() != inputs[1].len() || outputs[0].len() != inputs[0].len() || raw[0].len() != inputs[0].len() {
                return Err(Error::ShapeMismatch);
            }
            for index in 0..inputs[0].len() {
                let expected = inputs[0][index] as i128 * inputs[1][index] as i128;
                if raw[0][index] != expected || rq(expected, shift)? != outputs[0][index] as i128 {
                    return Err(Error::InvalidProof);
                }
            }
        }
        OpCode::Range => {
            let low = *config.first().ok_or(Error::InvalidConfig)?;
            let high = *config.get(1).ok_or(Error::InvalidConfig)?;
            if low > high || inputs.len() != 1 || outputs[0].len() != inputs[0].len() {
                return Err(Error::InvalidConfig);
            }
            let mut outside = Vec::new();
            for (index, value) in inputs[0].iter().enumerate() {
                let inside = *value >= low && *value <= high;
                if outputs[0][index] != i64::from(inside) {
                    return Err(Error::InvalidProof);
                }
                if !inside {
                    outside.push(index as i64);
                }
            }
            if helpers.first() != Some(&outside) {
                return Err(Error::InvalidProof);
            }
        }
        OpCode::DivRound => {
            if inputs.len() != 2 || inputs[0].len() != inputs[1].len() || outputs[0].len() != 2 * inputs[0].len() || quotients.len() != 1 {
                return Err(Error::ShapeMismatch);
            }
            for index in 0..inputs[0].len() {
                let divisor = inputs[1][index];
                if divisor <= 0 {
                    return Err(Error::InvalidConfig);
                }
                let quotient = quotients[0][index];
                let remainder = outputs[0][inputs[0].len() + index];
                if quotient as i128 * divisor as i128 + remainder as i128 != inputs[0][index] as i128
                    || remainder < 0
                    || remainder >= divisor
                    || outputs[0][index] != quotient
                {
                    return Err(Error::InvalidProof);
                }
            }
        }
        OpCode::IntSqrt => {
            if inputs.len() != 1 || outputs[0].len() != inputs[0].len() {
                return Err(Error::ShapeMismatch);
            }
            for (input, output) in inputs[0].iter().zip(&outputs[0]) {
                if *input < 0 || *output < 0 {
                    return Err(Error::InvalidProof);
                }
                let root = *output as u128;
                let next = root.checked_add(1).ok_or(Error::BoundViolation)?;
                if root * root > *input as u128 || next * next <= *input as u128 {
                    return Err(Error::InvalidProof);
                }
            }
        }
        OpCode::MoeDown => {
            let width = *config.first().ok_or(Error::InvalidConfig)? as usize;
            let out_cols = *config.get(1).ok_or(Error::InvalidConfig)? as usize;
            let shift = *config.get(2).ok_or(Error::InvalidConfig)? as u32;
            let expected_len = 3 + 2 * width * out_cols;
            if width == 0 || out_cols == 0 || config.len() != expected_len
                || inputs.len() != 2
                || inputs[0].len() != width
                || inputs[1].len() != width
                || outputs[0].len() != out_cols
                || raw.len() != 3
                || raw[0].len() != out_cols
                || raw[1].len() != out_cols
                || raw[2].len() != out_cols
            {
                return Err(Error::ShapeMismatch);
            }
            let shared = &config[3..3 + width * out_cols];
            let routed = &config[3 + width * out_cols..];
            for column in 0..out_cols {
                let mut shared_raw = 0i128;
                let mut routed_raw = 0i128;
                for index in 0..width {
                    shared_raw += inputs[0][index] as i128 * shared[index * out_cols + column] as i128;
                    routed_raw += inputs[1][index] as i128 * routed[index * out_cols + column] as i128;
                }
                let total = shared_raw + routed_raw;
                if raw[0][column] != shared_raw
                    || raw[1][column] != routed_raw
                    || raw[2][column] != total
                    || rq(total, shift)? != outputs[0][column] as i128
                {
                    return Err(Error::InvalidProof);
                }
            }
        }
        OpCode::KvCache => {
            let capacity = *config.first().ok_or(Error::InvalidConfig)? as usize;
            let width = *config.get(1).ok_or(Error::InvalidConfig)? as usize;
            let position = *config.get(2).ok_or(Error::InvalidConfig)? as usize;
            if position >= capacity || width == 0 || inputs.len() != 4 || outputs.len() != 2 {
                return Err(Error::ShapeMismatch);
            }
            let total = capacity * width;
            if inputs[0].len() != total
                || inputs[1].len() != total
                || inputs[2].len() != width
                || inputs[3].len() != width
                || outputs[0].len() != total
                || outputs[1].len() != total
            {
                return Err(Error::ShapeMismatch);
            }
            for index in 0..total {
                let slot = index / width;
                let expected_key = if slot == position { inputs[2][index % width] } else { inputs[0][index] };
                let expected_value = if slot == position { inputs[3][index % width] } else { inputs[1][index] };
                if outputs[0][index] != expected_key || outputs[1][index] != expected_value {
                    return Err(Error::InvalidProof);
                }
            }
        }
        OpCode::Patchify => {
            let patch = *config.first().ok_or(Error::InvalidConfig)? as usize;
            if patch == 0
                || inputs.len() != 1
                || inputs[0].len() % patch != 0
                || outputs[0].as_slice() != inputs[0].as_ref()
            {
                return Err(Error::InvalidProof);
            }
            let patches = inputs[0].len() / patch;
            if aux.first() != Some(&vec![patches as i64, patch as i64]) {
                return Err(Error::InvalidProof);
            }
        }
        OpCode::ProfileLayerNorm => {
            let rows = *config.first().ok_or(Error::InvalidConfig)? as usize;
            let cols = *config.get(1).ok_or(Error::InvalidConfig)? as usize;
            let valid = *config.get(2).ok_or(Error::InvalidConfig)? as usize;
            let output_shift = *config.get(3).ok_or(Error::InvalidConfig)? as u32;
            if valid == 0 || valid > cols || inputs.len() != 4
                || inputs[0].len() != rows * cols
                || inputs[1].len() != rows * cols
                || inputs[2].len() != rows * cols
                || inputs[3].is_empty()
                || outputs[0].len() != rows * cols
                || raw[0].len() != rows * cols
                || aux.len() != 3
                || aux[0].len() != rows
                || aux[1].len() != rows
                || aux[2].len() != rows
            {
                return Err(Error::ShapeMismatch);
            }
            for row in 0..rows {
                let start = row * cols;
                let sum: i128 = (0..valid).map(|index| inputs[0][start + index] as i128).sum();
                let mean = round_div_half_up(sum, valid as i128)?;
                let mut squares = 0i128;
                for index in 0..valid {
                    let centered = inputs[0][start + index] as i128 - mean;
                    squares += centered * centered;
                }
                let variance = round_div_half_up(squares, valid as i128)?;
                let variance_i64 = i64::try_from(variance).map_err(|_| Error::BoundViolation)?;
                let table_index = profile_rsqrt_index(variance_i64)?;
                if table_index as usize >= inputs[3].len() || aux[0][row] as i128 != mean || aux[1][row] != variance_i64 {
                    return Err(Error::InvalidProof);
                }
                let rstd = inputs[3][table_index as usize];
                if aux[2][row] != rstd {
                    return Err(Error::InvalidProof);
                }
                for index in 0..cols {
                    let image = if index < valid {
                        (inputs[0][start + index] as i128 - mean)
                            .checked_mul(rstd as i128)
                            .and_then(|value| value.checked_mul(inputs[1][start + index] as i128))
                            .ok_or(Error::BoundViolation)?
                    } else {
                        0
                    };
                    let expected = round_div_half_up(image, 1i128 << output_shift)?
                        + inputs[2][start + index] as i128;
                    let expected = i64::try_from(expected).map_err(|_| Error::BoundViolation)?;
                    if raw[0][start + index] != image
                        || outputs[0][start + index] != clamp_value(expected, config, 4)?
                    {
                        return Err(Error::InvalidProof);
                    }
                }
            }
        }
        OpCode::TableIndex => {
            let offset = *config.first().ok_or(Error::InvalidConfig)?;
            let table_len = *config.get(1).ok_or(Error::InvalidConfig)? as usize;
            if table_len == 0 || inputs.len() != 1 || outputs[0].len() != inputs[0].len() {
                return Err(Error::ShapeMismatch);
            }
            for (input, output) in inputs[0].iter().zip(&outputs[0]) {
                let expected = input
                    .saturating_add(offset)
                    .clamp(0, table_len as i64 - 1);
                if *output != expected {
                    return Err(Error::InvalidProof);
                }
            }
        }
        OpCode::EmbeddingIndex => {
            let hidden = *config.first().ok_or(Error::InvalidConfig)? as usize;
            let vocab = *config.get(1).ok_or(Error::InvalidConfig)? as usize;
            if hidden == 0 || vocab == 0 || inputs.len() != 1
                || outputs[0].len() != inputs[0].len() * hidden
            {
                return Err(Error::ShapeMismatch);
            }
            for (row, token) in inputs[0].iter().enumerate() {
                if *token < 0 || *token as usize >= vocab {
                    return Err(Error::InvalidProof);
                }
                for dimension in 0..hidden {
                    if outputs[0][row * hidden + dimension]
                        != (dimension * vocab) as i64 + *token
                    {
                        return Err(Error::InvalidProof);
                    }
                }
            }
        }
        OpCode::ProfileRmsNorm => {
            let rows = *config.first().ok_or(Error::InvalidConfig)? as usize;
            let cols = *config.get(1).ok_or(Error::InvalidConfig)? as usize;
            let valid = *config.get(2).ok_or(Error::InvalidConfig)? as usize;
            let output_shift = *config.get(3).ok_or(Error::InvalidConfig)? as u32;
            if (inputs.len() != 3 && inputs.len() != 4)
                || valid == 0
                || valid > cols
                || inputs[0].len() != rows * cols
                || inputs[1].len() != rows * cols
                || inputs[2].is_empty()
                || outputs[0].len() != rows * cols
                || raw[0].len() != rows * cols
                || aux.len() != 2
                || aux[0].len() != rows
                || aux[1].len() != rows
            {
                return Err(Error::ShapeMismatch);
            }
            for row in 0..rows {
                let start = row * cols;
                let mut square = 0i128;
                for index in 0..valid {
                    let value = inputs[0][start + index] as i128;
                    square += value * value;
                }
                let mean_square = round_div_half_up(square, valid as i128)?;
                let mean_square_i64 = i64::try_from(mean_square).map_err(|_| Error::BoundViolation)?;
                let table_index = profile_rsqrt_index(mean_square_i64)?;
                let base_len = inputs[2].len();
                let table_position = table_index as usize;
                let extension_value = if table_position < base_len {
                    None
                } else {
                    let offset = table_position - base_len;
                    Some(
                        inputs
                            .get(3)
                            .and_then(|extension| extension.get(offset))
                            .copied()
                            .ok_or(Error::InvalidProof)?,
                    )
                };
                if aux[0][row] != mean_square_i64 {
                    return Err(Error::InvalidProof);
                }
                let rstd = match extension_value {
                    Some(value) => value,
                    None => inputs[2][table_position],
                };
                if aux[1][row] != rstd {
                    return Err(Error::InvalidProof);
                }
                for index in 0..cols {
                    let image = if index < valid {
                        (inputs[0][start + index] as i128)
                            .checked_mul(rstd as i128)
                            .and_then(|value| value.checked_mul(inputs[1][start + index] as i128))
                            .ok_or(Error::BoundViolation)?
                    } else {
                        0
                    };
                    let expected = i64::try_from(round_div_half_up(image, 1i128 << output_shift)?)
                        .map_err(|_| Error::BoundViolation)?;
                    if raw[0][start + index] != image
                        || outputs[0][start + index] != clamp_value(expected, config, 4)?
                    {
                        return Err(Error::InvalidProof);
                    }
                }
            }
        }
        OpCode::ProfileRope => {
            let head_dim = *config.first().ok_or(Error::InvalidConfig)? as usize;
            let shift = *config.get(1).ok_or(Error::InvalidConfig)? as u32;
            if head_dim == 0 || head_dim % 2 != 0 || inputs.len() != 3
                || inputs[0].len() % head_dim != 0
                || inputs[1].len() != inputs[0].len() / 2
                || inputs[2].len() != inputs[0].len() / 2
                || outputs[0].len() != inputs[0].len()
                || raw[0].len() != inputs[0].len()
            {
                return Err(Error::ShapeMismatch);
            }
            let half = head_dim / 2;
            let rows = inputs[0].len() / head_dim;
            for row in 0..rows {
                for index in 0..half {
                    let even = inputs[0][row * head_dim + index] as i128;
                    let odd = inputs[0][row * head_dim + half + index] as i128;
                    let c = inputs[1][row * half + index] as i128;
                    let s = inputs[2][row * half + index] as i128;
                    let rotated_even = even.checked_mul(c).and_then(|v| v.checked_sub(odd.checked_mul(s)?)).ok_or(Error::BoundViolation)?;
                    let rotated_odd = even.checked_mul(s).and_then(|v| v.checked_add(odd.checked_mul(c)?)).ok_or(Error::BoundViolation)?;
                    if raw[0][row * head_dim + index] != rotated_even
                        || raw[0][row * head_dim + half + index] != rotated_odd
                        || outputs[0][row * head_dim + index] as i128 != rq(rotated_even, shift)?
                        || outputs[0][row * head_dim + half + index] as i128 != rq(rotated_odd, shift)?
                    {
                        return Err(Error::InvalidProof);
                    }
                }
            }
        }
        OpCode::ProfileStandardize => {
            let rows = *config.first().ok_or(Error::InvalidConfig)? as usize;
            let cols = *config.get(1).ok_or(Error::InvalidConfig)? as usize;
            let output_cols = *config.get(2).ok_or(Error::InvalidConfig)? as usize;
            let valid = *config.get(3).ok_or(Error::InvalidConfig)? as usize;
            let source_row = *config.get(4).ok_or(Error::InvalidConfig)? as usize;
            let output_shift = *config.get(5).ok_or(Error::InvalidConfig)? as u32;
            if (inputs.len() != 2 && inputs.len() != 3)
                || rows == 0
                || cols == 0
                || valid == 0
                || valid > cols
                || output_cols < cols
                || source_row >= rows
                || inputs[0].len() != rows * cols
                || inputs[1].is_empty()
                || outputs[0].len() != rows * output_cols
                || raw[0].len() != rows * output_cols
                || aux.len() != 3
                || aux[0].len() != 1
                || aux[1].len() != 1
                || aux[2].len() != 1
            {
                return Err(Error::ShapeMismatch);
            }
            let source = source_row * cols;
            let mut sum = 0i128;
            for index in 0..valid {
                sum += inputs[0][source + index] as i128;
            }
            let mean = round_div_half_up(sum, valid as i128)?;
            let mut squares = 0i128;
            for index in 0..valid {
                let centered = inputs[0][source + index] as i128 - mean;
                squares += centered * centered;
            }
            let variance = round_div_half_up(squares, valid as i128)?;
            let variance_i64 = i64::try_from(variance).map_err(|_| Error::BoundViolation)?;
            let table_index = profile_rsqrt_index(variance_i64)?;
            let base_len = inputs[1].len();
            let rstd = if (table_index as usize) < base_len {
                inputs[1][table_index as usize]
            } else {
                let offset = table_index as usize - base_len;
                inputs
                    .get(2)
                    .and_then(|extension| extension.get(offset))
                    .copied()
                    .ok_or(Error::InvalidProof)?
            };
            let mean_i64 = i64::try_from(mean).map_err(|_| Error::BoundViolation)?;
            if aux[0][0] != mean_i64 || aux[1][0] != variance_i64 || aux[2][0] != rstd {
                return Err(Error::InvalidProof);
            }
            for row in 0..rows {
                for index in 0..output_cols {
                    let image = if index < valid {
                        (inputs[0][row * cols + index] as i128 - mean_i64 as i128)
                            .checked_mul(rstd as i128)
                            .ok_or(Error::BoundViolation)?
                    } else {
                        0
                    };
                    let expected = i64::try_from(round_div_half_up(image, 1i128 << output_shift)?)
                        .map_err(|_| Error::BoundViolation)?;
                    if raw[0][row * output_cols + index] != image
                        || outputs[0][row * output_cols + index] != clamp_value(expected, config, 6)?
                    {
                        return Err(Error::InvalidProof);
                    }
                }
            }
        }
        OpCode::RowEmbeddingIndex => {
            let hidden = *config.first().ok_or(Error::InvalidConfig)? as usize;
            let vocab = *config.get(1).ok_or(Error::InvalidConfig)? as usize;
            if hidden == 0 || vocab == 0 || inputs.len() != 1
                || outputs[0].len() != inputs[0].len() * hidden
            {
                return Err(Error::ShapeMismatch);
            }
            for (row, token) in inputs[0].iter().enumerate() {
                if *token < 0 || *token as usize >= vocab {
                    return Err(Error::InvalidProof);
                }
                for dimension in 0..hidden {
                    if outputs[0][row * hidden + dimension] != *token * hidden as i64 + dimension as i64 {
                        return Err(Error::InvalidProof);
                    }
                }
            }
        }
        OpCode::CausalAttention => {
            let seq = *config.first().ok_or(Error::InvalidConfig)? as usize;
            let head_dim = *config.get(1).ok_or(Error::InvalidConfig)? as usize;
            let shift = *config.get(2).ok_or(Error::InvalidConfig)? as u32;
            if seq == 0 || head_dim == 0 || inputs.len() != 4 || outputs[0].len() != seq * head_dim || raw[0].len() != seq * head_dim {
                return Err(Error::ShapeMismatch);
            }
            if inputs[0].len() != seq * head_dim || inputs[1].len() != seq * head_dim || inputs[2].len() != seq * head_dim {
                return Err(Error::ShapeMismatch);
            }
            if aux.len() != 3 || aux[0].len() != seq || aux[2].len() != seq {
                return Err(Error::InvalidProof);
            }
            let scale = 1i128.checked_shl(shift).ok_or(Error::BoundViolation)?;
            let mut u_cursor = 0usize;
            for row in 0..seq {
                let mut scores = Vec::with_capacity(row + 1);
                for column in 0..=row {
                    let mut score = 0i128;
                    for index in 0..head_dim {
                        score += inputs[0][row * head_dim + index] as i128
                            * inputs[1][column * head_dim + index] as i128;
                    }
                    scores.push(score);
                }
                let max = *scores.iter().max().ok_or(Error::InvalidProof)?;
                if aux[0][row] as i128 != max {
                    return Err(Error::InvalidProof);
                }
                let mut denominator = 0i128;
                let mut u = Vec::with_capacity(scores.len());
                for score in &scores {
                    let distance = max.checked_sub(*score).ok_or(Error::BoundViolation)?;
                    let distance = i64::try_from(distance).map_err(|_| Error::BoundViolation)?;
                    let entry = exp_profile_value(&inputs[3], distance)?;
                    if entry < 0 {
                        return Err(Error::InvalidProof);
                    }
                    denominator += entry as i128;
                    u.push(entry);
                }
                if denominator == 0 || aux[2][row] as i128 != denominator {
                    return Err(Error::InvalidProof);
                }
                for entry in &u {
                    if aux[1].get(u_cursor).copied() != Some(*entry) {
                        return Err(Error::InvalidProof);
                    }
                    u_cursor += 1;
                }
                for index in 0..head_dim {
                    let mut numerator = 0i128;
                    for (column, entry) in u.iter().enumerate() {
                        numerator += *entry as i128 * inputs[2][column * head_dim + index] as i128;
                    }
                    let scaled = numerator.checked_mul(scale).ok_or(Error::BoundViolation)?;
                    let slot = row * head_dim + index;
                    if raw[0][slot] != scaled || outputs[0][slot] as i128 != scaled.div_euclid(denominator) {
                        return Err(Error::InvalidProof);
                    }
                }
            }
            if u_cursor != aux[1].len() {
                return Err(Error::InvalidProof);
            }
        }
        _ => return Err(Error::UnsupportedPrimitive),
    }
    Ok(())
}


/// Input values recovered from an authenticated relation proof. Used by the
/// verifier to compare the proof configuration with the operator configuration.
pub fn proof_input_values(proof: &OpRelationProof) -> Vec<Vec<i64>> {
    proof
        .inputs
        .iter()
        .zip(&proof.lengths.inputs)
        .map(|(openings, len)| decode(openings, *len))
        .collect()
}

pub fn proof_output_values(proof: &OpRelationProof) -> Vec<Vec<i64>> {
    proof
        .outputs
        .iter()
        .zip(&proof.lengths.outputs)
        .map(|(openings, len)| decode(openings, *len))
        .collect()
}
