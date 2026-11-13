//! Per-operator algebraic relations and `Operation::prove/verify` material.
//!
//! The prover commits input, output, raw, quotient, helper and auxiliary
//! vectors as authenticated oracles. The verifier checks the op-specific
//! equations from the opened vectors and never invokes the operator's forward
//! routine.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::op::OpCode;
use crate::proof::binding::{
    commit_rows_with_openings, oracle_leaf, verify_opening, OracleId, OracleOpening,
    OracleRegistry,
};
use zkie_engine::compute::cpu::merkle_build;
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

#[derive(Clone, Debug)]
pub struct OpRelationPublic {
    pub opcode: OpCode,
    pub config: Vec<i64>,
    pub registry: OracleRegistry,
    pub input_oracles: Vec<OracleId>,
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

impl RelationConfig for crate::argmax::ArgmaxOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(Vec::new())
    }
}

impl RelationConfig for crate::hadamard::HadamardOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![self.shift as i64])
    }
}

impl RelationConfig for crate::gated_ffn::GatedFfnOp {
    fn relation_config(&self, _inputs: &[Vec<i64>]) -> Result<Vec<i64>> {
        Ok(vec![self.shift as i64])
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
    let (_, mut openings) = commit_rows_with_openings(registry, label, &rows)?;
    for opening in &mut openings {
        opening.path.clear();
    }
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
    let (_, mut openings) = commit_rows_with_openings(registry, label, &[words])?;
    for opening in &mut openings {
        opening.path.clear();
    }
    Ok(openings)
}

fn relation_label(opcode: OpCode, config: &[i64]) -> Digest {
    let mut bytes = Vec::new();
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

fn input_widths(opcode: OpCode, config: &[i64], lengths: &[usize]) -> Vec<usize> {
    let count = lengths.len();
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
            if !lengths.is_empty() && lengths[0] > 0 {
                widths[0] = lengths[0];
            }
            if lengths.len() > 1 && lengths[1] > 0 {
                widths[1] = lengths[1];
            }
        }
        OpCode::Softmax => {
            if lengths.len() > 1 && lengths[1] > 0 {
                widths[1] = lengths[1];
            }
        }
        OpCode::SoftmaxRows => {
            if let Some(cols) = config.get(1) {
                if *cols > 0 {
                    widths[0] = *cols as usize;
                }
            }
            if lengths.len() > 1 && lengths[1] > 0 {
                widths[1] = lengths[1];
            }
        }
        OpCode::CausalAttention => {
            if lengths.len() > 3 && lengths[3] > 0 {
                widths[3] = lengths[3];
            }
        }
        OpCode::ProfileLayerNorm => {
            if lengths.len() > 3 && lengths[3] > 0 {
                widths[3] = lengths[3];
            }
            if lengths.len() > 2 {
                widths[1] = lengths[1];
                widths[2] = lengths[2];
            }
        }
        OpCode::ProfileRmsNorm => {
            if lengths.len() > 2 {
                widths[1] = lengths[1];
                widths[2] = lengths[2];
            }
            if lengths.len() > 3 {
                widths[3] = lengths[3];
            }
        }
        OpCode::ProfileRope => {
            for (index, length) in lengths.iter().enumerate() {
                widths[index] = *length;
            }
        }
        OpCode::RowEmbeddingIndex => {
            if let Some(length) = lengths.first() {
                widths[0] = *length;
            }
        }
        OpCode::ProfileStandardize => {
            if lengths.len() > 1 {
                widths[1] = lengths[1];
            }
            if lengths.len() > 2 {
                widths[2] = lengths[2];
            }
        }
        OpCode::Add => {
            for (index, length) in lengths.iter().enumerate() {
                widths[index] = *length;
            }
        }
        OpCode::Argmax | OpCode::TableIndex => {
            if let Some(length) = lengths.first() {
                widths[0] = *length;
            }
        }
        _ => {}
    }
    widths
}

fn output_widths(opcode: OpCode, config: &[i64], lengths: &[usize]) -> Vec<usize> {
    let mut widths = vec![1usize; lengths.len()];
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
            for (index, length) in lengths.iter().enumerate() {
                widths[index] = *length;
            }
            return widths;
        }
        _ => {}
    }
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
    prove_op_relation_with_inputs(op, inputs, config, &[])
}

/// Prove a relation with every input committed and opened in this proof. The
/// request path keeps this form.
pub fn prove_op_relation_with_inputs(
    op: &dyn PureForward,
    inputs: &[Vec<i64>],
    config: &[i64],
    external: &[usize],
) -> Result<(OpRelationPublic, OpRelationProof)> {
    prove_op_relation_inner(op, inputs, config, external, true)
}

/// Prove a relation where every input is supplied by the trusted plan and is
/// materialized by the verifier: no input oracle is committed and no input
/// opening is carried. The external list still marks the plan-mapped asset
/// ports; those inputs carry no opening in either form.
pub fn prove_op_relation_shared(
    op: &dyn PureForward,
    inputs: &[Vec<i64>],
    config: &[i64],
    external: &[usize],
) -> Result<(OpRelationPublic, OpRelationProof)> {
    prove_op_relation_inner(op, inputs, config, external, false)
}

fn prove_op_relation_inner(
    op: &dyn PureForward,
    inputs: &[Vec<i64>],
    config: &[i64],
    external: &[usize],
    commit_inputs: bool,
) -> Result<(OpRelationPublic, OpRelationProof)> {
    let _profile = zkie_core::profile::Scope::operator(op.opcode() as u16);
    let record = op.forward_values(inputs)?;
    let opcode = op.opcode();
    let label = relation_label(opcode, config);
    let mut registry = OracleRegistry::new();
    let input_lengths: Vec<usize> = inputs.iter().map(Vec::len).collect();
    let input_widths = input_widths(opcode, config, &input_lengths);
    let mut input_openings = Vec::new();
    let mut input_oracles = Vec::new();
    for (index, values) in inputs.iter().enumerate() {
        if !commit_inputs || external.contains(&index) {
            input_oracles.push(OracleId(u64::MAX));
            input_openings.push(Vec::new());
            continue;
        }
        let openings = commit_group_width(
            &mut registry,
            &label,
            values,
            *input_widths.get(index).unwrap_or(&1),
        )?;
        input_oracles.push(openings[0].oracle);
        input_openings.push(openings);
    }
    let output_lengths: Vec<usize> = record.outputs.iter().map(Vec::len).collect();
    let output_widths = output_widths(opcode, config, &output_lengths);
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

pub(crate) fn verify_group(
    registry: &OracleRegistry,
    openings: &[OracleOpening],
    len: usize,
    width: usize,
    wide: bool,
) -> Result<()> {
    if wide {
        let words = len.checked_mul(2).ok_or(Error::ResourceLimit)?;
        if openings.len() != 1 {
            return Err(Error::ShapeMismatch);
        }
        let opening = &openings[0];
        let root = registry
            .get(opening.oracle)
            .ok_or(Error::ProverStateMissing)?;
        if root.count != 1 {
            return Err(Error::StatementMismatch);
        }
        if opening.index != 0 {
            return Err(Error::InvalidIndex);
        }
        if opening.values.len() != words {
            return Err(Error::ShapeMismatch);
        }
        if !opening.path.is_empty() {
            return Err(Error::InvalidProof);
        }
        return verify_opening(registry, opening, 1);
    }
    if width == 0 {
        return Err(Error::InvalidConfig);
    }
    if openings.is_empty() {
        return Err(Error::ShapeMismatch);
    }
    let oracle = openings[0].oracle;
    let root = registry.get(oracle).ok_or(Error::ProverStateMissing)?;
    let count = openings.len() as u64;
    if root.count != count {
        return Err(Error::StatementMismatch);
    }
    if !count.is_power_of_two() {
        return Err(Error::InvalidConfig);
    }
    if len % width != 0 {
        return Err(Error::ShapeMismatch);
    }
    let data_rows = (len / width) as u64;
    let expected = data_rows
        .max(1)
        .checked_next_power_of_two()
        .ok_or(Error::ResourceLimit)?;
    if count != expected {
        return Err(Error::StatementMismatch);
    }
    let depth = count.trailing_zeros() as usize;
    for (position, opening) in openings.iter().enumerate() {
        if opening.oracle != oracle {
            return Err(Error::StatementMismatch);
        }
        if opening.index != position as u64 {
            return Err(Error::InvalidIndex);
        }
        if opening.values.len() != width {
            return Err(Error::ShapeMismatch);
        }
        if !opening.path.is_empty() && opening.path.len() != depth {
            return Err(Error::InvalidProof);
        }
        if position as u64 >= data_rows
            && opening.values.iter().any(|value| *value != 0)
        {
            return Err(Error::StatementMismatch);
        }
    }
    if openings.iter().all(|opening| opening.path.is_empty()) {
        // Every leaf of the tree is present and ordered by position, so the
        // root follows from the leaves alone: one root recomputation replaces
        // one Merkle path per opening. The order and padding checks above are
        // the same as those of the path form.
        let mut leaves = Vec::with_capacity(openings.len());
        for opening in openings {
            leaves.push(oracle_leaf(opening.index, &opening.values));
        }
        let rebuilt = merkle_build(&root.start, &leaves)?;
        if rebuilt.root != root.root {
            return Err(Error::InvalidProof);
        }
    } else {
        for opening in openings {
            verify_opening(registry, opening, count)?;
        }
    }
    Ok(())
}

fn decode(openings: &[OracleOpening], len: usize) -> Result<Vec<i64>> {
    if openings.is_empty() {
        return Ok(Vec::new());
    }
    let mut values = Vec::with_capacity(len);
    for (position, opening) in openings.iter().enumerate() {
        if opening.index != position as u64 {
            return Err(Error::InvalidIndex);
        }
        values.extend(opening.values.iter().map(|value| *value as i64));
        if values.len() >= len {
            break;
        }
    }
    if values.len() < len {
        return Err(Error::ShapeMismatch);
    }
    values.truncate(len);
    Ok(values)
}

fn decode_wide(openings: &[OracleOpening], len: usize) -> Result<Vec<i128>> {
    if openings.is_empty() {
        return Ok(Vec::new());
    }
    let mut values = Vec::with_capacity(len);
    for (position, opening) in openings.iter().enumerate() {
        if opening.index != position as u64 {
            return Err(Error::InvalidIndex);
        }
        if opening.values.len() % 2 != 0 {
            return Err(Error::ShapeMismatch);
        }
        for pair in opening.values.chunks_exact(2) {
            values.push(((pair[1] as i128) << 64) | pair[0] as i128);
        }
        if values.len() >= len {
            break;
        }
    }
    if values.len() < len {
        return Err(Error::ShapeMismatch);
    }
    values.truncate(len);
    Ok(values)
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

/// Number of output tensors every relation of this opcode produces.
fn output_tensor_count(opcode: OpCode) -> usize {
    match opcode {
        OpCode::KvCache => 2,
        _ => 1,
    }
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
    if proof.inputs.is_empty() || proof.outputs.len() != output_tensor_count(public.opcode) {
        return Err(Error::ShapeMismatch);
    }
    let input_widths = input_widths(public.opcode, &public.config, &proof.lengths.inputs);
    let output_widths = output_widths(public.opcode, &public.config, &proof.lengths.outputs);
    for (index, openings) in proof.inputs.iter().enumerate() {
        if let Some(values) = external_values.get(index).and_then(Option::as_ref) {
            if !openings.is_empty() || values.len() != proof.lengths.inputs[index] {
                return Err(Error::StatementMismatch);
            }
            continue;
        }
        if openings.first().map(|opening| opening.oracle) != Some(public.input_oracles[index]) {
            return Err(Error::StatementMismatch);
        }
        verify_group(
            &public.registry,
            openings,
            proof.lengths.inputs[index],
            *input_widths.get(index).unwrap_or(&1),
            false,
        )?;
    }
    for (index, openings) in proof.outputs.iter().enumerate() {
        if openings.first().map(|opening| opening.oracle) != Some(public.output_oracles[index]) {
            return Err(Error::StatementMismatch);
        }
        verify_group(
            &public.registry,
            openings,
            proof.lengths.outputs[index],
            *output_widths.get(index).unwrap_or(&1),
            false,
        )?;
    }
    for (index, openings) in proof.raw.iter().enumerate() {
        verify_group(
            &public.registry,
            openings,
            *proof.lengths.raw.get(index).ok_or(Error::ShapeMismatch)?,
            0,
            true,
        )?;
    }
    for (index, openings) in proof.quotients.iter().enumerate() {
        verify_group(
            &public.registry,
            openings,
            *proof
                .lengths
                .quotients
                .get(index)
                .ok_or(Error::ShapeMismatch)?,
            1,
            false,
        )?;
    }
    for (index, openings) in proof.helpers.iter().enumerate() {
        verify_group(
            &public.registry,
            openings,
            *proof
                .lengths
                .helpers
                .get(index)
                .ok_or(Error::ShapeMismatch)?,
            1,
            false,
        )?;
    }
    for (index, openings) in proof.aux.iter().enumerate() {
        verify_group(
            &public.registry,
            openings,
            *proof.lengths.aux.get(index).ok_or(Error::ShapeMismatch)?,
            1,
            false,
        )?;
    }
    let inputs: Vec<Vec<i64>> = proof
        .inputs
        .iter()
        .zip(&proof.lengths.inputs)
        .enumerate()
        .map(|(index, (openings, len))| -> Result<Vec<i64>> {
            if let Some(values) = external_values.get(index).and_then(Option::as_ref) {
                Ok(values.clone())
            } else {
                decode(openings, *len)
            }
        })
        .collect::<Result<_>>()?;
    let outputs: Vec<Vec<i64>> = proof
        .outputs
        .iter()
        .zip(&proof.lengths.outputs)
        .map(|(openings, len)| decode(openings, *len))
        .collect::<Result<_>>()?;
    let raw: Vec<Vec<i128>> = proof
        .raw
        .iter()
        .zip(&proof.lengths.raw)
        .map(|(openings, len)| decode_wide(openings, *len))
        .collect::<Result<_>>()?;
    let quotients: Vec<Vec<i64>> = proof
        .quotients
        .iter()
        .zip(&proof.lengths.quotients)
        .map(|(openings, len)| decode(openings, *len))
        .collect::<Result<_>>()?;
    let helpers: Vec<Vec<i64>> = proof
        .helpers
        .iter()
        .zip(&proof.lengths.helpers)
        .map(|(openings, len)| decode(openings, *len))
        .collect::<Result<_>>()?;
    let aux: Vec<Vec<i64>> = proof
        .aux
        .iter()
        .zip(&proof.lengths.aux)
        .map(|(openings, len)| decode(openings, *len))
        .collect::<Result<_>>()?;
    let config = &public.config;
    match public.opcode {
        OpCode::Layout => {
            let kind = config.first().copied().ok_or(Error::InvalidConfig)?;
            let from_x = *config.get(1).ok_or(Error::InvalidConfig)? as usize;
            let from_y = *config.get(2).ok_or(Error::InvalidConfig)? as usize;
            let to_x = *config.get(3).ok_or(Error::InvalidConfig)? as usize;
            let to_y = *config.get(4).ok_or(Error::InvalidConfig)? as usize;
            let cells = from_x
                .checked_mul(from_y)
                .ok_or(Error::ShapeMismatch)?;
            let targets = to_x
                .checked_mul(to_y)
                .ok_or(Error::ShapeMismatch)?;
            if cells != targets
                || inputs.len() != 1
                || inputs[0].len() != cells
                || outputs[0].len() != targets
            {
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
            let bias_header = 5usize.checked_add(bias_len).ok_or(Error::ResourceLimit)?;
            let left = m.checked_mul(k).ok_or(Error::ResourceLimit)?;
            let right = k.checked_mul(n).ok_or(Error::ResourceLimit)?;
            let output_len = m.checked_mul(n).ok_or(Error::ResourceLimit)?;
            if config.len() != bias_header
                || inputs.len() != 2
                || inputs[0].len() != left
                || inputs[1].len() != right
                || outputs[0].len() != output_len
                || raw.is_empty() || raw[0].len() != output_len
            {
                return Err(Error::ShapeMismatch);
            }
            let bias = &config[5..bias_header];
            let computed = crate::registry::matmul_raw(&inputs[0], &inputs[1], m, n, k)?;
            (0..output_len).into_par_iter().try_for_each(|slot| -> Result<()> {
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
            if inputs.len() != 2 || inputs[0].is_empty() || raw.is_empty() || outputs[0].len() != inputs[0].len() {
                return Err(Error::ShapeMismatch);
            }
            let length = inputs[0].len();
            if raw[0].len() != length {
                return Err(Error::ShapeMismatch);
            }
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
                denominator = denominator
                    .checked_add(entry as i128)
                    .ok_or(Error::BoundViolation)?;
            }
            if denominator == 0 || aux[2][0] as i128 != denominator {
                return Err(Error::InvalidProof);
            }
            let scale = 1i128.checked_shl(shift).ok_or(Error::BoundViolation)?;
            for index in 0..length {
                let numerator = (aux[1][index] as i128)
                    .checked_mul(scale)
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
                || raw.is_empty() || raw[0].len() != count
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
            let cells = rows.checked_mul(cols).ok_or(Error::ResourceLimit)?;
            if valid == 0 || valid > cols || inputs.len() != 2
                || inputs[0].len() != cells
                || inputs[1].len() != cells
                || outputs[0].len() != cells
                || raw.is_empty() || raw[0].len() != cells
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
            let doubled_shift = shift.checked_mul(2).ok_or(Error::BoundViolation)?;
            let scale_squared = 1u128
                .checked_shl(doubled_shift)
                .ok_or(Error::BoundViolation)?;
            for row in 0..rows {
                let start = row * cols;
                let mut mean = 0i128;
                if layer_norm {
                    for index in 0..valid {
                        mean = mean
                            .checked_add(inputs[0][start + index] as i128)
                            .ok_or(Error::BoundViolation)?;
                    }
                    mean = mean.div_euclid(valid as i128);
                } else {
                    for index in 0..valid {
                        let value = inputs[0][start + index] as i128;
                        let square = value.checked_mul(value).ok_or(Error::BoundViolation)?;
                        mean = mean.checked_add(square).ok_or(Error::BoundViolation)?;
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
                        let square = centered.checked_mul(centered).ok_or(Error::BoundViolation)?;
                        variance = variance.checked_add(square).ok_or(Error::BoundViolation)?;
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
            if inputs[1].len() != pairs || inputs[2].len() != pairs || outputs[0].len() != inputs[0].len() || raw.is_empty() || raw[0].len() != inputs[0].len() {
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
            if inputs.len() != 1 || k == 0 || k > inputs[0].len() {
                return Err(Error::ShapeMismatch);
            }
            let output_len = k.checked_mul(2).ok_or(Error::ResourceLimit)?;
            if outputs[0].len() != output_len {
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
            if inputs.len() != 1 || top_k == 0 || top_k > inputs[0].len() {
                return Err(Error::ShapeMismatch);
            }
            let output_len = top_k
                .checked_mul(2)
                .and_then(|value| value.checked_add(1))
                .ok_or(Error::ResourceLimit)?;
            if outputs[0].len() != output_len {
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
            for (index, value) in inputs[0].iter().zip(&inputs[1]) {
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
            if inputs.len() != 2 || cols == 0 || inputs[0].len() % cols != 0 {
                return Err(Error::ShapeMismatch);
            }
            let output_len = inputs[1]
                .len()
                .checked_mul(cols)
                .ok_or(Error::ResourceLimit)?;
            if outputs[0].len() != output_len {
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
            let left = batch
                .checked_mul(m)
                .and_then(|value| value.checked_mul(k))
                .ok_or(Error::ResourceLimit)?;
            let right = batch
                .checked_mul(k)
                .and_then(|value| value.checked_mul(n))
                .ok_or(Error::ResourceLimit)?;
            let output_len = batch
                .checked_mul(m)
                .and_then(|value| value.checked_mul(n))
                .ok_or(Error::ResourceLimit)?;
            if inputs.len() != 2
                || inputs[0].len() != left
                || inputs[1].len() != right
                || outputs[0].len() != output_len
                || raw.is_empty() || raw[0].len() != output_len
            {
                return Err(Error::ShapeMismatch);
            }
            for batch_index in 0..batch {
                for row in 0..m {
                    for column in 0..n {
                        let mut expected = 0i128;
                        for inner in 0..k {
                            let term = (inputs[0][batch_index * m * k + row * k + inner] as i128)
                                .checked_mul(
                                    inputs[1][batch_index * k * n + inner * n + column] as i128,
                                )
                                .ok_or(Error::BoundViolation)?;
                            expected = expected.checked_add(term).ok_or(Error::BoundViolation)?;
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
            if inputs.len() != 3 || cols == 0 || inputs[0].len() % cols != 0 {
                return Err(Error::ShapeMismatch);
            }
            let table_len = cols.checked_mul(n).ok_or(Error::ResourceLimit)?;
            if inputs[2].len() != table_len {
                return Err(Error::ShapeMismatch);
            }
            let rows = inputs[0].len() / cols;
            let output_len = inputs[1]
                .len()
                .checked_mul(n)
                .ok_or(Error::ResourceLimit)?;
            if outputs[0].len() != output_len || raw.is_empty() || raw[0].len() != output_len {
                return Err(Error::ShapeMismatch);
            }
            for (slot, index) in inputs[1].iter().enumerate() {
                if *index < 0 || *index as usize >= rows {
                    return Err(Error::InvalidIndex);
                }
                for column in 0..n {
                    let mut expected = 0i128;
                    for inner in 0..cols {
                        let term = (inputs[0][*index as usize * cols + inner] as i128)
                            .checked_mul(inputs[2][inner * n + column] as i128)
                            .ok_or(Error::BoundViolation)?;
                        expected = expected.checked_add(term).ok_or(Error::BoundViolation)?;
                    }
                    let position = slot * n + column;
                    if raw[0][position] != expected || rq(expected, shift)? != outputs[0][position] as i128 {
                        return Err(Error::InvalidProof);
                    }
                }
            }
        }
        OpCode::IndexGather => {
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
            if inputs.len() != 2 || inputs[0].len() != inputs[1].len() || outputs[0].len() != inputs[0].len() || raw.is_empty() || raw[0].len() != inputs[0].len() {
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
            let doubled = inputs[0]
                .len()
                .checked_mul(2)
                .ok_or(Error::ShapeMismatch)?;
            if inputs.len() != 2
                || inputs[0].len() != inputs[1].len()
                || outputs[0].len() != doubled
                || quotients.len() != 1
                || quotients[0].len() != inputs[0].len()
            {
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
            let table = width
                .checked_mul(out_cols)
                .ok_or(Error::ResourceLimit)?;
            let expected_len = table
                .checked_mul(2)
                .and_then(|value| value.checked_add(3))
                .ok_or(Error::ResourceLimit)?;
            if width == 0 || out_cols == 0 || config.len() != expected_len
                || inputs.len() != 2
                || inputs[0].len() != width
                || inputs[1].len() != width
                || outputs[0].len() != out_cols
                || raw.len() != 3
                || raw.is_empty() || raw[0].len() != out_cols
                || raw[1].len() != out_cols
                || raw[2].len() != out_cols
            {
                return Err(Error::ShapeMismatch);
            }
            let shared = &config[3..3 + table];
            let routed = &config[3 + table..];
            for column in 0..out_cols {
                let mut shared_raw = 0i128;
                let mut routed_raw = 0i128;
                for index in 0..width {
                    let shared_term = (inputs[0][index] as i128)
                        .checked_mul(shared[index * out_cols + column] as i128)
                        .ok_or(Error::BoundViolation)?;
                    shared_raw = shared_raw
                        .checked_add(shared_term)
                        .ok_or(Error::BoundViolation)?;
                    let routed_term = (inputs[1][index] as i128)
                        .checked_mul(routed[index * out_cols + column] as i128)
                        .ok_or(Error::BoundViolation)?;
                    routed_raw = routed_raw
                        .checked_add(routed_term)
                        .ok_or(Error::BoundViolation)?;
                }
                let total = shared_raw
                    .checked_add(routed_raw)
                    .ok_or(Error::BoundViolation)?;
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
            let total = capacity.checked_mul(width).ok_or(Error::ResourceLimit)?;
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
            if patch == 0 || inputs.len() != 1 || inputs[0].len() % patch != 0 || outputs[0] != inputs[0] {
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
            let cells = rows.checked_mul(cols).ok_or(Error::ResourceLimit)?;
            if valid == 0 || valid > cols || inputs.len() != 4
                || inputs[0].len() != cells
                || inputs[1].len() != cells
                || inputs[2].len() != cells
                || inputs[3].is_empty()
                || outputs[0].len() != cells
                || raw.is_empty() || raw[0].len() != cells
                || aux.len() != 3
                || aux[0].len() != rows
                || aux[1].len() != rows
                || aux[2].len() != rows
            {
                return Err(Error::ShapeMismatch);
            }
            for row in 0..rows {
                let start = row * cols;
                let mut sum = 0i128;
                for index in 0..valid {
                    sum = sum
                        .checked_add(inputs[0][start + index] as i128)
                        .ok_or(Error::BoundViolation)?;
                }
                let mean = round_div_half_up(sum, valid as i128)?;
                let mut squares = 0i128;
                for index in 0..valid {
                    let centered = inputs[0][start + index] as i128 - mean;
                    let square = centered.checked_mul(centered).ok_or(Error::BoundViolation)?;
                    squares = squares.checked_add(square).ok_or(Error::BoundViolation)?;
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
            if hidden == 0 || vocab == 0 || inputs.len() != 1 {
                return Err(Error::ShapeMismatch);
            }
            let output_len = inputs[0]
                .len()
                .checked_mul(hidden)
                .ok_or(Error::ResourceLimit)?;
            if outputs[0].len() != output_len {
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
            let cells = rows.checked_mul(cols).ok_or(Error::ResourceLimit)?;
            if (inputs.len() != 3 && inputs.len() != 4)
                || valid == 0
                || valid > cols
                || inputs[0].len() != cells
                || inputs[1].len() != cells
                || inputs[2].is_empty()
                || outputs[0].len() != cells
                || raw.is_empty() || raw[0].len() != cells
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
                    let term = value.checked_mul(value).ok_or(Error::BoundViolation)?;
                    square = square.checked_add(term).ok_or(Error::BoundViolation)?;
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
                || raw.is_empty() || raw[0].len() != inputs[0].len()
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
            let cells = rows.checked_mul(cols).ok_or(Error::ResourceLimit)?;
            let out_cells = rows
                .checked_mul(output_cols)
                .ok_or(Error::ResourceLimit)?;
            if (inputs.len() != 2 && inputs.len() != 3)
                || rows == 0
                || cols == 0
                || valid == 0
                || valid > cols
                || output_cols < cols
                || source_row >= rows
                || inputs[0].len() != cells
                || inputs[1].is_empty()
                || outputs[0].len() != out_cells
                || raw.is_empty() || raw[0].len() != out_cells
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
                sum = sum
                    .checked_add(inputs[0][source + index] as i128)
                    .ok_or(Error::BoundViolation)?;
            }
            let mean = round_div_half_up(sum, valid as i128)?;
            let mut squares = 0i128;
            for index in 0..valid {
                let centered = inputs[0][source + index] as i128 - mean;
                let square = centered.checked_mul(centered).ok_or(Error::BoundViolation)?;
                squares = squares.checked_add(square).ok_or(Error::BoundViolation)?;
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
            if hidden == 0 || vocab == 0 || inputs.len() != 1 {
                return Err(Error::ShapeMismatch);
            }
            let output_len = inputs[0]
                .len()
                .checked_mul(hidden)
                .ok_or(Error::ResourceLimit)?;
            if outputs[0].len() != output_len {
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
            let cells = seq.checked_mul(head_dim).ok_or(Error::ResourceLimit)?;
            if seq == 0 || head_dim == 0 || inputs.len() != 4 || outputs[0].len() != cells || raw.is_empty() || raw[0].len() != cells {
                return Err(Error::ShapeMismatch);
            }
            if inputs[0].len() != cells || inputs[1].len() != cells || inputs[2].len() != cells {
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
                        let term = (inputs[0][row * head_dim + index] as i128)
                            .checked_mul(inputs[1][column * head_dim + index] as i128)
                            .ok_or(Error::BoundViolation)?;
                        score = score.checked_add(term).ok_or(Error::BoundViolation)?;
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
                    denominator = denominator
                        .checked_add(entry as i128)
                        .ok_or(Error::BoundViolation)?;
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
                        let term = (*entry as i128)
                            .checked_mul(inputs[2][column * head_dim + index] as i128)
                            .ok_or(Error::BoundViolation)?;
                        numerator = numerator.checked_add(term).ok_or(Error::BoundViolation)?;
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
    }
    Ok(())
}


/// Input values recovered from a relation proof. The decode enforces
/// coordinate order and full length, so reordered or duplicated openings are
/// rejected even before the oracle roots are checked. Production callers use
/// this only on proofs that are verified in the same path.
pub fn proof_input_values(proof: &OpRelationProof) -> Result<Vec<Vec<i64>>> {
    proof
        .inputs
        .iter()
        .zip(&proof.lengths.inputs)
        .map(|(openings, len)| decode(openings, *len))
        .collect()
}

pub fn proof_output_values(proof: &OpRelationProof) -> Result<Vec<Vec<i64>>> {
    proof
        .outputs
        .iter()
        .zip(&proof.lengths.outputs)
        .map(|(openings, len)| decode(openings, *len))
        .collect()
}
