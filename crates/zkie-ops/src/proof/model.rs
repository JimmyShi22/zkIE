//! Layer-by-layer model binding: `logits = Model(weights, prompt)` is checked
//! as a chain of algebraic relations, not by re-running the forward code.
//!
//! Every layer opens its weights, input, output, raw accumulator, remainder and
//! (for a ReLU layer) selector bits from committed oracles. The verifier checks
//! the linear identity, the floor-rounding quotient relation, the ReLU
//! selector constraints, the weight range, and that each layer's input is the
//! previous layer's output. The final layer output is the logits row.

use zkie_core::error::{Error, Result};
use zkie_core::session::Statement;
use zkie_core::types::Digest;

use crate::proof::binding::{
    commit_rows_with_openings, verify_opening, OracleId, OracleOpening, OracleRegistry,
};
use crate::proof::request::output_policy::statement_digest;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelLayer {
    pub rows: usize,
    pub cols: usize,
    pub shift: u32,
    pub relu: bool,
    pub weights: Vec<i64>,
    pub bias: Vec<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerWitness {
    pub input: Vec<i64>,
    pub raw: Vec<i128>,
    pub remainders: Vec<i64>,
    pub pre_activation: Vec<i64>,
    pub output: Vec<i64>,
    pub bits: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelWitness {
    pub input: Vec<i64>,
    pub layers: Vec<LayerWitness>,
    pub logits: Vec<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerProof {
    pub input: Vec<OracleOpening>,
    pub weights: Vec<OracleOpening>,
    pub bias: Vec<OracleOpening>,
    pub raw: Vec<OracleOpening>,
    pub remainders: Vec<OracleOpening>,
    pub pre_activation: Vec<OracleOpening>,
    pub output: Vec<OracleOpening>,
    pub bits: Vec<OracleOpening>,
}

#[derive(Clone, Debug)]
pub struct ModelBindingPublic {
    pub statement: Statement,
    pub registry: OracleRegistry,
    pub input_oracle: OracleId,
    pub output_oracles: Vec<OracleId>,
    pub logits_oracle: OracleId,
    pub layers: Vec<(usize, usize, u32, bool)>,
    pub weight_bounds: (i64, i64),
    pub expected_token: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelBindingProof {
    pub statement_digest: Digest,
    pub layers: Vec<LayerProof>,
}

fn pad_rows(values: &[Vec<u64>]) -> Result<Vec<Vec<u64>>> {
    let mut rows = values.to_vec();
    let target = rows.len().max(1).next_power_of_two();
    while rows.len() < target {
        rows.push(vec![0u64; values.first().map(Vec::len).unwrap_or(1)]);
    }
    Ok(rows)
}

fn commit_scalars(
    registry: &mut OracleRegistry,
    label: &Digest,
    values: &[i64],
) -> Result<Vec<OracleOpening>> {
    let rows: Vec<Vec<u64>> = values.iter().map(|value| vec![*value as u64]).collect();
    let (_, mut openings) = commit_rows_with_openings(registry, label, &pad_rows(&rows)?)?;
    openings.truncate(values.len());
    Ok(openings)
}

fn commit_wide(
    registry: &mut OracleRegistry,
    label: &Digest,
    values: &[i128],
) -> Result<Vec<OracleOpening>> {
    let rows: Vec<Vec<u64>> = values
        .iter()
        .map(|value| vec![*value as u64, (*value >> 64) as u64])
        .collect();
    let (_, mut openings) = commit_rows_with_openings(registry, label, &pad_rows(&rows)?)?;
    openings.truncate(values.len());
    Ok(openings)
}

fn values_of(openings: &[OracleOpening]) -> Vec<i64> {
    openings
        .iter()
        .map(|opening| opening.values[0] as i64)
        .collect()
}

fn wide_values(openings: &[OracleOpening]) -> Vec<i128> {
    openings
        .iter()
        .map(|opening| ((opening.values[1] as i128) << 64) | opening.values[0] as i128)
        .collect()
}

pub fn prove_model(
    statement: &Statement,
    model: &[ModelLayer],
    witness: &ModelWitness,
    expected_token: u32,
) -> Result<(ModelBindingPublic, ModelBindingProof)> {
    if model.is_empty() || model.len() != witness.layers.len() {
        return Err(Error::ShapeMismatch);
    }
    if witness.input.len() != model[0].cols {
        return Err(Error::ShapeMismatch);
    }
    let label = statement_digest(statement)?;
    let mut registry = OracleRegistry::new();
    let input_rows: Vec<Vec<u64>> = witness
        .input
        .iter()
        .map(|value| vec![*value as u64])
        .collect();
    let (input_oracle, mut current_input) =
        commit_rows_with_openings(&mut registry, &label, &pad_rows(&input_rows)?)?;
    current_input.truncate(witness.input.len());
    let mut layers = Vec::new();
    let mut output_oracles = Vec::new();
    for (index, (layer, layer_witness)) in model.iter().zip(&witness.layers).enumerate() {
        if layer.weights.len() != layer.rows * layer.cols
            || layer.bias.len() != layer.rows
            || layer_witness.input.len() != layer.cols
            || layer_witness.raw.len() != layer.rows
            || layer_witness.remainders.len() != layer.rows
            || layer_witness.pre_activation.len() != layer.rows
            || layer_witness.output.len() != layer.rows
            || layer_witness.bits.len() != layer.rows
        {
            return Err(Error::ShapeMismatch);
        }
        if index > 0 && layer_witness.input != witness.layers[index - 1].output {
            return Err(Error::StatementMismatch);
        }
        let weights = commit_scalars(&mut registry, &label, &layer.weights)?;
        let bias = commit_scalars(&mut registry, &label, &layer.bias)?;
        let raw = commit_wide(&mut registry, &label, &layer_witness.raw)?;
        let remainders = commit_scalars(&mut registry, &label, &layer_witness.remainders)?;
        let pre_activation = commit_scalars(&mut registry, &label, &layer_witness.pre_activation)?;
        let output = commit_scalars(&mut registry, &label, &layer_witness.output)?;
        let bits = commit_scalars(
            &mut registry,
            &label,
            &layer_witness.bits.iter().map(|bit| *bit as i64).collect::<Vec<_>>(),
        )?;
        output_oracles.push(output[0].oracle);
        // The output oracle was registered by commit_scalars as its own tree.
        // Reopening the next layer's input uses these same openings, so the
        // edge is an oracle identity, not a value comparison.
        layers.push(LayerProof {
            input: current_input.clone(),
            weights,
            bias,
            raw,
            remainders,
            pre_activation,
            output: output.clone(),
            bits,
        });
        current_input = output;
    }
    if witness.logits != witness.layers.last().ok_or(Error::IncompleteArtifact)?.output {
        return Err(Error::StatementMismatch);
    }
    let logits_oracle = output_oracles.last().copied().ok_or(Error::IncompleteArtifact)?;
    Ok((
        ModelBindingPublic {
            statement: statement.clone(),
            registry,
            input_oracle,
            output_oracles,
            logits_oracle,
            layers: model
                .iter()
                .map(|layer| (layer.rows, layer.cols, layer.shift, layer.relu))
                .collect(),
            weight_bounds: (-(1i64 << 15), (1i64 << 15) - 1),
            expected_token,
        },
        ModelBindingProof {
            statement_digest: label,
            layers,
        },
    ))
}

fn verify_vector(
    registry: &OracleRegistry,
    openings: &[OracleOpening],
    oracle: OracleId,
) -> Result<()> {
    let root = registry.get(oracle).ok_or(Error::ProverStateMissing)?;
    for opening in openings {
        if opening.oracle != oracle {
            return Err(Error::StatementMismatch);
        }
        verify_opening(registry, opening, root.count)?;
    }
    Ok(())
}

pub fn verify_model(public: &ModelBindingPublic, proof: &ModelBindingProof) -> Result<()> {
    if proof.statement_digest != statement_digest(&public.statement)?
        || proof.layers.len() != public.layers.len()
        || proof.layers.len() != public.output_oracles.len()
    {
        return Err(Error::StatementMismatch);
    }
    let (bound_lo, bound_hi) = public.weight_bounds;
    let mut previous_oracle = public.input_oracle;
    for (index, layer_proof) in proof.layers.iter().enumerate() {
        let (rows, cols, shift, relu) = public.layers[index];
        verify_vector(&public.registry, &layer_proof.input, previous_oracle)?;
        let input = values_of(&layer_proof.input);
        if input.len() != cols {
            return Err(Error::ShapeMismatch);
        }
        let weight_oracle = layer_proof
            .weights
            .first()
            .map(|opening| opening.oracle)
            .ok_or(Error::ShapeMismatch)?;
        if layer_proof.weights.iter().any(|opening| opening.oracle != weight_oracle) {
            return Err(Error::StatementMismatch);
        }
        verify_vector(&public.registry, &layer_proof.weights, weight_oracle)?;
        let weights = values_of(&layer_proof.weights);
        if weights.len() != rows * cols
            || weights.iter().any(|weight| *weight < bound_lo || *weight > bound_hi)
        {
            return Err(Error::InvalidProof);
        }
        let bias_oracle = layer_proof.bias.first().map(|opening| opening.oracle).ok_or(Error::ShapeMismatch)?;
        verify_vector(&public.registry, &layer_proof.bias, bias_oracle)?;
        let bias = values_of(&layer_proof.bias);
        let raw_oracle = layer_proof.raw.first().map(|opening| opening.oracle).ok_or(Error::ShapeMismatch)?;
        verify_vector(&public.registry, &layer_proof.raw, raw_oracle)?;
        let raw = wide_values(&layer_proof.raw);
        let remainder_oracle = layer_proof.remainders.first().map(|opening| opening.oracle).ok_or(Error::ShapeMismatch)?;
        verify_vector(&public.registry, &layer_proof.remainders, remainder_oracle)?;
        let remainders = values_of(&layer_proof.remainders);
        let pre_oracle = layer_proof.pre_activation.first().map(|opening| opening.oracle).ok_or(Error::ShapeMismatch)?;
        verify_vector(&public.registry, &layer_proof.pre_activation, pre_oracle)?;
        let pre = values_of(&layer_proof.pre_activation);
        let output_oracle = layer_proof.output.first().map(|opening| opening.oracle).ok_or(Error::ShapeMismatch)?;
        if output_oracle != public.output_oracles[index] {
            return Err(Error::StatementMismatch);
        }
        verify_vector(&public.registry, &layer_proof.output, output_oracle)?;
        let output = values_of(&layer_proof.output);
        let bit_oracle = layer_proof.bits.first().map(|opening| opening.oracle).ok_or(Error::ShapeMismatch)?;
        verify_vector(&public.registry, &layer_proof.bits, bit_oracle)?;
        let bits = values_of(&layer_proof.bits);
        if bias.len() != rows
            || raw.len() != rows
            || remainders.len() != rows
            || pre.len() != rows
            || output.len() != rows
            || bits.len() != rows
        {
            return Err(Error::ShapeMismatch);
        }
        let scale = 1i128
            .checked_shl(shift)
            .ok_or(Error::BoundViolation)?;
        for row in 0..rows {
            let mut sum = bias[row] as i128;
            for column in 0..cols {
                sum += weights[row * cols + column] as i128 * input[column] as i128;
            }
            if sum != raw[row]
                || pre[row] as i128 * scale + remainders[row] as i128 != raw[row]
                || remainders[row] < 0
                || remainders[row] as i128 >= scale
            {
                return Err(Error::InvalidProof);
            }
            let bit = bits[row];
            if bit != 0 && bit != 1 {
                return Err(Error::InvalidProof);
            }
            if relu {
                if output[row] != bit * pre[row] {
                    return Err(Error::InvalidProof);
                }
                if bit == 0 && pre[row] > 0 {
                    return Err(Error::InvalidProof);
                }
                if bit == 1 && pre[row] <= 0 {
                    return Err(Error::InvalidProof);
                }
            } else if output[row] != pre[row] {
                return Err(Error::InvalidProof);
            }
        }
        previous_oracle = output_oracle;
    }
    if previous_oracle != public.logits_oracle {
        return Err(Error::StatementMismatch);
    }
    let logits = values_of(&proof.layers.last().ok_or(Error::IncompleteArtifact)?.output);
    if logits.is_empty() {
        return Err(Error::ShapeMismatch);
    }
    let mut best = 0usize;
    for index in 1..logits.len() {
        if logits[index] > logits[best] {
            best = index;
        }
    }
    if best as u32 != public.expected_token
        || public.statement.output_tokens.first().copied() != Some(public.expected_token)
    {
        return Err(Error::InvalidProof);
    }
    Ok(())
}
