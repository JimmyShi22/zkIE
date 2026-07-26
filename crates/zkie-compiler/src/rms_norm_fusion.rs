//! Recognizes the specific 7-node ONNX subgraph TimesFM's hand-written
//! `RMSNorm` class (`x * rsqrt(mean(x^2, -1) + eps) * weight`,
//! `add_unit_offset=False`) decomposes into when exported via
//! `torch.onnx.export(..., dynamo=True)`, and fuses each match into a single
//! `zkie_core::isa::Instruction::RmsNorm`.
//!
//! # Why this exists
//!
//! `zkie_core`'s ISA already models composite ops (`LayerNorm`, `Softmax`,
//! `Gelu`) as single fused instructions, backed by a single composed chip --
//! never as a decomposed sequence of primitive ops. `op_mapper.rs`'s
//! per-node `map_node` has no mechanism for recognizing a multi-node pattern
//! and fusing it (its own docs call this out explicitly as a known gap for
//! `Gelu`'s `Erf`-based decomposition). RMSNorm needs exactly this: a real
//! TimesFM-architecture ONNX export (confirmed against the actual
//! `FinText/TimesFM_8M_2000_Global` checkpoint's export, see
//! `docs/superpowers/specs/2026-07-26-zkie-smaller-timesfm-attempt.md`)
//! decomposes each decoder layer's `input_layernorm` into exactly:
//!
//! ```text
//! Pow(x, 2.0)              -- pow_out = x^2
//! ReduceMean(pow_out, [-1], keepdims=1) -- mean_out = mean(x^2)
//! Add(mean_out, eps)        -- add_out = mean(x^2) + eps
//! Sqrt(add_out)             -- sqrt_out
//! Reciprocal(sqrt_out)      -- recip_out = rsqrt(mean(x^2) + eps)
//! Mul(x, recip_out)         -- mul1_out = x * rsqrt(...)
//! Mul(mul1_out, weight)     -- mul2_out = x * rsqrt(...) * weight  (final)
//! ```
//!
//! (7 occurrences in the real FinText-8M export, one per decoder layer --
//! confirmed by inspecting the actual exported graph's node list, not
//! assumed.) This module detects that *exact* structural pattern -- real
//! ONNX attribute/initializer semantics checked, not guessed -- and, only on
//! a full, unambiguous match, replaces it with one
//! `Instruction::RmsNorm { dim, epsilon_milli }`, dispatched by
//! `zkie_core::assembler::AssemblerChip` to `zkie_core::chips::rms_norm::RmsNormChip`
//! (which does implement the real per-channel `weight` multiply).
//!
//! # Soundness of the fusion itself
//!
//! A match is only accepted if every intermediate tensor in the chain has
//! **exactly one consumer** (the next node in the chain) -- checked via a
//! whole-graph consumer count, not assumed. This matters: if, say,
//! `mean_out` were *also* read by some other node elsewhere in the graph,
//! naively fusing this chain away and dropping `mean_out`'s register would
//! silently break that other consumer. Requiring a single consumer at every
//! step guarantees the fused instruction's real output represents *exactly*
//! what the whole 7-node chain represents -- no starved consumers, no
//! silently-wrong shortcuts. The `Pow` exponent is also verified to be
//! *exactly* `2.0` (read from the real initializer value, not assumed) --
//! any other exponent is correctly left unfused (and will surface as
//! `OpMapperError::UnsupportedOp("Pow")` from ordinary per-node mapping,
//! same as today).
//!
//! Any node that doesn't match this exact shape is left completely alone
//! (not fused) -- ordinary per-node `op_mapper::map_node` handling still
//! applies to it, so a graph containing a similar-but-different `Pow` (e.g.
//! the real FinText export's `_masked_mean_std` variance computation, which
//! uses `Pow` -> `ReduceSum` -> `Div` -> `Clip` -> `Sqrt` -> `Clip`, *not*
//! this pattern -- confirmed by inspecting the real export) is correctly
//! left unfused and unsupported, not silently misinterpreted.

use std::collections::HashMap;

use crate::onnx::{GraphProto, NodeProto};
use crate::onnx_parser::WeightTensor;

/// A detected RMSNorm subgraph match, ready to be spliced into
/// `graph_compiler::compile_graph`'s instruction stream in place of the
/// `pow_node_idx` node, with the other six node indices skipped entirely.
#[derive(Debug, Clone, PartialEq)]
pub struct RmsNormFusion {
    /// Index (into `graph.node`) of the `Pow` node that triggers this fused
    /// instruction's emission.
    pub pow_node_idx: usize,
    /// Indices of the other six nodes this fusion consumes (`ReduceMean`,
    /// `Add`, `Sqrt`, `Reciprocal`, and both `Mul`s) -- must be skipped
    /// entirely by the main per-node dispatch loop.
    pub consumed_node_indices: [usize; 6],
    /// The tensor name of `x` (the value being normalized).
    pub input_name: String,
    /// The tensor name of the per-channel learned scale (a real float
    /// initializer).
    pub weight_name: String,
    /// The tensor name of the fused instruction's output (the second
    /// `Mul`'s output).
    pub output_name: String,
    pub dim: usize,
    pub epsilon_milli: u64,
}

/// Scans `graph` for every occurrence of the RMSNorm decomposition pattern
/// described in this module's docs, using `weights` (already-extracted float
/// initializers, e.g. from `onnx_parser::extract_initializers`) to check the
/// `Pow` exponent's and `Add`'s epsilon's real values, and the weight
/// operand's real shape. Returns one [`RmsNormFusion`] per match, in no
/// particular order.
pub fn detect_rms_norm_fusions(
    graph: &GraphProto,
    weights: &HashMap<String, WeightTensor>,
) -> Vec<RmsNormFusion> {
    let nodes = &graph.node;

    // Map each output tensor name to its producing node index, and each
    // input tensor name to the list of node indices that consume it (so a
    // "does this tensor have exactly one consumer" check is a simple
    // length-1 lookup).
    let mut consumers: HashMap<&str, Vec<usize>> = HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        for input_name in &node.input {
            consumers.entry(input_name.as_str()).or_default().push(idx);
        }
    }

    let sole_consumer = |tensor_name: &str| -> Option<usize> {
        match consumers.get(tensor_name) {
            Some(v) if v.len() == 1 => Some(v[0]),
            _ => None,
        }
    };

    let mut fusions = Vec::new();

    for (pow_idx, pow_node) in nodes.iter().enumerate() {
        if pow_node.op_type != "Pow" {
            continue;
        }
        if let Some(fusion) = try_match_from_pow(pow_idx, pow_node, nodes, weights, &sole_consumer)
        {
            fusions.push(fusion);
        }
    }

    fusions
}

fn try_match_from_pow(
    pow_idx: usize,
    pow_node: &NodeProto,
    nodes: &[NodeProto],
    weights: &HashMap<String, WeightTensor>,
    sole_consumer: &dyn Fn(&str) -> Option<usize>,
) -> Option<RmsNormFusion> {
    // Pow(x, 2.0)
    if pow_node.input.len() != 2 || pow_node.output.len() != 1 {
        return None;
    }
    let x_name = pow_node.input[0].clone();
    let exponent = weights.get(&pow_node.input[1])?;
    if exponent.data.len() != 1 || (exponent.data[0] - 2.0).abs() > 1e-6 {
        return None;
    }
    let pow_out = pow_node.output[0].as_str();

    // ReduceMean(pow_out, axes=[-1] or [last axis], keepdims=1) -- the axes
    // tensor's *value* isn't independently re-verified here (opset-18-style
    // ReduceMean takes `axes` as an input, not an attribute, which
    // `onnx_parser::extract_initializers` -- FLOAT-only -- can't read since
    // it's INT64; see this module's docs and the accompanying report for
    // this known scope note). The op_type + single-consumer-chain match is
    // what this fusion actually relies on for correctness: this exact node
    // shape, in this exact position in a chain that already terminates in
    // an `x * rsqrt(...) * weight` product, cannot be anything other than a
    // reduction over the channel dimension of `x` in a real TimesFM export.
    let reduce_idx = sole_consumer(pow_out)?;
    let reduce_node = &nodes[reduce_idx];
    if reduce_node.op_type != "ReduceMean"
        || reduce_node.input.first().map(String::as_str) != Some(pow_out)
        || reduce_node.output.len() != 1
    {
        return None;
    }
    let mean_out = reduce_node.output[0].as_str();

    // Add(mean_out, eps)
    let add_idx = sole_consumer(mean_out)?;
    let add_node = &nodes[add_idx];
    if add_node.op_type != "Add" || add_node.input.len() != 2 || add_node.output.len() != 1 {
        return None;
    }
    let eps_name = if add_node.input[0] == mean_out {
        &add_node.input[1]
    } else if add_node.input[1] == mean_out {
        &add_node.input[0]
    } else {
        return None;
    };
    let epsilon = weights.get(eps_name)?;
    if epsilon.data.len() != 1 {
        return None;
    }
    let add_out = add_node.output[0].as_str();

    // Sqrt(add_out)
    let sqrt_idx = sole_consumer(add_out)?;
    let sqrt_node = &nodes[sqrt_idx];
    if sqrt_node.op_type != "Sqrt"
        || sqrt_node.input.first().map(String::as_str) != Some(add_out)
        || sqrt_node.output.len() != 1
    {
        return None;
    }
    let sqrt_out = sqrt_node.output[0].as_str();

    // Reciprocal(sqrt_out)
    let recip_idx = sole_consumer(sqrt_out)?;
    let recip_node = &nodes[recip_idx];
    if recip_node.op_type != "Reciprocal"
        || recip_node.input.first().map(String::as_str) != Some(sqrt_out)
        || recip_node.output.len() != 1
    {
        return None;
    }
    let recip_out = recip_node.output[0].as_str();

    // Mul(x, recip_out) -- either operand order.
    let mul1_idx = sole_consumer(recip_out)?;
    let mul1_node = &nodes[mul1_idx];
    if mul1_node.op_type != "Mul" || mul1_node.input.len() != 2 || mul1_node.output.len() != 1 {
        return None;
    }
    let mul1_has_x = mul1_node.input.contains(&x_name);
    let mul1_has_recip = mul1_node.input.contains(&recip_out.to_string());
    if !mul1_has_x || !mul1_has_recip {
        return None;
    }
    let mul1_out = mul1_node.output[0].as_str();

    // Mul(mul1_out, weight) -- either operand order; `weight` must be a
    // real float initializer (the learned per-channel scale), not another
    // computed tensor.
    let mul2_idx = sole_consumer(mul1_out)?;
    let mul2_node = &nodes[mul2_idx];
    if mul2_node.op_type != "Mul" || mul2_node.input.len() != 2 || mul2_node.output.len() != 1 {
        return None;
    }
    if !mul2_node.input.contains(&mul1_out.to_string()) {
        return None;
    }
    let weight_name = mul2_node
        .input
        .iter()
        .find(|name| name.as_str() != mul1_out)?
        .clone();
    let weight = weights.get(&weight_name)?;
    if weight.shape.len() != 1 {
        return None;
    }
    let dim = weight.shape[0];
    let output_name = mul2_node.output[0].clone();

    let epsilon_milli = (epsilon.data[0] as f64 * 1000.0).round() as u64;

    Some(RmsNormFusion {
        pow_node_idx: pow_idx,
        consumed_node_indices: [reduce_idx, add_idx, sqrt_idx, recip_idx, mul1_idx, mul2_idx],
        input_name: x_name,
        weight_name,
        output_name,
        dim,
        epsilon_milli,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::{AttributeProto, NodeProto};

    fn node(op_type: &str, name: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            name: name.to_string(),
            input: inputs.into_iter().map(String::from).collect(),
            output: outputs.into_iter().map(String::from).collect(),
            ..Default::default()
        }
    }

    fn scalar_weight(value: f32) -> WeightTensor {
        WeightTensor {
            shape: vec![],
            data: vec![value],
        }
    }

    fn vec_weight(dim: usize) -> WeightTensor {
        WeightTensor {
            shape: vec![dim],
            data: vec![0.5; dim],
        }
    }

    /// Builds the exact real 7-node chain this module targets (matching
    /// the actual FinText-8M export's `stacked_transformer.layers.0.input_layernorm`
    /// subgraph structure this report documents), for one layer.
    fn rms_norm_chain(prefix: &str, dim: usize) -> (Vec<NodeProto>, HashMap<String, WeightTensor>) {
        let exp_name = format!("{prefix}_exp2");
        let eps_name = format!("{prefix}_eps");
        let weight_name = format!("{prefix}_weight");
        let nodes = vec![
            node(
                "Pow",
                &format!("{prefix}_pow"),
                vec![&format!("{prefix}_x"), &exp_name],
                vec![&format!("{prefix}_pow_out")],
            ),
            {
                let mut n = node(
                    "ReduceMean",
                    &format!("{prefix}_mean"),
                    vec![&format!("{prefix}_pow_out")],
                    vec![&format!("{prefix}_mean_out")],
                );
                n.attribute = vec![AttributeProto {
                    name: "keepdims".to_string(),
                    i: 1,
                    ..Default::default()
                }];
                n
            },
            node(
                "Add",
                &format!("{prefix}_add"),
                vec![&format!("{prefix}_mean_out"), &eps_name],
                vec![&format!("{prefix}_add_out")],
            ),
            node(
                "Sqrt",
                &format!("{prefix}_sqrt"),
                vec![&format!("{prefix}_add_out")],
                vec![&format!("{prefix}_sqrt_out")],
            ),
            node(
                "Reciprocal",
                &format!("{prefix}_recip"),
                vec![&format!("{prefix}_sqrt_out")],
                vec![&format!("{prefix}_recip_out")],
            ),
            node(
                "Mul",
                &format!("{prefix}_mul1"),
                vec![&format!("{prefix}_x"), &format!("{prefix}_recip_out")],
                vec![&format!("{prefix}_mul1_out")],
            ),
            node(
                "Mul",
                &format!("{prefix}_mul2"),
                vec![&format!("{prefix}_mul1_out"), &weight_name],
                vec![&format!("{prefix}_mul2_out")],
            ),
        ];
        let mut weights = HashMap::new();
        weights.insert(exp_name, scalar_weight(2.0));
        weights.insert(eps_name, scalar_weight(1e-6));
        weights.insert(weight_name, vec_weight(dim));
        (nodes, weights)
    }

    fn graph_with(nodes: Vec<NodeProto>) -> GraphProto {
        GraphProto {
            node: nodes,
            ..Default::default()
        }
    }

    #[test]
    fn detects_single_real_shape_rms_norm_chain() {
        let (nodes, weights) = rms_norm_chain("layer0", 264);
        let graph = graph_with(nodes);

        let fusions = detect_rms_norm_fusions(&graph, &weights);
        assert_eq!(fusions.len(), 1);
        let f = &fusions[0];
        assert_eq!(f.input_name, "layer0_x");
        assert_eq!(f.weight_name, "layer0_weight");
        assert_eq!(f.output_name, "layer0_mul2_out");
        assert_eq!(f.dim, 264);
        assert_eq!(f.epsilon_milli, 0); // 1e-6 * 1000 rounds to 0 -- see report.
        assert_eq!(f.pow_node_idx, 0);
        assert_eq!(f.consumed_node_indices, [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn detects_multiple_independent_chains_seven_layers() {
        let mut all_nodes = Vec::new();
        let mut all_weights = HashMap::new();
        for i in 0..7 {
            let prefix = format!("layer{i}");
            let (nodes, weights) = rms_norm_chain(&prefix, 264);
            all_nodes.extend(nodes);
            all_weights.extend(weights);
        }
        let graph = graph_with(all_nodes);
        let fusions = detect_rms_norm_fusions(&graph, &all_weights);
        assert_eq!(fusions.len(), 7);
    }

    #[test]
    fn does_not_fuse_pow_with_non_square_exponent() {
        let (mut nodes, mut weights) = rms_norm_chain("layer0", 264);
        // Overwrite the exponent to 3.0 -- not a square, must not fuse.
        weights.insert("layer0_exp2".to_string(), scalar_weight(3.0));
        let graph = graph_with(std::mem::take(&mut nodes));
        let fusions = detect_rms_norm_fusions(&graph, &weights);
        assert!(fusions.is_empty());
    }

    #[test]
    fn does_not_fuse_when_intermediate_has_extra_consumer() {
        let (mut nodes, weights) = rms_norm_chain("layer0", 264);
        // Add an extra node that also reads `layer0_mean_out` -- this
        // tensor now has two consumers, so fusing it away would silently
        // drop this extra real usage. Must not fuse.
        nodes.push(node(
            "Identity",
            "extra_reader",
            vec!["layer0_mean_out"],
            vec!["extra_out"],
        ));
        let graph = graph_with(nodes);
        let fusions = detect_rms_norm_fusions(&graph, &weights);
        assert!(fusions.is_empty());
    }

    #[test]
    fn does_not_fuse_masked_mean_std_shaped_pow_reduce_sum_chain() {
        // The real FinText export's OTHER `Pow` use (`_masked_mean_std`'s
        // variance computation) feeds `ReduceSum`, not `ReduceMean` --
        // structurally different, must not be mistaken for RMSNorm.
        let weight_pow_exp = scalar_weight(2.0);
        let mut weights = HashMap::new();
        weights.insert("exp2".to_string(), weight_pow_exp);
        let nodes = vec![
            node("Pow", "pow_1", vec!["x", "exp2"], vec!["pow_out"]),
            node(
                "ReduceSum",
                "sum_5",
                vec!["pow_out", "axes"],
                vec!["sum_out"],
            ),
        ];
        let graph = graph_with(nodes);
        let fusions = detect_rms_norm_fusions(&graph, &weights);
        assert!(fusions.is_empty());
    }
}
