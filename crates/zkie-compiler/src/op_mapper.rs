//! Maps a curated subset of ONNX operators — the ops TimesFM and similar
//! small transformer/FFN models actually need — onto zkIE's `Instruction`
//! ISA (see `zkie_core::isa`). This module does not aim for general ONNX
//! operator coverage.
//!
//! # Scope limitations (deliberate)
//!
//! - Only the curated op subset below is supported; any other `op_type`
//!   produces [`OpMapperError::UnsupportedOp`] rather than a panic.
//! - `ReduceMean` only supports a single reduction axis, matching zkIE's
//!   `Instruction::Reduce { axis: usize }`. ONNX's `ReduceMean` (opset
//!   1-13 attribute form) allows reducing over multiple axes, and reduces
//!   over ALL axes by default when the `axes` attribute is absent. Both
//!   "more than one axis" and "axes absent" are rejected here with
//!   [`OpMapperError::MultipleReduceAxesUnsupported`] rather than silently
//!   picking one axis.
//! - `Gelu` maps directly to the fused ONNX opset-20+ `Gelu` operator.
//!   Models exported with older opsets often decompose GELU into an
//!   `Erf`-based subgraph (`x * 0.5 * (1 + Erf(x / sqrt(2)))`); this stage
//!   does not pattern-match multi-node subgraphs back into a single
//!   `Instruction::Gelu` — that is a known gap, out of scope here.
//! - `MatMul`/`Gemm` -> `Instruction::DotGeneral` shape extraction (`m`,
//!   `n`, `k`) requires both input tensors' shapes to already be known
//!   (e.g. via `onnx_parser::extract_value_shapes` and/or initializer
//!   shapes), threaded in via the `shapes` parameter of [`map_node`].
//!   `Gemm` per the ONNX spec only supports rank-2 inputs (no batch dims);
//!   `MatMul` batch dims are taken from input A's leading dimensions only
//!   — broadcasting compatibility between A's and B's batch dims is not
//!   validated here.
//! - `Gather` is mapped to `Instruction::EmbedLookup` assuming standard
//!   embedding-table semantics: the data operand (`node.input[0]`) is a
//!   rank>=2 tensor shaped `[table_size, embed_dim, ...]`. General
//!   `Gather` semantics (arbitrary axis, arbitrary rank) are out of scope.

use std::collections::HashMap;
use std::fmt;

use crate::onnx::{AttributeProto, NodeProto};
use zkie_core::isa::{EltwiseOp, Instruction, ReduceOp};

/// A node that only affects shape/buffer bookkeeping and emits no zkIE ISA
/// instruction (e.g. `Reshape`, `Transpose`, `Concat` — see the parent
/// design doc's "Reshape/Transpose: No instruction (metadata)" /
/// "Concat: Buffer splicing" notes).
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataOp {
    pub op_type: String,
    pub input: Vec<String>,
    pub output: Vec<String>,
}

/// The result of mapping a single ONNX `NodeProto`.
#[derive(Debug, Clone, PartialEq)]
pub enum MappedOp {
    /// This node emits a zkIE ISA instruction, plus the tensor names needed
    /// for later graph wiring.
    Instruction {
        instruction: Instruction,
        input: Vec<String>,
        output: Vec<String>,
    },
    /// This node is metadata-only bookkeeping; see [`MetadataOp`].
    Metadata(MetadataOp),
}

/// Errors that can occur while mapping an ONNX node to a [`MappedOp`].
#[derive(Debug, Clone, PartialEq)]
pub enum OpMapperError {
    /// The node's `op_type` is not in this compiler's curated op subset.
    UnsupportedOp(String),
    /// A required input tensor's shape was not present in the `shapes`
    /// lookup passed to [`map_node`].
    MissingShape {
        op_type: String,
        tensor_name: String,
    },
    /// The node was missing an input tensor name at the given index.
    MissingInput { op_type: String, index: usize },
    /// A negative (relative-to-rank) ONNX axis attribute was encountered,
    /// but no shape was available to resolve it to an absolute index.
    NegativeAxisNeedsShape { op_type: String, axis: i64 },
    /// A resolved axis fell outside the tensor's rank.
    AxisOutOfRange {
        op_type: String,
        axis: i64,
        rank: usize,
    },
    /// A tensor had a rank this op mapping does not support (e.g. `Gemm`
    /// requires rank-2 inputs).
    UnsupportedRank {
        op_type: String,
        tensor_name: String,
        rank: usize,
    },
    /// The contracted (`k`) dimension computed from input A did not match
    /// the one computed from input B.
    KDimMismatch {
        op_type: String,
        k_from_a: usize,
        k_from_b: usize,
    },
    /// `ReduceMean`'s `axes` attribute did not specify exactly one axis
    /// (either it was absent — which per the ONNX spec means "reduce all
    /// axes" — or it explicitly listed more than one). See module docs.
    MultipleReduceAxesUnsupported { axes: Vec<i64> },
}

impl fmt::Display for OpMapperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpMapperError::UnsupportedOp(op_type) => {
                write!(f, "unsupported ONNX op_type '{op_type}'")
            }
            OpMapperError::MissingShape {
                op_type,
                tensor_name,
            } => write!(
                f,
                "{op_type}: no known shape for tensor '{tensor_name}'; static shape info is required to map this op"
            ),
            OpMapperError::MissingInput { op_type, index } => {
                write!(f, "{op_type}: missing required input at index {index}")
            }
            OpMapperError::NegativeAxisNeedsShape { op_type, axis } => write!(
                f,
                "{op_type}: negative axis {axis} requires a known tensor rank to resolve, but no shape was available"
            ),
            OpMapperError::AxisOutOfRange {
                op_type,
                axis,
                rank,
            } => write!(
                f,
                "{op_type}: axis {axis} is out of range for a tensor of rank {rank}"
            ),
            OpMapperError::UnsupportedRank {
                op_type,
                tensor_name,
                rank,
            } => write!(
                f,
                "{op_type}: tensor '{tensor_name}' has unsupported rank {rank}"
            ),
            OpMapperError::KDimMismatch {
                op_type,
                k_from_a,
                k_from_b,
            } => write!(
                f,
                "{op_type}: contracted dimension mismatch, k={k_from_a} from A but k={k_from_b} from B"
            ),
            OpMapperError::MultipleReduceAxesUnsupported { axes } => write!(
                f,
                "ReduceMean: only a single reduction axis is supported, got axes={axes:?}"
            ),
        }
    }
}

impl std::error::Error for OpMapperError {}

fn get_attr<'a>(node: &'a NodeProto, name: &str) -> Option<&'a AttributeProto> {
    node.attribute.iter().find(|a| a.name == name)
}

fn attr_i64(node: &NodeProto, name: &str, default: i64) -> i64 {
    get_attr(node, name).map(|a| a.i).unwrap_or(default)
}

fn attr_f32(node: &NodeProto, name: &str, default: f32) -> f32 {
    get_attr(node, name).map(|a| a.f).unwrap_or(default)
}

fn attr_ints(node: &NodeProto, name: &str) -> Option<Vec<i64>> {
    get_attr(node, name).map(|a| a.ints.clone())
}

fn instruction_op(node: &NodeProto, instruction: Instruction) -> MappedOp {
    MappedOp::Instruction {
        instruction,
        input: node.input.clone(),
        output: node.output.clone(),
    }
}

fn metadata_op(node: &NodeProto) -> MappedOp {
    MappedOp::Metadata(MetadataOp {
        op_type: node.op_type.clone(),
        input: node.input.clone(),
        output: node.output.clone(),
    })
}

fn required_input_name(node: &NodeProto, index: usize) -> Result<&str, OpMapperError> {
    node.input
        .get(index)
        .map(String::as_str)
        .ok_or_else(|| OpMapperError::MissingInput {
            op_type: node.op_type.clone(),
            index,
        })
}

fn required_shape<'a>(
    shapes: &'a HashMap<String, Vec<usize>>,
    op_type: &str,
    tensor_name: &str,
) -> Result<&'a [usize], OpMapperError> {
    shapes
        .get(tensor_name)
        .map(Vec::as_slice)
        .ok_or_else(|| OpMapperError::MissingShape {
            op_type: op_type.to_string(),
            tensor_name: tensor_name.to_string(),
        })
}

/// Resolves a possibly-negative ONNX axis attribute to a non-negative
/// index. A negative axis (counting from the back, per ONNX convention)
/// requires the tensor's rank — via `shape` — to resolve to an absolute
/// index.
fn resolve_axis(op_type: &str, axis: i64, shape: Option<&[usize]>) -> Result<usize, OpMapperError> {
    if axis >= 0 {
        return Ok(axis as usize);
    }
    let rank = shape
        .ok_or_else(|| OpMapperError::NegativeAxisNeedsShape {
            op_type: op_type.to_string(),
            axis,
        })?
        .len() as i64;
    let resolved = rank + axis;
    if resolved < 0 {
        return Err(OpMapperError::AxisOutOfRange {
            op_type: op_type.to_string(),
            axis,
            rank: rank as usize,
        });
    }
    Ok(resolved as usize)
}

/// Maps a single ONNX `NodeProto` to a [`MappedOp`]. `shapes` is a
/// name-keyed lookup of statically-known tensor shapes (see
/// `onnx_parser::extract_value_shapes` and initializer shapes), used by
/// ops whose ISA parameters depend on tensor shape (`MatMul`/`Gemm`,
/// `Softmax`, `LayerNormalization`, `ReduceMean`, `Gather`).
pub fn map_node(
    node: &NodeProto,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<MappedOp, OpMapperError> {
    match node.op_type.as_str() {
        "MatMul" => map_matmul_or_gemm(node, shapes, false, false, true),
        "Gemm" => {
            // ONNX Gemm: transA/transB are INT attributes, default 0 (false).
            let trans_a = attr_i64(node, "transA", 0) != 0;
            let trans_b = attr_i64(node, "transB", 0) != 0;
            map_matmul_or_gemm(node, shapes, trans_a, trans_b, false)
        }
        "Add" => Ok(instruction_op(
            node,
            Instruction::Eltwise { op: EltwiseOp::Add },
        )),
        "Mul" => Ok(instruction_op(
            node,
            Instruction::Eltwise { op: EltwiseOp::Mul },
        )),
        "Relu" => Ok(instruction_op(
            node,
            Instruction::Eltwise {
                op: EltwiseOp::Relu,
            },
        )),
        "Softmax" => map_softmax(node, shapes),
        "Gelu" => Ok(instruction_op(node, Instruction::Gelu)),
        "LayerNormalization" => map_layer_norm(node, shapes),
        "ReduceMean" => map_reduce_mean(node, shapes),
        "Gather" => map_gather(node, shapes),
        "Reshape" | "Transpose" | "Squeeze" | "Unsqueeze" | "Concat" => Ok(metadata_op(node)),
        other => Err(OpMapperError::UnsupportedOp(other.to_string())),
    }
}

/// Returns `(rows, cols)` of the EFFECTIVE (post-transpose) matrix formed
/// by the last two dims of `shape` — i.e. the `(m, k)` pair for operand A
/// or the `(k, n)` pair for operand B, after applying ONNX's
/// transpose-last-two-dims semantics.
fn last_two_dims(shape: &[usize], transposed: bool) -> (usize, usize) {
    let len = shape.len();
    let (d0, d1) = (shape[len - 2], shape[len - 1]);
    if transposed {
        (d1, d0)
    } else {
        (d0, d1)
    }
}

fn map_matmul_or_gemm(
    node: &NodeProto,
    shapes: &HashMap<String, Vec<usize>>,
    trans_a: bool,
    trans_b: bool,
    allow_batch_dims: bool,
) -> Result<MappedOp, OpMapperError> {
    let op_type = node.op_type.as_str();
    let a_name = required_input_name(node, 0)?;
    let b_name = required_input_name(node, 1)?;
    let a_shape = required_shape(shapes, op_type, a_name)?;
    let b_shape = required_shape(shapes, op_type, b_name)?;

    if a_shape.len() < 2 {
        return Err(OpMapperError::UnsupportedRank {
            op_type: op_type.to_string(),
            tensor_name: a_name.to_string(),
            rank: a_shape.len(),
        });
    }
    if b_shape.len() < 2 {
        return Err(OpMapperError::UnsupportedRank {
            op_type: op_type.to_string(),
            tensor_name: b_name.to_string(),
            rank: b_shape.len(),
        });
    }
    if !allow_batch_dims && a_shape.len() != 2 {
        return Err(OpMapperError::UnsupportedRank {
            op_type: op_type.to_string(),
            tensor_name: a_name.to_string(),
            rank: a_shape.len(),
        });
    }
    if !allow_batch_dims && b_shape.len() != 2 {
        return Err(OpMapperError::UnsupportedRank {
            op_type: op_type.to_string(),
            tensor_name: b_name.to_string(),
            rank: b_shape.len(),
        });
    }

    let (m, k_from_a) = last_two_dims(a_shape, trans_a);
    let (k_from_b, n) = last_two_dims(b_shape, trans_b);
    if k_from_a != k_from_b {
        return Err(OpMapperError::KDimMismatch {
            op_type: op_type.to_string(),
            k_from_a,
            k_from_b,
        });
    }

    // Batch dims (MatMul only — Gemm is always rank-2): A's leading dims.
    // Broadcasting compatibility with B's leading dims is not validated;
    // see module docs.
    let batch_dims = a_shape[..a_shape.len() - 2].to_vec();

    Ok(instruction_op(
        node,
        Instruction::DotGeneral {
            m,
            n,
            k: k_from_a,
            batch_dims,
            trans_a,
            trans_b,
        },
    ))
}

fn map_softmax(
    node: &NodeProto,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<MappedOp, OpMapperError> {
    let op_type = node.op_type.as_str();
    // ONNX Softmax opset 13+ default axis is -1 (last axis); this compiler
    // targets opset 13+ semantics (opset <13 defaulted to axis=1 instead).
    let axis_attr = attr_i64(node, "axis", -1);
    let input_name = required_input_name(node, 0)?;
    let shape = required_shape(shapes, op_type, input_name)?;
    let axis = resolve_axis(op_type, axis_attr, Some(shape))?;
    let axis_dim = *shape
        .get(axis)
        .ok_or_else(|| OpMapperError::AxisOutOfRange {
            op_type: op_type.to_string(),
            axis: axis_attr,
            rank: shape.len(),
        })?;
    Ok(instruction_op(node, Instruction::Softmax { axis_dim }))
}

fn map_layer_norm(
    node: &NodeProto,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<MappedOp, OpMapperError> {
    let op_type = node.op_type.as_str();
    // ONNX LayerNormalization defaults: axis = -1, epsilon = 1e-5.
    let axis_attr = attr_i64(node, "axis", -1);
    let epsilon = attr_f32(node, "epsilon", 1e-5);
    let input_name = required_input_name(node, 0)?;
    let shape = required_shape(shapes, op_type, input_name)?;
    let axis = resolve_axis(op_type, axis_attr, Some(shape))?;
    let dim = *shape
        .get(axis)
        .ok_or_else(|| OpMapperError::AxisOutOfRange {
            op_type: op_type.to_string(),
            axis: axis_attr,
            rank: shape.len(),
        })?;
    let epsilon_milli = (epsilon as f64 * 1000.0).round() as u64;
    Ok(instruction_op(
        node,
        Instruction::LayerNorm { dim, epsilon_milli },
    ))
}

fn map_reduce_mean(
    node: &NodeProto,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<MappedOp, OpMapperError> {
    let op_type = node.op_type.as_str();
    // ONNX ReduceMean (opset 1-13 attribute form): `axes` is an INTS
    // attribute; if absent, ALL axes are reduced per spec. zkIE's Reduce
    // instruction only supports a single axis, so both cases are rejected
    // here rather than silently picking one — see module docs.
    let axes = attr_ints(node, "axes").unwrap_or_default();
    if axes.len() != 1 {
        return Err(OpMapperError::MultipleReduceAxesUnsupported { axes });
    }
    let input_name = required_input_name(node, 0)?;
    let shape = shapes.get(input_name).map(Vec::as_slice);
    let axis = resolve_axis(op_type, axes[0], shape)?;
    Ok(instruction_op(
        node,
        Instruction::Reduce {
            op: ReduceOp::Mean,
            axis,
        },
    ))
}

fn map_gather(
    node: &NodeProto,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<MappedOp, OpMapperError> {
    let op_type = node.op_type.as_str();
    let data_name = required_input_name(node, 0)?;
    let shape = required_shape(shapes, op_type, data_name)?;
    if shape.len() < 2 {
        return Err(OpMapperError::UnsupportedRank {
            op_type: op_type.to_string(),
            tensor_name: data_name.to_string(),
            rank: shape.len(),
        });
    }
    let table_size = shape[0];
    let embed_dim = shape[1..].iter().product();
    Ok(instruction_op(
        node,
        Instruction::EmbedLookup {
            table_size,
            embed_dim,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::attribute_proto::AttributeType;

    fn node(op_type: &str, input: Vec<&str>, output: Vec<&str>) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            input: input.into_iter().map(String::from).collect(),
            output: output.into_iter().map(String::from).collect(),
            ..Default::default()
        }
    }

    fn int_attr(name: &str, value: i64) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            i: value,
            r#type: AttributeType::Int as i32,
            ..Default::default()
        }
    }

    fn float_attr(name: &str, value: f32) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            f: value,
            r#type: AttributeType::Float as i32,
            ..Default::default()
        }
    }

    fn ints_attr(name: &str, values: Vec<i64>) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            ints: values,
            r#type: AttributeType::Ints as i32,
            ..Default::default()
        }
    }

    fn shapes(pairs: &[(&str, Vec<usize>)]) -> HashMap<String, Vec<usize>> {
        pairs
            .iter()
            .map(|(name, shape)| (name.to_string(), shape.clone()))
            .collect()
    }

    #[test]
    fn maps_matmul_to_dot_general() {
        let n = node("MatMul", vec!["a", "b"], vec!["c"]);
        let shapes = shapes(&[("a", vec![4, 8]), ("b", vec![8, 16])]);

        let mapped = map_node(&n, &shapes).unwrap();
        match mapped {
            MappedOp::Instruction {
                instruction,
                input,
                output,
            } => {
                assert_eq!(
                    instruction,
                    Instruction::DotGeneral {
                        m: 4,
                        n: 16,
                        k: 8,
                        batch_dims: vec![],
                        trans_a: false,
                        trans_b: false,
                    }
                );
                assert_eq!(input, vec!["a", "b"]);
                assert_eq!(output, vec!["c"]);
            }
            other => panic!("expected Instruction, got {other:?}"),
        }
    }

    #[test]
    fn maps_gemm_with_transposed_a_to_dot_general() {
        let mut n = node("Gemm", vec!["a", "b"], vec!["c"]);
        n.attribute = vec![int_attr("transA", 1)];
        // A is stored as [k, m] since transA=1 will transpose it.
        let shapes = shapes(&[("a", vec![8, 4]), ("b", vec![8, 16])]);

        let mapped = map_node(&n, &shapes).unwrap();
        match mapped {
            MappedOp::Instruction { instruction, .. } => {
                assert_eq!(
                    instruction,
                    Instruction::DotGeneral {
                        m: 4,
                        n: 16,
                        k: 8,
                        batch_dims: vec![],
                        trans_a: true,
                        trans_b: false,
                    }
                );
            }
            other => panic!("expected Instruction, got {other:?}"),
        }
    }

    #[test]
    fn gemm_rejects_batched_rank() {
        let n = node("Gemm", vec!["a", "b"], vec!["c"]);
        let shapes = shapes(&[("a", vec![2, 4, 8]), ("b", vec![8, 16])]);
        let result = map_node(&n, &shapes);
        assert!(matches!(result, Err(OpMapperError::UnsupportedRank { .. })));
    }

    #[test]
    fn maps_add_mul_relu_to_eltwise() {
        let cases = [
            ("Add", EltwiseOp::Add),
            ("Mul", EltwiseOp::Mul),
            ("Relu", EltwiseOp::Relu),
        ];
        for (op_type, expected) in cases {
            let n = node(op_type, vec!["x", "y"], vec!["z"]);
            let mapped = map_node(&n, &HashMap::new()).unwrap();
            match mapped {
                MappedOp::Instruction { instruction, .. } => {
                    assert_eq!(instruction, Instruction::Eltwise { op: expected });
                }
                other => panic!("expected Instruction, got {other:?}"),
            }
        }
    }

    #[test]
    fn maps_softmax_with_default_axis() {
        let n = node("Softmax", vec!["x"], vec!["y"]);
        let shapes = shapes(&[("x", vec![2, 8, 32])]);
        let mapped = map_node(&n, &shapes).unwrap();
        match mapped {
            MappedOp::Instruction { instruction, .. } => {
                assert_eq!(instruction, Instruction::Softmax { axis_dim: 32 });
            }
            other => panic!("expected Instruction, got {other:?}"),
        }
    }

    #[test]
    fn maps_softmax_with_explicit_axis() {
        let mut n = node("Softmax", vec!["x"], vec!["y"]);
        n.attribute = vec![int_attr("axis", 1)];
        let shapes = shapes(&[("x", vec![2, 8, 32])]);
        let mapped = map_node(&n, &shapes).unwrap();
        match mapped {
            MappedOp::Instruction { instruction, .. } => {
                assert_eq!(instruction, Instruction::Softmax { axis_dim: 8 });
            }
            other => panic!("expected Instruction, got {other:?}"),
        }
    }

    #[test]
    fn maps_gelu_directly() {
        let n = node("Gelu", vec!["x"], vec!["y"]);
        let mapped = map_node(&n, &HashMap::new()).unwrap();
        match mapped {
            MappedOp::Instruction { instruction, .. } => {
                assert_eq!(instruction, Instruction::Gelu);
            }
            other => panic!("expected Instruction, got {other:?}"),
        }
    }

    #[test]
    fn maps_layer_normalization_with_defaults() {
        let n = node("LayerNormalization", vec!["x"], vec!["y"]);
        let shapes = shapes(&[("x", vec![2, 8, 64])]);
        let mapped = map_node(&n, &shapes).unwrap();
        match mapped {
            MappedOp::Instruction { instruction, .. } => {
                assert_eq!(
                    instruction,
                    Instruction::LayerNorm {
                        dim: 64,
                        epsilon_milli: (1e-5_f64 * 1000.0).round() as u64,
                    }
                );
            }
            other => panic!("expected Instruction, got {other:?}"),
        }
    }

    #[test]
    fn maps_layer_normalization_with_explicit_epsilon_and_axis() {
        let mut n = node("LayerNormalization", vec!["x"], vec!["y"]);
        n.attribute = vec![float_attr("epsilon", 0.001), int_attr("axis", 1)];
        let shapes = shapes(&[("x", vec![2, 8, 64])]);
        let mapped = map_node(&n, &shapes).unwrap();
        match mapped {
            MappedOp::Instruction { instruction, .. } => {
                assert_eq!(
                    instruction,
                    Instruction::LayerNorm {
                        dim: 8,
                        epsilon_milli: 1,
                    }
                );
            }
            other => panic!("expected Instruction, got {other:?}"),
        }
    }

    #[test]
    fn maps_reduce_mean_single_axis() {
        let mut n = node("ReduceMean", vec!["x"], vec!["y"]);
        n.attribute = vec![ints_attr("axes", vec![1])];
        let mapped = map_node(&n, &HashMap::new()).unwrap();
        match mapped {
            MappedOp::Instruction { instruction, .. } => {
                assert_eq!(
                    instruction,
                    Instruction::Reduce {
                        op: ReduceOp::Mean,
                        axis: 1,
                    }
                );
            }
            other => panic!("expected Instruction, got {other:?}"),
        }
    }

    #[test]
    fn reduce_mean_rejects_multiple_axes() {
        let mut n = node("ReduceMean", vec!["x"], vec!["y"]);
        n.attribute = vec![ints_attr("axes", vec![1, 2])];
        let result = map_node(&n, &HashMap::new());
        match result {
            Err(OpMapperError::MultipleReduceAxesUnsupported { axes }) => {
                assert_eq!(axes, vec![1, 2]);
            }
            other => panic!("expected MultipleReduceAxesUnsupported, got {other:?}"),
        }
    }

    #[test]
    fn reduce_mean_rejects_absent_axes_attribute() {
        // Absent `axes` means "reduce all axes" per the ONNX spec, which
        // this compiler also cannot represent with a single `axis: usize`.
        let n = node("ReduceMean", vec!["x"], vec!["y"]);
        let result = map_node(&n, &HashMap::new());
        match result {
            Err(OpMapperError::MultipleReduceAxesUnsupported { axes }) => {
                assert!(axes.is_empty());
            }
            other => panic!("expected MultipleReduceAxesUnsupported, got {other:?}"),
        }
    }

    #[test]
    fn maps_gather_to_embed_lookup() {
        let n = node("Gather", vec!["table", "indices"], vec!["out"]);
        let shapes = shapes(&[("table", vec![1000, 128])]);
        let mapped = map_node(&n, &shapes).unwrap();
        match mapped {
            MappedOp::Instruction { instruction, .. } => {
                assert_eq!(
                    instruction,
                    Instruction::EmbedLookup {
                        table_size: 1000,
                        embed_dim: 128,
                    }
                );
            }
            other => panic!("expected Instruction, got {other:?}"),
        }
    }

    #[test]
    fn maps_metadata_only_ops() {
        for op_type in ["Reshape", "Transpose", "Squeeze", "Unsqueeze", "Concat"] {
            let n = node(op_type, vec!["x"], vec!["y"]);
            let mapped = map_node(&n, &HashMap::new()).unwrap();
            match mapped {
                MappedOp::Metadata(meta) => {
                    assert_eq!(meta.op_type, op_type);
                    assert_eq!(meta.input, vec!["x"]);
                    assert_eq!(meta.output, vec!["y"]);
                }
                other => panic!("expected Metadata, got {other:?}"),
            }
        }
    }

    #[test]
    fn rejects_unsupported_op() {
        let n = node("Conv", vec!["x"], vec!["y"]);
        let result = map_node(&n, &HashMap::new());
        match result {
            Err(OpMapperError::UnsupportedOp(op_type)) => assert_eq!(op_type, "Conv"),
            other => panic!("expected UnsupportedOp, got {other:?}"),
        }
    }
}
