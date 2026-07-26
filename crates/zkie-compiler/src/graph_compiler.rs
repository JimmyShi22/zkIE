//! Ties `onnx_parser` and `op_mapper` together into a fully-wired,
//! topologically-ordered zkIE program.
//!
//! This is the graph-compilation stage: it walks an ONNX `GraphProto` in
//! dependency order, maps each node to either a zkIE ISA instruction or a
//! metadata-only bookkeeping op (see `op_mapper`), and resolves every ONNX
//! tensor name to a [`Register`] describing where its value comes from.
//!
//! # Design notes
//!
//! - **Topological order**: nodes are ordered via Kahn's algorithm (BFS over
//!   zero-in-degree nodes using an explicit `VecDeque` worklist), not
//!   recursion, so this remains robust on larger graphs and can detect
//!   cycles without risking a stack overflow or infinite loop.
//! - **Metadata passthrough**: `Reshape`/`Transpose`/`Squeeze`/`Unsqueeze`/
//!   `Concat` never emit a `CompiledInstruction`. Instead, the metadata
//!   node's output tensor name is aliased directly to the SAME `Register`
//!   its (first) input tensor name already resolves to. This means multiple
//!   distinct ONNX tensor names can end up pointing at the same underlying
//!   `Register` — e.g. `Reshape(x) -> x_reshaped` binds `"x_reshaped"` to
//!   whatever register `"x"` was already bound to, with no new value
//!   produced.
//! - **Weights stay as `f32` until requested**: `CompiledProgram::weights`
//!   holds the raw `WeightTensor`s from `onnx_parser::extract_initializers`.
//!   We deliberately do NOT eagerly convert every weight to `I18` here: a
//!   real model may have weights outside `I18`'s representable range that
//!   are never actually consumed by the supported instruction subset, and
//!   eagerly converting all of them would make this compiler fail on graphs
//!   it could otherwise successfully compile. Callers convert on demand via
//!   [`CompiledProgram::weight_as_i18`].

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use crate::onnx::{GraphProto, ModelProto};
use crate::onnx_parser::{self, OnnxParseError, WeightTensor};
use crate::op_mapper::{self, MappedOp, MetadataOp, OpMapperError};
use crate::rms_norm_fusion::{self, RmsNormFusion};
use zkie_core::fixed_point::I18;
use zkie_core::isa::Instruction;

/// Identifies where a tensor's value comes from.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Register {
    /// A graph-level input tensor, by ONNX name.
    GraphInput(String),
    /// An initializer (weight), by ONNX name.
    Weight(String),
    /// An intermediate value produced by instruction #`usize` (0-indexed,
    /// in program order).
    Virtual(usize),
}

/// A single compiled instruction, wired up to its resolved input registers.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledInstruction {
    pub instruction: Instruction,
    pub inputs: Vec<Register>,
    /// The ONNX tensor name this instruction's output corresponds to, kept
    /// for debugging/traceability.
    pub output_name: String,
}

/// The fully-compiled, topologically-ordered program produced by
/// [`compile_graph`]/[`compile_model`].
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledProgram {
    pub instructions: Vec<CompiledInstruction>,
    /// Raw `f32` weights, from `onnx_parser::extract_initializers`. See
    /// module docs for why these are not eagerly converted to `I18`.
    pub weights: HashMap<String, WeightTensor>,
    /// ONNX graph input names, in order.
    pub graph_inputs: Vec<String>,
    /// ONNX graph output names, resolved to their final `Register`.
    pub graph_outputs: Vec<(String, Register)>,
}

impl CompiledProgram {
    /// Converts a single named weight to `I18` fixed-point, on demand. This
    /// lets callers (e.g. a later circuit-assembly stage) convert only the
    /// weights they actually consume, and decide how to handle/report a
    /// specific out-of-range weight in context, rather than this compiler
    /// eagerly failing on every weight in the model up front.
    pub fn weight_as_i18(&self, name: &str) -> Result<Vec<I18>, OnnxParseError> {
        let tensor = self
            .weights
            .get(name)
            .ok_or_else(|| OnnxParseError::WeightNotFound(name.to_string()))?;
        onnx_parser::weight_to_i18(tensor)
    }
}

/// Errors that can occur while compiling an ONNX graph into a
/// [`CompiledProgram`].
///
/// Not `Clone`/`PartialEq`: it wraps [`OnnxParseError`], which itself wraps
/// non-`Clone` types like `std::io::Error`. Tests compare variants with
/// `matches!` instead.
#[derive(Debug)]
pub enum GraphCompilerError {
    /// `compile_model` was called on a `ModelProto` with no `graph` field.
    MissingGraph,
    /// A node could not be mapped to a zkIE instruction or metadata op.
    OpMapper(OpMapperError),
    /// Initializer/weight extraction failed.
    OnnxParse(OnnxParseError),
    /// A node's input tensor name was not yet resolved to a `Register` at
    /// the point the node was processed. This indicates either a
    /// dependency-ordering bug in the topological sort, or a graph tensor
    /// whose producer was never found (e.g. a graph input missing from
    /// `graph.input`).
    UnresolvedInput {
        node_name: String,
        op_type: String,
        tensor_name: String,
    },
    /// An instruction-emitting node had zero or more than one output,
    /// which this compiler cannot represent (every supported instruction
    /// op is expected to have exactly one output).
    UnexpectedOutputCount {
        node_name: String,
        op_type: String,
        count: usize,
    },
    /// A graph-level output name did not resolve to any known `Register`.
    UnresolvedGraphOutput { tensor_name: String },
    /// The node dependency graph contains a cycle, so no valid topological
    /// order exists. This should not happen for a valid ONNX graph, but a
    /// malformed one must be rejected rather than causing a panic or hang.
    CyclicGraph,
}

impl fmt::Display for GraphCompilerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphCompilerError::MissingGraph => {
                write!(f, "ModelProto has no graph field")
            }
            GraphCompilerError::OpMapper(e) => write!(f, "op mapping failed: {e}"),
            GraphCompilerError::OnnxParse(e) => write!(f, "onnx parsing failed: {e}"),
            GraphCompilerError::UnresolvedInput {
                node_name,
                op_type,
                tensor_name,
            } => write!(
                f,
                "node '{node_name}' ({op_type}): input tensor '{tensor_name}' has no known register (not a graph input, initializer, or prior node output)"
            ),
            GraphCompilerError::UnexpectedOutputCount {
                node_name,
                op_type,
                count,
            } => write!(
                f,
                "node '{node_name}' ({op_type}): expected exactly one output, got {count}"
            ),
            GraphCompilerError::UnresolvedGraphOutput { tensor_name } => write!(
                f,
                "graph output '{tensor_name}' has no known register"
            ),
            GraphCompilerError::CyclicGraph => {
                write!(f, "graph contains a cycle; no valid topological order exists")
            }
        }
    }
}

impl std::error::Error for GraphCompilerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GraphCompilerError::OpMapper(e) => Some(e),
            GraphCompilerError::OnnxParse(e) => Some(e),
            _ => None,
        }
    }
}

impl From<OpMapperError> for GraphCompilerError {
    fn from(e: OpMapperError) -> Self {
        GraphCompilerError::OpMapper(e)
    }
}

impl From<OnnxParseError> for GraphCompilerError {
    fn from(e: OnnxParseError) -> Self {
        GraphCompilerError::OnnxParse(e)
    }
}

/// Computes a topological ordering of `graph.node`, returning the node
/// indices in dependency order (a node's producers all appear before it).
///
/// Uses Kahn's algorithm: an explicit `VecDeque` worklist of zero-in-degree
/// nodes, no recursion. Detects cycles (which cannot occur in a valid ONNX
/// graph, but could in a malformed one) by checking whether every node was
/// eventually dequeued.
fn topological_order(graph: &GraphProto) -> Result<Vec<usize>, GraphCompilerError> {
    let n = graph.node.len();

    // Map each tensor name to the index of the node that produces it.
    let mut producer_of: HashMap<&str, usize> = HashMap::new();
    for (idx, node) in graph.node.iter().enumerate() {
        for output_name in &node.output {
            producer_of.insert(output_name.as_str(), idx);
        }
    }

    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_degree: Vec<usize> = vec![0; n];

    for (idx, node) in graph.node.iter().enumerate() {
        // A node may reference the same producer through multiple input
        // names (e.g. `Add(x, x)`); only count the dependency edge once.
        let mut counted_producers: Vec<usize> = Vec::new();
        for input_name in &node.input {
            if let Some(&producer_idx) = producer_of.get(input_name.as_str()) {
                if producer_idx != idx && !counted_producers.contains(&producer_idx) {
                    counted_producers.push(producer_idx);
                    adjacency[producer_idx].push(idx);
                    in_degree[idx] += 1;
                }
            }
        }
    }

    let mut worklist: VecDeque<usize> = (0..n).filter(|&idx| in_degree[idx] == 0).collect();
    let mut order: Vec<usize> = Vec::with_capacity(n);

    while let Some(idx) = worklist.pop_front() {
        order.push(idx);
        for &next in &adjacency[idx] {
            in_degree[next] -= 1;
            if in_degree[next] == 0 {
                worklist.push_back(next);
            }
        }
    }

    if order.len() != n {
        // Not every node reached zero in-degree: a cycle exists among the
        // remaining nodes.
        return Err(GraphCompilerError::CyclicGraph);
    }

    Ok(order)
}

/// Compiles an ONNX `GraphProto` into a topologically-ordered, fully-wired
/// [`CompiledProgram`]. Does not panic on malformed or unsupported input;
/// all failure modes surface as a typed [`GraphCompilerError`].
pub fn compile_graph(graph: &GraphProto) -> Result<CompiledProgram, GraphCompilerError> {
    let weights = onnx_parser::extract_initializers(graph)?;

    // Shape lookup for the op mapper: value-info-derived shapes (graph
    // inputs/outputs/intermediates), overlaid with initializer shapes.
    // Initializer shapes aren't captured by `extract_value_shapes` (it only
    // scans `graph.input`/`graph.output`/`graph.value_info`), but ops like
    // `MatMul`/`Gemm` need the shape of weight operands too.
    let mut shapes = onnx_parser::extract_value_shapes(graph);
    for (name, tensor) in &weights {
        shapes.insert(name.clone(), tensor.shape.clone());
    }

    let mut registers: HashMap<String, Register> = HashMap::new();
    for input in &graph.input {
        registers.insert(input.name.clone(), Register::GraphInput(input.name.clone()));
    }
    for initializer in &graph.initializer {
        registers.insert(
            initializer.name.clone(),
            Register::Weight(initializer.name.clone()),
        );
    }

    // RMSNorm fusion pre-pass: recognizes the real 7-node
    // `Pow`/`ReduceMean`/`Add`/`Sqrt`/`Reciprocal`/`Mul`/`Mul` decomposition
    // (see `rms_norm_fusion`'s module docs) and folds each match into a
    // single `Instruction::RmsNorm`, emitted when the topological order
    // reaches the group's `Pow` node; the other six nodes in each match are
    // skipped entirely by the main loop below (never passed to
    // `op_mapper::map_node`).
    let fusions = rms_norm_fusion::detect_rms_norm_fusions(graph, &weights);
    let mut fusion_by_pow_idx: HashMap<usize, RmsNormFusion> = HashMap::new();
    let mut fusion_consumed: HashSet<usize> = HashSet::new();
    for fusion in fusions {
        fusion_consumed.extend(fusion.consumed_node_indices.iter().copied());
        fusion_by_pow_idx.insert(fusion.pow_node_idx, fusion);
    }

    let order = topological_order(graph)?;
    let mut instructions: Vec<CompiledInstruction> = Vec::new();

    for node_idx in order {
        if let Some(fusion) = fusion_by_pow_idx.get(&node_idx) {
            let x_register = resolve(&registers, "rms_norm_fusion", "RmsNorm", &fusion.input_name)?;
            let weight_register = resolve(
                &registers,
                "rms_norm_fusion",
                "RmsNorm",
                &fusion.weight_name,
            )?;

            let program_index = instructions.len();
            instructions.push(CompiledInstruction {
                instruction: Instruction::RmsNorm {
                    dim: fusion.dim,
                    epsilon_milli: fusion.epsilon_milli,
                },
                inputs: vec![x_register, weight_register],
                output_name: fusion.output_name.clone(),
            });
            registers.insert(fusion.output_name.clone(), Register::Virtual(program_index));
            continue;
        }
        if fusion_consumed.contains(&node_idx) {
            continue;
        }

        let node = &graph.node[node_idx];
        match op_mapper::map_node(node, &shapes)? {
            MappedOp::Instruction {
                instruction,
                input,
                output,
            } => {
                let resolved_inputs = input
                    .iter()
                    .map(|name| resolve(&registers, node.name.as_str(), &node.op_type, name))
                    .collect::<Result<Vec<Register>, GraphCompilerError>>()?;

                let output_name = match output.as_slice() {
                    [single] => single.clone(),
                    other => {
                        return Err(GraphCompilerError::UnexpectedOutputCount {
                            node_name: node.name.clone(),
                            op_type: node.op_type.clone(),
                            count: other.len(),
                        })
                    }
                };

                let program_index = instructions.len();
                instructions.push(CompiledInstruction {
                    instruction,
                    inputs: resolved_inputs,
                    output_name: output_name.clone(),
                });
                registers.insert(output_name, Register::Virtual(program_index));
            }
            MappedOp::Metadata(MetadataOp {
                op_type,
                input,
                output,
            }) => {
                let source_name =
                    input
                        .first()
                        .ok_or_else(|| GraphCompilerError::UnresolvedInput {
                            node_name: node.name.clone(),
                            op_type: op_type.clone(),
                            tensor_name: String::new(),
                        })?;
                let source_register =
                    resolve(&registers, node.name.as_str(), &op_type, source_name)?;
                for output_name in &output {
                    registers.insert(output_name.clone(), source_register.clone());
                }
            }
        }
    }

    let graph_outputs = graph
        .output
        .iter()
        .map(|value_info| {
            registers
                .get(&value_info.name)
                .cloned()
                .map(|register| (value_info.name.clone(), register))
                .ok_or_else(|| GraphCompilerError::UnresolvedGraphOutput {
                    tensor_name: value_info.name.clone(),
                })
        })
        .collect::<Result<Vec<_>, GraphCompilerError>>()?;

    Ok(CompiledProgram {
        instructions,
        weights,
        graph_inputs: graph.input.iter().map(|i| i.name.clone()).collect(),
        graph_outputs,
    })
}

fn resolve(
    registers: &HashMap<String, Register>,
    node_name: &str,
    op_type: &str,
    tensor_name: &str,
) -> Result<Register, GraphCompilerError> {
    registers
        .get(tensor_name)
        .cloned()
        .ok_or_else(|| GraphCompilerError::UnresolvedInput {
            node_name: node_name.to_string(),
            op_type: op_type.to_string(),
            tensor_name: tensor_name.to_string(),
        })
}

/// Extracts `model.graph` and delegates to [`compile_graph`].
pub fn compile_model(model: &ModelProto) -> Result<CompiledProgram, GraphCompilerError> {
    let graph = model
        .graph
        .as_ref()
        .ok_or(GraphCompilerError::MissingGraph)?;
    compile_graph(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::{
        tensor_shape_proto, type_proto, NodeProto, TensorProto, TensorShapeProto, TypeProto,
        ValueInfoProto,
    };

    const ONNX_DATA_TYPE_FLOAT: i32 = 1;

    fn node(name: &str, op_type: &str, input: Vec<&str>, output: Vec<&str>) -> NodeProto {
        NodeProto {
            name: name.to_string(),
            op_type: op_type.to_string(),
            input: input.into_iter().map(String::from).collect(),
            output: output.into_iter().map(String::from).collect(),
            ..Default::default()
        }
    }

    fn graph_input(name: &str) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            ..Default::default()
        }
    }

    fn graph_input_with_shape(name: &str, dims: Vec<i64>) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: ONNX_DATA_TYPE_FLOAT,
                    shape: Some(TensorShapeProto {
                        dim: dims
                            .into_iter()
                            .map(|d| tensor_shape_proto::Dimension {
                                value: Some(tensor_shape_proto::dimension::Value::DimValue(d)),
                                ..Default::default()
                            })
                            .collect(),
                    }),
                })),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn initializer(name: &str, dims: Vec<i64>, data: Vec<f32>) -> TensorProto {
        TensorProto {
            name: name.to_string(),
            dims,
            data_type: ONNX_DATA_TYPE_FLOAT,
            float_data: data,
            ..Default::default()
        }
    }

    // ---- Topological sort ----------------------------------------------

    #[test]
    fn topo_sort_orders_linear_chain() {
        // x -> Relu -> a -> Relu -> b -> Relu -> c
        let graph = GraphProto {
            input: vec![graph_input("x")],
            node: vec![
                node("n0", "Relu", vec!["x"], vec!["a"]),
                node("n1", "Relu", vec!["a"], vec!["b"]),
                node("n2", "Relu", vec!["b"], vec!["c"]),
            ],
            output: vec![graph_input("c")],
            ..Default::default()
        };

        let compiled = compile_graph(&graph).expect("should compile");
        assert_eq!(compiled.instructions.len(), 3);
        assert_eq!(compiled.instructions[0].output_name, "a");
        assert_eq!(compiled.instructions[1].output_name, "b");
        assert_eq!(compiled.instructions[2].output_name, "c");
        assert_eq!(compiled.instructions[1].inputs, vec![Register::Virtual(0)]);
        assert_eq!(compiled.instructions[2].inputs, vec![Register::Virtual(1)]);
        assert_eq!(
            compiled.graph_outputs,
            vec![("c".to_string(), Register::Virtual(2))]
        );
    }

    #[test]
    fn topo_sort_handles_independent_branches_that_merge() {
        // x -> Relu -> a
        // x -> Relu -> b
        // Add(a, b) -> c
        let graph = GraphProto {
            input: vec![graph_input("x")],
            node: vec![
                node("n0", "Relu", vec!["x"], vec!["a"]),
                node("n1", "Relu", vec!["x"], vec!["b"]),
                node("n2", "Add", vec!["a", "b"], vec!["c"]),
            ],
            output: vec![graph_input("c")],
            ..Default::default()
        };

        let compiled = compile_graph(&graph).expect("should compile");
        assert_eq!(compiled.instructions.len(), 3);
        // Both branches must precede the merge; branch order between the
        // two independent Relus is insertion-stable (n0 before n1) because
        // Kahn's algorithm seeds its worklist in original node order.
        assert_eq!(compiled.instructions[0].output_name, "a");
        assert_eq!(compiled.instructions[1].output_name, "b");
        assert_eq!(compiled.instructions[2].output_name, "c");
        assert_eq!(
            compiled.instructions[2].inputs,
            vec![Register::Virtual(0), Register::Virtual(1)]
        );
    }

    #[test]
    fn cyclic_graph_is_rejected_not_panicking() {
        // n0 needs "b" (produced by n1), n1 needs "a" (produced by n0) --
        // a genuine cycle with no valid topological order.
        let graph = GraphProto {
            node: vec![
                node("n0", "Relu", vec!["b"], vec!["a"]),
                node("n1", "Relu", vec!["a"], vec!["b"]),
            ],
            ..Default::default()
        };

        let result = compile_graph(&graph);
        assert!(matches!(result, Err(GraphCompilerError::CyclicGraph)));
    }

    // ---- Metadata passthrough -------------------------------------------

    #[test]
    fn metadata_op_aliases_output_to_input_register_without_emitting_instruction() {
        let graph = GraphProto {
            input: vec![graph_input("x")],
            node: vec![
                node("n0", "Reshape", vec!["x", "shape"], vec!["x_reshaped"]),
                node("n1", "Relu", vec!["x_reshaped"], vec!["y"]),
            ],
            output: vec![graph_input("y")],
            ..Default::default()
        };

        let compiled = compile_graph(&graph).expect("should compile");
        // Only the Relu should emit an instruction; Reshape is metadata-only.
        assert_eq!(compiled.instructions.len(), 1);
        assert_eq!(compiled.instructions[0].output_name, "y");
        // Relu's input resolves through the Reshape alias straight to x's
        // GraphInput register -- Reshape produced no new Register of its own.
        assert_eq!(
            compiled.instructions[0].inputs,
            vec![Register::GraphInput("x".to_string())]
        );
    }

    // ---- Unresolvable input ----------------------------------------------

    #[test]
    fn unresolvable_input_is_rejected_with_typed_error() {
        // "ghost" is neither a graph input, an initializer, nor any node's
        // output.
        let graph = GraphProto {
            node: vec![node("n0", "Relu", vec!["ghost"], vec!["y"])],
            ..Default::default()
        };

        let result = compile_graph(&graph);
        match result {
            Err(GraphCompilerError::UnresolvedInput { tensor_name, .. }) => {
                assert_eq!(tensor_name, "ghost");
            }
            other => panic!("expected UnresolvedInput, got {other:?}"),
        }
    }

    // ---- compile_model / MissingGraph -------------------------------------

    #[test]
    fn compile_model_rejects_missing_graph() {
        let model = ModelProto {
            graph: None,
            ..Default::default()
        };
        let result = compile_model(&model);
        assert!(matches!(result, Err(GraphCompilerError::MissingGraph)));
    }

    // ---- End-to-end: linear layer (MatMul + bias Add) ---------------------

    /// The single most important test in this module: proves the full
    /// ONNX -> zkIE pipeline (onnx_parser -> op_mapper -> graph_compiler)
    /// produces a correctly-wired program for a tiny "linear + bias" model,
    /// structurally identical to a transformer projection layer:
    ///
    ///   y = MatMul(x, W)
    ///   z = Add(y, b)
    ///
    /// with `W` a 2x2 weight matrix and `b` a length-2 bias, both ONNX
    /// initializers; `x` a graph input; `z` the sole graph output.
    #[test]
    fn compiles_matmul_plus_bias_linear_layer_end_to_end() {
        let w_data = vec![1.0f32, 2.0, 3.0, 4.0]; // row-major 2x2
        let b_data = vec![0.5f32, -0.5];

        let graph = GraphProto {
            input: vec![graph_input_with_shape("x", vec![2, 2])],
            output: vec![graph_input("z")],
            initializer: vec![
                initializer("W", vec![2, 2], w_data.clone()),
                initializer("b", vec![2], b_data.clone()),
            ],
            node: vec![
                node("matmul", "MatMul", vec!["x", "W"], vec!["y"]),
                node("add", "Add", vec!["y", "b"], vec!["z"]),
            ],
            ..Default::default()
        };

        let model = ModelProto {
            graph: Some(graph),
            ..Default::default()
        };

        let compiled = compile_model(&model).expect("should compile end-to-end");

        // Exactly 2 instructions, in the right order: MatMul's DotGeneral
        // before Add's Eltwise.
        assert_eq!(compiled.instructions.len(), 2);

        let matmul_instr = &compiled.instructions[0];
        assert_eq!(
            matmul_instr.instruction,
            Instruction::DotGeneral {
                m: 2,
                n: 2,
                k: 2,
                batch_dims: vec![],
                trans_a: false,
                trans_b: false,
            }
        );
        assert_eq!(
            matmul_instr.inputs,
            vec![
                Register::GraphInput("x".to_string()),
                Register::Weight("W".to_string()),
            ]
        );
        assert_eq!(matmul_instr.output_name, "y");

        let add_instr = &compiled.instructions[1];
        assert_eq!(
            add_instr.instruction,
            Instruction::Eltwise {
                op: zkie_core::isa::EltwiseOp::Add
            }
        );
        // Add's first input must point at MatMul's output (Virtual(0));
        // its second input must resolve to the "b" weight.
        assert_eq!(
            add_instr.inputs,
            vec![Register::Virtual(0), Register::Weight("b".to_string())]
        );
        assert_eq!(add_instr.output_name, "z");

        // The graph output resolves to the Add's output (Virtual(1)).
        assert_eq!(
            compiled.graph_outputs,
            vec![("z".to_string(), Register::Virtual(1))]
        );

        // Graph inputs recorded in order.
        assert_eq!(compiled.graph_inputs, vec!["x".to_string()]);

        // Raw f32 weights are preserved as-is (not eagerly converted).
        let w = compiled.weights.get("W").expect("W weight present");
        assert_eq!(w.shape, vec![2, 2]);
        assert_eq!(w.data, w_data);

        let b = compiled.weights.get("b").expect("b weight present");
        assert_eq!(b.shape, vec![2]);
        assert_eq!(b.data, b_data);
    }

    // ---- weight_as_i18 ------------------------------------------------------

    #[test]
    fn weight_as_i18_converts_in_range_weight() {
        let graph = GraphProto {
            initializer: vec![initializer("b", vec![2], vec![0.5, -0.5])],
            ..Default::default()
        };
        let compiled = compile_graph(&graph).expect("should compile");

        let converted = compiled.weight_as_i18("b").expect("in-range conversion");
        assert_eq!(converted.len(), 2);
        assert!((converted[0].to_f64() - 0.5).abs() < 1e-9);
        assert!((converted[1].to_f64() - (-0.5)).abs() < 1e-9);
    }

    #[test]
    fn weight_as_i18_returns_typed_error_for_out_of_range_weight() {
        let graph = GraphProto {
            initializer: vec![initializer("huge", vec![1], vec![1000.0])],
            ..Default::default()
        };
        let compiled = compile_graph(&graph).expect("should compile");

        let result = compiled.weight_as_i18("huge");
        match result {
            Err(OnnxParseError::WeightOutOfRange { value, .. }) => {
                assert_eq!(value, 1000.0);
            }
            other => panic!("expected WeightOutOfRange, got {other:?}"),
        }
    }

    #[test]
    fn weight_as_i18_returns_typed_error_for_unknown_weight_name() {
        let graph = GraphProto::default();
        let compiled = compile_graph(&graph).expect("should compile");

        let result = compiled.weight_as_i18("does_not_exist");
        match result {
            Err(OnnxParseError::WeightNotFound(name)) => {
                assert_eq!(name, "does_not_exist");
            }
            other => panic!("expected WeightNotFound, got {other:?}"),
        }
    }
}
