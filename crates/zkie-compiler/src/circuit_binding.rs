//! Adapter: [`crate::graph_compiler::CompiledProgram`] (a structural,
//! value-agnostic computation graph) plus concrete host-side input tensor
//! values for one specific proving run, converted into a
//! `zkie_core::assembler::AssemblerProgram` -- the shape
//! [`zkie_core::assembler::AssemblerChip`] actually consumes to build a real
//! circuit.
//!
//! # Why this lives here, not in `graph_compiler.rs`
//!
//! `zkie_core::assembler` deliberately defines its own local `RegisterRef`/
//! `AssemblerInstruction`/`AssemblerProgram` types rather than importing
//! `graph_compiler::Register`/`CompiledProgram` directly, because
//! `zkie_core` cannot depend on `zkie_compiler` (this crate already depends
//! on `zkie_core` -- see `Cargo.toml` -- so the reverse edge would be
//! circular) -- see `zkie_core::assembler`'s own module docs for the full
//! reasoning (this is the same composed-chip soundness concern
//! `RangeCheckChip`'s disconnected-witness bug and
//! `chips::layer_norm::LayerNormChip` already ran into, one level further
//! up: instruction-to-instruction wiring across an entire compiled program).
//! This module is the thin, one-directional translation layer between the
//! two: `zkie_compiler` already depends on `zkie_core`, so building an
//! `zkie_core::assembler::AssemblerProgram` here is a normal forward
//! dependency, not a circular one. `graph_compiler.rs` itself stays
//! untouched.
//!
//! # What this adapter actually does
//!
//! - Graph inputs: `CompiledProgram::graph_inputs` gives the input tensors'
//!   names and order; this adapter requires the caller to supply concrete
//!   `I18` values for every one of them (`graph_input_values`), and assigns
//!   them stable `RegisterRef::Input` indices matching that order.
//! - Weights: only weights that some instruction actually references are
//!   converted (via `CompiledProgram::weight_as_i18`, on demand -- mirroring
//!   that method's own "don't eagerly fail on every weight in the model"
//!   design note), each assigned a stable `RegisterRef::Weight` index the
//!   first time it's referenced (and reused, not re-converted, on every
//!   subsequent reference to the same weight name).
//! - `Register::Virtual(i)` maps straight through to `RegisterRef::Virtual(i)`
//!   -- both conventions already agree on "the output of instruction `i`,
//!   0-indexed in program order".

use std::collections::HashMap;
use std::fmt;

use crate::graph_compiler::{CompiledProgram, Register};
use crate::onnx_parser::OnnxParseError;
use zkie_core::assembler::{AssemblerInstruction, AssemblerProgram, RegisterRef};
use zkie_core::fixed_point::I18;

/// Errors that can occur while converting a [`CompiledProgram`] into an
/// [`AssemblerProgram`].
#[derive(Debug)]
pub enum CircuitBindingError {
    /// `graph_input_values` did not contain an entry for one of
    /// `program.graph_inputs`.
    MissingGraphInput(String),
    /// Converting a referenced weight to `I18` failed (out of range, wrong
    /// dtype, etc.) -- see [`CompiledProgram::weight_as_i18`].
    Weight(OnnxParseError),
}

impl fmt::Display for CircuitBindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CircuitBindingError::MissingGraphInput(name) => write!(
                f,
                "no concrete input values were supplied for graph input '{name}'"
            ),
            CircuitBindingError::Weight(e) => write!(f, "failed to convert a weight: {e}"),
        }
    }
}

impl std::error::Error for CircuitBindingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CircuitBindingError::Weight(e) => Some(e),
            _ => None,
        }
    }
}

impl From<OnnxParseError> for CircuitBindingError {
    fn from(e: OnnxParseError) -> Self {
        CircuitBindingError::Weight(e)
    }
}

/// Converts `program` (structural) plus `graph_input_values` (concrete,
/// per-input-name tensors for one specific proving run) into an
/// `AssemblerProgram` ready for `zkie_core::assembler::AssemblerChip`.
pub fn to_assembler_program(
    program: &CompiledProgram,
    graph_input_values: &HashMap<String, Vec<I18>>,
) -> Result<AssemblerProgram, CircuitBindingError> {
    let mut input_values = Vec::with_capacity(program.graph_inputs.len());
    let mut input_index: HashMap<&str, usize> = HashMap::with_capacity(program.graph_inputs.len());
    for (idx, name) in program.graph_inputs.iter().enumerate() {
        let values = graph_input_values
            .get(name)
            .cloned()
            .ok_or_else(|| CircuitBindingError::MissingGraphInput(name.clone()))?;
        input_index.insert(name.as_str(), idx);
        input_values.push(values);
    }

    let mut weight_values: Vec<Vec<I18>> = Vec::new();
    let mut weight_index: HashMap<&str, usize> = HashMap::new();

    let mut instructions = Vec::with_capacity(program.instructions.len());
    for compiled_instr in &program.instructions {
        let inputs = compiled_instr
            .inputs
            .iter()
            .map(|reg| {
                resolve_register(
                    reg,
                    &input_index,
                    &mut weight_index,
                    &mut weight_values,
                    program,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        instructions.push(AssemblerInstruction {
            instruction: compiled_instr.instruction.clone(),
            inputs,
        });
    }

    Ok(AssemblerProgram {
        instructions,
        input_values,
        weight_values,
    })
}

/// Resolves a single `graph_compiler::Register` to its `zkie_core`-local
/// `RegisterRef` equivalent, converting and caching a weight's `I18` values
/// the first time that weight name is referenced.
fn resolve_register<'a>(
    reg: &'a Register,
    input_index: &HashMap<&'a str, usize>,
    weight_index: &mut HashMap<&'a str, usize>,
    weight_values: &mut Vec<Vec<I18>>,
    program: &CompiledProgram,
) -> Result<RegisterRef, CircuitBindingError> {
    match reg {
        Register::GraphInput(name) => {
            let idx = *input_index
                .get(name.as_str())
                .ok_or_else(|| CircuitBindingError::MissingGraphInput(name.clone()))?;
            Ok(RegisterRef::Input(idx))
        }
        Register::Weight(name) => {
            if let Some(&idx) = weight_index.get(name.as_str()) {
                return Ok(RegisterRef::Weight(idx));
            }
            let values = program.weight_as_i18(name)?;
            let idx = weight_values.len();
            weight_values.push(values);
            weight_index.insert(name.as_str(), idx);
            Ok(RegisterRef::Weight(idx))
        }
        Register::Virtual(i) => Ok(RegisterRef::Virtual(*i)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_compiler::compile_model;
    use crate::onnx::{
        tensor_shape_proto, type_proto, GraphProto, ModelProto, NodeProto, TensorProto,
        TensorShapeProto, TypeProto, ValueInfoProto,
    };
    use zkie_core::isa::{EltwiseOp, Instruction};

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

    fn graph_output(name: &str) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
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

    /// The exact linear-layer graph
    /// (`graph_compiler::compiles_matmul_plus_bias_linear_layer_end_to_end`
    /// compiles), used here to exercise the real ONNX -> compile ->
    /// adapter -> AssemblerProgram path end-to-end.
    fn linear_layer_model() -> ModelProto {
        let graph = GraphProto {
            input: vec![graph_input_with_shape("x", vec![2, 2])],
            output: vec![graph_output("z")],
            initializer: vec![
                initializer("W", vec![2, 2], vec![1.0, 0.5, -0.5, 1.0]),
                initializer("b", vec![2], vec![0.1, -0.2]),
            ],
            node: vec![
                node("matmul", "MatMul", vec!["x", "W"], vec!["y"]),
                node("add", "Add", vec!["y", "b"], vec!["z"]),
            ],
            ..Default::default()
        };
        ModelProto {
            graph: Some(graph),
            ..Default::default()
        }
    }

    fn i18(v: f64) -> I18 {
        I18::from_f64(v).unwrap()
    }

    #[test]
    fn converts_linear_layer_program_with_correctly_wired_registers() {
        let model = linear_layer_model();
        let compiled = compile_model(&model).expect("should compile");

        let mut graph_input_values = HashMap::new();
        graph_input_values.insert(
            "x".to_string(),
            vec![i18(0.5), i18(-0.25), i18(0.25), i18(0.5)],
        );

        let assembler_program =
            to_assembler_program(&compiled, &graph_input_values).expect("should convert");

        assert_eq!(assembler_program.instructions.len(), 2);
        assert_eq!(assembler_program.input_values.len(), 1);
        assert_eq!(
            assembler_program.input_values[0],
            vec![i18(0.5), i18(-0.25), i18(0.25), i18(0.5)]
        );

        // MatMul: DotGeneral over (Input(0)=x, Weight(0)=W).
        let matmul = &assembler_program.instructions[0];
        assert!(matches!(
            matmul.instruction,
            Instruction::DotGeneral {
                m: 2,
                n: 2,
                k: 2,
                ..
            }
        ));
        assert_eq!(
            matmul.inputs,
            vec![RegisterRef::Input(0), RegisterRef::Weight(0)]
        );

        // Add: Eltwise over (Virtual(0)=y, Weight(1)=b) -- Virtual(0) must be
        // the register that later becomes AssemblerChip's own `Virtual(0)`
        // resolution of MatMul's real output cell (this is exactly the
        // cross-instruction wiring this whole assembler exists to link
        // soundly -- see `zkie_core::assembler`'s own module docs).
        let add = &assembler_program.instructions[1];
        assert_eq!(add.instruction, Instruction::Eltwise { op: EltwiseOp::Add });
        assert_eq!(
            add.inputs,
            vec![RegisterRef::Virtual(0), RegisterRef::Weight(1)]
        );

        // W and b converted to I18 and cached under stable weight indices.
        // Compared with a tolerance (not `assert_eq!`) since the weights
        // round-trip through f32 (ONNX's storage type) before conversion to
        // I18, which loses a few bits of precision relative to the f64
        // literals used to construct the expected values here.
        assert_eq!(assembler_program.weight_values.len(), 2);
        assert_close(&assembler_program.weight_values[0], &[1.0, 0.5, -0.5, 1.0]);
        assert_close(&assembler_program.weight_values[1], &[0.1, -0.2]);
    }

    fn assert_close(got: &[I18], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g.to_f64() - w).abs() < 1e-6, "{} vs {w}", g.to_f64());
        }
    }

    #[test]
    fn missing_graph_input_value_is_rejected_with_typed_error() {
        let model = linear_layer_model();
        let compiled = compile_model(&model).expect("should compile");
        let graph_input_values = HashMap::new(); // "x" deliberately omitted

        let result = to_assembler_program(&compiled, &graph_input_values);
        match result {
            Err(CircuitBindingError::MissingGraphInput(name)) => assert_eq!(name, "x"),
            other => panic!("expected MissingGraphInput, got {other:?}"),
        }
    }

    #[test]
    fn out_of_range_weight_is_rejected_with_typed_error() {
        let graph = GraphProto {
            input: vec![graph_input_with_shape("x", vec![2])],
            output: vec![graph_output("z")],
            initializer: vec![initializer("huge", vec![2], vec![1000.0, 1000.0])],
            node: vec![node("add", "Add", vec!["x", "huge"], vec!["z"])],
            ..Default::default()
        };
        let model = ModelProto {
            graph: Some(graph),
            ..Default::default()
        };
        let compiled = compile_model(&model).expect("should compile");

        let mut graph_input_values = HashMap::new();
        graph_input_values.insert("x".to_string(), vec![i18(0.1), i18(0.2)]);

        let result = to_assembler_program(&compiled, &graph_input_values);
        assert!(matches!(result, Err(CircuitBindingError::Weight(_))));
    }

    #[test]
    fn repeated_weight_reference_reuses_the_same_index() {
        // Add(x, w) then Mul(y, w): "w" is referenced twice but should only
        // be converted/cached once, at a single stable index.
        let graph = GraphProto {
            input: vec![graph_input_with_shape("x", vec![2])],
            output: vec![graph_output("z")],
            initializer: vec![initializer("w", vec![2], vec![0.5, 0.5])],
            node: vec![
                node("add", "Add", vec!["x", "w"], vec!["y"]),
                node("mul", "Mul", vec!["y", "w"], vec!["z"]),
            ],
            ..Default::default()
        };
        let model = ModelProto {
            graph: Some(graph),
            ..Default::default()
        };
        let compiled = compile_model(&model).expect("should compile");

        let mut graph_input_values = HashMap::new();
        graph_input_values.insert("x".to_string(), vec![i18(0.1), i18(0.2)]);

        let assembler_program =
            to_assembler_program(&compiled, &graph_input_values).expect("should convert");

        assert_eq!(assembler_program.weight_values.len(), 1);
        assert_eq!(
            assembler_program.instructions[0].inputs[1],
            RegisterRef::Weight(0)
        );
        assert_eq!(
            assembler_program.instructions[1].inputs[1],
            RegisterRef::Weight(0)
        );
    }
}
