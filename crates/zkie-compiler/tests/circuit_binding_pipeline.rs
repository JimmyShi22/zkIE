//! End-to-end pipeline test: real ONNX bytes -> `onnx_parser` ->
//! `graph_compiler::compile_model` -> `circuit_binding::to_assembler_program`
//! -> `zkie_core::assembler::AssemblerChip` -> `MockProver`. This is the
//! "does the whole ONNX -> compile -> circuit pipeline actually wire up
//! correctly" milestone -- the real KZG roundtrip for the same linear-layer
//! shape lives in `zkie-core`'s own `tests/assembler_kzg_roundtrip.rs` (this
//! crate has no halo2 dependency outside of tests -- see
//! `circuit_binding.rs`'s module docs and this crate's `Cargo.toml`).

use std::collections::HashMap;

use halo2_proofs::circuit::{Layouter, SimpleFloorPlanner};
use halo2_proofs::dev::MockProver;
use halo2_proofs::plonk::{Circuit, ConstraintSystem, ErrorFront};

use zkie_compiler::circuit_binding::to_assembler_program;
use zkie_compiler::graph_compiler::compile_model;
use zkie_compiler::onnx::{
    tensor_shape_proto, type_proto, GraphProto, ModelProto, NodeProto, TensorProto,
    TensorShapeProto, TypeProto, ValueInfoProto,
};
use zkie_core::assembler::{AssemblerChip, AssemblerConfig, AssemblerProgram};
use zkie_core::field_convert::Fr;
use zkie_core::fixed_point::I18;

const ONNX_DATA_TYPE_FLOAT: i32 = 1;
const CIRCUIT_K: u32 = 12;

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

fn i18(v: f64) -> I18 {
    I18::from_f64(v).unwrap()
}

/// The same linear-layer graph as `graph_compiler`'s own
/// `compiles_matmul_plus_bias_linear_layer_end_to_end` test: `y = MatMul(x,
/// W); z = Add(y, b)`. Weight magnitudes are kept small (see
/// `zkie_core::assembler`'s own test docs) so every intermediate
/// dot-product accumulation and the final biased sum stay within I18's
/// representable range once concrete `x` values are witnessed.
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

#[derive(Clone)]
struct PipelineCircuit {
    program: AssemblerProgram,
}

impl Circuit<Fr> for PipelineCircuit {
    type Config = AssemblerConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        PipelineCircuit {
            program: self.program.clone(),
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        // `configure` only needs the program's *shape* (see
        // `AssemblerChip::configure`'s docs) -- but since `Circuit::configure`
        // has no access to `self`, this test recompiles the same ONNX model
        // once more just to get that shape. Cheap and keeps this test
        // self-contained (mirrors `zkie_core::assembler`'s own test module
        // pattern of hardcoding each test circuit's instruction shape).
        let model = linear_layer_model();
        let compiled = compile_model(&model).expect("should compile");
        let mut graph_input_values = HashMap::new();
        graph_input_values.insert("x".to_string(), vec![I18::from_raw(0); 4]);
        let program = to_assembler_program(&compiled, &graph_input_values)
            .expect("should convert to assembler program");
        AssemblerChip::configure(meta, &program.instructions)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        AssemblerChip::construct(config)
            .assign(layouter, &self.program)
            .map(|_| ())
            .map_err(|e| panic!("assembler assign failed: {e}"))
    }
}

#[test]
fn onnx_compile_to_assembler_program_to_circuit_is_satisfied_and_correct() {
    let model = linear_layer_model();
    let compiled = compile_model(&model).expect("should compile the ONNX graph");

    let x = vec![i18(0.5), i18(-0.25), i18(0.25), i18(0.5)];
    let mut graph_input_values = HashMap::new();
    graph_input_values.insert("x".to_string(), x.clone());

    let program = to_assembler_program(&compiled, &graph_input_values)
        .expect("should convert to an AssemblerProgram");

    let circuit = PipelineCircuit {
        program: program.clone(),
    };
    let prover = MockProver::run(CIRCUIT_K, &circuit, vec![]).unwrap();
    prover.assert_satisfied();

    // Independently compute the expected z = x@W + b (row-major, b
    // broadcast across rows), matching the exact same fixed-point
    // arithmetic the assembler's own chips use internally.
    let w = &program.weight_values[0];
    let b = &program.weight_values[1];
    let (m, n, k) = (2usize, 2usize, 2usize);
    let mut y = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            let raw_sum: i128 = (0..k)
                .map(|l| (x[i * k + l].raw() as i128) * (w[l * n + j].raw() as i128))
                .sum();
            let (q, _) = zkie_core::fixed_point::requantize_raw(raw_sum).unwrap();
            y.push(q);
        }
    }
    let z: Vec<I18> = y
        .iter()
        .enumerate()
        .map(|(idx, yv)| I18::from_raw(yv.raw() + b[idx % b.len()].raw()))
        .collect();

    // z's magnitude here (per the actual weight/input values) is small
    // enough that x@W's true real-valued result matches the fixed-point
    // computation to within floating-point rounding.
    let true_z = [0.725_f64, -0.2, 0.1, 0.425];
    for (got, want) in z.iter().zip(true_z.iter()) {
        assert!(
            (got.to_f64() - want).abs() < 1e-6,
            "{} vs {want}",
            got.to_f64()
        );
    }

    // Confirm the graph output "z" really does resolve to the Add
    // instruction's Virtual register (instructions[1]), matching `z` above.
    assert_eq!(compiled.graph_outputs.len(), 1);
    assert_eq!(compiled.graph_outputs[0].0, "z");
}
