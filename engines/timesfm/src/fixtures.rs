//! A small, synthetic, TimesFM-*shaped* `CompiledProgram` -- NOT the real
//! TimesFM 200M model (compiling that end-to-end is currently blocked on
//! `zkie-compiler`'s `op_mapper` missing `Split`/`Sub` and several
//! prologue-only ops; see the design doc). This mirrors the real
//! structural pattern found by directly analyzing
//! `models/timesfm_1_0_200m.onnx` (see
//! `docs/superpowers/specs/2026-07-27-zkie-dag-sharding-aggregation-design.md`
//! section 1): a prologue producing a plain sequential hand-off *and* two
//! broadcast values (one consumed by every repeated layer, one consumed
//! only by the epilogue, skipping every layer), 3 structurally identical
//! repeated layers, and an epilogue.
//!
//! Layout (must match `tests/fixtures/synthetic_partition.toml`):
//! instructions 0..3 = prologue, 3..13 = layer_0, 13..23 = layer_1,
//! 23..33 = layer_2, 33..34 = epilogue.
//!
//! `pub`, not `#[cfg(test)]`-gated, so both the integration test
//! (`tests/end_to_end_synthetic.rs`) and the `draft_partition` example can
//! use it.

use std::collections::HashMap;

use zkie_compiler::graph_compiler::{CompiledInstruction, CompiledProgram, Register};
use zkie_core::fixed_point::I18;
use zkie_core::isa::{EltwiseOp, Instruction};

const INSTRUCTIONS_PER_LAYER: usize = 10;
const NUM_LAYERS: usize = 3;
const TOTAL_INSTRUCTIONS: usize = 3 + NUM_LAYERS * INSTRUCTIONS_PER_LAYER + 1;

fn dot_general(inputs: Vec<Register>, output_name: &str) -> CompiledInstruction {
    CompiledInstruction {
        instruction: Instruction::DotGeneral {
            m: 1,
            n: 1,
            k: 1,
            batch_dims: vec![],
            trans_a: false,
            trans_b: false,
        },
        inputs,
        output_name: output_name.to_string(),
    }
}

fn eltwise(op: EltwiseOp, inputs: Vec<Register>, output_name: &str) -> CompiledInstruction {
    CompiledInstruction {
        instruction: Instruction::Eltwise { op },
        inputs,
        output_name: output_name.to_string(),
    }
}

fn weight(name: &str) -> Register {
    Register::Weight(name.to_string())
}

/// Appends one layer's 10 instructions to `instructions`, consuming
/// `hidden_state_in` (the previous layer's output, or the prologue's for
/// layer 0) and `mask` (broadcast from the prologue to every layer),
/// returning the layer's own final output register.
fn push_layer(
    instructions: &mut Vec<CompiledInstruction>,
    layer_idx: usize,
    hidden_state_in: Register,
    mask: Register,
) -> Register {
    let base = instructions.len();
    let prefix = format!("layer{layer_idx}");

    instructions.push(CompiledInstruction {
        instruction: Instruction::RmsNorm {
            dim: 4,
            epsilon_milli: 1,
        },
        inputs: vec![
            hidden_state_in.clone(),
            weight(&format!("{prefix}_rmsnorm_w")),
        ],
        output_name: format!("{prefix}_rmsnorm_out"),
    }); // base + 0
    instructions.push(dot_general(
        vec![Register::Virtual(base), weight(&format!("{prefix}_qkv_w"))],
        &format!("{prefix}_qkv"),
    )); // base + 1
    instructions.push(eltwise(
        EltwiseOp::Add,
        vec![Register::Virtual(base + 1), mask],
        &format!("{prefix}_masked"),
    )); // base + 2
    instructions.push(CompiledInstruction {
        instruction: Instruction::Softmax { axis_dim: 4 },
        inputs: vec![Register::Virtual(base + 2)],
        output_name: format!("{prefix}_attn"),
    }); // base + 3
    instructions.push(dot_general(
        vec![
            Register::Virtual(base + 3),
            weight(&format!("{prefix}_v_w")),
        ],
        &format!("{prefix}_attn_out"),
    )); // base + 4
    instructions.push(eltwise(
        EltwiseOp::Add,
        vec![Register::Virtual(base + 4), hidden_state_in],
        &format!("{prefix}_attn_residual"),
    )); // base + 5
    instructions.push(CompiledInstruction {
        instruction: Instruction::LayerNorm {
            dim: 4,
            epsilon_milli: 1,
        },
        inputs: vec![
            Register::Virtual(base + 5),
            weight(&format!("{prefix}_ln_w")),
        ],
        output_name: format!("{prefix}_ln_out"),
    }); // base + 6
    instructions.push(dot_general(
        vec![
            Register::Virtual(base + 6),
            weight(&format!("{prefix}_ffn_w1")),
        ],
        &format!("{prefix}_ffn_hidden"),
    )); // base + 7
    instructions.push(eltwise(
        EltwiseOp::Relu,
        vec![Register::Virtual(base + 7)],
        &format!("{prefix}_ffn_relu"),
    )); // base + 8
    instructions.push(eltwise(
        EltwiseOp::Add,
        vec![Register::Virtual(base + 8), Register::Virtual(base + 5)],
        &format!("{prefix}_out"),
    )); // base + 9

    assert_eq!(instructions.len(), base + INSTRUCTIONS_PER_LAYER);
    Register::Virtual(base + 9)
}

/// Builds the synthetic 34-instruction, 5-shard `CompiledProgram` described
/// in the module docs: prologue (3 instructions) + 3 repeated layers (10
/// instructions each) + epilogue (1 instruction).
pub fn synthetic_program() -> CompiledProgram {
    let mut instructions = Vec::new();

    // Prologue: 3 instructions producing hidden_state_0 (Virtual(0)), mask
    // (Virtual(1), broadcast to every layer), and denorm_stats
    // (Virtual(2), broadcast only to the epilogue).
    instructions.push(eltwise(
        EltwiseOp::Add,
        vec![Register::GraphInput("input_ts".into()), weight("zero")],
        "hidden_state_0",
    )); // Virtual(0)
    instructions.push(eltwise(
        EltwiseOp::Mul,
        vec![
            Register::GraphInput("input_padding".into()),
            weight("mask_w"),
        ],
        "mask",
    )); // Virtual(1)
    instructions.push(eltwise(
        EltwiseOp::Mul,
        vec![Register::GraphInput("input_ts".into()), weight("denorm_w")],
        "denorm_stats",
    )); // Virtual(2)

    let mask = Register::Virtual(1);
    let mut hidden_state = Register::Virtual(0);
    for layer_idx in 0..NUM_LAYERS {
        hidden_state = push_layer(&mut instructions, layer_idx, hidden_state, mask.clone());
    }

    // Epilogue: consumes the last layer's output (sequential) and
    // denorm_stats directly from the prologue (broadcast, skipping all 3
    // layers).
    instructions.push(eltwise(
        EltwiseOp::Mul,
        vec![hidden_state, Register::Virtual(2)],
        "output_ts",
    ));

    assert_eq!(instructions.len(), TOTAL_INSTRUCTIONS);

    CompiledProgram {
        instructions,
        weights: HashMap::new(),
        graph_inputs: vec!["input_ts".into(), "input_padding".into()],
        graph_outputs: vec![(
            "output_ts".into(),
            Register::Virtual(TOTAL_INSTRUCTIONS - 1),
        )],
    }
}

/// Arbitrary but deterministic `I18` witness values for every
/// `Register::Virtual` register `synthetic_program`'s instructions
/// produce. Only `Register::Virtual` entries are ever looked up by
/// `MockProver`: `build_dag` only tracks cross-shard `Virtual`
/// dependencies, so `Shard::inputs`/`outputs` never contain a
/// `GraphInput`/`Weight` register (each shard's own direct references to
/// those are shard-local/opaque to the Dag/Prover/Linker machinery, and
/// there is no plaintext interpreter yet to compute real values for them
/// anyway -- see the design doc).
pub fn synthetic_witness() -> HashMap<Register, Vec<I18>> {
    (0..TOTAL_INSTRUCTIONS)
        .map(|i| {
            (
                Register::Virtual(i),
                vec![I18::from_f64(i as f64 * 0.01).expect("in range")],
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_program_has_the_documented_shape() {
        let program = synthetic_program();
        assert_eq!(program.instructions.len(), 34);
        assert_eq!(program.graph_inputs, vec!["input_ts", "input_padding"]);
    }

    #[test]
    fn synthetic_witness_covers_every_virtual_register() {
        let witness = synthetic_witness();
        assert_eq!(witness.len(), 34);
        for i in 0..34 {
            assert!(witness.contains_key(&Register::Virtual(i)));
        }
    }
}
