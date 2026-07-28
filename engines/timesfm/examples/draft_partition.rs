//! Draft shard-boundary generator: scans a `CompiledProgram` for repeated
//! occurrences of an "anchor" instruction (here, `Instruction::RmsNorm`,
//! which starts every layer in the real TimesFM architecture -- see
//! `docs/superpowers/specs/2026-07-27-zkie-dag-sharding-aggregation-design.md`)
//! and prints candidate shard boundaries. This is a convenience draft
//! only -- the design doc's section 4 explains why the final
//! `partition.toml` is always human-reviewed, never generated and used
//! as-is.
//!
//! Demonstrated here against the synthetic fixture program (there is no
//! way to compile the real TimesFM ONNX graph end-to-end yet -- see
//! `src/fixtures.rs`'s module docs), so its output is illustrative, not
//! the real TimesFM boundaries. It should print exactly the same
//! boundaries as `tests/fixtures/synthetic_partition.toml`.
//!
//! Usage: `cargo run -p zkie-ie-timesfm --example draft_partition`

use zkie_core::isa::Instruction;
use zkie_ie_timesfm::fixtures;

fn main() {
    let program = fixtures::synthetic_program();

    let anchors: Vec<usize> = program
        .instructions
        .iter()
        .enumerate()
        .filter(|(_, instr)| matches!(instr.instruction, Instruction::RmsNorm { .. }))
        .map(|(idx, _)| idx)
        .collect();

    println!(
        "found {} RmsNorm anchor(s) at instruction indices: {anchors:?}",
        anchors.len()
    );
    println!("(this is a DRAFT only -- a human must review/adjust before writing partition.toml)");
    println!();

    if anchors.is_empty() {
        println!("no anchors found; nothing to draft");
        return;
    }

    println!("prologue: 0..{}", anchors[0]);
    for (i, pair) in anchors.windows(2).enumerate() {
        println!("layer_{i}: {}..{}", pair[0], pair[1]);
    }

    // All layers are assumed uniform width (true of the real TimesFM
    // architecture and of this synthetic fixture); the gap between the
    // last two anchors is used to guess where the final layer ends, since
    // there is no next anchor to mark it.
    let layer_width = anchors
        .windows(2)
        .last()
        .map(|pair| pair[1] - pair[0])
        .unwrap_or(program.instructions.len() - anchors[0]);
    let last_layer_start = *anchors.last().unwrap();
    let last_layer_end = (last_layer_start + layer_width).min(program.instructions.len());
    println!(
        "layer_{}: {}..{}",
        anchors.len() - 1,
        last_layer_start,
        last_layer_end
    );

    if last_layer_end < program.instructions.len() {
        println!(
            "epilogue: {}..{}",
            last_layer_end,
            program.instructions.len()
        );
    }
}
