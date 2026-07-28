//! Generic, model-agnostic DAG of "shards" -- contiguous instruction ranges
//! of a `CompiledProgram` that can be proven independently and in
//! parallel. See
//! `docs/superpowers/specs/2026-07-27-zkie-dag-sharding-aggregation-design.md`.
//!
//! This module only does def-use analysis over `CompiledProgram`'s
//! existing `Register`/`CompiledInstruction` data -- it has no knowledge
//! of any specific model (TimesFM, Gemma, ...). Which instruction ranges
//! form a shard is decided by the caller (see `engines/timesfm`'s
//! `partition` module for a concrete, human-reviewed source of those
//! ranges); this module only computes the *edges* between them.

use std::collections::HashMap;
use std::fmt;
use std::ops::Range;

use crate::graph_compiler::{CompiledProgram, Register};

/// One independently-provable unit: a contiguous range of instruction
/// indices into `CompiledProgram::instructions`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shard {
    pub id: usize,
    pub name: String,
    pub range: Range<usize>,
    /// Registers produced *outside* this shard's range that some
    /// instruction inside the range reads (only `Register::Virtual` --
    /// `GraphInput`/`Weight` are external to the whole program, not
    /// cross-shard dependencies, so they never appear here).
    pub inputs: Vec<Register>,
    /// Registers this shard produces (`Register::Virtual(i)` for `i` in
    /// `range`) that are read by at least one instruction in a *different*
    /// shard.
    pub outputs: Vec<Register>,
}

/// How a cross-shard dependency is classified. See module docs and the
/// design doc's section 1 for the real TimesFM examples this distinction
/// is modeled on. Does not change how `link` (see `linker.rs`) verifies an
/// edge -- both kinds are checked identically; this is a descriptive label
/// only (useful for humans reading a `Dag`, and for future
/// scheduling/distribution heuristics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    /// The producer is the immediately-preceding shard (by shard id), and
    /// it is the *only* shard reading this register -- a plain hand-off
    /// along the main chain (e.g. the residual stream between two
    /// consecutive transformer layers).
    Sequential,
    /// Anything else: either 2+ distinct shards read this register, or the
    /// single reader is not the immediately-next shard (the value bypasses
    /// one or more shards in between). Covers both a value broadcast to
    /// many shards (e.g. an attention mask) and a value produced early and
    /// consumed much later, skipping the shards in between (e.g.
    /// TimesFM's denormalization stats, produced by the prologue and
    /// consumed only by the epilogue).
    Broadcast,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edge {
    pub producer: usize,
    pub consumer: usize,
    pub register: Register,
    pub kind: EdgeKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dag {
    pub shards: Vec<Shard>,
    pub edges: Vec<Edge>,
}

/// A named instruction range a caller wants turned into one `Shard`. See
/// `engines/timesfm`'s `partition` module for how these are loaded from a
/// human-reviewed file rather than computed at runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardSpec {
    pub name: String,
    pub range: Range<usize>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum BuildDagError {
    /// `specs` was empty.
    NoShards,
    /// `specs` do not exactly and contiguously cover
    /// `[0, program.instructions.len())`: either the first spec doesn't
    /// start at 0, some spec doesn't end exactly where the next one
    /// starts, or the last spec doesn't end at
    /// `program.instructions.len()`.
    NotContiguous {
        expected_start: usize,
        got_start: usize,
        shard_index: usize,
    },
    /// An instruction's input is `Register::Virtual(i)` where `i` is not
    /// the index of any instruction in `program.instructions` -- e.g. a
    /// hand-built `CompiledProgram` with an out-of-range or forward
    /// reference. This must be rejected before `shard_of` is called on
    /// `i`, rather than panicking.
    UnknownRegister {
        consumer_instruction: usize,
        register: Register,
    },
}

impl fmt::Display for BuildDagError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildDagError::NoShards => write!(f, "no shard specs provided"),
            BuildDagError::NotContiguous {
                expected_start,
                got_start,
                shard_index,
            } => write!(
                f,
                "shard {shard_index} starts at {got_start}, expected {expected_start} (specs must exactly and contiguously cover every instruction)"
            ),
            BuildDagError::UnknownRegister {
                consumer_instruction,
                register,
            } => write!(
                f,
                "instruction {consumer_instruction} references {register:?}, which is not produced by any instruction in the program"
            ),
        }
    }
}

impl std::error::Error for BuildDagError {}

/// Builds a `Dag` from `program` and `specs`, deriving cross-shard edges
/// via def-use analysis: for every instruction's `Register::Virtual`
/// inputs, find which shard produced that register (the shard whose
/// `range` contains the producing instruction's index) and, if it differs
/// from the consuming instruction's own shard, record a cross-shard
/// dependency.
pub fn build_dag(program: &CompiledProgram, specs: &[ShardSpec]) -> Result<Dag, BuildDagError> {
    if specs.is_empty() {
        return Err(BuildDagError::NoShards);
    }

    let mut expected_start = 0usize;
    for (i, spec) in specs.iter().enumerate() {
        if spec.range.start != expected_start {
            return Err(BuildDagError::NotContiguous {
                expected_start,
                got_start: spec.range.start,
                shard_index: i,
            });
        }
        expected_start = spec.range.end;
    }
    if expected_start != program.instructions.len() {
        return Err(BuildDagError::NotContiguous {
            expected_start: program.instructions.len(),
            got_start: expected_start,
            shard_index: specs.len() - 1,
        });
    }

    let shard_of = |instr_idx: usize| -> usize {
        specs
            .iter()
            .position(|s| s.range.contains(&instr_idx))
            .expect("every instruction index is covered by exactly one shard spec (checked above)")
    };

    // For each Virtual register produced inside some shard, the distinct
    // set of *other* shards that read it (first-seen order, for
    // deterministic edge ordering).
    let mut consumers_of: HashMap<usize, Vec<usize>> = HashMap::new();

    for (consumer_instr_idx, instr) in program.instructions.iter().enumerate() {
        let consumer_shard = shard_of(consumer_instr_idx);
        for input in &instr.inputs {
            if let Register::Virtual(producer_instr_idx) = input {
                if *producer_instr_idx >= program.instructions.len() {
                    return Err(BuildDagError::UnknownRegister {
                        consumer_instruction: consumer_instr_idx,
                        register: Register::Virtual(*producer_instr_idx),
                    });
                }
                let producer_shard = shard_of(*producer_instr_idx);
                if producer_shard != consumer_shard {
                    let entry = consumers_of.entry(*producer_instr_idx).or_default();
                    if !entry.contains(&consumer_shard) {
                        entry.push(consumer_shard);
                    }
                }
            }
        }
    }

    let mut edges = Vec::new();
    let mut inputs_by_shard: Vec<Vec<Register>> = vec![Vec::new(); specs.len()];
    let mut outputs_by_shard: Vec<Vec<Register>> = vec![Vec::new(); specs.len()];

    let mut producer_indices: Vec<usize> = consumers_of.keys().copied().collect();
    producer_indices.sort_unstable();

    for producer_instr_idx in producer_indices {
        let producer_shard = shard_of(producer_instr_idx);
        let register = Register::Virtual(producer_instr_idx);
        let consumers = &consumers_of[&producer_instr_idx];

        outputs_by_shard[producer_shard].push(register.clone());

        let is_plain_handoff = consumers.len() == 1 && consumers[0] == producer_shard + 1;
        let kind = if is_plain_handoff {
            EdgeKind::Sequential
        } else {
            EdgeKind::Broadcast
        };

        for &consumer_shard in consumers {
            inputs_by_shard[consumer_shard].push(register.clone());
            edges.push(Edge {
                producer: producer_shard,
                consumer: consumer_shard,
                register: register.clone(),
                kind,
            });
        }
    }

    let shards = specs
        .iter()
        .enumerate()
        .map(|(id, spec)| Shard {
            id,
            name: spec.name.clone(),
            range: spec.range.clone(),
            inputs: std::mem::take(&mut inputs_by_shard[id]),
            outputs: std::mem::take(&mut outputs_by_shard[id]),
        })
        .collect();

    Ok(Dag { shards, edges })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_compiler::CompiledInstruction;
    use zkie_core::isa::{EltwiseOp, Instruction};

    fn add_instr(inputs: Vec<Register>, output_name: &str) -> CompiledInstruction {
        CompiledInstruction {
            instruction: Instruction::Eltwise { op: EltwiseOp::Add },
            inputs,
            output_name: output_name.to_string(),
        }
    }

    fn program_of(instructions: Vec<CompiledInstruction>) -> CompiledProgram {
        CompiledProgram {
            instructions,
            weights: Default::default(),
            graph_inputs: vec![],
            graph_outputs: vec![],
        }
    }

    #[test]
    fn linear_chain_across_two_shards_is_one_sequential_edge() {
        let program = program_of(vec![
            add_instr(vec![Register::GraphInput("x".into())], "a"),
            add_instr(vec![Register::Virtual(0)], "b"),
        ]);
        let specs = vec![
            ShardSpec {
                name: "s0".into(),
                range: 0..1,
            },
            ShardSpec {
                name: "s1".into(),
                range: 1..2,
            },
        ];

        let dag = build_dag(&program, &specs).expect("valid specs");

        assert_eq!(dag.edges.len(), 1);
        assert_eq!(dag.edges[0].producer, 0);
        assert_eq!(dag.edges[0].consumer, 1);
        assert_eq!(dag.edges[0].register, Register::Virtual(0));
        assert_eq!(dag.edges[0].kind, EdgeKind::Sequential);
        assert_eq!(dag.shards[0].outputs, vec![Register::Virtual(0)]);
        assert_eq!(dag.shards[1].inputs, vec![Register::Virtual(0)]);
    }

    #[test]
    fn internal_same_shard_register_use_produces_no_edge() {
        // instr1 reads instr0's output, but both are in shard 0 -- not a
        // cross-shard dependency, so no edge and no input/output entry.
        let program = program_of(vec![
            add_instr(vec![Register::GraphInput("x".into())], "a"),
            add_instr(vec![Register::Virtual(0)], "b"),
        ]);
        let specs = vec![ShardSpec {
            name: "s0".into(),
            range: 0..2,
        }];

        let dag = build_dag(&program, &specs).expect("valid specs");

        assert!(dag.edges.is_empty());
        assert!(dag.shards[0].inputs.is_empty());
        assert!(dag.shards[0].outputs.is_empty());
    }

    #[test]
    fn rejects_non_contiguous_specs() {
        let program = program_of(vec![add_instr(vec![], "a"), add_instr(vec![], "b")]);
        let specs = vec![
            ShardSpec {
                name: "s0".into(),
                range: 0..1,
            },
            ShardSpec {
                name: "s1".into(),
                range: 2..3,
            }, // gap: skips instruction 1
        ];

        let result = build_dag(&program, &specs);
        assert!(matches!(result, Err(BuildDagError::NotContiguous { .. })));
    }

    #[test]
    fn rejects_empty_specs() {
        let program = program_of(vec![]);
        let result = build_dag(&program, &[]);
        assert_eq!(result, Err(BuildDagError::NoShards));
    }

    #[test]
    fn register_consumed_by_two_shards_is_broadcast() {
        // Mirrors TimesFM's attention mask: produced once by the prologue,
        // read independently by every layer shard.
        let program = program_of(vec![
            add_instr(vec![], "mask"),                           // instr0, shard0
            add_instr(vec![Register::Virtual(0)], "layer0_out"), // instr1, shard1
            add_instr(vec![Register::Virtual(0)], "layer1_out"), // instr2, shard2
        ]);
        let specs = vec![
            ShardSpec {
                name: "prologue".into(),
                range: 0..1,
            },
            ShardSpec {
                name: "layer0".into(),
                range: 1..2,
            },
            ShardSpec {
                name: "layer1".into(),
                range: 2..3,
            },
        ];

        let dag = build_dag(&program, &specs).expect("valid specs");

        let broadcast_edges: Vec<_> = dag
            .edges
            .iter()
            .filter(|e| e.register == Register::Virtual(0))
            .collect();
        assert_eq!(broadcast_edges.len(), 2);
        assert!(broadcast_edges
            .iter()
            .all(|e| e.kind == EdgeKind::Broadcast));
        assert_eq!(dag.shards[0].outputs, vec![Register::Virtual(0)]);
    }

    #[test]
    fn single_far_consumer_that_skips_shards_is_broadcast_not_sequential() {
        // Mirrors TimesFM's denormalization stats: produced by the
        // prologue, consumed only by the epilogue, skipping every
        // transformer layer in between.
        let program = program_of(vec![
            add_instr(vec![], "denorm_stats"), // instr0, shard0
            add_instr(vec![Register::GraphInput("x".into())], "l0"), // instr1, shard1
            add_instr(vec![Register::Virtual(1)], "l1"), // instr2, shard2
            add_instr(vec![Register::Virtual(0), Register::Virtual(2)], "out"), // instr3, shard3
        ]);
        let specs = vec![
            ShardSpec {
                name: "prologue".into(),
                range: 0..1,
            },
            ShardSpec {
                name: "layer0".into(),
                range: 1..2,
            },
            ShardSpec {
                name: "layer1".into(),
                range: 2..3,
            },
            ShardSpec {
                name: "epilogue".into(),
                range: 3..4,
            },
        ];

        let dag = build_dag(&program, &specs).expect("valid specs");

        let edge = dag
            .edges
            .iter()
            .find(|e| e.register == Register::Virtual(0))
            .expect("edge exists");
        assert_eq!(edge.producer, 0);
        assert_eq!(edge.consumer, 3);
        assert_eq!(edge.kind, EdgeKind::Broadcast);
    }

    #[test]
    fn out_of_range_virtual_register_is_a_typed_error_not_a_panic() {
        // instr1 (the only instruction) references Virtual(5), but there
        // is no instruction 5 -- e.g. a hand-built `CompiledProgram` with
        // an off-by-one register index.
        let program = program_of(vec![add_instr(vec![Register::Virtual(5)], "a")]);
        let specs = vec![ShardSpec {
            name: "s0".into(),
            range: 0..1,
        }];

        let result = build_dag(&program, &specs);

        assert_eq!(
            result,
            Err(BuildDagError::UnknownRegister {
                consumer_instruction: 0,
                register: Register::Virtual(5),
            })
        );
    }
}
