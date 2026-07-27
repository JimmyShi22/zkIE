# zkIE DAG Sharding + Parallel Proving + Aggregation-Linking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a generic, model-agnostic mechanism (`zkie-compiler::dag`) that splits a compiled ISA program into independently-provable "shards", proves each with a pluggable (for now mocked) backend, and verifies cross-shard consistency via commitment-linking — plus a first concrete model-specific consumer (`engines/timesfm`) that defines TimesFM's shard boundaries from a human-reviewed static file.

**Architecture:** `zkie-compiler` gains a new `dag` module (`Shard`/`Dag`/`build_dag`/`Prover` trait/`MockProver`/`Linker`) that does pure def-use analysis over the existing `CompiledProgram`/`Register` types — no model-specific knowledge. A new top-level `engines/timesfm` crate (package `zkie-ie-timesfm`) loads a static `partition.toml` (shard boundaries, human-authored) and demonstrates the whole pipeline end-to-end against a synthetic, TimesFM-*shaped* `CompiledProgram` (real ONNX compilation of the full model is blocked on unrelated `op_mapper` gaps — see Global Constraints).

**Tech Stack:** Rust, existing `zkie-core`/`zkie-compiler` crates, `blake3` (commitments), `toml`+`serde` (partition file), `rayon` (parallel proving in tests).

**Design doc:** `docs/superpowers/specs/2026-07-27-zkie-dag-sharding-aggregation-design.md`

## Global Constraints

- Never run `cargo build`/`cargo test`/anything that compiles a crate with a build script from a `/tmp`-based path — always run from inside `/Users/jimmyshi/code/zkie/...` (see project `CLAUDE.md`).
- The new `dag` module lives in **`zkie-compiler`**, not `zkie-core`, even though the design doc's section 2 diagram shows it under the "zkie 执行引擎层" — `CompiledProgram`/`Register` are defined in `zkie-compiler::graph_compiler`, and `zkie-core` must not depend on `zkie-compiler` (it already depends the other way around — see `crates/zkie-compiler/src/circuit_binding.rs`'s own module docs, which document this exact one-directional-dependency constraint for the same reason). This is a placement correction discovered while writing this plan, not a change to the design's intent — `zkie-compiler` is just as model-agnostic as `zkie-core` (it's a generic ONNX→ISA compiler).
- Follow the existing error-handling convention already used by `graph_compiler::GraphCompilerError`/`onnx_parser::OnnxParseError`: hand-rolled `enum` + `impl fmt::Display` + `impl std::error::Error`, not `thiserror` (not a dependency anywhere in this workspace today — don't introduce it).
- `cargo fmt` and `cargo clippy --workspace --all-targets -- -D warnings` must stay clean after every task.
- **Out of scope for this plan** (see design doc section 0 and section 7 — do not attempt any of this): a real STARK proving backend, a real recursive/aggregation SNARK (the `Linker` in this plan is a plain commitment-consistency check, not a succinct proof), automatic repeated-subgraph detection, distributed shard dispatch.
- **No plaintext ONNX interpreter exists in this codebase** (confirmed by searching for `interpret`/`execute`/`run_program` — none found; the only thing that consumes a `CompiledProgram` today is `circuit_binding.rs`, which builds a real halo2 circuit, not a plain-value evaluator). Building one is out of scope here. Tests/fixtures in this plan supply witness values by hand instead of computing them by running the program.
- **The real TimesFM `models/timesfm_1_0_200m.onnx` cannot be compiled end-to-end today**: `op_mapper.rs` does not yet map `Split` (used by the real QKV split) or `Sub` (used by the real padding-mask multiply before each layer's residual add), nor the prologue's `Where`/`Cast`/`ArgMax`/`Div`/`Sigmoid`/`GatherND`/`Sin`/`Cos`/`Pad`/etc. Expanding `op_mapper` coverage is a distinct, separate piece of work and is explicitly out of scope for this plan. This plan instead builds and tests everything against a **synthetic, hand-constructed `CompiledProgram`** that is structurally shaped like TimesFM (a prologue producing a sequential hand-off plus two broadcast values, N structurally-identical repeated layers, an epilogue) — this is documented clearly in code (see Task 6) so nobody mistakes it for the real model.

---

### Task 1: `Shard`/`Dag`/`build_dag` with sequential edges only

**Files:**
- Create: `crates/zkie-compiler/src/dag/mod.rs`
- Create: `crates/zkie-compiler/src/dag/model.rs`
- Modify: `crates/zkie-compiler/src/lib.rs` (add `pub mod dag;`)

**Interfaces:**
- Produces: `zkie_compiler::dag::{Shard, EdgeKind, Edge, Dag, ShardSpec, BuildDagError, build_dag}` — used by every later task.
  - `pub struct ShardSpec { pub name: String, pub range: std::ops::Range<usize> }`
  - `pub fn build_dag(program: &CompiledProgram, specs: &[ShardSpec]) -> Result<Dag, BuildDagError>`
  - `pub struct Shard { pub id: usize, pub name: String, pub range: Range<usize>, pub inputs: Vec<Register>, pub outputs: Vec<Register> }`

- [ ] **Step 1: Write the failing tests**

Create `crates/zkie-compiler/src/dag/model.rs`:

```rust
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
            ShardSpec { name: "s0".into(), range: 0..1 },
            ShardSpec { name: "s1".into(), range: 1..2 },
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
        let specs = vec![ShardSpec { name: "s0".into(), range: 0..2 }];

        let dag = build_dag(&program, &specs).expect("valid specs");

        assert!(dag.edges.is_empty());
        assert!(dag.shards[0].inputs.is_empty());
        assert!(dag.shards[0].outputs.is_empty());
    }

    #[test]
    fn rejects_non_contiguous_specs() {
        let program = program_of(vec![add_instr(vec![], "a"), add_instr(vec![], "b")]);
        let specs = vec![
            ShardSpec { name: "s0".into(), range: 0..1 },
            ShardSpec { name: "s1".into(), range: 2..3 }, // gap: skips instruction 1
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
}
```

Create `crates/zkie-compiler/src/dag/mod.rs`:

```rust
//! Generic, model-agnostic DAG of independently-provable "shards" over a
//! `CompiledProgram`, plus a pluggable `Prover` and a
//! commitment-consistency `Linker`. See
//! `docs/superpowers/specs/2026-07-27-zkie-dag-sharding-aggregation-design.md`.

pub mod model;

pub use model::{build_dag, BuildDagError, Dag, Edge, EdgeKind, Shard, ShardSpec};
```

- [ ] **Step 2: Wire the module into the crate**

Modify `crates/zkie-compiler/src/lib.rs`, adding `pub mod dag;` alongside the other `pub mod` lines (order doesn't matter; keep alphabetical with the existing ones for consistency: after `circuit_binding`, before `graph_compiler`).

- [ ] **Step 3: Run the new tests, expect them to pass**

Run: `cd /Users/jimmyshi/code/zkie && cargo test -p zkie-compiler dag:: -- --nocapture`
Expected: 4 tests pass (`linear_chain_across_two_shards_is_one_sequential_edge`, `internal_same_shard_register_use_produces_no_edge`, `rejects_non_contiguous_specs`, `rejects_empty_specs`).

- [ ] **Step 4: Run clippy and fmt**

Run: `cargo clippy -p zkie-compiler --all-targets -- -D warnings && cargo fmt -p zkie-compiler -- --check`
Expected: no warnings, no formatting diffs (if `fmt --check` reports diffs, run `cargo fmt -p zkie-compiler` and re-check).

- [ ] **Step 5: Commit**

```bash
git add crates/zkie-compiler/src/dag crates/zkie-compiler/src/lib.rs
git commit -m "feat: add Shard/Dag/build_dag def-use analysis to zkie-compiler"
```

---

### Task 2: Extend `build_dag` for broadcast edges

**Files:**
- Modify: `crates/zkie-compiler/src/dag/model.rs` (tests only — the classification logic from Task 1 already handles this; this task's job is to *prove* it with the two real-world cases from the design doc, not to change the implementation)

**Interfaces:**
- Consumes: `build_dag`, `ShardSpec`, `EdgeKind` from Task 1 (unchanged signatures).
- Produces: nothing new — this task only adds test coverage confirming `EdgeKind::Broadcast` is classified correctly for both real cases found in the design doc (a value read by 2+ shards; a value read by exactly one shard that isn't the immediately-next one).

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `crates/zkie-compiler/src/dag/model.rs`:

```rust
    #[test]
    fn register_consumed_by_two_shards_is_broadcast() {
        // Mirrors TimesFM's attention mask: produced once by the prologue,
        // read independently by every layer shard.
        let program = program_of(vec![
            add_instr(vec![], "mask"),                            // instr0, shard0
            add_instr(vec![Register::Virtual(0)], "layer0_out"),  // instr1, shard1
            add_instr(vec![Register::Virtual(0)], "layer1_out"),  // instr2, shard2
        ]);
        let specs = vec![
            ShardSpec { name: "prologue".into(), range: 0..1 },
            ShardSpec { name: "layer0".into(), range: 1..2 },
            ShardSpec { name: "layer1".into(), range: 2..3 },
        ];

        let dag = build_dag(&program, &specs).expect("valid specs");

        let broadcast_edges: Vec<_> = dag
            .edges
            .iter()
            .filter(|e| e.register == Register::Virtual(0))
            .collect();
        assert_eq!(broadcast_edges.len(), 2);
        assert!(broadcast_edges.iter().all(|e| e.kind == EdgeKind::Broadcast));
        assert_eq!(dag.shards[0].outputs, vec![Register::Virtual(0)]);
    }

    #[test]
    fn single_far_consumer_that_skips_shards_is_broadcast_not_sequential() {
        // Mirrors TimesFM's denormalization stats: produced by the
        // prologue, consumed only by the epilogue, skipping every
        // transformer layer in between.
        let program = program_of(vec![
            add_instr(vec![], "denorm_stats"),                                   // instr0, shard0
            add_instr(vec![Register::GraphInput("x".into())], "l0"),             // instr1, shard1
            add_instr(vec![Register::Virtual(1)], "l1"),                         // instr2, shard2
            add_instr(vec![Register::Virtual(0), Register::Virtual(2)], "out"),  // instr3, shard3
        ]);
        let specs = vec![
            ShardSpec { name: "prologue".into(), range: 0..1 },
            ShardSpec { name: "layer0".into(), range: 1..2 },
            ShardSpec { name: "layer1".into(), range: 2..3 },
            ShardSpec { name: "epilogue".into(), range: 3..4 },
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
```

- [ ] **Step 2: Run the new tests, expect them to pass**

Run: `cargo test -p zkie-compiler dag::model::tests -- --nocapture`
Expected: all 6 tests in this module pass (the 4 from Task 1 plus these 2) — no implementation change was needed, since `build_dag`'s classification rule from Task 1 already handles both cases correctly.

- [ ] **Step 3: Run clippy and fmt**

Run: `cargo clippy -p zkie-compiler --all-targets -- -D warnings && cargo fmt -p zkie-compiler -- --check`

- [ ] **Step 4: Commit**

```bash
git add crates/zkie-compiler/src/dag/model.rs
git commit -m "test: cover broadcast-edge classification (multi-consumer and skip-ahead cases)"
```

---

### Task 3: `Prover` trait, `MockProver`, and BLAKE3 commitments

**Files:**
- Modify: `crates/zkie-compiler/Cargo.toml` (add `blake3` dependency)
- Create: `crates/zkie-compiler/src/dag/prover.rs`
- Modify: `crates/zkie-compiler/src/dag/mod.rs` (add `pub mod prover;` and re-exports)

**Interfaces:**
- Consumes: `Shard` from Task 1.
- Produces: `zkie_compiler::dag::{Commitment, Witness, ShardProof, Prover, MockProver}` — used by Task 4 (`Linker`) and Task 6 (integration test).
  - `pub struct Commitment(pub [u8; 32])`
  - `pub type Witness = HashMap<Register, Vec<I18>>`
  - `pub struct ShardProof { pub shard_id: usize, pub input_commitments: HashMap<Register, Commitment>, pub output_commitments: HashMap<Register, Commitment>, pub valid: bool }`
  - `pub trait Prover { fn prove(&self, shard: &Shard, witness: &Witness) -> ShardProof; }`
  - `pub struct MockProver;` (implements `Prover`)

- [ ] **Step 1: Add the `blake3` dependency**

Modify `crates/zkie-compiler/Cargo.toml`, adding to `[dependencies]`:

```toml
blake3 = "1"
```

Run: `cargo build -p zkie-compiler` (confirm it fetches/builds with the new dependency before writing code that uses it).
Expected: builds successfully (no code uses `blake3` yet, this just confirms the dependency resolves).

- [ ] **Step 2: Write the failing tests**

Create `crates/zkie-compiler/src/dag/prover.rs`:

```rust
//! A pluggable prover for one `Shard`, plus the only implementation used in
//! this sub-project: `MockProver`, which computes real BLAKE3 commitments
//! over witness values but never generates an actual proof (`valid` is
//! always `true`). See
//! `docs/superpowers/specs/2026-07-27-zkie-dag-sharding-aggregation-design.md`
//! section 0 for why a real STARK-backed `Prover` is out of scope here --
//! this exists so `Linker`'s commitment-matching logic (the part of this
//! sub-project that must be genuinely correct) can be tested end-to-end
//! today, without waiting on a real STARK backend. A future real `Prover`
//! implementation is a drop-in replacement: `Dag`/`Linker` never change.

use std::collections::HashMap;

use crate::graph_compiler::Register;
use zkie_core::fixed_point::I18;

use super::model::Shard;

/// A binding commitment to a register's concrete witness value(s) -- a
/// BLAKE3 hash of its `I18` raw `i64` values, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Commitment(pub [u8; 32]);

impl Commitment {
    pub fn of_i18_values(values: &[I18]) -> Self {
        let mut hasher = blake3::Hasher::new();
        for value in values {
            hasher.update(&value.raw().to_le_bytes());
        }
        Commitment(*hasher.finalize().as_bytes())
    }
}

/// The concrete `I18` values for every register a shard's `prove` call
/// needs -- in this sub-project, supplied directly by the caller (there is
/// no plaintext ONNX interpreter yet; see the design doc for why building
/// one is out of scope here). A real value must be present for every
/// register in `shard.inputs` and `shard.outputs`.
pub type Witness = HashMap<Register, Vec<I18>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardProof {
    pub shard_id: usize,
    pub input_commitments: HashMap<Register, Commitment>,
    pub output_commitments: HashMap<Register, Commitment>,
    /// Whether the shard's own proof is valid. Always `true` for
    /// `MockProver`; a real STARK-backed `Prover` would set this from
    /// actual proof verification.
    pub valid: bool,
}

/// Proves (or, for now, mocks proving) one `Shard`. Pluggable so a real
/// STARK-backed implementation can later replace `MockProver` without any
/// change to `Dag`/`Linker`.
pub trait Prover {
    fn prove(&self, shard: &Shard, witness: &Witness) -> ShardProof;
}

/// Computes real commitments over the supplied witness values, but never
/// generates an actual proof -- `valid` is always `true`. See module docs.
pub struct MockProver;

impl Prover for MockProver {
    fn prove(&self, shard: &Shard, witness: &Witness) -> ShardProof {
        let commit_all = |registers: &[Register]| -> HashMap<Register, Commitment> {
            registers
                .iter()
                .map(|register| {
                    let values = witness
                        .get(register)
                        .unwrap_or_else(|| panic!("missing witness value for {register:?}"));
                    (register.clone(), Commitment::of_i18_values(values))
                })
                .collect()
        };

        ShardProof {
            shard_id: shard.id,
            input_commitments: commit_all(&shard.inputs),
            output_commitments: commit_all(&shard.outputs),
            valid: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commitment_is_deterministic_and_content_sensitive() {
        let a = Commitment::of_i18_values(&[
            I18::from_f64(1.0).unwrap(),
            I18::from_f64(2.0).unwrap(),
        ]);
        let a_again = Commitment::of_i18_values(&[
            I18::from_f64(1.0).unwrap(),
            I18::from_f64(2.0).unwrap(),
        ]);
        let b = Commitment::of_i18_values(&[
            I18::from_f64(1.0).unwrap(),
            I18::from_f64(3.0).unwrap(),
        ]);

        assert_eq!(a, a_again);
        assert_ne!(a, b);
    }

    #[test]
    fn mock_prover_commits_declared_inputs_and_outputs() {
        let shard = Shard {
            id: 0,
            name: "s0".into(),
            range: 0..1,
            inputs: vec![Register::GraphInput("x".into())],
            outputs: vec![Register::Virtual(0)],
        };
        let mut witness: Witness = HashMap::new();
        witness.insert(
            Register::GraphInput("x".into()),
            vec![I18::from_f64(1.0).unwrap()],
        );
        witness.insert(Register::Virtual(0), vec![I18::from_f64(2.0).unwrap()]);

        let proof = MockProver.prove(&shard, &witness);

        assert!(proof.valid);
        assert_eq!(proof.shard_id, 0);
        assert!(proof
            .input_commitments
            .contains_key(&Register::GraphInput("x".into())));
        assert!(proof.output_commitments.contains_key(&Register::Virtual(0)));
    }

    #[test]
    #[should_panic(expected = "missing witness value")]
    fn missing_witness_value_panics_with_a_clear_message() {
        let shard = Shard {
            id: 0,
            name: "s0".into(),
            range: 0..1,
            inputs: vec![],
            outputs: vec![Register::Virtual(0)],
        };
        let witness: Witness = HashMap::new();

        MockProver.prove(&shard, &witness);
    }
}
```

Modify `crates/zkie-compiler/src/dag/mod.rs` to:

```rust
//! Generic, model-agnostic DAG of independently-provable "shards" over a
//! `CompiledProgram`, plus a pluggable `Prover` and a
//! commitment-consistency `Linker`. See
//! `docs/superpowers/specs/2026-07-27-zkie-dag-sharding-aggregation-design.md`.

pub mod model;
pub mod prover;

pub use model::{build_dag, BuildDagError, Dag, Edge, EdgeKind, Shard, ShardSpec};
pub use prover::{Commitment, MockProver, Prover, ShardProof, Witness};
```

- [ ] **Step 3: Run the new tests, expect them to pass**

Run: `cargo test -p zkie-compiler dag::prover::tests -- --nocapture`
Expected: all 3 tests pass.

- [ ] **Step 4: Run clippy and fmt**

Run: `cargo clippy -p zkie-compiler --all-targets -- -D warnings && cargo fmt -p zkie-compiler -- --check`

- [ ] **Step 5: Commit**

```bash
git add crates/zkie-compiler/Cargo.toml crates/zkie-compiler/src/dag
git commit -m "feat: add Prover trait, MockProver, and BLAKE3 commitments"
```

---

### Task 4: `Linker` (commitment-consistency verification)

**Files:**
- Create: `crates/zkie-compiler/src/dag/linker.rs`
- Modify: `crates/zkie-compiler/src/dag/mod.rs` (add `pub mod linker;` and re-exports)

**Interfaces:**
- Consumes: `Dag`, `EdgeKind` from Task 1; `ShardProof`, `Commitment` from Task 3.
- Produces: `zkie_compiler::dag::{link, LinkError}` — used by Task 6 (integration test).
  - `pub fn link(dag: &Dag, proofs: &[ShardProof]) -> Result<(), LinkError>` (`proofs[i]` must be the proof for `dag.shards[i]`)
  - `pub enum LinkError { InvalidShard { shard_id: usize }, CommitmentMismatch { producer: usize, consumer: usize, register: Register }, MissingProof { shard_id: usize } }`

- [ ] **Step 1: Write the failing tests**

Create `crates/zkie-compiler/src/dag/linker.rs`:

```rust
//! Verifies a set of per-shard `ShardProof`s against a `Dag`: every
//! shard's own validity, and every cross-shard edge's commitment
//! consistency. This is a plain consistency check, not a succinct proof --
//! see the design doc section 7 for turning this into a real
//! recursive/aggregation SNARK as follow-up work.

use std::fmt;

use super::model::Dag;
use super::prover::{Commitment, ShardProof};
use crate::graph_compiler::Register;

#[derive(Debug, PartialEq, Eq)]
pub enum LinkError {
    /// A shard's own proof was not valid (see `ShardProof::valid`).
    InvalidShard { shard_id: usize },
    /// The producer's committed output for `register` does not match the
    /// consumer's committed input for the same register.
    CommitmentMismatch {
        producer: usize,
        consumer: usize,
        register: Register,
    },
    /// A `ShardProof` for some shard referenced by `dag` was not supplied
    /// to `link` (caller error: `proofs` must contain exactly one entry
    /// per `dag.shards`, indexed by `shard.id`).
    MissingProof { shard_id: usize },
}

impl fmt::Display for LinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinkError::InvalidShard { shard_id } => {
                write!(f, "shard {shard_id}'s proof is not valid")
            }
            LinkError::CommitmentMismatch {
                producer,
                consumer,
                register,
            } => write!(
                f,
                "commitment mismatch on {register:?}: producer shard {producer}'s output commitment does not match consumer shard {consumer}'s input commitment"
            ),
            LinkError::MissingProof { shard_id } => {
                write!(f, "no proof supplied for shard {shard_id}")
            }
        }
    }
}

impl std::error::Error for LinkError {}

/// Verifies `proofs` against `dag`: every shard's own `valid` flag, and
/// every cross-shard edge's producer-output vs. consumer-input commitment
/// consistency (identical check for `Sequential` and `Broadcast` edges --
/// see `EdgeKind`'s doc comment).
///
/// `proofs[i]` must be the `ShardProof` for `dag.shards[i]`.
pub fn link(dag: &Dag, proofs: &[ShardProof]) -> Result<(), LinkError> {
    let get_proof = |shard_id: usize| -> Result<&ShardProof, LinkError> {
        proofs
            .get(shard_id)
            .ok_or(LinkError::MissingProof { shard_id })
    };

    for shard in &dag.shards {
        let proof = get_proof(shard.id)?;
        if !proof.valid {
            return Err(LinkError::InvalidShard { shard_id: shard.id });
        }
    }

    for edge in &dag.edges {
        let producer_proof = get_proof(edge.producer)?;
        let consumer_proof = get_proof(edge.consumer)?;

        let produced: &Commitment = producer_proof
            .output_commitments
            .get(&edge.register)
            .unwrap_or_else(|| {
                panic!(
                    "producer shard {} has no output commitment for {:?} (Prover implementation bug)",
                    edge.producer, edge.register
                )
            });
        let consumed: &Commitment = consumer_proof
            .input_commitments
            .get(&edge.register)
            .unwrap_or_else(|| {
                panic!(
                    "consumer shard {} has no input commitment for {:?} (Prover implementation bug)",
                    edge.consumer, edge.register
                )
            });

        if produced != consumed {
            return Err(LinkError::CommitmentMismatch {
                producer: edge.producer,
                consumer: edge.consumer,
                register: edge.register.clone(),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::model::{build_dag, ShardSpec};
    use crate::dag::prover::{MockProver, Prover, Witness};
    use crate::graph_compiler::{CompiledInstruction, CompiledProgram};
    use std::collections::HashMap;
    use zkie_core::fixed_point::I18;
    use zkie_core::isa::{EltwiseOp, Instruction};

    fn add_instr(inputs: Vec<Register>, output_name: &str) -> CompiledInstruction {
        CompiledInstruction {
            instruction: Instruction::Eltwise { op: EltwiseOp::Add },
            inputs,
            output_name: output_name.to_string(),
        }
    }

    /// instr0 (shard0) -> instr1 (shard1), a plain hand-off, plus a
    /// matching witness map with correct values.
    fn linear_chain_fixture() -> (Dag, Witness) {
        let program = CompiledProgram {
            instructions: vec![
                add_instr(vec![Register::GraphInput("x".into())], "a"),
                add_instr(vec![Register::Virtual(0)], "b"),
            ],
            weights: Default::default(),
            graph_inputs: vec![],
            graph_outputs: vec![],
        };
        let specs = vec![
            ShardSpec { name: "s0".into(), range: 0..1 },
            ShardSpec { name: "s1".into(), range: 1..2 },
        ];
        let dag = build_dag(&program, &specs).expect("valid specs");

        let mut witness: Witness = HashMap::new();
        witness.insert(
            Register::GraphInput("x".into()),
            vec![I18::from_f64(1.0).unwrap()],
        );
        witness.insert(Register::Virtual(0), vec![I18::from_f64(2.0).unwrap()]);

        (dag, witness)
    }

    #[test]
    fn consistent_proofs_link_successfully() {
        let (dag, witness) = linear_chain_fixture();
        let proofs: Vec<_> = dag
            .shards
            .iter()
            .map(|s| MockProver.prove(s, &witness))
            .collect();

        assert_eq!(link(&dag, &proofs), Ok(()));
    }

    #[test]
    fn corrupted_output_commitment_is_caught_precisely() {
        let (dag, witness) = linear_chain_fixture();
        let mut proofs: Vec<_> = dag
            .shards
            .iter()
            .map(|s| MockProver.prove(s, &witness))
            .collect();

        proofs[0]
            .output_commitments
            .insert(Register::Virtual(0), Commitment([0xAA; 32]));

        let result = link(&dag, &proofs);
        assert_eq!(
            result,
            Err(LinkError::CommitmentMismatch {
                producer: 0,
                consumer: 1,
                register: Register::Virtual(0),
            })
        );
    }

    #[test]
    fn invalid_shard_proof_is_rejected() {
        let (dag, witness) = linear_chain_fixture();
        let mut proofs: Vec<_> = dag
            .shards
            .iter()
            .map(|s| MockProver.prove(s, &witness))
            .collect();
        proofs[1].valid = false;

        assert_eq!(
            link(&dag, &proofs),
            Err(LinkError::InvalidShard { shard_id: 1 })
        );
    }

    #[test]
    fn broadcast_edge_corruption_flags_only_the_broken_consumer() {
        let program = CompiledProgram {
            instructions: vec![
                add_instr(vec![], "mask"),
                add_instr(vec![Register::Virtual(0)], "layer0_out"),
                add_instr(vec![Register::Virtual(0)], "layer1_out"),
            ],
            weights: Default::default(),
            graph_inputs: vec![],
            graph_outputs: vec![],
        };
        let specs = vec![
            ShardSpec { name: "prologue".into(), range: 0..1 },
            ShardSpec { name: "layer0".into(), range: 1..2 },
            ShardSpec { name: "layer1".into(), range: 2..3 },
        ];
        let dag = build_dag(&program, &specs).expect("valid specs");

        let mut witness: Witness = HashMap::new();
        witness.insert(Register::Virtual(0), vec![I18::from_f64(9.0).unwrap()]);

        let proofs: Vec<_> = dag
            .shards
            .iter()
            .map(|s| MockProver.prove(s, &witness))
            .collect();
        assert_eq!(link(&dag, &proofs), Ok(()));

        let mut broken_proofs = proofs;
        broken_proofs[2]
            .input_commitments
            .insert(Register::Virtual(0), Commitment([0xBB; 32]));

        assert_eq!(
            link(&dag, &broken_proofs),
            Err(LinkError::CommitmentMismatch {
                producer: 0,
                consumer: 2,
                register: Register::Virtual(0),
            })
        );
    }
}
```

Modify `crates/zkie-compiler/src/dag/mod.rs` to:

```rust
//! Generic, model-agnostic DAG of independently-provable "shards" over a
//! `CompiledProgram`, plus a pluggable `Prover` and a
//! commitment-consistency `Linker`. See
//! `docs/superpowers/specs/2026-07-27-zkie-dag-sharding-aggregation-design.md`.

pub mod linker;
pub mod model;
pub mod prover;

pub use linker::{link, LinkError};
pub use model::{build_dag, BuildDagError, Dag, Edge, EdgeKind, Shard, ShardSpec};
pub use prover::{Commitment, MockProver, Prover, ShardProof, Witness};
```

- [ ] **Step 2: Run the new tests, expect them to pass**

Run: `cargo test -p zkie-compiler dag::linker::tests -- --nocapture`
Expected: all 4 tests pass.

- [ ] **Step 3: Run the whole crate's test suite**

Run: `cargo test -p zkie-compiler`
Expected: all existing tests plus every `dag::*` test pass (nothing in `graph_compiler.rs`/`circuit_binding.rs`/etc. should be affected — this task only added new files and one new `pub mod` line).

- [ ] **Step 4: Run clippy and fmt**

Run: `cargo clippy -p zkie-compiler --all-targets -- -D warnings && cargo fmt -p zkie-compiler -- --check`

- [ ] **Step 5: Commit**

```bash
git add crates/zkie-compiler/src/dag
git commit -m "feat: add Linker commitment-consistency verification"
```

---

### Task 5: Scaffold `engines/timesfm` and load `partition.toml`

**Files:**
- Modify: `/Users/jimmyshi/code/zkie/Cargo.toml` (add `engines/timesfm` to workspace `members`)
- Create: `engines/timesfm/Cargo.toml`
- Create: `engines/timesfm/src/lib.rs`
- Create: `engines/timesfm/src/partition.rs`
- Create: `engines/timesfm/tests/fixtures/synthetic_partition.toml`

**Interfaces:**
- Consumes: `zkie_compiler::dag::ShardSpec` from Task 1.
- Produces: `zkie_ie_timesfm::partition::{load_partition_file, PartitionError}` — used by Task 6.
  - `pub fn load_partition_file(path: &std::path::Path) -> Result<Vec<ShardSpec>, PartitionError>`

- [ ] **Step 1: Add the new workspace member**

Modify `/Users/jimmyshi/code/zkie/Cargo.toml`:

```toml
[workspace]
resolver = "2"
members = ["crates/zkie-core", "crates/zkie-compiler", "engines/timesfm"]
```

- [ ] **Step 2: Create the crate's `Cargo.toml`**

Create `engines/timesfm/Cargo.toml`:

```toml
[package]
name = "zkie-ie-timesfm"
version = "0.1.0"
edition = "2021"

[dependencies]
zkie-core = { path = "../../crates/zkie-core" }
zkie-compiler = { path = "../../crates/zkie-compiler" }
serde = { version = "1", features = ["derive"] }
toml = "0.8"

[dev-dependencies]
rayon = "1"
```

- [ ] **Step 3: Write the failing tests**

Create the fixture directory and file `engines/timesfm/tests/fixtures/synthetic_partition.toml`:

```toml
# NOT the real TimesFM 200M partition file -- op_mapper coverage gaps
# (Split, Sub, and prologue-specific ops like Where/Cast/ArgMax) currently
# block compiling the real ONNX graph end-to-end (see the design doc,
# section on scope). This fixture matches the synthetic, TimesFM-shaped
# `CompiledProgram` built by `zkie_ie_timesfm::fixtures::synthetic_program`
# (see src/fixtures.rs) -- it exists to exercise the Dag/Prover/Linker
# machinery with realistic Sequential + Broadcast edge shapes (a mask
# broadcast to every layer, denormalization stats broadcast only to the
# epilogue), not to prove the real model.

[[shard]]
name = "prologue"
start = 0
end = 3

[[shard]]
name = "layer_0"
group = "layer"
start = 3
end = 13

[[shard]]
name = "layer_1"
group = "layer"
start = 13
end = 23

[[shard]]
name = "layer_2"
group = "layer"
start = 23
end = 33

[[shard]]
name = "epilogue"
start = 33
end = 34
```

Create `engines/timesfm/src/partition.rs`:

```rust
//! Loads a model's shard-boundary definition from a human-reviewed static
//! TOML file. See
//! `docs/superpowers/specs/2026-07-27-zkie-dag-sharding-aggregation-design.md`
//! section 4 for why this is a static file, not a runtime detection
//! algorithm: instruction indices only depend on the model's architecture
//! and the compiler/exporter code, not on trained weight values, so they
//! only need to be regenerated (and re-reviewed) when that code changes --
//! not on every re-run.

use std::fmt;
use std::fs;
use std::path::Path;

use serde::Deserialize;
use zkie_compiler::dag::ShardSpec;

#[derive(Debug, Deserialize)]
struct PartitionFile {
    shard: Vec<ShardDef>,
}

#[derive(Debug, Deserialize)]
struct ShardDef {
    name: String,
    /// Metadata for tooling/readability only (e.g. "this shard is one of
    /// the repeated layers") -- not consumed by `build_dag`.
    #[serde(default)]
    #[allow(dead_code)]
    group: Option<String>,
    start: usize,
    end: usize,
}

#[derive(Debug)]
pub enum PartitionError {
    Io(std::io::Error),
    Toml(toml::de::Error),
}

impl fmt::Display for PartitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PartitionError::Io(e) => write!(f, "failed to read partition file: {e}"),
            PartitionError::Toml(e) => write!(f, "failed to parse partition file: {e}"),
        }
    }
}

impl std::error::Error for PartitionError {}

impl From<std::io::Error> for PartitionError {
    fn from(e: std::io::Error) -> Self {
        PartitionError::Io(e)
    }
}

impl From<toml::de::Error> for PartitionError {
    fn from(e: toml::de::Error) -> Self {
        PartitionError::Toml(e)
    }
}

/// Loads a `partition.toml`-shaped file into the `ShardSpec`s
/// `zkie_compiler::dag::build_dag` expects, in file order.
pub fn load_partition_file(path: &Path) -> Result<Vec<ShardSpec>, PartitionError> {
    let raw = fs::read_to_string(path)?;
    let parsed: PartitionFile = toml::from_str(&raw)?;
    Ok(parsed
        .shard
        .into_iter()
        .map(|def| ShardSpec {
            name: def.name,
            range: def.start..def.end,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_path() -> &'static Path {
        Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/synthetic_partition.toml"
        ))
    }

    #[test]
    fn loads_fixture_partition_file() {
        let specs = load_partition_file(fixture_path()).expect("fixture should parse");

        assert_eq!(specs.len(), 5);
        assert_eq!(specs[0].name, "prologue");
        assert_eq!(specs[0].range, 0..3);
        assert_eq!(specs[1].name, "layer_0");
        assert_eq!(specs[1].range, 3..13);
        assert_eq!(specs[4].name, "epilogue");
        assert_eq!(specs[4].range, 33..34);
    }

    #[test]
    fn missing_file_is_a_typed_io_error() {
        let result = load_partition_file(Path::new("/does/not/exist.toml"));
        assert!(matches!(result, Err(PartitionError::Io(_))));
    }

    #[test]
    fn malformed_toml_is_a_typed_parse_error() {
        let result: Result<PartitionFile, _> = toml::from_str("this is not [[ valid toml");
        assert!(result.is_err());
    }
}
```

Create `engines/timesfm/src/lib.rs`:

```rust
pub mod partition;
```

- [ ] **Step 4: Run the new tests, expect them to pass**

Run: `cd /Users/jimmyshi/code/zkie && cargo build --workspace && cargo test -p zkie-ie-timesfm`
Expected: crate builds, 3 tests pass.

- [ ] **Step 5: Run clippy and fmt**

Run: `cargo clippy -p zkie-ie-timesfm --all-targets -- -D warnings && cargo fmt -p zkie-ie-timesfm -- --check`

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml engines/timesfm
git commit -m "feat: scaffold zkie-ie-timesfm crate with partition.toml loader"
```

---

### Task 6: Synthetic TimesFM-shaped fixture + end-to-end integration test

**Files:**
- Create: `engines/timesfm/src/fixtures.rs`
- Modify: `engines/timesfm/src/lib.rs` (add `pub mod fixtures;`)
- Create: `engines/timesfm/tests/end_to_end_synthetic.rs`

**Interfaces:**
- Consumes: `zkie_compiler::graph_compiler::{CompiledProgram, CompiledInstruction, Register}`, `zkie_compiler::dag::{build_dag, link, MockProver, Prover, Commitment}`, `zkie_ie_timesfm::partition::load_partition_file`.
- Produces: `zkie_ie_timesfm::fixtures::{synthetic_program, synthetic_witness}` — used by Task 6's own test and Task 7's example binary.
  - `pub fn synthetic_program() -> CompiledProgram`
  - `pub fn synthetic_witness() -> HashMap<Register, Vec<I18>>`

- [ ] **Step 1: Write the fixture module**

Create `engines/timesfm/src/fixtures.rs`:

```rust
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
        instruction: Instruction::RmsNorm { dim: 4, epsilon_milli: 1 },
        inputs: vec![hidden_state_in.clone(), weight(&format!("{prefix}_rmsnorm_w"))],
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
        vec![Register::Virtual(base + 3), weight(&format!("{prefix}_v_w"))],
        &format!("{prefix}_attn_out"),
    )); // base + 4
    instructions.push(eltwise(
        EltwiseOp::Add,
        vec![Register::Virtual(base + 4), hidden_state_in],
        &format!("{prefix}_attn_residual"),
    )); // base + 5
    instructions.push(CompiledInstruction {
        instruction: Instruction::LayerNorm { dim: 4, epsilon_milli: 1 },
        inputs: vec![Register::Virtual(base + 5), weight(&format!("{prefix}_ln_w"))],
        output_name: format!("{prefix}_ln_out"),
    }); // base + 6
    instructions.push(dot_general(
        vec![Register::Virtual(base + 6), weight(&format!("{prefix}_ffn_w1"))],
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
        vec![Register::GraphInput("input_padding".into()), weight("mask_w")],
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
        graph_outputs: vec![("output_ts".into(), Register::Virtual(TOTAL_INSTRUCTIONS - 1))],
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
```

- [ ] **Step 2: Wire the module into the crate**

Modify `engines/timesfm/src/lib.rs`:

```rust
pub mod fixtures;
pub mod partition;
```

- [ ] **Step 3: Run the fixture's own tests, expect them to pass**

Run: `cargo test -p zkie-ie-timesfm fixtures:: -- --nocapture`
Expected: both tests pass.

- [ ] **Step 4: Write the failing end-to-end integration test**

Create `engines/timesfm/tests/end_to_end_synthetic.rs`:

```rust
use std::path::Path;

use rayon::prelude::*;
use zkie_compiler::dag::{build_dag, link, Commitment, MockProver, Prover};
use zkie_ie_timesfm::{fixtures, partition};

fn fixture_partition_path() -> &'static Path {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/synthetic_partition.toml"
    ))
}

#[test]
fn synthetic_timesfm_shaped_program_links_successfully() {
    let program = fixtures::synthetic_program();
    let specs = partition::load_partition_file(fixture_partition_path()).expect("fixture parses");
    let dag = build_dag(&program, &specs).expect("valid partition");

    assert_eq!(dag.shards.len(), 5);
    // 4 sequential hand-offs (prologue->layer0->layer1->layer2->epilogue)
    // + 3 mask broadcast edges (prologue->each layer) + 1 denorm-stats
    // broadcast edge (prologue->epilogue) = 8 edges.
    assert_eq!(dag.edges.len(), 8);

    let witness = fixtures::synthetic_witness();
    // Every shard's proof is independent of every other's -- prove them
    // all in parallel, demonstrating the whole point of sharding.
    let proofs: Vec<_> = dag
        .shards
        .par_iter()
        .map(|shard| MockProver.prove(shard, &witness))
        .collect();

    assert_eq!(link(&dag, &proofs), Ok(()));
}

#[test]
fn corrupting_one_layer_shard_output_is_caught_by_link() {
    let program = fixtures::synthetic_program();
    let specs = partition::load_partition_file(fixture_partition_path()).expect("fixture parses");
    let dag = build_dag(&program, &specs).expect("valid partition");
    let witness = fixtures::synthetic_witness();

    let mut proofs: Vec<_> = dag
        .shards
        .iter()
        .map(|shard| MockProver.prove(shard, &witness))
        .collect();

    // layer_1 is shard id 2 (prologue=0, layer_0=1, layer_1=2, layer_2=3,
    // epilogue=4); corrupt its one declared output commitment.
    let layer1_output = dag.shards[2].outputs[0].clone();
    proofs[2]
        .output_commitments
        .insert(layer1_output, Commitment([0xFF; 32]));

    assert!(link(&dag, &proofs).is_err());
}
```

- [ ] **Step 5: Run the integration test, expect it to pass**

Run: `cargo test -p zkie-ie-timesfm --test end_to_end_synthetic -- --nocapture`
Expected: both tests pass.

- [ ] **Step 6: Run the whole crate's test suite**

Run: `cargo test -p zkie-ie-timesfm`
Expected: all tests (partition, fixtures, end_to_end_synthetic) pass.

- [ ] **Step 7: Run clippy and fmt**

Run: `cargo clippy -p zkie-ie-timesfm --all-targets -- -D warnings && cargo fmt -p zkie-ie-timesfm -- --check`

- [ ] **Step 8: Commit**

```bash
git add engines/timesfm
git commit -m "test: add synthetic TimesFM-shaped fixture and end-to-end DAG/proving/linking test"
```

---

### Task 7: Draft shard-boundary generator (example binary)

**Files:**
- Create: `engines/timesfm/examples/draft_partition.rs`

**Interfaces:**
- Consumes: `zkie_ie_timesfm::fixtures::synthetic_program`, `zkie_core::isa::Instruction`.
- Produces: nothing consumed by later tasks — this is a standalone, manually-run convenience tool (not part of `cargo test`, matching the existing `crates/zkie-core/examples/bench_rowcount.rs` convention).

- [ ] **Step 1: Write the example binary**

Create `engines/timesfm/examples/draft_partition.rs`:

```rust
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
        println!("epilogue: {}..{}", last_layer_end, program.instructions.len());
    }
}
```

- [ ] **Step 2: Run the example and verify its output**

Run: `cargo run -p zkie-ie-timesfm --example draft_partition`
Expected output (exactly matches `tests/fixtures/synthetic_partition.toml`'s boundaries):

```
found 3 RmsNorm anchor(s) at instruction indices: [3, 13, 23]
(this is a DRAFT only -- a human must review/adjust before writing partition.toml)

prologue: 0..3
layer_0: 3..13
layer_1: 13..23
layer_2: 23..33
epilogue: 33..34
```

If the printed boundaries don't match, fix the example's logic before proceeding (this is the "does the draft tool actually work" check called for in the design doc — there is no automated test for this example, matching how `bench_rowcount.rs` is also manually run, not unit-tested).

- [ ] **Step 3: Run clippy and fmt**

Run: `cargo clippy -p zkie-ie-timesfm --all-targets -- -D warnings && cargo fmt -p zkie-ie-timesfm -- --check`

- [ ] **Step 4: Commit**

```bash
git add engines/timesfm/examples
git commit -m "feat: add draft_partition example for shard-boundary drafting"
```

---

### Task 8: Final workspace-wide verification

**Files:** none (verification only)

**Interfaces:** none

- [ ] **Step 1: Full workspace build**

Run: `cd /Users/jimmyshi/code/zkie && cargo build --workspace`
Expected: succeeds with no errors.

- [ ] **Step 2: Full workspace test suite**

Run: `cargo test --workspace`
Expected: every test across `zkie-core`, `zkie-compiler`, and `zkie-ie-timesfm` passes, including every test added in Tasks 1-6.

- [ ] **Step 3: Full workspace clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: no warnings. If any surface (e.g. from interactions between crates not caught by per-crate runs in earlier tasks), fix them now.

- [ ] **Step 4: Full workspace format check**

Run: `cargo fmt --all -- --check`
Expected: no diffs. If any, run `cargo fmt --all` and re-check.

- [ ] **Step 5: Commit any final cleanup**

If Steps 3-4 required fixes:

```bash
git add -A
git commit -m "chore: fix workspace-wide clippy/fmt issues"
```

If no fixes were needed, this task requires no commit — just confirm the working tree is clean (`git status`).
