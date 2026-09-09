# zkIE Automatic Partition and N-ary Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce immutable resource-targeted shard plans, bind every leaf boundary inside Halo2, and combine independently verified leaves into a sound configurable N-ary native aggregation manifest.

**Architecture:** The compiler creates a deterministic contiguous partition plan and derives all cross-shard edges through the existing def-use analysis. Leaf circuits expose circuit-constrained Poseidon boundary commitments. An aggregation planner groups verified claims in topology-aware N-ary levels and closes every DAG edge exactly once before publishing a root manifest.

**Tech Stack:** Rust 2021, PSE Halo2 v0.4, BN254 Poseidon x5 parameters, BLAKE3 artifact/plan digests, serde.

**Spec:** `docs/superpowers/specs/2026-09-09-zkie-multibackend-sharded-proving-design.md`

## Global Constraints

- Complete `2026-09-09-zkie-proof-backend-foundation.md` first.
- Run Cargo with `CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie`.
- `BoundaryCommitment` is circuit constrained; BLAKE3 remains artifact integrity only.
- The partition plan is immutable after creation and included in `run_plan_digest`.
- Scheduler/runtime code may consume a plan but may never change its shard boundaries.
- Aggregation fan-in is public, `2 <= N <= 16`, default 4.
- A native manifest must identify itself as non-succinct and must not claim to be a recursive proof.
- Request a Rust review after every production-code task and a security review for Poseidon/public-statement/aggregation soundness changes.

---

## File Structure

- Create `crates/zkie-compiler/src/dag/partition.rs`: cost model, hints and deterministic planner.
- Modify `crates/zkie-compiler/Cargo.toml`: depend on shared `zkie-types`.
- Modify `crates/zkie-compiler/src/dag/model.rs`: stable edge IDs and plan-linked shard metadata.
- Modify `crates/zkie-compiler/src/dag/mod.rs`: exports.
- Create `crates/zkie-compiler/src/shard_binding.rs`: convert one compiled shard into a local assembler program.
- Modify `crates/zkie-core/Cargo.toml`: Poseidon parameter dependencies.
- Create `crates/zkie-core/src/chips/poseidon_boundary.rs`: native/circuit-compatible two-input Poseidon compression.
- Modify `crates/zkie-core/src/chips/mod.rs` and `crates/zkie-core/src/program_circuit.rs`: expose boundary commitments as instances.
- Create `crates/zkie-prover/src/statement.rs`: canonical leaf and root statements.
- Modify `crates/zkie-prover/src/halo2_cpu.rs`: prove and verify real public instances.
- Create `crates/zkie-prover/src/aggregation.rs`: N-ary planner, claim merge and manifest verification.
- Create `crates/zkie-compiler/tests/automatic_partition.rs`.
- Create `crates/zkie-prover/tests/boundary_soundness.rs`.
- Create `crates/zkie-prover/tests/nary_aggregation.rs`.

### Task 1: Deterministic resource-targeted partition plan

**Files:**
- Create: `crates/zkie-compiler/src/dag/partition.rs`
- Modify: `crates/zkie-compiler/Cargo.toml`
- Modify: `crates/zkie-compiler/src/dag/model.rs`
- Modify: `crates/zkie-compiler/src/dag/mod.rs`
- Test: `crates/zkie-compiler/tests/automatic_partition.rs`

**Interfaces:**
- Consumes: `CompiledProgram`, `Instruction`, existing `build_dag`.
- Produces: `PartitionRequest`, `InstructionEstimate`, `PartitionPlan`, `PartitionPlanner::plan`; uses `zkie_types::Digest32` without depending on `zkie-prover`.

- [ ] **Step 1: Write failing determinism and budget tests**

```rust
#[test]
fn planner_cuts_before_exceeding_target_and_is_deterministic() {
    let program = four_instruction_chain();
    let request = PartitionRequest::new(300, TestCostModel::fixed(100)).unwrap();
    let first = PartitionPlanner::plan(&program, &request).unwrap();
    let second = PartitionPlanner::plan(&program, &request).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.shards.iter().map(|s| s.range.clone()).collect::<Vec<_>>(),
               vec![0..3, 3..4]);
    assert_eq!(first.digest(), second.digest());
}
```

Also test one instruction larger than the target returns `PartitionError::UnsplittableInstruction`, gaps/overlaps are impossible, and a broadcast producer creates every expected edge.

- [ ] **Step 2: Run the test and confirm planner types are missing**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-compiler --test automatic_partition
```

- [ ] **Step 3: Implement exact planning types**

```rust
pub struct PartitionRequest<M> {
    pub target_shard_ram_bytes: u64,
    pub max_k: u32,
    pub cost_model: M,
    pub boundary_hints: Vec<BoundaryHint>,
}

pub trait InstructionCostModel {
    fn estimate_instruction(&self, instruction: &CompiledInstruction)
        -> Result<InstructionEstimate, PartitionError>;
    fn estimate_shard(&self, instructions: &[CompiledInstruction])
        -> Result<ShardEstimate, PartitionError>;
}

pub struct PartitionPlan {
    pub shards: Vec<PlannedShard>,
    pub dag: Dag,
    pub digest: Digest32,
}
```

Use a deterministic greedy pass: accumulate instructions; when the next instruction would exceed the byte target, choose the highest-priority valid boundary hint within the current shard, otherwise cut immediately before the instruction. Reject an empty shard and an individually unsplittable instruction. Re-run `build_dag` and validate every `Virtual` cross-boundary dependency has one producer.

- [ ] **Step 4: Add stable plan encoding**

Encode fixed-width integers little-endian and strings as `u64 length || UTF-8 bytes`; encode shards and edges in sorted ID order. Hash with BLAKE3 into `Digest32`. Add a mutation test showing that changing a range, edge, estimate, model digest or compiler version changes the plan digest.

- [ ] **Step 5: Run compiler tests and commit**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-compiler dag
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-compiler --test automatic_partition
git add crates/zkie-compiler
git commit -m "feat: add deterministic automatic shard planner"
```

### Task 2: Convert a global compiled program into a local shard program

**Files:**
- Create: `crates/zkie-compiler/src/shard_binding.rs`
- Modify: `crates/zkie-compiler/src/lib.rs`
- Test: `crates/zkie-compiler/tests/shard_binding.rs`

**Interfaces:**
- Consumes: `PartitionPlan`, `Shard`, global `CompiledProgram`, boundary register values.
- Produces: `BoundShardProgram { assembler_program, input_boundaries, output_boundaries, global_to_local }`.

- [ ] **Step 1: Write failing sequential and broadcast binding tests**

Construct a three-shard program where shard 0 broadcasts one register to shards 1 and 2. Assert each consumer maps that external virtual register to a local `RegisterRef::Input`, internal virtual indices start at zero, weights stay weights, and both consumers retain the same global edge/register identity.

- [ ] **Step 2: Run and confirm the adapter is absent**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-compiler --test shard_binding
```

- [ ] **Step 3: Implement local remapping**

```rust
pub fn bind_shard_program(
    program: &CompiledProgram,
    shard: &Shard,
    graph_inputs: &HashMap<String, Vec<I18>>,
    incoming: &HashMap<Register, Vec<I18>>,
) -> Result<BoundShardProgram, ShardBindingError>;
```

Map every external graph input and incoming cross-shard virtual value to a stable local input index sorted by canonical register ID. Map global virtual index `i` inside `shard.range` to `RegisterRef::Virtual(i - shard.range.start)`. Reject a missing boundary value, reference to a later shard and output register outside the shard.

- [ ] **Step 4: Cross-check against unsplit execution**

Run the supported synthetic program once unsplit and then shard-by-shard, passing only declared boundary outputs. Assert every final raw `I18` output is identical.

- [ ] **Step 5: Run and commit**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-compiler --test shard_binding
git add crates/zkie-compiler
git commit -m "feat: bind compiled programs into local shards"
```

### Task 3: Circuit-constrained boundary commitment

**Files:**
- Modify: `crates/zkie-core/Cargo.toml`
- Create: `crates/zkie-core/src/chips/poseidon_boundary.rs`
- Modify: `crates/zkie-core/src/chips/mod.rs`
- Modify: `crates/zkie-core/src/program_circuit.rs`
- Create: `crates/zkie-prover/src/statement.rs`
- Modify: `crates/zkie-prover/src/lib.rs`
- Test: `crates/zkie-prover/tests/boundary_soundness.rs`

**Interfaces:**
- Consumes: `AssignedProgram`, canonical tensor metadata, Halo2 `Fr` cells.
- Produces: `BoundaryCommitment`, `BoundaryDescriptor`, `LeafStatement`, public instance vector.

- [ ] **Step 1: Add native known-answer and circuit-equality tests**

Add `light-poseidon = "0.4.0"`, `ark-bn254 = "0.5.0"` and `ark-ff = "0.5.0"` as the parameter-source dependencies for Circom-compatible BN254 x5 Poseidon. Add a known-answer test for two inputs and a `MockProver` test showing the circuit output equals the native result. Add negative tests for changed role, dtype, shape, quantization scale, tensor length and value.

- [ ] **Step 2: Run and observe missing commitment support**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-prover --test boundary_soundness
```

- [ ] **Step 3: Implement one canonical hash chain**

Define the commitment as repeated two-input Poseidon compression:

```text
state_0 = protocol_domain_field
state_1 = Poseidon(state_0, tensor_role)
state_2 = Poseidon(state_1, dtype)
state_3 = Poseidon(state_2, rank)
state_* = Poseidon(state, each dimension)
state_* = Poseidon(state, quantization scale)
state_* = Poseidon(state, element count)
commitment = fold(Poseidon(state, canonical_field(raw_i18)))
```

Convert the published `light-poseidon` round constants and MDS entries into `halo2curves::bn256::Fr` using canonical little-endian field representations. Constrain every x5 S-box and MDS round inside `PoseidonBoundaryChip`; do not accept a host-computed hash as advice without round constraints.

- [ ] **Step 4: Bind exact assembler cells to public instances**

`AssemblerCircuit` selects declared input/output tensors from `AssignedProgram`, hashes their existing constrained cells, and exposes commitments through one instance column. `LeafStatement::instances()` produces the exact same ordered vector. Include shard ID, circuit digest, partition digest, model/weights digest and proof flavor before boundary commitments. Encode each 32-byte digest as two unsigned 16-byte little-endian limbs so conversion into BN254 `Fr` is injective; encode string IDs by first hashing their canonical length-prefixed UTF-8 bytes to `Digest32` and applying the same two-limb rule.

- [ ] **Step 5: Prove tampering fails**

Generate a valid real KZG leaf proof, then verify it against a statement with one altered input commitment and separately one altered output commitment. Both must return cryptographic verification failure. A correct BLAKE3 digest over the altered metadata must not make either proof valid.

- [ ] **Step 6: Run and commit**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-prover --test boundary_soundness -- --nocapture
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-core poseidon_boundary
git add crates/zkie-core crates/zkie-prover Cargo.lock
git commit -m "feat: constrain shard boundary commitments"
```

### Task 4: Configurable N-ary aggregation plan

**Files:**
- Create: `crates/zkie-prover/src/aggregation.rs`
- Modify: `crates/zkie-prover/src/lib.rs`
- Test: `crates/zkie-prover/tests/nary_aggregation.rs`

**Interfaces:**
- Consumes: verified leaf proofs, `PartitionPlan`, `LeafStatement`, fan-in N.
- Produces: `AggregationPlan`, `AggregationClaim`, `VerifiedManifest`.

- [ ] **Step 1: Write failing plan-shape tests**

For 10 leaves, assert N=4 produces groups `[4, 4, 2]` at level one and one root group of 3. Assert N=2 and N=8 produce different plan digests. Assert N=1 and N=17 return `AggregationError::InvalidFanIn`.

- [ ] **Step 2: Run and confirm aggregation types are missing**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-prover --test nary_aggregation
```

- [ ] **Step 3: Implement deterministic topology-aware grouping**

```rust
pub fn plan_aggregation(
    partition: &PartitionPlan,
    fan_in: NonZeroU8,
) -> Result<AggregationPlan, AggregationError>;

pub fn merge_verified_claims(
    node: &AggregationNode,
    children: &[VerifiedClaim],
    dag: &Dag,
) -> Result<VerifiedClaim, AggregationError>;
```

Order leaves by topological shard order, preserve model block groups when they fit, and never use dummy children. Each node stores ordered child IDs and actual arity. Encode N, actual arity and node topology in the plan digest.

- [ ] **Step 4: Close cross-shard edges exactly once**

Each claim carries covered shard IDs plus frontier commitments keyed by stable edge/register identity. When producer and consumer first coexist in one node, require exact equality and move the edge to `closed_edges`. Reject duplicate closing, missing boundaries, an edge closed without both endpoints, and a root with any internal edge still open.

- [ ] **Step 5: Publish a clearly typed native manifest**

`VerifiedManifest` is constructible only after all child `VerifiedProof`s pass independent backend verification and the root claim closes every internal edge. Serialize:

```rust
pub enum FinalArtifactKind {
    NativeVerifiedManifest,
    RecursiveRootProof,
}
```

The phase-one implementation always emits `NativeVerifiedManifest`; verification APIs reject callers that request a recursive proof from it.

- [ ] **Step 6: Add adversarial tests**

Cover altered downstream commitment, sibling substitution, child reordering, changed N, changed model/weights/partition/flavor/VK digest, one corrupted broadcast consumer and one missing leaf. Assert exact typed errors identify node, edge and shard.

- [ ] **Step 7: Run and commit**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-prover --test nary_aggregation
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test --workspace
git add crates/zkie-prover crates/zkie-compiler
git commit -m "feat: add sound N-ary aggregation manifests"
```

### Task 5: End-to-end supported-model proof graph

**Files:**
- Create: `crates/zkie-prover/tests/sharded_kzg_pipeline.rs`
- Modify: `engines/timesfm/tests/end_to_end_synthetic.rs`

**Interfaces:**
- Consumes: all preceding tasks in Plans 1 and 2.
- Produces: one executable library-level proof pipeline fixture and measured artifact metadata.

- [ ] **Step 1: Write the end-to-end test**

Compile a supported synthetic ONNX graph containing a broadcast dependency, automatically partition it into at least three shards, generate CPU witnesses, produce and independently verify real KZG leaf proofs, and generate an N=4 native manifest. Assert the root binds graph input/output, closes every DAG edge and records actual arity.

- [ ] **Step 2: Add an invalid-composition variant**

Replace one consumer proof with a valid proof over a different self-consistent input. Assert each leaf verifies independently but aggregation fails with `AggregationError::BoundaryMismatch`.

- [ ] **Step 3: Run the focused pipeline**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-prover --test sharded_kzg_pipeline -- --nocapture
```

- [ ] **Step 4: Run full verification**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test --workspace
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo clippy --workspace --all-targets -- -D warnings
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo fmt --all -- --check
```

- [ ] **Step 5: Commit**

```bash
git add crates/zkie-prover/tests/sharded_kzg_pipeline.rs engines/timesfm/tests/end_to_end_synthetic.rs
git commit -m "test: prove and bind a sharded KZG pipeline"
```
