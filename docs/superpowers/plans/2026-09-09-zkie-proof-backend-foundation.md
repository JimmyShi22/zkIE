# zkIE Proof Backend Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build proof-system-neutral backend contracts, a runtime-configurable assembler circuit, a deterministic CPU witness executor, and a real Halo2 KZG CPU leaf-proof roundtrip.

**Architecture:** Add a dependency-free `zkie-types` crate for cross-layer identities/resources and a `zkie-prover` crate above `zkie-core` and `zkie-compiler`. The core crate owns Halo2 circuit construction, the compiler owns backend-neutral programs, and the prover crate adapts compiled shards into witness and proof jobs without leaking Halo2 types through public orchestration interfaces.

**Tech Stack:** Rust 2021, PSE Halo2 v0.4 with `circuit-params`, BN256 KZG/SHPLONK, BLAKE3 artifact digests, serde, thiserror.

**Spec:** `docs/superpowers/specs/2026-09-09-zkie-multibackend-sharded-proving-design.md`

## Global Constraints

- Run Cargo with `CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie`.
- Do not execute target-directory binaries directly; use `cargo test`, `cargo run`, or an installed executable.
- Keep `execution_backend` separate from `proof_flavor`.
- One run uses one proof flavor; compatible CPU/GPU implementations may share it.
- First production flavor is `halo2-kzg-bn256-shplonk-v1`.
- Backend public types must not expose Halo2 concrete types.
- Existing unsupported ISA variants return typed errors rather than panic or silently skip work.
- Preserve unrelated untracked files and the existing `assembler.rs` column-reuse change.
- Request a Rust review after every production-code task; request a security review for proof identity, key handling and cryptographic verification changes.

---

## File Structure

- Modify `Cargo.toml`: add `crates/zkie-types` and `crates/zkie-prover` to the workspace.
- Create `crates/zkie-types/Cargo.toml` and `src/lib.rs`.
- Create `crates/zkie-types/src/identity.rs`: stable IDs and digests.
- Create `crates/zkie-types/src/resource.rs`: resource quantities and checked arithmetic.
- Modify `crates/zkie-core/Cargo.toml`: enable Halo2 `circuit-params`.
- Modify `crates/zkie-core/src/lib.rs`: export `program_circuit`.
- Modify `crates/zkie-core/src/assembler.rs`: return assigned input/output cells through a new trace API while preserving `assign`.
- Create `crates/zkie-core/src/program_circuit.rs`: runtime-shaped `AssemblerCircuit` using `Circuit::Params`.
- Create `crates/zkie-prover/Cargo.toml`.
- Create `crates/zkie-prover/src/lib.rs`.
- Create `crates/zkie-prover/src/backend.rs`: witness/proof contracts and typed errors.
- Create `crates/zkie-prover/src/witness_cpu.rs`: deterministic supported-ISA executor.
- Create `crates/zkie-prover/src/halo2_cpu.rs`: real KZG setup/prove/verify adapter.
- Create `crates/zkie-prover/tests/backend_contract.rs`.
- Create `crates/zkie-prover/tests/halo2_cpu_roundtrip.rs`.

### Task 1: Stable backend identity and resource types

**Files:**
- Modify: `Cargo.toml`
- Create: `crates/zkie-types/Cargo.toml`
- Create: `crates/zkie-types/src/lib.rs`
- Create: `crates/zkie-types/src/identity.rs`
- Create: `crates/zkie-types/src/resource.rs`
- Create: `crates/zkie-prover/Cargo.toml`
- Create: `crates/zkie-prover/src/lib.rs`
- Test: `crates/zkie-prover/tests/backend_contract.rs`

**Interfaces:**
- Produces: `Digest32`, `ProofFlavorId`, `ExecutionBackendId`, `ModelVisibility`, `RunIdentity`, `HardwareProfile`, `ResourceRequest`, `ResourceCapacity`.
- Consumes: no earlier plan interfaces.

- [ ] **Step 1: Add a failing identity test**

```rust
#[test]
fn execution_backend_does_not_change_proof_flavor() {
    let flavor = ProofFlavorId::parse("halo2-kzg-bn256-shplonk-v1").unwrap();
    let cpu = ExecutionBackendId::parse("halo2-cpu").unwrap();
    let cuda = ExecutionBackendId::parse("halo2-cuda").unwrap();
    assert_ne!(cpu, cuda);
    assert_eq!(flavor.as_str(), "halo2-kzg-bn256-shplonk-v1");
}

#[test]
fn resource_capacity_rejects_overcommit() {
    let capacity = ResourceCapacity::new(56, 400 << 30, 0, 0).unwrap();
    let request = ResourceRequest::new(57, 1 << 30, 0, 0).unwrap();
    assert!(!capacity.fits(&request));
}
```

- [ ] **Step 2: Run the test and confirm the types are absent**

Run:

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-prover --test backend_contract
```

Expected: compilation fails because `zkie_prover` and its types do not exist.

- [ ] **Step 3: Create the crate and exact types**

Implement the following in `zkie-types`, and re-export them from `zkie-prover`: validated non-empty ASCII newtypes for `ProofFlavorId` and `ExecutionBackendId`; `Digest32([u8; 32])` with lowercase hex display/parse; resource values in bytes and logical CPU cores:

```rust
pub struct ResourceRequest {
    pub cpu_cores: u32,
    pub ram_bytes: u64,
    pub gpu_count: u32,
    pub gpu_vram_bytes_per_device: u64,
}

pub struct ResourceCapacity {
    pub cpu_cores: u32,
    pub ram_bytes: u64,
    pub gpu_count: u32,
    pub gpu_vram_bytes_per_device: u64,
}

impl ResourceCapacity {
    pub fn fits(&self, request: &ResourceRequest) -> bool;
    pub fn checked_reserve(&self, request: &ResourceRequest) -> Option<Self>;
}
```

Define `ModelVisibility::{PublicModel, PrivateModel}` and a `RunIdentity` containing model/weights/compiler/ISA/quantization/partition/aggregation digests, proof flavor, visibility, fan-in and public-input schema version. Its canonical digest must change when any field changes. Reject zero CPU, GPU VRAM with zero GPUs, and arithmetic overflow. Add serde derives so the same representations can cross the worker boundary.

- [ ] **Step 4: Run the crate tests**

Run the Task 1 command. Expected: all tests pass.

- [ ] **Step 5: Commit the isolated foundation**

```bash
git add Cargo.toml Cargo.lock crates/zkie-types crates/zkie-prover
git commit -m "feat: define prover backend identities and resources"
```

### Task 2: Runtime-shaped assembler circuit

**Files:**
- Modify: `crates/zkie-core/Cargo.toml`
- Modify: `crates/zkie-core/src/lib.rs`
- Modify: `crates/zkie-core/src/assembler.rs`
- Create: `crates/zkie-core/src/program_circuit.rs`

**Interfaces:**
- Consumes: existing `AssemblerProgram`, `AssemblerChip`, `RsqrtDomain`.
- Produces: `AssemblerCircuitParams`, `AssemblerCircuit`, `AssignedProgram`.

- [ ] **Step 1: Write failing circuit-parameter tests**

Add tests proving two programs with different `DotGeneral.k` values configure through the same concrete `AssemblerCircuit` Rust type and that `without_witnesses()` preserves shape while removing values:

```rust
#[test]
fn runtime_params_configure_distinct_program_shapes() {
    let k2 = AssemblerCircuit::new(dot_program(2), 10, Default::default());
    let k3 = AssemblerCircuit::new(dot_program(3), 10, Default::default());
    MockProver::run(10, &k2, vec![]).unwrap().assert_satisfied();
    MockProver::run(10, &k3, vec![]).unwrap().assert_satisfied();
    assert_ne!(k2.params().shape_digest(), k3.params().shape_digest());
}
```

- [ ] **Step 2: Run the focused test and observe the missing generic circuit**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-core program_circuit
```

Expected: compilation fails because `AssemblerCircuit` is not defined.

- [ ] **Step 3: Enable and implement `Circuit::Params`**

Enable the Halo2 `circuit-params` feature. Define:

```rust
#[derive(Clone, Default)]
pub struct AssemblerCircuitParams {
    pub instructions: Vec<AssemblerInstruction>,
    pub rms_norm_domains: HashMap<(usize, u64), RsqrtDomain>,
}

#[derive(Clone)]
pub struct AssemblerCircuit {
    params: AssemblerCircuitParams,
    program: AssemblerProgram,
    k: u32,
}
```

`params()` returns the exact configuration shape; `configure_with_params()` calls `AssemblerChip::configure_with_rms_norm_domains`; `synthesize()` loads RMSNorm tables only when required and assigns the program. `configure()` must reject accidental use by delegating to the default empty params without panicking.

- [ ] **Step 4: Expose assigned cells without breaking existing callers**

Add:

```rust
pub struct AssignedTensor {
    pub values: Vec<I18>,
    pub cells: Vec<AssignedCell<Fr, Fr>>,
}

pub struct AssignedProgram {
    pub inputs: Vec<AssignedTensor>,
    pub weights: Vec<AssignedTensor>,
    pub virtuals: Vec<AssignedTensor>,
}

pub fn assign_with_cells(
    &self,
    layouter: impl Layouter<Fr>,
    program: &AssemblerProgram,
) -> Result<AssignedProgram, AssemblerError>;
```

Keep `assign()` as a compatibility wrapper returning only virtual values. Ensure every returned cell is the same constrained cell used by downstream instructions, not a re-witnessed copy.

- [ ] **Step 5: Run assembler and circuit tests**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-core assembler::tests
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-core program_circuit
```

Expected: both suites pass, including the existing shared-column tests.

- [ ] **Step 6: Commit**

```bash
git add crates/zkie-core/Cargo.toml crates/zkie-core/src/lib.rs crates/zkie-core/src/assembler.rs crates/zkie-core/src/program_circuit.rs Cargo.lock
git commit -m "feat: add runtime-shaped assembler circuit"
```

### Task 3: Proof-system-neutral backend contracts

**Files:**
- Create: `crates/zkie-prover/src/backend.rs`
- Modify: `crates/zkie-prover/src/lib.rs`
- Modify: `crates/zkie-prover/tests/backend_contract.rs`

**Interfaces:**
- Consumes: Task 1 IDs/resources; compiler `Shard`; `AssemblerProgram`.
- Produces: `WitnessJob`, `WitnessArtifact`, `PrepareJob`, `ProofJob`, `UnverifiedProof`, `VerifiedProof`, `WitnessBackend`, `ProofBackend`, typed backend errors.

- [ ] **Step 1: Add a compile-time contract test with a fake backend**

```rust
struct FakeBackend;

impl ProofBackend for FakeBackend {
    fn capabilities(&self) -> BackendCapabilities { fake_capabilities() }
    fn estimate_resources(&self, _: &ProofJob) -> Result<ResourceRequest, BackendError> {
        Ok(ResourceRequest::new(1, 1 << 30, 0, 0).unwrap())
    }
    fn prepare(&self, _: &PrepareJob, _: &dyn KeyStore) -> Result<PreparedKeys, BackendError> {
        Ok(fake_keys())
    }
    fn prove(&self, _: &ProofJob, _: &Path) -> Result<UnverifiedProof, BackendError> {
        Ok(fake_unverified_proof())
    }
    fn verify(&self, proof: &UnverifiedProof) -> Result<VerifiedProof, VerificationError> {
        VerifiedProof::try_from_unverified(proof.clone())
    }
}
```

- [ ] **Step 2: Run the contract test and confirm it fails on missing interfaces**

Run the Task 1 test command. Expected: missing trait/type errors.

- [ ] **Step 3: Implement contracts with explicit state separation**

`UnverifiedProof` contains proof path/digest, public-statement bytes/digest, circuit digest, VK digest, flavor, backend ID and shard ID. `VerifiedProof` has private fields and is constructible only by a backend verifier. Define `BackendError` variants for unsupported instruction, invalid job, missing key, I/O and proving failure; define `VerificationError` separately so callers cannot convert it into a successful boolean.

Use object-safe trait methods and owned serializable job specs so implementations can run in an independent process later.

- [ ] **Step 4: Add negative tests**

Assert that a mismatched proof flavor, circuit digest, VK digest or artifact digest returns a typed verification error before cryptographic verification is accepted.

- [ ] **Step 5: Run tests and commit**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-prover --test backend_contract
git add crates/zkie-prover
git commit -m "feat: add witness and proof backend contracts"
```

### Task 4: Deterministic CPU witness backend

**Files:**
- Create: `crates/zkie-prover/src/witness_cpu.rs`
- Modify: `crates/zkie-prover/src/lib.rs`
- Create: `crates/zkie-prover/tests/witness_cpu.rs`

**Interfaces:**
- Consumes: `WitnessBackend`, `WitnessJob`, `CompiledProgram`, `Shard`, `I18`, existing fixed-point helpers.
- Produces: `ZkieIsaCpuWitnessBackend`, typed register-value artifact.

- [ ] **Step 1: Write failing exact-semantics tests**

Cover `DotGeneral`, broadcast `Eltwise::Add`, `Eltwise::Mul`, and `RmsNorm` using the same FinText fixture already used by `rms_norm_fintext_real_weights.rs`. Assert raw `I18` values, not approximate formatted floats. Add a test that `Softmax` returns `BackendError::UnsupportedInstruction` until the assembler can prove the same operation.

- [ ] **Step 2: Run the tests and confirm the backend is absent**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-prover --test witness_cpu
```

- [ ] **Step 3: Implement execution over a shard range**

Resolve `GraphInput`, `Weight` and `Virtual` registers exactly once. Use `requantize_mul`/`requantize_raw` and the same RMSNorm domain/rounding helpers as the circuit. Produce only the shard inputs, outputs and internal values required by the proof job. Reject missing registers, forward references, shape mismatches and arithmetic overflow with typed errors.

- [ ] **Step 4: Cross-check host and circuit outputs**

For every supported fixture, run `AssemblerCircuit` with the generated witness and compare every virtual register's raw `I18` output against the CPU witness artifact.

- [ ] **Step 5: Run and commit**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-prover --test witness_cpu
git add crates/zkie-prover
git commit -m "feat: add deterministic CPU witness backend"
```

### Task 5: Real Halo2 KZG CPU backend

**Files:**
- Create: `crates/zkie-prover/src/halo2_cpu.rs`
- Modify: `crates/zkie-prover/src/lib.rs`
- Create: `crates/zkie-prover/tests/halo2_cpu_roundtrip.rs`
- Modify: `crates/zkie-compiler/tests/rms_norm_fintext_real_weights.rs`

**Interfaces:**
- Consumes: `ProofBackend`, `AssemblerCircuit`, `PreparedKeys`, `UnverifiedProof`.
- Produces: `Halo2KzgCpuBackend` for `halo2-kzg-bn256-shplonk-v1`.

- [ ] **Step 1: Move the existing real-proof expectations into a failing backend test**

Construct the real FinText RMSNorm fixture, prepare keys, prove to a temporary artifact directory, verify, flip one proof byte, and assert the modified proof returns `VerificationError::CryptographicVerificationFailed`.

- [ ] **Step 2: Run the focused test and confirm the backend is missing**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-prover --test halo2_cpu_roundtrip -- --nocapture
```

- [ ] **Step 3: Implement setup/keygen/prove/verify adapters**

Use the same concrete protocol already proven in the integration test:

```rust
create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<'_, Bn256>, _, _, _, _>(...)
verify_proof::<KZGCommitmentScheme<Bn256>, VerifierSHPLONK<Bn256>, _, _, _>(...)
```

Write the transcript directly to the provided output file rather than retaining an additional final `Vec<u8>`. Deserialize only keys whose metadata flavor, circuit digest and `k` match the job. `verify()` must reread proof bytes from disk and must not trust a prior success flag. Define `SrsPolicy::{ProductionExisting, DevelopmentGenerate}`: production returns `BackendError::MissingSrs` if the named SRS and source digest are absent, while development-generated SRS metadata is permanently marked non-production.

- [ ] **Step 4: Remove duplicated KZG ceremony code from the old test**

Keep the FinText semantic assertions, but call `Halo2KzgCpuBackend` for the real roundtrip so there is one production implementation of proof generation.

Add negative tests proving `PrivateModel` is rejected with `BackendError::UnsupportedModelVisibility` and that development SRS cannot satisfy a production prepare job.

- [ ] **Step 5: Run package and workspace verification**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-prover -- --nocapture
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test --workspace
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo clippy --workspace --all-targets -- -D warnings
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo fmt --all -- --check
```

Expected: all tests, clippy and formatting checks pass.

- [ ] **Step 6: Commit**

```bash
git add crates/zkie-prover crates/zkie-compiler/tests/rms_norm_fintext_real_weights.rs Cargo.lock
git commit -m "feat: add real Halo2 CPU leaf prover"
```
