# zkIE Resource-aware Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run and resume witness, keygen, leaf proof, verification and native aggregation jobs under explicit CPU/RAM/GPU budgets using isolated worker processes.

**Architecture:** A new `zkie-runtime` crate owns the durable state machine, SQLite metadata, artifact/key stores, resource estimation and deterministic scheduler. A thin `zkie-cli` binary creates plans and runs one scheduler; every heavy attempt executes in a separate `zkie worker` process so process exit releases proving memory.

**Tech Stack:** Rust 2021, rusqlite 0.40.2 with bundled SQLite, clap 4.6.6, serde/serde_json, sysinfo 0.39.6, Linux cgroup v2 with an explicit best-effort fallback.

**Spec:** `docs/superpowers/specs/2026-09-09-zkie-multibackend-sharded-proving-design.md`

## Global Constraints

- Complete the backend-foundation and partition/aggregation plans first.
- Run Cargo with `CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie`.
- Default test-machine limits are 400 GiB admission, 440 GiB hard limit, 56 logical CPU cores and aggregation fan-in 4.
- Interpret one GiB as exactly `1_073_741_824` bytes.
- Only verified dependencies make a job ready.
- SQLite state transitions have one writer: the scheduler.
- Worker temporary output is never a consumable artifact.
- No cgroup delegation means startup fails unless `--allow-best-effort-hard-limit` is explicit.
- Verification failure never enters the automatic execution retry path.
- Request a Rust review after every production-code task and a security review for path validation, process launch, artifact publication and CLI input handling.

---

## File Structure

- Modify `Cargo.toml`: add `crates/zkie-runtime` and `crates/zkie-cli`.
- Create `crates/zkie-runtime/Cargo.toml` and `src/lib.rs`.
- Create `crates/zkie-runtime/src/state.rs`: typed job states and legal transitions.
- Create `crates/zkie-runtime/src/db.rs`: SQLite schema/repository and recovery.
- Create `crates/zkie-runtime/src/store.rs`: atomic content-addressed artifact/key storage.
- Create `crates/zkie-runtime/src/estimate.rs`: static/history reservations.
- Create `crates/zkie-runtime/src/scheduler.rs`: ready calculation, best-fit, aging and retries.
- Create `crates/zkie-runtime/src/monitor.rs`: cgroup v2 and best-effort monitors.
- Create `crates/zkie-runtime/src/worker.rs`: immutable job spec/result protocol and launcher.
- Create `crates/zkie-cli/Cargo.toml`, `src/lib.rs`, `src/main.rs`; package `zkie-cli` exposes `[[bin]] name = "zkie"`.
- Create runtime integration tests for recovery, resources and worker isolation.

### Task 1: Durable typed state machine

**Files:**
- Modify: `Cargo.toml`
- Create: `crates/zkie-runtime/Cargo.toml`
- Create: `crates/zkie-runtime/src/lib.rs`
- Create: `crates/zkie-runtime/src/state.rs`
- Create: `crates/zkie-runtime/src/db.rs`
- Create: `crates/zkie-runtime/tests/state_recovery.rs`

**Interfaces:**
- Consumes: plan/job/proof identities from `zkie-compiler` and `zkie-prover`.
- Produces: `JobId`, `AttemptId`, `JobKind`, `JobState`, `FailureKind`, `RunDb`.

- [ ] **Step 1: Write failing transition tests**

```rust
#[test]
fn only_verified_predecessors_unlock_a_job() {
    let db = fixture_with_edge(JobState::VerificationFailed, JobState::Pending);
    assert_eq!(db.refresh_readiness().unwrap(), 0);
    assert_eq!(db.job("downstream").unwrap().state, JobState::Blocked);
}

#[test]
fn verification_failure_cannot_be_requeued_as_execution_failure() {
    assert!(JobState::VerificationFailed
        .transition(JobEvent::RetryExecution)
        .is_err());
}
```

Test every legal path in the spec and explicitly reject `proved -> verified` without a verification record.

- [ ] **Step 2: Run and confirm the runtime crate is absent**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-runtime --test state_recovery
```

- [ ] **Step 3: Implement state/event enums**

```rust
pub enum JobState {
    Pending, Ready, Witnessing, WitnessReady, Preparing, Proving, Proved,
    Verifying, Verified, ExecutionFailed, VerificationFailed,
    ResourceExceeded, Interrupted, Requeued, Blocked,
}

pub enum FailureKind {
    Execution, Verification, ResourceExceeded, Dependency, Interrupted,
}
```

`JobState::transition(event)` is the only place defining legal transitions. Store failure stage, typed code, sanitized summary, attempt count and blocking predecessor ID separately.

- [ ] **Step 4: Create and migrate SQLite schema**

Use `PRAGMA journal_mode=WAL`, `foreign_keys=ON`, `synchronous=FULL`. Create versioned tables `runs`, `jobs`, `dependencies`, `attempts`, `state_events`, `artifacts`, `resource_samples`, `resource_history`, and `schema_version`. Add uniqueness constraints for `(run_id, logical_job_id)` and artifact digest.

- [ ] **Step 5: Implement transaction-checked transitions and recovery**

`RunDb::apply_event` loads current state, validates transition, inserts an append-only event and updates the materialized job row in one transaction. On startup, `recover_interrupted()` changes active attempts to `Interrupted`, then `Requeued`; complete `Verified` jobs remain untouched.

- [ ] **Step 6: Run restart tests and commit**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-runtime --test state_recovery
git add Cargo.toml Cargo.lock crates/zkie-runtime
git commit -m "feat: persist proving job state transitions"
```

### Task 2: Atomic artifact and key stores

**Files:**
- Create: `crates/zkie-runtime/src/store.rs`
- Modify: `crates/zkie-runtime/src/lib.rs`
- Create: `crates/zkie-runtime/tests/store_atomicity.rs`

**Interfaces:**
- Consumes: `Digest32`, proof/key metadata.
- Produces: `ArtifactStore`, `KeyStore`, `StagedObject`, `PublishedObject`.

- [ ] **Step 1: Write failing publish/recovery tests**

Cover successful publish, hash mismatch, metadata mismatch, duplicate identical content, duplicate digest with different content, stale temporary files and an injected failure between file sync and DB commit.

- [ ] **Step 2: Run and confirm store types are absent**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-runtime --test store_atomicity
```

- [ ] **Step 3: Implement staged writes**

```rust
pub trait ContentStore {
    fn stage(&self, expected: &ObjectMetadata) -> Result<StagedObject, StoreError>;
    fn validate(&self, staged: StagedObject) -> Result<ValidatedObject, StoreError>;
    fn publish(&self, validated: ValidatedObject) -> Result<PublishedObject, StoreError>;
    fn open_verified(&self, digest: Digest32) -> Result<File, StoreError>;
}
```

Create temporary files under the same filesystem as the final object. Flush and `sync_all` content and metadata, calculate BLAKE3 by rereading the file, rename atomically, then sync the containing directory. Never overwrite an existing different object.

- [ ] **Step 4: Add proof-specific publication ordering**

The scheduler accepts a worker result only after store validation, backend cryptographic verification and final rename. Only then does one SQLite transaction insert the artifact and transition the job to `Verified`. Key publication follows the same ordering and records SRS source digest, flavor, circuit digest, `k` and actual aggregation arity.

- [ ] **Step 5: Run and commit**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-runtime --test store_atomicity
git add crates/zkie-runtime
git commit -m "feat: add atomic artifact and key stores"
```

### Task 3: Calibrated resource estimates and deterministic best-fit scheduling

**Files:**
- Create: `crates/zkie-runtime/src/estimate.rs`
- Create: `crates/zkie-runtime/src/scheduler.rs`
- Modify: `crates/zkie-runtime/src/lib.rs`
- Create: `crates/zkie-runtime/tests/scheduler_resources.rs`

**Interfaces:**
- Consumes: `RunDb`, backend estimates, `ResourceCapacity`, dependency graph.
- Produces: `ReservationKey`, `Reservation`, `SchedulerConfig`, `SchedulerDecision`.

- [ ] **Step 1: Write failing estimate tests**

Assert reservation equals `max(static_estimate, ceil(max_observed_peak * 1.15))`; history from another circuit, `k`, flavor, backend or hardware profile is ignored; overflow returns `EstimateError::Overflow`.

- [ ] **Step 2: Write failing scheduling tests**

Use ready jobs sized 250/150/100 GiB under a 400 GiB limit and assert deterministic selection fills 400 GiB without exceeding CPU. Assert non-verified predecessors are never selected. Advance the fake clock 600 seconds and assert the aged 300 GiB job reserves capacity instead of being starved by new 100 GiB jobs.

- [ ] **Step 3: Run and confirm missing scheduler**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-runtime --test scheduler_resources
```

- [ ] **Step 4: Implement reservation calibration**

```rust
pub struct ReservationKey {
    pub circuit_digest: Digest32,
    pub k: u32,
    pub proof_flavor: ProofFlavorId,
    pub execution_backend: ExecutionBackendId,
    pub hardware_profile: Digest32,
}

pub fn calibrated_reservation(
    static_request: ResourceRequest,
    max_observed_peak_bytes: Option<u64>,
) -> Result<ResourceRequest, EstimateError>;
```

Use integer ceiling arithmetic for the 15% margin. A resource-exceeded attempt records its observed peak before recalculation.

- [ ] **Step 5: Implement scheduler selection**

Refresh readiness transactionally. Mark jobs aged after exactly 600 seconds. If an aged job can fit the machine at all, suppress admission of jobs whose reservation would prevent it from starting when current workers finish. Otherwise choose fitting jobs by smallest normalized leftover across RAM, CPU, GPU count and per-device VRAM; tie-break by older ready time then stable JobId.

- [ ] **Step 6: Implement retry decisions**

Execution failure allows two retries; resource-exceeded allows three requeues with increased reservations; verification failure allows none. Permanent failures recursively mark dependent jobs `Blocked` with the direct predecessor ID.

- [ ] **Step 7: Run and commit**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-runtime --test scheduler_resources
git add crates/zkie-runtime
git commit -m "feat: schedule proving jobs within resource budgets"
```

### Task 4: Linux hard memory enforcement and portable monitoring

**Files:**
- Create: `crates/zkie-runtime/src/monitor.rs`
- Modify: `crates/zkie-runtime/src/scheduler.rs`
- Create: `crates/zkie-runtime/tests/resource_monitor.rs`

**Interfaces:**
- Consumes: worker PIDs, 400/440 GiB limits.
- Produces: `ResourceMonitor`, `CgroupV2Monitor`, `ProcessTreeMonitor`, `MemoryEnforcement`.

- [ ] **Step 1: Write failing fake-monitor tests**

Feed samples below 400 GiB, between 400 and 440 GiB, and three consecutive samples over 440 GiB. Assert respectively normal admission, frozen admission, and termination of the newest running attempt. Assert one lower sample resets the consecutive-over-limit counter.

- [ ] **Step 2: Run and confirm monitor types are absent**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-runtime --test resource_monitor
```

- [ ] **Step 3: Implement cgroup v2 detection and setup**

On Linux, verify `/sys/fs/cgroup/cgroup.controllers` contains `memory`, create a run-specific delegated child, write exact byte values to `memory.high` and `memory.max`, keep the scheduler PID outside, and move each worker PID into `cgroup.procs`. Read `memory.current`, `memory.events`, and peak when available. Return `MonitorError::NoDelegatedCgroup` on permission failure.

- [ ] **Step 4: Implement explicit best-effort fallback**

Use sysinfo to traverse descendants of registered worker PIDs and sum RSS once per PID. Startup without cgroup succeeds only when `allow_best_effort_hard_limit` is true, and the run metadata records `MemoryEnforcement::BestEffort`. Log a structured warning once, not on every sample.

- [ ] **Step 5: Implement controlled termination**

Freeze admission above 400 GiB. After three one-second samples above 440 GiB, send SIGTERM to the newest attempt, wait up to 10 seconds while continuing state observation, then force terminate its process group. Record `ResourceExceeded`, observed peak and the signal/exit status before requeueing.

- [ ] **Step 6: Run platform tests and commit**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-runtime --test resource_monitor
git add crates/zkie-runtime
git commit -m "feat: enforce scheduler memory limits"
```

### Task 5: Isolated worker protocol

**Files:**
- Create: `crates/zkie-runtime/src/worker.rs`
- Modify: `crates/zkie-runtime/src/scheduler.rs`
- Create: `crates/zkie-runtime/tests/worker_protocol.rs`

**Interfaces:**
- Consumes: immutable job specs, backend registry, staged output paths.
- Produces: `WorkerSpec`, `WorkerResult`, `WorkerLauncher`, `ProcessWorkerLauncher`.

- [ ] **Step 1: Write protocol and fake-launcher tests**

Roundtrip every job kind through JSON and reject unknown schema versions, changed run digest, relative output paths and a result for another attempt. Use an in-process `FakeWorkerLauncher` to prove scheduler behavior without directly executing a target-directory binary.

- [ ] **Step 2: Run and confirm worker protocol is absent**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-runtime --test worker_protocol
```

- [ ] **Step 3: Implement immutable specs and result files**

```rust
pub struct WorkerSpec {
    pub schema_version: u32,
    pub run_digest: Digest32,
    pub job_id: JobId,
    pub attempt_id: AttemptId,
    pub job: WorkerJob,
    pub staged_output_dir: PathBuf,
    pub rayon_threads: u32,
    pub cuda_devices: Vec<u32>,
}

pub struct WorkerResult {
    pub job_id: JobId,
    pub attempt_id: AttemptId,
    pub outcome: WorkerOutcome,
    pub measurements: WorkerMeasurements,
}
```

The scheduler writes the spec atomically, launches `zkie worker --spec <absolute-path> --result <absolute-path>`, sets `RAYON_NUM_THREADS` and `CUDA_VISIBLE_DEVICES`, and places the child in its own process group. The worker never writes SQLite.

- [ ] **Step 4: Route each job kind through the registry**

Support witness, prepare/keygen, leaf prove, leaf verify and native aggregate jobs. Reject a requested flavor/backend combination absent from capabilities. Sanitize error results so secrets and witness values are not written to the database.

- [ ] **Step 5: Run and commit**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-runtime --test worker_protocol
git add crates/zkie-runtime
git commit -m "feat: isolate proving attempts in worker processes"
```

### Task 6: CLI, resume flow and full scheduler verification

**Files:**
- Create: `crates/zkie-cli/Cargo.toml`
- Create: `crates/zkie-cli/src/lib.rs`
- Create: `crates/zkie-cli/src/main.rs`
- Create: `crates/zkie-cli/tests/cli_config.rs`
- Create: `crates/zkie-runtime/tests/end_to_end_queue.rs`
- Modify: `README.md`

**Interfaces:**
- Consumes: all runtime, compiler, prover and aggregation APIs.
- Produces: `zkie plan`, `prepare`, `prove-queue`, `status`, `verify`, internal `worker`.

- [ ] **Step 1: Write failing exact-value CLI tests**

Parse:

```text
prove-queue --memory-budget-gib 400 --memory-hard-limit-gib 440 \
  --cpu-budget-cores 56 --aggregation-fan-in 4
```

Assert exact byte conversion. Reject hard <= admission, zero CPU, N outside 2..=16, duplicate CUDA devices and best-effort fallback without the explicit flag.

- [ ] **Step 2: Run and confirm CLI is absent**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test -p zkie-cli --test cli_config
```

- [ ] **Step 3: Implement commands and status output**

`plan` writes immutable partition/aggregation plans; `prepare` schedules SRS/key jobs; `prove-queue` creates or resumes a run; `status` reports overall state plus exact failed stage/type/reason/attempt; `verify` independently reopens and verifies all referenced artifacts. `worker` is hidden from normal help but validates every input path and digest.

- [ ] **Step 4: Add crash/restart integration coverage**

Using fake backends and launcher, simulate crashes during witnessing, proving, after staged proof creation, and during verification. Assert recovery requeues active attempts, verifies complete staged proofs, reuses only matching verified artifacts and never consumes a temporary file.

- [ ] **Step 5: Add queue composition coverage**

Run a synthetic three-shard DAG with a broadcast edge under constrained fake resources so jobs must execute in two waves. Assert maximum simultaneous reservation stays within budget, each downstream starts only after predecessors verify, and the final artifact is `NativeVerifiedManifest`.

- [ ] **Step 6: Run full local verification**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo test --workspace
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo clippy --workspace --all-targets -- -D warnings
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo fmt --all -- --check
```

- [ ] **Step 7: Run a source-level CLI smoke test**

```bash
CARGO_TARGET_DIR=/Users/jimmyshi/.cargo/bin/build-targets/zkie cargo run -p zkie-cli -- status --run-dir .spike-test/scheduler-fixture
```

Expected: a structured status summary with overall state and no direct execution of a target-directory binary.

- [ ] **Step 8: Commit**

```bash
git add Cargo.toml Cargo.lock crates/zkie-cli crates/zkie-runtime README.md
git commit -m "feat: add resumable resource-aware proving queue"
```

### Task 7: Test-machine calibration

**Files:**
- Create: `docs/superpowers/specs/2026-09-09-zkie-scheduler-calibration-results.md`

**Interfaces:**
- Consumes: installed `zkie` CLI and current 32-core/64-thread, 495 GiB, 3×L20 machine.
- Produces: measured resource-history seed and a reproducible calibration report.

- [ ] **Step 1: Install the executable before running it**

On the test machine from `/data/jimmyshi/test/ie`, install into a user executable prefix and run only that installed path, never the build-tree artifact:

```bash
CARGO_TARGET_DIR=/data/jimmyshi/cargo-build/zkie cargo install --locked --path crates/zkie-cli --root /data/jimmyshi/zkie-install
/data/jimmyshi/zkie-install/bin/zkie --version
```

- [ ] **Step 2: Run one-shard baseline**

Use 56 CPU threads, 400 GiB admission and 440 GiB hard limit. Record circuit digest, `k`, backend/flavor, static reservation, peak `memory.current`, worker peak RSS, wall time, proof bytes, verify time and key load/keygen time.

- [ ] **Step 3: Run sequential and concurrent queue cases**

Run the same representative FinText shard three times sequentially to populate history, then run resource-compatible distinct shards concurrently. Confirm reservation becomes `max(static, max_observed * 1.15)` and the scheduler never admits more than 400 GiB or 56 logical cores.

- [ ] **Step 4: Exercise recovery and hard protection safely**

Interrupt one worker below the hard limit and verify resume. Exercise hard-limit behavior only with synthetic memory workers capped well below host capacity; verify cgroup events and `resource_exceeded` transitions without intentionally driving the host near OOM.

- [ ] **Step 5: Record results and commit**

Document exact commands, digests, samples, pass/fail results and deviations from estimates. Do not generalize TimesFM requirements from FinText without a TimesFM circuit measurement.

```bash
git add docs/superpowers/specs/2026-09-09-zkie-scheduler-calibration-results.md
git commit -m "docs: record zkIE scheduler calibration"
```
