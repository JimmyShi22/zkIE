# PCS two-shard audit — 2026-10-07

Audit of the committed cross-shard binding path in `crates/zkie-ops/src/compose.rs`
and the WHIR wrapper in `crates/zkie-core/src/pcs/{whir,committed}.rs`, followed by
the proof-transport fix and regression tests.

## Findings (pre-fix)

1. **Verifier regenerated opening proofs.** `verify_committed_cross_bind` called
   `whir.open_multi(committed.prover_data.clone(), ...)` and then verified the
   locally regenerated proof. The proof argument carried only the `same_poly`
   merge — the WHIR opening proof was never transported. This required shipping
   `prover_data` (witness data) to the verifier inside `Committed`. The same
   pattern existed in `verify_weight_batch` (`open_batch(prover_data.clone(), ...)`).
   `verify_projection_committed` already did this correctly: it verifies a
   transported `(Proof, value)` pair and never opens.

2. **`commit_batch` silently downgraded security.** `Whir::commit_batch` built its
   batch instance with `Whir::new_testing(total_num_vars)` (32-bit security, 10
   PoW-bit budget) regardless of the caller's configuration, so the "global
   weights commit" ran at test parameters even when the caller asked for 90-bit.

3. **PoW budget semantics.** `Whir::new` configures `(security_level = 90,
   pow_bits = 32)`; `new_testing` `(32, 10)`. In p3-whir 0.7.0 `pow_bits` is a
   grinding *budget/cap*, not the enforced amount: the algebraic part targets
   `security_level - pow_bits` bits and the per-round/final PoW is derived from
   the gap to the query coverage, capped by the budget (config errors only if
   the derived value exceeds it). The measured derived schedules are in the
   "Derived WHIR PoW/query schedule" section below: at n=6 the derived PoW
   reaches the full 32-bit budget, at n=19 the final phase derives 28 bits.
   Note the configured 90-bit level is the *PCS opening* security only — it is
   not a claim about the total GKR/sumcheck soundness of the composed proof,
   which is a separate question this configuration does not address.

4. **Unbound cross-shard tensor gap.** `verify_committed_shard_dag` iterated
   `proof.boundary_tensors` without checking that the set exactly equals the
   tensors claimed by more than one shard (the plain `verify_shard_dag` does
   check this). A proof omitting a genuinely cross-shard tensor would skip its
   binding.

5. **No KZG/BN254 wrapper exists.** No pairing/KZG dependencies anywhere in the
   workspace (only a doc comment referencing an external BN254 MSM crate).
   A KZG/BN254 path would be greenfield.

## Changes

- **Transport the opening proof.** `committed_cross_bind` now returns a
  `CommittedBindProof { open_proof, same }`; `CommittedShardDagProof` carries
  `boundary_commitments: Vec<CommittedPublic>` (commitment + protocol only, no
  `prover_data`) plus `cross_open_proofs: Vec<whir::Proof>` aligned with
  `boundary_tensors`/`cross_binds`. `verify_committed_cross_bind` takes the
  transported proof and a `CommittedPublic`; it only calls `verify_multi`.
  Weight batch: `open_weight_batch` (prover) produces the `(Proof, value)` pair,
  `verify_weight_batch(&BatchPublic, ...)` verifies it; `verify_projection_committed`
  now takes `&BatchPublic` too.
- **Verifier-only contexts.** New `CommittedPublic` and `BatchPublic` in
  `committed.rs`; `Committed::to_public()` / `BatchCtx::to_public()` build them.
  `BatchCtx::to_public()` reconstructs a fresh batch `Whir` with the identical
  security parameters (deterministic construction).
- **Protocol binding.** `verify_committed_cross_bind` rejects a transported
  `OpeningProtocol` that differs from the canonical one for `(tensor arity,
  claim count)` — both are public, so any deviation is a malformed proof.
- **Boundary-set check.** `verify_committed_shard_dag` now recomputes the
  expected cross-shard tensor set from the shard claims and requires it to equal
  `proof.boundary_tensors` exactly.
- **Security-parameter preservation.** `Whir` stores its `(security_level,
  pow_bits)`; `commit_batch` builds the batch instance with the caller's
  parameters. Exposed `Whir::security_params()`, `Whir::num_variables()`, and
  `Whir::opening_protocol()` (the canonical protocol builder, factored out of
  `commit_with_points`). No checks were weakened and forward recomputation was
  kept (see below).

## Tests

Red-first: `committed_two_shard_verify_never_opens` and
`verify_weight_batch_never_opens` assert the verifier's `open_stats` stay flat —
both failed on the old code (verifier regenerated 2 and 1 openings,
respectively) and pass after the fix. Added tamper coverage, each on an
independently re-proven proof (identical seed => identical proof):

- tampered commitment root (rejected),
- tampered shard claim eval (rejected),
- tampered `same_poly` coeff (rejected),
- substituted commitment to a different same-size tensor (rejected),
- tampered transported opening proof — OOD answers doubled (rejected),
- non-canonical transported protocol (rejected),
- tampered weight-batch opened value and wrong tensor at an index (rejected).

`commit_batch_inherits_caller_security_params` (zkie-core) checks, commit-only
and with no openings, that the batch instance carries the caller's
`(security_level, pow_bits)` — no grinding. Functional roundtrips use
`new_testing` throughout; the pre-existing `whir_prescribed_open_matches_mle_eval`
was switched to `new_testing` so no functional test grinds real PoW.

Evidence (release mode): zkie-core lib 22/22 passed; zkie-ops lib 39/39 passed;
`cargo check --release --workspace` clean.

## Remaining witness dependence (intentional, documented)

The committed shard-DAG verifier is still **not** a pure PCS verification:

- `verify_committed_shard_dag` clones the store and runs `forward_ops` to
  recompute the whole witness, then `verify_shard_precomputed` recomputes every
  MLE evaluation at the sumcheck challenge points from that fresh witness
  (proof claim evals are ignored there).
- `verify_committed_cross_bind` still requires the boundary tensor itself to
  check `verified == mle::eval(tensor, point)`, and `verify_weight_batch`
  requires the weight tensor.

In this local-verifier model the boundary commitment is redundant against
dishonest shard provers (the sumchecks already bind claims to the recomputed
witness). It only adds soundness once the verifier does not recompute the
tensor — e.g. remote shard provers or committed global weights. Removing the
forward recomputation is future work, not part of this stage.

## Remaining risks

- `BatchCtx::to_public()` allocates a fresh batch `Whir` (including its DFT
  tables); fine for tests, worth memoizing for large production batches.
- `whir::Proof`/`Commitment` are serde-serializable plonky3 types, but no wire
  format is defined yet — real cross-machine transport needs one.
- `GLOBAL_OPEN_COUNT` in whir.rs is process-global telemetry; tests assert only
  the per-instance `open_stats` cells, which are parallel-safe.
- p3-whir derives the actual per-round PoW from the security level and caps it
  by the budget; the derived schedules are measured below.

## Reviewer follow-up — metadata validation (2026-10-07)

Upstream `verify_at` **asserts** on metadata mismatches
(`assert_eq!(protocol.num_openings(), points.len())`, p3-whir adapter.rs:226,
267) rather than returning an error, and `mle::eval` asserts on tensor/point
dimension mismatch. A verifier must therefore reject malformed metadata itself:

- `Whir::verify_batch_checked` + `check_batch_metadata`: rejects non-canonical
  batch protocol (rebuilt via `Whir::batch_protocol(arity, num_tables)`),
  `num_tables == 0`, out-of-range `table_index`, wrong point dimension, and a
  batch size inconsistent with the instance's `num_variables`, all *before* any
  PCS verification. Returns `WhirVerifyError::{Malformed, Pcs}`.
- `verify_weight_batch` uses the checked path and rejects
  empty/non-power-of-two/wrong-size tensors before `mle::eval`.
  `verify_projection_committed` uses the checked batch path for its opening
  metadata; its tensor-shape guards remain inside `verify_projection` (the
  plain GKR verification it calls first).
- `verify_committed_cross_bind` rejects empty/non-power-of-two tensors, empty
  claims, and claim points whose dimension differs from the tensor arity before
  any MLE evaluation or PCS verification.
- Negative tests (all must return `false`, never panic):
  `verify_weight_batch_rejects_malformed_metadata` (zero `num_tables`,
  out-of-range index, wrong point length, changed protocol, non-power-of-two
  tensor, wrong-size tensor) and `verify_committed_cross_bind_rejects_malformed_inputs`
  (empty tensor, non-power-of-two tensor, wrong claim dimension, empty claims).

Evidence: zkie-core lib 22/22, zkie-ops lib 41/41, workspace `--all-targets`
check clean (release mode, real OS `timeout 120`).

## Two-shard benchmark evidence (`examples/two_shard_bench.rs`)

Fixed fixture (4 projections, 2 shards, 1 boundary tensor, m=4, d=8, seed
0x2525), explicitly-labelled testing parameters `Whir::new_testing(5)` (32-bit
security, 10 PoW-bit budget) — no real-security grinding:

```
prove: 6.339425ms  verify: 435.727µs  ok=true
shards=2 boundary_tensors=1 cross_open_proofs=1
after prove : open_stats=(2, 0.001815s) verify_stats=(2, _)
after verify: open_stats=(2, 0.001815s) verify_stats=(4, 0.000136s)
verifier produced no openings: OK (open_stats unchanged across verify)
```

## Derived WHIR PoW/query schedule (`examples/whir_pow_diag.rs`, config-only)

Configured `pow_bits` is a grinding budget/cap; p3-whir derives the schedule
(`protocol_security_level = security_level - pow_bits`, gap covered by queries
+ derived PoW, capped at the budget). Actual output (config-only run — no
commitments, openings, or grinding):

```
=== prod Whir::new (90-bit, 32 PoW budget): n=6 variables ===
configured: security_level=90 bits, pow_budget=32 bits (grinding cap), soundness_type=CapacityBound
derived:    folding_schedule=[5] final_sumcheck_rounds=1
derived:    starting_folding_pow_bits=0 final_folding_pow_bits=0
  final phase: num_queries=63 pow_bits=32
derived:    max_pow_bits=32 (<= budget: true)

=== testing Whir::new_testing (32-bit, 10 PoW budget): n=6 variables ===
configured: security_level=32 bits, pow_budget=10 bits (grinding cap), soundness_type=CapacityBound
derived:    folding_schedule=[5] final_sumcheck_rounds=1
derived:    starting_folding_pow_bits=0 final_folding_pow_bits=0
  final phase: num_queries=24 pow_bits=10
derived:    max_pow_bits=10 (<= budget: true)

=== prod Whir::new (90-bit, 32 PoW budget): n=19 variables ===
configured: security_level=90 bits, pow_budget=32 bits (grinding cap), soundness_type=CapacityBound
derived:    folding_schedule=[5, 5, 5] final_sumcheck_rounds=4
derived:    starting_folding_pow_bits=0 final_folding_pow_bits=0
  round 0: log_inv_rate=5 num_queries=63 pow_bits=32 folding_pow_bits=0 domain_size=1048576
  round 1: log_inv_rate=9 num_queries=12 pow_bits=31 folding_pow_bits=3 domain_size=524288
  final phase: num_queries=7 pow_bits=28
derived:    max_pow_bits=32 (<= budget: true)

=== testing Whir::new_testing (32-bit, 10 PoW budget): n=19 variables ===
configured: security_level=32 bits, pow_budget=10 bits (grinding cap), soundness_type=CapacityBound
derived:    folding_schedule=[5, 5, 5] final_sumcheck_rounds=4
derived:    starting_folding_pow_bits=0 final_folding_pow_bits=0
  round 0: log_inv_rate=5 num_queries=24 pow_bits=10 folding_pow_bits=0 domain_size=1048576
  round 1: log_inv_rate=9 num_queries=5 pow_bits=8 folding_pow_bits=0 domain_size=524288
  final phase: num_queries=3 pow_bits=6
derived:    max_pow_bits=10 (<= budget: true)
```

Note n=6 prod reaches the full 32-bit PoW budget (an open performs ~2^32
grinding trials per grinding stage, not exactly 2^32), so functional tests must
stay on `new_testing`.

## n=19 PCS measurements (experiments/pcs-comparison, 2026-10-07)

WHIR (Goldilocks base + quadratic extension challenges, 90-bit fixed target,
two tables of 2^19 batched in one witness, 3 rounds, RAYON_NUM_THREADS=16):

```
/usr/bin/time -v timeout 120 env RAYON_NUM_THREADS=16 PCS_N=19 PCS_ROUNDS=3 \
  PCS_POW_BUDGET=4 PCS_TESTING_BASELINE=0 cargo run --release --bin whir
```

- `PCS_POW_BUDGET=0` is REJECTED by p3-whir:
  `PowBitsExceedBudget { required: 4, budget: 0 }` (recorded, not weakened).
- `PCS_POW_BUDGET=4` is accepted: final_queries=10, max_pow_bits=4.
  setup 0.019 ms | dft_init 0.578 ms | commit 52.735 ms | open 80.667 ms
  (avg/round) | verify 3.962 ms (avg/round) | proof 107932 B (batched, both
  tables) | commitment 65 B. Wrong-point / wrong-commitment / mutated-value
  rejections: all true. Process peak RSS 190588 KiB (whole process incl.
  rejection-test overhead and the Cargo launcher, NOT per-proof), wall 0.75 s.

KZG (ark-poly-commit 0.5 multilinear PC, BN254, 2 tables opened separately,
3 rounds, RAYON_NUM_THREADS=16):

```
timeout 120 env RAYON_NUM_THREADS=16 PCS_N=19 PCS_ROUNDS=3 cargo run --release --bin kzg
```

- setup 5500.837 ms | commit 341.651 ms | open 2990.380 ms (avg/round) |
  verify 8.412 ms (avg/round, PCS-only; caller-side O(N) MLE eval excluded) |
  proof 2448 B (sum, 2 tables) | commitment 80 B (sum) | SRS 100664144 B.
  Process peak RSS 767304 KiB, total 16.80 s. All rejections true.

These are different fields and different batching (WHIR: one batched
witness/proof; KZG: separate commitments/proofs per table), so the numbers
compare at the PCS-operation level only; the configured 90-bit target is PCS
opening security, not total GKR soundness.
