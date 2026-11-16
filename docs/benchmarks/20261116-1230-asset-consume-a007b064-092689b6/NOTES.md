# asset-consume: later parameters, coverage, and the new candidate instance

## Scheme and provenance

- Scheme: asset-consume. Code under test: `c95fcb18` (later distance rule and the
  later-table candidate), `534843af` (production later derivation with the
  ordinary plan fault), `99213415` (projection consumer reduction claim over
  the raw output), `4c4dea27` (consumer coverage over the consumed hidden
  columns), `eb613a12` (declared later batch in the table plan; key budget
  separated from the instance total), `a007b064` (evidence record carries the
  instance total and the declared later batch), `23c36aef` (durable budget
  opened with the key batch size and the instance total).
- Previous scheme close: `092689b6` (the [table-key] round closure).
- The directory name carries the commit at instance registration
  (`a007b064`); the key evidence below ran at `23c36aef`.
- Purpose: A1/A2 of the round: the approved later-parameter candidate, the new
  candidate instance with the later distances in its public shape, the key
  re-authentication on that instance, and the first B1/B2 pieces (the
  reduction claim and the coverage plan).
- Boundary: partial round progress. This trace does not claim the consumer
  alternative chain, the removal of the asset opening, or a batch pass.

## Later parameters (A1; values approved by the review)

- The later rule uses the same point goal with the request count and does not
  repeat the record check. In code the later branch is a distinct call path
  (`largest_table_m(..., with_record=false)`) that returns after the point
  test; the `later >= once` test is supplementary evidence, not the proof of
  the removal.
- The production derivation (`derive_production_later_candidate`) takes only
  the fixed public geometry, derives the once distance and columns itself,
  uses `table_tail = code.r`, request count one, work bits zero, and runs the
  ordinary `shared_lin_plan_fault` over the later point, columns and
  comparison point.
- Derived candidate (`later-candidate.log`): m_once=284, m_later=284,
  one_time_columns=52, later_columns=52, spent=52, capacity_batches=1,
  declared one-batch total=104 <= r=128.
- The 104 is the table's column budget only. The W, index mask, lookup and
  Omega messages keep their own masks, tails, fresh identifiers and
  cross-chain Omega accounting; they are not folded into 104.

## Table plan and budget separation (A2)

- `derive_table_plan` now derives the later candidate, writes `m_later` into
  the table shape distances (m_once=284, m_later=284), requires exactly one
  declared later batch, and sets the later batch columns (52) and the total
  (104).
- The table configuration and identity change with the distances
  (configuration `94dd0177…`, model-level identity `7562efa1…`); the old
  qualification is not migrated and the old ledgers stay spent.
- The key stage's own column budget stays 52
  (`KeyPlanInputs.total_columns = one_time_columns`); the instance's durable
  ledger and hiding budget total 104. The key run charges 52; the declared
  later batch's columns are accounted in the same root total (the durable
  budget is opened with batch size 52 and life limit 104).

## New candidate instance

| item | value |
|---|---|
| commit outcome | PASS, stage 3,849 ms, VmHWM 358,412 KiB |
| table root | `5a7cd31b…` |
| model-level table identity | `7562efa1…` |
| reader identity (bound by the plan) | `a5d13086…` |
| table configuration | `94dd0177…` |
| record length | 1024 |
| previous instance root | `404cca82…` (kept) |
| registration | 2 passed, 0 failed |

`registration.log`: the registration test binds the model-level meta identity
and the reader identity against the new derived plan and checks the previous
instance's root differs.

## Key re-authentication on the new instance

- Command: `cargo test --release -p zkie --test key_chain_evidence --
  --ignored --nocapture` with `ZKIE_KEY_EV` and `ZKIE_KEY_INSTANCE` pointed at
  this directory and instance; true status EXIT=0 (log
  `key-chain-evidence.log`).
- The run seed is the operating-system generator (private; not logged or
  written); the bundle is written to `key-proof.bin` (binary, not committed,
  ignored by git) and verified from disk.

| item | value |
|---|---|
| test result | 1 passed, 0 failed, 279.60 s |
| proof bytes | 4,551,024 |
| first / shared / second bytes | 3,554,128 / 713,936 / 282,928 |
| bundle sha256 | `44278485…` (matches the on-disk file) |
| prove wall | 188,662 ms |
| verify wall | 59,168 ms |
| process VmHWM | 35,432,628 KiB |
| verify HWM delta | 0 (same-process cumulative peak; the verify stage raised no new peak) |
| verify RSS before/after | 16,884,176 / 16,884,388 KiB (+212 KiB) |
| durable ledger | 0 -> 52 |
| instance total columns | 104 |
| declared later batches | 1 |
| later batch columns | 52 |
| plan digest | `93911f51…` |
| table root | `5a7cd31b…` |
| reader identity | `a5d13086…` |

The run covers the first part, the shared part, the second part and the
independent verification of the on-disk bundle; the real-bundle byte negatives
and the spent read ledger retry are exercised as before.

## B1/B2 progress in this batch

- `consumer_reduction` (`99213415`) builds one head slice's reduction claim
  from the public activation rows, the raw output and the verifier's
  coefficients. The identity `Σ_p Σ_n a[p] b[n] Y[p,n] =
  Σ_k (Σ_p a[p] X[p,k]) · (Σ_n b[n] M[n,k])` holds on the raw accumulator; a
  changed raw entry breaks it, and the requantized output is never substituted.
  The old transposed wiring is not reused.
- `ConsumerCoverage` (`4c4dea27`) fixes the twelve 64-column head slices over
  the consumed hidden width 768 of the 1024-column stored table; the stored
  padding columns [768,1024) are never claimed as consumed, and the reference
  count is one output entry per position and consumed column.

## Tests

- `table-plan.log`: model plan tests 21 passed / 0 failed.
- `consumer-claims.log`: projection claim tests 7 passed / 0 failed.
- `registration.log`: 2 passed / 0 failed.
- `workspace-tests.log`: `cargo test --workspace` EXIT=0; 1,243 passed,
  0 failed, 21 ignored across all targets.

## Ledger

- This instance: 0 -> 52 of its 104-column total; one charge, no refund.
- The earlier instances of the round (the fixed-seed candidate `4b35cc42`, the
  private-seed candidate `404cca82` and the diagnostic roots) stay spent and
  are not reset.

## Open

- The product-sum proof wiring over the same `M`, the request consumer chain
  (one request; one first/shared(LATER)/second; the head pendings merged by the
  real J), the sidecar proof and the global acceptance order, the deletion of
  the full `L0_k_w` opening, the negative matrix and the seq16/seq512
  validation set remain.
