# Real per-switch sampling and a durable claim journal

## Scheme

- Scheme: `s7b-rework-r2r5`.
- Current code: `ceb634f5`.
- Previous code: `7a253542`; the sampling change alone is `7a253542` on
  `95ef31b5`.
- Scope: the real GPT-2 LM-head local chain. The production three-stage
  command is unchanged; this is the local stage harness.

## Purpose

Two rework items:

- R2: draw each switch's positions from the trusted compiled plan instead of
  the fixture count 2, and make the verifier pin each count to the trusted
  plan. Report logical draws and unique physical openings separately.
- R5: make the claim journal durable before its lock is released, and record
  the external trust boundary of snapshot rollback.

## What changed

- `LimitedSwitchInputs` and `LimitedSwitchPlan` carry `positions`.
  `plan_stage_chain` sets switch 0 to the initial group's compiled count and
  switch `i > 0` to stage `i - 1`'s count. The last stage's count remains the
  BASE position count.
- `run_switch` uses the input count; `verify_limited_chain` uses the public
  switch plan and rejects a count that differs from the trusted setup's
  `switch_positions`. The prover also compares the whole provided sequence to
  the session's trusted list. A zero count is rejected.
- The prove result records per switch: `positions` (logical draws),
  `unique_main` and `unique_mask` (physical openings).
- `claim_budget` writes the claimed state to a fresh temporary file, renames
  it over the journal, syncs the parent directory, removes the lock, and syncs
  the directory again. The file itself is synced by `write_bytes`.
- New tests: success leaves no lock or temporary behind and a fresh reader
  sees the spent journal; two concurrent claims have exactly one winner and
  the mask is not reused; a stale lock fails closed without consuming the
  journal.

## Measurement basis

- Same machine, release build.
- Stage rows: `cargo test -p zkie --release --lib lm_head_stage_processes --
  --ignored --nocapture`; one run per row.
- Real local chain: `cargo test -p zkie --release --lib
  real_gpt2_lm_head_chain -- --ignored --nocapture`.
- Unit tests: `cargo test -p zkie --release --lib lm_head -- --nocapture`.

## Results

| stage | wall s | VmHWM KiB | temp bytes | notes |
|---|---:|---:|---:|---|
| commit | 32.780 | 461,200 | 1,073,774,656 | private state 98,160,504 B |
| prove | 9.978 | 630,500 | 5,600,288 | setup 0.102 s, chain 9.683 s |
| verify | 5.584 | 70,144 | 0 | accept, same seal |

- Draws: switch 0 `47/47/47`, switch 1 `63/63/63`, switch 2 `57/57/56` as
  `positions/unique_main/unique_mask`. The second switch of the last row is a
  physical collision: 57 logical draws, 56 unique mask openings. The third
  count is not the BASE count (84); BASE keeps its own plan-bound count.
- Artifact: 5,600,288 B in the stage harness; 5,589,336 B in the real local
  chain. Base proof 62,504 B. The artifact grew by 221,088 B against the
  previous `5,378,440 B` because the switch openings now carry the real draw
  counts rather than 2; this is a security correction, not an optimization.
- Tokens/s (16 positions): prove 16 / 9.978 = 1.604; chain-only 16 / 9.683 =
  1.652. The previous `1.854 / 1.918` was measured with 2 draws per switch and
  is not comparable; it is not a performance claim.
- The independent real-chain verify reaches the same seal; all nine negative
  classes in the stage harness still reject.
- Unit tests: 52 passed, 0 failed, 3 ignored.

## Security accounting

- The real counts are 47, 63 and 57 logical draws at the three switches; the
  BASE count is 84. These replace the fixture count 2.
- Proof of work is **not implemented** in this local chain. The plan states 12
  work bits for position-drawing rounds, but no nonce is searched, carried or
  verified. Therefore the honest bound for this run is the without-work bound
  (116.01 bits at the largest round), not the with-work 126.21 bits. This run
  must not be presented as production-tested and must not claim the with-work
  number. Wiring the nonce through the position-drawing rounds (CS and the
  column round) is the remaining R2 work.

## R5 durability

- Order on success: temporary write and file sync, atomic rename, parent
  directory sync, lock removal, second directory sync. A crash before the first
  directory sync leaves the old journal or the lock and later claims fail
  closed; a crash after it leaves the spent state durable.
- Mask non-reuse: the journal marks the mask consumed in the same atomic
  replacement; a second claim, a concurrent claim and a stale lock all fail.
- Machine or directory snapshot rollback to an earlier state is an external
  deployment trust assumption and is not detected. The journal fences reuse
  within the directory lifetime only.

## Open items

- R2: 12-bit proof of work per position-drawing round is absent; see above.
- R1, R3, R4 and R6 remain open in the rework batch.
- This harness is local; the production CLI does not carry this chain yet.

## Logs

- `unit-tests.txt`, `real-chain.txt`, `stage-processes.txt`.
- `chain-artifact.bin`: real-chain artifact (binary, gitignored).
