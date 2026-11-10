# Proof of work on the position draws

## Scheme

- Scheme: `s7b-rework-r2pow`.
- Current code: `0e2c667e`.
- Previous code: `16c71ac0`; the sampling and journal rounds are `7a253542`
  and `ceb634f5` on `95ef31b5`.
- Scope: the real GPT-2 LM-head local chain. The production three-stage
  command is unchanged.

## Purpose

Complete R2: keep the trusted plan's 12 work bits on every position-drawing
round instead of dropping them for speed, and carry the nonce in the artifact
so the independent verifier redoes the test. The column draw and the three
switch draws are the position rounds of this chain; the FOLD, BASE and seal
rounds draw algebraic challenges and carry no work.

## What changed

- `FoldeqSession::draw_queries_with_work` builds the positions round's prefix
  with the plan's work bits, searches the least nonce on the prover side,
  checks a supplied nonce on the verifier side, then steps the chain. The
  verifier replays the column round with `artifact.column_nonce`.
- `CsMultiSession::set_work_bits` and
  `draw_positions_theta_with_nonce` do the same for a switch. The verifier
  replays each switch with the artifact's `nonce`.
- The artifact version is 2. It carries `column_nonce` after the column
  openings and a `nonce` in every switch record. A changed nonce is rejected.
- The work bits come from the trusted request plan: 12 for the once scope's
  column draw and 12 for the shared scope's switch draws.
- The search is bounded at `2^24` tries and fails closed; the plan's current
  work is 12 bits.

## Measurement basis

- Same machine, release build.
- Stage rows: `cargo test -p zkie --release --lib lm_head_stage_processes --
  --ignored --nocapture`; one run per row.
- Real local chain: `cargo test -p zkie --release --lib
  real_gpt2_lm_head_chain -- --ignored --nocapture`.
- Unit and workspace tests: `cargo test -p zkie --release --lib lm_head --`;
  `cargo test --workspace`.

## Results

| stage | wall s | VmHWM KiB | temp bytes | notes |
|---|---:|---:|---:|---|
| commit | 32.691 | 461,088 | 1,073,774,656 | private state 98,160,504 B |
| prove | 9.876 | 630,540 | 5,589,424 | setup 0.105 s, chain 9.582 s |
| verify | 5.566 | 70,344 | 0 | accept, same seal |

- Work round nonces: column 320; switches 2161, 2968, 457. Work bits 12 each.
- Draws: switch 0 `47/47/47`, switch 1 `63/62/62`, switch 2 `57/57/57` as
  `positions/unique_main/unique_mask`. Switch 1 has one physical collision.
- Artifact: 5,589,424 B in the stage harness; 5,598,728 B in the real local
  chain; base proof 61,992 B. The nonce fields add 32 B against the prior
  round; the remaining difference is the different draw set.
- Tokens/s (16 positions): prove 16 / 9.876 = 1.620; chain-only 16 / 9.582 =
  1.670. These are local lm_head/16-position numbers, not full-model numbers.
- Workspace tests: 48 suites green (`workspace-tests.txt`, exit 0).
- The stage harness negative matrix still rejects every entry. New negatives:
  flipping `column_nonce`, flipping a switch `nonce`, and a changed statement,
  roots, openings, identities, order, count and version all reject.

## Security accounting

- With the nonce verified, each of the four position rounds gets its 12 work
  bits. The largest round error with work counted is 116.01 + 12 bits, and the
  plan's own target is met with work counted. This run may state the with-work
  figures for these rounds.
- The shared-only limited-key derivation, the 12 ON_THE_CALLERS_WORD items and
  the full theorem assumptions are R4 and are not yet written. This run does
  not establish a production tested state.
- The without-work figure remains 116.01 bits; the with-work figure is not a
  claim about the whole model.

## Open items

- R1 real activation-to-LM-head request declaration closure.
- R3 trusted verifier anchor and plan reconstruction, malicious whole-handoff
  negatives.
- R4 ON_THE_CALLERS_WORD item table and shared-only derivation.
- R6 final-entry positives/negatives and performance labels.
- The local chain is not in the production CLI.

## Logs

- `unit-tests.txt`, `real-chain.txt`, `stage-processes.txt`,
  `workspace-tests.txt`.
- `chain-artifact.bin`: real-chain artifact (binary, gitignored).
