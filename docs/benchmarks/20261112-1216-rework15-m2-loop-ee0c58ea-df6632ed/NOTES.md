# rework15 M2 part 2: hidden-session multi-stage IOPP loop

## Scheme and baseline

- Scheme: `rework15-m2-loop`.
- Code: `ee0c58ea` (multi-stage loop module and its tests); baseline
  `df6632ed` (trusted stage lists in the plan).
- Purpose: implement the stage 0 BAT/FOLD -> (CS switch -> stage FOLD)* ->
  last CS -> BASE loop in a new module and check it end to end against a small
  fast fixture. The proof format beyond the new loop is unchanged; the hidden
  shared/own proximity paths are not routed through the loop yet.

## Loop summary

- Stage 0 input is the trusted BAT/FOLD handoff, its authentication basis, the
  folded rows, the member commitments and the two mask trees. It is validated
  against the first stage entry before any session starts.
- Every later stage runs the CS switch over the previous stage's rows with the
  plan-derived target, tree plan and drawn position count. The first switch
  authenticates old main rows through the supplied member trees and the old
  mask through the dedicated mask tree or the member mask part; later switches
  authenticate through the previous switch's committed message.
- Except the last stage, the combined switch claims are folded by a stage FOLD
  into the next stage's rows, handoff and basis. The last stage feeds BASE.
- The verifier rebuilds the complete stage chain from the stage list, checks
  the proof's switch targets against it before any proof-supplied root,
  opening or stage FOLD message is used, and replays the same transcript.

## Fixture construction

- Fast codes: stage 0 (5,8,1)/(5,16,1) `k=2,h=0`; switch 1
  (6,4,1)/(6,32,1) `k=h=1`; last stage (8,4,1)/(8,128,1) `k=h=0`.
- The member tree commits encoded member codewords; the stage 0 rows are the
  gamma-weighted fold of the member messages and tails, so the switch
  authentication targets equal the row evaluations. The old and fresh mask
  trees commit the exact mask rows and tails the rows carry.

## Tests

`cargo test -p zkie-ops --lib multi_stage_iopp`: 9 passed, 0 failed.

- Round trip: two switch messages, the first carries a stage FOLD and the last
  does not; the verifier seal equals the prover seal.
- Negatives rejected: missing stage, reordered stage, wrong old mask root,
  wrong old opening, wrong fresh opening, wrong mask room at the plan, wrong
  leaf class at the stage FOLD.
- The actual shared and own stage lists derive and validate.

`cargo check --workspace`: finished with no errors; the pre-existing workspace
warnings are unchanged.

## Not run

- No seq16x3 or case run: the hidden paths are not wired to the loop, so the
  protocol output and the measured single-stage run are unchanged. The next
  step wires the hidden shared/own paths and derives the per-stage planner
  positions/work/error before binding; the full seq16x3 and case decision must
  then be run.

## Open items

- Wire the hidden shared/own proximity paths to the loop.
- Derive planner positions/work/error per stage and record the derivation
  before binding.
- Kept rows stay empty; a non-empty kept list is rejected at the stage FOLD
  handoff.
- No push: origin remains `831e7134`.

## Ledger

- Tests passed: 9 focused tests, 0 failed; workspace check clean.
- Proof bytes/wall time/VmHWM: not measured; no protocol path changed.
