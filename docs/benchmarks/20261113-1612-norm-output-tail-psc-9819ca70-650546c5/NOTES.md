# Tail relation product instances in the second stage

- Scheme: `norm-output-tail-psc`
- Change: code commit `9819ca70` (five tail relation product-sumcheck instances
  per position, transcript section and verifier replay).
- Baseline: previous commit `650546c5` (trace of the tail mask allocation
  `da245e88`).
- Purpose: run the rounding relation algebra over the committed tail mask
  entries and carry the transcripts, before the rows bind the shown values to
  the committed slots.

## What changed

- `NormTailPscSecond`: one transcript and two shown point values per position
  and relation, with a versioned canonical codec and instance/variable bounds;
  plan-gated by the tail group inside the norm second message.
- `norm_tail_entries`: the committed mask entries of the tail instances in
  position-major and relation order.
- The prover draws the five instances per position over the value-level maps
  and the committed mask entries with the existing product-sumcheck machinery
  and its own transcript tags; the verifier replays the same instances with
  zero mask values and compares the shown point values.
- Instance order is position-major and relation-major, after the pointwise
  instances on both sides.

## Evidence

- The full output chain passes prove and verify with the tail relation
  instances.
- Structural negatives through the real verifier: a popped tail proof, a
  popped tail point value and a missing tail section are rejected.
- Focused: hidden_norm 36, norm_output 1, rstd 16, lookup 24,
  norm_param_table 4 passed (`focused.txt`).
- Workspace: 48 suites ok, 912 passed, 0 failed, exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 502 / task card) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| rounding relations over committed slots | five instances per position | chain + structural negatives | the shown values are not yet bound |
| committed mask consumption | entries from the committed mask message | chain pass | mask soundness closes with the rows |
| transcript transport | plan-gated versioned section | codec and structural negatives | value tampering is rejected only after the rows land |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 81 tests across the five filters.
- Workspace: 912 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- The rows binding each tail relation map to the committed tail and image
  group slots; until they land, a tampered shown value is not rejected by the
  transcript alone (recorded boundary, not a soundness claim).
- The bias addition, clamp and padding relations; the hd binding.
- The ledger categories at the zkie path; production source closure.
- No push in this round.
