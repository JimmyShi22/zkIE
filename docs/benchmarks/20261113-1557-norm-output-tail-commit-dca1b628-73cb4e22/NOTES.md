# Output tail slots committed in the first stage

- Scheme: `norm-output-tail-commit`
- Change: code commit `dca1b628` (tail slot layout, committed tail group in the
  first-stage norm message, prover consistency checks, verifier replay).
- Baseline: previous commit `73cb4e22` (trace of the tail width correction
  `122ef0c2`).
- Purpose: commit the rounding, bias and clamp witness slots before the
  relations over them are wired.

## What changed

- `NormOutputTailPlan::values`: per column the offset quotient digits, the
  remainder digits, the half selector, the offset shifted digits and the
  offset pre-clamp digits; signed quantities use a public half-range offset
  (`2^(8w-1)`), the remainder stays nonnegative; a value that does not fit
  its byte width is a bound violation.
- The slot values are position-major over the column domain; the message
  width covers every column's slots (64 for the fixture's 4 columns × 12
  slots).
- `NormOutputTailMessage` carries the committed tail group; the first-stage
  message includes it exactly when the output plan carries the tail section
  (plan-gated, bind version 12).
- The prover validates the committed group declarations, recomputes the slot
  values from the exact output rows and compares every message value, then
  absorbs the root, samples and openings; the verifier checks the same
  declarations and redraws the same round.

## Evidence

- The full output chain passes prove and verify with the tail group
  committed and opened.
- Real-verifier negatives: a changed tail root, a changed tail sample answer,
  a changed tail opening symbol and a missing tail section are rejected.
- Focused: hidden_norm 34, norm_output 1, norm_pointwise 1, rstd 16,
  lookup 24, norm_param_table 4 passed (`focused.txt`).
- Workspace: 48 suites ok, 910 passed, 0 failed, exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 502 / task card) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| committed rows for the tail | tail group committed in the first stage | chain + negatives | the relations are not proven yet |
| exact witness values | recomputed from the output rows and compared value by value | prover check | no plaintext in the message |
| signed quantities in digits | public half-range offsets | values layout | the no-wrap real bounds are next |
| transcript order | tail round after the output groups on both sides | chain replay | no challenge-order change |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 80 tests across the six filters.
- Workspace: 910 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- The rounding relations (image = q·2^shift + r, q/r digit reconstruction,
  half-up selector, shifted = q + s); the bias addition via the parameter
  member; the clamp interval; the padding rule.
- The no-wrap real-bound re-derivation of every tail equation (< p/2), with
  stop-and-report if a bound fails.
- The hd binding; the ledger categories at the zkie path; production source
  closure.
- No push in this round.
