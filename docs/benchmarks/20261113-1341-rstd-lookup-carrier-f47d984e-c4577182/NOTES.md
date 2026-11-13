# Frozen-table lookup plan and message carriers

- Scheme: `rstd-lookup-carrier`
- Change: code commit `f47d984e` (NormRstdPlan, plan/digest, rstd fields in
  the norm plan, first/second messages, witness and row binding).
- Baseline: previous commit `c4577182` (rsqrt enable trace `97797fd1`).
- Purpose: prepare the second lookup over the frozen rstd table, keyed by the
  committed index and the committed table value, with the trusted table
  identity/length/encoding bound in the plan.

## What changed

- `NormRstdPlan` carries the two-column lookup plan, the committed bit/mask
  message plan, the key plan over the round value slots and the trusted
  `TableConfig` (identity, content root, real length, encoding, version).
- `NormRstdPlan::from_round` reads column zero from the committed index value
  slot and column one from the committed table value slot, pads the remaining
  keys with the public first table row, sizes the multiplicity-bit message and
  the mask rows from the parent codes, and binds the digest.
- `NormPlan` gains the optional rstd plan; its validation checks the plan and
  requires the table real length to equal the rounding plan's trusted table
  length.
- `NormFirstMessages` and `NormPscSecond` gain plan-gated rstd fields; the
  wire layout omits them when the plan has no rstd lookup, so existing norm
  messages are unchanged. `NormWitness` and `NormRowBinding` carry the
  matching prover/binding material.
- The bind version moves to 6.
- A unit test checks the four key sources (two columns per position), the
  column order, the padding pair, the table identity binding and the
  two-column validation.

## Evidence

- Unit test: `rstd_plan_reads_the_committed_index_and_value_slots` passes.
- Norm suite: 29 passed (see `focused.txt`).
- Workspace: exit 0 (see `workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (see `proof-bytes.txt`).

## Four-column mapping

| TURN 492 / TL ② requirement | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| key column 0 = committed index, column 1 = committed table value | key sources read the round index and rstd value slots per position | unit test checks the column order and slot mapping | the prove/verify first/second stages are not wired yet |
| table identity and real length bound | `TableConfig` in the plan digest; NormPlan binds the real length to the rounding plan | unit test validates the plan and rejects a zero identity; plan validation checks the length | the table view argument and the actual reads are the next step |
| public first-row padding pair | `padding_values = [0, T(0)]` fixed by the constructor | unit test checks the pair | the verifier-side first-row check is the next step |
| index < N | rounding plan table length + the row constructor reject index >= N | the row constructor test from the previous round | the rstd lookup binding of the committed rstd value is the next step |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: norm 29 passed, carrier unit test passed.
- Workspace: exit 0.
- No wall time or memory claim.

## Open items

- Prove/verify first stage: commit the rstd bit/mask messages, SEND sample
  answers/openings, absorb roots.
- Prove/verify second stage: draw the rstd lookup seed, prove/verify with the
  frozen table view, bind the key claim rows into the hidden BAT/FOLD, include
  the new members in bound ids and sample rows, extend the mask budget.
- Hidden fixture: synthetic ranged table with the trusted config and rstd
  values matching the committed table-value slots.
- Negative matrix and the TURN 492 acceptance batch.
- No push in this round.
