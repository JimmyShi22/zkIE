# Pointwise product sumcheck transcripts in the norm second stage

- Scheme: `norm-pointwise-transcript`
- Change: code commit `9405c7f9` (the two equality-weighted product sumcheck
  transcripts per position: broadcast/gain and centered/product; column point
  drawn from the chain; transcript carried in the norm second message).
- Baseline: previous commit `b197cc66` (trace of the pointwise mask
  allocation `4a034263`).
- Purpose: run the product instances over the committed output rows and the
  parameter vector before their claim and end values are bound to the
  committed messages.

## What changed

- `NormOutputSecond`: the claim value and the general product-sumcheck proof
  of every pointwise instance, position-major, with a versioned canonical
  codec and bounds (round count, coefficient counts, two end values).
- The transcript is carried in the norm second message only when the plan
  carries the output tail; bind version stays 10 (the section is new).
- `PositionPscChallenges` also implements the general challenge trait on the
  same chain and tags, so the product instances use the existing transcript
  machinery and position indexing.
- The prover draws the column point per instance, computes the claim value
  from its own maps and proves the instance with the committed pointwise mask
  entries; the verifier draws the same point, replays the transcript against
  the carried claim and the derived hiding count, and records the checked
  output.
- The checked instances are carried in the row binding for the next step.

## Evidence

- The full output chain passes prove and verify with the two product
  instances per position; the transcript section round-trips through the
  message codec.
- Structural negatives through the real verifier: a missing output section,
  a popped round, a popped end value and a popped claim value are rejected.
- Focused: hidden_norm 34, norm_output 1, norm_pointwise 1, rstd 16,
  lookup 24, norm_param_table 4 passed (`focused.txt`).
- Workspace: 48 suites ok, 910 passed, 0 failed, exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 500 / task card) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| equality-weighted product instances | two instances per position over the column domain | chain prove/verify | the claim values are not yet bound |
| Product summand reuse | existing summand, weighted inner and general layer | same | no new summand form |
| committed mask consumption | the committed pointwise entries feed the prover | chain prove/verify | mask soundness closes with the rows |
| transcript transport | versioned, plan-gated second-stage section | codec/structural negatives | value tampering is rejected only after the rows land |
| challenge order | column point and rounds after the first-stage commitments | chain replay | the rows are next |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 80 tests across the six filters.
- Workspace: 910 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- The linear BAT rows binding each claim value to the committed product/image
  group, the broadcast and product end values to their groups, and the
  centered end value to the x/mean rows — until these land, a tampered claim
  or end value is not rejected by the transcript alone (recorded boundary,
  not a soundness claim).
- The rounding, bias, clamp and padding relations; the no-wrap real-bound
  re-derivation; the hd binding to the same-position parameters.
- The output group ledger categories at the zkie path; production source
  closure.
- No push in this round.
