# Tail relation masks in the committed mask plan

- Scheme: `norm-output-tail-masks`
- Change: code commit `da245e88` (one hiding and shown mask block per tail
  instance per position in the committed mask plan; derivation from the tail
  variable count; slots after the pointwise blocks).
- Baseline: previous commit `80f1ab23` (trace of the tail relation maps
  `1b973ddb`).
- Purpose: allocate the committed mask entries the five tail product
  instances will consume, and keep the mask budget singly derived.

## What changed

- `NormMaskPlan` gains the tail instance count per position and the hiding and
  shown counts of one tail instance; the counts are derived from the tail
  product-sumcheck variable count.
- Slot arithmetic: tail blocks follow the pointwise blocks and precede the
  reserved claims; `total_entries` and `claim_slot` follow the new base.
- The mask plan validates the derived counts and bounds the instance count at
  five; the norm plan requires the tail instances exactly when the tail
  section is present (fixture: five).
- The plan encoding absorbs the new fields; the committed mask message already
  covers the slots.

## Evidence

- Plan/chain tests pass with the tail mask allocation; the output chain still
  proves and verifies with the extended mask plan.
- Focused: hidden_norm 36, norm_output 1, rstd 16, lookup 24,
  norm_param_table 4 passed (`focused.txt`).
- Workspace: 48 suites ok, 912 passed, 0 failed, exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 502 / task card) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| mask budget single derivation | one helper derives the tail mask counts | plan tests | the transcripts are next |
| disjoint allocations | tail blocks after pointwise, claims last | slot arithmetic | no new message rows yet |
| allocation tied to the tail section | norm-plan rule | chain pass | the instances are next |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 81 tests across the five filters.
- Workspace: 912 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- The five tail product instances, their transcript section and the rows
  binding their maps to the committed tail and image groups.
- The bias addition, clamp and padding relations; the hd binding.
- The ledger categories at the zkie path; production source closure.
- No push in this round.
