# Pointwise product mask allocation in the norm mask plan

- Scheme: `norm-pointwise-masks`
- Change: code commit `4a034263` (the per-position pointwise product mask
  allocation after the relation blocks; single-derived hiding/shown counts,
  disjoint slots, plan-gated by the output tail; bind version 10).
- Baseline: previous commit `dec5a013` (trace of the output group
  commitments `1cefcd15`).
- Purpose: allocate the committed mask entries the two equality-weighted
  product instances will consume, before the sumcheck transcripts are wired.

## What changed

- `NormMaskPlan` carries the pointwise instance count per position and the
  hiding/shown counts of one instance; the counts are derived from the
  product-sumcheck variable count by `norm_pointwise_masks`.
- The slot arithmetic lays the pointwise blocks after every relation block
  and before the reserved claims; `total_entries` and `claim_slot` follow.
- The plan requires the pointwise instances exactly when the output tail is
  present and rejects wrong counts, wrong instance numbers and oversized
  allocations.
- `norm_pointwise_entries` builds the committed mask entries of the pointwise
  instances in position-major order.
- The committed mask message already covers the new slots, so no first-stage
  message or byte layout changes beyond the plan digest.

## Evidence

- `norm_pointwise_masks_are_allocated_after_the_relation_blocks`: the counts
  are derived, the entries cover exactly the pointwise blocks after the
  relation blocks, and the reserved claims stay at the end; a plan with the
  output but no allocation, a plan with the allocation but no output and a
  wrong hiding count are rejected.
- The output group chain still passes prove/verify with the extended mask
  plan.
- Focused: hidden_norm 34, norm_output 1, norm_pointwise 1, rstd 16,
  lookup 24, norm_param_table 4 passed (`focused.txt`).
- Workspace: 48 suites ok, 910 passed, 0 failed, exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 500 / task card) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| mask budget from a single derivation | one helper derives the pointwise hiding/shown counts | allocation test | the sumcheck transcripts are next |
| disjoint allocations | pointwise blocks after the relation blocks, claims last | slot assertions | no new message rows yet |
| allocation only with the output tail | plan rule ties the instances to the output section | rejection tests | the second-stage wiring is next |
| committed masks | entries read from the first-stage mask message | entries test | the masking soundness is proven with the transcripts |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 80 tests across the six filters.
- Workspace: 910 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- The two pointwise product sumcheck transcripts with the equality weights,
  their claim values and the linear rows binding them to the committed
  output groups; then the rounding, bias, clamp and padding relations; then
  the hd binding.
- The no-wrap real-bound re-derivation for the product, rounding and bias
  equations.
- The output group ledger categories at the zkie path; production source
  closure.
- No push in this round.
