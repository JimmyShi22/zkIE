# Pointwise claims and end values bound to their committed rows

- Scheme: `norm-output-rows`
- Change: code commit `219cf5f8` (the output groups registered as own
  proximity members and every pointwise instance bound by BAT rows: claim,
  two end values, hiding form).
- Baseline: previous commit `3d0bfde5` (trace of the pointwise transcripts
  `9405c7f9`).
- Purpose: bind the pointwise product claim and end values to the committed
  output rows so a tampered claim, end value, round coefficient or hiding sum
  is rejected by the real verifier, closing the boundary recorded in the
  transcript round.

## Root cause of the earlier blocker

The claim is drawn and computed at the original column point `z`
(`target = sum eq(z) a b`), but the first row construction used the
transcript's final challenge point to build the equality coefficients. Since
`z` differs from that point, the row was not the original claim and the honest
witness failed the own BASE target identity. The fix passes the original
point from the prover and the verifier, which draw it identically, into the
claim row; the end values and the hiding form keep their own transcript
points and masks.

## What changed

- `NormOutputBinding` carries the original column point next to the claim and
  the checked transcript output.
- The four output groups are registered as own members (with sample rows and
  witness trees), so the rows can reference them.
- The claim row puts `eq(z)` coefficients on the committed product (kind 0)
  or image (kind 1) row with the claim value as target.
- The broadcast and product end rows use `eq` of the transcript point on
  their groups; the centered end row is a combined x row (valid-masked
  coefficients) plus a mean row (public constant); the gain end row uses the
  evaluation coefficients of the parameter member.
- The hiding-form row puts the output's hiding coefficients on the committed
  pointwise mask slots with the last-round target.
- Every shown end carries its mask unit with the end's mask scale.

## Evidence

- Full output chain prove and verify passes with the two pointwise instances
  per position bound.
- Real-verifier negatives: missing output section, popped round, popped end
  value, popped claim, changed claim value, changed end values (both ends),
  changed round coefficient and changed hiding sum are all rejected.
- Focused: hidden_norm 34, norm_output 1, norm_pointwise 1, rstd 16,
  lookup 24, norm_param_table 4 passed (`focused.txt`).
- Workspace: 48 suites ok, 910 passed, 0 failed, exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Privacy check (TURN 502 item 4)

- The claim target enters the general sumcheck only through the masked
  start `hiding_sum + eps * target`, the same consumption form as the
  reference claim-and-mask treatment; the rounds are masked by the committed
  hiding polynomial and the hiding-form row binds its coefficients to the
  committed mask entries.
- The claim value and end values are random-point evaluations of committed
  rows, the same class of public claims already carried by the transcript
  round; no plaintext of x/mean/rstd/gain/bias/image/out is added to the
  proof.
- If the trusted plan later requires the claim target to be masked in the
  wire, that is a plan-level change and is not silently introduced here.

## Four-column mapping

| Requirement (TURN 500 / 502) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| claim value bound to product/image | claim row with `eq(z)` at the original point | chain + changed-claim negative | the point is drawn, not carried |
| end values bound to their maps | four end rows with mask units | changed-end negatives | centered uses x/mean rows |
| hiding form bound to masks | (U) row on the committed pointwise slots | changed-round/hiding-sum negatives | the mask message covers the slots |
| no challenge-order change | rows consume the existing transcript outputs | chain replay | the rounding relations are next |
| privacy | masked-start consumption form | analysis in this note | no new value classes |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 80 tests across the six filters.
- Workspace: 910 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- The rounding, bias, clamp and padding relations over the output rows; the
  no-wrap real-bound re-derivation; the hd binding to the same-position
  parameters.
- A direct wrong-point injection needs a prover-side harness (the point is
  drawn identically by both sides and is not carried in the message); the
  wire-visible negatives cover the equivalent effects.
- The output group ledger categories at the zkie path; production source
  closure.
- No push in this round.
