# Clamp comparison rows in the tail layout

- Scheme: `norm-output-tail-clamp-slots`
- Change: code commit `5794f907` (optional clamp widths and slots in the tail
  plan: offset distances to the low and high bounds plus their selectors).
- Baseline: previous commit `50f106d6` (trace of the bias binding
  `bae4fc2e`).
- Purpose: carry the committed values the clamp relation needs, with widths
  derived from the pre-clamp capacity and the clamp bounds.

## What changed

- `NormClampWidths` (low and high byte widths) on the tail plan; the tail
  slots and message width include the clamp rows.
- Validation: the clamp rows exist exactly when the output plan carries a
  clamp interval; both widths are bounded, below the modulus, and cover the
  pre-clamp capacity plus the farthest bound.
- The slot values append the offset distances `pre_clamp - low` and
  `pre_clamp - high` (public half-range offsets) and the two selectors
  `[pre_clamp < low]`, `[pre_clamp > high]`.
- The integrated fixture sizes the tail group to 128 columns (26 slots per
  column); the plan test follows the same derivation.
- All slot arithmetic (values, relation terms, bias row stride) uses the
  plan's slot count including the clamp rows.

## Evidence

- The plan test accepts the clamp widths, rejects a missing or spurious clamp
  section by plan rule, and the column width follows the new slot count.
- The full output chain passes prove and verify with the extended tail slots
  committed and opened.
- Focused: hidden_norm 36, norm_output 1, rstd 16, lookup 24,
  norm_param_table 4 passed (`focused.txt`).
- Workspace: 48 suites ok, 912 passed, 0 failed, exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 502 / task card) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| clamp interval rows | offset distances and selectors in the tail slots | chain + plan test | the clamp relation is not wired yet |
| widths from the trusted plan | capacity check against the pre-clamp bound and clamp bounds | rejection cases | clamp stays the output plan value |
| single slot arithmetic | values, terms and rows share the plan slot count | chain pass | the rows are next |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 81 tests across the five filters.
- Workspace: 912 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- The clamp relation (selector booleanity, one-hot, offset/slack rows) and
  the padding rule (non-valid columns output the clamped bias); the hd
  binding.
- The ledger categories at the zkie path; production source closure.
- No push in this round.
