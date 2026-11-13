# Tail relation maps bound to the committed slots

- Scheme: `norm-output-tail-rows`
- Change: code commit `93c19b4c` (tail group registered as an own member; BAT
  rows binding both shown maps, the mask units and the hiding form of every
  tail instance; single term enumeration shared with the value-level maps).
- Baseline: previous commit `6988b80c` (trace of the tail transcripts
  `9819ca70`).
- Purpose: close the boundary recorded in the transcript round; after this
  round, a changed tail shown value is rejected by the real verifier.

## What changed

- `output_rel_terms`: one enumeration of the tail relation products as
  `TailTerm` factors (committed tail slot with scale, committed image value,
  public constant); both the value-level maps and the BAT rows derive from it.
- The tail group is registered as an own proximity member (sample rows and
  witness trees included).
- The rows: the left shown map against the committed tail slots and the image
  row; the right shown map against the same objects (the public constant parts
  folded into the row targets); each end carries its mask unit with the end
  mask scale; the hiding form against the committed tail mask entries.
- `NormTailBinding` carries the checked tail outputs to the row builder on
  both sides.

## Evidence

- The full output chain passes prove and verify with the tail rows in the BAT
  closure.
- Real-verifier negatives: a changed tail point value is rejected; the earlier
  tail structural negatives (popped proof, popped point value, missing
  section) and the tail commitment negatives still hold.
- Focused: hidden_norm 36, norm_output 1, rstd 16, lookup 24,
  norm_param_table 4 passed (`focused.txt`).
- Workspace: 48 suites ok, 912 passed, 0 failed, exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 502 / task card) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| map factors bound to committed objects | single term enumeration; tail and image terms | chain + point-value negative | constants stay public |
| shown values bound | mask units with the end mask scales | chain pass | the hiding form is bound to the mask |
| hiding form bound | units on the committed tail mask slots | chain pass | no extra commitments |
| single source | maps and rows share the enumeration | chain pass | bias/clamp are next |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 81 tests across the five filters.
- Workspace: 912 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- The bias addition (`pre_clamp = shifted + bias` via the parameter member),
  the clamp interval and the padding rule.
- The hd binding; the ledger categories at the zkie path; production source
  closure.
- No push in this round.
