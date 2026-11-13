# Byte-table range lookup plan of the clamp slack digits

- Scheme: `norm-output-tail-slack-lookup-plan`
- Change: code commit `e9524624` (trusted lookup plan and key map of the
  committed clamp slack digits, with the byte table fixed at eight address
  bits).
- Baseline: previous commit `595bd9a6` (trace of the padding rows
  `a203f116`).
- Purpose: carry the plan side of the TURN 504 一① range lookup: every slack
  digit slot of every position becomes one lookup key over the public byte
  table.

## What changed

- `tail_slack_lookup_plan`: one value column, one key per slack digit slot of
  every position, `table_log` eight (the public byte table), the key count
  rounded to the plan bit width, the padding value zero, and every source
  addressing the tail group cell of its position by row and place.
- The builder validates the tail shape, requires a clamp interval, bounds the
  key count, and validates the key map against the lookup plan.

## Evidence

- Unit test: 64 slack-digit keys for the two-position four-column fixture,
  key count equal to the rounded power of two, every source's oracle equal to
  its position's tail group oracle, the padding value zero, and the plan
  rejected when the tail section has no clamp interval.
- Focused: hidden_norm 37, norm_output 1, rstd 16, lookup 25,
  norm_param_table 4 passed (`focused.txt`).
- Workspace: 48 suites ok, 913 passed, 0 failed, exit 0 (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 504 一① / 2b.2-2b.3) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| public byte table + committed key sources | plan/key builder over the tail slack slots | plan test | the lookup witness is not wired yet |
| table identity fixed before challenges | identity carried in the key map | plan test | the transcript wiring is next |
| multiplicity counted per actual slot | one key per slot, count rounded to the bit width | plan test | the multiplicity bits witness is next |
| budget single derivation | key count derived from the tail layout | plan test | mask/ledger wiring is next |

## Open items

- The multiplicity-bit witness built from the digit multiplicities, its
  commitment messages, sample and openings, and the mask rows of the lookup
  section.
- The prover/verifier key-row selection so the lookup transcript reads the
  tail group messages (the key map's oracles identify the group), with the
  lookup claims joined to the existing BAT/FOLD/CS/BASE rows and the tail
  members.
- The mask/ledger budget derivation and the plan/bind version bump.
- The clamp five-point acceptance and the negative matrix; the padding
  negatives; the hd binding.
- No push in this round.
