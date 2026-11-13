# Lookup key map over the committed rounding digits

## Scheme and baseline

- Scheme: `lookup-round-key-map`.
- Code: `81207d36`; previous commit `acda309c`.
- Purpose: derive the byte-lookup key plan from the trusted rounding layout so
  every committed digit slot is a key that reads its physical rounding cell,
  instead of hand-picked fixture sources.
- This is partial progress. The lookup commit plan is still assembled by the
  caller, and the BAT/FOLD registration of the lookup claims is pending the
  kept-map row decision.

## Validation basis

- The seq16 x3 rule is not applicable here: no production plan carries the
  lookup, no seq512 logic and no all-case structural path changes. The
  workspace suite, the focused tests and the byte-identical norm-free proof
  are the acceptance evidence.

## Changes

- `NormRoundLayout::digit_slots` lists the committed digit slots in plan order:
  the fixed-offset sum and mean rows, the squares and variance rows, and the
  four offset/complement rows. The six value slots (remainders, offsets and
  complements) are not keys.
- `round_digit_lookup_key` builds one `LookupKeySource` per digit slot and
  position: the oracle is the position's rounding oracle, the row and place
  follow the committed message layout, and the coefficient and offset are one
  and zero. Remaining key entries use the public padding byte; the table
  identity stays a plan value.
- `lookup_parameters` derives the minimal key-bit count (`2^key_bits >= slots`)
  and multiplicity-plane count (`2^(2^mu) > 2^key_bits`) from the digit count.
- The hidden fixture reads its real rounding messages through the new map; the
  unit test checks slot coverage, the excluded value slots, the per-cell
  mapping and the padding entries.

## Four-column mapping

| Requirement | Mechanism | zkIE landing | Correspondence / gap |
| --- | --- | --- | --- |
| Key indices fixed before challenges | Trusted round layout fixes every key-to-cell map | `round_digit_lookup_key`; plan digest binds the key map | Key map is plan data; no challenge-dependent selection |
| Physical source | Oracle, position, row, place of the committed rounding message | `LookupKeySource` from `digit_slots` and the message layout | Read by `LookupKeyPlan::read`; no virtual tree or root |
| Digit domain | Only base-256 digit rows are keys | `NormRoundLayout::digit_slots` excludes the six value slots | Value slots are bound by the existing relations, not the lookup |
| Padding and table identity | Public padding byte and plan table identity | `padding_byte`, `table_identity` parameters | Padding is a byte; the table identity is a plan value |
| Real message source | Keys read from committed rounding rows | Fixture test over `witness.round.messages` | Prover side uses the actual committed message |
| Parameter sizing | Minimal key and plane counts | `lookup_parameters` | Commit-plan sizing from the parent plan is still caller work |

## Results

- `cargo test --workspace`: 48 suites, 858 passed, 0 failed, 17 ignored.
- Focused `proof::lookup`: 8 passed; `proof::hidden_norm_psc`: 17 passed;
  `round_digit_key_plan_covers_the_committed_digits`: 1 passed;
  `production_key_map_reads_the_round_digits`: 1 passed.
- Hard gate: the norm-free toy output-head proof is 45,080 B, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`,
  byte-identical to the baseline dump.

## Explicitly deferred

- Sizing the lookup commit plan (bit and mask shapes, trees, openings) from the
  parent plan instead of caller-supplied values.
- Registering the key claim expansion and the multiplicity/end claims in the
  hidden second BAT/FOLD; this waits on the kept-map row decision.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
