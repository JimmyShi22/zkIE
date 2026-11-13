# Lookup binding matrix and production plan wiring

## Scheme and baseline

- Scheme: `lookup-binding-matrix`.
- Code: `e6135e89`; previous commit `2d53023b`.
- Purpose: close the binding-level negative matrix over the real hidden
  verifier and wire the lookup plan through the parent-plan constructor in the
  hidden fixture.
- This is partial progress. The rounding claim and the model-level plan
  assembly are not part of this step.

## Validation basis

- The seq16 x3 rule is not applicable here: no production model plan carries
  the lookup, no seq512 logic and no all-case structural path changes. The
  workspace suite, the focused tests and the byte-identical norm-free proof
  are the acceptance evidence.

## Changes

- The hidden fixture proximity main code is sized so the full digit-set
  multiplicity message fits (one more tree level), and the fixture builds its
  lookup plan with `NormLookupPlan::from_round` from the rounding layout and
  the real parent codes, mask code, group, oracle and tree identifiers, tree
  material, padding and table identity. The fixture's own mask tree rows are
  derived from the norm and lookup mask budgets. The constructor keeps failing
  closed when a parent code cannot hold the bit rows.
- Binding matrix through the real hidden verifier: changed bit root, changed
  mask root, changed bit sample answer, changed mask opening position, a
  missing second-stage lookup carrier, a changed committed digit source, and
  the earlier key value, leaf row target, leaf point, multiplicity count, bit
  end value and transcript round tampering. All are rejected.
- Unit matrix: duplicate physical key sources are rejected; keys read from two
  distinct physical messages; a changed committed digit breaks the transcript;
  a `d = 1` row has zero range digits (u, s, r and the four offset/complement
  digit rows), with the sum and mean magnitude rows still carrying their
  values.
- The verifier-visible proof is checked not to carry the private key vector.
  The existing SEND openings of the rounding and mask trees are unchanged and
  follow the established hiding budget.

## Four-column mapping

| Requirement | Mechanism | zkIE landing | Correspondence / gap |
| --- | --- | --- | --- |
| Changed source digit | Keys read from the committed rounding cells | Changed-digit test; row identity on committed bits | The transcript product no longer matches the committed multiplicity |
| Cross position/slot reorder | Plan fixes every key to one physical cell | Multi-message read test; plan source order | The key product is order independent, so the plan, not the order, is the binding |
| Changed root | Roots absorbed before the second stage | First-message bit and mask root tampering | First-stage seal check rejects |
| Missing source or claim | Decode and presence checks are plan driven | Missing second-stage carrier; duplicate source test | Missing material fails closed |
| Padding and table | Padding byte and table identity are plan values bound by the digest | Constructor binds both; key claim tampering rejects | No prover-chosen padding or table |
| Multiplicity and challenge | Binary counts and transcript rounds | Multiplicity count and transcript round tampering | Rejected by the recomputed claims |
| Multiple physical sources | One source per position and digit | Distinct-message read test; production key map | Each source reads its own message cell |
| Duplicate references and bounds | Unique physical references, padding entries, exact key count | Duplicate source test; key-map padding and boundary checks | Fail closed |
| Private digit plaintext | Private rows stay in the witness | Key-vector absence check in the proof bytes | Only the existing SEND openings are visible |
| `d = 1` boundary | Zero range digits | `unit_valid_row_has_zero_offset_digits` | u = s = r = 0; sum/mean magnitudes remain |

## Results

- `cargo test --workspace`: 48 suites, 865 passed, 0 failed, 17 ignored.
- Focused `proof::lookup`: 13 passed; `proof::hidden_norm_psc`: 19 passed;
  `norm_lookup_carrier_proves_and_verifies`: 1 passed;
  `merged_tests`: 34 passed.
- Hard gate: the norm-free toy output-head proof is 45,080 B, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`,
  byte-identical to the baseline dump.

## Explicitly deferred

- Model-level plan assembly: the constructor takes the parent values, but no
  production model builds and uses this lookup plan yet.
- The rounding claim itself; the digit byte lookup is not yet claimed proven
  as exact integer rounding.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
