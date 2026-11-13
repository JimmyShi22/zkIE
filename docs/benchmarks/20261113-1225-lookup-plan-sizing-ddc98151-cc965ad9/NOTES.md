# Lookup plan sized from the parent plan

## Scheme and baseline

- Scheme: `lookup-plan-sizing`.
- Code: `ddc98151`; previous commit `cc965ad9`.
- Purpose: derive the whole lookup plan from the rounding layout and the
  parent plan values instead of caller-supplied shapes, tree plans and
  opening counts.
- This is partial progress. The rounding claim itself and the production model
  wiring are not part of this step.

## Validation basis

- The seq16 x3 rule is not applicable here: no production plan carries the
  lookup, no seq512 logic and no all-case structural path changes. The
  workspace suite, the focused tests and the byte-identical norm-free proof
  are the acceptance evidence.

## Changes

- `NormLookupParams` carries the parent plan values: the main and mask codes,
  the group and oracle identifiers, the bit and mask tree numbers, the tree
  leaf class, salt bytes, run and configuration, the public padding byte and
  the table identity.
- `NormLookupPlan::from_round` derives the key and plane counts from the digit
  slot count, builds the digit key map, sizes the multiplicity-bit message
  from the parent main code, derives the mask row count from the mask budget,
  builds the mask tree and fixes one opening per message.
- The constructor fails closed when the parent main code cannot hold the bit
  message rows, when a parameter collides (equal oracles or tree numbers, zero
  identifiers, non-byte padding, zero table identity) or when the round layout
  cannot be laid out in its message.
- `lookup_mask_rows` is the single mask-size derivation; the test checks the
  free-entry budget and the tree symbol count against it.
- A test commits the constructed message plans from a synthetic bit witness,
  checks the derived sizes and the digest binding, and exercises the
  fail-closed paths.

## Four-column mapping

| Requirement | Mechanism | zkIE landing | Correspondence / gap |
| --- | --- | --- | --- |
| Shapes from the parent plan | Main and mask codes plus tree material are plan values | `NormLookupParams`; `NormLookupPlan::from_round` | No caller-supplied bit or mask shapes remain |
| Key and plane counts | Derived from the committed digit count | `lookup_parameters` inside the constructor | Key map still covers every digit slot |
| Mask budget | Free-entry budget of the parent mask code | `lookup_mask_rows` | Same derivation in the constructor and the test |
| Message feasibility | Parent main code must hold the bit rows | `main_rows <= 2^log_n` check | Fails closed for a too-small parent code |
| Commit material | One opening per message; trees from the parent values | `LookupCommitPlan` built in the constructor | A synthetic commit of both messages succeeds |

## Results

- `cargo test --workspace`: 48 suites, 861 passed, 0 failed, 17 ignored.
- Focused `proof::hidden_norm_psc`: 18 passed; `proof::lookup`: 10 passed;
  `norm_lookup_carrier_proves_and_verifies`: 1 passed.
- Hard gate: the norm-free toy output-head proof is 45,080 B, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`,
  byte-identical to the baseline dump.

## Explicitly deferred

- Wiring the constructor into a production model plan (the toy parent code is
  too small for the full digit set; the constructor now fails closed there).
- The rounding claim; the digit byte lookup is not yet claimed proven as exact
  integer rounding.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
