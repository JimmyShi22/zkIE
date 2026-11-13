# Norm offset and remainder relations

## Scheme and baseline

- Scheme: `norm-round-linear-relations`.
- Code: `f0e9ce1a`; previous commit `8d86836d`.
- This is partial progress on the norm rounding relation. The relation domain
  adds the mean and variance offset-complement identity `u+s=d-1` and the
  remainder-offset identity `r=lo+u`. The prover maps read the integer core of
  the same position; the BAT rows bind both endpoints to the committed rounding
  slots of the same position, with the public constants folded into the target
  and every mask entry taken from the same committed mask message. The rounding
  group also enters the proximity member list and the own-mask row count follows
  the derived size.

## Four-column mapping

| Requirement | Mechanism anchor | zkIE landing | Correspondence / gap |
| --- | --- | --- | --- |
| `u+s=d-1` | linear identity on committed rounding slots | map relations 4/5, BAT rows against the round member | Mean and variance each |
| `r=lo+u` | remainder offset identity | map relations 6/7, BAT rows with the public `lo` | Mean and variance each |
| t2Lin-style binding | coefficients from the plan and prior transcript, constants to target | `round_coefficients`, per-instance eq weights | Verifier constructs the same rows |
| Physical identity | round slots are one committed message per position | proximity member list includes the round group | Sample rows, trees, bound |
| Derived mask size | rows follow the relation count | `required_rows` from the previous step | No fixed row number |

## Results

- `cargo test --workspace`: 48 suites, 849 passed, 0 failed, 17 ignored.
- Focused `norm`: 25 passed, 0 failed.
- Norm-free byte gate: 45,080 B before and after, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical.
- Verification set: module-level block, no end-to-end path change; no seq512
  case was run.

## Coverage

- Every norm hidden positive now runs with relations 0-7; the honest fixture
  proves and verifies through the actual hidden verifier.
- The negative matrix covers every instance, including the new relations: a
  changed endpoint, point value or encoded message is rejected.
- A plan that lists a rounding relation without the rounding group is rejected.

## Scope and next

- The eight digit reconstruction relations (8-15), the lookup forest, the
  BAT/FOLD binding of their claims and the binding-level negatives are next.
- No claim that integer rounding is proven.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
