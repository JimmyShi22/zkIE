# Lookup key source plan

## Scheme and baseline

- Scheme: `lookup-key-source-plan`.
- Code: `e76cabde`; previous commit `dc9aab2f`.
- This is partial progress on the lookup key source. The lookup plan maps each
  key index to a physical committed rounding cell with a public coefficient and
  offset; the prover reads the key vector from those rows and builds the private
  key and product witness. No key message or key root is committed. The
  multiplicity-bit and mask messages stay as before.

## Four-column mapping

| Requirement | Mechanism anchor | zkIE landing | Correspondence / gap |
| --- | --- | --- | --- |
| Key to physical source | each key names its committed message cell and a public transform | `LookupKeySource`, `LookupKeyPlan::validate` | No virtual tree or root |
| Padding and table identity | padding value and table identity fixed in the plan | `LookupKeyPlan` fields | Table identity enters validation |
| Prover key read | keys read from the committed rows, not a key message | `LookupKeyPlan::read`, `prove_lookup` | Rows are the unencoded rounded messages |
| No key commitment | the key message, key oracle and key root are gone | `LookupCommitPlan`, `LookupMessages`, `LookupRoots` | Only bits and mask remain |
| Challenge absorption | bits and mask roots before alpha | `prove_lookup` / `verify_lookup` | Key root is not absorbed |
| Unit coverage | key plan and rows built directly in tests | lookup tests | Rows are synthetic in this step |

## Results

- `cargo test --workspace`: 48 suites, 850 passed, 0 failed, 17 ignored.
- Focused `lookup`: 4 passed, 0 failed.
- Norm-free byte gate: 45,080 B before and after, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical.
- Verification set: module-level block, no end-to-end path change; no seq512
  case was run.

## Scope and next

- The verifier-side key-leaf expansion into linear claims on the rounding
  oracle and its BAT/FOLD registration are next.
- The key plan is not yet wired to the norm round messages in the hidden path;
  the tests read synthetic rows.
- No claim that the digit byte proof is complete.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
