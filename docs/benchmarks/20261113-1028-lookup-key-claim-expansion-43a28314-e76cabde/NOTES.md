# Lookup key claim expansion

## Scheme and baseline

- Scheme: `lookup-key-claim-expansion`.
- Code: `43a28314`; previous commit `e76cabde`.
- This is partial progress on the key-leaf expansion. The key plan expands the
  lookup key claim at its random point into one linear term per physical
  committed rounding cell, a public padding constant and the claim mask terms.
  No virtual tree or root is created.

## Four-column mapping

| Requirement | Mechanism anchor | zkIE landing | Correspondence / gap |
| --- | --- | --- | --- |
| Point weights | equality weights over the claim point | `eq_weights`, `expand_claim` | Length must equal the key count |
| Physical terms | one term per source oracle/position/row/place | `KeyLinearTerm` | Caller registers on its member |
| Public transform | coefficient and offset folded per source | term coefficient, constant | Padding included in the constant |
| Mask terms | the claim's mask form is carried through | `KeyClaimExpansion::masks` | BAT registration next |
| Unit check | expansion equals a directly computed value | `key_claim_expands_to_linear_terms` | Synthetic cells |

## Results

- `cargo test --workspace`: 48 suites, 851 passed, 0 failed, 17 ignored.
- Focused `lookup`: 5 passed, 0 failed.
- Norm-free byte gate: 45,080 B before and after, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical.
- Verification set: module-level block, no end-to-end path change; no seq512
  case was run.

## Scope and next

- The BAT/FOLD registration of the expanded terms and the hidden round message
  source wiring are next.
- The test uses synthetic cells; no digit byte proof is claimed.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
