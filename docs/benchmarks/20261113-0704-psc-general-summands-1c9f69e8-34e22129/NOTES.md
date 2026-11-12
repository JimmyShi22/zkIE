# General summand sumcheck layer

## Scheme and baseline

- Scheme: `psc-general-summands`.
- Code: `1c9f69e8`; previous commit `34e22129`.
- This is partial progress on the shared sumcheck foundation for the byte
  range lookup. It adds the four summand forms, the kept-map output claims and
  the inner sumchecks. It does not add any commitment, message, challenge
  order or verification condition.

## Four-column mapping

| Requirement | Reference anchor (read) | zkIE landing | Correspondence / gap |
| --- | --- | --- | --- |
| Summand forms product, tie, leaves and bits, with shown/kept/degree | `blocks/iop/psc.h` `Summand`, `psc_vars`, `psc_rows`; `blocks/iop/lkp.h` summands | `proof/psc_general.rs` `Summand::{Product, TieFirst, TieNext, Bits, Leaves}` with `shown`, `kept`, `degree`, `at` | Formulas follow the reference shape; tie has no X variables; leaves is affine in the kept map and reads a public table |
| Output claims: masked shown ends, kept-map terms, `(U)` target and hiding form | `psc.h` `PscOutput`, `psc_rows` | `GeneralOutput { point, hiding_form, claim_scale, row_target, ends, row_maps }` | Same algebra; the caller supplies the mask slots and binds every output claim |
| Equality-weighted variables | `blocks/iop/hp.h` `Weight::equality`; `psc.h` `psc_vars` | `proof/psc.rs` `Weight::equality`; `psc_general` uses arbitrary X weights | Equality weights are the variable functional; inner sumchecks carry the future-variable equality factors |
| Inner GKR sumcheck | `psc.h` caller-supplied `SumcheckProver`; `lkp.h` `LkpProver` | `InnerSumcheck` with `ProductInner`, `WeightedProductInner`, `BitsInner`, `WeightedBitsInner`, `LeavesInner`, `ScalarInner` | Product and bit trees, tie scalars and the affine leaf sum check are concrete; the tree-level lookup driver remains for the next step |
| No new commitment or order | TURN 481 boundaries | No caller outside the new module; existing product/dot/closure/norm unchanged | Norm-free bytes and existing callers stay unchanged |

## Results

- `cargo test --workspace`: 48 suites, 836 passed, 0 failed, 17 ignored.
- Focused `psc_general`: 6 passed, 0 failed; focused `proof::psc`: 19 passed.
- Norm-free byte gate: 45,080 B before and after, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical.

## Scope and next

- Arithmetic and claim layout only: no lookup table identity, key or
  multiplicity group, mask slots, message codec or transcript order is wired.
- Next: the lookup module (byte table, key/multiplicity messages, alpha/theta,
  tie and tree driver, mask slots), then the norm digit/remainder rows.
- No round-half-up, soundness or complete-norm claim.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
