# Equality-weighted sumcheck variable

## Scheme and baseline

- Scheme: `psc-equality-weight`.
- Code: `b8e2650c`; previous commit `c9e1acb3`.
- This is partial progress on the shared sumcheck foundation needed by the byte
  range lookup. It adds the equality-weighted variable only; no new
  commitment, message, challenge order or verification condition is wired yet.

## Four-column mapping

| Requirement | Reference anchor (read) | zkIE landing | Correspondence / gap |
| --- | --- | --- | --- |
| A variable summed over the two Boolean points with equality weights `(1-z)g(0)+z g(1)`, `psi = 1`, fixed coefficient 0 | `blocks/iop/hp.h` `Weight::equality(z, d)` and `Weight::at_zero`; `blocks/iop/psc.h` `psc_vars` uses it for X variables | `crates/zkie-ops/src/proof/psc.rs` `Weight::equality(z, degree)`; `at_zero` now delegates to `equality(0, degree)` | Moments are `[1, z, z, ...]`, `psi = 1`, fixed coefficient 0, `fixed_inv = 1`; one implementation, no remaining zero-weight branch |
| No legacy path | project single-implementation rule | `at_zero` body deleted; all callers keep the same API and behavior | Existing product/dot/closure/norm calls compile unchanged |

## Results

- `cargo test --workspace`: 48 suites, 830 passed, 0 failed, 17 ignored.
- Focused `proof::psc`: 19 passed, 0 failed.
- Norm-free byte gate: 45,080 B before and after, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`;
  `cmp` reports the files identical.
- Added unit coverage for the equality functional, its `psi`, fixed
  coefficient, degree check and the `at_zero` equivalence.

## Scope and next

- This commit only extends the sumcheck variable. The summand forms
  (product, tie, leaves, bits), the kept-map output and the inner GKR
  sumcheck are the next part of the shared foundation.
- No new commitments, messages, challenge order or verification conditions;
  lookup and digit rows remain unwired. No round-half-up, soundness or
  complete-norm claim.

## Artifacts

- `workspace.txt`, `focused.txt`, `proof-bytes.txt`, `change.stat`, `NOTES.md`.
- No textual trace was deleted. No previous-scheme binary was deleted.
