# fold-handoff - f84cc118 / ebe35642

Trace of the candidate FOLD output handoff into the local CS/BASE algebraic
component.

## Change

Code commit `f84cc118` ("Hand off the candidate FOLD output to the CS/BASE
component"), based on `ebe35642`. The production verify entry is unchanged.

- New `FoldHandoff` maps the FOLD session output into the local claim and target
  on the verifier side.
- The verifier supplies a trusted anchor built from its own FOLD plan, group
  layout and accepted start message; the handoff checks the output against it
  before building any claim.
- The CS/BASE component consumes `FoldHandoff::statement()` on the verifier side
  and `FoldHandoff::single_rows()` with prover fixture rows on the prover side.
- The column session outcome now carries the candidate `FoldOutput` and its
  trusted anchor for the integration test.
- The local result remains `ProximityComponent::ComponentComplete`; the FOLD
  next stage stays fail-closed and the production entry is not switched.

This is a candidate FOLD output handoff. The FOLD input row is still a
self-reported fixture, and the plaintext CS witness is a fixture; this trace
does not claim a real FOLD output, a real FOLD witness, a complete FOLD result or
a complete IOPP/CS/BASE chain.

## Field mapping

| `FoldOutput` field | treatment |
| --- | --- |
| `code` | enters `SingleRows.code` |
| `mask_code` | enters `SingleRows.mask_code` |
| `coefficients` | enters `SingleRows.claim` |
| `mask_coefficients` | enters `SingleRows.mask_claims[0]` |
| `target` | enters `SingleRows.target` |
| plaintext main/mask rows | not in `FoldOutput`; supplied by the prover fixture to `single_rows` |
| `fresh_coefficients` | carried as `FoldHandoff.fresh_claim`, not in `SingleRows`; recomputed from the anchored gamma/shape and compared |
| `kept` | not in `SingleRows`; a non-empty kept form is rejected as unsupported |
| `kept_factor` | handoff identity; checked against `eps * product(1 - gamma)` |
| `gamma` | handoff identity; anchored to the verifier session |
| `eps` | handoff identity; anchored to the verifier session |
| `sum` | handoff identity; anchored to the accepted start message |
| `fresh` | handoff identity; anchored to the accepted start message |
| `fresh_root` | handoff identity; anchored to the accepted start message |
| `tree_number` | handoff identity; anchored to the accepted start message |
| `group` | handoff identity; anchored to the plan |
| `k`, `h` | handoff identity; anchored to the plan |
| `members` | handoff identity; anchored to the layout |
| `layout` | handoff identity; anchored to the verifier plan layout |

The implemented subset accepts only no-kept outputs. A kept output or a
non-empty kept form is rejected, not silently dropped.

## Data flow

1. The verifier runs the 2e/2f FOLDEq column session and obtains the candidate
   `FoldOutput` plus the trusted anchor (plan shape, layout, group, accepted
   start identity and sum, session gamma, epsilon and sum).
2. `FoldHandoff::from_output` validates the anchor, identity, shapes, kept
   status, gamma/eps/sum, kept factor and fresh coefficients.
3. `FoldHandoff::statement()` builds the `ProximityStatement` consumed by the
   local CS/BASE verifier.
4. The prover fixture supplies plaintext rows to `single_rows`, which takes the
   claim, mask claim and target from the handoff rather than from the rows.
5. The 3a CS/BASE component runs its opening authentication, CS registration and
   local BASE linear/symbol checks.

The plaintext rows are a fixture chosen to satisfy the mapped target; this is
not the FOLD end witness. The local BASE check is therefore a fixture relation
over the mapped claim/target, not a check of the real FOLD main/mask/fresh
witness.

## Boundary

The implemented subset is a candidate handoff. It is not an iopp_fault-equivalent
plan, not a real BAT/statement derivation, not a real FOLD output and not a
complete BASE/IOPP/proof acceptance. `FoldOutput::next_stage` remains
fail-closed; `verify_request` is unchanged.

## Negative coverage

| tampered input | layer | observed result |
| --- | --- | --- |
| target | local BASE linear check | `InvalidProof` |
| `coefficients` | local BASE linear check | `InvalidProof` |
| `mask_coefficients` | local BASE linear check | `InvalidProof` |
| `group` | handoff anchor | `StatementMismatch` |
| `fresh_root` | handoff anchor | `StatementMismatch` |
| `fresh_coefficients` | handoff recomputation | `StatementMismatch` |
| `gamma`, `eps` or `sum` | handoff anchor | `StatementMismatch` |
| `layout` or `members` | handoff anchor | `StatementMismatch` |
| truncated claim or mask coefficient vector | handoff shape | `ShapeMismatch` |
| kept rows or non-empty kept form | handoff no-kept subset | `StatementMismatch` or `UnsupportedPrimitive` |

Unit tests also cover a wrong shape, missing/truncated fields and a kept output
with a kept anchor. The real test covers target/claim tampering at the component
relation and identity/fresh/shape/kept tampering at the handoff.

## Tests and measurements

- Fold handoff unit: 3 passed.
- Proximity session unit: 12 passed.
- Regressions: foldeq 38, column 9, edge 2.
- Real gpt2-seq16 FOLD handoff: 1 passed (58.45 s test, 59.26 s wall, VmHWM
  4,239,992 kB; component session 0 ms, registration count 7, CS positions
  `[1, 0, 5]`, BASE positions `[6, 6, 3, 7]`).
- Real gpt2-seq16 CS/BASE component plus handoff in one process: 2 passed
  (55.71 s, VmHWM 8,434,572 kB).
- Real gpt2-seq16 column regression: 1 passed (72.86 s, VmHWM 6,377,344 kB;
  column session 362 ms).
- Workspace: 442 passed, 0 failed, 8 ignored.
- No proof bytes were produced; no performance gain, hiding or soundness bound
  is claimed.

## Validation scope

Only `gpt2-seq16` is used. The FOLD input row and the plaintext CS witness are
fixtures. The three seq16 cases were not run because this step does not change
the shared three-stage path or seq512 logic; the real gpt2 fixture is the
dependency case for the handoff. No five-case three-stage run was made.

## Unclosed items

1. Asset-to-FOLD witness derivation: the FOLD input row is a plan fixture, and
   the plaintext CS witness is a fixture chosen to satisfy the mapped target.
   The real FOLD end witness (main/mask/fresh rows) is not connected.
2. Fresh and kept contributions: fresh coefficients are anchored, but their
   target contribution is not checked against a fresh mask witness; kept outputs
   are unsupported and rejected.
3. Real BAT/statement and real FOLD output: the candidate output comes from a
   FOLD session whose input row is fixture.
4. Pending CS claims are registered but not proved by a later proximity test;
   a consistently re-committed CS output is caught only by the sampled BASE
   symbol check.
5. Kept rows, fresh mask rows and a second mask row remain unsupported; folding
   is unsupported and rejected.
6. Real oracle group, real SEND sample declaration and follow-up checks, real
   random masks/hiding, all stage counts and production security parameters are
   not connected.
7. Random symbol checks remain conditional on drawn positions.
8. Single-shot is not claimed.
9. Production wiring: `verify_request` unchanged; this is not an accepted proof
   format.

## Artifacts

- `tests/fold-handoff-unit.txt`
- `tests/proximity-session-unit.txt`
- `tests/fold-handoff-real.txt` (local fixture path replaced with `<fixture>`)
- `tests/proximity-session-real-both.txt` (local fixture path replaced with
  `<fixture>`)
- `tests/regression-foldeq.txt`
- `tests/regression-column-unit.txt`
- `tests/regression-column-real.txt` (local fixture path replaced with
  `<fixture>`)
- `tests/regression-edge.txt`
- `logs/cargo-test-workspace.txt`
- No binary artifacts were produced or deleted.

The 3a r3 package `t002-s6b3a-r3-review.tar.gz` sha256
`3a075309827790c2b736baf9a1c3011d8f153606ca51d7e4f028c434556c1056` is retained
unchanged; this 3b package is a new frozen evidence set.
