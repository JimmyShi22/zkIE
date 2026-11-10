# fold-projection r3 - 0ab31a6e / 4dad9114

Trace of the FOLD field projection fixture after removing the fresh coverage
path.

## Change

Code commit `0ab31a6e` ("Remove the fresh coverage path from the FOLD projection
fixture"), based on `4dad9114`. The production verify entry is unchanged.

- `FoldHandoff` no longer accepts a caller-supplied fresh witness for coverage.
- The public `fresh_covered` and `fresh_contribution` fields are deleted; a
  caller cannot mutate a flag to turn a projection into a handoff.
- `handoff_statement()` always fails closed with `UnsupportedPrimitive`.
- Only `projection_statement()` remains, and only for the local algebraic test.
- The module and NOTES state that no result can be mistaken for a valid
  FOLD-to-CS/BASE handoff before the fresh term is authenticated against its
  commitment and the full statement.

The previous package `t002-s6b3b-r2-review.tar.gz` sha256
`aaebc9a0f781fcb2c93c3e651ed57912546a1798108f1f0355dabbf47e7c97f8` is void and
retained unchanged. All earlier packages remain retained.

## Positioning

This step does not authenticate the fresh term. Every FOLD output stays on the
projection path: the projected component result describes only the algebraic
behaviour of the projected data and is not equivalent to the FOLD output claim.
The projection interface has no same-name external counterpart; it follows the
isolation rulings for this step.

`FoldHandoffAnchor` is caller-supplied. Its type and the handoff checks do not
establish independent trust; independent trust depends on the caller and the
production wiring, which remain unclosed. The length, recomputation and root
identity checks are internal consistency checks under that caller-trust
assumption.

## Field mapping

| `FoldOutput` field | treatment |
| --- | --- |
| `code`, `mask_code` | enter the projected row codes |
| `coefficients` | enters the projected main claim |
| `mask_coefficients` | enters the projected mask claim |
| `target` | enters the projected target |
| plaintext main/mask rows | not in `FoldOutput`; supplied by the projection fixture |
| `fresh_coefficients` | carried; length/recompute/root identity checked; not in `SingleRows`; the checks do not prove the fresh contribution is zero and do not bind a zero vector to the fresh root |
| `kept` | not in `SingleRows`; non-empty kept is rejected |
| `kept_factor` | internal consistency with `eps * product(1 - gamma)` |
| `gamma`, `eps`, `sum` | anchored to the caller's session state |
| `fresh`, `fresh_root`, `tree_number` | anchored to the caller's accepted start identity |
| `group`, `k`, `h`, `members`, `layout` | anchored to the caller's plan/layout |

There is no `fresh_covered` or `fresh_contribution` state. A handoff statement is
never produced; only the projection statement exists.

## Three negative cases

The unit test `no_output_obtains_a_valid_handoff` covers:

| case | projection algebra | authenticated zero | handoff |
| --- | --- | --- | --- |
| all-zero witness, not bound to the fresh root | dot is zero and the projected target equation is satisfiable | not available | `handoff_statement` is `UnsupportedPrimitive` |
| no witness | projection statement exists | not available | `handoff_statement` is `UnsupportedPrimitive` |
| nonzero witness | dot is nonzero; the projected equation is still satisfiable by a chosen fixture witness | not available | `handoff_statement` is `UnsupportedPrimitive` |

The zero-witness case distinguishes algebraic zero from an authenticated zero:
a zero dot product over an unbound vector is not an authentication of the fresh
term. The nonzero case shows that deleting the fresh term can still leave a
satisfiable projected equation; that is non-equivalence with the FOLD output
claim, not a failure of the projection equation. No state in this module means
"authenticated zero fresh contribution".

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
| kept rows or non-empty kept form | no-kept subset | `StatementMismatch` or `UnsupportedPrimitive` |
| any attempt to obtain a handoff | no handoff path | `UnsupportedPrimitive` |

## Tests and measurements

- Fold projection unit: 4 passed.
- Proximity session unit: 12 passed.
- Regressions: foldeq 38, column 9, edge 2.
- Real gpt2-seq16 projection: 1 passed (56.01 s test, 56.83 s wall, VmHWM
  4,240,208 kB; component session 0 ms, registration count 7, CS positions
  `[1, 0, 5]`, BASE positions `[6, 6, 3, 7]`).
- Real gpt2-seq16 3a + projection in one process: 2 passed (56.66 s, VmHWM
  8,433,592 kB).
- Real gpt2-seq16 column regression: 1 passed (75.09 s, VmHWM 6,376,372 kB;
  column session 362 ms).
- Workspace: 443 passed, 0 failed, 8 ignored.
- No proof bytes were produced; no performance gain, hiding or soundness bound
  is claimed.

## Validation scope

Only `gpt2-seq16` is used. The FOLD input row and the projected plaintext rows
are fixtures. The three seq16 cases were not run because this step does not
change the shared three-stage path or seq512 logic; the gpt2 fixture is the
dependency case for the projection. No five-case three-stage run was made.

## Unclosed items

1. Asset-to-FOLD witness derivation: the FOLD input row is a plan fixture, and
   the projected plaintext rows are a fixture chosen to satisfy the projected
   target. The real FOLD end witness is not connected.
2. Fresh authentication: the fresh term is not authenticated against its
   commitment or the full statement. There is no authenticated-zero state and
   no valid handoff path.
3. Handoff equivalence: the projected equation is not equivalent to the FOLD
   output claim; deleting the fresh term is a projection, not a handoff.
4. Real BAT/statement and real FOLD output: the candidate output comes from a
   FOLD session whose input row is fixture.
5. Pending CS claims are registered but not proved by a later proximity test.
6. Kept rows, fresh mask rows and a second mask row remain unsupported; folding
   is unsupported and rejected.
7. Real oracle group, real SEND sample declaration and follow-up checks, real
   random masks/hiding, all stage counts and production security parameters are
   not connected.
8. Random symbol checks remain conditional on drawn positions.
9. Single-shot is not claimed.
10. Production wiring: `verify_request` unchanged; this is not an accepted proof
    format.

## Artifacts

- `tests/fold-handoff-unit.txt`
- `tests/proximity-session-unit.txt`
- `tests/fold-projection-real.txt` (local fixture path replaced with `<fixture>`)
- `tests/proximity-session-real-both.txt` (local fixture path replaced with
  `<fixture>`)
- `tests/regression-foldeq.txt`
- `tests/regression-column-unit.txt`
- `tests/regression-column-real.txt` (local fixture path replaced with
  `<fixture>`)
- `tests/regression-edge.txt`
- `logs/cargo-test-workspace.txt`
- No binary artifacts were produced or deleted.

The 3a r3 package remains retained unchanged. This round is a new frozen
evidence set; the previous 3b r2 package is void.
