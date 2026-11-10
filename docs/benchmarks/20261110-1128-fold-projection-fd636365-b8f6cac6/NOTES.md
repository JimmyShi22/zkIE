# fold-projection - fd636365 / b8f6cac6

Trace of the FOLD field projection fixture rework.

## Change

Code commit `fd636365` ("Treat the FOLD handoff as a field projection fixture"),
based on `b8f6cac6`. The production verify entry is unchanged.

- `FoldHandoff` is explicitly a field projection fixture, not an equivalent
  FOLD-to-CS/BASE handoff.
- `handoff_statement()` returns a statement only when a caller-supplied fresh
  witness makes the fresh contribution to the target exactly zero. Otherwise it
  returns `UnsupportedPrimitive`; only `projection_statement()` exists for
  algebraic testing.
- `fresh_contribution` and `fresh_covered` are recorded. The length,
  recomputation and root identity checks do not prove the fresh contribution is
  zero.
- The real gpt2-seq16 test exercises the projection path: it asserts that no
  valid handoff is produced, then runs the local component on the projected
  data and reports only its algebraic behaviour.
- The anchor is documented as caller-supplied and caller-trusted; its checks are
  internal consistency checks under that assumption, not an independent trust
  establishment.

The previous package `t002-s6b3b-review.tar.gz` sha256
`bef9b7c81bcfcf17da4da569c4618ca5f2c1c7ae38eda2b88249b4e3437880f6` is void and
retained unchanged.

## Field projection

| `FoldOutput` field | treatment |
| --- | --- |
| `code`, `mask_code` | enter the projected row codes |
| `coefficients` | enters the projected main claim |
| `mask_coefficients` | enters the projected mask claim |
| `target` | enters the projected target |
| plaintext main/mask rows | not in `FoldOutput`; supplied by the projection fixture |
| `fresh_coefficients` | carried; length/recompute/root identity checked; not in `SingleRows`; these checks do not prove the fresh contribution is zero |
| `fresh_contribution` | computed only from a caller-supplied fresh witness; `Some(value)` records it |
| `fresh_covered` | true only when the caller-supplied witness makes the contribution exactly zero |
| `kept` | not in `SingleRows`; non-empty kept is rejected |
| `kept_factor` | internal consistency with `eps * product(1 - gamma)` |
| `gamma`, `eps`, `sum` | anchored to the caller's session state |
| `fresh`, `fresh_root`, `tree_number` | anchored to the caller's accepted start identity |
| `group`, `k`, `h`, `members`, `layout` | anchored to the caller's plan/layout |

`handoff_statement()` is available only in the `fresh_covered` case.
`projection_statement()` is the projection product used for algebraic testing.

## Fresh contribution boundary

When no fresh witness is supplied, the fresh term is uncovered, `fresh_covered`
is false and no handoff statement is produced. The real fixture test asserts
this for the candidate gpt2 output.

When a caller-supplied fresh witness makes the contribution nonzero, the
projection target still deletes the fresh term. The projected equation can be
satisfied by a chosen fixture witness, but that equation is not equivalent to
the FOLD output claim, which includes the fresh term. The negative test
`fresh_contribution_must_be_covered_for_a_handoff` states this explicitly: it
asserts that the projection equation is satisfiable while `handoff_statement()`
is still rejected, because deleting the term is not equivalence. The negative is
about non-equivalence, not about a failed projection equation.

## Anchor boundary

`FoldHandoffAnchor` is supplied by the caller. The type and the checks do not
make it independently trusted; independent trust depends on the caller and the
production wiring, which remain unclosed. The length, recomputation and root
identity checks in the handoff are internal consistency checks under the
caller-trust assumption.

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
| fresh contribution nonzero | handoff coverage | no handoff statement (`UnsupportedPrimitive`) |
| fresh contribution not supplied | handoff coverage | no handoff statement (`UnsupportedPrimitive`) |

## Tests and measurements

- Fold handoff unit: 4 passed.
- Proximity session unit: 12 passed.
- Regressions: foldeq 38, column 9, edge 2.
- Real gpt2-seq16 projection: 1 passed (55.80 s test, 55.93 s wall, VmHWM
  4,239,304 kB; component session 0 ms, registration count 7, CS positions
  `[1, 0, 5]`, BASE positions `[6, 6, 3, 7]`).
- Real gpt2-seq16 3a + projection in one process: 2 passed (56.44 s, VmHWM
  8,432,444 kB).
- Real gpt2-seq16 column regression: 1 passed (75.14 s, VmHWM 6,377,288 kB;
  column session 364 ms).
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
2. Fresh contribution: the fresh term is not verified against a real fresh
   witness. The projection deletes it from the target; the projected equation is
   not equivalent to the FOLD output claim.
3. Handoff equivalence: no valid FOLD-to-CS/BASE handoff is established while
   the fresh term is uncovered.
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
evidence set; the previous 3b package is void.
