# group-row r2 - b1bcbc99 / 13fbc523

Trace of the main-row demonstration rework for the isolated GroupRow source.

## Change

Code commit `b1bcbc99` ("Downgrade the GroupRow witness to a main-row
demonstration"), based on `13fbc523`. The production verify entry is unchanged.

- The step is now explicitly a **main-row/message demonstration**. Folding-tail
  derivation (`fold_low`/`fold_gather_mask` equivalent) is not implemented.
- The stored main, old-mask and fresh tail fields are fixtures; they do not
  participate in FOLD folding and are marked **not folded / not checked** in the
  module, outcome and tests. The outcome cannot be read as a completed witness
  fold.
- The candidate witness derivation conditions are incomplete: random tails did
  not participate in FOLD folding. This does not change the T-002 final
  acceptance.
- Shape and configuration come from a verifier-fixed `GroupRowPlan`, not from
  the source's self-report; source construction checks the table against it.
- The fresh check is renamed and documented as local fixture consistency, not
  trusted fresh authentication.
- The `Arc::ptr_eq` assertion only records that the demonstration consumes the
  same object as the comparison; it is not independent authentication.

The previous 3c package `t002-s6b3c-review.tar.gz` sha256
`3f960dac0edef2afdb54305c39e06bad52ca83ab60aa6a15c9f1e50bac33c1fa` is void and
retained unchanged.

## Two comparisons

| comparison | data | conclusion |
| --- | --- | --- |
| main-row comparison | authenticated asset opening vs the fixed demonstration slice under the anchor premise | full main-row/message comparison; table commitment rebuilt from the verifier-fixed shape and configuration |
| fresh local consistency | fresh fixture vs a root rebuilt from the same fixture data | local consistency only; **not** trusted fresh authentication |

The main-row comparison uses the verifier's fixed `GroupRowPlan` shape and
configuration. The source checks that its table matches the plan. The source's
own reported shape/configuration is not the comparison input.

## Folding-tail boundary

| tail | status |
| --- | --- |
| main tail | fixture; not folded; compared as part of the main-row fixture data |
| old-mask tail | fixture; not folded; not checked |
| fresh tail | fixture; not folded; only local fresh-root consistency is checked |

No test treats the existence of a tail field as a folding-tail test. The real
and unit tests assert the `*_tail_folded` flags are false; the tail lengths are
not used as folding evidence.

## Negative coverage

| change | layer | observed result |
| --- | --- | --- |
| opening value changed | opening authentication | authentication rejection (`InvalidProof`/`StatementMismatch`) |
| valid re-commitment with different main data | independent main-row comparison | `StatementMismatch` |
| main messages changed | independent comparison | `StatementMismatch` |
| main tail changed | independent comparison | `StatementMismatch` |
| old-mask messages changed | FOLD claim input | target changes |
| old-mask tail changed | not folded / not checked | target unchanged |
| fresh-mask rows changed | fresh local consistency | root consistency fails |
| fresh tail changed | fresh local consistency | root consistency fails |
| round drawn before epsilon | transcript order | `InvalidSessionState` |
| verifier plan shape/configuration mismatch | source plan check | `StatementMismatch` |

The opening-authentication negative and the valid-recommitment comparison
negative are separate and named by layer. No random sample is written as a
must-reject condition; the fixed coins are test-only.

## Order and separate components

The fixed order is fresh mask commitment and start sum, then epsilon, then one
fixed round message followed by that round's gamma. Main messages, old-mask
messages and fresh-mask rows remain separate vectors; the tail fixtures remain
separate but not folded. Challenges are used by each component's role; not every
witness entry depends on epsilon.

## Tests and measurements

- GroupRow unit: 6 passed.
- Fold projection unit: 4 passed.
- Proximity session unit: 12 passed.
- Regressions: foldeq 38, column 9, edge 2.
- Real gpt2-seq16 main-row demonstration: 1 passed (56.58 s test, 56.73 s wall,
  VmHWM 4,758,880 kB; GroupRow session 723 ms; comparison flags asset opening,
  main rows, table commitment and fresh local consistency all true; same object
  true).
- Workspace: 449 passed, 0 failed, 9 ignored.
- No proof bytes were produced; no performance gain, hiding or soundness bound
  is claimed.

Real roots:

| root | value |
| --- | --- |
| asset root | `9a8be4edbbc390fd6844710399441788506c2f1f3a1fc9c9c95e19e067971832` |
| table root | `83a62c3995b33823d29b5465f7a271d69b2cc9a4f73d4c403ee9e96c01f623b3` |
| fresh mask root | `75087d1c1bcf8dd963dcd5f44c799e0ebb709c6a4c804bf627cb3344a4547342` |

The roots are local component commitments, not a real GroupRow/G_u/BAT
authentication.

## Validation scope

Only `gpt2-seq16` is used. The main rows come from the complete asset opening;
old-mask and fresh-mask data are fixed fixtures. G_u and BAT are skipped. The
three seq16 cases beyond the dependency case were not run because this step does
not change the shared three-stage path or seq512 logic. No five-case three-stage
run was made.

## Unclosed items

- Real GroupRow, G_u and BAT verification.
- Folding-tail derivation (`fold_low`/`fold_gather_mask` equivalent): not
  implemented; random tails did not participate in FOLD folding.
- Real random mask and trusted fresh-mask authentication; fresh is only locally
  consistent with a root rebuilt from its own fixture.
- Complete authentication of sampled columns.
- A valid 3b FOLD-to-CS/BASE handoff; `handoff_statement` still rejects.
- Real SEND sample declaration and follow-up checks.
- Pending CS claims are registered but not proved by a later proximity test.
- Request identity/source authentication.
- Full graph and all upstream/downstream relations.
- Proximity, hiding and single-shot semantics.
- Production wiring: `verify_request` unchanged; this is not an accepted proof
  format.

## Artifacts

- `tests/group-row-unit.txt`
- `tests/fold-handoff-unit.txt`
- `tests/proximity-session-unit.txt`
- `tests/group-row-real.txt` (local fixture path replaced with `<fixture>`)
- `tests/regression-foldeq.txt`
- `tests/regression-column.txt`
- `tests/regression-edge.txt`
- `logs/cargo-test-workspace.txt`
- No binary artifacts were produced or deleted.

The earlier 3b r3 package remains retained unchanged. This is a new frozen
evidence set; the previous 3c package is void.
