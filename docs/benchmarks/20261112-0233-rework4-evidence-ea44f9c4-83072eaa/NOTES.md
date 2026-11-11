# rework4 evidence round: asset lifetime, joint view, transcript order

## Scheme and baseline

- Scheme: `rework4-evidence`.
- Code: `ea44f9c4`; previous candidate closeout `83072eaa`.
- Fixture: none in this round; the production proof code is unchanged from
  the rework3 basis `1f80e682`.
- Test command: `cargo test --workspace`.

## Purpose

The round adds documentation and test-only evidence: the bounded asset
lifetime inventory, the full-view linear-system and composition accounting,
the corrected Eval-order table, the `chi = 0` probability term, the
output-view boundary table, the executable fail-closed list, the
changed-message-with-original-tree test and a second group-table shape. It
does not change the proof relation, commitments, challenge order, parameters
or hiding boundary.

## Validation basis

The production code is unchanged; only test modules and documentation changed.
No seq512 logic is touched. The rework3 `seq16×3` runs and their proof hash
remain the performance/proof evidence; this round adds the workspace run
below.

## Workspace

`cargo test --workspace`: 48 test binaries, 726 passed, 0 failed, 17 ignored.
The complete log is `workspace.txt`.

## Proof invariant

The rework3 proof dumps remain 9,939,144 B with sha256
`f8f512c0e67407210f19dd92d038fd4c71a6bac3a12e1166cd4825d9413215f5`.
No production file in the proof path changed in this round.

## Open items

The security document still marks full joint-view hiding, the opening-union
rank proof, the composition theorem, output confidentiality, the trusted X
source and production wiring as `[UNPROVEN]` or `[GAP]`.

## Artifacts

- `workspace.txt`: complete workspace test log.
- No binary artifact; no proof dump was produced in this round.
