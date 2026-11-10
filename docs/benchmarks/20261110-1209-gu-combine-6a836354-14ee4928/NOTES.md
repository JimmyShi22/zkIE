# gu-combine - 6a836354 / 14ee4928

Trace of the isolated G_u declaration registration and BAT combination
demonstration.

## Change

Code commit `6a836354` ("Add the isolated G_u declaration combination
demonstration"), based on `14ee4928`. The production verify entry is unchanged.

- New isolated `gu_combine_session` module runs the fixed pre-FOLD
  G_u/record/column flow and takes the declarations registered by that same
  session.
- The session identity (reader root/configuration/shape/table identity), G_u
  root/oracle/group/tree, queries and all declaration coefficients/targets are
  fixed into the session chain snapshot; the next draw is the combination
  challenge `x`.
- All declarations are combined by powers of `x` into one pending row.
- A small `FoldeqSession::chain()` inspection accessor and a `open_gu_flow`
  helper return the real pre-FOLD session state. They are isolated inspection
  paths, not production acceptance paths.
- The combination is not connected to the 3c FOLD source; the 3c handoff stays
  fail-closed and no target is rewritten.

## Declaration sources

| declaration | source | note |
| --- | --- | --- |
| sample | `receive_gu_answer` in the same session | answer is a fake demonstration value (`0x31`) |
| record | `join_gu_record` in the same session | pending declaration under the bound index name |
| column | `read_queries` + `join_gu_read_columns` in the same session | opens authenticated only against the column root |

Supported declarations all come from the same session reader, G_u identity,
shape and opened columns. Not supported and listed in the outcome:
multi-oracle SEND, a real SEND provider, and a fresh-mask declaration. No extra
fixtures fill these gaps.

## Order and derivation

1. The fixed pre-FOLD session runs: first query, gamma, self-reported G_u root,
   shared seeds, fake sample answers, record join, query draw, column
   authentication and column join.
2. The session chain snapshot absorbs the session identity, G_u root, queries
   and every declaration's kind, oracle, group, coefficients and target.
3. The next chain draw is `x`; declarations are combined as
   `sum(x^i * coefficient_i)` and `sum(x^i * target_i)`.

The absorbed values are the already-bound session values; no additional digest
or unbound hash layer is introduced. The same declarations and session state
reproduce the same `x`; changing a declaration changes the challenge.

## G_u and BAT semantics

- The G_u row is a pending declaration. Registration binds the claimed root and
  coefficients/targets but does not prove the G_u value, its source or the
  declaration correctness.
- The G_u root is self-reported and the SEND answer is fake; both are flagged in
  the outcome.
- Column openings authenticate only against the column root; they do not
  authenticate the G_u root.
- BAT combines declarations; it does not judge truth. A wrong declaration on the
  valid path remains pending and is not rejected here.
- No later proximity/relation check is connected, so a successful combination
  is not a check pass and no proof is accepted.

## 3c connection boundary

The combined row is not connected to the 3c FOLD source in this step. The 3c
source uses its own fixture claim/target relation; rewriting its target to match
this combination would not establish real consistency. The independent
registration/combination result is kept, `three_c_connected` is false, and the
3b projection `handoff_statement` remains rejected. Fresh, old-mask and random
tails stay on their existing fixture/fail-closed boundaries.

## Tests and measurements

- G_u combine unit: 4 passed.
- GroupRow unit: 6 passed.
- Fold projection unit: 4 passed.
- Proximity session unit: 12 passed.
- Regressions: foldeq 38, column 9, edge 2.
- Real gpt2-seq16 G_u combination: 1 passed (54.47 s test, 55.29 s wall, VmHWM
  4,237,576 kB; combination session 362 ms).
- Workspace: 453 passed, 0 failed, 10 ignored.
- No proof bytes were produced; no performance gain, hiding or soundness bound
  is claimed.

Real session values:

| item | value |
| --- | --- |
| reader root | `83a62c3995b33823d29b5465f7a271d69b2cc9a4f73d4c403ee9e96c01f623b3` |
| G_u root | `4747474747474747474747474747474747474747474747474747474747474747` |
| G_u oracle/group/tree | `11` / `5` / `7` |
| queries | `[4, 7, 7, 4]` |
| sample / record / column declarations | `1` / `1` / `4` |
| combined declarations | `6` |
| declaration order | sample, record, column, column, column, column |
| challenge x | `Fp3([690512898248879634, 13892910747627118869, 6852393297469236421])` |

The roots are local/session demonstration values, not authentic G_u content.

The full asset opening verification and table build are included in the test
cost; this is a single `gpt2-seq16` case and is not a production performance or
memory measurement.

## Negative coverage

| change | layer | observed result |
| --- | --- | --- |
| declaration group/oracle differs | deterministic declaration check | `StatementMismatch` |
| declaration coefficient shape differs | deterministic declaration check | `StatementMismatch` |
| declaration target wrong on the valid path | BAT combination | still pending; combination succeeds, no truth decision |
| declaration changed | challenge derivation | different `x` |
| second run with the same data | deterministic rerun | same transcript, `x` and combined row |

The wrong-target case is deliberately not a rejection: BAT does not judge truth.
No random check is written as a must-reject condition; the demonstration uses
fixed session data and the test-only fake SEND answer.

## Validation scope

Only `gpt2-seq16` is used. The session is the same isolated column session used
by the earlier components; the G_u root and SEND answer are demonstration
values. The three seq16 cases beyond the dependency case were not run because
this step does not change the shared three-stage path or seq512 logic. No
five-case three-stage run was made.

## Unclosed items

- Real G_u authenticity/source and a real SEND provider.
- A later proximity/relation check for the pending declarations and the
  combined row.
- A valid 3c FOLD-to-CS/BASE handoff; `handoff_statement` still rejects.
- Fresh-mask handoff and trusted fresh authentication.
- Folding-tail derivation; random tails did not participate in FOLD folding.
- Hiding and single-shot semantics.
- Production wiring: `verify_request` unchanged; this is not an accepted proof
  format.
- Multi-oracle SEND and any declaration not listed above.

## Artifacts

- `tests/gu-combine-unit.txt`
- `tests/group-row-unit.txt`
- `tests/fold-handoff-unit.txt`
- `tests/proximity-session-unit.txt`
- `tests/gu-combine-real.txt` (local fixture path replaced with `<fixture>`)
- `tests/regression-foldeq.txt`
- `tests/regression-column.txt`
- `tests/regression-edge.txt`
- `logs/cargo-test-workspace.txt`
- No binary artifacts were produced or deleted.

The earlier 3c r2 package remains retained unchanged. This is a new frozen
evidence set; old packages are not deleted.
