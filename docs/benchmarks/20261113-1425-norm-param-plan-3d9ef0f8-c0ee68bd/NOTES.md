# Norm parameter table identities bound into the norm plan

- Scheme: `norm-param-plan`
- Change: code commit `3d9ef0f8` (norm plan field, validation, digest and
  bind-version upgrade; the parameter table digest encodes the committed reader
  inline; one plan test).
- Baseline: previous commit `c0ee68bd` (interface skeleton; trace `739369ad`).
- Purpose: make the parameter-table identities part of the trusted norm plan so
  their roots, configuration, shape and mapping are fixed and absorbed before
  any dependent challenge, as the frozen interface requires.

## What changed

- `NormPlan` gains the optional `params` tables plan; validation runs the
  frozen tables validation and requires both parameter widths to equal the
  padded column domain of the norm.
- The plan digest absorbs the tables plan digest; the bind version moves from
  6 to 7. The per-identity digest now encodes the committed reader (root,
  configuration, code/shape numbers, sampling distances and the record) inline,
  so digesting cannot silently fall back to a zero identity.
- A plan test checks that the digest changes when the tables plan is attached,
  and that a width that differs from the column domain, a swapped kind and a
  duplicated row inside one table are rejected.
- No message, witness or relation consumes the tables yet; the fixture still
  carries `params: None`.

## Evidence

- Focused: norm 33, rstd 16, lookup 24, norm_param_table 3 passed
  (`focused.txt`).
- Workspace: 48 suites ok, 898 passed, 0 failed, exit 0
  (`workspace.txt`).
- Hard gate: 45,080 bytes, sha256
  `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`
  (unchanged, `proof-bytes.txt`).

## Four-column mapping

| Requirement (TURN 498) | Change | Evidence | Boundary |
| --- | --- | --- | --- |
| table root/config/shape/mapping fixed before dependent challenges | plan field plus digest absorption; bind version 7 | digest-change plan test | no challenge draws the table yet |
| gain/bias shared across positions, one commitment | tables plan with one identity per kind, packed or separate tables | tables-plan validation tests | the fixture table build is the next step |
| mapping bound to the norm column domain | width equality with the plan columns | width-mismatch rejection test | the evaluation reduction is the next step |
| versioned plan/statement | bind version bump; old version rejected by the message check | existing version checks plus plan test | transcript fields are the next step |

## Key numbers

- Hard-gate proof bytes: 45,080 (unchanged, byte-identical sha256).
- Focused: 76 tests across the four filters.
- Workspace: 898 passed in 48 suites, exit 0.
- No wall time or memory claim in this round.

## Open items

- First/second message fields and the prover witness for the parameter tables;
  the fixture table built from read-only rows; column draws and openings; the
  ledger categories.
- Evaluation consumption through the LIN/FOLDEQ declarations; the output
  relations and the hd group.
- Production source closure remains a dependency; the i32 mapping flag is an
  assumption.
- No push in this round.
