# digit-lookup-commit

- Scheme: digit-lookup-commit (the digit lookups' message plans).
- Based on: the digit lookup plan (code d4f7d05f, trace 8ff416e7).
- Purpose: each digit lookup needs the plans of its two messages before the
  segment can commit and check them.
- Change:
  - `DigitLookupPlan` carries `commit: LookupCommitPlan`; `DigitLookupParams`
    holds the parent code, mask code, group, the two oracles, the two tree
    numbers, the leaf class, salt, run and configuration.
  - `digit_lookup_plan(range, plane, params)` sizes the lookup, enumerates the
    key sources, builds the multiplicity-bit message shape over the parent code
    and the mask rows and tree over the parent mask code, and validates the
    plan against the range object.
  - The plan check refuses a message plan that leaves the group, shares the
    digit message's identity or cannot carry its values and mask entries.
  - Tests additionally pin the group, both identities, the bit message shape
    and the mask row count.
- Key numbers: workspace 1108 passed / 0 failed (workspace.log); digit lookup 3
  passed (digit-lookup.log); hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
  (hard-gate.log, hard-gate.sha256).
- Conclusions: both calls have complete, validated message plans.
- Boundaries: the calls are not yet committed, proved or verified in the
  segment, and their claims are not consumed by the proximity group; no
  smallness or range closure is claimed. Not pushed.
