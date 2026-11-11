# Verifier buffer release and shared equality-vector reuse

## Scheme and baseline

- Scheme: `verifier-path-reuse`, round 1.
- Code: run basis `759db42b`; baseline is the instrumented run recorded in
  `20261111-2342-verifier-path-profile-759db42b`.
- Fixture: `/data/lin-fixture/gpt2-seq16`, same command and environment as the
  baseline.
- One same-machine run.

## Purpose

Apply the verifier-side implementation-level fixes identified by the profile:
release the prover witness and the verifier outcome before the negative replay,
remove the dead sample-row materialization in the shared verifier stage, and
reuse one equality vector per shared point instead of one evaluation per
column.

## Changes

- The fixture releases the prover witness (retained runtime trees and value
  rows) before the positive verification unless the optional agreement check
  is enabled, and releases the verifier outcome, the encoded proof copy and the
  witness before the negative section.
- `verify_shared` no longer builds a gu sample-row vector that is immediately
  shadowed; the gu member rows are built where they are consumed.
- `own_plain_rows` and `closure_plain_rows` build the equality vector for a
  shared point once and index it, instead of recomputing the evaluation for
  every column and for each of the three reconstruction rows.

Values, roots, openings, challenge order and the absorbed transcript are
unchanged.

## Results

- hidden prove 363,951 ms; hidden verify 50,086 ms (-2,549 ms, -4.84%).
- whole fixture test 690.23 s (-13.07 s, -1.86%).
- process maximum resident set 181,961,776 KiB (-25,176,748 KiB, -12.16%).
- hidden proof 9,939,144 B with the same sha256 as the baseline.
- positive verify start resident set 79,562,120 KiB, down from 87,214,432 KiB;
  the second-stage rows reach 136,237,648 KiB, down from 143,875,372 KiB.
- negative section 136,746 ms, down from 144,613 ms; the ten verifier entries
  take 102,108 ms and the eight proof clones take 34,630 ms (4.33 s each),
  confirming that the in-memory proof carries heavy witness material.

## Conclusion and open items

Releasing buffers removed 12.2% of the process high-water mark and the equality
reuse removed 4.8% of the positive verifier wall time. The dominant remaining
construction cost is the proof clone: the proof message still embeds the
committed value groups, whose value rows and retained trees are prover witness
state. The next round keeps only the committed group declaration in the proof
message.

## Logs

- `fixture.txt`: the complete instrumented formal fixture run.
