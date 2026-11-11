# Verifier proof-message reuse and final before/after measurement

## Scheme and baseline

- Scheme: `verifier-path-reuse`, round 2 (final).
- Code: run basis `759db42b`; baseline is the instrumented run recorded in
  `20261111-2342-verifier-path-profile-759db42b`; round 1 is recorded in
  `20261111-2356-verifier-path-reuse-r1-759db42b`.
- Fixture: `/data/lin-fixture/gpt2-seq16`, run with
  `ZKIE_SESSION_FIXTURE=/data/lin-fixture ZKIE_SESSION_FORMAL=1
  ZKIE_PROFILE_HIDDEN=1 ZKIE_SESSION_PROOF_DUMP=/tmp/p24-after2-hidden.bin`.
- The fixture, command, build profile and machine are the same in every round.
- One same-machine run; the positive verifier segment was also measured in
  round 1 to gauge the run-to-run spread.

## Purpose

Remove the remaining verifier-side duplicate materialization: the in-memory
proof message carried the prover's committed value rows and retained runtime
trees inside the closure groups, even though the verifier reads only the group
declaration, the roots and the SEND/opening messages. The message now carries
a committed-group declaration (the send group and its derived roots) and the
prover keeps the value rows and trees as witness state, which the fixture
releases before verifying.

## Changes

- `ClosureFirstMessages` carries `CommittedGroupMessage { group, roots }` for
  the slack, packed-bit and product groups instead of `CommittedValueGroup`.
  Encoding, decoding, roots and every verifier check are unchanged; the
  serialized message bytes are identical.
- The prover-built proof message no longer clones the value rows or the
  retained trees; the witness still owns them and is released before the
  positive verification unless the optional agreement check is enabled.
- The round-1 changes remain: released verifier outcome and encoded proof copy
  before the negative replay, removed dead shared-stage sample rows, and one
  equality vector per shared point in the row builders.

## Results

Same-machine before/after:

| Quantity | Before | After | Delta |
|---|---|---|---|
| hidden prove | 362,908 ms | 355,184 ms | -7,724 ms (-2.13%) |
| hidden verify | 52,635 ms | 50,453 ms | -2,182 ms (-4.15%) |
| whole fixture test | 703.30 s | 642.68 s | -60.62 s (-8.62%) |
| process maximum resident set | 207,138,524 KiB | 169,393,348 KiB | -37,745,176 KiB (-18.22%) |
| hidden proof bytes | 9,939,144 | 9,939,144 | unchanged |
| hidden proof sha256 | f8f512c0...3215f5 | f8f512c0...3215f5 | unchanged |

The proof sha256 is
`f8f512c0e67407210f19dd92d038fd4c71a6bac3a12e1166cd4825d9413215f5` in both
runs; the proof is byte-identical, and roots, openings, challenge order and
absorbed transcript are unchanged.

Positive hidden verify (52,635 -> 50,453 ms):

| Segment | Before ms | After ms |
|---|---|---|
| verify.first (openings 1.5) | 301 | 299 |
| verify.shared | 7,416 | 6,795 |
| verify.shared.column_rows | 3,768 | 3,661 |
| verify.shared.plain_rows | 449 | 469 |
| verify.shared.proximity | 3,153 | 2,624 |
| verify.second | 44,914 | 43,354 |
| verify.second.sample_rows | 10,239 | 10,363 |
| verify.second.plain_rows | 15,244 | 13,424 |
| verify.second.proximity | 19,430 | 19,566 |
| verify.hidden_lin | 52,631 | 50,448 |

Memory in the positive verify drops with the proof message: the verifier
starts at 67,174,512 KiB resident (was 87,214,432 KiB) and the second-stage
rows reach 123,902,772 KiB (was 143,875,372 KiB). Thread count is 66 in every
formal segment (64 workers plus the main and test threads).

Negative replay section:

| Segment | Before | Round 1 | Final |
|---|---|---|---|
| negative section | 144,613 ms | 136,746 ms | 100,939 ms |
| inside the verifier (10 entries) | 105,380 ms | 102,108 ms | 100,921 ms |
| eight proof clones | ~39,233 ms (derived) | 34,630 ms | 10.4 ms |
| resident set at section start | 99,834,732 KiB | 79,613,968 KiB | 67,279,092 KiB |
| proximity fold resident maximum | 175,588,092 KiB | 149,932,164 KiB | 130,263,920 KiB |

The before-value for the clones is derived from the section total minus the
time inside the verifier; round 1 measured the clones directly at 4.33 s each.
The final clones are ordinary message-sized copies, so the remaining negative
section time is the verifier replays themselves.

Run-to-run spread: the positive verifier measured 50,086 ms in round 1 and
50,453 ms in the final round (+0.73%); the hidden prove measured 362,908,
363,951 and 355,184 ms over the three rounds. The machine load was low and no
other workload was running during the measurements.

## Tests

The final source tree passes, with no failures:

| Command | Suites | Passed | Failed | Ignored |
|---|---|---|---|---|
| `cargo test -p zkie-ops` | 5 | 206 | 0 | 0 |
| `cargo test -p zkie` | 19 | 131 | 0 | 17 |
| `cargo test --workspace` | 48 | 716 | 0 | 17 |

The multi-GiB formal fixture is an ignored integration test and was run
explicitly as described above.

## Conclusion and open items

The verifier-side fixes are implementation-level: no proof relation,
commitment object, challenge order, protocol parameter or hiding boundary
changed, and the proof remained byte-identical. Process high-water memory is
now driven by the prover, not by the verifier replay.

The remaining verifier wall time is the protocol-required BAT/FOLD/BASE input
construction: the second-stage equality rows, the closure row materialization
and the proximity replay. Those segments were already parallelized on
order-preserving batches; further wall-time reduction would need a scheduling
or row-representation change outside this round's scope. The base-field
membership warranty, the joint-view hiding argument and the production
commit/prove/verify wiring remain open as before.

## Logs

- `fixture.txt`: the complete final instrumented formal fixture run.
- `tests-zkie-ops.txt`, `tests-zkie.txt`, `tests-workspace.txt`: the test
  suites above.
