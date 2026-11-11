# Exact integer output closure for the hidden output-head relation

## Scheme and baseline

- Scheme: `output-closure`.
- Run basis: commit `949c4e64` (the exact integer closure). The previous
  scheme commit is `bdc2e7d8` (independent claim and tilde masks with a public
  Dot residual). This trace directory is committed after the scheme commit
  and therefore names the scheme commit explicitly.
- Fixture: `/data/lin-fixture/gpt2-seq16`, the 4.16 GB production trace, run
  with `ZKIE_SESSION_FIXTURE=/data/lin-fixture ZKIE_SESSION_FORMAL=1`.
- Public closure parameters: `D = 40`, `b = 42`, `M = -(1 << 41)`,
  `B + 2(2^D - 1) + |M| < p`. The measured witness bounds on this fixture are
  `max |Y| = 622,653,884,972` and `max S = 2,143,462,175,194`, both inside the
  declared bounds.

## Purpose

Record the real-fixture execution of the exact output closure: the committed
slack, packed-bit, product and mask messages; the `u = Y + A`, `v = A - Y`
and slack bit decompositions; the comparison row `S_j = Z_a - Z_j - [j < a]`;
the public linear-functional binding of the product message to the bits; the
per-position booleanity product-sumcheck; and the relation-version-2 digest,
all checked by the same final verifier through the own-group
BAT/FOLD/CS/BASE test. The same run also executes the verifier negatives:
changed booleanity round, changed fresh-point functional, changed
slack/bit/product/mask roots, changed slack SEND answer, missing closure,
different trusted winner and truncated or trailing encodings.

## Results

Diagnostic path (unchanged measurement basis; the diagnostic setup keeps its
pre-closure fresh-mask row count):

- table commit 29,042 ms; trace verify 17,051 ms.
- diagnostic request proof 8,982,616 B; prove 34,264 ms; prepared verify
  8,302 ms; byte-level verify 24,492 ms.
- diagnostic manifest 4,170,807,683 B; total verification bytes
  4,170,765,190 B.

Formal hidden path:

- hidden prove 986,725 ms (16.4 min); hidden verify 196,644 ms (3.3 min).
- hidden proof 9,939,144 B.
- own-group FOLD rounds: 14.
- formal manifest 4,171,764,211 B
  (trace 4,161,782,574 + hidden proof 9,939,144 + public 42,493).
- commitment ledger: 655 logical openings; table reuse 0.
- whole fixture test: 1,772.25 s (29.5 min), 1 passed, 0 failed.
- whole-test peak resident set: 324,140,804 KiB (recorded from
  `/proc/<pid>/status` `VmHWM`).

Workspace release tests (the same source tree): 697 passed, 64 ignored,
0 failed across 48 suites. The full log is `workspace.txt`.

## Comparison with the previous scheme

The baseline `bdc2e7d8` formal measurement on the same fixture was hidden
proof 8,973,672 B, hidden prove 36,366 ms, hidden verify 8,690 ms and
whole-test peak RSS 19,232,416 KiB. The exact closure adds the packed-bit and
product messages, the dense random-point rows and the per-position
product-sumcheck: proof size grows by about 10.8%, prove time by about 27x
and verify time by about 23x on this shape, and peak memory is much higher.
The numbers are reported as measured; no gain is claimed. The exact per-cell
decomposition is the dominant cost, and the runtime/memory profile is an open
optimization item.

## Conclusion and open items

The final verifier accepts the honest real-fixture proof and rejects every
listed closure negative through the full verification path. This is not a
formal soundness or zero-knowledge proof: the reconstruction, comparison and
booleanity checks are random-point checks with Schwartz-Zippel terms; the
base-field membership of the committed reduced row remains the
activation/table warranty; the joint-view hiding argument, the opening-union
budget and the production `commit`/`prove`/`verify` wiring remain open. The
runtime and memory cost of the exact closure needs optimization before
production acceptance.

## Logs

- `fixture.txt`: the complete real-fixture run (diagnostic and formal
  sections, all verifier negatives, final result).
- `workspace.txt`: the complete workspace release test run.
