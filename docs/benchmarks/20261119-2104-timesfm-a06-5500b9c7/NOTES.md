# timesfm A06 minimal cut acceptance

- Scheme: the timesfm side of the segmented/streaming bounded-survival line, as
  the minimal cut approved for this lane: the same shared mechanism commits the
  larger slice landed, cherry-picked onto the timesfm branch with no local
  redesign.
- Branch state: base `066eed15`; cherry-picks `f9f7bfcd` (fold capacity release
  after the last round), `891e4bca` (frozen table column reads in bounded
  4096-row chunks), `da6a91c8` (parameter assets above a 1 MiB read cap are
  refused before the file is opened), `8177c8dc` (stored table commit bound and
  asset row buffer gate), `274ecdd7` (bounded table source identity bound in
  the segment statement; the timesfm statement carries no segment identity, so
  the check is inert here); component tests `5500b9c7`.
- Component evidence: the picks' unit tests pass (frozen table reads 8/8,
  parameter read 3/3, fold 75/75, stored table 43/43) and the new timesfm
  component tests pass 3/3: the chunked rstd value column equals the column
  rebuilt from the public file bit for bit with every range read inside the
  chunk bound, reads at or past the real length and the domain bound are
  refused, and a parameter asset above the read cap is refused before opening
  while a truncated file is an asset mismatch.

## Round numbers

| round | commit ms | prove ms | verify ms | artifact bytes | prove vmhwm KiB | verify vmhwm KiB |
|---|---|---|---|---|---|---|
| 1 (cold) | 151,994 | 374,012 | 133,415 | 5,259,835,925 | 58,567,796 | 37,938,492 |
| 2 | 151,417 | 372,380 | 132,937 | 5,259,830,226 | 58,567,724 | 37,959,648 |
| 3 | 154,246 | 372,070 | 132,513 | 5,259,834,647 | 58,566,224 | 37,942,344 |
| 4 | 152,888 | 373,271 | 133,382 | 5,259,811,867 | 58,572,040 | 37,939,896 |

Warm medians (rounds 2-4): commit 152,888 ms, prove 372,380 ms, verify
132,937 ms. Peak resident set size across all stages: 58,572,040 kB (GNU time,
includes the stage children). The proof bytes stay in the same band as the
accepted segment rounds (the mechanism change is a read-path and capacity
change, not a transcript change).

## Byte account (round 1)

| file | bytes |
|---|---|
| proof/statement.bin | 133,216 |
| proof/public-inputs.bin | 131,080 |
| proof/timesfm-norm-segment-carrier.bin | 1,644,035,480 |
| proof/proof.bin | 3,613,502,964 |
| proof/consumer_proof.bin | 2,028,416 |
| proof/manifest.json | 4,704 |
| proof/COMPLETE | 65 |
| total (the proof-bytes metric) | 5,259,835,925 |

The carrier is not a commit invariant: every commit samples fresh table
randomness, so the carrier and the surrounding artifact bytes vary across
rounds and setups. The acceptance therefore records the materialized verifier
PASS with `output_binding=expected` and the unchanged GT digest
`cc72805b7a845cb2b8505d6092dbb79477d083394a03c99cdac7f5665de56ba3` instead of a
carrier digest invariant; the per-round sizes above are the honest bytes.

## Hard gate and workspace

The twenty tracked `proof-45080.bin` copies are 45,080 B each with sha256
`7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb`; the docs
tree carries only this evidence directory. The workspace suite is green: 61
suites, 1,337 passed, 0 failed.

## Peak composition and the no-abuse rationale

The peak is essentially unchanged from the accepted pre-cut rounds (58,572,040
kB against 58,581,632 kB, about 9.6 MB lower), and that is the expected result:
this segment has no output-head table and no 2^18-level read surface. The
largest structures are the 2^17 code domains and the 2^16-row member groups of
the single L0 norm; the gain table has 2,048 columns and the frozen rstd table
is read by point lookups and, in the component test, in bounded chunks. The
peak composition is the norm-only own chain itself: the single owned norm
witness (borrowed by the second stage, never cloned), the member group tables
and their BAT/FOLD/CS rows, and the two message stages; the tail clamp and
padding rows already use the sparse unit representation. The fold capacity
release and the chunked reads land the shared bounded-survival mechanism on
this branch, but there is nothing large to stream in this segment, so the peak
does not move materially. The change is not used to claim a timesfm memory
reduction; the numbers above are recorded as they are.

## Boundaries

- Single-machine evidence; no push in this round.
- Time and memory are recorded, not signed off; the prove and verify walls sit
  in the same band as the accepted segment rounds.
- The cut follows the approved shared commits; no segment relation, parameter
  or proof byte was changed by this round.
