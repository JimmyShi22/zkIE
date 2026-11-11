# lin-bind implementation round — bounded work, reuse and parallel exact units

## Scheme

Three-stage hidden output-head relation, implementation-only round. The changes
are bounded lifetimes, row and tree reuse, copy elimination, order-preserving
bounded parallel scheduling of independent exact-domain units, and verifier-side
duplicate-state removal. No relation, commitment, challenge order, protocol
parameter or hiding-boundary change. Every cluster keeps `proof.bin`
byte-identical.

- Current commit: `3b3e8ecc`
- Previous commit: `759db42b`
- Round starting commit: `bdc2e7d8`
- Fixture: `/data/lin-fixture`, formal hidden session, GPT-2 output head
- Command: `cargo test --release -p zkie --test lin_session
  selected_gpt2_output_head_request_session -- --ignored --nocapture`

## Purpose

Reduce the exact-closure prove/verify cost without changing the proof content.
The measured targets were prove wall time, verifier wall time and peak memory on
one machine, one fixture and one build per comparison. Proof bytes and the
absorbed transcript are the hard invariant.

## Validation basis

This round touches implementation paths only. It does not modify seq512 logic or
the five-case structure. `seq16×3` here means three no-profile reruns of the same
GPT-2 formal fixture (`gpt2-seq16`), not three cases; only `gpt2-seq16` is covered.
The five-case set was not run because no seq512 logic and no all-case structural
path changed; the five-case set belongs to the full acceptance basis. The three
no-profile runs after the final cluster and the instrumented profile are
archived here. All GiB values in this note are KiB / 1048576.

The test machine was shared during the parallel-scheduling cluster. The
no-profile runs span 179,441 / 319,262 / 367,556 ms prove. The same-basis
representative is the run closest to the idle rerun (367.6 s versus 363.3 s),
not the fastest run. Both the representative and the idle rerun are recorded.
CPU share rose from 117% to 254–271%, so the parallel gain is real rather than a
clock or fixture change.

The idle rerun peak RSS is 207,153,712 KiB (197.6 GiB), +215,232 KiB (+0.104%)
over the 206,938,480 KiB (197.4 GiB) preceding value. This is inside the agreed 0.5% noise band and
is recorded as noise, not as a memory increase. No proof byte changed in any
cluster: 9,939,144 B, sha256
`f8f512c0e67407210f19dd92d038fd4c71a6bac3a12e1166cd4825d9413215f5`.

## Cluster measurements

| scheme point | hidden prove ms | hidden verify ms | whole test s | max RSS KiB | proof bytes |
|---|---:|---:|---:|---:|---:|
| fresh worktree at `bdc2e7d8` | 36,536 | 9,054 | 211.21 | — | 8,973,672 |
| shared point vector + row reuse, before (`fb9c31e6`) | 963,386 | 186,847 | 1,703.60 | 336,385,392 | 9,939,144 |
| shared point vector + row reuse, after | 955,806 | 182,916 | 1,692.66 | 336,303,764 | 9,939,144 |
| runtime-tree reuse + copy removal, before (`abc3f8dd`) | 955,806 | 182,916 | 1,692.66 | 336,303,764 | 9,939,144 |
| runtime-tree reuse + copy removal, after | 535,523 | 143,750 | 1,152.65 | 206,938,480 | 9,939,144 |
| bounded parallel exact units, before (`759db42b`) | 535,523 | 143,750 | 1,152.65 | 206,938,480 | 9,939,144 |
| bounded parallel exact units, run A | 179,441 | 42,007 | 488.36 | 217,570,556 | 9,939,144 |
| bounded parallel exact units, run B | 319,262 | 41,660 | 624.23 | 207,330,064 | 9,939,144 |
| bounded parallel exact units, run C (representative) | 367,556 | 52,635 | 706.47 | 207,064,604 | 9,939,144 |
| idle rerun at `759db42b` | 363,333 | 52,769 | 702.29 | 207,153,712 | 9,939,144 |
| verifier profile + duplicate-state removal, before | 362,908 | 52,635 | 701.02 | 207,138,524 | 9,939,144 |
| verifier profile + duplicate-state removal, after 1 | 363,951 | 50,086 | 688.70 | 181,961,776 | 9,939,144 |
| verifier profile + duplicate-state removal, after 2 | 355,184 | 50,453 | 641.20 | 169,393,348 | 9,939,144 |
| final no-profile seq16 run 3 | 355,755 | 50,607 | 642.64 | 169,395,556 | 9,939,144 |
| final instrumented profile | 355,164 | 50,589 | 640.72 | 170,170,260 | 9,939,144 |

The fresh worktree at `bdc2e7d8` predates the exact-closure commits, so its
proof is 8,973,672 B, not the closed 9,939,144 B. The closure adds the exact
per-unit range relation, the hidden argmax/winner relation and the output
binding; the 36.5 s baseline therefore compares a smaller proof and a smaller
relation. It confirms only that the pre-closure baseline is not a worktree
artifact; the ratio to the closure-bearing 355 s indicates magnitude, not a
controlled speedup. It is not used as a proof-content-equivalent cluster gate.
All cluster comparisons use the immediately preceding run of the same proof
content.

## Final segmented profile

Instrumented cold run at `3b3e8ecc`:

- first stage 227,882 ms (64.2%): bits commit 85,171 ms, map commit 84,273 ms,
  closure answers 23,950 ms, map answers 23,764 ms, closure openings 102.6 ms.
- second stage 99,858 ms (28.1%): proximity 47,486 ms (BAT 14,138 ms, fold
  rounds 30,876 ms, group tables 3,438 ms), sample rows 10,290 ms, plain rows
  13,498 ms, closure PSC 23,967 ms.
- shared 10,548 ms; verify 50,589 ms; verifier negative closure 100,451 ms.
- table bytes 268,435,456, one table pass, down from 805,306,368 before the
  round.

Compared with the pre-round profile (`ef46c560`): prove 940,640 → 355,164 ms
(−62.2%), first stage 399,363 → 227,882 ms (−43.0%), second stage
512,842 → 99,858 ms (−80.5%), proximity 242,207 → 47,486 ms (−80.4%).

## Attribution notes

- The runtime-tree cluster saved 420,283 ms, more than the 317.3 s predicted for
  tree reuse. The excess is attributed to copy removal and allocator churn in
  the same commit; the mechanism was checked as copy elimination.
- The verifier-side peak-memory drop (207,138,524 → 169,393,348 KiB, i.e.
  197.5 → 161.5 GiB across the two after runs) includes machine-load
  differences and the duplicate-proof-state removal; it is not presented as a
  single-factor attribution.
- The parallel-scheduling cluster used 179 / 319 / 367 s runs on a shared
  machine. The representative choice and the idle rerun are stated above; no
  best-run number is used as the cluster result.
- No equivalence, security or production-performance claim follows. The exact
  closure still costs 355 s prove and 50 s verify on this fixture; the remaining
  serial hotspots are bits/map commit and the proximity fold rounds.

## Conclusions and open items

Bounded work and reuse removed the copy-driven memory peak and the repeated tree
construction; bounded parallel scheduling reduced the remaining independent
exact-domain work without changing proof bytes. Joint-view hiding, opening-union
rank evidence, production commit/prove/verify wiring and a controlled
diagnostic remain open. The verifier negative section is still ~100 s.

## Artifacts

- `baseline-fresh-worktree.txt` — fresh `bdc2e7d8` baseline.
- `cluster-shared-vector-row-reuse.txt` — before/after record for the first
  cluster; the original `/tmp` log was overwritten, so the session record is
  archived with a provenance note.
- `cluster-f1f3-after.txt` — runtime-tree/copy-removal after-run tail.
- `run-1-no-profile.txt`, `run-2-no-profile.txt`, `run-3-no-profile.txt` — final
  `seq16×3` no-profile runs.
- `idle-rerun-759db42b.txt` — idle rerun for the parallel cluster and its
  +0.104% peak-RSS noise check.
- `profile-final.txt` — final instrumented segment profile.

`proof.bin` (9,939,144 B) remains on the test machine and is not committed; no
binary artifact was deleted for this directory. All textual records above are
committed.
