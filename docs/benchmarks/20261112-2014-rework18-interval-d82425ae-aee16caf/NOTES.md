# rework18: interval planner, accounting and the clean two-stage run

## Scheme and baseline

- Scheme: `rework18-interval`.
- Code: `d82425ae`; previous scheme head `aee16caf` (clean Q3 trace).
- Fixture: `/data/lin-fixture`, GPT-2 seq16 formal hidden session.
- Command:
  `ZKIE_SESSION_FIXTURE=/data/lin-fixture ZKIE_SESSION_FORMAL=1
  /usr/bin/time -v cargo test --release -p zkie --test lin_session
  selected_gpt2_output_head_request_session -- --ignored --nocapture`.
- One formal run on an otherwise idle machine.

## Purpose

Implement TURN 455 R1: replace the f64 + 0.2-bit planner decision with the
reference conservative interval formulas and search, expose the counted
round terms and their union caveat, pin the shared join count, and record the
nonce-search tail. R2/R3/R4 boundaries are in the bridge evidence notes. The
real three-stage reorder run and the five-case matrix are separate trace
directories.

## Plan values

- Own stage 0: `(17,65,536,8,192)/(17,512,256)`, `k=14,h=2`, Johnson
  m=616, positions 252, work 24.
- Own last: `(17,65,536,259)/(17,16,384,259)`, `k=h=0`, Johnson m=595,
  positions/r 259/259.
- Shared join: two stages, positions 2/2, work 0, pinned; its proximity is
  backed by the table columns at 603 positions with 12 work bits.

## Results

- hidden prove 389,493 ms; hidden verify 64,434 ms; proof 65,432,832 bytes;
  sha256
  `6d38a1e34f898a05090d9cf261f0eb2bc3ade27d782425a8021b0749412ec3c4`;
  VmHWM 169,342,676 KiB; Rust test 1,197.38 s; wall 20:12.00.
- Ledger before LastBase: 139 events / 18,129 logical / 18,058 unique;
  minimum remaining 3; exhausted 0. After LastBase: 145 / 18,912 / 18,841;
  minimum remaining 0; exhausted 1 (`Cs` oracle 1, tree 2049, rank 259,
  unique 259). The one exhausted key is the square Vandermonde `q = r`
  case; all positions are counted and the next event would be rejected.
- The earlier 254-position candidate measured 145/19,050/19,042 with two
  exhausted keys; the interval search removes two stage-0 positions and one
  exhausted key, and reduces the proof by 702,952 bytes.

## Accounting (bits)

`reference_interval_accounting_is_visible` prints the counted terms:
14 own FOLD rounds and 2 shared FOLD rounds at 2^-128.000035, own BASE at
2^-128.003677, own query 2^-128.138472, own CS 2^-128.002200, table columns
2^-128.164982, and the remaining FOLD/BAT/SAMPLE/range/PSC terms. Worst
single round 2^-128.000035; naive union sum 2^-123.692866. The union sum does
not meet 128 bits and is reported as such; the reference round-by-round rule
is separately stated. The prover nonce-search tail is now work_bits+8 bits
(capped at 40), fixing the 37% honest failure of the fixed 2^24 search at 24
work bits (run 1 failed this way).

## Tests

- `cargo test --workspace`: 48 suites, 800 passed, 0 failed, 17 ignored.
- `cargo check --workspace`: clean.
- Focused: interval 3/3, planner 3/3, hidden_send 45/45.

## Open items

- Naive union bound 2^-123.69 < 2^-128; parameter/composition decision is
  with the expert.
- Shared join proximity is explained by the table column test, not proved as
  a subsumption theorem.
- Production ledger entry remains harness-only (R3 boundary).
- No push in this round.

## Artifacts

- `run2-two-stage.txt`, `accounting.txt`, `workspace.txt`, `check.txt`,
  `focused-*.txt`, `summary.txt`.
- The 254-position prior run is not committed as a trace; its failure reason
  and numbers are in the bridge R1 note.
