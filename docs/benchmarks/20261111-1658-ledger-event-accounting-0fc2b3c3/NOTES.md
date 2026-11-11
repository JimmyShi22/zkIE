# Authoritative opening-event accounting on the real formal fixture

## Scheme and baseline

- Scheme: `ledger-event-accounting`.
- Code: `0fc2b3c3`; baseline `1e748ab0` (output-closure trace).
- Fixture: `/data/lin-fixture/gpt2-seq16`, 4.16 GB production trace,
  `ZKIE_SESSION_FIXTURE=/data/lin-fixture ZKIE_SESSION_FORMAL=1`.
- One same-machine run; no performance claim.

## Purpose

Record the real-fixture execution after the ledger gained authoritative
opening-event accounting: every event by oracle/stream identity, logical draw
counts, tail-union/rank checks, seed reservation before proof and success after
full verification.

## Results

- hidden prove 967,161 ms; hidden verify 189,064 ms; fixture test 1,714.16 s.
- hidden proof 9,939,144 B.
- formal manifest 4,171,764,211 B = trace 4,161,782,574 + hidden proof
  9,939,144 + public 42,493.
- ledger: 143 events; 887 logical openings; 887 unique positions;
  table 603 logical / 603 unique; activation rows 16 with 0 openings;
  closure 196 logical; pre-closure 691 logical / 691 unique; table reuse 0.
- The previous undercount (655) is replaced by the event table: 691
  pre-closure + 196 closure = 887 logical openings, with activation rows no
  longer charged as openings.

## Boundaries

- Not a formal soundness/ZK proof; the witness bounds and random-point checks
  keep their Schwartz-Zippel terms.
- The ledger is exercised on the formal fixture; production commit/prove/verify
  wiring remains open.
- Joint-view hiding, opening-union rank evidence and the runtime/memory cost
  optimization remain open.
- No equivalence or equal-security claim.

## Logs

- `fixture.txt`: the complete formal fixture run.
