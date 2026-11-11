# Hidden verifier segment profile on the real formal fixture

## Scheme and baseline

- Scheme: `verifier-path-profile`.
- Code: run basis `759db42b` plus environment-gated verifier segment scopes;
  disabled by default and inert when the variable is unset.
- Fixture: `/data/lin-fixture/gpt2-seq16`, run with
  `ZKIE_SESSION_FIXTURE=/data/lin-fixture ZKIE_SESSION_FORMAL=1
  ZKIE_PROFILE_HIDDEN=1 ZKIE_SESSION_PROOF_DUMP=/tmp/p24-before-hidden.bin`.
- One same-machine run; no performance change is claimed for this round.

## Purpose

Before changing the verifier path, measure where its wall time and memory go:
verify setup, trace check, the first/shared/second hidden stages, the
BAT/FOLD/BASE proximity replays, the ledger checks and the negative-proof
construction and replay.

## Results

- hidden prove 362,908 ms; hidden verify 52,635 ms.
- whole fixture test 703.30 s; process maximum resident set 207,138,524 KiB.
- hidden proof 9,939,144 B, sha256
  `f8f512c0e67407210f19dd92d038fd4c71a6bac3a12e1166cd4825d9413215f5`.
- diagnostic path: table commit 30,557 ms; trace check 16,217 ms; diagnostic
  prove 17,703 ms; prepared verify 8,618 ms; byte-level verify 24,286 ms.

Positive hidden verify (52,635 ms), nested scopes overlap:

| Segment | ms |
|---|---|
| verify.first (openings 1.5) | 301 |
| verify.shared | 7,416 |
| verify.shared.column_rows | 3,768 |
| verify.shared.plain_rows | 449 |
| verify.shared.proximity | 3,153 |
| verify.second | 44,914 |
| verify.second.sample_rows | 10,239 |
| verify.second.plain_rows | 15,244 |
| verify.second.proximity | 19,430 |
| verify.hidden_lin | 52,631 |

The verifier starts at 87,214,432 KiB (83.2 GiB) resident and raises the
process to its 197.6 GiB high-water mark (207,138,524 KiB / 1048576); the
second-stage rows reach 137.2 GiB resident.

Negative replay section (144,613 ms), ten verifier entries (nine full):

| Segment | ms | calls |
|---|---|---|
| verify.output_head_lin_hidden | 105,380 | 10 |
| verify.second.sample_rows | 20,143 | 2 |
| verify.second.plain_rows | 30,612 | 2 |
| verify.second.proximity | 38,745 | 2 |

The difference between the 144,613 ms negative section and the 105,380 ms
inside the verifier is 39,233 ms of construction and decode outside the
verifier entry. Inspection shows every negative starts from
`hidden_proof.clone()`, and the in-memory proof message carries the committed
value groups, including the packed-bit and product message rows and retained
runtime trees, although those are witness state. The clone cost is quantified
in the next round.

## Conclusion and open items

The verifier hotspots are the second-stage row construction and proximity
replay, the second-stage sample rows, and the shared-stage column rows. The
process high-water mark is driven by the negative replay. The round that
follows releases prover/verifier buffers earlier and reuses shared equality
vectors; a further round removes the witness material from the proof message.

## Logs

- `fixture.txt`: the complete instrumented formal fixture run, including all
  four profile blocks.
