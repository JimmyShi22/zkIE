# Limited output-head key: round accounting and caller-word items

## Scope and status

This note covers the limited output-head key relation of the local chain: one
committed table built from the output-head weights, its record and column
openings, the FOLD-to-CS stages, the final BASE and the seal. It is not a
full-model proof, not a single-shot decode proof and not a production tested
state. The table below is the plan's accounting; the implementation consumes
the same plan values, but its own round-by-round accounting is not mechanized
yet.

Target: 128 bits in every round. Work bits are counted only where the plan
assigns them; the four position rounds carry 12 bits each.

## The twelve caller-word rounds

An item is on the caller's word when its error bound is computed from a number
that the parameter file or the plan supplies and that nothing ties to the run
at that round. The error column is the bound without work; the effective bound
adds the round's work bits.

| # | round | actual value | trusted source | runtime constraint | count | error / work |
|---|---|---|---:|---:|---:|---|
| 1 | RECORD | table record, one answer per row | table shape and record seeds | record rows are rebuilt from the committed table and the seed; the table distances are not tied here | 1 | 2^-188.91 / 0 |
| 2 | LIN point | table v=10, answers=1, radius=16384, later radius=16384; sharing=1 | table plan and the linear shape | the statement's shape and the plan's table distances are checked; the distances remain the caller's | 1 | 2^-174.68 / 0 |
| 3 | SEND seed (table, folded row) | one seed, table and folded row lists | table and folded group shapes | the seed is drawn from the chain; the group list sizes are the caller's | 1 | 2^-170.46 / 0 |
| 4 | LOC (table, columns) | 603 positions | table v=10, answers=1, distances 16384/16384 | positions are drawn from the chain with 12 work bits; the verifier pins the count to the plan; openings are checked against the table root; the distances are the caller's | 1 | 2^-116.16 / 12 |
| 5 | SEND seed (table, Omega) | one seed, table and Omega lists | table and Omega group shapes | the seed is drawn from the chain; the group list sizes are the caller's | 1 | 2^-170.97 / 0 |
| 6 | BAT (folded row and Omega) | 607 rows | folded row and Omega group shapes | the BAT layout and group sizes are checked against the compiled plan; the row count is the plan's | 1 | 2^-173.53 / 0 |
| 7 | SEND seed (folded row and Omega, stage 1) | one seed | stage 1 group shapes | the seed is drawn from the chain; the following stage's list size is the caller's | 1 | 2^-169.73 / 0 |
| 8 | CS (folded row and Omega, stage 1) | 47 positions, 143 rows; main n=17 ell=4096 r=94; mask n=14 ell=64 r=47 | stage 1 compiled group and code lines | positions are drawn with 12 work bits and pinned to the plan; group codes and rows are checked against the compiled stage; the following stage's list size is the caller's | 1 | 2^-116.48 / 12 |
| 9 | SEND seed (folded row and Omega, stage 2) | one seed | stage 2 group shapes | as item 7 | 1 | 2^-170.21 / 0 |
| 10 | CS (folded row and Omega, stage 2) | 63 positions, 191 rows; main n=12 ell=256 r=63; mask n=12 ell=64 r=63 | stage 2 compiled group and code lines | as item 8 | 1 | 2^-116.01 / 12 |
| 11 | SEND seed (folded row and Omega, stage 3) | one seed | stage 3 group shapes | as item 7 | 1 | 2^-171.32 / 0 |
| 12 | CS (folded row and Omega, stage 3) | 57 positions, 173 rows; main n=12 ell=32 r=57; mask n=11 ell=64 r=57 | stage 3 compiled group and code lines | as item 8 | 1 | 2^-116.50 / 12 |

The remaining rounds of the trimmed once/shared plan are not caller-word
items: their bounds follow from the setting, the round's own fields or the
numbers the plan states beside them and the runtime compares.

## Shared-only derivation

Within one chain the rounds' errors are compared with the target one at a
time, not added to one another. With work counted, every round holds: the
largest round errors are 2^-128.00 in the FOLD rounds and the BASE lambda, and
the four position rounds hold at 2^-128.16, 2^-128.48, 2^-128.01 and
2^-128.50. The chain's smallest slack is 0.01 bit, in the stage 2 CS round.

Without work the same chain bottoms out at 2^-116.01, so the 12 work bits on
the four position rounds are not optional. A run that drops them does not meet
the 128-bit target and must not state the with-work figures.

The compiled bound on one request's proof, with the verifier's hash calls
counted, is

```
(t + q_V + 1) * 2^-128.00 + (17/2) * (t + q_V)^2 / 2^256
```

where t is the prover's hash calls and q_V the verifier's counted calls. With
work counted the bound is 126.21 bits; without work it is 116.01 bits. At
t = 2^64 the two terms are 2^-64.00 and 2^-124.91; at t = 2^96, 2^-32.00 and
2^-60.91; at t = 2^120, 2^-8.00 and 2^-12.91. The second term is the digests'
256 bits.

What the derivation assumes:

- g bits of work before a verifier's message multiply that round's error by
  2^-g. This is assumed for this hash-chain construction and is not proved
  here.
- The rounds with a closeness term rely on the code-proximity bounds the plan
  cites; some are taken as given and their proofs have not been checked here.
- Beyond a third of the code's distance a fold round has no factor for the
  number of interleaved rows; the plan's accounting states this as an argument
  on paper, not a proved theorem for these programs.
- That the programs compute the protocol is not proved. No figure in this note
  is a security level in the cryptographic-proof sense.
- Hiding is a separate bound, in the numbers of queries and proofs on one
  commitment, and is not part of these figures.

## Seal binding and what the verifier checks

The final seal is the value of the chain after the BASE round. It depends on
the statement, the trusted plan's first query, every root, answer, fold round,
switch identity and declaration, every nonce and every opening. The verifier
recomputes it from public inputs and the artifact and rejects a different
value. The tests flip the statement, roots, openings, identities, order,
positions, nonces and version and each is rejected.

The artifact binds the statement and the request declaration; the verifier
rebuilds the plan from the trusted policy and compiled plan, pins the switch
counts and work bits, and checks the final output relation through the trace
path where that path is supplied. The table entries' tie to the weight asset
root is not yet authenticated by an asset opening in this local chain; until
that closure is added, the table is only tied to the statement's identity and
the chain's commitments, and the caller-word items above remain open items.

## Open items

- The implementation's own round-by-round accounting is not mechanized; the
  figures above come from the plan's accounting.
- The table-to-asset opening authentication and the activation-to-output
  declaration are the remaining R1/R3 work.
- No production tested state is established by this note.


## Entry and performance scope

The positive and negative tests of this chain pass through the real public
serialization and the callable stage entry, not through the production CLI:

- commit: the stage worker builds the table, the public handoff and the
  private state; the handoff and the private files are encoded and decoded and
  the decoded values are used.
- prove: the worker reloads the handoff and private state, checks the plan and
  case identifiers, rebuilds the table commitment, runs the chain, and encodes
  the versioned artifact.
- verify: the worker reads only the public handoff and the artifact, decodes
  both, rebuilds the trusted setup from the policy, plan and statement, and
  runs the independent chain verifier. Any failure rejects.

The production `commit`, `prove` and `verify` commands and the pipeline
provider do not carry this chain yet. This boundary is a property of this
round, not a claim that the production entry was exercised.

Positives: the three processes accept with the same seal; the artifact
round-trips; a fresh decode verifies. Negatives include a repeated proof with
the spent budget, a missing or replaced private state, a stale journal lock, a
truncated artifact, trailing bytes, a changed handoff, a replaced plan
identity, a modified nonce, a different switch target, a changed statement,
root, opening, identity or order, and a version change. Each is rejected
before acceptance.

Performance numbers in this round are for the limited output-head chain at 16
positions on one machine, release build:

- artifact 5.59 MB; prove about 9.9 s; verify about 5.6 s; 1.62 tokens/s
  end-to-end and 1.67 tokens/s chain-only;
- the earlier 5.38 MB and 1.854 tokens/s figures used two draws per switch and
  are superseded, not comparable;
- full-model proof, real sampling at other sequence lengths, stage I/O,
  caching and the production baseline are not measured here. Memory is not an
  acceptance red line in this round.

## Shared-trace combination and confidentiality

A possible R1 combination verifies the existing shared trace independently and
uses its step outputs for X and Y. The trace proof carries the asset openings
that the step relations consume, and each opening contains the opened weight
rows as values. A verifier that reads the trace proof therefore sees those
weights in the clear. If the weights are private, the combination cannot be
presented as a hidden production proof; it can only be a controlled diagnostic
until the trace proof stops carrying the opened weights or another
weight-hiding path is built.

The full verification bytes of that combination are the trace proof
(4.16 GB in the measured production run) plus this chain's artifact (about
5.6 MB) plus the public bundle, policy and configuration. The new artifact's
size alone is not the proof size and must not be reported as such. The trace
proof is a required input, not an optional cache; a missing or trailing-byte
trace is rejected.
