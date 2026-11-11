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

## Trusted-policy table identity warranty

The limited table identity in the trusted policy binds the old model, anchor
and weight asset to the new committed encoded table through one canonical,
domain-separated, versioned digest. It covers the model id and version, the
case id, the anchor id, the asset id and asset table identity, the committed
table root, the full record digest, the table configuration, the code
parameters, the table shape and distances, the matrix mapping and the protocol
identity. The verifier rebuilds the committed side from the public reader, the
statement and the fixed protocol identity and checks it against the policy
value; the proof's own statement, roots and metadata are not a substitute. A
new table root or record therefore rejects until the operator updates the
policy. A prover-generated policy is not an independent anchor in production:
policy distribution, freshness and anti-rollback are deployment trust
assumptions.

This pin is an operator warranty, not a cryptographic permutation proof. The
trusted operator or publisher warrants that the two commitments -- the old
model/asset commitment and the new encoded table commitment -- correspond to
the same canonical weight matrix through the fixed mapping (table row and low
column to matrix row and column). The chain proves the encoded table's
relations; it does not prove that the encoding is a permutation of the
authenticated weights. That equality is the published-release trust assumption
recorded in the acceptance checklist.

Independently of this pin, hiding remains unclosed. The shared trace's
AssetOpening carries the opened weight rows as plaintext values to the
verifier, so a verifier that reads the trace proof sees private weights in the
clear. This combination must not be described as hidden acceptance; it is a
controlled diagnostic only.

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

The R1 combination verifies the existing shared trace independently and uses
its step outputs for X and Y. The trace proof carries the asset openings that
the step relations consume, and each opening contains the opened weight rows as
values. A verifier that reads the trace proof therefore sees those weights in
the clear.

The ruled scope for this round is therefore a **controlled diagnostic**, not a
hidden production proof. The combination may be accepted as evidence that the
request relation is real and independently checked; it must not be described
as hiding the weights, and it must not be used as the hidden production proof.
Replacing it with a weight-hiding relation is a later round. The acceptance
material must say that the combination sends plaintext weight rows and that
the hidden relation is outstanding.

The full verification bytes of that combination are the trace proof
(4.16 GB in the measured production run) plus this chain's artifact (about
5.6 MB) plus the public bundle, policy and configuration. The new artifact's
size alone is not the proof size and must not be reported as such. The trace
proof is a required input, not an optional cache; a missing or trailing-byte
trace is rejected.


## R1 controlled-diagnostic acceptance checklist

- The new chain verifies the LIN request itself over all 16 positions and the
  full column domain, with the first, shared and second checks and the checks
  they need. A tested key table or a recomputed shared trace alone is not the
  LIN request.
- X and Y are bound to step outputs of the verified trace; the verifier pins
  the model version, asset identity, step, port, shape and the upstream and
  downstream mapping from the trusted policy and plan.
- The declaration, its digests and its commitments use canonical encoding with
  domain separation and bind shape and identity; the verifier checks that both
  sides verify the same values or commitments before the challenges are drawn.
- Old and new roots are equal under the published-release trust assumption,
  not as a cryptographic equality claim.
- A versioned manifest lists every file the verifier needs, checks identity,
  missing files and trailing bytes. The full verification bytes are the trace
  proof (4.16 GB in the measured production run), this artifact and the public
  bundle/config; the new artifact's size is reported separately.
- Negatives: another valid trace, changed X or Y, another asset, another
  step or port, changed statement or root, wrong order, truncation and
  trailing bytes all reject.
- The material states plainly: the combination sends plaintext weight rows;
  it is a controlled diagnostic; the hidden relation is a later round.

## Independent trusted activation and hidden output closure

### What this round closes

The output-head request now has a formal path that consumes an independent
trusted activation `X` and keeps both `X` and the MatMul output `Y` out of the
verifier boundary:

- The operator policy carries an optional activation commitment identity: a
  canonical, domain-separated, versioned digest over the model, version, case,
  anchor and model ids plus the hiding row commitment's root, position count,
  hidden width, salt length and configuration. The identity binds the request
  to the request-level activation the operator warrants. The operator writes
  it after commit through the policy channel; the formal request verifier fails
  closed when the entry is absent. The verifier compares the commitment carried
  by the request session with this pin, so a prover-reported commitment cannot
  substitute for it.
- The prover commits `X` with fresh private salts, derives the reduced rows
  `W = X M` from the committed encoded table, and treats those rows as `Y`:
  they are committed as the SEND group members under the hiding product-form
  encoding. The verifier receives only the roots, the SEND samples and
  openings, the Eval targets and the BAT/FOLD/BASE checks. A separate
  plaintext `Y` input no longer exists on this path.
- The output relation is a trusted graph value: the vocabulary-mask step, its
  valid-vocabulary range, the separate argmax step of every position, the
  strict first-maximum tie rule and the expected final output. Its canonical
  digest is absorbed into the first statement, so a swapped mask, range,
  ordering rule or expected output changes the first query and the seals. The
  prover runs the complete full-row closure over all padded columns and must
  reproduce the trusted expected output; no argmax value is inferred from a
  sampled column.
- The exact integer output closure is checked by the same final verifier, not
  only bound by the digest. The trusted relation declares `D` (`|Y_j| < 2^D`),
  `b` (`0 <= S_j < 2^b`), the valid boundary `V`, the mask constant `M` and
  the public winner, and the proof commits four new messages: the slack row
  `S_j = Z_a - Z_j - [j < a]`, one packed bit message per position with the
  `D+1` bits of `u = Y + A`, the `D+1` bits of `v = A - Y` and the `b` bits of
  `S`, the product message `D(u) = eq(r,u) B(u)`, and one mask-only message
  carrying the one-time product-sumcheck and relation masks. The verifier rebuilds,
  from the transcript and the trusted plan:
  - one comparison row per position on the slack and reduced rows,
  - two reconstruction rows per position binding the `u` and `v` bit vectors
    to the committed reduced row, and one binding the slack bit vector to the
    committed slack row,
  - two public linear-functional rows and one equality row that tie the
    product message to the bit message,
  - one masked product-sumcheck per position proving
    `sum_u D(u)(1 - B(u)) = 0`; with `D = eq(r,.) B` this is the random-point
    booleanity sum `sum_u eq(r,u) B(u)(1 - B(u))`. The first map is the
    committed product message; the second map `1 - B` is affine in the
    committed bit message, so its end claim is one linear row over the bit
    point value and the closure mask.
  All rows join the existing own-group BAT/FOLD/CS/BASE test. The relation
  version is 2 and the no-wrap condition is `B + 2(2^D - 1) + |M| < p`. The
  comparison, reconstruction and booleanity checks are random-point checks
  over the extension field; the deterministic plan checks are `M <= -2^D`,
  `0 <= a_t < V`, the no-wrap inequality and the bound/shape checks.
- The complete hidden first, shared and second stages stay in one verifier:
  the group declaration, SEND samples/answers, drawn positions and openings,
  Eval rows, table record and column openings, FOLD rounds, the two proximity
  tests and the seals are all replayed. The limited-table policy pin is
  unchanged.

### Trust assumptions

- The correspondence of `X` to the request and the model is warranted by the
  operator beforehand. It is a request-level trust assumption, not a proved
  previous-layer relation. A prover self-report does not constitute a trusted
  input.
- The equality of the old and new static-weight commitments is a separate
  trusted publication assumption, not a cryptographic proof.
- `Y`, the weight rows and the other agreed private intermediates remain
  hidden on the formal path. The shared-trace path still carries opened weight
  rows and plaintext `X`/`Y`; it is a controlled diagnostic, not the hidden
  production proof.
- `D` and `b` are operator-declared model bounds. The proof proves the bound
  for the committed rows; a model whose logits or slacks exceed the declared
  bounds cannot produce an accepting proof. For the fixture, the measured
  largest magnitude is below `2^40` and the largest slack is below `2^42`.
- Passing this round does not imply the previous-layer hidden relation, a
  full-model proof or single-shot decode semantics.

### What the verifier does not check

- The hiding property of the activation row commitment is not proven here; the
  salted Merkle construction is used as the repository's existing primitive.
- The Dot identity `<X_t, Omega> = W_t(psi*)` is checked over the committed
  shown-map message, the committed `Omega` message and the committed reduced
  rows: the masked product-sumcheck transcript yields the `(U)` row and the
  two `(E_j)` end-value rows, one public difference row
  `sum_t chi^t g_t - sum_t chi^t c_t = R` joins the independent claim and
  tilde masks, and the own-group BAT/FOLD/BASE test checks every row. The
  residual `R` also adjusts the product-sumcheck target to
  `sum_t chi^t tilde_t - R`. The verifier pins the shown-map commitment and
  the activation commitment separately; that the two commitments carry the
  same `X` is an operator-policy warranty, not a cross-commitment proof. The
  mask joint-view analysis, the leakage-rank budget and the updated error
  terms are in `docs/hidden-dot-mask.md`; that argument is not a formal proof.
- The output closure is not a separate lookup/range proof over each cell. The
  range and comparison checks are random-point checks with the stated
  Schwartz-Zippel terms; they are not a formal soundness proof. The booleanity
  argument draws `r` after the bit commitment and before the product message,
  draws the sumcheck point after the product commitment, and draws the
  linear-functional point after it as well; the one-time closure-mask entries
  cover the sent product-sumcheck and relation scalars. The joint-view hiding
  and opening-union budgets are not mechanized. The base-field membership of the committed
  reduced row remains the activation/table warranty described above.
- The previous-layer relation from the model to `X` is not proven.
- Any plaintext `X` source is diagnostic only.
- The production `commit`/`prove`/`verify` entry wiring is pending.
- Formal soundness is not done. No equivalence with any reference system and
  no equal security or performance claim follows from this round.

### Negatives exercised

The unit and fixture tests reject a changed activation commitment against the
policy pin, an arbitrary activation not matching the pin, a changed output
commitment, a changed Eval target, a changed MatMul step position, a missing
output-relation declaration, a forged mask constant, a forged tie or expected
argmax, an unsigned modular encoding of a negative value, a masked-arithmetic
overflow, spliced request material and activation-opening bytes appended to the
request encoding. The output-closure tests additionally reject a changed
booleanity round, changed fresh-point functionals, changed slack/bit/product
roots, a changed slack SEND answer, a changed closure-mask root, a missing
closure block, a different trusted winner, a widened integer bound and
truncated or trailing proof encodings; the prover-side value checks reject a
cell outside `|Y| < 2^D` and a slack at or above `2^b`.
