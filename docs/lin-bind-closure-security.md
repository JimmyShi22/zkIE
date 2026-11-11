# Closed output-head relation: public view, opening union and composition accounting

## Status and labels

This note covers the closed hidden output-head relation implemented in the
local chain. It is an accounting and argument document, not a formal
zero-knowledge proof, not a mechanized composition theorem and not a claim that
the trusted X source is proved. Labels:

- `[FACT]` checked in the code or arithmetic at the stated snapshot.
- `[ASSUMPTION]` a condition the argument needs.
- `[UNPROVEN]` stated but not proved here.
- `[GAP]` a missing check or mechanism; the run must fail closed.

Snapshot: the closed relation at code basis `3b3e8ecc`, with the bounded-read
rollback `8d32d3de`; fixture `gpt2-seq16`, 16 positions, proof 9,939,144 B.

## 1. Public view

The verifier receives the trusted policy, the public statement and the proof
artifact. The relevant public items are:

| View item | Contents |
|---|---|
| policy pin | limited table identity: model, case, anchor, asset and table identity, committed table root, record digest, configuration, code parameters, shape, distances, matrix mapping, protocol identity |
| statement | activation commitment root, output plan (winner, valid range, tie rule), table identity, public claim items |
| committed group roots | W rows, index-mask message, shown map, Omega, G_u, closure slack, closure bits, closure map, closure mask, fresh/star messages |
| SEND material | shared seeds and their public answers, index-mask answers, masked Eval targets, drawn positions and their openings |
| Dot material | tilde values, the public residual `R`, the masked product-sumcheck transcript and its end values |
| FOLD/BASE material | fold start, round messages and challenges, lambda, base positions, base openings and final rows |
| output | first-max winner, valid range, tie result, seal |

`[FACT]` The verifier holds no table message rows, no activation rows, no
reduced weight rows and no private tails, salts or pads.

## 2. Random material and one-time masks

`[FACT]` The separate-mask schema allocates 73 entries for 16 positions:

| Class | Entries | Use |
|---|---:|---|
| PSC hiding coefficients | 23 | hides the masked product-sumcheck polynomial |
| shown-value masks | 2 | the two shown-map end values |
| claim masks `c_t` | 16 | one per position |
| tilde masks `g_t` | 16 | one per position |
| Eval masks `e_t` | 16 | one per position |

The classes occupy disjoint ranges. The claim and tilde entries are independent
draws from the same stage-1 stream; the public difference row ties their
chi-weighted sums to `R` without forcing equality. The Eval mask is fixed
before the Eval point is drawn, so the masked target `W_t(psi_t) + e_t` is
uniform. The PSC hiding coefficients are one-time.

`[FACT]` The table, W rows, shown map, Omega and G_u messages carry distinct
tails and salts drawn from the stage stream; the SEND answers use per-message
pads. The generation stream of one oracle is identified by a material
commitment, not by observed equal positions or values; a repeated material
commitment is rejected. `[ASSUMPTION]` The draws are independent uniform field
elements.

## 3. Joint-view hiding argument

The argument is made value by value. It is not a simulation proof.

1. `[FACT]` A table codeword symbol is one linear equation in the code tail's
   `r` random symbols plus the message symbol. The ledger records the union of
   distinct opened positions per tree; the hiding dimension is `r - union`.
   The union arithmetic is in section 4.
2. `[FACT]` A W-row opening is checked against the registered W root. The row's
   random tail and salt are not sent; the opened codeword symbols are the
   product-form encoding of the row.
3. `[FACT]` A SEND answer is `W_t(seed) + pad`. The pad is private and one-time
   per message; the answer is public.
4. `[FACT]` An Eval target is `W_t(psi_t) + e_t`. The mask `e_t` is fixed before
   `psi_t`; the target is uniform for a fixed message.
5. `[FACT]` A Dot value is masked by `c_t` or `g_t`. After the batching
   challenge `chi`, the prover sends the single public residual
   `R = sum_t chi^t (g_t - c_t)`. This is one linear equation in the 2*16
   independent masks. It reduces the joint entropy by one field element but
   leaves each individual mask marginal uniform; the marginals of the masked
   Dot values are therefore unchanged. `[ASSUMPTION]` `chi` and its powers are
   nonzero and the mask draws are independent uniform field elements. If
   `chi = 0`, the residual does not combine the positions as assumed and the
   run must fail closed.
6. `[FACT]` The closure bits, slack and product map are functions of the
   committed reduced output row and the public winner. Their openings reveal
   those functions at the drawn positions, not the weight rows.
7. `[FACT]` The transcript chain is sequential and domain-separated by stage
   labels and schema versions. The same challenge is not reused across stages.
   The ledger records zero table reuse and one-use material streams.

`[ASSUMPTION]` The reduced output row `Y` and the public winner are allowed to
be revealed to the verifier at the drawn positions. If output confidentiality
is required, the closure openings are a leak and the run must fail closed.

`[UNPROVEN]` A full joint simulation of the public view is not given. The
argument above covers each public item and the one public cross-position
combination `R`; it does not prove that no combination of roots, answers,
openings, transcript words and seals distinguishes the view. The rank and
union accounting in section 4 is a necessary condition, not a simulation.

`[GAP]` If any mask class is reused, if a pad is reused across messages, if a
material commitment repeats, or if the table union reaches its tail rank, the
argument does not apply. The ledger is the fail-closed gate for those cases.

## 4. Opening union and tail budget

`[FACT]` The formal closed run records:

| Category | Logical | Unique |
|---|---:|---:|
| activation rows | 0 | 0 (16 rows allocated, never opened) |
| table columns | 603 | 603 |
| W SEND and BASE | 64 | 64 |
| index mask SEND and BASE | 4 | 4 |
| shown map, Omega, G_u, fresh masks, stars | 20 | 20 |
| closure slack, bits, map, mask | 196 | 196 |
| total | 887 | 887 |

The 143 opening events and 887 logical draws are also the exact union count;
the closed run has no repeated table column. The earlier pre-closure artifact
with 603 draws and 601 distinct columns is recorded separately and is not
replaced by this run. Table reuse is zero.

`[FACT]` The ledger enforces `logical draws <= tail rank` at reservation and
reports, per code-bearing oracle, `rank - union`. The measured summary is:

- code-bearing oracles: 72;
- minimum tail remaining: 252;
- total tail rank: 558,080;
- total unique positions on code-bearing oracles: 879;
- exhausted tail oracles: 0.

The 887 total unique positions in the table include 8 positions on oracles
without a code tail (the star messages); the budget summary covers the 72
code-bearing oracles and accounts 879 of them. The minimum remaining 252 is
the conservative per-oracle bound; no code oracle is exhausted.

`[ASSUMPTION]` The relevant hiding budget is per tree: a union of `u` positions
against rank `r` leaves `r - u` random dimensions. Summing ranks across trees is
not meaningful because the trees are independent.

`[UNPROVEN]` The union count is accounting, not a rank proof of the code's
linear system. It assumes the product-form code's stated tail rank and the
ledger's position union. If the minimum remaining is zero, the run has no
hiding dimension in that tree and must fail closed.

## 5. Challenge order and verifier rejection chain

`[FACT]` The order of the closed hidden session is:

| Step | Absorbed before the challenge | Challenge | Verifier layer |
|---|---|---|---|
| 1 | activation commitment, W roots, mask and shown-map roots, group declaration | shared seeds | root and declaration checks |
| 2 | complete ordered answers | W/mask positions | sampled-answer and opening checks |
| 3 | closure group roots, closure mask root, closure seeds and answers | `r_bool` | closure root, answer and shape checks |
| 4 | map group and map answers | map positions | map root, answer and opening checks |
| 5 | index-mask answers | index-mask positions | mask opening and slot checks |
| 6 | Eval targets | Eval point `psi` | masked Eval row checks |
| 7 | tilde values and public residual `R` | PSC challenges | residual absorbed before PSC; PSC row and end-value checks |
| 8 | fold start | epsilon, then round challenges | fold shape, sum and round checks |
| 9 | base positions | lambda | base opening and final-row checks |
| 10 | final seal | none | seal recomputation and output relation |

`[FACT]` The negative matrix maps to these layers:

| Negative | Rejection layer |
|---|---|
| changed trusted activation commitment | policy pin |
| changed output winner or valid range | output plan validation |
| tampered closure PSC round | PSC verifier |
| changed closure point values | closure second-stage verifier |
| changed closure slack/bits/map/mask root | group root check |
| changed slack sample answer | sampled-answer check |
| missing closure section | plan/encoding check |
| changed fold round, position or nonce | fold/base verifier |
| changed W root or row | W root and opening check |
| tampered hidden request bytes | manifest/decode check |
| independent identical `finish_run` input | frozen-input ownership check |
| malformed group table | shape check |
| wrong retained tree plan or root | tree-association check |

`[GAP]` The matrix covers the implemented negative entries. It is not a proof
that every message field has a dedicated negative; fields reached only through
a shared root or a later relation are covered indirectly. A production
verifier must keep the full chain, not a subset.

## 6. Composition error accounting

`[FACT]` The existing round accounting gives a 128-bit target per round. With
work counted, the largest round errors are 2^-128.00 (FOLD rounds and BASE
lambda); the four position rounds are 2^-128.16, 2^-128.48, 2^-128.01 and
2^-128.50. Without work the chain bottoms out at 2^-116.01, so the 12 work bits
are not optional. The compiled per-request bound is

```
(t + q_V + 1) * 2^-128.00 + (17/2) * (t + q_V)^2 / 2^256
```

with 126.21 bits with work and 116.01 bits without work. At t = 2^64 the two
terms are 2^-64.00 and 2^-124.91; at t = 2^96, 2^-32.00 and 2^-60.91; at
t = 2^120, 2^-8.00 and 2^-12.91.

The closed relation adds:

- exact range decomposition over 40-bit `u`/`v` and 42-bit slack, with
  booleanity and range checks;
- the masked product-sumcheck for the dot identity;
- the closure PSC rows and their end values;
- the LKP membership and the proximity tests already in the chain.

`[UNPROVEN]` No independent composition theorem is proved here. The round
figures are taken from the plan's accounting; some code-proximity bounds are
assumed; the programs' computation of the protocol is not proved. A
conservative union bound over the components is not mechanized. The 126.21-bit
figure is therefore a plan-accounting figure, not an end-to-end security level.

`[GAP]` If the closure's range-decomposition terms, PSC terms or LKP terms are
not all instantiated at the stated parameters, the composition figure does not
apply. The run must fail closed rather than claim the 128-bit target.

## 7. Fail-closed summary

The following are open and must be stated as open:

- `[UNPROVEN]` full joint-view hiding / zero-knowledge simulation;
- `[UNPROVEN]` opening-union rank proof beyond the ledger's count and the
  code's stated tail rank;
- `[UNPROVEN]` a mechanized composition theorem and end-to-end 128-bit claim;
- `[GAP]` output confidentiality, if the output row is not allowed to be
  revealed at the drawn positions;
- `[GAP]` the trusted X source and the policy distribution/freshness warranty;
- `[GAP]` production commit/prove/verify CLI wiring; the measured path is a
  stage-harness controlled diagnostic.

No statement in this note claims that the trusted X source is proved, that the
weights are hidden under a formal ZK definition, or that the run is a
production acceptance.
