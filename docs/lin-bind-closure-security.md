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

### 2.1 Output-view boundary

This relation is within the approved output-binding boundary: the public
first-max winner, valid range and tie result are protocol outputs, and the full
reduced output row is not opened. The public items have the following form:

| Public item | Form | What the verifier can derive |
|---|---|---|
| closure slack/bits/map openings | encoded codeword symbols, linear in the code tail plus a function of the output row | linear equations in the tail; at most the union positions, leaving `r - union` tail dimensions under the stated rank |
| SEND answers | raw field values, message sample plus a one-time pad | only the padded value; the pad is one-use |
| Eval targets | raw field values, `W_t(psi_t) + e_t` | the target; `e_t` is fixed before `psi_t`, so the masked value is uniform for a fixed message |
| output winner/valid/tie | raw protocol output | the approved output semantics; no weight row |
| residual `R` | raw field value, one chi-weighted difference of the claim and tilde masks | one linear combination; no individual mask |
| table/W openings | encoded codeword symbols | linear equations in the tail; `r - union` dimensions under the rank assumption |
| roots and seal | digests | binding only; no raw value |

No item above is a plaintext weight row. The table and W messages are opened
as encoded symbols; the closure messages are functions of the reduced output
row. `[GAP]` If the deployment requires the output row itself to stay hidden
beyond the approved winner/valid/tie boundary, the closure openings are a leak
and the run must fail closed.

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
   nonzero and the mask draws are independent uniform field elements.
   `[FACT]` The implementation draws `chi` with the ordinary field challenge
   and has no explicit nonzero check. If `chi = 0`, the residual combines only
   the `t = 0` term; the probability is `1/p^3 = 2^-192` for the cubic field,
   and that term is included in the composition bound in section 6. No
   executable rejection of `chi = 0` is claimed.
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

### 3.5 Joint linear system over the full public view

`[ARGUMENT]` Write the public view as linear forms in the random material. The
mask block has 73 entries `(c, g, e, end, U)`. The code-tail block has one
random tail per code-bearing oracle, with total rank 558,080. The pads form a
third independent block. The coefficient structure is:

| Public item | Linear form | Random variables | Rank contribution |
|---|---|---|---:|
| `tilde_t` | `W_t + g_t` | `g_t` | 16 |
| `R` | `sum chi^t g_t - sum chi^t c_t` | `g_t, c_t` | 1 new direction |
| Eval target `tau_t` | `W_t(psi_t) + e_t` | `e_t` | 16 |
| shown end values | two forms on `end` | `end` | 2 |
| PSC transcript | affine bijection on `U`, plus one `sigma` direction | `U`, `c` | 23 |
| SEND answers | `message sample + pad` | `pad` | one per pad |
| table/W/closure openings | codeword symbol = tail combination + message symbol | per-tree tail | at most the union per tree |
| roots, seals | hash commitments | hash randomness | 0 algebraic |
| output winner/valid/tie | protocol output | none | 0 |

The mask block therefore has rank 58 on 73 entries and kernel 15, recovering
the existing `(tilde, R)` rank 17 / kernel 15 result and the 58/73 total. The
code-tail block contributes at most the 879 distinct code-oracle positions;
its kernel is at least `558,080 - 879 = 557,201`. The pads add one random
dimension and one revealed form each, so they cancel in the kernel.

`[ARGUMENT]` Under the independence assumptions in section 2, the joint
leakage dimension is

```text
rank <= 58 + 879 + P = 937 + P
dimension >= 73 + 558,080 + P = 558,153 + P
kernel >= 15 + 557,201 = 557,216
```

where `P` is the number of padded SEND answers. The kernel lower bound does
not depend on `P`. This is an argument about the linear forms, not a
simulation: roots and seals are hash commitments, and the code-tail rank is
the stated product-form rank.

`[UNPROVEN]` The full joint simulation and the all-challenge rank statement
are not proved. The count above is the algebraic accounting; the hash
commitments, the code's actual rank and the independence of the streams are
the assumptions listed in section 6.

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

`[ASSUMPTION]` The statement "hiding dimension = `r - union`" is conditional
on the code's actual tail rank being `r`. Known plan values are: table code
`r = 8192`; the folded/shared main code `r = 962`; the mask codes `r = 50`,
`57` or `47`; the closure codes use the setup's code parameters. If an actual
rank is lower than the plan value, use the lower value; the bound only
decreases. The ledger's measured minimum remaining 252 is the empirical check
over the actual plan.

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
| 6 | W/mask/closure positions and openings | Eval point `psi` | opening checks; `psi` is drawn after all openings |
| 7 | Eval targets computed from `psi` and absorbed after it | none at this step | masked Eval row checks; the mask `e_t` was fixed before `psi` |
| 8 | tilde values and public residual `R` | PSC challenges | residual absorbed before PSC; PSC row and end-value checks |
| 9 | fold start | epsilon, then round challenges | fold shape, sum and round checks |
| 10 | base positions | lambda | base opening and final-row checks |
| 11 | final seal | none | seal recomputation and output relation |

`[FACT]` The Eval targets are not absorbed before `psi`; they are computed from
`psi` and absorbed after it. The hiding property uses the fact that the mask
`e_t` is fixed before `psi`, not that the target is fixed before it. An earlier
version of this table stated the reverse order; the code order above is the
correct one.

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

`[ARGUMENT]` A parameterized union bound over the components is

```text
eps_total <= eps_hash + eps_chi + eps_IOPP + eps_LKP
           + eps_range + eps_PSC + eps_eq

eps_hash = (t + q_V + 1) * 2^-128.00
         + (17/2) * (t + q_V)^2 / 2^256
eps_chi  = 2^-192
```

The component terms use the plan's accounting: each audited proximity/LKP
round is at most `2^-128.00`, so eight such terms sum to about `2^-125`; the
range booleanity term is bounded by the decomposition degree over the cubic
field, `16 * 166 / p^3 <= 2^-180.6`; the PSC term by `10 * 3 / p^3 <= 2^-187.1`;
the single-equality term by `2 / p^3 <= 2^-191`. These are arithmetic bounds
under the stated degrees, not a mechanized proof.

At `t = 2^64`, `2^96` and `2^120` the hash term dominates: about `2^-64`,
`2^-32` and `2^-8`; the component sum is about `2^-125` and the `chi = 0` term
is `2^-192`. The figures `126.21` and `116.01` bits above remain plan-ledger
numbers, not a new end-to-end security level for the closed chain.

`[UNPROVEN]` No independent composition theorem is proved here. The round
figures are taken from the plan's accounting; some code-proximity bounds are
assumed; the programs' computation of the protocol is not proved. A
conservative union bound over the components is not mechanized. The 126.21-bit
figure is therefore a plan-accounting figure, not an end-to-end security level.

`[GAP]` If the closure's range-decomposition terms, PSC terms or LKP terms are
not all instantiated at the stated parameters, the composition figure does not
apply. The run must fail closed rather than claim the 128-bit target.

## 7. Runtime fail-closed controls

`[FACT]` The following controls are executable and have direct tests:

| Control | Executable evidence |
|---|---|
| material/seed/commitment reuse | ledger state validation and tests `the_same_material_stream_across_requests_is_rejected_even_with_another_root`, `the_same_seed_commitment_across_requests_is_rejected`, `the_same_commitment_root_under_another_stream_is_rejected` |
| table reuse | `TableReusePolicy::Forbidden` and `table_reuse_is_forbidden_or_counted_by_policy` |
| tail budget exhaustion | `LeafBudgetExhausted` and `a_declared_budget_above_the_tail_rank_is_rejected_before_any_event`, `tail_union_accumulates_across_stages_and_an_overrun_fails_closed` |
| state rollback or edited log | `StateRollbackSuspected`, `a_damaged_state_is_rejected`, `a_truncated_or_edited_event_log_is_rejected` |
| proof/verify lifecycle | `success_requires_the_complete_verification`, `a_failed_verification_marks_the_request_failed_and_blocks_retry` |
| frozen BAT input | `finish_run_rejects_independent_content_identical_input` |
| group-table shape | `group_table_rejects_malformed_members_and_cross_block_placement` |
| retained tree plan/root | `retained_tree_checks_plan_and_sealed_root` |
| mask slot schema | `mask_slot_schema_is_separate_and_one_time` |
| final proof negatives | hidden proof negative matrix: changed roots, PSC rows, point values, sample answers, missing closure, fold rounds, positions, nonces, wrong plan |

`[UNPROVEN]` or `[GAP]` items are not backed by an executable rejection:

- `chi = 0` has no explicit check; it is covered only by the `2^-192`
  probability term in the composition argument.
- Full joint-view hiding, the opening-union rank proof and the composition
  theorem are document-level arguments.
- Output confidentiality, the trusted X source, policy distribution/freshness
  and production CLI wiring are assumptions or gaps.

## 8. Fail-closed summary

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
