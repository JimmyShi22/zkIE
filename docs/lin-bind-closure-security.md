# Closed output-head relation: public view, opening union and composition accounting

## Status and labels

This note covers the closed hidden output-head relation implemented in the
local chain. It is an accounting and argument document, not a formal
zero-knowledge proof, not a mechanized composition theorem and not a claim that
the trusted X source is proved. Labels:

- `[FACT]` checked in the code or arithmetic at the stated snapshot.
- `[ASSUMPTION]` a condition the argument needs.
- `[UNPROVEN]` stated but not proved here.
- `[GAP]` a missing check or mechanism; an executable condition fails the
  run closed, a condition that cannot be checked at runtime withdraws the
  security conclusion.

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

`[FACT]` There are two separate mask allocations, in two different messages.
They are not summed.

The index-mask message (Dot relation) allocates 73 entries for 16 positions:

| Class | Entries | Use |
|---|---:|---|
| PSC hiding coefficients | 23 | hides the Dot product-sumcheck polynomial |
| shown-value masks | 2 | the two Dot shown-map end values |
| claim masks `c_t` | 16 | one per position |
| tilde masks `g_t` | 16 | one per position |
| Eval masks `e_t` | 16 | one per position |

The separate closure mask-only message allocates, per position, 49 PSC hiding
coefficients and 5 shown-value masks: two PSC end masks, one committed-bit
point mask and two relation-function masks (`CLOSURE_MASK_SHOWN = 5`). For 16
positions that is 864 entries. These entries are not part of the 73-entry
index-mask schema; the two plans are different oracles.

The classes in each message occupy disjoint ranges. The claim and tilde
entries are independent draws from the same stage-1 stream; the public
difference row ties their chi-weighted sums to `R` without forcing equality.
The Eval mask is fixed before the Eval point is drawn, so the masked target
`W_t(psi_t) + e_t` is uniform. The PSC hiding coefficients are one-time. The
closure mask's five shown entries are one-time per position, in this order:
entries 0 and 1 cover the shown-map ends; entries 2, 3 and 4 cover
`(B(pt), D(q), weighted B(q))`.

`[FACT]` The table, W rows, shown map, Omega and G_u messages carry distinct
tails and salts drawn from the stage stream; the SEND answers use per-message
pads. The generation stream of one oracle is identified by a material
commitment, not by observed equal positions or values; a repeated material
commitment is rejected. `[ASSUMPTION]` The draws are independent uniform field
elements.

### 2.1 Output view and full-row boundary

The approved output-binding boundary makes the public first-max winner, valid
range and tie result protocol outputs. The full reduced output row `Y` is not
public; hiding it beyond those three items is a condition of this round, not a
`[GAP]`.

| Public item | Actual form | What the verifier receives |
|---|---|---|
| closure slack/bits/map openings | codeword symbol `sum_{i<r} t_i z^i + sum_{i<ell} f_i(Y) z^{r+i}` | `q` linear equations; no plaintext `Y` |
| closure SEND answers | sample of `f(Y)` plus a one-time pad | a padded field value |
| closure PSC transcript | affine image of the 23 hiding coefficients plus the target shift | transcript; target covered by the hiding map |
| closure fresh-point values | three masked values per position | each covered by its one-time shown mask |
| W SEND answers / Eval targets | W sample plus pad / `W(psi)+e` | masked field values |
| output winner/valid/tie | protocol output | public by the approved boundary |
| residual `R` | chi-weighted difference of `c` and `g` | one mask combination |
| roots and seal | digest | binding only |

The codeword symbol is a function of `Y` plus a random tail; "no plaintext
`Y`" means the message part is masked by the tail, not that the symbol is
independent of `Y`. Section 3.6 states the same fact in linear-form terms.

`[ARGUMENT]` Every public item that depends on `Y` is one of: a codeword
symbol whose message part is masked by the code tail on the opened set, a
padded sample answer, or a PSC transcript whose target shift lies in the image
of the hiding map. Under the assumptions in section 3.10, the verifier cannot
recover the plaintext `Y` row beyond winner/valid/tie. This is a conditional
conclusion of the argument; if any assumption fails, the security conclusion
is withdrawn. Executable conditions (tail overrun, reuse) are ledger
rejections; conditions that are not runtime-checked are stated as conditional
assumptions, not as program fail-closed checks.

## 3. Joint-view hiding: linear-cover criterion

### 3.1 The criterion

For one public view write

```text
v = A r + B s,
```

where `r` is the random material, `s` the secret material, and `A`, `B` are the
coefficient matrices. Hiding holds if

```text
Im(B) subset Im(A),   equivalently rank([A | B]) = rank(A).
```

For every secret `s` there is then an `r'` with `A r' = A r + B s`, so the view
is independent of `s`. This is the criterion used below. A large kernel by
itself is not used as evidence: a large kernel can coexist with a directly
leaked secret.

### 3.2 Openings: Vandermonde submatrix

A codeword is the evaluation of a polynomial with tail coefficients at degrees
`0..r-1` and message coefficients at degrees `r..r+ell-1`. For an opening set
`S` of `q` distinct positions,

```text
A_tail = [ z_j^i ]_{j in S, i < r}       (q x r)
B_msg  = [ z_j^{r+i} ]_{j in S, i < ell} (q x ell)
```

`A_tail` is a Vandermonde matrix on distinct points. If `q <= r`, it has full
row rank `q`, so `Im(A_tail) = F^q` and `Im(B_msg) subset F^q = Im(A_tail)`.
The opened symbols are uniform over `F^q` and independent of the message. This
is the submatrix property used for every opening view. The union `S` of a
tree's SEND and BASE openings is used, so cross-stage reuse is included.

### 3.3 SEND answers and Eval targets

- SEND answer: `answer = sample(message) + pad`. `A = [1]` on the pad,
  `B = [sample]` on the secret; rank 1, covered.
- Eval target: `target = W(psi) + e`. `A = [1]` on `e`, `B = [W(psi)]`;
  rank 1, covered. `e` is fixed before `psi`.

### 3.4 Dot values, tilde and `R`

`tilde_t = W_t + g_t`: `A = [1]` on `g_t`, `B = [1]` on `W_t`; covered.
`R = sum chi^t g_t - sum chi^t c_t` has `B = 0` on the secret; it is a public
combination of masks only. The pair `(tilde, R)` has rank 17 on the 32
`(g, c)` entries and kernel 15, as in the existing mask analysis; `R` does not
add a secret direction. `chi = 0` is an exception event handled in section 6.

### 3.5 PSC transcript

The Dot PSC (transcript tags 16/17/18) sends 23 scalars. They are an affine
bijection of the 23 hiding coefficients of the 73-entry index-mask schema,
plus the `sigma` shift in one claim-mask direction. The hiding map is 23 x 23
and is checked full-rank at one deterministic challenge point; the
all-challenge statement is an assumption.

The closure PSC (tags 20/21/22) is a second, separate instance. The closure
mask plan allocates, per position, 49 hiding coefficients and five shown
masks (`CLOSURE_MASK_SHOWN = 5`): two PSC end masks, one committed-bit point
mask and two relation-function masks. Each shown mask is one-time and gives
one random dimension per masked value. The closure PSC transcript and the
three fresh-point values `(B(pt), D(q), weighted B(q))` are covered by that
per-position instance; the 23 Dot scalars and the 49 closure coefficients are
not the same mask plan and are not summed. The closure hiding map's
all-challenge full-rank statement is an assumption.

### 3.6 Closure messages and the full output row

The closure slack, packed-bit and product messages encode functions `f(Y)`.
Their openings are covered by the Vandermonde property in 3.2, their SEND
answers by the pads in 3.3, and their PSC transcript by the closure instance
in 3.5 (49 per-position hiding coefficients and five shown masks per
position). The Dot PSC's 23-entry index-mask schema is a different oracle and
is not used for the closure messages.
The same argument applies to every closure message and every position, so the
full `Y` row is hidden beyond the approved winner/valid/tie items. This is a
conditional conclusion, not a `[GAP]`.

### 3.7 Hash roots and commitment hiding

Roots are Merkle commitments over salted leaves. The algebraic argument is made
conditional on the roots: under the random-oracle or collision-resistant hash
model, a root hides its input when the input has sufficient min-entropy; the
salt, tail and pad supplies that entropy. The combination is a hybrid: first
replace the roots by random oracle outputs, run the algebraic cover argument on
the openings, answers and transcripts, then use the commitment hiding for the
roots. The statement "algebraic rank 0 means the root leaks nothing" is not
used. If the hash model or the min-entropy assumption fails, the conclusion is
withdrawn.

### 3.8 Cross-stage and multi-request view

- The opening union per tree is taken over every stage; the Vandermonde
  property is applied to that union.
- Mask slots are disjoint and one-time; claim, tilde and Eval masks are not
  reused across positions.
- Material streams are one-use; the ledger rejects a repeated material
  commitment, a repeated seed commitment and a reused table reservation.
- For multiple requests, the same per-request criterion applies; the table
  union is the cumulative union checked by the ledger. A global multi-request
  bound needs a bound on the number of requests; without one, no global figure
  is claimed.

### 3.9 FOLD, BASE and CS verification values

The verification transcript also includes the FOLD start sum/rounds and the
BASE symbols.

- FOLD: the fresh hiding polynomial supplies `2k+1` random coefficients. The
  start sum and every published round value are affine in those coefficients;
  the secret group-row contribution is a shift of that affine map. The
  construction uses the full coefficient set for the hiding polynomial (the
  reference check `fold_mask_as_drawn` verifies every coefficient and tail).
  The linear-cover criterion therefore holds under the assumption that the
  fresh-mask coefficient map has full rank at the drawn challenge; a
  degenerate challenge is an exception.
- BASE: the star main row, star mask row and their random tails are fresh
  draws. The public BASE rows and openings are affine in those draws, so the
  secret row contribution is covered by the star randomness one entry at a
  time. The BASE target includes the same star rows, so it does not add a
  separate secret direction.
- CS: the hidden output-head path runs no code-switch stage; the plan rejects
  a switch configuration. If a code-switch stage is added later, its fresh
  mask and star rows must be included in the same criterion; this document
  does not cover a CS path.
- The BAT `x` challenge is drawn after the frozen rows; the combined group
  row is internal and enters FOLD, where the fresh mask covers it. There is no
  separate public BAT row to analyze.
- The BASE `lambda` challenge is public; the BASE rows and openings are
  covered by the fresh star rows and tails as above.
- The FOLD/BASE public values are functions of committed codewords; they do
  not add a plaintext weight or output-row direction beyond the covered
  openings, answers and tails.

### 3.10 Assumptions and exception events

- The random draws are independent uniform field elements.
- The opening positions are distinct; the union per tree is used.
- `|S| <= r` for every actual opening set; section 4 checks the actual sets.
- The code tail rank is the stated `r`.
- The PSC hiding maps have full rank for every challenge, for both the
  23-entry Dot instance and the 49-coefficient closure instance; only one
  deterministic challenge point is tested for each.
- `chi != 0`; if `chi = 0`, the residual combines only the `t = 0` term and
  the corresponding term is in section 6.
- The SEND sample's beta is drawn nonzero by the implementation; the pad
  value itself may be zero and still hides (the verifier does not observe the
  zero event).
- The hash model is random-oracle or collision-resistant and the salted
  inputs have the stated min-entropy.
- The FOLD fresh-mask coefficient map is full-rank at every drawn challenge;
  the BASE star rows and tails are fresh and independent.
- The hidden path has no code-switch stage; a CS path would need its own
  fresh-mask/star coverage.
- No pad, mask class, material stream or table reservation is reused outside
  the ledger policy.
- The output-binding boundary is the approved winner/valid/tie boundary.

If an assumption fails, the linear-cover conclusion does not apply. Once
established, the linear-cover criterion is met for every view listed in this
section: the joint hiding argument is conditional and no secret leak is
identified. Assumptions that cannot be checked at runtime (PSC/FOLD rank,
hash model, output boundary) withdraw the security conclusion rather than
claiming that the program fail-closed. `[UNPROVEN]` A
full simulation and the adaptive composition of the hash and algebraic
arguments are not mechanized here; the argument above is the conditional
linear-cover argument, not a theorem.

### 3.10b Oracle encoding and hash order

| Oracle family | Code layout | Random values | Openings | Public values |
|---|---|---|---|---|
| table | base field, `log_n=17`, `ell=65536`, `r=8192` | row tail and leaf salt | 603 columns | root, record, column openings |
| W rows | cubic field, `log_n=17`, `ell=65536`, `r=8192` | per-tree tail and salts | 2 SEND + 2 BASE | W roots, openings, sample answers |
| closure slack/bits/map | cubic field, `log_n=17`, `ell=65536`, `r=8192` | per-tree tail and salts | 2 SEND + 2 BASE each | roots, answers, openings, PSC |
| index mask | cubic field, `log_n=17`, `ell=512`, `r=256` | tail, salt, 73-entry mask | 2 SEND + 2 BASE | root, answers, openings |
| closure mask | cubic field, `log_n=17`, `ell=512`, `r=256` | tail, salt, 49+5 per position | 2 SEND + 2 BASE | root, answers, openings |
| FOLD fresh | `2k+1` cubic-field coefficients | fresh mask polynomial | round messages | start sum, round values |
| BASE star | cubic-field rows and tails | fresh per value | base openings | base rows and openings |

The hash hybrid order is fixed: first replace every Merkle root by a random
oracle output (the commitment-hiding step), then run the algebraic cover
argument on the opened values, answers, transcripts and star rows, then use
the collision-resistant/random-oracle model for the roots. Roots are never
treated as independent random field values. The salt/tail/pad entropy is the
input min-entropy the commitment step needs. The transcript fixes positions
after the roots and answers are absorbed; conditional on a root, the tail is
uniform, and the only constraints are distinct positions and the root binding.

### 3.10c Shared random-variable attribution

| Random variable | Public items where it appears | Shared? |
|---|---|---|
| table/W/closure/mask tails | the corresponding tree openings | no across trees |
| `g_t` | `tilde_t`, `R` | yes: both rows, explicit |
| `c_t` | `R`, Dot PSC `sigma`; also the claim relation row (checked by proximity, not a sent field value) | yes: `R` and `sigma` explicit |
| `e_t` | the position's Eval target | no |
| Dot `U` (23) | Dot PSC transcript | no |
| closure `U` (49 per position) | closure PSC transcript | no |
| closure shown (5 per position) | two end values, three fresh-point values | no |
| SEND pads | the corresponding answer | no |
| FOLD fresh (`2k+1`) | FOLD start sum and rounds | no |
| BASE star rows/tails | the corresponding BASE values | no |

The only cross-public-item sharing is `g_t` between `tilde` and `R`, and `c_t`
between `R` and the Dot PSC `sigma`; `c_t` also appears in the claim relation,
which is checked by proximity rather than sent as a field value. The `R` and
`sigma` columns are present in the joint mask matrix; no other variable is
silently reused. This table is
the explicit check for the `v1 = s + r, v2 = r` counterexample.

### 3.11 Joint matrix checks

The test module constructs `A` and `B` from the actual protocol evaluation
paths at fixed oracle/challenge structure and computes the ranks numerically
over the cubic field. It does not assume that per-view full rank implies joint
rank; shared variables are explicit columns.

| Block | Random columns | Secret columns | Rows | Result |
|---|---|---:|---:|---|
| opening (table/W/closure/mask) | tail Vandermonde | message Vandermonde | `q` | `rank([A|B])=rank(A)=q` |
| mask joint (tilde, R, Eval, Dot PSC, ends) | `g16, c16, e16, U23, end2` | `W16` | 58 | rank 58, joint 58, kernel 15 |
| closure joint (PSC, ends, points) | `U49, shown5` | `f(Y)3` | 54 | rank 54, joint 54 |
| SEND/Eval/end/shown identity blocks | one per value | one per value | <=5 | rank equality |

The mask joint block explicitly places the same `c_t` in `R` and in the PSC
`sigma` column, so the `v1 = s + r, v2 = r` shape is represented rather than
assumed away. Across blocks, each tree has its own uniform tail and salts,
the material streams are one-use, the pads and the two mask plans are
separate, and the FOLD/BASE randomness is fresh. No random variable appears
in two blocks, so the concatenated matrix is block diagonal and the ranks add;
within the mask block the shared `c_t`/`g_t` columns are explicit. The test evaluates at deterministic and several random points.
No rank deficiency was found; if a run finds one, the run stops and reports a
possible leak. The tests are `joint_opening_submatrices_cover_the_message`,
`joint_mask_block_covers_the_secret_with_shared_claim_masks`,
`joint_closure_block_covers_psc_and_point_values`,
`joint_psc_hiding_maps_are_full_rank_at_multiple_points`,
`joint_fold_fresh_coefficients_map_is_full_rank`,
`joint_dot_mask_block_is_covered_for_zero_and_nonzero_chi` and the identity
block test.

### 3.12 PSC, FOLD and BASE coefficient maps

- Dot PSC: the 23 hiding entries map to the hiding sum and the sent round
  coefficients; the map is triangular by variable/degree, with unit diagonal
  on the entries that appear directly. The test computes rank 23 at three
  points. The determinant is a nonzero polynomial; a conservative degree
  bound is `D_PSC <= 23^2`, so a bad challenge has probability at most
  `D_PSC/p^3 < 2^-181`.
- Closure PSC: the same construction with 23 sumcheck variables and 49
  hiding entries; rank 49 at three points; `D_PSC <= 49^2`, bad challenge
  probability `< 2^-179`. The five shown masks cover the two end values and
  the three fresh-point values.
- FOLD: the fresh mask's `2k+1` coefficients fill the start sum and the sent
  `(c0,c2)` round values; the coefficient map is triangular by variable and
  degree with unit diagonal. Its determinant is a nonzero polynomial; a
  conservative degree bound is `(2k+1)^2`, giving a bad-challenge probability
  below `2^-178` for `k <= 32`.
- BASE: the star main row, star mask row and their tails are fresh per value;
  each public BASE value has its own star random dimension, so the multi-value
  map is an identity block (rank equal to the number of values), not a
  per-value implication. The joint test includes the multiple values in one
  matrix.
- The closed relation has no code-switch stage; a future CS path would need
  its own fresh-mask and star coverage and is not covered here.

## 4. Opening sets and submatrix property

The harness builds the code parameters actually used:

| Code family | `r` | Opening set per oracle | Remaining |
|---|---:|---:|---:|
| table | 8192 | 603 | 7589 |
| W and closure main | 8192 | up to 4 | at least 8188 |
| index and closure mask | 256 | up to 4 | at least 252 |

The ledger measures 72 code-bearing oracles, total rank 558,080, code-oracle
unique positions 879, minimum remaining 252 and zero exhausted tails. The
full opening accounting is 143 events, 887 logical draws and 887 unique
positions; the 8-position difference from the code-oracle total is on the
star messages, which have no code tail. Table reuse is zero.

`[ARGUMENT]` Every actual opening set has `|S| <= r` (603 <= 8192; each W and
closure set <= 4 <= 8192; each mask set <= 4 <= 256). By the Vandermonde
property in 3.2, `Im(B_msg) subset Im(A_tail)` for every such set, so the
message direction is masked on the opened positions. The old statement
"hiding dimension = r - union" is withdrawn: `r - q` is the unobserved tail
dimension, while message masking follows from surjectivity `|S| <= r`. The
ledger's minimum remaining 252 is the empirical margin; if a set exceeded
`r`, the argument would fail and the run must fail closed.

`[UNPROVEN]` The union count is accounting, not a formal rank theorem for the
programs. The Vandermonde property is exact for the stated polynomial code;
the program's execution of it is assumed.

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

`[UNPROVEN]` The matrix covers the implemented negative entries. It is not a
proof that every message field has a dedicated negative; fields reached only
through a shared root or a later relation are covered indirectly. A production
verifier must keep the full chain, not a subset.

## 6. Composition error accounting

`[FACT]` The base field is `p = 2^64 - 2^32 + 1`. Its cubic extension has
`p^3` elements with `log2(p^3) = 191.99999999899`, so `1/p^3` is slightly
larger than `2^-192`. The conservative bounds below use the looser powers:
`1/p^3 < 2^-191`, `30/p^3 < 2^-187`, `16*166/p^3 < 2^-180` and
`2/p^3 < 2^-190`.

The existing plan ledger gives a 128-bit target per round. With work counted,
the largest round errors are 2^-128.00 (FOLD rounds and BASE lambda); the four
position rounds are 2^-128.16, 2^-128.48, 2^-128.01 and 2^-128.50. Without work
the chain bottoms out at 2^-116.01. The plan's compiled per-request hash bound
is

```text
(t + q_V + 1) * 2^-128.00 + (17/2) * (t + q_V)^2 / 2^256,
```

with 126.21 bits with work and 116.01 without work. These are plan-ledger
numbers, not a new end-to-end security level.

`[ARGUMENT]` The component union bound used here is

```text
eps_total <= eps_hash + eps_components

eps_hash = (t + q_V + 1) * 2^-128.00
         + (17/2) * (t + q_V)^2 / 2^256
eps_components = 8 * 2^-128 + 2^-180 + 2^-187 + 2^-190 + 2^-191 + eps_rank_bad
eps_rank_bad <= (23^2 + 49^2 + (2k+1)^2) / p^3 < 2^-178
```

The terms are:

| Component | Count/degree | Bound |
|---|---|---:|
| audited proximity/LKP rounds | 8 rounds from the plan | `8 * 2^-128 = 2^-125` |
| range booleanity | 16 positions, degree at most 166 each | `16*166/p^3 < 2^-180` |
| PSC | 10 sumcheck variables, degree 3 each | `30/p^3 < 2^-187` |
| single equality | degree 2 | `2/p^3 < 2^-190` |
| `chi = 0` | one event | `1/p^3 < 2^-191` |
| PSC/FOLD rank determinant | degrees `23^2`, `49^2`, `(2k+1)^2` | `eps_rank_bad < 2^-178` for `k <= 32` |

The `t`/`q_V` factor is carried by the hash term, which is the adaptive
composition term; the algebraic component count is fixed per proof. If the
prover uses `Q_A` adaptive retry attempts, multiply the component terms by
`Q_A`; the hash term already carries the transcript query count. The challenge
draws are uniform over the cubic field. The hash term assumes the
plan's random-oracle/collision-resistant model and work bits; it is not
re-derived here. For `Q_R` independent requests, multiply the component sum by
`Q_R`; no global bound is claimed without a bound on `Q_R`.

The reference range check's error is counted as `2^-N` over its random
combination matrix, with a union bound over columns; this implementation's
closure booleanity is a polynomial identity checked at a random point, so the
degree-based `16*166/p^3` term is the relevant arithmetic model. The two are
not the same check and are not added as if they were; the plan's 128-bit round
accounting is used for the audited proximity/LKP rounds.

At `t = 2^64`, `2^96` and `2^120` the hash term dominates at about `2^-64`,
`2^-32` and `2^-8`; the component sum is about `2^-125` (before the `Q_R`
factor) and `chi = 0` is below `2^-191`. The figures `126.21` and `116.01`
bits remain plan-ledger numbers and are not presented as a new closed-chain
security level.

`[UNPROVEN]` No independent composition theorem is proved here. The round
figures are taken from the plan's accounting; the degrees and counts above are
arithmetic assumptions; some code-proximity bounds are assumed; the programs'
computation of the protocol is not proved. The `Q_R` multi-request factor is
not bounded. A conservative union bound over the components is not mechanized.

`[GAP]` If the range-decomposition, PSC or LKP terms are not instantiated at
the stated degrees/parameters, or if the number of requests is unbounded, the
composition figure does not apply; the security conclusion is conditional
and the 128-bit target is not claimed.

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

Items not backed by an executable rejection:

- `chi = 0` has no explicit check; it is covered only by the `1/p^3 < 2^-191`
  term in the composition argument.
- The full joint simulation, the all-challenge PSC rank and the composition
  theorem are document-level arguments.
- The output-binding boundary is the approved one; the full `Y` row is hidden
  by the conditional argument in 3.6. A change to the boundary requires a new
  review.
- The trusted X source, policy distribution/freshness and chain-based
  production verification are long-term direction records, not debts of this
  round.

## 8. Fail-closed summary

The following are open or conditional:

- `[UNPROVEN]` a full simulation and the adaptive composition of the hash and
  algebraic arguments; the linear-cover criterion itself is argued per view.
- `[UNPROVEN]` the all-challenge PSC rank; only one deterministic challenge
  point is tested.
- `[UNPROVEN]` a mechanized composition theorem and an end-to-end 128-bit
  claim; `126.21`/`116.01` remain plan-ledger numbers.
- `[ASSUMPTION]` the approved output-binding boundary (winner/valid/tie); the
  full `Y` row is hidden by the conditional argument, not by a `[GAP]` waiver.
- `[LONG-TERM]` the trusted X source, policy distribution/freshness and
  chain-based commitment/verification are recorded as long-term direction;
  they are not implemented here and are not debts of this round.
- `[EXCEPTION]` `chi = 0` is a public event included as a term. An opening
  union above the tail rank or a reused mask/pad/material stream is an
  executable ledger rejection. A failed hash model, a broken PSC/FOLD
  full-rank assumption or an unapproved output boundary withdraws the
  security conclusion; these are conditional assumptions, not runtime
  fail-closed checks.

No statement in this note claims that the trusted X source is proved, that the
weights are hidden under a formal ZK definition, or that the run is a
production acceptance.
