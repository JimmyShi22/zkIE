# Closed output-head relation: unbiased sampling evidence, joint views and the whole-chain conditions

## Status and labels

This note is the whole-chain argument for the closed hidden output-head
relation. It follows one path: actual random quantity -> every public
position -> sharing relations -> hiding or binding reason -> exception
probability -> code or plan basis. It is an accounting and argument document,
not a mechanized proof and not a claim that the trusted X source is proved.

Every claim carries two labels:

- a status label: `[FACT]` checked in the code or arithmetic at the snapshot;
  `[ASSUMPTION]` a condition the argument needs; `[UNPROVEN]` stated but not
  proved here; `[GAP]` a missing check or mechanism (an executable condition
  rejects the run when it fails, otherwise the security conclusion is
  conditional);
- an evidence grade: `NUM` a real-evaluation-path numerical test; `CODE` a
  manual derivation from the implementation; `MATH` a general mathematical
  result cited for the stated code (for example the Vandermonde theorem).

`NUM` is not a proof of the general statement and does not claim that every
actual opening set was numerically measured. Passing the test-only module is
not a production runtime detector for all bad ranks; the workspace and run
numbers in this round are reported results, not an independent external
verification.

Snapshot: the closed relation at code basis `3b3e8ecc`, with the bounded-read
rollback `8d32d3de`; fixture `gpt2-seq16`, 16 positions. The default two-stage
formal proof is 65,432,832 B (the previous single-stage snapshot was
9,939,144 B); the interval three-stage comparison is 212,523,920 B.
The committed document is version 15; version 15 adds the interval planner and its accounting; version 14 added the CS switch random
material and the wired multi-stage status.

## 1. The chain in one view

| # | Actual random quantity | Where it is used publicly | Shared with | Why the public view hides or binds | Exception probability | Evidence |
|---|---|---|---|---|---|---|
| 1 | table row tail and leaf salt | table column openings, table root | nothing across trees | Vandermonde tail covers the opened symbols; salt plus leaf hash hides unopened leaves | none for the rank step; hash model terms in section 6 | `NUM` q=603 block, `MATH` Vandermonde |
| 2 | W/closure/mask row tails and leaf salts | their SEND answers, openings and roots | nothing across trees | tail Vandermonde covers the opened symbols; the same tail is reused across stages of one tree (union); leaf hash hides unopened leaves | hash model terms | `NUM` q<=4 blocks, `CODE` union, `MATH` |
| 3 | index-mask 73 entries | mask root, SEND answer, Dot PSC transcript, tilde/Eval values, public residual | `g_t` in tilde and R; `c_t` in R and Dot PSC sigma | each entry is a one-time pad or a unit-triangular hiding coefficient; the shared columns are explicit in the joint matrix | `chi=0` and sumcheck degrees in section 6 | `NUM` 58-row joint matrix, `CODE` |
| 4 | closure-mask 49 hiding plus 5 shown per position | mask root and openings, per-position PSC transcript and end values, three point values | the same message entry appears in the encoding opening and in the PSC or point values | tail covers the opening symbols; the hiding map covers the message vector including the product-dependent last round; the shown masks one-time pad the two end values and the three point values | sumcheck degrees; beta nonzero by construction | `NUM` joint mask object and real-prover decomposition, `CODE` |
| 5 | SEND pads and per-message pads | SEND answers | the answer's own pad | one-time pad on the sampled value | none beyond the sample/answer check | `NUM` real answer function, `CODE` |
| 6 | Eval masks `e_t` | masked Eval targets | one per position | one-time pad on `W(psi)` | none | `NUM` real evaluation function, `CODE` |
| 7 | Dot PSC hiding coefficients | Dot PSC transcript and end values | the same 23-entry hiding class | the hiding map is triangular with a nonzero constant determinant; the two shown end masks one-time pad the end values | sumcheck degree terms in section 6 | `NUM` 23x23 map, `CODE` triangularity |
| 8 | closure PSC hiding coefficients | per-position PSC transcript | the same closure mask message | same triangular map, one instance per position | sumcheck degree terms; 16 instances | `NUM` 49x49 map, `CODE` |
| 9 | FOLD fresh coefficients | FOLD start sum and round values | the same fold instance | triangular coefficient map with constant nonzero determinant; the start sum and every sent round value are covered | sumcheck degree terms; two FOLD instances | `NUM` k=2 and k=14, `CODE` |
| 9a | switch main/mask tails, pad and salts | switch root, sampled answer and switch openings | the previous stage's tails are placed in the switched message; fresh switch values are one per switch | the fresh switch tail is the Vandermonde block on the opened symbols; old tails are fixed or fiber-uniform after the previous tree conditioning; the leaf hash hides unopened symbols | hash model terms; per-stage sampling terms | `NUM` real staged path, `CODE` |
| 10 | BASE star main/mask rows and tails | BASE target, combined rows and star openings | target, rows and openings of the same star | fresh one-time pads on the target dot product, each combined coordinate and each tail coordinate | BASE lambda/position terms are in `eps_proximity` (ineffective at current parameters; section 8) | `NUM` real star path, `CODE` |
| 11 | challenge draws | every PSC/FOLD/BASE/position challenge | none | drawn after the message they test; zero-fallback event bounded | 2^-192 per draw fallback, `chi=0` term | `CODE` draw functions, `NUM` order tests |

The rest of the note expands each column.

## 2. Actual random quantities

`[FACT]` `CODE` The generation paths are:

- table rows use the base field `Fp`; the row tail is drawn from the stage
  stream by the base-field draw;
- W, closure and mask rows use the cubic field `Fp3`; their tails, salts and
  mask entries are drawn by `ByteMaskCoins::value_at`, which builds three
  `Fp` coordinates from one base hash and a coordinate/attempt-specific retry
  domain, and returns `Fp3::new(c0, c1, c2)`;
- the challenges use the chain draw: `draw_f` takes a 64-bit word below the
  modulus by rejection sampling (`DRAW_TRIES = 6` attempts), `draw_e` takes
  three base draws, and `draw_e_nonzero` retries a fixed number of times and
  then returns one;
- the per-oracle streams are separated by a material commitment and a run
  seed; one stream is not reused by another oracle;
- each switch draws per-row main and mask tails, one nonzero pad and one salt
  per switched-tree column from a stage-indexed switch domain; each stage FOLD
  draws `2k+1` fresh coefficients from its stage seed and its own column
  salts; the last BASE star draws its main/mask rows and tails from a separate
  stream. `switched_rows_multi` places the previous stage's main/mask tails
  into the new mask message, so those old tail values are shared between the
  old tree and the switch tree rather than independently resampled.

`[FACT]` `CODE` Field and distribution transfer:

- an `Fp3` random quantity is a triple of independent `Fp` draws, so it is
  uniform over `Fp3`; a linear map over `Fp3` applied to such a vector is
  uniform over its image;
- the base-coded table uses `Fp` on both sides; its rank argument is over
  `Fp`;
- the secret values (weights, reduced rows, bit values) are `Fp` elements
  embedded in `Fp3`; the covering map has `Fp3` coefficients and the random
  vector is `Fp3`-uniform, so the rank criterion over `Fp3` applies to the
  embedded secret directions as well;
- the challenge draw uses rejection sampling, so the accepted word is uniform
  on `Fp`; the fixed-attempt fallback returns zero with probability below
  `(1 - p/2^64)^6 < 2^-192` per challenge draw. This is carried as a small
  exception term in section 6.

### 2.1 Unbiased mask sampling

`[FACT]` `CODE` The mask stream rejects a candidate word at or above the
modulus instead of reducing it:

- the base hash of a draw supplies one candidate word per coordinate; the
  draw number is the hash index and the seed is the hash data;
- a rejected coordinate is retried in a coordinate- and attempt-specific
  domain: the retry subdomain is `1 + coordinate` and the attempt number is
  part of the hash data, so the draw number, coordinate and retry number are
  domain-separated;
- the sequential `draw`, the random-access `redraw(number)` and the pure
  `value_at(number)` all call the same function, so they agree for the same
  draw number;
- exhaustion returns `EntropyUnavailable`. There is no modulo fallback.

`[FACT]` `CODE` `MATH` Ideal-word distribution. If each candidate word is
uniform on 64 bits, a candidate is accepted exactly when it is below `p`, so
the accepted value is uniform on `Fp`; rejection does not bias the accepted
distribution. The three coordinates use independent words (the base-hash word
index or distinct retry domains), so the resulting `Fp3` element is uniform on
`Fp3` in the ideal-word model. The retry-exhaustion probability per coordinate
is `(1 - p/2^64)^64 < 2^-2048`; for `N` coordinate draws the union is below
`N * 2^-2048`, which is far below every other term in section 6.

`[ASSUMPTION]` `CODE` Real finite seed. The implementation derives the
candidate words from `hash_labeled` with a per-run seed. The uniform
ideal-word statement is therefore a model-level statement: the hash is treated
as a random oracle (or a pseudorandom generator/function) so that its outputs
are computationally indistinguishable from independent uniform words. It is
not an unconditional infinite-independence claim about the concrete hash.

`[FACT]` `CODE` `NUM` The audit found four modulo reductions of random words:
the production `ByteMaskCoins::value_at`, and three test/fixture coin sources.
All four now use the rejection helper; no `word % p` reduction of a random
word remains. The production hidden chain draws masks, tails, pads and stars
through `ByteMaskCoins`; the test/fixture sources are used by component tests
and fixtures, not by the hidden production path. The table fixture's tails and
salts are deterministic fixture data supplied by the holder, not draws of the
per-proof mask stream; the production table commitment is external and
policy-pinned.

`[FACT]` `NUM` The tests cover: `p-1` accepted, `p` and `2^64-1` rejected and
retried, exhaustion after the fixed number of attempts, sequential/random
access agreement, deterministic replay for the same seed, distinct streams for
different seeds, and no repeated values in a sample. The source tests are in
the field, fold and hidden-session modules.

`[FACT]` `NUM` Real-path rejection evidence. A test-only forced-candidate
source makes the real `value_at` base candidate at or above `p`; the real
retry path then runs: the retry tag uses subdomain `1 + coordinate`, the draw
number as the hash index and `seed || attempt` as the data; a second retry is
forced, the coordinates and draw numbers are separated, and `draw`, `redraw`
and `value_at` agree on the same draw when a rejection occurs. The retry tag
is checked against the manual hash of the same domain, and exhaustion through
the real path returns `EntropyUnavailable`. The helper-level tests and the
real-path tests both cover the rejection branch.

`[FACT]` `NUM` The env-gated counter (`ZKIE_SAMPLE_COUNTS`) records the
accepted and rejected candidate counts in the run log. The counter is
process-level: after it is enabled, every `uniform_fp` call increments,
including `redraw` and repeated evaluations of the same draw. The accept and
reject totals are two separate `Relaxed` atomics, not a single atomic
snapshot of one draw, and the totals are not the number of independent random
coordinates or all randomness in the system. The three timed runs include the
atomic overhead; they are not exactly comparable to the earlier no-counter
baseline and are read within the recorded spread. The runs recorded
`sample_accepts = 276,343,251` and `sample_rejects = 0` in each run; the
identical proof hash is reported together with the measured rejection count,
not as an inference from the hash. The helper that follows the first accepted
retry uses attempts `1..SAMPLE_TRIES` (1..=63), matching the production
attempts 0..=63; attempt 64 cannot execute.

`[ASSUMPTION]` Unbiasedness is a property of the honest prover implementation:
the verifier does not rederive the hidden mask stream and cannot enforce the
sampler. The security argument assumes the prover follows the rejection
sampler; a prover that samples differently is outside the claim. This is not a
runtime check of the verifier.

`[FACT]` The schema and encoding versions are unchanged because the message
structure is unchanged and there is no deployed compatibility surface that
the version would have to distinguish. A version change would be a
security-semantics change and is not made here; if one were required, it
would be escalated rather than decided locally.

## 3. Public positions

`[FACT]` `CODE` The public view consists of:

| Public item | Actual form | Depends on random | Depends on secret |
|---|---|---|---|
| policy pin and statement | table identity, roots, output plan | none | no |
| table root and column openings | codeword symbols `sum_{i<r} t_i z^i + sum_{i<ell} m_i z^{r+i}` | tail, salt | table message rows |
| W and closure roots, SEND answers, openings | codeword symbols; sampled value plus pad | tail, pad, salt | W/closure message values |
| index-mask root, answer, openings | codeword symbols; sampled mask value plus pad | 73 entries, tail, salt | none (mask message is random) |
| tilde values, public residual R | `W_t + g_t`; `sum chi^t (g_t - c_t)` | `g_t`, `c_t` | W values enter tilde only |
| Dot PSC transcript | 23 values, an affine image of the 23 hiding entries plus the target shift | 23 hiding entries, two shown masks | claim target / witness message vector |
| Eval targets | `W_t(psi) + e_t` | `e_t` | W values |
| closure mask root and openings | encoded mask symbols | 49+5 entries per position, tail, salt | none |
| closure PSC transcripts | one per position, affine image of the 49 hiding entries and the two shown end masks | 49 hiding, 2 shown per position | message vector and target shift |
| closure point values | `(B(pt)+shown_2, D(q)+shown_3, weighted B(q)+shown_4)` | 3 shown masks per position | bit/map values |
| switched message roots, sample answers and openings | the new message's codeword symbols at the drawn positions; the sampled row evaluation | switch main/mask tails, pad and salts; old tails; stage-FOLD coefficients | old row values through the cut map; no plaintext |
| FOLD start and round values | start sum and `(c0,c2)` per round | 2k+1 fresh coefficients | group-row values |
| BASE target, combined rows and openings | `target*`, `star + lambda*row`, encoded star symbols | star main/mask/tails | final row and claims |
| output and seal | winner, valid range, tie result | none | approved output boundary |

The secret enters public positions only through the message values encoded in
the committed rows, the masked SEND/Eval/point values, and the PSC/FOLD
message vectors. It never appears as a plaintext row.

## 4. Sharing relations

`[FACT]` `CODE` `NUM` The shared-variable attribution is:

| Random variable | Public positions | Shared? |
|---|---|---|
| table/W/closure/mask tails and salts | the tree's own openings and answer samples | no across trees; union within one tree across stages |
| `g_t` | tilde value and R | yes, both rows, explicit |
| `c_t` | R and Dot PSC sigma; the claim relation is checked by proximity, not sent as a field value | yes, R and sigma explicit |
| `e_t` | one Eval target | no |
| Dot `U` (23 entries) | Dot PSC transcript | no |
| closure hiding `U` (49 per position) | that position's PSC transcript | the same message entry also appears in the mask opening |
| closure shown (5 per position) | two end values and three point values | the same message entry also appears in the mask opening |
| SEND pads | the answer they pad | no |
| switch main/mask tails, pad and salts | switch root, sample answer and switch openings | old tails are placed in the switched message and shared with the previous tree; fresh switch values are per switch |
| FOLD fresh (`2k+1`) | start sum and round values of one fold | no |
| BASE star main/mask/tails | target, combined rows and openings of one star | yes, the same star; explicit in the BASE matrix |

The explicit counterexample check is the pair `(tilde, R)` with shared `g_t`
and the pair `(R, sigma)` with shared `c_t`: the joint mask matrix keeps the
same column in both rows, so the `v1 = s + r, v2 = r` shape is represented
rather than assumed away.

The encoded mask message adds a second sharing: one mask entry appears in
the tree opening (inside an encoded symbol), in the PSC transcript (through
the hiding map), and in the published end or point value (for the shown
entries). Section 5.5 treats these three positions in one joint matrix.

The BASE star adds a third sharing: the same star main/mask row appears in
the target dot product, in each combined row and in the encoded opening. The
joint BASE test in section 5.6 uses the same star.

Across different oracles the tails, salts, streams, pads and mask plans are
different. This is the plan's source separation and the basis of the block
partition of section 5.7; section 5.7 proves the block-diagonal composition
(disjoint fresh variables plus the conditional block lemma). Within one oracle
the shared columns are explicit.

## 5. Hiding and binding reasons

### 5.1 Linear-cover criterion

For one public view write `v = A r + B s`, where `r` is the random material
and `s` the secret. Hiding holds if

```text
Im(B) subset Im(A),   equivalently rank([A | B]) = rank(A).
```

For every secret `s` there is an `r'` with `A r' = A r + B s`, so the view is
independent of `s`. A large kernel alone is not used as evidence. This is the
criterion applied below. `MATH`.

The field transfer from section 2 applies: when the random vector is a triple
of independent `Fp` draws, it is `Fp3`-uniform; when the code and message are
base-field, the argument is over `Fp`. The secret values embedded from `Fp`
are covered by the same inclusion. `CODE`.

### 5.2 Openings: Vandermonde submatrix

`[FACT]` `MATH` A codeword is the evaluation of a polynomial with tail
coefficients at degrees `0..r-1` and message coefficients at `r..r+ell-1`.
For an opening set `S` of `q` distinct positions,

```text
A_tail = [ z_j^i ]_{j in S, i < r}       (q x r)
B_msg  = [ z_j^{r+i} ]_{j in S, i < ell} (q x ell)
```

If `q <= r`, the q x q left block of `A_tail` is a Vandermonde matrix; it is
invertible for distinct points, so `A_tail` has row rank q and
`Im(B_msg) subset F^q = Im(A_tail)`. The opened symbols are uniform over
`F^q`, independent of the message. The union of a tree's SEND and BASE
openings is used, so cross-stage reuse is included.

`[FACT]` `NUM` The real code parameters and the actual opening counts are:
table `r = 8192`, `q = 603`; W and closure main `r = 8192`, `q <= 4`; index
and closure mask `r = 256`, `q <= 4`. The test builds the actual q x q
Vandermonde block with the code's point function and confirms full rank for
q = 603, 256, 16 and 4. The q x r matrix then has rank q because its q x q
block does. This is a numerical check at the actual q, not a claim that every
actual opening set was measured; distinctness of the actual set follows from
the transcript/ledger positions.

`[FACT]` `CODE` `q <= r` is the hiding budget: the Vandermonde tail block has
rank `q` only under this condition, so the opened symbols are uniform and the
message is covered. It is not a proximity soundness condition; the proximity
error comes from the plan's distance, query and work parameters.

`[FACT]` `CODE` The ledger measures 72 code-bearing oracles, total tail rank
558,080, code-oracle unique positions 879, minimum remaining 252 and zero
exhausted tails; the full opening accounting is 143 events, 887 logical and
887 unique positions. The 8-position difference is on the star messages, which
have no code tail. If a set exceeded `r`, the executable ledger rejects the
run.

`[UNPROVEN]` The union count is accounting, not a formal rank theorem for the
programs; the Vandermonde property is exact for the stated polynomial code,
and the program's execution of it is assumed.

### 5.3 SEND answers, Eval targets and shown values

`[FACT]` `CODE` `NUM`

- SEND answer: `answer = sample(message) + pad`. The answer function reads the
  mask/message entries at the sampled cube point and adds the reserved pad
  coefficient; the reserved coefficient is the nonzero beta of the seed. The
  numerical test builds the answer by calling the real answer function on unit
  messages and checks the coefficient of every entry: the ordinary entries
  carry the equal-weight coefficients, the last entry carries beta. The map is
  one row with a nonzero pad coefficient, so a uniform pad makes the answer
  uniform.
- Eval target: `target = W(psi) + e`. The test calls the real multilinear
  message evaluation on unit messages and checks every coefficient; the mask
  `e_t` is fixed before `psi` is drawn and gives one random dimension on the
  single target row.
- shown values: the closure point values are the real multilinear and
  equal-weight inner products plus the shown masks at offsets 2, 3 and 4; the
  two PSC end values are the two shown maps at beta plus the shown masks at
  offsets 0 and 1. Section 5.5 puts them in the joint matrix.

### 5.4 PSC hiding map and end values

`[FACT]` `CODE` The hiding polynomial is
`U(X) = sum_l w_l(X_l) prod_{j>l} psi_j(X_j)` with coefficients in
`hp_place` order. The sent values are the hiding sum and, for every variable,
the round coefficients with the fixed coefficient omitted. The map from the
entries to the sent values has a block-triangular form when processed from
the last variable backward:

- the last variable has three entries. Its power-0 entry appears in the sum
  row, and its power-1 and power-2 entries are the two sent coefficients of
  its own round; this is a 3 x 3 block with nonzero constant diagonal;
- for a plain variable `l`, the power-2 entry is its own sent power-2
  coefficient; the power-1 entry appears in the preceding variable's `after`
  value and hence in that variable's sent power-0 coefficient, with the
  nonzero constant factor from `apply_l` (for the first variable the sum row
  plays the role of the preceding row); this is a 2 x 2 block with nonzero
  constant diagonal.

The fixed coefficient is reconstructed by the verifier with `fixed_inv = 1`,
so the same triangular system covers the full public view. The determinant is
the product of these nonzero constants and is independent of the challenges;
there is no bad-challenge set for this map, so no Schwartz-Zippel term is
needed here.

`[FACT]` `CODE` The per-entry correspondence for k = 23 (49 entries, 49 sent
rows) is:

| Entry (hp_place order) | Count | First public occurrence | Direct coefficient | Challenge dependence |
|---|---:|---|---:|---|
| last variable, power 0 | 1 | sum row | 1 | none on the direct term |
| last variable, power 1 | 1 | last round, sent power 1 | 1 | none on the direct term |
| last variable, power 2 | 1 | last round, sent power 2 | 1 | none on the direct term |
| plain variable `l`, power 2 | 23 | its own round, sent power 2 | 1 | none on the direct term |
| plain variable `l`, power 1 | 23 | preceding variable's sent power 0; for `l=0`, the sum row | 1 | the preceding row also carries its own challenge-dependent terms |

The sent-row order is the hiding sum, then for every plain variable the two
non-fixed coefficients (powers 0 and 2), then the last variable's two
non-fixed coefficients (powers 1 and 2). In the order
`[sum, last round, variable 22 round, ..., variable 0 round]` the direct
entries above form the diagonal blocks; entries of a later variable that
appear in an earlier row are the strictly triangular part and are multiplied
by challenge-dependent `after`/`apply` factors. The determinant is the
product of the constant diagonal entries up to the sign of the block
ordering, so it is a nonzero constant independent of the challenges; the
observed signs (`+1` at k=10, `-1` at k=23) are numerical corroboration, not
the general proof.

`[FACT]` `CODE` The round/bind and first-term constraints are the actual code
steps:

1. `HidingPolynomial::new` fills the coefficient vector of each variable at
   powers `first..degree` (`first = 1` for a non-last variable, `0` for the
   last one) with the entries in `hp_place` order. The fixed index is the
   variable's `fixed()` (1 for a plain variable, 0 for the last one); its
   coefficient is not filled for a plain variable and is the last variable's
   power-0 entry.
2. `round` returns `coefficients + before * psi + after` (with `after` added
   to the constant coefficient). `sent_coefficients` omits the fixed index;
   the verifier's `complete_round` reconstructs it from the running claim and
   the sent coefficients with `fixed_inv = moment[fixed]^{-1}`, which is one
   for both variable kinds.
3. `bind` sets
   `before = before * psi(challenge) + poly_at(coefficients, challenge)`; this
   is the only way an earlier variable's fixed-power entry enters the later
   rows, and it is the triangular coupling in the table.
4. The first running value is `hiding_sum + eps * target`; every round must
   satisfy `apply(round) = running`, then the challenge binds and the next
   running value is the round at the challenge. The verifier repeats the same
   fixed-coefficient completion and the same checks. The end values are
   `ends_i + beta * shown_i` with `beta` nonzero.

The derivation assumes the plan's moments (`plain` with fixed index 1 and
`at_zero` with fixed index 0), the `fixed_inv = 1` completion, and the
`hp_place` ordering. If any of these preconditions fails, the constant
determinant statement is not claimed; the failure would be a configuration
error or a different map, not an unhandled challenge event. The three-point
numeric rank/determinant checks are corroboration of the derivation, not a
substitute for it.

`[FACT]` `NUM` The 23 x 23 Dot map and the 49 x 49 closure map are full rank
at three deterministic/random challenge points. Their determinants are `+1`
for k = 10 and `-1` for k = 23 on every tested point; the sign may depend on
the variable count, but it does not depend on the challenges. The FOLD map
has determinant `+1` for k = 2 and k = 14 on every tested point. These are
numerical confirmations of the constant-determinant derivation, not the
derivation itself.

`[FACT]` `CODE` The two shown end masks are used once each:
`end_i = ends_i + beta * shown_i`, with `beta` drawn nonzero and after the
round message. For fixed `ends`, the two end values are one-time padded; for
uniform shown masks they are uniform. The numerical test checks the affine
relation with the real prover.

`[FACT]` `CODE` The last round is the product of the two masked shown maps,
so it contains products of the shown masks. That dependence is part of the
unmasked message vector, not a separate public direction: the hiding
polynomial covers the whole sent message vector, including the last round.
The numerical test fixes the shown masks and checks that the public transcript
is affine in the hiding entries for the real prover; the end-value test checks
the separate one-time pad. The two facts together give the hiding reduction:
first the message vector is masked by the invertible hiding map, then the two
end values are padded by the nonzero beta.

### 5.5 The closure joint object

`[FACT]` `CODE` One position of the closure mask message has 49 hiding
entries, five shown entries and the encoding tail. The same message entry
appears in three public positions:

1. the encoded mask opening: a codeword symbol is
   `sum_{i<r} tail_i z^i + sum_{j<ell} message_j z^{r+j}`;
2. the per-position PSC transcript: the hiding map applied to the 49 hiding
   entries, plus the message vector and the two shown end values;
3. the published values: the two end values and the three point values, each
   padded by one of the five shown entries.

`[FACT]` `NUM` The joint matrix uses the real formulas: the hiding block is
built from the real hiding polynomial, the shown block is the identity on the
two end and three point rows, and the opening block is built by calling the
real encoder on unit message and tail vectors. The rows are the 2k+3
transcript values, the two end values, the three point values and the q
opening symbols; the random columns are the mask message entries and the
tail; the secret columns are the transcript/message coordinates and the
published values. At k = 4, ell = 32, r = 16, q = 2 the random rank is 18 and
the joint rank is 18: every secret column lies in the image of the random
columns. The real-prover decomposition test confirms that, with the shown
masks fixed, the transcript is affine in the hiding entries; the end-value
test confirms the one-time pad.

The derivation for the actual k = 23, ell = 512, r = 256 is the same block
argument: the 49 x 49 hiding map is invertible for every challenge by the
triangular constant determinant; the five shown entries are independent unit
columns on the end/point rows; the tail block has rank q for q <= r and its
image contains the opening parts of the message columns. The blocks are
orthogonal in row coordinates, so the ranks add; the secret rows are covered
by the hiding block and the shown columns. This is why the encoded mask
message, the PSC transcript and the published values are jointly hidden even
though they share the same message entries. `CODE` `MATH`.

`[FACT]` `NUM` The production-parameter check uses 16 positions, k = 23, two
mask rows, ell = 512, r = 256 and q = 4. The real one-symbol evaluation
builds an 872 x 1536 matrix: 16*49 PSC rows, 16*5 shown rows and 2*4 opening
rows; the random columns are the 2*(512+256) message and tail entries; the
secret columns are the 16*54 transcript/shown coordinates. The random rank is
872 and the joint rank is 872. This is a numerical confirmation of the block
derivation at the production shape. It is not a claim that the complete
multi-oracle transcript (index-mask, FOLD, BASE and closure together) was
extracted into one matrix; the blocks are analyzed separately, and the
cross-block joint composition is the open item of section 5.9, not a
direct-sum consequence of the sharing table.

`[UNPROVEN]` The closure point values are evaluated at points drawn after the
opening and before the next challenge; the adaptive composition of these
challenge draws with the hiding argument is not mechanized. The hybrid
argument in section 5.7 states the condition.

### 5.6 FOLD and BASE

`[FACT]` `CODE` `NUM` FOLD: the fresh mask has `2k+1` coefficients; the sent
values are the start sum and `(c0,c2)` for each of the k rounds. The same
block-triangular argument as the PSC applies to the fold round recursion: the
diagonal blocks are nonzero constants, so the determinant is a nonzero
constant independent of the challenges and the map is full rank for every
challenge. The test runs the real fold prover
for the test point k = 14 (the actual fixture own fold is also k = 14) and for k = 2 at three challenge seeds;
the 29 x 29 and 5 x 5 matrices are full rank and have determinant one. The single-stage
closed path has two FOLD instances (the shared and own proximity groups); a
staged group adds one stage FOLD after each switch, using the same
fresh-coefficient map, with fresh per-stage seeds and separate openings/tails.

`[FACT]` `CODE` The FOLD round/bind steps are the actual code behavior:

1. `FoldRoundProver::new` takes the `2k+1` fresh coefficients in order and
   builds the fresh mask table with `fold_mask_table`; `start(eps)` is called
   after the fold start is sent.
2. `u_round` reads `round_coefficient(coefficients, k, variable, 0/1/2)` and
   adds `after[variable]` to the constant and `before` to the linear
   coefficient; the main/mask terms and the kept term are added with `eps`.
3. The sent round values are `(c0, c2)`; the linear coefficient `c1 = before`
   is determined by the running claim and is not sent. `bind(gamma)` updates
   `before`, folds the main and mask tables, and updates `kept` and `later`.
4. The first running value is the fold start sum; the verifier checks
   `apply(round) = running` for every round, then binds the challenge. The
   verifier repeats the same coefficient completion.

The per-entry mapping is the same block-triangular form: the `2k+1` fresh
coefficients fill the start sum and the sent `(c0,c2)` rows with nonzero
constant diagonal blocks; the linear coefficient of each round is the
`before` value from the earlier entries and is not an independent public
entry. The three-point determinant check is corroboration of this derivation.

`[FACT]` `CODE` CS switch. For a non-last stage, `switched_rows_multi`
maps the old single rows into the target codes: old main values are cut by the
target index, old masks and old main/mask tails are placed in the target mask
message at the plan's rest places, and the plan pads are filled with one fresh
nonzero pad. The new message has fresh main and mask tails per row and one
salt per column; `commit_switch` commits it under the stage tree plan. The
public switch record carries the plan-derived target, the declared root, one
sampled answer, the old-member opening, the fresh-mask opening, and the stage
FOLD transcript when present. The target is compared with the trusted plan
before any opening is used; the root is an RO output; the sample answer is the
sampled row evaluation; the openings are the encoded symbols at the drawn
positions.

`[MATH]` The switch map is affine in the old row values, the old tails and the
fresh switch tails. The fresh switch tail is the new tree's tail Vandermonde
for `q <= r`; the old tails are the same variables that already appear in the
previous tree, so conditioning on that tree's root/opening either fixes them
or leaves them uniform over the unopened fiber. The old secret row values
enter through the deterministic cut map, not as a new plaintext position. The
switch opening is therefore covered by the previous tree's tail map plus the
fresh switch tail; the sample answer is a single evaluated row with the switch
pad. The following stage FOLD is the invertible fresh-coefficient map above,
and the fresh-mask opening uses the stage-fold tail. The new switch tree is a
new random source: its tree number, run, configuration and salt stream are
disjoint from the stage-0 group, the stage FOLD and the BASE star, and the
external ledger rejects reuse. No plaintext old row appears in the switch
record.

BASE: `LastBaseProver` draws a fresh star main row, mask row and tails from
its own stream. The public target is
`dot(claim, star.main) + dot(mask_claim, star.mask)`; each combined row is
`star + lambda * row`; each tail is combined the same way; the star opening
is the encoding of the same star rows. The real-path test checks:

- two different seeds give different star rows and tails;
- the target equals the two real dot products of the same star;
- every combined row and tail equals the real one-time pad formula.

`[FACT]` `CODE` Simultaneous elimination. The verifier's linear check uses
the same star in the target and in the combined rows:

```text
target* = <claim, star.main> + <mask_claim, star.mask>
main'   = star.main + lambda * main
mask'   = star.mask + lambda * mask

<claim, main'> + <mask_claim, mask'>
  = target* + lambda * (<claim, main> + <mask_claim, mask>)
  = target* + lambda * target
```

The star cancels. The relation left after elimination is the public
claim relation `target` (the verifier knows the claim and target from the
committed rows), not the secret row. The combined rows and the target alone
leave the star underdetermined: they give one dot product and one padded
coordinate per row, while the star has many coordinates. The openings reveal
the encoded star symbols at the drawn positions, masked by the star tail with
`q <= r`, and the old row symbols masked by the old row tail. The star is
fresh per base value and drawn from its own stream; the two-seed test is an
illustration, not the uniformity proof. The uniformity/one-time argument is
the generation relation: `draw_star` fills the star rows and tails from a
fresh `ByteMaskCoins` stream, and the public formulas above are the only
places they enter.

The joint BASE view is therefore a set of one-time pads on the secret rows
and on the target dot product; the star coordinates are fresh and appear in
the target, the rows and the opening consistently, which is exactly the
shared-variable pattern the joint criterion handles. `CODE` `NUM`.

The FOLD and BASE parts share the opening/tail accounting: a BASE opening is
an encoded star symbol masked by the star tail; a FOLD transcript is masked by
the fresh coefficients. They do not share random coordinates across the two
mechanisms; the per-block statements are composed by the block partition and
the conditional block lemma of section 5.7.

### 5.7 Hash roots and commitment hiding

`[FACT]` `CODE` A runtime leaf is
`hash(tag, leaf_index, run, width, salt_length, symbols, salt)`; inner nodes
use the tree configuration. The salt has 40 bytes. The leaf hash is the only
place the unopened symbols enter the root.

The correct model is a random-oracle commitment, not a generic collision-
resistant hash:

- **RO commitment (hiding).** In the random-oracle model the leaf digest is a
  uniform random value independent of the leaf symbols and salt; the root is a
  hash of these values. An unopened leaf's digest is therefore uniform and
  independent of its symbols. This is the unopened-leaf hiding lemma.
- **CR hashing (binding).** Collision resistance is used for binding: the
  root binds the opened leaves and the leaf index/plan bind the position. CR
  alone does not imply hiding; the two assumptions are not interchangeable.
- **Per-use record.** Root hiding uses RO; root binding uses CR; leaf binding
  uses CR plus the index/run/plan; salt entropy supports the RO input
  distinctness; the mask/tail/pad entropy supports the leaf input
  min-entropy. In the fixture the inputs meet the stated lengths.

`[FACT]` `CODE` Hybrid order:

1. Replace every root and leaf digest by an independent uniform RO output
   (or program the RO at those points in the simulator).
2. Run the algebraic cover argument on the openings, answers, transcripts and
   star rows, with the roots treated as fixed public values.
3. Use the RO commitment hiding for the unopened leaves and the CR binding
   for the opened ones.

Roots are never treated as independent random field elements. In particular the
old statement "conditional on a root the tail remains uniform" is withdrawn:
conditioning on a hash output can correlate with the input in a CR-only model.
The correct statement is the RO replacement above, where the root is uniform
and independent of the input by construction; after the replacement the tail
is uniform conditional on the programmed root.

`[CODE]` Adaptive challenges: the challenges are drawn after the roots and
answers are absorbed. Under the RO replacement, a root is independent of the
message and tail, so the challenge distribution is independent of the secret;
the algebraic cover argument can be conditioned on the root without changing
the secret distribution. This adaptation argument depends on the RO model for
the root and leaf outputs, not on CR.

`[FACT]` `CODE` `MATH` Online simulator construction. The simulator follows the
real transcript order; it does not preselect the PSC/FOLD rounds or the BASE
values before the challenges. Its inputs are the public statement and plan,
the random-oracle interface, and a witness `w'` that satisfies the same public
statement (the dummy-witness strategy below). It does not receive the future
opening set, and the salt is never reselected. Write `q` for the cumulative
number of random-oracle queries and calls (leaf, node, root, challenge and
opening) made by the adversary and the simulator up to the event.

1. **Prefix and roots.** Start from the fixed public statement: the output
   boundary (winner/valid/tie), the statement and plan digests, the public
   table identity and the fixed statement/activation roots. For every leaf
   choose one 40-byte salt at allocation time, store it, and announce a
   uniformly sampled digest `com_i` (lazily and recorded). The root is the
   Merkle hash of the recorded `com_i`; under the RO replacement it is a hash
   of uniform digests. The public statement/activation roots are fixed setup
   inputs: use them as given and do not program them. A prover-commitment root
   is part of the simulator's own message stream; it is generated by the
   lazy-sampled RO and, if a value must be matched, programmed only at an
   unqueried input, charged to the collision/programming event. The salt is
   never reselected.
2. **First-stage online messages.** Absorb the roots and declarations, then
   draw the shared seeds; generate the SEND answers with fresh pads; draw the
   W/mask positions; generate the openings from the fresh tails; draw `psi`;
   generate the Eval targets `W_t(psi) + e_t` with fresh `e_t`; absorb the Eval
   targets; generate the tilde values `W_t(psi*) + g_t` with fresh `g_t`;
   absorb the tilde values. Each message is generated at the point where the
   real prover sends it, after the challenges that precede it.
3. **Dot stage online.** After the tilde absorption, draw `chi`; compute the
   residual `R = sum_t chi^t (g_t - c_t)` from the fresh mask entries; absorb
   `R`; draw `eps` and then the PSC round challenges; generate each Dot PSC
   round from the hiding polynomial and the public claim target
   `T = sum_t chi^t tilde_t - R = sum_t chi^t (W_t(psi*) + c_t)`; after each
   round message, draw the next challenge. Generate the two end values
   `X_chi(point) + beta_dot*end_0` and `Omega(point) + beta_dot*end_1` after
   `beta_dot`, where `X` is the committed activation/shown-map message.
4. **Closure stage online.** For each position, after the closure roots and
   answers, draw `r_bool`; generate the closure PSC hiding sum and rounds from
   the 49 hiding entries and the product claim; draw each round challenge;
   generate the two end values and the three point values with the five shown
   entries; draw `q`, `r_diff` and `r_recon`; generate the reconstruction rows.
5. **Shared/own FOLD and BASE online.** Generate the fold start after its
   message; draw the fold challenges one round at a time and generate each
   round from the fresh fold coefficients (the round message is generated
   after the previous challenge and before the next one). In the own stage
   generate the public `target*` from the star before `lambda`; draw `lambda`;
   generate the combined rows `star + lambda * row`; draw the base positions;
   generate the star openings. The combined rows satisfy the public linear
   equations by construction, and the star/tail coordinates are fresh.
6. **Opening-time RO programming.** For an opened leaf, the leaf value is the
   one already generated in steps 2-5 from the same message/tail assignment;
   program `H(salt_i || leaf_i) = com_i` and reveal the path. Multiple adaptive
   openings use the same assignment; `q <= r` per tree and the block
   constant-determinant maps keep the opening equations solvable; no message
   or salt is reselected after an opening.
7. **Path consistency.** The opened path is recomputed from the recorded child
   digests; every queried leaf/node/FS point returns the recorded value, and a
   repeated query returns the same answer.

`[FACT]` `MATH` Per-block witness-independence. Conditional on the transcript
prefix and the challenges drawn so far, the public messages of each block are
an affine image of the fresh random coordinates, and the linear part is full
rank on the public rows: the SEND pads; the Vandermonde tail block for
`q <= r`; the closure 49 x 49 hiding map with constant nonzero determinant;
the five shown unit/`beta` pads; the index-mask tail; the `e_t`, `g_t` and
`c_t` classes; the Dot 23 x 23 hiding map and its two end masks; the FOLD
fresh-coefficient map with constant determinant; and the BASE star rows and
tails. The witness `w'` only shifts the image by a vector that lies in the
same image (the block rank conditions of sections 5.2-5.6), so the block's
public-message distribution is the same for every valid `w'`. The adaptive
step is an induction over the real order: a challenge is an RO output of a
prefix whose distribution is already witness-independent, so the next block
starts from the same conditional view. This is the online analogue of the
per-block covers, and it is the sense in which the dummy run and the real run
are claimed to have the same distribution.

`[FACT]` `CODE` `MATH` Block-diagonal closure. The fresh random variables of
the proof partition into blocks with no shared variable:

| Block | Fresh random variables | Public values generated from them |
|---|---|---|
| B0 activation | activation leaf salts (the rows are never opened in the hidden path) | activation root (public statement) |
| B1 W/SEND | W per-position tails, salts, pads | SEND answers, W openings; W roots are RO outputs |
| B2 index-mask/Dot | index-mask entries (23 hiding, 2 end, 16 claim, 16 tilde, 16 eval), its tail/salts/pad; Dot shown-map X tails/salts/pad | Eval targets, tilde, residual `R`, Dot PSC transcript/U, Dot end values, X answers/openings |
| B3 closure | per-position 49 hiding + 5 shown, closure mask tail/salts/pad; map/bit/slack trees, tails, salts, answers | closure PSC transcripts/U, ends/points, closure rows, closure openings |
| B4 shared | Gu/Omega trees; shared stage-0 tails and FOLD fresh `2k+1`; stage switch tails/pad/salts; stage-FOLD seed/salts | shared fold start/rounds, Gu/Omega answers/openings, switch root/answer/openings, shared base rows |
| B5 own | own stage-0 tails and FOLD fresh `2k+1`; stage switch tails/pad/salts; stage-FOLD seed/salts | own fold start/rounds, switch root/answer/openings, own base rows |
| B6 BASE | star main/mask rows and tails; BASE positions | `target*`, combined rows, star openings |

The fresh switch tails, pads and salts are disjoint per switch: the plan
allocates disjoint tree numbers, the stage seeds differ, and the external
ledger rejects reuse. The one deliberate sharing is the previous stage's
main/mask tails, which `switched_rows_multi` places in the new mask message:
those variables belong to the previous block by physical source, and the
switch cover conditions on the previous tree's root/opening and uses the fresh
switch tails as its new random columns. The quantities shared across blocks
are therefore the secret `W`/`Omega`/`Gu`/map/bit values, the old tails at the
switch boundary, and the public challenges.

`[FACT]` `MATH` Conditional block lemma (affine form). Let `X_b` be the fresh
vector of block `b`, uniform on a finite `Fp3`-vector space and independent of
the prior view `V` and the secret `S`. For fixed `(V,S)`, let the block public
values be `P_b = A X_b + c` with `A` linear and `c` fixed by `(V,S)`. Then the
pushforward of the uniform distribution is uniform on the affine image
`A X_b + c`; every point of the image has the same number of preimages
`|ker A|`, and points outside the image have none. This is the precise form:
the claim is not "surjective onto the whole public space and hence uniform";
it is uniform on the affine constraint space actually reached, with equal-size
fibers. If `A` has full row rank the image is the full public space; if not,
the deterministic linear constraints (for example the BASE relation below)
define the image. Conditioning on `V` with any posterior over `S` is a mixture
of the same uniform distribution on the same affine image, so the conditional
distribution is the same for every `S` in the support.

`[FACT]` `CODE` `MATH` Same-source commitment/opening conditioning. A fresh
source is not automatically independent of the prior view after its commitment
or opening is observed. For a tree, let `R` be the root and `O` the opened
symbols at the drawn positions. In the RO model `R` is a uniform RO output
independent of the leaf symbols; conditioning on `R` fixes the programmed leaf
digests but not the unopened symbols. For `q <= r` the opening map from the
tail to the `q` symbols is full row rank; conditioned on `R` and `O`, the
unopened tail coordinates are uniform over the affine fiber of the opening map,
with equal fiber size `|ker|`. The same derivation applies to a mask message
whose entries are opened: the opened entries are fixed, the unopened entries
are uniform over the fiber, and a later public value that reads the same
message is a linear functional of that fiber distribution. This replaces the
earlier unconditional "fresh source independent of the prior view" phrasing:
the conditioning is on the observed root/opening, and the residual distribution
is the uniform-on-fiber distribution.

`[FACT]` `CODE` `MATH` Block covers. For each block the random-to-public map
is affine; the rows below give its affine form, the image (the constraint
space reached), and the equal-fiber statement. The old formulation "surjective
onto the whole public space" is not used:

- **B1.** The answer pads and the tail Vandermonde: for any answer point and
  any distinct opening set with `q <= r`, the pad/tail coordinates solve the
  equations; the maps do not depend on the secret.
- **B2.** With `T = sum_t chi^t tilde_t - R = sum_t chi^t(W_t(psi*) + c_t)`,
  the public rows are the 16 tilde rows `tilde_t = W_t + g_t`, the difference
  row `R = sum_t chi^t(g_t - c_t)`, the 23 PSC sent rows (the invertible
  hiding map on `h` plus the secret terms), the `(U)` row
  `<hiding_form,h> + claim_scale*sum_t chi^t c_t`, and the two ends
  `X_chi(point) + beta_dot*end_0`, `Omega(point) + beta_dot*end_1`. The random
  columns are `g` (16), `c` (16), `h` (23) and `end` (2); the secret columns
  are `W` (16), `X_chi(point)` (1), `Omega(point)` (1). The `g` columns give
  the identity on the tilde rows and cover the W direction; the `c` columns
  give the `chi` combination on `R`/`T` and the claim term in `(U)`; the `h`
  columns are the invertible 23 x 23 PSC hiding map of section 5.4 (constant
  determinant) and cover the PSC secret directions; the `end` columns are the
  unit pivots of `X_chi(point)` and `Omega(point)`. `T` is the public
  combination `sum_t chi^t tilde_t - R`, not an independent row. The only
  challenge condition is `chi != 0`, the counted `chi = 0` event; the
  Vandermonde, identity and unit blocks are challenge-independent. The
  corrected 17-row tilde/difference matrix is exercised by the existing
  `joint_dot_mask_block_is_covered_for_zero_and_nonzero_chi` test (16 tilde
  rows plus the difference row with `chi` on `g` and `-chi` on `c`); the PSC
  hiding map and the ends are covered by the PSC and closure tests.
- **B3.** The per-position joint matrix of section 5.5 (49 hiding + 5 shown
  columns; transcript, ends, points, `(U)` and opening rows) is full rank at
  the production shape (872/872); the hiding map has constant determinant, the
  shown entries are the unit/`beta` pivots, the virtual/difference rows are
  mask-only combinations, and the opening is the tail Vandermonde for
  `q <= r`.
- **B4/B5 (FOLD).** The FOLD sent rows are the invertible fresh-coefficient
  map with constant determinant (section 5.6); the start sum and rounds are
  covered; the shared and own instances are separate.
- **B4/B5 (switch).** The switch message is affine in the old row values, the
  old tails and the fresh switch tails. The old-member opening is the encoded
  old symbol at the drawn positions, the fresh-mask opening is the stage-FOLD
  mask, and the sample answer is one row evaluation with the switch pad. The
  fresh switch tail is the Vandermonde row block for `q <= r`; the old tails
  are fixed (or fiber-uniform) after the previous tree conditioning, so the
  switch opening inherits the previous tree's cover plus the fresh tail. The
  switch target is plan-derived and adds no witness-dependent public value.
- **B6 (affine constraint space).** Write `a = claim`, `b = mask_claim`,
  `M`/`R` for the old main/mask rows, `s`/`t` for the star main/mask rows and
  `lambda` for the challenge. The public values are
  `target* = a*s + b*t`, `main' = s + lambda*M`, `mask' = t + lambda*R`. They
  are not free: substituting gives the deterministic public constraint
  `a*main' + b*mask' - target* = lambda*(a*M + b*R) = lambda*target`, where
  `target = a*M + b*R` is the public BASE claim target. The star values
  therefore lie on the affine graph of this relation, not on the whole public
  space. The map `(s,t) -> (target*, main', mask')` is injective (the combined
  rows recover `s,t`), with one preimage per image point; the image is the
  affine subspace defined by the displayed constraint. The star openings are
  the encoded star plus the star tail; they are affine in the same `(s,t)`,
  and their opening equations are included in the same image. For a fixed old
  row `M,R` the image is shifted by `lambda*(M,R)` but its direction and fiber
  size do not depend on the old row; the uniform star distribution pushes to
  the uniform distribution on that affine image. The right-hand side
  `lambda*target` is already-approved public data, so the constraint does not
  add a witness-dependent public value.

For each block, the listed random columns enter affinely; the image is the
affine space spanned by those columns plus the fixed secret terms, and the
fiber size is `|ker A|` for that block's matrix. The pivots, constant
determinants and Vandermonde full-row-rank statements above make the image
full-dimensional on the stated public rows; where the image is smaller (the B6
constraint), the displayed affine relation is the exact image, not an
approximation. This is the precise form required for the conditional block
lemma.

`[FACT]` `MATH` Online order and adaptive openings. The block covers are
challenge-independent (the only challenge condition is `chi != 0`, counted;
the Vandermonde blocks hold for any distinct positions; the determinants are
constant). Each block can therefore be generated online after the challenges
that precede it, and the conditional block lemma applies inductively over the
real order. For adaptive openings the ledger ensures `q <= r` per tree; the
tail Vandermonde `q x r` matrix has full row rank for any distinct position
set, so the affine image and its fiber size are unchanged for every opening
sequence; the simulator never reselects a message or salt. Hash hybrid: the
public activation/table roots are fixed setup inputs, not witness-dependent,
and are not programmed; only the prover's commitment trees are simulated by
RO lazy sampling with a consistent history, programming unqueried inputs and
charging the failure events `q * 2^-320` and `(q + 1)^2 / 2^256`.

`[FACT]` `CODE` Hidden definition and conclusion. The property the block
construction establishes is the distributional assertion
`view ⊥ witness | statement + public inputs`: for any two valid witnesses of
the same public statement, the transcript distribution is the same, with the
per-block affine images and equal fibers above. This is not a simulation-based
definition and it is not a witness-free simulator. If the public statement has
a unique valid witness, the "any two witnesses" assertion is vacuous; the
meaningful definition is simulation-based, and the present construction does
not meet it. A witness-free simulator would have to generate an accepting
transcript without a valid witness (for example the withdrawn offline
preselection with the full joint cover, or direct programming of the
algebraic checks); that is not provided. The B0-B6 construction is therefore a
simulator only in the weaker sense: it takes the statement, the public inputs
and a valid dummy witness and produces the view. The missing piece for the
simulation-based definition is exactly the witness-free generation step. The
per-block rank conditions (constant determinants, `q <= r`, `chi != 0`), the
ledger budget and the RO/CR model remain the conditions of this
witness-independence statement; no counterexample to the block partition or to
an affine block cover was found.

`[FACT]` `CODE` Root programming history. The roots split by setup history:
the public activation/table roots are fixed setup inputs, present before the
proof and not witness-dependent; they are used as given and are **not**
programmed. The prover's commitment roots (W, closure, mask, Dot-X, FOLD fresh,
BASE star) are generated during the proof. For those, the RO simulator uses
lazy sampling with a consistent history, programs only unqueried inputs, and
charges the failure events `q * 2^-320` (pre-opening guess) and
`(q + 1)^2 / 2^256` (collision/programming). The earlier sentence that
programmed a public root to a fixed value is withdrawn: a public root is an
input, not a programming target; programming applies only to prover
commitments in the simulator's own message stream.

`[FACT]` `MATH` Bad events. The simulation can fail only through:

- **Guess before opening.** A query made before a leaf is opened hits the
  exact input `salt_i || leaf_i`. Conditional on the view, the input has
  min-entropy at least `H_salt + H_leaf`, where the salt contributes 320 bits
  (chosen once at step 1 and hidden until step 6) and `H_leaf` is the
  conditional min-entropy supplied by the algebraic tail/pad/mask/star
  randomness of steps 2-5 (nonnegative; the pads and tails are the mask sources
  of sections 2-5). Over `q` cumulative queries the probability is at most
  `q * 2^{-(320 + H_leaf)} <= q * 2^-320`. The count `q` is cumulative over
  leaf, node, root, challenge and opening queries.
- **Collision or programming inconsistency.** Two distinct inputs receive the
  same random-oracle output, or a programmed value conflicts with an earlier
  answer: probability at most `(q + 1)^2 / 2^256`.

If neither event occurs, the generated view is identical to the real view for
every adaptive opening sequence, including repeated queries to
already-programmed points. The challenges are RO outputs independent of the
unopened leaves; no separate "conditioning on the root" step is used.

The mapping to the construction is literal: the leaf hash input is
`(tag, leaf_index, run, width, salt_length, symbols, salt)`, the node hash
input is `(tag, tree_number, configuration, left, right)`, the salt is 40
bytes, and the index/run/plan bind the leaf position. The argument uses RO for
hiding and CR for binding; it does not follow from CR alone. The concrete
hash's RO/CR advantage is not quantified, so the statement remains
model-conditional. `[ASSUMPTION]` The RO/CR model is not quantified against a
concrete hash. The hybrid failure probability is therefore a model-level
condition, not a numerical term; the composition expression in section 6
lists the algebraic terms and states the hash-model term separately.

### 5.8 Whole-chain conditional view

`[FACT]` `CODE` The whole-chain argument is the following stepwise conditional
distribution, not an extrapolation from the sub-block ranks:

**Sources and public positions.** The per-proof mask sources are the
`ByteMaskCoins` stream for the index-mask message (73 entries), the closure
mask message (16 positions of 49+5 entries), the W/Gu/Omega/closure tails and
salts, the SEND pads, the FOLD fresh coefficients and the BASE star rows and
tails. Their public positions are the roots and sampled answers, the drawn
openings, the tilde values and residual, the PSC transcripts and end values,
the closure point values, the FOLD start/rounds and the BASE target/combined
rows/openings. Each source is identified by its own stream/run seed; no stream
is reused across oracles.

**Conditional rounds.** Condition on the view produced so far and draw the
next challenge only after the message it tests:

1. after the roots/declarations, the shared seeds are drawn; the SEND answer
   rows are one-time pads on the sampled values;
2. after the answers, the W/mask positions are drawn; the openings are
   Vandermonde images masked by the tree tails;
3. after the closure roots/answers, `r_bool` is drawn; the closure PSC
   transcripts are the invertible hiding-map images plus the message vector,
   and the two end values are one-time padded;
4. after the map answers, the map positions are drawn;
5. after the index-mask answers, the mask positions are drawn;
6. after all openings, `psi` is drawn; the Eval targets are `W(psi)+e` with
   `e` fixed before `psi`;
7. after the Eval absorption, the tilde values and residual are public; the
   shared `g_t`/`c_t` columns are explicit in the mask block;
8. after the residual, the PSC challenges are drawn; the transcript rows are
   the invertible hiding-map images;
9. after the fold start, the fold challenges are drawn; the fold rows are the
   invertible fresh-coefficient images;
10. after the star commitment and target, `lambda` is drawn, then the combined
    rows are sent, then the base positions are drawn; the BASE rows are
    one-time pads `star + lambda * row`;
11. the seal.

At each step, the new public values are either one-time pads with fresh
uniform randomness or images of an invertible hiding map; the next challenge
is drawn afterwards; the conditional distribution of the secret given the view
so far is unchanged. The shared quantities (`g_t` in tilde/R, `c_t` in
R/Dot-sigma, the mask entry in the opening/PSC/point rows, the star in the
BASE target/rows/openings) are explicit columns in the corresponding joint
blocks; they are not treated as independent.

**Assembly.** The full view is the concatenation of these blocks. The block
structure is the partition of section 5.7; the blocks are not inferred to be
independent merely from the sub-block ranks. The composition is proved there
by the block partition (disjoint fresh variables) and the conditional block
lemma, applied inductively in the challenge order. The production-size closure
joint test (section 5.5) is a local numerical confirmation of one block; a
complete multi-oracle matrix extraction is not claimed and is not needed for
the block-diagonal proof.

**BASE elimination.** The equations are

```text
target* = <claim, star.main> + <mask_claim, star.mask>
main'   = star.main + lambda * main
mask'   = star.mask + lambda * mask
tail'   = star.tail + lambda * tail
```

The verifier's linear check is
`<claim, main'> + <mask_claim, mask'> = target* + lambda * target`. Substituting
the equations cancels the star and leaves

```text
<claim, main> + <mask_claim, mask> = target.
```

The left side is the public BASE claim relation; the right side is the claim
target that the verifier knows from the committed rows/statement. The hidden
row `Y` enters the committed rows, not the eliminated relation as a free
value. A claim value that depends on `Y` is never approved merely because it is
a claim: the approved output boundary is winner/valid/tie plus the public
statement, and the target inventory of section 5.9 shows that every BAT target
is either a public constant/deterministic public value or an already-published
transcript value whose dependence on `Y` is one-time masked. The combined rows
and the target alone leave the star underdetermined; the openings reveal
encoded star symbols masked by the star tail (`q <= r`) and the old row
symbols masked by the old tail. The star is a fresh one-time pad from its own
stream; the two-seed test is illustrative, not the uniformity proof.

`[FACT]` The multi-oracle conditional composition is supplied by the block
partition and conditional block lemma of section 5.7. The per-block ranks are
`CODE` derivations with `NUM` tests; a mechanized symbolic proof of the
partition and rank conditions is not delivered, so the statement is
model/derivation-conditional, not machine-checked. The production-size tests
remain local evidence for their blocks.

### 5.9 BAT target inventory and the joint mask/star view

`[FACT]` `CODE` The approved output boundary is winner/valid/tie plus the
public statement; it is not expanded to arbitrary functions of the hidden
row. The rows below are the complete target inventory of the three plain-row
families actually emitted in the hidden path (`closure_plain_rows`,
`dot_plain_rows`, `own_plain_rows`), followed by the shared/base rows. For
every row the table gives the exact target, the public coefficients, the
secret functional that the target is a masked version of, the one-time mask
entry, and the other public positions where the same value appears.

**Closure plain rows (one block of 11 rows per position `p`).** The mask
entries `shown0..shown4` are the five shown slots of position `p` in the
closure mask message; `hiding0..hiding48` are its 49 hiding slots. The
coefficients `eq_point`, `eq_q`, `eq_r`, `eq_diff`, `eq_recon` and
`hiding_form` are recomputed by the verifier from the public challenges; the
`mask_scale` of the two end claims is the nonzero last PSC challenge `beta`.

| Row | Target `row.target` | Row coefficients (all public) | Secret functional in the target | Pad (one-time mask entry) | Other public occurrence |
|---|---|---|---|---|---|
| slack/winner | `-(M*sum_{j>=valid} eq_diff_j + sum_{j<winner} eq_diff_j)` (`M = mask_value`) | `eq_diff` on the slack message; `eq_diff - c_sum*e_winner` on `W` | none: public plan/statement plus the public `r_diff` challenge | none | none; the value is a public constant of `winner`, `valid`, `mask_value`, `r_diff` |
| recon `u` | `offset` (public plan constant) | `weight*eq_recon` on the packed bits at `u_off`; `-eq_recon` on `W` | none in the target; the row binds the bit packing | none | none |
| recon `v` | `offset` | `eq_recon` powers at `v_off`; `+eq_recon` on `W` | none in the target | none | none |
| recon slack | `0` | `eq_recon` powers at `s_off`; `-eq_recon` on the slack message | none in the target | none | none |
| `(E_0)` | `ends[0].value` of the closure PSC | `eq_point` on the committed map message | `map(point)` (committed closure map at the PSC point) | `shown0`, coefficient `beta` | `DotProof.end_values[0]`, absorbed by `send_end_values`; also used to derive `row_target` |
| `B(point)` | `b_point` | `eq_point` on the committed bit message | `B(point)` (multilinear bit map at the PSC point) | `shown2`, coefficient `1` | `point_values[0]`, absorbed by `absorb_closure_point`; also in the virtual target below |
| virtual second map | `ends[1].value - 1 + b_point` | `shown2` with `1` and `shown1` with `beta`; no main term | none: since `Omega = 1 - B`, the target equals `beta*shown1 + shown2`; the row is mask-only | `shown1` (the ends[1] pad) and `shown2` (the B(point) pad) | derived from the two absorbed values; no new wire value |
| `(U)` | `row_target` (the PSC last round value minus `eps*ends[0].value*ends[1].value`) | `hiding_form[j]` on `hiding_j`, `j = 0..48` | none in the row: it is the mask-side linear form; the hidden map enters through the PSC transcript, whose hiding map is invertible (5.4) | `hiding_j` (the same entries appear in the PSC transcript as shared columns) | `row_target` is recomputed by the verifier from the absorbed round messages and end values; no separate wire value |
| `D(q)` | `d_value` | `eq_q` on the committed map message | `D(q) = <eq_q, map message>` | `shown3`, coefficient `1` | `point_values[1]`, absorbed by `absorb_closure_point`; also in the difference target |
| `B(q)` | `bq_value` | `eq_q[j]*eq_r[j]` on the committed bit message | `Bq = <eq_q*eq_r, bit message>` | `shown4`, coefficient `1` | `point_values[2]`, absorbed by `absorb_closure_point`; also in the difference target |
| `D(q)-B(q)` | `d_value - bq_value` | `shown3` with `1`, `shown4` with `-1`; no main term | none: mask-only difference of the two rows above | `shown3`, `shown4` (shared) | derived from the two absorbed values; no new wire value |

**Dot plain rows (4 rows per proof).** `chi` is the public batching challenge;
`tilde_t` and `claim_t` are the two disjoint per-position mask classes of the
index-mask message; `hiding_j` is its PSC hiding class; `end_0`, `end_1` are
its two end slots. `eq_index(point, value_at)` is the public equality-weight
coefficient; `beta_dot` is the nonzero last Dot PSC challenge.

| Row | Target `row.target` | Row coefficients (all public) | Secret functional in the target | Pad (one-time mask entry) | Other public occurrence |
|---|---|---|---|---|---|
| difference | `R = sum_t chi^t (g_t - c_t)` | `chi^t` on `g_t` (tilde class), `-chi^t` on `c_t` (claim class), `t = 0..positions-1` | none in the row: `g_t` and `c_t` are mask entries; the row checks `sum_t chi^t g_t - sum_t chi^t c_t = R` | `g_t`, `c_t` (both one-time) | `R` absorbed by `absorb_dot_residual`; `tilde_t = W_t(psi*)+g_t` is public; the adjusted PSC target `T = sum_t chi^t tilde_t - R = sum_t chi^t (W_t(psi*) + c_t)` is public and carries the W direction masked by `c_t` |
| `(U)` | `row_target` (the PSC last round value minus `eps*ends[0].value*ends[1].value`) | `hiding_form[j]` on `h_j`; `claim_scale*chi^t` on `c_t` | none in the row: it is the PSC output row `<hiding_form,h> + claim_scale*sigma = row_target` with `sigma = sum_t chi^t c_t`; the PSC transcript proves `sum_t chi^t <X_t,Omega> = sum_t chi^t W_t(psi*)` | `h_j`, `c_t` (the same entries appear in the Dot PSC transcript/openings) | `row_target` is recomputed by the verifier from the absorbed Dot PSC rounds and end values; no separate wire value |
| `(E_0)` | `ends[0].value` of the Dot PSC | `chi^t*eq_index(point, value_at)` on the committed activation/shown-map message `X` | `X_chi(point) = sum_t chi^t X_t(point)` (the chi-batched activation/shown-map X at the point), not a W evaluation | `end_0`, coefficient `beta_dot` | `DotProof.end_values[0]`, absorbed by `send_end_values` |
| `(E_1)` | `ends[1].value` of the Dot PSC | `eq_index(point, value_at)` on the Omega message entry `value_at` | `Omega(point)` (one committed Omega value; no chi batching) | `end_1`, coefficient `beta_dot` | `DotProof.end_values[1]`, absorbed by `send_end_values` |

`[FACT]` `CODE` The Dot equations in one place. With
`tilde_t = W_t(psi*) + g_t` and `R = sum_t chi^t (g_t - c_t)`, the public PSC
target is

```text
T = sum_t chi^t tilde_t - R = sum_t chi^t (W_t(psi*) + c_t).
```

The PSC proves `sum_t chi^t <X_t, Omega> = sum_t chi^t W_t(psi*)`: the claim
masks `c_t` cancel the W term in `T` against `sigma = sum_t chi^t c_t`, and the
difference row is the mask-entry relation
`sum_t chi^t g_t - sum_t chi^t c_t = R`. The W direction is therefore on the
PSC-target side (masked by `c_t`, with `g_t` in `tilde_t`), while `X` and
`Omega` are the two committed maps on the `(E_j)` side. The earlier reading
`sum_t chi^t c_t = sum_t chi^t tilde_t - R` is withdrawn: it replaced the
public `tilde_t` by its mask entry and dropped the W direction.

**Own plain rows (2 rows per position; the diagnostic rows are absent in the
hidden path).** `eval_t` and `tilde_t` are the eval and tilde mask classes of
the index-mask message; `psi` is the Eval point and `psi*` the shared point.

| Row | Target `row.target` | Row coefficients (all public) | Secret functional in the target | Pad (one-time mask entry) | Other public occurrence |
|---|---|---|---|---|---|
| Eval | `eval_targets[t]` | `eq(psi)` on the `W_t` message | `W_t(psi)` | `eval_t`, coefficient `1` | the serialized/absorbed Eval target itself; `eval_t` not separately published |
| tilde | `tilde_t` | `eq(psi*)` on the `W_t` message | `W_t(psi*)` | `tilde_t`, coefficient `1` | the absorbed tilde value; `tilde_t` also in the difference row |
| diagnostic Dot (only when `plan.weights` is non-empty) | `0` | public activation weights on Omega; `-eq(psi*)` on `W_t` | none | none | not emitted in the hidden-input path; excluded from this inventory |

**Shared and base rows.** `record_row`: the target is the public table record
weighted by the public `eq_gamma`; the witness is the message row, and no
secret enters the target. `column_row`: the target is the public column value
weighted by `eq_gamma`. `sample_row`/`oracle_sample_row`: the target is the
received SEND answer, i.e. the sampled message value plus the reserved SEND
pad; the pad is one-time per answer, the answer is serialized in `SendSamples`
and absorbed. `fold_plain_row`: the target is the public `G_u`/`Omega`
relation target with the public `psi`/`zeta` challenges; no hidden row enters
it. The BASE `handoff` target is supplied by the caller's verified FOLD output;
the caller/anchor trust is a deployment assumption, not part of this
derivation.

`[FACT]` `CODE` Every target in the three tables is a member of the public
view already present before the BAT rows: it is either a public plan/statement
constant or a value that the verifier itself reconstructs from the absorbed or
serialized transcript (the verifier calls the same row functions in
`verify_shared`/`verify_second`, so the target is not an extra wire value).
The only new data in a BAT row are its public coefficients, which are
recomputed from the challenges. Consequently the joint conditional
distribution of the BAT targets given the public view is degenerate: it adds
no public position and no secret coordinate.

`[FACT]` `CODE` `MATH` Joint conditional view of the targets and their masks
(per block). The target tables add no new wire value: every target is a public
plan/statement constant or a value the verifier reconstructs from the
absorbed/serialized transcript. The secret directions visible in the public
view are covered blockwise:

- closure block: the five shown columns are the pivots `shown0 -> ends[0].value`
  (coefficient `beta`), `shown1 -> ends[1].value` (`beta`),
  `shown2 -> b_point` (1), `shown3 -> d_value` (1),
  `shown4 -> bq_value` (1). The virtual row target
  `ends[1].value - 1 + b_point` is exactly `beta*shown1 + shown2`, and the
  difference row target `d_value - bq_value` is exactly `shown3 - shown4`;
  both are mask-only combinations of the pivoted rows. The `(U)` row is the
  PSC output row over the 49 hiding columns; the hiding map to the transcript
  has the constant nonzero determinant of 5.4; the encoded opening uses the
  tail Vandermonde for `q <= r`.
- dot block: the difference row is the mask-entry relation
  `sum_t chi^t (g_t - c_t) = R`; the public `tilde_t = W_t(psi*) + g_t` and the
  PSC target `T = sum_t chi^t tilde_t - R = sum_t chi^t (W_t(psi*) + c_t)`
  expose the W direction only through the masks `g_t` and `c_t`; the same
  `c_t` appears in `T`, in `sigma = sum_t chi^t c_t` (the `(U)` row claim term)
  and in the opening; the same `g_t` appears in `tilde_t`, `R` and the
  opening; `h_j` covers the PSC transcript and the `(U)` row; `end_0/end_1`
  are the unit/`beta_dot` pads of `X_chi(point)` and `Omega(point)`; the
  `(E_0)/(E_1)` rows are over the committed shown-map/Omega messages, not over
  W.
- own block: `eval_t` pads `W_t(psi)`, `tilde_t` pads `W_t(psi*)`; the same
  `tilde_t` is in the difference row and the opening; the W messages
  themselves are covered by their tails/openings.
- BASE: `target*` is one-time in the star; the same star is in the combined
  rows and the openings; the star block is separate from the mask classes.

For each block, `[A | B]` with the fresh random columns `A` and the secret
columns `B` has `Im(B) subset Im(A)` on the rows above (the pivots are
unit/`beta`/invertible), so the block's conditional distribution hides the
secret directions it touches.

`[FACT]` `CODE` Full joint composition. The per-block pivots above are not by
themselves a direct sum; the composition is supplied by the block partition
and the conditional block lemma of section 5.7. The fresh random variables
partition into blocks B0-B6; every public value uses fresh variables from
exactly one block; the shared `c_t`/`g_t` and W directions are within B2 or
are secret coupling handled by conditioning, not a shared random variable; and
the roots/star are RO/public values. Applying the conditional block lemma
inductively in the real challenge order gives the full joint distribution.
The earlier "full joint not assembled" status is superseded by this
block-diagonal proof; the scope condition (the dummy witness must satisfy the
same public statement; this is not a witness-free simulator) remains. The
joint tests are the per-block evidence; the block partition is the
code/ledger allocation fact. No counterexample to the partition or to a block
cover was found.

`[FACT]` `CODE` The BASE equations are
`target* = <claim, star.main> + <mask_claim, star.mask>`,
`main' = star.main + lambda * main`, `mask' = star.mask + lambda * mask`.
The claims and the target are public; the star is fresh; the hidden row
enters `main`/`mask`. The verifier's linear check cancels the star
(section 5.6/5.8) and leaves the public claim relation. The target* is a
masked public value; it has no secret direction to protect.

`[FACT]` `CODE` Joint mask-entry view. Condition on the public view before
the mask message. The 49 hiding and five shown entries are fresh uniform
draws. The encoded opening, the PSC transcript and the end/point values are
affine in the same entries and the tail; the block matrix of section 5.5 has
the hiding map as an invertible block, the shown identity on the end/point
rows, and the tail Vandermonde on the opening rows. Conditional on the view so
far, the joint distribution of these three public positions contains no
secret direction; the same entry appearing three times is a shared column in
one matrix, not three independent one-time pads.

`[FACT]` `CODE` Joint star view. Condition on the public view before the
star. The star main/mask rows and tails are fresh uniform draws. The target*,
the combined rows and the star openings are affine in the same star; the
target is public, the combined rows are `star + lambda * row`, and the
openings are the encoded star. The BASE elimination (section 5.6) cancels the
star and leaves the public claim relation; the combined rows and target alone
leave the star underdetermined, and the openings are masked by the star tail
(`q <= r`) and the old row tail. The three appearances are one shared random
vector, not three independent pads. The two-seed test is illustrative; the
uniformity is the `draw_star` generation relation.

## 6. Exception probabilities

`[FACT]` `CODE` The base field is `p = 2^64 - 2^32 + 1`; the cubic extension
has `p^3` elements with `log2(p^3) = 191.99999999899`, so `1/p^3` is slightly
larger than `2^-192` and `2/p^3` is slightly larger than `2^-191`. Every
strict bound below is exact or looser than the exact value:
`1/p^3 < 2^-191`, `2/p^3 < 2^-190.9`, `3/p^3 < 2^-190.4`,
`4/p^3 < 2^-189.9`, `6/p^3 < 2^-189.4`, `26/p^3 < 2^-187.2`,
`28/p^3 < 2^-187.1`, `30/p^3 < 2^-186.9`, `32/p^3 < 2^-186.9`,
`58/p^3 < 2^-186.1`, `1134/p^3 < 2^-181`, `16*166/p^3 < 2^-180`. The exact
quantities used in the self-check are
`1/p^3 = 2^-191.99999999899`, `2/p^3 = 2^-190.99999999899`,
`3/p^3 = 2^-190.4150374983`, `4/p^3 = 2^-189.99999999899`,
`6/p^3 = 2^-189.4150374983`, `26/p^3 = 2^-187.299560...`,
`28/p^3 = 2^-187.192645...`, `30/p^3 = 2^-187.0931094034`,
`32/p^3 = 2^-186.999999...`, `58/p^3 = 2^-186.142019...`,
`128/p^3 = 2^-184.99999999899`,
`1134/p^3 = 2^-181.852...` and `16*166/p^3 = 2^-180.625...`. In particular
`128/p^3` is slightly larger than `2^-185`, so no strict `< 2^-185` claim is
made for it; it is not used in the bound. The `3/p^3`, `4/p^3`, `6/p^3` and
`30/p^3`, `32/p^3` and `58/p^3` entries are listed for the arithmetic
self-check; the retained single-stage union terms are the FOLD `32/p^3`, the
range, the PSC and the `chi = 0` terms; the multi-stage loop's candidate FOLD
merge is `58/p^3`.

The hiding-rank maps have constant nonzero determinant (section 5.4), so the
former bad-challenge rank term is withdrawn: there is no Schwartz-Zippel
event for the PSC or FOLD hiding maps. The remaining degree terms are
soundness checks of the sumcheck and range relations, not hiding-rank terms.
The entries are:

| Component | Count/degree | Bound |
|---|---|---:|
| direct shared FOLD check | k = 2, 2 rounds, degree 2 | `4/p^3 < 2^-189.9` |
| direct own FOLD check | k = 14, 14 rounds, degree 2 | `28/p^3 < 2^-187.1` |
| BASE lambda and symbol checks | withdrawn from the algebraic union: no independent-uniform field-domain/order derivation for the Fp3 lambda challenge, and the two checks are codeword-index samples, not field challenges | carried in `eps_proximity` (unknown term, not in `eps_components`) |
| own base position sampling | interval planner: stage0 252 draws with 24 work bits, last 259 draws; shared join fixed at 2 and covered by the table column test | counted in the interval round table; the old two-position miss is withdrawn as the own-chain term |
| range booleanity | 16 positions, degree at most 166 each | `16*166/p^3 < 2^-180` |
| PSC sumcheck (Dot once, closure 16 positions) | 10 and 23 variables, degree 3 each | `(30 + 16*69)/p^3 < 2^-181` |
| single equality | the check exists, but its polynomial-degree/event derivation is not written in this note | not counted in `eps_components`; carried in `eps_proximity` (unknown term) |
| `chi = 0` | one event: a uniform `Fp3` challenge equals zero | `1/p^3 < 2^-191` |
| challenge-draw zero fallback | at most 6 rejected words per draw, `(1-p/2^64)^6` | below `2^-192` per draw |
| mask-coordinate retry exhaustion | at most 64 attempts, `(1-p/2^64)^64` | below `N * 2^-2048` over `N` coordinates |
| hash model | plan-ledger term | `(t + q_V + 1) * 2^-128 + (17/2) * (t + q_V)^2 / 2^256` |

`[ARGUMENT]` `CODE` The component union bound used here is

```text
eps_total <= eps_hash + eps_components + eps_proximity
eps_hash = (t + q_V + 1) * 2^-128 + (17/2) * (t + q_V)^2 / 2^256
eps_components <= 16*166/p^3 + 1134/p^3 + 32/p^3 + 1/p^3
                < 2^-180 + 2^-181.85 + 2^-186.9 + 2^-191
                < 2^-179.6
eps_proximity = the interval round terms: own stage0 252/24 and last 259,
                 shared join backed by the table columns at 603 draws with 12
                 work bits; the counted table is in the rework18 evidence
                 (worst round 2^-128.000035, naive union 2^-123.69)
eps_total also carries the retry exhaustion N * 2^-2048 and the Q_R / Q_A factors
```

`eps_proximity` is no longer the direct two-position FOLD-to-BASE miss for
the own chain. The interval planner fixes own stage0 at 252 positions with
24 work bits and the last stage at 259; the counted own FOLD, BASE and query
terms are in the rework18 interval table (worst single round
`2^-128.000035`; naive union sum `2^-123.69`). The shared group keeps two
positions, but its proximity is supplied by the table column test at 603
columns with 12 work bits (`2^-128.164982`); the two-position miss
(`0.3163976` main, `3.42e-5` mask) is retained only as the historical
single-stage observation and is not the own-chain bound. This note does not
claim a total security bit count; the union sum does not meet 128 bits and
is reported as such.

The `BASE 3/p^3` figure and the earlier `6/p^3 < 2^-189.4` merge are withdrawn
from `eps_components`: the lambda challenge is an `Fp3` field value whose
independent-uniform-domain/order derivation is not given here, the two symbol
checks are codeword-index samples of the real sampler, and two `2/p^3`
single-equality units lack a written degree/event derivation. The counted
terms are the FOLD merge `4/p^3 + 28/p^3 = 32/p^3 < 2^-186.9`, the range
term, the PSC term and the `chi = 0` event; each has an explicit degree or
event basis (FOLD round checks of degree 2, PSC round checks of degree 3,
range checks of degree at most 166 per position, and one
uniform-zero-challenge event). No term without a derivation is kept in the
union.

The `t`/`q_V` factor is carried by the hash term; the algebraic component
count is fixed per proof. If the prover uses `Q_A` adaptive retry attempts,
multiply the component terms by `Q_A`; the hash term already carries the
transcript query count. For `Q_R` independent requests, multiply the
component sum by `Q_R`; no global bound is claimed without a bound on `Q_R`.
The challenge draws are uniform on the cubic field except for the zero
fallback term above. The hash term assumes the plan's RO/CR model and work
bits; it is not re-derived here.

`[FACT]` `CODE` The local compiled plan's actual parameters are: table
distance radius 16,384, later radius 16,384, `m_once = m_later = 0`, `v = 10`,
`answers = 1`; column draws 603 with a 603 column budget and 12 column work
bits; shared/base positions 2 each. The direct FOLD checks have k = 2
(shared) and k = 14 (own). The plan does draw one `Fp3` BASE lambda and two
base positions, but their numeric error terms are not kept in
`eps_components`: they are carried in `eps_proximity`, which is ineffective at
the current parameters (section 8). The prior "8 rounds each <= 2^-128" term
is a multi-stage ledger figure that is not mapped to this direct FOLD-to-BASE
compiled plan; it is withdrawn from this chain's bound and replaced by the
per-round degree terms above.

`[FACT]` `CODE` The real base-position sampler is `draw -> unique_sorted`: the
plan's `plan_loc` convention draws `t = 2` codeword-index positions,
deduplicates them, and checks the resulting `s <= 2` positions; `s` is
variable (it is 2 unless the two draws collide). The soundness event is
whether the *original* draws hit the disagreement set; deduplication does not
change that event. If a draw lands on a disagreement position, that position is
among the unique positions and the check sees it; if both original draws land
in the agreement set, every unique position agrees. Thus the miss event for the
original sampling is "both draws agree", with probability `(1 - delta)^t` for a
word at relative distance `delta` (agreement `1 - delta`); the hit probability
is `1 - (1 - delta)^t`. This is not an argument from "the number of distinct
positions decreased": it is the same original-draw event. The actual
parameters give `n = 131,072`, `delta_main = 57,345/131,072 =
0.4375076294` and `delta_mask = 130,305/131,072 = 0.9941482544`; the
corresponding misses are `0.3163976670` and `3.424292663e-5` (section 8). The
base-position and proximity soundness is therefore **ineffective at the
current parameters** (the main code dominates), not zero and not a
conservative placeholder. The `Fp3` BASE lambda challenge and the two
codeword-index checks are likewise carried in `eps_proximity` (withdrawal
above), not in `eps_components`. The retry exhaustion (`N * 2^-2048`) and the
multi-request `Q_R`/adaptive `Q_A` factors are carried separately; they are
not dropped.

`[FACT]` `CODE` The numerical proximity miss at the actual code parameters is
computed above: `0.3163976670` (main) and `3.424292663e-5` (mask). `[UNPROVEN]`
The exact composition of this miss with the algebraic FOLD detection for each
error class (which differences are kernel-only and which are seen by the fold
polynomials) is not mechanized; the worst-case single-row error is the basis
of the ineffective-proximity conclusion.

`[UNPROVEN]` The `126.21` and `116.01` bit figures remain plan-ledger numbers,
not a new closed-chain theorem. The degrees and counts above are arithmetic
assumptions; a conservative union bound over the components is not mechanized.

`[FACT]` `CODE` The mask-coordinate rejection sampling is exact in the
ideal-word model (section 2.1); the only sampling term is the retry-exhaustion
abort probability above. The real finite-seed statement is computational under
the PRG/RO model for `hash_labeled`; it is not an unconditional statistical
independence claim. The table's rank and degree terms do not change under the
rejection sampling because the sampling only changes the distribution of the
mask values, not the algebraic checks.

## 7. Assumption support table

| Assumption | Support | Status |
|---|---|---|
| independent uniform `Fp`/`Fp3` draws | `ByteMaskCoins::value_at` rejection sampling and `draw_f/draw_e`; three `Fp` coordinates per `Fp3` | supported in the ideal-word model; real seed is PRG/RO-conditional (section 2.1) |
| challenge zero fallback bounded | `DRAW_TRIES = 6`, rejection sampling in `draw_f` | supported; term in section 6 |
| opening positions distinct, `q <= r` | transcript/ledger; actual counts 603/4/4; ledger min remaining 252 | supported by the ledger; the numerical rank check is `NUM` |
| tail union across SEND and BASE | the tree opening set is the union; ledger evidence | supported by the ledger |
| PSC hiding map constant nonzero determinant | triangular derivation from `HidingPolynomial`/`hp_place`; determinant `+1` at k=10 and `-1` at k=23 | `CODE` `NUM`; no SZ term |
| FOLD fresh map constant nonzero determinant | triangular derivation from `FoldRoundProver`; determinant `+1` at k=2 and k=14 | `CODE` `NUM`; no SZ term |
| shown end masks one-time, beta nonzero | `draw_e_nonzero`; real-prover end-value test | supported |
| closure joint mask object rank equality | real encoder + real hiding polynomial + joint matrix | `NUM` at k=4; `CODE` block derivation for k=23 |
| full joint composition of all blocks, shared columns and roots | block partition B0-B6 + conditional block lemma, per-block rank tests | `FACT` `CODE` `MATH` (section 5.7); the scope condition is witness-independence, not a witness-free simulator |
| BASE star fresh and one-time | `draw_star` from a fresh stream; real-path test | supported |
| local proximity distance/query soundness | `n = 131,072`; main `ell + r = 73,728`, mask `ell + r = 768`; two base positions | `FACT`: main relative distance `0.4375076294`, miss `0.3163976670`; mask miss `3.424292663e-5`; the main code makes the proximity ineffective at the current parameters (section 8) |
| BASE lambda and symbol-check terms | `Fp3` lambda challenge and two codeword-index samples; no independent-uniform-domain/order derivation here | withdrawn to `eps_proximity` (ineffective proximity; unknown lambda term), not in `eps_components` |
| single-equality term | the check is present; its polynomial-degree/event derivation is not written here | not counted; carried in `eps_proximity` (unknown term) |
| hash RO for hiding, CR for binding | leaf/root hash construction and salt lengths; model statement | supported as a model; no concrete-hash quantification |
| no mask/pad/stream/table reuse | ledger state validation and rejects | supported; executable |
| approved output boundary | policy/plan; the full `Y` row is hidden beyond winner/valid/tie | `FACT` boundary; per-block hiding and the block-diagonal composition of section 5.7 |
| closed path has two FOLD instances and one closure PSC per position | `member_fold_shape` call sites; `messages.psc.len() == positions` | `CODE`; actual k=14 own fold |
| no code-switch stage in the hidden path | plan rejects a switch configuration | supported |
| asset inventory only for this fixture | the stated shapes and code parameters | scoped, not extrapolated |
| concrete-hash PRG/RO indistinguishability | the ideal-word sampling proof plus the hash model | `UNPROVEN` model-conditional; no concrete-hash bound |

## 8. Soundness reduction, rejection chain and tree-reuse tamper

`[FACT]` `CODE` Acceptance conditions of the public verifier. The verifier
accepts only if all of the following hold; each is a deterministic binding, an
algebraic polynomial identity in the drawn challenges, or a sampled proximity
check:

1. **Plan/statement/reader binding.** Plan validation; reader
   configuration/shape/statement/roots; the activation and statement roots; no
   reuse/rollback. A mismatch rejects; the binding is deterministic (with the
   hash/CR events in the hash model).
2. **Transcript replay and seals.** The three chains absorb the declared
   messages in the fixed order; the seals equal the message seals; the first
   query and the second-stage chain start agree. Deterministic.
3. **SEND sampling.** Seed-to-cube, answer count/order, reserved pads; the
   answers are bound by the sample rows to the committed messages.
4. **Opening checks.** Runtime tree paths against the roots, leaf
   index/run/plan, distinct positions, `q <= r`. Deterministic under CR; a
   failure rejects (hash event in the model).
5. **Algebraic rows.** Eval `W_t(psi)+e_t`; tilde `W_t(psi*)+g_t`; residual
   `R = sum_t chi^t(g_t-c_t)`; the closure/dot/own BAT rows; the Dot and the
   16 closure PSC transcripts (degrees 3); the shared/own FOLD checks (degree
   2 per round); the range/booleanity checks (degree at most 166 per
   position); and the BASE `target*`/combined-row/lambda relations. After the
   committed messages and targets are fixed, these are polynomial identities
   in the challenges.
6. **Proximity.** The two direct FOLD-to-BASE groups, the base positions and
   the openings sampled by the verifier; a committed word that violates a
   sampled local check is rejected.
7. **Output/seal.** The winner/valid/tie output is bound by the closure/output
   plan and the final seal; the verifier accepts the public statement only if
   these bindings pass.

`[FACT]` `CODE` Statement component to binding object. The public statement
components and the objects that bind them are:

| Statement component | Bound object | Verifier check |
|---|---|---|
| output winner / valid range / tie | closure slack/winner/recon rows, closure PSC, output plan | closure rows and PSC; output plan |
| output `Y` / Eval relation | activation root, Eval targets, tilde values | first-statement root binding; Eval/tilde BAT rows; SEND samples |
| table identity / record | reader identity, record row | reader/statement binding; record row |
| activation commitment | activation root | first-chain statement binding |
| public claim / target | committed roots, absorbed transcript | BAT rows; PSC; BASE target* and combined rows |
| transcript order / seal | chain messages and seals | three-chain replay and seal equality |

Each row is either a deterministic binding (plan/reader/statement/root/seal)
or an algebraic row that the next subsections charge; a mismatch in the table
is a mismatch in the bound object, not a free public value.

`[ARGUMENT]` `CODE` Difference classes and nonzero conditions. For a fixed
prover message difference, the algebraic checks see a polynomial in the
challenges drawn after the difference is fixed:

| Difference class | Nonzero condition | Degree / term |
|---|---|---|
| FOLD round difference | the folded claim difference is nonzero in the round polynomial | degree 2 per variable; `4/p^3` shared, `28/p^3` own |
| PSC round difference | the sumcheck difference is nonzero in the round polynomial | degree 3 per variable; `1134/p^3` |
| range / booleanity difference | the range polynomial difference is nonzero | degree at most 166 per position; `2656/p^3` |
| BAT row difference | the row linear form differs at the batching challenge | degree 1; `1/p^3` |
| BASE lambda difference | the linear check differs at lambda | degree 1; currently in `eps_proximity` |
| proximity / folded-row difference | the committed word is far from the code | `(1-delta)^t`; `0.3163976` main / `3.42e-5` mask (ineffective) |

The FOLD terms enter only when the difference is nonzero in the folded claim
and the challenge is drawn after the difference is fixed; the order is the one
above. A zero difference is the kernel case of the next table, not a
polynomial-root event.

`[ARGUMENT]` `CODE` False-statement event. Fix the prover's messages. If the
public statement is false, at least one acceptance condition fails at the
drawn challenges or sampled positions, provided the semantic error is not in a
kernel that is uncovered by another layer. The detection probability is the
union over the violated conditions: algebraic violations are Schwartz-Zippel
terms (degree over challenge-space size); proximity violations are the
sampled-position miss terms; binding/hash failures are the hash-model events.
The union bound requires each difference to be fixed before its challenge and
the challenge spaces to be independent; the order below holds.

`[UNPROVEN]` Kernel coverage (a proposition, not a premise). A difference in
the kernel of a checked map leaves that map's public value unchanged.
Soundness is established only if every error that changes the verified
semantics is outside the kernel of at least one acceptance condition, or is
caught by a binding layer (roots, reader/table identity, statement binding,
closure/output plan). This note does not prove that coverage proposition. The
kernel cases are enumerated and either covered or marked open:

| Error class | Checked map / layer | Kernel case | Covered by | Status |
|---|---|---|---|---|
| output winner/valid/tie wrong | closure slack/winner/recon rows, closure PSC | the changed bits/map have the same closure public values | closure/output plan and closure checks; the composition of kernels is not proven | `UNPROVEN` |
| Eval/output relation wrong | Eval row, tilde/residual, Dot PSC, FOLD/BASE | the W difference is in the group-row kernel and leaves the Eval/tilde targets unchanged | BAT rows and roots; kernel composition not proven | `UNPROVEN` |
| table identity/record wrong | reader/statement binding, shared record row | a different table message with the same reader identity | reader identity and record row, under the stated reader/deployment trust | conditional |
| activation root wrong | first statement binding, activation root | a different activation with the same root | CR/hash binding | conditional on CR |
| closure point/end wrong | closure PSC, point/end rows | the difference leaves the published point/end values unchanged | closure PSC hiding map; the semantic kernel is not analyzed | `UNPROVEN` |
| BASE claim wrong | `target*`, combined rows, lambda/position checks | a difference in the BASE claim kernel | lambda/position checks; their numeric terms are in `eps_proximity` | `UNPROVEN` |
| opening/path wrong | runtime tree checks | a different path with the same root | CR/hash binding | conditional on CR |

The note does **not** assert "kernel differences are irrelevant to soundness";
that would require the coverage proposition above. In particular the statement
"a kernel difference is only a different witness for the same accepted
statement" is treated as a proposition to be proved by the per-error-class
coverage table, not as a premise of the reduction. Obstruction: no full
kernel/coverage matrix across BAT/FOLD/PSC/closure/BASE/output is built.

`[FACT]` `CODE` Kernel-capture outcome. The table separates three groups:

- Covered or conditional: plan/reader/statement/root/seal bindings and the
  CR/hash model for table identity, activation root and opening paths.
- Captured by another algebraic relation: BAT row differences by the PSC,
  range or Eval/tilde rows; closure point/end differences by the closure PSC;
  the corresponding polynomial terms are in `eps_components`.
- Not captured with an effective bound: (i) fold-consistent committed rows that
  are far from the code — the algebraic fold checks pass, the folded-row
  proximity is the only distance check, and its actual miss is `0.3163976` (main)
  / `3.42e-5` (mask), so it provides no bound; (ii) kernel differences of the
  fold relation that change the statement semantics and are not caught by
  another relation — the folded row is the honest one, so the proximity cannot
  see them; the capture must come from the Eval/closure/BAT/statement binding
  layers, and the composition is not proven. This is a real soundness gap at
  the current parameters, not a wording unknown.

Candidates for the open group: a complete IOPP proximity round set; increasing
the independent query rounds (`t >= 155` for `(1-delta_main)^t <= 2^-128`); or
changing the code parameters to a lower rate / larger `r`. The kernel/coverage
matrix, the final output-consistency check and the component-level kernel
negatives from the previous paragraph remain the structural candidates.
Candidate repairs: (i) build the kernel/coverage matrix for each semantic
error class; (ii) add a final output-consistency check that recomputes the
output from the committed rows; (iii) construct explicit kernel-error
negatives at the component level. Until one is supplied, the soundness
conclusion is conditional on kernel coverage.

`[FACT]` `CODE` Algebraic terms and preconditions. The shared/own FOLD terms
`4/p^3` and `28/p^3` enter only when the actual difference in the
corresponding folded claim is nonzero and the challenge is drawn after the
difference is fixed; if the difference is zero (kernel), those terms do not
apply to that difference, and the kernel case is handled (or left open) in the
kernel-coverage table above. The PSC, range and `chi` terms are listed with
their degrees in section 6; the terms without a written derivation (single
equality) and the BASE lambda/symbol terms remain in `eps_proximity`. The
union is an arithmetic assumption, not a mechanized composition.

`[FACT]` `CODE` `MATH` Proximity miss at the actual code parameters. The code
length is `n = 2^log_n = 131,072` (the product-form encoder returns `n`
symbols: `CodeParams::n()` and `product_form_fp3`), not the coefficient count
`ell + r`. The main W/closure code has `ell + r = 73,728` coefficients; the
table code (`ell = 65,539`) is only the reader/column-opening object and does
not enter the IOPP distance; the mask code has `ell + r = 768`. The
Reed-Solomon minimum distance is
`n - (ell + r) + 1` (a polynomial of degree below `ell + r` evaluated at the
`n` distinct coset points). The relative distances and the `(1 - delta)`
factors are

```text
main: distance 131072 - 73727 = 57345, delta = 57345/131072 = 0.4375076294,
      1 - delta = 73727/131072 = 0.5625152588;
mask: distance 131072 - 767 = 130305, delta = 130305/131072 = 0.9941482544,
      1 - delta = 767/131072 = 0.0058517456.
```

The BASE check samples `t = 2` uniformly drawn positions; for a word at
relative distance `delta`, the miss probability is `(1 - delta)^2` (the
original-draw event; deduplication does not change it). The actual values are

```text
main: (73727/131072)^2 = 0.3163976670 = 2^-1.6601,
mask: (767/131072)^2   = 3.424292663e-5 = 2^-14.8338.
```

The effective distance after folding: the FOLD is a linear combination of the
committed rows with the drawn fold coefficients. A fold-consistent error
pattern is a common error `e` added to every committed row (or any pattern
whose fold combination equals the claimed fold); its folded relative weight is
that of `e`, i.e. `delta_main`, and the coefficient combination is nonzero
except with the counted probability `1/p`. The FOLD checks see no difference
because the claimed fold includes the same error; the proximity is the only
distance check. The fold therefore does not contract the distance for this
pattern for `k = 2` or `k = 14`. If the errors were independent across the
`2^k` rows, the folded error weight would be near
`1 - (1 - delta)^{2^k}` and the check would be strong, but soundness must use
the worst-case fold-consistent pattern, so the effective `delta` is
`delta_main`.

`[FACT]` `CODE` Conclusion for the direct FOLD-to-BASE proximity. At the
current parameters the proximity check does not provide a meaningful security
bound: the main-code two-position miss is `0.3163976670` (about `2^-1.66`)
and the mask-code miss is `3.424292663e-5` (`2^-14.83`); the main code
dominates. For fold-consistent rows that are far from the code, the algebraic
fold checks see no difference (the fold relation holds), so the proximity is
the only distance check and it misses with probability `0.3163976` (main); the
algebraic terms cover the separate nonzero-difference class. This is the
explicit audit statement: **direct FOLD-to-BASE under the current parameters
does not provide a security bound for the proximity sub-term.** The `0.3163976`
figure is a single conditional event — two independent uniform index draws
missing a word at relative distance `delta_main`. It is not a proximity-gap
bound, not the acceptance probability of a complete proof, and not an overall
security bound; the 155 figure is the query count for that single event only.
No complete attack is constructed: the finding is a major security-bound
gap/risk, not a demonstrated `0.3163976` attack. Candidates:
(i) increase the number of independent query rounds:
`(1 - delta_main)^t <= 2^-128` needs
`t >= 154.2`, i.e. about 155 positions; (ii) add a complete IOPP proximity
round set at the plan's radius/work; (iii) change the code parameters to lower
the rate: increase `n` or decrease `ell + r`. Fixing `n` and `ell` and
increasing `r` raises the rate and **lowers** the distance, so it is the wrong
direction. Until one is adopted, `eps_proximity` is **ineffective, not
merely unknown**, and no total soundness bound is claimed.

`[FACT]` `CODE` The closed session absorbs messages and draws challenges in
this order: roots and declarations, answers, closure roots/answers and
`r_bool`, map answers and positions, index-mask answers and positions,
openings and `psi`, Eval targets after `psi`, tilde and residual, PSC
challenges, fold start and rounds, lambda, combined/base rows, base positions,
seal. Each
challenge is drawn after the data it tests.

`[FACT]` `NUM``CODE` The negative matrix maps changed roots, PSC rows, point
values, sample answers, missing closure, fold rounds, positions, nonces and
wrong plans to the verifier layer that rejects them. The changed-message
component test keeps the original tree and root and changes only the
prover-side message: the group table changes. This is a component result, not
a final-verifier run.

`[ARGUMENT]` `CODE` Tree-reuse tamper bound, itemized.

**Frozen difference.** The prover keeps the original committed tree/root and
uses a changed message row in the group table. Let `Delta` be the difference
of the affected folded/combined claim. The root and opening checks do not see
the change because the openings are the original tree's symbols. The
difference is fixed before the FOLD/BASE challenges are drawn.

**Nonzero condition.** If `Delta` is zero in the group-row map, the check is
not claimed to catch the change; this case is enumerated in the
kernel-coverage table in section 8, not assumed harmless. If `Delta` is
nonzero, the verifier's FOLD checks are polynomials in the challenges.

**Degree.** Each FOLD round polynomial has degree at most two in its
challenge; a nonzero difference polynomial has degree at most two per
variable. The own fold has k = 14 (the actual fixture's members total 8,258 main rows;
the next power of two is 16,384). The shared fold's members are the
Gu message (2 main rows) and the one-row Omega message, so its k is 2.
Charging k = 14 for both instances, the union over the two fold checks is at
most `2 * 2 * 14 / p^3`; the exact actual merge is `2*14 + 2*2 = 32` terms.

**Challenge order.** The fold challenges, the BASE lambda and the base
positions are drawn after the committed rows and roots; the difference is
fixed when they are drawn. The challenge draws are independent of the mask
sampling source (the masks are prover randomness; the challenges are verifier
randomness). The polynomial root bound therefore applies to the same
difference.

**BASE detection.** The BASE linear check cancels the star exactly (section
5.6) and reduces to the claim relation; the combined-row and symbol checks
compare the same star on both sides, so the star does not mask the difference.
The star and tail sampling change the values, not the check equations; the
tamper bound is unchanged by section 2.1.

**BASE lambda and symbol checks.** These are not counted in the algebraic
union: the lambda is an `Fp3` field challenge without an
independent-uniform-domain/order derivation here, and the two base positions
are codeword-index samples of the real `draw -> unique_sorted` sampler; both
are carried in `eps_proximity` (unknown term; the proximity itself is
ineffective, section 8).

**Proximity/opening.** The direct FOLD checks are the per-round degree terms
above (4/p^3 shared, 28/p^3 own); the BASE lambda and the two base-position
checks add no term to the algebraic union and are carried in `eps_proximity`
(withdrawal above). The opening sets satisfy `q <= r` by the ledger
(table 603, W/closure and mask at most 4, minimum remaining 252); `q <= r` is
the Vandermonde hiding-budget condition, not a proximity soundness term. The
tail union is the accounted union. The local proximity sub-term is the
computed result in the proximity block above: the plan samples `t = 2`
positions (`s <= 2` variable; deduplication does not change the original-draw
hit event), and the actual main/mask misses are `0.3163976`/`3.42e-5`, so the
proximity is ineffective at the current parameters.

**Combination conditions.** The union bound requires that the challenge
spaces are independent and that the difference is fixed before each challenge;
both hold by the order above. For a nonzero difference, the local polynomial
bound is

```text
(2 * 13 + 2 * 2) / p^3 = 30 / p^3 < 2^-186.9,
```

under the nonzero-difference and challenge-order preconditions. The
zero-difference (kernel) case is not charged to this bound; it is covered or
left open in the kernel-coverage table above. The proximity sub-term is
**ineffective** (`0.3163976` main / `3.42e-5` mask), so this polynomial bound is
only the algebraic part and does not yield a total soundness bound. This is a
loose bound under the degree assumption, not a measured probability. A
final-verifier exact negative for this tamper is not implemented because the
hidden prover has no message-injection point after the tree commit; the
component test (the group table changes) plus this bound are what the current
API supports.

**Kernel scope (cross-reference).** The tree-reuse component test changes the
group table while keeping the original root; its difference can be zero in the
group-row map, in which case the FOLD/BASE checks do not see it. This is not
claimed as a detected negative: the kernel/coverage requirement and the open
cases are in the kernel-coverage table above. The exact final-verifier negative
for the combined tamper is not implemented. The residual assumptions are: the
original statement/root is the true committed one, the claim/target wiring is
a deployment assumption, and the concrete hash's CR advantage is not
quantified.

**Honest soundness ledger.** The pieces of the reduction are:

```text
eps_algebraic  <= 16*166/p^3 + 1134/p^3 + 32/p^3 + 1/p^3 < 2^-179.6
eps_proximity  = INEFFECTIVE: main miss 0.3163976670, mask miss 3.424292663e-5
eps_hash       = the RO/CR model term of section 6 (not a concrete-hash bound)
eps_kernel     = UNPROVEN: the open error classes of the kernel-coverage table
```

The algebraic terms are the only part with a numeric bound, and they are
conditional on the nonzero-difference/challenge-order preconditions. The
proximity term is not a small unknown to be filled in later: at the current
code parameters the direct FOLD-to-BASE two-position miss is `0.3163976` (main),
so the proximity check does not provide a security bound. The kernel-coverage
table has open classes that the ineffective proximity was supposed to capture.
Therefore no total soundness bound is claimed for the current parameters; the
honest conclusion is that the chain is not closed on the proximity side, and
the candidates are the query-round increase, the full IOPP, or a lower-rate
code-parameter change listed above.

### 8.7 Multi-stage IOPP: per-group facts and the current plan

`[FACT]` `CODE` Stage-0 facts on the current tree (hidden fixture, not an old
snapshot). The shared proximity group uses main code `(log_n,ell,r) =
(17, 65,536, 8,192)` and mask code `(17, 512, 256)`; its members are Gu
(2 main rows) and Omega (1 main row), so `k = 2`, `h = 0`, sample answers 1
each; Gu/Omega leaf class 2, mask tree leaf class 3. The own proximity
group uses the same codes; its members are 16 W messages (2 rows each), Omega
(1), the Dot shown map (1), the index mask (2 mask rows), 16 slack messages
(2 rows each), 16 bits messages (256 rows each), 16 map messages (256 rows
each) and the closure mask (2 mask rows), so the main-row total is 8,258 and
`k = 14`; the mask-row total is 4 and `h = 2`; sample answers 1 each; W leaf
class 1, Omega/Dot/slack/bits/map leaf class 2, mask trees leaf class 3. The
bits/map row count is 256, not 128: `WShape` divides by `ell - answers` and
rounds the row count up to a power of two, so 8,388,608 entries need 129 rows
and are padded to 256. The direct own FOLD term is therefore `2*14 = 28/p^3`;
the earlier `4,162`/`k=13`/`26/p^3` figures (which divided by `ell` and
ignored the answer pad) and the earlier `22/28` figures (k=11/14) are
withdrawn.

`[FACT]` `CODE` The table does not enter the IOPP. The shared members are only
Gu and Omega; the own members are the W/Omega/Dot/closure messages listed
above. The table is the reader/column-opening object: it is the `GuShape.index`
and `HiddenSendPlan.table_shape`, its 603 column draws open table columns, and
the record/column rows are BAT plain rows, not proximity members. The table
code has `ell = 65,539 = 65,536 + 3` (the three pads) and its own sample
semantics; the layout is unchanged. Because the table is not a member, its
non-power-of-two length does not enter the IOPP shape check.

`[FACT]` `CODE` The compiled `lm_head_compiled` plan is not the hidden
fixture's run plan: it has main `ell = 4,096`, `r = 962`, mask `ell = 64/256`,
`k = 6/5` and three stages; the fixture has main `ell = 65,536`, `r = 8,192`,
mask `512` and the same two-stage own candidate described below. Using
`lm_head_compiled` directly would run a different protocol.

`[FACT]` `CODE` Reference multi-stage shape conditions: every non-last stage
has `k >= 1`, `h <= k`; main and mask message lengths are powers of two; the
next main length satisfies `ell_i = 2^{k_{i+1}} ell_{i+1}`; the next mask room
satisfies `ell_zk_{i+1} 2^{h_{i+1}} >= 2 ell_zk_i + r_i + 2 r_zk_i + answers`;
the last stage has `k = h = 0`. For the actual first stage (shared `k=2,h=0`,
own `k=14,h=2`, `ell=65,536`, `r=8,192`, mask `ell=512,r=256`, `answers=1`)
shape-legal continuations exist: for example `k'=2` gives `ell'=16,384` and a
next mask `ell=4,096` (room `16,384 >= 9,729`); the own group can step down
`65,536 -> ... -> 1` with `k=14` followed by `k=13` steps. These are
shape-feasible values, not yet the trusted parameters.

`[FACT]` `CODE` The fixture plan now carries the candidate trusted stage lists.
`IoppStagePlan` stores `main`, `mask`, `k`, `h`, `positions`, `samples` and
`work_bits`; `HiddenProximityPlan` has `shared_stages` and `own_stages`, and
`iopp_stage_fault` checks the reference shape conditions (non-last `k >= 1`,
`h <= k`, power-of-two message lengths, adjacent
`ell_i = 2^{k_{i+1}} ell_{i+1}`, next mask room, last `k = h = 0`). The lists
bound in the fixture are:

- shared: stage0 `(17, 65,536, 8,192)/(17, 512, 256)`, `k=2,h=0`,
  positions 2, samples 1, work 0; stage1 (last)
  `(17, 65,536, 256)/(17, 16,384, 256)`, `k=h=0`, positions 2, samples 1.
- own (default two-stage candidate): stage0
  `(17, 65,536, 8,192)/(17, 512, 256)`, `k=14,h=2`, positions 252, samples 1,
  work 24; stage1 (last) `(17, 65,536, 259)/(17, 16,384, 259)`, `k=h=0`,
  positions 259, samples 1, work 0.
- own (same-parameter three-stage comparison only): stage0 as above; stage1
  `(17, 8, 256)/(17, 16, 256)`, `k=13,h=13`, positions 252, samples 1,
  work 24; stage2 (last) `(17, 8, 259)/(17, 1,024, 259)`, `k=h=0`,
  positions 259, samples 1, work 0. The three-stage list is not the default
  and has no planner check for the middle stage; it is a comparison basis.

The first-stage codes equal the group codes and the first-stage `k`/`h` equal
the actual member counts (shared 2/0, own 14/2). Unit tests
`actual_shared_and_own_stage_lists_are_legal` and
`bad_stage_chains_are_rejected` cover the actual lists and the failures
(wrong first code, wrong folded length, non-last last stage). The lists are
plan data, not prover self-report.

`[FACT]` `CODE` The multi-stage loop is wired into the hidden path. The plan
carries the per-stage fresh ids, the stage-FOLD group/member id, the switch and
mask run/configuration digests, the base-star tree plan and the first-switch
old-mask source; the proof envelope is version 5. The default own stage list is
two stages with stage-0 positions/work 252/24 and last positions/r 259/259
(the shared group uses its measured 2/0 stages). The three-stage list is
retained only for the same-parameter comparison. The prover/verifier loop,
per-stage opening authentication, independent stage reconstruction, switch
target checks and the staged final-verifier negatives are exercised by the
formal seq16 run.

`[FACT]` `CODE` `MATH` Planner-level derivation for the multi-stage chain
(bound in the plan, not a final security parameter). The trusted interval
planner computes Johnson/list/MCA/FOLD/BASE/sample/field/batching terms from
outward-rounded bounds and returns stage-0 positions/work 252/24 and last
positions/r 259/259 for the own group; the shared join keeps its pinned two
positions. The naive union sum of the counted round terms is
`2^-123.69`, while the worst round is `2^-128.000035`; the per-round target
does not imply a 128-bit union bound, and that gap is carried as an explicit
boundary in the rework18 evidence note. The 155/129/155/15/20 values remain
reference-interval candidates only.

`[FACT]` `MATH` c-difference and `Im(A)`. For a public value `P = A X + c(V,S)`,
two secrets `(V,S)` and `(V',S')` give the same image exactly when
`c(V,S) - c(V',S')` lies in `Im(A)`. The block covers above exhibit each public
value's `A` (the fresh random columns) and `c` (the secret terms); the pivots,
constant determinants and Vandermonde full-row-rank statements put the
secret-difference directions into `Im(A)`. The `A = 0` values are the
winner/valid/tie outputs and the public statement constants; they are listed as
the approved public boundary and are not witness-hidden. SEND answers, PSC
transcripts, the same mask/star entries at their multiple sites and the
cross-tree openings are covered by the same affine maps. A fixed public root
is an input and cannot be replaced by a single tree's `q <= r` condition; the
joint root/RO history is part of the statement view.

`[UNPROVEN]` The multi-stage joint-hiding and soundness composition is
extended in sections 5.6/5.7 for the switch tree, but it remains a derivation,
not a mechanized proof. The reference interval planner port, the full
soundness composition and the trusted X source remain open. The table remains
outside the IOPP (N4); if a future plan moves a switched old main to
`ell=65,539`, the task requires stopping and handing the pad/representation
question to the expert.

## 9. Engineering boundaries and wording

`[FACT]` The joint-view module is test-only. It exercises the real encoder,
the real hiding polynomial and the real fold/base paths, but a passing test
module is not a production runtime detector for every bad rank or every
tampered message. The workspace result and the three seq16 runs are the
delivery evidence reported by the implementation side; they are not an
independent external verification. The 800-test workspace figure (17
ignored in 48 suites), the proof hashes and the HEAD are the
implementation-side delivery figures, not an independent expert validation of
the protocol. Editing the document or the
test-only module is not a new validation of the production protocol either;
the production messages, commitments and challenge order are unchanged.

`[FACT]` The staged loop is wired in this round: the proof envelope is
version 5, the hidden session layout is 14, and the LastBase verifier also
returns the drawn positions to the ledger harness. The earlier v7-v13
mask-sampling and joint-view changes were test-only; the loop wiring changes
the proof path, so the compatibility statement below applies to the staged
proof. The protocol relations, commitments, challenge order, parameters and
encoding format for the single-stage path are otherwise unchanged. The
previously closed counter-scope, sample-path and fixture-assumption wording is
maintained; the v10-v13 target inventory, online simulator and soundness
reduction do not reopen those items.

`[FACT]` The target row list of section 5.9 is the basis for the next round's
argument. The full joint composition of the fresh variables is supplied by the
block partition and conditional block lemma of section 5.7, not by a direct sum
from per-block pivots; its scope condition is witness-independence (a dummy
witness for the same public statement), not a witness-free simulator. The
soundness reduction of section 8 is not a total bound: the kernel-coverage
table has open classes, and the direct FOLD-to-BASE proximity is ineffective
at the current parameters (`0.3163976` main miss). The workspace counts, proof
hashes and HEAD remain the implementation side's reported figures; the TL
self-check is not an independent expert verification, and editing this
document or the test-only module is not a new validation of the production
protocol.

`[FACT]` `NUM` Compatibility evidence. The default two-stage formal seq16
proof is 65,432,832 bytes (three-stage same-parameter 212,523,920 bytes); the
old single-stage 9,939,144-byte artifact was a version-4 envelope and is not
accepted by the version-5 decoder (UnsupportedVersion; the formal outer
negative matrix checks this). The earlier mask-sampling counter evidence still
applies to the prover's draws: the sampling distribution is the unbiased one,
and the boundary/retry tests exercise the rejection path. The counter includes
`redraw` and repeated evaluations, and the timed runs include the atomic
overhead when the counter is enabled; they are not exactly comparable to the
earlier no-counter baseline.

`[FACT]` An old proof verifying would **not** prove that the old prover
sampled unbiasedly. The verification equations are the same; the mask
distribution is a prover-side implementation property. The version change is
an encoding-compatibility change, not a proof of the security source. The
unbiasedness conclusion is the honest-prover assumption of section 2.1, not
the compatibility test.

`[FACT]` `NUM` Sampling tests: boundary `p-1` accepted; `p` and `2^64-1`
rejected and retried; retry exhaustion returns an error after the fixed
number of attempts; sequential and random-access draws agree; the same seed
replays identically and different seeds give different streams; the sample
has no repeated values; cross-request material/seed/commitment reuse is
rejected by the existing ledger tests; the full prover/verifier suite and the
three seq16 runs pass.

`[ASSUMPTION]` The fixture's table tails and salts are deterministic fixture
data. They are not evidence that real private weights have independent
randomness across requests. The production table/asset commitment's random
source is a deployment trust assumption, listed separately from the per-proof
`ByteMaskCoins` stream; this document does not prove the deployment setup's
randomness.

`[FACT]` The five-case matrix covers gpt2 and gemma3 at seq16/seq512 and
timesfm at 16 patches; the detailed two-stage measurement is the gpt2-seq16
formal fixture. No performance or proof-size equivalence to another system is
claimed. All test numbers in this note are the implementation side's reported
results (TL self-check), not an independent expert verification.

`[FACT]` The envelope version is 5 and the hidden session layout is 14. A
version-4 or unknown version is rejected before decoding; there is no silent
fallback to the single-stage tail. A security-semantics version change beyond
this staged envelope is not made here; it would be escalated.

`[FACT]` The asset inventory and its bounds apply only to the listed
`gpt2-seq16` fixture path and its shapes/code parameters; they are not
extrapolated to other models, lengths, batches or production entries.

`[FACT]` The output-binding boundary keeps winner, valid range and tie result
public; the full reduced output row is hidden by the per-block conditional
arguments of 5.5/5.6. The full joint composition of all shared columns, roots
and PSC/BASE transcripts remains `[UNPROVEN]` (section 5.9); this note does not
claim a closed joint theorem.

## 10. Open items

- `[FACT]` The global online composition is supplied by the block partition
  and conditional block lemma of section 5.7. The remaining scope item is a
  witness-free simulator: the dummy witness must satisfy the same public
  statement, so the note proves witness-independence, not a witness-free
  simulation (section 5.7).
- `[UNPROVEN]` Kernel coverage in the soundness reduction: the note does not
  assert that kernel differences are irrelevant; the open error classes and
  candidate repairs are listed in the kernel-coverage table of section 8.
- `[UNPROVEN]` A mechanized symbolic proof of the constant determinant of the
  PSC/FOLD hiding maps; the triangular derivation and the numerical checks
  support it, but they are not a formal proof.
- `[UNPROVEN]` A concrete-hash (non-model) hiding bound. The mask sampling is
  exactly uniform in the ideal-word model; the real finite-seed statement is
  computational under the PRG/RO model, and the concrete hash's advantage is
  not quantified. This is a model-level condition, not a residual modulo
  bias; the modulo reduction has been removed.
- `[UNPROVEN]` An exact final-verifier negative for the tree-reuse tamper; the
  current API has no prover-side message injection point after the tree
  commit. The component test and the degree bound are delivered instead.
- `[GAP]` The direct FOLD-to-BASE proximity at the current parameters does not
  provide a security bound: `t = 2`, `delta_main = 0.4375076294`,
  `(1 - delta_main)^2 = 0.3163976670`; the mask miss is `3.424292663e-5`. The
  algebraic FOLD terms do not cover the folded-row error class. Candidates:
  about 155 independent query positions for `2^-128`, a complete IOPP
  proximity round set, or a lower-rate code parameter change (increase `n` or
  decrease `ell + r`; fixing `n`/`ell` and increasing `r` raises the rate and
  lowers the distance) (section 8). `eps_proximity` is ineffective, not a
  small unknown.
- `[GAP]` The multi-stage IOPP stage lists are in the plan and the
  BAT -> (FOLD -> CS)* -> BASE loop is implemented and exercised on the real
  seq16 fixture; the per-stage positions/work come from the candidate planner
  (252/24, 259/259) and are bound in the plan. The interval arithmetic and
  largest-m search are implemented; the union-sum gap (2^-123.69 versus
  per-round 2^-128.000035), a mechanized soundness/joint-hiding composition,
  and the resource decision for the 66 MB/170 GB shape remain expert/boss
  items (section 8.7).
- `[UNPROVEN]` A mechanized composition theorem and an end-to-end 128-bit
  claim; `126.21`/`116.01` remain plan-ledger numbers.
- `[LONG-TERM]` The trusted X source, policy distribution/freshness and
  chain-based commitment/verification are recorded as long-term direction;
  they are not implemented here and are not debts of this round.

No statement in this note claims that the trusted X source is proved, that
weights are hidden under a formal zero-knowledge definition, or that the run
is a production acceptance.
