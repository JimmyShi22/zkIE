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
rollback `8d32d3de`; fixture `gpt2-seq16`, 16 positions, proof 9,939,144 B.
The committed document is version 9.

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
| 10 | BASE star main/mask rows and tails | BASE target, combined rows and star openings | target, rows and openings of the same star | fresh one-time pads on the target dot product, each combined coordinate and each tail coordinate | BASE lambda/position terms are in `eps_proximity = UNKNOWN` (section 6) | `NUM` real star path, `CODE` |
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
  seed; one stream is not reused by another oracle.

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
different; the concatenated view is block diagonal and ranks add. Within one
oracle the shared columns are explicit.

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
extracted into one matrix; those blocks are separate and concatenate
diagonally by the sharing table in section 4.

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
for the actual own-fold size k = 14 and for k = 2 at three challenge seeds;
the 29 x 29 and 5 x 5 matrices are full rank and have determinant one. The
closed path has two FOLD instances (the shared and own proximity groups);
their randomness is fresh and their openings/tails are separate.

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
mechanisms, so their blocks concatenate diagonally.

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

`[FACT]` `CODE` `MATH` Simulator construction for the opened and unopened
leaves. The standard salted-Merkle random-oracle commitment lemma applies with
the following ordered simulator. Write `q` for the cumulative number of
random-oracle queries and calls made by the adversary and the simulator up to
the event, including leaf, node, root, challenge and opening queries. The
simulator receives only the public statement/plan and the random-oracle
interface; it receives neither the prover's witness nor the opening set.

1. **Domain-level public values (1).** Before any hash is fixed, the simulator
   chooses the algebraic public values it will publish: the SEND answers, Eval
   targets, tilde values and residual, the PSC transcripts and end values, the
   closure point values, the FOLD start/rounds, and the BASE target and
   combined rows. Each value is either prescribed by the public instance
   (public constants and the approved winner/valid/tie outputs) or sampled
   uniformly. The choices are recorded as the domain-value vector; they are
   not derived from a witness. The algebraic cover of sections 5.2-5.6 says
   that the tail/pad/mask/star coordinates can later realize these values.
2. **Salt once (2).** For every leaf the simulator chooses one 40-byte salt
   from its own randomness at allocation time and stores it. The salt is never
   reselected later and is not revealed before that leaf's opening.
3. **Uniform commitment (3).** For every leaf it announces a uniformly sampled
   digest `com_i`, lazily and recorded. It does not compute `com_i` from a
   leaf value; in the ideal RO model `com_i = H(salt_i || leaf_i)` is uniform
   and independent of the leaf.
4. **Root (4).** The root is the Merkle hash of the recorded `com_i`. Under
   the RO replacement it is a hash of uniform digests and is uniform and
   independent of the leaf symbols.
5. **Challenges (5).** The Fiat-Shamir challenges are answers of the same
   random-function table on the transcript so far. The table has no input that
   depends on an unopened leaf, so the challenges are independent of the
   unopened leaf values; the domain values of step 1 are fixed before these
   draws, matching the real order.
6. **Opening-time message/tail choice (6).** When an opening position is
   drawn, the simulator solves the algebraic hiding map for message and tail
   coordinates that (a) realize the domain values of step 1 and (b) produce the
   opened codeword symbols. The maps are the Vandermonde tail block for
   `q <= r` (section 5.2), the invertible PSC/FOLD coefficient maps (sections
   5.4 and 5.6) and the BASE star split (section 5.6). It maintains the set of
   message/tail assignments consistent with all revealed symbols; at each new
   opening it chooses an assignment in that set that produces the next symbol
   and keeps the unconstrained coordinates free. The Vandermonde rows of
   distinct opened positions are independent and the total number of opened
   positions is at most `r`, so the set stays nonempty for every adaptive
   opening sequence; the revealed symbols are therefore jointly consistent
   with one assignment, while the internal choice of the extension (the
   unopened tail coordinates) is never revealed. The PSC/FOLD/BASE blocks are
   separate systems with their own invertible maps, and the shared entries are
   explicit columns (sections 4 and 5.5), so the solutions agree on them. The
   inverse maps use only public challenges and the step-1 domain values, never
   the witness. The simulator then computes the leaf, programs
   `H(salt_i || leaf_i) = com_i`, reveals `(salt_i, leaf_i)`, and keeps the
   choice in its state. It does not reselect the salt or `com_i`.
7. **Path consistency (7).** The opened path is recomputed from the recorded
   child digests; every queried leaf/node/FS point returns the recorded value,
   and a repeated query returns the same answer. An opening set that violated
   `q <= r` or the block rank condition would make step 6 unsolvable; the
   protocol samples `q <= r` by the ledger, and the joint rank tests of
   section 5.5 cover the production shape, so this is not an
   adversary-chosen event.

`[FACT]` `MATH` Bad events. The simulation can fail only through:

- **Guess before opening.** A query made before a leaf is opened hits the
  exact input `salt_i || leaf_i`. Conditional on the view, the input has
  min-entropy at least `H_salt + H_leaf`, where the salt contributes 320 bits
  (chosen once at step 2 and hidden until step 6) and `H_leaf` is the
  conditional min-entropy supplied by the algebraic tail/pad/mask/star
  randomness of step 1 (nonnegative; the pads and tails are the mask sources of
  sections 2-5). Over `q` cumulative queries the probability is at most
  `q * 2^{-(320 + H_leaf)} <= q * 2^-320`. The count `q` is cumulative over
  leaf, node, root, challenge and opening queries.
- **Collision or programming inconsistency.** Two distinct inputs receive the
  same random-oracle output, or a programmed value conflicts with an earlier
  answer: probability at most `(q + 1)^2 / 2^256`.

If neither event occurs, the simulated view is identical to the real view for
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
structure is given by the distinct streams and by the explicit shared list;
the blocks are not inferred to be independent merely from the sub-block ranks.
The production-size closure joint test (section 5.5) is a local numerical
confirmation of one block; the whole-chain statement is this conditional/block
derivation. A complete multi-oracle matrix extraction is not claimed.

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

`[UNPROVEN]` The multi-oracle conditional composition is not mechanized; the
argument above is the per-round conditional/block derivation, and the
production-size tests are local evidence for their blocks.

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
| `(E_0)` | `ends[0].value` of the closure PSC | `eq_point` on the committed map message | `map(point)` (committed activation map at the PSC point) | `shown0`, coefficient `beta` | `DotProof.end_values[0]`, absorbed by `send_end_values`; also used to derive `row_target` |
| `B(point)` | `b_point` | `eq_point` on the committed bit message | `B(point)` (multilinear bit map at the PSC point) | `shown2`, coefficient `1` | `point_values[0]`, absorbed by `absorb_closure_point`; also in the virtual target below |
| virtual second map | `ends[1].value - 1 + b_point` | `shown2` with `1` and `shown1` with `beta`; no main term | none: since `Omega = 1 - B`, the target equals `beta*shown1 + shown2`; the row is mask-only | `shown1`, `shown2` (shared with the `(E_0)`/`B(point)` rows) | derived from the two absorbed values; no new wire value |
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
| difference | `residual R` | `chi^t` on `tilde_t`, `-chi^t` on `claim_t`, `t = 0..positions-1` | none: `tilde_t` is already public and `claim_t` is a random mask; the row publishes only the mask sum `sum_t chi^t claim_t = sum_t chi^t tilde_t - R` | `tilde_t`, `claim_t` (both one-time; `tilde_t` also in the public tilde) | `R` absorbed by `absorb_dot_residual`; `tilde_t` absorbed by `absorb_tilde`; `claim_t` not separately published |
| `(U)` | `output.row_target` of the Dot PSC | `hiding_form[j]` on `hiding_j`; `claim_scale*chi^t` on `claim_t` | none: the row is mask-side; the hidden batched maps enter through the Dot PSC transcript | `hiding_j`, `claim_t` | `row_target` recomputed by the verifier from the absorbed Dot PSC rounds and end values |
| `(E_0)` | `ends[0].value` of the Dot PSC | `chi^t*eq_index(point, value_at)` on the batched dot message entry `(t, value_at)` | `sum_t chi^t W_t(point)` (the batched activation map) | `end_0`, coefficient `beta_dot` | `DotProof.end_values[0]`, absorbed by `send_end_values` |
| `(E_1)` | `ends[1].value` of the Dot PSC | `eq_index(point, value_at)` on the Omega message entry `value_at` | `sum_t chi^t Omega_t(point)` (the committed Omega map) | `end_1`, coefficient `beta_dot` | `DotProof.end_values[1]`, absorbed by `send_end_values` |

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

`[FACT]` `CODE` `MATH` Joint conditional distribution of the values and the
masks. The existing public view already contains the secret directions, and
they are covered blockwise:

- closure block: per position the five shown columns `shown0..shown4` are the
  unit/`beta` pivots of the five values `ends[0].value`, `b_point`,
  `ends[1].value`, `d_value`, `bq_value`; the virtual and difference rows are
  the exact linear combinations `ends[1].value - 1 + b_point` and
  `d_value - bq_value` of those pivoted values, so they add no independent
  secret direction; the `(U)` row uses the 49 hiding columns whose map to the
  PSC transcript has the constant nonzero determinant of 5.4; the encoded
  opening uses the tail Vandermonde block for `q <= r`. This is the 54-column
  per-position block of the joint matrix in 5.5.
- dot block: the two end slots and the hidden map evaluations are distinct
  one-time columns; the difference and `(U)` rows are mask-only relations (no
  hidden functional); the hiding/claim classes are disjoint from the end
  class.
- own block: `eval_t` and `tilde_t` are distinct one-time columns, one per
  position; the Eval point and the shared point are distinct public
  challenges.
- SEND/base blocks: each answer has its own reserved pad; the BASE `target*`
  is one-time in the star, and the same star appears in the combined rows and
  openings (5.6); the star/mask blocks share no random coordinates and
  concatenate diagonally.

For each block the matrix `[A | B]` has the random pad/tail/star columns `A`
and the secret columns `B`; the pivots above put every secret column of `B`
into the image of `A` (row by row for the padded values, and through the
invertible hiding maps for the transcript/message coordinates), so
`rank([A | B]) = rank(A)`, i.e. `Im(B) subset Im(A)`. Because the blocks use
disjoint streams and disjoint mask classes, the direct sum inherits the
inclusion. This is the joint conditional statement for the targets together
with the same mask/opening/BASE star: conditional on the public view, the
targets are either fixed values or uniform one-time-padded values, and their
distribution contains no direction of `Y` beyond the approved winner/valid/tie
plus public statement boundary. No traced target reveals `Y` outside that
boundary, so the stop condition does not trigger.

`[UNPROVEN]` The adaptive composition and mechanization of this blockwise
conditional argument; the deployment randomness of the asset/table tails and
salts; and the concrete-hash hiding advantage. These remain explicit
assumptions/unknowns (sections 5.7 and 7), not silently covered by the target
argument.

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
`6/p^3 < 2^-189.4`, `22/p^3 < 2^-187`, `28/p^3 < 2^-186`,
`30/p^3 < 2^-186.9`, `50/p^3 < 2^-186.3`, `1134/p^3 < 2^-181`,
`16*166/p^3 < 2^-180`. The exact quantities used in the self-check are
`1/p^3 = 2^-191.99999999899`, `2/p^3 = 2^-190.99999999899`,
`3/p^3 = 2^-190.4150374983`, `6/p^3 = 2^-189.4150374983`,
`22/p^3 = 2^-187.5405683804`, `28/p^3 = 2^-187.1926450769`,
`30/p^3 = 2^-187.0931094034`, `50/p^3 = 2^-186.3561438092`,
`128/p^3 = 2^-184.99999999899`, `1134/p^3 = 2^-181.852...` and
`16*166/p^3 = 2^-180.625...`. In particular `128/p^3` is slightly larger than
`2^-185`, so no strict `< 2^-185` claim is made for it; it is not used in the
bound. The `3/p^3` and `6/p^3` entries are listed only for the arithmetic
self-check; neither is used as a retained union term below.

The hiding-rank maps have constant nonzero determinant (section 5.4), so the
former bad-challenge rank term is withdrawn: there is no Schwartz-Zippel
event for the PSC or FOLD hiding maps. The remaining degree terms are
soundness checks of the sumcheck and range relations, not hiding-rank terms.
The entries are:

| Component | Count/degree | Bound |
|---|---|---:|
| direct shared FOLD check | k <= 11, 11 rounds, degree 2 | `22/p^3 < 2^-187` |
| direct own FOLD check | k = 14, 14 rounds, degree 2 | `28/p^3 < 2^-186` |
| BASE lambda and symbol checks | withdrawn from the algebraic union: no independent-uniform field-domain/order derivation for the Fp3 lambda challenge, and the two checks are codeword-index samples, not field challenges | carried in `eps_proximity = UNKNOWN`, not in `eps_components` |
| base position sampling | real sampler `draw -> unique_sorted`: `t` draws, `s` unique positions, `t >= s`; distance parameter `delta` | miss `(1 - delta)^t`; the local `delta` convention is not derived, in `eps_proximity` |
| range booleanity | 16 positions, degree at most 166 each | `16*166/p^3 < 2^-180` |
| PSC sumcheck (Dot once, closure 16 positions) | 10 and 23 variables, degree 3 each | `(30 + 16*69)/p^3 < 2^-181` |
| single equality | the check exists, but its polynomial-degree/event derivation is not written in this note | not counted in `eps_components`; carried in `eps_proximity = UNKNOWN` |
| `chi = 0` | one event: a uniform `Fp3` challenge equals zero | `1/p^3 < 2^-191` |
| challenge-draw zero fallback | at most 6 rejected words per draw, `(1-p/2^64)^6` | below `2^-192` per draw |
| mask-coordinate retry exhaustion | at most 64 attempts, `(1-p/2^64)^64` | below `N * 2^-2048` over `N` coordinates |
| hash model | plan-ledger term | `(t + q_V + 1) * 2^-128 + (17/2) * (t + q_V)^2 / 2^256` |

`[ARGUMENT]` `CODE` The component union bound used here is

```text
eps_total <= eps_hash + eps_components + eps_proximity
eps_hash = (t + q_V + 1) * 2^-128 + (17/2) * (t + q_V)^2 / 2^256
eps_components <= 16*166/p^3 + 1134/p^3 + 50/p^3 + 1/p^3
                < 2^-180 + 2^-181.85 + 2^-186.3 + 2^-191
                < 2^-179.6
eps_proximity = UNKNOWN
eps_total also carries the retry exhaustion N * 2^-2048 and the Q_R / Q_A factors
```

`eps_proximity` is an explicit unknown, not zero: the base-position sampler
(`draw -> unique_sorted`), the local proximity distance/query soundness, the
single-equality term whose degree/event basis is not derived here, and the
withdrawn BASE lambda/symbol terms are carried here. The document does not
claim a total security bit count. The `BASE 3/p^3` figure of the previous
version is withdrawn from `eps_components`: the lambda challenge is an `Fp3`
field value whose independent-uniform-domain/order derivation is not given
here, and the two symbol checks are codeword-index samples produced by the
real sampler, not uniform field challenges. The earlier
`6/p^3 < 2^-189.4` merge is likewise not used, because two of its
`2/p^3` single-equality units lack a written degree/event derivation. The
counted terms are the FOLD merge `22/p^3 + 28/p^3 = 50/p^3 < 2^-186.3`, the
range term, the PSC term and the `chi = 0` event; each has an explicit degree
or event basis (FOLD round checks of degree 2, PSC round checks of degree 3,
range checks of degree at most 166 per position, and one uniform-zero-challenge
event). No term without a derivation is kept in the union.

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
bits; shared/base positions 2 each. The direct FOLD checks have k <= 11
(shared) and k = 14 (own). The plan does draw one `Fp3` BASE lambda and two
base positions, but their numeric error terms are not kept in
`eps_components`: they are carried in `eps_proximity = UNKNOWN` for the
domain/order reason above. The prior "8 rounds each <= 2^-128" term is a
multi-stage ledger figure that is not mapped to this direct FOLD-to-BASE
compiled plan; it is withdrawn from this chain's bound and replaced by the
per-round degree terms above.

`[FACT]` `CODE` The real base-position sampler is `draw -> unique_sorted`: the
plan's `plan_loc` convention draws `t` codeword-index positions and keeps the
`s <= t` unique ones; a repeated draw is discarded. If the committed word is at
relative distance `delta` from the code (agreement `1 - delta`), each
independent uniform draw misses the disagreement set with probability at most
`1 - delta`, so the miss probability after `t` draws is `(1 - delta)^t`; the
hit probability is `1 - (1 - delta)^t`. Deduplication can only reduce the
number of distinct tested positions, so it does not increase the miss
probability and the bound uses the draw count `t`, not `s`. The local plan
exposes `s = 2` and a radius parameter but does not derive a numeric
`delta`/agreement value from the radius and work bits; the base-position and
proximity soundness is therefore the explicit `eps_proximity = UNKNOWN` above,
not zero and not the withdrawn ledger term. The `Fp3` BASE lambda challenge
and the two codeword-index checks are likewise carried in `eps_proximity`
(withdrawal above), not in `eps_components`. The retry exhaustion
(`N * 2^-2048`) and the multi-request `Q_R`/adaptive `Q_A` factors are carried
separately; they are not dropped.

`[UNPROVEN]` The local proximity/base-position distance/query soundness
derivation; the plan parameters and the reference plan-loc convention are
stated, but the local numeric error term is not derived here.

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
| BASE star fresh and one-time | `draw_star` from a fresh stream; real-path test | supported |
| local proximity distance/query soundness | plan radius 16,384, 603 column draws, 12 work bits, two base positions | `UNPROVEN` numeric bound; the prior 8-round ledger term is withdrawn, not imported |
| BASE lambda and symbol-check terms | `Fp3` lambda challenge and two codeword-index samples; no independent-uniform-domain/order derivation here | withdrawn to `eps_proximity = UNKNOWN`; not in `eps_components` |
| single-equality term | the check is present; its polynomial-degree/event derivation is not written here | not counted; carried in `eps_proximity = UNKNOWN` |
| hash RO for hiding, CR for binding | leaf/root hash construction and salt lengths; model statement | supported as a model; no concrete-hash quantification |
| no mask/pad/stream/table reuse | ledger state validation and rejects | supported; executable |
| approved output boundary | policy/plan; the full `Y` row is hidden beyond winner/valid/tie | `FACT` boundary; conditional full-row hiding |
| closed path has two FOLD instances and one closure PSC per position | `member_fold_shape` call sites; `messages.psc.len() == positions` | `CODE`; actual k=14 own fold |
| no code-switch stage in the hidden path | plan rejects a switch configuration | supported |
| asset inventory only for this fixture | the stated shapes and code parameters | scoped, not extrapolated |
| concrete-hash PRG/RO indistinguishability | the ideal-word sampling proof plus the hash model | `UNPROVEN` model-conditional; no concrete-hash bound |

## 8. Rejection chain and tree-reuse tamper

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

**Nonzero condition.** If `Delta` is zero in the group-row map (the kernel),
the check is not claimed to catch the change; this document does not claim a
kernel negative. If `Delta` is nonzero, the verifier's FOLD checks are
polynomials in the challenges.

**Degree.** Each FOLD round polynomial has degree at most two in its
challenge; a nonzero difference polynomial has degree at most two per
variable. The own fold has k = 14 (the run reports 14 rounds); the shared
fold's members are the 1024-row Gu message, the one-row Omega message, the
record row and the opened-column rows (at most the table's 603 opening
events), so its k is at most 11. Charging k = 14 for both instances, the
union over the two fold checks is at most `2 * 2 * 14 / p^3`.

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
are carried in `eps_proximity = UNKNOWN` (section 6).

**Proximity/opening.** The direct FOLD checks are the per-round degree terms
above (22/p^3 shared, 28/p^3 own); the BASE lambda and the two base-position
checks add no term to the algebraic union and are carried in `eps_proximity`
(withdrawal above). The opening sets satisfy `q <= r` by the ledger
(table 603, W/closure and mask at most 4, minimum remaining 252); `q <= r` is
the Vandermonde hiding-budget condition, not a proximity soundness term. The
tail union is the accounted union. The local proximity distance/query/work
model (radius 16,384, 603 draws, 12 work bits) is the plan's parameter set; a
numerical proximity soundness bound for it is not derived here and is not
imported from the multi-stage ledger. `[UNPROVEN]` The local proximity
distance/query numeric bound.

**Combination conditions.** The union bound requires that the challenge
spaces are independent and that the difference is fixed before each challenge;
both hold by the order above. The local polynomial bound is

```text
(2 * 14 + 2 * 11) / p^3 = 50 / p^3 < 2^-186.3,
```

plus the conditional proximity distance/query term above. This is a loose
bound under the degree assumption, not a measured probability. A
final-verifier exact negative for this tamper is not implemented because the
hidden prover has no message-injection point after the tree commit; the
component test (the group table changes) plus this bound are what the current
API supports.

**Kernel scope and binding derivation.** If the changed message's difference
is in the kernel of the group-row map, the FOLD/BASE checks do not see it.
Only the following conditional connection can be stated; the kernel case
itself is not proven:

1. The verifier's public claim and target are functions of the committed
   roots and the public statement, not of the hidden row directly (section
   5.9). A false statement requires a different public claim or target. This
   is the intended binding; the wiring from the root to the claim/target is a
   deployment assumption, not independently trusted here.
2. If the changed message changes that public claim or target, its difference
   is nonzero in the group-row map (otherwise the public value is unchanged);
   the FOLD/BASE checks see a nonzero difference polynomial, and the
   challenges are drawn after the difference is fixed, so the polynomial
   bound applies.
3. If the difference is in the kernel, the public claim and target are
   unchanged and the verifier accepts the same public statement as for the
   original committed message. This does **not** by itself imply that the
   accepted statement is true: with the public target fixed, the changed
   message is a committed witness that may fail to satisfy that target, and
   the FOLD/BASE layer cannot see the kernel difference. The conclusion "a
   kernel difference is only another witness for the same true claim" is not
   valid without an additional proof that every committed witness satisfying
   the visible relation also satisfies the public target. No such proof is
   given here; the kernel case is therefore `[UNPROVEN]` and is not counted
   as a detected negative.
4. The root/leaf index/run bind the committed message under the CR hash
   assumption: a different message with the same root is a collision or a
   hash-model event (`(q+1)^2/2^256` in the RO analysis). If the changed
   message also changes the root or the statement, the binding checks reject
   it at the root/statement layer.

The residual assumptions are: the original statement/root is the true
committed one, the claim/target really are bound to the committed
statement/roots (the wiring is not independently trusted; a deployment
assumption), and the concrete hash's CR advantage is not quantified. The
document does not claim that the FOLD/BASE layer detects a kernel difference,
and it does not claim that a kernel difference is harmless: the exact
final-verifier negative for the combined tamper is not implemented, and the
kernel/accepted-statement connection is `[UNPROVEN]`.

## 9. Engineering boundaries and wording

`[FACT]` The joint-view module is test-only. It exercises the real encoder,
the real hiding polynomial and the real fold/base paths, but a passing test
module is not a production runtime detector for every bad rank or every
tampered message. The workspace result and the three seq16 runs are the
delivery evidence reported by the implementation side; they are not an
independent external verification. The 751-test workspace figure, the three
proof hashes and the HEAD are the TL-reported delivery figures, not an
independent expert validation of the protocol. Editing the document or the
test-only module is not a new validation of the production protocol either;
the production messages, commitments and challenge order are unchanged.

`[FACT]` The production proof code is unchanged in this round; the v7/v8/v9
changes are test-only and document. The earlier mask sampling source change
(section 2.1) changed only the prover's random values; the protocol
relations, commitments, challenge order, parameters and encoding format are
unchanged. The proof content and hash are the measured fixture result from
that source. The previously closed counter-scope, sample-path and
fixture-assumption wording is maintained; the v9 target inventory, simulator
construction and constant corrections do not reopen those items.

`[FACT]` `NUM` Compatibility evidence. The new proof length is 9,939,144
bytes and its sha256 is `f8f512c0...3215f5` in the three seq16 runs; the
process-level counter recorded `sample_accepts = 276,343,251` and
`sample_rejects = 0` in each run. The rejection sampling only changes a draw
when a candidate word is at or above the modulus; this deterministic fixture
produced no such word, so the proof content is byte-identical to the previous
round. The sampling distribution is still the unbiased one; the boundary and
retry tests exercise the rejection path. The counter includes `redraw` and
repeated evaluations, and the timed runs include the atomic overhead; they
are not exactly comparable to the earlier no-counter baseline. The encoding
version and the hidden schema version are unchanged because the message
structure is unchanged. A proof produced before the sampling change
(`9,939,144` bytes, the previous round's dump) was decoded and verified by the
new verifier through the bytes entry: `loaded_proof_ok=1`, with the same
`9,939,144` byte length. The verifier never re-derives the prover's mask
values, so the changed sampling does not invalidate old proofs under the same
plan and public statement.

`[FACT]` The old proof verifying does **not** prove that the old prover
sampled unbiasedly. The verification equations are the same; the mask
distribution is a prover-side implementation property. The version being
unchanged is an encoding-compatibility statement, not a proof of the security
source. The unbiasedness conclusion is the honest-prover assumption of
section 2.1, not the compatibility test.

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

`[FACT]` The GPT-2 single-fixture seq16 runs are not three models or all
models. No performance or proof-size equivalence to another system is claimed
from these runs. All test numbers in this note are the implementation side's
reported results (TL self-check), not an independent expert verification.

`[FACT]` The schema and encoding versions are unchanged because the message
structure is unchanged and there is no deployed compatibility surface the
version would need to distinguish. A security-semantics version change is not
made here; it would be escalated.

`[FACT]` The asset inventory and its bounds apply only to the listed
`gpt2-seq16` fixture path and its shapes/code parameters; they are not
extrapolated to other models, lengths, batches or production entries.

`[FACT]` The output-binding boundary keeps winner, valid range and tie result
public; the full reduced output row is hidden by the conditional argument in
5.5/5.6, not by a `[GAP]` waiver.

## 10. Open items

- `[UNPROVEN]` A full simulation and a mechanized adaptive composition of the
  hash and algebraic arguments; the linear-cover criterion is argued per view.
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
- `[UNPROVEN]` A local numeric proximity soundness bound for the direct
  FOLD-to-BASE plan (radius 16,384, 603 draws, 12 work bits); the prior
  eight-round ledger term is withdrawn rather than imported. `eps_proximity`
  is carried as `UNKNOWN` in section 6; no total security bit count is
  claimed.
- `[UNPROVEN]` A mechanized composition theorem and an end-to-end 128-bit
  claim; `126.21`/`116.01` remain plan-ledger numbers.
- `[LONG-TERM]` The trusted X source, policy distribution/freshness and
  chain-based commitment/verification are recorded as long-term direction;
  they are not implemented here and are not debts of this round.

No statement in this note claims that the trusted X source is proved, that
weights are hidden under a formal zero-knowledge definition, or that the run
is a production acceptance.
