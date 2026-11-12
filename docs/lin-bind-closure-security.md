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
The committed document is version 8.

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
| 10 | BASE star main/mask rows and tails | BASE target, combined rows and star openings | target, rows and openings of the same star | fresh one-time pads on the target dot product, each combined coordinate and each tail coordinate | lambda and position checks | `NUM` real star path, `CODE` |
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

The star cancels. The relation left after elimination is the approved public
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

`[FACT]` `CODE` `MATH` Specific hybrid for opened and unopened leaves. The
standard salted-Merkle random-oracle commitment lemma applies with the
following simulator, events and mapping:

1. **Query count.** `q` is the cumulative number of random-oracle queries and
   calls made by the adversary and the simulator up to the event, including
   leaf, node, root, challenge and opening queries. All bounds below use this
   cumulative `q`.
2. **Simulator inputs.** The simulator receives the public statement and plan,
   the random-oracle interface, and the algebraic hiding-map randomness it
   needs to choose the opened leaf values. It does **not** receive the opening
   set, and it does not precompute the leaves from the prover's witness: the
   opening positions are produced only after the roots and the Fiat-Shamir
   challenges.
3. **Commitment phase.** For every leaf it chooses a salt and a uniform
   digest `com_i` independently (lazily sampled); it keeps the pair
   `(salt_i, com_i)` and builds the inner nodes and the root from the
   digests. In the ideal RO model `com_i = H(salt_i || leaf_i)` is uniform and
   independent of the leaf, and the root is a hash of uniform digests.
4. **Query phase.** The simulator maintains a random-function table. A query
   to a new input is answered with a fresh uniform value and recorded; a
   repeated query to the same input returns the recorded value. This is
   consistent with the real random oracle.
5. **Adaptive opening phase.** After the root and challenges, the adversary
   chooses openings. For an opened leaf the simulator chooses its salt and,
   from the algebraic hiding simulation, its leaf value; it programs
   `H(salt || leaf) = com_i` and reveals `(salt, leaf)`. The path recomputes
   to the root because `com_i` is the digest used in the root. If the
   adversary had already queried the exact input `salt || leaf`, the simulator
   had returned a random value; matching it requires guessing the input (bad
   event). The opened leaf values are covered by the algebraic hiding
   argument, not by a per-leaf fresh pad.
6. **Bad events.** The simulation can fail only through:
   - a collision or a programming inconsistency between two distinct inputs
     in the random oracle, probability at most `(q + 1)^2 / 2^256`;
   - the adversary guessing an unopened or not-yet-open leaf's exact input
     (salt and leaf), probability at most `q / 2^320` for the cumulative
     query count `q`; the salt is 40 bytes, and the tail/pad entropy gives the
     message min-entropy the commitment needs.
   If neither event occurs, the simulated view is identical to the real view,
   including the adaptive openings and challenge queries. The challenges
   derived from the root/transcript are random-oracle outputs independent of
   the unopened leaf values; no separate "conditioning on the root" step is
   used.

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
value. If the claim itself is a function of `Y`, it is the approved public
claim (the relation the verifier checks), not the raw row. The combined rows
and the target alone leave the star underdetermined; the openings reveal
encoded star symbols masked by the star tail (`q <= r`) and the old row
symbols masked by the old tail. The star is a fresh one-time pad from its own
stream; the two-seed test is illustrative, not the uniformity proof.

`[UNPROVEN]` The multi-oracle conditional composition is not mechanized; the
argument above is the per-round conditional/block derivation, and the
production-size tests are local evidence for their blocks.

### 5.9 BASE target source and the joint mask/star view

`[FACT]` `CODE` The BASE target is not a function of the hidden row. The
actual target sources are:

- `record_row` and `column_row`: the target is the public table record or the
  public column value combined with the public `eq_gamma` weights;
- `sample_row` and `oracle_sample_row`: the target is the received SEND
  answer, which is the public sampled value plus its one-time pad;
- `fold_plain_row`: the target is the public `G_u`/`Omega` relation target
  combined with the public `psi` and `zeta` challenges;
- the BAT row target is the `row.target` above; the group/fold claim
  coefficients are the public relation coefficients and the public batching
  challenge.

The hidden reduced row `W`/`Y` enters the oracle message values (the
prover's witness), not the targets and not the public BAT relation
coefficients. The target values are therefore determined by the public
table/reader data, the public SEND answers and the public statement. This is not an expansion of the approved output boundary:
the target is a protocol message value derived from public data and masked
answers, not a new Y-dependent output. If a target were Y-dependent, it would
be a leak; none of the traced row families is. The handoff target is supplied
by the caller's verified FOLD output; the caller/anchor trust is a deployment
assumption, not part of this derivation.

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
larger than `2^-192`. The conservative bounds use the looser powers:
`1/p^3 < 2^-191`, `30/p^3 < 2^-187`, `1134/p^3 < 2^-181`,
`16*166/p^3 < 2^-180`, `2/p^3 < 2^-190`.

The hiding-rank maps have constant nonzero determinant (section 5.4), so the
former bad-challenge rank term is withdrawn: there is no Schwartz-Zippel
event for the PSC or FOLD hiding maps. The remaining degree terms are
soundness checks of the sumcheck and range relations, not hiding-rank terms.
The entries are:

| Component | Count/degree | Bound |
|---|---|---:|
| direct shared FOLD check | k <= 11, 11 rounds, degree 2 | `22/p^3 < 2^-187` |
| direct own FOLD check | k = 14, 14 rounds, degree 2 | `28/p^3 < 2^-186` |
| BASE lambda check | one Fp3 challenge, linear polynomial (`d = 1`) | `1/p^3 < 2^-192` |
| BASE symbol checks | two position checks, linear each | `2/p^3 < 2^-191` |
| base position sampling | `s = 2` codeword-index positions; distance parameter `delta` | miss `1 - (1 - delta)^s`; local `delta` convention not derived, in `eps_proximity` |
| range booleanity | 16 positions, degree at most 166 each | `16*166/p^3 < 2^-180` |
| PSC sumcheck (Dot once, closure 16 positions) | 10 and 23 variables, degree 3 each | `(30 + 16*69)/p^3 < 2^-181` |
| single equality | degree 2 | `2/p^3 < 2^-191` |
| `chi = 0` | one event | `1/p^3 < 2^-191` |
| challenge-draw zero fallback | at most 6 rejected words per draw, `(1-p/2^64)^6` | below `2^-192` per draw |
| mask-coordinate retry exhaustion | at most 64 attempts, `(1-p/2^64)^64` | below `N * 2^-2048` over `N` coordinates |
| hash model | plan-ledger term | `(t + q_V + 1) * 2^-128 + (17/2) * (t + q_V)^2 / 2^256` |

`[ARGUMENT]` `CODE` The component union bound used here is

```text
eps_total <= eps_hash + eps_components + eps_proximity
eps_hash = (t + q_V + 1) * 2^-128 + (17/2) * (t + q_V)^2 / 2^256
eps_components = 2^-180 + 2^-181 + 2^-186 + 2^-187 + 2^-189
                (BASE lambda 1/p^3 + BASE symbol 2/p^3 + single equality
                 2/p^3 + chi 1/p^3 merge to 6/p^3 < 2^-189.4; the shared and
                 own FOLD terms merge to 50/p^3 < 2^-186.4; the loose bounds
                 above are used)
eps_proximity = UNKNOWN
eps_total also carries the retry exhaustion N * 2^-2048 and the Q_R / Q_A factors
```

`eps_proximity` is an explicit unknown, not zero: the local base-position and
proximity distance/query soundness is not derived here. The document does not
claim a total security bit count. The `BASE 3/p^3` figure of the previous
version is split: the lambda challenge is one Fp3 challenge with a linear
check (`d = 1`), and the two position checks are the symbol checks; the
previous single `2^-190` merged term is replaced by the explicit
`6/p^3 < 2^-189.4` merge above, which does not fit under `2^-190`.

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
(shared) and k = 14 (own); the BASE lambda is linear and the two base
positions are symbol checks. The prior "8 rounds each <= 2^-128" term is a
multi-stage ledger figure that is not mapped to this direct FOLD-to-BASE
compiled plan; it is withdrawn from this chain's bound and replaced by the
per-round degree terms above.

`[FACT]` `CODE` The BASE lambda is one challenge in the cubic field and the
check is linear in it (`d = 1`), so the Schwartz-Zippel term is `1/p^3`; the
general form is `d/p^3` if a check of degree `d` is used. The two base
positions are codeword-index samples: the plan's `plan_loc` convention draws
`t` positions and charges the agreement bound raised to `t` (`agree^t`); in
the distance form the miss probability is `1 - (1 - delta)^s` for `s`
positions at relative distance `delta`. The local plan exposes `s = 2` but
does not turn its distance parameter into a numeric agreement/distance term in
this document, so the base-position and proximity soundness is the explicit
`eps_proximity = UNKNOWN` above, not zero and not the withdrawn ledger term.
The retry exhaustion (`N * 2^-2048`) and the multi-request `Q_R`/adaptive
`Q_A` factors are carried separately; they are not dropped.

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

**BASE lambda and symbol checks.** The lambda check is linear and adds
`1/p^3`; the base symbol checks are linear in the drawn positions.

**Proximity/opening.** The direct FOLD checks are the per-round degree terms
above (22/p^3 shared, 28/p^3 own); the BASE lambda and the two base-position
symbol checks add 3/p^3. The opening sets satisfy `q <= r` by the ledger
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
(2 * 14 + 2 * 11) / p^3 + 3 / p^3  <  2^-185,
```

plus the conditional proximity distance/query term above. This is a loose
bound under the degree assumption, not a measured probability. A
final-verifier exact negative for this tamper is not implemented because the
hidden prover has no message-injection point after the tree commit; the
component test (the group table changes) plus this bound are what the current
API supports.

**Kernel scope and binding derivation.** If the changed message's difference
is in the kernel of the group-row map, the FOLD/BASE checks do not see it.
The conditional conclusion is proved as follows under the stated binding
assumptions:

1. The verifier's public claim and target are functions of the committed
   roots and the public statement, not of the hidden row directly (section
   5.9). A false statement requires a different public claim or target.
2. If the changed message changes that public claim or target, its difference
   is nonzero in the group-row map (otherwise the public value is unchanged);
   the FOLD/BASE checks see a nonzero difference polynomial, and the
   challenges are drawn after the difference is fixed, so the polynomial
   bound applies.
3. If the difference is in the kernel, the public claim and target are
   unchanged. The verifier accepts the same public statement as for the
   original committed message. A kernel difference is a different private
   witness for the same committed claim, not a false-statement acceptance,
   provided the original statement/root is the true committed one.
4. The root/leaf index/run bind the committed message under the CR hash
   assumption: a different message with the same root is a collision or a
   hash-model event (`(q+1)^2/2^256` in the RO analysis). If the changed
   message also changes the root or the statement, the binding checks reject
   it at the root/statement layer.

The residual assumptions are: the original statement/root is the true
committed one, the claim/target really are bound to the committed
statement/roots (the wiring is not independently trusted; a deployment
assumption), and the concrete hash's CR advantage is not quantified. The
document does not claim that the FOLD/BASE layer detects a kernel difference;
it claims that a kernel difference does not create a false-statement
acceptance under these binding assumptions. `[UNPROVEN]` The exact
final-verifier negative for the combined tamper and the concrete-hash binding
bound.

## 9. Engineering boundaries and wording

`[FACT]` The joint-view module is test-only. It exercises the real encoder,
the real hiding polynomial and the real fold/base paths, but a passing test
module is not a production runtime detector for every bad rank or every
tampered message. The workspace result and the three seq16 runs are the
delivery evidence reported by the implementation side; they are not an
independent external verification.

`[FACT]` The production proof code is unchanged in this round; the v7/v8
changes are test-only and document. The earlier mask sampling source change
(section 2.1) changed only the prover's random values; the protocol
relations, commitments, challenge order, parameters and encoding format are
unchanged. The proof content and hash are the measured fixture result from
that source.

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
