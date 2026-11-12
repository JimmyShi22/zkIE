# Closed output-head relation: unbiased sampling, joint views and the whole-chain argument

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
The committed document is version 6.

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
following mapping and events:

1. The simulator chooses the salt and the message for every leaf, commits the
   opened leaves with the real leaf hash, and for every unopened leaf chooses
   a uniform digest and programs the random oracle at that leaf input only
   when the adversary queries it. Inner nodes are computed from the children;
   the root is the resulting digest.
2. The verifier's view is the root, the opened leaves and the answers and
   transcripts. The unopened leaves' symbols and salts are not in the view.
   The unopened digests are uniform in the ideal RO model, so the root is
   consistent with the opened leaves and independent of the unopened inputs.
3. The simulation can fail only through one of these events:
   - two distinct leaf or node inputs collide in the RO (including a
     programming inconsistency at the same input): probability at most
     `(q + 1)^2 / 2^256` over `q` queries;
   - the adversary guesses an unopened leaf's exact input (salt and message):
     the salt is 40 bytes, so the guessing probability is at most
     `q / 2^320`; the tail and pad entropy give the message min-entropy the
     commitment needs.
4. Adaptive queries. The adversary may query the RO, the root and the
   transcript before choosing its openings or challenges. The simulator
   answers every query consistently. Conditional on the opened leaves, the
   unopened leaf outputs and the root are uniform, so the challenge derived
   from them is independent of the unopened symbols. The adaptive composition
   is therefore covered by the same two events; no separate "conditioning on
   the root" step is used.

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
| audited proximity/LKP rounds | 8 rounds from the plan | `8 * 2^-128 = 2^-125` |
| range booleanity | 16 positions, degree at most 166 each | `16*166/p^3 < 2^-180` |
| PSC sumcheck (Dot once, closure 16 positions) | 10 and 23 variables, degree 3 each | `(30 + 16*69)/p^3 < 2^-181` |
| single equality | degree 2 | `2/p^3 < 2^-190` |
| `chi = 0` | one event | `1/p^3 < 2^-191` |
| challenge-draw zero fallback | at most 6 rejected words per draw, `(1-p/2^64)^6` | below `2^-192` per draw |
| mask-coordinate retry exhaustion | at most 64 attempts, `(1-p/2^64)^64` | below `N * 2^-2048` over `N` coordinates |
| hash model | plan-ledger term | `(t + q_V + 1) * 2^-128 + (17/2) * (t + q_V)^2 / 2^256` |

`[ARGUMENT]` `CODE` The component union bound used here is

```text
eps_total <= eps_hash + eps_components
eps_hash = (t + q_V + 1) * 2^-128 + (17/2) * (t + q_V)^2 / 2^256
eps_components = 8 * 2^-128 + 2^-180 + 2^-181 + 2^-190 + 2^-191
```

The `t`/`q_V` factor is carried by the hash term; the algebraic component
count is fixed per proof. If the prover uses `Q_A` adaptive retry attempts,
multiply the component terms by `Q_A`; the hash term already carries the
transcript query count. For `Q_R` independent requests, multiply the
component sum by `Q_R`; no global bound is claimed without a bound on `Q_R`.
The challenge draws are uniform on the cubic field except for the zero
fallback term above. The hash term assumes the plan's RO/CR model and work
bits; it is not re-derived here.

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
challenges, fold start and rounds, base positions and lambda, seal. Each
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

**Proximity/opening.** The audited proximity/LKP rounds contribute
`8 * 2^-128`; the opening sets satisfy `q <= r` by the ledger (table 603,
W/closure and mask at most 4, minimum remaining 252) and the tail union is the
accounted union. These terms are independent of the difference polynomial and
are added by the union bound.

**Combination conditions.** The union bound requires that the challenge
spaces are independent and that the difference is fixed before each challenge;
both hold by the order above. The bound is

```text
2 * (2 * 14) / p^3 + 2 / p^3 + 8 * 2^-128  <  2^-124.9,
```

dominated by the audited proximity term. This is a loose bound under the
degree assumption, not a measured probability. A final-verifier exact
negative for this tamper is not implemented because the hidden prover has no
message-injection point after the tree commit; the component test (the group
table changes) plus this bound are what the current API supports.
`[UNPROVEN]` The exact final-verifier negative; the kernel difference is not
claimed to be caught.

## 9. Engineering boundaries and wording

`[FACT]` The joint-view module is test-only. It exercises the real encoder,
the real hiding polynomial and the real fold/base paths, but a passing test
module is not a production runtime detector for every bad rank or every
tampered message. The workspace result and the three seq16 runs are the
delivery evidence reported by the implementation side; they are not an
independent external verification.

`[FACT]` The production proof code changed only in the mask sampling source
(section 2.1); the protocol relations, commitments, challenge order,
parameters and encoding format are unchanged. The proof content and hash
change because the prover's random values change.

`[FACT]` `NUM` Compatibility evidence. The new proof length is
9,939,144 bytes and its sha256 is `f8f512c0...3215f5` in the three seq16
runs. The rejection sampling only changes a draw when a candidate word is at
or above the modulus; this deterministic fixture produced no such word, so
the proof content is byte-identical to the previous round. The sampling
distribution is still the unbiased one; the boundary and retry tests exercise
the rejection path, and the fixture outcome is reported as measured, not
assumed. The
encoding version and the hidden schema version are unchanged because the
message structure is unchanged. A proof produced before the sampling change
(`9,939,144` bytes, the previous round's dump) was decoded and verified by the
new verifier through the bytes entry: `loaded_proof_ok=1`, with the same
`9,939,144` byte length. The verifier never re-derives the prover's mask
values, so the changed sampling does not invalidate old proofs under the same
plan and public statement. This is stronger than the expected "old artifact
does not verify"; it is recorded as measured, not assumed.

`[FACT]` `NUM` Sampling tests: boundary `p-1` accepted; `p` and `2^64-1`
rejected and retried; retry exhaustion returns an error after the fixed
number of attempts; sequential and random-access draws agree; the same seed
replays identically and different seeds give different streams; the sample
has no repeated values; cross-request material/seed/commitment reuse is
rejected by the existing ledger tests; the full prover/verifier suite and the
three seq16 runs pass.

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
- `[UNPROVEN]` A mechanized composition theorem and an end-to-end 128-bit
  claim; `126.21`/`116.01` remain plan-ledger numbers.
- `[LONG-TERM]` The trusted X source, policy distribution/freshness and
  chain-based commitment/verification are recorded as long-term direction;
  they are not implemented here and are not debts of this round.

No statement in this note claims that the trusted X source is proved, that
weights are hidden under a formal zero-knowledge definition, or that the run
is a production acceptance.
