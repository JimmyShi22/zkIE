# Hidden Dot masks: independent claim mask and public residual

## Scope and status

This note describes the implemented mask construction of the hidden Dot
identity and the joint hiding argument it supports. The construction is
implemented; the argument is stated and exercised by tests, not proved as a
formal zero-knowledge proof. The labels below separate the parts.

- `[FACT]` verified in the code or arithmetic checked against it.
- `[ASSUMPTION]` an explicit condition the argument needs.
- `[ARGUMENT]` a stated argument that is not a formal proof.
- `[GAP]` a missing check or mechanism inherited from the surrounding session.

Production shape: `P = 16` positions, `k = 10` product-sumcheck variables,
base field `p = 2^64 - 2^32 + 1`, extension `E = Fp3` with `|E| = p^3`.

## 1. Construction

`[FACT]` The sampled index-mask message (M1) carries 73 one-time entries in
five disjoint classes:

| Class | Entries | Primary use |
|---|---:|---|
| PSC hiding coefficients `U` | 23 | 23 sent product-sumcheck scalars |
| Shown-value end masks | 2 | two masked end values |
| Claim masks `c_t` | 16 | PSC claim form `sigma = sum_t chi^t c_t` |
| Tilde masks `g_t` | 16 | masked Dot right-hand side `tilde_t` |
| Eval masks `e_t` | 16 | masked Eval targets `tau_t` |

`[FACT]` The claim and tilde entries are independent draws from the same
stage-1 stream: the mask rows are drawn uniformly, the claim slot keeps its
own draw, and no value is copied from the tilde slot into the claim slot. Both
classes are committed in M1 before `chi` is drawn. The 16 per-position
equality rows of the previous schema are removed.

`[FACT]` The second-stage values are:

```text
tilde_t = W_t(psi*) + g_t                         (sent)
sigma   = sum_t chi^t c_t                         (claim, not sent)
R       = sum_t chi^t (g_t - c_t)                 (sent once)
T       = sum_t chi^t tilde_t - R                 (adjusted PSC target)
```

The prover sends exactly one residual `R`, never the individual `d_t`. The
verifier absorbs `R` and its relation description after `chi` and before the
product-sumcheck challenges. The committed rows are:

```text
difference:  sum_t chi^t g_t - sum_t chi^t c_t = R
(U):         <hiding_form, U> + claim_scale sigma = row_target
(E_0), (E_1): map(point) + beta_PSC end_mask_j = v_j
```

The difference row is a public linear claim on the same committed M1 mask
member and is checked by the own-group BAT/FOLD/BASE test. It binds only the
chi-weighted difference, not the individual values, and it does not force
`c_t = g_t`.

`[FACT]` Substituting the definitions cancels the masks exactly:

```text
x_chi . Omega + sigma
  = sum_t chi^t <X_t, Omega> + sum_t chi^t c_t
  = T
  = sum_t chi^t (W_t(psi*) + g_t) - sum_t chi^t (g_t - c_t)
  = sum_t chi^t W_t(psi*) + sum_t chi^t c_t.
```

After cancelling `sigma`, the PSC proves
`sum_t chi^t <X_t, Omega> = sum_t chi^t W_t(psi*)`. The residual is
transported into the public target and does not appear in the proved
relation.

`[FACT]` Canonical order: M1 root and sample answer (stage 1) -> first seal ->
shared stage -> `tilde` -> `chi` -> `R` with the relation description ->
`mu'` -> `eps` -> PSC rounds -> `beta_PSC` -> end values -> own proximity
test. Changing `R` changes `eps` and every later challenge, so a prover and
verifier that disagree on the residual or on its position in the transcript
derive different challenges and fail.

## 2. Every public value

`[FACT]` The verifier sees the following mask-dependent values.

| View | Count | Mask dependence | Rank contribution |
|---|---:|---|---:|
| `tilde_t` | 16 | `g_t` (given the secret `W_t(psi*)`) | 16 |
| `tau_t` | 16 | `e_t` (given the secret `W_t(psi_t)`) | 16 |
| End values `v_0, v_1` | 2 | end masks, `beta_PSC != 0` | 2 |
| Residual `R` | 1 | chi-weighted `g - c` | 1 new form |
| PSC transcript | 23 | affine bijection of `U` for fixed `sigma` | 23 |
| Mask sample answer | 1 | whole mask row, masked by the private non-zero pad | 0 (pad rank 1) |
| Openings | union `SEND ∪ BASE` | encoded symbols masked by the row tail | 0 if union `<= r` |
| Roots, salts, seals | digests | hash hiding | computational |

The 23 PSC sent scalars are the hiding sum (1) plus the 22 non-fixed round
coefficients (20 for the ten plain variables and 2 for the last variable).
The mask rows, tails, salts and pads are never serialized.

The relation rows are public predicates on the committed mask, not extra
field values: the verifier learns that the committed entries satisfy them,
but the entries themselves remain behind the commitment and the sampled
openings.

## 3. Corrected joint-view statement

### 3.1 What is false

`[FACT]` For fixed `g`, the pair `(R, sigma)` is **not** jointly uniform:

```text
R + sigma = sum_t chi^t g_t.
```

The two values lie on one affine line, so they are perfectly correlated for
fixed `g`. A statement of the form "for fixed `g`, `(R, sigma)` is jointly
uniform" is wrong. The correct statement must use the values that are
actually public.

### 3.2 The public pair

`[ARGUMENT]` For a fixed secret witness, the maps
`(g, c) -> (tilde, R)` are

```text
tilde_t = (secret W_t) + g_t
R       = sum_t chi^t g_t - sum_t chi^t c_t.
```

The 16 `tilde` forms span all 16 `g` coordinates. The residual has a non-zero
claim-mask component: the `t = 0` coefficient is `-1` for every `chi`, so no
linear combination of the `tilde` forms can produce it. Therefore the `17`
public forms have rank `17` on the `32`-dimensional `(g, c)` space for every
`chi`, and the kernel has dimension `15`. The kernel is exactly the claim-mask
directions orthogonal to `chi`:

```text
sum_t chi^t (delta c_t) = 0.
```

`[ARGUMENT]` Consequently `(tilde, R)` is uniform over `E^16 x E` for every
fixed witness, and `R` is independent of `tilde`. In particular `R` reveals no
individual `g_t`, no individual `c_t`, and no `W` or `X` value. The fact that
`R` determines `sigma` **given the witness** is not a leak, because `sigma`
is not a public value and the witness is secret.

### 3.3 The transcript does not add a second observation of the claim mask

`[ARGUMENT]` For a fixed adjusted target `T`, the 23 sent product-sumcheck
scalars are an affine bijection of the 23 hiding entries `U`. The `sigma` term
of the claim enters the rounds as
`eps sigma prod_t psi_t(X_t)`. In the hiding-coefficient order, the
polynomial `prod_t psi_t` is the degree-one coefficient of the first variable,
i.e. entry `hp_place(0, 1) = 0`:

```text
w_0(X_0) = sigma psi_0(X_0)  =>  U(X) includes sigma prod_t psi_t(X_t).
```

Therefore a change `sigma -> sigma + delta` is exactly a shift of one hiding
entry, and the image of the 23 sent scalars is unchanged. The transcript has
rank 23 in `U` for every `sigma`; it does not reveal `sigma` and adds no
independent rank on the claim mask beyond the one shared direction already
counted by `R`.

`[FACT]` The unit test
`production_hiding_coefficient_map_is_full_rank` builds the `23 x 23` matrix
from hiding entries to sent scalars for the production shape at a fixed
challenge point and checks rank 23. The check is at one point; the argument
for all challenges remains an assumption.

### 3.4 Rank budget

`[ARGUMENT]` The leakage linear map from the 73 mask entries to the public
mask-dependent values has rank

```text
16 (tilde) + 16 (tau) + 2 (end) + 23 (U -> transcript) + 1 (R's claim direction)
= 58,
```

with a kernel of dimension `15` (the claim-mask directions orthogonal to
`chi`). The `R` direction and the transcript's `sigma` direction are the same
one-dimensional claim-mask direction, so the correlation between `R` and
`sigma` does not add rank. The remaining mask randomness is not observed.

The sample answer and the openings use independent randomness, not entries:

- one non-zero pad with a public non-zero `beta_SEND` masks the answer; the
  map `pad -> beta_SEND pad` is a bijection on `E \ {0}`;
- a row tail of `r` fresh coefficients masks up to `r` distinct opened
  positions; the tail rank must cover the union `SEND ∪ BASE` of all tests on
  that row, not only the SEND count;
- the fresh fold mask (stage 4 for the shared group, stage 5 for the own
  group) is drawn from a separate stream and is independent of the 73
  entries.

### 3.5 Witness independence of the view

`[ARGUMENT]` The mask values `(g, c, U, e, end, pad, tails)` are drawn from
the secret stage-1 stream and are independent of the witness `(X, W, Omega)`.
For a fixed witness the public algebraic values have these distributions:

| Value | Form | Distribution |
|---|---|---|
| `tilde` | `W + g` | uniform over `E^16`, independent of `W` |
| `R` | `L(g) - sigma` | uniform over `E`, independent of `tilde` and `W` |
| transcript | affine in `U` for target `T` | uniform over the accepting tuples |
| `tau` | `W(psi_t) + e` | uniform, independent of `W` |
| end values | `map_j(point) + beta_PSC end_mask_j` | uniform, independent of the maps |
| sample answer | linear form of the mask row plus `beta_SEND pad` | uniform, pad-masked |
| openings | message plus tail polynomial | uniform if the tail covers the union |
| roots | hash commitments | computationally hiding |

Hence the honest view distribution is independent of the witness, up to the
computational hiding of the commitment roots and the commitment scheme's own
tails, salts and pads. The mask root is independent of the witness; the `W`,
shown-map and `Omega` roots are commitment values whose tails, salts and pads
are drawn from the `W` stream before the mask entries and are not part of the
73 entries.

`[ARGUMENT]` The rows do not force a particular witness. Fix the public
algebraic values and two witnesses `(X, W, Omega)` and `(X', W', Omega')`
consistent with the same statement. Starting from an honest mask assignment
for the first witness, define:

```text
g'_t          = g_t + W_t(psi*) - W'_t(psi*)
sigma'        = sigma + sum_t chi^t (W_t(psi*) - W'_t(psi*))
c'_t          = any claim vector with sum_t chi^t c'_t = sigma'
U'            = the unique hiding assignment that reproduces the original
                23 sent scalars under the new witness and sigma'
e'_t          = e_t + W_t(psi_t) - W'_t(psi_t)
end_mask'_j   = end_mask_j + beta_PSC^{-1} (v_j - v'_j)
```

`U'` exists because the hiding map is full rank: the sigma shift is the
first hiding direction (`w_0 = sigma psi_0`), and the product-summand change
from the different witness is another additive shift of the same 23 sent
scalars. The full-rank map has a preimage for the combined shift. The
conditions hold:

- `tilde` is unchanged by the definition of `g'`.
- `R` is unchanged because `sum chi^t g' - sigma' = sum chi^t g - sigma`.
- A claim vector `c'` exists because the `t = 0` coefficient is `1` for every
  `chi`: the equation `sum_t chi^t c'_t = sigma'` has one linear constraint on
  16 variables, so `15` degrees of freedom remain.
- The transcript is unchanged: the `sigma` shift is absorbed by `U'`, and the
  product-summand change from a different witness is absorbed by the same
  full-rank hiding map. The hidden sum, `eps`, every round coefficient and
  every challenge stay identical.
- `tau` is unchanged by the definition of `e'`.
- The end values are unchanged by the end-mask choice; `beta_PSC != 0` makes
  the adjustment possible.
- The tilde, Eval, difference, `(U)` and `(E_j)` rows hold for the new
  assignment: the difference row and the `(U)` row are preserved by the
  algebra above, and the other rows are preserved by construction.
- The public algebraic values are unchanged by construction. The commitment
  roots of the two assignments differ: each is a fresh hiding commitment with
  its own tails, salts and pads, and their distributions are computationally
  indistinguishable under the commitment hiding assumption and the fresh-seed
  requirement. This is not a same-root witness swap; the commitment is
  binding.

The algebraic compensation shows that the rows are consistent with every
witness that satisfies the public statement. The distributions in the table
above show that the honest view does not distinguish those witnesses, up to
the computational hiding of the roots. This is the argument the construction
supports; it is not a formal simulation proof.

### 3.6 Multi-stage view and same-commitment rows

`[FACT]` M1 is fixed in stage 1 before the SEND seed, the mask sample answer
and the SEND openings. `tilde` and `R` are stage-2 values; `chi` is drawn
after `tilde`, and `R` is absorbed before `eps`. The own-group opening is a
stage-2 BASE opening on the same M1 commitment. The sequential composition is
covered by the same rank budget on the union view: earlier public values are
functions of the same mask entries, and later challenges are functions of the
earlier values.

`[ARGUMENT]` The difference row, the `(U)` row and the two `(E_j)` rows are
public linear predicates on the same M1 commitment. They are not extra view
values and they consume no fresh entropy. Each `c_t` and `g_t` has one primary
role (claim form, tilde value) and one consistency use (the difference row);
the one-use statement is "one value, one primary role, one consistency role",
not "one row". The opening union is unchanged because the difference row names
entries that are already covered by the SEND/BASE openings.

### 3.7 Reuse and lifecycle

`[ASSUMPTION]` A fresh secret master seed and fresh commitments are used for
every accepted proof. The durable ledger rejects a repeated commitment root.
The verifier alone has no history and accepts a repeated root.

`[GAP]` Reusing the same mask values under a fresh root (fresh tails/salts) is
not detected by the verifier or by a root-only ledger. If the same `g`/`c`
values meet two runs with different `chi` or `psi*`, differences of `tilde`
and of the difference rows reveal linear equations on the secret values. The
independent-claim-mask construction does not add a new reuse channel beyond
the existing mask lifecycle: its entries are once-per-run values, and the
residual is a deterministic function of those values and the run challenge.

## 4. Error bound

`[FACT]` The chi-batching term. The acceptance condition after the target
adjustment is

```text
sum_t chi^t d_t = 0,   d_t = <X_t, Omega> - W_t(psi*).
```

The difference vector is fixed before `chi`: `g` and `c` are committed in M1
before `chi`, and `R` is bound to them by the difference row. If `d != 0`,
the polynomial `D(chi) = sum_t chi^t d_t` has degree at most `P - 1 = 15`, so

```text
Pr[D(chi) = 0] <= 15 / |E| = 15 / p^3 ~= 2^-188.09.
```

The term is already present in the batched Dot identity of the previous
schema; the new construction makes it explicit and does not add a second
mask-batching term when `R` is bound and the target adjustment is exact.

`[ASSUMPTION]` The proximity term. The own-group test is the existing
BAT/FOLD/BASE test. The change replaces 16 per-position equality rows with one
difference row: the Dot plain rows go from 19 (16 ties + `U` + 2 `E_j`) to 4
(difference + `U` + 2 `E_j`), so the BAT row-count term does not increase. The
SEND sample/opening and the FOLD/BASE round errors are unchanged. Let
`eps_prox` be the union of these terms. Taking the recorded per-round ceiling
of `2^-128.00` with the 12 work bits as a conservative ceiling for the
own-group rounds (the own plan's compiled values would replace it), with
`N_own` own-group rounds `eps_prox <= N_own * 2^-128`. The formal fixture
records 6 own-group FOLD rounds; with the BAT combination, the BASE round and
the sampling terms, `N_own <= 10` covers the own-group proximity accounting.

`[FACT]` The hash term of the chain is the recorded
`(17/2) (t + q_V)^2 / 2^256`, where `t` is the prover's hash calls and `q_V`
the verifier's counted calls.

`[ARGUMENT]` The updated total bound is

```text
eps_total <= (t + q_V + 1) * 2^-128.00
           + (17/2) (t + q_V)^2 / 2^256
           + eps_prox
           + 15 / |E|.
```

With `N_own <= 10`, the conservative union bound is

```text
eps_total <= 2^-126.21 + 10 * 2^-128.00 + 2^-188.09 <= 2^-124.25.
```

The recorded per-round chain bound in `docs/lin-bind-security.md` remains
`2^-126.21` with the work bits; the chi term is negligible. The 128-bit target
is not claimed by the conservative union bound; the per-round comparison basis
of the recorded chain accounting is unchanged. The hiding bound is separate
and is not part of these soundness figures.

## 5. Negatives and evidence

The following tests reach the verifier or the exact plaintext relation check:

- changed residual `R`, opposite sign convention, missing residual, residual
  without a Dot proof, and a residual that the prover does not absorb before
  the product-sumcheck challenges;
- changed claim entry, tilde entry, hiding entry and end entry against the
  plaintext relation check, and a wrong residual in the checked binding
  (which changes the difference-row target and the adjusted PSC target);
- changed mask root, sample seed, sample answer and opening symbols against
  the public verifier;
- missing mask message, missing Dot proof, missing message plan, a plan whose
  entry allocation merges or shortens a class, and a derived-row count that
  is not exactly the difference row plus the `(U)` and two `(E_j)` rows;
- canonical encoding: the residual survives a round trip, trailing bytes are
  rejected, and appending an individual `d_t` encoding is rejected;
- rank checks: the public `(tilde, R)` forms have rank `P + 1`, and the
  production hiding-coefficient map has rank 23.

## 6. Limitations

- The joint argument is a stated argument, not a formal proof. The hiding
  composition with the commitment sampling, FOLD/CS/BASE and the hash model is
  not mechanized.
- The hiding-coefficient rank is tested at one deterministic challenge point
  for the production shape; the all-challenge statement is an assumption.
- The tail-union check (`|SEND ∪ BASE| <= r`) is still a plan/ledger gap in
  the surrounding session.
- The Eval mask is drawn independently, but a value equality `e_t = g_t` is
  not rejected; the honest-prover draw is the basis.
- The durable ledger is exercised by the integration fixture, not yet by the
  production command path; value reuse under a fresh root is undetected.
- The output relation and the table/asset correspondence remain trusted-plan
  items outside the mask construction.
