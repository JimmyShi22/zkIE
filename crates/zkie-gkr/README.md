# zkie-gkr

Minimal, self-contained prototype of the **GKR / sum-check** route for proving
one matmul layer, as a first step toward "verifiable inference" that is matched
to the ML computation structure rather than a generic Halo2 chip ISA.

## What it proves

`C = A @ B` for field-element matrices (powers of two), via a single sum-check
over the contraction index:

```
C_tilde(u, v) == sum_w A_tilde(u, w) * B_tilde(w, v)
```

The proof is a degree-2 sum-check (`log k` rounds, three coefficients per
round), plus MLE evaluations of `A` and `B` at random points.

## Why it matters

The current `zkie-core::chips::dot_general::DotProductChip` expands every output
element into `K + 184` Plonkish rows (`184 = 64 + 60 + 60` range checks), so a
full matmul is `O(m*n*k)` rows. GKR keeps the *proof overhead* at `O(m*k + k*n)`
field ops: the actual `O(m*n*k)` matmul is still computed (to commit `C`), but
it is no longer re-expanded as a constraint system.

Measured on this laptop (naive, single-threaded, unoptimized):

```
matmul m=512 k=512 n=512
gkr_proof_field_ops = 527,360
halo2_rows          = 182,452,224
ratio               = 346x
```

## Field choice: Goldilocks (64-bit) for int16

The field is **Goldilocks** (`p = 2^64 - 2^32 + 1`). For large-model inference
we target `int16` fixed-point: products are 32-bit and a length-K dot product
adds `log2(K)` bits, so 64-bit arithmetic holds the accumulation with plenty of
margin. `p - 1 = 2^32 * (2^32 - 1)` gives the 2-adic subgroup FRI needs.

Implemented and tested:

- `merkle.rs` — Merkle commit/open/verify over field elements (hash is a
  deterministic placeholder; swap in Poseidon/Blake3 for production).
- `fri.rs` — the FRI building blocks: multilinear-to-univariate coefficient
  lift, Horner evaluation, LDE evaluation, LDE commitment, and the
  degree-halving fold.
- `lookup.rs` — a LogUp (log-derivative) lookup argument: proves
  `y_i = table[x_i]` as a parallel sum-of-rationals check, no permutation/sort.
- `softmax.rs` — quantized softmax as `exp lookup + reduction + division`.

## What is *not* here yet (deliberately)

- **FRI opening (evaluation proof)**: the verifier currently recomputes the MLE
  evaluations in `O(k)`. The fold/commit primitives are in place; the remaining
  piece is the DEEP-FRI evaluation proof that turns those into `O(polylog k)`
  openings.
- **Soundness field size**: Goldilocks is 64-bit; production soundness still
  needs a larger field, an extension field, or more FRI queries.
- **Softmax / activations**: not yet; those reduce to lookup arguments (Lasso /
  LogUp), which compose cleanly on top of this sum-check.
- **Folding across layers**: not yet; a transformer's repeated layers map to
  SuperNova/HyperNova steps around this per-layer matmul.

## Layout

- `src/field.rs` — Goldilocks (64-bit) field + xorshift PRNG.
- `src/mle.rs` — multilinear extension evaluation.
- `src/sumcheck.rs` — degree-2 sum-check prover/verifier.
- `src/matmul.rs` — single-layer matmul reduction.
- `src/merkle.rs` — Merkle commitment.
- `src/fri.rs` — FRI primitives (lift, LDE, fold).
- `src/lookup.rs` — LogUp lookup argument.
- `src/softmax.rs` — softmax via exp lookup.
- `examples/bench_matmul.rs` — the Halo2-vs-GKR comparison.

## Run

```bash
cargo test -p zkie-gkr
cargo run -p zkie-gkr --release --example bench_matmul
cargo run -p zkie-gkr --release --example bench_softmax
```
