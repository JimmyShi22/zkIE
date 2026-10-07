# Extension-field GKR chain — stage 1 (2026-10-07)

Two-shard GKR claim chain connected to quadratic-extension WHIR, with tensor
storage in base Goldilocks. Linear matmuls only: `Y = H @ W2`, `H = X @ W1`.

## Scope

- Included: the two-matmul chain, full-extension-field challenges/rounds/
  terminal claims, WHIR openings at full EF points, verifier = statement +
  proof only.
- Excluded (later stages): affine rounding, logUp/nonlinear ops, fanout
  merging (same_poly), multi-commit batching, end-to-end soundness analysis.
- The existing general composer (`compose.rs`) and its forward recomputation
  are unchanged; this module is a separate path.

## Protocol

1. **Public statement**: dimensions `(m, d, k, n)` (powers of two, >= 2) and
   the canonical commitment roots `[X, W1, H, W2, Y]`. The verifier takes the
   statement and the proof only — no witness store, no `prover_data`.

   **Dimensional restriction**: each dimension must be a power of two >= 2
   AND every derived tensor arity must be >= 5 (`log2 m + log2 d >= 5`, etc.) —
   arities below the WHIR folding factor (5) are rejected by the fallible
   constructors. The stage-1 fixture uses `m=4, d=8, k=8, n=16` (tensor
   arities 5–7).
2. All five tensors are committed (base field, single-point protocols) before
   the transcript runs.
3. A Poseidon2 Fiat–Shamir transcript (`common/transcript.rs`) is
   domain-separated by protocol label, dimensions, and all five roots. The
   output point for `Y` is a full `EF` point sampled from it (coordinate
   order: `v ++ u` — columns first, matching the LSB-first flat-index
   convention), and the claimed `Y` evaluation is absorbed before the rounds.
4. Shard 2 (`Y = H @ W2`): product sumcheck over the shared index. EACH round
   message is absorbed BEFORE that round's `EF` challenge is sampled.
   Terminal claims on `H` and `W2` are full `EF` points/values, absorbed into
   the transcript, and authenticated by WHIR openings against the statement
   roots. Claim points (each in the tensor's own low-bit-first variable
   order): `H` at `z ++ u`, `W2` at `v ++ z` (`z` = shard-2 challenges).
5. Shard 1 (`H = X @ W1`): reduced at the SAME `H` claim — identical `EF`
   point and value (the chain's output claim IS shard 1's input claim).
   Terminal claims: `X` at `r ++ u`, `W1` at `z ++ r` (`r` = shard-1
   challenges), WHIR-authenticated.
6. Sumcheck finals are enforced twice: the product of the terminal claims
   must equal the folded round evaluation, and each terminal claim must equal
   the WHIR-verified opening value at the same point. No EF claim is ever
   downcast to base; no same_poly merge is needed for this single-consumer
   chain.

## Files

- `crates/zkie-core/src/common/transcript.rs` — EF-capable Poseidon2 FS
  transcript (deterministic; challenges derive only from absorbed content).
- `crates/zkie-core/src/common/sumcheck.rs` — product-sumcheck round algebra
  generalized to any field (`RoundPolyF`, `product_round_f`, `fold_f`,
  `prove_product_f`, `verify_product_f`); the base-Goldilocks entry points
  call the shared circuit inline with no intermediate vector conversions.
  `product_round_f` uses the direct coefficient formula (c0 = Σa0·b0,
  c1 = Σa0·(b1−b0)+(a1−a0)·b0, c2 = Σ(a1−a0)·(b1−b0)) — no per-round
  inverse; a reference-equivalence test pins the Lagrange identity.
  **Reuse, not duplication**: the degree-2 round circuit is shared between
  the base and EF instantiations.
- `crates/zkie-core/src/common/mle.rs` — `eval_ef` / `partial_eval_ef`: MLE
  evaluation of base-stored tables with EF points (lift only the arithmetic).
- `crates/zkie-core/src/common/field.rs` — `EF = BinomialExtensionField<
  Goldilocks, 2>` alias; `BasedVectorSpace` re-export.
- `crates/zkie-core/src/pcs/whir.rs` —
  `Whir::new_target(num_variables: usize, security_level: usize, pow_budget: usize)
  -> Option<Self>` is genuinely fallible (tiny arities below the folding
  factor, PoW schedules exceeding the budget, FFT domains above 2^32
  elements) — never panics; the legacy `new`/`new_testing` constructors keep
  their previous behavior. `pow_bits_ok`/`max_pow_bits`, `open_ef`/
  `verify_ef` (full-EF prescribed points, no downcast, commitment-bound
  transcripts; `verify_ef` pre-validates point arity and the canonical
  single-opening protocol before upstream asserts), `Point` re-export.
  Note: the main PCS was **already** EF2 internally (quadratic-extension
  challenge field); what is new here is exposing prescribed openings at full
  EF points without base embedding — not a change from base-only WHIR.
- `crates/zkie-ops/src/extension_chain.rs` — `ChainStatement`, `ChainWhir`,
  `ExtensionChainProof`, `prove`, `verify` + 17 tests. **New orchestration**:
  the transcript ordering, the two chained reductions sharing one `H` claim,
  and the WHIR binding are new code on top of the reused primitives. Both
  `prove` and `verify` validate all five tensor sizes with checked
  arithmetic (huge power-of-two dimensions return `None`/`false`, never an
  overflowing multiply), and `verify` additionally checks the derived
  per-tensor arities against the `ChainWhir` instances before any transcript
  or algebra runs.
- `crates/zkie-ops/examples/extension_chain.rs` — tiny runnable demo
  (prove/verify timings and per-instance open/verify stats).

## Tests (all green, release)

`cargo test --release -p zkie-ops --lib extension_chain` — 17/17:
honest roundtrip (non-square power-of-two `m=4, d=8, k=8, n=16`), challenges
genuinely non-base (nonzero extension coefficient, asserted on every sampled
round challenge), terminal claims equal independent reference EF MLE
evaluations, verifier never opens (`open_stats` flat across verify), and
rejection for: tampered round coefficient, tampered terminal claim, tampered
Y claim, tampered boundary root (wrong statement root), swapped openings,
reordered rounds, truncated proof, wrong statement shape, spliced shard-2
proof from a different witness pair, huge power-of-two dimensions (usize
high bit), zero-round proofs, and unsupported `ChainWhir` configurations
(tiny arity, insufficient PoW budget); plus an isolated full-EF-point WHIR
open/verify regression (arities 7 and 8). Witness generation in tests uses a
deterministic XorShift; all protocol challenges come from the transcript.

Full suites: zkie-core lib 26/26 (including `new_target_is_fallible` and
`verify_ef_rejects_malformed_metadata`), zkie-ops lib 58/58, workspace
`--all-targets` check clean.

## Tiny example measurement (independently rerun)

`cargo run --release -p zkie-ops --example extension_chain`, dims
`m=4, d=8, k=8, n=16`, PCS target 90 bits, PoW budget 0:

```
prove 7.871 ms | verify 0.434 ms | ok=true
```

This is a tiny correctness-prototype timing, NOT full-model performance and
NOT a parity guarantee against any other PCS.

## Security statement

WHIR instances carry a **PCS target of 90 bits** with a caller-chosen PoW
budget (0 in the tests/example — no grinding, more queries). This is a
**correctness prototype**: no claim of total 90-bit soundness is made. The
sumcheck round soundness (EF challenges, degree-2 round polys), the
Fiat–Shamir reduction, and the commitment-root binding in the transcript are
not yet analyzed end-to-end. Known open items: fanout merging (same_poly)
when a tensor is consumed by more than one reduction, multi-open batching to
amortize the five openings, and per-round FS analysis.

## Debugging note (convention trap fixed)

The MLE convention is LSB-first on the flat index (columns are the low-order
variables). Claim points are written in each tensor's own variable order —
`Y: v++u`, `H: z++u`, `W2: v++z`, `X: r++u`, `W1: z++r` — and row-fixing uses
transposes (`H^T`, `X^T`) so `partial_eval` fixes rows. The WHIR boundary
reverses the whole point exactly once (p3 is MSB-first). The prover/verifier
point orders are covered by the tamper tests and the reference-eval test.
