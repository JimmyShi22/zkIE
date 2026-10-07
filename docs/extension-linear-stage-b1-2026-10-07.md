# Claim-driven linear-op reducer — stage B1 (2026-10-07)

Extension-field claim-driven reducer for the elementwise linear ops
(`Scale`, `Add`, `AddConst`, `Transpose`), the first per-family slice toward a
no-forward-recomputation GPT-2 verifier.

## Scope

- Included: single-op relations for Scale (EXACT — `shift == 0`), Add,
  AddConst (public constant in the statement), Transpose (MLE variable
  permutation, LSB-first convention). Full-EF output point sampled from the
  transcript; input claims mapped by linearity/permutation; all leaves
  WHIR-authenticated against the statement roots.
- Excluded (documented, not pretended): `Scale` with `shift > 0` (rounded
  division is piecewise — belongs to the affine/Projection reducer), any
  multi-op chain, any GPT-2 full-model claim, the existing composer path.
- The existing `compose.rs` verifier and its forward recomputation are
  unchanged; this module is a separate path.

## Design

- **Statement** (`LinearStatement`): op kind (public scalars included), input
  shape `m x k`, roots `[a, b?, out]`. No unbound caller data: every scalar
  and root is bound by the transcript.
- **Transcript** (Poseidon2 FS, `common/transcript.rs`): domain-separated by
  op-kind label, shapes, roots, and public scalars (factor/shift, constant).
  The output point is sampled from the transcript; the claimed output value
  is absorbed BEFORE any opening verification; the input claims are absorbed
  after the output claim for transcript continuity only. Tamper rejection
  does NOT rely on that absorption — it derives from the PCS verified-value
  equality (`verified == claimed`) enforced on every opening.
- **Proof** (`LinearProof`): the WHIR openings of the output and the sources;
  the claimed values are shared between WHIR authentication and the linear
  check. The verifier checks `verified == claimed` for every opening and then
  the exact relation in EF: `y = a + b`, `y = a + c`, `y = factor * a`, or
  `y = a` at the permuted point (Transpose: output `k x m` has `m`-vars low,
  so `input_point = out_point[k-part ++ m-part]`).
- **Verifier** takes statement + proof only: no witness, no forward
  recomputation, no `open_*` (asserted by `open_stats` staying flat).

## Files

- `crates/zkie-ops/src/extension_linear.rs` — `LinearOpKind`,
  `LinearStatement`, `LinearProof`, `prove`, `verify` + 13 tests.
- `crates/zkie-ops/examples/extension_linear.rs` — tiny runnable demo.
- `crates/zkie-core/src/common/transcript.rs` — added `absorb_base` (public
  scalar binding).

## Tests (all green, release)

13/13 in `extension_linear`: honest roundtrips for all four kinds (including
non-square Transpose `4x8`), output point genuinely non-base (nonzero
extension coordinate), claims equal independent reference `eval_ef`
evaluations (including the Transpose permutation), verifier never opens,
and rejection for: tampered output claim, tampered input claim (rejected via
the PCS verified-value equality, not transcript binding — see above),
tampered statement root, tampered statement scalar (factor), swapped
Transpose shape, rounded Scale (`shift > 0`), huge dimensions (usize high
bit), missing second source for Add, an attacker-supplied second opening on
each non-Add kind, and an Add proof missing its second opening.

Full suites: zkie-core lib 26/26, zkie-ops lib 71/71 (58 + 13), workspace
`--all-targets` check clean.

## Security statement

WHIR instances carry a PCS target of 90 bits with PoW budget 0 (more
queries, no grinding). Correctness prototype: no end-to-end soundness claim;
the FS reduction is not yet analyzed. Relation soundness here is exact-field
arithmetic — the linearity checks are identity checks in EF, and every value
they use is either a WHIR-verified opening or a transcript-bound public
scalar.

## Next slices (roadmap, not done here)

B2 (EF logUp lookups: gelu/softmax-exp tables), B3 (LayerNormCentered),
B4 (Softmax), then the claim-graph orchestrator over the GPT-2 op list. Until
those land, the existing recompute verifier remains the default.
