# Committed-path issues (store-free verification follow-up)

Branch `feature/verify-no-forward`. The committed shard-dag path
(`prove_committed_shard_dag` / `verify_committed_shard_dag`) is root-only,
serialized (P3/P4), and the weight commitment is now split into a one-time
`commit_model_weights` (commit + cache) phase plus a prove phase that only
restores prover data from cache (no re-commit).

## Resolved

1. **`OpeningProtocol` / `Whir` not serde** — `Whir` is serialized as
   `(num_variables, security_level, pow_bits)` and reconstructed; the opening
   protocol is rebuilt from the tensor arity.
2. **mmap weight `Store::get` panic** — `commit_model_weights` uses
   `store.materialize(t).into_owned()`.
3. **Weight commitment used `new_testing` (32-bit)** — now `Whir::with_params(
   arity, 90, pow_bits=17)` plus horizontal merge, borrowed from
   `feature/batch-pcs` (`docs/vertical-batching-2026-10-21.md`).

## Borrowed from `feature/batch-pcs` (already absorbed)

- 90-bit security + `pow_bits=17`.
- Horizontal merge: pack same-arity weights into groups sized
  `1 << (26 - arity)` so `num_vars <= 26` stays within the PoW budget.
- Lazy opening (restore one group's Merkle tree at a time) and bulk POD
  serialization of prover data (both already in the cache path).

## Cannot change without a decision (recorded, not yet acted on)

1. **Full vs partial weight commitment.** `feature/batch-pcs` commits only
   `Op::Projection { w }` (q/k/v/o_proj/fc/proj), skipping `Op::MatMul { b }`
   (`lm_head`, 2^26) and biases — that is why its commit is ~7s while a full
   commit of the `lm_head` is minutes.
   **Decision (2026-10-22): commit `lm_head` too — full weight commitment.**
   The one-time commit stays expensive; that is accepted as the baseline.
2. **Boundary `Whir` num_variables must match exactly.** The committed path
   commits every boundary tensor with a single `whir`; p3-whir's adapter asserts
   arity == configured num_variables, so it only works when all boundary tensors
   share one size. Unrelated to weight commitment; not addressed by
   `feature/batch-pcs`.
3. **Affine bias committed but never opened.** `collect_weight_tensors` still
   collects the projection bias, but the Projection op only claims x/w/out (not
   bias), so a bias group has no opening (currently skipped as `proof: None`).
   If we keep committing biases this gap stays; `feature/batch-pcs` sidesteps it
   by not committing biases at all.

## Status

Store-free verification (no forward re-run) is done and measured on the plain
path. The committed (root-only) path is implemented, unit-tested (tamper /
wrong-root / serialization roundtrip, commit -> cache -> restore -> open), with
90-bit weight commitments + horizontal merge. Real-model end-to-end timing for
the committed path is still to be measured; first-run commit is dominated by the
`lm_head` if it is committed (see "Cannot change" #1).

## Baseline (2026-10-22)

GPT-2 seq=512 committed path, 90-bit + pow_bits=17 + horizontal merge, full
weight commitment (including lm_head). Release, 64-thread CPU.

| phase | time |
| --- | --- |
| commit (first run, writes cache) | 16.66 s |
| prove (restore + batch open) | 51.17 s |
| verify | 8.84 s |

argmax 511/512 (one near-tie fixed-point), peak RSS ~30.3 GB, proof bytes ~1.73 GB.

The committed verify was initially single-threaded (`for ... in ranges.iter()`);
parallelizing the shard loop (`ranges.par_iter()`, like the plain path) dropped
verify from ~66.9s to ~8.84s (shard sumcheck verify 64.5s -> 6.4s).

Note: an earlier argmax=0/512 was a bench bug (the argmax loop used vocab 65536
instead of the real 50257); the committed path itself was correct.

## Next steps (from seq=512 baseline)

- proof bytes ~1.73 GB is large; the weight/boundary opening proofs dominate it.
  Worth compressing (share the horizontal-merged FRI opening across tables, keep
  only Merkle paths).
- verify (68.1s) is larger than prove (51.9s) now; worth profiling what verify
  spends on (weight opening verification vs boundary cross-bind vs sumcheck).

## Round 2 status (2026-10-22, later)

- Done: parallelize committed verify shard loop (par_iter), verify 66.9s -> 8.84s.
- Done: remove dead per-weight committed path (commit_weights_batch,
  open_weight_batch, verify_weight_batch, prove_projection_committed,
  verify_projection_committed, CommittedProjectionProof) and their tests.
- Done: open the projection bias (add a bias claim at the affine point), so the
  projection bias commitment is now verified.

### Remaining soundness gap (part of the original #1)

collect_weight_tensors still commits normalization weights that are never
opened:
- Op::Layernorm { w, b } — neither w nor b is claimed.
- Op::LayerNormCentered { w, b } — neither is claimed.
- Op::RmsNorm { w } — w is not claimed.

These need the same treatment as the projection bias: claim their w/b at the
normalization output point using the out VirtualProof's final_evals.

### Recorded for later (user researching)

- Boundary Whir num_variables must match exactly (single whir for all boundary
  tensors; needs per-size boundary grouping).
- cache-reuse not implemented: commit_model_weights always re-commits, no
  cache-existence check, so commit-once + reuse across runs is incomplete.

### Full weight commitment complete (2026-10-22)

Opened the remaining normalization weights, so every committed weight is now
opened and verified:
- Op::Layernorm { w, b } -> claimed at r_out.
- Op::LayerNormCentered { w, b } -> claimed at r_out.
- Op::RmsNorm { w } -> claimed at r_out.

GPT-2 seq=512 end to end: commit 16.59s / prove 67.15s / verify 10.52s /
argmax 511/512 / proof 1.74GB. The projection bias and the normalization
w/b were opened via `mle::eval` at their output point, matching the batch
opening. Full weight commitment (matmul b, projection w+bias, layernorm w+b,
layernormcentered w+b, rmsnorm w) is now sound.

## Round 3 status (2026-10-22, boundary fix + e2e)

- Fixed #2: boundary commitment now uses a per-boundary Whir
  (Whir::with_params(arity, 90, 17)) instead of a single shared whir, so models
  with non-uniform boundary sizes work.

### End-to-end committed-path results (90-bit + pow_bits=17 + horizontal merge)

| model | commit | prove | verify | argmax | proof |
| --- | --- | --- | --- | --- | --- |
| GPT-2 seq=512 | 16.52s | 68.24s | 10.98s | 511/512 | 1.74GB |
| TimesFM seq=16 | 23.45s | 95.86s | 6.35s | - | 219MB |
| Gemma 3 seq=16 | blocked | - | - | - | - |

Gemma 3 is blocked by PowBitsExceedBudget (required 26 > budget 17): its lm_head
is vocab 2^18 x H_PAD 2^10 = 2^28, which is larger than the 2^26 that
pow_bits=17 supports. Horizontal merge cannot help (it is one tensor).

## Round 4 status (2026-10-22, per-group pow_bits)

Fixed the pow_bits blocker: `commit_model_weights` now adapts `pow_bits` per
weight group via `Whir::required_pow_bits(num_variables, 90)` (see
[docs/model-adaptation.md](model-adaptation.md)), instead of a fixed 17. A
tensor larger than 2^26 (e.g. Gemma 3's 2^28 lm_head) now gets its required
budget automatically.

All three models now pass the committed path end to end:

| model | commit | prove | verify | argmax | proof |
| --- | --- | --- | --- | --- | --- |
| GPT-2 seq=512 | 16.52s | 68.24s | 10.98s | 511/512 | 1.74GB |
| TimesFM seq=16 | 23.45s | 95.86s | 6.35s | - | 219MB |
| Gemma 3 seq=16 | 17.81s | 89.47s | 5.54s | 16/16 | 82.9MB |

Remaining for later: none blocking — the boundary num_variables issue (#2) and
the pow_bits issue are both fixed. cache-reuse across runs is still a
performance item, not correctness.

## Final status (2026-10-22)

Correctness is complete. Performance/scale items remain.

### Done (correctness)

1. 90-bit weight commitments (was new_testing 32-bit).
2. mmap weights via materialize (was Store::get panic).
3. OpeningProtocol/Whir serialized via reconstruction (were not serde).
4. Commit split into one-time commit_model_weights + prove restore (was re-commit per prove).
5. Projection bias + layernorm/layernormcentered w+b + rmsnorm w all opened (were committed but never opened).
6. Boundary commitment per-arity Whir (was a single whir, broke on non-uniform sizes).
7. Per-group adaptive pow_bits via Whir::required_pow_bits (was fixed 17, broke >2^26 tensors).
8. Dead per-weight committed path removed.
9. Batch-opening gamma/challenges Fiat-Shamir (were prover-chosen).
10. Committed verify parallelized (was sequential; 66.9s -> 8.8s).

Three models pass end to end: GPT-2 511/512, TimesFM, Gemma 3 16/16.
cargo test -p zkie-ops --lib: 36 passed.

### Remaining (performance / scale, not correctness)

1. cache-reuse across runs: decided NOT needed (2026-10-22). Commit is a
   one-time per-model cost (input-independent), so the benchmark reports the
   real commit time (GPT-2 16.5s / TimesFM 23.5s / Gemma 3 17.8s) rather than
   a cached "2s" figure. Keep commit_model_weights as always-commit.
2. proof size: GPT-2 1.74GB, dominated by the store-free witness (softmax
   indices/e, etc.).
3. prove time: GPT-2 ~68s, the longest of the three phases.
4. DeepSeek-V2-Lite seq=16 now adapted end to end (see section below); seq=512 and MoE lazy-open / routing proof remain.


## DeepSeek-V2-Lite seq=16 (2026-10-22, committed path e2e)

DeepSeek-V2-Lite (M=16) now runs the committed path end to end. MoE is still
DENSE (all 64 routed experts committed, gate is a trusted constant, top-k routing
proof not yet wired) -- this is the known scope, not a correctness gap.

| model | commit | prove | verify | argmax | rss | proof |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V2-Lite seq=16 | 480.76s (8.0m) | 1990.72s (33.2m) | 185.04s (3.1m) | 16/16 | 193.8GB | 1.43GB |

Notes:

- The earlier `TooLarge(5, 1)` panic was NOT weights or boundaries -- it was the
  bench placeholder `Whir::new_testing(1)` (WHIR folding factor 5 cannot fold a
  1-variable instance). Fixed by `Whir::new_testing(5)`. All 10110 weight tensors
  and 137 boundary tensors are power-of-2 with arity >= 5, so the real commit
  paths never hit it.
- prove time is dominated by weight batch openings: restoring ~290GB of weight
  prover-data cache and running one FRI proof per weight group (~191 groups).
- verify is store-free (no forward / no intermediate activations), 3.1m.
- commit cache is ~290GB on disk (15.7B params, dense MoE, WHIR-encoded).

### Minor cleanups (non-blocking)

- prove_committed_shard_dag / verify_committed_shard_dag take a now-unused
  `_whir` placeholder (boundaries build their own per-tensor Whir).
- Lookup tables (rsqrt/exp/gelu) are trusted public inputs, not committed
  (allowed fallback in the store-free plan).

## Sequence certification (2026-10-22, one-pass argmax + shift)

Started the DeepProve-style one-pass sequence certification: prove the whole
[prompt + response] in one forward and certify every position's argmax, then
bind the sequence via a verifier-side shift check.

Done (committed):
- Generic `Argmax` op (top-1, n-position) in zkie-ops: reuses TopKSelect(k=1)
  and binds the public index via `sel[i][t[i]] == 1`; the d1/d2 range-check
  table is parameterized (`range_bits`) and computed with true i64 differences.
- GPT-2 in-proof embedding: `tokens -> Lookup(lm_head) -> +pos_emb`.
- GPT-2 argmax wiring: `logits -> Scale(shift 13) -> Argmax -> t` (the shift
  bounds the range check; right-shift is monotone so argmax is ~preserved).
- `CommittedShardDagProof` carries public `tokens` / `argmax_t`; added
  `verify_sequence_shift(tokens, argmax_t, prompt)` verifier-side check.

GPT-2 seq=512 end-to-end (committed path): commit 16.9s / prove 84.6s /
verify 33.3s / argmax 511/512 / proof 2.55GB / rss 38GB.

GPT-2 padding fixed: the padded vocab columns are masked to -(1<<26) after the
right-shift (Scale shift 13), and the Argmax range-check is 27 bits. The proven
index now matches the post-hoc argmax (n_mismatch=0).

Gemma 3 and DeepSeek-V2-Lite builders are also wired for the same
tokens->wte lookup + Argmax + shift path (Gemma VOCAB is 2^18 so no padding mask;
DeepSeek vocab 102400->131072 is masked). Both compile.

Remaining (all data-generation, blocked on onnx/tokenizer not present on the
test box):
- Run extract_gpt2.py / extract_gemma3.py / extract_deepseek.py on a machine
  with onnx+onnxruntime+transformers to emit tokens_i32.bin (and gt) for each
  model, then run each model's committed bench end to end and confirm the
  sequence_shift_ok becomes true for a real prompt+response.


## Sequence certification complete (GPT-2, range-decomposition) (2026-10-23)

Replaced the single d2 range lookup (which forced a 13-bit logit shift that
flipped a near-tie) with a 48-bit range decomposition: three 16-bit limbs,
each range-checked against a 2^16 table, plus one recombination sumcheck. The
argmax now runs on the exact (masked) logits, so the proven index equals the
true argmax and the one-pass shift check passes.

GPT-2 seq16 real sequence ("The sky is ..."): argmax 16/16,
argmax_op_matches=true (n_mismatch=0), sequence_shift_ok=true, commit 10.5s /
prove 48.5s / verify 21.5s / proof 101MB. Prove/verify are also faster than the
shifted 2^27 single-table version.


GPT-2 seq512 (range-decomposition, random-token input): argmax 511/512,
argmax_op_matches=true (n_mismatch=0), commit 18.0s / prove 89.6s / verify
33.8s / proof 3.35GB / rss 40.2GB. Prove/verify are faster than the shifted
2^27 single-table version (104s/53s) and the proven argmax is now exact.


GPT-2 seq512 REAL generated sequence ("The sky is ...", HF FP32 generation):
argmax 512/512, argmax_op_matches=true (n_mismatch=0), sequence_shift_ok=true,
commit 18.5s / prove 89.8s / verify 34.2s / proof 3.35GB / rss 40.3GB. The one-pass
sequence certification passes end to end on a real 512-token response.
