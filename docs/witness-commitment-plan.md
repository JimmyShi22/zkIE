# Witness commitment plan

## Goal
Replace every plaintext witness tensor embedded in op proofs (sel/d1/d2/e/indices/rstd_vals/rem_lo/hi/token_id/...) with a commit + batch opening, so a proof carries only sumcheck evals, claims and openings, and the verifier never re-runs forward or re-derives the LogUp fraction tree. Intermediate activations stay virtual (uncommitted).

## Baseline (GPT-2 seq16 committed path, measured)
Total proof ~100.9 MB: Argmax 50 MB, Projection 600x35 MB, Lookup 9.8 MB, Softmax 558 KB, LayerNormCentered 403 KB.

## Root cause
verify_fractional / verify_lookup_fractional (logup_gkr.rs) rebuild the full num/den fraction tree from full x/y witness vectors and mle::eval them at every layer. prove_virtual/verify_virtual are already store-free (they carry final_evals), so the fix is confined to the fractional-sumcheck core.

## Design decisions
1. Store-free verifier: verify_fractional uses layer.proof.final_evals instead of rebuilding the tree.
2. Chained GKR (DeepProve style): sample the full challenge point r=(r_0..r_{T-1}) once; layer k uses the prefix r[..T-k-1] as its sumcheck point. The verifier opens only the original merged num_0/den_0 columns at r, and enforces inter-layer consistency with an MLE folding check: next_num_{k-1}(r[..T-k]) == (1-r_{T-k-1})*num_low_k(r[..T-k-1]) + r_{T-k-1}*num_high_k(r[..T-k-1]) (same for den). No intermediate layer is committed.
3. Lookup tables are committed too (DeepProve parity), not trusted public input.
4. Security 90 for all commitments (weights, tables, witness, multiplicity).
5. Witness columns are batch-committed per shard, grouped by size, reusing the existing weight commit_batch / prove_batch_open_committed machinery.

## Transcript order (Fiat-Shamir)
1. model id + shard/op structure (types + shapes, deterministic encoding)
2. commitment roots (weights, tables, witness, boundary)
3. in proof order: sumcheck round messages, claims, openings

## Witness column inventory per op
- TopKSelect / Argmax: sel, d1, d2
- RangeDecomp: c0, c1, c2
- Projection: rem_off
- Lookup: x (indices), y (values)
- SoftmaxRounded: indices, e
- LayernormChain: rsqrt_idx, rsqrt_vals
- LayerNormCentered: s_index, rstd_vals
- RmsNorm: s_index, rstd_vals, rem_lo, rem_hi
- RoPE: rem_f, rem_s
- Scatter: token_id, expert_of, sel
- ScatterAdd: weighted, gate, out, token_id
- GatherMatMul: counts, table

## Chained GKR protocol (logup_gkr.rs)

prove_fractional builds the fraction-addition tree (num_0/den_0 down to
final_num/final_den). The full challenge point r has T=log2(N) entries and is
sampled once; layer k runs prove_virtual at point r[..T-k-1].

verify_fractional_chain checks, per layer, the sumcheck against the stored
final_evals (no num/den rebuild), then checks the inter-layer MLE folding so
next_num_{k-1} and num_low_k/num_high_k agree. The remaining soundness gap is
the leaf: eval(num_0, r) / eval(den_0, r) is committed and opened in P2/P3.

## Progress log

- Lookup op migrated to chained store-free verify (drop indices/values). GPT-2
  seq16: argmax 16/16, verify 21.5s -> 2.34s, Lookup proof 9.8MB -> 221KB,
  total proof 100.9MB -> 91.4MB. Remaining big heads: Argmax 50MB, Projection 35MB.

## Progress log

- Lookup migrated: verify 21.5s -> 2.34s, Lookup proof 9.8MB -> 221KB, total 100.9MB -> 91.4MB.
- TopK/Argmax migrated (range checks chained, drop d1/d2/c0/c1/c2): Argmax 50.4MB -> 8.45MB
  (sel kept for one-hot binding), total 91.4MB -> 49.4MB, verify 2.34s -> 2.25s. Remaining:
  Projection 35.8MB, Argmax sel 8.45MB, Softmax 558KB, LayerNormCentered 408KB.

## Progress log

- Lookup migrated: verify 21.5s -> 2.34s, Lookup proof 9.8MB -> 221KB, total 100.9MB -> 91.4MB.
- TopK/Argmax migrated (range checks chained, drop d1/d2/c0/c1/c2): Argmax 50.4MB -> 8.45MB,
  total 91.4MB -> 49.4MB, verify 2.34s -> 2.25s.
- Projection migrated (drop rem_off): Projection 35.8MB -> 5.5MB, total ~49.4MB -> ~18MB.
  Remaining: Argmax sel 8.45MB (needs opening), Softmax 558KB, LayerNormCentered 408KB.

## Progress log

- Lookup/TopK/Argmax/Projection plaintext witness removed; proof 100.9MB -> 22.8MB, verify 21.5s -> 2.11s.
- Lookup witness commit + opening (per-column) sound: argmax 16/16, prove 128s / rss 38GB.
- Batch commit experiments REVERTED: PoW is intrinsic (required_pow_bits grows with num_variables),
  batch does not amortise it and inflates peak memory. security 90 must not drop (user).
- Split LogUp protocol complete (logup_gkr.rs): witness/table sides + LogUp link,
  table-side (m+table) commit/open, merge_multiplicity, batch witness side.
  This is the protocol basis for P2 table commitment: table committed once globally,
  witness side committed once per batch, PoW drops an order of magnitude at security 90.

## Progress log

- Lookup/TopK/Argmax/Projection plaintext witness removed; proof 100.9MB -> 22.8MB, verify 21.5s -> 2.11s.
- Lookup witness commit + opening (per-column) sound: argmax 16/16, prove 128s / rss 38GB.
- Split LogUp protocol (witness/table sides) + two-phase collection landed: table side committed once,
  witness side batched. GPT-2 seq16: prove 128s -> 36.4s, rss 38GB -> 15.6GB, argmax 16/16.
  This is the P2 table-commitment PoW win, at security 90, without changing soundness.

## Phases
- P0: this document.
- P1: store-free fractional-sumcheck core (logup_gkr.rs) + leaf claims.
- P2: table + multiplicity commitment (compose.rs committed path).
- P3: per-op witness commitment + drop plaintext vectors (ordered by size).
- P4: end-to-end regression + proof-size metric in README/benchmarks.

## Acceptance
- Each op witness bytes drop to ~K level (bench_committed breakdown).
- Tamper claim / opening / wrong table root / wrong witness root => reject.
- GPT-2 seq16 (16/16) and seq512 (511/512) argmax unchanged; prove/verify not slower than current committed baseline.

- Table-side multiplicity m committed + opened per table (per-shape, no cross-shape merge, per user: batch only same-shape). GPT-2 seq16: prove 36.4s -> 51.48s, verify 2.05s, proof 21.1MB -> 22.86MB, rss ~14.5GB, argmax 16/16.

- Fiat-Shamir wired: fractional full_r/per-layer c now derived from a Transcript that absorbs the committed num/den/m roots (prover re-runs the chained GKR; verifier replays and checks the challenges match). GPT-2 seq16: argmax 16/16, prove 51.5s -> 66.2s (re-run cost), verify 2.05s -> 2.08s, rss ~14.4GB, proof ~21.8MB.

- Table body commitment landed (P2 done): table column padded + committed + opened at leaf; verifier checks den_leaf == alpha + eval(j) + beta*eval(table). GPT-2 seq16: argmax 16/16, prove 66.2s -> 82.1s (per-shard table commit, redundant across shards), verify 2.08s -> 2.12s, proof ~23.5MB, rss ~15.5GB.

- Witness x/y/num columns committed + LogUp alpha/beta bound via Fiat-Shamir (P3 core). Verifier opens x/y/num and checks den_leaf == alpha + eval(x) + beta*eval(y); alpha/beta/full_r/c all transcript-derived. Lookup op is now fully sound. GPT-2 seq16: argmax 16/16, prove ~82s, verify 2.15s, proof ~24.3MB, rss ~14.4GB.

- Argmax one-hot sel committed + opened at the m claimed output positions (sel_evals in ArgmaxProof, topk.sel cleared). GPT-2 seq16: argmax 16/16, proof 24.3MB -> 18.0MB, prove ~84s, verify 2.32s.

- Softmax migrated: softmax-exp table committed, softmax lookup moved into split witness/table path, indices/e plaintext dropped. GPT-2 seq16: argmax 16/16, prove ~87.8s, verify 2.15s, proof 18.0MB -> 21.9MB (table body commitment cost).

- LayerNormCentered migrated: rsqrt table committed, rsqrt lookup moved into split path, s_index/rstd_vals plaintext dropped. Also padded tiny witnesses to >=2^5 (WHIR folding factor 5 needs arity>=5). GPT-2 seq16: argmax 16/16, prove ~104.7s, verify 2.15s -> 0.96s, proof ~24.9MB.

- Layernorm + RmsNorm rsqrt migrated (rsqrt table committed, s_index/rstd_vals dropped). GPT-2 only uses LayerNormCentered; these are for other models, verified by unit tests + cargo check.

- Table body commitment made global-once (not per shard): each distinct lookup table committed once, openings per shard. GPT-2 seq16: argmax 16/16, prove 104.7s -> 101.1s, proof ~24.9MB.

- Range checks migrated: RmsNorm rem_lo/rem_hi and RoPE rem_f/rem_s committed (limb identity table 2^16 added to store, witness added to lookup_by_table). Verifier no longer regenerates the range table (succinct). gemma3/deepseek builders push limb_t.

- MoE Scatter permutation committed: PermutationProof split into a/b fractional sides, witness a/b/num committed and opened at both leaf points, alpha Fiat-Shamir bound. deepseek compiles, 48 tests green.

- limb/bin range-check tables: closed-form MLE eval (Sigma p_i*2^i) instead of commitment. Gather committed path: commit weighted/gate/out/num/den + open (witness still kept; store-free clearing is follow-up). 48 tests green, deepseek compiles.

- Gather histogram split into source/target fractional sides (cross-link S_source==S_target), committed path commits num_source/den_source/num_target/den_target/weighted/gate/out + opens. Fixed the negative-sign bug in target num. 48 tests green, deepseek compiles. Remaining: token_id/den binding + store-free witness clearing.

- Gather soundness closed: den=alpha-token_id bound via committed token_id sub-point opening; witness cleared (store-free); verify uses chain GKR + product-check final_evals + openings. 48 tests green, deepseek compiles. No remaining witness plaintext.

- P4 started: README benchmark table gains Proof size column; GPT-2 seq16 re-benchmarked (prove ~101.3s, verify ~1.0s, proof ~24.9MB). Other models need re-benchmark with the committed path.
