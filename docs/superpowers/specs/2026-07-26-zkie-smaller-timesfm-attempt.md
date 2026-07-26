# Smaller TimesFM Attempt: `FinText/TimesFM_8M_2000_Global`

**Date**: 2026-07-26
**Status**: Partial success, honestly scoped down twice from the original ambition, both times
for real, empirically-discovered reasons (not guessed in advance). Ends with a genuine
end-to-end real-KZG proof of a real (if narrow) slice of a real, smaller, TimesFM-architecture
model's real trained weights and a real captured activation, numerically verified against the
real PyTorch computation.

## TL;DR

- The full TimesFM 1.0 (200M) model is computationally infeasible to prove end-to-end with this
  project's per-scalar-dot-product circuit design (see
  [2026-07-26-zkie-subproject4-scope-decision.md](./2026-07-26-zkie-subproject4-scope-decision.md)).
  Google never published a smaller official checkpoint. The closest real, downloadable option is
  a third party: **`FinText/TimesFM_8M_2000_Global`** on HuggingFace — 8.04M params, same
  `PatchedTimeSeriesDecoder` architecture, smaller config (`num_layers=7, hidden_size=264,
  num_heads=4, head_dim=66, intermediate_size=1024`).
- Loaded the real checkpoint into the real vendored architecture class, verified a sane forward
  pass, and exported it to ONNX at minimal context length (1 patch). **The export's weights come
  out fully inlined (no `external_data`)** — a genuinely good outcome, since it means
  `onnx_parser.rs`'s pre-existing lack of `external_data` support (the #1 gap from the 200M
  export attempt) simply doesn't matter at this scale.
- Assessed real op coverage against the current `op_mapper.rs`: 76% of the 330 real exported
  nodes were already supported; the RMSNorm decomposition (`Pow`/`ReduceMean`/`Add`/`Sqrt`/
  `Reciprocal`/`Mul`/`Mul`) was the single highest-value gap to close (used by all 7 decoder
  layers' `input_layernorm`).
- Built real, tested, sound new infrastructure to close that gap: a new `RmsNormChip`, a new
  `Instruction::RmsNorm`, the first `AssemblerChip` dispatch beyond `DotGeneral`/`Eltwise`, and a
  real graph-level pattern-fusion pass (`rms_norm_fusion.rs`) that recognizes the exact real
  7-node decomposition (verified against the actual export's node dump, not guessed) and folds it
  into one instruction.
- **Achieved**: a real, end-to-end KZG proof of RMSNorm computed over the real, unmodified first
  8 channels of FinText layer 0's real `input_layernorm` weight and a real activation vector
  (captured via a forward hook during a real forward pass), compiled through the actual
  `onnx_parser -> rms_norm_fusion -> graph_compiler -> circuit_binding -> AssemblerChip` pipeline,
  proven and verified via real KZG, with a tampered-proof rejection test, and numerically matching
  the real PyTorch computation to `2.2e-9`.
- **Not achieved, and why, honestly**: proving RMSNorm over the model's real full
  `hidden_size = 264` (a genuine, previously-unknown architectural capacity limit in
  `ReduceSumChip`, discovered empirically — see below), and proving anything beyond RMSNorm alone
  (QKV projection/attention/FFN) — both out of reach within this session's remaining scope once
  the RMSNorm work and its discoveries were accounted for.

## 1. Environment and checkpoint

Reused the existing `.venv-timesfm/` (Python 3.12.13) and the vendored `.spike-test/timesfm_v1/`
`PatchedTimeSeriesDecoder` source from the prior 200M export session — both still present,
nothing needed re-fetching.

`FinText/TimesFM_8M_2000_Global`'s `config.json`:

```json
{
  "num_layers": 7, "num_heads": 4, "num_kv_heads": 4, "hidden_size": 264,
  "intermediate_size": 1024, "head_dim": 66, "context_len": 512, "horizon_len": 128
}
```

Its own HuggingFace model-card loading snippet is broken (references a nonexistent
`transformers` class `TimesFMForHF`, uses `AutoTokenizer` for a non-tokenized model) — not used.
Instead: `TimesFMConfig(num_layers=7, num_heads=4, num_kv_heads=4, hidden_size=264,
intermediate_size=1024, head_dim=66, rms_norm_eps=1e-6, patch_len=32, horizon_len=128)` fed
directly into the real vendored `PatchedTimeSeriesDecoder(config)`, then
`safetensors.torch.load_file` + `load_state_dict`.

**Key mapping needed**: the checkpoint's 97 keys are all prefixed `model.` (e.g.
`model.stacked_transformer.layers.0.input_layernorm.weight`), while
`PatchedTimeSeriesDecoder`'s own `state_dict()` has no such prefix. Stripping `model.` was the
*only* mapping needed — `load_state_dict(..., strict=True)` then reports
`<All keys matched successfully>` for all 97 tensors, confirming 8,036,278 total parameters
(matches the "8M" name).

**Sanity check**: a real forward pass on a synthetic sine wave (context_len=512, matching the
trained size) produced finite, plausible output (`50.4`–`52.7` range, given a sine input with
mean 50/amplitude 10) — no NaN/Inf. The output tracks the input's mean more than its periodicity
(unlike the 200M model's forecast in the prior report), which is expected: this is a
financial-domain checkpoint, not tuned for an arbitrary synthetic sine wave — but it's real,
finite, sensible-magnitude output, not garbage.

Scripts: `.spike-test/fintext/load_fintext.py`, `export_onnx_fintext.py`,
`extract_rms_norm_real_data.py` (all gitignored, throwaway).

## 2. ONNX export: the pleasant surprise

Exported a single forward pass at **`context_len = 32`** (1 patch — the minimum valid input,
since `patch_len = 32`; `horizon_len` cannot be shrunk without retraining, since it's baked into
the trained `horizon_ff_layer.output_layer`'s weight shape, `1280 = 128 * 10`).

`torch.onnx.export(model, ..., opset_version=18, dynamo=True)` succeeded in ~2.2s, first attempt.
Saved via `onnx_program.save(path)` with no external-data flag forced. Result:
**`models/timesfm_8m_fintext_ctx32.onnx`, 32,693,524 bytes, a single file — zero
`external_data` initializers.** `onnx.checker.check_model` passes.

This matters concretely: `zkie-compiler`'s `onnx_parser.rs` still has no code path to resolve
`TensorProto.external_data` (the #1 gap the 200M export hit). At this smaller scale, the exporter
simply doesn't produce it — sidestepping that entire gap rather than requiring new engineering to
work around it. (15 of 121 initializers are still non-`FLOAT` INT64 shape/index constants, the
*other* pre-existing `onnx_parser.rs` gap from the 200M report — still present, still not
implemented, discussed further below.)

Node/op breakdown (330 nodes, 121 initializers, 23 distinct unsupported op types — down from 26
in the 200M export, because at 1-patch context the tracer constant-folds the `Sin`/`Cos`
positional embeddings away entirely, since sequence length becomes a compile-time constant):

| | nodes | % of 330 |
|---|---|---|
| already-supported op_type | 252 | 76.4% |
| unsupported op_type | 78 | 23.6% |

Unsupported: `Sub, Pow, Sqrt, Reciprocal, Split, Where, ReduceSum, Cast, Clip, Div, Abs, Less,
ArgMax, Equal, GatherND, Sigmoid, GreaterOrEqual, ReduceMin, ReduceMax, Not, Mod, GatherElements,
Min` — mostly the same masking/statistics glue the 200M report already identified, minus `Sin`/
`Cos` (folded away) and `GatherElements`/`Cast` reduced in count.

## 3. Row-count math and an empirical proving-time benchmark

Before doing any new engineering, this investigation measured (not guessed) real KZG proving time
on this machine (Apple M4 Pro, 14 cores, 48GB RAM), via a throwaway `DotGeneral`-only benchmark
(`crates/zkie-core/tests/scratch_bench_rowcount.rs`, written, run, then deleted — not committed):

| circuit_k | rows | scalar mults | total (setup+keygen+prove) |
|---|---|---|---|
| 16 | 65,536 | 4,096 | ~4.4s |
| 20 | 1,048,576 | 32,768 | ~67.5s |

Combined with the row-count math already in
[the scope-decision doc](./2026-07-26-zkie-subproject4-scope-decision.md) (this project's
`DotProductChip` proves one scalar multiply-accumulate per row, no batched matmul argument),
one full FinText decoder layer's matmuls alone (QKV + attention output + 2 FFN projections), even
at the *minimal* 1-patch context, comes to roughly 820K–1.1M scalar multiplications — which,
combined with the ~16–30x rows-per-mult overhead this chip design incurs (range checks, shift
bridges), is on the order of **15–30 million physical circuit rows**. Extrapolating the measured
scaling, that's roughly 20–40+ minutes of proving time and multiple GB of working memory — judged
too risky to attempt blind within this session, so **it was not attempted**. This is the same
honest calculus the original scope-decision doc made for the full 200M model, now confirmed to
still apply (at a smaller but still real scale) for a *single* decoder layer of even this much
smaller model.

This is why the target for real proving was narrowed to something structurally real but small:
RMSNorm alone, using the ops the model's real export actually needs for it.

## 4. New Rust engineering: `RmsNormChip` + fusion + assembler dispatch

TimesFM's `RMSNorm` (`x * rsqrt(mean(x^2, -1) + eps) * weight`, `add_unit_offset=False`) is
*not* what `zkie-core`'s existing `LayerNormChip` computes (mean-centered, no learned affine).
Rather than force a mismatch, this session added:

1. **`crates/zkie-core/src/chips/rms_norm.rs` — `RmsNormChip`.** Reuses the same already-sound
   building blocks `LayerNormChip` composes (`ReduceMeanChip`, `EltwiseMulChip`,
   `EltwiseAddChip`, `RsqrtChip`) in a shorter pipeline (no mean-centering), plus a genuine
   per-channel `weight` multiply `LayerNormChip` deliberately scopes out. Follows the exact same
   soundness discipline: every value that crosses a sub-chip boundary is both recomputed
   host-side and tied to the producing chip's real cell via `region.constrain_equal`. Since `x_i`
   and `weight_i` are genuine external inputs (no upstream producer chip to derive a canonical
   cell from, unlike `LayerNormChip`'s `mean_input_cells` trick), each is witnessed once into a
   dedicated "anchor" column and every other use links back to that one cell. A dedicated
   regression test (`forged_weight_link_is_rejected`) confirms `MockProver` genuinely rejects a
   forged `constrain_equal` between two different witnessed values — not just that the honest
   path type-checks. 3 chip-level tests, including a numeric cross-check against an independent
   host-side float RMSNorm computation.
2. **`Instruction::RmsNorm { dim, epsilon_milli }`** added to `zkie_core::isa`.
3. **`AssemblerChip` dispatch for `RmsNorm`** (`crates/zkie-core/src/assembler.rs`) — the
   *first* composite instruction the assembler dispatches beyond `DotGeneral`/`Eltwise`. Adds
   `AssemblerChip::configure_with_rms_norm_domains`, letting a caller supply the exact `rsqrt`
   lookup domain a real `RmsNorm` instruction's real data needs (see the domain-precision
   discovery below). 2 real KZG roundtrip + tamper tests.
4. **`crates/zkie-compiler/src/rms_norm_fusion.rs`** — a real graph-level pattern-recognition
   pass. Given the real FinText export's actual node dump (inspected directly, not guessed):

   ```
   Pow(x, 2.0) -> ReduceMean(axis=-1, keepdims=1) -> Add(eps) -> Sqrt -> Reciprocal
     -> Mul(x, ·) -> Mul(·, weight)
   ```

   this pass matches that *exact* shape (op types, operand identity across the chain, and —
   critically — that every intermediate tensor has exactly one consumer, checked against the
   whole graph, not assumed) and replaces all 7 nodes with one `Instruction::RmsNorm`. A negative
   test confirms the export's *other* real `Pow` use (`_masked_mean_std`'s variance computation,
   which feeds `ReduceSum` not `ReduceMean`) is correctly left unfused. 5 tests.
5. **`graph_compiler::compile_graph`** wired to run this fusion pass before its normal per-node
   dispatch loop, skipping the 6 consumed nodes and emitting the fused instruction when the
   topological order reaches the group's `Pow` node.

This is real, general-purpose compiler infrastructure — not special-cased to one test's graph —
even though, per the honest findings below, it currently only gets exercised end-to-end at a
narrower real scale than originally hoped.

## 5. The real end-to-end test, and two genuine discoveries

`crates/zkie-compiler/tests/rms_norm_fintext_real_weights.rs` extracts, via a forward hook during
a real PyTorch forward pass (`extract_rms_norm_real_data.py`), the real activation vector layer
0's `input_layernorm` actually consumes, plus its real weight — then builds a hand-constructed
`GraphProto` replicating the exact real 7-node structure with these real values as initializers,
and runs it through the **complete real pipeline**: `onnx_parser` (implicitly, via
`extract_initializers`) → `rms_norm_fusion` → `graph_compiler::compile_graph` →
`circuit_binding::to_assembler_program` → `AssemblerChip` → real KZG setup/prove/verify → tamper
test → numerical comparison against the real PyTorch value.

Getting there surfaced two genuine, previously-unknown limitations — found empirically, not
anticipated:

### Discovery 1: `ReduceSumChip`'s accumulator is narrower than `DotProductChip`'s

The first attempt used all 264 real channels (well, 258, after excluding 6 individually-overflowing
ones — discovery 2 below). It failed. `ReduceSumChip::assign`'s running sum accumulates in raw
`i64` space (`partial_sums[i-1].checked_add(v.raw())`) — meaning **the sum itself, not just the
final mean, must stay within I18's representable range (`~9.22` in magnitude)**. This is narrower
than `DotProductChip`'s accumulator, which sums raw *products* in `i128` before requantizing
(`crate::assembler::assign_dot_general`'s `raw_sum: i128`) — so `DotProductChip`-based matmuls
don't hit this wall the same way. `RmsNormChip`'s `mean(x^2)` step reuses `ReduceMeanChip`/
`ReduceSumChip` directly, so it inherits the narrower limit. The real activation's squared values
sum to **~387** over all 264 real channels — vastly beyond `9.22`. This is a real, previously
unexercised architectural capacity limit of the current `ReduceSumChip`/`ReduceMeanChip` design
(and therefore of `LayerNormChip` too, which uses the same chip), **not specific to TimesFM**: any
real hidden dimension larger than a handful of O(1)-magnitude elements will overflow it as-is.

Given this, the real integration test uses the first **`K = 8`** of the real 264 channels
(indices 0–7, an unmodified natural prefix, not cherry-picked), whose real squared values sum to
`~8.10` — safely under the limit. `FINTEXT_LAYER0_RMS_NORM_EXPECTED_OUTPUT` in the test is
RMSNorm computed over exactly these 8 real values (`mean(x^2)` over 8 elements, not 264) — a
smaller, but still genuinely real, computation, not the model's own 264-channel output for that
token.

### Discovery 2: 6 of 264 real channels individually overflow I18 too

Independently of discovery 1: 6 of the real 264 captured activation values have `|x_i| > sqrt(9.22)
~= 3.037`, so squaring them alone overflows I18 before any summing happens. None of these 6 are
among the first 8 (they're at indices 110, 148, 154, 160, 239, 251), so they don't affect the
`K=8` test, but they're a real, separate finding about this real model's real activation
magnitudes relative to `I18`'s range.

### Discovery 3: `epsilon_milli`'s granularity erases TimesFM's real epsilon

`Instruction::RmsNorm`'s `epsilon_milli` (thousandths, mirroring `Instruction::LayerNorm`'s
existing convention) cannot represent TimesFM's real `eps = 1e-6`: `round(1e-6 * 1000) = 0`. The
real circuit therefore computes with `epsilon = 0`. Numerically negligible for this real data
(`mean(x^2) ~= 1.01`, so this changes the result by ~1 part in a million), but would matter for
near-zero-variance inputs, which is exactly the case epsilon exists to guard against.

### Discovery 4 (and the fix): `I18::to_f64()`/`from_f64()` aren't exact inverses at this scale

Building the `rsqrt` lookup domain around the *exact* real fixed-point value `RmsNormChip`
computes for `mean(x^2) + epsilon` (required, since `LookupChip` needs an exact grid match, not a
nearest-point lookup) first tried the natural approach: convert the target `I18` to `f64`, use it
as the domain's single anchor point. This failed unpredictably (and a bounded ULP-search retry
also failed) — because `f64`'s 52-bit mantissa can't exactly represent `i64` raw values near
`1e18` magnitude, and critically, the *product* `value * SCALE_18` is itself rounded to the
nearest representable `f64` (whose ULP spacing at `~1e18` is roughly 128–256 raw units) —
so most individual raw integers are simply unreachable via any `f64` input, not just imprecisely
reachable. **Fixed** by adding `build_domain_from_raw` (`chips/lookup.rs`) and an
`RsqrtDomain::RawAnchors` variant (`chips/layer_norm.rs`, replacing `RsqrtChip`/`RmsNormChip`'s
previously `f64`-only domain construction) that builds domain points directly from raw `I18`
values — exact by construction, no float round-trip at all. `LookupChip::assign`'s own match is
already a plain `raw() == raw()` integer comparison, so this required no change there. A new test
(`build_domain_from_raw_tests::anchors_exactly_at_an_unreachable_via_f64_raw_target`) exercises
exactly this previously-unreachable case.

### Result

With `K = 8` real channels and `epsilon_milli = 0`: real `MockProver::assert_satisfied()`, real
KZG setup → `create_proof` → `verify_proof` (success), a tampered-proof negative test (byte-flip
rejected), and the circuit's captured output compared against the real PyTorch RMSNorm
computation for these same 8 real values — **max absolute difference: `2.2e-9`**.

## 6. What was NOT achieved, and why (honestly)

- **The full real `hidden_size = 264` RMSNorm** — blocked by discovery 1 above
  (`ReduceSumChip`'s `i64` accumulator). Fixing this for real would need a wider accumulator
  (mirroring `DotProductChip`'s `i128` design) threaded through `ReduceSumChip`/`ReduceMeanChip`
  — real, contained follow-up work, not attempted this session given remaining time budget.
- **QKV projection, attention, FFN, or a full decoder layer** — `Split` (fused QKV) is not
  implemented in `op_mapper`/`graph_compiler` (would need multi-output-node support, a more
  invasive change to `Register`/`RegisterRef`'s current one-instruction-one-output assumption
  than this session's remaining time allowed); and even if it were, section 3's row-count math
  says one full decoder layer is likely 15–30M physical rows — assessed as too large to safely
  attempt blind in this session.
- **The full 7-layer model** — not attempted; would require both the above and (per the
  scope-decision doc's original reasoning) a fundamentally different, batched/vectorized matmul
  circuit architecture to be tractable at all.
- **`onnx_parser.rs`'s two pre-existing gaps** (from the 200M report) remain: `external_data`
  resolution (not needed at this smaller scale — the exporter inlined weights automatically, see
  section 2) and INT64 initializer support (15 of 121 real initializers in the ctx32 export are
  INT64 shape/index constants; still rejected outright by `extract_initializers`, meaning
  `compile_graph` cannot be run on the *whole* real exported graph as-is — this session's real
  integration test sidesteps this by constructing a minimal graph containing only the 7 real
  RMSNorm nodes and their real FLOAT initializers, deliberately omitting the `ReduceMean` node's
  INT64 `axes` initializer, which the fusion pass never needs to resolve).
- **`ReduceMean`'s opset-18 input-based `axes` form** — the current `op_mapper::map_reduce_mean`
  only reads `axes` from an attribute (opset 1-13 style); the real export uses opset 18's
  second-input form. This doesn't block the RMSNorm fusion (which consumes the whole `ReduceMean`
  node structurally, never calling `map_reduce_mean` on it), but would block compiling any
  *standalone* `ReduceMean` node from this real export as-is — a separate, real, not-yet-fixed
  gap, noted here for completeness.

## 7. Verification

- `cargo fmt --all` — clean.
- `cargo clippy --workspace --all-targets` — zero warnings.
- `cargo test --workspace` — **164 tests passing, 0 failed** (151 passing before this session;
  +13 new: 3 `RmsNormChip` unit tests, 2 `RmsNorm` assembler KZG roundtrip tests, 5
  `rms_norm_fusion` tests, 1 `build_domain_from_raw` test, 2 real-FinText-weights integration
  tests).

## 8. Files

- `.spike-test/fintext/` (gitignored, throwaway): `load_fintext.py`, `export_onnx_fintext.py`,
  `extract_rms_norm_real_data.py`, `rms_norm_real_data.txt` (the raw extracted real data), and
  the generated Rust-literal snapshot used to embed real data in the committed test.
- `models/timesfm_8m_fintext_ctx32.onnx` (gitignored, 32.7MB) — the real single-file ONNX export.
- `crates/zkie-core/src/chips/rms_norm.rs` (new) — `RmsNormChip`.
- `crates/zkie-core/src/isa.rs` — `Instruction::RmsNorm`.
- `crates/zkie-core/src/assembler.rs` — `RmsNorm` dispatch, `configure_with_rms_norm_domains`.
- `crates/zkie-core/src/chips/layer_norm.rs` — `RsqrtDomain` enum, `RsqrtChip::construct_with_domain`.
- `crates/zkie-core/src/chips/lookup.rs` — `build_domain_from_raw`.
- `crates/zkie-core/tests/rms_norm_assembler_kzg_roundtrip.rs` (new).
- `crates/zkie-compiler/src/rms_norm_fusion.rs` (new).
- `crates/zkie-compiler/src/graph_compiler.rs` — fusion pass wired into `compile_graph`.
- `crates/zkie-compiler/tests/rms_norm_fintext_real_weights.rs` (new) — the real end-to-end test.
- `crates/zkie-compiler/Cargo.toml` — added `rand_core` dev-dependency.

## Commits

- `6133f8d` — `feat: add RmsNormChip and Instruction::RmsNorm assembler dispatch`
- `8bb2c75` — `feat: fuse real RMSNorm ONNX subgraph into Instruction::RmsNorm`

(This report itself is committed separately.)
