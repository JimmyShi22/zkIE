# TimesFM 1.0 (200M) → ONNX Export Attempt

**Date**: 2026-07-26
**Status**: Succeeded (real checkpoint, real forward pass, real `torch.onnx.export`, numerically
verified, loadable by `zkie-compiler`'s own parser) — with two concrete, honestly-reported gaps
in what `zkie-compiler` can currently do with the result. This is item #2 from
[2026-07-26-zkie-subproject4-scope-decision.md](./2026-07-26-zkie-subproject4-scope-decision.md).

As far as this investigation can tell, this is the first successful ONNX export of TimesFM 1.0
anywhere — the two public GitHub issues asking for exactly this
([#252](https://github.com/google-research/timesfm/issues/252),
[#115](https://github.com/google-research/timesfm/issues/115)) remain open with zero replies.

## TL;DR

- Loaded the real `google/timesfm-1.0-200m-pytorch` checkpoint (203,568,960 parameters) into the
  actual `PatchedTimeSeriesDecoder` PyTorch module from `google-research/timesfm`'s `v1/` source.
- Ran a real sanity forecast via the library's own `tfm.forecast()` API on synthetic data —
  produced sensible, periodic output on a sine-wave input, confirming the loaded model works.
- Exported a single (non-autoregressive) forward pass — context length 512, horizon 128, which is
  exactly `output_patch_len`, so no autoregressive Python loop is involved — to ONNX using
  `torch.onnx.export(..., dynamo=True)` (torch 2.13's default exporter). **It worked on the first
  attempt**, in ~10 seconds, no workarounds needed.
- The exported graph is valid (`onnx.checker.check_model` passes) and numerically matches the
  original PyTorch model to within `max abs diff = 7.4e-6` on a random fixed input, verified via
  `onnxruntime`.
- `zkie-compiler`'s own `onnx_parser::load_model` (protobuf decode) loads the file without error.
- However, `zkie-compiler`'s op mapper would currently reject 134 of 778 nodes (17%, spanning 26
  distinct op types) as `UnsupportedOp`, and — separately and more importantly — its
  `extract_initializers` cannot currently read **the actual weight values** from this export at
  all, because the export uses ONNX's `external_data` mechanism (a sibling `.onnx.data` file),
  which the parser doesn't resolve, and because 17 of 279 initializers are non-float (INT64)
  constants that the parser deliberately rejects by design. Both gaps are pre-existing, documented
  scope limitations of `onnx_parser.rs` (see its module doc), now confirmed empirically against a
  real, non-toy model rather than being purely hypothetical.

## Environment setup

- New venv at `/Users/jimmyshi/code/zkie/.venv-timesfm/` (gitignored — added to `.gitignore`),
  built from `pyenv`'s Python **3.12.13**, not the system default of Python 3.14.2. Python 3.14
  is too new for `torch`'s current wheel matrix on macOS arm64 as of this writing (best not to
  find out via `pip install` failing 20 minutes in), so 3.12.13 (already installed via pyenv on
  this machine) was used instead.
- `pip install "timesfm[torch]"` (the straightforward path suggested in the task) resolved to
  **timesfm 2.0.2**, torch 2.13.0, and — notably — **pulled in zero JAX/JAXlib dependencies**.
  The packaging issue referenced in the task ([#254](https://github.com/google-research/timesfm/issues/254))
  appears to no longer apply to the current PyPI release.
- **But** that installed `timesfm` 2.0.2 package only contains code for **TimesFM 2.5**
  (`timesfm/timesfm_2p5/timesfm_2p5_torch.py` etc.) — there is no `pytorch_patched_decoder.py`,
  no `PatchedTimeSeriesDecoder`, and no TimesFM-1.0-shaped config anywhere in it. The PyPI
  `timesfm==1.0.0` release was also checked directly (downloaded, unzipped, inspected without
  installing) and turned out to be **JAX-only** (`jax==0.4.26`, `paxml==1.4.0`, `praxis==1.4.0`,
  no torch at all) — this is the old release that predates the repo's PyTorch port.
  **Neither PyPI release of `timesfm` gives you TimesFM 1.0's actual PyTorch architecture.**
- So, per the task's fallback instruction, the actual `v1/src/timesfm/` PyTorch-relevant source
  files were fetched directly from `raw.githubusercontent.com/google-research/timesfm/master/v1/src/timesfm/`
  (a git clone was tried first but failed — the repo uses `git-lfs`, which isn't installed on this
  machine, and per security guardrails only files needed were downloaded directly rather than
  pursuing LFS setup). Only the 3 files with zero JAX/TensorFlow/absl/sklearn dependencies were
  vendored, into `.spike-test/timesfm_v1/` (gitignored, throwaway):
  - `pytorch_patched_decoder.py` — the real `PatchedTimeSeriesDecoder` `nn.Module`, verbatim, zero
    internal deps beyond `torch`.
  - `timesfm_base.py` — the shared hparams/checkpoint dataclasses and `forecast()` orchestration
    (needs `pandas`, `utilsforecast`, both installed).
  - `timesfm_torch.py` — the `TimesFmTorch` wrapper that does `snapshot_download` +
    `PatchedTimeSeriesDecoder(...).load_state_dict(...)` (needs `huggingface_hub`). One import was
    rewritten (`from timesfm import timesfm_base` → `from . import timesfm_base`) so it works as
    a standalone package instead of colliding with the installed `timesfm` 2.0.2 package name.
  - Explicitly **not** vendored: `patched_decoder.py` (JAX), `timesfm_jax.py` (JAX/paxml/praxis),
    `data_loader.py` (tensorflow/absl/sklearn), `xreg_lib.py` (jax, only needed for the covariates
    feature, not plain forecasting), `time_features.py` (only used by `data_loader.py`).
  - Full copies of all 9 original `v1/src/timesfm/*.py` files (for reference/audit) are also
    saved at `.spike-test/timesfm_v1_src/` (gitignored).
- Additional packages installed into the venv: `pandas`, `utilsforecast`, `onnx==1.22.0`,
  `onnxscript==0.7.1` (required by the dynamo-based ONNX exporter), `onnxruntime==1.28.0` (for the
  numerical parity check, not required by the task but valuable extra evidence).

## Checkpoint gotcha: two different HF repos

`google/timesfm-1.0-200m` (the repo id from the task's research notes) **does not contain a
PyTorch checkpoint** — as of this writing its file listing is only a JAX/Orbax checkpoint
(`checkpoints/checkpoint_1100000/{descriptor,metadata,state}/...`). Attempting
`tfm.load_from_checkpoint(TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m"))`
downloads successfully but then fails with:

```
FileNotFoundError: [Errno 2] No such file or directory: '.../torch_model.ckpt'
```

The actual `torch_model.ckpt` file lives in a **separate** HuggingFace repo:
**`google/timesfm-1.0-200m-pytorch`**. Using that repo id, the checkpoint loads correctly.

## Sanity check: real forward pass before attempting export

Using the vendored `timesfm_v1` package's own `TimesFm.forecast()` API (not a hand-rolled call):

```python
hparams = tfm_pkg.TimesFmHparams(context_len=512, horizon_len=128, input_patch_len=32,
                                  output_patch_len=128, num_layers=20, model_dims=1280,
                                  per_core_batch_size=1, backend="cpu")
checkpoint = tfm_pkg.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m-pytorch")
tfm = tfm_pkg.TimesFm(hparams=hparams, checkpoint=checkpoint)
point_forecast, quantile_forecast = tfm.forecast([sine_wave, random_walk], freq=[0, 0])
```

- `type(tfm._model)` → `timesfm_v1.pytorch_patched_decoder.PatchedTimeSeriesDecoder`
- `sum(p.numel() for p in tfm._model.parameters())` → **203,568,960** (matches the "200M" name)
- Forecast on a synthetic sine wave (period 24, amplitude 10, offset 50) produced an oscillating
  forecast with the right period and amplitude (e.g. first 10 values
  `[58.5, 57.0, 55.1, 52.6, 50.1, 47.5, 45.0, 43.0, 41.3, 40.3]`), i.e. genuinely periodic,
  sensible forecasting behavior — not NaNs, not garbage, not a flat line. This confirms the
  checkpoint and code are correctly wired together before touching ONNX at all.

## The export itself

Target: a single static forward pass, `forward(input_ts, input_padding, freq)` on
`PatchedTimeSeriesDecoder` (obtained via `tfm._model`), with:

- `context_len = 512` (16 patches of `input_patch_len = 32`)
- `horizon_len = output_patch_len = 128` (native single-shot case — the task's guidance to avoid
  the autoregressive `decode()` loop for horizons > 128 was followed; `decode()` itself was never
  attempted)

Dummy inputs (fixed shape, `batch = 1`):

| name | shape | dtype | notes |
|---|---|---|---|
| `input_ts` | `[1, 512]` | `float32` | `torch.randn` |
| `input_padding` | `[1, 512]` | `float32` | all zeros (no padding). **Note**: `forward()`'s type hint says `torch.LongTensor`, but the actual runtime call site in `timesfm_torch.py`'s `_forecast()` builds it via `torch.Tensor(...)` (float) — the type hint is simply wrong/stale in the upstream source; float32 is what the real code path uses. |
| `freq` | `[1, 1]` | `int64` | `torch.zeros(...)`, frequency bucket 0 (high frequency) |

```python
torch.onnx.export(
    model, (input_ts, input_padding, freq), "timesfm_1_0_200m.onnx",
    input_names=["input_ts", "input_padding", "freq"],
    output_names=["output_ts"],
    opset_version=18,
    dynamo=True,   # torch 2.13's default exporter (torch.export–based)
)
```

**Result: succeeded on the first attempt, in ~10.2 seconds.** No unsupported-op errors, no
data-dependent control flow errors, no need to fall back to the legacy TorchScript-tracing
exporter (`dynamo=False`) — that fallback was written into the spike script but never triggered.
This matches the task's prediction that horizon ≤ 128 (no autoregressive loop) is the scenario
most likely to trace cleanly.

Output files:
- `/Users/jimmyshi/code/zkie/models/timesfm_1_0_200m.onnx` — 1,416,703 bytes (graph structure)
- `/Users/jimmyshi/code/zkie/models/timesfm_1_0_200m.onnx.data` — 814,546,944 bytes (external
  weight data; ≈203.6M params × 4 bytes/float32 ≈ 814MB, checks out)

Both already fall under the pre-existing `/models` entry in `.gitignore` — confirmed, nothing
extra needed there. Combined they are ~776MB, correctly excluded from git.

## Structural verification

`onnx.checker.check_model(model)` — **passes**.

- IR version: 10, opset: `{"": 18}`
- Inputs: `input_ts [1,512]`, `input_padding [1,512]`, `freq [1,1]`
- Output: `output_ts [1, 16, 128, 10]` (16 patches × horizon 128 × (1 mean + 9 quantiles))
- 778 nodes, 279 initializers, **39 distinct op types**

## Numerical verification

Ran the same fixed random input through (a) the original PyTorch model directly and (b) the
exported ONNX graph via `onnxruntime.InferenceSession` (CPU execution provider):

```
torch_out shape (1, 16, 128, 10)
onnx_out  shape (1, 16, 128, 10)
max abs diff: 7.390976e-06
mean abs diff: 9.692619e-07
allclose (atol=1e-3, rtol=1e-3): True
allclose (atol=1e-4, rtol=1e-4): True
```

The export is not merely structurally valid — it computes essentially the same numbers as the
real PyTorch model (differences at the level of float32 rounding across a different op
decomposition, not a semantic bug).

## Verification against `zkie-compiler`

Per the task's instruction, this was checked using `zkie-compiler`'s **own Rust parser**, run
from inside `/Users/jimmyshi/code/zkie` (never from `/tmp` or the scratchpad, per this repo's
`CLAUDE.md` build-path constraint) via a throwaway integration test
(`crates/zkie-compiler/tests/timesfm_onnx_spike.rs`, written, run, and then **deleted** afterward
— this investigation did not modify any existing Rust code, and left no new Rust files behind).

1. **`onnx_parser::load_model(path)`** (protobuf decode) — **succeeds**, instantly. Reports
   `ir_version: 10`, `node count: 778`, `initializer count: 279` — exactly matching the Python-side
   `onnx.load()` inspection above.

2. **Op-type coverage** against the op mapper's current supported set (`MatMul`, `Gemm`, `Add`,
   `Mul`, `Relu`, `Softmax`, `Gelu`, `LayerNormalization`, `ReduceMean`, `Gather`, `Reshape`,
   `Transpose`, `Squeeze`, `Unsqueeze`, `Concat`, from `op_mapper.rs`):

   | | nodes | % of 778 |
   |---|---|---|
   | supported op_type | 644 | 82.8% |
   | **unsupported** op_type | **134** | **17.2%** |

   13 of the mapper's 15 supported op names actually appear in this graph (`Gemm` and `Gelu` are
   not used — matmuls in this model are plain `MatMul` + separate `Add` for bias, and the FFN
   activation is `Relu`, not `Gelu`, matching the earlier scope-decision doc's finding that
   TimesFM's FFN uses ReLU).

   26 distinct **unsupported** op types account for the other 134 nodes:
   `Abs, ArgMax, Cast, Clip, Cos, Div, Equal, GatherElements, GatherND, GreaterOrEqual, Less, Min,
   Mod, Not, Pad, Pow, Reciprocal, ReduceMax, ReduceMin, ReduceSum, Sigmoid, Sin, Split, Sqrt, Sub,
   Where`. Most of these trace back to identifiable, explainable sources in the original code, not
   exporter noise:
   - `Pow`/`Sqrt`/`Reciprocal`/`ReduceMean` (×20 each) — the custom `RMSNorm` class used for each
     of the 20 decoder layers' `input_layernorm` (attention pre-norm), which is hand-written
     (`x * rsqrt(mean(x^2) + eps)`) rather than `nn.LayerNorm`, so it doesn't decompose to
     `LayerNormalization`. (The MLP's own `nn.LayerNorm` calls **do** show up as the 20
     `LayerNormalization` nodes that are supported.)
   - `Sigmoid`/`Mul` (×2) — the two top-level `ResidualBlock`s (`input_ff_layer`, `horizon_ff_layer`)
     use `nn.SiLU`, which decomposes to `Sigmoid` + `Mul`.
   - `Split` (×20) — the fused QKV linear projection in each attention layer is split into Q/K/V.
   - `ArgMax`/`Where`/`GreaterOrEqual`/`Equal`/`Clip`/`ReduceSum` — the padding/masking and
     per-patch mean/std statistics logic (`_masked_mean_std`, padding-value substitution).
   - `Sin`/`Cos` — the sinusoidal `PositionalEmbedding`.
   - `GatherND`/`Mod`/`GatherElements`/`Less`/`Not`/`Pad` — mostly `_shift_padded_seq`'s
     index-shifting logic for aligning the positional embedding to padding boundaries.
   - `Cast`/`Abs`/`Div`/`Min`/`Sub`/`ReduceMax`/`ReduceMin` — general glue in the above.

   None of this is surprising for a real, non-toy transformer with hand-written normalization and
   masking logic — but it means a real compile of this graph, as-is, through the *current*
   `op_mapper.rs` would hit `OpMapperError::UnsupportedOp` on 26 distinct op kinds well before
   reaching the end of the graph, even though the majority of individual nodes are of supported
   types.

3. **Initializer/weight extraction — the more significant gap.** `onnx_parser::extract_initializers`
   was run against this real export and revealed two concrete, previously-only-theoretical
   limitations documented in `onnx_parser.rs`'s own module docs:
   - Of the 279 initializers, **17 are non-`FLOAT` (`data_type == 7`, i.e. INT64)** — shape/index
     constants the dynamo exporter lifted into the graph (e.g. for `Reshape`/`Gather`/`Split`
     parameters). `onnx_parser` deliberately rejects non-float tensors
     (`OnnxParseError::UnsupportedDataType`) rather than misinterpreting them — calling
     `extract_initializers` on the *full* initializer list throws on the first INT64 tensor it
     hits (confirmed: fails on tensor `val_4`, `data_type: 7`).
   - Of the remaining 262 `FLOAT` initializers, **only 7 have non-empty extracted data; 255 come
     back empty.** This is because this export uses ONNX's `external_data` storage (the actual
     float32 bytes live in the sibling `timesfm_1_0_200m.onnx.data` file, referenced by an
     `external_data` field on each `TensorProto`, not inlined via `raw_data`/`float_data`).
     `onnx_parser::extract_tensor` only ever reads `float_data` or `raw_data` — it has no code
     path that resolves `external_data` at all, so for an export this large (which `external_data`
     is the normal/expected way to produce, and is `torch.onnx.export`'s default), **the parser
     currently cannot retrieve the real weight values, even though `load_model` reports success.**

   Put plainly: `zkie-compiler` can today parse the *shape* of this real TimesFM graph, and count
   and classify its nodes, but cannot yet pull the actual trained weights out of an export
   produced the normal way (external data, opset 18, dynamo exporter) — both because of the
   INT64-constant scope limitation and, more importantly, because external-data resolution isn't
   implemented. Both gaps are called out as deliberate, documented scope limitations in
   `onnx_parser.rs`'s own doc comments already (i.e. this is not new information about intent —
   this investigation just confirms both actually bite on a real, large model, not only in theory).

## What would be needed to compile more of this graph (not attempted here — out of scope)

This investigation did not attempt any compiler changes (explicitly out of scope per instructions
— Python/export-side only). For a future integration step, the concrete next needs are:
1. `onnx_parser`: resolve `TensorProto.external_data` (read the sibling `.onnx.data` file at the
   given byte offset/length) so real float weights can be extracted from an export like this one.
2. `onnx_parser`/`op_mapper`: a path for INT64 constant tensors used purely as static
   shape/index arguments (as opposed to being treated as unsupported data outright) — these are
   compile-time constants, not runtime tensor data, in this graph.
3. `op_mapper`: coverage for at least `Pow`+`Sqrt`+`Reciprocal` (or a fused `RMSNorm`-recognition
   pass), `Split`, `Sigmoid` (for `SiLU`), `Sin`/`Cos`, and the masking ops (`Where`, `Equal`,
   `GreaterOrEqual`, `Clip`, `ArgMax`, etc.) to get meaningfully further into this graph. A
   sensible incremental target, consistent with sub-project 4's decision to prove a single
   transformer block rather than the full 20-layer model, would be compiling just one decoder
   layer's worth of ops rather than all 26 missing types at once.

## Files produced by this investigation

- `/Users/jimmyshi/code/zkie/models/timesfm_1_0_200m.onnx` (1.4MB, graph) and
  `/Users/jimmyshi/code/zkie/models/timesfm_1_0_200m.onnx.data` (778MB, weights) — both already
  covered by the pre-existing `/models` `.gitignore` entry.
- `/Users/jimmyshi/code/zkie/.venv-timesfm/` — Python 3.12.13 venv, now gitignored (entry added).
- `/Users/jimmyshi/code/zkie/.spike-test/timesfm_v1/` — the minimal 3-file vendored PyTorch-only
  subset of `google-research/timesfm`'s `v1/` source actually used for loading/exporting.
- `/Users/jimmyshi/code/zkie/.spike-test/timesfm_v1_src/` — full copies of all 9 original
  `v1/src/timesfm/*.py` files, kept for reference/audit.
- `/Users/jimmyshi/code/zkie/.spike-test/export_onnx.py`, `verify_parity.py`, `export_log.txt` —
  the spike scripts and captured export log.
- This report.

All of the above under `.spike-test/` are gitignored (pre-existing `/.spike-test` entry) and
throwaway; nothing here was committed.

## Note on unrelated observation

During this session, unrelated uncommitted changes to `crates/zkie-core` (`chips/eltwise.rs`,
`chips/layer_norm.rs`, `chips/reduce.rs`, and two test files) were noticed in `git status` that
were not made by this investigation (this task only touched Python/export-side files, `.gitignore`,
and one throwaway Rust test file that was deleted before finishing). They look like real,
in-progress work on cell-linking infrastructure for composing chips — consistent with sub-project
4's item #1 (circuit assembler) from the scope-decision doc — likely from a concurrent session
working on this same repo. They were left untouched rather than reverted, since reverting
unfamiliar in-progress changes without full context risks destroying real work.
